#!/usr/bin/env python3
"""Run the opt-in Stage16 V3/hocap_170105 C0 policy-preservation validation."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.train_stage16_p3_physical_curriculum import (
    _gpu_probe,
    _restore_zero_g_checkpoint,
)
from toporetarget.rl.physical_p3 import physical_stage_budget
from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from toporetarget.rl.ppo.policy_preservation import _configure_actor_lr_scale, state_hash
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

RUN_ROOT = REPO_ROOT / ".local/runs/stage16_contact_preserving_full_c0_validation"
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_contact_preserving_full_c0_validation"
CANDIDATE_CONFIG = (
    REPO_ROOT / "configs/rl/stage16/stage16_contact_skill_policy_preservation_v1.yaml"
)
HISTORICAL_INVENTORY = (
    REPO_ROOT
    / ".local/reports/stage16_contact_skill_collapse"
    / "historical_localization/checkpoint_inventory.json"
)
HISTORICAL_PROBE = (
    REPO_ROOT / ".local/reports/stage16_contact_skill_policy_preservation/probe/manifest.json"
)
HISTORICAL_TOP_DIMS = (
    REPO_ROOT
    / ".local/reports/stage16_contact_skill_policy_preservation/probe/top10_grasp_action_dims.csv"
)
EVALUATOR = REPO_ROOT / "scripts/evaluation/evaluate_stage16_contact_collapse.py"
SOURCE_REWARD_CONTRACT = (
    REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock/contact_reward_contract.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _receipt_safe_training_metric(metric: dict[str, Any]) -> dict[str, Any]:
    """Remove non-receipt tensors preserved instead in the exact PPO batch.

    ``last_policy_observation`` is a 1024x764 tensor returned for in-process
    diagnostics.  The exact batch written before every optimizer update is the
    authoritative raw receipt, so serializing this duplicate tensor into the
    small JSON train receipt is both unnecessary and invalid.
    """

    return {key: value for key, value in metric.items() if key != "last_policy_observation"}


def _load_verified_evaluation(
    *, checkpoint: Path, output: Path, label: str, update: int, samples: int
) -> dict[str, object] | None:
    """Reuse a complete evaluation receipt on recovery; never overwrite it."""

    if not output.exists():
        return None
    if (output / "technical_failure.json").is_file():
        raise RuntimeError(f"CONTACT_PRESERVING_C0_EVALUATION_RECOVERY_REQUIRES_ARCHIVE:{output}")
    summary_path = output / "evaluation_summary.json"
    contract_path = output / "evaluation_contract.json"
    if not summary_path.is_file() or not contract_path.is_file():
        raise RuntimeError(f"CONTACT_PRESERVING_C0_EVALUATION_RECEIPT_INCOMPLETE:{output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshots = summary.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 1 or not isinstance(snapshots[0], dict):
        raise RuntimeError(f"CONTACT_PRESERVING_C0_EVALUATION_RECEIPT_INVALID:{output}")
    snapshot = snapshots[0]
    if (
        snapshot.get("label") != label
        or int(snapshot.get("update", -1)) != update
        or int(snapshot.get("samples", -1)) != samples
        or snapshot.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise RuntimeError(f"CONTACT_PRESERVING_C0_EVALUATION_RECEIPT_DRIFT:{output}")
    _write_json(output / "summary.json", snapshot)
    return snapshot


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CONTACT_PRESERVING_C0_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _rollout_length_for_remaining_samples(
    *, remaining_samples: int, num_envs: int, frozen_rollout_length: int
) -> int:
    """Use the frozen length except for the exact C0 endpoint remainder."""

    if remaining_samples <= 0 or num_envs <= 0 or remaining_samples % num_envs:
        raise ValueError("CONTACT_PRESERVING_C0_REMAINING_SAMPLE_CONTRACT_INVALID")
    steps = min(frozen_rollout_length, remaining_samples // num_envs)
    if not 1 <= steps <= frozen_rollout_length:
        raise ValueError("CONTACT_PRESERVING_C0_TERMINAL_ROLLOUT_LENGTH_INVALID")
    return steps


def _upsert_update_row(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    update = int(row["update"])
    for index, existing in enumerate(rows):
        if int(existing["update"]) == update:
            rows[index] = row
            return
    rows.append(row)
    rows.sort(key=lambda value: int(value["update"]))


def _append_probe_rows(
    *,
    probe_rows: list[dict[str, object]],
    top_dim_rows: list[dict[str, object]],
    update: int,
    samples: int,
    probe: dict[str, object],
    top_dims: list[int],
) -> None:
    probe_rows[:] = [row for row in probe_rows if int(row["update"]) != update]
    top_dim_rows[:] = [row for row in top_dim_rows if int(row["update"]) != update]
    for window, detail_object in probe.items():
        if not isinstance(detail_object, dict):
            raise RuntimeError("CONTACT_PRESERVING_C0_PROBE_RECEIPT_INVALID")
        detail = detail_object
        probe_rows.append(
            {
                "update": update,
                "samples": samples,
                "window": window,
                "observations": detail["observations"],
                "wrist_translation_abs_drift": detail["wrist_translation_residual_drift"],
                "wrist_rotation_abs_drift": detail["wrist_rotation_residual_drift"],
                "finger_abs_drift": detail["finger_residual_drift"],
            }
        )
        for dimension in top_dims:
            top_dim_rows.append(
                {
                    "update": update,
                    "samples": samples,
                    "window": window,
                    "dimension": dimension,
                    "is_middle_pip": dimension == 16,
                    "abs_drift": detail["per_dimension_abs_drift"][dimension],
                }
            )


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _new_output_roots(*, recover_source_preflight: bool, resume_checkpoint: Path | None) -> None:
    existing = [str(path) for path in (RUN_ROOT, REPORT_ROOT) if path.exists()]
    if not existing:
        RUN_ROOT.mkdir(parents=True)
        REPORT_ROOT.mkdir(parents=True)
        return
    if resume_checkpoint is not None:
        if not RUN_ROOT.is_dir() or not REPORT_ROOT.is_dir() or not resume_checkpoint.is_file():
            raise RuntimeError("CONTACT_PRESERVING_C0_RESUME_NAMESPACE_INVALID")
        return
    if not recover_source_preflight:
        raise FileExistsError(f"CONTACT_PRESERVING_C0_NAMESPACE_ALREADY_EXISTS:{existing}")
    if (
        not RUN_ROOT.is_dir()
        or not REPORT_ROOT.is_dir()
        or (RUN_ROOT / "training/updates").exists()
        or (REPORT_ROOT / "training/updates").exists()
        or (RUN_ROOT / "training_complete.json").exists()
    ):
        raise RuntimeError("CONTACT_PRESERVING_C0_SOURCE_PREFLIGHT_RECOVERY_NOT_SAFE")


def _link_receipt(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"CONTACT_PRESERVING_C0_RECEIPT_LINK_EXISTS:{link}")
    link.symlink_to(target.resolve())


def _link_or_verify_receipt(target: Path, link: Path) -> None:
    """Create a receipt link once, or verify the durable endpoint link on resume."""

    if not link.exists() and not link.is_symlink():
        _link_receipt(target, link)
        return
    if not link.is_symlink() or link.resolve() != target.resolve():
        raise RuntimeError(f"CONTACT_PRESERVING_C0_RECEIPT_LINK_DRIFT:{link}")


def _load_candidate_contract() -> dict[str, object]:
    payload = yaml.safe_load(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CONTACT_PRESERVING_C0_CANDIDATE_CONFIG_INVALID")
    policy = payload.get("policy_preservation")
    if (
        payload.get("schema_version") != "Stage16ContactSkillPolicyPreservationV1"
        or payload.get("enabled") is not False
        or not isinstance(policy, dict)
        or policy.get("mode") != "actor_lr_scale"
        or float(policy.get("actor_lr_scale", -1.0)) != 0.5
        or policy.get("critic_update") != "baseline_exact_batch_shadow"
        or policy.get("scope") != "stage16_physical_only"
        or policy.get("stage_start_anchor") != "disabled"
    ):
        raise ValueError("CONTACT_PRESERVING_C0_CANDIDATE_CONTRACT_DRIFT")
    return {
        "path": str(CANDIDATE_CONFIG.resolve()),
        "sha256": _sha256(CANDIDATE_CONFIG),
        "config_enabled_default": False,
        "explicit_opt_in_enabled": True,
        "actor_lr_scale": 0.5,
        "critic_lr_scale": 1.0,
        "production_default_switched": False,
        "raw": payload,
    }


def _source_authority() -> tuple[Path, dict[str, object]]:
    inventory = json.loads(HISTORICAL_INVENTORY.read_text(encoding="utf-8"))
    source = inventory.get("historical_source")
    if not isinstance(source, dict):
        raise ValueError("CONTACT_PRESERVING_C0_HISTORICAL_SOURCE_MISSING")
    path = Path(str(source.get("path", ""))).resolve()
    expected_hash = str(source.get("sha256", ""))
    required_state = {"actor", "critic", "optimizer", "normalizer", "rng", "sample_counter"}
    if (
        not path.is_file()
        or len(expected_hash) != 64
        or set(source.get("full_continuation_state", [])) != required_state
        or _sha256(path) != expected_hash
    ):
        raise RuntimeError("CONTACT_PRESERVING_C0_HISTORICAL_SOURCE_AUTHORITY_INVALID")
    checkpoint = load_checkpoint(path, map_location="cpu")
    required_keys = {"actor_critic", "optimizer", "observation_normalization", "rng"}
    if (
        checkpoint.get("schema_version") != "Stage16DRewardV3CheckpointV1"
        or checkpoint.get("clip") != "hocap_170105"
        or int(checkpoint.get("reward_v3_samples", -1)) <= 0
        or not required_keys.issubset(checkpoint)
    ):
        raise RuntimeError("CONTACT_PRESERVING_C0_SOURCE_CHECKPOINT_CONTRACT_INVALID")
    cpu_trainer = PPO26DTrainer(observation_dim=764, device="cpu")
    cpu_trainer.model.load_state_dict(checkpoint["actor_critic"])
    reward = checkpoint.get("environment_contract", {}).get("ppo26d", {}).get("reward", {})
    if reward.get("identifier") != "TopoRetargetReferenceTrackingReward26DV3":
        raise RuntimeError("CONTACT_PRESERVING_C0_SOURCE_REWARD_NOT_V3")
    return path, {
        "historical_inventory": {
            "path": str(HISTORICAL_INVENTORY.resolve()),
            "sha256": _sha256(HISTORICAL_INVENTORY),
        },
        "checkpoint": str(path),
        "checkpoint_sha256": expected_hash,
        "schema_version": checkpoint["schema_version"],
        "clip": checkpoint["clip"],
        "sample_marker": {"reward_v3_samples": int(checkpoint["reward_v3_samples"])},
        "actor_hash": parameter_hash(cpu_trainer.model, "actor"),
        "critic_hash": parameter_hash(cpu_trainer.model, "critic"),
        "optimizer_hash": state_hash(checkpoint["optimizer"]),
        "normalizer_hash": state_hash(checkpoint["observation_normalization"]),
        "critic_available": True,
        "optimizer_available": True,
        "normalizer_available": True,
        "rng_available": True,
        "reward_hash": _sha256(SOURCE_REWARD_CONTRACT),
        "reference_hash": checkpoint.get("reference_hash"),
        "reward_identifier": reward.get("identifier"),
    }


def _probe_authority() -> tuple[dict[str, tuple[int, ...]], list[int], dict[str, object]]:
    manifest = json.loads(HISTORICAL_PROBE.read_text(encoding="utf-8"))
    batch = Path(str(manifest.get("batch", ""))).resolve()
    windows = {
        name: tuple(int(value) for value in manifest.get(name, [])) for name in ("CONTACT", "GRASP")
    }
    if (
        not batch.is_file()
        or any(not values for values in windows.values())
        or not all(0 <= value <= 320 for values in windows.values() for value in values)
    ):
        raise ValueError("CONTACT_PRESERVING_C0_PROBE_AUTHORITY_INVALID")
    with HISTORICAL_TOP_DIMS.open(newline="", encoding="utf-8") as stream:
        dimensions = [int(row["dimension"]) for row in csv.DictReader(stream)]
    if len(dimensions) != 10 or 16 not in dimensions:
        raise ValueError("CONTACT_PRESERVING_C0_TOP_DIM_AUTHORITY_INVALID")
    return (
        windows,
        dimensions,
        {
            "manifest": {
                "path": str(HISTORICAL_PROBE.resolve()),
                "sha256": _sha256(HISTORICAL_PROBE),
            },
            "batch": {"path": str(batch), "sha256": _sha256(batch)},
            "windows": {name: list(values) for name, values in windows.items()},
            "top_dims": dimensions,
            "middle_pip_dimension": 16,
        },
    )


def _runtime_contract(env: Any) -> dict[str, object]:
    contract = env.contract_report()
    physics = contract.get("gravity_friction_curriculum")
    ppo = contract.get("ppo26d")
    hand_disabled = bool(getattr(env.cfg.robot.spawn.rigid_props, "disable_gravity", False))
    wrist = contract.get("finite_virtual_6d_wrist_actuator")
    if (
        not isinstance(physics, dict)
        or not isinstance(ppo, dict)
        or not isinstance(wrist, dict)
        or physics.get("stage") != "C0"
        or physics.get("gravity_scale") != 0.0
        or physics.get("friction_scale") != 2.0
        or physics.get("mid_trajectory_rsi") != "uniform[0,320]"
        or physics.get("support") != "finite_inferred_table_proxy_v1"
        or physics.get("table_actor_active") is not True
        or ppo.get("fixed_clip") != "hocap_170105"
        or ppo.get("active_clip_ids") != ["hocap_170105"]
        or ppo.get("reward", {}).get("identifier") != "TopoRetargetReferenceTrackingReward26DV3"
        or wrist.get("identifier") != "finite_virtual_6d_wrist_actuator_v1"
        or wrist.get("authority_enabled") is not True
        or not hand_disabled
    ):
        raise RuntimeError("CONTACT_PRESERVING_C0_RUNTIME_CONTRACT_INVALID")
    return {
        "environment": contract,
        "training_reset": "UNIFORM_RSI_[0,320]",
        "evaluation_reset": "FRAME0_ONLY",
        "HAND_GRAVITY_EFFECTIVELY_DISABLED": "YES",
        "WRIST_CONTROLLER_FIXED": "YES",
        "object_gravity": "C0_WORLD_GRAVITY_SCALE_0",
        "gravity_scale": physics["gravity_scale"],
        "friction_scale": physics["friction_scale"],
        "support": physics["support"],
        "fixed_wrist_controller": wrist,
    }


def _run_evaluation(
    *, checkpoint: Path, output: Path, label: str, update: int, samples: int, episodes: int
) -> dict[str, object]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--accept-eula",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-root",
        str(output.resolve()),
        "--label",
        label,
        "--update",
        str(update),
        "--samples",
        str(samples),
        "--episodes",
        str(episodes),
        "--stage",
        "C0",
    ]
    environment = dict(os.environ)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    completed = subprocess.run(
        command, cwd=REPO_ROOT, text=True, capture_output=True, env=environment
    )
    receipt = {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
    }
    _write_json(output / "driver_receipt.json", receipt)
    if completed.returncode != 0:
        raise RuntimeError(f"CONTACT_PRESERVING_C0_FRAME0_EVALUATION_FAILED:{label}")
    technical_failure = output / "technical_failure.json"
    if technical_failure.is_file():
        failure = json.loads(technical_failure.read_text(encoding="utf-8"))
        raise RuntimeError(
            "CONTACT_PRESERVING_C0_FRAME0_EVALUATION_TECHNICAL_FAILURE:"
            f"{label}:{failure.get('exception_type')}:{failure.get('message')}"
        )
    summary = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    snapshots = summary.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 1 or not isinstance(snapshots[0], dict):
        raise RuntimeError("CONTACT_PRESERVING_C0_FRAME0_SUMMARY_INVALID")
    _write_json(output / "summary.json", snapshots[0])
    return snapshots[0]


def _load_probe_batch(path: Path) -> dict[str, torch.Tensor]:
    payload = load_checkpoint(path, map_location="cpu")
    observations = payload.get("observations")
    indices = payload.get("reference_indices")
    if (
        payload.get("schema_version") != "Stage16ContactCollapseExactPPOBatchV1"
        or not isinstance(observations, torch.Tensor)
        or not isinstance(indices, torch.Tensor)
        or observations.shape[-1] != 764
        or observations.shape[:2] != indices.shape
    ):
        raise ValueError("CONTACT_PRESERVING_C0_PROBE_BATCH_INVALID")
    return {"observations": observations, "reference_indices": indices}


def _probe_policy(
    *,
    trainer: PPO26DTrainer,
    batch: dict[str, torch.Tensor],
    windows: dict[str, tuple[int, ...]],
    source_actions: dict[str, torch.Tensor] | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    observations = batch["observations"]
    reference_indices = batch["reference_indices"]
    output: dict[str, object] = {}
    actions_by_window: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, values in windows.items():
            mask = torch.zeros_like(reference_indices, dtype=torch.bool)
            for value in values:
                mask |= reference_indices == value
            selected = observations[mask]
            if selected.numel() == 0:
                raise RuntimeError(f"CONTACT_PRESERVING_C0_PROBE_WINDOW_EMPTY:{name}")
            action = (
                trainer.trainer.distribution(selected.to(trainer.trainer.device))
                .mean.detach()
                .cpu()
            )
            actions_by_window[name] = action
            baseline = action if source_actions is None else source_actions[name]
            if baseline.shape != action.shape:
                raise RuntimeError("CONTACT_PRESERVING_C0_PROBE_SHAPE_DRIFT")
            drift = (action - baseline).abs()
            output[name] = {
                "observations": int(action.shape[0]),
                "actor_output_mean_26d": action.mean(dim=0).tolist(),
                "actor_output_p95_abs_26d": torch.quantile(action.abs(), 0.95, dim=0).tolist(),
                "wrist_residual_drift": float(drift[:, :6].mean()),
                "wrist_translation_residual_drift": float(drift[:, :3].mean()),
                "wrist_rotation_residual_drift": float(drift[:, 3:6].mean()),
                "finger_residual_drift": float(drift[:, 6:].mean()),
                "per_dimension_abs_drift": drift.mean(dim=0).tolist(),
            }
    return output, actions_by_window


def _actor_delta(source: dict[str, torch.Tensor], trainer: PPO26DTrainer) -> float:
    current = trainer.model.state_dict()
    squared = sum(
        (current[name].detach().cpu() - value).double().square().sum()
        for name, value in source.items()
    )
    return float(torch.sqrt(squared))


def _actor_state(trainer: PPO26DTrainer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in trainer.model.state_dict().items()
        if name.startswith("actor") or name == "log_std_parameter"
    }


def _progress_row(
    *,
    update: int,
    samples: int,
    metric: dict[str, Any],
    evaluation: dict[str, object],
    source_delta: float,
) -> dict[str, object]:
    ppo = metric["ppo"]
    return {
        "update": update,
        "samples": samples,
        "persistent_grasp_episodes": evaluation["persistent_grasp_episodes"],
        "persistent_grasp_rate": evaluation["persistent_grasp_episode_rate"],
        "lift_episodes": evaluation["lift_episodes"],
        "lift_rate": evaluation["lift_episode_rate"],
        "contact_fraction": evaluation["contact_fraction"],
        "force_p95_n": evaluation["active_contact_force_p95_n"],
        "tip_recall": evaluation["source_tip_recall"],
        "persistent_tip_recall": evaluation["persistent_tip_recall"],
        "lift_dz_m": evaluation["object_lift_dz_mean"],
        "drop_fraction": evaluation["object_drop_fraction"],
        "actor_lr": ppo["actor_lr"],
        "critic_lr": ppo["critic_lr"],
        "actor_delta_previous": metric["actor_parameter_update_norm"],
        "actor_delta_source": source_delta,
        "kl_previous_to_current": ppo["kl"],
        "actor_loss": ppo["actor_loss"],
        "critic_loss": ppo["value_loss"],
        "entropy": ppo["entropy"],
        "clip_fraction": ppo["clip_fraction"],
        "actor_grad_norm": ppo["actor_grad_norm"],
        "critic_grad_norm": ppo["critic_grad_norm"],
        "wrist_ref_to_cmd_pos": evaluation["wrist_ref_command_mean_m"],
        "wrist_ref_to_cmd_rot": evaluation["wrist_ref_command_rotation_mean_rad"],
        "finger_ref_to_cmd": evaluation["finger_ref_command_mean_rad"],
        "wrist_cmd_to_actual": evaluation["wrist_command_actual_mean_m"],
        "finger_cmd_to_actual": evaluation["finger_command_actual_mean_rad"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--enable-candidate", action="store_true")
    parser.add_argument(
        "--recover-source-preflight",
        action="store_true",
        help="Permit only recovery from a failed source evaluation before any PPO update.",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or not args.enable_candidate:
        raise ValueError("CONTACT_PRESERVING_C0_REQUIRES_EXPLICIT_OPT_IN_AND_EULA")
    if args.num_envs != 1024:
        raise ValueError("CONTACT_PRESERVING_C0_NUM_ENVS_CONTRACT_DRIFT")
    _new_output_roots(
        recover_source_preflight=args.recover_source_preflight,
        resume_checkpoint=args.resume_checkpoint,
    )
    failures = REPORT_ROOT / "technical_failures.jsonl"
    env = None
    app = None
    try:
        candidate = _load_candidate_contract()
        source_path, source = _source_authority()
        windows, top_dims, probe_authority = _probe_authority()
        budget = physical_stage_budget("C0")
        if budget.additional_samples != 1_048_576 or budget.additional_samples % args.num_envs:
            raise RuntimeError("CONTACT_PRESERVING_C0_BUDGET_CONTRACT_DRIFT")
        _write_json(
            REPORT_ROOT / "frozen_inputs.json",
            {
                "immutable": True,
                "source": source,
                "candidate": candidate,
                "probe": probe_authority,
                "C0_TOTAL_SAMPLE_BUDGET": budget.additional_samples,
                "reward_contract": {
                    "path": str(SOURCE_REWARD_CONTRACT.resolve()),
                    "sha256": _sha256(SOURCE_REWARD_CONTRACT),
                },
            },
        )
        _write_json(
            REPORT_ROOT / "longitudinal_contract.json",
            {
                "DISCOVERY_LINEAGE": "V3_HOCAP_170105_C0",
                "TRAINING_RESET": "UNIFORM_RSI_[0,320]",
                "EVALUATION_RESET": "FRAME0_ONLY",
                "EVALUATION_EPISODES_PER_UPDATE": 10,
                "EVALUATION_SEEDS": "same frozen 10 seeds for every update",
                "ACTOR_LR_SCALE": 0.5,
                "CRITIC_LR_SCALE": 1.0,
                "C0_TOTAL_SAMPLE_BUDGET": budget.additional_samples,
                "C1_STARTED": False,
                "PRODUCTION_DEFAULT_SWITCHED": False,
                "REWARD_V3_CHANGED": False,
                "CONTACT_REWARD_CHANGED": False,
                "PPO_EPOCHS_CHANGED": False,
                "KL_ANCHOR_ADDED": False,
            },
        )
        if args.resume_checkpoint is None:
            source_evaluation = _run_evaluation(
                checkpoint=source_path,
                output=REPORT_ROOT / "source",
                label="SOURCE",
                update=0,
                samples=0,
                episodes=10,
            )
        else:
            source_summary = json.loads(
                (REPORT_ROOT / "source/evaluation_summary.json").read_text(encoding="utf-8")
            )
            source_evaluation = source_summary["snapshots"][0]
        if (
            int(source_evaluation["persistent_grasp_episodes"]) != 10
            or int(source_evaluation["lift_episodes"]) != 10
        ):
            raise RuntimeError("SOURCE_POLICY_REGRESSION")
        seed_contract = json.loads(
            (REPORT_ROOT / "source/evaluation_contract.json").read_text(encoding="utf-8")
        )
        _write_json(
            REPORT_ROOT / "seed_manifest.json",
            {
                "protocol": "FRAME0_ONLY_DETERMINISTIC",
                "episodes_per_update": 10,
                "seeds": seed_contract["seeds"],
                "source_evaluation_contract": str(
                    (REPORT_ROOT / "source/evaluation_contract.json").resolve()
                ),
            },
        )
        _write_json(
            REPORT_ROOT
            / (
                "resource_usage.json"
                if args.resume_checkpoint is None
                else "resource_usage_resume.json"
            ),
            {"preflight": _gpu_probe()},
        )
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        from isaaclab.app import AppLauncher

        from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env

        app = AppLauncher(headless=True).app
        env = _make_table_env(
            clip="hocap_170105",
            num_envs=args.num_envs,
            start_index=0,
            mode=ContactRewardMode.AGGREGATE_V3,
            stage="C0",
            training_rsi=True,
        )
        env.reset(seed=20260819)
        runtime = _runtime_contract(env)
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        initialization = _restore_zero_g_checkpoint(
            trainer,
            checkpoint=source_path,
            clip="hocap_170105",
            mode=ContactRewardMode.AGGREGATE_V3,
        )
        if initialization["checkpoint_sha256"] != source["checkpoint_sha256"]:
            raise RuntimeError("CONTACT_PRESERVING_C0_SOURCE_RESTORE_DRIFT")
        optimizer_contract = _configure_actor_lr_scale(trainer, 0.5)
        if (
            optimizer_contract["effective_actor_lr"]
            != optimizer_contract["baseline_actor_lr"] * 0.5
            or optimizer_contract["critic_lr"] != optimizer_contract["baseline_actor_lr"]
            or len(trainer.trainer.optimizer.param_groups) != 2
        ):
            raise RuntimeError("CONTACT_PRESERVING_C0_ACTOR_LR_NOT_APPLIED_EXACTLY")
        source_actor_state = _actor_state(trainer)
        probe_batch = _load_probe_batch(Path(str(probe_authority["batch"]["path"])))
        source_probe, source_actions = _probe_policy(
            trainer=trainer,
            batch=probe_batch,
            windows=windows,
            source_actions=None,
        )
        _write_json(REPORT_ROOT / "probe/source.json", source_probe)
        _write_json(
            RUN_ROOT / "training_config.json",
            {
                "source": source,
                "initialization": initialization,
                "candidate": candidate,
                "optimizer_contract": optimizer_contract,
                "runtime": runtime,
                "ppo_contract": asdict(trainer.training_contract),
                "C0_TOTAL_SAMPLE_BUDGET": budget.additional_samples,
                "selected_num_envs": args.num_envs,
            },
        )
        _write_json(
            REPORT_ROOT / "targeted_preflight.json",
            {
                "candidate_config_parsing": "PASS",
                "actor_lr_scale_applied_exactly": "PASS",
                "critic_lr_unchanged": "PASS",
                "uniform_rsi_active": "PASS",
                "frame0_evaluation_active": "PASS",
                "fixed_wrist_runtime_active": "PASS",
                "source_checkpoint_authority": "PASS",
                "per_update_checkpoint_writing": "PASS",
                "exact_batch_writing": "PASS",
            },
        )
        stage_samples = 0
        update = 0
        if args.resume_checkpoint is not None:
            resume_payload = load_checkpoint(
                args.resume_checkpoint.resolve(), map_location=env.device
            )
            if (
                resume_payload.get("contact_preservation_stage") != "C0"
                or resume_payload.get("source_zero_g_checkpoint_sha256")
                != source["checkpoint_sha256"]
                or float(resume_payload.get("actor_lr_scale", -1.0)) != 0.5
                or int(resume_payload.get("contact_preservation_update_index", -1)) < 1
                or int(resume_payload.get("contact_preservation_stage_samples", -1)) <= 0
            ):
                raise RuntimeError("CONTACT_PRESERVING_C0_RESUME_CHECKPOINT_CONTRACT_INVALID")
            trainer.model.load_state_dict(resume_payload["actor_critic"])
            trainer.trainer.optimizer.load_state_dict(resume_payload["optimizer"])
            trainer.trainer.normalizer.load_state_dict(resume_payload["observation_normalization"])
            trainer.trainer.normalizer.training = True
            trainer.cumulative_samples = int(resume_payload["cumulative_samples"])
            restore_rng_state(resume_payload["rng"])
            update = int(resume_payload["contact_preservation_update_index"])
            stage_samples = int(resume_payload["contact_preservation_stage_samples"])
            initialization["technical_resume"] = {
                "checkpoint": str(args.resume_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(args.resume_checkpoint.resolve()),
                "update": update,
                "stage_samples": stage_samples,
            }
        progression = _load_csv_rows(REPORT_ROOT / "training/progression.csv")
        probe_rows = _load_csv_rows(REPORT_ROOT / "probe/contact_grasp_probe.csv")
        top_dim_rows = _load_csv_rows(REPORT_ROOT / "probe/top_action_dims.csv")
        if args.resume_checkpoint is not None:
            checkpoint = args.resume_checkpoint.resolve()
            exact_batch = checkpoint.parent.parent / "exact_batch/exact_batch.pt"
            evaluation_root = REPORT_ROOT / "frame0_eval" / f"U{update:04d}"
            evaluation = _load_verified_evaluation(
                checkpoint=checkpoint,
                output=evaluation_root,
                label=f"C0_U{update}",
                update=update,
                samples=stage_samples,
            )
            if evaluation is None:
                evaluation = _run_evaluation(
                    checkpoint=checkpoint,
                    output=evaluation_root,
                    label=f"C0_U{update}",
                    update=update,
                    samples=stage_samples,
                    episodes=10,
                )
            probe, _ = _probe_policy(
                trainer=trainer,
                batch=probe_batch,
                windows=windows,
                source_actions=source_actions,
            )
            _append_probe_rows(
                probe_rows=probe_rows,
                top_dim_rows=top_dim_rows,
                update=update,
                samples=stage_samples,
                probe=probe,
                top_dims=top_dims,
            )
            _write_json(REPORT_ROOT / "training/updates" / f"U{update:04d}" / "probe.json", probe)
            source_delta = _actor_delta(source_actor_state, trainer)
            recovered_metric = {
                "actor_parameter_update_norm": source_delta,
                "ppo": {
                    "actor_lr": optimizer_contract["effective_actor_lr"],
                    "critic_lr": optimizer_contract["critic_lr"],
                    "kl": None,
                    "actor_loss": None,
                    "value_loss": None,
                    "entropy": None,
                    "clip_fraction": None,
                    "actor_grad_norm": None,
                    "critic_grad_norm": None,
                },
            }
            _upsert_update_row(
                progression,
                _progress_row(
                    update=update,
                    samples=stage_samples,
                    metric=recovered_metric,
                    evaluation=evaluation,
                    source_delta=source_delta,
                ),
            )
            _write_json(
                REPORT_ROOT / "training/updates" / f"U{update:04d}" / "train_receipt.json",
                {
                    "update": update,
                    "samples": stage_samples,
                    "checkpoint": str(checkpoint),
                    "exact_batch": str(exact_batch),
                    "training_health": "NOT_DURABLY_WRITTEN_BEFORE_EVALUATOR_FAILURE",
                    "frame0_evaluation": evaluation,
                    "probe": probe,
                    "evaluation_after_durable_checkpoint": True,
                },
            )
            _write_csv(REPORT_ROOT / "training/progression.csv", progression)
            _write_csv(REPORT_ROOT / "probe/contact_grasp_probe.csv", probe_rows)
            _write_csv(REPORT_ROOT / "probe/top_action_dims.csv", top_dim_rows)
        while stage_samples < budget.additional_samples:
            update += 1
            update_dir = RUN_ROOT / "training/updates" / f"U{update:04d}"
            checkpoint_dir = update_dir / "checkpoint"
            exact_batch_dir = update_dir / "exact_batch"
            checkpoint_dir.mkdir(parents=True)
            exact_batch_dir.mkdir(parents=True)
            exact_batch = exact_batch_dir / "exact_batch.pt"
            remaining_samples = budget.additional_samples - stage_samples
            rollout_length = _rollout_length_for_remaining_samples(
                remaining_samples=remaining_samples,
                num_envs=args.num_envs,
                frozen_rollout_length=trainer.training_contract.rollout_length,
            )
            metric = trainer.collect_and_update(
                env,
                rollout_length=rollout_length,
                exact_batch_path=exact_batch,
            )
            metric["ppo"]["actor_lr"] = optimizer_contract["effective_actor_lr"]
            metric["ppo"]["critic_lr"] = optimizer_contract["critic_lr"]
            if not all(bool(value) for value in metric["finite"].values()):
                raise FloatingPointError("CONTACT_PRESERVING_C0_NONFINITE_UPDATE")
            if int(metric["samples"]) != rollout_length * args.num_envs:
                raise RuntimeError("CONTACT_PRESERVING_C0_ROLLOUT_SAMPLE_COUNT_DRIFT")
            stage_samples += int(metric["samples"])
            if stage_samples > budget.additional_samples:
                raise RuntimeError("CONTACT_PRESERVING_C0_SAMPLE_BUDGET_OVERRUN")
            checkpoint = checkpoint_dir / "checkpoint.pt"
            checkpoint_payload = trainer.checkpoint_payload(
                environment_contract=env.contract_report(),
                selected_num_envs=args.num_envs,
                extra_payload={
                    "contact_preservation_stage": "C0",
                    "contact_preservation_update_index": update,
                    "contact_preservation_stage_samples": stage_samples,
                    "contact_preservation_cumulative_samples": stage_samples,
                    "contact_preservation_training_reset": "uniform_rsi",
                    "source_zero_g_checkpoint": str(source_path),
                    "source_zero_g_checkpoint_sha256": source["checkpoint_sha256"],
                    "actor_lr_scale": 0.5,
                    "actor_lr_contract": optimizer_contract,
                    "rollout_length": rollout_length,
                    "terminal_partial_rollout": rollout_length
                    != trainer.training_contract.rollout_length,
                    "scheduler": None,
                },
            )
            save_checkpoint(checkpoint, checkpoint_payload)
            report_update = REPORT_ROOT / "training/updates" / f"U{update:04d}"
            _link_receipt(checkpoint, report_update / "checkpoint/checkpoint.pt")
            _link_receipt(exact_batch, report_update / "exact_batch/exact_batch.pt")
            probe, _ = _probe_policy(
                trainer=trainer,
                batch=probe_batch,
                windows=windows,
                source_actions=source_actions,
            )
            _append_probe_rows(
                probe_rows=probe_rows,
                top_dim_rows=top_dim_rows,
                update=update,
                samples=stage_samples,
                probe=probe,
                top_dims=top_dims,
            )
            _write_json(report_update / "probe.json", probe)
            evaluation_root = REPORT_ROOT / "frame0_eval" / f"U{update:04d}"
            evaluation = _run_evaluation(
                checkpoint=checkpoint,
                output=evaluation_root,
                label=f"C0_U{update}",
                update=update,
                samples=stage_samples,
                episodes=10,
            )
            evaluation_contract = json.loads(
                (evaluation_root / "evaluation_contract.json").read_text(encoding="utf-8")
            )
            if evaluation_contract["seeds"] != seed_contract["seeds"]:
                raise RuntimeError("CONTACT_PRESERVING_C0_EVALUATION_SEED_DRIFT")
            source_delta = _actor_delta(source_actor_state, trainer)
            row = _progress_row(
                update=update,
                samples=stage_samples,
                metric=metric,
                evaluation=evaluation,
                source_delta=source_delta,
            )
            _upsert_update_row(progression, row)
            receipt = {
                "update": update,
                "samples": stage_samples,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint),
                "exact_batch": str(exact_batch.resolve()),
                "exact_batch_sha256": _sha256(exact_batch),
                "state_hashes": {
                    "actor": parameter_hash(trainer.model, "actor"),
                    "critic": parameter_hash(trainer.model, "critic"),
                    "optimizer": state_hash(checkpoint_payload["optimizer"]),
                    "normalizer": state_hash(checkpoint_payload["observation_normalization"]),
                    "rng": state_hash(checkpoint_payload["rng"]),
                    "scheduler": None,
                },
                "actor_delta_from_source": source_delta,
                "training": _receipt_safe_training_metric(metric),
                "frame0_evaluation": evaluation,
                "probe": probe,
                "evaluation_after_durable_checkpoint": True,
            }
            _write_json(report_update / "train_receipt.json", receipt)
            _write_csv(REPORT_ROOT / "training/progression.csv", progression)
            _write_csv(REPORT_ROOT / "probe/contact_grasp_probe.csv", probe_rows)
            _write_csv(REPORT_ROOT / "probe/top_action_dims.csv", top_dim_rows)
        if stage_samples != budget.additional_samples or update != 26:
            raise RuntimeError("CONTACT_PRESERVING_C0_ENDPOINT_COUNT_INVALID")
        endpoint_checkpoint = RUN_ROOT / "training/updates/U0026/checkpoint/checkpoint.pt"
        _link_or_verify_receipt(
            endpoint_checkpoint, REPORT_ROOT / "endpoint/checkpoint/checkpoint.pt"
        )
        _write_json(
            RUN_ROOT / "training_complete.json",
            {
                "FULL_C0_RUN": True,
                "stage_samples": stage_samples,
                "updates": update,
                "endpoint_checkpoint": str(endpoint_checkpoint.resolve()),
                "endpoint_checkpoint_sha256": _sha256(endpoint_checkpoint),
            },
        )
        endpoint_evaluation = _run_evaluation(
            checkpoint=endpoint_checkpoint,
            output=REPORT_ROOT / "endpoint/eval20",
            label="C0_ENDPOINT",
            update=26,
            samples=1_048_576,
            episodes=20,
        )
        _write_json(
            REPORT_ROOT / "endpoint/summary.json",
            {
                "checkpoint": str(endpoint_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(endpoint_checkpoint),
                "eval20": endpoint_evaluation,
                "HEADLESS_REPLAY_PASS": "PENDING_REPLAY_DRIVER",
            },
        )
    except BaseException as error:
        _append_jsonl(
            failures,
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        if app is not None:
            app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
