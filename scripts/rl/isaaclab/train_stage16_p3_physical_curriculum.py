#!/usr/bin/env python3
"""Run one recoverable Stage 16 P3 gravity/friction PPO curriculum segment.

The runner is deliberately stage-bound: a process owns exactly one fixed
gravity/friction condition.  Physics therefore cannot react to contact,
episode progress, reward, or any rollout signal.  C0 obtains its full PPO
state from the selected zero-g V3/V4 checkpoint; later stages accept only the
same stage or its immediate predecessor.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.gravity_friction_curriculum import (
    INITIAL_SAFE_BANKS,
    load_gravity_friction_curriculum,
    load_p3_entry_gate_v2,
)
from toporetarget.rl.physical_p3 import (
    PHYSICAL_PPO_CHECKPOINT_SCHEMA,
    PHYSICAL_PPO_RESULT_SCHEMA,
    checkpoint_state,
    physical_stage_budget,
    validate_resume_payload,
)
from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DTrainingConfigV1
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity"
DEFAULT_REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DEFAULT_SAFE_BANK_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2/p1"
DEFAULT_P0_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2"
DEFAULT_CURRICULUM = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
DEFAULT_ENTRY_V2 = REPO_ROOT / "configs/rl/stage16/stage16_p3_entry_gate_v2.yaml"
DEFAULT_V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
DEFAULT_V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_receipt(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PHYSICAL_PPO_REQUIRED_INPUT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _gpu_probe() -> dict[str, str]:
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "query": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "pmon": ["nvidia-smi", "pmon", "-c", "1"],
    }
    output: dict[str, str] = {}
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"PHYSICAL_PPO_GPU_PROBE_FAILED:{name}:{result.stderr.strip()}")
        output[name] = result.stdout.strip()
    return output


def _mode_paths(mode: ContactRewardMode) -> tuple[Path, Path, Path]:
    if mode is ContactRewardMode.AGGREGATE_V3:
        return (
            DEFAULT_V3_ROOT / "contact_reward_contract.json",
            REPO_ROOT / ".local/reports/stage16d_reward_v3_contact",
            DEFAULT_V3_ROOT,
        )
    return (DEFAULT_V4_ROOT / "strict_v4_contract.json", DEFAULT_V4_ROOT, DEFAULT_V4_ROOT)


def _selection_path(mode: ContactRewardMode, clip: str) -> Path:
    if mode is ContactRewardMode.AGGREGATE_V3:
        return DEFAULT_V3_ROOT / clip / "dev/checkpoint_selection.json"
    if clip == "hocap_170650":
        return DEFAULT_V4_ROOT / clip / "final_checkpoint_selection.json"
    return DEFAULT_V4_ROOT / clip / "checkpoint_selection.json"


def _selected_zero_g_checkpoint(mode: ContactRewardMode, clip: str) -> dict[str, Any]:
    selection_path = _selection_path(mode, clip)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    ranked = selection.get("ranked")
    if not isinstance(ranked, list) or not ranked or not isinstance(ranked[0], dict):
        raise ValueError("PHYSICAL_PPO_ZERO_G_SELECTION_INVALID")
    winner = ranked[0]
    checkpoint = Path(str(winner.get("checkpoint", ""))).resolve()
    selection_sha = winner.get("checkpoint_sha256")
    if not checkpoint.is_file() or not isinstance(selection_sha, str):
        raise ValueError("PHYSICAL_PPO_ZERO_G_SELECTED_CHECKPOINT_INVALID")
    actual_sha = _sha256(checkpoint)
    if actual_sha != selection_sha:
        raise RuntimeError("PHYSICAL_PPO_ZERO_G_SELECTED_CHECKPOINT_HASH_MISMATCH")
    return {
        "selection": _path_receipt(selection_path),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": actual_sha,
        "selection_rank": 0,
        "selection_metrics": winner.get("metrics"),
    }


def _restore_zero_g_checkpoint(
    trainer: PPO26DTrainer,
    *,
    checkpoint: Path,
    clip: str,
    mode: ContactRewardMode,
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    expected_schema = (
        "Stage16DRewardV3CheckpointV1"
        if mode is ContactRewardMode.AGGREGATE_V3
        else "Stage16DStrictPerFingerV4CheckpointV1"
    )
    sample_key = (
        "reward_v3_samples" if mode is ContactRewardMode.AGGREGATE_V3 else "reward_v4_samples"
    )
    if payload.get("schema_version") != expected_schema:
        raise ValueError("PHYSICAL_PPO_ZERO_G_SCHEMA_INVALID")
    if payload.get("clip") != clip:
        raise ValueError("PHYSICAL_PPO_ZERO_G_CLIP_MISMATCH")
    reward = payload.get("environment_contract", {}).get("ppo26d", {}).get("reward", {})
    expected_reward = (
        "TopoRetargetReferenceTrackingReward26DV3"
        if mode is ContactRewardMode.AGGREGATE_V3
        else "TopoRetargetReferenceTrackingReward26DV4"
    )
    if reward.get("identifier") != expected_reward:
        raise ValueError("PHYSICAL_PPO_ZERO_G_CONTACT_CONTRACT_MISMATCH")
    policy_samples = int(payload.get(sample_key, -1))
    if policy_samples <= 0:
        raise ValueError("PHYSICAL_PPO_ZERO_G_SAMPLE_COUNTER_INVALID")
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = policy_samples
    restore_rng_state(payload["rng"])
    return {
        "kind": "SELECTED_ZERO_G_FULL_PPO_STATE",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_schema": expected_schema,
        "policy_training_samples_before": policy_samples,
        "selected_contact_mode": mode.value,
        "optimizer_restored": True,
        "critic_restored": True,
        "normalizer_restored": True,
        "rng_restored": True,
    }


def _restore_physical_checkpoint(
    trainer: PPO26DTrainer,
    *,
    checkpoint: Path,
    clip: str,
    num_envs: int,
    mode: ContactRewardMode,
    stage: str,
) -> tuple[dict[str, Any], dict[str, object]]:
    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    metadata = validate_resume_payload(
        payload,
        expected_clip=clip,
        expected_num_envs=num_envs,
        expected_contact_mode=mode,
        target_stage=stage,
    )
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(metadata["policy_training_samples"])
    restore_rng_state(payload["rng"])
    return (
        {
            "kind": "PHYSICAL_PPO_RESUME",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_schema": payload["schema_version"],
            "optimizer_restored": True,
            "critic_restored": True,
            "normalizer_restored": True,
            "rng_restored": True,
            **metadata,
        },
        metadata,
    )


def _p3_entry_evidence(*, entry_path: Path, safe_bank_root: Path) -> dict[str, Any]:
    """Validate P0--P2 immutable evidence and emit the V2 entry decision receipt."""

    entry = load_p3_entry_gate_v2(entry_path)
    p0 = DEFAULT_P0_ROOT / "p0/config_smoke.json"
    p2 = DEFAULT_P0_ROOT / "p2/support_decision.json"
    if json.loads(p0.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("PHYSICAL_PPO_P0_PROVENANCE_NOT_PASS")
    support = json.loads(p2.read_text(encoding="utf-8"))
    clips = support.get("clips")
    if not isinstance(clips, dict):
        raise ValueError("PHYSICAL_PPO_P2_SUPPORT_DECISION_INVALID")
    safe_counts: dict[str, int] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        item = clips.get(clip)
        rsi_path = safe_bank_root / f"rsi_v2_contract_{clip.removeprefix('hocap_')}.json"
        bank_path = safe_bank_root / f"safe_bank_{clip.removeprefix('hocap_')}.npz"
        rsi = json.loads(rsi_path.read_text(encoding="utf-8"))
        if (
            not isinstance(item, dict)
            or item.get("support_mode") != "CONTACT_READY_ONLY_VALIDATED"
            or item.get("frame_zero_full_gravity_authorized") is not False
            or item.get("hidden_support") is not False
            or rsi.get("status") != "P1_RSI_V2_VALIDATED"
            or int(rsi.get("initial_p3_safe_state_count", 0)) < 1
            or not bank_path.is_file()
        ):
            raise RuntimeError(f"PHYSICAL_PPO_ENTRY_EVIDENCE_INVALID:{clip}")
        safe_counts[clip] = int(rsi["initial_p3_safe_state_count"])
    return {
        "schema_version": "Stage16P3EntryDecisionV2",
        "status": "P3_READY_WITH_CONSTRAINTS",
        "entry_contract": _path_receipt(entry_path),
        "p0_config_smoke": _path_receipt(p0),
        "p2_support_decision": _path_receipt(p2),
        "safe_state_counts": safe_counts,
        "allowed_initial_reset_banks": list(INITIAL_SAFE_BANKS),
        "frame_zero_full_gravity_authorized": False,
        "invented_support_allowed": False,
        "external_guidance": False,
        "initial_rsi_mode": entry["decision_contract"]["initial_rsi_mode"],
        "historical_v1_preserved": str(entry["historical_entry_gate"]),
        "g3_placement": entry["promotion_gate"]["placement"],
    }


def _validate_environment_contract(
    environment_contract: dict[str, Any],
    *,
    clip: str,
    stage: str,
    expected_physics: dict[str, object],
) -> None:
    """Assert causal runtime invariants before collecting any PPO transition."""

    active = environment_contract.get("ppo26d")
    physics = environment_contract.get("gravity_friction_curriculum")
    if not isinstance(active, dict) or not isinstance(physics, dict):
        raise RuntimeError("PHYSICAL_PPO_ENVIRONMENT_CONTRACT_INVALID")
    observed = {
        "fixed_clip": active.get("fixed_clip"),
        "active_clip_ids": active.get("active_clip_ids"),
        "stage": physics.get("stage"),
        "gravity_scale": physics.get("gravity_scale"),
        "friction_scale": physics.get("friction_scale"),
        "support": physics.get("support"),
        "external_guidance": physics.get("external_guidance"),
        "frame_zero_full_gravity_authorized": physics.get("frame_zero_full_gravity_authorized"),
        "object_rollout_state_writes": active.get("object_rollout_state_writes"),
        "wrist_root_state_writes_during_step": active.get("wrist_root_state_writes_during_step"),
    }
    expected = {
        "fixed_clip": clip,
        "active_clip_ids": [clip],
        "stage": stage,
        "gravity_scale": expected_physics["gravity_scale"],
        "friction_scale": expected_physics["friction_scale"],
        "support": "none",
        "external_guidance": False,
        "frame_zero_full_gravity_authorized": False,
        "object_rollout_state_writes": 0,
        "wrist_root_state_writes_during_step": 0,
    }
    mismatch = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatch:
        raise RuntimeError(f"PHYSICAL_PPO_ENVIRONMENT_CONTRACT_INVALID:{mismatch}")


def _output_dir(root: Path, *, mode: ContactRewardMode, clip: str, stage: str) -> Path:
    if stage in {"C0", "C1", "C2"}:
        mode_directory = "v3" if mode is ContactRewardMode.AGGREGATE_V3 else "v4"
        return root / "physical_pilot" / mode_directory / clip / stage.lower()
    return root / "selected_mode" / clip / stage.lower()


def _save_checkpoint(
    *,
    path: Path,
    trainer: PPO26DTrainer,
    environment_contract: dict[str, Any],
    num_envs: int,
    stage: str,
    physical_stage_samples: int,
    physical_cumulative_samples: int,
    mode: ContactRewardMode,
    curriculum_state: dict[str, object],
    initialization: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    payload = trainer.checkpoint_payload(
        environment_contract=environment_contract, selected_num_envs=num_envs
    )
    payload["schema_version"] = PHYSICAL_PPO_CHECKPOINT_SCHEMA
    payload.pop("cumulative_samples")
    payload.update(
        checkpoint_state(
            stage=stage,
            physical_stage_samples=physical_stage_samples,
            physical_cumulative_samples=physical_cumulative_samples,
            policy_training_samples=trainer.cumulative_samples,
            selected_contact_mode=mode,
            allowed_reset_banks=INITIAL_SAFE_BANKS,
            curriculum_state=curriculum_state,
        )
    )
    payload["physical_initialization"] = initialization
    payload["p3_entry_gate_v2"] = entry
    payload["reward_contract"] = environment_contract["ppo26d"]["reward"]
    payload["curriculum_training_only"] = True
    save_checkpoint(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument(
        "--contact-mode", choices=tuple(mode.value for mode in ContactRewardMode), required=True
    )
    parser.add_argument("--stage", choices=("C0", "C1", "C2", "C3", "C4"), required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--safe-bank-root", type=Path, default=DEFAULT_SAFE_BANK_ROOT)
    parser.add_argument("--curriculum-contract", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--entry-gate-v2", type=Path, default=DEFAULT_ENTRY_V2)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-rollout-steps", type=int, default=40)
    parser.add_argument(
        "--no-critical-dr",
        action="store_false",
        dest="critical_dr",
        help="Only allow an explicitly documented diagnostic without inherited V3/V4 noise.",
    )
    parser.set_defaults(critical_dr=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.num_envs <= 0:
        raise ValueError("PHYSICAL_PPO_NUM_ENVS_INVALID")
    if args.smoke_rollout_steps <= 0 or args.smoke_rollout_steps > 40:
        raise ValueError("PHYSICAL_PPO_SMOKE_ROLLOUT_STEPS_INVALID")
    if args.smoke_only and (args.num_envs * args.smoke_rollout_steps) % 32 != 0:
        raise ValueError("PHYSICAL_PPO_SMOKE_SAMPLE_COUNT_MUST_DIVIDE_32_MINIBATCHES")
    mode = ContactRewardMode.parse(args.contact_mode)
    stage = args.stage
    budget = physical_stage_budget(stage)
    if not args.smoke_only and budget.additional_samples % args.num_envs != 0:
        raise ValueError("PHYSICAL_PPO_BUDGET_MUST_ALIGN_WITH_NUM_ENVS")
    if stage == "C0" and args.resume_checkpoint is not None:
        raise ValueError("PHYSICAL_PPO_C0_MUST_START_FROM_SELECTED_ZERO_G_CHECKPOINT")
    if stage != "C0" and args.resume_checkpoint is None:
        raise ValueError("PHYSICAL_PPO_LATER_STAGE_REQUIRES_RESUME_CHECKPOINT")

    output = _output_dir(args.output_root.resolve(), mode=mode, clip=args.clip, stage=stage)
    output.mkdir(parents=True, exist_ok=True)
    entry = _p3_entry_evidence(
        entry_path=args.entry_gate_v2.resolve(), safe_bank_root=args.safe_bank_root.resolve()
    )
    _write_json(args.output_root.resolve() / "p3_entry_gate_v2.json", entry)
    curriculum = load_gravity_friction_curriculum(args.curriculum_contract.resolve())
    contract_physics = curriculum.physics(stage)
    curriculum_state = curriculum.checkpoint_state(
        stage=stage,
        allowed_reset_banks=INITIAL_SAFE_BANKS,
        selected_contact_mode=mode.value,
    )
    contact_contract, contact_mask_root, mode_root = _mode_paths(mode)
    safe_bank = args.safe_bank_root.resolve() / f"safe_bank_{args.clip.removeprefix('hocap_')}.npz"
    inputs = {
        "curriculum_contract": _path_receipt(args.curriculum_contract),
        "entry_gate_v2": _path_receipt(args.entry_gate_v2),
        "safe_bank": _path_receipt(safe_bank),
        "contact_contract": _path_receipt(contact_contract),
        "reference_170105": _path_receipt(
            args.reference_root.resolve() / "hocap_170105.reference_kinematics_v2.npz"
        ),
        "reference_170650": _path_receipt(
            args.reference_root.resolve() / "hocap_170650.reference_kinematics_v2.npz"
        ),
    }
    if args.resume_checkpoint is not None:
        inputs["resume_checkpoint"] = _path_receipt(args.resume_checkpoint)
    elif stage == "C0":
        inputs["zero_g_selection"] = _selected_zero_g_checkpoint(mode, args.clip)
    gpu_probe = _gpu_probe()
    _write_json(args.output_root.resolve() / "resource_usage.json", {"preflight": gpu_probe})
    _write_json(output / "launch_progress.json", {"phase": "gpu_preflight_complete"})
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env: Any | None = None
    try:
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo_cfg.configure_stage16d_ppo26d(
            cfg,
            num_envs=args.num_envs,
            clip=args.clip,
            rsi=True,
            critical_dr=args.critical_dr,
        )
        ppo_cfg.configure_stage16d_contact_reward(
            cfg,
            mode=mode,
            reference_root=args.reference_root.resolve(),
            contact_reward_contract=contact_contract.resolve(),
            contact_mask_root=contact_mask_root.resolve(),
        )
        ppo_cfg.configure_stage16_contact_ready_rsi_v2(cfg, safe_bank_path=safe_bank)
        ppo_cfg.configure_stage16_p3_p4_curriculum(
            cfg, curriculum_contract_path=args.curriculum_contract.resolve(), stage=stage
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        environment_contract = env.contract_report()
        _validate_environment_contract(
            environment_contract,
            clip=args.clip,
            stage=stage,
            expected_physics=contract_physics,
        )
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        if args.resume_checkpoint is not None:
            initialization, resume = _restore_physical_checkpoint(
                trainer,
                checkpoint=args.resume_checkpoint.resolve(),
                clip=args.clip,
                num_envs=args.num_envs,
                mode=mode,
                stage=stage,
            )
            stage_samples = int(resume["physical_stage_samples"])
            physical_cumulative_samples = int(resume["physical_cumulative_samples"])
        else:
            selected = inputs["zero_g_selection"]
            assert isinstance(selected, dict)
            initialization = _restore_zero_g_checkpoint(
                trainer,
                checkpoint=Path(str(selected["selected_checkpoint"])),
                clip=args.clip,
                mode=mode,
            )
            initialization["selection"] = selected
            stage_samples = 0
            physical_cumulative_samples = 0
        training_config = {
            "schema_version": "Stage16P3GravityFrictionTrainingConfigV1",
            "scientific_label": "ENGINEERING_CURRICULUM_V1",
            "clip": args.clip,
            "contact_mode": mode.value,
            "mode_comparison_semantics": "PHYSICAL_PIPELINE_COMPARISON",
            "curriculum_stage": stage,
            "stage_budget": budget.as_dict(),
            "physical_stage_samples_start": stage_samples,
            "physical_cumulative_samples_start": physical_cumulative_samples,
            "policy_training_samples_start": trainer.cumulative_samples,
            "selected_num_envs": args.num_envs,
            "critical_dr": args.critical_dr,
            "curriculum_state": curriculum_state,
            "p3_entry": entry,
            "inputs": inputs,
            "environment": environment_contract,
            "curriculum_training_only": True,
        }
        _write_json(output / "training_config.json", training_config)
        _write_json(output / "launch_progress.json", {"phase": "environment_and_state_ready"})
        if args.smoke_only:
            metric = trainer.collect_and_update(env, rollout_length=args.smoke_rollout_steps)
            metric.pop("last_policy_observation")
            receipt = {
                "schema_version": "Stage16P3GravityFrictionPPOSmokeV1",
                "status": "PHYSICAL_PPO_SMOKE_PASS",
                "clip": args.clip,
                "contact_mode": mode.value,
                "stage": stage,
                "discarded_physical_samples": metric["samples"],
                "finite": metric["finite"],
                "ppo": metric["ppo"],
                "environment": env.contract_report(),
                "initialization": initialization,
            }
            _write_json(output / "smoke.json", receipt)
            print(json.dumps(receipt, sort_keys=True))
            return 0

        metrics_path = output / "training_metrics.jsonl"
        checkpoint_records: list[dict[str, Any]] = []
        checkpoint: Path | None = None
        pending = [
            target
            for target in budget.checkpoint_stage_samples
            if stage_samples < target <= budget.additional_samples
        ]
        with metrics_path.open("a", encoding="utf-8") as stream:
            while stage_samples < budget.additional_samples:
                remaining = budget.additional_samples - stage_samples
                rollout_steps = min(
                    Stage16DPPO26DTrainingConfigV1().rollout_length, remaining // args.num_envs
                )
                if rollout_steps <= 0:
                    raise RuntimeError("PHYSICAL_PPO_EXACT_SAMPLE_ROLLOUT_INVALID")
                metric = trainer.collect_and_update(env, rollout_length=rollout_steps)
                metric.pop("last_policy_observation")
                stage_samples += int(metric["samples"])
                physical_cumulative_samples += int(metric["samples"])
                metric.update(
                    {
                        "clip": args.clip,
                        "contact_mode": mode.value,
                        "curriculum_stage": stage,
                        "physical_stage_samples": stage_samples,
                        "physical_cumulative_samples": physical_cumulative_samples,
                        "policy_training_samples": trainer.cumulative_samples,
                    }
                )
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
                stream.flush()
                crossed = [target for target in pending if stage_samples >= target]
                for target in crossed:
                    path = (
                        output
                        / "checkpoints"
                        / (
                            f"stage16_p3_{mode.value}_{stage.lower()}_physical_"
                            f"{physical_cumulative_samples}_stage_{stage_samples}.pt"
                        )
                    )
                    _save_checkpoint(
                        path=path,
                        trainer=trainer,
                        environment_contract=env.contract_report(),
                        num_envs=args.num_envs,
                        stage=stage,
                        physical_stage_samples=stage_samples,
                        physical_cumulative_samples=physical_cumulative_samples,
                        mode=mode,
                        curriculum_state=curriculum_state,
                        initialization=initialization,
                        entry=entry,
                    )
                    checkpoint = path
                    checkpoint_records.append(
                        {
                            "stage_checkpoint_target_samples": target,
                            "actual_stage_samples": stage_samples,
                            "actual_physical_cumulative_samples": physical_cumulative_samples,
                            "checkpoint": str(path.resolve()),
                            "checkpoint_sha256": _sha256(path),
                        }
                    )
                    pending.remove(target)
        if stage_samples != budget.additional_samples or checkpoint is None:
            raise RuntimeError("PHYSICAL_PPO_STAGE_BUDGET_OR_CHECKPOINT_INCOMPLETE")
        result = {
            "schema_version": PHYSICAL_PPO_RESULT_SCHEMA,
            "status": "PHYSICAL_PPO_STAGE_COMPLETE",
            "clip": args.clip,
            "contact_mode": mode.value,
            "curriculum_stage": stage,
            "physical_stage_samples": stage_samples,
            "physical_cumulative_samples": physical_cumulative_samples,
            "policy_training_samples": trainer.cumulative_samples,
            "iterations": math.ceil(
                (stage_samples - int(training_config["physical_stage_samples_start"]))
                / (args.num_envs * Stage16DPPO26DTrainingConfigV1().rollout_length)
            ),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_milestones": checkpoint_records,
            "metrics": str(metrics_path.resolve()),
            "environment": env.contract_report(),
            "initialization": initialization,
            "curriculum_training_only": True,
        }
        _write_json(output / "checkpoint_milestones.json", checkpoint_records)
        _write_json(output / "training_result.json", result)
        _write_json(output / "launch_progress.json", {"phase": "stage_complete"})
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        _write_json(
            output / "training_failure.json",
            {
                "schema_version": "Stage16P3GravityFrictionPPOFailureV1",
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
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
