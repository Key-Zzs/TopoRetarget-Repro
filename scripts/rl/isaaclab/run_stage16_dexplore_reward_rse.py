#!/usr/bin/env python3
"""Run the gated no-step sanity check or bounded Stage16 Dexplore refinement."""

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

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.train_stage16_p3_physical_curriculum import _restore_zero_g_checkpoint
from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from toporetarget.rl.ppo.policy_preservation import state_hash
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_dexplore_reward_rse"
RUN_ROOT = REPO_ROOT / ".local/runs/stage16_dexplore_reward_rse"
SOURCE_MANIFEST = (
    REPO_ROOT
    / ".local/reports/stage16_frozen_source_policy_gravity_sweep/sources/v4_hocap_170105.json"
)
OFFLINE_GATE = REPORT_ROOT / "offline/offline_gate.json"
EVALUATOR = REPO_ROOT / "scripts/evaluation/evaluate_stage16_dexplore_reward_rse.py"
MAX_UPDATES = 10
NUM_ENVS = 1024
SAMPLES_PER_UPDATE = NUM_ENVS * 40
CHECKPOINT_SCHEMA = "Stage16DGroupedMultiplicativeRSECheckpointV1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("runtime-sanity", "train"))
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("DEXPLORE_TRAINING_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_failure(scope: str, error: BaseException) -> None:
    path = REPORT_ROOT / "technical_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "scope": scope,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _source() -> tuple[Path, dict[str, object]]:
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    checkpoint = Path(str(source["checkpoint"])).resolve()
    if (
        source.get("id") != "v4_hocap_170105"
        or source.get("contact_mode") != "strict_per_finger_v4"
        or not checkpoint.is_file()
        or _sha256(checkpoint) != source.get("checkpoint_sha256")
        or source.get("checkpoint_sha256")
        != "90c7ddea923f2ba69b141f85e9c72680cddd5fb5a1d902d6cac086f73ce4c261"
    ):
        raise RuntimeError("DEXPLORE_HEALTHY_V4_SOURCE_AUTHORITY_INVALID")
    return checkpoint, source


def _require_offline_gate() -> dict[str, object]:
    gate = json.loads(OFFLINE_GATE.read_text(encoding="utf-8"))
    if (
        gate.get("classification") != "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED"
        or gate.get("passed") is not True
        or gate.get("ppo_training_run_authorized") is not True
    ):
        raise RuntimeError("DEXPLORE_OFFLINE_GATE_NOT_AUTHORIZED")
    return gate


def _gpu_probe() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _make_env() -> Any:
    from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env

    return _make_table_env(
        clip="hocap_170105",
        num_envs=NUM_ENVS,
        start_index=0,
        mode=ContactRewardMode.STRICT_PER_FINGER_V4,
        stage="C4",
        training_rsi=True,
        reward_aggregation_mode="grouped_multiplicative_v1",
        rse_enabled=True,
    )


def _runtime_contract(env: Any) -> dict[str, object]:
    report = env.contract_report()
    physics = report["gravity_friction_curriculum"]
    ppo = report["ppo26d"]
    wrist = report["finite_virtual_6d_wrist_actuator"]
    hand_gravity_off = bool(env.cfg.robot.spawn.rigid_props.disable_gravity)
    if (
        physics.get("stage") != "C4"
        or physics.get("gravity_scale") != 1.0
        or physics.get("friction_scale") != 1.0
        or physics.get("support") != "finite_inferred_table_proxy_v1"
        or physics.get("table_actor_active") is not True
        or physics.get("mid_trajectory_rsi") != "uniform[0,320]"
        or ppo.get("fixed_clip") != "hocap_170105"
        or ppo.get("reward", {}).get("identifier") != "TopoRetargetReferenceTrackingReward26DV4"
        or ppo.get("reward_aggregation", {}).get("mode") != "grouped_multiplicative_v1"
        or ppo.get("rse", {}).get("enabled") is not True
        or ppo.get("rse", {}).get("uniform_rsi_preserved") is not True
        or wrist.get("identifier") != "finite_virtual_6d_wrist_actuator_v1"
        or wrist.get("authority_enabled") is not True
        or not hand_gravity_off
    ):
        raise RuntimeError("DEXPLORE_C4_RUNTIME_CONTRACT_DRIFT")
    return {
        "environment": report,
        "physics_stage": "C4",
        "object_gravity": "1g",
        "nominal_friction": True,
        "table_active": True,
        "hand_gravity_off": hand_gravity_off,
        "virtual_wrist_gravity_off": hand_gravity_off,
        "fixed_wrist_repaired_runtime": True,
        "training_reset": "UNIFORM_RSI_[0,320]",
    }


def _restore_source(env: Any, checkpoint: Path) -> tuple[PPO26DTrainer, dict[str, object]]:
    trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
    initialization = _restore_zero_g_checkpoint(
        trainer,
        checkpoint=checkpoint,
        clip="hocap_170105",
        mode=ContactRewardMode.STRICT_PER_FINGER_V4,
    )
    return trainer, initialization


def _gradient_sanity(trainer: PPO26DTrainer, batch_path: Path) -> dict[str, object]:
    batch = load_checkpoint(batch_path, map_location=trainer.trainer.device)
    observations = batch["observations"].flatten(0, 1)
    actions = batch["actions"].flatten(0, 1)
    old_log_probs = batch["old_log_probs"].flatten(0, 1)
    advantages = batch["advantages"].flatten(0, 1)
    returns = batch["returns"].flatten(0, 1)
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1.0e-8
    )
    minibatch_size = len(observations) // trainer.training_contract.minibatches
    selected = slice(0, minibatch_size)
    actor_before = parameter_hash(trainer.model, "actor")
    critic_before = parameter_hash(trainer.model, "critic")
    optimizer_before = state_hash(trainer.trainer.optimizer.state_dict())
    normalizer_before = state_hash(trainer.trainer.normalizer.state_dict())
    distribution = trainer.trainer.distribution(observations[selected])
    new_log_probs = distribution.log_prob(actions[selected])
    ratio = torch.exp(new_log_probs - old_log_probs[selected])
    surrogate = torch.minimum(
        ratio * normalized_advantages[selected],
        torch.clamp(
            ratio,
            1.0 - trainer.training_contract.ppo_clip,
            1.0 + trainer.training_contract.ppo_clip,
        )
        * normalized_advantages[selected],
    )
    actor_loss = -surrogate.mean()
    value = trainer.model.value(trainer.trainer.normalizer.normalize(observations[selected]))
    critic_loss = torch.nn.functional.mse_loss(value, returns[selected])
    entropy = distribution.entropy().mean()
    loss = actor_loss + 0.5 * critic_loss - trainer.training_contract.entropy_coefficient * entropy
    trainer.trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None
    ]
    gradient_finite = bool(gradients) and all(
        bool(torch.isfinite(value).all()) for value in gradients
    )
    gradient_norm = float(
        torch.linalg.vector_norm(torch.stack([value.detach().norm() for value in gradients])).cpu()
    )
    trainer.trainer.optimizer.zero_grad(set_to_none=True)
    actor_after = parameter_hash(trainer.model, "actor")
    critic_after = parameter_hash(trainer.model, "critic")
    optimizer_after = state_hash(trainer.trainer.optimizer.state_dict())
    normalizer_after = state_hash(trainer.trainer.normalizer.state_dict())
    return {
        "schema_version": "Stage16DexploreNoStepGradientSanityV1",
        "backward_executed": True,
        "optimizer_step_executed": False,
        "minibatch_size": minibatch_size,
        "loss": float(loss.detach().cpu()),
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "gradient_finite": gradient_finite,
        "gradient_norm": gradient_norm,
        "actor_hash_before": actor_before,
        "actor_hash_after": actor_after,
        "critic_hash_before": critic_before,
        "critic_hash_after": critic_after,
        "optimizer_hash_before": optimizer_before,
        "optimizer_hash_after": optimizer_after,
        "normalizer_hash_before": normalizer_before,
        "normalizer_hash_after": normalizer_after,
        "parameters_unchanged": actor_before == actor_after and critic_before == critic_after,
        "optimizer_unchanged": optimizer_before == optimizer_after,
        "normalizer_unchanged": normalizer_before == normalizer_after,
    }


def _reward_scale(batch_path: Path, trainer: PPO26DTrainer) -> dict[str, object]:
    batch = load_checkpoint(batch_path, map_location="cpu")
    product = batch["rewards"].double().flatten()
    legacy = batch["reward_terms"]["total_legacy_additive"].double().flatten()

    def stats(value: torch.Tensor) -> dict[str, float]:
        return {
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
            "median": float(value.median()),
            "p05": float(torch.quantile(value, 0.05)),
            "p95": float(torch.quantile(value, 0.95)),
            "min": float(value.min()),
            "max": float(value.max()),
        }

    return {
        "schema_version": "Stage16DexploreRewardScaleAuditV1",
        "advantage_normalization_enabled": trainer.training_contract.advantage_normalization,
        "reward_rescaled": False,
        "grouped_multiplicative": stats(product),
        "same_rollout_legacy_additive": stats(legacy),
        "mean_ratio_product_to_legacy": float(product.mean() / legacy.mean()),
        "ppo_contract_comparable": bool(product.std(unbiased=False) > 0.0),
        "interpretation": (
            "Actor advantages retain the frozen per-batch normalization. Critic targets use the "
            "literal bounded product reward; no hidden scaling is applied."
        ),
    }


def _runtime_sanity(env: Any, trainer: PPO26DTrainer) -> dict[str, object]:
    root = RUN_ROOT / "runtime_sanity"
    if root.exists() or (REPORT_ROOT / "runtime_sanity/gate.json").exists():
        raise FileExistsError("DEXPLORE_RUNTIME_SANITY_ALREADY_EXISTS")
    root.mkdir(parents=True)
    exact_batch = root / "exact_batch.pt"
    metric = trainer.collect_and_update(env, update_policy=False, exact_batch_path=exact_batch)
    metric.pop("last_policy_observation")
    gradient = _gradient_sanity(trainer, exact_batch)
    reward_scale = _reward_scale(exact_batch, trainer)
    passed = bool(
        all(metric["finite"].values())
        and gradient["gradient_finite"]
        and gradient["parameters_unchanged"]
        and gradient["optimizer_unchanged"]
        and gradient["normalizer_unchanged"]
        and reward_scale["ppo_contract_comparable"]
        and metric["ppo"]["optimizer_steps"] == 0.0
    )
    _write_json(REPORT_ROOT / "runtime_sanity/reward_scale.json", reward_scale)
    _write_json(REPORT_ROOT / "runtime_sanity/gradient.json", gradient)
    gate = {
        "schema_version": "Stage16DexploreRuntimeSanityGateV1",
        "classification": "RUNTIME_GRADIENT_SANITY_PASS"
        if passed
        else "RUNTIME_GRADIENT_SANITY_FAIL",
        "passed": passed,
        "ppo_training_authorized": passed,
        "exact_batch": {"path": str(exact_batch.resolve()), "sha256": _sha256(exact_batch)},
        "rollout": metric,
        "gradient": gradient,
        "reward_scale": reward_scale,
    }
    _write_json(REPORT_ROOT / "runtime_sanity/gate.json", gate)
    return gate


def _require_runtime_gate() -> dict[str, object]:
    path = REPORT_ROOT / "runtime_sanity/gate.json"
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("ppo_training_authorized") is not True:
        raise RuntimeError("DEXPLORE_RUNTIME_GATE_NOT_AUTHORIZED")
    return gate


def _run_evaluation(
    *, checkpoint: Path, output: Path, episodes: int, update: int, samples: int
) -> dict[str, object]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--accept-eula",
        "--clip",
        "hocap_170105",
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output",
        str(output.resolve()),
        "--episodes",
        str(episodes),
        "--update",
        str(update),
        "--samples",
        str(samples),
    ]
    environment = dict(os.environ)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, text=True, capture_output=True
    )
    _write_json(
        output.parent / f"{output.name}_driver.json",
        {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-12000:],
            "stderr_tail": completed.stderr[-12000:],
        },
    )
    technical_failure = output / "technical_failure.json"
    if completed.returncode != 0 or technical_failure.is_file():
        raise RuntimeError(f"DEXPLORE_EVALUATION_FAILED:{output}")
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"DEXPLORE_REPORT_LINK_EXISTS:{link}")
    link.symlink_to(target.resolve())


def _progress_row(
    update: int, samples: int, metric: dict[str, Any], evaluation: dict[str, Any]
) -> dict[str, object]:
    counts = evaluation["counts"]
    groups = evaluation["group_means"]
    return {
        "update": update,
        "samples": samples,
        "PF": counts["PF"],
        "lift": counts["lift"],
        "DF_pose": counts["DF_pose"],
        "DF_linear": counts["DF_linear"],
        "DF_angular_v2": counts["DF_angular_v2"],
        "R_obj": groups["R_obj"],
        "R_hand": groups["R_hand"],
        "R_int": groups["R_int"],
        "R_reg": groups["R_reg"],
        "R_total": groups["R_total"],
        "kappa": metric["reference"]["rse"]["kappa"],
        "RSE_termination_rate": metric["termination"]["rse_primary_failure_rate"],
        "first_contact": evaluation["timing"]["first_contact_median"],
        "persistent_multi_contact": evaluation["timing"]["persistent_multi_contact_median"],
        "LIFT": evaluation["timing"]["LIFT"],
        "pre_LIFT_margin": evaluation["timing"]["pre_LIFT_margin_median"],
        "support_transfer": evaluation["support_transfer_episodes"],
        "lift_dz_m": evaluation["lift_dz_mean_m"],
        "table_object_contact_fraction": float(
            np.mean(
                [
                    float(row["table_object_contact_fraction"])
                    for row in csv.DictReader(
                        (Path(evaluation["traces"][0]["path"]).parents[1] / "per_episode.csv").open(
                            newline="", encoding="utf-8"
                        )
                    )
                ]
            )
        ),
    }


def _recovered_metric(batch_path: Path, checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    batch = load_checkpoint(batch_path, map_location="cpu")
    telemetry = batch["termination_telemetry"]
    failures = telemetry["rse_primary_failure"].float()
    rse = checkpoint_payload["environment_contract"]["ppo26d"]["rse"]
    return {
        "reference": {"rse": rse},
        "termination": {"rse_primary_failure_rate": float(failures.mean())},
        "finite": {
            name: bool(torch.isfinite(batch[name]).all())
            for name in ("rewards", "returns", "advantages", "values", "old_log_probs")
        },
        "exact_batch_recovery": True,
        "advantage_diagnostic": batch["advantage_diagnostic"],
        "return_diagnostic": batch["return_diagnostic"],
        "value_diagnostic": batch["value_diagnostic"],
    }


def _train(
    env: Any,
    trainer: PPO26DTrainer,
    source: dict[str, object],
    *,
    resume_payload: dict[str, Any] | None,
) -> dict[str, object]:
    if resume_payload is None and (
        (RUN_ROOT / "training").exists() or (REPORT_ROOT / "training/progression.csv").exists()
    ):
        raise FileExistsError("DEXPLORE_TRAINING_NAMESPACE_EXISTS")
    rows: list[dict[str, object]] = []
    stage_samples = (
        0 if resume_payload is None else int(resume_payload["dexplore_refinement_samples"])
    )
    completed_updates = (
        0 if resume_payload is None else int(resume_payload["dexplore_refinement_update"])
    )
    accepted: dict[str, object] | None = None
    if resume_payload is not None:
        update = completed_updates
        run_update = RUN_ROOT / "training" / f"U{update:02d}"
        report_update = REPORT_ROOT / "training" / f"U{update:02d}"
        checkpoint_path = run_update / "checkpoint/checkpoint.pt"
        batch_path = run_update / "exact_batch/exact_batch.pt"
        if not checkpoint_path.is_file() or not batch_path.is_file():
            raise FileNotFoundError("DEXPLORE_RESUME_DURABLE_UPDATE_ARTIFACT_MISSING")
        metric = _recovered_metric(batch_path, resume_payload)
        evaluation = _run_evaluation(
            checkpoint=checkpoint_path,
            output=report_update / "eval10",
            episodes=10,
            update=update,
            samples=stage_samples,
        )
        rows.append(_progress_row(update, stage_samples, metric, evaluation))
        receipt = {
            "schema_version": "Stage16DexploreRewardRSEUpdateReceiptV1",
            "update": update,
            "samples": stage_samples,
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "exact_batch": {"path": str(batch_path.resolve()), "sha256": _sha256(batch_path)},
            "state_hashes": {
                "actor": parameter_hash(trainer.model, "actor"),
                "critic": parameter_hash(trainer.model, "critic"),
                "optimizer": state_hash(resume_payload["optimizer"]),
                "normalizer": state_hash(resume_payload["observation_normalization"]),
                "rng": state_hash(resume_payload["rng"]),
            },
            "training": metric,
            "training_health": "RECOVERED_FROM_DURABLE_CHECKPOINT_AND_EXACT_BATCH",
            "eval10": evaluation,
            "evaluation_after_durable_checkpoint": True,
        }
        _write_json(report_update / "receipt.json", receipt)
        _write_csv(REPORT_ROOT / "training/progression.csv", rows)
        if int(evaluation["counts"]["PF"]) == 10:
            confirm = _run_evaluation(
                checkpoint=checkpoint_path,
                output=REPORT_ROOT / "confirm20" / f"U{update:02d}",
                episodes=20,
                update=update,
                samples=stage_samples,
            )
            receipt["confirm20"] = confirm
            _write_json(report_update / "receipt.json", receipt)
            if confirm["accepted"]:
                accepted = {
                    "update": update,
                    "samples": stage_samples,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "confirm20": confirm,
                }
                _write_json(REPORT_ROOT / "confirm20/best_accepted_checkpoint.json", accepted)
    for update in range(completed_updates + 1, MAX_UPDATES + 1):
        if accepted is not None:
            break
        run_update = RUN_ROOT / "training" / f"U{update:02d}"
        report_update = REPORT_ROOT / "training" / f"U{update:02d}"
        checkpoint_path = run_update / "checkpoint/checkpoint.pt"
        batch_path = run_update / "exact_batch/exact_batch.pt"
        checkpoint_path.parent.mkdir(parents=True)
        batch_path.parent.mkdir(parents=True)
        metric = trainer.collect_and_update(env, exact_batch_path=batch_path)
        metric.pop("last_policy_observation")
        if int(metric["samples"]) != SAMPLES_PER_UPDATE or not all(metric["finite"].values()):
            raise RuntimeError("DEXPLORE_TRAINING_UPDATE_CONTRACT_INVALID")
        stage_samples += int(metric["samples"])
        payload = trainer.checkpoint_payload(
            environment_contract=env.contract_report(),
            selected_num_envs=NUM_ENVS,
            extra_payload={
                "dexplore_refinement_update": update,
                "dexplore_refinement_samples": stage_samples,
                "source_v4_checkpoint": source["checkpoint"],
                "source_v4_checkpoint_sha256": source["checkpoint_sha256"],
                "reward_mode": "grouped_multiplicative_v1",
                "rse_enabled": True,
                "max_updates": MAX_UPDATES,
            },
        )
        payload["schema_version"] = CHECKPOINT_SCHEMA
        save_checkpoint(checkpoint_path, payload)
        _link(checkpoint_path, report_update / "checkpoint/checkpoint.pt")
        _link(batch_path, report_update / "exact_batch/exact_batch.pt")
        evaluation = _run_evaluation(
            checkpoint=checkpoint_path,
            output=report_update / "eval10",
            episodes=10,
            update=update,
            samples=stage_samples,
        )
        row = _progress_row(update, stage_samples, metric, evaluation)
        rows.append(row)
        receipt = {
            "schema_version": "Stage16DexploreRewardRSEUpdateReceiptV1",
            "update": update,
            "samples": stage_samples,
            "checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            },
            "exact_batch": {"path": str(batch_path.resolve()), "sha256": _sha256(batch_path)},
            "state_hashes": {
                "actor": parameter_hash(trainer.model, "actor"),
                "critic": parameter_hash(trainer.model, "critic"),
                "optimizer": state_hash(payload["optimizer"]),
                "normalizer": state_hash(payload["observation_normalization"]),
                "rng": state_hash(payload["rng"]),
            },
            "training": metric,
            "eval10": evaluation,
            "evaluation_after_durable_checkpoint": True,
        }
        _write_json(report_update / "receipt.json", receipt)
        _write_csv(REPORT_ROOT / "training/progression.csv", rows)
        if int(evaluation["counts"]["PF"]) == 10:
            confirm = _run_evaluation(
                checkpoint=checkpoint_path,
                output=REPORT_ROOT / "confirm20" / f"U{update:02d}",
                episodes=20,
                update=update,
                samples=stage_samples,
            )
            receipt["confirm20"] = confirm
            _write_json(report_update / "receipt.json", receipt)
            if confirm["accepted"]:
                accepted = {
                    "update": update,
                    "samples": stage_samples,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "confirm20": confirm,
                }
                _write_json(REPORT_ROOT / "confirm20/best_accepted_checkpoint.json", accepted)
                break
        _write_json(
            RUN_ROOT / "training/progress.json",
            {"completed_updates": update, "samples": stage_samples, "accepted": accepted},
        )
    result = {
        "schema_version": "Stage16DexploreRewardRSEBoundedTrainingV1",
        "PPO_TRAINING_RUN": True,
        "PPO_MAX_UPDATES": MAX_UPDATES,
        "PPO_UPDATES_ACTUALLY_RUN": int(rows[-1]["update"]),
        "ACTUAL_SAMPLES": stage_samples,
        "accepted": accepted is not None,
        "best_accepted": accepted,
        "status": (
            "MULTIPLICATIVE_RSE_REFINEMENT_ACCEPTED"
            if accepted is not None
            else "BOUNDED_U10_COMPLETE_NO_ACCEPTED_CHECKPOINT"
        ),
    }
    _write_json(RUN_ROOT / "training/complete.json", result)
    return result


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.num_envs != NUM_ENVS:
        raise ValueError("DEXPLORE_RUN_REQUIRES_EULA_AND_1024_ENVS")
    _require_offline_gate()
    source_path, source = _source()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(
        REPORT_ROOT / "resource_usage.json",
        {"before": _gpu_probe(), "mode": args.mode, "num_envs": NUM_ENVS},
    )
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        if args.mode == "train":
            _require_runtime_gate()
        env = _make_env()
        env.reset(seed=20260822)
        runtime = _runtime_contract(env)
        trainer, initialization = _restore_source(env, source_path)
        resume_payload = None
        if args.resume_checkpoint is not None:
            if args.mode != "train":
                raise ValueError("DEXPLORE_RESUME_ONLY_VALID_FOR_TRAIN")
            resume_path = args.resume_checkpoint.resolve()
            resume_payload = load_checkpoint(resume_path, map_location=env.device)
            if (
                resume_payload.get("schema_version") != CHECKPOINT_SCHEMA
                or resume_payload.get("source_v4_checkpoint_sha256") != source["checkpoint_sha256"]
                or not 1 <= int(resume_payload.get("dexplore_refinement_update", 0)) <= MAX_UPDATES
            ):
                raise RuntimeError("DEXPLORE_RESUME_CHECKPOINT_CONTRACT_INVALID")
            trainer.model.load_state_dict(resume_payload["actor_critic"])
            trainer.trainer.optimizer.load_state_dict(resume_payload["optimizer"])
            trainer.trainer.normalizer.load_state_dict(resume_payload["observation_normalization"])
            trainer.trainer.normalizer.training = True
            trainer.cumulative_samples = int(resume_payload["cumulative_samples"])
            restore_rng_state(resume_payload["rng"])
            rse = resume_payload["environment_contract"]["ppo26d"]["rse"]
            env.restore_rse_state(
                fail_count=int(rse["fail_count"]), total_count=int(rse["total_count"])
            )
            initialization["technical_resume"] = {
                "checkpoint": str(resume_path),
                "checkpoint_sha256": _sha256(resume_path),
                "update": resume_payload["dexplore_refinement_update"],
                "samples": resume_payload["dexplore_refinement_samples"],
                "rse_state_restored": True,
            }
        _write_json(
            RUN_ROOT / f"{args.mode}_contract.json",
            {
                "source": source,
                "initialization": initialization,
                "runtime": runtime,
                "ppo_contract": asdict(trainer.training_contract),
                "MAX_UPDATES": MAX_UPDATES,
                "SAMPLES_PER_UPDATE": SAMPLES_PER_UPDATE,
            },
        )
        result = (
            _runtime_sanity(env, trainer)
            if args.mode == "runtime-sanity"
            else _train(env, trainer, source, resume_payload=resume_payload)
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        _append_failure(args.mode, error)
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
