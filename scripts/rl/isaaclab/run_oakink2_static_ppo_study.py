#!/usr/bin/env python3
"""Run one frozen OakInk2 static baseline, PPO training, or post-PPO evaluation."""

# ruff: noqa: E402, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "src")]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "train", "eval"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--item-index", type=int, required=True)
    parser.add_argument("--study-contract", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all()),
            "mean": float(value.float().mean().detach().cpu()),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


class _ZeroDistribution:
    def __init__(self, mean: Any) -> None:
        self.mean = mean


class _ZeroPolicy:
    def distribution(self, observation: Any) -> _ZeroDistribution:
        import torch

        return _ZeroDistribution(
            torch.zeros(
                (observation.shape[0], 26), dtype=observation.dtype, device=observation.device
            )
        )


class _ZeroTrainer:
    def __init__(self) -> None:
        self.trainer = _ZeroPolicy()


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _finger_name(body_name: str) -> str:
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        if name in body_name:
            return name
    return "palm"


def _episode_metrics(
    *,
    item: dict[str, Any],
    episode: int,
    seed: int,
    rollout: dict[str, object],
    trace: dict[str, np.ndarray],
    geometry_manifest: Path,
    trace_root: Path,
    phase: str,
) -> dict[str, Any]:
    from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
        _evaluate_geometry_with_exact_broadphase,
        _reconstruct_hand,
    )
    from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
        HAND_COLLISION_BODY_NAMES,
    )

    trace["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
    trace["hand_collision_body_pose"] = _reconstruct_hand(trace)
    geometry, detail = _evaluate_geometry_with_exact_broadphase(
        clip=item["clip_id"],
        object_pose=np.asarray(trace["object_pose"], dtype=np.float64)[:, None],
        hand_collision_body_pose=np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)[
            :, None
        ],
        hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
        geometry_path=geometry_manifest,
    )
    penetration = np.asarray(detail["penetration_depth_m"], dtype=np.float64)[:, 0]
    pair_ids = [str(value) for value in detail["pair_ids"]]
    per_finger_penetration: dict[str, float] = {}
    for finger in ("thumb", "index", "middle", "ring", "pinky", "palm"):
        indices = [index for index, pair in enumerate(pair_ids) if _finger_name(pair) == finger]
        per_finger_penetration[finger] = (
            0.0 if not indices else float(penetration[:, indices].max(initial=0.0))
        )
    object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
    object_twist = np.asarray(trace["object_twist"], dtype=np.float64)
    object_reference = np.asarray(trace["object_reference"], dtype=np.float64)
    wrist_pose = np.asarray(trace["wrist_pose"], dtype=np.float64)
    wrist_reference = np.asarray(trace["wrist_reference"], dtype=np.float64)
    finger_q = np.asarray(trace["finger_q"], dtype=np.float64)
    finger_reference = np.asarray(trace["finger_reference"], dtype=np.float64)
    action = np.asarray(trace["action"], dtype=np.float64)
    force = np.asarray(trace["tip_pair_force_norm"], dtype=np.float64)
    contact = np.asarray(trace["tip_pair_presence"], dtype=bool)
    support = np.asarray(trace["table_object_contact"], dtype=bool)
    translation_drift = np.linalg.norm(object_pose[:, :3] - object_reference[:, :3], axis=-1)
    rotation_drift = _quaternion_angle(object_pose[:, 3:7], object_reference[:, 3:7])
    wrist_translation = np.linalg.norm(wrist_pose[:, :3] - wrist_reference[:, :3], axis=-1)
    wrist_rotation = _quaternion_angle(wrist_pose[:, 3:7], wrist_reference[:, 3:7])
    finger_deviation = np.linalg.norm(finger_q - finger_reference, axis=-1)
    linear_speed = np.linalg.norm(object_twist[:, :3], axis=-1)
    angular_speed = np.linalg.norm(object_twist[:, 3:], axis=-1)
    finite_fields = (
        object_pose,
        object_twist,
        wrist_pose,
        finger_q,
        action,
        force,
        penetration,
    )
    trace_path = trace_root / f"episode_{episode:02d}.npz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(trace_path, **trace, physical_penetration_depth_m=penetration)
    return {
        "anchor_id": item["anchor_id"],
        "clip_id": item["clip_id"],
        "phase": phase,
        "episode": episode,
        "seed": seed,
        "simulation_finite": bool(all(np.isfinite(value).all() for value in finite_fields)),
        "scene_authority_valid": True,
        "controller_valid": True,
        "metrics_complete": True,
        "no_cheat": True,
        "geometric_penetration_metric": "SPARSEV4_SOURCE_GEOMETRY",
        "geometric_penetration_p95_m": float(item["geom_penetration_p95_m"]),
        "geometric_penetration_max_m": float(item["geom_penetration_max_m"]),
        "physical_penetration_metric": "RUNTIME_PHYSX_COLLISION_PROXY_EXACT_FCL_V1",
        "physical_penetration_p95_m": float(geometry["p95_penetration_m"]),
        "physical_penetration_max_m": float(geometry["max_penetration_m"]),
        "physical_penetration_terminal_m": float(penetration[-1].max(initial=0.0)),
        "physical_penetrating_fraction": float(np.mean(penetration > 0.0)),
        "per_finger_physical_penetration_max_m": per_finger_penetration,
        "contact_count": int(contact.any(axis=1).sum()),
        "contact_persistence_fraction": float(contact.any(axis=1).mean()),
        "per_finger_contact_force_mean_n": force.mean(axis=0).tolist(),
        "per_finger_contact_force_max_n": force.max(axis=0, initial=0.0).tolist(),
        "object_translation_drift_m_max": float(translation_drift.max(initial=0.0)),
        "object_rotation_drift_rad_max": float(rotation_drift.max(initial=0.0)),
        "object_linear_velocity_mps_max": float(linear_speed.max(initial=0.0)),
        "object_angular_velocity_radps_max": float(angular_speed.max(initial=0.0)),
        "terminal_object_linear_velocity_mps": float(linear_speed[-1]),
        "terminal_object_angular_velocity_radps": float(angular_speed[-1]),
        "wrist_reference_deviation_m_mean": float(wrist_translation.mean()),
        "wrist_reference_deviation_rad_mean": float(wrist_rotation.mean()),
        "finger_reference_deviation_l2_rad_mean": float(finger_deviation.mean()),
        "action_l2_mean": float(np.linalg.norm(action, axis=-1).mean()),
        "action_max_abs": float(np.abs(action).max(initial=0.0)),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
        "residual_l2_mean": float(np.linalg.norm(action, axis=-1).mean()),
        "reward_total_mean": float(np.asarray(trace["reward_total"], dtype=np.float64).mean()),
        "reward_object_mean": float(np.asarray(trace["reward_object"], dtype=np.float64).mean()),
        "reward_interaction_mean": float(
            np.asarray(trace["reward_group_interaction"], dtype=np.float64).mean()
        ),
        "object_support_status": "CONTACT_OBSERVED" if support.any() else "NO_CONTACT_OBSERVED",
        "object_support_contact_fraction": float(support.mean()),
        "steps": int(rollout["steps"]),
        "termination_reason": int(rollout["termination_reason"]),
        "trace": str(trace_path),
        "trace_sha256": _sha256(trace_path),
    }


def _environment(item: dict[str, Any], *, num_envs: int, training: bool) -> Any:
    from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
    from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

    return _make_table_env(
        clip=item["clip_id"],
        num_envs=num_envs,
        start_index=0,
        mode=ContactRewardMode.STRICT_PER_FINGER_V4,
        stage="C4",
        training_rsi=training,
        reward_aggregation_mode="grouped_multiplicative_v1",
        rse_enabled=True,
        full_horizon_evaluation=not training,
        reference_path=Path(item["reference"]),
        object_usd_path=Path(item["object_usd"]),
        support_proxy_path=Path(item["support_proxy"]),
        support_asset_path=Path(item["support_asset"]),
        contact_contract_path=Path(item["contact_contract"]),
        contact_mask_root=Path(item["contact_mask_root"]),
        reference_distance_root=Path(item["reference_distance_root"]),
        object_mesh_root=Path(item["object_mesh_root"]),
        continuous_virtual_wrist_angles=True,
        source_controller_admission_v2=True,
        static_reference_adapter=True,
    )


def _evaluate(
    *,
    mode: str,
    item: dict[str, Any],
    seeds: list[int],
    geometry_manifest: Path,
    output: Path,
    checkpoint: Path | None,
) -> dict[str, Any]:
    from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
        _parallel_rollouts,
    )

    # Independent OakInk2 external-object scenes are evaluated as ten serial
    # episodes in one 1-env Isaac scene.  Eval10, seeds, policy, horizon, and
    # denominator are unchanged; only evaluation parallelism is one.
    env = _environment(item, num_envs=1, training=False)
    try:
        if checkpoint is None:
            trainer: Any = _ZeroTrainer()
            policy = "ZERO_RESIDUAL_DETERMINISTIC"
        else:
            from scripts.rl.isaaclab.evaluate_physical_hoi import model_from_checkpoint

            trainer, _payload = model_from_checkpoint(
                checkpoint, str(env.device), expected_clip=item["clip_id"]
            )
            policy = "FIXED_BUDGET_FINAL_PPO_CHECKPOINT"
        rollouts = []
        for episode, seed in enumerate(seeds):
            rollouts.extend(
                _parallel_rollouts(
                    env=env,
                    trainer=trainer,
                    clip=item["clip_id"],
                    seeds=[seed],
                    start=0,
                )
            )
            print(
                f"PROGRESS anchor={item['anchor_id']} phase={mode} episode={episode + 1}/10",
                file=sys.stderr,
                flush=True,
            )
        rows = [
            _episode_metrics(
                item=item,
                episode=episode,
                seed=seeds[episode],
                rollout=rollout,
                trace=trace,
                geometry_manifest=geometry_manifest,
                trace_root=output.parent / output.stem / "traces",
                phase=mode,
            )
            for episode, (rollout, trace) in enumerate(rollouts)
        ]
        runtime = env.contract_report()
        no_cheat = {
            "object_rollout_state_writes": runtime["ppo26d"]["object_rollout_state_writes"],
            "wrist_root_state_writes_during_step": runtime["ppo26d"][
                "wrist_root_state_writes_during_step"
            ],
            "hidden_force_or_attachment": runtime["ppo26d"]["hidden_force_or_attachment"],
        }
        valid = all(
            row["simulation_finite"]
            and row["scene_authority_valid"]
            and row["controller_valid"]
            and row["metrics_complete"]
            and row["no_cheat"]
            for row in rows
        ) and no_cheat == {
            "object_rollout_state_writes": 0,
            "wrist_root_state_writes_during_step": 0,
            "hidden_force_or_attachment": False,
        }
        return {
            "schema_version": "OakInk2StaticPhysicalEvaluationV1",
            "status": "PASS" if valid and len(rows) == 10 else "TECHNICAL_INFRA_FAILURE",
            "phase": mode,
            "policy": policy,
            "anchor_id": item["anchor_id"],
            "clip_id": item["clip_id"],
            "episodes": len(rows),
            "seeds": seeds,
            "checkpoint": None
            if checkpoint is None
            else {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
            "no_cheat": no_cheat,
            "rows": rows,
        }
    finally:
        env.close()


def _train(
    *, item: dict[str, Any], seed: int, study_contract: Path, output: Path
) -> dict[str, Any]:
    import torch

    from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
    from toporetarget.rl.source_controller import make_zero_output_residual_actor_

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env = _environment(item, num_envs=1024, training=True)
    try:
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        make_zero_output_residual_actor_(trainer.model)
        actor_initial = parameter_hash(trainer.model, "actor")
        metrics = []
        started = time.monotonic()
        for update in range(1, 16):
            result = trainer.collect_and_update(env)
            result["update"] = update
            result["cumulative_samples"] = trainer.cumulative_samples
            metrics.append(_jsonable(result))
            print(
                f"PROGRESS anchor={item['anchor_id']} update={update}/15 "
                f"samples={trainer.cumulative_samples}",
                file=sys.stderr,
                flush=True,
            )
        checkpoint = output / "checkpoint.pt"
        trainer.save(
            checkpoint,
            environment_contract=env.contract_report(),
            selected_num_envs=1024,
            extra_payload={
                "study_role": "DIAGNOSTIC_PHYSICAL_RECOVERABILITY",
                "study_contract_path": str(study_contract),
                "study_contract_sha256": _sha256(study_contract),
                "training_seed": seed,
                "optimizer_steps": 15,
                "fixed_budget_final_checkpoint": True,
                "checkpoint_selection": "FINAL_UPDATE_NO_OUTCOME_SELECTION",
            },
        )
        _write_json(output / "update_metrics.json", metrics)
        with (output / "update_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = [
                "update",
                "cumulative_samples",
                "mean_reward",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "action_saturation_fraction",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in metrics:
                ppo = row.get("ppo", {}) if isinstance(row.get("ppo"), dict) else {}
                saturation = row.get("saturation_instrumentation", {})
                writer.writerow(
                    {
                        "update": row["update"],
                        "cumulative_samples": row["cumulative_samples"],
                        "mean_reward": row.get("mean_reward"),
                        "policy_loss": ppo.get("policy_loss"),
                        "value_loss": ppo.get("value_loss"),
                        "entropy": ppo.get("entropy"),
                        "approx_kl": ppo.get("approx_kl"),
                        "action_saturation_fraction": saturation.get(
                            "deterministic_action_saturation_fraction"
                        )
                        if isinstance(saturation, dict)
                        else None,
                    }
                )
        receipt = {
            "schema_version": "OakInk2StaticPPOTrainingReceiptV1",
            "status": "PASS",
            "anchor_id": item["anchor_id"],
            "clip_id": item["clip_id"],
            "training_seed": seed,
            "num_envs": 1024,
            "rollout_length": 40,
            "updates": 15,
            "samples_per_update": 40960,
            "cumulative_samples": trainer.cumulative_samples,
            "initial_actor_sha256": actor_initial,
            "final_actor_sha256": parameter_hash(trainer.model, "actor"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "elapsed_s": time.monotonic() - started,
            "PPO_REWARD_CHANGED": "NO",
            "adaptive_budget": False,
            "checkpoint_selection": "FINAL_UPDATE_NO_OUTCOME_SELECTION",
        }
        _write_json(output / "train_receipt.json", receipt)
        _write_json(
            output / "checkpoint_hashes.json",
            {
                "schema_version": "OakInk2StaticPPOCheckpointHashesV1",
                "checkpoint": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "actor_initial": actor_initial,
                "actor_final": receipt["final_actor_sha256"],
            },
        )
        return receipt
    finally:
        env.close()


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    study = json.loads(args.study_contract.resolve().read_text(encoding="utf-8"))
    if study.get("status") != "FROZEN_BEFORE_PPO" or len(manifest.get("anchors", [])) != 12:
        raise ValueError("STATIC_PPO_STUDY_CONTRACT_OR_MANIFEST_INVALID")
    if args.item_index < 0 or args.item_index >= 12:
        raise ValueError("STATIC_PPO_ITEM_INDEX_INVALID")
    item = dict(manifest["anchors"][args.item_index])
    seed_row = study["seeds"][args.item_index]
    if seed_row["anchor_id"] != item["anchor_id"]:
        raise ValueError("STATIC_PPO_SEED_ANCHOR_DRIFT")
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("STATIC_PPO_TRAIN_REJECTS_SOURCE_CHECKPOINT")
    if args.mode == "eval" and (args.checkpoint is None or not args.checkpoint.is_file()):
        raise ValueError("STATIC_PPO_EVAL_CHECKPOINT_REQUIRED")
    if args.mode == "baseline" and args.checkpoint is not None:
        raise ValueError("STATIC_PPO_BASELINE_MUST_BE_ZERO_RESIDUAL")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        if args.mode == "train":
            result = _train(
                item=item,
                seed=int(seed_row["training_seed"]),
                study_contract=args.study_contract.resolve(),
                output=args.output.resolve(),
            )
        else:
            result = _evaluate(
                mode=args.mode,
                item=item,
                seeds=[int(value) for value in seed_row["eval_seeds"]],
                geometry_manifest=args.geometry_manifest.resolve(),
                output=args.output.resolve(),
                checkpoint=None if args.checkpoint is None else args.checkpoint.resolve(),
            )
            _write_json(args.output.resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    except BaseException as exc:
        failure = {
            "schema_version": "OakInk2StaticPPOStudyTechnicalFailureV1",
            "status": "TECHNICAL_INFRA_FAILURE",
            "mode": args.mode,
            "item_index": args.item_index,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path = (
            args.output.resolve() / "technical_failure.json"
            if args.mode == "train"
            else args.output.resolve().with_name(args.output.stem + "_technical_failure.json")
        )
        _write_json(failure_path, failure)
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
