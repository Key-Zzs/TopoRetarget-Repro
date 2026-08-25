#!/usr/bin/env python3
"""Qualify the production zero-residual source controller under frozen safety gates."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.audit_stage16_zero_g_frozen_actor_contact import _full_start
from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
    FROZEN_GATES,
    GEOMETRY,
    _evaluate_geometry_with_exact_broadphase,
    _inter_finger_penetration,
    _load_gate,
    _parallel_rollouts,
    _reconstruct_hand,
    _seeds,
)
from toporetarget.rl.contact_skill_collapse import command_tracking_metrics
from toporetarget.rl.environments.isaaclab_backend.explicit_virtual_wrist import (
    serial_xyz_singularity_margin_deg,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.source_controller import SourceControllerSafetyContractV1

DEFAULT_CONTRACT = REPO_ROOT / "configs/contracts/source_controller_auto_v1.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--optimizer-steps", type=int, default=0)
    parser.add_argument("--training-samples", type=int, default=0)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--object-usd", type=Path)
    parser.add_argument("--support-proxy", type=Path)
    parser.add_argument("--support-asset", type=Path)
    parser.add_argument("--contact-contract", type=Path)
    parser.add_argument("--contact-mask-root", type=Path)
    parser.add_argument("--reference-distance-root", type=Path)
    parser.add_argument("--object-mesh-root", type=Path)
    parser.add_argument("--runtime-geometry-manifest", type=Path)
    parser.add_argument("--frozen-evaluation-gates", type=Path)
    parser.add_argument("--seed-manifest", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_EMPTY_ROWS")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class _ZeroResidualDistribution:
    def __init__(self, mean: Any) -> None:
        self.mean = mean


class _ZeroResidualPolicy:
    def distribution(self, observation: Any) -> _ZeroResidualDistribution:
        import torch

        return _ZeroResidualDistribution(
            torch.zeros(
                (observation.shape[0], 26),
                dtype=observation.dtype,
                device=observation.device,
            )
        )


class _ZeroResidualTrainer:
    def __init__(self) -> None:
        self.trainer = _ZeroResidualPolicy()


def _seeds_for(args: argparse.Namespace, *, independent: bool) -> list[int]:
    if not independent:
        return _seeds(args.clip, count=args.episodes)
    assert args.seed_manifest is not None
    payload = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    seeds = payload.get("eval10")
    if (
        payload.get("schema_version") != "IndependentPhysicalEvaluationSeedManifestV1"
        or payload.get("clip_id") != args.clip
        or not isinstance(seeds, list)
        or len(seeds) < args.episodes
    ):
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_SEED_MANIFEST_INVALID")
    return [int(value) for value in seeds[: args.episodes]]


def _contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    qualification = payload.get("zero_residual_eval", {}).get("qualification", {})
    required = {
        "minimum_pass_fraction",
        "wrist_command_to_actual_position_mean_m_max",
        "wrist_command_to_actual_rotation_mean_deg_max",
        "finger_command_to_actual_mean_rad_max",
        "minimum_contact_fraction",
        "reference_end_required",
    }
    if payload.get("schema_version") != "SourceControllerAutoContractV1" or not required.issubset(
        qualification
    ):
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_CONTRACT_INVALID")
    return payload, qualification


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.episodes != 10:
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_REQUIRES_EVAL10_AND_EULA")
    if args.checkpoint is None:
        if args.optimizer_steps != 0 or args.training_samples != 0:
            raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_COST_MUST_BE_ZERO")
    elif not args.checkpoint.is_file() or args.optimizer_steps <= 0 or args.training_samples <= 0:
        raise ValueError("CORRECTED_L0_SOURCE_CONTROLLER_COST_OR_CHECKPOINT_INVALID")
    independent_inputs = (
        args.reference,
        args.object_usd,
        args.support_proxy,
        args.support_asset,
        args.contact_contract,
        args.contact_mask_root,
        args.reference_distance_root,
        args.object_mesh_root,
        args.runtime_geometry_manifest,
        args.frozen_evaluation_gates,
        args.seed_manifest,
    )
    if any(value is not None for value in independent_inputs) and not all(
        value is not None for value in independent_inputs
    ):
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_INDEPENDENT_INPUT_SET_INCOMPLETE")
    independent = args.reference is not None
    contract_path = args.contract.resolve()
    contract, qualification = _contract(contract_path)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"ZERO_RESIDUAL_SOURCE_CONTROLLER_OUTPUT_EXISTS:{output}")
    output.mkdir(parents=True)
    seeds = _seeds_for(args, independent=independent)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
        from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

        start = 0 if independent else _full_start(args.clip)
        env = _make_table_env(
            clip=args.clip,
            num_envs=args.episodes,
            start_index=start,
            mode=ContactRewardMode.STRICT_PER_FINGER_V4,
            stage="C4",
            training_rsi=False,
            reward_aggregation_mode="grouped_multiplicative_v1",
            rse_enabled=True,
            full_horizon_evaluation=True,
            reference_path=args.reference,
            object_usd_path=args.object_usd,
            support_proxy_path=args.support_proxy,
            support_asset_path=args.support_asset,
            contact_contract_path=args.contact_contract,
            contact_mask_root=args.contact_mask_root,
            reference_distance_root=args.reference_distance_root,
            object_mesh_root=args.object_mesh_root,
            enforce_joint_position_limits=True,
            continuous_virtual_wrist_angles=True,
        )
        runtime = env.contract_report()
        wrist_contract = runtime["finite_virtual_6d_wrist_actuator"]
        mapping = runtime["joint_mapping"]
        if (
            wrist_contract["continuous_angle_branch"] is not True
            or wrist_contract["joint_position_limits_enforced"] is not False
            or mapping["joint_position_target_limits_enforced"] is not True
        ):
            raise RuntimeError("ZERO_RESIDUAL_SOURCE_CONTROLLER_LIMIT_CONTRACT_DRIFT")
        if args.checkpoint is None:
            trainer: Any = _ZeroResidualTrainer()
        else:
            from scripts.rl.isaaclab.evaluate_physical_hoi import model_from_checkpoint

            trainer, _ = model_from_checkpoint(
                args.checkpoint.resolve(), str(env.device), expected_clip=args.clip
            )
        rollouts = _parallel_rollouts(
            env=env,
            trainer=trainer,
            clip=args.clip,
            seeds=seeds,
            start=start,
        )
        gate = _load_gate(args.frozen_evaluation_gates or FROZEN_GATES, clip=args.clip)
        finger_lower = env.joint_lower.detach().cpu().numpy()
        finger_upper = env.joint_upper.detach().cpu().numpy()
        wrist_effort_indices = np.asarray(env._virtual_wrist_joint_ids, dtype=np.int64)
        finger_effort_indices = np.asarray(env._finger_target_joint_ids, dtype=np.int64)
        rows: list[dict[str, object]] = []
        receipts: list[dict[str, object]] = []
        for episode, (rollout, trace) in enumerate(rollouts):
            command = command_tracking_metrics(trace)
            trace["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
            trace["hand_collision_body_pose"] = _reconstruct_hand(trace)
            geometry, _ = _evaluate_geometry_with_exact_broadphase(
                clip=args.clip,
                object_pose=np.asarray(trace["object_pose"], dtype=np.float64)[:, None],
                hand_collision_body_pose=np.asarray(
                    trace["hand_collision_body_pose"], dtype=np.float64
                )[:, None],
                hand_collision_body_names=HAND_COLLISION_BODY_NAMES,
                geometry_path=args.runtime_geometry_manifest or GEOMETRY,
            )
            inter = _inter_finger_penetration(trace["hand_collision_body_pose"])
            action = np.asarray(trace["action"], dtype=np.float64)
            finger_q = np.asarray(trace["finger_q"], dtype=np.float64)
            effort = np.asarray(trace["actuator_effort"], dtype=np.float64)
            wrist_qdot = np.asarray(trace["virtual_wrist_qdot"], dtype=np.float64)
            finger_qdot = np.asarray(trace["finger_qdot"], dtype=np.float64)
            singularity_margin_deg = (
                serial_xyz_singularity_margin_deg(
                    torch.as_tensor(trace["virtual_wrist_q"], dtype=torch.float64)
                )
                .detach()
                .cpu()
                .numpy()
            )
            finite = bool(
                all(
                    np.isfinite(np.asarray(trace[name])).all()
                    for name in (
                        "wrist_pose",
                        "finger_q",
                        "object_pose",
                        "action",
                        "actuator_effort",
                    )
                )
            )
            joint_safe = bool(
                np.all(finger_q >= finger_lower[None] - 1.0e-6)
                and np.all(finger_q <= finger_upper[None] + 1.0e-6)
                and int(rollout["termination_reason"]) != 2
            )
            actuator_safe = bool(
                np.isfinite(effort).all()
                and np.max(np.abs(effort[:, wrist_effort_indices]), initial=0.0) <= 500.0 + 1.0e-3
                and np.max(np.abs(effort[:, finger_effort_indices]), initial=0.0) <= 0.6 + 1.0e-3
                and np.max(np.abs(wrist_qdot[:, :3]), initial=0.0) <= 2.0 + 1.0e-3
                and np.max(np.abs(wrist_qdot[:, 3:]), initial=0.0) <= 6.0 + 1.0e-3
                and np.max(np.abs(finger_qdot), initial=0.0) <= 12.0 + 1.0e-3
            )
            singularity_safe = bool(
                np.isfinite(singularity_margin_deg).all()
                and np.min(singularity_margin_deg, initial=180.0) > 5.0
            )
            action_safe = bool(np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0)
            wrist_position_mean = float(command["wrist_position_command_to_actual_m"]["mean"])
            wrist_rotation_mean_deg = math.degrees(
                float(command["wrist_rotation_command_to_actual_rad"]["mean"])
            )
            finger_mean = float(command["finger_command_to_actual_rad"]["mean"])
            tracking = bool(
                wrist_position_mean
                <= float(qualification["wrist_command_to_actual_position_mean_m_max"])
                and wrist_rotation_mean_deg
                <= float(qualification["wrist_command_to_actual_rotation_mean_deg_max"])
                and finger_mean <= float(qualification["finger_command_to_actual_mean_rad_max"])
            )
            contact = float(rollout["contact_fraction"]) >= float(
                qualification["minimum_contact_fraction"]
            )
            progression = bool(rollout["reached_reference_end"])
            collision = bool(
                float(geometry["max_penetration_m"]) < float(gate["catastrophic_penetration_m"])
                and float(geometry["p95_penetration_m"]) <= float(gate["p95_penetration_m"])
                and float(inter.max(initial=0.0))
                <= float(gate["maximum_inter_finger_penetration_m"])
            )
            qualified = bool(
                tracking
                and contact
                and progression
                and finite
                and joint_safe
                and actuator_safe
                and action_safe
                and singularity_safe
                and collision
            )
            trace_path = output / "traces" / f"episode_{episode:02d}.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, **trace)
            row = {
                "clip_id": args.clip,
                "episode": episode,
                "seed": seeds[episode],
                "optimizer_steps": args.optimizer_steps,
                "training_samples": args.training_samples,
                "action_max_abs": float(np.max(np.abs(action))),
                "wrist_command_actual_position_mean_m": wrist_position_mean,
                "wrist_command_actual_rotation_mean_deg": wrist_rotation_mean_deg,
                "finger_command_actual_mean_rad": finger_mean,
                "contact_fraction": float(rollout["contact_fraction"]),
                "reference_end": progression,
                "finite_safe": finite,
                "joint_limits_safe": joint_safe,
                "actuator_limits_safe": actuator_safe,
                "singularity_safe": singularity_safe,
                "minimum_singularity_margin_deg": float(
                    np.min(singularity_margin_deg, initial=180.0)
                ),
                "action_bounds_safe": action_safe,
                "collision_safety_pass": collision,
                "qualified": qualified,
                "trace": str(trace_path),
                "trace_sha256": _sha256(trace_path),
            }
            rows.append(row)
            receipt = {
                **row,
                "schema_version": "ZeroResidualSourceControllerEpisodeReceiptV1",
                "reference_tracking_pass": tracking,
                "contact_execution_pass": contact,
                "reference_progression_pass": progression,
                "controller_authority_pass": tracking,
            }
            receipts.append(receipt)
            _write_json(output / "per_episode" / f"episode_{episode:02d}.json", receipt)
        passed = sum(bool(row["qualified"]) for row in rows)
        minimum = math.ceil(float(qualification["minimum_pass_fraction"]) * args.episodes)
        summary = {
            "schema_version": "ZeroResidualSourceControllerQualificationV1",
            "status": "PASS" if passed >= minimum else "FAIL",
            "clip_id": args.clip,
            "mode": "ZERO_RESIDUAL" if args.checkpoint is None else "CORRECTED_L0",
            "optimizer_steps": args.optimizer_steps,
            "training_samples": args.training_samples,
            "checkpoint": (
                None
                if args.checkpoint is None
                else {
                    "path": str(args.checkpoint.resolve()),
                    "sha256": _sha256(args.checkpoint.resolve()),
                }
            ),
            "episodes": args.episodes,
            "qualified_episodes": passed,
            "qualification_required": minimum,
            "source_policy_ready_no_l0": passed >= minimum,
            "continuous_virtual_wrist_angles": True,
            "real_finger_joint_limits_enforced": True,
            "safety_contract": SourceControllerSafetyContractV1().as_dict(),
            "contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "runtime_contract": runtime,
            "seed_manifest": seeds,
            "per_episode_receipts": receipts,
        }
        _write_csv(output / "per_episode.csv", rows)
        _write_json(output / "qualification.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        active_error = sys.exc_info()[0]
        if active_error is None and env is not None:
            env.close()
        app.close(skip_cleanup=active_error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
