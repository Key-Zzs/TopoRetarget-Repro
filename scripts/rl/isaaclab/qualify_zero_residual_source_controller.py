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
from toporetarget.rl.source_controller import (
    SourceControllerExecutableContractV2,
    source_controller_executability_v2,
    source_controller_fidelity_v2,
)

DEFAULT_CONTRACT = REPO_ROOT / "configs/contracts/source_controller_auto_v2.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--controller-mode",
        choices=("ZERO_RESIDUAL_DETERMINISTIC", "ZERO_RESIDUAL_NETWORK", "CORRECTED_L0"),
    )
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


def _quaternion_error_deg(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    actual_norm = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    target_norm = target / np.linalg.norm(target, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(actual_norm * target_norm, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


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


def _contract(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    admission = payload.get("admission", {})
    fidelity = payload.get("fidelity", {})
    fidelity_required = {
        "minimum_pass_fraction",
    }
    fidelity_thresholds = {
        "wrist_command_to_actual_position_mean_m_max",
        "wrist_command_to_actual_rotation_mean_deg_max",
        "finger_command_to_actual_mean_rad_max",
        "link_reference_error_mean_m_max",
        "source_contact_recall_min",
        "object_position_error_mean_m_max",
        "object_rotation_error_mean_deg_max",
        "command_clamp_fraction_max",
        "actuator_saturation_fraction_max",
    }
    if (
        payload.get("schema_version") != "SourceControllerAutoContractV2"
        or admission.get("schema_version") != "SourceControllerExecutableV2"
        or not fidelity_required.issubset(admission)
        or not fidelity_thresholds.issubset(fidelity)
    ):
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_CONTRACT_INVALID")
    expected = set(SourceControllerExecutableContractV2().as_dict()) - {
        "virtual_wrist_angle_authority"
    }
    if set(admission.get("hard_requirements", ())) != expected:
        raise ValueError("SOURCE_CONTROLLER_EXECUTABLE_V2_REQUIREMENTS_DRIFT")
    return payload, admission, fidelity


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.episodes != 10:
        raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_REQUIRES_EVAL10_AND_EULA")
    controller_mode = args.controller_mode or (
        "ZERO_RESIDUAL_DETERMINISTIC" if args.checkpoint is None else "CORRECTED_L0"
    )
    if controller_mode == "ZERO_RESIDUAL_DETERMINISTIC":
        if (
            args.checkpoint is not None
            or args.optimizer_steps != 0
            or args.training_samples != 0
        ):
            raise ValueError("ZERO_RESIDUAL_SOURCE_CONTROLLER_COST_MUST_BE_ZERO")
    elif controller_mode == "ZERO_RESIDUAL_NETWORK":
        if (
            args.checkpoint is None
            or not args.checkpoint.is_file()
            or args.optimizer_steps != 0
            or args.training_samples != 0
        ):
            raise ValueError("ZERO_RESIDUAL_NETWORK_COST_OR_CHECKPOINT_INVALID")
    elif (
        args.checkpoint is None
        or not args.checkpoint.is_file()
        or args.optimizer_steps <= 0
        or args.training_samples <= 0
    ):
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
    contract, admission, fidelity_thresholds = _contract(contract_path)
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
            continuous_virtual_wrist_angles=True,
            source_controller_admission_v2=True,
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

            trainer, checkpoint_payload = model_from_checkpoint(
                args.checkpoint.resolve(), str(env.device), expected_clip=args.clip
            )
            if controller_mode == "ZERO_RESIDUAL_NETWORK":
                actor_state = {
                    name: value
                    for name, value in checkpoint_payload["actor_critic"].items()
                    if name.startswith("actor.")
                }
                if (
                    checkpoint_payload.get("source_controller_route") != "ZERO_RESIDUAL"
                    or not actor_state
                    or not all(
                        bool(torch.count_nonzero(value) == 0)
                        for value in actor_state.values()
                    )
                ):
                    raise ValueError("ZERO_RESIDUAL_NETWORK_CHECKPOINT_NOT_IDENTICALLY_ZERO")
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
            wrist_q = np.asarray(trace["virtual_wrist_q"], dtype=np.float64)
            wrist_qdot = np.asarray(trace["virtual_wrist_qdot"], dtype=np.float64)
            finger_qdot = np.asarray(trace["finger_qdot"], dtype=np.float64)
            reference_index = np.asarray(trace["reference_index"], dtype=np.int64)
            singularity_margin_deg = (
                serial_xyz_singularity_margin_deg(
                    torch.as_tensor(wrist_q, dtype=torch.float64)
                )
                .detach()
                .cpu()
                .numpy()
            )
            state_fields = (
                "wrist_pose",
                "wrist_twist_world",
                "finger_q",
                "finger_qdot",
                "virtual_wrist_q",
                "virtual_wrist_qdot",
                "object_pose",
                "object_twist",
            )
            target_fields = (
                "wrist_reference",
                "finger_reference",
                "object_reference",
                "tracked_link_reference",
            )
            command_fields = ("wrist_target", "finger_target", "virtual_wrist_target_q")
            required_fields = state_fields + target_fields + command_fields + ("action",)
            expected_rows = int(rollout["steps"]) + 1
            rows_readable = bool(
                all(name in trace for name in required_fields)
                and all(len(np.asarray(trace[name])) == expected_rows for name in required_fields)
                and len(reference_index) == expected_rows
            )
            state_finite = bool(
                rows_readable
                and all(np.isfinite(np.asarray(trace[name])).all() for name in state_fields)
            )
            target_finite = bool(
                rows_readable
                and all(np.isfinite(np.asarray(trace[name])).all() for name in target_fields)
            )
            command_finite = bool(
                rows_readable
                and all(np.isfinite(np.asarray(trace[name])).all() for name in command_fields)
            )
            action_finite = bool(np.isfinite(action).all())
            reference_advances = bool(
                len(reference_index) > 1
                and np.all(np.diff(reference_index) >= 0)
                and int(reference_index[-1]) > int(reference_index[0])
            )
            controller_fresh = bool(
                rows_readable
                and reference_advances
                and np.array_equal(
                    reference_index,
                    np.arange(reference_index[0], reference_index[0] + len(reference_index)),
                )
            )
            finger_joint_safe = bool(
                np.all(finger_q >= finger_lower[None] - 1.0e-6)
                and np.all(finger_q <= finger_upper[None] + 1.0e-6)
            )
            wrist_translation_safe = bool(
                np.isfinite(wrist_q[:, :3]).all()
                and np.all(wrist_q[:, :3] >= -0.4 - 1.0e-6)
                and np.all(wrist_q[:, :3] <= 0.4 + 1.0e-6)
            )
            effort_safe = bool(
                np.isfinite(effort).all()
                and np.max(np.abs(effort[:, wrist_effort_indices]), initial=0.0) <= 500.0 + 1.0e-3
                and np.max(np.abs(effort[:, finger_effort_indices]), initial=0.0) <= 0.6 + 1.0e-3
            )
            velocity_safe = bool(
                np.isfinite(wrist_qdot).all()
                and np.isfinite(finger_qdot).all()
                and np.max(np.abs(wrist_qdot[:, :3]), initial=0.0) <= 2.0 + 1.0e-3
                and np.max(np.abs(wrist_qdot[:, 3:]), initial=0.0) <= 6.0 + 1.0e-3
                and np.max(np.abs(finger_qdot), initial=0.0) <= 12.0 + 1.0e-3
            )
            singularity_safe = bool(
                np.isfinite(singularity_margin_deg).all()
                and np.min(singularity_margin_deg, initial=180.0) > 5.0
            )
            action_safe = bool(action_finite and np.max(np.abs(action), initial=0.0) <= 1.0)
            wrist_position_mean = float(command["wrist_position_command_to_actual_m"]["mean"])
            wrist_rotation_mean_deg = math.degrees(
                float(command["wrist_rotation_command_to_actual_rad"]["mean"])
            )
            finger_mean = float(command["finger_command_to_actual_rad"]["mean"])
            tracked_link_error_mean = float(
                np.linalg.norm(
                    np.asarray(trace["tracked_link_positions"], dtype=np.float64)
                    - np.asarray(trace["tracked_link_reference"], dtype=np.float64),
                    axis=-1,
                ).mean()
            )
            source_contact = np.asarray(trace["source_contact_mask"], dtype=bool)
            actual_contact = np.asarray(trace["actual_contact_mask"], dtype=bool)
            source_contact_count = int(source_contact.sum())
            source_contact_recall = float(
                1.0
                if source_contact_count == 0
                else np.count_nonzero(source_contact & actual_contact) / source_contact_count
            )
            object_actual = np.asarray(trace["object_pose"], dtype=np.float64)
            object_reference = np.asarray(trace["object_reference"], dtype=np.float64)
            object_position_error_mean = float(
                np.linalg.norm(object_actual[:, :3] - object_reference[:, :3], axis=-1).mean()
            )
            object_rotation_error_mean_deg = float(
                _quaternion_error_deg(object_actual[:, 3:7], object_reference[:, 3:7]).mean()
            )
            finger_fraction = float(runtime["action"]["finger_joint_range_fraction"])
            pre_finger_target = (
                np.asarray(trace["finger_reference"], dtype=np.float64)
                + action[:, 6:] * (finger_upper - finger_lower)[None] * finger_fraction
            )
            post_finger_target = np.asarray(trace["finger_target"], dtype=np.float64)
            command_clamp_fraction = float(
                np.mean(np.abs(pre_finger_target - post_finger_target) > 1.0e-7)
            )
            wrist_saturation = np.abs(effort[:, wrist_effort_indices]) >= 500.0 * 0.999
            finger_saturation = np.abs(effort[:, finger_effort_indices]) >= 0.6 * 0.999
            actuator_saturation_fraction = float(
                np.mean(np.concatenate((wrist_saturation, finger_saturation), axis=1))
            )
            reference_completion = bool(rollout["reached_reference_end"])
            interaction_progression = bool(
                reference_advances
                and (int(reference_index[-1]) - int(reference_index[0]))
                >= max(1, expected_rows // 2)
            )
            catastrophic_collision_safe = bool(
                float(geometry["max_penetration_m"]) < float(gate["catastrophic_penetration_m"])
                and float(inter.max(initial=0.0))
                <= float(gate["maximum_inter_finger_penetration_m"])
            )
            hard_reason = int(rollout["termination_reason"]) in (1, 5, 6)
            executable_receipt = {
                "state_finite": state_finite,
                "target_finite": target_finite,
                "command_finite": command_finite,
                "action_finite": action_finite,
                "reference_index_advances": reference_advances,
                "trajectory_rows_readable": rows_readable,
                "controller_state_fresh": controller_fresh,
                "real_finger_joint_limits_safe": finger_joint_safe,
                "virtual_wrist_translation_limits_safe": wrist_translation_safe,
                "actuator_effort_limits_safe": effort_safe,
                "actuator_velocity_limits_safe": velocity_safe,
                "action_bounds_safe": action_safe,
                "singularity_safety_pass": singularity_safe,
                "catastrophic_collision_safe": catastrophic_collision_safe,
                "nonfinite_dynamics_absent": state_finite,
                "controller_divergence_absent": not hard_reason,
            }
            fidelity_receipt = {
                "wrist_position_tracking_pass": wrist_position_mean
                <= float(
                    fidelity_thresholds["wrist_command_to_actual_position_mean_m_max"]
                ),
                "wrist_rotation_tracking_pass": wrist_rotation_mean_deg
                <= float(
                    fidelity_thresholds["wrist_command_to_actual_rotation_mean_deg_max"]
                ),
                "finger_tracking_pass": finger_mean
                <= float(fidelity_thresholds["finger_command_to_actual_mean_rad_max"]),
                "link_tracking_pass": tracked_link_error_mean
                <= float(fidelity_thresholds["link_reference_error_mean_m_max"]),
                "source_contact_recall_pass": source_contact_recall
                >= float(fidelity_thresholds["source_contact_recall_min"]),
                "object_tracking_pass": object_position_error_mean
                <= float(fidelity_thresholds["object_position_error_mean_m_max"])
                and object_rotation_error_mean_deg
                <= float(fidelity_thresholds["object_rotation_error_mean_deg_max"]),
                "interaction_progression_pass": interaction_progression,
                "command_clamp_pass": command_clamp_fraction
                <= float(fidelity_thresholds["command_clamp_fraction_max"]),
                "actuator_saturation_pass": actuator_saturation_fraction
                <= float(fidelity_thresholds["actuator_saturation_fraction_max"]),
                "reference_completion_pass": reference_completion,
            }
            combined_receipt = {**executable_receipt, **fidelity_receipt}
            executability = source_controller_executability_v2(combined_receipt)
            fidelity = source_controller_fidelity_v2(combined_receipt)
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
                "link_reference_error_mean_m": tracked_link_error_mean,
                "source_contact_recall": source_contact_recall,
                "object_position_error_mean_m": object_position_error_mean,
                "object_rotation_error_mean_deg": object_rotation_error_mean_deg,
                "command_clamp_fraction": command_clamp_fraction,
                "actuator_saturation_fraction": actuator_saturation_fraction,
                "contact_fraction": float(rollout["contact_fraction"]),
                "reference_end": reference_completion,
                "finite_safe": state_finite and target_finite and command_finite and action_finite,
                "joint_limits_safe": finger_joint_safe and wrist_translation_safe,
                "actuator_limits_safe": effort_safe and velocity_safe,
                "singularity_safe": singularity_safe,
                "minimum_singularity_margin_deg": float(
                    np.min(singularity_margin_deg, initial=180.0)
                ),
                "action_bounds_safe": action_safe,
                "collision_safety_pass": catastrophic_collision_safe,
                "executability_v2": executability.value,
                "fidelity_v2": fidelity.value,
                "qualified": executability.value == "PASS",
                "trace": str(trace_path),
                "trace_sha256": _sha256(trace_path),
            }
            rows.append(row)
            receipt = {
                **row,
                **combined_receipt,
                "schema_version": "SourceControllerEpisodeReceiptV2",
                "reference_tracking_pass": bool(
                    fidelity_receipt["wrist_position_tracking_pass"]
                    and fidelity_receipt["wrist_rotation_tracking_pass"]
                    and fidelity_receipt["finger_tracking_pass"]
                ),
                "contact_execution_pass": fidelity_receipt["source_contact_recall_pass"],
                "reference_progression_pass": fidelity_receipt[
                    "interaction_progression_pass"
                ],
                "controller_authority_pass": executability.value == "PASS",
                "normalized_wrist_tracking_error": (
                    wrist_position_mean
                    / float(
                        fidelity_thresholds[
                            "wrist_command_to_actual_position_mean_m_max"
                        ]
                    )
                    + wrist_rotation_mean_deg
                    / float(
                        fidelity_thresholds[
                            "wrist_command_to_actual_rotation_mean_deg_max"
                        ]
                    )
                ),
                "normalized_finger_tracking_error": finger_mean
                / float(fidelity_thresholds["finger_command_to_actual_mean_rad_max"]),
                "object_tracking_score": 1.0
                / (
                    1.0
                    + object_position_error_mean
                    / float(
                        fidelity_thresholds["object_position_error_mean_m_max"]
                    )
                    + object_rotation_error_mean_deg
                    / float(
                        fidelity_thresholds["object_rotation_error_mean_deg_max"]
                    )
                ),
            }
            receipts.append(receipt)
            _write_json(output / "per_episode" / f"episode_{episode:02d}.json", receipt)
        passed = sum(row["executability_v2"] == "PASS" for row in rows)
        minimum = math.ceil(float(admission["minimum_pass_fraction"]) * args.episodes)
        summary = {
            "schema_version": "SourceControllerQualificationV2",
            "status": "PASS" if passed >= minimum else "FAIL",
            "clip_id": args.clip,
            "mode": controller_mode,
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
            "source_policy_ready_no_l0": controller_mode.startswith("ZERO_RESIDUAL")
            and passed >= minimum,
            "source_controller_executability_v2": (
                "PASS" if passed >= minimum else "FAIL"
            ),
            "source_controller_fidelity_v2": (
                "PASS"
                if all(row["fidelity_v2"] == "PASS" for row in rows)
                else (
                    "FAIL"
                    if all(row["fidelity_v2"] == "FAIL" for row in rows)
                    else "DEGRADED"
                )
            ),
            "continuous_virtual_wrist_angles": True,
            "real_finger_joint_limits_enforced": True,
            "executable_contract": SourceControllerExecutableContractV2().as_dict(),
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
