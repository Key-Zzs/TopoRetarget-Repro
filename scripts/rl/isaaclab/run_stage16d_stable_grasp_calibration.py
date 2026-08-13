#!/usr/bin/env python3
"""Run one bounded object-canonical free-object stable-grasp calibration candidate."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.calibration_initialization import (  # noqa: E402
    RESET_MAXIMUM_PAIR_PENETRATION_M,
    ObjectCanonicalFrameV1,
    apply_bounded_pose_delta,
    initialize_object_between_contacts,
    refine_balanced_contact_pose,
)
from toporetarget.rl.geometry_audit.stable_grasp_calibration import (  # noqa: E402
    CALIBRATION_GROUP_ORDER,
    StableGraspCalibrationActionScheduleV1,
    StableGraspCalibrationGateV1,
    qualify_stable_grasp,
)
from toporetarget.rl.geometry_audit.transforms import (  # noqa: E402
    compose_poses,
    quaternion_matrix_wxyz,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"
MANIFEST_PATH = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)
OBJECT_IDS = ("hocap_170105", "hocap_170650")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", choices=OBJECT_IDS, required=True)
    parser.add_argument("--level", choices=("C1", "C2"), default="C1")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--replicas", type=int, choices=(4, 20), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--raw-geometry", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_report(path: Path, result: dict[str, Any]) -> None:
    """Atomically persist recoverable evidence before Isaac shutdown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate(args: argparse.Namespace) -> dict[str, Any]:
    matrix = _load(args.contract_root / f"stable_grasp_candidate_matrix_{args.level.lower()}.json")
    rows = matrix["objects"][args.object]
    matches = [row for row in rows if row["candidate_id"] == args.candidate_id]
    if len(matches) != 1:
        raise ValueError(f"candidate is not uniquely frozen: {args.candidate_id}")
    return matches[0]


def _close_env(env: Any) -> None:
    if env is not None:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def _freeze_calibration_reference(env: Any, *, clip_index: int) -> None:
    """Freeze a pre-grasp hand reference without using any object trajectory pose."""

    import torch

    from toporetarget.rl.environments.isaaclab_backend.explicit_wrist_reference import (
        ExplicitWristJointReferenceV2,
    )

    bank = env.reference_bank
    lower = env.joint_lower
    upper = env.joint_upper
    pregrasp = lower + 0.65 * (upper - lower)
    for field in bank.REQUIRED_FIELDS:
        if field == "timestamps":
            continue
        values = getattr(bank, field)
        if field == "q_finger_ref":
            values[clip_index] = pregrasp.expand_as(values[clip_index])
        elif field in {"qdot_finger_ref", "wrist_twist_world_ref", "object_twist_world_ref"}:
            values[clip_index].zero_()
        else:
            frozen = values[clip_index, 0].clone()
            values[clip_index] = frozen.expand_as(values[clip_index])
    env._explicit_joint_reference = ExplicitWristJointReferenceV2.from_reference_bank(bank)
    if not bool(torch.isfinite(env._explicit_joint_reference.q_wrist_ref).all()):
        raise RuntimeError("STAGE16D_CALIBRATION_REFERENCE_NONFINITE")


def _settle_calibration_hand(
    env: Any,
    *,
    clip_index: int,
    open_action: np.ndarray,
    minimum_steps: int = 24,
    maximum_steps: int = 240,
) -> dict[str, Any]:
    """Settle the reset hand while the active object is parked out of reach."""

    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import (
        raw_control_step,
        synchronize_reset_boundary,
    )

    ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    parking_position = env.scene.env_origins + torch.tensor(
        (8.0, 5.0, -5.0), dtype=torch.float32, device=env.device
    )
    parking_state = torch.cat(
        (
            parking_position,
            torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device).expand(env.num_envs, -1),
            torch.zeros((env.num_envs, 6), device=env.device),
        ),
        dim=-1,
    )
    obj = env._object_170105 if clip_index == 0 else env._object_170650
    obj.write_root_state_to_sim(parking_state, env_ids=ids)
    synchronize_reset_boundary(env)
    held_open_action = torch.as_tensor(open_action, dtype=torch.float32, device=env.device).expand(
        env.num_envs, -1
    )
    stable_intervals = 0
    completed_steps = 0
    final_max_joint_speed = float("inf")
    for step in range(maximum_steps):
        raw_control_step(env, held_open_action)
        completed_steps = step + 1
        final_max_joint_speed = float(
            env._robot.data.joint_vel[:, env._finger_target_joint_ids].abs().amax().item()
        )
        if completed_steps >= minimum_steps and final_max_joint_speed <= 1.0e-3:
            stable_intervals += 1
        else:
            stable_intervals = 0
        if stable_intervals >= 12:
            break
    # These are framework counters, not simulator state.  The 321-step
    # qualification trace starts only after the object-canonical reset below.
    env.episode_length_buf.zero_()
    env.reset_buf.zero_()
    env.reset_terminated.zero_()
    env.reset_time_outs.zero_()
    env._contact_substep_records.clear()
    env._contact_substep_record_total = 0
    return {
        "schema_version": "CalibrationHandSettleV1",
        "object_parked_out_of_reach": True,
        "settled_at_first_schedule_action": True,
        "completed_control_intervals": completed_steps,
        "required_consecutive_stable_intervals": 12,
        "joint_speed_threshold_radps": 1.0e-3,
        "final_max_finger_joint_speed_radps": final_max_joint_speed,
        "settled": stable_intervals >= 12,
    }


def _group_collision_geometry(
    env: Any,
    *,
    group: str,
    env_id: int,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )

    slots = env._group_slots[group]
    distal = [slot for slot in slots if "distal" in HAND_COLLISION_BODY_NAMES[slot]]
    slot = distal[-1] if distal else slots[-1]
    body_name = HAND_COLLISION_BODY_NAMES[slot]
    rows = [row for row in manifest["hand_shapes"] if row["body_name"] == body_name]
    if len(rows) != 1:
        raise RuntimeError(f"STAGE16D_CALIBRATION_HAND_PROXY_RESOLUTION_FAILURE:{body_name}")
    row = rows[0]
    vertices = np.asarray(row["convex_vertices_m"], dtype=np.float64) * np.asarray(
        row["scale_xyz"], dtype=np.float64
    )
    shape_rotation = quaternion_matrix_wxyz(
        np.asarray(row["local_transform"]["rotation_wxyz"], dtype=np.float64)
    )
    shape_translation = np.asarray(row["local_transform"]["translation_xyz_m"], dtype=np.float64)
    body_vertices = vertices @ shape_rotation.T + shape_translation
    body_id = env._robot.body_names.index(body_name)
    position = (
        (env._robot.data.body_link_pos_w[env_id, body_id] - env.scene.env_origins[env_id])
        .detach()
        .cpu()
        .numpy()
    )
    quaternion = env._robot.data.body_link_quat_w[env_id, body_id].detach().cpu().numpy()
    rotation = quaternion_matrix_wxyz(quaternion)
    world_vertices = body_vertices @ rotation.T + position
    return world_vertices.mean(axis=0), world_vertices


def _initialize_candidate(
    env: Any,
    *,
    clip_index: int,
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    initialization_contract: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )
    from toporetarget.rl.isaaclab_oracle.history_replay import synchronize_reset_boundary

    first, first_vertices = _group_collision_geometry(
        env,
        group=candidate["first_opposition_group"],
        env_id=0,
        manifest=manifest,
    )
    second, second_vertices = _group_collision_geometry(
        env,
        group=candidate["second_opposition_group"],
        env_id=0,
        manifest=manifest,
    )
    wrist_body = env._robot.body_names.index(HAND_COLLISION_BODY_NAMES[0])
    wrist_position = (
        (env._robot.data.body_link_pos_w[0, wrist_body] - env.scene.env_origins[0])
        .detach()
        .cpu()
        .numpy()
    )
    wrist_quaternion = env._robot.data.body_link_quat_w[0, wrist_body].detach().cpu().numpy()
    wrist_rotation = quaternion_matrix_wxyz(wrist_quaternion)
    frame_row = initialization_contract["object_frames"][candidate["object_id"]]["canonical_frame"]
    initialization = initialize_object_between_contacts(
        frame=ObjectCanonicalFrameV1(**frame_row),
        first_contact_center_scene=first,
        second_contact_center_scene=second,
        first_contact_vertices_scene=first_vertices,
        second_contact_vertices_scene=second_vertices,
        palm_center_scene=wrist_position,
        palm_rotation_scene=wrist_rotation,
        approach_offset_m=float(candidate["approach_offset_m"]),
    )
    from toporetarget.rl.geometry_audit.convex_query import PythonFCLConvexQueryBackend
    from toporetarget.rl.geometry_audit.runtime_geometry import load_runtime_geometry_manifest

    hand_proxies, object_proxies = load_runtime_geometry_manifest(MANIFEST_PATH)
    if tuple(proxy.body_name for proxy in hand_proxies) != tuple(HAND_COLLISION_BODY_NAMES):
        raise RuntimeError("STAGE16D_CALIBRATION_HAND_BODY_ORDER_DRIFT")
    object_rows = object_proxies[candidate["object_id"]]
    if len(object_rows) != 1:
        raise RuntimeError("STAGE16D_CALIBRATION_OBJECT_PROXY_COUNT_DRIFT")
    backend = PythonFCLConvexQueryBackend()
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_proxy = object_rows[0]
    object_shape = backend.proxy_shape(object_proxy)
    hand_body_ids = [env._robot.body_names.index(name) for name in HAND_COLLISION_BODY_NAMES]
    hand_position = (
        (env._robot.data.body_link_pos_w[0, hand_body_ids] - env.scene.env_origins[0])
        .detach()
        .cpu()
        .numpy()
    )
    hand_quaternion = env._robot.data.body_link_quat_w[0, hand_body_ids].detach().cpu().numpy()
    hand_root_poses = np.concatenate((hand_position, hand_quaternion), axis=-1)
    hand_world_poses = [
        compose_poses(root_pose, proxy.local_pose_xyz_wxyz)
        for root_pose, proxy in zip(hand_root_poses, hand_proxies, strict=True)
    ]

    def query_pose(object_root_pose: np.ndarray) -> list[Any]:
        object_world = compose_poses(object_root_pose, object_proxy.local_pose_xyz_wxyz)
        return [
            backend.query(hand_shape, hand_world, object_shape, object_world)
            for hand_shape, hand_world in zip(hand_shapes, hand_world_poses, strict=True)
        ]

    canonical_pose: np.ndarray = np.asarray(
        initialization.object_pose_scene_xyz_wxyz, dtype=np.float64
    )
    fine_orientation_grid_deg = (-10.0, -5.0, 0.0, 5.0, 10.0)
    coarse_orientation_grid_deg = (-90.0, 0.0, 90.0, 180.0)
    orientation_seeds_deg = sorted(
        set(itertools.product(fine_orientation_grid_deg, repeat=3))
        | set(itertools.product(coarse_orientation_grid_deg, repeat=3))
    )
    refinement_rows: list[tuple[tuple[Any, ...], tuple[float, float, float], Any]] = []
    for raw_rotation_deg in orientation_seeds_deg:
        rotation_deg = (
            float(raw_rotation_deg[0]),
            float(raw_rotation_deg[1]),
            float(raw_rotation_deg[2]),
        )
        seed_delta = np.asarray([0.0, 0.0, 0.0, *(np.radians(rotation_deg))], dtype=np.float64)
        seed_pose = apply_bounded_pose_delta(canonical_pose, seed_delta)
        row = refine_balanced_contact_pose(
            initial_pose_scene_xyz_wxyz=seed_pose,
            selected_groups=(
                str(candidate["first_opposition_group"]),
                str(candidate["second_opposition_group"]),
            ),
            group_slots=env._group_slots,
            query_pose=query_pose,
            maximum_iterations=8,
        )
        signed_error = max(
            abs(value - row.target_signed_separation_m)
            for value in row.selected_signed_separation_after_m
        )
        static_contact_pass = bool(
            row.safe_reset
            and row.maximum_pair_penetration_after_m <= RESET_MAXIMUM_PAIR_PENETRATION_M
        )
        maximum_selected_gap = max(row.selected_signed_separation_after_m)
        key: tuple[Any, ...] = (
            not static_contact_pass,
            maximum_selected_gap if static_contact_pass else row.maximum_pair_penetration_after_m,
            row.selected_direction_balance_norm,
            signed_error,
            row.maximum_pair_penetration_after_m,
            rotation_deg,
        )
        refinement_rows.append((key, rotation_deg, row))
    refinement_rows.sort(key=lambda value: value[0])
    refinement_key, refinement_rotation_deg, refinement = refinement_rows[0]
    refined_pose: np.ndarray = np.asarray(refinement.refined_pose_scene_xyz_wxyz, dtype=np.float64)
    object_centroid = refined_pose[:3]
    direction_scene = object_centroid - wrist_position
    direction_local: np.ndarray = wrist_rotation.T @ direction_scene
    report = initialization.as_dict()
    refinement_report = refinement.as_dict()
    refinement_report.update(
        {
            "fine_orientation_grid_deg": list(fine_orientation_grid_deg),
            "coarse_orientation_grid_deg": list(coarse_orientation_grid_deg),
            "orientation_grid_evaluations": len(refinement_rows),
            "selected_orientation_seed_deg": list(refinement_rotation_deg),
            "static_contact_pass": not bool(refinement_key[0]),
            "static_reset_safety_pass": not bool(refinement_key[0]),
            "target_clearance_converged": refinement.converged,
            "static_contact_pass_requires_solver_convergence": False,
            "dynamic_required_contact_groups": list(candidate["contact_groups"]),
            "selection_uses_dynamic_rollout_result": False,
            "object_pose_written_at_reset": False,
        }
    )
    report["exact_balanced_contact_refinement"] = refinement_report
    if refinement_report["static_contact_pass"]:
        ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        pose = torch.as_tensor(
            refined_pose,
            dtype=torch.float32,
            device=env.device,
        ).expand(env.num_envs, -1)
        root_state = torch.cat(
            (
                pose[:, :3] + env.scene.env_origins,
                pose[:, 3:],
                torch.zeros((env.num_envs, 6), device=env.device),
            ),
            dim=-1,
        )
        obj = env._object_170105 if clip_index == 0 else env._object_170650
        obj.write_root_state_to_sim(root_state, env_ids=ids)
        synchronize_reset_boundary(env)
        refinement_report["object_pose_written_at_reset"] = True
    return report, direction_local


def _contact_presence(env: Any, *, clip_index: int) -> tuple[Any, Any]:
    import torch

    first = env._object_contact_sensors["Object170105"].data.force_matrix_w
    second = env._object_contact_sensors["Object170650"].data.force_matrix_w
    if first is None or second is None:
        raise RuntimeError("STAGE16D_CALIBRATION_CONTACT_FORCE_MATRIX_UNAVAILABLE")
    forces = first[:, 0] if clip_index == 0 else second[:, 0]
    pair_presence = torch.linalg.vector_norm(forces, dim=-1) > 1.0e-4
    grouped = torch.stack(
        tuple(
            pair_presence[:, env._group_slots[group]].any(dim=-1)
            if env._group_slots.get(group)
            else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for group in CALIBRATION_GROUP_ORDER
        ),
        dim=-1,
    )
    return forces, grouped


def _interval_contact_presence(env: Any) -> tuple[Any, Any]:
    """Aggregate real filtered contacts over all substeps in one control interval."""

    import torch

    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )

    pair_presence = torch.zeros(
        (env.num_envs, len(HAND_COLLISION_BODY_NAMES)),
        dtype=torch.bool,
        device=env.device,
    )
    expected = env.num_envs * env.cfg.decimation
    records = list(env._contact_substep_records)[-expected:]
    for record in records:
        env_id = int(record["env_id"])
        for body_name in record["present_hand_body_names"]:
            pair_presence[env_id, HAND_COLLISION_BODY_NAMES.index(str(body_name))] = True
    grouped = torch.stack(
        tuple(
            pair_presence[:, env._group_slots[group]].any(dim=-1)
            if env._group_slots.get(group)
            else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for group in CALIBRATION_GROUP_ORDER
        ),
        dim=-1,
    )
    return pair_presence, grouped


def _run_physx(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    import torch
    from optimize_stage16d_physics_trajectory import make_env

    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )
    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    env = None
    started = time.perf_counter()
    clip_index = OBJECT_IDS.index(args.object)
    manifest = _load(MANIFEST_PATH)
    initialization_contract = _load(args.contract_root / "calibration_initialization_contract.json")
    schedule = StableGraspCalibrationActionScheduleV1()
    try:
        env = make_env(num_envs=args.replicas, clip=args.object, telemetry="aggregate")
        _freeze_calibration_reference(env, clip_index=clip_index)
        reset_frozen_clip_frame_zero(env, clip_index=clip_index)
        open_action = schedule.actions(
            contact_groups=candidate["contact_groups"],
            closure_amplitude=float(candidate["closure_amplitude"]),
            wrist_approach_direction_local=np.zeros(3, dtype=np.float64),
        )[0]
        hand_settle = _settle_calibration_hand(env, clip_index=clip_index, open_action=open_action)
        initialization, approach_direction_local = _initialize_candidate(
            env,
            clip_index=clip_index,
            candidate=candidate,
            manifest=manifest,
            initialization_contract=initialization_contract,
        )
        initialization["hand_settle"] = hand_settle
        initialization_static_pass = bool(
            initialization["exact_balanced_contact_refinement"]["static_contact_pass"]
        )
        initialization["rollout_boundary"] = {
            "schema_version": "StableGraspCalibrationRolloutBoundaryV1",
            "free_object_control_intervals_before_schedule": 0,
            "first_free_object_control_interval_is_schedule_step": 0,
            "schedule_control_intervals": 321,
            "object_state_writes_after_schedule_start": 0,
            "wrist_state_writes_after_schedule_start": 0,
        }
        if not initialization_static_pass:
            runtime = {
                "schema_version": "StableFreeObjectGraspCalibrationRunV1",
                "object_id": args.object,
                "candidate": candidate,
                "replicas": args.replicas,
                "steps": 0,
                "requested_steps": 321,
                "rollout_started": False,
                "initialization": initialization,
                "schedule": schedule.as_dict(),
                "free_object": True,
                "gravity": 0,
                "ground": False,
                "support": False,
                "initialization_object_state_writes": 2,
                "initialization_wrist_state_writes": 1,
                "rollout_object_state_writes": 0,
                "rollout_wrist_state_writes": 0,
                "hidden_force": False,
                "hidden_attachment": False,
                "corrected_trajectory_used": False,
                "source_object_pose_used": False,
                "initialization_static_pass": False,
                "development_pass": False,
                "exact_audit_pending": False,
                "wall_time_physx_s": time.perf_counter() - started,
                "status": "STAGE16D_STABLE_GRASP_DEVELOPMENT_STATIC_INITIALIZATION_FAILED",
            }
            _write_report(args.output, runtime)
            return runtime
        actions = schedule.actions(
            contact_groups=candidate["contact_groups"],
            closure_amplitude=float(candidate["closure_amplitude"]),
            wrist_approach_direction_local=approach_direction_local,
        )
        hand_ids = torch.tensor(
            [env._robot.body_names.index(name) for name in HAND_COLLISION_BODY_NAMES],
            dtype=torch.long,
            device=env.device,
        )
        object_rows: list[np.ndarray] = []
        hand_rows: list[np.ndarray] = []
        force_rows: list[np.ndarray] = []
        pair_rows: list[np.ndarray] = []
        group_rows: list[np.ndarray] = []
        twist_rows: list[np.ndarray] = []
        effort_rows: list[np.ndarray] = []
        finite_rows: list[np.ndarray] = []
        reason_rows: list[np.ndarray] = []
        initial_object_position: np.ndarray | None = None
        for action_row in actions:
            action = torch.as_tensor(action_row, device=env.device).expand(args.replicas, -1)
            raw_control_step(env, action)
            state = env._state()
            object_pose = torch.cat(
                (state["object_position_scene"], state["object_quaternion_wxyz"]), dim=-1
            )
            if initial_object_position is None:
                initial_object_position = object_pose[:, :3].detach().cpu().numpy().copy()
            hand_position = env._robot.data.body_link_pos_w.index_select(1, hand_ids)
            hand_position = hand_position - env.scene.env_origins[:, None, :]
            hand_quaternion = env._robot.data.body_link_quat_w.index_select(1, hand_ids)
            force, _ = _contact_presence(env, clip_index=clip_index)
            interval_pairs, grouped = _interval_contact_presence(env)
            object_rows.append(object_pose.detach().cpu().numpy().copy())
            hand_rows.append(
                torch.cat((hand_position, hand_quaternion), dim=-1).detach().cpu().numpy().copy()
            )
            force_rows.append(force.detach().cpu().numpy().copy())
            pair_rows.append(interval_pairs.detach().cpu().numpy().copy())
            group_rows.append(grouped.detach().cpu().numpy().copy())
            twist_rows.append(state["object_twist_world"].detach().cpu().numpy().copy())
            effort_rows.append(
                env._robot.data.applied_torque.abs().mean(dim=-1).detach().cpu().numpy().copy()
            )
            finite_rows.append(
                torch.cat(
                    (
                        object_pose,
                        state["object_twist_world"],
                        state["finger_q"],
                    ),
                    dim=-1,
                )
                .isfinite()
                .all(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            reason_rows.append(env._reason_codes.detach().cpu().numpy().copy())
        trace: dict[str, Any] = {
            "object_pose": np.stack(object_rows),
            "hand_collision_body_pose": np.stack(hand_rows),
            "hand_collision_body_names": np.asarray(HAND_COLLISION_BODY_NAMES),
            "contact_force_world": np.stack(force_rows),
            "contact_pair_presence": np.stack(pair_rows),
            "contact_group_presence": np.stack(group_rows),
            "contact_group_names": np.asarray(CALIBRATION_GROUP_ORDER),
            "object_twist": np.stack(twist_rows),
            "mean_absolute_effort": np.stack(effort_rows),
            "finite": np.stack(finite_rows),
            "reason_code": np.stack(reason_rows),
            "actions": actions,
            "initialization_static_pass": np.asarray(
                initialization["exact_balanced_contact_refinement"]["static_contact_pass"],
                dtype=bool,
            ),
        }
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace, **trace)
        assert initial_object_position is not None
        displacement = np.linalg.norm(
            trace["object_pose"][..., :3] - initial_object_position[None, ...], axis=-1
        )
        selected_group_indices = [
            CALIBRATION_GROUP_ORDER.index(group) for group in candidate["contact_groups"]
        ]
        selected_presence = trace["contact_group_presence"][..., selected_group_indices]
        initialization_static_pass = bool(trace["initialization_static_pass"])
        runtime = {
            "schema_version": "StableFreeObjectGraspCalibrationRunV1",
            "object_id": args.object,
            "candidate": candidate,
            "replicas": args.replicas,
            "steps": 321,
            "initialization": initialization,
            "schedule": schedule.as_dict(),
            "free_object": True,
            "gravity": 0,
            "ground": False,
            "support": False,
            "initialization_object_state_writes": 3,
            "initialization_wrist_state_writes": 1,
            "rollout_object_state_writes": 0,
            "rollout_wrist_state_writes": 0,
            "hidden_force": False,
            "hidden_attachment": False,
            "corrected_trajectory_used": False,
            "source_object_pose_used": False,
            "rollout_started": True,
            "initialization_static_pass": initialization_static_pass,
            "action_bounds_pass": bool(np.max(np.abs(actions)) <= 1.0),
            "finite_replica_pass": trace["finite"].all(axis=0).tolist(),
            "real_contact_replica_pass": selected_presence.any(axis=(0, 2)).tolist(),
            "topology_replica_pass": selected_presence.any(axis=0).all(axis=-1).tolist(),
            "terminal_topology_replica_pass": (
                selected_presence[-100:].any(axis=0).all(axis=-1).tolist()
            ),
            "workspace_replica_pass": (displacement.max(axis=0) <= 0.5).tolist(),
            "trace": str(args.trace.relative_to(REPO_ROOT)),
            "trace_sha256": _sha256(args.trace),
            "wall_time_physx_s": time.perf_counter() - started,
            "exact_audit_pending": True,
            "status": "STAGE16D_STABLE_GRASP_PHYSX_COMPLETE_EXACT_PENDING",
        }
        # Some Isaac builds terminate the interpreter while closing the env.
        # Persist the complete PhysX/init record first so exact-only recovery
        # cannot lose provenance or silently reconstruct weaker evidence.
        _write_report(args.output, runtime)
        return runtime
    finally:
        _close_env(env)


def _exact_audit(
    args: argparse.Namespace, candidate: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    from toporetarget.rl.geometry_audit.exact_evaluator import evaluate_runtime_proxy_state

    started = time.perf_counter()
    with np.load(args.trace, allow_pickle=False) as trace:
        aggregate, raw = evaluate_runtime_proxy_state(
            manifest_path=MANIFEST_PATH,
            clip=args.object,
            object_pose=np.asarray(trace["object_pose"], dtype=np.float64),
            hand_collision_body_pose=np.asarray(
                trace["hand_collision_body_pose"], dtype=np.float64
            ),
            hand_collision_body_names=tuple(
                str(value) for value in trace["hand_collision_body_names"]
            ),
        )
        presence = np.asarray(trace["contact_group_presence"], dtype=bool)
        group_names = tuple(str(value) for value in trace["contact_group_names"])
        selected_presence = presence[
            ...,
            [group_names.index(group) for group in candidate["contact_groups"]],
        ]
        twist = np.asarray(trace["object_twist"], dtype=np.float64)
        finite = np.asarray(trace["finite"], dtype=bool).all(axis=0)
        object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
        displacement = np.linalg.norm(object_pose[..., :3] - object_pose[:1, :, :3], axis=-1)
        actions = np.asarray(trace["actions"], dtype=np.float64)
        effort = np.asarray(trace["mean_absolute_effort"], dtype=np.float64)
        initialization_static_pass = bool(trace["initialization_static_pass"])
    args.raw_geometry.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_geometry, **cast(Any, raw))
    frame_values = np.asarray(raw["frame_worst_penetration_m"], dtype=np.float64)
    replica_max = frame_values.max(axis=0)
    replica_p95 = np.asarray(
        [
            np.quantile(values[values > 0.0], 0.95) if np.any(values > 0.0) else 0.0
            for values in frame_values.T
        ]
    )
    runtime.update(
        {
            "formal_geometry": aggregate,
            "formal_geometry_raw": str(args.raw_geometry.relative_to(REPO_ROOT)),
            "formal_geometry_raw_sha256": _sha256(args.raw_geometry),
            "replica_max_penetration_m": replica_max.tolist(),
            "replica_active_p95_penetration_m": replica_p95.tolist(),
            "wall_time_exact_s": time.perf_counter() - started,
            "initialization_static_pass": initialization_static_pass,
            "selection_metrics": {
                "topology_coverage": float(selected_presence.any(axis=0).all(axis=-1).mean()),
                "terminal_hold_stability": float(selected_presence[-100:].all(axis=-1).mean()),
                "contact_persistence": float(selected_presence.all(axis=-1).mean()),
                "terminal_linear_speed_p95_max_mps": float(
                    np.quantile(np.linalg.norm(twist[-100:, :, :3], axis=-1), 0.95, axis=0).max()
                ),
                "terminal_angular_speed_p95_max_radps": float(
                    np.quantile(np.linalg.norm(twist[-100:, :, 3:], axis=-1), 0.95, axis=0).max()
                ),
                "active_p95_penetration_m": float(aggregate["p95_penetration_m"]),
                "max_penetration_m": float(aggregate["max_penetration_m"]),
                "mean_absolute_effort": float(effort.mean()),
                "mean_action_variation": float(
                    np.linalg.norm(np.diff(actions, axis=0), axis=-1).mean()
                ),
            },
        }
    )
    if args.replicas == 20:
        qualification = qualify_stable_grasp(
            contact_group_presence=selected_presence,
            object_twist=twist,
            finite=finite,
            action_bounds_pass=np.full(20, np.max(np.abs(actions)) <= 1.0),
            workspace_pass=displacement.max(axis=0) <= 0.5,
            exact_replica_max_penetration_m=replica_max,
            exact_replica_active_p95_m=replica_p95,
            gate=StableGraspCalibrationGateV1(),
        )
        qualification["hard_gates"]["initialization_static_pass"] = initialization_static_pass
        qualification["passed"] = bool(qualification["passed"] and initialization_static_pass)
        qualification["status"] = (
            "STAGE16D_STABLE_GRASP_CALIBRATION_VALIDATED"
            if qualification["passed"]
            else "STAGE16D_STABLE_GRASP_CALIBRATION_FAILED"
        )
        runtime["qualification"] = qualification
        runtime["status"] = qualification["status"]
    else:
        group_ever = selected_presence.any(axis=0).all(axis=-1)
        final_presence = selected_presence[-100:].all(axis=-1)
        linear_p95 = np.quantile(np.linalg.norm(twist[-100:, :, :3], axis=-1), 0.95, axis=0)
        angular_p95 = np.quantile(np.linalg.norm(twist[-100:, :, 3:], axis=-1), 0.95, axis=0)
        development_pass = bool(
            initialization_static_pass
            and finite.all()
            and selected_presence.any(axis=(0, 2)).all()
            and group_ever.all()
            and final_presence.mean() >= 0.95
            and np.all(linear_p95 <= 0.01)
            and np.all(angular_p95 <= 0.10)
            and np.all(replica_max < 0.010)
            and np.all(replica_p95 <= 0.003)
        )
        runtime["development_pass"] = development_pass
        runtime["status"] = (
            "STAGE16D_STABLE_GRASP_DEVELOPMENT_PASSED"
            if development_pass
            else "STAGE16D_STABLE_GRASP_DEVELOPMENT_FAILED"
        )
    runtime["exact_audit_pending"] = False
    return runtime


def _runtime_from_trace(args: argparse.Namespace, candidate: dict[str, Any]) -> dict[str, Any]:
    with np.load(args.trace, allow_pickle=False) as trace:
        replicas = int(trace["object_pose"].shape[1])
        initialization_static_pass = bool(trace["initialization_static_pass"])
    return {
        "schema_version": "StableFreeObjectGraspCalibrationRunV1",
        "object_id": args.object,
        "candidate": candidate,
        "replicas": replicas,
        "steps": 321,
        "free_object": True,
        "rollout_object_state_writes": 0,
        "rollout_wrist_state_writes": 0,
        "hidden_force": False,
        "hidden_attachment": False,
        "corrected_trajectory_used": False,
        "initialization_static_pass": initialization_static_pass,
        "trace": str(args.trace.relative_to(REPO_ROOT)),
        "trace_sha256": _sha256(args.trace),
        "recovered_from_existing_trace": True,
    }


def _fail_closed_exact_audit(runtime: dict[str, Any], error: RuntimeError) -> dict[str, Any]:
    """Turn an exact-backend disagreement into an explicit candidate failure."""

    message = str(error)
    if not message.startswith("STAGE16D_CONVEX_QUERY_"):
        raise error
    runtime.update(
        {
            "exact_audit_pending": False,
            "development_pass": False,
            "exact_audit_failure": {
                "schema_version": "StableGraspExactAuditFailureV1",
                "backend": "python-fcl==0.7.0.11",
                "error": message,
                "fallback_used": False,
                "metric_value_imputed": False,
                "candidate_rejected_fail_closed": True,
            },
            "status": "STAGE16D_STABLE_GRASP_EXACT_GEOMETRY_AUDIT_FAILED",
        }
    )
    return runtime


def main() -> int:
    args = _parser().parse_args()
    args.output = args.output.resolve()
    args.trace = args.trace.resolve()
    args.raw_geometry = args.raw_geometry.resolve()
    args.contract_root = args.contract_root.resolve()
    if not args.accept_eula:
        raise SystemExit("Stage16D Isaac calibration requires --accept-eula")
    candidate = _candidate(args)
    pending_runtime: dict[str, Any] | None = None
    if args.exact_only and args.output.exists():
        pending_runtime = _load(args.output)
        expected_status = "STAGE16D_STABLE_GRASP_PHYSX_COMPLETE_EXACT_PENDING"
        if pending_runtime.get("status") != expected_status:
            raise FileExistsError(args.output)
        if pending_runtime.get("trace_sha256") != _sha256(args.trace):
            raise RuntimeError("STAGE16D_EXACT_RECOVERY_TRACE_HASH_MISMATCH")
    elif args.output.exists() or (args.raw_geometry.exists() and not args.exact_only):
        raise FileExistsError(args.output if args.output.exists() else args.raw_geometry)

    def write_result(result: dict[str, Any]) -> None:
        _write_report(args.output, result)
        print(
            json.dumps({"status": result["status"], "output": str(args.output)}),
            flush=True,
        )

    if args.exact_only:
        if not args.trace.is_file():
            raise FileNotFoundError(args.trace)
        runtime = (
            pending_runtime if pending_runtime is not None else _runtime_from_trace(args, candidate)
        )
        try:
            result = _exact_audit(args, candidate, runtime)
        except RuntimeError as error:
            result = _fail_closed_exact_audit(runtime, error)
        write_result(result)
    else:
        if args.trace.exists():
            raise FileExistsError(args.trace)
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True).app
        try:
            runtime = _run_physx(args, candidate)
            if runtime["exact_audit_pending"]:
                try:
                    result = _exact_audit(args, candidate, runtime)
                except RuntimeError as error:
                    result = _fail_closed_exact_audit(runtime, error)
            else:
                result = runtime
            # Kit shutdown may terminate the interpreter in some Isaac builds.
            write_result(result)
        finally:
            app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
