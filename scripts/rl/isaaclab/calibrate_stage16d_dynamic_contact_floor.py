#!/usr/bin/env python3
"""Run 20-replica no-contact, source-following, or stable-contact calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_geometry_aware_optimization_ppo"
BASELINE_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
MANIFEST_PATH = BASELINE_ROOT / "runtime_collision_geometry_manifest.json"
CLIPS = ("hocap_170105", "hocap_170650")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("no-contact", "source-following", "stable-contact"), required=True
    )
    parser.add_argument("--clip", choices=CLIPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--raw-geometry", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--steps", type=int, default=321)
    parser.add_argument("--contact-frame", type=int)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close_env(env: Any) -> None:
    if env is not None:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def _source_contact_frame(clip: str) -> int:
    suffix = clip.removeprefix("hocap_")
    path = BASELINE_ROOT / f"source_runtime_penetration_pairs_{suffix}.npz"
    with np.load(path, allow_pickle=False) as source:
        values = np.asarray(source["frame_worst_penetration_m"], dtype=np.float64)[:, 0]
    if not np.any(values > 0.0):
        raise RuntimeError(f"STAGE16D_SOURCE_CONTACT_FRAME_MISSING:{clip}")
    return int(np.argmax(values))


def _freeze_source_contact_reference(env: Any, *, clip_index: int, frame: int) -> None:
    """Repeat one source-only contact key in process; source files remain immutable."""

    import torch

    from toporetarget.rl.environments.isaaclab_backend.explicit_wrist_reference import (
        ExplicitWristJointReferenceV2,
    )

    bank = env.reference_bank
    if not 0 <= frame < bank.frame_count:
        raise ValueError("stable contact frame is outside the retimed source")
    for field in bank.REQUIRED_FIELDS:
        if field == "timestamps":
            continue
        values = getattr(bank, field)
        frozen = values[clip_index, frame].clone()
        values[clip_index] = frozen.expand_as(values[clip_index])
    bank.wrist_twist_world_ref[clip_index].zero_()
    bank.qdot_finger_ref[clip_index].zero_()
    bank.object_twist_world_ref[clip_index].zero_()
    env._explicit_joint_reference = ExplicitWristJointReferenceV2.from_reference_bank(bank)
    if not bool(torch.isfinite(env._explicit_joint_reference.q_wrist_ref).all()):
        raise RuntimeError("STAGE16D_STABLE_CONTACT_REFERENCE_NONFINITE")


def _separate_active_object(env: Any, *, clip_index: int, ids: Any) -> None:
    """Apply one initialization-only separation write before the rollout begins."""

    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import synchronize_reset_boundary

    obj = env._object_170105 if clip_index == 0 else env._object_170650
    state = obj.data.root_state_w.index_select(0, ids).clone()
    state[:, :3] += torch.tensor([0.0, 0.0, 1.0], device=env.device)
    state[:, 7:] = 0.0
    obj.write_root_state_to_sim(state, env_ids=ids)
    synchronize_reset_boundary(env)


def _run_physx(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from optimize_stage16d_physics_trajectory import make_env

    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )
    from toporetarget.rl.isaaclab_oracle.history_replay import (
        raw_control_step,
        synchronize_reset_boundary,
    )
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    if args.replicas != 20 or args.steps != 321:
        raise ValueError("formal dynamic calibration requires exactly 20 replicas and 321 steps")
    clip_index = CLIPS.index(args.clip)
    env = None
    started = time.perf_counter()
    try:
        env = make_env(num_envs=args.replicas, clip=args.clip, telemetry="aggregate")
        ids = torch.arange(args.replicas, dtype=torch.long, device=env.device)
        contact_frame = args.contact_frame
        if args.mode == "stable-contact":
            contact_frame = (
                contact_frame if contact_frame is not None else _source_contact_frame(args.clip)
            )
            _freeze_source_contact_reference(env, clip_index=clip_index, frame=contact_frame)
        reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=ids)
        if args.mode == "stable-contact":
            synchronize_reset_boundary(env)
        initialization_state_writes = 1
        if args.mode == "no-contact":
            _separate_active_object(env, clip_index=clip_index, ids=ids)
            initialization_state_writes += 1

        hand_body_ids = torch.tensor(
            [env._robot.body_names.index(name) for name in HAND_COLLISION_BODY_NAMES],
            dtype=torch.long,
            device=env.device,
        )
        required_groups = tuple(env._topology_contracts[args.clip]["required_body_groups"])
        required_slots = {
            group: torch.tensor(env._group_slots[group], dtype=torch.long, device=env.device)
            for group in required_groups
        }
        object_rows: list[np.ndarray] = []
        hand_rows: list[np.ndarray] = []
        force_rows: list[np.ndarray] = []
        pair_presence_rows: list[np.ndarray] = []
        required_presence_rows: list[np.ndarray] = []
        object_twist_rows: list[np.ndarray] = []
        reason_rows: list[np.ndarray] = []
        zero_action = torch.zeros((args.replicas, 26), dtype=torch.float32, device=env.device)
        for _ in range(args.steps):
            raw_control_step(env, zero_action)
            state = env._state()
            object_pose = torch.cat(
                (state["object_position_scene"], state["object_quaternion_wxyz"]), dim=-1
            )
            hand_position = env._robot.data.body_link_pos_w.index_select(1, hand_body_ids)
            hand_position = hand_position - env.scene.env_origins[:, None, :]
            hand_quaternion = env._robot.data.body_link_quat_w.index_select(1, hand_body_ids)
            hand_pose = torch.cat((hand_position, hand_quaternion), dim=-1)
            first_force = env._object_contact_sensors["Object170105"].data.force_matrix_w
            second_force = env._object_contact_sensors["Object170650"].data.force_matrix_w
            if first_force is None or second_force is None:
                raise RuntimeError("STAGE16D_CALIBRATION_CONTACT_FORCE_MATRIX_UNAVAILABLE")
            pair_force = torch.where(
                (env._clip_index == 0)[:, None, None], first_force[:, 0], second_force[:, 0]
            )
            pair_presence = torch.linalg.vector_norm(pair_force, dim=-1) > 1.0e-4
            required_presence = torch.stack(
                tuple(
                    pair_presence.index_select(1, slots).any(dim=-1)
                    for slots in required_slots.values()
                ),
                dim=-1,
            ).all(dim=-1)
            object_rows.append(object_pose.detach().cpu().numpy().copy())
            hand_rows.append(hand_pose.detach().cpu().numpy().copy())
            force_rows.append(pair_force.sum(dim=1).detach().cpu().numpy().copy())
            pair_presence_rows.append(pair_presence.detach().cpu().numpy().copy())
            required_presence_rows.append(required_presence.detach().cpu().numpy().copy())
            object_twist_rows.append(state["object_twist_world"].detach().cpu().numpy().copy())
            reason_rows.append(env._reason_codes.detach().cpu().numpy().copy())
        trace = {
            "object_pose": np.stack(object_rows),
            "hand_collision_body_pose": np.stack(hand_rows),
            "hand_collision_body_names": np.asarray(HAND_COLLISION_BODY_NAMES),
            "contact_force_world": np.stack(force_rows),
            "contact_pair_presence": np.stack(pair_presence_rows),
            "required_contact_presence": np.stack(required_presence_rows),
            "object_twist": np.stack(object_twist_rows),
            "reason_code": np.stack(reason_rows),
        }
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.trace, **trace)
        force_norm = np.linalg.norm(trace["contact_force_world"], axis=-1)
        required_presence = trace["required_contact_presence"]
        return {
            "schema_version": "Stage16DDynamicContactCalibrationTraceV1",
            "mode": args.mode,
            "clip": args.clip,
            "replicas": args.replicas,
            "control_steps": args.steps,
            "contact_frame": contact_frame,
            "contact_frame_provenance": (
                "source-only RuntimeCollisionProxyPenetrationV1 maximum frame"
                if args.mode == "stable-contact"
                else "not_applicable"
            ),
            "source_files_modified": False,
            "free_object": True,
            "initialization_state_writes": initialization_state_writes,
            "rollout_object_state_writes": 0,
            "rollout_wrist_state_writes": 0,
            "hidden_force": False,
            "hidden_attachment": False,
            "required_contact_groups": list(required_groups),
            "required_contact_present_rate": float(required_presence.mean()),
            "required_contact_final100_rate": float(required_presence[-100:].mean()),
            "contact_force_max_n": float(force_norm.max()),
            "contact_force_active_p95_n": (
                float(np.quantile(force_norm[force_norm > 0.0], 0.95))
                if np.any(force_norm > 0.0)
                else 0.0
            ),
            "object_linear_speed_max_mps": float(
                np.linalg.norm(trace["object_twist"][..., :3], axis=-1).max()
            ),
            "object_angular_speed_max_radps": float(
                np.linalg.norm(trace["object_twist"][..., 3:], axis=-1).max()
            ),
            "trace": str(args.trace.relative_to(REPO_ROOT)),
            "trace_sha256": _sha256(args.trace),
            "wall_time_physx_s": time.perf_counter() - started,
        }
    finally:
        _close_env(env)


def _exact_audit(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    from toporetarget.rl.geometry_audit.exact_evaluator import evaluate_runtime_proxy_state

    started = time.perf_counter()
    with np.load(args.trace, allow_pickle=False) as trace:
        aggregate, raw = evaluate_runtime_proxy_state(
            manifest_path=MANIFEST_PATH,
            clip=args.clip,
            object_pose=np.asarray(trace["object_pose"], dtype=np.float64),
            hand_collision_body_pose=np.asarray(
                trace["hand_collision_body_pose"], dtype=np.float64
            ),
            hand_collision_body_names=tuple(
                str(value) for value in trace["hand_collision_body_names"]
            ),
        )
    args.raw_geometry.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw_geometry, **raw)
    replica_max = np.asarray(raw["frame_worst_penetration_m"], dtype=np.float64).max(axis=0)
    runtime.update(
        {
            "formal_geometry": aggregate,
            "formal_geometry_raw": str(args.raw_geometry.relative_to(REPO_ROOT)),
            "formal_geometry_raw_sha256": _sha256(args.raw_geometry),
            "replica_p99_max_penetration_m": float(np.quantile(replica_max, 0.99)),
            "pooled_active_p95_penetration_m": float(aggregate["p95_penetration_m"]),
            "wall_time_exact_s": time.perf_counter() - started,
        }
    )
    if args.mode == "no-contact":
        passed = aggregate["max_penetration_m"] <= 5.0e-7
        runtime["status"] = (
            "STAGE16D_NO_CONTACT_GEOMETRY_FLOOR_VALIDATED"
            if passed
            else "STAGE16D_NO_CONTACT_GEOMETRY_FLOOR_BLOCKED"
        )
    elif args.mode == "stable-contact":
        passed = runtime["required_contact_final100_rate"] >= 0.95
        runtime["status"] = (
            "STAGE16D_STABLE_DYNAMIC_CONTACT_CALIBRATED"
            if passed
            else "STAGE16D_STABLE_DYNAMIC_CONTACT_NOT_ESTABLISHED"
        )
    else:
        runtime["status"] = "STAGE16D_SOURCE_DYNAMIC_CONTACT_COMPARISON_RECORDED"
    return runtime


def _runtime_from_trace(args: argparse.Namespace) -> dict[str, Any]:
    """Recover exact-audit metadata when Kit exited after writing the PhysX trace."""

    topology_path = (
        REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting/contact_topology.json"
    )
    topology = json.loads(topology_path.read_text(encoding="utf-8"))["clips"][args.clip]
    with np.load(args.trace, allow_pickle=False) as trace:
        force = np.asarray(trace["contact_force_world"], dtype=np.float64)
        force_norm = np.linalg.norm(force, axis=-1)
        required_presence = np.asarray(trace["required_contact_presence"], dtype=bool)
        object_twist = np.asarray(trace["object_twist"], dtype=np.float64)
        steps, replicas = required_presence.shape
    contact_frame = args.contact_frame
    if args.mode == "stable-contact" and contact_frame is None:
        contact_frame = _source_contact_frame(args.clip)
    return {
        "schema_version": "Stage16DDynamicContactCalibrationTraceV1",
        "mode": args.mode,
        "clip": args.clip,
        "replicas": replicas,
        "control_steps": steps,
        "contact_frame": contact_frame,
        "contact_frame_provenance": (
            "source-only RuntimeCollisionProxyPenetrationV1 maximum frame"
            if args.mode == "stable-contact"
            else "not_applicable"
        ),
        "source_files_modified": False,
        "free_object": True,
        "initialization_state_writes": 2 if args.mode == "no-contact" else 1,
        "rollout_object_state_writes": 0,
        "rollout_wrist_state_writes": 0,
        "hidden_force": False,
        "hidden_attachment": False,
        "required_contact_groups": list(topology["required_body_groups"]),
        "required_contact_present_rate": float(required_presence.mean()),
        "required_contact_final100_rate": float(required_presence[-100:].mean()),
        "contact_force_max_n": float(force_norm.max()),
        "contact_force_active_p95_n": (
            float(np.quantile(force_norm[force_norm > 0.0], 0.95))
            if np.any(force_norm > 0.0)
            else 0.0
        ),
        "object_linear_speed_max_mps": float(np.linalg.norm(object_twist[..., :3], axis=-1).max()),
        "object_angular_speed_max_radps": float(
            np.linalg.norm(object_twist[..., 3:], axis=-1).max()
        ),
        "trace": str(args.trace.relative_to(REPO_ROOT)),
        "trace_sha256": _sha256(args.trace),
        "wall_time_physx_s": None,
        "recovered_from_existing_trace": True,
    }


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise SystemExit("Stage16D Isaac calibration requires --accept-eula")
    if args.output.exists() or args.raw_geometry.exists():
        raise FileExistsError(args.output if args.output.exists() else args.raw_geometry)
    if args.exact_only:
        if not args.trace.is_file():
            raise FileNotFoundError(args.trace)
        result = _exact_audit(args, _runtime_from_trace(args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 0
    if args.trace.exists():
        raise FileExistsError(args.trace)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        runtime = _run_physx(args)
        result = _exact_audit(args, runtime)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 0
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
