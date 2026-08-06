#!/usr/bin/env python3
"""Replay one Stage 16-D action trace over 20 independent GPU PhysX replicas."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_actions(clip: str, path: Path | None) -> tuple[Path, np.ndarray, str]:
    if path is not None:
        actions = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        source = path
        role = "physics_correction_candidate"
    else:
        from optimize_stage16d_physics_trajectory import source_actions

        source, actions = source_actions(clip)
        role = "frozen_stage16c_source_demonstration"
    if actions.shape != (321, 26) or not np.isfinite(actions).all():
        raise ValueError("Stage16D qualification actions must be finite [321,26]")
    if float(np.max(np.abs(actions))) > 1.0:
        raise ValueError("Stage16D qualification action bounds failed")
    return source, actions, role


def close_env(env: Any) -> None:
    if env is not None:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def run(args: argparse.Namespace) -> int:
    import torch
    from optimize_stage16d_physics_trajectory import make_env

    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )
    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    if args.replicas != 20:
        raise ValueError("formal Stage16D qualification requires exactly 20 replicas")
    if args.output.exists() or args.trace.exists():
        raise FileExistsError("Stage16D qualification refuses overwrite")
    source, actions_np, role = load_actions(args.clip, args.actions)
    env = None
    started = time.perf_counter()
    try:
        env = make_env(num_envs=args.replicas, clip=args.clip, telemetry="aggregate")
        ids = torch.arange(args.replicas, device=env.device)
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=ids)
        actions = torch.as_tensor(actions_np, device=env.device)
        active = torch.ones(args.replicas, dtype=torch.bool, device=env.device)
        ever_success = torch.zeros_like(active)
        ever_causality = torch.zeros_like(active)
        numerical_pass = torch.ones_like(active)
        max_progress = torch.zeros(args.replicas, device=env.device)
        max_contact = torch.zeros_like(max_progress)
        first_reason = torch.zeros(args.replicas, dtype=torch.long, device=env.device)
        terminal_stable = torch.zeros_like(active)
        early_failure = torch.zeros_like(active)
        traces: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "observation",
                "action",
                "wrist_pose",
                "wrist_twist",
                "finger_q",
                "finger_qd",
                "virtual_wrist_q",
                "virtual_wrist_qd",
                "joint_targets",
                "actuator_effort",
                "saturation",
                "object_pose",
                "object_twist",
                "object_axis_points",
                "hand_collision_body_pose",
                "replica_object_pose",
                "replica_hand_collision_body_pose",
                "semantic_progress",
                "contact_recall",
                "contact_persistence",
                "contact_causality",
                "contact_pair_presence",
                "contact_force_world",
                "contact_impulse_world",
                "terminal_stable",
                "reason_code",
            )
        }
        hand_body_ids = torch.tensor(
            [env._robot.body_names.index(name) for name in HAND_COLLISION_BODY_NAMES],
            dtype=torch.long,
            device=env.device,
        )
        for step in range(321):
            batch = actions[step].expand(args.replicas, -1)
            batch = torch.where(active[:, None], batch, torch.zeros_like(batch))
            terminated, timed_out = raw_control_step(env, batch)
            stage = env.extras["stage16d"]
            state = env._state()
            observation = env._get_observations()["policy"]
            finite = torch.isfinite(observation).all(dim=-1)
            numerical_pass &= finite
            max_progress = torch.maximum(max_progress, stage["semantic_progress"])
            max_contact = torch.maximum(max_contact, stage["contact_recall"])
            ever_success |= stage["success"]
            ever_causality |= stage["contact_causality"]
            terminal_stable = stage["terminal_stable"]
            newly_done = active & (terminated | timed_out)
            first_reason = torch.where(newly_done, stage["primary_reason_code"], first_reason)
            early_failure |= (
                newly_done
                & (stage["primary_reason_code"] >= 2)
                & (stage["primary_reason_code"] <= 8)
            )
            active &= ~newly_done
            wrist_pose = torch.cat(
                (state["wrist_position_scene"], state["wrist_quaternion_wxyz"]), dim=-1
            )
            object_pose = torch.cat(
                (state["object_position_scene"], state["object_quaternion_wxyz"]), dim=-1
            )
            hand_position = env._robot.data.body_link_pos_w.index_select(1, hand_body_ids)
            hand_position = hand_position - env.scene.env_origins[:, None, :]
            hand_quaternion = env._robot.data.body_link_quat_w.index_select(1, hand_body_ids)
            hand_pose = torch.cat((hand_position, hand_quaternion), dim=-1)
            virtual_q = env._robot.data.joint_pos[:, env._virtual_wrist_joint_ids]
            virtual_qd = env._robot.data.joint_vel[:, env._virtual_wrist_joint_ids]
            finger_target = env.action_adapter.isaac_to_canonical(env._joint_target_isaac)
            joint_targets = torch.cat((env._explicit_wrist_joint_target, finger_target), dim=-1)
            actuator_effort = torch.cat(
                (
                    env._robot.data.applied_torque[:, env._virtual_wrist_joint_ids],
                    env._robot.data.applied_torque[:, env._joint_ids],
                ),
                dim=-1,
            )
            saturation = torch.stack(
                (
                    env._force_saturated,
                    env._torque_saturated,
                    env._velocity_saturated,
                ),
                dim=-1,
            )
            first_force = env._object_contact_sensors["Object170105"].data.force_matrix_w
            second_force = env._object_contact_sensors["Object170650"].data.force_matrix_w
            if first_force is None or second_force is None:
                raise RuntimeError("Stage16D qualification contact force matrix unavailable")
            pair_force = torch.where(
                (env._clip_index == 0)[:, None, None], first_force[:, 0], second_force[:, 0]
            )
            pair_presence = torch.linalg.vector_norm(pair_force, dim=-1) > 1.0e-4
            contact_force = pair_force.sum(dim=1)
            row = {
                "observation": observation[0],
                "action": batch[0],
                "wrist_pose": wrist_pose[0],
                "wrist_twist": state["wrist_twist_world"][0],
                "finger_q": state["finger_q"][0],
                "finger_qd": state["finger_qdot"][0],
                "virtual_wrist_q": virtual_q[0],
                "virtual_wrist_qd": virtual_qd[0],
                "joint_targets": joint_targets[0],
                "actuator_effort": actuator_effort[0],
                "saturation": saturation[0],
                "object_pose": object_pose[0],
                "object_twist": state["object_twist_world"][0],
                "object_axis_points": state["object_axis_points_scene"][0],
                "hand_collision_body_pose": hand_pose[0],
                "replica_object_pose": object_pose,
                "replica_hand_collision_body_pose": hand_pose,
                "semantic_progress": stage["semantic_progress"][0],
                "contact_recall": stage["contact_recall"][0],
                "contact_persistence": stage["contact_persistence"][0],
                "contact_causality": stage["contact_causality"][0],
                "contact_pair_presence": pair_presence[0],
                "contact_force_world": contact_force[0],
                "contact_impulse_world": contact_force[0] * env.physics_dt,
                "terminal_stable": stage["terminal_stable"][0],
                "reason_code": stage["primary_reason_code"][0],
            }
            for name, value in row.items():
                traces[name].append(value.detach().cpu().numpy().copy())
        episodes = []
        for env_id in range(args.replicas):
            episodes.append(
                {
                    "replica": env_id,
                    "success": bool(ever_success[env_id].cpu()),
                    "semantic_progress": float(max_progress[env_id].cpu()),
                    "contact_recall": float(max_contact[env_id].cpu()),
                    "contact_causality_pass": bool(ever_causality[env_id].cpu()),
                    "terminal_stability_pass": bool(terminal_stable[env_id].cpu()),
                    "numerical_pass": bool(numerical_pass[env_id].cpu()),
                    "complete_trajectory": not bool(early_failure[env_id].cpu()),
                    "termination_reason_code": int(first_reason[env_id].cpu()),
                    "formal_object_state_writes": 0,
                    "formal_wrist_state_writes": 0,
                    "no_hidden_control": True,
                    "action_bounds_pass": True,
                    "penetration": "NOT_MEASURED_INDEPENDENT_AUDIT_REQUIRED",
                }
            )
        success_rate = float(np.mean([row["success"] for row in episodes]))
        progress_rate = float(np.mean([row["semantic_progress"] >= 0.30 for row in episodes]))
        contact_rate = float(np.mean([row["contact_recall"] >= 0.50 for row in episodes]))
        nonzero_progress = max(row["semantic_progress"] for row in episodes) > 0.0
        empirical_class = (
            "EMPIRICAL_SEED_CANDIDATE_PENDING_GEOMETRY"
            if success_rate >= 0.80 and progress_rate >= 0.80 and contact_rate >= 0.80
            else "EMPIRICAL_PARTIAL_CANDIDATE_PENDING_GEOMETRY"
            if success_rate >= 0.30 and contact_rate >= 0.50 and nonzero_progress
            else "DEGENERATE_SEED"
        )
        payload = {
            "schema_version": "Stage16DTrajectoryQualificationV1",
            "status": "STAGE16D_TRAJECTORY_QUALIFICATION_BLOCKED",
            "clip": args.clip,
            "replicas": args.replicas,
            "frame_zero_full_steps": 321,
            "action_source": str(source),
            "action_source_sha256": sha256(source),
            "action_role": role,
            "success_rate": success_rate,
            "semantic_reach_rate": progress_rate,
            "contact_topology_pass_rate": contact_rate,
            "contact_causality_pass_rate": float(
                np.mean([row["contact_causality_pass"] for row in episodes])
            ),
            "terminal_stability_pass_rate": float(
                np.mean([row["terminal_stability_pass"] for row in episodes])
            ),
            "numerical_pass_rate": float(np.mean([row["numerical_pass"] for row in episodes])),
            "complete_trajectory_rate": float(
                np.mean([row["complete_trajectory"] for row in episodes])
            ),
            "empirical_classification": empirical_class,
            "formal_classification": "BLOCKED_INDEPENDENT_PENETRATION_AUDIT_MISSING",
            "ppo_entry": "PPO_NOT_AUTHORIZED_FOR_CLIP",
            "penetration_audit": "NOT_RUN_NO_CORRECTED_QUALIFIED_TRAJECTORY",
            "formal_object_state_writes": 0,
            "formal_wrist_state_writes": 0,
            "hidden_force": False,
            "hidden_attachment": False,
            "episodes": episodes,
            "trace": str(args.trace),
            "wall_time_s": time.perf_counter() - started,
        }
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.trace,
            **{name: np.stack(rows) for name, rows in traces.items()},
            hand_collision_body_names=np.asarray(HAND_COLLISION_BODY_NAMES),
        )
        payload["trace_sha256"] = sha256(args.trace)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": payload["status"], "output": str(args.output)}))
        return 0
    finally:
        close_env(env)


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("Stage16D Isaac execution requires --accept-eula")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        return run(args)
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
