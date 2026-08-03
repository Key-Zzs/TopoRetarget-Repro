#!/usr/bin/env python3
"""Run the fenced C3-0 fully kinematic replay before actuator qualification.

This is the sole Stage 16-C.3 mode that writes wrist, finger, and object
states at every reference frame.  It is deliberately isolated from execution
rollouts and reports its writes so it cannot be misrepresented as dynamic
tracking or contact causality evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3r2_c5/c3/c3_0_fully_kinematic.json",
    )
    return parser.parse_args()


def _rotation_error_rad(torch: Any, current: Any, reference: Any) -> Any:
    from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
        relative_rotation_log_local,
    )

    return torch.linalg.vector_norm(relative_rotation_log_local(current, reference), dim=-1)


def _scalar(value: Any) -> float:
    return float(value.detach().cpu())


def _write_frame(env: Any, torch: Any, *, clip_index: int, frame: int) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    bank = env.reference_bank
    clip = torch.full((env.num_envs,), clip_index, dtype=torch.long, device=env.device)
    index = torch.full((env.num_envs,), frame, dtype=torch.long, device=env.device)
    wrist_position = bank.gather("wrist_pose_translation_world_ref", clip, index)
    wrist_quaternion = bank.gather("wrist_pose_quaternion_world_ref_wxyz", clip, index)
    wrist_twist = bank.gather("wrist_twist_world_ref", clip, index)
    wrist_state = torch.cat((wrist_position, wrist_quaternion, wrist_twist), dim=-1)
    env._robot.write_root_state_to_sim(wrist_state, env_ids=env_ids)
    env._robot.write_joint_state_to_sim(
        env.action_adapter.canonical_to_isaac(bank.gather("q_finger_ref", clip, index)),
        env.action_adapter.canonical_to_isaac(bank.gather("qdot_finger_ref", clip, index)),
        env_ids=env_ids,
    )
    object_position = bank.gather("object_pose_translation_world_ref", clip, index)
    object_quaternion = bank.gather("object_pose_quaternion_world_ref_wxyz", clip, index)
    object_twist = bank.gather("object_twist_world_ref", clip, index)
    active_state = torch.cat((object_position, object_quaternion, object_twist), dim=-1)
    inactive_position = env.scene.env_origins + torch.tensor(
        env.cfg.inactive_object_scene_offset, dtype=torch.float32, device=env.device
    )
    inactive_state = torch.cat(
        (
            inactive_position,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1),
            torch.zeros((env.num_envs, 6), device=env.device),
        ),
        dim=-1,
    )
    if clip_index == 0:
        env._object_170105.write_root_state_to_sim(active_state, env_ids=env_ids)
        env._object_170650.write_root_state_to_sim(inactive_state, env_ids=env_ids)
    else:
        env._object_170105.write_root_state_to_sim(inactive_state, env_ids=env_ids)
        env._object_170650.write_root_state_to_sim(active_state, env_ids=env_ids)
    env._clip_index.copy_(clip)
    env._reference_index.copy_(index)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def _measure_frame(env: Any, torch: Any, *, clip_index: int, frame: int) -> dict[str, Any]:
    bank = env.reference_bank
    state = env._state()
    current_object = env._active_object_state()
    reference_wrist_position = bank.wrist_pose_translation_world_ref[clip_index, frame]
    reference_wrist_quaternion = bank.wrist_pose_quaternion_world_ref_wxyz[clip_index, frame]
    reference_object_position = bank.object_pose_translation_world_ref[clip_index, frame]
    reference_object_quaternion = bank.object_pose_quaternion_world_ref_wxyz[clip_index, frame]
    reference_fingers = bank.q_finger_ref[clip_index, frame]
    reference_axis = bank.object_axis_points_world_ref[clip_index, frame]
    reference_links = bank.tracked_link_positions_world_ref[clip_index, frame]
    wrist_position_error = torch.linalg.vector_norm(
        env._robot.data.root_pos_w[0] - reference_wrist_position
    )
    wrist_rotation_error = _rotation_error_rad(
        torch, env._robot.data.root_quat_w, reference_wrist_quaternion[None]
    )[0]
    finger_error = torch.max(
        torch.abs(
            env.action_adapter.isaac_to_canonical(env._robot.data.joint_pos)[0] - reference_fingers
        )
    )
    object_position_error = torch.linalg.vector_norm(
        current_object[0, :3] - reference_object_position
    )
    object_rotation_error = _rotation_error_rad(
        torch, current_object[:, 3:7], reference_object_quaternion[None]
    )[0]
    object_axis_error = torch.max(
        torch.linalg.vector_norm(state["object_axis_points_scene"][0] - reference_axis, dim=-1)
    )
    tracked_link_errors = torch.linalg.vector_norm(
        env._robot.data.body_pos_w[0, env._tracked_link_ids] - reference_links, dim=-1
    )
    tracked_link_error, tracked_link_index = torch.max(tracked_link_errors, dim=0)
    return {
        "wrist_position_m": _scalar(wrist_position_error),
        "wrist_rotation_rad": _scalar(wrist_rotation_error),
        "finger_rad": _scalar(finger_error),
        "object_position_m": _scalar(object_position_error),
        "object_rotation_rad": _scalar(object_rotation_error),
        "object_axis_m": _scalar(object_axis_error),
        "tracked_link_m": _scalar(tracked_link_error),
        "tracked_link_name": env.reference_bank.tracked_link_names[int(tracked_link_index.item())],
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.frames != 41:
        raise SystemExit("C3-0 requires exactly the frozen 41 reference frames")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.balanced_clip_assignment = False
        cfg.contact_telemetry = "off"
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260803)
        tolerances = {
            "wrist_position_m": 1.0e-5,
            "wrist_rotation_rad": 1.0e-4,
            "finger_rad": 1.0e-5,
            "object_position_m": 1.0e-5,
            "object_rotation_rad": 1.0e-4,
            "object_axis_m": 1.0e-4,
            "tracked_link_m": 1.0e-4,
        }
        clip_reports: list[dict[str, object]] = []
        for clip_index, clip_id in enumerate(env.reference_bank.clip_ids):
            maxima = {name: 0.0 for name in tolerances}
            worst_tracked_link: dict[str, object] | None = None
            for frame in range(args.frames):
                _write_frame(env, torch, clip_index=clip_index, frame=frame)
                measured = _measure_frame(env, torch, clip_index=clip_index, frame=frame)
                for name, value in measured.items():
                    if name in maxima:
                        maxima[name] = max(maxima[name], value)
                if (
                    worst_tracked_link is None
                    or measured["tracked_link_m"] > worst_tracked_link["error_m"]
                ):
                    worst_tracked_link = {
                        "frame": frame,
                        "link": measured["tracked_link_name"],
                        "error_m": measured["tracked_link_m"],
                    }
            clip_reports.append(
                {
                    "clip": clip_id,
                    "frames": args.frames,
                    "maxima": maxima,
                    "worst_tracked_link": worst_tracked_link,
                    "status": "PASS"
                    if all(maxima[name] <= tolerance for name, tolerance in tolerances.items())
                    else "FAIL",
                }
            )
        passed = all(report["status"] == "PASS" for report in clip_reports)
        result = {
            "status": "C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED"
            if passed
            else "C3_REFERENCE_OR_FRAME_CONTRACT_FAILURE",
            "mode": "fully_kinematic_reference_replay",
            "scope": "C3-0 only; every wrist/finger/object state write is diagnostic replay setup",
            "tolerances": tolerances,
            "clips": clip_reports,
            "execution_state_writes": {
                "wrist_or_object_during_rollout": 0,
                "kinematic_replay_frames_written": args.frames * len(clip_reports),
            },
            "contact_telemetry": "off",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if passed else 1
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
