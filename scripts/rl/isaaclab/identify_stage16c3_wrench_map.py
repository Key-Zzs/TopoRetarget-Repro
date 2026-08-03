#!/usr/bin/env python3
"""Identify the bounded Path-A free-root wrench map in batched GPU PhysX probes.

This diagnostic never uses ``DirectRLEnv.step``.  It resets isolated probe
states, removes both task objects, and measures signed one-substep root
accelerations.  The generated JSON is a read-only controller input; it is not
an execution rollout and writes no state during a rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
PROFILE_PATH = REPO_ROOT / "configs/rl/stage16/isaaclab_inverse_wrench.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3r2_c5/free_root/wrench_map.json",
    )
    return parser.parse_args()


def _profile() -> dict[str, Any]:
    profile = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile["profile"]["identifier"] != "identified_inverse_wrench_v1":
        raise ValueError("unexpected Stage 16-C.3 Path-A profile")
    return profile


def _list(value: Any) -> list[Any]:
    return value.detach().cpu().tolist()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    profile = _profile()
    probe_profile = profile["profile"]["probe"]
    frame_indices = tuple(int(value) for value in probe_profile["frame_indices"])
    if frame_indices != (0, 10, 20, 30, 40):
        raise SystemExit("the frozen Path-A probe frame set must be [0, 10, 20, 30, 40]")
    if int(probe_profile["num_envs"]) != 12:
        raise SystemExit("Path-A signed-basis probe requires exactly 12 parallel environments")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.inverse_wrench_controller import (
            BatchedEffectiveWrenchMapIdentifier,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = int(probe_profile["num_envs"])
        cfg.balanced_clip_assignment = False
        cfg.contact_telemetry = "off"
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env_ids = torch.arange(env.num_envs, device=env.device)
        wrist_body = torch.tensor([env._wrist_body_id], device=env.device)
        signed_axes = torch.arange(6, device=env.device).repeat_interleave(2)
        signed_values = torch.tensor([-1.0, 1.0] * 6, dtype=torch.float32, device=env.device)
        amplitudes = torch.tensor(
            [
                float(probe_profile["force_amplitude_n"]),
                float(probe_profile["force_amplitude_n"]),
                float(probe_profile["force_amplitude_n"]),
                float(probe_profile["torque_amplitude_nm"]),
                float(probe_profile["torque_amplitude_nm"]),
                float(probe_profile["torque_amplitude_nm"]),
            ],
            device=env.device,
        )
        signed_wrenches = torch.zeros((env.num_envs, 6), device=env.device)
        signed_wrenches[torch.arange(env.num_envs, device=env.device), signed_axes] = (
            signed_values * amplitudes[signed_axes]
        )

        def reset_probe_state(*, clip_index: int, frame: int, condition: str) -> None:
            env._clip_index.fill_(clip_index)
            env._reset_idx(env_ids)
            # ``_reset_idx`` intentionally resets normal rollouts to frame 0.
            # This isolated map identifier explicitly replaces that setup with
            # the requested frozen probe frame before any raw PhysX step.
            env._clip_index.fill_(clip_index)
            env._reference_index.fill_(frame)
            env._target_reference_index.fill_(frame)
            wrist_position = (
                env.reference_bank.gather(
                    "wrist_pose_translation_world_ref", env._clip_index, env._reference_index
                )
                + env.scene.env_origins
            )
            wrist_quaternion = env.reference_bank.gather(
                "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, env._reference_index
            )
            wrist_twist = env.reference_bank.gather(
                "wrist_twist_world_ref", env._clip_index, env._reference_index
            )
            env._robot.write_root_state_to_sim(
                torch.cat((wrist_position, wrist_quaternion, wrist_twist), dim=-1), env_ids=env_ids
            )
            q_ref = env.action_adapter.canonical_to_isaac(
                env.reference_bank.gather("q_finger_ref", env._clip_index, env._reference_index)
            )
            qdot_ref = env.action_adapter.canonical_to_isaac(
                env.reference_bank.gather("qdot_finger_ref", env._clip_index, env._reference_index)
            )
            env._robot.write_joint_state_to_sim(q_ref, qdot_ref, env_ids=env_ids)
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
            # Explicitly fenced diagnostic setup: objects are absent from every
            # raw PhysX probe and cannot create contact or causal motion.
            env._object_170105.write_root_state_to_sim(inactive_state, env_ids=env_ids)
            env._object_170650.write_root_state_to_sim(inactive_state, env_ids=env_ids)
            if condition == "finger_locked_hold_target":
                env._robot.write_joint_state_to_sim(q_ref, torch.zeros_like(q_ref), env_ids=env_ids)
                target = q_ref
            elif condition == "finger_reference_target_active":
                target = q_ref
            else:
                raise ValueError(f"unknown Path-A finger condition: {condition}")
            env._robot.set_joint_position_target(target)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)

        def raw_step(wrench_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            before_twist = torch.cat(
                (env._robot.data.root_lin_vel_w, env._robot.data.root_ang_vel_w), dim=-1
            )
            before_q = env._robot.data.joint_pos.clone()
            env._robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=wrench_world[:, None, :3],
                torques=wrench_world[:, None, 3:],
                body_ids=wrist_body,
                is_global=True,
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            after_twist = torch.cat(
                (env._robot.data.root_lin_vel_w, env._robot.data.root_ang_vel_w), dim=-1
            )
            return (after_twist - before_twist) / env.physics_dt, (
                env._robot.data.joint_pos - before_q
            )

        reports: list[dict[str, object]] = []
        control_response: list[list[list[list[float]]]] = []
        control_drift: list[list[list[float]]] = []
        all_pass = True
        for clip_index, clip_id in enumerate(env.reference_bank.clip_ids):
            clip_control_response: list[list[list[float]]] = []
            clip_control_drift: list[list[float]] = []
            for frame in frame_indices:
                per_condition: dict[str, dict[str, object]] = {}
                for condition in probe_profile["finger_conditions"]:
                    reset_probe_state(clip_index=clip_index, frame=frame, condition=condition)
                    zero_acceleration, zero_joint_delta = raw_step(
                        torch.zeros((env.num_envs, 6), device=env.device)
                    )
                    reset_probe_state(clip_index=clip_index, frame=frame, condition=condition)
                    signed_acceleration, signed_joint_delta = raw_step(signed_wrenches)
                    negative = signed_acceleration[0::2]
                    positive = signed_acceleration[1::2]
                    response = BatchedEffectiveWrenchMapIdentifier.central_difference(
                        positive_acceleration=positive,
                        negative_acceleration=negative,
                        amplitudes=amplitudes,
                    )
                    diagnostics = BatchedEffectiveWrenchMapIdentifier.diagnostics(response)
                    condition_number = float(diagnostics["condition_number"].item())
                    condition_pass = condition_number <= float(
                        profile["profile"]["controller"]["condition_number_max"]
                    )
                    finite = bool(
                        torch.isfinite(response).all()
                        and torch.isfinite(zero_acceleration).all()
                        and torch.isfinite(signed_acceleration).all()
                    )
                    locked_joint_delta = float(
                        torch.max(
                            torch.abs(torch.cat((zero_joint_delta, signed_joint_delta)))
                        ).item()
                    )
                    entry = {
                        "finite": finite,
                        "condition_number": condition_number,
                        "condition_gate_pass": condition_pass,
                        "singular_values": _list(diagnostics["singular_values"]),
                        "cross_axis_coupling_ratio": _list(
                            diagnostics["cross_axis_coupling_ratio"]
                        ),
                        "zero_wrench_acceleration_world": _list(zero_acceleration.mean(dim=0)),
                        "response_acceleration_per_wrench_world": _list(response),
                        "finger_joint_max_delta_rad": locked_joint_delta,
                        "signed_wrench_world": _list(signed_wrenches),
                    }
                    per_condition[condition] = entry
                    all_pass = all_pass and finite and condition_pass
                    if condition == "finger_reference_target_active":
                        clip_control_response.append(_list(response))
                        clip_control_drift.append(_list(zero_acceleration.mean(dim=0)))
                reports.append(
                    {
                        "clip": clip_id,
                        "frame": frame,
                        "conditions": per_condition,
                    }
                )
            control_response.append(clip_control_response)
            control_drift.append(clip_control_drift)
        result = {
            "status": "C3_EFFECTIVE_WRENCH_MAP_IDENTIFIED"
            if all_pass
            else "C3_EFFECTIVE_WRENCH_MAP_CONDITION_BLOCKED",
            "profile_path": str(PROFILE_PATH),
            "profile": profile["profile"],
            "scope": (
                "isolated GPU PhysX raw probes; both objects absent; no DirectRLEnv rollout; "
                "signed central difference is baseline-subtracted by zero-wrench acceleration"
            ),
            "control_map": {
                "clip_ids": list(env.reference_bank.clip_ids),
                "frame_indices": list(frame_indices),
                "finger_condition": "finger_reference_target_active",
                "response_acceleration_per_wrench_world": control_response,
                "zero_wrench_acceleration_world": control_drift,
                "selection": "nearest frozen probe frame at each physics substep",
            },
            "probes": reports,
            "execution_state_writes": {
                "wrist_or_object_during_rollout": 0,
                "diagnostic_probe_state_setup": "explicitly fenced raw PhysX setup only",
            },
            "contract": env.contract_report(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if all_pass else 1
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
