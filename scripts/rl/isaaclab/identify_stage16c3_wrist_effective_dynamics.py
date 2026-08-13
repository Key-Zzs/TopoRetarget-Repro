#!/usr/bin/env python3
"""Identify the shared 6-D wrist response in the real Isaac Lab/PhysX model.

The probe is deliberately outside an execution rollout: both HO-Cap objects
are moved to the already-defined inactive scene position, a fresh reset is
used for every sample, and the result is baseline-subtracted against the same
finger-drive state.  It measures the wrist link's local response rather than
claiming whole-hand rigid-body inertia.
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
    parser.add_argument(
        "--condition",
        choices=("F0_no_finger_drives", "F1_zero_finger_targets", "F2_reference_finger_targets"),
        required=True,
        help="Run exactly one condition per Isaac Sim process.",
    )
    parser.add_argument("--clip-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--force-amplitude-n", type=float, default=0.25)
    parser.add_argument("--torque-amplitude-nm", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / ".local/reports/stage16c3_repair_c5_oracle/wrist_effective_dynamics.json",
    )
    return parser.parse_args()


def _list(value: Any) -> list[float]:
    return value.detach().cpu().tolist()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.force_amplitude_n <= 0.0 or args.torque_amplitude_nm <= 0.0:
        raise SystemExit("probe amplitudes must be positive")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.wrist_controller import (
            WristEffectiveDynamicsIdentifier,
        )

        reports: list[dict[str, object]] = []
        for condition in (args.condition,):
            cfg = IsaacWorldWristFingerDirectRLEnvCfg()
            cfg.scene.num_envs = 1
            cfg.balanced_clip_assignment = False
            if condition == "F0_no_finger_drives":
                actuator = cfg.robot.actuators["fingers"]
                actuator.stiffness = 0.0
                actuator.damping = 0.0
                actuator.effort_limit_sim = 0.0
            env = IsaacWorldWristFingerDirectRLEnv(cfg)
            try:
                env_ids = torch.arange(env.num_envs, device=env.device)
                wrist_body = torch.tensor([env._wrist_body_id], device=env.device)

                def reset_probe_state(
                    env: IsaacWorldWristFingerDirectRLEnv = env,
                    env_ids: torch.Tensor = env_ids,
                    condition: str = condition,
                ) -> torch.Tensor:
                    env._clip_index.fill_(args.clip_index)
                    env._reference_index.zero_()
                    env._reset_idx(env_ids)
                    inactive_position = env.scene.env_origins + torch.tensor(
                        [5.0, 5.0, -5.0], device=env.device
                    )
                    inactive_state = torch.cat(
                        (
                            inactive_position,
                            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(1, -1),
                            torch.zeros((1, 6), device=env.device),
                        ),
                        dim=-1,
                    )
                    env._object_170105.write_root_state_to_sim(inactive_state, env_ids=env_ids)
                    env._object_170650.write_root_state_to_sim(inactive_state, env_ids=env_ids)
                    if condition == "F2_reference_finger_targets":
                        q_ref = env.reference_bank.gather(
                            "q_finger_ref", env._clip_index, env._reference_index
                        )
                        target = env.action_adapter.canonical_to_isaac(q_ref)
                    else:
                        target = torch.zeros_like(env._robot.data.joint_pos)
                    env._robot.set_joint_position_target(target)
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.scene.update(env.physics_dt)
                    return target

                def sample(
                    wrench: torch.Tensor,
                    env: IsaacWorldWristFingerDirectRLEnv = env,
                    wrist_body: torch.Tensor = wrist_body,
                ) -> torch.Tensor:
                    target = reset_probe_state()
                    zero = torch.zeros((1, 1, 3), device=env.device)
                    before = torch.cat(
                        (env._robot.data.root_lin_vel_w, env._robot.data.root_ang_vel_w), dim=-1
                    )
                    env._robot.set_joint_position_target(target)
                    env._robot.instantaneous_wrench_composer.set_forces_and_torques(
                        forces=zero,
                        torques=zero,
                        body_ids=wrist_body,
                        is_global=True,
                    )
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    baseline = (
                        torch.cat(
                            (env._robot.data.root_lin_vel_w, env._robot.data.root_ang_vel_w), dim=-1
                        )
                        - before
                    )
                    target = reset_probe_state()
                    before = torch.cat(
                        (env._robot.data.root_lin_vel_w, env._robot.data.root_ang_vel_w), dim=-1
                    )
                    env._robot.set_joint_position_target(target)
                    env._robot.instantaneous_wrench_composer.set_forces_and_torques(
                        forces=wrench[:, None, :3],
                        torques=wrench[:, None, 3:],
                        body_ids=wrist_body,
                        is_global=True,
                    )
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    raw = (
                        torch.cat(
                            (env._robot.data.root_lin_vel_w, env._robot.data.root_ang_vel_w), dim=-1
                        )
                        - before
                    )
                    return (raw - baseline).squeeze(0)

                wrenches = []
                responses = []
                trials = []
                reset_probe_state()
                initial_root_quaternion_wxyz = env._robot.data.root_quat_w[0].clone()
                initial_root_position_world_m = env._robot.data.root_pos_w[0].clone()
                for axis in range(6):
                    amplitude = args.force_amplitude_n if axis < 3 else args.torque_amplitude_nm
                    for sign in (-1.0, 1.0):
                        wrench = torch.zeros((1, 6), device=env.device)
                        wrench[0, axis] = sign * amplitude
                        response = sample(wrench)
                        wrenches.append(wrench.squeeze(0))
                        responses.append(response)
                        trials.append(
                            {
                                "axis": axis,
                                "sign": sign,
                                "applied_wrench_world": _list(wrench.squeeze(0)),
                                "baseline_subtracted_delta_twist_world": _list(response),
                                "aligned_response": float((response[axis] * sign).item()),
                            }
                        )
                applied = torch.stack(wrenches)
                delta_twist = torch.stack(responses)
                estimate = WristEffectiveDynamicsIdentifier.estimate(
                    applied_wrench=applied, delta_twist=delta_twist, dt_s=env.physics_dt
                )
                singular = torch.linalg.svdvals(estimate.inverse_spatial_inertia)
                reports.append(
                    {
                        "condition": condition,
                        "clip": env.reference_bank.clip_ids[args.clip_index],
                        "clip_index": args.clip_index,
                        "initial_root_quaternion_wxyz": _list(initial_root_quaternion_wxyz),
                        "initial_root_position_world_m": _list(initial_root_position_world_m),
                        "status": "PASS"
                        if bool(torch.all(torch.isfinite(estimate.inverse_spatial_inertia)))
                        and float(singular.min().item()) > 1.0e-8
                        else "FAIL",
                        "dt_s": env.physics_dt,
                        "probe_amplitudes": {
                            "force_n": args.force_amplitude_n,
                            "torque_nm": args.torque_amplitude_nm,
                        },
                        "response_matrix_delta_twist_per_wrench": _list(
                            estimate.response_matrix_s_per_kg
                        ),
                        "inverse_effective_spatial_inertia": _list(
                            estimate.inverse_spatial_inertia
                        ),
                        "effective_spatial_inertia": _list(
                            torch.linalg.pinv(estimate.inverse_spatial_inertia)
                        ),
                        "condition_number": float(estimate.condition.item()),
                        "singular_values": _list(singular),
                        "residual_rms_delta_twist": _list(estimate.residual_rms),
                        "trials": trials,
                    }
                )
            finally:
                env.close()
                env.sim.clear_all_callbacks()
                env.sim.clear_instance()
        status = "PASS" if all(report["status"] == "PASS" for report in reports) else "FAIL"
        result = {
            "status": status,
            "scope": (
                "isolated, baseline-subtracted local wrist dynamics identification; "
                "objects inactive and no execution rollout"
            ),
            "input_frame": "world wrench at r_wrist link via instantaneous composer",
            "response_frame": "world root [linear_velocity, angular_velocity]",
            "conditions": reports,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if status == "PASS" else 1
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
