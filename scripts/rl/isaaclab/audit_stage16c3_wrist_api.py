#!/usr/bin/env python3
"""Audit the live PhysX wrist force/torque convention with isolated 6-D probes."""

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
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3_repair_c5_oracle/wrist_api_conventions.json",
    )
    return parser.parse_args()


def _value(tensor: Any) -> list[float]:
    return tensor.detach().cpu().tolist()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
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
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260803)
        env_ids = torch.arange(env.num_envs, device=env.device)
        wrist_body = torch.tensor([env._wrist_body_id], device=env.device)

        def reset_isolated_state() -> None:
            env._clip_index.zero_()
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
            # This is an isolated API diagnostic, not an execution rollout.
            env._object_170105.write_root_state_to_sim(inactive_state, env_ids=env_ids)
            env._object_170650.write_root_state_to_sim(inactive_state, env_ids=env_ids)
            env._robot.set_joint_position_target(env._robot.data.joint_pos.clone())
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)

        def one_substep(
            force: torch.Tensor, torque: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            before_linear = env._robot.data.root_lin_vel_w.clone()
            before_angular = env._robot.data.root_ang_vel_w.clone()
            env._robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=force,
                torques=torque,
                body_ids=wrist_body,
                is_global=True,
            )
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            return (
                env._robot.data.root_lin_vel_w - before_linear,
                env._robot.data.root_ang_vel_w - before_angular,
            )

        probes: list[dict[str, object]] = []
        for wrench_type, component_name in (("force", "linear"), ("torque", "angular")):
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    reset_isolated_state()
                    zero_force = torch.zeros((1, 1, 3), device=env.device)
                    baseline_linear, baseline_angular = one_substep(zero_force, zero_force)
                    reset_isolated_state()
                    force = torch.zeros((1, 1, 3), device=env.device)
                    torque = torch.zeros_like(force)
                    target = force if wrench_type == "force" else torque
                    target[..., axis] = sign
                    raw_linear, raw_angular = one_substep(force, torque)
                    delta_linear = raw_linear - baseline_linear
                    delta_angular = raw_angular - baseline_angular
                    response = delta_linear if component_name == "linear" else delta_angular
                    axis_direction = torch.zeros(3, device=env.device)
                    axis_direction[axis] = sign
                    projection = float((response[0] @ axis_direction).item())
                    probes.append(
                        {
                            "input": f"{sign:+.0f}{'F' if wrench_type == 'force' else 'T'}{axis}",
                            "input_frame": "world",
                            "response_frame": "root world velocity",
                            "delta_linear_velocity_world_mps": _value(delta_linear[0]),
                            "delta_angular_velocity_world_radps": _value(delta_angular[0]),
                            "baseline_linear_velocity_delta_world_mps": _value(baseline_linear[0]),
                            "baseline_angular_velocity_delta_world_radps": _value(
                                baseline_angular[0]
                            ),
                            "axis_projection": projection,
                            "expected_positive": projection > 0.0,
                        }
                    )
        result = {
            "status": "PASS" if all(item["expected_positive"] for item in probes) else "FAIL",
            "root_quaternion_order": "wxyz",
            "root_linear_velocity_frame": "world",
            "root_angular_velocity_frame": "world",
            "external_wrench_input_frame": "world via instantaneous_wrench_composer",
            "wrench_application_body": (
                "r_wrist at its link origin; torque is about the body COM API"
            ),
            "per_substep_rule": (
                "instantaneous composer is set before every sim step and reset by write_data_to_sim"
            ),
            "object_and_finger_scope": (
                "objects moved to diagnostic non-contact location; no execution rollout"
            ),
            "probes": probes,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
