#!/usr/bin/env python3
"""Probe the explicit virtual 3P+3R wrist through real GPU articulation tensors."""

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
    parser.add_argument("--num-envs", type=int, choices=(1, 128), default=1)
    parser.add_argument("--steps-per-axis", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _serialize(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.steps_per_axis < 1:
        raise SystemExit("--steps-per-axis must be positive")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    sim = None
    try:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

        from toporetarget.rl.environments.isaaclab_backend.d6_wrist_asset import (
            D6_WRIST_PROFILES,
        )
        from toporetarget.rl.environments.isaaclab_backend.explicit_virtual_wrist import (
            EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER,
        )
        from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
            quaternion_geodesic,
        )

        wrapper = (
            REPO_ROOT
            / ".local/generated_assets/isaaclab/wuji_hand2_beta1_explicit_virtual_wrist"
            / "wujihand2_explicit_virtual_wrist.usda"
        ).resolve()
        if not wrapper.is_file():
            raise FileNotFoundError(f"C3_EXPLICIT_VIRTUAL_WRIST_MISSING: {wrapper}")
        profile = next(item for item in D6_WRIST_PROFILES if item.identifier == "nominal")
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / 120.0,
                device="cuda:0",
                gravity=(0.0, 0.0, 0.0),
                physx=sim_utils.PhysxCfg(
                    solver_type=1,
                    min_position_iteration_count=4,
                    max_position_iteration_count=8,
                    min_velocity_iteration_count=1,
                    max_velocity_iteration_count=2,
                ),
            )
        )
        scene_cfg = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=0.75)
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(wrapper),
                copy_from_source=False,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
            ),
            actuators={
                "virtual_translation": ImplicitActuatorCfg(
                    joint_names_expr=list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[:3]),
                    stiffness=profile.translation_stiffness_npm,
                    damping=profile.translation_damping_ns_per_m,
                    effort_limit_sim=profile.translation_effort_limit_n,
                    velocity_limit_sim=profile.translation_velocity_limit_mps,
                ),
                "virtual_rotation": ImplicitActuatorCfg(
                    joint_names_expr=list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[3:]),
                    stiffness=profile.rotation_stiffness_nm_per_rad,
                    damping=profile.rotation_damping_nm_s_per_rad,
                    effort_limit_sim=profile.rotation_effort_limit_nm,
                    velocity_limit_sim=profile.rotation_velocity_limit_radps,
                ),
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=["r_.*"],
                    stiffness=4.0,
                    damping=0.2,
                    effort_limit_sim=0.6,
                    velocity_limit_sim=12.0,
                ),
            },
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        expected = list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER)
        virtual_ids = [robot.joint_names.index(name) for name in expected]
        wrist_id = robot.body_names.index("r_wrist")
        anchor_id = robot.body_names.index("VirtualWristAnchor")
        axis_results = []
        command_values = [0.02, 0.02, 0.02, 0.10, 0.10, 0.10]
        for axis_index, (joint_name, command) in enumerate(
            zip(expected, command_values, strict=True)
        ):
            zero_position = torch.zeros_like(robot.data.joint_pos)
            zero_velocity = torch.zeros_like(robot.data.joint_vel)
            robot.write_joint_state_to_sim(zero_position, zero_velocity)
            robot.set_joint_position_target(zero_position)
            robot.set_joint_velocity_target(zero_velocity)
            robot.reset()
            scene.write_data_to_sim()
            sim.forward()
            scene.update(sim.get_physics_dt())
            before_joint = robot.data.joint_pos[:, virtual_ids].clone()
            before_wrist_position = robot.data.body_link_pos_w[:, wrist_id].clone()
            before_wrist_quaternion = robot.data.body_link_quat_w[:, wrist_id].clone()
            before_anchor_position = robot.data.body_link_pos_w[:, anchor_id].clone()
            target = zero_position.clone()
            target[:, virtual_ids[axis_index]] = command
            robot.set_joint_position_target(target)
            robot.set_joint_velocity_target(zero_velocity)
            finite = True
            for _ in range(args.steps_per_axis):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                finite = finite and bool(torch.isfinite(robot.data.joint_pos).all())
            joint_delta = robot.data.joint_pos[:, virtual_ids] - before_joint
            commanded_delta = joint_delta[:, axis_index]
            wrist_position_delta = torch.linalg.vector_norm(
                robot.data.body_link_pos_w[:, wrist_id] - before_wrist_position, dim=-1
            )
            wrist_rotation_delta = quaternion_geodesic(
                before_wrist_quaternion, robot.data.body_link_quat_w[:, wrist_id]
            )
            anchor_delta = torch.linalg.vector_norm(
                robot.data.body_link_pos_w[:, anchor_id] - before_anchor_position, dim=-1
            )
            response = {
                "joint": joint_name,
                "command": command,
                "joint_delta_min": float(commanded_delta.min().item()),
                "joint_delta_max": float(commanded_delta.max().item()),
                "command_direction_correct": bool((commanded_delta * command > 0.0).all()),
                "wrist_position_delta_max_m": float(wrist_position_delta.max().item()),
                "wrist_rotation_delta_max_rad": float(wrist_rotation_delta.max().item()),
                "anchor_position_delta_max_m": float(anchor_delta.max().item()),
                "finite": finite,
            }
            is_translation = axis_index < 3
            response["axis_gate_pass"] = bool(
                finite
                and response["command_direction_correct"]
                and abs(response["joint_delta_min"]) > 1.0e-4
                and response["anchor_position_delta_max_m"] <= 1.0e-6
                and (
                    response["wrist_position_delta_max_m"] > 1.0e-4
                    if is_translation
                    else response["wrist_rotation_delta_max_rad"] > 1.0e-4
                )
            )
            axis_results.append(response)
        exact_inventory = (
            len(robot.joint_names) == 26
            and [name for name in robot.joint_names if name.startswith("virtual_")] == expected
        )
        passed = exact_inventory and all(item["axis_gate_pass"] for item in axis_results)
        result = {
            "status": (
                "C3_EXPLICIT_VIRTUAL_WRIST_GPU_TENSOR_VALIDATED"
                if passed
                else "C3_EXPLICIT_VIRTUAL_WRIST_GPU_TENSOR_BLOCKED"
            ),
            "implementation": "finite_virtual_6d_wrist_actuator_v1",
            "articulation_model": "explicit_serial_3p3r",
            "engineering_model": "abstract_6dof_wrist_not_real_arm",
            "num_envs": args.num_envs,
            "device": str(robot.device),
            "wrapper": str(wrapper),
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "virtual_joint_order": expected,
            "virtual_joint_ids": virtual_ids,
            "exact_26dof_inventory": exact_inventory,
            "axis_results": axis_results,
            "rollout_root_state_writes": 0,
            "rollout_object_state_writes": 0,
            "real_arm_present": False,
        }
        output = args.output or (
            REPO_ROOT
            / ".local/reports/stage16c3r2_c5"
            / f"explicit_virtual_wrist_micro_{args.num_envs}env.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_serialize(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(_serialize(result), sort_keys=True))
        return 0 if passed else 2
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
