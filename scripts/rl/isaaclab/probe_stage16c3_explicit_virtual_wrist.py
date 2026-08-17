#!/usr/bin/env python3
"""Probe the explicit virtual 3P+3R wrist through real GPU articulation tensors."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_FINGER_JOINT_ORDER = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an Isaac Sim window; GUI is the default for interactive replay.",
    )
    parser.add_argument("--num-envs", type=int, choices=(1, 128), default=1)
    parser.add_argument("--steps-per-axis", type=int, default=120)
    parser.add_argument(
        "--world-gravity-z",
        type=float,
        default=0.0,
        help="Diagnostic world gravity; production C4 uses -9.81 while all hand bodies opt out.",
    )
    parser.add_argument(
        "--force-hand-gravity-off",
        action="store_true",
        help="Apply Isaac Lab's runtime rigid-body gravity override to the imported articulation.",
    )
    parser.add_argument(
        "--profile",
        choices=("conservative", "nominal", "high_authority_bounded"),
        default="high_authority_bounded",
        help="Production uses high_authority_bounded; other profiles are diagnostics only.",
    )
    parser.add_argument(
        "--rotation-angles-deg",
        nargs="+",
        type=float,
        default=(5.0, 15.0, 30.0),
        help="Positive magnitudes for the required +/- single-axis 3R probes.",
    )
    parser.add_argument(
        "--mixed-target-deg",
        nargs=3,
        type=float,
        metavar=("RX", "RY", "RZ"),
        help="One full-chain 3R static target, in radians-native joint degrees.",
    )
    parser.add_argument(
        "--finger-target-trace",
        type=Path,
        help="Initialize/hold its frame-zero canonical finger q/qd in the mixed static test.",
    )
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

    app = AppLauncher(headless=args.headless).app
    sim = None
    try:
        import isaaclab.sim as sim_utils
        import numpy as np
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
        if not args.rotation_angles_deg or any(value <= 0.0 for value in args.rotation_angles_deg):
            raise ValueError("rotation angles must be positive")
        profile = next(item for item in D6_WRIST_PROFILES if item.identifier == args.profile)
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / 120.0,
                device="cuda:0",
                gravity=(0.0, 0.0, args.world_gravity_z),
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
                rigid_props=(
                    sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
                    if args.force_hand_gravity_off
                    else None
                ),
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
        # The six-DoF inventory smoke above is intentionally retained.  The
        # following cases are the controller decision tree's production-path
        # single-axis tests: hand/object/table are absent from this scene and
        # each explicit 3R actuator is driven independently through the same
        # Isaac Lab position/velocity target API used by the environment.
        rotation_results = []
        for axis_index, joint_name in enumerate(expected[3:], start=3):
            for angle_deg in args.rotation_angles_deg:
                for sign in (-1.0, 1.0):
                    zero_position = torch.zeros_like(robot.data.joint_pos)
                    zero_velocity = torch.zeros_like(robot.data.joint_vel)
                    robot.write_joint_state_to_sim(zero_position, zero_velocity)
                    robot.set_joint_position_target(zero_position)
                    robot.set_joint_velocity_target(zero_velocity)
                    robot.reset()
                    scene.write_data_to_sim()
                    sim.forward()
                    scene.update(sim.get_physics_dt())
                    before_quaternion = robot.data.body_link_quat_w[:, wrist_id].clone()
                    target = zero_position.clone()
                    target_value = sign * math.radians(angle_deg)
                    target[:, virtual_ids[axis_index]] = target_value
                    robot.set_joint_position_target(target)
                    robot.set_joint_velocity_target(zero_velocity)
                    finite = True
                    for _ in range(args.steps_per_axis):
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(sim.get_physics_dt())
                        finite = finite and bool(torch.isfinite(robot.data.joint_pos).all())
                    actual_q = robot.data.joint_pos[:, virtual_ids]
                    q_error_deg = torch.rad2deg(torch.abs(actual_q[:, axis_index] - target_value))
                    wrist_rotation_delta = quaternion_geodesic(
                        before_quaternion, robot.data.body_link_quat_w[:, wrist_id]
                    )
                    drive_effort = robot.data.applied_torque[:, virtual_ids[axis_index]]
                    result = {
                        "joint": joint_name,
                        "axis": ("X", "Y", "Z")[axis_index - 3],
                        "target_deg": target_value * 180.0 / math.pi,
                        "actual_deg_min": float(
                            torch.rad2deg(actual_q[:, axis_index]).min().item()
                        ),
                        "actual_deg_max": float(
                            torch.rad2deg(actual_q[:, axis_index]).max().item()
                        ),
                        "joint_error_deg_max": float(q_error_deg.max().item()),
                        "cartesian_rotation_delta_deg_min": float(
                            torch.rad2deg(wrist_rotation_delta).min().item()
                        ),
                        "cartesian_rotation_delta_deg_max": float(
                            torch.rad2deg(wrist_rotation_delta).max().item()
                        ),
                        "effort_abs_max_nm": float(torch.abs(drive_effort).max().item()),
                        "effort_saturated": bool(
                            torch.any(
                                torch.abs(drive_effort) >= profile.rotation_effort_limit_nm - 1.0e-6
                            ).item()
                        ),
                        "finite": finite,
                    }
                    actual_sign = actual_q[:, axis_index] * target_value > 0.0
                    result["axis_gate_pass"] = bool(
                        finite
                        and bool(actual_sign.all())
                        and result["joint_error_deg_max"] <= 2.0
                        and result["cartesian_rotation_delta_deg_min"] >= abs(target_value) * 0.75
                    )
                    rotation_results.append(result)
        mixed_result = None
        if args.mixed_target_deg is not None:
            zero_position = torch.zeros_like(robot.data.joint_pos)
            zero_velocity = torch.zeros_like(robot.data.joint_vel)
            target = zero_position.clone()
            target_radians = (
                torch.tensor(args.mixed_target_deg, dtype=target.dtype, device=target.device)
                * math.pi
                / 180.0
            )
            target[:, virtual_ids[3:]] = target_radians
            initial_position = target.clone()
            initial_velocity = zero_velocity.clone()
            finger_ids = [robot.joint_names.index(name) for name in _FINGER_JOINT_ORDER]
            finger_velocity_target = zero_velocity.clone()
            if args.finger_target_trace is not None:
                with np.load(args.finger_target_trace, allow_pickle=False) as trace:
                    canonical_position = np.asarray(trace["finger_target_q"][0], dtype=np.float32)
                    canonical_velocity = np.asarray(trace["finger_qdot"][0], dtype=np.float32)
                for canonical_index, joint_id in enumerate(finger_ids):
                    initial_position[:, joint_id] = float(canonical_position[canonical_index])
                    initial_velocity[:, joint_id] = float(canonical_velocity[canonical_index])
                    finger_velocity_target[:, joint_id] = float(canonical_velocity[canonical_index])
            robot.write_joint_state_to_sim(initial_position, initial_velocity)
            robot.set_joint_position_target(initial_position)
            robot.set_joint_velocity_target(initial_velocity)
            robot.reset()
            scene.write_data_to_sim()
            sim.forward()
            scene.update(sim.get_physics_dt())
            robot.set_joint_position_target(target)
            robot.set_joint_velocity_target(zero_velocity[:, virtual_ids], joint_ids=virtual_ids)
            if args.finger_target_trace is not None:
                robot.set_joint_position_target(target[:, finger_ids], joint_ids=finger_ids)
                robot.set_joint_velocity_target(
                    finger_velocity_target[:, finger_ids], joint_ids=finger_ids
                )
            finite = True
            for _ in range(args.steps_per_axis):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                finite = finite and bool(torch.isfinite(robot.data.joint_pos).all())
            actual_q = robot.data.joint_pos[:, virtual_ids[3:]]
            q_error_deg = torch.rad2deg(torch.abs(actual_q - target_radians))
            drive_effort = robot.data.applied_torque[:, virtual_ids[3:]]
            mixed_result = {
                "target_deg": list(args.mixed_target_deg),
                "finger_state": (
                    "c4_frame_zero_trace" if args.finger_target_trace is not None else "zero"
                ),
                "actual_deg": torch.rad2deg(actual_q[0]).detach().cpu().tolist(),
                "joint_error_deg_max": float(q_error_deg.max().item()),
                "effort_abs_max_nm": float(torch.abs(drive_effort).max().item()),
                "effort_saturated": bool(
                    torch.any(
                        torch.abs(drive_effort) >= profile.rotation_effort_limit_nm - 1.0e-6
                    ).item()
                ),
                "finite": finite,
            }
            mixed_result["static_gate_pass"] = bool(
                finite and mixed_result["joint_error_deg_max"] <= 2.0
            )
        exact_inventory = (
            len(robot.joint_names) == 26
            and [name for name in robot.joint_names if name.startswith("virtual_")] == expected
        )
        passed = (
            exact_inventory
            and all(item["axis_gate_pass"] for item in axis_results)
            and all(item["axis_gate_pass"] for item in rotation_results)
            and (mixed_result is None or mixed_result["static_gate_pass"])
        )
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
            "profile": args.profile,
            "world_gravity_mps2": [0.0, 0.0, args.world_gravity_z],
            "runtime_hand_gravity_override": args.force_hand_gravity_off,
            "device": str(robot.device),
            "wrapper": str(wrapper),
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "virtual_joint_order": expected,
            "virtual_joint_ids": virtual_ids,
            "exact_26dof_inventory": exact_inventory,
            "axis_results": axis_results,
            "rotation_single_axis_results": rotation_results,
            "mixed_rotation_static_result": mixed_result,
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
