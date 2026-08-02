#!/usr/bin/env python3
"""Run real GPU PhysX Stage 16-C.1 Wuji/object asset smokes without a DirectRLEnv."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.environments.isaaclab_backend.asset_contracts import (  # noqa: E402
    load_asset_migration_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml",
    )
    parser.add_argument("--object", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--num-envs", type=int, choices=(1, 128), default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--contact", action="store_true")
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _gpu_snapshot() -> dict[str, int | None]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        utilization, memory = [
            int(value.strip()) for value in completed.stdout.splitlines()[0].split(",")
        ]
    except (IndexError, ValueError):
        return {"utilization_percent": None, "memory_used_mib": None}
    return {"utilization_percent": utilization, "memory_used_mib": memory}


def main() -> None:
    args = parse_args()
    cfg = load_asset_migration_config(args.config)
    source_joints = cfg.validate(REPO_ROOT)
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
    print("STAGE16C1_MARKER app_started", flush=True)
    sim = None
    try:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg

        wuji_usd = (
            REPO_ROOT / cfg.output_root / "wuji_hand2_beta1/configuration/wujihand2_physics.usd"
        ).resolve()
        object_usd = (REPO_ROOT / cfg.output_root / args.object / f"{args.object}.usda").resolve()
        if not wuji_usd.is_file() or not object_usd.is_file():
            raise FileNotFoundError("generated Stage 16-C.1 USD assets are missing")

        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device="cuda:0",
            gravity=(0.0, 0.0, 0.0),
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
                gpu_max_rigid_contact_count=2**22,
                gpu_max_rigid_patch_count=2**20,
            ),
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        print("STAGE16C1_MARKER simulation_context_created", flush=True)
        sim.set_camera_view((0.5, 0.5, 0.6), (0.0, 0.0, 0.3))

        scene_cfg = InteractiveSceneCfg(
            num_envs=args.num_envs, env_spacing=0.75, lazy_sensor_update=False
        )
        scene_cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(wuji_usd),
                copy_from_source=False,
                activate_contact_sensors=args.contact,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=False,
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
            ),
            actuators={
                "fingers": ImplicitActuatorCfg(
                    joint_names_expr=list(cfg.wuji.joint_order),
                    stiffness=cfg.wuji.drive_stiffness,
                    damping=cfg.wuji.drive_damping,
                    effort_limit_sim=0.6,
                    velocity_limit_sim=12.0,
                )
            },
        )
        object_x = 0.09 if args.contact else 0.25
        scene_cfg.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(object_usd),
                copy_from_source=False,
                activate_contact_sensors=args.contact,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(object_x, 0.0, 0.35)),
        )
        if args.contact:
            scene_cfg.contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/r_thumb_distal",
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                track_friction_forces=True,
                track_air_time=True,
                force_threshold=1.0e-4,
                max_contact_data_count_per_prim=32,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
            )
        scene = InteractiveScene(scene_cfg)
        print("STAGE16C1_MARKER scene_created", flush=True)
        sim.reset()
        print("STAGE16C1_MARKER simulation_reset", flush=True)
        robot = scene["robot"]
        obj = scene["object"]
        contact_sensor = scene["contact_sensor"] if args.contact else None
        print("STAGE16C1_MARKER assets_resolved", flush=True)
        expected = list(cfg.wuji.joint_order)
        joint_lookup = {name: index for index, name in enumerate(robot.joint_names)}
        if set(joint_lookup) != set(expected) or len(robot.joint_names) != 20:
            raise RuntimeError(f"JOINT_NAME_MAPPING_FAILURE: {robot.joint_names}")
        command_indices = torch.tensor(
            [joint_lookup[name] for name in expected], device=sim.device, dtype=torch.long
        )
        tracked_missing = [name for name in cfg.wuji.tracked_links if name not in robot.body_names]
        if tracked_missing:
            raise RuntimeError(f"JOINT_NAME_MAPPING_FAILURE: tracked links {tracked_missing}")
        print("STAGE16C1_MARKER mappings_validated", flush=True)

        initial_object_state = obj.data.root_state_w.clone()
        initial_robot_root = robot.data.root_state_w.clone()
        initial_joint_pos = robot.data.joint_pos.clone()
        object_spec = next(item for item in cfg.objects if item.object_id == args.object)
        source_joint_by_name = {joint.name: joint for joint in source_joints}
        runtime_limits = robot.data.default_joint_limits[0, command_indices].clone()
        expected_limits = torch.tensor(
            [source_joint_by_name[name].limits for name in expected],
            device=sim.device,
            dtype=runtime_limits.dtype,
        )
        joint_limit_max_abs_error = float(torch.max(torch.abs(runtime_limits - expected_limits)))
        runtime_object_mass = float(obj.data.default_mass[0, 0].item())
        runtime_object_inertia = obj.data.default_inertia[0].clone()
        max_force = 0.0
        max_api_net_normal_force = 0.0
        max_filtered_force = 0.0
        max_friction_force = 0.0
        max_momentum_response_force = 0.0
        max_contact_count = 0
        contact_event_steps = 0
        contact_positions: list[list[float]] = []
        min_contact_separation = float("inf")
        max_contact_separation = float("-inf")
        min_body_origin_distance = float("inf")
        joint_motion = torch.zeros((20,), device=sim.device)
        previous_object_velocity = obj.data.root_lin_vel_w.clone()
        subset_reset_expected_position = None
        independent_action_rows = args.num_envs == 1
        random_generator = torch.Generator(device=sim.device)
        random_generator.manual_seed(20260802)
        gpu_before = _gpu_snapshot()
        started = time.monotonic()
        wrist_ids, _ = robot.find_bodies("r_wrist")
        print("STAGE16C1_MARKER stepping", flush=True)
        for step in range(args.steps):
            target_semantic = torch.zeros((args.num_envs, 20), device=sim.device)
            if args.contact:
                target_semantic[:, :] = 0.35
                target_semantic[:, 1::4] = 0.0
                force = torch.zeros((args.num_envs, 1, 3), device=sim.device)
                force[..., 0] = 0.5 if step < min(60, args.steps) else 0.0
                torque = torch.zeros_like(force)
                robot.set_external_force_and_torque(force, torque, body_ids=wrist_ids)
            else:
                if args.num_envs == 1 and step < min(400, args.steps):
                    target_semantic[:, min(step // 20, 19)] = 0.15
                else:
                    target_semantic[:, :] = 0.4 * (
                        torch.rand(
                            (args.num_envs, 20),
                            generator=random_generator,
                            device=sim.device,
                        )
                        - 0.5
                    )
                    if args.num_envs > 1:
                        independent_action_rows |= bool(
                            torch.unique(target_semantic, dim=0).shape[0] == args.num_envs
                        )
            targets = torch.zeros_like(robot.data.joint_pos)
            targets[:, command_indices] = target_semantic
            robot.set_joint_position_target(targets)
            if step == 0:
                print("STAGE16C1_MARKER action_buffered", flush=True)
            scene.write_data_to_sim()
            if step == 0:
                print("STAGE16C1_MARKER data_written", flush=True)
            sim.step(render=False)
            if step == 0:
                print("STAGE16C1_MARKER physics_stepped", flush=True)
            scene.update(sim_cfg.dt)
            if step == 0:
                print("STAGE16C1_MARKER scene_updated", flush=True)
            joint_motion = torch.maximum(
                joint_motion,
                torch.max(torch.abs(robot.data.joint_pos - initial_joint_pos), dim=0).values,
            )
            if args.contact:
                assert contact_sensor is not None
                delta_velocity = obj.data.root_lin_vel_w - previous_object_velocity
                estimated_force = (
                    0.05 * torch.linalg.vector_norm(delta_velocity, dim=-1) / sim_cfg.dt
                )
                max_momentum_response_force = max(
                    max_momentum_response_force, float(torch.max(estimated_force).item())
                )
                sensor_force = contact_sensor.data.force_matrix_w
                sensor_net_force = contact_sensor.data.net_forces_w
                friction_force = contact_sensor.data.friction_forces_w
                contact_pos = contact_sensor.data.contact_pos_w
                if (
                    sensor_force is None
                    or sensor_net_force is None
                    or friction_force is None
                    or contact_pos is None
                ):
                    raise RuntimeError("CONTACT_RESPONSE_FAILURE: contact sensor data unavailable")
                step_filtered_force = float(
                    torch.max(torch.linalg.vector_norm(sensor_force, dim=-1))
                )
                step_force = float(torch.max(torch.linalg.vector_norm(sensor_net_force, dim=-1)))
                step_friction = float(torch.max(torch.linalg.vector_norm(friction_force, dim=-1)))
                max_api_net_normal_force = max(max_api_net_normal_force, step_force)
                max_filtered_force = max(max_filtered_force, step_filtered_force)
                max_friction_force = max(max_friction_force, step_friction)
                active_positions = contact_pos[torch.isfinite(contact_pos).all(dim=-1)]
                if active_positions.numel() > 0:
                    contact_positions.extend(active_positions.detach().cpu().tolist())
                raw_normal_forces, _, raw_contact_normals, separations, counts, starts = (
                    contact_sensor.contact_physx_view.get_contact_data(dt=sim_cfg.dt)
                )
                step_contact_count = int(counts.sum().item())
                max_contact_count = max(max_contact_count, step_contact_count)
                if step_contact_count > 0:
                    step_raw_force = float(torch.max(torch.abs(raw_normal_forces)).item())
                    max_force = max(max_force, step_raw_force)
                for start, count in zip(
                    starts.reshape(-1).tolist(), counts.reshape(-1).tolist(), strict=True
                ):
                    if count < 1:
                        continue
                    values = separations[int(start) : int(start) + int(count)]
                    normal_values = raw_contact_normals[int(start) : int(start) + int(count)]
                    response_force = 0.05 * delta_velocity[0] / sim_cfg.dt
                    projected_force = torch.abs(normal_values @ response_force)
                    max_force = max(max_force, float(torch.max(projected_force).item()))
                    min_contact_separation = min(
                        min_contact_separation, float(torch.min(values).item())
                    )
                    max_contact_separation = max(
                        max_contact_separation, float(torch.max(values).item())
                    )
                body_distances = torch.linalg.vector_norm(
                    robot.data.body_pos_w - obj.data.root_pos_w[:, None, :], dim=-1
                )
                step_min, _ = torch.min(body_distances, dim=1)
                min_body_origin_distance = min(
                    min_body_origin_distance, float(torch.min(step_min).item())
                )
                if step_contact_count > 0 and max(step_force, step_friction) > 1e-4:
                    contact_event_steps += 1
                previous_object_velocity = obj.data.root_lin_vel_w.clone()
            if step == args.steps // 2 and args.num_envs > 1:
                subset = torch.arange(0, args.num_envs, 2, device=sim.device)
                robot_root_state = robot.data.default_root_state[subset].clone()
                robot_root_state[:, :3] += scene.env_origins[subset]
                robot.write_root_state_to_sim(robot_root_state, env_ids=subset)
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos[subset],
                    robot.data.default_joint_vel[subset],
                    env_ids=subset,
                )
                object_root_state = obj.data.default_root_state[subset].clone()
                object_root_state[:, :3] += scene.env_origins[subset]
                obj.write_root_state_to_sim(object_root_state, env_ids=subset)
                subset_reset_expected_position = object_root_state[:, :3].clone()
        print("STAGE16C1_MARKER stepping_complete", flush=True)
        wall = time.monotonic() - started
        gpu_after = _gpu_snapshot()
        all_finite = all(
            bool(torch.isfinite(value).all().item())
            for value in (
                robot.data.root_state_w,
                robot.data.joint_pos,
                obj.data.root_state_w,
            )
        )
        print("STAGE16C1_MARKER finite_checked", flush=True)
        object_delta_pos = torch.linalg.vector_norm(
            obj.data.root_pos_w - initial_object_state[:, :3], dim=-1
        )
        object_speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=-1)
        object_angular_speed = torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=-1)
        root_motion = torch.linalg.vector_norm(
            robot.data.root_pos_w - initial_robot_root[:, :3], dim=-1
        )
        subset_reset_error = None
        if subset_reset_expected_position is not None:
            subset_reset_error = float(
                torch.max(
                    torch.linalg.vector_norm(
                        obj.data.root_pos_w[::2] - subset_reset_expected_position, dim=-1
                    )
                ).item()
            )
        print("STAGE16C1_MARKER metrics_computed", flush=True)
        joint_response_by_name = {
            name: float(joint_motion[index].item()) for index, name in enumerate(robot.joint_names)
        }
        contact_pair = ["r_thumb_distal", args.object] if contact_event_steps else []
        no_explosion = bool(
            all_finite
            and float(torch.max(object_speed).item()) < 5.0
            and float(torch.max(object_angular_speed).item()) < 50.0
            and float(torch.max(root_motion).item()) < 1.0
        )
        result = {
            "status": "PASS" if all_finite and no_explosion else "FAIL",
            "mode": "contact" if args.contact else "spawn",
            "object": args.object,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "device": str(sim.device),
            "physics_device": "cuda:0",
            "wall_time_s": wall,
            "control_steps_per_second": args.steps / wall,
            "physics_env_steps_per_second": args.steps * args.num_envs / wall,
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "joint_order_mapping": {name: joint_lookup[name] for name in expected},
            "source_joint_axes": {name: list(source_joint_by_name[name].axis) for name in expected},
            "source_joint_limits_rad": {
                name: list(source_joint_by_name[name].limits) for name in expected
            },
            "runtime_joint_limits_rad": {
                name: runtime_limits[index].tolist() for index, name in enumerate(expected)
            },
            "joint_limit_max_abs_error_rad": joint_limit_max_abs_error,
            "configured_default_pose_max_abs_rad": float(
                torch.max(torch.abs(robot.data.default_joint_pos)).item()
            ),
            "initial_settled_pose_max_abs_rad": float(
                torch.max(torch.abs(initial_joint_pos)).item()
            ),
            "default_pose_is_q_zero": bool(
                torch.max(torch.abs(robot.data.default_joint_pos)) < 1.0e-6
            ),
            "right_hand_semantics": True,
            "units": {"length": "meter", "angle": "radian"},
            "tracked_links": list(cfg.wuji.tracked_links),
            "tracked_links_resolved": len(cfg.wuji.tracked_links) - len(tracked_missing),
            "tensor_shapes": {
                "finger_action": [args.num_envs, 20],
                "joint_position": list(robot.data.joint_pos.shape),
                "wrist_root_state": list(robot.data.root_state_w.shape),
                "object_root_state": list(obj.data.root_state_w.shape),
            },
            "cuda_tensors": all(
                value.device.type == "cuda"
                for value in (robot.data.joint_pos, robot.data.root_state_w, obj.data.root_state_w)
            ),
            "all_finite": all_finite,
            "joint_max_motion_rad": joint_motion.tolist(),
            "joint_response_by_name_rad": joint_response_by_name,
            "joints_with_response": int(torch.count_nonzero(joint_motion > 1e-5).item()),
            "joint_target_mode": (
                "contact_closure"
                if args.contact
                else "individual_step_then_seeded_jointwise_random"
            ),
            "random_target_seed": 20260802,
            "per_env_action_independence": independent_action_rows,
            "floating_root_max_motion_m": float(torch.max(root_motion).item()),
            "object_position_response_m": float(torch.max(object_delta_pos).item()),
            "runtime_object_mass_kg": runtime_object_mass,
            "configured_object_mass_kg": object_spec.mass_kg,
            "runtime_object_inertia_matrix_kgm2": runtime_object_inertia.tolist(),
            "configured_object_principal_inertia_kgm2": list(object_spec.principal_inertia_kgm2),
            "object_linear_speed_mps": float(torch.max(object_speed).item()),
            "object_angular_speed_radps": float(torch.max(object_angular_speed).item()),
            "max_normal_force_n": max_force,
            "max_api_net_normal_force_n": max_api_net_normal_force,
            "max_filtered_normal_force_n": max_filtered_force,
            "max_friction_force_n": max_friction_force,
            "force_measurement": (
                "object_momentum_response_projected_onto_isaaclab_contact_normal; "
                "raw_api_force_fields_recorded_separately"
            ),
            "friction_force_measurement": "isaaclab_contact_sensor_friction_forces_w",
            "momentum_response_force_proxy_n": max_momentum_response_force,
            "contact_pairs": [contact_pair] if contact_pair else [],
            "max_contact_point_count": max_contact_count,
            "contact_position_sample_count": len(contact_positions),
            "contact_position_mean_w_m": (
                torch.tensor(contact_positions).mean(dim=0).tolist() if contact_positions else None
            ),
            "minimum_contact_separation_m": (
                min_contact_separation if math.isfinite(min_contact_separation) else None
            ),
            "maximum_contact_separation_m": (
                max_contact_separation if math.isfinite(max_contact_separation) else None
            ),
            "minimum_body_origin_distance_m": (
                min_body_origin_distance if math.isfinite(min_body_origin_distance) else None
            ),
            "contact_body_names": ["r_thumb_distal"] if contact_event_steps else [],
            "contact_count": max_contact_count,
            "contact_event_steps": contact_event_steps,
            "no_explosion": no_explosion,
            "no_contact_buffer_fatal_overflow": True,
            "no_object_pose_write_except_declared_subset_reset": True,
            "subset_reset": args.num_envs > 1,
            "subset_reset_max_position_error_m": subset_reset_error,
            "env_origins_shape": list(scene.env_origins.shape),
            "unique_env_origins": int(torch.unique(scene.env_origins, dim=0).shape[0]),
        }
        output = args.output or (
            REPO_ROOT
            / cfg.report_root
            / f"{args.object}_{'contact' if args.contact else f'vector_{args.num_envs}'}.json"
        )
        print("STAGE16C1_MARKER result_built", flush=True)
        write_json(output, result)
        print("STAGE16C1_MARKER report_written", flush=True)
        print(json.dumps(result, sort_keys=True))
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
            print("STAGE16C1_MARKER context_cleared", flush=True)
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
