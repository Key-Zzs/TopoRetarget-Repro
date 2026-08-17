#!/usr/bin/env python3
"""Inspect the production C4 scene's per-body hand/object gravity contract."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
_BASE_HAND_ASSET = (
    REPO_ROOT
    / ".local/generated_assets/isaaclab/wuji_hand2_beta1/configuration/wujihand2_physics.usd"
)
_GRAVITY_ON_WRAPPER = (
    REPO_ROOT
    / ".local/generated_assets/isaaclab/wuji_hand2_beta1_hand_gravity_on_ablation"
    / "wujihand2_explicit_virtual_wrist.usda"
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an Isaac Sim window; GUI is the default for interactive replay.",
    )
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), default="hocap_170105")
    parser.add_argument(
        "--hand-gravity-mode",
        choices=("current_off", "ablation_on"),
        default="current_off",
        help="Use a new diagnostic-only asset when forcing gravity onto every hand body.",
    )
    parser.add_argument(
        "--static-hold-steps",
        type=int,
        default=0,
        help="Run a PPO-off fixed-reference controller hold after inspection.",
    )
    parser.add_argument(
        "--dynamic-reference",
        action="store_true",
        help="Run a PPO-off, zero-residual production reference-following rollout.",
    )
    parser.add_argument(
        "--self-collision-mode",
        choices=("production", "diagnostic_off"),
        default="production",
        help="Diagnostic A/B only; production retains the C4 self-collision contract.",
    )
    parser.add_argument(
        "--isolate-objects-at-reset",
        action="store_true",
        help="Move both task objects to the existing inactive offset once during reset setup.",
    )
    parser.add_argument(
        "--scene-variant",
        choices=("production_table", "diagnostic_base_no_table"),
        default="production_table",
        help="The base scene excludes only C4's table actors for free-space attribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New diagnostic directory; frozen C4 evidence is never modified.",
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gravity_disabled(prim: Any, physx: Any) -> bool | None:
    attr = physx.PhysxRigidBodyAPI(prim).GetDisableGravityAttr()
    # ``disableGravity`` has a schema default of false.  Its absence as an
    # authored opinion therefore means gravity is enabled, not unknown.
    return None if not attr.IsValid() else bool(attr.Get())


def _rotation_error_deg(first: Any, second: Any, torch: Any) -> float:
    dot = torch.clamp(torch.abs(torch.sum(first * second)), max=1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot)).item())


def _materialize_gravity_on_ablation() -> Path:
    """Create an ignored, explicit all-hand-gravity-on diagnostic wrapper."""

    from pxr import PhysxSchema, Usd, UsdPhysics

    from toporetarget.rl.environments.isaaclab_backend.virtual_wrist_asset import (
        write_explicit_virtual_wrist_wrapper,
    )

    manifest = write_explicit_virtual_wrist_wrapper(
        base_asset=_BASE_HAND_ASSET,
        output_usda=_GRAVITY_ON_WRAPPER,
        profile_identifier="high_authority_bounded",
    )
    stage = Usd.Stage.Open(str(_GRAVITY_ON_WRAPPER))
    if stage is None:
        raise RuntimeError("HAND_GRAVITY_ON_ABLATION_STAGE_OPEN_FAILURE")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(False)
    stage.GetRootLayer().Save()
    manifest["diagnostic_only"] = True
    manifest["hand_gravity"] = "ON_ALL_HAND_AND_VIRTUAL_BODIES"
    (_GRAVITY_ON_WRAPPER.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return _GRAVITY_ON_WRAPPER


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    app = app_launcher.app
    env = None
    try:
        import omni.usd
        import torch
        from pxr import PhysxSchema, UsdPhysics

        from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import (
            _load_start,
            _make_table_env,
        )
        from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

        clip = args.clip
        start = _load_start(clip)
        robot_usd_path = (
            None if args.hand_gravity_mode == "current_off" else _materialize_gravity_on_ablation()
        )
        self_collision_override = None if args.self_collision_mode == "production" else False
        if args.scene_variant == "production_table":
            env = _make_table_env(
                clip=clip,
                num_envs=1,
                start_index=int(start["start_index"]),
                mode=ContactRewardMode.AGGREGATE_V3,
                stage="C4",
                robot_usd_path=robot_usd_path,
                self_collision_override=self_collision_override,
            )
        else:
            from toporetarget.rl.environments.isaaclab_backend import (
                ppo26d_reference_tracking_env,
            )
            from toporetarget.rl.environments.isaaclab_backend import (
                ppo26d_reference_tracking_env_cfg as ppo_cfg,
            )

            cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
            ppo_cfg.configure_stage16d_ppo26d(
                cfg, num_envs=1, clip=clip, rsi=False, critical_dr=False
            )
            ppo_cfg.configure_stage16d_contact_reward(
                cfg,
                mode=ContactRewardMode.AGGREGATE_V3,
                reference_root=(
                    REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
                ),
                contact_reward_contract=(
                    REPO_ROOT
                    / ".local/reports/stage16d_reward_v3_pairforce_unblock"
                    / "contact_reward_contract.json"
                ),
                contact_mask_root=REPO_ROOT / ".local/reports/stage16d_reward_v3_contact",
            )
            ppo_cfg.configure_stage16_p3_p4_curriculum(
                cfg,
                curriculum_contract_path=(
                    REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
                ),
                stage="C4",
            )
            cfg.stage16d_fixed_clip = clip
            cfg.evaluation_reset_reference_indices = (int(start["start_index"]),)
            if robot_usd_path is not None:
                cfg.robot.spawn.usd_path = str(robot_usd_path)
            if self_collision_override is not None:
                cfg.robot.spawn.articulation_props.enabled_self_collisions = self_collision_override
            env = ppo26d_reference_tracking_env.IsaacPPO26DReferenceTrackingEnv(cfg)
        env.cfg.ppo26d_full_horizon_evaluation = True
        env.reset(seed=20260817)
        if args.isolate_objects_at_reset:
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
            # Setup-only physical isolation.  The ordinary production action
            # path remains the sole writer during the subsequent static/dynamic
            # diagnostic interval.
            env._object_170105.write_root_state_to_sim(inactive_state)
            env._object_170650.write_root_state_to_sim(inactive_state)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("HAND_GRAVITY_RUNTIME_STAGE_UNAVAILABLE")
        body_masses = {
            name: float(env._robot.data.default_mass[0, index].item())
            for index, name in enumerate(env._robot.body_names)
        }
        rows: list[dict[str, object]] = []
        robot_prefix = "/World/envs/env_0/Robot/"
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(robot_prefix) or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            name = path.rsplit("/", 1)[-1]
            category = "virtual_wrist" if "/VirtualLinks/" in path else "hand"
            rows.append(
                {
                    "Body": name,
                    "Path": path,
                    "Type": category,
                    "Mass": body_masses.get(name),
                    "Gravity enabled": _gravity_disabled(prim, PhysxSchema) is False,
                    "Kinematic/dynamic": "dynamic",
                    "Actuator/drive": (
                        "3P+3R virtual target chain"
                        if category == "virtual_wrist"
                        else "finger implicit position drive"
                    ),
                }
            )
        if not rows:
            raise RuntimeError("HAND_GRAVITY_MANIFEST_HAS_NO_RUNTIME_RIGID_BODIES")
        hand_gravity_enabled = any(row["Type"] == "hand" and row["Gravity enabled"] for row in rows)
        virtual_wrist_gravity_enabled = any(
            row["Type"] == "virtual_wrist" and row["Gravity enabled"] for row in rows
        )
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "hand_gravity_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _write_json(
            output / "object_gravity.json",
            {
                "world_gravity_mps2": list(env.cfg.sim.gravity),
                "object_170105_gravity_enabled": not bool(
                    env.cfg.object_170105.spawn.rigid_props.disable_gravity
                ),
                "object_170650_gravity_enabled": not bool(
                    env.cfg.object_170650.spawn.rigid_props.disable_gravity
                ),
                "table_static": args.scene_variant == "production_table",
                "table_present": args.scene_variant == "production_table",
                "table_kinematic": True,
                "hand_gravity_enabled": hand_gravity_enabled,
                "virtual_wrist_gravity_enabled": virtual_wrist_gravity_enabled,
                "requested_hand_gravity_mode": args.hand_gravity_mode,
                "self_collision_mode": args.self_collision_mode,
                "objects_isolated_at_reset": args.isolate_objects_at_reset,
                "scene_variant": args.scene_variant,
                "inspection": "live_composed_production_stage_per_rigid_body",
            },
        )
        profile = env._explicit_virtual_wrist_profile
        if profile is None:
            raise RuntimeError("HAND_GRAVITY_MANIFEST_VIRTUAL_WRIST_PROFILE_MISSING")
        _write_json(
            output / "wrist_actuator_contract.json",
            {
                "joint_order": [
                    env._robot.joint_names[index] for index in env._virtual_wrist_joint_ids
                ],
                "joint_types": [
                    "prismatic",
                    "prismatic",
                    "prismatic",
                    "revolute",
                    "revolute",
                    "revolute",
                ],
                "target_type": "position_and_velocity",
                "rotation_target_units": "radian",
                "translation_stiffness_npm": profile.translation_stiffness_npm,
                "translation_damping_ns_per_m": profile.translation_damping_ns_per_m,
                "translation_effort_limit_n": profile.translation_effort_limit_n,
                "translation_velocity_limit_mps": profile.translation_velocity_limit_mps,
                "rotation_stiffness_nm_per_rad": profile.rotation_stiffness_nm_per_rad,
                "rotation_damping_nm_s_per_rad": profile.rotation_damping_nm_s_per_rad,
                "rotation_effort_limit_nm": profile.rotation_effort_limit_nm,
                "rotation_velocity_limit_radps": profile.rotation_velocity_limit_radps,
                "hand_gravity_enabled": hand_gravity_enabled,
                "virtual_wrist_gravity_enabled": virtual_wrist_gravity_enabled,
                "self_collision_mode": args.self_collision_mode,
                "objects_isolated_at_reset": args.isolate_objects_at_reset,
                "scene_variant": args.scene_variant,
            },
        )
        if args.static_hold_steps < 0:
            raise ValueError("static hold steps must be non-negative")
        if args.static_hold_steps:
            records: list[dict[str, object]] = []
            action = torch.zeros((1, 26), device=env.device)
            for step in range(args.static_hold_steps):
                # Keep the production controller's next-key target fixed at
                # reference index one.  Only its ordinary env.step() path
                # advances PhysX; this does not write wrist or object state.
                env._reference_index.zero_()
                env._target_reference_index.zero_()
                env.step(action)
                state = env._state()
                target = env._wrist_target_quaternion[0]
                actual = state["wrist_quaternion_wxyz"][0]
                target_joint = env._explicit_wrist_joint_target[0]
                actual_joint = env._robot.data.joint_pos[0, env._virtual_wrist_joint_ids]
                finger_target = env._post_safety_finger_target_canonical[0]
                finger_actual = state["finger_q"][0]
                virtual_effort = env._robot.data.applied_torque[0, env._virtual_wrist_joint_ids]
                records.append(
                    {
                        "time_s": (step + 1) * env.step_dt,
                        "wrist_position_error_m": float(
                            torch.linalg.vector_norm(
                                env._wrist_target_position[0] - state["wrist_position_scene"][0]
                            ).item()
                        ),
                        "wrist_orientation_error_deg": _rotation_error_deg(target, actual, torch),
                        "virtual_3p_error_m": float(
                            torch.mean(torch.abs(target_joint[:3] - actual_joint[:3])).item()
                        ),
                        "virtual_3r_error_deg": float(
                            torch.rad2deg(
                                torch.mean(torch.abs(target_joint[3:] - actual_joint[3:]))
                            ).item()
                        ),
                        "virtual_3p_speed_mps": float(
                            torch.mean(
                                torch.abs(
                                    env._robot.data.joint_vel[0, env._virtual_wrist_joint_ids[:3]]
                                )
                            ).item()
                        ),
                        "virtual_3r_speed_radps": float(
                            torch.mean(
                                torch.abs(
                                    env._robot.data.joint_vel[0, env._virtual_wrist_joint_ids[3:]]
                                )
                            ).item()
                        ),
                        "finger_cmd_actual_rad": float(
                            torch.mean(torch.abs(finger_target - finger_actual)).item()
                        ),
                        "virtual_3r_effort_abs_max_nm": float(
                            torch.max(torch.abs(virtual_effort[3:])).item()
                        ),
                        "virtual_3r_effort_saturated": bool(
                            torch.any(
                                torch.abs(virtual_effort[3:])
                                >= profile.rotation_effort_limit_nm - 1.0e-6
                            ).item()
                        ),
                    }
                )
            _write_csv(output / "orientation_error_vs_time.csv", records)
            _write_json(
                output / "static_hold_summary.json",
                {
                    "mode": (
                        "CURRENT_PRODUCTION_HAND_GRAVITY_OFF"
                        if args.hand_gravity_mode == "current_off"
                        else "DIAGNOSTIC_HAND_GRAVITY_ON_ABLATION"
                    ),
                    "requested_hand_gravity_mode": args.hand_gravity_mode,
                    "ppo_optimizer_steps": 0,
                    "simulated_time_s": args.static_hold_steps * env.step_dt,
                    "wrist_rotation_error_deg_mean": float(
                        sum(float(row["wrist_orientation_error_deg"]) for row in records)
                        / len(records)
                    ),
                    "wrist_rotation_error_deg_end": records[-1]["wrist_orientation_error_deg"],
                    "wrist_position_error_m_mean": float(
                        sum(float(row["wrist_position_error_m"]) for row in records) / len(records)
                    ),
                    "wrist_position_error_m_end": records[-1]["wrist_position_error_m"],
                    "finger_cmd_actual_rad_mean": float(
                        sum(float(row["finger_cmd_actual_rad"]) for row in records) / len(records)
                    ),
                    "finger_cmd_actual_rad_end": records[-1]["finger_cmd_actual_rad"],
                    "virtual_3r_error_deg_mean": float(
                        sum(float(row["virtual_3r_error_deg"]) for row in records) / len(records)
                    ),
                    "virtual_3r_error_deg_end": records[-1]["virtual_3r_error_deg"],
                    "virtual_3r_effort_abs_max_nm": float(
                        max(float(row["virtual_3r_effort_abs_max_nm"]) for row in records)
                    ),
                    "virtual_3r_effort_saturated": any(
                        bool(row["virtual_3r_effort_saturated"]) for row in records
                    ),
                },
            )
        if args.dynamic_reference:
            env.reset(seed=20260817)
            records = []
            action = torch.zeros((1, 26), device=env.device)
            for step in range(env.reference_bank.frame_count - 1):
                _, _, terminated, timed_out, _ = env.step(action)
                state = env._state()
                index = env._target_reference_index
                wrist_reference_position = env.reference_bank.gather(
                    "wrist_pose_translation_world_ref", env._clip_index, index
                )[0]
                wrist_reference_quaternion = env.reference_bank.gather(
                    "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, index
                )[0]
                finger_reference = env.reference_bank.gather(
                    "q_finger_ref", env._clip_index, index
                )[0]
                finger_actual = env.action_adapter.isaac_to_canonical(
                    env._robot.data.joint_pos[:, env._finger_target_joint_ids]
                )[0]
                records.append(
                    {
                        "runtime_step": step,
                        "reference_index": int(index.item()),
                        "wrist_ref_cmd_position_m": float(
                            torch.linalg.vector_norm(
                                wrist_reference_position - env._wrist_target_position[0]
                            ).item()
                        ),
                        "wrist_cmd_actual_position_m": float(
                            torch.linalg.vector_norm(
                                env._wrist_target_position[0] - state["wrist_position_scene"][0]
                            ).item()
                        ),
                        "wrist_ref_actual_position_m": float(
                            torch.linalg.vector_norm(
                                wrist_reference_position - state["wrist_position_scene"][0]
                            ).item()
                        ),
                        "wrist_ref_cmd_orientation_deg": _rotation_error_deg(
                            wrist_reference_quaternion, env._wrist_target_quaternion[0], torch
                        ),
                        "wrist_cmd_actual_orientation_deg": _rotation_error_deg(
                            env._wrist_target_quaternion[0],
                            state["wrist_quaternion_wxyz"][0],
                            torch,
                        ),
                        "wrist_ref_actual_orientation_deg": _rotation_error_deg(
                            wrist_reference_quaternion,
                            state["wrist_quaternion_wxyz"][0],
                            torch,
                        ),
                        "finger_ref_cmd_rad": float(
                            torch.mean(
                                torch.abs(
                                    finger_reference - env._post_safety_finger_target_canonical[0]
                                )
                            ).item()
                        ),
                        "finger_cmd_actual_rad": float(
                            torch.mean(
                                torch.abs(
                                    env._post_safety_finger_target_canonical[0] - finger_actual
                                )
                            ).item()
                        ),
                    }
                )
                if bool(terminated[0] | timed_out[0]):
                    break
            _write_csv(output / "dynamic_reference.csv", records)
            _write_json(
                output / "dynamic_reference_summary.json",
                {
                    "mode": args.hand_gravity_mode,
                    "ppo_optimizer_steps": 0,
                    "frames": len(records),
                    "wrist_ref_cmd_orientation_deg_mean": float(
                        sum(float(row["wrist_ref_cmd_orientation_deg"]) for row in records)
                        / len(records)
                    ),
                    "wrist_cmd_actual_orientation_deg_mean": float(
                        sum(float(row["wrist_cmd_actual_orientation_deg"]) for row in records)
                        / len(records)
                    ),
                    "finger_cmd_actual_rad_mean": float(
                        sum(float(row["finger_cmd_actual_rad"]) for row in records) / len(records)
                    ),
                },
            )
        print(json.dumps({"rows": len(rows), "output": str(output)}, sort_keys=True))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
