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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--static-hold-steps",
        type=int,
        default=0,
        help="Run a PPO-off fixed-reference controller hold after inspection.",
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


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
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

        clip = "hocap_170105"
        start = _load_start(clip)
        env = _make_table_env(
            clip=clip,
            num_envs=1,
            start_index=int(start["start_index"]),
            mode=ContactRewardMode.AGGREGATE_V3,
            stage="C4",
        )
        env.cfg.ppo26d_full_horizon_evaluation = True
        env.reset(seed=20260817)
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
                "table_static": True,
                "table_kinematic": True,
                "hand_gravity_enabled": hand_gravity_enabled,
                "virtual_wrist_gravity_enabled": virtual_wrist_gravity_enabled,
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
                    }
                )
            _write_csv(output / "orientation_error_vs_time.csv", records)
            _write_json(
                output / "static_hold_summary.json",
                {
                    "mode": "CURRENT_PRODUCTION_HAND_GRAVITY_OFF",
                    "ppo_optimizer_steps": 0,
                    "simulated_time_s": args.static_hold_steps * env.step_dt,
                    "wrist_rotation_error_deg_mean": float(
                        sum(float(row["wrist_orientation_error_deg"]) for row in records)
                        / len(records)
                    ),
                    "wrist_rotation_error_deg_end": records[-1]["wrist_orientation_error_deg"],
                    "virtual_3r_error_deg_mean": float(
                        sum(float(row["virtual_3r_error_deg"]) for row in records) / len(records)
                    ),
                    "virtual_3r_error_deg_end": records[-1]["virtual_3r_error_deg"],
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
