#!/usr/bin/env python3
"""Probe whether the authored D6 wrist reaches Isaac Lab's GPU tensor API."""

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
        default=REPO_ROOT / ".local/reports/stage16c3r2_c5/d6_tensor_contract.json",
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
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    sim = None
    try:
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sim import SimulationContext

        wrapper_path = (
            REPO_ROOT
            / ".local/generated_assets/isaaclab/wuji_hand2_beta1_d6_wrist/wujihand2_d6_wrist.usda"
        )
        if not wrapper_path.is_file():
            raise FileNotFoundError(f"C3_D6_WRAPPER_MISSING: {wrapper_path}")
        sim = SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0", gravity=(0.0, 0.0, 0.0))
        )
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/D6Hand",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(wrapper_path),
                    copy_from_source=False,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        fix_root_link=False,
                        enabled_self_collisions=False,
                    ),
                ),
                actuators={
                    "all_tensor_dofs": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=0.0,
                        damping=0.0,
                        effort_limit_sim=70.0,
                        velocity_limit_sim=6.0,
                    )
                },
            )
        )
        sim.reset()
        robot.update(sim.get_physics_dt())
        d6_ids = [index for index, name in enumerate(robot.joint_names) if "WristD6" in name]
        wrist_id = robot.body_names.index("r_wrist")
        before_joint = robot.data.joint_pos.clone()
        before_wrist = robot.data.body_link_pos_w[:, wrist_id].clone()
        target = before_joint.clone()
        if d6_ids:
            target[:, d6_ids[0]] = target[:, d6_ids[0]] + 0.01
        robot.set_joint_position_target(target)
        for _ in range(12):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
        d6_joint_delta = (
            torch.max(torch.abs(robot.data.joint_pos[:, d6_ids] - before_joint[:, d6_ids])).item()
            if d6_ids
            else 0.0
        )
        wrist_delta = torch.max(
            torch.linalg.vector_norm(robot.data.body_link_pos_w[:, wrist_id] - before_wrist, dim=-1)
        ).item()
        control_available = len(d6_ids) == 6 and d6_joint_delta > 1.0e-5 and wrist_delta > 1.0e-5
        result = {
            "status": (
                "D6_GPU_TENSOR_CONTROL_AVAILABLE"
                if control_available
                else "D6_GPU_TENSOR_CONTROL_UNAVAILABLE"
            ),
            "wrapper": str(wrapper_path),
            "implementation": "finite_d6_wrist_actuator_v1",
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "d6_tensor_joint_names": [robot.joint_names[index] for index in d6_ids],
            "d6_tensor_joint_count": len(d6_ids),
            "d6_first_axis_command": 0.01,
            "d6_joint_position_delta_max": d6_joint_delta,
            "wrist_position_delta_max_m": wrist_delta,
            "gpu_device": str(robot.device),
            "state_write_scope": "setup_and_tensor_target_only_no_rollout",
            "fallback_permitted": not control_available,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(_serialize(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(_serialize(result), sort_keys=True))
        return 0 if control_available else 2
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
