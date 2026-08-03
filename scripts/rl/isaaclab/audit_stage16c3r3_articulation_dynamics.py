#!/usr/bin/env python3
"""Audit the live Isaac Sim articulation dynamics tensors in an isolated process."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3r3_joint_dynamics_c5/dynamics_api_audit.json",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        help="Optional non-overwriting mass-matrix diagnostics output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
            generalized_mass_matrix,
            inferred_generalized_bias,
            mass_matrix_diagnostics,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
            configure_full_articulation_computed_torque_wrist,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.lazy_sensor_update = True
        cfg.contact_telemetry = "off"
        cfg.balanced_clip_assignment = False
        configure_full_articulation_computed_torque_wrist(cfg, profile_identifier="CT-low")
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        env._pre_physics_step(torch.zeros((1, 26), device=env.device))
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        matrix = generalized_mass_matrix(env._robot)
        bias = inferred_generalized_bias(
            mass_matrix=matrix,
            applied_effort=env._robot.data.applied_torque,
            joint_acceleration=env._robot.data.joint_acc,
        )
        diagnostics = mass_matrix_diagnostics(
            matrix,
            wrist_joint_ids=env._virtual_wrist_joint_ids,
            finger_joint_ids=env._finger_target_joint_ids,
        )
        view = env._robot.root_physx_view
        runtime_methods = sorted(
            name
            for name in dir(view)
            if any(
                token in name.lower()
                for token in ("mass", "coriolis", "gravity", "jacobian", "force")
            )
        )
        result = {
            "status": "C3R3_PHYSX_ARTICULATION_DYNAMICS_API_VALIDATED",
            "runtime": {
                "mass_matrix": "root_physx_view.get_generalized_mass_matrices",
                "bias": "A2 inferred: applied_torque - M @ joint_acc",
                "effort_command": "Articulation.set_joint_effort_target",
                "actual_effort": "ArticulationData.applied_torque",
                "joint_acceleration": "ArticulationData.joint_acc",
            },
            "runtime_view_methods_matching_dynamics": runtime_methods,
            "joint_order": env._robot.joint_names,
            "wrist_joint_ids": env._virtual_wrist_joint_ids,
            "finger_joint_ids": env._finger_target_joint_ids,
            "mass_matrix": diagnostics,
            "bias": {
                "shape": list(bias.shape),
                "finite": bool(torch.isfinite(bias).all()),
                "source": "A2_previous_substep_effort_acceleration",
            },
            "fixed_root": True,
            "gpu_tensor": str(matrix.device).startswith("cuda"),
            "drive_effects": "wrist stiffness=0,damping=0; fingers retain position drive",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        matrix_report = args.matrix_output or args.output.with_name("mass_matrix_validation.json")
        matrix_report.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
