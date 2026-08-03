#!/usr/bin/env python3
"""Trace one bounded-MPC control interval at every Isaac physics boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r4_mpc_holdout_c4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--clip-index", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--output", type=Path, default=REPORT_ROOT / "mpc_first_interval_trace.json"
    )
    return parser.parse_args()


def _cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def _tensor_summary(torch: Any, value: Any) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all()),
        "min": float(value.amin().detach().cpu()),
        "max": float(value.amax().detach().cpu()),
        "max_abs": float(value.abs().amax().detach().cpu()),
    }


def _synchronize(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _state_snapshot(env: Any, torch: Any) -> dict[str, Any]:
    ids = env._virtual_wrist_joint_ids
    q = env._robot.data.joint_pos[:, ids]
    qd = env._robot.data.joint_vel[:, ids]
    qdd = env._robot.data.joint_acc[:, ids]
    applied = env._robot.data.applied_torque[:, ids]
    return {
        "reference_index": int(env._reference_index[0].item()),
        "target_reference_index": int(env._target_reference_index[0].item()),
        "physics_substep": int(env._physics_substep),
        "q": _cpu(q[0]),
        "qd": _cpu(qd[0]),
        "qdd": _cpu(qdd[0]),
        "applied_effort": _cpu(applied[0]),
        "finite": bool(
            torch.isfinite(q).all()
            and torch.isfinite(qd).all()
            and torch.isfinite(qdd).all()
            and torch.isfinite(applied).all()
        ),
    }


def _controller_snapshot(env: Any, torch: Any) -> dict[str, Any]:
    latest = env._mpc_latest
    if latest is None:
        raise RuntimeError("C3R4_MPC_DIAGNOSTIC_MISSING_CONTROLLER_RESULT")
    tensor_keys = (
        "dynamics_a",
        "dynamics_b",
        "mass_wrist",
        "hessian",
        "linear",
        "state_error",
        "unconstrained_control",
        "unconstrained_control_sequence",
        "projected_control_sequence",
        "lower_effort_delta",
        "upper_effort_delta",
        "hessian_lambda_max",
        "projected_gradient_step",
        "projected_gradient_stability_product",
        "feedforward",
        "feedback",
        "command",
        "applied",
        "saturation",
        "coupling",
    )
    result: dict[str, Any] = {"model_source": latest["model_source"]}
    for key in tensor_keys:
        value = latest[key]
        result[key] = _cpu(value[0])
        if value.dtype != torch.bool:
            result[f"{key}_summary"] = _tensor_summary(torch, value)
    result["condition_numbers"] = {
        "A": float(torch.linalg.cond(latest["dynamics_a"])[0].detach().cpu()),
        "B": float(torch.linalg.cond(latest["dynamics_b"])[0].detach().cpu()),
        "M_ww": float(torch.linalg.cond(latest["mass_wrist"])[0].detach().cpu()),
        "H": float(torch.linalg.cond(latest["hessian"])[0].detach().cpu()),
    }
    return result


def _reset_clip(env: Any, torch: Any, clip_index: int) -> dict[str, float]:
    env._clip_index.fill_(clip_index)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    position = env.scene.env_origins + torch.tensor(
        env.cfg.inactive_object_scene_offset, dtype=torch.float32, device=env.device
    )
    inactive = torch.cat(
        (
            position,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1),
            torch.zeros((env.num_envs, 6), device=env.device),
        ),
        dim=-1,
    )
    env._object_170105.write_root_state_to_sim(inactive)
    env._object_170650.write_root_state_to_sim(inactive)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    finger_reference = env.reference_bank.gather(
        "q_finger_ref", env._clip_index, env._reference_index
    )
    env._robot.set_joint_position_target(
        env.action_adapter.canonical_to_isaac(finger_reference),
        joint_ids=env._finger_target_joint_ids,
    )
    return env.calibrate_computed_torque_bias()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.output.exists():
        raise FileExistsError(f"C3R4_DIAGNOSTIC_REFUSES_OVERWRITE: {args.output}")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    report: dict[str, Any] = {
        "status": "C3R4_MPC_FIRST_INTERVAL_NOT_STARTED",
        "clip_index": args.clip_index,
        "model_path": str(args.model_path),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "substeps": [],
    }
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
            configure_bounded_mpc_wrist,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.lazy_sensor_update = True
        cfg.contact_telemetry = "off"
        cfg.balanced_clip_assignment = False
        cfg.identified_tvlqr_model_path = str(args.model_path)
        configure_bounded_mpc_wrist(cfg)
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        report["bias_calibration"] = _reset_clip(env, torch, args.clip_index)
        action = torch.zeros((1, 26), dtype=torch.float32, device=env.device)
        env._pre_physics_step(action)
        for substep in range(cfg.decimation):
            entry: dict[str, Any] = {
                "substep": substep,
                "before_apply_action": _state_snapshot(env, torch),
            }
            env._apply_action()
            _synchronize(torch)
            entry["after_apply_action"] = _state_snapshot(env, torch)
            entry["controller"] = _controller_snapshot(env, torch)
            entry["before_scene_write"] = _state_snapshot(env, torch)
            env.scene.write_data_to_sim()
            _synchronize(torch)
            entry["after_scene_write"] = _state_snapshot(env, torch)
            entry["before_sim_step"] = _state_snapshot(env, torch)
            env.sim.step(render=False)
            _synchronize(torch)
            entry["after_sim_step"] = _state_snapshot(env, torch)
            entry["before_scene_update"] = _state_snapshot(env, torch)
            env.scene.update(env.physics_dt)
            _synchronize(torch)
            entry["after_scene_update"] = _state_snapshot(env, torch)
            report["substeps"].append(entry)
        report.update(
            {
                "status": "C3R4_MPC_FIRST_INTERVAL_TRACED",
                "substeps_completed": len(report["substeps"]),
                "wrist_root_state_writes_during_step": int(
                    env._wrist_step_state_write_count.sum().item()
                ),
                "object_rollout_state_writes": 0,
            }
        )
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "C3R4_MPC_FIRST_INTERVAL_EXCEPTION",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 2
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
