#!/usr/bin/env python3
"""Run the single bounded-TVLQR Path B qualification on both frozen clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
_STEPS = 41
_GATE = {
    "position_max_m": 0.02,
    "rotation_max_deg": 10.0,
    "position_rmse_m": 0.01,
    "rotation_rmse_deg": 5.0,
    "saturation": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--controller", choices=("tvlqr", "mpc"), default="tvlqr", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3r3_joint_dynamics_c5/tvlqr_qualification.json",
    )
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


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


def _sample(env: Any, torch: Any) -> tuple[float, float, bool]:
    from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
        relative_rotation_log_local,
    )

    state = env._state()
    position_ref = env.reference_bank.gather(
        "wrist_pose_translation_world_ref", env._clip_index, env._reference_index
    )
    quaternion_ref = env.reference_bank.gather(
        "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, env._reference_index
    )
    position_error = torch.linalg.vector_norm(state["wrist_position_scene"] - position_ref, dim=-1)
    rotation_error = torch.linalg.vector_norm(
        relative_rotation_log_local(state["wrist_quaternion_wxyz"], quaternion_ref), dim=-1
    )
    return (
        _scalar(position_error.amax()),
        _scalar(rotation_error.amax()) * 180.0 / math.pi,
        bool(torch.isfinite(env._get_observations()["policy"]).all()),
    )


def _interval(env: Any, action: Any) -> None:
    env._pre_physics_step(action)
    for _ in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env._reference_index.copy_(env._target_reference_index)


def _clip_report(env: Any, torch: Any, clip_index: int) -> dict[str, Any]:
    bias_calibration = _reset_clip(env, torch, clip_index)
    action = torch.zeros((env.num_envs, 26), device=env.device)
    pos, rot, finite = _sample(env, torch)
    pos_sq, rot_sq = pos * pos, rot * rot
    pos_max, rot_max = pos, rot
    term_max = {key: 0.0 for key in ("feedforward", "feedback", "command", "applied")}
    gain_max = 0.0
    saturation_count = torch.zeros(6, dtype=torch.long, device=env.device)
    for _ in range(_STEPS - 1):
        _interval(env, action)
        pos, rot, finite_now = _sample(env, torch)
        finite = finite and finite_now
        pos_max, rot_max = max(pos_max, pos), max(rot_max, rot)
        pos_sq, rot_sq = pos_sq + pos * pos, rot_sq + rot * rot
        latest = (
            env._tvlqr_latest
            if env.cfg.wrist_controller_mode == "bounded_tvlqr_wrist_v1"
            else env._mpc_latest
        )
        if latest is None:
            raise RuntimeError("C3_PREVIEW_CONTROLLER_NO_SUBSTEP_EVIDENCE")
        for key in term_max:
            term_max[key] = max(term_max[key], _scalar(latest[key].abs().amax()))
        gain_max = max(gain_max, _scalar(latest["gain"].abs().amax()))
        saturation_count += latest["saturation"].any(dim=0).to(torch.long)
    saturation = saturation_count.to(torch.float32) / float((_STEPS - 1) * env.cfg.decimation)
    report = {
        "clip": env.reference_bank.clip_ids[clip_index],
        "frames_completed": _STEPS,
        "finite": finite,
        "max_position_m": pos_max,
        "max_rotation_deg": rot_max,
        "position_rmse_m": math.sqrt(pos_sq / _STEPS),
        "rotation_rmse_deg": math.sqrt(rot_sq / _STEPS),
        "per_joint_saturation_ratio": saturation.detach().cpu().tolist(),
        "aggregate_torque_saturation_ratio": _scalar(saturation.max()),
        "terms_max_abs": term_max,
        "gain_max_abs": gain_max,
        "bias_calibration": bias_calibration,
        "actual_effort_source": "ArticulationData.applied_torque",
        "wrist_root_state_writes_during_step": int(env._wrist_step_state_write_count.sum().item()),
        "object_rollout_state_writes": 0,
    }
    report["pass"] = bool(
        finite
        and pos_max <= _GATE["position_max_m"]
        and rot_max <= _GATE["rotation_max_deg"]
        and report["position_rmse_m"] <= _GATE["position_rmse_m"]
        and report["rotation_rmse_deg"] <= _GATE["rotation_rmse_deg"]
        and report["aggregate_torque_saturation_ratio"] <= _GATE["saturation"]
    )
    return report


def _worker(args: argparse.Namespace) -> int:
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
            configure_bounded_mpc_wrist,
            configure_bounded_tvlqr_wrist,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.lazy_sensor_update = True
        cfg.balanced_clip_assignment = False
        cfg.contact_telemetry = "off"
        cfg.identified_tvlqr_model_path = str(args.model_path)
        if args.controller == "tvlqr":
            configure_bounded_tvlqr_wrist(cfg)
            controller_identifier = "bounded_tvlqr_wrist_v1"
        else:
            configure_bounded_mpc_wrist(cfg)
            controller_identifier = "bounded_mpc_wrist_v1"
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        clips = [_clip_report(env, torch, index) for index in range(2)]
        result = {
            "controller": controller_identifier,
            "identified_model_path": str(args.model_path),
            "gate": _GATE,
            "clips": clips,
            "pass": all(clip["pass"] for clip in clips),
            "contract": env.contract_report(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def main() -> int:
    args = parse_args()
    if args.worker:
        return _worker(args)
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--worker",
        "--controller",
        args.controller,
        "--model-path",
        str(args.model_path),
        "--output",
        str(args.output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        check=False,
    )
    if completed.returncode != 0 or not args.output.is_file():
        if args.controller != "mpc":
            raise RuntimeError("C3_TVLQR_WORKER_FAILURE")
        failure = {
            "controller": "bounded_mpc_wrist_v1",
            "identified_model_path": str(args.model_path),
            "pass": False,
            "status": "C3_MPC_WORKER_TERMINATED",
            "failure": {
                "worker_returncode": completed.returncode,
                "reason": (
                    "The isolated PhysX worker terminated after the first bounded MPC interval "
                    "without emitting JSON; C3 requires a safe numerical/physical stop."
                ),
            },
            "gate": _GATE,
            "clips": [],
            "c4_c5_permitted": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("C3_MPC_WORKER_TERMINATED")
        return 0
    result = json.loads(args.output.read_text(encoding="utf-8"))
    status_prefix = "C3_TVLQR" if args.controller == "tvlqr" else "C3_MPC"
    print(f"{status_prefix}_WRIST_TRACKING_" + ("PASS" if result["pass"] else "FAIL"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
