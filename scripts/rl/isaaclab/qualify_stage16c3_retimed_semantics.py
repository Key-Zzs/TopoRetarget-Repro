#!/usr/bin/env python3
"""Run C3-1 through C3-4 with the qualified uniformly retimed wrist."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_FINGER_FINAL_RMSE_RAD = 0.02
_LINK_FINAL_RMSE_M = 0.005
_WRIST_GATE = {
    "position_max_m": 0.02,
    "position_rmse_m": 0.01,
    "rotation_max_deg": 10.0,
    "rotation_rmse_deg": 5.0,
    "saturation": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--reference-time-scale", type=int, default=8)
    parser.add_argument("--profile", default="high_authority_bounded")
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--c3-0-report", type=Path, required=True)
    parser.add_argument("--contact-causality-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-mode", choices=("kinematic", "dynamic"))
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrist_errors(env: Any, torch: Any) -> tuple[float, float]:
    from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
        relative_rotation_log_local,
    )

    state = env._state()
    position = env.reference_bank.gather(
        "wrist_pose_translation_world_ref", env._clip_index, env._reference_index
    )
    quaternion = env.reference_bank.gather(
        "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, env._reference_index
    )
    position_error = torch.linalg.vector_norm(state["wrist_position_scene"] - position, dim=-1)
    rotation_error = torch.linalg.vector_norm(
        relative_rotation_log_local(state["wrist_quaternion_wxyz"], quaternion), dim=-1
    )
    return (
        float(position_error.amax().detach().cpu()),
        float(rotation_error.amax().detach().cpu()) * 180.0 / math.pi,
    )


def _finger_link_errors(env: Any, torch: Any) -> tuple[float, float]:
    state = env._state()
    q_reference = env.reference_bank.gather("q_finger_ref", env._clip_index, env._reference_index)
    link_reference = env.reference_bank.gather(
        "tracked_link_positions_world_ref", env._clip_index, env._reference_index
    )
    finger_rmse = torch.sqrt(torch.mean((state["finger_q"] - q_reference).square(), dim=-1))
    link_rmse = torch.sqrt(
        torch.mean((state["tracked_links_scene"] - link_reference).square(), dim=(-2, -1))
    )
    return float(finger_rmse.amax().detach().cpu()), float(link_rmse.amax().detach().cpu())


def _final_finger_diagnostics(env: Any, torch: Any) -> dict[str, Any]:
    state = env._state()
    q_reference = env.reference_bank.gather("q_finger_ref", env._clip_index, env._reference_index)
    qdot_reference = env.reference_bank.gather(
        "qdot_finger_ref", env._clip_index, env._reference_index
    )
    error = state["finger_q"] - q_reference
    absolute_error = error.abs()
    worst = int(absolute_error[0].argmax().detach().cpu())
    return {
        "joint_order": list(env.reference_bank.joint_order),
        "error_rad": error[0].detach().cpu().tolist(),
        "absolute_error_rad": absolute_error[0].detach().cpu().tolist(),
        "actual_q_rad": state["finger_q"][0].detach().cpu().tolist(),
        "reference_q_rad": q_reference[0].detach().cpu().tolist(),
        "actual_qdot_radps": state["finger_qdot"][0].detach().cpu().tolist(),
        "reference_qdot_radps": qdot_reference[0].detach().cpu().tolist(),
        "worst_joint": env.reference_bank.joint_order[worst],
        "worst_absolute_error_rad": float(absolute_error[0, worst].detach().cpu()),
        "applied_torque_nm": env._robot.data.applied_torque[0, env._finger_target_joint_ids]
        .detach()
        .cpu()
        .tolist(),
        "effort_limit_nm": env._robot.data.joint_effort_limits[0, env._finger_target_joint_ids]
        .detach()
        .cpu()
        .tolist(),
    }


def _raw_interval(env: Any, action: Any) -> None:
    env._pre_physics_step(action)
    for _ in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env._record_completed_contact_substep()
    if env.cfg.diagnostic_kinematic_object:
        env._write_kinematic_object_diagnostic()
    env._reference_index.copy_(env._target_reference_index)


def _mode_report(
    env: Any,
    torch: Any,
    clip_index: int,
    *,
    kinematic_object: bool,
) -> dict[str, Any]:
    env.cfg.diagnostic_kinematic_object = kinematic_object
    env._clip_index.fill_(clip_index)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    action = torch.zeros((1, 26), device=env.device)
    position, rotation = _wrist_errors(env, torch)
    finger, link = _finger_link_errors(env, torch)
    position_max, rotation_max = position, rotation
    position_squared, rotation_squared = position * position, rotation * rotation
    finger_max, link_max = finger, link
    for _ in range(env.reference_bank.frame_count - 1):
        _raw_interval(env, action)
        position, rotation = _wrist_errors(env, torch)
        finger, link = _finger_link_errors(env, torch)
        position_max = max(position_max, position)
        rotation_max = max(rotation_max, rotation)
        position_squared += position * position
        rotation_squared += rotation * rotation
        finger_max = max(finger_max, finger)
        link_max = max(link_max, link)
    samples = env.reference_bank.frame_count
    wrist = {
        "max_position_m": position_max,
        "position_rmse_m": math.sqrt(position_squared / samples),
        "max_rotation_deg": rotation_max,
        "rotation_rmse_deg": math.sqrt(rotation_squared / samples),
        "force_saturation_ratio": float(
            env._force_saturation_substeps.max().detach().cpu()
            / env._wrist_substeps.max().clamp_min(1).detach().cpu()
        ),
        "torque_saturation_ratio": float(
            env._torque_saturation_substeps.max().detach().cpu()
            / env._wrist_substeps.max().clamp_min(1).detach().cpu()
        ),
    }
    wrist_pass = (
        wrist["max_position_m"] <= _WRIST_GATE["position_max_m"]
        and wrist["position_rmse_m"] <= _WRIST_GATE["position_rmse_m"]
        and wrist["max_rotation_deg"] <= _WRIST_GATE["rotation_max_deg"]
        and wrist["rotation_rmse_deg"] <= _WRIST_GATE["rotation_rmse_deg"]
        and wrist["force_saturation_ratio"] <= _WRIST_GATE["saturation"]
        and wrist["torque_saturation_ratio"] <= _WRIST_GATE["saturation"]
    )
    final_finger, final_link = _finger_link_errors(env, torch)
    contract = env.contract_report()
    no_rollout_writes = contract["wrist_root_state_writes_during_step"] == 0 and (
        kinematic_object or contract["object_rollout_state_writes"] == 0
    )
    passed = (
        wrist_pass
        and final_finger <= _FINGER_FINAL_RMSE_RAD
        and final_link <= _LINK_FINAL_RMSE_M
        and no_rollout_writes
    )
    return {
        "clip": env.reference_bank.clip_ids[clip_index],
        "mode": "dynamic_wrist_fingers_kinematic_object"
        if kinematic_object
        else "dynamic_wrist_fingers_free_object_zero_residual",
        "retimed_control_steps": samples,
        "wrist": wrist,
        "finger_rmse_rad": {"maximum": finger_max, "final": final_finger},
        "final_finger_diagnostics": _final_finger_diagnostics(env, torch),
        "tracked_link_rmse_m": {"maximum": link_max, "final": final_link},
        "diagnostic_object_state_writes": contract["diagnostic_object_state_writes"],
        "formal_object_rollout_state_writes": contract["object_rollout_state_writes"],
        "wrist_root_state_writes_during_step": contract["wrist_root_state_writes_during_step"],
        "pass": passed,
    }


def _basis_report(env: Any, torch: Any) -> dict[str, Any]:
    env.cfg.diagnostic_kinematic_object = False
    env._clip_index.zero_()
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    rows = []
    for dimension in range(26):
        positive = torch.zeros((1, 26), device=env.device)
        negative = torch.zeros_like(positive)
        positive[:, dimension] = 0.25
        negative[:, dimension] = -0.25
        env._pre_physics_step(positive)
        positive_finger = env._joint_target_isaac.clone()
        positive_position = env._wrist_target_position.clone()
        positive_quaternion = env._wrist_target_quaternion.clone()
        env._pre_physics_step(negative)
        if dimension < 3:
            response = torch.linalg.vector_norm(positive_position - env._wrist_target_position)
        elif dimension < 6:
            response = torch.linalg.vector_norm(positive_quaternion - env._wrist_target_quaternion)
        else:
            joint = int(env.action_adapter.isaac_from_canonical[dimension - 6].item())
            response = torch.abs(positive_finger[:, joint] - env._joint_target_isaac[:, joint])
        value = float(response.detach().cpu())
        rows.append({"dimension": dimension, "response": value, "pass": value > 0.0})
    return {"basis_count": 26, "all_pass": all(row["pass"] for row in rows), "rows": rows}


def _trace_report(env: Any, torch: Any, trace: dict[str, Any]) -> dict[str, Any]:
    clip_index = env.reference_bank.clip_ids.index(trace["clip"])
    with np.load(trace["action_trace"], allow_pickle=False) as payload:
        source_actions = np.asarray(payload["actions"], dtype=np.float32)
    if source_actions.shape != (40, 26) or not np.isfinite(source_actions).all():
        raise RuntimeError(f"C3_MUJOCO_ACTION_TRACE_INVALID: {trace['action_trace']}")
    actions = np.repeat(source_actions, env.cfg.reference_time_scale, axis=0)
    env.cfg.diagnostic_kinematic_object = False
    env._clip_index.fill_(clip_index)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    first_reason = None
    steps = 0
    for action_array in actions:
        action = torch.as_tensor(action_array[None], device=env.device)
        _, _, terminated, truncated, extras = env.step(action)
        steps += 1
        if bool((terminated | truncated).any()):
            code = int(extras["stage16"]["primary_reason_code"][0].item())
            first_reason = extras["stage16"]["termination_reasons"][code]
            break
    return {
        "clip": trace["clip"],
        "source_horizon": trace["horizon"],
        "source_trace": trace["action_trace"],
        "source_trace_sha256": _sha256(Path(trace["action_trace"])),
        "source_action_shape": list(source_actions.shape),
        "retimed_action_shape": list(actions.shape),
        "retiming_rule": "repeat each source action for each uniform retimed interval",
        "action_bounds_pass": bool(np.max(np.abs(source_actions)) <= 1.0),
        "executed_control_steps": steps,
        "first_termination": first_reason,
        "classification": first_reason or "NO_TERMINATION_WITHIN_TRACE",
        "classified": True,
        "formal_object_rollout_state_writes": env.contract_report()["object_rollout_state_writes"],
    }


def _close_env(env: Any) -> None:
    env.close()
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


def _run_worker(args: argparse.Namespace) -> int:
    if args.worker_output is None:
        raise RuntimeError("--worker-output is required in worker mode")
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
            configure_explicit_virtual_wrist,
            configure_uniform_reference_retiming,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.lazy_sensor_update = True
        cfg.balanced_clip_assignment = False
        cfg.contact_telemetry = "aggregate"
        configure_uniform_reference_retiming(cfg, time_scale=args.reference_time_scale)
        configure_explicit_virtual_wrist(cfg, profile_identifier=args.profile)
        if args.worker_mode == "kinematic":
            cfg.diagnostic_kinematic_object = True
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        if args.worker_mode == "kinematic":
            payload = {
                "worker_mode": args.worker_mode,
                "C3-1": [
                    _mode_report(env, torch, clip, kinematic_object=True) for clip in range(2)
                ],
                "contract": env.contract_report(),
            }
        else:
            trace_manifest = json.loads(args.trace_manifest.read_text(encoding="utf-8"))
            payload = {
                "worker_mode": args.worker_mode,
                "C3-2": [
                    _mode_report(env, torch, clip, kinematic_object=False) for clip in range(2)
                ],
                "C3-3": _basis_report(env, torch),
                "C3-4": [_trace_report(env, torch, trace) for trace in trace_manifest["traces"]],
                "contract": env.contract_report(),
            }
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0
    finally:
        if env is not None:
            _close_env(env)
        app.close(wait_for_replicator=False)


def _worker_command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--reference-time-scale",
        str(args.reference_time_scale),
        "--profile",
        args.profile,
        "--trace-manifest",
        str(args.trace_manifest),
        "--c3-0-report",
        str(args.c3_0_report),
        "--contact-causality-report",
        str(args.contact_causality_report),
        "--output",
        str(args.output),
        "--worker-mode",
        mode,
        "--worker-output",
        str(output),
    ]


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required")
    if args.worker_mode is not None:
        return _run_worker(args)
    c3_0 = json.loads(args.c3_0_report.read_text(encoding="utf-8"))
    causality = json.loads(args.contact_causality_report.read_text(encoding="utf-8"))
    if c3_0["status"] != "C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED":
        raise RuntimeError("C3-0 prerequisite is not validated")
    if causality["status"] != "C3_CONTACT_CAUSALITY_VALIDATED":
        raise RuntimeError("C3-5 prerequisite is not validated")
    causality_time_scale = int(causality.get("reference_time_scale", 1))
    worker_root = args.output.parent / f"{args.output.stem}.workers"
    worker_paths = {mode: worker_root / f"{mode}.json" for mode in ("kinematic", "dynamic")}
    worker_exits = {}
    for mode, output in worker_paths.items():
        completed = subprocess.run(_worker_command(args, mode, output), check=False)
        worker_exits[mode] = completed.returncode
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"C3_{mode.upper()}_WORKER_FAILED: exit={completed.returncode}, output={output}"
            )
    kinematic = json.loads(worker_paths["kinematic"].read_text(encoding="utf-8"))
    dynamic = json.loads(worker_paths["dynamic"].read_text(encoding="utf-8"))
    c3_1, c3_2 = kinematic["C3-1"], dynamic["C3-2"]
    c3_3, c3_4 = dynamic["C3-3"], dynamic["C3-4"]
    passes = {
        "C3-0": True,
        "C3-1": all(row["pass"] for row in c3_1),
        "C3-2": all(row["pass"] for row in c3_2),
        "C3-3": c3_3["all_pass"],
        "C3-4": all(row["classified"] for row in c3_4),
        "C3-5": causality_time_scale == args.reference_time_scale,
    }
    result = {
        "status": "STAGE16C3_SEMANTIC_QUALIFICATION_VALIDATED"
        if all(passes.values())
        else "STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED",
        "controller": "finite_virtual_6d_wrist_actuator_v1",
        "profile": args.profile,
        "reference_time_scale": args.reference_time_scale,
        "source_keyframes": 41,
        "retimed_control_steps": (41 - 1) * args.reference_time_scale + 1,
        "process_isolation": {
            "worker_exits": worker_exits,
            "worker_reports": {key: str(value) for key, value in worker_paths.items()},
        },
        "passes": passes,
        "gates": {
            "wrist": _WRIST_GATE,
            "finger_final_rmse_rad": _FINGER_FINAL_RMSE_RAD,
            "tracked_link_final_rmse_m": _LINK_FINAL_RMSE_M,
        },
        "C3-0": {"report": str(args.c3_0_report), "status": c3_0["status"]},
        "C3-1": c3_1,
        "C3-1-kinematic-contract": {
            "actor_mode": "dynamic_with_post_interval_reference_state_reset",
            "diagnostic_object_state_writes": kinematic["contract"][
                "diagnostic_object_state_writes"
            ],
            "formal_object_rollout_state_writes": kinematic["contract"][
                "object_rollout_state_writes"
            ],
        },
        "C3-2": c3_2,
        "C3-3": c3_3,
        "C3-4": c3_4,
        "C3-5": {
            "report": str(args.contact_causality_report),
            "status": causality["status"],
            "reference_time_scale": causality_time_scale,
            "matches_qualification_time_scale": (causality_time_scale == args.reference_time_scale),
        },
        "contract": dynamic["contract"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if all(passes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
