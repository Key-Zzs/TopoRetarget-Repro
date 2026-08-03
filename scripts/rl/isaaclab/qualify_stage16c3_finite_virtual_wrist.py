#!/usr/bin/env python3
"""Qualify the fixed finite virtual six-DoF Stage 16-C.3 wrist actuator."""

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
_POSITION_MAX_M = 0.02
_ROTATION_MAX_DEG = 10.0
_POSITION_RMSE_M = 0.01
_ROTATION_RMSE_DEG = 5.0
_SATURATION_MAX = 0.05
_TELEPORT_CONTROL_STEP_DELTA_MAX_M = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT / ".local/reports/stage16c3r2_c5/c3/path_b_finite_virtual_noncontact.json"
        ),
    )
    parser.add_argument(
        "--worker-mode",
        choices=("tracking", "disturbance", "authority_removed"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-profile",
        choices=("conservative", "nominal", "high_authority_bounded"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _reset_for_clip(env: Any, torch: Any, clip_index: int) -> None:
    env._clip_index.fill_(clip_index)
    env._reset_idx(torch.arange(env.num_envs, device=env.device))
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
    # Path B's wrist gate excludes task-object contact.  Both free bodies are
    # moved to the pre-existing inactive scene offset during reset setup only;
    # no rollout action or per-step object-state write is introduced.
    env._object_170105.write_root_state_to_sim(inactive_state)
    env._object_170650.write_root_state_to_sim(inactive_state)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def _wrist_sample(env: Any, torch: Any) -> dict[str, Any]:
    from toporetarget.rl.environments.isaaclab_backend.tensor_math import (
        relative_rotation_log_local,
    )

    state = env._state()
    index = env._reference_index
    position_reference = env.reference_bank.gather(
        "wrist_pose_translation_world_ref", env._clip_index, index
    )
    quaternion_reference = env.reference_bank.gather(
        "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, index
    )
    position_error = torch.linalg.vector_norm(
        state["wrist_position_scene"] - position_reference, dim=-1
    ).amax()
    rotation_error = torch.linalg.vector_norm(
        relative_rotation_log_local(state["wrist_quaternion_wxyz"], quaternion_reference), dim=-1
    ).amax()
    observation = env._get_observations()["policy"]
    return {
        "position_m": _scalar(position_error),
        "rotation_deg": _scalar(rotation_error) * 180.0 / math.pi,
        "finite": bool(torch.isfinite(observation).all()),
        "wrist_position_scene": state["wrist_position_scene"].clone(),
    }


def _raw_control_interval(env: Any, torch: Any, action: Any) -> None:
    """Run the C.2 physics substeps without formal object termination/reset."""

    env._pre_physics_step(action)
    for _ in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    # This is a wrist-only non-contact gate.  The formal immutable task object
    # termination is intentionally not evaluated while both task objects are
    # absent; the reference cursor still advances exactly one 20 Hz key.
    env._reference_index.copy_(env._target_reference_index)


def _run_zero_action_clip(env: Any, torch: Any, *, clip_index: int) -> dict[str, Any]:
    """Measure all 41 frozen keys with dynamic PhysX wrist substeps only."""

    from toporetarget.rl.environments.isaaclab_backend.explicit_virtual_wrist import (
        explicit_3p3r_rotation_matrix,
    )
    from toporetarget.rl.environments.isaaclab_backend.reference_bank import (
        quaternion_to_matrix_wxyz,
    )

    def joint_diagnostics() -> dict[str, float]:
        state = env._state()
        current = env._robot.data.joint_pos[:, env._virtual_wrist_joint_ids]
        target = env._explicit_wrist_joint_target
        actual_rotation = quaternion_to_matrix_wxyz(state["wrist_quaternion_wxyz"])
        joint_rotation = explicit_3p3r_rotation_matrix(current)
        relative_trace = torch.diagonal(
            actual_rotation.transpose(-1, -2) @ joint_rotation, dim1=-2, dim2=-1
        ).sum(dim=-1)
        kinematic_angle = torch.acos(((relative_trace - 1.0) * 0.5).clamp(-1.0, 1.0))
        return {
            "translation_joint_target_error_m": _scalar(
                torch.linalg.vector_norm(target[:, :3] - current[:, :3], dim=-1).amax()
            ),
            "rotation_joint_target_error_deg": _scalar(
                torch.linalg.vector_norm(target[:, 3:] - current[:, 3:], dim=-1).amax()
            )
            * 180.0
            / math.pi,
            "translation_fk_mismatch_m": _scalar(
                torch.linalg.vector_norm(
                    state["wrist_position_scene"] - current[:, :3], dim=-1
                ).amax()
            ),
            "rotation_fk_mismatch_deg": _scalar(kinematic_angle.amax()) * 180.0 / math.pi,
        }

    _reset_for_clip(env, torch, clip_index)
    action = torch.zeros((env.num_envs, 26), device=env.device)
    initial = _wrist_sample(env, torch)
    previous_wrist = initial["wrist_position_scene"]
    maxima = {
        "position_m": initial["position_m"],
        "rotation_deg": initial["rotation_deg"],
        "control_step_delta_m": 0.0,
    }
    squared = {
        "position_m": initial["position_m"] ** 2,
        "rotation_deg": initial["rotation_deg"] ** 2,
    }
    finite = initial["finite"]
    state_evolved = False
    diagnostic_maxima = joint_diagnostics()
    for _ in range(_STEPS - 1):
        _raw_control_interval(env, torch, action)
        sample = _wrist_sample(env, torch)
        maxima["position_m"] = max(maxima["position_m"], sample["position_m"])
        maxima["rotation_deg"] = max(maxima["rotation_deg"], sample["rotation_deg"])
        squared["position_m"] += sample["position_m"] ** 2
        squared["rotation_deg"] += sample["rotation_deg"] ** 2
        finite = finite and sample["finite"]
        current_wrist = sample["wrist_position_scene"]
        step_delta = _scalar(
            torch.linalg.vector_norm(current_wrist - previous_wrist, dim=-1).amax()
        )
        maxima["control_step_delta_m"] = max(maxima["control_step_delta_m"], step_delta)
        state_evolved = state_evolved or step_delta > 1.0e-8
        previous_wrist = current_wrist.clone()
        diagnostics = joint_diagnostics()
        diagnostic_maxima = {
            key: max(diagnostic_maxima[key], value) for key, value in diagnostics.items()
        }
    total_physics_substeps = int(env._wrist_substeps.max().item())
    return {
        "clip": env.reference_bank.clip_ids[clip_index],
        "frames_requested": _STEPS,
        "frames_completed": _STEPS,
        "control_intervals": _STEPS - 1,
        "formal_task_termination_evaluated": False,
        "finite": finite,
        "physx_state_evolved": state_evolved,
        "maxima": maxima,
        "saturation": {
            "force_ratio": _scalar(
                env._force_saturation_substeps.max().to(torch.float32)
                / env._wrist_substeps.max().clamp_min(1).to(torch.float32)
            ),
            "torque_ratio": _scalar(
                env._torque_saturation_substeps.max().to(torch.float32)
                / env._wrist_substeps.max().clamp_min(1).to(torch.float32)
            ),
            "velocity_ratio": _scalar(
                env._velocity_saturation_substeps.max().to(torch.float32)
                / env._wrist_substeps.max().clamp_min(1).to(torch.float32)
            ),
            "physics_substeps": total_physics_substeps,
        },
        "diagnostic_maxima": diagnostic_maxima,
        "rmse": {key: math.sqrt(value / _STEPS) for key, value in squared.items()},
    }


def _run_finite_disturbance(env: Any, torch: Any) -> dict[str, Any]:
    """Apply one bounded action residual and observe physical, finite response."""

    _reset_for_clip(env, torch, 0)
    before = env._state()["wrist_position_scene"].clone()
    action = torch.zeros((env.num_envs, 26), device=env.device)
    action[:, 0] = 0.5
    _raw_control_interval(env, torch, action)
    sample = _wrist_sample(env, torch)
    after = env._state()["wrist_position_scene"]
    return {
        "action": "wrist_translation_x_residual=+0.5",
        "finite": sample["finite"],
        "wrist_control_step_delta_m": _scalar(
            torch.linalg.vector_norm(after - before, dim=-1).amax()
        ),
        "wrist_position_error_m": sample["position_m"],
    }


def _tracking_pass(report: dict[str, Any]) -> bool:
    maxima = report["maxima"]
    rmse = report["rmse"]
    return bool(
        report["frames_completed"] == _STEPS
        and report["finite"]
        and report["physx_state_evolved"]
        and maxima["position_m"] <= _POSITION_MAX_M
        and maxima["rotation_deg"] <= _ROTATION_MAX_DEG
        and rmse["position_m"] <= _POSITION_RMSE_M
        and rmse["rotation_deg"] <= _ROTATION_RMSE_DEG
        and maxima["control_step_delta_m"] <= _TELEPORT_CONTROL_STEP_DELTA_MAX_M
        and report["saturation"]["force_ratio"] <= _SATURATION_MAX
        and report["saturation"]["torque_ratio"] <= _SATURATION_MAX
        and report["saturation"]["velocity_ratio"] <= _SATURATION_MAX
    )


def _reference_envelope(env: Any, torch: Any) -> dict[str, float]:
    twist = env.reference_bank.wrist_twist_world_ref
    acceleration = (twist[:, 1:] - twist[:, :-1]) / env.step_dt
    return {
        "linear_velocity_max_mps": _scalar(torch.linalg.vector_norm(twist[..., :3], dim=-1).amax()),
        "angular_velocity_max_radps": _scalar(
            torch.linalg.vector_norm(twist[..., 3:], dim=-1).amax()
        ),
        "linear_acceleration_max_mps2": _scalar(
            torch.linalg.vector_norm(acceleration[..., :3], dim=-1).amax()
        ),
        "angular_acceleration_max_radps2": _scalar(
            torch.linalg.vector_norm(acceleration[..., 3:], dim=-1).amax()
        ),
    }


def _worker_command(args: argparse.Namespace, *, mode: str, profile: str, output: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--worker-mode",
        mode,
        "--worker-profile",
        profile,
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"C3_EXPLICIT_WRIST_WORKER_FAILURE: mode={mode} profile={profile} "
            f"returncode={completed.returncode}"
        )


def _orchestrate(args: argparse.Namespace) -> int:
    from toporetarget.rl.environments.isaaclab_backend.d6_wrist_asset import D6_WRIST_PROFILES

    worker_root = args.output.parent / f".{args.output.stem}_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    profile_reports: dict[str, dict[str, Any]] = {}
    reference_envelope: dict[str, float] | None = None
    selected_profile: str | None = None
    for candidate in D6_WRIST_PROFILES:
        profile = candidate.identifier
        output = worker_root / f"tracking_{profile}.json"
        _worker_command(args, mode="tracking", profile=profile, output=output)
        worker = json.loads(output.read_text(encoding="utf-8"))
        reference_envelope = worker["reference_envelope"]
        profile_reports[profile] = worker["profile_report"]
        if worker["profile_report"]["tracking_gate_pass"]:
            selected_profile = profile
            break
    assert reference_envelope is not None
    ablation_profile = selected_profile or "nominal"
    disturbance_output = worker_root / f"disturbance_{ablation_profile}.json"
    _worker_command(args, mode="disturbance", profile=ablation_profile, output=disturbance_output)
    disturbance_worker = json.loads(disturbance_output.read_text(encoding="utf-8"))
    authority_removed_output = worker_root / f"authority_removed_{ablation_profile}.json"
    _worker_command(
        args,
        mode="authority_removed",
        profile=ablation_profile,
        output=authority_removed_output,
    )
    authority_worker = json.loads(authority_removed_output.read_text(encoding="utf-8"))
    disturbance = disturbance_worker["finite_disturbance"]
    authority_removed = authority_worker["two_clip_results"]
    enabled_contract = disturbance_worker["contract"]
    authority_removed_contract = authority_worker["contract"]
    disturbance_ok = disturbance["finite"] and disturbance["wrist_control_step_delta_m"] > 1e-8
    ablation_enabled = profile_reports[ablation_profile]["two_clip_results"]
    enabled_position_rmse = sum(report["rmse"]["position_m"] for report in ablation_enabled)
    disabled_position_rmse = sum(report["rmse"]["position_m"] for report in authority_removed)
    authority_degrades = disabled_position_rmse > enabled_position_rmse + 1e-5
    no_state_write = all(
        contract["wrist_root_state_writes_during_step"] == 0
        and contract["object_rollout_state_writes"] == 0
        for contract in (enabled_contract, authority_removed_contract)
    )
    passed = (
        selected_profile is not None and disturbance_ok and authority_degrades and no_state_write
    )
    result = {
        "status": (
            "C3_FINITE_6DOF_WRIST_ACTUATION_VALIDATED"
            if passed
            else "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED"
        ),
        "mode": "finite_virtual_6d_wrist_actuator_v1",
        "articulation_model": "explicit_serial_3p3r",
        "real_arm": False,
        "profile_candidates": [profile.identifier for profile in D6_WRIST_PROFILES],
        "profile_selection": "first_two_clip_passing_global_profile_in_frozen_order",
        "selected_profile": selected_profile,
        "ablation_profile": ablation_profile,
        "frames": _STEPS,
        "object_scope": "both task objects inactive and non-contact during wrist gate",
        "contact_sensor_update": "lazy_and_unread_for_noncontact_wrist_gate",
        "process_isolation": "one_fresh_isaac_process_per_profile_or_ablation",
        "acceptance": {
            "position_max_m": _POSITION_MAX_M,
            "rotation_max_deg": _ROTATION_MAX_DEG,
            "position_rmse_m": _POSITION_RMSE_M,
            "rotation_rmse_deg": _ROTATION_RMSE_DEG,
            "force_torque_velocity_saturation_ratio_max": _SATURATION_MAX,
            "control_step_teleport_max_m": _TELEPORT_CONTROL_STEP_DELTA_MAX_M,
        },
        "reference_envelope": reference_envelope,
        "profile_reports": profile_reports,
        "finite_disturbance": disturbance,
        "authority_removed": authority_removed,
        "authority_removal_degrades_tracking": authority_degrades,
        "authority_enabled_position_rmse_sum_m": enabled_position_rmse,
        "authority_removed_position_rmse_sum_m": disabled_position_rmse,
        "no_rollout_pose_velocity_or_object_writes": no_state_write,
        "contract": enabled_contract,
        "authority_removed_contract": authority_removed_contract,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 1


def _worker_main(args: argparse.Namespace) -> int:
    if args.worker_profile is None or args.worker_mode is None:
        raise RuntimeError("worker profile and mode are required together")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.d6_wrist_asset import (
            D6_WRIST_PROFILES,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
            configure_explicit_virtual_wrist,
        )

        candidate = next(
            profile for profile in D6_WRIST_PROFILES if profile.identifier == args.worker_profile
        )
        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.lazy_sensor_update = True
        cfg.balanced_clip_assignment = False
        cfg.contact_telemetry = "off"
        configure_explicit_virtual_wrist(
            cfg,
            profile_identifier=candidate.identifier,
            authority_enabled=args.worker_mode != "authority_removed",
        )
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260803)
        if args.worker_mode == "tracking":
            reference_envelope = _reference_envelope(env, torch)
            clips = [_run_zero_action_clip(env, torch, clip_index=index) for index in range(2)]
            tracking_ok = all(_tracking_pass(report) for report in clips)
            result = {
                "worker_mode": args.worker_mode,
                "profile": candidate.identifier,
                "reference_envelope": reference_envelope,
                "profile_report": {
                    "profile": candidate.identifier,
                    "velocity_envelope_covered": (
                        candidate.translation_velocity_limit_mps
                        >= reference_envelope["linear_velocity_max_mps"]
                        and candidate.rotation_velocity_limit_radps
                        >= reference_envelope["angular_velocity_max_radps"]
                    ),
                    "tracking_gate_pass": tracking_ok,
                    "two_clip_results": clips,
                },
                "contract": env.contract_report(),
            }
        elif args.worker_mode == "disturbance":
            result = {
                "worker_mode": args.worker_mode,
                "profile": candidate.identifier,
                "finite_disturbance": _run_finite_disturbance(env, torch),
                "contract": env.contract_report(),
            }
        else:
            result = {
                "worker_mode": args.worker_mode,
                "profile": candidate.identifier,
                "two_clip_results": [
                    _run_zero_action_clip(env, torch, clip_index=index) for index in range(2)
                ],
                "contract": env.contract_report(),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "explicit_wrist_worker_complete",
                    "mode": args.worker_mode,
                    "profile": candidate.identifier,
                    "output": str(args.output),
                },
                sort_keys=True,
            ),
            flush=True,
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
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.worker_mode is None:
        return _orchestrate(args)
    return _worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
