#!/usr/bin/env python3
"""Run withheld 1-step and 6-substep gates for the affine V2 wrist model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r4_mpc_holdout_c4"

_CONTROL_DT_S = 1.0 / 20.0
_Q_SCALE = np.asarray([0.01] * 3 + [math.radians(5.0)] * 3, dtype=np.float32)
_STATE_SCALE = np.concatenate((_Q_SCALE, _Q_SCALE / _CONTROL_DT_S))
_EFFORT_SCALE = np.asarray([500.0] * 6, dtype=np.float32)
_STATE_DIRECTION = np.asarray(
    [1.0, -0.7, 0.5, -0.8, 0.6, -0.4, -0.5, 0.8, -0.6, 0.7, -0.4, 1.0],
    dtype=np.float32,
)
_EFFORT_DIRECTION = np.asarray([1.0, -0.6, 0.4, -0.8, 0.5, -0.3], dtype=np.float32)
_STATE_FRACTION = 0.05
_EFFORT_FRACTION = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=REPORT_ROOT / "local_dynamics_v2_holdout.json"
    )
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _state(env: Any, torch: Any) -> Any:
    return torch.cat(
        (
            env._robot.data.joint_pos[:, env._virtual_wrist_joint_ids],
            env._robot.data.joint_vel[:, env._virtual_wrist_joint_ids],
        ),
        dim=-1,
    )


def _inactive_objects(env: Any, torch: Any) -> None:
    position = env.scene.env_origins + torch.tensor(
        env.cfg.inactive_object_scene_offset, dtype=torch.float32, device=env.device
    )
    state = torch.cat(
        (
            position,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1),
            torch.zeros((env.num_envs, 6), device=env.device),
        ),
        dim=-1,
    )
    env._object_170105.write_root_state_to_sim(state)
    env._object_170650.write_root_state_to_sim(state)


def _samples(env: Any, torch: Any, *, clip: int, frame: int, substep: int) -> tuple[Any, Any]:
    reference = env._explicit_joint_reference
    clips = torch.full((env.num_envs,), clip, dtype=torch.long, device=env.device)
    frames = torch.full((env.num_envs,), frame, dtype=torch.long, device=env.device)
    current = reference.sample(clips, frames, substep=substep, decimation=env.cfg.decimation)
    following = reference.sample(clips, frames, substep=substep + 1, decimation=env.cfg.decimation)
    return current, following


def _following_state(following: Any, torch: Any) -> Any:
    if isinstance(following, torch.Tensor):
        return following
    return torch.cat((following.q_wrist, following.qd_wrist), dim=-1)


def _setup_interval(env: Any, torch: Any, *, clip: int, frame: int, state_delta: Any) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._clip_index.fill_(clip)
    env._reset_idx(env_ids)
    env._reference_index.fill_(frame)
    env._target_reference_index.fill_(frame + 1)
    current, _ = _samples(env, torch, clip=clip, frame=frame, substep=0)
    q_wrist = current.q_wrist.clone()
    qd_wrist = current.qd_wrist.clone()
    q_wrist[1] += state_delta[:6]
    qd_wrist[1] += state_delta[6:]
    q_finger = env.action_adapter.canonical_to_isaac(current.q_finger)
    qd_finger = env.action_adapter.canonical_to_isaac(current.qd_finger)
    env._robot.write_joint_state_to_sim(
        q_wrist, qd_wrist, joint_ids=env._virtual_wrist_joint_ids, env_ids=env_ids
    )
    env._robot.write_joint_state_to_sim(
        q_finger, qd_finger, joint_ids=env._finger_target_joint_ids, env_ids=env_ids
    )
    finger_target = env.action_adapter.canonical_to_isaac(
        env._explicit_joint_reference.q_finger_ref[clip, frame + 1].expand(env.num_envs, -1)
    )
    env._robot.set_joint_position_target(finger_target, joint_ids=env._finger_target_joint_ids)
    env._robot.set_joint_velocity_target(qd_finger, joint_ids=env._finger_target_joint_ids)
    _inactive_objects(env, torch)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def _metrics(errors: list[Any], torch: Any) -> dict[str, float | bool]:
    value = torch.stack(errors)
    return {
        "finite": bool(torch.isfinite(value).all()),
        "normalized_rmse": _scalar(torch.sqrt(torch.square(value).mean())),
        "normalized_max_abs": _scalar(value.abs().amax()),
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.output.exists():
        raise FileExistsError(f"C3R4_V2_HOLDOUT_REFUSES_OVERWRITE: {args.output}")
    payload = np.load(args.model_path, allow_pickle=False)
    if set(payload.files) != {"A", "B", "C", "U_NOMINAL"}:
        raise RuntimeError(f"C3R4_V2_MODEL_SCHEMA_INVALID: {payload.files}")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    report: dict[str, Any] = {"status": "C3R4_V2_HOLDOUT_NOT_STARTED"}
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.tvlqr_wrist import (
            BoundedMPCWristControllerV1,
            BoundedMPCWristProfileV1,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
            configure_bounded_mpc_wrist,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 2
        cfg.scene.lazy_sensor_update = True
        cfg.contact_telemetry = "off"
        cfg.balanced_clip_assignment = False
        configure_bounded_mpc_wrist(cfg)
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        a = torch.tensor(payload["A"], dtype=torch.float32, device=env.device)
        b = torch.tensor(payload["B"], dtype=torch.float32, device=env.device)
        c = torch.tensor(payload["C"], dtype=torch.float32, device=env.device)
        u_nominal = torch.tensor(payload["U_NOMINAL"], dtype=torch.float32, device=env.device)
        expected_prefix = (
            len(env.reference_bank.clip_ids),
            env.reference_bank.frame_count - 1,
            env.cfg.decimation,
        )
        if a.shape != (*expected_prefix, 12, 12) or b.shape != (*expected_prefix, 12, 6):
            raise RuntimeError("C3R4_V2_MODEL_SHAPE_INVALID")
        state_scale = torch.tensor(_STATE_SCALE, dtype=torch.float32, device=env.device)
        effort_scale = torch.tensor(_EFFORT_SCALE, dtype=torch.float32, device=env.device)
        state_delta = state_scale * torch.tensor(
            _STATE_DIRECTION * _STATE_FRACTION, dtype=torch.float32, device=env.device
        )
        effort_delta = effort_scale * torch.tensor(
            _EFFORT_DIRECTION * _EFFORT_FRACTION,
            dtype=torch.float32,
            device=env.device,
        )
        one_step_difference_errors: list[Any] = []
        six_step_difference_errors: list[Any] = []
        one_step_absolute_errors: list[Any] = []
        six_step_absolute_errors: list[Any] = []
        records: list[dict[str, Any]] = []
        for clip in range(expected_prefix[0]):
            for frame in range(expected_prefix[1]):
                _setup_interval(env, torch, clip=clip, frame=frame, state_delta=state_delta)
                nominal_prediction = torch.zeros(12, device=env.device)
                difference_prediction = state_delta.clone()
                horizons: dict[str, Any] = {}
                for substep in range(env.cfg.decimation):
                    _, following = _samples(env, torch, clip=clip, frame=frame, substep=substep)
                    following_state = _following_state(following, torch)
                    effort = torch.stack(
                        (
                            u_nominal[clip, frame, substep],
                            u_nominal[clip, frame, substep] + effort_delta,
                        )
                    )
                    env._robot.set_joint_effort_target(
                        effort, joint_ids=env._virtual_wrist_joint_ids
                    )
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(env.physics_dt)
                    actual_error = _state(env, torch) - following_state
                    nominal_prediction = (
                        a[clip, frame, substep] @ nominal_prediction + c[clip, frame, substep]
                    )
                    difference_prediction = (
                        a[clip, frame, substep] @ difference_prediction
                        + b[clip, frame, substep] @ effort_delta
                    )
                    predicted_error = torch.stack(
                        (nominal_prediction, nominal_prediction + difference_prediction)
                    )
                    absolute_error = (actual_error - predicted_error) / state_scale
                    difference_actual = actual_error[1] - actual_error[0]
                    difference_error = (difference_actual - difference_prediction) / state_scale
                    horizon = substep + 1
                    if horizon in {1, 6}:
                        horizons[str(horizon)] = {
                            "difference_normalized_rmse": _scalar(
                                torch.sqrt(torch.square(difference_error).mean())
                            ),
                            "absolute_normalized_rmse": _scalar(
                                torch.sqrt(torch.square(absolute_error).mean())
                            ),
                            "difference_normalized_max_abs": _scalar(difference_error.abs().amax()),
                            "absolute_normalized_max_abs": _scalar(absolute_error.abs().amax()),
                        }
                    if horizon == 1:
                        one_step_difference_errors.append(difference_error)
                        one_step_absolute_errors.append(absolute_error)
                    if horizon == 6:
                        six_step_difference_errors.append(difference_error)
                        six_step_absolute_errors.append(absolute_error)
                records.append(
                    {
                        "clip": env.reference_bank.clip_ids[clip],
                        "interval": frame,
                        "horizons": horizons,
                    }
                )
        profile = BoundedMPCWristProfileV1()
        controller = BoundedMPCWristControllerV1(profile, device=env.device)
        hessian_conditions: list[float] = []
        spectral_products: list[float] = []
        q_block = torch.block_diag(*([controller.q[0]] * profile.horizon))[None]
        r_block = torch.block_diag(*([controller.r[0]] * profile.horizon))[None]
        flat_a = a.reshape(expected_prefix[0], -1, 12, 12)
        flat_b = b.reshape(expected_prefix[0], -1, 12, 6)
        flat_c = c.reshape(expected_prefix[0], -1, 12)
        for clip in range(expected_prefix[0]):
            for node in range(flat_a.shape[1]):
                indices = torch.arange(node, node + profile.horizon, device=env.device).clamp_max(
                    flat_a.shape[1] - 1
                )
                _, g, _ = controller._lifted_time_varying_dynamics(
                    flat_a[clip, indices][None],
                    flat_b[clip, indices][None],
                    flat_c[clip, indices][None],
                )
                hessian = g.mT @ q_block @ g + r_block
                eigen_max = torch.linalg.eigvalsh(hessian).amax()
                step = min(profile.projected_gradient_step, 1.0 / _scalar(eigen_max))
                hessian_conditions.append(_scalar(torch.linalg.cond(hessian)[0]))
                spectral_products.append(step * _scalar(eigen_max))
        one_difference = _metrics(one_step_difference_errors, torch)
        six_difference = _metrics(six_step_difference_errors, torch)
        one_absolute = _metrics(one_step_absolute_errors, torch)
        six_absolute = _metrics(six_step_absolute_errors, torch)
        report = {
            "status": "C3R4_V2_LOCAL_DYNAMICS_HOLDOUT_COMPLETE",
            "model_path": str(args.model_path),
            "holdout": {
                "direction_excluded_from_hadamard_fit": True,
                "state_scale_fraction": _STATE_FRACTION,
                "effort_scale_fraction": _EFFORT_FRACTION,
                "one_step": {
                    "difference_model": one_difference,
                    "absolute_affine_model": one_absolute,
                },
                "six_substeps_20hz": {
                    "difference_model": six_difference,
                    "absolute_affine_model": six_absolute,
                },
            },
            "thresholds_frozen_before_run": {
                "one_step_normalized_rmse_max": 0.05,
                "six_step_normalized_rmse_max": 0.25,
            },
            "gates": {
                "one_step": bool(
                    one_difference["finite"]
                    and one_absolute["finite"]
                    and one_difference["normalized_rmse"] <= 0.05
                    and one_absolute["normalized_rmse"] <= 0.05
                ),
                "six_step": bool(
                    six_difference["finite"]
                    and six_absolute["finite"]
                    and six_difference["normalized_rmse"] <= 0.25
                    and six_absolute["normalized_rmse"] <= 0.25
                ),
                "spectral_projected_gradient": bool(max(spectral_products) <= 1.0 + 1.0e-6),
            },
            "solver": {
                "hessian_condition_min": min(hessian_conditions),
                "hessian_condition_median": float(np.median(hessian_conditions)),
                "hessian_condition_max": max(hessian_conditions),
                "spectral_step_product_max": max(spectral_products),
            },
            "records": records,
            "setup_state_writes_only": True,
            "rollout_wrist_state_writes": 0,
            "rollout_object_state_writes": 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(report["status"])
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "C3R4_V2_HOLDOUT_EXCEPTION",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 2
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
