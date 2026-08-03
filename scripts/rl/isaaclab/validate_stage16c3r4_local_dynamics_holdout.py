#!/usr/bin/env python3
"""Validate the C3R3 wrist model beyond its fitted axial one-step probes."""

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
_QD_SCALE = _Q_SCALE / _CONTROL_DT_S
_STATE_SCALE = np.concatenate((_Q_SCALE, _QD_SCALE))
_QDD_SCALE = _Q_SCALE / (_CONTROL_DT_S * _CONTROL_DT_S)
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
        "--output", type=Path, default=REPORT_ROOT / "local_dynamics_holdout_v1.json"
    )
    return parser.parse_args()


def _cpu(value: Any) -> Any:
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value


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


def _live_bias(env: Any, torch: Any) -> tuple[Any, Any, Any]:
    view = env._robot.root_physx_view
    coriolis = view.get_coriolis_and_centrifugal_compensation_forces()
    gravity = view.get_gravity_compensation_forces()
    if not isinstance(coriolis, torch.Tensor) or not isinstance(gravity, torch.Tensor):
        raise RuntimeError("C3R4_PHYSX_BIAS_TENSOR_UNAVAILABLE")
    if coriolis.shape != gravity.shape or coriolis.shape[-1] != env._robot.num_joints:
        raise RuntimeError(
            f"C3R4_PHYSX_BIAS_SHAPE_INVALID: coriolis={coriolis.shape} gravity={gravity.shape}"
        )
    bias = coriolis + gravity
    if not bool(torch.isfinite(bias).all()):
        raise RuntimeError("C3R4_PHYSX_BIAS_NONFINITE")
    return bias, coriolis, gravity


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


def _setup_state(
    env: Any,
    torch: Any,
    *,
    clip: int,
    frame: int,
    state_delta: Any | None = None,
) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._clip_index.fill_(clip)
    env._reset_idx(env_ids)
    env._reference_index.fill_(frame)
    env._target_reference_index.fill_(frame)
    reference = env._explicit_joint_reference
    q_wrist = reference.q_wrist_ref[clip, frame].expand(env.num_envs, -1).clone()
    qd_wrist = reference.qd_wrist_ref[clip, frame].expand(env.num_envs, -1).clone()
    if state_delta is not None:
        q_wrist[1] += state_delta[:6]
        qd_wrist[1] += state_delta[6:]
    q_finger = env.action_adapter.canonical_to_isaac(
        reference.q_finger_ref[clip, frame].expand(env.num_envs, -1)
    )
    qd_finger = env.action_adapter.canonical_to_isaac(
        reference.qd_finger_ref[clip, frame].expand(env.num_envs, -1)
    )
    env._robot.write_joint_state_to_sim(
        q_wrist, qd_wrist, joint_ids=env._virtual_wrist_joint_ids, env_ids=env_ids
    )
    env._robot.write_joint_state_to_sim(
        q_finger, qd_finger, joint_ids=env._finger_target_joint_ids, env_ids=env_ids
    )
    env._robot.set_joint_position_target(q_wrist, joint_ids=env._virtual_wrist_joint_ids)
    env._robot.set_joint_velocity_target(qd_wrist, joint_ids=env._virtual_wrist_joint_ids)
    env._robot.set_joint_position_target(q_finger, joint_ids=env._finger_target_joint_ids)
    env._robot.set_joint_velocity_target(qd_finger, joint_ids=env._finger_target_joint_ids)
    env._robot.set_joint_effort_target(
        torch.zeros((env.num_envs, 6), device=env.device),
        joint_ids=env._virtual_wrist_joint_ids,
    )
    _inactive_objects(env, torch)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def _physics_step(env: Any, effort: Any) -> None:
    env._robot.set_joint_effort_target(effort, joint_ids=env._virtual_wrist_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)


def _component_errors(error: Any, scale: Any, torch: Any) -> dict[str, float]:
    normalized = error / scale
    return {
        "q_translation_rmse_m": _scalar(torch.sqrt(torch.square(error[:, :3]).mean())),
        "q_rotation_rmse_rad": _scalar(torch.sqrt(torch.square(error[:, 3:6]).mean())),
        "qd_translation_rmse_mps": _scalar(torch.sqrt(torch.square(error[:, 6:9]).mean())),
        "qd_rotation_rmse_radps": _scalar(torch.sqrt(torch.square(error[:, 9:]).mean())),
        "normalized_rmse": _scalar(torch.sqrt(torch.square(normalized).mean())),
        "normalized_max_abs": _scalar(normalized.abs().amax()),
    }


def _paired_model_holdout(
    env: Any,
    torch: Any,
    *,
    clip: int,
    frame: int,
    a: Any,
    b: Any,
    state_delta: Any,
    effort_delta: Any,
    state_scale: Any,
) -> dict[str, Any]:
    _setup_state(env, torch, clip=clip, frame=frame, state_delta=state_delta)
    predicted = state_delta.clone()
    effort = torch.stack((torch.zeros_like(effort_delta), effort_delta))
    horizons: dict[str, Any] = {}
    inferred_vs_live: dict[str, Any] | None = None
    for step in range(1, 7):
        _physics_step(env, effort)
        actual = _state(env, torch)[1] - _state(env, torch)[0]
        predicted = a @ predicted + b @ effort_delta
        error = (actual - predicted)[None]
        if step == 1:
            from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
                generalized_mass_matrix,
                inferred_generalized_bias,
            )

            mass = generalized_mass_matrix(env._robot)
            inferred = inferred_generalized_bias(
                mass_matrix=mass,
                applied_effort=env._robot.data.applied_torque,
                joint_acceleration=env._robot.data.joint_acc,
            )
            live, coriolis, gravity = _live_bias(env, torch)
            wrist = env._virtual_wrist_joint_ids
            inferred_vs_live = {
                "inferred_wrist": _cpu(inferred[:, wrist]),
                "live_wrist": _cpu(live[:, wrist]),
                "coriolis_wrist": _cpu(coriolis[:, wrist]),
                "gravity_wrist": _cpu(gravity[:, wrist]),
                "difference_max_abs": _scalar((inferred[:, wrist] - live[:, wrist]).abs().amax()),
            }
        if step in {1, 6}:
            entry = {
                "actual_delta": _cpu(actual),
                "predicted_delta": _cpu(predicted),
                **_component_errors(error, state_scale, torch),
            }
            entry["normalized_signal_rmse"] = _scalar(
                torch.sqrt(torch.square(actual / state_scale).mean())
            )
            entry["relative_normalized_rmse"] = entry["normalized_rmse"] / max(
                entry["normalized_signal_rmse"], 1.0e-12
            )
            horizons[str(step)] = entry
    return {"horizons": horizons, "inferred_vs_live_bias": inferred_vs_live}


def _reference_endpoint_holdout(
    env: Any,
    torch: Any,
    *,
    clip: int,
    frame: int,
    state_scale: Any,
) -> dict[str, Any]:
    """Compare one frozen reset bias with live q/qd-conditioned PhysX bias."""

    _setup_state(env, torch, clip=clip, frame=frame)
    reference = env._explicit_joint_reference
    clip_index = torch.full((env.num_envs,), clip, dtype=torch.long, device=env.device)
    key_index = torch.full((env.num_envs,), frame, dtype=torch.long, device=env.device)
    initial_bias, initial_coriolis, initial_gravity = _live_bias(env, torch)
    fixed_bias = initial_bias[0].clone()
    bias_variation_max = 0.0
    for substep in range(env.cfg.decimation):
        sample = reference.sample(
            clip_index, key_index, substep=substep, decimation=env.cfg.decimation
        )
        finger_target = env.action_adapter.canonical_to_isaac(
            reference.q_finger_ref[clip, min(frame + 1, reference.frame_count - 1)].expand(
                env.num_envs, -1
            )
        )
        env._robot.set_joint_position_target(finger_target, joint_ids=env._finger_target_joint_ids)
        from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
            generalized_mass_matrix,
        )

        mass = generalized_mass_matrix(env._robot)
        wrist_ids = torch.tensor(env._virtual_wrist_joint_ids, device=env.device)
        finger_ids = torch.tensor(env._finger_target_joint_ids, device=env.device)
        mww = mass.index_select(1, wrist_ids).index_select(2, wrist_ids)
        mwf = mass.index_select(1, wrist_ids).index_select(2, finger_ids)
        finger_qdd = env.action_adapter.canonical_to_isaac(sample.qdd_finger)
        live_bias, _, _ = _live_bias(env, torch)
        bias_variation_max = max(
            bias_variation_max,
            _scalar((live_bias[:, wrist_ids] - fixed_bias[wrist_ids]).abs().amax()),
        )
        feedforward = torch.bmm(mww, sample.qdd_wrist.unsqueeze(-1)).squeeze(-1) + torch.bmm(
            mwf, finger_qdd.unsqueeze(-1)
        ).squeeze(-1)
        effort = feedforward.clone()
        effort[0] += fixed_bias[wrist_ids]
        effort[1] += live_bias[1, wrist_ids]
        _physics_step(env, effort)
    endpoint_q = reference.q_wrist_ref[clip, min(frame + 1, reference.frame_count - 1)]
    endpoint_qd = reference.qd_wrist_ref[clip, min(frame + 1, reference.frame_count - 1)]
    endpoint = torch.cat((endpoint_q, endpoint_qd))[None]
    error = _state(env, torch) - endpoint
    return {
        "fixed_reset_bias": _component_errors(error[0:1], state_scale, torch),
        "live_physx_bias": _component_errors(error[1:2], state_scale, torch),
        "bias_variation_max_abs": bias_variation_max,
        "initial_coriolis_wrist_max_abs": _scalar(
            initial_coriolis[:, env._virtual_wrist_joint_ids].abs().amax()
        ),
        "initial_gravity_wrist_max_abs": _scalar(
            initial_gravity[:, env._virtual_wrist_joint_ids].abs().amax()
        ),
    }


def _aggregate(records: list[dict[str, Any]], horizon: str) -> dict[str, float | bool]:
    values = [record["model_holdout"]["horizons"][horizon] for record in records]
    return {
        "finite": all(math.isfinite(value["normalized_rmse"]) for value in values),
        "normalized_rmse": math.sqrt(
            sum(value["normalized_rmse"] ** 2 for value in values) / len(values)
        ),
        "normalized_max_abs": max(value["normalized_max_abs"] for value in values),
        "relative_normalized_rmse": math.sqrt(
            sum(value["relative_normalized_rmse"] ** 2 for value in values) / len(values)
        ),
        "q_translation_rmse_m": math.sqrt(
            sum(value["q_translation_rmse_m"] ** 2 for value in values) / len(values)
        ),
        "q_rotation_rmse_rad": math.sqrt(
            sum(value["q_rotation_rmse_rad"] ** 2 for value in values) / len(values)
        ),
        "qd_translation_rmse_mps": math.sqrt(
            sum(value["qd_translation_rmse_mps"] ** 2 for value in values) / len(values)
        ),
        "qd_rotation_rmse_radps": math.sqrt(
            sum(value["qd_rotation_rmse_radps"] ** 2 for value in values) / len(values)
        ),
    }


def _condition_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.output.exists():
        raise FileExistsError(f"C3R4_HOLDOUT_REFUSES_OVERWRITE: {args.output}")
    model = np.load(args.model_path, allow_pickle=False)
    if set(model.files) != {"A", "B"}:
        raise RuntimeError(f"C3R4_MODEL_SCHEMA_UNEXPECTED: {model.files}")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    report: dict[str, Any] = {
        "status": "C3R4_LOCAL_DYNAMICS_HOLDOUT_NOT_STARTED",
        "model_path": str(args.model_path),
    }
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
            generalized_mass_matrix,
        )
        from toporetarget.rl.environments.isaaclab_backend.tvlqr_wrist import (
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
        cfg.identified_tvlqr_model_path = str(args.model_path)
        configure_bounded_mpc_wrist(cfg)
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        state_scale = torch.tensor(_STATE_SCALE, dtype=torch.float32, device=env.device)
        state_delta = state_scale * torch.tensor(
            _STATE_DIRECTION * _STATE_FRACTION, dtype=torch.float32, device=env.device
        )
        effort_delta = torch.tensor(
            _EFFORT_SCALE * _EFFORT_DIRECTION * _EFFORT_FRACTION,
            dtype=torch.float32,
            device=env.device,
        )
        profile = BoundedMPCWristProfileV1()
        q_diag = torch.tensor(
            [profile.q_translation] * 3 + [profile.q_rotation] * 3 + [profile.q_velocity] * 6,
            dtype=torch.float32,
            device=env.device,
        )
        q = torch.diag(q_diag)
        r = torch.eye(6, dtype=torch.float32, device=env.device) * profile.r_effort
        records: list[dict[str, Any]] = []
        mww_raw_conditions: list[float] = []
        mww_scaled_conditions: list[float] = []
        a_raw_conditions: list[float] = []
        a_scaled_conditions: list[float] = []
        b_raw_conditions: list[float] = []
        b_scaled_conditions: list[float] = []
        hessian_conditions: list[float] = []
        fixed_step_products: list[float] = []
        qdd_scale = torch.tensor(_QDD_SCALE, dtype=torch.float32, device=env.device)
        effort_scale = torch.tensor(_EFFORT_SCALE, dtype=torch.float32, device=env.device)
        sx = torch.diag(state_scale)
        sx_inverse = torch.diag(1.0 / state_scale)
        su = torch.diag(effort_scale)
        for clip in range(len(env.reference_bank.clip_ids)):
            for frame in range(env.reference_bank.frame_count):
                a = torch.tensor(model["A"][clip, frame], device=env.device)
                b = torch.tensor(model["B"][clip, frame], device=env.device)
                _setup_state(env, torch, clip=clip, frame=frame)
                mass = generalized_mass_matrix(env._robot)
                wrist_ids = torch.tensor(env._virtual_wrist_joint_ids, device=env.device)
                mww = mass.index_select(1, wrist_ids).index_select(2, wrist_ids)[0]
                mww_scaled = torch.diag(1.0 / effort_scale) @ mww @ torch.diag(qdd_scale)
                a_scaled = sx_inverse @ a @ sx
                b_scaled = sx_inverse @ b @ su
                _, g = env._mpc_wrist_controller._lifted_dynamics(a[None], b[None], profile.horizon)
                q_block = torch.block_diag(*([q] * profile.horizon))[None]
                r_block = torch.block_diag(*([r] * profile.horizon))[None]
                hessian = g.mT @ q_block @ g + r_block
                eigen_max = torch.linalg.eigvalsh(hessian)[0].amax()
                fixed_step_product = profile.projected_gradient_step * _scalar(eigen_max)
                mww_raw_conditions.append(_scalar(torch.linalg.cond(mww)))
                mww_scaled_conditions.append(_scalar(torch.linalg.cond(mww_scaled)))
                a_raw_conditions.append(_scalar(torch.linalg.cond(a)))
                a_scaled_conditions.append(_scalar(torch.linalg.cond(a_scaled)))
                b_raw_conditions.append(_scalar(torch.linalg.cond(b)))
                b_scaled_conditions.append(_scalar(torch.linalg.cond(b_scaled)))
                hessian_conditions.append(_scalar(torch.linalg.cond(hessian)[0]))
                fixed_step_products.append(fixed_step_product)
                model_holdout = _paired_model_holdout(
                    env,
                    torch,
                    clip=clip,
                    frame=frame,
                    a=a,
                    b=b,
                    state_delta=state_delta,
                    effort_delta=effort_delta,
                    state_scale=state_scale,
                )
                record: dict[str, Any] = {
                    "clip": env.reference_bank.clip_ids[clip],
                    "frame": frame,
                    "model_holdout": model_holdout,
                    "conditions": {
                        "A_raw": a_raw_conditions[-1],
                        "A_unit_scaled": a_scaled_conditions[-1],
                        "B_raw": b_raw_conditions[-1],
                        "B_unit_scaled": b_scaled_conditions[-1],
                        "M_ww_raw": mww_raw_conditions[-1],
                        "M_ww_unit_scaled": mww_scaled_conditions[-1],
                        "H": hessian_conditions[-1],
                        "fixed_gradient_step_times_lambda_max": fixed_step_product,
                    },
                }
                if frame < env.reference_bank.frame_count - 1:
                    record["reference_interval_feedforward"] = _reference_endpoint_holdout(
                        env,
                        torch,
                        clip=clip,
                        frame=frame,
                        state_scale=state_scale,
                    )
                records.append(record)
        one_step = _aggregate(records, "1")
        six_step = _aggregate(records, "6")
        fixed_feedforward = [
            record["reference_interval_feedforward"]["fixed_reset_bias"]["normalized_rmse"]
            for record in records
            if "reference_interval_feedforward" in record
        ]
        live_feedforward = [
            record["reference_interval_feedforward"]["live_physx_bias"]["normalized_rmse"]
            for record in records
            if "reference_interval_feedforward" in record
        ]
        report.update(
            {
                "status": "C3R4_LOCAL_DYNAMICS_HOLDOUT_COMPLETE",
                "scale_contract": {
                    "q": {
                        "translation_m": 0.01,
                        "rotation_rad": math.radians(5.0),
                    },
                    "qd": {
                        "translation_mps": 0.01 / _CONTROL_DT_S,
                        "rotation_radps": math.radians(5.0) / _CONTROL_DT_S,
                    },
                    "qdd": {
                        "translation_mps2": 0.01 / (_CONTROL_DT_S**2),
                        "rotation_radps2": math.radians(5.0) / (_CONTROL_DT_S**2),
                    },
                    "effort": {
                        "prismatic_force_n": 500.0,
                        "revolute_torque_nm": 500.0,
                    },
                    "note": (
                        "Force and torque share a numeric bound but remain distinct physical units."
                    ),
                },
                "holdout": {
                    "not_used_for_identification": True,
                    "combined_non_axial_state_fraction": _STATE_FRACTION,
                    "combined_non_axial_effort_fraction": _EFFORT_FRACTION,
                    "one_physics_step": one_step,
                    "six_physics_substeps_20hz_duration": six_step,
                },
                "diagnostic_thresholds_frozen_before_run": {
                    "one_step_normalized_rmse_max": 0.05,
                    "six_step_normalized_rmse_max": 0.25,
                    "projected_gradient_stability": "step_times_lambda_max_less_than_2",
                },
                "diagnostic_gates": {
                    "one_step": bool(one_step["finite"] and one_step["normalized_rmse"] <= 0.05),
                    "six_step": bool(six_step["finite"] and six_step["normalized_rmse"] <= 0.25),
                    "fixed_projected_gradient_step": bool(max(fixed_step_products) < 2.0),
                },
                "conditioning": {
                    "A_raw": _condition_summary(a_raw_conditions),
                    "A_unit_scaled": _condition_summary(a_scaled_conditions),
                    "B_raw": _condition_summary(b_raw_conditions),
                    "B_unit_scaled": _condition_summary(b_scaled_conditions),
                    "M_ww_raw": _condition_summary(mww_raw_conditions),
                    "M_ww_unit_scaled": _condition_summary(mww_scaled_conditions),
                    "H": _condition_summary(hessian_conditions),
                    "fixed_gradient_step_times_lambda_max": _condition_summary(fixed_step_products),
                },
                "bias_contract": {
                    "live_source": (
                        "get_coriolis_and_centrifugal_compensation_forces + "
                        "get_gravity_compensation_forces"
                    ),
                    "conditioned_on_live_q_qdot_each_substep": True,
                    "fixed_reset_bias_feedforward_normalized_rmse": math.sqrt(
                        sum(value * value for value in fixed_feedforward) / len(fixed_feedforward)
                    ),
                    "live_bias_feedforward_normalized_rmse": math.sqrt(
                        sum(value * value for value in live_feedforward) / len(live_feedforward)
                    ),
                    "gravity_expected_zero": True,
                },
                "records": records,
                "setup_state_writes_only": True,
                "rollout_wrist_state_writes": 0,
                "rollout_object_state_writes": 0,
            }
        )
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "C3R4_LOCAL_DYNAMICS_HOLDOUT_EXCEPTION",
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
