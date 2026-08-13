#!/usr/bin/env python3
"""Identify unit-scaled affine wrist dynamics at every reference substep."""

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
_EFFORT_SCALE = np.asarray([500.0] * 6, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--model-output", type=Path, default=REPORT_ROOT / "identified_local_dynamics_v2.npz"
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPORT_ROOT / "local_dynamics_identification_v2.json",
    )
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _hadamard(order: int) -> np.ndarray:
    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


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


def _node_samples(env: Any, torch: Any, *, clip: int, frame: int, substep: int) -> tuple[Any, Any]:
    reference = env._explicit_joint_reference
    clip_index = torch.full((env.num_envs,), clip, dtype=torch.long, device=env.device)
    key_index = torch.full((env.num_envs,), frame, dtype=torch.long, device=env.device)
    current = reference.sample(
        clip_index, key_index, substep=substep, decimation=env.cfg.decimation
    )
    following = reference.sample(
        clip_index,
        key_index,
        substep=substep + 1,
        decimation=env.cfg.decimation,
    )
    return current, following


def _setup_node(
    env: Any,
    torch: Any,
    *,
    clip: int,
    frame: int,
    substep: int,
    state_delta: Any,
) -> tuple[Any, Any]:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._clip_index.fill_(clip)
    env._reset_idx(env_ids)
    env._reference_index.fill_(frame)
    env._target_reference_index.fill_(frame + 1)
    current, following = _node_samples(env, torch, clip=clip, frame=frame, substep=substep)
    q_wrist = current.q_wrist.clone() + state_delta[:, :6]
    qd_wrist = current.qd_wrist.clone() + state_delta[:, 6:]
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
    env._robot.set_joint_velocity_target(
        env.action_adapter.canonical_to_isaac(current.qd_finger),
        joint_ids=env._finger_target_joint_ids,
    )
    _inactive_objects(env, torch)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    return current, following


def _nominal_effort(env: Any, current: Any, torch: Any) -> tuple[Any, Any, Any]:
    from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
        generalized_bias_compensation,
        generalized_mass_matrix,
    )

    mass = generalized_mass_matrix(env._robot)
    bias = generalized_bias_compensation(env._robot)
    wrist_ids = torch.tensor(env._virtual_wrist_joint_ids, device=env.device)
    finger_ids = torch.tensor(env._finger_target_joint_ids, device=env.device)
    central = env.num_envs - 1
    mww = mass[central].index_select(0, wrist_ids).index_select(1, wrist_ids)
    mwf = mass[central].index_select(0, wrist_ids).index_select(1, finger_ids)
    qdd_finger = env.action_adapter.canonical_to_isaac(current.qdd_finger[central : central + 1])[0]
    effort = mww @ current.qdd_wrist[central] + mwf @ qdd_finger + bias[central, wrist_ids]
    if not bool(torch.isfinite(effort).all()):
        raise RuntimeError("C3R4_V2_NOMINAL_EFFORT_NONFINITE")
    return effort, mww, bias[central, wrist_ids]


def _identify_node(
    env: Any,
    torch: Any,
    *,
    clip: int,
    frame: int,
    substep: int,
    design: Any,
    state_scale: Any,
    effort_scale: Any,
    profile: Any,
) -> dict[str, Any]:
    directions = design.shape[0]
    z_plus = design.clone()
    z_plus[:, :12] *= profile.state_scale_fraction
    z_plus[:, 12:] *= profile.effort_scale_fraction
    z = torch.cat((z_plus, -z_plus, torch.zeros((1, 18), device=env.device)), dim=0)
    state_delta = z[:, :12] * state_scale
    effort_delta = z[:, 12:] * effort_scale
    current, following = _setup_node(
        env,
        torch,
        clip=clip,
        frame=frame,
        substep=substep,
        state_delta=state_delta,
    )
    nominal_effort, mww, live_bias = _nominal_effort(env, current, torch)
    effort = nominal_effort[None] + effort_delta
    if bool((effort.abs() > 500.0).any()):
        raise RuntimeError("C3R4_V2_IDENTIFICATION_EFFORT_OUTSIDE_FROZEN_BOX")
    env._robot.set_joint_effort_target(effort, joint_ids=env._virtual_wrist_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    next_reference = torch.cat((following.q_wrist, following.qd_wrist), dim=-1)
    next_error = _state(env, torch) - next_reference
    paired_response = (next_error[:directions] - next_error[directions : 2 * directions]) / 2.0
    paired_response_normalized = paired_response / state_scale
    gram = z_plus.mT @ z_plus
    jacobian_transpose = torch.linalg.solve(gram, z_plus.mT @ paired_response_normalized)
    jacobian_normalized = jacobian_transpose.mT
    sx = torch.diag(state_scale)
    sx_inverse = torch.diag(1.0 / state_scale)
    su_inverse = torch.diag(1.0 / effort_scale)
    a = sx @ jacobian_normalized[:, :12] @ sx_inverse
    b = sx @ jacobian_normalized[:, 12:] @ su_inverse
    predicted = z_plus @ jacobian_transpose
    residual = paired_response_normalized - predicted
    actual_ss = torch.square(paired_response_normalized).sum()
    residual_ss = torch.square(residual).sum()
    r2 = 1.0 - residual_ss / actual_ss.clamp_min(1.0e-12)
    c = next_error[-1]
    return {
        "A": a,
        "B": b,
        "C": c,
        "U_NOMINAL": nominal_effort,
        "r2": r2,
        "normalized_residual": residual,
        "normalized_actual": paired_response_normalized,
        "M_ww_condition": torch.linalg.cond(mww),
        "A_condition": torch.linalg.cond(jacobian_normalized[:, :12]),
        "B_condition": torch.linalg.cond(jacobian_normalized[:, 12:]),
        "bias_max_abs": live_bias.abs().amax(),
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("C3R4_V2_IDENTIFICATION_REFUSES_OVERWRITE")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    report: dict[str, Any] = {"status": "C3R4_V2_IDENTIFICATION_NOT_STARTED"}
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.tvlqr_wrist import (
            ExplicitWristLocalDynamicsIdentifierV2,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
            configure_bounded_mpc_wrist,
        )

        profile = ExplicitWristLocalDynamicsIdentifierV2()
        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = profile.num_envs
        cfg.scene.lazy_sensor_update = True
        cfg.contact_telemetry = "off"
        cfg.balanced_clip_assignment = False
        configure_bounded_mpc_wrist(cfg)
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        clips = len(env.reference_bank.clip_ids)
        intervals = env.reference_bank.frame_count - 1
        substeps = cfg.decimation
        a = torch.empty((clips, intervals, substeps, 12, 12), device=env.device)
        b = torch.empty((clips, intervals, substeps, 12, 6), device=env.device)
        c = torch.empty((clips, intervals, substeps, 12), device=env.device)
        u_nominal = torch.empty((clips, intervals, substeps, 6), device=env.device)
        design = torch.tensor(
            _hadamard(profile.direction_count)[:, :18],
            dtype=torch.float32,
            device=env.device,
        )
        state_scale = torch.tensor(_STATE_SCALE, dtype=torch.float32, device=env.device)
        effort_scale = torch.tensor(_EFFORT_SCALE, dtype=torch.float32, device=env.device)
        residuals: list[Any] = []
        actuals: list[Any] = []
        r2_nodes: list[Any] = []
        mww_conditions: list[Any] = []
        a_conditions: list[Any] = []
        b_conditions: list[Any] = []
        bias_values: list[Any] = []
        for clip in range(clips):
            for frame in range(intervals):
                for substep in range(substeps):
                    node = _identify_node(
                        env,
                        torch,
                        clip=clip,
                        frame=frame,
                        substep=substep,
                        design=design,
                        state_scale=state_scale,
                        effort_scale=effort_scale,
                        profile=profile,
                    )
                    a[clip, frame, substep] = node["A"]
                    b[clip, frame, substep] = node["B"]
                    c[clip, frame, substep] = node["C"]
                    u_nominal[clip, frame, substep] = node["U_NOMINAL"]
                    residuals.append(node["normalized_residual"])
                    actuals.append(node["normalized_actual"])
                    r2_nodes.append(node["r2"])
                    mww_conditions.append(node["M_ww_condition"])
                    a_conditions.append(node["A_condition"])
                    b_conditions.append(node["B_condition"])
                    bias_values.append(node["bias_max_abs"])
                print(
                    f"C3R4_ID_V2 clip={env.reference_bank.clip_ids[clip]} "
                    f"interval={frame + 1}/{intervals}",
                    flush=True,
                )
        residual = torch.cat(residuals)
        actual = torch.cat(actuals)
        global_r2 = 1.0 - torch.square(residual).sum() / torch.square(actual).sum().clamp_min(
            1.0e-12
        )
        report = {
            "status": "C3R4_GPU_SUBSTEP_AFFINE_DYNAMICS_IDENTIFICATION_COMPLETE",
            "identifier": profile.identifier,
            "schema": "explicit_wrist_local_dynamics_v2_substep_affine",
            "model_path": str(args.model_output),
            "clips": list(env.reference_bank.clip_ids),
            "intervals": intervals,
            "physics_substeps_per_interval": substeps,
            "nodes": clips * intervals * substeps,
            "design": {
                "name": profile.design,
                "direction_count": profile.direction_count,
                "paired_environments": profile.direction_count * 2,
                "central_environments": 1,
                "num_envs": profile.num_envs,
                "state_scale_fraction": profile.state_scale_fraction,
                "effort_scale_fraction": profile.effort_scale_fraction,
                "holdout_direction_excluded": True,
            },
            "scale_contract": {
                "state": _STATE_SCALE.tolist(),
                "effort": _EFFORT_SCALE.tolist(),
                "force_and_torque_units_kept_distinct": True,
            },
            "fit": {
                "normalized_global_r2": _scalar(global_r2),
                "node_r2_min": _scalar(torch.stack(r2_nodes).amin()),
                "node_r2_median": _scalar(torch.stack(r2_nodes).median()),
                "normalized_residual_rmse": _scalar(torch.sqrt(torch.square(residual).mean())),
            },
            "conditioning": {
                "M_ww_max": _scalar(torch.stack(mww_conditions).amax()),
                "M_ww_median": _scalar(torch.stack(mww_conditions).median()),
                "A_unit_scaled_max": _scalar(torch.stack(a_conditions).amax()),
                "B_unit_scaled_max": _scalar(torch.stack(b_conditions).amax()),
            },
            "affine": {
                "C_max_abs": _scalar(c.abs().amax()),
                "nominal_effort_max_abs": _scalar(u_nominal.abs().amax()),
                "live_bias_max_abs_at_reference": _scalar(torch.stack(bias_values).amax()),
            },
            "finite": bool(
                torch.isfinite(a).all()
                and torch.isfinite(b).all()
                and torch.isfinite(c).all()
                and torch.isfinite(u_nominal).all()
            ),
            "setup_state_writes_only": True,
            "rollout_wrist_state_writes": 0,
            "rollout_object_state_writes": 0,
        }
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.model_output,
            A=a.detach().cpu().numpy(),
            B=b.detach().cpu().numpy(),
            C=c.detach().cpu().numpy(),
            U_NOMINAL=u_nominal.detach().cpu().numpy(),
        )
        args.report_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(report["status"])
        return 0
    except BaseException as exc:
        report.update(
            {
                "status": "C3R4_V2_IDENTIFICATION_EXCEPTION",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
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
