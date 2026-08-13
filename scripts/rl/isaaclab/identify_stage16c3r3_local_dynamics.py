#!/usr/bin/env python3
"""GPU finite-difference local A/B identification for the explicit wrist."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r3_joint_dynamics_c5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--model-output", type=Path, default=REPORT_ROOT / "identified_local_dynamics_v1.npz"
    )
    parser.add_argument(
        "--report-output", type=Path, default=REPORT_ROOT / "local_dynamics_identification.json"
    )
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _set_reference_state(env: Any, torch: Any, *, clip: int, frame: int) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device)
    env._clip_index.fill_(clip)
    env._reset_idx(env_ids)
    env._reference_index.fill_(frame)
    env._target_reference_index.fill_(frame)
    reference = env._explicit_joint_reference
    q_wrist = reference.q_wrist_ref[clip, frame].expand(env.num_envs, -1)
    qd_wrist = reference.qd_wrist_ref[clip, frame].expand(env.num_envs, -1)
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
    inactive_position = env.scene.env_origins + torch.tensor(
        env.cfg.inactive_object_scene_offset, device=env.device, dtype=torch.float32
    )
    inactive_state = torch.cat(
        (
            inactive_position,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1),
            torch.zeros((env.num_envs, 6), device=env.device),
        ),
        dim=-1,
    )
    env._object_170105.write_root_state_to_sim(inactive_state)
    env._object_170650.write_root_state_to_sim(inactive_state)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def _state(env: Any, torch: Any) -> Any:
    return torch.cat(
        (
            env._robot.data.joint_pos[:, env._virtual_wrist_joint_ids],
            env._robot.data.joint_vel[:, env._virtual_wrist_joint_ids],
        ),
        dim=-1,
    )


def _step(env: Any, torch: Any, effort: Any) -> Any:
    env._robot.set_joint_effort_target(effort, joint_ids=env._virtual_wrist_joint_ids)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    return _state(env, torch)


def _paired_columns(
    env: Any,
    torch: Any,
    *,
    clip: int,
    frame: int,
    kind: str,
    magnitude: float,
) -> tuple[Any, Any, Any, Any]:
    """Return finite-difference columns, central response, and pair residuals."""

    _set_reference_state(env, torch, clip=clip, frame=frame)
    start = _state(env, torch)
    zero = torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)
    if kind == "effort":
        basis = torch.eye(6, device=env.device).repeat_interleave(2, dim=0)
        sign = torch.tensor([1.0, -1.0], device=env.device).repeat(6)[:, None]
        next_state = _step(env, torch, sign * magnitude * basis)
        delta = sign * magnitude * basis
    elif kind in {"position", "velocity"}:
        source = 0 if kind == "position" else 6
        perturbation = torch.eye(6, device=env.device).repeat_interleave(2, dim=0)
        sign = torch.tensor([1.0, -1.0], device=env.device).repeat(6)[:, None]
        shifted = start.clone()
        shifted[:, source : source + 6] += sign * magnitude * perturbation
        env._robot.write_joint_state_to_sim(
            shifted[:, :6], shifted[:, 6:], joint_ids=env._virtual_wrist_joint_ids
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.physics_dt)
        next_state = _step(env, torch, zero)
        delta = sign * magnitude * perturbation
    else:
        raise ValueError(kind)
    _set_reference_state(env, torch, clip=clip, frame=frame)
    central = _step(env, torch, zero)[0]
    plus = next_state[0::2]
    minus = next_state[1::2]
    columns = ((plus - minus) / (2.0 * magnitude)).mT
    plus_delta = delta[0::2]
    predicted_plus = central[None] + plus_delta @ columns.mT
    predicted_minus = central[None] - plus_delta @ columns.mT
    residual = torch.cat((plus - predicted_plus, minus - predicted_minus), dim=0)
    actual = torch.cat((plus - central, minus - central), dim=0)
    predicted = torch.cat((predicted_plus - central, predicted_minus - central), dim=0)
    return columns, residual, actual, predicted


def _identify(env: Any, torch: Any) -> dict[str, Any]:
    from toporetarget.rl.environments.isaaclab_backend.tvlqr_wrist import (
        ExplicitWristLocalDynamicsIdentifierV1,
    )

    profile = ExplicitWristLocalDynamicsIdentifierV1()
    clips = len(env.reference_bank.clip_ids)
    frames = env.reference_bank.frame_count
    a = torch.empty((clips, frames, 12, 12), device=env.device)
    b = torch.empty((clips, frames, 12, 6), device=env.device)
    residuals: list[Any] = []
    actuals: list[Any] = []
    predictions: list[Any] = []
    controllability_conditions: list[Any] = []
    for clip in range(clips):
        for frame in range(frames):
            a_q, residual, actual, predicted = _paired_columns(
                env,
                torch,
                clip=clip,
                frame=frame,
                kind="position",
                magnitude=profile.state_position_perturbation,
            )
            a_qd, residual_qd, actual_qd, predicted_qd = _paired_columns(
                env,
                torch,
                clip=clip,
                frame=frame,
                kind="velocity",
                magnitude=profile.state_velocity_perturbation,
            )
            b_frame, residual_b, actual_b, predicted_b = _paired_columns(
                env,
                torch,
                clip=clip,
                frame=frame,
                kind="effort",
                magnitude=profile.effort_perturbation,
            )
            a[clip, frame] = torch.cat((a_q, a_qd), dim=1)
            b[clip, frame] = b_frame
            residuals.extend((residual, residual_qd, residual_b))
            actuals.extend((actual, actual_qd, actual_b))
            predictions.extend((predicted, predicted_qd, predicted_b))
            blocks = [b_frame]
            propagated = b_frame
            for _ in range(11):
                propagated = a[clip, frame] @ propagated
                blocks.append(propagated)
            controllability_conditions.append(torch.linalg.cond(torch.cat(blocks, dim=1)))
    residual = torch.cat(residuals)
    actual = torch.cat(actuals)
    predicted = torch.cat(predictions)
    residual_ss = torch.square(actual - predicted).sum()
    total_ss = torch.square(actual - actual.mean(dim=0, keepdim=True)).sum().clamp_min(1.0e-12)
    r2 = 1.0 - residual_ss / total_ss
    covariance = residual.mT @ residual / max(residual.shape[0] - 1, 1)
    eigenvalue = torch.linalg.eigvals(a).abs().amax(dim=-1)
    return {
        "identifier": profile.identifier,
        "A": a,
        "B": b,
        "report": {
            "identifier": profile.identifier,
            "state": profile.state,
            "action": profile.action,
            "num_envs": profile.num_envs,
            "frames": frames,
            "clips": list(env.reference_bank.clip_ids),
            "perturbations": {
                "position": profile.state_position_perturbation,
                "velocity": profile.state_velocity_perturbation,
                "effort": profile.effort_perturbation,
            },
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "residual_covariance": covariance.detach().cpu().tolist(),
            "r2": _scalar(r2),
            "controllability_condition_max": _scalar(
                torch.stack(controllability_conditions).amax()
            ),
            "spectral_radius_max": _scalar(eigenvalue.amax()),
            "numeric_stable": bool(torch.isfinite(eigenvalue).all() and eigenvalue.amax() < 1.1),
            "setup_state_writes_only": True,
            "rollout_state_writes": 0,
            "object_rollout_state_writes": 0,
        },
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.model_output.exists() or args.report_output.exists():
        raise FileExistsError("C3R3_IDENTIFICATION_REFUSES_OVERWRITE")
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
            configure_bounded_tvlqr_wrist,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 12
        cfg.scene.lazy_sensor_update = True
        cfg.contact_telemetry = "off"
        cfg.balanced_clip_assignment = False
        configure_bounded_tvlqr_wrist(cfg)
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        result = _identify(env, torch)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.model_output,
            A=result["A"].detach().cpu().numpy(),
            B=result["B"].detach().cpu().numpy(),
        )
        report = {
            "status": "C3R3_GPU_LOCAL_DYNAMICS_IDENTIFICATION_COMPLETE",
            "model_path": str(args.model_output),
            **result["report"],
        }
        args.report_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(report["status"])
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
