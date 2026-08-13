#!/usr/bin/env python3
"""Run the bounded two-profile full-articulation computed-torque gate."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / ".local/reports/stage16c3r3_joint_dynamics_c5/computed_torque_qualification.json",
    )
    parser.add_argument("--worker-profile", help=argparse.SUPPRESS)
    parser.add_argument("--worker-ablation", default="full", help=argparse.SUPPRESS)
    return parser.parse_args()


def _scalar(value: Any) -> float:
    return float(value.detach().cpu().item())


def _reset_clip(env: Any, torch: Any, clip_index: int) -> None:
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
    env.calibrate_computed_torque_bias()


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


def _interval(env: Any, action: Any, *, observe_substep: Callable[[dict[str, Any]], None]) -> None:
    env._pre_physics_step(action)
    for _ in range(env.cfg.decimation):
        env._apply_action()
        latest = env._computed_torque_latest
        if latest is None:
            raise RuntimeError("C3_COMPUTED_TORQUE_NO_SUBSTEP_EVIDENCE")
        observe_substep(latest)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env._reference_index.copy_(env._target_reference_index)


def _clip_report(env: Any, torch: Any, clip_index: int) -> dict[str, Any]:
    _reset_clip(env, torch, clip_index)
    action = torch.zeros((env.num_envs, 26), device=env.device)
    pos, rot, finite = _sample(env, torch)
    pos_sq, rot_sq = pos * pos, rot * rot
    pos_max, rot_max = pos, rot
    term_max = {
        key: 0.0
        for key in (
            "feedforward",
            "coupling",
            "bias",
            "feedback",
            "effort_command",
            "effort_applied",
        )
    }
    saturation_count = torch.zeros(6, dtype=torch.long, device=env.device)

    def observe_substep(latest: dict[str, Any]) -> None:
        for key in term_max:
            term_max[key] = max(term_max[key], _scalar(latest[key].abs().amax()))
        saturation_count.add_(latest["saturation"].any(dim=0).to(torch.long))

    for _ in range(_STEPS - 1):
        _interval(env, action, observe_substep=observe_substep)
        pos, rot, finite_now = _sample(env, torch)
        finite = finite and finite_now
        pos_max, rot_max = max(pos_max, pos), max(rot_max, rot)
        pos_sq, rot_sq = pos_sq + pos * pos, rot_sq + rot * rot
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
            configure_full_articulation_computed_torque_wrist,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = 1
        cfg.scene.lazy_sensor_update = True
        cfg.balanced_clip_assignment = False
        cfg.contact_telemetry = "off"
        cfg.computed_torque_ablation = args.worker_ablation
        configure_full_articulation_computed_torque_wrist(
            cfg, profile_identifier=args.worker_profile
        )
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        clips = [_clip_report(env, torch, index) for index in range(2)]
        result = {
            "profile": args.worker_profile,
            "ablation": args.worker_ablation,
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


def _run_worker(
    args: argparse.Namespace, profile: str, ablation: str, output: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--worker-profile",
        profile,
        "--worker-ablation",
        ablation,
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}, check=False
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"C3_COMPUTED_TORQUE_WORKER_FAILURE: {profile}/{ablation}")
    return json.loads(output.read_text(encoding="utf-8"))


def _orchestrate(args: argparse.Namespace) -> int:
    from toporetarget.rl.environments.isaaclab_backend.articulation_dynamics import (
        FULL_ARTICULATION_COMPUTED_TORQUE_PROFILES,
    )

    worker_root = args.output.parent / f".{args.output.stem}_workers"
    if args.output.exists() or worker_root.exists():
        raise FileExistsError(
            f"C3_COMPUTED_TORQUE_REFUSES_OVERWRITE: output={args.output} workers={worker_root}"
        )
    worker_root.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    selected: str | None = None
    for profile in FULL_ARTICULATION_COMPUTED_TORQUE_PROFILES:
        result = _run_worker(
            args, profile.identifier, "full", worker_root / f"{profile.identifier}_full.json"
        )
        candidates.append(result)
        if result["pass"]:
            selected = profile.identifier
            break
    ablations: dict[str, dict[str, Any]] = {}
    if selected is not None:
        for name in ("zero_effort", "feedforward_only", "feedback_only", "coupling_off"):
            ablations[name] = _run_worker(
                args, selected, name, worker_root / f"{selected}_{name}.json"
            )
    result = {
        "status": "C3_COMPUTED_TORQUE_WRIST_TRACKING_VALIDATED"
        if selected
        else "C3_COMPUTED_TORQUE_WRIST_TRACKING_EXHAUSTED",
        "controller": "full_articulation_computed_torque_v1",
        "profiles": candidates,
        "selected_profile": selected,
        "counterfactuals": ablations,
        "path_b_permitted": selected is None,
        "no_cartesian_wrench": True,
        "no_rollout_pose_or_velocity_writes": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if selected else 1


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required")
    return _worker(args) if args.worker_profile else _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
