#!/usr/bin/env python3
"""Qualify task contact-to-momentum causality with the retimed active wrist."""

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

_FORCE_THRESHOLD_N = 1.0e-4
_MOMENTUM_EPSILON = 1.0e-7
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
    parser.add_argument("--child-case", choices=("no_contact", "hocap_170105", "hocap_170650"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _norm(value: list[float]) -> float:
    return math.sqrt(sum(component * component for component in value))


def _sample_wrist(env: Any, torch: Any) -> tuple[float, float, bool]:
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
        bool(torch.isfinite(env._get_observations()["policy"]).all()),
    )


def _interval(env: Any, action: Any) -> None:
    env._pre_physics_step(action)
    for _ in range(env.cfg.decimation):
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env._record_completed_contact_substep()
    env._reference_index.copy_(env._target_reference_index)


def _run_child(args: argparse.Namespace) -> int:
    if not args.accept_eula or args.child_case is None:
        raise SystemExit("child mode requires --accept-eula and --child-case")
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
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        env.reset(seed=20260804)
        clip_index = 0 if args.child_case != "hocap_170650" else 1
        env._clip_index.fill_(clip_index)
        env._reset_idx(torch.arange(env.num_envs, device=env.device))
        if args.child_case == "no_contact":
            inactive_position = env.scene.env_origins + torch.tensor(
                env.cfg.inactive_object_scene_offset, dtype=torch.float32, device=env.device
            )
            inactive_170105 = torch.cat(
                (
                    inactive_position,
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(1, -1),
                    torch.zeros((1, 6), device=env.device),
                ),
                dim=-1,
            )
            inactive_170650 = inactive_170105.clone()
            inactive_170650[:, 0] += 1.0
            env._object_170105.write_root_state_to_sim(inactive_170105)
            env._object_170650.write_root_state_to_sim(inactive_170650)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.physics_dt)
            intervals = 20
        else:
            intervals = env.reference_bank.frame_count - 1
        action = torch.zeros((1, 26), device=env.device)
        initial_position, initial_rotation, finite = _sample_wrist(env, torch)
        position_max = initial_position
        rotation_max = initial_rotation
        position_squared = initial_position * initial_position
        rotation_squared = initial_rotation * initial_rotation
        for _ in range(intervals):
            _interval(env, action)
            position, rotation, finite_now = _sample_wrist(env, torch)
            position_max = max(position_max, position)
            rotation_max = max(rotation_max, rotation)
            position_squared += position * position
            rotation_squared += rotation * rotation
            finite = finite and finite_now
        records = env.contact_substep_records
        sample_count = intervals + 1
        result = {
            "case": args.child_case,
            "clip": None if args.child_case == "no_contact" else args.child_case,
            "profile": args.profile,
            "reference_time_scale": args.reference_time_scale,
            "source_keyframes": 41,
            "retimed_control_steps": env.reference_bank.frame_count,
            "control_intervals": intervals,
            "physics_substeps": intervals * env.cfg.decimation,
            "finite": finite,
            "wrist": {
                "max_position_m": position_max,
                "position_rmse_m": math.sqrt(position_squared / sample_count),
                "max_rotation_deg": rotation_max,
                "rotation_rmse_deg": math.sqrt(rotation_squared / sample_count),
                "force_saturation_ratio": float(
                    env._force_saturation_substeps.max().detach().cpu()
                    / env._wrist_substeps.max().clamp_min(1).detach().cpu()
                ),
                "torque_saturation_ratio": float(
                    env._torque_saturation_substeps.max().detach().cpu()
                    / env._wrist_substeps.max().clamp_min(1).detach().cpu()
                ),
            },
            "contact_records": records,
            "contact_sensor_contract": env.contact_sensor_contract(),
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


def _run_worker(args: argparse.Namespace, case: str, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--accept-eula",
        "--reference-time-scale",
        str(args.reference_time_scale),
        "--profile",
        args.profile,
        "--child-case",
        case,
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
        raise RuntimeError(f"C3_RETIMED_CONTACT_WORKER_FAILURE: {case}")
    return json.loads(output.read_text(encoding="utf-8"))


def _wrist_pass(report: dict[str, Any]) -> bool:
    wrist = report["wrist"]
    return bool(
        report["finite"]
        and wrist["max_position_m"] <= _WRIST_GATE["position_max_m"]
        and wrist["position_rmse_m"] <= _WRIST_GATE["position_rmse_m"]
        and wrist["max_rotation_deg"] <= _WRIST_GATE["rotation_max_deg"]
        and wrist["rotation_rmse_deg"] <= _WRIST_GATE["rotation_rmse_deg"]
        and wrist["force_saturation_ratio"] <= _WRIST_GATE["saturation"]
        and wrist["torque_saturation_ratio"] <= _WRIST_GATE["saturation"]
    )


def _summarize_clip(report: dict[str, Any], noise_delta_v: float) -> dict[str, Any]:
    records = report["contact_records"]
    contacts = [
        record
        for record in records
        if record["contact_count"] > 0
        and _norm(record["net_contact_force_world_on_object_n"]) > _FORCE_THRESHOLD_N
    ]
    causal = [
        record
        for record in contacts
        if _norm(record["object_delta_v_world_mps"]) > noise_delta_v + _MOMENTUM_EPSILON
        or _norm(record["object_delta_omega_world_radps"]) > _MOMENTUM_EPSILON
    ]
    contract = report["contract"]
    no_hidden_control = (
        contract["wrist_root_state_writes_during_step"] == 0
        and contract["object_rollout_state_writes"] == 0
    )
    passes = {
        "complete_retimed_reference": report["control_intervals"]
        == report["retimed_control_steps"] - 1,
        "finite": report["finite"],
        "wrist_gate": _wrist_pass(report),
        "contact_event": bool(contacts),
        "finite_nonzero_force_and_impulse": bool(contacts)
        and all(
            math.isfinite(_norm(record["net_contact_force_world_on_object_n"]))
            and _norm(record["impulse_world_on_object_ns"]) > 0.0
            for record in contacts
        ),
        "contact_coincides_with_momentum_change": bool(causal),
        "no_rollout_state_write_or_hidden_force": no_hidden_control,
    }
    first_contact = contacts[0] if contacts else None
    first_causal = causal[0] if causal else None
    return {
        "status": "C3_CONTACT_CAUSALITY_VALIDATED"
        if all(passes.values())
        else "C3_CONTACT_CAUSALITY_BLOCKED",
        "clip": report["clip"],
        "passes": passes,
        "wrist": report["wrist"],
        "contact_record_count": len(contacts),
        "causal_record_count": len(causal),
        "peak_contact_force_n": max(
            [_norm(record["net_contact_force_world_on_object_n"]) for record in contacts] or [0.0]
        ),
        "peak_impulse_ns": max(
            [_norm(record["impulse_world_on_object_ns"]) for record in contacts] or [0.0]
        ),
        "peak_delta_v_mps": max(
            [_norm(record["object_delta_v_world_mps"]) for record in records] or [0.0]
        ),
        "peak_delta_omega_radps": max(
            [_norm(record["object_delta_omega_world_radps"]) for record in records] or [0.0]
        ),
        "first_contact": first_contact,
        "first_causal_contact": first_causal,
        "angular_causality_precision": (
            "approximate_from_aggregate_force; contact point unavailable"
        ),
        "contract": contract,
    }


def _orchestrate(args: argparse.Namespace) -> int:
    worker_root = args.output.parent / f".{args.output.stem}_workers"
    if args.output.exists() or worker_root.exists():
        raise FileExistsError(
            f"C3_RETIMED_CONTACT_REFUSES_OVERWRITE: output={args.output} workers={worker_root}"
        )
    worker_root.mkdir(parents=True)
    no_contact = _run_worker(args, "no_contact", worker_root / "no_contact.json")
    no_contact_records = no_contact["contact_records"]
    noise_force = max(
        [_norm(record["net_contact_force_world_on_object_n"]) for record in no_contact_records]
        or [0.0]
    )
    noise_delta_v = max(
        [_norm(record["object_delta_v_world_mps"]) for record in no_contact_records] or [0.0]
    )
    clip_reports = []
    for clip in ("hocap_170105", "hocap_170650"):
        raw = _run_worker(args, clip, worker_root / f"{clip}.json")
        clip_reports.append(_summarize_clip(raw, noise_delta_v))
    baseline_pass = noise_force <= _FORCE_THRESHOLD_N and noise_delta_v <= _MOMENTUM_EPSILON
    status = (
        "C3_CONTACT_CAUSALITY_VALIDATED"
        if baseline_pass
        and all(report["status"] == "C3_CONTACT_CAUSALITY_VALIDATED" for report in clip_reports)
        else "C3_CONTACT_CAUSALITY_BLOCKED"
    )
    result = {
        "status": status,
        "controller": "finite_virtual_6d_wrist_actuator_v1",
        "profile": args.profile,
        "reference_time_scale": args.reference_time_scale,
        "contact_readout_prerequisite": "C3_CONTACT_READOUT_VALIDATED",
        "no_contact_baseline": {
            "max_force_n": noise_force,
            "max_delta_v_mps": noise_delta_v,
            "pass": baseline_pass,
            "physics_substeps": no_contact["physics_substeps"],
        },
        "clips": clip_reports,
        "gate": _WRIST_GATE,
        "no_contact_force_threshold_n": _FORCE_THRESHOLD_N,
        "momentum_epsilon": _MOMENTUM_EPSILON,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "C3_CONTACT_CAUSALITY_VALIDATED" else 1


def main() -> int:
    args = parse_args()
    if args.reference_time_scale < 1:
        raise SystemExit("--reference-time-scale must be positive")
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required")
    return _run_child(args) if args.child_case else _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
