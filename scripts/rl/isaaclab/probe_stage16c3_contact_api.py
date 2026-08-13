#!/usr/bin/env python3
"""Isolate Stage 16-C.3 object-centric contact telemetry in child processes.

An Isaac process can close without Python receiving a useful exception.  The
parent therefore qualifies a probe only when its child exits cleanly *and*
writes an ordered, flushed progress trace through the force-matrix read.
This is deliberately a contact API probe, not a C.3 semantic qualification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-envs", type=int, choices=(1, 128), default=1)
    parser.add_argument(
        "--steps", type=int, default=2, help="Control steps when --physics-steps is absent."
    )
    parser.add_argument(
        "--physics-steps",
        type=int,
        help="Minimum physics steps; the probe rounds up to complete control decimation.",
    )
    parser.add_argument(
        "--clone-mode",
        choices=("frozen_fabric", "usd_clone"),
        default="frozen_fabric",
        help=(
            "Probe cloning choice; usd_clone disables Fabric cloning without "
            "changing physics replication."
        ),
    )
    parser.add_argument(
        "--case",
        choices=("settled_no_contact", "single_finger_preload", "random_actions"),
        default="settled_no_contact",
    )
    parser.add_argument(
        "--fixture-object",
        choices=("Object170105", "Object170650"),
        default="Object170105",
        help="Object used only by the pre-rollout single-finger diagnostic fixture.",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--events", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c3r2_c5/contact/c1_probe.json",
    )
    return parser.parse_args()


def _event(path: Path, stage: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "time_unix_s": time.time(), **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _force_matrix_summary(force_matrix: Any, torch: Any) -> dict[str, Any]:
    if force_matrix is None:
        return {"shape": None, "finite": None, "max_pair_force_n": None, "present_pairs": None}
    pair_norm = torch.linalg.vector_norm(force_matrix, dim=-1)
    slots = torch.unique(torch.nonzero(pair_norm > 1.0e-4, as_tuple=False)[..., -1])
    return {
        "shape": list(force_matrix.shape),
        "finite": bool(torch.isfinite(force_matrix).all()),
        "max_pair_force_n": float(pair_norm.amax().detach().cpu()),
        "present_pairs": int((pair_norm > 1.0e-4).sum().item()),
        "present_filter_slots": slots.detach().cpu().tolist(),
    }


def _sensor_force_summary(env: Any, torch: Any) -> dict[str, dict[str, Any]]:
    return {
        object_name: _force_matrix_summary(sensor.data.force_matrix_w, torch)
        for object_name, sensor in env._object_contact_sensors.items()
    }


def _child(args: argparse.Namespace) -> int:
    if args.events is None:
        raise SystemExit("--events is required in child mode")
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    if args.steps < 1 or (args.physics_steps is not None and args.physics_steps < 1):
        raise SystemExit("--steps must be positive")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    _event(
        args.events,
        "child_started",
        num_envs=args.num_envs,
        case=args.case,
        clone_mode=args.clone_mode,
        fixture_object=args.fixture_object,
        physics_steps_requested=args.physics_steps,
        steps=args.steps,
    )
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
        )

        _event(args.events, "app_ready")
        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.scene.clone_in_fabric = args.clone_mode == "frozen_fabric"
        cfg.balanced_clip_assignment = True
        cfg.contact_telemetry = "aggregate"
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        _event(
            args.events,
            "environment_constructed",
            object_sensor_bodies={
                object_name: list(sensor.body_names)
                for object_name, sensor in env._object_contact_sensors.items()
            },
            filter_count_per_sensor={
                object_name: sensor.contact_physx_view.filter_count
                for object_name, sensor in env._object_contact_sensors.items()
            },
        )
        observation, _ = env.reset(seed=20260803)
        inactive_position = env.scene.env_origins + torch.tensor(
            env.cfg.inactive_object_scene_offset, device=env.device
        )
        inactive_state = torch.cat(
            (
                inactive_position,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1),
                torch.zeros((env.num_envs, 6), device=env.device),
            ),
            dim=-1,
        )
        if args.case == "settled_no_contact":
            env._object_170105.write_root_state_to_sim(inactive_state)
            env._object_170650.write_root_state_to_sim(inactive_state)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            _event(
                args.events,
                "no_contact_fixture",
                object_state_write_scope="before rollout; contact API diagnostic fixture only",
            )
        elif args.case == "single_finger_preload":
            finger_body_id = env._robot.body_names.index("r_index_finger_distal")
            fixture_object = (
                env._object_170105 if args.fixture_object == "Object170105" else env._object_170650
            )
            other_object = (
                env._object_170650 if args.fixture_object == "Object170105" else env._object_170105
            )
            fixture_state = fixture_object.data.root_state_w.clone()
            fixture_state[:, :3] = env._robot.data.body_pos_w[:, finger_body_id]
            fixture_state[:, 7:].zero_()
            other_object.write_root_state_to_sim(inactive_state)
            fixture_object.write_root_state_to_sim(fixture_state)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            _event(
                args.events,
                "single_finger_preload_fixture",
                object_name=args.fixture_object,
                target_hand_body="r_index_finger_distal",
                object_state_write_scope="before rollout; contact API diagnostic fixture only",
            )
        _event(
            args.events,
            "reset_complete",
            observation_shape=list(observation["policy"].shape),
            device=str(observation["policy"].device),
            reset_reference_index=int(env._reference_index[0].item()),
        )
        _event(
            args.events,
            "force_matrix_read_after_reset",
            object_sensor_force_matrices=_sensor_force_summary(env, torch),
        )
        if args.case == "settled_no_contact":
            physics_steps = args.physics_steps or args.steps * env.cfg.decimation
            joint_target = env._robot.data.joint_pos.clone()
            for physics_step in range(physics_steps):
                env._robot.set_joint_position_target(joint_target)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                _event(
                    args.events,
                    "force_matrix_read_after_step",
                    physics_step=physics_step,
                    object_sensor_force_matrices=_sensor_force_summary(env, torch),
                    scope="raw PhysX no-contact isolation; no DirectRLEnv reset or rollout",
                )
            _event(args.events, "child_completed")
            return 0
        generator = torch.Generator(device=env.device)
        generator.manual_seed(20260803)
        control_steps = (
            args.steps
            if args.physics_steps is None
            else math.ceil(args.physics_steps / env.cfg.decimation)
        )
        for step in range(control_steps):
            if args.case == "random_actions":
                action = 0.05 * torch.rand(
                    (args.num_envs, 26), device=env.device, generator=generator
                )
            else:
                action = torch.zeros((args.num_envs, 26), device=env.device)
            _event(args.events, "before_env_step", step=step)
            observation, _, _, _, _ = env.step(action)
            _event(
                args.events,
                "force_matrix_read_after_step",
                step=step,
                object_sensor_force_matrices=_sensor_force_summary(env, torch),
                substep_records=len(env.contact_substep_records),
                observation_finite=bool(torch.isfinite(observation["policy"]).all()),
            )
        _event(args.events, "child_completed")
        return 0
    except BaseException as error:
        _event(
            args.events,
            "child_exception",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _parent(args: argparse.Namespace) -> int:
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    events = args.output.with_name(
        f"{args.output.stem}_{args.case}_{args.num_envs}env.events.jsonl"
    )
    stdout = events.with_suffix(".stdout.txt")
    stderr = events.with_suffix(".stderr.txt")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in (events, stdout, stderr):
        path.unlink(missing_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--num-envs",
        str(args.num_envs),
        "--steps",
        str(args.steps),
        "--case",
        args.case,
        "--fixture-object",
        args.fixture_object,
        "--clone-mode",
        args.clone_mode,
        *(["--physics-steps", str(args.physics_steps)] if args.physics_steps is not None else []),
        "--events",
        str(events),
        "--accept-eula",
    ]
    started = time.monotonic()
    with (
        stdout.open("w", encoding="utf-8") as stdout_stream,
        stderr.open("w", encoding="utf-8") as stderr_stream,
    ):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
            timeout=600,
        )
    event_records = _read_events(events)
    stages = [record["stage"] for record in event_records]
    stage_counts = {stage: stages.count(stage) for stage in sorted(set(stages))}
    completed_marker = bool(stages) and stages[-1] == "child_completed"
    result = {
        "status": "STAGE16C3_CONTACT_API_PROBE_PASS"
        if completed.returncode == 0 and completed_marker
        else "STAGE16C3_CONTACT_API_PROBE_FAIL",
        "case": args.case,
        "clone_mode": args.clone_mode,
        "fixture_object": args.fixture_object,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "physics_steps_requested": args.physics_steps,
        "physics_steps_executed": (
            args.steps * 6
            if args.physics_steps is None
            else (
                args.physics_steps
                if args.case == "settled_no_contact"
                else math.ceil(args.physics_steps / 6) * 6
            )
        ),
        "child_returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "events_path": str(events),
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
        "event_count": len(event_records),
        "event_stage_counts": stage_counts,
        "completed_marker": completed_marker,
        "last_event": event_records[-1] if event_records else None,
        "qualification_scope": "C1 object-centric ContactSensor API isolation only",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "STAGE16C3_CONTACT_API_PROBE_PASS" else 1


def main() -> int:
    args = parse_args()
    return _child(args) if args.child else _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
