#!/usr/bin/env python3
"""Run the frozen Stage 16-C.5A-R3 contact-topology matrix.

The parent process is deliberately Isaac-free.  Every scene cell is a fresh
headless Isaac child process under the retained G0 physics contract.  Workers
use only frame-zero resets and ordinary control steps: neither candidate-state
restoration nor object/wrist rollout state writes are available here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--identifier", help=argparse.SUPPRESS)
    parser.add_argument("--scene-env-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--active-contact-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--schedule", help=argparse.SUPPRESS)
    parser.add_argument("--shard-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R3_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_frames(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("clips") if isinstance(payload, dict) else None
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("R3 requires the frozen two-clip replication-frame report")
    for row in rows:
        frames = row.get("frames")
        if not isinstance(row.get("clip"), str) or not isinstance(frames, dict):
            raise ValueError("malformed frozen replication frame row")
        onset = frames.get("contact_onset")
        if not isinstance(onset, int) or not 0 < onset < 321:
            raise ValueError("contact onset must be an interior factor-8 runtime frame")
    return rows


def _fingerprint(tensors: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"R3 fingerprint needs tensor {name}")
        cpu = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(repr(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _max_abs(first: Any, second: Any) -> float:
    import torch

    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("R3 comparison needs tensors")
    if first.shape != second.shape:
        raise ValueError(f"R3 tensor shape mismatch: {first.shape} != {second.shape}")
    if first.dtype in {torch.bool, torch.long}:
        return float((first != second).to(torch.float32).amax().detach().cpu())
    return float((first - second).abs().amax().detach().cpu())


def _canonical_raw(env: Any, ids: Any) -> dict[str, Any]:
    """Capture per-environment live state with only the known origin offset removed."""

    from toporetarget.rl.isaaclab_oracle.candidate_state import capture_candidate_state

    tensors = capture_candidate_state(env, ids).tensors
    origins = tensors.pop("source_env_origins")
    for name in (
        "robot_root_state",
        "object_170105_root_state",
        "object_170650_root_state",
        "wrist_target_position",
    ):
        if name in tensors:
            tensors[name] = tensors[name].clone()
            tensors[name][..., :3] -= origins
    return tensors


def _measurement(env: Any, ids: Any) -> dict[str, Any]:
    import torch

    from toporetarget.rl.isaaclab_oracle.runtime import state_view

    if str(env.device).startswith("cuda"):
        torch.cuda.synchronize(env.device)
    raw = _canonical_raw(env, ids)
    derived = {
        "state": state_view(env, ids),
        "observation": env._get_observations()["policy"].index_select(0, ids).detach().clone(),
        "reward": {
            name: value.index_select(0, ids).detach().clone()
            for name, value in env._last_reward_terms.items()
        },
        "terminated": env.reset_terminated.index_select(0, ids).detach().clone(),
        "timed_out": env.reset_time_outs.index_select(0, ids).detach().clone(),
        "reason_codes": env._reason_codes.index_select(0, ids).detach().clone(),
    }
    return {
        "raw": raw,
        "derived": derived,
        "raw_fingerprint": _fingerprint(raw),
        "derived_fingerprint": _fingerprint(
            {
                **derived["state"],
                "observation": derived["observation"],
                **{f"reward.{name}": value for name, value in derived["reward"].items()},
                "terminated": derived["terminated"],
                "timed_out": derived["timed_out"],
                "reason_codes": derived["reason_codes"],
            }
        ),
    }


def _select(measurement: Mapping[str, Any], index: int) -> dict[str, Any]:
    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {name: visit(child) for name, child in value.items()}
        return value[index : index + 1]

    return {"raw": visit(measurement["raw"]), "derived": visit(measurement["derived"])}


def _compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, object]:
    import torch

    from toporetarget.rl.isaaclab_oracle.metrics import state_differences

    raw = {name: _max_abs(value, second["raw"][name]) for name, value in first["raw"].items()}
    state = state_differences(first["derived"]["state"], second["derived"]["state"])
    rewards = {
        name: _max_abs(value, second["derived"]["reward"][name])
        for name, value in first["derived"]["reward"].items()
    }
    termination_exact = bool(
        torch.equal(first["derived"]["terminated"], second["derived"]["terminated"])
        and torch.equal(first["derived"]["timed_out"], second["derived"]["timed_out"])
        and torch.equal(first["derived"]["reason_codes"], second["derived"]["reason_codes"])
    )
    return {
        "raw_max_abs": raw,
        "metrics": {**state, "reward": rewards["total"]},
        "observation_max_abs": _max_abs(
            first["derived"]["observation"], second["derived"]["observation"]
        ),
        "reward_components_max_abs": rewards,
        "termination_exact": termination_exact,
    }


def _hold_inactive_at_current_reference(env: Any, inactive_ids: Any, current_index: Any) -> None:
    """Keep T1/T3 unreleased environments at a frozen reference key.

    This changes only when a candidate starts its normal trajectory.  It does
    not edit source reference values or actions and never writes root/object
    state to PhysX.
    """

    if inactive_ids.numel() == 0:
        return
    clips = env._clip_index.index_select(0, inactive_ids)
    frames = current_index.index_select(0, inactive_ids)
    env._target_reference_index.index_copy_(0, inactive_ids, frames)
    finger = env.reference_bank.gather("q_finger_ref", clips, frames)
    env._joint_target_isaac.index_copy_(
        0, inactive_ids, env.action_adapter.canonical_to_isaac(finger)
    )
    position = env.reference_bank.gather("wrist_pose_translation_world_ref", clips, frames)
    quaternion = env.reference_bank.gather("wrist_pose_quaternion_world_ref_wxyz", clips, frames)
    twist = env.reference_bank.gather("wrist_twist_world_ref", clips, frames)
    for attribute, value in (
        ("_wrist_interval_start_position", position),
        ("_wrist_interval_end_position", position),
        ("_wrist_interval_start_quaternion", quaternion),
        ("_wrist_interval_end_quaternion", quaternion),
        ("_wrist_interval_start_twist", twist),
        ("_wrist_interval_end_twist", twist),
    ):
        getattr(env, attribute).index_copy_(0, inactive_ids, value)
    env._wrist_translation_residual.index_fill_(0, inactive_ids, 0.0)
    env._wrist_rotation_residual.index_fill_(0, inactive_ids, 0.0)


def _scheduled_control_step(env: Any, actions: Any, active_ids: Any) -> tuple[Any, Any]:
    """Exact DirectRLEnv step order with only candidate start scheduling masked."""

    import torch

    previous_index = env._reference_index.clone()
    env._pre_physics_step(actions.to(env.device))
    active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active[active_ids] = True
    inactive_ids = torch.nonzero(~active, as_tuple=False).squeeze(-1)
    _hold_inactive_at_current_reference(env, inactive_ids, previous_index)
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    terminated, timed_out = env._get_dones()
    env.reset_terminated.copy_(terminated)
    env.reset_time_outs.copy_(timed_out)
    env.reset_buf.copy_(terminated | timed_out)
    rewards = env._get_rewards()
    if hasattr(env, "reward_buf"):
        env.reward_buf.copy_(rewards)
    else:
        env.reward_buf = rewards.clone()
    return terminated.clone(), timed_out.clone()


def _contact_counts(env: Any, active_ids: Sequence[int]) -> dict[str, int]:
    active = set(active_ids)
    active_count = 0
    inactive_count = 0
    for row in getattr(env, "contact_substep_records", []):
        if not isinstance(row, Mapping):
            continue
        env_id = row.get("env_id")
        if not isinstance(env_id, int):
            continue
        if int(row.get("contact_count", 0)) == 0:
            continue
        if env_id in active:
            active_count += 1
        else:
            inactive_count += 1
    return {"active": active_count, "inactive": inactive_count}


def _collect_standard_trial(
    env: Any, *, clip_index: int, onset: int, active_count: int, schedule: str
) -> dict[int, dict[str, Any]]:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    reset_frozen_clip_frame_zero(env, clip_index=clip_index)
    actions = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    active_ids = torch.arange(active_count, dtype=torch.long, device=env.device)
    if schedule == "all_simultaneous":
        for step in range(onset + 1):
            terminated, timed_out = raw_control_step(env, actions)
            if bool((terminated | timed_out).any()) and step < onset:
                raise RuntimeError("R3 early termination before contact onset")
        measured = _measurement(env, torch.arange(env.num_envs, device=env.device))
        return {index: _select(measured, index) for index in range(env.num_envs)}
    if schedule != "one_active":
        raise ValueError("standard topology collector only supports simultaneous or one-active")
    for step in range(onset + 1):
        terminated, timed_out = _scheduled_control_step(env, actions, active_ids)
        if bool((terminated | timed_out).index_select(0, active_ids).any()) and step < onset:
            raise RuntimeError("R3 active contact environment terminated before onset")
    measured = _measurement(env, active_ids)
    counts = _contact_counts(env, [0])
    if counts["active"] == 0 or counts["inactive"] != 0:
        raise RuntimeError(
            "R3 T1 dummy-contact contract failed: "
            f"active_records={counts['active']} inactive_records={counts['inactive']}"
        )
    return {0: _select(measured, 0)}


def _collect_staggered_trial(env: Any, *, clip_index: int, onset: int) -> dict[int, dict[str, Any]]:
    import torch

    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    reset_frozen_clip_frame_zero(env, clip_index=clip_index)
    actions = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    delays = torch.arange(env.num_envs, dtype=torch.long, device=env.device) * 4
    captured: dict[int, dict[str, Any]] = {}
    last_step = onset + int(delays.max().detach().cpu())
    for global_step in range(last_step + 1):
        active_ids = torch.nonzero(delays <= global_step, as_tuple=False).squeeze(-1)
        _scheduled_control_step(env, actions, active_ids)
        reached = torch.nonzero(env._reference_index == onset, as_tuple=False).squeeze(-1)
        for env_id in reached.detach().cpu().tolist():
            if env_id not in captured:
                captured[env_id] = _select(
                    _measurement(env, torch.tensor([env_id], device=env.device)), 0
                )
    if len(captured) != env.num_envs:
        raise RuntimeError("R3 staggered schedule did not capture every contact onset")
    return captured


def _comparison_summary(
    trial_measurements: Sequence[Mapping[int, Mapping[str, Any]]], *, schedule: str
) -> dict[str, object]:
    from toporetarget.rl.isaaclab_oracle.tolerance import freeze_tolerances

    baseline = trial_measurements[0]
    comparisons: list[dict[str, object]] = []
    if schedule == "all_simultaneous" and len(baseline) > 1:
        # One anchor covers both same-scene equality and repeat-after-reset
        # equality.  Peer-only comparisons could miss a divergence that is
        # identical across an entire trial but changes after the next reset.
        source = baseline[0]
        for trial_index, row in enumerate(trial_measurements):
            for env_id in sorted(row):
                if trial_index == 0 and env_id == 0:
                    continue
                comparisons.append(_compare(source, row[env_id]))
    else:
        for row in trial_measurements[1:]:
            for env_id in sorted(baseline):
                comparisons.append(_compare(baseline[env_id], row[env_id]))
    if not comparisons:
        raise RuntimeError("R3 topology comparison has no repeated measurement")
    metric_samples: dict[str, list[float]] = {}
    for comparison in comparisons:
        metrics = comparison["metrics"]
        assert isinstance(metrics, Mapping)
        for name, value in metrics.items():
            metric_samples.setdefault(str(name), []).append(float(value))
    tolerance = freeze_tolerances(metric_samples)
    raw_maxima: dict[str, float] = {}
    for comparison in comparisons:
        raw = comparison["raw_max_abs"]
        assert isinstance(raw, Mapping)
        for name, value in raw.items():
            raw_maxima[name] = max(raw_maxima.get(str(name), 0.0), float(value))
    observation_maxima: list[float] = []
    for comparison in comparisons:
        observation = comparison["observation_max_abs"]
        if not isinstance(observation, int | float):
            raise TypeError("R3 observation comparison must be numeric")
        observation_maxima.append(float(observation))
    derived_max = max(
        [float(value) for values in metric_samples.values() for value in values]
        + observation_maxima
    )
    return {
        "comparison_count": len(comparisons),
        "raw_state_max_abs": raw_maxima,
        "raw_state_stable": all(value == 0.0 for value in raw_maxima.values()),
        "derived_state_max_abs": derived_max,
        "derived_state_stable": derived_max == 0.0
        and all(bool(comparison["termination_exact"]) for comparison in comparisons),
        "all_termination_exact": all(
            bool(comparison["termination_exact"]) for comparison in comparisons
        ),
        "frozen_tolerances": tolerance,
    }


def _gpu_memory_mib() -> float | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return float(values[0]) if values and values[0].replace(".", "", 1).isdigit() else None


def _run_worker(args: argparse.Namespace) -> int:
    if (
        not args.accept_eula
        or args.trials != 20
        or args.identifier is None
        or args.scene_env_count is None
        or args.active_contact_count is None
        or args.schedule is None
        or args.shard_id is None
        or args.output is None
    ):
        raise SystemExit("R3 worker requires frozen topology arguments and exactly 20 trials")
    if args.output.exists():
        raise FileExistsError(f"R3 topology worker output already exists: {args.output}")
    from toporetarget.rl.isaaclab_oracle.topology import ContactTopologyExperimentV1

    experiment = ContactTopologyExperimentV1(
        identifier=args.identifier,
        scene_env_count=args.scene_env_count,
        active_contact_count=args.active_contact_count,
        shard_sizes=(args.scene_env_count,),
        schedule=args.schedule,
        trials=args.trials,
    )
    frames = _load_frames(args.frames)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    started = time.perf_counter()
    memory_before = _gpu_memory_mib()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env

        env = make_stage16c5_env(num_envs=experiment.scene_env_count, contact_telemetry="aggregate")
        trial_rows: list[dict[str, object]] = []
        all_pass = True
        raw_stable = True
        derived_stable = True
        finite = True
        for clip_index, frame_row in enumerate(frames):
            onset = int(frame_row["frames"]["contact_onset"])
            trials: list[Mapping[int, Mapping[str, Any]]] = []
            for _ in range(args.trials):
                if experiment.schedule == "staggered":
                    measured = _collect_staggered_trial(env, clip_index=clip_index, onset=onset)
                else:
                    measured = _collect_standard_trial(
                        env,
                        clip_index=clip_index,
                        onset=onset,
                        active_count=experiment.active_contact_count,
                        schedule=experiment.schedule,
                    )
                trials.append(measured)
                finite = finite and all(
                    bool(torch.isfinite(value).all())
                    for measurement in measured.values()
                    for value in measurement["raw"].values()
                    if value.is_floating_point()
                )
            summary = _comparison_summary(trials, schedule=experiment.schedule)
            tolerance = summary["frozen_tolerances"]
            assert isinstance(tolerance, Mapping)
            passes = bool(
                tolerance["status"] == "REPLICATION_TOLERANCES_FROZEN"
                and summary["all_termination_exact"]
                and finite
            )
            all_pass = all_pass and passes
            raw_stable = raw_stable and bool(summary["raw_state_stable"])
            derived_stable = derived_stable and bool(summary["derived_state_stable"])
            trial_rows.append(
                {
                    "clip": frame_row["clip"],
                    "contact_onset_frame": onset,
                    "summary": summary,
                    "passes_frozen_gate": passes,
                }
            )
        contract = env.contract_report()
        no_hidden_writes = bool(
            contract["object_rollout_state_writes"] == 0
            and contract["wrist_root_state_writes_during_step"] == 0
        )
        payload = {
            "schema_version": "stage16c5a_r3_contact_topology_worker_v1",
            "identifier": experiment.identifier,
            "shard_id": args.shard_id,
            "experiment": experiment.as_dict(),
            "device": str(env.device),
            "trials": args.trials,
            "contact_reference": "frozen_factor8_contact_onset_only",
            "reference_or_action_modified": False,
            "candidate_state_restore_used": False,
            "object_pose_write_used": False,
            "hidden_force_or_teleport_used": False,
            "no_hidden_execution_state_writes": no_hidden_writes,
            "execution_write_audit": {
                "object": contract["object_rollout_state_writes"],
                "wrist": contract["wrist_root_state_writes_during_step"],
            },
            "raw_state_stable": raw_stable,
            "derived_state_stable": derived_stable,
            "finite": finite,
            "passes_frozen_gate": all_pass and no_hidden_writes,
            "clip_rows": trial_rows,
            "latency_s": time.perf_counter() - started,
            "gpu_memory_before_mib": memory_before,
            "gpu_memory_peak_mib": _gpu_memory_mib(),
            "ipc_overhead_s": 0.0,
        }
        _write(args.output, payload)
        print(json.dumps({"identifier": args.identifier, "result": payload["passes_frozen_gate"]}))
        return 0
    except BaseException as error:
        failure = args.output.with_suffix(".failure.json")
        if not failure.exists():
            _write(
                failure,
                {
                    "schema_version": "stage16c5a_r3_contact_topology_worker_failure_v1",
                    "identifier": args.identifier,
                    "shard_id": args.shard_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _run_parent(args: argparse.Namespace) -> int:
    if not args.accept_eula or args.trials != 20 or args.output_dir is None:
        raise SystemExit("R3 topology gate requires --accept-eula and exactly 20 trials")
    if args.output_dir.exists():
        raise FileExistsError(f"R3 topology output directory already exists: {args.output_dir}")
    from toporetarget.rl.isaaclab_oracle.topology import (
        classify_contact_topology,
        r3_topology_matrix,
    )

    matrix = r3_topology_matrix()
    _load_frames(args.frames)
    args.output_dir.mkdir(parents=True)
    worker_dir = args.output_dir / "workers"
    worker_dir.mkdir()
    aggregate_rows: list[dict[str, object]] = []
    for phase, experiments in matrix.items():
        for experiment in experiments:
            shard_rows: list[dict[str, Any]] = []
            parent_started = time.perf_counter()
            for shard_id, size in enumerate(experiment.shard_sizes):
                output = worker_dir / f"{experiment.identifier}_shard{shard_id}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--accept-eula",
                    "--frames",
                    str(args.frames.resolve()),
                    "--trials",
                    "20",
                    "--identifier",
                    experiment.identifier,
                    "--scene-env-count",
                    str(size),
                    "--active-contact-count",
                    str(1 if experiment.schedule == "one_active" else size),
                    "--schedule",
                    experiment.schedule,
                    "--shard-id",
                    str(shard_id),
                    "--output",
                    str(output.resolve()),
                ]
                launched = time.perf_counter()
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
                    check=False,
                    capture_output=True,
                    text=True,
                )
                parent_elapsed = time.perf_counter() - launched
                log_path = worker_dir / f"{experiment.identifier}_shard{shard_id}.log"
                if log_path.exists():
                    raise AssertionError("R3 worker log path collision")
                log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    raise RuntimeError(
                        "R3 topology worker failed "
                        f"identifier={experiment.identifier} shard={shard_id} "
                        f"returncode={completed.returncode} log={log_path}"
                    )
                payload = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("R3 worker report is malformed")
                payload["ipc_overhead_s"] = max(0.0, parent_elapsed - float(payload["latency_s"]))
                shard_rows.append(payload)
            elapsed = time.perf_counter() - parent_started
            passes = all(bool(row["passes_frozen_gate"]) for row in shard_rows)
            aggregate_rows.append(
                {
                    "phase": phase,
                    "identifier": experiment.identifier,
                    "experiment": experiment.as_dict(),
                    "worker_reports": [
                        str(worker_dir / f"{experiment.identifier}_shard{index}.json")
                        for index in range(len(shard_rows))
                    ],
                    "shard_count": len(shard_rows),
                    "raw_state_stable": all(bool(row["raw_state_stable"]) for row in shard_rows),
                    "derived_state_stable": all(
                        bool(row["derived_state_stable"]) for row in shard_rows
                    ),
                    "passes_frozen_gate": passes,
                    "throughput": {
                        "candidate_scenes": experiment.scene_env_count,
                        "wall_time_s": elapsed,
                        "effective_scene_rollouts_per_s": (
                            experiment.scene_env_count * args.trials * 2 / elapsed
                        ),
                        "gpu_memory_peak_mib": max(
                            float(row["gpu_memory_peak_mib"] or 0.0) for row in shard_rows
                        ),
                        "ipc_overhead_s": sum(float(row["ipc_overhead_s"]) for row in shard_rows),
                    },
                }
            )
    diagnosis = classify_contact_topology(aggregate_rows)
    result = {
        "schema_version": "stage16c5a_r3_contact_topology_diagnosis_v1",
        "frozen_inputs": {
            "reference": "factor8_321_samples_20hz_control_120hz_physics_decimation6",
            "controller": "finite_virtual_6d_wrist_actuator_v1",
            "physics_contract": "R2_G0_no_solver_mutation",
            "frames": str(args.frames),
        },
        "matrix": aggregate_rows,
        "diagnosis": diagnosis,
        "next_contract": (
            "SHARDED_DETERMINISTIC_ORACLE_CANDIDATE"
            if diagnosis["classification"] == "SINGLE_SCENE_CONTACT_BATCHING_FAILURE"
            else "ROBUST_STATISTICAL_ORACLE_CANDIDATE"
            if diagnosis["classification"] == "TRUE_CONTACT_SOLVER_NONDETERMINISM"
            else "HARNESS_REPAIR_REQUIRED"
        ),
        "ppo": {"status": "NOT_STARTED", "samples": 0, "checkpoints": 0},
    }
    _write(args.output_dir / "contact_topology_diagnosis.json", result)
    print(json.dumps({"classification": diagnosis["classification"], "rows": len(aggregate_rows)}))
    return 0


def main() -> int:
    args = parse_args()
    return _run_worker(args) if args.worker else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
