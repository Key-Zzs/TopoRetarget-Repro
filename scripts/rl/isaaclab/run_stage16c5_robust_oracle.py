#!/usr/bin/env python3
"""Run the Stage 16-C.5A-R3 robust-oracle contract without state restoration.

The contact-topology gate can establish that simultaneous contact populations
are not a valid deterministic candidate pool.  This runner then evaluates an
already selected action trace through independent *one-environment* frame-zero
PhysX rollouts.  It is deliberately not an optimizer, does not import the
candidate-state restore helpers, and never writes object or wrist root state
during a rollout.

The parent process remains Isaac-free.  It launches a fresh worker process for
each clip qualification and benchmark cell; a worker owns one Isaac scene and
restarts that scene at frame zero before every replica.  The serial topology is
intentional: no result here claims that a simultaneous multi-contact batch is
valid after the R3 topology diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_RUNTIME_STEPS = 320
_REPLICA_QUALIFICATION_COUNT = 20
_ACTION_SHAPE = (40, 26)
_BENCHMARK_CANDIDATE_COUNTS = (32, 96)
_BENCHMARK_REPLICA_COUNTS = (1, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--clip", help=argparse.SUPPRESS)
    parser.add_argument("--action-trace", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--replica-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-candidate-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-replicas", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R3_ROBUST_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trace_manifest(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("traces"), list):
        raise ValueError("R3 robust runner needs a manifest with a traces list")
    traces: dict[str, dict[str, object]] = {}
    for row in payload["traces"]:
        if not isinstance(row, Mapping):
            raise ValueError("R3 robust trace manifest row is malformed")
        clip = row.get("clip")
        trace = row.get("action_trace")
        expected_hash = row.get("action_trace_sha256")
        if (
            not isinstance(clip, str)
            or not isinstance(trace, str)
            or not isinstance(expected_hash, str)
        ):
            raise ValueError("R3 robust trace manifest lacks clip, trace, or hash")
        trace_path = Path(trace)
        if clip in traces or not trace_path.is_file() or _sha256(trace_path) != expected_hash:
            raise ValueError(f"R3 robust selected trace validation failed: {clip}")
        traces[clip] = {
            "clip": clip,
            "action_trace": str(trace_path.resolve()),
            "action_trace_sha256": expected_hash,
            "source_protocol": payload.get("source_protocol"),
        }
    if tuple(sorted(traces)) != ("hocap_170105", "hocap_170650"):
        raise ValueError("R3 robust runner requires exactly the frozen two clip traces")
    return traces


def _load_actions(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        actions = np.asarray(payload["actions"], dtype=np.float32)
    if actions.shape != _ACTION_SHAPE or not np.isfinite(actions).all():
        raise ValueError(f"R3 robust trace must be finite shape {_ACTION_SHAPE}: {path}")
    if float(np.abs(actions).max()) > 1.0:
        raise ValueError(f"R3 robust trace is outside frozen action bounds: {path}")
    retimed = np.repeat(actions, _RUNTIME_STEPS // actions.shape[0], axis=0)
    if retimed.shape != (_RUNTIME_STEPS, 26):
        raise AssertionError("R3 robust factor-8 retiming mismatch")
    return retimed


def _nvidia_sample() -> tuple[float | None, float | None]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None, None
    line = next((value.strip() for value in completed.stdout.splitlines() if value.strip()), "")
    values = [value.strip() for value in line.split(",")]
    if len(values) != 2:
        return None, None
    try:
        return float(values[0]), float(values[1])
    except ValueError:
        return None, None


class _GpuSampler:
    """Bounded sampling for a report; it has no simulator interaction."""

    def __init__(self) -> None:
        self._samples: list[tuple[float | None, float | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(_nvidia_sample())
            self._stop.wait(0.5)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float | None]:
        self._stop.set()
        self._thread.join(timeout=2.0)
        utilization = [value for value, _ in self._samples if value is not None]
        memory = [value for _, value in self._samples if value is not None]
        return {
            "gpu_utilization_mean_pct": fmean(utilization) if utilization else None,
            "gpu_utilization_peak_pct": max(utilization) if utilization else None,
            "gpu_memory_peak_mib": max(memory) if memory else None,
            "gpu_sample_count": len(self._samples),
        }


def _reason(env: Any) -> str:
    stage = env.extras.get("stage16")
    if not isinstance(stage, Mapping):
        return "MISSING_STAGE16_TERMINATION"
    codes: Any = stage.get("primary_reason_code")
    names: Any = stage.get("termination_reasons")
    if not hasattr(codes, "detach") or not isinstance(names, tuple):
        return "MALFORMED_STAGE16_TERMINATION"
    code = int(codes[0].detach().cpu())
    return names[code] if 0 <= code < len(names) else f"UNKNOWN_REASON_{code}"


def _stage_value(env: Any, name: str) -> float:
    stage = env.extras.get("stage16")
    if not isinstance(stage, Mapping) or name not in stage:
        raise RuntimeError(f"R3 robust rollout missing stage16 field: {name}")
    value: Any = stage[name]
    return float(value[0].detach().cpu())


def _run_one_rollout(
    env: Any, *, clip_index: int, actions: np.ndarray, replica_id: int
) -> dict[str, object]:
    """Run exactly one fresh frame-zero physical rollout and measure it."""

    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.robust import (
        OBJECT_AXIS_GATE_M,
        OBJECT_POSITION_GATE_M,
        OBJECT_ROTATION_GATE_DEG,
    )
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    # The seed is recorded for independence bookkeeping.  There is no task
    # randomization in the frozen contract, so a different seed cannot be
    # misrepresented as a physics or reference mutation.
    env.reset(seed=20260804 + replica_id)
    reset_frozen_clip_frame_zero(env, clip_index=clip_index)
    effort_samples: list[float] = []
    contact_records_before = len(getattr(env, "contact_substep_records", []))
    terminated = False
    timed_out = False
    steps = 0
    for action_array in actions:
        action = torch.as_tensor(action_array[None], dtype=torch.float32, device=env.device)
        terminated_tensor, timed_out_tensor = raw_control_step(env, action)
        effort_samples.append(float(env._robot.data.applied_torque[0].abs().mean().detach().cpu()))
        steps += 1
        terminated = bool(terminated_tensor[0].detach().cpu())
        timed_out = bool(timed_out_tensor[0].detach().cpu())
        if terminated or timed_out:
            break
    position = _stage_value(env, "object_position_error_m")
    axis = _stage_value(env, "object_axis_error_m")
    rotation = math.degrees(_stage_value(env, "object_orientation_error_rad"))
    success = bool(env._success[0].detach().cpu())
    final_reach = int(env._reference_index[0].detach().cpu()) == env.reference_bank.frame_count - 1
    contact_rows = getattr(env, "contact_substep_records", [])[contact_records_before:]
    has_nonzero_contact = any(
        isinstance(row, Mapping) and int(row.get("contact_count", 0)) > 0 for row in contact_rows
    )
    stage = env.extras["stage16"]
    force_sat = float(stage["force_saturation_ratio"][0].detach().cpu())
    torque_sat = float(stage["torque_saturation_ratio"][0].detach().cpu())
    contact_stability_penalty = float(not has_nonzero_contact) + force_sat + torque_sat
    action_differences = np.linalg.norm(np.diff(actions[:steps], axis=0), axis=1)
    action_smoothness = float(action_differences.mean()) if action_differences.size else 0.0
    effort = fmean(effort_samples) if effort_samples else 0.0
    normalized_error_sum = (
        position / OBJECT_POSITION_GATE_M
        + rotation / OBJECT_ROTATION_GATE_DEG
        + axis / OBJECT_AXIS_GATE_M
    )
    cost = (
        normalized_error_sum
        + float(not (success and final_reach))
        + 0.01 * (contact_stability_penalty + action_smoothness + effort)
    )
    return {
        "replica_id": replica_id,
        "seed": 20260804 + replica_id,
        "steps_executed": steps,
        "terminated": terminated,
        "timed_out_success_boundary": timed_out,
        "cost": cost,
        "object_position_error_m": position,
        "object_rotation_error_deg": rotation,
        "object_axis_error_m": axis,
        "success": success,
        "final_reach": final_reach,
        "contact_stability_penalty": contact_stability_penalty,
        "action_smoothness": action_smoothness,
        "effort": effort,
        "termination_reason": _reason(env),
    }


def _numeric_field(row: Mapping[str, object], name: str) -> float:
    value = row.get(name)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"R3 robust replica field is not numeric: {name}")
    return float(value)


def _replica_result(row: Mapping[str, object]) -> Any:
    from toporetarget.rl.isaaclab_oracle.robust import RobustReplicaResultV1

    return RobustReplicaResultV1(
        cost=_numeric_field(row, "cost"),
        object_position_error_m=_numeric_field(row, "object_position_error_m"),
        object_rotation_error_deg=_numeric_field(row, "object_rotation_error_deg"),
        object_axis_error_m=_numeric_field(row, "object_axis_error_m"),
        success=bool(row["success"]),
        final_reach=bool(row["final_reach"]),
        contact_stability_penalty=_numeric_field(row, "contact_stability_penalty"),
        action_smoothness=_numeric_field(row, "action_smoothness"),
        effort=_numeric_field(row, "effort"),
        termination_reason=str(row["termination_reason"]),
    )


def _run_worker(args: argparse.Namespace) -> int:
    if (
        not args.accept_eula
        or not args.clip
        or args.action_trace is None
        or args.replica_count is None
        or args.output is None
    ):
        raise SystemExit("R3 robust worker requires the frozen explicit arguments")
    if args.replica_count < 1 or args.output.exists():
        raise ValueError("R3 robust worker needs a positive replica count and fresh output")
    benchmark = args.benchmark_candidate_count is not None or args.benchmark_replicas is not None
    if benchmark:
        if (
            args.benchmark_candidate_count not in _BENCHMARK_CANDIDATE_COUNTS
            or args.benchmark_replicas not in _BENCHMARK_REPLICA_COUNTS
            or args.replica_count != args.benchmark_candidate_count * args.benchmark_replicas
        ):
            raise ValueError("R3 robust benchmark count must equal candidates times replicas")
    actions = _load_actions(args.action_trace)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    started = time.perf_counter()
    sampler = _GpuSampler()
    app = AppLauncher(headless=True).app
    env = None
    try:
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env

        env = make_stage16c5_env(num_envs=1, contact_telemetry="aggregate")
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        sampler.start()
        rows = [
            _run_one_rollout(env, clip_index=clip_index, actions=actions, replica_id=index)
            for index in range(args.replica_count)
        ]
        gpu = sampler.stop()
        contract = env.contract_report()
        no_hidden_writes = bool(
            contract["object_rollout_state_writes"] == 0
            and contract["wrist_root_state_writes_during_step"] == 0
        )
        payload: dict[str, object] = {
            "schema_version": "stage16c5a_r3_robust_oracle_worker_v1",
            "clip": args.clip,
            "action_trace": str(args.action_trace.resolve()),
            "action_trace_sha256": _sha256(args.action_trace),
            "reference_or_action_modified": False,
            "candidate_state_restore_used": False,
            "object_pose_write_used": False,
            "hidden_force_or_teleport_used": False,
            "rollout_topology": "one_environment_fresh_frame_zero_per_replica",
            "replica_count": args.replica_count,
            "replicas": rows if not benchmark else [],
            "benchmark_outcome_summary": (
                {
                    "success_rate": fmean(float(bool(row["success"])) for row in rows),
                    "final_reach_rate": fmean(float(bool(row["final_reach"])) for row in rows),
                    "mean_steps_executed": fmean(
                        _numeric_field(row, "steps_executed") for row in rows
                    ),
                }
                if benchmark
                else None
            ),
            "no_hidden_execution_state_writes": no_hidden_writes,
            "execution_write_audit": {
                "object": contract["object_rollout_state_writes"],
                "wrist": contract["wrist_root_state_writes_during_step"],
            },
            "latency_s": time.perf_counter() - started,
            "gpu": gpu,
        }
        if benchmark:
            payload["benchmark"] = {
                "candidate_count": args.benchmark_candidate_count,
                "replicas_per_candidate": args.benchmark_replicas,
                "effective_rollouts": args.replica_count,
                "workload": "factor8_selected_contact_trace_one_env_serial_dispatch",
                "concurrency_claim": "none; simultaneous contact populations are not valid",
            }
        _write(args.output, payload)
        print(json.dumps({"clip": args.clip, "replicas": args.replica_count, "ok": True}))
        return 0
    finally:
        if sampler._thread.is_alive():
            sampler.stop()
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _launch_worker(
    args: argparse.Namespace,
    *,
    clip: str,
    trace: Mapping[str, object],
    replica_count: int,
    output: Path,
    benchmark_candidate_count: int | None = None,
    benchmark_replicas: int | None = None,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--accept-eula",
        "--trace-manifest",
        str(args.trace_manifest.resolve()),
        "--clip",
        clip,
        "--action-trace",
        str(trace["action_trace"]),
        "--replica-count",
        str(replica_count),
        "--output",
        str(output.resolve()),
    ]
    if benchmark_candidate_count is not None and benchmark_replicas is not None:
        command.extend(
            [
                "--benchmark-candidate-count",
                str(benchmark_candidate_count),
                "--benchmark-replicas",
                str(benchmark_replicas),
            ]
        )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"},
        check=False,
        capture_output=True,
        text=True,
    )
    log_path = output.with_suffix(".log")
    if log_path.exists():
        raise FileExistsError(f"R3 robust worker log collision: {log_path}")
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"R3 robust worker failed clip={clip} returncode={completed.returncode} log={log_path}"
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("R3 robust worker payload is malformed")
    payload["ipc_overhead_s"] = max(
        0.0, time.perf_counter() - started - float(payload["latency_s"])
    )
    return payload


def _evaluate_clip(clip: str, worker: Mapping[str, object]) -> dict[str, object]:
    from toporetarget.rl.isaaclab_oracle.robust import (
        RobustCandidateEvaluatorV1,
        RobustCandidateSelector,
        RobustOracleContractV1,
        qualify_c5c_independent_replicas,
    )

    rows = worker.get("replicas")
    if not isinstance(rows, list) or len(rows) != _REPLICA_QUALIFICATION_COUNT:
        raise ValueError("R3 robust qualification worker must return twenty replica rows")
    replicas = [_replica_result(row) for row in rows if isinstance(row, Mapping)]
    if len(replicas) != _REPLICA_QUALIFICATION_COUNT:
        raise ValueError("R3 robust replica row is malformed")
    four = RobustCandidateEvaluatorV1(RobustOracleContractV1(replica_count=4)).evaluate(
        clip, replicas[:4]
    )
    eight = RobustCandidateEvaluatorV1(RobustOracleContractV1(replica_count=8)).evaluate(
        clip, replicas[:8]
    )
    # The selector remains in the runtime path even for this one frozen input;
    # a later optimizer may supply multiple evaluations without changing its
    # lexical contract.
    selected = RobustCandidateSelector().select([four])
    return {
        "clip": clip,
        "candidate_trace_role": "frozen_mujoco_selected_trace",
        "one_selected_trace": dict(rows[0]),
        "independent_replicas": qualify_c5c_independent_replicas(replicas),
        "default_replicas_4": four.as_dict(),
        "upgraded_replicas_8": eight.as_dict(),
        "selector_selected_candidate": selected.candidate_id,
        "no_hidden_execution_state_writes": bool(worker["no_hidden_execution_state_writes"]),
        "worker_report": str(worker.get("_report_path", "")),
    }


def _run_parent(args: argparse.Namespace) -> int:
    if not args.accept_eula or args.output_dir is None:
        raise SystemExit("R3 robust parent requires --accept-eula and --output-dir")
    if args.output_dir.exists():
        raise FileExistsError(f"STAGE16C5A_R3_ROBUST_OUTPUT_ALREADY_EXISTS: {args.output_dir}")
    from toporetarget.rl.isaaclab_oracle.robust import RobustOracleContractV1

    traces = _load_trace_manifest(args.trace_manifest)
    qualification_dir = args.output_dir / "qualification_workers"
    qualification: dict[str, dict[str, object]] = {}
    for clip in sorted(traces):
        path = qualification_dir / f"{clip}.json"
        worker = _launch_worker(
            args,
            clip=clip,
            trace=traces[clip],
            replica_count=_REPLICA_QUALIFICATION_COUNT,
            output=path,
        )
        worker["_report_path"] = str(path)
        qualification[clip] = _evaluate_clip(clip, worker)
    c5c_pass = True
    for row in qualification.values():
        c5c = row.get("independent_replicas")
        if not isinstance(c5c, Mapping):
            raise ValueError("R3 robust qualification lacks C5C aggregate")
        c5c_pass = c5c_pass and bool(c5c.get("passes_frozen_gate"))
        c5c_pass = c5c_pass and bool(row.get("no_hidden_execution_state_writes"))

    benchmark_dir = args.output_dir / "benchmark_workers"
    benchmark_rows: list[dict[str, object]] = []
    ordered_traces = [traces[clip] for clip in sorted(traces)]
    for candidate_count in _BENCHMARK_CANDIDATE_COUNTS:
        for replicas in _BENCHMARK_REPLICA_COUNTS:
            # Alternate clips deterministically.  A worker remains a single
            # scene to avoid reintroducing invalid simultaneous contact pools.
            trace = ordered_traces[(candidate_count + replicas) % len(ordered_traces)]
            clip = str(trace["clip"])
            path = benchmark_dir / f"candidates{candidate_count}_replicas{replicas}.json"
            worker = _launch_worker(
                args,
                clip=clip,
                trace=trace,
                replica_count=candidate_count * replicas,
                output=path,
                benchmark_candidate_count=candidate_count,
                benchmark_replicas=replicas,
            )
            wall_time = _numeric_field(worker, "latency_s") + _numeric_field(
                worker, "ipc_overhead_s"
            )
            benchmark_rows.append(
                {
                    "candidate_count": candidate_count,
                    "replicas_per_candidate": replicas,
                    "effective_rollouts": candidate_count * replicas,
                    "clip": clip,
                    "worker_report": str(path),
                    "wall_time_s": wall_time,
                    "effective_rollouts_per_s": (
                        candidate_count * replicas / wall_time if wall_time > 0.0 else float("inf")
                    ),
                    "gpu": worker["gpu"],
                    "ipc_overhead_s": worker["ipc_overhead_s"],
                    "workload_status": "MEASURED_CONTACT_TRACE_SERIAL_DISPATCH",
                    "no_hidden_execution_state_writes": worker["no_hidden_execution_state_writes"],
                }
            )
    result = {
        "schema_version": "stage16c5a_r3_robust_oracle_report_v1",
        "frozen_inputs": {
            "reference": "factor8_321_samples_20hz_control_120hz_physics_decimation6",
            "controller": "finite_virtual_6d_wrist_actuator_v1",
            "action_trace_manifest": str(args.trace_manifest.resolve()),
            "action_trace_manifest_sha256": _sha256(args.trace_manifest),
            "trace_source": "preexisting_mujoco_selected_actions; no R3 optimization run",
        },
        "contract": {
            "default": asdict(RobustOracleContractV1(replica_count=4)),
            "upgrade": asdict(RobustOracleContractV1(replica_count=8)),
            "c5c": (
                "one frozen selected trace plus exactly 20 independent frame-zero replicas per clip"
            ),
            "selector": (
                "failure_probability,cvar_formal_gate_violation,worst_normalized_gate_margin,"
                "mean_object_error,mean_rotation_error,contact_stability_penalty,"
                "action_smoothness,effort"
            ),
        },
        "qualification": qualification,
        "benchmark": benchmark_rows,
        "c5b_authorization": (
            "STAGE16C5B_ROBUST_ORACLE_AUTHORIZED"
            if c5c_pass
            else "STAGE16C5_CONTACT_ORACLE_BLOCKED"
        ),
        "authorization_rationale": (
            "All selected traces satisfy C5C's unchanged physical gates across twenty replicas."
            if c5c_pass
            else (
                "At least one selected trace fails the unchanged C5C physical gate; "
                "robust statistics do not waive failures."
            )
        ),
        "ppo": {"status": "NOT_STARTED", "samples": 0, "checkpoints": 0},
    }
    _write(args.output_dir / "robust_oracle_report.json", result)
    print(
        json.dumps(
            {
                "c5b_authorization": result["c5b_authorization"],
                "benchmark_rows": len(benchmark_rows),
            }
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    return _run_worker(args) if args.worker else _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
