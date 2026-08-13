#!/usr/bin/env python3
"""Benchmark bounded C.5A candidate capacity; this is not CEM or an Oracle rollout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--candidate-envs", type=int, choices=(32, 96, 144), required=True)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--measurement-batches", type=int, default=100)
    parser.add_argument(
        "--replication-qualification",
        type=Path,
        required=True,
        help="C.5A O1 report; benchmark refuses to run unless it passed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _gpu() -> dict[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    return {
        "utilization_percent": int(fields[0]),
        "vram_used_mib": int(fields[1]),
        "vram_total_mib": int(fields[2]),
    }


def _write(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_BENCHMARK_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_replication_qualification(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != (
        "STAGE16C5A_STATE_REPLICATION_VALIDATED"
    ):
        raise RuntimeError("C5A_BENCHMARK_REPLICATION_QUALIFICATION_REQUIRED")


def _batch(env: Any, pool: Any, horizons: tuple[int, ...]) -> dict[str, float]:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step

    capture_started = time.perf_counter()
    state = pool.capture_execution_state()
    capture_s = time.perf_counter() - capture_started
    restore_started = time.perf_counter()
    pool.replicate_execution_state(state)
    restore_s = time.perf_counter() - restore_started
    rollout_started = time.perf_counter()
    action = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    control_steps = 0
    for horizon in horizons:
        for _ in range(horizon):
            raw_control_step(env, action)
            control_steps += 1
    return {
        "capture_s": capture_s,
        "restore_s": restore_s,
        "rollout_s": time.perf_counter() - rollout_started,
        "control_steps": float(control_steps),
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.warmup_batches != 20 or args.measurement_batches != 100:
        raise SystemExit("C.5A freezes warmup=20 and measurement=100 batches")
    if args.output.exists():
        raise FileExistsError(f"STAGE16C5A_BENCHMARK_REFUSES_OVERWRITE: {args.output}")
    _require_replication_qualification(args.replication_qualification)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from toporetarget.rl.isaaclab_oracle.candidate_pool import PhysXOracleCandidatePoolV1
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env

        env = make_stage16c5_env(num_envs=1 + args.candidate_envs)
        population = 32 if args.candidate_envs == 96 else 48 if args.candidate_envs == 144 else 32
        pool = PhysXOracleCandidatePoolV1(
            env,
            candidate_count=args.candidate_envs,
            population_per_horizon=population,
        )
        horizons = (1, 5, 10)
        for _ in range(args.warmup_batches):
            _batch(env, pool, horizons)
        gpu_samples = [_gpu()]
        rows = [_batch(env, pool, horizons) for _ in range(args.measurement_batches)]
        gpu_samples.append(_gpu())
        elapsed_s = sum(row["capture_s"] + row["restore_s"] + row["rollout_s"] for row in rows)
        total_control_steps = sum(row["control_steps"] for row in rows)
        total_sequences = args.candidate_envs * len(horizons) * args.measurement_batches
        control_per_s = total_control_steps * args.candidate_envs / elapsed_s
        report = {
            "status": "STAGE16C5A_CANDIDATE_POOL_BENCHMARK_VALIDATED",
            "candidate_envs": args.candidate_envs,
            "total_envs": env.num_envs,
            "horizons": list(horizons),
            "placeholder_actions_only": True,
            "cem_updates": 0,
            "warmup_batches": args.warmup_batches,
            "measurement_batches": args.measurement_batches,
            "elapsed_s": elapsed_s,
            "candidate_sequences_per_s": total_sequences / elapsed_s,
            "simulated_control_steps_per_s": control_per_s,
            "physics_steps_per_s": control_per_s * env.cfg.decimation,
            "capture_ms_mean": 1000.0 * sum(row["capture_s"] for row in rows) / len(rows),
            "restore_ms_mean": 1000.0 * sum(row["restore_s"] for row in rows) / len(rows),
            "candidate_rollout_ms_mean": 1000.0 * sum(row["rollout_s"] for row in rows) / len(rows),
            "history_replay_latency": "measured separately by O1; no synthetic history shortcut",
            "gpu": gpu_samples,
            "cuda_device": str(env.device),
            "pool": pool.validate_layout(),
            "write_audit": pool.write_audit.as_dict(),
            "clean_exit": True,
        }
        _write(args.output, report)
        print(json.dumps({"status": report["status"], "candidate_envs": args.candidate_envs}))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
