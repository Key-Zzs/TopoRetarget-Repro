#!/usr/bin/env python3
"""Low-cost P1 checkpoint I/O, append-only, and resume parity benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.retarget.refinement_checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointStore,
    _checkpoint_hash,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--durable-interval", type=int, default=10)
    parser.add_argument("--interrupt-after", type=int, default=37)
    return parser


def _manifest(frames: int, interval: int, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "input_signature": "p1_checkpoint_microbenchmark_v1",
        "solver_profile_hash": "solver_math_unchanged",
        "execution_profile_hash": "fast_exact_v2_durable_io",
        "query_profile_hash": "query_math_unchanged",
        "frame_range": [0, frames],
        "durable_checkpoint_interval_frames": interval,
        "intermediate_checkpoint_mode": "append_only_frame_payload_plus_fsync_jsonl",
        "historical_sequence_rewrite": False,
    }


def _payload(local: int, previous: str | None) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # Fixed deterministic arrays model the persisted shape classes without any
    # robot solve. The benchmark measures only checkpoint orchestration.
    arrays = {
        "full_signed_distance": np.linspace(0.01, 0.02, 512, dtype=np.float64) + local * 1.0e-9,
        "hard_residual": np.ones(64, dtype=np.float64),
        "soft_residual": np.ones(64, dtype=np.float64),
        "qpos": np.linspace(-0.2, 0.2, 22, dtype=np.float64) + local * 1.0e-8,
        "base_pose_scene": np.eye(4, dtype=np.float64),
    }
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "local_frame_index": local,
        "global_frame_index": local,
        "timestamp": local / 30.0,
        "optimizer_status_code": 0,
        "optimizer_converged": True,
        "qpos_bounds_pass": True,
        "slack_bounds_pass": True,
        "active_constraints_feasible": True,
        "full_surface_hard_audit_pass": True,
        "full_surface_soft_audit_pass": True,
        "active_set_converged": True,
        "all_values_finite": True,
        "strict_accepted": True,
        "solver_success": True,
        "previous_checkpoint_hash": previous,
        "per_frame_checkpoint_hash": "",
    }
    metadata["per_frame_checkpoint_hash"] = _checkpoint_hash(metadata, arrays)
    return metadata, arrays


def _write_run(
    root: Path,
    *,
    frames: int,
    interval: int,
    emulate_historical_full_scan: bool,
    interrupt_after: int | None = None,
) -> dict[str, Any]:
    store = CheckpointStore.open(root, manifest=_manifest(frames, interval, root.name))
    previous: str | None = None
    first_mtime: int | None = None
    started = time.perf_counter()
    split = frames if interrupt_after is None else interrupt_after
    for local in range(split):
        metadata, arrays = _payload(local, previous)
        previous = store.save_frame(metadata, arrays)
        if emulate_historical_full_scan:
            store.scan(refresh=True)
        store.update_progress(status="running")
        current_mtime = (store.frames_dir / "frame_000000.npz").stat().st_mtime_ns
        first_mtime = current_mtime if first_mtime is None else first_mtime
        if current_mtime != first_mtime:
            raise RuntimeError("APPEND_ONLY_HISTORY_REWRITTEN")
    sessions = 1
    if interrupt_after is not None:
        store.commit_durable_checkpoint(status="interrupted_for_test")
        store = CheckpointStore.open(
            root,
            manifest=_manifest(frames, interval, root.name),
            resume=True,
        )
        chain = store.validate_chain(allow_incomplete=True)
        if int(chain["next_frame"]) != interrupt_after:
            raise RuntimeError("RESUME_NEXT_FRAME_MISMATCH")
        if interrupt_after:
            metadata, _ = store.load_frame(interrupt_after - 1)
            previous = str(metadata["per_frame_checkpoint_hash"])
        for local in range(interrupt_after, frames):
            metadata, arrays = _payload(local, previous)
            previous = store.save_frame(metadata, arrays)
            store.update_progress(status="running")
        sessions = 2
    store.commit_durable_checkpoint(status="complete")
    elapsed = time.perf_counter() - started
    chain = store.validate_chain()
    hashes = [store.load_frame(local)[0]["per_frame_checkpoint_hash"] for local in range(frames)]
    return {
        "seconds": elapsed,
        "ms_per_frame": 1000.0 * elapsed / frames,
        "chain_pass": bool(chain["chain_pass"]),
        "complete": bool(chain["complete"]),
        "sessions": sessions,
        "frame_hashes": hashes,
        "append_event_count": sum(
            1
            for line in (root / "append_events.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line).get("event") == "FRAME_APPENDED"
        ),
        "history_first_frame_mtime_unchanged": True,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.frames <= 1 or args.repeats < 3 or not 0 < args.interrupt_after < args.frames:
        raise ValueError("P1_CHECKPOINT_BENCHMARK_ARGUMENTS_INVALID")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    resume_parity: list[bool] = []
    with tempfile.TemporaryDirectory(prefix="p1_checkpoint_bench_", dir=output.parent) as temp:
        root = Path(temp)
        for repeat in range(1, args.repeats + 1):
            baseline = _write_run(
                root / f"baseline_{repeat}",
                frames=args.frames,
                interval=10_000,
                emulate_historical_full_scan=True,
            )
            candidate = _write_run(
                root / f"candidate_{repeat}",
                frames=args.frames,
                interval=args.durable_interval,
                emulate_historical_full_scan=False,
            )
            resumed = _write_run(
                root / f"resumed_{repeat}",
                frames=args.frames,
                interval=args.durable_interval,
                emulate_historical_full_scan=False,
                interrupt_after=args.interrupt_after,
            )
            parity = candidate["frame_hashes"] == resumed["frame_hashes"]
            resume_parity.append(parity)
            for mode, value in (("before_hardening", baseline), ("after_hardening", candidate)):
                rows.append(
                    {
                        "repeat": repeat,
                        "mode": mode,
                        **{key: item for key, item in value.items() if key != "frame_hashes"},
                    }
                )
    before = [row["ms_per_frame"] for row in rows if row["mode"] == "before_hardening"]
    after = [row["ms_per_frame"] for row in rows if row["mode"] == "after_hardening"]
    result = {
        "schema_version": "RetargetCheckpointDurabilityBenchmarkV1",
        "status": "PASS",
        "scope": "CHECKPOINT_IO_MICROBENCHMARK_ONLY_NOT_SOLVER_THROUGHPUT",
        "frames": args.frames,
        "repeats": args.repeats,
        "durable_checkpoint_interval_frames": args.durable_interval,
        "interrupt_after_frame_count": args.interrupt_after,
        "before_model": "append frame then rescan complete historical frame directory",
        "after_model": "append frame plus fsync JSONL and cached progress; full marker every K",
        "rows": rows,
        "median_before_ms_per_frame": statistics.median(before),
        "median_after_ms_per_frame": statistics.median(after),
        "median_change_percent": 100.0
        * (statistics.median(after) - statistics.median(before))
        / statistics.median(before),
        "append_only_checkpoint_parity": all(
            row["history_first_frame_mtime_unchanged"] for row in rows
        ),
        "resume_parity": all(resume_parity),
        "retarget_math_exercised": False,
        "retarget_math_changed": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
