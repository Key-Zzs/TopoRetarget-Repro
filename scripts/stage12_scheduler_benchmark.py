#!/usr/bin/env python3
"""Run the Stage-12 A/B/C scheduler measurement without touching formal checkpoints.

Each worker is a separate Python OS process.  It restores a read-only copy of
the five-frame checkpoint into a shadow root and only appends new shadow frame
files.  A ready-file barrier excludes imports, data loads and BVH construction
from the reported steady-state interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ENV_PYTHON = Path("/home/deepcybo/miniconda3/envs/topo-retarget/bin/python")
PROFILE = "wuji_continuous_sequential_fast_exact_v4_compiled_sign"


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threads() -> int:
    return len(list((Path("/proc") / str(os.getpid()) / "task").iterdir()))


def _rss_bytes() -> int:
    for line in (Path("/proc") / str(os.getpid()) / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _p95(values: list[float]) -> float | None:
    """Match NumPy's default linear percentile used by the formal health gate."""

    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=100, method="inclusive")[94])


def _paths(source: Path) -> dict[str, Path]:
    checkpoint = max(
        (path for path in (source / "checkpoints").iterdir() if (path / "manifest.json").is_file()),
        key=lambda path: int(_read(path / "progress.json").get("next_frame", -1)),
    )
    return {
        "canonical": source / "canonical" / "canonical_hoi_v2.zarr",
        "warm": source / "warm" / "warm_start.zarr",
        "graph": source / "exports" / "interaction_graph.zarr",
        "collision": source / "exports" / "wuji_collision_samples.npz",
        "checkpoint": checkpoint,
    }


def _shadow_checkpoint(source: Path, shadow_root: Path, *, seed_before: int) -> Path:
    original = _paths(source)["checkpoint"]
    destination = shadow_root / source.parent.name / source.name / "checkpoints" / original.name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original / "manifest.json", destination / "manifest.json")
    if (original / "progress.json").is_file():
        # A formal checkpoint can have advanced after an earlier benchmark.
        # Its copied progress must describe only the frames actually linked into
        # this shadow; otherwise resume validation could treat absent later
        # frames as accepted and silently skip the requested interval.
        progress = _read(original / "progress.json")
        accepted = [
            int(index) for index in progress.get("accepted_frames", []) if int(index) < seed_before
        ]
        progress["accepted_frames"] = accepted
        progress["invalid_frames"] = [
            int(index) for index in progress.get("invalid_frames", []) if int(index) < seed_before
        ]
        progress["orphan_frames"] = [
            int(index) for index in progress.get("orphan_frames", []) if int(index) < seed_before
        ]
        progress["last_accepted_frame"] = accepted[-1] if accepted else None
        progress["next_frame"] = seed_before
        progress["status"] = "paused"
        _write(destination / "progress.json", progress)
    (destination / "frames").mkdir(exist_ok=True)
    for frame in sorted((original / "frames").glob("frame_*.npz")):
        if int(frame.stem.rsplit("_", 1)[1]) >= seed_before:
            continue
        link = destination / "frames" / frame.name
        if not link.exists():
            link.symlink_to(frame.resolve())
    return destination


def worker(args: argparse.Namespace) -> int:
    # Set every numeric library limit before importing the runner and NumPy.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    started = time.monotonic()
    source = Path(args.source).resolve()
    start_frame = int(args.start_frame)
    stop_after_frame = int(args.stop_after_frame)
    shadow = _shadow_checkpoint(source, Path(args.shadow_root).resolve(), seed_before=start_frame)
    paths = _paths(source)
    from toporetarget.cli.retarget import _run_checkpoint_refinement
    from toporetarget.retarget.refinement_checkpoint import CheckpointStore

    ready = Path(args.ready)
    go = Path(args.go)

    def barrier() -> None:
        _write(
            ready,
            {
                "pid": os.getpid(),
                "ready_monotonic_s": time.monotonic(),
                "threads": _threads(),
                "rss": _rss_bytes(),
            },
        )
        deadline = time.monotonic() + 900.0
        while not go.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("scheduler benchmark barrier timed out")
            time.sleep(0.02)

    result = _run_checkpoint_refinement(
        canonical=paths["canonical"],
        warm_start=paths["warm"],
        graph_path=paths["graph"],
        robot="wuji_hand2_beta1_rh",
        collision_samples=paths["collision"],
        query_profile_id="adaptive_active_set_v1",
        coordinate_profile_id="local_seed_delta_v1",
        solver_profile_id="wuji_continuous_sequential_v1",
        execution_profile_id=PROFILE,
        start_frame=0,
        end_frame=60,
        checkpoint_root=shadow,
        output=None,
        asset_root=None,
        resume=True,
        max_wall_time=None,
        stop_after_frame=stop_after_frame,
        progress_json=shadow / "benchmark_progress.json",
        progress_log=None,
        force=False,
        pause_check=lambda: False,
        allow_shadow_while_queue_paused=True,
        ready_callback=barrier,
    )
    store = CheckpointStore(shadow)
    new_rows = [
        store.load_frame(index)[0]
        for index in range(start_frame, stop_after_frame + 1)
        if (shadow / "frames" / f"frame_{index:06d}.npz").is_file()
    ]
    times = [float(row["solve_time_s"]) for row in new_rows]
    _write(
        Path(args.result),
        {
            "source": str(source),
            "shadow_checkpoint": str(shadow),
            "setup_wall_s": time.monotonic() - started - sum(times),
            "steady_state_end_monotonic_s": time.monotonic(),
            "completed_frames": len(new_rows),
            "latency_s": times,
            "p50_s": statistics.median(times) if times else None,
            "p95_s": _p95(times),
            "max_s": max(times) if times else None,
            "threads": _threads(),
            "rss_bytes": _rss_bytes(),
            "status": result.get("status"),
        },
    )
    return 0


def _run_mode(
    mode: str, sources: list[Path], root: Path, *, start_frame: int, stop_after_frame: int
) -> dict[str, Any]:
    mode_root = root / mode
    go = mode_root / "GO"
    commands: list[tuple[subprocess.Popen[str], Path]] = []
    for index, source in enumerate(sources):
        ready = mode_root / f"worker_{index}.ready.json"
        result = mode_root / f"worker_{index}.result.json"
        command = [
            str(ENV_PYTHON),
            str(Path(__file__).resolve()),
            "worker",
            "--source",
            str(source),
            "--shadow-root",
            str(mode_root / "shadow"),
            "--ready",
            str(ready),
            "--go",
            str(go),
            "--result",
            str(result),
            "--start-frame",
            str(start_frame),
            "--stop-after-frame",
            str(stop_after_frame),
        ]
        commands.append((subprocess.Popen(command, text=True), result))
    deadline = time.monotonic() + 900.0
    while not all(
        (mode_root / f"worker_{index}.ready.json").is_file() for index in range(len(sources))
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{mode} workers did not reach the ready barrier")
        if any(process.poll() is not None for process, _ in commands):
            raise RuntimeError(f"{mode} worker exited before readiness")
        time.sleep(0.05)
    steady_start = time.monotonic()
    go.parent.mkdir(parents=True, exist_ok=True)
    go.write_text(str(steady_start), encoding="utf-8")
    exits = [process.wait() for process, _ in commands]
    if any(exits):
        raise RuntimeError(f"{mode} worker exits: {exits}")
    workers = [_read(result) for _, result in commands]
    expected_frames = stop_after_frame - start_frame + 1
    if any(int(row["completed_frames"]) != expected_frames for row in workers):
        raise RuntimeError(f"{mode} did not complete its fixed interval: {workers}")
    steady_end = max(float(row["steady_state_end_monotonic_s"]) for row in workers)
    frames = sum(int(row["completed_frames"]) for row in workers)
    return {
        "mode": mode,
        "workers": len(workers),
        "setup_s": max(float(row["setup_wall_s"]) for row in workers),
        "steady_state_s": steady_end - steady_start,
        "frames": frames,
        "aggregate_fps": frames / (steady_end - steady_start),
        "worker_rows": workers,
    }


def parent(args: argparse.Namespace) -> int:
    host = REPO.parent / ".toporetarget_host_control"
    host.mkdir(exist_ok=True)
    request = host / "compiled_exact_sign_benchmark.request.json"
    if request.exists():
        raise RuntimeError(
            "host exact-sign benchmark request is active; final benchmark is not permitted"
        )
    from toporetarget.retarget.final_jobs import pause_final_jobs, resume_final_jobs

    output = Path(args.output_root).resolve()
    bench_root = (
        REPO
        / ".local"
        / "experiments"
        / "stage12_scheduler_closeout"
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    dex = Path(args.source_a).resolve()
    oak = Path(args.source_b).resolve()
    for label, source in (("source_a", dex), ("source_b", oak)):
        if not (source / "checkpoints").is_dir():
            raise ValueError(f"{label} is not a Stage-12 formal selection root: {source}")
    resume_final_jobs(
        REPO, max_final_workers=2, reason="Stage-12 isolated scheduler A/B/C measurement"
    )
    try:
        modes = [
            _run_mode(
                "A_dexycb_single",
                [dex],
                bench_root,
                start_frame=args.start_frame,
                stop_after_frame=args.stop_after_frame,
            ),
            _run_mode(
                "B_oakink_single",
                [oak],
                bench_root,
                start_frame=args.start_frame,
                stop_after_frame=args.stop_after_frame,
            ),
            _run_mode(
                "C_dual",
                [dex, oak],
                bench_root,
                start_frame=args.start_frame,
                stop_after_frame=args.stop_after_frame,
            ),
        ]
    finally:
        pause_final_jobs(REPO, reason="Stage-12 scheduler benchmark complete")
    singles = [float(row["aggregate_fps"]) for row in modes[:2]]
    dual = modes[2]
    latency_ok = all(
        float(worker["p95_s"] or float("inf")) <= 30.0
        and float(worker["p50_s"] or float("inf"))
        <= 1.3 * float(single["worker_rows"][0]["p50_s"] or float("inf"))
        for worker, single in zip(dual["worker_rows"], modes[:2], strict=True)
    )
    selected = 2 if dual["aggregate_fps"] >= 1.10 * max(singles) and latency_ok else 1
    result = {
        "schema_version": "toporetarget.stage12.scheduler_closeout.v1",
        "benchmark_root": str(bench_root),
        "modes": modes,
        "selected_final_workers": selected,
        "selection_reason": "dual_thresholds_passed"
        if selected == 2
        else "dual_thresholds_not_met_fallback_to_single",
    }
    _write(output / "scheduler_closeout.json", result)
    with (output / "scheduler_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "workers", "setup_s", "steady_state_s", "frames", "aggregate_fps"],
        )
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in modes])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker_parser = sub.add_parser("worker")
    for name in ("source", "shadow-root", "ready", "go", "result"):
        worker_parser.add_argument(f"--{name}", required=True)
    worker_parser.add_argument("--start-frame", type=int, default=5)
    worker_parser.add_argument("--stop-after-frame", type=int, default=14)
    parent_parser = sub.add_parser("run")
    parent_parser.add_argument(
        "--output-root", type=Path, default=REPO / ".local/reports/stage12_completion"
    )
    parent_parser.add_argument("--start-frame", type=int, default=5)
    parent_parser.add_argument("--stop-after-frame", type=int, default=14)
    parent_parser.add_argument(
        "--source-a", type=Path, required=True, help="first completed v4 formal selection root"
    )
    parent_parser.add_argument(
        "--source-b", type=Path, required=True, help="second completed v4 formal selection root"
    )
    args = parser.parse_args()
    return worker(args) if args.command == "worker" else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
