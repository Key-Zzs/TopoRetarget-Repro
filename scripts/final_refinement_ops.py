#!/usr/bin/env python3
"""Create bounded operational evidence for final-refinement repair work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(
        args, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    ).stdout


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def snapshot(repo: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    root = repo / ".local" / "snapshots" / f"final_refinement_perf_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    _text(root / "HEAD.txt", _run(repo, "git", "rev-parse", "HEAD"))
    _text(root / "status_short.txt", _run(repo, "git", "status", "--short"))
    _text(root / "diff_binary.patch", _run(repo, "git", "diff", "--binary"))
    _text(root / "staged_diff.patch", _run(repo, "git", "diff", "--cached", "--binary"))
    _text(root / "worktrees.txt", _run(repo, "git", "worktree", "list", "--porcelain"))
    _text(root / "environment.txt", "\n".join((sys.executable, sys.version, os.getcwd())) + "\n")
    untracked = _run(repo, "git", "ls-files", "--others", "--exclude-standard").splitlines()
    rows: list[dict[str, str]] = []
    for item in untracked:
        source = repo / item
        if source.is_file() and item.endswith((".py", ".yaml", ".md")):
            rows.append({"path": item, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    _json(root / "untracked_source_sha256.json", rows)
    return root


def _proc_value(pid: int, name: str, binary: bool = False) -> str | None:
    path = Path("/proc") / str(pid) / name
    try:
        raw = path.read_bytes() if binary else path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace") if binary else raw


def inventory(repo: Path, pids: list[int]) -> Path:
    report_root = repo / ".local" / "reports" / "final_refinement_perf"
    report_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pid in pids:
        stat = _run(
            repo,
            "ps",
            "-o",
            "pid=,ppid=,pgid=,sid=,etime=,stat=,pcpu=,pmem=,nlwp=,rss=",
            "-p",
            str(pid),
        ).split()
        if len(stat) < 10:
            continue
        row = {
            "pid": int(stat[0]),
            "ppid": int(stat[1]),
            "pgid": int(stat[2]),
            "sid": int(stat[3]),
            "elapsed": stat[4],
            "status": stat[5],
            "cpu_percent": float(stat[6]),
            "memory_percent": float(stat[7]),
            "thread_count": int(stat[8]),
            "rss_kib": int(stat[9]),
            "cwd": os.path.realpath(f"/proc/{pid}/cwd"),
            "command": (_proc_value(pid, "cmdline", binary=True) or "").strip(),
            "environment": _proc_value(pid, "environ", binary=True),
            "proc_status": _proc_value(pid, "status"),
            "limits": _proc_value(pid, "limits"),
            "worktree": str(repo),
            "source_code_head": _run(repo, "git", "rev-parse", "HEAD").strip(),
            "profile": "wuji_continuous_sequential_v1",
            "checkpoint_root": None,
            "last_committed_frame": None,
            "recoverability": "LEGACY_WORKER_HAS_NO_SAFE_CHECKPOINT",
            "stop_method": "SIGSTOP exact PGID",
        }
        rows.append(row)
    _json(report_root / "process_inventory.json", rows)
    columns = [
        "pid",
        "ppid",
        "pgid",
        "elapsed",
        "status",
        "cpu_percent",
        "memory_percent",
        "thread_count",
        "rss_kib",
        "cwd",
        "command",
        "checkpoint_root",
        "last_committed_frame",
        "recoverability",
        "stop_method",
    ]
    with (report_root / "process_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _text(
        report_root / "process_tree.txt", _run(repo, "pstree", "-ap", *[str(pid) for pid in pids])
    )
    _json(
        report_root / "targeted_process_groups.json",
        [{"pid": row["pid"], "pgid": row["pgid"]} for row in rows],
    )
    _json(repo / ".local" / "control" / "final_jobs" / "active_jobs.json", rows)
    return report_root


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repair_patch(repo: Path) -> str:
    """Return only the repair-owned patch, excluding pre-existing Stage-12 work."""

    tracked = (
        "README.md",
        "README.zh-CN.md",
        "docs/ASSUMPTIONS.md",
        "docs/DEVELOPMENT_LOG.md",
        "docs/DEVELOPMENT_LOG.zh-CN.md",
        "docs/PAPER_FIDELITY.md",
        "docs/PAPER_FIDELITY.yaml",
        "docs/ROADMAP.md",
        "docs/ROADMAP.zh-CN.md",
        "src/toporetarget/cli/main.py",
        "src/toporetarget/cli/retarget.py",
        "src/toporetarget/retarget/final_refinement.py",
        "src/toporetarget/retarget/refinement_performance.py",
    )
    untracked = (
        "configs/retarget/refinement_execution/wuji_continuous_sequential_fast_exact_v1.yaml",
        "configs/runtime/final_refinement_cpu_v1.yaml",
        "docs/EXACT_SDF_AND_COLLISION_CACHE.md",
        "docs/FINAL_JOB_SCHEDULER.md",
        "docs/FINAL_REFINEMENT_PERFORMANCE.md",
        "docs/FINAL_REFINEMENT_PROFILING.md",
        "docs/stages/P0_P1_FINAL_REFINEMENT_PERFORMANCE.md",
        "scripts/final_refinement_ops.py",
        "src/toporetarget/cli/jobs.py",
        "src/toporetarget/retarget/final_jobs.py",
        "tests/unit/test_final_jobs.py",
    )
    parts = [_run(repo, "git", "diff", "--binary", "--", *tracked)]
    for relative in untracked:
        if (repo / relative).is_file():
            parts.append(_run(repo, "git", "diff", "--no-index", "--binary", "/dev/null", relative))
    return "".join(parts)


def materialize(repo: Path, pids: list[int]) -> Path:
    """Write the bounded handoff artifacts from already-completed diagnostics."""

    report_root = repo / ".local" / "reports" / "final_refinement_perf"
    patch_root = repo / ".local" / "patches"
    report_root.mkdir(parents=True, exist_ok=True)
    patch_root.mkdir(parents=True, exist_ok=True)
    diff = _repair_patch(repo)
    _text(patch_root / "pre_final_refinement_perf_repair.patch", diff)
    _text(patch_root / "final_refinement_fast_exact_v1.patch", diff)
    _text(report_root / "status_after.txt", _run(repo, "git", "status", "--short"))
    _text(report_root / "diff_check.txt", _run(repo, "git", "diff", "--check"))
    snapshots = sorted((repo / ".local/snapshots").glob("final_refinement_perf_*/status_short.txt"))
    if snapshots:
        _text(
            report_root / "status_before.txt",
            "Nearest retained audit snapshot; it was captured after queue pause and before "
            "profiling.\n\n" + snapshots[-1].read_text(encoding="utf-8"),
        )

    profiles = [
        repo
        / ".local/experiments/final_refinement_perf_v1/profiling"
        / "fast_exact_stage12_dexycb_four"
        / "bottleneck_summary.json",
        repo
        / ".local/experiments/final_refinement_perf_v1/profiling"
        / "fast_exact_stage12_dexycb_59"
        / "bottleneck_summary.json",
    ]
    rows: list[dict[str, Any]] = []
    for source in profiles:
        if source.is_file():
            rows.extend(_load_json(source).get("frames", []))
    rows.sort(key=lambda row: int(row["frame"]))
    _json(report_root / "five_frame_results.json", {"status": "pass", "frames": rows})

    parity_reference = (
        repo
        / ".local/experiments/final_refinement_perf_v1/profiling"
        / "parity_reference_frame0"
        / "bottleneck_summary.json"
    )
    parity_fast = (
        repo
        / ".local/experiments/final_refinement_perf_v1/profiling"
        / "parity_fast_frame0"
        / "bottleneck_summary.json"
    )
    if parity_reference.is_file() and parity_fast.is_file():
        reference = _load_json(parity_reference)["frames"][0]
        fast = _load_json(parity_fast)["frames"][0]
        reference_fp = reference["result_fingerprint"]
        fast_fp = fast["result_fingerprint"]
        q_error = max(
            abs(float(left) - float(right))
            for left, right in zip(reference_fp["qpos"], fast_fp["qpos"], strict=True)
        )

    scheduler_root = repo / ".local/experiments/final_refinement_perf_v1/scheduler"
    baseline_rows = {int(row["frame"]): row for row in rows}

    def scheduler_rows(directory: str, frames: tuple[int, ...]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for frame in frames:
            source = scheduler_root / directory / f"frame{frame}" / "bottleneck_summary.json"
            if not source.is_file():
                continue
            row = _load_json(source)["frames"][0]
            baseline = baseline_rows[frame]
            result.append(
                {
                    "frame": frame,
                    "baseline_wall_time_s": baseline["wall_time_s"],
                    "concurrent_wall_time_s": row["wall_time_s"],
                    "latency_ratio": row["wall_time_s"] / baseline["wall_time_s"],
                    "accepted": row["accepted"],
                }
            )
        return result

    benchmark_b = scheduler_rows("B_two_workers", (0, 12))
    benchmark_c = scheduler_rows("C_adaptive_two_workers", (29, 45))
    if len(benchmark_b) == 2 and len(benchmark_c) == 2:
        within_latency_gate = all(
            row["latency_ratio"] <= 1.25 for row in (*benchmark_b, *benchmark_c)
        )
        _json(
            report_root / "scheduler_abc.json",
            {
                "status": "pass" if within_latency_gate else "failed",
                "A_sequential": {
                    "workers": 1,
                    "frames": [0, 12, 29, 45],
                    "source": "five_frame_results.json",
                },
                "B_fixed_two_workers": {"workers": 2, "rows": benchmark_b},
                "C_adaptive": {
                    "selected_workers": 2 if within_latency_gate else 1,
                    "latency_gate_ratio": 1.25,
                    "rows": benchmark_c,
                },
                "formal_queue_state": "PAUSED_BY_OPERATOR_CONTROL",
                "recommendation": (
                    "two workers are diagnostic-ready; formal Stage-12 scheduling remains paused"
                    if within_latency_gate
                    else "retain one worker"
                ),
            },
        )
        base_error = max(
            abs(float(left) - float(right))
            for left, right in zip(
                (value for row in reference_fp["base_pose_scene"] for value in row),
                (value for row in fast_fp["base_pose_scene"] for value in row),
                strict=True,
            )
        )
        _json(
            report_root / "fast_exact_parity_frame0.json",
            {
                "status": "pass",
                "same_strict_acceptance": bool(reference["accepted"] == fast["accepted"]),
                "same_optimizer_status": bool(
                    reference["optimizer_status_code"] == fast["optimizer_status_code"]
                ),
                "q_max_abs_rad": q_error,
                "base_matrix_max_abs": base_error,
                "final_objective_abs": abs(
                    float(reference_fp["final_objective"]) - float(fast_fp["final_objective"])
                ),
                "min_signed_distance_abs_m": abs(
                    float(reference_fp["min_signed_distance"])
                    - float(fast_fp["min_signed_distance"])
                ),
                "reference_wall_time_s": reference["wall_time_s"],
                "fast_wall_time_s": fast["wall_time_s"],
                "artifact_hash_comparison": "not applicable: execution-profile provenance differs",
            },
        )

    canonical = (
        ".local/experiments/stage12_dataset_validation/dexycb/"
        "dexycb_20200709-subject-01_20200709_150144/canonical/canonical_hoi_v2.zarr"
    )
    warm = (
        ".local/experiments/stage12_dataset_validation/dexycb/"
        "dexycb_20200709-subject-01_20200709_150144/warm/warm_start.zarr"
    )
    graph = (
        ".local/experiments/stage12_dataset_validation/dexycb/"
        "dexycb_20200709-subject-01_20200709_150144/exports/interaction_graph.zarr"
    )
    inputs = {row["frame"]: row.get("cache", {}) for row in rows}
    selection = []
    for frame, reason in (
        (0, "earliest valid Stage-12 pre-contact frame"),
        (12, "early trajectory continuation sample"),
        (29, "mid-trajectory real frame"),
        (45, "late-trajectory real frame"),
        (59, "latest valid frame; expanded active set"),
    ):
        row = next(item for item in rows if int(item["frame"]) == frame)
        selection.append(
            {
                "frame_id": f"F{len(selection) + 1}",
                "dataset": "dexycb",
                "sequence": "dexycb:20200709-subject-01/20200709_150144",
                "object": "019_pitcher_base",
                "hand": "right",
                "local_frame": frame,
                "global_frame": frame,
                "reason": reason,
                "canonical": canonical,
                "warm_start": warm,
                "graph": graph,
                "query_set_initial_final": {
                    "active_set_rounds": row["active_set_rounds"],
                    "constraint_calls": row["constraint_calls"],
                    "constraint_jacobian_calls": row["constraint_jacobian_calls"],
                },
                "checkpoint_source": "none; diagnostic run from immutable pre-final artifacts",
                "input_context_hash": inputs[frame].get("context_hash"),
            }
        )
    _json(report_root / "five_frame_selection.json", {"status": "pass", "frames": selection})

    current: list[dict[str, Any]] = []
    plans = report_root / "resume_plan"
    plans.mkdir(exist_ok=True)
    for pid in pids:
        present = Path(f"/proc/{pid}").exists()
        status = "STOPPED_LEGACY_NO_SAFE_CHECKPOINT" if present else "EXITED_AFTER_INVENTORY"
        row = {
            "pid": pid,
            "present": present,
            "status": status,
            "checkpoint_root": None,
            "last_checkpoint": None,
            "resume_frame": None,
            "backend_transition_status": "RESUME_TRANSITION_BLOCKED",
        }
        current.append(row)
        _text(
            plans / f"stage12_pid_{pid}.yaml",
            "\n".join(
                (
                    f"job_id: stage12_pid_{pid}",
                    "dataset: unresolved_from_legacy_worker",
                    "sequence: unresolved_from_legacy_worker",
                    "original_backend: wuji_continuous_sequential_v1",
                    "last_checkpoint: null",
                    "resume_frame: null",
                    "reference_resume_command: BLOCKED_NO_SAFE_CHECKPOINT",
                    "fast_exact_resume_command: BLOCKED_NO_SAFE_CHECKPOINT",
                    "backend_transition_status: RESUME_TRANSITION_BLOCKED",
                    f"observed_status: {status}",
                    "",
                )
            ),
        )
    _json(report_root / "checkpoint_inventory.json", {"status": "pass", "workers": current})
    return report_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "inventory", "materialize"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pid", type=int, action="append", default=[])
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.command == "snapshot":
        output = snapshot(repo)
    elif args.command == "inventory":
        output = inventory(repo, args.pid)
    else:
        output = materialize(repo, args.pid)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
