#!/usr/bin/env python3
"""Qualify one frozen retarget execution profile before the five-clip run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    BatchContractError,
    atomic_write_json,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

SOLVER_PROFILE_ID = "wuji_continuous_sequential_v1"
EXECUTION_PROFILE_ID = "wuji_continuous_sequential_fast_exact_v2"
BASELINE_EXECUTION_PROFILE_ID = "cached_checkpoint_cpu_float64_v1"
DEFAULT_FRAMES = (0, 12, 29, 45, 59)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--max-seconds-per-frame", type=float, default=30.0)
    parser.add_argument("--max-q-difference-rad", type=float, default=1.0e-5)
    parser.add_argument("--asset-root", type=Path)
    return parser


def _metadata(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise BatchContractError(f"GEOMETRIC_QUALIFICATION_CHECKPOINT_MISSING:{path}")
    with np.load(path, allow_pickle=False) as value:
        qpos = np.asarray(value["qpos"], dtype=np.float64)
        metadata = json.loads(str(value["metadata_json"].item()))
    return qpos, metadata


def _command(
    *,
    canonical: Path,
    warm_start: Path,
    graph: Path,
    frame: int,
    execution_profile: str,
    checkpoint_root: Path,
    final: Path,
    asset_root: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "toporetarget",
        "retarget",
        "refine",
        "--canonical",
        str(canonical),
        "--warm-start",
        str(warm_start),
        "--graph",
        str(graph),
        "--robot",
        "wuji_hand2_beta1_rh",
        "--solver-profile",
        SOLVER_PROFILE_ID,
        "--execution-profile",
        execution_profile,
        "--start-frame",
        str(frame),
        "--end-frame",
        str(frame + 1),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output",
        str(final),
    ]
    if asset_root is not None:
        command.extend(["--asset-root", str(asset_root)])
    return command


def _execute(command: list[str], *, environment: dict[str, str], log: Path) -> dict[str, Any]:
    tick = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - tick
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    return {
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": wall_seconds,
        "log": str(log.resolve()),
        "log_sha256": sha256_file(log),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.report.exists() or args.output_root.exists():
        raise FileExistsError("GEOMETRIC_QUALIFICATION_REFUSES_OVERWRITE")
    if args.max_seconds_per_frame <= 0.0 or args.max_q_difference_rad < 0.0:
        raise ValueError("GEOMETRIC_QUALIFICATION_THRESHOLD_INVALID")
    frames = tuple(int(frame) for frame in args.frames)
    if len(frames) != len(set(frames)) or any(frame < 0 for frame in frames):
        raise ValueError("GEOMETRIC_QUALIFICATION_FRAMES_INVALID")
    execution = RefinementExecutionProfile.load(EXECUTION_PROFILE_ID, REPO_ROOT)
    baseline_execution = RefinementExecutionProfile.load(BASELINE_EXECUTION_PROFILE_ID, REPO_ROOT)
    if not (
        execution.math_equivalent
        and execution.paper_objective_unchanged
        and execution.paper_constraints_unchanged
        and execution.continuity_contract_unchanged
        and execution.final_full_surface_audit
    ):
        raise BatchContractError("GEOMETRIC_EXECUTION_PROFILE_NOT_QUALIFIED")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    rows: list[dict[str, Any]] = []
    for frame in frames:
        frame_root = args.output_root / f"frame_{frame:06d}"
        baseline_root = frame_root / "baseline"
        candidate_root = frame_root / "candidate"
        baseline_run = _execute(
            _command(
                canonical=args.canonical,
                warm_start=args.warm_start,
                graph=args.graph,
                frame=frame,
                execution_profile=baseline_execution.profile_id,
                checkpoint_root=baseline_root / "checkpoints",
                final=baseline_root / "final.zarr",
                asset_root=args.asset_root,
            ),
            environment=environment,
            log=baseline_root / "run.log",
        )
        candidate_run = _execute(
            _command(
                canonical=args.canonical,
                warm_start=args.warm_start,
                graph=args.graph,
                frame=frame,
                execution_profile=execution.profile_id,
                checkpoint_root=candidate_root / "checkpoints",
                final=candidate_root / "final.zarr",
                asset_root=args.asset_root,
            ),
            environment=environment,
            log=candidate_root / "run.log",
        )
        candidate_path = candidate_root / "checkpoints/frames" / f"frame_{frame:06d}.npz"
        baseline_path = baseline_root / "checkpoints/frames" / f"frame_{frame:06d}.npz"
        row: dict[str, Any] = {
            "frame": frame,
            "baseline": baseline_run,
            "candidate": candidate_run,
        }
        if baseline_run["returncode"] == 0 and candidate_run["returncode"] == 0:
            candidate_q, metadata = _metadata(candidate_path)
            baseline_q, baseline_metadata = _metadata(baseline_path)
            if candidate_q.shape != baseline_q.shape:
                raise BatchContractError("GEOMETRIC_QUALIFICATION_QPOS_SHAPE_MISMATCH")
            row.update(
                {
                    "strict_accepted": metadata.get("strict_accepted") is True,
                    "optimizer_status_code": metadata.get("optimizer_status_code"),
                    "solve_time_s": metadata.get("solve_time_s"),
                    "baseline_strict_accepted": baseline_metadata.get("strict_accepted") is True,
                    "baseline_optimizer_status_code": baseline_metadata.get(
                        "optimizer_status_code"
                    ),
                    "baseline_solve_time_s": baseline_metadata.get("solve_time_s"),
                    "qpos_max_abs_difference_rad": float(np.max(np.abs(candidate_q - baseline_q))),
                    "baseline_execution_profile_id": baseline_metadata.get(
                        "execution_profile", {}
                    ).get("profile_id"),
                }
            )
        row["checks"] = {
            "baseline_process_pass": baseline_run["returncode"] == 0,
            "candidate_process_pass": candidate_run["returncode"] == 0,
            "baseline_strict_accepted": row.get("baseline_strict_accepted") is True,
            "baseline_optimizer_status_zero": row.get("baseline_optimizer_status_code") == 0,
            "strict_accepted": row.get("strict_accepted") is True,
            "optimizer_status_zero": row.get("optimizer_status_code") == 0,
            "runtime_pass": candidate_run["wall_seconds"] <= args.max_seconds_per_frame,
            "parity_pass": row.get("qpos_max_abs_difference_rad", float("inf"))
            <= args.max_q_difference_rad,
        }
        row["status"] = "PASS" if all(row["checks"].values()) else "FAIL"
        rows.append(row)
        if row["status"] != "PASS":
            break

    complete = len(rows) == len(frames)
    failures = [row["frame"] for row in rows if row["status"] != "PASS"]
    report = {
        "schema_version": "IndependentGeometricExecutionQualificationV1",
        "status": "PASS" if complete and not failures else "FAIL",
        "solver_profile_id": SOLVER_PROFILE_ID,
        "baseline_execution_profile_id": baseline_execution.profile_id,
        "baseline_execution_profile_sha256": baseline_execution.profile_hash,
        "execution_profile_id": execution.profile_id,
        "execution_profile_sha256": execution.profile_hash,
        "frames": list(frames),
        "max_seconds_per_frame": args.max_seconds_per_frame,
        "max_q_difference_rad": args.max_q_difference_rad,
        "completed_frames": len(rows),
        "failed_frames": failures,
        "rows": rows,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
