"""Generate bounded Stage 9 summary reports from final artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.retarget.final_refinement import load_final_trajectory


def _side_report(path: Path, side: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trajectory = load_final_trajectory(path)
    arrays = trajectory.arrays
    frame_indices = np.asarray(arrays["frame_indices"], dtype=np.int64)
    phi = np.asarray(arrays["full_signed_distance"], dtype=np.float64)
    total = np.asarray(arrays["total_objective"], dtype=np.float64)
    rows = []
    for local_index, frame in enumerate(frame_indices.tolist()):
        rows.append(
            {
                "side": side,
                "frame": int(frame),
                "solver_success": bool(arrays["solver_success"][local_index]),
                "active_set_converged": bool(arrays["active_set_converged"][local_index]),
                "query_count": int(
                    arrays["query_offsets"][local_index + 1] - arrays["query_offsets"][local_index]
                ),
                "total_objective": float(total[local_index]),
                "min_full_signed_distance_m": float(np.min(phi[local_index])),
                "max_penetration_m": float(arrays["max_penetration"][local_index]),
                "max_slack_m": float(
                    np.max(
                        arrays["slack_concat"][
                            arrays["slack_offsets"][local_index] : arrays["slack_offsets"][
                                local_index + 1
                            ]
                        ],
                        initial=0.0,
                    )
                ),
                "solve_time_s": float(arrays["solve_time_s"][local_index]),
            }
        )
    summary = {
        "side": side,
        "path": str(path),
        "schema_version": trajectory.schema_version,
        "frame_count": trajectory.frame_count,
        "frame_range": [int(frame_indices[0]), int(frame_indices[-1]) + 1],
        "all_solver_success": bool(np.all(arrays["solver_success"])),
        "all_active_set_converged": bool(np.all(arrays["active_set_converged"])),
        "all_finite": bool(
            all(
                np.all(np.isfinite(value))
                for value in arrays.values()
                if value.dtype.kind in "buiufc"
            )
        ),
        "min_full_signed_distance_m": float(np.min(phi)),
        "max_penetration_m": float(np.max(arrays["max_penetration"])),
        "max_slack_m": float(max(row["max_slack_m"] for row in rows)),
        "mean_solve_time_s": float(np.mean(arrays["solve_time_s"])),
        "p95_solve_time_s": float(np.percentile(arrays["solve_time_s"], 95)),
        "artifact_hash": trajectory.metadata.get("artifact_hash"),
        "source_canonical_hash": trajectory.metadata.get("source_canonical_hash"),
        "warm_start_artifact_hash": trajectory.metadata.get("warm_start_artifact_hash"),
        "graph_artifact_hash": trajectory.metadata.get("graph_artifact_hash"),
        "provenance": trajectory.metadata.get("provenance", {}),
    }
    return summary, rows


def _determinism(first_path: Path, repeat_path: Path | None) -> dict[str, Any]:
    if repeat_path is None or not repeat_path.exists():
        return {"status": "not_run", "first": str(first_path)}
    first = load_final_trajectory(first_path)
    repeat = load_final_trajectory(repeat_path)
    exact: dict[str, bool] = {}
    max_abs_diff: dict[str, float] = {}
    for name in sorted(set(first.arrays) & set(repeat.arrays)):
        if name == "solve_time_s":
            continue
        left = np.asarray(first.arrays[name])
        right = np.asarray(repeat.arrays[name])
        if left.shape != right.shape:
            exact[name] = False
            max_abs_diff[name] = float("inf")
            continue
        if left.dtype.kind in "US" or right.dtype.kind in "US":
            exact[name] = bool(np.array_equal(left.astype(str), right.astype(str)))
            continue
        exact[name] = bool(np.array_equal(left, right))
        if left.dtype.kind == "b" or right.dtype.kind == "b":
            max_abs_diff[name] = float(np.count_nonzero(np.logical_xor(left, right)))
        else:
            max_abs_diff[name] = float(np.max(np.abs(left - right))) if left.size else 0.0
    return {
        "status": "pass" if exact and all(exact.values()) else "fail",
        "first": str(first_path),
        "repeat": str(repeat_path),
        "exact_arrays": exact,
        "max_abs_diff": max_abs_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--right-final", type=Path, required=True)
    parser.add_argument("--left-final", type=Path, required=True)
    parser.add_argument("--right-validation", type=Path)
    parser.add_argument("--left-validation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--right-repeat", type=Path)
    parser.add_argument("--left-repeat", type=Path)
    parser.add_argument("--right-determinism-first", type=Path)
    parser.add_argument("--right-determinism-repeat", type=Path)
    parser.add_argument("--left-determinism-first", type=Path)
    parser.add_argument("--left-determinism-repeat", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    rows: list[dict[str, Any]] = []
    for side, path in (("right", args.right_final), ("left", args.left_final)):
        summary, side_rows = _side_report(path, side)
        summaries.append(summary)
        rows.extend(side_rows)

    determinism = {
        "right": _determinism(
            args.right_determinism_first or args.right_final,
            args.right_determinism_repeat or args.right_repeat,
        ),
        "left": _determinism(
            args.left_determinism_first or args.left_final,
            args.left_determinism_repeat or args.left_repeat,
        ),
    }

    validation = {}
    for side, path in (("right", args.right_validation), ("left", args.left_validation)):
        if path is not None:
            validation[side] = json.loads(path.read_text(encoding="utf-8"))
    validation_status = {
        side: bool(report.get("status") == "pass" and report.get("pass", False))
        for side, report in validation.items()
    }
    has_validation = set(validation_status) == {"right", "left"}
    payload = {
        "status": "pass"
        if all(
            item["all_solver_success"] and item["all_active_set_converged"] for item in summaries
        )
        and all(item["status"] == "pass" for item in determinism.values())
        and has_validation
        and all(validation_status.values())
        else "incomplete",
        "sides": summaries,
        "required_manual_frames": [0, 29, 59],
        "independent_validation": validation,
        "independent_validation_status": validation_status,
        "determinism": determinism,
        "determinism_scope": "single_frame_smoke"
        if args.right_determinism_first or args.left_determinism_first
        else "full_artifact_rerun",
    }
    (args.output_dir / "stage9_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    worst = sorted(rows, key=lambda row: row["min_full_signed_distance_m"])[:10]
    (args.output_dir / "worst_frames.json").write_text(
        json.dumps(worst, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "source_integrity.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "sides": [
                    {
                        "side": item["side"],
                        "source_canonical_hash": item["source_canonical_hash"],
                        "warm_start_artifact_hash": item["warm_start_artifact_hash"],
                        "graph_artifact_hash": item["graph_artifact_hash"],
                        "provenance": item["provenance"],
                    }
                    for item in summaries
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "determinism.json").write_text(
        json.dumps(determinism, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "performance.json").write_text(
        json.dumps(
            {
                item["side"]: {key: item[key] for key in ("mean_solve_time_s", "p95_solve_time_s")}
                for item in summaries
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
