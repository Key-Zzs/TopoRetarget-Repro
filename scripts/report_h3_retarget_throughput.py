#!/usr/bin/env python3
"""Compare the frozen H3-B baseline and optimized retarget matrices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.retarget.final_refinement import load_final_trajectory

RUNTIME_ONLY_ARRAYS = frozenset({"solve_time_s"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--baseline-report-root", type=Path, required=True)
    parser.add_argument("--optimized-run-root", type=Path, required=True)
    parser.add_argument("--optimized-report-root", type=Path, required=True)
    parser.add_argument("--resume-parity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("H3B_REPORT_ROWS_EMPTY")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _artifact_path(run_root: Path, case: dict[str, Any], repeat: int) -> Path:
    root = run_root / str(case["case_id"]) / f"repeat_{repeat:02d}"
    if "prepared_root" not in case:
        root = root / str(case["episode_id"])
    return root / "retarget/final_continuous.zarr"


def _validation_path(report_root: Path, case: dict[str, Any], repeat: int) -> Path:
    root = report_root / "runs" / str(case["case_id"]) / f"repeat_{repeat:02d}"
    if "prepared_root" not in case:
        root = root / "episodes" / str(case["episode_id"])
    return root / "retarget/continuous_final_validation.json"


def _selected_attempt_profile(
    run_root: Path, case: dict[str, Any], repeat: int
) -> dict[str, float | int | str]:
    checkpoint_root = _artifact_path(run_root, case, repeat).parent / "continuous_checkpoints"
    frames = sorted((checkpoint_root / "frames").glob("frame_*.npz"))
    if not frames:
        raise FileNotFoundError(f"H3B_PROFILE_CHECKPOINT_FRAMES_MISSING:{checkpoint_root}")
    totals = {
        "selected_attempt_slsqp_seconds": 0.0,
        "active_set_discovery_seconds": 0.0,
        "final_full_audit_seconds": 0.0,
        "selected_attempt_solve_seconds": 0.0,
    }
    reused = 0
    physical_final_queries = 0
    for path in frames:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        diagnostics = metadata.get("diagnostics", {})
        elapsed = diagnostics.get("timers", {}).get("elapsed_s", {})
        totals["selected_attempt_slsqp_seconds"] += float(elapsed.get("slsqp_total", 0.0))
        totals["active_set_discovery_seconds"] += float(elapsed.get("active_set_discovery", 0.0))
        totals["final_full_audit_seconds"] += float(elapsed.get("final_full_audit", 0.0))
        totals["selected_attempt_solve_seconds"] += float(metadata["solve_time_s"])
        reused += int(bool(diagnostics.get("final_audit_query_reused", False)))
        physical_final_queries += int(
            diagnostics.get("physical_reference_final_audit_query_count", 1)
        )
    count = len(frames)
    return {
        "case_id": str(case["case_id"]),
        "repeat": repeat,
        "frames": count,
        **totals,
        "selected_attempt_slsqp_ms_per_frame": 1000.0
        * totals["selected_attempt_slsqp_seconds"]
        / count,
        "active_set_discovery_ms_per_frame": 1000.0
        * totals["active_set_discovery_seconds"]
        / count,
        "final_full_audit_ms_per_frame": 1000.0 * totals["final_full_audit_seconds"] / count,
        "final_audit_query_reuse_frames": reused,
        "physical_reference_final_audit_query_count": physical_final_queries,
    }


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(left.dtype, np.complexfloating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _compare_artifacts(baseline: Path, optimized: Path) -> dict[str, Any]:
    before = load_final_trajectory(baseline)
    after = load_final_trajectory(optimized)
    before_names = set(before.arrays) - RUNTIME_ONLY_ARRAYS
    after_names = set(after.arrays) - RUNTIME_ONLY_ARRAYS
    rows: list[dict[str, Any]] = []
    for name in sorted(before_names | after_names):
        left = before.arrays.get(name)
        right = after.arrays.get(name)
        present = left is not None and right is not None
        equal = bool(present and _arrays_equal(np.asarray(left), np.asarray(right)))
        maximum_absolute_difference: float | None = None
        if present and np.issubdtype(np.asarray(left).dtype, np.number):
            a = np.asarray(left, dtype=np.float64)
            b = np.asarray(right, dtype=np.float64)
            if a.shape == b.shape and a.size:
                finite = np.isfinite(a) & np.isfinite(b)
                maximum_absolute_difference = (
                    float(np.max(np.abs(a[finite] - b[finite]))) if np.any(finite) else 0.0
                )
        rows.append(
            {
                "array": name,
                "present_both": present,
                "shape_before": None if left is None else list(np.asarray(left).shape),
                "shape_after": None if right is None else list(np.asarray(right).shape),
                "dtype_before": None if left is None else str(np.asarray(left).dtype),
                "dtype_after": None if right is None else str(np.asarray(right).dtype),
                "sha256_before": None if left is None else _array_hash(np.asarray(left)),
                "sha256_after": None if right is None else _array_hash(np.asarray(right)),
                "maximum_absolute_difference": maximum_absolute_difference,
                "bitwise_equal_with_nan_equivalence": equal,
            }
        )
    return {
        "status": (
            "PASS"
            if before_names == after_names
            and all(row["bitwise_equal_with_nan_equivalence"] for row in rows)
            else "FAIL"
        ),
        "runtime_only_arrays_excluded": sorted(RUNTIME_ONLY_ARRAYS),
        "arrays": rows,
    }


def _compare_validation(baseline: Path, optimized: Path) -> dict[str, Any]:
    before = json.loads(baseline.read_text(encoding="utf-8"))
    after = json.loads(optimized.read_text(encoding="utf-8"))
    before.pop("final", None)
    after.pop("final", None)
    return {
        "status": "PASS" if before == after else "FAIL",
        "path_only_fields_excluded": ["final"],
        "before": str(baseline.resolve()),
        "after": str(optimized.resolve()),
    }


def _median_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    fields = (
        "frames",
        "solver_ms_per_frame",
        "checkpoint_payload_seconds",
        "checkpoint_serialization_seconds",
        "append_write_seconds",
        "durable_checkpoint_seconds",
        "assembly_validation_seconds",
        "full_frame_validation_seconds",
        "html_generation_seconds",
        "orchestration_seconds",
        "total_seconds",
    )
    return {
        case: {field: statistics.median(float(row[field]) for row in case_rows) for field in fields}
        for case, case_rows in grouped.items()
    }


def _before_after(
    baseline: list[dict[str, str]], optimized: list[dict[str, str]]
) -> list[dict[str, object]]:
    before = _median_rows(baseline)
    after = _median_rows(optimized)
    if set(before) != set(after):
        raise ValueError("H3B_CASE_SET_DRIFT")
    rows: list[dict[str, object]] = []
    for case in before:
        b = before[case]
        a = after[case]
        frames = int(b["frames"])
        if frames != int(a["frames"]):
            raise ValueError(f"H3B_FRAME_COUNT_DRIFT:{case}")
        b_io = sum(
            b[key]
            for key in (
                "checkpoint_payload_seconds",
                "checkpoint_serialization_seconds",
                "append_write_seconds",
                "durable_checkpoint_seconds",
            )
        )
        a_io = sum(
            a[key]
            for key in (
                "checkpoint_payload_seconds",
                "checkpoint_serialization_seconds",
                "append_write_seconds",
                "durable_checkpoint_seconds",
            )
        )
        b_validation = b["assembly_validation_seconds"] + b["full_frame_validation_seconds"]
        a_validation = a["assembly_validation_seconds"] + a["full_frame_validation_seconds"]
        b_total_ms = 1000.0 * b["total_seconds"] / frames
        a_total_ms = 1000.0 * a["total_seconds"] / frames
        rows.append(
            {
                "case_id": case,
                "frames": frames,
                "baseline_solver_ms_per_frame": b["solver_ms_per_frame"],
                "optimized_solver_ms_per_frame": a["solver_ms_per_frame"],
                "baseline_io_ms_per_frame": 1000.0 * b_io / frames,
                "optimized_io_ms_per_frame": 1000.0 * a_io / frames,
                "baseline_validation_ms_per_frame": 1000.0 * b_validation / frames,
                "optimized_validation_ms_per_frame": 1000.0 * a_validation / frames,
                "baseline_html_seconds": b["html_generation_seconds"],
                "optimized_html_seconds": a["html_generation_seconds"],
                "baseline_orchestration_ms_per_frame": 1000.0 * b["orchestration_seconds"] / frames,
                "optimized_orchestration_ms_per_frame": 1000.0
                * a["orchestration_seconds"]
                / frames,
                "baseline_total_ms_per_frame": b_total_ms,
                "optimized_total_ms_per_frame": a_total_ms,
                "total_improvement_fraction": (b_total_ms - a_total_ms) / b_total_ms,
            }
        )
    return rows


def main() -> int:
    args = _parser().parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    baseline_report = args.baseline_report_root.resolve()
    optimized_report = args.optimized_report_root.resolve()
    baseline_rows = _read_rows(baseline_report / "fast_exact_v2_benchmark.csv")
    optimized_rows = _read_rows(optimized_report / "fast_exact_v2_benchmark.csv")
    keys = {(row["case_id"], row["repeat"]) for row in baseline_rows}
    if keys != {(row["case_id"], row["repeat"]) for row in optimized_rows}:
        raise ValueError("H3B_REPEAT_MATRIX_DRIFT")
    output = args.output_root.resolve()
    _write_rows(output / "baseline.csv", baseline_rows)
    _write_rows(output / "optimized.csv", optimized_rows)
    comparison_rows = _before_after(baseline_rows, optimized_rows)
    _write_rows(output / "before_after.csv", comparison_rows)

    profiling_rows: list[dict[str, object]] = []
    for phase, run_root in (
        ("baseline", args.baseline_run_root.resolve()),
        ("optimized", args.optimized_run_root.resolve()),
    ):
        for case in config["cases"]:
            repeats = int(case.get("repeats", config["repeats"]))
            for repeat in range(1, repeats + 1):
                profiling_rows.append(
                    {"phase": phase, **_selected_attempt_profile(run_root, case, repeat)}
                )
    _write_rows(output / "selected_attempt_profiling.csv", profiling_rows)

    parity_rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        repeats = int(case.get("repeats", config["repeats"]))
        for repeat in range(1, repeats + 1):
            artifact = _compare_artifacts(
                _artifact_path(args.baseline_run_root.resolve(), case, repeat),
                _artifact_path(args.optimized_run_root.resolve(), case, repeat),
            )
            validation = _compare_validation(
                _validation_path(baseline_report, case, repeat),
                _validation_path(optimized_report, case, repeat),
            )
            parity_rows.append(
                {
                    "case_id": case["case_id"],
                    "repeat": repeat,
                    "trajectory": artifact,
                    "validation_receipt": validation,
                    "status": (
                        "PASS" if artifact["status"] == validation["status"] == "PASS" else "FAIL"
                    ),
                }
            )
    math_parity = {
        "schema_version": "H3RetargetMathParityV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in parity_rows) else "FAIL",
        "RETARGET_MATH_PARITY": (
            "PASS" if all(row["status"] == "PASS" for row in parity_rows) else "FAIL"
        ),
        "comparison": "bitwise_equal_with_nan_equivalence",
        "rows": parity_rows,
    }
    _write_json(output / "math_parity.json", math_parity)
    resume = json.loads(args.resume_parity.resolve().read_text(encoding="utf-8"))
    if resume.get("status") not in {"PASS", "FAIL"}:
        raise ValueError("H3B_RESUME_PARITY_RECEIPT_INVALID")
    mean_improvement = statistics.mean(
        float(row["total_improvement_fraction"]) for row in comparison_rows
    )
    all_validation = all(row["validation_receipt"]["status"] == "PASS" for row in parity_rows)
    gates_pass = math_parity["status"] == resume["status"] == "PASS" and all_validation
    baseline_profile = baseline_rows[0]["execution_profile"]
    optimized_profile = optimized_rows[0]["execution_profile"]
    if not gates_pass:
        decision = "H3B_INCONCLUSIVE"
        selected = baseline_profile
    elif mean_improvement > 0.01:
        decision = "H3B_THROUGHPUT_HARDENING_VALIDATED"
        selected = optimized_profile
    elif mean_improvement >= -0.01:
        decision = "H3B_NO_MEASURABLE_SPEEDUP"
        selected = baseline_profile
    else:
        decision = "H3B_REGRESSION_REVERTED"
        selected = baseline_profile
    final = {
        "schema_version": "H3RetargetThroughputDecisionV1",
        "status": "PASS" if gates_pass else "FAIL",
        "decision": decision,
        "H3B_SELECTED_RETARGET_EXECUTION_CONTRACT": selected,
        "RETARGET_MATH_PARITY": math_parity["status"],
        "RESUME_PARITY": resume["status"],
        "ALL_FRAME_VALIDATION": "PASS" if all_validation else "FAIL",
        "RETARGET_MATH_CHANGED": "NO" if math_parity["status"] == "PASS" else "UNKNOWN",
        "mean_case_total_improvement_fraction": mean_improvement,
        "selection_threshold_fraction": 0.01,
        "candidate_profile": optimized_profile,
        "baseline_profile": baseline_profile,
        "selected_attempt_profiling": str((output / "selected_attempt_profiling.csv").resolve()),
    }
    _write_json(output / "final_decision.json", final)
    lines = [
        "# H3-B retarget throughput analysis",
        "",
        f"Decision: `{decision}`.",
        "",
        f"Math parity: `{math_parity['status']}`. Resume parity: `{resume['status']}`. "
        f"All-frame validation: `{'PASS' if all_validation else 'FAIL'}`.",
        "",
        "| Case | Frames | Solver before/after ms/f | I/O before/after ms/f | "
        "Validation before/after ms/f | HTML before/after s | Total before/after ms/f |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['case_id']} | {row['frames']} | "
            f"{row['baseline_solver_ms_per_frame']:.3f} / "
            f"{row['optimized_solver_ms_per_frame']:.3f} | "
            f"{row['baseline_io_ms_per_frame']:.3f} / "
            f"{row['optimized_io_ms_per_frame']:.3f} | "
            f"{row['baseline_validation_ms_per_frame']:.3f} / "
            f"{row['optimized_validation_ms_per_frame']:.3f} | "
            f"{row['baseline_html_seconds']:.3f} / {row['optimized_html_seconds']:.3f} | "
            f"{row['baseline_total_ms_per_frame']:.3f} / "
            f"{row['optimized_total_ms_per_frame']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`solver ms/frame` above is the persisted end-to-end per-frame refinement time. "
            "`selected_attempt_profiling.csv` separately reports SLSQP core, active-set "
            "discovery, and final-audit scheduling time; retries remain orchestration cost.",
            "",
            "The candidate changes exact-reference validation scheduling only. Objective, "
            "constraints, precision, iterations, sequential warm-start, recovery, and final "
            "independent all-frame validation remain unchanged.",
        ]
    )
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
