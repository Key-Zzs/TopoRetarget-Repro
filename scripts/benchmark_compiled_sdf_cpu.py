#!/usr/bin/env python3
"""Summarize fixed-frame v2/v3 records without touching Stage-12 artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np


def _read_frame(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["frames"][0]


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min_s": float(np.min(array)),
        "max_s": float(np.max(array)),
        "mean_s": float(np.mean(array)),
        "median_s": float(np.median(array)),
        "p90_s": float(np.percentile(array, 90)),
        "p95_s": float(np.percentile(array, 95)),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    local = root / ".local"
    baseline = Path(
        "/home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro/.local/reports/final_refinement_p2/reports/performance.json"
    )
    v2 = {
        int(row["frame"]): row
        for row in json.loads(baseline.read_text(encoding="utf-8"))["per_frame"]
    }
    roots = {
        0: local / "experiments/compiled_sdf_cpu_v1/smoke_v3_frame0/bottleneck_summary.json",
        **{
            frame: local
            / "experiments/compiled_sdf_cpu_v1"
            / "qualification"
            / f"v3_frame{frame}"
            / "bottleneck_summary.json"
            for frame in (12, 29, 45, 59)
        },
    }
    repeats = {
        frame: local
        / "experiments/compiled_sdf_cpu_v1"
        / "qualification"
        / f"v3_repeat_frame{frame}"
        / "bottleneck_summary.json"
        for frame in (0, 12, 29, 45, 59)
    }
    rows: list[dict[str, object]] = []
    parity: list[dict[str, object]] = []
    for frame in (0, 12, 29, 45, 59):
        current = _read_frame(roots[frame])
        repeat = _read_frame(repeats[frame])
        timers = current["timers"]["elapsed_s"]
        old = v2[frame]
        rows.append(
            {
                "frame": frame,
                "v2_total_s": float(old["v2_total_s"]),
                "v3_total_s": float(current["wall_time_s"]),
                "overall_speedup": float(old["v2_total_s"]) / float(current["wall_time_s"]),
                "v2_fd_s": float(old["fd_fallback_s"]),
                "v3_fd_s": float(timers["spatial_fd_fallback"]),
                "fd_speedup": float(old["fd_fallback_s"]) / float(timers["spatial_fd_fallback"]),
                "ambiguous_rows": int(current["gradient"]["ambiguous_point_count"]),
                "compiled_calls": int(current["timers"]["counts"].get("compiled_kernel", 0)),
                "accepted": bool(current["accepted"]),
            }
        )
        left, right = current["result_fingerprint"], repeat["result_fingerprint"]
        parity.append(
            {
                "frame": frame,
                "q_max_diff_rad": float(
                    np.max(np.abs(np.asarray(left["qpos"]) - np.asarray(right["qpos"])))
                ),
                "base_max_diff": float(
                    np.max(
                        np.abs(
                            np.asarray(left["base_pose_scene"])
                            - np.asarray(right["base_pose_scene"])
                        )
                    ),
                ),
                "objective_abs_diff": abs(
                    float(left["final_objective"]) - float(right["final_objective"])
                ),
                "min_sdf_abs_diff_m": abs(
                    float(left["min_signed_distance"]) - float(right["min_signed_distance"])
                ),
                "same_accepted": bool(current["accepted"]) == bool(repeat["accepted"]),
            }
        )
    frame_stats = _stats([float(row["v3_total_s"]) for row in rows])
    fd_speedup = float(statistics.median([float(row["fd_speedup"]) for row in rows]))
    overall_speedup = float(statistics.median([float(row["overall_speedup"]) for row in rows]))
    status = (
        "COMPILED_KERNEL_LIMITED_VALUE"
        if fd_speedup > 1.0 and overall_speedup < 1.1
        else "COMPILED_KERNEL_NOT_RECOMMENDED"
    )
    report_root = local / "reports/compiled_sdf_cpu_v1"
    report_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "five_frame": rows,
        "determinism": parity,
        "v3_frame_stats": frame_stats,
        "median_fd_speedup": fd_speedup,
        "median_overall_speedup": overall_speedup,
    }
    (report_root / "five_frame_qualification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (report_root / "five_frame_qualification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
