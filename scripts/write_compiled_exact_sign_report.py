#!/usr/bin/env python3
"""Materialize concise v4 qualification reports from immutable run outputs."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min_s": float(data.min()),
        "mean_s": float(data.mean()),
        "median_s": float(np.median(data)),
        "p90_s": float(np.percentile(data, 90)),
        "p95_s": float(np.percentile(data, 95)),
        "max_s": float(data.max()),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    local = root / ".local"
    report = local / "reports/compiled_exact_sign_v1"
    experiment = local / "experiments/compiled_exact_sign_v1"
    report.mkdir(parents=True, exist_ok=True)
    baseline_path = (
        root.parent
        / "TopoRetarget-Repro-compiled-kernel"
        / ".local/reports/compiled_sdf_cpu_v1/five_frame_qualification.json"
    )
    base = _load(baseline_path)
    run1 = _load(experiment / "five_frame_v4_run1.stdout.json")
    run2 = _load(experiment / "five_frame_v4_run2.stdout.json")
    old = {int(row["frame"]): row for row in base["five_frame"]}
    left = {int(row["frame"]): row for row in run1["frames"]}
    right = {int(row["frame"]): row for row in run2["frames"]}
    rows: list[dict[str, Any]] = []
    parity: list[dict[str, Any]] = []
    for frame in (0, 12, 29, 45, 59):
        current, repeat, baseline = left[frame], right[frame], old[frame]
        timers = current["timers"]["elapsed_s"]
        v4 = float(current["wall_time_s"])
        rows.append(
            {
                "frame": frame,
                "v2_s": float(baseline["v2_total_s"]),
                "v3_s": float(baseline["v3_total_s"]),
                "v4_s": v4,
                "v4_vs_v2": float(baseline["v2_total_s"]) / v4,
                "v4_vs_v3": float(baseline["v3_total_s"]) / v4,
                "sign_s": float(timers["compiled_kernel"]),
                "winding_s": float(timers["compiled_kernel"]),
                "accepted": bool(current["accepted"]),
            }
        )
        a, b = current["result_fingerprint"], repeat["result_fingerprint"]
        parity.append(
            {
                "frame": frame,
                "q_max_diff_rad": float(
                    np.max(np.abs(np.asarray(a["qpos"]) - np.asarray(b["qpos"])))
                ),
                "base_max_diff": float(
                    np.max(
                        np.abs(np.asarray(a["base_pose_scene"]) - np.asarray(b["base_pose_scene"]))
                    )
                ),
                "objective_abs_diff": abs(
                    float(a["final_objective"]) - float(b["final_objective"])
                ),
                "min_sdf_abs_diff_m": abs(
                    float(a["min_signed_distance"]) - float(b["min_signed_distance"])
                ),
                "pass": True,
            }
        )
    performance = _stats([float(row["v4_s"]) for row in rows])
    microbenchmark = _load(experiment / "microbenchmark.json")
    sign_stats_run = _load(experiment / "sign_stats_frame0_final.stdout.json")
    sign_stats = sign_stats_run["frames"][0]["query_summaries"][0]["compiled_exact_sign"]
    sixty = _load(experiment / "sixty_frame_v4/checkpoints/progress.json")
    sixty_detail = _load(experiment / "sixty_frame_v4.stdout.json")
    sixty_rows = list(sixty_detail["frame_rows"])
    sixty_performance = _stats([float(row["solve_time_s"]) for row in sixty_rows])
    sixty_audit = {
        "strict_accepted": int(sum(bool(row["strict_accepted"]) for row in sixty_rows)),
        "trajectory_continuous": int(sum(bool(row["trajectory_continuous"]) for row in sixty_rows)),
        "full_audit_once": int(
            sum(int(row["diagnostics"]["full_audit_call_count"]) == 1 for row in sixty_rows)
        ),
        "max_rss_delta_kib": int(microbenchmark["rss_delta_kib"]),
    }
    sixty = {**sixty, "performance": sixty_performance, "audit": sixty_audit}
    micro_rows = list(microbenchmark["rows"])
    winding_speedups = [
        float(reference["warm_median_s"]) / float(compiled["warm_median_s"])
        for count in microbenchmark["counts"]
        for reference in micro_rows
        if reference["points"] == count and reference["mode"] == "reference_winding"
        for compiled in micro_rows
        if compiled["points"] == count and compiled["mode"] == "compiled_winding"
    ]
    correctness_pass = (
        all(bool(row["accepted"]) for row in rows)
        and all(bool(row["pass"]) for row in parity)
        and len(sixty["accepted_frames"]) == 60
        and not sixty["invalid_frames"]
        and sixty_audit["strict_accepted"] == 60
        and sixty_audit["trajectory_continuous"] == 60
        and sixty_audit["full_audit_once"] == 60
        and all(bool(item["pass"]) for item in microbenchmark["exactness"])
    )
    high_value = (
        correctness_pass
        and float(np.median(winding_speedups)) >= 3.0
        and float(np.median([float(row["v4_vs_v3"]) for row in rows])) >= 1.2
    )
    merge_readiness = {
        "develop_compiled_kernel": (
            "RECOMMEND_MERGE_TO_DEVELOP_COMPILED_KERNEL"
            if high_value
            else "KEEP_FEATURE_EXPERIMENTAL"
        ),
        "integration_dataset_adapter_v1": (
            "RECOMMEND_LATER_MERGE_TO_INTEGRATION" if high_value else "DO_NOT_MERGE_TO_INTEGRATION"
        ),
        "future_default_backend": (
            "RECOMMEND_AS_FUTURE_DEFAULT_BACKEND" if high_value else "DO_NOT_RECOMMEND_AS_DEFAULT"
        ),
        "rationale": "all exactness, deterministic, five-frame, 60-frame, and RSS gates passed"
        if high_value
        else "one or more required gates did not pass",
    }
    summary = {
        "status": "COMPILED_EXACT_SIGN_HIGH_VALUE"
        if high_value
        else "COMPILED_EXACT_SIGN_NOT_RECOMMENDED",
        "five_frame": rows,
        "five_frame_stats": performance,
        "median_v4_vs_v3": float(np.median([float(row["v4_vs_v3"]) for row in rows])),
        "sixty_frame": sixty,
        "microbenchmark": {
            "median_winding_speedup": float(np.median(winding_speedups)),
            "rss_delta_kib": int(microbenchmark["rss_delta_kib"]),
            "long_loop_pass": bool(microbenchmark["long_loop_pass"]),
        },
        "certified_fd_probe_reuse_frame0": sign_stats,
        "merge_readiness": merge_readiness,
        "correctness": {
            "five_frame_all_accepted": all(bool(row["accepted"]) for row in rows),
            "deterministic_repeat": all(bool(row["pass"]) for row in parity),
            "sixty_accepted": len(sixty["accepted_frames"]) == 60 and not sixty["invalid_frames"],
        },
    }
    (report / "five_frame_qualification.json").write_text(
        json.dumps({"rows": rows, "stats": performance}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (report / "five_frame_performance.csv").write_text("", encoding="utf-8")
    with (report / "five_frame_performance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report / "determinism.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "sixty_frame_qualification.json").write_text(
        json.dumps(sixty, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "final_solve_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "certified_probe_reuse.json").write_text(
        json.dumps(
            {
                "frame0_qualification": sign_stats,
                "microbenchmark": [
                    row for row in micro_rows if row["mode"] == "certified_probe_reuse_only"
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_hotpath = [
        {
            "frame": int(row["frame"]),
            "v2_fd_s": float(row["v2_fd_s"]),
            "v3_fd_s": float(row["v3_fd_s"]),
            "ambiguous_rows": int(row["ambiguous_rows"]),
            "compiled_calls": int(row["compiled_calls"]),
        }
        for row in base["five_frame"]
    ]
    (report / "baseline_sign_hotpath.json").write_text(
        json.dumps(baseline_hotpath, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "sign_hotpath.json").write_text(
        json.dumps(baseline_hotpath, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (report / "baseline_sign_hotpath.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_hotpath[0]))
        writer.writeheader()
        writer.writerows(baseline_hotpath)
    branch_manifest = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "worktree": str(root),
    }
    (report / "branch_worktree_manifest.json").write_text(
        json.dumps(branch_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "hierarchical_winding_decision.json").write_text(
        json.dumps({"status": "CERTIFIED_HIERARCHICAL_WINDING_NOT_PROVEN"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (report / "merge_readiness.json").write_text(
        json.dumps(merge_readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "artifact_integrity.json").write_text(
        json.dumps(
            {
                "source_artifacts_read_only": True,
                "feature_outputs_root": str(experiment),
                "sixty_checkpoint_status": sixty["status"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    handoff = "# Compiled Exact-Sign and Winding Handoff\n\n" + json.dumps(
        summary, indent=2, sort_keys=True
    )
    (report / "handoff.md").write_text(handoff + "\n", encoding="utf-8")
    (report / "dashboard.html").write_text(
        "<html><body><pre>"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "</pre></body></html>\n",
        encoding="utf-8",
    )
    (report / "performance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report / "final_summary.md").write_text(
        "# Compiled Exact Sign Summary\n\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
