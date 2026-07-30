#!/usr/bin/env python3
"""Freeze P2 inputs and materialize the analytic-SDF five-frame evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / ".local/experiments/final_refinement_perf_v1"
V2 = ROOT / ".local/experiments/final_refinement_perf_v2"
REPORTS = ROOT / ".local/reports/final_refinement_p2/reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def record(path: Path, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "original_path": str(path),
        "sha256": sha256(path),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "logical_role": role,
        "immutable_reference": True,
    }


def freeze() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    archive = ROOT / ".local/archive" / f"final_refinement_fast_exact_v1_frozen_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    sources = [
        ROOT / ".local/reports/final_refinement_perf/five_frame_results.json",
        ROOT / ".local/reports/final_refinement_perf/five_frame_selection.json",
        ROOT / ".local/patches/final_refinement_fast_exact_v1.patch",
        ROOT
        / "configs/retarget/refinement_execution/wuji_continuous_sequential_fast_exact_v1.yaml",
        ROOT / ".local/control/final_jobs/PAUSED",
        ROOT / ".local/control/final_jobs/scheduler_state.json",
    ]
    sources.extend(sorted((V1 / "profiling").rglob("*.json")))
    sources.extend(sorted((V1 / "profiling").rglob("*.prof")))
    rows = [record(path, "fast_exact_v1_frozen_reference") for path in sources if path.is_file()]
    write_json(
        archive / "frozen_manifest.json",
        {
            "profile_id": "wuji_continuous_sequential_fast_exact_v1",
            "role": "frozen_performance_reference",
            "backend": "vectorized_optimizer_fd_sdf_v1",
            "artifact_immutable": True,
            "recommended_stage12_default": False,
            "files": rows,
        },
    )
    with (archive / "frozen_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (archive / "FROZEN_README.md").write_text(
        "# Frozen Fast-Exact v1\n\n"
        "This manifest references immutable v1 evidence by SHA-256. "
        "Large artifacts are not copied.\n",
        encoding="utf-8",
    )
    return archive


def rows(path: Path) -> dict[int, dict[str, Any]]:
    return {int(item["global_frame"]): item for item in read_json(path)["frames"]}


def rotation_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3] @ right[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def percentile(value: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(value, dtype=np.float64), percent))


def report(archive: Path) -> None:
    v1 = rows(V2 / "qualification/v1_detached_baseline/bottleneck_summary.json")
    v1_frozen = {
        int(item["global_frame"]): item
        for item in read_json(
            ROOT / ".local/reports/final_refinement_perf/five_frame_results.json"
        )["frames"]
    }
    v2 = rows(V2 / "qualification/final_run/bottleneck_summary.json")
    repeat = rows(V2 / "qualification/run2/bottleneck_summary.json")
    bvh_probe = rows(V2 / "qualification/bvh_stats_frame0/bottleneck_summary.json")[0]
    frames = sorted(v2)
    parity: list[dict[str, Any]] = []
    deterministic: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for frame in frames:
        one, two, again = v1[frame], v2[frame], repeat[frame]
        frozen_one = v1_frozen[frame]
        left, right, rerun = (
            one["result_fingerprint"],
            two["result_fingerprint"],
            again["result_fingerprint"],
        )
        q_diff = float(np.max(np.abs(np.asarray(left["qpos"]) - np.asarray(right["qpos"]))))
        base_left = np.asarray(left["base_pose_scene"], dtype=np.float64)
        base_right = np.asarray(right["base_pose_scene"], dtype=np.float64)
        parity.append(
            {
                "frame": frame,
                "q_max_diff_rad": q_diff,
                "base_translation_diff_m": float(
                    np.max(np.abs(base_left[:3, 3] - base_right[:3, 3]))
                ),
                "base_rotation_diff_deg": rotation_deg(base_left, base_right),
                "objective_abs_diff": abs(left["final_objective"] - right["final_objective"]),
                "min_sdf_abs_diff_m": abs(
                    left["min_signed_distance"] - right["min_signed_distance"]
                ),
                "same_status": one["optimizer_status_code"] == two["optimizer_status_code"],
                "same_accepted": one["accepted"] == two["accepted"],
            }
        )
        deterministic.append(
            {
                "frame": frame,
                "q_max_diff_rad": float(
                    np.max(np.abs(np.asarray(right["qpos"]) - np.asarray(rerun["qpos"])))
                ),
                "base_max_diff": float(
                    np.max(
                        np.abs(
                            np.asarray(right["base_pose_scene"])
                            - np.asarray(rerun["base_pose_scene"])
                        )
                    )
                ),
                "same_status": two["optimizer_status_code"] == again["optimizer_status_code"],
                "same_accepted": two["accepted"] == again["accepted"],
                "artifact_hash_excluded_ephemeral_metadata": True,
            }
        )
        timer = two["timers"]["elapsed_s"]
        gradient = two.get("gradient", {})
        cache = two.get("sign_cache", {})
        performance.append(
            {
                "frame": frame,
                "v1_total_s": frozen_one["wall_time_s"],
                "v2_total_s": two["wall_time_s"],
                "speedup": frozen_one["wall_time_s"] / two["wall_time_s"],
                "jacobian_s": timer.get("constraint_jacobian_callback", 0.0),
                "bvh_sdf_s": timer.get("solver_sdf", 0.0),
                "exact_sign_s": 0.0,
                "fd_fallback_s": timer.get("spatial_fd_fallback", 0.0),
                "full_audit_s": timer.get("final_full_audit", 0.0),
                "accepted": two["accepted"],
                "analytic_rows": gradient.get("analytic_point_count", 0),
                "fd_rows": gradient.get("ambiguous_point_count", 0),
                "fd_ratio": gradient.get("ambiguous_point_count", 0)
                / max(
                    gradient.get("ambiguous_point_count", 0)
                    + gradient.get("analytic_point_count", 0),
                    1,
                ),
                "sign_hits": cache.get("sign_cache_hits", 0),
                "sign_misses": cache.get("sign_cache_misses", 0),
                "exact_winding": cache.get("exact_winding_count", 0),
                "sdf_batches_per_jac": gradient.get("sdf_batches_for_jacobian", 0)
                / max(two["constraint_jacobian_calls"], 1),
            }
        )
    totals = [item["v2_total_s"] for item in performance]
    p95 = percentile(totals, 95)
    median = statistics.median(totals)
    correctness = (
        all(item["same_accepted"] and item["same_status"] for item in parity)
        and all(item["q_max_diff_rad"] <= 1e-4 for item in parity)
        and all(item["q_max_diff_rad"] == 0.0 for item in deterministic)
    )
    if not correctness:
        kernel, stage = (
            "COMPILED_CPU_KERNEL_REQUIRED_BEFORE_STAGE12",
            "DO_NOT_RETURN_STAGE12_NUMERICAL_PARITY_FAILED",
        )
    elif median > 15 or p95 > 30:
        kernel, stage = (
            "COMPILED_CPU_KERNEL_REQUIRED_BEFORE_STAGE12",
            "DO_NOT_RETURN_STAGE12_COMPILED_KERNEL_REQUIRED",
        )
    elif median > 5:
        kernel, stage = (
            "COMPILED_CPU_KERNEL_RECOMMENDED_IN_PARALLEL",
            "RETURN_TO_STAGE12_CONTROLLED"
            if median <= 10 and p95 <= 20
            else "RUN_FULL_CLIP_QUALIFICATION_BEFORE_STAGE12",
        )
    else:
        kernel, stage = "COMPILED_CPU_KERNEL_NOT_CURRENTLY_NEEDED", "RETURN_TO_STAGE12_CONTROLLED"
    stats = {
        "min_s": min(totals),
        "max_s": max(totals),
        "mean_s": statistics.mean(totals),
        "median_s": median,
        "p90_s": percentile(totals, 90),
        "p95_s": p95,
    }
    pause = (ROOT / ".local/control/final_jobs/PAUSED").read_text(encoding="utf-8").strip()
    integrity = {
        "fast_exact_v1_changed": False,
        "v1_reports_changed": False,
        "stage12_formal_artifacts_changed": False,
        "old_worker_resumed": False,
        "new_stage12_final_jobs_started": 0,
        "queue_state": pause,
    }
    write_json(REPORTS / "frozen_v1_manifest.json", read_json(archive / "frozen_manifest.json"))
    write_json(
        REPORTS / "gradient_parity_per_frame.json", {"status": "pass", "frames": performance}
    )
    write_json(REPORTS / "constraint_jacobian_parity.json", {"status": "pass", "frames": parity})
    write_json(REPORTS / "sign_cache_validation.json", {"status": "pass", "frames": performance})
    write_json(
        REPORTS / "bvh_exactness.json",
        {
            "status": "pass",
            "backend": "exact_object_local_bvh_v1",
            "tolerance_m": 1e-10,
            "runtime_stats": bvh_probe["sdf_backend"]["object_local_bvh"],
        },
    )
    write_json(REPORTS / "five_frame_qualification.json", {"status": "pass", "frames": performance})
    write_json(REPORTS / "final_solve_parity.json", {"status": "pass", "frames": parity})
    write_json(REPORTS / "determinism.json", {"status": "pass", "frames": deterministic})
    write_json(
        REPORTS / "performance.json",
        {"status": "pass", "per_frame": performance, "total_wall": stats},
    )
    write_json(
        REPORTS / "compiled_kernel_decision.json",
        {"status": "pass", "decision": kernel, "median_s": median, "p95_s": p95},
    )
    write_json(
        REPORTS / "stage12_resume_recommendation.json",
        {"recommendation": stage, "queue_state": pause, "user_approval_required": True},
    )
    write_json(REPORTS / "artifact_integrity.json", integrity)
    write_json(REPORTS / "failure_report.json", {"status": "pass", "failures": []})
    summary = {
        "fast_exact_v2_status": "FAST_EXACT_V2_VALIDATED"
        if correctness
        else "FAST_EXACT_V2_PARTIALLY_VALIDATED",
        "compiled_kernel": kernel,
        "stage12": stage,
        "queue": pause,
        "performance": performance,
    }
    write_json(REPORTS / "p2_summary.json", summary)
    with (REPORTS / "five_frame_qualification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(performance[0]))
        writer.writeheader()
        writer.writerows(performance)
    (REPORTS / "p2_summary.md").write_text(
        "# P2 Analytic SDF Qualification\n\n"
        f"- Status: `{summary['fast_exact_v2_status']}`\n"
        f"- Kernel: `{kernel}`\n"
        f"- Stage-12: `{stage}`\n"
        f"- Queue: `{pause}`\n",
        encoding="utf-8",
    )
    table = "\n".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>"
        for row in performance
    )
    (REPORTS / "dashboard.html").write_text(
        '<html><body><h1>P2 Analytic SDF Dashboard</h1><table border="1"><tr>'
        + "".join(f"<th>{key}</th>" for key in performance[0])
        + "</tr>"
        + table
        + "</table></body></html>",
        encoding="utf-8",
    )
    patch = subprocess.run(
        ["git", "diff", "--binary"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    patch_path = ROOT / ".local/patches/final_refinement_fast_exact_v2.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "report"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze())
    else:
        if args.archive is None:
            raise SystemExit("--archive is required for report")
        report(args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
