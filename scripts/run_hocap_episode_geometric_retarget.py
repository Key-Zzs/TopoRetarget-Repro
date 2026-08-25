#!/usr/bin/env python3
"""Run geometric retargeting for one frozen HOCap EpisodeV1 unit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.data.materialize_hocap_episode import load_episode_row  # noqa: E402
from toporetarget.retarget.final_refinement import load_final_trajectory  # noqa: E402
from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    BatchContractError,
    assert_frozen_manifest,
    atomic_write_json,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

SOLVER_PROFILE_ID = "wuji_continuous_sequential_v1"
EXECUTION_PROFILE_ID = "wuji_continuous_sequential_fast_exact_v2"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mano-model-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Frozen EpisodeV1 held-out manifest; required for downstream held-out execution.",
    )
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--benchmark-first-frames",
        type=int,
        help="Use only the first N episode frames for an explicitly labeled benchmark.",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Benchmark/debug only; production manual workflow keeps HTML enabled.",
    )
    parser.add_argument(
        "--execution-profile",
        choices=(EXECUTION_PROFILE_ID,),
        default=EXECUTION_PROFILE_ID,
    )
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_step(name: str, command: list[str], *, log_root: Path) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    receipt_path = log_root / f"{name}.receipt.json"
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS" and previous.get("command") == command:
            return {**previous, "resumed_from_pass_receipt": True}
    tick = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    environment.update(THREAD_ENVIRONMENT)
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = log_root / f"{name}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    receipt = {
        "stage": name,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "command": command,
        "wall_seconds": time.perf_counter() - tick,
        "returncode": result.returncode,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }
    atomic_write_json(receipt_path, receipt)
    if result.returncode != 0:
        raise BatchContractError(f"HOCAP_EPISODE_RETARGET_STAGE_FAILED:{name}:{log_path}")
    return receipt


def _robot_for_hand(hand: str) -> str:
    if hand == "right":
        return "wuji_hand2_beta1_rh"
    if hand == "left":
        return "wuji_hand2_beta1_lh"
    raise BatchContractError("HOCAP_EPISODE_RETARGET_SINGLE_HAND_REQUIRED")


def _timing_summary(
    stages: list[dict[str, Any]], final: Path, execution: RefinementExecutionProfile
) -> dict[str, Any]:
    by_name = {str(row["stage"]): row for row in stages}
    trajectory = load_final_trajectory(final)
    solve_times = np.asarray(trajectory.arrays["solve_time_s"], dtype=np.float64)
    iterations = np.asarray(
        trajectory.arrays.get("optimizer_iterations", trajectory.arrays["iterations"]),
        dtype=np.float64,
    )
    solver_seconds = float(np.sum(solve_times))
    solver_wall = float(by_name["continuous_refinement"]["wall_seconds"])
    return {
        "schema_version": "FastExactV2SeparatedTimingV1",
        "frames": int(len(solve_times)),
        "input_quality_scan_seconds": float(by_name["input_quality_precheck"]["wall_seconds"]),
        "raw_loading_seconds": float(by_name["raw_conversion"]["wall_seconds"]),
        "solver_seconds": solver_seconds,
        "solver_ms_per_frame": 1000.0 * solver_seconds / len(solve_times),
        "iterations_per_frame_mean": float(np.mean(iterations)),
        "iterations_per_frame_median": float(np.median(iterations)),
        "full_frame_validation_seconds": float(
            by_name["validate_continuous_refinement"]["wall_seconds"]
        ),
        "mesh_validation_seconds": float(
            by_name["validate_object_samples"]["wall_seconds"]
            + by_name["validate_interaction_graph"]["wall_seconds"]
        ),
        "html_generation_seconds": (
            None if "render_html" not in by_name else float(by_name["render_html"]["wall_seconds"])
        ),
        "serialization_seconds": max(0.0, solver_wall - solver_seconds),
        "serialization_timing_authority": (
            "continuous_refinement_wall_minus_sum_per_frame_solve_time; includes checkpoint/final "
            "serialization and command orchestration"
        ),
        "total_seconds": float(sum(float(row["wall_seconds"]) for row in stages)),
        "warm_start_enabled": True,
        "solver_profile": SOLVER_PROFILE_ID,
        "execution_profile": execution.profile_id,
        "execution_profile_sha256": execution.profile_hash,
        "math_equivalent": execution.math_equivalent,
        "paper_objective_unchanged": execution.paper_objective_unchanged,
        "paper_constraints_unchanged": execution.paper_constraints_unchanged,
    }


def main() -> int:
    args = _parser().parse_args()
    execution = RefinementExecutionProfile.load(args.execution_profile, REPO_ROOT)
    if not (
        execution.profile_id == EXECUTION_PROFILE_ID
        and execution.math_equivalent
        and execution.final_full_surface_audit
        and execution.device == "cpu"
        and execution.dtype == "float64"
    ):
        raise BatchContractError("HOCAP_EPISODE_RETARGET_EXECUTION_PROFILE_INVALID")
    index_path = args.episode_index.resolve()
    row = load_episode_row(index_path, args.episode_id)
    selection_manifest_sha256: str | None = None
    if args.selection_manifest is not None:
        selection_manifest = json.loads(
            args.selection_manifest.resolve().read_text(encoding="utf-8")
        )
        assert_frozen_manifest(selection_manifest)
        matches = [
            item
            for item in selection_manifest["clips"]
            if item.get("episode_id") == args.episode_id and item.get("clip_id") == args.episode_id
        ]
        if (
            len(matches) != 1
            or matches[0].get("primary_object_id") != row.get("target_object")
            or matches[0].get("selected_frame_range")
            != [row.get("start_frame"), row.get("end_frame")]
        ):
            raise BatchContractError("HOCAP_EPISODE_RETARGET_SELECTION_MANIFEST_MISMATCH")
        selection_manifest_sha256 = str(selection_manifest["manifest_sha256"])
    if row.get("physicalization_v1_eligible") is not True:
        raise BatchContractError("HOCAP_EPISODE_RETARGET_REQUIRES_ELIGIBLE_EPISODE")
    hand = str(row["active_hand"])
    robot = _robot_for_hand(hand)
    object_id = str(row["target_object"])
    primary_object_authority_sha256 = hashlib.sha256(
        json.dumps(
            {
                "authority": "HOCapSingleHandObjectEpisodeV1",
                "contract_sha256": row["contract_sha256"],
                "episode_id": row["episode_id"],
                "target_object": object_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    episode_frames = int(row["end_frame"]) - int(row["start_frame"])
    if args.benchmark_first_frames is not None and not (
        2 <= args.benchmark_first_frames <= episode_frames
    ):
        raise BatchContractError("HOCAP_EPISODE_RETARGET_BENCHMARK_RANGE_INVALID")
    expected_frames = args.benchmark_first_frames or episode_frames

    run_root = args.run_root.resolve() / args.episode_id
    report_root = args.report_root.resolve() / "episodes" / args.episode_id
    raw_root = run_root / "raw_contract"
    retarget = run_root / "retarget"
    reports = report_root / "retarget"
    logs = report_root / "logs"
    canonical = raw_root / "canonical_episode.zarr"
    canonical_receipt = raw_root / "canonical_episode.receipt.json"
    quality_receipt = reports / "retarget_input_quality.json"
    quality_csv = reports / "retarget_input_quality_per_frame.csv"
    repaired_input = raw_root / "retarget_input_quality_repaired.npz"
    warm = retarget / "warm_start.npz"
    samples = retarget / "object_samples.npz"
    graph = retarget / "interaction_graph.npz"
    evaluation = retarget / "interaction_evaluation.npz"
    final = retarget / "final_continuous.zarr"
    python = sys.executable
    common_asset = [] if args.asset_root is None else ["--asset-root", str(args.asset_root)]
    materialize = [
        python,
        "scripts/data/materialize_hocap_episode.py",
        "--episode-index",
        str(index_path),
        "--episode-id",
        args.episode_id,
        "--data-root",
        str(args.data_root.resolve()),
        "--mano-model-root",
        str(args.mano_model_root.resolve()),
        "--output",
        str(canonical),
        "--receipt",
        str(canonical_receipt),
        "--retarget-input-quality-receipt",
        str(quality_receipt),
    ]
    if args.benchmark_first_frames is not None:
        materialize.extend(["--benchmark-first-frames", str(args.benchmark_first_frames)])
    steps: list[tuple[str, list[str]]] = [
        (
            "input_quality_precheck",
            [
                python,
                "scripts/retarget/scan_hocap_retarget_input_quality.py",
                "--episode-index",
                str(index_path),
                "--episode-id",
                args.episode_id,
                "--data-root",
                str(args.data_root.resolve()),
                "--mano-model-root",
                str(args.mano_model_root.resolve()),
                "--report",
                str(quality_receipt),
                "--per-frame-csv",
                str(quality_csv),
                "--repaired-output",
                str(repaired_input),
            ],
        ),
        ("raw_conversion", materialize),
        (
            "warm_start",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "warm-start",
                "--canonical",
                str(canonical),
                "--hand",
                hand,
                "--robot",
                robot,
                "--output",
                str(warm),
                *common_asset,
            ],
        ),
        (
            "validate_warm_start",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "validate-warm-start",
                "--canonical",
                str(canonical),
                "--warm-start",
                str(warm),
                "--report",
                str(reports / "warm_start_validation.json"),
                "--csv",
                str(reports / "warm_start_validation.csv"),
                *common_asset,
            ],
        ),
        (
            "sample_object_surface",
            [
                python,
                "-m",
                "toporetarget",
                "geometry",
                "sample-object",
                "--canonical",
                str(canonical),
                "--object-id",
                object_id,
                "--output",
                str(samples),
                "--report",
                str(reports / "object_samples.json"),
            ],
        ),
        (
            "validate_object_samples",
            [
                python,
                "-m",
                "toporetarget",
                "geometry",
                "validate-samples",
                "--samples",
                str(samples),
                "--canonical",
                str(canonical),
                "--object-id",
                object_id,
                "--report",
                str(reports / "object_samples_validation.json"),
                "--csv",
                str(reports / "object_samples_validation.csv"),
            ],
        ),
        (
            "build_interaction_graph",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "build-interaction-graph",
                "--canonical",
                str(canonical),
                "--hand",
                hand,
                "--object-id",
                object_id,
                "--object-samples",
                str(samples),
                "--output",
                str(graph),
                "--report",
                str(reports / "interaction_graph.json"),
            ],
        ),
        (
            "validate_interaction_graph",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "validate-interaction-graph",
                "--canonical",
                str(canonical),
                "--object-samples",
                str(samples),
                "--graph",
                str(graph),
                "--report",
                str(reports / "interaction_graph_validation.json"),
                "--csv",
                str(reports / "interaction_graph_validation.csv"),
            ],
        ),
        (
            "evaluate_interaction",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "evaluate-interaction",
                "--graph",
                str(graph),
                "--warm-start",
                str(warm),
                "--robot",
                robot,
                "--output",
                str(evaluation),
                *common_asset,
            ],
        ),
        (
            "validate_interaction",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "validate-interaction",
                "--graph",
                str(graph),
                "--evaluation",
                str(evaluation),
                "--warm-start",
                str(warm),
                "--report",
                str(reports / "interaction_validation.json"),
                "--csv",
                str(reports / "interaction_validation.csv"),
            ],
        ),
        (
            "continuous_refinement",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "refine",
                "--canonical",
                str(canonical),
                "--warm-start",
                str(warm),
                "--graph",
                str(graph),
                "--robot",
                robot,
                "--solver-profile",
                SOLVER_PROFILE_ID,
                "--execution-profile",
                execution.profile_id,
                "--checkpoint-root",
                str(retarget / "continuous_checkpoints"),
                "--progress-json",
                str(reports / "continuous_refine_progress.json"),
                "--progress-log",
                str(reports / "continuous_refine_progress.jsonl"),
                "--output",
                str(final),
                *common_asset,
            ],
        ),
        (
            "validate_continuous_refinement",
            [
                python,
                "-m",
                "toporetarget",
                "retarget",
                "validate-refinement",
                "--canonical",
                str(canonical),
                "--warm-start",
                str(warm),
                "--graph",
                str(graph),
                "--final",
                str(final),
                "--robot",
                robot,
                "--report",
                str(reports / "continuous_final_validation.json"),
                "--csv",
                str(reports / "continuous_final_validation.csv"),
                *common_asset,
            ],
        ),
    ]
    receipts: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for name, command in steps:
            receipts.append(_run_step(name, command, log_root=logs))
    except BatchContractError as error:
        atomic_write_json(
            report_root / "geometric_retarget_receipt.json",
            {
                "schema_version": "HOCapEpisodeGeometricRetargetFailureV1",
                "status": "FAIL",
                "reason": str(error),
                "episode_id": args.episode_id,
                "completed_stages": receipts,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise

    artifacts = {
        "retarget_input_quality": {"path": str(quality_receipt)},
        "retarget_input_quality_per_frame": {"path": str(quality_csv)},
        "retarget_input_repair": {"path": str(repaired_input)},
        "canonical": {"path": str(canonical)},
        "canonical_receipt": {"path": str(canonical_receipt)},
        "warm_start": {"path": str(warm)},
        "graph": {"path": str(graph)},
        "evaluation": {"path": str(evaluation)},
        "final": {"path": str(final)},
        "checkpoint_manifest": {"path": str(retarget / "continuous_checkpoints" / "manifest.json")},
    }
    html_path: Path | None = None
    if not args.skip_html:
        html_manifest = {
            "schema_version": "HOCapEpisodeRetargetHtmlVisualizationManifestV1",
            "run_id": f"{args.report_root.name}_{args.episode_id}",
            "episode_id": args.episode_id,
            "source_sequence": row["raw_sequence"],
            "selected_frame_range": [
                row["start_frame"],
                int(row["start_frame"]) + expected_frames,
            ],
            "robot": robot,
            "primary_object_id": object_id,
            "primary_object_authority_sha256": primary_object_authority_sha256,
            "retarget_method": {
                "solver_profile_id": SOLVER_PROFILE_ID,
                "execution_profile_id": execution.profile_id,
                "execution_profile_sha256": execution.profile_hash,
                "math_equivalent": execution.math_equivalent,
            },
            "run_root": str(retarget),
            "artifacts": artifacts,
        }
        html_manifest_path = reports / "html_visualization_manifest.json"
        atomic_write_json(html_manifest_path, html_manifest)
        html_path = reports / "continuous_refinement_visualization.html"
        receipts.append(
            _run_step(
                "render_html",
                [
                    python,
                    "-m",
                    "toporetarget",
                    "workflow",
                    "visualize-mesh",
                    "--run",
                    str(html_manifest_path),
                    "--mode",
                    "combined",
                    "--max-object-points",
                    "50000",
                    "--output",
                    str(html_path),
                    *common_asset,
                ],
                log_root=logs,
            )
        )
    timing = _timing_summary(receipts, final, execution)
    if timing["frames"] != expected_frames:
        raise BatchContractError(
            f"HOCAP_EPISODE_RETARGET_FRAME_COUNT_DRIFT:{timing['frames']}:{expected_frames}"
        )
    receipt = {
        "schema_version": "HOCapEpisodeGeometricRetargetReceiptV1",
        "status": "PASS",
        "episode_id": args.episode_id,
        "clip_id": args.episode_id,
        "raw_sequence": row["raw_sequence"],
        "active_hand": hand,
        "target_object": object_id,
        "primary_object_id": object_id,
        "primary_object_authority_sha256": primary_object_authority_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "episode_index": {"path": str(index_path), "sha256": sha256_file(index_path)},
        "benchmark_first_frames": args.benchmark_first_frames,
        "robot": robot,
        "solver_profile_id": SOLVER_PROFILE_ID,
        "execution_profile_id": execution.profile_id,
        "execution_profile_sha256": execution.profile_hash,
        "math_equivalent": execution.math_equivalent,
        "thread_environment": THREAD_ENVIRONMENT,
        "artifacts": artifacts,
        "html": None if html_path is None else str(html_path),
        "stages": receipts,
        "timing": timing,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(report_root / "geometric_retarget_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
