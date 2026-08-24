#!/usr/bin/env python3
"""Benchmark frozen fast_exact_v2 on historical and EpisodeV1 HOCap inputs."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_hocap_episode_geometric_retarget import (  # noqa: E402
    EXECUTION_PROFILE_ID,
    SOLVER_PROFILE_ID,
    THREAD_ENVIRONMENT,
    _run_step,
    _timing_summary,
)
from toporetarget.contracts.canonical import (  # noqa: E402
    load_canonical_hoi,
    save_canonical_hoi,
)
from toporetarget.geometry.surface_artifacts import load_surface_artifact  # noqa: E402
from toporetarget.retarget.artifacts import load_warm_start  # noqa: E402
from toporetarget.retarget.final_refinement import load_final_trajectory  # noqa: E402
from toporetarget.retarget.interaction_artifacts import (  # noqa: E402
    load_interaction_evaluation,
    load_interaction_graph,
)
from toporetarget.retarget.refinement_performance import (  # noqa: E402
    RefinementExecutionProfile,
)
from toporetarget.rl.independent_physical_refinement import atomic_write_json  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmarks/hocap_fast_exact_v2_benchmark_v1.yaml"),
    )
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mano-model-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    return parser


def _hash_tree(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _hash_tree(path) if path.is_dir() else sha256_file(path),
    }


def _numeric_array_manifest(value: Any, *, prefix: str = "root") -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        manifest[prefix] = {
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            manifest.update(
                _numeric_array_manifest(
                    getattr(value, field.name),
                    prefix=f"{prefix}.{field.name}",
                )
            )
    elif isinstance(value, dict):
        for key in sorted(value):
            manifest.update(_numeric_array_manifest(value[key], prefix=f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            manifest.update(_numeric_array_manifest(item, prefix=f"{prefix}.{index}"))
    return manifest


def _historical_authority_canonical(
    original: Path,
    destination: Path,
    *,
    primary_object_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Create an additive metadata-only primary-object repair for HTML authority."""

    source = load_canonical_hoi(original)
    source_arrays = _numeric_array_manifest(source)
    authority_sha256 = hashlib.sha256(
        json.dumps(
            {
                "authority": "STAGE12_SELECTION_PRIMARY_OBJECT_METADATA_REPAIR_V1",
                "original_tree_sha256": _hash_tree(original),
                "primary_object_id": primary_object_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if not destination.exists():
        repaired = load_canonical_hoi(original)
        repaired.metadata = dataclasses.replace(
            repaired.metadata,
            metadata={
                **repaired.metadata.metadata,
                "primary_object_id": primary_object_id,
                "primary_object_authority_sha256": authority_sha256,
            },
            provenance=dataclasses.replace(
                repaired.metadata.provenance,
                conversion_options={
                    **repaired.metadata.provenance.conversion_options,
                    "primary_object_id": primary_object_id,
                    "primary_object_authority_sha256": authority_sha256,
                    "metadata_only_additive_repair": True,
                },
            ),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_canonical_hoi(repaired, destination)
    reloaded = load_canonical_hoi(destination)
    repaired_arrays = _numeric_array_manifest(reloaded)
    if source_arrays != repaired_arrays:
        raise RuntimeError("HOCAP_HISTORICAL_PRIMARY_AUTHORITY_NUMERIC_DRIFT")
    if (
        reloaded.metadata.metadata.get("primary_object_id") != primary_object_id
        or reloaded.metadata.provenance.conversion_options.get("primary_object_id")
        != primary_object_id
    ):
        raise RuntimeError("HOCAP_HISTORICAL_PRIMARY_AUTHORITY_METADATA_INVALID")
    receipt = {
        "schema_version": "HOCapHistoricalPrimaryObjectAuthorityRepairV1",
        "status": "PASS",
        "authority_sha256": authority_sha256,
        "primary_object_id": primary_object_id,
        "original": _artifact_identity(original),
        "repaired": _artifact_identity(destination),
        "numeric_array_manifest_identical": True,
        "numeric_array_count": len(source_arrays),
        "scope": "metadata_only_additive_repair; immutable source not mutated",
    }
    return destination, receipt


def _raw_load_probe(
    canonical: Path,
    warm: Path,
    graph: Path,
    evaluation: Path,
    samples: Path,
) -> tuple[float, dict[str, Any]]:
    tick = time.perf_counter()
    sequence = load_canonical_hoi(canonical)
    warm_value = load_warm_start(warm)
    graph_value = load_interaction_graph(graph)
    evaluation_value = load_interaction_evaluation(evaluation)
    surface_value = load_surface_artifact(samples)
    elapsed = time.perf_counter() - tick
    return elapsed, {
        "frames": int(sequence.num_frames),
        "warm_frames": int(len(warm_value.arrays["qpos"])),
        "graph_frames": int(graph_value.frame_count),
        "evaluation_frames": int(evaluation_value.frame_count),
        "surface_samples": int(surface_value.count),
    }


def _historical_repeat(
    case: dict[str, Any],
    *,
    repeat: int,
    run_root: Path,
    report_root: Path,
    execution: RefinementExecutionProfile,
    asset_root: Path | None,
) -> dict[str, Any]:
    prepared = (REPO_ROOT / str(case["prepared_root"])).resolve()
    original_canonical = prepared / "canonical/canonical_hoi_v2.zarr"
    html_canonical, authority_receipt = _historical_authority_canonical(
        original_canonical,
        run_root / str(case["case_id"]) / "frozen_input_authority/canonical_hoi_v2.zarr",
        primary_object_id=str(case["object_id"]),
    )
    # Every source-hash-sensitive validation and solver step remains bound to the
    # immutable historical canonical.  The additive metadata repair is only an
    # HTML authority view; using it as solver input would correctly invalidate
    # the frozen graph's source-cache hash even though all numeric arrays match.
    canonical = original_canonical
    warm = prepared / "warm/warm_start.zarr"
    graph = prepared / "exports/interaction_graph.zarr"
    evaluation = prepared / "exports/interaction_evaluation.zarr"
    samples = prepared / "exports/object_samples.npz"
    repeat_run = run_root / str(case["case_id"]) / f"repeat_{repeat:02d}"
    repeat_report = report_root / "runs" / str(case["case_id"]) / f"repeat_{repeat:02d}"
    retarget = repeat_run / "retarget"
    reports = repeat_report / "retarget"
    logs = repeat_report / "logs"
    final = retarget / "final_continuous.zarr"
    python = sys.executable
    common_asset = [] if asset_root is None else ["--asset-root", str(asset_root.resolve())]

    raw_seconds, raw_probe = _raw_load_probe(canonical, warm, graph, evaluation, samples)
    raw_receipt = {
        "stage": "raw_loading",
        "status": "PASS",
        "command": ["in_process_load_frozen_prepared_artifacts"],
        "wall_seconds": raw_seconds,
        "returncode": 0,
        "probe": raw_probe,
    }
    steps = [
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
                str(case["object_id"]),
                "--report",
                str(reports / "object_samples_validation.json"),
                "--csv",
                str(reports / "object_samples_validation.csv"),
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
                str(case["robot"]),
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
                str(case["robot"]),
                "--report",
                str(reports / "continuous_final_validation.json"),
                "--csv",
                str(reports / "continuous_final_validation.csv"),
                *common_asset,
            ],
        ),
    ]
    receipts = [_run_step(name, command, log_root=logs) for name, command in steps]
    manifest = {
        "schema_version": "HOCapFastExactV2BenchmarkHtmlManifestV1",
        "run_id": f"{case['case_id']}_repeat_{repeat:02d}",
        "source_sequence": case["sequence"],
        "selected_frame_range": [0, int(case["frames"])],
        "robot": case["robot"],
        "primary_object_id": case["object_id"],
        "primary_object_authority_sha256": authority_receipt["authority_sha256"],
        "retarget_method": {
            "solver_profile_id": SOLVER_PROFILE_ID,
            "execution_profile_id": execution.profile_id,
            "execution_profile_sha256": execution.profile_hash,
            "math_equivalent": execution.math_equivalent,
        },
        "run_root": str(retarget),
        "artifacts": {
            "canonical": {"path": str(html_canonical)},
            "warm_start": {"path": str(warm)},
            "graph": {"path": str(graph)},
            "evaluation": {"path": str(evaluation)},
            "final": {"path": str(final)},
        },
    }
    manifest_path = reports / "html_visualization_manifest.json"
    atomic_write_json(manifest_path, manifest)
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
                str(manifest_path),
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
    timing_stages = [
        {"stage": "raw_conversion", "wall_seconds": raw_seconds},
        *receipts,
    ]
    timing = _timing_summary(timing_stages, final, execution)
    timing["raw_loading_authority"] = "in-process deserialization of five frozen inputs"
    trajectory = load_final_trajectory(final)
    if timing["frames"] != int(case["frames"]):
        raise RuntimeError("HOCAP_BENCHMARK_HISTORICAL_FRAME_COUNT_DRIFT")
    receipt = {
        "schema_version": "HOCapFastExactV2HistoricalRepeatV1",
        "status": "PASS",
        "case_id": case["case_id"],
        "repeat": repeat,
        "input_artifacts": {
            name: _artifact_identity(path)
            for name, path in {
                "canonical": canonical,
                "warm": warm,
                "graph": graph,
                "evaluation": evaluation,
                "samples": samples,
            }.items()
        },
        "primary_object_authority_repair": authority_receipt,
        "final": _artifact_identity(final),
        "html": _artifact_identity(html_path),
        "optimizer_iterations": np.asarray(
            trajectory.arrays.get("optimizer_iterations", trajectory.arrays["iterations"])
        ).tolist(),
        "stages": [raw_receipt, *receipts],
        "timing": timing,
        "thread_environment": THREAD_ENVIRONMENT,
    }
    atomic_write_json(repeat_report / "benchmark_repeat_receipt.json", receipt)
    return receipt


def _episode_repeat(
    case: dict[str, Any],
    *,
    repeat: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_root = args.run_root.resolve() / str(case["case_id"]) / f"repeat_{repeat:02d}"
    case_report = (
        args.report_root.resolve() / "runs" / str(case["case_id"]) / f"repeat_{repeat:02d}"
    )
    command = [
        sys.executable,
        "scripts/run_hocap_episode_geometric_retarget.py",
        "--episode-index",
        str(args.episode_index.resolve()),
        "--episode-id",
        str(case["episode_id"]),
        "--data-root",
        str(args.data_root.resolve()),
        "--mano-model-root",
        str(args.mano_model_root.resolve()),
        "--run-root",
        str(case_root),
        "--report-root",
        str(case_report),
    ]
    if args.asset_root is not None:
        command.extend(["--asset-root", str(args.asset_root.resolve())])
    if "benchmark_first_frames" in case:
        command.extend(["--benchmark-first-frames", str(case["benchmark_first_frames"])])
    outer = _run_step(
        "episode_pipeline",
        command,
        log_root=case_report / "outer_logs",
    )
    inner_path = (
        case_report / "episodes" / str(case["episode_id"]) / "geometric_retarget_receipt.json"
    )
    inner = json.loads(inner_path.read_text(encoding="utf-8"))
    if inner.get("status") != "PASS" or inner["timing"]["frames"] != int(case["frames"]):
        raise RuntimeError("HOCAP_BENCHMARK_EPISODE_REPEAT_INVALID")
    receipt = {
        "schema_version": "HOCapFastExactV2EpisodeRepeatV1",
        "status": "PASS",
        "case_id": case["case_id"],
        "repeat": repeat,
        "outer": outer,
        "episode_receipt": str(inner_path),
        "timing": inner["timing"],
        "thread_environment": inner["thread_environment"],
    }
    atomic_write_json(case_report / "benchmark_repeat_receipt.json", receipt)
    return receipt


def _system_receipt() -> dict[str, Any]:
    commands = {}
    for name, command in {
        "lscpu": ["lscpu"],
        "uptime": ["uptime"],
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    }.items():
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        commands[name] = {"returncode": result.returncode, "output": result.stdout.strip()}
    return {
        "schema_version": "HOCapFastExactV2BenchmarkHostV1",
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "thread_environment": THREAD_ENVIRONMENT,
        "commands": commands,
    }


def _write_outputs(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    report_root: Path,
) -> None:
    columns = [
        "case_id",
        "repeat",
        "frames",
        "raw_loading_seconds",
        "solver_seconds",
        "solver_ms_per_frame",
        "iterations_per_frame_mean",
        "iterations_per_frame_median",
        "full_frame_validation_seconds",
        "mesh_validation_seconds",
        "html_generation_seconds",
        "serialization_seconds",
        "total_seconds",
        "warm_start_enabled",
        "solver_profile",
        "execution_profile",
        "execution_profile_sha256",
        "math_equivalent",
        "paper_objective_unchanged",
        "paper_constraints_unchanged",
    ]
    csv_path = report_root / "fast_exact_v2_benchmark.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    medians: dict[str, dict[str, float]] = {}
    for case in config["cases"]:
        case_rows = [row for row in rows if row["case_id"] == case["case_id"]]
        medians[str(case["case_id"])] = {
            key: statistics.median(float(row[key]) for row in case_rows)
            for key in (
                "solver_ms_per_frame",
                "iterations_per_frame_mean",
                "raw_loading_seconds",
                "full_frame_validation_seconds",
                "mesh_validation_seconds",
                "html_generation_seconds",
                "serialization_seconds",
                "total_seconds",
            )
        }
    regression = config["historical_regression_baseline"]
    summary = {
        "schema_version": "HOCapFastExactV2BenchmarkSummaryV1",
        "status": "PASS",
        "repeats": config["repeats"],
        "medians": medians,
        "regression_conclusion": "INCONCLUSIVE",
        "regression_reason": regression["reason"],
        "comparison_note": (
            "Cross-case timing diagnoses input/episode effects; it is not a same-input "
            "historical regression baseline."
        ),
    }
    atomic_write_json(report_root / "fast_exact_v2_benchmark_summary.json", summary)
    analysis = [
        "# fast_exact_v2 HOCap benchmark analysis",
        "",
        "## Result",
        "",
        "`REGRESSION=INCONCLUSIVE`. " + str(regression["reason"]),
        "",
        "All rows use one process at a time, the same conda environment, exact v2 math, "
        "float64 CPU execution, warm start, and six BLAS/OpenMP thread limits fixed to 1.",
        "",
        "## Median results",
        "",
        "| case | frames | core solver ms/frame | iterations/frame | raw load s | "
        "full validation s | mesh validation s | HTML s | serialization residual s | "
        "total s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_case = {str(case["case_id"]): case for case in config["cases"]}
    for case_id, values in medians.items():
        analysis.append(
            f"| {case_id} | {by_case[case_id]['frames']} | "
            f"{values['solver_ms_per_frame']:.3f} | {values['iterations_per_frame_mean']:.3f} | "
            f"{values['raw_loading_seconds']:.3f} | "
            f"{values['full_frame_validation_seconds']:.3f} | "
            f"{values['mesh_validation_seconds']:.3f} | "
            f"{values['html_generation_seconds']:.3f} | "
            f"{values['serialization_seconds']:.3f} | {values['total_seconds']:.3f} |"
        )
    analysis.extend(
        [
            "",
            "## Regression investigation controls",
            "",
            "- Warm start: enabled for every row.",
            "- Iterations: recorded per run and summarized above.",
            "- Threading: OMP/MKL/OpenBLAS/NumExpr/BLIS/vecLib all fixed to one thread.",
            "- Cache: every repeat has a distinct output/checkpoint tree; frozen inputs "
            "remain read-only.",
            "- Validation and HTML: timed separately and excluded from core solver ms/frame.",
            "- Fallback: execution profile is exact CPU float64 v2 with final full-surface "
            "audit; no approximate fallback is enabled.",
            "- CPU load/frequency and host identity: captured in `benchmark_host.json`.",
            "- Historical authority limitation: the 170650 accepted completion is v4, so "
            "it is not used as a v2 baseline.",
            "",
            "`serialization_seconds` is a residual (refinement command wall time minus "
            "summed per-frame solver time); it also contains process load/orchestration and "
            "is not pure serializer CPU time.",
        ]
    )
    (report_root / "fast_exact_v2_analysis.md").write_text(
        "\n".join(analysis) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["schema_version"] != "HOCapFastExactV2BenchmarkV1":
        raise ValueError("HOCAP_FAST_EXACT_V2_BENCHMARK_CONFIG_INVALID")
    if config["repeats"] < 3:
        raise ValueError("HOCAP_FAST_EXACT_V2_BENCHMARK_REPEATS_TOO_SMALL")
    if config["thread_environment"] != THREAD_ENVIRONMENT:
        raise ValueError("HOCAP_FAST_EXACT_V2_BENCHMARK_THREAD_CONTRACT_DRIFT")
    execution = RefinementExecutionProfile.load(EXECUTION_PROFILE_ID, REPO_ROOT)
    if not (
        execution.profile_id == EXECUTION_PROFILE_ID
        and execution.math_equivalent
        and execution.final_full_surface_audit
        and execution.device == "cpu"
        and execution.dtype == "float64"
    ):
        raise ValueError("HOCAP_FAST_EXACT_V2_BENCHMARK_EXECUTION_PROFILE_INVALID")
    os.environ.update(THREAD_ENVIRONMENT)
    report_root = args.report_root.resolve()
    run_root = args.run_root.resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_root / "benchmark_host.json", _system_receipt())
    atomic_write_json(
        report_root / "benchmark_config_receipt.json",
        {
            "schema_version": "HOCapFastExactV2BenchmarkConfigReceiptV1",
            "status": "PASS",
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "config": config,
        },
    )
    receipts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for case in config["cases"]:
        for repeat in range(1, int(config["repeats"]) + 1):
            if "prepared_root" in case:
                receipt = _historical_repeat(
                    case,
                    repeat=repeat,
                    run_root=run_root,
                    report_root=report_root,
                    execution=execution,
                    asset_root=args.asset_root,
                )
            else:
                receipt = _episode_repeat(case, repeat=repeat, args=args)
            receipts.append(receipt)
            rows.append({"case_id": case["case_id"], "repeat": repeat, **receipt["timing"]})
            atomic_write_json(
                report_root / "benchmark_progress.json",
                {
                    "schema_version": "HOCapFastExactV2BenchmarkProgressV1",
                    "status": "RUNNING",
                    "completed": len(receipts),
                    "total": len(config["cases"]) * int(config["repeats"]),
                    "rows": rows,
                },
            )
    _write_outputs(rows, config, report_root)
    atomic_write_json(
        report_root / "benchmark_receipt.json",
        {
            "schema_version": "HOCapFastExactV2BenchmarkReceiptV1",
            "status": "PASS",
            "completed": len(receipts),
            "rows": rows,
            "csv": _artifact_identity(report_root / "fast_exact_v2_benchmark.csv"),
            "analysis": _artifact_identity(report_root / "fast_exact_v2_analysis.md"),
        },
    )
    print(json.dumps({"status": "PASS", "completed": len(receipts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
