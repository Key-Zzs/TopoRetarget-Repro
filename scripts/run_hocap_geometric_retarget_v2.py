#!/usr/bin/env python3
"""Run one manifest-bound HOCap raw-to-continuous-retarget lineage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.hocap_primary_object import (  # noqa: E402
    load_primary_object_authority,
    primary_object_from_authority,
)
from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    BatchContractError,
    assert_frozen_manifest,
    atomic_write_json,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-object-authority", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
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
    started = _utc()
    tick = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - tick
    log_path = log_root / f"{name}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    receipt = {
        "stage": name,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "command": command,
        "started_utc": started,
        "ended_utc": _utc(),
        "wall_seconds": elapsed,
        "returncode": result.returncode,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
    }
    atomic_write_json(receipt_path, receipt)
    if result.returncode != 0:
        raise BatchContractError(f"GEOMETRIC_RETARGET_STAGE_FAILED:{name}:{log_path}")
    return receipt


def main() -> int:
    args = _parser().parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert_frozen_manifest(manifest)
    authority = load_primary_object_authority(args.primary_object_authority)
    rows = [row for row in manifest["clips"] if row["clip_id"] == args.clip_id]
    if len(rows) != 1:
        raise BatchContractError(f"CLIP_ID_CARDINALITY:{args.clip_id}:{len(rows)}")
    clip = rows[0]
    primary = primary_object_from_authority(
        authority,
        sequence=clip["sequence"],
        available_object_ids=clip["object_ids"],
    )
    if clip.get("object_id") != primary or clip.get("primary_object_id") != primary:
        raise BatchContractError("SELECTION_PRIMARY_OBJECT_AUTHORITY_MISMATCH")
    if manifest.get("primary_object_authority_sha256") != authority["authority_sha256"]:
        raise BatchContractError("SELECTION_PRIMARY_OBJECT_AUTHORITY_HASH_MISMATCH")

    clip_run = args.run_root / args.clip_id
    clip_report = args.report_root / "clips" / args.clip_id
    raw_root = clip_run / "raw_contract"
    retarget = clip_run / "retarget"
    reports = clip_report / "retarget"
    logs = clip_report / "logs"
    canonical = raw_root / "canonical.zarr"
    warm = retarget / "warm_start.npz"
    samples = retarget / "object_samples.npz"
    graph = retarget / "interaction_graph.npz"
    evaluation = retarget / "interaction_evaluation.npz"
    final = retarget / "final_continuous.zarr"
    python = sys.executable
    common_asset = [] if args.asset_root is None else ["--asset-root", str(args.asset_root)]
    steps: list[tuple[str, list[str]]] = [
        (
            "raw_conversion",
            [
                python,
                "-m",
                "toporetarget",
                "data",
                "convert",
                "--dataset",
                "hocap",
                "--sequence",
                clip["sequence"],
                "--data-root",
                str(args.data_root.parent),
                "--primary-object-authority",
                str(args.primary_object_authority),
                "--start-frame",
                "0",
                "--end-frame",
                "41",
                "--output",
                str(canonical),
            ],
        ),
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
                "--robot",
                "wuji_hand2_beta1_rh",
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
                primary,
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
                primary,
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
                "--object-id",
                primary,
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
                "wuji_hand2_beta1_rh",
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
                "wuji_hand2_beta1_rh",
                "--solver-profile",
                "wuji_continuous_sequential_v1",
                "--execution-profile",
                "cached_checkpoint_cpu_float64_v1",
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
                "wuji_hand2_beta1_rh",
                "--report",
                str(reports / "continuous_final_validation.json"),
                "--csv",
                str(reports / "continuous_final_validation.csv"),
                *common_asset,
            ],
        ),
    ]
    receipts = []
    started = time.perf_counter()
    try:
        for name, command in steps:
            receipts.append(_run_step(name, command, log_root=logs))
    except BatchContractError as exc:
        atomic_write_json(
            clip_report / "geometric_retarget_receipt.json",
            {
                "status": "FAIL",
                "reason": str(exc),
                "clip_id": args.clip_id,
                "primary_object_id": primary,
                "completed_stages": receipts,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise

    html_manifest = {
        "schema_version": "IndependentHOCapRetargetHtmlVisualizationManifestV2",
        "run_id": f"independent_multiclip_hocap_pilot_v2_{args.clip_id}",
        "source_sequence": clip["sequence"],
        "selected_frame_range": [0, 41],
        "robot": "wuji_hand2_beta1_rh",
        "primary_object_id": primary,
        "primary_object_authority_sha256": authority["authority_sha256"],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "run_root": str(retarget),
        "artifacts": {
            "canonical": {"path": str(canonical)},
            "warm_start": {"path": str(warm)},
            "graph": {"path": str(graph)},
            "evaluation": {"path": str(evaluation)},
            "final": {"path": str(final)},
        },
    }
    html_manifest_path = reports / "html_visualization_manifest.v2.json"
    atomic_write_json(html_manifest_path, html_manifest)
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
                str(reports / "continuous_refinement_visualization.html"),
                *common_asset,
            ],
            log_root=logs,
        )
    )
    receipt = {
        "status": "PASS",
        "clip_id": args.clip_id,
        "sequence": clip["sequence"],
        "primary_object_id": primary,
        "primary_object_authority_sha256": authority["authority_sha256"],
        "selection_manifest_sha256": manifest["manifest_sha256"],
        "frame_range": [0, 41],
        "frame_count": 41,
        "artifacts": html_manifest["artifacts"],
        "html": str((reports / "continuous_refinement_visualization.html").resolve()),
        "stages": receipts,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(clip_report / "geometric_retarget_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
