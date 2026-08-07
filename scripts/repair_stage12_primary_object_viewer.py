#!/usr/bin/env python3
"""Rebuild Stage 12 viewer-derived artifacts for an explicitly selected object.

The command leaves the canonical, warm-start, final retarget, and source data
untouched.  It writes a versioned repair lineage and only replaces the legacy
HTML path when ``--replace-html`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.contracts.canonical import load_canonical_hoi  # noqa: E402
from toporetarget.geometry.object_geometry import sample_object_track  # noqa: E402
from toporetarget.geometry.surface_sampling import load_surface_profile  # noqa: E402
from toporetarget.quality.html import render_clip_html, smoke_html  # noqa: E402
from toporetarget.quality.schema import ClipSpec  # noqa: E402
from toporetarget.retarget.artifacts import load_warm_start  # noqa: E402
from toporetarget.retarget.interaction_artifacts import (  # noqa: E402
    interaction_artifact_hash,
    save_interaction_evaluation,
    save_interaction_graph,
)
from toporetarget.retarget.interaction_evaluation import (  # noqa: E402
    evaluate_interaction_graph,
)
from toporetarget.retarget.interaction_graph import (  # noqa: E402
    build_source_interaction_graph,
)
from toporetarget.robots.registry import get_robot_registry  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--primary-object", required=True)
    parser.add_argument("--robot", default="wuji_hand2_beta1_rh")
    parser.add_argument(
        "--repair-name",
        help="Versioned repair directory name; defaults to primary_object_<id>_v1",
    )
    parser.add_argument(
        "--replace-html",
        action="store_true",
        help="Archive and replace html/source_warm_final_wuji.html after validation",
    )
    return parser.parse_args()


def _load_report(root: Path) -> dict[str, object]:
    path = root / "metrics" / "retarget_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("retarget report must be a JSON object")
    return payload


def _artifact_path(root: Path, report: dict[str, object], name: str) -> Path:
    paths = report.get("paths")
    if not isinstance(paths, dict) or name not in paths:
        raise ValueError(f"retarget report has no {name!r} artifact path")
    path = Path(str(paths[name])).expanduser()
    return path if path.is_absolute() else root / path


def main() -> int:
    args = parse_args()
    root = args.selection_root.expanduser().resolve()
    report = _load_report(root)
    canonical_path = _artifact_path(root, report, "canonical")
    warm_path = _artifact_path(root, report, "warm")
    final_path = _artifact_path(root, report, "final")
    for path in (canonical_path, warm_path, final_path):
        if not path.is_dir():
            raise FileNotFoundError(path)

    canonical = load_canonical_hoi(canonical_path)
    object_track = canonical.rigid_object(args.primary_object)
    context_object_ids = {
        item.object_id
        for item in canonical.rigid_objects
        if item.object_id != object_track.object_id
    }
    repair_name = args.repair_name or f"primary_object_{args.primary_object}_v1"
    repair_root = root / "repairs" / repair_name
    if repair_root.exists():
        raise FileExistsError(
            f"repair lineage already exists: {repair_root}; choose a new --repair-name"
        )
    graph_path = repair_root / "interaction_graph.zarr"
    evaluation_path = repair_root / "interaction_evaluation.zarr"
    samples_path = repair_root / "object_samples.npz"
    repaired_html = repair_root / "source_warm_final_wuji.html"

    surface_profile = load_surface_profile("paper_strict_area_uniform", repo_root=REPO_ROOT)
    samples = sample_object_track(object_track, surface_profile)
    samples.save(samples_path)
    graph = build_source_interaction_graph(
        canonical,
        "right_hand",
        object_track.object_id,
        samples,
        source_cache=canonical_path,
        object_sample_path=samples_path,
        frame_indices=np.arange(canonical.num_frames, dtype=np.int64),
    )
    if graph.metadata.get("object_id") != object_track.object_id:
        raise ValueError("rebuilt interaction graph object id does not match selection")
    save_interaction_graph(graph, graph_path)

    warm = load_warm_start(warm_path)
    robot = get_robot_registry(repo_root=REPO_ROOT).load(args.robot)
    evaluation = evaluate_interaction_graph(graph, warm, robot)
    save_interaction_evaluation(evaluation, evaluation_path)

    selection = report.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("retarget report has no selection mapping")
    frame_range = selection.get("frame_range", [0, canonical.num_frames])
    if not isinstance(frame_range, list) or len(frame_range) != 2:
        raise ValueError("retarget report has an invalid frame_range")
    retarget = report.get("retarget")
    if not isinstance(retarget, dict):
        raise ValueError("retarget report has no retarget mapping")
    profile = str(retarget.get("profile", "wuji_continuous_sequential_v1"))
    sequence = str(report.get("sequence", canonical.metadata.sequence_id))
    clip = ClipSpec(
        unit_id=f"stage12_repair_{root.name}",
        sequence=sequence,
        subject=str(report.get("dataset", "unknown")),
        object_name=object_track.object_id,
        start_frame=int(frame_range[0]),
        end_frame=int(frame_range[0]) + canonical.num_frames,
        hand=str(selection.get("hand", "right")),
        robot=args.robot,
        native_fps=float(canonical.metadata.native_fps or 30.0),
    )
    render_clip_html(
        clip=clip,
        canonical_path=canonical_path,
        source_path=canonical_path,
        profile_paths={
            "paper_warm": (warm_path, True, "warm Wuji"),
            profile: (final_path, False, "final Wuji"),
        },
        output=repaired_html,
        asset_root=None,
        recommended_profile=profile,
        graph_path=graph_path,
        evaluation_path=evaluation_path,
    )
    smoke = smoke_html(
        repaired_html,
        expected_frames=canonical.num_frames,
        profiles=2,
        expected_object_id=object_track.object_id,
        expected_context_object_ids=context_object_ids,
    )
    if smoke["status"] != "pass":
        raise ValueError(f"repaired HTML smoke failed: {smoke}")

    legacy_html = root / "html" / "source_warm_final_wuji.html"
    archived_html: Path | None = None
    if args.replace_html:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive_root = root / "html" / "archive"
        archive_root.mkdir(parents=True, exist_ok=True)
        if legacy_html.is_file():
            archived_html = archive_root / f"source_warm_final_wuji.stale_{timestamp}.html"
            shutil.copy2(legacy_html, archived_html)
        temporary = legacy_html.with_name(f".{legacy_html.name}.repair-tmp")
        shutil.copy2(repaired_html, temporary)
        os.replace(temporary, legacy_html)

    manifest = {
        "schema_version": "toporetarget.stage12.primary_object_viewer_repair.v1",
        "status": "STAGE12_PRIMARY_OBJECT_VIEWER_REPAIRED",
        "selection_root": str(root),
        "primary_object": object_track.object_id,
        "context_objects": sorted(context_object_ids),
        "canonical_untouched": True,
        "warm_untouched": True,
        "final_untouched": True,
        "source_untouched": True,
        "object_samples": str(samples_path.resolve()),
        "interaction_graph": str(graph_path.resolve()),
        "interaction_graph_sha256": interaction_artifact_hash(graph_path),
        "interaction_evaluation": str(evaluation_path.resolve()),
        "interaction_evaluation_sha256": interaction_artifact_hash(evaluation_path),
        "repaired_html": str(repaired_html.resolve()),
        "repaired_html_sha256": sha256_file(repaired_html),
        "legacy_html_replaced": bool(args.replace_html),
        "legacy_html": str(legacy_html.resolve()),
        "archived_legacy_html": str(archived_html.resolve()) if archived_html else None,
        "html_smoke": smoke,
    }
    manifest_path = repair_root / "repair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
