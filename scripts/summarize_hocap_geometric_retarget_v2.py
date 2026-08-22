#!/usr/bin/env python3
"""Fail-closed aggregate validation for the five HOCap retarget HTML lineages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    BatchContractError,
    assert_frozen_manifest,
    atomic_write_json,
    atomic_write_text,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

PRODUCTION_SOLVER_PROFILE_ID = "wuji_continuous_sequential_v1"
PRODUCTION_EXECUTION_PROFILE_ID = "wuji_continuous_sequential_fast_exact_v2"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-object-authority", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BatchContractError(f"REQUIRED_ARTIFACT_MISSING:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BatchContractError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _validate_clip(
    *,
    clip: dict[str, Any],
    report_root: Path,
    manifest_hash: str,
    authority_hash: str,
) -> dict[str, Any]:
    clip_id = str(clip["clip_id"])
    expected_object = str(clip["primary_object_id"])
    selected_range = clip.get("selected_frame_range")
    if not isinstance(selected_range, list) or len(selected_range) != 2:
        raise BatchContractError(f"SELECTED_FRAME_RANGE_MISSING:{clip_id}")
    expected_frames = int(selected_range[1]) - int(selected_range[0])
    root = report_root / "clips" / clip_id
    receipt_path = root / "geometric_retarget_receipt.json"
    validation_path = root / "retarget/continuous_final_validation.json"
    html_manifest_path = root / "retarget/html_visualization_manifest.v2.json"
    render_log_path = root / "logs/render_html.log"
    html_path = root / "retarget/continuous_refinement_visualization.html"
    receipt = _load_json(receipt_path)
    validation = _load_json(validation_path)
    html_manifest = _load_json(html_manifest_path)
    render = _load_json(render_log_path)
    frames = validation.get("frames")
    checks = {
        "geometric_receipt_pass": receipt.get("status") == "PASS",
        "continuous_validation_pass": validation.get("pass") is True,
        "frame_count_matches_manifest": validation.get("frame_count") == expected_frames,
        "all_frames_accepted": isinstance(frames, list)
        and len(frames) == expected_frames
        and all(row.get("accepted") is True for row in frames),
        "source_integrity_pass": validation.get("source_integrity_pass") is True,
        "object_mesh_hash_match": validation.get("object_mesh_hash_match") is True,
        "html_exists_nonempty": html_path.is_file() and html_path.stat().st_size > 0,
        "render_pass": render.get("status") == "pass",
        "object_triangle_mesh_present": int(render.get("source_faces", 0)) > 0,
        "receipt_primary_matches": receipt.get("primary_object_id") == expected_object,
        "html_primary_matches": html_manifest.get("primary_object_id") == expected_object,
        "render_primary_matches": render.get("primary_object_id") == expected_object,
        "receipt_authority_matches": receipt.get("primary_object_authority_sha256")
        == authority_hash,
        "html_authority_matches": html_manifest.get("primary_object_authority_sha256")
        == authority_hash,
        "render_authority_matches": render.get("primary_object_authority_sha256") == authority_hash,
        "selection_manifest_matches": receipt.get("selection_manifest_sha256") == manifest_hash,
        "solver_profile_matches_production": receipt.get("solver_profile_id")
        == PRODUCTION_SOLVER_PROFILE_ID,
        "execution_profile_matches_production": receipt.get("execution_profile_id")
        == PRODUCTION_EXECUTION_PROFILE_ID,
        "html_execution_profile_matches": html_manifest.get("retarget_method", {}).get(
            "execution_profile_id"
        )
        == PRODUCTION_EXECUTION_PROFILE_ID,
        "execution_profile_hash_matches_html": receipt.get("execution_profile_sha256")
        == html_manifest.get("retarget_method", {}).get("execution_profile_sha256"),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    return {
        "clip_id": clip_id,
        "sequence": clip["sequence"],
        "primary_object_id": expected_object,
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "frame_count": validation.get("frame_count"),
        "source_vertices": render.get("source_vertices"),
        "source_faces": render.get("source_faces"),
        "html": str(html_path.resolve()),
        "html_bytes": html_path.stat().st_size if html_path.is_file() else 0,
        "html_sha256": sha256_file(html_path) if html_path.is_file() else None,
        "geometric_wall_seconds": receipt.get("wall_seconds"),
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
    }


def main() -> int:
    args = _parser().parse_args()
    manifest = _load_json(args.manifest)
    assert_frozen_manifest(manifest)
    authority = _load_json(args.primary_object_authority)
    manifest_hash = str(manifest["manifest_sha256"])
    authority_hash = str(authority["authority_sha256"])
    if manifest.get("primary_object_authority_sha256") != authority_hash:
        raise BatchContractError("SELECTION_PRIMARY_OBJECT_AUTHORITY_HASH_MISMATCH")
    rows = [
        _validate_clip(
            clip=clip,
            report_root=args.report_root,
            manifest_hash=manifest_hash,
            authority_hash=authority_hash,
        )
        for clip in manifest["clips"]
    ]
    failures = [row["clip_id"] for row in rows if row["status"] != "PASS"]
    result = {
        "schema_version": "IndependentHOCapGeometricRetargetAggregateV2",
        "status": "PASS" if not failures else "FAIL",
        "held_out_count": len(rows),
        "passed_count": len(rows) - len(failures),
        "failed_clips": failures,
        "selection_manifest": str(args.manifest.resolve()),
        "selection_manifest_sha256": manifest_hash,
        "primary_object_authority": str(args.primary_object_authority.resolve()),
        "primary_object_authority_sha256": authority_hash,
        "clips": rows,
    }
    aggregate = args.report_root / "aggregate"
    atomic_write_json(aggregate / "geometric_retarget_html_validation.json", result)
    commands = [
        "# Five held-out HOCap geometric-retarget HTML visualizations",
        "",
        (
            "Each HTML is self-contained and includes the authority-bound primary object "
            "triangle mesh."
        ),
        "",
    ]
    for row in rows:
        commands.extend(
            [
                f"## {row['clip_id']} -> {row['primary_object_id']}",
                "",
                "```bash",
                f"xdg-open {row['html']}",
                "```",
                "",
            ]
        )
    atomic_write_text(args.report_root / "replay/visualization_commands.md", "\n".join(commands))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
