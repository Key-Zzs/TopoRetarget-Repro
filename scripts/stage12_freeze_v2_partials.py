#!/usr/bin/env python3
"""Freeze, without modifying, the legacy Stage-12 v2 partial evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "benchmarks" / "stage12_selection.yaml"
DEFAULT_SOURCE = REPO / ".local" / "experiments" / "stage12_dataset_validation"
DEFAULT_ARCHIVE = REPO / ".local" / "archive"
V4_PROFILE = "wuji_continuous_sequential_fast_exact_v4_compiled_sign"


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in value).strip("_")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for checkpoint in sorted((root / "checkpoints").glob("*")):
        if not checkpoint.is_dir():
            continue
        progress = _read_json(checkpoint / "progress.json")
        if not progress:
            continue
        accepted = [int(value) for value in progress.get("accepted_frames", [])]
        item = {
            "checkpoint": str(checkpoint),
            "manifest_present": (checkpoint / "manifest.json").is_file(),
            "progress": progress,
            "accepted_count": len(accepted),
            "accepted_prefix_contiguous": accepted == list(range(len(accepted))),
            "frame_file_count": len(list((checkpoint / "frames").glob("frame_*.npz"))),
        }
        items.append(item)
    latest = max(items, key=lambda item: int(item["progress"].get("next_frame", -1)), default={})
    return items, latest


def _artifact_hashes(root: Path) -> list[dict[str, Any]]:
    hashes: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        hashes.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return hashes


def _upstream_reusable(root: Path) -> dict[str, bool]:
    return {
        "canonical": (root / "canonical" / "canonical_hoi_v2.zarr").is_dir(),
        "source": (root / "canonical" / "canonical_hoi_v2.zarr").is_dir(),
        "warm": (root / "warm" / "warm_start.zarr").is_dir(),
        "graph": (root / "exports" / "interaction_graph.zarr").is_dir(),
        "object_samples": (root / "exports" / "object_samples.npz").is_file(),
        "collision_samples": (root / "exports" / "wuji_collision_samples.npz").is_file(),
    }


def _freeze_one(
    source: Path, destination: Path, selection: dict[str, Any], *, hocap_g10: bool
) -> dict[str, Any]:
    root = source / str(selection["dataset"]) / _safe(str(selection["sequence"]))
    checkpoints, latest = _checkpoint_inventory(root)
    accepted = [int(value) for value in latest.get("progress", {}).get("accepted_frames", [])]
    report = _read_json(root / "metrics" / "retarget_report.json")
    metadata = {
        "schema_version": "toporetarget.stage12.v2_partial_freeze.v1",
        "status": "V2_PARTIAL_SUPERSEDED_BY_V4_FORMAL_RERUN",
        "backend": "fast_exact_v2",
        "formal_complete": False,
        "accepted_prefix_preserved": bool(latest) and accepted == list(range(len(accepted))),
        "resumable_for_diagnostics": bool(latest)
        and bool(latest.get("manifest_present"))
        and accepted == list(range(len(accepted))),
        "included_in_stage12_final_metrics": False,
        "superseding_profile": V4_PROFILE,
        "superseding_start_frame": 0,
        "selection": selection,
        "legacy_root": str(root),
        "upstream_reusable": _upstream_reusable(root),
    }
    if hocap_g10:
        metadata["historical_status"] = "HISTORICAL_RESULT_PROVENANCE_INCONSISTENT"
        metadata["included_in_final_stage12"] = False
        metadata["v4_rerun_required"] = True
    accepted_inventory = {
        "selection": selection,
        "accepted_frames": accepted,
        "accepted_count": len(accepted),
        "next_frame": latest.get("progress", {}).get("next_frame"),
        "checkpoint": latest.get("checkpoint"),
    }
    lineage = {
        "legacy_provenance": _read_json(root / "provenance.json"),
        "legacy_report_status": report.get("status"),
        "legacy_report_error": report.get("error"),
        "freeze_status": metadata["status"],
        "v4_rerun_start_frame": 0,
        "v4_profile": V4_PROFILE,
    }
    _write_json(destination / "frozen_partial_manifest.json", metadata)
    _write_json(destination / "checkpoint_inventory.json", {"checkpoints": checkpoints})
    _write_json(
        destination / "artifact_hashes.json",
        {"legacy_root": str(root), "files": _artifact_hashes(root)},
    )
    _write_json(destination / "accepted_frame_inventory.json", accepted_inventory)
    _write_json(destination / "lineage_report.json", lineage)
    (destination / "README.md").write_text(
        "# Frozen Stage-12 v2 partial\n\n"
        "This is an immutable evidence inventory. The legacy input was not modified, copied, "
        "or used to resume v4 final refinement. v4 restarts final refinement at frame 0.\n",
        encoding="utf-8",
    )
    return {**metadata, "archive_path": str(destination), **accepted_inventory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=REPO / ".local" / "reports" / "stage12_completion_v4",
    )
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    source = args.source_root.expanduser().resolve()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    selections = list(config.get("selections", []))
    indices = (1, 3, 4)
    destination = (
        args.archive_root.expanduser().resolve() / f"stage12_v2_partial_frozen_{args.timestamp}"
    )
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing freeze directory: {destination}")
    frozen = []
    for index in indices:
        selection = selections[index]
        frozen.append(
            _freeze_one(
                source,
                destination / str(selection["dataset"]) / _safe(str(selection["sequence"])),
                selection,
                hocap_g10=index == 4,
            )
        )
    summary = {
        "schema_version": "toporetarget.stage12.v2_partial_freeze_summary.v1",
        "archive_root": str(destination),
        "v4_profile": V4_PROFILE,
        "frozen_runs": frozen,
    }
    _write_json(destination / "v2_partial_freeze.json", summary)
    _write_json(args.reports_root.expanduser().resolve() / "v2_partial_freeze.json", summary)
    print(
        json.dumps(
            {"status": "pass", "archive_root": str(destination), "frozen_count": len(frozen)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
