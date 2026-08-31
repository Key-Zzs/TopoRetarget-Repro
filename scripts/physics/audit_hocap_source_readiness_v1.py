#!/usr/bin/env python3
"""Audit future HOCap physicalization readiness without launching PhysX.

The scan is deliberately metadata-only.  Missing source support does not
become a claim that support is absent, and a mesh does not become evidence
that runtime mass/COM/inertia have already been resolved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-results", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--hocap-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else ["episode_id"]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _taxonomy(*, semantic_pass: bool, mesh_available: bool, support_known: bool) -> str:
    if not semantic_pass:
        return "EXCLUDED_NON_SEMANTIC_PASS"
    if not mesh_available:
        return "QUARANTINE_DYNAMICS_UNRESOLVED"
    if not support_known:
        return "QUARANTINE_SUPPORT_UNDERDETERMINED"
    return "PROXY_PHYSICALIZATION_REQUIRED"


def main() -> int:
    args = _parser().parse_args()
    semantic_path = args.semantic_results.resolve()
    index_path = args.episode_index.resolve()
    hocap_root = args.hocap_root.resolve()
    index_value = _json(index_path)
    if not isinstance(index_value, list):
        raise ValueError("HOCAP_READINESS_EPISODE_INDEX_LIST_REQUIRED")
    index_by_id = {str(row["episode_id"]): row for row in index_value if isinstance(row, dict)}

    results: list[dict[str, str]] = []
    with semantic_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            results.append({str(key): str(value) for key, value in row.items()})
    semantic_ids = {row.get("episode_id", "") for row in results}
    if len(semantic_ids) != len(results):
        raise ValueError("HOCAP_READINESS_SEMANTIC_RESULT_DUPLICATE")

    output_rows: list[dict[str, object]] = []
    for result in results:
        episode_id = result.get("episode_id", "")
        source = index_by_id.get(episode_id)
        if source is None:
            raise ValueError(f"HOCAP_READINESS_EPISODE_MISSING:{episode_id}")
        semantic_pass = result.get("status") == "SEMANTIC_PREFLIGHT_PASS"
        target_object = str(source.get("target_object", ""))
        mesh_path = hocap_root / "data" / "models" / target_object / "textured_mesh.obj"
        support = source.get("source_support_metadata")
        if not isinstance(support, dict):
            support = {}
        source_explicit = _bool(support.get("source_explicit_support_present"))
        source_reconstructed = _bool(support.get("source_reconstructed_support_candidate_present"))
        support_known = source_explicit or source_reconstructed
        eligible = (
            semantic_pass
            and _bool(source.get("complete"))
            and _bool(source.get("physicalization_v1_eligible"))
            and source.get("episode_type") == "SINGLE_HAND_PICK_PLACE"
        )
        mesh_available = mesh_path.is_file()
        row: dict[str, object] = {
            "episode_id": episode_id,
            "raw_sequence": source.get("raw_sequence", ""),
            "subject": source.get("subject", ""),
            "active_hand": source.get("active_hand", ""),
            "object_id": target_object,
            "semantic_pass": semantic_pass,
            "physicalization_v1_eligible": eligible,
            "source_support_explicit": source_explicit,
            "source_support_reconstructed": source_reconstructed,
            "support_existence_status": (
                "SOURCE_SUPPORT_KNOWN" if support_known else "UNDERDETERMINED"
            ),
            "hand_support_signal": _bool(source.get("other_hand_same_target")),
            "other_object_support_signal": _bool(source.get("overlapping_other_hand_other_object")),
            "object_mesh_available": mesh_available,
            "object_mesh_sha256": _sha256(mesh_path) if mesh_available else "",
            "object_dynamics_explicit": False,
            "object_dynamics_derived": False,
            "geometry_derived_dynamics_possible": mesh_available,
            "object_dynamics_status": "UNRESOLVED",
            "proxy_physicalizable_candidate": bool(eligible and mesh_available),
            "likely_physical_scene_gpu_candidate": bool(
                eligible and mesh_available and support_known
            ),
            "primary_batch_taxonomy": _taxonomy(
                semantic_pass=semantic_pass,
                mesh_available=mesh_available,
                support_known=support_known,
            ),
            "reason": (
                "no_source_support_metadata_and_no_runtime_dynamic_asset_receipt"
                if semantic_pass and not support_known
                else "no_runtime_dynamic_asset_receipt"
                if semantic_pass
                else "semantic_preflight_not_pass"
            ),
        }
        output_rows.append(row)

    semantic_pass_rows = [row for row in output_rows if row["semantic_pass"] is True]
    summary = {
        "schema_version": "HOCapSourceReadinessAuditV1",
        "status": "PASS",
        "scope": "CPU_ONLY_STATIC_METADATA_MESH_SCAN",
        "full_corpus_gpu": False,
        "inputs": {
            "semantic_results": {"path": str(semantic_path), "sha256": _sha256(semantic_path)},
            "episode_index": {"path": str(index_path), "sha256": _sha256(index_path)},
            "hocap_root": str(hocap_root),
        },
        "counts": {
            "all_scanned_episode_rows": len(output_rows),
            "semantic_pass": len(semantic_pass_rows),
            "source_support_known": sum(
                bool(row["source_support_explicit"] or row["source_support_reconstructed"])
                for row in semantic_pass_rows
            ),
            "source_support_explicit": sum(
                bool(row["source_support_explicit"]) for row in semantic_pass_rows
            ),
            "source_support_reconstructed": sum(
                bool(row["source_support_reconstructed"]) for row in semantic_pass_rows
            ),
            "proxy_physicalizable": sum(
                bool(row["proxy_physicalizable_candidate"]) for row in semantic_pass_rows
            ),
            "support_underdetermined": sum(
                row["support_existence_status"] == "UNDERDETERMINED" for row in semantic_pass_rows
            ),
            "hand_supported_signal": sum(
                bool(row["hand_support_signal"]) for row in semantic_pass_rows
            ),
            "other_object_supported_signal": sum(
                bool(row["other_object_support_signal"]) for row in semantic_pass_rows
            ),
            "object_dynamics_explicit": sum(
                bool(row["object_dynamics_explicit"]) for row in semantic_pass_rows
            ),
            "object_dynamics_derived": sum(
                bool(row["object_dynamics_derived"]) for row in semantic_pass_rows
            ),
            "object_dynamics_unresolved": sum(
                row["object_dynamics_status"] == "UNRESOLVED" for row in semantic_pass_rows
            ),
            "likely_physical_scene_gpu_candidates": sum(
                bool(row["likely_physical_scene_gpu_candidate"]) for row in semantic_pass_rows
            ),
        },
        "interpretation": {
            "support_unknown_is_not_support_absent": True,
            "mesh_presence_is_not_runtime_dynamics_resolution": True,
            "runtime_physx_not_run": True,
            "taxonomy_requires_fresh_support_and_dynamics_authority": True,
        },
        "primary_taxonomy_counts": {
            taxonomy: sum(row["primary_batch_taxonomy"] == taxonomy for row in output_rows)
            for taxonomy in sorted({str(row["primary_batch_taxonomy"]) for row in output_rows})
        },
    }
    _write_csv(args.output_csv.resolve(), output_rows)
    _write_json(args.output_summary.resolve(), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
