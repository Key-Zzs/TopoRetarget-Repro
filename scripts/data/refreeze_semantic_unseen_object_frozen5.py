#!/usr/bin/env python3
"""Freeze the post-certification object-disjoint HOCap semantic held-out set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.data.freeze_h3_unseen_object_frozen5 import _manifest_episode  # noqa: E402
from toporetarget.geometry.mesh_audit import audit_mesh  # noqa: E402
from toporetarget.geometry.object_geometry import load_mesh_file  # noqa: E402

SEED = 20260830
COUNT = 5
SPLIT = "UNSEEN_OBJECT_INSTANCE_HELDOUT"

# These are explicit object identities exposed by the integrated development,
# hardening, audit, and physicalization lanes.  The historical H3D manifest is
# deliberately not read or used as a selection input.
DEVELOPMENT_OBJECTS: dict[str, str] = {
    **{f"G10_{index}": "170105_and_development" for index in range(1, 5)},
    **{f"G04_{index}": "170650_and_development" for index in range(1, 5)},
    "G16_3": "H3A_H3B_H3C_hardening",
    "G09_4": "H3A_H3C_hardening",
    "G22_3": "H3A_H3C_hardening",
    "G16_2": "H3A_H3B_H3C_hardening",
    "G06_4": "downstream_multiclip_outcome",
    "G21_1": "downstream_multiclip_and_wrong_target_audit",
    "G19_1": "downstream_multiclip_and_wrong_target_audit",
    "G02_1": "downstream_multiclip_outcome",
    "G09_1": "downstream_multiclip_outcome",
    "G18_1": "golden_or_manual_development",
    "G15_1": "golden_or_manual_development",
    "G05_1": "semantic_canary_2",
    "G19_2": "wrong_target_audit_correct_object",
    "G19_4": "wrong_target_audit_sequence_context",
    "G21_2": "wrong_target_audit_sequence_context",
    "G21_3": "semantic_canary_1_and_wrong_target_sequence_context",
    "G21_4": "wrong_target_audit_sequence_context",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--p4-root", type=Path, required=True)
    parser.add_argument("--p5-manifest", type=Path, required=True)
    parser.add_argument("--p6-decision", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"P7_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    _write_new(
        path,
        _csv_text(rows, fields),
    )


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _identity_map(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sources: dict[str, tuple[Path, str]] = {}
    for episode in episodes:
        object_id = str(episode["target_object"])
        mesh = episode.get("provenance", {}).get("object_mesh", {})
        path = Path(str(mesh.get("path", ""))).resolve()
        expected = str(mesh.get("sha256", ""))
        if not path.is_file() or len(expected) != 64:
            raise ValueError(f"P7_OBJECT_MESH_AUTHORITY_INVALID:{object_id}")
        previous = sources.setdefault(object_id, (path, expected))
        if previous != (path, expected):
            raise ValueError(f"P7_OBJECT_IDENTITY_DRIFT:{object_id}")
    identities: dict[str, dict[str, Any]] = {}
    for object_id, (path, expected) in sorted(sources.items()):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"P7_OBJECT_MESH_SHA_DRIFT:{object_id}")
        vertices, faces = load_mesh_file(path)
        audit = audit_mesh(vertices, faces, source_path=path, mesh_id=object_id)
        dimensions = [float(value) for value in vertices.max(axis=0) - vertices.min(axis=0)]
        identities[object_id] = {
            "object_id": object_id,
            "canonical_object_asset_path": str(path),
            "canonical_mesh_sha256": actual,
            "geometry_hash": audit.mesh_hash,
            "topology_hash": audit.topology_hash,
            "dimensions_m": dimensions,
            "known_aliases": [object_id],
            "alias_authority": "SELF_ONLY_NO_OFFICIAL_ALIAS_CROSSWALK",
            "object_family_category": None,
            "category_authority": "UNAVAILABLE_IN_OFFICIAL_RELEASE_METADATA",
            "object_prefix": object_id.split("_", 1)[0],
            "object_prefix_is_category_authority": False,
        }
    return identities


def _selection_key(row: dict[str, Any], identity: dict[str, Any]) -> str:
    return _stable_hash(
        {
            "seed": SEED,
            "episode_id": row["episode_id"],
            "raw_sequence": row["raw_sequence"],
            "object_id": identity["object_id"],
            "mesh_sha256": identity["canonical_mesh_sha256"],
        }
    )


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    objects: set[str] = set()
    sequences: set[str] = set()
    subjects: set[str] = set()
    prefixes: set[str] = set()
    for require_subject, require_prefix in ((True, True), (True, False), (False, False)):
        for row in rows:
            if len(selected) == COUNT:
                return selected
            if row["target_object"] in objects or row["raw_sequence"] in sequences:
                continue
            if require_subject and row["subject"] in subjects:
                continue
            if require_prefix and row["object_prefix"] in prefixes:
                continue
            selected.append(row)
            objects.add(str(row["target_object"]))
            sequences.add(str(row["raw_sequence"]))
            subjects.add(str(row["subject"]))
            prefixes.add(str(row["object_prefix"]))
    return selected


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"P7_OUTPUT_EXISTS:{output}")
    p6 = json.loads(args.p6_decision.resolve().read_text(encoding="utf-8"))
    if p6.get("status") != "PASS":
        raise ValueError("P7_P6_CERTIFICATION_REQUIRED")
    p5 = json.loads(args.p5_manifest.resolve().read_text(encoding="utf-8"))
    if p5.get("downstream_outcomes_used") is not False:
        raise ValueError("P7_P5_OUTCOME_CONTAMINATION")
    p5_manifest_hash = str(p5.get("manifest_sha256", ""))
    p4_root = args.p4_root.resolve()
    source_rows = list(csv.DictReader((p4_root / "all_episode_candidates.csv").open()))
    episodes = json.loads(args.episode_index.resolve().read_text(encoding="utf-8"))
    if not isinstance(episodes, list):
        raise ValueError("P7_EPISODE_INDEX_LIST_REQUIRED")
    episode_by_id = {str(row["episode_id"]): row for row in episodes}
    identities = _identity_map(episodes)

    excluded_ids = set(DEVELOPMENT_OBJECTS)
    excluded_meshes = {identities[item]["canonical_mesh_sha256"] for item in excluded_ids}
    excluded_geometry = {identities[item]["geometry_hash"] for item in excluded_ids}
    excluded_aliases = {
        alias for item in excluded_ids for alias in identities[item]["known_aliases"]
    }

    candidates: list[dict[str, Any]] = []
    audited: list[dict[str, Any]] = []
    for source in source_rows:
        if source.get("semantic_preflight_status") != "SEMANTIC_PREFLIGHT_PASS":
            continue
        if (
            source.get("active_hand") != "right"
            or source.get("episode_type") != "SINGLE_HAND_PICK_PLACE"
        ):
            continue
        if source.get("complete") != "True" or source.get("physicalization_v1_eligible") != "True":
            continue
        if (
            source.get("binding_status") != "PASS"
            or source.get("target_authority_status") != "TARGET_OBJECT_PASS"
        ):
            continue
        episode = episode_by_id.get(source["episode_id"])
        if episode is None:
            raise ValueError(f"P7_EPISODE_MISSING:{source['episode_id']}")
        object_id = str(source["target_object"])
        identity = identities[object_id]
        overlap: list[str] = []
        if object_id in excluded_ids:
            overlap.append("DEVELOPMENT_OBJECT_ID_OVERLAP")
        if identity["canonical_mesh_sha256"] in excluded_meshes:
            overlap.append("DEVELOPMENT_MESH_SHA256_OVERLAP")
        if identity["geometry_hash"] in excluded_geometry:
            overlap.append("DEVELOPMENT_GEOMETRY_HASH_OVERLAP")
        if set(identity["known_aliases"]) & excluded_aliases:
            overlap.append("DEVELOPMENT_KNOWN_ALIAS_OVERLAP")
        row = {
            **source,
            "object_prefix": identity["object_prefix"],
            "canonical_mesh_sha256": identity["canonical_mesh_sha256"],
            "geometry_hash": identity["geometry_hash"],
            "known_aliases": identity["known_aliases"],
            "development_overlap_reasons": overlap,
            "selection_key": _selection_key(source, identity),
            "metadata_exposure_only": True,
            "downstream_outcome_observed_for_selection": False,
            "episode_contract_sha256": episode.get("contract_sha256"),
        }
        audited.append(row)
        if not overlap:
            candidates.append(row)
    candidates.sort(key=lambda row: (row["selection_key"], row["episode_id"]))
    selected = _select(candidates)
    if len(selected) != COUNT:
        raise RuntimeError(f"P7_UNSEEN_OBJECT_SPLIT_UNAVAILABLE:{len(selected)}")
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank

    selected_ids = {row["target_object"] for row in selected}
    selected_meshes = {row["canonical_mesh_sha256"] for row in selected}
    selected_geometry = {row["geometry_hash"] for row in selected}
    selected_aliases = {alias for row in selected for alias in row["known_aliases"]}
    selected_sequences = {row["raw_sequence"] for row in selected}
    if not (
        len(selected_ids) == COUNT
        and len(selected_meshes) == COUNT
        and len(selected_geometry) == COUNT
        and len(selected_aliases) == COUNT
        and len(selected_sequences) == COUNT
        and not selected_ids & excluded_ids
        and not selected_meshes & excluded_meshes
        and not selected_geometry & excluded_geometry
        and not selected_aliases & excluded_aliases
    ):
        raise RuntimeError("P7_OBJECT_IDENTITY_OR_SEQUENCE_DISJOINTNESS_FAILED")

    clips: list[dict[str, Any]] = []
    for row in selected:
        clip = _manifest_episode({**episode_by_id[row["episode_id"]], **row})
        clip.update(
            {
                "object_identity": identities[row["target_object"]],
                "selection_rank": row["selection_rank"],
                "selection_key": row["selection_key"],
                "exclusion_audit": {
                    "object_id_disjoint": True,
                    "mesh_sha256_disjoint": True,
                    "geometry_hash_disjoint": True,
                    "known_alias_disjoint": True,
                    "sequence_disjoint": True,
                    "outcome_observed": False,
                },
            }
        )
        clips.append(clip)

    manifest: dict[str, Any] = {
        "schema_version": "DatasetSemanticAuthorityUnseenObjectFrozen5ManifestV1",
        "status": "FROZEN_NOT_EXECUTED",
        "HELD_OUT_SET_FROZEN": "YES",
        "dataset": "hocap",
        "dataset_role": SPLIT,
        "split_type": SPLIT,
        "selection_unit": "CanonicalHOIRecordV1",
        "selection_seed": SEED,
        "selection_basis": (
            "metadata_only_semantic_pass_right_hand_complete_single_hand_unique_sequence_and_object"
        ),
        "selection_algorithm": (
            "sha256_seed_then_unique_object_sequence_subject_prefix_diversity_v1"
        ),
        "category_overlap": "UNKNOWN_NO_AUTHORITY",
        "category_authority": "UNAVAILABLE_IN_OFFICIAL_RELEASE_METADATA",
        "category_disjoint_claim": False,
        "alias_authority": "SELF_ONLY_NO_OFFICIAL_ALIAS_CROSSWALK",
        "object_prefix_is_category_authority": False,
        "development_exclusion_ids": sorted(excluded_ids),
        "development_exclusion_sources": [
            "P0-P4 integrated reports",
            "P1 wrong-target audit",
            "P3 golden suite",
            "P5 approved semantic canaries",
            "historical H3A/H3B/H3C object lanes",
        ],
        "episode_index_sha256": _sha256(args.episode_index.resolve()),
        "p4_semantic_results_sha256": _sha256(p4_root / "semantic_preflight_results.csv"),
        "p5_manifest_sha256": p5_manifest_hash,
        "p6_certification_sha256": _sha256(args.p6_decision.resolve()),
        "held_out_count": COUNT,
        "clips": clips,
        "episodes": clips,
        "geometric_retarget_run": False,
        "source_controller_run": False,
        "support_physx_run": False,
        "frozen_evaluation_run": False,
        "physical_ppo_run": False,
        "downstream_outcomes_observed": False,
        "downstream_outcomes_used_for_selection": False,
        "shared_policy_zero_shot_claim": False,
    }
    manifest["manifest_sha256"] = _stable_hash(manifest)
    receipt = {
        "schema_version": "DatasetSemanticAuthorityUnseenObjectSelectionReceiptV1",
        "status": "FROZEN_NOT_EXECUTED",
        "selection_seed": SEED,
        "selection_unit": manifest["selection_unit"],
        "selection_algorithm": manifest["selection_algorithm"],
        "metadata_only": True,
        "outcome_fields_used": [],
        "downstream_outcomes_used": False,
        "candidate_count": len(candidates),
        "audited_pass_count": len(audited),
        "selected_episode_ids": [row["episode_id"] for row in selected],
        "selected_object_ids": sorted(selected_ids),
        "selected_sequences": sorted(selected_sequences),
        "selected_objects_unique": len(selected_ids) == COUNT,
        "selected_meshes_unique": len(selected_meshes) == COUNT,
        "selected_geometry_unique": len(selected_geometry) == COUNT,
        "selected_known_aliases_unique": len(selected_aliases) == COUNT,
        "selected_sequences_unique": len(selected_sequences) == COUNT,
        "object_id_overlap_with_development": len(selected_ids & excluded_ids),
        "mesh_sha256_overlap_with_development": len(selected_meshes & excluded_meshes),
        "geometry_hash_overlap_with_development": len(selected_geometry & excluded_geometry),
        "known_alias_overlap_with_development": len(selected_aliases & excluded_aliases),
        "category_overlap": "UNKNOWN_NO_AUTHORITY",
        "category_disjoint_claim": False,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    identity_rows = list(identities.values())
    exclusion_rows = [
        {"object_id": object_id, "exposure_class": reason, **identities[object_id]}
        for object_id, reason in sorted(DEVELOPMENT_OBJECTS.items())
    ]
    candidate_rows = [
        {
            **row,
            "known_aliases": json.dumps(row["known_aliases"], sort_keys=True),
            "development_overlap_reasons": json.dumps(row["development_overlap_reasons"]),
        }
        for row in audited
    ]
    identity_fields = list(identity_rows[0])
    _write_new(
        output / "unseen_object_frozen5_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        output / "unseen_object_frozen5_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False)
    )
    _write_new(output / "manifest_sha256.txt", manifest["manifest_sha256"] + "\n")
    _write_new(
        output / "selection_receipt.json", json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(
        output / "development_object_exclusions.csv",
        exclusion_rows,
        ["object_id", "exposure_class", *identity_fields[1:]],
    )
    _write_csv(output / "object_identity_table.csv", identity_rows, identity_fields)
    _write_csv(
        output / "candidate_pool.csv",
        candidate_rows,
        list(candidate_rows[0]) if candidate_rows else ["status"],
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
