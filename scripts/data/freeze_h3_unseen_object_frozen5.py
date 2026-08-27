#!/usr/bin/env python3
"""Freeze the H3 object/mesh-disjoint HOCap EpisodeV1 Frozen5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.data.freeze_hocap_episode_held_out import _manifest_episode  # noqa: E402
from toporetarget.geometry.mesh_audit import audit_mesh  # noqa: E402
from toporetarget.geometry.object_geometry import load_mesh_file  # noqa: E402

SELECTION_SEED = 20260826
SPLIT_LABEL = "UNSEEN_OBJECT_INSTANCE_HELDOUT"
CATEGORY_AUDIT_SOURCES = (
    "https://irvlutd.github.io/HOCap/",
    "https://github.com/IRVLUTD/HO-Cap",
    "https://openreview.net/pdf/425863500ffe9488949557e52e96eb7a959b1f22.pdf",
    "https://arxiv.org/abs/2406.06843",
)


@dataclass(frozen=True)
class ObjectIdentity:
    object_id: str
    asset_path: str
    mesh_sha256: str
    geometry_hash: str
    topology_hash: str
    dimensions_m: tuple[float, float, float]
    aliases: tuple[str, ...]
    alias_authority: str
    category: str | None
    category_authority: str

    @property
    def prefix(self) -> str:
        return self.object_id.split("_", 1)[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "canonical_object_asset_path": self.asset_path,
            "canonical_mesh_sha256": self.mesh_sha256,
            "geometry_hash": self.geometry_hash,
            "topology_hash": self.topology_hash,
            "dimensions_m": list(self.dimensions_m),
            "aliases": list(self.aliases),
            "alias_authority": self.alias_authority,
            "object_family_category": self.category,
            "category_authority": self.category_authority,
            "object_prefix": self.prefix,
            "object_prefix_is_category_authority": False,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--development-exclusions", type=Path, required=True)
    parser.add_argument("--h3-protocol", type=Path, required=True)
    parser.add_argument("--h3-protocol-hash", type=Path, required=True)
    parser.add_argument("--old-p6-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
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


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3D_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"H3D_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (list, dict, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    os.replace(temporary, path)


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"H3D_MAPPING_REQUIRED:{path}")
    return dict(value)


def _identity_rows(episodes: list[dict[str, Any]]) -> dict[str, ObjectIdentity]:
    sources: dict[str, tuple[Path, str]] = {}
    for episode in episodes:
        object_id = str(episode["target_object"])
        mesh = episode.get("provenance", {}).get("object_mesh", {})
        path = Path(str(mesh.get("path", ""))).resolve()
        expected = str(mesh.get("sha256", ""))
        if not path.is_file() or len(expected) != 64:
            raise ValueError(f"H3D_OBJECT_MESH_AUTHORITY_INVALID:{object_id}")
        previous = sources.setdefault(object_id, (path, expected))
        if previous != (path, expected):
            raise ValueError(f"H3D_OBJECT_IDENTITY_DRIFT:{object_id}")
    result: dict[str, ObjectIdentity] = {}
    for object_id, (path, expected) in sorted(sources.items()):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"H3D_OBJECT_MESH_SHA_DRIFT:{object_id}")
        vertices, faces = load_mesh_file(path)
        report = audit_mesh(vertices, faces, source_path=path, mesh_id=object_id)
        lower = np.min(vertices, axis=0)
        upper = np.max(vertices, axis=0)
        dimensions = tuple(float(value) for value in upper - lower)
        result[object_id] = ObjectIdentity(
            object_id=object_id,
            asset_path=str(path),
            mesh_sha256=actual,
            geometry_hash=report.mesh_hash,
            topology_hash=report.topology_hash,
            dimensions_m=dimensions,
            aliases=(object_id,),
            alias_authority="SELF_ONLY_NO_OFFICIAL_ALIAS_CROSSWALK",
            category=None,
            category_authority="UNAVAILABLE_IN_OFFICIAL_RELEASE_METADATA",
        )
    return result


def _eligible(episode: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = {
        "RIGHT_HAND_REQUIRED": episode.get("active_hand") == "right",
        "EPISODE_V1_ELIGIBLE_REQUIRED": bool(episode.get("physicalization_v1_eligible")),
        "COMPLETE_REQUIRED": bool(episode.get("complete")),
        "SINGLE_HAND_PICK_PLACE_REQUIRED": episode.get("episode_type") == "SINGLE_HAND_PICK_PLACE",
        "SAME_OBJECT_BIMANUAL_FORBIDDEN": not bool(episode.get("other_hand_same_target")),
        "HANDOVER_FORBIDDEN": not bool(episode.get("handover", False)),
        "OTHER_OBJECT_OVERLAP_FORBIDDEN": not bool(
            episode.get("overlapping_other_hand_other_object")
        ),
        "RETURN_LIFECYCLE_REQUIRED": episode.get("return_semantics")
        == "RETURN_TO_NON_INTERACTING_IDLE",
    }
    lifecycle = (
        "approach_frame",
        "contact_frame",
        "pickup_frame",
        "transport_frame",
        "place_frame",
        "release_frame",
        "retreat_frame",
    )
    checks["LIFECYCLE_FRAMES_REQUIRED"] = all(episode.get(key) is not None for key in lifecycle)
    for reason, passed in checks.items():
        if not passed:
            failures.append(reason)
    return not failures, failures


def _selection_key(episode: dict[str, Any], identity: ObjectIdentity, *, seed: int) -> str:
    return _stable_hash(
        {
            "seed": seed,
            "episode_id": episode["episode_id"],
            "raw_sequence": episode["raw_sequence"],
            "object_id": identity.object_id,
            "mesh_sha256": identity.mesh_sha256,
        }
    )


def _select(candidates: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    objects: set[str] = set()
    sequences: set[str] = set()
    subjects: set[str] = set()
    prefixes: set[str] = set()
    for require_subject, require_prefix in ((True, True), (True, False), (False, False)):
        for row in candidates:
            if len(selected) == count:
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
    if args.count != 5 or args.seed != SELECTION_SEED:
        raise ValueError("H3D_FROZEN_SELECTION_PARAMETERS_REQUIRED")
    episode_path = args.episode_index.resolve()
    episodes = json.loads(episode_path.read_text(encoding="utf-8"))
    if not isinstance(episodes, list):
        raise ValueError("H3D_EPISODE_INDEX_LIST_REQUIRED")
    exclusions_path = args.development_exclusions.resolve()
    exclusions = _load_mapping(exclusions_path)
    if exclusions.get("schema_version") != "H3DevelopmentObjectExclusionsV1":
        raise ValueError("H3D_DEVELOPMENT_EXCLUSIONS_INVALID")
    entries = exclusions.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("H3D_DEVELOPMENT_EXCLUSIONS_EMPTY")
    for entry in entries:
        evidence = REPO_ROOT / str(entry.get("evidence_path", ""))
        if not evidence.exists():
            raise FileNotFoundError(f"H3D_EXCLUSION_EVIDENCE_MISSING:{evidence}")

    protocol_path = args.h3_protocol.resolve()
    protocol = _load_mapping(protocol_path)
    if protocol.get("schema_version") != "H3PhysicalizationProtocolV1":
        raise ValueError("H3D_PROTOCOL_INVALID")
    protocol_hash = args.h3_protocol_hash.resolve().read_text(encoding="utf-8").strip()
    if protocol_hash != _stable_hash(protocol):
        raise ValueError("H3D_PROTOCOL_HASH_DRIFT")
    execution_head = str(protocol.get("freeze", {}).get("H3_EXECUTION_HEAD", ""))
    if len(execution_head) != 40:
        raise ValueError("H3D_EXECUTION_HEAD_INVALID")

    identities = _identity_rows(episodes)
    excluded_ids = {str(entry["object_id"]) for entry in entries}
    missing_ids = excluded_ids - set(identities)
    if missing_ids:
        raise ValueError(f"H3D_EXCLUSION_OBJECT_UNKNOWN:{sorted(missing_ids)}")
    excluded_meshes = {identities[object_id].mesh_sha256 for object_id in excluded_ids}
    excluded_geometry = {identities[object_id].geometry_hash for object_id in excluded_ids}
    excluded_aliases = {
        alias for object_id in excluded_ids for alias in identities[object_id].aliases
    }

    audited: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for episode in episodes:
        object_id = str(episode["target_object"])
        identity = identities[object_id]
        episode_ok, failures = _eligible(episode)
        overlap_reasons: list[str] = []
        if object_id in excluded_ids:
            overlap_reasons.append("DEVELOPMENT_OBJECT_ID_OVERLAP")
        if identity.mesh_sha256 in excluded_meshes:
            overlap_reasons.append("DEVELOPMENT_MESH_SHA256_OVERLAP")
        if identity.geometry_hash in excluded_geometry:
            overlap_reasons.append("DEVELOPMENT_GEOMETRY_HASH_OVERLAP")
        if set(identity.aliases) & excluded_aliases:
            overlap_reasons.append("DEVELOPMENT_KNOWN_ALIAS_OVERLAP")
        row = {
            **episode,
            **identity.as_dict(),
            "static_episode_eligible": episode_ok,
            "static_episode_exclusion_reasons": failures,
            "development_overlap_reasons": overlap_reasons,
            "metadata_exposure_only": True,
            "downstream_outcome_observed_for_selection": False,
            "method_tuning_from_candidate": False,
            "selection_key": _selection_key(episode, identity, seed=args.seed),
        }
        audited.append(row)
        if episode_ok and not overlap_reasons:
            candidates.append(row)
    candidates.sort(key=lambda row: (str(row["selection_key"]), str(row["episode_id"])))
    selected = _select(candidates, count=args.count)
    if len(selected) != args.count:
        raise RuntimeError("H3D_UNSEEN_OBJECT_SPLIT_UNAVAILABLE")
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank

    selected_ids = {str(row["target_object"]) for row in selected}
    selected_meshes = {str(row["canonical_mesh_sha256"]) for row in selected}
    selected_geometry = {str(row["geometry_hash"]) for row in selected}
    selected_aliases = {alias for row in selected for alias in row["aliases"]}
    selected_sequences = {str(row["raw_sequence"]) for row in selected}
    if (
        len(selected_ids) != args.count
        or len(selected_meshes) != args.count
        or len(selected_geometry) != args.count
        or len(selected_aliases) != args.count
        or len(selected_sequences) != args.count
        or selected_ids & excluded_ids
        or selected_meshes & excluded_meshes
        or selected_geometry & excluded_geometry
        or selected_aliases & excluded_aliases
    ):
        raise RuntimeError("H3D_OBJECT_MESH_ALIAS_OR_SEQUENCE_DISJOINTNESS_FAILED")

    clips: list[dict[str, Any]] = []
    for row in selected:
        clip = _manifest_episode(row)
        clip.update(
            {
                "object_identity": identities[str(row["target_object"])].as_dict(),
                "selection_rank": row["selection_rank"],
                "selection_key": row["selection_key"],
                "exclusion_audit": {
                    "outcome_observed": False,
                    "metadata_exposure_only": True,
                    "object_id_disjoint": True,
                    "mesh_sha256_disjoint": True,
                    "geometry_hash_disjoint": True,
                    "known_alias_disjoint": True,
                },
            }
        )
        clips.append(clip)
    manifest: dict[str, Any] = {
        "schema_version": "H3UnseenObjectFrozen5ManifestV1",
        "status": "FROZEN_NOT_EXECUTED",
        "HELD_OUT_SET_FROZEN": "YES",
        "dataset": "hocap",
        "dataset_role": "UNSEEN_OBJECT_INSTANCE_HELDOUT",
        "held_out": True,
        "split_type": SPLIT_LABEL,
        "claim_boundary": "METHOD_LEVEL_NOT_SHARED_POLICY_ZERO_SHOT",
        "selection_unit": "HOCapSingleHandObjectEpisodeV1",
        "physicalization_protocol": "H3PhysicalizationProtocolV1",
        "h3_protocol_hash": protocol_hash,
        "H3_EXECUTION_HEAD": execution_head,
        "held_out_count": args.count,
        "selection_seed": args.seed,
        "selection_basis": (
            "static_metadata_episode_v1_object_mesh_disjoint_sequence_subject_diversity"
        ),
        "selection_algorithm": (
            "sha256_seed_then_unique_object_sequence_subject_prefix_diversity_v1"
        ),
        "category_authority": "UNAVAILABLE_IN_OFFICIAL_RELEASE_METADATA",
        "object_prefix_is_category_authority": False,
        "alias_authority": "SELF_ONLY_NO_OFFICIAL_ALIAS_CROSSWALK",
        "development_exclusions_sha256": _sha256(exclusions_path),
        "episode_index_sha256": _sha256(episode_path),
        "clips": clips,
        "episodes": clips,
        "geometric_retarget_run": False,
        "source_controller_run": False,
        "support_physx_run": False,
        "frozen_evaluation_run": False,
        "physical_ppo_run": False,
        "downstream_outcomes_observed": False,
        "shared_policy_zero_shot_claim": False,
        "independent_ppo_per_episode": True,
    }
    manifest["manifest_sha256"] = _stable_hash(manifest)

    old_p6_path = args.old_p6_manifest.resolve()
    old_p6 = _load_mapping(old_p6_path)
    old_p6_embedded = str(old_p6.get("manifest_sha256", ""))
    old_p6_core = dict(old_p6)
    old_p6_core.pop("manifest_sha256", None)
    old_p6_supersession = {
        "schema_version": "H3OldP6SupersessionReceiptV1",
        "status": "SUPERSEDED_FOR_UNSEEN_OBJECT_SPLIT",
        "historical_manifest_retained": True,
        "historical_manifest_path": str(old_p6_path),
        "historical_manifest_file_sha256": _sha256(old_p6_path),
        "historical_manifest_embedded_sha256": old_p6_embedded,
        "historical_manifest_canonical_sha256": _stable_hash(old_p6_core),
        "historical_hash_valid": old_p6_embedded == _stable_hash(old_p6_core),
        "reason": "old P6 overlaps development/hardening objects and is not object-disjoint",
    }
    if not old_p6_supersession["historical_hash_valid"]:
        raise ValueError("H3D_OLD_P6_HISTORICAL_HASH_INVALID")

    category_audit = {
        "schema_version": "H3HOCapObjectCategoryAuditV1",
        "status": "CATEGORY_AUTHORITY_UNAVAILABLE",
        "sources": list(CATEGORY_AUDIT_SOURCES),
        "official_release_category_field_found": False,
        "official_alias_crosswalk_found": False,
        "g21_examples": {
            "G21_1": "bottle",
            "G21_2": "playing_cards",
            "G21_3": "spatula",
            "G21_4": "spatula",
        },
        "object_prefix_is_category": False,
        "inference": (
            "shared G21 prefix spans distinct object categories; prefix may only be a "
            "diversity heuristic"
        ),
        "selected_split_type": SPLIT_LABEL,
        "category_disjoint_claim": False,
    }
    receipt = {
        "schema_version": "H3UnseenObjectSelectionReceiptV1",
        "status": "FROZEN_NOT_EXECUTED",
        "selection_seed": args.seed,
        "selection_algorithm": manifest["selection_algorithm"],
        "metadata_only": True,
        "outcome_fields_used": [],
        "downstream_outcomes_used": False,
        "candidate_count": len(candidates),
        "selected_episode_ids": [row["episode_id"] for row in selected],
        "selected_object_ids": [row["target_object"] for row in selected],
        "selected_sequences_unique": len(selected_sequences) == args.count,
        "selected_objects_unique": len(selected_ids) == args.count,
        "selected_meshes_unique": len(selected_meshes) == args.count,
        "selected_subject_count": len({str(row["subject"]) for row in selected}),
        "selected_prefix_count": len({str(row["object_prefix"]) for row in selected}),
        "object_id_overlap_with_development": len(selected_ids & excluded_ids),
        "mesh_sha256_overlap_with_development": len(selected_meshes & excluded_meshes),
        "geometry_hash_overlap_with_development": len(selected_geometry & excluded_geometry),
        "known_alias_overlap_with_development": len(selected_aliases & excluded_aliases),
        "category_overlap_with_development": "UNKNOWN_NO_AUTHORITY",
        "manifest_sha256": manifest["manifest_sha256"],
        "h3_protocol_hash": protocol_hash,
        "H3_EXECUTION_HEAD": execution_head,
        "downstream_execution": "FORBIDDEN_UNTIL_H3C_READINESS",
    }

    output = args.output_root.resolve()
    identity_rows = [identity.as_dict() for identity in identities.values()]
    exclusion_rows = [
        {
            **entry,
            **identities[str(entry["object_id"])].as_dict(),
        }
        for entry in entries
    ]
    candidate_fields = [
        "episode_id",
        "raw_sequence",
        "subject",
        "active_hand",
        "target_object",
        "start_frame",
        "end_frame",
        "duration_frames",
        "canonical_mesh_sha256",
        "geometry_hash",
        "object_prefix",
        "selection_key",
        "selection_rank",
        "metadata_exposure_only",
        "downstream_outcome_observed_for_selection",
    ]
    identity_fields = list(identity_rows[0])
    exclusion_fields = [
        "object_id",
        "exposure_class",
        "evidence_path",
        "evidence_scope",
        *[field for field in identity_fields if field != "object_id"],
    ]
    _write_new(
        output / "unseen_object_frozen5_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        output / "unseen_object_frozen5_manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
    )
    _write_new(output / "manifest_sha256.txt", manifest["manifest_sha256"] + "\n")
    _write_new(
        output / "selection_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        output / "category_audit.json",
        json.dumps(category_audit, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        output / "old_p6_supersession.json",
        json.dumps(old_p6_supersession, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "object_identity_table.csv", identity_rows, identity_fields)
    _write_csv(output / "development_object_exclusions.csv", exclusion_rows, exclusion_fields)
    _write_csv(output / "candidate_pool.csv", candidates, candidate_fields)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
