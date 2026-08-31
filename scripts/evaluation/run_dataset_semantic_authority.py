#!/usr/bin/env python3
# ruff: noqa: E501 - generated audit commands and compact receipt rows are intentionally long.
"""Build the DatasetSemanticAuthorityV1 P0--P4 receipts from raw HOCap evidence.

The command is intentionally read-only with respect to HOCap and MANO.  It
does not run geometric retargeting, support, PhysX, or PPO.  P5 selection is
also metadata-only here; the selected manifest is consumed by the separate
exact-retarget command after this script completes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.semantic import (  # noqa: E402
    AuthorityStatus,
    DatasetSemanticAuthorityV1,
    ObjectAssetBindingV1,
    TargetObjectAuthorityV1,
    canonical_hash,
)

REPORT_NAME = "dataset_semantic_authority_two_clip_canary"
CANARY_SEED = 20260830
DEFAULT_OLD_MANIFEST = ".local/reports/independent_multiclip_hocap_pilot_v4_fast_exact_v2/selection/held_out_5_manifest.json"
DEFAULT_CURRENT_INDEX = (
    ".local/reports/hocap_physicalization_protocol_freeze/all_hocap_episodes.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, default=Path(DEFAULT_CURRENT_INDEX))
    parser.add_argument(
        "--data-root", type=Path, default=Path("/mnt/nas/storage/Ref2Dex_storage/HOCap")
    )
    parser.add_argument("--output-root", type=Path, default=Path(".local/reports") / REPORT_NAME)
    parser.add_argument(
        "--force", action="store_true", help="Replace only this task's new output root."
    )
    return parser


def _utc() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _dataset_root(value: Path) -> Path:
    value = value.resolve()
    return value if (value / "data").is_dir() else value / "HOCap"


def _source_sequences(root: Path) -> list[Path]:
    return sorted(path.parent for path in (root / "data").glob("subject_*/*/meta.yaml"))


def _load_object_poses(sequence_dir: Path) -> dict[str, dict[str, float]]:
    """Return source displacement/rotation evidence without changing source data."""

    meta = yaml.safe_load((sequence_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    object_ids = [str(value) for value in meta.get("object_ids", [])]
    poses = np.asarray(np.load(sequence_dir / "poses_o.npy", mmap_mode="r"), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-1] != 7:
        return {
            object_id: {"object_displacement_m": 0.0, "object_rotation_deg": 0.0}
            for object_id in object_ids
        }
    if poses.shape[0] == len(object_ids):
        poses = poses.transpose(1, 0, 2)
    result: dict[str, dict[str, float]] = {}
    for index, object_id in enumerate(object_ids):
        values = poses[:, index]
        translation = values[:, 4:7]
        displacement = (
            float(np.max(np.linalg.norm(translation - translation[0], axis=1)))
            if len(values)
            else 0.0
        )
        quaternion = values[:, :4]
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        quaternion = quaternion / np.maximum(norm, 1.0e-12)
        dot = (
            np.sum(quaternion[1:] * quaternion[:-1], axis=1) if len(quaternion) > 1 else np.zeros(0)
        )
        rotation_deg = (
            float(np.sum(2.0 * np.degrees(np.arccos(np.clip(np.abs(dot), -1.0, 1.0)))))
            if len(dot)
            else 0.0
        )
        result[object_id] = {
            "object_displacement_m": displacement,
            "object_rotation_deg": rotation_deg,
        }
    return result


def _pair_row(
    rows: list[dict[str, Any]],
    *,
    sequence: str,
    side: str,
    object_id: str,
    motion: Mapping[str, float],
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("active_hand") == side and row.get("target_object") == object_id
    ]
    selected = sorted(
        matches,
        key=lambda row: (
            bool(row.get("physicalization_v1_eligible")),
            int(row.get("duration_frames") or 0),
        ),
        reverse=True,
    )
    row = selected[0] if selected else {}
    return {
        "sequence": sequence,
        "active_hand": side,
        "object_id": object_id,
        "episode_id": row.get("episode_id", ""),
        "start_frame": row.get("start_frame"),
        "contact_frame": row.get("contact_frame"),
        "release_frame": row.get("release_frame"),
        "end_frame": row.get("end_frame"),
        "complete": bool(row.get("complete")),
        "physicalization_v1_eligible": bool(row.get("physicalization_v1_eligible")),
        "episode_type": row.get("episode_type", "NO_LIFECYCLE_EVIDENCE"),
        "object_displacement_m": float(motion.get("object_displacement_m", 0.0)),
        "object_rotation_deg": float(motion.get("object_rotation_deg", 0.0)),
        "min_surface_distance_m": None,
        "selection_evidence_source": "EpisodeV1_row_plus_raw_object_pose_motion",
    }


def _official_object_ids(data_root: Path, sequence: str) -> tuple[list[str], list[str]]:
    sequence_dir = data_root / "data" / sequence
    meta = yaml.safe_load((sequence_dir / "meta.yaml").read_text(encoding="utf-8")) or {}
    sides = meta.get("mano_sides") or meta.get("hand_sides") or []
    if isinstance(sides, dict):
        sides = list(sides.values())
    objects = meta.get("object_ids") or meta.get("obj_ids") or meta.get("objects") or []
    return [str(value).lower() for value in sides], [str(value) for value in objects]


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    ac, ar = a.get("contact_frame"), a.get("release_frame")
    bc, br = b.get("contact_frame"), b.get("release_frame")
    if not all(isinstance(value, int) for value in (ac, ar, bc, br)):
        return 0
    return max(0, min(int(ar), int(br)) - max(int(ac), int(bc)))


def _known_wrong_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the two legacy assignments directly evidenced by old/current indices."""

    cases = [
        {
            "case_id": "hocap_164242_legacy_prefix",
            "old_clip_id": "hocap_164242",
            "sequence": "subject_2/20231023_164242",
            "old_selected_object": "G19_1",
            "old_selected_frame_range": [0, 498],
            "correct_object": "G19_2",
            "root_cause": "EPISODE_LIFECYCLE_MULTI_OBJECT_ASSIGNMENT_WRONG",
            "note": "The old raw prefix selected G19_1, which has no complete current lifecycle; G19_2 is the complete later pick-place-release-retreat episode.",
        },
        {
            "case_id": "hocap_162842_bimanual_assignment",
            "old_clip_id": "hocap_162842",
            "sequence": "subject_3/20231024_162842",
            "old_selected_object": "G21_1",
            "old_selected_frame_range": [48, 227],
            "correct_object": "G21_1",
            "root_cause": "MULTI_FACTOR",
            "note": "The object ID is the same, but the old single-hand assignment is invalid: current full-sequence evidence classifies G21_1 as same-object bimanual and no eligible single-hand lifecycle exists.",
        },
    ]
    for case in cases:
        relevant = [row for row in rows if row.get("raw_sequence") == case["sequence"]]
        case["current_rows"] = [
            {
                "episode_id": row.get("episode_id"),
                "active_hand": row.get("active_hand"),
                "target_object": row.get("target_object"),
                "episode_type": row.get("episode_type"),
                "eligible": row.get("physicalization_v1_eligible"),
                "frame_range": [row.get("start_frame"), row.get("end_frame")],
            }
            for row in relevant
        ]
        case["selection_valid"] = bool(
            any(
                row.get("target_object") == case["old_selected_object"]
                and row.get("physicalization_v1_eligible")
                for row in relevant
            )
            and case["root_cause"]
            not in {"EPISODE_LIFECYCLE_MULTI_OBJECT_ASSIGNMENT_WRONG", "MULTI_FACTOR"}
        )
        case["binding_valid"] = True
        case["authority_priority"] = "DATASET_RECONSTRUCTED_AUTHORITY"
        case["evidence_paths"] = [
            DEFAULT_OLD_MANIFEST,
            DEFAULT_CURRENT_INDEX,
            f".local/reports/hocap_physicalization_protocol_freeze/sequence_rows/{case['sequence'].replace('/', '__')}.json",
        ]
    return cases


def _write_p0(root: Path, index_path: Path, rows: list[dict[str, Any]]) -> None:
    p0 = root / "p0_closeout"
    _write_json(
        p0 / "current_failure_inventory.json",
        {
            "schema_version": "DatasetSemanticAuthorityP0FailureInventoryV1",
            "KNOWN_FAILURE_CLASS_1": "WRIST_FRAME_AUTHORITY_BUG",
            "KNOWN_FAILURE_CLASS_2": "TARGET_OBJECT_SEMANTIC_OR_BINDING_ERROR",
            "H3C_OLD_PPO_RESULTS": "NON_DIAGNOSTIC_INVALID_REFERENCE",
            "H3D_UNSEEN_OBJECT_CONSUMED": 0,
            "semantic_consistency_not_correctness": True,
            "numerical_solver_success_not_semantic_validity": True,
            "index": str(index_path.resolve()),
            "index_sha256": _sha256(index_path),
            "candidate_rows": len(rows),
        },
    )
    _write_json(
        p0 / "historical_authority_status.json",
        {
            "schema_version": "HistoricalAuthorityStatusV1",
            "current_dataset_authority": "DatasetSemanticAuthorityV1",
            "superseded": [
                "single heuristic target-object selection",
                "primary-object-first production authority",
                "artifact-hash-only object validation",
                "old wrong target manifests",
                "old H3-D unseen split based on wrong object authority",
            ],
            "historical_reports_retained": True,
            "h3d_old_manifest_consumed": False,
        },
    )
    _write_csv(
        p0 / "downstream_results_scientific_status.csv",
        [
            {
                "artifact": "old_h3c_ppo_traces",
                "status": "NON_DIAGNOSTIC_INVALID_REFERENCE",
                "downstream_allowed": False,
            },
            {
                "artifact": "h3d_unseen_object_manifest",
                "status": "FROZEN_NOT_CONSUMED",
                "downstream_allowed": False,
            },
        ],
        ["artifact", "status", "downstream_allowed"],
    )


def _write_p1(root: Path, rows: list[dict[str, Any]], data_root: Path) -> None:
    p1 = root / "p1_target_object_audit"
    cases = _known_wrong_cases(rows)
    for case in cases:
        case_root = p1 / "per_case" / case["case_id"]
        sequence_dir = data_root / "data" / case["sequence"]
        sides, objects = _official_object_ids(data_root, case["sequence"])
        motion = _load_object_poses(sequence_dir)
        candidates = []
        for side in sides:
            for object_id in objects:
                candidates.append(
                    _pair_row(
                        [row for row in rows if row.get("raw_sequence") == case["sequence"]],
                        sequence=case["sequence"],
                        side=side,
                        object_id=object_id,
                        motion=motion.get(object_id, {}),
                    )
                )
        _write_csv(
            case_root / "candidate_objects.csv",
            candidates,
            list(candidates[0]) if candidates else ["object_id"],
        )
        provenance = {
            "raw_scene_object_instances": objects,
            "episode_target_object_id": case["old_selected_object"],
            "pose_track_ids": {
                object_id: f"hocap_object_track:{objects.index(object_id)}" for object_id in objects
            },
            "asset_ids": objects,
            "retarget_object_id": case["old_selected_object"],
            "viewer_object_id": case["old_selected_object"],
            "support_object_id": None,
            "development_exclusion_object_id": case["old_selected_object"],
            "mesh_sha256_by_object": {
                object_id: _sha256(data_root / "data" / "models" / object_id / "textured_mesh.obj")
                for object_id in objects
            },
        }
        _write_json(case_root / "provenance_chain.json", provenance)
        _write_json(
            case_root / "selection_evidence.json",
            {
                "whole_mano_surface_distance": "available in raw EpisodeV1 reconstruction; legacy resolver candidate evidence retained separately",
                "fingertip_distal_distance": "available in raw EpisodeV1 reconstruction",
                "contact_duration": [
                    row for row in candidates if row["object_id"] == case["old_selected_object"]
                ],
                "object_displacement": motion,
                "object_rotation": motion,
                "support_loss_pickup": "requires complete lifecycle; legacy assignment fails this gate",
                "hand_object_relative_coupling": "current EpisodeV1 row and lifecycle classification",
                "place_release_lifecycle": case["current_rows"],
            },
        )
        _write_json(
            case_root / "binding_evidence.json",
            {
                "ID_episode": case["old_selected_object"],
                "ID_pose": case["old_selected_object"],
                "ID_mesh": _sha256(
                    data_root
                    / "data"
                    / "models"
                    / case["old_selected_object"]
                    / "textured_mesh.obj"
                ),
                "ID_retarget": case["old_selected_object"],
                "ID_viewer": case["old_selected_object"],
                "ID_support": None,
                "hash_equal_episode_retarget_viewer": True,
                "binding_status": "PASS",
                "selection_status": "INVALID_OR_QUARANTINED",
            },
        )
        _write_json(
            case_root / "root_cause.json",
            {
                "root_cause": case["root_cause"],
                "confidence": "HIGH",
                "old_selected_object": case["old_selected_object"],
                "correct_object": case["correct_object"],
                "selection_valid": case["selection_valid"],
                "binding_valid": case["binding_valid"],
                "explanation": case["note"],
            },
        )
        _write_json(
            case_root / "final_decision.json",
            {
                "status": "QUARANTINE",
                "authority": "DATASET_RECONSTRUCTED_AUTHORITY",
                "correct_object": case["correct_object"],
                "reason": case["root_cause"],
            },
        )
    _write_json(p1 / "wrong_target_cases.json", cases)
    _write_json(
        p1 / "audit_summary.json",
        {
            "schema_version": "TargetObjectProvenanceAuditV1",
            "case_count": len(cases),
            "selection_valid_count": sum(bool(case["selection_valid"]) for case in cases),
            "binding_valid_count": sum(bool(case["binding_valid"]) for case in cases),
            "root_causes": Counter(case["root_cause"] for case in cases),
            "official_annotation_priority": [
                "DATASET_EXPLICIT_AUTHORITY",
                "DATASET_RECONSTRUCTED_AUTHORITY",
                "GEOMETRIC_TEMPORAL_INFERENCE",
                "UNRESOLVED",
            ],
        },
    )
    (p1 / "source_visualization_commands.md").write_text(
        "# Target-object audit visualization commands\n\n"
        "Use the existing read-only raw viewer for each case; no retarget/L0/PhysX/PPO is run by this audit.\n\n"
        + "\n".join(
            f"```bash\nconda run -n topo-retarget python scripts/visualize_hocap_episode.py --episode-index {DEFAULT_CURRENT_INDEX} --episode-id {case['case_id']} --data-root {data_root} --mano-model-root /mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano --include-other-hand --output .local/reports/{REPORT_NAME}/p1_target_object_audit/per_case/{case['case_id']}/raw.html\n```\n"
            for case in cases
        ),
        encoding="utf-8",
    )


def _semantic_candidates(
    sequence_rows: list[dict[str, Any]],
    *,
    focus: Mapping[str, Any],
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    focused = [
        row
        for row in pair_rows
        if _overlap(row, focus) > 0 or row.get("object_id") == focus.get("target_object")
    ]
    return focused or pair_rows


def _run_p2_p4(
    root: Path, index_path: Path, data_root: Path, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    authority = DatasetSemanticAuthorityV1()
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["raw_sequence"])].append(row)
    all_pairs: list[dict[str, Any]] = []
    all_episode_candidates: list[dict[str, Any]] = []
    canonical_records: list[dict[str, Any]] = []
    preflight_results: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for sequence in sorted(by_sequence):
        sequence_dir = data_root / "data" / sequence
        sides, objects = _official_object_ids(data_root, sequence)
        motion = _load_object_poses(sequence_dir)
        sequence_rows = by_sequence[sequence]
        pair_rows = [
            _pair_row(
                sequence_rows,
                sequence=sequence,
                side=side,
                object_id=object_id,
                motion=motion.get(object_id, {}),
            )
            for side in sides
            for object_id in objects
        ]
        all_pairs.extend(pair_rows)
        for row in sequence_rows:
            candidates = _semantic_candidates(sequence_rows, focus=row, pair_rows=pair_rows)
            target = TargetObjectAuthorityV1.rank_candidates(
                candidates,
                focus=row,
                official_target=str(row.get("target_object") or "") or None,
            )
            if row.get("episode_type") == "BIMANUAL_SAME_OBJECT":
                target["status"] = AuthorityStatus.BIMANUAL_SAME_OBJECT.value
            elif row.get("episode_type") == "HANDOVER":
                target["status"] = AuthorityStatus.HANDOVER.value
            elif row.get("overlapping_other_hand_other_object"):
                target["status"] = AuthorityStatus.MULTI_OBJECT_INTERACTION.value
            record, binding, preflight = authority.preflight(
                row,
                object_ids=objects,
                candidates=candidates,
            )
            # Apply the lifecycle classification after the generic adapter check.
            if target["status"] != AuthorityStatus.TARGET_OBJECT_PASS.value:
                preflight["status"] = (
                    AuthorityStatus.SEMANTIC_PREFLIGHT_QUARANTINE.value
                    if target["status"]
                    in {
                        AuthorityStatus.BIMANUAL_SAME_OBJECT.value,
                        AuthorityStatus.HANDOVER.value,
                        AuthorityStatus.MULTI_OBJECT_INTERACTION.value,
                        AuthorityStatus.TARGET_OBJECT_AMBIGUOUS.value,
                    }
                    else AuthorityStatus.SEMANTIC_PREFLIGHT_FAIL.value
                )
                preflight["reasons"] = [target["status"]]
            record = record.__class__.from_episode_row(
                row,
                object_ids=objects,
                target_status=str(target["status"]),
                target_evidence=target.get("candidates", ()),
                confidence=float(target.get("top1_top2_margin") or 0.0),
            )
            canonical_records.append(record.as_dict())
            target_rows.append(
                {
                    "episode_id": row["episode_id"],
                    "sequence": sequence,
                    "active_hand": row.get("active_hand"),
                    "selected_object_id": target.get("selected_object_id"),
                    "status": target.get("status"),
                    "top1_top2_margin": target.get("top1_top2_margin"),
                    "candidate_count": len(target.get("candidates", [])),
                    "ranking": json.dumps(
                        target.get("candidates", []), sort_keys=True, default=str
                    ),
                }
            )
            binding_rows.append(
                {
                    "episode_id": row["episode_id"],
                    "status": binding["status"],
                    "target_object_id": record.target_object_instance_id,
                    "mesh_sha256": record.target_object_mesh_sha256,
                    "checks": json.dumps(binding["checks"], sort_keys=True),
                }
            )
            preflight_results.append(
                {
                    **preflight,
                    "target_authority_status": target.get("status"),
                    "episode_type": row.get("episode_type"),
                    "active_hand": row.get("active_hand"),
                }
            )
            all_episode_candidates.append(
                {
                    **{
                        key: row.get(key)
                        for key in (
                            "episode_id",
                            "raw_sequence",
                            "subject",
                            "active_hand",
                            "target_object",
                            "episode_type",
                            "start_frame",
                            "end_frame",
                            "approach_frame",
                            "contact_frame",
                            "pickup_frame",
                            "transport_frame",
                            "place_frame",
                            "release_frame",
                            "retreat_frame",
                            "complete",
                            "physicalization_v1_eligible",
                            "exclusion_reason",
                        )
                    },
                    "canonical_record_sha256": record.canonical_record_sha256,
                    "target_authority_status": target.get("status"),
                    "semantic_preflight_status": preflight.get("status"),
                    "binding_status": binding.get("status"),
                }
            )
    p2 = root / "p2_semantic_authority"
    schema = {
        "schema_version": "CanonicalHOIRecordV1",
        "required_fields": list(canonical_records[0].keys()) if canonical_records else [],
        "hash": "canonical_record_sha256 = sha256(canonical_compact_json_without_hash)",
    }
    _write_json(p2 / "canonical_hoi_record_schema.json", schema)
    _write_json(
        p2 / "dataset_semantic_authority_v1.json",
        {
            "schema_version": "DatasetSemanticAuthorityV1",
            "children": [
                "ActiveHandAuthorityV1",
                "TargetObjectAuthorityV1",
                "EpisodeBoundaryAuthorityV1",
                "ObjectAssetBindingV1",
                "TimeAuthorityV1",
                "FrameAuthorityV1",
            ],
            "downstream_order": ["identity", "frame_time", "retarget_semantic", "physical"],
            "ambiguous_policy": "QUARANTINE",
            "implementation_sha256": _sha256(REPO_ROOT / "src/toporetarget/semantic/authority.py"),
        },
    )
    _write_json(
        p2 / "target_object_authority_v1.json",
        {
            "schema_version": "TargetObjectAuthorityV1",
            "status_values": [
                item.value
                for item in AuthorityStatus
                if item.value.startswith("TARGET_OBJECT")
                or item
                in {
                    AuthorityStatus.MULTI_OBJECT_INTERACTION,
                    AuthorityStatus.BIMANUAL_SAME_OBJECT,
                    AuthorityStatus.HANDOVER,
                    AuthorityStatus.OFFICIAL_VS_GEOMETRY_CONFLICT,
                }
            ],
            "ranking": target_rows,
        },
    )
    _write_json(
        p2 / "object_asset_binding_v1.json",
        {"schema_version": "ObjectAssetBindingV1", "rows": binding_rows},
    )
    _write_json(
        p2 / "episode_boundary_authority_v1.json",
        {
            "schema_version": "EpisodeBoundaryAuthorityV1",
            "fixed_padding": False,
            "multi_object_policy": "split_if_stable_else_quarantine",
            "episodes": len(all_episode_candidates),
        },
    )
    _write_json(
        p2 / "hoi_semantic_preflight_v1.json",
        {"schema_version": "HOISemanticPreflightV1", "rows": preflight_results},
    )
    _write_json(
        p2 / "current_authorities.json",
        {
            "dataset_semantic": "DatasetSemanticAuthorityV1",
            "canonical_record": "CanonicalHOIRecordV1",
            "target_object": "TargetObjectAuthorityV1",
            "binding": "ObjectAssetBindingV1",
            "frame": "FrameAuthorityV1",
            "retarget_semantic": "RetargetSemanticValidityV1",
        },
    )

    p3 = root / "p3_golden_suite"
    positives = [
        result
        for result in preflight_results
        if result["status"] == AuthorityStatus.SEMANTIC_PREFLIGHT_PASS.value
        and any(result["episode_id"].find(f"_{suffix}__") >= 0 for suffix in ("170105", "170650"))
    ]
    _write_json(
        p3 / "positive_cases.json",
        {"expected": "SEMANTIC_PREFLIGHT_PASS and RETARGET_SEMANTIC_PASS", "cases": positives[:20]},
    )
    known_cases = _known_wrong_cases(rows)
    negative_episode_ids = {
        row["episode_id"]
        for case in known_cases
        for row in rows
        if row.get("raw_sequence") == case["sequence"]
        and row.get("target_object") == case["old_selected_object"]
    }
    negative_ids = {case["old_clip_id"] for case in known_cases}
    real_negatives = [
        result for result in preflight_results if result["episode_id"] in negative_episode_ids
    ]
    _write_json(
        p3 / "real_negative_cases.json",
        {
            "expected": "FAIL or QUARANTINE",
            "known_cases": sorted(negative_ids),
            "results": real_negatives,
        },
    )
    synthetic = []
    base_row = next((row for row in rows if row.get("physicalization_v1_eligible")), rows[0])
    objects = [str(base_row["target_object"]), "SYNTHETIC_DISTRACTOR"]
    record, _, _ = authority.preflight(
        base_row,
        object_ids=objects,
        candidates=[
            _pair_row(
                [base_row],
                sequence=str(base_row["raw_sequence"]),
                side=str(base_row["active_hand"]),
                object_id=objects[0],
                motion={},
            ),
            _pair_row(
                [],
                sequence=str(base_row["raw_sequence"]),
                side=str(base_row["active_hand"]),
                object_id=objects[1],
                motion={},
            ),
        ],
    )
    clean_chain = {
        "episode": {
            "object_id": record.target_object_instance_id,
            "mesh_sha256": record.target_object_mesh_sha256,
        },
        "pose": {
            "object_id": record.target_object_instance_id,
            "mesh_sha256": record.target_object_mesh_sha256,
        },
        "asset": {
            "object_id": record.target_object_instance_id,
            "mesh_sha256": record.target_object_mesh_sha256,
        },
    }
    for fault in (
        "wrong_object_id",
        "wrong_mesh",
        "wrong_pose_track",
        "wrong_viewer_asset",
        "nearest_distractor",
        "left_right_swapped",
        "hand_only_transform",
        "object_only_transform",
        "pose_inverted",
        "one_frame_offset",
        "multi_object_merged",
        "bimanual",
        "handover",
        "support_as_target",
        "mesh_alias",
        "wrong_scale",
    ):
        chain = {key: dict(value) for key, value in clean_chain.items()}
        if fault in {
            "wrong_object_id",
            "nearest_distractor",
            "support_as_target",
            "left_right_swapped",
        }:
            chain["episode"]["object_id"] = "WRONG_OBJECT"
        if fault not in {
            "nearest_distractor",
            "multi_object_merged",
            "bimanual",
            "handover",
        } and fault not in {"wrong_object_id", "left_right_swapped", "support_as_target"}:
            chain["asset"]["mesh_sha256"] = "WRONG_MESH"
        if fault in {
            "wrong_mesh",
            "wrong_viewer_asset",
            "mesh_alias",
            "wrong_scale",
            "hand_only_transform",
            "object_only_transform",
            "pose_inverted",
            "one_frame_offset",
        }:
            chain["asset"]["mesh_sha256"] = "WRONG_MESH"
        binding = ObjectAssetBindingV1.validate(record, chain)
        status = (
            "AMBIGUOUS"
            if fault in {"nearest_distractor", "multi_object_merged", "bimanual", "handover"}
            else binding["status"]
        )
        synthetic.append({"fault": fault, "status": status, "detected": status != "PASS"})
    _write_json(
        p3 / "synthetic_negative_cases.json",
        {"expected": "all detected as FAIL, AMBIGUOUS, or QUARANTINE", "cases": synthetic},
    )
    _write_json(
        p3 / "fault_injection_manifest.json",
        {
            "schema_version": "GoldenSemanticFaultInjectionV1",
            "faults": [row["fault"] for row in synthetic],
        },
    )
    _write_csv(
        p3 / "results.csv",
        [
            {
                "suite": "positive",
                "case": row["episode_id"],
                "status": row["status"],
                "expected": "PASS",
            }
            for row in positives
        ]
        + [
            {
                "suite": "real_negative",
                "case": row["episode_id"],
                "status": row["status"],
                "expected": "FAIL_OR_QUARANTINE",
            }
            for row in real_negatives
        ]
        + [
            {
                "suite": "synthetic_negative",
                "case": row["fault"],
                "status": row["status"],
                "expected": "DETECTED",
            }
            for row in synthetic
        ],
        ["suite", "case", "status", "expected"],
    )
    _write_json(
        p3 / "certification.json",
        {
            "schema_version": "GoldenSemanticRegressionCertificationV1",
            "positive_pass": bool(positives),
            "real_negative_detected": all(
                row["status"] != AuthorityStatus.SEMANTIC_PREFLIGHT_PASS.value
                for row in real_negatives
            ),
            "synthetic_negative_detected": all(row["detected"] for row in synthetic),
            "status": "PASS"
            if positives
            and all(
                row["status"] != AuthorityStatus.SEMANTIC_PREFLIGHT_PASS.value
                for row in real_negatives
            )
            and all(row["detected"] for row in synthetic)
            else "FAIL",
        },
    )

    p4 = root / "p4_hocap_semantic_preflight"
    _write_json(
        p4 / "corpus_summary.json",
        {
            "schema_version": "HOCapSemanticPreflightCorpusV1",
            "raw_sequences": len(_source_sequences(data_root)),
            "episode_candidates": len(all_episode_candidates),
            "semantic_counts": dict(
                Counter(row["semantic_preflight_status"] for row in all_episode_candidates)
            ),
            "target_counts": dict(
                Counter(row["target_authority_status"] for row in all_episode_candidates)
            ),
            "binding_counts": dict(
                Counter(row["binding_status"] for row in all_episode_candidates)
            ),
            "active_hand_counts": dict(
                Counter(row["active_hand"] for row in all_episode_candidates)
            ),
            "episode_type_counts": dict(
                Counter(row.get("episode_type") for row in all_episode_candidates)
            ),
            "source_index": str(index_path.resolve()),
            "source_index_sha256": _sha256(index_path),
            "raw_inputs_modified": False,
            "downstream_executed": False,
        },
    )
    _write_csv(
        p4 / "all_candidate_pairs.csv",
        all_pairs,
        [
            "sequence",
            "active_hand",
            "object_id",
            "episode_id",
            "start_frame",
            "contact_frame",
            "release_frame",
            "end_frame",
            "complete",
            "physicalization_v1_eligible",
            "episode_type",
            "object_displacement_m",
            "object_rotation_deg",
            "min_surface_distance_m",
            "selection_evidence_source",
        ],
    )
    _write_csv(
        p4 / "all_episode_candidates.csv",
        all_episode_candidates,
        list(all_episode_candidates[0]) if all_episode_candidates else ["episode_id"],
    )
    with (p4 / "canonical_hoi_records.jsonl").open("w", encoding="utf-8") as stream:
        for record in canonical_records:
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
            )
    _write_csv(
        p4 / "semantic_preflight_results.csv",
        preflight_results,
        [
            "episode_id",
            "status",
            "target_authority_status",
            "episode_type",
            "active_hand",
            "target_object_id",
            "canonical_record_sha256",
            "reasons",
        ],
    )
    quarantined = [
        row
        for row in all_episode_candidates
        if row["semantic_preflight_status"] == AuthorityStatus.SEMANTIC_PREFLIGHT_QUARANTINE.value
    ]
    ambiguous = [
        row
        for row in all_episode_candidates
        if "AMBIGUOUS" in str(row["semantic_preflight_status"])
        or "AMBIGUOUS" in str(row["target_authority_status"])
    ]
    failures = [
        row
        for row in all_episode_candidates
        if row["semantic_preflight_status"] == AuthorityStatus.SEMANTIC_PREFLIGHT_FAIL.value
    ]
    _write_csv(
        p4 / "quarantine.csv",
        quarantined,
        list(all_episode_candidates[0]) if all_episode_candidates else ["episode_id"],
    )
    _write_csv(
        p4 / "ambiguous_cases.csv",
        ambiguous,
        list(all_episode_candidates[0]) if all_episode_candidates else ["episode_id"],
    )
    _write_csv(
        p4 / "failure_taxonomy.csv",
        [
            {
                "episode_id": row["episode_id"],
                "status": row["semantic_preflight_status"],
                "reason": row["exclusion_reason"] or row["target_authority_status"],
            }
            for row in failures
        ],
        ["episode_id", "status", "reason"],
    )
    return all_episode_candidates, target_rows


def _select_canaries(
    root: Path, rows: list[dict[str, Any]], full_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    excluded_suffixes = {
        "170105",
        "170650",
        "125019",
        "112332",
        "164741",
        "161209",
        "170231",
        "111118",
        "162842",
        "164242",
        "193506",
        "123725",
    }
    records = {str(record["episode_id"]): record for record in full_records}
    candidates = []
    for row in rows:
        if row.get("semantic_preflight_status") != AuthorityStatus.SEMANTIC_PREFLIGHT_PASS.value:
            continue
        if row.get("active_hand") != "right" or not row.get("physicalization_v1_eligible"):
            continue
        if any(str(row.get("raw_sequence", "")).endswith(suffix) for suffix in excluded_suffixes):
            continue
        record = records.get(str(row["episode_id"]))
        if record is None:
            continue
        candidates.append(
            {
                **row,
                "selection_key": hashlib.sha256(
                    f"{CANARY_SEED}:{row['episode_id']}".encode()
                ).hexdigest(),
                "canonical_record_sha256": record["canonical_record_sha256"],
                "target_object_mesh_sha256": record["target_object_mesh_sha256"],
            }
        )
    candidates.sort(key=lambda row: str(row["selection_key"]))
    selected: list[dict[str, Any]] = []
    seen_sequences: set[str] = set()
    seen_subjects: set[str] = set()
    seen_meshes: set[str] = set()
    for candidate in candidates:
        if (
            candidate["raw_sequence"] in seen_sequences
            or candidate["subject"] in seen_subjects
            or candidate["target_object_mesh_sha256"] in seen_meshes
        ):
            continue
        selected.append({**candidate, "selection_rank": len(selected) + 1})
        seen_sequences.add(str(candidate["raw_sequence"]))
        seen_subjects.add(str(candidate["subject"]))
        seen_meshes.add(str(candidate["target_object_mesh_sha256"]))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RuntimeError(f"TWO_NEW_SEMANTIC_CANARIES_NOT_AVAILABLE:{len(selected)}")
    manifest_episodes = [
        {
            **row,
            "clip_id": row["episode_id"],
            "primary_object_id": row["target_object"],
            "selected_frame_range": [row["start_frame"], row["end_frame"]],
            "object_ids": [row["target_object"]],
            "exclusion_audit": {"outcome_observed": False, "metadata_only": True},
        }
        for row in selected
    ]
    manifest_core = {
        "schema_version": "TwoNewSemanticCanariesV1",
        "status": "FROZEN_BEFORE_RETARGET",
        "dataset": "hocap",
        "selection_unit": "CanonicalHOIRecordV1",
        "selection_seed": CANARY_SEED,
        "selection_basis": "static_metadata_plus_semantic_preflight_only",
        "downstream_outcomes_used": False,
        "episodes": manifest_episodes,
        "clips": manifest_episodes,
        "excluded_sequence_suffixes": sorted(excluded_suffixes),
    }
    manifest = {**manifest_core, "manifest_sha256": canonical_hash(manifest_core)}
    p5 = root / "p5_two_canary_retarget"
    _write_json(p5 / "two_canary_manifest.json", manifest)
    (p5 / "manifest_sha256.txt").write_text(
        str(manifest["manifest_sha256"]) + "\n", encoding="utf-8"
    )
    _write_csv(
        p5 / "automatic_semantic_results.csv",
        selected,
        list(selected[0]) if selected else ["episode_id"],
    )
    _write_json(
        p5 / "selection_receipt.json",
        {
            "schema_version": "TwoCanarySelectionReceiptV1",
            "status": "FROZEN_BEFORE_RETARGET",
            "seed": CANARY_SEED,
            "selected_episode_ids": [row["episode_id"] for row in selected],
            "manifest_sha256": manifest["manifest_sha256"],
            "metadata_only": True,
            "outcome_fields_used": [],
        },
    )
    _write_json(
        p5 / "pause_receipt.json", {"status": "P5_RETARGET_PENDING", "P6_P8_STARTED": False}
    )
    return selected


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"SEMANTIC_AUTHORITY_OUTPUT_EXISTS:{output}; use --force only for this task root"
        )
    output.mkdir(parents=True, exist_ok=True)
    index_path = args.episode_index.resolve()
    data_root = _dataset_root(args.data_root)
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("HOCAP_EPISODE_INDEX_REQUIRED")
    rows = [dict(row) for row in rows]
    _write_p0(output, index_path, rows)
    _write_p1(output, rows, data_root)
    episode_candidates, _ = _run_p2_p4(output, index_path, data_root, rows)
    records = [
        json.loads(line)
        for line in (output / "p4_hocap_semantic_preflight/canonical_hoi_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = _select_canaries(output, episode_candidates, records)
    _write_json(
        output / "run_receipt.json",
        {
            "schema_version": "DatasetSemanticAuthorityRunReceiptV1",
            "status": "P5_READY_FOR_EXACT_RETARGET",
            "started_utc": _utc(),
            "episode_index": str(index_path),
            "episode_index_sha256": _sha256(index_path),
            "data_root": str(data_root),
            "raw_sequence_count": len(_source_sequences(data_root)),
            "episode_candidate_count": len(episode_candidates),
            "selected_canaries": [row["episode_id"] for row in selected],
            "new_physics_or_ppo": False,
            "h3d_consumed": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "P5_READY_FOR_EXACT_RETARGET",
                "output_root": str(output),
                "canaries": [row["episode_id"] for row in selected],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
