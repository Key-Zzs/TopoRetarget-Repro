"""Dataset semantic authority contracts.

This module is deliberately independent of the numerical retarget solver.  It
turns an EpisodeV1 row into a small, hashable semantic record, keeps all
object candidates in a deterministic ranking, and blocks ambiguous or
inconsistently bound records before geometric retargeting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any


def canonical_hash(value: object) -> str:
    """Hash JSON using the repository's canonical compact representation."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _file_hash(path: object) -> str | None:
    if not path:
        return None
    candidate = Path(str(path))
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AuthorityStatus(str, Enum):
    TARGET_OBJECT_PASS = "TARGET_OBJECT_PASS"
    TARGET_OBJECT_AMBIGUOUS = "TARGET_OBJECT_AMBIGUOUS"
    TARGET_OBJECT_UNRESOLVED = "TARGET_OBJECT_UNRESOLVED"
    MULTI_OBJECT_INTERACTION = "MULTI_OBJECT_INTERACTION"
    BIMANUAL_SAME_OBJECT = "BIMANUAL_SAME_OBJECT"
    HANDOVER = "HANDOVER"
    OFFICIAL_VS_GEOMETRY_CONFLICT = "OFFICIAL_VS_GEOMETRY_CONFLICT"
    OBJECT_ASSET_BINDING_FAIL = "OBJECT_ASSET_BINDING_FAIL"
    SEMANTIC_PREFLIGHT_PASS = "SEMANTIC_PREFLIGHT_PASS"
    SEMANTIC_PREFLIGHT_QUARANTINE = "SEMANTIC_PREFLIGHT_QUARANTINE"
    SEMANTIC_PREFLIGHT_FAIL = "SEMANTIC_PREFLIGHT_FAIL"


@dataclass(frozen=True)
class CanonicalHOIRecordV1:
    """One canonical, provenance-bound hand/object episode."""

    dataset: str
    dataset_version: str
    subject_id: str
    sequence_id: str
    episode_id: str
    active_hand: str
    hand_model: str
    hand_parameter_authority: str
    target_object_instance_id: str
    target_object_track_id: str
    target_object_asset_id: str
    target_object_mesh_sha256: str | None
    target_object_geometry_hash: str | None
    other_object_ids: tuple[str, ...]
    support_object_ids: tuple[str, ...]
    source_start_frame: int | None
    source_end_frame: int | None
    timestamps: tuple[float, ...]
    fps: float
    approach_frame: int | None
    contact_frame: int | None
    pickup_frame: int | None
    transport_frame: int | None
    place_frame: int | None
    release_frame: int | None
    retreat_frame: int | None
    hand_world_frame_authority: str
    object_world_frame_authority: str
    wrist_frame_authority: str
    time_authority: str
    source_hand_hash: str | None
    object_pose_track_hash: str | None
    object_mesh_hash: str | None
    support_authority: str
    target_selection_method: str
    target_selection_confidence: float
    target_selection_evidence: tuple[Mapping[str, Any], ...]
    semantic_validation_status: str
    semantic_failure_reason: str
    canonical_record_sha256: str = ""

    @classmethod
    def from_episode_row(
        cls,
        row: Mapping[str, Any],
        *,
        object_ids: Sequence[str] = (),
        target_status: str = AuthorityStatus.TARGET_OBJECT_PASS.value,
        target_evidence: Sequence[Mapping[str, Any]] = (),
        confidence: float = 1.0,
    ) -> CanonicalHOIRecordV1:
        provenance = row.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        raw_mano = provenance.get("raw_mano")
        raw_mano = raw_mano if isinstance(raw_mano, Mapping) else {}
        raw_object = provenance.get("raw_object")
        raw_object = raw_object if isinstance(raw_object, Mapping) else {}
        mesh = provenance.get("object_mesh")
        mesh = mesh if isinstance(mesh, Mapping) else {}
        support = row.get("source_support_metadata")
        support = support if isinstance(support, Mapping) else {}
        start = row.get("start_frame")
        end = row.get("end_frame")
        fps = float(row.get("fps") or row.get("timestamps_fps") or 30.0)
        frame_count = (
            max(0, int(end) - int(start)) if isinstance(start, int) and isinstance(end, int) else 0
        )
        timestamps = tuple(float(index) / fps for index in range(frame_count))
        target = str(row.get("target_object") or "")
        all_objects = tuple(sorted({str(value) for value in object_ids if str(value)}))
        others = tuple(value for value in all_objects if value != target)
        object_index = raw_object.get("official_object_index", "unknown")
        mesh_path = mesh.get("path")
        geometry_hash = _file_hash(mesh_path)
        explicit_support = bool(support.get("source_explicit_support_present"))
        reconstructed_support = bool(support.get("source_reconstructed_support_candidate_present"))
        support_authority = (
            "SOURCE_EXPLICIT"
            if explicit_support
            else "SOURCE_RECONSTRUCTED"
            if reconstructed_support
            else "UNRESOLVED"
        )
        values: dict[str, Any] = {
            "dataset": "hocap",
            "dataset_version": "hocap_extracted_v1",
            "subject_id": str(row.get("subject") or ""),
            "sequence_id": str(row.get("raw_sequence") or ""),
            "episode_id": str(row.get("episode_id") or ""),
            "active_hand": str(row.get("active_hand") or ""),
            "hand_model": "MANO_PCA45",
            "hand_parameter_authority": "HOCAP_POSES_M_PCA45",
            "target_object_instance_id": target,
            "target_object_track_id": f"hocap_object_track:{object_index}",
            "target_object_asset_id": target,
            "target_object_mesh_sha256": mesh.get("sha256"),
            "target_object_geometry_hash": geometry_hash,
            "other_object_ids": others,
            "support_object_ids": tuple(
                str(value) for value in row.get("support_object_ids", ()) or ()
            ),
            "source_start_frame": start if isinstance(start, int) else None,
            "source_end_frame": end if isinstance(end, int) else None,
            "timestamps": timestamps,
            "fps": fps,
            "approach_frame": row.get("approach_frame"),
            "contact_frame": row.get("contact_frame"),
            "pickup_frame": row.get("pickup_frame"),
            "transport_frame": row.get("transport_frame"),
            "place_frame": row.get("place_frame"),
            "release_frame": row.get("release_frame"),
            "retreat_frame": row.get("retreat_frame"),
            "hand_world_frame_authority": "HOCAP_WORLD_SCENE_V1",
            "object_world_frame_authority": "HOCAP_OBJECT_POSE_WORLD_V1",
            "wrist_frame_authority": "CANONICAL_KEYPOINT_WRIST_FRAME_V1",
            "time_authority": "HOCAP_NATIVE_FRAME_INDEX_V1",
            "source_hand_hash": raw_mano.get("sha256"),
            "object_pose_track_hash": raw_object.get("sha256"),
            "object_mesh_hash": mesh.get("sha256"),
            "support_authority": support_authority,
            "target_selection_method": "DATASET_RECONSTRUCTED_AUTHORITY_EPISODE_V1",
            "target_selection_confidence": float(confidence),
            "target_selection_evidence": tuple(dict(item) for item in target_evidence),
            "semantic_validation_status": target_status,
            "semantic_failure_reason": ""
            if target_status == AuthorityStatus.TARGET_OBJECT_PASS.value
            else target_status,
        }
        record = cls(**values)
        return replace(
            record, canonical_record_sha256=canonical_hash(record.as_dict(include_hash=False))
        )

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": "CanonicalHOIRecordV1",
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "subject_id": self.subject_id,
            "sequence_id": self.sequence_id,
            "episode_id": self.episode_id,
            "active_hand": self.active_hand,
            "hand_model": self.hand_model,
            "hand_parameter_authority": self.hand_parameter_authority,
            "target_object_instance_id": self.target_object_instance_id,
            "target_object_track_id": self.target_object_track_id,
            "target_object_asset_id": self.target_object_asset_id,
            "target_object_mesh_sha256": self.target_object_mesh_sha256,
            "target_object_geometry_hash": self.target_object_geometry_hash,
            "other_object_ids": list(self.other_object_ids),
            "support_object_ids": list(self.support_object_ids),
            "source_start_frame": self.source_start_frame,
            "source_end_frame": self.source_end_frame,
            "timestamps": list(self.timestamps),
            "fps": self.fps,
            "event_frames": {
                "approach_frame": self.approach_frame,
                "contact_frame": self.contact_frame,
                "pickup_frame": self.pickup_frame,
                "transport_frame": self.transport_frame,
                "place_frame": self.place_frame,
                "release_frame": self.release_frame,
                "retreat_frame": self.retreat_frame,
            },
            "approach_frame": self.approach_frame,
            "contact_frame": self.contact_frame,
            "pickup_frame": self.pickup_frame,
            "transport_frame": self.transport_frame,
            "place_frame": self.place_frame,
            "release_frame": self.release_frame,
            "retreat_frame": self.retreat_frame,
            "hand_world_frame_authority": self.hand_world_frame_authority,
            "object_world_frame_authority": self.object_world_frame_authority,
            "wrist_frame_authority": self.wrist_frame_authority,
            "time_authority": self.time_authority,
            "source_hand_hash": self.source_hand_hash,
            "object_pose_track_hash": self.object_pose_track_hash,
            "object_mesh_hash": self.object_mesh_hash,
            "support_authority": self.support_authority,
            "target_selection_method": self.target_selection_method,
            "target_selection_confidence": self.target_selection_confidence,
            "target_selection_evidence": [dict(item) for item in self.target_selection_evidence],
            "semantic_validation_status": self.semantic_validation_status,
            "semantic_failure_reason": self.semantic_failure_reason,
        }
        if include_hash:
            value["canonical_record_sha256"] = self.canonical_record_sha256
        return value


class TargetObjectAuthorityV1:
    """Rank every candidate pair and fail closed on ambiguity."""

    schema_version = "TargetObjectAuthorityV1"
    clear_margin = 0.15

    @staticmethod
    def _interval(row: Mapping[str, Any]) -> tuple[int, int]:
        contact = row.get("contact_frame")
        release = row.get("release_frame")
        if not isinstance(contact, int) or not isinstance(release, int):
            return (0, 0)
        return contact, max(contact, release)

    @classmethod
    def _score(cls, candidate: Mapping[str, Any], focus: Mapping[str, Any] | None) -> float:
        start, end = cls._interval(candidate)
        f_start, f_end = cls._interval(focus or candidate)
        overlap = max(0, min(end, f_end) - max(start, f_start))
        contact_duration = max(0, end - start)
        displacement = float(candidate.get("object_displacement_m") or 0.0)
        surface = float(candidate.get("min_surface_distance_m") or 1.0)
        complete = bool(candidate.get("complete"))
        eligible = bool(candidate.get("physicalization_v1_eligible"))
        return (
            (1_000_000.0 if eligible else 0.0)
            + (100_000.0 if complete else 0.0)
            + overlap * 1_000.0
            + min(contact_duration, 10_000) * 2.0
            + min(displacement, 1.0) * 100.0
            + max(0.0, 0.01 - min(surface, 0.01)) * 10.0
        )

    @classmethod
    def rank_candidates(
        cls,
        candidates: Sequence[Mapping[str, Any]],
        *,
        focus: Mapping[str, Any] | None = None,
        official_target: str | None = None,
    ) -> dict[str, Any]:
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            item["object_id"] = str(item.get("object_id") or item.get("target_object") or "")
            item["score"] = cls._score(item, focus)
            item["evidence"] = {
                "complete": bool(item.get("complete")),
                "physicalization_v1_eligible": bool(item.get("physicalization_v1_eligible")),
                "contact_duration_frames": max(0, cls._interval(item)[1] - cls._interval(item)[0]),
                "object_displacement_m": float(item.get("object_displacement_m") or 0.0),
                "min_surface_distance_m": float(item.get("min_surface_distance_m") or 1.0),
            }
            ranked.append(item)
        ranked.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["object_id"]),
                str(item.get("episode_id", "")),
            )
        )
        if not ranked or not ranked[0]["object_id"]:
            return {
                "schema_version": cls.schema_version,
                "status": AuthorityStatus.TARGET_OBJECT_UNRESOLVED.value,
                "selected_object_id": None,
                "top1_top2_margin": None,
                "candidates": ranked,
                "official_target": official_target,
            }
        top = float(ranked[0]["score"])
        second = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
        margin = (top - second) / max(abs(top), 1.0)
        tied = len(ranked) > 1 and margin < cls.clear_margin
        selected = str(ranked[0]["object_id"])
        status = (
            AuthorityStatus.TARGET_OBJECT_AMBIGUOUS.value
            if tied
            else AuthorityStatus.TARGET_OBJECT_PASS.value
        )
        if official_target and str(official_target) != selected:
            status = AuthorityStatus.OFFICIAL_VS_GEOMETRY_CONFLICT.value
        return {
            "schema_version": cls.schema_version,
            "status": status,
            "selected_object_id": selected,
            "top1_top2_margin": margin,
            "candidates": ranked,
            "official_target": official_target,
        }


class ObjectAssetBindingV1:
    """Validate identity and mesh authority independently of selection."""

    schema_version = "ObjectAssetBindingV1"

    @classmethod
    def validate(
        cls,
        record: CanonicalHOIRecordV1,
        chain: Mapping[str, Mapping[str, Any] | None],
    ) -> dict[str, Any]:
        checks: dict[str, bool] = {}
        expected_id = record.target_object_instance_id
        expected_mesh = record.target_object_mesh_sha256
        for name in (
            "episode",
            "pose",
            "asset",
            "retarget",
            "viewer",
            "support",
            "development_exclusion",
        ):
            item = chain.get(name)
            checks[f"{name}_object_id"] = (
                item is None or str(item.get("object_id", expected_id)) == expected_id
            )
            checks[f"{name}_mesh_sha256"] = (
                item is None or item.get("mesh_sha256", expected_mesh) == expected_mesh
            )
        checks["episode_pose_asset_identity"] = bool(
            checks["episode_object_id"] and checks["pose_object_id"] and checks["asset_object_id"]
        )
        checks["mesh_identity"] = bool(
            checks["episode_mesh_sha256"] and checks["asset_mesh_sha256"]
        )
        passed = all(checks.values())
        return {
            "schema_version": cls.schema_version,
            "status": "PASS" if passed else AuthorityStatus.OBJECT_ASSET_BINDING_FAIL.value,
            "episode_id": record.episode_id,
            "target_object_id": expected_id,
            "target_object_mesh_sha256": expected_mesh,
            "checks": checks,
            "identity_tuple": {
                "ID_episode": chain.get("episode", {}).get("object_id")
                if chain.get("episode")
                else expected_id,
                "ID_pose": chain.get("pose", {}).get("object_id")
                if chain.get("pose")
                else expected_id,
                "ID_asset": chain.get("asset", {}).get("object_id")
                if chain.get("asset")
                else expected_id,
                "ID_retarget": chain.get("retarget", {}).get("object_id")
                if chain.get("retarget")
                else expected_id,
                "ID_viewer": chain.get("viewer", {}).get("object_id")
                if chain.get("viewer")
                else expected_id,
                "ID_support": chain.get("support", {}).get("object_id")
                if chain.get("support")
                else expected_id,
            },
        }


class HOISemanticPreflightV1:
    """Admission gate used before exact geometric retargeting."""

    schema_version = "HOISemanticPreflightV1"

    @classmethod
    def evaluate(
        cls,
        record: CanonicalHOIRecordV1,
        *,
        target_authority: Mapping[str, Any],
        binding: Mapping[str, Any],
        episode_type: str,
        frame_pass: bool = True,
        time_pass: bool = True,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        target_status = str(target_authority.get("status", ""))
        if target_status in {
            AuthorityStatus.TARGET_OBJECT_AMBIGUOUS.value,
            AuthorityStatus.MULTI_OBJECT_INTERACTION.value,
            AuthorityStatus.BIMANUAL_SAME_OBJECT.value,
            AuthorityStatus.HANDOVER.value,
        }:
            reasons.append(target_status)
        elif target_status != AuthorityStatus.TARGET_OBJECT_PASS.value:
            reasons.append(target_status or AuthorityStatus.TARGET_OBJECT_UNRESOLVED.value)
        if str(binding.get("status")) != "PASS":
            reasons.append(AuthorityStatus.OBJECT_ASSET_BINDING_FAIL.value)
        if record.active_hand not in {"left", "right"}:
            reasons.append("ACTIVE_HAND_AUTHORITY_INVALID")
        if episode_type != "SINGLE_HAND_PICK_PLACE":
            reasons.append(f"EPISODE_TYPE_NOT_SINGLE_HAND:{episode_type}")
        event_values = (
            record.source_start_frame,
            record.source_end_frame,
            record.approach_frame,
            record.contact_frame,
            record.pickup_frame,
            record.transport_frame,
            record.place_frame,
            record.release_frame,
            record.retreat_frame,
        )
        if any(value is None for value in event_values):
            reasons.append("COMPLETE_LIFECYCLE_REQUIRED")
        if not frame_pass:
            reasons.append("FRAME_AUTHORITY_FAIL")
        if not time_pass:
            reasons.append("TIME_AUTHORITY_FAIL")
        if reasons:
            status = (
                AuthorityStatus.SEMANTIC_PREFLIGHT_QUARANTINE.value
                if target_status
                in {
                    AuthorityStatus.TARGET_OBJECT_AMBIGUOUS.value,
                    AuthorityStatus.MULTI_OBJECT_INTERACTION.value,
                    AuthorityStatus.BIMANUAL_SAME_OBJECT.value,
                    AuthorityStatus.HANDOVER.value,
                }
                else AuthorityStatus.SEMANTIC_PREFLIGHT_FAIL.value
            )
        else:
            status = AuthorityStatus.SEMANTIC_PREFLIGHT_PASS.value
        return {
            "schema_version": cls.schema_version,
            "status": status,
            "episode_id": record.episode_id,
            "target_object_id": record.target_object_instance_id,
            "reasons": reasons,
            "checks": {
                "active_hand_authority": record.active_hand in {"left", "right"},
                "target_object_authority": target_status
                == AuthorityStatus.TARGET_OBJECT_PASS.value,
                "object_asset_binding": str(binding.get("status")) == "PASS",
                "episode_lifecycle": not any(value is None for value in event_values),
                "frame_authority": frame_pass,
                "time_authority": time_pass,
                "canonical_record_hash": bool(record.canonical_record_sha256),
            },
            "canonical_record_sha256": record.canonical_record_sha256,
        }


class DatasetSemanticAuthorityV1:
    """Facade coordinating the independent semantic authorities."""

    schema_version = "DatasetSemanticAuthorityV1"

    def build_record(
        self,
        row: Mapping[str, Any],
        *,
        object_ids: Sequence[str],
        target_authority: Mapping[str, Any],
    ) -> CanonicalHOIRecordV1:
        return CanonicalHOIRecordV1.from_episode_row(
            row,
            object_ids=object_ids,
            target_status=str(
                target_authority.get("status", AuthorityStatus.TARGET_OBJECT_UNRESOLVED.value)
            ),
            target_evidence=target_authority.get("candidates", ()),
            confidence=float(target_authority.get("top1_top2_margin") or 0.0),
        )

    def preflight(
        self,
        row: Mapping[str, Any],
        *,
        object_ids: Sequence[str],
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[CanonicalHOIRecordV1, dict[str, Any], dict[str, Any]]:
        target = TargetObjectAuthorityV1.rank_candidates(
            candidates,
            focus=row,
            official_target=str(row.get("target_object") or "") or None,
        )
        record = self.build_record(row, object_ids=object_ids, target_authority=target)
        provenance = row.get("provenance") if isinstance(row.get("provenance"), Mapping) else {}
        mesh = provenance.get("object_mesh") if isinstance(provenance, Mapping) else {}
        mesh = mesh if isinstance(mesh, Mapping) else {}
        binding = ObjectAssetBindingV1.validate(
            record,
            {
                "episode": {
                    "object_id": row.get("target_object"),
                    "mesh_sha256": mesh.get("sha256"),
                },
                "pose": {"object_id": row.get("target_object"), "mesh_sha256": mesh.get("sha256")},
                "asset": {"object_id": row.get("target_object"), "mesh_sha256": mesh.get("sha256")},
            },
        )
        preflight = HOISemanticPreflightV1.evaluate(
            record,
            target_authority=target,
            binding=binding,
            episode_type=str(row.get("episode_type") or ""),
        )
        return record, binding, preflight


__all__ = [
    "AuthorityStatus",
    "CanonicalHOIRecordV1",
    "DatasetSemanticAuthorityV1",
    "HOISemanticPreflightV1",
    "ObjectAssetBindingV1",
    "TargetObjectAuthorityV1",
    "canonical_hash",
]
