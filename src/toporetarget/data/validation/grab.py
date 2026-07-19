"""Structured validation for canonical GRAB sequences and cache round-trips."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.se3 import object_to_scene
from toporetarget.viz.errors import ComparisonMetrics


@dataclass
class GrabValidationReport:
    status: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass" and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.passed else "fail",
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "metrics": self.metrics,
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return destination

    def write_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["category", "name", "available", "value", "status"])
            for category, payload in (("check", self.checks), ("metric", self.metrics)):
                for name, value in payload.items():
                    available = value.get("available", True) if isinstance(value, dict) else True
                    writer.writerow(
                        [
                            category,
                            name,
                            available,
                            json.dumps(value, default=str),
                            _csv_status(value),
                        ]
                    )
            for error in self.errors:
                writer.writerow(["error", error, True, "", "fail"])
        return destination


def _csv_status(value: Any) -> str:
    """Report a useful CSV status instead of marking every emitted row pass."""

    failing_flags = {
        "finite",
        "in_expected_range",
        "frame_alignment",
        "vertex_alignment",
        "contact_vertex_count_match",
        "contact_frame_count_match",
        "timestamp_monotonic",
        "source_tracks_preserved",
        "no_temporal_resampling",
        "no_spatial_sampling",
        "fully_mapped",
        "readable",
    }

    def contains_failure(item: Any) -> bool:
        if isinstance(item, dict):
            if item.get("equal") is False:
                return True
            if any(key in failing_flags and child is False for key, child in item.items()):
                return True
            return any(contains_failure(child) for child in item.values())
        if isinstance(item, list):
            return any(contains_failure(child) for child in item)
        return False

    if contains_failure(value):
        return "fail"
    if isinstance(value, dict) and value.get("available") is False:
        return "unavailable"
    return "pass" if value is not False else "fail"


def _summary(values: np.ndarray | None) -> dict[str, Any]:
    if values is None:
        return {"available": False, "reason": "unavailable"}
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"available": False, "reason": "no finite values"}
    return {
        "available": True,
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "max": float(np.max(np.abs(finite))),
        "mean": float(np.mean(np.abs(finite))),
    }


def _track_error(first: np.ndarray | None, second: np.ndarray | None) -> dict[str, Any]:
    if first is None or second is None or first.shape != second.shape:
        return {"available": False, "reason": "track unavailable or shape mismatch"}
    return _summary(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))


def _unavailable(reason: str = "unavailable") -> dict[str, Any]:
    return {"available": False, "reason": reason}


def _contact_semantic_checks(sequence: HOISequence) -> dict[str, Any]:
    """Summarize raw, binary, and official semantic contact invariants."""

    if not sequence.contacts:
        return {
            "contact_mode": "none",
            "raw_labels_present": False,
            "binary_mask_present": False,
            "semantic_ids_present": False,
            "semantic_mapping_present": False,
            "source_mapping_id": None,
            "source_mapping_version": None,
            "source_mapping_hash": None,
            "mapping_config_hash": None,
            "observed_labels": [],
            "mapped_labels": [],
            "unmapped_labels": [],
            "label_range_valid": _unavailable("no contact track"),
            "contact_vertex_count": _unavailable("no contact track"),
            "object_vertex_count": _unavailable("no contact track"),
            "contact_vertex_count_match": _unavailable("no contact track"),
            "contact_frame_count": _unavailable("no contact track"),
            "canonical_frame_count": sequence.num_frames,
            "contact_frame_count_match": _unavailable("no contact track"),
            "left_hand_contact_vertex_count": _unavailable("no contact track"),
            "right_hand_contact_vertex_count": _unavailable("no contact track"),
            "body_contact_vertex_count": _unavailable("no contact track"),
            "no_contact_vertex_count": _unavailable("no contact track"),
            "semantic_roundtrip_exact": _unavailable("canonical cache not supplied"),
            "raw_roundtrip_exact": _unavailable("canonical cache not supplied"),
            "binary_roundtrip_exact": _unavailable("canonical cache not supplied"),
            "mapping_hash_consistent": _unavailable("canonical cache not supplied"),
            "fully_mapped": _unavailable("no contact track"),
        }
    contact = sequence.contacts[0]
    object_track = sequence.rigid_object(contact.object_id)
    labels = contact.labels
    binary = contact.binary
    semantic = contact.semantic_ids
    mapping = contact.semantic_mapping
    observed = sorted(int(item) for item in np.unique(labels)) if labels is not None else []
    mapped = sorted(int(item) for item in observed if mapping is not None and int(item) in mapping)
    unmapped = sorted(set(observed) - set(mapped)) if mapping is not None else []
    label_range = _unavailable("raw labels unavailable")
    if labels is not None:
        expected_range = contact.metadata.get("expected_label_range")
        label_range = {
            "available": True,
            "min": int(labels.min()) if labels.size else None,
            "max": int(labels.max()) if labels.size else None,
            "finite": bool(np.all(np.isfinite(labels))),
            "expected": expected_range,
            "in_expected_range": (
                all(expected_range[0] <= int(item) <= expected_range[1] for item in observed)
                if expected_range is not None
                else _unavailable("official range not attached")
            ),
        }
    category_counts: dict[str, int] = {
        "left_hand": 0,
        "right_hand": 0,
        "body": 0,
        "none": 0,
        "unknown": 0,
    }
    if semantic is not None and mapping is not None:
        for semantic_id, definition in mapping.items():
            category = str(definition.get("category", "unknown"))
            category_counts[category] = category_counts.get(category, 0) + int(
                np.count_nonzero(semantic == int(semantic_id))
            )
    contact_vertices: int | dict[str, Any] = _unavailable("raw labels unavailable")
    object_vertices = int(object_track.mesh.vertices_local.shape[0])
    contact_frames: int | dict[str, Any] = _unavailable("raw labels unavailable")
    if labels is not None:
        contact_vertices = int(labels.shape[-1])
        contact_frames = int(labels.shape[0])
    return {
        "contact_mode": contact.metadata.get("mode", "unknown"),
        "raw_labels_present": labels is not None,
        "binary_mask_present": binary is not None,
        "semantic_ids_present": semantic is not None,
        "semantic_mapping_present": mapping is not None,
        "source_mapping_id": contact.metadata.get("mapping_id"),
        "source_mapping_version": contact.metadata.get("mapping_version"),
        "source_mapping_hash": contact.metadata.get("source_mapping_hash"),
        "mapping_config_hash": contact.metadata.get("mapping_config_sha256"),
        "observed_labels": observed,
        "mapped_labels": mapped,
        "unmapped_labels": unmapped,
        "unmapped_vertex_count": contact.metadata.get("unmapped_vertex_count", 0),
        "unmapped_frame_count": contact.metadata.get("unmapped_frame_count", 0),
        "label_range_valid": label_range,
        "contact_vertex_count": contact_vertices,
        "object_vertex_count": object_vertices,
        "contact_vertex_count_match": (
            contact_vertices == object_vertices
            if isinstance(contact_vertices, int)
            else _unavailable("raw labels unavailable")
        ),
        "contact_frame_count": contact_frames,
        "canonical_frame_count": sequence.num_frames,
        "contact_frame_count_match": (
            contact_frames == sequence.num_frames
            if isinstance(contact_frames, int)
            else _unavailable("raw labels unavailable")
        ),
        "left_hand_contact_vertex_count": category_counts["left_hand"]
        if semantic is not None
        else _unavailable("semantic IDs unavailable"),
        "right_hand_contact_vertex_count": category_counts["right_hand"]
        if semantic is not None
        else _unavailable("semantic IDs unavailable"),
        "body_contact_vertex_count": category_counts["body"]
        if semantic is not None
        else _unavailable("semantic IDs unavailable"),
        "no_contact_vertex_count": category_counts["none"]
        if semantic is not None
        else _unavailable("semantic IDs unavailable"),
        "semantic_roundtrip_exact": _unavailable("canonical cache not supplied"),
        "raw_roundtrip_exact": _unavailable("canonical cache not supplied"),
        "binary_roundtrip_exact": _unavailable("canonical cache not supplied"),
        "mapping_hash_consistent": _unavailable("canonical cache not supplied"),
        "fully_mapped": (
            bool(contact.metadata.get("fully_mapped"))
            if semantic is not None
            else _unavailable("semantic IDs unavailable")
        ),
    }


def validate_grab_sequence(
    sequence: HOISequence,
    *,
    canonical: str | Path | HOISequence | None = None,
    source_path: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Return source/hand/object/table/contact/temporal/cache checks."""

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    try:
        errors.extend(sequence.validate(raise_on_error=False))
    except Exception as exc:
        errors.append(str(exc))
    source = Path(source_path or sequence.metadata.provenance.source_file or "")
    checks["source"] = {
        "path": str(source) if source else None,
        "readable": source.is_file() if source else False,
        "source_hash": sequence.metadata.provenance.source_hash,
        "source_tracks_preserved": bool(
            sequence.metadata.metadata.get("source_tracks_preserved", False)
        ),
    }
    if source_path is not None and not source.is_file():
        errors.append(f"source path is not readable: {source}")
    checks["temporal"] = {
        "frame_count": sequence.num_frames,
        "native_fps": sequence.metadata.native_fps,
        "timestamp_monotonic": bool(np.all(np.diff(sequence.timestamps) > 0))
        if sequence.num_frames > 1
        else True,
        "timestamp_formula_or_source_preserved": True,
        "no_temporal_resampling": sequence.metadata.provenance.no_temporal_resampling,
        "no_spatial_sampling": sequence.metadata.provenance.no_spatial_sampling,
    }
    if not checks["temporal"]["timestamp_monotonic"]:
        errors.append("timestamps are not strictly increasing")
    hand_checks: dict[str, Any] = {}
    for hand in sequence.hands:
        prefix = hand.hand_id
        vertices = hand.vertices_scene
        native = next(
            (track for name, track in hand.keypoint_tracks.items() if name != "mediapipe21"), None
        )
        mediapipe = hand.keypoint_tracks.get("mediapipe21")
        roundtrip = None
        if vertices is not None:
            try:
                roundtrip = _track_error(
                    vertices,
                    sequence.wrist_to_scene(prefix, sequence.scene_to_wrist(prefix, vertices)),
                )
            except Exception as exc:
                roundtrip = {"available": False, "reason": str(exc)}
        bones = None
        if mediapipe is not None and mediapipe.positions_scene.shape[1] > 1:
            lengths = np.linalg.norm(np.diff(mediapipe.positions_scene, axis=1), axis=-1)
            bones = {
                "zero_length_count": int(np.count_nonzero(lengths <= 1e-12)),
                "available": True,
            }
        hand_checks[prefix] = {
            "side": hand.side,
            "vertices_shape": None if vertices is None else list(vertices.shape),
            "native_joint_shape": None if native is None else list(native.positions_scene.shape),
            "mediapipe21_shape": None
            if mediapipe is None
            else list(mediapipe.positions_scene.shape),
            "finite_vertices": vertices is None or bool(np.all(np.isfinite(vertices))),
            "wrist_pose_shape": list(hand.wrist_pose_scene.pose_scene.shape),
            "personalized_vtemp": hand.mano_parameters is not None
            and hand.mano_parameters.personalized_v_template_reference is not None,
            "mano_model_hash": hand.metadata.get("mano_model_hash"),
            "mapping_profile": None
            if mediapipe is None
            else mediapipe.provenance.get("mapping_profile_id"),
            "scene_wrist_roundtrip": roundtrip,
            "bone_degeneracy": bones or {"available": False, "reason": "mediapipe21 unavailable"},
        }
    checks["hands"] = hand_checks
    if not sequence.rigid_objects:
        errors.append("primary object track is missing")
    object_checks: dict[str, Any] = {}
    for obj in sequence.rigid_objects:
        world = object_to_scene(
            obj.pose_scene.pose_scene,
            np.broadcast_to(
                obj.mesh.vertices_local, (sequence.num_frames, obj.mesh.vertices_local.shape[0], 3)
            ),
        )
        object_checks[obj.object_id] = {
            "role": obj.metadata.get("role"),
            "mesh_shape": [list(obj.mesh.vertices_local.shape), list(obj.mesh.faces.shape)],
            "pose_shape": list(obj.pose_scene.pose_scene.shape),
            "finite_world_vertices": bool(np.all(np.isfinite(world))),
            "mesh_hash": obj.mesh.mesh_hash,
            "frame_count": obj.pose_scene.pose_scene.shape[0],
        }
    checks["objects"] = object_checks
    checks["table"] = {
        item.object_id: {
            "role": item.metadata.get("role"),
            "mesh_present": True,
            "frame_count": item.pose_scene.pose_scene.shape[0],
        }
        for item in sequence.rigid_objects
        if item.object_id == "table"
    }
    checks["contacts"] = []
    for contact in sequence.contacts:
        payload = {
            "hand_id": contact.hand_id,
            "object_id": contact.object_id,
            "mode": contact.metadata.get("mode"),
            "source_shape": None if contact.labels is None else list(contact.labels.shape),
            "binary_shape": None if contact.binary is None else list(contact.binary.shape),
            "frame_alignment": contact.valid.shape[0] == sequence.num_frames,
            "vertex_alignment": contact.labels is None
            or contact.labels.shape[-1]
            == sequence.rigid_object(contact.object_id).mesh.vertices_local.shape[0],
            "unique_labels": contact.metadata.get("unique_numeric_labels", []),
            "semantic_mapping_status": contact.metadata.get("semantic_mapping_status"),
            "semantic_ids_present": contact.semantic_ids is not None,
            "semantic_mapping_present": contact.semantic_mapping is not None,
            "unmapped_labels": contact.metadata.get("unmapped_labels", []),
            "fully_mapped": contact.metadata.get("fully_mapped"),
        }
        checks["contacts"].append(payload)
    checks["contact_semantics"] = _contact_semantic_checks(sequence)
    if canonical is not None:
        other = load_hoi_sequence(canonical) if isinstance(canonical, (str, Path)) else canonical
        comparison = ComparisonMetrics.compute(sequence, other).as_dict()
        metrics = comparison.get("metrics", {})
        for name in (
            "hand_vertex_rmse_m",
            "hand_vertex_max_error_m",
            "hand_keypoint_rmse_m",
            "hand_keypoint_max_error_m",
            "wrist_translation_error_m",
            "wrist_rotation_geodesic_deg",
            "object_pose_translation_error_m",
            "object_pose_rotation_geodesic_deg",
            "object_world_vertex_rmse_m",
            "object_world_vertex_max_error_m",
            "timestamp_abs_error_s",
            "timestamp_max_abs_error_s",
        ):
            metrics.setdefault(name, {"available": False, "reason": "unavailable"})
        checks["round_trip"] = {
            "frame_count_match": comparison.get("frame_count_match"),
            "timestamp_count_match": comparison.get("timestamp_count_match"),
            "contact_exact_equality": _contact_equality(sequence, other),
            "source_tracks_preserved": checks["source"]["source_tracks_preserved"],
        }
        equality = checks["round_trip"]["contact_exact_equality"]
        checks["contact_semantics"].update(
            {
                "semantic_roundtrip_exact": equality["semantic"],
                "raw_roundtrip_exact": equality["raw"],
                "binary_roundtrip_exact": equality["binary"],
                "mapping_hash_consistent": equality["mapping"],
            }
        )
    else:
        metrics = {"round_trip": {"available": False, "reason": "canonical cache not supplied"}}
    status = "pass" if not errors else "fail"
    if strict and any(
        item.get("semantic_mapping_status") == "unavailable_no_verified_official_mapping"
        for item in checks["contacts"]
    ):
        warnings.append(
            "semantic contact mapping is unavailable; numeric labels remain authoritative"
        )
    report = GrabValidationReport(status, errors, warnings, checks, metrics)
    return report.as_dict()


def _contact_equality(first: HOISequence, second: HOISequence) -> dict[str, Any]:
    left = {(item.hand_id, item.object_id): item for item in first.contacts}
    right = {(item.hand_id, item.object_id): item for item in second.contacts}
    if set(left) != set(right):
        mismatch = {"available": True, "equal": False, "reason": "contact track IDs differ"}
        return {
            "available": True,
            "equal": False,
            "raw": mismatch,
            "binary": mismatch,
            "semantic": mismatch,
            "mapping": mismatch,
        }

    def equal_or_missing(first: np.ndarray | None, second: np.ndarray | None) -> bool:
        if first is None or second is None:
            return first is None and second is None
        return bool(np.array_equal(first, second))

    raw_equal = all(equal_or_missing(left[key].labels, right[key].labels) for key in left)
    binary_equal = all(equal_or_missing(left[key].binary, right[key].binary) for key in left)
    semantic_equal = all(
        equal_or_missing(left[key].semantic_ids, right[key].semantic_ids) for key in left
    )
    mapping_equal = all(
        left[key].metadata.get("mapping_config_sha256")
        == right[key].metadata.get("mapping_config_sha256")
        and left[key].semantic_mapping == right[key].semantic_mapping
        for key in left
    )
    return {
        "available": True,
        "equal": raw_equal and binary_equal and semantic_equal and mapping_equal,
        "raw": {"available": True, "equal": raw_equal},
        "binary": {"available": True, "equal": binary_equal},
        "semantic": {"available": True, "equal": semantic_equal},
        "mapping": {"available": True, "equal": mapping_equal},
    }


__all__ = ["GrabValidationReport", "validate_grab_sequence"]
