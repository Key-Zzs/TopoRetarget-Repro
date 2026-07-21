"""Deterministic, bounded GRAB contact-window selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache import path_hash
from .schema import WorkflowRequest, stable_hash

CONTACT_THRESHOLD_ASSUMPTION = "A_WORKFLOW_CONTACT_WINDOW_THRESHOLD_001"
SOURCE_SANITY_ASSUMPTION = "A_WORKFLOW_SOURCE_CONTACT_SANITY_001"


@dataclass(frozen=True)
class ContactWindow:
    sequence: str
    hand: str
    start_frame: int
    end_frame: int
    contact_frame_count: int
    contact_frame_ratio: float
    total_hand_contact_vertices: int
    median_hand_contact_vertices: float
    max_hand_contact_vertices: int
    no_contact_frames: list[int]
    observed_semantic_labels: list[int]
    contact_frames: list[int] = field(default_factory=list)
    contact_counts: dict[str, int] = field(default_factory=dict)
    source_contact_median_distance_m: float | None = None
    source_contact_min_distance_m: float | None = None
    source_geometry_status: str = "not_checked"
    object_mesh_audit: dict[str, Any] = None  # type: ignore[assignment]
    rejection_reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.object_mesh_audit is None:
            object.__setattr__(self, "object_mesh_audit", {})
        if self.rejection_reasons is None:
            object.__setattr__(self, "rejection_reasons", [])

    @property
    def score(self) -> tuple[float, int, float, int, str]:
        distance = self.source_contact_median_distance_m
        return (
            -self.contact_frame_ratio,
            -self.total_hand_contact_vertices,
            math.inf if distance is None else distance,
            self.start_frame,
            self.sequence,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "hand": self.hand,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_range": [self.start_frame, self.end_frame],
            "window_length": self.end_frame - self.start_frame,
            "contact_frame_count": self.contact_frame_count,
            "contact_frame_ratio": self.contact_frame_ratio,
            "total_hand_contact_vertices": self.total_hand_contact_vertices,
            "median_hand_contact_vertices": self.median_hand_contact_vertices,
            "max_hand_contact_vertices": self.max_hand_contact_vertices,
            "no_contact_frames": list(self.no_contact_frames),
            "observed_semantic_labels": list(self.observed_semantic_labels),
            "contact_frames": list(self.contact_frames),
            "contact_counts": dict(self.contact_counts),
            "source_contact_median_distance_m": self.source_contact_median_distance_m,
            "source_contact_min_distance_m": self.source_contact_min_distance_m,
            "source_geometry_status": self.source_geometry_status,
            "object_mesh_audit": self.object_mesh_audit,
            "rejection_reasons": self.rejection_reasons,
            "score": list(self.score),
        }


def _index_entries(index: Path) -> tuple[dict[str, Any], Path]:
    index = index.expanduser()
    index_file = index / "index.jsonl"
    manifest_file = index / "manifest.json"
    if not index_file.is_file() or not manifest_file.is_file():
        raise FileNotFoundError(f"GRAB index requires index.jsonl and manifest.json: {index}")
    entries: dict[str, Any] = {}
    for line in index_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("status") != "deleted":
                entries[str(item["sequence_id"])] = item
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return entries, Path(str(manifest["grab_root"])).expanduser()


def resolve_index_sequence(index: str | Path, sequence: str) -> tuple[dict[str, Any], Path]:
    entries, root = _index_entries(Path(index))
    try:
        item = entries[sequence]
    except KeyError as exc:
        raise ValueError(f"sequence {sequence!r} is not in the explicit GRAB index") from exc
    return item, (root / str(item["relative_path"])).resolve()


def _semantic_contact_counts(
    labels: np.ndarray, *, hand: str, mapping: Any
) -> tuple[np.ndarray, list[int]]:
    labels = np.asarray(labels, dtype=np.int64)
    allowed = {
        int(label_id)
        for label_id, definition in mapping.labels.items()
        if definition.get("is_hand") and definition.get("side") == hand
    }
    if not allowed:
        raise ValueError(f"official semantic mapping has no labels for {hand}")
    per_frame = np.count_nonzero(np.isin(labels, sorted(allowed)), axis=1)
    observed = sorted(int(value) for value in np.unique(labels[np.isin(labels, sorted(allowed))]))
    return per_frame.astype(np.int64), observed


def _object_mesh_audit(
    source_path: Path, root: Path
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    # Import the adapter package before the raw reader.  The existing
    # ``data.adapters`` package re-exports the GRAB adapter, whose import also
    # references the reader; preloading it avoids a reader/adapters package
    # initialization cycle in selector-only processes.
    import importlib

    importlib.import_module("toporetarget.data.adapters.grab")
    from toporetarget.data.readers.grab import load_ply_mesh, read_grab_npz, resolve_grab_resource
    from toporetarget.geometry.mesh_audit import audit_mesh

    record = read_grab_npz(source_path)
    mesh_path = resolve_grab_resource(root, record.object.mesh_relative, "GRAB object mesh")
    vertices, faces = load_ply_mesh(mesh_path)
    audit = audit_mesh(vertices, faces, source_path=mesh_path)
    return audit.as_dict(), vertices, faces


def _nearest_distances(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(queries) == 0:
        return np.asarray([], dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(points).query(queries, k=1)[0], dtype=np.float64)
    except ImportError:
        result = np.full(len(queries), np.inf, dtype=np.float64)
        for start in range(0, len(queries), 256):
            chunk = queries[start : start + 256]
            delta = chunk[:, None, :] - points[None, :, :]
            result[start : start + len(chunk)] = np.sqrt(
                np.min(np.sum(delta * delta, axis=-1), axis=1)
            )
        return result


def _source_geometry_sanity(
    source_path: Path,
    root: Path,
    *,
    hand: str,
    start: int,
    end: int,
    labels: np.ndarray,
    mapping: Any,
    mano_model_root: Path | None,
) -> tuple[str, float | None, float | None, dict[str, Any]]:
    """Check source contact proximity for one candidate without changing data."""

    from toporetarget.data.adapters.base import FrameRange
    from toporetarget.data.adapters.grab import GrabDatasetAdapter, GrabLoadOptions

    audit, vertices, _ = _object_mesh_audit(source_path, root)
    if audit.get("sign_reliability") != "reliable_watertight":
        return "rejected_non_watertight_object", None, None, audit
    if mano_model_root is None:
        return "not_checked_missing_mano_model_root", None, None, audit
    adapter = GrabDatasetAdapter(
        sequence_path=source_path,
        mano_model_root=mano_model_root,
        options=GrabLoadOptions(
            hands=hand,
            start_frame=start,
            end_frame=end,
            include_table=True,
            contact_mode="semantic",
            include_mediapipe21=False,
        ),
    )
    sequence = adapter.load_sequence(
        sequence=f"{source_path.parent.name}/{source_path.stem}",
        frame_range=FrameRange(start, end),
        options=GrabLoadOptions(
            hands=hand,
            start_frame=start,
            end_frame=end,
            include_table=True,
            contact_mode="semantic",
            include_mediapipe21=False,
        ),
    )
    hand_track = sequence.hands[0]
    object_track = sequence.rigid_objects[0]
    contact_ids = {
        int(label_id)
        for label_id, definition in mapping.labels.items()
        if definition.get("is_hand") and definition.get("side") == hand
    }
    distances: list[np.ndarray] = []
    raw_labels = labels[start:end]
    for local_frame in range(end - start):
        selected = np.flatnonzero(np.isin(raw_labels[local_frame], sorted(contact_ids)))
        if len(selected) == 0:
            continue
        object_points = (
            np.asarray(
                object_track.pose_scene.pose_scene[local_frame, :3, :3] @ vertices[selected].T
            ).T
            + object_track.pose_scene.pose_scene[local_frame, :3, 3]
        )
        source_vertices = (
            np.asarray(hand_track.vertices_scene[local_frame])
            if hand_track.vertices_scene is not None
            else np.asarray(hand_track.keypoint_tracks["mano_native"].positions_scene[local_frame])
        )
        distances.append(_nearest_distances(source_vertices, object_points))
    if not distances:
        return "rejected_no_source_contact_vertices", None, None, audit
    values = np.concatenate(distances)
    return "pass", float(np.median(values)), float(np.min(values)), audit


def _candidate(
    sequence: str,
    hand: str,
    start: int,
    end: int,
    per_frame: np.ndarray,
    observed: list[int],
) -> ContactWindow:
    selected = np.asarray(per_frame[start:end], dtype=np.int64)
    contact_frames = int(np.count_nonzero(selected > 0))
    return ContactWindow(
        sequence=sequence,
        hand=hand,
        start_frame=start,
        end_frame=end,
        contact_frame_count=contact_frames,
        contact_frame_ratio=float(contact_frames / len(selected)),
        total_hand_contact_vertices=int(np.sum(selected)),
        median_hand_contact_vertices=float(np.median(selected)) if len(selected) else 0.0,
        max_hand_contact_vertices=int(np.max(selected)) if len(selected) else 0,
        no_contact_frames=[int(start + index) for index in np.flatnonzero(selected == 0)],
        contact_frames=[int(start + index) for index in np.flatnonzero(selected > 0)],
        contact_counts={
            str(start + index): int(value) for index, value in enumerate(selected.tolist())
        },
        observed_semantic_labels=observed,
    )


def select_contact_windows(
    request: WorkflowRequest,
    *,
    mano_model_root: Path | None = None,
    max_sanity_candidates: int = 5,
) -> dict[str, Any]:
    """Select from one explicit sequence or an explicitly constrained query."""

    import importlib

    importlib.import_module("toporetarget.data.adapters.grab")
    from toporetarget.data.contacts.grab import load_grab_contact_mapping
    from toporetarget.data.readers.grab import load_grab_auxiliary, read_grab_npz

    request.validate()
    entry, source_path = resolve_index_sequence(request.index, request.sequence)
    root = Path(json.loads((request.index / "manifest.json").read_text())["grab_root"])
    record = read_grab_npz(source_path)
    if request.hand not in record.hands:
        raise ValueError(f"selected sequence has no {request.hand} hand")
    mapping = load_grab_contact_mapping()
    auxiliary = load_grab_auxiliary(source_path, contact_mode="semantic", include_table=False)
    raw_contact = auxiliary.get("contact")
    if not isinstance(raw_contact, dict) or "object" not in raw_contact:
        raise ValueError("selected GRAB sequence has no official object contact labels")
    labels = np.asarray(raw_contact["object"], dtype=np.int64)
    per_frame, observed = _semantic_contact_counts(labels, hand=request.hand, mapping=mapping)
    if request.start_frame is not None:
        starts = [request.start_frame]
    else:
        starts = list(range(0, record.num_frames - request.window_length + 1))
    candidates: list[ContactWindow] = []
    for start in starts:
        end = start + request.window_length
        if end > record.num_frames:
            continue
        candidates.append(
            _candidate(request.sequence, request.hand, start, end, per_frame, observed)
        )
    if not candidates:
        raise ValueError("no complete contact-window candidates in the selected sequence")

    ranked = sorted(candidates, key=lambda item: item.score)
    inspected: list[ContactWindow] = []
    for candidate in ranked[:max_sanity_candidates]:
        status, median, minimum, audit = _source_geometry_sanity(
            source_path,
            root,
            hand=request.hand,
            start=candidate.start_frame,
            end=candidate.end_frame,
            labels=labels,
            mapping=mapping,
            mano_model_root=mano_model_root,
        )
        reasons: list[str] = []
        if candidate.contact_frame_ratio < request.minimum_contact_frame_ratio:
            reasons.append("contact_frame_ratio_below_engineering_gate")
        if status == "rejected_non_watertight_object":
            reasons.append("object_mesh_not_strictly_watertight")
        if status == "rejected_no_source_contact_vertices":
            reasons.append("source_contact_geometry_empty")
        if median is not None and median > request.maximum_source_contact_median_distance_m:
            reasons.append("source_contact_median_distance_above_engineering_gate")
        if status == "not_checked_missing_mano_model_root":
            reasons.append("source_geometry_not_checked_missing_mano_model_root")
        inspected.append(
            ContactWindow(
                **{
                    **candidate.__dict__,
                    "source_contact_median_distance_m": median,
                    "source_contact_min_distance_m": minimum,
                    "source_geometry_status": status,
                    "object_mesh_audit": audit,
                    "rejection_reasons": reasons,
                }
            )
        )
    accepted = [
        item
        for item in inspected
        if not item.rejection_reasons and item.source_geometry_status == "pass"
    ]
    selected = sorted(accepted, key=lambda item: item.score)[0] if accepted else None
    return {
        "schema_version": "toporetarget.contact_window_selection.v1",
        "status": "pass" if selected is not None else "fail",
        "sequence": request.sequence,
        "source_path": str(source_path),
        "source_hash": entry.get("sha256") or entry.get("source_hash") or path_hash(source_path),
        "subject": record.subject_id,
        "object": record.object_name,
        "action": record.motion_intent,
        "hand": request.hand,
        "window_length": request.window_length,
        "thresholds": {
            "minimum_contact_frame_ratio": request.minimum_contact_frame_ratio,
            "maximum_source_contact_median_distance_m": (
                request.maximum_source_contact_median_distance_m
            ),
            "assumptions": [CONTACT_THRESHOLD_ASSUMPTION, SOURCE_SANITY_ASSUMPTION],
        },
        "mapping": {
            "mapping_id": mapping.mapping_id,
            "mapping_version": mapping.mapping_version,
            "mapping_config_sha256": mapping.config_sha256,
            "source_sha256": mapping.source_sha256,
            "observed_semantic_labels": observed,
            "hand_semantic_category": f"{request.hand}_hand",
        },
        "candidate_count": len(candidates),
        "candidates": [item.as_dict() for item in inspected],
        "selected": None if selected is None else selected.as_dict(),
        "selection_hash": stable_hash(None if selected is None else selected.as_dict()),
    }


__all__ = [
    "CONTACT_THRESHOLD_ASSUMPTION",
    "SOURCE_SANITY_ASSUMPTION",
    "ContactWindow",
    "resolve_index_sequence",
    "select_contact_windows",
]
