"""Mapping consistency metrics and machine-readable validation reports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence, KeypointTrack
from toporetarget.geometry.se3 import scene_to_wrist, wrist_to_scene
from toporetarget.keypoints.profiles import MappingProfile
from toporetarget.keypoints.registry import get_layout


def _rmse_max(error: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(error, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return None, None
    return float(np.sqrt(np.mean(values**2))), float(np.max(values))


def _same_array(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return first is second
    return bool(np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True))


def _track_valid(track: KeypointTrack) -> np.ndarray:
    if track.valid is None:
        valid = np.ones(track.positions_scene.shape[:2], dtype=bool)
    elif track.valid.shape == track.positions_scene.shape[:1]:
        valid = np.broadcast_to(track.valid[:, None], track.positions_scene.shape[:2]).copy()
    else:
        valid = np.asarray(track.valid, dtype=bool).copy()
    return valid & np.isfinite(track.positions_scene).all(axis=-1)


@dataclass
class MappingConsistencyReport:
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.metrics)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    def write_csv(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("metric", "value"))
            for key, value in sorted(self.metrics.items()):
                writer.writerow(
                    (
                        key,
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value,
                    )
                )
        return destination


def _source_track(
    sequence: HOISequence, profile: MappingProfile, hand_id: str
) -> tuple[KeypointTrack | None, str | None]:
    hand = sequence.hand(hand_id)
    if profile.source_joint_layout in hand.keypoint_tracks:
        return hand.keypoint_tracks[profile.source_joint_layout], profile.source_joint_layout
    layout = get_layout(profile.source_joint_layout)
    for alias in layout.aliases:
        if alias in hand.keypoint_tracks:
            return hand.keypoint_tracks[alias], alias
    return None, None


def validate_mapping(
    original: HOISequence,
    converted: HOISequence,
    *,
    hand_id: str,
    profile: MappingProfile,
    roundtrip_tolerance_m: float = 1e-12,
) -> MappingConsistencyReport:
    """Measure implementation consistency; these metrics are not detector accuracy."""

    source_hand = original.hand(hand_id)
    target_hand = converted.hand(hand_id)
    target = target_hand.keypoint_tracks[profile.target_layout]
    target_layout = get_layout(profile.target_layout)
    target_valid = _track_valid(target)
    source_track, source_layout_name = _source_track(original, profile, hand_id)
    non_tip_errors: list[np.ndarray] = []
    non_tip_masks: list[np.ndarray] = []
    if source_track is not None:
        source_names = list(
            source_track.semantic_names or get_layout(profile.source_joint_layout).semantic_names
        )
        source_indices = {name: index for index, name in enumerate(source_names)}
        source_valid = _track_valid(source_track)
        for target_name, source_name in profile.joint_mapping.items():
            target_index = target_layout.index_by_name[target_name]
            source_index = source_indices[source_name]
            mask = target_valid[:, target_index] & source_valid[:, source_index]
            non_tip_errors.append(
                np.linalg.norm(
                    target.positions_scene[:, target_index]
                    - source_track.positions_scene[:, source_index],
                    axis=-1,
                )
            )
            non_tip_masks.append(mask)
    if non_tip_errors:
        joint_error = np.concatenate(non_tip_errors)
        joint_mask = np.concatenate(non_tip_masks)
    else:
        joint_error = np.empty(0)
        joint_mask = np.empty(0, dtype=bool)

    tip_errors: list[np.ndarray] = []
    tip_masks: list[np.ndarray] = []
    if source_hand.vertices_scene is not None:
        for target_name, anchor in profile.fingertip_mapping.items():
            target_index = target_layout.index_by_name[target_name]
            mask = target_valid[:, target_index] & np.isfinite(
                source_hand.vertices_scene[:, anchor.vertex_index]
            ).all(axis=-1)
            tip_errors.append(
                np.linalg.norm(
                    target.positions_scene[:, target_index]
                    - source_hand.vertices_scene[:, anchor.vertex_index],
                    axis=-1,
                )
            )
            tip_masks.append(mask)
    if tip_errors:
        tip_error = np.concatenate(tip_errors)
        tip_mask = np.concatenate(tip_masks)
    else:
        tip_error = np.empty(0)
        tip_mask = np.empty(0, dtype=bool)

    wrist_pose = target_hand.wrist_pose_scene.pose_scene
    wrist_points = scene_to_wrist(wrist_pose, target.positions_scene)
    reconstructed = wrist_to_scene(wrist_pose, wrist_points)
    roundtrip_mask = target_valid
    roundtrip_error = np.linalg.norm(reconstructed - target.positions_scene, axis=-1)
    roundtrip_rmse, roundtrip_max = _rmse_max(roundtrip_error, roundtrip_mask)
    wrist_origin = np.linalg.norm(wrist_points[:, target_layout.wrist_index], axis=-1)
    wrist_origin_rmse, wrist_origin_max = _rmse_max(
        wrist_origin, target_valid[:, target_layout.wrist_index]
    )

    bone_lengths: list[np.ndarray] = []
    bone_masks: list[np.ndarray] = []
    for parent, child in target_layout.edges:
        bone_lengths.append(
            np.linalg.norm(
                target.positions_scene[:, child] - target.positions_scene[:, parent], axis=-1
            )
        )
        bone_masks.append(target_valid[:, child] & target_valid[:, parent])
    all_bones = np.concatenate(bone_lengths) if bone_lengths else np.empty(0)
    bone_mask = np.concatenate(bone_masks) if bone_masks else np.empty(0, dtype=bool)
    finite_bones = bool(np.isfinite(all_bones[bone_mask]).all()) if bone_mask.any() else True
    zero_count = (
        int(np.count_nonzero(all_bones[bone_mask] <= roundtrip_tolerance_m))
        if bone_mask.any()
        else 0
    )

    source_tracks_preserved = set(original.hand(hand_id).keypoint_tracks).issubset(
        target_hand.keypoint_tracks
    )
    if source_tracks_preserved:
        for name, track in original.hand(hand_id).keypoint_tracks.items():
            output_track = target_hand.keypoint_tracks[name]
            source_tracks_preserved &= (
                output_track.layout_name == track.layout_name
                and _same_array(output_track.positions_scene, track.positions_scene)
                and _same_array(output_track.valid, track.valid)
                and output_track.semantic_names == track.semantic_names
            )
    objects_unchanged = len(original.rigid_objects) == len(converted.rigid_objects)
    object_mesh_unchanged = objects_unchanged
    object_pose_unchanged = objects_unchanged
    if objects_unchanged:
        for before, after in zip(original.rigid_objects, converted.rigid_objects, strict=True):
            object_mesh_unchanged &= (
                before.object_id == after.object_id
                and _same_array(before.mesh.vertices_local, after.mesh.vertices_local)
                and _same_array(before.mesh.faces, after.mesh.faces)
            )
            object_pose_unchanged &= before.object_id == after.object_id and _same_array(
                before.pose_scene.pose_scene, after.pose_scene.pose_scene
            )

    joint_rmse, joint_max = _rmse_max(joint_error, joint_mask)
    tip_rmse, tip_max = _rmse_max(tip_error, tip_mask)
    timestamps_equal = _same_array(original.timestamps, converted.timestamps)
    source_meta = converted.metadata.provenance.conversion_options.get("mano_to_mediapipe21", {})
    provenance = target.provenance
    metrics: dict[str, Any] = {
        "non_tip_joint_copy_rmse_m": joint_rmse,
        "non_tip_joint_copy_max_m": joint_max,
        "fingertip_anchor_rmse_m": tip_rmse,
        "fingertip_anchor_max_m": tip_max,
        "scene_wrist_roundtrip_rmse_m": roundtrip_rmse,
        "scene_wrist_roundtrip_max_m": roundtrip_max,
        "wrist_origin_error_m": wrist_origin_max,
        "timestamp_max_abs_error_s": float(
            np.max(np.abs(original.timestamps - converted.timestamps))
        )
        if timestamps_equal
        else float(np.max(np.abs(original.timestamps - converted.timestamps))),
        "frame_count_match": original.num_frames == converted.num_frames,
        "native_fps_match": original.metadata.native_fps == converted.metadata.native_fps,
        "object_pose_unchanged": object_pose_unchanged,
        "object_mesh_unchanged": object_mesh_unchanged,
        "wrist_pose_unchanged": _same_array(
            source_hand.wrist_pose_scene.pose_scene, target_hand.wrist_pose_scene.pose_scene
        ),
        "source_tracks_preserved": source_tracks_preserved,
        "output_layout_valid": target.layout_name == profile.target_layout
        and target.positions_scene.shape == (converted.num_frames, 21, 3)
        and target.semantic_names == list(target_layout.semantic_names),
        "all_bone_lengths_finite": finite_bones,
        "zero_length_bone_count": zero_count,
        "profile_id": provenance.get("mapping_profile_id", profile.profile_id),
        "profile_hash": provenance.get("mapping_profile_hash", profile.sha256),
        "model_hash": provenance.get("mano_model_hash"),
        "source_layout": provenance.get("source_layout", source_layout_name),
        "mapping_profile_metadata": source_meta,
        "target_frame": target.frame_name,
        "target_units": target.units,
        "valid_point_count": int(target_valid.sum()),
    }
    return MappingConsistencyReport(metrics)


__all__ = ["MappingConsistencyReport", "validate_mapping"]
