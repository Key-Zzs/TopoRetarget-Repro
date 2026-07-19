"""Deterministic, robot-free HOI data for tests and local smoke checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange, HOIDatasetAdapter
from toporetarget.data.schema import (
    HandTrack,
    HOISequence,
    KeypointTrack,
    MeshDefinition,
    PoseTrack,
    ProvenanceRecord,
    RigidObjectTrack,
    SequenceMetadata,
)
from toporetarget.geometry.se3 import wrist_to_scene


def _pose_track(num_frames: int, *, x_offset: float = 0.0, z_angle_step: float = 0.0) -> PoseTrack:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, ...], num_frames, axis=0)
    poses[:, 0, 3] = x_offset + np.arange(num_frames, dtype=np.float64) * 0.002
    angles = np.arange(num_frames, dtype=np.float64) * z_angle_step
    poses[:, 0, 0] = np.cos(angles)
    poses[:, 0, 1] = -np.sin(angles)
    poses[:, 1, 0] = np.sin(angles)
    poses[:, 1, 1] = np.cos(angles)
    return PoseTrack(poses, frame_name="S", child_frame_name="H")


def make_synthetic_sequence(
    *,
    sequence_id: str = "demo",
    num_frames: int = 8,
    native_fps: float = 30.0,
    irregular_timestamps: bool = False,
) -> HOISequence:
    """Build a small deterministic sequence without touching the filesystem."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if irregular_timestamps:
        deltas = np.linspace(1.0 / native_fps, 1.5 / native_fps, num_frames)
        timestamps = np.cumsum(deltas) - deltas[0]
    else:
        timestamps = np.arange(num_frames, dtype=np.float64) / native_fps

    wrist = _pose_track(num_frames, x_offset=0.15, z_angle_step=0.01)
    hand_local = np.array(
        [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.04, 0.03, 0.0], [0.0, 0.03, 0.01]],
        dtype=np.float64,
    )
    hand_mesh = MeshDefinition(
        hand_local,
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        mesh_frame_name="H_R",
        mesh_id="synthetic_hand",
    )
    hand_vertices = wrist_to_scene(
        wrist.pose_scene, np.broadcast_to(hand_local, (num_frames, 4, 3))
    )
    keypoint_positions = hand_vertices[:, :3, :]
    hand = HandTrack(
        hand_id="hand_r",
        side="right",
        wrist_pose_scene=wrist,
        valid=np.ones(num_frames, dtype=bool),
        keypoint_tracks={
            "dataset_native": KeypointTrack(
                keypoint_positions,
                layout_name="dataset_native",
                valid=np.ones((num_frames, 3), dtype=bool),
                semantic_names=["wrist", "index", "middle"],
            )
        },
        mesh=hand_mesh,
        vertices_scene=hand_vertices,
        metadata={"generator": "deterministic_synthetic", "hand_side_preserved": True},
    )

    object_mesh = MeshDefinition(
        np.array(
            [[-0.03, -0.03, 0.0], [0.03, -0.03, 0.0], [0.03, 0.03, 0.0], [-0.03, 0.03, 0.0]],
            dtype=np.float64,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        mesh_frame_name="O",
        mesh_id="synthetic_object",
    )
    object_pose = _pose_track(num_frames, x_offset=0.22, z_angle_step=-0.005)
    obj = RigidObjectTrack(
        object_id="object_0",
        mesh=object_mesh,
        pose_scene=PoseTrack(object_pose.pose_scene, frame_name="S", child_frame_name="O"),
        valid=np.ones(num_frames, dtype=bool),
        metadata={"object_name": "synthetic_square"},
    )
    metadata = SequenceMetadata(
        dataset_name="synthetic",
        sequence_id=sequence_id,
        native_fps=native_fps,
        timestamps=timestamps,
        source_frame_name="synthetic_source",
        scene_frame_name="S",
        provenance=ProvenanceRecord(
            source_dataset="synthetic",
            source_sequence=sequence_id,
            adapter_name="synthetic",
            adapter_version="1",
            source_coordinate_convention="synthetic right-handed scene frame",
            conversion_options={"irregular_timestamps": irregular_timestamps},
        ),
        metadata={"axis_semantics": "unknown", "units": "m", "angles": "rad", "time": "s"},
    )
    sequence = HOISequence(metadata=metadata, hands=[hand], rigid_objects=[obj])
    sequence.validate()
    return sequence


def _slice_pose(track: PoseTrack, start: int, end: int) -> PoseTrack:
    valid = np.ones(track.pose_scene.shape[0], dtype=bool) if track.valid is None else track.valid
    return PoseTrack(
        track.pose_scene[start:end],
        valid[start:end],
        track.frame_name,
        track.child_frame_name,
    )


class SyntheticAdapter(HOIDatasetAdapter):
    adapter_name = "synthetic"
    adapter_version = "1"

    def describe_sequence(self, sequence: str = "demo", **kwargs: Any) -> dict[str, Any]:
        item = make_synthetic_sequence(sequence_id=sequence, **kwargs)
        return {
            "dataset_name": item.metadata.dataset_name,
            "sequence_id": item.metadata.sequence_id,
            "num_frames": item.num_frames,
            "native_fps": item.metadata.native_fps,
            "hands": [hand.hand_id for hand in item.hands],
            "rigid_objects": [obj.object_id for obj in item.rigid_objects],
            "timestamps": item.timestamps.tolist(),
        }

    def load_sequence(
        self,
        sequence: str = "demo",
        *,
        frame_range: FrameRange | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        full = make_synthetic_sequence(sequence_id=sequence, **kwargs)
        if frame_range is None:
            return full
        start, end = frame_range.resolve(full.num_frames)
        metadata = replace(
            full.metadata,
            timestamps=full.timestamps[start:end].copy(),
            num_frames=end - start,
            provenance=replace(
                full.metadata.provenance,
                conversion_options={
                    **full.metadata.provenance.conversion_options,
                    "frame_range": [start, end],
                },
            ),
        )
        hands: list[HandTrack] = []
        for hand in full.hands:
            keypoints = {
                layout: KeypointTrack(
                    track.positions_scene[start:end].copy(),
                    layout_name=track.layout_name,
                    valid=None if track.valid is None else track.valid[start:end].copy(),
                    semantic_names=None
                    if track.semantic_names is None
                    else list(track.semantic_names),
                    confidence=None
                    if track.confidence is None
                    else track.confidence[start:end].copy(),
                )
                for layout, track in hand.keypoint_tracks.items()
            }
            hands.append(
                HandTrack(
                    hand_id=hand.hand_id,
                    side=hand.side,
                    wrist_pose_scene=_slice_pose(hand.wrist_pose_scene, start, end),
                    valid=None if hand.valid is None else hand.valid[start:end].copy(),
                    keypoint_tracks=keypoints,
                    mesh=hand.mesh,
                    vertices_scene=None
                    if hand.vertices_scene is None
                    else hand.vertices_scene[start:end].copy(),
                    mano_parameters=hand.mano_parameters,
                    metadata=dict(hand.metadata),
                )
            )
        objects = [
            RigidObjectTrack(
                object_id=obj.object_id,
                mesh=obj.mesh,
                pose_scene=_slice_pose(obj.pose_scene, start, end),
                valid=None if obj.valid is None else obj.valid[start:end].copy(),
                scale=obj.scale,
                metadata=dict(obj.metadata),
            )
            for obj in full.rigid_objects
        ]
        result = HOISequence(metadata=metadata, hands=hands, rigid_objects=objects)
        result.validate()
        return result

    def canonicalize(self, sequence: HOISequence, **kwargs: Any) -> HOISequence:
        sequence.validate()
        return sequence

    def supported_fields(self) -> tuple[str, ...]:
        return (
            "timestamps",
            "native_fps",
            "hand.wrist_pose_scene",
            "hand.vertices_scene",
            "hand.keypoint_tracks.dataset_native",
            "rigid_object.mesh",
            "rigid_object.pose_scene",
        )


__all__ = ["SyntheticAdapter", "make_synthetic_sequence"]
