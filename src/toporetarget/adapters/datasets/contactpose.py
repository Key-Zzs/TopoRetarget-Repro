"""Lazy ContactPose adapter with explicit contact-attribution gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.contracts.dataset import DatasetCapabilities, DatasetDescriptor
from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.schema import HOISequence, ManoParameterTrack

from .stage12_base import (
    Stage12AdapterBase,
    Stage12AdapterError,
    contactpose_annotation_mano21_track,
    identity_poses,
    load_mesh,
    make_hand,
    make_object,
    pose_json_wxyz,
    render_mano_pca,
    sequence_metadata,
    sha256_paths,
    transform_points,
)


def static_mano_to_object_transform(fit: dict[str, Any]) -> np.ndarray:
    """Return the static ContactPose MANO-to-object transform.

    ``hTo`` belongs to the RGB-D rigid-observation stream. It must not be
    composed into a one-frame MANO fit: doing so places every source joint in
    a second, unrelated hand frame and produces an apparent finger error.
    """

    return np.linalg.inv(pose_json_wxyz(fit["mTc"]))


class ContactPoseAdapterV1(Stage12AdapterBase):
    """Load one ContactPose object in object coordinates.

    The raw ContactPose object PLY is retained, but Eq. (10)/(11) contact
    attribution is intentionally not synthesized: the current canonical
    contract has no verified hand-bone attribution source for this adapter.
    """

    adapter_name = "contactpose_stage12_adapter"
    descriptor = DatasetDescriptor(
        name="contactpose",
        version="contactpose_raw_v1",
        capabilities=DatasetCapabilities(
            canonical_hoi=True,
            contact_annotation=False,
            articulated_object=False,
            bimanual=True,
            body_model=True,
            rgb=True,
            depth=True,
        ),
        provenance={
            "raw_contract": "ContactPose annotations.json + mano_fits_15.json + object PLY",
            "contact_annotation_available": False,
            "contact_benchmark_status": "NOT_AVAILABLE",
            "contact_ply_preserved_as_raw_evidence": True,
        },
    )

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "ContactPose" / "data" / "contactpose_data"

    def _discover_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for session in sorted(self.dataset_dir.glob("full*_use")):
            for object_dir in sorted(session.iterdir()):
                if not object_dir.is_dir() or object_dir.name in {"hands", "palm_print"}:
                    continue
                annotation_path = object_dir / "annotations.json"
                fits_path = object_dir / "mano_fits_15.json"
                mesh_path = object_dir / f"{object_dir.name}.ply"
                if (
                    not annotation_path.is_file()
                    or not fits_path.is_file()
                    or not mesh_path.is_file()
                ):
                    continue
                try:
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                    frame_count = len(annotation.get("frames", []))
                except (OSError, json.JSONDecodeError):
                    continue
                rows.append(
                    {
                        "dataset": "contactpose",
                        "sequence": f"{session.name}/{object_dir.name}",
                        "p_num": int(session.name.removeprefix("full").removesuffix("_use")),
                        "intent": "use",
                        "object_name": object_dir.name,
                        "num_frames": frame_count,
                        "source_path": str(object_dir),
                        "contact_annotation_available": False,
                        "contact_benchmark_status": "NOT_AVAILABLE",
                    }
                )
        if not rows:
            raise Stage12AdapterError(
                f"ContactPose extracted payload is missing: {self.dataset_dir}"
            )
        return rows

    def _row(self, sequence: str) -> dict[str, Any]:
        key = sequence.removeprefix("contactpose:")
        for row in self._discover_rows():
            if row["sequence"] == key:
                return row
        raise Stage12AdapterError(f"ContactPose sequence is not indexed: {sequence!r}")

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        del kwargs
        row = dict(self._row(sequence))
        row.update({"lazy": True, "frames_loaded": 0})
        return row

    @staticmethod
    def _native_timestamps(frames: list[dict[str, Any]]) -> np.ndarray | None:
        values: list[float] = []
        for frame in frames:
            clocks = frame.get("time", {})
            clock = clocks.get("kinect2_middle") or next(iter(clocks.values()), None)
            if not isinstance(clock, dict) or "sec" not in clock or "nsec" not in clock:
                return None
            values.append(float(clock["sec"]) + float(clock["nsec"]) * 1e-9)
        if len(values) < 2 or not np.all(np.diff(values) > 0):
            return None
        return np.asarray(values, dtype=np.float64) - values[0]

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        del kwargs
        row = self._row(sequence)
        sequence_dir = self.dataset_dir / row["sequence"]
        annotation_path = sequence_dir / "annotations.json"
        fits_path = sequence_dir / "mano_fits_15.json"
        object_mesh_path = sequence_dir / f"{row['object_name']}.ply"
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        fits = json.loads(fits_path.read_text(encoding="utf-8"))
        frames = list(annotation.get("frames", []))
        requested_start, requested_stop = (frame_range or FrameRange()).resolve(len(frames))
        # A ContactPose entry is one fitted hand articulation.  RGB-D frames
        # may show rigid observation motion, but never form a MANO trajectory.
        start = requested_start
        selected = [frames[start]]
        hands = list(annotation.get("hands", []))
        if len(hands) < 2 or not bool(hands[1].get("valid")):
            raise Stage12AdapterError(
                f"ContactPose selected trajectory has no valid right hand: {sequence}"
            )
        if len(fits) < 2 or not bool(fits[1].get("valid")):
            raise Stage12AdapterError(f"ContactPose has no valid right MANO fit: {fits_path}")
        fit = fits[1]
        base = render_mano_pca(
            np.asarray(fit["pose"], dtype=np.float64),
            side="right",
            mano_model_root=self.mano_model_root,
            betas=np.asarray(fit["betas"], dtype=np.float64),
            dataset_name="contactpose",
            source_annotation_path=fits_path,
            source_annotation_hash=sha256_paths([annotation_path, fits_path, object_mesh_path]),
        )
        htm = static_mano_to_object_transform(fit)
        base_vertices = np.asarray(base.vertices[0], dtype=np.float64)
        base_wrist = np.asarray(base.wrist_pose_scene[0], dtype=np.float64)
        vertices: list[np.ndarray] = []
        wrists: list[np.ndarray] = []
        for _frame in selected:
            otm = htm
            vertices.append(transform_points(otm[None, ...], base_vertices[None, ...])[0])
            wrists.append(otm @ base_wrist)
        hand_vertices = np.stack(vertices, axis=0)
        hand_wrist = np.stack(wrists, axis=0)
        valid = np.ones(1, dtype=bool)
        raw_joints = np.asarray(hands[1].get("joints"), dtype=np.float64)
        if raw_joints.shape != (21, 3):
            raise Stage12AdapterError(
                "ContactPose right-hand annotation must provide exactly 21 OpenPose joints, "
                f"got {raw_joints.shape} in {annotation_path}"
            )
        official_joints = contactpose_annotation_mano21_track(
            raw_joints[None, ...], valid=valid, source_path=str(annotation_path)
        )
        hand = make_hand(
            hand_id="right_hand",
            side="right",
            vertices_scene=hand_vertices,
            faces=base.faces,
            wrist_pose_scene=hand_wrist,
            valid=valid,
            mano_parameters=ManoParameterTrack(
                global_orient_aa=base.global_orient_axis_angle,
                hand_pose_aa=base.hand_pose_axis_angle,
                transl=base.translation,
                betas=base.betas,
                model_profile="contactpose_mano_fits_15_pca_explicit_contract_v2",
            ),
            mano_model_root=self.mano_model_root,
            metadata={
                "source": "ContactPose fitted MANO mesh plus annotated OpenPose-21 keypoints",
                "source_pca_pose": np.asarray(fit["pose"], dtype=np.float64).tolist(),
                "mano_representation": "pca",
                "num_pca_components": 15,
                "flat_hand_mean": False,
                "mano_reconstruction": base.reconstruction_manifest,
                "joint_source": "contactpose_annotation_openpose21",
                "joint_mesh_contract": (
                    "official annotated OpenPose-21 joints drive retargeting; "
                    "fitted PCA15 MANO mesh remains the source visualization surface"
                ),
                "wrist_transform_source": "oTm=inv(mTc); hTo retained as observation evidence only",
                "contact_annotation_available": False,
                "contact_benchmark_status": "NOT_AVAILABLE",
            },
            native_joint_track=official_joints,
        )
        object_vertices, object_faces = load_mesh(object_mesh_path)
        object_track = make_object(
            object_id=row["object_name"],
            vertices=object_vertices,
            faces=object_faces,
            poses_scene=identity_poses(1),
            valid=valid,
            mesh_hash=sha256_paths([object_mesh_path]),
            metadata={
                "role": "primary_manipulation_object",
                "object_coordinate_scene": True,
                "raw_contact_map_path": str(object_mesh_path),
            },
        )
        metadata = sequence_metadata(
            dataset="contactpose",
            sequence_id=row["sequence"],
            frame_count=1,
            fps=30.0,
            source_file=annotation_path,
            source_hash=sha256_paths([annotation_path, fits_path, object_mesh_path]),
            adapter_name=self.adapter_name,
            coordinate_convention=(
                "ContactPose object scene; static MANO fit uses inv(mTc), while "
                "moving-hand hTo remains rigid-observation evidence only"
            ),
            conversion_options={
                "requested_observation_frame_range": [requested_start, requested_stop],
                "selected_static_observation_frame": start,
                "right_hand_index": 1,
                "contact_benchmark_status": "NOT_AVAILABLE",
                "official_contact_attribution": False,
                "sample_type": "static_contact_evaluation_only",
                "articulated_frame_count": 1,
                "temporal_metrics_applicable": False,
                "rigid_observation_sequence_available": bool(hands[1].get("moving")),
                "hTo_applied_to_static_mano": False,
            },
            metadata={
                "participant": row["p_num"],
                "intent": row["intent"],
                "object_name": row["object_name"],
                "contact_annotation_available": False,
                "contact_benchmark_status": "NOT_AVAILABLE",
                "contact_attribution_blocker": (
                    "raw object contact-map PLY has no verified hand-bone "
                    "attribution in this adapter contract"
                ),
                "classification": "static_contact_evaluation_only",
                "frame_count": 1,
                "articulated_frame_count": 1,
                "articulated_motion": False,
                "temporal_metrics": "NOT_APPLICABLE",
                "rigid_observation_sequence_available": bool(hands[1].get("moving")),
                "hTo_applied_to_static_mano": False,
                "repeated_pose_manufacturing": False,
                "contactpose_official_tip_vertex_ids": {
                    "index": 333,
                    "middle": 444,
                    "pinky": 672,
                    "ring": 555,
                    "thumb": 745,
                },
                "contactpose_keypoint_contract": {
                    "source": "annotations.hands[1].joints",
                    "layout": "contactpose_openpose21",
                    "semantic_route": "identity_to_mediapipe21",
                    "fitted_mano_used_for_keypoints": False,
                    "fitted_mano_mesh_retained_for_visualization": True,
                },
            },
        )
        timestamps = self._native_timestamps(selected)
        if timestamps is not None:
            metadata.timestamps = timestamps
            metadata.num_frames = len(timestamps)
        result = HOISequence(metadata=metadata, hands=[hand], rigid_objects=[object_track])
        result.validate()
        return result


__all__ = ["ContactPoseAdapterV1"]
