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
        start, stop = (frame_range or FrameRange()).resolve(len(frames))
        selected = frames[start:stop]
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
        )
        htm = np.linalg.inv(pose_json_wxyz(fit["mTc"]))
        base_vertices = np.asarray(base.vertices_scene[0], dtype=np.float64)
        base_wrist = np.asarray(base.wrist_pose_scene[0], dtype=np.float64)
        vertices: list[np.ndarray] = []
        wrists: list[np.ndarray] = []
        for frame in selected:
            if bool(hands[1].get("moving")) and "hTo" in frame:
                oth = np.linalg.inv(pose_json_wxyz(frame["hTo"][1]))
            else:
                oth = np.eye(4, dtype=np.float64)
            otm = oth @ htm
            vertices.append(transform_points(otm[None, ...], base_vertices[None, ...])[0])
            wrists.append(otm @ base_wrist)
        hand_vertices = np.stack(vertices, axis=0)
        hand_wrist = np.stack(wrists, axis=0)
        valid = np.ones(stop - start, dtype=bool)
        hand = make_hand(
            hand_id="right_hand",
            side="right",
            vertices_scene=hand_vertices,
            faces=base.faces,
            wrist_pose_scene=hand_wrist,
            valid=valid,
            mano_parameters=ManoParameterTrack(
                global_orient_aa=np.broadcast_to(
                    np.asarray(fit["pose"][:3], dtype=np.float64), (stop - start, 3)
                ).copy(),
                transl=np.zeros((stop - start, 3), dtype=np.float64),
                betas=np.broadcast_to(
                    np.asarray(fit["betas"], dtype=np.float64), (stop - start, 10)
                ).copy(),
                model_profile="contactpose_mano_fits_15_pca",
            ),
            mano_model_root=self.mano_model_root,
            metadata={
                "source": "ContactPose MANO fit transformed into object frame",
                "source_pca_pose": np.asarray(fit["pose"], dtype=np.float64).tolist(),
                "contact_annotation_available": False,
                "contact_benchmark_status": "NOT_AVAILABLE",
            },
        )
        object_vertices, object_faces = load_mesh(object_mesh_path)
        object_track = make_object(
            object_id=row["object_name"],
            vertices=object_vertices,
            faces=object_faces,
            poses_scene=identity_poses(stop - start),
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
            frame_count=stop - start,
            fps=30.0,
            source_file=annotation_path,
            source_hash=sha256_paths([annotation_path, fits_path, object_mesh_path]),
            adapter_name=self.adapter_name,
            coordinate_convention=(
                "ContactPose object scene; MANO fit and moving-hand hTo are "
                "composed into object coordinates"
            ),
            conversion_options={
                "selected_frame_range": [start, stop],
                "right_hand_index": 1,
                "contact_benchmark_status": "NOT_AVAILABLE",
                "official_contact_attribution": False,
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
