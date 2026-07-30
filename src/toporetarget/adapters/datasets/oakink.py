"""Lazy OakInk-Image adapter for Stage 12."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.contracts.dataset import DatasetCapabilities, DatasetDescriptor
from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.schema import HOISequence

from .stage12_base import (
    Stage12AdapterBase,
    Stage12AdapterError,
    identity_poses,
    load_mesh,
    load_pickle,
    make_hand,
    make_object,
    sequence_metadata,
    sha256_paths,
)


class OakInkAdapterV1(Stage12AdapterBase):
    """Read OakInk image annotations, excluding RGB streams and shape caches."""

    adapter_name = "oakink_stage12_adapter"
    descriptor = DatasetDescriptor(
        name="oakink",
        version="oakink_image_annotation_v1",
        capabilities=DatasetCapabilities(
            canonical_hoi=True,
            contact_annotation=False,
            articulated_object=False,
            bimanual=False,
            body_model=True,
            rgb=False,
            depth=False,
        ),
        provenance={
            "raw_contract": "OakInk-Image image/anno/hand_v + obj_transf",
            "source_mode": "annotation_only",
            "rgb_depth_read": False,
            "contact_annotation_available": False,
        },
    )

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "OakInk"

    @property
    def annotation_dir(self) -> Path:
        return self.dataset_dir / "downloads" / "image" / "anno"

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "outputs" / "dataset_audit" / "manifest.csv"

    def _discover_rows(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise Stage12AdapterError(f"OakInk manifest is missing: {self.manifest_path}")
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("status") != "image_sequence_available":
                    continue
                rows.append(
                    {
                        "dataset": "oakink",
                        "item": str(raw.get("item", "")),
                        "sequence": str(raw.get("sequence", "")),
                        "object_name": str(raw.get("object_name", "")),
                        "view_id": int(raw.get("view_id") or 0),
                        "num_frames": int(raw.get("num_frames") or 0),
                        "source_path": str(raw.get("source_path", "")),
                    }
                )
        return rows

    @staticmethod
    def _parse_item(sequence: str) -> tuple[str, int]:
        body = sequence.removeprefix("image_sequence:")
        seq_id, separator, view = body.rpartition(":view")
        if not separator or not seq_id:
            raise Stage12AdapterError("OakInk sequence must be image_sequence:<seq_id>:view<id>")
        try:
            return seq_id, int(view)
        except ValueError as exc:
            raise Stage12AdapterError(f"invalid OakInk view in {sequence!r}") from exc

    def _row(self, sequence: str) -> dict[str, Any]:
        seq_id, view = self._parse_item(sequence)
        for row in self._discover_rows():
            if row["sequence"] == seq_id and row["view_id"] == view:
                return row
            if row["item"] == sequence:
                return row
        raise Stage12AdapterError(f"OakInk sequence is not indexed: {sequence!r}")

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        del kwargs
        row = dict(self._row(sequence))
        row.update({"lazy": True, "frames_loaded": 0, "rgb_depth_read": False})
        return row

    def _seq_infos(self, seq_id: str, view_id: int) -> list[list[Any]]:
        path = self.annotation_dir / "seq_all.json"
        if not path.is_file():
            raise Stage12AdapterError(f"OakInk seq_all index is missing: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        infos = [
            list(item)
            for item in values
            if isinstance(item, list)
            and len(item) >= 4
            and str(item[0]) == seq_id
            and int(item[3]) == view_id
        ]
        infos.sort(key=lambda item: (int(item[1]), int(item[2])))
        if not infos:
            raise Stage12AdapterError(f"OakInk has no indexed frames for {seq_id}:view{view_id}")
        return infos

    @staticmethod
    def _info_name(info: list[Any]) -> str:
        return "__".join(str(value) for value in info).replace("/", "__")

    def _object_mesh_path(self, object_name: str) -> Path:
        root = self.dataset_dir / "downloads" / "image" / "obj"
        for suffix in (".obj", ".ply"):
            path = root / f"{object_name}{suffix}"
            if path.is_file():
                return path
        raise Stage12AdapterError(f"OakInk object mesh is missing for {object_name}: {root}")

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        del kwargs
        row = self._row(sequence)
        seq_id, view_id = self._parse_item(row["item"] or sequence)
        infos = self._seq_infos(seq_id, view_id)
        start, stop = (frame_range or FrameRange()).resolve(len(infos))
        selected = infos[start:stop]
        object_mesh_path = self._object_mesh_path(row["object_name"])
        object_vertices, object_faces = load_mesh(object_mesh_path)
        hand_frames: list[np.ndarray] = []
        object_poses: list[np.ndarray] = []
        source_paths = [self.annotation_dir / "seq_all.json", object_mesh_path]
        hand_dir = self.annotation_dir / "hand_v"
        object_transform_dir = self.annotation_dir / "obj_transf"
        for info in selected:
            name = self._info_name(info)
            hand_path = hand_dir / f"{name}.pkl"
            object_path = object_transform_dir / f"{name}.pkl"
            if not hand_path.is_file() or not object_path.is_file():
                raise Stage12AdapterError(
                    "OakInk frame annotations are incomplete: "
                    f"hand={hand_path}, object={object_path}"
                )
            hand_vertices = np.asarray(load_pickle(hand_path), dtype=np.float64)
            if hand_vertices.shape != (778, 3) or not np.isfinite(hand_vertices).all():
                raise Stage12AdapterError(f"OakInk hand_v must be finite [778,3]: {hand_path}")
            transform = np.asarray(load_pickle(object_path), dtype=np.float64)
            if transform.shape == (3, 4):
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3] = transform
                transform = matrix
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise Stage12AdapterError(f"OakInk obj_transf must be finite [4,4]: {object_path}")
            hand_frames.append(hand_vertices)
            object_poses.append(transform)
            source_paths.extend([hand_path, object_path])
        vertices = np.stack(hand_frames, axis=0)
        poses = np.stack(object_poses, axis=0)
        valid = np.ones(stop - start, dtype=bool)
        hand = make_hand(
            hand_id="right_hand",
            side="right",
            vertices_scene=vertices,
            faces=load_mano_faces(self.mano_model_root),
            wrist_pose_scene=identity_poses(stop - start),
            valid=valid,
            mano_parameters=None,
            mano_model_root=self.mano_model_root,
            metadata={
                "source": "OakInk-Image hand_v",
                "source_joint_annotation": (
                    "hand_j available but canonicalized through MANO geometry"
                ),
                "view_id": view_id,
            },
        )
        object_track = make_object(
            object_id=row["object_name"],
            vertices=object_vertices,
            faces=object_faces,
            poses_scene=poses,
            valid=valid,
            mesh_hash=sha256_paths([object_mesh_path]),
            metadata={"role": "primary_manipulation_object", "object_name": row["object_name"]},
        )
        metadata = sequence_metadata(
            dataset="oakink",
            sequence_id=seq_id,
            frame_count=stop - start,
            fps=30.0,
            source_file=self.annotation_dir / "seq_all.json",
            source_hash=sha256_paths(source_paths),
            adapter_name=self.adapter_name,
            coordinate_convention=(
                "OakInk camera scene; hand_v and obj_transf are T_c_o/hand camera annotations"
            ),
            conversion_options={
                "selected_frame_range": [start, stop],
                "view_id": view_id,
                "object_name": row["object_name"],
                "rgb_depth_read": False,
            },
            metadata={
                "object_name": row["object_name"],
                "view_id": view_id,
                "source_frame_indices": [int(info[2]) for info in selected],
                "contact_annotation_available": False,
            },
        )
        result = HOISequence(metadata=metadata, hands=[hand], rigid_objects=[object_track])
        result.validate()
        return result


def load_mano_faces(model_root: Path) -> np.ndarray:
    """Return MANO topology without importing the SMPL-X runtime."""

    from toporetarget.keypoints.mano_to_mediapipe import load_mano_model_geometry

    faces = load_mano_model_geometry(model_root, side="right", expected_vertex_count=778).faces
    if faces is None:
        raise Stage12AdapterError("MANO_RIGHT.pkl has no faces")
    return np.asarray(faces, dtype=np.int64)


__all__ = ["OakInkAdapterV1"]
