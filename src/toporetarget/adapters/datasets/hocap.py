"""Lazy HO-Cap adapter for Stage 12."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.contracts.dataset import DatasetCapabilities, DatasetDescriptor
from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.schema import HOISequence, ManoParameterTrack

from .stage12_base import (
    Stage12AdapterBase,
    Stage12AdapterError,
    load_mesh,
    make_hand,
    make_object,
    pose_hocap_qxyzw,
    render_mano_fullpose,
    sequence_metadata,
    sha256_paths,
)


class HOCapAdapterV1(Stage12AdapterBase):
    """Load one HOCap subject/timestamp and its declared object parts."""

    adapter_name = "hocap_stage12_adapter"
    descriptor = DatasetDescriptor(
        name="hocap",
        version="hocap_extracted_v1",
        capabilities=DatasetCapabilities(
            canonical_hoi=True,
            contact_annotation=False,
            articulated_object=True,
            bimanual=True,
            body_model=True,
            rgb=False,
            depth=False,
        ),
        provenance={
            "raw_contract": "HOCap poses_m.npy + poses_o.npy + meta.yaml",
            "source_mode": "extracted_sequence_only",
            "rgb_depth_read": False,
            "contact_annotation_available": False,
        },
    )

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "HOCap"

    @property
    def sequence_root(self) -> Path:
        return self.dataset_dir / "data"

    def _discover_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for meta_path in sorted(self.sequence_root.glob("subject_*/*/meta.yaml")):
            values = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            sides = values.get("mano_sides") or values.get("hand_sides") or []
            if isinstance(sides, dict):
                sides = list(sides.values())
            object_ids = (
                values.get("object_ids") or values.get("obj_ids") or values.get("objects") or []
            )
            rows.append(
                {
                    "dataset": "hocap",
                    "sequence": str(meta_path.parent.relative_to(self.sequence_root)),
                    "source_path": str(meta_path),
                    "num_frames": int(values.get("num_frames") or 0),
                    "mano_sides": [str(item).lower() for item in sides],
                    "object_ids": [str(item) for item in object_ids],
                }
            )
        if not rows:
            raise Stage12AdapterError(
                f"HOCap extracted sequences are missing: {self.sequence_root}"
            )
        return rows

    def _row(self, sequence: str) -> dict[str, Any]:
        key = sequence.removeprefix("hocap:")
        for row in self._discover_rows():
            if row["sequence"] == key:
                return row
        raise Stage12AdapterError(f"HOCap sequence is not indexed: {sequence!r}")

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        del kwargs
        row = dict(self._row(sequence))
        row.update({"lazy": True, "frames_loaded": 0, "rgb_depth_read": False})
        return row

    @staticmethod
    def _object_ids(meta: dict[str, Any], count: int) -> list[str]:
        values = meta.get("object_ids") or meta.get("obj_ids") or meta.get("objects") or []
        if isinstance(values, dict):
            values = list(values.values())
        result = [str(item) for item in values]
        if len(result) != count:
            raise Stage12AdapterError(
                f"HOCap meta object count {len(result)} does not match poses_o count {count}"
            )
        return result

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        hand: str = "right",
        **kwargs: Any,
    ) -> HOISequence:
        del kwargs
        if hand != "right":
            raise Stage12AdapterError(
                "Stage 12 is frozen to wuji_hand2_beta1_rh; HOCap hand must be right"
            )
        row = self._row(sequence)
        sequence_dir = self.sequence_root / row["sequence"]
        meta_path = sequence_dir / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        poses_m_path = sequence_dir / "poses_m.npy"
        poses_o_path = sequence_dir / "poses_o.npy"
        poses_m = np.load(poses_m_path, mmap_mode="r").astype(np.float64)
        poses_o = np.load(poses_o_path, mmap_mode="r").astype(np.float64)
        if poses_m.ndim != 3 or poses_m.shape[2:] != (51,):
            raise Stage12AdapterError(
                f"HOCap poses_m must have shape [H,T,51], got {poses_m.shape}"
            )
        if poses_o.ndim != 3 or poses_o.shape[2:] != (7,):
            raise Stage12AdapterError(
                f"HOCap poses_o must have shape [O,T,7] or [T,O,7], got {poses_o.shape}"
            )
        # The extracted HOCap payload stores poses_o as [object,T,7].  The
        # official viewer transposes it to [T,object,7] before use.
        if poses_o.shape[1] == poses_m.shape[1] and poses_o.shape[0] != poses_m.shape[1]:
            poses_o = np.asarray(poses_o).transpose(1, 0, 2)
        full_count = min(int(poses_m.shape[1]), int(poses_o.shape[0]))
        start, stop = (frame_range or FrameRange()).resolve(full_count)
        sides = [str(item).lower() for item in (meta.get("mano_sides") or [])]
        if not sides:
            sides = ["right"]
        if "right" not in sides:
            raise Stage12AdapterError(f"HOCap sequence has no right hand: {row['sequence']}")
        hand_index = 0 if sides[0] == "right" else 1
        pose_values = np.asarray(poses_m[hand_index, start:stop], dtype=np.float64)
        valid = np.asarray(np.isfinite(pose_values).all(axis=1), dtype=bool)
        if not valid.all():
            raise Stage12AdapterError(
                "HOCap selected clip contains invalid right MANO frames: "
                f"{np.flatnonzero(~valid).tolist()}"
            )
        render = render_mano_fullpose(
            pose_values, side="right", mano_model_root=self.mano_model_root
        )
        mano_parameters = ManoParameterTrack(
            global_orient_aa=pose_values[:, :3],
            hand_pose_aa=pose_values[:, 3:48],
            transl=pose_values[:, 48:51],
            model_profile="hocap_poses_m_world_fullpose",
        )
        hand_track = make_hand(
            hand_id="right_hand",
            side="right",
            vertices_scene=render.vertices_scene,
            faces=render.faces,
            wrist_pose_scene=render.wrist_pose_scene,
            valid=valid,
            mano_parameters=mano_parameters,
            mano_model_root=self.mano_model_root,
            metadata={"source": "HOCap poses_m", "source_hand_index": hand_index},
        )
        object_ids = self._object_ids(meta, poses_o.shape[1])
        objects = []
        object_pose_values = np.asarray(poses_o[start:stop], dtype=np.float64)
        for object_index, object_id in enumerate(object_ids):
            mesh_path = self.dataset_dir / "data" / "models" / object_id / "textured_mesh.obj"
            if not mesh_path.is_file():
                raise Stage12AdapterError(f"HOCap object mesh is missing: {mesh_path}")
            vertices, faces = load_mesh(mesh_path)
            poses = np.stack(
                [pose_hocap_qxyzw(value) for value in object_pose_values[:, object_index]], axis=0
            )
            object_valid = np.asarray(
                np.isfinite(object_pose_values[:, object_index]).all(axis=1), dtype=bool
            )
            objects.append(
                make_object(
                    object_id=object_id,
                    vertices=vertices,
                    faces=faces,
                    poses_scene=poses,
                    valid=object_valid,
                    mesh_hash=sha256_paths([mesh_path]),
                    metadata={"role": "hocap_object_part", "object_index": object_index},
                )
            )
        source_paths = [meta_path, poses_m_path, poses_o_path]
        source_paths.extend(
            self.dataset_dir / "data" / "models" / object_id / "textured_mesh.obj"
            for object_id in object_ids
        )
        fps = float(meta.get("fps") or meta.get("frame_rate") or 30.0)
        metadata = sequence_metadata(
            dataset="hocap",
            sequence_id=row["sequence"],
            frame_count=stop - start,
            fps=fps,
            source_file=meta_path,
            source_hash=sha256_paths(source_paths),
            adapter_name=self.adapter_name,
            coordinate_convention=(
                "HOCap world scene; poses_m and poses_o use the extracted HOCap world convention"
            ),
            conversion_options={
                "selected_frame_range": [start, stop],
                "selected_hand": "right",
                "object_ids": object_ids,
                "rgb_depth_read": False,
            },
            metadata={
                "object_ids": object_ids,
                "source_hand_index": hand_index,
                "contact_annotation_available": False,
                "articulated_object_parts": True,
            },
        )
        result = HOISequence(metadata=metadata, hands=[hand_track], rigid_objects=objects)
        result.validate()
        return result


__all__ = ["HOCapAdapterV1"]
