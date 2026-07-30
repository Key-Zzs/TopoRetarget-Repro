"""Lazy DexYCB adapter for the Stage 12 canonical HOI v2 lane."""

from __future__ import annotations

import csv
import json
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
    identity_poses,
    load_mesh,
    make_hand,
    make_object,
    render_mano_fullpose,
    sequence_metadata,
    sha256_paths,
)


class DexYCBAdapterV1(Stage12AdapterBase):
    """Read one DexYCB annotation sequence and one synchronized camera only."""

    adapter_name = "dexycb_stage12_adapter"
    descriptor = DatasetDescriptor(
        name="dexycb",
        version="dexycb_raw_annotation_v1",
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
            "raw_contract": "DexYCB per-camera labels_*.npz",
            "source_mode": "annotation_only",
            "rgb_depth_read": False,
            "contact_annotation_available": False,
        },
    )

    @property
    def dataset_dir(self) -> Path:
        return self.data_root / "DexYCB"

    @property
    def manifest_path(self) -> Path:
        return self.dataset_dir / "outputs" / "dataset_audit" / "manifest.csv"

    def _discover_rows(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise Stage12AdapterError(f"DexYCB manifest is missing: {self.manifest_path}")
        rows: list[dict[str, Any]] = []
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                if raw.get("status") != "dexycb_sequence_available":
                    continue
                item = str(raw.get("item", ""))
                sequence_path = item.removeprefix("dexycb:")
                serials = raw.get("camera_serials", "[]")
                try:
                    serial_list = json.loads(serials)
                except json.JSONDecodeError:
                    serial_list = [item.strip() for item in serials.strip("[]").split(",") if item]
                rows.append(
                    {
                        "dataset": "dexycb",
                        "sequence": sequence_path,
                        "native_sequence": str(raw.get("sequence", "")),
                        "item": item,
                        "source_path": str(raw.get("source_path", "")),
                        "num_frames_declared": int(raw.get("num_frames_declared") or 0),
                        "camera_serials": [str(item) for item in serial_list],
                        "object_name": str(raw.get("object_name", "")),
                        "object_id": str(raw.get("object_id", "")),
                        "object_mesh_path": str(raw.get("object_mesh_path", "")),
                        "mano_side": str(raw.get("mano_side", "")),
                        "meta_path": str(raw.get("meta_path", "")),
                    }
                )
        return rows

    def _row(self, sequence: str) -> dict[str, Any]:
        key = sequence.removeprefix("dexycb:")
        for row in self._discover_rows():
            if row["sequence"] == key or row["item"] == sequence or row["item"] == f"dexycb:{key}":
                return row
        raise Stage12AdapterError(f"DexYCB sequence is not indexed: {sequence!r}")

    def describe_sequence(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        del kwargs
        row = dict(self._row(sequence))
        row["lazy"] = True
        row["frames_loaded"] = 0
        row["rgb_depth_read"] = False
        return row

    def _resolve_sequence_dir(self, row: dict[str, Any]) -> Path:
        relative = Path(row["sequence"])
        path = self.dataset_dir / "data" / relative
        if not path.is_dir():
            raise Stage12AdapterError(f"DexYCB sequence directory is missing: {path}")
        return path

    def _choose_camera(self, sequence_dir: Path, serials: list[str], indices: range) -> str:
        scores: list[tuple[int, str]] = []
        for serial in serials:
            valid_count = 0
            for frame in indices:
                path = sequence_dir / serial / f"labels_{frame:06d}.npz"
                if not path.is_file():
                    continue
                with np.load(path) as data:
                    if "pose_m" not in data or "joint_3d" not in data:
                        continue
                    pose = np.asarray(data["pose_m"])[0]
                    # DexYCB's annotation-only payload may omit per-camera
                    # joint_3d while pose_m remains the valid MANO source.
                    # The latter is the canonical source of geometry here;
                    # joint_3d is retained only as source evidence.
                    if (
                        pose.shape == (51,)
                        and np.isfinite(pose).all()
                        and not np.allclose(pose, 0.0)
                    ):
                        valid_count += 1
            scores.append((valid_count, serial))
        if not scores or max(scores)[0] == 0:
            raise Stage12AdapterError(
                f"DexYCB has no valid MANO annotation in requested range; cameras={serials}"
            )
        return max(scores, key=lambda item: (item[0], item[1]))[1]

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        camera_serial: str | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        del kwargs
        row = self._row(sequence)
        sequence_dir = self._resolve_sequence_dir(row)
        meta_path = sequence_dir / "meta.yml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        full_count = int(meta.get("num_frames", row["num_frames_declared"]))
        start, stop = (frame_range or FrameRange()).resolve(full_count)
        indices = range(start, stop)
        serials = row["camera_serials"] or [str(item) for item in meta.get("serials", [])]
        if camera_serial is not None:
            if camera_serial not in serials:
                raise Stage12AdapterError(f"camera {camera_serial!r} is not listed for {sequence}")
            serial = camera_serial
        else:
            serial = self._choose_camera(sequence_dir, serials, indices)

        labels: list[dict[str, Any]] = []
        source_paths = [meta_path]
        for frame in indices:
            path = sequence_dir / serial / f"labels_{frame:06d}.npz"
            if not path.is_file():
                raise Stage12AdapterError(f"DexYCB label is missing: {path}")
            with np.load(path) as data:
                labels.append({key: np.asarray(data[key]) for key in data.files})
            source_paths.append(path)

        object_ids = [int(item) for item in meta.get("ycb_ids", [])]
        try:
            object_index = object_ids.index(int(row["object_id"]))
        except (ValueError, TypeError):
            raise Stage12AdapterError(
                f"DexYCB object id {row['object_id']!r} is not in meta ycb_ids={object_ids}"
            ) from None
        object_mesh_path = (
            self.dataset_dir / "data" / "models" / row["object_name"] / "textured_simple.obj"
        )
        if not object_mesh_path.is_file():
            object_mesh_path = Path(row["object_mesh_path"])
            if not object_mesh_path.is_file():
                raise Stage12AdapterError(f"DexYCB object mesh is missing: {object_mesh_path}")
        object_vertices, object_faces = load_mesh(object_mesh_path)
        source_paths.append(object_mesh_path)

        valid = np.zeros(stop - start, dtype=bool)
        pose_values = np.full((stop - start, 51), np.nan, dtype=np.float64)
        wrist = identity_poses(stop - start)
        source_vertices = np.full((stop - start, 778, 3), np.nan, dtype=np.float64)
        render_faces: np.ndarray | None = None
        valid_local: list[int] = []
        for local, item in enumerate(labels):
            pose = np.asarray(item.get("pose_m"), dtype=np.float64).reshape(-1)
            ok = pose.shape == (51,) and np.isfinite(pose).all() and not np.allclose(pose, 0.0)
            if ok:
                valid[local] = True
                pose_values[local] = pose
                valid_local.append(local)
        if not valid_local:
            raise Stage12AdapterError(f"DexYCB sequence {sequence} has no valid requested frames")
        render = render_mano_fullpose(
            pose_values[valid], side="right", mano_model_root=self.mano_model_root
        )
        render_faces = render.faces
        source_vertices[valid] = render.vertices_scene
        wrist[valid] = render.wrist_pose_scene

        object_poses = identity_poses(stop - start)
        object_valid = np.zeros(stop - start, dtype=bool)
        for local, item in enumerate(labels):
            pose_y = np.asarray(item.get("pose_y"), dtype=np.float64)
            if pose_y.shape == (len(object_ids), 3, 4) and np.isfinite(pose_y[object_index]).all():
                object_poses[local, :3, :4] = pose_y[object_index]
                object_valid[local] = valid[local]
        if not object_valid[valid].all():
            raise Stage12AdapterError(
                f"DexYCB target object pose is incomplete in {sequence} camera {serial}"
            )

        mano_parameters = ManoParameterTrack(
            global_orient_aa=pose_values[:, :3],
            hand_pose_aa=pose_values[:, 3:48],
            transl=pose_values[:, 48:51],
            model_profile="dexycb_pose_m_51_fullpose",
        )
        hand = make_hand(
            hand_id="right_hand",
            side="right",
            vertices_scene=source_vertices,
            faces=render_faces,
            wrist_pose_scene=wrist,
            valid=valid,
            mano_parameters=mano_parameters,
            mano_model_root=self.mano_model_root,
            metadata={
                "source": "DexYCB pose_m MANO reconstruction",
                "source_camera_serial": serial,
                "native_joint_3d_preserved_as_source_evidence": True,
            },
        )
        object_track = make_object(
            object_id=row["object_name"],
            vertices=object_vertices,
            faces=object_faces,
            poses_scene=object_poses,
            valid=object_valid,
            mesh_hash=sha256_paths([object_mesh_path]),
            metadata={"role": "primary_manipulation_object", "ycb_id": int(row["object_id"])},
        )
        source_hash = sha256_paths(source_paths)
        metadata = sequence_metadata(
            dataset="dexycb",
            sequence_id=row["sequence"],
            frame_count=stop - start,
            fps=30.0,
            source_file=meta_path,
            source_hash=source_hash,
            adapter_name=self.adapter_name,
            coordinate_convention=(
                "selected DexYCB camera scene; pose_m and pose_y are camera-frame annotations"
            ),
            conversion_options={
                "selected_frame_range": [start, stop],
                "selected_camera_serial": serial,
                "object_id": int(row["object_id"]),
                "raw_rgb_depth_read": False,
            },
            metadata={
                "subject": sequence_dir.parent.name,
                "source_frames": [start, stop],
                "source_camera_serial": serial,
                "object_name": row["object_name"],
                "contact_annotation_available": False,
                "invalid_source_frames": np.flatnonzero(~valid).tolist(),
            },
        )
        result = HOISequence(metadata=metadata, hands=[hand], rigid_objects=[object_track])
        result.validate()
        return result


__all__ = ["DexYCBAdapterV1"]
