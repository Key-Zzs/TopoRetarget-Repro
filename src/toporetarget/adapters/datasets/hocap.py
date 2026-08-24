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
    backend_posed_joint_track,
    load_mesh,
    make_hand,
    make_object,
    pose_hocap_qxyzw,
    render_mano_pca45,
    sequence_metadata,
    sha256_paths,
)


def hocap_mano_storage_index(side: str) -> int:
    """Return the fixed HOCap poses_m storage slot for an official side."""

    normalized = side.lower()
    if normalized not in {"left", "right"}:
        raise Stage12AdapterError(f"HOCap MANO side is invalid: {side!r}")
    return 0 if normalized == "right" else 1


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

    @staticmethod
    def _primary_object_id(object_ids: list[str], requested: str | None) -> str | None:
        """Validate the frozen selection's semantic target, if supplied."""

        if requested is None:
            return None
        primary = str(requested)
        if primary not in object_ids:
            raise Stage12AdapterError(
                "HOCap primary object is absent from this sequence: "
                f"primary={primary!r}, available={object_ids!r}"
            )
        return primary

    def _subject_betas(
        self, *, meta: dict[str, Any], sequence_dir: Path
    ) -> tuple[np.ndarray, Path]:
        """Load required subject calibration rather than silently using zero betas."""

        subject = str(meta.get("subject_id") or sequence_dir.parent.name)
        if subject != sequence_dir.parent.name:
            raise Stage12AdapterError(
                "HOCAP_REQUIRED_MANO_BETAS_MISSING: sequence subject and metadata disagree"
            )
        path = self.dataset_dir / "data" / "calibration" / "mano" / f"{subject}.yaml"
        if not path.is_file():
            raise Stage12AdapterError(
                f"HOCAP_REQUIRED_MANO_BETAS_MISSING: calibration file missing: {path}"
            )
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        betas = np.asarray(payload.get("betas"), dtype=np.float64)
        if betas.shape != (10,) or not np.isfinite(betas).all():
            raise Stage12AdapterError(
                "HOCAP_REQUIRED_MANO_BETAS_MISSING: calibration betas must be finite [10]"
            )
        return betas, path

    def load_sequence(
        self,
        sequence: str = "",
        *,
        frame_range: FrameRange | None = None,
        hand: str = "right",
        primary_object_id: str | None = None,
        **kwargs: Any,
    ) -> HOISequence:
        del kwargs
        selected_hand = str(hand).lower()
        if selected_hand not in {"left", "right"}:
            raise Stage12AdapterError("HOCap hand must be left or right")
        row = self._row(sequence)
        sequence_dir = self.sequence_root / row["sequence"]
        meta_path = sequence_dir / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        poses_m_path = sequence_dir / "poses_m.npy"
        poses_o_path = sequence_dir / "poses_o.npy"
        subject_betas, calibration_path = self._subject_betas(meta=meta, sequence_dir=sequence_dir)
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
        if selected_hand not in sides:
            raise Stage12AdapterError(
                f"HOCap sequence has no {selected_hand} hand: {row['sequence']}"
            )
        hand_index = hocap_mano_storage_index(selected_hand)
        if poses_m.shape[0] <= hand_index:
            raise Stage12AdapterError(
                f"HOCap poses_m is missing fixed {selected_hand} slot {hand_index}: {poses_m.shape}"
            )
        pose_values = np.asarray(poses_m[hand_index, start:stop], dtype=np.float64)
        valid = np.asarray(np.isfinite(pose_values).all(axis=1), dtype=bool)
        if not valid.all():
            raise Stage12AdapterError(
                f"HOCap selected clip contains invalid {selected_hand} MANO frames: "
                f"{np.flatnonzero(~valid).tolist()}"
            )
        mano_source_hash = sha256_paths([meta_path, poses_m_path, calibration_path])
        render = render_mano_pca45(
            pose_values,
            side=selected_hand,
            mano_model_root=self.mano_model_root,
            betas=subject_betas,
            dataset_name="hocap",
            source_annotation_path=poses_m_path,
            source_annotation_hash=mano_source_hash,
        )
        mano_parameters = ManoParameterTrack(
            global_orient_aa=pose_values[:, :3],
            hand_pose_aa=render.hand_pose_axis_angle,
            transl=pose_values[:, 48:51],
            betas=render.betas,
            model_profile="hocap_poses_m_pca45_explicit_contract_v2",
        )
        hand_track = make_hand(
            hand_id=f"{selected_hand}_hand",
            side=selected_hand,
            vertices_scene=render.vertices,
            faces=render.faces,
            wrist_pose_scene=render.wrist_pose_scene,
            valid=valid,
            mano_parameters=mano_parameters,
            mano_model_root=self.mano_model_root,
            metadata={
                "source": "HOCap poses_m PCA45 reconstruction",
                "source_hand_index": hand_index,
                "mano_representation": "pca",
                "num_pca_components": 45,
                "flat_hand_mean": False,
                "native_pca_coefficients": pose_values[:, 3:48].tolist(),
                "mano_reconstruction": render.reconstruction_manifest,
                "calibration_path": str(calibration_path),
                "calibration_hash": sha256_paths([calibration_path]),
            },
            native_joint_track=backend_posed_joint_track(render, valid=valid),
        )
        object_ids = self._object_ids(meta, poses_o.shape[1])
        primary_object_id = self._primary_object_id(object_ids, primary_object_id)
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
                    metadata={
                        "role": (
                            "primary_manipulation_object"
                            if object_id == primary_object_id
                            else "hocap_object_part"
                        ),
                        "object_index": object_index,
                        "primary_object_selection": (
                            "explicit_selection_contract"
                            if object_id == primary_object_id
                            else "context_object_part"
                        ),
                    },
                )
            )
        source_paths = [meta_path, poses_m_path, poses_o_path, calibration_path]
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
                "selected_hand": selected_hand,
                "object_ids": object_ids,
                "primary_object_id": primary_object_id,
                "rgb_depth_read": False,
            },
            metadata={
                "object_ids": object_ids,
                "primary_object_id": primary_object_id,
                "source_hand_index": hand_index,
                "mano_calibration_path": str(calibration_path),
                "mano_calibration_hash": sha256_paths([calibration_path]),
                "contact_annotation_available": False,
                "articulated_object_parts": True,
            },
        )
        result = HOISequence(metadata=metadata, hands=[hand_track], rigid_objects=objects)
        result.validate()
        return result


__all__ = ["HOCapAdapterV1"]
