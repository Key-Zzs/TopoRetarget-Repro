"""Explicit MANO geometry to MediaPipe-style 21-point conversion."""

from __future__ import annotations

import copy
import hashlib
import inspect
import pickle
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HandTrack, HOISequence, KeypointTrack
from toporetarget.keypoints.profiles import MappingProfile
from toporetarget.keypoints.registry import get_layout, load_layouts, load_profiles


class MappingError(ValueError):
    """Raised when a source hand cannot satisfy an explicit mapping profile."""


@dataclass(frozen=True)
class ManoModelGeometry:
    model_path: Path
    model_hash: str
    vertex_count: int
    joint_regressor: np.ndarray
    v_template: np.ndarray
    faces: np.ndarray | None = None


def _model_file(model_root: str | Path, side: str) -> Path:
    root = Path(model_root).expanduser()
    if side not in {"left", "right"}:
        raise MappingError(f"MANO side must be left or right, got {side!r}")
    filename = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
    if root.is_file():
        path = root
    elif (root / filename).is_file():
        path = root / filename
    elif (root / "mano" / filename).is_file():
        path = root / "mano" / filename
    else:
        raise MappingError(
            f"MANO model file {filename} was not found below {root}; pass a model directory "
            "containing MANO_RIGHT.pkl/MANO_LEFT.pkl"
        )
    return path


def _compat_pickle_imports() -> None:
    """Patch legacy MANO/chumpy names only when an external pkl is explicitly read."""

    import numpy as numpy_module

    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in numpy_module.__dict__:
            setattr(numpy_module, name, value)
    if not hasattr(inspect, "getargspec"):
        arg_spec = namedtuple(  # type: ignore[name-match]
            "ArgSpec", "args varargs keywords defaults"
        )

        def getargspec(function: Any) -> Any:
            full = inspect.getfullargspec(function)
            return arg_spec(full.args, full.varargs, full.varkw, full.defaults)

        inspect.getargspec = getargspec  # type: ignore[attr-defined]


def _dense_regressor(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise MappingError(f"MANO J_regressor must be 2-D, got {array.shape}")
    return array


def load_mano_model_geometry(
    model_root: str | Path,
    *,
    side: str,
    expected_vertex_count: int | None = None,
) -> ManoModelGeometry:
    """Load only the local MANO regressor/topology metadata needed by vertex conversion."""

    path = _model_file(model_root, side)
    _compat_pickle_imports()
    try:
        with path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
    except (OSError, pickle.PickleError, ImportError, AttributeError, ValueError) as exc:
        raise MappingError(f"could not load MANO model file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise MappingError(f"MANO model file {path} did not contain a mapping")
    template = np.asarray(data.get("v_template"), dtype=np.float64)
    if template.ndim != 2 or template.shape[1:] != (3,):
        raise MappingError(f"MANO v_template must have shape [V,3], got {template.shape}")
    vertex_count = int(template.shape[0])
    if expected_vertex_count is not None and vertex_count != expected_vertex_count:
        raise MappingError(
            f"MANO model topology mismatch: profile expects {expected_vertex_count} vertices, "
            f"model has {vertex_count}"
        )
    regressor = _dense_regressor(data.get("J_regressor"))
    if regressor.shape[1] != vertex_count:
        raise MappingError(
            f"MANO J_regressor shape {regressor.shape} is incompatible with {vertex_count} vertices"
        )
    faces_value = data.get("f")
    faces = None if faces_value is None else np.asarray(faces_value, dtype=np.int64)
    return ManoModelGeometry(
        model_path=path,
        model_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        vertex_count=vertex_count,
        joint_regressor=regressor,
        v_template=template,
        faces=faces,
    )


def _track_valid(track: KeypointTrack, frame_count: int) -> np.ndarray:
    if track.valid is None:
        result = np.ones(track.positions_scene.shape[:2], dtype=bool)
    elif track.valid.shape == (frame_count,):
        result = np.broadcast_to(track.valid[:, None], track.positions_scene.shape[:2]).copy()
    elif track.valid.shape == track.positions_scene.shape[:2]:
        result = track.valid.copy()
    else:
        raise MappingError(f"source track valid mask has unsupported shape {track.valid.shape}")
    return result & np.isfinite(track.positions_scene).all(axis=-1)


def _hand_valid(hand: HandTrack, frame_count: int) -> np.ndarray:
    if hand.valid is None:
        return np.ones(frame_count, dtype=bool)
    if hand.valid.shape != (frame_count,):
        raise MappingError(
            f"hand valid mask must have shape [{frame_count}], got {hand.valid.shape}"
        )
    return hand.valid.copy()


class ManoToMediaPipe21Converter:
    """Convert one canonical hand without resampling, mirroring, or normalization."""

    def __init__(
        self,
        profile: MappingProfile | str = "mano_v1_2_smplx_to_mediapipe21",
        *,
        config_root: str | Path | None = None,
    ) -> None:
        self.config_root = config_root
        if isinstance(profile, str):
            try:
                self.profile = load_profiles(config_root)[profile]
            except KeyError as exc:
                raise MappingError(f"unknown mapping profile: {profile}") from exc
        else:
            self.profile = profile
        layouts = load_layouts(config_root)
        self.source_layout = layouts[self.profile.source_joint_layout]
        self.target_layout = get_layout(self.profile.target_layout, config_root)
        self.profile.validate(layouts)

    def describe_profile(self) -> dict[str, Any]:
        return self.profile.as_dict()

    def _source_track(self, hand: HandTrack) -> tuple[KeypointTrack | None, str | None]:
        if self.profile.source_joint_layout in hand.keypoint_tracks:
            return hand.keypoint_tracks[
                self.profile.source_joint_layout
            ], self.profile.source_joint_layout
        for alias in self.source_layout.aliases:
            if alias in hand.keypoint_tracks:
                return hand.keypoint_tracks[alias], alias
        return None, None

    def _model_hash_if_requested(
        self, hand: HandTrack, mano_model_root: str | Path | None
    ) -> str | None:
        if mano_model_root is None:
            return None
        geometry = load_mano_model_geometry(
            mano_model_root,
            side=hand.side,
            expected_vertex_count=self.profile.expected_vertex_count,
        )
        if (
            hand.vertices_scene is not None
            and hand.vertices_scene.shape[1] != geometry.vertex_count
        ):
            raise MappingError(
                f"source vertices have {hand.vertices_scene.shape[1]} vertices but MANO model has "
                f"{geometry.vertex_count}"
            )
        return geometry.model_hash

    def _regressed_joints(
        self,
        hand: HandTrack,
        frame_count: int,
        mano_model_root: str | Path | None,
    ) -> tuple[np.ndarray, np.ndarray, str | None]:
        if hand.vertices_scene is None:
            raise MappingError(
                "vertices_with_joint_regressor requires hand.vertices_scene; "
                "no virtual joints are generated"
            )
        if mano_model_root is None:
            raise MappingError(
                "vertices_with_joint_regressor requires --mano-model-root or mano_model_root"
            )
        geometry = load_mano_model_geometry(
            mano_model_root,
            side=hand.side,
            expected_vertex_count=self.profile.expected_vertex_count,
        )
        vertices = np.asarray(hand.vertices_scene, dtype=np.float64)
        if vertices.shape != (frame_count, geometry.vertex_count, 3):
            raise MappingError(
                f"source vertices must have shape [{frame_count},{geometry.vertex_count},3], "
                f"got {vertices.shape}"
            )
        finite = np.isfinite(vertices).all(axis=(1, 2)) & _hand_valid(hand, frame_count)
        joints = np.einsum("jv,tvc->tjc", geometry.joint_regressor, vertices)
        joint_valid = np.broadcast_to(finite[:, None], joints.shape[:2]).copy()
        return joints, joint_valid, geometry.model_hash

    def convert_hand_track(
        self,
        hand: HandTrack,
        *,
        frame_count: int,
        mano_model_root: str | Path | None = None,
        overwrite: bool = False,
    ) -> KeypointTrack:
        """Return a new ``mediapipe21`` track; the input hand is not modified."""

        if "mediapipe21" in hand.keypoint_tracks and not overwrite:
            raise MappingError(
                "hand already contains mediapipe21; pass overwrite=True to replace it explicitly"
            )
        if hand.side not in {"left", "right"}:
            raise MappingError(f"hand side must be left or right, got {hand.side!r}")
        source_track, source_name = self._source_track(hand)
        model_hash: str | None = None
        if hand.vertices_scene is not None:
            vertices = np.asarray(hand.vertices_scene, dtype=np.float64)
            if vertices.ndim != 3 or vertices.shape[0] != frame_count or vertices.shape[2:] != (3,):
                raise MappingError(
                    f"source vertices must have shape [{frame_count},V,3], got {vertices.shape}"
                )
            if vertices.shape[1] != self.profile.expected_vertex_count:
                raise MappingError(
                    f"profile {self.profile.profile_id} expects "
                    f"{self.profile.expected_vertex_count} vertices, "
                    f"got {vertices.shape[1]}"
                )
            model_hash = self._model_hash_if_requested(hand, mano_model_root)
        elif any(
            anchor.source_type == "vertex" for anchor in self.profile.fingertip_mapping.values()
        ):
            vertices = None
        else:
            vertices = None

        regressed_joints: np.ndarray | None = None
        regressed_valid: np.ndarray | None = None
        if source_track is None and self.profile.mapping_mode == "vertices_with_joint_regressor":
            regressed_joints, regressed_valid, model_hash = self._regressed_joints(
                hand, frame_count, mano_model_root
            )
        if (
            source_track is None
            and regressed_joints is None
            and self.profile.mapping_mode != "validated_mano21_reorder"
        ):
            raise MappingError(
                f"no verified source layout {self.profile.source_joint_layout!r} "
                "(or alias) is present; "
                "refusing shape-only semantic guessing"
            )

        source_positions: dict[str, np.ndarray] = {}
        source_valid: dict[str, np.ndarray] = {}
        if source_track is not None:
            if source_track.positions_scene.shape[0] != frame_count:
                raise MappingError("source keypoint frame count does not match sequence")
            names = list(source_track.semantic_names or self.source_layout.semantic_names)
            if len(names) != source_track.positions_scene.shape[1] or len(set(names)) != len(names):
                raise MappingError(
                    "source keypoint semantic names are missing, duplicated, or wrong-sized"
                )
            source_positions = {
                name: source_track.positions_scene[:, index] for index, name in enumerate(names)
            }
            source_valid_array = _track_valid(source_track, frame_count)
            source_valid = {name: source_valid_array[:, index] for index, name in enumerate(names)}
        elif regressed_joints is not None and regressed_valid is not None:
            source_positions = {
                name: regressed_joints[:, index]
                for index, name in enumerate(self.source_layout.semantic_names)
            }
            source_valid = {
                name: regressed_valid[:, index]
                for index, name in enumerate(self.source_layout.semantic_names)
            }

        output = np.full((frame_count, self.target_layout.point_count, 3), np.nan, dtype=np.float64)
        output_valid = np.zeros((frame_count, self.target_layout.point_count), dtype=bool)
        hand_valid = _hand_valid(hand, frame_count)
        target_indices = self.target_layout.index_by_name
        for target_semantic, source_semantic in self.profile.joint_mapping.items():
            if source_semantic not in source_positions:
                raise MappingError(
                    f"profile source semantic {source_semantic!r} is not available in "
                    f"{source_name or 'regressed joints'}"
                )
            target_index = target_indices[target_semantic]
            valid = source_valid[source_semantic] & hand_valid
            output[valid, target_index] = source_positions[source_semantic][valid]
            output_valid[:, target_index] = valid

        for target_semantic, anchor in self.profile.fingertip_mapping.items():
            target_index = target_indices[target_semantic]
            if vertices is None:
                continue
            valid = hand_valid & np.isfinite(vertices[:, anchor.vertex_index]).all(axis=1)
            output[valid, target_index] = vertices[valid, anchor.vertex_index]
            output_valid[:, target_index] = valid

        if self.profile.mapping_mode == "validated_mano21_reorder" and source_track is not None:
            for _target_semantic, source_semantic in self.profile.joint_mapping.items():
                if source_semantic not in source_positions:
                    raise MappingError(
                        f"validated MANO-21 source semantic is unavailable: {source_semantic}"
                    )

        provenance = {
            "layout_name": "mediapipe21",
            "layout_version": "1.0.0",
            "coordinate_source": "mano_geometry",
            "coordinate_frame": "scene",
            "units": "m",
            "detector_output": False,
            "semantic_approximation": True,
            "mapping_profile_id": self.profile.profile_id,
            "mapping_profile_version": self.profile.version,
            "mapping_profile_hash": self.profile.sha256,
            "mapping_mode": self.profile.mapping_mode,
            "source_layout": source_name or self.profile.source_joint_layout,
            "source_track_present": source_track is not None,
            "fingertip_vertex_indices": {
                key: value.vertex_index for key, value in self.profile.fingertip_mapping.items()
            },
            "mano_model_hash": model_hash,
            "overwrite": overwrite,
            "assumptions": list(self.profile.assumptions),
        }
        return KeypointTrack(
            positions_scene=output,
            layout_name="mediapipe21",
            valid=output_valid,
            semantic_names=list(self.target_layout.semantic_names),
            frame_name="S",
            units="m",
            provenance=provenance,
        )

    def convert_sequence(
        self,
        sequence: HOISequence,
        *,
        hand_id: str,
        mano_model_root: str | Path | None = None,
        overwrite: bool = False,
    ) -> HOISequence:
        sequence.validate()
        result = copy.deepcopy(sequence)
        hand = result.hand(hand_id)
        output_track = self.convert_hand_track(
            hand,
            frame_count=result.num_frames,
            mano_model_root=mano_model_root,
            overwrite=overwrite,
        )
        hand.keypoint_tracks["mediapipe21"] = output_track
        conversion = {
            "mapping_profile_id": self.profile.profile_id,
            "mapping_profile_version": self.profile.version,
            "mapping_profile_hash": self.profile.sha256,
            "hand_id": hand_id,
            "source_layout": output_track.provenance["source_layout"],
            "target_layout": "mediapipe21",
            "overwrite": overwrite,
            "no_temporal_resampling": True,
            "no_spatial_sampling": True,
            "no_mirroring": True,
            "no_wrist_recentering": True,
        }
        result.metadata.provenance.conversion_options = {
            **result.metadata.provenance.conversion_options,
            "mano_to_mediapipe21": conversion,
        }
        result.metadata.metadata = {
            **result.metadata.metadata,
            "keypoint_layout_semantics": {
                "mediapipe21": {
                    "coordinate_source": "mano_geometry",
                    "detector_output": False,
                    "semantic_approximation": True,
                    "mapping_profile_id": self.profile.profile_id,
                    "mapping_profile_version": self.profile.version,
                    "mapping_profile_hash": self.profile.sha256,
                }
            },
        }
        hand.metadata = {
            **hand.metadata,
            "mediapipe_mapping": output_track.provenance,
            "no_mediapipe_detector": True,
        }
        result.validate()
        return result


def convert_sequence_to_mediapipe21(
    sequence: HOISequence,
    *,
    hand_id: str,
    profile: MappingProfile | str = "mano_v1_2_smplx_to_mediapipe21",
    mano_model_root: str | Path | None = None,
    overwrite: bool = False,
    config_root: str | Path | None = None,
) -> HOISequence:
    return ManoToMediaPipe21Converter(profile, config_root=config_root).convert_sequence(
        sequence,
        hand_id=hand_id,
        mano_model_root=mano_model_root,
        overwrite=overwrite,
    )


__all__ = [
    "ManoModelGeometry",
    "ManoToMediaPipe21Converter",
    "MappingError",
    "convert_sequence_to_mediapipe21",
    "load_mano_model_geometry",
]
