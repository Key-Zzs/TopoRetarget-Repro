"""Explicit source-hand keypoint layouts and MANO-to-MediaPipe21 conversion."""

from toporetarget.keypoints.frames import mediapipe21_scene_to_wrist, mediapipe21_wrist_to_scene
from toporetarget.keypoints.layouts import KeypointLayoutDefinition
from toporetarget.keypoints.mano_to_mediapipe import (
    ManoModelGeometry,
    ManoToMediaPipe21Converter,
    MappingError,
    convert_sequence_to_mediapipe21,
    load_mano_model_geometry,
)
from toporetarget.keypoints.profiles import MappingProfile
from toporetarget.keypoints.registry import get_layout, load_layouts, load_profiles
from toporetarget.keypoints.reports import MappingConsistencyReport, validate_mapping

__all__ = [
    "KeypointLayoutDefinition",
    "ManoModelGeometry",
    "ManoToMediaPipe21Converter",
    "MappingConsistencyReport",
    "MappingError",
    "MappingProfile",
    "convert_sequence_to_mediapipe21",
    "get_layout",
    "load_layouts",
    "load_mano_model_geometry",
    "load_profiles",
    "mediapipe21_scene_to_wrist",
    "mediapipe21_wrist_to_scene",
    "validate_mapping",
]
