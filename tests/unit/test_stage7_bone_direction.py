from __future__ import annotations

import numpy as np
import pytest
import torch

from toporetarget.retarget.bones import (
    ZeroLengthBoneError,
    extract_bone_features,
    load_bone_profile,
)
from toporetarget.retarget.frames import FrameDegeneracyError, load_frame_profile
from toporetarget.retarget.objectives import equation_1_report


def _hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]
    paths = [
        [[0.02, 0.02, 0.0], [0.04, 0.04, 0.0], [0.06, 0.05, 0.0], [0.08, 0.05, 0.0]],
        [[0.05, 0.01, 0.0], [0.09, 0.01, 0.0], [0.12, 0.01, 0.0], [0.15, 0.01, 0.0]],
        [[0.05, 0.03, 0.0], [0.09, 0.03, 0.0], [0.12, 0.03, 0.0], [0.15, 0.03, 0.0]],
        [[0.05, 0.05, 0.0], [0.09, 0.05, 0.0], [0.12, 0.05, 0.0], [0.15, 0.05, 0.0]],
        [[0.05, 0.07, 0.0], [0.085, 0.07, 0.0], [0.115, 0.07, 0.0], [0.14, 0.07, 0.0]],
    ]
    indices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    for path, target in zip(paths, indices, strict=True):
        points[target] = path
    return points


def _rigid(points: np.ndarray) -> np.ndarray:
    angle = 0.41
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    return points @ rotation.T + np.array([0.3, -0.2, 0.4])


def test_profiles_have_exact_stage7_topology() -> None:
    full = load_bone_profile("mediapipe21_full_finger_chain_v1")
    phalange = load_bone_profile("mediapipe21_phalange_only_diagnostic")
    assert len(full.bones) == 20
    assert len(full.pairs) == 15
    assert len(phalange.bones) == 15
    assert len(phalange.pairs) == 10
    assert all(
        full.bones[p.first_bone].finger == full.bones[p.second_bone].finger for p in full.pairs
    )
    assert full.bones[0].parent_name == "wrist"
    assert full.bones[3].child_name == "thumb_tip"


def test_local_frame_is_rigid_transform_invariant_and_right_handed() -> None:
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    first = profile.frame_transform(_hand())
    profile.frame_transform(_rigid(_hand()))
    assert np.max(np.abs(first[:3, :3].T @ first[:3, :3] - np.eye(3))) < 1e-12
    assert np.linalg.det(first[:3, :3]) == pytest.approx(1.0, abs=1e-12)
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    a = extract_bone_features(_hand(), profile, bone)
    b = extract_bone_features(_rigid(_hand()), profile, bone)
    np.testing.assert_allclose(a.local_keypoints, b.local_keypoints, atol=1e-12)
    np.testing.assert_allclose(a.adjacent_features, b.adjacent_features, atol=1e-12)


def test_translation_centered_profile_retains_rotation_observability() -> None:
    local = _hand()
    rotated = _rigid(local)
    profile = load_frame_profile("translation_centered_scene_axes")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    first = extract_bone_features(local, profile, bone)
    second = extract_bone_features(rotated, profile, bone)
    assert not np.allclose(first.unit_directions, second.unit_directions)


def test_frame_and_zero_length_fail_strictly_without_identity_fallback() -> None:
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    degenerate = _hand()
    degenerate[9] = degenerate[0]
    with pytest.raises(FrameDegeneracyError):
        profile.frame_transform(degenerate)
    zero_bone = _hand()
    zero_bone[1] = zero_bone[0]
    with pytest.raises(ZeroLengthBoneError):
        extract_bone_features(zero_bone, profile, bone)


def test_eq1_is_exact_sum_and_feature_difference_is_not_renormalized() -> None:
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    source = extract_bone_features(_hand(), profile, bone)
    robot = extract_bone_features(_hand(), profile, bone)
    report = equation_1_report(source, robot)
    assert report["sum_loss"] == pytest.approx(0.0, abs=1e-24)
    assert len(report["per_pair_loss"]) == 15
    perturbed = _hand()
    perturbed[2] += np.array([0.0, 0.001, 0.0])
    changed = extract_bone_features(perturbed, profile, bone)
    residual = changed.adjacent_features - source.adjacent_features
    assert np.max(np.linalg.norm(residual, axis=-1)) < 2.0
    assert not np.allclose(np.linalg.norm(residual, axis=-1), 1.0)


def test_direction_features_support_float32_float64_and_torch_autograd() -> None:
    profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    value = torch.tensor(_hand(), dtype=torch.float64, requires_grad=True)
    features = extract_bone_features(value, profile, bone)
    loss = torch.sum(features.adjacent_features**2)
    loss.backward()
    assert features.unit_directions.dtype == torch.float64
    assert value.grad is not None and torch.isfinite(value.grad).all()
    value32 = extract_bone_features(torch.tensor(_hand(), dtype=torch.float32), profile, bone)
    assert value32.unit_directions.dtype == torch.float32
