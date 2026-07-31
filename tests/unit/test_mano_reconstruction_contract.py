"""Unit coverage for the explicit MANO v2 request contract."""

from __future__ import annotations

import numpy as np
import pytest

from toporetarget.data.mano_backends.contracts import (
    AmbiguousManoPoseRepresentationError,
    InvalidManoPoseDimensionError,
    ManoModelProvenanceError,
    ManoPoseRepresentation,
    ManoReconstructionRequest,
    MissingRequiredManoBetasError,
)


def _request(
    representation: ManoPoseRepresentation = ManoPoseRepresentation.AXIS_ANGLE,
    *,
    components: int | None = None,
    betas: np.ndarray | None = None,
    flat_hand_mean: bool | None = False,
) -> ManoReconstructionRequest:
    count = 2
    if representation is ManoPoseRepresentation.AXIS_ANGLE:
        components = None
        pose = np.zeros((count, 45), dtype=np.float64)
    else:
        assert components is not None
        pose = np.zeros((count, components), dtype=np.float64)
    return ManoReconstructionRequest(
        side="right",
        pose_representation=representation,
        global_orient=np.zeros((count, 3), dtype=np.float64),
        hand_pose=pose,
        num_pca_components=components,
        translation=np.zeros((count, 3), dtype=np.float64),
        betas=np.zeros(10, dtype=np.float64) if betas is None else betas,
        flat_hand_mean=flat_hand_mean,
        units="metre",
        dtype=np.float64,
        model_path="/explicit/MANO_RIGHT.pkl",
        model_hash="declared-model-hash",
        model_version="MANO v1.2",
        dataset_name="test",
        source_annotation_path="/explicit/source.npz",
        source_annotation_hash="declared-source-hash",
    )


def test_axis_angle_45_is_an_explicit_valid_request() -> None:
    request = _request()
    assert request.pose_representation is ManoPoseRepresentation.AXIS_ANGLE
    assert request.hand_pose.shape == (2, 45)
    assert request.num_pca_components is None


@pytest.mark.parametrize("components", [15, 45])
def test_pca_components_are_explicit_and_match_pose_width(components: int) -> None:
    request = _request(ManoPoseRepresentation.PCA, components=components)
    assert request.pose_representation is ManoPoseRepresentation.PCA
    assert request.num_pca_components == components
    assert request.hand_pose.shape == (2, components)


def test_45_values_without_an_explicit_representation_fail_closed() -> None:
    with pytest.raises(AmbiguousManoPoseRepresentationError):
        ManoReconstructionRequest(
            side="right",
            pose_representation=None,
            global_orient=np.zeros((2, 3), dtype=np.float64),
            hand_pose=np.zeros((2, 45), dtype=np.float64),
            num_pca_components=None,
            translation=np.zeros((2, 3), dtype=np.float64),
            betas=np.zeros(10, dtype=np.float64),
            flat_hand_mean=False,
            units="metre",
            dtype=np.float64,
            model_path="/explicit/MANO_RIGHT.pkl",
            model_hash="declared-model-hash",
            model_version="MANO v1.2",
            dataset_name="test",
            source_annotation_path="/explicit/source.npz",
            source_annotation_hash="declared-source-hash",
        )


def test_pca45_cannot_be_labeled_axis_angle_with_a_pca_count() -> None:
    with pytest.raises(InvalidManoPoseDimensionError):
        request = _request()
        request.num_pca_components = 45
        request.__post_init__()


def test_pca_width_must_match_declared_component_count() -> None:
    with pytest.raises(InvalidManoPoseDimensionError):
        request = _request(ManoPoseRepresentation.PCA, components=15)
        request.hand_pose = np.zeros((2, 45), dtype=np.float64)
        request.__post_init__()


def test_missing_flat_hand_mean_fails_closed() -> None:
    with pytest.raises(ManoModelProvenanceError, match="flat_hand_mean"):
        _request(flat_hand_mean=None)


def test_required_betas_cannot_be_omitted() -> None:
    with pytest.raises(MissingRequiredManoBetasError):
        request = _request()
        request.betas = None
        request.__post_init__()


@pytest.mark.parametrize("shape", [(10,), (1, 10), (2, 10)])
def test_beta_broadcast_is_explicit_and_auditable(shape: tuple[int, ...]) -> None:
    request = _request(betas=np.zeros(shape, dtype=np.float64))
    assert request.broadcast_betas() is not None
    assert request.broadcast_betas().shape == (2, 10)  # type: ignore[union-attr]


def test_non_float64_input_fails_closed() -> None:
    with pytest.raises(ManoModelProvenanceError, match="hand_pose"):
        request = _request()
        request.hand_pose = request.hand_pose.astype(np.float32)
        request.__post_init__()
