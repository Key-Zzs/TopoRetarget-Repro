"""Opt-in real-model tests for the explicit Stage 12.5 MANO backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.data.mano_backends.contracts import (
    AmbiguousManoPoseRepresentationError,
    ManoPoseRepresentation,
    ManoReconstructionRequest,
)
from toporetarget.data.mano_backends.smplx_backend import SmplxManoBackend

pytestmark = pytest.mark.licensed_data

MODEL_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")


def _backend() -> SmplxManoBackend:
    if not MODEL_ROOT.is_dir():
        pytest.skip("shared local MANO model is unavailable")
    try:
        return SmplxManoBackend(MODEL_ROOT)
    except Exception as exc:  # optional smplx/torch dependency is intentionally opt-in
        pytest.skip(f"real MANO backend unavailable: {exc}")


def _request(
    backend: SmplxManoBackend,
    *,
    side: str = "right",
    representation: ManoPoseRepresentation,
    components: int | None,
    flat_hand_mean: bool,
) -> ManoReconstructionRequest:
    count = 2
    provenance = backend.model_provenance(
        side=side,
        dataset_name="stage12.5.real_backend_test",
        source_annotation_path="/explicit/test/source.npy",
        source_annotation_hash="real-model-test-source-hash",
    )
    width = 45 if representation is ManoPoseRepresentation.AXIS_ANGLE else components
    assert width is not None
    return ManoReconstructionRequest(
        side=side,
        pose_representation=representation,
        global_orient=np.zeros((count, 3), dtype=np.float64),
        hand_pose=np.zeros((count, width), dtype=np.float64),
        num_pca_components=components,
        translation=np.zeros((count, 3), dtype=np.float64),
        betas=np.linspace(-0.02, 0.02, 10, dtype=np.float64),
        flat_hand_mean=flat_hand_mean,
        units="metre",
        dtype=np.float64,
        model_path=provenance.model_path,
        model_hash=provenance.model_hash,
        model_version=provenance.model_version,
        dataset_name=provenance.dataset_name,
        source_annotation_path=provenance.source_annotation_path,
        source_annotation_hash=provenance.source_annotation_hash,
    )


@pytest.mark.parametrize(
    ("representation", "components"),
    [
        (ManoPoseRepresentation.AXIS_ANGLE, None),
        (ManoPoseRepresentation.PCA, 15),
        (ManoPoseRepresentation.PCA, 45),
    ],
)
def test_real_backend_preserves_explicit_representation_contract(
    representation: ManoPoseRepresentation, components: int | None
) -> None:
    backend = _backend()
    request = _request(
        backend,
        representation=representation,
        components=components,
        flat_hand_mean=False,
    )
    result = backend.reconstruct(request)
    assert result.vertices.shape == (2, 778, 3)
    assert result.posed_joints_native.shape == (2, 16, 3)
    assert result.posed_joint_layout == "mano16_smplx"
    assert result.full_pose_axis_angle.shape == (2, 48)
    assert result.vertices.dtype == np.float64
    assert result.posed_joints_native.dtype == np.float64
    assert np.isfinite(result.vertices).all()
    assert np.isfinite(result.posed_joints_native).all()
    assert np.allclose(result.full_pose_axis_angle[:, :3], request.global_orient)
    if representation is ManoPoseRepresentation.PCA:
        assert result.reconstruction_manifest["num_pca_components"] == components
        assert result.model_provenance.pca_basis_hash
        assert result.model_provenance.hand_mean_hash


def test_pca_expansion_is_the_backend_layer_basis_times_coefficients_plus_mean() -> None:
    backend = _backend()
    request = _request(
        backend,
        representation=ManoPoseRepresentation.PCA,
        components=15,
        flat_hand_mean=False,
    )
    request.hand_pose[0, :3] = [0.1, -0.2, 0.3]
    result = backend.reconstruct(request)
    layer = backend._layer(  # noqa: SLF001 - verifies the exact cached SMPL-X layer contract
        request,
        model_path=Path(request.model_path),
        use_pca=True,
        v_template=None,
    )
    basis, mean = backend._basis_and_mean(layer, 15)  # noqa: SLF001
    assert np.allclose(result.hand_pose_axis_angle, request.hand_pose @ basis + mean)


def test_cache_identity_separates_k_flat_hand_mean_and_side() -> None:
    backend = _backend()
    requests = [
        _request(
            backend,
            representation=ManoPoseRepresentation.PCA,
            components=15,
            flat_hand_mean=False,
        ),
        _request(
            backend,
            representation=ManoPoseRepresentation.PCA,
            components=45,
            flat_hand_mean=False,
        ),
        _request(
            backend,
            representation=ManoPoseRepresentation.PCA,
            components=15,
            flat_hand_mean=True,
        ),
        _request(
            backend,
            side="left",
            representation=ManoPoseRepresentation.PCA,
            components=15,
            flat_hand_mean=False,
        ),
    ]
    for request in requests:
        backend.reconstruct(request)
    keys = backend.cache_keys
    assert {key[0] for key in keys} == {"left", "right"}
    assert {key[2] for key in keys} == {15, 45}
    assert {key[3] for key in keys} == {False, True}


def test_legacy_render_entrypoint_fails_instead_of_guessing() -> None:
    backend = _backend()
    with pytest.raises(AmbiguousManoPoseRepresentationError):
        backend.render(
            params={},
            v_template=np.zeros((778, 3), dtype=np.float64),
            side="right",
            frame_count=1,
        )
