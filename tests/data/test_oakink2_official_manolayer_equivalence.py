from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from toporetarget.adapters.datasets.oakink2 import reconstruct_mano_geometry


@pytest.mark.licensed_data
def test_reconstruction_matches_official_manolayer_quaternion_and_root_centre(
    tmp_path: Path,
) -> None:
    """Keep the local CPU viewer equivalent to OakInk2's official MANO preview.

    Set ``OAKINK2_OFFICIAL_PYTHON`` to the Python executable in the independent
    official OakInk2 environment.  Keeping that interpreter separate avoids
    mixing its legacy NumPy/chumpy ABI with the current project environment.
    """
    official_python = Path(
        os.environ.get(
            "OAKINK2_OFFICIAL_PYTHON",
            "/home/deepcybo/miniconda3/envs/ref2dex-oakink/bin/python",
        )
    )
    if not official_python.is_file():
        pytest.skip("set OAKINK2_OFFICIAL_PYTHON to run official ManoLayer equivalence")
    model_path = Path(
        os.environ.get(
            "MANO_RIGHT_MODEL",
            "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/MANO_RIGHT.pkl",
        )
    )
    if not model_path.is_file():
        pytest.skip(f"licensed MANO right model unavailable: {model_path}")

    asset_root = tmp_path / "mano_v1_2"
    models = asset_root / "models"
    models.mkdir(parents=True)
    (models / "MANO_RIGHT.pkl").symlink_to(model_path)
    official_vertices = tmp_path / "official_vertices.npz"
    source_root = Path(__file__).parents[2] / "src"
    script = textwrap.dedent(
        """
        import sys

        import numpy as np
        import torch
        from manotorch.manolayer import ManoLayer

        pose = np.zeros((2, 16, 4), dtype=np.float32)
        pose[..., 0] = 1.0
        pose[1, 0] = [0.9393727, 0.0, 0.0, 0.3428978]
        pose[1, 1] = [0.9800666, 0.1986693, 0.0, 0.0]
        pose[1, 6] = [0.9887711, 0.0, -0.1494381, 0.0]
        pose[1, 11] = [0.9950042, 0.0, 0.0, 0.0998334]
        betas = np.array(
            [[0.0] * 10, [0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.0, 0.01, 0.0, -0.02]],
            dtype=np.float32,
        )
        translation = np.array([[0.1, -0.2, 0.3], [-0.05, 0.04, 0.2]], dtype=np.float32)
        layer = ManoLayer(
            mano_assets_root=sys.argv[1],
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        ).to("cpu")
        official = layer(
            pose_coeffs=torch.from_numpy(pose), betas=torch.from_numpy(betas)
        )
        uncentred_layer = ManoLayer(
            mano_assets_root=sys.argv[1],
            rot_mode="quat",
            side="right",
            center_idx=None,
            use_pca=False,
            flat_hand_mean=True,
        ).to("cpu")
        uncentred = uncentred_layer(
            pose_coeffs=torch.from_numpy(pose), betas=torch.from_numpy(betas)
        )
        np.savez(
            sys.argv[2],
            pose=pose,
            betas=betas,
            translation=translation,
            vertices=official.verts.detach().cpu().numpy() + translation[:, None, :],
            joints=official.joints.detach().cpu().numpy() + translation[:, None, :],
            uncentred_vertices=uncentred.verts.detach().cpu().numpy() + translation[:, None, :],
        )
        """
    )
    subprocess.run(
        [str(official_python), "-c", script, str(asset_root), str(official_vertices)],
        check=True,
        cwd=source_root.parent,
    )
    with np.load(official_vertices) as payload:
        actual_vertices, actual_joints, _ = reconstruct_mano_geometry(
            payload["pose"], payload["translation"], payload["betas"], model_path
        )
        np.testing.assert_allclose(actual_vertices, payload["vertices"], rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(actual_joints, payload["joints"], rtol=2e-6, atol=2e-7)
        assert np.max(np.abs(payload["vertices"] - payload["uncentred_vertices"])) > 1e-3

        wrong_xyzw = payload["pose"][..., [1, 2, 3, 0]]
        wrong_vertices, _, _ = reconstruct_mano_geometry(
            wrong_xyzw, payload["translation"], payload["betas"], model_path
        )
        assert np.max(np.linalg.norm(wrong_vertices - payload["vertices"], axis=-1)) > 1e-3
