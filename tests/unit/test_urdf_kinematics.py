from pathlib import Path

import numpy as np
import torch

from toporetarget.robots.registry import RobotHandRegistry

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic_robot"


def _model():
    return RobotHandRegistry(config_root=FIXTURE_ROOT, repo_root=FIXTURE_ROOT).load(
        "synthetic_hand", asset_root=FIXTURE_ROOT
    )


def test_synthetic_fk_matches_analytic_chain_and_reference() -> None:
    model = _model()
    q = np.array([np.pi / 2, 0.0, 0.2])
    fk = model.forward_kinematics_base(torch.tensor(q, dtype=torch.float64))
    np_fk = model.forward_kinematics_reference(q)
    np.testing.assert_allclose(
        fk["synthetic_tip"].detach().numpy(), np_fk["synthetic_tip"], atol=1e-12
    )
    np.testing.assert_allclose(
        fk["synthetic_tip"].detach().numpy()[:3, 3], [1.0, 2.4, 0.0], atol=1e-12
    )


def test_synthetic_fk_batch_base_equivariance_and_named_qpos() -> None:
    model = _model()
    q = torch.zeros((2, model.num_dofs), dtype=torch.float32)
    q[1, 2] = 0.2
    result = model.forward_kinematics_base(q)
    assert result["synthetic_tip"].shape == (2, 4, 4)
    base = torch.eye(4)
    base[:3, 3] = torch.tensor([0.1, 0.2, 0.3])
    scene = model.forward_kinematics_scene(q, base)
    torch.testing.assert_close(scene["synthetic_tip"], base @ result["synthetic_tip"])
    restored = model.qpos_from_named_dict(model.qpos_to_named_dict(q[0]))
    torch.testing.assert_close(restored, q[0])
