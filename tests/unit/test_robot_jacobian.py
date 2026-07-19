from pathlib import Path

import numpy as np
import torch

from toporetarget.robots.registry import RobotHandRegistry
from toporetarget.robots.reports import jacobian_check

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic_robot"


def test_synthetic_jacobian_is_differentiable_and_matches_finite_difference() -> None:
    model = RobotHandRegistry(config_root=FIXTURE_ROOT, repo_root=FIXTURE_ROOT).load(
        "synthetic_hand", asset_root=FIXTURE_ROOT
    )
    q = torch.tensor([0.2, -0.1, 0.15], dtype=torch.float64)
    jacobian = model.keypoint_jacobian_qpos(q)
    assert jacobian.shape == (21, 3, 3)
    report = jacobian_check(model, np.asarray(q), dtype="float64")
    assert report["passed"]
    assert report["maximum_absolute_error"] < 1e-5
