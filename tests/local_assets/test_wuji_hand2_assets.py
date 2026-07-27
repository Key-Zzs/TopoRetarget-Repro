from __future__ import annotations

from pathlib import Path

import pytest

from toporetarget.robots.registry import RobotHandRegistry
from toporetarget.robots.reports import validate_robot_model

REPO_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("robot", "root", "urdf_hash", "mjcf_hash"),
    [
        (
            "wuji_hand2_beta1_rh",
            "r_wrist",
            "1ae70be3f5e64532203e599eaa98d2af368d0214be9c949a358b7abaa8b6265a",
            "1bc53b7ca6f2eb84fc66ac736027984cd1734fa30d27f9c1f640495258d626f9",
        ),
        (
            "wuji_hand2_beta1_lh",
            "l_wrist",
            "cec0a7eb6a34fd82e200def7b75c1d477fad790b2de903aec58e59991994c471",
            "ebaa3b07854c8df1847ffbf54d9d6527e82d838d02329e4c35c6ca667b8fef89",
        ),
    ],
)
def test_tracked_wuji_hand2_generic_validation(
    robot: str, root: str, urdf_hash: str, mjcf_hash: str
) -> None:
    model = RobotHandRegistry(repo_root=REPO_ROOT).load(robot)
    assert model.base_link == root
    assert model.num_dofs == 20
    assert len(model.link_names) == 26
    assert len(model.joint_names) == 25
    assert model.urdf_hash == urdf_hash
    assert model.spec.simulation.source_hash == mjcf_hash
    report = validate_robot_model(model)
    assert report.status == "pass", report.as_dict()
