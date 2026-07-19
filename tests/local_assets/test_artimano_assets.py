import os
from pathlib import Path

import pytest

from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.reports import jacobian_check

pytestmark = pytest.mark.licensed_data


def _asset_root() -> Path:
    return Path(
        os.environ.get(
            "ARTIMANO_ASSET_ROOT", Path(__file__).parents[2] / ".local" / "assets" / "artimano"
        )
    )


def _model(side: str):
    if os.environ.get("TOPORETARGET_RUN_LOCAL_ASSET_TESTS") != "1":
        pytest.skip("set TOPORETARGET_RUN_LOCAL_ASSET_TESTS=1 for imported Arti-MANO assets")
    return load_artimano_model(side, asset_root=_asset_root())


@pytest.mark.parametrize("side", ["right", "left"])
def test_real_artimano_topology_fk_anchors_and_jacobian(side: str) -> None:
    model = _model(side)
    assert len(model.link_names) == 28
    assert len(model.joint_names) == 27
    assert model.num_dofs == 22
    assert model.base_link == "palm"
    assert model.keypoints_base(model.neutral_q).shape == (21, 3)
    assert model.visual_geometry_instances(model.neutral_q)
    assert model.collision_geometry_instances(model.neutral_q)
    result = jacobian_check(model, model.neutral_q, dtype="float64")
    assert result["passed"], result
