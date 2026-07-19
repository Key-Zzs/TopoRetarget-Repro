import os

import pytest

from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.reports import jacobian_check

pytestmark = pytest.mark.skipif(
    os.environ.get("TOPORETARGET_RUN_LOCAL_ASSET_TESTS") != "1",
    reason="requires imported local Arti-MANO assets",
)


@pytest.mark.parametrize("side", ["right", "left"])
def test_real_artimano_side_validates_independently(side: str) -> None:
    model = load_artimano_model(side)
    report = model.validate(seed=4, dtype="float64")
    assert report.status == "pass", report.as_dict()
    assert len(model.link_names) == 28
    assert len(model.joint_names) == 27
    assert len(model.dof_names) == 22
    result = jacobian_check(
        model, model.neutral_q + 0.2 * (model.joint_upper - model.joint_lower), dtype="float64"
    )
    assert result["passed"], result
