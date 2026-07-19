import os

import pytest

from toporetarget.geometry.robot_surface import (
    load_robot_surface_profile,
    sample_robot_collision_surface,
)
from toporetarget.robots.artimano import load_artimano_model

pytestmark = pytest.mark.skipif(
    os.environ.get("TOPORETARGET_RUN_LOCAL_ASSET_TESTS") != "1",
    reason="requires imported local Arti-MANO assets",
)


@pytest.mark.parametrize("side", ["right", "left"])
def test_real_artimano_collision_surface_is_collision_only(side: str) -> None:
    model = load_artimano_model(side)
    samples = sample_robot_collision_surface(
        model, model.neutral_q, load_robot_surface_profile("engineering_collision_32_per_geometry")
    )
    assert samples.count == 512
    assert samples.source_provenance["collision_only"] is True
    assert samples.source_provenance["visual_fallback"] is False
    assert samples.source_provenance["tip_visual_fallback"] is False
