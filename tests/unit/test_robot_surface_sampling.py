from pathlib import Path

from toporetarget.geometry.robot_surface import (
    RobotSurfaceSamplingProfile,
    sample_robot_collision_surface,
)
from toporetarget.robots.registry import RobotHandRegistry

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic_robot"


def test_collision_sampling_excludes_visual_geometry_and_preserves_links() -> None:
    model = RobotHandRegistry(config_root=FIXTURE_ROOT, repo_root=FIXTURE_ROOT).load(
        "synthetic_hand", asset_root=FIXTURE_ROOT
    )
    profile = RobotSurfaceSamplingProfile(
        "test",
        "1",
        8,
        "area_uniform_triangles",
        4,
        "engineering",
        "not_paper_specified",
        False,
        False,
    )
    samples = sample_robot_collision_surface(model, model.neutral_q, profile)
    assert samples.count == len(model.collision_geometry_instances(model.neutral_q)) * 8
    assert len(set(samples.link_names.tolist())) > 0
    assert all("visual" not in value for value in samples.geometry_ids.tolist())
    assert samples.source_provenance["collision_only"] is True
