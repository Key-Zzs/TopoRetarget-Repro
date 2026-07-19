from pathlib import Path

from toporetarget.robots.registry import RobotHandRegistry

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic_robot"


def test_synthetic_visual_and_collision_geometry_are_separate() -> None:
    model = RobotHandRegistry(config_root=FIXTURE_ROOT, repo_root=FIXTURE_ROOT).load(
        "synthetic_hand", asset_root=FIXTURE_ROOT
    )
    visual = model.visual_geometry_instances(model.neutral_q)
    collision = model.collision_geometry_instances(model.neutral_q)
    assert {item.kind for item in visual} == {"visual"}
    assert {item.kind for item in collision} == {"collision"}
    assert len(visual) == 5
    assert len(collision) == 4
    assert any(item.geometry_type == "sphere" for item in collision)
    assert any(item.geometry_type == "cylinder" for item in visual)
    assert model.urdf.links["synthetic_tip"].collisions == ()
