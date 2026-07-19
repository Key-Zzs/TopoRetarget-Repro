from pathlib import Path

from toporetarget.robots.registry import RobotHandRegistry

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic_robot"


def test_synthetic_robot_validation_passes() -> None:
    model = RobotHandRegistry(config_root=FIXTURE_ROOT, repo_root=FIXTURE_ROOT).load(
        "synthetic_hand", asset_root=FIXTURE_ROOT
    )
    report = model.validate()
    assert report.status == "pass", report.as_dict()
