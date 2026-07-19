from pathlib import Path

import yaml

from toporetarget.robots.registry import RobotHandRegistry


def test_registry_lists_yaml_robot_without_loading_asset(tmp_path: Path) -> None:
    root = tmp_path / "robots"
    root.mkdir()
    (root / "synthetic.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "synthetic",
                "side": "right",
                "asset_id": "fixture",
                "urdf_relative_path": "hand.urdf",
                "base_link": "palm",
                "semantic_keypoint_layout": "mediapipe21",
                "keypoint_anchor_profile": "synthetic",
                "dof_order": ["bend"],
                "neutral_q": [0],
                "expected_link_count": 1,
                "expected_total_joint_count": 0,
                "expected_actuated_joint_count": 0,
                "expected_fixed_joint_count": 0,
            }
        ),
        encoding="utf-8",
    )
    registry = RobotHandRegistry(root, repo_root=tmp_path)
    listed = registry.list()
    assert listed[0]["name"] == "synthetic"
    assert listed[0]["asset"]["available"] is False
