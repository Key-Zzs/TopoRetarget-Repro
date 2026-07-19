from toporetarget.robots.spec import RobotHandSpec


def test_robot_spec_is_serializable_and_hashed_without_machine_paths() -> None:
    spec = RobotHandSpec.from_mapping(
        {
            "name": "fixture",
            "version": "1",
            "side": "synthetic",
            "asset_id": "fixture",
            "urdf_relative_path": "hand.urdf",
            "base_link": "palm",
            "semantic_keypoint_layout": "mediapipe21",
            "keypoint_anchor_profile": "fixture",
            "dof_order": ["joint"],
            "neutral_q": [0],
            "expected_link_count": 1,
            "expected_total_joint_count": 0,
            "expected_actuated_joint_count": 0,
            "expected_fixed_joint_count": 0,
            "expected_tip_links": [],
            "upstream_provenance": {"repository": "fixture", "relative_asset_path": "assets/hand"},
        }
    )
    assert spec.as_dict()["urdf_relative_path"] == "hand.urdf"
    assert len(spec.sha256) == 64
    assert "/home/" not in str(spec.as_dict())
