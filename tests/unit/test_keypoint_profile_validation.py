from toporetarget.keypoints.registry import load_profiles


def test_default_profile_has_explicit_provenance_and_five_tip_anchors() -> None:
    profile = load_profiles()["mano_v1_2_smplx_to_mediapipe21"]
    assert profile.expected_vertex_count == 778
    assert profile.mapping_mode == "joint_map_plus_tip_vertices"
    assert set(profile.fingertip_mapping) == {
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    }
    assert "A_MANO_MEDIAPIPE_SEMANTICS_001" in profile.assumptions
    assert profile.sha256
