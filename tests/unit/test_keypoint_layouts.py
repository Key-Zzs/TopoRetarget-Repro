from toporetarget.keypoints.registry import get_layout, load_layouts


def test_mediapipe21_layout_is_explicit_and_acyclic() -> None:
    layout = get_layout("mediapipe21")
    assert layout.point_count == 21
    assert layout.semantic_names == (
        "wrist",
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
        "index_mcp",
        "index_pip",
        "index_dip",
        "index_tip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "middle_tip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "ring_tip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    )
    assert layout.wrist_index == 0
    assert layout.fingertip_indices == (4, 8, 12, 16, 20)
    assert len(layout.edges) == 20
    assert all(
        len(children) == 0
        for index, children in layout.children.items()
        if index in layout.fingertip_indices
    )
    assert len(load_layouts()) == 4


def test_source_mano_layout_is_distinct_from_target() -> None:
    source = get_layout("mano16")
    target = get_layout("mediapipe21")
    assert source.name == "mano16_smplx"
    assert source.point_count == 16
    assert source.fingertip_indices == ()
    assert source.semantic_names != target.semantic_names
