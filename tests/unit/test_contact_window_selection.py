from toporetarget.workflows.contact_window import ContactWindow


def test_contact_window_serializes_frame_counts_and_deterministic_score() -> None:
    value = ContactWindow(
        sequence="s1/airplane_lift",
        hand="right",
        start_frame=240,
        end_frame=300,
        contact_frame_count=60,
        contact_frame_ratio=1.0,
        total_hand_contact_vertices=27266,
        median_hand_contact_vertices=400.0,
        max_hand_contact_vertices=600,
        no_contact_frames=[],
        observed_semantic_labels=[43, 46, 52, 55],
        contact_frames=list(range(240, 300)),
        contact_counts={str(frame): 400 for frame in range(240, 300)},
        source_contact_median_distance_m=0.003,
        source_geometry_status="pass",
    )
    payload = value.as_dict()
    assert payload["frame_range"] == [240, 300]
    assert payload["contact_frames"][0] == 240
    assert payload["contact_counts"]["299"] == 400
    assert value.score[:3] == (-1.0, -27266, 0.003)
