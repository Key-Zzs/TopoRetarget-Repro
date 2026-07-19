import numpy as np


def test_interactive_viewer_callbacks_and_artist_stability() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from toporetarget.data.synthetic import make_synthetic_sequence
    from toporetarget.viz.grab_viewer import GrabViewerOptions, InteractiveHOIViewer

    sequence = make_synthetic_sequence(num_frames=5)
    viewer = InteractiveHOIViewer(
        canonical=sequence,
        options=GrabViewerOptions(mode="canonical", show_native_joints=True),
    )
    initial = len(viewer.artists)
    viewer.on_slider_changed(2)
    viewer.previous_frame()
    viewer.next_frame()
    viewer.first_frame()
    viewer.last_frame()
    viewer.set_reference_frame("object")
    viewer.set_visibility("object", False)
    viewer.set_visibility("object", True)
    viewer.toggle_play()
    viewer.toggle_play()
    assert viewer.frame == 4
    assert len(viewer.artists) == initial
    assert not viewer.is_playing
    viewer.close()
    assert not viewer.is_playing


def test_display_stride_changes_navigation_only(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from toporetarget.data.synthetic import make_synthetic_sequence
    from toporetarget.viz.grab_viewer import GrabViewerOptions, InteractiveHOIViewer

    sequence = make_synthetic_sequence(num_frames=5)
    timestamps = sequence.timestamps.copy()
    viewer = InteractiveHOIViewer(
        canonical=sequence,
        options=GrabViewerOptions(mode="canonical", display_stride=2),
    )
    viewer.first_frame()
    viewer.next_frame()
    assert viewer.frame == 2
    viewer.next_frame()
    assert viewer.frame == 4
    viewer.previous_frame()
    assert viewer.frame == 2
    np.testing.assert_array_equal(sequence.timestamps, timestamps)
    viewer.render_headless(tmp_path / "canonical.png")
    viewer.render_headless(tmp_path / "canonical.gif", start_frame=0, end_frame=5)
    assert (tmp_path / "canonical.png").is_file()
    assert (tmp_path / "canonical.gif").is_file()
    viewer.close()


def test_compare_viewer_supports_overlay_and_side_by_side(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from toporetarget.data.synthetic import make_synthetic_sequence
    from toporetarget.viz.grab_viewer import GrabViewerOptions, InteractiveHOIViewer

    raw = make_synthetic_sequence(num_frames=3)
    canonical = make_synthetic_sequence(num_frames=3)
    for layout in ("overlay", "side-by-side"):
        viewer = InteractiveHOIViewer(
            raw=raw,
            canonical=canonical,
            options=GrabViewerOptions(mode="compare", layout=layout),
        )
        viewer.render_headless(tmp_path / f"compare_{layout}.png")
        viewer.close()
        assert (tmp_path / f"compare_{layout}.png").is_file()


def test_semantic_contact_viewer_colors_and_artist_count_are_stable() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from toporetarget.data.contacts.grab import build_grab_contacts
    from toporetarget.data.synthetic import make_synthetic_sequence
    from toporetarget.viz.grab_viewer import GrabViewerOptions, InteractiveHOIViewer

    sequence = make_synthetic_sequence(num_frames=120)
    vertex_count = sequence.rigid_objects[0].mesh.vertices_local.shape[0]
    labels = np.zeros((sequence.num_frames, vertex_count), dtype=np.int16)
    labels[:, 0] = 21
    labels[:, 1] = 41
    sequence.contacts = build_grab_contacts(
        {"contact": {"object": labels}},
        hand_ids=[sequence.hands[0].hand_id],
        object_id=sequence.rigid_objects[0].object_id,
        object_vertex_count=vertex_count,
        frame_count=sequence.num_frames,
        mode="semantic",
    )
    viewer = InteractiveHOIViewer(
        canonical=sequence,
        options=GrabViewerOptions(
            mode="canonical", show_contacts=True, contact_color_mode="semantic"
        ),
    )
    initial = len(viewer.artists)
    for frame in range(100):
        viewer.update(frame)
    assert len(viewer.artists) == initial
    assert "grab_smplx_contact_joint" in viewer.figure._suptitle.get_text()
    assert "L_Hand" in viewer.contact_legend.get_text()
    assert "R_Index1" in viewer.contact_legend.get_text()
    viewer.set_contact_color_mode("binary")
    viewer.set_contact_color_mode("semantic")
    viewer.close()
