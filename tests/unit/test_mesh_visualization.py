from toporetarget.workflows.mesh_visualization import _html_document


def test_mesh_html_is_self_contained_and_has_three_mesh_layers() -> None:
    payload = {
        "schema_version": "toporetarget.mesh_viewer.v1",
        "title": "test",
        "frame_count": 1,
        "source_sequence": "s1/test",
        "robot": "artimano_rh",
        "source": {"vertices": [[[0, 0, 0]]], "faces": []},
        "warm": {"parts": []},
        "final": {"parts": []},
        "object": {"object_id": None, "vertices": [], "poses": []},
        "metrics": {"frames": [{"local_frame": 0, "source_frame": 0}]},
        "bounds": [[-1, -1, -1], [1, 1, 1]],
    }
    html = _html_document(payload)

    assert "toporetarget.mesh_viewer" in html
    assert "Source MANO mesh" in html
    assert "Warm-start robot mesh" in html
    assert "Final robot mesh" in html
    assert "https://" not in html
    assert "${item.local_frame}" in html
