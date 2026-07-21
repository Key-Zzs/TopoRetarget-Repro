from types import SimpleNamespace

import numpy as np

from toporetarget.workflows.interaction_html import _html_document as _interaction_html_document
from toporetarget.workflows.mesh_visualization import (
    _edge_category_codes,
    _filter_edge_indices,
    _html_document,
    _residual_summary,
    _visual_undirected_weights,
)


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


def test_interaction_graph_display_helpers_preserve_stage8_semantics() -> None:
    edges = np.asarray([[0, 1], [0, 21], [21, 22]], dtype=np.int64)
    assert _edge_category_codes(edges).tolist() == [0, 1, 2]
    directed = SimpleNamespace(
        source_index=np.asarray([0, 1, 0, 21, 21, 22], dtype=np.int64),
        destination_index=np.asarray([1, 0, 21, 0, 22, 21], dtype=np.int64),
        weights=np.asarray([0.2, 0.4, 0.1, 0.3, 0.8, 0.6], dtype=np.float64),
    )
    np.testing.assert_allclose(_visual_undirected_weights(directed, edges), [0.3, 0.2, 0.7])
    np.testing.assert_array_equal(
        _filter_edge_indices(
            edges,
            np.asarray([0, 1, 2]),
            np.asarray([0.3, 0.2, 0.7]),
            threshold=0.25,
            top_k=1,
            hand_object_only=False,
        ),
        [2],
    )
    np.testing.assert_array_equal(
        _filter_edge_indices(
            edges,
            np.asarray([0, 1, 2]),
            np.asarray([0.3, 0.2, 0.7]),
            hand_object_only=True,
        ),
        [1],
    )


def test_interaction_residual_summary_and_html_modes() -> None:
    residual = np.zeros((71, 3), dtype=np.float64)
    residual[3] = [0.0, 0.0, 0.2]
    residual[25] = [0.0, 0.3, 0.0]
    metadata = [{"semantic_name": f"v{i}"} for i in range(71)]
    summary = _residual_summary(residual, metadata)
    assert summary["max"] == 0.3
    assert summary["top"][0]["vertex_id"] == 25
    html = _interaction_html_document(
        {
            "title": "interaction",
            "frame_count": 1,
            "initial_mode": "figure4-style",
            "interaction": {"source_graph_artifact_hash": "graph-hash"},
        }
    )
    for mode in ("mesh", "full-graph", "figure4-style", "laplacian-diagnostic", "combined"):
        assert f'value="{mode}"' in html
    assert "handObjectOnly" in html
    assert "modeInput.value=DATA.initial_mode||'mesh'" in html
    assert "function drawMeshLayers()" in html
