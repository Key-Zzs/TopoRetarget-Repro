from __future__ import annotations

import json
from pathlib import Path

from toporetarget.quality.html import _HTML_TEMPLATE, smoke_html


def _write_minimal_viewer(path: Path, object_id: str) -> None:
    profile = {"robot_mesh": {"parts": []}}
    payload = {
        "frame_count": 1,
        "clip": {"object_name": object_id},
        "profiles": {"source_mano": profile, "paper_warm": profile},
        "interaction_profiles": {"paper_warm": {}},
        "source_mesh": {"vertices": [[0.0, 0.0, 0.0]], "faces": [[0, 0, 0]]},
        "object_mesh": {
            "object_id": object_id,
            "vertices": [[0.0, 0.0, 0.0]],
            "faces": [[0, 0, 0]],
        },
        "context_object_meshes": [
            {
                "object_id": "G10_1",
                "vertices": [[1.0, 0.0, 0.0]],
                "faces": [[0, 0, 0]],
                "poses": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            }
        ],
        "robot_topology": {"parts": [{}]},
        "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    }
    path.write_text(
        _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload)).replace("__LAYERS__", "test"),
        encoding="utf-8",
    )


def test_smoke_html_checks_semantic_object_identity(tmp_path: Path) -> None:
    viewer = tmp_path / "viewer.html"
    _write_minimal_viewer(viewer, "G10_2")

    assert (
        smoke_html(
            viewer,
            expected_frames=1,
            profiles=1,
            expected_object_id="G10_2",
            expected_context_object_ids={"G10_1"},
        )["status"]
        == "pass"
    )
    assert (
        smoke_html(viewer, expected_frames=1, profiles=1, expected_object_id="G10_1")["status"]
        == "fail"
    )
