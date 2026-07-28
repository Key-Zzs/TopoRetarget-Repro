from __future__ import annotations

import json

import numpy as np

from toporetarget.quality.contact import CONTACT_GRID, _huber
from toporetarget.quality.html import _clustered_mesh_preview
from toporetarget.quality.orchestrator import _classify_baseline_failure
from toporetarget.quality.schema import CLIPS, QUALITY_SCHEMA_VERSION, profile_label


def test_quality_clips_are_frozen_native_right_hand_units() -> None:
    assert [(item.sequence, item.start_frame, item.end_frame) for item in CLIPS] == [
        ("s1/airplane_lift", 240, 300),
        ("s1/apple_eat_1", 212, 272),
        ("s1/banana_lift", 1658, 1718),
        ("s1/alarmclock_lift", 407, 467),
    ]
    assert all(item.hand == "right" and item.robot == "artimano_rh" for item in CLIPS)
    assert all(item.length == 60 for item in CLIPS)


def test_contact_grid_and_huber_are_fixed() -> None:
    assert CONTACT_GRID == (
        ("P1", 0.25, 0.0),
        ("P2", 1.0, 0.0),
        ("P3", 4.0, 0.0),
        ("PD1", 1.0, 0.1),
        ("PD2", 1.0, 0.5),
    )
    assert _huber.__name__ == "_huber"
    assert _huber.__module__.endswith("quality.contact")


def test_profile_labels_and_schema_are_explicit() -> None:
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        **profile_label(paper_method=False, paper_external_extension=True),
    }
    assert json.loads(json.dumps(payload))["paper_external_extension"] is True
    assert payload["paper_method"] is False


def test_baseline_failure_classification_preserves_data_vs_solver_blockers() -> None:
    assert (
        _classify_baseline_failure(RuntimeError("strict signed distance requires watertight"))
        == "raw_grab_object_mesh_not_watertight_for_strict_signed_distance"
    )
    assert (
        _classify_baseline_failure(RuntimeError("Inequality constraints incompatible"))
        == "solver_incompatible_constraints_at_fixed_configuration"
    )


def test_clustered_mesh_preview_is_bounded_and_non_degenerate() -> None:
    axis = np.linspace(0.0, 1.0, 12)
    x, y = np.meshgrid(axis, axis)
    vertices = np.column_stack((x.reshape(-1), y.reshape(-1), np.zeros(x.size)))
    faces = []
    for row in range(11):
        for column in range(11):
            first = row * 12 + column
            faces.extend(
                (
                    (first, first + 1, first + 12),
                    (first + 1, first + 13, first + 12),
                )
            )
    preview_vertices, preview_faces = _clustered_mesh_preview(
        vertices,
        np.asarray(faces, dtype=np.int64),
        max_faces=50,
    )
    assert 0 < len(preview_faces) <= 50
    assert np.all(np.isfinite(preview_vertices))
    assert int(np.min(preview_faces)) >= 0
    assert int(np.max(preview_faces)) < len(preview_vertices)
    assert np.all(
        (preview_faces[:, 0] != preview_faces[:, 1])
        & (preview_faces[:, 1] != preview_faces[:, 2])
        & (preview_faces[:, 0] != preview_faces[:, 2])
    )
