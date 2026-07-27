from __future__ import annotations

import json

from toporetarget.quality.contact import CONTACT_GRID, _huber
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
