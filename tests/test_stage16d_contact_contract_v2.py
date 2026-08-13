from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toporetarget.evaluation.full_hand_contact import (  # noqa: E402
    FINGERTIP_NAMES,
    hand_body_group,
    hand_body_manifest,
)
from toporetarget.evaluation.reference_contact_contract import (  # noqa: E402
    FINGER_ORDER,
    evaluate_reference_contact,
    persistent_windows,
)


def test_v2_preserves_v3_mask_and_labels_proximity() -> None:
    distance = np.full((321, 5), 0.06)
    distance[:3, 0] = 0.02
    distance[:3, 1] = 0.025
    result = evaluate_reference_contact(distance)

    assert (
        result["historical_v3_primary_mask"][:3].tolist() == [[True, True, False, False, False]] * 3
    )
    assert (
        result["strong_contact_expected"][:3].tolist() == [[True, False, False, False, False]] * 3
    )
    assert result["proximity_only"][:3].tolist() == [[False, True, False, False, False]] * 3
    assert result["reference_evidence_class"][0, 0] == "GEOMETRIC_STRONG_CONTACT_CANDIDATE"
    assert result["reference_evidence_class"][0, 1] == "PROXIMITY_ONLY_AMBIGUOUS"


def test_source_evidence_overrides_distance_unless_conflicted() -> None:
    distance = np.full((321, 5), 0.06)
    source = np.zeros((321, 5), dtype=bool)
    source[10:13, 2] = True
    result = evaluate_reference_contact(distance, source_explicit_contact=source)

    assert not result["strong_contact_expected"][0].any()
    assert result["reference_contact_evidence_conflict"][10:13, 2].all()
    assert (
        result["reference_evidence_class"][10:13, 2] == "REFERENCE_CONTACT_EVIDENCE_CONFLICT"
    ).all()


def test_persistent_windows_record_evidence_composition() -> None:
    distance = np.full((321, 5), 0.06)
    distance[4:7, 3] = 0.01
    result = evaluate_reference_contact(distance)
    windows = persistent_windows(
        result["strong_contact_expected"],
        evidence_source=result["reference_evidence_source"],
        distances_m=result["reference_distance_m"],
    )

    assert len(windows) == 1
    window = windows[0]
    assert window["finger"] == FINGER_ORDER[3]
    assert (window["start"], window["end"], window["length_control_steps"]) == (4, 7, 3)
    assert window["evidence_source_composition"] == {"GEOMETRIC_PROXIMITY": 3}
    assert window["source_or_topology_supported_fraction"] == 0.0
    assert window["distance_m"] == pytest.approx({"min": 0.01, "mean": 0.01, "max": 0.01})


def test_full_hand_manifest_does_not_relabel_wrist_as_palm() -> None:
    names = (
        "r_wrist",
        "r_index_finger_proximal",
        "r_index_finger_proximal_abd",
        "r_index_finger_middle",
        "r_index_finger_distal",
        "r_middle_finger_proximal",
        "r_middle_finger_proximal_abd",
        "r_middle_finger_middle",
        "r_middle_finger_distal",
        "r_pinky_proximal",
        "r_pinky_proximal_abd",
        "r_pinky_middle",
        "r_pinky_distal",
        "r_ring_finger_proximal",
        "r_ring_finger_proximal_abd",
        "r_ring_finger_middle",
        "r_ring_finger_distal",
        "r_thumb_proximal",
        "r_thumb_proximal_abd",
        "r_thumb_middle",
        "r_thumb_distal",
    )
    manifest = hand_body_manifest(names, repo_root=ROOT)

    assert manifest["palm_mapping"]["palm_body_name"] is None
    assert manifest["palm_mapping"]["r_wrist_interpretation"] == "WRIST_BASE_CONTACT_BODY"
    assert tuple(manifest["fingertip_body_names"]) == FINGERTIP_NAMES
    assert hand_body_group("r_wrist") == "palm_or_wrist"
    assert hand_body_group("r_ring_finger_distal") == "ring"
