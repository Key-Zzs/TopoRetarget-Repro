"""Pure P3/P4 contact-ready evaluation contracts and derived metric tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.physical_evaluation import (
    ContactReadyEpisodePairV1,
    ContactReadyPairSetV1,
    contact_metrics,
    flight_metrics,
    load_contact_ready_evaluation_pairs,
    physical_failure_status,
    twist_metrics,
    validate_pair_set_against_safe_indices,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_frozen_pair_contract_is_20_per_kind_and_uses_contact_ready_safe_indices() -> None:
    pairs = load_contact_ready_evaluation_pairs(
        _root() / "configs/rl/stage16/stage16_p3_p4_contact_ready_evaluation_pairs_v1.yaml"
    )
    for clip, allowed in {
        "hocap_170105": {186, 203, 205, 206, 210, 211, 222, 238},
        "hocap_170650": {
            117,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            152,
            153,
            157,
            158,
            161,
            167,
            168,
            191,
            222,
            247,
            248,
            251,
            254,
            257,
            263,
            295,
        },
    }.items():
        development = pairs[clip]["development"]
        formal = pairs[clip]["formal"]
        validate_pair_set_against_safe_indices(development, safe_indices=allowed)
        validate_pair_set_against_safe_indices(formal, safe_indices=allowed)
        assert len(development.pairs) == len(formal.pairs) == 20
        assert {pair.seed for pair in development.pairs}.isdisjoint(
            {pair.seed for pair in formal.pairs}
        )
        assert "development" in development.identifier
        assert "formal" in formal.identifier


def test_pair_set_rejects_outside_safe_bank() -> None:
    pair_set = ContactReadyPairSetV1(
        identifier="test",
        source_seed_manifest="x.json",
        source_seed_set="development",
        pairs=tuple(ContactReadyEpisodePairV1(seed=index, reset_index=7) for index in range(20)),
    )
    with pytest.raises(ValueError, match="OUTSIDE_CONTACT_READY_SAFE_BANK"):
        validate_pair_set_against_safe_indices(pair_set, safe_indices=(1, 2))


def test_contact_flight_and_twist_metrics_preserve_distinct_semantics() -> None:
    expected = np.zeros((6, 5), dtype=bool)
    expected[1:5, 0] = True
    actual = np.zeros_like(expected)
    actual[1, 0] = True
    actual[2:4, 1] = True
    actual[4, 0] = True
    valid = np.array([False, True, True, True, True, True])
    interaction, per_finger = contact_metrics(expected=expected, actual=actual, valid=valid)
    assert interaction["source_tip_recall"] is not None
    assert interaction["source_persistent_tip_recall"] is not None
    assert interaction["cross_finger_compensation"] is not None
    assert len(per_finger) == 5

    pose = np.zeros((6, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    pose[2:4, 2] = -0.02
    twist = np.zeros((6, 6), dtype=np.float64)
    twist[2:4, 2] = -0.4
    flight = flight_metrics(
        tip_contact=actual.any(axis=-1),
        hand_contact=np.array([False, True, False, False, True, True]),
        valid=valid,
        object_pose=pose,
        object_twist=twist,
    )
    assert flight["flight_event_count"] == 1
    assert flight["recontact_count"] == 1
    assert flight["events"][0]["z_displacement_m"] <= 0.0

    twist_summary = twist_metrics(actual=twist, reference=np.zeros_like(twist), valid=valid)
    assert twist_summary["Delta_v_mps"]["p95"] > 0.0
    assert twist_summary["Delta_omega_radps"]["terminal"] == 0.0


def test_p95_geometry_violation_is_not_misreported_as_catastrophic_contact() -> None:
    failure = physical_failure_status(
        termination_reason=7,
        finite=True,
        absolute_geometry_pass=False,
        inter_finger_pass=True,
        max_penetration_m=0.005,
        catastrophic_penetration_m=0.01,
    )
    assert failure["catastrophic_contact"] is False
    assert failure["absolute_geometry_violation"] is True
