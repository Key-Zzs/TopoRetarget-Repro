from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/audit_stage16d_per_finger_contact.py"
    )
    spec = importlib.util.spec_from_file_location("stage16d_per_finger_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_primary_mask_and_strong_ambiguous_evidence_partition() -> None:
    module = _module()
    distance = np.full((321, 5), 0.04)
    distance[:3, 0] = 0.01
    distance[:3, 1] = 0.02
    distance[:3, 2] = 0.025
    evidence = module._reference_evidence(distance)
    assert (
        evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"][:3].tolist()
        == [[True, True, True, False, False]] * 3
    )
    assert (
        evidence["REFERENCE_STRONG_CONTACT_EVIDENCE"][:3].tolist()
        == [[True, True, False, False, False]] * 3
    )
    assert (
        evidence["REFERENCE_PROXIMITY_ONLY_AMBIGUOUS"][:3].tolist()
        == [[False, False, True, False, False]] * 3
    )
    assert evidence["REFERENCE_NO_CONTACT_EXPECTED"][:3, 3:].all()


def test_actual_contact_requires_frozen_pair_validity() -> None:
    module = _module()
    actual = np.ones((321, 20, 5), dtype=bool)
    valid = np.ones((321, 20), dtype=bool)
    valid[0] = False
    valid[3, 2] = False
    result = module._actual_valid_contact(actual, valid)
    assert not result[0].any()
    assert not result[3, 2].any()
    assert result[1, 2].all()


def test_per_finger_persistent_windows_and_recontact_detection() -> None:
    module = _module()
    expected = np.zeros(321, dtype=bool)
    expected[4:12] = True
    actual = np.zeros(321, dtype=bool)
    actual[4:6] = True
    actual[9:12] = True
    assert module._persistent_mask(expected).sum() == 8
    assert module._longest_run(expected & ~actual) == 3
    assert module._event_counts(expected & ~actual) == (1, 1)


def test_compensation_uses_other_expected_fingers_and_relative_reward() -> None:
    module = _module()
    distance = np.full((321, 5), 0.04)
    distance[1:4, :2] = 0.01
    evidence = module._reference_evidence(distance)
    actual = np.zeros((321, 20, 5), dtype=bool)
    actual[1:4, :, 1] = True
    actual[4:6, :, 0] = True
    valid = np.ones((321, 20), dtype=bool)
    valid[0] = False
    magnitude = np.zeros((321, 20, 5))
    magnitude[1:4, :, 1] = 2.0
    magnitude[4:6, :, 0] = 1.0
    reward = np.zeros((321, 20))
    reward[1:4] = 0.8
    reward[4:6] = 0.4
    result = module._compensation(
        evidence=evidence,
        actual=actual,
        valid=valid,
        magnitude=magnitude,
        reward=reward,
    )
    assert result["per_finger"]["thumb"]["compensation_ratio"]["p50"] > 0.99
    assert result["full_coverage_reward_distribution"]["n"] == 0
    assert result["reward_compensation_ratio"] is None


def test_force_concentration_uses_actual_covered_expected_fingers() -> None:
    module = _module()
    distance = np.full((321, 5), 0.04)
    distance[1:3, :2] = 0.01
    evidence = module._reference_evidence(distance)
    actual = np.zeros((321, 20, 5), dtype=bool)
    actual[1:3, :, 0] = True
    valid = np.ones((321, 20), dtype=bool)
    valid[0] = False
    magnitude = np.zeros((321, 20, 5))
    magnitude[1:3, :, 0] = 2.0
    result = module._force_concentration(
        evidence=evidence,
        actual=actual,
        valid=valid,
        magnitude=magnitude,
    )
    assert result["expected_finger_count"]["2"]["coverage"] == 0.5
    assert result["largest_finger_force_share"]["p50"] > 0.99


def test_reference_evidence_is_diagnostic_and_does_not_mutate_inputs() -> None:
    module = _module()
    distance = np.full((321, 5), 0.025)
    before = distance.copy()
    evidence = module._reference_evidence(distance)
    assert np.array_equal(distance, before)
    assert evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"].all()
    assert evidence["REFERENCE_PROXIMITY_ONLY_AMBIGUOUS"].all()
