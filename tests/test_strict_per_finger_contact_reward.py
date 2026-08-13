"""Pure-contract tests for Stage 16-D Strict Per-Finger Contact Reward V4."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from toporetarget.rl.reference_tracking.strict_per_finger_contact import (
    StrictPerFingerContactRewardV4,
    strict_per_finger_contact_reward,
    strict_source_contact_mask,
)


def _reward(
    mask: list[list[bool]],
    forces: list[list[float]],
    *,
    presence: list[list[bool]] | None = None,
) -> dict[str, torch.Tensor]:
    scalar = torch.tensor(forces, dtype=torch.float32)
    force = torch.zeros((*scalar.shape, 3), dtype=torch.float32)
    force[..., 0] = scalar
    return strict_per_finger_contact_reward(
        source_contact_mask=torch.tensor(mask),
        fingertip_object_pair_force_world=force,
        pair_presence=None if presence is None else torch.tensor(presence),
        lambda_tip_n=1.0,
    )


def test_only_confirmed_and_persistent_source_labels_are_required() -> None:
    labels = np.full((321, 5), "SOURCE_NO_CONTACT", dtype="U32")
    labels[0, 0] = "SOURCE_CONTACT_CONFIRMED"
    labels[1, 1] = "SOURCE_CONTACT_PERSISTENT"
    labels[2, 2] = "SOURCE_CONTACT_PROBABLE"
    labels[3, 3] = "SOURCE_CONTACT_TRANSITION"
    labels[4, 4] = "SOURCE_PROXIMITY_ONLY"
    mask = strict_source_contact_mask(labels)
    assert mask[0, 0]
    assert mask[1, 1]
    assert not mask[2, 2]
    assert not mask[3, 3]
    assert not mask[4, 4]
    with pytest.raises(ValueError, match="STRICT_V4_SOURCE_CLASS_SHAPE_INVALID"):
        strict_source_contact_mask(labels[:320])


def test_one_expected_finger_only_gets_its_own_tip_credit() -> None:
    reward = _reward([[False, False, True, False, False]], [[1000, 1000, 0, 1000, 1000]])
    assert reward["per_finger_contact_reward"][0, 2].item() == 0.0
    assert reward["r_contact_v4"].item() == 0.0
    assert reward["source_satisfied_tip_count"].item() == 0


def test_other_finger_force_cannot_compensate_for_missing_required_finger() -> None:
    absent = _reward([[False, False, True, False, False]], [[0, 0, 0, 0, 0]])
    huge_other = _reward([[False, False, True, False, False]], [[1.0e6, 0, 0, 0, 0]])
    assert huge_other["r_contact_v4"].item() == absent["r_contact_v4"].item() == 0.0
    assert huge_other["per_finger_contact_reward"][0, 2].item() == 0.0


def test_expected_count_normalization_and_partial_coverage() -> None:
    one = _reward([[True, False, False, False, False]], [[1, 0, 0, 0, 0]])
    three = _reward([[True, True, True, False, False]], [[1, 1, 1, 0, 0]])
    partial = _reward([[True, True, True, False, False]], [[1, 0, 1, 0, 0]])
    assert one["r_contact_v4"].item() == pytest.approx(three["r_contact_v4"].item())
    assert partial["source_expected_finger_count"].item() == 3
    assert partial["source_satisfied_tip_count"].item() == 2
    assert partial["source_tip_coverage_ratio"].item() == pytest.approx(2 / 3)
    assert partial["r_contact_v4"].item() == pytest.approx(one["r_contact_v4"].item() * 2 / 3)


def test_zero_required_pair_absence_and_numerical_noise_are_zero() -> None:
    zero_expected = _reward([[False] * 5], [[10] * 5])
    absent = _reward(
        [[True, False, False, False, False]], [[10, 0, 0, 0, 0]], presence=[[False] * 5]
    )
    noise = _reward([[True, False, False, False, False]], [[1.0e-5, 0, 0, 0, 0]])
    assert zero_expected["r_contact_v4"].item() == 0.0
    assert absent["r_contact_v4"].item() == 0.0
    assert noise["r_contact_v4"].item() == 0.0


def test_reward_is_monotonic_bounded_and_has_exp_minus_one_at_lambda() -> None:
    low = _reward([[True, False, False, False, False]], [[0.25, 0, 0, 0, 0]])
    at_lambda = _reward([[True, False, False, False, False]], [[1.0, 0, 0, 0, 0]])
    high = _reward([[True, False, False, False, False]], [[1000.0, 0, 0, 0, 0]])
    assert (
        0.0
        < low["r_contact_v4"].item()
        < at_lambda["r_contact_v4"].item()
        < high["r_contact_v4"].item()
        <= 1.0
    )
    assert at_lambda["r_contact_v4"].item() == pytest.approx(np.exp(-1.0), abs=1.0e-5)


def test_contract_is_frozen_to_v4_strict_semantics() -> None:
    contract = StrictPerFingerContactRewardV4()
    assert contract.identifier == "StrictPerFingerContactRewardV4"
    assert contract.aggregation == "mean_over_source_required_fingers_only"
    with pytest.raises(ValueError, match="STRICT_V4_SOURCE_CLASS_POLICY_DRIFT"):
        StrictPerFingerContactRewardV4(source_required_classes=("SOURCE_CONTACT_PROBABLE",))
