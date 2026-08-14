from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.full_trajectory_episode_start import (
    FULL_TRAJECTORY_EPISODE_START_SCHEMA,
    select_full_trajectory_episode_start,
    validate_full_trajectory_start,
)


def _rows(valid: list[bool]) -> dict[str, np.ndarray]:
    count = len(valid)
    return {
        "runtime_index": np.arange(count),
        "semantic_class": np.asarray(["PRE_CONTACT"] * count),
        "support_state": np.asarray(["TABLE_SUPPORTED"] * count),
        "overall_reference_geometry_valid": np.asarray(valid),
        "reference_object_twist": np.ones((count, 6)),
    }


def test_frame_zero_is_preferred_and_uses_resting_velocity() -> None:
    selected = select_full_trajectory_episode_start(
        clip="hocap_170105",
        validity_rows=_rows([True, True, True]),
        stable_indices=(0, 1, 2),
        reference_hash="reference",
        support_contract_hash="support",
    )
    receipt = selected.as_dict()
    assert receipt["schema_version"] == FULL_TRAJECTORY_EPISODE_START_SCHEMA
    assert receipt["start_index"] == 0
    assert receipt["object_linear_velocity_mps"] == [0.0, 0.0, 0.0]
    assert receipt["random_state_init"] is False
    assert (
        validate_full_trajectory_start(receipt, clip="hocap_170105")["table_actor_active"] is True
    )


def test_uses_earliest_individual_valid_frame_without_an_eight_frame_requirement() -> None:
    selected = select_full_trajectory_episode_start(
        clip="hocap_170650",
        validity_rows=_rows([False, False, True, False]),
        stable_indices=(0, 1, 2, 3),
        reference_hash="reference",
        support_contract_hash="support",
    )
    assert selected.start_index == 2


def test_missing_eligible_start_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="P3_RESTART_BLOCKED_EPISODE_START"):
        select_full_trajectory_episode_start(
            clip="hocap_170105",
            validity_rows=_rows([False, False]),
            stable_indices=(0, 1),
            reference_hash="reference",
            support_contract_hash="support",
        )
