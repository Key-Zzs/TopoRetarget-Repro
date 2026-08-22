from __future__ import annotations

import numpy as np
import pytest

from toporetarget.evaluation.stage16_pf_v2_causal_lift import (
    Stage16PhysicalFunctionalityV2Contract,
    evaluate_stage16_physical_functionality_v2,
)


def _poses(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    object_pose = np.zeros((len(z), 7), dtype=np.float64)
    object_pose[:, 2] = z
    object_pose[:, 3] = 1.0
    wrist_pose = np.zeros((len(z), 7), dtype=np.float64)
    wrist_pose[:, 2] = np.linspace(0.1, 0.25, len(z))
    wrist_pose[:, 3] = 1.0
    return object_pose, wrist_pose


def _evaluate(
    *,
    z: np.ndarray,
    tip_contact: np.ndarray,
    hand_contact: np.ndarray | None = None,
    table: np.ndarray | None = None,
    interaction_valid: np.ndarray | None = None,
    support_valid: np.ndarray | None = None,
    causal: bool = True,
    hidden: bool = True,
) -> dict[str, object]:
    object_pose, wrist_pose = _poses(z)
    count = len(z)
    tips = np.zeros((count, 5), dtype=bool)
    tips[:, :2] = tip_contact[:, None]
    hand = np.zeros((count, 21), dtype=bool)
    hand[:, 0] = tip_contact if hand_contact is None else hand_contact
    return evaluate_stage16_physical_functionality_v2(
        object_pose_wxyz=object_pose,
        wrist_pose_wxyz=wrist_pose,
        tip_pair_presence=tips,
        hand_object_pair_presence=hand,
        table_object_contact=np.ones(count, dtype=bool) if table is None else table,
        interaction_valid=(
            np.ones(count, dtype=bool) if interaction_valid is None else interaction_valid
        ),
        support_valid=support_valid,
        reference_lift_onset=3,
        causal_execution=causal,
        geometry_safe=True,
        action_bounds_safe=True,
        no_hidden_control=hidden,
    )


def test_stable_prelift_grasp_then_lift_passes() -> None:
    result = _evaluate(
        z=np.asarray((0.0, 0.0, 0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11)),
        tip_contact=np.asarray((False, True, True, True, True, True, True, True, True)),
        table=np.asarray((True, True, True, True, True, False, False, False, False)),
    )
    assert result["pf_v2"] is True
    assert result["interaction_timing"]["pre_reference_lift_multifinger_contact"] is True


def test_dynamic_grasp_can_consolidate_after_reference_lift_and_pass() -> None:
    result = _evaluate(
        z=np.asarray((0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11)),
        tip_contact=np.asarray((False, False, False, False, True, True, True, True, True, True)),
        table=np.asarray((True, True, True, True, True, True, False, False, False, False)),
    )
    assert result["pf_v2"] is True
    assert result["interaction_timing"]["pre_reference_lift_multifinger_contact"] is False
    assert result["interaction_timing"]["pre_actual_lift_margin"] == 2


def test_reset_table_support_is_not_masked_by_invalid_hand_pair_force() -> None:
    count = 10
    result = _evaluate(
        z=np.asarray((0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11)),
        tip_contact=np.asarray((False, False, False, False, True, True, True, True, True, True)),
        table=np.asarray((True, False, False, False, False, False, False, False, False, False)),
        interaction_valid=np.asarray((False,) + (True,) * (count - 1)),
        support_valid=np.ones(count, dtype=bool),
    )
    assert result["support_transfer"]["support_present_before_release"] is True
    assert result["support_transfer_success"] is True
    assert result["pf_v2"] is True


def test_flick_lift_without_post_lift_contact_fails() -> None:
    result = _evaluate(
        z=np.asarray((0.0, 0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13)),
        tip_contact=np.asarray((False, True, True, True, False, False, False, False, False)),
        table=np.asarray((True, True, True, True, False, False, False, False, False)),
    )
    assert result["pf_v2"] is False
    assert "causal_hand_object_lift" in result["pf_v2_failure_reasons"]
    assert result["causal_interaction"]["ballistic_or_flick_rejected"] is False


def test_teleport_is_rejected_by_causal_and_hidden_gates() -> None:
    result = _evaluate(
        z=np.asarray((0.0, 0.0, 0.0, 0.0, 0.07, 0.07, 0.07, 0.07, 0.07)),
        tip_contact=np.ones(9, dtype=bool),
        table=np.asarray((True, True, True, True, False, False, False, False, False)),
        causal=False,
        hidden=False,
    )
    assert result["pf_v2"] is False
    assert {"causal_execution", "no_hidden_control"}.issubset(result["pf_v2_failure_reasons"])


@pytest.mark.parametrize(
    ("z", "table", "reason"),
    (
        (
            np.asarray((0.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.04, 0.04, 0.04)),
            np.asarray((True, True, True, True, False, False, False, False, False)),
            "physical_lift_success",
        ),
        (
            np.asarray((0.0, 0.0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13)),
            np.ones(9, dtype=bool),
            "physical_lift_success",
        ),
    ),
)
def test_insufficient_lift_and_table_supported_lift_fail(
    z: np.ndarray, table: np.ndarray, reason: str
) -> None:
    result = _evaluate(z=z, tip_contact=np.ones(len(z), dtype=bool), table=table)
    assert result["pf_v2"] is False
    assert reason in result["pf_v2_failure_reasons"]


def test_contract_rejects_a_reference_timing_hard_gate() -> None:
    with pytest.raises(ValueError, match="PF_V2_FORBIDDEN_SEMANTIC_CLAIM"):
        Stage16PhysicalFunctionalityV2Contract(reference_lift_hard_gate=True)
