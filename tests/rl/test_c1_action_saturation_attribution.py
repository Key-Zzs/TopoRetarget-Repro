from __future__ import annotations

import pytest

from toporetarget.rl.c1_action_saturation_attribution import (
    ACTION_SEMANTICS,
    ACTION_THRESHOLD,
    FRACTION_LIMIT,
    action_dimension_rows,
    classify_trend,
    conclusion,
    metric_contract,
    parse_failure_metric,
)


def test_metric_contract_is_the_bounded_deterministic_mean_gate() -> None:
    contract = metric_contract()
    assert contract["thresholds"] == {
        "per_element_absolute_action_threshold": ACTION_THRESHOLD,
        "fail_fast_fraction_strictly_greater_than": FRACTION_LIMIT,
    }
    assert "tanh(actor_location" in str(contract["numerator"])
    assert "actuator_effort_saturation" in contract["not_measured"]


def test_failure_receipt_uses_tail_denominator_and_strict_gate() -> None:
    failure = parse_failure_metric(
        {"message": "PPO26D_ACTION_SATURATION_FAIL_FAST: phase=before_update fraction=0.260371"},
        rollout_steps=24,
        num_envs=1024,
    )
    assert failure.denominator == 638_976
    assert failure.fraction > FRACTION_LIMIT


def test_failure_receipt_rejects_non_saturation_errors() -> None:
    with pytest.raises(ValueError, match="RECEIPT_INVALID"):
        parse_failure_metric({"message": "different error"}, rollout_steps=24, num_envs=1024)


def test_all_26_dimensions_have_real_semantics_and_fail_closed_values() -> None:
    rows = action_dimension_rows()
    assert len(rows) == 26
    assert [row["semantic"] for row in rows] == list(ACTION_SEMANTICS)
    assert {row["unavailable_reason"] for row in rows} == {"NO_PERSISTED_C1_ACTION_TELEMETRY"}


def test_increasing_history_is_not_misclassified_as_tail_only() -> None:
    history = [
        {"deterministic_action_saturation_fraction": 0.018},
        {"deterministic_action_saturation_fraction": 0.070},
        {"deterministic_action_saturation_fraction": 0.068},
        {"deterministic_action_saturation_fraction": 0.120},
        {"deterministic_action_saturation_fraction": 0.165},
        {"deterministic_action_saturation_fraction": 0.207},
    ]
    failure = parse_failure_metric(
        {"message": "PPO26D_ACTION_SATURATION_FAIL_FAST: phase=before_update fraction=0.260371"},
        rollout_steps=24,
        num_envs=1024,
    )
    assert classify_trend(history) == "PERSISTENT_INCREASING"
    result = conclusion(c1_rows=history, failure=failure)
    assert result["primary_root_cause"] == "POLICY_OUTPUT_SATURATION_PRIMARY"
    assert result["partial_rollout_estimator_is_primary"] == "INCONCLUSIVE"
