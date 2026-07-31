"""Contract tests for static singleton runtime health classification."""

from __future__ import annotations

import copy
import hashlib
import json

from toporetarget.retarget.static_runtime_policy import (
    STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING,
    STATIC_FRAME_HARD_RUNTIME_FAILURE,
    classify_runtime_health,
    is_static_single_frame_contract,
)


def test_dynamic_105_seconds_remains_a_hard_failure() -> None:
    decision = classify_runtime_health(
        elapsed_s=105.0, frame_times_s=[105.0], static_single_frame=False
    )
    assert decision.terminal_reason == "single_frame_over_90s:105.000"


def test_dynamic_rolling_p95_retains_the_existing_linear_percentile_gate() -> None:
    below_gate = classify_runtime_health(
        elapsed_s=1.0, frame_times_s=[0.0] * 9 + [31.0], static_single_frame=False
    )
    above_gate = classify_runtime_health(
        elapsed_s=1.0, frame_times_s=[0.0] * 9 + [60.0], static_single_frame=False
    )
    assert below_gate.terminal_reason is None
    assert above_gate.terminal_reason == "rolling_10_frame_p95_over_30s"


def test_static_105_seconds_is_accepted_with_a_warning() -> None:
    decision = classify_runtime_health(
        elapsed_s=105.0, frame_times_s=[105.0], static_single_frame=True
    )
    assert decision.status == STATIC_FRAME_ACCEPTED_WITH_RUNTIME_WARNING
    assert decision.terminal_reason is None
    assert decision.rolling_p95_gate == "NOT_APPLICABLE"
    assert decision.consecutive_slow_frame_gate == "NOT_APPLICABLE"


def test_static_301_seconds_is_a_hard_failure() -> None:
    decision = classify_runtime_health(
        elapsed_s=301.0, frame_times_s=[301.0], static_single_frame=True
    )
    assert decision.status == STATIC_FRAME_HARD_RUNTIME_FAILURE
    assert decision.terminal_reason == "static_frame_over_300s:301.000"


def test_static_contract_has_no_dataset_name_conditional_or_artifact_side_effect() -> None:
    contract = {
        "sample_type": "static_contact_evaluation_only",
        "articulated_frame_count": 1,
        "temporal_metrics_applicable": False,
        "solver_inputs": {"tau": 0.1, "slack_penalty": 2.0},
    }
    canonical = {"frame_count": 1}
    before = json.dumps(contract, sort_keys=True)
    before_hash = hashlib.sha256(before.encode("utf-8")).hexdigest()
    contactpose = {**copy.deepcopy(contract), "dataset": "contactpose"}
    another_dataset = {**copy.deepcopy(contract), "dataset": "another_static_dataset"}
    assert is_static_single_frame_contract(contactpose, canonical, frame_count=1)
    assert is_static_single_frame_contract(another_dataset, canonical, frame_count=1)
    classify_runtime_health(elapsed_s=105.0, frame_times_s=[105.0], static_single_frame=True)
    after = json.dumps(contract, sort_keys=True)
    assert after == before
    assert hashlib.sha256(after.encode("utf-8")).hexdigest() == before_hash
