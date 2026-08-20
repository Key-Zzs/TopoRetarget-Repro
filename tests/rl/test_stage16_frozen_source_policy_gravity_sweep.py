from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.finalize_stage16_frozen_source_policy_gravity_sweep import (
    _lineage_decision,
    classify_condition,
)
from scripts.evaluation.finalize_stage16_full_gravity_capability_closure import _receipt_metrics
from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
    _completion_layers,
    _slice_parallel_trace,
)


def test_condition_classification_is_fail_closed() -> None:
    assert classify_condition(None) == "TECHNICALLY_INCONCLUSIVE"
    assert (
        classify_condition({"persistent_grasp_episodes": 10, "lift_episodes": 10}) == "FUNCTIONAL"
    )
    assert (
        classify_condition({"persistent_grasp_episodes": 10, "lift_episodes": 0})
        == "PARTIALLY_FUNCTIONAL"
    )
    assert (
        classify_condition({"persistent_grasp_episodes": 0, "lift_episodes": 0}) == "NON_FUNCTIONAL"
    )


def test_lineage_reports_technical_c4_without_converting_it_to_failure() -> None:
    rows = [
        {"source": "v3_hocap_170105", "stage": "C0", "classification": "FUNCTIONAL"},
        {"source": "v3_hocap_170105", "stage": "C1", "classification": "PARTIALLY_FUNCTIONAL"},
        {"source": "v3_hocap_170105", "stage": "C2", "classification": "TECHNICALLY_INCONCLUSIVE"},
        {"source": "v3_hocap_170105", "stage": "C3", "classification": "PARTIALLY_FUNCTIONAL"},
        {"source": "v3_hocap_170105", "stage": "C4", "classification": "TECHNICALLY_INCONCLUSIVE"},
    ]

    result = _lineage_decision(rows)

    assert result["LAST_FUNCTIONAL_STAGE"] == "C0"
    assert result["FIRST_NON_FUNCTIONAL_STAGE"] == "C1"
    assert result["C4_FULL_GRAVITY_FUNCTIONAL"] == "TECHNICALLY_INCONCLUSIVE"
    assert result["NON_MONOTONIC_CAPABILITY"] == "NOT_IDENTIFIABLE_DUE_TECHNICAL_GAP"


def test_parallel_trace_excludes_rows_after_an_early_vector_reset() -> None:
    """A reset's later terminal phase must not make an early episode complete."""

    trace = {
        "reference_index": np.stack((np.arange(321), np.arange(321)), axis=1),
        "phase_code": np.stack(
            (np.minimum(np.arange(321) // 46, 6), np.minimum(np.arange(321) // 46, 6)), axis=1
        ),
        "object_pose": np.zeros((321, 2, 7), dtype=np.float32),
    }
    sliced = _slice_parallel_trace(
        trace, replica=0, replicas=2, clip="hocap_170650", expected_frames=273
    )
    completion = _completion_layers(
        trace=sliced,
        rollout={"steps": 272, "reached_reference_end": False, "termination_reason": 4},
        start=0,
    )

    assert sliced["reference_index"].tolist() == list(range(273))
    assert "TERMINAL" not in set(sliced["phase"].tolist())
    assert completion["SIMULATION_COMPLETED"] is False
    assert completion["terminal_semantic_recorded"] is False


def test_completion_layers_require_the_authoritative_terminal_reference() -> None:
    trace = {
        "reference_index": np.arange(321, dtype=np.int64),
        "phase": np.asarray(["TERMINAL"] * 321),
        "object_pose": np.zeros((321, 7), dtype=np.float32),
        "object_twist": np.zeros((321, 6), dtype=np.float32),
        "wrist_pose": np.zeros((321, 7), dtype=np.float32),
        "finger_q": np.zeros((321, 20), dtype=np.float32),
        "action": np.zeros((321, 26), dtype=np.float32),
    }
    completion = _completion_layers(
        trace=trace,
        rollout={"steps": 320, "reached_reference_end": True, "termination_reason": 7},
        start=0,
    )

    assert completion["SIMULATION_COMPLETED"] is True
    assert completion["TRACE_COMPLETED"] is True
    assert completion["terminal_semantic_recorded"] is True


def test_closure_metrics_keep_early_physical_failure_distinct_from_a_pass() -> None:
    result = _receipt_metrics(
        {
            "status": "COMPLETE_DIAGNOSTIC_SWEEP_WITH_PHYSICAL_FAILURE",
            "persistent_grasp_episodes": 10,
            "lift_episodes": 0,
            "object_lift_dz_m": -1.0,
            "active_force_p95_n": None,
            "completion": {
                "SIMULATION_COMPLETED": False,
                "TRACE_COMPLETED": False,
                "QUALIFICATION_COMPLETED": False,
            },
        }
    )

    assert result["technical_complete"] is False
    assert result["grasp"] == 10
    assert result["lift"] == 0
    assert result["force_p95_n"] == 0.0
