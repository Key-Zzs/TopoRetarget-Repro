from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.finalize_stage16_frozen_source_policy_gravity_sweep import (
    _lineage_decision,
    classify_condition,
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
