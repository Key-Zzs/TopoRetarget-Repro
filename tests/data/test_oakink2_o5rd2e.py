"""CPU-only contracts for O5R-D2E/D2F gate alignment and V3 recertification."""

from __future__ import annotations

from pathlib import Path

from scripts.data.run_oakink2_o5rd2e import (
    EPSILON_NUM,
    EXECUTION_NAME,
    TAU,
    certification_gate_v2,
    evaluate_gate_v2_rows,
    exact_nonregression,
    threshold_aware_nonregression,
)


def _row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "stratum": "HIGH",
        "old_e_im": 2.0e-4,
        "new_e_im": 8.0e-5,
        "technical_completion": True,
        "finite": True,
        "semantic_hard_pass": True,
        "wrist_pass": True,
        "bone_pass": True,
        "continuity_pass": True,
        "collision_pass": True,
        "joint_limits_pass": True,
        "reflection_pass": True,
        "scale_pass": True,
    }
    row.update(changes)
    return row


def _passing_rows() -> list[dict[str, object]]:
    invalid = [_row() for _ in range(20)]
    low = [_row(stratum="LOW", old_e_im=7.0e-5, new_e_im=9.0e-5) for _ in range(10)]
    return invalid + low


def test_threshold_aware_rule_matches_thresholded_objective_semantics() -> None:
    assert threshold_aware_nonregression(7.0e-5, 9.0e-5)
    assert not exact_nonregression(7.0e-5, 9.0e-5)
    assert threshold_aware_nonregression(2.0e-4, 2.0e-4 + EPSILON_NUM)
    assert not threshold_aware_nonregression(2.0e-4, 2.0e-4 + 2 * EPSILON_NUM)


def test_gate_v2_keeps_original_recovery_requirements_and_freezes_epsilon() -> None:
    gate = certification_gate_v2()
    assert gate["tau"] == TAU
    assert gate["epsilon_num"] == EPSILON_NUM == 1.0e-12
    assert gate["old_invalid_recovery_rate_minimum"] == 0.80
    assert gate["old_invalid_median_relative_reduction_minimum"] == 0.50
    assert gate["threshold_aware_nonregression_required_fraction"] == 1.0
    assert gate["exact_monotonic_fraction_role"] == "DIAGNOSTIC_ONLY"
    assert gate["mutable_after_freeze"] is False


def test_gate_v2_passes_valid_set_secondary_movement() -> None:
    result = evaluate_gate_v2_rows(_passing_rows(), determinism_pass=True)
    assert result["decision"] == "PASS"
    assert result["threshold_aware_nonregression_count"] == 30
    assert result["exact_monotonic_count_DIAGNOSTIC_ONLY"] == 20


def test_gate_v2_fails_each_structural_requirement() -> None:
    rows = _passing_rows()
    assert evaluate_gate_v2_rows(rows, determinism_pass=False)["decision"] == "FAIL"

    broken = [dict(row) for row in rows]
    broken[0]["technical_completion"] = False
    assert evaluate_gate_v2_rows(broken, determinism_pass=True)["decision"] == "FAIL"

    broken = [dict(row) for row in rows]
    broken[-1]["new_e_im"] = TAU + 2 * EPSILON_NUM
    assert evaluate_gate_v2_rows(broken, determinism_pass=True)["decision"] == "FAIL"


def test_gate_evaluator_normalizes_numpy_and_csv_style_booleans() -> None:
    rows = _passing_rows()
    for row in rows:
        for key in (
            "technical_completion",
            "finite",
            "semantic_hard_pass",
            "wrist_pass",
            "bone_pass",
            "continuity_pass",
            "collision_pass",
            "joint_limits_pass",
            "reflection_pass",
            "scale_pass",
        ):
            row[key] = "True"
    assert evaluate_gate_v2_rows(rows, determinism_pass=True)["decision"] == "PASS"


def test_cli_exposes_full_fail_closed_state_machine() -> None:
    source = Path("scripts/data/run_oakink2_o5rd2e.py").read_text(encoding="utf-8")
    for action in (
        "audit-gate-v1-authority",
        "audit-sparse-v2-low",
        "audit-qold-selection",
        "decide-d2e-root-cause",
        "freeze-gate-v2-if-authorized",
        "freeze-sparse-v3",
        "run-sparse-v3",
        "freeze-window-v3",
        "run-window-v3",
        "run-dev2-frame0-if-authorized",
        "run-dev2-full-if-authorized",
        "run-dev2-semantic-v1",
        "render-dev2-viewer",
    ):
        assert f'"{action}"' in source
    assert EXECUTION_NAME == "S1_DETERMINISTIC_MULTI_START"
    assert "CONTRACTS[EXECUTION_NAME]" in source


def test_dev2_missing_qold_fails_without_state_substitution() -> None:
    source = Path("scripts/data/run_oakink2_o5rd2e.py").read_text(encoding="utf-8")
    assert "MISSING_FROZEN_S1_OLD_PRODUCTION_Q_OLD_AUTHORITY" in source
    assert '"old_stage7_failure_terminal_is_q_old": False' in source
    assert '"optimizer_started": False' in source
    assert '"DEV2_FRAME0_RUN_COUNT": 3' in source
    assert '"DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0' in source


def test_historical_sparse_v2_is_never_promoted_by_posthoc_gate() -> None:
    source = Path("scripts/data/run_oakink2_o5rd2e.py").read_text(encoding="utf-8")
    assert '"SPARSE_VALIDATION_V2": "FAIL"' in source
    assert '"SPARSE_V2_GATE_V2_CERTIFICATION_STATUS": "NOT_APPLICABLE_POST_HOC"' in source
    assert '"historical_result_rewritten": False' in source
