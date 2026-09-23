from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd3r4 as r4


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_historical_hashes_are_exact() -> None:
    assert (
        r4.sha256_file(r4.R2_ROOT / "design/refinement_v2_design.json") == r4.EXPECTED_DESIGN_SHA256
    )
    assert (
        r4.sha256_file(r4.R2_ROOT / "design/refinement_v2_development_gate.json")
        == r4.EXPECTED_GATE_V1_SHA256
    )
    assert (
        r4.sha256_file(r4.R3_ROOT / "run_authority/development_validation_manifest.json")
        == r4.EXPECTED_R3_MANIFEST_SHA256
    )


def test_historical_r3_fail_is_immutable() -> None:
    summary = r4.read_json(r4.R3_ROOT / "final_summary.json")
    decision = r4.read_json(r4.R3_ROOT / "gate/decision.json")
    assert summary["D3_R3_STATUS"] == "FAIL_DEVELOPMENT_GATE"
    assert summary["REFINEMENT_V2_DEVELOPMENT_VALIDATION"] == "FAIL"
    assert decision["G3_MEDIAN_REDUCTION"] == "FAIL"


def test_action_surface_has_no_optimizer_or_fresh_execution() -> None:
    assert not (set(r4.ACTIONS) & r4.FORBIDDEN_EXECUTION_ACTIONS)
    source = Path(r4.__file__).read_text(encoding="utf-8")
    assert "search_frame(" not in source
    assert "D2ARuntime(" not in source


def test_authority_enums_are_exhaustive() -> None:
    assert all(row["authority_type"] in r4.AUTHORITY_TYPES for row in r4.gate_v1_criteria())
    assert "CONSERVATIVE_PREREGISTERED_HEURISTIC" in r4.MEDIAN_AUTHORITY_TYPES


def test_paper_absence_is_not_contradiction(tmp_path: Path) -> None:
    result = r4.audit_paper_authority(tmp_path)
    assert result["PAPER_DOES_NOT_DEFINE_THIS_GATE"] is True
    assert result["paper_rejects_this_gate"] is False
    assert result["PAPER_SUPPORTS_50_PERCENT_MEDIAN_REDUCTION_GATE"] == "NO"


def test_semantic_v1_threshold_is_read_from_frozen_contract(tmp_path: Path) -> None:
    result = r4.audit_semantic_v1_authority(tmp_path)
    assert result["contract"]["interaction_e_im_p95_limit"] == pytest.approx(1e-4)
    assert result["SEMANTIC_V1_HAS_RELATIVE_REDUCTION_GATE"] == "NO"
    assert result["SEMANTIC_V1_HAS_50_PERCENT_REDUCTION_GATE"] == "NO"


def test_objective_v2_has_zero_pressure_at_tau_and_no_half_tau_target(tmp_path: Path) -> None:
    result = r4.audit_objective_v2_authority(tmp_path)
    assert result["primary_pressure_at_or_below_tau"] == 0.0
    assert result["requires_tau_over_2"] is False
    assert result["OBJECTIVE_V2_SUPPORTS_50_PERCENT_REDUCTION_GATE"] == "NO"


def test_recoverability_cannot_invent_posthoc_threshold(tmp_path: Path) -> None:
    result = r4.audit_recoverability_authority(tmp_path)
    assert result["pre_registered_mapping_median_50_percent_to_recoverability"] is False
    assert result["post_hoc_threshold_invention_allowed"] is False


def test_gate_v2_cannot_be_designed_before_authority_decision(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MISSING"):
        r4.design_gate_v2_if_authorized(tmp_path)


def test_gate_v2_cannot_be_frozen_before_all_authority_audits(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MISSING"):
        r4.freeze_gate_v2(tmp_path)


def test_gate_v2_does_not_use_r3_outcome_as_threshold_source() -> None:
    gate = r4.gate_v2_contract()
    assert gate["R3_outcome_used_as_threshold_source"] is False
    assert gate["median_invalid_relative_E_IM_reduction"] == "DIAGNOSTIC_ONLY"
    assert gate["tau"] == pytest.approx(1e-4)
    assert gate["old_invalid_recovery_rate_minimum"] == pytest.approx(0.8)


def test_gate_v2_serialization_is_deterministic() -> None:
    gate = r4.gate_v2_contract()
    assert r4.canonical_sha(gate) == r4.canonical_sha(json.loads(json.dumps(gate)))


def test_stored_compatibility_cannot_run_before_freeze(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="GATE_V2_NOT_FROZEN"):
        r4.evaluate_stored_r3_gate_v2_compatibility(tmp_path)


def test_fresh_certification_cannot_start_before_freeze(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="PREREQUISITES_MISSING"):
        r4.authorize_fresh_certification(tmp_path)


def test_historical_r3_cannot_become_pass_under_compatibility_label() -> None:
    source = Path(r4.__file__).read_text(encoding="utf-8")
    assert '"EVIDENCE_ROLE": "POST_HOC_COMPATIBILITY_ONLY"' in source
    assert '"R3_REEVALUATED_PASS": False' in source
    assert '"HISTORICAL_R3_GATE_V1_RESULT": "FAIL"' in source


def test_no_d3v2_execution_surface() -> None:
    assert "run-d3v2" not in r4.ACTIONS
    assert r4.gate_v2_contract()["trajectory_semantic_v1_role"].startswith("must be established")


def test_refinement_v2_design_hash_remains_unchanged() -> None:
    assert (
        r4.sha256_file(r4.R2_ROOT / "design/refinement_v2_design.json") == r4.EXPECTED_DESIGN_SHA256
    )


def test_effect_size_analysis_uses_all_stored_frames(tmp_path: Path) -> None:
    write_json(tmp_path / "preflight/historical_integrity.json", {"status": "PASS"})
    result = r4.analyze_effect_size_semantics(tmp_path)
    assert result["N"] == 145
    assert result["recovered_count"] == 143
    assert result["r_min_max"] < 0.2
    assert result["fraction_r_min_lt_0p5"] == 1.0


def test_remaining_frames_are_threshold_aware_unrecovered_not_special_cases(tmp_path: Path) -> None:
    write_json(tmp_path / "preflight/historical_integrity.json", {"status": "PASS"})
    r4.analyze_effect_size_semantics(tmp_path)
    comparison = r4.read_json(tmp_path / "effect_size/comparison.json")
    assert [row["source_frame"] for row in comparison["remaining_invalid_frames"]] == [4202, 4209]
    assert "no special case" in comparison["remaining_classification_under_threshold_aware_gate"]


def test_cli_help_lists_real_contract_actions() -> None:
    help_text = r4.parser().format_help()
    for action in (
        "preflight",
        "verify-r3-history",
        "verify-gate-v1-authority",
        "trace-gate-v1-provenance",
        "audit-paper-authority",
        "audit-semantic-v1-authority",
        "audit-objective-v2-authority",
        "audit-physical-authority",
        "audit-recoverability-authority",
        "analyze-effect-size-semantics",
        "classify-median-reduction-authority",
        "decide-gate-semantic-alignment",
        "design-gate-v2-if-authorized",
        "freeze-gate-v2",
        "evaluate-stored-r3-gate-v2-compatibility",
        "authorize-fresh-certification",
        "generate-fresh-certification-plan",
        "summarize",
    ):
        assert action in help_text
