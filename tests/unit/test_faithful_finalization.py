import json
from pathlib import Path

import pytest
import yaml

from toporetarget.retarget.final_refinement import regularization_profile_for_solver
from toporetarget.workflows.faithful_finalization import (
    FINALIZATION_BOOLEAN_CHECKS,
    REQUIRED_REVIEW_FRAMES,
    _decision_policy,
    _paper_fidelity_statement,
    build_manual_acceptance_template,
    validate_finalization_manual_acceptance,
)
from toporetarget.workflows.validation import validate_manual_acceptance


def test_profile_classification_separates_legacy_and_faithful() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (root / "configs/retarget/finalization/faithful_reproduction_profiles.yaml").read_text(
            encoding="utf-8"
        )
    )
    legacy = payload["legacy_profile"]
    faithful = payload["faithful_profile"]
    assert legacy["id"] == "scipy_slsqp_active_set_contact_rich_v2"
    assert legacy["paper_faithful"] is False
    assert legacy["status"] == "historical_accepted"
    assert "base correction included in temporal regularization" in legacy["known_deviation"]
    assert faithful["id"] == "scipy_slsqp_active_set_contact_rich_v3_fixed"
    assert faithful["paper_faithful"] is True
    assert faithful["canonical_faithful"] is True
    assert faithful["status"] == "validated_quality_neutral"
    assert regularization_profile_for_solver(faithful["id"]) == "faithful_regularization_fix_v1"
    assert regularization_profile_for_solver(legacy["id"]) == "faithful_current_baseline"


def test_manual_template_covers_required_and_worst_frames_without_faking_human_pass(
    tmp_path: Path,
) -> None:
    payload = build_manual_acceptance_template()
    assert set(REQUIRED_REVIEW_FRAMES).issubset(payload["reviewed_frames"])
    assert payload["status"] == "pending_human_review"
    assert payload["reviewer"] == ""
    assert payload["decision_rationale"] is None
    assert payload["decision_evidence_frames"] == []
    path = tmp_path / "manual_acceptance.template.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="status must be pass"):
        validate_manual_acceptance(path)


def test_finalization_acceptance_requires_all_case_a_visual_checks(tmp_path: Path) -> None:
    payload = build_manual_acceptance_template()
    payload.update(
        {
            "status": "pass",
            "reviewer": "human",
            "contact_rich_clip_validated": True,
            "decision_case": "A",
        }
    )
    for name in FINALIZATION_BOOLEAN_CHECKS:
        payload[name] = True
    path = tmp_path / "manual_acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_finalization_manual_acceptance(path)["decision_case"] == "A"
    payload["thumb_opposition_preserved"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="every finalization visual check"):
        validate_finalization_manual_acceptance(path)


@pytest.mark.parametrize(
    ("case", "status", "quality_improved", "production_recommended"),
    [
        ("A", "FAITHFUL_REPRODUCTION_FINALIZED_CASE_A", False, True),
        ("B", "FAITHFUL_REPRODUCTION_FINALIZED_CASE_B", False, False),
        ("C", "FAITHFUL_REPRODUCTION_FINALIZED_CASE_C", True, True),
    ],
)
def test_decision_policy_preserves_a_b_c_semantics(
    case: str,
    status: str,
    quality_improved: bool,
    production_recommended: bool,
) -> None:
    root = Path(__file__).resolve().parents[2]
    profile = yaml.safe_load(
        (root / "configs/retarget/finalization/faithful_reproduction_profiles.yaml").read_text(
            encoding="utf-8"
        )
    )
    decision = _decision_policy(profile, {"decision_case": case})
    assert decision["case"] == case
    assert decision["finalization_status"] == status
    assert decision["quality_improvement_claimed"] is quality_improved
    assert (decision["production_recommended_profile"] is not None) is production_recommended
    statement = _paper_fidelity_statement(profile, decision)
    assert f"decision case {case}" in statement
    if case == "B":
        assert "not recommended for production" in statement


def test_case_b_requires_regression_rationale_and_evidence_frames(tmp_path: Path) -> None:
    payload = build_manual_acceptance_template()
    payload.update(
        {
            "status": "pass",
            "reviewer": "human",
            "contact_rich_clip_validated": True,
            "decision_case": "B",
        }
    )
    for name in FINALIZATION_BOOLEAN_CHECKS:
        payload[name] = True
    path = tmp_path / "manual_acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="at least one recorded visual regression"):
        validate_finalization_manual_acceptance(path)
    payload["index_middle_surface_relation_preserved"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision_rationale"):
        validate_finalization_manual_acceptance(path)
    payload["decision_rationale"] = "Fixed index visibly separates from the object."
    payload["decision_evidence_frames"] = [36]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_finalization_manual_acceptance(path)["decision_case"] == "B"


def test_case_c_requires_all_checks_rationale_and_evidence_frames(tmp_path: Path) -> None:
    payload = build_manual_acceptance_template()
    payload.update(
        {
            "status": "pass",
            "reviewer": "human",
            "contact_rich_clip_validated": True,
            "decision_case": "C",
            "decision_rationale": "Fixed thumb contact is visibly more stable.",
            "decision_evidence_frames": [30, 36],
        }
    )
    for name in FINALIZATION_BOOLEAN_CHECKS:
        payload[name] = True
    path = tmp_path / "manual_acceptance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_finalization_manual_acceptance(path)["decision_case"] == "C"
    payload["thumb_opposition_preserved"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="every finalization visual check"):
        validate_finalization_manual_acceptance(path)
