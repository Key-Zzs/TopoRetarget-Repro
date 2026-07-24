from pathlib import Path

from toporetarget.workflows.stage9_4 import (
    ABLATION_PROFILES,
    build_regularization_code_map,
    classify_projection_state,
    select_lowest_formal_candidate,
    temporal_indices,
    temporal_scope_for_profile,
)


def test_stage9_4_declares_the_fixed_c0_c7_profiles_and_frames() -> None:
    assert len(ABLATION_PROFILES) == 8
    assert ABLATION_PROFILES[0] == "faithful_current_baseline"
    assert ABLATION_PROFILES[-1] == "projection_or_warm_initialized_faithful"


def test_temporal_membership_is_explicit_and_non_overlapping() -> None:
    base = set(temporal_indices("base_only"))
    finger = set(temporal_indices("finger_only"))
    both = set(temporal_indices("base_and_finger"))
    assert base == set(range(6))
    assert finger == set(range(6, 28))
    assert base.isdisjoint(finger)
    assert both == base | finger
    assert temporal_indices("none") == ()


def test_repair_profile_uses_finger_temporal_term_only() -> None:
    assert temporal_scope_for_profile("faithful_current_baseline") == "base_and_finger"
    assert temporal_scope_for_profile("faithful_regularization_fix_v1") == "finger_only"


def test_projection_state_contract_accepts_only_declared_states() -> None:
    assert (
        classify_projection_state({"diagnostic_status": "ANALYTIC_IDENTITY_PROJECTION"})
        == "ANALYTIC_IDENTITY_PROJECTION"
    )
    assert classify_projection_state({"strict_projection_acceptance": True}) == (
        "SOLVED_AND_VALIDATED"
    )
    assert classify_projection_state({"projection_feasible": True}) == ("FEASIBLE_UPPER_BOUND_ONLY")
    assert classify_projection_state({}) == "INVALID_CONTRACT"


def test_formal_candidate_selection_rejects_nonaccepted_candidates() -> None:
    candidates = [
        {"strict_accepted": False, "formal_objective": 0.0},
        {"strict_accepted": True, "formal_objective": 3.0, "label": "high"},
        {"strict_accepted": True, "formal_objective": 1.0, "label": "low"},
    ]
    assert select_lowest_formal_candidate(candidates)["label"] == "low"
    assert select_lowest_formal_candidate([candidates[0]]) is None


def test_eq9_code_map_records_the_base_prior_and_mapping_contract() -> None:
    code_map = build_regularization_code_map(Path.cwd())
    terms = {entry["term"] for entry in code_map["entries"]}
    assert {
        "E_temporal_q",
        "E_base_position_prior",
        "E_base_rotation_prior",
        "E_IM_and_Ebone_reduction",
    } <= terms
    assert code_map["audit_answers"]["previous_final_mapping"]
    assert code_map["audit_answers"]["base_duplicate_regularization"] is True
    assert code_map["diagnostic_only"] is True
    assert code_map["paper_method"] is False
