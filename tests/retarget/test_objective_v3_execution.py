from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.objective_v3_execution import (
    ColdStartSearchV2Candidate,
    ExecutionBaselineAuthority,
    ExecutionFrameInputsV3,
    ExecutionInputAuthorityError,
    RetargetMode,
    cold_start_seeds_v3,
    default_cold_start_search_v2_candidates,
    default_execution_v3_candidates,
    refinement_seeds_v3,
    screen_whole_hand_bootstrap_states,
)


def _inputs(mode: RetargetMode, *, step: int = 0) -> ExecutionFrameInputsV3:
    previous_q = None if step == 0 else np.full(4, 0.4)
    previous_base = None if step == 0 else np.eye(4)
    return ExecutionFrameInputsV3(
        mode=mode,
        runtime_step_index=step,
        old_production_q=np.full(4, 0.3) if mode is RetargetMode.REFINEMENT else None,
        previous_accepted_q=previous_q,
        previous_accepted_base=previous_base,
    )


def test_mode_authority_is_explicit_and_fail_closed() -> None:
    with pytest.raises(ExecutionInputAuthorityError, match="REFINEMENT_REQUIRES"):
        ExecutionFrameInputsV3(RetargetMode.REFINEMENT, 0, None, None, None).validate()
    with pytest.raises(ExecutionInputAuthorityError, match="COLD_START_FORBIDS"):
        ExecutionFrameInputsV3(RetargetMode.COLD_START, 0, np.zeros(4), None, None).validate()
    _inputs(RetargetMode.COLD_START).validate()


def test_frame_zero_and_runtime_previous_are_distinct_authorities() -> None:
    with pytest.raises(ExecutionInputAuthorityError, match="frame zero forbids"):
        ExecutionFrameInputsV3(RetargetMode.COLD_START, 0, None, np.zeros(4), np.eye(4)).validate()
    _inputs(RetargetMode.COLD_START, step=1).validate()


def test_t_gt_zero_runtime_state_is_not_old_production_authority() -> None:
    inputs = _inputs(RetargetMode.COLD_START, step=1).validate()
    assert inputs.old_production_q is None
    assert inputs.previous_accepted_q is not None
    assert (
        ExecutionBaselineAuthority.PREVIOUS_ACCEPTED_RUNTIME_STATE
        is not ExecutionBaselineAuthority.OLD_PRODUCTION_TRAJECTORY
    )


def test_cold_start_never_aliases_q_old() -> None:
    lower, upper = np.zeros(4), np.ones(4)
    seeds = cold_start_seeds_v3(
        candidate=default_execution_v3_candidates()[1],
        frame_inputs=_inputs(RetargetMode.COLD_START, step=1),
        neutral_q=np.full(4, 0.2),
        lower_q=lower,
        upper_q=upper,
    )
    assert [name for name, _ in seeds] == [
        "wuji_canonical_rest",
        "joint_range_midpoint",
        "previous_accepted_runtime",
    ]
    assert all(name != "old_production" for name, _ in seeds)
    assert not np.shares_memory(seeds[0][1], seeds[-1][1])


def test_cold_start_frame_zero_omits_previous_seed() -> None:
    candidate = default_execution_v3_candidates()[1]
    seeds = cold_start_seeds_v3(
        candidate=candidate,
        frame_inputs=_inputs(RetargetMode.COLD_START),
        neutral_q=np.full(4, -0.25),
        lower_q=np.full(4, -1.0),
        upper_q=np.ones(4),
    )
    assert [name for name, _ in seeds] == ["wuji_canonical_rest", "joint_range_midpoint"]


def test_source_geometric_authority_is_required_for_candidate_c() -> None:
    with pytest.raises(ExecutionInputAuthorityError, match="source_geometric"):
        cold_start_seeds_v3(
            candidate=default_execution_v3_candidates()[2],
            frame_inputs=_inputs(RetargetMode.COLD_START),
            neutral_q=np.zeros(4),
            lower_q=np.full(4, -1.0),
            upper_q=np.ones(4),
        )


def test_refinement_delegates_to_frozen_s1_seed_order() -> None:
    seeds = refinement_seeds_v3(
        frame_inputs=_inputs(RetargetMode.REFINEMENT),
        neutral_q=np.full(4, 0.2),
        lower_q=np.zeros(4),
        upper_q=np.ones(4),
        block=(0, 1),
    )
    assert [name for name, _ in seeds] == [
        "old_production",
        "wuji_canonical_rest",
        "joint_range_midpoint",
    ]


def test_candidates_are_generic_and_contain_no_episode_literals() -> None:
    payload = repr([candidate.as_dict() for candidate in default_execution_v3_candidates()])
    for forbidden in ("C11001", "10704", "DEV2", "thumb"):
        assert forbidden not in payload


def test_search_v2_candidate_is_full_hand_qold_free_and_budget_frozen() -> None:
    (candidate,) = default_cold_start_search_v2_candidates()
    assert candidate.validate() is candidate
    assert candidate.bootstrap_free_dofs == "ALL_20_FINGER_DOFS"
    assert candidate.post_bootstrap_contributor_ranking
    assert candidate.top_k == 1
    assert candidate.bootstrap_max_nfev == 250
    assert candidate.contributor_probe_max_nfev == 24
    assert all("old" not in source for source in candidate.bootstrap_seed_sources)
    with pytest.raises(ValueError, match="budgets must remain frozen"):
        ColdStartSearchV2Candidate(contributor_probe_max_nfev=25).validate()


def test_whole_hand_bootstrap_screen_is_deterministic_and_complete() -> None:
    lower = np.full(20, -1.0)
    upper = np.ones(20)
    first = np.linspace(-0.5, 0.5, 20)
    duplicate = first.copy()
    accepted = screen_whole_hand_bootstrap_states(
        states=(("first", first), ("duplicate", duplicate)),
        lower_q=lower,
        upper_q=upper,
        expected_dofs=20,
    )
    assert [name for name, _q in accepted] == ["first"]
    assert accepted[0][1].shape == (20,)
    assert np.all(np.isfinite(accepted[0][1]))
    with pytest.raises(ValueError, match="full hand"):
        screen_whole_hand_bootstrap_states(
            states=(("local_only", np.zeros(4)),),
            lower_q=lower,
            upper_q=upper,
            expected_dofs=20,
        )


def test_whole_hand_bootstrap_rejects_invalid_states_without_fabrication() -> None:
    lower = np.full(20, -1.0)
    upper = np.ones(20)
    with pytest.raises(ExecutionInputAuthorityError, match="NO_VALID_COLDSTART"):
        screen_whole_hand_bootstrap_states(
            states=(
                ("nan", np.full(20, np.nan)),
                ("out_of_bounds", np.full(20, 2.0)),
            ),
            lower_q=lower,
            upper_q=upper,
            expected_dofs=20,
        )
