from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.objective_v3_execution import (
    ExecutionFrameInputsV3,
    ExecutionInputAuthorityError,
    RetargetMode,
    cold_start_seeds_v3,
    default_execution_v3_candidates,
    refinement_seeds_v3,
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
