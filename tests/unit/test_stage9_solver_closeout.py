from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.final_refinement import (
    STRICT_ACCEPTANCE_POLICY_ID,
    CollisionQuerySet,
    RefinementSolverProfile,
    active_set_is_monotonic,
    continue_active_set_initial,
    strict_acceptance_decision,
)


def _query_set(ids: list[int]) -> CollisionQuerySet:
    return CollisionQuerySet(
        np.asarray(ids, dtype=np.int64),
        tuple("test" for _ in ids),
        np.arange(len(ids), dtype=np.int64),
        np.zeros(len(ids), dtype=np.float64),
        "test",
    )


def test_active_set_continuation_copies_base_q_and_remaps_old_slack() -> None:
    previous = _query_set([1, 3])
    expanded = _query_set([3, 0, 1, 2])
    result_x = np.asarray([10.0, 11.0, 12.0, 0.11, 0.22])
    continued = continue_active_set_initial(
        result_x,
        previous,
        expanded,
        new_query_ids=np.asarray([0, 2]),
        signed_distance=np.asarray([-0.004, 0.9, -0.02, 0.8]),
        tau=0.001,
        b=0.03,
    )
    assert np.array_equal(continued[:3], result_x[:3])
    assert np.array_equal(continued[3:], [0.22, 0.003, 0.11, 0.019])


def test_active_set_continuation_new_slack_is_minimum_bounded_feasible_value() -> None:
    previous = _query_set([4])
    expanded = _query_set([4, 2])
    continued = continue_active_set_initial(
        np.asarray([1.0, 0.7]),
        previous,
        expanded,
        new_query_ids=np.asarray([2]),
        signed_distance=np.asarray([0.0, 0.0, -0.04, 0.0, 0.02]),
        tau=0.001,
        b=0.03,
    )
    assert continued[-1] == pytest.approx(0.029)


def test_active_set_growth_is_monotonic_and_reorder_safe() -> None:
    previous = _query_set([5, 7])
    reordered = _query_set([7, 2, 5])
    assert active_set_is_monotonic(previous, reordered)
    assert not active_set_is_monotonic(_query_set([5, 7]), _query_set([5]))


def test_strict_status_9_rejects_feasible_candidate() -> None:
    result = strict_acceptance_decision(
        optimizer_converged=False,
        optimizer_status_code=9,
        qpos_bounds_pass=True,
        slack_bounds_pass=True,
        active_constraints_feasible=True,
        full_surface_hard_audit_pass=True,
        full_surface_soft_audit_pass=True,
        active_set_converged=True,
        all_values_finite=True,
    )
    assert result["accepted"] is False
    assert "optimizer_converged" in result["acceptance_reason"]


def test_feasible_and_converged_are_separate_contract_fields() -> None:
    result = strict_acceptance_decision(
        optimizer_converged=True,
        optimizer_status_code=0,
        qpos_bounds_pass=True,
        slack_bounds_pass=True,
        active_constraints_feasible=True,
        full_surface_hard_audit_pass=True,
        full_surface_soft_audit_pass=True,
        active_set_converged=True,
        all_values_finite=True,
    )
    assert result["accepted"] is True
    assert result["acceptance_policy_id"] == STRICT_ACCEPTANCE_POLICY_ID


def test_v1_profile_and_frozen_continuation_behavior_remain_distinct() -> None:
    v1 = RefinementSolverProfile.load("scipy_slsqp_active_set_v1")
    v2 = RefinementSolverProfile.load("scipy_slsqp_active_set_contact_rich_v2")
    assert v1.maxiter == 30
    assert v1.active_set_continuation_policy == "warm_seed_reinitialized_v1"
    assert v2.profile_id != v1.profile_id
    assert v2.active_set_continuation_policy == "result_x_query_id_slack_remap_v2"
    assert v1.profile_hash != v2.profile_hash


def test_v2_profile_declares_hash_termination_and_benchmark_contract() -> None:
    profile = RefinementSolverProfile.load("scipy_slsqp_active_set_contact_rich_v2")
    assert profile.profile_hash
    assert profile.version == "2.0.0"
    assert profile.termination_contract
    assert profile.maxiter_provenance["source"] == "stage9_1_fixed_benchmark_grid"
    assert profile.benchmark_grid == (30, 60, 100, 200, 400)
    assert profile.acceptance_policy_id == STRICT_ACCEPTANCE_POLICY_ID
