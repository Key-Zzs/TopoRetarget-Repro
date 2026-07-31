from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from toporetarget.retarget.final_refinement import (
    FINAL_REFINEMENT_SCHEMA_VERSION_V2,
    CollisionQuerySet,
    FinalRetargetTrajectory,
    RefinementSolverProfile,
    active_set_is_monotonic,
    continue_active_set_initial,
    restore_slack_feasibility,
    strict_acceptance_decision,
    strict_recovery_trigger,
)
from toporetarget.retarget.solver_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    choose_uniform_maxiter,
    validate_benchmark_report,
)


def _query(ids: list[int]) -> CollisionQuerySet:
    return CollisionQuerySet(
        np.asarray(ids, dtype=np.int64),
        tuple(f"q{item}" for item in ids),
        np.zeros(len(ids), dtype=np.int64),
        np.zeros(len(ids), dtype=np.float64),
        "hash",
    )


def test_result_x_continuation_preserves_base_q_and_old_slack() -> None:
    previous = _query([4, 9])
    expanded = _query([1, 4, 7, 9])
    result_x = np.asarray([0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.007, 0.019])
    initial = continue_active_set_initial(
        result_x,
        previous,
        expanded,
        new_query_ids=np.asarray([1, 7]),
        signed_distance=np.asarray([0.0, -0.004, 0.0, 0.0, 0.01, 0.0, 0.0, -0.02, 0.0, -0.003]),
        tau=0.001,
        b=0.030,
    )
    assert np.array_equal(initial[:6], result_x[:6])
    # Query IDs were reordered; q4/q9 still receive exactly their old values.
    assert np.array_equal(initial[[1 + 6, 3 + 6]], np.asarray([0.007, 0.019]))
    assert initial[0 + 6] == 0.003
    assert initial[2 + 6] == 0.019


def test_query_reorder_and_active_set_growth_are_monotonic() -> None:
    previous = _query([2, 8])
    expanded = _query([1, 2, 8, 10])
    assert active_set_is_monotonic(previous, expanded)
    assert not active_set_is_monotonic(expanded, previous)


def test_strict_status_9_rejects_feasible_candidate() -> None:
    common = {
        "optimizer_converged": True,
        "qpos_bounds_pass": True,
        "slack_bounds_pass": True,
        "active_constraints_feasible": True,
        "full_surface_hard_audit_pass": True,
        "full_surface_soft_audit_pass": True,
        "active_set_converged": True,
        "all_values_finite": True,
    }
    rejected = strict_acceptance_decision(**common, optimizer_status_code=9)
    assert rejected["accepted"] is False
    assert "optimizer_converged" in rejected["acceptance_reason"]
    accepted = strict_acceptance_decision(**common, optimizer_status_code=0)
    assert accepted["accepted"] is True


def test_feasible_but_not_converged_is_distinct_from_acceptance() -> None:
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
    assert result["checks"]["active_constraints_feasible"] is True
    assert result["checks"]["full_surface_hard_audit_pass"] is True
    assert result["accepted"] is False


def test_status_8_reference_recovery_is_generic_and_bound_checked() -> None:
    context = SimpleNamespace(
        robot_model=SimpleNamespace(
            joint_lower=np.asarray([-1.0, -2.0]),
            joint_upper=np.asarray([1.0, 2.0]),
        ),
        paper=SimpleNamespace(b=0.03, tau=0.001),
        unpack=lambda value: (value[:3], value[3:6], value[6:8], value[8:]),
    )
    value = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.5, 0.001])
    allowed, trace = strict_recovery_trigger(
        status_code=8,
        recovery_policy="reference_batched_from_primary_result_v1",
        value=value,
        context=context,
    )
    assert allowed is True
    assert trace["recoverable_status"] is True
    assert trace["candidate_q_bounds_pass"] is True
    assert trace["candidate_slack_bounds_pass"] is True

    nonrecoverable, _ = strict_recovery_trigger(
        status_code=7,
        recovery_policy="reference_batched_from_primary_result_v1",
        value=value,
        context=context,
    )
    out_of_bounds, bounds_trace = strict_recovery_trigger(
        status_code=8,
        recovery_policy="reference_batched_from_primary_result_v1",
        value=np.asarray([*value[:-1], 0.04]),
        context=context,
    )
    assert nonrecoverable is False
    assert out_of_bounds is False
    assert bounds_trace["candidate_slack_bounds_pass"] is False


def test_slack_feasibility_restoration_only_adjusts_representable_slack() -> None:
    context = SimpleNamespace(
        paper=SimpleNamespace(b=0.03, tau=0.001),
        unpack=lambda value: (value[:3], value[3:6], value[6:7], value[7:]),
        constraint_query=lambda *_args: SimpleNamespace(signed_distance=np.asarray([-0.001001])),
    )
    restored, trace = restore_slack_feasibility(context, np.zeros(8, dtype=np.float64), _query([0]))
    assert trace["hard_feasible"] is True
    assert trace["soft_slack_representable"] is True
    assert trace["slack_adjusted_count"] == 1
    assert restored[-1] > 0.0


def test_v1_profile_is_unchanged_and_v2_has_independent_hash() -> None:
    root = Path(__file__).resolve().parents[2]
    v1_path = root / "configs/retarget/refinement_solvers/scipy_slsqp_active_set_v1.yaml"
    v1 = RefinementSolverProfile.load("scipy_slsqp_active_set_v1", root=root)
    v2 = RefinementSolverProfile.load("scipy_slsqp_active_set_contact_rich_v2", root=root)
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == v1.profile_hash
    assert v1.maxiter == 30
    assert v1.active_set_continuation_policy == "warm_seed_reinitialized_v1"
    assert v2.profile_id != v1.profile_id
    assert v2.profile_hash != v1.profile_hash
    assert v2.active_set_continuation_policy == "result_x_query_id_slack_remap_v2"
    assert v2.maxiter_provenance["source"] == "stage9_1_fixed_benchmark_grid"


def test_v2_artifact_contains_termination_contract_fields() -> None:
    arrays = {
        "qpos": np.zeros((1, 22)),
        "timestamps": np.zeros(1),
        "base_pose_scene": np.eye(4)[None],
        "base_corrections": np.zeros((1, 6)),
        "robot_keypoints_base": np.zeros((1, 21, 3)),
        "robot_keypoints_scene": np.zeros((1, 21, 3)),
        "collision_points_scene": np.zeros((1, 512, 3)),
        "slack_concat": np.zeros(1),
        "query_offsets": np.asarray([0, 1]),
        "full_signed_distance": np.ones((1, 512)),
        "full_closest_points": np.zeros((1, 512, 3)),
        "full_surface_normals": np.zeros((1, 512, 3)),
        "full_hard_residual": np.ones((1, 512)),
        "full_soft_violation_count": np.zeros(1, dtype=np.int64),
        "unqueried_soft_violation_count": np.zeros(1, dtype=np.int64),
        "active_set_converged": np.ones(1, dtype=bool),
        "robot_link_poses": np.eye(4)[None, None],
        "valid_mask": np.ones(1, dtype=bool),
    }
    for name, dtype in {
        "optimizer_converged": bool,
        "optimizer_status_code": np.int64,
        "optimizer_message": "S256",
        "optimizer_iterations": np.int64,
        "optimizer_function_evaluations": np.int64,
        "optimizer_jacobian_evaluations": np.int64,
        "qpos_bounds_pass": bool,
        "slack_bounds_pass": bool,
        "active_constraints_feasible": bool,
        "full_surface_hard_audit_pass": bool,
        "full_surface_soft_audit_pass": bool,
        "all_values_finite": bool,
        "stationarity_checked": bool,
        "stationarity_residual": np.float64,
        "accepted": bool,
        "acceptance_reason": "S512",
        "initial_objective": np.float64,
        "final_objective": np.float64,
        "final_objective_change": np.float64,
        "final_step_norm": np.float64,
    }.items():
        arrays[name] = np.asarray([False if dtype is bool else 0], dtype=dtype)
    trajectory = FinalRetargetTrajectory(
        {
            "schema_version": FINAL_REFINEMENT_SCHEMA_VERSION_V2,
            "solver_profile_id": "scipy_slsqp_active_set_contact_rich_v2",
        },
        arrays,
    )
    assert trajectory.validate().frame_count == 1


def _record(case_id: str, budget: int, passed: bool) -> dict[str, object]:
    return {
        "case_id": case_id,
        "budget": budget,
        "result_success": passed,
        "status_code": 0 if passed else 9,
        "message": "ok" if passed else "Iteration limit reached",
        "nit": budget,
        "nfev": 1,
        "njev": 1,
        "initial_objective": 1.0,
        "final_objective": 0.5,
        "final_objective_change": 0.5,
        "final_step_norm": 0.0,
        "min_hard_residual_m": 0.1,
        "min_soft_residual_m": 0.1,
        "full_surface_min_signed_distance_m": 0.1,
        "active_set_rounds": 1,
        "runtime_s": 0.1,
        "strict_acceptance": passed,
        "independent_full_surface_audit_pass": passed,
        "deterministic_repeat": passed,
    }


def test_maxiter_report_selects_minimum_uniform_budget() -> None:
    records = [
        _record("a", 30, False),
        _record("b", 30, True),
        _record("a", 60, True),
        _record("b", 60, True),
    ]
    assert choose_uniform_maxiter(records, (30, 60), case_ids=("a", "b")) == 60
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "budget_grid": [30, 60],
        "selected_maxiter": 60,
        "records": records,
    }
    assert validate_benchmark_report(payload)["selected_maxiter"] == 60
