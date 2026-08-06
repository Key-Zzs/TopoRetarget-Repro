from __future__ import annotations

import inspect

import pytest
import torch

from toporetarget.rl.physics_retargeting.contracts import PhysicsConsistentTaskGateV1
from toporetarget.rl.physics_retargeting.recovery import Stage16DRecoveryStateMachine
from toporetarget.rl.physics_retargeting.rewards import (
    PhysicsConsistentRewardProfileV1,
    physics_consistent_reward_terms,
)
from toporetarget.rl.physics_retargeting.robust_optimizer import (
    PhaseWiseRobustSplineCEMV1,
    PhaseWiseSplineCEMConfigV1,
    PhysicsCandidateEvaluationV1,
    PhysicsCandidateReplicaV1,
)
from toporetarget.rl.physics_retargeting.spline_actions import PiecewiseSplineResidualV1
from toporetarget.rl.physics_retargeting.termination import physics_consistent_termination


def _reward_metrics() -> dict[str, torch.Tensor]:
    names = (
        "wrist_fidelity",
        "finger_fidelity",
        "link_fidelity",
        "contact_coverage",
        "contact_persistence",
        "contact_onset_alignment",
        "final_topology",
        "forbidden_contact",
        "penetration_m",
        "impulse_outlier",
        "object_instability",
        "action_effort",
        "action_first_difference",
        "action_second_difference",
        "semantic_progress",
        "relative_pose_progress",
        "source_object_soft_prior",
        "terminal_success",
        "catastrophic_failure",
    )
    return {name: torch.ones(2) for name in names}


def _gate() -> PhysicsConsistentTaskGateV1:
    return PhysicsConsistentTaskGateV1(
        clip="synthetic",
        object_bbox_diagonal_m=0.1,
        minimum_contact_recall=0.5,
        minimum_semantic_progress=0.3,
        minimum_object_motion_m=0.01,
        minimum_object_rotation_deg=0.0,
        terminal_window_control_steps=5,
        workspace_radius_m=0.5,
    )


def test_reward_prefers_semantic_progress_and_contact() -> None:
    profile = PhysicsConsistentRewardProfileV1()
    good = _reward_metrics()
    bad = _reward_metrics()
    bad["semantic_progress"] = torch.zeros(2)
    bad["contact_coverage"] = torch.zeros(2)
    assert torch.all(
        physics_consistent_reward_terms(good, profile)["total"]
        > physics_consistent_reward_terms(bad, profile)["total"]
    )


def test_termination_has_no_strict_object_tracking_input() -> None:
    signature = inspect.signature(physics_consistent_termination)
    assert "object_position_error" not in signature.parameters
    metrics = {
        "finite": torch.tensor([True]),
        "penetration_m": torch.tensor([0.0]),
        "workspace_distance_m": torch.tensor([0.1]),
        "wrist_safe": torch.tensor([True]),
        "joint_limits_safe": torch.tensor([True]),
        "action_valid": torch.tensor([True]),
        "object_speed_mps": torch.tensor([0.0]),
        "semantic_progress": torch.tensor([1.0]),
        "contact_recall": torch.tensor([1.0]),
        "contact_causality": torch.tensor([True]),
        "terminal_stable": torch.tensor([True]),
        "object_motion_m": torch.tensor([0.02]),
    }
    result = physics_consistent_termination(metrics, _gate(), final_step=torch.tensor([True]))
    assert result["success"].item()


def test_spline_is_bounded_continuous_and_complete() -> None:
    spline = PiecewiseSplineResidualV1(knot_count=16)
    knots = torch.linspace(-2.0, 2.0, 16 * 26).reshape(16, 26)
    actions = spline.materialize(knots)
    assert actions.shape == (321, 26)
    assert float(actions.abs().max()) <= 1.0
    assert float(torch.diff(actions, dim=0).abs().max()) < 0.2


def _replica(*, catastrophic: bool, progress: float) -> PhysicsCandidateReplicaV1:
    return PhysicsCandidateReplicaV1(
        catastrophic_failure=catastrophic,
        semantic_failure=progress < 0.5,
        contact_topology_failure=False,
        penetration_m=0.0,
        safety_violation=0.0,
        semantic_progress=progress,
        contact_recall=1.0,
        contact_persistence=1.0,
        terminal_stability=1.0,
        robot_fidelity_error=0.0,
        source_object_soft_prior_error=0.0,
        action_smoothness=0.0,
        effort=0.0,
    )


def test_robust_rank_prioritizes_catastrophic_probability() -> None:
    safe = PhysicsCandidateEvaluationV1(
        1, tuple(_replica(catastrophic=False, progress=0.5) for _ in range(4))
    )
    unsafe = PhysicsCandidateEvaluationV1(
        0, tuple(_replica(catastrophic=True, progress=1.0) for _ in range(4))
    )
    assert safe.lexical_key() < unsafe.lexical_key()


def test_cem_has_bounded_upgrade_and_object_is_not_a_variable() -> None:
    config = PhaseWiseSplineCEMConfigV1()
    optimizer = PhaseWiseRobustSplineCEMV1(config)
    knots, actions = optimizer.ask()
    assert knots.shape == (64, 16, 26)
    assert actions.shape == (64, 321, 26)
    assert "object" not in inspect.signature(optimizer.tell).parameters
    with pytest.raises(ValueError):
        PhaseWiseSplineCEMConfigV1(population=128, replicas=8, knot_count=32, iterations=8)


def test_recovery_budgets_and_fail_closed() -> None:
    state = Stage16DRecoveryStateMachine()
    state.transition("TASK_SEMANTICS", reason="inputs frozen")
    state.repair("TASK_CLASSIFICATION", reason="sparse contact")
    assert state.phase == "TASK_SEMANTICS"
    with pytest.raises(RuntimeError, match="STAGE16D_FAIL_CLOSED"):
        state.repair("SOURCE_HASH_DRIFT", reason="changed")
