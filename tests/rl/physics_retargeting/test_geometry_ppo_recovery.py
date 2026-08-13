"""Budget tests for Stage 16-D metric qualification and PPO recovery."""

from __future__ import annotations

import pytest

from toporetarget.rl.physics_retargeting.recovery import (
    Stage16DGeometryAndPPORecoveryStateMachine,
    Stage16DStableGraspGeometryPPORecoveryStateMachine,
)


def test_geometry_and_optimizer_budgets_are_fail_closed() -> None:
    state = Stage16DGeometryAndPPORecoveryStateMachine()
    state.register_geometry_backend("python-fcl==0.7.0.11")
    with pytest.raises(RuntimeError, match="GEOMETRY_BACKEND_BUDGET"):
        state.register_geometry_backend("another-backend")

    terminal = Stage16DGeometryAndPPORecoveryStateMachine()
    terminal.register_terminal_refinement_profile("knots8_pop96_rep4_iter8_elite12")
    with pytest.raises(RuntimeError, match="TERMINAL_REFINEMENT_PROFILE_BUDGET"):
        terminal.register_terminal_refinement_profile("a-second-profile")

    global_fallback = Stage16DGeometryAndPPORecoveryStateMachine()
    global_fallback.register_global_optimizer_upgrade(reason="T1 formal replay failed")
    with pytest.raises(RuntimeError, match="GLOBAL_OPTIMIZER_UPGRADE_BUDGET"):
        global_fallback.register_global_optimizer_upgrade(reason="not authorized")


def test_ppo_budgets_and_transition_limit() -> None:
    state = Stage16DGeometryAndPPORecoveryStateMachine()
    state.register_ppo_run("hocap_170105", seed=0, samples=67_108_864)
    state.register_ppo_run("hocap_170105", seed=1, samples=0)
    with pytest.raises(RuntimeError, match="PPO_SEED_BUDGET"):
        state.register_ppo_run("hocap_170105", seed=2, samples=0)
    with pytest.raises(RuntimeError, match="PPO_SAMPLE_BUDGET"):
        state.register_ppo_run("hocap_170650", seed=0, samples=67_108_865)

    for index in range(48):
        state.transition("CLOSEOUT", reason=f"bounded transition {index}")
    with pytest.raises(RuntimeError, match="MAJOR_TRANSITION_BUDGET"):
        state.transition("CLOSEOUT", reason="not authorized")


def test_stable_grasp_recovery_enforces_calibration_and_two_clip_prerequisites() -> None:
    state = Stage16DStableGraspGeometryPPORecoveryStateMachine()
    with pytest.raises(RuntimeError, match="C2_REQUIRES_C1"):
        state.register_calibration_level("C2")
    state.register_calibration_level("C1")
    state.register_calibration_level("C2")
    for index in range(48):
        state.register_calibration_candidate("hocap_170105", f"candidate-{index}")
    with pytest.raises(RuntimeError, match="CALIBRATION_CANDIDATE_BUDGET"):
        state.register_calibration_candidate("hocap_170105", "candidate-48")
    with pytest.raises(RuntimeError, match="TWO_CLIP_PPO_PREREQUISITES"):
        state.transition("TWO_CLIP", reason="not yet authorized")
    state.authorize_single_ppo_success("hocap_170105")
    state.authorize_single_ppo_success("hocap_170650")
    state.transition("TWO_CLIP", reason="both single policies passed")
