from __future__ import annotations

import numpy as np

from toporetarget.rl.physics_retargeting.geometry_aware_optimizer import G1_CONFIG, G2_CONFIG
from toporetarget.rl.physics_retargeting.geometry_ranking import (
    GeometryAwareCandidateEvaluationV2,
    GeometryAwareCandidateReplicaV2,
)
from toporetarget.rl.physics_retargeting.recovery import (
    Stage16DGeometryAwarePPORecoveryStateMachine,
)
from toporetarget.rl.physics_retargeting.window_selection import (
    extract_geometry_optimization_windows,
)


def _row(*, absolute_failure: bool, progress: float) -> GeometryAwareCandidateReplicaV2:
    return GeometryAwareCandidateReplicaV2(
        catastrophic_failure=False,
        absolute_geometry_failure=absolute_failure,
        relative_geometry_failure=absolute_failure,
        semantic_failure=False,
        contact_topology_failure=False,
        terminal_stability_failure=False,
        contact_causality_failure=False,
        max_penetration_m=0.004 if absolute_failure else 0.001,
        active_p95_penetration_m=0.0031 if absolute_failure else 0.0005,
        contact_coverage=1.0,
        contact_persistence=1.0,
        semantic_progress=progress,
        robot_fidelity_error=0.0,
        source_object_soft_prior_error=0.0,
        action_smoothness=0.0,
        effort=0.0,
    )


def test_hard_geometry_gate_ranks_before_progress() -> None:
    unsafe = GeometryAwareCandidateEvaluationV2(0, (_row(absolute_failure=True, progress=100.0),))
    safe = GeometryAwareCandidateEvaluationV2(1, (_row(absolute_failure=False, progress=0.1),))
    assert safe.lexical_key() < unsafe.lexical_key()


def test_frozen_g1_g2_budgets() -> None:
    assert (
        G1_CONFIG.knots,
        G1_CONFIG.population,
        G1_CONFIG.replicas,
        G1_CONFIG.iterations,
    ) == (16, 96, 4, 8)
    assert (
        G2_CONFIG.knots,
        G2_CONFIG.population,
        G2_CONFIG.replicas,
        G2_CONFIG.iterations,
    ) == (32, 96, 8, 8)


def test_automatic_windows_merge_overlap_with_margin() -> None:
    geometry = np.zeros(321, dtype=bool)
    terminal = np.zeros(321, dtype=bool)
    geometry[100:110] = True
    terminal[118:125] = True
    windows = extract_geometry_optimization_windows({"geometry": geometry, "terminal": terminal})
    assert len(windows) == 1
    assert windows[0].start == 90
    assert windows[0].end == 135
    assert windows[0].reasons == ("geometry", "terminal")


def test_recovery_budget_and_two_clip_prerequisite() -> None:
    state = Stage16DGeometryAwarePPORecoveryStateMachine()
    state.register_optimizer_level("hocap_170650", "G1")
    state.register_formal_evaluation("hocap_170650", "G1")
    with np.testing.assert_raises_regex(RuntimeError, "FORMAL20_BUDGET"):
        state.register_formal_evaluation("hocap_170650", "G1")
    with np.testing.assert_raises_regex(RuntimeError, "PREREQUISITES"):
        state.transition("TWO_CLIP", reason="must fail before both single PPOs")
    state.authorize_single_ppo_success("hocap_170105")
    state.authorize_single_ppo_success("hocap_170650")
    state.transition("TWO_CLIP", reason="both single PPOs validated")
