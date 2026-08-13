from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.c2_geometry_attribution import (
    PHYSICS_VARIANTS,
    GeometryGateV1,
    controller_indicators,
    decision_contract,
    first_violation,
    physics_attribution,
    root_cause_matrix,
    temporal_classification,
)


@pytest.fixture
def gate() -> GeometryGateV1:
    return GeometryGateV1(0.01, 0.003, 0.003)


def test_first_violation_and_initial_temporal_classification(gate: GeometryGateV1) -> None:
    values = np.array([0.005, 0.001, 0.012])
    assert first_violation(values, gate=gate) == 0
    assert (
        temporal_classification(
            first_violation_frame=0,
            maximum_frame=0,
            frame_count=3,
            first_contact_frame=0,
            reset_violates=True,
        )
        == "INITIAL_GEOMETRY_INVALID"
    )


def test_temporal_classification_is_deterministic() -> None:
    assert (
        temporal_classification(
            first_violation_frame=7,
            maximum_frame=7,
            frame_count=40,
            first_contact_frame=3,
            reset_violates=False,
        )
        == "CONTACT_TRANSIENT_GEOMETRY_FAILURE"
    )
    assert (
        temporal_classification(
            first_violation_frame=33,
            maximum_frame=35,
            frame_count=40,
            first_contact_frame=3,
            reset_violates=False,
        )
        == "LATE_POLICY_GEOMETRY_FAILURE"
    )


def test_counterfactual_decision_contract_freezes_abcd() -> None:
    contract = decision_contract(GeometryGateV1(0.01, 0.003, 0.003))
    assert set(contract["physics_variants"]) == set(PHYSICS_VARIANTS)
    assert "ppo_training" in contract["prohibited"]
    assert "collision_geometry" in contract["frozen_invariants"]


def test_friction_gravity_and_mixed_attribution() -> None:
    baseline = {
        "A": {"p95_penetration_m": 0.005, "max_penetration_m": 0.006, "gate_pass": False},
        "B": {"p95_penetration_m": 0.001, "max_penetration_m": 0.002, "gate_pass": True},
        "C": {"p95_penetration_m": 0.004, "max_penetration_m": 0.005, "gate_pass": False},
        "D": {"p95_penetration_m": 0.001, "max_penetration_m": 0.002, "gate_pass": True},
    }
    assert (
        physics_attribution(rows=baseline, mode="closed_loop") == "HIGH_FRICTION_STICKING_PRIMARY"
    )
    gravity = {**baseline, "B": baseline["C"], "C": baseline["B"]}
    assert physics_attribution(rows=gravity, mode="closed_loop") == "GRAVITY_LOAD_PRIMARY"
    mixed = {
        "A": baseline["A"],
        "B": {"p95_penetration_m": 0.0048, "max_penetration_m": 0.0058, "gate_pass": False},
        "C": {"p95_penetration_m": 0.0048, "max_penetration_m": 0.0058, "gate_pass": False},
        "D": baseline["B"],
    }
    assert physics_attribution(rows=mixed, mode="closed_loop") == "GRAVITY_FRICTION_COUPLING"


def test_controller_indicators_and_root_cause_matrix() -> None:
    values = np.zeros((21, 20))
    report = controller_indicators(
        finger_target=values + 0.4,
        finger_actual=values,
        wrist_target=np.zeros((21, 7)),
        wrist_actual=np.zeros((21, 7)),
        actuator_effort=np.ones((21, 26)),
        effort_limit=1.0,
        contact_force_world=np.ones((21, 3)),
        center_frame=10,
    )
    assert report["joint_target_error_peak_rad"] == pytest.approx(0.4)
    assert report["effort_saturation_fraction"] == pytest.approx(1.0)
    matrix = root_cause_matrix(
        reset_fraction=0.9,
        friction_label="PHYSICS_PARAMETER_EFFECT_NOT_SUPPORTED",
        gravity_label="PHYSICS_PARAMETER_EFFECT_NOT_SUPPORTED",
        controller_overdrive=False,
        policy_reaction=False,
        proxy_discrepancy=None,
    )
    assert matrix["Reset"] == "STRONG"
    assert matrix["Proxy"] == "WEAK"
