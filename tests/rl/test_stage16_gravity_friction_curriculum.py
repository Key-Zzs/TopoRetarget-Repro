"""P3/P4 curriculum guards are pure-Python and must fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from toporetarget.rl.gravity_friction_curriculum import (
    CURRICULUM_STAGES,
    INITIAL_SAFE_BANKS,
    load_gravity_friction_curriculum,
    load_p3_entry_gate_v2,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_frozen_schedule_reaches_exact_nominal_endpoint() -> None:
    contract = load_gravity_friction_curriculum(
        _root() / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
    )
    assert tuple(stage.identifier for stage in contract.stages) == CURRICULUM_STAGES
    assert [contract.physics(stage)["gravity_scale"] for stage in CURRICULUM_STAGES] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert [contract.physics(stage)["friction_scale"] for stage in CURRICULUM_STAGES] == [
        2.0,
        1.75,
        1.5,
        1.25,
        1.0,
    ]
    endpoint = contract.physics("C4")
    assert endpoint["gravity_world_mps2"] == [0.0, 0.0, -9.81]
    assert endpoint["material_roles"]["global_default_rigid_body"]["static_friction"] == 0.5
    assert endpoint["material_roles"]["hocap_bound_object_material"]["dynamic_friction"] == 1.0


def test_checkpoint_state_carries_stage_physics_and_safe_banks() -> None:
    contract = load_gravity_friction_curriculum(
        _root() / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
    )
    state = contract.checkpoint_state(
        stage="C2", allowed_reset_banks=INITIAL_SAFE_BANKS, selected_contact_mode="aggregate_v3"
    )
    assert state["curriculum_stage"] == "C2"
    assert state["gravity_scale"] == 0.5
    assert state["friction_scale"] == 1.5
    assert state["selected_contact_mode"] == "aggregate_v3"
    with pytest.raises(ValueError, match="RESET_BANKS_DRIFT"):
        contract.checkpoint_state(
            stage="C2", allowed_reset_banks=("PRE_CONTACT",), selected_contact_mode="aggregate_v3"
        )


def test_v2_entry_moves_g3_to_promotion_without_overwriting_v1() -> None:
    entry = load_p3_entry_gate_v2(_root() / "configs/rl/stage16/stage16_p3_entry_gate_v2.yaml")
    assert entry["historical_entry_gate"].endswith("stage16_p3_entry_gate_v1.yaml")
    assert entry["decision_contract"]["status_on_pass"] == "P3_READY_WITH_CONSTRAINTS"
    assert entry["promotion_gate"]["placement"] == "between_C2_and_C3"
    assert entry["decision_contract"]["external_guidance"] is False
