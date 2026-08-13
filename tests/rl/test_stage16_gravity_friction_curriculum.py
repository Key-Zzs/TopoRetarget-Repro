"""P3/P4 curriculum guards are pure-Python and must fail closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from toporetarget.rl.full_gravity_promotion import (
    G3_BLOCKED,
    G3_PASS_WITH_FILTERED_BANK,
    decide_g3_promotion,
    expected_g3_state_replica_pairs,
    validate_g3_contract,
)
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


def _g3_row(state: int, replica: int, *, initial_valid: bool = True) -> dict[str, object]:
    return {
        "runtime_index": state,
        "replica": replica,
        "initial_geometry_valid": initial_valid,
        "finite": True,
        "absolute_geometry_pass": True,
        "interfinger_pass": True,
        "joint_safe": True,
        "action_safe": True,
        "no_actuator_explosion": True,
        "no_contact_solver_instability": True,
    }


def test_g3_uses_all_safe_states_and_only_predeclared_initial_geometry_filtering() -> None:
    curriculum = _root() / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"
    import yaml

    contract = yaml.safe_load(curriculum.read_text(encoding="utf-8"))["g3_promotion"]
    assert validate_g3_contract(contract)["gravity_scale"] == 1.0
    safe = (7, 11)
    assert expected_g3_state_replica_pairs(safe) == {
        (state, replica) for state in safe for replica in range(4)
    }
    rows = [
        _g3_row(state, replica, initial_valid=state != 7)
        for state in safe
        for replica in (0, 1, 2, 3)
    ]
    result = decide_g3_promotion(
        safe_indices=safe,
        rows=rows,
        rollout_object_state_writes=0,
        rollout_wrist_root_writes=0,
    )
    assert result["status"] == G3_PASS_WITH_FILTERED_BANK
    assert result["filtered_initial_geometry_count"] == 1


def test_g3_blocks_on_dynamic_failure_and_rejects_missing_safe_replica() -> None:
    rows = [_g3_row(7, replica) for replica in range(4)]
    rows[0]["finite"] = False
    result = decide_g3_promotion(
        safe_indices=(7,),
        rows=rows,
        rollout_object_state_writes=0,
        rollout_wrist_root_writes=0,
    )
    assert result["status"] == G3_BLOCKED
    with pytest.raises(ValueError, match="ROSTER_INCOMPLETE"):
        decide_g3_promotion(
            safe_indices=(7,),
            rows=rows[:-1],
            rollout_object_state_writes=0,
            rollout_wrist_root_writes=0,
        )
