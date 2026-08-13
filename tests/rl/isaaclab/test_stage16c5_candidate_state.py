"""CPU-only schema tests for Stage16C5CandidateStateV1."""

from __future__ import annotations

import torch

from toporetarget.rl.isaaclab_oracle.candidate_state import (
    hash_candidate_state,
    validate_candidate_state,
)
from toporetarget.rl.isaaclab_oracle.contracts import (
    REQUIRED_STATE_FIELDS,
    CandidateStateValidationError,
    Stage16C5CandidateStateV1,
)


def _state() -> Stage16C5CandidateStateV1:
    tensors = {
        field.name: torch.zeros((1, 1), dtype=torch.float32) for field in REQUIRED_STATE_FIELDS
    }
    tensors["robot_joint_pos"] = torch.zeros((1, 26))
    tensors["robot_joint_vel"] = torch.zeros((1, 26))
    tensors["robot_root_state"] = torch.tensor([[0.0, 0.0, 0.0, 1.0] + [0.0] * 9])
    tensors["object_170105_root_state"] = torch.tensor([[0.0, 0.0, 0.0, 1.0] + [0.0] * 9])
    tensors["object_170650_root_state"] = torch.tensor([[0.0, 0.0, 0.0, 1.0] + [0.0] * 9])
    tensors["source_env_origins"] = torch.zeros((1, 3))
    tensors["actions"] = torch.zeros((1, 26))
    for name in {
        "clip_index",
        "reference_index",
        "target_reference_index",
        "reason_codes",
        "identified_map_selected_reference_frame",
        "base_episode_length_buf",
    }:
        tensors[name] = torch.zeros((1, 1), dtype=torch.long)
    for name in {
        "force_saturated",
        "torque_saturated",
        "velocity_saturated",
        "success",
        "identified_map_condition_gate_pass",
        "base_reset_terminated",
        "base_reset_time_outs",
    }:
        tensors[name] = torch.zeros((1, 1), dtype=torch.bool)
    return Stage16C5CandidateStateV1(
        config_hashes={
            "candidate_state_contract": "a",
            "runtime_profile": "b",
            "reference": "c",
            "direct_env_contract": "d",
            "controller": "e",
        },
        tensors=tensors,
        scalars={
            "physics_substep": 0,
            "contact_software_state": {},
            "object_activation": {},
            "rng": {},
            "canonical_joint_order": list(range(26)),
            "canonical_finger_joint_order": list(range(20)),
        },
        env_count=1,
    )


def test_candidate_state_schema_hash_and_manifest_are_deterministic() -> None:
    state = _state()
    validate_candidate_state(state)
    assert hash_candidate_state(state) == hash_candidate_state(state)
    fields = {row["field"] for row in state.field_manifest()}
    assert {field.name for field in REQUIRED_STATE_FIELDS}.issubset(fields)
    assert state.tensors["robot_joint_pos"].shape == (1, 26)
    assert state.tensors["actions"].shape == (1, 26)


def test_candidate_state_rejects_nonfinite_required_tensor() -> None:
    state = _state()
    state.tensors["actions"][0, 0] = float("nan")
    try:
        validate_candidate_state(state)
    except CandidateStateValidationError as error:
        assert "NaN/Inf" in str(error)
    else:
        raise AssertionError("non-finite candidate action state was accepted")


def test_candidate_state_rejects_wrong_discrete_dtype() -> None:
    state = _state()
    state.tensors["clip_index"] = state.tensors["clip_index"].float()
    try:
        validate_candidate_state(state)
    except CandidateStateValidationError as error:
        assert "torch.long" in str(error)
    else:
        raise AssertionError("float clip index was accepted")
