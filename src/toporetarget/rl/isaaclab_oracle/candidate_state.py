"""Capture, validate, hash, restore, and replicate C.5A candidate tensors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol

import torch

from .contracts import (
    REQUIRED_STATE_FIELDS,
    CandidateStateValidationError,
    Stage16C5CandidateStateV1,
    Stage16C5WriteAuditV1,
)


class _LiveEnv(Protocol):
    """Structural subset of the live Isaac environment used by this module."""

    device: torch.device | str
    num_envs: int


_ENV_TENSOR_ATTRIBUTES: dict[str, str] = {
    "clip_index": "_clip_index",
    "reference_index": "_reference_index",
    "target_reference_index": "_target_reference_index",
    "actions": "_actions",
    "previous_actions": "_previous_actions",
    "second_previous_actions": "_second_previous_actions",
    "joint_target_isaac": "_joint_target_isaac",
    "explicit_wrist_joint_target": "_explicit_wrist_joint_target",
    "explicit_wrist_joint_velocity_target": "_explicit_wrist_joint_velocity_target",
    "previous_explicit_wrist_joint_target": "_previous_explicit_wrist_joint_target",
    "explicit_wrist_singularity_margin_deg": "_explicit_wrist_singularity_margin_deg",
    "computed_torque_bias_estimate": "_computed_torque_bias_estimate",
    "wrist_target_position": "_wrist_target_position",
    "wrist_target_quaternion": "_wrist_target_quaternion",
    "wrist_target_twist": "_wrist_target_twist",
    "wrist_target_acceleration": "_wrist_target_acceleration",
    "wrist_interval_start_position": "_wrist_interval_start_position",
    "wrist_interval_end_position": "_wrist_interval_end_position",
    "wrist_interval_start_quaternion": "_wrist_interval_start_quaternion",
    "wrist_interval_end_quaternion": "_wrist_interval_end_quaternion",
    "wrist_interval_start_twist": "_wrist_interval_start_twist",
    "wrist_interval_end_twist": "_wrist_interval_end_twist",
    "object_interval_start_position": "_object_interval_start_position",
    "object_interval_end_position": "_object_interval_end_position",
    "object_interval_start_quaternion": "_object_interval_start_quaternion",
    "object_interval_end_quaternion": "_object_interval_end_quaternion",
    "object_interval_start_twist": "_object_interval_start_twist",
    "object_interval_end_twist": "_object_interval_end_twist",
    "wrist_translation_residual": "_wrist_translation_residual",
    "wrist_rotation_residual": "_wrist_rotation_residual",
    "force_saturated": "_force_saturated",
    "torque_saturated": "_torque_saturated",
    "force_saturation_substeps": "_force_saturation_substeps",
    "torque_saturation_substeps": "_torque_saturation_substeps",
    "velocity_saturated": "_velocity_saturated",
    "velocity_saturation_substeps": "_velocity_saturation_substeps",
    "wrist_substeps": "_wrist_substeps",
    "success": "_success",
    "reason_codes": "_reason_codes",
    "identified_map_condition_number": "_identified_map_condition_number",
    "identified_map_condition_gate_pass": "_identified_map_condition_gate_pass",
    "identified_map_selected_reference_frame": "_identified_map_selected_reference_frame",
    "base_episode_length_buf": "episode_length_buf",
    "base_reset_terminated": "reset_terminated",
    "base_reset_time_outs": "reset_time_outs",
}

_WORLD_POSITION_FIELDS = {
    "robot_root_state",
    "object_170105_root_state",
    "object_170650_root_state",
    "wrist_target_position",
}

# The task state has a deliberately narrow integer/bool surface.  Making this
# explicit catches accidental float casts during snapshot construction before a
# candidate is ever restored into PhysX.
_INTEGER_STATE_FIELDS = {
    "clip_index",
    "reference_index",
    "target_reference_index",
    "reason_codes",
    "identified_map_selected_reference_frame",
    "base_episode_length_buf",
}
_BOOLEAN_STATE_FIELDS = {
    "force_saturated",
    "torque_saturated",
    "velocity_saturated",
    "success",
    "identified_map_condition_gate_pass",
    "base_reset_terminated",
    "base_reset_time_outs",
}

_REQUIRED_CONFIG_HASHES = {
    "candidate_state_contract",
    "runtime_profile",
    "reference",
    "direct_env_contract",
    "controller",
}
_REQUIRED_SCALARS = {
    "physics_substep",
    "contact_software_state",
    "object_activation",
    "rng",
    "canonical_joint_order",
    "canonical_finger_joint_order",
}


def _as_env_ids(env: _LiveEnv, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    result = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if result.ndim != 1 or result.numel() == 0:
        raise CandidateStateValidationError("candidate env ids must be a nonempty vector")
    if bool((result < 0).any()) or bool((result >= env.num_envs).any()):
        raise CandidateStateValidationError("candidate env id outside live environment")
    return result


def _indexed_clone(value: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
    if value.ndim < 1 or value.shape[0] <= int(env_ids.max().item()):
        raise CandidateStateValidationError(
            f"state tensor cannot be indexed by env ids: shape={tuple(value.shape)}"
        )
    return value.index_select(0, env_ids).detach().clone().contiguous()


def _data_tensor(owner: Any, name: str) -> torch.Tensor | None:
    value = getattr(owner, name, None)
    return value if isinstance(value, torch.Tensor) else None


def _hash_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _runtime_hashes(env: Any) -> dict[str, str]:
    manifest = env.reference_bank.manifest.as_dict()
    contract = env.contract_report()
    return {
        "candidate_state_contract": "stage16c5_candidate_state_v1",
        "runtime_profile": _hash_json(manifest),
        "reference": _hash_json(manifest["hashes"]),
        "direct_env_contract": _hash_json(
            {
                "action": contract["action"],
                "observation_dimension": contract["observation_dimension"],
                "reference_timing": contract["reference_timing"],
                "termination": contract["termination"],
            }
        ),
        "controller": _hash_json(
            contract.get("finite_virtual_6d_wrist_actuator")
            or contract.get("full_articulation_computed_torque")
            or contract.get("bounded_tvlqr_wrist")
            or contract.get("bounded_mpc_wrist")
            or {}
        ),
    }


def _robot_state(env: Any, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    robot_data = env._robot.data
    required = {
        "robot_joint_pos": _data_tensor(robot_data, "joint_pos"),
        "robot_joint_vel": _data_tensor(robot_data, "joint_vel"),
        "robot_root_state": _data_tensor(robot_data, "root_state_w"),
    }
    absent = [name for name, value in required.items() if value is None]
    if absent:
        raise CandidateStateValidationError(f"live articulation lacks required tensors: {absent}")
    state = {
        name: _indexed_clone(value, env_ids)
        for name, value in required.items()
        if value is not None
    }
    for source, target in (
        ("joint_pos_target", "robot_joint_pos_target"),
        ("joint_vel_target", "robot_joint_vel_target"),
        ("joint_effort_target", "robot_joint_effort_target"),
    ):
        value = _data_tensor(robot_data, source)
        if value is not None:
            state[target] = _indexed_clone(value, env_ids)
    return state


def capture_candidate_state(
    env: Any, env_ids: Sequence[int] | torch.Tensor | None = None
) -> Stage16C5CandidateStateV1:
    """Capture every currently API-restorable causal tensor from a live env."""

    indices = _as_env_ids(env, env_ids)
    tensors = _robot_state(env, indices)
    for object_name, object_handle in (
        ("object_170105_root_state", env._object_170105),
        ("object_170650_root_state", env._object_170650),
    ):
        root_state = _data_tensor(object_handle.data, "root_state_w")
        if root_state is None:
            raise CandidateStateValidationError(f"live object lacks root_state_w: {object_name}")
        tensors[object_name] = _indexed_clone(root_state, indices)
    tensors["source_env_origins"] = _indexed_clone(env.scene.env_origins, indices)
    for target_name, attribute_name in _ENV_TENSOR_ATTRIBUTES.items():
        value = getattr(env, attribute_name, None)
        if isinstance(value, torch.Tensor):
            tensors[target_name] = _indexed_clone(value, indices)
    scalars: dict[str, Any] = {
        "physics_substep": int(getattr(env, "_physics_substep", 0)),
        "contact_software_state": {
            "pending_sample_present": getattr(env, "_pending_contact_sample", None) is not None,
            "record_history_role": "diagnostic_only",
            "sensor_cache_restore": "unavailable_recompute_from_physx",
        },
        "object_activation": {
            "active_object_by_clip": {"0": "Object170105", "1": "Object170650"},
            "inactive_object_preserved": True,
        },
        "rng": {
            "torch_initial_seed": int(torch.initial_seed()),
            "formal_candidate_rollout_randomized": False,
        },
        "canonical_joint_order": list(env._robot.joint_names),
        "canonical_finger_joint_order": list(env.reference_bank.joint_order),
    }
    state = Stage16C5CandidateStateV1(
        config_hashes=_runtime_hashes(env),
        tensors=tensors,
        scalars=scalars,
        env_count=int(indices.numel()),
    )
    validate_candidate_state(state, expected_device=env.device)
    return state


def validate_candidate_state(
    state: Stage16C5CandidateStateV1,
    *,
    expected_device: torch.device | str | None = None,
) -> None:
    """Fail closed on missing, nonfinite, incorrectly shaped state tensors."""

    if state.version != "Stage16C5CandidateStateV1":
        raise CandidateStateValidationError(f"unsupported candidate state version: {state.version}")
    if state.env_count < 1:
        raise CandidateStateValidationError("candidate state must contain at least one environment")
    missing_hashes = _REQUIRED_CONFIG_HASHES.difference(state.config_hashes)
    if missing_hashes or any(not value for value in state.config_hashes.values()):
        raise CandidateStateValidationError(
            f"candidate state config hashes are incomplete: {sorted(missing_hashes)}"
        )
    missing_scalars = _REQUIRED_SCALARS.difference(state.scalars)
    if missing_scalars:
        raise CandidateStateValidationError(
            f"candidate state scalar contract is incomplete: {sorted(missing_scalars)}"
        )
    if len(state.scalars["canonical_joint_order"]) != 26:
        raise CandidateStateValidationError(
            "candidate state joint order must contain exactly 26 joints"
        )
    expected = None if expected_device is None else torch.device(expected_device)
    for definition in REQUIRED_STATE_FIELDS:
        if definition.required and definition.name not in state.tensors:
            raise CandidateStateValidationError(f"missing required state field: {definition.name}")
    for name, value in state.tensors.items():
        if not isinstance(value, torch.Tensor):
            raise CandidateStateValidationError(f"state field {name} is not a tensor")
        if value.ndim == 0 or value.shape[0] != state.env_count:
            raise CandidateStateValidationError(
                f"state field {name} must start with env_count={state.env_count}; "
                f"shape={tuple(value.shape)}"
            )
        if expected is not None and value.device != expected:
            raise CandidateStateValidationError(
                f"state field {name} device mismatch: {value.device} != {expected}"
            )
        if name in _INTEGER_STATE_FIELDS and value.dtype != torch.long:
            raise CandidateStateValidationError(
                f"state field {name} must be torch.long, got {value.dtype}"
            )
        if name in _BOOLEAN_STATE_FIELDS and value.dtype != torch.bool:
            raise CandidateStateValidationError(
                f"state field {name} must be torch.bool, got {value.dtype}"
            )
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all()):
                raise CandidateStateValidationError(f"state field {name} contains NaN/Inf")
    if state.tensors["robot_joint_pos"].shape[-1] != 26:
        raise CandidateStateValidationError("Stage 16-C.5 requires exactly 26 articulation DoF")
    if state.tensors["actions"].shape[-1] != 26:
        raise CandidateStateValidationError("Stage 16-C.5 requires exactly 26 action dimensions")
    for object_name in ("object_170105_root_state", "object_170650_root_state"):
        if state.tensors[object_name].shape[-1] != 13:
            raise CandidateStateValidationError(f"{object_name} must be pose(7)+twist(6)")


def hash_candidate_state(state: Stage16C5CandidateStateV1) -> str:
    """Hash metadata and raw tensor bytes in canonical field-name order."""

    validate_candidate_state(state)
    digest = hashlib.sha256()
    digest.update(json.dumps(state.config_hashes, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(state.scalars, sort_keys=True, default=str).encode("utf-8"))
    for name in sorted(state.tensors):
        value = state.tensors[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _expand_field(
    value: torch.Tensor, source_indices: torch.Tensor, target_count: int
) -> torch.Tensor:
    if source_indices.shape != (target_count,):
        raise CandidateStateValidationError("source_indices must align with target env ids")
    if bool((source_indices < 0).any()) or bool((source_indices >= value.shape[0]).any()):
        raise CandidateStateValidationError("source index outside captured candidate state")
    return value.index_select(0, source_indices).clone()


def _rebase_world_position(
    value: torch.Tensor, source_origins: torch.Tensor, target_origins: torch.Tensor
) -> torch.Tensor:
    result = value.clone()
    result[..., :3] += target_origins - source_origins
    return result


def _copy_env_tensor(
    env: Any, attribute_name: str, value: torch.Tensor, env_ids: torch.Tensor
) -> None:
    destination = getattr(env, attribute_name, None)
    if not isinstance(destination, torch.Tensor):
        return
    destination.index_copy_(0, env_ids, value)


def _state_selection(
    state: Stage16C5CandidateStateV1,
    env: Any,
    target_ids: torch.Tensor,
    source_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    selected = {
        name: _expand_field(value, source_indices, int(target_ids.numel()))
        for name, value in state.tensors.items()
    }
    target_origins = env.scene.env_origins.index_select(0, target_ids)
    source_origins = selected["source_env_origins"]
    for name in _WORLD_POSITION_FIELDS:
        if name in selected:
            selected[name] = _rebase_world_position(selected[name], source_origins, target_origins)
    return selected


def _write_live_state(
    env: Any, target_ids: torch.Tensor, selected: dict[str, torch.Tensor]
) -> None:
    env._robot.write_joint_state_to_sim(
        selected["robot_joint_pos"], selected["robot_joint_vel"], env_ids=target_ids
    )
    env._robot.write_root_state_to_sim(selected["robot_root_state"], env_ids=target_ids)
    env._object_170105.write_root_state_to_sim(
        selected["object_170105_root_state"], env_ids=target_ids
    )
    env._object_170650.write_root_state_to_sim(
        selected["object_170650_root_state"], env_ids=target_ids
    )
    for state_name, setter in (
        ("robot_joint_pos_target", env._robot.set_joint_position_target),
        ("robot_joint_vel_target", env._robot.set_joint_velocity_target),
        ("robot_joint_effort_target", env._robot.set_joint_effort_target),
    ):
        if state_name in selected:
            setter(selected[state_name], env_ids=target_ids)
    for state_name, attribute_name in _ENV_TENSOR_ATTRIBUTES.items():
        if state_name in selected:
            _copy_env_tensor(env, attribute_name, selected[state_name], target_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)


def restore_candidate_state(
    env: Any,
    state: Stage16C5CandidateStateV1,
    target_env_ids: Sequence[int] | torch.Tensor,
    *,
    source_indices: Sequence[int] | torch.Tensor | None = None,
    write_audit: Stage16C5WriteAuditV1 | None = None,
) -> None:
    """Restore a snapshot into candidate IDs without writing the source IDs."""

    validate_candidate_state(state, expected_device=env.device)
    target_ids = _as_env_ids(env, target_env_ids)
    if source_indices is None:
        sources = torch.zeros(target_ids.numel(), dtype=torch.long, device=env.device)
    else:
        sources = torch.as_tensor(source_indices, dtype=torch.long, device=env.device)
    selected = _state_selection(state, env, target_ids, sources)
    _write_live_state(env, target_ids, selected)
    audit = write_audit or getattr(env, "_stage16c5_write_audit", None)
    if audit is not None:
        audit.record(
            category="candidate_setup",
            operation="restore_candidate_state",
            env_ids=target_ids,
            tensor_names=sorted(selected),
        )


def replicate_candidate_state(
    env: Any,
    state: Stage16C5CandidateStateV1,
    target_env_ids: Sequence[int] | torch.Tensor,
    *,
    write_audit: Stage16C5WriteAuditV1 | None = None,
) -> None:
    """Replicate source state zero to every candidate environment."""

    restore_candidate_state(
        env,
        state,
        target_env_ids,
        source_indices=None,
        write_audit=write_audit,
    )


__all__ = [
    "capture_candidate_state",
    "hash_candidate_state",
    "replicate_candidate_state",
    "restore_candidate_state",
    "validate_candidate_state",
]
