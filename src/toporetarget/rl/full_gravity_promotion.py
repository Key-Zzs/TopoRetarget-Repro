"""Pure G3 full-gravity promotion gates for every contact-ready safe state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

G3_PROMOTION_SCHEMA = "Stage16P3FullGravityPromotionV1"
G3_BLOCKED = "G3_PROMOTION_BLOCKED"
G3_PASS = "G3_PROMOTION_PASS"
G3_PASS_WITH_FILTERED_BANK = "G3_PROMOTION_PASS_WITH_FILTERED_BANK"


def validate_g3_contract(contract: Mapping[str, Any]) -> dict[str, object]:
    """Validate the pre-registered C4 physics and bounded G3 diagnostic budget."""

    if contract.get("physics_stage") != "C4":
        raise ValueError("G3_REQUIRES_C4_PHYSICS_STAGE")
    if int(contract.get("replicas_per_safe_state", -1)) != 4:
        raise ValueError("G3_REQUIRES_FOUR_REPLICAS_PER_SAFE_STATE")
    if int(contract.get("control_steps", -1)) != 20:
        raise ValueError("G3_REQUIRES_TWENTY_CONTROL_STEPS")
    if int(contract.get("minimum_retained_safe_states_per_clip", -1)) != 1:
        raise ValueError("G3_MINIMUM_RETAINED_SAFE_STATES_DRIFT")
    requirements = contract.get("controller_requirements")
    if not isinstance(requirements, Mapping) or requirements != {
        "nonfinite_count": 0,
        "systematic_joint_limit_failure": False,
        "actuator_explosion": False,
        "contact_solver_instability": False,
        "rollout_object_state_writes": 0,
        "rollout_wrist_root_writes": 0,
    }:
        raise ValueError("G3_CONTROLLER_REQUIREMENTS_DRIFT")
    return {
        "physics_stage": "C4",
        "gravity_scale": 1.0,
        "friction_scale": 1.0,
        "replicas_per_safe_state": 4,
        "control_steps": 20,
        "minimum_retained_safe_states_per_clip": 1,
    }


def expected_g3_state_replica_pairs(safe_indices: Sequence[int]) -> set[tuple[int, int]]:
    """Return the exhaustive, deterministic G3 roster; sampling is forbidden."""

    indices = tuple(int(index) for index in safe_indices)
    if not indices or len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        raise ValueError("G3_SAFE_STATE_ROSTER_INVALID")
    return {(index, replica) for index in indices for replica in range(4)}


def decide_g3_promotion(
    *,
    safe_indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    rollout_object_state_writes: int,
    rollout_wrist_root_writes: int,
) -> dict[str, object]:
    """Apply frozen G3 gates, permitting only initial-geometry filtering.

    Rows must include every safe-state/replica pair even when a state is
    filtered.  This makes an unavailable policy or a failed initial geometry
    auditable rather than silently removing difficult starts.
    """

    expected = expected_g3_state_replica_pairs(safe_indices)
    by_pair: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in rows:
        pair = (int(row.get("runtime_index", -1)), int(row.get("replica", -1)))
        if pair in by_pair or pair not in expected:
            raise ValueError("G3_STATE_REPLICA_ROSTER_MISMATCH")
        by_pair[pair] = row
    if set(by_pair) != expected:
        raise ValueError("G3_STATE_REPLICA_ROSTER_INCOMPLETE")
    states: list[dict[str, object]] = []
    for state in sorted({pair[0] for pair in expected}):
        replicas = [by_pair[(state, replica)] for replica in range(4)]
        initial_geometry_valid = all(bool(row.get("initial_geometry_valid")) for row in replicas)
        if not initial_geometry_valid:
            states.append(
                {
                    "runtime_index": state,
                    "status": "FILTERED",
                    "filter_reason": "INITIAL_GEOMETRY_INVALID",
                    "replica_count": 4,
                    "replica_pass_count": 0,
                }
            )
            continue
        required = (
            "finite",
            "absolute_geometry_pass",
            "interfinger_pass",
            "joint_safe",
            "action_safe",
            "no_actuator_explosion",
            "no_contact_solver_instability",
        )
        pass_count = sum(all(bool(row.get(key)) for key in required) for row in replicas)
        states.append(
            {
                "runtime_index": state,
                "status": "PASS" if pass_count == 4 else "DYNAMIC_FAILURE",
                "filter_reason": None,
                "replica_count": 4,
                "replica_pass_count": pass_count,
            }
        )
    retained = [state for state in states if state["status"] != "FILTERED"]
    dynamic_fail = [state for state in states if state["status"] == "DYNAMIC_FAILURE"]
    controller_pass = (
        rollout_object_state_writes == 0 and rollout_wrist_root_writes == 0 and not dynamic_fail
    )
    if len(retained) < 1 or not controller_pass:
        status = G3_BLOCKED
    elif len(retained) == len(states):
        status = G3_PASS
    else:
        status = G3_PASS_WITH_FILTERED_BANK
    return {
        "schema_version": G3_PROMOTION_SCHEMA,
        "status": status,
        "candidate_safe_state_count": len(states),
        "retained_safe_state_count": len(retained),
        "filtered_initial_geometry_count": len(states) - len(retained),
        "dynamic_failure_state_count": len(dynamic_fail),
        "controller_pass": controller_pass,
        "rollout_object_state_writes": int(rollout_object_state_writes),
        "rollout_wrist_root_writes": int(rollout_wrist_root_writes),
        "states": states,
    }


__all__ = [
    "G3_BLOCKED",
    "G3_PASS",
    "G3_PASS_WITH_FILTERED_BANK",
    "G3_PROMOTION_SCHEMA",
    "decide_g3_promotion",
    "expected_g3_state_replica_pairs",
    "validate_g3_contract",
]
