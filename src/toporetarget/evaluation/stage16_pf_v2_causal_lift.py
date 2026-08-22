"""Versioned Stage16 physical-functionality V2 causal-lift semantics.

The evaluator is deliberately trace-only.  It distinguishes the reference
``LIFT`` annotation from an actual physical lift, and never turns a delay
relative to that annotation into a physical-functionality failure.  Exact
support wrench transfer and surface-relative slip are not present in the
recorded traces, so the support and coupling conclusions below are explicitly
named proxies rather than hidden force-closure claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from toporetarget.evaluation.human_object_interaction_profile import (
    build_human_object_interaction_profile,
)
from toporetarget.rl.physical_evaluation import persistent_mask

STAGE16_PHYSICAL_FUNCTIONALITY_V2: Final = "Stage16PhysicalFunctionalityV2"
ACTUAL_LIFT_ONSET_V1: Final = "Stage16ActualLiftOnsetV1"
SUPPORT_TRANSFER_PROXY_V1: Final = "Stage16SupportTransferProxyV1"
CAUSAL_LOAD_BEARING_INTERACTION_V1: Final = "CausalLoadBearingInteractionV1"
DF_INTERACTION_TIMING_V1: Final = "DFInteractionTimingV1"


@dataclass(frozen=True)
class Stage16PhysicalFunctionalityV2Contract:
    """Frozen, outcome-independent causal-lift contract for Stage16 traces."""

    schema_version: str = STAGE16_PHYSICAL_FUNCTIONALITY_V2
    actual_lift_schema: str = ACTUAL_LIFT_ONSET_V1
    support_transfer_schema: str = SUPPORT_TRANSFER_PROXY_V1
    causal_interaction_schema: str = CAUSAL_LOAD_BEARING_INTERACTION_V1
    interaction_timing_schema: str = DF_INTERACTION_TIMING_V1
    lift_threshold_m: float = 0.05
    persistence_control_steps: int = 3
    multifinger_minimum: int = 2
    control_period_s: float = 0.05
    vertical_velocity_rule: str = "pose_derived_centered_vertical_velocity_strictly_positive"
    support_signal: str = "table_object_contact_binary_proxy_no_exact_normal_wrench"
    support_validity_rule: str = (
        "table_contact_sensor_validity_is_independent_of_hand_object_pair_force_validity; "
        "a recorded reset support sample is retained"
    )
    coupling_rule: str = (
        "persistent_multifinger_contact_through_actual_lift_plus_"
        "finite_pose_derived_relative_motion"
    )
    reference_lift_hard_gate: bool = False
    exact_wrench_transfer_claimed: bool = False
    exact_surface_slip_claimed: bool = False
    outcome_tuned: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != STAGE16_PHYSICAL_FUNCTIONALITY_V2:
            raise ValueError("PF_V2_SCHEMA_DRIFT")
        if (self.lift_threshold_m, self.persistence_control_steps, self.multifinger_minimum) != (
            0.05,
            3,
            2,
        ):
            raise ValueError("PF_V2_FROZEN_PERSISTENCE_OR_LIFT_DRIFT")
        if self.control_period_s != 0.05:
            raise ValueError("PF_V2_CONTROL_PERIOD_DRIFT")
        if (
            self.reference_lift_hard_gate
            or self.exact_wrench_transfer_claimed
            or self.exact_surface_slip_claimed
            or self.outcome_tuned
        ):
            raise ValueError("PF_V2_FORBIDDEN_SEMANTIC_CLAIM")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector(values: np.ndarray, *, name: str, count: int, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape != (count,):
        raise ValueError(f"PF_V2_{name}_MUST_BE_[T]")
    return result


def _matrix(values: np.ndarray, *, name: str, count: int) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    if result.ndim != 2 or result.shape[0] != count:
        raise ValueError(f"PF_V2_{name}_MUST_BE_[T,N]")
    return result


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not len(indices) else int(indices[0])


def _persistent(values: np.ndarray, *, minimum_steps: int) -> np.ndarray:
    return np.asarray(persistent_mask(np.asarray(values, dtype=bool), minimum_steps=minimum_steps))


def _at_least(values: np.ndarray, start: int | None, *, count: int) -> bool:
    if start is None or start + count > len(values):
        return False
    return bool(np.all(np.asarray(values, dtype=bool)[start : start + count]))


def _summary(values: np.ndarray, *, start: int | None, count: int) -> dict[str, float | None]:
    if start is None or start + count > len(values):
        return {"mean": None, "max": None}
    window = np.asarray(values, dtype=np.float64)[start : start + count]
    return {"mean": float(window.mean()), "max": float(window.max())}


def _actual_lift_event(
    *,
    object_pose_wxyz: np.ndarray,
    table_object_contact: np.ndarray,
    support_valid: np.ndarray,
    contract: Stage16PhysicalFunctionalityV2Contract,
) -> dict[str, object]:
    pose = np.asarray(object_pose_wxyz, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7 or len(pose) < contract.persistence_control_steps:
        raise ValueError("PF_V2_OBJECT_POSE_MUST_BE_[T,7]_WITH_PERSISTENCE")
    count = len(pose)
    table = _vector(table_object_contact, name="TABLE_OBJECT_CONTACT", count=count, dtype=bool)
    support_rows = _vector(support_valid, name="SUPPORT_VALID", count=count, dtype=bool)
    timestamps = np.arange(count, dtype=np.float64) * contract.control_period_s
    vertical_velocity = np.gradient(pose[:, 2], timestamps, edge_order=1)
    displacement = pose[:, 2] - pose[0, 2]
    support_free = (~table) & support_rows
    support_release = _first_true(
        _persistent(support_free, minimum_steps=contract.persistence_control_steps)
    )
    actual_lift_mask = (
        support_rows
        & support_free
        & (displacement >= contract.lift_threshold_m)
        & (vertical_velocity > 0.0)
    )
    onset = _first_true(
        _persistent(actual_lift_mask, minimum_steps=contract.persistence_control_steps)
    )
    return {
        "actual_lift_onset": onset,
        "support_release_event": support_release,
        "object_vertical_displacement_m": displacement,
        "object_vertical_velocity_mps": vertical_velocity,
        "actual_lift_candidate": actual_lift_mask,
        "physical_lift_success": onset is not None,
        "support_present_before_release": bool(
            support_release is not None
            and bool(np.any(table[:support_release] & support_rows[:support_release]))
        ),
    }


def evaluate_stage16_physical_functionality_v2(
    *,
    object_pose_wxyz: np.ndarray,
    wrist_pose_wxyz: np.ndarray,
    tip_pair_presence: np.ndarray,
    hand_object_pair_presence: np.ndarray,
    table_object_contact: np.ndarray,
    valid: np.ndarray | None = None,
    interaction_valid: np.ndarray | None = None,
    support_valid: np.ndarray | None = None,
    reference_lift_onset: int | None,
    causal_execution: bool,
    geometry_safe: bool,
    action_bounds_safe: bool,
    no_hidden_control: bool,
    contract: Stage16PhysicalFunctionalityV2Contract | None = None,
) -> dict[str, object]:
    """Evaluate causal physical lift while retaining timing as a DF diagnostic.

    A physical pass requires actual, persistent support-free lift with observed
    hand-object interaction through the actual event.  Hand-object pair-force
    validity and table-contact validity are separate: frame zero has no
    post-physics hand-pair sample, but a recorded reset table-support sample
    remains admissible evidence of support before release.  The recorded
    support signal is binary; this proves a *support-transfer proxy*, not exact
    normal load transfer.  Relative-motion fields are reported from the
    established pose-derived estimator family but intentionally receive no
    fitted cutoff.
    """

    frozen = contract or Stage16PhysicalFunctionalityV2Contract()
    pose = np.asarray(object_pose_wxyz, dtype=np.float64)
    wrist = np.asarray(wrist_pose_wxyz, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7 or wrist.shape != pose.shape:
        raise ValueError("PF_V2_WRIST_AND_OBJECT_POSE_MUST_MATCH_[T,7]")
    count = len(pose)
    if interaction_valid is not None and valid is not None:
        raise ValueError("PF_V2_INTERACTION_VALID_AND_LEGACY_VALID_BOTH_PROVIDED")
    if interaction_valid is None:
        if valid is None:
            raise ValueError("PF_V2_INTERACTION_VALID_REQUIRED")
        interaction_valid = valid
    rows = _vector(interaction_valid, name="INTERACTION_VALID", count=count, dtype=bool)
    support_rows = (
        np.ones(count, dtype=bool)
        if support_valid is None
        else _vector(support_valid, name="SUPPORT_VALID", count=count, dtype=bool)
    )
    tips = _matrix(tip_pair_presence, name="TIP_PAIR_PRESENCE", count=count) & rows[:, None]
    hand = _matrix(hand_object_pair_presence, name="HAND_OBJECT_PAIR_PRESENCE", count=count)
    hand &= rows[:, None]

    event = _actual_lift_event(
        object_pose_wxyz=pose,
        table_object_contact=table_object_contact,
        support_valid=support_rows,
        contract=frozen,
    )
    onset_value = event["actual_lift_onset"]
    support_release_value = event["support_release_event"]
    onset = int(onset_value) if isinstance(onset_value, int) else None
    support_release = int(support_release_value) if isinstance(support_release_value, int) else None
    vertical_displacement = np.asarray(event["object_vertical_displacement_m"], dtype=np.float64)
    vertical_velocity = np.asarray(event["object_vertical_velocity_mps"], dtype=np.float64)

    persistent_tips = np.stack(
        [
            _persistent(tips[:, index], minimum_steps=frozen.persistence_control_steps)
            for index in range(tips.shape[1])
        ],
        axis=1,
    )
    persistent_multi = persistent_tips.sum(axis=1) >= frozen.multifinger_minimum
    any_hand = np.asarray(hand.any(axis=1), dtype=np.bool_)
    persistent_hand = _persistent(any_hand, minimum_steps=frozen.persistence_control_steps)
    persistent_multi_onset = _first_true(persistent_multi)
    first_hand_contact = _first_true(any_hand)
    hand_contact_at_lift = bool(onset is not None and persistent_hand[onset])
    multi_contact_at_lift = bool(onset is not None and persistent_multi[onset])
    sustained_contact = _at_least(persistent_multi, onset, count=frozen.persistence_control_steps)

    coupling = build_human_object_interaction_profile(
        timestamps_s=np.arange(count, dtype=np.float64) * frozen.control_period_s,
        hand_pose_world_wxyz=wrist,
        object_pose_world_wxyz=pose,
        region_contact=tips,
    )
    relative_linear = np.asarray(coupling["relative_linear_speed_mps"], dtype=np.float64)
    relative_angular = np.asarray(coupling["relative_angular_speed_radps"], dtype=np.float64)
    finite_relative_motion = bool(
        onset is not None
        and np.isfinite(relative_linear[onset : onset + frozen.persistence_control_steps]).all()
        and np.isfinite(relative_angular[onset : onset + frozen.persistence_control_steps]).all()
    )
    causal_hand_object_lift = bool(
        event["physical_lift_success"]
        and hand_contact_at_lift
        and multi_contact_at_lift
        and persistent_multi_onset is not None
        and onset is not None
        and persistent_multi_onset <= onset
    )
    support_transfer = bool(
        event["support_present_before_release"]
        and support_release is not None
        and onset is not None
        and support_release <= onset
        and hand_contact_at_lift
    )
    sustained_coupling = bool(sustained_contact and finite_relative_motion)

    timing = {
        "schema_version": DF_INTERACTION_TIMING_V1,
        "reference_lift_onset": reference_lift_onset,
        "first_hand_object_contact": first_hand_contact,
        "persistent_multifinger_contact": persistent_multi_onset,
        "actual_lift_onset": onset,
        "pre_reference_lift_multifinger_contact": bool(
            reference_lift_onset is not None
            and persistent_multi_onset is not None
            and persistent_multi_onset <= reference_lift_onset
        ),
        "pre_reference_lift_margin": (
            None
            if reference_lift_onset is None or persistent_multi_onset is None
            else int(reference_lift_onset - persistent_multi_onset)
        ),
        "pre_actual_lift_margin": (
            None
            if onset is None or persistent_multi_onset is None
            else int(onset - persistent_multi_onset)
        ),
        "interaction_timing_fidelity": "PASS"
        if reference_lift_onset is not None
        and persistent_multi_onset is not None
        and persistent_multi_onset <= reference_lift_onset
        else "FAIL_OR_UNAVAILABLE",
    }
    gates = {
        "causal_execution": bool(causal_execution),
        "geometry_safe": bool(geometry_safe),
        "action_bounds_safe": bool(action_bounds_safe),
        "physical_lift_success": bool(event["physical_lift_success"]),
        "causal_hand_object_lift": causal_hand_object_lift,
        "support_transfer_success": support_transfer,
        "sustained_hand_object_coupling": sustained_coupling,
        "no_hidden_control": bool(no_hidden_control),
    }
    failure_reasons = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": STAGE16_PHYSICAL_FUNCTIONALITY_V2,
        "contract": frozen.as_dict(),
        "pf_v2": not failure_reasons,
        "pf_v2_failure_reasons": failure_reasons,
        **gates,
        "actual_lift": {
            "schema_version": ACTUAL_LIFT_ONSET_V1,
            "onset": onset,
            "support_release_event": support_release,
            "lift_threshold_m": frozen.lift_threshold_m,
            "vertical_velocity_rule": frozen.vertical_velocity_rule,
            "vertical_displacement_at_onset_m": (
                None if onset is None else float(vertical_displacement[onset])
            ),
            "vertical_velocity_at_onset_mps": (
                None if onset is None else float(vertical_velocity[onset])
            ),
        },
        "support_transfer": {
            "schema_version": SUPPORT_TRANSFER_PROXY_V1,
            "is_exact_wrench_transfer": False,
            "signal": frozen.support_signal,
            "validity_rule": frozen.support_validity_rule,
            "observed_support_rows": int(support_rows.sum()),
            "support_present_before_release": event["support_present_before_release"],
            "support_release_event": support_release,
            "support_transfer_success": support_transfer,
        },
        "causal_interaction": {
            "schema_version": CAUSAL_LOAD_BEARING_INTERACTION_V1,
            "hand_contact_at_actual_lift": hand_contact_at_lift,
            "multifinger_contact_at_actual_lift": multi_contact_at_lift,
            "post_lift_persistent_multifinger_contact": sustained_contact,
            "relative_motion_continuity": finite_relative_motion,
            "exact_surface_slip_identifiable": False,
            "causal_hand_object_lift": causal_hand_object_lift,
            "ballistic_or_flick_rejected": bool(causal_hand_object_lift and sustained_coupling),
            "relative_linear_speed_mps": _summary(
                relative_linear, start=onset, count=frozen.persistence_control_steps
            ),
            "relative_angular_speed_radps": _summary(
                relative_angular, start=onset, count=frozen.persistence_control_steps
            ),
        },
        "interaction_timing": timing,
    }


__all__ = [
    "ACTUAL_LIFT_ONSET_V1",
    "CAUSAL_LOAD_BEARING_INTERACTION_V1",
    "DF_INTERACTION_TIMING_V1",
    "STAGE16_PHYSICAL_FUNCTIONALITY_V2",
    "SUPPORT_TRANSFER_PROXY_V1",
    "Stage16PhysicalFunctionalityV2Contract",
    "evaluate_stage16_physical_functionality_v2",
]
