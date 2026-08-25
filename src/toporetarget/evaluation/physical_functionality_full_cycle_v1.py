"""Trace-only full-cycle physical-functionality qualification.

``PhysicalFunctionalityFullCycleV1`` is additive to the frozen Stage16
``PhysicalFunctionalityV2`` causal-lift evaluator.  Pick is exactly the V2
result.  Later phases qualify transport, placement, release, and retreat
without turning reference interaction timing into a physical hard gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal, cast

import numpy as np

from toporetarget.evaluation.stage16_pf_v2_causal_lift import (
    Stage16PhysicalFunctionalityV2Contract,
    evaluate_stage16_physical_functionality_v2,
)
from toporetarget.rl.physical_evaluation import persistent_mask

PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1: Final = "PhysicalFunctionalityFullCycleV1"
SUPPORT_TRANSFER_HAND_TO_SURFACE_PROXY_V1: Final = "SupportTransferHandToSurfaceProxyV1"
DF_INTERACTION_TIMING_FULL_CYCLE_V1: Final = "DFInteractionTimingFullCycleV1"

PhaseStatus = Literal["PASS", "FAIL", "NOT_IDENTIFIABLE", "NOT_REACHED"]


@dataclass(frozen=True)
class PhysicalFunctionalityFullCycleV1Contract:
    """Frozen outcome-independent thresholds for full-cycle trace qualification."""

    schema_version: str = PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1
    pick_authority: str = "Stage16PhysicalFunctionalityV2"
    placement_support_schema: str = SUPPORT_TRANSFER_HAND_TO_SURFACE_PROXY_V1
    timing_diagnostic_schema: str = DF_INTERACTION_TIMING_FULL_CYCLE_V1
    persistence_control_steps: int = 3
    required_transport_progress_m: float = 0.05
    maximum_object_translation_step_m: float = 0.08
    release_stability_translation_m: float = 0.02
    release_stability_rotation_rad: float = 0.35
    retreat_clearance_m: float = 0.10
    retreat_object_translation_m: float = 0.02
    retreat_object_rotation_rad: float = 0.35
    placement_support_is_exact_force: bool = False
    reference_timing_is_hard_gate: bool = False
    outcome_tuned: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1:
            raise ValueError("PF_FULL_CYCLE_V1_SCHEMA_DRIFT")
        if self.pick_authority != "Stage16PhysicalFunctionalityV2":
            raise ValueError("PF_FULL_CYCLE_V1_PICK_AUTHORITY_DRIFT")
        if (
            self.persistence_control_steps,
            self.required_transport_progress_m,
            self.maximum_object_translation_step_m,
            self.release_stability_translation_m,
            self.release_stability_rotation_rad,
            self.retreat_clearance_m,
            self.retreat_object_translation_m,
            self.retreat_object_rotation_rad,
        ) != (3, 0.05, 0.08, 0.02, 0.35, 0.10, 0.02, 0.35):
            raise ValueError("PF_FULL_CYCLE_V1_FROZEN_THRESHOLD_DRIFT")
        if (
            self.placement_support_is_exact_force
            or self.reference_timing_is_hard_gate
            or self.outcome_tuned
        ):
            raise ValueError("PF_FULL_CYCLE_V1_FORBIDDEN_SEMANTIC_CLAIM")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _vector(values: np.ndarray, *, name: str, count: int, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.shape != (count,):
        raise ValueError(f"PF_FULL_CYCLE_V1_{name}_MUST_BE_[T]")
    return result


def _matrix(values: np.ndarray, *, name: str, count: int) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    if result.ndim != 2 or result.shape[0] != count:
        raise ValueError(f"PF_FULL_CYCLE_V1_{name}_MUST_BE_[T,N]")
    return result


def _first_persistent(values: np.ndarray, *, minimum_steps: int, start: int = 0) -> int | None:
    persistent = np.asarray(
        persistent_mask(np.asarray(values, dtype=bool), minimum_steps=minimum_steps),
        dtype=bool,
    )
    indices = np.flatnonzero(persistent & (np.arange(len(persistent)) >= start))
    return None if not len(indices) else int(indices[0])


def _phase(
    status: PhaseStatus,
    *,
    failure_reasons: list[str] | None = None,
    events: Mapping[str, object] | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "passed": True if status == "PASS" else False if status == "FAIL" else None,
        "failure_reasons": list(failure_reasons or []),
        "events": dict(events or {}),
        "diagnostics": dict(diagnostics or {}),
    }


def _not_reached(upstream_phase: str, upstream_status: str) -> dict[str, object]:
    return _phase(
        "NOT_REACHED",
        failure_reasons=[f"upstream_{upstream_phase}_{upstream_status.lower()}"],
        diagnostics={"upstream_phase": upstream_phase, "upstream_status": upstream_status},
    )


def _quaternion_distance_rad(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> np.ndarray:
    left = np.asarray(left_wxyz, dtype=np.float64)
    right = np.asarray(right_wxyz, dtype=np.float64)
    left_norm = np.linalg.norm(left, axis=-1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=-1, keepdims=True)
    if np.any(left_norm <= 0.0) or np.any(right_norm <= 0.0):
        return np.full(left.shape[:-1], np.nan, dtype=np.float64)
    dot = np.abs(np.sum((left / left_norm) * (right / right_norm), axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def _pose_drift(pose: np.ndarray, *, anchor: int, indices: np.ndarray) -> tuple[float, float]:
    if not len(indices):
        return float("inf"), float("inf")
    translation = np.linalg.norm(pose[indices, :3] - pose[anchor, :3], axis=1)
    rotation = _quaternion_distance_rad(pose[indices, 3:7], pose[anchor, 3:7])
    return float(np.max(translation)), float(np.max(rotation))


def _timing_entry(reference: int | None, observed: int | None) -> dict[str, object]:
    return {
        "reference_onset": reference,
        "observed_onset": observed,
        "delta_control_steps": (
            None if reference is None or observed is None else int(observed - reference)
        ),
        "identifiable": reference is not None and observed is not None,
    }


def evaluate_physical_functionality_full_cycle_v1(
    *,
    object_pose_wxyz: np.ndarray,
    wrist_pose_wxyz: np.ndarray,
    tip_pair_presence: np.ndarray,
    hand_object_pair_presence: np.ndarray,
    table_object_contact: np.ndarray,
    destination_region: np.ndarray,
    destination_support_contact: np.ndarray,
    reference_lift_onset: int | None,
    causal_execution: bool,
    geometry_safe: bool,
    action_bounds_safe: bool,
    no_hidden_control: bool,
    valid: np.ndarray | None = None,
    interaction_valid: np.ndarray | None = None,
    support_valid: np.ndarray | None = None,
    destination_region_valid: np.ndarray | None = None,
    destination_support_valid: np.ndarray | None = None,
    reference_events: Mapping[str, int | None] | None = None,
    contract: PhysicalFunctionalityFullCycleV1Contract | None = None,
    pick_contract: Stage16PhysicalFunctionalityV2Contract | None = None,
) -> dict[str, object]:
    """Evaluate a complete pick-place-release-retreat trace.

    Phase ordering only controls reachability.  Every reached phase has its own
    physical gates.  If an upstream phase does not pass, all later phases are
    ``NOT_REACHED`` and the composite is false.  Reference timing is returned
    only under ``DF_interaction_timing``.
    """

    frozen = contract or PhysicalFunctionalityFullCycleV1Contract()
    pose = np.asarray(object_pose_wxyz, dtype=np.float64)
    wrist = np.asarray(wrist_pose_wxyz, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7 or wrist.shape != pose.shape:
        raise ValueError("PF_FULL_CYCLE_V1_WRIST_AND_OBJECT_POSE_MUST_MATCH_[T,7]")
    count = len(pose)
    if count < frozen.persistence_control_steps:
        raise ValueError("PF_FULL_CYCLE_V1_TRACE_TOO_SHORT")
    if not np.isfinite(pose).all() or not np.isfinite(wrist).all():
        raise ValueError("PF_FULL_CYCLE_V1_POSE_MUST_BE_FINITE")

    if valid is not None and interaction_valid is not None:
        raise ValueError("PF_FULL_CYCLE_V1_INTERACTION_VALID_AND_LEGACY_VALID_BOTH_PROVIDED")
    interaction_rows = interaction_valid if interaction_valid is not None else valid
    if interaction_rows is None:
        raise ValueError("PF_FULL_CYCLE_V1_INTERACTION_VALID_REQUIRED")
    interaction_rows = _vector(interaction_rows, name="INTERACTION_VALID", count=count, dtype=bool)
    support_rows = (
        np.ones(count, dtype=bool)
        if support_valid is None
        else _vector(support_valid, name="SUPPORT_VALID", count=count, dtype=bool)
    )
    region_rows = (
        np.ones(count, dtype=bool)
        if destination_region_valid is None
        else _vector(
            destination_region_valid,
            name="DESTINATION_REGION_VALID",
            count=count,
            dtype=bool,
        )
    )
    destination_support_rows = (
        np.ones(count, dtype=bool)
        if destination_support_valid is None
        else _vector(
            destination_support_valid,
            name="DESTINATION_SUPPORT_VALID",
            count=count,
            dtype=bool,
        )
    )
    hand = _matrix(hand_object_pair_presence, name="HAND_OBJECT_PAIR_PRESENCE", count=count)
    hand_any = hand.any(axis=1) & interaction_rows
    table = _vector(table_object_contact, name="TABLE_OBJECT_CONTACT", count=count, dtype=bool)
    region = _vector(destination_region, name="DESTINATION_REGION", count=count, dtype=bool)
    region &= region_rows
    destination_support = _vector(
        destination_support_contact,
        name="DESTINATION_SUPPORT_CONTACT",
        count=count,
        dtype=bool,
    )
    destination_support &= destination_support_rows

    pick_detail = evaluate_stage16_physical_functionality_v2(
        object_pose_wxyz=pose,
        wrist_pose_wxyz=wrist,
        tip_pair_presence=tip_pair_presence,
        hand_object_pair_presence=hand_object_pair_presence,
        table_object_contact=table,
        interaction_valid=interaction_rows,
        support_valid=support_rows,
        reference_lift_onset=reference_lift_onset,
        causal_execution=causal_execution,
        geometry_safe=geometry_safe,
        action_bounds_safe=action_bounds_safe,
        no_hidden_control=no_hidden_control,
        contract=pick_contract,
    )
    actual_lift = cast(Mapping[str, object], pick_detail["actual_lift"])
    lift_value = actual_lift["onset"]
    lift_onset = int(lift_value) if isinstance(lift_value, int) else None
    pick_reason_values = cast(list[object], pick_detail["pf_v2_failure_reasons"])
    pick_reasons = [str(value) for value in pick_reason_values]
    pick = _phase(
        "PASS" if bool(pick_detail["pf_v2"]) else "FAIL",
        failure_reasons=pick_reasons,
        events={"actual_lift_onset": lift_onset},
        diagnostics={
            "authority": "Stage16PhysicalFunctionalityV2",
            "parity_preserved": True,
            "detail": pick_detail,
        },
    )

    transport_end: int | None = None
    place_onset: int | None = None
    release_onset: int | None = None
    retreat_onset: int | None = None
    if pick["status"] != "PASS":
        transport = _not_reached("PF_pick", str(pick["status"]))
    elif not bool(np.any(region_rows[lift_onset:])):
        transport = _phase(
            "NOT_IDENTIFIABLE",
            failure_reasons=["destination_region_signal_unavailable"],
        )
    else:
        assert lift_onset is not None
        transport_end = _first_persistent(
            region,
            minimum_steps=frozen.persistence_control_steps,
            start=lift_onset,
        )
        transport_reasons: list[str] = []
        progress = None
        maximum_step = None
        coupling = False
        no_initial_support_recontact = False
        if transport_end is None:
            transport_reasons.append("required_destination_region_not_reached")
        else:
            progress = float(np.linalg.norm(pose[transport_end, :3] - pose[lift_onset, :3]))
            window = slice(lift_onset, transport_end + 1)
            coupling = bool(np.all(hand_any[window]))
            no_initial_support_recontact = not bool(
                np.any(table[lift_onset:transport_end] & support_rows[lift_onset:transport_end])
            )
            steps = np.linalg.norm(
                np.diff(pose[lift_onset : transport_end + 1, :3], axis=0), axis=1
            )
            maximum_step = 0.0 if not len(steps) else float(np.max(steps))
            if not coupling:
                transport_reasons.append("hand_object_coupling_lost")
            if progress < frozen.required_transport_progress_m:
                transport_reasons.append("required_transport_progress_not_met")
            if not no_initial_support_recontact:
                transport_reasons.append("initial_support_recontact_before_placement")
            if maximum_step > frozen.maximum_object_translation_step_m:
                transport_reasons.append("object_translation_discontinuity_teleport")
        if not causal_execution:
            transport_reasons.append("causal_execution")
        if not geometry_safe:
            transport_reasons.append("geometry_safe")
        if not action_bounds_safe:
            transport_reasons.append("action_bounds_safe")
        if not no_hidden_control:
            transport_reasons.append("no_hidden_control")
        transport = _phase(
            "PASS" if not transport_reasons else "FAIL",
            failure_reasons=transport_reasons,
            events={"destination_region_onset": transport_end},
            diagnostics={
                "transport_progress_m": progress,
                "required_transport_progress_m": frozen.required_transport_progress_m,
                "maximum_object_translation_step_m": maximum_step,
                "maximum_allowed_translation_step_m": (frozen.maximum_object_translation_step_m),
                "sustained_hand_object_coupling": coupling,
                "initial_support_recontact_absent": no_initial_support_recontact,
                "strict_reference_timing_required": False,
            },
        )

    if transport["status"] != "PASS":
        place = _not_reached("PF_transport", str(transport["status"]))
    elif not bool(np.any(destination_support_rows[transport_end:])):
        place = _phase(
            "NOT_IDENTIFIABLE",
            failure_reasons=["destination_support_signal_unavailable"],
        )
    else:
        assert transport_end is not None
        place_onset = _first_persistent(
            destination_support,
            minimum_steps=frozen.persistence_control_steps,
            start=transport_end,
        )
        place_reasons: list[str] = []
        hand_supported_at_acquisition = bool(place_onset is not None and hand_any[place_onset])
        if place_onset is None:
            place_reasons.append("destination_support_never_acquired")
        elif not hand_supported_at_acquisition:
            place_reasons.append("hand_support_absent_at_destination_acquisition")
        place = _phase(
            "PASS" if not place_reasons else "FAIL",
            failure_reasons=place_reasons,
            events={"destination_support_acquisition_onset": place_onset},
            diagnostics={
                "support_schema": SUPPORT_TRANSFER_HAND_TO_SURFACE_PROXY_V1,
                "is_exact_support_force": False,
                "hand_supported_at_acquisition": hand_supported_at_acquisition,
                "support_persistence_steps": frozen.persistence_control_steps,
            },
        )

    if place["status"] != "PASS":
        release = _not_reached("PF_place", str(place["status"]))
    elif not bool(np.any(interaction_rows[place_onset:])):
        release = _phase(
            "NOT_IDENTIFIABLE", failure_reasons=["post_place_interaction_signal_unavailable"]
        )
    else:
        assert place_onset is not None
        release_onset = _first_persistent(
            (~hand_any) & interaction_rows,
            minimum_steps=frozen.persistence_control_steps,
            start=place_onset,
        )
        release_reasons: list[str] = []
        release_translation_drift = None
        release_rotation_drift = None
        if release_onset is None:
            release_reasons.append("hand_object_contact_did_not_release")
        else:
            stability_end = release_onset + frozen.persistence_control_steps
            stability_indices = np.arange(release_onset, min(stability_end, count))
            release_translation_drift, release_rotation_drift = _pose_drift(
                pose, anchor=release_onset, indices=stability_indices
            )
            if stability_end > count:
                release_reasons.append("insufficient_post_release_stability_window")
            if not bool(np.all(destination_support[release_onset:])):
                release_reasons.append("destination_support_lost_after_release")
            if bool(np.any(hand_any[release_onset:])):
                release_reasons.append("object_regrasped_after_release")
            if release_translation_drift > frozen.release_stability_translation_m:
                release_reasons.append("object_translation_unstable_during_release")
            if release_rotation_drift > frozen.release_stability_rotation_rad:
                release_reasons.append("object_rotation_unstable_during_release")
        release = _phase(
            "PASS" if not release_reasons else "FAIL",
            failure_reasons=release_reasons,
            events={"release_onset": release_onset},
            diagnostics={
                "release_translation_drift_m": release_translation_drift,
                "release_rotation_drift_rad": release_rotation_drift,
                "stable_support_required_through_terminal": True,
                "no_regrasp_required_through_terminal": True,
            },
        )

    if release["status"] != "PASS":
        retreat = _not_reached("PF_release", str(release["status"]))
    else:
        assert release_onset is not None
        clearance = np.linalg.norm(wrist[:, :3] - pose[:, :3], axis=1)
        retreat_onset = _first_persistent(
            clearance >= frozen.retreat_clearance_m,
            minimum_steps=frozen.persistence_control_steps,
            start=release_onset,
        )
        retreat_reasons: list[str] = []
        terminal_translation_drift = None
        terminal_rotation_drift = None
        if retreat_onset is None:
            retreat_reasons.append("hand_did_not_clear_object")
        else:
            terminal_indices = np.arange(release_onset, count)
            terminal_translation_drift, terminal_rotation_drift = _pose_drift(
                pose, anchor=release_onset, indices=terminal_indices
            )
            if bool(np.any(hand_any[release_onset:])):
                retreat_reasons.append("post_release_hand_object_interaction")
            if not bool(np.all(destination_support[release_onset:])):
                retreat_reasons.append("placed_object_support_disturbed")
            if terminal_translation_drift > frozen.retreat_object_translation_m:
                retreat_reasons.append("retreat_disturbed_object_translation")
            if terminal_rotation_drift > frozen.retreat_object_rotation_rad:
                retreat_reasons.append("retreat_disturbed_object_rotation")
        retreat = _phase(
            "PASS" if not retreat_reasons else "FAIL",
            failure_reasons=retreat_reasons,
            events={"retreat_clearance_onset": retreat_onset},
            diagnostics={
                "required_clearance_m": frozen.retreat_clearance_m,
                "terminal_translation_drift_m": terminal_translation_drift,
                "terminal_rotation_drift_rad": terminal_rotation_drift,
                "exact_return_to_initial_pose_required": False,
            },
        )

    phases = {
        "PF_pick": pick,
        "PF_transport": transport,
        "PF_place": place,
        "PF_release": release,
        "PF_retreat": retreat,
    }
    full_pass = all(phase["status"] == "PASS" for phase in phases.values())
    full_cycle = _phase(
        "PASS" if full_pass else "FAIL",
        failure_reasons=[
            f"{name}_{str(phase['status']).lower()}"
            for name, phase in phases.items()
            if phase["status"] != "PASS"
        ],
        events={
            "actual_lift_onset": lift_onset,
            "destination_region_onset": transport_end,
            "destination_support_acquisition_onset": place_onset,
            "release_onset": release_onset,
            "retreat_clearance_onset": retreat_onset,
        },
    )

    references = dict(reference_events or {})
    pick_timing = cast(Mapping[str, int | None], pick_detail["interaction_timing"])
    timing = {
        "schema_version": DF_INTERACTION_TIMING_FULL_CYCLE_V1,
        "diagnostic_only": True,
        "included_in_pf_hard_gate": False,
        "source_contact_timing": _timing_entry(
            references.get("source_contact"), pick_timing["first_hand_object_contact"]
        ),
        "persistent_contact_timing": _timing_entry(
            references.get("persistent_contact"),
            pick_timing["persistent_multifinger_contact"],
        ),
        "pickup_timing": _timing_entry(references.get("pickup"), lift_onset),
        "place_timing": _timing_entry(references.get("place"), place_onset),
        "release_timing": _timing_entry(references.get("release"), release_onset),
    }
    return {
        "schema_version": PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1,
        "contract": frozen.as_dict(),
        **phases,
        "PF_full_cycle": full_cycle,
        "pf_full_cycle": full_pass,
        "DF_interaction_timing": timing,
    }


__all__ = [
    "DF_INTERACTION_TIMING_FULL_CYCLE_V1",
    "PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1",
    "SUPPORT_TRANSFER_HAND_TO_SURFACE_PROXY_V1",
    "PhysicalFunctionalityFullCycleV1Contract",
    "evaluate_physical_functionality_full_cycle_v1",
]
