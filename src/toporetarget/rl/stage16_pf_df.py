"""Offline Stage16 contact-timing, angular-twist, PF, and DF helpers.

The helpers in this module only interpret recorded arrays.  They do not load a
policy, advance a simulator, or alter the immutable Evaluation Suite V2 and
Stage16 Dynamic Physical Qualification V1 contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Final, cast

import numpy as np

from toporetarget.evaluation.reference_contact_contract import FINGER_ORDER
from toporetarget.rl.physical_evaluation import persistent_mask
from toporetarget.rl.reference_tracking.reference_kinematics import (
    derive_angular_velocity_world_wxyz,
    quaternion_to_matrix_wxyz,
    so3_log,
)

CONTACT_TIMING_SCHEMA: Final = "Stage16ContactTimingLayerAttributionV1"
ANGULAR_AUDIT_SCHEMA: Final = "Stage16AngularTwistAuditV1"
PHYSICAL_FUNCTIONALITY_SCHEMA: Final = "Stage16PhysicalFunctionalityV1"
DEMONSTRATION_FIDELITY_SCHEMA: Final = "Stage16DemonstrationFidelityV1"


@dataclass(frozen=True)
class ContactTimingContract:
    """Frozen layer-attribution semantics, independent of observed outcomes."""

    schema_version: str = CONTACT_TIMING_SCHEMA
    finger_order: tuple[str, ...] = FINGER_ORDER
    raw_contact_authority: str = "StrictV4SourceContactMaskContractV1"
    retarget_contact_authority: str = "ReferenceContactContractV2.strong_contact_expected"
    retarget_strong_distance_m: float = 0.02
    actual_contact_authority: str = "named PhysX fingertip-active-object pair presence"
    persistence_control_steps: int = 3
    multifinger_minimum: int = 2
    material_delay_control_steps: int = 3
    ready_on_lift_is_prelift_ready: bool = True

    def __post_init__(self) -> None:
        if self.finger_order != FINGER_ORDER:
            raise ValueError("CONTACT_TIMING_FINGER_ORDER_DRIFT")
        if self.retarget_strong_distance_m != 0.02:
            raise ValueError("CONTACT_TIMING_RETARGET_DISTANCE_AUTHORITY_DRIFT")
        if (self.persistence_control_steps, self.multifinger_minimum) != (3, 2):
            raise ValueError("CONTACT_TIMING_PERSISTENCE_CONTRACT_DRIFT")
        if self.material_delay_control_steps != self.persistence_control_steps:
            raise ValueError("CONTACT_TIMING_MATERIAL_DELAY_CONTRACT_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AngularAuditContract:
    """Frozen structural audit rules; no dynamic-fidelity threshold is tuned."""

    schema_version: str = ANGULAR_AUDIT_SCHEMA
    estimator: str = "ReferenceKinematicsV2.SO3_log_centered_world_with_one_sided_endpoints"
    transient_max_consecutive_control_steps: int = 2
    persistent_min_consecutive_control_steps: int = 3
    contributor_primary_fraction: float = 0.5
    contributor_partial_fraction: float = 0.25
    endpoint_frames: tuple[int, int] = (0, 320)
    threshold_source: str = "legacy_inherited_contact_or_free_V1_limits_unchanged"
    angular_threshold_tuned: bool = False

    def __post_init__(self) -> None:
        if self.transient_max_consecutive_control_steps != 2:
            raise ValueError("ANGULAR_AUDIT_TRANSIENT_CONTRACT_DRIFT")
        if self.persistent_min_consecutive_control_steps != 3:
            raise ValueError("ANGULAR_AUDIT_PERSISTENCE_CONTRACT_DRIFT")
        if (self.contributor_partial_fraction, self.contributor_primary_fraction) != (0.25, 0.5):
            raise ValueError("ANGULAR_AUDIT_ATTRIBUTION_FRACTION_DRIFT")
        if self.angular_threshold_tuned:
            raise ValueError("ANGULAR_THRESHOLD_TUNING_FORBIDDEN")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalFunctionalityContract:
    """Physical task outcome gates, deliberately independent of fidelity."""

    schema_version: str = PHYSICAL_FUNCTIONALITY_SCHEMA
    persistent_control_steps: int = 3
    multifinger_minimum: int = 2
    lift_threshold_m: float = 0.05
    hard_gates: tuple[str, ...] = (
        "causal_execution",
        "geometry_safe",
        "action_bounds_safe",
        "prelift_multifinger_grasp_ready",
        "lift_success",
        "no_hidden_control",
    )
    named_source_contact_readiness: str = "reported_supporting_metric_not_hard_gate_v1"
    support_transfer: str = "reported_supporting_metric_no_frozen_fraction_threshold"
    hand_object_coupling: str = "reported_supporting_metric_no_frozen_threshold"

    def __post_init__(self) -> None:
        if (self.persistent_control_steps, self.multifinger_minimum, self.lift_threshold_m) != (
            3,
            2,
            0.05,
        ):
            raise ValueError("PHYSICAL_FUNCTIONALITY_FROZEN_GATE_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DemonstrationFidelityContract:
    """Dimensioned fidelity profile using only previously frozen thresholds."""

    schema_version: str = DEMONSTRATION_FIDELITY_SCHEMA
    object_rotation_threshold_deg: float = 30.0
    object_translation_threshold_cm: float = 3.0
    hand_joint_threshold_cm: float = 8.0
    fingertip_threshold_cm: float = 6.0
    terminal_window_control_steps: int = 20
    threshold_provenance: str = "LEGACY_INHERITED_NOT_NEWLY_VALIDATED"
    overall_boolean_defined: bool = False

    def __post_init__(self) -> None:
        if (
            self.object_rotation_threshold_deg,
            self.object_translation_threshold_cm,
            self.hand_joint_threshold_cm,
            self.fingertip_threshold_cm,
            self.terminal_window_control_steps,
        ) != (30.0, 3.0, 8.0, 6.0, 20):
            raise ValueError("DEMONSTRATION_FIDELITY_FROZEN_GATE_DRIFT")
        if self.overall_boolean_defined:
            raise ValueError("DEMONSTRATION_FIDELITY_UNVALIDATED_OVERALL_BOOL_FORBIDDEN")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _mask_2d(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.ndim != 2 or result.shape[1] != len(FINGER_ORDER):
        raise ValueError(f"{name}_MUST_BE_[T,5]")
    return result


def first_true(mask: np.ndarray) -> int | None:
    """Return the first true index, or ``None`` when the event never occurs."""

    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return None if not len(indices) else int(indices[0])


def persistent_finger_mask(mask: np.ndarray, *, minimum_steps: int = 3) -> np.ndarray:
    """Apply the established Stage16 run-length convention to each named finger."""

    values = _mask_2d(mask, name="PERSISTENT_FINGER_MASK")
    return np.stack(
        [persistent_mask(values[:, index], minimum_steps=minimum_steps) for index in range(5)],
        axis=-1,
    )


def contact_timing_metrics(
    *,
    raw_contact: np.ndarray,
    retarget_contact: np.ndarray,
    actual_contact: np.ndarray,
    actual_valid: np.ndarray,
    lift_onset: int,
    timestamps_s: np.ndarray,
    raw_timestamps_s: np.ndarray,
    raw_frame_float: np.ndarray,
    contract: ContactTimingContract | None = None,
) -> dict[str, object]:
    """Attribute named-contact readiness across raw, retarget, and PhysX layers."""

    frozen = contract or ContactTimingContract()
    raw = _mask_2d(raw_contact, name="RAW_CONTACT")
    retarget = _mask_2d(retarget_contact, name="RETARGET_CONTACT")
    actual = _mask_2d(actual_contact, name="ACTUAL_CONTACT")
    if raw.shape != retarget.shape or raw.shape != actual.shape:
        raise ValueError("CONTACT_TIMING_LAYER_SHAPE_MISMATCH")
    valid = np.asarray(actual_valid, dtype=bool)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    raw_timestamps = np.asarray(raw_timestamps_s, dtype=np.float64)
    raw_frames = np.asarray(raw_frame_float, dtype=np.float64)
    if (
        valid.shape != (len(raw),)
        or timestamps.shape != valid.shape
        or raw_timestamps.shape != valid.shape
        or raw_frames.shape != valid.shape
    ):
        raise ValueError("CONTACT_TIMING_TIME_OR_VALID_SHAPE_INVALID")
    if (
        not np.isfinite(timestamps).all()
        or not np.isfinite(raw_timestamps).all()
        or not np.all(np.diff(timestamps) > 0.0)
        or not np.all(np.diff(raw_timestamps) > 0.0)
    ):
        raise ValueError("CONTACT_TIMING_TIMESTAMPS_INVALID")
    if lift_onset < 0 or lift_onset >= len(raw):
        raise ValueError("CONTACT_TIMING_LIFT_ONSET_INVALID")
    actual = actual & valid[:, None]
    raw_persistent = persistent_finger_mask(raw, minimum_steps=frozen.persistence_control_steps)
    retarget_persistent = persistent_finger_mask(
        retarget, minimum_steps=frozen.persistence_control_steps
    )
    actual_persistent = persistent_finger_mask(
        actual, minimum_steps=frozen.persistence_control_steps
    )

    def ready(mask: np.ndarray) -> int | None:
        return first_true(mask.sum(axis=-1) >= frozen.multifinger_minimum)

    raw_ready = ready(raw_persistent)
    retarget_ready = ready(retarget_persistent)
    actual_ready = ready(actual_persistent)

    def delta(later: int | None, earlier: int | None) -> int | None:
        return None if later is None or earlier is None else later - earlier

    def margin(onset: int | None) -> int | None:
        return None if onset is None else lift_onset - onset

    def seconds(frames: int | None) -> float | None:
        if frames is None:
            return None
        return float(frames * np.median(np.diff(timestamps)))

    per_finger: list[dict[str, object]] = []
    for index, finger in enumerate(frozen.finger_order):
        raw_onset = first_true(raw[:, index])
        raw_persistent_onset = first_true(raw_persistent[:, index])
        retarget_onset = first_true(retarget[:, index])
        retarget_persistent_onset = first_true(retarget_persistent[:, index])
        actual_first = first_true(actual[:, index])
        actual_persistent_onset = first_true(actual_persistent[:, index])
        per_finger.append(
            {
                "finger": finger,
                "raw_onset": raw_onset,
                "raw_onset_runtime_time_s": (
                    None if raw_onset is None else float(timestamps[raw_onset])
                ),
                "raw_onset_source_time_s": (
                    None if raw_onset is None else float(raw_timestamps[raw_onset])
                ),
                "raw_onset_source_frame_float": (
                    None if raw_onset is None else float(raw_frames[raw_onset])
                ),
                "raw_persistent": raw_persistent_onset,
                "retarget_onset": retarget_onset,
                "retarget_onset_runtime_time_s": (
                    None if retarget_onset is None else float(timestamps[retarget_onset])
                ),
                "retarget_persistent": retarget_persistent_onset,
                "actual_first": actual_first,
                "actual_first_runtime_time_s": (
                    None if actual_first is None else float(timestamps[actual_first])
                ),
                "actual_persistent": actual_persistent_onset,
                "actual_persistent_runtime_time_s": (
                    None
                    if actual_persistent_onset is None
                    else float(timestamps[actual_persistent_onset])
                ),
                "lift_onset": lift_onset,
                "raw_to_retarget_frames": delta(retarget_onset, raw_onset),
                "raw_to_retarget_seconds": seconds(delta(retarget_onset, raw_onset)),
                "retarget_to_actual_first_frames": delta(actual_first, retarget_onset),
                "retarget_to_actual_first_seconds": seconds(delta(actual_first, retarget_onset)),
                "retarget_to_actual_persistent_frames": delta(
                    actual_persistent_onset, retarget_persistent_onset
                ),
                "retarget_to_actual_persistent_seconds": seconds(
                    delta(actual_persistent_onset, retarget_persistent_onset)
                ),
                "raw_required_at_lift": bool(raw[lift_onset, index]),
                "retarget_ready_at_lift": bool(retarget_persistent[lift_onset, index]),
                "actual_persistent_at_lift": bool(actual_persistent[lift_onset, index]),
            }
        )

    source_required_at_lift = raw[lift_onset]
    required_count = int(source_required_at_lift.sum())
    identity_matched = actual_persistent[lift_onset] & source_required_at_lift
    return {
        "raw_ready": raw_ready,
        "retarget_ready": retarget_ready,
        "actual_ready": actual_ready,
        "lift_onset": lift_onset,
        "lift_runtime_time_s": float(timestamps[lift_onset]),
        "lift_raw_time_s": float(raw_timestamps[lift_onset]),
        "lift_raw_frame_float": float(raw_frames[lift_onset]),
        "raw_ready_runtime_time_s": (None if raw_ready is None else float(timestamps[raw_ready])),
        "raw_ready_source_time_s": (
            None if raw_ready is None else float(raw_timestamps[raw_ready])
        ),
        "raw_ready_source_frame_float": (
            None if raw_ready is None else float(raw_frames[raw_ready])
        ),
        "retarget_ready_runtime_time_s": (
            None if retarget_ready is None else float(timestamps[retarget_ready])
        ),
        "actual_ready_runtime_time_s": (
            None if actual_ready is None else float(timestamps[actual_ready])
        ),
        "raw_margin_frames": margin(raw_ready),
        "raw_margin_seconds": seconds(margin(raw_ready)),
        "retarget_margin_frames": margin(retarget_ready),
        "retarget_margin_seconds": seconds(margin(retarget_ready)),
        "actual_margin_frames": margin(actual_ready),
        "actual_margin_seconds": seconds(margin(actual_ready)),
        "raw_to_retarget_delay_frames": delta(retarget_ready, raw_ready),
        "raw_to_retarget_delay_seconds": seconds(delta(retarget_ready, raw_ready)),
        "retarget_to_actual_delay_frames": delta(actual_ready, retarget_ready),
        "retarget_to_actual_delay_seconds": seconds(delta(actual_ready, retarget_ready)),
        "prelift_multifinger_grasp_ready": bool(
            actual_ready is not None and actual_ready <= lift_onset
        ),
        "source_required_fingers_at_lift": [
            finger
            for finger, selected in zip(frozen.finger_order, source_required_at_lift, strict=True)
            if selected
        ],
        "actual_persistent_fingers_at_lift": [
            finger
            for finger, selected in zip(
                frozen.finger_order, actual_persistent[lift_onset], strict=True
            )
            if selected
        ],
        "named_source_contact_match_at_lift": bool(
            required_count > 0 and int(identity_matched.sum()) == required_count
        ),
        "named_source_contact_recall_at_lift": (
            None if required_count == 0 else float(identity_matched.sum() / required_count)
        ),
        "per_finger": per_finger,
    }


def timing_attribution(
    negative_summary: Mapping[str, object],
    positive_summary: Mapping[str, object],
    *,
    contract: ContactTimingContract | None = None,
) -> dict[str, object]:
    """Apply the frozen layer-delay decision tree to negative and positive controls."""

    frozen = contract or ContactTimingContract()

    def integer(row: Mapping[str, object], name: str) -> int | None:
        value = row.get(name)
        return None if value is None else int(cast(int, value))

    raw_margin = integer(negative_summary, "raw_margin_frames_median")
    retarget_margin = integer(negative_summary, "retarget_margin_frames_median")
    actual_margin = integer(negative_summary, "actual_margin_frames_median")
    raw_delay = integer(negative_summary, "raw_to_retarget_delay_frames_median")
    physics_delay = integer(negative_summary, "retarget_to_actual_delay_frames_median")
    positive_actual_margin = integer(positive_summary, "actual_margin_frames_median")
    if None in (raw_margin, retarget_margin, actual_margin, raw_delay, physics_delay):
        root = "INCONCLUSIVE"
        confidence = "LOW"
    else:
        assert raw_margin is not None
        assert retarget_margin is not None
        assert actual_margin is not None
        assert raw_delay is not None
        assert physics_delay is not None
        raw_layer = (
            raw_margin >= frozen.material_delay_control_steps
            and raw_delay >= frozen.material_delay_control_steps
            and retarget_margin < raw_margin
        )
        physics_layer = (
            raw_margin >= 0
            and retarget_margin >= 0
            and actual_margin < 0
            and physics_delay >= frozen.material_delay_control_steps
        )
        if raw_layer and physics_layer:
            root = "MULTI_STAGE_TIMING_LOSS_PRIMARY"
        elif raw_layer:
            root = "RAW_TO_RETARGET_TIMING_LOSS_PRIMARY"
        elif physics_layer:
            root = "RETARGET_TO_PHYSICS_CONTACT_ACQUISITION_LAG_PRIMARY"
        elif raw_margin >= 0 and retarget_margin >= 0 and actual_margin >= 0:
            root = "NO_MATERIAL_TIMING_LOSS"
        else:
            root = "INCONCLUSIVE"
        confidence = (
            "LOW"
            if root == "INCONCLUSIVE"
            else "HIGH"
            if positive_actual_margin is not None and positive_actual_margin >= 0
            else "MEDIUM"
        )
    return {
        "CONTACT_TIMING_LAYER_ROOT_CAUSE": root,
        "CONFIDENCE": confidence,
        "negative_clip": dict(negative_summary),
        "positive_control": dict(positive_summary),
    }


def _norm(value: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(value, dtype=np.float64), axis=-1)


def distribution(values: np.ndarray) -> dict[str, float]:
    """Return the fixed descriptive distribution used throughout the audit."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("ANGULAR_AUDIT_DISTRIBUTION_INVALID")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open true runs in a one-dimensional boolean series."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("ANGULAR_AUDIT_RUN_MASK_MUST_BE_1D")
    changes = np.diff(np.concatenate(([False], values, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def angular_episode_audit(
    *,
    actual_object_pose_wxyz: np.ndarray,
    actual_object_twist_world: np.ndarray,
    reference_object_pose_wxyz: np.ndarray,
    reference_object_twist_world: np.ndarray,
    wrist_pose_wxyz: np.ndarray,
    wrist_twist_world: np.ndarray,
    timestamps_s: np.ndarray,
    phase: np.ndarray,
    hand_object_contact: np.ndarray,
    valid: np.ndarray,
    contact_angular_limit_radps: float,
    free_angular_limit_radps: float,
    terminal_window_control_steps: int = 20,
    contract: AngularAuditContract | None = None,
) -> dict[str, object]:
    """Compare trace and pose-derived world angular velocities for one episode."""

    frozen = contract or AngularAuditContract()
    actual_pose = np.asarray(actual_object_pose_wxyz, dtype=np.float64)
    reference_pose = np.asarray(reference_object_pose_wxyz, dtype=np.float64)
    wrist_pose = np.asarray(wrist_pose_wxyz, dtype=np.float64)
    actual_twist = np.asarray(actual_object_twist_world, dtype=np.float64)
    reference_twist = np.asarray(reference_object_twist_world, dtype=np.float64)
    wrist_twist = np.asarray(wrist_twist_world, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    labels = np.asarray(phase).astype("U24")
    contacts = np.asarray(hand_object_contact, dtype=bool)
    rows = np.asarray(valid, dtype=bool)
    frame_count = len(timestamps)
    if (
        actual_pose.shape != (frame_count, 7)
        or reference_pose.shape != (frame_count, 7)
        or wrist_pose.shape != (frame_count, 7)
        or actual_twist.shape != (frame_count, 6)
        or reference_twist.shape != (frame_count, 6)
        or wrist_twist.shape != (frame_count, 6)
        or labels.shape != (frame_count,)
        or contacts.shape != (frame_count,)
        or rows.shape != (frame_count,)
    ):
        raise ValueError("ANGULAR_AUDIT_TRACE_SHAPE_INVALID")
    if not rows.any() or not np.isfinite(actual_pose).all() or not np.isfinite(actual_twist).all():
        raise ValueError("ANGULAR_AUDIT_TRACE_VALUES_INVALID")
    actual_pose_omega = derive_angular_velocity_world_wxyz(actual_pose[:, 3:], timestamps)
    reference_pose_omega = derive_angular_velocity_world_wxyz(reference_pose[:, 3:], timestamps)
    wrist_pose_omega = derive_angular_velocity_world_wxyz(wrist_pose[:, 3:], timestamps)
    actual_trace_omega = actual_twist[:, 3:]
    reference_trace_omega = reference_twist[:, 3:]
    wrist_trace_omega = wrist_twist[:, 3:]
    measurement_error = _norm(actual_trace_omega - actual_pose_omega)
    reference_estimator_error = _norm(reference_trace_omega - reference_pose_omega)
    delta_trace = _norm(actual_trace_omega - reference_trace_omega)
    delta_pose = _norm(actual_pose_omega - reference_trace_omega)
    relative_trace = _norm(actual_trace_omega - wrist_trace_omega)
    relative_pose = _norm(actual_pose_omega - wrist_pose_omega)
    limits = np.where(contacts, contact_angular_limit_radps, free_angular_limit_radps)
    exceed_trace = (delta_trace > limits) & rows
    exceed_pose = (delta_pose > limits) & rows
    runs = true_runs(exceed_trace)
    selected = np.flatnonzero(rows)
    terminal = selected[-min(terminal_window_control_steps, len(selected)) :]
    actual_rotation = quaternion_to_matrix_wxyz(actual_pose[:, 3:])
    incremental_angle = np.zeros(frame_count, dtype=np.float64)
    incremental_angle[1:] = _norm(
        so3_log(actual_rotation[1:] @ np.swapaxes(actual_rotation[:-1], -1, -2))
    )
    angular_acceleration = np.zeros(frame_count, dtype=np.float64)
    angular_acceleration[1:] = _norm(
        np.diff(actual_pose_omega, axis=0) / np.diff(timestamps)[:, None]
    )

    phase_rows: list[dict[str, object]] = []
    phase_groups = {
        "APPROACH": labels == "APPROACH",
        "CONTACT": labels == "CONTACT",
        "GRASP": labels == "GRASP",
        "LIFT": labels == "LIFT",
        "LATE_MOTION": np.isin(labels, ("MANIPULATION", "TERMINAL")),
    }
    for name, mask in phase_groups.items():
        selected_phase = mask & rows
        if not selected_phase.any():
            continue
        phase_rows.append(
            {
                "phase": name,
                "frame_count": int(selected_phase.sum()),
                "Delta_omega_trace_mean_radps": float(delta_trace[selected_phase].mean()),
                "Delta_omega_trace_p95_radps": float(
                    np.quantile(delta_trace[selected_phase], 0.95)
                ),
                "Delta_omega_trace_max_radps": float(delta_trace[selected_phase].max()),
                "Delta_omega_pose_mean_radps": float(delta_pose[selected_phase].mean()),
                "Delta_omega_pose_p95_radps": float(np.quantile(delta_pose[selected_phase], 0.95)),
                "Delta_omega_pose_max_radps": float(delta_pose[selected_phase].max()),
                "trace_pose_mismatch_mean_radps": float(measurement_error[selected_phase].mean()),
                "trace_pose_mismatch_p95_radps": float(
                    np.quantile(measurement_error[selected_phase], 0.95)
                ),
                "relative_angular_twist_mean_radps": float(relative_trace[selected_phase].mean()),
                "relative_angular_twist_p95_radps": float(
                    np.quantile(relative_trace[selected_phase], 0.95)
                ),
            }
        )

    return {
        "series": {
            "omega_ref": reference_trace_omega,
            "omega_ref_pose": reference_pose_omega,
            "omega_actual_trace": actual_trace_omega,
            "omega_actual_pose": actual_pose_omega,
            "Delta_omega_trace": delta_trace,
            "Delta_omega_pose": delta_pose,
            "trace_pose_mismatch": measurement_error,
            "reference_estimator_mismatch": reference_estimator_error,
            "relative_angular_twist_trace": relative_trace,
            "relative_angular_twist_pose": relative_pose,
            "incremental_rotation_angle": incremental_angle,
            "angular_acceleration_proxy": angular_acceleration,
            "angular_limit": limits,
            "exceedance_trace": exceed_trace,
            "exceedance_pose": exceed_pose,
        },
        "measurement_consistency": distribution(measurement_error[rows]),
        "reference_estimator_consistency": distribution(reference_estimator_error[rows]),
        "Delta_omega_trace": distribution(delta_trace[rows]),
        "Delta_omega_pose": distribution(delta_pose[rows]),
        "relative_angular_twist_trace": distribution(relative_trace[rows]),
        "relative_angular_twist_pose": distribution(relative_pose[rows]),
        "high_frequency": {
            "incremental_rotation_angle": distribution(incremental_angle[rows]),
            "angular_acceleration_proxy": distribution(angular_acceleration[rows]),
        },
        "exceedance": {
            "frame_fraction": float(exceed_trace[rows].mean()),
            "pose_frame_fraction": float(exceed_pose[rows].mean()),
            "longest_consecutive_run": max((end - start for start, end in runs), default=0),
            "number_of_segments": len(runs),
            "transient_segment_count": sum(
                end - start <= frozen.transient_max_consecutive_control_steps for start, end in runs
            ),
            "persistent_segment_count": sum(
                end - start >= frozen.persistent_min_consecutive_control_steps
                for start, end in runs
            ),
        },
        "terminal": {
            "frame_count": int(len(terminal)),
            "trace_pass_under_v1": bool(np.all(delta_trace[terminal] <= limits[terminal])),
            "pose_pass_under_v1": bool(np.all(delta_pose[terminal] <= limits[terminal])),
            "trace_exceedance_count": int(exceed_trace[terminal].sum()),
            "pose_exceedance_count": int(exceed_pose[terminal].sum()),
            "endpoint_trace_error_radps": float(delta_trace[-1]),
            "endpoint_pose_error_radps": float(delta_pose[-1]),
            "endpoint_reference_estimator_mismatch_radps": float(reference_estimator_error[-1]),
        },
        "phase_rows": phase_rows,
    }


def angular_root_cause(
    aggregate: Mapping[str, object], *, contract: AngularAuditContract | None = None
) -> dict[str, object]:
    """Classify the aggregate with dimensionless, pre-frozen contribution rules."""

    frozen = contract or AngularAuditContract()
    trace = aggregate["Delta_omega_trace"]
    measurement = aggregate["measurement_consistency"]
    reference = aggregate["reference_estimator_consistency"]
    relative = aggregate["relative_angular_twist_trace"]
    exceedance = aggregate["exceedance"]
    if not all(
        isinstance(value, Mapping)
        for value in (trace, measurement, reference, relative, exceedance)
    ):
        raise ValueError("ANGULAR_AUDIT_AGGREGATE_STRUCTURE_INVALID")
    trace = cast(Mapping[str, object], trace)
    measurement = cast(Mapping[str, object], measurement)
    reference = cast(Mapping[str, object], reference)
    relative = cast(Mapping[str, object], relative)
    exceedance = cast(Mapping[str, object], exceedance)
    denominator_p95 = max(float(cast(float, trace["p95"])), 1.0e-12)
    denominator_mean = max(float(cast(float, trace["mean"])), 1.0e-12)
    measurement_fraction = float(cast(float, measurement["p95"])) / denominator_p95
    reference_fraction = float(cast(float, reference["p95"])) / denominator_p95
    relative_fraction = float(cast(float, relative["mean"])) / denominator_mean
    measurement_primary = measurement_fraction >= frozen.contributor_primary_fraction
    reference_primary = reference_fraction >= frozen.contributor_primary_fraction
    persistent = int(cast(int, exceedance["longest_consecutive_run_max"])) >= (
        frozen.persistent_min_consecutive_control_steps
    )
    transient_only = not persistent and int(cast(int, exceedance["number_of_segments_total"])) > 0
    relative_primary = relative_fraction >= frozen.contributor_primary_fraction and persistent
    if measurement_primary and reference_primary:
        root = "MULTI_FACTOR_PRIMARY"
    elif measurement_primary:
        root = "ANGULAR_VELOCITY_MEASUREMENT_SEMANTICS_MISMATCH_PRIMARY"
    elif reference_primary:
        root = "REFERENCE_ANGULAR_ESTIMATION_ARTIFACT_PRIMARY"
    elif relative_primary:
        root = "HAND_OBJECT_RELATIVE_ROTATION_PRIMARY"
    elif transient_only:
        root = "TRANSIENT_ANGULAR_SPIKES_PRIMARY"
    elif persistent:
        root = "PERSISTENT_ROTATIONAL_WOBBLE_PRIMARY"
    else:
        root = "INCONCLUSIVE"
    if measurement_fraction < frozen.contributor_partial_fraction:
        match = "YES"
    elif measurement_fraction >= frozen.contributor_primary_fraction:
        match = "NO"
    else:
        match = "PARTIALLY"
    if persistent and int(cast(int, exceedance["transient_segment_count_total"])) > 0:
        structure = "MIXED"
    elif persistent:
        structure = "PERSISTENT"
    elif transient_only:
        structure = "MOSTLY_TRANSIENT_SPIKES"
    else:
        structure = "PHASE_LOCALIZED"
    return {
        "ANGULAR_TWIST_ROOT_CAUSE": root,
        "DOES_TRACE_OMEGA_MATCH_POSE_DERIVED_OMEGA": match,
        "IS_LARGE_DELTA_OMEGA": structure,
        "measurement_contribution_fraction_p95": measurement_fraction,
        "reference_estimator_contribution_fraction_p95": reference_fraction,
        "relative_rotation_fraction_mean": relative_fraction,
        "ANGULAR_THRESHOLD_TUNED": "NO",
    }


def terminal_threshold_pass(
    error: np.ndarray,
    *,
    contact: np.ndarray,
    valid: np.ndarray,
    contact_limit: float,
    free_limit: float,
    terminal_steps: int = 20,
) -> bool:
    """Apply unchanged contact/free V1 limits to one scalar error series."""

    values = np.asarray(error, dtype=np.float64)
    contacts = np.asarray(contact, dtype=bool)
    rows = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or contacts.shape != values.shape or rows.shape != values.shape:
        raise ValueError("DEMONSTRATION_FIDELITY_TERMINAL_SERIES_INVALID")
    selected = np.flatnonzero(rows)
    if not len(selected):
        return False
    terminal = selected[-min(terminal_steps, len(selected)) :]
    limits = np.where(contacts[terminal], contact_limit, free_limit)
    return bool(np.all(values[terminal] <= limits))


def evaluate_physical_functionality(
    *,
    causal_execution: bool,
    geometry_safe: bool,
    action_bounds_safe: bool,
    prelift_multifinger_grasp_ready: bool,
    lift_dz_m: float,
    no_hidden_control: bool,
    contract: PhysicalFunctionalityContract | None = None,
) -> dict[str, object]:
    """Compose PF without consulting pose or twist fidelity metrics."""

    frozen = contract or PhysicalFunctionalityContract()
    gates = {
        "causal_execution": bool(causal_execution),
        "geometry_safe": bool(geometry_safe),
        "action_bounds_safe": bool(action_bounds_safe),
        "prelift_multifinger_grasp_ready": bool(prelift_multifinger_grasp_ready),
        "lift_success": bool(float(lift_dz_m) >= frozen.lift_threshold_m),
        "no_hidden_control": bool(no_hidden_control),
    }
    reasons = [name for name, passed in gates.items() if not passed]
    return {"pf": not reasons, "pf_failure_reasons": reasons, **gates}


def evaluate_demonstration_fidelity(
    *,
    e_r_mean_deg: float,
    e_t_mean_cm: float,
    e_j_mean_cm: float,
    e_ft_mean_cm: float,
    linear_pass_under_v1: bool,
    angular_trace_pass_under_v1: bool,
    angular_pose_pass_under_v1: bool,
    contract: DemonstrationFidelityContract | None = None,
) -> dict[str, object]:
    """Return the orthogonal DF profile; no unvalidated total boolean is created."""

    frozen = contract or DemonstrationFidelityContract()
    pose = bool(
        e_r_mean_deg < frozen.object_rotation_threshold_deg
        and e_t_mean_cm < frozen.object_translation_threshold_cm
        and e_j_mean_cm < frozen.hand_joint_threshold_cm
        and e_ft_mean_cm < frozen.fingertip_threshold_cm
    )
    return {
        "df_pose": pose,
        "df_linear": bool(linear_pass_under_v1),
        "df_angular": bool(angular_trace_pass_under_v1),
        "df_angular_pose_derived": bool(angular_pose_pass_under_v1),
        "DF_POSE_STATUS": "PASS" if pose else "FAIL",
        "DF_LINEAR_STATUS": (
            "PASS_UNDER_CURRENT_V1_THRESHOLD"
            if linear_pass_under_v1
            else "FAIL_UNDER_CURRENT_V1_THRESHOLD"
        ),
        "DF_ANGULAR_STATUS": (
            "PASS_UNDER_CURRENT_V1_THRESHOLD"
            if angular_trace_pass_under_v1
            else "FAIL_UNDER_CURRENT_V1_THRESHOLD"
        ),
        "THRESHOLD_PROVENANCE": frozen.threshold_provenance,
        "DF_OVERALL_BOOL": "NOT_DEFINED",
    }


__all__ = [
    "ANGULAR_AUDIT_SCHEMA",
    "CONTACT_TIMING_SCHEMA",
    "DEMONSTRATION_FIDELITY_SCHEMA",
    "PHYSICAL_FUNCTIONALITY_SCHEMA",
    "AngularAuditContract",
    "ContactTimingContract",
    "DemonstrationFidelityContract",
    "PhysicalFunctionalityContract",
    "angular_episode_audit",
    "angular_root_cause",
    "contact_timing_metrics",
    "distribution",
    "evaluate_demonstration_fidelity",
    "evaluate_physical_functionality",
    "first_true",
    "persistent_finger_mask",
    "terminal_threshold_pass",
    "timing_attribution",
    "true_runs",
]
