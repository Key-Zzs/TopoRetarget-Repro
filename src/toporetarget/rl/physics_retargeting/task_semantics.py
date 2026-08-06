"""Automatic Stage 16-D task-semantic extraction from frozen references."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .contact_topology import collapse_contact_records, consecutive_runs
from .contracts import ContactWindowV1, TaskSemanticContractV1


def quaternion_angle_deg(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    first = np.asarray(first_wxyz, dtype=np.float64)
    second = np.asarray(second_wxyz, dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("quaternion angle requires two wxyz quaternions")
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError("relative transform must be finite [4,4]")
    return tuple(tuple(float(item) for item in row) for row in array)


def _classify_motion(
    *,
    translation_m: float,
    rotation_deg: float,
    relative_translation_m: float,
    relative_rotation_deg: float,
) -> str:
    if translation_m >= 0.02 and relative_translation_m < max(0.02, 0.50 * translation_m):
        return "transport"
    if relative_rotation_deg >= 15.0 or rotation_deg >= 30.0:
        return "in_hand_rotation"
    if relative_translation_m >= 0.02:
        return "in_hand_translation"
    if translation_m < 0.01 and rotation_deg < 10.0:
        return "grasp_and_hold"
    return "mixed_or_ambiguous"


def extract_task_semantics(
    *,
    clip: str,
    reference: Mapping[str, np.ndarray],
    contact_records: Sequence[Mapping[str, object]],
    reference_time_scale: int = 8,
    control_hz: float = 20.0,
) -> TaskSemanticContractV1:
    """Extract one contract with no clip-identity conditionals."""

    required = {
        "object_pose_translation_world_ref",
        "object_pose_quaternion_world_ref_wxyz",
        "object_twist_world_ref",
        "T_wrist_object_ref",
    }
    missing = required - set(reference)
    if missing:
        raise ValueError(f"semantic reference misses fields: {sorted(missing)}")
    object_position = np.asarray(reference["object_pose_translation_world_ref"], dtype=np.float64)
    object_quaternion = np.asarray(
        reference["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64
    )
    object_twist = np.asarray(reference["object_twist_world_ref"], dtype=np.float64)
    wrist_object = np.asarray(reference["T_wrist_object_ref"], dtype=np.float64)
    frames = object_position.shape[0]
    if (
        object_position.shape != (frames, 3)
        or object_quaternion.shape != (frames, 4)
        or object_twist.shape != (frames, 6)
        or wrist_object.shape != (frames, 4, 4)
        or frames < 2
    ):
        raise ValueError("semantic reference fields have incompatible shapes")
    if not all(
        np.isfinite(value).all()
        for value in (object_position, object_quaternion, object_twist, wrist_object)
    ):
        raise ValueError("semantic reference contains non-finite values")
    if reference_time_scale < 1 or control_hz <= 0.0:
        raise ValueError("retiming scale and control frequency must be positive")
    retimed_frames = (frames - 1) * reference_time_scale + 1
    contacts = collapse_contact_records(contact_records)
    steps = sorted(step for step in contacts if 0 <= step < retimed_frames)
    runs = consecutive_runs(steps)
    if not steps:
        onset = persistent_start = persistent_end = contact_end = 0
        longest = 0
    else:
        onset, contact_end = steps[0], steps[-1]
        persistent_start, persistent_end = max(runs, key=lambda row: row[1] - row[0] + 1)
        longest = persistent_end - persistent_start + 1
    margin = max(2, int(round(0.025 * retimed_frames)))
    final_hold_start = max(0, retimed_frames - max(3, retimed_frames // 16))
    final_contact = bool(steps and steps[-1] >= final_hold_start)
    translation = float(np.linalg.norm(object_position[-1] - object_position[0]))
    rotation = quaternion_angle_deg(object_quaternion[0], object_quaternion[-1])
    relative_translation = float(np.linalg.norm(wrist_object[-1, :3, 3] - wrist_object[0, :3, 3]))
    relative_rotation = math.degrees(
        math.acos(
            float(
                np.clip(
                    (np.trace(wrist_object[0, :3, :3].T @ wrist_object[-1, :3, :3]) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
    )
    motion_class = _classify_motion(
        translation_m=translation,
        rotation_deg=rotation,
        relative_translation_m=relative_translation,
        relative_rotation_deg=relative_rotation,
    )
    contact_density = len(steps) / retimed_frames
    persistence_ratio = longest / max(1, len(steps))
    confidence = min(1.0, 0.35 + 2.5 * contact_density + 0.25 * persistence_ratio)
    ambiguous = confidence < 0.60 or len(steps) < 3
    limitations: list[str] = []
    if ambiguous:
        limitations.extend(
            (
                "TASK_SEMANTIC_CLASSIFICATION_AMBIGUOUS",
                "validated_c3_trace_has_sparse_contact_support",
                "generic_contact_preserving_motion_fallback",
            )
        )
    return TaskSemanticContractV1(
        clip=clip,
        task_class="generic_contact_preserving_motion" if ambiguous else motion_class,
        classification_confidence=confidence,
        classification_status=(
            "TASK_SEMANTIC_CLASSIFICATION_AMBIGUOUS" if ambiguous else "CLASSIFIED"
        ),
        source_motion_class=motion_class,
        source_frame_count=frames,
        retimed_frame_count=retimed_frames,
        initial_object_pose_wxyz=tuple(
            float(value) for value in np.concatenate((object_position[0], object_quaternion[0]))
        ),
        initial_wrist_object_transform=_matrix_tuple(wrist_object[0]),
        final_wrist_object_transform=_matrix_tuple(wrist_object[-1]),
        contact_onset_window=ContactWindowV1(
            max(0, onset - margin), min(retimed_frames - 1, onset + margin)
        ),
        persistent_contact_window=ContactWindowV1(persistent_start, persistent_end),
        contact_end_window=ContactWindowV1(
            max(0, contact_end - margin), min(retimed_frames - 1, contact_end + margin)
        ),
        final_hold_window=ContactWindowV1(
            final_hold_start if final_contact else contact_end,
            retimed_frames - 1 if final_contact else contact_end,
        ),
        observed_contact_bodies=tuple(
            sorted({body for bodies in contacts.values() for body in bodies})
        ),
        observed_contact_groups=tuple(
            sorted(
                {
                    group
                    for body in {body for bodies in contacts.values() for body in bodies}
                    for group in (body.split("_")[1] if "_" in body else "",)
                    if group in {"thumb", "index", "middle", "ring", "pinky", "palm"}
                }
            )
        ),
        source_contact_control_steps=len(steps),
        source_contact_duration_s=len(steps) / control_hz,
        source_object_translation_m=translation,
        source_object_rotation_deg=rotation,
        source_object_relative_palm_translation_m=relative_translation,
        source_object_relative_palm_rotation_deg=relative_rotation,
        source_final_linear_speed_mps=float(np.linalg.norm(object_twist[-1, :3])),
        source_final_angular_speed_radps=float(np.linalg.norm(object_twist[-1, 3:])),
        limitations=tuple(limitations),
    )


__all__ = ["extract_task_semantics", "quaternion_angle_deg"]
