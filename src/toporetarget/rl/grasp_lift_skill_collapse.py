"""Frozen, outcome-independent metrics for Stage16 grasp/lift localization."""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def first_persistent_true(values: np.ndarray, consecutive: int = 3) -> int | None:
    """Return the start of the first pre-registered all-true run."""

    if consecutive < 1:
        raise ValueError("GRASP_LIFT_CONSECUTIVE_FRAMES_INVALID")
    values = np.asarray(values, dtype=bool)
    if values.ndim != 1:
        raise ValueError("GRASP_LIFT_BOOLEAN_SERIES_MUST_BE_1D")
    if values.size < consecutive:
        return None
    window = np.convolve(values.astype(np.int8), np.ones(consecutive, dtype=np.int8), mode="valid")
    found = np.flatnonzero(window == consecutive)
    return None if not found.size else int(found[0])


def persistent_mask(values: np.ndarray, consecutive: int) -> np.ndarray:
    """Mark every element belonging to a qualifying true run."""

    boolean = np.asarray(values, dtype=bool)
    result = np.zeros(boolean.shape, dtype=bool)
    start = 0
    while start < boolean.size:
        if not bool(boolean[start]):
            start += 1
            continue
        stop = start + 1
        while stop < boolean.size and bool(boolean[stop]):
            stop += 1
        if stop - start >= consecutive:
            result[start:stop] = True
        start = stop
    return result


def grasp_lift_episode_metrics(
    trace: Mapping[str, np.ndarray], *, consecutive_frames: int = 3, lift_threshold_m: float = 0.05
) -> dict[str, Any]:
    """Measure contact quality and semantic-LIFT readiness from one frozen trace.

    ``persistent_grasp`` is intentionally diagnostic-only: two fingertips must
    concurrently remain in actual contact for the frozen three control-frame
    convention.  It is never a training reward or success gate.
    """

    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    hand = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1) & valid
    tip = np.asarray(trace["actual_contact_mask"], dtype=bool) & valid[:, None]
    force = np.linalg.norm(
        np.asarray(trace["fingertip_object_pair_force_world"], dtype=np.float64), axis=-1
    )
    expected = np.asarray(trace["reference_contact_mask"], dtype=bool) & valid[:, None]
    reward = np.asarray(trace["contact_reward"], dtype=np.float64)
    if reward.size == valid.size - 1:
        reward = np.concatenate((np.zeros(1), reward))
    if reward.size != valid.size:
        raise ValueError("GRASP_LIFT_CONTACT_REWARD_LENGTH_INVALID")
    persistent_fingers = np.stack(
        [persistent_mask(tip[:, finger], consecutive_frames) for finger in range(tip.shape[1])],
        axis=-1,
    )
    multi = persistent_fingers.astype(np.int8).sum(axis=-1) >= 2
    persistent_grasp = multi & valid
    phase = np.asarray(trace["phase"])
    ref_contact = np.flatnonzero(expected.any(axis=-1))
    ref_lift = np.flatnonzero(phase == "LIFT")
    grasp = np.flatnonzero(phase == "GRASP")
    first_hand = np.flatnonzero(hand)
    object_z = np.asarray(trace["object_pose"], dtype=np.float64)[:, 2]
    lift = object_z - object_z[0]
    lift_onset = first_persistent_true(lift > 0.005, consecutive_frames)
    active_force = force[tip]
    lift_success = bool(lift[-1] >= lift_threshold_m and persistent_grasp.any())
    if not hand.any():
        category = "NO_CONTACT"
    elif not persistent_grasp.any():
        category = "GRAZING_CONTACT"
    elif lift_success:
        category = "GRASP_AND_LIFT"
    else:
        category = "GRASP_NO_LIFT"
    at_lift = None if not ref_lift.size else bool(persistent_grasp[int(ref_lift[0])])
    return {
        "any_contact": bool(hand.any()),
        "persistent_contact": bool(first_persistent_true(hand, consecutive_frames) is not None),
        "persistent_grasp": bool(persistent_grasp.any()),
        "grasp_and_lift": lift_success,
        "category": category,
        "first_contact": None if not first_hand.size else int(first_hand[0]),
        "first_persistent_contact": first_persistent_true(hand, consecutive_frames),
        "first_persistent_grasp": first_persistent_true(persistent_grasp, consecutive_frames),
        "reference_contact_onset": None if not ref_contact.size else int(ref_contact[0]),
        "reference_grasp_onset": None if not grasp.size else int(grasp[0]),
        "reference_lift_onset": None if not ref_lift.size else int(ref_lift[0]),
        "object_lift_onset": lift_onset,
        "persistent_grasp_at_semantic_lift": at_lift,
        "lift_without_grasp": bool(ref_lift.size and not bool(persistent_grasp[int(ref_lift[0])])),
        "lift_dz_m": float(lift[-1]),
        "max_force_n": float(force[valid].max(initial=0.0)),
        "mean_active_force_n": None if not active_force.size else float(active_force.mean()),
        "p95_active_force_n": None
        if not active_force.size
        else float(np.quantile(active_force, 0.95)),
        "contact_reward_positive_fraction": float((reward[valid] > 0).mean()),
        "contact_reward_mean": float(reward[valid].mean()),
        "contact_reward_max": float(reward[valid].max(initial=0.0)),
        "contact_fraction": float(hand.sum() / valid.sum()),
        "persistent_multi_finger_fraction": float(persistent_grasp.sum() / valid.sum()),
        "max_simultaneous_fingertips": int(tip.astype(np.int8).sum(axis=-1).max(initial=0)),
        "per_finger_contact_fraction": tip.astype(np.int8).sum(axis=0) / valid.sum(),
        "per_finger_persistent_fraction": persistent_fingers.astype(np.int8).sum(axis=0)
        / valid.sum(),
        "per_finger_mean_active_force_n": [
            None if not tip[:, finger].any() else float(force[tip[:, finger], finger].mean())
            for finger in range(tip.shape[1])
        ],
        "per_finger_p95_active_force_n": [
            None
            if not tip[:, finger].any()
            else float(np.quantile(force[tip[:, finger], finger], 0.95))
            for finger in range(tip.shape[1])
        ],
    }


def lift_milestones(
    rows: list[Mapping[str, Any]], baseline_rate: float
) -> dict[str, dict[str, Any] | None]:
    """Locate frozen lift milestones without treating any-contact as grasp."""

    ordered = sorted(rows, key=lambda row: int(row["update"]))
    stable = [row for row in ordered if float(row["lift_episode_rate"]) >= baseline_rate]
    first_bad = next(
        (row for row in ordered if float(row["lift_episode_rate"]) < baseline_rate), None
    )
    major = next((row for row in ordered if float(row["lift_episode_rate"]) <= 0.5), None)
    zero = next((row for row in ordered if float(row["lift_episode_rate"]) == 0.0), None)
    persistent: Mapping[str, Any] | None = None
    for index in range(2, len(ordered)):
        run = ordered[index - 2 : index + 1]
        if all(float(row["lift_episode_rate"]) == 0.0 for row in run):
            persistent = {**ordered[index], "run_start_update": run[0]["update"]}
            break
    keys = ("update", "samples", "checkpoint_sha256")

    def compact(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        return None if row is None else {key: row.get(key) for key in keys}

    return {
        "U_LAST_LIFT_STABLE": compact(stable[-1] if stable else None),
        "U_FIRST_LIFT_DEGRADATION": compact(first_bad),
        "U_MAJOR_LIFT_DEGRADATION": compact(major),
        "U_ZERO_LIFT": compact(zero),
        "U_PERSISTENT_ZERO_LIFT": compact(persistent),
    }
