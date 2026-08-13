"""Pure contracts and metrics for P3/P4 contact-ready physical evaluation.

The IsaacLab driver deliberately imports this small module rather than
reimplementing its seed/reset, contact, flight, and twist semantics.  Keeping
the deterministic contracts here lets tests reject a formal/dev leak before a
simulator process is started.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PHYSICAL_EVALUATION_PAIR_SCHEMA = "Stage16P3P4ContactReadyEvaluationPairsV1"
PHYSICAL_EVALUATION_SCHEMA = "Stage16P3P4ContactReadyPhysicalEvaluationV1"
P4_QUALIFICATION_SCHEMA = "ContactReadyFullGravityQualificationV1"
CLIPS = ("hocap_170105", "hocap_170650")
KINDS = ("development", "formal")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PERSISTENCE_STEPS = 3


@dataclass(frozen=True)
class ContactReadyEpisodePairV1:
    """One pre-registered deterministic evaluation reset."""

    seed: int
    reset_index: int

    def as_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "reset_index": self.reset_index}


@dataclass(frozen=True)
class ContactReadyPairSetV1:
    identifier: str
    source_seed_manifest: str
    source_seed_set: str
    pairs: tuple[ContactReadyEpisodePairV1, ...]

    def __post_init__(self) -> None:
        if not self.identifier or not self.source_seed_manifest or not self.source_seed_set:
            raise ValueError("PHYSICAL_EVALUATION_PAIR_IDENTITY_INVALID")
        if len(self.pairs) != 20:
            raise ValueError("PHYSICAL_EVALUATION_REQUIRES_EXACTLY_20_PAIRS")
        if len({pair.seed for pair in self.pairs}) != len(self.pairs):
            raise ValueError("PHYSICAL_EVALUATION_SEED_DUPLICATE")
        if any(pair.seed < 0 or pair.reset_index < 0 for pair in self.pairs):
            raise ValueError("PHYSICAL_EVALUATION_PAIR_NEGATIVE_VALUE")

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "source_seed_manifest": self.source_seed_manifest,
            "source_seed_set": self.source_seed_set,
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_MUST_BE_MAPPING")
    return value


def load_contact_ready_evaluation_pairs(
    path: Path,
) -> dict[str, dict[str, ContactReadyPairSetV1]]:
    """Load exact development/formal `(seed, reset_index)` contracts."""

    document = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="PAIR_CONTRACT")
    if document.get("schema_version") != PHYSICAL_EVALUATION_PAIR_SCHEMA:
        raise ValueError("PHYSICAL_EVALUATION_PAIR_SCHEMA_INVALID")
    clips = _mapping(document.get("clips"), name="PAIR_CONTRACT_CLIPS")
    if set(clips) != set(CLIPS):
        raise ValueError("PHYSICAL_EVALUATION_PAIR_CLIP_SET_INVALID")
    result: dict[str, dict[str, ContactReadyPairSetV1]] = {}
    for clip in CLIPS:
        kinds = _mapping(clips[clip], name=f"PAIR_CONTRACT_{clip}")
        if set(kinds) != set(KINDS):
            raise ValueError("PHYSICAL_EVALUATION_PAIR_KIND_SET_INVALID")
        result[clip] = {}
        for kind in KINDS:
            item = _mapping(kinds[kind], name=f"PAIR_CONTRACT_{clip}_{kind}")
            pairs_raw = item.get("pairs")
            if not isinstance(pairs_raw, list):
                raise ValueError("PHYSICAL_EVALUATION_PAIRS_INVALID")
            pairs = tuple(
                ContactReadyEpisodePairV1(
                    seed=int(_mapping(row, name="PHYSICAL_EVALUATION_PAIR")["seed"]),
                    reset_index=int(_mapping(row, name="PHYSICAL_EVALUATION_PAIR")["reset_index"]),
                )
                for row in pairs_raw
            )
            result[clip][kind] = ContactReadyPairSetV1(
                identifier=str(item.get("identifier", "")),
                source_seed_manifest=str(item.get("source_seed_manifest", "")),
                source_seed_set=str(item.get("source_seed_set", "")),
                pairs=pairs,
            )
    return result


def validate_pair_set_against_safe_indices(
    pair_set: ContactReadyPairSetV1, *, safe_indices: Sequence[int]
) -> None:
    """Fail closed unless every frozen reset belongs to the P1 safe bank."""

    allowed = {int(value) for value in safe_indices}
    observed = {pair.reset_index for pair in pair_set.pairs}
    if not allowed or not observed.issubset(allowed):
        raise ValueError("PHYSICAL_EVALUATION_RESET_OUTSIDE_CONTACT_READY_SAFE_BANK")


def persistent_mask(mask: np.ndarray, *, minimum_steps: int = PERSISTENCE_STEPS) -> np.ndarray:
    """Mark only runs that meet the fixed P3/P4 contact persistence rule."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or minimum_steps <= 0:
        raise ValueError("PHYSICAL_EVALUATION_PERSISTENCE_INPUT_INVALID")
    result = np.zeros_like(values)
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        if end - start >= minimum_steps:
            result[start:end] = True
        start = end
    return result


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("PHYSICAL_EVALUATION_RUN_MASK_INVALID")
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        result.append((start, end))
        start = end
    return result


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    denominator_values = np.asarray(denominator, dtype=bool)
    count = int(denominator_values.sum())
    if count == 0:
        return None
    return float((np.asarray(numerator, dtype=bool) & denominator_values).sum() / count)


def contact_metrics(
    *, expected: np.ndarray, actual: np.ndarray, valid: np.ndarray
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compute mode-neutral five-tip interaction metrics from a physical trace."""

    expected_values = np.asarray(expected, dtype=bool)
    actual_values = np.asarray(actual, dtype=bool)
    valid_values = np.asarray(valid, dtype=bool)
    if (
        expected_values.ndim != 2
        or expected_values.shape[1] != len(FINGERS)
        or actual_values.shape != expected_values.shape
        or valid_values.shape != expected_values.shape[:1]
    ):
        raise ValueError("PHYSICAL_EVALUATION_CONTACT_SHAPE_INVALID")
    expected_valid = expected_values & valid_values[:, None]
    actual_valid = actual_values & valid_values[:, None]
    per_finger: list[dict[str, object]] = []
    recall_values: list[float] = []
    persistent_recall_values: list[float] = []
    cross_values: list[float] = []
    persistent_cross_values: list[float] = []
    for finger, name in enumerate(FINGERS):
        required = expected_valid[:, finger]
        persistent = persistent_mask(expected_values[:, finger]) & valid_values
        present = actual_valid[:, finger]
        other = actual_valid.copy()
        other[:, finger] = False
        missing = required & ~present
        persistent_missing = persistent & ~present
        cross = missing & other.any(axis=-1)
        persistent_cross = persistent_missing & other.any(axis=-1)
        recall = _rate(present, required)
        persistent_recall = _rate(present, persistent)
        cross_rate = _rate(cross, missing)
        persistent_cross_rate = _rate(persistent_cross, persistent_missing)
        if recall is not None:
            recall_values.append(recall)
        if persistent_recall is not None:
            persistent_recall_values.append(persistent_recall)
        if cross_rate is not None:
            cross_values.append(cross_rate)
        if persistent_cross_rate is not None:
            persistent_cross_values.append(persistent_cross_rate)
        per_finger.append(
            {
                "finger": name,
                "source_tip_recall": recall,
                "persistent_source_tip_recall": persistent_recall,
                "cross_finger_compensation": cross_rate,
                "persistent_cross_finger_compensation": persistent_cross_rate,
                "required_frame_count": int(required.sum()),
                "persistent_required_frame_count": int(persistent.sum()),
                "missing_required_frame_count": int(missing.sum()),
                "fully_missing_required_frame_count": int(
                    (required & ~actual_valid.any(axis=-1)).sum()
                ),
            }
        )
    any_required = expected_valid.any(axis=-1)
    any_matched = (expected_valid & actual_valid).any(axis=-1)
    fully_covered = ~expected_valid | actual_valid
    return (
        {
            "source_tip_recall": None if not recall_values else float(np.mean(recall_values)),
            "source_persistent_tip_recall": (
                None if not persistent_recall_values else float(np.mean(persistent_recall_values))
            ),
            "cross_finger_compensation": None if not cross_values else float(np.mean(cross_values)),
            "persistent_cross_finger_compensation": (
                None if not persistent_cross_values else float(np.mean(persistent_cross_values))
            ),
            "fully_missing_source_contact": _rate(any_required & ~any_matched, any_required),
            "source_contact_full_coverage": _rate(fully_covered.all(axis=-1), any_required),
        },
        per_finger,
    )


def flight_metrics(
    *,
    tip_contact: np.ndarray,
    hand_contact: np.ndarray,
    valid: np.ndarray,
    object_pose: np.ndarray,
    object_twist: np.ndarray,
) -> dict[str, object]:
    """Report distinct fingertip and complete-hand flight windows."""

    tips = np.asarray(tip_contact, dtype=bool)
    hand = np.asarray(hand_contact, dtype=bool)
    valid_values = np.asarray(valid, dtype=bool)
    pose = np.asarray(object_pose, dtype=np.float64)
    twist = np.asarray(object_twist, dtype=np.float64)
    if (
        tips.shape != valid_values.shape
        or hand.shape != valid_values.shape
        or pose.shape != (len(valid_values), 7)
        or twist.shape != (len(valid_values), 6)
    ):
        raise ValueError("PHYSICAL_EVALUATION_FLIGHT_SHAPE_INVALID")
    no_tip = valid_values & ~tips
    no_hand = valid_values & ~hand
    runs = _runs(no_hand)
    events: list[dict[str, object]] = []
    for start, end in runs:
        recontact = next((index for index in range(end, len(hand)) if hand[index]), None)
        events.append(
            {
                "event_type": "NO_HAND_OBJECT_CONTACT_FLIGHT",
                "start_frame": start,
                "end_frame_exclusive": end,
                "duration_control_steps": end - start,
                "onset_vz_mps": float(twist[start, 2]),
                "z_displacement_m": float(pose[end - 1, 2] - pose[start, 2]),
                "recontact_frame": recontact,
            }
        )
    gaps = [end - start for start, end in runs]
    recontacts = sum(event["recontact_frame"] is not None for event in events)
    return {
        "no_tip_contact_fraction": _rate(no_tip, valid_values),
        "no_hand_object_contact_fraction": _rate(no_hand, valid_values),
        "flight_event_count": len(events),
        "longest_flight_gap": max(gaps, default=0),
        "mean_flight_gap": None if not gaps else float(np.mean(gaps)),
        "recontact_count": int(recontacts),
        "events": events,
    }


def twist_metrics(
    *, actual: np.ndarray, reference: np.ndarray, valid: np.ndarray, terminal_steps: int = 20
) -> dict[str, object]:
    """Compute reference-relative twist residuals only over captured physical rows."""

    actual_values = np.asarray(actual, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    valid_values = np.asarray(valid, dtype=bool)
    if actual_values.shape != reference_values.shape or actual_values.shape != (
        len(valid_values),
        6,
    ):
        raise ValueError("PHYSICAL_EVALUATION_TWIST_SHAPE_INVALID")
    if not valid_values.any():
        raise ValueError("PHYSICAL_EVALUATION_TWIST_NO_VALID_ROWS")
    delta = actual_values - reference_values
    linear = np.linalg.norm(delta[:, :3], axis=-1)[valid_values]
    angular = np.linalg.norm(delta[:, 3:], axis=-1)[valid_values]
    terminal_index = np.flatnonzero(valid_values)[-min(terminal_steps, int(valid_values.sum())) :]
    terminal_linear = np.linalg.norm(delta[terminal_index, :3], axis=-1)
    terminal_angular = np.linalg.norm(delta[terminal_index, 3:], axis=-1)
    actual_terminal_linear = np.linalg.norm(actual_values[terminal_index, :3], axis=-1)
    actual_terminal_angular = np.linalg.norm(actual_values[terminal_index, 3:], axis=-1)
    return {
        "Delta_v_mps": {
            "mean": float(linear.mean()),
            "p95": float(np.percentile(linear, 95)),
            "terminal": float(np.median(terminal_linear)),
        },
        "Delta_omega_radps": {
            "mean": float(angular.mean()),
            "p95": float(np.percentile(angular, 95)),
            "terminal": float(np.median(terminal_angular)),
        },
        "terminal_abs_v_mps": float(np.median(actual_terminal_linear)),
        "terminal_abs_omega_radps": float(np.median(actual_terminal_angular)),
    }


def physical_failure_status(
    *,
    termination_reason: int,
    finite: bool,
    absolute_geometry_pass: bool,
    inter_finger_pass: bool,
    max_penetration_m: float,
    catastrophic_penetration_m: float,
) -> dict[str, object]:
    """Keep the frozen P95 gate distinct from a true catastrophic contact event."""

    return {
        "finite": bool(finite),
        "object_drop": termination_reason in {4, 5},
        "joint_limit": termination_reason == 2,
        "catastrophic_contact": max_penetration_m >= catastrophic_penetration_m,
        "absolute_geometry_violation": not absolute_geometry_pass,
        "interfinger_limit_violation": not inter_finger_pass,
        "termination_reason_code": int(termination_reason),
    }


__all__ = [
    "CLIPS",
    "FINGERS",
    "KINDS",
    "P4_QUALIFICATION_SCHEMA",
    "PHYSICAL_EVALUATION_PAIR_SCHEMA",
    "PHYSICAL_EVALUATION_SCHEMA",
    "ContactReadyEpisodePairV1",
    "ContactReadyPairSetV1",
    "contact_metrics",
    "flight_metrics",
    "load_contact_ready_evaluation_pairs",
    "persistent_mask",
    "physical_failure_status",
    "twist_metrics",
    "validate_pair_set_against_safe_indices",
]
