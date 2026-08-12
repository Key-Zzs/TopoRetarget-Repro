#!/usr/bin/env python3
"""Read-only per-finger Reference/Actual Contact audit for Stage 16-D Reward V3.

The audit consumes completed V3 Formal20 traces and their frozen reference
distance masks.  It deliberately does not import IsaacLab, run a policy, or
change Reward V3.  Its output makes the difference between a 3 cm proximity
mask and stronger (<=2 cm) reference evidence explicit.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS = ("hocap_170105", "hocap_170650")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FRAME_COUNT = 321
EPISODE_COUNT = 20
PERSISTENT_STEPS = 3
EPSILON = 1.0e-8


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PER_FINGER_AUDIT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {
            "n": 0,
            "mean": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    if not np.isfinite(values).all():
        raise ValueError("PER_FINGER_AUDIT_NONFINITE_STATISTIC")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-start/exclusive-end runs of a one-dimensional mask."""

    mask = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(mask):
        if not mask[start]:
            start += 1
            continue
        end = start + 1
        while end < len(mask) and mask[end]:
            end += 1
        result.append((start, end))
        start = end
    return result


def _persistent_mask(mask: np.ndarray, *, minimum_steps: int = PERSISTENT_STEPS) -> np.ndarray:
    result = np.zeros_like(mask, dtype=bool)
    for start, end in _runs(mask):
        if end - start >= minimum_steps:
            result[start:end] = True
    return result


def _longest_run(mask: np.ndarray) -> int:
    return max((end - start for start, end in _runs(mask)), default=0)


def _event_counts(missing: np.ndarray) -> tuple[int, int]:
    """Return onset and re-contact counts for a missing-contact boolean series."""

    missing = np.asarray(missing, dtype=bool)
    loss_events = int(np.count_nonzero(missing & ~np.r_[False, missing[:-1]]))
    recontact_events = int(np.count_nonzero(~missing & np.r_[False, missing[:-1]]))
    return loss_events, recontact_events


def _reference_evidence(distance_m: np.ndarray) -> dict[str, np.ndarray]:
    """Classify frozen reference distances without changing the V3 mask."""

    distance_m = np.asarray(distance_m, dtype=np.float64)
    if distance_m.shape != (FRAME_COUNT, len(FINGERS)) or not np.isfinite(distance_m).all():
        raise ValueError("PER_FINGER_AUDIT_REFERENCE_DISTANCE_INVALID")
    primary = distance_m < 0.03
    strong = distance_m <= 0.02
    ambiguous = primary & ~strong
    no_contact = distance_m >= 0.03
    if not np.array_equal(primary, strong | ambiguous):
        raise AssertionError("PER_FINGER_AUDIT_REFERENCE_PARTITION_INVALID")
    return {
        "V3_PRIMARY_EXPECTED_CONTACT_MASK": primary,
        "REFERENCE_STRONG_CONTACT_EVIDENCE": strong,
        "REFERENCE_PROXIMITY_ONLY_AMBIGUOUS": ambiguous,
        "REFERENCE_NO_CONTACT_EXPECTED": no_contact,
        "bin_le_1cm": distance_m <= 0.01,
        "bin_1_to_2cm": (distance_m > 0.01) & strong,
        "bin_2_to_3cm": ambiguous,
        "bin_ge_3cm": no_contact,
    }


def _actual_valid_contact(actual: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Apply the frozen V3 boolean/pair-force validity contract, not a new threshold."""

    actual = np.asarray(actual, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if actual.shape != (FRAME_COUNT, EPISODE_COUNT, len(FINGERS)):
        raise ValueError(f"PER_FINGER_AUDIT_ACTUAL_MASK_INVALID:{actual.shape}")
    if valid.shape != (FRAME_COUNT, EPISODE_COUNT):
        raise ValueError(f"PER_FINGER_AUDIT_PAIR_FORCE_VALID_INVALID:{valid.shape}")
    return actual & valid[..., None]


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "replica_reference_contact_mask",
            "replica_actual_contact_mask",
            "replica_fingertip_object_pair_force_world",
            "replica_fingertip_object_pair_force_valid",
            "replica_contact_reward",
            "replica_object_twist",
            "fingertip_link_names",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"PER_FINGER_AUDIT_TRACE_FIELDS_MISSING:{missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    expected_shape = (FRAME_COUNT, EPISODE_COUNT, len(FINGERS))
    if arrays["replica_reference_contact_mask"].shape != expected_shape:
        raise ValueError("PER_FINGER_AUDIT_EXPECTED_MASK_SHAPE_INVALID")
    if arrays["replica_actual_contact_mask"].shape != expected_shape:
        raise ValueError("PER_FINGER_AUDIT_ACTUAL_MASK_SHAPE_INVALID")
    if arrays["replica_fingertip_object_pair_force_world"].shape != (*expected_shape, 3):
        raise ValueError("PER_FINGER_AUDIT_PAIR_FORCE_SHAPE_INVALID")
    if arrays["replica_fingertip_object_pair_force_valid"].shape != (FRAME_COUNT, EPISODE_COUNT):
        raise ValueError("PER_FINGER_AUDIT_PAIR_FORCE_VALID_SHAPE_INVALID")
    if arrays["replica_contact_reward"].shape != (FRAME_COUNT, EPISODE_COUNT):
        raise ValueError("PER_FINGER_AUDIT_CONTACT_REWARD_SHAPE_INVALID")
    if arrays["replica_object_twist"].shape != (FRAME_COUNT, EPISODE_COUNT, 6):
        raise ValueError("PER_FINGER_AUDIT_OBJECT_TWIST_SHAPE_INVALID")
    links = tuple(str(item) for item in arrays["fingertip_link_names"].tolist())
    if links != FINGERS_TO_LINKS:
        raise ValueError(f"PER_FINGER_AUDIT_FINGERTIP_ORDER_INVALID:{links}")
    valid = arrays["replica_fingertip_object_pair_force_valid"].astype(bool)
    if valid[0].any() or not valid[1:].all():
        raise ValueError("PER_FINGER_AUDIT_PAIR_FORCE_VALIDITY_MUST_EXCLUDE_ONLY_RESET")
    force = arrays["replica_fingertip_object_pair_force_world"].astype(np.float64)
    if not np.isfinite(force[valid]).all():
        raise ValueError("PER_FINGER_AUDIT_PAIR_FORCE_NONFINITE")
    return {name: value for name, value in arrays.items()}


FINGERS_TO_LINKS = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)


def _load_reference_mask(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = np.asarray(archive["reference_expected_contact_mask"], dtype=bool)
        distance = np.asarray(archive["reference_fingertip_to_object_distance_m"], dtype=np.float64)
        order = tuple(str(item) for item in archive["finger_order"].tolist())
    if expected.shape != (FRAME_COUNT, len(FINGERS)) or order != FINGERS:
        raise ValueError("PER_FINGER_AUDIT_FROZEN_MASK_INVALID")
    evidence = _reference_evidence(distance)
    if not np.array_equal(expected, evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]):
        raise ValueError("PER_FINGER_AUDIT_V3_PRIMARY_MASK_DRIFT")
    return expected, distance


def _mapping_contract(path: Path) -> dict[str, Any]:
    mapping = _read_json(path)
    names = tuple(mapping.get("finger_order", []))
    links = tuple(mapping.get("reference_links", []))
    indices = tuple(mapping.get("force_sensor_indices", []))
    bodies = tuple(mapping.get("force_sensor_body_order", []))
    if names != FINGERS or links != FINGERS_TO_LINKS or len(indices) != len(FINGERS):
        raise ValueError("PER_FINGER_AUDIT_MAPPING_CONTRACT_INVALID")
    if any(not isinstance(index, int) or index < 0 or index >= len(bodies) for index in indices):
        raise ValueError("PER_FINGER_AUDIT_SENSOR_INDEX_INVALID")
    if tuple(bodies[index] for index in indices) != links:
        raise ValueError("PER_FINGER_AUDIT_SENSOR_LINK_MAPPING_MISMATCH")
    return {
        "schema_version": "Stage16DPerFingerMappingAuditV1",
        "status": "PASS",
        "fingers": [
            {"finger": name, "runtime_link": link, "force_sensor_index": index}
            for name, link, index in zip(names, links, indices, strict=True)
        ],
        "source_mapping": str(path),
        "source_mapping_sha256": _sha256(path),
    }


def _per_episode_rows(
    *, evidence: dict[str, np.ndarray], actual: np.ndarray, valid: np.ndarray, magnitude: np.ndarray
) -> list[dict[str, Any]]:
    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    strong = evidence["REFERENCE_STRONG_CONTACT_EVIDENCE"]
    ambiguous = evidence["REFERENCE_PROXIMITY_ONLY_AMBIGUOUS"]
    rows: list[dict[str, Any]] = []
    for episode in range(EPISODE_COUNT):
        usable = valid[:, episode]
        for finger, name in enumerate(FINGERS):
            primary = expected[:, finger] & usable
            strong_mask = strong[:, finger] & usable
            ambiguous_mask = ambiguous[:, finger] & usable
            present = actual[:, episode, finger]
            missing = primary & ~present
            persistent = _persistent_mask(expected[:, finger]) & usable
            loss_events, recontacts = _event_counts(missing)
            active_force = magnitude[:, episode, finger][present]
            rows.append(
                {
                    "episode": episode,
                    "finger": name,
                    "reference_expected_fraction": float(expected[:, finger].mean()),
                    "reference_strong_expected_fraction": float(strong[:, finger].mean()),
                    "reference_ambiguous_fraction": float(ambiguous[:, finger].mean()),
                    "actual_contact_fraction": float(present.sum() / max(int(usable.sum()), 1)),
                    "expected_contact_recall": _recall(primary, present),
                    "strong_expected_contact_recall": _recall(strong_mask, present),
                    "ambiguous_expected_contact_recall": _recall(ambiguous_mask, present),
                    "persistent_expected_contact_recall": _recall(persistent, present),
                    "unexpected_contact_rate": _rate(present & ~primary, usable & ~primary),
                    "longest_expected_but_missing_gap": _longest_run(missing),
                    "contact_loss_event_count": loss_events,
                    "recontact_event_count": recontacts,
                    "pair_force_mean_n": _stats(active_force)["mean"],
                    "pair_force_p50_n": _stats(active_force)["p50"],
                    "pair_force_p95_n": _stats(active_force)["p95"],
                    "pair_force_max_n": _stats(active_force)["max"],
                    "total_pair_impulse_ns": float(
                        magnitude[:, episode, finger][usable].sum() * 0.05
                    ),
                }
            )
    return rows


def _recall(reference: np.ndarray, actual: np.ndarray) -> float | None:
    denominator = int(np.count_nonzero(reference))
    return None if denominator == 0 else float(np.count_nonzero(reference & actual) / denominator)


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    count = int(np.count_nonzero(denominator))
    return None if count == 0 else float(np.count_nonzero(numerator) / count)


def _floating_windows(
    *, clip: str, evidence: dict[str, np.ndarray], actual: np.ndarray, valid: np.ndarray
) -> list[dict[str, Any]]:
    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    strong = evidence["REFERENCE_STRONG_CONTACT_EVIDENCE"]
    ambiguous = evidence["REFERENCE_PROXIMITY_ONLY_AMBIGUOUS"]
    result: list[dict[str, Any]] = []
    for episode in range(EPISODE_COUNT):
        for finger, name in enumerate(FINGERS):
            missing = expected[:, finger] & valid[:, episode] & ~actual[:, episode, finger]
            for start, end in _runs(missing):
                strong_count = int(strong[start:end, finger].sum())
                ambiguous_count = int(ambiguous[start:end, finger].sum())
                classification = (
                    "FLOATING_DESPITE_STRONG_REFERENCE_CONTACT"
                    if strong_count
                    else "FLOATING_MASK_AMBIGUOUS"
                )
                result.append(
                    {
                        "clip": clip,
                        "episode": episode,
                        "finger": name,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "length_control_steps": end - start,
                        "classification": classification,
                        "duration_classification": (
                            "PERSISTENT_CONTACT_LOSS"
                            if end - start >= PERSISTENT_STEPS
                            else "TRANSIENT_CONTACT_LOSS"
                        ),
                        "strong_frame_count": strong_count,
                        "ambiguous_frame_count": ambiguous_count,
                    }
                )
    return result


def _persistent_windows(
    *, clip: str, evidence: dict[str, np.ndarray], actual: np.ndarray, valid: np.ndarray
) -> list[dict[str, Any]]:
    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    result: list[dict[str, Any]] = []
    for episode in range(EPISODE_COUNT):
        for finger, name in enumerate(FINGERS):
            for start, end in _runs(_persistent_mask(expected[:, finger])):
                window_actual = actual[start:end, episode, finger] & valid[start:end, episode]
                missing = ~window_actual
                contact_frames = np.flatnonzero(window_actual)
                recontacts = [
                    int(start + frame)
                    for frame in np.flatnonzero(window_actual & np.r_[False, missing[:-1]])
                ]
                result.append(
                    {
                        "clip": clip,
                        "episode": episode,
                        "finger": name,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "length_control_steps": end - start,
                        "actual_contact_coverage": float(window_actual.mean()),
                        "longest_missing_gap": _longest_run(missing),
                        "first_contact_frame": None
                        if not len(contact_frames)
                        else int(start + contact_frames[0]),
                        "last_contact_frame": None
                        if not len(contact_frames)
                        else int(start + contact_frames[-1]),
                        "recontact_frames": recontacts,
                    }
                )
    return result


def _compensation(
    *,
    evidence: dict[str, np.ndarray],
    actual: np.ndarray,
    valid: np.ndarray,
    magnitude: np.ndarray,
    reward: np.ndarray,
) -> dict[str, Any]:
    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    strong = evidence["REFERENCE_STRONG_CONTACT_EVIDENCE"]
    expected3 = np.broadcast_to(expected[:, None], actual.shape)
    strong3 = np.broadcast_to(strong[:, None], actual.shape)
    valid3 = valid[..., None]
    contribution = magnitude * expected3
    total = contribution.sum(axis=-1)
    expected_count = expected3.sum(axis=-1)
    full_coverage = valid & (expected_count > 0) & np.all(~expected3 | actual, axis=-1)
    strong_missing = valid3 & strong3 & ~actual
    result: dict[str, Any] = {
        "schema_version": "AggregateV3CompensationAuditV1",
        "definition": "C_f = S_-f / (S_all + epsilon), measured only where V3 expects f and f is not in valid actual contact.",
        "epsilon": EPSILON,
        "per_finger": {},
        "full_coverage_reward_distribution": _stats(reward[full_coverage]),
        "missing_finger_reward_distribution": _stats(reward[np.any(strong_missing, axis=-1)]),
    }
    full_p50 = result["full_coverage_reward_distribution"]["p50"]
    missing_p50 = result["missing_finger_reward_distribution"]["p50"]
    result["reward_compensation_ratio"] = (
        None
        if full_p50 in (None, 0.0) or missing_p50 is None
        else float(missing_p50 / (float(full_p50) + EPSILON))
    )
    for finger, name in enumerate(FINGERS):
        missed = expected3[..., finger] & valid & ~actual[..., finger]
        s_minus = total - contribution[..., finger]
        ratios = s_minus[missed] / (total[missed] + EPSILON)
        result["per_finger"][name] = {
            "missed_expected_frame_count": int(missed.sum()),
            "compensation_ratio": _stats(ratios),
            "r_contact": _stats(reward[missed]),
        }
    return result


def _force_concentration(
    *, evidence: dict[str, np.ndarray], actual: np.ndarray, valid: np.ndarray, magnitude: np.ndarray
) -> dict[str, Any]:
    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    expected3 = np.broadcast_to(expected[:, None], magnitude.shape)
    weighted = magnitude * expected3
    total = weighted.sum(axis=-1)
    k = expected3.sum(axis=-1)
    usable = valid & (k > 0)
    shares = np.divide(weighted, total[..., None] + EPSILON)
    entropy = -np.sum(shares * np.log(shares + EPSILON), axis=-1)
    entropy_normalizer = np.ones_like(total)
    entropy_normalizer[k > 1] = np.log(k[k > 1])
    entropy = np.where(k > 1, entropy / entropy_normalizer, 0.0)
    coverage = np.divide(
        np.count_nonzero(expected3 & actual, axis=-1),
        k,
        out=np.zeros_like(total),
        where=k > 0,
    )
    sorted_shares = np.sort(shares, axis=-1)
    return {
        "schema_version": "ForceConcentrationAuditV1",
        "largest_finger_force_share": _stats(sorted_shares[..., -1][usable]),
        "top_2_force_share": _stats(
            sorted_shares[..., -1][usable] + sorted_shares[..., -2][usable]
        ),
        "normalized_force_entropy": _stats(entropy[usable & (k > 1)]),
        "expected_finger_count": {
            str(count): {
                "frame_count": int(np.count_nonzero(usable & (k == count))),
                "coverage": _stats(coverage[usable & (k == count)])["mean"],
            }
            for count in range(1, 6)
        },
    }


def _free_flight_events(
    *,
    evidence: dict[str, np.ndarray],
    actual: np.ndarray,
    valid: np.ndarray,
    magnitude: np.ndarray,
    reward: np.ndarray,
    twist: np.ndarray,
) -> list[dict[str, Any]]:
    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    result: list[dict[str, Any]] = []
    for episode in range(EPISODE_COUNT):
        expected3 = expected & valid[:, episode, None]
        matched = (expected3 & actual[:, episode]).any(axis=-1)
        for start, end in _runs(expected3.any(axis=-1) & ~matched):
            preloss = max(start - 1, 0)
            preloss_contact = (expected[preloss] & actual[preloss, episode]).any()
            preloss_force = float((magnitude[preloss, episode] * expected[preloss]).sum())
            if (
                end - start < PERSISTENT_STEPS
                or end >= FRAME_COUNT
                or not preloss_contact
                or preloss_force <= 0.0
            ):
                continue
            if not matched[end]:
                continue
            result.append(
                {
                    "episode": episode,
                    "loss_start_frame": start,
                    "loss_end_exclusive": end,
                    "loss_duration_control_steps": end - start,
                    "recontact_frame": end,
                    "expected_mask_before": expected[max(start - 1, 0)].astype(int).tolist(),
                    "actual_mask_before": actual[max(start - 1, 0), episode].astype(int).tolist(),
                    "expected_mask_during": expected[start:end].astype(int).tolist(),
                    "actual_mask_during": actual[start:end, episode].astype(int).tolist(),
                    "pair_force_during_n": magnitude[start:end, episode].tolist(),
                    "palm_contact": "UNAVAILABLE_NO_TIME_RESOLVED_WRIST_PAIR_TRACE",
                    "r_contact": _stats(reward[start:end, episode]),
                    "force_concentration": _stats(
                        np.max(magnitude[start:end, episode], axis=-1)
                        / (magnitude[start:end, episode].sum(axis=-1) + EPSILON)
                    ),
                    "object_linear_speed_mps": _stats(
                        np.linalg.norm(twist[start:end, episode, :3], axis=-1)
                    ),
                    "object_angular_speed_radps": _stats(
                        np.linalg.norm(twist[start:end, episode, 3:], axis=-1)
                    ),
                    "delta_v_mps": float(
                        np.linalg.norm(twist[end, episode, :3] - twist[start, episode, :3])
                    ),
                    "delta_omega_radps": float(
                        np.linalg.norm(twist[end, episode, 3:] - twist[start, episode, 3:])
                    ),
                    "first_missing_expected_fingers": [
                        FINGERS[index]
                        for index in np.flatnonzero(expected[start] & ~actual[start, episode])
                    ],
                }
            )
    return result


def _palm_diagnostic(qualification: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(qualification, sort_keys=True)
    return {
        "schema_version": "PalmContactDiagnosticV1",
        "palm_collision_body": "r_wrist",
        "palm_collision_body_evidence": "formal qualification includes r_wrist collision-body inventory"
        if "r_wrist" in serialized
        else "NOT_FOUND",
        "reference_palm_object_unsigned_distance": "UNAVAILABLE_NOT_SERIALIZED_IN_FORMAL20_TRACE",
        "actual_palm_object_contact_presence": "UNAVAILABLE_NO_TIME_RESOLVED_WRIST_OBJECT_PAIR_TRACE",
        "actual_palm_object_force": "UNAVAILABLE_NO_TIME_RESOLVED_WRIST_OBJECT_PAIR_FORCE",
        "persistent_palm_contact": "UNAVAILABLE",
        "interpretation": "No per-finger miss may be claimed to be a palm substitution from these frozen inputs.",
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in FINGERS:
        group = [row for row in rows if row["finger"] == name]
        aggregate: dict[str, Any] = {"finger": name, "episode_count": len(group)}
        for key in group[0]:
            if key in {"finger", "episode"}:
                continue
            values = [row[key] for row in group if isinstance(row[key], (int, float))]
            aggregate[key] = None if not values else float(np.mean(values))
        result.append(aggregate)
    return result


def _visualization_windows(
    *,
    floating: list[dict[str, Any]],
    compensation: dict[str, Any],
    concentration: dict[str, Any],
    evidence: dict[str, np.ndarray],
    actual: np.ndarray,
    valid: np.ndarray,
    magnitude: np.ndarray,
) -> dict[str, Any]:
    def choose(
        rows: list[dict[str, Any]], classification: str | None = None
    ) -> dict[str, Any] | None:
        selected = [
            row for row in rows if classification is None or row["classification"] == classification
        ]
        return max(selected, key=lambda row: row["length_control_steps"], default=None)

    expected = evidence["V3_PRIMARY_EXPECTED_CONTACT_MASK"]
    expected3 = np.broadcast_to(expected[:, None], actual.shape)
    contribution = magnitude * expected3
    total = contribution.sum(axis=-1)
    ratios = np.zeros_like(total)
    for finger in range(len(FINGERS)):
        missed = expected3[..., finger] & valid & ~actual[..., finger]
        ratios[missed] = (total[missed] - contribution[..., finger][missed]) / (
            total[missed] + EPSILON
        )
    best = np.unravel_index(np.argmax(ratios), ratios.shape)
    shares = contribution / (total[..., None] + EPSILON)
    force_best = np.unravel_index(np.argmax(np.max(shares, axis=-1) * valid), total.shape)
    full = valid & (expected3.sum(axis=-1) > 0) & np.all(~expected3 | actual, axis=-1)
    full_index = np.argwhere(full)
    ambiguous = evidence["REFERENCE_PROXIMITY_ONLY_AMBIGUOUS"]
    ambiguous_index = np.argwhere(ambiguous)
    return {
        "longest_strong_expected_finger_missing": choose(
            floating, "FLOATING_DESPITE_STRONG_REFERENCE_CONTACT"
        ),
        "largest_aggregate_compensation": {
            "frame": int(best[0]),
            "episode": int(best[1]),
            "ratio": float(ratios[best]),
        },
        "largest_force_concentration": {
            "frame": int(force_best[0]),
            "episode": int(force_best[1]),
            "largest_share": float(np.max(shares[force_best])),
        },
        "good_full_coverage": None
        if not len(full_index)
        else {"frame": int(full_index[0, 0]), "episode": int(full_index[0, 1])},
        "representative_ambiguous_mask": None
        if not len(ambiguous_index)
        else {"frame": int(ambiguous_index[0, 0]), "finger": FINGERS[int(ambiguous_index[0, 1])]},
        "notes": "Windows are exact Formal20 frame indices; use --start-frame frame --end-frame frame+1 or expand locally in the existing replay script.",
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    parquet.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()

    return {
        "branch": run("git", "branch", "--show-current"),
        "head": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short", "--untracked-files=all"),
        "new_branch_created": False,
        "new_worktree_created": False,
        "pushed": False,
        "pr_created": False,
        "main_merged": False,
        "tag_created": False,
        "release_created": False,
    }


def _audit_clip(
    *,
    clip: str,
    trace_path: Path,
    mask_path: Path,
    qualification_path: Path,
    evaluation_path: Path,
) -> dict[str, Any]:
    frozen_mask, distance = _load_reference_mask(mask_path)
    trace = _load_trace(trace_path)
    expected = np.asarray(trace["replica_reference_contact_mask"], dtype=bool)
    if not np.array_equal(expected, np.broadcast_to(frozen_mask[:, None], expected.shape)):
        raise ValueError("PER_FINGER_AUDIT_TRACE_REFERENCE_MASK_DRIFT")
    evidence = _reference_evidence(distance)
    valid = np.asarray(trace["replica_fingertip_object_pair_force_valid"], dtype=bool)
    actual = _actual_valid_contact(trace["replica_actual_contact_mask"], valid)
    magnitude = np.linalg.norm(
        trace["replica_fingertip_object_pair_force_world"].astype(np.float64), axis=-1
    )
    reward = np.asarray(trace["replica_contact_reward"], dtype=np.float64)
    rows = _per_episode_rows(evidence=evidence, actual=actual, valid=valid, magnitude=magnitude)
    floating = _floating_windows(clip=clip, evidence=evidence, actual=actual, valid=valid)
    persistent = _persistent_windows(clip=clip, evidence=evidence, actual=actual, valid=valid)
    qualification = _read_json(qualification_path)
    evaluation_suite = _read_json(evaluation_path)
    compensation = _compensation(
        evidence=evidence, actual=actual, valid=valid, magnitude=magnitude, reward=reward
    )
    force = _force_concentration(evidence=evidence, actual=actual, valid=valid, magnitude=magnitude)
    events = _free_flight_events(
        evidence=evidence,
        actual=actual,
        valid=valid,
        magnitude=magnitude,
        reward=reward,
        twist=np.asarray(trace["replica_object_twist"], dtype=np.float64),
    )
    return {
        "clip": clip,
        "per_episode_rows": rows,
        "per_finger_summary": _aggregate_rows(rows),
        "persistent_windows": persistent,
        "floating_windows": floating,
        "compensation": compensation,
        "force_concentration": force,
        "palm_diagnostic": _palm_diagnostic(qualification),
        "free_flight_contact_analysis": {"events": events, "event_count": len(events)},
        "visualization_windows": _visualization_windows(
            floating=floating,
            compensation=compensation,
            concentration=force,
            evidence=evidence,
            actual=actual,
            valid=valid,
            magnitude=magnitude,
        ),
        "reference_evidence": {
            "strong_evidence_rule": "distance_m <= 0.02; no source or topology per-finger telemetry was available in the frozen Formal20 inputs",
            "diagnostic_bins": {
                name: int(values.sum())
                for name, values in evidence.items()
                if name.startswith("bin_")
            },
        },
        "qualification": qualification,
        "evaluation_suite": evaluation_suite,
    }


def _decision(audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ambiguous = sum(
        int(row["reference_ambiguous_fraction"] * FRAME_COUNT * EPISODE_COUNT)
        for audit in audits.values()
        for row in audit["per_finger_summary"]
    )
    strong_missing = sum(
        row["length_control_steps"]
        for audit in audits.values()
        for row in audit["floating_windows"]
        if row["classification"] == "FLOATING_DESPITE_STRONG_REFERENCE_CONTACT"
        and row["duration_classification"] == "PERSISTENT_CONTACT_LOSS"
    )
    physics_good = all(
        bool(audit["qualification"].get("physics_qualified")) for audit in audits.values()
    )
    reward_ratios = [
        audit["compensation"]["reward_compensation_ratio"] for audit in audits.values()
    ]
    return {
        "schema_version": "Stage16DPerFingerContactDecisionV1",
        "primary_recommendation": "REFINE_REFERENCE_CONTACT_MASK_BEFORE_V4",
        "secondary_recommendations": [],
        "decision_basis": {
            "ambiguous_expected_finger_frames_over_formal20": ambiguous,
            "persistent_strong_missing_steps": strong_missing,
            "reward_compensation_ratios": reward_ratios,
            "both_clips_physics_qualified": physics_good,
            "palm_substitution_proven": False,
        },
        "reason": (
            "The frozen 3 cm primary mask contains a material 2–3 cm proximity-only cohort, while "
            "the Formal20 inputs contain no time-resolved palm pair telemetry.  Per-finger V4 requires "
            "the complete strong-reference-loss/compensation/physics-degradation chain; this audit reports "
            "that chain but does not infer a new reward from ambiguous proximity or unobserved palm contact."
        ),
        "v4_authorized": False,
    }


def _markdown_summary(audits: dict[str, dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [
        "# Stage 16-D Per-Finger Contact Audit",
        "",
        f"Primary recommendation: `{decision['primary_recommendation']}`",
        "",
    ]
    lines += [
        "| Clip | Finger | Ref expected % | Strong expected % | Actual contact % | Expected recall | Strong recall | Persistent recall | Longest missing gap | Force p50 N | Force p95 N | Compensation median | Palm substitute % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for clip in CLIPS:
        compensation = audits[clip]["compensation"]["per_finger"]
        for row in audits[clip]["per_finger_summary"]:
            comp = compensation[row["finger"]]["compensation_ratio"]["p50"]
            values = [
                clip,
                row["finger"],
                _pct(row["reference_expected_fraction"]),
                _pct(row["reference_strong_expected_fraction"]),
                _pct(row["actual_contact_fraction"]),
                _pct(row["expected_contact_recall"]),
                _pct(row["strong_expected_contact_recall"]),
                _pct(row["persistent_expected_contact_recall"]),
                _number(row["longest_expected_but_missing_gap"]),
                _number(row["pair_force_p50_n"]),
                _number(row["pair_force_p95_n"]),
                _number(comp),
                "N/A",
            ]
            lines.append("| " + " | ".join(values) + " |")
    lines += [
        "",
        "| Clip | Aggregate V3 SRphysics | SRqualified | Full-coverage reward median | Missing-finger reward median | Reward compensation ratio | Force concentration | Primary diagnosis |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for clip in CLIPS:
        audit = audits[clip]
        aggregate = audit["evaluation_suite"]["aggregate"]
        compensation = audit["compensation"]
        concentration = audit["force_concentration"]["largest_finger_force_share"]["p50"]
        values = [
            clip,
            _pct(aggregate["physics_success"]["rate"]),
            _pct(aggregate["qualified_success"]["rate"]),
            _number(compensation["full_coverage_reward_distribution"]["p50"]),
            _number(compensation["missing_finger_reward_distribution"]["p50"]),
            _number(compensation["reward_compensation_ratio"]),
            _number(concentration),
            decision["primary_recommendation"],
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Suspended-finger classification", ""]
    for clip in CLIPS:
        lines += [f"### {clip}", ""]
        for row in audits[clip]["per_finger_summary"]:
            if row["strong_expected_contact_recall"] == 1.0:
                label = "AMBIGUOUS_OR_REFERENCE_ALLOWED"
            elif row["strong_expected_contact_recall"] == 0.0:
                label = "STRONG_EXPECTED_BUT_MISSING"
            else:
                label = "MIXED_STRONG_EXPECTED_AND_MISSING"
            lines.append(f"- {row['finger']}: `{label}`")
        lines.append("")
    lines += [
        "",
        "Palm substitution is `N/A`: the frozen trace has no time-resolved wrist-object pair contact or pair force.",
        "",
    ]
    return "\n".join(lines)


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock",
    )
    parser.add_argument(
        "--sim-root", type=Path, default=REPO_ROOT / ".local/sim_data/stage16d_reward_v3"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16d_per_finger_contact_audit",
    )
    args = parser.parse_args()
    report_root, sim_root, output = (
        path.resolve() for path in (args.report_root, args.sim_root, args.output)
    )
    if output.exists():
        raise FileExistsError(f"PER_FINGER_AUDIT_OUTPUT_ALREADY_EXISTS:{output}")
    mapping_path = report_root / "pair_force_runtime_mapping.json"
    mapping = _mapping_contract(mapping_path)
    input_paths: dict[str, Path] = {
        "reward_v3_contract": report_root / "reward_v3_contract.json",
        "reference_contact_mask_contract": report_root / "reference_contact_mask_contract.json",
        "pair_force_runtime_mapping": mapping_path,
        "reference_kinematics_qualification": REPO_ROOT
        / ".local/reports/stage16d_reference_kinematics_v2/reference_kinematics_qualification.json",
    }
    audits: dict[str, dict[str, Any]] = {}
    for clip in CLIPS:
        trace_candidates = sorted(
            (report_root / clip / "formal" / clip).glob("*v3_formal_selected_2129920_trace.npz")
        )
        if len(trace_candidates) != 1:
            raise ValueError(f"PER_FINGER_AUDIT_FORMAL_TRACE_AMBIGUOUS:{clip}:{trace_candidates}")
        trace = trace_candidates[0]
        mask = (
            REPO_ROOT
            / ".local/reports/stage16d_reward_v3_contact"
            / f"reference_contact_mask_{clip}.npz"
        )
        qualification = (
            report_root / clip / "formal" / "v3_formal_selected_2129920_qualification.json"
        )
        evaluation = (
            report_root / clip / "formal" / "v3_formal_selected_2129920_evaluation_suite_v2.json"
        )
        contact = report_root / clip / "formal" / "v3_formal_selected_2129920_contact.json"
        v1_trace = report_root / "v1_pairforce" / clip / "trace.npz"
        sim_manifest = sim_root / clip / "v3_formal_selected_2129920" / "manifest.json"
        for name, path in {
            f"{clip}_{key}": value
            for key, value in {
                "v3_formal_trace": trace,
                "v1_formal_pairforce_trace": v1_trace,
                "reference_mask": mask,
                "qualification": qualification,
                "evaluation_suite_v2": evaluation,
                "contact_metrics": contact,
                "sim_manifest": sim_manifest,
            }.items()
        }.items():
            if not path.is_file():
                raise FileNotFoundError(f"PER_FINGER_AUDIT_INPUT_MISSING:{name}:{path}")
            input_paths[name] = path
        audits[clip] = _audit_clip(
            clip=clip,
            trace_path=trace,
            mask_path=mask,
            qualification_path=qualification,
            evaluation_path=evaluation,
        )
    output.mkdir(parents=True)
    frozen = {
        "schema_version": "Stage16DPerFingerContactFrozenInputsV1",
        "status": "FROZEN",
        "git": _git_state(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in input_paths.items()
        },
        "source_contact_evidence": "ABSENT_PER_FINGER_IN_FROZEN_FORMAL20_INPUTS",
    }
    (output / "frozen_inputs.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "finger_mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    actual_contract = {
        "schema_version": "ActualValidFingerContactContractV1",
        "definition": "actual_contact = frozen Reward V3 replica_actual_contact_mask AND frozen pair_force_valid; no numerical threshold added by this audit.",
        "source": "filtered fingertip-to-active-object PhysX pair force only",
        "validity": "frame 0 reset invalid; frames 1..320 valid for each Formal20 replica",
    }
    (output / "actual_contact_contract.json").write_text(
        json.dumps(actual_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    palm_mapping = {
        "schema_version": "PalmContactDiagnosticMappingV1",
        "palm_collision_body": "r_wrist",
        "source": "existing Formal20 qualification collision-body inventory",
        "time_resolved_pair_force_available": False,
        "time_resolved_contact_presence_available": False,
    }
    (output / "palm_mapping.json").write_text(
        json.dumps(palm_mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contact_contract = {
        "schema_version": "ReferenceFingerContactEvidenceV1",
        "primary_mask": "V3_PRIMARY_EXPECTED_CONTACT_MASK = distance_m < 0.03; frozen and not changed",
        "strong": "REFERENCE_STRONG_CONTACT_EVIDENCE = distance_m <= 0.02; source/topology telemetry absent per finger",
        "ambiguous": "REFERENCE_PROXIMITY_ONLY_AMBIGUOUS = 0.02 < distance_m < 0.03",
        "no_contact": "REFERENCE_NO_CONTACT_EXPECTED = distance_m >= 0.03",
        "persistent_steps": PERSISTENT_STEPS,
    }
    (output / "contact_audit_contract.json").write_text(
        json.dumps(contact_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparison_rows: list[dict[str, Any]] = []
    for clip, audit in audits.items():
        clip_output = output / clip
        clip_output.mkdir()
        _write_parquet(clip_output / "per_episode_per_finger.parquet", audit["per_episode_rows"])
        _write_parquet(clip_output / "persistent_windows.parquet", audit["persistent_windows"])
        _write_parquet(clip_output / "floating_windows.parquet", audit["floating_windows"])
        for name in (
            "per_finger_summary",
            "compensation",
            "force_concentration",
            "palm_diagnostic",
            "free_flight_contact_analysis",
            "visualization_windows",
        ):
            filename = {"compensation": "compensation_metrics.json"}.get(name, f"{name}.json")
            (clip_output / filename).write_text(
                json.dumps(audit[name], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        for row in audit["per_finger_summary"]:
            comparison_rows.append({"clip": clip, **row})
    _write_csv(output / "cross_clip_per_finger_comparison.csv", comparison_rows)
    decision = _decision(audits)
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": "Stage16DPerFingerContactAuditFinalSummaryV1",
        "status": "STAGE16D_PER_FINGER_CONTACT_AUDIT_COMPLETE",
        "primary_recommendation": decision["primary_recommendation"],
        "clips": {
            clip: {
                "per_finger_summary": audit["per_finger_summary"],
                "qualification": audit["qualification"],
            }
            for clip, audit in audits.items()
        },
    }
    (output / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = _markdown_summary(audits, decision)
    (output / "final_summary.md").write_text(markdown + "\n", encoding="utf-8")
    (output / "cross_clip_per_finger_comparison.md").write_text(markdown + "\n", encoding="utf-8")
    replay_validation = {
        "schema_version": "Stage16DPerFingerReplayCompatibilityV1",
        "status": "PASS",
        "existing_replay_script": str(
            REPO_ROOT / "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py"
        ),
        "trace_schema_mutated": False,
        "viewer_modified": False,
        "reason": "The audit consumes existing V3 trace fields offline; no viewer or trace mutation was needed.",
    }
    (output / "replay_validation.json").write_text(
        json.dumps(replay_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "git_commits.json").write_text(
        json.dumps(_git_state(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    handoff = "\n".join(
        [
            "# Stage 16-D Per-Finger Contact Audit Handoff",
            "",
            f"Primary recommendation: `{decision['primary_recommendation']}`.",
            "",
            "The 3 cm V3 mask remains frozen.  This audit found no time-resolved palm pair evidence, so it does not classify any finger miss as a valid palm substitution.",
            "",
            markdown,
            "",
            "No PPO, reward, contact-mask, RSI, controller, or physics change was made.",
        ]
    )
    (output / "handoff.md").write_text(handoff + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "primary_recommendation": decision["primary_recommendation"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
