#!/usr/bin/env python3
"""Audit a V4-mode Formal20 trace against the frozen source-tip contract.

The script is intentionally offline.  It does not start IsaacLab, step a
policy, mutate a checkpoint, or reinterpret source labels.  It only consumes
the exact named-tip and optional full-hand active-object telemetry already
captured by the V4 evaluation mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# The stable audit JSON keys are intentionally descriptive.
# ruff: noqa: E501

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TIP_NAMES = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)
TIP_INDICES = (20, 4, 8, 16, 12)
FRAME_COUNT = 321
FORMAL_REPLICAS = 20
PERSISTENCE_STEPS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_AUDIT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("STRICT_V4_AUDIT_NONFINITE_STATISTIC")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(values, dtype=bool)
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


def _persistent(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=bool)
    for start, end in _runs(values):
        if end - start >= PERSISTENCE_STEPS:
            result[start:end] = True
    return result


def _persistent_3d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if values.shape != (FRAME_COUNT, FORMAL_REPLICAS):
        raise ValueError("STRICT_V4_AUDIT_PERSISTENCE_SHAPE_INVALID")
    result = np.zeros_like(values)
    for replica in range(FORMAL_REPLICAS):
        result[:, replica] = _persistent(values[:, replica])
    return result


def _rate(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    denominator = np.asarray(denominator, dtype=bool)
    if not denominator.any():
        return None
    return float(
        np.count_nonzero(np.asarray(numerator, dtype=bool) & denominator) / denominator.sum()
    )


def _longest(values: np.ndarray) -> int:
    return max((end - start for start, end in _runs(values)), default=0)


def _load_source_mask(path: Path, *, clip: str, contract: dict[str, Any]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        required = {"strict_source_contact_mask", "finger_names", "control_index"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"STRICT_V4_AUDIT_SOURCE_MASK_FIELDS_MISSING:{missing}")
        mask = np.asarray(archive["strict_source_contact_mask"], dtype=bool)
        names = tuple(str(value) for value in archive["finger_names"].tolist())
        control = np.asarray(archive["control_index"], dtype=np.int64)
    frozen = contract.get("frozen_parameters", {})
    if (
        mask.shape != (FRAME_COUNT, len(FINGERS))
        or names != FINGERS
        or not np.array_equal(control, np.arange(FRAME_COUNT))
        or tuple(frozen.get("finger_order", ())) != FINGERS
        or contract.get("status") != "STRICT_V4_CONTACT_CONTRACT_FROZEN"
        or not clip.startswith("hocap_")
    ):
        raise ValueError("STRICT_V4_AUDIT_SOURCE_MASK_CONTRACT_DRIFT")
    return mask


def _group_indices(names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], int]:
    groups: dict[str, np.ndarray] = {}
    prefixes = {
        "thumb": "r_thumb_",
        "index": "r_index_finger_",
        "middle": "r_middle_finger_",
        "ring": "r_ring_finger_",
        "pinky": "r_pinky_",
    }
    for finger, prefix in prefixes.items():
        indices = np.asarray([index for index, name in enumerate(names) if name.startswith(prefix)])
        if len(indices) != 4:
            raise ValueError(f"STRICT_V4_AUDIT_FINGER_GROUP_INVALID:{finger}:{indices.tolist()}")
        groups[finger] = indices
    if len(names) != 21 or len(set(names)) != 21 or "r_wrist" not in names:
        raise ValueError("STRICT_V4_AUDIT_HAND_BODY_MAPPING_INVALID")
    if tuple(names[index] for index in TIP_INDICES) != TIP_NAMES:
        raise ValueError("STRICT_V4_AUDIT_TIP_SENSOR_MAPPING_DRIFT")
    return groups, names.index("r_wrist")


def _load_trace(path: Path, *, mask: np.ndarray) -> dict[str, np.ndarray | tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "replica_source_contact_mask",
            "replica_tip_pair_presence",
            "replica_tip_pair_force_world",
            "replica_hand_object_pair_force_world",
            "replica_hand_object_pair_presence",
            "replica_hand_object_pair_force_valid",
            "replica_object_pose",
            "replica_object_twist",
            "replica_reward_total",
            "hand_body_names",
            "fingertip_link_names",
            "fingertip_force_sensor_indices",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"STRICT_V4_AUDIT_TRACE_FIELDS_MISSING:{missing}")
        result: dict[str, np.ndarray | tuple[str, ...]] = {
            name: np.asarray(archive[name])
            for name in required
            if name not in {"hand_body_names", "fingertip_link_names"}
        }
        result["hand_body_names"] = tuple(
            str(value) for value in archive["hand_body_names"].tolist()
        )
        result["fingertip_link_names"] = tuple(
            str(value) for value in archive["fingertip_link_names"].tolist()
        )
        for optional in ("replica_r_contact_v4", "replica_per_finger_contact_reward"):
            if optional in archive.files:
                result[optional] = np.asarray(archive[optional])
    source = np.asarray(result["replica_source_contact_mask"], dtype=bool)
    tip_presence = np.asarray(result["replica_tip_pair_presence"], dtype=bool)
    tip_force = np.asarray(result["replica_tip_pair_force_world"], dtype=np.float64)
    hand_force = np.asarray(result["replica_hand_object_pair_force_world"], dtype=np.float64)
    hand_presence = np.asarray(result["replica_hand_object_pair_presence"], dtype=bool)
    valid = np.asarray(result["replica_hand_object_pair_force_valid"], dtype=bool)
    names = result["hand_body_names"]
    tip_names = result["fingertip_link_names"]
    indices = tuple(
        int(value) for value in np.asarray(result["fingertip_force_sensor_indices"]).tolist()
    )
    if not isinstance(names, tuple) or not isinstance(tip_names, tuple):
        raise AssertionError("STRICT_V4_AUDIT_INTERNAL_TRACE_TYPE")
    if (
        source.shape != (FRAME_COUNT, FORMAL_REPLICAS, len(FINGERS))
        or tip_presence.shape != source.shape
        or tip_force.shape != (*source.shape, 3)
        or hand_force.shape != (FRAME_COUNT, FORMAL_REPLICAS, 21, 3)
        or hand_presence.shape != hand_force.shape[:-1]
        or valid.shape != hand_presence.shape[:2]
        or not np.array_equal(source, np.broadcast_to(mask[:, None], source.shape))
        or tip_names != TIP_NAMES
        or indices != TIP_INDICES
        or valid[0].any()
        or not valid[1:].all()
        or hand_presence[0].any()
        or np.any(hand_force[0] != 0.0)
        or not np.isfinite(tip_force[valid]).all()
        or not np.isfinite(hand_force[valid]).all()
    ):
        raise ValueError("STRICT_V4_AUDIT_TRACE_SHAPE_OR_SEMANTICS_INVALID")
    pair_presence = hand_presence[:, :, TIP_INDICES] & valid[..., None]
    if not np.array_equal(tip_presence & valid[..., None], pair_presence):
        raise ValueError("STRICT_V4_AUDIT_NAMED_TIP_PRESENCE_MISMATCH")
    valid_tips = np.broadcast_to(valid[..., None, None], tip_force.shape)
    if not np.allclose(tip_force[valid_tips], hand_force[:, :, TIP_INDICES][valid_tips]):
        raise ValueError("STRICT_V4_AUDIT_NAMED_TIP_FORCE_MISMATCH")
    return result


def _flight_events(
    *,
    clip: str,
    expected: np.ndarray,
    tip_presence: np.ndarray,
    named_tip: np.ndarray,
    any_hand: np.ndarray,
    object_pose: np.ndarray,
    object_twist: np.ndarray,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    expected_any = expected.any(axis=-1)
    for replica in range(FORMAL_REPLICAS):
        event_masks: dict[str, np.ndarray] = {
            "NO_TIP_CONTACT_FLIGHT": expected_any[:, replica] & ~named_tip[:, replica],
            "NO_HAND_OBJECT_CONTACT_FLIGHT": expected_any[:, replica] & ~any_hand[:, replica],
        }
        for finger, name in enumerate(FINGERS):
            event_masks[f"SOURCE_REQUIRED_{name.upper()}_TIP_LOSS"] = (
                expected[:, replica, finger] & ~tip_presence[:, replica, finger]
            )
        for name, missing in event_masks.items():
            for start, end in _runs(missing):
                if end - start < PERSISTENCE_STEPS:
                    continue
                target_contact = (
                    tip_presence[:, replica, FINGERS.index(name.split("_")[2].lower())]
                    if name.startswith("SOURCE_REQUIRED_")
                    else named_tip[:, replica]
                )
                recontact = next(
                    (index for index in range(end, FRAME_COUNT) if target_contact[index]), None
                )
                result.append(
                    {
                        "clip": clip,
                        "replica": replica,
                        "event_type": name,
                        "start_control_index": start,
                        "end_control_index_exclusive": end,
                        "duration_control_steps": end - start,
                        "recontact_control_index": recontact,
                        "expected_fingers": [
                            FINGERS[index]
                            for index in np.flatnonzero(expected[start:end, replica].any(axis=0))
                        ],
                        "object_z_displacement_m": float(
                            object_pose[end - 1, replica, 2] - object_pose[start, replica, 2]
                        ),
                        "object_vz_at_onset_mps": float(object_twist[start, replica, 2]),
                        "object_vz_mean_mps": float(object_twist[start:end, replica, 2].mean()),
                        "object_vz_max_abs_mps": float(
                            np.abs(object_twist[start:end, replica, 2]).max()
                        ),
                    }
                )
    return result


def audit(
    *, clip: str, trace_path: Path, source_mask_path: Path, contract_path: Path
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    mask = _load_source_mask(source_mask_path, clip=clip, contract=contract)
    trace = _load_trace(trace_path, mask=mask)
    names = trace["hand_body_names"]
    if not isinstance(names, tuple):
        raise AssertionError("STRICT_V4_AUDIT_INTERNAL_NAMES_TYPE")
    groups, wrist = _group_indices(names)
    valid = np.asarray(trace["replica_hand_object_pair_force_valid"], dtype=bool)
    expected = np.asarray(trace["replica_source_contact_mask"], dtype=bool) & valid[..., None]
    hand_presence = (
        np.asarray(trace["replica_hand_object_pair_presence"], dtype=bool) & valid[..., None]
    )
    hand_force = np.linalg.norm(
        np.asarray(trace["replica_hand_object_pair_force_world"], dtype=np.float64), axis=-1
    )
    tip_presence = np.asarray(trace["replica_tip_pair_presence"], dtype=bool) & valid[..., None]
    tip_force = np.linalg.norm(
        np.asarray(trace["replica_tip_pair_force_world"], dtype=np.float64), axis=-1
    )
    named_tip = tip_presence.any(axis=-1)
    any_hand = hand_presence.any(axis=-1)
    rows: list[dict[str, Any]] = []
    aggregate_expected = expected.sum(axis=(0, 2))
    aggregate_matched = (expected & tip_presence).sum(axis=(0, 2))
    persistent_expected = np.zeros_like(expected)
    for replica in range(FORMAL_REPLICAS):
        for finger in range(len(FINGERS)):
            persistent_expected[:, replica, finger] = _persistent(expected[:, replica, finger])
    for finger_index, finger in enumerate(FINGERS):
        tip_index = TIP_INDICES[finger_index]
        group = groups[finger]
        non_tip = group[group != tip_index]
        other_groups = np.concatenate([groups[key] for key in FINGERS if key != finger])
        other_tips = np.asarray(
            [TIP_INDICES[index] for index in range(len(FINGERS)) if index != finger_index]
        )
        active = expected[:, :, finger_index]
        own_tip = hand_presence[:, :, tip_index]
        same_group = hand_presence[:, :, non_tip].any(axis=-1)
        cross_tip = hand_presence[:, :, other_tips].any(axis=-1)
        cross_group = hand_presence[:, :, other_groups].any(axis=-1)
        wrist_contact = hand_presence[:, :, wrist]
        missing = active & ~own_tip & ~same_group & ~cross_group & ~wrist_contact
        persistent = persistent_expected[:, :, finger_index]
        cross_group_compensation = active & ~own_tip & ~same_group & cross_group
        rows.append(
            {
                "clip": clip,
                "finger": finger,
                "source_required_valid_samples": int(active.sum()),
                "source_required_runtime_frames": int(mask[:, finger_index].sum()),
                "source_tip_recall": _rate(own_tip, active),
                "persistent_source_tip_recall": _rate(own_tip, persistent),
                "same_finger_non_tip_substitution_fraction": _rate(
                    active & ~own_tip & same_group, active
                ),
                "cross_finger_tip_compensation_fraction": _rate(
                    active & ~own_tip & ~same_group & cross_tip, active
                ),
                "cross_finger_group_compensation_fraction": _rate(cross_group_compensation, active),
                "persistent_cross_finger_group_compensation_fraction": _rate(
                    _persistent_3d(cross_group_compensation), active
                ),
                "wrist_base_substitution_fraction": _rate(
                    active & ~own_tip & ~same_group & ~cross_group & wrist_contact, active
                ),
                "fully_missing_fraction": _rate(missing, active),
                "longest_fully_missing_control_steps": int(
                    max(
                        (_longest(missing[:, replica]) for replica in range(FORMAL_REPLICAS)),
                        default=0,
                    )
                ),
                "tip_pair_force_n_when_source_required": _stats(
                    tip_force[:, :, finger_index][active]
                ),
                "tip_pair_force_n_when_satisfied": _stats(
                    tip_force[:, :, finger_index][active & own_tip]
                ),
                "other_finger_group_force_n_when_compensating": _stats(
                    hand_force[:, :, other_groups][active & ~own_tip & ~same_group & cross_group]
                ),
            }
        )
    per_replica: list[dict[str, Any]] = []
    for replica in range(FORMAL_REPLICAS):
        active = expected[:, replica]
        matched = active & tip_presence[:, replica]
        persistent = persistent_expected[:, replica]
        expected_any = active.any(axis=-1)
        loss = expected_any & ~matched.any(axis=-1)
        per_replica.append(
            {
                "replica": replica,
                "source_tip_recall": _rate(matched, active),
                "persistent_source_tip_recall": _rate(matched, persistent),
                "full_source_tip_coverage_rate": _rate(
                    (matched.sum(axis=-1) == active.sum(axis=-1)), expected_any
                ),
                "longest_source_contact_loss_gap": _longest(loss),
                "source_contact_loss_event_count": len(_runs(loss)),
                "recontact_event_count": sum(
                    int(not loss[index] and loss[index - 1]) for index in range(1, FRAME_COUNT)
                ),
            }
        )
    object_pose = np.asarray(trace["replica_object_pose"], dtype=np.float64)
    object_twist = np.asarray(trace["replica_object_twist"], dtype=np.float64)
    flights = _flight_events(
        clip=clip,
        expected=expected,
        tip_presence=tip_presence,
        named_tip=named_tip,
        any_hand=any_hand,
        object_pose=object_pose,
        object_twist=object_twist,
    )
    v4_reward = trace.get("replica_r_contact_v4")
    per_finger_reward = trace.get("replica_per_finger_contact_reward")
    result = {
        "schema_version": "Stage16DStrictPerFingerV4SourceAuditV1",
        "status": "STRICT_V4_SOURCE_CONTACT_AUDIT_COMPLETE",
        "clip": clip,
        "trace": {"path": str(trace_path.resolve()), "sha256": _sha256(trace_path)},
        "source_mask": {
            "path": str(source_mask_path.resolve()),
            "sha256": _sha256(source_mask_path),
        },
        "strict_v4_contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "semantics": {
            "source_required_classes": contract["frozen_parameters"]["source_required_classes"],
            "named_tip_links": list(TIP_NAMES),
            "named_tip_sensor_indices": list(TIP_INDICES),
            "force_floor_n": contract["frozen_parameters"]["numerical_floor_n"],
            "cross_finger_definition": "other named tip/group only after own tip and same-finger non-tip are absent",
            "wrist_or_palm": "r_wrist is a wrist-base body; no separate palm body exists",
        },
        "aggregate": {
            "source_tip_recall": _rate(tip_presence, expected),
            "persistent_source_tip_recall": _rate(tip_presence, persistent_expected),
            "full_source_tip_coverage_rate": _rate(
                (expected & tip_presence).sum(axis=-1) == expected.sum(axis=-1),
                expected.any(axis=-1),
            ),
            "longest_source_contact_loss_gap": float(
                np.mean([row["longest_source_contact_loss_gap"] for row in per_replica])
            ),
            "source_contact_loss_event_count": float(
                np.mean([row["source_contact_loss_event_count"] for row in per_replica])
            ),
            "recontact_event_count": float(
                np.mean([row["recontact_event_count"] for row in per_replica])
            ),
            "tip_pair_force_n": _stats(tip_force[expected]),
            "r_contact_v4": _stats(np.asarray(v4_reward)[expected.any(axis=-1)])
            if isinstance(v4_reward, np.ndarray)
            else None,
            "per_finger_contact_reward": _stats(np.asarray(per_finger_reward)[expected])
            if isinstance(per_finger_reward, np.ndarray)
            else None,
            "valid_source_tip_samples": int(aggregate_expected.sum()),
            "satisfied_source_tip_samples": int(aggregate_matched.sum()),
            "no_tip_contact_flight_fraction": _rate(
                expected.any(axis=-1) & ~named_tip, expected.any(axis=-1)
            ),
            "no_hand_object_contact_flight_fraction": _rate(
                expected.any(axis=-1) & ~any_hand, expected.any(axis=-1)
            ),
            "longest_no_tip_flight_gap": float(
                np.mean(
                    [
                        _longest(expected[:, replica].any(axis=-1) & ~named_tip[:, replica])
                        for replica in range(FORMAL_REPLICAS)
                    ]
                )
            ),
            "longest_no_hand_flight_gap": float(
                np.mean(
                    [
                        _longest(expected[:, replica].any(axis=-1) & ~any_hand[:, replica])
                        for replica in range(FORMAL_REPLICAS)
                    ]
                )
            ),
        },
        "per_finger": rows,
        "per_replica": per_replica,
        "no_tip_no_hand_flight_events": flights,
        "no_tip_no_hand_flight_event_counts": {
            event_type: sum(1 for row in flights if row["event_type"] == event_type)
            for event_type in ("NO_TIP_CONTACT_FLIGHT", "NO_HAND_OBJECT_CONTACT_FLIGHT")
        },
        "reward_trace_available": isinstance(v4_reward, np.ndarray),
        "trace_only": True,
    }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"empty": True}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-mask", type=Path, required=True)
    parser.add_argument("--strict-v4-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-finger-csv", type=Path, required=True)
    parser.add_argument("--flight-csv", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        clip=args.clip,
        trace_path=args.trace.resolve(),
        source_mask_path=args.source_mask.resolve(),
        contract_path=args.strict_v4_contract.resolve(),
    )
    _write_json(args.output.resolve(), result)
    _write_csv(args.per_finger_csv.resolve(), result["per_finger"])
    _write_csv(args.flight_csv.resolve(), result["no_tip_no_hand_flight_events"])
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
