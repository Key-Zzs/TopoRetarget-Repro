#!/usr/bin/env python3
"""Classify impulse--free-flight--re-catch evidence from exact Formal20 pair force."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["reference_expected_contact_mask"], dtype=bool)
    if mask.shape != (321, 5):
        raise ValueError("V3_FREE_FLIGHT_REFERENCE_MASK_SHAPE_INVALID")
    return mask


def _load_trace(
    path: Path, *, v3: bool, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "replica_fingertip_object_pair_force_world",
            "replica_fingertip_object_pair_force_valid",
            "replica_object_pose",
            "replica_object_twist",
        }
        if v3:
            required.update({"replica_reference_contact_mask", "replica_actual_contact_mask"})
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"V3_FREE_FLIGHT_TRACE_FIELDS_MISSING:{missing}")
        force = np.asarray(archive["replica_fingertip_object_pair_force_world"], dtype=np.float64)
        valid = np.asarray(archive["replica_fingertip_object_pair_force_valid"], dtype=bool)
        pose = np.asarray(archive["replica_object_pose"], dtype=np.float64)
        twist = np.asarray(archive["replica_object_twist"], dtype=np.float64)
        if v3:
            expected = np.asarray(archive["replica_reference_contact_mask"], dtype=bool)
            actual = np.asarray(archive["replica_actual_contact_mask"], dtype=bool)
        else:
            expected = np.broadcast_to(mask[:, None], (321, 20, 5)).copy()
            actual = np.linalg.norm(force, axis=-1) > 1.0e-4
    if (
        force.shape != (321, 20, 5, 3)
        or valid.shape != (321, 20)
        or pose.shape != (321, 20, 7)
        or twist.shape != (321, 20, 6)
        or expected.shape != (321, 20, 5)
        or actual.shape != (321, 20, 5)
        or valid[0].any()
        or not valid[1:].all()
    ):
        raise ValueError("V3_FREE_FLIGHT_TRACE_SHAPE_OR_VALIDITY_INVALID")
    if v3 and not np.array_equal(expected, np.broadcast_to(mask[:, None], expected.shape)):
        raise ValueError("V3_FREE_FLIGHT_REFERENCE_MASK_DRIFT")
    return force, valid, actual, pose, twist, expected


def _events(
    *,
    force: np.ndarray,
    valid: np.ndarray,
    actual: np.ndarray,
    pose: np.ndarray,
    twist: np.ndarray,
    expected: np.ndarray,
) -> list[dict[str, Any]]:
    magnitude = np.linalg.norm(force, axis=-1)
    result: list[dict[str, Any]] = []
    for replica in range(20):
        expected_any = expected[:, replica].any(axis=-1) & valid[:, replica]
        matched = (expected[:, replica] & actual[:, replica]).any(axis=-1) & valid[:, replica]
        loss = expected_any & ~matched
        start = 0
        while start < len(loss):
            if not loss[start]:
                start += 1
                continue
            end = start
            while end < len(loss) and loss[end]:
                end += 1
            recontact = end < len(loss) and bool(matched[end])
            preloss = max(start - 1, 0)
            s_contact_pre_loss = float(
                (magnitude[preloss, replica] * expected[preloss, replica]).sum()
            )
            if end - start >= 3 and recontact and s_contact_pre_loss > 0.0:
                result.append(
                    {
                        "replica": replica,
                        "loss_start_frame": start,
                        "loss_end_exclusive": end,
                        "loss_duration_control_steps": end - start,
                        "recontact_frame": end,
                        "pre_loss_S_contact_n": s_contact_pre_loss,
                        "free_flight_object_displacement_m": float(
                            np.linalg.norm(pose[end - 1, replica, :3] - pose[start, replica, :3])
                        ),
                        "free_flight_max_linear_speed_mps": float(
                            np.linalg.norm(twist[start:end, replica, :3], axis=-1).max()
                        ),
                        "free_flight_max_angular_speed_radps": float(
                            np.linalg.norm(twist[start:end, replica, 3:], axis=-1).max()
                        ),
                    }
                )
            start = end
    return result


def _classification(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> str:
    if not baseline:
        return "NO_CLEAR_FREE_FLIGHT_PATTERN"
    if not candidate:
        return "FREE_FLIGHT_RECATCH_RESOLVED"
    if len(candidate) < len(baseline):
        return "FREE_FLIGHT_RECATCH_REDUCED"
    return "FREE_FLIGHT_RECATCH_PERSISTS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-pairforce-trace", type=Path, required=True)
    parser.add_argument("--v3-trace", type=Path, required=True)
    parser.add_argument("--frozen-contact-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mask_path = args.frozen_contact_mask.resolve()
    mask = _load_mask(mask_path)
    v1_path = args.v1_pairforce_trace.resolve()
    v3_path = args.v3_trace.resolve()
    v1_force, v1_valid, v1_actual, v1_pose, v1_twist, v1_expected = _load_trace(
        v1_path, v3=False, mask=mask
    )
    v3_force, v3_valid, v3_actual, v3_pose, v3_twist, v3_expected = _load_trace(
        v3_path, v3=True, mask=mask
    )
    baseline = _events(
        force=v1_force,
        valid=v1_valid,
        actual=v1_actual,
        pose=v1_pose,
        twist=v1_twist,
        expected=v1_expected,
    )
    candidate = _events(
        force=v3_force,
        valid=v3_valid,
        actual=v3_actual,
        pose=v3_pose,
        twist=v3_twist,
        expected=v3_expected,
    )
    result = {
        "schema_version": "Stage16DRewardV3FreeFlightRecontactAnalysisV1",
        "status": _classification(baseline, candidate),
        "definition": (
            "reference-expected contact loss for >=3 control steps after nonzero pre-loss "
            "S_contact and followed by expected-finger recontact; object displacement "
            "and twist are recorded through the entire loss window"
        ),
        "inputs": {
            "v1_pairforce_trace": {"path": str(v1_path), "sha256": _sha256(v1_path)},
            "v3_trace": {"path": str(v3_path), "sha256": _sha256(v3_path)},
            "frozen_contact_mask": {"path": str(mask_path), "sha256": _sha256(mask_path)},
        },
        "v1_event_count": len(baseline),
        "v3_event_count": len(candidate),
        "v1_events": baseline,
        "v3_events": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": result["status"], "v1_events": len(baseline), "v3_events": len(candidate)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
