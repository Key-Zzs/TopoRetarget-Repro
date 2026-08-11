#!/usr/bin/env python3
"""Summarize exact V3 fingertip--active-object contact evidence offline.

This tool is deliberately trace-only.  It neither launches IsaacLab nor derives
fingertip force from aggregate force, so its contact metrics remain auditable
against the V1 Formal20 pair-force calibration source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.reference_gated_contact import (  # noqa: E402
    EVALUATION_FINGERTIP_LINKS,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _longest_run(values: np.ndarray) -> tuple[int, int, int]:
    """Return longest loss gap, loss-onset count, and recontact count for `[T]`."""

    longest = current = losses = recontacts = 0
    previous = False
    for value in values.astype(bool):
        if value:
            current += 1
            losses += int(not previous)
        else:
            recontacts += int(previous)
            current = 0
        longest = max(longest, current)
        previous = bool(value)
    return longest, losses, recontacts


def _persistent_windows(expected_any: np.ndarray, *, minimum_steps: int = 3) -> np.ndarray:
    """Mark every sample belonging to an expected-contact run of at least three."""

    result = np.zeros_like(expected_any, dtype=bool)
    start = 0
    while start < expected_any.shape[0]:
        end = start
        while end < expected_any.shape[0] and expected_any[end]:
            end += 1
        if end - start >= minimum_steps:
            result[start:end] = True
        start = end + 1 if end == start else end
    return result


def _statistics(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("V3_CONTACT_FORCE_STATISTICS_NONFINITE")
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def _load_frozen_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["reference_expected_contact_mask"], dtype=bool)
    if mask.shape != (321, 5):
        raise ValueError(f"V3_CONTACT_FROZEN_MASK_SHAPE_INVALID:{mask.shape}")
    return mask


def _load_pair_force(
    path: Path, *, require_v3_contact_fields: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "replica_fingertip_object_pair_force_world",
            "replica_fingertip_object_pair_force_valid",
        }
        if require_v3_contact_fields:
            required.update({"replica_reference_contact_mask", "replica_actual_contact_mask"})
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"V3_CONTACT_TRACE_FIELDS_MISSING:{missing}")
        force = np.asarray(archive["replica_fingertip_object_pair_force_world"])
        valid = np.asarray(archive["replica_fingertip_object_pair_force_valid"], dtype=bool)
        if require_v3_contact_fields:
            reference = np.asarray(archive["replica_reference_contact_mask"], dtype=bool)
            actual = np.asarray(archive["replica_actual_contact_mask"], dtype=bool)
        else:
            reference = np.empty((0,), dtype=bool)
            actual = np.linalg.vector_norm(force, axis=-1) > 1.0e-4
        links = tuple(str(value) for value in archive["fingertip_link_names"].tolist())
        if links != EVALUATION_FINGERTIP_LINKS:
            raise ValueError(f"V3_CONTACT_FINGERTIP_LINK_ORDER_INVALID:{links}")
        if "pair_force_frame" in archive.files and _scalar(archive["pair_force_frame"]) != "world":
            raise ValueError("V3_CONTACT_PAIR_FORCE_NOT_WORLD_FRAME")
    if force.shape[:2] != valid.shape or force.shape[-2:] != (5, 3):
        raise ValueError(f"V3_CONTACT_PAIR_FORCE_SHAPE_INVALID:{force.shape}:{valid.shape}")
    if force.shape[0] != 321 or force.shape[1] != 20:
        raise ValueError("V3_CONTACT_REQUIRES_EXACT_FORMAL20_SHAPE")
    if valid[0].any() or not valid[1:].all():
        raise ValueError("V3_CONTACT_PAIR_FORCE_VALIDITY_MUST_EXCLUDE_ONLY_RESET")
    if not np.isfinite(force[valid]).all():
        raise ValueError("V3_CONTACT_PAIR_FORCE_VALID_SAMPLES_NONFINITE")
    return force.astype(np.float64), valid, reference, actual


def _replica_metrics(
    *, force: np.ndarray, valid: np.ndarray, expected: np.ndarray, actual: np.ndarray
) -> tuple[list[dict[str, Any]], np.ndarray]:
    magnitudes = np.linalg.vector_norm(force, axis=-1)
    rows: list[dict[str, Any]] = []
    for replica in range(force.shape[1]):
        usable = valid[:, replica]
        expected_row = expected[:, replica] & usable[:, None]
        actual_row = actual[:, replica] & usable[:, None]
        expected_any = expected_row.any(axis=-1)
        matched_any = (expected_row & actual_row).any(axis=-1)
        loss = expected_any & ~matched_any
        persistent = _persistent_windows(expected_any)
        expected_count = int(expected_row.sum())
        persistent_count = int((expected_row & persistent[:, None]).sum())
        longest, loss_events, recontact_events = _longest_run(loss)
        active_force = magnitudes[:, replica][actual_row]
        scale = (magnitudes[:, replica] * expected_row).sum(axis=-1)
        rows.append(
            {
                "replica": replica,
                "expected_contact_recall": (
                    float((expected_row & actual_row).sum() / expected_count)
                    if expected_count
                    else None
                ),
                "persistent_contact_recall": (
                    float(
                        ((expected_row & actual_row & persistent[:, None]).sum()) / persistent_count
                    )
                    if persistent_count
                    else None
                ),
                "unexpected_contact_rate": float(
                    (actual_row & ~expected_row).sum()
                    / max(int((usable[:, None] & ~expected_row).sum()), 1)
                ),
                "actual_contact_fraction": float(actual_row.sum() / max(int(usable.sum() * 5), 1)),
                "longest_contact_loss_gap": int(longest),
                "contact_loss_event_count": int(loss_events),
                "recontact_event_count": int(recontact_events),
                "terminal_contact": bool(actual_row[-20:].any()),
                "contact_force": _statistics(active_force),
                "total_contact_impulse_ns": float(scale[usable].sum() / 20.0),
            }
        )
    return rows, magnitudes


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row[key], (float, int))]
    return None if not values else float(np.mean(values))


def _p95_ratio(v3_p95: float | None, baseline_p95: float | None) -> float | None:
    """Return a force-farming diagnostic only when both cohorts contacted."""

    if not isinstance(v3_p95, float) or not isinstance(baseline_p95, float) or baseline_p95 <= 0.0:
        return None
    return v3_p95 / baseline_p95


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frozen-contact-mask", type=Path, required=True)
    parser.add_argument("--v1-pairforce-trace", type=Path, required=True)
    parser.add_argument(
        "--actual-contact-from-pair-force-threshold-n",
        type=float,
        help=(
            "Use direct exact pair-force magnitude > this threshold as the actual-contact "
            "indicator for a V1 development baseline. Omit for a V3 reward-sensor trace."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = args.trace.resolve()
    frozen_mask = _load_frozen_mask(args.frozen_contact_mask.resolve())
    pair_force_threshold = args.actual_contact_from_pair_force_threshold_n
    if pair_force_threshold is None:
        force, valid, expected, actual = _load_pair_force(trace, require_v3_contact_fields=True)
        if expected.shape != (321, 20, 5) or actual.shape != expected.shape:
            raise ValueError("V3_CONTACT_MASK_TRACE_SHAPE_INVALID")
        if not np.array_equal(expected, np.broadcast_to(frozen_mask[:, None], expected.shape)):
            raise ValueError("V3_CONTACT_REFERENCE_MASK_DRIFT")
        actual_contact_source = "RewardV3 current exact pair-force contact mask"
    else:
        if not np.isfinite(pair_force_threshold) or pair_force_threshold <= 0.0:
            raise ValueError("V3_CONTACT_PAIR_FORCE_THRESHOLD_INVALID")
        force, valid, _, _ = _load_pair_force(trace, require_v3_contact_fields=False)
        expected = np.broadcast_to(frozen_mask[:, None], (321, 20, 5)).copy()
        actual = np.linalg.vector_norm(force, axis=-1) > pair_force_threshold
        actual_contact_source = (
            "direct exact active-object pair-force magnitude "
            f"> {pair_force_threshold:g} N; no aggregate force"
        )
    rows, magnitudes = _replica_metrics(force=force, valid=valid, expected=expected, actual=actual)
    v1_force, v1_valid, _, v1_actual = _load_pair_force(
        args.v1_pairforce_trace.resolve(), require_v3_contact_fields=False
    )
    v1_active = np.linalg.vector_norm(v1_force, axis=-1)[v1_actual & v1_valid[..., None]]
    v3_active = magnitudes[actual & valid[..., None]]
    baseline_force = _statistics(v1_active)
    v3_force = _statistics(v3_active)
    baseline_p95 = baseline_force["p95"]
    v3_p95 = v3_force["p95"]
    result = {
        "schema_version": "ReferenceContactEvaluationV1",
        "status": "REFERENCE_CONTACT_EVALUATION_COMPLETE",
        "trace": str(trace),
        "trace_sha256": _hash(trace),
        "actual_contact_source": actual_contact_source,
        "frozen_contact_mask": str(args.frozen_contact_mask.resolve()),
        "frozen_contact_mask_sha256": _hash(args.frozen_contact_mask.resolve()),
        "v1_pairforce_baseline": {
            "trace": str(args.v1_pairforce_trace.resolve()),
            "trace_sha256": _hash(args.v1_pairforce_trace.resolve()),
            "active_pair_force_n": baseline_force,
        },
        "aggregate": {
            "expected_contact_recall": _mean(rows, "expected_contact_recall"),
            "persistent_contact_recall": _mean(rows, "persistent_contact_recall"),
            "unexpected_contact_rate": _mean(rows, "unexpected_contact_rate"),
            "actual_contact_fraction": _mean(rows, "actual_contact_fraction"),
            "longest_contact_loss_gap": _mean(rows, "longest_contact_loss_gap"),
            "contact_loss_event_count": _mean(rows, "contact_loss_event_count"),
            "recontact_event_count": _mean(rows, "recontact_event_count"),
            "terminal_contact_rate": _mean(rows, "terminal_contact"),
            "contact_force_n": v3_force,
            "total_contact_impulse_ns": _mean(rows, "total_contact_impulse_ns"),
            "force_p95_over_v1": _p95_ratio(v3_p95, baseline_p95),
        },
        "per_replica": rows,
        "persistent_contact_definition": (
            "reference mask any fingertip active for >=3 control steps"
        ),
        "force_farming_rule": (
            "suspected only when V3 force p95 > 3x V1 Formal20 p95 and penetration worsens "
            "or SR_physics declines; this trace-only report does not assert that combined gate"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
