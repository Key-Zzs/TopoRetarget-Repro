#!/usr/bin/env python3
"""Apply the frozen full-cycle evaluator to a directory of PhysX traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toporetarget.evaluation.physical_functionality_full_cycle_v1 import (
    evaluate_physical_functionality_full_cycle_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-events", type=Path, required=True)
    parser.add_argument(
        "--destination-signal-root",
        type=Path,
        help=(
            "Optional directory of trace-matched NPZ files containing destination_region, "
            "destination_support_contact, and their validity masks. Missing signals remain "
            "unidentifiable; source-table contact is never substituted."
        ),
    )
    parser.add_argument("--geometry-safe", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PF_FULL_CYCLE_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _destination_signals(
    *, trace_path: Path, count: int, signal_root: Path | None
) -> tuple[dict[str, np.ndarray], bool]:
    if signal_root is None:
        unavailable = np.zeros(count, dtype=bool)
        return {
            "destination_region": unavailable,
            "destination_support_contact": unavailable,
            "destination_region_valid": unavailable,
            "destination_support_valid": unavailable,
        }, False
    path = signal_root / trace_path.name
    if not path.is_file():
        raise FileNotFoundError(f"PF_FULL_CYCLE_DESTINATION_SIGNAL_MISSING:{path}")
    with np.load(path, allow_pickle=False) as signals:
        required = {
            "destination_region",
            "destination_support_contact",
            "destination_region_valid",
            "destination_support_valid",
        }
        missing = sorted(required - set(signals.files))
        if missing:
            raise ValueError(f"PF_FULL_CYCLE_DESTINATION_FIELDS_MISSING:{path}:{missing}")
        result = {name: np.asarray(signals[name], dtype=bool) for name in required}
    if any(value.shape != (count,) for value in result.values()):
        raise ValueError(f"PF_FULL_CYCLE_DESTINATION_SIGNAL_LENGTH_DRIFT:{path}")
    return result, True


def qualify(
    *,
    trace_root: Path,
    output: Path,
    runtime_events: Path,
    destination_signal_root: Path | None,
    geometry_safe: bool,
) -> dict[str, object]:
    events = _read_json(runtime_events)
    if events.get("schema_version") != "HardeningV2RuntimeEventsV1":
        raise ValueError("PF_FULL_CYCLE_HARDENING_RUNTIME_EVENTS_REQUIRED")
    traces = sorted(trace_root.glob("*.npz"))
    if not traces:
        raise ValueError(f"PF_FULL_CYCLE_TRACE_ROOT_EMPTY:{trace_root}")
    output.mkdir(parents=True, exist_ok=True)
    phase_names = (
        "PF_pick",
        "PF_transport",
        "PF_place",
        "PF_release",
        "PF_retreat",
        "PF_full_cycle",
    )
    counts = {
        phase: {status: 0 for status in ("PASS", "FAIL", "NOT_IDENTIFIABLE", "NOT_REACHED")}
        for phase in phase_names
    }
    rows: list[dict[str, object]] = []
    all_destination_signals_available = True
    for trace_path in traces:
        with np.load(trace_path, allow_pickle=False) as trace:
            count = len(trace["object_pose"])
            destination, destination_available = _destination_signals(
                trace_path=trace_path,
                count=count,
                signal_root=destination_signal_root,
            )
            causal = bool(
                np.isfinite(trace["object_pose"]).all() and np.isfinite(trace["wrist_pose"]).all()
            )
            action_safe = bool(
                np.isfinite(trace["action"]).all() and (np.abs(trace["action"]) <= 1.0).all()
            )
            result = evaluate_physical_functionality_full_cycle_v1(
                object_pose_wxyz=trace["object_pose"],
                wrist_pose_wxyz=trace["wrist_pose"],
                tip_pair_presence=trace["tip_pair_presence"],
                hand_object_pair_presence=trace["hand_object_pair_presence"],
                table_object_contact=trace["table_object_contact"],
                interaction_valid=trace["fingertip_object_pair_force_valid"],
                support_valid=(
                    trace["table_object_contact_valid"]
                    if "table_object_contact_valid" in trace
                    else np.ones(count, dtype=bool)
                ),
                reference_lift_onset=int(events["pickup"]),
                reference_events={
                    "source_contact": int(events["contact"]),
                    "persistent_contact": int(events["contact"]),
                    "pickup": int(events["pickup"]),
                    "place": int(events["place"]),
                    "release": int(events["release"]),
                },
                causal_execution=causal,
                geometry_safe=geometry_safe,
                action_bounds_safe=action_safe,
                no_hidden_control=causal,
                **destination,
            )
        statuses = {phase: str(result[phase]["status"]) for phase in phase_names}
        for phase, status in statuses.items():
            counts[phase][status] += 1
        receipt = {
            "schema_version": "HardeningV2FullCycleTraceReceiptV1",
            "trace": str(trace_path.resolve()),
            "destination_signals_available": destination_available,
            "causal_execution": causal,
            "geometry_safe": geometry_safe,
            "action_bounds_safe": action_safe,
            "result": result,
        }
        receipt_path = output / "per_trace" / f"{trace_path.stem}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append({"trace": trace_path.name, **statuses})
        all_destination_signals_available &= destination_available
    summary = {
        "schema_version": "HardeningV2FullCycleQualificationV1",
        "trace_count": len(traces),
        "destination_signals_available_for_all_traces": all_destination_signals_available,
        "source_table_contact_substituted_for_destination_support": False,
        "phase_status_counts": counts,
        "per_trace": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = _parser().parse_args()
    summary = qualify(
        trace_root=args.trace_root,
        output=args.output,
        runtime_events=args.runtime_events,
        destination_signal_root=args.destination_signal_root,
        geometry_safe=args.geometry_safe,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
