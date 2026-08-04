#!/usr/bin/env python3
"""Summarize the E3 origin and E6 contact-telemetry portions of C.5A R1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--telemetry-off", type=Path, required=True)
    parser.add_argument("--telemetry-diagnostic", type=Path, required=True)
    parser.add_argument("--e3-output", type=Path, required=True)
    parser.add_argument("--e6-output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, telemetry: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"C5A diagnostic is not an object: {path}")
    if (
        report.get("mode") != "vector"
        or report.get("process_mode") != "same_process"
        or report.get("num_envs") != 33
        or report.get("trials") != 20
        or report.get("telemetry") != telemetry
        or report.get("snapshot_restore_used") is not False
    ):
        raise ValueError(f"C5A vector diagnostic contract mismatch: {path}")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError(f"C5A vector diagnostic requires 2 clips x 4 phases: {path}")
    return report


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_SUMMARY_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fingerprints(row: dict[str, Any]) -> list[dict[str, str]]:
    values = row.get("measurement_fingerprints")
    if not isinstance(values, list) or len(values) != 20:
        raise ValueError("C5A diagnostic row lacks 20 fingerprints")
    result: list[dict[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("C5A diagnostic fingerprint is malformed")
        raw, derived = value.get("raw"), value.get("derived")
        if not isinstance(raw, str) or not isinstance(derived, str):
            raise ValueError("C5A diagnostic fingerprint keys are malformed")
        result.append({"raw": raw, "derived": derived})
    return result


def _aligned_rows(reports: list[dict[str, Any]]) -> list[tuple[dict[str, Any], ...]]:
    reference = reports[0]["rows"]
    assert isinstance(reference, list)
    aligned: list[tuple[dict[str, Any], ...]] = []
    for index, row in enumerate(reference):
        if not isinstance(row, dict):
            raise ValueError("C5A diagnostic row is malformed")
        rows: list[dict[str, Any]] = [row]
        for report in reports[1:]:
            candidate_rows = report["rows"]
            assert isinstance(candidate_rows, list)
            candidate = candidate_rows[index]
            if not isinstance(candidate, dict) or any(
                candidate.get(key) != row.get(key) for key in ("clip", "phase", "frame")
            ):
                raise ValueError("C5A diagnostics do not share the same phase matrix")
            rows.append(candidate)
        aligned.append(tuple(rows))
    return aligned


def summarize_telemetry(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare byte-identical physics state over off/aggregate/diagnostic modes."""

    modes = ("off", "aggregate", "diagnostic")
    rows: list[dict[str, Any]] = []
    for aligned in _aligned_rows(reports):
        fingerprints = [_fingerprints(row) for row in aligned]
        physical_trials = [
            len({fingerprints[mode_index][trial]["raw"] for mode_index in range(3)}) == 1
            for trial in range(20)
        ]
        derived_trials = [
            len({fingerprints[mode_index][trial]["derived"] for mode_index in range(3)}) == 1
            for trial in range(20)
        ]
        reference = aligned[0]
        rows.append(
            {
                "clip": reference["clip"],
                "phase": reference["phase"],
                "frame": reference["frame"],
                "raw_physics_fingerprint_identical_all_trials": all(physical_trials),
                "derived_fingerprint_identical_all_trials": all(derived_trials),
                "raw_physics_identical_trial_count": sum(physical_trials),
                "derived_identical_trial_count": sum(derived_trials),
            }
        )
    physics_identical = all(row["raw_physics_fingerprint_identical_all_trials"] for row in rows)
    return {
        "schema_version": "stage16c5_contact_telemetry_effect_v1",
        "modes": list(modes),
        "trajectory_contract": (
            "same frozen reset, same clip/frame/action history, same vector scene"
        ),
        "comparison": (
            "byte-level fingerprints of raw simulator/candidate state and derived task state"
        ),
        "rows": rows,
        "result": (
            "CONTACT_TELEMETRY_READ_ONLY_CONFIRMED"
            if physics_identical
            else "CONTACT_TELEMETRY_ALTERS_DYNAMICS"
        ),
    }


def summarize_origin(report: dict[str, Any]) -> dict[str, Any]:
    """Show that origin subtraction is correct without hiding genuine local drift."""

    rows: list[dict[str, Any]] = []
    for row in report["rows"]:
        assert isinstance(row, dict)
        checks = row.get("origin_invariance")
        if not isinstance(checks, list) or len(checks) != 20:
            raise ValueError("C5A vector diagnostic row lacks 20 origin checks")
        unique_counts = {
            check.get("unique_origin_count") for check in checks if isinstance(check, dict)
        }
        if unique_counts != {33}:
            raise ValueError("C5A E3 did not preserve 33 unique environment origins")
        fields: dict[str, dict[str, float]] = {}
        for name in (
            "robot_root_state",
            "object_170105_root_state",
            "object_170650_root_state",
        ):
            world = []
            local = []
            rebased = []
            for check in checks:
                assert isinstance(check, dict)
                field = check.get(name)
                if not isinstance(field, dict):
                    raise ValueError(f"C5A E3 lacks {name}")
                world.append(float(field["world_position_max_abs"]))
                local.append(float(field["scene_local_max_abs"]))
                rebased.append(float(field["world_minus_origin_delta_max_abs"]))
            fields[name] = {
                "global_position_max_abs": max(world),
                "scene_local_max_abs": max(local),
                "world_minus_origin_delta_max_abs": max(rebased),
                "origin_subtraction_consistent": max(
                    abs(left - right) for left, right in zip(local, rebased, strict=True)
                )
                <= 1.0e-6,
            }
        rows.append(
            {
                "clip": row["clip"],
                "phase": row["phase"],
                "frame": row["frame"],
                "unique_origin_count": 33,
                "fields": fields,
            }
        )
    valid = all(
        value["origin_subtraction_consistent"] for row in rows for value in row["fields"].values()
    )
    return {
        "schema_version": "stage16c5_env_origin_invariance_v1",
        "comparison": "world coordinates versus world coordinates rebased by each env origin",
        "rows": rows,
        "result": (
            "ENV_ORIGIN_NORMALIZATION_VALID" if valid else "ENV_ORIGIN_NORMALIZATION_FAILURE"
        ),
        "interpretation": (
            "A nonzero scene-local value is retained as physical peer divergence; "
            "it is not hidden by global origin offsets."
        ),
    }


def main() -> int:
    args = parse_args()
    aggregate = _load(args.aggregate, "aggregate")
    telemetry_off = _load(args.telemetry_off, "off")
    telemetry_diagnostic = _load(args.telemetry_diagnostic, "diagnostic")
    _write(args.e3_output, summarize_origin(aggregate))
    _write(
        args.e6_output,
        summarize_telemetry([telemetry_off, aggregate, telemetry_diagnostic]),
    )
    print(json.dumps({"result": "STAGE16C5A_E3_E6_SUMMARY_COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
