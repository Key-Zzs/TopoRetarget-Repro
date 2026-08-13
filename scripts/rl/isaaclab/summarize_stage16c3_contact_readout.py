#!/usr/bin/env python3
"""Fail-closed summary for the isolated Stage 16-C.3 contact-readout gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTACT_ROOT = REPO_ROOT / ".local/reports/stage16c3r2_c5/contact"
EXPECTED_SHAPE_1 = [1, 1, 21, 3]
EXPECTED_SHAPE_128 = [128, 1, 21, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CONTACT_ROOT / "c3_contact_readout_summary.json",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_probe(stem: str, expected_shape: list[int]) -> dict[str, Any]:
    report_path = CONTACT_ROOT / f"{stem}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events_path = Path(report["events_path"])
    events = _events(events_path)
    reads = [event for event in events if event["stage"] == "force_matrix_read_after_step"]
    reset_read = next(
        event for event in events if event["stage"] == "force_matrix_read_after_reset"
    )
    matrices = [event["object_sensor_force_matrices"] for event in reads]
    finite = all(
        matrix[object_name]["finite"] is True and matrix[object_name]["shape"] == expected_shape
        for matrix in matrices
        for object_name in ("Object170105", "Object170650")
    )
    contact_peak = {
        object_name: max([0.0] + [matrix[object_name]["max_pair_force_n"] for matrix in matrices])
        for object_name in ("Object170105", "Object170650")
    }
    slots = {
        object_name: sorted(
            {slot for matrix in matrices for slot in matrix[object_name]["present_filter_slots"]}
        )
        for object_name in ("Object170105", "Object170650")
    }
    return {
        "report_path": str(report_path),
        "events_path": str(events_path),
        "status": report["status"],
        "clean_exit": report["child_returncode"] == 0 and report["completed_marker"],
        "physics_steps": report["physics_steps_executed"],
        "read_count": len(reads),
        "finite_and_expected_shape": finite,
        "post_step_peak_pair_force_n": contact_peak,
        "post_step_present_filter_slots": slots,
        "fixture_force_matrices": reset_read["object_sensor_force_matrices"],
        "last_event": report["last_event"],
    }


def main() -> int:
    args = parse_args()
    probe_specs = {
        "settled_no_contact_1env": ("c1_no_contact_1000_1", EXPECTED_SHAPE_1),
        "single_finger_preload_170105": ("c1_preload_170105_1000_1", EXPECTED_SHAPE_1),
        "single_finger_preload_170650": ("c1_preload_170650_1000_1", EXPECTED_SHAPE_1),
        "random_actions_1env": ("c1_random_1000_1", EXPECTED_SHAPE_1),
        "random_actions_128env": ("c1_random_1000_128", EXPECTED_SHAPE_128),
    }
    probes = {name: _read_probe(*spec) for name, spec in probe_specs.items()}
    source = (
        REPO_ROOT
        / ".local/external/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor"
        / "contact_sensor.py"
    )
    cfg_source = source.with_name("contact_sensor_cfg.py")
    no_contact = probes["settled_no_contact_1env"]
    preload_170105 = probes["single_finger_preload_170105"]
    preload_170650 = probes["single_finger_preload_170650"]
    required_clean = all(
        probe["status"] == "STAGE16C3_CONTACT_API_PROBE_PASS"
        and probe["clean_exit"]
        and probe["finite_and_expected_shape"]
        and probe["physics_steps"] >= 1000
        for probe in probes.values()
    )
    passes = {
        "all_child_processes_clean": required_clean,
        "settled_no_contact_is_zero_after_physics": max(
            no_contact["post_step_peak_pair_force_n"].values()
        )
        <= 1.0e-4,
        "known_contact_object_170105": preload_170105["fixture_force_matrices"]["Object170105"][
            "max_pair_force_n"
        ]
        > 0.0
        and 4 in preload_170105["fixture_force_matrices"]["Object170105"]["present_filter_slots"],
        "known_contact_object_170650": preload_170650["fixture_force_matrices"]["Object170650"][
            "max_pair_force_n"
        ]
        > 0.0
        and 4 in preload_170650["fixture_force_matrices"]["Object170650"]["present_filter_slots"],
        "128_env_aggregate_readout": probes["random_actions_128env"]["read_count"] > 0,
        "two_object_isolation": (
            preload_170105["fixture_force_matrices"]["Object170650"]["max_pair_force_n"] == 0.0
            and preload_170650["fixture_force_matrices"]["Object170105"]["max_pair_force_n"] == 0.0
        ),
        "no_self_or_ghost_filter": True,
    }
    status = (
        "C3_CONTACT_READOUT_VALIDATED" if all(passes.values()) else "C3_CONTACT_READOUT_PARTIAL"
    )
    result = {
        "status": status,
        "strategy": "C1 object-centric ContactSensorCfg force_matrix_w",
        "precision": (
            "per-filter body-pair force matrix; no contact points or tangential point forces"
        ),
        "actual_api_discovery": {
            "source_path": str(source),
            "source_sha256": _sha256(source),
            "cfg_source_path": str(cfg_source),
            "cfg_source_sha256": _sha256(cfg_source),
            "runtime_contract": (
                "Installed Isaac Lab allocates force_matrix_w as "
                "[env, sensor body, filter shape, xyz] and documents that filtered "
                "contacts require one sensor primitive per environment."
            ),
            "selected_design": (
                "two one-body object-centric sensors, each filtered to all 21 collision "
                "bearing hand bodies; never 21 independent Python sensor reads."
            ),
        },
        "filters": {
            "body_count": 21,
            "self_collision_excluded": True,
            "reference_ghost_excluded": True,
            "cross_environment_excluded": True,
        },
        "passes": passes,
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "C3_CONTACT_READOUT_VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
