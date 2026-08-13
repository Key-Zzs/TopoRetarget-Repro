#!/usr/bin/env python3
"""Aggregate the bounded E4 or E5 child-process reproducibility diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER = REPO_ROOT / "scripts/rl/isaaclab/diagnose_stage16c5_natural_nondeterminism.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--mode", choices=("single", "vector"), required=True)
    parser.add_argument(
        "--telemetry", choices=("off", "aggregate", "diagnostic"), default="aggregate"
    )
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--workers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_CROSS_PROCESS_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("result") != "DIAGNOSTIC_COMPLETE":
        raise ValueError(f"malformed C5A cross-process worker report: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError(f"C5A cross-process worker lacks 2x4 phase rows: {path}")
    if payload.get("trials") != 1 or payload.get("process_mode") != "cross_process_worker":
        raise ValueError(f"C5A cross-process worker mode mismatch: {path}")
    return payload


def summarize_workers(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare byte-level physics and derived-state fingerprints per phase."""

    if len(reports) != 20:
        raise ValueError("E4/E5 require exactly 20 independent child reports")
    reference_rows = reports[0]["rows"]
    assert isinstance(reference_rows, list)
    summary: list[dict[str, Any]] = []
    for row_index, reference_row in enumerate(reference_rows):
        assert isinstance(reference_row, dict)
        raw_hashes: list[str] = []
        derived_hashes: list[str] = []
        for report in reports:
            rows = report["rows"]
            assert isinstance(rows, list)
            row = rows[row_index]
            if not isinstance(row, dict):
                raise ValueError("C5A cross-process row is not an object")
            if (
                row.get("clip") != reference_row.get("clip")
                or row.get("phase") != reference_row.get("phase")
                or row.get("frame") != reference_row.get("frame")
            ):
                raise ValueError("C5A cross-process phase alignment mismatch")
            fingerprints = row.get("measurement_fingerprints")
            if not isinstance(fingerprints, list) or len(fingerprints) != 1:
                raise ValueError("C5A cross-process worker lacks one measurement fingerprint")
            fingerprint = fingerprints[0]
            if not isinstance(fingerprint, dict):
                raise ValueError("C5A cross-process fingerprint is malformed")
            raw = fingerprint.get("raw")
            derived = fingerprint.get("derived")
            if not isinstance(raw, str) or not isinstance(derived, str):
                raise ValueError("C5A cross-process fingerprint keys are malformed")
            raw_hashes.append(raw)
            derived_hashes.append(derived)
        summary.append(
            {
                "clip": reference_row["clip"],
                "phase": reference_row["phase"],
                "frame": reference_row["frame"],
                "raw_fingerprint_identical": len(set(raw_hashes)) == 1,
                "derived_fingerprint_identical": len(set(derived_hashes)) == 1,
                "raw_unique_fingerprint_count": len(set(raw_hashes)),
                "derived_unique_fingerprint_count": len(set(derived_hashes)),
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    if not args.accept_eula or args.trials != 20:
        raise SystemExit("E4/E5 require --accept-eula and exactly 20 child processes")
    if args.output.exists() or args.workers_dir.exists():
        raise FileExistsError("STAGE16C5A_CROSS_PROCESS_REFUSES_OVERWRITE")
    args.workers_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    report_paths: list[Path] = []
    for worker_index in range(args.trials):
        report_path = args.workers_dir / f"worker_{worker_index:02d}.json"
        log_path = args.workers_dir / f"worker_{worker_index:02d}.log"
        command = [
            sys.executable,
            str(WORKER),
            "--accept-eula",
            "--cross-process-worker",
            "--mode",
            args.mode,
            "--telemetry",
            args.telemetry,
            "--trials",
            "1",
            "--frames",
            str(args.frames.resolve()),
            "--output",
            str(report_path),
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "STAGE16C5A_CROSS_PROCESS_WORKER_FAILED: "
                f"worker={worker_index} returncode={completed.returncode} log={log_path}"
            )
        report_paths.append(report_path)
    reports = [_load(path) for path in report_paths]
    phase_summary = summarize_workers(reports)
    all_raw_identical = all(row["raw_fingerprint_identical"] for row in phase_summary)
    all_derived_identical = all(row["derived_fingerprint_identical"] for row in phase_summary)
    report = {
        "schema_version": "stage16c5_cross_process_diagnostic_v1",
        "mode": args.mode,
        "telemetry": args.telemetry,
        "process_mode": "20_independent_child_processes",
        "formal_gate_role": "diagnostic_only_not_same_process_candidate_gate",
        "trials": args.trials,
        "phase_summary": phase_summary,
        "result": (
            "CROSS_PROCESS_FINGERPRINT_IDENTICAL"
            if all_raw_identical and all_derived_identical
            else "CROSS_PROCESS_FINGERPRINT_DIFFERENT"
        ),
    }
    _write(args.output, report)
    print(json.dumps({"result": report["result"], "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
