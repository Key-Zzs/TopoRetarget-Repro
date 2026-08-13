#!/usr/bin/env python3
"""Write non-execution closeout metadata for a completed C.5A-R1 report."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--test-command",
        action="append",
        default=[],
        help="Completed validation command, preserved verbatim; repeat as needed.",
    )
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R1_METADATA_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command(*args: str) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    recent = _command("git", "-C", str(repo_root), "log", "-6", "--format=%H%x1f%s").splitlines()
    return {
        "branch": _command("git", "-C", str(repo_root), "branch", "--show-current"),
        "head": _command("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        "recent_commits": [
            {"commit": row.split("\x1f", 1)[0], "subject": row.split("\x1f", 1)[1]}
            for row in recent
        ],
        "remote_operation": "NOT_RUN",
    }


def _resource_snapshot() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "status": "CLOSEOUT_METADATA_ONLY",
        "capture_scope": "current host snapshot; not a C5A benchmark",
        "nvidia_smi_exit_code": completed.returncode,
        "nvidia_smi_stdout": completed.stdout.strip(),
        "nvidia_smi_stderr": completed.stderr.strip(),
        "O1_C5B_C5C_PPO_resource_measurements": "NOT_RUN_GATE_BLOCKED",
    }


def main() -> int:
    args = parse_args()
    root = args.report_dir.resolve()
    summary_path = root / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("status") != (
        "STAGE16C5A_PHYSICS_CONTRACT_CHANGE_REQUIRED"
    ):
        raise RuntimeError("STAGE16C5A_R1_METADATA_REQUIRES_FAIL_CLOSED_SUMMARY")

    reason = str(summary["reason"])
    _write(
        root / "tests.json",
        {
            "status": "PASS",
            "scope": "post-closeout static, unit, full-suite, and fidelity validation",
            "commands": args.test_command,
            "runtime_gate": summary["natural_baseline"],
            "authorization": summary["authorization"],
        },
    )
    _write(root / "git_commits.json", _git_metadata())
    _write(root / "resource_usage.json", _resource_snapshot())
    transition = {
        "phase": "C5A_R1_CLOSEOUT",
        "classification": reason,
        "status": summary["status"],
        "action": (
            "stop O1/history replay/C5B/C5C/PPO; require separate physics-contract authorization"
        ),
        "evidence": [
            "e1_single_env_same_process.json",
            "e2_vector_same_process.json",
            "e3_env_origin_invariance.json",
            "e4_single_env_cross_process.json",
            "e5_vector_cross_process.json",
            "e6_contact_telemetry_effect.json",
        ],
    }
    _write(root / "c5_failure_transitions.jsonl", json.dumps(transition, sort_keys=True) + "\n")
    _write(
        root / "dashboard.html",
        """<!doctype html>
<meta charset=\"utf-8\">
<title>Stage 16-C.5A-R1 closeout</title>
<h1>Stage 16-C.5A-R1: physics contract change required</h1>
<p>Same-process 33-environment peers diverge after contact under frozen inputs.</p>
<ul>
  <li>E1: exact one-environment same-process replication</li>
  <li>E3: environment-origin normalization valid</li>
  <li>E4/E5: cross-process fingerprints identical</li>
  <li>E6: contact telemetry is read-only</li>
</ul>
<p>O1, C5B, C5C, and PPO are not authorized.</p>
<p>This is a numeric status view, not a visual-success claim.</p>
""",
    )
    print(json.dumps({"status": "STAGE16C5A_R1_METADATA_COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
