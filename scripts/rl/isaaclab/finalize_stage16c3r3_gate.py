#!/usr/bin/env python3
"""Emit immutable C.3R3/C.4/C.5 closeout after the finite-effort gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r3_joint_dynamics_c5"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"C3R3_CLOSEOUT_INPUT_MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _assert_qualification(qualification: dict[str, Any], *, controller: str) -> None:
    if qualification.get("controller") != controller or qualification.get("pass") is not False:
        raise RuntimeError(f"C3R3_CLOSEOUT_UNEXPECTED_{controller}")
    clips = qualification.get("clips", [])
    if len(clips) != 2 or any(clip.get("frames_completed") != 41 for clip in clips):
        raise RuntimeError(f"C3R3_CLOSEOUT_INCOMPLETE_{controller}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "c3r3_gate_summary.json")
    parser.add_argument("--c4-c5-output", type=Path, default=REPORT_ROOT / "c4_c5_not_run.json")
    parser.add_argument("--handoff-output", type=Path, default=REPORT_ROOT / "handoff.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = (args.output, args.c4_c5_output, args.handoff_output)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"C3R3_CLOSEOUT_REFUSES_OVERWRITE: {existing}")

    inputs = {
        "frozen_baseline": REPORT_ROOT / "frozen_baseline.json",
        "joint_reference": REPORT_ROOT / "explicit_joint_reference_rerun.json",
        "dynamics_api": REPORT_ROOT / "dynamics_api_audit_repair1.json",
        "mass_matrix": REPORT_ROOT / "mass_matrix_validation_repair1.json",
        "computed_torque": REPORT_ROOT / "computed_torque_qualification_repair3.json",
        "local_dynamics": REPORT_ROOT / "local_dynamics_identification.json",
        "tvlqr": REPORT_ROOT / "tvlqr_qualification_empirical_v1.json",
        "mpc": REPORT_ROOT / "mpc_qualification_empirical_v1.json",
    }
    artifacts = {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()
    }
    frozen = _json(inputs["frozen_baseline"])
    reference = _json(inputs["joint_reference"])
    dynamics = _json(inputs["dynamics_api"])
    computed = _json(inputs["computed_torque"])
    local_dynamics = _json(inputs["local_dynamics"])
    tvlqr = _json(inputs["tvlqr"])
    mpc = _json(inputs["mpc"])
    if computed.get("status") != "C3_COMPUTED_TORQUE_WRIST_TRACKING_EXHAUSTED":
        raise RuntimeError("C3R3_CLOSEOUT_COMPUTED_TORQUE_NOT_EXHAUSTED")
    _assert_qualification(tvlqr, controller="bounded_tvlqr_wrist_v1")
    if mpc.get("status") != "C3_MPC_WORKER_TERMINATED" or mpc.get("pass") is not False:
        raise RuntimeError("C3R3_CLOSEOUT_MPC_STATUS_UNEXPECTED")
    if dynamics.get("status") != "C3R3_PHYSX_ARTICULATION_DYNAMICS_API_VALIDATED":
        raise RuntimeError("C3R3_CLOSEOUT_DYNAMICS_API_NOT_VALIDATED")
    if not reference.get("keyframes_preserved") or not reference.get("joint_limits_pass"):
        raise RuntimeError("C3R3_CLOSEOUT_REFERENCE_CONTRACT_FAILURE")

    summary = {
        "status": "C3_EXPLICIT_WRIST_FINITE_EFFORT_TRACKING_EXHAUSTED",
        "schema_version": "toporetarget.stage16c3r3.gate_closeout.v1",
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "frozen_inputs": {
            "status": frozen["status"],
            "head": frozen["head"],
            "hashes": frozen["hashes"],
            "hash_drift_status_at_freeze": frozen["hash_drift_status"],
        },
        "reference": reference,
        "dynamics": {
            "status": dynamics["status"],
            "mass_matrix_api": dynamics["runtime"]["mass_matrix"],
            "effort_api": dynamics["runtime"]["effort_command"],
            "actual_effort": dynamics["runtime"]["actual_effort"],
            "mass_matrix": dynamics["mass_matrix"],
        },
        "path_a": {
            "status": computed["status"],
            "profiles": [
                {
                    "profile": entry["profile"],
                    "pass": entry["pass"],
                    "clips": entry["clips"],
                }
                for entry in computed["profiles"]
            ],
            "no_cartesian_wrench": computed["no_cartesian_wrench"],
            "no_rollout_pose_or_velocity_writes": computed["no_rollout_pose_or_velocity_writes"],
        },
        "path_b": {
            "local_dynamics": local_dynamics,
            "tvlqr": {
                "status": "C3_TVLQR_WRIST_TRACKING_FAIL",
                "controller": tvlqr["controller"],
                "clips": tvlqr["clips"],
            },
            "mpc": {
                "status": mpc["status"],
                "failure": mpc["failure"],
            },
        },
        "gates": {
            "c3": "FAIL",
            "c4": "NOT_RUN_BLOCKED_BY_C3",
            "c5": "NOT_RUN_BLOCKED_BY_C3",
            "ppo": "NOT_STARTED_PROHIBITED_BY_C3",
        },
        "artifacts": artifacts,
    }
    blocked = {
        "status": "C4_C5_NOT_RUN_C3_EXPLICIT_WRIST_FINITE_EFFORT_TRACKING_EXHAUSTED",
        "c3_gate_report": str(args.output),
        "c4": {
            "status": "NOT_RUN",
            "reason": "C3 wrist finite-effort tracking gate failed on both frozen clips.",
        },
        "c5": {
            "status": "NOT_RUN",
            "reason": "C3 wrist finite-effort tracking gate failed; PPO is prohibited.",
        },
        "ppo_started": False,
    }
    handoff = "\n".join(
        (
            "# Stage 16-C.3R3 joint-dynamics closeout",
            "",
            "`C3_EXPLICIT_WRIST_FINITE_EFFORT_TRACKING_EXHAUSTED`",
            "",
            "Both frozen clips failed Path A (CT-low and CT-nominal) and the identified "
            "bounded TVLQR Path B; the required box-constrained MPC worker then terminated "
            "safely before a result.  C4/C5 and PPO were not run.",
            "",
            f"- Gate report: `{args.output}`",
            f"- C4/C5 status: `{args.c4_c5_output}`",
            "",
        )
    )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.c4_c5_output.write_text(
        json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.handoff_output.write_text(handoff, encoding="utf-8")
    print(summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
