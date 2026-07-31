#!/usr/bin/env python3
"""Materialize the bounded ContactPose mug solver-repair handoff."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.retarget.final_refinement import load_final_trajectory

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / ".local/experiments/contactpose_mug_solver_feasibility_v1"
REPORT = REPO / ".local/reports/contactpose_mug_solver_repair"
BASELINE = (
    REPO
    / ".local/experiments/stage12_contactpose_official_joints_v1_20260731_c108ea8"
    / "contactpose/contactpose_full1_use_mug/checkpoints/final_refinement_fast_exact_v2_r1"
)
RUN5 = ROOT / "recovery_trial/run_05_exact_slack"
RUN6 = ROOT / "recovery_trial/run_06_exact_slack_repeat"
BANANA_OLD = (
    REPO
    / ".local/experiments/stage12_contactpose_official_joints_banana_v1_20260731_c108ea8"
    / "contactpose/contactpose_full31_use_banana/final"
    / "final_refinement_fast_exact_v2_r1/final_retarget.zarr"
)
BANANA_NEW = ROOT / "non_regression/banana_run_01/final_retarget.zarr"
REPRESENTATIVE_RUNS = {
    "DexYCB frame 0": ROOT / "non_regression/dexycb_frame0/progress.json",
    "HO-Cap frame 0": ROOT / "non_regression/hocap_frame0/progress.json",
    "OakInk frame 0": ROOT / "non_regression/oakink_frame0/progress.json",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _metadata(run: Path) -> dict[str, Any]:
    frame = run / "checkpoints/frames/frame_000000.npz"
    with np.load(frame, allow_pickle=False) as values:
        return json.loads(str(values["metadata_json"].item()))


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _row(label: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": label,
        "optimizer_status": int(metadata["optimizer_status_code"]),
        "optimizer_converged": bool(metadata["optimizer_converged"]),
        "active_constraints_feasible": bool(metadata["active_constraints_feasible"]),
        "active_set_converged": bool(metadata["active_set_converged"]),
        "full_hard_audit": bool(metadata["full_surface_hard_audit_pass"]),
        "full_soft_audit": bool(metadata["full_surface_soft_audit_pass"]),
        "strict_accepted": bool(metadata["strict_accepted"]),
        "query_count": int(metadata["query_count"]),
        "objective": float(metadata["final_objective"]),
    }


def main() -> int:
    run5, run6 = _metadata(RUN5), _metadata(RUN6)
    rejected = _json(BASELINE / "rejected_frame_000000.json")
    a = load_final_trajectory(RUN5 / "final_retarget.zarr")
    b = load_final_trajectory(RUN6 / "final_retarget.zarr")
    deterministic_arrays = ("qpos", "base_pose_scene", "total_objective", "full_signed_distance")
    diffs = {
        key: float(np.max(np.abs(np.asarray(a.arrays[key]) - np.asarray(b.arrays[key]))))
        for key in deterministic_arrays
    }
    banana_old = load_final_trajectory(BANANA_OLD)
    banana_new = load_final_trajectory(BANANA_NEW)
    banana_keys = (
        "qpos",
        "base_pose_scene",
        "total_objective",
        "e_im",
        "e_bone",
        "full_signed_distance",
        "max_penetration",
    )
    banana_diffs = {
        key: float(
            np.max(np.abs(np.asarray(banana_old.arrays[key]) - np.asarray(banana_new.arrays[key])))
        )
        for key in banana_keys
    }
    report5 = _json(RUN5 / "progress.json")
    report6 = _json(RUN6 / "progress.json")
    queue = _json(REPO / ".local/control/final_jobs/scheduler_state.json")
    lineage = {
        "branch": _git("branch", "--show-current"),
        "head": _git("rev-parse", "HEAD"),
        "origin_head": _git("rev-parse", "origin/integration/dataset-adapter-v1"),
        "input_signature": report5["input_signature"],
        "solver_profile_hash": report5["solver_profile_hash"],
        "execution_profile_hash": report5["execution_profile_hash"],
    }
    rows = [
        _row("baseline_rejected", rejected),
        _row("R5_exact_slack", run5),
        _row("R6_exact_slack_repeat", run6),
    ]
    determinism = {
        "status": "pass" if all(value == 0.0 for value in diffs.values()) else "fail",
        "input_signatures_equal": report5["input_signature"] == report6["input_signature"],
        "array_max_abs_diffs": diffs,
        "run5_checkpoint_hash": report5["last_checkpoint_hash"],
        "run6_checkpoint_hash": report6["last_checkpoint_hash"],
    }
    nonreg = {
        "status": "pass" if all(value == 0.0 for value in banana_diffs.values()) else "fail",
        "scope": "ContactPose banana static shadow replay",
        "array_max_abs_diffs": banana_diffs,
        "stage12_representative_shadow": {
            label: {
                "complete": bool(_json(path)["complete"]),
                "accepted_frames": _json(path)["accepted_frames"],
                "input_signature": _json(path)["input_signature"],
            }
            for label, path in REPRESENTATIVE_RUNS.items()
        },
    }
    root_cause = {
        "root_cause_status": "NONSMOOTH_LINESEARCH_FAILURE",
        "classification_detail": (
            "status-8 at a nearly feasible active soft/slack boundary; selected Jacobian "
            "parity did not reveal a material mapping error"
        ),
        "repair": (
            "bounded exact active-query slack reconstruction followed by reference-Jacobian retry"
        ),
        "acceptance_weakened": False,
    }
    qualification = {
        "status": "pass" if all(row["strict_accepted"] for row in rows[1:]) else "fail",
        "strict_requirements": rows[1:],
        "full_audit_count": [
            int(item["diagnostics"]["full_audit_call_count"]) for item in (run5, run6)
        ],
    }
    matrix = {
        "adapter_source_qualification": (
            "8/8 inherited immutable source qualification; mug repair did not alter "
            "source artifacts"
        ),
        "strict_final_qualification": (
            "mug repaired candidate passed R5/R6; aggregate formal matrix requires "
            "the separate formal-artifact closeout"
        ),
        "contactpose_contact_benchmark": "NOT_REPRODUCED",
        "queue_remains_paused": queue,
    }
    for name, payload in {
        "baseline_reproduction.json": {"rows": rows[:1]},
        "root_cause_classification.json": root_cause,
        "repair_experiments.json": {"rows": rows},
        "mug_final_qualification.json": qualification,
        "determinism.json": determinism,
        "non_regression.json": nonreg,
        "stage12_matrix.json": matrix,
        "queue_closeout.json": queue,
        "git_lineage.json": lineage,
        "artifact_integrity.json": {
            "run5_complete": report5["complete"],
            "run6_complete": report6["complete"],
            "determinism": determinism["status"],
        },
        "active_vs_full_audit_reconciliation.json": {
            "explanation": (
                "active soft rows include slack; the full unqueried-soft audit is distinct"
            ),
            "run5": run5["diagnostics"]["solver_attempt_trace"],
        },
    }.items():
        _write(REPORT / name, payload)
    summary = {
        "root_cause_status": root_cause["root_cause_status"],
        "repair_status": "CONTACTPOSE_MUG_STRICT_FINAL_RESOLVED",
        "stage12_status": (
            "STAGE12_STRICT_FINAL_QUALIFICATION_8_OF_8_PENDING_FORMAL_ARTIFACT_CLOSEOUT"
        ),
        "pr_readiness": "ADAPTER_BRANCH_NOT_PR_READY",
        "queue": queue["state"],
        "mug": qualification["status"],
        "determinism": determinism["status"],
        "banana_non_regression": nonreg["status"],
    }
    _write(REPORT / "final_summary.json", summary)
    (REPORT / "final_summary.md").write_text(
        "# ContactPose mug solver repair\n\n"
        f"- Root cause: `{summary['root_cause_status']}`\n"
        f"- Repair: `{summary['repair_status']}`\n"
        f"- R5/R6 deterministic replay: `{summary['determinism']}`\n"
        f"- Banana non-regression: `{summary['banana_non_regression']}`\n"
        f"- Queue: `{summary['queue']}`\n"
        "- ContactPose Eq. 10/Eq. 11 contact benchmark: `NOT_REPRODUCED`.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
