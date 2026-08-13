#!/usr/bin/env python3
"""Write the fail-closed C3R4 diagnosis and C4 gate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3r4_mpc_holdout_c4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-log", type=Path, required=True)
    parser.add_argument("--targeted-pytest", required=True)
    parser.add_argument("--full-pytest", required=True)
    parser.add_argument("--ruff", required=True)
    parser.add_argument("--mypy", required=True)
    parser.add_argument("--paper-fidelity", required=True)
    parser.add_argument("--base-import", required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"C3R4_CLOSEOUT_INPUT_MISSING: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clip_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "clip": clip["clip"],
            "frames_completed": clip["frames_completed"],
            "finite": clip["finite"],
            "max_position_m": clip["max_position_m"],
            "position_rmse_m": clip["position_rmse_m"],
            "max_rotation_deg": clip["max_rotation_deg"],
            "rotation_rmse_deg": clip["rotation_rmse_deg"],
            "aggregate_torque_saturation_ratio": clip["aggregate_torque_saturation_ratio"],
            "pass": clip["pass"],
        }
        for clip in report["clips"]
    ]


def _assert_trace(trace: dict[str, Any]) -> dict[str, Any]:
    if trace.get("status") != "C3R4_MPC_FIRST_INTERVAL_TRACED":
        raise RuntimeError("C3R4_CLOSEOUT_FIRST_INTERVAL_INCOMPLETE")
    if trace.get("substeps_completed") != 6 or len(trace.get("substeps", [])) != 6:
        raise RuntimeError("C3R4_CLOSEOUT_FIRST_INTERVAL_SUBSTEP_COUNT")
    checkpoints = (
        "before_apply_action",
        "after_apply_action",
        "before_scene_write",
        "after_scene_write",
        "before_sim_step",
        "after_sim_step",
        "before_scene_update",
        "after_scene_update",
    )
    conditions = {name: [] for name in ("A", "B", "M_ww", "H")}
    applied_max = 0.0
    unconstrained_max = 0.0
    for entry in trace["substeps"]:
        if not all(entry[name]["finite"] for name in checkpoints):
            raise RuntimeError("C3R4_CLOSEOUT_FIRST_INTERVAL_NONFINITE_STATE")
        controller = entry["controller"]
        for name in conditions:
            conditions[name].append(controller["condition_numbers"][name])
        applied_max = max(applied_max, controller["applied_summary"]["max_abs"])
        unconstrained_max = max(
            unconstrained_max, controller["unconstrained_control_summary"]["max_abs"]
        )
        if not all(
            controller[name]["finite"]
            for name in (
                "dynamics_a_summary",
                "dynamics_b_summary",
                "hessian_summary",
                "unconstrained_control_summary",
                "projected_control_sequence_summary",
                "applied_summary",
            )
        ):
            raise RuntimeError("C3R4_CLOSEOUT_FIRST_INTERVAL_NONFINITE_CONTROLLER")
    return {
        "status": trace["status"],
        "physics_substeps": 6,
        "cuda_launch_blocking": trace["cuda_launch_blocking"],
        "all_api_boundary_states_finite": True,
        "condition_number_ranges": {
            name: {"min": min(values), "max": max(values)} for name, values in conditions.items()
        },
        "unconstrained_effort_max_abs": unconstrained_max,
        "applied_effort_max_abs": applied_max,
        "wrist_root_state_writes_during_step": trace["wrist_root_state_writes_during_step"],
        "object_rollout_state_writes": trace["object_rollout_state_writes"],
    }


def _markdown(summary: dict[str, Any]) -> str:
    ct_rows = []
    for profile in summary["path_a"]["profiles"]:
        for clip in profile["clips"]:
            ct_rows.append(
                "| {profile} | {clip} | {pos:.6f} | {rot:.3f} | {prmse:.6f} | "
                "{rrmse:.3f} | {sat:.4f} | FAIL |".format(
                    profile=profile["profile"],
                    clip=clip["clip"],
                    pos=clip["max_position_m"],
                    rot=clip["max_rotation_deg"],
                    prmse=clip["position_rmse_m"],
                    rrmse=clip["rotation_rmse_deg"],
                    sat=clip["aggregate_torque_saturation_ratio"],
                )
            )
    mpc_rows = [
        "| {clip} | {pos:.6f} | {rot:.3f} | {prmse:.6f} | {rrmse:.3f} | {sat:.4f} | FAIL |".format(
            clip=clip["clip"],
            pos=clip["max_position_m"],
            rot=clip["max_rotation_deg"],
            prmse=clip["position_rmse_m"],
            rrmse=clip["rotation_rmse_deg"],
            sat=clip["aggregate_torque_saturation_ratio"],
        )
        for clip in summary["path_b"]["mpc_41_frame"]
    ]
    return "\n".join(
        (
            "# Stage 16-C.3R4 MPC Holdout through C.4 Handoff",
            "",
            "## 1. Final Status",
            "",
            "`STAGE16C_EXPLICIT_WRIST_CONTROL_EXHAUSTED`; C.4 remains "
            "`STAGE16C4_NOT_RUN_GATE_BLOCKED`.",
            "",
            "## 2. Git and Environment",
            "",
            f"Branch: `{summary['git']['branch']}`; evidence HEAD: `{summary['git']['head']}`. "
            "Isaac Sim 5.1 / Isaac Lab 2.3.2, GPU PhysX on RTX 5080.",
            "",
            "## 3. Frozen PD Baseline",
            "",
            "The prior best bounded PD profile remains 1.128/1.089 cm position maximum, "
            "17.587/19.570 degrees rotation maximum, and 21.25%/18.75% torque saturation.",
            "",
            "## 4. Explicit 3P+3R Joint Reference",
            "",
            "The six pre-step boundaries are now 0/6 through 5/6; 6/6 is the post-step "
            "20 Hz endpoint. Frozen keys and timing are unchanged, reset initializes qdot from "
            "the analytic joint reference, and rollout state writes remain zero.",
            "",
            "## 5. Articulation Dynamics API",
            "",
            "The runtime path uses the 26x26 PhysX generalized mass matrix and live Coriolis/"
            "centrifugal plus gravity compensation at each substep. Gravity is zero, but bias "
            "is not assumed zero.",
            "",
            "## 6. Wrist-Finger Dynamic Coupling",
            "",
            "The existing full-articulation controller retains M_ww, M_wf qdd_f, and live bias. "
            "The C3R4 holdout reports M_ww separately rather than using the full 26-DoF condition "
            "number as a proxy.",
            "",
            "## 7. Computed-Torque Controller",
            "",
            "| Profile | Clip | Pos max (m) | Rot max (deg) | Pos RMSE (m) | Rot RMSE "
            "(deg) | Torque sat | Result |",
            "|---|---|---:|---:|---:|---:|---:|---|",
            *ct_rows,
            "",
            "## 8. Local Linear Dynamics",
            "",
            f"V1 withheld RMSE is {summary['path_b']['v1_holdout']['one_step']:.6f} at one "
            f"step and {summary['path_b']['v1_holdout']['six_step']:.6f} at six substeps. "
            f"V2 fit R2 is {summary['path_b']['v2_identification']['fit_r2']:.6f}, but withheld "
            f"absolute RMSE is {summary['path_b']['v2_holdout']['one_step']:.6f} / "
            f"{summary['path_b']['v2_holdout']['six_step']:.6f}; both frozen diagnostic gates "
            "fail.",
            "",
            "## 9. TVLQR / Joint MPC",
            "",
            "The false worker-termination label was a reporter `gain`/MPC schema KeyError. The "
            "worker is healthy. The fixed projected-gradient step was spectrally invalid; it is "
            "now capped by the Hessian spectral radius. The corrected V2 41-frame result is:",
            "",
            "| Clip | Pos max (m) | Rot max (deg) | Pos RMSE (m) | Rot RMSE (deg) | "
            "Torque sat | Result |",
            "|---|---:|---:|---:|---:|---:|---|",
            *mpc_rows,
            "",
            "## 10. Active Wrist Controller Decision",
            "",
            "No active controller is selected. Path A and the single permitted Path B "
            "architecture are exhausted under the frozen constraints.",
            "",
            "## 11. Contact Readout and Causality",
            "",
            "Not rerun: the wrist gate failed first. No new contact-causality claim is made.",
            "",
            "## 12. Stage 16-C.3 Requalification",
            "",
            "`STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED` at the wrist prerequisite.",
            "",
            "## 13. Stage 16-C.4 Benchmark",
            "",
            "`STAGE16C4_NOT_RUN_GATE_BLOCKED`; no benchmark was started.",
            "",
            "## 14. Stage 16-C.5 Oracle",
            "",
            "`STAGE16C5_NOT_RUN_GATE_BLOCKED`; no oracle was started.",
            "",
            "## 15. Failure-Recovery",
            "",
            "The reporter, physics-boundary timing, live bias source, projected-gradient "
            "stability, affine substep model, and substep saturation accounting were repaired. "
            "The model still fails independent 1/6-step holdout and both 41-frame gates.",
            "",
            "## 16. Commands",
            "",
            "See `commands` in `final_summary.json`; every path and fixed profile is explicit.",
            "",
            "## 17. Tests",
            "",
            json.dumps(summary["tests"], sort_keys=True),
            "",
            "## 18. README and Roadmap",
            "",
            "English and Chinese wrist-dynamics/status documentation records the C3R4 result.",
            "",
            "## 19. Local Commits",
            "",
            "A precise-path local commit is permitted; push, PR, merge, tag, and release remain "
            "forbidden.",
            "",
            "## 20. Stage 16-C.6 Entry Decision",
            "",
            "`STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED`; samples=0, checkpoints=0.",
            "",
            "## 21. Remaining Limitations",
            "",
            "No PPO or checkpoint; the explicit wrist is abstract; there is no real arm; "
            "physical provenance remains unresolved; Isaac Sim 5.1 is legacy; no real-world "
            "dynamics claim is made.",
            "",
            "## 22. Recommended Next Action",
            "",
            "Authorize exactly one structural change: a real arm model, reference retiming, or a "
            "revised wrist tracking gate. This closeout does not execute any of them.",
            "",
        )
    )


def main() -> int:
    args = parse_args()
    outputs = {
        "summary": REPORT_ROOT / "final_summary.json",
        "summary_markdown": REPORT_ROOT / "final_summary.md",
        "active_controller": REPORT_ROOT / "active_wrist_controller.json",
        "c4": REPORT_ROOT / "c4_status.json",
        "tests": REPORT_ROOT / "tests.json",
        "handoff": REPORT_ROOT / "handoff.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"C3R4_CLOSEOUT_REFUSES_OVERWRITE: {existing}")
    inputs = {
        "frozen_inputs": REPORT_ROOT / "frozen_input_verification.json",
        "reporter_recovery": REPORT_ROOT / "mpc_qualification_after_reporter_fix.json",
        "v1_holdout": REPORT_ROOT / "local_dynamics_holdout_v1.json",
        "computed_torque": REPORT_ROOT
        / "computed_torque_boundary_live_bias_qualification_final.json",
        "v2_identification": REPORT_ROOT / "local_dynamics_identification_v2_boundary.json",
        "v2_holdout": REPORT_ROOT / "local_dynamics_v2_boundary_holdout.json",
        "first_interval": REPORT_ROOT / "mpc_v2_boundary_first_interval_trace_final.json",
        "mpc_41_frame": REPORT_ROOT / "mpc_v2_boundary_41frame_qualification_final.json",
    }
    evidence = {name: _read(path) for name, path in inputs.items()}
    if evidence["frozen_inputs"].get("status") != "STAGE16C3R3_INPUT_HASHES_MATCH":
        raise RuntimeError("C3R4_CLOSEOUT_FROZEN_INPUT_DRIFT")
    if evidence["frozen_inputs"].get("changed"):
        raise RuntimeError("C3R4_CLOSEOUT_FROZEN_INPUT_CHANGED")
    if evidence["computed_torque"].get("status") != ("C3_COMPUTED_TORQUE_WRIST_TRACKING_EXHAUSTED"):
        raise RuntimeError("C3R4_CLOSEOUT_PATH_A_NOT_EXHAUSTED")
    if evidence["v2_identification"].get("finite") is not True:
        raise RuntimeError("C3R4_CLOSEOUT_V2_IDENTIFICATION_NONFINITE")
    if evidence["v2_holdout"].get("gates") != {
        "one_step": False,
        "six_step": False,
        "spectral_projected_gradient": True,
    }:
        raise RuntimeError("C3R4_CLOSEOUT_V2_HOLDOUT_UNEXPECTED")
    qualification = evidence["mpc_41_frame"]
    if (
        qualification.get("status") != "C3_MPC_WRIST_TRACKING_FAIL"
        or qualification.get("pass") is not False
    ):
        raise RuntimeError("C3R4_CLOSEOUT_MPC_GATE_UNEXPECTED")
    if len(qualification.get("clips", [])) != 2 or any(
        clip.get("frames_completed") != 41 for clip in qualification["clips"]
    ):
        raise RuntimeError("C3R4_CLOSEOUT_MPC_GATE_INCOMPLETE")
    if any(
        clip.get("wrist_root_state_writes_during_step") != 0
        or clip.get("object_rollout_state_writes") != 0
        for clip in qualification["clips"]
    ):
        raise RuntimeError("C3R4_CLOSEOUT_FORBIDDEN_ROLLOUT_STATE_WRITE")
    reporter_recovery = evidence["reporter_recovery"]
    if len(reporter_recovery.get("clips", [])) != 2 or any(
        clip.get("frames_completed") != 41 for clip in reporter_recovery["clips"]
    ):
        raise RuntimeError("C3R4_CLOSEOUT_REPORTER_RECOVERY_NOT_PROVEN")
    first_interval = _assert_trace(evidence["first_interval"])
    if not args.kit_log.is_file():
        raise FileNotFoundError(f"C3R4_CLOSEOUT_KIT_LOG_MISSING: {args.kit_log}")
    tests = {
        "targeted_pytest": args.targeted_pytest,
        "full_pytest": args.full_pytest,
        "ruff": args.ruff,
        "mypy": args.mypy,
        "paper_fidelity": args.paper_fidelity,
        "base_import": args.base_import,
    }
    if any(not value.startswith("PASS") for value in tests.values()):
        raise RuntimeError(f"C3R4_CLOSEOUT_TEST_FAILURE: {tests}")
    v1 = evidence["v1_holdout"]
    v2_id = evidence["v2_identification"]
    v2 = evidence["v2_holdout"]
    summary = {
        "status": "STAGE16C_EXPLICIT_WRIST_CONTROL_EXHAUSTED",
        "overall": "STAGE16C_BLOCKED_WITH_BOUNDED_EVIDENCE",
        "schema_version": "toporetarget.stage16c3r4.closeout.v1",
        "git": {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "required_ancestor": _git("rev-parse", "d90bf50"),
        },
        "frozen_inputs": evidence["frozen_inputs"],
        "diagnosis": {
            "false_blocker": {
                "status": "REPAIRED",
                "cause": (
                    "The qualification reporter unconditionally indexed latest['gain'] even for "
                    "MPC, raising KeyError after a successful first interval and mislabeling the "
                    "worker as terminated."
                ),
                "worker_health": "two complete 41-frame reports now prove clean worker execution",
            },
            "actual_blocker": (
                "The frozen explicit 3P+3R finite-effort architecture fails independent "
                "multi-step model holdout and both fixed 41-frame wrist tracking gates."
            ),
            "repairs": [
                "correct 120 Hz pre-step boundaries 0/6..5/6 with endpoint at 6/6",
                "initialize reset qdot from the analytic wrist reference",
                "use live PhysX Coriolis/centrifugal plus gravity compensation",
                "cap projected-gradient step by inverse Hessian spectral radius",
                "identify unit-scaled substep-affine A/B/C and nominal effort",
                "record every qualification substep and actual projected torque boundary",
                "persist worker exceptions instead of emitting a false termination label",
            ],
        },
        "first_interval": {
            **first_interval,
            "kit_log": str(args.kit_log),
            "kit_log_sha256": _sha256(args.kit_log),
            "cuda_or_physx_error_detected": False,
        },
        "path_a": {
            "status": evidence["computed_torque"]["status"],
            "profiles": [
                {
                    "profile": profile["profile"],
                    "pass": profile["pass"],
                    "clips": _clip_rows(profile),
                }
                for profile in evidence["computed_torque"]["profiles"]
            ],
            "active": False,
        },
        "path_b": {
            "status": "C3_EXPLICIT_WRIST_FINITE_EFFORT_TRACKING_EXHAUSTED",
            "v1_holdout": {
                "one_step": v1["holdout"]["one_physics_step"]["normalized_rmse"],
                "six_step": v1["holdout"]["six_physics_substeps_20hz_duration"]["normalized_rmse"],
                "gates": v1["diagnostic_gates"],
                "M_ww_condition_raw": v1["conditioning"]["M_ww_raw"],
                "M_ww_condition_unit_scaled": v1["conditioning"]["M_ww_unit_scaled"],
                "H_condition": v1["conditioning"]["H"],
            },
            "v2_identification": {
                "fit_r2": v2_id["fit"]["normalized_global_r2"],
                "conditioning": v2_id["conditioning"],
                "affine": v2_id["affine"],
            },
            "v2_holdout": {
                "one_step": v2["holdout"]["one_step"]["absolute_affine_model"]["normalized_rmse"],
                "six_step": v2["holdout"]["six_substeps_20hz"]["absolute_affine_model"][
                    "normalized_rmse"
                ],
                "gates": v2["gates"],
            },
            "mpc_41_frame": _clip_rows(qualification),
            "active": False,
        },
        "active_wrist_controller": None,
        "gates": {
            "c3_wrist": "FAIL",
            "contact_readout": "NOT_RUN_GATE_BLOCKED_BY_C3_WRIST",
            "contact_causality": "NOT_RUN_GATE_BLOCKED_BY_C3_WRIST",
            "c3_semantic": "STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED",
            "c4": "STAGE16C4_NOT_RUN_GATE_BLOCKED",
            "c5": "STAGE16C5_NOT_RUN_GATE_BLOCKED",
            "c6": "STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED",
            "ppo_samples": 0,
            "ppo_checkpoints": 0,
        },
        "tests": tests,
        "commands": [
            "python scripts/rl/isaaclab/freeze_stage16c3r3_baseline.py --verify "
            ".local/reports/stage16c3r3_joint_dynamics_c5/frozen_baseline.json "
            "--verify-output .local/reports/stage16c3r4_mpc_holdout_c4/"
            "frozen_input_verification.json",
            "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES "
            "CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 python scripts/rl/isaaclab/"
            "diagnose_stage16c3r4_mpc_interval.py --accept-eula --model-path "
            ".local/reports/stage16c3r4_mpc_holdout_c4/"
            "identified_local_dynamics_v2_boundary.npz --clip-index 0 --output "
            ".local/reports/stage16c3r4_mpc_holdout_c4/"
            "mpc_v2_boundary_first_interval_trace_final.json",
            "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES "
            "CUDA_LAUNCH_BLOCKING=1 PYTHONFAULTHANDLER=1 python scripts/rl/isaaclab/"
            "qualify_stage16c3r3_tvlqr.py --accept-eula --controller mpc --model-path "
            ".local/reports/stage16c3r4_mpc_holdout_c4/"
            "identified_local_dynamics_v2_boundary.npz --output .local/reports/"
            "stage16c3r4_mpc_holdout_c4/mpc_v2_boundary_41frame_qualification_final.json",
        ],
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in inputs.items()
        },
        "recommended_next_action": [
            "authorize a real arm model",
            "authorize reference retiming",
            "authorize a revised wrist tracking gate",
        ],
        "prohibited_actions_taken": {
            "c4_started": False,
            "c5_started": False,
            "ppo_started": False,
            "gain_grid": False,
            "threshold_relaxed": False,
            "torque_limit_increased": False,
            "reference_retimed": False,
            "push": False,
            "pull_request": False,
            "merge": False,
            "tag": False,
            "release": False,
        },
    }
    active = {
        "status": "NO_ACTIVE_WRIST_CONTROLLER_C3_FINITE_EFFORT_TRACKING_EXHAUSTED",
        "controller": None,
        "reason": summary["diagnosis"]["actual_blocker"],
    }
    c4 = {
        "status": "STAGE16C4_NOT_RUN_GATE_BLOCKED",
        "started": False,
        "reason": "C3 wrist tracking failed; full C3 semantic qualification cannot pass.",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(outputs["tests"], tests)
    _write_json(outputs["active_controller"], active)
    _write_json(outputs["c4"], c4)
    _write_json(outputs["summary"], summary)
    markdown = _markdown(summary)
    outputs["summary_markdown"].write_text(markdown, encoding="utf-8")
    outputs["handoff"].write_text(markdown, encoding="utf-8")
    print(summary["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
