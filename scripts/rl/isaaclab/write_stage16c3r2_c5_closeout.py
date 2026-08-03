#!/usr/bin/env python3
"""Write the fail-closed Stage 16-C.3R2--C.5 closeout from immutable reports."""

from __future__ import annotations

import argparse
import json
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
    state.transition(RecoveryStage.CONTACT_READOUT, reason="two-object readout validated")
    state.transition(
        RecoveryStage.FREE_ROOT_FINAL_ATTEMPT,
        reason=(
            "identified inverse-wrench map exceeded frozen condition-number gate at "
            f"{len(_failed_conditions(path_a))} sampled conditions"
        ),
    )
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

    profile_summary = _profile_summary(sources["path_b"])
    recovery = _state_machine(sources["path_a"], sources["path_b"])
    failure_reason = "C3 wrist architecture failed all frozen Path B profiles on both clips"
    downstream_not_run = _not_run(
        status="NOT_RUN_GATE_BLOCKED_BY_C3_WRIST_ARCHITECTURE", reason=failure_reason
    )
    artifacts = {
        "c3_0_reconciled": str(source_paths["c3_0"].relative_to(REPO_ROOT)),
        "contact_readout": str(source_paths["contact"].relative_to(REPO_ROOT)),
        "path_a_wrench_map": str(source_paths["path_a"].relative_to(REPO_ROOT)),
        "d6_tensor_contract": str(source_paths["d6"].relative_to(REPO_ROOT)),
        "path_b_noncontact": str(source_paths["path_b"].relative_to(REPO_ROOT)),
        "d6_wrapper_manifest": str(source_paths["wrapper_manifest"].relative_to(REPO_ROOT)),
        "failure_transitions": str(
            (output_dir / "c3_failure_transitions.jsonl").relative_to(REPO_ROOT)
        ),
    }
    summary = {
        "schema_version": "toporetarget.stage16c3r2_c5.closeout.v2",
        "status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "final_status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
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
        "c2_regression": _not_run(
            status="STAGE16C2_D6_PROFILE_REGRESSION_BLOCKED",
            reason="No Path B profile passed; there is no active D6 profile to regress.",
        ),
        "c3": {
            "status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
            "c3_0": sources["c3_0"]["status"],
            "modes_1_through_5": downstream_not_run,
            "contact_momentum_causality": downstream_not_run,
        },
        "c4_gpu_benchmark": downstream_not_run,
        "c5_state_replication": downstream_not_run,
        "c5_physx_oracle": downstream_not_run,
        "c6_ppo": _not_run(status="NOT_AUTHORIZED", reason="C.5 PhysX oracle was not run."),
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
    c3_qualification = {
        "status": "C3_WRIST_ACTUATION_ARCHITECTURE_BLOCKED",
        "mode_0": sources["c3_0"]["status"],
        "modes_1_through_5": downstream_not_run,
        "reason": failure_reason,
    }
    c4 = {"status": downstream_not_run["status"], "reason": downstream_not_run["reason"]}
    c5 = {"status": downstream_not_run["status"], "reason": downstream_not_run["reason"]}
    active_profile = {
        "status": "NO_ACTIVE_WRIST_PROFILE",
        "reason": "No frozen finite virtual profile passed both clips.",
        "selected_profile": None,
    }
    contact_causality = {
        "status": "C3_CONTACT_MOMENTUM_CAUSALITY_NOT_RUN_GATE_BLOCKED",
        "reason": failure_reason,
        "readout_prerequisite": sources["contact"]["status"],
    }
    for name, value in {
        "final_summary.json": summary,
        "wrist_architecture_decision.json": decision,
        "d6_wrist_manifest.json": sources["wrapper_manifest"],
        "active_wrist_profile.json": active_profile,
        "c2_d6_profile_regression.json": summary["c2_regression"],
        "c3_full_qualification.json": c3_qualification,
        "c3_contact_momentum_causality.json": contact_causality,
        "c4_vector_benchmark.json": c4,
        "c5_state_replication.json": c5,
        "c5_physx_oracle.json": c5,
    }.items():
        _write_json(output_dir / name, value)
    transitions_path = output_dir / "c3_failure_transitions.jsonl"
    transitions_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in recovery["transitions"]),
        encoding="utf-8",
    )
    (output_dir / "final_summary.md").write_text(
        _markdown(summary, profile_summary), encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
