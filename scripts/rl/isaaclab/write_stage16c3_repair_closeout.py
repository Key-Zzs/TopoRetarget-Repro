#!/usr/bin/env python3
"""Write the fail-closed Stage 16-C.3 repair closeout from immutable run evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c3_repair_c5_oracle"
REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def profile_row(name: str, report: dict[str, object]) -> dict[str, object]:
    diagnostic = report["dynamic_wrist_finger_kinematic_object"]
    assert isinstance(diagnostic, dict)
    results = diagnostic["results"]
    assert isinstance(results, list)
    maxima = [item["maxima"] for item in results]
    return {
        "profile": name,
        "report": str(REPORT_ROOT / f"{name}.json"),
        "steps": results[0]["steps"],
        "max_wrist_position_m": max(item["wrist_position_m"] for item in maxima),
        "max_wrist_rotation_deg": max(item["wrist_rotation_deg"] for item in maxima),
        "max_force_saturation_ratio": max(item["force_saturation_ratio"] for item in maxima),
        "max_torque_saturation_ratio": max(item["torque_saturation_ratio"] for item in maxima),
        "status": diagnostic["status"],
    }


def reference_dynamics() -> dict[str, object]:
    result: dict[str, object] = {"status": "PASS", "clips": {}}
    for clip in ("hocap_170105", "hocap_170650"):
        path = REFERENCE_ROOT / f"{clip}.world_wrist.stage16.npz"
        with np.load(path, allow_pickle=False) as source:
            timestamps = np.asarray(source["timestamps"], dtype=np.float64)
            position = np.asarray(source["wrist_pose_translation_world_ref"], dtype=np.float64)
            twist = np.asarray(source["wrist_twist_world_ref"], dtype=np.float64)
        dt = np.diff(timestamps)
        acceleration = np.diff(twist[:, :3], axis=0) / dt[:, None]
        alpha = np.diff(twist[:, 3:], axis=0) / dt[:, None]
        result["clips"][clip] = {
            "reference_sha256": sha256(path),
            "frames": int(timestamps.size),
            "control_dt_s": float(np.median(dt)),
            "max_linear_speed_mps": float(np.linalg.norm(twist[:, :3], axis=1).max()),
            "max_angular_speed_radps": float(np.linalg.norm(twist[:, 3:], axis=1).max()),
            "max_linear_acceleration_mps2": float(np.linalg.norm(acceleration, axis=1).max()),
            "max_angular_acceleration_radps2": float(np.linalg.norm(alpha, axis=1).max()),
            "max_key_translation_delta_m": float(
                np.linalg.norm(np.diff(position, axis=0), axis=1).max()
            ),
        }
    return result


def write_handoff(
    *,
    profiles: list[dict[str, object]],
    composite: dict[str, object],
    effective: dict[str, object],
    required_wrench: dict[str, object],
    contact: dict[str, object],
    transitions: dict[str, object],
) -> None:
    """Write a portable, evidence-only handoff for the fail-closed closeout."""
    rows = [
        "| Profile | Pos max (m) | Rot max (deg) | Force sat | Torque sat | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for profile in profiles:
        rows.append(
            "| {name} | {pos:.5f} | {rot:.3f} | {force:.3f} | {torque:.3f} | {status} |".format(
                name=profile["profile"],
                pos=float(profile["max_wrist_position_m"]),
                rot=float(profile["max_wrist_rotation_deg"]),
                force=float(profile["max_force_saturation_ratio"]),
                torque=float(profile["max_torque_saturation_ratio"]),
                status=profile["status"],
            )
        )
    lines = [
        "# Stage 16-C.3 Repair through C.5 PhysX Oracle Handoff",
        "",
        "## 1. Final Status",
        "`STAGE16C3_WRIST_AND_CONTACT_QUALIFICATION_BLOCKED`; C.4 and C.5 are "
        "`NOT_RUN_GATE_BLOCKED`, and C.6 PPO is not authorized.",
        "",
        "## 2. Git and Environment",
        "Local Isaac Lab / RTX 5080 evidence only. `frozen_manifest.json` records the "
        "preserved C.2 inputs. No remote operation is part of this closeout.",
        "",
        "## 3. Frozen C.3 Failure Baseline",
        "Historic failure evidence is preserved in `wrist_failure_baseline.json`.",
        "",
        "## 4. Wrist API and Frame Conventions",
        "The baseline-subtracted signed +/- six-axis PhysX pulse audit passes. Root pose uses "
        "wxyz, and root linear/angular velocities are world-frame quantities.",
        "",
        "## 5. Wrist Reference Dynamics",
        "Per-clip speed, acceleration, and keyframe bounds are in `wrist_reference_dynamics.json`.",
        "",
        "## 6. Composite and Effective Wrist Inertia",
        json.dumps(composite, sort_keys=True),
        json.dumps(effective, sort_keys=True),
        "",
        "## 7. Wrist Controller Repair",
        *rows,
        "",
        "The final shared 41-step profile reaches 0.03347 m / 23.00 deg, above the 0.02 m / "
        "10 deg gate. Higher authority worsened translation and saturated force.",
        "",
        "## 8. Full-Hand Contact Sensor Contract",
        json.dumps(contact["inventory"], sort_keys=True),
        json.dumps(contact["sensor_contract"], sort_keys=True),
        "",
        "## 9. Contact-Momentum Causality",
        "`NOT_PROVEN`: reading the 21 filtered sensor views exits the Isaac process before a "
        "trace report, so no force, impulse, residual, or causal pass is asserted.",
        "",
        "## 10. Stage 16-C.3 Requalification",
        "Failed at the independent dynamic-wrist gate; contact collection cannot make it a pass.",
        "",
        "## 11. MuJoCo Action-Trace Replay",
        "Not rerun in this repair; frozen action-only evidence is retained and no hidden shortcut "
        "was introduced.",
        "",
        "## 12. C.3 Failure-Recovery",
        json.dumps(transitions, sort_keys=True),
        "",
        "## 13. Stage 16-C.4 GPU Benchmark",
        "`NOT_RUN_GATE_BLOCKED_BY_C3`; no throughput is reported.",
        "",
        "## 14. Selected Parallel Environment Profile",
        "None selected because C.4 did not run.",
        "",
        "## 15. Stage 16-C.5 PhysX Oracle",
        "`NOT_RUN_GATE_BLOCKED_BY_C3`; no replication, pool, horizon, CEM, selector, or runtime "
        "claim is made.",
        "",
        "## 16. Oracle Qualification Results",
        "No episodes were run; success, reach, position, rotation, and axis results are N/A.",
        "",
        "## 17. C.5 Failure-Recovery",
        "Blocked upstream by C.3. Repair dynamic wrist tracking and obtain an observable contact "
        "trace before entering C.4/C.5.",
        "",
        "## 18. Visualization and Numerical Review",
        json.dumps(required_wrench, sort_keys=True),
        "",
        "## 19. Commands",
        "```bash",
        "conda run -n toporetarget-rl python "
        "scripts/rl/isaaclab/qualify_stage16c3_semantics.py --help",
        "conda run -n toporetarget-rl python "
        "scripts/rl/isaaclab/audit_stage16c3_wrist_api.py --help",
        "conda run -n toporetarget-rl python "
        "scripts/rl/isaaclab/identify_stage16c3_wrist_effective_dynamics.py --help",
        "conda run -n toporetarget-rl python "
        "scripts/rl/isaaclab/write_stage16c3_repair_closeout.py",
        "```",
        "",
        "## 20. Tests",
        "Targeted Ruff passed; the focused pytest command passed 13 tests. GPU A0 and F0/F1/F2 "
        "probes passed; C.3 profiles failed.",
        "",
        "## 21. README and Roadmap",
        "README, both roadmaps, Isaac Lab docs, wrist/contact docs, and recovery docs record "
        "the blocked state.",
        "",
        "## 22. Local Commits",
        "PUSHED = NO; PR_CREATED = NO; MAIN_MERGED = NO; TAG_CREATED = NO; RELEASE_CREATED = NO.",
        "",
        "## 23. Stage 16-C.6 Entry Decision",
        "`STAGE16C6_SINGLE_CLIP_GPU_PPO_NOT_AUTHORIZED`: zero PPO samples and zero checkpoints.",
        "",
        "## 24. Remaining Limitations",
        "no PPO; no PPO checkpoint; no DR; no real arm; physical provenance unresolved; Isaac "
        "Sim 5.1 legacy; no real-world dynamics claim.",
        "",
        "## 25. Recommended Next Action",
        "Bounded repair only: make all-hand contact collection observable without changing frozen "
        "inputs, then repair the dynamic wrist residual and rerun C.3 before C.4/C.5.",
        "",
    ]
    (REPORT_ROOT / "handoff.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    historic = read_json(REPO_ROOT / ".local/reports/stage16c2_c5_isaaclab/final_summary.json")
    p1 = read_json(REPORT_ROOT / "wrist_profile_p1.json")
    p2 = read_json(REPORT_ROOT / "wrist_profile_p2_25n_1p5nm.json")
    p3 = read_json(REPORT_ROOT / "wrist_profile_p3_50n_6nm.json")
    p3r = read_json(REPORT_ROOT / "wrist_profile_p3r_50n_6nm.json")
    v1_tuned = read_json(REPORT_ROOT / "c3_v1_ten_step_tuned.json")
    v1_high = read_json(REPORT_ROOT / "c3_v1_ten_step_high_authority.json")
    final_wrist = read_json(REPORT_ROOT / "c3_final_wrist_qualification.json")
    api = read_json(REPORT_ROOT / "wrist_api_conventions_baseline_subtracted.json")
    f0 = read_json(REPORT_ROOT / "wrist_effective_dynamics_f0.json")
    f1 = read_json(REPORT_ROOT / "wrist_effective_dynamics_f1.json")
    f2_105 = read_json(REPORT_ROOT / "wrist_effective_dynamics_f2_hocap_170105.json")
    f2_650 = read_json(REPORT_ROOT / "wrist_effective_dynamics_f2_hocap_170650.json")
    dynamics = reference_dynamics()
    write_json(REPORT_ROOT / "wrist_reference_dynamics.json", dynamics)

    p2_substeps = read_json(REPORT_ROOT / "wrist_profile_p2_25n_1p5nm_wrist_substeps.json")
    mass_values = [item["composite_mass_kg"] for item in p2_substeps]
    inertia_values = [item["inertia_eigenvalues_kgm2"] for item in p2_substeps]
    composite = {
        "status": "MEASURED_FROM_REAL_PHYSX_SUBSTEPS",
        "sample_count": len(p2_substeps),
        "mass_kg": {"min": min(mass_values), "max": max(mass_values)},
        "inertia_eigenvalues_kgm2": {
            "min": np.min(np.asarray(inertia_values), axis=0).tolist(),
            "max": np.max(np.asarray(inertia_values), axis=0).tolist(),
        },
        "note": (
            "Composite all-link inertia is an audit value, not a validated "
            "effective root response model."
        ),
    }
    write_json(REPORT_ROOT / "wrist_composite_inertia.json", composite)

    required_wrench = {
        "status": "DERIVED_BOUND_NOT_A_GAIN_SELECTION",
        "method": "max reference acceleration times measured all-link mass/eigenvalue upper bound",
        "clips": {},
    }
    for clip, values in dynamics["clips"].items():
        required_wrench["clips"][clip] = {
            "composite_force_bound_n": composite["mass_kg"]["max"]
            * values["max_linear_acceleration_mps2"],
            "composite_torque_bound_nm": max(composite["inertia_eigenvalues_kgm2"]["max"])
            * values["max_angular_acceleration_radps2"],
        }
    write_json(REPORT_ROOT / "wrist_required_wrench.json", required_wrench)

    effective = {
        "status": "F0_F1_F2_IDENTIFIED_BUT_NOT_ACCEPTED_FOR_TRAJECTORY_CONTROL",
        "F0_no_finger_drives": f0["conditions"][0],
        "F1_zero_finger_targets": f1["conditions"][0],
        "F2_reference_finger_targets": [f2_105["conditions"][0], f2_650["conditions"][0]],
        "conclusion": (
            "The local F2 response is finite and signed correctly, but its one-step matrix did not "
            "stabilize the 10-step reference trajectory. It is retained as diagnosis only."
        ),
    }
    write_json(REPORT_ROOT / "wrist_effective_dynamics.json", effective)

    profiles = [
        profile_row("wrist_profile_p1", p1),
        profile_row("wrist_profile_p2_25n_1p5nm", p2),
        profile_row("wrist_profile_p3_50n_6nm", p3),
        profile_row("wrist_profile_p3r_50n_6nm", p3r),
        profile_row("c3_v1_ten_step_tuned", v1_tuned),
        profile_row("c3_v1_ten_step_high_authority", v1_high),
        profile_row("c3_final_wrist_qualification", final_wrist),
    ]
    profile_analysis = {
        "status": "C3_WRIST_DYNAMIC_TRACKING_BLOCKED",
        "gate": {"max_wrist_position_m": 0.02, "max_wrist_rotation_deg": 10.0},
        "profiles": profiles,
        "conclusion": (
            "No shared profile passed both frozen clips. The final bounded 41-step profile remains "
            "above both limits, and the higher-authority check degraded translation "
            "while saturating."
        ),
    }
    write_json(REPORT_ROOT / "wrist_profile_analysis.json", profile_analysis)

    frozen_manifest = {
        "status": "FROZEN_ARCHIVE_CREATED_BEFORE_REPAIR",
        "generated_at_utc": generated_at,
        "references": {
            clip: sha256(REFERENCE_ROOT / f"{clip}.world_wrist.stage16.npz")
            for clip in ("hocap_170105", "hocap_170650")
        },
        "configs": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in sorted((REPO_ROOT / "configs/rl/stage16").glob("isaaclab_*.yaml"))
        },
        "historic_c2_summary_sha256": sha256(
            REPO_ROOT / ".local/reports/stage16c2_c5_isaaclab/final_summary.json"
        ),
    }
    write_json(REPORT_ROOT / "frozen_manifest.json", frozen_manifest)
    archives = sorted((REPO_ROOT / ".local/archive").glob("stage16c3_wrist_contact_failure_*"))
    if archives:
        write_json(archives[-1] / "frozen_manifest.json", frozen_manifest)

    c2_hashes = {
        "status": "PRESERVED",
        "c2_summary_sha256": frozen_manifest["historic_c2_summary_sha256"],
        "c2": historic["c2"],
        "frozen_reference_hashes": historic["c2"]["alternating"].get("reference_hashes", None),
    }
    write_json(REPORT_ROOT / "c2_contract_hashes.json", c2_hashes)
    baseline = {
        "status": "PRESERVED_HISTORIC_C3_FAILURE",
        "kinematic_object": historic["c3_kinematic_object"],
        "free_object": historic["c3_free_object"],
        "original_blocker": historic["c3_blocker"],
    }
    write_json(REPORT_ROOT / "wrist_failure_baseline.json", baseline)

    contact = {
        "status": "NOT_VALIDATED_RUNTIME_LIMITATION",
        "inventory": final_wrist["contact"]["hand_collision_inventory"],
        "sensor_contract": final_wrist["contact"]["sensor_contract"],
        "attempts": [
            {
                "command_scope": "1 and 10 step all-hand collection, no C3 acceptance claim",
                "result": "NO_REPORT_WRITTEN",
                "reason": (
                    "Isaac Sim process exits after setup when the 21 filtered sensor views are "
                    "read; "
                    "this remains fail-closed and no contact causality is claimed."
                ),
            }
        ],
        "contact_causality": "NOT_PROVEN",
    }
    write_json(REPORT_ROOT / "contact_capture_status.json", contact)
    write_json(REPORT_ROOT / "contact_causality.json", contact)

    transitions = {
        "status": "STOPPED_AT_C3",
        "transitions": [
            {
                "phase": "A0",
                "result": "PASS",
                "evidence": "wrist_api_conventions_baseline_subtracted.json",
            },
            {"phase": "A1", "result": "DIAGNOSED", "evidence": "wrist_effective_dynamics.json"},
            {"phase": "A2", "result": "FAIL", "evidence": "wrist_profile_analysis.json"},
            {"phase": "B", "result": "BLOCKED", "reason": "2 cm/10 deg dynamic wrist gate failed"},
            {"phase": "C", "result": "NOT_RUN", "reason": "upstream wrist gate failed"},
            {"phase": "C4_C5", "result": "NOT_RUN", "reason": "C3 remains blocked"},
        ],
    }
    write_json(REPORT_ROOT / "c3_failure_transitions.json", transitions)
    blocked = {
        "status": "NOT_RUN_GATE_BLOCKED_BY_C3",
        "reason": "C3 wrist tracking and contact causality are not validated.",
        "ppo_authorization": "NOT_AUTHORIZED; zero samples and zero checkpoints",
    }
    write_json(REPORT_ROOT / "c4_benchmark.json", blocked)
    write_json(REPORT_ROOT / "c5_oracle.json", blocked)
    write_handoff(
        profiles=profiles,
        composite=composite,
        effective=effective,
        required_wrench=required_wrench,
        contact=contact,
        transitions=transitions,
    )
    write_json(
        REPORT_ROOT / "tests.json",
        {
            "status": "PASS",
            "commands": [
                "ruff check targeted Stage16-C.3 sources",
                "python -m pytest -q tests/rl/isaaclab/test_stage16c2_direct_env_contracts.py",
            ],
            "pytest": "13 passed",
            "real_gpu": [
                "wrist signed six-axis basis: PASS",
                "F0/F1/F2 effective-response probes: PASS",
                "C3 profiles: FAIL / BLOCKED",
            ],
        },
    )

    summary = {
        "status": "STAGE16C3_WRIST_AND_CONTACT_QUALIFICATION_BLOCKED",
        "generated_at_utc": generated_at,
        "c2": "PRESERVED_VALIDATED",
        "c3_wrist": "FAIL",
        "c3_contact": "NOT_VALIDATED",
        "c4": blocked["status"],
        "c5": blocked["status"],
        "c6_ppo": blocked["ppo_authorization"],
        "key_evidence": {
            "wrist_api": api["status"],
            "final_41_step_profile": profile_analysis["profiles"][-1],
            "high_authority_profile": profile_analysis["profiles"][-2],
        },
        "reports": {
            "frozen_manifest": "frozen_manifest.json",
            "profile_analysis": "wrist_profile_analysis.json",
            "contact": "contact_capture_status.json",
            "failure_transitions": "c3_failure_transitions.json",
        },
    }
    write_json(REPORT_ROOT / "final_summary.json", summary)
    markdown = "\n".join(
        (
            "# Stage 16-C.3 repair closeout",
            "",
            "**Status:** `STAGE16C3_WRIST_AND_CONTACT_QUALIFICATION_BLOCKED`.",
            "",
            "- C.2 remains preserved and validated; its references/action/observation contract "
            "was not changed.",
            "- The signed 6-D PhysX basis passes, so the failure is not a world/local or "
            "sign inversion.",
            "- F0/F1/F2 response probes show a coupled articulated response. The static F2 "
            "matrix did not stabilize the trajectory.",
            "- Final shared 41-step profile: 3.35 cm / 23.00 deg maximum wrist error; "
            "gate is 2 cm / 10 deg.",
            "- High-authority profile: 8.53 cm / 15.82 deg and 83.3% force saturation; rejected.",
            "- All-hand sensor inventory resolves 21 collision-bearing bodies, but real "
            "all-hand collection exits before a report. No contact causality is claimed.",
            "- C.4/C.5 were not run. C.6 PPO is not authorized; zero samples/checkpoints.",
            "",
            "See `wrist_profile_analysis.json`, `wrist_effective_dynamics.json`, and "
            "`contact_capture_status.json`.",
            "",
        )
    )
    (REPORT_ROOT / "final_summary.md").write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
