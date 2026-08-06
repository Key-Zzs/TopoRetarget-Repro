#!/usr/bin/env python3
"""Write the bounded D.4R3 calibration-blocked closeout and downstream stop ledger."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_stable_grasp_geometry_ppo"
OBJECT_IDS = ("hocap_170105", "hocap_170650")
CALIBRATION_BLOCKER = "STAGE16D_STABLE_FREE_OBJECT_GRASP_CALIBRATION_BLOCKED"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--tests-status", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--test-summary", required=True)
    return parser


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _not_run(stage: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "Stage16DStableGraspDownstreamStopV1",
        "stage": stage,
        "status": f"STAGE16D_{stage.upper()}_NOT_RUN_GATE_BLOCKED",
        "reason": reason,
        "runs": 0,
        "samples": 0,
        "checkpoints": [],
        "unauthorized_work_started": False,
    }


def _candidate_row(path: Path) -> dict[str, Any]:
    report = _load(path)
    metrics = report.get("selection_metrics", {})
    candidate = report["candidate"]
    geometry = report.get("formal_geometry", {})
    return {
        "object_id": report["object_id"],
        "level": candidate["level"],
        "family_id": candidate["family_id"],
        "candidate_id": candidate["candidate_id"],
        "approach_offset_m": candidate["approach_offset_m"],
        "closure_amplitude": candidate["closure_amplitude"],
        "contact_groups": candidate["contact_groups"],
        "replicas": report["replicas"],
        "steps": report["steps"],
        "initialization_static_pass": report.get("initialization_static_pass"),
        "contact_persistence": metrics.get("contact_persistence"),
        "topology_coverage": metrics.get("topology_coverage"),
        "terminal_hold_stability": metrics.get("terminal_hold_stability"),
        "terminal_linear_speed_p95_max_mps": metrics.get("terminal_linear_speed_p95_max_mps"),
        "terminal_angular_speed_p95_max_radps": metrics.get("terminal_angular_speed_p95_max_radps"),
        "max_penetration_m": metrics.get("max_penetration_m"),
        "active_p95_penetration_m": metrics.get("active_p95_penetration_m"),
        "geometry_backend": geometry.get("backend"),
        "development_pass": report.get("development_pass", False),
        "status": report["status"],
        "exact_audit_failure": report.get("exact_audit_failure"),
        "report": str(path.relative_to(REPO_ROOT)),
        "report_sha256": _sha256(path),
    }


def _validate_and_collect(output: Path) -> list[dict[str, Any]]:
    expected: set[tuple[str, str]] = set()
    for level in ("c1", "c2"):
        matrix = _load(output / f"stable_grasp_candidate_matrix_{level}.json")
        for object_id, candidates in matrix["objects"].items():
            expected.update((object_id, row["candidate_id"]) for row in candidates)
    if len(expected) != 20:
        raise RuntimeError(f"STAGE16D_CALIBRATION_MATRIX_CARDINALITY_DRIFT:{len(expected)}")
    rows: list[dict[str, Any]] = []
    for object_id, candidate_id in sorted(expected):
        path = output / f"calibration_dev_{object_id}_{candidate_id}.json"
        if not path.is_file():
            raise RuntimeError(f"STAGE16D_CALIBRATION_RESULT_MISSING:{path}")
        row = _candidate_row(path)
        if row["object_id"] != object_id or row["candidate_id"] != candidate_id:
            raise RuntimeError("STAGE16D_CALIBRATION_RESULT_IDENTITY_FAILURE")
        rows.append(row)
    if any(row["development_pass"] for row in rows):
        raise RuntimeError("STAGE16D_CLOSEOUT_REFUSES_BLOCKED_STATUS_WITH_ELIGIBLE_CANDIDATE")
    if any(row["status"].endswith("EXACT_PENDING") for row in rows):
        raise RuntimeError("STAGE16D_CLOSEOUT_REFUSES_PENDING_EXACT_AUDIT")
    return rows


def _best_diagnostic(rows: list[dict[str, Any]], object_id: str) -> dict[str, Any]:
    candidates = [row for row in rows if row["object_id"] == object_id]

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            -float(row["terminal_hold_stability"] or 0.0),
            -float(row["topology_coverage"] or 0.0),
            -float(row["contact_persistence"] or 0.0),
            float(row["terminal_angular_speed_p95_max_radps"] or float("inf")),
            row["candidate_id"],
        )

    return min(candidates, key=key)


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    rows = _validate_and_collect(output)
    generated = datetime.now(UTC).isoformat()
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    frozen = _load(output / "frozen_inputs.json")
    topology = _load(output / "grasp_topology_family_contract.json")
    _load(output / "calibration_action_schedule.json")
    trajectory_baseline = _load(output / "trajectory_baseline.json")
    exact_failures = [row for row in rows if row["exact_audit_failure"] is not None]
    completed_exact = [row for row in rows if row["geometry_backend"] is not None]
    best = {object_id: _best_diagnostic(rows, object_id) for object_id in OBJECT_IDS}

    for object_id in OBJECT_IDS:
        short = object_id.removeprefix("hocap_")
        object_rows = [row for row in rows if row["object_id"] == object_id]
        _write(
            output / f"calibration_results_{short}.json",
            {
                "schema_version": "StableGraspCalibrationObjectResultsV1",
                "object_id": object_id,
                "status": "STAGE16D_STABLE_GRASP_CALIBRATION_BLOCKED",
                "development_candidates": object_rows,
                "development_candidate_count": len(object_rows),
                "eligible_candidate_count": 0,
                "formal20_runs": 0,
                "best_diagnostic_only": best[object_id],
            },
        )

    qualification = {
        "schema_version": "StableGraspCalibrationQualificationSetV1",
        "status": "STAGE16D_STABLE_GRASP_CALIBRATION_BLOCKED",
        "stop_marker": CALIBRATION_BLOCKER,
        "reason": (
            "No frozen C1 or unique C2 development candidate passed contact, topology, "
            "terminal hold, terminal twist, and exact geometry gates together."
        ),
        "development_candidates": len(rows),
        "development_replicas_per_candidate": 4,
        "development_control_steps_per_replica": 321,
        "eligible_candidates": 0,
        "formal20_runs": 0,
        "formal20_authorized": False,
        "c1_selection": _load(output / "stable_grasp_selection_c1.json"),
        "c2_selection": _load(output / "stable_grasp_selection_c2.json"),
        "best_diagnostic_only": best,
        "corrected_trajectory_used": False,
        "rollout_state_writes": 0,
        "hidden_support": False,
    }
    _write(output / "stable_grasp_qualification.json", qualification)
    _write(
        output / "stable_grasp_exact_geometry.json",
        {
            "schema_version": "StableGraspExactGeometryAuditV1",
            "backend": "python-fcl==0.7.0.11",
            "metric": "RuntimeCollisionProxyPenetrationV1",
            "development_candidates_exact_complete": len(completed_exact),
            "development_candidates_exact_failed_closed": len(exact_failures),
            "formal20_geometry_rows": [],
            "formal20_not_run_reason": CALIBRATION_BLOCKER,
            "rows": rows,
        },
    )
    _write(
        output / "geometry_v1_attainability.json",
        {
            "schema_version": "RuntimePenetrationGateAttainabilityAuditV1",
            "status": "STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED",
            "v1_attainable": False,
            "v1_attainability_determined": False,
            "reason": "stable free-object calibration prerequisite was not established",
            "absolute_gate_unchanged": True,
            "stable_calibration": qualification["status"],
        },
    )
    _write(
        output / "empirical_dynamic_contact_reference.json",
        {
            "schema_version": "EmpiricalStableDynamicContactReferenceV1",
            "status": "NOT_CREATED_GATE_BLOCKED",
            "reason": CALIBRATION_BLOCKER,
            "physical_truth_claimed": False,
            "mathematical_lower_bound_claimed": False,
            "bootstrap_ucb_computed": False,
        },
    )
    _write(
        output / "geometry_v1_v2_decision.json",
        {
            "schema_version": "Stage16DGeometryV1V2DecisionV1",
            "status": "STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED",
            "v1_retained_as_current_contract": True,
            "v2_created": False,
            "reason": CALIBRATION_BLOCKER,
            "absolute_gate_unchanged": True,
        },
    )
    _write(
        output / "geometry_v2_contract.json",
        {
            "schema_version": "RuntimeCollisionProxyPenetrationV2",
            "status": "NOT_CREATED_GATE_BLOCKED",
            "reason": CALIBRATION_BLOCKER,
            "parent_v1": "geometry_v1_contract.json",
        },
    )

    downstream = (
        "online_geometry_signal_qualification",
        "optimizer_170650_g1",
        "optimizer_170650_g2",
        "optimizer_170105_g1",
        "optimizer_170105_g2",
        "demonstration_manifest",
        "bc_training_170650",
        "bc_training_170105",
        "ppo_env_benchmark",
        "ppo_training_170650",
        "ppo_training_170105",
        "ppo_evaluation_170650",
        "ppo_evaluation_170105",
        "two_clip_ppo",
        "v2_data_inventory",
        "sensitivity_audit",
        "visual_review",
    )
    for name in downstream:
        _write(output / f"{name}.json", _not_run(name, CALIBRATION_BLOCKER))
    for short in ("170105", "170650"):
        baseline = trajectory_baseline["geometry_aware_closeout"]["trajectories"][f"hocap_{short}"]
        _write(
            output / f"trajectory_qualification_{short}.json",
            {
                "schema_version": "Stage16DStableGraspTrajectoryStopV1",
                "status": baseline,
                "new_optimization_run": False,
                "new_formal20_run": False,
                "reason": CALIBRATION_BLOCKER,
            },
        )

    transitions = [
        {"from": "INPUT_FREEZE", "to": "C1_DEVELOPMENT", "reason": "inputs frozen"},
        {
            "from": "C1_DEVELOPMENT",
            "to": "C2_DEVELOPMENT",
            "reason": "0 eligible families after complete C1 matrix",
        },
        {
            "from": "C2_DEVELOPMENT",
            "to": "CLOSEOUT",
            "reason": CALIBRATION_BLOCKER,
        },
    ]
    (output / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions),
        encoding="utf-8",
    )
    _write(
        output / "resource_usage.json",
        {
            "schema_version": "Stage16DStableGraspResourceUsageV1",
            "generated_at": generated,
            "development_candidates": 20,
            "development_replicas": 80,
            "scheduled_control_intervals": 25680,
            "formal20_runs": 0,
            "optimizer_runs": 0,
            "ppo_samples": 0,
            "unrelated_processes_terminated": 0,
        },
    )
    tests = {
        "schema_version": "Stage16DStableGraspTestsV1",
        "status": args.tests_status,
        "summary": args.test_summary,
        "base_package_import_without_isaac": "PASS" if args.tests_status == "PASS" else "FAIL",
        "local_tracked_files": _git("ls-files", ".local").splitlines(),
    }
    _write(output / "tests.json", tests)
    _write(
        output / "git_commits.json",
        {
            "schema_version": "Stage16DStableGraspGitCommitsV1",
            "branch": branch,
            "start_head": "7488657d4423e2856666bc582e2d711b4a9a98c5",
            "frozen_calibration_head": frozen["head"],
            "closeout_head": head,
            "commits": _git("log", "--oneline", "7488657..HEAD").splitlines(),
            "pushed": False,
            "pr_created": False,
            "main_merged": False,
            "tag_created": False,
            "release_created": False,
        },
    )
    final = {
        "schema_version": "Stage16DStableGraspGeometryPPOCloseoutV1",
        "generated_at": generated,
        "branch": branch,
        "start_head": "7488657d4423e2856666bc582e2d711b4a9a98c5",
        "frozen_calibration_head": frozen["head"],
        "closeout_head": head,
        "calibration": qualification["status"],
        "calibration_stop_marker": CALIBRATION_BLOCKER,
        "v1_attainability": "NOT_DETERMINED_STABLE_CALIBRATION_MISSING",
        "v2_decision": "NOT_CREATED_STABLE_CALIBRATION_MISSING",
        "online_geometry_signal": "NOT_RUN_GATE_BLOCKED",
        "trajectories": trajectory_baseline["geometry_aware_closeout"]["trajectories"],
        "ppo": {
            "hocap_170105": "STAGE16D_170105_PPO_NOT_RUN_GATE_BLOCKED",
            "hocap_170650": "STAGE16D_170650_PPO_NOT_RUN_GATE_BLOCKED",
            "two_clip": "STAGE16D_TWO_CLIP_PPO_NOT_RUN_GATE_BLOCKED",
            "samples": 0,
            "checkpoints": [],
        },
        "v2_export": "STAGE16D_PHYSICS_DATA_V2_BLOCKED",
        "absolute_geometry_gate_unchanged": True,
        "runtime_collision_geometry_modified": False,
        "formal20_runs": 0,
        "overall": "STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE",
        "blocker": CALIBRATION_BLOCKER,
        "tests": tests["status"],
    }
    _write(output / "final_summary.json", final)
    summary_md = f"""# Stage 16-D stable-grasp calibration closeout

- Overall: `{final["overall"]}`
- Calibration: `{qualification["status"]}` / `{CALIBRATION_BLOCKER}`
- Frozen search: C1 12 + unique C2 8 development candidates, 4 replicas × 321 steps
- Eligible candidates: 0; formal 20-replica runs: 0
- V2, optimizer, BC, PPO, two-clip PPO, and V2 export: not run, gate blocked
- Absolute geometry gate: unchanged at max <10 mm and active-p95 <=3 mm
- Frozen calibration HEAD: `{frozen["head"]}`
- Closeout HEAD: `{head}`
"""
    (output / "final_summary.md").write_text(summary_md, encoding="utf-8")

    handoff = f"""# Stage 16-D Stable Dynamic Contact Geometry and PPO Handoff

## 1. Final Status

`{final["overall"]}`. Calibration is `{qualification["status"]}` with stop marker
`{CALIBRATION_BLOCKER}`. V2 and all downstream training are not authorized.

## 2. Git and Environment

Branch `{branch}`; start `7488657`; frozen calibration `{frozen["head"]}`; closeout `{head}`.
Isaac used CUDA on the RTX 5080 with python-fcl 0.7.0.11. No push/PR/merge/tag/release.

## 3. Frozen Stage16D Baseline

`frozen_inputs.json` references `{frozen["archive"]}`. Source, Stage-12, collision assets,
physics, controller, and the absolute 10 mm / 3 mm gate were unchanged.

## 4. Grasp-Topology Family Contract

Data-derived families: `{", ".join(row["identifier"] for row in topology["families"])}`.
170105 uses the derived thumb/index opposition family; 170650 uses the derived
thumb/index/pinky multi-finger family. The extraction algorithm is shared.

## 5. Object-Canonical Initialization

Runtime convex-proxy centroid, PCA/OBB axes, extents, support directions, and exact
python-fcl reset refinement define the reset pose. It does not use corrected trajectories.

## 6. Stable-Grasp Calibration Matrix

C1 has 6 candidates/object at -6/0/+6 mm × closure 0.5/1.0. Unique C2 has
4 candidates/object at -10/+10 mm × the same closure grid. No post-result candidate was added.

## 7. Stable-Grasp Qualification

All 20 development candidates ran 4 replicas × 321 scheduled steps. Eligible count is 0;
formal20 count is 0. 170105's best diagnostic is `{best["hocap_170105"]["candidate_id"]}`
with terminal hold `{best["hocap_170105"]["terminal_hold_stability"]}`, linear p95
`{best["hocap_170105"]["terminal_linear_speed_p95_max_mps"]}` m/s and angular p95
`{best["hocap_170105"]["terminal_angular_speed_p95_max_radps"]}` rad/s. 170650 never
established its required three-group topology.

## 8. Exact Calibration Geometry Audit

{len(completed_exact)} candidates completed exact V1 audit. {len(exact_failures)} candidate
failed closed on a signed-distance/contact-MTD disagreement; no fallback or imputation was used.

## 9. V1 Attainability

Not determined because no stable calibration passed the prerequisite gate.

## 10. Empirical Dynamic Contact Reference

Not created. Such a reference is empirical engineering evidence, never physical truth or a
mathematical lower bound.

## 11. V1/V2 Geometry Contract Decision

V1 remains current. V2 was not created. The absolute gate remains unchanged.

## 12. Online Geometry Signal

Not run or qualified because calibration blocked before V1/V2 decision.

## 13. 170650 Geometry Refinement

Not run; previous trajectory remains partial/blocked.

## 14. 170105 Geometry and Terminal Refinement

Not run; previous trajectory remains partial/blocked.

## 15. Per-Clip Trajectory Qualification

No new formal trajectory qualification was authorized.

## 16. Demonstration Dataset

Not generated.

## 17. Behavior Cloning

Not run; zero samples/checkpoints.

## 18. 170650 Single PPO

Not run; zero samples/checkpoints.

## 19. 170105 Single PPO

Not run; zero samples/checkpoints.

## 20. Two-Clip PPO

Not run; both single policies are unqualified.

## 21. PhysicsQualifiedIsaacTrajectoryV2

Not exported.

## 22. Sensitivity Audit

Not run because no V2 artifact exists.

## 23. Failure-Recovery

C1 exhausted → unique C2 exhausted → fail-closed closeout. No gate, physics, collision,
controller, or topology thresholds were relaxed.

## 24. Commands

```bash
conda run -n toporetarget-rl python scripts/rl/isaaclab/freeze_stage16d_stable_grasp_inputs.py
conda run -n toporetarget-rl python scripts/rl/isaaclab/build_stage16d_grasp_topology_families.py
conda run -n toporetarget-rl python \
  scripts/rl/isaaclab/freeze_stage16d_stable_grasp_matrix.py --level C1
conda run -n toporetarget-rl python \
  scripts/rl/isaaclab/freeze_stage16d_stable_grasp_matrix.py --level C2
conda run -n toporetarget-rl python \
  scripts/rl/isaaclab/qualify_stage16d_stable_grasp.py \
  --phase select-development --level C1
conda run -n toporetarget-rl python \
  scripts/rl/isaaclab/qualify_stage16d_stable_grasp.py \
  --phase select-development --level C2
```

The exact candidate commands and IDs are frozen in the two matrix JSON files. Downstream
optimizer/BC/PPO commands are intentionally not authorized by this result.

## 25. Tests

`{args.test_summary}`

## 26. README and Roadmap

Updated to record calibration-blocked status and the prohibited downstream transition.

## 27. Local Commits

See `git_commits.json`. PUSHED=NO, PR_CREATED=NO, MAIN_MERGED=NO, TAG_CREATED=NO,
RELEASE_CREATED=NO.

## 28. Remaining Limitations

Collision proxies are not visual truth; factor-8 changes time semantics; the explicit virtual
wrist is not a real arm; physics is not physically calibrated; no sim-to-real claim is made.
No V2 data can be exported without qualified PPO.

## 29. Recommended Next Action

Open a new bounded research stage for a shared passive-stability controller/initializer design,
then freeze a new calibration generation before any new candidate outcome. Do not define V2
from these failed development traces.
"""
    (output / "handoff.md").write_text(handoff, encoding="utf-8")
    dashboard = f"""<!doctype html><html><head><meta charset="utf-8"><title>Stage16D D.4R3</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse}}
td,th{{border:1px solid #aaa;padding:.35rem}}
.fail{{color:#a00}}</style>
</head><body><h1>Stage 16-D.4R3 stable-grasp calibration</h1>
<p class="fail"><strong>{html.escape(CALIBRATION_BLOCKER)}</strong></p>
<p>20 frozen development candidates; 0 eligible; formal20/V2/optimizer/PPO not run.</p>
<table><tr><th>Object</th><th>Best diagnostic</th><th>Topology</th><th>Hold</th>
<th>Linear p95</th><th>Angular p95</th></tr>
<tr><td>170105</td><td>{best["hocap_170105"]["candidate_id"]}</td><td>{best["hocap_170105"]["topology_coverage"]}</td><td>{best["hocap_170105"]["terminal_hold_stability"]}</td><td>{best["hocap_170105"]["terminal_linear_speed_p95_max_mps"]}</td><td>{best["hocap_170105"]["terminal_angular_speed_p95_max_radps"]}</td></tr>
<tr><td>170650</td><td>{best["hocap_170650"]["candidate_id"]}</td><td>{best["hocap_170650"]["topology_coverage"]}</td><td>{best["hocap_170650"]["terminal_hold_stability"]}</td><td>{best["hocap_170650"]["terminal_linear_speed_p95_max_mps"]}</td><td>{best["hocap_170650"]["terminal_angular_speed_p95_max_radps"]}</td></tr>
</table><p>Generated {html.escape(generated)}</p></body></html>"""
    (output / "dashboard.html").write_text(dashboard, encoding="utf-8")
    print(json.dumps({"status": final["overall"], "blocker": CALIBRATION_BLOCKER}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
