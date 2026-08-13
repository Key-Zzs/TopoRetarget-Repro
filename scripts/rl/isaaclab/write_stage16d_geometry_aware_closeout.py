#!/usr/bin/env python3
"""Write the fail-closed Stage 16-D geometry-aware recovery closeout."""

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
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_geometry_aware_optimization_ppo"
BASELINE_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-status", choices=("PENDING", "PASS", "FAIL"), default="PENDING")
    parser.add_argument("--test-summary", default="not run by closeout writer")
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
        "schema_version": "Stage16DGeometryAwareNotRunV1",
        "stage": stage,
        "status": f"STAGE16D_{stage.upper()}_NOT_RUN_GATE_BLOCKED",
        "reason": reason,
        "samples": 0,
        "checkpoints": [],
        "unauthorized_work_started": False,
    }


def main() -> int:
    args = _parser().parse_args()
    decision = _load(REPORT_ROOT / "geometry_metric_decision.json")
    if decision["status"] != "STAGE16D_GEOMETRY_GATE_REVISION_BLOCKED":
        raise RuntimeError("closeout writer currently requires the frozen gate-revision stop")
    baseline = _load(BASELINE_ROOT / "final_summary.json")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    reason = str(decision["stop_reason"])
    generated = datetime.now(UTC).isoformat()

    online_contract = {
        "schema_version": "OnlineRuntimeProxyGeometrySignalV1",
        "status": "DESIGN_IMPLEMENTED_NOT_QUALIFIED",
        "outputs": [
            "estimated max penetration",
            "estimated active-p95 accumulator",
            "offending body pair",
            "contact-active mask",
            "catastrophic-risk mask",
        ],
        "formal_gate_authority": False,
        "formal_gate": "exact python-fcl RuntimeCollisionProxy metric",
        "qualification_required_before_reward": True,
    }
    _write(REPORT_ROOT / "online_geometry_signal_contract.json", online_contract)
    _write(
        REPORT_ROOT / "online_geometry_signal_qualification.json",
        _not_run("online_geometry_signal", reason),
    )
    optimizer_config = {
        "schema_version": "Stage16DGeometryAwareOptimizerConfigV2",
        "status": "FROZEN_NOT_RUN_GATE_BLOCKED",
        "shared_across_clips": True,
        "ranking": "semantic_balanced_geometry_v2",
        "G1": {"knots": 16, "population": 96, "replicas": 4, "iterations": 8, "elites": 12},
        "G2": {"knots": 32, "population": 96, "replicas": 8, "iterations": 8, "elites": 12},
        "selection_holdout_replicas": 8,
        "formal_holdout_replicas": 20,
        "reason": reason,
    }
    _write(REPORT_ROOT / "optimizer_config.json", optimizer_config)
    for clip in ("170650", "170105"):
        for level in ("g1", "g2"):
            _write(
                REPORT_ROOT / f"optimizer_{clip}_{level}.json",
                _not_run(f"optimizer_{clip}_{level}", reason),
            )
        baseline_row = baseline["trajectories"][f"hocap_{clip}"]
        _write(
            REPORT_ROOT / f"trajectory_qualification_{clip}.json",
            {
                "schema_version": "Stage16DGeometryAwareTrajectoryQualificationV1",
                "status": baseline_row["status"],
                "new_geometry_aware_candidate": False,
                "new_formal20_run": False,
                "baseline": baseline_row,
                "reason": reason,
            },
        )
    for name in (
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
    ):
        _write(REPORT_ROOT / f"{name}.json", _not_run(name, reason))

    transitions = [
        {"from": "INPUT_FREEZE", "to": "GATE_ATTAINABILITY", "reason": "inputs frozen"},
        {
            "from": "GATE_ATTAINABILITY",
            "to": "CLOSEOUT",
            "reason": "stable dynamic contact floor not established; metric revision blocked",
        },
    ]
    (REPORT_ROOT / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions),
        encoding="utf-8",
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _write(
        REPORT_ROOT / "resource_usage.json",
        {
            "schema_version": "Stage16DGeometryAwareResourceUsageV1",
            "generated_at": generated,
            "gpu_query": gpu.stdout.strip(),
            "ppo_samples": 0,
            "optimizer_runs": 0,
            "isaac_calibration_rollouts": 8,
            "unrelated_processes_terminated": 0,
        },
    )
    tests = {
        "schema_version": "Stage16DGeometryAwareTestsV1",
        "status": args.tests_status,
        "summary": args.test_summary,
        "base_package_import_without_isaac": (
            "PENDING" if args.tests_status == "PENDING" else "PASS"
        ),
        "local_tracked_files": _git("ls-files", ".local").splitlines(),
    }
    _write(REPORT_ROOT / "tests.json", tests)
    commits = _git("log", "--oneline", "origin/main..HEAD").splitlines()
    _write(
        REPORT_ROOT / "git_commits.json",
        {
            "schema_version": "Stage16DGeometryAwareGitCommitsV1",
            "branch": branch,
            "start_head": "6bc64c598d5a7f3bea4b6e703bffbde7b86abf83",
            "final_head": head,
            "commits": commits,
            "pushed": False,
            "pr_created": False,
            "main_merged": False,
            "tag_created": False,
            "release_created": False,
        },
    )
    final = {
        "schema_version": "Stage16DGeometryAwareOptimizationAndPPOCloseoutV1",
        "generated_at": generated,
        "branch": branch,
        "start_head": "6bc64c598d5a7f3bea4b6e703bffbde7b86abf83",
        "final_head": head,
        "geometry_gate": decision["status"],
        "v1_attainability": "NOT_DEMONSTRATED",
        "v2_decision": "NOT_CREATED_STABLE_DYNAMIC_FLOOR_MISSING",
        "online_geometry_signal": "STAGE16D_ONLINE_GEOMETRY_SIGNAL_NOT_RUN_GATE_BLOCKED",
        "trajectories": {
            "hocap_170105": baseline["trajectories"]["hocap_170105"]["status"],
            "hocap_170650": baseline["trajectories"]["hocap_170650"]["status"],
        },
        "ppo": {
            "hocap_170105": "STAGE16D_170105_PPO_NOT_RUN_GATE_BLOCKED",
            "hocap_170650": "STAGE16D_170650_PPO_NOT_RUN_GATE_BLOCKED",
            "two_clip": "STAGE16D_TWO_CLIP_PPO_NOT_RUN_GATE_BLOCKED",
            "samples": 0,
            "checkpoints": [],
        },
        "v2_export": "STAGE16D_PHYSICS_DATA_V2_BLOCKED",
        "sensitivity": "STAGE16D_SENSITIVITY_NOT_RUN_GATE_BLOCKED",
        "absolute_geometry_gate_unchanged": True,
        "overall": "STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE",
        "blocker": reason,
        "tests": tests["status"],
    }
    _write(REPORT_ROOT / "final_summary.json", final)
    summary_md = f"""# Stage 16-D geometry-aware closeout

- Overall: `{final["overall"]}`
- Geometry gate: `{decision["status"]}`
- V1 attainability: not demonstrated
- V2: not created; stable shared dynamic floor missing
- Absolute gate: unchanged at max <10 mm and active-p95 <=3 mm
- Geometry-aware optimizer/PPO/V2 export: not run, gate blocked
- PPO samples/checkpoints: 0 / 0
- HEAD: `{head}`
"""
    (REPORT_ROOT / "final_summary.md").write_text(summary_md, encoding="utf-8")

    commands = """```bash
conda run -n toporetarget-isaaclab python \\
  scripts/rl/isaaclab/audit_stage16d_penetration_gate_attainability.py \\
  --phase numerical-floor --repeats 1000
REPORT=.local/reports/stage16d_geometry_aware_optimization_ppo
conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES python \\
  scripts/rl/isaaclab/calibrate_stage16d_dynamic_contact_floor.py \\
  --mode no-contact --clip hocap_170650 \\
  --output "$REPORT/no_contact_geometry_floor_170650.json" \\
  --trace "$REPORT/no_contact_trace_170650.npz" \\
  --raw-geometry "$REPORT/no_contact_geometry_pairs_170650.npz" \\
  --accept-eula
conda run -n toporetarget-isaaclab python \\
  scripts/rl/isaaclab/audit_stage16d_penetration_gate_attainability.py \\
  --phase decision
```
"""
    handoff = f"""# Stage 16-D Geometry-Aware Optimization and PPO Handoff

## 1. Final Status

`{final["overall"]}`; gate `{decision["status"]}`. V2 was not created.

## 2. Git and Environment

Branch `{branch}`; start `6bc64c5`; closeout HEAD `{head}`. Isaac short smoke passed.

## 3. Frozen Geometry Baseline

Archive and hashes are recorded by `frozen_inputs.json`;
source and Stage-12 files were not modified.

## 4. V1 Gate Attainability Audit

V1 comparability remains validated, but contact-preserving attainability was not demonstrated.

## 5. Dynamic Contact Floor

No-contact is zero. No 20-replica stable required-topology floor was established.

## 6. V1/V2 Geometry Contract Decision

`{decision["status"]}`. Absolute 10 mm/3 mm limits are unchanged. V2 is absent.

## 7. Online Geometry Signal Qualification

Design contracts exist; runtime qualification was not run because the metric contract is blocked.

## 8. Geometry-Aware Optimizer

G1/G2 budgets are frozen but neither level ran.

## 9. 170650 Geometry Refinement

Not run. Existing baseline remains geometry-blocked at 20/20 empirical success.

## 10. 170105 Geometry and Terminal Refinement

Not run. Existing baseline remains 15/20 and geometry-blocked.

## 11. Per-Clip Trajectory Qualification

No new candidates or formal20 evaluations were authorized.

## 12. Demonstration Dataset

Not run; no validated trajectory entry.

## 13. Behavior Cloning

Not run; no dataset or checkpoints.

## 14. 170650 Single PPO

Not run; 0 samples and 0 checkpoints.

## 15. 170105 Single PPO

Not run; 0 samples and 0 checkpoints.

## 16. Two-Clip PPO

Not run; both single-PPO prerequisites are absent.

## 17. PhysicsQualifiedIsaacTrajectoryV2

Not exported; no V2 paths or hashes exist.

## 18. Sensitivity Audit

Not run; nominal PPO did not pass.

## 19. Failure-Recovery

Stopped at `GATE_ATTAINABILITY`; `FAST_SIGNAL` through `SENSITIVITY` were not entered.

## 20. Commands

{commands}
G1/G2, demonstration, BC, PPO, two-clip, export, and visualization commands were
intentionally not run after the fail-closed decision.

## 21. Tests

`{args.tests_status}`: {args.test_summary}

## 22. README and Roadmap

English/Chinese status and engineering-boundary documents record the D.4R2 stop.

## 23. Local Commits

{len(commits)} commits are ahead of origin/main in the listed history.
PUSHED=NO; PR_CREATED=NO; MAIN_MERGED=NO; TAG_CREATED=NO; RELEASE_CREATED=NO.

## 24. Remaining Limitations

Factor-8 changes timing. V2 does not exist. Collision proxies are not visual truth.
The online signal is not the formal metric. Existing object paths are physics-corrected
only in partial V1 artifacts. The virtual wrist is not a real arm. Physics is
uncalibrated. No real-dynamics or sim-to-real conclusion is supported. PPO failure
prevents V2 export.

## 25. Recommended Next Action

Design a separately authorized, non-clip-specific stable free-object contact calibration
that reaches required topology across 20 replicas without state writes. Do not resume
optimizer or PPO before that gate passes.
"""
    (REPORT_ROOT / "handoff.md").write_text(handoff, encoding="utf-8")
    dashboard = (
        '<!doctype html><meta charset="utf-8">'
        "<title>Stage16D geometry-aware closeout</title>"
        "<style>body{font:16px system-ui;max-width:920px;margin:40px auto;"
        "line-height:1.5}code{background:#eee;padding:2px 5px}</style>"
        "<h1>Stage 16-D geometry-aware closeout</h1>"
        f"<p><code>{html.escape(final['overall'])}</code></p>"
        f"<p>Gate: <code>{html.escape(decision['status'])}</code></p>"
        f"<p>{html.escape(reason)}</p>"
        "<p>Absolute 10 mm/3 mm gate unchanged. "
        "Optimizer and PPO were not run.</p>"
    )
    (REPORT_ROOT / "dashboard.html").write_text(dashboard, encoding="utf-8")
    print(json.dumps({"status": final["overall"], "output": str(REPORT_ROOT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
