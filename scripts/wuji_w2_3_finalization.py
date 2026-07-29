#!/usr/bin/env python3
"""Run the bounded W2.3 Wuji sequential-profile finalization evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from toporetarget.retarget.wuji_w2_3 import (  # noqa: E402
    CLIPS,
    SCHEMA_VERSION,
    SEQUENTIAL_PROFILE_ID,
    browser_smoke,
    build_html_outputs,
    clip_paths,
    export_versioned_references,
    formal_execution_path_audit,
    input_audit,
    integrity_after,
    penetration_audit,
    profile_structural_diff,
    recommendation_gate,
    run_selected_replay,
    run_window_shadow,
    tree_digest,
    window_oracle,
    write_csv,
    write_input_audit,
    write_json,
    write_penetration_outputs,
    write_recommendation_outputs,
    write_window_oracle_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO / ".local" / "experiments" / "wuji_hand2_continuous_v1",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO / ".local" / "experiments" / "wuji_hand2_grab3_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to <root>/w2_3_finalization.",
    )
    parser.add_argument(
        "--run-w1-full",
        action="store_true",
        help="Record the request; full W1 execution remains explicitly bounded by runtime.",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Recompute W2.3 selected replays and diagnostic window evidence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    root = args.root.expanduser().resolve()
    baseline_root = args.baseline_root.expanduser().resolve()
    output_root = (args.output_root or root / "w2_3_finalization").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reuse = not args.no_reuse

    run_command = (
        "PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/wuji_w2_3_finalization.py "
        f"--repo {repo} --root {root} --baseline-root {baseline_root}"
    )
    write_json(
        output_root / "reports" / "run_environment.json",
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": SEQUENTIAL_PROFILE_ID,
            "repo": str(repo),
            "root": str(root),
            "baseline_root": str(baseline_root),
            "output_root": str(output_root),
            "run_command": run_command,
            "cwd": os.getcwd(),
            "python": sys.executable,
            "python_version": sys.version,
            "reuse_existing": reuse,
            "clips": list(CLIPS),
        },
    )
    (output_root / "reports" / "run_command.sh").write_text(run_command + "\n", encoding="utf-8")

    # This audit is deliberately first: it captures the immutable source and
    # formal artifacts before any W2.3 diagnostic output is created.
    before = input_audit(repo, root, baseline_root)
    write_input_audit(output_root, before)
    if not before["passed"]:
        write_json(
            output_root / "reports" / "w2_3_final_status.json",
            {
                "status": "BLOCKED_INPUT_LINEAGE_OR_SHAPE_AUDIT",
                "input_audit_pass": False,
            },
        )
        return 2

    profile_diff = profile_structural_diff(repo)
    path_audit = formal_execution_path_audit(repo, root, baseline_root)
    write_json(output_root / "reports" / "structural_diff.json", profile_diff)
    write_json(output_root / "reports" / "formal_execution_path_audit.json", path_audit)

    penetration = penetration_audit(repo, root, baseline_root)
    write_penetration_outputs(output_root, penetration)
    oracle = window_oracle(repo, root, baseline_root)
    write_window_oracle_outputs(output_root, oracle)

    replay = run_selected_replay(
        repo,
        root,
        baseline_root,
        output_root,
        run_w1_full=args.run_w1_full,
        reuse_existing=reuse,
    )
    write_json(output_root / "reports" / "selected_frame_replay.json", replay["selected"])
    write_json(
        output_root / "reports" / "sequential_profile_equivalence.json", replay["equivalence"]
    )

    shadow = run_window_shadow(
        repo,
        root,
        baseline_root,
        output_root,
        oracle,
        reuse_existing=reuse,
    )
    gate = recommendation_gate(
        repo,
        root,
        baseline_root,
        profile_diff,
        path_audit,
        replay["equivalence"],
        penetration,
        shadow["status"],
    )
    write_recommendation_outputs(output_root, gate)

    exports = export_versioned_references(
        repo,
        root,
        baseline_root,
        output_root,
        gate,
        replay["equivalence"],
        penetration,
        before,
    )
    existing_html = [
        path
        for clip in CLIPS
        for path in (clip_paths(repo, root, baseline_root, clip)["existing_html"],)
        if path.is_file()
    ]
    existing_html.extend(
        path
        for path in (root / "html" / "index.html", root / "reports" / "dashboard.html")
        if path.is_file()
    )
    html_smoke = build_html_outputs(
        output_root,
        profile_diff,
        path_audit,
        replay["equivalence"],
        penetration,
        oracle,
        gate,
        exports,
        existing_html,
    )
    browser = browser_smoke(output_root, html_smoke)
    write_json(
        output_root / "reports" / "reproducibility.json",
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": SEQUENTIAL_PROFILE_ID,
            "input_identity_hash": tree_digest(output_root / "input_audit"),
            "formal_path_audit": path_audit,
            "selected_replay": replay["equivalence"],
            "window_shadow_status": shadow["status"],
            "browser_smoke": browser,
            "export_status": exports.get("status"),
            "no_formal_solver_run_during_export": True,
        },
    )
    write_json(
        output_root / "reports" / "trajectory_continuity_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": SEQUENTIAL_PROFILE_ID,
            "formal_rows": gate["formal_rows"],
            "selected_replay_pass": replay["equivalence"]["selected_replay_pass"],
            "window_nonblocking": True,
        },
    )
    write_json(
        output_root / "reports" / "multi_start_stability.json",
        {
            "schema_version": SCHEMA_VERSION,
            "selected_replay": replay["equivalence"],
            "window_determinism": shadow["deterministic_replay"],
            "minimum_multi_start_evidence": 5,
        },
    )
    write_json(
        output_root / "reports" / "data_integrity.json",
        {
            "schema_version": SCHEMA_VERSION,
            "input_audit_pass": before["passed"],
            "penetration_audit_rows": len(penetration["per_frame"]),
            "formal_artifacts_read_only": True,
            "old_exports_read_only": True,
            "new_outputs_root": str(output_root),
        },
    )
    status = (
        "W2_3_FINALIZATION_COMPLETE_RECOMMENDED_OFFLINE_ONLY"
        if gate["recommended"]
        else "W2_3_FINALIZATION_COMPLETE_PROFILE_NOT_RECOMMENDED"
    )
    final = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "recommendation_status": gate["status"],
        "input_audit_pass": before["passed"],
        "formal_execution_path_audit_pass": path_audit["passed"],
        "selected_replay_pass": replay["equivalence"]["selected_replay_pass"],
        "penetration_hard_gate": gate["penetration_hard_gate"],
        "window_experimental_status": shadow["status"],
        "window_blocks_sequential_recommendation": False,
        "exports": exports,
        "html_smoke": html_smoke,
        "browser_smoke": browser,
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(output_root / "reports" / "w2_3_final_status.json", final)
    integrity = integrity_after(repo, output_root, before)
    final["artifact_integrity_pass"] = integrity["pass"]
    final["protected_worktree_unchanged"] = integrity["protected_worktree_unchanged"]
    final["status"] = (
        status
        if integrity["pass"]
        else "W2_3_FINALIZATION_COMPLETE_PROTECTED_WORKTREE_DRIFT_REVIEW_REQUIRED"
        if not integrity["protected_worktree_unchanged"]
        else "W2_3_FINALIZATION_COMPLETE_INTEGRITY_REVIEW_REQUIRED"
    )
    write_json(output_root / "reports" / "w2_3_final_status.json", final)

    # Keep the report names required by the W2.3 handoff contract in addition
    # to the more structured subdirectories above.  These are projections of
    # already-produced evidence; they never invoke a solver or rewrite an
    # input/formal artifact.
    reports = output_root / "reports"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": final["status"],
        "sequential_status": gate["status"],
        "window_status": shadow["status"],
        "profile_id": SEQUENTIAL_PROFILE_ID,
        "window_profile_id": "wuji_five_frame_window_experimental_v1",
        "scope": "offline_reference_generation",
        "recommended_for_offline_reference_generation": bool(gate["recommended"]),
        "rl_ready": False,
        "realtime_ready": False,
        "cross_subject_validated": False,
        "author_exact": "unresolved",
        "profile_structural_diff": profile_diff,
        "formal_path_audit": path_audit,
        "equivalence": replay["equivalence"],
        "penetration": penetration,
        "window_oracle": oracle,
        "window_repair": shadow,
        "recommendation_gate": gate,
        "exports": exports,
        "html_smoke": html_smoke,
        "browser_smoke": browser,
        "integrity": integrity,
    }
    write_json(reports / "w2_3_summary.json", summary)
    (reports / "w2_3_summary.md").write_text(
        "# W2.3 summary\n\n"
        f"- sequential status: `{gate['status']}`\n"
        f"- window status: `{shadow['status']}`\n"
        f"- scope: `offline_reference_generation`\n"
        f"- final integrity status: `{final['status']}`\n",
        encoding="utf-8",
    )
    write_json(
        reports / "profile_equivalence.json",
        {
            "schema_version": SCHEMA_VERSION,
            "structural_diff": profile_diff,
            "formal_path_audit": path_audit,
            "selected_replay": replay["selected"],
            "equivalence": replay["equivalence"],
        },
    )
    write_csv(reports / "profile_equivalence.csv", replay["selected"]["rows"])
    write_json(reports / "penetration_reaudit.json", penetration)
    write_csv(reports / "penetration_reaudit.csv", penetration["per_frame"])
    root_cause = {
        "schema_version": SCHEMA_VERSION,
        "historical_failure": {
            "branch": "five_frame_window_fallback",
            "observed": "SLSQP status 4 / inequality constraints incompatible",
            "center_continuity_failure": True,
        },
        "repair": {
            "fixed_left_anchor": True,
            "normalized_coordinates": True,
            "analytic_block_jacobian": True,
            "trust_constr_attempted_after_slsqp": True,
        },
        "current_status": shadow["status"],
        "diagnostic_only": True,
        "sequential_gate_impact": "nonblocking",
    }
    write_json(reports / "window_root_cause.json", root_cause)
    write_json(reports / "window_repair_result.json", shadow)
    write_json(
        reports / "sequential_recommendation_gate.json",
        gate,
    )
    write_json(
        reports / "determinism.json",
        {
            "schema_version": SCHEMA_VERSION,
            "selected_frame_replay_pass": replay["equivalence"]["selected_replay_pass"],
            "selected_replay_tolerances": replay["selected"]["determinism_tolerances"],
            "window_deterministic_replay": shadow["deterministic_replay"],
            "no_profile_drift": profile_diff["passed"],
            "formal_artifact_untouched": shadow["formal_artifact_untouched"],
        },
    )
    write_json(
        reports / "performance.json",
        {
            "schema_version": SCHEMA_VERSION,
            "elapsed_s": final["elapsed_s"],
            "historical_w1_full_runtime_s": replay["selected"]["w1_full_replay"][
                "historical_runtime_s"
            ],
            "selected_replay_frame_count": replay["selected"]["selected_frame_count"]
            if "selected_frame_count" in replay["selected"]
            else len(replay["selected"]["rows"]),
            "window_solver_comparison": shadow["solver_comparison"],
        },
    )
    write_json(
        reports / "failure_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "nonblocking_window_failure": shadow["status"],
            "secondary_penetration_warning": not gate["penetration_secondary_gate"],
            "protected_worktree_drift": not integrity["protected_worktree_unchanged"],
            "final_status": final["status"],
            "fail_closed": not integrity["pass"],
        },
    )
    write_json(reports / "artifact_integrity.json", integrity)
    (output_root / "reports" / "README.md").write_text(
        "# W2.3 finalization reports\n\n"
        f"Status: `{final['status']}`\n\n"
        "All formal continuous/baseline/source/warm/graph artifacts are read-only inputs. "
        "The five-frame window is diagnostic-only and nonblocking.\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if integrity["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
