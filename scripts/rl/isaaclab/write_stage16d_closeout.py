#!/usr/bin/env python3
# ruff: noqa: E501
"""Assemble the truthful Stage 16-D closeout from ignored runtime evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLIPS = ("170105", "170650")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(".local/reports/stage16d_physics_consistent_retargeting"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(".local/physics_consistent_retargeting"),
    )
    parser.add_argument("--ruff", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--format", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--mypy", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--pytest", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--paper-fidelity", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--pytest-summary", default="not run")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, text=True, capture_output=True).stdout.strip()


def trajectory_summary(report_root: Path, clip: str) -> dict[str, Any]:
    qualification_path = report_root / f"trajectory_qualification_{clip}_v3.json"
    geometry_path = report_root / f"geometry_audit_{clip}_v3.json"
    optimizer_path = report_root / f"optimizer_{clip}_s3.json"
    qualification = load(qualification_path)
    geometry = load(geometry_path)
    optimizer = load(optimizer_path)
    collision = geometry["collision_proxy"]
    return {
        "schema_version": "Stage16DIntegratedTrajectoryQualificationV1",
        "clip": f"hocap_{clip}",
        "status": f"STAGE16D_{clip}_TRAJECTORY_PARTIAL_BLOCKED",
        "empirical_classification": qualification["empirical_classification"],
        "formal_geometry_gate": geometry["formal_geometry_gate"],
        "formal_classification": "BLOCKED_METRIC_COMPARABILITY_AND_VISUAL_SIGN",
        "replicas": qualification["replicas"],
        "success_rate": qualification["success_rate"],
        "semantic_reach_rate": qualification["semantic_reach_rate"],
        "contact_topology_pass_rate": qualification["contact_topology_pass_rate"],
        "contact_causality_pass_rate": qualification["contact_causality_pass_rate"],
        "terminal_stability_pass_rate": qualification["terminal_stability_pass_rate"],
        "complete_trajectory_rate": qualification["complete_trajectory_rate"],
        "numerical_pass_rate": qualification["numerical_pass_rate"],
        "max_penetration_lower_bound_m": collision["max_penetration_lower_bound_m"],
        "p95_penetration_lower_bound_m": collision["p95_penetration_lower_bound_m"],
        "penetration_metric_role": collision["metric_role"],
        "formal_object_rollout_state_writes": 0,
        "formal_wrist_rollout_state_writes": 0,
        "hidden_force": False,
        "hidden_attachment": False,
        "optimizer": {
            "knots": optimizer["config"]["knot_count"],
            "population": optimizer["config"]["population"],
            "replicas": optimizer["config"]["replicas"],
            "iterations": optimizer["config"]["iterations"],
            "wall_time_s": optimizer["wall_time_s"],
            "action_trace": optimizer["action_trace"],
        },
        "evidence": {
            "qualification": str(qualification_path.resolve()),
            "geometry": str(geometry_path.resolve()),
            "optimizer": str(optimizer_path.resolve()),
        },
        "blockers": [
            "penetration result is a lower bound rather than an upper bound",
            "runtime visual OBJ is non-watertight and has no signed penetration",
            "Stage 12 SDF and runtime proxy metrics are not directly comparable",
        ],
    }


def failure_transitions() -> list[dict[str, object]]:
    return [
        {"phase": "D.0", "from": "START", "to": "INPUTS_FROZEN", "repair": 0},
        {
            "phase": "D.1",
            "from": "SEMANTIC_EXTRACTION",
            "to": "PARTIAL_SPARSE_C3_CONTACTS",
            "repair": 0,
        },
        {
            "phase": "D.2",
            "from": "ENV_IMPLEMENTED",
            "to": "GPU_SMOKES_PASS_1_128_4096",
            "repair": 0,
        },
        {
            "phase": "D.3-S1",
            "from": "INITIAL_SEGMENT",
            "to": "DEGENERATE_OR_NO_PROGRESS",
            "repair": 0,
        },
        {"phase": "D.3-S2", "from": "BOUNDED_RERUN", "to": "INSUFFICIENT_PROGRESS", "repair": 1},
        {
            "phase": "D.3",
            "from": "TRACE_SELECTION_DEFECT",
            "to": "BEST_EVALUATED_TRACE_RETAINED",
            "repair": 1,
        },
        {
            "phase": "D.3-S3",
            "from": "FULL_DEFAULT_SEARCH",
            "to": "EMPIRICAL_CANDIDATES",
            "repair": 0,
        },
        {
            "phase": "D.4",
            "from": "20_REPLICA_REPLAY",
            "to": "105_SUCCESS_0.75_650_SUCCESS_1.00",
            "repair": 0,
        },
        {
            "phase": "D.4",
            "from": "GEOMETRY_AUDIT",
            "to": "BLOCKED_METRIC_COMPARABILITY_AND_VISUAL_SIGN",
            "repair": 2,
        },
        {
            "phase": "D.4",
            "from": "QUALIFICATION_BLOCKED",
            "to": "V1_PARTIAL_ARTIFACTS_EXPORTED",
            "repair": 0,
        },
        {"phase": "D.5", "from": "ENTRY_GATE", "to": "BC_AND_SINGLE_PPO_NOT_RUN", "repair": 0},
        {"phase": "D.6", "from": "SINGLE_PPO_GATE", "to": "TWO_CLIP_PPO_NOT_RUN", "repair": 0},
        {
            "phase": "D.7",
            "from": "NOMINAL_PPO_GATE",
            "to": "SENSITIVITY_NOT_RUN_V2_NOT_EXPORTED",
            "repair": 0,
        },
    ]


def make_handoff(summaries: dict[str, dict[str, Any]], commit_log: str) -> str:
    s105, s650 = summaries["170105"], summaries["170650"]
    artifact_root = Path(".local/physics_consistent_retargeting").resolve()
    sections = {
        1: "`STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE`; D.0 pass, D.1 partial, D.2 pass, D.3/D.4 partial, D.5-D.6 not run, D.7 partial inventory only.",
        2: f"Branch `feature/reference-tracking-isaaclab`; start `363f2c88506c89e69656211d4cbe38901a21f472`; current `{git('rev-parse', 'HEAD')}`; GPU PhysX on RTX 5080.",
        3: "Stage 16-C source NPZs, Stage 12 outputs, hashes, and failure ledger are frozen under the ignored report root; none were overwritten.",
        4: "The source wrist/finger intent and semantic/contact contracts remain priors, while the free object trajectory is generated causally by PhysX and may differ from source.",
        5: "Both clips use the low-confidence `generic_contact_preserving_motion` fallback: 170105 requires index; 170650 requires index+pinky. No clip-specific controller rule exists.",
        6: "Groups are thumb/index/middle/ring/pinky/palm. Onset, duration, persistence, final coverage, and forbidden contact are explicit; point-level pairs were unavailable in the aggregate C3 trace.",
        7: "Real `DirectRLEnv`, 26-D actions, 764-D observations, 120 Hz simulation, decimation 6, 20 Hz control, factor-8 321-step references, free zero-gravity object, no support.",
        8: "Shared `semantic_balanced_v1` reward; hard gates include complete trace, action bounds, wrist/finger/workspace safety, contact causality/topology, terminal semantics, and independent penetration.",
        9: f"Shared phase-wise spline CEM: 16 knots, population 64, 4 replicas, 5 iterations. Runtime {s105['optimizer']['wall_time_s']:.2f}s / {s650['optimizer']['wall_time_s']:.2f}s.",
        10: f"170105 replay: success {s105['success_rate']:.2f}; 170650: {s650['success_rate']:.2f}; semantic/contact/causality all 1.00. Both remain formally blocked by geometry comparability.",
        11: f"V1 partial artifacts: `{artifact_root / 'hocap_170105'}` and `{artifact_root / 'hocap_170650'}` with NPZ, Zarr, manifest, quality, Parquet contacts, action trace, and comparison CSV.",
        12: "Demonstration export not authorized; dataset absent, actor-only BC not run, best/last checkpoints absent.",
        13: "170105 PPO not run: 0 samples and no checkpoints because the corrected-trajectory entry gate failed closed.",
        14: "170650 PPO not run: 0 samples and no checkpoints because formal geometry did not pass despite empirical replay success.",
        15: "Two-clip PPO not run; both single-clip PPO validations are mandatory and absent.",
        16: "`PhysicsQualifiedIsaacTrajectoryV2` was not exported. The inventory contains only explicitly partial V1 packages.",
        17: "Wrist/finger/link and source-object deviations are stored in trajectory arrays; PPO comparison is unavailable. The source object path is a soft prior, not a hard target.",
        18: "Sensitivity audit not run because no nominal PPO is validated.",
        19: "S1/S2 degeneracy triggered bounded S3 search; a trace-selection implementation defect was repaired; geometry stayed blocked; all downstream training stopped at the gate.",
        20: "Headless numerical dashboard generated; no raster video/contact sheet or interactive visual acceptance exists.",
        21: "Executable commands are documented in `docs/stages/STAGE16D_PHYSICS_CONSISTENT_RETARGETING.md` and `docs/rl/PHYSICS_CONSISTENT_RETARGETING.md`.",
        22: "See ignored `tests.json` for the complete repository validation results.",
        23: "English/Chinese README and roadmap plus Stage 16-D environment, optimizer, PPO, recovery, fidelity, and assumption docs are synchronized.",
        24: f"Local-only commits:\n\n```text\n{commit_log}\n```\n\nPUSHED=NO; PR_CREATED=NO; MAIN_MERGED=NO; TAG_CREATED=NO; RELEASE_CREATED=NO.",
        25: "Factor-8 changes timing; virtual wrist is not a robot arm; mass/inertia/friction lack physical calibration; simulation data are not real-robot data; no sim-to-real claim; optimized seeds are not PPO data.",
        26: "Build one metric-compatible, signed, runtime-geometry penetration audit and requalify both candidates. Only after both formal trajectory gates pass may BC and single-clip PPO start.",
    }
    headings = [
        "Final Status",
        "Git and Environment",
        "Frozen Stage 16-C Evidence",
        "Stage 16-D Goal and Contract",
        "Task Semantic Classification",
        "Contact-Topology Contract",
        "Physics-Correction Environment",
        "Reward, Termination, and Anti-Degenerate Gates",
        "Robust Trajectory Optimization",
        "Corrected Trajectory Qualification",
        "PhysicsConsistentRetargetedTrajectoryV1 Artifacts",
        "Demonstration Dataset and BC",
        "170105 Single-Clip PPO",
        "170650 Single-Clip PPO",
        "Two-Clip PPO",
        "PhysicsQualifiedIsaacTrajectoryV2 Export",
        "Source / Corrected / PPO Comparison",
        "Sensitivity Audit",
        "Failure-Recovery",
        "Visualization",
        "Commands",
        "Tests",
        "README and Roadmap",
        "Local Commits",
        "Remaining Limitations",
        "Recommended Next Action",
    ]
    body = ["# Stage 16-D Physics-Consistent Retargeting and PPO Handoff", ""]
    for index, heading in enumerate(headings, start=1):
        body.extend((f"## {index}. {heading}", "", sections[index], ""))
    return "\n".join(body)


def main() -> int:
    args = parse_args()
    root = args.report_root
    root.mkdir(parents=True, exist_ok=True)
    summaries = {clip: trajectory_summary(root, clip) for clip in CLIPS}
    for clip, summary in summaries.items():
        write_json(root / f"trajectory_qualification_{clip}.json", summary)
        optimizer = load(root / f"optimizer_{clip}_s3.json")
        write_json(
            root / f"optimizer_{clip}.json",
            {
                "schema_version": "Stage16DOptimizerCloseoutV1",
                "clip": f"hocap_{clip}",
                "status": "STAGE16D_OPTIMIZATION_CANDIDATE_PRODUCED",
                "stage": optimizer["stage"],
                "config": optimizer["config"],
                "wall_time_s": optimizer["wall_time_s"],
                "action_trace": optimizer["action_trace"],
                "action_trace_sha256": optimizer["action_trace_sha256"],
                "object_is_decision_variable": False,
                "formal_result": summaries[clip]["formal_classification"],
                "raw_report": str((root / f"optimizer_{clip}_s3.json").resolve()),
            },
        )
    write_json(
        root / "optimizer_config.json",
        {
            "schema_version": "Stage16DOptimizerConfigCloseoutV1",
            "shared_across_clips": True,
            "config": load(root / "optimizer_170105_s3.json")["config"],
            "action_shape": [321, 26],
            "object_trajectory": "free_physx_rollout_output_not_decision_variable",
            "selection": "best_lexically_ranked_evaluated_candidate_across_iterations",
        },
    )
    write_json(
        root / "penetration_audit.json",
        {
            "schema_version": "Stage16DPenetrationAuditCloseoutV1",
            "status": "BLOCKED_METRIC_COMPARABILITY_AND_VISUAL_SIGN",
            "clips": {
                clip: {
                    "max_lower_bound_m": summaries[clip]["max_penetration_lower_bound_m"],
                    "p95_lower_bound_m": summaries[clip]["p95_penetration_lower_bound_m"],
                    "raw_report": summaries[clip]["evidence"]["geometry"],
                }
                for clip in CLIPS
            },
            "limitation": "values are collision-proxy lower bounds and cannot prove the formal upper-bound gate",
        },
    )
    transitions = failure_transitions()
    (root / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions),
        encoding="utf-8",
    )
    env_smokes = {
        name: load(root / f"env_smoke_{name}.json")["wall_time_s"]
        for name in ("170105", "170650", "128", "4096")
    }
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "Stage16DResourceUsageV1",
            "gpu": "NVIDIA GeForce RTX 5080 16 GiB",
            "env_smoke_wall_time_s": env_smokes,
            "optimizer_wall_time_s": {
                clip: summaries[clip]["optimizer"]["wall_time_s"] for clip in CLIPS
            },
            "ppo_samples": {clip: 0 for clip in CLIPS},
            "ppo_checkpoints": [],
        },
    )
    tests = {
        "schema_version": "Stage16DTestCloseoutV1",
        "ruff_check": args.ruff,
        "ruff_format_check": args.format,
        "mypy": args.mypy,
        "pytest": args.pytest,
        "pytest_summary": args.pytest_summary,
        "paper_fidelity": args.paper_fidelity,
        "isaac_platform_smoke_100_steps": "PASS",
        "isaac_env_smokes": {"1": "PASS", "128": "PASS", "4096": "PASS"},
        "isaac_targeted_failures": 0,
        "local_tracked_files": len(git("ls-files", ".local").splitlines()),
    }
    write_json(root / "tests.json", tests)
    commit_log = git("log", "--oneline", "origin/main..HEAD")
    commits = {
        "schema_version": "Stage16DGitCloseoutV1",
        "branch": git("branch", "--show-current"),
        "start_head": "363f2c88506c89e69656211d4cbe38901a21f472",
        "final_head": git("rev-parse", "HEAD"),
        "origin_main_to_head": commit_log.splitlines(),
        "pushed": False,
        "pr_created": False,
        "main_merged": False,
        "tag_created": False,
        "release_created": False,
    }
    write_json(root / "git_commits.json", commits)
    final = {
        "schema_version": "Stage16DPhysicsConsistentRetargetingCloseoutV1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE",
        "stages": {
            "D.0": "VALIDATED",
            "D.1": "PARTIAL",
            "D.2": "VALIDATED",
            "D.3": "PARTIAL_BLOCKED",
            "D.4": "PARTIAL_BLOCKED",
            "D.5": "NOT_RUN_GATE_BLOCKED",
            "D.6": "NOT_RUN_GATE_BLOCKED",
            "D.7": "PARTIAL_BLOCKED",
        },
        "trajectories": summaries,
        "ppo": {
            clip: {"status": f"STAGE16D_{clip}_PPO_NOT_RUN", "samples": 0, "checkpoints": []}
            for clip in CLIPS
        },
        "two_clip_ppo": "STAGE16D_TWO_CLIP_PPO_NOT_RUN",
        "data": "STAGE16D_PHYSICS_DATA_PARTIAL_BLOCKED",
        "next_action": "establish a signed metric-compatible runtime-geometry penetration audit, then requalify",
    }
    write_json(root / "final_summary.json", final)
    handoff = make_handoff(summaries, commit_log)
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    (root / "final_summary.md").write_text(
        "# Stage 16-D closeout\n\n"
        "`STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE`\n\n"
        "Both 321-step action candidates produced causal free-object PhysX rollouts. "
        "The 20-replica empirical success rates are 0.75 (170105) and 1.00 (170650), "
        "but the independent penetration result is only a lower bound and the runtime "
        "visual meshes are non-watertight. BC/PPO therefore remained unauthorized: "
        "zero samples and no checkpoints. See `handoff.md` for the complete handoff.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": final["status"], "report_root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
