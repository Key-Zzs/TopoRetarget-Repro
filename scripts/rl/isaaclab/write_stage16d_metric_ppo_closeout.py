#!/usr/bin/env python3
# ruff: noqa: E501
"""Assemble the fail-closed Stage 16-D runtime-geometry/PPO closeout."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.physics_retargeting.recovery import (  # noqa: E402
    Stage16DGeometryAndPPORecoveryStateMachine,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
OLD_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
START_HEAD = "dbbf2f3e30d2ce6d8267864bf398b07aff5ab83c"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruff", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--format", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--mypy", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--pytest", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--paper-fidelity", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--pytest-summary", default="not run")
    parser.add_argument("--preserve-guard-evidence", action="store_true")
    return parser


def _integrated_trajectory_reports() -> dict[str, dict[str, Any]]:
    terminal_path = REPORT_ROOT / "trajectory_requalification_170105.json"
    raw_backup = REPORT_ROOT / "trajectory_requalification_170105_terminal_refined_raw.json"
    current = _read(terminal_path)
    if current.get("schema_version") == "Stage16DTrajectoryQualificationV1":
        shutil.copy2(terminal_path, raw_backup)
        terminal = current
    else:
        terminal = _read(raw_backup)
    global_result = _read(REPORT_ROOT / "trajectory_requalification_170105_global.json")
    geometry_105 = _read(REPORT_ROOT / "geometry_qualification_170105_terminal_refined.json")
    raw_650 = _read(OLD_ROOT / "trajectory_qualification_170650_v3.json")
    geometry_650 = _read(REPORT_ROOT / "geometry_qualification_170650.json")

    q105 = {
        "schema_version": "Stage16DIntegratedTrajectoryQualificationV2",
        "clip": "hocap_170105",
        "selected_candidate": "terminal_refined",
        "status": "STAGE16D_170105_TRAJECTORY_PARTIAL_BLOCKED",
        "success_rate": terminal["success_rate"],
        "success_count": int(round(20 * terminal["success_rate"])),
        "required_success_count": 16,
        "semantic_reach_rate": terminal["semantic_reach_rate"],
        "contact_topology_pass_rate": terminal["contact_topology_pass_rate"],
        "contact_causality_pass_rate": terminal["contact_causality_pass_rate"],
        "terminal_stability_pass_rate": terminal["terminal_stability_pass_rate"],
        "complete_trajectory_rate": terminal["complete_trajectory_rate"],
        "numerical_pass_rate": terminal["numerical_pass_rate"],
        "geometry_pass": geometry_105["formal_pass"],
        "no_hidden_control": True,
        "formal_object_state_writes": terminal["formal_object_state_writes"],
        "formal_wrist_state_writes": terminal["formal_wrist_state_writes"],
        "action_bounds_pass": all(row["action_bounds_pass"] for row in terminal["episodes"]),
        "ppo_authorization": "PPO_NOT_AUTHORIZED_FOR_CLIP",
        "blockers": [
            "formal 20-replica success and terminal stability are 15/20, below 16/20",
            "corrected runtime-proxy max and contact-active p95 exceed source-relative limits",
        ],
        "bounded_repairs": {
            "baseline": "15/20",
            "terminal_refinement": "15/20",
            "single_global_fallback": f"{round(20 * global_result['success_rate'])}/20",
            "additional_optimizer_repairs_authorized": False,
        },
        "evidence": {
            "terminal_raw": str(raw_backup.relative_to(REPO_ROOT)),
            "global_raw": str(
                (REPORT_ROOT / "trajectory_requalification_170105_global.json").relative_to(
                    REPO_ROOT
                )
            ),
            "geometry": str(
                (REPORT_ROOT / "geometry_qualification_170105_terminal_refined.json").relative_to(
                    REPO_ROOT
                )
            ),
        },
    }
    q650 = {
        "schema_version": "Stage16DIntegratedTrajectoryQualificationV2",
        "clip": "hocap_170650",
        "selected_candidate": "stage16d_v3",
        "status": "STAGE16D_170650_TRAJECTORY_PARTIAL_BLOCKED",
        "success_rate": raw_650["success_rate"],
        "success_count": int(round(20 * raw_650["success_rate"])),
        "required_success_count": 16,
        "semantic_reach_rate": raw_650["semantic_reach_rate"],
        "contact_topology_pass_rate": raw_650["contact_topology_pass_rate"],
        "contact_causality_pass_rate": raw_650["contact_causality_pass_rate"],
        "terminal_stability_pass_rate": raw_650["terminal_stability_pass_rate"],
        "complete_trajectory_rate": raw_650["complete_trajectory_rate"],
        "numerical_pass_rate": raw_650["numerical_pass_rate"],
        "geometry_pass": geometry_650["formal_pass"],
        "no_hidden_control": True,
        "formal_object_state_writes": raw_650["formal_object_state_writes"],
        "formal_wrist_state_writes": raw_650["formal_wrist_state_writes"],
        "action_bounds_pass": all(row["action_bounds_pass"] for row in raw_650["episodes"]),
        "ppo_authorization": "PPO_NOT_AUTHORIZED_FOR_CLIP",
        "blockers": [
            "corrected runtime-proxy max and contact-active p95 exceed source-relative limits"
        ],
        "evidence": {
            "qualification": str(
                (OLD_ROOT / "trajectory_qualification_170650_v3.json").relative_to(REPO_ROOT)
            ),
            "geometry": str(
                (REPORT_ROOT / "geometry_qualification_170650.json").relative_to(REPO_ROOT)
            ),
        },
    }
    _write(terminal_path, q105)
    _write(REPORT_ROOT / "trajectory_requalification_170650.json", q650)
    return {"hocap_170105": q105, "hocap_170650": q650}


def _recovery_and_failures() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = Stage16DGeometryAndPPORecoveryStateMachine()
    failures: list[dict[str, Any]] = []

    def move(target: str, reason: str) -> None:
        previous = state.phase
        state.transition(target, reason=reason)
        failures.append({"from": previous, "to": target, "reason": reason})

    move("GEOMETRY_INVENTORY", "frozen hashes and exact C.1 collision meshes inventoried")
    move("QUERY_BACKEND", "hpp-fcl convex overlap returned non-finite distance/normal")
    state.register_geometry_repair(reason="reject non-finite hpp-fcl query path")
    failures.append(
        {
            "phase": "QUERY_BACKEND",
            "failure": "HPP_FCL_CONVEX_OVERLAP_NONFINITE",
            "recovery": "fixed python-fcl 0.7.0.11 backend",
        }
    )
    state.register_geometry_backend("python-fcl==0.7.0.11")
    state.register_geometry_repair(
        reason="record absent python-fcl iteration control as null instead of claiming 1000"
    )
    failures.append(
        {
            "phase": "QUERY_BACKEND",
            "failure": "PYTHON_FCL_MAX_ITERATIONS_NOT_EXPOSED",
            "recovery": "metadata corrected to null; unchanged queries and all formal audits rerun",
        }
    )
    move("QUERY_VALIDATION", "13 analytic signed-query tests passed")
    move("METRIC_FREEZE", "RuntimeCollisionProxyPenetrationV1 frozen")
    state.register_geometry_repair(reason="replace all-frame p95 draft with contact-active p95")
    failures.append(
        {
            "phase": "METRIC_FREEZE",
            "failure": "DRAFT_ALL_FRAME_P95_DILUTION",
            "recovery": "invalidated draft and reran all formal audits with active-only p95",
        }
    )
    move("SOURCE_GEOMETRY", "source trajectories audited with frozen metric")
    move("CORRECTED_GEOMETRY", "both corrected trajectories failed source-relative gates")
    failures.extend(
        [
            {
                "phase": "CORRECTED_GEOMETRY",
                "clip": "hocap_170105",
                "result": "BLOCKED_SOURCE_RELATIVE_MAX_AND_P95",
            },
            {
                "phase": "CORRECTED_GEOMETRY",
                "clip": "hocap_170650",
                "result": "BLOCKED_SOURCE_RELATIVE_MAX_AND_P95",
            },
        ]
    )
    move("D4_REQUALIFICATION", "170105 remained 15/20; 170650 was 20/20 but geometry failed")
    move("TERMINAL_FAILURE_ANALYSIS", "five 170105 failures classified terminal object twist")
    move("TERMINAL_REFINEMENT", "single T1 profile authorized")
    state.register_terminal_refinement_profile("knots8_pop96_rep4_iter8_elite12")
    failures.append(
        {
            "phase": "TERMINAL_REFINEMENT",
            "result": "TRAINING_4_OF_4_BUT_FORMAL_15_OF_20",
            "recovery": "use only authorized global fallback",
        }
    )
    move("GLOBAL_OPTIMIZER_FALLBACK", "T1 formal replay failed 16/20 gate")
    state.register_global_optimizer_upgrade(reason="single authorized knots32 fallback")
    failures.append(
        {
            "phase": "GLOBAL_OPTIMIZER_FALLBACK",
            "result": "FORMAL_12_OF_20_AND_GEOMETRY_BLOCKED",
            "recovery": "stop optimization; budget exhausted",
        }
    )
    move("DEMONSTRATIONS", "both per-clip trajectory gates blocked")
    move("BC", "not authorized; zero demonstrations")
    move("PPO_BENCHMARK", "not authorized before a qualified trajectory")
    move("SINGLE_PPO", "not authorized; zero samples and checkpoints")
    move("TWO_CLIP_PPO", "both single PPO validations absent")
    move("V2_EXPORT", "no validated PPO episodes")
    move("SENSITIVITY", "nominal PPO absent")
    failures.append(
        {
            "phase": "CLOSEOUT",
            "failure": "VISUAL_DASHBOARD_SINGLE_ROOT_REJECTED",
            "recovery": "rerun with both required clip roots; no formal metric changed",
        }
    )
    move("CLOSEOUT", "bounded evidence complete; no further optimizer profile authorized")
    return state.as_dict(), failures


def _guarded_outputs(trajectories: dict[str, dict[str, Any]]) -> None:
    decisions = []
    for clip, qualification in trajectories.items():
        decisions.append(
            {
                "clip": clip,
                "trajectory_status": qualification["status"],
                "geometry_pass": qualification["geometry_pass"],
                "authorization": "PPO_NOT_AUTHORIZED_FOR_CLIP",
                "blockers": qualification["blockers"],
            }
        )
    _write(
        REPORT_ROOT / "demonstration_manifest.json",
        {
            "schema_version": "PhysicsCorrectionDemonstrationManifestV1",
            "status": "STAGE16D_DEMONSTRATION_EXPORT_NOT_AUTHORIZED",
            "trajectory_level_split": {"train": 0.8, "validation": 0.2},
            "frame_level_random_split": False,
            "trajectories": [],
            "decisions": decisions,
        },
    )
    for suffix in ("170105", "170650"):
        clip = f"hocap_{suffix}"
        _write(
            REPORT_ROOT / f"bc_training_{suffix}.json",
            {
                "schema_version": "Stage16DActorOnlyBCTrainingV1",
                "clip": clip,
                "status": f"STAGE16D_{suffix}_BC_NOT_RUN",
                "reason": "per-clip trajectory gate blocked",
                "epochs": 0,
                "checkpoints": [],
                "demonstration_trajectories": 0,
            },
        )
        training = {
            "schema_version": "Stage16DSingleClipPPOTrainingV2",
            "clip": clip,
            "status": f"STAGE16D_{suffix}_PPO_NOT_RUN",
            "reason": "formal trajectory entry gate did not authorize PPO",
            "samples": 0,
            "workers_started": 0,
            "seeds": [],
            "lr_fallbacks": 0,
            "checkpoints": [],
            "sample_ladder": [1_048_576, 4_194_304, 16_777_216, 67_108_864],
        }
        _write(REPORT_ROOT / f"ppo_training_{suffix}.json", training)
        _write(
            REPORT_ROOT / f"ppo_evaluation_{suffix}.json",
            {
                "schema_version": "Stage16DSingleClipPPOEvaluationV2",
                "clip": clip,
                "status": f"STAGE16D_{suffix}_PPO_NOT_RUN",
                "episodes": 0,
                "checkpoint": None,
                "qualification": "NOT_RUN_TRAJECTORY_GATE_BLOCKED",
            },
        )
    _write(
        REPORT_ROOT / "ppo_env_benchmark.json",
        {
            "schema_version": "Stage16DPPOEnvironmentBenchmarkV1",
            "status": "STAGE16D_PPO_ENV_BENCHMARK_NOT_RUN_GATE_BLOCKED",
            "requested_env_counts": [1024, 2048, 4096],
            "executed_env_counts": [],
            "reason": "no per-clip trajectory was PPO-authorized",
        },
    )
    _write(
        REPORT_ROOT / "two_clip_ppo.json",
        {
            "schema_version": "Stage16DTwoClipPPOV2",
            "status": "STAGE16D_TWO_CLIP_PPO_NOT_RUN",
            "samples": 0,
            "checkpoints": [],
            "reason": "both single-clip PPO validations are absent",
        },
    )
    _write(
        REPORT_ROOT / "v2_data_inventory.json",
        {
            "schema_version": "Stage16DPhysicsQualifiedIsaacTrajectoryV2Inventory",
            "status": "STAGE16D_PHYSICS_DATA_V2_BLOCKED",
            "artifacts": [],
            "reason": "unvalidated trajectories and no validated PPO episodes",
            "v1_artifacts_are_not_v2": True,
        },
    )
    _write(
        REPORT_ROOT / "sensitivity_audit.json",
        {
            "schema_version": "Stage16DSensitivityAuditV2",
            "status": "STAGE16D_SENSITIVITY_NOT_RUN",
            "reason": "nominal PPO did not run or validate",
            "runs": [],
        },
    )


def _validate_preserved_guard_outputs() -> None:
    expected = {
        "demonstration_manifest.json": "STAGE16D_DEMONSTRATION_EXPORT_NOT_AUTHORIZED",
        "bc_training_170105.json": "STAGE16D_BC_NOT_RUN",
        "bc_training_170650.json": "STAGE16D_BC_NOT_RUN",
        "ppo_training_170105.json": "STAGE16D_170105_PPO_NOT_RUN",
        "ppo_training_170650.json": "STAGE16D_170650_PPO_NOT_RUN",
        "ppo_evaluation_170105.json": "STAGE16D_170105_PPO_NOT_RUN",
        "ppo_evaluation_170650.json": "STAGE16D_170650_PPO_NOT_RUN",
        "two_clip_ppo.json": "STAGE16D_TWO_CLIP_PPO_NOT_RUN",
        "v2_data_inventory.json": "STAGE16D_PHYSICS_DATA_PARTIAL_BLOCKED",
        "sensitivity_audit.json": "STAGE16D_SENSITIVITY_NOT_RUN",
    }
    for name, status in expected.items():
        payload = _read(REPORT_ROOT / name)
        if payload.get("status") != status:
            raise RuntimeError(f"STAGE16D_GUARD_EVIDENCE_STATUS_DRIFT:{name}")
        if int(payload.get("samples", 0)) != 0:
            raise RuntimeError(f"STAGE16D_GUARD_EVIDENCE_SAMPLE_DRIFT:{name}")
        checkpoints = payload.get("checkpoints", [])
        if checkpoints:
            raise RuntimeError(f"STAGE16D_GUARD_EVIDENCE_CHECKPOINT_DRIFT:{name}")


def _metric_row(clip: str, kind: str, data: dict[str, Any]) -> str:
    return (
        f"| {clip} | {kind} | {1e3 * data['max_penetration_m']:.6f} | "
        f"{1e3 * data['all_frame_p95_penetration_m']:.6f} | "
        f"{1e3 * data['p95_penetration_m']:.6f} | {data['over_3mm_frame_replica_count']} | "
        f"`{data['worst_pair']}` |"
    )


def _handoff(
    trajectories: dict[str, dict[str, Any]], tests: dict[str, Any], commits: dict[str, Any]
) -> str:
    manifest = _read(REPORT_ROOT / "runtime_collision_geometry_manifest.json")
    backend = _read(REPORT_ROOT / "geometry_query_backend_contract.json")
    metric = _read(REPORT_ROOT / "geometry_metric_contract.json")
    g105 = _read(REPORT_ROOT / "geometry_qualification_170105_terminal_refined.json")
    g650 = _read(REPORT_ROOT / "geometry_qualification_170650.json")
    visual = _read(REPORT_ROOT / "visual_proxy_diagnostics.json")
    terminal = _read(REPORT_ROOT / "terminal_failure_analysis_170105.json")
    refinement = _read(REPORT_ROOT / "terminal_refinement_170105.json")
    global_result = _read(REPORT_ROOT / "trajectory_requalification_170105_global.json")
    shape_rows = [
        (
            "Wuji hand",
            len(manifest["hand_shapes"]),
            "convex_hull",
            len({row["geometry_sha256"] for row in manifest["hand_shapes"]}),
        ),
        (
            "hocap_170105",
            len(manifest["object_shapes"]["hocap_170105"]),
            "convex_hull",
            len({row["geometry_sha256"] for row in manifest["object_shapes"]["hocap_170105"]}),
        ),
        (
            "hocap_170650",
            len(manifest["object_shapes"]["hocap_170650"]),
            "convex_hull",
            len({row["geometry_sha256"] for row in manifest["object_shapes"]["hocap_170650"]}),
        ),
    ]
    geometry_table = [
        "| Asset | Shapes | Types | Unique hashes | Scale | Transform validation |",
        "|---|---:|---|---:|---|---|",
        *[
            f"| {asset} | {count} | {kind} | {hashes} | positive, authored | finite, FK crosschecked |"
            for asset, count, kind, hashes in shape_rows
        ],
    ]
    metric_table = [
        "| Clip | Trajectory | Max mm | all-frame P95 mm | contact-active P95 mm | >3 mm samples | Worst pair |",
        "|---|---|---:|---:|---:|---:|---|",
        _metric_row("170105", "source", g105["source"]),
        _metric_row("170105", "terminal refined", g105["corrected"]),
        _metric_row("170650", "source", g650["source"]),
        _metric_row("170650", "corrected", g650["corrected"]),
    ]
    command_lines = [
        "python scripts/rl/isaaclab/capture_stage16d_metric_preflight.py --run-smoke --accept-eula",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py --phase inventory --accept-eula",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py --phase backend",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py --phase audit",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/audit_stage16d_runtime_collision_geometry.py --phase candidate --clip hocap_170105 --corrected-trace .local/reports/stage16d_metric_qualification_and_ppo/trajectory_trace_170105_terminal_refined.npz --candidate-label terminal_refined",
        "python scripts/rl/isaaclab/analyze_stage16d_terminal_failures.py --clip hocap_170105 --output .local/reports/stage16d_metric_qualification_and_ppo/terminal_failure_analysis_170105.json --csv-output .local/reports/stage16d_metric_qualification_and_ppo/successful_vs_failed_170105.csv",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/refine_stage16d_terminal_tail.py --clip hocap_170105 --actions .local/reports/stage16d_physics_consistent_retargeting/optimizer_170105_s3.actions.npy --failure-analysis .local/reports/stage16d_metric_qualification_and_ppo/terminal_failure_analysis_170105.json --output .local/reports/stage16d_metric_qualification_and_ppo/terminal_refinement_170105.json --accept-eula",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/qualify_stage16d_trajectory.py --clip hocap_170105 --actions .local/reports/stage16d_metric_qualification_and_ppo/terminal_refinement_170105.actions.npy --output .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170105.json --trace .local/reports/stage16d_metric_qualification_and_ppo/trajectory_trace_170105_terminal_refined.npz --replicas 20 --accept-eula",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/optimize_stage16d_physics_trajectory.py --accept-eula --stage d3-s3 --clip hocap_170105 --knots 32 --population 96 --replicas 4 --iterations 8 --elites 12 --output .local/reports/stage16d_metric_qualification_and_ppo/optimizer_upgrade_170105.json",
        "python scripts/rl/isaaclab/reextract_stage16d_semantics.py",
        "python scripts/rl/isaaclab/build_stage16d_demonstrations.py --qualification .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170105.json --qualification .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170650.json --geometry .local/reports/stage16d_metric_qualification_and_ppo/geometry_qualification_170105_terminal_refined.json --geometry .local/reports/stage16d_metric_qualification_and_ppo/geometry_qualification_170650.json --output .local/reports/stage16d_metric_qualification_and_ppo/demonstration_manifest.json",
        "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/optimize_stage16d_physics_trajectory.py --accept-eula --stage env-smoke --clip hocap_170650 --num-envs 1024 --steps 321 --output .local/reports/stage16d_metric_qualification_and_ppo/ppo_env_benchmark_1024.json",
        "python scripts/rl/isaaclab/train_stage16d_single_ppo.py --clip hocap_170650 --qualification .local/reports/stage16d_metric_qualification_and_ppo/trajectory_requalification_170650.json --geometry .local/reports/stage16d_metric_qualification_and_ppo/geometry_qualification_170650.json --bc-output .local/reports/stage16d_metric_qualification_and_ppo/bc_training_170650.json --output .local/reports/stage16d_metric_qualification_and_ppo/ppo_training_170650.json",
        "python scripts/rl/isaaclab/train_stage16d_two_clip_ppo.py --evaluation .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170105.json --evaluation .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170650.json --output .local/reports/stage16d_metric_qualification_and_ppo/two_clip_ppo.json",
        "python scripts/rl/isaaclab/export_stage16d_physics_data.py --trajectory-root .local/physics_consistent_retargeting --ppo-evaluation .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170650.json --output .local/reports/stage16d_metric_qualification_and_ppo/v2_data_inventory.json",
        "python scripts/rl/isaaclab/audit_stage16d_sensitivity.py --ppo-evaluation .local/reports/stage16d_metric_qualification_and_ppo/ppo_evaluation_170650.json --output .local/reports/stage16d_metric_qualification_and_ppo/sensitivity_audit.json",
        "python scripts/rl/isaaclab/visualize_stage16d_physics_retargeting.py --trajectory optimized --mode headless --clip-root .local/physics_consistent_retargeting/hocap_170105 --clip-root .local/physics_consistent_retargeting/hocap_170650 --show-reference-hand --show-corrected-hand --show-source-object --show-corrected-object --show-contacts --show-contact-topology --show-penetration --show-task-progress --output-dashboard .local/reports/stage16d_metric_qualification_and_ppo/dashboard.html --output-review .local/reports/stage16d_metric_qualification_and_ppo/visual_review.json",
    ]
    sections = [
        (
            "Final Status",
            "`STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE`. Runtime metric validated; both per-clip trajectories blocked; BC/PPO/two-clip/V2/sensitivity not run.",
        ),
        (
            "Git and Environment",
            f"Branch `{commits['branch']}`; start `{START_HEAD}`; final `{commits['final_head']}`. Isaac 100-step full smoke: PASS.",
        ),
        (
            "Frozen Stage16D Baseline",
            "The prior reports, configs, manifests, assets, source traces, corrected traces, and hashes were archived before formal geometry queries; frozen inputs were not overwritten.",
        ),
        ("Runtime Collision Geometry Manifest", "\n".join(geometry_table)),
        (
            "Convex Query Backend Qualification",
            f"`{backend['backend']}=={backend['backend_version']}`; {backend['algorithm']}; tolerance `{backend['numerical_tolerance_m']}` m; metric epsilon `{backend['metric_epsilon_m']}` m. Maximum-iteration control is `{backend['max_iterations']}` because this binding does not expose it. Analytic tests 13/13 and all formal queries converged.",
        ),
        (
            "Formal Penetration Metric Contract",
            f"`{metric['schema_version']}`: positive separation, zero touching, negative overlap; penetration is `max(0,-signed)`. Per-frame value is the worst allowed hand-object pair. Formal p95 uses only contact-active per-frame-worst samples; all-frame p95 is diagnostic. Corrected max and p95 must each be no greater than source×1.10+`{backend['metric_epsilon_m']}` m; max <10 mm and p95 <=3 mm.",
        ),
        (
            "Source Runtime-Proxy Audit",
            "Source audits used the exact same 21×1 runtime convex-pair set, transforms, scale, backend, sign convention, tolerance, and aggregation as corrected trajectories.",
        ),
        (
            "Corrected Runtime-Proxy Audit",
            "\n".join(metric_table)
            + "\n\nBoth clips pass absolute thresholds but fail both source-relative thresholds.",
        ),
        (
            "Source-vs-Corrected Comparability",
            "The metrics are directly comparable. Runtime FK pose crosschecks and PhysX contact/sign checks passed. The old lower-bound-only audit is superseded, not rewritten.",
        ),
        (
            "Visual-Mesh Diagnostics",
            f"Unsigned triangle-surface diagnostics only; formal authority is false. Visual meshes are non-watertight. Results: `{json.dumps(visual['clips'], sort_keys=True)}`.",
        ),
        (
            "170105 Terminal Failure Analysis",
            f"Failed replicas `{terminal['failed_replicas']}` were classified terminal object twist; semantic/contact/causality gates passed.",
        ),
        (
            "170105 Refinement",
            f"T1 exact 8-knot profile ran {refinement['wall_time_s']:.3f}s: optimizer-internal 4/4, formal replay 15/20. The only global fallback then produced {round(20 * global_result['success_rate'])}/20; no further optimizer upgrade is authorized.",
        ),
        (
            "Per-Clip Trajectory Qualification",
            f"170105: {trajectories['hocap_170105']['success_count']}/20 plus geometry fail. 170650: {trajectories['hocap_170650']['success_count']}/20 but geometry fail. Both are `PARTIAL_BLOCKED`.",
        ),
        (
            "Demonstration Dataset and BC",
            "Not authorized for either clip. Demonstration trajectories=0, BC epochs=0, checkpoints=0. The 80/20 trajectory/replica split contract is recorded but not executed.",
        ),
        (
            "170105 Single-Clip PPO",
            "`STAGE16D_170105_PPO_NOT_RUN`; samples=0, checkpoints=0, workers=0.",
        ),
        (
            "170650 Single-Clip PPO",
            "`STAGE16D_170650_PPO_NOT_RUN`; samples=0, checkpoints=0, workers=0. A 20/20 empirical replay does not override the failed geometry gate.",
        ),
        (
            "Two-Clip PPO",
            "`STAGE16D_TWO_CLIP_PPO_NOT_RUN`; samples=0, checkpoints=0 because neither single-clip PPO validated.",
        ),
        (
            "PhysicsQualifiedIsaacTrajectoryV2",
            "Not exported; artifact paths and hashes are absent by design. Existing V1 packages remain partial and are not relabeled.",
        ),
        ("Sensitivity Audit", "Not run because nominal PPO never became authorized or validated."),
        (
            "Failure-Recovery",
            "hpp-fcl non-finite overlap was rejected; one fixed python-fcl backend passed. An invalid all-frame-p95 draft was invalidated before formal acceptance and all results rerun. T1 and the only global fallback failed formal 20-replica gates. Optimization and training stopped.",
        ),
        ("Commands", "```text\n" + "\n".join(command_lines) + "\n```"),
        ("Tests", f"`{json.dumps(tests, sort_keys=True)}`"),
        (
            "README and Roadmap",
            "English/Chinese status and the Stage16D PPO document are synchronized to the metric-compatible blocked result.",
        ),
        (
            "Local Commits",
            f"Local commits: `{commits['commits']}`. PUSHED=NO; PR_CREATED=NO; MAIN_MERGED=NO; TAG_CREATED=NO; RELEASE_CREATED=NO.",
        ),
        (
            "Remaining Limitations",
            "Collision proxies are not visual truth. Factor-8 changes time semantics. The free object trajectory is corrected by physics. The virtual wrist is not a real robot arm. Physics parameters are uncalibrated. No real dynamics or sim-to-real conclusion is supported. Failed PPO cannot be exported as V2.",
        ),
        (
            "Recommended Next Action",
            "Do not run more optimizer profiles under this frozen budget. Design a new, separately authorized correction objective that explicitly penalizes the frozen RuntimeCollisionProxyPenetrationV1 source-relative excess while preserving terminal stability, then restart qualification from frame-zero replicas. Real-arm integration and PPO remain downstream.",
        ),
    ]
    lines = ["# Stage 16-D Runtime Geometry Qualification and PPO Handoff", ""]
    for index, (title, body) in enumerate(sections, start=1):
        lines.extend((f"## {index}. {title}", "", body, ""))
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    trajectories = _integrated_trajectory_reports()
    recovery, failures = _recovery_and_failures()
    _write(REPORT_ROOT / "recovery_state.json", recovery)
    (REPORT_ROOT / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures),
        encoding="utf-8",
    )
    if args.preserve_guard_evidence:
        _validate_preserved_guard_outputs()
    else:
        _guarded_outputs(trajectories)
    terminal = _read(REPORT_ROOT / "terminal_refinement_170105.json")
    global_opt = _read(REPORT_ROOT / "optimizer_upgrade_170105.json")
    _write(
        REPORT_ROOT / "resource_usage.json",
        {
            "schema_version": "Stage16DMetricPPOResourceUsageV1",
            "isaac_platform_smoke_100_steps": "PASS",
            "terminal_refinement_wall_time_s": terminal["wall_time_s"],
            "global_optimizer_wall_time_s": global_opt["wall_time_s"],
            "formal_geometry_queries": {
                "baseline": 321 * 20 * 21 * 2,
                "source": 321 * 21 * 2,
                "additional_170105_candidates": 321 * 20 * 21 * 2,
            },
            "ppo_samples": {"hocap_170105": 0, "hocap_170650": 0, "two_clip": 0},
            "ppo_checkpoints": [],
            "unrelated_processes_terminated": 0,
        },
    )
    visual = _read(REPORT_ROOT / "visual_proxy_diagnostics.json")
    _write(
        REPORT_ROOT / "visual_review.json",
        {
            "schema_version": "Stage16DMetricVisualReviewV1",
            "status": "NUMERICAL_HTML_DASHBOARD_GENERATED",
            "formal_gate_authority": False,
            "raster_render": "NOT_AVAILABLE_NOT_REQUIRED_FOR_NUMERICAL_GATE",
            "diagnostics": visual,
            "dashboard": ".local/reports/stage16d_metric_qualification_and_ppo/dashboard.html",
        },
    )
    tests = {
        "schema_version": "Stage16DMetricPPOTestSummaryV1",
        "ruff_check": args.ruff,
        "ruff_format_check": args.format,
        "mypy": args.mypy,
        "pytest": args.pytest,
        "pytest_summary": args.pytest_summary,
        "paper_fidelity": args.paper_fidelity,
        "isaac_platform_smoke_100_steps": "PASS",
        "geometry_analytic_tests": "13/13 PASS",
        "formal_geometry_query_nonconvergence": 0,
    }
    _write(REPORT_ROOT / "tests.json", tests)
    commits = {
        "schema_version": "Stage16DMetricPPOGitCloseoutV1",
        "branch": _git("branch", "--show-current"),
        "start_head": START_HEAD,
        "final_head": _git("rev-parse", "HEAD"),
        "commits": _git("log", "--oneline", f"{START_HEAD}..HEAD").splitlines(),
        "pushed": False,
        "pr_created": False,
        "main_merged": False,
        "tag_created": False,
        "release_created": False,
    }
    _write(REPORT_ROOT / "git_commits.json", commits)
    summary = {
        "schema_version": "Stage16DRuntimeGeometryQualificationAndPPOCloseoutV1",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": commits["branch"],
        "start_head": START_HEAD,
        "final_head": commits["final_head"],
        "geometry_backend": "python-fcl==0.7.0.11",
        "metric_contract": "RuntimeCollisionProxyPenetrationV1",
        "runtime_geometry_metric": "STAGE16D_RUNTIME_GEOMETRY_METRIC_VALIDATED",
        "trajectories": trajectories,
        "ppo": {
            "hocap_170105": "STAGE16D_170105_PPO_NOT_RUN",
            "hocap_170650": "STAGE16D_170650_PPO_NOT_RUN",
            "two_clip": "STAGE16D_TWO_CLIP_PPO_NOT_RUN",
        },
        "v2_export": "STAGE16D_PHYSICS_DATA_V2_BLOCKED",
        "sensitivity": "STAGE16D_SENSITIVITY_NOT_RUN",
        "overall": "STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE",
        "blockers": {
            "hocap_170105": trajectories["hocap_170105"]["blockers"],
            "hocap_170650": trajectories["hocap_170650"]["blockers"],
        },
        "actual_ppo_samples": 0,
        "actual_ppo_checkpoints": [],
        "actual_v2_paths": [],
    }
    _write(REPORT_ROOT / "final_summary.json", summary)
    handoff = _handoff(trajectories, tests, commits)
    (REPORT_ROOT / "handoff.md").write_text(handoff, encoding="utf-8")
    (REPORT_ROOT / "final_summary.md").write_text(
        "# Stage 16-D Runtime Geometry Qualification and PPO\n\n"
        "`STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE`\n\n"
        "The runtime collision metric is validated, but both corrected trajectories fail "
        "the frozen source-relative geometry gates. 170105 also remains 15/20 after the "
        "single terminal repair; its only global fallback regressed to 12/20. BC/PPO, "
        "two-clip PPO, V2 export, and sensitivity therefore remain not run with zero samples.\n",
        encoding="utf-8",
    )
    dashboard_rows = []
    for clip in ("170105", "170650"):
        q = _read(REPORT_ROOT / f"geometry_qualification_{clip}.json")
        dashboard_rows.append(
            "<tr>"
            f"<td>{clip}</td><td>{1e3 * q['source']['max_penetration_m']:.4f}</td>"
            f"<td>{1e3 * q['corrected']['max_penetration_m']:.4f}</td>"
            f"<td>{1e3 * q['corrected']['p95_penetration_m']:.4f}</td>"
            f"<td>{html.escape(q['status'])}</td></tr>"
        )
    dashboard = (
        "<!doctype html><meta charset='utf-8'><title>Stage16D metric qualification</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.5rem}"
        "code{background:#eee;padding:.15rem}</style><h1>Stage 16-D Runtime Geometry</h1>"
        "<p><code>STAGE16D_BLOCKED_WITH_BOUNDED_EVIDENCE</code></p>"
        "<p>Formal authority: runtime collision proxies. Visual distances are unsigned diagnostics only.</p>"
        "<table><thead><tr><th>Clip</th><th>Source max mm</th><th>Corrected max mm</th>"
        "<th>Corrected active P95 mm</th><th>Result</th></tr></thead><tbody>"
        + "".join(dashboard_rows)
        + "</tbody></table><p>See <code>handoff.md</code> and the pairwise NPZ/Parquet timelines.</p>"
    )
    (REPORT_ROOT / "dashboard.html").write_text(dashboard, encoding="utf-8")
    print(json.dumps({"overall": summary["overall"], "report_root": str(REPORT_ROOT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
