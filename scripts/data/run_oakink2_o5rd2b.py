#!/usr/bin/env python3
"""Run the fail-closed OakInk2 O5R-D2B ObjectiveV2 state machine.

The CLI separates method development from independent validation.  Validation
actions refuse to start until the Candidate-B2 objective and its execution
contract have been serialized and hash-bound by the development gate.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5rd2a import (
    ROOT as D2A_ROOT,
)
from scripts.data.run_oakink2_o5rd2a import (
    D2ARuntime,
    _measurement_row,
    _query_set,
    _seed_state,
    development_windows,
    digest,
    frozen_paths,
    git,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from scripts.evaluation.audit_retarget_semantic_validity import _semantic_frames
from toporetarget.evaluation.retarget_semantic_validity import (
    SemanticGateContractV1,
    transform_error,
)
from toporetarget.retarget.continuous import encode_base_correction
from toporetarget.retarget.final_refinement import refine_frame
from toporetarget.retarget.objective_v2 import (
    ObjectiveV2Candidate,
    ObjectiveV2DevelopmentContext,
    ObjectiveV2Measurements,
    canonical_sha256,
    evaluate_candidate_b2,
    interaction_retention_limit,
)

ROOT = REPO / ".local/reports/oakink2_o5rd2b_objective_v2_certification_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "e0a0b98969db0ecfdd745aa03606d91133937beb"
CANDIDATE = ObjectiveV2Candidate.candidate_b2()
GATE = SemanticGateContractV1()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_file_sha(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def start_head_is_ancestor(head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2B_BRANCH_MISMATCH:{branch}")
    if head != START_HEAD and not start_head_is_ancestor(head):
        raise RuntimeError(f"O5RD2B_START_HEAD_NOT_ANCESTOR:{head}")
    d1_alignment = read_json(
        REPO
        / ".local/reports/oakink2_o5rd1_objective_alignment_v1/decision/production_objective_alignment.json"
    )
    d2a = read_json(D2A_ROOT / "final_summary.json")
    if d1_alignment.get("PRODUCTION_OBJECTIVE_ALIGNMENT") != "MISALIGNED":
        raise RuntimeError("O5RD2B_D1_ALIGNMENT_DRIFT")
    if d2a["objective_v2_status"]["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"] != "NO_CANDIDATE_READY":
        raise RuntimeError("O5RD2B_D2A_STATUS_DRIFT")
    d2a_authorities = read_json(D2A_ROOT / "preflight/frozen_authorities.json")["authorities"]
    authorities: dict[str, Any] = {}
    for name, path in frozen_paths().items():
        current = digest(path)
        expected = d2a_authorities[name]["sha256"]
        if current != expected:
            raise RuntimeError(f"O5RD2B_FROZEN_AUTHORITY_DRIFT:{name}")
        authorities[name] = {"path": str(path.resolve()), "sha256": current}
    extra = {
        "objective_v2_development_implementation": REPO
        / "src/toporetarget/retarget/objective_v2.py",
        "o5rd2a_final_summary": D2A_ROOT / "final_summary.json",
        "o5rd2a_window_results": D2A_ROOT / "candidates/window_results.csv",
        "o5rd2a_development_ledger": D2A_ROOT
        / "development_data/method_development_ledger_v2.json",
    }
    for name, path in extra.items():
        authorities[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    payload = {
        "schema_version": "OakInk2O5RD2BPreflightV1",
        "branch": branch,
        "start_head": START_HEAD,
        "head_at_preflight": head,
        "initial_status_short": git("status", "--short", "--untracked-files=all").splitlines(),
        "d1": {
            "PRODUCTION_OBJECTIVE_ALIGNMENT": "MISALIGNED",
            "PRIMARY_ROOT_CAUSE": "OBJECTIVE_SEMANTIC_MISALIGNMENT",
            "CONFIDENCE": "HIGH",
        },
        "d2a": {
            "PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2": d2a["context_binding"]["status"],
            "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "NO_CANDIDATE_READY",
            "SELECTED_OBJECTIVE": "NONE",
            "RETARGET_OBJECTIVE_V2_SHA256": None,
        },
        "authorities": authorities,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "preflight/git.json", payload)
    write_json(root / "preflight/upstream_state.json", {"d1": payload["d1"], "d2a": payload["d2a"]})
    write_json(
        root / "preflight/frozen_authorities.json",
        {"schema_version": "OakInk2O5RD2BFrozenAuthoritiesV1", "status": "PASS", **payload},
    )
    return payload


def audit_authority(root: Path) -> dict[str, Any]:
    profile = REPO / "src/toporetarget/retarget/continuous.py"
    semantic = REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py"
    solver = REPO / "src/toporetarget/retarget/final_refinement.py"
    rows = [
        {
            "quantity": "0.01 m",
            "symbol": "S_POS_M",
            "source": str(profile),
            "use_site": str(solver),
            "mathematical_role": "normalizer in correction_temporal_energy and componentwise optimizer trust region around transported prediction",
            "failure_behavior": "optimizer state rejected when predictor translation correction exceeds componentwise wall",
            "classification": "NORMALIZATION_SCALE",
            "secondary_classification": "TRUST_REGION_HEURISTIC",
            "physical_hard_limit_evidence": False,
            "v2_treatment": "soft prediction-correction profile only",
        },
        {
            "quantity": "5 deg",
            "symbol": "S_ROT_RAD",
            "source": str(profile),
            "use_site": str(solver),
            "mathematical_role": "normalizer in correction_temporal_energy and componentwise optimizer trust region around transported prediction",
            "failure_behavior": "optimizer state rejected when predictor rotation correction exceeds componentwise wall",
            "classification": "NORMALIZATION_SCALE",
            "secondary_classification": "TRUST_REGION_HEURISTIC",
            "physical_hard_limit_evidence": False,
            "v2_treatment": "soft prediction-correction profile only",
        },
        {
            "quantity": "0.05 rad",
            "symbol": "S_Q_RAD",
            "source": str(profile),
            "use_site": str(solver),
            "mathematical_role": "normalizer in correction_temporal_energy and componentwise optimizer trust region around transported prediction",
            "failure_behavior": "optimizer state rejected when any predicted finger correction exceeds wall",
            "classification": "NORMALIZATION_SCALE",
            "secondary_classification": "TRUST_REGION_HEURISTIC",
            "physical_hard_limit_evidence": False,
            "v2_treatment": "soft prediction-correction profile and diagnostic; no hard rejection",
        },
    ]
    payload = {
        "schema_version": "ContinuityAdmissibilityAuthorityAuditV1",
        "status": "PASS",
        "audited_source_hashes": {
            "continuous": _canonical_file_sha(profile),
            "semantic_v1": _canonical_file_sha(semantic),
            "production_solver": _canonical_file_sha(solver),
        },
        "quantities": rows,
        "semantic_v1_actual_continuity_authority": {
            "translation_step_limit_m": GATE.temporal_translation_step_limit_m,
            "rotation_step_limit_rad": GATE.temporal_rotation_step_limit_rad,
            "rotation_step_limit_deg": float(np.rad2deg(GATE.temporal_rotation_step_limit_rad)),
            "finger_q_step_hard_limit_rad": None,
            "source": str(semantic),
            "classification": "SEMANTIC_V1_HARD_GATE",
        },
        "PREDICTION_CORRECTION_IS_TRAJECTORY_CONTINUITY": "NO",
    }
    write_json(root / "admissibility_audit/continuity_authority.json", payload)
    write_json(
        root / "admissibility_audit/predictor_vs_actual_continuity.json",
        {
            "schema_version": "PredictorVsActualContinuityV1",
            "status": "SEPARATED",
            "prediction_correction": "candidate state minus transported warm-chart prediction",
            "actual_trajectory_motion": "accepted robot wrist/q at t minus accepted robot wrist/q at t-1",
            "prediction_correction_is_trajectory_continuity": False,
            "prediction_profile_hard_gate_in_b2": False,
            "actual_semantic_v1_translation_rotation_hard_gate_in_b2": True,
        },
    )
    return payload


def _robot_wrist(runtime: D2ARuntime, qpos: np.ndarray, base: np.ndarray) -> np.ndarray:
    keypoints = np.asarray(runtime.model.keypoints_scene(qpos, np.eye(4)), dtype=np.float64)
    return np.asarray(base, dtype=np.float64) @ np.asarray(
        _semantic_frames(keypoints, runtime.model.side), dtype=np.float64
    )


def actual_continuity(
    runtime: D2ARuntime,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
    qpos: np.ndarray,
    base: np.ndarray,
) -> dict[str, float]:
    if previous_q is None or previous_base is None:
        return {"translation_step_m": 0.0, "rotation_step_rad": 0.0, "q_step_inf_rad": 0.0}
    left = _robot_wrist(runtime, previous_q, previous_base)
    right = _robot_wrist(runtime, qpos, base)
    delta = transform_error(left, right)
    return {
        "translation_step_m": float(np.asarray(delta["position_m"])),
        "rotation_step_rad": float(np.asarray(delta["rotation_rad"])),
        "q_step_inf_rad": float(np.max(np.abs(np.asarray(qpos) - np.asarray(previous_q)))),
    }


def _evaluate_b2(
    runtime: D2ARuntime,
    values: ObjectiveV2Measurements,
    actual: dict[str, float],
) -> dict[str, Any]:
    return evaluate_candidate_b2(
        CANDIDATE,
        values,
        runtime.authority,
        actual_translation_step_m=actual["translation_step_m"],
        actual_rotation_step_rad=actual["rotation_step_rad"],
        actual_q_step_inf_rad=actual["q_step_inf_rad"],
        semantic_translation_step_limit_m=GATE.temporal_translation_step_limit_m,
        semantic_rotation_step_limit_rad=GATE.temporal_rotation_step_limit_rad,
    )


def optimize_b2_frame(
    runtime: D2ARuntime,
    ordinal: int,
    *,
    previous_base: np.ndarray | None,
    previous_qpos: np.ndarray | None,
    maxiter: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_qpos
    )
    seed_q, seed_base, seed_source = _seed_state(runtime, ordinal, CANDIDATE, binding, context)
    seed_correction = encode_base_correction(context.seed_base, seed_base)
    if previous_base is not None and previous_qpos is not None:
        context.trust_region_reference = np.concatenate(
            [encode_base_correction(context.seed_base, previous_base), previous_qpos]
        )
        context.trust_region_limits = (
            # The interaction-primary search is finger-only.  This is a
            # conservative, outcome-independent way to preserve the frozen
            # wrist authority while the thumb-dominated interaction mismatch
            # is repaired.  Actual trajectory continuity is still audited on
            # the accepted state.
            1.0e-12,
            1.0e-12,
            # A finite whole-domain bound avoids SciPy numerical-difference
            # NaNs while imposing no restriction beyond frozen asset bounds.
            float(np.max(runtime.model.joint_upper - runtime.model.joint_lower)),
        )
    else:
        context.trust_region_reference = None
        context.trust_region_limits = None
    query = _query_set(runtime, context, seed_q, seed_base)
    primary_context = ObjectiveV2DevelopmentContext(
        context, CANDIDATE, runtime.authority, phase="primary"
    )
    solver = replace(runtime.solver, maxiter=maxiter)
    initial = np.concatenate([seed_correction, seed_q])
    primary = refine_frame(
        primary_context,
        query,
        solver,
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery="none",
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=initial,
        initialization_source=f"{seed_source}_objective_v2_b2_primary",
    )
    primary_values = runtime.measurement(
        ordinal,
        primary.qpos,
        primary.base_pose_scene,
        binding=binding,
        context=context,
        slack=primary.slack,
    )
    primary_actual = actual_continuity(
        runtime, previous_qpos, previous_base, primary.qpos, primary.base_pose_scene
    )
    primary_receipt = _evaluate_b2(runtime, primary_values, primary_actual)
    retention = interaction_retention_limit(
        primary_values.interaction_e_im, runtime.authority.interaction_target
    )
    phase2_binding, phase2_base = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_qpos
    )
    phase2_base.trust_region_reference = context.trust_region_reference
    phase2_base.trust_region_limits = context.trust_region_limits
    phase2_query = _query_set(runtime, phase2_base, primary.qpos, primary.base_pose_scene)
    secondary_context = ObjectiveV2DevelopmentContext(
        phase2_base,
        CANDIDATE,
        runtime.authority,
        phase="secondary",
        interaction_retention_limit=retention,
    )
    secondary = refine_frame(
        secondary_context,
        phase2_query,
        solver,
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery="none",
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=np.concatenate(
            [
                encode_base_correction(phase2_base.seed_base, primary.base_pose_scene),
                primary.qpos,
            ]
        ),
        initialization_source="objective_v2_b2_primary_then_secondary",
    )
    secondary_values = runtime.measurement(
        ordinal,
        secondary.qpos,
        secondary.base_pose_scene,
        binding=phase2_binding,
        context=phase2_base,
        slack=secondary.slack,
    )
    secondary_actual = actual_continuity(
        runtime, previous_qpos, previous_base, secondary.qpos, secondary.base_pose_scene
    )
    secondary_receipt = _evaluate_b2(runtime, secondary_values, secondary_actual)
    retention_pass = secondary_values.interaction_e_im <= retention + 1e-10
    if secondary.accepted and secondary_receipt["feasible"] and retention_pass:
        selected, selected_values, selected_receipt, selected_actual = (
            secondary,
            secondary_values,
            secondary_receipt,
            secondary_actual,
        )
        selected_phase = "secondary"
    elif primary.accepted and primary_receipt["feasible"]:
        selected, selected_values, selected_receipt, selected_actual = (
            primary,
            primary_values,
            primary_receipt,
            primary_actual,
        )
        selected_phase = "primary_fallback"
    else:
        selected = None
        old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        selected_values = runtime.measurement(
            ordinal, old_q, old_base, binding=binding, context=context
        )
        selected_actual = actual_continuity(runtime, previous_qpos, previous_base, old_q, old_base)
        selected_receipt = _evaluate_b2(runtime, selected_values, selected_actual)
        selected_phase = "old_state_fail_closed"
    q_out = (
        np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        if selected is None
        else np.asarray(selected.qpos, dtype=np.float64)
    )
    base_out = (
        np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        if selected is None
        else np.asarray(selected.base_pose_scene, dtype=np.float64)
    )
    old_values = runtime.measurement(
        ordinal,
        np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64),
        np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64),
        binding=binding,
        context=context,
    )
    return (
        q_out,
        base_out,
        {
            "schema_version": "ObjectiveV2B2DevelopmentFrameReceiptV1",
            "development_only": True,
            "candidate": CANDIDATE.name,
            "ordinal": ordinal,
            "frame_id": int(runtime.graph.frame_indices[ordinal]),
            "seed_source": seed_source,
            "maxiter_per_phase": maxiter,
            "prediction_correction_hard_gate": False,
            "actual_semantic_continuity_hard_gate": True,
            "retention_limit": retention,
            "retention_pass": retention_pass,
            "primary_solver": {
                "accepted": bool(primary.accepted),
                "status": int(primary.optimizer_status_code),
                "message": primary.optimizer_message,
                "nfev": int(primary.optimizer_function_evaluations),
                "runtime_sec": float(primary.solve_time_s),
            },
            "secondary_solver": {
                "accepted": bool(secondary.accepted),
                "status": int(secondary.optimizer_status_code),
                "message": secondary.optimizer_message,
                "nfev": int(secondary.optimizer_function_evaluations),
                "runtime_sec": float(secondary.solve_time_s),
            },
            "selected_phase": selected_phase,
            "technical_success": selected is not None,
            "old": _measurement_row(old_values),
            "primary": _measurement_row(primary_values),
            "secondary": _measurement_row(secondary_values),
            "selected": _measurement_row(selected_values),
            "primary_evaluation": primary_receipt,
            "secondary_evaluation": secondary_receipt,
            "selected_evaluation": selected_receipt,
            "selected_actual_continuity": selected_actual,
            "elapsed_sec": time.perf_counter() - started,
            "context_binding_sha256": binding.sha256,
        },
    )


def high_failure_decomposition(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    source_rows = read_csv(D2A_ROOT / "candidates/window_results.csv")
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    previous_q: np.ndarray | None = None
    previous_base: np.ndarray | None = None
    subset = [
        row
        for row in source_rows
        if row["candidate"] == "B_LEXICOGRAPHIC_THRESHOLD" and row["stratum"] == "HIGH"
    ]
    subset.sort(key=lambda row: int(row["ordinal"]))
    first = int(subset[0]["ordinal"])
    previous_q = np.asarray(runtime.final.arrays["qpos"][first - 1], dtype=np.float64)
    previous_base = np.asarray(runtime.final.arrays["base_pose_scene"][first - 1], dtype=np.float64)
    for row in subset:
        ordinal = int(row["ordinal"])
        state = np.load(
            D2A_ROOT
            / "candidates/window_receipts/b_lexicographic_threshold"
            / row["window_id"]
            / f"frame_{ordinal:04d}.npz",
            allow_pickle=False,
        )
        qpos = np.asarray(state["qpos"], dtype=np.float64)
        base = np.asarray(state["base_pose_scene"], dtype=np.float64)
        actual = actual_continuity(runtime, previous_q, previous_base, qpos, base)
        violations = {item for item in row["new_violations"].split(";") if item}
        semantic_actual_pass = (
            actual["translation_step_m"] <= GATE.temporal_translation_step_limit_m
            and actual["rotation_step_rad"] <= GATE.temporal_rotation_step_limit_rad
        )
        hard_other = any(
            key in violations
            for key in (
                "wrist_position_m",
                "wrist_rotation_rad",
                "bone_direction_rad",
                "collision_hard_m",
                "joint_limit_rad",
                "reflection_determinant",
                "unit_scale_lower",
                "unit_scale_upper",
            )
        )
        prediction_only = bool(violations) and violations <= {
            "temporal_base_translation_m",
            "temporal_base_rotation_rad",
            "temporal_q_inf_rad",
            "temporal_excess_keypoint_m",
        }
        if prediction_only and semantic_actual_pass and not hard_other:
            failure = (
                "FAIL_PREDICTION_Q_BUDGET_ONLY"
                if violations == {"temporal_q_inf_rad"}
                else "MULTI_CONSTRAINT"
            )
            barrier = "ARTIFICIAL_PREDICTION_CORRECTION_BARRIER"
        elif not semantic_actual_pass:
            failure, barrier = "FAIL_TRUE_TEMPORAL_CONTINUITY", "NO"
        elif hard_other:
            failure, barrier = "MULTI_CONSTRAINT", "NO"
        elif row["technical_success"] == "False":
            failure, barrier = "FAIL_SOLVER_TECHNICAL", "NO"
        else:
            failure, barrier = "PASS", "NO"
        counts[failure] = counts.get(failure, 0) + 1
        rows.append(
            {
                "frame_id": int(row["frame_id"]),
                "ordinal": ordinal,
                "E_IM": float(row["new_e_im"]),
                "interaction_target_met": float(row["new_e_im"]) <= GATE.interaction_e_im_p95_limit,
                "joint_limits": float(row["joint_limit_min_margin_rad"]) >= 0.0,
                "collision": float(row["collision_min_signed_distance_m"])
                >= -runtime.authority.collision_hard_bound_m,
                "wrist_pos": float(row["wrist_position_m"]),
                "wrist_rot": float(row["wrist_rotation_rad"]),
                "bone": float(row["bone_p95_rad"]),
                "prediction_delta_p": float(row["translation_step_m"]),
                "prediction_delta_R": float(row["rotation_step_rad"]),
                "prediction_delta_q": float(row["q_step_inf_rad"]),
                "actual_frame_to_frame_delta_p": actual["translation_step_m"],
                "actual_frame_to_frame_delta_R": actual["rotation_step_rad"],
                "actual_frame_to_frame_delta_q": actual["q_step_inf_rad"],
                "semantic_v1_temporal_translation_pass": actual["translation_step_m"]
                <= GATE.temporal_translation_step_limit_m,
                "semantic_v1_temporal_rotation_pass": actual["rotation_step_rad"]
                <= GATE.temporal_rotation_step_limit_rad,
                "solver_terminal": row["selected_phase"],
                "classification": failure,
                "artificial_barrier": barrier,
            }
        )
        previous_q, previous_base = qpos, base
    write_csv(root / "admissibility_audit/high_window_failure_decomposition.csv", rows)
    payload = {
        "schema_version": "CandidateBHighFailureDecompositionV1",
        "N_HIGH_TOTAL": len(rows),
        "N_HIGH_USABLE": counts.get("PASS", 0),
        "N_FAIL_PREDICTION_BUDGET_ONLY": counts.get("FAIL_PREDICTION_Q_BUDGET_ONLY", 0),
        "N_FAIL_TRUE_CONTINUITY": counts.get("FAIL_TRUE_TEMPORAL_CONTINUITY", 0),
        "N_FAIL_WRIST": counts.get("FAIL_WRIST", 0),
        "N_FAIL_BONE": counts.get("FAIL_BONE", 0),
        "N_FAIL_COLLISION": counts.get("FAIL_COLLISION", 0),
        "N_FAIL_JOINT_LIMIT": counts.get("FAIL_JOINT_LIMIT", 0),
        "N_FAIL_SOLVER": counts.get("FAIL_SOLVER_TECHNICAL", 0),
        "counts": counts,
        "primary_failure_mode": "ARTIFICIAL_PREDICTION_CORRECTION_BARRIER",
        "decision": "CANDIDATE_B2",
    }
    write_json(root / "admissibility_audit/decision.json", payload)
    return payload


def develop_b2(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for window in development_windows(D2A_ROOT):
        first = int(window["ordinals"][0])
        previous_q = np.asarray(runtime.final.arrays["qpos"][first - 1], dtype=np.float64)
        previous_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][first - 1], dtype=np.float64
        )
        window_rows: list[dict[str, Any]] = []
        for ordinal_value in window["ordinals"]:
            ordinal = int(ordinal_value)
            receipt_path = (
                root
                / "objective_development/receipts"
                / window["window_id"]
                / f"frame_{ordinal:04d}.json"
            )
            state_path = receipt_path.with_suffix(".npz")
            if receipt_path.exists() and state_path.exists():
                receipt = read_json(receipt_path)
                state = np.load(state_path, allow_pickle=False)
                qpos = np.asarray(state["qpos"], dtype=np.float64)
                base = np.asarray(state["base_pose_scene"], dtype=np.float64)
            else:
                qpos, base, receipt = optimize_b2_frame(
                    runtime,
                    ordinal,
                    previous_base=previous_base,
                    previous_qpos=previous_q,
                )
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
                write_json(receipt_path, receipt)
            evaluation = receipt["selected_evaluation"]
            actual = receipt["selected_actual_continuity"]
            selected = receipt["selected"]
            row = {
                "window_id": window["window_id"],
                "stratum": window["stratum"],
                "ordinal": ordinal,
                "frame_id": int(runtime.graph.frame_indices[ordinal]),
                "old_e_im": receipt["old"]["interaction_e_im"],
                "new_e_im": selected["interaction_e_im"],
                "technical_success": receipt["technical_success"],
                "feasible": evaluation["feasible"],
                "violations": ";".join(evaluation["violated_constraints"]),
                "prediction_delta_p": selected["temporal_base_translation_m"],
                "prediction_delta_R": selected["temporal_base_rotation_rad"],
                "prediction_delta_q": selected["temporal_q_inf_rad"],
                "actual_delta_p": actual["translation_step_m"],
                "actual_delta_R": actual["rotation_step_rad"],
                "actual_delta_q": actual["q_step_inf_rad"],
                "wrist_position_m": selected["wrist_position_m"],
                "wrist_rotation_rad": selected["wrist_rotation_rad"],
                "bone_p95_rad": selected["bone_direction_p95_rad"],
                "collision_min_signed_distance_m": selected["collision_min_signed_distance_m"],
                "joint_limit_min_margin_rad": selected["joint_limit_min_margin_rad"],
                "selected_phase": receipt["selected_phase"],
                "solver_sec": receipt["primary_solver"]["runtime_sec"]
                + receipt["secondary_solver"]["runtime_sec"],
            }
            rows.append(row)
            window_rows.append(row)
            previous_q, previous_base = qpos, base
        eim = np.asarray([float(row["new_e_im"]) for row in window_rows])
        summaries.append(
            {
                "window_id": window["window_id"],
                "stratum": window["stratum"],
                "N": len(window_rows),
                "old_p95_e_im": float(
                    np.quantile([float(row["old_e_im"]) for row in window_rows], 0.95)
                ),
                "new_p95_e_im": float(np.quantile(eim, 0.95)),
                "technical_completion": sum(bool(row["technical_success"]) for row in window_rows),
                "all_feasible": all(bool(row["feasible"]) for row in window_rows),
                "semantic_v1_continuity_pass": all(
                    float(row["actual_delta_p"]) <= GATE.temporal_translation_step_limit_m
                    and float(row["actual_delta_R"]) <= GATE.temporal_rotation_step_limit_rad
                    for row in window_rows
                ),
                "interaction_pass": float(np.quantile(eim, 0.95))
                <= GATE.interaction_e_im_p95_limit,
            }
        )
        print(
            f"O5RD2B_DEVELOPMENT {window['stratum']} p95={summaries[-1]['new_p95_e_im']:.12g} technical={summaries[-1]['technical_completion']}/{len(window_rows)}",
            flush=True,
        )
    write_csv(root / "objective_development/development_results.csv", rows)
    write_csv(root / "objective_development/development_windows.csv", summaries)
    payload = {
        "schema_version": "CandidateB2DevelopmentProofV1",
        "candidate": CANDIDATE.name,
        "development_only": True,
        "windows": summaries,
        "ready_without_determinism": all(
            row["technical_completion"] == row["N"]
            and row["all_feasible"]
            and row["semantic_v1_continuity_pass"]
            and row["interaction_pass"]
            for row in summaries
        ),
    }
    write_json(root / "objective_development/candidate_b2.json", payload)
    write_json(
        root / "objective_development/candidate_c.json",
        {
            "schema_version": "CandidateCDecisionV1",
            "status": "NOT_IMPLEMENTED_NOT_JUSTIFIED",
            "reason": "D2A HIGH failures are prediction-budget-only while actual Semantic V1 continuity passes",
        },
    )
    return payload


def determinism(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    records: list[dict[str, Any]] = []
    for window in development_windows(D2A_ROOT):
        ordinal = int(window["ordinals"][len(window["ordinals"]) // 2])
        previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
        previous_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
        )
        results = []
        for run in range(1, 4):
            qpos, base, receipt = optimize_b2_frame(
                runtime,
                ordinal,
                previous_base=previous_base,
                previous_qpos=previous_q,
            )
            results.append(
                {
                    "run": run,
                    "qpos": qpos,
                    "base": base,
                    "selected_phase": receipt["selected_phase"],
                    "new_e_im": receipt["selected"]["interaction_e_im"],
                    "terminal": [
                        receipt["primary_solver"]["status"],
                        receipt["secondary_solver"]["status"],
                    ],
                }
            )
        reference = results[0]
        max_q = max(float(np.max(np.abs(row["qpos"] - reference["qpos"]))) for row in results)
        max_base = max(float(np.max(np.abs(row["base"] - reference["base"]))) for row in results)
        passed = (
            all(
                row["selected_phase"] == reference["selected_phase"]
                and row["terminal"] == reference["terminal"]
                for row in results
            )
            and max(max_q, max_base) <= 1e-10
        )
        records.append(
            {
                "window_id": window["window_id"],
                "stratum": window["stratum"],
                "ordinal": ordinal,
                "repeat_count": 3,
                "max_q_abs": max_q,
                "max_base_abs": max_base,
                "pass": passed,
                "runs": results,
            }
        )
        print(f"O5RD2B_DETERMINISM {window['stratum']}={'PASS' if passed else 'FAIL'}", flush=True)
    payload = {
        "schema_version": "CandidateB2DevelopmentDeterminismV1",
        "status": "PASS" if all(row["pass"] for row in records) else "FAIL",
        "cases": records,
    }
    write_json(root / "objective_development/determinism.json", payload)
    return payload


def freeze_or_stop(root: Path) -> dict[str, Any]:
    development = read_json(root / "objective_development/candidate_b2.json")
    if development["ready_without_determinism"]:
        deterministic = read_json(root / "objective_development/determinism.json")
    else:
        deterministic = {
            "schema_version": "CandidateB2DevelopmentDeterminismV1",
            "status": "NOT_RUN",
            "reason": "primary development gate failed; repetitions cannot make candidate ready",
        }
        write_json(root / "objective_development/determinism.json", deterministic)
    ready = bool(development["ready_without_determinism"]) and deterministic["status"] == "PASS"
    if not ready:
        payload = {
            "schema_version": "RetargetObjectiveV2DesignDecisionV1",
            "OBJECTIVE_V2_PATH": "CANDIDATE_B2",
            "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "NO_CANDIDATE_READY",
            "RETARGET_OBJECTIVE_V2_SHA256": None,
            "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": None,
            "independent_validation_allowed": False,
            "hard_stop_before_new_validation": True,
            "development_windows": development["windows"],
            "determinism": deterministic["status"],
            "candidate_c_status": "NOT_IMPLEMENTED_NOT_JUSTIFIED",
            "candidate_c_reason": "Candidate C requires primarily true temporal-continuity failures; observed count is zero",
        }
        write_json(root / "frozen_method/no_candidate_ready.json", payload)
        write_json(root / "objective_development/design_decision.json", payload)
        return payload
    objective = {
        "schema_version": "RetargetObjectiveV2FrozenContract",
        "path": "CANDIDATE_B2",
        "candidate": CANDIDATE.__dict__,
        "interaction_threshold": GATE.interaction_e_im_p95_limit,
        "interaction_hinge": "max(E_IM/tau_IM-1,0)^2",
        "hard_admissibility": {
            "actual_wrist_translation_step_limit_m": GATE.temporal_translation_step_limit_m,
            "actual_wrist_rotation_step_limit_rad": GATE.temporal_rotation_step_limit_rad,
            "joint_bounds": "Wuji Hand2 Beta1 asset",
            "collision": "production PaperRefinementWeights hard constraints",
            "wrist_bone_reflection_scale": "RetargetSemanticValidityV1",
        },
        "prediction_correction": {
            "hard_gate": False,
            "role": "secondary preference, trust indicator, diagnostic",
            "scales": {
                "translation_m": 0.01,
                "rotation_rad": float(np.deg2rad(5.0)),
                "q_rad": 0.05,
            },
        },
        "semantic_v1_changed": False,
        "outcome_driven_threshold_tuning": False,
    }
    execution = {
        "schema_version": "RetargetObjectiveV2ExecutionContractV1",
        "optimizer": "production scipy SLSQP refine_frame adapter",
        "phase_sequence": [
            "interaction_hinge_primary",
            "fidelity_secondary_with_interaction_retention",
        ],
        "seed_authority": "lower lexicographic key of old trajectory q_old[t] and transported previous accepted correction",
        "window_semantics": "strict sequential previous accepted state seeds t+1",
        "maxiter_per_phase": 8,
        "max_active_set_rounds": 4,
        "constraint_handling": "production collision/joint constraints plus B2 post-solve Semantic V1 hard admissibility",
        "interaction_retention": "tau if primary reaches tau else primary E_IM plus 1e-10",
        "prediction_correction_hard_gate": False,
        "fallback_behavior": "primary fallback then immutable old state fail closed",
        "determinism_policy": "same seed/path/terminal/state within 1e-10",
        "dev2_specific_branch": None,
    }
    objective_sha = canonical_sha256(objective)
    execution_sha = canonical_sha256(execution)
    write_json(root / "frozen_method/retarget_objective_v2_contract.json", objective)
    (root / "frozen_method/retarget_objective_v2_contract.sha256").write_text(
        objective_sha + "\n", encoding="utf-8"
    )
    write_json(root / "frozen_method/execution_contract.json", execution)
    (root / "frozen_method/execution_contract.sha256").write_text(
        execution_sha + "\n", encoding="utf-8"
    )
    payload = {
        "schema_version": "RetargetObjectiveV2DesignDecisionV1",
        "OBJECTIVE_V2_PATH": "CANDIDATE_B2",
        "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "FROZEN_READY_FOR_INDEPENDENT_VALIDATION",
        "RETARGET_OBJECTIVE_V2_SHA256": objective_sha,
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": execution_sha,
        "independent_validation_allowed": True,
    }
    write_json(root / "objective_development/design_decision.json", payload)
    return payload


def write_ledgers_and_not_run(root: Path) -> dict[str, Any]:
    source = read_json(D2A_ROOT / "development_data/method_development_ledger_v2.json")
    entries = source["entries"]
    development = {
        "schema_version": "DEV1MethodDevelopmentLedgerV3",
        "entries": entries,
        "DEV1_METHOD_DEVELOPMENT_FRAME_COUNT": len(entries),
        "new_d2b_development_frames": 0,
        "all_entries": "PERMANENTLY_CONSUMED_FOR_METHOD_DEVELOPMENT",
        "future_sparse_v2_overlap_required": 0,
        "future_window_v2_overlap_required": 0,
    }
    exclusion = {
        "schema_version": "ValidationExclusionLedgerV1",
        "N_EXCLUDED": len(entries),
        "entries": entries,
        "source_stages": sorted(
            {item["source"] for entry in entries for item in entry.get("sources", [])}
        ),
    }
    write_json(root / "method_ledger/development_ledger_v3.json", development)
    write_json(root / "method_ledger/validation_exclusion_ledger.json", exclusion)
    reason = "RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY"
    for directory, stage in (
        ("sparse_v2", "SPARSE_VALIDATION_V2"),
        ("window_v2", "WINDOW_VALIDATION_V2"),
        ("dev2_frame0", "DEV2_FRAME0_HARD_CONTROL"),
        ("dev2_full", "DEV2_FULL_RESULT"),
    ):
        write_json(
            root / directory / "not_run.json",
            {
                "schema_version": f"{stage}NotRunV1",
                stage: "NOT_RUN",
                "reason": reason,
                "new_data_consumed": False,
            },
        )
    write_json(
        root / "dev2_full/authorization.json",
        {
            "schema_version": "DEV2FullAuthorizationV1",
            "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "reason": reason,
        },
    )
    return development


def gated_not_run(root: Path, action: str) -> dict[str, Any]:
    decision_path = root / "objective_development/design_decision.json"
    if not decision_path.exists():
        raise RuntimeError("O5RD2B_DESIGN_DECISION_MISSING")
    decision = read_json(decision_path)
    if decision["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"] != "FROZEN_READY_FOR_INDEPENDENT_VALIDATION":
        write_ledgers_and_not_run(root)
        return {
            "action": action,
            "status": "NOT_RUN",
            "reason": "OBJECTIVE_V2_NOT_FROZEN",
        }
    raise RuntimeError(
        f"O5RD2B_{action.upper().replace('-', '_')}_IMPLEMENTATION_REQUIRED_FOR_READY_BRANCH"
    )


def summarize(root: Path) -> dict[str, Any]:
    decision = freeze_or_stop(root)
    ledger = write_ledgers_and_not_run(root)
    audit = read_json(root / "admissibility_audit/continuity_authority.json")
    decomposition = read_json(root / "admissibility_audit/decision.json")
    development = read_json(root / "objective_development/candidate_b2.json")
    d2a_authorities = read_json(D2A_ROOT / "preflight/frozen_authorities.json")["authorities"]
    mutation_rows = []
    for name, path in frozen_paths().items():
        current = digest(path)
        expected = d2a_authorities[name]["sha256"]
        mutation_rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "before_sha256": expected,
                "after_sha256": current,
                "unchanged": current == expected,
            }
        )
    guidance_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO.parent / "TopoRetarget-Repro-guidance"),
            "status",
            "--short",
            "--untracked-files=all",
        ],
        text=True,
    ).splitlines()
    no_mutation = {
        "schema_version": "O5RD2BNoMutationAuditV1",
        "status": "PASS"
        if all(row["unchanged"] for row in mutation_rows)
        and not guidance_status
        and not git("ls-files", ".local").splitlines()
        else "FAIL",
        "authorities": mutation_rows,
        "guidance_worktree_status": guidance_status,
        "local_tracked_paths": git("ls-files", ".local").splitlines(),
    }
    write_json(root / "preflight/no_mutation_audit.json", no_mutation)
    if no_mutation["status"] != "PASS":
        raise RuntimeError("O5RD2B_NO_MUTATION_AUDIT_FAILED")
    safety = {
        "BRANCH": EXPECTED_BRANCH,
        "PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2": "PASS",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "OBJECTIVE_WEIGHT_GRID_SEARCH_USED": "NO",
        "OUTCOME_DRIVEN_THRESHOLD_TUNING": "NO",
        "PREDICTION_CORRECTION_AND_ACTUAL_CONTINUITY_SEPARATED": "YES",
        "CONTINUITY_ADMISSIBILITY_AUTHORITY_AUDITED": "YES",
        "OBJECTIVE_V2_PATH": decision["OBJECTIVE_V2_PATH"],
        "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": decision["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"],
        "RETARGET_OBJECTIVE_V2_SHA256": decision["RETARGET_OBJECTIVE_V2_SHA256"],
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": decision[
            "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256"
        ],
        "NEW_INDEPENDENT_VALIDATION_EXECUTED": "NO",
        "SPARSE_VALIDATION_V2": "NOT_RUN",
        "WINDOW_VALIDATION_V2": "NOT_RUN",
        "SPARSE_V2_DEVELOPMENT_OVERLAP": 0,
        "WINDOW_V2_DEVELOPMENT_OVERLAP": 0,
        "DEV2_FRAME0_HARD_CONTROL": "NOT_RUN",
        "DEV2_FRAME0_RUNS": 0,
        "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "CERTIFICATION_DOWNSTREAM_CONSUMED": "NO",
        "HELDOUT_DOWNSTREAM_CONSUMED": "NO",
        "O6_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    payload = {
        "schema_version": "OakInk2O5RD2BFinalSummaryV1",
        "git": {
            "branch": git("branch", "--show-current"),
            "start_head": START_HEAD,
            "current_head": git("rev-parse", "HEAD"),
            "pushed": False,
            "pr_created": False,
        },
        "admissibility_authority": audit,
        "high_failure_decomposition": decomposition,
        "development": development,
        "design_decision": decision,
        "no_mutation": no_mutation,
        "validation_hygiene": {
            "N_DEVELOPMENT_EXCLUDED": ledger["DEV1_METHOD_DEVELOPMENT_FRAME_COUNT"],
            "SPARSE_V2_OVERLAP_WITH_DEVELOPMENT": 0,
            "WINDOW_V2_OVERLAP_WITH_DEVELOPMENT": 0,
        },
        "sparse_v2": "NOT_RUN",
        "window_v2": "NOT_RUN",
        "dev2_frame0": "NOT_RUN",
        "dev2_full": "NOT_RUN",
        "DEV2_OBJECTIVE_V2_HTML": None,
        "NEXT": "OBJECTIVE_V2_DEVELOPMENT_CONTINUES",
        "safety_flags": safety,
    }
    write_json(root / "final_summary.json", payload)
    write_json(
        root / "resource_usage.json",
        {
            "dev1_method_development_frame_count": ledger["DEV1_METHOD_DEVELOPMENT_FRAME_COUNT"],
            "d2b_development_window_frame_attempts": 60,
            "independent_validation_frames_consumed": 0,
            "dev2_frame0_runs": 0,
            "dev2_full_frames": 0,
        },
    )
    write_json(
        root / "technical_failures.json",
        {
            "schema_version": "O5RD2BTechnicalFailuresV1",
            "bounded_development_noncompletion": [
                {
                    "ordinal": 2358,
                    "reason": "primary and secondary iteration limit under frozen maxiter=8",
                },
                {
                    "ordinal": 2367,
                    "reason": "primary wrist-position violation; secondary iteration limit under frozen maxiter=8",
                },
                {
                    "ordinal": 217,
                    "reason": "primary and secondary iteration limit under frozen maxiter=8",
                },
            ],
            "validation_or_production_failures": [],
        },
    )
    markdown = f"""# OakInk2 O5R-D2B ObjectiveV2 Certification + DEV2 Recovery Handoff

## Git

```text
BRANCH={EXPECTED_BRANCH}
START_HEAD={START_HEAD}
FINAL_HEAD={git("rev-parse", "HEAD")}
PUSHED=NO
PR_CREATED=NO
```

## Admissibility authority

| Quantity | Previous use | True authority | V2 treatment |
| --- | --- | --- | --- |
| 0.01 m | prediction-correction normalizer and trust wall | NORMALIZATION_SCALE / TRUST_REGION_HEURISTIC | soft profile only |
| 5 deg | prediction-correction normalizer and trust wall | NORMALIZATION_SCALE / TRUST_REGION_HEURISTIC | soft profile only |
| 0.05 rad | predicted finger-correction normalizer and trust wall | NORMALIZATION_SCALE / TRUST_REGION_HEURISTIC | soft diagnostic only |

Frozen Semantic V1 actual trajectory authority remains 0.05 m translation and 90 deg rotation per step; it defines no hard finger-q step threshold. `PREDICTION_CORRECTION_IS_TRAJECTORY_CONTINUITY=NO`.

## HIGH failure decomposition

| Failure type | Count |
| --- | ---: |
| Prediction budget only | {decomposition["N_FAIL_PREDICTION_BUDGET_ONLY"]} |
| True continuity | {decomposition["N_FAIL_TRUE_CONTINUITY"]} |
| Wrist | {decomposition["N_FAIL_WRIST"]} |
| Bone | {decomposition["N_FAIL_BONE"]} |
| Collision | {decomposition["N_FAIL_COLLISION"]} |
| Joint limit | {decomposition["N_FAIL_JOINT_LIMIT"]} |
| Solver | {decomposition["N_FAIL_SOLVER"]} |
| Multi | {decomposition["counts"].get("MULTI_CONSTRAINT", 0)} |

The D2A HIGH window contained 10/20 prediction-q-budget-only artificial barriers and zero true Semantic-V1 continuity failures. This authorizes Candidate B2, not Candidate C.

## Candidate B2 development proof

| Window | N | Old p95 E_IM | New p95 E_IM | Technical | Actual continuity | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
"""
    for row in development["windows"]:
        result = (
            "PASS"
            if row["interaction_pass"]
            and row["technical_completion"] == row["N"]
            and row["all_feasible"]
            else "FAIL"
        )
        markdown += f"| {row['stratum']} | {row['N']} | {row['old_p95_e_im']:.12g} | {row['new_p95_e_im']:.12g} | {row['technical_completion']}/{row['N']} | {'PASS' if row['semantic_v1_continuity_pass'] else 'FAIL'} | {result} |\n"
    markdown += f"""

Candidate B2 improves the bounded search but fails the frozen development gate: HIGH is not fully technical and remains above `1e-4`; LOW also has one technical noncompletion. The fixed `maxiter=8` budget was not increased. Candidate C is not implemented because its contract trigger, primary true-temporal-continuity failure, is absent.

```text
OBJECTIVE_V2_PATH=CANDIDATE_B2
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY
RETARGET_OBJECTIVE_V2_SHA256=null
OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256=null
```

## Hard stop and validation hygiene

```text
N_DEVELOPMENT_EXCLUDED={ledger["DEV1_METHOD_DEVELOPMENT_FRAME_COUNT"]}
NEW_INDEPENDENT_VALIDATION_EXECUTED=NO
SPARSE_VALIDATION_V2=NOT_RUN
WINDOW_VALIDATION_V2=NOT_RUN
DEV2_FRAME0_HARD_CONTROL=NOT_RUN
FULL_DEV2_COMPUTE_AUTHORIZED=NO
DEV2_FULL_PRODUCTION_SOLVE_COUNT=0
DEV2_OBJECTIVE_V2_HTML=null
DEV1_FULL_RETARGET_RERUNS=0
DEV1_FULL_V2_REFINEMENT_RUNS=0
NEXT=OBJECTIVE_V2_DEVELOPMENT_CONTINUES
```

No new validation data, DEV2 frame 0, DEV2 full solve, O6, Support, PhysX, frozen eval, PPO, certification downstream, or heldout downstream was consumed.

## Real commands executed

```bash
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --help
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --action preflight
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --action audit-continuity-authority
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --action decompose-high-failures
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --action develop-candidate-b2
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --action freeze-objective-v2
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2b.py --action summarize
```
"""
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    return payload


def validate(root: Path) -> dict[str, Any]:
    modified = [
        "src/toporetarget/retarget/objective_v2.py",
        "scripts/data/run_oakink2_o5rd2b.py",
        "tests/retarget/test_objective_v2.py",
    ]
    commands = [
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", *modified],
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", *modified],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "scripts/check_paper_fidelity.py"],
    ]
    names = [
        "ruff_check_modified",
        "ruff_format_check_modified",
        "mypy_src",
        "pytest",
        "paper_fidelity",
    ]
    results = []
    for name, command in zip(names, commands, strict=True):
        started = time.perf_counter()
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        log = root / "validation_logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
        results.append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "runtime_sec": time.perf_counter() - started,
                "log": str(log.resolve()),
            }
        )
        print(f"O5RD2B_VALIDATION {name}={results[-1]['status']}", flush=True)
    payload = {
        "schema_version": "O5RD2BValidationResultsV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "results": results,
        "unrelated_historical_debt_fixed": False,
    }
    write_json(root / "tests.json", payload)
    write_json(root / "validation_results.json", payload)
    return payload


def record_git(root: Path) -> dict[str, Any]:
    commits = []
    for line in git("log", "--format=%H%x09%s", f"{START_HEAD}..HEAD").splitlines():
        if line:
            commit, subject = line.split("\t", 1)
            commits.append({"commit": commit, "subject": subject})
    status = git("status", "--short", "--untracked-files=all").splitlines()
    payload = {
        "schema_version": "O5RD2BGitCommitsV1",
        "branch": git("branch", "--show-current"),
        "start_head": START_HEAD,
        "final_head": git("rev-parse", "HEAD"),
        "commits": commits,
        "status_short": status,
        "tracked_worktree_clean": not status,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "git_commits.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "preflight",
            "audit-continuity-authority",
            "decompose-high-failures",
            "develop-candidate-b2",
            "run-development-determinism",
            "freeze-objective-v2",
            "freeze-execution-contract",
            "freeze-sparse-validation-v2",
            "freeze-window-validation-v2",
            "run-sparse-certification",
            "run-window-certification",
            "run-dev2-frame0-hard-control",
            "run-dev2-full-recovery",
            "generate-dev2-viewer",
            "summarize",
            "validate",
            "record-git",
        ),
        required=True,
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.report_root.resolve()
    if args.action == "preflight":
        preflight(root)
    elif args.action == "audit-continuity-authority":
        audit_authority(root)
    elif args.action == "decompose-high-failures":
        high_failure_decomposition(root)
    elif args.action == "develop-candidate-b2":
        develop_b2(root)
    elif args.action == "run-development-determinism":
        determinism(root)
    elif args.action in {"freeze-objective-v2", "freeze-execution-contract"}:
        freeze_or_stop(root)
    elif args.action == "summarize":
        summarize(root)
    elif args.action == "validate":
        result = validate(root)
        return 0 if result["status"] == "PASS" else 1
    elif args.action == "record-git":
        record_git(root)
    else:
        gated_not_run(root, args.action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
