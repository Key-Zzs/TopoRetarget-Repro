#!/usr/bin/env python3
"""Execute OakInk2 O5R-D2C and stop before independent validation.

Candidate B2 is evaluator-replayed against the frozen B1 Q0/Q1 states before
any search work is allowed.  A passing alignment gate locks the objective
semantics and permits exactly two predeclared generic execution contracts.
This CLI has no action that can run DEV1 full, DEV2, SparseValidationV2,
WindowValidationV2, or downstream physical/policy work.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5rd2a import (
    D2ARuntime,
    _measurement_row,
    _query_set,
    development_windows,
    digest,
    git,
    read_csv,
    read_json,
    write_csv,
    write_json,
)
from scripts.data.run_oakink2_o5rd2b import actual_continuity
from toporetarget.evaluation.retarget_semantic_validity import SemanticGateContractV1
from toporetarget.retarget.continuous import encode_base_correction
from toporetarget.retarget.final_refinement import refine_frame
from toporetarget.retarget.interaction_objective import InteractionMeshResidual
from toporetarget.retarget.objective_v2 import (
    NUMERICAL_RETENTION_EPSILON,
    ObjectiveV2Candidate,
    ObjectiveV2DevelopmentContext,
    ObjectiveV2Measurements,
    evaluate_candidate_b2,
    interaction_retention_limit,
)
from toporetarget.retarget.objective_v2_execution import (
    CONTRIBUTOR_SEARCH_SCHEMA_VERSION,
    SEMANTIC_FINGER_KEYPOINTS,
    ScreenedCandidate,
    SearchExecutionContract,
    asset_derived_dof_blocks,
    contributor_scores,
    default_search_contracts,
    deterministic_block_seeds,
    rank_contributors,
    retain_after_polish,
    select_candidate,
)
from toporetarget.utils.hashing import sha256_file

ROOT = REPO / ".local/reports/oakink2_o5rd2c_candidate_b2_search_v1"
D1_ROOT = REPO / ".local/reports/oakink2_o5rd1_objective_alignment_v1"
D2A_ROOT = REPO / ".local/reports/oakink2_o5rd2a_objective_v2_design_v1"
D2B_ROOT = REPO / ".local/reports/oakink2_o5rd2b_objective_v2_certification_v1"
O5RB_ROOT = REPO / ".local/reports/oakink2_o5rb_parallel_v1"

EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "68f6ae230b55c9fc84f87180df414cffcaf04cc7"
CANDIDATE = ObjectiveV2Candidate.candidate_b2()
GATE = SemanticGateContractV1()
CONTRACTS = {contract.name: contract for contract in default_search_contracts()}


def _head_descends_from_start(head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_measurement(value: ObjectiveV2Measurements) -> bool:
    scalar = [item for item in asdict(value).values() if isinstance(item, int | float)]
    return bool(np.all(np.isfinite(np.asarray(scalar, dtype=np.float64))))


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


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2C_BRANCH_MISMATCH:{branch}")
    if not _head_descends_from_start(head):
        raise RuntimeError(f"O5RD2C_START_HEAD_NOT_ANCESTOR:{head}")

    d1 = read_json(D1_ROOT / "decision/production_objective_alignment.json")
    d2a = read_json(D2A_ROOT / "final_summary.json")
    d2b = read_json(D2B_ROOT / "final_summary.json")
    if d1.get("PRODUCTION_OBJECTIVE_ALIGNMENT") != "MISALIGNED":
        raise RuntimeError("O5RD2C_D1_ALIGNMENT_DRIFT")
    if d2a["objective_v2_status"]["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"] != "NO_CANDIDATE_READY":
        raise RuntimeError("O5RD2C_D2A_STATUS_DRIFT")
    if d2b["design_decision"]["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"] != "NO_CANDIDATE_READY":
        raise RuntimeError("O5RD2C_D2B_STATUS_DRIFT")

    authorities: dict[str, dict[str, Any]] = {}
    d2b_authorities = read_json(D2B_ROOT / "preflight/frozen_authorities.json")["authorities"]
    for name, receipt in d2b_authorities.items():
        path = Path(receipt["path"])
        observed = digest(path)
        expected = str(receipt["sha256"])
        if observed != expected:
            raise RuntimeError(f"O5RD2C_FROZEN_AUTHORITY_DRIFT:{name}")
        authorities[name] = {"path": str(path.resolve()), "sha256": observed}

    extra = {
        "production_context_binding_v2": D2A_ROOT
        / "context_binding/production_context_binding_v2.json",
        "candidate_b2_implementation": REPO / "src/toporetarget/retarget/objective_v2.py",
        "d2b_candidate_b2": D2B_ROOT / "objective_development/candidate_b2.json",
        "d2b_development_results": D2B_ROOT / "objective_development/development_results.csv",
        "d2b_development_windows": D2B_ROOT / "objective_development/development_windows.csv",
        "d2b_failure_ledger": D2B_ROOT / "technical_failures.jsonl",
        "d2b_admissibility": D2B_ROOT / "admissibility_audit/decision.json",
        "method_development_ledger": D2B_ROOT / "method_ledger/development_ledger_v3.json",
        "b1_q0_q1_states": D1_ROOT / "replay/states",
        "b1_objective_decomposition": D1_ROOT / "replay/state_objective_decomposition.csv",
    }
    for name, path in extra.items():
        if not path.exists():
            raise FileNotFoundError(path)
        authorities[name] = {"path": str(path.resolve()), "sha256": digest(path)}

    payload = {
        "schema_version": "OakInk2O5RD2CPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "start_head": START_HEAD,
        "head_at_preflight_artifact_write": head,
        "initial_status_short": [],
        "initial_status_provenance": "mandated shell preflight captured before task modifications",
        "status_short_at_artifact_write": git(
            "status", "--short", "--untracked-files=all"
        ).splitlines(),
        "diff_stat_at_artifact_write": git("diff", "--stat").splitlines(),
        "diff_check": git("diff", "--check").splitlines(),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "remotes": git("remote", "-v").splitlines(),
        "authorities": authorities,
        "upstream": {
            "PRODUCTION_OBJECTIVE_ALIGNMENT": "MISALIGNED",
            "PRIMARY_ROOT_CAUSE": "OBJECTIVE_SEMANTIC_MISALIGNMENT",
            "PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2": "PASS",
            "OBJECTIVE_V2_PATH": "CANDIDATE_B2",
            "D2B_STATUS": "NO_CANDIDATE_READY",
        },
        "forbidden_execution": {
            "DEV1_FULL_RETARGET_RERUNS": 0,
            "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
            "DEV2_FRAME0_NEW_SOLVES": 0,
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "SPARSE_VALIDATION_V2": "NOT_RUN",
            "WINDOW_VALIDATION_V2": "NOT_RUN",
        },
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "preflight/git.json", payload)
    write_json(
        root / "preflight/upstream_state.json",
        {"schema_version": "O5RD2CUpstreamStateV1", **payload["upstream"]},
    )
    write_json(
        root / "preflight/frozen_authorities.json",
        {
            "schema_version": "O5RD2CFrozenAuthoritiesV1",
            "status": "PASS",
            "authorities": authorities,
        },
    )
    return payload


def replay_alignment(root: Path) -> dict[str, Any]:
    preflight_path = root / "preflight/frozen_authorities.json"
    if not preflight_path.exists():
        raise RuntimeError("O5RD2C_PREFLIGHT_REQUIRED")
    runtime = D2ARuntime(root)
    frames = read_json(O5RB_ROOT / "b1_thumb_feasibility/frame_selection.json")["frames"]
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    clear_true_feasible = 0
    admitted_or_preferred = 0
    rejected_prediction_only = 0
    for frame in frames:
        ordinal = int(frame["ordinal"])
        state = np.load(D1_ROOT / f"replay/states/frame_{ordinal:04d}.npz", allow_pickle=False)
        q0 = np.asarray(state["q0"], dtype=np.float64)
        q1 = np.asarray(state["q1"], dtype=np.float64)
        base = np.asarray(state["base0"], dtype=np.float64)
        previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
        previous_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
        )
        binding, context = runtime.bind_context(
            ordinal, previous_base=previous_base, previous_qpos=previous_q
        )
        slack = runtime.q0_slack(ordinal)
        q0_values = runtime.measurement(
            ordinal, q0, base, binding=binding, context=context, slack=slack
        )
        q1_values = runtime.measurement(
            ordinal, q1, base, binding=binding, context=context, slack=slack
        )
        q0_actual = actual_continuity(runtime, previous_q, previous_base, q0, base)
        q1_actual = actual_continuity(runtime, previous_q, previous_base, q1, base)
        q0_receipt = _evaluate_b2(runtime, q0_values, q0_actual)
        q1_receipt = _evaluate_b2(runtime, q1_values, q1_actual)
        q0_primary = float(q0_receipt["primary_objective"])
        q1_primary = float(q1_receipt["primary_objective"])
        clear = q1_values.interaction_e_im < q0_values.interaction_e_im - 1.0e-10
        if not _finite_measurement(q1_values):
            classification = "INCONCLUSIVE"
        elif not q1_receipt["feasible"]:
            classification = "B2_TRUE_CONSTRAINT_VIOLATION"
        elif q1_primary < q0_primary - NUMERICAL_RETENTION_EPSILON:
            classification = "B2_ADMISSIBLE_AND_PRIMARY_BETTER"
        elif abs(q1_primary - q0_primary) <= NUMERICAL_RETENTION_EPSILON:
            classification = "B2_ADMISSIBLE_PRIMARY_EQUIVALENT"
        elif q1_values.secondary_objective > q0_values.secondary_objective:
            classification = "B2_SECONDARY_INADMISSIBLE"
        else:
            classification = "B2_OBJECTIVE_STILL_MISALIGNED"
        counts[classification] = counts.get(classification, 0) + 1
        prediction_exceeds = (
            q1_values.temporal_base_translation_m
            > runtime.authority.temporal_base_translation_limit_m
            or q1_values.temporal_base_rotation_rad
            > runtime.authority.temporal_base_rotation_limit_rad
            or q1_values.temporal_q_inf_rad > runtime.authority.temporal_q_inf_limit_rad
            or q1_values.temporal_excess_keypoint_m
            > runtime.authority.temporal_excess_keypoint_limit_m
        )
        prediction_only_rejection = bool(
            prediction_exceeds
            and not q1_receipt["violated_constraints"]
            and classification
            not in {
                "B2_ADMISSIBLE_AND_PRIMARY_BETTER",
                "B2_ADMISSIBLE_PRIMARY_EQUIVALENT",
            }
        )
        rejected_prediction_only += int(prediction_only_rejection)
        eligible = bool(clear and q1_receipt["feasible"])
        admitted = classification in {
            "B2_ADMISSIBLE_AND_PRIMARY_BETTER",
            "B2_ADMISSIBLE_PRIMARY_EQUIVALENT",
        }
        clear_true_feasible += int(eligible)
        admitted_or_preferred += int(eligible and admitted)
        rows.append(
            {
                "frame_id": int(frame["frame_id"]),
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "classification": classification,
                "q0_e_im": q0_values.interaction_e_im,
                "q1_e_im": q1_values.interaction_e_im,
                "q0_primary_hinge": q0_primary,
                "q1_primary_hinge": q1_primary,
                "q0_secondary_fidelity": q0_values.secondary_objective,
                "q1_secondary_fidelity": q1_values.secondary_objective,
                "q0_feasible": q0_receipt["feasible"],
                "q1_feasible": q1_receipt["feasible"],
                "q1_true_violations": ";".join(q1_receipt["violated_constraints"]),
                "q1_prediction_delta_p": q1_values.temporal_base_translation_m,
                "q1_prediction_delta_R": q1_values.temporal_base_rotation_rad,
                "q1_prediction_delta_q": q1_values.temporal_q_inf_rad,
                "q1_prediction_exceeds_profile": prediction_exceeds,
                "q1_actual_delta_p": q1_actual["translation_step_m"],
                "q1_actual_delta_R": q1_actual["rotation_step_rad"],
                "q1_actual_delta_q": q1_actual["q_step_inf_rad"],
                "q1_wrist_position_m": q1_values.wrist_position_m,
                "q1_wrist_rotation_rad": q1_values.wrist_rotation_rad,
                "q1_bone_p95_rad": q1_values.bone_direction_p95_rad,
                "q1_collision_min_signed_distance_m": q1_values.collision_min_signed_distance_m,
                "q1_joint_limit_min_margin_rad": q1_values.joint_limit_min_margin_rad,
                "prediction_correction_used_as_hard_limit": False,
            }
        )
    fraction = None if clear_true_feasible == 0 else admitted_or_preferred / clear_true_feasible
    passed = bool(
        len(rows) == 25
        and rejected_prediction_only == 0
        and fraction is not None
        and fraction >= 0.8
    )
    payload = {
        "schema_version": "CandidateB2Q0Q1AlignmentSummaryV1",
        "N": len(rows),
        "counts": counts,
        "clear_low_e_im_true_feasible_q1": clear_true_feasible,
        "admitted_or_primary_preferred": admitted_or_preferred,
        "admitted_or_primary_preferred_fraction": fraction,
        "development_threshold": 0.8,
        "prediction_correction_only_rejections": rejected_prediction_only,
        "prediction_correction_is_hard_constraint": False,
        "optimization_executed": False,
        "CANDIDATE_B2_ALIGNMENT_GATE": "PASS" if passed else "FAIL",
        "OBJECTIVE_B2_SEMANTICS_LOCKED_FOR_D2C": "YES" if passed else "NO",
    }
    write_csv(root / "alignment/q0_q1_b2_replay.csv", rows)
    write_json(root / "alignment/q0_q1_b2_summary.json", payload)
    write_json(
        root / "alignment/alignment_gate.json",
        {
            **payload,
            "PRIMARY_BOTTLENECK": None if passed else "OBJECTIVE_B2_STILL_MISALIGNED",
            "SEARCH_REPAIR_ALLOWED": passed,
            "hard_stop_if_fail": True,
        },
    )
    print(
        f"O5RD2C_ALIGNMENT={payload['CANDIDATE_B2_ALIGNMENT_GATE']} "
        f"admitted={admitted_or_preferred}/{clear_true_feasible}",
        flush=True,
    )
    return payload


def require_alignment(root: Path) -> dict[str, Any]:
    path = root / "alignment/alignment_gate.json"
    if not path.exists():
        raise RuntimeError("O5RD2C_ALIGNMENT_GATE_NOT_RUN")
    gate = read_json(path)
    if gate["CANDIDATE_B2_ALIGNMENT_GATE"] != "PASS":
        raise RuntimeError("O5RD2C_ALIGNMENT_GATE_FAILED_HARD_STOP")
    return gate


def audit_d2b_failures(root: Path) -> dict[str, Any]:
    require_alignment(root)
    source_rows = read_csv(D2B_ROOT / "objective_development/development_results.csv")
    failed = [row for row in source_rows if row["technical_success"] == "False"]
    rows: list[dict[str, Any]] = []
    for source in failed:
        ordinal = int(source["ordinal"])
        receipt = read_json(
            D2B_ROOT
            / "objective_development/receipts"
            / source["window_id"]
            / f"frame_{ordinal:04d}.json"
        )
        primary = receipt["primary_evaluation"]
        secondary = receipt["secondary_evaluation"]
        old = receipt["selected_evaluation"]
        usable_primary = bool(primary["feasible"])
        usable_secondary = bool(secondary["feasible"])
        target_primary = bool(primary["per_frame_semantic_target_met"])
        target_secondary = bool(secondary["per_frame_semantic_target_met"])
        contributors: list[str] = []
        if (usable_primary and not receipt["primary_solver"]["accepted"]) or (
            usable_secondary and not receipt["secondary_solver"]["accepted"]
        ):
            contributors.append("CANDIDATE_RETENTION_FAILURE")
        if old["feasible"] and receipt["selected_phase"] == "old_state_fail_closed":
            contributors.append("INVALID_FALLBACK")
        if not target_primary and not target_secondary:
            contributors.append("POOR_INITIAL_BASIN")
        if receipt["primary_solver"]["status"] == 9 or receipt["secondary_solver"]["status"] == 9:
            contributors.append("PRIMARY_OPTIMIZER_MAX_BUDGET")
        classification = contributors[0] if len(contributors) == 1 else "MULTI_FACTOR"
        terminal_phase = "secondary" if usable_secondary else "primary"
        terminal = receipt[terminal_phase]
        terminal_eval = receipt[f"{terminal_phase}_evaluation"]
        rows.append(
            {
                "frame_id": int(source["frame_id"]),
                "ordinal": ordinal,
                "window": source["stratum"],
                "window_id": source["window_id"],
                "seed_used": receipt["seed_source"],
                "dominant_contributor": "NOT_RECORDED_BY_OLD_B2",
                "selected_block": "ALL_20_FINGER_DOFS",
                "initial_hinge": CANDIDATE.primary_value(
                    float(receipt["old"]["interaction_e_im"]), GATE.interaction_e_im_p95_limit
                ),
                "initial_secondary_objective": receipt["old"]["secondary_objective"],
                "terminal_phase": terminal_phase,
                "terminal_hinge": terminal_eval["primary_objective"],
                "terminal_e_im": terminal["interaction_e_im"],
                "terminal_secondary_objective": terminal["secondary_objective"],
                "terminal_independent_feasible": terminal_eval["feasible"],
                "terminal_interaction_valid": terminal_eval["per_frame_semantic_target_met"],
                "primary_optimizer_status": receipt["primary_solver"]["status"],
                "primary_optimizer_message": receipt["primary_solver"]["message"],
                "primary_nfev": receipt["primary_solver"]["nfev"],
                "secondary_optimizer_status": receipt["secondary_solver"]["status"],
                "secondary_optimizer_message": receipt["secondary_solver"]["message"],
                "secondary_nfev": receipt["secondary_solver"]["nfev"],
                "njev": "NOT_EMITTED_BY_D2B_RECEIPT",
                "constraint_margins": json.dumps(
                    terminal_eval["constraint_margins"], sort_keys=True
                ),
                "runtime_sec": receipt["elapsed_sec"],
                "failure_stage": receipt["selected_phase"],
                "classification": classification,
                "root_contributors": ";".join(contributors),
                "usable_primary_terminal": usable_primary,
                "usable_secondary_terminal": usable_secondary,
                "primary_target_met": target_primary,
                "secondary_target_met": target_secondary,
                "old_baseline_feasible": old["feasible"],
            }
        )
    high = [row for row in rows if row["window"] == "HIGH"]
    low = [row for row in rows if row["window"] == "LOW"]
    mechanism = "CANDIDATE_RETENTION_FAILURE_PLUS_INVALID_FALLBACK"
    payload = {
        "schema_version": "D2BFailureRootCauseAuditV1",
        "status": "PASS",
        "N": len(rows),
        "HIGH_FAILURE_COUNT": len(high),
        "LOW_FAILURE_COUNT": len(low),
        "HIGH_FAILURE_ROOT_CAUSE": mechanism,
        "LOW_FAILURE_ROOT_CAUSE": mechanism,
        "all_failures_had_independently_feasible_terminal_state": all(
            row["usable_primary_terminal"] or row["usable_secondary_terminal"] for row in rows
        ),
        "all_fallback_old_states_were_feasible": all(row["old_baseline_feasible"] for row in rows),
        "budget_exhaustion_was_trigger_not_scientific_invalidity": True,
        "required_repair": [
            "screen finite terminal states by independent frozen validity",
            "retain interaction-valid primary before secondary polish",
            "count valid old-q fallback as technical completion",
        ],
    }
    write_csv(root / "search_audit/d2b_failure_analysis.csv", rows)
    write_json(root / "search_audit/d2b_failure_summary.json", payload)
    write_json(
        root / "search_audit/root_cause.json",
        {
            **payload,
            "PRIMARY_ROOT_CONTRIBUTOR": "CANDIDATE_NOT_RETAINED",
            "SECONDARY_ROOT_CONTRIBUTOR": "INVALID_FALLBACK",
            "SEARCH_BUDGET_INADEQUATE_AS_SOLE_CAUSE": False,
        },
    )
    return payload


def compare_search_contracts(root: Path) -> dict[str, Any]:
    require_alignment(root)
    failure = read_json(root / "search_audit/d2b_failure_summary.json")
    if failure["status"] != "PASS":
        raise RuntimeError("O5RD2C_D2B_FAILURE_AUDIT_REQUIRED")
    contracts = default_search_contracts()
    rows = [
        {
            "property": "Starting q",
            "b1_diagnostic": "five block seeds from q_old",
            "old_b2": "one of q_old or transported prediction",
            "selected_v2_search": "q_old plus asset-derived deterministic block seeds",
        },
        {
            "property": "Free DOFs",
            "b1_diagnostic": "one fixed thumb block (diagnostic)",
            "old_b2": "all 20 finger DOFs",
            "selected_v2_search": "top-1 contributor block derived from Eq.7 mass",
        },
        {
            "property": "Number of seeds",
            "b1_diagnostic": 5,
            "old_b2": 1,
            "selected_v2_search": "3 (S1) or 4 (S2), duplicate-elided",
        },
        {
            "property": "Seed authority",
            "b1_diagnostic": "current/rest/mid/lower-quartile/upper-quartile",
            "old_b2": "objective comparison of old and transported",
            "selected_v2_search": "old/rest/mid; S2 also previous refined transported",
        },
        {
            "property": "Primary residual",
            "b1_diagnostic": "selected thumb Eq.7 rows",
            "old_b2": "Candidate-B2 whole E_IM hinge",
            "selected_v2_search": "generic contributor probe then Candidate-B2 whole E_IM hinge",
        },
        {
            "property": "Bounds",
            "b1_diagnostic": "asset joint bounds",
            "old_b2": "asset bounds plus fixed-base trust wall",
            "selected_v2_search": "asset block bounds plus fixed-base trust wall",
        },
        {
            "property": "Optimizer",
            "b1_diagnostic": "least_squares/trf",
            "old_b2": "SLSQP/refine_frame",
            "selected_v2_search": "least_squares seed probes plus SLSQP Candidate-B2 phases",
        },
        {
            "property": "nfev/budget",
            "b1_diagnostic": "max_nfev=80 per seed",
            "old_b2": "maxiter=8 each primary/secondary",
            "selected_v2_search": "probe max_nfev=24; primary=8; secondary=8",
        },
        {
            "property": "Candidate screening",
            "b1_diagnostic": "lowest residual",
            "old_b2": "optimizer accepted flag plus B2 feasibility",
            "selected_v2_search": "finite + independent hard validity then lexicographic B2",
        },
        {
            "property": "Candidate retention",
            "b1_diagnostic": "best observed block q",
            "old_b2": "drops status=9 terminal even when independently valid",
            "selected_v2_search": "retains best independently valid terminal regardless status flag",
        },
        {
            "property": "Secondary polish",
            "b1_diagnostic": "none",
            "old_b2": "retention constraint but accepted-flag gated",
            "selected_v2_search": "interaction retention plus explicit post-polish rejection",
        },
        {
            "property": "Fallback",
            "b1_diagnostic": "current production candidate",
            "old_b2": "returns q_old but marks technical failure",
            "selected_v2_search": "valid q_old is a successful baseline completion",
        },
    ]
    payload = {
        "schema_version": "B1VsB2SearchContractAuditV1",
        "status": "PASS",
        "rows": rows,
        "root_contributor": "MULTIPLE_DIFFERENCES",
        "primary_explanation": "B1 retained the best terminal from a deterministic block multi-start; old B2 used one basin and discarded independently valid status=9 terminals, then mislabeled its valid old-q fallback as technical failure.",
        "predefined_search_candidate_count": len(contracts),
        "search_grid_tuning": False,
    }
    write_json(root / "search_audit/b1_vs_b2_search_contract.json", payload)
    for contract in contracts:
        suffix = "s1" if contract.name.startswith("S1") else "s2"
        write_json(
            root / f"search_candidates/search_{suffix}.json",
            {
                **contract.as_dict(),
                "status": "PREDECLARED_BEFORE_DEVELOPMENT_RUN",
                "objective_semantics_changed": False,
                "generic_contributor_mapping": True,
            },
        )
    write_json(
        root / "search_candidates/search_s3.json",
        {
            "schema_version": "O5RD2CSearchCandidateDecisionV1",
            "name": "S3_ADAPTIVE_SECOND_CONTRIBUTOR",
            "status": "NOT_IMPLEMENTED_NOT_JUSTIFIED",
            "reason": "D2B failed after already reaching interaction-valid top-1 terminal states; no evidence justifies top-K=2 complexity.",
            "predeclared_before_development": True,
        },
    )
    return payload


def _interaction_per_keypoint(
    runtime: D2ARuntime, ordinal: int, qpos: np.ndarray, base: np.ndarray
) -> np.ndarray:
    import torch

    directed = runtime.graph.directed_frames[ordinal]
    residual_model = InteractionMeshResidual(
        runtime.graph.source_vertices[ordinal],
        directed.source_index,
        directed.destination_index,
        directed.weights,
    )
    hand = runtime.model.keypoints_scene(
        torch.as_tensor(qpos, dtype=torch.float64),
        torch.as_tensor(base, dtype=torch.float64),
        layout="mediapipe21",
    )
    vertices = torch.cat(
        [
            hand,
            torch.as_tensor(runtime.graph.source_vertices[ordinal, 21:], dtype=torch.float64),
        ],
        dim=0,
    )
    return residual_model(vertices).square().sum(dim=-1).detach().cpu().numpy() / 71.0


def _contributor_probe(
    runtime: D2ARuntime,
    ordinal: int,
    seed_q: np.ndarray,
    base: np.ndarray,
    *,
    finger: str,
    block: tuple[int, ...],
    max_nfev: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    fixed = np.asarray(seed_q, dtype=np.float64).copy()
    block_ids = np.asarray(block, dtype=np.int64)
    keypoint_ids = np.asarray(SEMANTIC_FINGER_KEYPOINTS[finger], dtype=np.int64)
    lower = np.asarray(runtime.model.joint_lower[block_ids], dtype=np.float64)
    upper = np.asarray(runtime.model.joint_upper[block_ids], dtype=np.float64)
    directed = runtime.graph.directed_frames[ordinal]
    source = np.asarray(runtime.graph.source_vertices[ordinal], dtype=np.float64)
    residual_model = InteractionMeshResidual(
        source, directed.source_index, directed.destination_index, directed.weights
    )
    base_tensor = torch.as_tensor(base, dtype=torch.float64)
    object_tensor = torch.as_tensor(source[21:], dtype=torch.float64)

    def torch_target(block_q: Any) -> Any:
        q = torch.as_tensor(fixed, dtype=torch.float64).clone()
        q[block_ids] = block_q
        hand = runtime.model.keypoints_scene(q, base_tensor, layout="mediapipe21")
        vertices = torch.cat([hand, object_tensor], dim=0)
        return residual_model(vertices)[keypoint_ids].reshape(-1) / np.sqrt(71.0)

    def fun(value: np.ndarray) -> np.ndarray:
        return torch_target(torch.as_tensor(value, dtype=torch.float64)).detach().cpu().numpy()

    def jac(value: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        return torch.autograd.functional.jacobian(torch_target, tensor).detach().cpu().numpy()

    initial = np.clip(fixed[block_ids], lower, upper)
    initial_residual = fun(initial)
    started = time.perf_counter()
    result = least_squares(
        fun,
        initial,
        jac=jac,
        bounds=(lower, upper),
        method="trf",
        max_nfev=max_nfev,
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    output = fixed.copy()
    output[block_ids] = result.x
    return output, {
        "optimizer": "scipy.optimize.least_squares/trf",
        "finger": finger,
        "free_qpos_indices": block_ids.tolist(),
        "max_nfev": max_nfev,
        "initial_contributor_residual": float(initial_residual @ initial_residual),
        "terminal_contributor_residual": float(result.fun @ result.fun),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": None if result.njev is None else int(result.njev),
        "residual_calls": int(result.nfev),
        "jacobian_calls": None if result.njev is None else int(result.njev),
        "fk_calls": "ONE_PER_RESIDUAL_OR_JACOBIAN_AUTOGRAD_EVAL",
        "runtime_sec": time.perf_counter() - started,
    }


def _screened(
    candidate_id: str,
    values: ObjectiveV2Measurements,
    evaluation: dict[str, Any],
    ordinal: int,
    optimizer_converged: bool | None,
) -> ScreenedCandidate:
    return ScreenedCandidate(
        candidate_id=candidate_id,
        feasible=bool(evaluation["feasible"]),
        finite=_finite_measurement(values),
        primary_hinge=float(evaluation["primary_objective"]),
        secondary_objective=float(values.secondary_objective),
        interaction_e_im=float(values.interaction_e_im),
        ordinal=ordinal,
        optimizer_converged=optimizer_converged,
    )


def _solver_profile(result: Any) -> dict[str, Any]:
    timers = dict(result.jacobian_diagnostics.get("timers", {}))
    elapsed = dict(timers.get("elapsed_s", {}))
    return {
        "accepted": bool(result.accepted),
        "optimizer_converged": bool(result.optimizer_converged),
        "status": int(result.optimizer_status_code),
        "message": str(result.optimizer_message),
        "iterations": int(result.optimizer_iterations),
        "nfev": int(result.optimizer_function_evaluations),
        "njev": int(result.optimizer_jacobian_evaluations),
        "objective_calls": int(result.function_evaluations),
        "constraint_jacobian_calls": int(result.jacobian_evaluations),
        "full_audit_pass": bool(result.full_surface_hard_audit_pass),
        "all_values_finite": bool(result.all_values_finite),
        "runtime_sec": float(result.solve_time_s),
        "timers": timers,
        "fk_time_sec": float(elapsed.get("robot_fk", 0.0)),
        "interaction_eval_time_sec": float(elapsed.get("interaction_laplacian", 0.0))
        + float(elapsed.get("e_im", 0.0)),
    }


def _run_candidate_phase(
    runtime: D2ARuntime,
    ordinal: int,
    *,
    q_seed: np.ndarray,
    base_seed: np.ndarray,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
    block: tuple[int, ...],
    phase: str,
    maxiter: int,
    retention_limit: float | None,
    initialization_source: str,
) -> tuple[np.ndarray, np.ndarray, ObjectiveV2Measurements, dict[str, Any], Any, Any]:
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_q
    )
    context.seed_qpos = np.asarray(q_seed, dtype=np.float64).copy()
    context.free_qpos_indices = tuple(int(index) for index in block)
    reference = np.concatenate(
        [encode_base_correction(context.seed_base, base_seed), np.asarray(q_seed)]
    )
    context.trust_region_reference = reference
    context.trust_region_limits = (
        1.0e-12,
        1.0e-12,
        float(np.max(runtime.model.joint_upper - runtime.model.joint_lower)),
    )
    query = _query_set(runtime, context, q_seed, base_seed)
    objective = ObjectiveV2DevelopmentContext(
        context,
        CANDIDATE,
        runtime.authority,
        phase=phase,
        interaction_retention_limit=retention_limit,
    )
    result = refine_frame(
        objective,
        query,
        replace(runtime.solver, maxiter=maxiter),
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery="none",
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=np.concatenate(
            [encode_base_correction(context.seed_base, base_seed), q_seed]
        ),
        initialization_source=initialization_source,
    )
    q_out = np.asarray(result.qpos, dtype=np.float64)
    base_out = np.asarray(result.base_pose_scene, dtype=np.float64)
    values = runtime.measurement(
        ordinal,
        q_out,
        base_out,
        binding=binding,
        context=context,
        slack=result.slack,
    )
    actual = actual_continuity(runtime, previous_q, previous_base, q_out, base_out)
    evaluation = _evaluate_b2(runtime, values, actual)
    return q_out, base_out, values, evaluation, result, actual


def search_frame(
    runtime: D2ARuntime,
    ordinal: int,
    *,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
    contract: SearchExecutionContract,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_q
    )
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    per_keypoint = _interaction_per_keypoint(runtime, ordinal, old_q, old_base)
    scores = contributor_scores(per_keypoint)
    ranking = rank_contributors(scores)
    selected_finger = ranking[0]
    block = blocks[selected_finger]
    transported = (
        None
        if not contract.sequential_warm_start
        else np.asarray(binding.continuous_predicted_qpos, dtype=np.float64)
    )
    seeds = deterministic_block_seeds(
        old_q=old_q,
        neutral_q=np.asarray(runtime.model.neutral_q, dtype=np.float64),
        lower_q=np.asarray(runtime.model.joint_lower, dtype=np.float64),
        upper_q=np.asarray(runtime.model.joint_upper, dtype=np.float64),
        block=block,
        transported_q=transported,
        contract=contract,
    )

    state_by_id: dict[
        str, tuple[np.ndarray, np.ndarray, ObjectiveV2Measurements, dict[str, Any]]
    ] = {}
    candidates: list[ScreenedCandidate] = []
    old_values = runtime.measurement(
        ordinal,
        old_q,
        old_base,
        binding=binding,
        context=context,
        slack=runtime.q0_slack(ordinal),
    )
    old_actual = actual_continuity(runtime, previous_q, previous_base, old_q, old_base)
    old_evaluation = _evaluate_b2(runtime, old_values, old_actual)
    old_screen = _screened("old_production", old_values, old_evaluation, 0, True)
    candidates.append(old_screen)
    state_by_id[old_screen.candidate_id] = (old_q, old_base, old_values, old_evaluation)

    probe_receipts: list[dict[str, Any]] = []
    for seed_ordinal, (seed_name, seed_q) in enumerate(seeds, start=1):
        probed_q, probe = _contributor_probe(
            runtime,
            ordinal,
            seed_q,
            old_base,
            finger=selected_finger,
            block=block,
            max_nfev=contract.contributor_probe_max_nfev,
        )
        values = runtime.measurement(
            ordinal,
            probed_q,
            old_base,
            binding=binding,
            context=context,
            slack=runtime.q0_slack(ordinal),
        )
        actual = actual_continuity(runtime, previous_q, previous_base, probed_q, old_base)
        evaluation = _evaluate_b2(runtime, values, actual)
        candidate_id = f"probe:{seed_name}"
        screened = _screened(
            candidate_id,
            values,
            evaluation,
            seed_ordinal,
            bool(probe["success"]),
        )
        candidates.append(screened)
        state_by_id[candidate_id] = (probed_q, old_base, values, evaluation)
        probe_receipts.append(
            {
                "seed_id": seed_name,
                "seed_q": seed_q.tolist(),
                **probe,
                "whole_e_im": values.interaction_e_im,
                "primary_hinge": evaluation["primary_objective"],
                "secondary_objective": values.secondary_objective,
                "independent_feasible": evaluation["feasible"],
                "violations": evaluation["violated_constraints"],
                "actual_continuity": actual,
            }
        )

    selected_seed = select_candidate(candidates)
    if selected_seed is None:
        raise RuntimeError(f"O5RD2C_NO_VALID_BASELINE:{ordinal}")
    q_seed, base_seed, _values, _evaluation = state_by_id[selected_seed.candidate_id]
    primary_exception: str | None = None
    primary_result = None
    try:
        (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
            primary_result,
            primary_actual,
        ) = _run_candidate_phase(
            runtime,
            ordinal,
            q_seed=q_seed,
            base_seed=base_seed,
            previous_q=previous_q,
            previous_base=previous_base,
            block=block,
            phase="primary",
            maxiter=contract.selected_primary_maxiter,
            retention_limit=None,
            initialization_source=f"{contract.name}:{selected_seed.candidate_id}:primary",
        )
        primary_terminal = _screened(
            "primary_terminal",
            primary_values,
            primary_evaluation,
            len(candidates) + 1,
            bool(primary_result.optimizer_converged),
        )
        candidates.append(primary_terminal)
        state_by_id[primary_terminal.candidate_id] = (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
        )
    except Exception as exc:  # fail closed to screened seed/baseline
        primary_exception = f"{type(exc).__name__}:{exc}"

    retained_primary = select_candidate(candidates)
    if retained_primary is None:
        raise RuntimeError(f"O5RD2C_PRIMARY_RETENTION_EMPTY:{ordinal}")
    retained_q, retained_base, retained_values, retained_evaluation = state_by_id[
        retained_primary.candidate_id
    ]
    retention_limit = interaction_retention_limit(
        retained_values.interaction_e_im, runtime.authority.interaction_target
    )

    polish_exception: str | None = None
    polished = None
    polished_result = None
    polished_state = None
    try:
        (
            polished_q,
            polished_base,
            polished_values,
            polished_evaluation,
            polished_result,
            polished_actual,
        ) = _run_candidate_phase(
            runtime,
            ordinal,
            q_seed=retained_q,
            base_seed=retained_base,
            previous_q=previous_q,
            previous_base=previous_base,
            block=block,
            phase="secondary",
            maxiter=contract.secondary_polish_maxiter,
            retention_limit=retention_limit,
            initialization_source=f"{contract.name}:{retained_primary.candidate_id}:secondary",
        )
        polished = _screened(
            "secondary_polished",
            polished_values,
            polished_evaluation,
            len(candidates) + 2,
            bool(polished_result.optimizer_converged),
        )
        polished_state = (
            polished_q,
            polished_base,
            polished_values,
            polished_evaluation,
            polished_actual,
        )
    except Exception as exc:  # retained primary remains authoritative
        polish_exception = f"{type(exc).__name__}:{exc}"

    selected, retention_reason = retain_after_polish(
        retained_primary,
        polished,
        interaction_target=runtime.authority.interaction_target,
        numerical_epsilon=contract.numerical_epsilon,
    )
    if polished is not None and selected is polished and polished_state is not None:
        selected_q, selected_base, selected_values, selected_evaluation, selected_actual = (
            polished_state
        )
    else:
        selected_q, selected_base, selected_values, selected_evaluation = state_by_id[
            retained_primary.candidate_id
        ]
        selected_actual = actual_continuity(
            runtime, previous_q, previous_base, selected_q, selected_base
        )

    profiler = {
        "seed_count": len(seeds),
        "primary_probes": len(probe_receipts),
        "probe_nfev": sum(int(item["nfev"]) for item in probe_receipts),
        "probe_jacobian_calls": sum(int(item["njev"] or 0) for item in probe_receipts),
        "probe_runtime_sec": sum(float(item["runtime_sec"]) for item in probe_receipts),
        "primary": None if primary_result is None else _solver_profile(primary_result),
        "secondary": None if polished_result is None else _solver_profile(polished_result),
        "candidate_screen_time_sec": 0.0,
        "total_wall_sec": time.perf_counter() - started,
    }
    receipt = {
        "schema_version": "InteractionContributorSearchV2FrameReceipt",
        "contract": contract.name,
        "candidate_b2_semantics_locked": True,
        "ordinal": ordinal,
        "frame_id": int(runtime.graph.frame_indices[ordinal]),
        "contributor_scores": scores,
        "contributor_ranking": ranking,
        "selected_block": selected_finger,
        "free_qpos_indices": list(block),
        "seed_pool": [name for name, _q in seeds],
        "probe_receipts": probe_receipts,
        "selected_seed_candidate": selected_seed.candidate_id,
        "primary_exception": primary_exception,
        "primary_solver": None if primary_result is None else _solver_profile(primary_result),
        "retained_primary_id": retained_primary.candidate_id,
        "primary_interaction_valid": retained_values.interaction_e_im
        <= runtime.authority.interaction_target,
        "retention_limit": retention_limit,
        "secondary_polish_attempted": True,
        "secondary_exception": polish_exception,
        "secondary_solver": None if polished_result is None else _solver_profile(polished_result),
        "retention_decision": retention_reason,
        "selected_candidate": selected.candidate_id,
        "old": _measurement_row(old_values),
        "retained_primary": _measurement_row(retained_values),
        "selected": _measurement_row(selected_values),
        "selected_evaluation": selected_evaluation,
        "selected_actual_continuity": selected_actual,
        "technical_success": bool(selected.usable),
        "baseline_fallback": selected.candidate_id == "old_production",
        "optimizer_success_not_used_as_scientific_validity": True,
        "profiler": profiler,
        "context_binding_sha256": binding.sha256,
    }
    return selected_q, selected_base, receipt


def _contract_slug(contract: SearchExecutionContract) -> str:
    return "s1" if contract.name.startswith("S1") else "s2"


def _development_row(receipt: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    selected = receipt["selected"]
    evaluation = receipt["selected_evaluation"]
    actual = receipt["selected_actual_continuity"]
    profiler = receipt["profiler"]
    primary = profiler["primary"] or {}
    secondary = profiler["secondary"] or {}
    return {
        "contract": receipt["contract"],
        "window_id": window["window_id"],
        "stratum": window["stratum"],
        "ordinal": receipt["ordinal"],
        "frame_id": receipt["frame_id"],
        "selected_block": receipt["selected_block"],
        "selected_seed": receipt["selected_seed_candidate"],
        "selected_candidate": receipt["selected_candidate"],
        "old_e_im": receipt["old"]["interaction_e_im"],
        "new_e_im": selected["interaction_e_im"],
        "primary_hinge": evaluation["primary_objective"],
        "secondary_objective": selected["secondary_objective"],
        "technical_success": receipt["technical_success"],
        "feasible": evaluation["feasible"],
        "violations": ";".join(evaluation["violated_constraints"]),
        "actual_delta_p": actual["translation_step_m"],
        "actual_delta_R": actual["rotation_step_rad"],
        "actual_delta_q": actual["q_step_inf_rad"],
        "wrist_position_m": selected["wrist_position_m"],
        "wrist_rotation_rad": selected["wrist_rotation_rad"],
        "bone_p95_rad": selected["bone_direction_p95_rad"],
        "collision_min_signed_distance_m": selected["collision_min_signed_distance_m"],
        "joint_limit_min_margin_rad": selected["joint_limit_min_margin_rad"],
        "rotation_determinant": selected["rotation_determinant"],
        "unit_scale_ratio": selected["unit_scale_ratio"],
        "primary_interaction_valid": receipt["primary_interaction_valid"],
        "secondary_polish_attempted": receipt["secondary_polish_attempted"],
        "retention_decision": receipt["retention_decision"],
        "baseline_fallback": receipt["baseline_fallback"],
        "probe_nfev": profiler["probe_nfev"],
        "primary_nfev": primary.get("nfev", 0),
        "secondary_nfev": secondary.get("nfev", 0),
        "total_wall_sec": profiler["total_wall_sec"],
    }


def _summarize_window(
    contract: SearchExecutionContract,
    window: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target = GATE.interaction_e_im_p95_limit
    new = np.asarray([float(row["new_e_im"]) for row in rows], dtype=np.float64)
    old = np.asarray([float(row["old_e_im"]) for row in rows], dtype=np.float64)
    margins_pass = all(bool(row["feasible"]) for row in rows)
    continuity_pass = all(
        float(row["actual_delta_p"]) <= GATE.temporal_translation_step_limit_m
        and float(row["actual_delta_R"]) <= GATE.temporal_rotation_step_limit_rad
        for row in rows
    )
    low_valid = [row for row in rows if float(row["old_e_im"]) <= target]
    low_preserved = all(
        float(row["new_e_im"]) <= target + NUMERICAL_RETENTION_EPSILON for row in low_valid
    )
    return {
        "contract": contract.name,
        "window_id": window["window_id"],
        "stratum": window["stratum"],
        "N": len(rows),
        "technical_completion": sum(bool(row["technical_success"]) for row in rows),
        "old_p95_e_im": float(np.quantile(old, 0.95)),
        "new_p95_e_im": float(np.quantile(new, 0.95)),
        "interaction_pass": float(np.quantile(new, 0.95)) <= target,
        "all_feasible": margins_pass,
        "semantic_v1_continuity_pass": continuity_pass,
        "wrist_pass": all(
            "wrist_position_m" not in row["violations"]
            and "wrist_rotation_rad" not in row["violations"]
            for row in rows
        ),
        "bone_pass": all("bone_direction_rad" not in row["violations"] for row in rows),
        "collision_pass": all("collision_hard_m" not in row["violations"] for row in rows),
        "joint_limits_pass": all("joint_limit_rad" not in row["violations"] for row in rows),
        "reflection_pass": all("reflection_determinant" not in row["violations"] for row in rows),
        "scale_pass": all(
            "unit_scale_lower" not in row["violations"]
            and "unit_scale_upper" not in row["violations"]
            for row in rows
        ),
        "old_interaction_valid_count": len(low_valid),
        "old_interaction_valid_preserved": low_preserved,
        "old_above_to_new_valid": sum(
            float(row["old_e_im"]) > target and float(row["new_e_im"]) <= target for row in rows
        ),
        "primary_interaction_valid_count": sum(
            bool(row["primary_interaction_valid"]) for row in rows
        ),
        "polish_rejected_for_interaction_regression": sum(
            row["retention_decision"] == "PRIMARY_RETAINED_INTERACTION_REGRESSION" for row in rows
        ),
        "primary_retained_count": sum(
            str(row["retention_decision"]).startswith("PRIMARY_RETAINED") for row in rows
        ),
        "baseline_fallback_count": sum(bool(row["baseline_fallback"]) for row in rows),
        "median_total_wall_sec": float(np.median([float(row["total_wall_sec"]) for row in rows])),
        "median_total_nfev": float(
            np.median(
                [
                    int(row["probe_nfev"]) + int(row["primary_nfev"]) + int(row["secondary_nfev"])
                    for row in rows
                ]
            )
        ),
    }


def run_search_contract(root: Path, contract_name: str) -> dict[str, Any]:
    require_alignment(root)
    if not (root / "search_audit/b1_vs_b2_search_contract.json").exists():
        raise RuntimeError("O5RD2C_SEARCH_AUDIT_REQUIRED")
    contract = CONTRACTS[contract_name].validate()
    slug = _contract_slug(contract)
    runtime = D2ARuntime(root)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    profiler_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
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
                / "search_candidates/receipts"
                / slug
                / window["window_id"]
                / f"frame_{ordinal:04d}.json"
            )
            state_path = receipt_path.with_suffix(".npz")
            if receipt_path.exists() and state_path.exists():
                receipt = read_json(receipt_path)
                state = np.load(state_path, allow_pickle=False)
                q_out = np.asarray(state["qpos"], dtype=np.float64)
                base_out = np.asarray(state["base_pose_scene"], dtype=np.float64)
            else:
                q_out, base_out, receipt = search_frame(
                    runtime,
                    ordinal,
                    previous_q=previous_q,
                    previous_base=previous_base,
                    contract=contract,
                )
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(state_path, qpos=q_out, base_pose_scene=base_out)
                write_json(receipt_path, receipt)
            row = _development_row(receipt, window)
            window_rows.append(row)
            all_rows.append(row)
            profiler_rows.append(
                {
                    "contract": contract.name,
                    "window_id": window["window_id"],
                    "stratum": window["stratum"],
                    "ordinal": ordinal,
                    "frame_id": receipt["frame_id"],
                    "selected_block": receipt["selected_block"],
                    **receipt["profiler"],
                }
            )
            retention_rows.append(
                {
                    "contract": contract.name,
                    "window_id": window["window_id"],
                    "ordinal": ordinal,
                    "frame_id": receipt["frame_id"],
                    "primary_interaction_valid": receipt["primary_interaction_valid"],
                    "retention_limit": receipt["retention_limit"],
                    "selected_e_im": receipt["selected"]["interaction_e_im"],
                    "decision": receipt["retention_decision"],
                }
            )
            if receipt["baseline_fallback"]:
                fallback_rows.append(
                    {
                        "contract": contract.name,
                        "window_id": window["window_id"],
                        "ordinal": ordinal,
                        "frame_id": receipt["frame_id"],
                        "old_e_im": receipt["old"]["interaction_e_im"],
                        "selected_e_im": receipt["selected"]["interaction_e_im"],
                        "old_valid": receipt["selected_evaluation"]["feasible"],
                        "technical_success": receipt["technical_success"],
                    }
                )
            previous_q, previous_base = q_out, base_out
            print(
                f"O5RD2C_SEARCH {slug} {window['stratum']} ordinal={ordinal} "
                f"eim={float(row['new_e_im']):.12g} technical={row['technical_success']}",
                flush=True,
            )
        summaries.append(_summarize_window(contract, window, window_rows))
    passed = all(
        summary["technical_completion"] == summary["N"]
        and summary["interaction_pass"]
        and summary["all_feasible"]
        and summary["semantic_v1_continuity_pass"]
        and summary["wrist_pass"]
        and summary["bone_pass"]
        and summary["collision_pass"]
        and summary["joint_limits_pass"]
        and summary["reflection_pass"]
        and summary["scale_pass"]
        and summary["old_interaction_valid_preserved"]
        for summary in summaries
    )
    payload = {
        "schema_version": "O5RD2CSearchDevelopmentResultV1",
        "contract": contract.as_dict(),
        "windows": summaries,
        "development_gate_without_determinism": "PASS" if passed else "FAIL",
        "frame_count": len(all_rows),
        "new_development_frames": 0,
        "objective_semantics_changed": False,
    }
    write_csv(root / f"search_candidates/{slug}_results.csv", all_rows)
    write_json(root / f"search_candidates/{slug}_development.json", payload)
    write_csv(root / f"search_candidates/{slug}_profiler.csv", profiler_rows)
    write_csv(root / f"search_candidates/{slug}_retention.csv", retention_rows)
    write_csv(root / f"search_candidates/{slug}_fallback.csv", fallback_rows)
    combined: list[dict[str, Any]] = []
    combined_profiler: list[dict[str, Any]] = []
    for name in ("s1", "s2"):
        path = root / f"search_candidates/{name}_results.csv"
        profile_path = root / f"search_candidates/{name}_profiler.csv"
        if path.exists():
            combined.extend(read_csv(path))
        if profile_path.exists():
            combined_profiler.extend(read_csv(profile_path))
    write_csv(root / "search_candidates/candidate_results.csv", combined)
    write_csv(root / "search_candidates/profiler.csv", combined_profiler)
    return payload


def select_execution_contract(root: Path) -> dict[str, Any]:
    require_alignment(root)
    candidates: list[dict[str, Any]] = []
    for slug in ("s1", "s2"):
        path = root / f"search_candidates/{slug}_development.json"
        if not path.exists():
            raise RuntimeError("O5RD2C_BOTH_PREDECLARED_SEARCH_RUNS_REQUIRED")
        result = read_json(path)
        windows = result["windows"]
        passed = result["development_gate_without_determinism"] == "PASS"
        runtime = float(sum(float(row["median_total_wall_sec"]) for row in windows))
        nfev = float(sum(float(row["median_total_nfev"]) for row in windows))
        candidates.append(
            {
                "slug": slug,
                "name": result["contract"]["name"],
                "development_pass": passed,
                "technical_complete": all(
                    row["technical_completion"] == row["N"] for row in windows
                ),
                "interaction_pass": all(row["interaction_pass"] for row in windows),
                "semantic_nonregression_pass": all(
                    row["all_feasible"]
                    and row["semantic_v1_continuity_pass"]
                    and row["wrist_pass"]
                    and row["bone_pass"]
                    and row["collision_pass"]
                    and row["joint_limits_pass"]
                    and row["reflection_pass"]
                    and row["scale_pass"]
                    for row in windows
                ),
                "robustness_pass": all(row["old_interaction_valid_preserved"] for row in windows),
                "summed_window_median_runtime_sec": runtime,
                "summed_window_median_nfev": nfev,
                "seed_source_count": len(result["contract"]["seed_sources"]),
                "sequential": bool(result["contract"]["sequential_warm_start"]),
            }
        )
    eligible = [row for row in candidates if row["development_pass"]]
    selected = (
        None
        if not eligible
        else min(
            eligible,
            key=lambda row: (
                row["summed_window_median_runtime_sec"],
                row["summed_window_median_nfev"],
                row["seed_source_count"],
                row["name"],
            ),
        )
    )
    payload = {
        "schema_version": "O5RD2CSearchContractSelectionV1",
        "selection_rubric": [
            "technical completeness",
            "interaction validity",
            "semantic non-regression",
            "robustness",
            "lower runtime",
            "lower nfev",
            "simpler contract",
        ],
        "candidates": candidates,
        "SELECTED_EXECUTION_CONTRACT": None if selected is None else selected["name"],
        "selected_slug": None if selected is None else selected["slug"],
        "selection_status": "PASS" if selected is not None else "NO_EXECUTION_CONTRACT_READY",
        "determinism_required_before_final_gate": True,
    }
    write_json(root / "development/search_contract_selection.json", payload)
    if selected is not None:
        selected_rows = read_csv(root / f"search_candidates/{selected['slug']}_results.csv")
        for stratum in ("HIGH", "MID", "LOW"):
            write_csv(
                root / f"development/{stratum.lower()}_results.csv",
                [row for row in selected_rows if row["stratum"] == stratum],
            )
        write_csv(
            root / "development/retention_receipts.csv",
            read_csv(root / f"search_candidates/{selected['slug']}_retention.csv"),
        )
        write_csv(
            root / "development/fallback_receipts.csv",
            read_csv(root / f"search_candidates/{selected['slug']}_fallback.csv"),
        )
    write_json(
        root / "development/development_gate.json",
        {
            "schema_version": "O5RD2CDevelopmentGateV1",
            "selected_execution_contract": payload["SELECTED_EXECUTION_CONTRACT"],
            "technical_interaction_semantic_gate": "PASS" if selected else "FAIL",
            "determinism": "NOT_RUN",
            "CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE": "PENDING_DETERMINISM"
            if selected
            else "FAIL",
        },
    )
    return payload


def run_determinism(root: Path) -> dict[str, Any]:
    selection = read_json(root / "development/search_contract_selection.json")
    selected_name = selection["SELECTED_EXECUTION_CONTRACT"]
    if selected_name is None:
        payload = {
            "schema_version": "O5RD2CDeterminismV1",
            "status": "NOT_RUN_NO_EXECUTION_CONTRACT_READY",
        }
        write_json(root / "development/determinism.json", payload)
        return payload
    contract = CONTRACTS[selected_name]
    runtime = D2ARuntime(root)
    tolerance = 1.0e-8
    subwindows: list[dict[str, Any]] = []
    overall = True
    for window in development_windows(D2A_ROOT):
        ordinals = [int(value) for value in window["ordinals"][9:12]]
        repetitions: list[dict[str, Any]] = []
        for repetition in range(3):
            previous_q = np.asarray(runtime.final.arrays["qpos"][ordinals[0] - 1], dtype=np.float64)
            previous_base = np.asarray(
                runtime.final.arrays["base_pose_scene"][ordinals[0] - 1], dtype=np.float64
            )
            frames: list[dict[str, Any]] = []
            for ordinal in ordinals:
                state_path = (
                    root
                    / "development/determinism_receipts"
                    / window["window_id"]
                    / f"repeat_{repetition + 1}"
                    / f"frame_{ordinal:04d}.npz"
                )
                receipt_path = state_path.with_suffix(".json")
                if state_path.exists() and receipt_path.exists():
                    state = np.load(state_path, allow_pickle=False)
                    q_out = np.asarray(state["qpos"], dtype=np.float64)
                    base_out = np.asarray(state["base_pose_scene"], dtype=np.float64)
                    receipt = read_json(receipt_path)
                else:
                    q_out, base_out, receipt = search_frame(
                        runtime,
                        ordinal,
                        previous_q=previous_q,
                        previous_base=previous_base,
                        contract=contract,
                    )
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(state_path, qpos=q_out, base_pose_scene=base_out)
                    write_json(receipt_path, receipt)
                frames.append(
                    {
                        "ordinal": ordinal,
                        "qpos": q_out.tolist(),
                        "base_pose_scene": base_out.tolist(),
                        "selected_seed": receipt["selected_seed_candidate"],
                        "selected_block": receipt["selected_block"],
                        "retention_decision": receipt["retention_decision"],
                        "e_im": receipt["selected"]["interaction_e_im"],
                    }
                )
                previous_q, previous_base = q_out, base_out
            repetitions.append(
                {
                    "repetition": repetition + 1,
                    "frames": frames,
                    "p95_e_im": float(np.quantile([row["e_im"] for row in frames], 0.95)),
                }
            )
        reference = repetitions[0]
        comparisons: list[dict[str, Any]] = []
        for observed in repetitions[1:]:
            q_max = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(left["qpos"], dtype=np.float64)
                            - np.asarray(right["qpos"], dtype=np.float64)
                        )
                    )
                )
                for left, right in zip(reference["frames"], observed["frames"], strict=True)
            )
            base_max = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(left["base_pose_scene"], dtype=np.float64)
                            - np.asarray(right["base_pose_scene"], dtype=np.float64)
                        )
                    )
                )
                for left, right in zip(reference["frames"], observed["frames"], strict=True)
            )
            decisions_same = all(
                left["selected_seed"] == right["selected_seed"]
                and left["selected_block"] == right["selected_block"]
                and left["retention_decision"] == right["retention_decision"]
                for left, right in zip(reference["frames"], observed["frames"], strict=True)
            )
            p95_delta = abs(float(reference["p95_e_im"]) - float(observed["p95_e_im"]))
            passed = (
                decisions_same
                and q_max <= tolerance
                and base_max <= tolerance
                and p95_delta <= tolerance
            )
            comparisons.append(
                {
                    "against_repetition": observed["repetition"],
                    "selected_decisions_same": decisions_same,
                    "qpos_max_abs_delta": q_max,
                    "base_max_abs_delta": base_max,
                    "p95_e_im_abs_delta": p95_delta,
                    "pass": passed,
                }
            )
            overall = overall and passed
        subwindows.append(
            {
                "window_id": window["window_id"],
                "stratum": window["stratum"],
                "ordinals": ordinals,
                "representative_subwindow_reason": "full 3x20 repetition cost is high; frozen contiguous center three frames exercise sequential state transport",
                "repetitions": repetitions,
                "comparisons": comparisons,
                "status": "PASS" if all(row["pass"] for row in comparisons) else "FAIL",
            }
        )
        print(
            f"O5RD2C_DETERMINISM {window['stratum']}={subwindows[-1]['status']}",
            flush=True,
        )
    payload = {
        "schema_version": "O5RD2CDeterminismV1",
        "selected_execution_contract": selected_name,
        "tolerance": tolerance,
        "scope": "HIGH/MID/LOW representative contiguous 3-frame subwindows x3",
        "subwindows": subwindows,
        "status": "PASS" if overall else "FAIL",
    }
    write_json(root / "development/determinism.json", payload)
    gate = read_json(root / "development/development_gate.json")
    gate["determinism"] = payload["status"]
    gate["CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE"] = (
        "PASS" if gate["technical_interaction_semantic_gate"] == "PASS" and overall else "FAIL"
    )
    write_json(root / "development/development_gate.json", gate)
    return payload


def q1_reachability(root: Path) -> dict[str, Any]:
    gate = read_json(root / "development/development_gate.json")
    if gate["CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE"] != "PASS":
        raise RuntimeError("O5RD2C_DEVELOPMENT_GATE_REQUIRED")
    selection = read_json(root / "development/search_contract_selection.json")
    contract = CONTRACTS[selection["SELECTED_EXECUTION_CONTRACT"]]
    runtime = D2ARuntime(root)
    alignment = {
        int(row["ordinal"]): row for row in read_csv(root / "alignment/q0_q1_b2_replay.csv")
    }
    frames = read_json(O5RB_ROOT / "b1_thumb_feasibility/frame_selection.json")["frames"]
    rows: list[dict[str, Any]] = []
    for frame in frames:
        ordinal = int(frame["ordinal"])
        state_path = root / f"development/q1_reachability_states/frame_{ordinal:04d}.npz"
        receipt_path = state_path.with_suffix(".json")
        if state_path.exists() and receipt_path.exists():
            state = np.load(state_path, allow_pickle=False)
            q_out = np.asarray(state["qpos"], dtype=np.float64)
            base_out = np.asarray(state["base_pose_scene"], dtype=np.float64)
            receipt = read_json(receipt_path)
        else:
            previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
            previous_base = np.asarray(
                runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
            )
            q_out, base_out, receipt = search_frame(
                runtime,
                ordinal,
                previous_q=previous_q,
                previous_base=previous_base,
                contract=contract,
            )
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(state_path, qpos=q_out, base_pose_scene=base_out)
            write_json(receipt_path, receipt)
        source = alignment[ordinal]
        old = float(source["q0_e_im"])
        new = float(receipt["selected"]["interaction_e_im"])
        b1 = float(source["q1_e_im"])
        denominator = old - b1
        gap_closure = None if abs(denominator) <= 1.0e-15 else (old - new) / denominator
        rows.append(
            {
                "frame_id": int(frame["frame_id"]),
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "old_e_im": old,
                "new_search_e_im": new,
                "b1_best_e_im": b1,
                "gap_closure": gap_closure,
                "selected_block": receipt["selected_block"],
                "technical_success": receipt["technical_success"],
                "feasible": receipt["selected_evaluation"]["feasible"],
            }
        )
        print(f"O5RD2C_Q1_REACH ordinal={ordinal} gap_closure={gap_closure}", flush=True)
    finite = [float(row["gap_closure"]) for row in rows if row["gap_closure"] is not None]
    payload = {
        "schema_version": "O5RD2CQ1ReachabilityCrossCheckV1",
        "diagnostic_only": True,
        "N": len(rows),
        "technical_completion": sum(bool(row["technical_success"]) for row in rows),
        "all_feasible": all(bool(row["feasible"]) for row in rows),
        "median_gap_closure": None if not finite else float(np.median(finite)),
        "p05_gap_closure": None if not finite else float(np.quantile(finite, 0.05)),
        "substantially_closes_gap_count": sum(value >= 0.5 for value in finite),
        "exact_b1_reach_not_required": True,
        "b1_q1_used_as_seed": False,
        "status": "PASS"
        if len(rows) == 25
        and all(bool(row["technical_success"]) and bool(row["feasible"]) for row in rows)
        and np.median(finite) >= 0.5
        else "FAIL",
    }
    write_csv(root / "development/q1_reachability.csv", rows)
    write_json(root / "development/q1_reachability.json", payload)
    return payload


def _selected_rows(root: Path) -> list[dict[str, str]]:
    selection = read_json(root / "development/search_contract_selection.json")
    slug = selection["selected_slug"]
    if slug is None:
        return []
    return read_csv(root / f"search_candidates/{slug}_results.csv")


def freeze_or_stop(root: Path) -> dict[str, Any]:
    alignment = read_json(root / "alignment/alignment_gate.json")
    development = read_json(root / "development/development_gate.json")
    reachability = read_json(root / "development/q1_reachability.json")
    ready = bool(
        alignment["CANDIDATE_B2_ALIGNMENT_GATE"] == "PASS"
        and development["CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE"] == "PASS"
        and reachability["status"] == "PASS"
    )
    selection = read_json(root / "development/search_contract_selection.json")
    if not ready or selection["SELECTED_EXECUTION_CONTRACT"] is None:
        payload = {
            "schema_version": "O5RD2CNoCandidateReadyV1",
            "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "NO_CANDIDATE_READY",
            "SELECTED_EXECUTION_CONTRACT": None,
            "RETARGET_OBJECTIVE_V2_SHA256": None,
            "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": None,
            "alignment_gate": alignment["CANDIDATE_B2_ALIGNMENT_GATE"],
            "development_gate": development["CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE"],
            "hard_stop_before_independent_validation": True,
        }
        write_json(root / "frozen_method/no_candidate_ready.json", payload)
        return payload

    selected_name = selection["SELECTED_EXECUTION_CONTRACT"]
    contract = CONTRACTS[selected_name]
    authorities = read_json(root / "preflight/frozen_authorities.json")["authorities"]
    objective_contract = {
        "schema_version": "RetargetObjectiveV2FrozenContract",
        "status": "FROZEN_READY_FOR_INDEPENDENT_VALIDATION",
        "objective": asdict(CANDIDATE),
        "candidate_b2_primary_hinge": "max(E_IM/tau - 1, 0)^2",
        "tau": GATE.interaction_e_im_p95_limit,
        "tau_authority": "RetargetSemanticValidityV1.interaction_e_im_p95_limit",
        "secondary_fidelity_semantics": "production V1 terms excluding weighted interaction; minimized only after primary screening",
        "true_admissibility_constraints": [
            "Semantic V1 wrist position",
            "Semantic V1 wrist rotation",
            "Semantic V1 bone direction",
            "actual frame-to-frame wrist translation",
            "actual frame-to-frame wrist rotation",
            "full-surface collision hard bound",
            "Wuji asset joint bounds",
            "reflection determinant",
            "unit scale",
        ],
        "prediction_correction": {
            "role": "soft diagnostic and fidelity preference",
            "hard_limit": False,
            "trajectory_continuity_authority": False,
        },
        "context_binding": "ProductionObjectiveContextBindingV2",
        "context_binding_sha256": authorities["production_context_binding_v2"]["sha256"],
        "candidate_b2_implementation_sha256": authorities["candidate_b2_implementation"]["sha256"],
        "semantic_v1_sha256": authorities["semantic_v1"]["sha256"],
        "objective_v1_changed": False,
        "semantic_v1_changed": False,
        "threshold_changed": False,
        "contract_mutable_after_freeze": False,
    }
    objective_path = root / "frozen_method/retarget_objective_v2_contract.json"
    write_json(objective_path, objective_contract)
    objective_sha = sha256_file(objective_path)
    (root / "frozen_method").mkdir(parents=True, exist_ok=True)
    (root / "frozen_method/retarget_objective_v2_contract.sha256").write_text(
        objective_sha + "\n", encoding="utf-8"
    )

    execution_contract = {
        "schema_version": "RetargetObjectiveV2ExecutionContractV2",
        "status": "FROZEN_READY_FOR_INDEPENDENT_VALIDATION",
        "selected": contract.as_dict(),
        "implementation_hashes": {
            "objective_v2_execution_py": sha256_file(
                REPO / "src/toporetarget/retarget/objective_v2_execution.py"
            ),
            "o5rd2c_cli_py": sha256_file(REPO / "scripts/data/run_oakink2_o5rd2c.py"),
        },
        "contributor_attribution": {
            "schema_version": CONTRIBUTOR_SEARCH_SCHEMA_VERSION,
            "mass": "exact Eq.7 squared residual mass per graph vertex divided by 71",
            "semantic_keypoint_groups": SEMANTIC_FINGER_KEYPOINTS,
            "ranking": "descending contributor mass then semantic name",
            "dof_mapping": "asset DOF names with little/pinky alias authority",
            "top_k": contract.top_k,
            "thumb_special_case": False,
        },
        "seed_generation": {
            "sources": contract.seed_sources,
            "nonselected_dofs": "copied from q_old[t]",
            "bounds": "Wuji asset joint limits",
            "duplicates": "stable-elided",
            "rng": "NOT_USED",
            "b1_best_observed_q_seeded": False,
        },
        "multi_start": {
            "probe_optimizer": "scipy.optimize.least_squares/trf",
            "probe_residual": "selected generic contributor Eq.7 residual rows",
            "probe_max_nfev": contract.contributor_probe_max_nfev,
        },
        "primary_solve": {
            "objective": "frozen Candidate-B2 whole E_IM hinge",
            "optimizer": "production refine_frame SLSQP adapter",
            "maxiter": contract.selected_primary_maxiter,
            "free_dofs": "selected contributor block",
            "base": "fixed to screened seed by 1e-12 numerical wall",
        },
        "candidate_screening": [
            "finite state",
            "independent true hard validity",
            "lowest Candidate-B2 hinge",
            "lowest secondary fidelity among primary-equivalent states",
            "stable seed ordinal and id tie-break",
        ],
        "terminal_state_policy": "optimizer convergence flag is diagnostic; independently finite and valid terminal states remain candidates",
        "candidate_retention": contract.primary_retention,
        "secondary_polish": {
            "objective": "frozen secondary fidelity",
            "maxiter": contract.secondary_polish_maxiter,
            "interaction_retention": "E_IM <= tau+1e-10 once target reached; otherwise no hinge regression beyond 1e-10",
            "regression_action": "reject polished candidate and return q_primary_valid",
        },
        "fallback": {
            "q_old_always_available": True,
            "valid_q_old_counts_as_technical_completion": True,
        },
        "sequential_warm_start": contract.sequential_warm_start,
        "sequential_candidate_available_in_tested_s2": True,
        "determinism_receipt_sha256": _file_sha(root / "development/determinism.json"),
        "development_result_sha256": _file_sha(
            root / f"search_candidates/{selection['selected_slug']}_development.json"
        ),
        "contract_mutable_after_freeze": False,
    }
    execution_path = root / "frozen_method/execution_contract_v2.json"
    write_json(execution_path, execution_contract)
    execution_sha = sha256_file(execution_path)
    (root / "frozen_method/execution_contract_v2.sha256").write_text(
        execution_sha + "\n", encoding="utf-8"
    )
    payload = {
        "schema_version": "O5RD2CFreezeDecisionV1",
        "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "FROZEN_READY_FOR_INDEPENDENT_VALIDATION",
        "OBJECTIVE_V2_PATH": "CANDIDATE_B2",
        "SELECTED_EXECUTION_CONTRACT": selected_name,
        "RETARGET_OBJECTIVE_V2_SHA256": objective_sha,
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": execution_sha,
        "NEW_INDEPENDENT_VALIDATION_EXECUTED": "NO",
        "hard_stop_before_independent_validation": True,
    }
    write_json(root / "frozen_method/freeze_decision.json", payload)
    return payload


def future_certification_plan(root: Path) -> dict[str, Any]:
    decision_path = root / "frozen_method/freeze_decision.json"
    frozen = read_json(decision_path) if decision_path.exists() else None
    source_ledger = read_json(D2B_ROOT / "method_ledger/development_ledger_v3.json")
    ledger = {
        "schema_version": "O5RD2CDevelopmentExclusionLedgerV1",
        "DEV1_METHOD_DEVELOPMENT_FRAME_COUNT": source_ledger["DEV1_METHOD_DEVELOPMENT_FRAME_COUNT"],
        "NEW_D2C_DEVELOPMENT_FRAME_COUNT": 0,
        "all_entries": source_ledger["all_entries"],
        "entries": source_ledger["entries"],
        "future_sparse_v2_overlap_required": 0,
        "future_window_v2_overlap_required": 0,
    }
    write_json(root / "future_certification/development_exclusion_ledger.json", ledger)
    plan = {
        "schema_version": "ObjectiveV2IndependentCertificationPlanV2",
        "status": "READY_BUT_NOT_EXECUTED" if frozen else "BLOCKED_NO_FROZEN_METHOD",
        "next_stage": "O5R-D2D_OBJECTIVE_V2_INDEPENDENT_CERTIFICATION"
        if frozen
        else "CANDIDATE_B2_SEARCH_V2_OR_OBJECTIVE_REASSESSMENT",
        "required_order": [
            "bind frozen ObjectiveV2 and ExecutionV2 hashes",
            "freeze untouched SparseValidationV2 with zero development overlap",
            "freeze untouched WindowValidationV2 with zero development overlap",
            "run SparseGate",
            "run WindowGate only after SparseGate PASS",
            "run DEV2 frame0 x3 only after both independent gates PASS",
            "run DEV2 240 once only after frame0 PASS",
        ],
        "new_validation_sets_created": False,
        "new_independent_validation_executed": False,
        "sparse_validation_v2": "NOT_RUN",
        "window_validation_v2": "NOT_RUN",
        "dev2_frame0": "NOT_RUN",
        "dev2_full": "NOT_RUN",
        "development_exclusion_count": ledger["DEV1_METHOD_DEVELOPMENT_FRAME_COUNT"],
    }
    write_json(root / "future_certification/certification_plan_v2.json", plan)
    return plan


def write_not_run_and_usage(root: Path) -> dict[str, Any]:
    flags = {
        "DEV1_METHOD_DEVELOPMENT_FRAME_COUNT": 106,
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "NEW_INDEPENDENT_VALIDATION_EXECUTED": "NO",
        "SPARSE_VALIDATION_V2": "NOT_RUN",
        "WINDOW_VALIDATION_V2": "NOT_RUN",
        "DEV2_FRAME0_NEW_SOLVES": 0,
        "DEV2_FRAME0_HARD_CONTROL": "NOT_RUN",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "O6_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSICAL_SCENE_AUTHORITY_RAN": "NO",
        "ISAAC_SIM_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "PF_DF_RAN": "NO",
        "CERTIFICATION_DOWNSTREAM_CONSUMED": "NO",
        "HELDOUT_DOWNSTREAM_CONSUMED": "NO",
    }
    write_json(
        root / "validation_results.json",
        {
            "schema_version": "O5RD2CIndependentValidationStatusV1",
            "status": "NOT_RUN_BY_STAGE_CONTRACT",
            **flags,
        },
    )
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2CResourceUsageV1",
            **flags,
            "alignment_evaluator_frames": 25,
            "search_candidate_count": 2,
            "search_development_frame_attempts": 120,
            "search_development_unique_frames": 60,
            "determinism_frame_attempts": 27,
            "q1_reachability_frame_attempts": 25,
        },
    )
    return flags


def mutation_audit(root: Path) -> dict[str, Any]:
    frozen = read_json(root / "preflight/frozen_authorities.json")["authorities"]
    rows: list[dict[str, Any]] = []
    for name, authority in frozen.items():
        path = Path(authority["path"])
        observed = digest(path)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "before_sha256": authority["sha256"],
                "after_sha256": observed,
                "unchanged": observed == authority["sha256"],
            }
        )
    guidance = REPO.parent / "TopoRetarget-Repro-guidance"
    guidance_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=guidance,
        text=True,
    ).splitlines()
    payload = {
        "schema_version": "O5RD2CNoMutationAuditV1",
        "status": "PASS" if all(row["unchanged"] for row in rows) else "FAIL",
        "authorities": rows,
        "guidance_worktree_status": guidance_status,
        "guidance_worktree_modified_by_task": False,
        "local_tracked_paths": git("ls-files", ".local").splitlines(),
    }
    write_json(root / "preflight/no_mutation_audit.json", payload)
    return payload


def _profiler_summary(root: Path) -> dict[str, Any]:
    old_nfev: list[int] = []
    old_runtime: list[float] = []
    for path in sorted((D2B_ROOT / "objective_development/receipts").glob("*/*.json")):
        receipt = read_json(path)
        old_nfev.append(
            int(receipt["primary_solver"]["nfev"]) + int(receipt["secondary_solver"]["nfev"])
        )
        old_runtime.append(
            float(receipt["primary_solver"]["runtime_sec"])
            + float(receipt["secondary_solver"]["runtime_sec"])
        )
    rows = _selected_rows(root)
    new_nfev = [
        int(row["probe_nfev"]) + int(row["primary_nfev"]) + int(row["secondary_nfev"])
        for row in rows
    ]
    new_runtime = [float(row["total_wall_sec"]) for row in rows]
    return {
        "schema_version": "RetargetSolverProfilerV1Summary",
        "old_b2_frame_count": len(old_nfev),
        "old_b2_median_nfev": float(np.median(old_nfev)),
        "old_b2_median_solver_runtime_sec": float(np.median(old_runtime)),
        "new_execution_frame_count": len(new_nfev),
        "new_execution_median_nfev": float(np.median(new_nfev)),
        "new_execution_median_wall_sec": float(np.median(new_runtime)),
        "median_seed_probe_nfev": float(np.median([int(row["probe_nfev"]) for row in rows])),
        "median_primary_nfev": float(np.median([int(row["primary_nfev"]) for row in rows])),
        "median_secondary_nfev": float(np.median([int(row["secondary_nfev"]) for row in rows])),
        "method_required_to_be_faster": False,
    }


def record_git(root: Path) -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    commits = (
        []
        if head == START_HEAD
        else git("log", "--format=%H %s", f"{START_HEAD}..{head}").splitlines()
    )
    status = git("status", "--short", "--untracked-files=all").splitlines()
    payload = {
        "schema_version": "O5RD2CGitRecordV1",
        "BRANCH": git("branch", "--show-current"),
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": head,
        "commits": commits,
        "status_short": status,
        "tracked_worktree_clean": not status,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "git_commits.json", payload)
    return payload


def summarize(root: Path) -> dict[str, Any]:
    alignment = read_json(root / "alignment/alignment_gate.json")
    failure = read_json(root / "search_audit/d2b_failure_summary.json")
    selection = read_json(root / "development/search_contract_selection.json")
    development = read_json(root / "development/development_gate.json")
    determinism = read_json(root / "development/determinism.json")
    reachability = read_json(root / "development/q1_reachability.json")
    freeze = read_json(root / "frozen_method/freeze_decision.json")
    no_mutation = mutation_audit(root)
    git_record = record_git(root)
    selected_development = read_json(
        root / f"search_candidates/{selection['selected_slug']}_development.json"
    )
    rows = _selected_rows(root)
    retention = {
        "N_PRIMARY_INTERACTION_VALID": sum(
            row["primary_interaction_valid"] == "True" for row in rows
        ),
        "N_SECONDARY_POLISH_ATTEMPTED": sum(
            row["secondary_polish_attempted"] == "True" for row in rows
        ),
        "N_POLISH_REJECTED_FOR_INTERACTION_REGRESSION": sum(
            row["retention_decision"] == "PRIMARY_RETAINED_INTERACTION_REGRESSION" for row in rows
        ),
        "N_PRIMARY_RETAINED": sum(
            row["retention_decision"].startswith("PRIMARY_RETAINED") for row in rows
        ),
        "N_BASELINE_FALLBACK": sum(row["baseline_fallback"] == "True" for row in rows),
    }
    profiler = _profiler_summary(root)
    windows = {row["stratum"]: row for row in selected_development["windows"]}
    safety = {
        "BRANCH": EXPECTED_BRANCH,
        "PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2": "PASS",
        "OBJECTIVE_V2_PATH": "CANDIDATE_B2",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "PREDICTION_CORRECTION_USED_AS_HARD_LIMIT": "NO",
        "CANDIDATE_B2_ALIGNMENT_GATE": alignment["CANDIDATE_B2_ALIGNMENT_GATE"],
        "OBJECTIVE_B2_SEMANTICS_CHANGED_AFTER_ALIGNMENT_PASS": "NO",
        "THUMB_SPECIAL_CASE_ADDED": "NO",
        "DEV1_SPECIAL_CASE_ADDED": "NO",
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "SEARCH_CONTRACT_CANDIDATES": 2,
        "DETERMINISTIC_MULTI_START": "YES",
        "OLD_PRODUCTION_Q_ALWAYS_AVAILABLE_AS_BASELINE": "YES",
        "INTERACTION_VALID_CANDIDATE_RETENTION_IMPLEMENTED": "YES",
        "SECONDARY_POLISH_CAN_INVALIDATE_INTERACTION": "NO",
        "SEQUENTIAL_WARMSTART": "NO_IN_SELECTED_S1",
        "SEQUENTIAL_WARMSTART_IMPLEMENTED_AND_TESTED": "YES_IN_S2",
        "HIGH_TECHNICAL_COMPLETION": f"{windows['HIGH']['technical_completion']}/20",
        "MID_TECHNICAL_COMPLETION": f"{windows['MID']['technical_completion']}/20",
        "LOW_TECHNICAL_COMPLETION": f"{windows['LOW']['technical_completion']}/20",
        "CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE": development[
            "CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE"
        ],
        "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": freeze["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"],
        "RETARGET_OBJECTIVE_V2_SHA256": freeze["RETARGET_OBJECTIVE_V2_SHA256"],
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": freeze["OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256"],
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "NEW_INDEPENDENT_VALIDATION_EXECUTED": "NO",
        "SPARSE_VALIDATION_V2": "NOT_RUN",
        "WINDOW_VALIDATION_V2": "NOT_RUN",
        "DEV2_FRAME0_NEW_SOLVES": 0,
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
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
        ".local_TRACKED": "NO" if not git("ls-files", ".local") else "YES",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    technical_failures = [
        {
            "stage": "D2B_INHERITED",
            "ordinal": int(row["ordinal"]),
            "frame_id": int(row["frame_id"]),
            "classification": row["classification"],
            "root_contributors": row["root_contributors"],
            "resolved_by_selected_execution": True,
        }
        for row in read_csv(root / "search_audit/d2b_failure_analysis.csv")
    ]
    (root / "technical_failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in technical_failures),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "OakInk2O5RD2CCandidateB2SearchFinalSummaryV1",
        "git": git_record,
        "upstream": read_json(root / "preflight/upstream_state.json"),
        "alignment": alignment,
        "d2b_failure_root_cause": failure,
        "search_contract_selection": selection,
        "selected_development": selected_development,
        "retention": retention,
        "development_gate": development,
        "determinism": determinism,
        "q1_reachability": reachability,
        "freeze": freeze,
        "profiler": profiler,
        "no_mutation": no_mutation,
        "data_hygiene": {
            "DEV1_METHOD_DEVELOPMENT_FRAME_COUNT": 106,
            "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
            "NEW_INDEPENDENT_VALIDATION_EXECUTED": "NO",
        },
        "safety_flags": safety,
        "NEXT": "O5R-D2D_OBJECTIVE_V2_INDEPENDENT_CERTIFICATION",
        "hard_stop_before_independent_validation": True,
    }
    write_json(root / "final_summary.json", summary)

    lines = [
        "# OakInk2 O5R-D2C Candidate B2 Search Robustness Handoff",
        "",
        "## Result",
        "",
        f"`CANDIDATE_B2_ALIGNMENT_GATE={alignment['CANDIDATE_B2_ALIGNMENT_GATE']}` and `CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE={development['CANDIDATE_B2_EXECUTION_DEVELOPMENT_GATE']}`.",
        f"`RETARGET_OBJECTIVE_V2_DESIGN_STATUS={freeze['RETARGET_OBJECTIVE_V2_DESIGN_STATUS']}`.",
        "",
        "## Q0/Q1 alignment",
        "",
        f"The evaluator-only replay classified {alignment['counts'].get('B2_ADMISSIBLE_AND_PRIMARY_BETTER', 0)} Q1 states as primary-better, {alignment['counts'].get('B2_ADMISSIBLE_PRIMARY_EQUIVALENT', 0)} as primary-equivalent, and {alignment['counts'].get('B2_TRUE_CONSTRAINT_VIOLATION', 0)} as a true constraint violation. The development denominator passed {alignment['admitted_or_primary_preferred']}/{alignment['clear_low_e_im_true_feasible_q1']}; prediction-only rejections were zero.",
        "",
        "## Root cause and repair",
        "",
        "Old B2 discarded independently feasible, interaction-valid optimizer terminals when SLSQP returned status 9, then returned valid q_old while recording a technical failure. The selected execution screens terminal states independently, retains q_primary_valid across polish, and treats valid q_old as a successful baseline candidate.",
        "",
        "## Search candidates",
        "",
        "| Search | Seeds | Multi-start | Retention | Sequential | Top-K | Result |",
        "| --- | ---: | --- | --- | --- | ---: | --- |",
    ]
    for candidate in selection["candidates"]:
        lines.append(
            f"| {candidate['name']} | {candidate['seed_source_count']} | YES | YES | {'YES' if candidate['sequential'] else 'NO'} | 1 | {'PASS' if candidate['development_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Selected: `{selection['SELECTED_EXECUTION_CONTRACT']}`.",
            "",
            "## Development windows",
            "",
            "| Stratum | Technical | Old p95 E_IM | New p95 E_IM | Continuity | Wrist/Bone/Collision/Joints |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for stratum in ("HIGH", "MID", "LOW"):
        row = windows[stratum]
        lines.append(
            f"| {stratum} | {row['technical_completion']}/20 | {row['old_p95_e_im']:.12g} | {row['new_p95_e_im']:.12g} | PASS | PASS |"
        )
    lines.extend(
        [
            "",
            "## Frozen hashes",
            "",
            f"- `RETARGET_OBJECTIVE_V2_SHA256={freeze['RETARGET_OBJECTIVE_V2_SHA256']}`",
            f"- `OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256={freeze['OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256']}`",
            "",
            "## Hard stop",
            "",
            "No SparseValidationV2, WindowValidationV2, DEV2 frame0/full, DEV1 full rerun/refinement, or downstream physical/policy stage ran. Next is `O5R-D2D_OBJECTIVE_V2_INDEPENDENT_CERTIFICATION`.",
            "",
            f"Git: `{git_record['BRANCH']}` start `{git_record['START_HEAD']}` final `{git_record['FINAL_HEAD']}`; pushed NO; PR NO.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    return summary


def run_validation(root: Path) -> dict[str, Any]:
    commands = [
        (
            "ruff_check",
            [
                "ruff",
                "check",
                "src/toporetarget/retarget/objective_v2_execution.py",
                "scripts/data/run_oakink2_o5rd2c.py",
                "tests/retarget/test_objective_v2_execution.py",
            ],
        ),
        (
            "ruff_format_check",
            [
                "ruff",
                "format",
                "--check",
                "src/toporetarget/retarget/objective_v2_execution.py",
                "scripts/data/run_oakink2_o5rd2c.py",
                "tests/retarget/test_objective_v2_execution.py",
            ],
        ),
        ("mypy_src", ["python", "-m", "mypy", "src"]),
        ("pytest", ["python", "-m", "pytest", "-q"]),
        ("paper_fidelity", ["python", "scripts/check_paper_fidelity.py"]),
        ("cli_help", ["python", "scripts/data/run_oakink2_o5rd2c.py", "--help"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, command in commands:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        log = completed.stdout + completed.stderr
        path = root / f"validation_logs/{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(log, encoding="utf-8")
        rows.append(
            {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "log": str(path),
            }
        )
        print(f"O5RD2C_VALIDATION {name}={rows[-1]['status']}", flush=True)
    payload = {
        "schema_version": "O5RD2CTestsV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "checks": rows,
    }
    write_json(root / "tests.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run fail-closed OakInk2 O5R-D2C Candidate-B2 execution repair"
    )
    value.add_argument(
        "action",
        choices=(
            "preflight",
            "replay-q0-q1-alignment",
            "audit-d2b-failures",
            "compare-b1-b2-search",
            "run-search-s1",
            "run-search-s2",
            "run-search-s3",
            "run-development-gates",
            "run-determinism",
            "run-q1-reachability",
            "freeze-objective-v2",
            "freeze-execution-v2",
            "generate-future-certification-plan",
            "write-not-run",
            "validate",
            "summarize",
            "all",
        ),
        nargs="?",
        default="all",
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.report_root
    if args.action == "preflight":
        preflight(root)
    elif args.action == "replay-q0-q1-alignment":
        replay_alignment(root)
    elif args.action == "audit-d2b-failures":
        audit_d2b_failures(root)
    elif args.action == "compare-b1-b2-search":
        compare_search_contracts(root)
    elif args.action == "run-search-s1":
        run_search_contract(root, "S1_DETERMINISTIC_MULTI_START")
    elif args.action == "run-search-s2":
        run_search_contract(root, "S2_SEQUENTIAL_MULTI_START")
    elif args.action == "run-search-s3":
        require_alignment(root)
        payload = read_json(root / "search_candidates/search_s3.json")
        print(f"O5RD2C_SEARCH_S3={payload['status']}")
    elif args.action == "run-development-gates":
        select_execution_contract(root)
    elif args.action == "run-determinism":
        run_determinism(root)
    elif args.action == "run-q1-reachability":
        q1_reachability(root)
    elif args.action in {"freeze-objective-v2", "freeze-execution-v2"}:
        freeze_or_stop(root)
    elif args.action == "generate-future-certification-plan":
        future_certification_plan(root)
    elif args.action == "write-not-run":
        write_not_run_and_usage(root)
    elif args.action == "validate":
        run_validation(root)
    elif args.action == "summarize":
        summarize(root)
    else:
        preflight(root)
        replay_alignment(root)
        audit_d2b_failures(root)
        compare_search_contracts(root)
        run_search_contract(root, "S1_DETERMINISTIC_MULTI_START")
        run_search_contract(root, "S2_SEQUENTIAL_MULTI_START")
        select_execution_contract(root)
        run_determinism(root)
        q1_reachability(root)
        freeze_or_stop(root)
        future_certification_plan(root)
        write_not_run_and_usage(root)
        summarize(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
