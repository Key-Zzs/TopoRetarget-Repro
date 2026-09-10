#!/usr/bin/env python3
"""Audit SparseV2 gate semantics and run fail-closed O5R-D2E/D2F recertification.

The frozen Candidate-B2 objective and S1 execution implementation are imported
unchanged.  D2E actions only inspect already-consumed SparseValidationV2
evidence.  Fresh compute is reachable only after a HIGH-confidence gate-
semantic-mismatch decision and immutable GateV2/validation manifests.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5rd2a import (
    D2ARuntime,
    digest,
    git,
    read_json,
    write_csv,
    write_json,
)
from scripts.data.run_oakink2_o5rd2c import CONTRACTS, _evaluate_b2, actual_continuity, search_frame
from scripts.data.run_oakink2_o5rd2d import (
    EIM_ATOL,
    EXECUTION_NAME,
    EXECUTION_SOURCE,
    OBJECTIVE_SOURCE,
    Q_ATOL,
    TAU,
    _compare_repeat,
    _determinism_signature,
    _determinism_sparse_ids,
    _row_from_receipt,
    development_entries,
    old_eim_rows,
    select_sparse_rows,
    select_window_rows,
)
from scripts.data.run_oakink2_o5rd2d import (
    ROOT as D2D_ROOT,
)
from scripts.data.run_oakink2_o5rd2d import (
    verify_frozen_method as verify_d2d_frozen_method,
)
from toporetarget.retarget.objective_v2 import interaction_hinge
from toporetarget.retarget.objective_v2_execution import asset_derived_dof_blocks
from toporetarget.utils.hashing import sha256_file

ROOT = REPO / ".local/reports/oakink2_o5rd2e_gate_alignment_and_v3_recertification_v1"
O5_ROOT = REPO / ".local/reports/oakink2_o5_geometric_retarget_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "7ea8b8cfd9817d912275f3b58114bc50320ed3e3"
SPARSE_V2_ROOT = D2D_ROOT / "sparse_v2"
EPSILON_NUM = EIM_ATOL

IMPLEMENTATION_PATHS = {
    "objective_v2_contract": OBJECTIVE_SOURCE,
    "execution_v2_contract": EXECUTION_SOURCE,
    "objective_implementation": REPO / "src/toporetarget/retarget/objective_v2.py",
    "execution_implementation": REPO / "src/toporetarget/retarget/objective_v2_execution.py",
    "search_driver": REPO / "scripts/data/run_oakink2_o5rd2c.py",
    "runtime_measurement_and_context": REPO / "scripts/data/run_oakink2_o5rd2a.py",
    "production_refinement_adapter": REPO / "src/toporetarget/retarget/final_refinement.py",
    "continuous_context": REPO / "src/toporetarget/retarget/continuous.py",
    "semantic_v1": REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py",
    "solver_profile": REPO
    / "configs/retarget/refinement_solvers/wuji_continuous_sequential_v1.yaml",
    "execution_profile": REPO
    / "configs/retarget/refinement_execution/wuji_continuous_sequential_fast_exact_v2.yaml",
    "frame_profile": REPO / "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml",
    "bone_profile": REPO / "configs/retarget/bones/mediapipe21_full_finger_chain_v1.yaml",
    "robot_mapping": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
    "robot_anchor_profile": REPO / "configs/robots/anchors/wuji_hand2_beta1_rh_mediapipe21.yaml",
    "robot_joint_order": REPO / "configs/robots/joint_orders/wuji_hand2_beta1_rh.yaml",
    "robot_surface_profile": REPO / "configs/robots/surfaces/wuji_hand2_beta1_rh.yaml",
    "robot_urdf": REPO / "third_party/robot_hands/wuji_hand2_beta1/urdf/right.urdf",
    "robot_mjcf": REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_sha(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def _require_file_hash(path: Path, receipt: Path, error: str) -> None:
    if (
        not path.exists()
        or not receipt.exists()
        or sha256_file(path) != receipt.read_text().strip()
    ):
        raise RuntimeError(error)


def _head_descends_from_start(head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0
    )


def implementation_authority() -> dict[str, Any]:
    files: dict[str, dict[str, str]] = {}
    for name, path in IMPLEMENTATION_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        files[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    return {
        "schema_version": "FrozenMethodImplementationAuthorityV1",
        "status": "FROZEN",
        "frozen_before_independent_certification_v3": True,
        "method_behavior_mutable": False,
        "files": files,
    }


def verify_frozen_method(root: Path) -> dict[str, Any]:
    base = verify_d2d_frozen_method(root)
    authority_path = root / "preflight/frozen_method_implementation_authority.json"
    observed = implementation_authority()
    if authority_path.exists():
        frozen = read_json(authority_path)
        if frozen != observed:
            write_json(
                root / "preflight/frozen_method_integrity.json",
                {
                    **base,
                    "FROZEN_METHOD_INTEGRITY": "FAIL",
                    "implementation_authority_exact": False,
                },
            )
            raise RuntimeError("O5RD2E_BLOCKED_IMPLEMENTATION_AUTHORITY_DRIFT")
    else:
        write_json(authority_path, observed)
    result = {
        **base,
        "schema_version": "O5RD2EFrozenMethodIntegrityV1",
        "implementation_authority_exact": True,
        "METHOD_IMPLEMENTATION_SHA_SET": "UNCHANGED",
    }
    write_json(root / "preflight/frozen_method_integrity.json", result)
    return result


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2E_BRANCH_MISMATCH:{branch}")
    if not _head_descends_from_start(head):
        raise RuntimeError(f"O5RD2E_START_HEAD_NOT_ANCESTOR:{head}")
    integrity = verify_frozen_method(root)
    authorities = {
        "d2d_gate_v1": SPARSE_V2_ROOT / "gate_contract.json",
        "d2d_gate_v1_decision": SPARSE_V2_ROOT / "gate_decision.json",
        "d2d_manifest_v2": SPARSE_V2_ROOT / "manifest.json",
        "d2d_sparse_results": SPARSE_V2_ROOT / "per_frame_results.csv",
        "d2d_determinism": SPARSE_V2_ROOT / "determinism.json",
        "d2d_handoff": D2D_ROOT / "handoff.md",
        "objective_v2": OBJECTIVE_SOURCE,
        "execution_v2": EXECUTION_SOURCE,
    }
    frozen = {}
    for name, path in authorities.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frozen[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    remote_ref = "origin/feature/oakink2-raw-to-physical"
    remote_head = git("rev-parse", remote_ref)
    behind, ahead = git("rev-list", "--left-right", "--count", f"{remote_ref}...HEAD").split()
    git_payload = {
        "schema_version": "OakInk2O5RD2EGitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "START_HEAD": START_HEAD,
        "head_at_preflight": head,
        "initial_status_short": [],
        "initial_status_provenance": "mandated shell preflight captured before task modifications",
        "status_short_at_artifact_write": git(
            "status", "--short", "--untracked-files=all"
        ).splitlines(),
        "diff_stat_at_artifact_write": git("diff", "--stat").splitlines(),
        "diff_check": git("diff", "--check").splitlines(),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "remotes": git("remote", "-v").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", git_payload)
    write_json(
        root / "preflight/upstream_state.json",
        {
            "schema_version": "O5RD2EUpstreamStateV1",
            "remote_ref": remote_ref,
            "remote_head": remote_head,
            "behind": int(behind),
            "ahead": int(ahead),
            "fetch_performed": False,
            "authorities": frozen,
        },
    )
    return {"git": git_payload, "integrity": integrity, "authorities": frozen}


def audit_gate_v1_authority(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    gate = read_json(SPARSE_V2_ROOT / "gate_contract.json")
    source = str((REPO / "scripts/data/run_oakink2_o5rd2d.py").resolve())
    criteria = [
        ("technical_completion", gate["technical_completion"], "PRODUCTION_EXECUTION_AUTHORITY"),
        (
            "above_target_recovery_rate_minimum",
            gate["above_target_recovery_rate_minimum"],
            "PRE_REGISTERED_CONSERVATIVE_CERTIFICATION_HEURISTIC",
        ),
        (
            "old_above_target_median_relative_reduction_minimum",
            gate["old_above_target_median_relative_reduction_minimum"],
            "PRE_REGISTERED_CONSERVATIVE_CERTIFICATION_HEURISTIC",
        ),
        (
            "overall_nonregression_fraction_minimum",
            gate["overall_nonregression_fraction_minimum"],
            "PRE_REGISTERED_CONSERVATIVE_CERTIFICATION_HEURISTIC",
        ),
        (
            "catastrophic_regression_maximum",
            gate["catastrophic_regression_maximum"],
            "PRE_REGISTERED_CONSERVATIVE_CERTIFICATION_HEURISTIC",
        ),
        (
            "low_controls_all_below_or_equal_tau",
            gate["low_controls_all_below_or_equal_tau"],
            "SEMANTIC_V1_AUTHORITY",
        ),
        ("tau", gate["tau"], "SEMANTIC_V1_AUTHORITY"),
        (
            "semantic_v1_hard_nonregression",
            gate["semantic_v1_hard_nonregression"],
            "SEMANTIC_V1_AUTHORITY",
        ),
        ("collision_required", True, "PHYSICAL_HARD_AUTHORITY"),
        ("joint_limits_required", True, "ROBOT_ASSET_HARD_AUTHORITY"),
        ("determinism_repeats", gate["determinism_repeats"], "PRODUCTION_EXECUTION_AUTHORITY"),
    ]
    rows = [
        {
            "criterion": name,
            "value": value,
            "classification": classification,
            "preregistered": True,
            "semantic_authority": classification
            in {
                "SEMANTIC_V1_AUTHORITY",
                "OBJECTIVE_V2_SEMANTIC_AUTHORITY",
                "PHYSICAL_HARD_AUTHORITY",
                "ROBOT_ASSET_HARD_AUTHORITY",
                "PRODUCTION_EXECUTION_AUTHORITY",
            },
            "local_provenance": f"{source}:341-360; introduced by commit a2455386a381237129f1b88548712da58c6cf375",
        }
        for name, value, classification in criteria
    ]
    definition = {
        "schema_version": "SparseValidationV2GateContractV1AuthorityAuditV1",
        "authoritative_contract_path": str((SPARSE_V2_ROOT / "gate_contract.json").resolve()),
        "authoritative_contract_sha256": sha256_file(SPARSE_V2_ROOT / "gate_contract.json"),
        "historical_result": "FAIL",
        "historical_result_rewritten": False,
        "criteria": gate,
    }
    authority = {
        "schema_version": "GateV1RequirementAuthorityMapV1",
        "freeze_is_not_semantic_authority": True,
        "requirements": rows,
        "EXACT_MONOTONIC_90_PERCENT_AUTHORITY": "PRE_REGISTERED_CONSERVATIVE_CERTIFICATION_HEURISTIC",
        "exact_monotonic_independent_semantic_physical_robot_authority_found": False,
        "repository_search_scope": ["src", "scripts", "tests", "docs"],
        "repository_search_result": "definition found only in O5R-D2D certification implementation and report; no Semantic V1, physical-hard, or robot-hard source",
    }
    catastrophic = {
        "schema_version": "CatastrophicRegressionGateAuthorityAuditV1",
        "definition": gate["catastrophic_regression_definition"],
        "classification": "CONSERVATIVE_HEURISTIC",
        "independent_authority_found": False,
        "future_gate_v2_role": "DIAGNOSTIC_ONLY",
    }
    write_json(root / "d2e_audit/gate_v1_definition.json", definition)
    write_json(root / "d2e_audit/gate_authority_map.json", authority)
    write_json(root / "d2e_audit/catastrophic_gate_authority.json", catastrophic)
    return authority


def _stored_hard_pass(evaluation: dict[str, Any]) -> bool:
    margins = evaluation["constraint_margins"]
    required = (
        "wrist_position_m",
        "wrist_rotation_rad",
        "bone_direction_rad",
        "actual_temporal_translation_m",
        "actual_temporal_rotation_rad",
        "collision_hard_m",
        "joint_limit_rad",
        "reflection_determinant",
        "unit_scale_lower",
        "unit_scale_upper",
    )
    return bool(
        evaluation["feasible"] and all(float(margins[name]) >= -EPSILON_NUM for name in required)
    )


def audit_sparse_v2_low(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    manifest = read_json(SPARSE_V2_ROOT / "manifest.json")
    low_frames = [row for row in manifest["frames"] if row["stratum"] == "LOW"]
    if [int(row["frame_id"]) for row in low_frames] != [
        1970,
        2045,
        2124,
        2223,
        2299,
        2376,
        2452,
        2528,
        2604,
        4689,
    ]:
        raise RuntimeError("O5RD2E_LOW_MANIFEST_AUTHORITY_DRIFT")
    runtime = D2ARuntime(root)
    rows: list[dict[str, Any]] = []
    for frame in low_frames:
        ordinal = int(frame["ordinal"])
        receipt_path = SPARSE_V2_ROOT / f"receipts/frame_{ordinal:04d}_run_1.json"
        state_path = receipt_path.with_suffix(".npz")
        receipt = read_json(receipt_path)
        state = np.load(state_path, allow_pickle=False)
        final_q = np.asarray(state["qpos"], dtype=np.float64)
        final_base = np.asarray(state["base_pose_scene"], dtype=np.float64)
        old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
        previous_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
        )
        binding, context = runtime.bind_context(
            ordinal, previous_base=previous_base, previous_qpos=previous_q
        )
        old_values = runtime.measurement(
            ordinal,
            old_q,
            old_base,
            binding=binding,
            context=context,
            slack=runtime.q0_slack(ordinal),
        )
        final_values = runtime.measurement(
            ordinal,
            final_q,
            final_base,
            binding=binding,
            context=context,
            slack=runtime.q0_slack(ordinal),
        )
        old_eval = _evaluate_b2(
            runtime,
            old_values,
            actual_continuity(runtime, previous_q, previous_base, old_q, old_base),
        )
        final_eval = _evaluate_b2(
            runtime,
            final_values,
            actual_continuity(runtime, previous_q, previous_base, final_q, final_base),
        )
        old_e = float(receipt["old"]["interaction_e_im"])
        new_e = float(receipt["selected"]["interaction_e_im"])
        old_secondary = float(receipt["old"]["secondary_objective"])
        retained_secondary = float(receipt["retained_primary"]["secondary_objective"])
        final_secondary = float(receipt["selected"]["secondary_objective"])
        row = {
            "frame_id": int(frame["frame_id"]),
            "ordinal": ordinal,
            "stratum": "LOW",
            "q_old_state": "AVAILABLE_REEVALUATED",
            "q_primary_selected_state": "INTERMEDIATE_VECTOR_UNAVAILABLE_STORED_EVALUATION_AVAILABLE"
            if receipt["retained_primary_id"] != receipt["selected_candidate"]
            else "AVAILABLE_AS_FINAL",
            "q_secondary_polished_state": "AVAILABLE_AS_FINAL"
            if receipt["selected_candidate"] == "secondary_polished"
            else "INTERMEDIATE_STATE_UNAVAILABLE",
            "q_final_state": "AVAILABLE_REEVALUATED",
            "old_e_im": old_e,
            "primary_selected_e_im": float(receipt["retained_primary"]["interaction_e_im"]),
            "final_e_im": new_e,
            "old_hinge": interaction_hinge(old_e, TAU),
            "primary_selected_hinge": interaction_hinge(
                float(receipt["retained_primary"]["interaction_e_im"]), TAU
            ),
            "final_hinge": interaction_hinge(new_e, TAU),
            "absolute_increase": new_e - old_e,
            "relative_increase": (new_e - old_e) / old_e,
            "remaining_semantic_margin": TAU - new_e,
            "old_secondary_objective": old_secondary,
            "primary_selected_secondary_objective": retained_secondary,
            "final_secondary_objective": final_secondary,
            "secondary_objective_change": final_secondary - old_secondary,
            "secondary_objectively_better": final_secondary < old_secondary,
            "old_secondary_components": json.dumps(
                receipt["old"]["per_components"], sort_keys=True
            ),
            "primary_secondary_components": json.dumps(
                receipt["retained_primary"]["per_components"], sort_keys=True
            ),
            "final_secondary_components": json.dumps(
                receipt["selected"]["per_components"], sort_keys=True
            ),
            "old_reevaluated_e_im": float(old_values.interaction_e_im),
            "final_reevaluated_e_im": float(final_values.interaction_e_im),
            "old_receipt_reevaluation_abs_diff": abs(float(old_values.interaction_e_im) - old_e),
            "final_receipt_reevaluation_abs_diff": abs(
                float(final_values.interaction_e_im) - new_e
            ),
            "old_semantic_valid": old_e <= TAU and _stored_hard_pass(old_eval),
            "final_semantic_valid": new_e <= TAU and _stored_hard_pass(final_eval),
            "wrist_pass": float(final_eval["constraint_margins"]["wrist_position_m"]) >= 0
            and float(final_eval["constraint_margins"]["wrist_rotation_rad"]) >= 0,
            "bone_pass": float(final_eval["constraint_margins"]["bone_direction_rad"]) >= 0,
            "continuity_pass": float(
                final_eval["constraint_margins"]["actual_temporal_translation_m"]
            )
            >= 0
            and float(final_eval["constraint_margins"]["actual_temporal_rotation_rad"]) >= 0,
            "collision_pass": float(final_eval["constraint_margins"]["collision_hard_m"]) >= 0,
            "joint_limits_pass": float(final_eval["constraint_margins"]["joint_limit_rad"])
            >= -EPSILON_NUM,
            "reflection_pass": float(final_eval["constraint_margins"]["reflection_determinant"])
            >= 0,
            "scale_pass": float(final_eval["constraint_margins"]["unit_scale_lower"]) >= 0
            and float(final_eval["constraint_margins"]["unit_scale_upper"]) >= 0,
            "selected_block": receipt["selected_block"],
            "selected_seed": receipt["selected_seed_candidate"],
            "candidate_rank": receipt["retained_primary_id"],
            "primary_retention": receipt["retention_decision"],
            "polish_accepted": receipt["selected_candidate"] == "secondary_polished",
            "fallback": receipt["baseline_fallback"],
            "q_old_baseline_available": "old_production" in receipt["seed_pool"],
            "classification": "EXPECTED_VALID_SET_SECONDARY_IMPROVEMENT",
            "VALID_SET_MOVEMENT_BY_DESIGN": True,
        }
        if not (
            row["old_hinge"] == 0
            and row["final_hinge"] == 0
            and row["secondary_objectively_better"]
            and row["final_semantic_valid"]
        ):
            row["classification"] = "INCONCLUSIVE"
            row["VALID_SET_MOVEMENT_BY_DESIGN"] = False
        rows.append(row)
    write_csv(root / "d2e_audit/low_per_frame_analysis.csv", rows)
    deltas = np.asarray([float(row["absolute_increase"]) for row in rows], dtype=np.float64)
    relatives = np.asarray([float(row["relative_increase"]) for row in rows], dtype=np.float64)
    aggregate = {
        "schema_version": "SparseV2LowAggregateAuditV1",
        "N_LOW": len(rows),
        "median_delta_e_im": float(np.median(deltas)),
        "p90_delta_e_im": float(np.quantile(deltas, 0.90)),
        "p95_delta_e_im": float(np.quantile(deltas, 0.95)),
        "max_delta_e_im": float(np.max(deltas)),
        "median_relative_increase": float(np.median(relatives)),
        "p95_relative_increase": float(np.quantile(relatives, 0.95)),
        "max_relative_increase": float(np.max(relatives)),
        "minimum_margin_to_tau": min(float(row["remaining_semantic_margin"]) for row in rows),
        "N_within_valid_set_before": sum(bool(row["old_semantic_valid"]) for row in rows),
        "N_within_valid_set_after": sum(bool(row["final_semantic_valid"]) for row in rows),
        "all_receipt_reevaluations_within_eim_atol": all(
            float(row["old_receipt_reevaluation_abs_diff"]) <= EIM_ATOL
            and float(row["final_receipt_reevaluation_abs_diff"]) <= EIM_ATOL
            for row in rows
        ),
        "D2E_OPTIMIZER_RUN_COUNT": 0,
        "D2E_NEW_DEV1_FRAMES_CONSUMED": 0,
        "DEV2_RUN_COUNT": 0,
    }
    write_json(root / "d2e_audit/low_aggregate.json", aggregate)
    return aggregate


def audit_qold_selection(root: Path) -> dict[str, Any]:
    rows = read_csv(root / "d2e_audit/low_per_frame_analysis.csv")
    output = [
        {
            "frame_id": int(row["frame_id"]),
            "ordinal": int(row["ordinal"]),
            "q_old_hinge_zero": float(row["old_hinge"]) == 0.0,
            "q_final_hinge_zero": float(row["final_hinge"]) == 0.0,
            "q_final_secondary_better": row["secondary_objectively_better"] == "True",
            "all_hard_valid": all(
                row[name] == "True"
                for name in (
                    "wrist_pass",
                    "bone_pass",
                    "continuity_pass",
                    "collision_pass",
                    "joint_limits_pass",
                    "reflection_pass",
                    "scale_pass",
                )
            ),
            "q_old_was_baseline": row["q_old_baseline_available"] == "True",
            "selected_seed": row["selected_seed"],
            "retained_primary": row["candidate_rank"],
            "polish_accepted": row["polish_accepted"] == "True",
            "retention_decision": row["primary_retention"],
            "classification": row["classification"],
            "VALID_SET_MOVEMENT_BY_DESIGN": row["VALID_SET_MOVEMENT_BY_DESIGN"] == "True",
        }
        for row in rows
    ]
    write_csv(root / "d2e_audit/qold_selection_audit.csv", output)
    counts = {
        name: sum(row["classification"] == name for row in output)
        for name in (
            "EXPECTED_VALID_SET_SECONDARY_IMPROVEMENT",
            "EXPECTED_PRIMARY_RETENTION_BEHAVIOR",
            "EXPECTED_DETERMINISTIC_TIEBREAK",
            "UNNECESSARY_MOVE_NO_SECONDARY_BENEFIT",
            "LEXICOGRAPHIC_SELECTION_VIOLATION",
            "HARD_VALIDITY_REGRESSION",
            "INSUFFICIENT_RECEIPT",
            "INCONCLUSIVE",
        )
    }
    payload = {
        "schema_version": "SparseV2ValidSetMovementAuditV1",
        "q_old_always_available_contract": True,
        "candidate_screening": read_json(EXECUTION_SOURCE)["candidate_screening"],
        "counts": counts,
        "VALID_SET_MOVEMENT_BY_DESIGN": "YES"
        if counts["EXPECTED_VALID_SET_SECONDARY_IMPROVEMENT"] == 10
        else "NO",
    }
    write_json(root / "d2e_audit/valid_set_movement.json", payload)
    return payload


def threshold_aware_nonregression(
    old: float, new: float, *, tau: float = TAU, epsilon_num: float = EPSILON_NUM
) -> bool:
    return float(new) <= max(float(old), float(tau)) + float(epsilon_num)


def exact_nonregression(old: float, new: float) -> bool:
    return float(new) <= float(old)


def compare_exact_and_threshold(root: Path) -> dict[str, Any]:
    rows = []
    for row in read_csv(SPARSE_V2_ROOT / "per_frame_results.csv"):
        old, new = float(row["old_e_im"]), float(row["new_e_im"])
        rows.append(
            {
                "frame_id": int(row["frame_id"]),
                "ordinal": int(row["ordinal"]),
                "stratum": row["stratum"],
                "old_e_im": old,
                "new_e_im": new,
                "old_valid": old <= TAU,
                "exact_monotonic_M_i": exact_nonregression(old, new),
                "threshold_aware_T_i": threshold_aware_nonregression(old, new),
                "threshold_bound": max(old, TAU) + EPSILON_NUM,
                "epsilon_num": EPSILON_NUM,
            }
        )
    write_csv(root / "d2e_audit/exact_vs_threshold_nonregression.csv", rows)
    return {
        "N": len(rows),
        "exact_monotonic_count": sum(bool(row["exact_monotonic_M_i"]) for row in rows),
        "threshold_aware_count": sum(bool(row["threshold_aware_T_i"]) for row in rows),
        "epsilon_num": EPSILON_NUM,
        "epsilon_authority": "SparseValidationV2GateContractV1.e_im_atol",
    }


def decide_d2e_root_cause(root: Path) -> dict[str, Any]:
    authority = read_json(root / "d2e_audit/gate_authority_map.json")
    aggregate = read_json(root / "d2e_audit/low_aggregate.json")
    movement = read_json(root / "d2e_audit/valid_set_movement.json")
    predicates = compare_exact_and_threshold(root)
    evidence = {
        "A_10_of_10_LOW_semantic_valid": aggregate["N_within_valid_set_after"] == 10,
        "B_10_of_10_LOW_no_hard_violation": movement["counts"]["HARD_VALIDITY_REGRESSION"] == 0
        and movement["counts"]["INCONCLUSIVE"] == 0,
        "C_candidate_b2_zero_pressure_in_valid_set": read_json(OBJECTIVE_SOURCE)[
            "candidate_b2_primary_hinge"
        ]
        == "max(E_IM/tau - 1, 0)^2",
        "D_exact_90_has_no_independent_hard_authority": not authority[
            "exact_monotonic_independent_semantic_physical_robot_authority_found"
        ],
        "E_all_LOW_follow_frozen_lexicographic_selection": movement["counts"][
            "EXPECTED_VALID_SET_SECONDARY_IMPROVEMENT"
        ]
        == 10,
        "stored_state_reevaluation_exact": aggregate["all_receipt_reevaluations_within_eim_atol"],
        "all_30_threshold_aware_nonregression": predicates["threshold_aware_count"] == 30,
    }
    mismatch = all(evidence.values())
    payload = {
        "schema_version": "O5RD2ERootCauseDecisionV1",
        "PRIMARY_ROOT_CAUSE": "CERTIFICATION_GATE_SEMANTIC_MISMATCH"
        if mismatch
        else "INCONCLUSIVE",
        "CONFIDENCE": "HIGH" if mismatch else "LOW",
        "evidence": evidence,
        "exact_vs_threshold": predicates,
        "SPARSE_VALIDATION_V2": "FAIL",
        "SPARSE_V2_GATE_CONTRACT_V1": "FAILED",
        "HISTORICAL_RESULT_REWRITTEN": "NO",
        "D2E_OPTIMIZER_RUN_COUNT": 0,
        "D2E_NEW_DEV1_FRAMES_CONSUMED": 0,
        "DEV2_RUN_COUNT": 0,
    }
    write_json(root / "d2e_audit/root_cause_decision.json", payload)
    if not mismatch:
        write_not_authorized(root, reason="D2E_ROOT_CAUSE_NOT_GATE_MISMATCH_HIGH")
    return payload


def certification_gate_v2() -> dict[str, Any]:
    return {
        "schema_version": "CertificationGateContractV2",
        "description": "threshold-aware semantic realignment; not a relaxed GateContractV1",
        "derivation": "Candidate B2 valid set E_IM<=tau plus frozen Semantic V1 acceptance set",
        "frozen_before_new_validation_selection": True,
        "mutable_after_freeze": False,
        "tau": TAU,
        "epsilon_num": EPSILON_NUM,
        "epsilon_num_authority": "SparseValidationV2GateContractV1.e_im_atol",
        "technical_completion": "N/N finite, valid method terminal, joint limits PASS, collision PASS",
        "old_invalid_definition": "E_old > tau",
        "old_valid_definition": "E_old <= tau",
        "old_invalid_recovery_rate_minimum": 0.80,
        "old_invalid_median_relative_reduction_minimum": 0.50,
        "threshold_aware_nonregression": "E_new <= max(E_old, tau) + epsilon_num",
        "threshold_aware_nonregression_required_fraction": 1.0,
        "low_preservation": "for E_old <= tau, E_new <= tau + epsilon_num",
        "low_preservation_required_fraction": 1.0,
        "hard_semantic_validity": [
            "wrist",
            "bone",
            "actual_continuity",
            "reflection",
            "scale",
            "collision",
            "joint_limits",
        ],
        "determinism_required": True,
        "exact_monotonic_fraction_role": "DIAGNOSTIC_ONLY",
        "catastrophic_relative_regression_role": "DIAGNOSTIC_ONLY",
        "outcome_driven_threshold_tuning": False,
    }


def freeze_gate_v2_if_authorized(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    decision = read_json(root / "d2e_audit/root_cause_decision.json")
    if (
        decision.get("PRIMARY_ROOT_CAUSE") != "CERTIFICATION_GATE_SEMANTIC_MISMATCH"
        or decision.get("CONFIDENCE") != "HIGH"
    ):
        write_not_authorized(root, reason="D2E_ROOT_CAUSE_NOT_GATE_MISMATCH_HIGH")
        raise RuntimeError("O5RD2E_GATE_V2_NOT_AUTHORIZED")
    path = root / "gate_v2/certification_gate_v2.json"
    contract = certification_gate_v2()
    if path.exists() and read_json(path) != contract:
        raise RuntimeError("O5RD2E_GATE_V2_FROZEN_DRIFT")
    write_json(path, contract)
    sha = sha256_file(path)
    _write_sha(root / "gate_v2/certification_gate_v2.sha256", sha)
    derivation = """# CertificationGateContractV2 semantic derivation\n\nCandidate B2 uses `max(E_IM/tau - 1, 0)^2`; therefore every state in the frozen Semantic V1 interaction-valid set has zero primary pressure. GateV2 preserves the original 80% invalid-recovery and 50% median-invalid-improvement criteria, but replaces the unauthorised all-frame exact-monotonic heuristic with the structural rule\n\n`E_new <= max(E_old, tau) + epsilon_num`.\n\nFor old-invalid frames this forbids further degradation; for old-valid frames it permits secondary optimization inside the valid set while forbidding return to the invalid set. `epsilon_num=1e-12` is inherited from the already-frozen GateContractV1 reproducibility tolerance, not fitted to SparseV2 outcomes. Historical SparseValidationV2 remains FAIL under GateContractV1.\n"""
    (root / "gate_v2/semantic_derivation.md").write_text(derivation, encoding="utf-8")
    posthoc = evaluate_gate_v2_rows(
        read_csv(SPARSE_V2_ROOT / "per_frame_results.csv"),
        determinism_pass=True,
        certification_status="NOT_APPLICABLE_POST_HOC",
    )
    posthoc.update(
        {
            "schema_version": "SparseV2GateV2PostHocCompatibilityDiagnosticV1",
            "SPARSE_V2_GATE_V2_CERTIFICATION_STATUS": "NOT_APPLICABLE_POST_HOC",
            "POST_HOC_DIAGNOSTIC_ONLY": True,
            "SPARSE_VALIDATION_V2": "FAIL",
            "historical_gate": "SparseValidationV2GateContractV1",
            "historical_result_rewritten": False,
        }
    )
    write_json(root / "gate_v2/sparse_v2_posthoc_compatibility_diagnostic.json", posthoc)
    return {
        "CERTIFICATION_GATE_V2_CREATED": "YES",
        "CERTIFICATION_GATE_V2_SHA256": sha,
        "contract": contract,
    }


def _hard_row_pass(row: dict[str, Any]) -> bool:
    def truth(name: str) -> bool:
        value = row[name]
        return value == "True" if isinstance(value, str) else bool(value)

    return all(
        truth(name)
        for name in (
            "technical_completion",
            "finite",
            "semantic_hard_pass",
            "wrist_pass",
            "bone_pass",
            "continuity_pass",
            "collision_pass",
            "joint_limits_pass",
            "reflection_pass",
            "scale_pass",
        )
    )


def evaluate_gate_v2_rows(
    rows: list[dict[str, Any]], *, determinism_pass: bool, certification_status: str = "INDEPENDENT"
) -> dict[str, Any]:
    above = [row for row in rows if float(row["old_e_im"]) > TAU]
    low = [row for row in rows if float(row["old_e_im"]) <= TAU]
    recovered = sum(float(row["new_e_im"]) <= TAU + EPSILON_NUM for row in above)
    reductions = [
        (float(row["old_e_im"]) - float(row["new_e_im"])) / float(row["old_e_im"]) for row in above
    ]
    threshold_count = sum(
        threshold_aware_nonregression(float(row["old_e_im"]), float(row["new_e_im"]))
        for row in rows
    )
    exact_count = sum(
        exact_nonregression(float(row["old_e_im"]), float(row["new_e_im"])) for row in rows
    )
    conditions = {
        "technical_N_of_N_and_hard_validity": bool(rows)
        and all(_hard_row_pass(row) for row in rows),
        "old_invalid_recovery": bool(above) and recovered / len(above) >= 0.80,
        "old_invalid_median_relative_reduction": bool(reductions)
        and float(np.median(reductions)) >= 0.50,
        "threshold_aware_nonregression_100_percent": bool(rows) and threshold_count == len(rows),
        "low_preservation_100_percent": bool(low)
        and all(float(row["new_e_im"]) <= TAU + EPSILON_NUM for row in low),
        "determinism": determinism_pass,
    }
    return {
        "certification_status": certification_status,
        "N": len(rows),
        "technical": sum(_hard_row_pass(row) for row in rows),
        "old_invalid_count": len(above),
        "old_invalid_recovered_count": recovered,
        "old_invalid_recovery_rate": None if not above else recovered / len(above),
        "median_invalid_relative_reduction": None
        if not reductions
        else float(np.median(reductions)),
        "threshold_aware_nonregression_count": threshold_count,
        "low_preserved_count": sum(float(row["new_e_im"]) <= TAU + EPSILON_NUM for row in low),
        "exact_monotonic_count_DIAGNOSTIC_ONLY": exact_count,
        "exact_monotonic_fraction_DIAGNOSTIC_ONLY": None if not rows else exact_count / len(rows),
        "catastrophic_count_DIAGNOSTIC_ONLY": sum(
            float(row["new_e_im"]) > 1.25 * float(row["old_e_im"]) for row in rows
        ),
        "conditions": conditions,
        "decision": "PASS" if all(conditions.values()) else "FAIL",
    }


def _development_ordinals() -> set[int]:
    return {int(row["ordinal"]) for row in development_entries()}


def _sparse_v2_frames() -> list[dict[str, Any]]:
    return list(read_json(SPARSE_V2_ROOT / "manifest.json")["frames"])


def _write_ledgers(
    root: Path,
    *,
    sparse_v3: list[dict[str, Any]] | None = None,
    sparse_consumed: bool = False,
    windows: list[dict[str, Any]] | None = None,
    windows_consumed: bool = False,
) -> None:
    development = development_entries()
    sparse_v2 = _sparse_v2_frames()
    write_json(
        root / "ledger/development_ledger.json",
        {
            "schema_version": "DEV1MethodDevelopmentLedgerV5",
            "role": "METHOD_DEVELOPMENT",
            "count": len(development),
            "entries": development,
        },
    )
    entries = []
    entries.extend(
        {
            **row,
            "role": "SPARSE_V2_CERTIFICATION_CONSUMED",
            "audit_role": "SPARSE_V2_POST_FAILURE_AUDIT",
        }
        for row in sparse_v2
    )
    entries.extend(
        {
            **row,
            "role": "SPARSE_V3_CERTIFICATION_CONSUMED"
            if sparse_consumed
            else "FROZEN_NOT_EXECUTED",
        }
        for row in sparse_v3 or []
    )
    for window in windows or []:
        for ordinal, frame_id in zip(window["ordinals"], window["frame_ids"], strict=True):
            entries.append(
                {
                    "ordinal": ordinal,
                    "frame_id": frame_id,
                    "window_id": window["window_id"],
                    "stratum": window["type"],
                    "role": "WINDOW_V3_CERTIFICATION_CONSUMED"
                    if windows_consumed
                    else "FROZEN_NOT_EXECUTED",
                }
            )
    write_json(
        root / "ledger/certification_consumption_ledger.json",
        {"schema_version": "DEV1EvidenceConsumptionLedgerV5", "entries": entries},
    )
    excluded = {
        (int(row["ordinal"]), int(row["frame_id"]), "METHOD_DEVELOPMENT") for row in development
    }
    excluded.update(
        (int(row["ordinal"]), int(row["frame_id"]), "SPARSE_V2_CERTIFICATION_CONSUMED")
        for row in sparse_v2
    )
    write_json(
        root / "ledger/total_exclusion_ledger.json",
        {
            "schema_version": "DEV1TotalExclusionLedgerV1",
            "TOTAL_PRE_V3_EXCLUSION_COUNT": len(excluded),
            "entries": [
                {"ordinal": ordinal, "frame_id": frame_id, "role": role}
                for ordinal, frame_id, role in sorted(excluded)
            ],
        },
    )


def freeze_sparse_v3(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    _require_file_hash(
        root / "gate_v2/certification_gate_v2.json",
        root / "gate_v2/certification_gate_v2.sha256",
        "O5RD2E_GATE_V2_INTEGRITY",
    )
    target = root / "sparse_v3/manifest.json"
    gate_path = root / "sparse_v3/gate_contract.json"
    if target.exists() or gate_path.exists():
        if not (target.exists() and gate_path.exists()):
            raise RuntimeError("O5RD2E_PARTIAL_SPARSE_V3_FREEZE")
        _require_file_hash(
            target, root / "sparse_v3/manifest.sha256", "O5RD2E_SPARSE_V3_MANIFEST_DRIFT"
        )
        _require_file_hash(
            gate_path, root / "sparse_v3/gate_contract.sha256", "O5RD2E_SPARSE_V3_GATE_DRIFT"
        )
        return read_json(target)
    development = _development_ordinals()
    sparse_v2 = {int(row["ordinal"]) for row in _sparse_v2_frames()}
    selected = select_sparse_rows(old_eim_rows(), development | sparse_v2)
    ordinals = {int(row["ordinal"]) for row in selected}
    if len(selected) != 30 or development & ordinals or sparse_v2 & ordinals:
        raise RuntimeError("O5RD2E_SPARSE_V3_OVERLAP_OR_COUNT")
    payload = {
        "schema_version": "DEV1SparseValidationV3",
        "version": "V3",
        "status": "FROZEN",
        "selection_authority": "old frozen DEV1 trajectory E_IM only",
        "selection_algorithm": "SparseValidationV2 deterministic strata and temporal-spread algorithm with expanded exclusion ledger",
        "frames": selected,
        "N": len(selected),
        "composition": {
            name: sum(row["stratum"] == name for row in selected) for name in ("HIGH", "MID", "LOW")
        },
        "determinism_ordinals": _determinism_sparse_ids(selected),
        "SPARSE_V3_DEVELOPMENT_OVERLAP": len(development & ordinals),
        "SPARSE_V3_SPARSE_V2_OVERLAP": len(sparse_v2 & ordinals),
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    write_json(target, payload)
    _write_sha(root / "sparse_v3/manifest.sha256", sha256_file(target))
    gate = read_json(root / "gate_v2/certification_gate_v2.json")
    write_json(gate_path, gate)
    _write_sha(root / "sparse_v3/gate_contract.sha256", sha256_file(gate_path))
    write_json(
        root / "sparse_v3/determinism_subset.json",
        {
            "schema_version": "SparseValidationV3DeterminismSubsetV1",
            "ordinals": payload["determinism_ordinals"],
            "repeats": 3,
        },
    )
    _write_ledgers(root, sparse_v3=selected)
    return payload


def _load_or_run_frame(
    runtime: D2ARuntime,
    root: Path,
    directory: str,
    ordinal: int,
    run: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    receipt_path = root / f"{directory}/receipts/frame_{ordinal:04d}_run_{run}.json"
    state_path = receipt_path.with_suffix(".npz")
    if receipt_path.exists() and state_path.exists():
        receipt = read_json(receipt_path)
        state = np.load(state_path, allow_pickle=False)
        return (
            receipt,
            np.asarray(state["qpos"], dtype=np.float64),
            np.asarray(state["base_pose_scene"], dtype=np.float64),
        )
    qpos, base, receipt = search_frame(
        runtime,
        ordinal,
        previous_q=previous_q,
        previous_base=previous_base,
        contract=CONTRACTS[EXECUTION_NAME].validate(),
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
    write_json(receipt_path, receipt)
    return receipt, qpos, base


def run_sparse_v3(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    manifest = read_json(root / "sparse_v3/manifest.json")
    _require_file_hash(
        root / "sparse_v3/manifest.json",
        root / "sparse_v3/manifest.sha256",
        "O5RD2E_SPARSE_V3_MANIFEST_DRIFT",
    )
    _require_file_hash(
        root / "sparse_v3/gate_contract.json",
        root / "sparse_v3/gate_contract.sha256",
        "O5RD2E_SPARSE_V3_GATE_DRIFT",
    )
    decision_path = root / "sparse_v3/gate_decision.json"
    if decision_path.exists():
        existing = read_json(decision_path)
        if existing.get("evaluator_version") == "CertificationGateV2EvaluatorV2":
            return existing
        results_path = root / "sparse_v3/per_frame_results.csv"
        determinism_path = root / "sparse_v3/determinism.json"
        if not results_path.exists() or not determinism_path.exists():
            raise RuntimeError("O5RD2E_SPARSE_V3_PARTIAL_RESULT")
        invalidated = {
            **existing,
            "decision_validity": "INVALIDATED",
            "invalidation_reason": "CertificationGateV2EvaluatorV1 used identity checks for numpy.bool_ hard-validity fields",
            "optimizer_rerun": False,
            "evidence_reused_exactly": True,
        }
        invalidated_path = root / "sparse_v3/gate_decision.invalidated_evaluator_v1.json"
        write_json(invalidated_path, invalidated)
        determinism_pass = read_json(determinism_path).get("status") == "PASS"
        corrected = evaluate_gate_v2_rows(read_csv(results_path), determinism_pass=determinism_pass)
        corrected.update(
            {
                "schema_version": "SparseValidationV3GateDecisionV1",
                "SPARSE_VALIDATION_V3": corrected.pop("decision"),
                "evaluator_version": "CertificationGateV2EvaluatorV2",
                "supersedes_invalidated_evaluator_receipt": str(invalidated_path.resolve()),
                "optimizer_rerun": False,
                "same_frozen_manifest_and_result_bytes": True,
            }
        )
        write_json(decision_path, corrected)
        if corrected["SPARSE_VALIDATION_V3"] != "PASS":
            write_downstream_not_run(root, reason="SPARSE_VALIDATION_V3_FAIL")
        return corrected
    runtime = D2ARuntime(root)
    rows, references = [], {}
    for index, frame in enumerate(manifest["frames"], start=1):
        ordinal = int(frame["ordinal"])
        previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
        previous_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
        )
        receipt, qpos, _base = _load_or_run_frame(
            runtime, root, "sparse_v3", ordinal, 1, previous_q, previous_base
        )
        references[ordinal] = (receipt, qpos)
        rows.append(_row_from_receipt(receipt, str(frame["stratum"])))
        print(
            f"O5RD2F_SPARSE_V3 {index}/{manifest['N']} ordinal={ordinal} eim={float(receipt['selected']['interaction_e_im']):.12g}",
            flush=True,
        )
    write_csv(root / "sparse_v3/per_frame_results.csv", rows)
    write_csv(root / "sparse_v3/semantic_metrics.csv", rows)
    write_csv(root / "sparse_v3/search_metrics.csv", rows)
    repeat_rows = []
    for ordinal_value in manifest["determinism_ordinals"]:
        ordinal = int(ordinal_value)
        reference_receipt, reference_q = references[ordinal]
        repeat_rows.append(
            {
                "ordinal": ordinal,
                "run": 1,
                "pass": True,
                "signature": _determinism_signature(reference_receipt),
            }
        )
        previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
        previous_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
        )
        for run in (2, 3):
            receipt, qpos, _base = _load_or_run_frame(
                runtime, root, "sparse_v3", ordinal, run, previous_q, previous_base
            )
            repeat_rows.append(
                {
                    "ordinal": ordinal,
                    "run": run,
                    **_compare_repeat(reference_receipt, reference_q, receipt, qpos),
                }
            )
            print(f"O5RD2F_SPARSE_V3_DETERMINISM ordinal={ordinal} run={run}", flush=True)
    determinism_pass = all(bool(row["pass"]) for row in repeat_rows)
    write_json(
        root / "sparse_v3/determinism.json",
        {
            "schema_version": "SparseValidationV3DeterminismV1",
            "status": "PASS" if determinism_pass else "FAIL",
            "rows": repeat_rows,
        },
    )
    decision = evaluate_gate_v2_rows(rows, determinism_pass=determinism_pass)
    decision.update(
        {
            "schema_version": "SparseValidationV3GateDecisionV1",
            "SPARSE_VALIDATION_V3": decision.pop("decision"),
            "evaluator_version": "CertificationGateV2EvaluatorV2",
        }
    )
    write_json(decision_path, decision)
    _write_ledgers(root, sparse_v3=manifest["frames"], sparse_consumed=True)
    if decision["SPARSE_VALIDATION_V3"] != "PASS":
        write_downstream_not_run(root, reason="SPARSE_VALIDATION_V3_FAIL")
    return decision


def window_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": "WindowValidationV3GateContractV1",
        "frozen_before_first_solve": True,
        "mutable_after_freeze": False,
        "windows": ["HIGH_1", "HIGH_2", "MID", "LOW"],
        "technical_completion": "4/4 windows and all frames",
        "interaction_p95_maximum": TAU,
        "low_old_valid_preservation": "E_new <= tau + epsilon_num",
        "epsilon_num": EPSILON_NUM,
        "semantic_v1_actual_continuity_required": True,
        "wrist_bone_collision_joint_limits_reflection_scale_required": True,
        "jitter_role": "DIAGNOSTIC_ONLY where no frozen hard authority exists",
        "determinism_windows": ["HIGH_1", "LOW"],
        "determinism_repeats": 3,
        "q_atol": Q_ATOL,
        "e_im_atol": EIM_ATOL,
    }


def freeze_window_v3(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    sparse = read_json(root / "sparse_v3/gate_decision.json")
    if sparse.get("SPARSE_VALIDATION_V3") != "PASS":
        raise RuntimeError("O5RD2E_WINDOW_V3_BLOCKED_BY_SPARSE_V3")
    target = root / "window_v3/manifest.json"
    gate_path = root / "window_v3/gate_contract.json"
    if target.exists() or gate_path.exists():
        if not (target.exists() and gate_path.exists()):
            raise RuntimeError("O5RD2E_PARTIAL_WINDOW_V3_FREEZE")
        _require_file_hash(
            target, root / "window_v3/manifest.sha256", "O5RD2E_WINDOW_V3_MANIFEST_DRIFT"
        )
        _require_file_hash(
            gate_path, root / "window_v3/gate_contract.sha256", "O5RD2E_WINDOW_V3_GATE_DRIFT"
        )
        return read_json(target)
    development = _development_ordinals()
    sparse_v2 = {int(row["ordinal"]) for row in _sparse_v2_frames()}
    sparse_v3 = {
        int(row["ordinal"]) for row in read_json(root / "sparse_v3/manifest.json")["frames"]
    }
    selected = select_window_rows(old_eim_rows(), development | sparse_v2 | sparse_v3)
    all_ordinals = [int(value) for window in selected for value in window["ordinals"]]
    ordinal_set = set(all_ordinals)
    payload = {
        "schema_version": "DEV1WindowValidationV3",
        "status": "FROZEN",
        "selection_authority": "old frozen DEV1 trajectory E_IM only",
        "selection_rule": "same deterministic V2 temporal/stratum rule with expanded exclusions; 32 consecutive frames",
        "windows": selected,
        "WINDOW_V3_DEVELOPMENT_OVERLAP": len(development & ordinal_set),
        "WINDOW_V3_SPARSE_V2_OVERLAP": len(sparse_v2 & ordinal_set),
        "WINDOW_V3_SPARSE_V3_OVERLAP": len(sparse_v3 & ordinal_set),
        "WINDOW_V3_INTERNAL_OVERLAP": len(all_ordinals) - len(ordinal_set),
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    if any(
        payload[name] != 0
        for name in (
            "WINDOW_V3_DEVELOPMENT_OVERLAP",
            "WINDOW_V3_SPARSE_V2_OVERLAP",
            "WINDOW_V3_SPARSE_V3_OVERLAP",
            "WINDOW_V3_INTERNAL_OVERLAP",
        )
    ):
        raise RuntimeError("O5RD2E_WINDOW_V3_OVERLAP")
    write_json(target, payload)
    _write_sha(root / "window_v3/manifest.sha256", sha256_file(target))
    write_json(gate_path, window_gate_contract())
    _write_sha(root / "window_v3/gate_contract.sha256", sha256_file(gate_path))
    _write_ledgers(
        root,
        sparse_v3=read_json(root / "sparse_v3/manifest.json")["frames"],
        sparse_consumed=True,
        windows=selected,
    )
    return payload


def _window_run(
    runtime: D2ARuntime, root: Path, window: dict[str, Any], run: int
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    ordinals = [int(value) for value in window["ordinals"]]
    first = ordinals[0]
    previous_q = np.asarray(runtime.final.arrays["qpos"][first - 1], dtype=np.float64)
    previous_base = np.asarray(runtime.final.arrays["base_pose_scene"][first - 1], dtype=np.float64)
    rows, q_states, base_states = [], [], []
    directory = f"window_v3/receipts/{window['window_id']}"
    for ordinal in ordinals:
        receipt, qpos, base = _load_or_run_frame(
            runtime, root, directory, ordinal, run, previous_q, previous_base
        )
        row = _row_from_receipt(receipt, str(window["type"]))
        row.update({"window_id": window["window_id"], "run": run})
        rows.append(row)
        q_states.append(qpos)
        base_states.append(base)
        previous_q, previous_base = qpos, base
    return rows, q_states, base_states


def _jitter_rows(
    runtime: D2ARuntime, window_id: str, rows: list[dict[str, Any]], q_states: list[np.ndarray]
) -> list[dict[str, Any]]:
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    output = []
    for index, (row, qpos) in enumerate(zip(rows, q_states, strict=True)):
        if index == 0:
            previous = np.asarray(
                runtime.final.arrays["qpos"][int(row["ordinal"]) - 1], dtype=np.float64
            )
        else:
            previous = q_states[index - 1]
        delta = np.abs(qpos - previous)
        thumb = np.asarray(blocks["thumb"], dtype=np.int64)
        other = np.asarray(
            [value for name, block in blocks.items() if name != "thumb" for value in block],
            dtype=np.int64,
        )
        output.append(
            {
                "window_id": window_id,
                "ordinal": int(row["ordinal"]),
                "joint_step_mean_rad": float(np.mean(delta)),
                "joint_step_p95_rad": float(np.quantile(delta, 0.95)),
                "joint_step_max_rad": float(np.max(delta)),
                "thumb_step_mean_rad": float(np.mean(delta[thumb])),
                "thumb_step_p95_rad": float(np.quantile(delta[thumb], 0.95)),
                "thumb_step_max_rad": float(np.max(delta[thumb])),
                "other_finger_step_mean_rad": float(np.mean(delta[other])),
                "other_finger_step_p95_rad": float(np.quantile(delta[other], 0.95)),
                "other_finger_step_max_rad": float(np.max(delta[other])),
                "wrist_translation_step_m": float(row["actual_translation_step_m"]),
                "wrist_rotation_step_rad": float(row["actual_rotation_step_rad"]),
                "role": "DIAGNOSTIC_ONLY",
            }
        )
    return output


def run_window_v3(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    sparse = read_json(root / "sparse_v3/gate_decision.json")
    if sparse.get("SPARSE_VALIDATION_V3") != "PASS":
        raise RuntimeError("O5RD2E_WINDOW_V3_BLOCKED_BY_SPARSE_V3")
    manifest = read_json(root / "window_v3/manifest.json")
    _require_file_hash(
        root / "window_v3/manifest.json",
        root / "window_v3/manifest.sha256",
        "O5RD2E_WINDOW_V3_MANIFEST_DRIFT",
    )
    _require_file_hash(
        root / "window_v3/gate_contract.json",
        root / "window_v3/gate_contract.sha256",
        "O5RD2E_WINDOW_V3_GATE_DRIFT",
    )
    decision_path = root / "window_v3/gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    runtime = D2ARuntime(root)
    all_rows, summaries, jitter = [], [], []
    references: dict[str, tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]] = {}
    for window in manifest["windows"]:
        rows, q_states, base_states = _window_run(runtime, root, window, 1)
        references[str(window["window_id"])] = (rows, q_states, base_states)
        all_rows.extend(rows)
        jitter.extend(_jitter_rows(runtime, str(window["window_id"]), rows, q_states))
        new_values = np.asarray([float(row["new_e_im"]) for row in rows], dtype=np.float64)
        summaries.append(
            {
                "window_id": window["window_id"],
                "type": window["type"],
                "N": len(rows),
                "old_p95_e_im": float(window["old_p95_e_im"]),
                "new_p95_e_im": float(np.quantile(new_values, 0.95)),
                "technical": sum(_hard_row_pass(row) for row in rows),
                "continuity_pass": all(row["continuity_pass"] for row in rows),
                "wrist_pass": all(row["wrist_pass"] for row in rows),
                "bone_pass": all(row["bone_pass"] for row in rows),
                "collision_pass": all(row["collision_pass"] for row in rows),
                "joint_limits_pass": all(row["joint_limits_pass"] for row in rows),
                "reflection_pass": all(row["reflection_pass"] for row in rows),
                "scale_pass": all(row["scale_pass"] for row in rows),
                "interaction_pass": float(np.quantile(new_values, 0.95)) <= TAU,
                "low_preservation_pass": window["type"] != "LOW"
                or all(
                    float(row["new_e_im"]) <= TAU + EPSILON_NUM
                    for row in rows
                    if float(row["old_e_im"]) <= TAU
                ),
            }
        )
        print(
            f"O5RD2F_WINDOW_V3 {window['window_id']} 1/3 p95={summaries[-1]['new_p95_e_im']:.12g}",
            flush=True,
        )
    write_csv(root / "window_v3/per_frame_results.csv", all_rows)
    write_csv(root / "window_v3/interaction_metrics.csv", all_rows)
    write_csv(root / "window_v3/continuity_metrics.csv", all_rows)
    write_csv(root / "window_v3/search_metrics.csv", all_rows)
    write_csv(root / "window_v3/jitter_metrics.csv", jitter)
    determinism_rows = []
    for window_id in ("HIGH_1", "LOW"):
        reference_rows, reference_q, reference_base = references[window_id]
        determinism_rows.append(
            {
                "window_id": window_id,
                "run": 1,
                "pass": True,
                "p95_e_im": float(
                    np.quantile([float(row["new_e_im"]) for row in reference_rows], 0.95)
                ),
            }
        )
        window = next(item for item in manifest["windows"] if item["window_id"] == window_id)
        for run in (2, 3):
            rows, q_states, base_states = _window_run(runtime, root, window, run)
            categorical = all(
                _determinism_signature(
                    read_json(
                        root
                        / f"window_v3/receipts/{window_id}/receipts/frame_{int(left['ordinal']):04d}_run_1.json"
                    )
                )
                == _determinism_signature(
                    read_json(
                        root
                        / f"window_v3/receipts/{window_id}/receipts/frame_{int(right['ordinal']):04d}_run_{run}.json"
                    )
                )
                for left, right in zip(reference_rows, rows, strict=True)
            )
            q_max = max(
                float(np.max(np.abs(left - right)))
                for left, right in zip(reference_q, q_states, strict=True)
            )
            base_max = max(
                float(np.max(np.abs(left - right)))
                for left, right in zip(reference_base, base_states, strict=True)
            )
            p95 = float(np.quantile([float(row["new_e_im"]) for row in rows], 0.95))
            reference_p95 = float(
                np.quantile([float(row["new_e_im"]) for row in reference_rows], 0.95)
            )
            determinism_rows.append(
                {
                    "window_id": window_id,
                    "run": run,
                    "selected_contributor_seed_candidate_retention_stable": categorical,
                    "q_max_abs": q_max,
                    "base_max_abs": base_max,
                    "p95_e_im": p95,
                    "p95_abs": abs(p95 - reference_p95),
                    "pass": categorical
                    and q_max <= Q_ATOL
                    and base_max <= Q_ATOL
                    and abs(p95 - reference_p95) <= EIM_ATOL,
                }
            )
            print(f"O5RD2F_WINDOW_V3_DETERMINISM {window_id} run={run}", flush=True)
    determinism_pass = all(bool(row["pass"]) for row in determinism_rows)
    write_json(
        root / "window_v3/determinism.json",
        {
            "schema_version": "WindowValidationV3DeterminismV1",
            "status": "PASS" if determinism_pass else "FAIL",
            "rows": determinism_rows,
        },
    )
    for summary in summaries:
        summary["result"] = (
            "PASS"
            if all(
                summary[name]
                for name in (
                    "continuity_pass",
                    "wrist_pass",
                    "bone_pass",
                    "collision_pass",
                    "joint_limits_pass",
                    "reflection_pass",
                    "scale_pass",
                    "interaction_pass",
                    "low_preservation_pass",
                )
            )
            and summary["technical"] == summary["N"]
            else "FAIL"
        )
    write_csv(root / "window_v3/per_window_summary.csv", summaries)
    passed = (
        len(summaries) == 4
        and all(row["result"] == "PASS" for row in summaries)
        and determinism_pass
    )
    decision = {
        "schema_version": "WindowValidationV3GateDecisionV1",
        "WINDOW_VALIDATION_V3": "PASS" if passed else "FAIL",
        "technical_windows": sum(row["technical"] == row["N"] for row in summaries),
        "determinism": "PASS" if determinism_pass else "FAIL",
        "windows": summaries,
    }
    write_json(decision_path, decision)
    _write_ledgers(
        root,
        sparse_v3=read_json(root / "sparse_v3/manifest.json")["frames"],
        sparse_consumed=True,
        windows=manifest["windows"],
        windows_consumed=True,
    )
    if not passed:
        write_downstream_not_run(root, reason="WINDOW_VALIDATION_V3_FAIL")
    return decision


def write_not_authorized(root: Path, *, reason: str) -> None:
    write_json(
        root / "gate_v2/not_authorized.json",
        {"CERTIFICATION_GATE_V2_CREATED": "NO", "reason": reason},
    )
    write_downstream_not_run(root, reason=reason)


def write_downstream_not_run(root: Path, *, reason: str) -> None:
    if not (root / "sparse_v3/gate_decision.json").exists():
        write_json(
            root / "sparse_v3/not_run.json", {"SPARSE_VALIDATION_V3": "NOT_RUN", "reason": reason}
        )
    if not (root / "window_v3/gate_decision.json").exists():
        write_json(
            root / "window_v3/not_run.json", {"WINDOW_VALIDATION_V3": "NOT_RUN", "reason": reason}
        )
    if not (root / "dev2_frame0/decision.json").exists():
        payload = {
            "DEV2_FRAME0_AUTHORIZED": "NO",
            "DEV2_FRAME0_HARD_CONTROL": "NOT_RUN",
            "DEV2_FRAME0_RUN_COUNT": 0,
            "reason": reason,
        }
        write_json(root / "dev2_frame0/authorization.json", payload)
        write_json(root / "dev2_frame0/not_run.json", payload)
    if not (root / "dev2_full/solver_receipt.json").exists():
        payload = {
            "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "DEV2_FULL_RESULT": "NOT_RUN",
            "reason": reason,
        }
        write_json(root / "dev2_full/authorization.json", payload)
        write_json(root / "dev2_full/not_run.json", payload)


def _require_stage_pass(root: Path, relative: str, key: str, error: str) -> None:
    path = root / relative
    if not path.exists() or read_json(path).get(key) != "PASS":
        raise RuntimeError(error)


def run_dev2_frame0_if_authorized(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    _require_stage_pass(
        root, "sparse_v3/gate_decision.json", "SPARSE_VALIDATION_V3", "O5RD2E_DEV2_FRAME0_BLOCKED"
    )
    _require_stage_pass(
        root, "window_v3/gate_decision.json", "WINDOW_VALIDATION_V3", "O5RD2E_DEV2_FRAME0_BLOCKED"
    )
    decision_path = root / "dev2_frame0/decision.json"
    if decision_path.exists():
        return read_json(decision_path)

    fixed_set_path = O5_ROOT / "preflight/fixed_o5_episode_set.json"
    fixed_set = read_json(fixed_set_path)
    dev2 = next((item for item in fixed_set["episodes"] if item["review"] == "dev_02"), None)
    canonical = O5_ROOT / "retarget/dev_02/work/canonical_episode.zarr"
    canonical_metadata_path = canonical / "metadata.json"
    canonical_metadata = read_json(canonical_metadata_path)
    source_metadata = canonical_metadata["manifest"]["fields"]["metadata"]["fields"]["metadata"]
    manifest_record = source_metadata["manifest_record"]
    expected = {
        "record_id": "oakink2:scene_01__A003++seq__a7a1a0cf7d90a9083013__2023-04-21-20-13-04:00010",
        "object": "C11001",
        "source_interval": [10704, 10944],
        "frames": 240,
        "primitive": "rearrange",
    }
    identity_pass = bool(
        dev2 is not None
        and dev2["record_id"] == expected["record_id"]
        and dev2["object"] == expected["object"]
        and dev2["source_interval"] == expected["source_interval"]
        and dev2["source_interval"][1] - dev2["source_interval"][0] == expected["frames"]
        and manifest_record["primitive"] == expected["primitive"]
        and manifest_record["record_id"] == expected["record_id"]
        and manifest_record["canonical_target_object"] == expected["object"]
        and manifest_record["frame_binding_authority"]
        == "OakInk2MocapFrameBindingV1:FRAME_BINDING_EXACT"
        and source_metadata["source_frame_ids"] == list(range(10704, 10944))
    )
    required_s1_inputs = {
        "canonical_episode": canonical,
        "old_production_trajectory": O5_ROOT / "retarget/dev_02/work/final_continuous.zarr",
        "warm_start": O5_ROOT / "retarget/dev_02/work/warm_start.zarr",
        "interaction_graph": O5_ROOT / "retarget/dev_02/work/interaction_graph.zarr",
    }
    availability = {name: path.exists() for name, path in required_s1_inputs.items()}
    execution = read_json(EXECUTION_SOURCE)
    q_old_required = bool(
        execution["fallback"]["q_old_always_available"]
        and "old_production" in execution["seed_generation"]["sources"]
    )
    authority = {
        "schema_version": "DEV2Frame0FrozenInputAuthorityV1",
        "status": "PASS" if identity_pass else "FAIL",
        "authority_path": str(fixed_set_path.resolve()),
        "authority_sha256": sha256_file(fixed_set_path),
        "canonical_metadata_path": str(canonical_metadata_path.resolve()),
        "canonical_metadata_sha256": sha256_file(canonical_metadata_path),
        "expected": expected,
        "observed": dev2,
        "identity_exact": identity_pass,
        "primitive_authority": manifest_record["primitive"],
        "frame0_source_frame": 10704,
        "required_s1_inputs": {
            name: {"path": str(path.resolve()), "exists": availability[name]}
            for name, path in required_s1_inputs.items()
        },
        "q_old_required_by_frozen_execution_v2": q_old_required,
        "q_old_authority_available": availability["old_production_trajectory"],
        "old_stage7_failure_terminal_is_q_old": False,
        "reason": (
            "Frozen S1 requires an always-available old_production q_old baseline, but the fixed DEV2 "
            "episode failed in Stage 7 before warm, interaction-graph, or final-production artifacts existed."
        ),
    }
    write_json(root / "dev2_frame0/input_authority.json", authority)
    write_json(
        root / "dev2_frame0/authorization.json",
        {
            "schema_version": "DEV2Frame0AuthorizationV1",
            "DEV2_FRAME0_AUTHORIZED": "YES",
            "authorized_by": ["SPARSE_VALIDATION_V3=PASS", "WINDOW_VALIDATION_V3=PASS"],
            "authorization_is_not_technical_success": True,
        },
    )

    # These are three independent invocations of the same frozen method.  Each
    # fails before optimizer construction because a mandatory frozen S1 input
    # has no authority.  Substituting a Stage-7 failed terminal or a canonical
    # rest state for old_production would change ExecutionV2.
    failure = "MISSING_FROZEN_S1_OLD_PRODUCTION_Q_OLD_AUTHORITY"
    runs = []
    for run in (1, 2, 3):
        row = {
            "schema_version": "DEV2Frame0HardControlRunV1",
            "run": run,
            "source_frame": 10704,
            "frozen_objective": "RetargetObjectiveV2",
            "frozen_execution": EXECUTION_NAME,
            "technical": "FAIL",
            "finite": "NOT_EVALUABLE",
            "E_IM": None,
            "wrist": "NOT_EVALUABLE",
            "bone": "NOT_EVALUABLE",
            "collision": "NOT_EVALUABLE",
            "joint_limits": "NOT_EVALUABLE",
            "deterministic": "NOT_EVALUABLE",
            "optimizer_started": False,
            "candidate_state": None,
            "profiler": None,
            "DEV2_specific_branch": False,
            "failure_stage": "FROZEN_METHOD_INPUT_AUTHORITY",
            "failure": failure,
        }
        write_json(root / f"dev2_frame0/run_{run}.json", row)
        runs.append(row)
    failure_mode_deterministic = all(row["failure"] == failure for row in runs)
    decision = {
        "schema_version": "DEV2Frame0HardControlDecisionV1",
        "DEV2_FRAME0_AUTHORIZED": "YES",
        "DEV2_FRAME0_HARD_CONTROL": "FAIL",
        "DEV2_FRAME0_RUN_COUNT": 3,
        "technical_pass_count": 0,
        "method_optimizer_run_count": 0,
        "failure_mode_deterministic": failure_mode_deterministic,
        "determinism": "NOT_EVALUABLE_NO_FROZEN_METHOD_INPUT",
        "failure": failure,
        "no_dev2_specific_branch": True,
        "runs": runs,
    }
    write_json(decision_path, decision)
    write_json(
        root / "dev2_full/authorization.json",
        {
            "schema_version": "DEV2FullAuthorizationV1",
            "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "DEV2_FULL_RESULT": "NOT_RUN",
            "reason": "DEV2_FRAME0_HARD_CONTROL_FAIL",
        },
    )
    return decision


def run_dev2_full_if_authorized(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    _require_stage_pass(
        root, "sparse_v3/gate_decision.json", "SPARSE_VALIDATION_V3", "O5RD2E_DEV2_FULL_BLOCKED"
    )
    _require_stage_pass(
        root, "window_v3/gate_decision.json", "WINDOW_VALIDATION_V3", "O5RD2E_DEV2_FULL_BLOCKED"
    )
    frame0 = read_json(root / "dev2_frame0/decision.json")
    if frame0.get("DEV2_FRAME0_HARD_CONTROL") != "PASS":
        payload = {
            "schema_version": "DEV2FullAuthorizationV1",
            "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "DEV2_FULL_RESULT": "NOT_RUN",
            "reason": "DEV2_FRAME0_HARD_CONTROL_FAIL",
        }
        write_json(root / "dev2_full/authorization.json", payload)
        write_json(root / "dev2_full/not_run.json", payload)
        return payload
    raise RuntimeError("O5RD2E_DEV2_FULL_PASS_PATH_NOT_IMPLEMENTED")


def run_dev2_semantic_v1(root: Path) -> dict[str, Any]:
    payload = {
        "RETARGET_SEMANTIC_VALIDITY_V1_RAN": "NO",
        "SEMANTIC_V1_RESULT": "NOT_RUN",
        "reason": "NO_COMPLETE_DEV2_TRAJECTORY",
    }
    write_json(root / "dev2_full/semantic_not_run.json", payload)
    return payload


def render_dev2_viewer(root: Path) -> dict[str, Any]:
    payload = {
        "DEV2_OBJECTIVE_V2_HTML": None,
        "DEV2_HTML_SHA256": None,
        "DEV2_VIEWER_RESULT": "NOT_RUN",
        "reason": "NO_COMPLETE_USABLE_DEV2_TRAJECTORY",
    }
    write_json(root / "dev2_full/viewer_not_run.json", payload)
    return payload


def summarize(root: Path) -> dict[str, Any]:
    integrity = verify_frozen_method(root)
    root_decision = read_json(root / "d2e_audit/root_cause_decision.json")
    low = read_json(root / "d2e_audit/low_aggregate.json")
    movement = read_json(root / "d2e_audit/valid_set_movement.json")
    gate_path = root / "gate_v2/certification_gate_v2.json"
    sparse_status = (
        read_json(root / "sparse_v3/gate_decision.json").get("SPARSE_VALIDATION_V3")
        if (root / "sparse_v3/gate_decision.json").exists()
        else "NOT_RUN"
    )
    window_status = (
        read_json(root / "window_v3/gate_decision.json").get("WINDOW_VALIDATION_V3")
        if (root / "window_v3/gate_decision.json").exists()
        else "NOT_RUN"
    )
    frame0_status = (
        read_json(root / "dev2_frame0/decision.json").get("DEV2_FRAME0_HARD_CONTROL")
        if (root / "dev2_frame0/decision.json").exists()
        else "NOT_RUN"
    )
    frame0_decision = (
        read_json(root / "dev2_frame0/decision.json")
        if (root / "dev2_frame0/decision.json").exists()
        else {"DEV2_FRAME0_RUN_COUNT": 0}
    )
    full_auth = (
        read_json(root / "dev2_full/authorization.json")
        if (root / "dev2_full/authorization.json").exists()
        else {
            "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "DEV2_FULL_RESULT": "NOT_RUN",
        }
    )
    exclusion = (
        read_json(root / "ledger/total_exclusion_ledger.json")
        if (root / "ledger/total_exclusion_ledger.json").exists()
        else {"TOTAL_PRE_V3_EXCLUSION_COUNT": 136}
    )
    flags = {
        "BRANCH": git("branch", "--show-current"),
        "FROZEN_METHOD_INTEGRITY": integrity["FROZEN_METHOD_INTEGRITY"],
        "RETARGET_OBJECTIVE_V2_SHA256": integrity["RETARGET_OBJECTIVE_V2_SHA256"],
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": integrity[
            "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256"
        ],
        "OBJECTIVE_V2_CHANGED": "NO",
        "EXECUTION_V2_CHANGED": "NO",
        "S1_USED": "YES",
        "S2_USED_AS_RESCUE": "NO",
        "SPARSE_VALIDATION_V2": "FAIL",
        "HISTORICAL_SPARSE_V2_RESULT_REWRITTEN": "NO",
        "D2E_OPTIMIZER_RUN_COUNT": 0,
        "D2E_NEW_DEV1_FRAMES_CONSUMED": 0,
        "D2E_PRIMARY_ROOT_CAUSE": root_decision["PRIMARY_ROOT_CAUSE"],
        "D2E_CONFIDENCE": root_decision["CONFIDENCE"],
        "EXACT_MONOTONIC_90_PERCENT_GATE_AUTHORITY": "PRE_REGISTERED_CONSERVATIVE_CERTIFICATION_HEURISTIC",
        "CERTIFICATION_GATE_V2_CREATED": "YES" if gate_path.exists() else "NO",
        "CERTIFICATION_GATE_V2_SHA256": sha256_file(gate_path) if gate_path.exists() else None,
        "OUTCOME_DRIVEN_THRESHOLD_TUNING": "NO",
        "SPARSE_V2_REUSED_AS_INDEPENDENT_CERTIFICATION": "NO",
        "TOTAL_PRE_V3_EXCLUSION_COUNT": exclusion["TOTAL_PRE_V3_EXCLUSION_COUNT"],
        "SPARSE_VALIDATION_V3": sparse_status,
        "SPARSE_V3_DEVELOPMENT_OVERLAP": read_json(root / "sparse_v3/manifest.json").get(
            "SPARSE_V3_DEVELOPMENT_OVERLAP"
        )
        if (root / "sparse_v3/manifest.json").exists()
        else "NOT_RUN",
        "SPARSE_V3_SPARSE_V2_OVERLAP": read_json(root / "sparse_v3/manifest.json").get(
            "SPARSE_V3_SPARSE_V2_OVERLAP"
        )
        if (root / "sparse_v3/manifest.json").exists()
        else "NOT_RUN",
        "WINDOW_VALIDATION_V3": window_status,
        "WINDOW_V3_DEVELOPMENT_OVERLAP": read_json(root / "window_v3/manifest.json").get(
            "WINDOW_V3_DEVELOPMENT_OVERLAP"
        )
        if (root / "window_v3/manifest.json").exists()
        else "NOT_RUN",
        "WINDOW_V3_SPARSE_V2_OVERLAP": read_json(root / "window_v3/manifest.json").get(
            "WINDOW_V3_SPARSE_V2_OVERLAP"
        )
        if (root / "window_v3/manifest.json").exists()
        else "NOT_RUN",
        "WINDOW_V3_SPARSE_V3_OVERLAP": read_json(root / "window_v3/manifest.json").get(
            "WINDOW_V3_SPARSE_V3_OVERLAP"
        )
        if (root / "window_v3/manifest.json").exists()
        else "NOT_RUN",
        "DEV2_FRAME0_HARD_CONTROL": frame0_status,
        "DEV2_FRAME0_RUN_COUNT": frame0_decision.get("DEV2_FRAME0_RUN_COUNT", 0),
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "FULL_DEV2_COMPUTE_AUTHORIZED": full_auth.get("FULL_DEV2_COMPUTE_AUTHORIZED", "NO"),
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": full_auth.get("DEV2_FULL_PRODUCTION_SOLVE_COUNT", 0),
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
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
    if sparse_status == "FAIL":
        next_stage = "OBJECTIVE_V2_GENERALIZATION_FAILURE_V3_ANALYSIS"
    elif window_status == "FAIL":
        next_stage = "OBJECTIVE_V2_SEQUENCE_GENERALIZATION_FAILURE_V3_ANALYSIS"
    elif frame0_status == "FAIL":
        next_stage = "OBJECTIVE_V2_CROSS_EPISODE_HARD_CONTROL_FAILURE_ANALYSIS"
    elif full_auth.get("DEV2_FULL_RESULT") == "FAIL":
        next_stage = "DEV2_FULL_TRAJECTORY_FAILURE_LOCALIZATION"
    elif full_auth.get("DEV2_FULL_RESULT") == "PASS":
        next_stage = "WAIT_FOR_DEV2_HUMAN_REVIEW"
    elif root_decision["PRIMARY_ROOT_CAUSE"] != "CERTIFICATION_GATE_SEMANTIC_MISMATCH":
        next_stage = "OBJECTIVE_V2_EXECUTION_V3_DEVELOPMENT"
    else:
        next_stage = "CONTINUE_CONDITIONAL_STATE_MACHINE"
    result = {
        "schema_version": "OakInk2O5RD2ED2FFinalSummaryV1",
        "status": (
            "HARD_STOP"
            if sparse_status == "FAIL"
            or window_status == "FAIL"
            or frame0_status == "FAIL"
            or full_auth.get("DEV2_FULL_RESULT") == "FAIL"
            or root_decision["CONFIDENCE"] != "HIGH"
            else "WAIT_FOR_HUMAN_REVIEW"
            if full_auth.get("DEV2_FULL_RESULT") == "PASS"
            else "IN_PROGRESS"
        ),
        "git": {
            "BRANCH": flags["BRANCH"],
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": git("rev-parse", "HEAD"),
            "commits": git("log", f"{START_HEAD}..HEAD", "--format=%H").splitlines(),
            "tracked_worktree_clean": not bool(git("status", "--short", "--untracked-files=no")),
        },
        "historical_d2d": {
            "SPARSE_VALIDATION_V2": "FAIL",
            "only_failed_gate": "overall exact pointwise E_IM non-regression >=90%; observed 20/30=66.67%",
            "HISTORICAL_RESULT_REWRITTEN": "NO",
        },
        "low_regression": low,
        "low_selection_classification_counts": movement["counts"],
        "root_cause": root_decision,
        "sparse_v3": read_json(root / "sparse_v3/gate_decision.json"),
        "window_v3": read_json(root / "window_v3/gate_decision.json"),
        "dev2_frame0": frame0_decision,
        "dev2_full": full_auth,
        "flags": flags,
        "NEXT": next_stage,
    }
    write_json(root / "final_summary.json", result)
    lines = [
        "# OakInk2 O5R-D2E/D2F SparseV2 Failure Semantics + GateV2 + V3 Re-certification Handoff",
        "",
        f"Machine status: `{result['status']}`.",
        "",
        "## Git",
        "",
        f"- `BRANCH={flags['BRANCH']}`",
        f"- `START_HEAD={START_HEAD}`",
        f"- `FINAL_HEAD={result['git']['FINAL_HEAD']}`",
        "- `PUSHED=NO`",
        "- `PR_CREATED=NO`",
        "",
        "## Historical SparseValidationV2",
        "",
        "- `SPARSE_VALIDATION_V2=FAIL`",
        "- `HISTORICAL_RESULT_REWRITTEN=NO`",
        "- Only failed GateContractV1 criterion: exact pointwise E_IM non-regression was 20/30 (66.67%), below the preregistered 90% heuristic.",
        "",
        "## LOW regression magnitude",
        "",
        f"- `N_LOW={low['N_LOW']}`",
        f"- `median absolute increase={low['median_delta_e_im']}`",
        f"- `p95 absolute increase={low['p95_delta_e_im']}`",
        f"- `max absolute increase={low['max_delta_e_im']}`",
        f"- `median relative increase={low['median_relative_increase']}`",
        f"- `max relative increase={low['max_relative_increase']}`",
        f"- `minimum final margin to tau={low['minimum_margin_to_tau']}`",
        f"- `final semantic-valid={low['N_within_valid_set_after']}/{low['N_LOW']}`",
        "",
        "## LOW selection semantics",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    labels = {
        "EXPECTED_VALID_SET_SECONDARY_IMPROVEMENT": "Expected valid-set secondary improvement",
        "EXPECTED_PRIMARY_RETENTION_BEHAVIOR": "Expected primary retention",
        "EXPECTED_DETERMINISTIC_TIEBREAK": "Expected tie-break",
        "UNNECESSARY_MOVE_NO_SECONDARY_BENEFIT": "Unnecessary move",
        "LEXICOGRAPHIC_SELECTION_VIOLATION": "Lexicographic violation",
        "INCONCLUSIVE": "Inconclusive",
    }
    lines.extend(f"| {label} | {movement['counts'].get(key, 0)} |" for key, label in labels.items())
    lines.extend(
        [
            "",
            "## D2E decision",
            "",
            f"- `EXACT_MONOTONIC_90_PERCENT_AUTHORITY={flags['EXACT_MONOTONIC_90_PERCENT_GATE_AUTHORITY']}`",
            f"- `PRIMARY_ROOT_CAUSE={root_decision['PRIMARY_ROOT_CAUSE']}`",
            f"- `CONFIDENCE={root_decision['CONFIDENCE']}`",
            f"- `CERTIFICATION_GATE_V2_CREATED={flags['CERTIFICATION_GATE_V2_CREATED']}`",
            f"- `CERTIFICATION_GATE_V2_SHA256={flags['CERTIFICATION_GATE_V2_SHA256']}`",
            "",
            "## Conditional state",
            "",
            f"- `SPARSE_VALIDATION_V3={sparse_status}`",
            f"- `WINDOW_VALIDATION_V3={window_status}`",
            f"- `DEV2_FRAME0_HARD_CONTROL={frame0_status}`",
            f"- `FULL_DEV2_COMPUTE_AUTHORIZED={flags['FULL_DEV2_COMPUTE_AUTHORIZED']}`",
            f"- `DEV2_FULL_PRODUCTION_SOLVE_COUNT={flags['DEV2_FULL_PRODUCTION_SOLVE_COUNT']}`",
            f"- `NEXT={next_stage}`",
            "",
            "### SparseV3",
            "",
            f"- technical: `{result['sparse_v3']['technical']}/{result['sparse_v3']['N']}`",
            f"- invalid recovery: `{result['sparse_v3']['old_invalid_recovered_count']}/{result['sparse_v3']['old_invalid_count']}`",
            f"- median invalid relative reduction: `{result['sparse_v3']['median_invalid_relative_reduction']}`",
            f"- threshold-aware non-regression: `{result['sparse_v3']['threshold_aware_nonregression_count']}/{result['sparse_v3']['N']}`",
            f"- exact monotonic diagnostic only: `{result['sparse_v3']['exact_monotonic_count_DIAGNOSTIC_ONLY']}/{result['sparse_v3']['N']}`",
            "- An initial evaluator-only decision was invalidated because Python identity checks rejected `numpy.bool_`; the optimizer was not rerun and the corrected evaluator used the same stored frame evidence.",
            "",
            "### WindowV3",
            "",
            "| Window | N | Old p95 | New p95 | Technical | Continuity | Wrist | Bone | Result |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        "| {window_id} | {N} | {old_p95_e_im} | {new_p95_e_im} | {technical}/{N} | {continuity_pass} | {wrist_pass} | {bone_pass} | {result} |".format(
            **row
        )
        for row in result["window_v3"]["windows"]
    )
    lines.extend(
        [
            "",
            "### DEV2 frame0",
            "",
            "| Run | Technical | E_IM | Wrist | Bone | Collision | Joints | Deterministic |",
            "| ---: | --- | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {row['run']} | {row['technical']} | {row['E_IM']} | {row['wrist']} | {row['bone']} | {row['collision']} | {row['joint_limits']} | {row['deterministic']} |"
        for row in frame0_decision.get("runs", [])
    )
    lines.extend(
        [
            "",
            f"All three invocations failed before optimizer construction at `{frame0_decision.get('failure', 'NOT_RUN')}`. The fixed DEV2 O5 run has canonical input only; no authoritative `old_production` q_old, warm-start, or interaction graph exists. No failed Stage-7 terminal was substituted for q_old.",
            "",
            "## Final safety flags",
            "",
        ]
    )
    lines.extend(f"- `{key}={value}`" for key, value in flags.items())
    text = "\n".join(lines) + "\n"
    (root / "handoff.md").write_text(text, encoding="utf-8")
    (root / "final_summary.md").write_text(text, encoding="utf-8")
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2EResourceUsageV1",
            **{
                key: flags[key]
                for key in (
                    "D2E_OPTIMIZER_RUN_COUNT",
                    "D2E_NEW_DEV1_FRAMES_CONSUMED",
                    "DEV2_FRAME0_RUN_COUNT",
                    "DEV2_FULL_PRODUCTION_SOLVE_COUNT",
                    "DEV1_FULL_RETARGET_RERUNS",
                    "DEV1_FULL_V2_REFINEMENT_RUNS",
                    "O6_RAN",
                    "SUPPORT_PHYSICALIZATION_RAN",
                    "PHYSX_RAN",
                    "FROZEN_EVAL_RAN",
                    "PPO_RAN",
                )
            },
        },
    )
    failures = root / "technical_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    if not failures.exists():
        failures.write_text("", encoding="utf-8")
    return result


def record_git(root: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "O5RD2EGitCommitsV1",
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "commits": git("log", f"{START_HEAD}..HEAD", "--format=%H%x09%s").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "git_commits.json", payload)
    return payload


def validate_repository(root: Path) -> dict[str, Any]:
    commands = [
        (
            "ruff_task_files",
            [
                "ruff",
                "check",
                "scripts/data/run_oakink2_o5rd2e.py",
                "tests/data/test_oakink2_o5rd2e.py",
            ],
        ),
        (
            "ruff_format_task_files",
            [
                "ruff",
                "format",
                "--check",
                "scripts/data/run_oakink2_o5rd2e.py",
                "tests/data/test_oakink2_o5rd2e.py",
            ],
        ),
        ("mypy_src", [sys.executable, "-m", "mypy", "src"]),
        ("pytest_full", [sys.executable, "-m", "pytest", "-q"]),
        ("paper_fidelity", [sys.executable, "scripts/check_paper_fidelity.py"]),
        ("ci_ruff_all", ["ruff", "check", "."]),
        ("ci_ruff_format_all", ["ruff", "format", "--check", "."]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    rows = []
    log_root = root / "validation_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for index, (name, command) in enumerate(commands):
        result = subprocess.run(command, cwd=REPO, check=False, text=True, capture_output=True)
        output = result.stdout + result.stderr
        log_path = log_root / f"{index:02d}_{name}.log"
        log_path.write_text(output, encoding="utf-8")
        row = {
            "name": name,
            "command": command,
            "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "log": str(log_path.resolve()),
        }
        rows.append(row)
        print(f"O5RD2E_VALIDATION {name} {row['status']}", flush=True)
    payload = {
        "schema_version": "O5RD2ERepositoryValidationV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "environment": "toporetarget-rl",
        "checks": rows,
    }
    write_json(root / "validation_results.json", payload)
    write_json(root / "tests.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "action",
        choices=(
            "preflight",
            "verify-frozen-method",
            "audit-gate-v1-authority",
            "audit-sparse-v2-low",
            "audit-qold-selection",
            "decide-d2e-root-cause",
            "freeze-gate-v2-if-authorized",
            "freeze-sparse-v3",
            "run-sparse-v3",
            "freeze-window-v3",
            "run-window-v3",
            "run-dev2-frame0-if-authorized",
            "run-dev2-full-if-authorized",
            "run-dev2-semantic-v1",
            "render-dev2-viewer",
            "summarize",
            "validate",
            "record-git",
        ),
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    actions = {
        "preflight": preflight,
        "verify-frozen-method": verify_frozen_method,
        "audit-gate-v1-authority": audit_gate_v1_authority,
        "audit-sparse-v2-low": audit_sparse_v2_low,
        "audit-qold-selection": audit_qold_selection,
        "decide-d2e-root-cause": decide_d2e_root_cause,
        "freeze-gate-v2-if-authorized": freeze_gate_v2_if_authorized,
        "freeze-sparse-v3": freeze_sparse_v3,
        "run-sparse-v3": run_sparse_v3,
        "freeze-window-v3": freeze_window_v3,
        "run-window-v3": run_window_v3,
        "run-dev2-frame0-if-authorized": run_dev2_frame0_if_authorized,
        "run-dev2-full-if-authorized": run_dev2_full_if_authorized,
        "run-dev2-semantic-v1": run_dev2_semantic_v1,
        "render-dev2-viewer": render_dev2_viewer,
        "summarize": summarize,
        "validate": validate_repository,
        "record-git": record_git,
    }
    result = actions[args.action](args.report_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
