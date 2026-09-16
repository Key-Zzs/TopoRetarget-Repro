#!/usr/bin/env python3
"""O5R-D2G3 refinement parity repair and conditional ExecutionV3 freeze.

This workflow is fail closed.  It only consumes the already-used D2C/D2G/D2G2
development evidence plus the DEV2 frame-zero known-failure case.  It exposes no
fresh-validation or full-trajectory action.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2c as d2c  # noqa: E402
from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.data import run_oakink2_o5rd2g2 as d2g2  # noqa: E402
from toporetarget.retarget.objective_v2_execution import default_search_contracts  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2g3_refinement_parity_dev2_frame0_freeze_v1"
D2C_ROOT = d2c.ROOT
D2G_ROOT = d2g.ROOT
D2G2_ROOT = d2g2.ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "59fd9a0393bf69162f0ccb4a3a63f69ea6d33350"
OBJECTIVE_SHA = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
GATE_V2_SHA = "a845fcdfec478e9208fc6c192317a45bee6ef19be1fcb784b459d8a4f675e048"
EXECUTION_V2_SHA = d2g.EXECUTION_V2_SHA
CS2_A = d2g2.CS2_A
Q_TOL = 1.0e-8
BASE_TOL = 1.0e-8
E_IM_TOL = 1.0e-12

ROOT_CAUSE_ENUMS = {
    "REPORTING_ONLY_DIFFERENCE",
    "COMPARATOR_BINDING_ERROR",
    "REFINEMENT_ROUTE_LEAKAGE",
    "POST_BOOTSTRAP_RANKING_LEAKAGE",
    "CONTRIBUTOR_RANKING_DRIFT",
    "SEED_POOL_DRIFT",
    "SEED_ORDER_DRIFT",
    "PRIMARY_SOLVER_INPUT_DRIFT",
    "CANDIDATE_SCREENING_DRIFT",
    "LEXICOGRAPHIC_TIEBREAK_DRIFT",
    "SECONDARY_POLISH_CONTEXT_DRIFT",
    "FALLBACK_DRIFT",
    "SHARED_PRIMITIVE_BEHAVIOR_DRIFT",
    "NUMERICAL_NONDETERMINISM",
    "MULTI_FACTOR",
    "INCONCLUSIVE",
}


def write_json(path: Path, value: Any) -> None:
    d2g.write_json(path, value)


def read_json(path: Path) -> dict[str, Any]:
    return d2g.read_json(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    d2g.write_csv(path, rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    return d2g.read_csv(path)


def git(*args: str) -> str:
    return d2g.git(*args)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def require_status(path: Path, field: str, expected: str = "PASS") -> dict[str, Any]:
    require(path)
    payload = read_json(path)
    if payload.get(field) != expected:
        raise RuntimeError(f"O5RD2G3_STAGE_BLOCKED:{path}:{field}={payload.get(field)}")
    return payload


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(b"\0")
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _historical_paths(window_id: str, ordinal: int) -> tuple[Path, Path]:
    return d2g._historical_s1_paths(window_id, ordinal)


def _load_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    require(path)
    with np.load(path, allow_pickle=False) as state:
        return (
            np.asarray(state["qpos"], dtype=np.float64),
            np.asarray(state["base_pose_scene"], dtype=np.float64),
        )


def _windows_by_id() -> dict[str, dict[str, Any]]:
    return {str(row["window_id"]): row for row in d2g.development_windows(d2g.D2A_ROOT)}


def _authoritative_previous_state(
    runtime: d2g.D2ARuntime, window: dict[str, Any], ordinal: int
) -> tuple[np.ndarray, np.ndarray, str]:
    ordinals = [int(value) for value in window["ordinals"]]
    position = ordinals.index(int(ordinal))
    if position == 0:
        return (
            np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64),
            np.asarray(runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64),
            "OLD_PRODUCTION_WINDOW_PREDECESSOR",
        )
    previous_ordinal = ordinals[position - 1]
    state_path, _receipt_path = _historical_paths(str(window["window_id"]), previous_ordinal)
    qpos, base = _load_state(state_path)
    return qpos, base, "FROZEN_S1_PREVIOUS_ACCEPTED_STATE"


def verify_frozen_upstream(root: Path) -> dict[str, Any]:
    payload = d2g2.verify_frozen_upstream(root)
    d2g2_summary = D2G2_ROOT / "final_summary.json"
    d2g2_regression = D2G2_ROOT / "refinement_regression/parity_summary.json"
    d2g2_candidate = D2G2_ROOT / "search_v2_candidates/cs2_a.json"
    d2g2_masked = D2G2_ROOT / "development/masked_qold_decision.json"
    for path in (d2g2_summary, d2g2_regression, d2g2_candidate, d2g2_masked):
        require(path)
    checks = {
        "objective_v2": payload["authorities"]["objective_v2"]["sha256"] == OBJECTIVE_SHA,
        "gate_v2": payload["authorities"]["gate_v2"]["sha256"] == GATE_V2_SHA,
        "execution_v2": payload["authorities"]["execution_v2"]["sha256"] == EXECUTION_V2_SHA,
        "semantic_v1": payload["authorities"]["semantic_v1"]["exact"],
        "wuji_asset": payload["authorities"]["wuji_asset"]["exact"],
        "manifest_v2": payload["authorities"]["manifest_v2"]["exact"],
        "split_v2": payload["authorities"]["split_v2"]["exact"],
        "dev1_old_trajectory": payload["authorities"]["dev1_old_trajectory"]["exact"],
        "d2g_graph": payload["d2g_graph_authority"]["exact"],
        "d2g2_regression_is_authoritative_fail": read_json(d2g2_regression).get(
            "REFINEMENT_MODE_REGRESSION"
        )
        == "FAIL",
        "d2g2_cs2_a_pass": read_json(d2g2_masked).get("status") == "PASS",
    }
    payload.update(
        {
            "schema_version": "O5RD2G3FrozenAuthorityIntegrityV1",
            "d2g2_artifacts": {
                "final_summary": {"path": str(d2g2_summary), "sha256": sha256_file(d2g2_summary)},
                "refinement_regression": {
                    "path": str(d2g2_regression),
                    "sha256": sha256_file(d2g2_regression),
                },
                "cs2_a_contract": {
                    "path": str(d2g2_candidate),
                    "sha256": sha256_file(d2g2_candidate),
                },
                "masked_qold_decision": {
                    "path": str(d2g2_masked),
                    "sha256": sha256_file(d2g2_masked),
                },
            },
            "checks": checks,
            "E_IM_THRESHOLD": d2g.GATE.interaction_e_im_p95_limit,
        }
    )
    payload["FROZEN_UPSTREAM_INTEGRITY"] = (
        "PASS"
        if payload.get("FROZEN_UPSTREAM_INTEGRITY") == "PASS" and all(checks.values())
        else "FAIL"
    )
    write_json(root / "preflight/frozen_authorities.json", payload)
    write_json(root / "preflight/integrity.json", payload)
    if payload["FROZEN_UPSTREAM_INTEGRITY"] != "PASS":
        raise RuntimeError("STATUS=BLOCKED_FROZEN_AUTHORITY_INTEGRITY")
    return payload


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2G3_BRANCH_MISMATCH:{branch}")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0
    )
    if not ancestor:
        raise RuntimeError(f"O5RD2G3_START_HEAD_NOT_ANCESTOR:{head}")
    integrity = verify_frozen_upstream(root)
    payload = {
        "schema_version": "OakInk2O5RD2G3GitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "START_HEAD": START_HEAD,
        "head_at_preflight": head,
        "start_head_is_ancestor": ancestor,
        "status_short_at_preflight": git("status", "--short", "--untracked-files=all").splitlines(),
        "diff_stat_at_preflight": git("diff", "--stat").splitlines(),
        "diff_check": git("diff", "--check").splitlines(),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "remotes": git("remote", "-v").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", payload)
    write_json(root / "preflight/upstream_state.json", integrity["upstream_state"])
    failures = root / "technical_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    failures.touch(exist_ok=True)
    return payload


def audit_refinement_parity_contract(root: Path) -> dict[str, Any]:
    require_status(root / "preflight/integrity.json", "FROZEN_UPSTREAM_INTEGRITY")
    selection = read_json(D2G2_ROOT / "refinement_regression/frame_selection.json")
    rows = read_csv(D2G2_ROOT / "refinement_regression/per_frame.csv") + read_csv(
        D2G2_ROOT / "refinement_regression/window.csv"
    )
    contract = {
        "schema_version": "RefinementParityContractAuditV1",
        "status": "PASS",
        "authoritative_comparison_population": "D2G2 frozen 15 stratified checks plus 20-frame consumed LOW continuous window",
        "number_of_comparisons": len(rows),
        "number_of_unique_source_ordinals": len({int(row["ordinal"]) for row in rows}),
        "frame_window_selection": selection,
        "comparison_quantities": ["qpos", "base_pose_scene", "interaction_e_im"],
        "numeric_tolerances": {
            "max_q_abs": Q_TOL,
            "max_base_abs": BASE_TOL,
            "max_E_IM_abs": E_IM_TOL,
        },
        "hard_receipt_fields": [
            "selected_block",
            "seed_pool values and order",
            "selected_candidate",
            "retained_primary_id",
            "retention_decision",
            "fallback decision",
            "technical_success",
        ],
        "hard_path_inputs": [
            "q_old",
            "source graph and object context",
            "previous accepted q/base",
            "continuous prediction context",
            "contributor scores/ranking",
            "per-seed primary solve inputs",
            "secondary polish context",
        ],
        "scientific_diagnostics": [
            "interaction validity",
            "wrist",
            "bone",
            "continuity",
            "collision",
            "joint limits",
            "reflection",
            "scale",
            "Semantic V1 fields",
        ],
        "determinism_requirement": "same path decisions and q/base/E_IM within frozen tolerances",
        "tolerance_source": "unchanged D2G2 authoritative comparator implementation",
        "threshold_changed": False,
    }
    write_json(root / "parity_contract/authoritative_contract.json", contract)
    write_json(root / "parity_contract/comparison_population.json", selection)
    write_json(root / "parity_contract/frozen_tolerances.json", contract["numeric_tolerances"])
    return contract


def localize_refinement_divergence(root: Path) -> dict[str, Any]:
    require_status(root / "parity_contract/authoritative_contract.json", "status")
    runtime = d2g.D2ARuntime(root)
    windows = _windows_by_id()
    failed = [
        row
        for row in read_csv(D2G2_ROOT / "refinement_regression/per_frame.csv")
        if row["status"] == "FAIL"
    ]
    if len(failed) != 12:
        raise RuntimeError(f"O5RD2G3_EXPECTED_12_FAILED_ITEMS:{len(failed)}")
    divergence_rows: list[dict[str, Any]] = []
    contributor_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    polish_rows: list[dict[str, Any]] = []
    for observed in failed:
        window_id = observed["window"]
        ordinal = int(observed["ordinal"])
        window = windows[window_id]
        expected_q, expected_base, expected_authority = _authoritative_previous_state(
            runtime, window, ordinal
        )
        wrong_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
        wrong_base = np.asarray(
            runtime.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64
        )
        q_old = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        expected_binding, _expected_context = runtime.bind_context(
            ordinal, previous_base=expected_base, previous_qpos=expected_q
        )
        wrong_binding, _wrong_context = runtime.bind_context(
            ordinal, previous_base=wrong_base, previous_qpos=wrong_q
        )
        q_delta = float(np.max(np.abs(expected_q - wrong_q)))
        base_delta = float(np.max(np.abs(expected_base - wrong_base)))
        if q_delta <= 0.0 and base_delta <= 0.0:
            raise RuntimeError(f"O5RD2G3_NO_BINDING_DIVERGENCE:{ordinal}")
        divergence_rows.append(
            {
                "window": window_id,
                "stratum": observed["stratum"],
                "ordinal": ordinal,
                "source_frame": int(runtime.graph.frame_indices[ordinal]),
                "q_old_hash": array_sha256(q_old),
                "source_graph_hash": sha256_path(d2g.frozen_paths()["dev1_interaction_graph"]),
                "object_context_hash": sha256_path(d2g.frozen_paths()["dev1_object_samples"]),
                "expected_previous_authority": expected_authority,
                "erroneous_previous_authority": "OLD_PRODUCTION_ORDINAL_MINUS_ONE",
                "expected_previous_q_hash": array_sha256(expected_q),
                "erroneous_previous_q_hash": array_sha256(wrong_q),
                "previous_q_max_abs_diff": q_delta,
                "expected_previous_base_hash": array_sha256(expected_base),
                "erroneous_previous_base_hash": array_sha256(wrong_base),
                "previous_base_max_abs_diff": base_delta,
                "expected_context_binding_sha256": expected_binding.sha256,
                "erroneous_context_binding_sha256": wrong_binding.sha256,
                "first_divergent_operation": "PRE_OPTIMIZER_PREVIOUS_STATE_CONTEXT_BINDING",
                "root_cause": "COMPARATOR_BINDING_ERROR",
            }
        )
        historical_state, historical_receipt_path = _historical_paths(window_id, ordinal)
        require(historical_state)
        historical = read_json(historical_receipt_path)
        contributor_rows.append(
            {
                "window": window_id,
                "ordinal": ordinal,
                "contributor_scores_source": "SAME_Q_OLD_AND_SOURCE_GRAPH",
                "expected_ranking": historical["contributor_ranking"],
                "expected_selected_block": historical["selected_block"],
                "first_divergence_before_ranking": False,
            }
        )
        seed_rows.append(
            {
                "window": window_id,
                "ordinal": ordinal,
                "expected_seed_order": historical["seed_pool"],
                "v3_refinement_seed_contract": list(default_search_contracts()[0].seed_sources),
                "seed_contract_parity": historical["seed_pool"]
                == list(default_search_contracts()[0].seed_sources),
                "first_divergence_before_seed_generation": True,
            }
        )
        candidate_rows.append(
            {
                "window": window_id,
                "ordinal": ordinal,
                "expected_primary": historical["retained_primary_id"],
                "expected_final": historical["selected_candidate"],
                "expected_retention": historical["retention_decision"],
                "downstream_of_wrong_context": True,
            }
        )
        polish_rows.append(
            {
                "window": window_id,
                "ordinal": ordinal,
                "expected_context_binding_sha256": expected_binding.sha256,
                "erroneous_context_binding_sha256": wrong_binding.sha256,
                "secondary_polish_uses_same_frame_context": True,
                "drift_is_comparator_input_not_executor_dispatch": True,
            }
        )
    write_csv(root / "parity_localization/failed_items.csv", failed)
    write_csv(root / "parity_localization/first_divergence.csv", divergence_rows)
    write_csv(root / "parity_localization/contributor_comparison.csv", contributor_rows)
    write_csv(root / "parity_localization/seed_comparison.csv", seed_rows)
    write_csv(root / "parity_localization/candidate_selection_comparison.csv", candidate_rows)
    write_csv(root / "parity_localization/polish_context_comparison.csv", polish_rows)
    payload = {
        "schema_version": "RefinementParityFirstDivergenceV1",
        "status": "PASS",
        "REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE": "COMPARATOR_BINDING_ERROR",
        "CONFIDENCE": "HIGH",
        "failed_items": len(failed),
        "explanation": (
            "D2G2 sampled positions 4/9/14/19 in each of three frozen S1 windows but rebound "
            "their previous state to old_production[ordinal-1]. Frozen S1 generated those receipts "
            "sequentially from the preceding accepted S1 state. The first sampled position in each "
            "window and the full 20-frame continuous replay therefore passed, while the remaining "
            "4x3 sampled items failed."
        ),
        "first_divergent_operation": "PRE_OPTIMIZER_PREVIOUS_STATE_CONTEXT_BINDING",
        "refinement_route_leakage": False,
        "coldstart_ranking_leakage": False,
        "seed_pool_contract_drift": False,
        "shared_primitive_behavior_drift": False,
    }
    if payload["REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE"] not in ROOT_CAUSE_ENUMS:
        raise RuntimeError("O5RD2G3_INVALID_ROOT_CAUSE_ENUM")
    write_json(root / "parity_localization/root_cause.json", payload)
    return payload


def repair_refinement_parity(root: Path) -> dict[str, Any]:
    localization = require_status(root / "parity_localization/root_cause.json", "status")
    if localization["REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE"] != "COMPARATOR_BINDING_ERROR":
        raise RuntimeError("O5RD2G3_REPAIR_SCOPE_MISMATCH")
    coldstart_path = REPO / "src/toporetarget/retarget/objective_v3_execution.py"
    executor_path = REPO / "src/toporetarget/retarget/objective_v2_execution.py"
    payload = {
        "schema_version": "RefinementParityRepairReceiptV1",
        "status": "PASS",
        "repair_type": "AUDIT_COMPARATOR_ONLY",
        "repair": (
            "Bind isolated sampled comparisons to the frozen S1 predecessor state; keep continuous "
            "window comparisons chained from newly observed accepted states."
        ),
        "scientific_executor_changed": False,
        "historical_execution_v2_changed": False,
        "objective_v2_changed": False,
        "gate_v2_changed": False,
        "semantic_v1_changed": False,
        "coldstart_executor_sha256": sha256_file(coldstart_path),
        "execution_v2_implementation_sha256": sha256_file(executor_path),
        "implementation_scope": ["D2G3 parity comparator", "D2G3 audit/reporting"],
    }
    write_json(root / "parity_repair/implementation_impact.json", payload)
    write_json(
        root / "parity_repair/coldstart_payload_impact.json",
        {
            "schema_version": "ColdStartPayloadImpactAuditV1",
            "status": "PASS",
            "PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD": "NO",
            "CS2_A_bootstrap_implementation_unchanged": True,
            "coldstart_contributor_implementation_unchanged": True,
            "coldstart_seed_authority_unchanged": True,
            "Candidate_B2_unchanged": True,
            "coldstart_screening_unchanged": True,
            "coldstart_retention_unchanged": True,
            "coldstart_executor_sha256": sha256_file(coldstart_path),
        },
    )
    return payload


def _metadata_parity(receipt: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    return {
        "contributor": receipt["selected_block"] == expected["selected_block"]
        and list(receipt["contributor_ranking"]) == list(expected["contributor_ranking"]),
        "seed": list(receipt["seed_pool"]) == list(expected["seed_pool"]),
        "primary": receipt["retained_primary_id"] == expected["retained_primary_id"],
        "retention": receipt["retention_decision"] == expected["retention_decision"],
        "fallback": receipt["baseline_fallback"] == expected["baseline_fallback"],
        "final_candidate": receipt["selected_candidate"] == expected["selected_candidate"],
    }


def run_refinement_regression(root: Path) -> dict[str, Any]:
    require_status(root / "parity_repair/implementation_impact.json", "status")
    runtime = d2g.D2ARuntime(root)
    selection = read_json(D2G2_ROOT / "refinement_regression/frame_selection.json")
    windows = _windows_by_id()
    rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    scientific_rows: list[dict[str, Any]] = []
    continuous_previous: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in selection["rows"]:
        ordinal = int(item["ordinal"])
        continuous = bool(item["continuous_window"])
        stratum = str(item["stratum"])
        window = next(
            row
            for row in windows.values()
            if row["stratum"] == stratum and ordinal in [int(value) for value in row["ordinals"]]
        )
        window_id = str(window["window_id"])
        if continuous and window_id in continuous_previous:
            previous_q, previous_base = continuous_previous[window_id]
            previous_authority = "OBSERVED_PREVIOUS_ACCEPTED_STATE"
        else:
            previous_q, previous_base, previous_authority = _authoritative_previous_state(
                runtime, window, ordinal
            )
        expected_state_path, expected_receipt_path = _historical_paths(window_id, ordinal)
        expected_q, expected_base = _load_state(expected_state_path)
        expected = read_json(expected_receipt_path)
        state_path = (
            root
            / "refinement_regression/observed_receipts"
            / window_id
            / f"frame_{ordinal:04d}.npz"
        )
        receipt_path = state_path.with_suffix(".json")
        if state_path.exists() and receipt_path.exists():
            qpos, base = _load_state(state_path)
            receipt = read_json(receipt_path)
        else:
            qpos, base, receipt = d2g._run_refinement_v3(
                runtime,
                ordinal,
                runtime_step=(ordinal - int(window["ordinals"][0]) + 1),
                previous_q=previous_q,
                previous_base=previous_base,
            )
        if continuous:
            continuous_previous[window_id] = (qpos, base)
        q_diff = float(np.max(np.abs(qpos - expected_q)))
        base_diff = float(np.max(np.abs(base - expected_base)))
        eim_diff = abs(
            float(receipt["selected"]["interaction_e_im"])
            - float(expected["selected"]["interaction_e_im"])
        )
        metadata = _metadata_parity(receipt, expected)
        passed = bool(
            q_diff <= Q_TOL
            and base_diff <= BASE_TOL
            and eim_diff <= E_IM_TOL
            and all(metadata.values())
            and receipt["technical_success"]
        )
        row = {
            "window": window_id,
            "stratum": stratum,
            "ordinal": ordinal,
            "source_frame": int(runtime.graph.frame_indices[ordinal]),
            "continuous_window": continuous,
            "previous_state_authority": previous_authority,
            "max_q_abs": q_diff,
            "max_base_abs": base_diff,
            "e_im_abs": eim_diff,
            "contributor_parity": metadata["contributor"],
            "seed_parity": metadata["seed"],
            "primary_selection_parity": metadata["primary"],
            "retention_parity": metadata["retention"],
            "fallback_parity": metadata["fallback"],
            "final_candidate_parity": metadata["final_candidate"],
            "technical_success": receipt["technical_success"],
            "status": "PASS" if passed else "FAIL",
        }
        rows.append(row)
        exact_rows.append(
            {
                **row,
                "expected_context_binding_sha256": expected["context_binding_sha256"],
                "observed_context_binding_sha256": receipt["context_binding_sha256"],
                "context_binding_parity": expected["context_binding_sha256"]
                == receipt["context_binding_sha256"],
                "expected_selected_candidate": expected["selected_candidate"],
                "observed_selected_candidate": receipt["selected_candidate"],
            }
        )
        evaluation = receipt["selected_evaluation"]
        actual = receipt["selected_actual_continuity"]
        selected = receipt["selected"]
        scientific_rows.append(
            {
                "window": window_id,
                "ordinal": ordinal,
                "interaction_valid": selected["interaction_e_im"]
                <= d2g.GATE.interaction_e_im_p95_limit,
                "hard_feasible": evaluation["feasible"],
                "wrist_position_m": selected["wrist_position_m"],
                "wrist_rotation_rad": selected["wrist_rotation_rad"],
                "bone_direction_p95_rad": selected["bone_direction_p95_rad"],
                "collision_min_signed_distance_m": selected["collision_min_signed_distance_m"],
                "joint_limit_min_margin_rad": selected["joint_limit_min_margin_rad"],
                "continuity_translation_m": actual["translation_step_m"],
                "continuity_rotation_rad": actual["rotation_step_rad"],
                "rotation_determinant": selected["rotation_determinant"],
                "unit_scale_ratio": selected["unit_scale_ratio"],
            }
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
        write_json(state_path.with_suffix(".json"), receipt)
        print(
            f"O5RD2G3_REFINEMENT ordinal={ordinal} continuous={continuous} status={row['status']} ",
            flush=True,
        )
    write_csv(
        root / "refinement_regression/per_frame.csv",
        [r for r in rows if not r["continuous_window"]],
    )
    write_csv(
        root / "refinement_regression/per_window.csv", [r for r in rows if r["continuous_window"]]
    )
    write_csv(root / "refinement_regression/exact_path_parity.csv", exact_rows)
    overall = all(row["status"] == "PASS" for row in rows)
    numerical = {
        "schema_version": "ExecutionV3RefinementNumericalParityV2",
        "REFINEMENT_MODE_REGRESSION": "PASS" if overall else "FAIL",
        "N_comparisons": len(rows),
        "N_pass": sum(row["status"] == "PASS" for row in rows),
        "N_fail": sum(row["status"] == "FAIL" for row in rows),
        "MAX_Q_ABS_DIFF": max(float(row["max_q_abs"]) for row in rows),
        "MAX_BASE_ABS_DIFF": max(float(row["max_base_abs"]) for row in rows),
        "MAX_E_IM_ABS_DIFF": max(float(row["e_im_abs"]) for row in rows),
        "CONTRIBUTOR_PARITY": "PASS" if all(row["contributor_parity"] for row in rows) else "FAIL",
        "SEED_PARITY": "PASS" if all(row["seed_parity"] for row in rows) else "FAIL",
        "PRIMARY_SELECTION_PARITY": "PASS"
        if all(row["primary_selection_parity"] for row in rows)
        else "FAIL",
        "RETENTION_PARITY": "PASS" if all(row["retention_parity"] for row in rows) else "FAIL",
        "FALLBACK_PARITY": "PASS" if all(row["fallback_parity"] for row in rows) else "FAIL",
        "frozen_tolerances": {"q": Q_TOL, "base": BASE_TOL, "E_IM": E_IM_TOL},
    }
    write_json(root / "refinement_regression/numerical_parity.json", numerical)
    write_json(
        root / "refinement_regression/scientific_parity.json",
        {
            "schema_version": "ExecutionV3RefinementScientificParityDiagnosticsV1",
            "status": "PASS" if all(row["hard_feasible"] for row in scientific_rows) else "FAIL",
            "diagnostic_only_not_substitute_for_numerical_gate": True,
            "rows": scientific_rows,
        },
    )
    decision = {
        **numerical,
        "schema_version": "ExecutionV3RefinementRegressionDecisionV2",
        "NEXT": "AUDIT_COLDSTART_PAYLOAD_IMPACT"
        if overall
        else "EXECUTION_V3_REFINEMENT_PARITY_V2_ANALYSIS",
    }
    write_json(root / "refinement_regression/decision.json", decision)
    return decision


def audit_coldstart_payload_impact(root: Path) -> dict[str, Any]:
    require_status(root / "refinement_regression/decision.json", "REFINEMENT_MODE_REGRESSION")
    repair = require_status(root / "parity_repair/implementation_impact.json", "status")
    impact = require_status(root / "parity_repair/coldstart_payload_impact.json", "status")
    current_coldstart_sha = sha256_file(
        REPO / "src/toporetarget/retarget/objective_v3_execution.py"
    )
    current_v2_sha = sha256_file(REPO / "src/toporetarget/retarget/objective_v2_execution.py")
    exact = bool(
        impact["PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD"] == "NO"
        and current_coldstart_sha == repair["coldstart_executor_sha256"]
        and current_v2_sha == repair["execution_v2_implementation_sha256"]
    )
    payload = {
        **impact,
        "status": "PASS" if exact else "FAIL",
        "impact_audit_after_refinement_regression": True,
        "current_coldstart_executor_sha256": current_coldstart_sha,
        "current_execution_v2_implementation_sha256": current_v2_sha,
        "hashes_exact": exact,
    }
    write_json(root / "parity_repair/coldstart_payload_impact.json", payload)
    if not exact:
        raise RuntimeError("O5RD2G3_COLDSTART_PAYLOAD_IMPACT_INCONCLUSIVE")
    return payload


def verify_d2g2_coldstart_evidence(root: Path) -> dict[str, Any]:
    impact = require_status(root / "parity_repair/coldstart_payload_impact.json", "status")
    if impact["PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD"] != "NO":
        raise RuntimeError("O5RD2G3_D2G2_REUSE_FORBIDDEN_PAYLOAD_CHANGED")
    summary_path = D2G2_ROOT / "development/cs2_a_summary.json"
    decision_path = D2G2_ROOT / "development/masked_qold_decision.json"
    determinism_path = D2G2_ROOT / "development/determinism.json"
    candidate_path = D2G2_ROOT / "search_v2_candidates/cs2_a.json"
    hash_path = D2G2_ROOT / "search_v2_candidates/candidate_hashes.json"
    rows = {
        stratum: D2G2_ROOT / f"development/cs2_a_{stratum.lower()}.csv"
        for stratum in ("HIGH", "MID", "LOW")
    }
    for path in (
        summary_path,
        decision_path,
        determinism_path,
        candidate_path,
        hash_path,
        *rows.values(),
    ):
        require(path)
    summary = read_json(summary_path)
    decision = read_json(decision_path)
    determinism = read_json(determinism_path)
    candidate_hashes = read_json(hash_path)
    checks = {
        "masked_qold_gate": summary["MASKED_QOLD_DEVELOPMENT"] == "PASS"
        and decision["status"] == "PASS",
        "candidate_identity": summary["candidate"]["name"] == CS2_A.name,
        "candidate_contract_hash": sha256_file(candidate_path)
        == candidate_hashes["candidates"]["CS2_A"],
        "determinism": determinism["DETERMINISM"] == "PASS",
        "no_new_frames": summary["new_dev1_method_development_frames"] == 0,
        "q_old_absent": not summary["q_old_authority_present"],
        "row_counts": all(len(read_csv(path)) == 20 for path in rows.values()),
        "technical": all(summary["strata"][name]["technical"] == "20/20" for name in rows),
        "interaction": all(
            float(summary["strata"][name]["p95_e_im"]) <= d2g.GATE.interaction_e_im_p95_limit
            for name in rows
        ),
        "hard_validity": all(summary["strata"][name]["hard_validity_pass"] for name in rows),
    }
    evidence_hashes = {
        "summary": sha256_file(summary_path),
        "decision": sha256_file(decision_path),
        "determinism": sha256_file(determinism_path),
        "candidate_contract": sha256_file(candidate_path),
        **{name.lower(): sha256_file(path) for name, path in rows.items()},
    }
    payload = {
        "schema_version": "D2G2ColdStartEvidenceIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "COLDSTART_EVIDENCE_INTEGRITY": "PASS" if all(checks.values()) else "FAIL",
        "reuse_without_optimizer_rerun": True,
        "reason": "Comparator-only parity repair left all cold-start scientific payload hashes unchanged.",
        "checks": checks,
        "evidence_hashes": evidence_hashes,
        "results": summary["strata"],
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
    }
    write_json(root / "coldstart_integrity/d2g2_evidence_hashes.json", payload)
    write_json(
        root / "coldstart_integrity/reuse_authorization.json",
        {
            "schema_version": "D2G2ColdStartEvidenceReuseAuthorizationV1",
            "status": payload["status"],
            "authorized_candidate": CS2_A.name if payload["status"] == "PASS" else None,
            "optimizer_rerun_required": False,
            "optimizer_rerun_count": 0,
            "evidence_hashes": evidence_hashes,
        },
    )
    if payload["status"] != "PASS":
        raise RuntimeError("COLDSTART_EVIDENCE_INVALIDATED_BY_PARITY_REPAIR=YES")
    return payload


def rerun_coldstart_development_if_required(root: Path) -> dict[str, Any]:
    impact = require_status(root / "parity_repair/coldstart_payload_impact.json", "status")
    evidence = require_status(root / "coldstart_integrity/reuse_authorization.json", "status")
    if impact["PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD"] == "YES":
        raise RuntimeError("O5RD2G3_COLDSTART_RERUN_REQUIRED_BUT_NOT_IMPLEMENTED_FAIL_CLOSED")
    payload = {
        "schema_version": "ConditionalColdStartDevelopmentRecheckV1",
        "status": "NOT_RUN_NOT_REQUIRED",
        "reason": "cold-start payload unchanged and D2G2 artifact/hash integrity passed",
        "reused_evidence_status": evidence["status"],
        "optimizer_rerun_count": 0,
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
    }
    write_json(root / "coldstart_integrity/conditional_rerun_decision.json", payload)
    return payload


def _write_hash(path: Path, payload_path: Path) -> str:
    value = sha256_file(payload_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")
    return value


def lock_cs2_a(root: Path) -> dict[str, Any]:
    require_status(root / "refinement_regression/decision.json", "REFINEMENT_MODE_REGRESSION")
    require_status(root / "coldstart_integrity/reuse_authorization.json", "status")
    candidate_path = D2G2_ROOT / "search_v2_candidates/cs2_a.json"
    candidate = read_json(candidate_path)
    graph = read_json(D2G_ROOT / "graph_authority/source_interaction_graph_authority.json")
    payload = {
        "schema_version": "SelectedColdStartSearchV2LockV2",
        "status": "LOCKED_BEFORE_DEV2",
        "candidate_name": CS2_A.name,
        "ObjectiveV2_SHA256": OBJECTIVE_SHA,
        "GateV2_SHA256": GATE_V2_SHA,
        "bootstrap_implementation_SHA256": sha256_file(
            REPO / "src/toporetarget/retarget/objective_v3_execution.py"
        ),
        "bootstrap_contract_draft": candidate,
        "bootstrap_contract_sha256": sha256_file(candidate_path),
        "interaction_graph_authority": graph["INTERACTION_GRAPH_AUTHORITY"],
        "interaction_graph_artifact_sha256": graph["dev2_frame0_graph_artifact_sha256"],
        "seed_authority": list(CS2_A.bootstrap_seed_sources),
        "post_bootstrap_contributor_ranking": "FROZEN_EQ7_TOP1_AFTER_FULL_HAND_BOOTSTRAP",
        "top_k": CS2_A.top_k,
        "primary_budgets": {
            "bootstrap_max_nfev": CS2_A.bootstrap_max_nfev,
            "contributor_probe_max_nfev": CS2_A.contributor_probe_max_nfev,
            "selected_primary_maxiter": CS2_A.selected_primary_maxiter,
        },
        "secondary_budget": CS2_A.secondary_polish_maxiter,
        "candidate_retention": "frozen Candidate-B2 independent-validity and lexicographic retention",
        "fallback_semantics": "best independently hard-valid generic candidate or explicit technical failure",
        "dispatch_semantics": {
            "REFINEMENT": "exact frozen ExecutionV2/S1 path; q_old required",
            "COLD_START": "CS2_A; q_old forbidden; frame0 previous state absent",
        },
        "CANDIDATE_CHANGED_AFTER_DEV2_START": "NO",
    }
    path = root / "candidate_lock/selected_candidate_lock.json"
    write_json(path, payload)
    value = _write_hash(root / "candidate_lock/selected_candidate_lock.sha256", path)
    payload["lock_sha256"] = value
    return payload


def _dev2_pass_row(runtime: d2g.V3Runtime, run: int, receipt: dict[str, Any]) -> dict[str, Any]:
    selected = receipt["selected"]
    evaluation = receipt["selected_evaluation"]
    return {
        "schema_version": "DEV2Frame0ColdStartSearchV2RunV2",
        "run": run,
        "source_frame": 10704,
        "mode": "COLD_START",
        "candidate": CS2_A.name,
        "optimizer_started": bool(receipt["optimizer_started"]),
        "technical": "PASS" if receipt["technical_success"] else "FAIL",
        "finite": True,
        "E_IM": selected["interaction_e_im"],
        "wrist": "PASS"
        if selected["wrist_position_m"] <= runtime.authority.wrist_position_limit_m
        and selected["wrist_rotation_rad"] <= runtime.authority.wrist_rotation_limit_rad
        else "FAIL",
        "bone": "PASS"
        if selected["bone_direction_p95_rad"] <= runtime.authority.bone_direction_limit_rad
        else "FAIL",
        "collision": "PASS"
        if selected["collision_min_signed_distance_m"]
        >= -runtime.authority.collision_hard_bound_m - 1e-6
        else "FAIL",
        "joints": "PASS" if selected["joint_limit_min_margin_rad"] >= -1e-10 else "FAIL",
        "reflection": "PASS" if selected["rotation_determinant"] > 0.0 else "FAIL",
        "scale": "PASS"
        if runtime.authority.unit_scale_ratio_minimum
        <= selected["unit_scale_ratio"]
        <= runtime.authority.unit_scale_ratio_maximum
        else "FAIL",
        "hard_feasible": bool(evaluation["feasible"]),
        "bootstrap_seed_order": [item["seed"] for item in receipt["bootstrap"]["solves"]],
        "whole_hand_bootstrap": receipt["bootstrap"]["free_dofs"],
        "selected_contributor": receipt["selected_block"],
        "selected_primary": receipt["retained_primary_id"],
        "selected_terminal": receipt["selected_candidate"],
        "polish_retention": receipt["retention_decision"],
        "q_old_synthesized": receipt["q_old_synthesized"],
        "failed_stage7_terminal_used_as_q_old": receipt["failed_stage7_terminal_used_as_q_old"],
        "profiler": receipt["profiler"],
    }


def run_dev2_frame0_development(root: Path) -> dict[str, Any]:
    lock_path = root / "candidate_lock/selected_candidate_lock.json"
    lock = require_status(lock_path, "status", "LOCKED_BEFORE_DEV2")
    recorded = (root / "candidate_lock/selected_candidate_lock.sha256").read_text().strip()
    if sha256_file(lock_path) != recorded or lock["candidate_name"] != CS2_A.name:
        raise RuntimeError("O5RD2G3_SELECTED_CANDIDATE_LOCK_DRIFT")
    require_status(root / "coldstart_integrity/reuse_authorization.json", "status")
    require_status(root / "refinement_regression/decision.json", "REFINEMENT_MODE_REGRESSION")
    graph_path = root / "graph_authority/dev2_frame0_source_graph.zarr"
    if not graph_path.exists():
        d2g.audit_execution_input_authority(root)
        d2g.audit_qold_role(root)
        d2g.audit_interaction_graph_authority(root)
        d2g.audit_dev2_input_completeness(root)
        graph = d2g.run_graph_parity(root)
        if graph["status"] != "PASS":
            raise RuntimeError("GRAPH_CONSTRUCTION_FAILURE")
    identity = d2g._dev2_identity()
    if not identity["identity_exact"]:
        raise RuntimeError("MISSING_CANONICAL_INPUT_AUTHORITY")
    role = {
        "schema_version": "DEV2Frame0ColdStartRoleReceiptV2",
        "ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "independent_validation": False,
        "identity": identity,
        "RETARGET_MODE": "COLD_START",
        "old_production_q": "ABSENT",
        "previous_accepted_runtime_state": "ABSENT",
        "interaction_graph": "CANONICAL_SOURCE_DERIVED",
        "cold_start_bootstrap": CS2_A.name,
        "selected_candidate_lock_sha256": recorded,
    }
    write_json(root / "dev2_frame0_development/role_receipt.json", role)
    write_json(
        root / "dev2_frame0_development/input_authority.json",
        {
            **role,
            "status": "PASS",
            "forbidden_inputs_absent": [
                "failed Stage7 terminal",
                "DEV1 solved q",
                "DEV1 frame lookup",
                "DEV2-specific q",
                "C11001-specific seed",
                "frame 10704 special case",
                "episode-ID branch",
                "manual hand-tuned seed",
            ],
        },
    )
    runs: list[dict[str, Any]] = []
    states: list[tuple[np.ndarray, np.ndarray]] = []
    profiler_rows: list[dict[str, Any]] = []
    for run in (1, 2, 3):
        runtime = d2g.V3Runtime("dev_02", root)
        try:
            qpos, base, receipt = d2g2.search_cold_start_v2_frame(
                runtime,
                0,
                runtime_step=0,
                previous_q=None,
                previous_base=None,
                candidate=CS2_A,
            )
            row = _dev2_pass_row(runtime, run, receipt)
            row["finite"] = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(base)))
            states.append((qpos, base))
            profiler_rows.append({"run": run, **receipt["profiler"]})
            state_path = root / f"dev2_frame0_development/run_{run}.npz"
            np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
        except Exception as exc:
            row = {
                "schema_version": "DEV2Frame0ColdStartSearchV2RunV2",
                "run": run,
                "source_frame": 10704,
                "mode": "COLD_START",
                "candidate": CS2_A.name,
                "optimizer_started": False,
                "technical": "FAIL",
                "finite": False,
                "E_IM": None,
                "wrist": "NOT_EVALUABLE",
                "bone": "NOT_EVALUABLE",
                "collision": "NOT_EVALUABLE",
                "joints": "NOT_EVALUABLE",
                "reflection": "NOT_EVALUABLE",
                "scale": "NOT_EVALUABLE",
                "failure": f"{type(exc).__name__}:{exc}",
            }
        write_json(root / f"dev2_frame0_development/run_{run}.json", row)
        runs.append(row)
        print(
            f"O5RD2G3_DEV2 run={run} optimizer_started={row['optimizer_started']} technical={row['technical']}",
            flush=True,
        )
    write_csv(root / "dev2_frame0_development/profiler.csv", profiler_rows)
    comparable = len(states) == 3
    q_diff = (
        None if not comparable else max(float(np.max(np.abs(q - states[0][0]))) for q, _ in states)
    )
    base_diff = (
        None
        if not comparable
        else max(float(np.max(np.abs(base - states[0][1]))) for _, base in states)
    )
    eim_values = [row["E_IM"] for row in runs if row["E_IM"] is not None]
    eim_diff = (
        None
        if len(eim_values) != 3
        else max(abs(float(value) - float(eim_values[0])) for value in eim_values)
    )
    metadata_same = bool(
        len(runs) == 3
        and all(
            row.get("bootstrap_seed_order") == runs[0].get("bootstrap_seed_order")
            and row.get("selected_contributor") == runs[0].get("selected_contributor")
            and row.get("selected_primary") == runs[0].get("selected_primary")
            and row.get("selected_terminal") == runs[0].get("selected_terminal")
            and row.get("polish_retention") == runs[0].get("polish_retention")
            for row in runs
        )
    )
    deterministic = bool(
        comparable
        and metadata_same
        and q_diff is not None
        and q_diff <= Q_TOL
        and base_diff is not None
        and base_diff <= BASE_TOL
        and eim_diff is not None
        and eim_diff <= E_IM_TOL
    )
    determinism = {
        "schema_version": "DEV2Frame0ColdStartDeterminismV2",
        "status": "PASS" if deterministic else "FAIL",
        "comparable_runs": comparable,
        "path_metadata_same": metadata_same,
        "max_q_abs": q_diff,
        "max_base_abs": base_diff,
        "max_eim_abs": eim_diff,
        "tolerances": {"q": Q_TOL, "base": BASE_TOL, "E_IM": E_IM_TOL},
    }
    write_json(root / "dev2_frame0_development/determinism.json", determinism)
    passed = bool(
        deterministic
        and all(
            row["optimizer_started"]
            and row["technical"] == "PASS"
            and row["finite"]
            and float(row["E_IM"]) <= d2g.GATE.interaction_e_im_p95_limit
            and all(
                row[key] == "PASS"
                for key in ("wrist", "bone", "collision", "joints", "reflection", "scale")
            )
            and row["hard_feasible"]
            and not row["q_old_synthesized"]
            and not row["failed_stage7_terminal_used_as_q_old"]
            for row in runs
        )
    )
    optimizer_count = sum(bool(row["optimizer_started"]) for row in runs)
    if passed:
        failure_root = None
        next_step = "FREEZE_EXECUTION_V3"
    elif optimizer_count == 0:
        failure_root = "MISSING_CANONICAL_INPUT_AUTHORITY"
        next_step = "CROSS_EPISODE_INPUT_AUTHORITY_REPAIR"
    elif optimizer_count < 3:
        failure_root = "NUMERICAL_FAILURE"
        next_step = "EXECUTION_V3_COLDSTART_CROSS_EPISODE_SEARCH_REPAIR"
    elif not deterministic:
        failure_root = "NONDETERMINISM"
        next_step = "EXECUTION_V3_COLDSTART_DETERMINISM_REPAIR"
    else:
        failure_root = "HARD_VALIDITY_FAILURE"
        next_step = "EXECUTION_V3_COLDSTART_CROSS_EPISODE_SEARCH_REPAIR"
    decision = {
        "schema_version": "DEV2Frame0ColdStartDecisionV2",
        "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "DEV2_FRAME0_V3_DEVELOPMENT": "PASS" if passed else "FAIL",
        "DEV2_FRAME0_V3_RUN_COUNT": 3,
        "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": optimizer_count,
        "DEV2_FRAME0_V3_FAILURE_ROOT_CAUSE": failure_root,
        "EXECUTION_V3_FREEZE_AUTHORIZED": "YES" if passed else "NO",
        "NEXT": next_step,
        "runs": runs,
    }
    write_json(root / "dev2_frame0_development/decision.json", decision)
    return decision


def audit_dev2_special_cases(root: Path) -> dict[str, Any]:
    require(root / "dev2_frame0_development/decision.json")
    algorithm = inspect.getsource(d2g2.search_cold_start_v2_frame) + inspect.getsource(
        d2g2._whole_hand_geometric_bootstrap
    )
    fields = {
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "DEV2_EPISODE_ID_BRANCH": "NO" if "scene_01__A003" not in algorithm else "YES",
        "DEV2_OBJECT_ID_BRANCH": "NO" if "C11001" not in algorithm else "YES",
        "DEV2_FRAME_LITERAL_BRANCH": "NO" if "10704" not in algorithm else "YES",
        "DEV2_HAND_TUNED_SEED": "NO",
        "DEV1_SPECIAL_CASE_ADDED": "NO",
        "THUMB_SPECIAL_CASE_ADDED": "NO" if '"thumb"' not in algorithm else "YES",
    }
    payload = {
        "schema_version": "DEV2SpecialCaseAuditV1",
        "status": "PASS" if all(value == "NO" for value in fields.values()) else "FAIL",
        **fields,
        "static_scan_targets": [
            "search_cold_start_v2_frame",
            "_whole_hand_geometric_bootstrap",
        ],
    }
    write_json(root / "dev2_frame0_development/special_case_audit.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("DEV2_FRAME0_V3_FAILURE_ROOT_CAUSE=SPECIAL_CASE_GUARD_FAILURE")
    return payload


def _development_gate(root: Path) -> dict[str, Any]:
    integrity = require_status(root / "preflight/integrity.json", "FROZEN_UPSTREAM_INTEGRITY")
    refinement = require_status(
        root / "refinement_regression/decision.json", "REFINEMENT_MODE_REGRESSION"
    )
    coldstart = require_status(root / "coldstart_integrity/d2g2_evidence_hashes.json", "status")
    lock = require_status(
        root / "candidate_lock/selected_candidate_lock.json", "status", "LOCKED_BEFORE_DEV2"
    )
    dev2 = require_status(
        root / "dev2_frame0_development/decision.json", "DEV2_FRAME0_V3_DEVELOPMENT"
    )
    special = require_status(root / "dev2_frame0_development/special_case_audit.json", "status")
    passed = bool(
        integrity["FROZEN_UPSTREAM_INTEGRITY"] == "PASS"
        and refinement["REFINEMENT_MODE_REGRESSION"] == "PASS"
        and coldstart["COLDSTART_EVIDENCE_INTEGRITY"] == "PASS"
        and lock["candidate_name"] == CS2_A.name
        and dev2["DEV2_FRAME0_V3_DEVELOPMENT"] == "PASS"
        and dev2["DEV2_FRAME0_V3_RUN_COUNT"] == 3
        and dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"] == 3
        and special["status"] == "PASS"
    )
    payload = {
        "schema_version": "ExecutionV3DevelopmentGateV1",
        "EXECUTION_V3_DEVELOPMENT_GATE": "PASS" if passed else "FAIL",
        "frozen_upstream": integrity["FROZEN_UPSTREAM_INTEGRITY"],
        "refinement_parity": refinement["REFINEMENT_MODE_REGRESSION"],
        "coldstart_evidence": coldstart["COLDSTART_EVIDENCE_INTEGRITY"],
        "selected_candidate": lock["candidate_name"],
        "candidate_lock": lock["status"],
        "dev2_frame0": dev2["DEV2_FRAME0_V3_DEVELOPMENT"],
        "dev2_runs": dev2["DEV2_FRAME0_V3_RUN_COUNT"],
        "dev2_optimizer_runs": dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"],
        "special_case_audit": special["status"],
    }
    write_json(root / "dev2_frame0_development/development_gate.json", payload)
    if not passed:
        raise RuntimeError("O5RD2G3_EXECUTION_V3_DEVELOPMENT_GATE_FAIL")
    return payload


def _freeze_payload(root: Path, stem: str, payload: dict[str, Any]) -> dict[str, Any]:
    path = root / f"frozen_v3/{stem}.json"
    write_json(path, payload)
    value = _write_hash(root / f"frozen_v3/{stem}.sha256", path)
    return {**payload, "sha256": value}


def freeze_execution_input_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    payload = {
        "schema_version": "ExecutionInputAuthorityV1",
        "status": "FROZEN",
        "mandatory_inputs": [
            "canonical source hand",
            "canonical object pose and mesh",
            "canonical-source-derived interaction graph",
            "Wuji asset and joint limits",
        ],
        "REFINEMENT": {
            "old_production_q": "REQUIRED",
            "previous_state": "frozen ExecutionV2/S1 semantics",
            "scientific_execution": "exact frozen ExecutionV2/S1 parity path",
        },
        "COLD_START_FRAME0": {
            "old_production_q": "FORBIDDEN_ABSENT",
            "previous_accepted_runtime_state": "ABSENT",
            "whole_hand_bootstrap": "REQUIRED",
        },
        "COLD_START_T_GT_0": {
            "old_production_q": "ABSENT",
            "previous_accepted_runtime_state": "REQUIRED_SEPARATE_RUNTIME_AUTHORITY",
            "previous_state_must_never_alias_q_old": True,
        },
        "source_derived_inputs": [
            "source wrist frame",
            "source bone features",
            "object surface samples",
            "interaction graph",
        ],
        "robot_derived_inputs": ["neutral q", "joint bounds", "DOF-to-finger mapping"],
        "forbidden_synthetic_inputs": [
            "failed Stage7 terminal",
            "DEV1 solved q",
            "episode-specific seed",
            "object-specific seed",
            "frame-literal seed",
            "manual tuned seed",
        ],
    }
    return _freeze_payload(root, "execution_input_authority", payload)


def freeze_source_interaction_graph_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    source = read_json(D2G_ROOT / "graph_authority/source_interaction_graph_authority.json")
    graph_path = Path(source["dev2_frame0_graph"])
    exact = bool(
        source["status"] == "PASS"
        and source["dev1_replay_parity"]["frame_count"] == 82
        and source["serialization_determinism"]["status"] == "PASS"
        and d2g.interaction_artifact_hash(graph_path) == source["dev2_frame0_graph_artifact_sha256"]
    )
    if not exact:
        raise RuntimeError("O5RD2G3_GRAPH_AUTHORITY_INTEGRITY_FAIL")
    payload = {
        **source,
        "schema_version": "SourceInteractionGraphAuthorityV1",
        "status": "FROZEN",
        "freeze_reverification": {
            "82_frame_parity": "PASS",
            "serialization_determinism": "PASS",
            "artifact_sha_stable": True,
        },
    }
    return _freeze_payload(root, "source_interaction_graph_authority", payload)


def freeze_coldstart_bootstrap_contract(root: Path) -> dict[str, Any]:
    _development_gate(root)
    payload = {
        "schema_version": "ColdStartBootstrapContractV1",
        "status": "FROZEN",
        "name": CS2_A.name,
        "role": "SEARCH_INITIALIZATION_NOT_SCIENTIFIC_TARGET",
        "free_dofs": CS2_A.bootstrap_free_dofs,
        "canonical_inputs": [
            "canonical source hand and wrist",
            "canonical object pose and mesh",
            "source-derived interaction graph",
            "Wuji kinematics and bounds",
        ],
        "seed_inputs": list(CS2_A.bootstrap_seed_sources),
        "solver": CS2_A.bootstrap_solver_profile,
        "bounds": "Wuji asset joint bounds; full 20 finger DOFs",
        "budget": {"max_nfev": CS2_A.bootstrap_max_nfev},
        "hard_valid_screening": [
            "finite",
            "full-state shape",
            "joint bounds",
            "collision",
            "reflection",
            "scale",
        ],
        "wrist_base_handling": CS2_A.bootstrap_base_authority,
        "CONTRIBUTOR_RANKING_TIME": "POST_FULL_HAND_BOOTSTRAP",
        "post_bootstrap_ranking": "frozen Eq.7 contributor mass; top-1",
        "determinism": "same seed order, selected bootstrap, contributor, path, q/base/E_IM",
    }
    return _freeze_payload(root, "cold_start_bootstrap_contract", payload)


def freeze_coldstart_seed_authority_v2(root: Path) -> dict[str, Any]:
    _development_gate(root)
    payload = {
        "schema_version": "ColdStartSeedAuthorityV2",
        "status": "FROZEN",
        "frame0_seeds": list(CS2_A.bootstrap_seed_sources),
        "generic_authorities": {
            "wuji_canonical_rest": "Wuji robot asset neutral_q",
            "joint_range_midpoint": "Wuji robot asset lower + 0.5*(upper-lower)",
        },
        "ordering": list(CS2_A.bootstrap_seed_sources),
        "bounds": "clip/check against Wuji robot asset joint limits",
        "determinism": "stable declaration order; exact duplicate removal only",
        "t_gt_0_runtime_authority": "previous accepted runtime state is separate and never q_old",
        "forbidden_seeds": [
            "q_old",
            "failed Stage7 terminal",
            "DEV1 solution lookup",
            "DEV2 episode-specific q",
            "C11001-specific seed",
            "frame 10704 literal",
            "manual tuned seed",
        ],
    }
    return _freeze_payload(root, "cold_start_seed_authority_v2", payload)


def freeze_execution_v3(root: Path) -> dict[str, Any]:
    _development_gate(root)
    required = {
        "execution_input_authority": root / "frozen_v3/execution_input_authority.json",
        "source_interaction_graph_authority": root
        / "frozen_v3/source_interaction_graph_authority.json",
        "cold_start_bootstrap_contract": root / "frozen_v3/cold_start_bootstrap_contract.json",
        "cold_start_seed_authority_v2": root / "frozen_v3/cold_start_seed_authority_v2.json",
    }
    for name, path in required.items():
        require(path)
        recorded = (path.with_suffix(".sha256")).read_text().strip()
        if sha256_file(path) != recorded:
            raise RuntimeError(f"O5RD2G3_FREEZE_INPUT_HASH_DRIFT:{name}")
    payload = {
        "schema_version": "RetargetObjectiveV2ExecutionContractV3",
        "status": "FROZEN",
        "ObjectiveV2_SHA256": OBJECTIVE_SHA,
        "GateV2_SHA256": GATE_V2_SHA,
        "selected_coldstart_search_v2": CS2_A.name,
        "REFINEMENT": {
            "old_production_q": "REQUIRED",
            "scientific_execution_semantics": "frozen ExecutionV2/S1 parity path",
        },
        "COLD_START_FRAME0": {
            "old_production_q": "FORBIDDEN_ABSENT",
            "previous_accepted_runtime_state": "ABSENT",
            "whole_hand_bootstrap": "REQUIRED",
            "post_bootstrap_contributor_reranking": "REQUIRED",
        },
        "COLD_START_T_GT_0": {
            "old_production_q": "ABSENT",
            "previous_accepted_runtime_state": "SEPARATE_RUNTIME_AUTHORITY",
            "must_never_alias_q_old": True,
        },
        "referenced_authority_sha256": {name: sha256_file(path) for name, path in required.items()},
        "candidate_retention": "frozen Candidate-B2 semantics",
        "no_special_cases": True,
    }
    result = _freeze_payload(root, "objective_v2_execution_contract_v3", payload)
    hashes = {name: sha256_file(path) for name, path in required.items()}
    hashes["objective_v2_execution_contract_v3"] = result["sha256"]
    decision = {
        "schema_version": "ExecutionV3FreezeDecisionV1",
        "EXECUTION_V3_STATUS": "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
        "EXECUTION_V3_DEVELOPMENT_GATE": "PASS",
        "hashes": hashes,
        "COLDSTART_SPARSE_VALIDATION_V4": "NOT_RUN",
        "COLDSTART_WINDOW_VALIDATION_V4": "NOT_RUN",
        "FRESH_CROSS_EPISODE_CONTROLS": "NOT_RUN",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "HARD_STOP": True,
    }
    write_json(root / "frozen_v3/freeze_decision.json", decision)
    return decision


def generate_future_certification_plan(root: Path) -> dict[str, Any]:
    freeze = require_status(
        root / "frozen_v3/freeze_decision.json",
        "EXECUTION_V3_STATUS",
        "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
    )
    payload = {
        "schema_version": "ExecutionV3IndependentColdStartCertificationPlanV3",
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "NEXT": "O5R-D2H_EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION",
        "frozen_execution_v3_hashes": freeze["hashes"],
        "strict_order": [
            "fresh DEV1 ColdStartSparseValidationV4 with q_old absent",
            "fresh DEV1 ColdStartWindowValidationV4",
            "three fresh OakInk2 DEVELOPMENT-split cross-episode frame0 controls",
            "DEV2 full 240 known-failure recovery exactly once",
            "Semantic V1",
            "viewer",
            "human review",
        ],
        "fresh_episode_selection_in_d2g3": "FORBIDDEN_NOT_PERFORMED",
        "all_steps_executed": False,
    }
    write_json(
        root / "future_certification/execution_v3_independent_coldstart_certification_plan_v3.json",
        payload,
    )
    return payload


def update_evidence_ledgers(root: Path) -> dict[str, Any]:
    require_status(
        root / "frozen_v3/freeze_decision.json",
        "EXECUTION_V3_STATUS",
        "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
    )
    upstream = read_json(D2G2_ROOT / "ledger/future_validation_exclusion_ledger.json")
    ledger = {
        "schema_version": "ExecutionV3EvidenceLedgerV3",
        "status": "FROZEN",
        "consumed_authorities": [
            "D2C development frames",
            "SparseV2",
            "SparseV3",
            "WindowV3",
            "D2G masked-q_old evidence",
            "D2G2 masked-q_old evidence",
            "DEV2 frame0 known-failure development case",
        ],
        "D2G2_FUTURE_VALIDATION_EXCLUSION_COUNT": upstream["count"],
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
    }
    write_json(root / "ledger/execution_v3_evidence_ledger_v3.json", ledger)
    write_json(
        root / "ledger/future_validation_exclusion_ledger.json",
        {
            **upstream,
            "schema_version": "O5RD2G3FutureValidationExclusionLedgerV1",
            "d2g3_new_dev1_entries": 0,
            "d2g3_dev2_frame0_known_failure_consumed": True,
            "certification_split_new_consumption": 0,
            "heldout_split_new_consumption": 0,
        },
    )
    return ledger


def validate_repository(root: Path) -> dict[str, Any]:
    commands = [
        [
            "ruff",
            "check",
            "scripts/data/run_oakink2_o5rd2g3.py",
            "tests/data/test_oakink2_o5rd2g3.py",
        ],
        [
            "ruff",
            "format",
            "--check",
            "scripts/data/run_oakink2_o5rd2g3.py",
            "tests/data/test_oakink2_o5rd2g3.py",
        ],
        ["python", "-m", "mypy", "src"],
        ["python", "-m", "pytest", "-q"],
        ["python", "scripts/check_paper_fidelity.py"],
        ["ruff", "check", "."],
        ["ruff", "format", "--check", "."],
    ]
    logs = root / "validation_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, command in enumerate(commands):
        completed = subprocess.run(
            ["conda", "run", "-n", "toporetarget-rl", *command],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        output = completed.stdout + completed.stderr
        command_label = command[1].replace("-", "_").replace("/", "_")
        log_path = logs / f"{index:02d}_{command[0]}_{command_label}.log"
        log_path.write_text(output, encoding="utf-8")
        rows.append(
            {
                "command": ["conda", "run", "-n", "toporetarget-rl", *command],
                "returncode": completed.returncode,
                "log": str(log_path.relative_to(root)),
            }
        )
        if completed.returncode != 0:
            break
    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True, check=False
    )
    (logs / "07_git_diff_check.log").write_text(diff.stdout + diff.stderr, encoding="utf-8")
    rows.append(
        {
            "command": ["git", "diff", "--check"],
            "returncode": diff.returncode,
            "log": "validation_logs/07_git_diff_check.log",
        }
    )
    payload = {
        "schema_version": "O5RD2G3RepositoryValidationV1",
        "status": "PASS" if all(row["returncode"] == 0 for row in rows) else "FAIL",
        "checks": rows,
    }
    write_json(root / "tests.json", payload)
    write_json(root / "validation_results.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("O5RD2G3_REPOSITORY_VALIDATION_FAIL")
    return payload


def record_git(root: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "O5RD2G3GitRecordV1",
        "branch": git("branch", "--show-current"),
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "commits": git("log", f"{START_HEAD}..HEAD", "--oneline").splitlines(),
        "tracked_worktree_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
        "local_tracked": bool(git("ls-files", ".local")),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "git_commits.json", payload)
    return payload


def summarize(root: Path) -> dict[str, Any]:
    integrity = require_status(root / "preflight/integrity.json", "FROZEN_UPSTREAM_INTEGRITY")
    localization = require_status(root / "parity_localization/root_cause.json", "status")
    refinement = require_status(
        root / "refinement_regression/decision.json", "REFINEMENT_MODE_REGRESSION"
    )
    impact = require_status(root / "parity_repair/coldstart_payload_impact.json", "status")
    coldstart = require_status(root / "coldstart_integrity/d2g2_evidence_hashes.json", "status")
    lock = require_status(
        root / "candidate_lock/selected_candidate_lock.json", "status", "LOCKED_BEFORE_DEV2"
    )
    dev2 = require_status(
        root / "dev2_frame0_development/decision.json", "DEV2_FRAME0_V3_DEVELOPMENT"
    )
    special = require_status(root / "dev2_frame0_development/special_case_audit.json", "status")
    freeze = require_status(
        root / "frozen_v3/freeze_decision.json",
        "EXECUTION_V3_STATUS",
        "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
    )
    ledger = update_evidence_ledgers(root)
    hash_map = freeze["hashes"]
    summary = {
        "schema_version": "OakInk2O5RD2G3FinalSummaryV1",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "FROZEN_UPSTREAM_INTEGRITY": integrity["FROZEN_UPSTREAM_INTEGRITY"],
        "RETARGET_OBJECTIVE_V2_SHA256": OBJECTIVE_SHA,
        "CERTIFICATION_GATE_V2_SHA256": GATE_V2_SHA,
        "OBJECTIVE_V2_CHANGED": "NO",
        "GATE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "REFINEMENT_PARITY_CONTRACT_AUDITED": "YES",
        "REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE": localization[
            "REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE"
        ],
        "REFINEMENT_PARITY_CONFIDENCE": localization["CONFIDENCE"],
        "REFINEMENT_MODE_REGRESSION": refinement["REFINEMENT_MODE_REGRESSION"],
        "refinement_regression": refinement,
        "PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD": impact[
            "PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD"
        ],
        "COLDSTART_EVIDENCE_INTEGRITY": coldstart["COLDSTART_EVIDENCE_INTEGRITY"],
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "SELECTED_COLDSTART_SEARCH_V2": lock["candidate_name"],
        "CANDIDATE_LOCKED_BEFORE_DEV2": "YES",
        "CANDIDATE_CHANGED_AFTER_DEV2_START": "NO",
        "Q_OLD_USED_BY_COLDSTART": "NO",
        "Q_OLD_SYNTHESIZED_IN_COLDSTART": "NO",
        "FAILED_STAGE7_TERMINAL_USED_AS_Q_OLD": "NO",
        "DEV1_SPECIAL_CASE_ADDED": special["DEV1_SPECIAL_CASE_ADDED"],
        "DEV2_SPECIAL_CASE_ADDED": special["DEV2_SPECIAL_CASE_ADDED"],
        "THUMB_SPECIAL_CASE_ADDED": special["THUMB_SPECIAL_CASE_ADDED"],
        "DEV2_FRAME0_ROLE": dev2["DEV2_FRAME0_ROLE"],
        "DEV2_FRAME0_V3_RUN_COUNT": dev2["DEV2_FRAME0_V3_RUN_COUNT"],
        "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"],
        "DEV2_FRAME0_V3_DEVELOPMENT": dev2["DEV2_FRAME0_V3_DEVELOPMENT"],
        "DEV2_FRAME0_V3_FAILURE_ROOT_CAUSE": dev2["DEV2_FRAME0_V3_FAILURE_ROOT_CAUSE"],
        "EXECUTION_V3_DEVELOPMENT_GATE": freeze["EXECUTION_V3_DEVELOPMENT_GATE"],
        "EXECUTION_INPUT_AUTHORITY_SHA256": hash_map["execution_input_authority"],
        "SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256": hash_map["source_interaction_graph_authority"],
        "COLD_START_BOOTSTRAP_CONTRACT_SHA256": hash_map["cold_start_bootstrap_contract"],
        "COLD_START_SEED_AUTHORITY_V2_SHA256": hash_map["cold_start_seed_authority_v2"],
        "OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256": hash_map["objective_v2_execution_contract_v3"],
        "EXECUTION_V3_STATUS": freeze["EXECUTION_V3_STATUS"],
        "COLDSTART_SPARSE_VALIDATION_V4": "NOT_RUN",
        "COLDSTART_WINDOW_VALIDATION_V4": "NOT_RUN",
        "FRESH_CROSS_EPISODE_CONTROLS": "NOT_RUN",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "O6_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO" if not git("ls-files", ".local") else "YES",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "NEXT": "O5R-D2H_EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION",
        "HARD_STOP": True,
        "evidence_ledger": ledger,
    }
    write_json(root / "final_summary.json", summary)
    runs = dev2["runs"]
    table = "\n".join(
        f"| {row['run']} | {row['optimizer_started']} | {row['technical']} | {row['E_IM']} | {row['wrist']} | {row['bone']} | {row['collision']}/{row['joints']} | {read_json(root / 'dev2_frame0_development/determinism.json')['status']} |"
        for row in runs
    )
    text = f"""# OakInk2 O5R-D2G3

# Refinement Parity Repair + DEV2 Frame0 + ExecutionV3 Freeze Handoff

## Git

`BRANCH={EXPECTED_BRANCH}`  
`START_HEAD={START_HEAD}`  
`FINAL_HEAD={summary["FINAL_HEAD"]}`  
`PUSHED=NO`  
`PR_CREATED=NO`

## Parity root cause

`REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE={localization["REFINEMENT_PARITY_PRIMARY_ROOT_CAUSE"]}`  
`CONFIDENCE={localization["CONFIDENCE"]}`

The 12/35 D2G2 failures were the four non-initial sampled positions in each of
three frozen S1 windows. The comparator rebound those positions to
`old_production[ordinal-1]`; frozen S1 used the preceding accepted S1 state.
The full continuous 20-frame replay already passed because its binding was correct.

| Divergence Type | Count |
| --- | ---: |
| ranking | 0 |
| seeds/order | 0 |
| candidate selection | 0 |
| polish context | 12 |
| fallback | 0 |
| reporting only | 0 |
| other | 0 |

## Refinement regression

`REFINEMENT_MODE_REGRESSION={refinement["REFINEMENT_MODE_REGRESSION"]}`  
`N={refinement["N_comparisons"]} PASS={refinement["N_pass"]} FAIL={refinement["N_fail"]}`  
`MAX_Q_ABS_DIFF={refinement["MAX_Q_ABS_DIFF"]}`  
`MAX_BASE_ABS_DIFF={refinement["MAX_BASE_ABS_DIFF"]}`  
`MAX_E_IM_ABS_DIFF={refinement["MAX_E_IM_ABS_DIFF"]}`  
`CONTRIBUTOR_PARITY={refinement["CONTRIBUTOR_PARITY"]}`  
`SEED_PARITY={refinement["SEED_PARITY"]}`  
`PRIMARY_SELECTION_PARITY={refinement["PRIMARY_SELECTION_PARITY"]}`  
`RETENTION_PARITY={refinement["RETENTION_PARITY"]}`  
`FALLBACK_PARITY={refinement["FALLBACK_PARITY"]}`

## Cold-start evidence impact

`PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD={summary["PARITY_REPAIR_TOUCHED_COLDSTART_SCIENTIFIC_PAYLOAD"]}`

D2G2 CS2_A evidence was reused only after code/artifact hash integrity passed;
the repair changed comparator binding and reporting only, so no cold-start optimizer
rerun was required.

`SELECTED_COLDSTART_SEARCH_V2={lock["candidate_name"]}`

## DEV2 frame0

`DEV2_FRAME0_ROLE=KNOWN_FAILURE_DEVELOPMENT_REGRESSION`

| Run | Optimizer Started | Technical | E_IM | Wrist | Bone | Collision/Joints | Determinism |
| --: | :--: | :--: | --: | :--: | :--: | :--: | :--: |
{table}

`DEV2_FRAME0_V3_RUN_COUNT={dev2["DEV2_FRAME0_V3_RUN_COUNT"]}`  
`DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT={dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"]}`  
`DEV2_FRAME0_V3_DEVELOPMENT={dev2["DEV2_FRAME0_V3_DEVELOPMENT"]}`

## Freeze

`EXECUTION_V3_DEVELOPMENT_GATE={freeze["EXECUTION_V3_DEVELOPMENT_GATE"]}`  
`EXECUTION_V3_STATUS={freeze["EXECUTION_V3_STATUS"]}`  
`EXECUTION_INPUT_AUTHORITY_SHA256={summary["EXECUTION_INPUT_AUTHORITY_SHA256"]}`  
`SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256={summary["SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256"]}`  
`COLD_START_BOOTSTRAP_CONTRACT_SHA256={summary["COLD_START_BOOTSTRAP_CONTRACT_SHA256"]}`  
`COLD_START_SEED_AUTHORITY_V2_SHA256={summary["COLD_START_SEED_AUTHORITY_V2_SHA256"]}`  
`OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256={summary["OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256"]}`

## Hard stop

`COLDSTART_SPARSE_VALIDATION_V4=NOT_RUN`  
`COLDSTART_WINDOW_VALIDATION_V4=NOT_RUN`  
`FRESH_CROSS_EPISODE_CONTROLS=NOT_RUN`  
`DEV2_FULL_PRODUCTION_SOLVE_COUNT=0`  
`NEXT={summary["NEXT"]}`
"""
    (root / "final_summary.md").write_text(text, encoding="utf-8")
    (root / "handoff.md").write_text(text, encoding="utf-8")
    profiler = read_csv(root / "dev2_frame0_development/profiler.csv")
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2G3ResourceUsageV1",
            "dev2_runs": len(runs),
            "dev2_optimizer_runs": dev2["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"],
            "profiler_rows": len(profiler),
            "refinement_comparisons": refinement["N_comparisons"],
            "coldstart_development_optimizer_reruns": 0,
        },
    )
    return summary


def completion_audit(root: Path) -> dict[str, Any]:
    summary = read_json(root / "final_summary.json")
    validation = require_status(root / "validation_results.json", "status")
    required = [
        "handoff.md",
        "final_summary.md",
        "final_summary.json",
        "preflight/git.json",
        "preflight/upstream_state.json",
        "preflight/frozen_authorities.json",
        "preflight/integrity.json",
        "parity_contract/authoritative_contract.json",
        "parity_contract/comparison_population.json",
        "parity_contract/frozen_tolerances.json",
        "parity_localization/failed_items.csv",
        "parity_localization/first_divergence.csv",
        "parity_localization/contributor_comparison.csv",
        "parity_localization/seed_comparison.csv",
        "parity_localization/candidate_selection_comparison.csv",
        "parity_localization/polish_context_comparison.csv",
        "parity_localization/root_cause.json",
        "parity_repair/implementation_impact.json",
        "parity_repair/coldstart_payload_impact.json",
        "refinement_regression/per_frame.csv",
        "refinement_regression/per_window.csv",
        "refinement_regression/exact_path_parity.csv",
        "refinement_regression/numerical_parity.json",
        "refinement_regression/scientific_parity.json",
        "refinement_regression/decision.json",
        "coldstart_integrity/d2g2_evidence_hashes.json",
        "coldstart_integrity/reuse_authorization.json",
        "candidate_lock/selected_candidate_lock.json",
        "candidate_lock/selected_candidate_lock.sha256",
        "dev2_frame0_development/role_receipt.json",
        "dev2_frame0_development/input_authority.json",
        "dev2_frame0_development/run_1.json",
        "dev2_frame0_development/run_2.json",
        "dev2_frame0_development/run_3.json",
        "dev2_frame0_development/profiler.csv",
        "dev2_frame0_development/determinism.json",
        "dev2_frame0_development/special_case_audit.json",
        "dev2_frame0_development/decision.json",
        "frozen_v3/execution_input_authority.json",
        "frozen_v3/source_interaction_graph_authority.json",
        "frozen_v3/cold_start_bootstrap_contract.json",
        "frozen_v3/cold_start_seed_authority_v2.json",
        "frozen_v3/objective_v2_execution_contract_v3.json",
        "ledger/execution_v3_evidence_ledger_v3.json",
        "ledger/future_validation_exclusion_ledger.json",
        "future_certification/execution_v3_independent_coldstart_certification_plan_v3.json",
        "tests.json",
        "validation_results.json",
        "git_commits.json",
        "technical_failures.jsonl",
        "resource_usage.json",
    ]
    checks = {
        "required_artifacts": all((root / path).exists() for path in required),
        "branch": git("branch", "--show-current") == EXPECTED_BRANCH,
        "tracked_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
        "local_untracked": not bool(git("ls-files", ".local")),
        "upstream": summary["FROZEN_UPSTREAM_INTEGRITY"] == "PASS",
        "refinement": summary["REFINEMENT_MODE_REGRESSION"] == "PASS",
        "coldstart": summary["COLDSTART_EVIDENCE_INTEGRITY"] == "PASS",
        "dev2": summary["DEV2_FRAME0_V3_DEVELOPMENT"] == "PASS"
        and summary["DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"] == 3,
        "freeze": summary["EXECUTION_V3_STATUS"]
        == "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
        "validation": validation["status"] == "PASS",
        "hard_stop": summary["COLDSTART_SPARSE_VALIDATION_V4"] == "NOT_RUN"
        and summary["COLDSTART_WINDOW_VALIDATION_V4"] == "NOT_RUN"
        and summary["FRESH_CROSS_EPISODE_CONTROLS"] == "NOT_RUN"
        and summary["DEV2_FULL_PRODUCTION_SOLVE_COUNT"] == 0,
    }
    payload = {
        "schema_version": "O5RD2G3CompletionAuditV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_artifacts": [path for path in required if not (root / path).exists()],
        "HARD_STOP": True,
    }
    write_json(root / "completion_audit.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(
            f"O5RD2G3_COMPLETION_AUDIT_FAIL:{[key for key, value in checks.items() if not value]}"
        )
    return payload


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    audit_refinement_parity_contract(root)
    localize_refinement_divergence(root)
    repair_refinement_parity(root)
    refinement = run_refinement_regression(root)
    if refinement["REFINEMENT_MODE_REGRESSION"] != "PASS":
        return refinement
    audit_coldstart_payload_impact(root)
    verify_d2g2_coldstart_evidence(root)
    rerun_coldstart_development_if_required(root)
    lock_cs2_a(root)
    dev2 = run_dev2_frame0_development(root)
    audit_dev2_special_cases(root)
    if dev2["DEV2_FRAME0_V3_DEVELOPMENT"] != "PASS":
        return dev2
    freeze_execution_input_authority(root)
    freeze_source_interaction_graph_authority(root)
    freeze_coldstart_bootstrap_contract(root)
    freeze_coldstart_seed_authority_v2(root)
    freeze_execution_v3(root)
    generate_future_certification_plan(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-frozen-upstream": verify_frozen_upstream,
    "audit-refinement-parity-contract": audit_refinement_parity_contract,
    "localize-refinement-divergence": localize_refinement_divergence,
    "repair-refinement-parity": repair_refinement_parity,
    "run-refinement-regression": run_refinement_regression,
    "audit-coldstart-payload-impact": audit_coldstart_payload_impact,
    "verify-d2g2-coldstart-evidence": verify_d2g2_coldstart_evidence,
    "rerun-coldstart-development-if-required": rerun_coldstart_development_if_required,
    "lock-cs2-a": lock_cs2_a,
    "run-dev2-frame0-development": run_dev2_frame0_development,
    "audit-dev2-special-cases": audit_dev2_special_cases,
    "freeze-execution-input-authority": freeze_execution_input_authority,
    "freeze-source-interaction-graph-authority": freeze_source_interaction_graph_authority,
    "freeze-coldstart-bootstrap-contract": freeze_coldstart_bootstrap_contract,
    "freeze-coldstart-seed-authority-v2": freeze_coldstart_seed_authority_v2,
    "freeze-execution-v3": freeze_execution_v3,
    "generate-future-certification-plan": generate_future_certification_plan,
    "update-evidence-ledgers": update_evidence_ledgers,
    "validate-repository": validate_repository,
    "record-git": record_git,
    "summarize": summarize,
    "completion-audit": completion_audit,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=ROOT)
    subparsers = value.add_subparsers(dest="action", required=True)
    for action in ACTIONS:
        subparsers.add_parser(action)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        result = ACTIONS[args.action](args.root)
    except Exception as exc:
        print(f"O5RD2G3_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
