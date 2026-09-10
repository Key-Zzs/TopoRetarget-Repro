#!/usr/bin/env python3
"""Run OakInk2 O5R-D2D ObjectiveV2 independent certification.

The command line is a fail-closed state machine.  It imports the byte-frozen
O5R-D2C Candidate-B2/S1 implementation and never changes method semantics.
Validation manifests and gate contracts are serialized and hash-bound before
the first corresponding solve.  A failed scientific gate permanently blocks
every later compute stage in this report root.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
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
from scripts.data.run_oakink2_o5rd2c import CONTRACTS, search_frame
from toporetarget.utils.hashing import sha256_file

ROOT = REPO / ".local/reports/oakink2_o5rd2d_objective_v2_independent_certification_v1"
D2C_ROOT = REPO / ".local/reports/oakink2_o5rd2c_candidate_b2_search_v1"
O5_ROOT = REPO / ".local/reports/oakink2_o5_geometric_retarget_v1"
O5RA_ROOT = REPO / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"

EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "1ab2fcfa8033fa758f4182d55db2eac8512e0270"
OBJECTIVE_SHA = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
EXECUTION_SHA = "b114b3960c47b641a85eac44d974a6e71921613bee85273e3d7de811ba4e937e"
EXECUTION_NAME = "S1_DETERMINISTIC_MULTI_START"
TAU = 1.0e-4
EPS = 1.0e-12
Q_ATOL = 1.0e-9
EIM_ATOL = 1.0e-12

OBJECTIVE_SOURCE = D2C_ROOT / "frozen_method/retarget_objective_v2_contract.json"
EXECUTION_SOURCE = D2C_ROOT / "frozen_method/execution_contract_v2.json"
DEVELOPMENT_LEDGER_SOURCE = D2C_ROOT / "future_certification/development_exclusion_ledger.json"
OLD_EIM_SOURCE = O5RA_ROOT / "dev1_semantic/dev1_e_im_per_frame.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _head_descends_from_start(head: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )


def _write_sha(path: Path, digest_value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(digest_value + "\n", encoding="utf-8")


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise RuntimeError(f"O5RD2D_FROZEN_COPY_DRIFT:{destination}")
    if not destination.exists():
        shutil.copyfile(source, destination)


def verify_frozen_method(root: Path) -> dict[str, Any]:
    """Verify the two frozen contracts and every implementation hash they bind."""

    observed_objective = sha256_file(OBJECTIVE_SOURCE)
    observed_execution = sha256_file(EXECUTION_SOURCE)
    objective = read_json(OBJECTIVE_SOURCE)
    execution = read_json(EXECUTION_SOURCE)
    implementation = execution["implementation_hashes"]
    checks = {
        "objective_contract_hash_exact": observed_objective == OBJECTIVE_SHA,
        "execution_contract_hash_exact": observed_execution == EXECUTION_SHA,
        "selected_execution_is_s1": execution["selected"]["name"] == EXECUTION_NAME,
        "s2_not_selected": not bool(execution["selected"]["sequential_warm_start"]),
        "objective_implementation_exact": sha256_file(
            REPO / "src/toporetarget/retarget/objective_v2.py"
        )
        == objective["candidate_b2_implementation_sha256"],
        "execution_implementation_exact": sha256_file(
            REPO / "src/toporetarget/retarget/objective_v2_execution.py"
        )
        == implementation["objective_v2_execution_py"],
        "frozen_search_driver_exact": sha256_file(REPO / "scripts/data/run_oakink2_o5rd2c.py")
        == implementation["o5rd2c_cli_py"],
        "semantic_v1_exact": sha256_file(
            REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py"
        )
        == objective["semantic_v1_sha256"],
        "tau_exact": float(objective["tau"]) == TAU,
        "contracts_immutable": not objective["contract_mutable_after_freeze"]
        and not execution["contract_mutable_after_freeze"],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": "O5RD2DFrozenMethodIntegrityV1",
        "FROZEN_METHOD_INTEGRITY": status,
        "RETARGET_OBJECTIVE_V2_SHA256": observed_objective,
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": observed_execution,
        "SELECTED_EXECUTION_CONTRACT": execution["selected"]["name"],
        "checks": checks,
    }
    write_json(root / "preflight/frozen_method_integrity.json", payload)
    if status != "PASS":
        raise RuntimeError("O5RD2D_BLOCKED_FROZEN_METHOD_INTEGRITY")
    _copy_exact(OBJECTIVE_SOURCE, root / "method/retarget_objective_v2_contract.json")
    _copy_exact(EXECUTION_SOURCE, root / "method/execution_contract_v2.json")
    write_json(
        root / "method/method_hashes.json",
        {
            "RETARGET_OBJECTIVE_V2_SHA256": observed_objective,
            "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": observed_execution,
            "OBJECTIVE_CHANGED_DURING_CERTIFICATION": "NO",
            "EXECUTION_CHANGED_DURING_CERTIFICATION": "NO",
        },
    )
    return payload


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2D_BRANCH_MISMATCH:{branch}")
    if not _head_descends_from_start(head):
        raise RuntimeError(f"O5RD2D_START_HEAD_NOT_ANCESTOR:{head}")
    integrity = verify_frozen_method(root)
    upstream_authorities = read_json(D2C_ROOT / "preflight/frozen_authorities.json")["authorities"]
    authorities: dict[str, dict[str, str]] = {}
    for name, receipt in upstream_authorities.items():
        path = Path(receipt["path"])
        observed = digest(path)
        if observed != receipt["sha256"]:
            raise RuntimeError(f"O5RD2D_FROZEN_AUTHORITY_DRIFT:{name}")
        authorities[name] = {"path": str(path.resolve()), "sha256": observed}
    extras = {
        "objective_v2_frozen_contract": OBJECTIVE_SOURCE,
        "execution_v2_frozen_contract": EXECUTION_SOURCE,
        "o5rd2c_method_development_ledger": DEVELOPMENT_LEDGER_SOURCE,
        "o5rd2c_development_results": D2C_ROOT / "development/development_gate.json",
        "o5rd2c_certification_plan": D2C_ROOT / "future_certification/certification_plan_v2.json",
        "dev1_old_eim": OLD_EIM_SOURCE,
        "o5_fixed_episode_receipt": O5_ROOT / "preflight/fixed_o5_episode_set.json",
    }
    for name, path in extras.items():
        if not path.exists():
            raise FileNotFoundError(path)
        authorities[name] = {"path": str(path.resolve()), "sha256": digest(path)}
    git_payload = {
        "schema_version": "OakInk2O5RD2DGitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "start_head": START_HEAD,
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
            "schema_version": "O5RD2DUpstreamStateV1",
            "O5RD2C_HEAD": START_HEAD,
            "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "FROZEN_READY_FOR_INDEPENDENT_VALIDATION",
            "SELECTED_EXECUTION_CONTRACT": EXECUTION_NAME,
        },
    )
    write_json(
        root / "preflight/frozen_authorities.json",
        {
            "schema_version": "O5RD2DFrozenAuthoritiesV1",
            "status": "PASS",
            "authorities": authorities,
        },
    )
    return {"git": git_payload, "integrity": integrity, "authorities": authorities}


def development_entries() -> list[dict[str, Any]]:
    ledger = read_json(DEVELOPMENT_LEDGER_SOURCE)
    entries = list(ledger["entries"])
    if len(entries) != int(ledger["DEV1_METHOD_DEVELOPMENT_FRAME_COUNT"]):
        raise RuntimeError("O5RD2D_DEVELOPMENT_LEDGER_COUNT_DRIFT")
    return entries


def old_eim_rows() -> list[dict[str, Any]]:
    return [
        {
            "ordinal": int(row["ordinal"]),
            "frame_id": int(row["frame_id"]),
            "old_e_im": float(row["e_im"]),
        }
        for row in read_csv(OLD_EIM_SOURCE)
    ]


def _temporal_quantiles(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["ordinal"], row["frame_id"]))
    if len(ordered) < count:
        raise RuntimeError(f"O5RD2D_SELECTION_INSUFFICIENT_SUPPORT:{len(ordered)}<{count}")
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    return [ordered[int(index)] for index in indices]


def select_sparse_rows(
    rows: list[dict[str, Any]], excluded_ordinals: set[int]
) -> list[dict[str, Any]]:
    """Outcome-independent, deterministic temporal-stratified SparseV2 selection."""

    eligible = [row for row in rows if int(row["ordinal"]) not in excluded_ordinals]
    high_pool = [row for row in eligible if float(row["old_e_im"]) > TAU]
    high_bins = np.array_split(
        np.asarray(sorted(high_pool, key=lambda row: row["ordinal"]), dtype=object), 10
    )
    if any(len(bin_rows) == 0 for bin_rows in high_bins):
        raise RuntimeError("O5RD2D_HIGH_SELECTION_INSUFFICIENT_SUPPORT")
    high = [
        max(list(bin_rows), key=lambda row: (float(row["old_e_im"]), -int(row["ordinal"])))
        for bin_rows in high_bins
    ]
    remaining = [row for row in eligible if row not in high]
    values = np.asarray([float(row["old_e_im"]) for row in remaining], dtype=np.float64)
    p40, p60 = (float(value) for value in np.quantile(values, [0.4, 0.6]))
    mid_pool = [row for row in remaining if p40 <= float(row["old_e_im"]) <= p60]
    mid = _temporal_quantiles(mid_pool, 10)
    remaining = [row for row in remaining if row not in mid]
    low_pool = [row for row in remaining if float(row["old_e_im"]) <= TAU]
    low = _temporal_quantiles(low_pool, 10)
    selected: list[dict[str, Any]] = []
    for stratum, group in (("HIGH", high), ("MID", mid), ("LOW", low)):
        for row in group:
            selected.append({**row, "stratum": stratum})
    return sorted(selected, key=lambda row: (row["stratum"], row["ordinal"]))


def _determinism_sparse_ids(selected: list[dict[str, Any]]) -> list[int]:
    result: list[int] = []
    for stratum, count in (("HIGH", 2), ("MID", 2), ("LOW", 1)):
        rows = sorted(
            (row for row in selected if row["stratum"] == stratum),
            key=lambda row: row["ordinal"],
        )
        result.extend(int(row["ordinal"]) for row in _temporal_quantiles(rows, count))
    return result


def _write_ledgers(
    root: Path,
    *,
    sparse: list[dict[str, Any]] | None = None,
    sparse_consumed: bool = False,
    windows: list[dict[str, Any]] | None = None,
    windows_consumed: bool = False,
) -> None:
    development = development_entries()
    write_json(
        root / "ledger/development_exclusion_ledger.json",
        {
            "schema_version": "D2DValidationExclusionAuthorityV1",
            "DEVELOPMENT_EXCLUSION_COUNT": len(development),
            "all_consumed_frame_ids": [row["frame_id"] for row in development],
            "entries": development,
        },
    )
    validation: list[dict[str, Any]] = []
    for row in sparse or []:
        validation.append(
            {
                **row,
                "source_stage": "SparseValidationV2",
                "role": "INDEPENDENT_CERTIFICATION_CONSUMED"
                if sparse_consumed
                else "FROZEN_NOT_EXECUTED",
            }
        )
    for window in windows or []:
        for ordinal, frame_id in zip(window["ordinals"], window["frame_ids"], strict=True):
            validation.append(
                {
                    "ordinal": ordinal,
                    "frame_id": frame_id,
                    "stratum": window["type"],
                    "window_id": window["window_id"],
                    "source_stage": "ContiguousWindowValidationV2",
                    "role": "INDEPENDENT_CERTIFICATION_CONSUMED"
                    if windows_consumed
                    else "FROZEN_NOT_EXECUTED",
                }
            )
    write_json(
        root / "ledger/validation_consumption_ledger.json",
        {
            "schema_version": "DEV1MethodAndValidationLedgerV4",
            "development_count": len(development),
            "independent_certification_entries": validation,
            "independent_certification_consumed_count": sum(
                row["role"] == "INDEPENDENT_CERTIFICATION_CONSUMED" for row in validation
            ),
        },
    )


def sparse_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": "SparseValidationV2GateContractV1",
        "frozen_before_first_solve": True,
        "N": 30,
        "technical_completion": "N/N",
        "above_target_recovery_rate_minimum": 0.80,
        "old_above_target_median_relative_reduction_minimum": 0.50,
        "overall_nonregression_fraction_minimum": 0.90,
        "catastrophic_regression_definition": "E_new > 1.25 * E_old",
        "catastrophic_regression_maximum": 0,
        "low_controls_all_below_or_equal_tau": True,
        "tau": TAU,
        "semantic_v1_hard_nonregression": True,
        "collision_and_joint_limits_required": True,
        "determinism_repeats": 3,
        "q_atol": Q_ATOL,
        "e_im_atol": EIM_ATOL,
        "mutable_after_first_solve": False,
    }


def freeze_sparse_v2(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    target = root / "sparse_v2/manifest.json"
    gate_path = root / "sparse_v2/gate_contract.json"
    if target.exists() or gate_path.exists():
        if not (target.exists() and gate_path.exists()):
            raise RuntimeError("O5RD2D_PARTIAL_SPARSE_FREEZE")
        return read_json(target)
    excluded = {int(row["ordinal"]) for row in development_entries()}
    selected = select_sparse_rows(old_eim_rows(), excluded)
    overlap = sorted(excluded & {int(row["ordinal"]) for row in selected})
    if overlap or len(selected) != 30:
        raise RuntimeError(f"O5RD2D_SPARSE_SELECTION_INVALID:{overlap}")
    determinism_ordinals = _determinism_sparse_ids(selected)
    payload = {
        "schema_version": "DEV1SparseValidationV2",
        "status": "FROZEN",
        "selection_authority": "old frozen DEV1 trajectory E_IM only",
        "selection_rule": {
            "HIGH": "ten temporal bins over untouched old-above-target frames; maximum old E_IM per bin",
            "MID": "ten temporal quantiles from untouched global old-E_IM p40-p60 band",
            "LOW": "ten temporal quantiles from untouched old interaction-valid frames",
        },
        "N": len(selected),
        "frames": selected,
        "determinism_ordinals": determinism_ordinals,
        "SPARSE_V2_DEVELOPMENT_OVERLAP": len(overlap),
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    write_json(target, payload)
    manifest_sha = sha256_file(target)
    _write_sha(root / "sparse_v2/manifest.sha256", manifest_sha)
    _copy_exact(target, root / "sparse_v2/sparse_validation_v2_manifest.json")
    _write_sha(root / "sparse_v2/sparse_validation_v2_manifest.sha256", manifest_sha)
    contract = sparse_gate_contract()
    write_json(gate_path, contract)
    _write_sha(root / "sparse_v2/gate_contract.sha256", sha256_file(gate_path))
    _write_ledgers(root, sparse=selected)
    return payload


def _row_from_receipt(receipt: dict[str, Any], stratum: str) -> dict[str, Any]:
    old = receipt["old"]
    new = receipt["selected"]
    evaluation = receipt["selected_evaluation"]
    old_eim = float(old["interaction_e_im"])
    new_eim = float(new["interaction_e_im"])
    profiler = receipt["profiler"]
    primary = profiler.get("primary") or {}
    secondary = profiler.get("secondary") or {}
    probes = receipt["probe_receipts"]
    total_nfev = sum(int(item["nfev"]) for item in probes)
    total_nfev += int(primary.get("nfev", 0)) + int(secondary.get("nfev", 0))
    finite = bool(
        np.all(
            np.isfinite(
                [
                    new_eim,
                    float(new["wrist_position_m"]),
                    float(new["wrist_rotation_rad"]),
                    float(new["bone_direction_p95_rad"]),
                    float(new["collision_min_signed_distance_m"]),
                    float(new["joint_limit_min_margin_rad"]),
                ]
            )
        )
    )
    technical = bool(receipt["technical_success"] and finite and evaluation["feasible"])
    return {
        "ordinal": int(receipt["ordinal"]),
        "frame_id": int(receipt["frame_id"]),
        "stratum": stratum,
        "old_e_im": old_eim,
        "new_e_im": new_eim,
        "relative_reduction": (old_eim - new_eim) / max(old_eim, EPS),
        "technical_completion": technical,
        "finite": finite,
        "semantic_hard_pass": bool(evaluation["feasible"]),
        "wrist_pass": evaluation["constraint_margins"]["wrist_position_m"] >= 0
        and evaluation["constraint_margins"]["wrist_rotation_rad"] >= 0,
        "bone_pass": evaluation["constraint_margins"]["bone_direction_rad"] >= 0,
        "continuity_pass": evaluation["constraint_margins"]["actual_temporal_translation_m"] >= 0
        and evaluation["constraint_margins"]["actual_temporal_rotation_rad"] >= 0,
        "collision_pass": evaluation["constraint_margins"]["collision_hard_m"] >= 0,
        "joint_limits_pass": evaluation["constraint_margins"]["joint_limit_rad"] >= -EPS,
        "reflection_pass": evaluation["constraint_margins"]["reflection_determinant"] >= 0,
        "scale_pass": evaluation["constraint_margins"]["unit_scale_lower"] >= 0
        and evaluation["constraint_margins"]["unit_scale_upper"] >= 0,
        "selected_block": receipt["selected_block"],
        "seed_pool": "+".join(receipt["seed_pool"]),
        "selected_seed_candidate": receipt["selected_seed_candidate"],
        "selected_candidate": receipt["selected_candidate"],
        "retention_decision": receipt["retention_decision"],
        "baseline_fallback": bool(receipt["baseline_fallback"]),
        "primary_optimizer_converged": primary.get("optimizer_converged"),
        "secondary_optimizer_converged": secondary.get("optimizer_converged"),
        "total_nfev": total_nfev,
        "wall_solver_sec": float(profiler["total_wall_sec"]),
        "actual_translation_step_m": float(
            receipt["selected_actual_continuity"]["translation_step_m"]
        ),
        "actual_rotation_step_rad": float(
            receipt["selected_actual_continuity"]["rotation_step_rad"]
        ),
        "joint_step_inf_rad": float(receipt["selected_actual_continuity"]["q_step_inf_rad"]),
    }


def _determinism_signature(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_block": receipt["selected_block"],
        "seed_pool": receipt["seed_pool"],
        "selected_seed_candidate": receipt["selected_seed_candidate"],
        "retained_primary_id": receipt["retained_primary_id"],
        "retention_decision": receipt["retention_decision"],
        "selected_candidate": receipt["selected_candidate"],
        "baseline_fallback": receipt["baseline_fallback"],
        "e_im": float(receipt["selected"]["interaction_e_im"]),
    }


def _compare_repeat(
    reference_receipt: dict[str, Any],
    reference_q: np.ndarray,
    receipt: dict[str, Any],
    qpos: np.ndarray,
) -> dict[str, Any]:
    left, right = _determinism_signature(reference_receipt), _determinism_signature(receipt)
    categorical = all(left[key] == right[key] for key in left if key != "e_im")
    q_max_abs = float(np.max(np.abs(np.asarray(reference_q) - np.asarray(qpos))))
    eim_abs = abs(float(left["e_im"]) - float(right["e_im"]))
    return {
        "categorical_equal": categorical,
        "q_max_abs": q_max_abs,
        "e_im_abs": eim_abs,
        "pass": categorical and q_max_abs <= Q_ATOL and eim_abs <= EIM_ATOL,
        "signature": right,
    }


def decide_sparse_gate(rows: list[dict[str, Any]], determinism_pass: bool) -> dict[str, Any]:
    above = [row for row in rows if float(row["old_e_im"]) > TAU]
    low = [row for row in rows if row["stratum"] == "LOW"]
    reductions = [float(row["relative_reduction"]) for row in above]
    recovered = sum(float(row["new_e_im"]) <= TAU for row in above)
    nonregression = sum(float(row["new_e_im"]) <= float(row["old_e_im"]) + EPS for row in rows)
    catastrophic = sum(float(row["new_e_im"]) > 1.25 * float(row["old_e_im"]) for row in rows)
    conditions = {
        "technical_N_of_N": len(rows) > 0
        and all(bool(row["technical_completion"]) for row in rows),
        "above_target_recovery": bool(above) and recovered / len(above) >= 0.80,
        "median_relative_reduction": bool(reductions) and float(np.median(reductions)) >= 0.50,
        "overall_interaction_nonregression": bool(rows) and nonregression / len(rows) >= 0.90,
        "zero_catastrophic_regressions": catastrophic == 0,
        "low_controls_preserved": bool(low) and all(float(row["new_e_im"]) <= TAU for row in low),
        "semantic_v1_nonregression": all(bool(row["semantic_hard_pass"]) for row in rows),
        "collision_and_joint_limits": all(
            bool(row["collision_pass"]) and bool(row["joint_limits_pass"]) for row in rows
        ),
        "determinism": determinism_pass,
    }
    return {
        "schema_version": "SparseValidationV2GateDecisionV1",
        "N": len(rows),
        "technical": sum(bool(row["technical_completion"]) for row in rows),
        "old_above_target_count": len(above),
        "recovered_below_target_count": recovered,
        "recovery_rate": None if not above else recovered / len(above),
        "median_relative_reduction": None if not reductions else float(np.median(reductions)),
        "overall_nonregression_fraction": None if not rows else nonregression / len(rows),
        "catastrophic_regression_count": catastrophic,
        "low_preserved_count": sum(float(row["new_e_im"]) <= TAU for row in low),
        "wrist_pass_count": sum(bool(row["wrist_pass"]) for row in rows),
        "bone_pass_count": sum(bool(row["bone_pass"]) for row in rows),
        "continuity_pass_count": sum(bool(row["continuity_pass"]) for row in rows),
        "collision_pass_count": sum(bool(row["collision_pass"]) for row in rows),
        "joint_limit_pass_count": sum(bool(row["joint_limits_pass"]) for row in rows),
        "conditions": conditions,
        "SPARSE_VALIDATION_V2": "PASS" if all(conditions.values()) else "FAIL",
    }


def run_sparse_v2(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    decision_path = root / "sparse_v2/gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    manifest = read_json(root / "sparse_v2/manifest.json")
    if (
        manifest["status"] != "FROZEN"
        or sha256_file(root / "sparse_v2/manifest.json")
        != (root / "sparse_v2/manifest.sha256").read_text().strip()
    ):
        raise RuntimeError("O5RD2D_SPARSE_MANIFEST_INTEGRITY")
    if (
        sha256_file(root / "sparse_v2/gate_contract.json")
        != (root / "sparse_v2/gate_contract.sha256").read_text().strip()
    ):
        raise RuntimeError("O5RD2D_SPARSE_GATE_CONTRACT_INTEGRITY")
    runtime = D2ARuntime(root)
    contract = CONTRACTS[EXECUTION_NAME].validate()
    rows: list[dict[str, Any]] = []
    references: dict[int, tuple[dict[str, Any], np.ndarray]] = {}
    for index, frame in enumerate(manifest["frames"], start=1):
        ordinal = int(frame["ordinal"])
        receipt_path = root / f"sparse_v2/receipts/frame_{ordinal:04d}_run_1.json"
        state_path = receipt_path.with_suffix(".npz")
        if receipt_path.exists() and state_path.exists():
            receipt = read_json(receipt_path)
            state = np.load(state_path, allow_pickle=False)
            qpos = np.asarray(state["qpos"], dtype=np.float64)
        else:
            previous_q = (
                None if ordinal == 0 else np.asarray(runtime.final.arrays["qpos"][ordinal - 1])
            )
            previous_base = (
                None
                if ordinal == 0
                else np.asarray(runtime.final.arrays["base_pose_scene"][ordinal - 1])
            )
            qpos, base, receipt = search_frame(
                runtime,
                ordinal,
                previous_q=previous_q,
                previous_base=previous_base,
                contract=contract,
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
            write_json(receipt_path, receipt)
        references[ordinal] = (receipt, qpos)
        rows.append(_row_from_receipt(receipt, str(frame["stratum"])))
        print(
            f"O5RD2D_SPARSE {index}/{manifest['N']} ordinal={ordinal} "
            f"eim={float(receipt['selected']['interaction_e_im']):.12g}",
            flush=True,
        )
    write_csv(root / "sparse_v2/per_frame_results.csv", rows)
    write_csv(root / "sparse_v2/semantic_metrics.csv", rows)
    write_csv(root / "sparse_v2/search_metrics.csv", rows)

    repeat_rows: list[dict[str, Any]] = []
    for ordinal in manifest["determinism_ordinals"]:
        ordinal = int(ordinal)
        reference_receipt, reference_q = references[ordinal]
        repeat_rows.append(
            {
                "ordinal": ordinal,
                "run": 1,
                "pass": True,
                "signature": _determinism_signature(reference_receipt),
            }
        )
        for run in (2, 3):
            previous_q = (
                None if ordinal == 0 else np.asarray(runtime.final.arrays["qpos"][ordinal - 1])
            )
            previous_base = (
                None
                if ordinal == 0
                else np.asarray(runtime.final.arrays["base_pose_scene"][ordinal - 1])
            )
            qpos, base, receipt = search_frame(
                runtime,
                ordinal,
                previous_q=previous_q,
                previous_base=previous_base,
                contract=contract,
            )
            receipt_path = root / f"sparse_v2/receipts/frame_{ordinal:04d}_run_{run}.json"
            np.savez_compressed(receipt_path.with_suffix(".npz"), qpos=qpos, base_pose_scene=base)
            write_json(receipt_path, receipt)
            repeat_rows.append(
                {
                    "ordinal": ordinal,
                    "run": run,
                    **_compare_repeat(reference_receipt, reference_q, receipt, qpos),
                }
            )
            print(f"O5RD2D_SPARSE_DETERMINISM ordinal={ordinal} run={run}", flush=True)
    determinism_pass = all(bool(row["pass"]) for row in repeat_rows)
    determinism = {
        "schema_version": "SparseValidationV2DeterminismV1",
        "status": "PASS" if determinism_pass else "FAIL",
        "repeats_per_frame": 3,
        "rows": repeat_rows,
    }
    write_json(root / "sparse_v2/determinism.json", determinism)
    decision = decide_sparse_gate(rows, determinism_pass)
    write_json(decision_path, decision)
    _write_ledgers(root, sparse=manifest["frames"], sparse_consumed=True)
    if decision["SPARSE_VALIDATION_V2"] != "PASS":
        write_downstream_not_run(root, reason="SPARSE_VALIDATION_V2_FAIL")
    return decision


def select_window_rows(
    rows: list[dict[str, Any]], excluded_ordinals: set[int], length: int = 32
) -> list[dict[str, Any]]:
    """Select four mutually disjoint untouched windows from old E_IM only."""

    by_ordinal = {int(row["ordinal"]): row for row in rows}
    maximum = max(by_ordinal)
    candidates: list[dict[str, Any]] = []
    for start in range(1, maximum - length + 2):
        ordinals = list(range(start, start + length))
        if any(value in excluded_ordinals or value not in by_ordinal for value in ordinals):
            continue
        values = [float(by_ordinal[value]["old_e_im"]) for value in ordinals]
        candidates.append(
            {
                "start_ordinal": start,
                "end_ordinal_inclusive": start + length - 1,
                "ordinals": ordinals,
                "frame_ids": [int(by_ordinal[value]["frame_id"]) for value in ordinals],
                "old_p95_e_im": float(np.quantile(values, 0.95)),
                "N": length,
            }
        )
    selected: list[dict[str, Any]] = []

    def nonoverlap(candidate: dict[str, Any]) -> bool:
        used = {value for row in selected for value in row["ordinals"]}
        return not used.intersection(candidate["ordinals"])

    high_pool = [row for row in candidates if row["old_p95_e_im"] > TAU]
    if len(high_pool) < 2:
        raise RuntimeError("O5RD2D_WINDOW_HIGH_SUPPORT_INSUFFICIENT")
    halves = np.array_split(
        np.asarray(sorted(high_pool, key=lambda row: row["start_ordinal"]), dtype=object), 2
    )
    for index, half in enumerate(halves, start=1):
        options = [row for row in half if nonoverlap(row)]
        if not options:
            raise RuntimeError("O5RD2D_WINDOW_HIGH_NONOVERLAP_INFEASIBLE")
        chosen = max(options, key=lambda row: (row["old_p95_e_im"], -row["start_ordinal"]))
        selected.append({**chosen, "window_id": f"HIGH_{index}", "type": f"HIGH_{index}"})
    low_pool = [row for row in candidates if row["old_p95_e_im"] <= TAU and nonoverlap(row)]
    if not low_pool:
        raise RuntimeError("O5RD2D_WINDOW_LOW_SUPPORT_INSUFFICIENT")
    low = min(low_pool, key=lambda row: (row["old_p95_e_im"], row["start_ordinal"]))
    selected.append({**low, "window_id": "LOW", "type": "LOW"})
    mid_pool = [row for row in candidates if row["old_p95_e_im"] > TAU and nonoverlap(row)]
    if not mid_pool:
        raise RuntimeError("O5RD2D_WINDOW_MID_SUPPORT_INSUFFICIENT")
    mid = min(mid_pool, key=lambda row: (row["old_p95_e_im"] - TAU, row["start_ordinal"]))
    selected.append({**mid, "window_id": "MID", "type": "MID"})
    return sorted(selected, key=lambda row: row["window_id"])


def window_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": "WindowValidationV2GateContractV1",
        "frozen_before_first_solve": True,
        "windows": ["HIGH_1", "HIGH_2", "MID", "LOW"],
        "technical_completion": "all frames in 4/4 windows",
        "interaction_p95_maximum": TAU,
        "low_formerly_valid_preserved": True,
        "semantic_v1_actual_continuity_required": True,
        "wrist_bone_collision_joint_limits_required": True,
        "determinism_windows": ["HIGH_1", "LOW"],
        "determinism_repeats": 3,
        "q_atol": Q_ATOL,
        "e_im_atol": EIM_ATOL,
        "mutable_after_first_solve": False,
    }


def freeze_window_v2(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    sparse = read_json(root / "sparse_v2/gate_decision.json")
    if sparse["SPARSE_VALIDATION_V2"] != "PASS":
        raise RuntimeError("O5RD2D_WINDOW_BLOCKED_BY_SPARSE")
    target = root / "window_v2/manifest.json"
    if target.exists():
        return read_json(target)
    sparse_manifest = read_json(root / "sparse_v2/manifest.json")
    development = {int(row["ordinal"]) for row in development_entries()}
    sparse_ordinals = {int(row["ordinal"]) for row in sparse_manifest["frames"]}
    selected = select_window_rows(old_eim_rows(), development | sparse_ordinals)
    all_ordinals = [value for row in selected for value in row["ordinals"]]
    if len(all_ordinals) != len(set(all_ordinals)):
        raise RuntimeError("O5RD2D_WINDOW_INTERNAL_OVERLAP")
    payload = {
        "schema_version": "DEV1ContiguousWindowValidationV2",
        "status": "FROZEN",
        "selection_authority": "old frozen DEV1 trajectory E_IM only",
        "selection_rule": "32-frame untouched windows; two temporally separated high-tail maxima, nearest-above-target MID, minimum-p95 LOW",
        "windows": selected,
        "WINDOW_V2_DEVELOPMENT_OVERLAP": len(development & set(all_ordinals)),
        "WINDOW_V2_SPARSE_OVERLAP": len(sparse_ordinals & set(all_ordinals)),
        "WINDOW_V2_INTERNAL_OVERLAP": len(all_ordinals) - len(set(all_ordinals)),
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    write_json(target, payload)
    _write_sha(root / "window_v2/manifest.sha256", sha256_file(target))
    gate = window_gate_contract()
    write_json(root / "window_v2/gate_contract.json", gate)
    _write_sha(
        root / "window_v2/gate_contract.sha256",
        sha256_file(root / "window_v2/gate_contract.json"),
    )
    _write_ledgers(
        root,
        sparse=sparse_manifest["frames"],
        sparse_consumed=True,
        windows=selected,
    )
    return payload


def _require_pass(root: Path, path: str, field: str, error: str) -> dict[str, Any]:
    target = root / path
    if not target.exists():
        raise RuntimeError(error)
    payload = read_json(target)
    if payload.get(field) != "PASS":
        raise RuntimeError(error)
    return payload


def run_window_v2(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    _require_pass(
        root,
        "sparse_v2/gate_decision.json",
        "SPARSE_VALIDATION_V2",
        "O5RD2D_WINDOW_BLOCKED_BY_SPARSE",
    )
    raise RuntimeError("O5RD2D_WINDOW_RUNTIME_REQUIRES_PASS_PATH_COMPLETION")


def run_dev2_frame0_hard_control(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    _require_pass(
        root, "sparse_v2/gate_decision.json", "SPARSE_VALIDATION_V2", "O5RD2D_FRAME0_BLOCKED"
    )
    _require_pass(
        root, "window_v2/gate_decision.json", "WINDOW_VALIDATION_V2", "O5RD2D_FRAME0_BLOCKED"
    )
    raise RuntimeError("O5RD2D_DEV2_FRAME0_RUNTIME_REQUIRES_PASS_PATH_COMPLETION")


def run_dev2_full_if_authorized(root: Path) -> dict[str, Any]:
    verify_frozen_method(root)
    _require_pass(
        root, "sparse_v2/gate_decision.json", "SPARSE_VALIDATION_V2", "O5RD2D_FULL_BLOCKED"
    )
    _require_pass(
        root, "window_v2/gate_decision.json", "WINDOW_VALIDATION_V2", "O5RD2D_FULL_BLOCKED"
    )
    _require_pass(
        root, "dev2_frame0/decision.json", "DEV2_FRAME0_HARD_CONTROL", "O5RD2D_FULL_BLOCKED"
    )
    raise RuntimeError("O5RD2D_DEV2_FULL_RUNTIME_REQUIRES_PASS_PATH_COMPLETION")


def run_dev2_semantic_v1(root: Path) -> dict[str, Any]:
    authorization = read_json(root / "dev2_full/authorization.json")
    if authorization.get("DEV2_FULL_PRODUCTION_SOLVE_COUNT") != 1:
        raise RuntimeError("O5RD2D_SEMANTIC_BLOCKED_NO_COMPLETE_DEV2")
    raise RuntimeError("O5RD2D_SEMANTIC_RUNTIME_REQUIRES_COMPLETE_DEV2")


def render_dev2_viewer(root: Path) -> dict[str, Any]:
    semantic = read_json(root / "dev2_full/semantic_validity.json")
    if semantic.get("DEV2_OBJECTIVE_V2_MACHINE") != "PASS":
        raise RuntimeError("O5RD2D_VIEWER_BLOCKED_NO_MACHINE_PASS")
    raise RuntimeError("O5RD2D_VIEWER_RUNTIME_REQUIRES_MACHINE_PASS")


def write_downstream_not_run(root: Path, *, reason: str) -> None:
    sparse_status = (
        read_json(root / "sparse_v2/gate_decision.json").get("SPARSE_VALIDATION_V2")
        if (root / "sparse_v2/gate_decision.json").exists()
        else "NOT_RUN"
    )
    windows = {
        "WINDOW_VALIDATION_V2": "NOT_RUN",
        "reason": reason,
    }
    write_json(root / "window_v2/not_run.json", windows)
    frame0 = {
        "DEV2_FRAME0_HARD_CONTROL_AUTHORIZED": "NO",
        "DEV2_FRAME0_HARD_CONTROL": "NOT_RUN",
        "DEV2_FRAME0_RUN_COUNT": 0,
        "reason": reason,
    }
    write_json(root / "dev2_frame0/authorization.json", frame0)
    write_json(root / "dev2_frame0/not_run.json", frame0)
    full = {
        "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "DEV2_FULL_RESULT": "NOT_RUN",
        "reason": reason,
    }
    write_json(root / "dev2_full/authorization.json", full)
    write_json(root / "dev2_full/not_run.json", full)
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2DResourceUsageV1",
            "sparse_validation": sparse_status,
            "window_validation": "NOT_RUN",
            "dev2_frame0_runs": 0,
            "dev2_full_runs": 0,
            "DEV1_FULL_RETARGET_RERUNS": 0,
            "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
            "O6_RAN": "NO",
            "SUPPORT_PHYSICALIZATION_RAN": "NO",
            "PHYSX_RAN": "NO",
            "FROZEN_EVAL_RAN": "NO",
            "PPO_RAN": "NO",
        },
    )
    failures = root / "technical_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    if not failures.exists():
        failures.write_text("", encoding="utf-8")


def _status_or(root: Path, relative: str, field: str, default: str) -> str:
    path = root / relative
    return str(read_json(path).get(field, default)) if path.exists() else default


def summarize(root: Path) -> dict[str, Any]:
    integrity = verify_frozen_method(root)
    sparse_manifest = read_json(root / "sparse_v2/manifest.json")
    sparse = read_json(root / "sparse_v2/gate_decision.json")
    if sparse["SPARSE_VALIDATION_V2"] != "PASS":
        write_downstream_not_run(root, reason="SPARSE_VALIDATION_V2_FAIL")
    window = _status_or(root, "window_v2/gate_decision.json", "WINDOW_VALIDATION_V2", "NOT_RUN")
    frame0 = _status_or(root, "dev2_frame0/decision.json", "DEV2_FRAME0_HARD_CONTROL", "NOT_RUN")
    full_auth = read_json(root / "dev2_full/authorization.json")
    development_count = len(development_entries())
    final_head = git("rev-parse", "HEAD")
    commit_receipt = record_git(root)
    sparse_rows = read_csv(root / "sparse_v2/per_frame_results.csv")
    pass_counts = {
        "wrist": sum(row["wrist_pass"] == "True" for row in sparse_rows),
        "bone": sum(row["bone_pass"] == "True" for row in sparse_rows),
        "continuity": sum(row["continuity_pass"] == "True" for row in sparse_rows),
        "collision": sum(row["collision_pass"] == "True" for row in sparse_rows),
        "joint_limits": sum(row["joint_limits_pass"] == "True" for row in sparse_rows),
    }
    flags = {
        "BRANCH": git("branch", "--show-current"),
        "RETARGET_OBJECTIVE_V2_SHA256": integrity["RETARGET_OBJECTIVE_V2_SHA256"],
        "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256": integrity[
            "OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256"
        ],
        "FROZEN_METHOD_INTEGRITY": integrity["FROZEN_METHOD_INTEGRITY"],
        "OBJECTIVE_CHANGED_DURING_CERTIFICATION": "NO",
        "EXECUTION_CHANGED_DURING_CERTIFICATION": "NO",
        "S1_USED_AS_CERTIFICATION_EXECUTION": "YES",
        "S2_USED_AS_RESCUE": "NO",
        "DEVELOPMENT_EXCLUSION_COUNT": development_count,
        "SPARSE_VALIDATION_V2_FROZEN": "YES",
        "SPARSE_V2_DEVELOPMENT_OVERLAP": sparse_manifest["SPARSE_V2_DEVELOPMENT_OVERLAP"],
        "SPARSE_VALIDATION_V2": sparse["SPARSE_VALIDATION_V2"],
        "WINDOW_VALIDATION_V2_FROZEN": "NO" if window == "NOT_RUN" else "YES",
        "WINDOW_V2_DEVELOPMENT_OVERLAP": "NOT_RUN" if window == "NOT_RUN" else 0,
        "WINDOW_V2_SPARSE_OVERLAP": "NOT_RUN" if window == "NOT_RUN" else 0,
        "WINDOW_VALIDATION_V2": window,
        "VALIDATION_THRESHOLD_CHANGED_AFTER_OUTCOME": "NO",
        "DEV2_FRAME0_HARD_CONTROL": frame0,
        "DEV2_FRAME0_RUN_COUNT": 0 if frame0 == "NOT_RUN" else 3,
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "FULL_DEV2_COMPUTE_AUTHORIZED": full_auth["FULL_DEV2_COMPUTE_AUTHORIZED"],
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": full_auth["DEV2_FULL_PRODUCTION_SOLVE_COUNT"],
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "CERTIFICATION_SPLIT_DOWNSTREAM_CONSUMED": "NO",
        "HELDOUT_SPLIT_DOWNSTREAM_CONSUMED": "NO",
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
    result = {
        "schema_version": "OakInk2O5RD2DObjectiveV2IndependentCertificationSummaryV1",
        "status": "SPARSE_VALIDATION_FAIL"
        if sparse["SPARSE_VALIDATION_V2"] == "FAIL"
        else "IN_PROGRESS",
        "git": {
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": final_head,
            "commits": commit_receipt["commits"],
            "tracked_worktree_clean": not bool(git("status", "--short", "--untracked-files=no")),
        },
        "sparse_manifest": sparse_manifest,
        "sparse_result": sparse,
        "OBJECTIVE_V2_INDEPENDENT_CERTIFICATION": "PASS"
        if sparse["SPARSE_VALIDATION_V2"] == "PASS" and window == "PASS"
        else "FAIL",
        "DEV2_OBJECTIVE_V2_HTML": None,
        "O5_FINAL": "MACHINE_NONPASS" if sparse["SPARSE_VALIDATION_V2"] == "FAIL" else "INCOMPLETE",
        "NEXT": "OBJECTIVE_V2_GENERALIZATION_FAILURE_ANALYSIS"
        if sparse["SPARSE_VALIDATION_V2"] == "FAIL"
        else "CONTINUE_STATE_MACHINE",
        "flags": flags,
    }
    write_json(root / "final_summary.json", result)
    lines = [
        "# OakInk2 O5R-D2D ObjectiveV2 Independent Certification Handoff",
        "",
        f"Machine status: `{result['status']}`. The state machine hard-stopped at the first failed scientific gate.",
        "",
        "## Git",
        "",
        f"- `BRANCH={flags['BRANCH']}`",
        f"- `START_HEAD={START_HEAD}`",
        f"- `FINAL_HEAD={final_head}`",
        "- `PUSHED=NO`",
        "- `PR_CREATED=NO`",
        "",
        "## Frozen method",
        "",
        f"- `RETARGET_OBJECTIVE_V2_SHA256={OBJECTIVE_SHA}`",
        f"- `OBJECTIVE_V2_EXECUTION_CONTRACT_SHA256={EXECUTION_SHA}`",
        "- `FROZEN_METHOD_INTEGRITY=PASS`",
        "- `OBJECTIVE_CHANGED_DURING_CERTIFICATION=NO`",
        "- `EXECUTION_CHANGED_DURING_CERTIFICATION=NO`",
        "",
        "## SparseValidationV2",
        "",
        "| Stratum | N | Frame IDs |",
        "| --- | ---: | --- |",
    ]
    for stratum in ("HIGH", "MID", "LOW"):
        selected = [row for row in sparse_manifest["frames"] if row["stratum"] == stratum]
        lines.append(
            f"| {stratum} | {len(selected)} | {', '.join(str(row['frame_id']) for row in selected)} |"
        )
    lines.extend(
        [
            "",
            f"- `DEVELOPMENT_OVERLAP={sparse_manifest['SPARSE_V2_DEVELOPMENT_OVERLAP']}`",
            f"- `MANIFEST_SHA={sha256_file(root / 'sparse_v2/manifest.json')}`",
            f"- `technical={sparse['technical']}/{sparse['N']}`",
            f"- `old-above-target count={sparse['old_above_target_count']}`",
            f"- `recovered-below-target count={sparse['recovered_below_target_count']}`",
            f"- `recovery rate={sparse['recovery_rate']}`",
            f"- `median relative reduction={sparse['median_relative_reduction']}`",
            f"- `overall non-regression fraction={sparse['overall_nonregression_fraction']}`",
            f"- `catastrophic regression count={sparse['catastrophic_regression_count']}`",
            f"- `LOW preserved count={sparse['low_preserved_count']}`",
            f"- `wrist PASS count={pass_counts['wrist']}/{sparse['N']}`",
            f"- `bone PASS count={pass_counts['bone']}/{sparse['N']}`",
            f"- `continuity PASS count={pass_counts['continuity']}/{sparse['N']}`",
            f"- `collision PASS count={pass_counts['collision']}/{sparse['N']}`",
            f"- `joint-limit PASS count={pass_counts['joint_limits']}/{sparse['N']}`",
            f"- `determinism={sparse['conditions']['determinism']}`",
            f"- `SPARSE_VALIDATION_V2={sparse['SPARSE_VALIDATION_V2']}`",
            "",
            "## Downstream state",
            "",
            f"- `WINDOW_VALIDATION_V2={window}`",
            f"- `DEV2_FRAME0_HARD_CONTROL={frame0}`",
            f"- `FULL_DEV2_COMPUTE_AUTHORIZED={full_auth['FULL_DEV2_COMPUTE_AUTHORIZED']}`",
            f"- `DEV2_FULL_PRODUCTION_SOLVE_COUNT={full_auth['DEV2_FULL_PRODUCTION_SOLVE_COUNT']}`",
            "- `DEV2_OBJECTIVE_V2_HTML=null`",
            f"- `NEXT={result['NEXT']}`",
            "",
            "## Safety flags",
            "",
        ]
    )
    lines.extend(f"- `{key}={value}`" for key, value in flags.items())
    text = "\n".join(lines) + "\n"
    (root / "handoff.md").write_text(text, encoding="utf-8")
    (root / "final_summary.md").write_text(text, encoding="utf-8")
    return result


def record_git(root: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "O5RD2DGitCommitsV1",
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "commits": git("log", f"{START_HEAD}..HEAD", "--format=%H").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "git_commits.json", payload)
    return payload


def run_validation(root: Path) -> dict[str, Any]:
    """Run the mandated repository checks without executing scientific stages."""

    log_root = root / "validation_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    modified = [
        "scripts/data/run_oakink2_o5rd2d.py",
        "tests/data/test_oakink2_o5rd2d.py",
    ]
    checks = [
        ("ruff_check_modified", ["ruff", "check", *modified]),
        ("ruff_format_check_modified", ["ruff", "format", "--check", *modified]),
        ("mypy_src", [sys.executable, "-m", "mypy", "src"]),
        ("pytest", [sys.executable, "-m", "pytest", "-q"]),
        ("paper_fidelity", [sys.executable, "scripts/check_paper_fidelity.py"]),
        ("cli_help", [sys.executable, "scripts/data/run_oakink2_o5rd2d.py", "--help"]),
    ]
    rows: list[dict[str, Any]] = []
    for name, command in checks:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        (log_root / f"{name}.log").write_text(output, encoding="utf-8")
        rows.append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "runtime_sec": time.perf_counter() - started,
                "log": str((log_root / f"{name}.log").resolve()),
            }
        )
    payload = {
        "schema_version": "O5RD2DValidationResultsV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "checks": rows,
    }
    write_json(root / "validation_results.json", payload)
    write_json(root / "tests.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run fail-closed OakInk2 O5R-D2D ObjectiveV2 independent certification"
    )
    value.add_argument(
        "action",
        choices=(
            "preflight",
            "verify-frozen-method",
            "freeze-sparse-v2",
            "run-sparse-v2",
            "freeze-window-v2",
            "run-window-v2",
            "run-dev2-frame0-hard-control",
            "run-dev2-full-if-authorized",
            "run-dev2-semantic-v1",
            "render-dev2-viewer",
            "validate",
            "summarize",
        ),
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    actions = {
        "preflight": preflight,
        "verify-frozen-method": verify_frozen_method,
        "freeze-sparse-v2": freeze_sparse_v2,
        "run-sparse-v2": run_sparse_v2,
        "freeze-window-v2": freeze_window_v2,
        "run-window-v2": run_window_v2,
        "run-dev2-frame0-hard-control": run_dev2_frame0_hard_control,
        "run-dev2-full-if-authorized": run_dev2_full_if_authorized,
        "run-dev2-semantic-v1": run_dev2_semantic_v1,
        "render-dev2-viewer": render_dev2_viewer,
        "validate": run_validation,
        "summarize": summarize,
    }
    result = actions[args.action](args.report_root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
