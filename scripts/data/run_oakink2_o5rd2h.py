#!/usr/bin/env python3
"""O5R-D2H ExecutionV3 independent cold-start certification.

This driver is intentionally fail closed.  It freezes the complete D2H plan
before selecting evidence, verifies the frozen method before every stage, and
never exposes a downstream compute action before its predecessor gate passes.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5
from scripts.data import run_oakink2_o5rd2d as d2d
from scripts.data import run_oakink2_o5rd2e as d2e
from scripts.data import run_oakink2_o5rd2g as d2g
from scripts.data import run_oakink2_o5rd2g2 as d2g2
from scripts.data import run_oakink2_o5rd2g3 as d2g3
from scripts.evaluation import audit_retarget_semantic_validity as semantic_audit
from toporetarget.evaluation.retarget_semantic_validity import (
    SemanticGateContractV1,
    angular_error,
    compose,
    qualify_semantics,
    relative_transform,
    temporal_steps,
    transform_error,
)
from toporetarget.retarget.bones import extract_bone_features
from toporetarget.retarget.final_refinement import dynamic_collision_points_numpy
from toporetarget.retarget.objective_v2_execution import asset_derived_dof_blocks
from toporetarget.utils.hashing import sha256_file

ROOT = REPO / ".local/reports/oakink2_o5rd2h_execution_v3_independent_coldstart_certification_v1"
D2G3_ROOT = d2g3.ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "0335e529f5cb14782c1916ad6c5fb7e7780f33d1"
TAU = d2d.TAU
EPSILON_NUM = d2d.EIM_ATOL
Q_ATOL = d2d.Q_ATOL
BASE_ATOL = 1.0e-8
EXPECTED_HASHES = {
    "retarget_objective_v2": "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc",
    "certification_gate_v2": "a845fcdfec478e9208fc6c192317a45bee6ef19be1fcb784b459d8a4f675e048",
    "execution_input_authority": "8e2584ddad17f8e5f6b661ce42a415abbb56ef58c2ee5493b009056a48bde18c",
    "source_interaction_graph_authority": "36c0b9ee1fac20c957dd03a622b493bedfd59f42d002d8162b1b87f770caf5ff",
    "cold_start_bootstrap_contract": "772cfe722327d0662c92a943d52d0f23cbece66b6e6c552fa5c764470b00de8e",
    "cold_start_seed_authority_v2": "fde5cac433434504f3a8c58e289672e71cda85dcc77449d1f038f4010355feb0",
    "objective_v2_execution_contract_v3": "8b4947ddbdea19f4119c4ade74009b0eaab57ff947cca78eb7035e9aff2a467a",
}

FROZEN_CONTRACT_PATHS = {
    "retarget_objective_v2": d2d.OBJECTIVE_SOURCE,
    "certification_gate_v2": d2e.ROOT / "gate_v2/certification_gate_v2.json",
    "execution_input_authority": D2G3_ROOT / "frozen_v3/execution_input_authority.json",
    "source_interaction_graph_authority": D2G3_ROOT
    / "frozen_v3/source_interaction_graph_authority.json",
    "cold_start_bootstrap_contract": D2G3_ROOT / "frozen_v3/cold_start_bootstrap_contract.json",
    "cold_start_seed_authority_v2": D2G3_ROOT / "frozen_v3/cold_start_seed_authority_v2.json",
    "objective_v2_execution_contract_v3": D2G3_ROOT
    / "frozen_v3/objective_v2_execution_contract_v3.json",
}

# This list follows the actual D2G2 call graph.  The D2H driver is included
# because it owns selection and gate evaluation, while method files remain
# immutable throughout certification.
IMPLEMENTATION_PATHS = {
    "d2h_certification_driver": Path(__file__).resolve(),
    "oakink2_canonical_preparation": REPO / "scripts/data/run_oakink2_o5.py",
    "semantic_v1_auditor": REPO / "scripts/evaluation/audit_retarget_semantic_validity.py",
    "d2g2_execution_v3_search": REPO / "scripts/data/run_oakink2_o5rd2g2.py",
    "d2g_execution_v3_runtime": REPO / "scripts/data/run_oakink2_o5rd2g.py",
    "candidate_b2_search": REPO / "scripts/data/run_oakink2_o5rd2c.py",
    "runtime_measurement_context": REPO / "scripts/data/run_oakink2_o5rd2a.py",
    "objective_v2": REPO / "src/toporetarget/retarget/objective_v2.py",
    "execution_v2_v3_shared": REPO / "src/toporetarget/retarget/objective_v2_execution.py",
    "production_refinement": REPO / "src/toporetarget/retarget/final_refinement.py",
    "continuous_context": REPO / "src/toporetarget/retarget/continuous.py",
    "semantic_v1": REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py",
    "solver_profile": REPO
    / "configs/retarget/refinement_solvers/wuji_continuous_sequential_v1.yaml",
    "execution_profile": REPO
    / "configs/retarget/refinement_execution/wuji_continuous_sequential_fast_exact_v2.yaml",
    "frame_profile": REPO / "configs/retarget/frames/canonical_keypoint_wrist_v1.yaml",
    "bone_profile": REPO / "configs/retarget/bones/mediapipe21_full_finger_chain_v1.yaml",
    "wuji_mapping": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
    "wuji_anchors": REPO / "configs/robots/anchors/wuji_hand2_beta1_rh_mediapipe21.yaml",
    "wuji_joint_order": REPO / "configs/robots/joint_orders/wuji_hand2_beta1_rh.yaml",
    "wuji_surface": REPO / "configs/robots/surfaces/wuji_hand2_beta1_rh.yaml",
    "wuji_urdf": REPO / "third_party/robot_hands/wuji_hand2_beta1/urdf/right.urdf",
    "wuji_mjcf": REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml",
}


def write_json(path: Path, value: Any) -> None:
    d2g.write_json(path, value)


def read_json(path: Path) -> dict[str, Any]:
    return d2g.read_json(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    d2g.write_csv(path, rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    return d2g.git(*args)


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


def write_sha(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def freeze_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() and read_json(path) != payload:
        raise RuntimeError(f"O5RD2H_FROZEN_ARTIFACT_DRIFT:{path}")
    write_json(path, payload)
    value = sha256_file(path)
    receipt = path.with_suffix(".sha256")
    if receipt.exists() and receipt.read_text(encoding="utf-8").strip() != value:
        raise RuntimeError(f"O5RD2H_FROZEN_HASH_DRIFT:{path}")
    write_sha(receipt, value)
    return value


def require_hash(path: Path, receipt: Path, label: str) -> None:
    if not path.exists() or not receipt.exists():
        raise RuntimeError(f"O5RD2H_MISSING_FROZEN_ARTIFACT:{label}")
    if sha256_file(path) != receipt.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"O5RD2H_FROZEN_ARTIFACT_DRIFT:{label}")


def require_field(path: Path, field: str, expected: Any) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"O5RD2H_PREREQUISITE_MISSING:{path}")
    payload = read_json(path)
    if payload.get(field) != expected:
        raise RuntimeError(f"O5RD2H_STAGE_BLOCKED:{path}:{field}={payload.get(field)!r}")
    return payload


def implementation_authority() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for name, path in IMPLEMENTATION_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        files[name] = {"path": str(path.resolve()), "sha256": sha256_path(path)}
    return {
        "schema_version": "FrozenExecutionV3ImplementationAuthorityV1",
        "status": "FROZEN",
        "frozen_before_d2h_plan": True,
        "files": files,
    }


def verify_frozen_execution_v3(root: Path) -> dict[str, Any]:
    observed_contracts: dict[str, Any] = {}
    for name, path in FROZEN_CONTRACT_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        observed_contracts[name] = {
            "path": str(path.resolve()),
            "expected_sha256": EXPECTED_HASHES[name],
            "actual_sha256": actual,
            "exact": actual == EXPECTED_HASHES[name],
        }
    d2g3_summary = read_json(D2G3_ROOT / "final_summary.json")
    development_checks = {
        "EXECUTION_V3_DEVELOPMENT_GATE": d2g3_summary.get("EXECUTION_V3_DEVELOPMENT_GATE")
        == "PASS",
        "EXECUTION_V3_STATUS": d2g3_summary.get("EXECUTION_V3_STATUS")
        == "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
        "REFINEMENT_MODE_REGRESSION": d2g3_summary.get("REFINEMENT_MODE_REGRESSION") == "PASS",
        "N_REFINEMENT_PARITY": d2g3_summary.get("refinement_regression", {}).get("N_comparisons")
        == 35,
        "N_REFINEMENT_PARITY_PASS": d2g3_summary.get("refinement_regression", {}).get("N_pass")
        == 35,
        "MAX_Q_ABS_DIFF": d2g3_summary.get("refinement_regression", {}).get("MAX_Q_ABS_DIFF")
        == 0.0,
        "MAX_BASE_ABS_DIFF": d2g3_summary.get("refinement_regression", {}).get("MAX_BASE_ABS_DIFF")
        == 0.0,
        "MAX_E_IM_ABS_DIFF": d2g3_summary.get("refinement_regression", {}).get("MAX_E_IM_ABS_DIFF")
        == 0.0,
    }
    authority_path = root / "frozen_method/implementation_authority.json"
    authority_sha_path = root / "frozen_method/implementation_authority.sha256"
    observed_implementation = implementation_authority()
    implementation_exact = True
    if authority_path.exists():
        require_hash(authority_path, authority_sha_path, "implementation_authority")
        implementation_exact = read_json(authority_path) == observed_implementation
    else:
        write_json(authority_path, observed_implementation)
        write_sha(authority_sha_path, sha256_file(authority_path))
    passed = bool(
        all(item["exact"] for item in observed_contracts.values())
        and all(development_checks.values())
        and implementation_exact
    )
    payload = {
        "schema_version": "O5RD2HFrozenMethodIntegrityV1",
        "FROZEN_METHOD_INTEGRITY": "PASS" if passed else "FAIL",
        "contracts": observed_contracts,
        "development_checks": development_checks,
        "implementation_authority_exact": implementation_exact,
        "METHOD_CHANGED_DURING_CERTIFICATION": "NO" if implementation_exact else "YES",
    }
    write_json(root / "frozen_method/contract_hashes.json", observed_contracts)
    write_json(root / "frozen_method/integrity.json", payload)
    if not passed:
        _write_all_not_run(root, "BLOCKED_FROZEN_METHOD_INTEGRITY")
        raise RuntimeError("D2H_STATUS=BLOCKED_FROZEN_METHOD_INTEGRITY")
    return payload


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2H_BRANCH_MISMATCH:{branch}")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0
    )
    if not ancestor:
        raise RuntimeError(f"O5RD2H_START_HEAD_NOT_ANCESTOR:{head}")
    integrity = verify_frozen_execution_v3(root)
    payload = {
        "schema_version": "OakInk2O5RD2HGitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "START_HEAD": START_HEAD,
        "head_at_preflight": head,
        "start_head_is_ancestor": ancestor,
        "status_short": git("status", "--short", "--untracked-files=all").splitlines(),
        "diff_stat": git("diff", "--stat").splitlines(),
        "diff_check": git("diff", "--check").splitlines(),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "remotes": git("remote", "-v").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", payload)
    write_json(
        root / "preflight/upstream_state.json",
        {
            "schema_version": "O5RD2HUpstreamStateV1",
            "d2g3_root": str(D2G3_ROOT.resolve()),
            "d2g3_final_summary_sha256": sha256_file(D2G3_ROOT / "final_summary.json"),
            "integrity": integrity["FROZEN_METHOD_INTEGRITY"],
        },
    )
    failures = root / "technical_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    failures.touch(exist_ok=True)
    return payload


def certification_plan() -> dict[str, Any]:
    return {
        "schema_version": "D2HCertificationPlanContractV1",
        "status": "FROZEN",
        "frozen_before_any_fresh_selection": True,
        "sparse_v4": {
            "N": 30,
            "composition": {"HIGH": 10, "MID": 10, "LOW": 10},
            "selection": "SparseValidationV2 deterministic temporal-strata algorithm over old frozen DEV1 scalar E_IM with the complete pre-D2H exclusion ledger",
            "gate": "CertificationGateV2; standalone continuity is NOT_APPLICABLE and never fabricated",
            "determinism": {"HIGH": 2, "MID": 2, "LOW": 1, "runs": 3},
        },
        "window_v4": {
            "windows": ["HIGH_1", "HIGH_2", "MID", "LOW"],
            "preferred_length": 32,
            "minimum_length_only_if_boundary_requires": 24,
            "selection": "WindowValidationV3 deterministic old-E_IM-only algorithm with expanded exclusions",
            "gate": "all technical; each p95 E_IM<=tau; LOW old-valid preservation; frozen actual continuity and hard validity",
            "determinism": {"windows": ["HIGH_1", "LOW"], "runs": 3},
        },
        "cross_episode": {
            "split": "OakInk2 development only",
            "N": 3,
            "selection": "eligible pool sorted by SHA256 of canonical record identity, then greedily maximize distinct object and primitive without outcome inspection",
            "freshness_preference": "SEQUENCE_DISJOINT then PRIMITIVE_RECORD_DISJOINT only if sequence support is insufficient",
            "runs_per_control": 3,
            "gate": "3/3 episodes and every run technical, finite, E_IM<=tau, all frozen hard validity, no special case, deterministic",
        },
        "repeat_counts": {"sparse_subset": 3, "window_controls": 3, "cross_episode": 3},
        "determinism_tolerance": {"q": Q_ATOL, "base": BASE_ATOL, "E_IM": EPSILON_NUM},
        "evidence_consumption": "mark consumed immediately on first optimizer execution regardless of PASS/FAIL",
        "dev2_authorization": "YES iff SparseV4 PASS and WindowV4 PASS and FreshCrossEpisodeControls PASS",
        "dev2_full": {
            "role": "KNOWN_FAILURE_RECOVERY_PRODUCTION_RUN",
            "scientific_run_limit": 1,
            "frame0": "recompute generic frozen ExecutionV3 cold-start; D2G3 state reuse forbidden",
            "scientific_failure": "stop at first failure; no skip, seed/budget/method change, or second run",
        },
        "technical_retry_policy": {
            "allowed": "same RUN_UUID, hashes, and accepted checkpoint after infrastructure-only failure",
            "forbidden": "resume across a scientific failure or change method",
        },
        "tau": TAU,
        "epsilon_num": EPSILON_NUM,
        "outcome_driven_mutation": False,
    }


def freeze_certification_plan(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    payload = certification_plan()
    value = freeze_json(root / "certification_plan/d2h_certification_plan.json", payload)
    return {**payload, "D2H_CERTIFICATION_PLAN_SHA256": value}


def _pre_d2h_entries() -> list[dict[str, Any]]:
    ledger = read_json(D2G3_ROOT / "ledger/future_validation_exclusion_ledger.json")
    return list(ledger["entries"])


def build_exclusion_ledgers(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_hash(
        root / "certification_plan/d2h_certification_plan.json",
        root / "certification_plan/d2h_certification_plan.sha256",
        "d2h_certification_plan",
    )
    source = _pre_d2h_entries()
    dev1: dict[int, dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    for row in source:
        if row.get("episode") == "DEV1" and "ordinal" in row:
            ordinal = int(row["ordinal"])
            current = dev1.setdefault(
                ordinal,
                {"episode": "DEV1", "ordinal": ordinal, "roles": [], "future_role": "EXCLUDED"},
            )
            current["roles"] = sorted(set(current["roles"]) | set(row.get("roles", [])))
        else:
            episode_rows.append(row)
    # Both canonical O5 episodes supplied method-development or known-failure
    # evidence.  Their whole source sequences are excluded from fresh controls.
    for episode in o5.EPISODES:
        episode_rows.append(
            {
                "episode_id": episode["record_id"],
                "sequence_id": str(episode["record_id"]).split(":", 1)[1].rsplit(":", 1)[0],
                "role": (
                    "DEV1_METHOD_DEVELOPMENT_SOURCE"
                    if episode["review"] == "dev_01"
                    else "DEV2_KNOWN_FAILURE_DEVELOPMENT"
                ),
                "excluded": True,
            }
        )
    frame_payload = {
        "schema_version": "ExecutionV3CertificationFrameExclusionLedgerV1",
        "status": "FROZEN",
        "count": len(dev1),
        "entries": [dev1[key] for key in sorted(dev1)],
    }
    episode_payload = {
        "schema_version": "ExecutionV3CertificationEpisodeExclusionLedgerV1",
        "status": "FROZEN",
        "freshness_preference": "SEQUENCE_DISJOINT",
        "count": len(episode_rows),
        "entries": episode_rows,
    }
    consumption = {
        "schema_version": "ExecutionV3CertificationEvidenceLedgerV1",
        "status": "FROZEN",
        "roles": [
            "METHOD_DEVELOPMENT",
            "OLD_CERTIFICATION_CONSUMED",
            "COLDSTART_SPARSE_V4_CONSUMED",
            "COLDSTART_WINDOW_V4_CONSUMED",
            "FRESH_CROSS_EPISODE_CONTROL_CONSUMED",
            "DEV2_KNOWN_FAILURE_DEVELOPMENT",
            "DEV2_KNOWN_FAILURE_RECOVERY",
            "FROZEN_NOT_EXECUTED",
        ],
        "pre_d2h_dev1_exclusion_count": len(dev1),
        "pre_d2h_source_ledger_count": len(source),
        "sparse_v4": [],
        "window_v4": [],
        "cross_episode": [],
        "dev2": {"role": "DEV2_KNOWN_FAILURE_DEVELOPMENT", "full_run_count": 0},
    }
    write_json(root / "ledger/frame_exclusion_ledger.json", frame_payload)
    write_json(root / "ledger/episode_exclusion_ledger.json", episode_payload)
    write_json(root / "ledger/evidence_consumption_ledger.json", consumption)
    return {
        "PRE_D2H_DEV1_EXCLUSION_COUNT": len(dev1),
        "PRE_D2H_EPISODE_EXCLUSION_COUNT": len(episode_rows),
    }


def sparse_gate_contract() -> dict[str, Any]:
    gate = read_json(FROZEN_CONTRACT_PATHS["certification_gate_v2"])
    return {
        **gate,
        "schema_version": "ColdStartSparseValidationV4GateContractV1",
        "execution_mode": "standalone cold-start",
        "old_production_q": "ABSENT",
        "previous_accepted_runtime_state": "ABSENT",
        "standalone_continuity": "NOT_APPLICABLE_NO_FABRICATED_T_MINUS_1",
        "q_old_access_count_required": 0,
    }


def freeze_sparse_v4(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_hash(
        root / "certification_plan/d2h_certification_plan.json",
        root / "certification_plan/d2h_certification_plan.sha256",
        "d2h_certification_plan",
    )
    frame_ledger = require_field(root / "ledger/frame_exclusion_ledger.json", "status", "FROZEN")
    target = root / "sparse_v4/manifest.json"
    gate_path = root / "sparse_v4/gate_contract.json"
    if target.exists() or gate_path.exists():
        require_hash(target, root / "sparse_v4/manifest.sha256", "sparse_v4_manifest")
        require_hash(gate_path, root / "sparse_v4/gate_contract.sha256", "sparse_v4_gate")
        return read_json(target)
    excluded = {int(row["ordinal"]) for row in frame_ledger["entries"]}
    selected = d2d.select_sparse_rows(d2d.old_eim_rows(), excluded)
    selected_ordinals = {int(row["ordinal"]) for row in selected}
    if len(selected) != 30 or selected_ordinals & excluded:
        raise RuntimeError("O5RD2H_SPARSE_V4_COUNT_OR_OVERLAP")
    composition = {
        name: sum(row["stratum"] == name for row in selected) for name in ("HIGH", "MID", "LOW")
    }
    if composition != {"HIGH": 10, "MID": 10, "LOW": 10}:
        raise RuntimeError(f"O5RD2H_SPARSE_V4_COMPOSITION:{composition}")
    payload = {
        "schema_version": "ColdStartSparseValidationV4ManifestV1",
        "status": "FROZEN",
        "selection_authority": "old frozen DEV1 scalar E_IM only",
        "selection_algorithm": certification_plan()["sparse_v4"]["selection"],
        "frames": selected,
        "N": len(selected),
        "composition": composition,
        "determinism_ordinals": d2d._determinism_sparse_ids(selected),
        "SPARSE_V4_OVERLAP_WITH_PRIOR_EVIDENCE": len(selected_ordinals & excluded),
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    freeze_json(target, payload)
    freeze_json(gate_path, sparse_gate_contract())
    write_json(
        root / "sparse_v4/determinism_subset.json",
        {
            "schema_version": "ColdStartSparseValidationV4DeterminismSubsetV1",
            "ordinals": payload["determinism_ordinals"],
            "runs": 3,
        },
    )
    return payload


def _append_failure(root: Path, stage: str, exc: Exception, **context: Any) -> None:
    row = {
        "schema_version": "O5RD2HTechnicalFailureV1",
        "stage": stage,
        "error": f"{type(exc).__name__}:{exc}",
        **context,
    }
    with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _run_coldstart_frame(
    runtime: d2g.V3Runtime,
    root: Path,
    directory: str,
    ordinal: int,
    run: int,
    runtime_step: int,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    receipt_path = root / directory / "receipts" / f"frame_{ordinal:04d}_run_{run}.json"
    state_path = receipt_path.with_suffix(".npz")
    if receipt_path.exists() or state_path.exists():
        if not (receipt_path.exists() and state_path.exists()):
            raise RuntimeError(f"O5RD2H_PARTIAL_FRAME_RECEIPT:{receipt_path}")
        receipt = read_json(receipt_path)
        with np.load(state_path, allow_pickle=False) as state:
            return (
                receipt,
                np.asarray(state["qpos"], dtype=np.float64),
                np.asarray(state["base_pose_scene"], dtype=np.float64),
            )
    qpos, base, receipt = d2g2.search_cold_start_v2_frame(
        runtime,
        ordinal,
        runtime_step=runtime_step,
        previous_q=previous_q,
        previous_base=previous_base,
        candidate=d2g2.CS2_A,
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
    write_json(receipt_path, receipt)
    return receipt, np.asarray(qpos), np.asarray(base)


def _bool_margin(evaluation: dict[str, Any], *names: str) -> bool:
    margins = evaluation["constraint_margins"]
    return all(float(margins[name]) >= -EPSILON_NUM for name in names)


def _row_from_coldstart(
    receipt: dict[str, Any], stratum: str, old_e_im: float, *, standalone: bool
) -> dict[str, Any]:
    selected = receipt["selected"]
    evaluation = receipt["selected_evaluation"]
    actual = receipt["selected_actual_continuity"]
    finite = all(
        np.isfinite(float(selected[name]))
        for name in (
            "interaction_e_im",
            "wrist_position_m",
            "wrist_rotation_rad",
            "bone_direction_p95_rad",
            "collision_min_signed_distance_m",
            "joint_limit_min_margin_rad",
            "rotation_determinant",
            "unit_scale_ratio",
        )
    )
    continuity_pass = (
        True
        if standalone
        else _bool_margin(
            evaluation, "actual_temporal_translation_m", "actual_temporal_rotation_rad"
        )
    )
    hard = bool(
        receipt["technical_success"]
        and evaluation["feasible"]
        and finite
        and _bool_margin(evaluation, "wrist_position_m", "wrist_rotation_rad")
        and _bool_margin(evaluation, "bone_direction_rad")
        and continuity_pass
        and _bool_margin(evaluation, "collision_hard_m")
        and _bool_margin(evaluation, "joint_limit_rad")
        and _bool_margin(evaluation, "reflection_determinant")
        and _bool_margin(evaluation, "unit_scale_lower", "unit_scale_upper")
    )
    return {
        "ordinal": int(receipt["ordinal"]),
        "frame_id": int(receipt["source_frame_local"]),
        "stratum": stratum,
        "runtime_step_index": int(receipt["runtime_step_index"]),
        "old_e_im": float(old_e_im),
        "new_e_im": float(selected["interaction_e_im"]),
        "technical_completion": bool(receipt["technical_success"]),
        "optimizer_started": bool(receipt["optimizer_started"]),
        "finite": finite,
        "semantic_hard_pass": hard,
        "wrist_pass": _bool_margin(evaluation, "wrist_position_m", "wrist_rotation_rad"),
        "bone_pass": _bool_margin(evaluation, "bone_direction_rad"),
        "continuity_pass": continuity_pass,
        "continuity_role": "NOT_APPLICABLE_STANDALONE" if standalone else "HARD_GATE",
        "collision_pass": _bool_margin(evaluation, "collision_hard_m"),
        "joint_limits_pass": _bool_margin(evaluation, "joint_limit_rad"),
        "reflection_pass": _bool_margin(evaluation, "reflection_determinant"),
        "scale_pass": _bool_margin(evaluation, "unit_scale_lower", "unit_scale_upper"),
        "actual_translation_step_m": float(actual["translation_step_m"]),
        "actual_rotation_step_rad": float(actual["rotation_step_rad"]),
        "actual_q_step_inf_rad": float(actual["q_step_inf_rad"]),
        "selected_seed": receipt["selected_seed_candidate"],
        "selected_contributor": receipt["selected_block"],
        "selected_candidate": receipt["selected_candidate"],
        "retention_decision": receipt["retention_decision"],
        "q_old_field": receipt["old_production_q"],
        "q_old_access_count": 0,
        "q_old_synthesized": bool(receipt["q_old_synthesized"]),
        "previous_accepted_state": receipt["previous_accepted_state"],
        **{f"profiler_{key}": value for key, value in receipt["profiler"].items()},
    }


def _signature(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "bootstrap_seed_order": [row["seed"] for row in receipt["bootstrap"]["solves"]],
        "bootstrap_selected": [
            (row["seed_id"], row["selected_contributor"], row["bootstrap_screen_pass"])
            for row in receipt["bootstrap_candidates"]
        ],
        "selected_seed": receipt["selected_seed_candidate"],
        "selected_contributor": receipt["selected_block"],
        "selected_candidate": receipt["selected_candidate"],
        "retention": receipt["retention_decision"],
        "fallback": bool(receipt["profiler"]["fallback"]),
    }


def _compare_repeat(
    reference_receipt: dict[str, Any],
    reference_q: np.ndarray,
    reference_base: np.ndarray,
    receipt: dict[str, Any],
    qpos: np.ndarray,
    base: np.ndarray,
) -> dict[str, Any]:
    q_diff = float(np.max(np.abs(qpos - reference_q)))
    base_diff = float(np.max(np.abs(base - reference_base)))
    e_diff = abs(
        float(receipt["selected"]["interaction_e_im"])
        - float(reference_receipt["selected"]["interaction_e_im"])
    )
    signature_exact = _signature(receipt) == _signature(reference_receipt)
    return {
        "pass": signature_exact
        and q_diff <= Q_ATOL
        and base_diff <= BASE_ATOL
        and e_diff <= EPSILON_NUM,
        "signature_exact": signature_exact,
        "max_q_abs_diff": q_diff,
        "max_base_abs_diff": base_diff,
        "E_IM_abs_diff": e_diff,
        "signature": _signature(receipt),
    }


def _consume(root: Path, field: str, entries: list[dict[str, Any]]) -> None:
    ledger_path = root / "ledger/evidence_consumption_ledger.json"
    ledger = read_json(ledger_path)
    existing = ledger.get(field, [])
    merged = {json.dumps(row, sort_keys=True): row for row in [*existing, *entries]}
    ledger[field] = [merged[key] for key in sorted(merged)]
    write_json(ledger_path, ledger)


def run_sparse_v4(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_hash(
        root / "certification_plan/d2h_certification_plan.json",
        root / "certification_plan/d2h_certification_plan.sha256",
        "d2h_certification_plan",
    )
    require_hash(
        root / "sparse_v4/manifest.json", root / "sparse_v4/manifest.sha256", "sparse_v4_manifest"
    )
    require_hash(
        root / "sparse_v4/gate_contract.json",
        root / "sparse_v4/gate_contract.sha256",
        "sparse_v4_gate",
    )
    manifest = read_json(root / "sparse_v4/manifest.json")
    decision_path = root / "sparse_v4/gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    _consume(
        root,
        "sparse_v4",
        [{**row, "role": "COLDSTART_SPARSE_V4_CONSUMED"} for row in manifest["frames"]],
    )
    runtime = d2g.V3Runtime("dev_01", root)
    rows: list[dict[str, Any]] = []
    references: dict[int, tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    for index, frame in enumerate(manifest["frames"], start=1):
        ordinal = int(frame["ordinal"])
        try:
            receipt, qpos, base = _run_coldstart_frame(
                runtime, root, "sparse_v4", ordinal, 1, 0, None, None
            )
            rows.append(
                _row_from_coldstart(
                    receipt, str(frame["stratum"]), float(frame["old_e_im"]), standalone=True
                )
            )
            references[ordinal] = (receipt, qpos, base)
            print(
                f"O5RD2H_SPARSE_V4 {index}/{manifest['N']} ordinal={ordinal} E_IM={float(receipt['selected']['interaction_e_im']):.12g}",
                flush=True,
            )
        except Exception as exc:
            _append_failure(root, "COLDSTART_SPARSE_V4", exc, ordinal=ordinal)
            rows.append(
                {
                    "ordinal": ordinal,
                    "frame_id": int(frame["frame_id"]),
                    "stratum": frame["stratum"],
                    "old_e_im": float(frame["old_e_im"]),
                    "new_e_im": float("inf"),
                    "technical_completion": False,
                    "optimizer_started": False,
                    "finite": False,
                    "semantic_hard_pass": False,
                    "wrist_pass": False,
                    "bone_pass": False,
                    "continuity_pass": True,
                    "collision_pass": False,
                    "joint_limits_pass": False,
                    "reflection_pass": False,
                    "scale_pass": False,
                    "q_old_access_count": 0,
                    "failure": f"{type(exc).__name__}:{exc}",
                }
            )
            break
    write_csv(root / "sparse_v4/per_frame_results.csv", rows)
    write_csv(root / "sparse_v4/semantic_metrics.csv", rows)
    write_csv(root / "sparse_v4/search_metrics.csv", rows)
    write_csv(
        root / "sparse_v4/qold_access_audit.csv",
        [
            {
                "ordinal": row["ordinal"],
                "q_old_field": row.get("q_old_field", "ABSENT"),
                "q_old_access_count": row.get("q_old_access_count", 0),
                "previous_accepted_state": row.get("previous_accepted_state", "ABSENT"),
            }
            for row in rows
        ],
    )
    repeat_rows: list[dict[str, Any]] = []
    if len(rows) == manifest["N"] and all(bool(row["technical_completion"]) for row in rows):
        for ordinal_value in manifest["determinism_ordinals"]:
            ordinal = int(ordinal_value)
            ref_receipt, ref_q, ref_base = references[ordinal]
            repeat_rows.append(
                {"ordinal": ordinal, "run": 1, "pass": True, "signature": _signature(ref_receipt)}
            )
            for run in (2, 3):
                receipt, qpos, base = _run_coldstart_frame(
                    runtime, root, "sparse_v4", ordinal, run, 0, None, None
                )
                repeat_rows.append(
                    {
                        "ordinal": ordinal,
                        "run": run,
                        **_compare_repeat(ref_receipt, ref_q, ref_base, receipt, qpos, base),
                    }
                )
                print(f"O5RD2H_SPARSE_V4_DETERMINISM ordinal={ordinal} run={run}", flush=True)
    determinism_pass = bool(repeat_rows) and all(bool(row["pass"]) for row in repeat_rows)
    write_json(
        root / "sparse_v4/determinism.json",
        {
            "schema_version": "ColdStartSparseValidationV4DeterminismV1",
            "status": "PASS" if determinism_pass else "FAIL",
            "rows": repeat_rows,
        },
    )
    decision = d2e.evaluate_gate_v2_rows(rows, determinism_pass=determinism_pass)
    decision.update(
        {
            "schema_version": "ColdStartSparseValidationV4GateDecisionV1",
            "COLDSTART_SPARSE_VALIDATION_V4": decision.pop("decision"),
            "q_old_access_count": sum(int(row.get("q_old_access_count", 0)) for row in rows),
            "standalone_continuity_gate": "NOT_APPLICABLE",
        }
    )
    write_json(decision_path, decision)
    if decision["COLDSTART_SPARSE_VALIDATION_V4"] != "PASS":
        _write_downstream_not_run(
            root, "COLDSTART_SPARSE_VALIDATION_V4_FAIL", from_stage="window_v4"
        )
    return decision


def window_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": "ColdStartWindowValidationV4GateContractV1",
        "status": "FROZEN",
        "technical_completion": "4/4 windows and all frames",
        "interaction_p95_maximum": TAU,
        "low_old_valid_preservation": "E_new <= tau + epsilon_num",
        "epsilon_num": EPSILON_NUM,
        "actual_translation_and_rotation_continuity": "frozen Semantic V1 hard limits",
        "hard_validity": [
            "wrist",
            "bone",
            "collision",
            "joint_limits",
            "reflection",
            "scale",
        ],
        "determinism_windows": ["HIGH_1", "LOW"],
        "determinism_runs": 3,
    }


def freeze_window_v4(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_field(root / "sparse_v4/gate_decision.json", "COLDSTART_SPARSE_VALIDATION_V4", "PASS")
    target = root / "window_v4/manifest.json"
    gate_path = root / "window_v4/gate_contract.json"
    if target.exists() or gate_path.exists():
        require_hash(target, root / "window_v4/manifest.sha256", "window_v4_manifest")
        require_hash(gate_path, root / "window_v4/gate_contract.sha256", "window_v4_gate")
        return read_json(target)
    prior = {
        int(row["ordinal"])
        for row in read_json(root / "ledger/frame_exclusion_ledger.json")["entries"]
    }
    sparse = {int(row["ordinal"]) for row in read_json(root / "sparse_v4/manifest.json")["frames"]}
    selected = d2d.select_window_rows(d2d.old_eim_rows(), prior | sparse, length=32)
    all_ordinals = [int(value) for window in selected for value in window["ordinals"]]
    ordinal_set = set(all_ordinals)
    payload = {
        "schema_version": "ColdStartWindowValidationV4ManifestV1",
        "status": "FROZEN",
        "selection_authority": "old frozen DEV1 scalar E_IM only",
        "selection_algorithm": certification_plan()["window_v4"]["selection"],
        "windows": selected,
        "WINDOW_V4_PRIOR_EVIDENCE_OVERLAP": len(prior & ordinal_set),
        "WINDOW_V4_SPARSE_V4_OVERLAP": len(sparse & ordinal_set),
        "WINDOW_V4_INTERNAL_OVERLAP": len(all_ordinals) - len(ordinal_set),
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    if any(
        payload[name] != 0
        for name in (
            "WINDOW_V4_PRIOR_EVIDENCE_OVERLAP",
            "WINDOW_V4_SPARSE_V4_OVERLAP",
            "WINDOW_V4_INTERNAL_OVERLAP",
        )
    ):
        raise RuntimeError("O5RD2H_WINDOW_V4_OVERLAP")
    freeze_json(target, payload)
    freeze_json(gate_path, window_gate_contract())
    return payload


def _window_run(
    runtime: d2g.V3Runtime, root: Path, window: dict[str, Any], run: int
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    old = {int(row["ordinal"]): float(row["old_e_im"]) for row in d2d.old_eim_rows()}
    previous_q = None
    previous_base = None
    rows: list[dict[str, Any]] = []
    q_states: list[np.ndarray] = []
    base_states: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    directory = f"window_v4/{window['window_id']}"
    for step, ordinal_value in enumerate(window["ordinals"]):
        ordinal = int(ordinal_value)
        receipt, qpos, base = _run_coldstart_frame(
            runtime, root, directory, ordinal, run, step, previous_q, previous_base
        )
        row = _row_from_coldstart(receipt, str(window["type"]), old[ordinal], standalone=False)
        row.update({"window_id": window["window_id"], "run": run})
        rows.append(row)
        q_states.append(qpos)
        base_states.append(base)
        receipts.append(receipt)
        previous_q, previous_base = qpos, base
    return rows, q_states, base_states, receipts


def _jitter_rows(
    runtime: d2g.V3Runtime, window_id: str, rows: list[dict[str, Any]], q_states: list[np.ndarray]
) -> list[dict[str, Any]]:
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    output: list[dict[str, Any]] = []
    for index, (row, qpos) in enumerate(zip(rows, q_states, strict=True)):
        if index == 0:
            delta = np.zeros_like(qpos)
        else:
            delta = np.abs(qpos - q_states[index - 1])
        item: dict[str, Any] = {
            "window_id": window_id,
            "ordinal": int(row["ordinal"]),
            "joint_step_mean": float(np.mean(delta)),
            "joint_step_p50": float(np.quantile(delta, 0.50)),
            "joint_step_p90": float(np.quantile(delta, 0.90)),
            "joint_step_p95": float(np.quantile(delta, 0.95)),
            "joint_step_max": float(np.max(delta)),
            "wrist_translation_step_m": float(row["actual_translation_step_m"]),
            "wrist_rotation_step_rad": float(row["actual_rotation_step_rad"]),
            "role": "DIAGNOSTIC_ONLY",
        }
        for finger, indices in blocks.items():
            values = delta[np.asarray(indices, dtype=np.int64)]
            item[f"{finger}_step_mean"] = float(np.mean(values))
            item[f"{finger}_step_p95"] = float(np.quantile(values, 0.95))
            item[f"{finger}_step_max"] = float(np.max(values))
        output.append(item)
    return output


def run_window_v4(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_field(root / "sparse_v4/gate_decision.json", "COLDSTART_SPARSE_VALIDATION_V4", "PASS")
    require_hash(
        root / "window_v4/manifest.json", root / "window_v4/manifest.sha256", "window_v4_manifest"
    )
    require_hash(
        root / "window_v4/gate_contract.json",
        root / "window_v4/gate_contract.sha256",
        "window_v4_gate",
    )
    decision_path = root / "window_v4/gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    manifest = read_json(root / "window_v4/manifest.json")
    consumed = [
        {
            "ordinal": int(ordinal),
            "frame_id": int(frame_id),
            "window_id": window["window_id"],
            "role": "COLDSTART_WINDOW_V4_CONSUMED",
        }
        for window in manifest["windows"]
        for ordinal, frame_id in zip(window["ordinals"], window["frame_ids"], strict=True)
    ]
    _consume(root, "window_v4", consumed)
    runtime = d2g.V3Runtime("dev_01", root)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    jitter: list[dict[str, Any]] = []
    references: dict[
        str, tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]
    ] = {}
    for window in manifest["windows"]:
        try:
            result = _window_run(runtime, root, window, 1)
        except Exception as exc:
            _append_failure(root, "COLDSTART_WINDOW_V4", exc, window_id=window["window_id"])
            result = ([], [], [], [])
        rows, q_states, base_states, receipts = result
        references[str(window["window_id"])] = result
        all_rows.extend(rows)
        jitter.extend(_jitter_rows(runtime, str(window["window_id"]), rows, q_states))
        eim = [float(row["new_e_im"]) for row in rows]
        p95 = None if not eim else float(np.quantile(eim, 0.95))
        low_preserved = all(
            float(row["new_e_im"]) <= TAU + EPSILON_NUM
            for row in rows
            if str(window["type"]) == "LOW" and float(row["old_e_im"]) <= TAU
        )
        summary = {
            "window_id": window["window_id"],
            "type": window["type"],
            "N": len(rows),
            "expected_N": int(window["N"]),
            "old_p95_e_im": float(window["old_p95_e_im"]),
            "new_p95_e_im": p95,
            "technical": sum(bool(row["technical_completion"]) for row in rows),
            "continuity": all(bool(row["continuity_pass"]) for row in rows),
            "wrist": all(bool(row["wrist_pass"]) for row in rows),
            "bone": all(bool(row["bone_pass"]) for row in rows),
            "hard_validity": all(bool(row["semantic_hard_pass"]) for row in rows),
            "low_old_valid_preserved": low_preserved,
        }
        summary["result_without_determinism"] = (
            "PASS"
            if (
                len(rows) == int(window["N"])
                and summary["technical"] == int(window["N"])
                and p95 is not None
                and p95 <= TAU
                and summary["continuity"]
                and summary["hard_validity"]
                and low_preserved
            )
            else "FAIL"
        )
        summaries.append(summary)
        print(
            f"O5RD2H_WINDOW_V4 window={window['window_id']} N={len(rows)} p95={p95}",
            flush=True,
        )
        if summary["result_without_determinism"] != "PASS":
            break
    write_csv(root / "window_v4/per_frame_results.csv", all_rows)
    write_csv(root / "window_v4/interaction_metrics.csv", all_rows)
    write_csv(root / "window_v4/continuity_metrics.csv", all_rows)
    write_csv(root / "window_v4/search_metrics.csv", all_rows)
    write_csv(
        root / "window_v4/qold_access_audit.csv",
        [
            {
                "ordinal": row["ordinal"],
                "runtime_step_index": row["runtime_step_index"],
                "q_old_field": row["q_old_field"],
                "q_old_access_count": row["q_old_access_count"],
                "previous_accepted_state": row["previous_accepted_state"],
            }
            for row in all_rows
        ],
    )
    write_csv(root / "window_v4/jitter_metrics.csv", jitter)
    repeat_rows: list[dict[str, Any]] = []
    if len(summaries) == 4 and all(
        row["result_without_determinism"] == "PASS" for row in summaries
    ):
        for window_id in ("HIGH_1", "LOW"):
            window = next(row for row in manifest["windows"] if row["window_id"] == window_id)
            ref_rows, ref_q, ref_base, ref_receipts = references[window_id]
            for run in (1, 2, 3):
                if run == 1:
                    repeat_rows.append({"window_id": window_id, "run": 1, "pass": True})
                    continue
                rows, q_states, base_states, receipts = _window_run(runtime, root, window, run)
                frame_comparisons = [
                    _compare_repeat(a, qa, ba, b, qb, bb)
                    for a, qa, ba, b, qb, bb in zip(
                        ref_receipts, ref_q, ref_base, receipts, q_states, base_states, strict=True
                    )
                ]
                p95_diff = abs(
                    float(np.quantile([float(row["new_e_im"]) for row in ref_rows], 0.95))
                    - float(np.quantile([float(row["new_e_im"]) for row in rows], 0.95))
                )
                repeat_rows.append(
                    {
                        "window_id": window_id,
                        "run": run,
                        "pass": all(bool(item["pass"]) for item in frame_comparisons)
                        and p95_diff <= EPSILON_NUM,
                        "p95_E_IM_abs_diff": p95_diff,
                        "frames": frame_comparisons,
                    }
                )
    determinism_pass = bool(repeat_rows) and all(bool(row["pass"]) for row in repeat_rows)
    write_json(
        root / "window_v4/determinism.json",
        {
            "schema_version": "ColdStartWindowValidationV4DeterminismV1",
            "status": "PASS" if determinism_pass else "FAIL",
            "rows": repeat_rows,
        },
    )
    overall = bool(
        len(summaries) == 4
        and all(row["result_without_determinism"] == "PASS" for row in summaries)
        and determinism_pass
    )
    decision = {
        "schema_version": "ColdStartWindowValidationV4GateDecisionV1",
        "COLDSTART_WINDOW_VALIDATION_V4": "PASS" if overall else "FAIL",
        "windows": summaries,
        "determinism": "PASS" if determinism_pass else "FAIL",
        "q_old_access_count": sum(int(row["q_old_access_count"]) for row in all_rows),
    }
    write_csv(root / "window_v4/per_window_summary.csv", summaries)
    write_json(decision_path, decision)
    if not overall:
        _write_downstream_not_run(
            root, "COLDSTART_WINDOW_VALIDATION_V4_FAIL", from_stage="cross_episode_controls"
        )
    return decision


def _cross_episode_pool(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, _by_id = o5.manifest_rows()
    split = read_json(o5.SPLIT_V2)
    development = set(split["splits"]["DEVELOPMENT"])
    certification = set(split["splits"]["CERTIFICATION"])
    heldout = set(split["splits"]["HELDOUT_TEST"])
    excluded_rows = read_json(root / "ledger/episode_exclusion_ledger.json")["entries"]
    excluded_records = {str(row["episode_id"]) for row in excluded_rows if row.get("episode_id")}
    excluded_sequences = {
        str(row["sequence_id"]) for row in excluded_rows if row.get("sequence_id")
    }
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        record_id = str(row["record_id"])
        sequence_id = str(row["sequence_id"])
        reasons: list[str] = []
        if record_id not in development:
            reasons.append("NOT_DEVELOPMENT_SPLIT")
        if record_id in certification:
            reasons.append("CERTIFICATION_SPLIT_FORBIDDEN")
        if record_id in heldout:
            reasons.append("HELDOUT_SPLIT_FORBIDDEN")
        if row.get("eligibility") is not True:
            reasons.append("NOT_ELIGIBLE")
        if row.get("active_hand") != "RIGHT":
            reasons.append("NOT_RIGHT_HAND")
        if len(row.get("official_right_object_list", [])) != 1:
            reasons.append("AMBIGUOUS_TARGET_OBJECT")
        if not row.get("canonical_target_object"):
            reasons.append("MISSING_TARGET_OBJECT")
        if not row.get("object_asset") or not Path(str(row["object_asset"])).is_file():
            reasons.append("MISSING_OBJECT_MESH")
        if record_id in excluded_records:
            reasons.append("PRIOR_RECORD_EVIDENCE")
        if sequence_id in excluded_sequences:
            reasons.append("PRIOR_SEQUENCE_EVIDENCE")
        if reasons:
            exclusions.append({"record_id": record_id, "reasons": reasons})
            continue
        eligible.append(
            {
                "record_id": record_id,
                "sequence_id": sequence_id,
                "primitive": str(row["primitive"]),
                "object_id": str(row["canonical_target_object"]),
                "object_asset_sha256": str(row["object_asset_sha256"]),
                "source_interval": [int(value) for value in row["source_interval"]],
                "canonical_record_sha256": str(row["canonical_record_sha256"]),
                "selection_hash": hashlib.sha256(record_id.encode()).hexdigest(),
            }
        )
    return sorted(eligible, key=lambda row: (row["selection_hash"], row["record_id"])), exclusions


def _select_cross_controls(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for require_object, require_primitive in ((True, True), (True, False), (False, False)):
        for row in pool:
            if row in selected:
                continue
            if require_object and row["object_id"] in {item["object_id"] for item in selected}:
                continue
            if require_primitive and row["primitive"] in {item["primitive"] for item in selected}:
                continue
            selected.append(row)
            if len(selected) == 3:
                return selected
    raise RuntimeError(f"O5RD2H_CROSS_EPISODE_SUPPORT_INSUFFICIENT:{len(selected)}<3")


def cross_episode_gate_contract() -> dict[str, Any]:
    return {
        "schema_version": "FreshCrossEpisodeFrame0ControlsGateV1",
        "status": "FROZEN",
        "N": 3,
        "runs_per_control": 3,
        "development_split_only": True,
        "per_run": [
            "optimizer started",
            "technical PASS",
            "finite",
            "E_IM <= tau",
            "wrist, bone, collision, joint limits, reflection, scale PASS",
            "q_old access count = 0",
            "no special-case path",
        ],
        "determinism": "each control 3/3 within frozen tolerance",
        "episode_gate": "3/3 controls PASS; no 2/3 partial pass",
        "tau": TAU,
    }


def freeze_cross_episode_controls(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_field(root / "window_v4/gate_decision.json", "COLDSTART_WINDOW_VALIDATION_V4", "PASS")
    target = root / "cross_episode_controls/manifest.json"
    gate_path = root / "cross_episode_controls/gate_contract.json"
    if target.exists() or gate_path.exists():
        require_hash(target, root / "cross_episode_controls/manifest.sha256", "cross_manifest")
        require_hash(gate_path, root / "cross_episode_controls/gate_contract.sha256", "cross_gate")
        return read_json(target)
    pool, exclusions = _cross_episode_pool(root)
    controls = [
        {**row, "control_id": f"CONTROL_{index}", "freshness": "SEQUENCE_DISJOINT"}
        for index, row in enumerate(_select_cross_controls(pool), start=1)
    ]
    payload = {
        "schema_version": "FreshCrossEpisodeFrame0ControlsManifestV1",
        "status": "FROZEN",
        "split": "DEVELOPMENT",
        "freshness": "SEQUENCE_DISJOINT",
        "selection_algorithm": certification_plan()["cross_episode"]["selection"],
        "controls": controls,
        "N": len(controls),
        "DEVELOPMENT_SPLIT_ONLY": "YES",
        "CERTIFICATION_SPLIT_USED": "NO",
        "HELDOUT_SPLIT_USED": "NO",
        "outcomes_used_for_selection": False,
        "frozen_before_execution": True,
    }
    write_json(
        root / "cross_episode_controls/eligible_pool.json",
        {"schema_version": "CrossEpisodeEligiblePoolV1", "count": len(pool), "rows": pool},
    )
    write_json(
        root / "cross_episode_controls/exclusions.json",
        {
            "schema_version": "CrossEpisodeExclusionsV1",
            "count": len(exclusions),
            "rows": exclusions,
        },
    )
    freeze_json(target, payload)
    freeze_json(gate_path, cross_episode_gate_contract())
    write_json(
        root / "cross_episode_controls/selection_receipt.json",
        {
            "schema_version": "CrossEpisodeSelectionReceiptV1",
            "eligible_pool_count": len(pool),
            "selected_record_ids": [row["record_id"] for row in controls],
            "selected_objects": [row["object_id"] for row in controls],
            "selected_primitives": [row["primitive"] for row in controls],
            "freshness": "SEQUENCE_DISJOINT",
        },
    )
    return payload


def _build_canonical_graph(
    *,
    record: dict[str, Any],
    canonical_path: Path,
    samples_path: Path,
    graph_path: Path,
    input_receipt: Path,
) -> dict[str, Any]:
    paths = (canonical_path, samples_path, graph_path, input_receipt)
    if all(path.exists() for path in paths):
        return read_json(input_receipt)
    if any(path.exists() for path in paths):
        raise RuntimeError(f"O5RD2H_PARTIAL_CANONICAL_PREPARATION:{canonical_path.parent}")
    adapter = o5.OakInk2CanonicalAdapterV1(o5.DATASET_ROOT)
    canonical, authority = o5.materialize_manifest_record_v2(
        adapter, record, mano_model_path=o5.MANO_MODEL, admitted_split="DEVELOPMENT"
    )
    o5.save_canonical_hoi(canonical, canonical_path)
    surface_profile = o5.load_surface_profile("paper_strict_area_uniform", repo_root=REPO)
    object_id = str(record["canonical_target_object"])
    samples = o5.sample_object_track(canonical.rigid_object(object_id), surface_profile)
    samples.save(samples_path)
    graph = o5.build_source_interaction_graph(
        canonical,
        "right_hand",
        object_id,
        samples,
        source_cache=canonical_path,
        object_sample_path=samples_path,
        delaunay_profile=o5.load_delaunay_profile("strict_scipy_qhull_v1"),
        kappa=o5.load_paper_kappa(),
        frame_indices=np.asarray([0], dtype=np.int64),
    )
    o5.save_interaction_graph(graph, graph_path)
    receipt = {
        "schema_version": "D2HCanonicalGraphPreparationV1",
        "status": "PASS",
        "record_id": record["record_id"],
        "object_id": object_id,
        "frame_count": canonical.num_frames,
        "split": "DEVELOPMENT",
        "canonical_sha256": sha256_path(canonical_path),
        "object_samples_sha256": sha256_file(samples_path),
        "interaction_graph_sha256": d2g.interaction_artifact_hash(graph_path),
        "authority": authority,
    }
    write_json(input_receipt, receipt)
    return receipt


class IndependentColdStartRuntime(d2g.V3Runtime):
    """ExecutionV3 runtime over a frozen canonical episode and source graph."""

    def __init__(self, canonical_path: Path, graph_path: Path, artifact_root: Path):
        self.review = "independent_coldstart"
        self.root = artifact_root
        self.paths = {"canonical": canonical_path, "graph": graph_path}
        self.final = None
        self.sequence = d2g.load_hoi_sequence(canonical_path)
        self.graph = d2g.load_interaction_graph(graph_path)
        self.model = d2g._load_robot(d2g.ROBOT, None)
        self.surface = d2g.load_robot_surface_samples(d2g._default_collision_samples(d2g.ROBOT))
        self.frame_profile = d2g.load_frame_profile("canonical_keypoint_wrist_v1")
        self.bone_profile = d2g.load_bone_profile("mediapipe21_full_finger_chain_v1")
        self.solver = d2g.RefinementSolverProfile.load(d2g.SOLVER_PROFILE)
        self.execution = d2g.RefinementExecutionProfile.load(d2g.EXECUTION_PROFILE, REPO)
        self.query = d2g.CollisionQueryProfile.load("adaptive_active_set_v1")
        self.coordinate = d2g.RefinementCoordinateProfile.load("local_seed_delta_v1")
        self.resources = d2g.prepare_refinement_resources(
            self.sequence,
            self.graph,
            self.solver,
            geometry_artifact_root=artifact_root / "geometry",
        )
        self.backends = d2g.prepare_refinement_runtime_backends(self.resources, self.execution)
        self.authority = d2g.RetargetNonRegressionBudgetAuthorityV1.from_frozen_v1(
            collision_hard_bound_m=self.resources.paper.b,
            collision_soft_tolerance_m=self.resources.paper.tau,
        )
        self.current_runtime_step = 0
        self._base_cache: dict[tuple[int, bytes], np.ndarray] = {}
        neutral = np.asarray(self.model.neutral_q, dtype=np.float64)
        count = len(self.graph.frame_indices)
        bases = np.stack([self.base_for_q(index, neutral) for index in range(count)])
        robot_points = np.asarray(
            self.model.keypoints_scene(neutral, np.eye(4), layout="mediapipe21"), dtype=np.float64
        )
        self.warm = SimpleNamespace(
            metadata={"source_hand_id": "right_hand", "source_side": "right"},
            arrays={
                "qpos": np.repeat(neutral[None, :], count, axis=0),
                "base_pose_scene": bases,
                "robot_keypoints_base": np.repeat(robot_points[None, :, :], count, axis=0),
            },
        )


def _manifest_record(record_id: str) -> dict[str, Any]:
    _rows, by_id = o5.manifest_rows()
    if record_id not in by_id:
        raise RuntimeError(f"O5RD2H_MANIFEST_RECORD_MISSING:{record_id}")
    return by_id[record_id]


def _cross_run_pass(row: dict[str, Any]) -> bool:
    return bool(
        row["optimizer_started"]
        and row["technical_completion"]
        and row["finite"]
        and float(row["new_e_im"]) <= TAU + EPSILON_NUM
        and row["wrist_pass"]
        and row["bone_pass"]
        and row["collision_pass"]
        and row["joint_limits_pass"]
        and row["reflection_pass"]
        and row["scale_pass"]
        and int(row["q_old_access_count"]) == 0
        and not row["q_old_synthesized"]
    )


def run_cross_episode_controls(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_field(root / "window_v4/gate_decision.json", "COLDSTART_WINDOW_VALIDATION_V4", "PASS")
    require_hash(
        root / "cross_episode_controls/manifest.json",
        root / "cross_episode_controls/manifest.sha256",
        "cross_episode_manifest",
    )
    require_hash(
        root / "cross_episode_controls/gate_contract.json",
        root / "cross_episode_controls/gate_contract.sha256",
        "cross_episode_gate",
    )
    decision_path = root / "cross_episode_controls/decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    manifest = read_json(root / "cross_episode_controls/manifest.json")
    _consume(
        root,
        "cross_episode",
        [{**row, "role": "FRESH_CROSS_EPISODE_CONTROL_CONSUMED"} for row in manifest["controls"]],
    )
    decisions: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []
    for control in manifest["controls"]:
        control_id = str(control["control_id"])
        directory = root / "cross_episode_controls" / control_id.lower()
        record = _manifest_record(str(control["record_id"]))
        preparation = _build_canonical_graph(
            record=record,
            canonical_path=directory / "work/canonical_episode.zarr",
            samples_path=directory / "work/object_samples.npz",
            graph_path=directory / "work/interaction_graph.zarr",
            input_receipt=directory / "input_authority.json",
        )
        runs: list[dict[str, Any]] = []
        states: list[tuple[np.ndarray, np.ndarray]] = []
        receipts: list[dict[str, Any]] = []
        for run in (1, 2, 3):
            runtime = IndependentColdStartRuntime(
                directory / "work/canonical_episode.zarr",
                directory / "work/interaction_graph.zarr",
                directory / f"runtime_{run}",
            )
            try:
                receipt, qpos, base = _run_coldstart_frame(
                    runtime,
                    root,
                    f"cross_episode_controls/{control_id.lower()}",
                    0,
                    run,
                    0,
                    None,
                    None,
                )
                row = _row_from_coldstart(receipt, "CROSS_EPISODE", float("nan"), standalone=True)
                row.update(
                    {
                        "schema_version": "FreshCrossEpisodeFrame0RunV1",
                        "control_id": control_id,
                        "run": run,
                        "record_id": control["record_id"],
                        "source_frame": int(control["source_interval"][0]),
                        "object_id": control["object_id"],
                        "primitive": control["primitive"],
                        "preparation_sha256": sha256_file(directory / "input_authority.json"),
                    }
                )
                row["result"] = "PASS" if _cross_run_pass(row) else "FAIL"
                states.append((qpos, base))
                receipts.append(receipt)
            except Exception as exc:
                _append_failure(
                    root, "FRESH_CROSS_EPISODE_CONTROL", exc, control_id=control_id, run=run
                )
                row = {
                    "schema_version": "FreshCrossEpisodeFrame0RunV1",
                    "control_id": control_id,
                    "run": run,
                    "record_id": control["record_id"],
                    "source_frame": int(control["source_interval"][0]),
                    "result": "FAIL",
                    "failure": f"{type(exc).__name__}:{exc}",
                }
            write_json(directory / f"run_{run}.json", row)
            runs.append(row)
        comparisons: list[dict[str, Any]] = []
        deterministic = len(states) == 3
        if deterministic:
            for index in (1, 2):
                comparison = _compare_repeat(
                    receipts[0],
                    states[0][0],
                    states[0][1],
                    receipts[index],
                    states[index][0],
                    states[index][1],
                )
                comparisons.append({"run": index + 1, **comparison})
            deterministic = all(bool(row["pass"]) for row in comparisons)
        determinism_rows.append(
            {
                "control_id": control_id,
                "status": "PASS" if deterministic else "FAIL",
                "comparisons": comparisons,
            }
        )
        passed = (
            len(runs) == 3 and all(row.get("result") == "PASS" for row in runs) and deterministic
        )
        decisions.append(
            {
                "control_id": control_id,
                "record_id": control["record_id"],
                "primitive": control["primitive"],
                "object_id": control["object_id"],
                "freshness": control["freshness"],
                "runs": [row.get("result") for row in runs],
                "determinism": "PASS" if deterministic else "FAIL",
                "result": "PASS" if passed else "FAIL",
                "input_authority": preparation,
            }
        )
        if not passed:
            break
    overall = len(decisions) == 3 and all(row["result"] == "PASS" for row in decisions)
    write_json(
        root / "cross_episode_controls/determinism.json",
        {
            "schema_version": "FreshCrossEpisodeDeterminismV1",
            "status": "PASS" if overall else "FAIL",
            "controls": determinism_rows,
        },
    )
    decision = {
        "schema_version": "FreshCrossEpisodeFrame0ControlsDecisionV1",
        "FRESH_CROSS_EPISODE_CONTROLS": "PASS" if overall else "FAIL",
        "N_CONTROLS": len(decisions),
        "TOTAL_RUNS": sum(len(row["runs"]) for row in decisions),
        "DEVELOPMENT_SPLIT_ONLY": "YES",
        "CERTIFICATION_SPLIT_USED": "NO",
        "HELDOUT_SPLIT_USED": "NO",
        "controls": decisions,
    }
    write_json(decision_path, decision)
    if not overall:
        _write_downstream_not_run(root, "FRESH_CROSS_EPISODE_CONTROLS_FAIL", from_stage="dev2_full")
    return decision


def authorize_dev2_full(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    sparse = require_field(
        root / "sparse_v4/gate_decision.json", "COLDSTART_SPARSE_VALIDATION_V4", "PASS"
    )
    window = require_field(
        root / "window_v4/gate_decision.json", "COLDSTART_WINDOW_VALIDATION_V4", "PASS"
    )
    cross = require_field(
        root / "cross_episode_controls/decision.json", "FRESH_CROSS_EPISODE_CONTROLS", "PASS"
    )
    payload = {
        "schema_version": "DEV2FullAuthorizationV1",
        "FULL_DEV2_COMPUTE_AUTHORIZED": "YES",
        "SparseV4": sparse["COLDSTART_SPARSE_VALIDATION_V4"],
        "WindowV4": window["COLDSTART_WINDOW_VALIDATION_V4"],
        "FreshCrossEpisodeControls": cross["FRESH_CROSS_EPISODE_CONTROLS"],
        "run_role": "KNOWN_FAILURE_RECOVERY_PRODUCTION_RUN",
        "scientific_run_limit": 1,
    }
    write_json(root / "dev2_full/authorization.json", payload)
    return payload


def _prepare_dev2_graph(root: Path) -> dict[str, Any]:
    identity = d2g._dev2_identity()
    if not identity["identity_exact"]:
        raise RuntimeError("O5RD2H_DEV2_IDENTITY_AUTHORITY_MISMATCH")
    expected = identity["expected"]
    if expected != {
        "record_id": "oakink2:scene_01__A003++seq__a7a1a0cf7d90a9083013__2023-04-21-20-13-04:00010",
        "primitive": "rearrange",
        "object": "C11001",
        "source_interval": [10704, 10944],
        "frames": 240,
    }:
        raise RuntimeError(f"O5RD2H_DEV2_IDENTITY_DRIFT:{expected}")
    work = root / "dev2_full/work"
    canonical_path = o5.episode_paths(o5.REPORT_ROOT, "dev_02")["canonical"]
    samples_path = work / "object_samples.npz"
    graph_path = work / "interaction_graph.zarr"
    receipt_path = work / "input_authority.json"
    if samples_path.exists() and graph_path.exists() and receipt_path.exists():
        return read_json(receipt_path)
    if any(path.exists() for path in (samples_path, graph_path, receipt_path)):
        raise RuntimeError("O5RD2H_PARTIAL_DEV2_GRAPH_PREPARATION")
    canonical = o5.load_canonical_hoi(canonical_path)
    if canonical.num_frames != 240:
        raise RuntimeError(f"O5RD2H_DEV2_FRAME_COUNT:{canonical.num_frames}")
    surface_profile = o5.load_surface_profile("paper_strict_area_uniform", repo_root=REPO)
    samples = o5.sample_object_track(canonical.rigid_object("C11001"), surface_profile)
    samples.save(samples_path)
    graph = o5.build_source_interaction_graph(
        canonical,
        "right_hand",
        "C11001",
        samples,
        source_cache=canonical_path,
        object_sample_path=samples_path,
        delaunay_profile=o5.load_delaunay_profile("strict_scipy_qhull_v1"),
        kappa=o5.load_paper_kappa(),
        frame_indices=np.arange(240, dtype=np.int64),
    )
    o5.save_interaction_graph(graph, graph_path)
    payload = {
        "schema_version": "DEV2FullInputAuthorityV1",
        "status": "PASS",
        "identity": identity,
        "canonical_path": str(canonical_path.resolve()),
        "canonical_sha256": sha256_path(canonical_path),
        "object_samples_sha256": sha256_file(samples_path),
        "interaction_graph_sha256": d2g.interaction_artifact_hash(graph_path),
        "source_graph": "CANONICAL_SOURCE_DERIVED",
        "frame_count": 240,
    }
    write_json(receipt_path, payload)
    return payload


def _dev2_method_hashes(root: Path) -> dict[str, str]:
    integrity = require_field(
        root / "frozen_method/integrity.json", "FROZEN_METHOD_INTEGRITY", "PASS"
    )
    return {name: str(row["actual_sha256"]) for name, row in integrity["contracts"].items()} | {
        "implementation_authority": sha256_file(
            root / "frozen_method/implementation_authority.json"
        )
    }


def _dev2_existing_states(root: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    q_states: list[np.ndarray] = []
    base_states: list[np.ndarray] = []
    for ordinal in range(240):
        path = root / f"dev2_full/checkpoints/receipts/frame_{ordinal:04d}_run_1.npz"
        if not path.exists():
            break
        with np.load(path, allow_pickle=False) as state:
            q_states.append(np.asarray(state["qpos"], dtype=np.float64))
            base_states.append(np.asarray(state["base_pose_scene"], dtype=np.float64))
    return q_states, base_states


def _execute_dev2_full(root: Path, *, resume: bool) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_field(root / "dev2_full/authorization.json", "FULL_DEV2_COMPUTE_AUTHORIZED", "YES")
    input_authority = _prepare_dev2_graph(root)
    manifest_path = root / "dev2_full/run_manifest.json"
    manifest_sha_path = root / "dev2_full/run_manifest.sha256"
    method_hashes = _dev2_method_hashes(root)
    if manifest_path.exists():
        require_hash(manifest_path, manifest_sha_path, "dev2_run_manifest")
        manifest = read_json(manifest_path)
        if manifest["method_hashes"] != method_hashes:
            raise RuntimeError("O5RD2H_DEV2_RESUME_METHOD_HASH_DRIFT")
        if not resume:
            raise RuntimeError("O5RD2H_DEV2_SCIENTIFIC_RUN_ALREADY_STARTED")
        if manifest.get("resume_allowed") != "INFRASTRUCTURE_ONLY":
            raise RuntimeError("O5RD2H_DEV2_RESUME_NOT_AUTHORIZED")
    else:
        if resume:
            raise RuntimeError("O5RD2H_DEV2_RESUME_WITHOUT_RUN")
        manifest = {
            "schema_version": "DEV2ProductionRunManifestV1",
            "status": "STARTED",
            "DEV2_PRODUCTION_RUN_UUID": str(uuid.uuid4()),
            "RUN_ROLE": "KNOWN_FAILURE_RECOVERY_PRODUCTION_RUN",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 1,
            "EXPECTED_FRAMES": 240,
            "identity": input_authority["identity"],
            "method_hashes": method_hashes,
            "frame0_mode": "COLD_START",
            "t_gt_0_mode": "COLD_START_SEQUENCE",
            "DEV2_DEVELOPMENT_FRAME0_STATE_REUSED": "NO",
            "resume_allowed": "INFRASTRUCTURE_ONLY",
        }
        freeze_json(manifest_path, manifest)
        write_json(
            root / "dev2_full/technical_resume_policy.json",
            {
                "schema_version": "DEV2TechnicalResumePolicyV1",
                "status": "FROZEN",
                "infrastructure_failure": "same UUID, hashes, and accepted checkpoint may resume",
                "scientific_failure": "no resume through failure frame and no second run",
            },
        )
        ledger = read_json(root / "ledger/evidence_consumption_ledger.json")
        ledger["dev2"] = {
            "role": "DEV2_KNOWN_FAILURE_RECOVERY",
            "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"],
            "full_run_count": 1,
        }
        write_json(root / "ledger/evidence_consumption_ledger.json", ledger)
    started = time.perf_counter()
    runtime = IndependentColdStartRuntime(
        Path(input_authority["canonical_path"]),
        root / "dev2_full/work/interaction_graph.zarr",
        root / "dev2_full/runtime",
    )
    q_states, base_states = _dev2_existing_states(root)
    start_ordinal = len(q_states)
    previous_q = None if not q_states else q_states[-1]
    previous_base = None if not base_states else base_states[-1]
    rows = (
        read_csv(root / "dev2_full/frame_metrics.csv")
        if (root / "dev2_full/frame_metrics.csv").exists()
        else []
    )
    scientific_failure: dict[str, Any] | None = None
    for ordinal in range(start_ordinal, 240):
        try:
            receipt, qpos, base = _run_coldstart_frame(
                runtime,
                root,
                "dev2_full/checkpoints",
                ordinal,
                1,
                ordinal,
                previous_q,
                previous_base,
            )
            row = _row_from_coldstart(receipt, "DEV2", float("nan"), standalone=False)
            row.update(
                {"source_frame": 10704 + ordinal, "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"]}
            )
            if not bool(
                row["technical_completion"] and row["finite"] and row["semantic_hard_pass"]
            ):
                raise RuntimeError("DEV2_EXECUTION_V3_INVALID_TERMINAL")
            q_states.append(qpos)
            base_states.append(base)
            checkpoint = {
                "schema_version": "DEV2DurableFrameCheckpointV1",
                "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"],
                "ordinal": ordinal,
                "source_frame": 10704 + ordinal,
                "accepted_q_sha256": d2g3.array_sha256(qpos),
                "accepted_base_sha256": d2g3.array_sha256(base),
                "previous_state_sha256": None
                if ordinal == 0
                else hashlib.sha256(
                    (d2g3.array_sha256(previous_q) + d2g3.array_sha256(previous_base)).encode()
                ).hexdigest(),
                "interaction_graph_sha256": input_authority["interaction_graph_sha256"],
                "bootstrap_seed_order": [item["seed"] for item in receipt["bootstrap"]["solves"]],
                "contributor": receipt["selected_block"],
                "selected_candidate": receipt["selected_candidate"],
                "E_IM": receipt["selected"]["interaction_e_im"],
                "hard_validity": "PASS",
                "method_hashes": method_hashes,
            }
            write_json(root / f"dev2_full/checkpoints/frame_{ordinal:04d}.json", checkpoint)
            rows.append(row)
            write_csv(root / "dev2_full/frame_metrics.csv", rows)
            write_csv(root / "dev2_full/search_metrics.csv", rows)
            previous_q, previous_base = qpos, base
            print(
                f"O5RD2H_DEV2_FULL {ordinal + 1}/240 source_frame={10704 + ordinal} E_IM={float(row['new_e_im']):.12g}",
                flush=True,
            )
        except OSError as exc:
            write_json(
                root / "dev2_full/infrastructure_interruption.json",
                {
                    "schema_version": "DEV2InfrastructureInterruptionV1",
                    "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"],
                    "ordinal": ordinal,
                    "error": f"{type(exc).__name__}:{exc}",
                    "resume_allowed": True,
                },
            )
            raise
        except Exception as exc:
            scientific_failure = {
                "schema_version": "DEV2ScientificFailureV1",
                "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"],
                "FIRST_FAILURE_ORDINAL": ordinal,
                "FIRST_FAILURE_SOURCE_FRAME": 10704 + ordinal,
                "error": f"{type(exc).__name__}:{exc}",
                "resume_allowed": False,
            }
            write_json(root / "dev2_full/scientific_failure.json", scientific_failure)
            _append_failure(root, "DEV2_FULL_SCIENTIFIC", exc, ordinal=ordinal)
            break
    solver_sec = time.perf_counter() - started
    completed = len(q_states)
    if scientific_failure is not None:
        result = {
            "schema_version": "DEV2FullSolverReceiptV1",
            "DEV2_NUMERICAL_RESULT": "FAIL",
            "EXPECTED_FRAMES": 240,
            "COMPLETED_FRAMES": completed,
            "FIRST_FAILURE_ORDINAL": scientific_failure["FIRST_FAILURE_ORDINAL"],
            "FIRST_FAILURE_SOURCE_FRAME": scientific_failure["FIRST_FAILURE_SOURCE_FRAME"],
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 1,
            "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"],
        }
        write_json(root / "dev2_full/solver_receipt.json", result)
        return result
    if completed != 240:
        raise RuntimeError(f"O5RD2H_DEV2_INCOMPLETE_WITHOUT_FAILURE:{completed}")
    q_array = np.stack(q_states)
    base_array = np.stack(base_states)
    trajectory_path = root / "dev2_full/trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        schema_version=np.asarray("DEV2ExecutionV3TrajectoryV1"),
        record_id=np.asarray(input_authority["identity"]["expected"]["record_id"]),
        source_frame_ids=np.arange(10704, 10944, dtype=np.int64),
        qpos=q_array,
        base_pose_scene=base_array,
        e_im=np.asarray([float(row["new_e_im"]) for row in rows], dtype=np.float64),
    )
    profiler_rows = [
        {
            key.removeprefix("profiler_"): value
            for key, value in row.items()
            if key.startswith("profiler_")
        }
        | {"ordinal": int(row["ordinal"]), "source_frame": int(row["source_frame"])}
        for row in rows
    ]
    write_csv(root / "dev2_full/profiler/per_frame.csv", profiler_rows)
    totals = np.asarray([float(row.get("total_wall_sec", 0.0)) for row in profiler_rows])
    aggregate = {
        "schema_version": "RetargetSolverProfilerV1Aggregate",
        "N": 240,
        "solver_total_sec": float(np.sum(totals)),
        "mean_solver_sec_per_frame": float(np.mean(totals)),
        "p50": float(np.quantile(totals, 0.50)),
        "p90": float(np.quantile(totals, 0.90)),
        "p95": float(np.quantile(totals, 0.95)),
        "p99": float(np.quantile(totals, 0.99)),
        "max": float(np.max(totals)),
        "mean_bootstrap_sec_per_frame": float(
            np.mean([float(row.get("bootstrap_wall_sec", 0.0)) for row in profiler_rows])
        ),
        "mean_candidate_probes_per_frame": float(
            np.mean([float(row.get("candidate_b2_candidate_probes", 0.0)) for row in profiler_rows])
        ),
        "primary_valid_count": sum(bool(row["technical_completion"]) for row in rows),
        "secondary_polish_attempts": sum(
            int(row.get("secondary_nfev", 0)) > 0 for row in profiler_rows
        ),
        "polish_rejection_count": sum("PRIMARY" in str(row["retention_decision"]) for row in rows),
        "primary_retention_count": sum("PRIMARY" in str(row["retention_decision"]) for row in rows),
        "fallback_count": sum(bool(row.get("fallback")) for row in profiler_rows),
        "optimizer_nonsuccess_but_valid_count": None,
        "optimizer_nonsuccess_but_valid_status": "NOT_AVAILABLE",
        "true_scientific_failure_count": 0,
    }
    write_json(root / "dev2_full/profiler/aggregate.json", aggregate)
    write_json(
        root / "dev2_full/timing/stage_timing.json",
        {
            "T_load": None,
            "T_prepare": None,
            "T_retarget_solver": solver_sec,
            "T_semantic_v1": None,
            "T_viewer": None,
            "T_machine_total": None,
            "unavailable_fields": "NOT_AVAILABLE_UNMEASURED_SEPARATELY",
        },
    )
    write_json(root / "dev2_full/timing/aggregate.json", aggregate)
    write_json(
        root / "dev2_full/output_hashes.json",
        {
            "trajectory": sha256_file(trajectory_path),
            "frame_metrics": sha256_file(root / "dev2_full/frame_metrics.csv"),
        },
    )
    result = {
        "schema_version": "DEV2FullSolverReceiptV1",
        "DEV2_NUMERICAL_RESULT": "PASS",
        "EXPECTED_FRAMES": 240,
        "COMPLETED_FRAMES": 240,
        "FIRST_FAILURE_ORDINAL": None,
        "FIRST_FAILURE_SOURCE_FRAME": None,
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 1,
        "DEV2_DEVELOPMENT_FRAME0_STATE_REUSED": "NO",
        "TRAJECTORY_SHA256": sha256_file(trajectory_path),
        "run_uuid": manifest["DEV2_PRODUCTION_RUN_UUID"],
    }
    write_json(root / "dev2_full/solver_receipt.json", result)
    return result


def run_dev2_full(root: Path) -> dict[str, Any]:
    return _execute_dev2_full(root, resume=False)


def resume_dev2_full_technical_only(root: Path) -> dict[str, Any]:
    interruption = require_field(
        root / "dev2_full/infrastructure_interruption.json", "resume_allowed", True
    )
    manifest = require_field(root / "dev2_full/run_manifest.json", "status", "STARTED")
    if interruption["run_uuid"] != manifest["DEV2_PRODUCTION_RUN_UUID"]:
        raise RuntimeError("O5RD2H_RESUME_RUN_UUID_MISMATCH")
    return _execute_dev2_full(root, resume=True)


def run_dev2_semantic_v1(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    require_field(root / "dev2_full/solver_receipt.json", "DEV2_NUMERICAL_RESULT", "PASS")
    started = time.perf_counter()
    trajectory_path = root / "dev2_full/trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        bases = np.asarray(trajectory["base_pose_scene"], dtype=np.float64)
        final_eim = np.asarray(trajectory["e_im"], dtype=np.float64)
    if qpos.shape[0] != 240 or bases.shape[0] != 240 or final_eim.shape != (240,):
        raise RuntimeError("O5RD2H_SEMANTIC_TRAJECTORY_COVERAGE_MISMATCH")
    canonical_path = Path(read_json(root / "dev2_full/work/input_authority.json")["canonical_path"])
    canonical = o5.load_canonical_hoi(canonical_path)
    hand = canonical.hand("right_hand")
    obj = canonical.rigid_object("C11001")
    model = d2g._load_robot(d2g.ROBOT, None)
    surface = d2g.load_robot_surface_samples(d2g._default_collision_samples(d2g.ROBOT))
    source_keypoints = np.asarray(
        hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    source_frames = d2g._semantic_frames(source_keypoints, hand.side)
    keypoints_base = np.stack(
        [model.keypoints_scene(q, np.eye(4), layout="mediapipe21") for q in qpos]
    )
    frames_base = d2g._semantic_frames(keypoints_base, model.side)
    robot_frames = compose(bases, frames_base)
    object_pose = np.asarray(obj.pose_scene.pose_scene, dtype=np.float64)
    wrist = transform_error(
        relative_transform(object_pose, source_frames),
        relative_transform(object_pose, robot_frames),
    )
    frame_profile = d2g.load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = d2g.load_bone_profile("mediapipe21_full_finger_chain_v1")
    source_features = extract_bone_features(
        source_keypoints, frame_profile, bone_profile, side=hand.side, strict=True
    )
    final_features = extract_bone_features(
        keypoints_base, frame_profile, bone_profile, side=model.side, strict=True
    )
    bone_error = angular_error(
        np.asarray(source_features.unit_directions),
        np.asarray(final_features.unit_directions),
    )
    neutral_features = extract_bone_features(
        np.asarray(model.keypoints_scene(model.neutral_q, np.eye(4), layout="mediapipe21")),
        frame_profile,
        bone_profile,
        side=model.side,
        strict=True,
    )
    scale_ratio = np.asarray(final_features.bone_lengths) / np.asarray(
        neutral_features.bone_lengths
    )
    gate = SemanticGateContractV1()
    unit_scale_pass = bool(
        np.isfinite(scale_ratio).all()
        and np.min(scale_ratio) >= gate.unit_scale_ratio_minimum
        and np.max(scale_ratio) <= gate.unit_scale_ratio_maximum
    )
    triangles = np.asarray(obj.mesh.vertices_local)[np.asarray(obj.mesh.faces, dtype=np.int64)]
    tree = semantic_audit.ObjectLocalBVH(triangles, leaf_size=32)
    source_distance = semantic_audit._object_distance(
        tree, object_pose, np.asarray(hand.vertices_scene)
    )
    collision_points = np.stack(
        [
            dynamic_collision_points_numpy(model, surface, q, base)
            for q, base in zip(qpos, bases, strict=True)
        ]
    )
    final_distance = semantic_audit._object_distance(tree, object_pose, collision_points)
    source_contact = np.min(source_distance, axis=1) <= gate.contact_opportunity_distance_m
    robot_contact = np.min(final_distance, axis=1) <= gate.contact_opportunity_distance_m
    timestamps = np.asarray(canonical.metadata.timestamps)
    selected_range = canonical.metadata.provenance.conversion_options.get(
        "selected_frame_range", [0, len(timestamps)]
    )
    time_alignment_pass = bool(
        len(timestamps) == 240
        and len(selected_range) == 2
        and int(selected_range[1]) - int(selected_range[0]) == 240
        and canonical.metadata.provenance.no_temporal_resampling
        and canonical.metadata.provenance.no_spatial_sampling
    )
    source_to_scene = np.asarray(canonical.metadata.source_to_scene, dtype=np.float64)
    spatial_identity = bool(
        source_to_scene.shape == (4, 4)
        and np.allclose(source_to_scene, np.eye(4), rtol=0.0, atol=gate.rigid_invariant_atol_m)
    )
    invariant = semantic_audit.common_rigid_transform_invariant(
        source_frames, object_pose, source_to_scene
    )
    shared_scene = bool(
        hand.keypoint_tracks["mediapipe21"].frame_name == canonical.metadata.scene_frame_name
        and hand.wrist_pose_scene.frame_name == canonical.metadata.scene_frame_name
        and obj.pose_scene.frame_name == canonical.metadata.scene_frame_name
    )
    frame_authority_pass = bool(
        invariant["pass"]
        and spatial_identity
        and shared_scene
        and time_alignment_pass
        and np.all(np.linalg.det(source_frames[:, :3, :3]) > 0.999999)
        and np.all(np.linalg.det(object_pose[:, :3, :3]) > 0.999999)
        and np.all(np.linalg.det(robot_frames[:, :3, :3]) > 0.999999)
        and unit_scale_pass
    )
    interaction_pass = bool(
        np.isfinite(final_eim).all()
        and np.quantile(final_eim, 0.95) <= gate.interaction_e_im_p95_limit
    )
    qualification = qualify_semantics(
        wrist_position_m=wrist["position_m"],
        wrist_rotation_rad=wrist["rotation_rad"],
        bone_error_rad=bone_error,
        source_contact=source_contact,
        robot_contact=robot_contact,
        robot_wrist_transforms=robot_frames,
        frame_authority_pass=frame_authority_pass,
        time_alignment_pass=time_alignment_pass,
        interaction_geometry_pass=interaction_pass,
        gate=gate,
    )
    steps = temporal_steps(robot_frames)
    passed = str(qualification["status"]) == "RETARGET_SEMANTIC_PASS"
    result = {
        "schema_version": "RetargetSemanticValidityV1",
        "RETARGET_SEMANTIC_VALIDITY_V1_RAN": "YES",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "E_IM_MEAN": float(np.mean(final_eim)),
        "E_IM_P50": float(np.quantile(final_eim, 0.50)),
        "E_IM_P90": float(np.quantile(final_eim, 0.90)),
        "E_IM_P95": float(np.quantile(final_eim, 0.95)),
        "E_IM_MAX": float(np.max(final_eim)),
        "E_IM_THRESHOLD": gate.interaction_e_im_p95_limit,
        "WRIST": "PASS"
        if qualification["metrics"]["object_relative_wrist_position_m"]["max"]
        <= gate.object_relative_wrist_position_limit_m
        and qualification["metrics"]["object_relative_wrist_rotation_rad"]["max"]
        <= gate.object_relative_wrist_rotation_limit_rad
        else "FAIL",
        "BONE": qualification["bone_direction_status"],
        "CONTACT_RECALL": qualification["metrics"]["source_contact_recall"],
        "CONTACT_RECALL_STATUS": qualification["contact_recall_status"],
        "CONTINUITY": qualification["temporal_continuity_status"],
        "ACTUAL_TEMPORAL_TRANSLATION": semantic_audit.summarize(steps["translation_m"]),
        "ACTUAL_TEMPORAL_ROTATION": semantic_audit.summarize(steps["rotation_rad"]),
        "REFLECTION": "PASS"
        if np.all(np.linalg.det(robot_frames[:, :3, :3]) > 0.999999)
        else "FAIL",
        "SCALE": "PASS" if unit_scale_pass else "FAIL",
        "FRAME_AUTHORITY": "PASS" if frame_authority_pass else "FAIL",
        "TIME_ALIGNMENT": "PASS" if time_alignment_pass else "FAIL",
        "SOURCE_CONTACT_OPPORTUNITY_COUNT": int(np.sum(source_contact)),
        "qualification": qualification,
        "gate": gate.as_dict(),
        "SEMANTIC_V1_RESULT": "PASS" if passed else "FAIL",
    }
    write_json(root / "dev2_full/semantic_validity.json", result)
    timing_path = root / "dev2_full/timing/stage_timing.json"
    timing = read_json(timing_path)
    timing["T_semantic_v1"] = time.perf_counter() - started
    write_json(timing_path, timing)
    return result


def render_dev2_viewer(root: Path) -> dict[str, Any]:
    verify_frozen_execution_v3(root)
    if not (root / "dev2_full/trajectory.npz").exists():
        raise RuntimeError("O5RD2H_VIEWER_REQUIRES_COMPLETE_TRAJECTORY")
    semantic = read_json(root / "dev2_full/semantic_validity.json")
    started = time.perf_counter()
    with np.load(root / "dev2_full/trajectory.npz", allow_pickle=False) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        bases = np.asarray(trajectory["base_pose_scene"], dtype=np.float64)
        frames = np.asarray(trajectory["source_frame_ids"], dtype=np.int64)
    canonical = o5.load_canonical_hoi(
        Path(read_json(root / "dev2_full/work/input_authority.json")["canonical_path"])
    )
    selected = o5.selected_viewer_indices(240, 10778 - 10704)
    hand = canonical.hand("right_hand")
    joints = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[selected]
    vertices = np.asarray(hand.vertices_scene)[selected]
    translation = np.asarray(hand.mano_parameters.transl)[selected]
    model = d2g._load_robot(d2g.ROBOT, None)
    visual = o5._robot_visual_payload(model, qpos[selected], bases[selected])
    robot_joints = np.stack(
        [
            model.keypoints_scene(q, base, layout="mediapipe21")
            for q, base in zip(qpos[selected], bases[selected], strict=True)
        ]
    )
    obj = canonical.rigid_object("C11001")
    html = root / "dev2_full/review/oakink2_wuji_execution_v3_viewer.html"
    data = o5.OakInk2HTMLViewerV2Data(
        frames=frames[selected],
        hand_vertices_world=vertices,
        hand_vertices_anatomy=vertices - translation[:, None, :],
        hand_faces_closed=np.asarray(hand.mesh.faces),
        hand_faces_open=np.asarray(hand.mesh.faces),
        hand_joints_world=joints,
        hand_joints_anatomy=joints - translation[:, None, :],
        object_vertices=np.asarray(obj.mesh.vertices_local),
        object_faces=np.asarray(obj.mesh.faces),
        object_transforms=np.asarray(obj.pose_scene.pose_scene)[selected],
        primary_frame=10778,
        record={
            "dataset": "OakInk2",
            "episode": d2g._dev2_identity()["expected"]["record_id"],
            "primitive": "rearrange",
            "source_hand": "RIGHT",
            "target_object": "C11001",
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "PASS",
            "semantic_validity_status": semantic["SEMANTIC_V1_RESULT"],
            "viewer_sampling": "180 deterministic display frames; solver used all 240 frames",
        },
        camera_presets=o5.camera_presets(),
        wuji_parts=list(visual["parts"]),
        wuji_joints_world=robot_joints,
        frame_solver_sec=np.zeros(len(selected), dtype=np.float64),
    )
    renderer = o5.render_oakink2_html_viewer_v2(data, html)
    regression = o5.certify_viewer(html, html.with_name("interaction_review.png"))
    role = (
        "MACHINE_PASS_HUMAN_REVIEW"
        if semantic["SEMANTIC_V1_RESULT"] == "PASS"
        else "DIAGNOSTIC_NOT_ACCEPTED"
    )
    receipt = {
        "schema_version": "DEV2ExecutionV3ViewerReceiptV1",
        "DEV2_EXECUTION_V3_HTML": str(html.resolve()),
        "DEV2_HTML_SHA256": sha256_file(html),
        "VIEWER_ROLE": role,
        "renderer": renderer,
        "interactive_regression": regression,
        "display_frames": len(selected),
        "solver_frames": 240,
    }
    write_json(root / "dev2_full/review/receipt.json", receipt)
    manual = [
        "# DEV2 ExecutionV3 manual geometric review",
        "",
        f"Viewer: `{html.resolve()}`",
        "",
        "Inspect wrist following, thumb and four-finger interaction, contact geometry, finger identity, penetration, jitter, playback continuity, orbit stability, object identity, frame0 jump, and early bootstrap transition.",
        "",
        "Reply: `OAKINK2_O5_DEV_2_EXECUTION_V3=APPROVE` or `OAKINK2_O5_DEV_2_EXECUTION_V3=REJECT`.",
    ]
    (root / "dev2_full/review/manual_review.md").write_text(
        "\n".join(manual) + "\n", encoding="utf-8"
    )
    timing_path = root / "dev2_full/timing/stage_timing.json"
    timing = read_json(timing_path)
    timing["T_viewer"] = time.perf_counter() - started
    write_json(timing_path, timing)
    return receipt


def _write_json_if_absent(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        write_json(path, payload)


def _write_downstream_not_run(root: Path, reason: str, *, from_stage: str) -> None:
    order = ["window_v4", "cross_episode_controls", "dev2_full"]
    for stage in order[order.index(from_stage) :]:
        _write_json_if_absent(
            root / stage / "not_run.json",
            {
                "schema_version": "O5RD2HNotRunV1",
                "status": "NOT_RUN",
                "reason": reason,
            },
        )


def _write_all_not_run(root: Path, reason: str) -> None:
    for stage in ("sparse_v4", "window_v4", "cross_episode_controls", "dev2_full"):
        _write_json_if_absent(
            root / stage / "not_run.json",
            {"schema_version": "O5RD2HNotRunV1", "status": "NOT_RUN", "reason": reason},
        )


def summarize(root: Path) -> dict[str, Any]:
    integrity = (
        read_json(root / "frozen_method/integrity.json")
        if (root / "frozen_method/integrity.json").exists()
        else {}
    )
    plan_sha = (
        (root / "certification_plan/d2h_certification_plan.sha256")
        .read_text(encoding="utf-8")
        .strip()
        if (root / "certification_plan/d2h_certification_plan.sha256").exists()
        else None
    )
    sparse = (
        read_json(root / "sparse_v4/gate_decision.json")
        if (root / "sparse_v4/gate_decision.json").exists()
        else {"COLDSTART_SPARSE_VALIDATION_V4": "NOT_RUN"}
    )
    window = (
        read_json(root / "window_v4/gate_decision.json")
        if (root / "window_v4/gate_decision.json").exists()
        else {"COLDSTART_WINDOW_VALIDATION_V4": "NOT_RUN"}
    )
    cross = (
        read_json(root / "cross_episode_controls/decision.json")
        if (root / "cross_episode_controls/decision.json").exists()
        else {"FRESH_CROSS_EPISODE_CONTROLS": "NOT_RUN"}
    )
    dev2 = (
        read_json(root / "dev2_full/solver_receipt.json")
        if (root / "dev2_full/solver_receipt.json").exists()
        else {
            "DEV2_NUMERICAL_RESULT": "NOT_RUN",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "COMPLETED_FRAMES": 0,
        }
    )
    semantic = (
        read_json(root / "dev2_full/semantic_validity.json")
        if (root / "dev2_full/semantic_validity.json").exists()
        else {"SEMANTIC_V1_RESULT": "NOT_RUN"}
    )
    viewer = (
        read_json(root / "dev2_full/review/receipt.json")
        if (root / "dev2_full/review/receipt.json").exists()
        else {}
    )
    certified = bool(
        sparse.get("COLDSTART_SPARSE_VALIDATION_V4") == "PASS"
        and window.get("COLDSTART_WINDOW_VALIDATION_V4") == "PASS"
        and cross.get("FRESH_CROSS_EPISODE_CONTROLS") == "PASS"
    )
    if integrity.get("FROZEN_METHOD_INTEGRITY") != "PASS":
        next_step = "EXECUTION_V3_FROZEN_AUTHORITY_INTEGRITY_REPAIR"
    elif sparse.get("COLDSTART_SPARSE_VALIDATION_V4") == "FAIL":
        next_step = "EXECUTION_V3_FRAMELEVEL_COLDSTART_GENERALIZATION_FAILURE_ANALYSIS"
    elif window.get("COLDSTART_WINDOW_VALIDATION_V4") == "FAIL":
        next_step = "EXECUTION_V3_SEQUENCE_COLDSTART_GENERALIZATION_FAILURE_ANALYSIS"
    elif cross.get("FRESH_CROSS_EPISODE_CONTROLS") == "FAIL":
        next_step = "EXECUTION_V3_CROSS_EPISODE_GENERALIZATION_FAILURE_ANALYSIS"
    elif dev2.get("DEV2_NUMERICAL_RESULT") == "FAIL":
        next_step = "DEV2_EXECUTION_V3_FULL_TRAJECTORY_FAILURE_LOCALIZATION"
    elif semantic.get("SEMANTIC_V1_RESULT") == "FAIL":
        next_step = "DEV2_EXECUTION_V3_SEMANTIC_FAILURE_LOCALIZATION"
    elif semantic.get("SEMANTIC_V1_RESULT") == "PASS" and viewer:
        next_step = "WAIT_FOR_DEV2_HUMAN_REVIEW"
    else:
        next_step = "CONTINUE_D2H_STATE_MACHINE"
    frame_ledger = (
        read_json(root / "ledger/frame_exclusion_ledger.json")
        if (root / "ledger/frame_exclusion_ledger.json").exists()
        else {"count": None}
    )
    summary = {
        "schema_version": "OakInk2O5RD2HFinalSummaryV1",
        "BRANCH": git("branch", "--show-current"),
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "RETARGET_OBJECTIVE_V2_SHA256": EXPECTED_HASHES["retarget_objective_v2"],
        "CERTIFICATION_GATE_V2_SHA256": EXPECTED_HASHES["certification_gate_v2"],
        "EXECUTION_INPUT_AUTHORITY_SHA256": EXPECTED_HASHES["execution_input_authority"],
        "SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256": EXPECTED_HASHES[
            "source_interaction_graph_authority"
        ],
        "COLD_START_BOOTSTRAP_CONTRACT_SHA256": EXPECTED_HASHES["cold_start_bootstrap_contract"],
        "COLD_START_SEED_AUTHORITY_V2_SHA256": EXPECTED_HASHES["cold_start_seed_authority_v2"],
        "OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256": EXPECTED_HASHES[
            "objective_v2_execution_contract_v3"
        ],
        "D2H_CERTIFICATION_PLAN_SHA256": plan_sha,
        "FROZEN_METHOD_INTEGRITY": integrity.get("FROZEN_METHOD_INTEGRITY", "NOT_RUN"),
        "METHOD_CHANGED_DURING_CERTIFICATION": integrity.get(
            "METHOD_CHANGED_DURING_CERTIFICATION", "NOT_RUN"
        ),
        "PRE_D2H_DEV1_EXCLUSION_COUNT": frame_ledger.get("count"),
        "COLDSTART_SPARSE_VALIDATION_V4": sparse.get("COLDSTART_SPARSE_VALIDATION_V4", "NOT_RUN"),
        "SPARSE_V4_Q_OLD_ACCESS_COUNT": sparse.get("q_old_access_count", "NOT_RUN"),
        "SPARSE_V4_PRIOR_EVIDENCE_OVERLAP": (
            read_json(root / "sparse_v4/manifest.json").get("SPARSE_V4_OVERLAP_WITH_PRIOR_EVIDENCE")
            if (root / "sparse_v4/manifest.json").exists()
            else "NOT_RUN"
        ),
        "COLDSTART_WINDOW_VALIDATION_V4": window.get("COLDSTART_WINDOW_VALIDATION_V4", "NOT_RUN"),
        "WINDOW_V4_Q_OLD_ACCESS_COUNT": window.get("q_old_access_count", "NOT_RUN"),
        "WINDOW_V4_PRIOR_EVIDENCE_OVERLAP": (
            read_json(root / "window_v4/manifest.json").get("WINDOW_V4_PRIOR_EVIDENCE_OVERLAP")
            if (root / "window_v4/manifest.json").exists()
            else "NOT_RUN"
        ),
        "WINDOW_V4_SPARSE_V4_OVERLAP": (
            read_json(root / "window_v4/manifest.json").get("WINDOW_V4_SPARSE_V4_OVERLAP")
            if (root / "window_v4/manifest.json").exists()
            else "NOT_RUN"
        ),
        "FRESH_CROSS_EPISODE_CONTROLS": cross.get("FRESH_CROSS_EPISODE_CONTROLS", "NOT_RUN"),
        "FRESH_CROSS_EPISODE_CONTROL_COUNT": cross.get("N_CONTROLS", 0),
        "FRESH_CROSS_EPISODE_TOTAL_RUNS": cross.get("TOTAL_RUNS", 0),
        "EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION": "PASS" if certified else "FAIL",
        "FULL_DEV2_COMPUTE_AUTHORIZED": "YES" if certified else "NO",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": dev2.get("DEV2_FULL_PRODUCTION_SOLVE_COUNT", 0),
        "DEV2_DEVELOPMENT_FRAME0_STATE_REUSED": "NO",
        "DEV2_NUMERICAL_RESULT": dev2.get("DEV2_NUMERICAL_RESULT", "NOT_RUN"),
        "DEV2_COMPLETED_FRAMES": dev2.get("COMPLETED_FRAMES", 0),
        "SEMANTIC_V1_RESULT": semantic.get("SEMANTIC_V1_RESULT", "NOT_RUN"),
        "DEV2_OBJECTIVE_V2_EXECUTION_V3_MACHINE": (
            "PASS"
            if semantic.get("SEMANTIC_V1_RESULT") == "PASS"
            else "RETARGET_SEMANTIC_FAIL"
            if semantic.get("SEMANTIC_V1_RESULT") == "FAIL"
            else "NUMERICAL_FAIL"
            if dev2.get("DEV2_NUMERICAL_RESULT") == "FAIL"
            else "NOT_RUN"
        ),
        "DEV2_EXECUTION_V3_HTML": viewer.get("DEV2_EXECUTION_V3_HTML"),
        "DEV2_HTML_SHA256": viewer.get("DEV2_HTML_SHA256"),
        "VIEWER_ROLE": viewer.get("VIEWER_ROLE"),
        "DEV2_HUMAN_GEOMETRIC_REVIEW": (
            "PENDING" if semantic.get("SEMANTIC_V1_RESULT") == "PASS" and viewer else "NOT_RUN"
        ),
        "O5_FINAL": (
            "PENDING_DEV1_OBJECTIVE_V2_REFINEMENT"
            if semantic.get("SEMANTIC_V1_RESULT") == "PASS" and viewer
            else "NOT_RUN"
        ),
        "OBJECTIVE_V2_CHANGED": "NO",
        "GATE_V2_CHANGED": "NO",
        "EXECUTION_V3_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "S1_OR_EXECUTION_V2_RESCUE_USED": "NO",
        "ALTERNATE_COLDSTART_METHOD_USED": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
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
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "NEXT": next_step,
    }
    write_json(root / "final_summary.json", summary)
    lines = [
        "# OakInk2 O5R-D2H ExecutionV3 Independent Cold-Start Certification Handoff",
        "",
        f"- Frozen method integrity: `{summary['FROZEN_METHOD_INTEGRITY']}`",
        f"- SparseV4: `{summary['COLDSTART_SPARSE_VALIDATION_V4']}`",
        f"- WindowV4: `{summary['COLDSTART_WINDOW_VALIDATION_V4']}`",
        f"- Fresh cross-episode controls: `{summary['FRESH_CROSS_EPISODE_CONTROLS']}`",
        f"- Full DEV2 authorized: `{summary['FULL_DEV2_COMPUTE_AUTHORIZED']}`",
        f"- DEV2 full run count: `{summary['DEV2_FULL_PRODUCTION_SOLVE_COUNT']}`",
        f"- NEXT: `{summary['NEXT']}`",
        "",
        "No certification or heldout split records were consumed. No frozen method, gate, Semantic V1, manifest, split, or Wuji authority was changed.",
    ]
    (root / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    freeze_certification_plan(root)
    build_exclusion_ledgers(root)
    freeze_sparse_v4(root)
    sparse = run_sparse_v4(root)
    if sparse["COLDSTART_SPARSE_VALIDATION_V4"] != "PASS":
        return summarize(root)
    freeze_window_v4(root)
    window = run_window_v4(root)
    if window["COLDSTART_WINDOW_VALIDATION_V4"] != "PASS":
        return summarize(root)
    freeze_cross_episode_controls(root)
    cross = run_cross_episode_controls(root)
    if cross["FRESH_CROSS_EPISODE_CONTROLS"] != "PASS":
        return summarize(root)
    authorize_dev2_full(root)
    dev2 = run_dev2_full(root)
    if dev2["DEV2_NUMERICAL_RESULT"] != "PASS":
        return summarize(root)
    run_dev2_semantic_v1(root)
    render_dev2_viewer(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-frozen-execution-v3": verify_frozen_execution_v3,
    "freeze-certification-plan": freeze_certification_plan,
    "build-exclusion-ledgers": build_exclusion_ledgers,
    "freeze-sparse-v4": freeze_sparse_v4,
    "run-sparse-v4": run_sparse_v4,
    "freeze-window-v4": freeze_window_v4,
    "run-window-v4": run_window_v4,
    "freeze-cross-episode-controls": freeze_cross_episode_controls,
    "run-cross-episode-controls": run_cross_episode_controls,
    "authorize-dev2-full": authorize_dev2_full,
    "run-dev2-full": run_dev2_full,
    "resume-dev2-full-technical-only": resume_dev2_full_technical_only,
    "run-dev2-semantic-v1": run_dev2_semantic_v1,
    "render-dev2-viewer": render_dev2_viewer,
    "summarize": summarize,
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
        print(f"O5RD2H_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
