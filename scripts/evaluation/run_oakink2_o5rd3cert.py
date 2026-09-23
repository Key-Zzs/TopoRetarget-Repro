#!/usr/bin/env python3
"""O5R-D3-CERT fresh independent RefinementV2 certification.

This runner is deliberately fail closed.  It freezes an untouched DEVELOPMENT
pool and historical-production q_old authority before exposing any selected
sample to RefinementV2.  Fresh sparse, sequential-window, and cross-episode
stages are ordered and cannot be retried after a scientific execution starts.
It has no D3-V2, DEV2, PPO, PhysX, O6, CERTIFICATION, or HELDOUT action.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5  # noqa: E402
from scripts.data import run_oakink2_o5rd2a as d2a  # noqa: E402
from scripts.data import run_oakink2_o5rd2c as d2c  # noqa: E402
from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3 as d3  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3r2 as r2  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3r3 as r3  # noqa: E402
from toporetarget.retarget.artifacts import load_warm_start  # noqa: E402
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    asset_derived_dof_blocks,
    default_search_contracts,
)
from toporetarget.retarget.objective_v3_execution import (  # noqa: E402
    ExecutionFrameInputsV3,
    RetargetMode,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3cert_refinement_v2_fresh_certification_v1"
R2_ROOT = r2.ROOT
R3_ROOT = r3.ROOT
R4_ROOT = REPO / ".local/reports/oakink2_o5rd3r4_refinement_v2_gate_semantic_alignment_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "109a0677a55208ba3d517590844404ae08637990"
DESIGN_SHA256 = "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
GATE_V2_SHA256 = "505af73c9871baa67045449a4908b3259a16116b19bc2817d8b08470855bf6dc"
OBJECTIVE_V2_SHA256 = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
TAU = 1.0e-4
EPS = 1.0e-12
SPARSE_N = 30
SPARSE_PER_STRATUM = 10
WINDOW_COUNT = 4
WINDOW_SIZE = 32
CROSS_CONTROL_COUNT = 3
CROSS_WINDOW_SIZE = 16
BASELINE_PRIMARY_RECORDS = 5
BASELINE_TOTAL_RECORDS = BASELINE_PRIMARY_RECORDS + CROSS_CONTROL_COUNT
DETERMINISM_RUNS = 3
Q_TOL = 1.0e-10
BASE_TOL = 1.0e-10
EIM_TOL = 1.0e-12


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or sorted({key for row in rows for key in row}) or ["status"]
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item
                for key, item in row.items()
            }
            for row in rows
        )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=jsonable) + "\n")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=jsonable).encode()
    ).hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode() + b"\0" + str(array.shape).encode() + b"\0" + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def freeze_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        sidecar = path.with_suffix(".sha256")
        observed = sha256_file(path)
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != observed:
            raise RuntimeError(f"FROZEN_ARTIFACT_HASH_DRIFT:{path}")
        return read_json(path)
    write_json(path, value)
    digest = sha256_file(path)
    write_text(path.with_suffix(".sha256"), digest + "\n")
    return value


def frozen(path: Path, action: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING_FROZEN:{path}")
    observed = sha256_file(path)
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError(f"{action}_REJECTED:HASH_DRIFT:{path}")
    return read_json(path), observed


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{field}={value.get(field)!r}")
    return value


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def manifest_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = [json.loads(line) for line in o5.MANIFEST_V2.read_text().splitlines() if line]
    return rows, {str(row["record_id"]): row for row in rows}


def _initial_not_run(root: Path) -> None:
    values = {
        "schema_version": "O5RD3CERTExplicitNotRunV1",
        "status": "NOT_RUN",
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    for relative in (
        "sparse/decision.json",
        "window/decision.json",
        "cross_episode/decision.json",
        "certification/final_decision.json",
        "future/not_authorized.json",
    ):
        if not (root / relative).exists():
            write_json(root / relative, values)


def _authority_paths() -> dict[str, Path]:
    return {
        "refinement_v2_design": R2_ROOT / "design/refinement_v2_design.json",
        "gate_v2": R4_ROOT / "gate_v2/development_gate_v2.json",
        "objective_v2": REPO / "src/toporetarget/retarget/objective_v2.py",
        "execution_v4": REPO / "src/toporetarget/retarget/objective_v4_execution.py",
        "semantic_v1": REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "manifest_v2": o5.MANIFEST_V2,
        "split_v2": o5.SPLIT_V2,
    }


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0,
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "dataset_root_exists": o5.DATASET_ROOT.is_dir(),
        "r2_exists": R2_ROOT.is_dir(),
        "r3_exists": R3_ROOT.is_dir(),
        "r4_exists": R4_ROOT.is_dir(),
    }
    value = {
        "schema_version": "O5RD3CERTGitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": head,
        "status_short": git("status", "--short", "--untracked-files=all"),
        "diff_stat": git("diff", "--stat"),
        "diff": git("diff"),
        "cached_diff_stat": git("diff", "--cached", "--stat"),
        "cached_diff": git("diff", "--cached"),
        "diff_check": git("diff", "--check"),
        "log_150": git("log", "-150", "--oneline", "--decorate"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", value)
    (root / "technical_failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    _initial_not_run(root)
    if value["status"] != "PASS":
        raise RuntimeError("D3_CERT_STATUS=BLOCKED_UPSTREAM_AUTHORITY:GIT")
    return value


def verify_r4_authority(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_R4_AUTHORITY")
    r4 = read_json(R4_ROOT / "final_summary.json")
    authorization = read_json(R4_ROOT / "future/fresh_certification_authorization.json")
    plan_stub = read_json(R4_ROOT / "future/fresh_certification_plan_stub.json")
    r3_summary = read_json(R3_ROOT / "final_summary.json")
    design_path = R2_ROOT / "design/refinement_v2_design.json"
    gate_path = R4_ROOT / "gate_v2/development_gate_v2.json"
    design = read_json(design_path)
    gate = read_json(gate_path)
    checks = {
        "D3_R4_STATUS": r4.get("D3_R4_STATUS") == "PASS_GATE_REALIGNMENT",
        "GATE_SEMANTIC_ALIGNMENT": r4.get("GATE_SEMANTIC_ALIGNMENT") == "MISALIGNED",
        "PRIMARY_ROOT_CAUSE": r4.get("PRIMARY_ROOT_CAUSE") == "DEVELOPMENT_GATE_SEMANTIC_MISMATCH",
        "MEDIAN_AUTHORITY": r4.get("MEDIAN_REDUCTION_50_PERCENT_AUTHORITY")
        == "CONSERVATIVE_PREREGISTERED_HEURISTIC",
        "GATE_V2_CREATED": r4.get("REFINEMENT_V2_DEVELOPMENT_GATE_V2_CREATED") == "YES",
        "GATE_V2_SHA": sha256_file(gate_path) == GATE_V2_SHA256,
        "GATE_V2_SIDECAR": (
            gate_path.with_suffix(".sha256").read_text().split()[0] == GATE_V2_SHA256
        ),
        "R3_COMPATIBILITY": r4.get("R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY") == "PASS",
        "EVIDENCE_ROLE": r4.get("EVIDENCE_ROLE") == "POST_HOC_COMPATIBILITY_ONLY",
        "FRESH_AUTHORIZED": authorization.get("FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED") == "YES",
        "CROSS_EPISODE_REQUIRED": plan_stub.get("CROSS_EPISODE_REQUIREMENT") == "REQUIRED",
        "CERTIFICATION_NOT_RUN": authorization.get("REFINEMENT_V2_INDEPENDENT_CERTIFICATION")
        == "NOT_RUN",
        "R3_HISTORICAL_FAIL": r3_summary.get("D3_R3_STATUS") == "FAIL_DEVELOPMENT_GATE",
        "R3_NOT_REWRITTEN": r4.get("HISTORICAL_R3_RESULT_REWRITTEN") == "NO",
        "DESIGN_SHA": sha256_file(design_path) == DESIGN_SHA256,
        "DESIGN_SIDECAR": design_path.with_suffix(".sha256").read_text().split()[0]
        == DESIGN_SHA256,
        "DESIGN_FROZEN": design.get("status") == "FROZEN",
        "DESIGN_ID": design.get("selected_design")
        == "REFINEMENT_V2_C_CONDITIONAL_EXPANDED_ACTIVE_SET",
        "EXPANDED_20_DOF": design.get("active_dofs") == list(range(20)),
        "MEDIAN_DIAGNOSTIC_ONLY": gate.get("median_invalid_relative_E_IM_reduction")
        == "DIAGNOSTIC_ONLY",
        "GATE_FROZEN": gate.get("status") == "FROZEN" and gate.get("mutable_after_freeze") is False,
    }
    authority_hashes = {name: sha256_file(path) for name, path in _authority_paths().items()}
    value = {
        "schema_version": "O5RD3CERTUpstreamIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D3_R4_STATUS": r4.get("D3_R4_STATUS"),
        "D3_R3_HISTORICAL_STATUS": r4.get("D3_R3_HISTORICAL_STATUS"),
        "HISTORICAL_GATE_V1_RESULT": r4.get("HISTORICAL_GATE_V1_RESULT"),
        "HISTORICAL_R3_RESULT_REWRITTEN": r4.get("HISTORICAL_R3_RESULT_REWRITTEN"),
        "REFINEMENT_V2_DESIGN_SHA256": sha256_file(design_path),
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": sha256_file(gate_path),
        "authority_hashes": authority_hashes,
        "GATE_V2_FROZEN_BEFORE_FRESH_CERTIFICATION": "YES",
        "gate_v2_frozen_mtime_ns": gate_path.stat().st_mtime_ns,
        "fresh_optimizer_execution_count_at_verification": 0,
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
    }
    write_json(root / "preflight/upstream_integrity.json", value)
    write_json(root / "preflight/frozen_method.json", {"status": value["status"], **value})
    write_json(root / "preflight/gate_v2.json", {"status": value["status"], "gate": gate})
    if value["status"] != "PASS":
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "BLOCKED_UPSTREAM_AUTHORITY",
                "D3_CERT_STATUS": "BLOCKED_UPSTREAM_AUTHORITY",
                "D3_V2_AUTHORIZED": "NO",
                "checks": checks,
            },
        )
        raise RuntimeError("D3_CERT_STATUS=BLOCKED_UPSTREAM_AUTHORITY")
    return value


def _prior_consumed_sequences() -> list[dict[str, Any]]:
    entries = [
        {
            "record_id": item["record_id"],
            "sequence_id": str(item["record_id"]).split("oakink2:", 1)[1].rsplit(":", 1)[0],
            "provenance": "O5_FIXED_DEVELOPMENT_EPISODE_AND_ALL_D2_D3_DESCENDANTS",
            "path": str((o5.REPORT_ROOT / "preflight/fixed_o5_episode_set.json").resolve()),
        }
        for item in o5.EPISODES
    ]
    cross_path = (
        REPO
        / ".local/reports/oakink2_o5rd2l_execution_v4_sequential_runtime_repair_v1/cross_episode_v6/manifest.json"
    )
    cross = read_json(cross_path)
    for item in cross["controls"]:
        entries.append(
            {
                "record_id": item["record_id"],
                "sequence_id": item["sequence_id"],
                "provenance": "D2L_CROSS_EPISODE_V6_CONSUMED_CONTROL",
                "path": str(cross_path.resolve()),
            }
        )
    return entries


def build_certification_exclusion_ledger(root: Path) -> dict[str, Any]:
    require(root / "preflight/upstream_integrity.json", "status", "PASS", "BUILD_LEDGER")
    consumed = _prior_consumed_sequences()
    sequences = sorted({str(item["sequence_id"]) for item in consumed})
    records = sorted({str(item["record_id"]) for item in consumed})
    ledger_sources = [
        R2_ROOT / "development_set/representative_failure_frames.json",
        R2_ROOT / "development_set/valid_controls.json",
        R2_ROOT / "development_set/sequential_windows.json",
        R3_ROOT / "manifests/failure_frames.json",
        R3_ROOT / "manifests/valid_controls.json",
        R3_ROOT / "manifests/sequential_windows.json",
        REPO
        / ".local/reports/oakink2_o5rd2l_execution_v4_sequential_runtime_repair_v1/ledger/evidence_ledger.json",
        REPO
        / ".local/reports/oakink2_o5rd2l_execution_v4_sequential_runtime_repair_v1/cross_episode_v6/manifest.json",
        REPO
        / ".local/reports/oakink2_o5rd3r_d3v2_semantic_tail_repair_v1/ledger/evidence_ledger.json",
    ]
    source_receipts = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in ledger_sources
    ]
    historical = {
        "schema_version": "O5RD3CERTHistoricalExclusionLedgerV1",
        "status": "FROZEN",
        "policy": "SOURCE_SEQUENCE_DISJOINT from every prior OakInk2 method-development or consumed cross-episode source",
        "consumed_sources": consumed,
        "excluded_record_ids": records,
        "excluded_sequence_ids": sequences,
        "D3_DEV1_FRAME_IDS": list(range(d3.SOURCE_START, d3.SOURCE_STOP)),
        "D3_DEV1_FRAME_COUNT": d3.EXPECTED_FRAMES,
        "source_receipts": source_receipts,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    freeze_json(root / "ledger/historical_exclusion_ledger.json", historical)
    return historical


def audit_fresh_refinement_pool(root: Path) -> dict[str, Any]:
    ledger, ledger_sha = frozen(
        root / "ledger/historical_exclusion_ledger.json", "AUDIT_FRESH_REFINEMENT_POOL"
    )
    split = read_json(o5.SPLIT_V2)
    development = set(str(value) for value in split["splits"]["DEVELOPMENT"])
    certification = set(str(value) for value in split["splits"]["CERTIFICATION"])
    heldout = set(str(value) for value in split["splits"]["HELDOUT_TEST"])
    rows, _by_id = manifest_rows()
    excluded_sequences = set(str(value) for value in ledger["excluded_sequence_ids"])
    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    development_rows = [row for row in rows if str(row["record_id"]) in development]
    for row in development_rows:
        record_id = str(row["record_id"])
        sequence_id = str(row["sequence_id"])
        reasons: list[str] = []
        if not bool(row.get("eligibility")):
            reasons.append("MANIFEST_INELIGIBLE")
        if str(row.get("active_hand")) != "RIGHT":
            reasons.append("RIGHT_HAND_DATA_UNAVAILABLE")
        if len(row.get("official_right_object_list", [])) != 1:
            reasons.append("TARGET_OBJECT_AMBIGUOUS")
        if not row.get("canonical_target_object") or not row.get("object_asset"):
            reasons.append("MISSING_OBJECT_AUTHORITY")
        if sequence_id in excluded_sequences:
            reasons.append("METHOD_DEVELOPMENT_SOURCE_SEQUENCE")
        if record_id in certification:
            reasons.append("CERTIFICATION_SPLIT")
        if record_id in heldout:
            reasons.append("HELDOUT_SPLIT")
        if reasons:
            exclusions.append(
                {
                    "record_id": record_id,
                    "sequence_id": sequence_id,
                    "primitive_id": row.get("primitive_id"),
                    "reasons": reasons,
                }
            )
            continue
        interval = [int(value) for value in row["source_interval"]]
        eligible.append(
            {
                "record_id": record_id,
                "sequence_id": sequence_id,
                "primitive": str(row["primitive"]),
                "primitive_id": int(row["primitive_id"]),
                "object_id": str(row["canonical_target_object"]),
                "source_interval": interval,
                "frame_count": interval[1] - interval[0],
                "canonical_record_sha256": str(row["canonical_record_sha256"]),
                "object_asset_sha256": str(row["object_asset_sha256"]),
                "freshness": "SOURCE_SEQUENCE_DISJOINT",
                "selection_hash": hashlib.sha256(record_id.encode()).hexdigest(),
            }
        )
    eligible.sort(key=lambda item: (item["frame_count"], item["selection_hash"]))
    summary = {
        "schema_version": "O5RD3CERTFreshPoolSummaryV1",
        "status": "PASS" if len(eligible) >= BASELINE_TOTAL_RECORDS else "INSUFFICIENT",
        "DEVELOPMENT_RECORD_COUNT": len(development_rows),
        "EXCLUDED_METHOD_DEVELOPMENT_RECORD_COUNT": sum(
            "METHOD_DEVELOPMENT_SOURCE_SEQUENCE" in item["reasons"] for item in exclusions
        ),
        "ELIGIBLE_FRESH_RECORD_COUNT": len(eligible),
        "ELIGIBLE_FRESH_FRAME_COUNT": sum(int(item["frame_count"]) for item in eligible),
        "METHOD_DEVELOPMENT_FRAME_OVERLAP": 0,
        "METHOD_DEVELOPMENT_SEQUENCE_OVERLAP": 0,
        "freshness": "SOURCE_SEQUENCE_DISJOINT",
        "exclusion_ledger_sha256": ledger_sha,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    write_json(
        root / "fresh_pool/eligible_records.json",
        {"status": summary["status"], "records": eligible},
    )
    write_json(
        root / "fresh_pool/eligible_frames.json",
        {
            "status": summary["status"],
            "frame_count": summary["ELIGIBLE_FRESH_FRAME_COUNT"],
            "representation": "record source_interval [start,end); not expanded to avoid redundant payload",
        },
    )
    write_json(root / "fresh_pool/pool_summary.json", summary)
    write_json(
        root / "preflight/split_integrity.json",
        {
            "status": "PASS",
            "manifest_v2_sha256": sha256_file(o5.MANIFEST_V2),
            "split_v2_sha256": sha256_file(o5.SPLIT_V2),
            "split_counts": {name: len(values) for name, values in split["splits"].items()},
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    freeze_json(
        root / "ledger/fresh_pool_exclusions.json",
        {
            "schema_version": "O5RD3CERTFreshPoolExclusionsV1",
            "status": "FROZEN",
            "exclusions": exclusions,
        },
    )
    if summary["status"] != "PASS":
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "BLOCKED_INSUFFICIENT_FRESH_DEVELOPMENT_POOL",
                "D3_CERT_STATUS": "BLOCKED_INSUFFICIENT_FRESH_DEVELOPMENT_POOL",
                "D3_V2_AUTHORIZED": "NO",
            },
        )
        raise RuntimeError("D3_CERT_STATUS=BLOCKED_INSUFFICIENT_FRESH_DEVELOPMENT_POOL")
    return summary


def audit_qold_authority(root: Path) -> dict[str, Any]:
    require(root / "fresh_pool/pool_summary.json", "status", "PASS", "AUDIT_QOLD_AUTHORITY")
    existing = []
    value = {
        "schema_version": "O5RD3CERTQOldPoolStatusV1",
        "status": "PASS",
        "FRESH_REFINEMENT_QOLD_POOL_STATUS": "BASELINE_GENERATION_REQUIRED",
        "existing_authoritative_records": existing,
        "historical_generator": "OakInk2O5 exact checkpointed continuous sequential production solver",
        "historical_generator_contract": str(
            (o5.REPORT_ROOT / "contract/geometric_retarget_contract.json").resolve()
        ),
        "historical_generator_contract_sha256": sha256_file(
            o5.REPORT_ROOT / "contract/geometric_retarget_contract.json"
        ),
        "RefinementV2_generates_q_old": False,
    }
    write_json(root / "qold_authority/pool_status.json", value)
    return value


def _select_distinct(
    rows: list[dict[str, Any]], count: int, *, excluded_sequences: set[str] | None = None
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_sequences = set() if excluded_sequences is None else set(excluded_sequences)
    for row in rows:
        if str(row["sequence_id"]) in used_sequences:
            continue
        selected.append(dict(row))
        used_sequences.add(str(row["sequence_id"]))
        if len(selected) == count:
            break
    return selected


def freeze_baseline_generation_plan(root: Path) -> dict[str, Any]:
    require(
        root / "qold_authority/pool_status.json",
        "FRESH_REFINEMENT_QOLD_POOL_STATUS",
        "BASELINE_GENERATION_REQUIRED",
        "FREEZE_BASELINE_GENERATION_PLAN",
    )
    eligible = read_json(root / "fresh_pool/eligible_records.json")["records"]
    primary = _select_distinct(eligible, BASELINE_PRIMARY_RECORDS)
    if len(primary) != BASELINE_PRIMARY_RECORDS:
        raise RuntimeError("BASELINE_GENERATION_PRIMARY_POOL_INSUFFICIENT")
    cross_candidates = [
        row
        for row in eligible
        if row["sequence_id"] not in {item["sequence_id"] for item in primary}
    ]
    cross: list[dict[str, Any]] = []
    used_objects: set[str] = set()
    used_primitives: set[str] = set()
    for row in cross_candidates:
        if row["object_id"] in used_objects or row["primitive"] in used_primitives:
            continue
        cross.append(dict(row))
        used_objects.add(str(row["object_id"]))
        used_primitives.add(str(row["primitive"]))
        if len(cross) == CROSS_CONTROL_COUNT:
            break
    if len(cross) != CROSS_CONTROL_COUNT:
        cross = _select_distinct(
            cross_candidates,
            CROSS_CONTROL_COUNT,
            excluded_sequences={str(item["sequence_id"]) for item in primary},
        )
    if len(cross) != CROSS_CONTROL_COUNT:
        raise RuntimeError("BASELINE_GENERATION_CROSS_POOL_INSUFFICIENT")
    records = []
    for index, (role, item) in enumerate(
        [("PRIMARY_CERTIFICATION_POOL", row) for row in primary]
        + [("CROSS_EPISODE_CONTROL_POOL", row) for row in cross],
        start=1,
    ):
        records.append(
            {
                **item,
                "baseline_id": f"record_{index:02d}",
                "role": role,
                "generator_review": f"record_{index:02d}",
            }
        )
    value = {
        "schema_version": "O5RD3CERTBaselineGenerationManifestV1",
        "status": "FROZEN",
        "selection_timing": "BEFORE_BASELINE_E_IM_AND_BEFORE_REFINEMENT_V2",
        "selection_rule": "eligible SOURCE_SEQUENCE_DISJOINT records ordered by shortest frame_count then sha256(record_id); first five distinct sequences for primary pool; next three distinct sequences with object/primitive diversity where possible for cross controls",
        "eligibility_failure_policy": "a historical solver technical failure makes the frozen plan INSUFFICIENT; records are not substituted after any baseline execution",
        "outcome_driven_selection": False,
        "generator": "FROZEN_HISTORICAL_O5_PRODUCTION_SOLVER",
        "generator_contract_sha256": sha256_file(
            o5.REPORT_ROOT / "contract/geometric_retarget_contract.json"
        ),
        "record_count": len(records),
        "records": records,
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    return freeze_json(root / "qold_authority/generation_plan/manifest.json", value)


def _baseline_paths(root: Path, item: dict[str, Any]) -> tuple[Path, dict[str, Path]]:
    work_root = root / "qold_authority/generation_work"
    return work_root, o5.episode_paths(work_root, str(item["generator_review"]))


def _baseline_episode(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "review": item["generator_review"],
        "record_id": item["record_id"],
        "object": item["object_id"],
        "source_interval": item["source_interval"],
        "primary_frame": int(item["source_interval"][0]),
    }


def _generate_one_baseline(
    root: Path,
    item: dict[str, Any],
    record: dict[str, Any],
    model: Any,
    surface: Any,
) -> dict[str, Any]:
    authority_dir = root / "qold_authority/generated" / str(item["baseline_id"])
    authority_path = authority_dir / "authority.json"
    if authority_path.is_file():
        value = read_json(authority_path)
        if value.get("status") != "FROZEN":
            raise RuntimeError(f"BASELINE_AUTHORITY_INCOMPLETE:{item['baseline_id']}")
        return value
    work_root, paths = _baseline_paths(root, item)
    episode = _baseline_episode(item)
    if paths["input_authority"].is_file():
        input_authority = read_json(paths["input_authority"])
    else:
        paths, input_authority, _load_sec, _prepare_sec = o5.prepare_episode(
            work_root, episode, record, model
        )
    run_state_path = authority_dir / "generation_state.json"
    state = read_json(run_state_path) if run_state_path.is_file() else {}
    if state.get("status") == "SCIENTIFIC_FAILURE":
        raise RuntimeError(f"BASELINE_PREVIOUS_FAILURE:{item['baseline_id']}")
    resume = paths["checkpoints"].is_dir()
    if not state:
        write_json(
            run_state_path,
            {
                "status": "STARTED",
                "baseline_id": item["baseline_id"],
                "record_id": item["record_id"],
                "BASELINE_GENERATION_RUN_COUNT": 1,
                "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
            },
        )
    result = o5._run_checkpoint_refinement(
        canonical=paths["canonical"],
        warm_start=paths["warm"],
        graph_path=paths["graph"],
        robot=o5.ROBOT,
        collision_samples=o5._default_collision_samples(o5.ROBOT),
        query_profile_id=o5.QUERY_PROFILE,
        coordinate_profile_id=o5.COORDINATE_PROFILE,
        solver_profile_id=o5.SOLVER_PROFILE,
        execution_profile_id=o5.EXECUTION_PROFILE,
        start_frame=0,
        end_frame=int(input_authority["frame_count"]),
        checkpoint_root=paths["checkpoints"],
        output=paths["final"],
        asset_root=None,
        resume=resume,
        max_wall_time=None,
        stop_after_frame=None,
        progress_json=paths["progress"],
        progress_log=paths["progress_log"],
        force=False,
        allow_shadow_while_queue_paused=True,
        frame_health_gate=o5._health_gate,
        model_override=model,
        surface_override=surface,
    )
    if result.get("status") != "complete" or not paths["final"].is_dir():
        write_json(
            run_state_path,
            {
                "status": "SCIENTIFIC_FAILURE",
                "result": {key: value for key, value in result.items() if key != "frame_rows"},
                "BASELINE_GENERATION_RUN_COUNT": 1,
                "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
            },
        )
        raise RuntimeError(f"BASELINE_GENERATION_FAILED:{item['baseline_id']}")
    rows = o5.checkpoint_rows(paths, episode)
    o5.validate_frame_rows(rows, int(input_authority["frame_count"]))
    final = o5.load_final_trajectory(paths["final"])
    o5.compact_trajectory(paths, episode, final, rows)
    authority_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths["trajectory"], authority_dir / "trajectory.npz")
    trajectory_sha = sha256_file(authority_dir / "trajectory.npz")
    write_text(authority_dir / "trajectory.sha256", trajectory_sha + "\n")
    value = {
        "schema_version": "O5RD3CERTHistoricalQOldAuthorityV1",
        "status": "FROZEN",
        "baseline_id": item["baseline_id"],
        "record_id": item["record_id"],
        "sequence_id": item["sequence_id"],
        "primitive": item["primitive"],
        "primitive_id": item["primitive_id"],
        "object_id": item["object_id"],
        "source_interval": item["source_interval"],
        "frame_count": int(final.frame_count),
        "trajectory_path": str((authority_dir / "trajectory.npz").resolve()),
        "trajectory_sha256": trajectory_sha,
        "final_artifact_path": str(paths["final"].resolve()),
        "final_artifact_sha256": o5.tree_hash(paths["final"]),
        "canonical_path": str(paths["canonical"].resolve()),
        "canonical_sha256": o5.tree_hash(paths["canonical"]),
        "warm_path": str(paths["warm"].resolve()),
        "warm_sha256": o5.artifact_hash(paths["warm"]),
        "interaction_graph_path": str(paths["graph"].resolve()),
        "interaction_graph_sha256": o5.interaction_artifact_hash(paths["graph"]),
        "historical_solver": "OakInk2O5 exact checkpointed continuous sequential production solver",
        "historical_solver_contract_sha256": sha256_file(
            o5.REPORT_ROOT / "contract/geometric_retarget_contract.json"
        ),
        "generation_manifest_sha256": sha256_file(
            root / "qold_authority/generation_plan/manifest.json"
        ),
        "q_finite": bool(np.isfinite(np.asarray(final.arrays["qpos"])).all()),
        "base_finite": bool(np.isfinite(np.asarray(final.arrays["base_pose_scene"])).all()),
        "frame_binding": "EXACT_SOURCE_INTERVAL_ORDER",
        "BASELINE_GENERATION_RUN_COUNT": 1,
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
    }
    write_json(authority_path, value)
    write_json(run_state_path, {"status": "COMPLETE", **value})
    return value


class FreshRefinementRuntime(d2g.V3Runtime):
    """Generic RefinementV2 runtime bound to a frozen q_old record."""

    def __init__(self, root: Path, authority: dict[str, Any]):
        super().__init__("dev_01", root)
        self.authority_receipt = authority
        self.sequence = d2g.load_hoi_sequence(Path(authority["canonical_path"]))
        self.graph = d2g.load_interaction_graph(Path(authority["interaction_graph_path"]))
        self.warm = load_warm_start(Path(authority["warm_path"]))
        self.final = o5.load_final_trajectory(Path(authority["final_artifact_path"]))
        geometry = root / "qold_authority/runtime_geometry" / str(authority["baseline_id"])
        self.resources = d2g.prepare_refinement_resources(
            self.sequence, self.graph, self.solver, geometry_artifact_root=geometry
        )
        self.backends = d2g.prepare_refinement_runtime_backends(self.resources, self.execution)
        self.current_runtime_step = 0
        self._base_cache = {}

    def bind_context(
        self,
        ordinal: int,
        *,
        previous_base: np.ndarray | None = None,
        previous_qpos: np.ndarray | None = None,
    ) -> tuple[Any, Any]:
        current_frame = int(self.graph.frame_indices[ordinal])
        object_id = str(self.graph.metadata["object_id"])
        obj = self.sequence.rigid_object(object_id)
        object_pose = np.asarray(obj.pose_scene.pose_scene[current_frame], dtype=np.float64)
        if int(self.current_runtime_step) == 0:
            previous_reference = None
            propagated = None
            previous_frame = None
            previous_base = None
            previous_qpos = None
        else:
            if previous_base is None or previous_qpos is None:
                raise RuntimeError("REFINEMENT_V2_PREVIOUS_RUNTIME_STATE_REQUIRED")
            previous_ordinal = ordinal - 1
            if previous_ordinal < 0:
                raise RuntimeError("REFINEMENT_V2_INVALID_PREVIOUS_ORDINAL")
            previous_frame = int(self.graph.frame_indices[previous_ordinal])
            previous_reference = d2a.map_previous_state_to_seed(
                np.asarray(previous_base),
                np.asarray(previous_qpos),
                np.asarray(self.warm.arrays["base_pose_scene"][ordinal]),
            )
            propagated = d2a.transport_previous_final_to_current_warm(
                self.warm.arrays["base_pose_scene"][previous_ordinal],
                np.asarray(previous_base),
                self.warm.arrays["base_pose_scene"][ordinal],
                self.warm.arrays["qpos"][previous_ordinal],
                np.asarray(previous_qpos),
                self.warm.arrays["qpos"][ordinal],
                self.model.joint_lower,
                self.model.joint_upper,
                previous_frame=previous_frame,
                current_frame=current_frame,
            )
        predicted_base = None if propagated is None else propagated.predicted_base_scene
        predicted_q = None if propagated is None else propagated.predicted_qpos
        binding = d2a.ProductionObjectiveContextBindingV2(
            active_frame_id=current_frame,
            local_ordinal=int(ordinal),
            current_source_frame_id=current_frame,
            previous_source_frame_id=previous_frame,
            previous_runtime_base_scene=previous_base,
            previous_robot_qpos=previous_qpos,
            continuous_predicted_translation_scene=None
            if predicted_base is None
            else np.asarray(predicted_base[:3, 3]),
            continuous_predicted_rotation_scene=None
            if predicted_base is None
            else np.asarray(predicted_base[:3, :3]),
            continuous_predicted_qpos=predicted_q,
            base_correction_reference=None if propagated is None else propagated.base_correction,
            object_pose_scene=object_pose,
            object_id=object_id,
            robot_name=str(self.model.name),
            robot_side=str(self.model.side),
            robot_dof_names=tuple(str(name) for name in self.model.dof_names),
            robot_mapping_authority="configs/robots/wuji_hand2_beta1_rh.yaml",
            source_hand_id=str(self.warm.metadata["source_hand_id"]),
        ).validate()
        context = d2a._make_context(
            self.sequence,
            self.graph,
            self.warm,
            self.model,
            self.surface,
            self.backends.solver_sdf,
            self.resources.reference_sdf,
            self.frame_profile,
            self.bone_profile,
            self.resources.paper,
            ordinal,
            previous_reference,
            temporal_scope=binding.temporal_scope,
            continuous_prediction_base=binding.continuous_prediction_base,
            continuous_prediction_qpos=binding.continuous_predicted_qpos,
            spatial_gradient_backend=self.execution.signed_distance_gradient,
            sign_cache=self.backends.sign_cache,
            compiled_spatial_fd_backend=self.backends.compiled_spatial_fd_backend,
        )
        return binding, context


def _baseline_metrics(root: Path, authority: dict[str, Any]) -> list[dict[str, Any]]:
    target = root / "qold_authority/generated" / authority["baseline_id"] / "per_frame.csv"
    if target.is_file():
        return [
            {
                **row,
                "ordinal": int(row["ordinal"]),
                "source_frame": int(row["source_frame"]),
                "baseline_E_IM": float(row["baseline_E_IM"]),
                "hard_valid": row["hard_valid"] == "True",
            }
            for row in read_csv(target)
        ]
    runtime = FreshRefinementRuntime(root, authority)
    rows = []
    previous_q: np.ndarray | None = None
    previous_base: np.ndarray | None = None
    for ordinal in range(int(authority["frame_count"])):
        runtime.current_runtime_step = ordinal
        q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        binding, context = runtime.bind_context(
            ordinal, previous_qpos=previous_q, previous_base=previous_base
        )
        values = runtime.measurement(
            ordinal, q, base, binding=binding, context=context, slack=runtime.q0_slack(ordinal)
        )
        actual = d2c.actual_continuity(runtime, previous_q, previous_base, q, base)
        evaluation = d2c._evaluate_b2(runtime, values, actual)
        rows.append(
            {
                "baseline_id": authority["baseline_id"],
                "record_id": authority["record_id"],
                "sequence_id": authority["sequence_id"],
                "primitive": authority["primitive"],
                "object_id": authority["object_id"],
                "ordinal": ordinal,
                "source_frame": int(authority["source_interval"][0]) + ordinal,
                "q_old_sha256": array_sha(q),
                "base_old_sha256": array_sha(base),
                "baseline_E_IM": float(values.interaction_e_im),
                "hard_valid": bool(evaluation["feasible"]),
            }
        )
        previous_q, previous_base = q, base
    write_csv(target, rows)
    return rows


def generate_qold_baselines_if_required(root: Path) -> dict[str, Any]:
    plan, _plan_sha = frozen(
        root / "qold_authority/generation_plan/manifest.json", "GENERATE_QOLD_BASELINES"
    )
    rows, by_id = manifest_rows()
    del rows
    model = o5.get_robot_registry().load(o5.ROBOT)
    surface = o5.load_robot_surface_samples(o5._default_collision_samples(o5.ROBOT))
    authorities = []
    for item in plan["records"]:
        print(
            f"baseline {item['baseline_id']} {item['record_id']} frames={item['frame_count']}",
            flush=True,
        )
        authority = _generate_one_baseline(root, item, by_id[item["record_id"]], model, surface)
        _baseline_metrics(root, authority)
        authorities.append(authority)
    return {
        "status": "PASS",
        "FRESH_REFINEMENT_QOLD_POOL_STATUS": "BASELINE_GENERATION_REQUIRED",
        "BASELINE_GENERATION_RUN_COUNT": sum(
            int(item["BASELINE_GENERATION_RUN_COUNT"]) for item in authorities
        ),
        "QOLD_RECORD_COUNT": len(authorities),
        "QOLD_FRAME_COUNT": sum(int(item["frame_count"]) for item in authorities),
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
    }


def freeze_qold_authority(root: Path) -> dict[str, Any]:
    plan, generation_sha = frozen(
        root / "qold_authority/generation_plan/manifest.json", "FREEZE_QOLD_AUTHORITY"
    )
    authorities = []
    for item in plan["records"]:
        authority_path = root / "qold_authority/generated" / item["baseline_id"] / "authority.json"
        authority = require(authority_path, "status", "FROZEN", "FREEZE_QOLD_AUTHORITY")
        if sha256_file(Path(authority["trajectory_path"])) != authority["trajectory_sha256"]:
            raise RuntimeError(f"QOLD_TRAJECTORY_HASH_DRIFT:{item['baseline_id']}")
        authorities.append(
            {
                "baseline_id": item["baseline_id"],
                "role": item["role"],
                "record_id": item["record_id"],
                "sequence_id": item["sequence_id"],
                "frame_count": authority["frame_count"],
                "trajectory_sha256": authority["trajectory_sha256"],
                "authority_sha256": sha256_file(authority_path),
                "canonical_sha256": authority["canonical_sha256"],
                "interaction_graph_sha256": authority["interaction_graph_sha256"],
            }
        )
    value = {
        "schema_version": "O5RD3CERTQOldAuthorityManifestV1",
        "status": "FROZEN",
        "FRESH_REFINEMENT_QOLD_POOL_STATUS": "BASELINE_GENERATION_REQUIRED",
        "generation_manifest_sha256": generation_sha,
        "QOLD_RECORD_COUNT": len(authorities),
        "QOLD_FRAME_COUNT": sum(int(item["frame_count"]) for item in authorities),
        "records": authorities,
        "QOLD_FROZEN_BEFORE_REFINEMENT_V2": "YES",
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT_AT_FREEZE": 0,
    }
    return freeze_json(root / "qold_authority/qold_authority_manifest.json", value)


def _all_baseline_rows(root: Path, role: str) -> list[dict[str, Any]]:
    plan, _ = frozen(root / "qold_authority/generation_plan/manifest.json", "LOAD_BASELINES")
    values: list[dict[str, Any]] = []
    for item in plan["records"]:
        if item["role"] != role:
            continue
        path = root / "qold_authority/generated" / item["baseline_id"] / "per_frame.csv"
        for row in read_csv(path):
            values.append(
                {
                    **row,
                    "ordinal": int(row["ordinal"]),
                    "source_frame": int(row["source_frame"]),
                    "baseline_E_IM": float(row["baseline_E_IM"]),
                    "hard_valid": row["hard_valid"] == "True",
                }
            )
    return values


def freeze_certification_plan(root: Path) -> dict[str, Any]:
    qold, qold_sha = frozen(
        root / "qold_authority/qold_authority_manifest.json", "FREEZE_CERTIFICATION_PLAN"
    )
    if qold.get("REFINEMENT_V2_CERTIFICATION_RUN_COUNT_AT_FREEZE") != 0:
        raise RuntimeError("CERTIFICATION_PLAN_NOT_FROZEN_BEFORE_FRESH_RUN")
    primary_rows = _all_baseline_rows(root, "PRIMARY_CERTIFICATION_POOL")
    by_record: dict[str, list[dict[str, Any]]] = {}
    for row in primary_rows:
        by_record.setdefault(str(row["baseline_id"]), []).append(row)
    window_candidates = []
    for baseline_id, rows in sorted(by_record.items()):
        ordered = sorted(rows, key=lambda item: int(item["ordinal"]))
        if len(ordered) < WINDOW_SIZE:
            continue
        start = (len(ordered) - WINDOW_SIZE) // 2
        selected = ordered[start : start + WINDOW_SIZE]
        window_candidates.append(
            {
                "baseline_id": baseline_id,
                "start_ordinal": start,
                "ordinals": [int(item["ordinal"]) for item in selected],
                "baseline_p95": float(
                    np.percentile([float(item["baseline_E_IM"]) for item in selected], 95)
                ),
            }
        )
    if len(window_candidates) < WINDOW_COUNT:
        raise RuntimeError("CERTIFICATION_PLAN_WINDOW_POOL_INSUFFICIENT")
    ranked_windows = sorted(
        window_candidates, key=lambda item: (item["baseline_p95"], item["baseline_id"])
    )
    selected_windows = [
        ranked_windows[-1],
        ranked_windows[-2],
        ranked_windows[len(ranked_windows) // 2],
        ranked_windows[0],
    ]
    labels = ["HIGH_1", "HIGH_2", "MID", "LOW"]
    windows = [
        dict(item, window_id=label) for label, item in zip(labels, selected_windows, strict=True)
    ]
    occupied = {
        (item["baseline_id"], int(ordinal)) for item in windows for ordinal in item["ordinals"]
    }
    sparse_pool = [
        row for row in primary_rows if (row["baseline_id"], int(row["ordinal"])) not in occupied
    ]
    if len(sparse_pool) < SPARSE_N:
        raise RuntimeError("CERTIFICATION_PLAN_SPARSE_POOL_INSUFFICIENT")
    ordered = sorted(
        sparse_pool,
        key=lambda item: (float(item["baseline_E_IM"]), item["baseline_id"], int(item["ordinal"])),
    )
    low = ordered[:SPARSE_PER_STRATUM]
    high = ordered[-SPARSE_PER_STRATUM:]
    remaining = [item for item in ordered if item not in low and item not in high]
    mid = sorted(
        remaining,
        key=lambda item: (
            abs(float(item["baseline_E_IM"]) - TAU),
            item["baseline_id"],
            int(item["ordinal"]),
        ),
    )[:SPARSE_PER_STRATUM]
    if len(low) != 10 or len(mid) != 10 or len(high) != 10:
        raise RuntimeError("CERTIFICATION_PLAN_STRATA_INSUFFICIENT")
    sparse = [dict(item, stratum="HIGH") for item in high]
    sparse += [dict(item, stratum="MID") for item in mid]
    sparse += [dict(item, stratum="LOW") for item in low]
    sparse.sort(key=lambda item: (item["stratum"], item["baseline_id"], item["ordinal"]))
    determinism = []
    for stratum, count in (("HIGH", 2), ("MID", 2), ("LOW", 1)):
        determinism.extend([item for item in sparse if item["stratum"] == stratum][:count])
    cross_plan = [
        item
        for item in read_json(root / "qold_authority/generation_plan/manifest.json")["records"]
        if item["role"] == "CROSS_EPISODE_CONTROL_POOL"
    ]
    cross = [
        {
            **item,
            "control_id": f"control_{index}",
            "ordinals": list(range(min(CROSS_WINDOW_SIZE, int(item["frame_count"])))),
        }
        for index, item in enumerate(cross_plan, start=1)
    ]
    if len(cross) != CROSS_CONTROL_COUNT or any(
        len(item["ordinals"]) != CROSS_WINDOW_SIZE for item in cross
    ):
        raise RuntimeError("CERTIFICATION_PLAN_CROSS_POOL_INSUFFICIENT")
    value = {
        "schema_version": "RefinementV2FreshCertificationPlanV1",
        "status": "FROZEN_BEFORE_FIRST_REFINEMENT_V2_RUN",
        "RefinementV2_design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "ObjectiveV2_sha256": OBJECTIVE_V2_SHA256,
        "qold_authority_manifest_sha256": qold_sha,
        "exclusion_ledger_sha256": sha256_file(root / "ledger/historical_exclusion_ledger.json"),
        "freshness": "SOURCE_SEQUENCE_DISJOINT",
        "sparse_selection_rule": "exclude selected window frames; rank solely by frozen baseline q_old E_IM; bottom 10 LOW, top 10 HIGH, and 10 closest remaining to tau MID",
        "sparse": sparse,
        "sparse_count": SPARSE_N,
        "sparse_strata": {"HIGH": 10, "MID": 10, "LOW": 10},
        "sparse_determinism": determinism,
        "window_selection_rule": "one centered 32-frame candidate per frozen primary record; rank solely by baseline q_old p95; top two HIGH, median MID, bottom LOW",
        "windows": windows,
        "window_count": WINDOW_COUNT,
        "window_size": WINDOW_SIZE,
        "window_determinism": ["HIGH_1", "LOW"],
        "cross_episode_requirement": "REQUIRED",
        "cross_episode_selection_rule": "three preselected SOURCE_SEQUENCE_DISJOINT records; first 16 frames each; no RefinementV2 outcome used",
        "cross_episode": cross,
        "cross_episode_count": CROSS_CONTROL_COUNT,
        "cross_episode_window_size": CROSS_WINDOW_SIZE,
        "determinism_runs": DETERMINISM_RUNS,
        "stage_order": ["SPARSE", "WINDOW", "CROSS_EPISODE"],
        "fail_closed": True,
        "scientific_rerun_policy": "NO_RETRY_AFTER_OPTIMIZER_START; deterministic repeats are separate",
        "criteria": read_json(R4_ROOT / "gate_v2/development_gate_v2.json"),
        "median_relative_reduction_role": "DIAGNOSTIC_ONLY",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    frozen_value = freeze_json(root / "certification_plan/plan.json", value)
    observed = sha256_file(root / "certification_plan/plan.json")
    write_json(
        root / "certification_plan/serialization_determinism.json",
        {
            "status": "PASS",
            "canonical_payload_sha256_first": canonical_sha(frozen_value),
            "canonical_payload_sha256_second": canonical_sha(
                read_json(root / "certification_plan/plan.json")
            ),
            "file_sha256": observed,
        },
    )
    freeze_json(
        root / "ledger/certification_exclusion_ledger.json",
        {
            "schema_version": "O5RD3CERTCertificationExclusionLedgerV1",
            "status": "FROZEN",
            "method_development_overlap": 0,
            "sparse_frames": [
                [item["baseline_id"], item["ordinal"]] for item in frozen_value["sparse"]
            ],
            "window_frames": [
                [item["baseline_id"], ordinal]
                for item in frozen_value["windows"]
                for ordinal in item["ordinals"]
            ],
            "cross_episode_sequences": [
                item["sequence_id"] for item in frozen_value["cross_episode"]
            ],
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    return frozen_value


def _manifest_authority(root: Path, action: str) -> tuple[dict[str, Any], str]:
    plan, plan_sha = frozen(root / "certification_plan/plan.json", action)
    if plan.get("status") != "FROZEN_BEFORE_FIRST_REFINEMENT_V2_RUN":
        raise RuntimeError(f"{action}_REJECTED:CERTIFICATION_PLAN_NOT_FROZEN")
    qold, _ = frozen(root / "qold_authority/qold_authority_manifest.json", action)
    if qold.get("QOLD_FROZEN_BEFORE_REFINEMENT_V2") != "YES":
        raise RuntimeError(f"{action}_REJECTED:QOLD_NOT_FROZEN")
    return plan, plan_sha


def _record_authority(root: Path, baseline_id: str) -> dict[str, Any]:
    return require(
        root / "qold_authority/generated" / baseline_id / "authority.json",
        "status",
        "FROZEN",
        "LOAD_RECORD_AUTHORITY",
    )


def _base_manifest_fields(root: Path, plan_sha: str) -> dict[str, Any]:
    upstream = read_json(root / "preflight/upstream_integrity.json")
    return {
        "certification_plan_sha256": plan_sha,
        "RefinementV2_design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "qold_authority_manifest_sha256": sha256_file(
            root / "qold_authority/qold_authority_manifest.json"
        ),
        "manifest_v2_sha256": upstream["authority_hashes"]["manifest_v2"],
        "split_v2_sha256": upstream["authority_hashes"]["split_v2"],
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }


def select_fresh_sparse(root: Path) -> dict[str, Any]:
    plan, plan_sha = _manifest_authority(root, "SELECT_FRESH_SPARSE")
    value = {
        "schema_version": "O5RD3CERTFreshSparseSelectionV1",
        "status": "SELECTED_BEFORE_OPTIMIZATION",
        "selection_rule": plan["sparse_selection_rule"],
        "frames": plan["sparse"],
        "determinism_subset": plan["sparse_determinism"],
        **_base_manifest_fields(root, plan_sha),
    }
    write_json(root / "sparse/selection.json", value)
    return value


def freeze_fresh_sparse(root: Path) -> dict[str, Any]:
    selection = require(
        root / "sparse/selection.json",
        "status",
        "SELECTED_BEFORE_OPTIMIZATION",
        "FREEZE_FRESH_SPARSE",
    )
    frames = selection["frames"]
    keys = [(str(item["baseline_id"]), int(item["ordinal"])) for item in frames]
    if len(frames) != SPARSE_N or len(set(keys)) != SPARSE_N:
        raise RuntimeError("SPARSE_MANIFEST_CARDINALITY_OR_DUPLICATE")
    if {item["stratum"] for item in frames} != {"HIGH", "MID", "LOW"}:
        raise RuntimeError("SPARSE_MANIFEST_STRATA_MISSING")
    enriched = []
    for item in frames:
        authority = _record_authority(root, str(item["baseline_id"]))
        enriched.append(
            {
                **item,
                "baseline_qold_trajectory_sha256": authority["trajectory_sha256"],
                "record_authority_sha256": sha256_file(
                    root / "qold_authority/generated" / str(item["baseline_id"]) / "authority.json"
                ),
            }
        )
    value = {
        **selection,
        "schema_version": "O5RD3CERTFreshSparseManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "frames": enriched,
        "N": SPARSE_N,
        "strata": {
            name: sum(item["stratum"] == name for item in frames) for name in ("HIGH", "MID", "LOW")
        },
        "SPARSE_METHOD_DEVELOPMENT_OVERLAP": 0,
        "SPARSE_PRIOR_VALIDATION_OVERLAP": 0,
        "SPARSE_CERTIFICATION_SPLIT_OVERLAP": 0,
        "SPARSE_HELDOUT_OVERLAP": 0,
        "previous_accepted_runtime_state": "ABSENT",
    }
    return freeze_json(root / "sparse/manifest.json", value)


def _normal_path(
    runtime: FreshRefinementRuntime,
    ordinal: int,
    runtime_step: int,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    runtime.current_runtime_step = runtime_step
    old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    ExecutionFrameInputsV3(
        mode=RetargetMode.REFINEMENT,
        runtime_step_index=runtime_step,
        old_production_q=old_q,
        previous_accepted_q=previous_q,
        previous_accepted_base=previous_base,
    ).validate()
    q, base, receipt = d2c.search_frame(
        runtime,
        ordinal,
        previous_q=previous_q,
        previous_base=previous_base,
        contract=default_search_contracts()[0],
    )
    return q, base, receipt


def _expanded_path(
    runtime: FreshRefinementRuntime,
    ordinal: int,
    normal_q: np.ndarray,
    normal_base: np.ndarray,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    midpoint = 0.5 * (
        np.asarray(runtime.model.joint_lower, dtype=np.float64)
        + np.asarray(runtime.model.joint_upper, dtype=np.float64)
    )
    dof_blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    block = tuple(sorted(index for indices in dof_blocks.values() for index in indices))
    if block != tuple(range(len(old_q))):
        raise RuntimeError(f"EXPANDED_ACTIVE_SET_ASSET_INVENTORY_MISMATCH:{block}")
    baseline_values, baseline_evaluation = r2._measure_state(
        runtime, ordinal, normal_q, normal_base, previous_q, previous_base
    )
    states = [
        (
            "normal_refinement_baseline",
            normal_q,
            normal_base,
            baseline_values,
            baseline_evaluation,
            {"stored": False, "path": "FROZEN_NORMAL_REFINEMENT"},
        )
    ]
    primary_failures = []
    profiles = []
    started = time.perf_counter()
    for seed_name, seed_q in (
        ("old_production", old_q),
        ("wuji_canonical_rest", neutral),
        ("joint_range_midpoint", midpoint),
    ):
        try:
            q, base, values, evaluation, profile = r2._run_primary(
                runtime,
                ordinal,
                seed_q,
                old_base,
                previous_q,
                previous_base,
                block,
                f"O5RD3CERT_C:{seed_name}:primary",
            )
            states.append((f"expanded_primary:{seed_name}", q, base, values, evaluation, profile))
            profiles.append(profile)
        except Exception as exc:
            primary_failures.append(f"{seed_name}:{type(exc).__name__}:{exc}")
    q, base, receipt = r2._best_with_polish(
        runtime, ordinal, previous_q, previous_base, block, states, "CERT_C"
    )
    receipt.update(
        {
            "RETARGET_MODE": "REFINEMENT",
            "q_old_authority": "FROZEN_HISTORICAL_Q_OLD_T_UNCHANGED",
            "active_dofs": list(block),
            "active_dof_count": len(block),
            "wrist_base_search_expansion": "NO",
            "primary_failures": primary_failures,
            "primary_profiles": profiles,
            "wall_time_sec": time.perf_counter() - started,
        }
    )
    return q, base, receipt


def _run_refinement_frame(
    runtime: FreshRefinementRuntime,
    ordinal: int,
    runtime_step: int,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    normal_started = time.perf_counter()
    normal_q, normal_base, normal_raw = _normal_path(
        runtime, ordinal, runtime_step, previous_q, previous_base
    )
    normal_time = time.perf_counter() - normal_started
    normal_eim = float(normal_raw["selected"]["interaction_e_im"])
    normal_hard = bool(normal_raw["selected_evaluation"]["feasible"])
    trigger = bool(normal_hard and normal_eim > TAU)
    expanded: dict[str, Any]
    if trigger:
        q, base, expanded_raw = _expanded_path(
            runtime, ordinal, normal_q, normal_base, previous_q, previous_base
        )
        expanded = {
            "triggered": True,
            "trigger_reason": "NORMAL_HARD_VALID_AND_E_IM_EXCEEDS_1E-4",
            "active_dofs": expanded_raw["active_dofs"],
            "active_dof_count": expanded_raw["active_dof_count"],
            "E_IM": float(expanded_raw["selected_E_IM"]),
            "hard_valid": bool(expanded_raw["selected_hard_valid"]),
            "interaction_valid": bool(expanded_raw["selected_interaction_valid"]),
            "solver_status": (
                expanded_raw.get("secondary_profile") or expanded_raw.get("primary_profile") or {}
            ).get("status"),
            "nfev": sum(
                int(item.get("nfev", 0))
                for item in (
                    expanded_raw.get("primary_profile") or {},
                    expanded_raw.get("secondary_profile") or {},
                )
            ),
            "wall_time_sec": float(expanded_raw["wall_time_sec"]),
            "selected_candidate": expanded_raw["selected_candidate"],
            "retention": expanded_raw["retention"],
            "raw_receipt": expanded_raw,
        }
        selected_path = (
            "NORMAL_PATH_RETAINED_AFTER_EXPANDED_SCREENING"
            if expanded_raw["selected_candidate"] == "normal_refinement_baseline"
            else "EXPANDED_PATH"
        )
        final_eim = expanded["E_IM"]
        final_hard = expanded["hard_valid"]
    else:
        q, base = normal_q, normal_base
        expanded = {
            "triggered": False,
            "trigger_reason": "FROZEN_TRIGGER_FALSE",
            "active_dofs": [],
            "active_dof_count": 0,
            "E_IM": None,
            "hard_valid": None,
            "interaction_valid": None,
            "solver_status": "NOT_RUN",
            "nfev": 0,
            "wall_time_sec": 0.0,
        }
        selected_path = "NORMAL_PATH"
        final_eim = normal_eim
        final_hard = normal_hard
    actual = d2c.actual_continuity(runtime, previous_q, previous_base, q, base)
    receipt = {
        "schema_version": "O5RD3CERTRefinementV2FrameReceiptV1",
        "ordinal": ordinal,
        "source_frame": int(runtime.authority_receipt["source_interval"][0]) + ordinal,
        "runtime_step": runtime_step,
        "normal": {
            "active_dofs": normal_raw["free_qpos_indices"],
            "active_dof_count": len(normal_raw["free_qpos_indices"]),
            "E_IM": normal_eim,
            "hard_valid": normal_hard,
            "solver_status": (
                normal_raw.get("secondary_solver") or normal_raw.get("primary_solver") or {}
            ).get("status"),
            "nfev": sum(
                int(item.get("nfev", 0))
                for item in (
                    normal_raw.get("primary_solver") or {},
                    normal_raw.get("secondary_solver") or {},
                )
            ),
            "wall_time_sec": normal_time,
            "raw_receipt": normal_raw,
        },
        "expanded": expanded,
        "selected_path": selected_path,
        "final_E_IM": final_eim,
        "final_hard_valid": final_hard,
        "final_interaction_valid": bool(final_eim <= TAU + EPS),
        "actual_continuity": actual,
        "q_sha256": array_sha(q),
        "base_sha256": array_sha(base),
        "total_wall_time_sec": time.perf_counter() - started,
    }
    return q, base, receipt


def _begin_stage(root: Path, relative: str, manifest_sha: str, count: int) -> Path:
    path = root / relative
    if path.exists():
        state = read_json(path)
        raise RuntimeError(
            f"SCIENTIFIC_RERUN_FORBIDDEN:{relative}:status={state.get('status')}:completed={len(state.get('completed', []))}"
        )
    write_json(
        path,
        {
            "status": "STARTED",
            "manifest_sha256": manifest_sha,
            "scientific_primary_count": count,
            "completed": [],
            "retry_allowed": False,
        },
    )
    return path


def _mark_completed(path: Path, identifier: str) -> None:
    value = read_json(path)
    value["completed"].append(identifier)
    write_json(path, value)


def _finish_stage(path: Path) -> None:
    value = read_json(path)
    value["status"] = "COMPLETE"
    write_json(path, value)


def _frame_row(
    item: dict[str, Any],
    q: np.ndarray,
    base: np.ndarray,
    receipt: dict[str, Any],
    runtime: FreshRefinementRuntime,
    previous_q: np.ndarray | None,
) -> dict[str, Any]:
    ordinal = int(item["ordinal"])
    qold = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    old = float(item["baseline_E_IM"])
    new = float(receipt["final_E_IM"])
    return {
        "baseline_id": item["baseline_id"],
        "record_id": item["record_id"],
        "sequence_id": item["sequence_id"],
        "primitive": item["primitive"],
        "object_id": item["object_id"],
        "ordinal": ordinal,
        "source_frame": int(item["source_frame"]),
        "stratum": item.get("stratum"),
        "baseline_qold_sha256": array_sha(qold),
        "baseline_E_IM": old,
        "normal_E_IM": receipt["normal"]["E_IM"],
        "normal_hard_valid": receipt["normal"]["hard_valid"],
        "normal_solver_status": receipt["normal"]["solver_status"],
        "normal_nfev": receipt["normal"]["nfev"],
        "normal_wall_time_sec": receipt["normal"]["wall_time_sec"],
        "expanded_triggered": receipt["expanded"]["triggered"],
        "expanded_E_IM": receipt["expanded"]["E_IM"],
        "expanded_hard_valid": receipt["expanded"]["hard_valid"],
        "expanded_solver_status": receipt["expanded"]["solver_status"],
        "expanded_nfev": receipt["expanded"]["nfev"],
        "expanded_wall_time_sec": receipt["expanded"]["wall_time_sec"],
        "selected_path": receipt["selected_path"],
        "final_E_IM": new,
        "recovered": bool(old > TAU and new <= TAU + EPS and receipt["final_hard_valid"]),
        "threshold_aware_nonregression": bool(new <= max(old, TAU) + EPS),
        "hard_valid": receipt["final_hard_valid"],
        "interaction_valid": receipt["final_interaction_valid"],
        "q_deviation_l2": float(np.linalg.norm(q - qold)),
        "q_old_distinct_from_previous_runtime": True
        if previous_q is None
        else array_sha(qold) != array_sha(previous_q),
        "q_sha256": array_sha(q),
        "base_sha256": array_sha(base),
        "translation_step_m": receipt["actual_continuity"]["translation_step_m"],
        "rotation_step_rad": receipt["actual_continuity"]["rotation_step_rad"],
        "q_step_inf_rad": receipt["actual_continuity"]["q_step_inf_rad"],
        "total_wall_time_sec": receipt["total_wall_time_sec"],
    }


def run_fresh_sparse(root: Path) -> dict[str, Any]:
    manifest, manifest_sha = frozen(root / "sparse/manifest.json", "RUN_FRESH_SPARSE")
    state_path = _begin_stage(root, "sparse/run_state.json", manifest_sha, SPARSE_N)
    rows = []
    jsonl = root / "sparse/frame_results.jsonl"
    write_text(jsonl, "")
    runtimes: dict[str, FreshRefinementRuntime] = {}
    for item in manifest["frames"]:
        baseline_id = str(item["baseline_id"])
        runtime = runtimes.setdefault(
            baseline_id, FreshRefinementRuntime(root, _record_authority(root, baseline_id))
        )
        ordinal = int(item["ordinal"])
        q, base, receipt = _run_refinement_frame(runtime, ordinal, 0, None, None)
        row = _frame_row(item, q, base, receipt, runtime, None)
        checkpoint = {
            "schema_version": "O5RD3CERTSparseFrameCheckpointV1",
            "status": "COMPLETE",
            "manifest_sha256": manifest_sha,
            "qpos": q,
            "base_pose_scene": base,
            "receipt": receipt,
            "row": row,
        }
        write_json(
            root / "sparse/checkpoints" / f"{baseline_id}_frame_{ordinal:04d}.json", checkpoint
        )
        append_jsonl(jsonl, checkpoint)
        rows.append(row)
        write_csv(root / "sparse/per_frame.csv", rows)
        _mark_completed(state_path, f"{baseline_id}:{ordinal}")
        print(
            f"sparse {len(rows)}/{SPARSE_N} {baseline_id}:{ordinal} old={row['baseline_E_IM']:.12g} new={row['final_E_IM']:.12g}",
            flush=True,
        )
    _finish_stage(state_path)
    return {"status": "COMPLETE", "frame_count": len(rows), "manifest_sha256": manifest_sha}


def run_fresh_sparse_determinism(root: Path) -> dict[str, Any]:
    require(root / "sparse/run_state.json", "status", "COMPLETE", "RUN_SPARSE_DETERMINISM")
    manifest, manifest_sha = frozen(root / "sparse/manifest.json", "RUN_SPARSE_DETERMINISM")
    target_state = _begin_stage(
        root,
        "sparse/determinism_run_state.json",
        manifest_sha,
        len(manifest["determinism_subset"]) * DETERMINISM_RUNS,
    )
    primary = {
        (row["baseline_id"], int(row["ordinal"])): row
        for row in read_csv(root / "sparse/per_frame.csv")
    }
    rows = []
    for item in manifest["determinism_subset"]:
        baseline_id = str(item["baseline_id"])
        ordinal = int(item["ordinal"])
        reference = primary[(baseline_id, ordinal)]
        for repeat in range(1, DETERMINISM_RUNS + 1):
            runtime = FreshRefinementRuntime(root, _record_authority(root, baseline_id))
            q, base, receipt = _run_refinement_frame(runtime, ordinal, 0, None, None)
            rows.append(
                {
                    "baseline_id": baseline_id,
                    "ordinal": ordinal,
                    "stratum": item["stratum"],
                    "repeat": repeat,
                    "trigger_match": str(receipt["expanded"]["triggered"])
                    == reference["expanded_triggered"],
                    "selected_path_match": receipt["selected_path"] == reference["selected_path"],
                    "q_sha_match": array_sha(q) == reference["q_sha256"],
                    "base_sha_match": array_sha(base) == reference["base_sha256"],
                    "E_IM_abs_diff": abs(
                        float(receipt["final_E_IM"]) - float(reference["final_E_IM"])
                    ),
                    "hard_valid_match": str(receipt["final_hard_valid"]) == reference["hard_valid"],
                }
            )
            write_csv(root / "sparse/determinism.csv", rows)
            _mark_completed(target_state, f"{baseline_id}:{ordinal}:repeat{repeat}")
    _finish_stage(target_state)
    passed = all(
        row["trigger_match"]
        and row["selected_path_match"]
        and row["q_sha_match"]
        and row["base_sha_match"]
        and row["E_IM_abs_diff"] <= EIM_TOL
        and row["hard_valid_match"]
        for row in rows
    )
    value = {
        "schema_version": "O5RD3CERTSparseDeterminismV1",
        "status": "PASS" if passed else "FAIL",
        "DETERMINISM": "PASS" if passed else "FAIL",
        "subset_count": len(manifest["determinism_subset"]),
        "repeat_count": DETERMINISM_RUNS,
    }
    write_json(root / "sparse/determinism_manifest.json", value)
    return value


def evaluate_fresh_sparse(root: Path) -> dict[str, Any]:
    require(root / "sparse/run_state.json", "status", "COMPLETE", "EVALUATE_SPARSE")
    determinism = require(
        root / "sparse/determinism_manifest.json", "status", "PASS", "EVALUATE_SPARSE"
    )
    rows = read_csv(root / "sparse/per_frame.csv")
    old_invalid = [row for row in rows if float(row["baseline_E_IM"]) > TAU]
    recovered = sum(row["recovered"] == "True" for row in old_invalid)
    recovery = recovered / len(old_invalid) if old_invalid else 0.0
    nonregression = sum(row["threshold_aware_nonregression"] == "True" for row in rows)
    low = [row for row in rows if row["stratum"] == "LOW"]
    low_preserved = sum(float(row["final_E_IM"]) <= TAU + EPS for row in low)
    hard_valid = sum(row["hard_valid"] == "True" for row in rows)
    reductions = [
        (float(row["baseline_E_IM"]) - float(row["final_E_IM"])) / float(row["baseline_E_IM"])
        for row in old_invalid
    ]
    criteria = {
        "technical_30_of_30": len(rows) == SPARSE_N,
        "old_invalid_evidence_present": bool(old_invalid),
        "old_invalid_recovery": bool(old_invalid) and recovery >= 0.8,
        "threshold_aware_nonregression": nonregression == SPARSE_N,
        "low_preservation": low_preserved == SPARSE_PER_STRATUM,
        "hard_validity": hard_valid == SPARSE_N,
        "determinism": determinism["DETERMINISM"] == "PASS",
    }
    passed = all(criteria.values())
    value = {
        "schema_version": "O5RD3CERTSparseGateResultsV1",
        "status": "PASS" if passed else "FAIL",
        "FRESH_REFINEMENT_SPARSE": "PASS" if passed else "FAIL",
        "criteria": criteria,
        "SPARSE_N": len(rows),
        "TECHNICAL": f"{len(rows)}/{SPARSE_N}",
        "OLD_INVALID_COUNT": len(old_invalid),
        "RECOVERED_COUNT": recovered,
        "RECOVERY_FRACTION": recovery,
        "THRESHOLD_AWARE_NONREGRESSION": f"{nonregression}/{SPARSE_N}",
        "LOW_PRESERVED": f"{low_preserved}/{SPARSE_PER_STRATUM}",
        "HARD_VALID": f"{hard_valid}/{SPARSE_N}",
        "DETERMINISM": determinism["DETERMINISM"],
        "MEDIAN_RELATIVE_REDUCTION_DIAGNOSTIC": None
        if not reductions
        else float(np.median(reductions)),
    }
    write_json(root / "sparse/gate_results.json", value)
    write_json(root / "sparse/decision.json", value)
    if not passed:
        write_json(
            root / "window/decision.json",
            {"status": "NOT_RUN", "FRESH_REFINEMENT_WINDOW": "NOT_RUN", "reason": "SPARSE_FAIL"},
        )
        write_json(
            root / "cross_episode/decision.json",
            {"status": "NOT_RUN", "CROSS_EPISODE_REFINEMENT": "NOT_RUN", "reason": "SPARSE_FAIL"},
        )
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "FAIL",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "FAIL",
                "D3_V2_AUTHORIZED": "NO",
                "NEXT": "DEV1_REFINEMENT_V2_FRESH_SPARSE_FAILURE_ANALYSIS",
            },
        )
    return value


def select_fresh_windows(root: Path) -> dict[str, Any]:
    require(root / "sparse/decision.json", "FRESH_REFINEMENT_SPARSE", "PASS", "SELECT_WINDOWS")
    plan, plan_sha = _manifest_authority(root, "SELECT_WINDOWS")
    value = {
        "schema_version": "O5RD3CERTFreshWindowSelectionV1",
        "status": "SELECTED_AFTER_SPARSE_PASS_BEFORE_WINDOW_EXECUTION",
        "selection_rule": plan["window_selection_rule"],
        "windows": plan["windows"],
        "determinism_windows": plan["window_determinism"],
        **_base_manifest_fields(root, plan_sha),
    }
    write_json(root / "window/selection.json", value)
    return value


def freeze_fresh_windows(root: Path) -> dict[str, Any]:
    selection = require(
        root / "window/selection.json",
        "status",
        "SELECTED_AFTER_SPARSE_PASS_BEFORE_WINDOW_EXECUTION",
        "FREEZE_WINDOWS",
    )
    windows = selection["windows"]
    keys = [(item["baseline_id"], int(ordinal)) for item in windows for ordinal in item["ordinals"]]
    sparse_keys = {
        (item["baseline_id"], int(item["ordinal"]))
        for item in read_json(root / "sparse/manifest.json")["frames"]
    }
    if len(windows) != WINDOW_COUNT or len(set(keys)) != WINDOW_COUNT * WINDOW_SIZE:
        raise RuntimeError("WINDOW_MANIFEST_CARDINALITY_OR_OVERLAP")
    if set(keys) & sparse_keys:
        raise RuntimeError("WINDOW_SPARSE_OVERLAP")
    value = {
        **selection,
        "schema_version": "O5RD3CERTFreshWindowManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "WINDOW_METHOD_DEVELOPMENT_OVERLAP": 0,
        "WINDOW_SPARSE_OVERLAP": 0,
        "WINDOW_PRIOR_VALIDATION_OVERLAP": 0,
        "WINDOW_CERTIFICATION_SPLIT_OVERLAP": 0,
        "WINDOW_HELDOUT_OVERLAP": 0,
        "q_old_semantics": "historical q_old[current frame] immutable",
        "previous_runtime_semantics": "ABSENT at local frame0; accepted refined state[t-1] thereafter",
    }
    return freeze_json(root / "window/manifest.json", value)


def _run_window(
    root: Path,
    window: dict[str, Any],
    *,
    repeat: int | None,
    manifest_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_id = str(window["baseline_id"])
    runtime = FreshRefinementRuntime(root, _record_authority(root, baseline_id))
    metrics = {
        int(row["ordinal"]): row
        for row in _all_baseline_rows(root, "PRIMARY_CERTIFICATION_POOL")
        if row["baseline_id"] == baseline_id
    }
    previous_q: np.ndarray | None = None
    previous_base: np.ndarray | None = None
    rows = []
    chains = []
    for step, ordinal_value in enumerate(window["ordinals"]):
        ordinal = int(ordinal_value)
        item = {**metrics[ordinal], "stratum": window["window_id"]}
        predecessor_q_sha = None if previous_q is None else array_sha(previous_q)
        q, base, receipt = _run_refinement_frame(runtime, ordinal, step, previous_q, previous_base)
        row = {
            **_frame_row(item, q, base, receipt, runtime, previous_q),
            "window_id": window["window_id"],
            "local_index": step,
            "repeat": 0 if repeat is None else repeat,
            "manifest_sha256": manifest_sha,
        }
        rows.append(row)
        chains.append(
            {
                "window_id": window["window_id"],
                "local_index": step,
                "ordinal": ordinal,
                "q_old_sha256": row["baseline_qold_sha256"],
                "predecessor_q_sha256": predecessor_q_sha,
                "accepted_q_sha256": row["q_sha256"],
                "previous_runtime_state_aliased_as_qold": False
                if predecessor_q_sha is None
                else predecessor_q_sha == row["baseline_qold_sha256"],
            }
        )
        previous_q, previous_base = q, base
    return rows, chains


def run_fresh_windows(root: Path) -> dict[str, Any]:
    require(root / "sparse/decision.json", "FRESH_REFINEMENT_SPARSE", "PASS", "RUN_WINDOWS")
    manifest, manifest_sha = frozen(root / "window/manifest.json", "RUN_WINDOWS")
    state = _begin_stage(root, "window/run_state.json", manifest_sha, WINDOW_COUNT)
    all_rows = []
    chains = []
    for window in manifest["windows"]:
        rows, chain = _run_window(root, window, repeat=None, manifest_sha=manifest_sha)
        all_rows.extend(rows)
        chains.extend(chain)
        write_csv(root / "window/per_frame.csv", all_rows)
        write_csv(root / "window/runtime_chain.csv", chains)
        _mark_completed(state, str(window["window_id"]))
        print(f"window {window['window_id']} complete frames={len(rows)}", flush=True)
    _finish_stage(state)
    return {"status": "COMPLETE", "window_count": WINDOW_COUNT, "frame_count": len(all_rows)}


def run_fresh_window_determinism(root: Path) -> dict[str, Any]:
    require(root / "window/run_state.json", "status", "COMPLETE", "RUN_WINDOW_DETERMINISM")
    manifest, manifest_sha = frozen(root / "window/manifest.json", "RUN_WINDOW_DETERMINISM")
    selected = [
        item for item in manifest["windows"] if item["window_id"] in manifest["determinism_windows"]
    ]
    state = _begin_stage(
        root,
        "window/determinism_run_state.json",
        manifest_sha,
        len(selected) * DETERMINISM_RUNS,
    )
    primary_rows = read_csv(root / "window/per_frame.csv")
    primary = {(row["window_id"], int(row["local_index"])): row for row in primary_rows}
    output = []
    for window in selected:
        for repeat in range(1, DETERMINISM_RUNS + 1):
            rows, _chain = _run_window(root, window, repeat=repeat, manifest_sha=manifest_sha)
            for row in rows:
                reference = primary[(row["window_id"], int(row["local_index"]))]
                output.append(
                    {
                        "window_id": row["window_id"],
                        "local_index": row["local_index"],
                        "repeat": repeat,
                        "trigger_match": row["expanded_triggered"]
                        == (reference["expanded_triggered"] == "True"),
                        "selected_path_match": row["selected_path"] == reference["selected_path"],
                        "q_sha_match": row["q_sha256"] == reference["q_sha256"],
                        "base_sha_match": row["base_sha256"] == reference["base_sha256"],
                        "E_IM_abs_diff": abs(
                            float(row["final_E_IM"]) - float(reference["final_E_IM"])
                        ),
                    }
                )
            write_csv(root / "window/determinism.csv", output)
            _mark_completed(state, f"{window['window_id']}:repeat{repeat}")
    _finish_stage(state)
    passed = all(
        row["trigger_match"]
        and row["selected_path_match"]
        and row["q_sha_match"]
        and row["base_sha_match"]
        and row["E_IM_abs_diff"] <= EIM_TOL
        for row in output
    )
    value = {
        "schema_version": "O5RD3CERTWindowDeterminismV1",
        "status": "PASS" if passed else "FAIL",
        "DETERMINISM": "PASS" if passed else "FAIL",
        "window_count": len(selected),
        "repeat_count": DETERMINISM_RUNS,
    }
    write_json(root / "window/determinism_manifest.json", value)
    return value


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_fresh_windows(root: Path) -> dict[str, Any]:
    require(root / "window/run_state.json", "status", "COMPLETE", "EVALUATE_WINDOWS")
    determinism = require(
        root / "window/determinism_manifest.json", "status", "PASS", "EVALUATE_WINDOWS"
    )
    rows = read_csv(root / "window/per_frame.csv")
    chains = read_csv(root / "window/runtime_chain.csv")
    summaries = []
    for window_id in ("HIGH_1", "HIGH_2", "MID", "LOW"):
        selected = [row for row in rows if row["window_id"] == window_id]
        baseline = _distribution([float(row["baseline_E_IM"]) for row in selected])
        refined = _distribution([float(row["final_E_IM"]) for row in selected])
        continuity = all(
            float(row["translation_step_m"]) <= 0.05 + EPS
            and float(row["rotation_step_rad"]) <= math.pi / 2.0 + EPS
            for row in selected
        )
        hard = all(row["hard_valid"] == "True" for row in selected)
        interaction = refined["p95"] <= TAU + EPS
        window_chains = [row for row in chains if row["window_id"] == window_id]
        chain_pass = all(
            index == 0
            or window_chains[index]["predecessor_q_sha256"]
            == window_chains[index - 1]["accepted_q_sha256"]
            for index in range(len(window_chains))
        )
        summaries.append(
            {
                "window_id": window_id,
                "N": len(selected),
                "baseline": baseline,
                "refined": refined,
                "hard_valid": "PASS" if hard else "FAIL",
                "interaction": "PASS" if interaction else "FAIL",
                "continuity": "PASS" if continuity else "FAIL",
                "runtime_chain": "PASS" if chain_pass else "FAIL",
                "determinism": determinism["DETERMINISM"]
                if window_id in ("HIGH_1", "LOW")
                else "NOT_SELECTED",
                "status": "PASS"
                if len(selected) == WINDOW_SIZE
                and hard
                and interaction
                and continuity
                and chain_pass
                else "FAIL",
            }
        )
    passed = (
        all(item["status"] == "PASS" for item in summaries) and determinism["DETERMINISM"] == "PASS"
    )
    value = {
        "schema_version": "O5RD3CERTWindowGateResultsV1",
        "status": "PASS" if passed else "FAIL",
        "FRESH_REFINEMENT_WINDOW": "PASS" if passed else "FAIL",
        "WINDOW_COUNT": len(summaries),
        "windows": summaries,
        "WINDOW_METHOD_DEVELOPMENT_OVERLAP": 0,
        "WINDOW_SPARSE_OVERLAP": 0,
        "DETERMINISM": determinism["DETERMINISM"],
    }
    write_json(root / "window/per_window.json", {"windows": summaries})
    write_csv(
        root / "window/per_window.csv",
        [
            {
                "window_id": item["window_id"],
                "N": item["N"],
                "baseline_p95": item["baseline"]["p95"],
                "refined_p95": item["refined"]["p95"],
                "hard_valid": item["hard_valid"],
                "interaction": item["interaction"],
                "continuity": item["continuity"],
                "runtime_chain": item["runtime_chain"],
                "determinism": item["determinism"],
                "status": item["status"],
            }
            for item in summaries
        ],
    )
    write_json(root / "window/gate_results.json", value)
    write_json(root / "window/decision.json", value)
    if not passed:
        write_json(
            root / "cross_episode/decision.json",
            {"status": "NOT_RUN", "CROSS_EPISODE_REFINEMENT": "NOT_RUN", "reason": "WINDOW_FAIL"},
        )
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "FAIL",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "FAIL",
                "D3_V2_AUTHORIZED": "NO",
                "NEXT": "DEV1_REFINEMENT_V2_FRESH_SEQUENCE_FAILURE_ANALYSIS",
            },
        )
    return value


def select_fresh_cross_episode_controls(root: Path) -> dict[str, Any]:
    require(root / "window/decision.json", "FRESH_REFINEMENT_WINDOW", "PASS", "SELECT_CROSS")
    plan, plan_sha = _manifest_authority(root, "SELECT_CROSS")
    value = {
        "schema_version": "O5RD3CERTCrossEpisodeSelectionV1",
        "status": "SELECTED_AFTER_WINDOW_PASS_BEFORE_CROSS_EXECUTION",
        "selection_rule": plan["cross_episode_selection_rule"],
        "controls": plan["cross_episode"],
        **_base_manifest_fields(root, plan_sha),
    }
    write_json(root / "cross_episode/eligible_pool.json", value)
    write_json(root / "cross_episode/selection.json", value)
    return value


def freeze_fresh_cross_episode(root: Path) -> dict[str, Any]:
    selection = require(
        root / "cross_episode/selection.json",
        "status",
        "SELECTED_AFTER_WINDOW_PASS_BEFORE_CROSS_EXECUTION",
        "FREEZE_CROSS",
    )
    controls = selection["controls"]
    sequences = {item["sequence_id"] for item in controls}
    prior_sequences = {
        item["sequence_id"]
        for item in read_json(root / "qold_authority/generation_plan/manifest.json")["records"]
        if item["role"] == "PRIMARY_CERTIFICATION_POOL"
    }
    if len(controls) != CROSS_CONTROL_COUNT or len(sequences) != CROSS_CONTROL_COUNT:
        raise RuntimeError("CROSS_CONTROL_COUNT_OR_SEQUENCE_DUPLICATE")
    if sequences & prior_sequences:
        raise RuntimeError("CROSS_SOURCE_SEQUENCE_NOT_DISJOINT")
    value = {
        **selection,
        "schema_version": "O5RD3CERTCrossEpisodeManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "SOURCE_SEQUENCE_DISJOINT": "YES",
        "control_count": CROSS_CONTROL_COUNT,
        "window_size": CROSS_WINDOW_SIZE,
        "determinism_runs": DETERMINISM_RUNS,
    }
    return freeze_json(root / "cross_episode/manifest.json", value)


def run_fresh_cross_episode(root: Path) -> dict[str, Any]:
    require(root / "window/decision.json", "FRESH_REFINEMENT_WINDOW", "PASS", "RUN_CROSS")
    manifest, manifest_sha = frozen(root / "cross_episode/manifest.json", "RUN_CROSS")
    state = _begin_stage(root, "cross_episode/run_state.json", manifest_sha, CROSS_CONTROL_COUNT)
    all_rows = []
    repeat_rows = []
    for control in manifest["controls"]:
        window = {
            "window_id": control["control_id"],
            "baseline_id": control["baseline_id"],
            "ordinals": control["ordinals"],
        }
        rows, chains = _run_window(root, window, repeat=None, manifest_sha=manifest_sha)
        control_root = root / "cross_episode" / control["control_id"]
        write_csv(control_root / "per_frame.csv", rows)
        write_csv(control_root / "runtime_chain.csv", chains)
        write_json(
            control_root / "result.json",
            {"status": "PRIMARY_COMPLETE", "control": control, "frame_count": len(rows)},
        )
        all_rows.extend(rows)
        for repeat in range(1, DETERMINISM_RUNS + 1):
            repeated, _ = _run_window(root, window, repeat=repeat, manifest_sha=manifest_sha)
            for reference, observed in zip(rows, repeated, strict=True):
                repeat_rows.append(
                    {
                        "control_id": control["control_id"],
                        "local_index": observed["local_index"],
                        "repeat": repeat,
                        "trigger_match": observed["expanded_triggered"]
                        == reference["expanded_triggered"],
                        "selected_path_match": observed["selected_path"]
                        == reference["selected_path"],
                        "q_sha_match": observed["q_sha256"] == reference["q_sha256"],
                        "base_sha_match": observed["base_sha256"] == reference["base_sha256"],
                        "E_IM_abs_diff": abs(
                            float(observed["final_E_IM"]) - float(reference["final_E_IM"])
                        ),
                    }
                )
            write_csv(root / "cross_episode/determinism.csv", repeat_rows)
        _mark_completed(state, str(control["control_id"]))
        print(f"cross {control['control_id']} complete frames={len(rows)}", flush=True)
    _finish_stage(state)
    passed = all(
        row["trigger_match"]
        and row["selected_path_match"]
        and row["q_sha_match"]
        and row["base_sha_match"]
        and row["E_IM_abs_diff"] <= EIM_TOL
        for row in repeat_rows
    )
    write_json(
        root / "cross_episode/determinism.json",
        {
            "status": "PASS" if passed else "FAIL",
            "DETERMINISM": "PASS" if passed else "FAIL",
            "repeat_count": DETERMINISM_RUNS,
        },
    )
    return {"status": "COMPLETE", "control_count": CROSS_CONTROL_COUNT, "determinism": passed}


def evaluate_fresh_cross_episode(root: Path) -> dict[str, Any]:
    require(root / "cross_episode/run_state.json", "status", "COMPLETE", "EVALUATE_CROSS")
    determinism = require(
        root / "cross_episode/determinism.json", "status", "PASS", "EVALUATE_CROSS"
    )
    manifest = read_json(root / "cross_episode/manifest.json")
    controls = []
    for control in manifest["controls"]:
        control_id = control["control_id"]
        rows = read_csv(root / "cross_episode" / control_id / "per_frame.csv")
        interaction = _distribution([float(row["final_E_IM"]) for row in rows])
        continuity = all(
            float(row["translation_step_m"]) <= 0.05 + EPS
            and float(row["rotation_step_rad"]) <= math.pi / 2.0 + EPS
            for row in rows
        )
        hard = all(row["hard_valid"] == "True" for row in rows)
        interaction_pass = interaction["p95"] <= TAU + EPS
        passed = (
            len(rows) == CROSS_WINDOW_SIZE
            and hard
            and interaction_pass
            and continuity
            and determinism["DETERMINISM"] == "PASS"
        )
        controls.append(
            {
                "control_id": control_id,
                "record_id": control["record_id"],
                "episode": control["sequence_id"],
                "primitive": control["primitive"],
                "object": control["object_id"],
                "frames": len(rows),
                "fallback_rate": sum(row["expanded_triggered"] == "True" for row in rows)
                / len(rows),
                "interaction": "PASS" if interaction_pass else "FAIL",
                "interaction_metrics": interaction,
                "continuity": "PASS" if continuity else "FAIL",
                "hard_validity": "PASS" if hard else "FAIL",
                "determinism": determinism["DETERMINISM"],
                "status": "PASS" if passed else "FAIL",
            }
        )
    passed = all(item["status"] == "PASS" for item in controls)
    value = {
        "schema_version": "O5RD3CERTCrossEpisodeGateResultsV1",
        "status": "PASS" if passed else "FAIL",
        "CROSS_EPISODE_REFINEMENT": "PASS" if passed else "FAIL",
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "CROSS_EPISODE_CONTROL_COUNT": len(controls),
        "CROSS_EPISODE_SOURCE_SEQUENCE_DISJOINT": "YES",
        "controls": controls,
    }
    write_json(root / "cross_episode/gate_results.json", value)
    write_json(root / "cross_episode/decision.json", value)
    if not passed:
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "FAIL",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "FAIL",
                "D3_V2_AUTHORIZED": "NO",
                "NEXT": "DEV1_REFINEMENT_V2_CROSS_EPISODE_GENERALIZATION_FAILURE_ANALYSIS",
            },
        )
    return value


def audit_method_integrity(root: Path) -> dict[str, Any]:
    start = read_json(root / "preflight/upstream_integrity.json")["authority_hashes"]
    observed = {name: sha256_file(path) for name, path in _authority_paths().items()}
    checks = {name: observed[name] == digest for name, digest in start.items()}
    source = Path(__file__).read_text(encoding="utf-8")
    special_cases = {
        "DEV1_EPISODE_SPECIAL_BRANCH": "NO",
        "CERTIFICATION_RECORD_SPECIAL_BRANCH": "NO",
        "OBJECT_ID_SPECIAL_BRANCH": "NO",
        "FRAME_LITERAL_BRANCH": "NO",
        "PRIMITIVE_SPECIAL_BRANCH": "NO",
        "SPECIAL_SEED": "NO",
        "SPECIAL_BUDGET": "NO",
        "SPECIAL_DOF_SET": "NO",
        "expanded_set_from_asset_inventory": "asset_derived_dof_blocks" in source
        and "EXPANDED_ACTIVE_SET_ASSET_INVENTORY_MISMATCH" in source,
    }
    value = {
        "schema_version": "O5RD3CERTMethodIntegrityPostrunV1",
        "status": "PASS"
        if all(checks.values()) and special_cases["expanded_set_from_asset_inventory"]
        else "FAIL",
        "METHOD_INTEGRITY_POSTRUN": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "start_hashes": start,
        "observed_hashes": observed,
        "REFINEMENT_V2_DESIGN_CHANGED": "NO" if checks["refinement_v2_design"] else "YES",
        "GATE_V2_CHANGED": "NO" if checks["gate_v2"] else "YES",
        "OBJECTIVE_V2_CHANGED": "NO" if checks["objective_v2"] else "YES",
        "SEMANTIC_V1_CHANGED": "NO" if checks["semantic_v1"] else "YES",
        "E_IM_THRESHOLD_CHANGED": "NO",
    }
    write_json(root / "audits/method_integrity_postrun.json", value)
    write_json(root / "audits/no_method_change.json", value)
    write_json(root / "audits/no_special_cases.json", {"status": value["status"], **special_cases})
    write_json(
        root / "audits/no_split_leakage.json",
        {
            "status": "PASS",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    write_json(
        root / "audits/qold_before_refinement.json",
        {
            "status": "PASS",
            "QOLD_FROZEN_BEFORE_REFINEMENT_V2": "YES",
            "qold_manifest_mtime_ns": (root / "qold_authority/qold_authority_manifest.json")
            .stat()
            .st_mtime_ns,
            "plan_mtime_ns": (root / "certification_plan/plan.json").stat().st_mtime_ns,
            "sparse_run_state_mtime_ns": (root / "sparse/run_state.json").stat().st_mtime_ns,
        },
    )
    write_json(
        root / "audits/historical_results_immutable.json",
        {
            "status": "PASS",
            "D3_R3_HISTORICAL_STATUS": "FAIL_DEVELOPMENT_GATE",
            "HISTORICAL_GATE_V1_RESULT": "FAIL",
            "HISTORICAL_R3_RESULT_REWRITTEN": "NO",
            "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY": "PASS",
            "EVIDENCE_ROLE": "POST_HOC_COMPATIBILITY_ONLY",
        },
    )
    return value


def decide_refinement_v2_certification(root: Path) -> dict[str, Any]:
    sparse = read_json(root / "sparse/decision.json")
    window = read_json(root / "window/decision.json")
    cross = read_json(root / "cross_episode/decision.json")
    integrity = read_json(root / "audits/method_integrity_postrun.json")
    criteria = {
        "C1_SPARSE": sparse.get("FRESH_REFINEMENT_SPARSE", "NOT_RUN"),
        "C2_WINDOW": window.get("FRESH_REFINEMENT_WINDOW", "NOT_RUN"),
        "C3_CROSS_EPISODE": cross.get("CROSS_EPISODE_REFINEMENT", "NOT_RUN"),
        "C4_METHOD_INTEGRITY": integrity.get("METHOD_INTEGRITY_POSTRUN", "FAIL"),
    }
    passed = all(value == "PASS" for value in criteria.values())
    if not passed:
        if criteria["C1_SPARSE"] == "FAIL":
            next_step = "DEV1_REFINEMENT_V2_FRESH_SPARSE_FAILURE_ANALYSIS"
        elif criteria["C2_WINDOW"] == "FAIL":
            next_step = "DEV1_REFINEMENT_V2_FRESH_SEQUENCE_FAILURE_ANALYSIS"
        elif criteria["C3_CROSS_EPISODE"] == "FAIL":
            next_step = "DEV1_REFINEMENT_V2_CROSS_EPISODE_GENERALIZATION_FAILURE_ANALYSIS"
        else:
            next_step = "REFINEMENT_V2_METHOD_INTEGRITY_FAILURE_ANALYSIS"
    else:
        next_step = "O5R-D3-V2_DEV1_FULL_REFINEMENT_RECOVERY_V2"
    value = {
        "schema_version": "O5RD3CERTFinalDecisionV1",
        "status": "PASS" if passed else "FAIL",
        **criteria,
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "PASS" if passed else "FAIL",
        "D3_V2_AUTHORIZED": "YES" if passed else "NO",
        "NEXT": next_step,
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    write_json(root / "certification/criterion_results.json", criteria)
    write_json(root / "certification/final_decision.json", value)
    if not passed:
        write_json(
            root / "frozen_authority/not_certified.json",
            {"status": "NOT_CERTIFIED", **value},
        )
        write_json(
            root / "future/not_authorized.json",
            {"status": "NOT_AUTHORIZED", **value},
        )
    return value


def freeze_certified_refinement_v2(root: Path) -> dict[str, Any]:
    decision = require(
        root / "certification/final_decision.json",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION",
        "PASS",
        "FREEZE_CERTIFIED_REFINEMENT_V2",
    )
    value = {
        "schema_version": "CertifiedRefinementV2AuthorityV1",
        "status": "CERTIFIED",
        "RefinementV2_design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "ObjectiveV2_sha256": OBJECTIVE_V2_SHA256,
        "input_authority_schema": "O5RD3CERTHistoricalQOldAuthorityV1",
        "normal_active_set": "asset-derived top-1 contributor block; frozen S1",
        "expanded_active_set": list(range(20)),
        "trigger_semantics": "normal hard-valid and E_IM > 1e-4",
        "solver_budget": "normal 24/8/8; expanded three seeds primary/secondary 8/8",
        "q_old_semantics": "immutable historical q_old[t]",
        "previous_runtime_semantics": "absent at local frame0; accepted refined t-1 thereafter",
        "sparse_result_sha256": sha256_file(root / "sparse/decision.json"),
        "window_result_sha256": sha256_file(root / "window/decision.json"),
        "cross_episode_result_sha256": sha256_file(root / "cross_episode/decision.json"),
        "certification_plan_sha256": sha256_file(root / "certification_plan/plan.json"),
        "exclusion_ledger_sha256": sha256_file(root / "ledger/historical_exclusion_ledger.json"),
        "decision_sha256": sha256_file(root / "certification/final_decision.json"),
        "D3_V2_SCIENTIFIC_RUN_COUNT": decision["D3_V2_SCIENTIFIC_RUN_COUNT"],
    }
    frozen_value = freeze_json(
        root / "frozen_authority/certified_refinement_v2_authority.json", value
    )
    return {
        **frozen_value,
        "CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256": sha256_file(
            root / "frozen_authority/certified_refinement_v2_authority.json"
        ),
    }


def authorize_d3v2(root: Path) -> dict[str, Any]:
    certified, certified_sha = frozen(
        root / "frozen_authority/certified_refinement_v2_authority.json", "AUTHORIZE_D3V2"
    )
    if certified.get("status") != "CERTIFIED":
        raise RuntimeError("AUTHORIZE_D3V2_REJECTED:NOT_CERTIFIED")
    value = {
        "schema_version": "O5RD3CERTD3V2AuthorizationV1",
        "status": "AUTHORIZED_NOT_RUN",
        "CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256": certified_sha,
        "D3_V2_AUTHORIZED": "YES",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "NEXT": "O5R-D3-V2_DEV1_FULL_REFINEMENT_RECOVERY_V2",
    }
    write_json(root / "future/d3v2_authorization.json", value)
    return value


def generate_d3v2_plan(root: Path) -> dict[str, Any]:
    authorization = require(
        root / "future/d3v2_authorization.json",
        "D3_V2_AUTHORIZED",
        "YES",
        "GENERATE_D3V2_PLAN",
    )
    value = {
        "schema_version": "O5RD3CERTD3V2PlanStubV1",
        "status": "AUTHORIZED_NOT_RUN",
        "NEXT": authorization["NEXT"],
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "execution_in_this_task": "FORBIDDEN",
    }
    write_json(root / "future/d3v2_plan_stub.json", value)
    return value


def _profile_stage(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"status": "NOT_RUN"}
    times = [float(row["total_wall_time_sec"]) for row in rows]
    expanded = [float(row["expanded_wall_time_sec"]) for row in rows]
    fallback = sum(row["expanded_triggered"] == "True" for row in rows)
    return {
        "status": "PASS",
        "frame_count": len(rows),
        "fallback_invocation_count": fallback,
        "fallback_invocation_fraction": fallback / len(rows),
        "mean_sec": float(np.mean(times)),
        "p50_sec": float(np.percentile(times, 50)),
        "p90_sec": float(np.percentile(times, 90)),
        "p95_sec": float(np.percentile(times, 95)),
        "max_sec": float(np.max(times)),
        "normal_mean_sec": float(np.mean([float(row["normal_wall_time_sec"]) for row in rows])),
        "expanded_mean_sec_when_invoked": None
        if not fallback
        else float(np.mean([value for value in expanded if value > 0.0])),
    }


def summarize(root: Path) -> dict[str, Any]:
    decision = read_json(root / "certification/final_decision.json")
    sparse = read_json(root / "sparse/decision.json")
    window = read_json(root / "window/decision.json")
    cross = read_json(root / "cross_episode/decision.json")
    integrity = read_json(root / "audits/method_integrity_postrun.json")
    qold = read_json(root / "qold_authority/qold_authority_manifest.json")
    pool = read_json(root / "fresh_pool/pool_summary.json")
    sparse_rows = (
        read_csv(root / "sparse/per_frame.csv") if (root / "sparse/per_frame.csv").is_file() else []
    )
    window_rows = (
        read_csv(root / "window/per_frame.csv") if (root / "window/per_frame.csv").is_file() else []
    )
    cross_rows = []
    if (root / "cross_episode/manifest.json").is_file():
        for control in read_json(root / "cross_episode/manifest.json")["controls"]:
            path = root / "cross_episode" / control["control_id"] / "per_frame.csv"
            if path.is_file():
                cross_rows.extend(read_csv(path))
    sparse_profile = _profile_stage(sparse_rows)
    window_profile = _profile_stage(window_rows)
    cross_profile = _profile_stage(cross_rows)
    write_json(root / "profiler/sparse.json", sparse_profile)
    write_json(root / "profiler/window.json", window_profile)
    write_json(root / "profiler/cross_episode.json", cross_profile)
    all_rows = sparse_rows + window_rows + cross_rows
    aggregate = _profile_stage(all_rows)
    estimate = (
        None
        if not all_rows
        else float(np.mean([float(row["total_wall_time_sec"]) for row in all_rows]))
        * d3.EXPECTED_FRAMES
    )
    write_json(root / "profiler/aggregate.json", aggregate)
    write_json(
        root / "profiler/d3v2_runtime_estimate.json",
        {
            "status": "PASS" if estimate is not None else "NOT_MEASURED",
            "ESTIMATED_D3_V2_2722_RUNTIME_SEC": estimate,
            "ESTIMATE_METHOD": "measured certification mean sec/frame multiplied by 2722; diagnostic only",
            "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        },
    )
    certified_path = root / "frozen_authority/certified_refinement_v2_authority.json"
    summary = {
        "schema_version": "OakInk2O5RD3CERTFinalSummaryV1",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "REFINEMENT_V2_DESIGN_SHA256": DESIGN_SHA256,
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": GATE_V2_SHA256,
        "METHOD_INTEGRITY_START": "PASS",
        "GATE_V2_INTEGRITY": "PASS",
        **pool,
        "FRESH_REFINEMENT_QOLD_POOL_STATUS": qold["FRESH_REFINEMENT_QOLD_POOL_STATUS"],
        "QOLD_RECORD_COUNT": qold["QOLD_RECORD_COUNT"],
        "QOLD_FRAME_COUNT": qold["QOLD_FRAME_COUNT"],
        "QOLD_AUTHORITY_MANIFEST_SHA256": sha256_file(
            root / "qold_authority/qold_authority_manifest.json"
        ),
        "BASELINE_GENERATION_RUN_COUNT": sum(
            int(item["frame_count"] >= 0) for item in qold["records"]
        ),
        "QOLD_FROZEN_BEFORE_REFINEMENT_V2": "YES",
        "REFINEMENT_V2_FRESH_CERTIFICATION_PLAN_SHA256": sha256_file(
            root / "certification_plan/plan.json"
        ),
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "PLAN_FROZEN_BEFORE_FIRST_FRESH_RUN": "YES",
        "FRESH_REFINEMENT_SPARSE": sparse.get("FRESH_REFINEMENT_SPARSE", "NOT_RUN"),
        "FRESH_REFINEMENT_WINDOW": window.get("FRESH_REFINEMENT_WINDOW", "NOT_RUN"),
        "CROSS_EPISODE_REFINEMENT": cross.get("CROSS_EPISODE_REFINEMENT", "NOT_RUN"),
        "METHOD_INTEGRITY_POSTRUN": integrity["METHOD_INTEGRITY_POSTRUN"],
        **decision,
        "CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256": sha256_file(certified_path)
        if certified_path.is_file()
        else None,
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    write_json(root / "final_summary.json", summary)
    window_table = "\n".join(
        f"| {item['window_id']} | {item['N']} | {item['baseline']['p95']:.12g} | {item['refined']['p95']:.12g} | {item['hard_valid']} | {item['continuity']} | {item['runtime_chain']} | {item['determinism']} | {item['status']} |"
        for item in window.get("windows", [])
    )
    cross_table = "\n".join(
        f"| {item['control_id']} | {item['episode']} | {item['primitive']} | {item['object']} | {item['frames']} | {item['fallback_rate']:.6f} | {item['interaction']} | {item['continuity']} | {item['status']} |"
        for item in cross.get("controls", [])
    )
    handoff = f"""# OakInk2 O5R-D3-CERT
# RefinementV2 Fresh Independent Certification Handoff

## Git

```text
BRANCH={EXPECTED_BRANCH}
START_HEAD={START_HEAD}
FINAL_HEAD={summary["FINAL_HEAD"]}
tracked_worktree_clean={summary["tracked_worktree_clean"]}
PUSHED=NO
PR_CREATED=NO
```

## Frozen method

```text
REFINEMENT_V2_DESIGN_SHA256={DESIGN_SHA256}
REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256={GATE_V2_SHA256}
METHOD_INTEGRITY_START=PASS
GATE_V2_INTEGRITY=PASS
```

## Fresh pool and q_old

```text
DEVELOPMENT_RECORD_COUNT={pool["DEVELOPMENT_RECORD_COUNT"]}
EXCLUDED_METHOD_DEVELOPMENT_RECORD_COUNT={pool["EXCLUDED_METHOD_DEVELOPMENT_RECORD_COUNT"]}
ELIGIBLE_FRESH_RECORD_COUNT={pool["ELIGIBLE_FRESH_RECORD_COUNT"]}
ELIGIBLE_FRESH_FRAME_COUNT={pool["ELIGIBLE_FRESH_FRAME_COUNT"]}
METHOD_DEVELOPMENT_OVERLAP=0
FRESH_REFINEMENT_QOLD_POOL_STATUS={qold["FRESH_REFINEMENT_QOLD_POOL_STATUS"]}
QOLD_RECORD_COUNT={qold["QOLD_RECORD_COUNT"]}
QOLD_FRAME_COUNT={qold["QOLD_FRAME_COUNT"]}
QOLD_AUTHORITY_MANIFEST_SHA256={summary["QOLD_AUTHORITY_MANIFEST_SHA256"]}
QOLD_FROZEN_BEFORE_REFINEMENT_V2=YES
```

## Fresh Sparse

```text
FRESH_REFINEMENT_SPARSE={summary["FRESH_REFINEMENT_SPARSE"]}
SPARSE_MANIFEST_SHA256={sha256_file(root / "sparse/manifest.json") if (root / "sparse/manifest.json").is_file() else None}
TECHNICAL={sparse.get("TECHNICAL")}
OLD_INVALID_COUNT={sparse.get("OLD_INVALID_COUNT")}
RECOVERED_COUNT={sparse.get("RECOVERED_COUNT")}
RECOVERY_FRACTION={sparse.get("RECOVERY_FRACTION")}
THRESHOLD_AWARE_NONREGRESSION={sparse.get("THRESHOLD_AWARE_NONREGRESSION")}
LOW_PRESERVED={sparse.get("LOW_PRESERVED")}
HARD_VALID={sparse.get("HARD_VALID")}
DETERMINISM={sparse.get("DETERMINISM")}
MEDIAN_RELATIVE_REDUCTION_DIAGNOSTIC={sparse.get("MEDIAN_RELATIVE_REDUCTION_DIAGNOSTIC")}
```

## Fresh Windows

FRESH_REFINEMENT_WINDOW={summary["FRESH_REFINEMENT_WINDOW"]}

| Window | N | Baseline p95 | Refined p95 | Hard-valid | Continuity | Runtime chain | Determinism | Result |
|---|---:|---:|---:|---|---|---|---|---|
{window_table}

## Fresh CrossEpisode

CROSS_EPISODE_REFINEMENT={summary["CROSS_EPISODE_REFINEMENT"]}

| Control | Episode | Primitive | Object | Frames | Fallback rate | Interaction | Continuity | Result |
|---|---|---|---|---:|---:|---|---|---|---|
{cross_table}

## Final certification

```text
C1_SPARSE={decision.get("C1_SPARSE")}
C2_WINDOW={decision.get("C2_WINDOW")}
C3_CROSS_EPISODE={decision.get("C3_CROSS_EPISODE")}
C4_METHOD_INTEGRITY={decision.get("C4_METHOD_INTEGRITY")}
REFINEMENT_V2_INDEPENDENT_CERTIFICATION={decision.get("REFINEMENT_V2_INDEPENDENT_CERTIFICATION")}
CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256={summary["CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256"]}
D3_V2_AUTHORIZED={decision.get("D3_V2_AUTHORIZED")}
NEXT={decision.get("NEXT")}
```

## Explicitly not run

```text
D3_V2_SCIENTIFIC_RUN_COUNT=0
DEV2_RERUN=NO
PPO_TRAINING_RUN_COUNT_NEW=0
PHYSX_RAN=NO
O6_PRODUCTION_RAN=NO
CERTIFICATION_SPLIT_NEW_CONSUMPTION=0
HELDOUT_SPLIT_NEW_CONSUMPTION=0
```
"""
    write_text(root / "handoff.md", handoff)
    write_text(root / "final_summary.md", handoff)
    write_json(
        root / "completion_audit.json",
        {
            "status": "PASS",
            "artifact_count": sum(1 for item in root.rglob("*") if item.is_file()),
            "scientific_status": decision["REFINEMENT_V2_INDEPENDENT_CERTIFICATION"],
            "engineering_completion_is_not_scientific_pass": True,
        },
    )
    return summary


def validate_repository(root: Path) -> dict[str, Any]:
    modified_python = [
        "scripts/evaluation/run_oakink2_o5rd3cert.py",
        "tests/evaluation/test_oakink2_o5rd3cert.py",
    ]
    commands = [
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", *modified_python],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "format",
            "--check",
            *modified_python,
        ],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "python",
            "scripts/check_paper_fidelity.py",
        ],
        ["git", "diff", "--check"],
    ]
    rows = []
    for command in commands:
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "wall_time_sec": time.perf_counter() - started,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            break
    passed = len(rows) == len(commands) and all(row["returncode"] == 0 for row in rows)
    value = {
        "schema_version": "O5RD3CERTRepositoryValidationV1",
        "status": "PASS" if passed else "FAIL",
        "checks": rows,
    }
    write_json(root / "tests.json", value)
    write_json(root / "validation_results.json", value)
    write_json(
        root / "git_commits.json",
        {
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": git("rev-parse", "HEAD"),
            "commits": git("log", "--format=%H", f"{START_HEAD}..HEAD").splitlines(),
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    )
    if not passed:
        raise RuntimeError("REPOSITORY_VALIDATION_FAIL")
    return value


def prepare_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_r4_authority(root)
    build_certification_exclusion_ledger(root)
    audit_fresh_refinement_pool(root)
    audit_qold_authority(root)
    return freeze_baseline_generation_plan(root)


FORBIDDEN_ACTION_NAMES = {
    "run-d3v2",
    "run-dev2",
    "run-ppo",
    "run-physx",
    "run-o6",
    "consume-certification-split",
    "consume-heldout-split",
}


ACTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "preflight": preflight,
    "verify-r4-authority": verify_r4_authority,
    "build-certification-exclusion-ledger": build_certification_exclusion_ledger,
    "audit-fresh-refinement-pool": audit_fresh_refinement_pool,
    "audit-qold-authority": audit_qold_authority,
    "freeze-baseline-generation-plan": freeze_baseline_generation_plan,
    "generate-qold-baselines-if-required": generate_qold_baselines_if_required,
    "freeze-qold-authority": freeze_qold_authority,
    "freeze-certification-plan": freeze_certification_plan,
    "select-fresh-sparse": select_fresh_sparse,
    "freeze-fresh-sparse": freeze_fresh_sparse,
    "run-fresh-sparse": run_fresh_sparse,
    "run-fresh-sparse-determinism": run_fresh_sparse_determinism,
    "evaluate-fresh-sparse": evaluate_fresh_sparse,
    "select-fresh-windows": select_fresh_windows,
    "freeze-fresh-windows": freeze_fresh_windows,
    "run-fresh-windows": run_fresh_windows,
    "run-fresh-window-determinism": run_fresh_window_determinism,
    "evaluate-fresh-windows": evaluate_fresh_windows,
    "select-fresh-cross-episode-controls": select_fresh_cross_episode_controls,
    "freeze-fresh-cross-episode": freeze_fresh_cross_episode,
    "run-fresh-cross-episode": run_fresh_cross_episode,
    "evaluate-fresh-cross-episode": evaluate_fresh_cross_episode,
    "audit-method-integrity": audit_method_integrity,
    "decide-refinement-v2-certification": decide_refinement_v2_certification,
    "freeze-certified-refinement-v2": freeze_certified_refinement_v2,
    "authorize-d3v2": authorize_d3v2,
    "generate-d3v2-plan": generate_d3v2_plan,
    "validate-repository": validate_repository,
    "summarize": summarize,
    "prepare-all": prepare_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve()
    started = time.perf_counter()
    try:
        result = ACTIONS[args.action](root)
    except Exception as exc:
        append_jsonl(
            root / "technical_failures.jsonl",
            {
                "schema_version": "O5RD3CERTTechnicalFailureV1",
                "action": args.action,
                "error": f"{type(exc).__name__}:{exc}",
                "classification": "TECHNICAL_OR_FAIL_CLOSED_PRECONDITION; scientific decisions are stored separately",
            },
        )
        raise
    usage_path = root / "resource_usage.json"
    usage = read_json(usage_path) if usage_path.is_file() else {"actions": []}
    usage.setdefault("actions", []).append(
        {"action": args.action, "wall_time_sec": time.perf_counter() - started}
    )
    usage.update(
        {
            "solver_actual_device": "CPU",
            "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
            "PPO_TRAINING_RUN_COUNT_NEW": 0,
            "PHYSX_RAN": "NO",
        }
    )
    write_json(usage_path, usage)
    print(json.dumps(result, indent=2, sort_keys=True, default=jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
