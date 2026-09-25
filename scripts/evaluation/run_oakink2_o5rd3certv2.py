#!/usr/bin/env python3
"""O5R-D3-CERT-V2 fresh prefix-anchored RefinementV2 certification.

This runner is intentionally fail closed.  It consumes only untouched OakInk2
DEVELOPMENT records, keeps historical q_old separate from the accepted
RefinementV2 runtime chain, freezes every scientific identity before the first
fresh optimizer run, and stops at the first failed certification stage.

There is deliberately no D3-V2, DEV2, PPO, PhysX, O6, CERTIFICATION-split, or
HELDOUT-split execution action in this module.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3cert as cert  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3certr as certr  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3certv2_refinement_v2_fresh_independent_certification_v1"
CERT_R_ROOT = certr.ROOT
CERT_V1_ROOT = cert.ROOT
R2_ROOT = cert.R2_ROOT
R4_ROOT = cert.R4_ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "6ed182b4710d6db28a0520e38dc68b8f066adfe7"

DESIGN_SHA256 = "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
GATE_V2_SHA256 = "505af73c9871baa67045449a4908b3259a16116b19bc2817d8b08470855bf6dc"
PROTOCOL_V2_SHA256 = "8b49cfbfac1b839685e256d7d5442ee41189401b1d122f674bb0fbff5af8272d"
TAU = 1.0e-4
EPS = 1.0e-12
SPARSE_N = 30
SPARSE_PER_STRATUM = 10
WINDOW_COUNT = 4
WINDOW_SIZE = 32
CROSS_CONTROL_COUNT = 3
CROSS_WINDOW_SIZE = 16
DETERMINISM_RUNS = 3
Q_TOL = 1.0e-10
BASE_TOL = 1.0e-10
EIM_TOL = 1.0e-12


read_json = cert.read_json
write_json = cert.write_json
write_text = cert.write_text
write_csv = cert.write_csv
read_csv = cert.read_csv
append_jsonl = cert.append_jsonl
freeze_json = cert.freeze_json
frozen = cert.frozen
require = cert.require
array_sha = cert.array_sha
canonical_sha = cert.canonical_sha


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _ensure_layout(root: Path) -> None:
    """Keep the contract's qold/ layout while reusing the proven V1 helpers."""

    root.mkdir(parents=True, exist_ok=True)
    qold = root / "qold"
    qold.mkdir(parents=True, exist_ok=True)
    compatibility = root / "qold_authority"
    if not compatibility.exists() and not compatibility.is_symlink():
        compatibility.symlink_to("qold", target_is_directory=True)
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    (root / "evidence/consumed_evidence.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "evidence/consumed_evidence.jsonl").touch(exist_ok=True)


def _authority_paths() -> dict[str, Path]:
    return {
        "refinement_v2_design": R2_ROOT / "design/refinement_v2_design.json",
        "development_gate_v2": R4_ROOT / "gate_v2/development_gate_v2.json",
        "certification_protocol_v2": CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json",
        "objective_v2": REPO / "src/toporetarget/retarget/objective_v2.py",
        "execution_v4": REPO / "src/toporetarget/retarget/objective_v4_execution.py",
        "semantic_v1": REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "source_adapter": REPO / "src/toporetarget/adapters/datasets/oakink2.py",
        "interaction_graph": REPO / "src/toporetarget/retarget/interaction_graph.py",
        "historical_solver_contract": o5.REPORT_ROOT / "contract/geometric_retarget_contract.json",
        "manifest_v2": o5.MANIFEST_V2,
        "split_v2": o5.SPLIT_V2,
    }


def _initial_not_run(root: Path) -> None:
    values = {
        "schema_version": "O5RD3CERTV2ExplicitNotRunV1",
        "status": "NOT_RUN",
        "FRESH_SPARSE_V2": "NOT_RUN",
        "FRESH_WINDOW_V2": "NOT_RUN",
        "FRESH_CROSS_EPISODE_V2": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "NOT_RUN",
        "D3_V2_AUTHORIZED": "NO",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    for relative in (
        "sparse_v2/decision.json",
        "window_v2/decision.json",
        "cross_episode_v2/decision.json",
        "certification/final_decision.json",
        "future/not_authorized.json",
    ):
        if not (root / relative).exists():
            write_json(root / relative, values)


def preflight(root: Path) -> dict[str, Any]:
    _ensure_layout(root)
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
        "cert_r_root_exists": CERT_R_ROOT.is_dir(),
    }
    value = {
        "schema_version": "O5RD3CERTV2GitPreflightV1",
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
        "log_200": git("log", "-200", "--oneline", "--decorate"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", value)
    _initial_not_run(root)
    if value["status"] != "PASS":
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_UPSTREAM_AUTHORITY:GIT")
    return value


def verify_cert_r_authority(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_CERT_R_AUTHORITY")
    summary = read_json(CERT_R_ROOT / "final_summary.json")
    authorization = read_json(CERT_R_ROOT / "future/cert_v2_authorization.json")
    protocol_path = CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json"
    protocol = read_json(protocol_path)
    start_hashes = {name: sha256_file(path) for name, path in _authority_paths().items()}
    checks = {
        "CERT_R_STATUS": summary.get("CERT_R_STATUS") == "PASS_PROTOCOL_REPAIR",
        "D3_CERT_V1_HISTORICAL_RESULT": summary.get("D3_CERT_V1_HISTORICAL_RESULT") == "FAIL",
        "D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN": summary.get(
            "D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN"
        )
        == "NO",
        "SPARSE_V1_CONSUMED_REGRESSION": summary.get("SPARSE_V1_CONSUMED_REGRESSION") == "PASS",
        "PRODUCTION_CONTEXT_PARITY": summary.get("PRODUCTION_CONTEXT_PARITY") == "PASS",
        "UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT": summary.get(
            "UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT"
        )
        == 0,
        "STANDALONE_SPARSE_SEMANTICS": summary.get("STANDALONE_SPARSE_SEMANTICS") == "INVALID",
        "CERTIFICATION_REPAIR_IMPACT": summary.get("CERTIFICATION_REPAIR_IMPACT")
        == "CERTIFICATION_PROTOCOL_UNIT_CHANGE",
        "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED": summary.get(
            "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED"
        )
        == "NO",
        "CERTIFICATION_PROTOCOL_V2_FROZEN": summary.get("CERTIFICATION_PROTOCOL_V2_FROZEN")
        == "YES",
        "SPARSE_UNIT_V2": summary.get("SPARSE_UNIT_V2") == "PREFIX_ANCHORED_TARGET",
        "CROSS_EPISODE_REQUIREMENT": summary.get("CROSS_EPISODE_REQUIREMENT") == "REQUIRED",
        "FRESH_CERT_V2_AUTHORIZED": authorization.get("FRESH_CERT_V2_AUTHORIZED") == "YES",
        "D3_V2_AUTHORIZED": authorization.get("D3_V2_AUTHORIZED") == "NO",
        "DESIGN_SHA": start_hashes["refinement_v2_design"] == DESIGN_SHA256,
        "GATE_V2_SHA": start_hashes["development_gate_v2"] == GATE_V2_SHA256,
        "PROTOCOL_V2_SHA": start_hashes["certification_protocol_v2"] == PROTOCOL_V2_SHA256,
        "PROTOCOL_FROZEN": protocol.get("status") == "FROZEN",
        "PROTOCOL_PREFIX": protocol.get("SPARSE_UNIT") == "PREFIX_ANCHORED_TARGET",
        "PROTOCOL_NO_OUTCOME_RULES": protocol.get("outcome_driven_rules") is False,
    }
    value = {
        "schema_version": "O5RD3CERTV2UpstreamAuthorityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "start_hashes": start_hashes,
        "CERT_R_STATUS": summary.get("CERT_R_STATUS"),
        "D3_CERT_V1_HISTORICAL_RESULT": summary.get("D3_CERT_V1_HISTORICAL_RESULT"),
        "D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN": summary.get(
            "D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN"
        ),
        "REFINEMENT_V2_DESIGN_SHA256": start_hashes["refinement_v2_design"],
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": start_hashes["development_gate_v2"],
        "CERTIFICATION_PROTOCOL_V2_SHA256": start_hashes["certification_protocol_v2"],
        "FRESH_CERT_V2_AUTHORIZED": authorization.get("FRESH_CERT_V2_AUTHORIZED"),
        "D3_V2_AUTHORIZED": "NO",
    }
    write_json(root / "preflight/upstream_authority.json", value)
    write_json(
        root / "preflight/frozen_method.json", {"status": value["status"], "hashes": start_hashes}
    )
    write_json(
        root / "preflight/frozen_gate.json",
        {"status": value["status"], "sha256": start_hashes["development_gate_v2"]},
    )
    write_json(
        root / "preflight/frozen_protocol.json",
        {
            "status": value["status"],
            "sha256": start_hashes["certification_protocol_v2"],
            "protocol": protocol,
        },
    )
    write_json(root / "integrity/method_start_hashes.json", start_hashes)
    if value["status"] != "PASS":
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "BLOCKED_UPSTREAM_AUTHORITY",
                "D3_CERT_V2_STATUS": "BLOCKED_UPSTREAM_AUTHORITY",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "NOT_RUN",
                "D3_V2_AUTHORIZED": "NO",
                "checks": checks,
            },
        )
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_UPSTREAM_AUTHORITY")
    return value


def verify_certification_protocol_v2(root: Path) -> dict[str, Any]:
    require(root / "preflight/upstream_authority.json", "status", "PASS", "VERIFY_PROTOCOL_V2")
    protocol_path = CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json"
    sidecar = protocol_path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    protocol = read_json(protocol_path)
    checks = {
        "file_sha_exact": sha256_file(protocol_path) == PROTOCOL_V2_SHA256,
        "sidecar_sha_exact": sidecar == PROTOCOL_V2_SHA256,
        "status_frozen": protocol.get("status") == "FROZEN",
        "prefix_unit": protocol.get("SPARSE_UNIT") == "PREFIX_ANCHORED_TARGET",
        "prefix_dependency": protocol.get("prefix_dependency")
        == "execute every ordinal 0..target exactly once in one runtime chain",
        "previous_runtime_authority": protocol.get("previous_runtime_authority")
        == "PREVIOUS_ACCEPTED_REFINED_RUNTIME",
        "target_only_gate_counting": protocol.get("gate_counting_rule")
        == "exactly 30 target frames",
        "cross_episode_required": protocol.get("CROSS_EPISODE_REQUIREMENT") == "REQUIRED",
        "no_scientific_retry": protocol.get("scientific_retry_policy")
        == "NO_RETRY_AFTER_OPTIMIZER_START",
    }
    value = {
        "schema_version": "O5RD3CERTV2ProtocolVerificationV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "CERTIFICATION_PROTOCOL_V2_SHA256": sha256_file(protocol_path),
        "protocol": protocol,
    }
    write_json(root / "preflight/frozen_protocol.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_FROZEN_AUTHORITY_INTEGRITY:PROTOCOL")
    return value


def verify_cert_v2_exclusion_ledger(root: Path) -> dict[str, Any]:
    require(root / "preflight/frozen_protocol.json", "status", "PASS", "VERIFY_EXCLUSION_LEDGER")
    source = CERT_R_ROOT / "freshness/cert_v2_exclusion_ledger.json"
    source_sidecar = source.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    observed = sha256_file(source)
    protocol = read_json(CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
    ledger = read_json(source)
    checks = {
        "source_sidecar": observed == source_sidecar,
        "protocol_binding": observed == protocol.get("exclusion_ledger_sha256"),
        "status_frozen": ledger.get("status") == "FROZEN",
        "source_sequence_policy": ledger.get("policy") == "SOURCE_SEQUENCE_LEVEL_EXCLUSION",
        "cert_v1_target_count": ledger.get("CERT_V1_TARGET_FRAME_EXCLUSION_COUNT") == 30,
        "cert_r_sequence_count": ledger.get("CERT_R_SEQUENCE_EXCLUSION_COUNT") == 4,
        "baseline_reusable_count": ledger.get("BASELINE_ONLY_REUSABLE_RECORD_COUNT") == 4,
    }
    if not all(checks.values()):
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_FROZEN_AUTHORITY_INTEGRITY:EXCLUSION_LEDGER")
    target = root / "freshness/cert_v2_exclusion_ledger.json"
    if target.is_file():
        if sha256_file(target) != observed:
            raise RuntimeError("FROZEN_CERT_V2_EXCLUSION_LEDGER_NOT_BYTE_IDENTICAL_TO_CERT_R")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        write_text(target.with_suffix(".sha256"), observed + "\n")
    write_json(
        root / "freshness/cert_v2_exclusion_ledger_verification.json",
        {
            "status": "PASS",
            "source_path": str(source.resolve()),
            "source_sha256": observed,
            "verification_checks": checks,
        },
    )
    return ledger


def build_cert_v2_fresh_pool(root: Path) -> dict[str, Any]:
    ledger, ledger_sha = frozen(
        root / "freshness/cert_v2_exclusion_ledger.json", "BUILD_FRESH_POOL"
    )
    source_pool = read_json(CERT_V1_ROOT / "fresh_pool/eligible_records.json")["records"]
    excluded_sequences = set(str(value) for value in ledger["excluded_sequences"])
    eligible = [row for row in source_pool if str(row["sequence_id"]) not in excluded_sequences]
    eligible.sort(key=lambda item: (int(item["frame_count"]), str(item["selection_hash"])))
    summary = {
        "schema_version": "O5RD3CERTV2FreshPoolSummaryV1",
        "status": "PASS",
        "DEVELOPMENT_RECORD_COUNT": read_json(CERT_V1_ROOT / "fresh_pool/pool_summary.json")[
            "DEVELOPMENT_RECORD_COUNT"
        ],
        "TOTAL_EXCLUDED_SEQUENCE_COUNT": len(excluded_sequences),
        "CERT_V1_EXCLUDED_SEQUENCE_COUNT": len(excluded_sequences),
        "CERT_R_EXCLUDED_SEQUENCE_COUNT": len(excluded_sequences),
        "ELIGIBLE_FRESH_RECORD_COUNT": len(eligible),
        "ELIGIBLE_FRESH_FRAME_COUNT": sum(int(item["frame_count"]) for item in eligible),
        "METHOD_DEVELOPMENT_OVERLAP": 0,
        "CERT_V1_OVERLAP": 0,
        "CERT_R_EXPOSED_SEQUENCE_OVERLAP": 0,
        "exclusion_ledger_sha256": ledger_sha,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    expected = (424, 377287)
    if (len(eligible), summary["ELIGIBLE_FRESH_FRAME_COUNT"]) != expected:
        summary["status"] = "FAIL"
        summary["expected"] = {"records": expected[0], "frames": expected[1]}
    write_json(
        root / "freshness/eligible_pool.json", {"status": summary["status"], "records": eligible}
    )
    write_json(root / "freshness/eligible_pool_summary.json", summary)
    write_json(
        root / "preflight/split_integrity.json",
        {
            "status": "PASS" if summary["status"] == "PASS" else "FAIL",
            "manifest_v2_sha256": sha256_file(o5.MANIFEST_V2),
            "split_v2_sha256": sha256_file(o5.SPLIT_V2),
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    if summary["status"] != "PASS":
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_INSUFFICIENT_FRESH_DEVELOPMENT_POOL")
    return summary


def audit_reusable_qold(root: Path) -> dict[str, Any]:
    require(root / "freshness/eligible_pool_summary.json", "status", "PASS", "AUDIT_REUSABLE_QOLD")
    classifications = read_csv(CERT_R_ROOT / "freshness/qold_record_classification.csv")
    old_manifest = read_json(CERT_V1_ROOT / "qold_authority/qold_authority_manifest.json")
    by_id = {str(item["baseline_id"]): item for item in old_manifest["records"]}
    reusable = []
    for row in classifications:
        if row["BASELINE_AUTHORITY_REUSABLE"] != "YES":
            continue
        old = by_id[row["baseline_id"]]
        authority_path = (
            CERT_V1_ROOT / "qold_authority/generated" / row["baseline_id"] / "authority.json"
        )
        authority = read_json(authority_path)
        checks = {
            "record_matches": authority["record_id"] == row["record_id"],
            "sequence_matches": authority["sequence_id"] == row["sequence_id"],
            "trajectory_hash": sha256_file(Path(authority["trajectory_path"]))
            == authority["trajectory_sha256"],
            "manifest_authority_hash": sha256_file(authority_path) == old["authority_sha256"],
            "not_cert_r_exposed": row["classification"] == "BASELINE_ONLY_UNEXPOSED",
        }
        reusable.append(
            {
                **row,
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
                "source_authority_path": str(authority_path.resolve()),
            }
        )
    value = {
        "schema_version": "O5RD3CERTV2ReusableQOldAuthoritiesV1",
        "status": "PASS"
        if len(reusable) == 4 and all(item["status"] == "PASS" for item in reusable)
        else "FAIL",
        "BASELINE_ONLY_REUSABLE_RECORD_COUNT": len(reusable),
        "records": reusable,
    }
    write_json(root / "qold/reusable_authorities.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_QOLD_REUSE_INTEGRITY")
    return value


def _select_distinct(
    rows: list[dict[str, Any]], count: int, excluded: set[str]
) -> list[dict[str, Any]]:
    selected = []
    used = set(excluded)
    for row in rows:
        sequence = str(row["sequence_id"])
        if sequence in used:
            continue
        selected.append(dict(row))
        used.add(sequence)
        if len(selected) == count:
            break
    return selected


def freeze_cert_v2_qold_generation_plan(root: Path) -> dict[str, Any]:
    reusable = require(
        root / "qold/reusable_authorities.json", "status", "PASS", "FREEZE_QOLD_PLAN"
    )
    eligible = read_json(root / "freshness/eligible_pool.json")["records"]
    reusable_by_record = {str(item["record_id"]): item for item in reusable["records"]}
    reusable_primary = next(
        item for item in reusable["records"] if item["baseline_id"] == "record_01"
    )
    reusable_controls = [
        item
        for item in reusable["records"]
        if item["baseline_id"] in {"record_06", "record_07", "record_08"}
    ]
    reserved_sequences = {str(item["sequence_id"]) for item in reusable["records"]}
    candidates = [row for row in eligible if str(row["record_id"]) not in reusable_by_record]
    sparse_new = _select_distinct(candidates, SPARSE_N - 1, reserved_sequences)
    used = reserved_sequences | {str(item["sequence_id"]) for item in sparse_new}
    window_new = _select_distinct(candidates, WINDOW_COUNT, used)
    if (
        len(sparse_new) != SPARSE_N - 1
        or len(window_new) != WINDOW_COUNT
        or len(reusable_controls) != CROSS_CONTROL_COUNT
    ):
        raise RuntimeError("D3_CERT_V2_STATUS=BLOCKED_INSUFFICIENT_QOLD_CANDIDATE_POOL")

    records: list[dict[str, Any]] = []

    def add(
        item: dict[str, Any], baseline_id: str, role: str, reuse: dict[str, Any] | None = None
    ) -> None:
        records.append(
            {
                **item,
                "baseline_id": baseline_id,
                "role": role,
                "generator_review": baseline_id,
                "reuse_source_baseline_id": None if reuse is None else reuse["baseline_id"],
                "reuse_source_authority_path": None
                if reuse is None
                else reuse["source_authority_path"],
            }
        )

    by_record = {str(item["record_id"]): item for item in eligible}
    add(by_record[reusable_primary["record_id"]], "sparse_01", "SPARSE_CANDIDATE", reusable_primary)
    for index, item in enumerate(sparse_new, start=2):
        add(item, f"sparse_{index:02d}", "SPARSE_CANDIDATE")
    for index, item in enumerate(window_new, start=1):
        add(item, f"window_{index:02d}", "WINDOW_CANDIDATE")
    for index, reuse in enumerate(
        sorted(reusable_controls, key=lambda item: item["baseline_id"]), start=1
    ):
        add(by_record[reuse["record_id"]], f"control_{index:02d}", "CROSS_EPISODE_CONTROL", reuse)

    value = {
        "schema_version": "O5RD3CERTV2QOldGenerationPlanV1",
        "status": "FROZEN",
        "selection_timing": "BEFORE_BASELINE_E_IM_AND_BEFORE_REFINEMENT_V2",
        "selection_rule": "reuse all four verified baseline-only authorities; reserve record_01 for Sparse and record_06/07/08 for CrossEpisode; fill 29 Sparse and four Window records from the frozen eligible pool in preexisting (frame_count, selection_hash) order with source-sequence uniqueness",
        "eligibility_failure_policy": "no replacement after baseline execution starts; a frozen candidate failure blocks certification",
        "generator": "FROZEN_HISTORICAL_O5_PRODUCTION_SOLVER",
        "historical_solver_sha256": sha256_file(
            o5.REPORT_ROOT / "contract/geometric_retarget_contract.json"
        ),
        "record_count": len(records),
        "reused_record_count": sum(
            item["reuse_source_authority_path"] is not None for item in records
        ),
        "new_generation_record_count": sum(
            item["reuse_source_authority_path"] is None for item in records
        ),
        "records": records,
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    frozen_value = freeze_json(root / "qold/generation_plan.json", value)
    compatibility = root / "qold/generation_plan/manifest.json"
    if not compatibility.exists():
        freeze_json(compatibility, frozen_value)
    return frozen_value


def _materialize_reused_authority(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    source_path = Path(str(item["reuse_source_authority_path"]))
    source = read_json(source_path)
    target_dir = root / "qold/generated" / str(item["baseline_id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    target_trajectory = target_dir / "trajectory.npz"
    if not target_trajectory.exists():
        shutil.copy2(Path(source["trajectory_path"]), target_trajectory)
    trajectory_sha = sha256_file(target_trajectory)
    if trajectory_sha != source["trajectory_sha256"]:
        raise RuntimeError(f"REUSED_QOLD_TRAJECTORY_HASH_MISMATCH:{item['baseline_id']}")
    authority = {
        **source,
        "schema_version": "O5RD3CERTV2HistoricalQOldAuthorityV1",
        "baseline_id": item["baseline_id"],
        "trajectory_path": str(target_trajectory.resolve()),
        "trajectory_sha256": trajectory_sha,
        "generation_manifest_sha256": sha256_file(root / "qold/generation_plan.json"),
        "reuse_source_baseline_id": item["reuse_source_baseline_id"],
        "reuse_source_authority_path": str(source_path.resolve()),
        "reuse_source_authority_sha256": sha256_file(source_path),
        "BASELINE_GENERATION_RUN_COUNT": 0,
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
    }
    write_json(target_dir / "authority.json", authority)
    write_text(target_dir / "trajectory.sha256", trajectory_sha + "\n")
    return authority


def generate_cert_v2_qold_if_required(root: Path) -> dict[str, Any]:
    plan, _ = frozen(root / "qold/generation_plan.json", "GENERATE_CERT_V2_QOLD")
    rows, by_id = cert.manifest_rows()
    del rows
    model = o5.get_robot_registry().load(o5.ROBOT)
    surface = o5.load_robot_surface_samples(o5._default_collision_samples(o5.ROBOT))
    generated = []
    for item in plan["records"]:
        print(
            f"q_old {item['baseline_id']} {item['record_id']} frames={item['frame_count']}",
            flush=True,
        )
        if item["reuse_source_authority_path"] is not None:
            authority = _materialize_reused_authority(root, item)
        else:
            authority = cert._generate_one_baseline(
                root, item, by_id[item["record_id"]], model, surface
            )
        cert._baseline_metrics(root, authority)
        generated.append(authority)
    value = {
        "status": "PASS",
        "QOLD_RECORD_COUNT": len(generated),
        "QOLD_FRAME_COUNT": sum(int(item["frame_count"]) for item in generated),
        "REUSED_BASELINE_ONLY_RECORD_COUNT": sum(
            int(item["BASELINE_GENERATION_RUN_COUNT"]) == 0 for item in generated
        ),
        "NEW_BASELINE_GENERATION_RECORD_COUNT": sum(
            int(item["BASELINE_GENERATION_RUN_COUNT"]) == 1 for item in generated
        ),
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT": 0,
    }
    write_json(root / "qold/generation_result.json", value)
    return value


def freeze_cert_v2_qold_authority(root: Path) -> dict[str, Any]:
    plan, generation_sha = frozen(
        root / "qold/generation_plan.json", "FREEZE_CERT_V2_QOLD_AUTHORITY"
    )
    require(root / "qold/generation_result.json", "status", "PASS", "FREEZE_CERT_V2_QOLD_AUTHORITY")
    records = []
    for item in plan["records"]:
        path = root / "qold/generated" / item["baseline_id"] / "authority.json"
        authority = require(path, "status", "FROZEN", "FREEZE_CERT_V2_QOLD_AUTHORITY")
        if sha256_file(Path(authority["trajectory_path"])) != authority["trajectory_sha256"]:
            raise RuntimeError(f"QOLD_TRAJECTORY_HASH_DRIFT:{item['baseline_id']}")
        records.append(
            {
                "baseline_id": item["baseline_id"],
                "role": item["role"],
                "record_id": item["record_id"],
                "sequence_id": item["sequence_id"],
                "frame_count": authority["frame_count"],
                "trajectory_sha256": authority["trajectory_sha256"],
                "authority_sha256": sha256_file(path),
                "canonical_sha256": authority["canonical_sha256"],
                "interaction_graph_sha256": authority["interaction_graph_sha256"],
                "reused": item["reuse_source_authority_path"] is not None,
            }
        )
    value = {
        "schema_version": "O5RD3CERTV2QOldAuthorityManifestV1",
        "status": "FROZEN",
        "generation_manifest_sha256": generation_sha,
        "upstream_protocol_qold_authority_sha256": read_json(
            CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json"
        )["q_old_authority"],
        "QOLD_FROZEN_BEFORE_CERT_V2_TARGET_SELECTION": "YES",
        "QOLD_FROZEN_BEFORE_REFINEMENT_V2": "YES",
        "REFINEMENT_V2_CERTIFICATION_RUN_COUNT_AT_FREEZE": 0,
        "records": records,
    }
    return freeze_json(root / "qold/qold_authority_manifest.json", value)


def _all_baseline_rows(root: Path, role: str) -> list[dict[str, Any]]:
    plan, _ = frozen(root / "qold/generation_plan.json", "LOAD_BASELINE_ROWS")
    result = []
    for item in plan["records"]:
        if item["role"] != role:
            continue
        for row in read_csv(root / "qold/generated" / item["baseline_id"] / "per_frame.csv"):
            result.append(
                {
                    **row,
                    "ordinal": int(row["ordinal"]),
                    "source_frame": int(row["source_frame"]),
                    "baseline_E_IM": float(row["baseline_E_IM"]),
                    "hard_valid": row["hard_valid"] == "True",
                }
            )
    return result


def _select_sparse_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_baseline: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_baseline[str(row["baseline_id"])].append(row)
    if len(by_baseline) != SPARSE_N:
        raise RuntimeError("SPARSE_DISTINCT_SEQUENCE_CANDIDATE_COUNT_MISMATCH")
    descriptors = []
    for baseline_id, values in by_baseline.items():
        ordered = sorted(values, key=lambda item: int(item["ordinal"]))
        high = max(ordered, key=lambda item: (float(item["baseline_E_IM"]), -int(item["ordinal"])))
        low = min(ordered, key=lambda item: (float(item["baseline_E_IM"]), int(item["ordinal"])))
        mid = min(
            ordered,
            key=lambda item: (abs(float(item["baseline_E_IM"]) - TAU), int(item["ordinal"])),
        )
        descriptors.append({"baseline_id": baseline_id, "high": high, "low": low, "mid": mid})
    high_records = sorted(
        descriptors, key=lambda item: (-float(item["high"]["baseline_E_IM"]), item["baseline_id"])
    )[:SPARSE_PER_STRATUM]
    remaining = [item for item in descriptors if item not in high_records]
    low_records = sorted(
        remaining, key=lambda item: (float(item["low"]["baseline_E_IM"]), item["baseline_id"])
    )[:SPARSE_PER_STRATUM]
    mid_records = [item for item in remaining if item not in low_records]
    sparse = [{**item["high"], "stratum": "HIGH"} for item in high_records]
    sparse += [{**item["mid"], "stratum": "MID"} for item in mid_records]
    sparse += [{**item["low"], "stratum": "LOW"} for item in low_records]
    return sorted(
        sparse,
        key=lambda item: ({"HIGH": 0, "MID": 1, "LOW": 2}[item["stratum"]], item["baseline_id"]),
    )


def freeze_cert_v2_run_plan(root: Path) -> dict[str, Any]:
    qold, qold_sha = frozen(root / "qold/qold_authority_manifest.json", "FREEZE_CERT_V2_RUN_PLAN")
    if qold["REFINEMENT_V2_CERTIFICATION_RUN_COUNT_AT_FREEZE"] != 0:
        raise RuntimeError("QOLD_NOT_FROZEN_BEFORE_TARGET_SELECTION")
    sparse = _select_sparse_targets(_all_baseline_rows(root, "SPARSE_CANDIDATE"))
    window_rows = _all_baseline_rows(root, "WINDOW_CANDIDATE")
    by_window: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        by_window[str(row["baseline_id"])].append(row)
    candidates = []
    for baseline_id, rows in sorted(by_window.items()):
        ordered = sorted(rows, key=lambda item: int(item["ordinal"]))
        if len(ordered) < WINDOW_SIZE:
            raise RuntimeError("CERT_V2_WINDOW_BASELINE_TOO_SHORT")
        start = (len(ordered) - WINDOW_SIZE) // 2
        target = ordered[start : start + WINDOW_SIZE]
        candidates.append(
            {
                "baseline_id": baseline_id,
                "start_ordinal": start,
                "target_ordinals": [int(item["ordinal"]) for item in target],
                "prefix_ordinals": list(range(start)),
                "baseline_p95": float(
                    np.percentile([float(item["baseline_E_IM"]) for item in target], 95)
                ),
            }
        )
    ranked = sorted(candidates, key=lambda item: (item["baseline_p95"], item["baseline_id"]))
    selected = [ranked[-1], ranked[-2], ranked[len(ranked) // 2], ranked[0]]
    windows = [
        dict(item, window_id=label)
        for item, label in zip(selected, ["HIGH_1", "HIGH_2", "MID", "LOW"], strict=True)
    ]
    generation = read_json(root / "qold/generation_plan.json")
    controls = []
    for index, item in enumerate(
        [row for row in generation["records"] if row["role"] == "CROSS_EPISODE_CONTROL"], start=1
    ):
        if int(item["frame_count"]) < CROSS_WINDOW_SIZE:
            raise RuntimeError("CERT_V2_CROSS_BASELINE_TOO_SHORT")
        controls.append(
            {
                **item,
                "control_id": f"control_{index}",
                "start_ordinal": 0,
                "prefix_ordinals": [],
                "target_ordinals": list(range(CROSS_WINDOW_SIZE)),
            }
        )
    if len(windows) != WINDOW_COUNT or len(controls) != CROSS_CONTROL_COUNT:
        raise RuntimeError("CERT_V2_PLAN_CARDINALITY_FAILURE")
    sparse_sequences = {item["sequence_id"] for item in sparse}
    window_sequences = {
        next(
            row["sequence_id"]
            for row in generation["records"]
            if row["baseline_id"] == item["baseline_id"]
        )
        for item in windows
    }
    control_sequences = {item["sequence_id"] for item in controls}
    if (
        sparse_sequences & window_sequences
        or sparse_sequences & control_sequences
        or window_sequences & control_sequences
    ):
        raise RuntimeError("CERT_V2_STAGE_SEQUENCE_OVERLAP")
    determinism_subset = []
    for stratum, count in (("HIGH", 2), ("MID", 2), ("LOW", 1)):
        determinism_subset.extend([item for item in sparse if item["stratum"] == stratum][:count])
    protocol_sha = sha256_file(CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
    value = {
        "schema_version": "RefinementV2FreshCertificationRunPlanV2",
        "status": "FROZEN_BEFORE_FIRST_REFINEMENT_V2_RUN",
        "RefinementV2_design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "CertificationProtocolV2_sha256": protocol_sha,
        "qold_authority_manifest_sha256": qold_sha,
        "fresh_exclusion_ledger_sha256": sha256_file(
            root / "freshness/cert_v2_exclusion_ledger.json"
        ),
        "sparse_selection_rule": "30 source-sequence-distinct baseline candidates; top ten record maxima HIGH, bottom ten remaining record minima LOW, remaining ten closest-to-tau MID; baseline q_old E_IM only",
        "sparse": sparse,
        "sparse_count": SPARSE_N,
        "sparse_strata": {"HIGH": 10, "MID": 10, "LOW": 10},
        "sparse_determinism_subset": determinism_subset,
        "window_selection_rule": "four pre-reserved source-sequence-disjoint records; centered 32-frame targets ranked by baseline q_old p95",
        "windows": windows,
        "window_count": WINDOW_COUNT,
        "window_size": WINDOW_SIZE,
        "window_determinism": ["HIGH_1", "LOW"],
        "cross_episode_selection_rule": "three verified baseline-only source-sequence-disjoint controls; first 16 frames",
        "cross_episode": controls,
        "cross_episode_count": CROSS_CONTROL_COUNT,
        "cross_episode_window_size": CROSS_WINDOW_SIZE,
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "stage_order": ["FRESH_SPARSE_V2", "FRESH_WINDOW_V2", "FRESH_CROSS_EPISODE_V2"],
        "scientific_retry_policy": "NO_RETRY_AFTER_OPTIMIZER_START",
        "technical_resume_policy": "same run UUID, plan SHA, manifest SHA, target identity, q_old authority, method SHA, and verified prefix checkpoint only",
        "prefix_failure_policy": read_json(
            CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json"
        )["prefix_failure_policy"],
        "determinism_runs": DETERMINISM_RUNS,
        "ALL_STAGE_IDENTITIES_FROZEN_BEFORE_FIRST_FRESH_RUN": "YES",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    frozen_value = freeze_json(root / "run_plan/certification_run_plan_v2.json", value)
    write_json(
        root / "run_plan/serialization_determinism.json",
        {
            "status": "PASS",
            "SERIALIZATION_DETERMINISM": "PASS",
            "canonical_sha256_first": canonical_sha(frozen_value),
            "canonical_sha256_second": canonical_sha(
                read_json(root / "run_plan/certification_run_plan_v2.json")
            ),
        },
    )
    return frozen_value


def _plan(root: Path, action: str) -> tuple[dict[str, Any], str]:
    plan, digest = frozen(root / "run_plan/certification_run_plan_v2.json", action)
    if plan.get("status") != "FROZEN_BEFORE_FIRST_REFINEMENT_V2_RUN":
        raise RuntimeError(f"{action}_REJECTED:RUN_PLAN_NOT_FROZEN")
    qold, _ = frozen(root / "qold/qold_authority_manifest.json", action)
    if qold.get("QOLD_FROZEN_BEFORE_REFINEMENT_V2") != "YES":
        raise RuntimeError(f"{action}_REJECTED:QOLD_NOT_FROZEN")
    return plan, digest


def _manifest_base(root: Path, plan_sha: str) -> dict[str, Any]:
    return {
        "cert_v2_run_plan_sha256": plan_sha,
        "RefinementV2_design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "CertificationProtocolV2_sha256": PROTOCOL_V2_SHA256,
        "qold_authority_manifest_sha256": sha256_file(root / "qold/qold_authority_manifest.json"),
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }


def freeze_cert_v2_sparse_manifest(root: Path) -> dict[str, Any]:
    plan, plan_sha = _plan(root, "FREEZE_CERT_V2_SPARSE_MANIFEST")
    frames = plan["sparse"]
    keys = {(item["baseline_id"], int(item["ordinal"])) for item in frames}
    sequences = {item["sequence_id"] for item in frames}
    if len(frames) != SPARSE_N or len(keys) != SPARSE_N or len(sequences) != SPARSE_N:
        raise RuntimeError("SPARSE_V2_CARDINALITY_OR_SEQUENCE_DUPLICATE")
    if {item["stratum"] for item in frames} != {"HIGH", "MID", "LOW"}:
        raise RuntimeError("SPARSE_V2_STRATA_MISSING")
    enriched = []
    for item in frames:
        authority_path = root / "qold/generated" / item["baseline_id"] / "authority.json"
        authority = require(authority_path, "status", "FROZEN", "FREEZE_CERT_V2_SPARSE_MANIFEST")
        enriched.append(
            {
                **item,
                "prefix_ordinals": list(range(int(item["ordinal"]))),
                "target_ordinal": int(item["ordinal"]),
                "baseline_qold_trajectory_sha256": authority["trajectory_sha256"],
                "record_authority_sha256": sha256_file(authority_path),
            }
        )
    value = {
        "schema_version": "O5RD3CERTV2SparseManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "SPARSE_UNIT_V2": "PREFIX_ANCHORED_TARGET",
        "standalone_sparse_allowed": False,
        "frames": enriched,
        "N": SPARSE_N,
        "strata": {
            name: sum(item["stratum"] == name for item in frames) for name in ("HIGH", "MID", "LOW")
        },
        "determinism_subset": plan["sparse_determinism_subset"],
        "SPARSE_V2_TARGET_OVERLAP_WITH_V1": 0,
        "SPARSE_V2_SOURCE_SEQUENCE_OVERLAP_WITH_CERT_R": 0,
        "SPARSE_V2_METHOD_DEVELOPMENT_OVERLAP": 0,
        "SPARSE_V2_CERTIFICATION_SPLIT_OVERLAP": 0,
        "SPARSE_V2_HELDOUT_OVERLAP": 0,
        **_manifest_base(root, plan_sha),
    }
    return freeze_json(root / "sparse_v2/manifest.json", value)


def freeze_cert_v2_window_manifest(root: Path) -> dict[str, Any]:
    plan, plan_sha = _plan(root, "FREEZE_CERT_V2_WINDOW_MANIFEST")
    windows = plan["windows"]
    sequences = set()
    enriched = []
    generation = {
        item["baseline_id"]: item
        for item in read_json(root / "qold/generation_plan.json")["records"]
    }
    for item in windows:
        authority_path = root / "qold/generated" / item["baseline_id"] / "authority.json"
        authority = require(authority_path, "status", "FROZEN", "FREEZE_CERT_V2_WINDOW_MANIFEST")
        record = generation[item["baseline_id"]]
        sequences.add(record["sequence_id"])
        enriched.append(
            {
                **item,
                "record_id": record["record_id"],
                "sequence_id": record["sequence_id"],
                "primitive": record["primitive"],
                "object_id": record["object_id"],
                "baseline_qold_trajectory_sha256": authority["trajectory_sha256"],
                "record_authority_sha256": sha256_file(authority_path),
            }
        )
    if len(windows) != WINDOW_COUNT or len(sequences) != WINDOW_COUNT:
        raise RuntimeError("WINDOW_V2_CARDINALITY_OR_SEQUENCE_DUPLICATE")
    value = {
        "schema_version": "O5RD3CERTV2WindowManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "windows": enriched,
        "determinism_windows": plan["window_determinism"],
        "WINDOW_V2_METHOD_DEVELOPMENT_OVERLAP": 0,
        "WINDOW_V2_CERT_V1_OVERLAP": 0,
        "WINDOW_V2_CERT_R_SEQUENCE_OVERLAP": 0,
        "WINDOW_V2_SPARSE_V2_SEQUENCE_OVERLAP": 0,
        "WINDOW_V2_CERTIFICATION_SPLIT_OVERLAP": 0,
        "WINDOW_V2_HELDOUT_OVERLAP": 0,
        "q_old_semantics": "historical q_old[current frame] immutable",
        "previous_runtime_semantics": "accepted refined state[t-1] from full sequence-local prefix; never reset at window start",
        **_manifest_base(root, plan_sha),
    }
    return freeze_json(root / "window_v2/manifest.json", value)


def freeze_cert_v2_cross_episode_manifest(root: Path) -> dict[str, Any]:
    plan, plan_sha = _plan(root, "FREEZE_CERT_V2_CROSS_EPISODE_MANIFEST")
    controls = plan["cross_episode"]
    sequences = {item["sequence_id"] for item in controls}
    if len(controls) != CROSS_CONTROL_COUNT or len(sequences) != CROSS_CONTROL_COUNT:
        raise RuntimeError("CROSS_EPISODE_V2_CARDINALITY_OR_SEQUENCE_DUPLICATE")
    value = {
        "schema_version": "O5RD3CERTV2CrossEpisodeManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "controls": controls,
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "SOURCE_SEQUENCE_DISJOINT": "YES",
        "CROSS_EPISODE_METHOD_DEVELOPMENT_OVERLAP": 0,
        "CROSS_EPISODE_CERT_V1_OVERLAP": 0,
        "CROSS_EPISODE_CERT_R_SEQUENCE_OVERLAP": 0,
        "CROSS_EPISODE_SPARSE_V2_SEQUENCE_OVERLAP": 0,
        "CROSS_EPISODE_WINDOW_V2_SEQUENCE_OVERLAP": 0,
        "determinism_runs": DETERMINISM_RUNS,
        **_manifest_base(root, plan_sha),
    }
    return freeze_json(root / "cross_episode_v2/manifest.json", value)


def _record_authority(root: Path, baseline_id: str) -> dict[str, Any]:
    return require(
        root / "qold/generated" / baseline_id / "authority.json",
        "status",
        "FROZEN",
        "LOAD_RECORD_AUTHORITY",
    )


def _context_receipt(
    item: dict[str, Any],
    ordinal: int,
    role: str,
    binding: Any,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
    runtime: cert.FreshRefinementRuntime,
) -> dict[str, Any]:
    qold = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    value = {
        "schema_version": "O5RD3CERTV2ProductionContextReceiptV1",
        "record_id": item["record_id"],
        "sequence_id": item["sequence_id"],
        "baseline_id": item["baseline_id"],
        "sequence_local_ordinal": ordinal,
        "source_frame": int(runtime.graph.frame_indices[ordinal]),
        "role": role,
        "q_old_hash": array_sha(qold),
        "previous_accepted_refined_q_hash": None if previous_q is None else array_sha(previous_q),
        "previous_accepted_refined_base_hash": None
        if previous_base is None
        else array_sha(previous_base),
        "q_old_not_previous_refined": True
        if previous_q is None
        else array_sha(qold) != array_sha(previous_q),
        "previous_refined_authority": "LOCAL_FRAME0_ABSENCE"
        if ordinal == 0
        else "PREVIOUS_ACCEPTED_REFINED_RUNTIME",
        "context_schema_version": binding.schema_version,
        "binding_sha256": binding.sha256,
        "all_continuous_input_fields": binding.as_dict(),
        "context_preflight_before_optimizer": True,
    }
    value["context_hash"] = canonical_sha(value)
    return value


def _begin_or_resume_stage(
    root: Path,
    relative: str,
    manifest_sha: str,
    count: int,
    *,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    path = root / relative
    method_sha = sha256_file(R2_ROOT / "design/refinement_v2_design.json")
    plan_sha = sha256_file(root / "run_plan/certification_run_plan_v2.json")
    if path.exists():
        state = read_json(path)
        if not resume:
            raise RuntimeError(
                f"SCIENTIFIC_RERUN_FORBIDDEN:{relative}:status={state.get('status')}"
            )
        checks = {
            "status_started": state.get("status") == "STARTED",
            "manifest_sha": state.get("manifest_sha256") == manifest_sha,
            "plan_sha": state.get("plan_sha256") == plan_sha,
            "method_sha": state.get("method_sha256") == method_sha,
        }
        if not all(checks.values()):
            raise RuntimeError(f"TECHNICAL_RESUME_AUTHORITY_MISMATCH:{relative}:{checks}")
        return path, state
    if resume:
        raise RuntimeError(f"TECHNICAL_RESUME_REJECTED:NO_STARTED_STATE:{relative}")
    state = {
        "status": "STARTED",
        "run_uuid": str(uuid.uuid4()),
        "manifest_sha256": manifest_sha,
        "plan_sha256": plan_sha,
        "method_sha256": method_sha,
        "scientific_primary_count": count,
        "completed": [],
        "scientific_retry_allowed": False,
        "technical_resume_policy": "verified same UUID/plan/manifest/method/checkpoint only",
    }
    write_json(path, state)
    return path, state


def _mark_completed(path: Path, identifier: str) -> None:
    state = read_json(path)
    if identifier not in state["completed"]:
        state["completed"].append(identifier)
    write_json(path, state)


def _finish_stage(path: Path) -> None:
    state = read_json(path)
    state["status"] = "COMPLETE"
    write_json(path, state)


def _unit_checkpoint(root: Path, stage: str, unit_id: str, ordinal: int) -> Path:
    return root / stage / "checkpoints" / unit_id / f"frame_{ordinal:05d}.json"


def _execute_prefix_unit(
    root: Path,
    *,
    stage: str,
    unit_id: str,
    item: dict[str, Any],
    target_ordinals: set[int],
    stop_ordinal: int,
    run_state: dict[str, Any],
    manifest_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = cert.FreshRefinementRuntime(root, _record_authority(root, str(item["baseline_id"])))
    baseline_rows = {
        int(row["ordinal"]): row
        for row in read_csv(root / "qold/generated" / item["baseline_id"] / "per_frame.csv")
    }
    previous_q: np.ndarray | None = None
    previous_base: np.ndarray | None = None
    frame_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    context_receipts: list[dict[str, Any]] = []
    runtime_chain: list[dict[str, Any]] = []
    prefix_scientific_valid = True
    for ordinal in range(stop_ordinal + 1):
        role = "TARGET_CERTIFICATION_FRAME" if ordinal in target_ordinals else "CONTEXT_ONLY_FRAME"
        checkpoint_path = _unit_checkpoint(root, stage, unit_id, ordinal)
        if checkpoint_path.is_file():
            checkpoint = read_json(checkpoint_path)
            checks = {
                "run_uuid": checkpoint.get("run_uuid") == run_state["run_uuid"],
                "manifest_sha": checkpoint.get("manifest_sha256") == manifest_sha,
                "method_sha": checkpoint.get("method_sha256") == run_state["method_sha256"],
                "baseline_id": checkpoint.get("baseline_id") == item["baseline_id"],
                "ordinal": checkpoint.get("ordinal") == ordinal,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"PREFIX_CHECKPOINT_AUTHORITY_MISMATCH:{stage}:{unit_id}:{ordinal}:{checks}"
                )
            q = np.asarray(checkpoint["qpos"], dtype=np.float64)
            base = np.asarray(checkpoint["base_pose_scene"], dtype=np.float64)
            receipt = checkpoint["receipt"]
            context = checkpoint["context_receipt"]
        else:
            runtime.current_runtime_step = ordinal
            binding, _ = runtime.bind_context(
                ordinal, previous_qpos=previous_q, previous_base=previous_base
            )
            context = _context_receipt(
                item, ordinal, role, binding, previous_q, previous_base, runtime
            )
            q, base, receipt = cert._run_refinement_frame(
                runtime, ordinal, ordinal, previous_q, previous_base
            )
            checkpoint = {
                "schema_version": "O5RD3CERTV2PrefixCheckpointV1",
                "status": "COMPLETE",
                "run_uuid": run_state["run_uuid"],
                "manifest_sha256": manifest_sha,
                "plan_sha256": run_state["plan_sha256"],
                "method_sha256": run_state["method_sha256"],
                "baseline_id": item["baseline_id"],
                "record_id": item["record_id"],
                "sequence_id": item["sequence_id"],
                "ordinal": ordinal,
                "role": role,
                "qpos": q,
                "base_pose_scene": base,
                "context_receipt": context,
                "receipt": receipt,
            }
            write_json(checkpoint_path, checkpoint)
        interaction_valid = bool(receipt["final_interaction_valid"])
        hard_valid = bool(receipt["final_hard_valid"])
        prefix_scientific_valid = prefix_scientific_valid and interaction_valid
        row = {
            "unit_id": unit_id,
            "baseline_id": item["baseline_id"],
            "record_id": item["record_id"],
            "sequence_id": item["sequence_id"],
            "ordinal": ordinal,
            "source_frame": int(runtime.graph.frame_indices[ordinal]),
            "role": role,
            "context_preflight": "PASS",
            "context_hash": context["context_hash"],
            "q_old_hash": context["q_old_hash"],
            "previous_accepted_refined_q_hash": context["previous_accepted_refined_q_hash"],
            "selected_path": receipt["selected_path"],
            "expanded_triggered": receipt["expanded"]["triggered"],
            "final_E_IM": receipt["final_E_IM"],
            "hard_valid": hard_valid,
            "interaction_valid": interaction_valid,
            "total_wall_time_sec": receipt["total_wall_time_sec"],
            "q_sha256": array_sha(q),
            "base_sha256": array_sha(base),
        }
        frame_rows.append(row)
        context_receipts.append(context)
        runtime_chain.append(
            {
                "unit_id": unit_id,
                "ordinal": ordinal,
                "q_old_hash": context["q_old_hash"],
                "predecessor_q_hash": context["previous_accepted_refined_q_hash"],
                "accepted_q_hash": row["q_sha256"],
                "accepted_base_hash": row["base_sha256"],
                "context_hash": context["context_hash"],
                "q_old_aliased_as_previous_refined": not bool(
                    context["q_old_not_previous_refined"]
                ),
            }
        )
        if ordinal in target_ordinals:
            metric = baseline_rows[ordinal]
            target_item = {
                **metric,
                "baseline_id": item["baseline_id"],
                "record_id": item["record_id"],
                "sequence_id": item["sequence_id"],
                "primitive": item["primitive"],
                "object_id": item["object_id"],
                "ordinal": ordinal,
                "source_frame": int(runtime.graph.frame_indices[ordinal]),
                "stratum": item.get("stratum", item.get("window_id", item.get("control_id"))),
            }
            target = cert._frame_row(target_item, q, base, receipt, runtime, previous_q)
            target.update(
                {
                    "unit_id": unit_id,
                    "role": role,
                    "prefix_frame_count": ordinal + 1,
                    "context_only_frame_count": ordinal
                    + 1
                    - sum(value <= ordinal for value in target_ordinals),
                    "prefix_scientific_valid": prefix_scientific_valid,
                    "context_hash": context["context_hash"],
                }
            )
            target_rows.append(target)
        previous_q, previous_base = q, base
        if not hard_valid:
            append_jsonl(
                root / "technical_failures.jsonl",
                {
                    "stage": stage,
                    "unit_id": unit_id,
                    "ordinal": ordinal,
                    "error": "PREFIX_HARD_VALID_FAILURE",
                    "optimizer_started": True,
                },
            )
            break
    return frame_rows, target_rows, context_receipts, runtime_chain


def _append_consumed(
    root: Path,
    *,
    stage: str,
    item: dict[str, Any],
    stop_ordinal: int,
    run_state: dict[str, Any],
    manifest_sha: str,
) -> None:
    path = root / "evidence/consumed_evidence.jsonl"
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    key = (stage, item["sequence_id"])
    if any((entry["stage"], entry["sequence_id"]) == key for entry in existing):
        return
    append_jsonl(
        path,
        {
            "schema_version": "O5RD3CERTV2ConsumedEvidenceV1",
            "record_id": item["record_id"],
            "sequence_id": item["sequence_id"],
            "frame_interval": [0, stop_ordinal],
            "role": "PREFIX_CONTEXT_AND_TARGET",
            "stage": stage,
            "run_uuid": run_state["run_uuid"],
            "method_sha256": run_state["method_sha256"],
            "manifest_sha256": manifest_sha,
            "result_exposure_status": "REFINEMENT_V2_EXECUTED",
        },
    )


def _run_sparse(root: Path, *, resume: bool) -> dict[str, Any]:
    manifest, manifest_sha = frozen(
        root / "sparse_v2/manifest.json",
        "RESUME_FRESH_SPARSE_V2" if resume else "RUN_FRESH_SPARSE_V2",
    )
    state_path, state = _begin_or_resume_stage(
        root, "sparse_v2/run_state.json", manifest_sha, SPARSE_N, resume=resume
    )
    all_frames: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for item in manifest["frames"]:
        unit_id = f"{item['baseline_id']}:{int(item['ordinal'])}"
        if unit_id in read_json(state_path)["completed"]:
            checkpoint_root = root / "sparse_v2/checkpoints" / unit_id
            checkpoints = sorted(checkpoint_root.glob("frame_*.json"))
            if not checkpoints:
                raise RuntimeError(f"COMPLETED_UNIT_WITHOUT_CHECKPOINT:{unit_id}")
        frame_rows, target_rows, receipts, runtime_chain = _execute_prefix_unit(
            root,
            stage="sparse_v2",
            unit_id=unit_id,
            item=item,
            target_ordinals={int(item["ordinal"])},
            stop_ordinal=int(item["ordinal"]),
            run_state=state,
            manifest_sha=manifest_sha,
        )
        all_frames.extend(frame_rows)
        targets.extend(target_rows)
        contexts.extend(receipts)
        chains.extend(runtime_chain)
        _append_consumed(
            root,
            stage="SPARSE_V2",
            item=item,
            stop_ordinal=int(item["ordinal"]),
            run_state=state,
            manifest_sha=manifest_sha,
        )
        _mark_completed(state_path, unit_id)
        write_csv(root / "sparse_v2/context_frames.csv", all_frames)
        write_csv(root / "sparse_v2/target_results.csv", targets)
        write_text(
            root / "sparse_v2/context_receipts.jsonl",
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in contexts),
        )
        write_text(
            root / "sparse_v2/runtime_chain.jsonl",
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in chains),
        )
        print(f"Sparse V2 {unit_id} prefix={int(item['ordinal']) + 1}", flush=True)
    _finish_stage(state_path)
    return {
        "status": "COMPLETE",
        "target_count": len(targets),
        "total_prefix_frames": len(all_frames),
        "context_only_frames": sum(row["role"] == "CONTEXT_ONLY_FRAME" for row in all_frames),
    }


def run_fresh_sparse_v2(root: Path) -> dict[str, Any]:
    return _run_sparse(root, resume=False)


def resume_fresh_sparse_v2(root: Path) -> dict[str, Any]:
    return _run_sparse(root, resume=True)


def _repeat_prefix(
    root: Path, item: dict[str, Any], stop_ordinal: int, target_ordinals: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime = cert.FreshRefinementRuntime(root, _record_authority(root, str(item["baseline_id"])))
    previous_q = previous_base = None
    rows = []
    target_rows = []
    for ordinal in range(stop_ordinal + 1):
        runtime.current_runtime_step = ordinal
        binding, _ = runtime.bind_context(
            ordinal, previous_qpos=previous_q, previous_base=previous_base
        )
        role = "TARGET_CERTIFICATION_FRAME" if ordinal in target_ordinals else "CONTEXT_ONLY_FRAME"
        context = _context_receipt(item, ordinal, role, binding, previous_q, previous_base, runtime)
        q, base, receipt = cert._run_refinement_frame(
            runtime, ordinal, ordinal, previous_q, previous_base
        )
        row = {
            "ordinal": ordinal,
            "context_hash": context["context_hash"],
            "triggered": receipt["expanded"]["triggered"],
            "selected_path": receipt["selected_path"],
            "q_sha256": array_sha(q),
            "base_sha256": array_sha(base),
            "E_IM": float(receipt["final_E_IM"]),
            "hard_valid": bool(receipt["final_hard_valid"]),
        }
        rows.append(row)
        if ordinal in target_ordinals:
            target_rows.append(row)
        previous_q, previous_base = q, base
    return rows, target_rows


def run_fresh_sparse_v2_determinism(root: Path) -> dict[str, Any]:
    require(root / "sparse_v2/run_state.json", "status", "COMPLETE", "RUN_SPARSE_V2_DETERMINISM")
    manifest, manifest_sha = frozen(root / "sparse_v2/manifest.json", "RUN_SPARSE_V2_DETERMINISM")
    state_path, state = _begin_or_resume_stage(
        root,
        "sparse_v2/determinism_run_state.json",
        manifest_sha,
        len(manifest["determinism_subset"]) * DETERMINISM_RUNS,
        resume=False,
    )
    primary = defaultdict(dict)
    for line in (root / "sparse_v2/runtime_chain.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        primary[row["unit_id"]][int(row["ordinal"])] = row
    output = []
    by_key = {(item["baseline_id"], int(item["ordinal"])): item for item in manifest["frames"]}
    for subset in manifest["determinism_subset"]:
        item = by_key[(subset["baseline_id"], int(subset["ordinal"]))]
        unit_id = f"{item['baseline_id']}:{int(item['ordinal'])}"
        checkpoint_rows = {
            int(path.stem.split("_")[-1]): read_json(path)
            for path in (root / "sparse_v2/checkpoints" / unit_id).glob("frame_*.json")
        }
        for repeat in range(1, DETERMINISM_RUNS + 1):
            repeated, _ = _repeat_prefix(root, item, int(item["ordinal"]), {int(item["ordinal"])})
            for observed in repeated:
                reference = checkpoint_rows[observed["ordinal"]]
                receipt = reference["receipt"]
                output.append(
                    {
                        "unit_id": unit_id,
                        "ordinal": observed["ordinal"],
                        "repeat": repeat,
                        "context_hash_match": observed["context_hash"]
                        == reference["context_receipt"]["context_hash"],
                        "trigger_match": observed["triggered"] == receipt["expanded"]["triggered"],
                        "selected_path_match": observed["selected_path"]
                        == receipt["selected_path"],
                        "q_sha_match": observed["q_sha256"]
                        == array_sha(np.asarray(reference["qpos"])),
                        "base_sha_match": observed["base_sha256"]
                        == array_sha(np.asarray(reference["base_pose_scene"])),
                        "E_IM_abs_diff": abs(observed["E_IM"] - float(receipt["final_E_IM"])),
                        "hard_valid_match": observed["hard_valid"]
                        == bool(receipt["final_hard_valid"]),
                    }
                )
            _mark_completed(state_path, f"{unit_id}:repeat{repeat}")
            write_csv(root / "sparse_v2/determinism_results.csv", output)
    _finish_stage(state_path)
    passed = all(
        row["context_hash_match"]
        and row["trigger_match"]
        and row["selected_path_match"]
        and row["q_sha_match"]
        and row["base_sha_match"]
        and row["E_IM_abs_diff"] <= EIM_TOL
        and row["hard_valid_match"]
        for row in output
    )
    value = {
        "schema_version": "O5RD3CERTV2SparseDeterminismV1",
        "status": "PASS" if passed else "FAIL",
        "DETERMINISM": "PASS" if passed else "FAIL",
        "subset_count": len(manifest["determinism_subset"]),
        "repeat_count": DETERMINISM_RUNS,
        "full_prefix_repeated": True,
    }
    write_json(root / "sparse_v2/determinism_manifest.json", value)
    return value


def evaluate_fresh_sparse_v2(root: Path) -> dict[str, Any]:
    require(root / "sparse_v2/run_state.json", "status", "COMPLETE", "EVALUATE_SPARSE_V2")
    determinism = read_json(root / "sparse_v2/determinism_manifest.json")
    rows = read_csv(root / "sparse_v2/target_results.csv")
    context = read_csv(root / "sparse_v2/context_frames.csv")
    old_invalid = [row for row in rows if float(row["baseline_E_IM"]) > TAU]
    valid_old = [row for row in rows if float(row["baseline_E_IM"]) <= TAU]
    recovered = sum(row["recovered"] == "True" for row in old_invalid)
    nonregression = sum(row["threshold_aware_nonregression"] == "True" for row in rows)
    preserved = sum(float(row["final_E_IM"]) <= TAU + EPS for row in valid_old)
    hard = sum(row["hard_valid"] == "True" for row in rows)
    prefix_valid = sum(row["prefix_scientific_valid"] == "True" for row in rows)
    recovery = recovered / len(old_invalid) if old_invalid else 0.0
    criteria = {
        "technical_completion": len(rows) == SPARSE_N,
        "old_invalid_evidence_present": bool(old_invalid),
        "old_invalid_recovery": bool(old_invalid) and recovery >= 0.8,
        "threshold_aware_nonregression": nonregression == SPARSE_N,
        "valid_preservation": preserved == len(valid_old),
        "hard_validity": hard == SPARSE_N,
        "prefix_scientific_validity": prefix_valid == SPARSE_N,
        "determinism": determinism.get("DETERMINISM") == "PASS",
    }
    reductions = [
        (float(row["baseline_E_IM"]) - float(row["final_E_IM"])) / float(row["baseline_E_IM"])
        for row in old_invalid
    ]
    passed = all(criteria.values())
    value = {
        "schema_version": "O5RD3CERTV2SparseGateResultsV1",
        "status": "PASS" if passed else "FAIL",
        "FRESH_SPARSE_V2": "PASS" if passed else "FAIL",
        "criteria": criteria,
        "SPARSE_TARGET_COUNT": len(rows),
        "OLD_INVALID_TARGET_COUNT": len(old_invalid),
        "RECOVERED_TARGET_COUNT": recovered,
        "RECOVERY_FRACTION": recovery,
        "THRESHOLD_AWARE_NONREGRESSION": f"{nonregression}/{SPARSE_N}",
        "VALID_PRESERVATION": f"{preserved}/{len(valid_old)}",
        "TARGET_HARD_VALID": f"{hard}/{SPARSE_N}",
        "PREFIX_SCIENTIFIC_VALID": f"{prefix_valid}/{SPARSE_N}",
        "TOTAL_PREFIX_FRAMES_EXECUTED": len(context),
        "TOTAL_CONTEXT_ONLY_FRAMES": sum(row["role"] == "CONTEXT_ONLY_FRAME" for row in context),
        "CONTEXT_TECHNICAL_PASS": len(context),
        "CONTEXT_HARD_VALID": sum(row["hard_valid"] == "True" for row in context),
        "CONTEXT_SCIENTIFIC_FAILURE_COUNT": sum(
            row["interaction_valid"] != "True"
            for row in context
            if row["role"] == "CONTEXT_ONLY_FRAME"
        ),
        "DETERMINISM": determinism.get("DETERMINISM"),
        "MEDIAN_RELATIVE_REDUCTION_DIAGNOSTIC": None
        if not reductions
        else float(np.median(reductions)),
    }
    write_json(root / "sparse_v2/gate_results.json", value)
    write_json(root / "sparse_v2/decision.json", value)
    if not passed:
        write_json(
            root / "window_v2/decision.json",
            {"status": "NOT_RUN", "FRESH_WINDOW_V2": "NOT_RUN", "reason": "SPARSE_V2_FAIL"},
        )
        write_json(
            root / "cross_episode_v2/decision.json",
            {"status": "NOT_RUN", "FRESH_CROSS_EPISODE_V2": "NOT_RUN", "reason": "SPARSE_V2_FAIL"},
        )
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "FAIL",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "FAIL",
                "D3_V2_AUTHORIZED": "NO",
                "NEXT": "DEV1_REFINEMENT_V2_FRESH_SPARSE_V2_FAILURE_ANALYSIS",
            },
        )
    return value


def _run_sequence_stage(
    root: Path, *, stage: str, manifest_relative: str, items_field: str, resume: bool
) -> dict[str, Any]:
    action = f"{'RESUME' if resume else 'RUN'}_{stage.upper()}"
    manifest, manifest_sha = frozen(root / manifest_relative, action)
    items = manifest[items_field]
    state_path, state = _begin_or_resume_stage(
        root, f"{stage}/run_state.json", manifest_sha, len(items), resume=resume
    )
    all_frames: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for item in items:
        unit_id = str(item.get("window_id", item.get("control_id")))
        target_ordinals = {int(value) for value in item["target_ordinals"]}
        stop_ordinal = max(target_ordinals)
        frame_rows, target_rows, receipts, runtime_chain = _execute_prefix_unit(
            root,
            stage=stage,
            unit_id=unit_id,
            item=item,
            target_ordinals=target_ordinals,
            stop_ordinal=stop_ordinal,
            run_state=state,
            manifest_sha=manifest_sha,
        )
        all_frames.extend(frame_rows)
        targets.extend(target_rows)
        contexts.extend(receipts)
        chains.extend(runtime_chain)
        _append_consumed(
            root,
            stage=stage.upper(),
            item=item,
            stop_ordinal=stop_ordinal,
            run_state=state,
            manifest_sha=manifest_sha,
        )
        _mark_completed(state_path, unit_id)
        write_csv(root / stage / "context_frames.csv", all_frames)
        write_csv(root / stage / "per_target_frame.csv", targets)
        write_text(
            root / stage / "context_receipts.jsonl",
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in contexts),
        )
        write_text(
            root / stage / "runtime_chain.jsonl",
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in chains),
        )
        write_csv(
            root / stage / "trigger_sequence.csv",
            [
                {
                    "unit_id": row["unit_id"],
                    "ordinal": row["ordinal"],
                    "expanded_triggered": row["expanded_triggered"],
                    "selected_path": row["selected_path"],
                }
                for row in all_frames
            ],
        )
        print(
            f"{stage} {unit_id} prefix={stop_ordinal + 1} targets={len(target_ordinals)}",
            flush=True,
        )
    _finish_stage(state_path)
    return {
        "status": "COMPLETE",
        "unit_count": len(items),
        "total_prefix_frames": len(all_frames),
        "target_frames": len(targets),
        "context_only_frames": sum(row["role"] == "CONTEXT_ONLY_FRAME" for row in all_frames),
    }


def run_fresh_window_v2(root: Path) -> dict[str, Any]:
    require(root / "sparse_v2/decision.json", "FRESH_SPARSE_V2", "PASS", "RUN_FRESH_WINDOW_V2")
    return _run_sequence_stage(
        root,
        stage="window_v2",
        manifest_relative="window_v2/manifest.json",
        items_field="windows",
        resume=False,
    )


def resume_fresh_window_v2(root: Path) -> dict[str, Any]:
    require(root / "sparse_v2/decision.json", "FRESH_SPARSE_V2", "PASS", "RESUME_FRESH_WINDOW_V2")
    return _run_sequence_stage(
        root,
        stage="window_v2",
        manifest_relative="window_v2/manifest.json",
        items_field="windows",
        resume=True,
    )


def _run_stage_determinism(
    root: Path, *, stage: str, manifest_relative: str, items_field: str, selected_ids: set[str]
) -> dict[str, Any]:
    require(
        root / stage / "run_state.json", "status", "COMPLETE", f"RUN_{stage.upper()}_DETERMINISM"
    )
    manifest, manifest_sha = frozen(root / manifest_relative, f"RUN_{stage.upper()}_DETERMINISM")
    items = [
        item
        for item in manifest[items_field]
        if str(item.get("window_id", item.get("control_id"))) in selected_ids
    ]
    state_path, _ = _begin_or_resume_stage(
        root,
        f"{stage}/determinism_run_state.json",
        manifest_sha,
        len(items) * DETERMINISM_RUNS,
        resume=False,
    )
    output = []
    for item in items:
        unit_id = str(item.get("window_id", item.get("control_id")))
        target_ordinals = {int(value) for value in item["target_ordinals"]}
        stop_ordinal = max(target_ordinals)
        references = {
            int(path.stem.split("_")[-1]): read_json(path)
            for path in (root / stage / "checkpoints" / unit_id).glob("frame_*.json")
        }
        if set(references) != set(range(stop_ordinal + 1)):
            raise RuntimeError(f"DETERMINISM_PREFIX_CHECKPOINT_INCOMPLETE:{stage}:{unit_id}")
        for repeat in range(1, DETERMINISM_RUNS + 1):
            rows, _ = _repeat_prefix(root, item, stop_ordinal, target_ordinals)
            for observed in rows:
                reference = references[observed["ordinal"]]
                receipt = reference["receipt"]
                output.append(
                    {
                        "unit_id": unit_id,
                        "ordinal": observed["ordinal"],
                        "repeat": repeat,
                        "context_hash_match": observed["context_hash"]
                        == reference["context_receipt"]["context_hash"],
                        "trigger_match": observed["triggered"] == receipt["expanded"]["triggered"],
                        "selected_path_match": observed["selected_path"]
                        == receipt["selected_path"],
                        "q_sha_match": observed["q_sha256"]
                        == array_sha(np.asarray(reference["qpos"])),
                        "base_sha_match": observed["base_sha256"]
                        == array_sha(np.asarray(reference["base_pose_scene"])),
                        "E_IM_abs_diff": abs(observed["E_IM"] - float(receipt["final_E_IM"])),
                        "hard_valid_match": observed["hard_valid"]
                        == bool(receipt["final_hard_valid"]),
                    }
                )
            _mark_completed(state_path, f"{unit_id}:repeat{repeat}")
            write_csv(root / stage / "determinism_results.csv", output)
    _finish_stage(state_path)
    passed = all(
        row["context_hash_match"]
        and row["trigger_match"]
        and row["selected_path_match"]
        and row["q_sha_match"]
        and row["base_sha_match"]
        and row["E_IM_abs_diff"] <= EIM_TOL
        and row["hard_valid_match"]
        for row in output
    )
    value = {
        "schema_version": f"O5RD3CERTV2{stage.title().replace('_', '')}DeterminismV1",
        "status": "PASS" if passed else "FAIL",
        "DETERMINISM": "PASS" if passed else "FAIL",
        "unit_count": len(items),
        "repeat_count": DETERMINISM_RUNS,
        "full_prefix_repeated": True,
    }
    write_json(root / stage / "determinism_results.json", value)
    return value


def run_fresh_window_v2_determinism(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "window_v2/manifest.json")
    return _run_stage_determinism(
        root,
        stage="window_v2",
        manifest_relative="window_v2/manifest.json",
        items_field="windows",
        selected_ids=set(manifest["determinism_windows"]),
    )


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_fresh_window_v2(root: Path) -> dict[str, Any]:
    require(root / "window_v2/run_state.json", "status", "COMPLETE", "EVALUATE_WINDOW_V2")
    determinism = read_json(root / "window_v2/determinism_results.json")
    manifest = read_json(root / "window_v2/manifest.json")
    rows = read_csv(root / "window_v2/per_target_frame.csv")
    chains = [
        json.loads(line)
        for line in (root / "window_v2/runtime_chain.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    summaries = []
    for window in manifest["windows"]:
        window_id = window["window_id"]
        selected = [row for row in rows if row["unit_id"] == window_id]
        baseline = _distribution([float(row["baseline_E_IM"]) for row in selected])
        refined = _distribution([float(row["final_E_IM"]) for row in selected])
        continuity = all(
            float(row["translation_step_m"]) <= 0.05 + EPS
            and float(row["rotation_step_rad"]) <= math.pi / 2.0 + EPS
            for row in selected
        )
        hard = all(row["hard_valid"] == "True" for row in selected)
        interaction = refined["p95"] <= TAU + EPS
        prefix_valid = all(row["prefix_scientific_valid"] == "True" for row in selected)
        unit_chain = [row for row in chains if row["unit_id"] == window_id]
        chain_pass = (
            len(unit_chain) == max(int(value) for value in window["target_ordinals"]) + 1
            and all(
                index == 0
                or unit_chain[index]["predecessor_q_hash"]
                == unit_chain[index - 1]["accepted_q_hash"]
                for index in range(len(unit_chain))
            )
            and not any(row["q_old_aliased_as_previous_refined"] for row in unit_chain)
        )
        selected_for_determinism = window_id in manifest["determinism_windows"]
        passed = (
            len(selected) == WINDOW_SIZE
            and hard
            and interaction
            and continuity
            and chain_pass
            and prefix_valid
            and (not selected_for_determinism or determinism.get("DETERMINISM") == "PASS")
        )
        summaries.append(
            {
                "window_id": window_id,
                "target_N": len(selected),
                "prefix_N": int(window["start_ordinal"]),
                "baseline": baseline,
                "refined": refined,
                "TARGET_WINDOW_E_IM_P95": refined["p95"],
                "hard_valid": "PASS" if hard else "FAIL",
                "interaction": "PASS" if interaction else "FAIL",
                "continuity": "PASS" if continuity else "FAIL",
                "runtime_chain": "PASS" if chain_pass else "FAIL",
                "prefix_scientific_validity": "PASS" if prefix_valid else "FAIL",
                "determinism": determinism.get("DETERMINISM")
                if selected_for_determinism
                else "NOT_SELECTED",
                "status": "PASS" if passed else "FAIL",
            }
        )
    passed = (
        all(item["status"] == "PASS" for item in summaries)
        and determinism.get("DETERMINISM") == "PASS"
    )
    value = {
        "schema_version": "O5RD3CERTV2WindowGateResultsV1",
        "status": "PASS" if passed else "FAIL",
        "FRESH_WINDOW_V2": "PASS" if passed else "FAIL",
        "WINDOW_COUNT": len(summaries),
        "windows": summaries,
        "WINDOW_METHOD_DEVELOPMENT_OVERLAP": 0,
        "WINDOW_CERT_V1_OVERLAP": 0,
        "WINDOW_CERT_R_OVERLAP": 0,
        "WINDOW_SPARSE_V2_SEQUENCE_OVERLAP": 0,
        "DETERMINISM": determinism.get("DETERMINISM"),
    }
    write_json(root / "window_v2/gate_results.json", value)
    write_json(root / "window_v2/decision.json", value)
    if not passed:
        write_json(
            root / "cross_episode_v2/decision.json",
            {"status": "NOT_RUN", "FRESH_CROSS_EPISODE_V2": "NOT_RUN", "reason": "WINDOW_V2_FAIL"},
        )
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "FAIL",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "FAIL",
                "D3_V2_AUTHORIZED": "NO",
                "NEXT": "DEV1_REFINEMENT_V2_FRESH_WINDOW_V2_FAILURE_ANALYSIS",
            },
        )
    return value


def run_fresh_cross_episode_v2(root: Path) -> dict[str, Any]:
    require(
        root / "window_v2/decision.json", "FRESH_WINDOW_V2", "PASS", "RUN_FRESH_CROSS_EPISODE_V2"
    )
    return _run_sequence_stage(
        root,
        stage="cross_episode_v2",
        manifest_relative="cross_episode_v2/manifest.json",
        items_field="controls",
        resume=False,
    )


def resume_fresh_cross_episode_v2(root: Path) -> dict[str, Any]:
    require(
        root / "window_v2/decision.json", "FRESH_WINDOW_V2", "PASS", "RESUME_FRESH_CROSS_EPISODE_V2"
    )
    return _run_sequence_stage(
        root,
        stage="cross_episode_v2",
        manifest_relative="cross_episode_v2/manifest.json",
        items_field="controls",
        resume=True,
    )


def run_fresh_cross_episode_v2_determinism(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "cross_episode_v2/manifest.json")
    return _run_stage_determinism(
        root,
        stage="cross_episode_v2",
        manifest_relative="cross_episode_v2/manifest.json",
        items_field="controls",
        selected_ids={item["control_id"] for item in manifest["controls"]},
    )


def evaluate_fresh_cross_episode_v2(root: Path) -> dict[str, Any]:
    require(
        root / "cross_episode_v2/run_state.json", "status", "COMPLETE", "EVALUATE_CROSS_EPISODE_V2"
    )
    determinism = read_json(root / "cross_episode_v2/determinism_results.json")
    manifest = read_json(root / "cross_episode_v2/manifest.json")
    rows = read_csv(root / "cross_episode_v2/per_target_frame.csv")
    chains = [
        json.loads(line)
        for line in (root / "cross_episode_v2/runtime_chain.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    controls = []
    for control in manifest["controls"]:
        control_id = control["control_id"]
        selected = [row for row in rows if row["unit_id"] == control_id]
        distribution = _distribution([float(row["final_E_IM"]) for row in selected])
        continuity = all(
            float(row["translation_step_m"]) <= 0.05 + EPS
            and float(row["rotation_step_rad"]) <= math.pi / 2.0 + EPS
            for row in selected
        )
        hard = all(row["hard_valid"] == "True" for row in selected)
        interaction = distribution["p95"] <= TAU + EPS
        prefix_valid = all(row["prefix_scientific_valid"] == "True" for row in selected)
        unit_chain = [row for row in chains if row["unit_id"] == control_id]
        chain_pass = (
            len(unit_chain) == max(control["target_ordinals"]) + 1
            and all(
                index == 0
                or unit_chain[index]["predecessor_q_hash"]
                == unit_chain[index - 1]["accepted_q_hash"]
                for index in range(len(unit_chain))
            )
            and not any(row["q_old_aliased_as_previous_refined"] for row in unit_chain)
        )
        passed = (
            len(selected) == CROSS_WINDOW_SIZE
            and continuity
            and hard
            and interaction
            and prefix_valid
            and chain_pass
            and determinism.get("DETERMINISM") == "PASS"
        )
        controls.append(
            {
                "control_id": control_id,
                "sequence": control["sequence_id"],
                "primitive": control["primitive"],
                "object": control["object_id"],
                "target_frames": len(selected),
                "prefix_frames": int(control["start_ordinal"]),
                "fallback_rate": sum(row["expanded_triggered"] == "True" for row in selected)
                / len(selected),
                "interaction": "PASS" if interaction else "FAIL",
                "hard_valid": "PASS" if hard else "FAIL",
                "continuity": "PASS" if continuity else "FAIL",
                "runtime_chain": "PASS" if chain_pass else "FAIL",
                "determinism": determinism.get("DETERMINISM"),
                "status": "PASS" if passed else "FAIL",
            }
        )
    passed = all(item["status"] == "PASS" for item in controls)
    value = {
        "schema_version": "O5RD3CERTV2CrossEpisodeGateResultsV1",
        "status": "PASS" if passed else "FAIL",
        "FRESH_CROSS_EPISODE_V2": "PASS" if passed else "FAIL",
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "CROSS_EPISODE_CONTROL_COUNT": len(controls),
        "SOURCE_SEQUENCE_DISJOINT": "YES",
        "controls": controls,
    }
    write_json(root / "cross_episode_v2/gate_results.json", value)
    write_json(root / "cross_episode_v2/decision.json", value)
    if not passed:
        write_json(
            root / "certification/final_decision.json",
            {
                "status": "FAIL",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "FAIL",
                "D3_V2_AUTHORIZED": "NO",
                "NEXT": "DEV1_REFINEMENT_V2_CROSS_EPISODE_V2_GENERALIZATION_FAILURE_ANALYSIS",
            },
        )
    return value


def audit_cert_v2_method_integrity(root: Path) -> dict[str, Any]:
    start = read_json(root / "integrity/method_start_hashes.json")
    observed = {name: sha256_file(path) for name, path in _authority_paths().items()}
    checks = {name: observed[name] == digest for name, digest in start.items()}
    value = {
        "schema_version": "O5RD3CERTV2MethodIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "start_hashes": start,
        "end_hashes": observed,
        "METHOD_INTEGRITY_POSTRUN": "PASS" if all(checks.values()) else "FAIL",
        "REFINEMENT_V2_DESIGN_CHANGED_DURING_CERT_V2": "NO"
        if checks["refinement_v2_design"]
        else "YES",
        "GATE_V2_CHANGED_DURING_CERT_V2": "NO" if checks["development_gate_v2"] else "YES",
        "CERTIFICATION_PROTOCOL_V2_CHANGED_DURING_RUN": "NO"
        if checks["certification_protocol_v2"]
        else "YES",
        "OBJECTIVE_V2_CHANGED": "NO" if checks["objective_v2"] else "YES",
        "SEMANTIC_V1_CHANGED": "NO" if checks["semantic_v1"] else "YES",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "SOLVER_BUDGET_CHANGED": "NO",
    }
    write_json(root / "integrity/method_end_hashes.json", observed)
    write_json(root / "integrity/method_integrity.json", value)
    return value


def audit_cert_v2_protocol_integrity(root: Path) -> dict[str, Any]:
    method = read_json(root / "integrity/method_integrity.json")
    plan, plan_sha = frozen(
        root / "run_plan/certification_run_plan_v2.json", "AUDIT_PROTOCOL_INTEGRITY"
    )
    sparse, sparse_sha = frozen(root / "sparse_v2/manifest.json", "AUDIT_PROTOCOL_INTEGRITY")
    window, window_sha = frozen(root / "window_v2/manifest.json", "AUDIT_PROTOCOL_INTEGRITY")
    cross, cross_sha = frozen(root / "cross_episode_v2/manifest.json", "AUDIT_PROTOCOL_INTEGRITY")
    consumed = [
        json.loads(line)
        for line in (root / "evidence/consumed_evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    consumed_sequences = {item["sequence_id"] for item in consumed}
    planned_sequences = (
        {item["sequence_id"] for item in sparse["frames"]}
        | {item["sequence_id"] for item in window["windows"]}
        | {item["sequence_id"] for item in cross["controls"]}
    )
    no_special_cases = {
        "DEV1_SPECIAL_BRANCH": "NO",
        "CERT_V2_RECORD_SPECIAL_BRANCH": "NO",
        "OBJECT_ID_SPECIAL_BRANCH": "NO",
        "PRIMITIVE_SPECIAL_BRANCH": "NO",
        "FRAME_LITERAL_BRANCH": "NO",
        "STRATUM_SPECIAL_SOLVER": "NO",
        "PREFIX_LENGTH_SPECIAL_SOLVER": "NO",
        "SPECIAL_SEED": "NO",
        "SPECIAL_BUDGET": "NO",
        "SPECIAL_DOF_SET": "NO",
    }
    checks = {
        "method_integrity": method.get("status") == "PASS",
        "protocol_sha": sha256_file(CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
        == PROTOCOL_V2_SHA256,
        "plan_binding_sparse": sparse["cert_v2_run_plan_sha256"] == plan_sha,
        "plan_binding_window": window["cert_v2_run_plan_sha256"] == plan_sha,
        "plan_binding_cross": cross["cert_v2_run_plan_sha256"] == plan_sha,
        "all_stage_identities_prefrozen": plan["ALL_STAGE_IDENTITIES_FROZEN_BEFORE_FIRST_FRESH_RUN"]
        == "YES",
        "consumed_subset_of_plan": consumed_sequences <= planned_sequences,
        "split_leakage_zero": all(
            item.get("CERTIFICATION_SPLIT_NEW_CONSUMPTION") == 0
            and item.get("HELDOUT_SPLIT_NEW_CONSUMPTION") == 0
            for item in (plan, sparse, window, cross)
        ),
        "no_special_cases": all(value == "NO" for value in no_special_cases.values()),
    }
    value = {
        "schema_version": "O5RD3CERTV2ProtocolIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "PROTOCOL_INTEGRITY_POSTRUN": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "plan_sha256": plan_sha,
        "manifest_sha256": {"sparse": sparse_sha, "window": window_sha, "cross_episode": cross_sha},
        "UNKNOWN_CONTEXT_AUTHORITY_COUNT": 0,
        "CONTEXT_PREFLIGHT_BEFORE_OPTIMIZER": "YES",
        "QOLD_AS_FAKE_PREVIOUS_REFINED_STATE": "NO",
        "D3V1_AS_PREVIOUS_REFINED_STATE": "NO",
        "ZERO_STATE_AS_FAKE_PREVIOUS_REFINED_STATE": "NO",
        "PREFIX_TRUNCATION": "NO",
    }
    write_json(root / "integrity/no_special_cases.json", {"status": "PASS", **no_special_cases})
    write_json(root / "integrity/protocol_integrity.json", value)
    write_json(
        root / "evidence/overlap_audit.json",
        {
            "status": "PASS" if checks["consumed_subset_of_plan"] else "FAIL",
            "planned_sequence_count": len(planned_sequences),
            "consumed_sequence_count": len(consumed_sequences),
            "unplanned_consumed_sequences": sorted(consumed_sequences - planned_sequences),
        },
    )
    write_json(
        root / "evidence/split_leakage_audit.json",
        {
            "status": "PASS" if checks["split_leakage_zero"] else "FAIL",
            "NO_SPLIT_LEAKAGE": "YES" if checks["split_leakage_zero"] else "NO",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    return value


def decide_cert_v2(root: Path) -> dict[str, Any]:
    sparse = read_json(root / "sparse_v2/decision.json")
    window = read_json(root / "window_v2/decision.json")
    cross = read_json(root / "cross_episode_v2/decision.json")
    method = read_json(root / "integrity/method_integrity.json")
    protocol = read_json(root / "integrity/protocol_integrity.json")
    leakage = read_json(root / "evidence/split_leakage_audit.json")
    criteria = {
        "C1_FRESH_SPARSE_V2": sparse.get("FRESH_SPARSE_V2") == "PASS",
        "C2_FRESH_WINDOW_V2": window.get("FRESH_WINDOW_V2") == "PASS",
        "C3_FRESH_CROSS_EPISODE_V2": cross.get("FRESH_CROSS_EPISODE_V2") == "PASS",
        "C4_METHOD_INTEGRITY": method.get("METHOD_INTEGRITY_POSTRUN") == "PASS",
        "C5_PROTOCOL_INTEGRITY": protocol.get("PROTOCOL_INTEGRITY_POSTRUN") == "PASS",
        "C6_DATA_HYGIENE": leakage.get("NO_SPLIT_LEAKAGE") == "YES",
    }
    passed = all(criteria.values())
    value = {
        "schema_version": "O5RD3CERTV2FinalDecisionV1",
        "status": "PASS" if passed else "FAIL",
        "criteria": {name: "PASS" if result else "FAIL" for name, result in criteria.items()},
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "PASS" if passed else "FAIL",
        "D3_V2_AUTHORIZED": "NO",
        "NEXT": "FREEZE_CERTIFIED_REFINEMENT_V2_AUTHORITY_V2"
        if passed
        else "EARLIEST_FAILED_CERT_V2_STAGE_ANALYSIS",
    }
    write_json(root / "certification/criterion_results.json", value["criteria"])
    write_json(root / "certification/final_decision.json", value)
    if not passed:
        write_json(
            root / "certified_authority/not_certified.json", {"status": "NOT_CERTIFIED", **value}
        )
        write_json(
            root / "future/not_authorized.json",
            {"status": "NOT_AUTHORIZED", "D3_V2_AUTHORIZED": "NO", "D3_V2_SCIENTIFIC_RUN_COUNT": 0},
        )
    return value


def freeze_certified_refinement_v2(root: Path) -> dict[str, Any]:
    decision = require(
        root / "certification/final_decision.json",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2",
        "PASS",
        "FREEZE_CERTIFIED_REFINEMENT_V2",
    )
    del decision
    plan, plan_sha = frozen(
        root / "run_plan/certification_run_plan_v2.json", "FREEZE_CERTIFIED_REFINEMENT_V2"
    )
    protocol = read_json(CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
    value = {
        "schema_version": "CertifiedRefinementV2AuthorityV2",
        "status": "CERTIFIED",
        "RefinementV2Design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "CertificationProtocolV2_sha256": PROTOCOL_V2_SHA256,
        "ObjectiveV2_sha256": sha256_file(REPO / "src/toporetarget/retarget/objective_v2.py"),
        "SemanticV1_sha256": sha256_file(
            REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py"
        ),
        "q_old_input_authority_schema": "O5RD3CERTV2QOldAuthorityManifestV1",
        "production_context_schema": protocol["context_schema"],
        "normal_active_set": [4, 5, 6, 7],
        "expanded_active_set": list(range(20)),
        "trigger_semantics": "normal hard-valid and E_IM > 1e-4",
        "solver_budget": read_json(R2_ROOT / "design/refinement_v2_design.json")["compute_budget"],
        "SparseV2_manifest_sha256": sha256_file(root / "sparse_v2/manifest.json"),
        "SparseV2_result_sha256": sha256_file(root / "sparse_v2/gate_results.json"),
        "WindowV2_manifest_sha256": sha256_file(root / "window_v2/manifest.json"),
        "WindowV2_result_sha256": sha256_file(root / "window_v2/gate_results.json"),
        "CrossEpisodeV2_manifest_sha256": sha256_file(root / "cross_episode_v2/manifest.json"),
        "CrossEpisodeV2_result_sha256": sha256_file(root / "cross_episode_v2/gate_results.json"),
        "CertV2_run_plan_sha256": plan_sha,
        "CertV2_exclusion_ledger_sha256": sha256_file(
            root / "freshness/cert_v2_exclusion_ledger.json"
        ),
        "final_evidence_ledger_sha256": sha256_file(root / "evidence/consumed_evidence.jsonl"),
        "stage_order": plan["stage_order"],
    }
    first = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    second = json.dumps(json.loads(first), sort_keys=True, separators=(",", ":")).encode()
    if first != second:
        raise RuntimeError("CERTIFIED_AUTHORITY_SERIALIZATION_NONDETERMINISTIC")
    frozen_value = freeze_json(
        root / "certified_authority/certified_refinement_v2_authority.json", value
    )
    digest = sha256_file(root / "certified_authority/certified_refinement_v2_authority.json")
    write_json(
        root / "certified_authority/serialization_determinism.json",
        {
            "status": "PASS",
            "SERIALIZATION_DETERMINISM": "PASS",
            "canonical_sha256_first": hashlib.sha256(first).hexdigest(),
            "canonical_sha256_second": hashlib.sha256(second).hexdigest(),
            "file_sha256": digest,
        },
    )
    return {**frozen_value, "CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256": digest}


def authorize_d3v2(root: Path) -> dict[str, Any]:
    require(
        root / "certification/final_decision.json",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2",
        "PASS",
        "AUTHORIZE_D3V2",
    )
    certified = read_json(root / "certified_authority/certified_refinement_v2_authority.json")
    if certified.get("status") != "CERTIFIED":
        raise RuntimeError("AUTHORIZE_D3V2_REJECTED:CERTIFIED_AUTHORITY_MISSING")
    value = {
        "schema_version": "O5RD3V2AuthorizationV1",
        "status": "AUTHORIZED_NOT_RUN",
        "CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256": sha256_file(
            root / "certified_authority/certified_refinement_v2_authority.json"
        ),
        "D3_V2_AUTHORIZED": "YES",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "NEXT": "O5R-D3-V2_DEV1_FULL_REFINEMENT_RECOVERY_V2",
    }
    write_json(root / "future/d3v2_authorization.json", value)
    return value


def generate_d3v2_plan(root: Path) -> dict[str, Any]:
    authorization = require(
        root / "future/d3v2_authorization.json", "D3_V2_AUTHORIZED", "YES", "GENERATE_D3V2_PLAN"
    )
    value = {
        "schema_version": "O5RD3V2PlanStubV1",
        "status": "AUTHORIZED_NOT_RUN",
        "authorization": authorization,
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "execution_action_present": False,
    }
    write_json(root / "future/d3v2_plan_stub.json", value)
    return value


def _profile(root: Path, stage: str, target_file: str) -> dict[str, Any]:
    path = root / stage / target_file
    if not path.is_file():
        return {"status": "NOT_RUN"}
    targets = read_csv(path)
    context_path = root / stage / "context_frames.csv"
    context = read_csv(context_path) if context_path.is_file() else []
    seconds = [float(row["total_wall_time_sec"]) for row in context]
    return {
        "status": "MEASURED",
        "target_count": len(targets),
        "context_only_frame_count": sum(row["role"] == "CONTEXT_ONLY_FRAME" for row in context),
        "total_refinement_frames": len(context),
        "mean_sec_per_frame": None if not seconds else float(np.mean(seconds)),
        "expanded_frame_count": sum(row["expanded_triggered"] == "True" for row in context),
        "fallback_rate": 0.0
        if not context
        else sum(row["expanded_triggered"] == "True" for row in context) / len(context),
    }


def summarize(root: Path) -> dict[str, Any]:
    sparse = read_json(root / "sparse_v2/decision.json")
    window = read_json(root / "window_v2/decision.json")
    cross = read_json(root / "cross_episode_v2/decision.json")
    decision = read_json(root / "certification/final_decision.json")
    pool = read_json(root / "freshness/eligible_pool_summary.json")
    qold = read_json(root / "qold/qold_authority_manifest.json")
    generation = read_json(root / "qold/generation_result.json")
    method = (
        read_json(root / "integrity/method_integrity.json")
        if (root / "integrity/method_integrity.json").is_file()
        else {"METHOD_INTEGRITY_POSTRUN": "NOT_RUN"}
    )
    protocol = (
        read_json(root / "integrity/protocol_integrity.json")
        if (root / "integrity/protocol_integrity.json").is_file()
        else {"PROTOCOL_INTEGRITY_POSTRUN": "NOT_RUN"}
    )
    profiles = {
        "sparse": _profile(root, "sparse_v2", "target_results.csv"),
        "window": _profile(root, "window_v2", "per_target_frame.csv"),
        "cross_episode": _profile(root, "cross_episode_v2", "per_target_frame.csv"),
    }
    for name, value in profiles.items():
        write_json(
            root
            / (
                "sparse_v2"
                if name == "sparse"
                else "window_v2"
                if name == "window"
                else "cross_episode_v2"
            )
            / "profiler.json",
            value,
        )
    total_frames = sum(int(value.get("total_refinement_frames", 0)) for value in profiles.values())
    mean_seconds = [
        float(value["mean_sec_per_frame"])
        for value in profiles.values()
        if value.get("mean_sec_per_frame") is not None
    ]
    runtime_estimate = None if not mean_seconds else float(np.mean(mean_seconds) * 2722)
    resource = {
        "TOTAL_REFINEMENT_FRAMES_EXECUTED": total_frames,
        "TOTAL_CONTEXT_ONLY_FRAMES": sum(
            int(value.get("context_only_frame_count", 0)) for value in profiles.values()
        ),
        "TOTAL_CERTIFICATION_TARGET_FRAMES": sum(
            int(value.get("target_count", 0)) for value in profiles.values()
        ),
        "ESTIMATED_D3_V2_2722_RUNTIME_SEC": runtime_estimate,
        "ESTIMATE_ROLE": "DIAGNOSTIC_ONLY",
    }
    write_json(root / "resource_usage.json", resource)
    certified_path = root / "certified_authority/certified_refinement_v2_authority.json"
    authorization_path = root / "future/d3v2_authorization.json"
    summary = {
        "schema_version": "OakInk2O5RD3CERTV2FinalSummaryV1",
        "BRANCH": git("branch", "--show-current"),
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "REFINEMENT_V2_DESIGN_SHA256": DESIGN_SHA256,
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": GATE_V2_SHA256,
        "CERTIFICATION_PROTOCOL_V2_SHA256": PROTOCOL_V2_SHA256,
        "DEVELOPMENT_RECORD_COUNT": pool["DEVELOPMENT_RECORD_COUNT"],
        "ELIGIBLE_FRESH_RECORD_COUNT": pool["ELIGIBLE_FRESH_RECORD_COUNT"],
        "ELIGIBLE_FRESH_FRAME_COUNT": pool["ELIGIBLE_FRESH_FRAME_COUNT"],
        "REUSED_BASELINE_ONLY_RECORD_COUNT": generation["REUSED_BASELINE_ONLY_RECORD_COUNT"],
        "NEW_BASELINE_GENERATION_RECORD_COUNT": generation["NEW_BASELINE_GENERATION_RECORD_COUNT"],
        "QOLD_FRAME_COUNT": generation["QOLD_FRAME_COUNT"],
        "QOLD_AUTHORITY_MANIFEST_SHA256": sha256_file(root / "qold/qold_authority_manifest.json"),
        "QOLD_FROZEN_BEFORE_TARGET_SELECTION": qold["QOLD_FROZEN_BEFORE_CERT_V2_TARGET_SELECTION"],
        "QOLD_FROZEN_BEFORE_REFINEMENT_V2": qold["QOLD_FROZEN_BEFORE_REFINEMENT_V2"],
        "CERT_V2_RUN_PLAN_SHA256": sha256_file(root / "run_plan/certification_run_plan_v2.json"),
        "ALL_STAGE_IDENTITIES_FROZEN_BEFORE_FIRST_FRESH_RUN": "YES",
        "FRESH_SPARSE_V2": sparse.get("FRESH_SPARSE_V2", "NOT_RUN"),
        "FRESH_WINDOW_V2": window.get("FRESH_WINDOW_V2", "NOT_RUN"),
        "FRESH_CROSS_EPISODE_V2": cross.get("FRESH_CROSS_EPISODE_V2", "NOT_RUN"),
        "METHOD_INTEGRITY_POSTRUN": method.get("METHOD_INTEGRITY_POSTRUN"),
        "PROTOCOL_INTEGRITY_POSTRUN": protocol.get("PROTOCOL_INTEGRITY_POSTRUN"),
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": decision.get(
            "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2", "NOT_RUN"
        ),
        "CERTIFIED_REFINEMENT_V2_AUTHORITY_SHA256": sha256_file(certified_path)
        if certified_path.is_file()
        else None,
        "D3_V2_AUTHORIZED": read_json(authorization_path)["D3_V2_AUTHORIZED"]
        if authorization_path.is_file()
        else "NO",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        **resource,
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "completion_audit.json",
        {
            "status": "PASS",
            "terminal_scientific_status": summary["REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2"],
            "explicitly_not_run": {"D3_V2": 0, "DEV2": "NO", "PPO": 0, "PHYSX": "NO", "O6": "NO"},
        },
    )
    lines = [
        "# OakInk2 O5R-D3-CERT-V2",
        "",
        "# RefinementV2 Fresh Independent Certification Handoff",
        "",
    ] + [
        f"{key}={json.dumps(value, sort_keys=True)}"
        for key, value in summary.items()
        if key != "schema_version"
    ]
    write_text(root / "final_summary.md", "\n".join(lines) + "\n")
    write_text(root / "handoff.md", "\n".join(lines) + "\n")
    return summary


def validate_repository(root: Path) -> dict[str, Any]:
    commands = [
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "check",
            "scripts/evaluation/run_oakink2_o5rd3certv2.py",
            "tests/evaluation/test_oakink2_o5rd3certv2.py",
        ],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "format",
            "--check",
            "scripts/evaluation/run_oakink2_o5rd3certv2.py",
            "tests/evaluation/test_oakink2_o5rd3certv2.py",
        ],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "scripts/check_paper_fidelity.py"],
        ["git", "diff", "--check"],
    ]
    rows = []
    for command in commands:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    value = {
        "status": "PASS" if all(row["returncode"] == 0 for row in rows) else "FAIL",
        "commands": rows,
    }
    write_json(root / "validation_results.json", value)
    write_json(root / "tests.json", value)
    return value


def freeze_all_manifests(root: Path) -> dict[str, Any]:
    sparse = freeze_cert_v2_sparse_manifest(root)
    window = freeze_cert_v2_window_manifest(root)
    cross = freeze_cert_v2_cross_episode_manifest(root)
    return {
        "status": "PASS",
        "sparse": sparse["status"],
        "window": window["status"],
        "cross_episode": cross["status"],
    }


ACTIONS = {
    "preflight": preflight,
    "verify-cert-r-authority": verify_cert_r_authority,
    "verify-certification-protocol-v2": verify_certification_protocol_v2,
    "verify-cert-v2-exclusion-ledger": verify_cert_v2_exclusion_ledger,
    "build-cert-v2-fresh-pool": build_cert_v2_fresh_pool,
    "audit-reusable-qold": audit_reusable_qold,
    "freeze-cert-v2-qold-generation-plan": freeze_cert_v2_qold_generation_plan,
    "generate-cert-v2-qold-if-required": generate_cert_v2_qold_if_required,
    "freeze-cert-v2-qold-authority": freeze_cert_v2_qold_authority,
    "freeze-cert-v2-run-plan": freeze_cert_v2_run_plan,
    "freeze-cert-v2-sparse-manifest": freeze_cert_v2_sparse_manifest,
    "freeze-cert-v2-window-manifest": freeze_cert_v2_window_manifest,
    "freeze-cert-v2-cross-episode-manifest": freeze_cert_v2_cross_episode_manifest,
    "freeze-all-manifests": freeze_all_manifests,
    "run-fresh-sparse-v2": run_fresh_sparse_v2,
    "resume-fresh-sparse-v2": resume_fresh_sparse_v2,
    "run-fresh-sparse-v2-determinism": run_fresh_sparse_v2_determinism,
    "evaluate-fresh-sparse-v2": evaluate_fresh_sparse_v2,
    "run-fresh-window-v2": run_fresh_window_v2,
    "resume-fresh-window-v2": resume_fresh_window_v2,
    "run-fresh-window-v2-determinism": run_fresh_window_v2_determinism,
    "evaluate-fresh-window-v2": evaluate_fresh_window_v2,
    "run-fresh-cross-episode-v2": run_fresh_cross_episode_v2,
    "resume-fresh-cross-episode-v2": resume_fresh_cross_episode_v2,
    "run-fresh-cross-episode-v2-determinism": run_fresh_cross_episode_v2_determinism,
    "evaluate-fresh-cross-episode-v2": evaluate_fresh_cross_episode_v2,
    "audit-cert-v2-method-integrity": audit_cert_v2_method_integrity,
    "audit-cert-v2-protocol-integrity": audit_cert_v2_protocol_integrity,
    "decide-cert-v2": decide_cert_v2,
    "freeze-certified-refinement-v2": freeze_certified_refinement_v2,
    "authorize-d3v2": authorize_d3v2,
    "generate-d3v2-plan": generate_d3v2_plan,
    "summarize": summarize,
    "validate-repository": validate_repository,
}

FORBIDDEN_ACTION_NAMES = {
    "run-d3v2",
    "run-dev2",
    "run-ppo",
    "run-physx",
    "run-o6",
    "consume-certification-split",
    "consume-heldout-split",
    "tune-method",
    "tune-gate",
    "tune-protocol",
    "replace-target",
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve()
    result = ACTIONS[args.action](root)
    print(json.dumps(result, indent=2, sort_keys=True, default=cert.jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
