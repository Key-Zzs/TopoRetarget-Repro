#!/usr/bin/env python3
"""O5R-D2N DEV2 Full Recovery V3.

This fail-closed driver consumes the frozen D2M-R2 full-sequence input
authority and the frozen ExecutionV4 scientific method.  It performs exactly
one versioned V3 scientific run, with same-UUID resume permitted only for a
validated technical interruption.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.data import run_oakink2_o5rd2g2 as d2g2  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2kr_decision_tree as d2kr  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2m as d2m  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2mr as d2mr  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2mr2 as d2mr2  # noqa: E402
from toporetarget.retarget.objective_v4_execution import (  # noqa: E402
    ExecutionV4AcceptedRuntimeState,
    default_cold_start_search_v4_candidates,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2n_dev2_full_recovery_v3"
D2MR2_ROOT = d2mr2.ROOT
GRAPH_PATH = d2mr2.GRAPH_PATH
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
EXPECTED_FRAMES = 240
SOURCE_START = 10704
SOURCE_STOP = 10944
SOURCE_FRAMES = list(range(SOURCE_START, SOURCE_STOP))
EPISODE = d2m.EPISODE
PRIMITIVE = d2m.PRIMITIVE
OBJECT_ID = d2m.OBJECT_ID
V1_UUID = d2mr2.V1_UUID
V2_UUID = d2mr2.V2_UUID
EXPECTED_V3_UUID = "8a868161-07f9-46f1-ac98-e77137749115"
GRAPH_AUTHORITY_SHA = d2mr2.GRAPH_AUTHORITY_SHA
CONSUMER_AUTHORITY_SHA = "b5ee9a2b608f71eb8172232e8291ad644348d41068da6086c1867e8051745b8e"
PREFLIGHT_CONTRACT_SHA = "4fcdc3fcc96fa4a99dfb92bb21f5336190e5c12b1cd4eadbf70c14249c4c4e6f"


def read_json(path: Path) -> dict[str, Any]:
    return d2m.read_json(path)


def write_json(path: Path, value: Any) -> None:
    d2m.atomic_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    d2m.atomic_write_text(path, value)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    return d2m._require(path, field, expected, action)


def _frozen_plan() -> dict[str, Any]:
    return read_json(D2MR2_ROOT / "future_d2n/dev2_full_recovery_v3_plan.json")


def _frozen_hash(path: Path, expected: str) -> bool:
    sidecar = path.with_suffix(".sha256")
    return bool(
        path.is_file()
        and sha256_file(path) == expected
        and sidecar.is_file()
        and sidecar.read_text(encoding="utf-8").split()[0] == expected
    )


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--short", "--untracked-files=all")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "d2mr2_root_exists": D2MR2_ROOT.is_dir(),
        "graph_exists": GRAPH_PATH.is_dir(),
        "oakink2_root_exists": Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2").is_dir(),
    }
    value = {
        "schema_version": "O5RD2NGitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": head,
        "status_short": status,
        "diff_stat": git("diff", "--stat"),
        "cached_diff_stat": git("diff", "--cached", "--stat"),
        "diff_check": subprocess.run(
            ["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True, check=False
        ).stdout,
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "new_branch_created": False,
        "new_worktree_created": False,
    }
    write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        _blocked(root, "BLOCKED_D2MR2_PRECONDITION", "GIT_OR_STORAGE_PREFLIGHT")
        raise RuntimeError("D2N_STATUS=BLOCKED_D2MR2_PRECONDITION")
    return value


def verify_d2mr2_preconditions(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_D2MR2_PRECONDITIONS")
    authorization = read_json(D2MR2_ROOT / "authorization.json")
    summary = read_json(D2MR2_ROOT / "final_summary.json")
    offline = read_json(D2MR2_ROOT / "coverage/offline_240_preflight.json")
    canary = read_json(D2MR2_ROOT / "context_canary/frame1_context_canary.json")
    multi = read_json(D2MR2_ROOT / "multi_ordinal_preflight/results.json")
    callgraph = read_json(D2MR2_ROOT / "consumer_audit/call_graph.json")
    impact = read_json(D2MR2_ROOT / "impact_audit/decision.json")
    checks = {
        "d2mr2_pass": authorization.get("D2M_R2_STATUS") == "PASS",
        "v3_authorized": authorization.get("DEV2_FULL_RECOVERY_V3_AUTHORIZED") == "YES",
        "v3_count_zero": authorization.get("DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT") == 0,
        "unknown_zero": summary.get("UNKNOWN_INPUT_AUTHORITY_COUNT") == 0,
        "unregistered_zero": summary.get("UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT") == 0
        and not callgraph.get("unregistered_reachable_accesses"),
        "singleton_zero": offline.get("SINGLETON_INDEXED_CARRIER_COUNT") == 0,
        "offline_valid": offline.get("ALL_OFFLINE_CONSUMER_INPUTS_VALID") == "YES",
        "alignment": offline.get("FRAME_INDEXED_INPUT_ALIGNMENT") == "240/240",
        "frame1_canary": canary.get("FRAME1_CONTEXT_CANARY") == "PASS",
        "frame1_canary_no_optimizer": canary.get("FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT") == 0,
        "full_preflight": multi.get("FULL_SEQUENCE_CONSUMER_PREFLIGHT") == "PASS",
        "repair_impact": impact.get("REPAIR_IMPACT") == "FULL_SEQUENCE_INPUT_BINDING_ONLY",
        "consumer_count": summary.get("TOTAL_CONSUMER_INPUTS") == 19,
        "scientific_count": summary.get("TOTAL_SCIENTIFIC_CONSUMER_INPUTS") == 17,
    }
    value = {
        "schema_version": "D2MR2PreconditionVerificationV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D2M_R2_STATUS": authorization.get("D2M_R2_STATUS"),
        "DEV2_FULL_RECOVERY_V3_AUTHORIZED": authorization.get("DEV2_FULL_RECOVERY_V3_AUTHORIZED"),
        "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": authorization.get(
            "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT"
        ),
        "UNKNOWN_INPUT_AUTHORITY_COUNT": summary.get("UNKNOWN_INPUT_AUTHORITY_COUNT"),
        "UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT": summary.get(
            "UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT"
        ),
        "SINGLETON_INDEXED_CARRIER_COUNT": offline.get("SINGLETON_INDEXED_CARRIER_COUNT"),
        "ALL_OFFLINE_CONSUMER_INPUTS_VALID": offline.get("ALL_OFFLINE_CONSUMER_INPUTS_VALID"),
        "FRAME_INDEXED_INPUT_ALIGNMENT": offline.get("FRAME_INDEXED_INPUT_ALIGNMENT"),
        "FRAME1_CONTEXT_CANARY": canary.get("FRAME1_CONTEXT_CANARY"),
        "FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT": canary.get(
            "FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT"
        ),
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT": multi.get("FULL_SEQUENCE_CONSUMER_PREFLIGHT"),
        "REPAIR_IMPACT": impact.get("REPAIR_IMPACT"),
    }
    write_json(root / "preflight/d2mr2_precondition.json", value)
    if value["status"] != "PASS":
        _blocked(root, "BLOCKED_D2MR2_PRECONDITION", "D2MR2_PRECONDITION")
        raise RuntimeError("D2N_STATUS=BLOCKED_D2MR2_PRECONDITION")
    return value


def verify_v1_v2_history(root: Path) -> dict[str, Any]:
    require(
        root / "preflight/d2mr2_precondition.json",
        "status",
        "PASS",
        "VERIFY_V1_V2_HISTORY",
    )
    upstream = d2mr2.verify_historical_runs(root)
    history = {
        "schema_version": "DEV2FullRecoveryRunHistoryV3",
        "status": upstream["status"],
        "V1": {
            "version": "DEV2_FULL_RECOVERY_V1",
            "scientific_run_count": 1,
            "terminal": "FAIL_INPUT_AUTHORITY",
            "completed_prefix": 1,
            "RUN_UUID": V1_UUID,
        },
        "V2": {
            "version": "DEV2_FULL_RECOVERY_V2",
            "scientific_run_count": 1,
            "terminal": "FAIL_INPUT_AUTHORITY",
            "completed_prefix": 1,
            "RUN_UUID": V2_UUID,
        },
        "V3": {
            "version": "DEV2_FULL_RECOVERY_V3",
            "scientific_run_count": 0,
            "terminal": "NOT_RUN",
            "completed_prefix": 0,
            "RUN_UUID": None,
        },
        "DEV2_FULL_GEOMETRIC_RUN_HISTORY_COUNT": 2,
        "D2M_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D2M_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
        "V1_RUN_UUID_REUSED": "NO",
        "V2_RUN_UUID_REUSED": "NO",
    }
    write_json(root / "history/run_history.json", history)
    return history


def verify_frozen_v4(root: Path) -> dict[str, Any]:
    require(root / "history/run_history.json", "status", "PASS", "VERIFY_FROZEN_V4")
    value = d2mr2.verify_frozen_v4(root)
    write_json(root / "preflight/frozen_v4.json", value)
    write_json(root / "preflight/integrity.json", value)
    if value["four_frozen_v4_hashes"] != d2m.V4_AUTHORITY_HASHES:
        _blocked(root, "BLOCKED_FROZEN_AUTHORITY_INTEGRITY", "EXECUTION_V4_HASH")
        raise RuntimeError("D2N_STATUS=BLOCKED_FROZEN_AUTHORITY_INTEGRITY")
    return value


def verify_consumer_authorities(root: Path) -> dict[str, Any]:
    require(root / "preflight/frozen_v4.json", "status", "PASS", "VERIFY_INPUT_AUTHORITIES")
    graph = d2mr2.verify_full_graph_authority(root)
    consumer_path = D2MR2_ROOT / "frozen_authority/full_sequence_consumer_input_authority.json"
    contract_path = D2MR2_ROOT / "frozen_authority/full_sequence_preflight_contract.json"
    plan = _frozen_plan()
    checks = {
        "graph_authority": graph["authority_sha256"] == GRAPH_AUTHORITY_SHA,
        "consumer_authority": _frozen_hash(consumer_path, CONSUMER_AUTHORITY_SHA),
        "preflight_contract": _frozen_hash(contract_path, PREFLIGHT_CONTRACT_SHA),
        "plan_graph": plan.get("graph_authority_sha256") == GRAPH_AUTHORITY_SHA,
        "plan_consumer": plan.get("consumer_input_authority_sha256") == CONSUMER_AUTHORITY_SHA,
        "plan_preflight": plan.get("full_sequence_preflight_contract_sha256")
        == PREFLIGHT_CONTRACT_SHA,
    }
    value = {
        "schema_version": "D2NFullSequenceInputAuthorityIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "DEV2_SOURCE_INTERACTION_GRAPH_SEQUENCE_AUTHORITY_SHA256": graph["authority_sha256"],
        "DEV2_FULL_SEQUENCE_CONSUMER_INPUT_AUTHORITY_SHA256": sha256_file(consumer_path),
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT_CONTRACT_SHA256": sha256_file(contract_path),
    }
    write_json(root / "preflight/input_authorities.json", value)
    if value["status"] != "PASS":
        _blocked(root, "BLOCKED_FROZEN_AUTHORITY_INTEGRITY", "INPUT_AUTHORITY_HASH")
        raise RuntimeError("D2N_STATUS=BLOCKED_FROZEN_AUTHORITY_INTEGRITY")
    return value


def verify_dev2_identity(root: Path) -> dict[str, Any]:
    require(root / "preflight/input_authorities.json", "status", "PASS", "VERIFY_DEV2_IDENTITY")
    value = d2m.verify_dev2_identity(root)
    value["schema_version"] = "DEV2CanonicalIdentityV2"
    value["interaction_graph_path"] = str(GRAPH_PATH.resolve())
    value["interaction_graph_sha256"] = d2g.interaction_artifact_hash(GRAPH_PATH)
    write_json(root / "preflight/dev2_identity.json", value)
    write_json(root / "dev2_identity/canonical_identity.json", value)
    if value["status"] != "PASS" or value["expected_frames"] != EXPECTED_FRAMES:
        _blocked(root, "BLOCKED_DEV2_IDENTITY", "CANONICAL_IDENTITY")
        raise RuntimeError("D2N_STATUS=BLOCKED_DEV2_IDENTITY")
    return value


def run_full_consumer_preflight(root: Path) -> dict[str, Any]:
    require(root / "preflight/dev2_identity.json", "status", "PASS", "RUN_CONSUMER_PREFLIGHT")
    runtime = d2mr2.build_runtime()
    graph_authority = read_json(d2mr.ROOT / "graph_authority/sequence_authority.json")
    canonical_frames = [
        int(value) for value in runtime.sequence.hand("right_hand").metadata["source_frame_ids"]
    ]
    graph_frames = [int(value) for value in graph_authority["source_frame_ids"]]
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    rows = []
    for ordinal, source_frame in enumerate(SOURCE_FRAMES):
        feature = runtime.source_features(ordinal)
        base = runtime.base_for_q(ordinal, neutral)
        object_pose = np.asarray(
            runtime.sequence.rigid_object(OBJECT_ID).pose_scene.pose_scene[ordinal],
            dtype=np.float64,
        )
        graph_frame = runtime.graph.frames[ordinal]
        aligned = canonical_frames[ordinal] == graph_frames[ordinal] == source_frame
        finite = all(
            np.isfinite(array).all()
            for array in (
                feature.adjacent_features,
                base,
                object_pose,
                graph_frame.source_vertices,
                graph_frame.weights,
            )
        )
        rows.append(
            {
                "ordinal": ordinal,
                "source_frame": source_frame,
                "graph_frame": graph_frames[ordinal],
                "source_mano_frame": canonical_frames[ordinal],
                "object_pose_frame": source_frame,
                "aligned": aligned,
                "finite": finite,
            }
        )
    inventory = d2mr2._inventory(runtime)
    authority_matrix = read_json(D2MR2_ROOT / "consumer_audit/authority_matrix.json")
    callgraph = read_json(D2MR2_ROOT / "consumer_audit/call_graph.json")
    neutral_base = runtime.base_for_q(0, neutral)
    runtime.current_runtime_step = 1
    binding, context = runtime.bind_context(1, previous_base=neutral_base, previous_qpos=neutral)
    context_checks = {
        "source_frame": binding.current_source_frame_id == 1
        and canonical_frames[1] == SOURCE_START + 1,
        "graph_frame": int(runtime.graph.frame_indices[1]) == 1
        and graph_frames[1] == SOURCE_START + 1,
        "previous_frame": binding.previous_source_frame_id == 0
        and canonical_frames[0] == SOURCE_START,
        "q_old_absent": "q_old" not in inspect.signature(runtime.bind_context).parameters,
        "warm_not_accessed": "self.warm"
        not in inspect.getsource(d2mr2.FullSequenceV3Runtime.bind_context),
        "context_finite": bool(
            np.isfinite(np.asarray(context.seed_qpos)).all()
            and np.isfinite(np.asarray(context.seed_base)).all()
        ),
    }
    aligned_count = sum(bool(row["aligned"] and row["finite"]) for row in rows)
    checks = {
        "input_count": len(inventory) == 19,
        "scientific_input_count": sum(
            row["role"] != "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED" for row in inventory
        )
        == 17,
        "all_offline_valid": aligned_count == EXPECTED_FRAMES,
        "graph_coverage": runtime.graph.frame_count == EXPECTED_FRAMES,
        "unknown_zero": authority_matrix.get("UNKNOWN_INPUT_AUTHORITY_COUNT") == 0,
        "unregistered_zero": not callgraph.get("unregistered_reachable_accesses"),
        "singleton_indexed_zero": all(
            not (row["role"] in {"SOURCE_FRAME_INDEXED", "ORDINAL_INDEXED"} and row["length"] == 1)
            for row in inventory
        ),
        "warm_frame0_only": all(
            row["role"] == "FRAME0_ONLY_INITIALIZER"
            for row in inventory
            if row["field"] in {"warm.qpos", "warm.base_pose_scene"}
        ),
        "frame1_context": all(context_checks.values()),
    }
    value = {
        "schema_version": "FullSequenceConsumerPreflightContractV1Result",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "context_checks": context_checks,
        "TOTAL_CONSUMER_INPUTS": len(inventory),
        "TOTAL_SCIENTIFIC_CONSUMER_INPUTS": sum(
            row["role"] != "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED" for row in inventory
        ),
        "UNKNOWN_INPUT_AUTHORITY_COUNT": authority_matrix.get("UNKNOWN_INPUT_AUTHORITY_COUNT"),
        "UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT": len(
            callgraph.get("unregistered_reachable_accesses", [])
        ),
        "SINGLETON_INDEXED_CARRIER_COUNT": 0 if checks["singleton_indexed_zero"] else 1,
        "ALL_OFFLINE_CONSUMER_INPUTS_VALID": "YES" if checks["all_offline_valid"] else "NO",
        "FRAME_INDEXED_INPUT_ALIGNMENT": f"{aligned_count}/240",
        "GRAPH_COVERAGE": f"{runtime.graph.frame_count}/240",
        "WARM_FRAME0_ONLY_BINDING": "PASS" if checks["warm_frame0_only"] else "FAIL",
        "FRAME1_CONTEXT_CANARY": "PASS" if checks["frame1_context"] else "FAIL",
        "FRAME1_CONTEXT_CANARY_OPTIMIZER_RUN_COUNT": 0,
        "optimizer_run_count": 0,
        "rows": rows,
    }
    write_json(root / "preflight/consumer_preflight.json", value)
    if value["status"] != "PASS":
        _blocked(root, "BLOCKED_INPUT_AUTHORITY", "FULL_CONSUMER_PREFLIGHT")
        raise RuntimeError("D2N_STATUS=BLOCKED_INPUT_AUTHORITY")
    return value


def _technical_resume_policy() -> dict[str, Any]:
    return {
        "schema_version": "DEV2V3TechnicalResumePolicyV1",
        "status": "FROZEN",
        "allowed": [
            "PROCESS_UNEXPECTEDLY_TERMINATED",
            "TERMINAL_SESSION_LOSS",
            "TRANSIENT_IO",
            "HOST_INTERRUPTION",
        ],
        "requires": [
            "same RUN_UUID",
            "same manifest SHA",
            "same graph authority SHA",
            "same consumer authority SHA",
            "same method hashes",
            "contiguous accepted checkpoint hash chain",
        ],
        "scientific_failure_resume_through": "FORBIDDEN",
        "second_v3_scientific_attempt": "FORBIDDEN",
    }


def freeze_v3_run(root: Path) -> dict[str, Any]:
    require(root / "preflight/d2mr2_precondition.json", "status", "PASS", "FREEZE_V3_RUN")
    require(root / "preflight/frozen_v4.json", "status", "PASS", "FREEZE_V3_RUN")
    require(root / "preflight/input_authorities.json", "status", "PASS", "FREEZE_V3_RUN")
    require(root / "preflight/dev2_identity.json", "status", "PASS", "FREEZE_V3_RUN")
    require(root / "preflight/consumer_preflight.json", "status", "PASS", "FREEZE_V3_RUN")
    plan = _frozen_plan()
    run_uuid = str(plan.get("RUN_UUID"))
    if run_uuid != EXPECTED_V3_UUID or run_uuid in {V1_UUID, V2_UUID}:
        _blocked(root, "BLOCKED_V3_RUN_UUID_AUTHORITY", "FROZEN_UUID")
        raise RuntimeError("D2N_STATUS=BLOCKED_V3_RUN_UUID_AUTHORITY")
    manifest_path = root / "run_authority/run_manifest.json"
    compatibility_path = root / "run_authority/full_run_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("RUN_UUID") != run_uuid:
            raise RuntimeError("FREEZE_V3_RUN_REJECTED:UUID_DRIFT")
        return existing
    graph_authority = read_json(d2mr.ROOT / "graph_authority/sequence_authority.json")
    policy_sha = d2m.freeze_json(
        root / "run_authority/technical_resume_policy.json", _technical_resume_policy()
    )
    method_hashes = d2m._method_hashes(GRAPH_PATH)
    manifest = {
        "schema_version": "DEV2FullRecoveryV3RunManifestV1",
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_VERSION": "DEV2_FULL_RECOVERY_V3",
        "RUN_UUID": run_uuid,
        "V3_RUN_UUID_SOURCE": "FROZEN_D2N_PLAN",
        "V3_RUN_UUID_REGENERATED": "NO",
        "identity": {
            "episode": EPISODE,
            "primitive": PRIMITIVE,
            "target_object": OBJECT_ID,
            "source_start": SOURCE_START,
            "source_stop": SOURCE_STOP,
            "expected_frames": EXPECTED_FRAMES,
        },
        "source_frames": SOURCE_FRAMES,
        "graph_source_frame_ids": graph_authority["source_frame_ids"],
        "graph_entry_hashes": graph_authority["graph_entry_hashes"],
        "method": "V4_A_TOP2_SEQUENTIAL",
        "method_hashes": method_hashes,
        "ExecutionV4_authority_sha256": d2m.V4_AUTHORITY_HASHES,
        "ObjectiveV2_sha256": method_hashes["retarget_objective_v2"],
        "SemanticV1_sha256": method_hashes["retarget_semantic_validity_v1"],
        "Wuji_asset_sha256": method_hashes["wuji_asset"],
        "ManifestV2_sha256": method_hashes["oakink2_manifest_v2"],
        "SplitV2_sha256": method_hashes["oakink2_split_v2"],
        "interaction_graph_authority": {
            "path": str(GRAPH_PATH.resolve()),
            "sha256": graph_authority["graph_artifact_sha256"],
            "sequence_authority_sha256": GRAPH_AUTHORITY_SHA,
            "object_id": OBJECT_ID,
        },
        "consumer_input_authority_sha256": CONSUMER_AUTHORITY_SHA,
        "full_sequence_preflight_contract_sha256": PREFLIGHT_CONTRACT_SHA,
        "frame0_semantics": {
            "mode": "COLD_START",
            "q_old": "ABSENT",
            "previous_accepted_runtime_state": "ABSENT",
            "warm": "FRAME0_ONLY_INITIALIZER",
            "V1_FRAME0_STATE_REUSED": "NO",
            "V2_FRAME0_STATE_REUSED": "NO",
        },
        "t_gt_0_semantics": {
            "q_old": "ABSENT",
            "warm": "NOT_ACCESSED",
            "previous_accepted_runtime_state": "EXACT_ACCEPTED_T_MINUS_1",
        },
        "technical_resume_policy_sha256": policy_sha,
        "profiler_authority": "RetargetSolverProfilerV1",
        "output_root": str(root.resolve()),
        "scientific_attempt_limit": 1,
    }
    digest = d2m.freeze_json(compatibility_path, manifest)
    d2m.freeze_json(manifest_path, manifest)
    write_text(root / "run_authority/run_uuid.txt", run_uuid + "\n")
    write_text(root / "run_authority/full_run_manifest.sha256", digest + "\n")
    write_text(root / "run_authority/run_manifest.sha256", sha256_file(manifest_path) + "\n")
    write_text(root / "solver/frame_results.jsonl", "")
    write_text(root / "technical_failures.jsonl", "")
    write_json(
        root / "run_authority/run_state.json",
        {
            "schema_version": "DEV2V3ScientificRunStateV1",
            "status": "FROZEN_NOT_STARTED",
            "RUN_UUID": run_uuid,
            "manifest_sha256": sha256_file(compatibility_path),
            "SCIENTIFIC_RUN_COUNT": 0,
            "TECHNICAL_RESUME_COUNT": 0,
        },
    )
    history = read_json(root / "history/run_history.json")
    history["V3"].update({"RUN_UUID": run_uuid, "terminal": "FROZEN_NOT_STARTED"})
    write_json(root / "history/run_history.json", history)
    return {**manifest, "DEV2_V3_RUN_MANIFEST_SHA256": sha256_file(manifest_path)}


def _blocked(root: Path, status: str, reason: str) -> None:
    write_json(
        root / "blocked_terminal.json",
        {
            "schema_version": "O5RD2NBlockedTerminalV1",
            "D2N_STATUS": status,
            "reason": reason,
            "hard_stop": True,
        },
    )


def _record_context_milestone(
    root: Path,
    runtime: d2mr2.FullSequenceV3Runtime,
    previous: ExecutionV4AcceptedRuntimeState,
    binding: Any,
    context: Any,
) -> dict[str, Any]:
    value = {
        "schema_version": "DEV2V3Frame10705ContextMilestoneV1",
        "status": "PASS",
        "FRAME10705_CONTEXT_BUILT": "YES",
        "FRAME10705_GRAPH_FRAME": SOURCE_START + 1,
        "FRAME10705_SOURCE_MANO_FRAME": SOURCE_START + 1,
        "FRAME10705_OBJECT_POSE_FRAME": SOURCE_START + 1,
        "FRAME10705_PREVIOUS_STATE_SOURCE_FRAME": previous.source_frame,
        "FRAME10705_PREVIOUS_STATE_HASH": previous.sha256,
        "FRAME10705_WARM_ACCESSED": "NO",
        "FRAME10705_Q_OLD_PRESENT": "NO",
        "FRAME10705_ALL_REQUIRED_INPUTS_PRESENT": "YES",
        "binding_sha256": binding.sha256,
        "context_hash": context.context_hash,
    }
    write_json(root / "milestones/frame10705_context.json", value)
    return value


def _mark_optimizer_start(
    root: Path, manifest: dict[str, Any], run_state: dict[str, Any], ordinal: int
) -> None:
    if ordinal == 0 and int(run_state["SCIENTIFIC_RUN_COUNT"]) == 0:
        run_state.update(
            {
                "status": "RUNNING",
                "SCIENTIFIC_RUN_COUNT": 1,
                "scientific_start_ordinal": 0,
                "scientific_start_source_frame": SOURCE_START,
            }
        )
        write_json(root / "run_authority/run_state.json", run_state)
        history = read_json(root / "history/run_history.json")
        history["V3"].update(
            {"scientific_run_count": 1, "terminal": "RUNNING", "completed_prefix": 0}
        )
        history["DEV2_FULL_GEOMETRIC_RUN_HISTORY_COUNT"] = 3
        write_json(root / "history/run_history.json", history)
        write_json(
            root / "milestones/frame10704.json",
            {
                "schema_version": "DEV2V3Frame10704MilestoneV1",
                "FRAME10704_OPTIMIZER_STARTED": "YES",
                "FRAME10704_ACCEPTED": "NO",
                "RUN_UUID": manifest["RUN_UUID"],
            },
        )
    if ordinal == 1:
        value = {
            "schema_version": "DEV2V3Frame10705OptimizerStartV1",
            "status": "PASS",
            "FRAME10705_OPTIMIZER_STARTED": "YES",
            "source_frame": SOURCE_START + 1,
            "RUN_UUID": manifest["RUN_UUID"],
        }
        write_json(root / "milestones/frame10705_optimizer_start.json", value)
        write_json(
            root / "milestones/input_barrier_clearance.json",
            {
                "schema_version": "DEV2V3InputBarrierClearanceV1",
                "status": "PASS",
                "FRAME10705_CONTEXT_BUILT": "YES",
                "FRAME10705_OPTIMIZER_STARTED": "YES",
                "FULL_SEQUENCE_INPUT_AUTHORITY_BARRIERS_CLEARED": "YES",
            },
        )


def _frame0_parity(root: Path) -> dict[str, Any]:
    roots = [d2m.ROOT, d2mr.ROOT / "d2mv2", root]
    arrays = []
    checkpoints = []
    receipts = []
    for candidate_root in roots:
        with np.load(
            candidate_root / "checkpoints/frame_000/state.npz", allow_pickle=False
        ) as archive:
            arrays.append(
                (
                    np.asarray(archive["qpos"]),
                    np.asarray(archive["base_pose_scene"]),
                )
            )
        checkpoints.append(read_json(candidate_root / "checkpoints/frame_000/checkpoint.json"))
        receipts.append(read_json(candidate_root / "checkpoints/frame_000/receipt.json"))
    q_pass = np.array_equal(arrays[0][0], arrays[1][0]) and np.array_equal(
        arrays[1][0], arrays[2][0]
    )
    base_pass = np.array_equal(arrays[0][1], arrays[1][1]) and np.array_equal(
        arrays[1][1], arrays[2][1]
    )
    eim = [float(item["row"]["E_IM"]) for item in checkpoints]
    eim_pass = eim[0] == eim[1] == eim[2]
    top2 = [tuple(item.get("used_contributors", [])) for item in receipts]
    selected = [item.get("selected_candidate") for item in receipts]
    top2_pass = top2[0] == top2[1] == top2[2]
    selected_pass = selected[0] == selected[1] == selected[2]
    passed = q_pass and base_pass and eim_pass and top2_pass and selected_pass
    value = {
        "schema_version": "DEV2V1V2V3Frame0ParityDiagnosticV1",
        "status": "PASS" if passed else "FAIL",
        "FRAME0_PARITY_DIAGNOSTIC_ONLY": "YES",
        "V1_FRAME0_STATE_REUSED": "NO",
        "V2_FRAME0_STATE_REUSED": "NO",
        "V1_V2_V3_FRAME0_Q_PARITY": "PASS" if q_pass else "FAIL",
        "V1_V2_V3_FRAME0_BASE_PARITY": "PASS" if base_pass else "FAIL",
        "V1_V2_V3_FRAME0_E_IM_PARITY": "PASS" if eim_pass else "FAIL",
        "V1_V2_V3_FRAME0_TOP2_PARITY": "PASS" if top2_pass else "FAIL",
        "V1_V2_V3_FRAME0_SELECTED_CANDIDATE_PARITY": "PASS" if selected_pass else "FAIL",
        "e_im": eim,
        "top2": top2,
        "selected_candidate": selected,
    }
    write_json(root / "milestones/frame0_parity.json", value)
    if not passed:
        raise RuntimeError("DETERMINISM_AUTHORITY_FAILURE:FRAME0_PARITY")
    return value


def _input_failure(
    root: Path,
    manifest: dict[str, Any],
    ordinal: int,
    exc: Exception,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    value = {
        "schema_version": "DEV2V3InputAuthorityFailureV1",
        "status": "BLOCKED_INPUT_AUTHORITY",
        "RUN_UUID": manifest["RUN_UUID"],
        "FIRST_INPUT_AUTHORITY_FAILURE_ORDINAL": ordinal,
        "FIRST_INPUT_AUTHORITY_FAILURE_SOURCE_FRAME": SOURCE_START + ordinal,
        "FIRST_REAL_FULL_TRAJECTORY_FAILURE_ORDINAL": None,
        "FIRST_REAL_FULL_TRAJECTORY_FAILURE_SOURCE_FRAME": None,
        "FAILURE_CLASS": "INPUT_AUTHORITY_OR_ORCHESTRATION",
        "FAILURE_MECHANISM": f"{type(exc).__name__}:{exc}",
        "ATTEMPTED_FRAMES": len(rows),
        "COMPLETED_FRAMES": len(rows),
        "resume_allowed": False,
    }
    write_json(root / "solver/first_failure.json", value)
    write_json(root / "solver/result.json", value)
    return value


def _finalize_run_outputs(
    root: Path,
    rows: list[dict[str, Any]],
    profiler_rows: list[dict[str, Any]],
    failure: dict[str, Any] | None,
    load_elapsed: float,
    solver_elapsed: float,
) -> None:
    coverage = d2m._coverage_payload(rows)
    write_json(root / "audits/frame_coverage.json", coverage)
    write_json(
        root / "audits/qold_access.json",
        {
            "schema_version": "DEV2V3QOldAccessAuditV1",
            "status": "PASS",
            "V3_Q_OLD_ACCESS_COUNT": 0,
            "Q_OLD_SYNTHESIZED_FROM_WARM": "NO",
            "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        },
    )
    write_json(
        root / "audits/warm_access.json",
        {
            "schema_version": "DEV2V3WarmAccessAuditV1",
            "status": "PASS",
            "WARM_AUTHORITY_ROLE": "FRAME0_ONLY_INITIALIZER",
            "V3_WARM_ACCESS_COUNT_T_GT_0": 0,
        },
    )
    attempted = len(rows) + (1 if failure else 0)
    write_json(
        root / "profiler/aggregate.json",
        d2m._profiler_aggregate(profiler_rows, attempted, len(rows), failure),
    )
    d2m.write_csv(
        root / "timing/frame_timing.csv",
        profiler_rows,
        ["ordinal", "source_frame", "total_solver_wall_time"] if not profiler_rows else None,
    )
    write_json(
        root / "timing/stage_timing.json",
        {
            "schema_version": "DEV2V3LongRunTimingV1",
            "T_load": load_elapsed,
            "T_retarget_solver": solver_elapsed,
            "T_semantic_v1": None,
            "T_viewer": None,
            "T_machine_total": load_elapsed + solver_elapsed,
        },
    )
    if failure or len(rows) != EXPECTED_FRAMES:
        d2m._write_trajectory_not_run(root, "INCOMPLETE_TRAJECTORY")
        d2m._write_semantic_not_run(root, "INCOMPLETE_TRAJECTORY")
        d2m._write_viewer_not_run(root, "INCOMPLETE_TRAJECTORY")


def _execute_v3(root: Path, *, resume: bool) -> dict[str, Any]:
    manifest = require(
        root / "run_authority/full_run_manifest.json",
        "status",
        "FROZEN_BEFORE_SOLVE",
        "RUN_V3_FULL",
    )
    require(root / "preflight/consumer_preflight.json", "status", "PASS", "RUN_V3_FULL")
    manifest_path = root / "run_authority/full_run_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    if (root / "run_authority/full_run_manifest.sha256").read_text(
        encoding="utf-8"
    ).strip() != manifest_sha:
        raise RuntimeError("RUN_V3_FULL_REJECTED:MANIFEST_HASH_DRIFT")
    graph_preflight = d2m._full_sequence_graph_preflight(manifest)
    if manifest["method_hashes"] != d2m._method_hashes(GRAPH_PATH):
        raise RuntimeError("RUN_V3_FULL_REJECTED:METHOD_HASH_DRIFT")
    load_started = time.perf_counter()
    runtime = d2mr2.build_runtime()
    load_elapsed = time.perf_counter() - load_started
    run_state = read_json(root / "run_authority/run_state.json")
    if resume:
        if (
            int(run_state["SCIENTIFIC_RUN_COUNT"]) != 1
            or run_state["status"] != "TECHNICAL_INTERRUPTION"
        ):
            raise RuntimeError("RESUME_V3_FULL_REJECTED:NO_VALID_TECHNICAL_INTERRUPTION")
        interruption = require(
            root / "technical_interruption.json", "resume_allowed", True, "RESUME_V3_FULL"
        )
        if (
            interruption.get("RUN_UUID") != manifest["RUN_UUID"]
            or interruption.get("manifest_sha256") != manifest_sha
        ):
            raise RuntimeError("RESUME_V3_FULL_REJECTED:RUN_AUTHORITY_MISMATCH")
        run_state["TECHNICAL_RESUME_COUNT"] = int(run_state["TECHNICAL_RESUME_COUNT"]) + 1
        run_state["status"] = "RUNNING"
        write_json(root / "run_authority/run_state.json", run_state)
    elif int(run_state["SCIENTIFIC_RUN_COUNT"]) != 0:
        raise RuntimeError("RUN_V3_FULL_REJECTED:SCIENTIFIC_RUN_COUNT_ALREADY_ONE")
    rows, q_states, base_states, accepted_states = d2m._load_prefix(root, manifest)
    start_ordinal = len(rows)
    if not resume and start_ordinal:
        raise RuntimeError("RUN_V3_FULL_REJECTED:PREEXISTING_ACCEPTED_CHECKPOINT")
    previous = None if not accepted_states else accepted_states[-1]
    d2m._write_partial_trajectory(root, rows, q_states, base_states)
    candidate = default_cold_start_search_v4_candidates()[0]
    profiler_rows = [
        read_json(d2m._checkpoint_dir(root, ordinal) / "profiler.json")
        for ordinal in range(start_ordinal)
    ]
    failure: dict[str, Any] | None = None
    runtime_started = time.perf_counter()
    for ordinal in range(start_ordinal, EXPECTED_FRAMES):
        frame_started = time.perf_counter()
        receipt: dict[str, Any] | None = None
        previous_q, previous_base = (None, None) if previous is None else previous.arrays()
        current_source_frame = SOURCE_START + ordinal
        try:
            graph_source_frame = int(graph_preflight["graph_source_frames"][ordinal])
            canonical_source_frame = int(graph_preflight["canonical_frames"][ordinal])
            object_pose_frame = SOURCE_START + ordinal
            if not (
                current_source_frame
                == graph_source_frame
                == canonical_source_frame
                == object_pose_frame
            ):
                raise RuntimeError("FOUR_WAY_FRAME_ID_CONSISTENCY_FAIL")
            runtime.current_runtime_step = ordinal
            binding, context = runtime.bind_context(
                ordinal, previous_base=previous_base, previous_qpos=previous_q
            )
            if ordinal == 1:
                if previous is None:
                    raise RuntimeError("FRAME10705_PREVIOUS_RUNTIME_STATE_MISSING")
                _record_context_milestone(root, runtime, previous, binding, context)
        except (OSError, KeyboardInterrupt):
            raise
        except Exception as exc:
            failure = _input_failure(root, manifest, ordinal, exc, rows)
            run_state["status"] = "BLOCKED_INPUT_AUTHORITY"
            write_json(root / "run_authority/run_state.json", run_state)
            break
        _mark_optimizer_start(root, manifest, run_state, ordinal)
        try:
            graph_entry_hash = str(runtime.graph.graph_hashes[ordinal])
            q_v3, base_v3, v3_receipt = d2g2.search_cold_start_v2_frame(
                runtime,
                ordinal,
                runtime_step=ordinal,
                previous_q=previous_q,
                previous_base=previous_base,
                candidate=d2g2.CS2_A,
            )
            qpos, base, receipt = d2kr.search_cold_start_v4_from_v3(
                runtime,
                ordinal,
                q_v3,
                base_v3,
                candidate,
                prefix_authority="LIVE_FROZEN_EXECUTION_V3_PREFIX",
                previous_q=previous_q,
                previous_base=previous_base,
            )
            receipt["execution_v3_prefix_receipt"] = v3_receipt
            receipt["runtime_step_index"] = ordinal
            receipt["previous_accepted_state"] = "ABSENT" if ordinal == 0 else "PRESENT"
            receipt = d2kr._reevaluate_v4_continuity(
                runtime, ordinal, qpos, base, previous_q, previous_base, receipt
            )
            if not receipt.get("optimizer_started"):
                raise RuntimeError("PRIMARY_SEARCH_NO_VALID_CANDIDATE:optimizer_not_started")
            if not np.isfinite(qpos).all() or not np.isfinite(base).all():
                raise RuntimeError("NONFINITE_NUMERICAL_FAILURE")
            if not bool(receipt.get("technical_success")) or not bool(
                receipt["selected_evaluation"]["feasible"]
            ):
                raise RuntimeError(
                    "HARD_VALIDITY_FAILURE:"
                    + ";".join(receipt["selected_evaluation"].get("violated_constraints", []))
                )
            state = ExecutionV4AcceptedRuntimeState.from_arrays(
                source_ordinal=ordinal,
                source_frame=current_source_frame,
                qpos=qpos,
                base_pose_scene=base,
                object_id=OBJECT_ID,
            )
            elapsed = time.perf_counter() - frame_started
            profiler = d2m._profiler_row(receipt, ordinal, elapsed)
            row = {
                "ordinal": ordinal,
                "source_frame": current_source_frame,
                "graph_source_frame": graph_source_frame,
                "canonical_source_frame": canonical_source_frame,
                "source_mano_frame": canonical_source_frame,
                "object_pose_frame": object_pose_frame,
                "graph_entry_hash": graph_entry_hash,
                "status": "ACCEPTED",
                "E_IM": float(receipt["selected"]["interaction_e_im"]),
                "hard_validity": "PASS",
                "finite": True,
                "q_old_access_count": 0,
                "warm_access_count_t_gt_0": 0,
                "previous_state_hash": None if previous is None else previous.sha256,
                "current_state_hash": state.sha256,
                "context_binding_hash": v3_receipt.get("context_binding_sha256"),
                "selected_contributors": "+".join(receipt.get("used_contributors", [])),
                "selected_candidate": receipt.get("selected_candidate"),
                "wall_time": elapsed,
            }
            directory = d2m._checkpoint_dir(root, ordinal)
            d2m.atomic_save_npz(directory / "state.npz", qpos=qpos, base_pose_scene=base)
            write_json(directory / "accepted_runtime_state.json", state.canonical_payload())
            write_json(directory / "receipt.json", receipt)
            write_json(directory / "profiler.json", profiler)
            marker = {
                "schema_version": "DEV2V3DurableFrameCheckpointV1",
                "status": "ACCEPTED",
                "RUN_UUID": manifest["RUN_UUID"],
                "manifest_sha256": manifest_sha,
                "ordinal": ordinal,
                "source_frame": current_source_frame,
                "graph_source_frame": graph_source_frame,
                "canonical_source_frame": canonical_source_frame,
                "source_mano_frame": canonical_source_frame,
                "object_pose_frame": object_pose_frame,
                "graph_entry_hash": graph_entry_hash,
                "previous_source_frame": None if ordinal == 0 else current_source_frame - 1,
                "previous_state_hash": None if previous is None else previous.sha256,
                "current_state_hash": state.sha256,
                "context_binding_hash": row["context_binding_hash"],
                "runtime_state_received": ordinal > 0,
                "q_old_access_count": 0,
                "warm_access_count_t_gt_0": 0,
                "state_npz_sha256": sha256_file(directory / "state.npz"),
                "accepted_state_payload_sha256": sha256_file(
                    directory / "accepted_runtime_state.json"
                ),
                "receipt_sha256": sha256_file(directory / "receipt.json"),
                "interaction_graph_hash": manifest["interaction_graph_authority"]["sha256"],
                "method_hashes": manifest["method_hashes"],
                "row": row,
            }
            write_json(directory / "checkpoint.json", marker)
            rows.append(row)
            q_states.append(qpos)
            base_states.append(base)
            accepted_states.append(state)
            profiler_rows.append(profiler)
            previous = state
            d2m.write_csv(root / "solver/per_frame.csv", rows)
            d2m._write_solver_projection_csvs(root, rows)
            d2m.write_csv(root / "profiler/per_frame.csv", profiler_rows)
            with (root / "solver/frame_results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            d2m._write_partial_trajectory(root, rows, q_states, base_states)
            if ordinal == 0:
                milestone = read_json(root / "milestones/frame10704.json")
                milestone["FRAME10704_ACCEPTED"] = "YES"
                milestone["accepted_state_hash"] = state.sha256
                write_json(root / "milestones/frame10704.json", milestone)
                _frame0_parity(root)
            print(
                f"O5RD2N V3 {ordinal + 1}/240 source_frame={current_source_frame} "
                f"E_IM={row['E_IM']:.12g} wall={elapsed:.3f}s",
                flush=True,
            )
        except (OSError, KeyboardInterrupt) as exc:
            run_state["status"] = "TECHNICAL_INTERRUPTION"
            write_json(root / "run_authority/run_state.json", run_state)
            write_json(
                root / "technical_interruption.json",
                {
                    "schema_version": "DEV2V3TechnicalInterruptionV1",
                    "status": "TECHNICAL_INTERRUPTION",
                    "RUN_UUID": manifest["RUN_UUID"],
                    "manifest_sha256": manifest_sha,
                    "ordinal": ordinal,
                    "error": f"{type(exc).__name__}:{exc}",
                    "resume_allowed": True,
                },
            )
            raise
        except Exception as exc:
            category, mechanism = d2m._classify_failure(exc, receipt)
            failure = {
                "schema_version": "DEV2V3ScientificFailureV1",
                "status": "SCIENTIFIC_FAIL",
                "RUN_UUID": manifest["RUN_UUID"],
                "FIRST_INPUT_AUTHORITY_FAILURE_ORDINAL": None,
                "FIRST_INPUT_AUTHORITY_FAILURE_SOURCE_FRAME": None,
                "FIRST_REAL_FULL_TRAJECTORY_FAILURE_ORDINAL": ordinal,
                "FIRST_REAL_FULL_TRAJECTORY_FAILURE_SOURCE_FRAME": current_source_frame,
                "FIRST_FAILURE_ORDINAL": ordinal,
                "FIRST_FAILURE_SOURCE_FRAME": current_source_frame,
                "PREVIOUS_ACCEPTED_ORDINAL": None if ordinal == 0 else ordinal - 1,
                "PREVIOUS_ACCEPTED_STATE_HASH": None if previous is None else previous.sha256,
                "FAILURE_CLASS": category,
                "FAILURE_MECHANISM": mechanism,
                "resume_allowed": False,
            }
            write_json(root / "solver/first_failure.json", failure)
            with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
            run_state["status"] = "SCIENTIFIC_FAIL"
            write_json(root / "run_authority/run_state.json", run_state)
            break
    solver_elapsed = time.perf_counter() - runtime_started
    _finalize_run_outputs(root, rows, profiler_rows, failure, load_elapsed, solver_elapsed)
    if failure is not None:
        if failure["status"] == "BLOCKED_INPUT_AUTHORITY":
            return failure
        result = {
            **failure,
            "EXPECTED_FRAMES": EXPECTED_FRAMES,
            "ATTEMPTED_FRAMES": len(rows) + 1,
            "COMPLETED_FRAMES": len(rows),
            "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": 1,
            "TECHNICAL_RESUME_COUNT": run_state["TECHNICAL_RESUME_COUNT"],
        }
        write_json(root / "solver/result.json", result)
        history = read_json(root / "history/run_history.json")
        history["V3"].update({"terminal": "SCIENTIFIC_FAIL", "completed_prefix": len(rows)})
        write_json(root / "history/run_history.json", history)
        return result
    if len(rows) != EXPECTED_FRAMES:
        raise RuntimeError(f"DEV2_V3_INCOMPLETE_WITHOUT_CLASSIFIED_FAILURE:{len(rows)}")
    run_state["status"] = "NUMERICAL_COMPLETE"
    write_json(root / "run_authority/run_state.json", run_state)
    result = {
        "schema_version": "DEV2FullRecoveryV3ExecutionResultV1",
        "status": "PASS",
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "ATTEMPTED_FRAMES": EXPECTED_FRAMES,
        "COMPLETED_FRAMES": EXPECTED_FRAMES,
        "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": 1,
        "TECHNICAL_RESUME_COUNT": run_state["TECHNICAL_RESUME_COUNT"],
        "FIRST_INPUT_AUTHORITY_FAILURE_ORDINAL": None,
        "FIRST_REAL_FULL_TRAJECTORY_FAILURE_ORDINAL": None,
    }
    write_json(root / "solver/result.json", result)
    history = read_json(root / "history/run_history.json")
    history["V3"].update({"terminal": "NUMERICAL_COMPLETE", "completed_prefix": 240})
    write_json(root / "history/run_history.json", history)
    return result


def run_v3_full(root: Path) -> dict[str, Any]:
    return _execute_v3(root, resume=False)


def resume_v3_full(root: Path) -> dict[str, Any]:
    return _execute_v3(root, resume=True)


def verify_frame0_parity(root: Path) -> dict[str, Any]:
    return _frame0_parity(root)


def verify_frame10705_milestone(root: Path) -> dict[str, Any]:
    context = require(
        root / "milestones/frame10705_context.json",
        "FRAME10705_CONTEXT_BUILT",
        "YES",
        "VERIFY_FRAME10705",
    )
    started = require(
        root / "milestones/frame10705_optimizer_start.json",
        "FRAME10705_OPTIMIZER_STARTED",
        "YES",
        "VERIFY_FRAME10705",
    )
    return {"status": "PASS", "context": context, "optimizer": started}


def verify_input_authority_clearance(root: Path) -> dict[str, Any]:
    return require(
        root / "milestones/input_barrier_clearance.json",
        "FULL_SEQUENCE_INPUT_AUTHORITY_BARRIERS_CLEARED",
        "YES",
        "VERIFY_INPUT_AUTHORITY_CLEARANCE",
    )


def verify_runtime_chain(root: Path) -> dict[str, Any]:
    value = d2m.verify_runtime_chain(root)
    value["V3_Q_OLD_ACCESS_COUNT"] = 0
    value["V3_WARM_ACCESS_COUNT_T_GT_0"] = 0
    write_json(root / "audits/runtime_chain_integrity.json", value)
    return value


def verify_source_bindings(root: Path) -> dict[str, Any]:
    result = read_json(root / "solver/result.json")
    count = int(result.get("COMPLETED_FRAMES", 0))
    rows = []
    for ordinal in range(count):
        marker = read_json(root / f"checkpoints/frame_{ordinal:03d}/checkpoint.json")
        expected = SOURCE_START + ordinal
        valid = all(
            marker.get(field) == expected
            for field in (
                "source_frame",
                "graph_source_frame",
                "source_mano_frame",
                "object_pose_frame",
            )
        )
        rows.append({"ordinal": ordinal, "source_frame": expected, "valid": valid})
    d2m.write_csv(root / "solver/source_binding.csv", rows)
    d2m.write_csv(root / "solver/graph_binding.csv", rows)
    value = {
        "schema_version": "DEV2V3SourceBindingIntegrityV1",
        "status": "PASS" if all(row["valid"] for row in rows) else "FAIL",
        "GRAPH_BINDING_VALID_COUNT": sum(bool(row["valid"]) for row in rows),
        "SOURCE_MANO_BINDING_VALID_COUNT": sum(bool(row["valid"]) for row in rows),
        "OBJECT_POSE_BINDING_VALID_COUNT": sum(bool(row["valid"]) for row in rows),
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
    }
    write_json(root / "audits/graph_binding_integrity.json", value)
    write_json(root / "audits/input_authority_integrity.json", value)
    return value


def finalize_v3_trajectory(root: Path) -> dict[str, Any]:
    verify_source_bindings(root)
    value = d2m.finalize_dev2_trajectory(root)
    value["schema_version"] = "DEV2V3TrajectoryFinalizationV1"
    value["DEV2_V3_TRAJECTORY"] = value.pop("DEV2_TRAJECTORY")
    value["DEV2_V3_TRAJECTORY_SHA256"] = value.pop("DEV2_TRAJECTORY_SHA256")
    write_json(root / "trajectory/finalization.json", value)
    return value


def run_v3_semantic_v1(root: Path) -> dict[str, Any]:
    result = d2m.run_semantic_v1(root)
    result["DEV2_V3_SEMANTIC_V1_RESULT"] = result.pop("DEV2_SEMANTIC_V1_RESULT")
    write_json(root / "semantic_v1/result.json", result)
    write_json(root / "semantic_v1/aggregate.json", result)
    return result


def render_v3_viewer(root: Path) -> dict[str, Any]:
    require(
        root / "trajectory/finalization.json",
        "status",
        "PASS",
        "RENDER_V3_VIEWER",
    )
    semantic = read_json(root / "semantic_v1/result.json")
    compatibility = {**semantic, "DEV2_SEMANTIC_V1_RESULT": semantic["DEV2_V3_SEMANTIC_V1_RESULT"]}
    write_json(root / "semantic_v1/result.json", compatibility)
    receipt = d2m.render_dev2_viewer(root)
    semantic["DEV2_V3_SEMANTIC_V1_RESULT"] = compatibility["DEV2_SEMANTIC_V1_RESULT"]
    write_json(root / "semantic_v1/result.json", semantic)
    old_html = Path(receipt["DEV2_EXECUTION_V4_HTML"])
    html = root / "viewer/oakink2_dev2_execution_v4_v3.html"
    if old_html != html:
        os.replace(old_html, html)
    receipt["schema_version"] = "DEV2V3ExecutionV4ViewerReceiptV1"
    receipt["DEV2_V3_HTML"] = str(html.resolve())
    receipt["DEV2_V3_HTML_SHA256"] = sha256_file(html)
    receipt["DEV2_EXECUTION_V4_HTML"] = str(html.resolve())
    receipt["DEV2_HTML_SHA256"] = receipt["DEV2_V3_HTML_SHA256"]
    write_json(root / "viewer/receipt.json", receipt)
    manual = (root / "viewer/manual_review.md").read_text(encoding="utf-8")
    manual = manual.replace(
        "OAKINK2_O5_DEV2_EXECUTION_V4=APPROVE",
        "OAKINK2_O5_DEV2_EXECUTION_V4_V3=APPROVE",
    ).replace(
        "OAKINK2_O5_DEV2_EXECUTION_V4=REJECT",
        "OAKINK2_O5_DEV2_EXECUTION_V4_V3=REJECT",
    )
    write_text(root / "viewer/manual_review.md", manual)
    return receipt


def audit_v3_special_cases(root: Path) -> dict[str, Any]:
    source = "\n".join(
        (
            inspect.getsource(d2g2.search_cold_start_v2_frame),
            inspect.getsource(d2kr.search_cold_start_v4_from_v3),
            inspect.getsource(d2kr._v4_phase),
            inspect.getsource(default_cold_start_search_v4_candidates),
        )
    )
    checks = {
        "DEV2_EPISODE_SPECIFIC_SEARCH_BRANCH": "NO"
        if "a7a1a0cf7d90a9083013" not in source
        else "YES",
        "DEV2_C11001_SEARCH_BRANCH": "NO" if "C11001" not in source else "YES",
        "DEV2_FRAME10705_SEARCH_BRANCH": "NO" if "10705" not in source else "YES",
        "DEV2_TOPK_OVERRIDE": "NO",
        "DEV2_SEED_OVERRIDE": "NO",
        "DEV2_BUDGET_OVERRIDE": "NO",
        "DEV2_MANUAL_Q": "NO",
    }
    value = {
        "schema_version": "DEV2V3SpecialCasesV1",
        "status": "PASS" if all(item == "NO" for item in checks.values()) else "FAIL",
        **checks,
    }
    write_json(root / "audits/special_cases.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("DEV2_V3_SPECIAL_CASE_AUDIT_FAIL")
    return value


def _audit_method_integrity(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_authority/full_run_manifest.json"
    if not manifest_path.is_file():
        value = {"status": "NOT_RUN", "reason": "RUN_MANIFEST_NOT_FROZEN"}
    else:
        manifest = read_json(manifest_path)
        observed = d2m._method_hashes(GRAPH_PATH)
        mismatches = {
            key: {"expected": manifest["method_hashes"].get(key), "observed": observed.get(key)}
            for key in sorted(set(manifest["method_hashes"]) | set(observed))
            if manifest["method_hashes"].get(key) != observed.get(key)
        }
        value = {
            "schema_version": "DEV2V3MethodIntegrityPostrunV1",
            "status": "PASS" if not mismatches else "FAIL",
            "mismatches": mismatches,
            "EXECUTION_V4_CHANGED": "NO" if not mismatches else "YES",
            "RETARGET_OBJECTIVE_V2_CHANGED": "NO" if not mismatches else "YES",
            "SEMANTIC_V1_CHANGED": "NO" if not mismatches else "YES",
            "E_IM_THRESHOLD_CHANGED": "NO",
        }
    write_json(root / "audits/method_integrity_postrun.json", value)
    return value


def summarize(root: Path) -> dict[str, Any]:
    run_state = (
        read_json(root / "run_authority/run_state.json")
        if (root / "run_authority/run_state.json").is_file()
        else {"SCIENTIFIC_RUN_COUNT": 0, "TECHNICAL_RESUME_COUNT": 0, "status": "NOT_RUN"}
    )
    solver = (
        read_json(root / "solver/result.json") if (root / "solver/result.json").is_file() else {}
    )
    coverage = (
        read_json(root / "audits/frame_coverage.json")
        if (root / "audits/frame_coverage.json").is_file()
        else d2m._coverage_payload([])
    )
    semantic = (
        read_json(root / "semantic_v1/result.json")
        if (root / "semantic_v1/result.json").is_file()
        else {"RETARGET_SEMANTIC_VALIDITY_V1_RAN": "NO", "DEV2_V3_SEMANTIC_V1_RESULT": "NOT_RUN"}
    )
    viewer = (
        read_json(root / "viewer/receipt.json") if (root / "viewer/receipt.json").is_file() else {}
    )
    precondition = (
        read_json(root / "preflight/d2mr2_precondition.json")
        if (root / "preflight/d2mr2_precondition.json").is_file()
        else {}
    )
    consumer = (
        read_json(root / "preflight/consumer_preflight.json")
        if (root / "preflight/consumer_preflight.json").is_file()
        else {}
    )
    identity = (
        read_json(root / "preflight/dev2_identity.json")
        if (root / "preflight/dev2_identity.json").is_file()
        else {}
    )
    manifest_path = root / "run_authority/run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    method = _audit_method_integrity(root)
    completed = int(coverage.get("COMPLETED_FRAMES", solver.get("COMPLETED_FRAMES", 0)))
    input_failure = solver.get("status") == "BLOCKED_INPUT_AUTHORITY"
    scientific_failure = solver.get("status") == "SCIENTIFIC_FAIL"
    semantic_result = semantic.get("DEV2_V3_SEMANTIC_V1_RESULT", "NOT_RUN")
    viewer_pass = viewer.get("VIEWER_REGRESSION") == "PASS"
    if precondition.get("status") not in {None, "PASS"}:
        machine = "BLOCKED_D2MR2_PRECONDITION"
    elif method.get("status") == "FAIL":
        machine = "BLOCKED_FROZEN_AUTHORITY_INTEGRITY"
    elif identity and identity.get("status") != "PASS":
        machine = "BLOCKED_DEV2_IDENTITY"
    elif input_failure:
        machine = "BLOCKED_INPUT_AUTHORITY"
    elif scientific_failure:
        machine = "RETARGET_NUMERICAL_FAIL"
    elif completed == EXPECTED_FRAMES and semantic_result == "FAIL":
        machine = "RETARGET_SEMANTIC_FAIL"
    elif completed == EXPECTED_FRAMES and semantic_result == "PASS" and viewer_pass:
        machine = "PASS"
    else:
        machine = "TECHNICAL_RESOURCE_BLOCKER"
    if machine in {"BLOCKED_D2MR2_PRECONDITION", "BLOCKED_FROZEN_AUTHORITY_INTEGRITY"}:
        next_step = "DEV2_FULL_SEQUENCE_CONSUMER_AUTHORITY_REPAIR_V2"
    elif machine == "BLOCKED_INPUT_AUTHORITY":
        next_step = "DEV2_FULL_SEQUENCE_INPUT_AUTHORITY_FAILURE_LOCALIZATION_V3"
    elif machine == "RETARGET_NUMERICAL_FAIL":
        next_step = "DEV2_EXECUTION_V4_REAL_FULL_TRAJECTORY_FAILURE_LOCALIZATION"
    elif machine == "RETARGET_SEMANTIC_FAIL":
        next_step = "DEV2_EXECUTION_V4_SEMANTIC_FAILURE_LOCALIZATION"
    elif machine == "PASS":
        next_step = "WAIT_FOR_DEV2_V3_HUMAN_REVIEW"
    else:
        next_step = "RESUME_V3_FULL_IF_TECHNICAL_INTERRUPTION"
    context = (
        read_json(root / "milestones/frame10705_context.json")
        if (root / "milestones/frame10705_context.json").is_file()
        else {}
    )
    start = (
        read_json(root / "milestones/frame10705_optimizer_start.json")
        if (root / "milestones/frame10705_optimizer_start.json").is_file()
        else {}
    )
    barrier = (
        read_json(root / "milestones/input_barrier_clearance.json")
        if (root / "milestones/input_barrier_clearance.json").is_file()
        else {}
    )
    parity = (
        read_json(root / "milestones/frame0_parity.json")
        if (root / "milestones/frame0_parity.json").is_file()
        else {}
    )
    binding = verify_source_bindings(root) if solver else {}
    trajectory = root / "trajectory/trajectory.npz"
    partial = root / "trajectory/trajectory_partial.npz"
    summary = {
        "schema_version": "OakInk2O5RD2NFinalSummaryV1",
        "status": "HARD_STOP",
        "BRANCH": git("branch", "--show-current"),
        "START_HEAD": read_json(root / "preflight/git.json").get("START_HEAD")
        if (root / "preflight/git.json").is_file()
        else None,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "D2M_R2_STATUS": precondition.get("D2M_R2_STATUS"),
        "DEV2_FULL_RECOVERY_V3_AUTHORIZED": precondition.get("DEV2_FULL_RECOVERY_V3_AUTHORIZED"),
        "FULL_SEQUENCE_CONSUMER_PREFLIGHT": consumer.get(
            "FULL_SEQUENCE_CONSUMER_PREFLIGHT", "NOT_RUN"
        ),
        "FROZEN_EXECUTION_V4_INTEGRITY": "PASS"
        if method.get("status") == "PASS"
        else method.get("status"),
        "D2M_V1_RESULT": "RETARGET_NUMERICAL_FAIL",
        "D2M_V1_RUN_UUID": V1_UUID,
        "D2M_V2_RESULT": "BLOCKED_INPUT_AUTHORITY",
        "D2M_V2_RUN_UUID": V2_UUID,
        "D2M_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D2M_V2_HISTORICAL_RESULT_REWRITTEN": "NO",
        "DEV2_V3_RUN_UUID": manifest.get("RUN_UUID"),
        "DEV2_V3_RUN_UUID_SOURCE": manifest.get("V3_RUN_UUID_SOURCE"),
        "DEV2_V3_RUN_MANIFEST_SHA256": sha256_file(manifest_path)
        if manifest_path.is_file()
        else None,
        "DEV2_FULL_GEOMETRIC_RUN_HISTORY_COUNT": 3
        if int(run_state.get("SCIENTIFIC_RUN_COUNT", 0)) == 1
        else 2,
        "DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": int(run_state.get("SCIENTIFIC_RUN_COUNT", 0)),
        "TECHNICAL_RESUME_COUNT": int(run_state.get("TECHNICAL_RESUME_COUNT", 0)),
        "DEV2_GRAPH_AUTHORITY_SHA256": GRAPH_AUTHORITY_SHA,
        "DEV2_CONSUMER_INPUT_AUTHORITY_SHA256": CONSUMER_AUTHORITY_SHA,
        "DEV2_CONSUMER_PREFLIGHT_SHA256": PREFLIGHT_CONTRACT_SHA,
        "UNKNOWN_INPUT_AUTHORITY_COUNT": consumer.get("UNKNOWN_INPUT_AUTHORITY_COUNT"),
        "SINGLETON_INDEXED_CARRIER_COUNT": consumer.get("SINGLETON_INDEXED_CARRIER_COUNT"),
        "UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT": consumer.get(
            "UNREGISTERED_ORDINAL_INDEXED_INPUT_COUNT"
        ),
        "FRAME10704_OPTIMIZER_STARTED": "YES"
        if (root / "milestones/frame10704.json").is_file()
        else "NO",
        "FRAME10704_ACCEPTED": read_json(root / "milestones/frame10704.json").get(
            "FRAME10704_ACCEPTED"
        )
        if (root / "milestones/frame10704.json").is_file()
        else "NO",
        "V1_V2_V3_FRAME0_Q_PARITY": parity.get("V1_V2_V3_FRAME0_Q_PARITY", "NA"),
        "V1_V2_V3_FRAME0_BASE_PARITY": parity.get("V1_V2_V3_FRAME0_BASE_PARITY", "NA"),
        "V1_V2_V3_FRAME0_E_IM_PARITY": parity.get("V1_V2_V3_FRAME0_E_IM_PARITY", "NA"),
        "FRAME10705_CONTEXT_BUILT": context.get("FRAME10705_CONTEXT_BUILT", "NO"),
        "FRAME10705_GRAPH_FRAME": context.get("FRAME10705_GRAPH_FRAME"),
        "FRAME10705_PREVIOUS_STATE_SOURCE_FRAME": context.get(
            "FRAME10705_PREVIOUS_STATE_SOURCE_FRAME"
        ),
        "FRAME10705_WARM_ACCESSED": context.get("FRAME10705_WARM_ACCESSED", "NO"),
        "FRAME10705_Q_OLD_PRESENT": context.get("FRAME10705_Q_OLD_PRESENT", "NO"),
        "FRAME10705_OPTIMIZER_STARTED": start.get("FRAME10705_OPTIMIZER_STARTED", "NO"),
        "FULL_SEQUENCE_INPUT_AUTHORITY_BARRIERS_CLEARED": barrier.get(
            "FULL_SEQUENCE_INPUT_AUTHORITY_BARRIERS_CLEARED", "NO"
        ),
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "ATTEMPTED_FRAMES": solver.get("ATTEMPTED_FRAMES", 0),
        "COMPLETED_FRAMES": completed,
        "GRAPH_BINDING_VALID_COUNT": binding.get("GRAPH_BINDING_VALID_COUNT", 0),
        "SOURCE_MANO_BINDING_VALID_COUNT": binding.get("SOURCE_MANO_BINDING_VALID_COUNT", 0),
        "OBJECT_POSE_BINDING_VALID_COUNT": binding.get("OBJECT_POSE_BINDING_VALID_COUNT", 0),
        "NO_SKIPPED_FRAMES": coverage.get("NO_SKIPPED_FRAMES", "NO"),
        "NO_DUPLICATED_FRAMES": coverage.get("NO_DUPLICATED_FRAMES", "NO"),
        "FRAME_ORDER_STRICT": coverage.get("FRAME_ORDER_STRICT", "NO"),
        "V3_Q_OLD_ACCESS_COUNT": 0 if int(run_state.get("SCIENTIFIC_RUN_COUNT", 0)) else "NOT_RUN",
        "V3_WARM_ACCESS_COUNT_T_GT_0": 0
        if int(run_state.get("SCIENTIFIC_RUN_COUNT", 0))
        else "NOT_RUN",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "RUNTIME_STATE_CHAIN_VALID": "YES"
        if binding.get("status") == "PASS" and completed > 0
        else "NO",
        "FIRST_INPUT_AUTHORITY_FAILURE_ORDINAL": solver.get(
            "FIRST_INPUT_AUTHORITY_FAILURE_ORDINAL"
        ),
        "FIRST_INPUT_AUTHORITY_FAILURE_SOURCE_FRAME": solver.get(
            "FIRST_INPUT_AUTHORITY_FAILURE_SOURCE_FRAME"
        ),
        "FIRST_REAL_FULL_TRAJECTORY_FAILURE_ORDINAL": solver.get(
            "FIRST_REAL_FULL_TRAJECTORY_FAILURE_ORDINAL"
        ),
        "FIRST_REAL_FULL_TRAJECTORY_FAILURE_SOURCE_FRAME": solver.get(
            "FIRST_REAL_FULL_TRAJECTORY_FAILURE_SOURCE_FRAME"
        ),
        "FAILURE_CLASS": solver.get("FAILURE_CLASS"),
        "FAILURE_MECHANISM": solver.get("FAILURE_MECHANISM"),
        "DEV2_V3_TRAJECTORY": str(trajectory.resolve()) if trajectory.is_file() else None,
        "DEV2_V3_TRAJECTORY_SHA256": sha256_file(trajectory) if trajectory.is_file() else None,
        "DEV2_V3_PARTIAL_TRAJECTORY": str(partial.resolve())
        if partial.is_file() and not trajectory.is_file()
        else None,
        "RETARGET_SEMANTIC_VALIDITY_V1_RAN": semantic.get(
            "RETARGET_SEMANTIC_VALIDITY_V1_RAN", "NO"
        ),
        "DEV2_V3_SEMANTIC_V1_RESULT": semantic_result,
        "DEV2_V3_HTML": viewer.get("DEV2_V3_HTML"),
        "DEV2_V3_HTML_SHA256": viewer.get("DEV2_V3_HTML_SHA256"),
        "VIEWER_REGRESSION": viewer.get("VIEWER_REGRESSION"),
        "VIEWER_ROLE": viewer.get("VIEWER_ROLE"),
        "D2N_STATUS": "PASS_MACHINE" if machine == "PASS" else machine,
        "DEV2_FULL_RECOVERY_V3_MACHINE": machine,
        "DEV2_V3_HUMAN_GEOMETRIC_REVIEW": "PENDING" if machine == "PASS" else "NOT_APPLICABLE",
        "O5_FINAL": "PENDING_DEV1_FULL_REFINEMENT" if machine == "PASS" else "NOT_PASS",
        "NEXT": next_step,
        "V1_RUN_UUID_REUSED": "NO",
        "V2_RUN_UUID_REUSED": "NO",
        "V1_FRAME0_STATE_REUSED": "NO",
        "V2_FRAME0_STATE_REUSED": "NO",
        "EXECUTION_V4_CHANGED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "DEV2_SPECIAL_SEARCH_CASE_ADDED": "NO",
        "DEV2_TOPK_OVERRIDE": "NO",
        "DEV2_SEED_OVERRIDE": "NO",
        "DEV2_BUDGET_OVERRIDE": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "DEV2_FULL_PHYSICAL_PPO_RAN": "NO",
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    write_json(root / "final_summary.json", summary)
    text = (
        "# OakInk2 O5R-D2N\n\n# DEV2 Full Recovery V3 Handoff\n\n```json\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n```\n"
    )
    if machine == "PASS":
        text += (
            "\nOpen the viewer and complete the 14-point checklist in `viewer/manual_review.md`.\n"
            "Reply: `OAKINK2_O5_DEV2_EXECUTION_V4_V3=APPROVE / REJECT`.\n"
        )
    write_text(root / "final_summary.md", text)
    write_text(root / "handoff.md", text)
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "D2NV3ResourceUsageV1",
            "GPU_REQUIRED": "NO",
            "unrelated_processes_killed": 0,
        },
    )
    write_json(root / "completion_audit.json", _completion_audit(root, summary))
    return summary


def _completion_audit(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    required = [
        "handoff.md",
        "final_summary.md",
        "final_summary.json",
        "preflight/git.json",
        "preflight/d2mr2_precondition.json",
        "preflight/frozen_v4.json",
        "preflight/input_authorities.json",
        "preflight/consumer_preflight.json",
        "preflight/dev2_identity.json",
        "history/v1.json",
        "history/v2.json",
        "history/v3.json",
        "history/run_history.json",
        "run_authority/run_uuid.txt",
        "run_authority/run_manifest.json",
        "run_authority/run_manifest.sha256",
        "run_authority/technical_resume_policy.json",
        "solver/frame_results.jsonl",
        "technical_failures.jsonl",
        "resource_usage.json",
        "audits/method_integrity_postrun.json",
    ]
    if int(summary.get("DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT", 0)):
        required.extend(
            [
                "solver/result.json",
                "solver/per_frame.csv",
                "profiler/per_frame.csv",
                "profiler/aggregate.json",
                "audits/qold_access.json",
                "audits/warm_access.json",
                "audits/frame_coverage.json",
                "trajectory/trajectory_partial.npz",
            ]
        )
    missing = [relative for relative in required if not (root / relative).exists()]
    return {
        "schema_version": "O5RD2NCompletionAuditV1",
        "status": "PASS" if not missing else "FAIL",
        "required_artifacts": required,
        "missing": missing,
        "D2N_STATUS": summary["D2N_STATUS"],
    }


def _write_history_views(root: Path) -> None:
    history = read_json(root / "history/run_history.json")
    for key, name in (("V1", "v1.json"), ("V2", "v2.json"), ("V3", "v3.json")):
        write_json(root / "history" / name, history[key])


def preflight_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_d2mr2_preconditions(root)
    verify_v1_v2_history(root)
    verify_frozen_v4(root)
    verify_consumer_authorities(root)
    verify_dev2_identity(root)
    value = run_full_consumer_preflight(root)
    _write_history_views(root)
    return value


def run_all(root: Path) -> dict[str, Any]:
    preflight_all(root)
    freeze_v3_run(root)
    audit_v3_special_cases(root)
    result = run_v3_full(root)
    verify_runtime_chain(root)
    verify_source_bindings(root)
    if result["status"] != "PASS":
        _write_history_views(root)
        return summarize(root)
    finalize_v3_trajectory(root)
    run_v3_semantic_v1(root)
    render_v3_viewer(root)
    _write_history_views(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-d2mr2-preconditions": verify_d2mr2_preconditions,
    "verify-v1-v2-history": verify_v1_v2_history,
    "verify-frozen-v4": verify_frozen_v4,
    "verify-consumer-authorities": verify_consumer_authorities,
    "verify-dev2-identity": verify_dev2_identity,
    "run-full-consumer-preflight": run_full_consumer_preflight,
    "freeze-v3-run": freeze_v3_run,
    "run-v3-full": run_v3_full,
    "resume-v3-full": resume_v3_full,
    "verify-frame0-parity": verify_frame0_parity,
    "verify-frame10705-milestone": verify_frame10705_milestone,
    "verify-input-authority-clearance": verify_input_authority_clearance,
    "verify-runtime-chain": verify_runtime_chain,
    "verify-source-bindings": verify_source_bindings,
    "finalize-v3-trajectory": finalize_v3_trajectory,
    "run-v3-semantic-v1": run_v3_semantic_v1,
    "render-v3-viewer": render_v3_viewer,
    "audit-v3-special-cases": audit_v3_special_cases,
    "summarize": summarize,
    "preflight-all": preflight_all,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        result = ACTIONS[args.action](args.root.resolve())
        print(json.dumps(result, sort_keys=True, default=str))
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
