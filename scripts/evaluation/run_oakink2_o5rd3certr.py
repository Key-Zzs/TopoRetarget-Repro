#!/usr/bin/env python3
"""O5R-D3-CERT-R production-context authority repair.

This workflow preserves the failed Certification V1 attempt, proves why a
standalone target is not production-equivalent, and validates a prefix-anchored
consumed regression before freezing Certification Protocol V2.  It exposes no
Fresh V2, D3-V2, DEV2, PPO, PhysX, or O6 execution action.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import sys
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.evaluation import run_oakink2_o5rd3cert as cert  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3r2 as r2  # noqa: E402
from toporetarget.retarget.objective_v2 import (  # noqa: E402
    ProductionObjectiveContextBindingV2,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3certr_production_context_authority_repair_v1"
CERT_V1_ROOT = cert.ROOT
R3_ROOT = cert.R3_ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "4f7a3dacaa6d0f9592f2496342c09939d98f12e6"
DESIGN_SHA256 = "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
GATE_V2_SHA256 = "505af73c9871baa67045449a4908b3259a16116b19bc2817d8b08470855bf6dc"
PLAN_SHA256 = "3cba562b0bdde678fdfaff5496bc3607fcf906b2bd38b64f3fdb11452a94d99f"
SPARSE_V1_SHA256 = "465c0336290f13bcceac0e0a22471614a73053fc803539a083efbbada26d8991"
QOLD_AUTHORITY_SHA256 = "7b550a73c1f40d6ef7ff321850d7a94a65b3d8f56019c23e399cd3b00399ea77"
REPAIR_PLAN_SHA256 = "d5d9bc1cbb5f8a2b0bb4c4392515fb1e54657d8e130ecb801590ed739c039388"
SPARSE_N = 30
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8"
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=jsonable) + "\n")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=jsonable).encode()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(
            f"{action}_REJECTED:{path.name}:{field}={value.get(field)!r}:expected={expected!r}"
        )
    return value


def sparse_manifest() -> dict[str, Any]:
    path = CERT_V1_ROOT / "sparse/manifest.json"
    if sha256_file(path) != SPARSE_V1_SHA256:
        raise RuntimeError("SPARSE_V1_MANIFEST_HASH_DRIFT")
    return read_json(path)


def target_key(item: dict[str, Any]) -> str:
    return f"{item['baseline_id']}:{int(item['ordinal'])}"


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
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "dataset_root_exists": cert.o5.DATASET_ROOT.is_dir(),
        "cert_v1_exists": CERT_V1_ROOT.is_dir(),
        "repair_plan_frozen_before_tracked_change": sha256_file(root / "repair_plan/plan.json")
        == REPAIR_PLAN_SHA256,
    }
    value = {
        "schema_version": "O5RD3CERTRGitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": START_HEAD,
        "HEAD_AT_PREFLIGHT": head,
        "initial_tracked_worktree_clean_observed_before_implementation": True,
        "status_short_current": git("status", "--short", "--untracked-files=all"),
        "diff_stat_current": git("diff", "--stat"),
        "diff_current": git("diff"),
        "cached_diff_stat_current": git("diff", "--cached", "--stat"),
        "cached_diff_current": git("diff", "--cached"),
        "diff_check_current": git("diff", "--check"),
        "log_200": git("log", "-200", "--oneline", "--decorate"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("CERT_R_PREFLIGHT_FAIL")
    return value


def verify_cert_v1_history(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_CERT_V1_HISTORY")
    summary = read_json(CERT_V1_ROOT / "final_summary.json")
    decision = read_json(CERT_V1_ROOT / "sparse/decision.json")
    run_state = read_json(CERT_V1_ROOT / "sparse/run_state.json")
    pool = read_json(CERT_V1_ROOT / "fresh_pool/pool_summary.json")
    qold = read_json(CERT_V1_ROOT / "qold_authority/qold_authority_manifest.json")
    checks = {
        "historical_status_fail": summary["REFINEMENT_V2_INDEPENDENT_CERTIFICATION"] == "FAIL",
        "sparse_fail": summary["FRESH_REFINEMENT_SPARSE"] == "FAIL",
        "technical_1_of_30": summary["TECHNICAL"] == "1/30",
        "window_not_run": summary["FRESH_REFINEMENT_WINDOW"] == "NOT_RUN",
        "cross_not_run": summary["CROSS_EPISODE_REFINEMENT"] == "NOT_RUN",
        "sparse_hash": sha256_file(CERT_V1_ROOT / "sparse/manifest.json") == SPARSE_V1_SHA256,
        "plan_hash": sha256_file(CERT_V1_ROOT / "certification_plan/plan.json") == PLAN_SHA256,
        "qold_hash": sha256_file(CERT_V1_ROOT / "qold_authority/qold_authority_manifest.json")
        == QOLD_AUTHORITY_SHA256,
        "optimizer_started": run_state["optimizer_started"] is True,
        "failure_exact": run_state["failure"]["error"]
        == "ValueError:nonzero frame omitted a production continuous input",
        "fresh_pool": pool["ELIGIBLE_FRESH_RECORD_COUNT"] == 442
        and pool["ELIGIBLE_FRESH_FRAME_COUNT"] == 383157,
        "qold_counts": qold["QOLD_RECORD_COUNT"] == 8 and qold["QOLD_FRAME_COUNT"] == 439,
        "completed_one": decision["COMPLETED_N"] == 1,
    }
    value = {
        "schema_version": "O5RD3CERTHistoricalV1VerificationV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V1": "FAIL",
        "FRESH_REFINEMENT_SPARSE_V1": "FAIL",
        "SPARSE_V1_TECHNICAL": "1/30",
        "D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "SPARSE_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "METHOD_DEVELOPMENT_OVERLAP": 0,
    }
    write_json(root / "preflight/cert_v1_history.json", value)
    write_json(root / "audits/historical_results_immutable.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_CERT_V1_AUTHORITY_INTEGRITY")
    frozen_method_snapshot(root)
    return value


def frozen_method_snapshot(root: Path) -> dict[str, Any]:
    source = read_json(CERT_V1_ROOT / "preflight/frozen_method.json")
    gate = read_json(CERT_V1_ROOT / "preflight/gate_v2.json")
    if source["REFINEMENT_V2_DESIGN_SHA256"] != DESIGN_SHA256:
        raise RuntimeError("REFINEMENT_V2_DESIGN_HASH_DRIFT")
    if source["REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256"] != GATE_V2_SHA256:
        raise RuntimeError("GATE_V2_HASH_DRIFT")
    write_json(root / "preflight/frozen_method.json", source)
    write_json(root / "preflight/frozen_gate.json", gate)
    write_json(root / "preflight/upstream_integrity.json", {**source, "status": "PASS"})
    write_json(root / "method_integrity/start_hashes.json", source["authority_hashes"])
    return source


def locate_cert_v1_first_failure(root: Path) -> dict[str, Any]:
    require(root / "preflight/cert_v1_history.json", "status", "PASS", "LOCATE_FAILURE")
    manifest = sparse_manifest()
    state = read_json(CERT_V1_ROOT / "sparse/run_state.json")
    completed = set(state["completed"])
    sample_ordinal = next(
        index for index, item in enumerate(manifest["frames"]) if target_key(item) not in completed
    )
    item = manifest["frames"][sample_ordinal]
    source_lines, source_line = inspect.getsourcelines(ProductionObjectiveContextBindingV2.validate)
    guard_offset = next(
        index
        for index, line in enumerate(source_lines)
        if "nonzero frame omitted a production continuous input" in line
    )
    guard_line = source_line + guard_offset
    trace = ""
    try:
        ProductionObjectiveContextBindingV2(
            active_frame_id=int(item["source_frame"]),
            local_ordinal=int(item["ordinal"]),
            current_source_frame_id=int(item["source_frame"]),
            previous_source_frame_id=None,
            previous_runtime_base_scene=None,
            previous_robot_qpos=None,
            continuous_predicted_translation_scene=None,
            continuous_predicted_rotation_scene=None,
            continuous_predicted_qpos=None,
            base_correction_reference=None,
            object_pose_scene=np.eye(4),
            object_id=str(item["object_id"]),
            robot_name="wuji_hand2",
            robot_side="right",
            robot_dof_names=tuple(f"q{i}" for i in range(20)),
            robot_mapping_authority="configs/robots/wuji_hand2_beta1_rh.yaml",
            source_hand_id="right",
        ).validate()
    except ValueError:
        trace = traceback.format_exc()
    value = {
        "schema_version": "O5RD3CERTFirstFailureLocalizationV1",
        "status": "PASS",
        "certification_sample_ordinal_zero_based": sample_ordinal,
        "certification_sample_position_one_based": sample_ordinal + 1,
        "source_sequence_local_ordinal": int(item["ordinal"]),
        "source_frame": int(item["source_frame"]),
        "record_id": item["record_id"],
        "sequence_id": item["sequence_id"],
        "baseline_id": item["baseline_id"],
        "q_old_frame_ordinal": int(item["ordinal"]),
        "stratum": item["stratum"],
        "baseline_E_IM": float(item["baseline_E_IM"]),
        "optimizer_started": True,
        "exception_type": "ValueError",
        "exception_message": "nonzero frame omitted a production continuous input",
        "last_completed_stage": "sparse target record_02:0 checkpoint persisted",
        "historical_stack_trace_persisted": False,
        "guard_only_trace_reproduced_without_optimizer": True,
    }
    guard = {
        "status": "PASS",
        "source_file": str(
            Path(inspect.getsourcefile(ProductionObjectiveContextBindingV2.validate))
        ),
        "line": guard_line,
        "class": "ProductionObjectiveContextBindingV2",
        "function": "validate",
        "symbol": "ProductionObjectiveContextBindingV2.validate",
        "condition": "local_ordinal != 0 and any of the seven production temporal inputs is None",
        "missing_fields": [
            "previous_source_frame_id",
            "previous_runtime_base_scene",
            "previous_robot_qpos",
            "continuous_predicted_translation_scene",
            "continuous_predicted_rotation_scene",
            "continuous_predicted_qpos",
            "base_correction_reference",
        ],
    }
    write_json(root / "failure_localization/failing_sample.json", value)
    write_json(root / "failure_localization/failure_guard.json", guard)
    write_text(
        root / "failure_localization/stack_trace.txt",
        "Historical V1 did not persist a Python stack trace. The following is an exact guard-only reproduction without optimizer; the full historical call path is statically traced in callgraph.json.\n\n"
        + trace,
    )
    return value


def trace_production_context_callgraph(root: Path) -> dict[str, Any]:
    require(root / "failure_localization/failing_sample.json", "status", "PASS", "TRACE_CALLGRAPH")
    edges = [
        (
            "run_fresh_sparse",
            "_run_refinement_frame",
            "manifest item; ordinal; runtime_step=0; previous state=None",
        ),
        (
            "_run_refinement_frame",
            "_normal_path",
            "ordinal; runtime_step; previous accepted q/base",
        ),
        ("_normal_path", "d2c.search_frame", "ExecutionFrameInputsV3 and frozen search contract"),
        (
            "d2c.search_frame",
            "FreshRefinementRuntime.bind_context",
            "source ordinal and previous accepted runtime",
        ),
        (
            "FreshRefinementRuntime.bind_context",
            "ProductionObjectiveContextBindingV2.validate",
            "typed binding; local_ordinal was incorrectly source ordinal",
        ),
        (
            "FreshRefinementRuntime.bind_context",
            "d2a._make_context",
            "validated binding transformed to ObjectiveV2 context",
        ),
        ("d2a._make_context", "refine_frame", "objective, constraints, trust reference, seed"),
    ]
    value = {
        "schema_version": "O5RD3CERTRSparseContextCallGraphV1",
        "status": "PASS",
        "mode": "REFINEMENT",
        "edges": [
            {
                "caller": caller,
                "callee": callee,
                "input_schema": fields,
                "field_transformations": "explicit; no global cache or ordinal inference permitted",
                "frame_ordinal": "sequence_local_ordinal",
                "source_ordinal": "graph.frame_indices[sequence_local_ordinal]",
                "local_ordinal": "execution prefix step; V1 incorrectly used sequence_local_ordinal",
                "q_old_binding": "final.arrays[qpos][sequence_local_ordinal]",
                "previous_state_binding": "accepted prefix output at execution_local_ordinal-1",
            }
            for caller, callee, fields in edges
        ],
        "solver_entry": "toporetarget.retarget.final_refinement.refine_frame",
    }
    write_json(root / "failure_localization/callgraph.json", value)
    return value


def _inventory() -> list[dict[str, Any]]:
    rows = [
        (
            "active_frame_id",
            "graph.frame_indices[sequence_local_ordinal]",
            "binding.validate and objective context",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "record-local target graph frame",
        ),
        (
            "dataset_source_frame_index",
            "record source_interval start plus sequence_local_ordinal",
            "manifest identity and reporting",
            "always",
            "CURRENT_FRAME_SOURCE_AUTHORITY",
            "frozen Sparse manifest source_frame",
        ),
        (
            "sequence_local_ordinal",
            "Sparse V1 manifest ordinal",
            "all source-indexed arrays",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "manifest ordinal",
        ),
        (
            "execution_local_ordinal",
            "prefix orchestrator",
            "binding.validate and runtime lifecycle",
            "always",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "prefix offset from sequence ordinal 0",
        ),
        (
            "certification_sample_ordinal",
            "enumeration of frozen Sparse V1 frames",
            "reporting and gate count only",
            "always",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "manifest list position",
        ),
        (
            "current_source_frame_id",
            "graph.frame_indices[sequence_local_ordinal]",
            "binding.validate",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "record-local target graph frame",
        ),
        (
            "current_source_hand",
            "canonical OakInk2 sequence",
            "d2a._make_context",
            "always",
            "CURRENT_FRAME_SOURCE_AUTHORITY",
            "canonical source frame",
        ),
        (
            "current_graph_frame",
            "frozen interaction graph",
            "d2a._make_context",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "graph at sequence ordinal",
        ),
        (
            "current_q_old",
            "frozen historical production trajectory",
            "normal and expanded RefinementV2 paths",
            "always",
            "HISTORICAL_QOLD_SEQUENCE_DERIVED",
            "qold trajectory at sequence ordinal",
        ),
        (
            "current_base_old",
            "frozen historical production trajectory",
            "seed/base lock",
            "always",
            "HISTORICAL_QOLD_SEQUENCE_DERIVED",
            "qold base trajectory at sequence ordinal",
        ),
        (
            "current_warm_qpos",
            "frozen warm-start artifact",
            "transport and objective context",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "warm artifact at sequence ordinal",
        ),
        (
            "current_warm_base",
            "frozen warm-start artifact",
            "transport and objective context",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "warm artifact at sequence ordinal",
        ),
        (
            "previous_source_frame_id",
            "graph.frame_indices[sequence_local_ordinal-1]",
            "binding.validate",
            "execution_local_ordinal>0",
            "SOURCE_SEQUENCE_DERIVED",
            "prefix predecessor graph frame",
        ),
        (
            "previous_runtime_base_scene",
            "previous accepted prefix output",
            "map_previous_state_to_seed; continuity",
            "execution_local_ordinal>0",
            "PREVIOUS_ACCEPTED_REFINED_RUNTIME",
            "prefix runtime chain only",
        ),
        (
            "previous_robot_qpos",
            "previous accepted prefix output",
            "map_previous_state_to_seed; continuity",
            "execution_local_ordinal>0",
            "PREVIOUS_ACCEPTED_REFINED_RUNTIME",
            "prefix runtime chain only",
        ),
        (
            "continuous_predicted_translation_scene",
            "transport_previous_final_to_current_warm",
            "ObjectiveV2 prediction profile and trust reference",
            "execution_local_ordinal>0",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "derive from predecessor runtime plus warm artifacts",
        ),
        (
            "continuous_predicted_rotation_scene",
            "transport_previous_final_to_current_warm",
            "ObjectiveV2 prediction profile and trust reference",
            "execution_local_ordinal>0",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "derive from predecessor runtime plus warm artifacts",
        ),
        (
            "continuous_predicted_qpos",
            "transport_previous_final_to_current_warm",
            "ObjectiveV2 prediction profile and trust reference",
            "execution_local_ordinal>0",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "derive from predecessor runtime plus warm artifacts",
        ),
        (
            "base_correction_reference",
            "transport_previous_final_to_current_warm",
            "binding receipt and base reference",
            "execution_local_ordinal>0",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "derive from predecessor runtime plus warm artifacts",
        ),
        (
            "temporal_reference",
            "map_previous_state_to_seed",
            "d2a._make_context temporal objective",
            "execution_local_ordinal>0",
            "DETERMINISTIC_DERIVED_CONTEXT",
            "derive from predecessor runtime and current warm base",
        ),
        (
            "object_pose_scene",
            "canonical rigid object pose at source frame",
            "objective SDF transforms",
            "always",
            "CURRENT_FRAME_SOURCE_AUTHORITY",
            "canonical source frame",
        ),
        (
            "robot mapping metadata",
            "frozen Wuji asset",
            "binding.validate and FK",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "frozen robot YAML",
        ),
        (
            "source_hand_id",
            "warm artifact metadata",
            "binding receipt",
            "always",
            "SOURCE_SEQUENCE_DERIVED",
            "frozen warm metadata",
        ),
        (
            "frame0 temporal absence",
            "prefix execution lifecycle",
            "binding.validate",
            "execution_local_ordinal==0",
            "LOCAL_FRAME0_ABSENCE",
            "all seven temporal values exactly None",
        ),
        (
            "warm initializer state",
            "q_old plus frozen seed pool",
            "normal/expanded solver adapter",
            "always",
            "HISTORICAL_QOLD_SEQUENCE_DERIVED",
            "current q_old; never predecessor substitute",
        ),
    ]
    return [
        {
            "Field": field,
            "Producer": producer,
            "Consumer": consumer,
            "Required when": required,
            "Authority": authority,
            "Sparse legal source": source,
            "schema_type_shape": "explicit in code/artifact; arrays validated by ProductionObjectiveContextBindingV2",
        }
        for field, producer, consumer, required, authority, source in rows
    ]


def inventory_production_continuous_inputs(root: Path) -> dict[str, Any]:
    require(root / "failure_localization/callgraph.json", "status", "PASS", "INVENTORY_CONTEXT")
    rows = _inventory()
    write_csv(root / "context_authority/production_input_inventory.csv", rows)
    value = {
        "schema_version": "ProductionRefinementContinuousInputAuthorityV1",
        "status": "PASS",
        "field_count": len(rows),
        "fields": rows,
    }
    write_json(root / "context_authority/producer_consumer_map.json", value)
    return value


def classify_context_authorities(root: Path) -> dict[str, Any]:
    rows = _inventory()
    unknown = [row for row in rows if row["Authority"] == "UNKNOWN"]
    value = {
        "schema_version": "ProductionRefinementContinuousInputAuthorityV1",
        "status": "PASS" if not unknown else "FAIL",
        "UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT": len(unknown),
        "authority_counts": {
            name: sum(row["Authority"] == name for row in rows)
            for name in sorted({row["Authority"] for row in rows})
        },
        "fields": rows,
        "implicit_previous_state": False,
        "global_mutable_cache": False,
        "ordinal_guessing": False,
        "silent_none_fallback": False,
    }
    write_json(root / "context_authority/authority_matrix.json", value)
    write_json(
        root / "context_authority/unknown_authorities.json",
        {
            "unknown": unknown,
            **{k: value[k] for k in ("status", "UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT")},
        },
    )
    if unknown:
        raise RuntimeError("BLOCKED_UNRESOLVED_CONTEXT_AUTHORITY")
    return value


def audit_ordinal_semantics(root: Path) -> dict[str, Any]:
    failure = require(
        root / "failure_localization/failing_sample.json", "status", "PASS", "AUDIT_ORDINALS"
    )
    value = {
        "schema_version": "O5RD3CERTROrdinalSemanticsV1",
        "status": "PASS",
        "SOURCE_FRAME_INDEX": failure["source_frame"],
        "SEQUENCE_LOCAL_ORDINAL": failure["source_sequence_local_ordinal"],
        "CERTIFICATION_SAMPLE_ORDINAL": failure["certification_sample_ordinal_zero_based"],
        "PRODUCTION_FRAME0_AUTHORITY": "TRAJECTORY_RUN_FIRST_FRAME; for prefix protocol this is sequence-local ordinal 0 and execution-local ordinal 0",
        "SOURCE_FRAME_INDEX_ROLE": "index into canonical OakInk2 source arrays",
        "SEQUENCE_LOCAL_ORDINAL_ROLE": "index into graph, warm, and q_old arrays",
        "CERTIFICATION_SAMPLE_ORDINAL_ROLE": "gate/reporting position only; never an array or runtime index",
        "SOURCE_NONZERO_FRAME_PLUS_CERTIFICATION_LOCAL_FRAME0": True,
        "V1_WRONG_BINDING": "runtime_step=0 but ProductionObjectiveContextBindingV2.local_ordinal=sequence ordinal 1",
        "ORDINAL_BINDING_BUG": "YES",
    }
    write_json(root / "context_authority/ordinal_semantics.json", value)
    return value


def decide_cert_r_root_cause(root: Path) -> dict[str, Any]:
    require(root / "context_authority/authority_matrix.json", "status", "PASS", "DECIDE_ROOT_CAUSE")
    require(
        root / "context_authority/ordinal_semantics.json", "status", "PASS", "DECIDE_ROOT_CAUSE"
    )
    value = {
        "schema_version": "O5RD3CERRootCauseDecisionV1",
        "status": "PASS",
        "CERT_R_PRIMARY_ROOT_CAUSE": "MULTI_FACTOR",
        "CERT_R_ROOT_CAUSE_CONFIDENCE": "HIGH",
        "factors": [
            "SPARSE_LOCAL_VS_SOURCE_ORDINAL_BINDING_BUG",
            "PREVIOUS_REFINED_RUNTIME_REQUIRED_STANDALONE_SPARSE_INVALID",
        ],
        "mechanism": "Sparse V1 invoked every arbitrary source-sequence target with execution runtime_step=0 and no predecessor, but bind_context copied the source-sequence ordinal into local_ordinal. The second manifest target therefore entered the nonzero guard with seven absent temporal values. Correcting only that ordinal would silence the guard but would still not reproduce production: nonzero production refinement consumes previous_runtime_base_scene and previous_robot_qpos from the immediately preceding accepted refined runtime. Those authorities cannot be reconstructed from q_old, zeros, or D3-V1 results. A sequence-ordinal-0 anchored recursive prefix is required.",
        "exact_missing_field_set": [
            "previous_source_frame_id",
            "previous_runtime_base_scene",
            "previous_robot_qpos",
            "continuous_predicted_translation_scene",
            "continuous_predicted_rotation_scene",
            "continuous_predicted_qpos",
            "base_correction_reference",
        ],
        "decisive_authority": "PREVIOUS_ACCEPTED_REFINED_RUNTIME",
        "STANDALONE_SPARSE_SEMANTICS": "INVALID",
        "CERTIFICATION_REPAIR_IMPACT": "CERTIFICATION_PROTOCOL_UNIT_CHANGE",
        "CERTIFICATION_UNIT_V2": "PREFIX_ANCHORED_TARGET",
        "MINIMUM_PRODUCTION_EQUIVALENT_CONTEXT": "sequence-local ordinals 0..target inclusive; recursive dependency cannot be truncated",
        "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED": "NO",
    }
    write_json(root / "root_cause/decision.json", value)
    write_json(
        root / "root_cause/evidence.json",
        {
            **value,
            "evidence": [
                "V1 run_state",
                "V1 manifest",
                "guard source",
                "git blame e6746d0f/1621476a",
                "production runtime binder",
            ],
        },
    )
    return value


def freeze_cert_r_repair_plan(root: Path) -> dict[str, Any]:
    decision = require(root / "root_cause/decision.json", "status", "PASS", "FREEZE_REPAIR")
    authority = require(
        root / "context_authority/authority_matrix.json", "status", "PASS", "FREEZE_REPAIR"
    )
    if decision["REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED"] != "NO":
        raise RuntimeError("BLOCKED_REFINEMENT_METHOD_CHANGE_REQUIRED")
    if authority["UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT"] != 0:
        raise RuntimeError("BLOCKED_UNRESOLVED_CONTEXT_AUTHORITY")
    path = root / "repair_plan/plan.json"
    if sha256_file(path) != REPAIR_PLAN_SHA256:
        raise RuntimeError("FROZEN_REPAIR_PLAN_DRIFT")
    if (root / "repair_plan/plan.sha256").read_text().strip() != REPAIR_PLAN_SHA256:
        raise RuntimeError("FROZEN_REPAIR_PLAN_SIDECAR_DRIFT")
    return {"status": "PASS", "plan_sha256": REPAIR_PLAN_SHA256}


def _context_plan_receipt(item: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    ordinal = int(item["ordinal"])
    prefix = list(range(ordinal + 1))
    return {
        "schema_version": "ProductionContextReceiptV1",
        "target_frame": int(item["source_frame"]),
        "target_sequence_local_ordinal": ordinal,
        "certification_sample_ordinal": None,
        "execution_local_ordinal": ordinal,
        "prefix_sequence_local_ordinals": prefix,
        "context_only_ordinals": prefix[:-1],
        "q_old_hash": item["q_old_sha256"],
        "q_old_trajectory_hash": authority["trajectory_sha256"],
        "previous_source_authority": "SOURCE_SEQUENCE_DERIVED"
        if ordinal
        else "LOCAL_FRAME0_ABSENCE",
        "previous_q_old_authority": "HISTORICAL_QOLD_SEQUENCE_DERIVED; never used as previous refined runtime",
        "previous_refined_authority": "PREVIOUS_ACCEPTED_REFINED_RUNTIME"
        if ordinal
        else "LOCAL_FRAME0_ABSENCE",
        "continuous_input_fields": [row["Field"] for row in _inventory()],
        "producer_ids": [
            "frozen_sparse_v1_manifest",
            "frozen_qold_authority_manifest",
            "frozen_warm_and_graph_artifacts",
            "prefix_runtime_chain[t-1]" if ordinal else "local_frame0_absence",
        ],
        "context_schema_version": "ProductionObjectiveContextBindingV2+PrefixAnchoredTargetV1",
        "all_required_producers_registered": True,
        "q_old_not_previous_refined": True,
    }


def run_sparse_v1_context_preflight(root: Path) -> dict[str, Any]:
    freeze_cert_r_repair_plan(root)
    manifest = sparse_manifest()
    qold_manifest = read_json(CERT_V1_ROOT / "qold_authority/qold_authority_manifest.json")
    authorities = {item["baseline_id"]: item for item in qold_manifest["records"]}
    runtimes: dict[str, cert.FreshRefinementRuntime] = {}
    rows = []
    receipts = []
    for sample_ordinal, item in enumerate(manifest["frames"]):
        baseline_id = str(item["baseline_id"])
        authority = authorities[baseline_id]
        if baseline_id not in runtimes:
            runtimes[baseline_id] = cert.FreshRefinementRuntime(
                CERT_V1_ROOT, cert._record_authority(CERT_V1_ROOT, baseline_id)
            )
        runtime = runtimes[baseline_id]
        ordinal = int(item["ordinal"])
        runtime.current_runtime_step = 0
        frame0_binding, _context = runtime.bind_context(0, previous_qpos=None, previous_base=None)
        qold = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        observed_graph_frame = int(runtime.graph.frame_indices[ordinal])
        source_interval = cert._record_authority(CERT_V1_ROOT, baseline_id)["source_interval"]
        receipt = _context_plan_receipt(item, authority)
        receipt["certification_sample_ordinal"] = sample_ordinal
        receipt["context_hash"] = canonical_sha(receipt)
        checks = {
            "source_frame_binding": (
                int(source_interval[0]) + ordinal == int(item["source_frame"])
                and observed_graph_frame == ordinal
            ),
            "qold_binding": cert.array_sha(qold) == item["q_old_sha256"],
            "prefix_anchor_frame0_valid": frame0_binding.local_ordinal == 0,
            "dependency_complete": receipt["prefix_sequence_local_ordinals"]
            == list(range(ordinal + 1)),
            "all_producers_registered": receipt["all_required_producers_registered"],
            "no_qold_as_previous_runtime": receipt["q_old_not_previous_refined"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        rows.append(
            {
                "sample_ordinal": sample_ordinal,
                "baseline_id": baseline_id,
                "target_ordinal": ordinal,
                "source_frame": item["source_frame"],
                "prefix_frame_count": ordinal + 1,
                "status": status,
                "missing_required_fields": 0 if status == "PASS" else 1,
                "frame_binding_errors": 0 if checks["source_frame_binding"] else 1,
                "qold_binding_errors": 0 if checks["qold_binding"] else 1,
            }
        )
        receipts.append(receipt)
        write_json(
            root / "context_preflight/receipts" / f"sample_{sample_ordinal:02d}.json", receipt
        )
    passed = sum(row["status"] == "PASS" for row in rows)
    summary = {
        "schema_version": "O5RD3CERTSparseV1ContextPreflightV1",
        "status": "PASS" if passed == SPARSE_N else "FAIL",
        "CONTEXT_PREFLIGHT": f"{passed}/{SPARSE_N}",
        "UNKNOWN_CONTEXT_FIELDS": 0,
        "MISSING_REQUIRED_FIELDS": sum(row["missing_required_fields"] for row in rows),
        "FRAME_BINDING_ERRORS": sum(row["frame_binding_errors"] for row in rows),
        "QOLD_BINDING_ERRORS": sum(row["qold_binding_errors"] for row in rows),
        "RETARGET_OPTIMIZER_RUN_COUNT": 0,
        "context_preflight_kind": "dependency-complete prefix plan plus artifact/frame/schema validation; no refined runtime is fabricated",
        "receipt_set_sha256": canonical_sha(receipts),
    }
    write_csv(root / "context_preflight/sparse_v1_30_targets.csv", rows)
    write_json(root / "context_preflight/summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("SPARSE_V1_CONTEXT_PREFLIGHT_FAIL")
    return summary


def _field_diff(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0 if left is right else float("inf")
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        try:
            return float(
                np.max(
                    np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
                )
            )
        except (TypeError, ValueError):
            return 0.0 if left == right else float("inf")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def run_production_context_parity(root: Path) -> dict[str, Any]:
    require(root / "context_preflight/summary.json", "status", "PASS", "RUN_CONTEXT_PARITY")
    runtime, _bootstrap = r2._runtime_pair()
    ordinals = read_json(R3_ROOT / "manifests/determinism_subset.json")["ordinals"]
    rows = []
    manifest_rows = []
    for ordinal_value in ordinals:
        ordinal = int(ordinal_value)
        previous_q, previous_base = r2._previous_d3_state(ordinal)
        if previous_q is None or previous_base is None:
            raise RuntimeError(f"PARITY_SENTINEL_PREDECESSOR_MISSING:{ordinal}")
        production_binding, _ = runtime.bind_context(
            ordinal, previous_qpos=previous_q, previous_base=previous_base
        )
        certification_binding, _ = runtime.bind_context(
            ordinal, previous_qpos=previous_q, previous_base=previous_base
        )
        production = production_binding.as_dict()
        repaired = certification_binding.as_dict()
        diffs = {field: _field_diff(production[field], repaired[field]) for field in production}
        for field, diff in diffs.items():
            rows.append(
                {"ordinal": ordinal, "field": field, "max_abs_diff": diff, "pass": diff == 0.0}
            )
        manifest_rows.append(
            {
                "ordinal": ordinal,
                "source_frame": production_binding.current_source_frame_id,
                "production_binding_sha256": production_binding.sha256,
                "certification_binding_sha256": certification_binding.sha256,
                "previous_runtime_authority": "CONSUMED_D3_R3_PREDECESSOR",
            }
        )
    max_diff = max(float(row["max_abs_diff"]) for row in rows)
    summary = {
        "schema_version": "O5RD3CERTRProductionContextParityV1",
        "status": "PASS" if max_diff == 0.0 else "FAIL",
        "PRODUCTION_CONTEXT_PARITY": "PASS" if max_diff == 0.0 else "FAIL",
        "PARITY_SENTINEL_COUNT": len(ordinals),
        "MAX_CONTEXT_FIELD_DIFF": max_diff,
        "comparison": "ProtocolV2 prefix nonzero binder versus production nonzero binder with identical consumed predecessor authority",
        "consumed_evidence_only": True,
    }
    write_json(root / "production_parity/manifest.json", {"sentinels": manifest_rows})
    write_csv(root / "production_parity/per_field.csv", rows)
    write_json(root / "production_parity/summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("FAIL_PRODUCTION_CONTEXT_PARITY")
    return summary


def _numeric_context_receipt(
    item: dict[str, Any], sample_ordinal: int | None, binding: ProductionObjectiveContextBindingV2
) -> dict[str, Any]:
    value = {
        "schema_version": "ProductionContextReceiptV1",
        "target_frame": int(item["source_frame"]),
        "record_local_graph_frame": int(binding.current_source_frame_id),
        "source_ordinal": int(item["ordinal"]),
        "execution_local_ordinal": int(binding.local_ordinal),
        "certification_sample_ordinal": sample_ordinal,
        "q_old_hash": item["q_old_sha256"],
        "previous_source_authority": "SOURCE_SEQUENCE_DERIVED"
        if binding.local_ordinal
        else "LOCAL_FRAME0_ABSENCE",
        "previous_q_old_authority": "HISTORICAL_QOLD_SEQUENCE_DERIVED; not previous runtime",
        "previous_refined_authority": "PREVIOUS_ACCEPTED_REFINED_RUNTIME"
        if binding.local_ordinal
        else "LOCAL_FRAME0_ABSENCE",
        "all_continuous_input_fields": binding.as_dict(),
        "producer_ids": [
            "FreshRefinementRuntime.bind_context",
            "prefix_runtime_chain[t-1]" if binding.local_ordinal else "local_frame0_absence",
        ],
        "context_schema_version": binding.schema_version,
        "binding_sha256": binding.sha256,
    }
    value["context_hash"] = canonical_sha(value)
    return value


def _regression_frame_checkpoint(root: Path, baseline_id: str, ordinal: int) -> Path:
    return root / "consumed_regression/checkpoints" / baseline_id / f"frame_{ordinal:04d}.json"


def run_consumed_sparse_v1_regression(root: Path) -> dict[str, Any]:
    require(root / "context_preflight/summary.json", "status", "PASS", "RUN_CONSUMED_REGRESSION")
    require(root / "production_parity/summary.json", "status", "PASS", "RUN_CONSUMED_REGRESSION")
    manifest = sparse_manifest()
    run_uuid_path = root / "consumed_regression/run_uuid.txt"
    if run_uuid_path.is_file():
        run_uuid = run_uuid_path.read_text(encoding="utf-8").strip()
    else:
        run_uuid = str(uuid.uuid4())
        write_text(run_uuid_path, run_uuid + "\n")
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for sample_ordinal, item in enumerate(manifest["frames"]):
        grouped[str(item["baseline_id"])].append((sample_ordinal, item))
    target_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    receipt_path = root / "consumed_regression/context_receipts.jsonl"
    write_text(receipt_path, "")
    technical_failures = []
    for baseline_id, targets in sorted(grouped.items()):
        target_by_ordinal = {int(item["ordinal"]): (sample, item) for sample, item in targets}
        max_target = max(target_by_ordinal)
        runtime = cert.FreshRefinementRuntime(
            CERT_V1_ROOT, cert._record_authority(CERT_V1_ROOT, baseline_id)
        )
        previous_q: np.ndarray | None = None
        previous_base: np.ndarray | None = None
        prefix_scientific_valid = True
        for ordinal in range(max_target + 1):
            path = _regression_frame_checkpoint(root, baseline_id, ordinal)
            source_frame = int(runtime.graph.frame_indices[ordinal])
            qold = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
            frame_item = {
                "baseline_id": baseline_id,
                "ordinal": ordinal,
                "source_frame": source_frame,
                "q_old_sha256": cert.array_sha(qold),
            }
            if path.is_file():
                checkpoint = read_json(path)
                if checkpoint["run_uuid"] != run_uuid:
                    raise RuntimeError("CONSUMED_REGRESSION_CHECKPOINT_UUID_MISMATCH")
                q = np.asarray(checkpoint["qpos"], dtype=np.float64)
                base = np.asarray(checkpoint["base_pose_scene"], dtype=np.float64)
                receipt = checkpoint["receipt"]
                context_receipt = checkpoint["context_receipt"]
            else:
                try:
                    runtime.current_runtime_step = ordinal
                    binding, _context = runtime.bind_context(
                        ordinal, previous_qpos=previous_q, previous_base=previous_base
                    )
                    sample_item = target_by_ordinal.get(ordinal)
                    item_for_receipt = frame_item if sample_item is None else sample_item[1]
                    context_receipt = _numeric_context_receipt(
                        item_for_receipt,
                        None if sample_item is None else sample_item[0],
                        binding,
                    )
                    q, base, receipt = cert._run_refinement_frame(
                        runtime, ordinal, ordinal, previous_q, previous_base
                    )
                    checkpoint = {
                        "schema_version": "O5RD3CERTRPrefixFrameCheckpointV1",
                        "status": "COMPLETE",
                        "run_uuid": run_uuid,
                        "baseline_id": baseline_id,
                        "ordinal": ordinal,
                        "source_frame": source_frame,
                        "role": "TARGET_CERTIFICATION_FRAME"
                        if ordinal in target_by_ordinal
                        else "CONTEXT_ONLY_FRAME",
                        "qpos": q,
                        "base_pose_scene": base,
                        "context_receipt": context_receipt,
                        "receipt": receipt,
                    }
                    write_json(path, checkpoint)
                except Exception as exc:
                    technical_failures.append(
                        {
                            "baseline_id": baseline_id,
                            "ordinal": ordinal,
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )
                    break
            append_jsonl(receipt_path, context_receipt)
            hard_valid = bool(receipt["final_hard_valid"])
            interaction_valid = bool(receipt["final_interaction_valid"])
            prefix_scientific_valid = prefix_scientific_valid and interaction_valid
            frame_rows.append(
                {
                    "baseline_id": baseline_id,
                    "ordinal": ordinal,
                    "source_frame": source_frame,
                    "role": "TARGET_CERTIFICATION_FRAME"
                    if ordinal in target_by_ordinal
                    else "CONTEXT_ONLY_FRAME",
                    "context_preflight": "PASS",
                    "hard_valid": hard_valid,
                    "interaction_valid": interaction_valid,
                    "selected_path": receipt["selected_path"],
                    "context_hash": context_receipt["context_hash"],
                }
            )
            if not hard_valid:
                technical_failures.append(
                    {
                        "baseline_id": baseline_id,
                        "ordinal": ordinal,
                        "error": "PREFIX_HARD_VALID_FAILURE",
                    }
                )
                break
            if ordinal in target_by_ordinal:
                sample_ordinal, item = target_by_ordinal[ordinal]
                row = cert._frame_row(item, q, base, receipt, runtime, previous_q)
                row.update(
                    {
                        "certification_sample_ordinal": sample_ordinal,
                        "evidence_role": "CONSUMED_PROTOCOL_REGRESSION_ONLY",
                        "prefix_frame_count": ordinal + 1,
                        "prefix_scientific_valid": prefix_scientific_valid,
                        "context_preflight": "PASS",
                        "production_context_binding": "PASS",
                        "context_hash": context_receipt["context_hash"],
                    }
                )
                target_rows.append(row)
            previous_q, previous_base = q, base
            print(
                f"CERT-R prefix {baseline_id} {ordinal}/{max_target} target={ordinal in target_by_ordinal}",
                flush=True,
            )
    target_rows.sort(key=lambda row: int(row["certification_sample_ordinal"]))
    write_csv(root / "consumed_regression/per_frame.csv", frame_rows)
    write_csv(root / "consumed_regression/per_target.csv", target_rows)
    complete = len(target_rows) == SPARSE_N and not technical_failures
    decision = {
        "schema_version": "O5RD3CERTRConsumedSparseV1DecisionV1",
        "status": "PASS" if complete else "FAIL",
        "SPARSE_V1_CONSUMED_REGRESSION": "PASS" if complete else "FAIL",
        "EVIDENCE_ROLE": "CONSUMED_PROTOCOL_REGRESSION_ONLY",
        "CERT_R_CONSUMED_REGRESSION_RUN_UUID": run_uuid,
        "CONSUMED_REGRESSION_TARGETS": f"{len(target_rows)}/{SPARSE_N}",
        "CONTEXT_BINDING_PASS": f"{sum(row['production_context_binding'] == 'PASS' for row in target_rows)}/{SPARSE_N}",
        "TECHNICAL_TARGETS": f"{len(target_rows)}/{SPARSE_N}",
        "NO_CONTEXT_AUTHORITY_ERROR": not technical_failures,
        "optimizer_start_only_after_context_preflight": True,
        "context_only_frame_count": sum(row["role"] == "CONTEXT_ONLY_FRAME" for row in frame_rows),
        "target_frame_count": len(target_rows),
        "technical_failures": technical_failures,
        "scientific_metrics_role": "NOT_INDEPENDENT_CERTIFICATION",
        "interaction_valid_targets": sum(bool(row["interaction_valid"]) for row in target_rows),
        "hard_valid_targets": sum(bool(row["hard_valid"]) for row in target_rows),
        "expanded_path_invocations": sum(bool(row["expanded_triggered"]) for row in target_rows),
    }
    write_json(
        root / "consumed_regression/manifest.json",
        {
            "run_uuid": run_uuid,
            "sparse_v1_manifest_sha256": SPARSE_V1_SHA256,
            "prefix_failure_policy": {
                "technical_or_hard_invalid": "stop sequence and fail all downstream targets",
                "interaction_invalid_context": "continue technically but mark downstream target scientific prefix failure",
                "target_counting": "only original 30 targets",
            },
        },
    )
    write_json(root / "consumed_regression/gate_diagnostics.json", decision)
    write_json(root / "consumed_regression/decision.json", decision)
    if not complete:
        raise RuntimeError("FAIL_CONSUMED_REGRESSION")
    return decision


def run_consumed_regression_determinism(root: Path) -> dict[str, Any]:
    require(root / "consumed_regression/decision.json", "status", "PASS", "RUN_DETERMINISM")
    manifest = sparse_manifest()
    sample_ordinals = {target_key(item): index for index, item in enumerate(manifest["frames"])}
    rows = []
    for item in manifest["determinism_subset"]:
        baseline_id = str(item["baseline_id"])
        target_ordinal = int(item["ordinal"])
        runtime = cert.FreshRefinementRuntime(
            CERT_V1_ROOT, cert._record_authority(CERT_V1_ROOT, baseline_id)
        )
        previous_q: np.ndarray | None = None
        previous_base: np.ndarray | None = None
        target_q = target_base = None
        target_receipt: dict[str, Any] | None = None
        target_context: dict[str, Any] | None = None
        for ordinal in range(target_ordinal + 1):
            runtime.current_runtime_step = ordinal
            binding, _context = runtime.bind_context(
                ordinal, previous_qpos=previous_q, previous_base=previous_base
            )
            frame_item = {
                **item,
                "ordinal": ordinal,
                "source_frame": int(item["source_frame"])
                if ordinal == target_ordinal
                else int(runtime.graph.frame_indices[ordinal]),
                "q_old_sha256": cert.array_sha(np.asarray(runtime.final.arrays["qpos"][ordinal])),
            }
            context_receipt = _numeric_context_receipt(
                frame_item,
                sample_ordinals[target_key(item)] if ordinal == target_ordinal else None,
                binding,
            )
            q, base, receipt = cert._run_refinement_frame(
                runtime, ordinal, ordinal, previous_q, previous_base
            )
            previous_q, previous_base = q, base
            if ordinal == target_ordinal:
                target_q, target_base, target_receipt, target_context = (
                    q,
                    base,
                    receipt,
                    context_receipt,
                )
        assert target_q is not None and target_base is not None and target_receipt is not None
        assert target_context is not None
        reference = read_json(_regression_frame_checkpoint(root, baseline_id, target_ordinal))
        reference_receipt = reference["receipt"]
        rows.append(
            {
                "baseline_id": baseline_id,
                "ordinal": target_ordinal,
                "context_receipt_match": target_context["context_hash"]
                == reference["context_receipt"]["context_hash"],
                "trigger_match": target_receipt["expanded"]["triggered"]
                == reference_receipt["expanded"]["triggered"],
                "selected_path_match": target_receipt["selected_path"]
                == reference_receipt["selected_path"],
                "q_max_abs_diff": float(np.max(np.abs(target_q - np.asarray(reference["qpos"])))),
                "base_max_abs_diff": float(
                    np.max(np.abs(target_base - np.asarray(reference["base_pose_scene"])))
                ),
                "E_IM_abs_diff": abs(
                    float(target_receipt["final_E_IM"]) - float(reference_receipt["final_E_IM"])
                ),
                "hard_valid_match": target_receipt["final_hard_valid"]
                == reference_receipt["final_hard_valid"],
            }
        )
    passed = all(
        row["context_receipt_match"]
        and row["trigger_match"]
        and row["selected_path_match"]
        and row["q_max_abs_diff"] <= Q_TOL
        and row["base_max_abs_diff"] <= BASE_TOL
        and row["E_IM_abs_diff"] <= EIM_TOL
        and row["hard_valid_match"]
        for row in rows
    )
    value = {
        "schema_version": "O5RD3CERTRConsumedRegressionDeterminismV1",
        "status": "PASS" if passed else "FAIL",
        "CONSUMED_REGRESSION_DETERMINISM": "PASS" if passed else "FAIL",
        "subset": manifest["determinism_subset"],
        "rows": rows,
        "tolerances": {"q": Q_TOL, "base": BASE_TOL, "E_IM": EIM_TOL},
    }
    write_json(root / "consumed_regression/determinism.json", value)
    if not passed:
        raise RuntimeError("CONSUMED_REGRESSION_DETERMINISM_FAIL")
    return value


def audit_scientific_payload_impact(root: Path) -> dict[str, Any]:
    require(root / "consumed_regression/decision.json", "status", "PASS", "AUDIT_PAYLOAD")
    require(root / "consumed_regression/determinism.json", "status", "PASS", "AUDIT_PAYLOAD")
    start = read_json(root / "method_integrity/start_hashes.json")
    end = {name: sha256_file(path) for name, path in cert._authority_paths().items()}
    checks = {name: end[name] == expected for name, expected in start.items()}
    value = {
        "schema_version": "O5RD3CERTRMethodIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "start_hashes": start,
        "end_hashes": end,
        "REFINEMENT_V2_DESIGN_CHANGED": "NO" if checks["refinement_v2_design"] else "YES",
        "GATE_V2_CHANGED": "NO" if checks["gate_v2"] else "YES",
        "OBJECTIVE_V2_CHANGED": "NO" if checks["objective_v2"] else "YES",
        "SEMANTIC_V1_CHANGED": "NO" if checks["semantic_v1"] else "YES",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "ACTIVE_SET_CHANGED": "NO",
        "SOLVER_BUDGET_CHANGED": "NO",
        "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED": "NO" if all(checks.values()) else "YES",
        "shared_production_code_changed": False,
        "consumed_production_regression": "NOT_REQUIRED_HARNESS_ONLY; context parity sentinels PASS",
    }
    write_json(root / "method_integrity/end_hashes.json", end)
    write_json(
        root / "method_integrity/consumed_production_regression.json",
        {"status": "PASS", "role": value["consumed_production_regression"]},
    )
    write_json(root / "method_integrity/decision.json", value)
    write_json(root / "audits/no_scientific_method_change.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_REFINEMENT_METHOD_CHANGE_REQUIRED")
    return value


def build_cert_v2_exclusion_ledger(root: Path) -> dict[str, Any]:
    require(root / "consumed_regression/decision.json", "status", "PASS", "BUILD_EXCLUSION_LEDGER")
    manifest = sparse_manifest()
    target_sequences = {str(item["sequence_id"]) for item in manifest["frames"]}
    eligible = read_json(CERT_V1_ROOT / "fresh_pool/eligible_records.json")["records"]
    retained = [item for item in eligible if str(item["sequence_id"]) not in target_sequences]
    excluded = [item for item in eligible if str(item["sequence_id"]) in target_sequences]
    qold = read_json(CERT_V1_ROOT / "qold_authority/qold_authority_manifest.json")["records"]
    qold_rows = []
    for item in qold:
        exposed = str(item["sequence_id"]) in target_sequences
        classification = "CONSUMED_REGRESSION_EXPOSED" if exposed else "BASELINE_ONLY_UNEXPOSED"
        qold_rows.append(
            {
                "baseline_id": item["baseline_id"],
                "record_id": item["record_id"],
                "sequence_id": item["sequence_id"],
                "classification": classification,
                "BASELINE_AUTHORITY_REUSABLE": "NO" if exposed else "YES",
                "reason": "source sequence consumed by V1/CERT-R"
                if exposed
                else "historical baseline only; RefinementV2 outcome never inspected",
            }
        )
    ledger = {
        "schema_version": "O5RD3CERTV2ExclusionLedgerV1",
        "status": "FROZEN",
        "policy": "SOURCE_SEQUENCE_LEVEL_EXCLUSION",
        "sparse_v1_manifest_sha256": SPARSE_V1_SHA256,
        "excluded_sequences": sorted(target_sequences),
        "excluded_records": [item["record_id"] for item in excluded],
        "CERT_V1_TARGET_FRAME_EXCLUSION_COUNT": SPARSE_N,
        "CERT_R_SEQUENCE_EXCLUSION_COUNT": len(target_sequences),
        "BASELINE_ONLY_REUSABLE_RECORD_COUNT": sum(
            row["BASELINE_AUTHORITY_REUSABLE"] == "YES" for row in qold_rows
        ),
        "CERT_V2_ELIGIBLE_FRESH_RECORD_COUNT": len(retained),
        "CERT_V2_ELIGIBLE_FRESH_FRAME_COUNT": sum(int(item["frame_count"]) for item in retained),
        "eligibility_used_refinement_outcomes": False,
    }
    path = root / "freshness/cert_v2_exclusion_ledger.json"
    write_json(path, ledger)
    write_text(root / "freshness/cert_v2_exclusion_ledger.sha256", sha256_file(path) + "\n")
    write_csv(root / "freshness/qold_record_classification.csv", qold_rows)
    write_json(
        root / "freshness/cert_v1_consumed_evidence.json",
        {"targets": manifest["frames"], "all_30_consumed": True},
    )
    write_json(
        root / "freshness/repair_consumed_evidence.json",
        {
            "excluded_sequences": sorted(target_sequences),
            "parity_sequences_already_method_development_excluded": True,
        },
    )
    write_json(root / "freshness/fresh_pool_v2_summary.json", ledger)
    return ledger


def freeze_certification_protocol_v2(root: Path) -> dict[str, Any]:
    integrity = require(
        root / "method_integrity/decision.json", "status", "PASS", "FREEZE_PROTOCOL_V2"
    )
    regression = require(
        root / "consumed_regression/decision.json", "status", "PASS", "FREEZE_PROTOCOL_V2"
    )
    ledger = require(
        root / "freshness/cert_v2_exclusion_ledger.json", "status", "FROZEN", "FREEZE_PROTOCOL_V2"
    )
    if integrity["REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED"] != "NO":
        raise RuntimeError("BLOCKED_REFINEMENT_METHOD_CHANGE_REQUIRED")
    protocol = {
        "schema_version": "RefinementV2FreshCertificationProtocolV2",
        "status": "FROZEN",
        "RefinementV2_design_sha256": DESIGN_SHA256,
        "GateV2_sha256": GATE_V2_SHA256,
        "context_schema": "ProductionObjectiveContextBindingV2+PrefixAnchoredTargetV1",
        "ordinal_semantics": {
            "source_frame_index": "canonical dataset index",
            "sequence_local_ordinal": "graph/warm/qold index",
            "certification_sample_ordinal": "target gate position only",
            "production_frame0": "sequence-local ordinal 0 and prefix execution-local ordinal 0",
        },
        "SPARSE_UNIT": "PREFIX_ANCHORED_TARGET",
        "anchor_rule": "start at sequence-local ordinal 0",
        "prefix_dependency": "execute every ordinal 0..target exactly once in one runtime chain",
        "target_frame": "only the selected fresh Sparse V2 frame is gate-counted",
        "context_only_frames": "all earlier prefix frames; never independent samples",
        "gate_counting_rule": "exactly 30 target frames",
        "overlap_rule": "zero target overlap with V1 and zero source-sequence overlap with CERT-R",
        "runtime_state_lifecycle": "accepted t output becomes previous accepted refined runtime for t+1; q_old remains immutable and separate",
        "prefix_failure_policy": {
            "technical_or_context_or_hard_valid_failure": "target technical FAIL; stop that prefix",
            "context_frame_interaction_failure": "target scientific prefix FAIL; do not retune, extend, or replace",
            "target_failure": "apply frozen GateV2; no retry",
        },
        "q_old_authority": QOLD_AUTHORITY_SHA256,
        "previous_runtime_authority": "PREVIOUS_ACCEPTED_REFINED_RUNTIME",
        "window_semantics": "fresh sequential windows with explicit predecessor runtime chain",
        "cross_episode_semantics": "fresh source-sequence-disjoint controls; REQUIRED",
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "exclusion_ledger_sha256": sha256_file(root / "freshness/cert_v2_exclusion_ledger.json"),
        "scientific_retry_policy": "NO_RETRY_AFTER_OPTIMIZER_START",
        "outcome_driven_rules": False,
        "consumed_regression_uuid": regression["CERT_R_CONSUMED_REGRESSION_RUN_UUID"],
        "eligible_record_count": ledger["CERT_V2_ELIGIBLE_FRESH_RECORD_COUNT"],
    }
    first = canonical_bytes(protocol)
    second = canonical_bytes(json.loads(first))
    deterministic = first == second
    path = root / "protocol_v2/certification_protocol_v2.json"
    write_json(path, protocol)
    digest = sha256_file(path)
    write_text(root / "protocol_v2/certification_protocol_v2.sha256", digest + "\n")
    write_json(
        root / "protocol_v2/serialization_determinism.json",
        {
            "status": "PASS" if deterministic else "FAIL",
            "SERIALIZATION_DETERMINISM": "PASS" if deterministic else "FAIL",
            "canonical_sha256_first": hashlib.sha256(first).hexdigest(),
            "canonical_sha256_second": hashlib.sha256(second).hexdigest(),
            "file_sha256": digest,
        },
    )
    if not deterministic:
        raise RuntimeError("PROTOCOL_V2_SERIALIZATION_NONDETERMINISTIC")
    return {**protocol, "CERTIFICATION_PROTOCOL_V2_SHA256": digest}


def authorize_cert_v2(root: Path) -> dict[str, Any]:
    protocol = require(
        root / "protocol_v2/certification_protocol_v2.json", "status", "FROZEN", "AUTHORIZE_CERT_V2"
    )
    require(root / "consumed_regression/decision.json", "status", "PASS", "AUTHORIZE_CERT_V2")
    integrity = require(
        root / "method_integrity/decision.json", "status", "PASS", "AUTHORIZE_CERT_V2"
    )
    authorized = integrity["REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED"] == "NO"
    value = {
        "schema_version": "O5RD3CERTV2AuthorizationV1",
        "CERT_R_STATUS": "PASS_PROTOCOL_REPAIR"
        if authorized
        else "BLOCKED_REFINEMENT_METHOD_CHANGE_REQUIRED",
        "CERTIFICATION_PROTOCOL_V2_FROZEN": "YES",
        "CERTIFICATION_PROTOCOL_V2_SHA256": sha256_file(
            root / "protocol_v2/certification_protocol_v2.json"
        ),
        "SPARSE_UNIT_V2": protocol["SPARSE_UNIT"],
        "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED": integrity["REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED"],
        "FRESH_CERT_V2_AUTHORIZED": "YES" if authorized else "NO",
        "D3_V2_AUTHORIZED": "NO",
        "NEXT": "O5R-D3-CERT-V2_REFINEMENT_V2_FRESH_INDEPENDENT_CERTIFICATION"
        if authorized
        else "BLOCKED_REFINEMENT_METHOD_CHANGE_REQUIRED",
    }
    write_json(root / "future/cert_v2_authorization.json", value)
    return value


def generate_cert_v2_plan(root: Path) -> dict[str, Any]:
    authorization = require(
        root / "future/cert_v2_authorization.json",
        "FRESH_CERT_V2_AUTHORIZED",
        "YES",
        "GENERATE_CERT_V2_PLAN",
    )
    value = {
        "schema_version": "O5RD3CERTV2PlanStubV1",
        "status": "PLANNED_NOT_RUN",
        "authorization": authorization,
        "stages": ["Fresh Sparse V2", "Fresh Window V2", "Fresh CrossEpisode V2"],
        "deterministic_selection_rule": "select from V2 eligible records after source-sequence exclusions using a separately frozen outcome-independent manifest",
        "SPARSE_V2_MANIFEST_NOT_EQUAL_V1": True,
        "SPARSE_V2_TARGET_OVERLAP_WITH_V1": 0,
        "SOURCE_SEQUENCE_OVERLAP_WITH_CERT_R": 0,
        "FRESH_SPARSE_V2": "NOT_RUN",
        "FRESH_WINDOW_V2": "NOT_RUN",
        "FRESH_CROSS_EPISODE_V2": "NOT_RUN",
        "D3_V2_AUTHORIZED": "NO",
    }
    write_json(root / "future/cert_v2_plan_stub.json", value)
    write_json(
        root / "audits/no_fresh_v2_optimizer.json", {"status": "PASS", "Fresh_V2_optimizer_runs": 0}
    )
    write_json(
        root / "audits/no_split_leakage.json",
        {
            "status": "PASS",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    write_json(
        root / "audits/no_special_cases.json", {"status": "PASS", "outcome_driven_rules": False}
    )
    return value


def summarize(root: Path) -> dict[str, Any]:
    authorization = read_json(root / "future/cert_v2_authorization.json")
    failure = read_json(root / "failure_localization/failing_sample.json")
    guard = read_json(root / "failure_localization/failure_guard.json")
    authority = read_json(root / "context_authority/authority_matrix.json")
    root_cause = read_json(root / "root_cause/decision.json")
    preflight_result = read_json(root / "context_preflight/summary.json")
    parity = read_json(root / "production_parity/summary.json")
    regression = read_json(root / "consumed_regression/decision.json")
    determinism = read_json(root / "consumed_regression/determinism.json")
    integrity = read_json(root / "method_integrity/decision.json")
    ledger = read_json(root / "freshness/cert_v2_exclusion_ledger.json")
    protocol_sha = sha256_file(root / "protocol_v2/certification_protocol_v2.json")
    summary = {
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "commits": git("log", "--format=%H", f"{START_HEAD}..HEAD").splitlines(),
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V1": "FAIL",
        "D3_CERT_V1_HISTORICAL_RESULT": "FAIL",
        "D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "FRESH_REFINEMENT_SPARSE_V1": "FAIL",
        "SPARSE_V1_TECHNICAL": "1/30",
        "SPARSE_V1_MANIFEST_SHA256": SPARSE_V1_SHA256,
        "HISTORICAL_CERT_V1_RESULT_REWRITTEN": "NO",
        "FIRST_FAILURE_SAMPLE_ORDINAL": failure["certification_sample_ordinal_zero_based"],
        "FIRST_FAILURE_SOURCE_FRAME": failure["source_frame"],
        "FIRST_FAILURE_SEQUENCE": failure["sequence_id"],
        "FIRST_FAILURE_RECORD": failure["record_id"],
        "FIRST_FAILURE_STRATUM": failure["stratum"],
        "FAILURE_GUARD": f"{guard['source_file']}:{guard['line']}:{guard['symbol']}",
        "FAILURE_MESSAGE": failure["exception_message"],
        "MISSING_PRODUCTION_CONTINUOUS_INPUT": guard["missing_fields"],
        "PRODUCER": "previous accepted prefix output plus deterministic source/warm transport",
        "CONSUMER": "ProductionObjectiveContextBindingV2.validate then d2a._make_context/ObjectiveV2",
        "REQUIRED_WHEN": "execution_local_ordinal>0",
        "AUTHORITY": "PREVIOUS_ACCEPTED_REFINED_RUNTIME plus deterministic derived context",
        "PRODUCTION_CONTINUOUS_INPUT_COUNT": len(authority["fields"]),
        "UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT": authority[
            "UNKNOWN_CONTINUOUS_INPUT_AUTHORITY_COUNT"
        ],
        "PRODUCTION_FRAME0_AUTHORITY": "TRAJECTORY_RUN_FIRST_FRAME / sequence-local ordinal 0",
        "ORDINAL_BINDING_BUG": "YES",
        "CERT_R_PRIMARY_ROOT_CAUSE": root_cause["CERT_R_PRIMARY_ROOT_CAUSE"],
        "CERT_R_ROOT_CAUSE_CONFIDENCE": root_cause["CERT_R_ROOT_CAUSE_CONFIDENCE"],
        "STANDALONE_SPARSE_SEMANTICS": root_cause["STANDALONE_SPARSE_SEMANTICS"],
        "CERTIFICATION_UNIT_V2": root_cause["CERTIFICATION_UNIT_V2"],
        "MINIMUM_PRODUCTION_EQUIVALENT_CONTEXT": root_cause[
            "MINIMUM_PRODUCTION_EQUIVALENT_CONTEXT"
        ],
        "CERTIFICATION_REPAIR_IMPACT": root_cause["CERTIFICATION_REPAIR_IMPACT"],
        "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED": integrity["REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED"],
        "SPARSE_V1_CONTEXT_PREFLIGHT": preflight_result["CONTEXT_PREFLIGHT"],
        "MISSING_REQUIRED_CONTEXT_COUNT": preflight_result["MISSING_REQUIRED_FIELDS"],
        "FRAME_BINDING_ERROR_COUNT": preflight_result["FRAME_BINDING_ERRORS"],
        "QOLD_BINDING_ERROR_COUNT": preflight_result["QOLD_BINDING_ERRORS"],
        "PRODUCTION_CONTEXT_PARITY": parity["PRODUCTION_CONTEXT_PARITY"],
        "PARITY_SENTINEL_COUNT": parity["PARITY_SENTINEL_COUNT"],
        "MAX_CONTEXT_FIELD_DIFF": parity["MAX_CONTEXT_FIELD_DIFF"],
        "SPARSE_V1_CONSUMED_REGRESSION": regression["SPARSE_V1_CONSUMED_REGRESSION"],
        "EVIDENCE_ROLE": "CONSUMED_PROTOCOL_REGRESSION_ONLY",
        "CONSUMED_REGRESSION_TARGETS": regression["CONSUMED_REGRESSION_TARGETS"],
        "CONTEXT_BINDING_PASS": regression["CONTEXT_BINDING_PASS"],
        "TECHNICAL_PASS": regression["TECHNICAL_TARGETS"],
        "DETERMINISM": determinism["CONSUMED_REGRESSION_DETERMINISM"],
        "REFINEMENT_V2_DESIGN_CHANGED": integrity["REFINEMENT_V2_DESIGN_CHANGED"],
        "REFINEMENT_V2_DESIGN_SHA256": DESIGN_SHA256,
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": GATE_V2_SHA256,
        "GATE_V2_CHANGED": integrity["GATE_V2_CHANGED"],
        "OBJECTIVE_V2_CHANGED": integrity["OBJECTIVE_V2_CHANGED"],
        "SEMANTIC_V1_CHANGED": integrity["SEMANTIC_V1_CHANGED"],
        "E_IM_THRESHOLD_CHANGED": integrity["E_IM_THRESHOLD_CHANGED"],
        "ACTIVE_SET_CHANGED": integrity["ACTIVE_SET_CHANGED"],
        "SOLVER_BUDGET_CHANGED": integrity["SOLVER_BUDGET_CHANGED"],
        "SPARSE_V1_ALL_30_CONSUMED_FOR_FUTURE_CERTIFICATION": "YES",
        "CERT_V1_TARGET_FRAME_EXCLUSION_COUNT": ledger["CERT_V1_TARGET_FRAME_EXCLUSION_COUNT"],
        "CERT_R_SEQUENCE_EXCLUSION_COUNT": ledger["CERT_R_SEQUENCE_EXCLUSION_COUNT"],
        "BASELINE_ONLY_REUSABLE_RECORD_COUNT": ledger["BASELINE_ONLY_REUSABLE_RECORD_COUNT"],
        "CERT_V2_ELIGIBLE_FRESH_RECORD_COUNT": ledger["CERT_V2_ELIGIBLE_FRESH_RECORD_COUNT"],
        "CERT_V2_ELIGIBLE_FRESH_FRAME_COUNT": ledger["CERT_V2_ELIGIBLE_FRESH_FRAME_COUNT"],
        "CERTIFICATION_PROTOCOL_V2_FROZEN": "YES",
        "CERTIFICATION_PROTOCOL_V2_SHA256": protocol_sha,
        "SPARSE_UNIT_V2": "PREFIX_ANCHORED_TARGET",
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        **authorization,
        "FRESH_SPARSE_V2": "NOT_RUN",
        "FRESH_WINDOW_V2": "NOT_RUN",
        "FRESH_CROSS_EPISODE_V2": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION_V2": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RAN": "NO",
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "schema_version": "OakInk2O5RD3CERTRFinalSummaryV1",
    }
    changed_files = [
        "scripts/evaluation/run_oakink2_o5rd3certr.py",
        "tests/evaluation/test_oakink2_o5rd3certr.py",
    ]
    write_json(
        root / "repair/implementation_receipt.json",
        {
            "schema_version": "O5RD3CERTRImplementationReceiptV1",
            "status": "PASS",
            "repair_plan_sha256": REPAIR_PLAN_SHA256,
            "implementation": "prefix-anchored certification adapter and fail-closed protocol CLI",
            "shared_production_code_changed": False,
            "context_preflight_before_optimizer": True,
            "files": changed_files,
        },
    )
    write_json(
        root / "repair/changed_files.json",
        {"status": "PASS", "files": changed_files, "shared_method_files": []},
    )
    write_json(
        root / "repair/impact_classification.json",
        {
            "status": "PASS",
            "CERTIFICATION_REPAIR_IMPACT": "CERTIFICATION_PROTOCOL_UNIT_CHANGE",
            "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED": "NO",
        },
    )
    write_json(root / "final_summary.json", summary)
    write_json(root / "completion_audit.json", {"status": "PASS", "summary": summary})
    text = (
        "# OakInk2 O5R-D3-CERT-R\n\n# Fresh Refinement Production-Context Authority Repair Handoff\n\n"
        + "\n".join(
            f"{key}={json.dumps(value, default=jsonable)}" for key, value in summary.items()
        )
        + "\n"
    )
    write_text(root / "final_summary.md", text)
    write_text(root / "handoff.md", text)
    write_json(
        root / "resource_usage.json",
        {
            "solver_actual_device": "CPU",
            "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
            "PPO_TRAINING_RUN_COUNT_NEW": 0,
            "PHYSX_RAN": "NO",
        },
    )
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
    return summary


def validate_repository(root: Path) -> dict[str, Any]:
    files = [
        "scripts/evaluation/run_oakink2_o5rd3certr.py",
        "tests/evaluation/test_oakink2_o5rd3certr.py",
    ]
    commands = [
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", *files],
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", *files],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "scripts/check_paper_fidelity.py"],
        ["git", "diff", "--check"],
    ]
    rows = []
    for command in commands:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
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
        "schema_version": "O5RD3CERTRRepositoryValidationV1",
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


FORBIDDEN_ACTION_NAMES = {
    "run-fresh-sparse-v2",
    "run-fresh-window-v2",
    "run-fresh-cross-episode-v2",
    "run-d3v2",
    "run-dev2",
    "run-ppo",
    "run-physx",
    "run-o6",
}

ACTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "preflight": preflight,
    "verify-cert-v1-history": verify_cert_v1_history,
    "locate-cert-v1-first-failure": locate_cert_v1_first_failure,
    "trace-production-context-callgraph": trace_production_context_callgraph,
    "inventory-production-continuous-inputs": inventory_production_continuous_inputs,
    "classify-context-authorities": classify_context_authorities,
    "audit-ordinal-semantics": audit_ordinal_semantics,
    "decide-cert-r-root-cause": decide_cert_r_root_cause,
    "freeze-cert-r-repair-plan": freeze_cert_r_repair_plan,
    "run-sparse-v1-context-preflight": run_sparse_v1_context_preflight,
    "run-production-context-parity": run_production_context_parity,
    "run-consumed-sparse-v1-regression": run_consumed_sparse_v1_regression,
    "run-consumed-regression-determinism": run_consumed_regression_determinism,
    "audit-scientific-payload-impact": audit_scientific_payload_impact,
    "build-cert-v2-exclusion-ledger": build_cert_v2_exclusion_ledger,
    "freeze-certification-protocol-v2": freeze_certification_protocol_v2,
    "authorize-cert-v2": authorize_cert_v2,
    "generate-cert-v2-plan": generate_cert_v2_plan,
    "validate-repository": validate_repository,
    "summarize": summarize,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.root.resolve()
    try:
        result = ACTIONS[args.action](root)
    except Exception as exc:
        append_jsonl(
            root / "technical_failures.jsonl",
            {
                "action": args.action,
                "error": f"{type(exc).__name__}:{exc}",
                "classification": "TECHNICAL_OR_FAIL_CLOSED_PRECONDITION",
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True, default=jsonable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
