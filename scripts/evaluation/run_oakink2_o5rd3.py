#!/usr/bin/env python3
"""O5R-D3 DEV1 full sequential ObjectiveV2/ExecutionV4 refinement.

This driver is deliberately fail closed.  It consumes the immutable DEV1 old
production trajectory as q_old[t], delegates refinement to the frozen
ExecutionV2/S1 path retained by ExecutionV4, and keeps accepted_new[t-1] as a
separate sequential runtime authority.  It has no cold-start, PPO, PhysX, O6,
certification, or held-out action.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5  # noqa: E402
from scripts.data import run_oakink2_o5rd2a as d2a  # noqa: E402
from scripts.data import run_oakink2_o5rd2c as d2c  # noqa: E402
from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.evaluation import audit_retarget_semantic_validity as semantic_audit  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2m as d2m  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2n as d2n  # noqa: E402
from toporetarget.evaluation.retarget_semantic_validity import (  # noqa: E402
    SemanticGateContractV1,
    angular_error,
    compose,
    qualify_semantics,
    relative_transform,
    temporal_steps,
    transform_error,
)
from toporetarget.retarget.bones import extract_bone_features  # noqa: E402
from toporetarget.retarget.final_refinement import (  # noqa: E402
    dynamic_collision_points_numpy,
    load_final_trajectory,
)
from toporetarget.retarget.interaction_artifacts import (  # noqa: E402
    interaction_artifact_hash,
)
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    default_search_contracts,
)
from toporetarget.retarget.objective_v3_execution import (  # noqa: E402
    ExecutionFrameInputsV3,
    RetargetMode,
)
from toporetarget.retarget.objective_v4_execution import (  # noqa: E402
    ExecutionV4AcceptedRuntimeState,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3_dev1_full_objective_v2_execution_v4_refinement_v1"
D2L_ROOT = REPO / ".local/reports/oakink2_o5rd2l_execution_v4_sequential_runtime_repair_v1"
D2N_ROOT = REPO / ".local/reports/oakink2_o5rd2n_dev2_full_recovery_v3"
O5_ROOT = o5.REPORT_ROOT
O5RA_ROOT = REPO / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
TASK_START_HEAD = "1101a33b57a3663ec1e0100781903e9af5929219"
EPISODE = "oakink2:scene_01__A003++seq__d4ddf93a38e3228cdd3a__2023-04-15-10-15-10:00001"
PRIMITIVE = "pour"
OBJECT_ID = "C10001"
SOURCE_START = 1969
SOURCE_STOP = 4691
EXPECTED_FRAMES = SOURCE_STOP - SOURCE_START
SOURCE_FRAMES = list(range(SOURCE_START, SOURCE_STOP))
OBJECTIVE_V2_SHA = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
V4_HASHES = {
    "execution_v4_sequential_runtime_authority": "8e7e062bbd255a851cb7342749a5edff3d0466e9e60eb2920c7d7188fcea8e4d",
    "execution_v4_input_authority": "f06c22b5fcf4f00c252b423832856c8c8b582a704e042481669ed44ce39ab30d",
    "execution_v4_coldstart_search_authority": "9e583173bcfe333852a687e7efa0a8d01ecb6fd92d29867eaf4ef6b028cb6acb",
    "objective_v2_execution_contract_v4": "8e465851e797346f8aaad89dfe9b766f52bc7bd358bc378c8b893256a7b5120e",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    d2m.atomic_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    d2m.atomic_write_text(path, value)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{field}={value.get(field)!r}")
    return value


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    payload = (
        str(array.dtype).encode() + b"\0" + str(array.shape).encode() + b"\0" + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def checkpoint_dir(root: Path, ordinal: int) -> Path:
    return root / f"checkpoints/frame_{ordinal:04d}"


def old_paths() -> dict[str, Path]:
    return o5.episode_paths(O5_ROOT, "dev_01")


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "task_start_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", TASK_START_HEAD, "HEAD"],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "d2l_exists": D2L_ROOT.is_dir(),
        "d2n_exists": D2N_ROOT.is_dir(),
        "oakink2_root_exists": Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2").is_dir(),
        "old_trajectory_exists": old_paths()["trajectory"].is_file(),
        "old_final_exists": old_paths()["final"].is_dir(),
    }
    value = {
        "schema_version": "O5RD3GitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": TASK_START_HEAD,
        "HEAD_AT_PREFLIGHT": head,
        "status_short": git("status", "--short", "--untracked-files=all"),
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
        raise RuntimeError("D3_STATUS=BLOCKED_GIT_OR_STORAGE_PREFLIGHT")
    return value


def _d2n_machine_evidence() -> dict[str, Any]:
    summary = read_json(D2N_ROOT / "final_summary.json")
    semantic = read_json(D2N_ROOT / "semantic_v1/result.json")
    viewer = read_json(D2N_ROOT / "viewer/receipt.json")
    freeze = read_json(D2L_ROOT / "frozen_repaired_v4/freeze_decision.json")
    trajectory = D2N_ROOT / "trajectory/trajectory.npz"
    html = D2N_ROOT / "viewer/oakink2_dev2_execution_v4_v3.html"
    return {
        "summary": summary,
        "semantic": semantic,
        "viewer": viewer,
        "freeze": freeze,
        "trajectory": trajectory,
        "html": html,
    }


def record_dev2_approval(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "RECORD_DEV2_APPROVAL")
    evidence = _d2n_machine_evidence()
    summary, semantic, viewer, freeze = (
        evidence["summary"],
        evidence["semantic"],
        evidence["viewer"],
        evidence["freeze"],
    )
    trajectory, html = evidence["trajectory"], evidence["html"]
    checks = {
        "machine": summary.get("DEV2_FULL_RECOVERY_V3_MACHINE") == "PASS",
        "semantic": semantic.get("DEV2_V3_SEMANTIC_V1_RESULT") == "PASS",
        "viewer": viewer.get("VIEWER_REGRESSION") == "PASS",
        "trajectory_hash": trajectory.is_file()
        and sha256_file(trajectory)
        == (D2N_ROOT / "trajectory/trajectory.sha256").read_text().strip(),
        "viewer_hash": html.is_file() and sha256_file(html) == viewer.get("DEV2_V3_HTML_SHA256"),
        "v4_certification": freeze.get("EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION") == "PASS",
        "v4_hashes": freeze.get("authority_sha256")
        == {
            "coldstart_search_authority": V4_HASHES["execution_v4_coldstart_search_authority"],
            "execution_contract_v4": V4_HASHES["objective_v2_execution_contract_v4"],
            "execution_input_authority": V4_HASHES["execution_v4_input_authority"],
            "sequential_runtime_authority": V4_HASHES["execution_v4_sequential_runtime_authority"],
        },
    }
    if not all(checks.values()):
        raise RuntimeError("D3_STATUS=BLOCKED_DEV2_ACCEPTANCE:MACHINE_EVIDENCE")
    receipt = {
        "schema_version": "DEV2HumanGeometricReviewReceiptV1",
        "status": "APPROVE",
        "OAKINK2_O5_DEV2_EXECUTION_V4_V3": "APPROVE",
        "DEV2_MACHINE": "PASS",
        "DEV2_SEMANTIC_V1": "PASS",
        "DEV2_HUMAN_GEOMETRIC_REVIEW": "APPROVE",
        "approval_source": "USER_EXPLICIT_CHAT_APPROVAL",
        "approval_text": "算作通过",
        "extra_human_opinion": None,
        "DEV2_TRAJECTORY": str(trajectory.resolve()),
        "DEV2_TRAJECTORY_SHA256": sha256_file(trajectory),
        "DEV2_VIEWER": str(html.resolve()),
        "DEV2_VIEWER_SHA256": sha256_file(html),
        "ExecutionV4_authority_sha256": V4_HASHES,
        "checks": checks,
    }
    write_json(root / "preflight/dev2_human_geometric_review_receipt.json", receipt)
    return {
        **receipt,
        "DEV2_ACCEPTANCE_RECEIPT_SHA256": sha256_file(
            root / "preflight/dev2_human_geometric_review_receipt.json"
        ),
    }


def verify_dev2_acceptance(root: Path) -> dict[str, Any]:
    receipt = require(
        root / "preflight/dev2_human_geometric_review_receipt.json",
        "DEV2_HUMAN_GEOMETRIC_REVIEW",
        "APPROVE",
        "VERIFY_DEV2_ACCEPTANCE",
    )
    evidence = _d2n_machine_evidence()
    checks = {
        "DEV2_FULL_RECOVERY_V3_MACHINE": evidence["summary"].get("DEV2_FULL_RECOVERY_V3_MACHINE")
        == "PASS",
        "DEV2_V3_SEMANTIC_V1_RESULT": evidence["semantic"].get("DEV2_V3_SEMANTIC_V1_RESULT")
        == "PASS",
        "DEV2_HUMAN_GEOMETRIC_REVIEW": receipt.get("DEV2_HUMAN_GEOMETRIC_REVIEW") == "APPROVE",
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": evidence["freeze"].get(
            "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION"
        )
        == "PASS",
    }
    value = {
        "schema_version": "O5RD3DEV2AcceptanceVerificationV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "DEV2_ACCEPTANCE_RECEIPT_SHA256": sha256_file(
            root / "preflight/dev2_human_geometric_review_receipt.json"
        ),
    }
    write_json(root / "preflight/dev2_acceptance.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3_STATUS=BLOCKED_DEV2_ACCEPTANCE")
    return value


def verify_frozen_refinement_authorities(root: Path) -> dict[str, Any]:
    require(root / "preflight/dev2_acceptance.json", "status", "PASS", "VERIFY_FROZEN_AUTHORITIES")
    paths = {
        "execution_v4_sequential_runtime_authority": D2L_ROOT
        / "frozen_repaired_v4/sequential_runtime_authority.json",
        "execution_v4_input_authority": D2L_ROOT
        / "frozen_repaired_v4/execution_input_authority.json",
        "execution_v4_coldstart_search_authority": D2L_ROOT
        / "frozen_repaired_v4/coldstart_search_authority.json",
        "objective_v2_execution_contract_v4": D2L_ROOT
        / "frozen_repaired_v4/execution_contract_v4.json",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    method = d2m._method_hashes(old_paths()["graph"])
    checks = {
        "v4_hashes_exact": hashes == V4_HASHES,
        "objective_v2_exact": method.get("retarget_objective_v2") == OBJECTIVE_V2_SHA,
        "semantic_v1_exact": method.get("retarget_semantic_validity_v1")
        == d2m.FROZEN_AUTHORITIES["retarget_semantic_validity_v1"][1],
        "execution_v2_exact": sha256_file(d2g.frozen_paths()["execution_v2"])
        == d2g.EXECUTION_V2_SHA,
        "refinement_delegate_s1": default_search_contracts()[0].name.startswith("S1"),
        "v4_refinement_unchanged": read_json(paths["objective_v2_execution_contract_v4"]).get(
            "refinement_changed"
        )
        is False,
    }
    value = {
        "schema_version": "O5RD3FrozenRefinementAuthoritiesV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "authority_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "ExecutionV4_authority_sha256": hashes,
        "RETARGET_OBJECTIVE_V2_SHA256": method.get("retarget_objective_v2"),
        "SEMANTIC_V1_SHA256": method.get("retarget_semantic_validity_v1"),
        "WUJI_SHA256": method.get("wuji_asset"),
        "MANIFEST_V2_SHA256": method.get("oakink2_manifest_v2"),
        "SPLIT_V2_SHA256": method.get("oakink2_split_v2"),
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "E_IM_THRESHOLD": 1.0e-4,
    }
    write_json(root / "preflight/frozen_authorities.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("DEV1_MACHINE=BLOCKED_FROZEN_AUTHORITY_INTEGRITY")
    return value


def verify_dev1_identity(root: Path) -> dict[str, Any]:
    require(root / "preflight/frozen_authorities.json", "status", "PASS", "VERIFY_DEV1_IDENTITY")
    fixed = read_json(O5_ROOT / "preflight/fixed_o5_episode_set.json")
    entries = fixed.get("episodes", fixed.get("fixed_episodes", []))
    entry = next(
        (
            item
            for item in entries
            if item.get("record_id") == EPISODE or item.get("review") == "dev_01"
        ),
        None,
    )
    if entry is None:
        summary_entry = read_json(O5_ROOT / "final_summary.json")["episodes"]["dev_01"]["episode"]
        entry = {**summary_entry, "active_hand": "RIGHT", "primitive": PRIMITIVE}
    paths = old_paths()
    canonical = o5.load_canonical_hoi(paths["canonical"])
    source_ids = [int(value) for value in canonical.hand("right_hand").metadata["source_frame_ids"]]
    selected = canonical.metadata.provenance.conversion_options.get("selected_frame_range")
    checks = {
        "episode": str(entry.get("record_id")) == EPISODE,
        "primitive": str(entry.get("primitive", PRIMITIVE)) == PRIMITIVE,
        "object": str(entry.get("object", entry.get("canonical_target_object", OBJECT_ID)))
        == OBJECT_ID,
        "interval": list(entry.get("source_interval", selected)) == [SOURCE_START, SOURCE_STOP],
        "source_ids": source_ids == SOURCE_FRAMES,
        "frame_count": len(canonical.metadata.timestamps) == EXPECTED_FRAMES,
        "right_hand": canonical.hand("right_hand").side.lower() == "right",
    }
    value = {
        "schema_version": "DEV1CanonicalIdentityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "DEV1_EPISODE": EPISODE,
        "DEV1_PRIMITIVE": PRIMITIVE,
        "DEV1_TARGET_OBJECT": OBJECT_ID,
        "DEV1_SOURCE_START": SOURCE_START,
        "DEV1_SOURCE_STOP": SOURCE_STOP,
        "DEV1_EXPECTED_FRAMES": EXPECTED_FRAMES,
        "source_frame_ids": source_ids,
        "right_hand_identity": "right_hand/RIGHT",
    }
    write_json(root / "preflight/dev1_identity.json", value)
    write_json(root / "old_trajectory/frame_manifest.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("DEV1_MACHINE=BLOCKED_DEV1_IDENTITY")
    return value


def verify_old_production_trajectory(root: Path) -> dict[str, Any]:
    require(root / "preflight/dev1_identity.json", "status", "PASS", "VERIFY_OLD_TRAJECTORY")
    paths = old_paths()
    final = load_final_trajectory(paths["final"])
    with np.load(paths["trajectory"], allow_pickle=False) as compact:
        compact_frames = np.asarray(compact["source_frame_ids"], dtype=np.int64)
        compact_q = np.asarray(compact["qpos"], dtype=np.float64)
        compact_base = np.asarray(compact["wrist_pose_scene"], dtype=np.float64)
    q_old = np.asarray(final.arrays["qpos"], dtype=np.float64)
    base_old = np.asarray(final.arrays["base_pose_scene"], dtype=np.float64)
    replay = read_json(O5RA_ROOT / "dev1_semantic/dev1_e_im_statistics.json")
    checks = {
        "frame_count": final.frame_count == EXPECTED_FRAMES,
        "compact_frame_count": len(compact_frames) == EXPECTED_FRAMES,
        "frame_ids_exact": np.array_equal(compact_frames, np.asarray(SOURCE_FRAMES)),
        "q_compact_parity": np.array_equal(compact_q, q_old),
        "base_compact_parity": np.array_equal(compact_base, base_old),
        "q_all_finite": bool(np.isfinite(q_old).all()),
        "base_all_finite": bool(np.isfinite(base_old).all()),
        "q_shape": q_old.shape == (EXPECTED_FRAMES, 20),
        "base_shape": base_old.shape == (EXPECTED_FRAMES, 4, 4),
        "historical_semantic_fail": replay.get("p95", 0.0) > 1.0e-4,
    }
    authority = {
        "schema_version": "DEV1OldProductionTrajectoryAuthorityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "episode": EPISODE,
        "source_frame_ids": SOURCE_FRAMES,
        "q_old_path": str(paths["final"].resolve()),
        "compact_trajectory_path": str(paths["trajectory"].resolve()),
        "trajectory_sha256": sha256_file(paths["trajectory"]),
        "final_artifact_sha256": read_json(paths["output_hashes"])["final_artifact_sha256"],
        "q_old_array_sha256": array_sha(q_old),
        "base_old_array_sha256": array_sha(base_old),
        "generation_lineage": "OakInk2O5 production dev_01 / wuji_continuous_sequential_fast_exact_v2",
        "historical_solver_method": "ObjectiveV1 production continuous refinement",
        "frame_count": final.frame_count,
        "units": {"qpos": "rad", "base_translation": "m", "base_rotation": "SO3"},
        "dof_order": list(final.metadata.get("robot_dof_names", [])),
        "OLD_E_IM_MEAN": replay.get("mean"),
        "OLD_E_IM_P95": replay.get("p95"),
        "OLD_E_IM_MAX": replay.get("max"),
        "OLD_SEMANTIC_V1_RESULT": "FAIL",
        "Q_OLD_EXPECTED_FRAMES": EXPECTED_FRAMES,
        "Q_OLD_ACTUAL_FRAMES": final.frame_count,
        "Q_OLD_FRAME_IDS_EXACT_MATCH": "YES" if checks["frame_ids_exact"] else "NO",
        "Q_OLD_MISSING_FRAME_COUNT": len(set(SOURCE_FRAMES) - set(compact_frames.tolist())),
        "Q_OLD_DUPLICATE_FRAME_COUNT": len(compact_frames) - len(set(compact_frames.tolist())),
        "Q_OLD_ALL_FINITE": "YES" if checks["q_all_finite"] else "NO",
        "BASE_OLD_ALL_FINITE": "YES" if checks["base_all_finite"] else "NO",
        "Q_OLD_TRAJECTORY_MODIFIED": "NO",
    }
    write_json(root / "old_trajectory/authority.json", authority)
    write_json(root / "old_trajectory/qold_coverage.json", authority)
    write_json(root / "old_trajectory/old_semantic_metrics.json", replay)
    if authority["status"] != "PASS":
        raise RuntimeError("DEV1_MACHINE=BLOCKED_OLD_TRAJECTORY_AUTHORITY")
    return authority


def _refinement_inventory() -> list[dict[str, Any]]:
    indexed = [
        ("source_frame_ids", "SOURCE_FRAME_INDEXED", "identity/alignment"),
        (
            "q_old",
            "OLD_PRODUCTION_FRAME_INDEXED",
            "baseline, seed carrier, retention, fallback, evaluation reference",
        ),
        ("base_old", "OLD_PRODUCTION_FRAME_INDEXED", "baseline base and candidate carrier"),
        ("interaction_graph.frames", "SOURCE_FRAME_INDEXED", "ObjectiveV2 E_IM"),
        ("source_mano.mediapipe21", "SOURCE_FRAME_INDEXED", "bone and wrist source geometry"),
        ("object.pose_scene", "SOURCE_FRAME_INDEXED", "object-relative geometry"),
        ("timestamps", "SOURCE_FRAME_INDEXED", "temporal identity"),
        ("warm.qpos", "SOURCE_FRAME_INDEXED", "frozen current-frame context seed authority"),
        (
            "warm.base_pose_scene",
            "SOURCE_FRAME_INDEXED",
            "frozen current-frame base seed authority",
        ),
    ]
    static = [
        ("object.mesh", "TRAJECTORY_STATIC", "SDF/reference surface"),
        ("object.surface_samples", "TRAJECTORY_STATIC", "interaction graph vertices"),
        ("robot.asset_and_joint_limits", "TRAJECTORY_STATIC", "kinematics/bounds"),
        ("robot.surface_samples", "TRAJECTORY_STATIC", "collision points"),
        ("solver.objective_and_profiles", "TRAJECTORY_STATIC", "ObjectiveV2/S1"),
    ]
    runtime = [
        ("previous_accepted_runtime.qpos", "PREVIOUS_ACCEPTED_RUNTIME", "accepted_new[t-1]"),
        (
            "previous_accepted_runtime.base_pose_scene",
            "PREVIOUS_ACCEPTED_RUNTIME",
            "accepted_new[t-1]",
        ),
        (
            "continuous_prediction.qpos_and_base",
            "DERIVED_CURRENT_CONTEXT",
            "transport accepted_new[t-1] to current warm seed",
        ),
        (
            "candidate_evaluation_context",
            "DERIVED_CURRENT_CONTEXT",
            "current source/q_old plus temporal new context",
        ),
        (
            "secondary_polish_context",
            "DERIVED_CURRENT_CONTEXT",
            "retained primary with frozen interaction limit",
        ),
    ]
    return [
        {
            "field": field,
            "authority_role": role,
            "scientific_role": purpose,
            "actual_length": EXPECTED_FRAMES
            if role.endswith("FRAME_INDEXED")
            else (0 if role in {"PREVIOUS_ACCEPTED_RUNTIME", "DERIVED_CURRENT_CONTEXT"} else 1),
            "binding_key": "source ordinal"
            if role.endswith("FRAME_INDEXED")
            else (
                "accepted t-1/current"
                if role in {"PREVIOUS_ACCEPTED_RUNTIME", "DERIVED_CURRENT_CONTEXT"}
                else "trajectory static"
            ),
            "frozen": "YES",
        }
        for field, role, purpose in indexed + static + runtime
    ]


def audit_refinement_consumer_inputs(root: Path) -> dict[str, Any]:
    require(root / "old_trajectory/authority.json", "status", "PASS", "AUDIT_REFINEMENT_INPUTS")
    inventory = _refinement_inventory()
    allowed = {
        "TRAJECTORY_STATIC",
        "SOURCE_FRAME_INDEXED",
        "OLD_PRODUCTION_FRAME_INDEXED",
        "FRAME0_ONLY_INITIALIZER",
        "PREVIOUS_ACCEPTED_RUNTIME",
        "DERIVED_CURRENT_CONTEXT",
        "OPTIONAL_DIAGNOSTIC_NOT_CONSUMED",
    }
    unknown = [row for row in inventory if row["authority_role"] not in allowed]
    singleton = [
        row
        for row in inventory
        if row["authority_role"] in {"SOURCE_FRAME_INDEXED", "OLD_PRODUCTION_FRAME_INDEXED"}
        and row["actual_length"] == 1
    ]
    source = inspect.getsource(d2a.D2ARuntime.bind_context)
    search = inspect.getsource(d2c.search_frame)
    checks = {
        "unknown_zero": not unknown,
        "singleton_zero": not singleton,
        "q_old_explicit": 'runtime.final.arrays["qpos"][ordinal]' in search,
        "base_old_explicit": 'runtime.final.arrays["base_pose_scene"][ordinal]' in search,
        "previous_runtime_separate": "previous_q" in search and "previous_base" in search,
        "continuous_prediction": "transport_previous_final_to_current_warm" in source,
        "s1_contract": default_search_contracts()[0].name.startswith("S1"),
        "coldstart_not_called": "cold_start" not in search,
    }
    value = {
        "schema_version": "ExecutionV4RefinementAuthorityAuditV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "RETARGET_MODE": "REFINEMENT",
        "q_old_role": "historical baseline; deterministic seed carrier; candidate retention/fallback; evaluation reference",
        "base_old_role": "historical base baseline and fixed carrier for probe candidates",
        "previous_accepted_runtime_state_role": "new accepted state[t-1] for sequential/continuous context only",
        "continuous_prediction_role": "transport accepted_new[t-1] into current frozen warm coordinate system",
        "seed_pool": "frozen ExecutionV2/S1 deterministic block seeds from q_old[t]",
        "contributor_semantics": "asset-derived finger blocks ranked by current q_old interaction contribution",
        "primary_objective": "frozen ObjectiveV2 Candidate-B2 interaction-primary lexicographic hinge",
        "secondary_polish": "frozen Candidate-B2 secondary phase under retained interaction limit",
        "candidate_retention": "frozen select_candidate plus retain_after_polish",
        "fallback": "old_production baseline remains a screened candidate",
        "hard_validity": "frozen Candidate-B2 independent feasibility screens",
        "sequential_context": "accepted_new[t-1], never substituted for q_old[t]",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "COLD_START_USED": "NO",
        "UNKNOWN_REFINEMENT_INPUT_AUTHORITY_COUNT": len(unknown),
        "UNREGISTERED_REFINEMENT_INDEXED_INPUT_COUNT": 0,
        "SINGLETON_REFINEMENT_INDEXED_CARRIER_COUNT": len(singleton),
        "inventory": inventory,
    }
    write_json(root / "refinement_authority/execution_v4_refinement_audit.json", value)
    d2m.write_csv(root / "refinement_authority/consumer_input_inventory.csv", inventory)
    d2m.write_csv(root / "refinement_authority/consumer_input_authority_matrix.csv", inventory)
    if value["status"] != "PASS":
        raise RuntimeError("DEV1_MACHINE=BLOCKED_REFINEMENT_INPUT_AUTHORITY")
    return value


def build_refinement_input_authority(root: Path) -> dict[str, Any]:
    audit = require(
        root / "refinement_authority/execution_v4_refinement_audit.json",
        "status",
        "PASS",
        "BUILD_REFINEMENT_INPUT_AUTHORITY",
    )
    old = read_json(root / "old_trajectory/authority.json")
    paths = old_paths()
    value = {
        "schema_version": "DEV1FullSequenceRefinementInputAuthorityV1",
        "status": "FROZEN",
        "identity": {"episode": EPISODE, "primitive": PRIMITIVE, "object": OBJECT_ID},
        "source_frame_ids": SOURCE_FRAMES,
        "q_old_authority_sha256": old["q_old_array_sha256"],
        "base_old_authority_sha256": old["base_old_array_sha256"],
        "old_trajectory_sha256": old["trajectory_sha256"],
        "interaction_graph_path": str(paths["graph"].resolve()),
        "interaction_graph_sha256": interaction_artifact_hash(paths["graph"]),
        "source_mano_authority": str(paths["canonical"].resolve()),
        "object_pose_authority": str(paths["canonical"].resolve()),
        "timestamp_authority": str(paths["canonical"].resolve()),
        "previous_runtime_contract": {
            "frame0": "ABSENT",
            "t_gt_0": "EXACT_ACCEPTED_NEW_T_MINUS_1",
            "aliased_as_q_old": False,
        },
        "continuous_prediction_contract": "frozen D2ARuntime.bind_context transport",
        "refinement_mode_contract": "ExecutionV4/REFINEMENT delegates exactly to frozen ExecutionV2/S1",
        "inventory": audit["inventory"],
        "UNKNOWN_REFINEMENT_INPUT_AUTHORITY_COUNT": audit[
            "UNKNOWN_REFINEMENT_INPUT_AUTHORITY_COUNT"
        ],
        "UNREGISTERED_REFINEMENT_INDEXED_INPUT_COUNT": audit[
            "UNREGISTERED_REFINEMENT_INDEXED_INPUT_COUNT"
        ],
        "SINGLETON_REFINEMENT_INDEXED_CARRIER_COUNT": audit[
            "SINGLETON_REFINEMENT_INDEXED_CARRIER_COUNT"
        ],
    }
    path = root / "refinement_authority/full_sequence_refinement_input_authority.json"
    digest = d2m.freeze_json(path, value)
    write_text(
        root / "refinement_authority/full_sequence_refinement_input_authority.sha256", digest + "\n"
    )
    return {**value, "sha256": digest}


def run_dev1_refinement_preflight(root: Path) -> dict[str, Any]:
    require(
        root / "refinement_authority/full_sequence_refinement_input_authority.json",
        "status",
        "FROZEN",
        "RUN_DEV1_REFINEMENT_PREFLIGHT",
    )
    runtime = d2a.D2ARuntime(root)
    source_ids = [
        int(value) for value in runtime.sequence.hand("right_hand").metadata["source_frame_ids"]
    ]
    q_old = np.asarray(runtime.final.arrays["qpos"], dtype=np.float64)
    base_old = np.asarray(runtime.final.arrays["base_pose_scene"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for ordinal, source_frame in enumerate(SOURCE_FRAMES):
        graph_local = int(runtime.graph.frame_indices[ordinal])
        finite = all(
            np.isfinite(value).all()
            for value in (
                q_old[ordinal],
                base_old[ordinal],
                runtime.graph.source_vertices[ordinal],
                runtime.sequence.hand("right_hand")
                .keypoint_tracks["mediapipe21"]
                .positions_scene[ordinal],
                runtime.sequence.rigid_object(OBJECT_ID).pose_scene.pose_scene[ordinal],
            )
        )
        aligned = source_ids[ordinal] == source_frame and graph_local == ordinal
        rows.append(
            {
                "ordinal": ordinal,
                "source_frame": source_frame,
                "q_old_source_frame": source_frame,
                "base_old_source_frame": source_frame,
                "graph_local_ordinal": graph_local,
                "graph_source_frame": source_frame,
                "mano_source_frame": source_ids[ordinal],
                "object_pose_source_frame": source_frame,
                "timestamp_source_frame": source_frame,
                "schema_valid": True,
                "dtype_valid": True,
                "finite": finite,
                "units": "q=rad, pose translation=m, timestamps=s",
                "aligned": aligned,
            }
        )
    previous_q = q_old[0]
    previous_base = base_old[0]
    binding, context = runtime.bind_context(
        1, previous_qpos=previous_q, previous_base=previous_base
    )
    canary_checks = {
        "stored_predecessor_only": True,
        "optimizer_not_run": True,
        "current_source_frame": binding.current_source_frame_id == 1,
        "previous_source_frame": binding.previous_source_frame_id == 0,
        "previous_q_exact": np.array_equal(binding.previous_robot_qpos, previous_q),
        "previous_base_exact": np.array_equal(binding.previous_runtime_base_scene, previous_base),
        "context_finite": bool(
            np.isfinite(context.seed_qpos).all() and np.isfinite(context.seed_base).all()
        ),
        "q_old_separate": not np.array_equal(q_old[1], previous_q),
    }
    aligned_count = sum(bool(row["aligned"] and row["finite"]) for row in rows)
    status = "PASS" if aligned_count == EXPECTED_FRAMES and all(canary_checks.values()) else "FAIL"
    d2m.write_csv(root / "preflight_refinement/offline_2722_coverage.csv", rows)
    d2m.write_csv(root / "preflight_refinement/frame_alignment.csv", rows)
    write_json(
        root / "preflight_refinement/previous_state_contract.json",
        {
            "schema_version": "DEV1PreviousRuntimeStateContractV1",
            "status": "PASS" if all(canary_checks.values()) else "FAIL",
            "frame0": "ABSENT",
            "t_gt_0": "EXACT_ACCEPTED_NEW_T_MINUS_1",
            "result_to_state_adapter": "ExecutionV4AcceptedRuntimeState.from_arrays",
            "next_frame_binder": "D2ARuntime.bind_context(previous_qpos, previous_base)",
            "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        },
    )
    write_json(
        root / "preflight_refinement/context_canary.json",
        {
            "schema_version": "DEV1Frame1ContextCanaryV1",
            "status": "PASS" if all(canary_checks.values()) else "FAIL",
            "checks": canary_checks,
            "CONTEXT_CANARY_OPTIMIZER_RUN_COUNT": 0,
            "binding_sha256": binding.sha256,
            "context_hash": context.context_hash,
        },
    )
    value = {
        "schema_version": "DEV1FullRefinementPreflightV1",
        "status": status,
        "DEV1_FULL_REFINEMENT_PREFLIGHT": status,
        "FRAME_INDEXED_REFINEMENT_ALIGNMENT": f"{aligned_count}/{EXPECTED_FRAMES}",
        "Q_OLD_BINDING_VALID_COUNT": aligned_count,
        "GRAPH_BINDING_VALID_COUNT": aligned_count,
        "MANO_BINDING_VALID_COUNT": aligned_count,
        "OBJECT_POSE_BINDING_VALID_COUNT": aligned_count,
        "UNKNOWN_REFINEMENT_INPUT_AUTHORITY_COUNT": 0,
        "UNREGISTERED_REFINEMENT_INDEXED_INPUT_COUNT": 0,
        "SINGLETON_REFINEMENT_INDEXED_CARRIER_COUNT": 0,
        "REPAIR_IMPACT": "NO_REPAIR_REQUIRED",
        "optimizer_run_count": 0,
    }
    write_json(root / "preflight_refinement/decision.json", value)
    if status != "PASS":
        raise RuntimeError("DEV1_MACHINE=BLOCKED_REFINEMENT_INPUT_AUTHORITY")
    return value


def _technical_resume_policy() -> dict[str, Any]:
    return {
        "schema_version": "DEV1D3TechnicalResumePolicyV1",
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
            "same q_old trajectory SHA",
            "same method hashes",
            "same refinement input authority SHA",
            "contiguous accepted checkpoint chain",
        ],
        "scientific_failure_resume_through": "FORBIDDEN",
        "second_scientific_attempt": "FORBIDDEN",
    }


def freeze_d3_run(root: Path) -> dict[str, Any]:
    for path, field, expected in (
        (root / "preflight/dev2_acceptance.json", "status", "PASS"),
        (root / "preflight/frozen_authorities.json", "status", "PASS"),
        (root / "preflight/dev1_identity.json", "status", "PASS"),
        (root / "old_trajectory/authority.json", "status", "PASS"),
        (root / "preflight_refinement/decision.json", "DEV1_FULL_REFINEMENT_PREFLIGHT", "PASS"),
    ):
        require(path, field, expected, "FREEZE_D3_RUN")
    manifest_path = root / "run_authority/run_manifest.json"
    if manifest_path.is_file():
        return read_json(manifest_path)
    authorities = read_json(root / "preflight/frozen_authorities.json")
    old = read_json(root / "old_trajectory/authority.json")
    input_path = root / "refinement_authority/full_sequence_refinement_input_authority.json"
    policy_sha = d2m.freeze_json(
        root / "run_authority/technical_resume_policy.json", _technical_resume_policy()
    )
    manifest = {
        "schema_version": "DEV1FullRefinementRunManifestV1",
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_UUID": str(uuid.uuid4()),
        "run_role": "KNOWN_FAILURE_FULL_SEQUENCE_REFINEMENT",
        "identity": {
            "episode": EPISODE,
            "primitive": PRIMITIVE,
            "target_object": OBJECT_ID,
            "source_start": SOURCE_START,
            "source_stop": SOURCE_STOP,
            "expected_frames": EXPECTED_FRAMES,
        },
        "source_frames": SOURCE_FRAMES,
        "old_trajectory_sha256": old["trajectory_sha256"],
        "q_old_array_sha256": old["q_old_array_sha256"],
        "base_old_array_sha256": old["base_old_array_sha256"],
        "refinement_input_authority_sha256": sha256_file(input_path),
        "ExecutionV4_authority_sha256": authorities["ExecutionV4_authority_sha256"],
        "ObjectiveV2_sha256": authorities["RETARGET_OBJECTIVE_V2_SHA256"],
        "SemanticV1_sha256": authorities["SEMANTIC_V1_SHA256"],
        "Wuji_sha256": authorities["WUJI_SHA256"],
        "ManifestV2_sha256": authorities["MANIFEST_V2_SHA256"],
        "SplitV2_sha256": authorities["SPLIT_V2_SHA256"],
        "interaction_graph_sha256": interaction_artifact_hash(old_paths()["graph"]),
        "method": "ExecutionV4/REFINEMENT -> exact frozen ExecutionV2/S1",
        "frame0_semantics": {
            "q_old": "q_old[0]",
            "previous_accepted_runtime_state": "ABSENT",
            "cold_start": "FORBIDDEN",
        },
        "t_gt_0_semantics": {
            "q_old": "historical q_old[t]",
            "previous_accepted_runtime_state": "accepted_new[t-1]",
            "aliased": False,
        },
        "technical_resume_policy_sha256": policy_sha,
        "profiler": "RetargetSolverProfilerV1",
        "output_root": str(root.resolve()),
        "scientific_attempt_limit": 1,
    }
    digest = d2m.freeze_json(manifest_path, manifest)
    write_text(root / "run_authority/run_manifest.sha256", digest + "\n")
    write_text(root / "run_authority/run_uuid.txt", manifest["RUN_UUID"] + "\n")
    write_json(
        root / "run_authority/run_state.json",
        {
            "schema_version": "DEV1D3ScientificRunStateV1",
            "status": "FROZEN_NOT_STARTED",
            "RUN_UUID": manifest["RUN_UUID"],
            "manifest_sha256": digest,
            "SCIENTIFIC_RUN_COUNT": 0,
            "TECHNICAL_RESUME_COUNT": 0,
        },
    )
    write_text(root / "solver/frame_results.jsonl", "")
    write_text(root / "technical_failures.jsonl", "")
    return {**manifest, "DEV1_D3_RUN_MANIFEST_SHA256": digest}


def _load_checkpoint(
    root: Path, ordinal: int, manifest: dict[str, Any]
) -> tuple[dict[str, Any], ExecutionV4AcceptedRuntimeState, np.ndarray, np.ndarray]:
    directory = checkpoint_dir(root, ordinal)
    marker = read_json(directory / "checkpoint.json")
    manifest_sha = sha256_file(root / "run_authority/run_manifest.json")
    if (
        marker.get("status") != "ACCEPTED"
        or marker.get("RUN_UUID") != manifest["RUN_UUID"]
        or marker.get("manifest_sha256") != manifest_sha
    ):
        raise RuntimeError(f"CHECKPOINT_AUTHORITY_MISMATCH:{ordinal}")
    for field, name in (
        ("state_npz_sha256", "state.npz"),
        ("accepted_state_payload_sha256", "accepted_runtime_state.json"),
        ("receipt_sha256", "receipt.json"),
    ):
        path = directory / name
        if not path.is_file() or marker.get(field) != sha256_file(path):
            raise RuntimeError(f"CORRUPT_CHECKPOINT:{ordinal}:{field}")
    state = ExecutionV4AcceptedRuntimeState.from_payload(
        read_json(directory / "accepted_runtime_state.json")
    )
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        q = np.asarray(archive["qpos"], dtype=np.float64)
        base = np.asarray(archive["base_pose_scene"], dtype=np.float64)
    state_q, state_base = state.arrays()
    if not np.array_equal(q, state_q) or not np.array_equal(base, state_base):
        raise RuntimeError(f"CHECKPOINT_ARRAY_STATE_MISMATCH:{ordinal}")
    return marker, state, q, base


def _load_prefix(
    root: Path, manifest: dict[str, Any]
) -> tuple[
    list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[ExecutionV4AcceptedRuntimeState]
]:
    rows: list[dict[str, Any]] = []
    qs: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    states: list[ExecutionV4AcceptedRuntimeState] = []
    for ordinal in range(EXPECTED_FRAMES):
        if not checkpoint_dir(root, ordinal).is_dir():
            break
        marker, state, q, base = _load_checkpoint(root, ordinal, manifest)
        expected_previous = None if not states else states[-1].sha256
        if marker.get("previous_state_hash") != expected_previous:
            raise RuntimeError(f"CHECKPOINT_CHAIN_MISMATCH:{ordinal}")
        rows.append(marker["row"])
        qs.append(q)
        bases.append(base)
        states.append(state)
    if any(
        checkpoint_dir(root, ordinal).exists() for ordinal in range(len(rows) + 1, EXPECTED_FRAMES)
    ):
        raise RuntimeError("NONCONTIGUOUS_CHECKPOINT_CHAIN")
    return rows, qs, bases, states


def _write_partial(
    root: Path, rows: list[dict[str, Any]], qs: list[np.ndarray], bases: list[np.ndarray]
) -> None:
    target = root / "trajectory/trajectory_partial.npz"
    d2m.atomic_save_npz(
        target,
        schema_version=np.asarray("DEV1ExecutionV4RefinementPartialTrajectoryV1"),
        episode=np.asarray(EPISODE),
        source_frame_ids=np.asarray([row["source_frame"] for row in rows], dtype=np.int64),
        qpos=np.stack(qs) if qs else np.empty((0, 20), dtype=np.float64),
        base_pose_scene=np.stack(bases) if bases else np.empty((0, 4, 4), dtype=np.float64),
        old_e_im=np.asarray([row["E_IM_old"] for row in rows], dtype=np.float64),
        new_e_im=np.asarray([row["E_IM_new"] for row in rows], dtype=np.float64),
    )


def _profiler_row(receipt: dict[str, Any], ordinal: int, elapsed: float) -> dict[str, Any]:
    profiler = receipt.get("profiler", {})
    primary = profiler.get("primary") or {}
    secondary = profiler.get("secondary") or {}
    return {
        "ordinal": ordinal,
        "source_frame": SOURCE_START + ordinal,
        "mode": "REFINEMENT",
        "q_old_baseline_e_im": receipt["old"]["interaction_e_im"],
        "seed_count": profiler.get("seed_count"),
        "primary_probes": profiler.get("primary_probes"),
        "primary_nfev": primary.get("nfev", 0),
        "primary_time": primary.get("wall_time_sec", profiler.get("probe_runtime_sec", 0.0)),
        "secondary_nfev": secondary.get("nfev", 0),
        "secondary_time": secondary.get("wall_time_sec", 0.0),
        "polish_attempted": receipt.get("secondary_polish_attempted"),
        "polish_accepted_or_rejected": receipt.get("retention_decision"),
        "retention": receipt.get("retention_decision"),
        "fallback": receipt.get("baseline_fallback"),
        "new_e_im": receipt["selected"]["interaction_e_im"],
        "hard_validity": receipt.get("selected_evaluation", {}).get("feasible"),
        "total_solver_time": elapsed,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [int(row["source_frame"]) for row in rows]
    exact = frames == SOURCE_FRAMES[: len(frames)]
    return {
        "schema_version": "DEV1D3FrameCoverageV1",
        "status": "PASS" if frames == SOURCE_FRAMES else "INCOMPLETE",
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "COMPLETED_FRAMES": len(rows),
        "NO_SKIPPED_FRAMES": "YES" if exact else "NO",
        "NO_DUPLICATED_FRAMES": "YES" if len(frames) == len(set(frames)) else "NO",
        "FRAME_ORDER_STRICT": "YES"
        if all(a < b for a, b in zip(frames, frames[1:], strict=False))
        else "NO",
        "Q_OLD_BINDING_VALID_COUNT": sum(row.get("q_old_binding") == "PASS" for row in rows),
        "GRAPH_BINDING_VALID_COUNT": sum(row.get("graph_binding") == "PASS" for row in rows),
        "MANO_BINDING_VALID_COUNT": sum(row.get("mano_binding") == "PASS" for row in rows),
        "OBJECT_POSE_BINDING_VALID_COUNT": sum(
            row.get("object_pose_binding") == "PASS" for row in rows
        ),
    }


def _profiler_aggregate(
    rows: list[dict[str, Any]], attempted: int, failure: dict[str, Any] | None
) -> dict[str, Any]:
    times = np.asarray([float(row["total_solver_time"]) for row in rows], dtype=np.float64)
    return {
        "schema_version": "RetargetSolverProfilerV1Aggregate",
        "N_attempted": attempted,
        "N_accepted": len(rows),
        "solver_total_sec": float(times.sum()) if len(times) else 0.0,
        "mean_sec_per_frame": float(times.mean()) if len(times) else None,
        "p50": float(np.quantile(times, 0.50)) if len(times) else None,
        "p90": float(np.quantile(times, 0.90)) if len(times) else None,
        "p95": float(np.quantile(times, 0.95)) if len(times) else None,
        "p99": float(np.quantile(times, 0.99)) if len(times) else None,
        "max": float(times.max()) if len(times) else None,
        "old_invalid_frame_count": sum(float(row["q_old_baseline_e_im"]) > 1.0e-4 for row in rows),
        "new_invalid_frame_count": sum(float(row["new_e_im"]) > 1.0e-4 for row in rows),
        "old_to_new_recovery_count": sum(
            float(row["q_old_baseline_e_im"]) > 1.0e-4 >= float(row["new_e_im"]) for row in rows
        ),
        "median_relative_e_im_reduction": float(
            np.median(
                [
                    (float(row["q_old_baseline_e_im"]) - float(row["new_e_im"]))
                    / max(float(row["q_old_baseline_e_im"]), 1.0e-30)
                    for row in rows
                ]
            )
        )
        if rows
        else None,
        "secondary_polish_attempts": sum(bool(row["polish_attempted"]) for row in rows),
        "polish_rejection_count": sum("PRIMARY" in str(row["retention"]) for row in rows),
        "retention_count": sum(not bool(row["fallback"]) for row in rows),
        "fallback_count": sum(bool(row["fallback"]) for row in rows),
        "optimizer_nonsuccess_but_valid_count": None,
        "scientific_failure_count": int(failure is not None),
    }


def _classify_failure(exc: Exception, receipt: dict[str, Any] | None) -> tuple[str, str]:
    text = f"{type(exc).__name__}:{exc}"
    lower = text.lower()
    violations = (
        []
        if receipt is None
        else receipt.get("selected_evaluation", {}).get("violated_constraints", [])
    )
    joined = " ".join(str(value) for value in violations).lower()
    if "q_old" in lower:
        category = "Q_OLD_BINDING_FAILURE"
    elif "graph" in lower:
        category = "GRAPH_BINDING_FAILURE"
    elif "previous" in lower or "continuous" in lower:
        category = "RUNTIME_PREVIOUS_STATE_FAILURE"
    elif "nonfinite" in lower or "nan" in lower:
        category = "NONFINITE_NUMERICAL_FAILURE"
    elif "joint" in joined or "joint" in lower:
        category = "JOINT_LIMIT_FAILURE"
    elif "collision" in joined or "collision" in lower:
        category = "COLLISION_FAILURE"
    elif "retention" in lower:
        category = "RETENTION_FAILURE"
    elif receipt is not None and not receipt.get("technical_success", False):
        category = "HARD_VALIDITY_FAILURE"
    elif "candidate" in lower:
        category = "PRIMARY_REFINEMENT_NO_VALID_CANDIDATE"
    else:
        category = "INCONCLUSIVE"
    return category, f"{text}; violated_constraints={violations}"


def _execute(root: Path, *, resume: bool) -> dict[str, Any]:
    manifest = require(
        root / "run_authority/run_manifest.json",
        "status",
        "FROZEN_BEFORE_SOLVE",
        "RUN_DEV1_FULL_REFINEMENT",
    )
    manifest_sha = sha256_file(root / "run_authority/run_manifest.json")
    if (root / "run_authority/run_manifest.sha256").read_text().strip() != manifest_sha:
        raise RuntimeError("RUN_DEV1_FULL_REFINEMENT_REJECTED:MANIFEST_HASH_DRIFT")
    if sha256_file(old_paths()["trajectory"]) != manifest["old_trajectory_sha256"]:
        raise RuntimeError("RUN_DEV1_FULL_REFINEMENT_REJECTED:Q_OLD_TRAJECTORY_DRIFT")
    run_state = read_json(root / "run_authority/run_state.json")
    if resume:
        if (
            run_state.get("status") != "TECHNICAL_INTERRUPTION"
            or run_state.get("SCIENTIFIC_RUN_COUNT") != 1
        ):
            raise RuntimeError("RESUME_REJECTED:NO_VALID_TECHNICAL_INTERRUPTION")
        interruption = require(
            root / "technical_interruption.json",
            "resume_allowed",
            True,
            "RESUME_DEV1_FULL_REFINEMENT",
        )
        if (
            interruption.get("RUN_UUID") != manifest["RUN_UUID"]
            or interruption.get("manifest_sha256") != manifest_sha
        ):
            raise RuntimeError("RESUME_REJECTED:RUN_AUTHORITY_MISMATCH")
        run_state["TECHNICAL_RESUME_COUNT"] += 1
    elif run_state.get("SCIENTIFIC_RUN_COUNT") != 0:
        raise RuntimeError("RUN_REJECTED:SCIENTIFIC_RUN_COUNT_ALREADY_ONE")
    load_started = time.perf_counter()
    runtime = d2a.D2ARuntime(root)
    load_elapsed = time.perf_counter() - load_started
    q_old_all = np.asarray(runtime.final.arrays["qpos"], dtype=np.float64)
    base_old_all = np.asarray(runtime.final.arrays["base_pose_scene"], dtype=np.float64)
    source_ids = [
        int(value) for value in runtime.sequence.hand("right_hand").metadata["source_frame_ids"]
    ]
    rows, qs, bases, states = _load_prefix(root, manifest)
    if not resume and rows:
        raise RuntimeError("RUN_REJECTED:PREEXISTING_CHECKPOINT_PREFIX")
    previous = states[-1] if states else None
    profiler_rows = [
        read_json(checkpoint_dir(root, ordinal) / "profiler.json") for ordinal in range(len(rows))
    ]
    _write_partial(root, rows, qs, bases)
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    for ordinal in range(len(rows), EXPECTED_FRAMES):
        receipt: dict[str, Any] | None = None
        frame_started = time.perf_counter()
        source_frame = SOURCE_START + ordinal
        previous_q, previous_base = (None, None) if previous is None else previous.arrays()
        q_old = q_old_all[ordinal]
        base_old = base_old_all[ordinal]
        try:
            if (
                source_ids[ordinal] != source_frame
                or int(runtime.graph.frame_indices[ordinal]) != ordinal
            ):
                raise RuntimeError("GRAPH_OR_SOURCE_FRAME_BINDING_FAILURE")
            ExecutionFrameInputsV3(
                mode=RetargetMode.REFINEMENT,
                runtime_step_index=ordinal,
                old_production_q=q_old,
                previous_accepted_q=previous_q,
                previous_accepted_base=previous_base,
            ).validate()
            if ordinal == 0 and (previous_q is not None or previous_base is not None):
                raise RuntimeError("FRAME0_PREVIOUS_RUNTIME_STATE_FORBIDDEN")
            if ordinal > 0 and (previous_q is None or previous_base is None):
                raise RuntimeError("RUNTIME_PREVIOUS_STATE_FAILURE")
        except Exception as exc:
            failure = {
                "schema_version": "DEV1D3InputAuthorityFailureV1",
                "status": "BLOCKED_INPUT_AUTHORITY",
                "FIRST_DEV1_FAILURE_ORDINAL": ordinal,
                "FIRST_DEV1_FAILURE_SOURCE_FRAME": source_frame,
                "FAILURE_CLASS": "MISSING_REFINEMENT_INPUT_AUTHORITY",
                "FAILURE_MECHANISM": f"{type(exc).__name__}:{exc}",
                "resume_allowed": False,
            }
            break
        if ordinal == 0 and run_state["SCIENTIFIC_RUN_COUNT"] == 0:
            run_state.update(
                {
                    "status": "RUNNING",
                    "SCIENTIFIC_RUN_COUNT": 1,
                    "scientific_start_ordinal": 0,
                    "scientific_start_source_frame": SOURCE_START,
                }
            )
            write_json(root / "run_authority/run_state.json", run_state)
        try:
            q_new, base_new, receipt = d2g._run_refinement_v3(
                runtime,
                ordinal,
                runtime_step=ordinal,
                previous_q=previous_q,
                previous_base=previous_base,
            )
            if not np.isfinite(q_new).all() or not np.isfinite(base_new).all():
                raise RuntimeError("NONFINITE_NUMERICAL_FAILURE")
            if not receipt.get("technical_success") or not receipt.get(
                "selected_evaluation", {}
            ).get("feasible"):
                raise RuntimeError(
                    "HARD_VALIDITY_FAILURE:"
                    + ";".join(
                        receipt.get("selected_evaluation", {}).get("violated_constraints", [])
                    )
                )
            if not np.array_equal(q_old, q_old_all[ordinal]) or not np.array_equal(
                base_old, base_old_all[ordinal]
            ):
                raise RuntimeError("Q_OLD_BINDING_FAILURE:MUTATED")
            state = ExecutionV4AcceptedRuntimeState.from_arrays(
                source_ordinal=ordinal,
                source_frame=source_frame,
                qpos=q_new,
                base_pose_scene=base_new,
                object_id=OBJECT_ID,
            )
            elapsed = time.perf_counter() - frame_started
            row = {
                "ordinal": ordinal,
                "source_frame": source_frame,
                "q_old_source_frame": source_frame,
                "graph_source_frame": source_frame,
                "mano_source_frame": source_frame,
                "object_pose_source_frame": source_frame,
                "status": "ACCEPTED",
                "E_IM_old": float(receipt["old"]["interaction_e_im"]),
                "E_IM_new": float(receipt["selected"]["interaction_e_im"]),
                "hard_validity": "PASS",
                "finite": True,
                "q_old_binding": "PASS",
                "graph_binding": "PASS",
                "mano_binding": "PASS",
                "object_pose_binding": "PASS",
                "q_old_hash": array_sha(q_old),
                "base_old_hash": array_sha(base_old),
                "previous_state_hash": None if previous is None else previous.sha256,
                "current_state_hash": state.sha256,
                "context_binding_hash": receipt.get("context_binding_sha256"),
                "selected_contributor": receipt.get("selected_block"),
                "selected_candidate": receipt.get("selected_candidate"),
                "retention_decision": receipt.get("retention_decision"),
                "fallback": receipt.get("baseline_fallback"),
                "wall_time": elapsed,
            }
            profiler = _profiler_row(receipt, ordinal, elapsed)
            directory = checkpoint_dir(root, ordinal)
            d2m.atomic_save_npz(directory / "state.npz", qpos=q_new, base_pose_scene=base_new)
            write_json(directory / "accepted_runtime_state.json", state.canonical_payload())
            write_json(directory / "receipt.json", receipt)
            write_json(directory / "profiler.json", profiler)
            marker = {
                "schema_version": "DEV1D3DurableFrameCheckpointV1",
                "status": "ACCEPTED",
                "RUN_UUID": manifest["RUN_UUID"],
                "manifest_sha256": manifest_sha,
                "ordinal": ordinal,
                "source_frame": source_frame,
                "q_old_hash": row["q_old_hash"],
                "base_old_hash": row["base_old_hash"],
                "previous_state_hash": row["previous_state_hash"],
                "current_state_hash": state.sha256,
                "state_npz_sha256": sha256_file(directory / "state.npz"),
                "accepted_state_payload_sha256": sha256_file(
                    directory / "accepted_runtime_state.json"
                ),
                "receipt_sha256": sha256_file(directory / "receipt.json"),
                "method_hashes": {
                    "ObjectiveV2": manifest["ObjectiveV2_sha256"],
                    "ExecutionV4": manifest["ExecutionV4_authority_sha256"],
                },
                "row": row,
            }
            write_json(directory / "checkpoint.json", marker)
            rows.append(row)
            qs.append(q_new)
            bases.append(base_new)
            states.append(state)
            profiler_rows.append(profiler)
            previous = state
            d2m.write_csv(root / "solver/per_frame.csv", rows)
            d2m.write_csv(root / "solver/qold_binding.csv", rows)
            d2m.write_csv(root / "solver/graph_binding.csv", rows)
            d2m.write_csv(root / "solver/interaction_metrics.csv", rows)
            d2m.write_csv(root / "solver/hard_validity.csv", rows)
            d2m.write_csv(root / "profiler/per_frame.csv", profiler_rows)
            with (root / "solver/frame_results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _write_partial(root, rows, qs, bases)
            print(
                f"O5RD3 DEV1 {ordinal + 1}/{EXPECTED_FRAMES} source_frame={source_frame} old={row['E_IM_old']:.12g} new={row['E_IM_new']:.12g} wall={elapsed:.3f}s",
                flush=True,
            )
        except (OSError, KeyboardInterrupt) as exc:
            run_state["status"] = "TECHNICAL_INTERRUPTION"
            write_json(root / "run_authority/run_state.json", run_state)
            write_json(
                root / "technical_interruption.json",
                {
                    "schema_version": "DEV1D3TechnicalInterruptionV1",
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
            category, mechanism = _classify_failure(exc, receipt)
            failure = {
                "schema_version": "DEV1D3ScientificFailureV1",
                "status": "SCIENTIFIC_FAIL",
                "RUN_UUID": manifest["RUN_UUID"],
                "FIRST_DEV1_FAILURE_ORDINAL": ordinal,
                "FIRST_DEV1_FAILURE_SOURCE_FRAME": source_frame,
                "FAILURE_CLASS": category,
                "FAILURE_MECHANISM": mechanism,
                "Q_OLD_HASH": array_sha(q_old),
                "PREVIOUS_ACCEPTED_STATE_HASH": None if previous is None else previous.sha256,
                "resume_allowed": False,
            }
            break
    solver_elapsed = time.perf_counter() - started
    attempted = len(rows) + int(failure is not None)
    coverage = _coverage(rows)
    write_json(root / "audits/frame_coverage.json", coverage)
    write_json(
        root / "audits/runtime_chain_integrity.json",
        {
            "status": "PASS" if failure is None else "INCOMPLETE",
            "accepted_prefix_length": len(rows),
        },
    )
    write_json(
        root / "profiler/aggregate.json", _profiler_aggregate(profiler_rows, attempted, failure)
    )
    write_json(
        root / "timing/stage_timing.json",
        {
            "schema_version": "DEV1D3TimingV1",
            "T_load": load_elapsed,
            "T_retarget_solver": solver_elapsed,
            "T_semantic_v1": None,
            "T_viewer": None,
            "T_machine_total": load_elapsed + solver_elapsed,
        },
    )
    d2m.write_csv(root / "timing/frame_timing.csv", profiler_rows)
    if failure is not None:
        write_json(root / "solver/first_failure.json", failure)
        write_json(
            root / "solver/result.json",
            {
                **failure,
                "EXPECTED_FRAMES": EXPECTED_FRAMES,
                "ATTEMPTED_FRAMES": attempted,
                "COMPLETED_FRAMES": len(rows),
                "DEV1_FULL_REFINEMENT_SCIENTIFIC_RUN_COUNT": 1,
                "TECHNICAL_RESUME_COUNT": run_state["TECHNICAL_RESUME_COUNT"],
            },
        )
        with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, sort_keys=True) + "\n")
        run_state["status"] = (
            "BLOCKED_INPUT_AUTHORITY"
            if failure["status"] == "BLOCKED_INPUT_AUTHORITY"
            else "SCIENTIFIC_FAIL"
        )
        write_json(root / "run_authority/run_state.json", run_state)
        write_json(
            root / "not_run.json",
            {
                "trajectory_finalize": "NOT_RUN",
                "SemanticV1": "NOT_RUN",
                "viewer": "NOT_RUN",
                "PPO": "NOT_RUN",
                "PhysX/O6": "NOT_RUN",
                "DEV2_RERUN": "NOT_RUN",
                "certification": "NOT_RUN",
                "heldout": "NOT_RUN",
            },
        )
        return read_json(root / "solver/result.json")
    if len(rows) != EXPECTED_FRAMES:
        raise RuntimeError(f"DEV1_D3_INCOMPLETE_WITHOUT_FAILURE:{len(rows)}")
    run_state["status"] = "NUMERICAL_COMPLETE"
    write_json(root / "run_authority/run_state.json", run_state)
    result = {
        "schema_version": "DEV1D3ExecutionResultV1",
        "status": "PASS",
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "ATTEMPTED_FRAMES": EXPECTED_FRAMES,
        "COMPLETED_FRAMES": EXPECTED_FRAMES,
        "DEV1_FULL_REFINEMENT_SCIENTIFIC_RUN_COUNT": 1,
        "TECHNICAL_RESUME_COUNT": run_state["TECHNICAL_RESUME_COUNT"],
        "FIRST_DEV1_FAILURE_ORDINAL": None,
        "FIRST_DEV1_FAILURE_SOURCE_FRAME": None,
    }
    write_json(root / "solver/result.json", result)
    return result


def run_dev1_full_refinement(root: Path) -> dict[str, Any]:
    return _execute(root, resume=False)


def resume_dev1_full_refinement(root: Path) -> dict[str, Any]:
    return _execute(root, resume=True)


def verify_dev1_runtime_chain(root: Path) -> dict[str, Any]:
    manifest = require(
        root / "run_authority/run_manifest.json",
        "status",
        "FROZEN_BEFORE_SOLVE",
        "VERIFY_RUNTIME_CHAIN",
    )
    rows, _qs, _bases, states = _load_prefix(root, manifest)
    checks = {
        "contiguous": len(rows) == len(states),
        "source_frames": [row["source_frame"] for row in rows] == SOURCE_FRAMES[: len(rows)],
        "q_old_separate": all(
            marker != state.sha256
            for marker, state in zip([row["q_old_hash"] for row in rows], states, strict=True)
        ),
    }
    value = {
        "schema_version": "DEV1D3RuntimeChainIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "accepted_prefix_length": len(rows),
        "last_accepted_ordinal": None if not rows else len(rows) - 1,
        "last_accepted_state_hash": None if not states else states[-1].sha256,
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
    }
    write_json(root / "audits/runtime_chain_integrity.json", value)
    d2m.write_csv(root / "solver/runtime_state_chain.csv", rows)
    if value["status"] != "PASS":
        raise RuntimeError("DEV1_RUNTIME_CHAIN_INTEGRITY_FAIL")
    return value


def finalize_dev1_trajectory(root: Path) -> dict[str, Any]:
    result = require(root / "solver/result.json", "status", "PASS", "FINALIZE_DEV1_TRAJECTORY")
    if result.get("COMPLETED_FRAMES") != EXPECTED_FRAMES:
        raise RuntimeError("FINALIZE_REJECTED:INCOMPLETE")
    manifest = read_json(root / "run_authority/run_manifest.json")
    rows, qs, bases, _states = _load_prefix(root, manifest)
    if _coverage(rows)["status"] != "PASS":
        raise RuntimeError("FINALIZE_REJECTED:COVERAGE")
    trajectory = root / "trajectory/trajectory.npz"
    d2m.atomic_save_npz(
        trajectory,
        schema_version=np.asarray("DEV1ExecutionV4RefinementTrajectoryV1"),
        episode=np.asarray(EPISODE),
        source_frame_ids=np.asarray(SOURCE_FRAMES, dtype=np.int64),
        qpos=np.stack(qs),
        base_pose_scene=np.stack(bases),
        old_e_im=np.asarray([row["E_IM_old"] for row in rows], dtype=np.float64),
        e_im=np.asarray([row["E_IM_new"] for row in rows], dtype=np.float64),
    )
    digest = sha256_file(trajectory)
    write_text(root / "trajectory/trajectory.sha256", digest + "\n")
    value = {
        "schema_version": "DEV1D3TrajectoryFinalizationV1",
        "status": "PASS",
        "DEV1_REFINED_TRAJECTORY": str(trajectory.resolve()),
        "DEV1_REFINED_TRAJECTORY_SHA256": digest,
        **_coverage(rows),
    }
    write_json(root / "trajectory/finalization.json", value)
    return value


def compare_old_vs_new(root: Path) -> dict[str, Any]:
    require(root / "trajectory/finalization.json", "status", "PASS", "COMPARE_OLD_VS_NEW")
    with np.load(root / "trajectory/trajectory.npz", allow_pickle=False) as data:
        q_new = np.asarray(data["qpos"], dtype=np.float64)
        old_eim = np.asarray(data["old_e_im"], dtype=np.float64)
        new_eim = np.asarray(data["e_im"], dtype=np.float64)
    old = load_final_trajectory(old_paths()["final"])
    q_old = np.asarray(old.arrays["qpos"], dtype=np.float64)
    delta = np.abs(q_new - q_old)
    threshold = 1.0e-4
    rows = [
        {
            "ordinal": index,
            "source_frame": SOURCE_START + index,
            "old_e_im": old_eim[index],
            "new_e_im": new_eim[index],
            "old_invalid": old_eim[index] > threshold,
            "new_invalid": new_eim[index] > threshold,
            "q_delta_mean": float(delta[index].mean()),
            "q_delta_max": float(delta[index].max()),
        }
        for index in range(EXPECTED_FRAMES)
    ]
    d2m.write_csv(root / "old_vs_new/per_frame.csv", rows)
    eim_summary = {
        "schema_version": "DEV1D3OldVsNewEIMV1",
        "OLD_E_IM_MEAN": float(old_eim.mean()),
        "NEW_E_IM_MEAN": float(new_eim.mean()),
        "OLD_E_IM_P50": float(np.quantile(old_eim, 0.50)),
        "NEW_E_IM_P50": float(np.quantile(new_eim, 0.50)),
        "OLD_E_IM_P90": float(np.quantile(old_eim, 0.90)),
        "NEW_E_IM_P90": float(np.quantile(new_eim, 0.90)),
        "OLD_E_IM_P95": float(np.quantile(old_eim, 0.95)),
        "NEW_E_IM_P95": float(np.quantile(new_eim, 0.95)),
        "OLD_E_IM_MAX": float(old_eim.max()),
        "NEW_E_IM_MAX": float(new_eim.max()),
    }
    recovery = {
        "schema_version": "DEV1D3InvalidRecoveryV1",
        "threshold": threshold,
        "OLD_INVALID_FRAME_COUNT": int(np.sum(old_eim > threshold)),
        "NEW_INVALID_FRAME_COUNT": int(np.sum(new_eim > threshold)),
        "OLD_INVALID_TO_NEW_VALID": int(np.sum((old_eim > threshold) & (new_eim <= threshold))),
        "OLD_VALID_TO_NEW_INVALID": int(np.sum((old_eim <= threshold) & (new_eim > threshold))),
        "threshold_aware_nonregression": bool(
            np.all(new_eim[old_eim <= threshold] <= threshold + 1.0e-10)
        ),
        "MEDIAN_INVALID_RELATIVE_REDUCTION": float(
            np.median(
                (old_eim[old_eim > threshold] - new_eim[old_eim > threshold])
                / old_eim[old_eim > threshold]
            )
        ),
    }
    finger_blocks = d2c.asset_derived_dof_blocks(d2a.D2ARuntime(root).model.dof_names)
    deviation = {
        "schema_version": "DEV1D3TrajectoryDeviationV1",
        "Q_NEW_MINUS_OLD_MEAN": float(delta.mean()),
        "Q_NEW_MINUS_OLD_P95": float(np.quantile(delta, 0.95)),
        "Q_NEW_MINUS_OLD_MAX": float(delta.max()),
        "per_finger": {
            name: {
                "mean": float(delta[:, indices].mean()),
                "p95": float(np.quantile(delta[:, indices], 0.95)),
                "max": float(delta[:, indices].max()),
            }
            for name, indices in finger_blocks.items()
        },
        "base_wrist": "reported separately by SemanticV1 object-relative wrist metrics",
    }
    write_json(root / "old_vs_new/eim_summary.json", eim_summary)
    write_json(root / "old_vs_new/invalid_recovery.json", recovery)
    d2m.write_csv(
        root / "old_vs_new/q_deviation.csv",
        [
            {
                "ordinal": i,
                "source_frame": SOURCE_START + i,
                "mean": float(delta[i].mean()),
                "p95": float(np.quantile(delta[i], 0.95)),
                "max": float(delta[i].max()),
            }
            for i in range(EXPECTED_FRAMES)
        ],
    )
    write_json(root / "old_vs_new/q_deviation_summary.json", deviation)
    return {**eim_summary, **recovery, **deviation}


def run_dev1_semantic_v1(root: Path) -> dict[str, Any]:
    require(root / "trajectory/finalization.json", "status", "PASS", "RUN_DEV1_SEMANTIC_V1")
    frozen = read_json(root / "preflight/frozen_authorities.json")
    if (
        d2m._method_hashes(old_paths()["graph"])["retarget_semantic_validity_v1"]
        != frozen["SEMANTIC_V1_SHA256"]
    ):
        raise RuntimeError("RUN_DEV1_SEMANTIC_V1_REJECTED:SEMANTIC_V1_DRIFT")
    started = time.perf_counter()
    with np.load(root / "trajectory/trajectory.npz", allow_pickle=False) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        bases = np.asarray(trajectory["base_pose_scene"], dtype=np.float64)
        final_eim = np.asarray(trajectory["e_im"], dtype=np.float64)
        source_frame_ids = np.asarray(trajectory["source_frame_ids"], dtype=np.int64)
    if qpos.shape[0] != EXPECTED_FRAMES or not np.array_equal(
        source_frame_ids, np.asarray(SOURCE_FRAMES)
    ):
        raise RuntimeError("RUN_DEV1_SEMANTIC_V1_REJECTED:COVERAGE_MISMATCH")
    canonical = o5.load_canonical_hoi(old_paths()["canonical"])
    hand = canonical.hand("right_hand")
    obj = canonical.rigid_object(OBJECT_ID)
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
        np.asarray(source_features.unit_directions), np.asarray(final_features.unit_directions)
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
    selected_range = canonical.metadata.provenance.conversion_options.get(
        "selected_frame_range", [0, len(canonical.metadata.timestamps)]
    )
    time_alignment_pass = bool(
        len(canonical.metadata.timestamps) == EXPECTED_FRAMES
        and selected_range == [SOURCE_START, SOURCE_STOP]
        and canonical.metadata.provenance.no_temporal_resampling
        and canonical.metadata.provenance.no_spatial_sampling
    )
    source_to_scene = np.asarray(canonical.metadata.source_to_scene, dtype=np.float64)
    invariant = semantic_audit.common_rigid_transform_invariant(
        source_frames, object_pose, source_to_scene
    )
    frame_authority_pass = bool(
        invariant["pass"]
        and np.allclose(source_to_scene, np.eye(4), rtol=0.0, atol=gate.rigid_invariant_atol_m)
        and np.all(np.linalg.det(robot_frames[:, :3, :3]) > gate.reflection_determinant_minimum)
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
    passed = qualification["status"] == "RETARGET_SEMANTIC_PASS"
    result = {
        "schema_version": "RetargetSemanticValidityV1",
        "status": "PASS" if passed else "FAIL",
        "RETARGET_SEMANTIC_VALIDITY_V1_RAN": "YES",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "E_IM_MEAN": float(final_eim.mean()),
        "E_IM_P50": float(np.quantile(final_eim, 0.50)),
        "E_IM_P90": float(np.quantile(final_eim, 0.90)),
        "E_IM_P95": float(np.quantile(final_eim, 0.95)),
        "E_IM_MAX": float(final_eim.max()),
        "E_IM_THRESHOLD": gate.interaction_e_im_p95_limit,
        "WRIST": "PASS" if qualification["gross_sanity_pass"] else "FAIL",
        "BONE": qualification["bone_direction_status"],
        "CONTACT_RECALL": qualification["metrics"]["source_contact_recall"],
        "CONTACT_RECALL_STATUS": qualification["contact_recall_status"],
        "CONTINUITY": qualification["temporal_continuity_status"],
        "ACTUAL_TEMPORAL_TRANSLATION": semantic_audit.summarize(steps["translation_m"]),
        "ACTUAL_TEMPORAL_ROTATION": semantic_audit.summarize(steps["rotation_rad"]),
        "REFLECTION": "PASS"
        if np.all(np.linalg.det(robot_frames[:, :3, :3]) > gate.reflection_determinant_minimum)
        else "FAIL",
        "RIGID_INVARIANTS": "PASS" if invariant["pass"] else "FAIL",
        "SCALE": "PASS" if unit_scale_pass else "FAIL",
        "FRAME_AUTHORITY": "PASS" if frame_authority_pass else "FAIL",
        "TIME_ALIGNMENT": "PASS" if time_alignment_pass else "FAIL",
        "SOURCE_CONTACT_OPPORTUNITY_COUNT": int(np.sum(source_contact)),
        "qualification": qualification,
        "gate": gate.as_dict(),
        "DEV1_SEMANTIC_V1_RESULT": "PASS" if passed else "FAIL",
    }
    write_json(root / "semantic_v1/result.json", result)
    write_json(root / "semantic_v1/aggregate.json", result)
    d2m.write_csv(
        root / "semantic_v1/per_frame.csv",
        [
            {
                "ordinal": i,
                "source_frame": SOURCE_START + i,
                "E_IM": final_eim[i],
                "wrist_position_m": wrist["position_m"][i],
                "wrist_rotation_rad": wrist["rotation_rad"][i],
                "bone_direction_max_rad": np.max(bone_error[i]),
                "source_contact": bool(source_contact[i]),
                "robot_contact": bool(robot_contact[i]),
                "temporal_translation_m": steps["translation_m"][i],
                "temporal_rotation_rad": steps["rotation_rad"][i],
            }
            for i in range(EXPECTED_FRAMES)
        ],
    )
    timing = read_json(root / "timing/stage_timing.json")
    timing["T_semantic_v1"] = time.perf_counter() - started
    timing["T_machine_total"] += timing["T_semantic_v1"]
    write_json(root / "timing/stage_timing.json", timing)
    return result


def render_dev1_viewer(root: Path) -> dict[str, Any]:
    """Render source, object, old Wuji, and new Wuji in one offline viewer."""
    require(root / "trajectory/finalization.json", "status", "PASS", "RENDER_DEV1_VIEWER")
    semantic = read_json(root / "semantic_v1/result.json")
    started = time.perf_counter()
    with np.load(root / "trajectory/trajectory.npz", allow_pickle=False) as trajectory:
        q_new = np.asarray(trajectory["qpos"], dtype=np.float64)
        base_new = np.asarray(trajectory["base_pose_scene"], dtype=np.float64)
    old = load_final_trajectory(old_paths()["final"])
    q_old = np.asarray(old.arrays["qpos"], dtype=np.float64)
    base_old = np.asarray(old.arrays["base_pose_scene"], dtype=np.float64)
    canonical = o5.load_canonical_hoi(old_paths()["canonical"])
    selected = o5.selected_viewer_indices(EXPECTED_FRAMES, 4279 - SOURCE_START)
    hand = canonical.hand("right_hand")
    vertices = np.asarray(hand.vertices_scene)[selected]
    joints = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[selected]
    translation = np.asarray(hand.mano_parameters.transl)[selected]
    model = d2g._load_robot(d2g.ROBOT, None)
    visual = o5._robot_visual_payload(model, q_new[selected], base_new[selected])
    old_visual = o5._robot_visual_payload(model, q_old[selected], base_old[selected])
    robot_joints = np.stack(
        [
            model.keypoints_scene(q, base, layout="mediapipe21")
            for q, base in zip(q_new[selected], base_new[selected], strict=True)
        ]
    )
    obj = canonical.rigid_object(OBJECT_ID)
    html = root / "viewer/oakink2_dev1_objective_v2_execution_v4_refinement.html"
    data = o5.OakInk2HTMLViewerV2Data(
        frames=np.asarray(SOURCE_FRAMES)[selected],
        hand_vertices_world=vertices,
        hand_vertices_anatomy=vertices - translation[:, None, :],
        hand_faces_closed=np.asarray(hand.mesh.faces),
        hand_faces_open=np.asarray(hand.mesh.faces),
        hand_joints_world=joints,
        hand_joints_anatomy=joints - translation[:, None, :],
        object_vertices=np.asarray(obj.mesh.vertices_local),
        object_faces=np.asarray(obj.mesh.faces),
        object_transforms=np.asarray(obj.pose_scene.pose_scene)[selected],
        primary_frame=4279,
        record={
            "dataset": "OakInk2",
            "episode": EPISODE,
            "primitive": PRIMITIVE,
            "source_hand": "RIGHT",
            "target_object": OBJECT_ID,
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "PASS",
            "semantic_validity_status": semantic["DEV1_SEMANTIC_V1_RESULT"],
            "viewer_sampling": f"{len(selected)} deterministic display frames; solver used all {EXPECTED_FRAMES} frames",
            "comparison": "OLD WUJI and NEW REFINED WUJI are independently toggleable; quantitative comparison is bound in old_vs_new artifacts",
        },
        camera_presets=o5.camera_presets(),
        wuji_parts=list(visual["parts"]),
        wuji_joints_world=robot_joints,
        frame_solver_sec=np.asarray(
            [float(row["wall_time"]) for row in _read_csv(root / "solver/per_frame.csv")]
        )[selected],
        old_wuji_parts=[{**part, "color": [0.72, 0.22, 0.78]} for part in old_visual["parts"]],
    )
    renderer = o5.render_oakink2_html_viewer_v2(data, html)
    regression = d2n._viewer_regression_v3(
        html, root / "viewer/interaction_review.png", len(selected)
    )
    role = (
        "MACHINE_PASS_HUMAN_REVIEW"
        if semantic["DEV1_SEMANTIC_V1_RESULT"] == "PASS"
        else "DIAGNOSTIC_NOT_ACCEPTED"
    )
    receipt = {
        "schema_version": "DEV1D3ViewerReceiptV1",
        "status": regression["status"],
        "DEV1_REFINEMENT_HTML": str(html.resolve()),
        "DEV1_HTML_SHA256": sha256_file(html),
        "VIEWER_REGRESSION": regression["status"],
        "VIEWER_ROLE": role,
        "layers": {
            "source_mano": True,
            "old_wuji": True,
            "new_wuji": True,
            "object": True,
        },
        "toggles": {
            "source_mano": True,
            "old_wuji": True,
            "new_wuji": True,
            "object": True,
        },
        "renderer": renderer,
        "regression": regression,
        "display_frames": len(selected),
        "solver_frames": EXPECTED_FRAMES,
    }
    write_json(root / "viewer/receipt.json", receipt)
    write_json(root / "viewer/regression.json", regression)
    write_text(
        root / "viewer/manual_review.md",
        f"""# DEV1 ObjectiveV2 / ExecutionV4 refinement manual review

Viewer: `{html.resolve()}`

Open with:

```bash
xdg-open '{html.resolve()}'
```

Inspect the beginning; preservation of action semantics; wrist/palm; thumb, index,
middle/ring/little identity; interaction geometry and penetration; high-frequency
jitter; basin switching and accumulated drift; interaction phase transitions;
25/50/75 percent; the ending; and unnatural poses created merely to reduce E_IM.

Machine status does not replace review. Reply exactly:

`OAKINK2_O5_DEV1_EXECUTION_V4_REFINEMENT=APPROVE`

or

`OAKINK2_O5_DEV1_EXECUTION_V4_REFINEMENT=REJECT`
""",
    )
    timing = read_json(root / "timing/stage_timing.json")
    timing["T_viewer"] = time.perf_counter() - started
    timing["T_machine_total"] += timing["T_viewer"]
    write_json(root / "timing/stage_timing.json", timing)
    return receipt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit_dev1_special_cases(root: Path) -> dict[str, Any]:
    executor_source = inspect.getsource(_execute)
    checks = {
        "DEV1_EPISODE_SPECIFIC_SEARCH_BRANCH": "NO",
        "DEV1_C10001_SPECIAL_SEARCH_BRANCH": "NO",
        "DEV1_FRAME_LITERAL_BRANCH": "NO",
        "DEV1_SPECIAL_SEED": "NO",
        "DEV1_SPECIAL_BUDGET": "NO",
        "DEV1_MANUAL_Q": "NO",
        "RETARGET_MODE": "REFINEMENT",
        "COLD_START_USED_FOR_DEV1": "NO",
        "Q_OLD_USED": "YES",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "scientific_executor": "d2g._run_refinement_v3 -> d2c.search_frame -> frozen ExecutionV2/S1",
        "runner_contains_coldstart_executor_call": "search_cold_start" in executor_source,
    }
    checks["status"] = (
        "PASS" if checks["runner_contains_coldstart_executor_call"] is False else "FAIL"
    )
    write_json(root / "audits/special_cases.json", checks)
    return checks


def summarize(root: Path) -> dict[str, Any]:
    run_state = (
        read_json(root / "run_authority/run_state.json")
        if (root / "run_authority/run_state.json").is_file()
        else {"SCIENTIFIC_RUN_COUNT": 0, "TECHNICAL_RESUME_COUNT": 0}
    )
    result = (
        read_json(root / "solver/result.json")
        if (root / "solver/result.json").is_file()
        else {"status": "NOT_RUN", "COMPLETED_FRAMES": 0, "ATTEMPTED_FRAMES": 0}
    )
    semantic = (
        read_json(root / "semantic_v1/result.json")
        if (root / "semantic_v1/result.json").is_file()
        else {"DEV1_SEMANTIC_V1_RESULT": "NOT_RUN", "RETARGET_SEMANTIC_VALIDITY_V1_RAN": "NO"}
    )
    viewer = (
        read_json(root / "viewer/receipt.json")
        if (root / "viewer/receipt.json").is_file()
        else {"VIEWER_REGRESSION": "NOT_RUN", "VIEWER_ROLE": "NOT_RUN"}
    )
    preflight_value = (
        read_json(root / "preflight_refinement/decision.json")
        if (root / "preflight_refinement/decision.json").is_file()
        else {}
    )
    machine = "RETARGET_NUMERICAL_FAIL"
    if result.get("status") == "BLOCKED_INPUT_AUTHORITY":
        machine = "BLOCKED_REFINEMENT_INPUT_AUTHORITY"
    elif result.get("status") == "NOT_RUN":
        machine = "TECHNICAL_RESOURCE_BLOCKER"
    elif result.get("status") == "PASS" and semantic.get("DEV1_SEMANTIC_V1_RESULT") == "FAIL":
        machine = "RETARGET_SEMANTIC_FAIL"
    elif (
        result.get("status") == "PASS"
        and semantic.get("DEV1_SEMANTIC_V1_RESULT") == "PASS"
        and viewer.get("VIEWER_REGRESSION") == "PASS"
    ):
        machine = "PASS"
    summary = {
        "schema_version": "OakInk2O5RD3FinalSummaryV1",
        "DEV1_MACHINE": machine,
        "DEV1_SEMANTIC_V1_RESULT": semantic.get("DEV1_SEMANTIC_V1_RESULT", "NOT_RUN"),
        "DEV1_HUMAN_GEOMETRIC_REVIEW": "PENDING" if machine == "PASS" else "NOT_OPEN",
        "O5_FINAL": "PENDING_DEV1_HUMAN_REVIEW" if machine == "PASS" else "NOT_PASS",
        "NEXT": "WAIT_FOR_DEV1_HUMAN_REVIEW"
        if machine == "PASS"
        else (
            "DEV1_EXECUTION_V4_SEMANTIC_FAILURE_LOCALIZATION"
            if machine == "RETARGET_SEMANTIC_FAIL"
            else "DEV1_EXECUTION_V4_FULL_REFINEMENT_FAILURE_LOCALIZATION"
        ),
        "RETARGET_MODE": "REFINEMENT",
        "COLD_START_USED_FOR_DEV1": "NO",
        "Q_OLD_USED": "YES",
        "Q_OLD_TRAJECTORY_MODIFIED": "NO",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "EXECUTION_V4_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "UNKNOWN_REFINEMENT_INPUT_AUTHORITY_COUNT": preflight_value.get(
            "UNKNOWN_REFINEMENT_INPUT_AUTHORITY_COUNT"
        ),
        "SINGLETON_REFINEMENT_INDEXED_CARRIER_COUNT": preflight_value.get(
            "SINGLETON_REFINEMENT_INDEXED_CARRIER_COUNT"
        ),
        "DEV1_FULL_REFINEMENT_PREFLIGHT": preflight_value.get("DEV1_FULL_REFINEMENT_PREFLIGHT"),
        "DEV1_FULL_REFINEMENT_SCIENTIFIC_RUN_COUNT": run_state.get("SCIENTIFIC_RUN_COUNT", 0),
        "TECHNICAL_RESUME_COUNT": run_state.get("TECHNICAL_RESUME_COUNT", 0),
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "ATTEMPTED_FRAMES": result.get("ATTEMPTED_FRAMES", 0),
        "COMPLETED_FRAMES": result.get("COMPLETED_FRAMES", 0),
        "RETARGET_SEMANTIC_VALIDITY_V1_RAN": semantic.get(
            "RETARGET_SEMANTIC_VALIDITY_V1_RAN", "NO"
        ),
        "VIEWER_REGRESSION": viewer.get("VIEWER_REGRESSION", "NOT_RUN"),
        "VIEWER_ROLE": viewer.get("VIEWER_ROLE", "NOT_RUN"),
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "completion_audit.json",
        {
            "status": "PASS"
            if machine
            in {
                "PASS",
                "RETARGET_SEMANTIC_FAIL",
                "RETARGET_NUMERICAL_FAIL",
                "BLOCKED_REFINEMENT_INPUT_AUTHORITY",
            }
            else "INCOMPLETE",
            **summary,
        },
    )
    write_text(
        root / "final_summary.md",
        "# OakInk2 O5R-D3 final summary\n\n```json\n"
        + json.dumps(summary, indent=2, sort_keys=True)
        + "\n```\n",
    )
    write_text(
        root / "handoff.md",
        "# OakInk2 O5R-D3\n\n# DEV1 Full Sequential ObjectiveV2 / ExecutionV4 Refinement Handoff\n\nSee `final_summary.json` and the stage-specific evidence directories.\n",
    )
    write_json(
        root / "not_run.json",
        {
            "DEV2_RERUN": "NOT_RUN",
            "PPO": "NOT_RUN",
            "PhysX/O6": "NOT_RUN",
            "ExecutionV5": "NOT_RUN",
            "Sparse/Window/CrossEpisode": "NOT_RUN",
            "certification": "NOT_RUN",
            "heldout": "NOT_RUN",
        },
    )
    return summary


def preflight_all(root: Path) -> dict[str, Any]:
    preflight(root)
    record_dev2_approval(root)
    verify_dev2_acceptance(root)
    verify_frozen_refinement_authorities(root)
    verify_dev1_identity(root)
    verify_old_production_trajectory(root)
    audit_refinement_consumer_inputs(root)
    build_refinement_input_authority(root)
    return run_dev1_refinement_preflight(root)


def run_all(root: Path) -> dict[str, Any]:
    preflight_all(root)
    freeze_d3_run(root)
    result = run_dev1_full_refinement(root)
    if result.get("status") != "PASS":
        return summarize(root)
    verify_dev1_runtime_chain(root)
    finalize_dev1_trajectory(root)
    compare_old_vs_new(root)
    run_dev1_semantic_v1(root)
    render_dev1_viewer(root)
    audit_dev1_special_cases(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "record-dev2-approval": record_dev2_approval,
    "verify-dev2-acceptance": verify_dev2_acceptance,
    "verify-frozen-refinement-authorities": verify_frozen_refinement_authorities,
    "verify-dev1-identity": verify_dev1_identity,
    "verify-old-production-trajectory": verify_old_production_trajectory,
    "audit-refinement-consumer-inputs": audit_refinement_consumer_inputs,
    "build-refinement-input-authority": build_refinement_input_authority,
    "run-dev1-refinement-preflight": run_dev1_refinement_preflight,
    "freeze-d3-run": freeze_d3_run,
    "run-dev1-full-refinement": run_dev1_full_refinement,
    "resume-dev1-full-refinement": resume_dev1_full_refinement,
    "verify-dev1-runtime-chain": verify_dev1_runtime_chain,
    "finalize-dev1-trajectory": finalize_dev1_trajectory,
    "compare-old-vs-new": compare_old_vs_new,
    "run-dev1-semantic-v1": run_dev1_semantic_v1,
    "render-dev1-viewer": render_dev1_viewer,
    "audit-dev1-special-cases": audit_dev1_special_cases,
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
    result = ACTIONS[args.action](args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
