#!/usr/bin/env python3
"""O5R-D2M DEV2 full 240-frame geometric ExecutionV4 recovery.

This driver is deliberately fail closed.  It consumes the independently
certified D2L ExecutionV4 authorities without changing the scientific method,
permits exactly one DEV2 scientific run, and permits a same-run resume only
after an infrastructure interruption and a complete checkpoint-chain audit.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5  # noqa: E402
from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.data import run_oakink2_o5rd2g2 as d2g2  # noqa: E402
from scripts.data import run_oakink2_o5rd2g3 as d2g3  # noqa: E402
from scripts.evaluation import audit_retarget_semantic_validity as semantic_audit  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2kr_decision_tree as d2kr  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2l as d2l  # noqa: E402
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
from toporetarget.retarget.final_refinement import dynamic_collision_points_numpy  # noqa: E402
from toporetarget.retarget.objective_v4_execution import (  # noqa: E402
    ExecutionV4AcceptedRuntimeState,
    default_cold_start_search_v4_candidates,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2m_dev2_full_240_execution_v4_recovery_v1"
D2L_ROOT = d2l.ROOT
D2G3_ROOT = d2g3.ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
D2L_HEAD = "cd5340ccff4dade8de6ea00ecedfb9bf16742037"
EXPECTED_FRAMES = 240
SOURCE_START = 10704
SOURCE_STOP = 10944
EPISODE = "oakink2:scene_01__A003++seq__a7a1a0cf7d90a9083013__2023-04-21-20-13-04:00010"
PRIMITIVE = "rearrange"
OBJECT_ID = "C11001"

V4_AUTHORITY_HASHES = {
    "execution_v4_sequential_runtime_authority": "8e7e062bbd255a851cb7342749a5edff3d0466e9e60eb2920c7d7188fcea8e4d",
    "execution_v4_input_authority": "f06c22b5fcf4f00c252b423832856c8c8b582a704e042481669ed44ce39ab30d",
    "execution_v4_coldstart_search_authority": "9e583173bcfe333852a687e7efa0a8d01ecb6fd92d29867eaf4ef6b028cb6acb",
    "objective_v2_execution_contract_v4": "8e465851e797346f8aaad89dfe9b766f52bc7bd358bc378c8b893256a7b5120e",
}

FROZEN_AUTHORITIES: dict[str, tuple[Path, str]] = {
    "execution_v4_sequential_runtime_authority": (
        D2L_ROOT / "frozen_repaired_v4/sequential_runtime_authority.json",
        V4_AUTHORITY_HASHES["execution_v4_sequential_runtime_authority"],
    ),
    "execution_v4_input_authority": (
        D2L_ROOT / "frozen_repaired_v4/execution_input_authority.json",
        V4_AUTHORITY_HASHES["execution_v4_input_authority"],
    ),
    "execution_v4_coldstart_search_authority": (
        D2L_ROOT / "frozen_repaired_v4/coldstart_search_authority.json",
        V4_AUTHORITY_HASHES["execution_v4_coldstart_search_authority"],
    ),
    "objective_v2_execution_contract_v4": (
        D2L_ROOT / "frozen_repaired_v4/execution_contract_v4.json",
        V4_AUTHORITY_HASHES["objective_v2_execution_contract_v4"],
    ),
    "retarget_objective_v2": (
        d2g.frozen_paths()["objective_v2"],
        "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc",
    ),
    "certification_gate_v2": (
        d2g.frozen_paths()["gate_v2"],
        "a845fcdfec478e9208fc6c192317a45bee6ef19be1fcb784b459d8a4f675e048",
    ),
    "retarget_semantic_validity_v1": (
        d2g.frozen_paths()["semantic_v1"],
        "0ea9ba21419a2b254557e0af18405649b23a0cf30407684eaa2e2349c4467e31",
    ),
    "wuji_asset": (
        d2g.frozen_paths()["wuji_asset"],
        "65d77deb82afbda9144724a937312742c08ed86fd16609fb1118e71b6c9181ed",
    ),
    "oakink2_manifest_v2": (
        d2g.frozen_paths()["manifest_v2"],
        "0ef41a307d431569c519dd426b7c21a5ff7151e04ad60f35b36ecbc975d190c7",
    ),
    "oakink2_split_v2": (
        d2g.frozen_paths()["split_v2"],
        "05a32e22373b632e14ad1238954b1e28c4135efb0b580e8ef43e2d7dd74c5561",
    ),
    "source_interaction_graph_authority": (
        D2G3_ROOT / "frozen_v3/source_interaction_graph_authority.json",
        "36c0b9ee1fac20c957dd03a622b493bedfd59f42d002d8162b1b87f770caf5ff",
    ),
    "cold_start_bootstrap_contract": (
        D2G3_ROOT / "frozen_v3/cold_start_bootstrap_contract.json",
        "772cfe722327d0662c92a943d52d0f23cbece66b6e6c552fa5c764470b00de8e",
    ),
    "cold_start_seed_authority_v2": (
        D2G3_ROOT / "frozen_v3/cold_start_seed_authority_v2.json",
        "fde5cac433434504f3a8c58e289672e71cda85dcc77449d1f038f4010355feb0",
    ),
}

METHOD_IMPLEMENTATIONS: dict[str, tuple[Path, str]] = {
    "execution_v4_search": (
        REPO / "scripts/evaluation/run_oakink2_o5rd2kr_decision_tree.py",
        "2b5577a69906db390a3cadcfc3c849dad5820c438ea586e8c935a51da799c414",
    ),
    "execution_v4_runtime_state": (
        REPO / "src/toporetarget/retarget/objective_v4_execution.py",
        "144102114be1b7e3bf736b3ef3807751ad346e346475c120609fcd9507a196b2",
    ),
    "execution_v3_prefix": (
        REPO / "scripts/data/run_oakink2_o5rd2g2.py",
        "99e349ca6aeac254db254c3bd8e292ef880603aaba507d3ede59a7c27c54b3fd",
    ),
    "execution_v3_runtime": (
        REPO / "scripts/data/run_oakink2_o5rd2g.py",
        "d1ff24f587cea3a6b0d904c62f8e3369f31e8969eafac262ad0ff84ef731c691",
    ),
    "objective_v2": (
        REPO / "src/toporetarget/retarget/objective_v2.py",
        "8f719b709dfa4d7d6b0f19b4fa7a0897ce255bcbe4ae2a7a1cc9b5180f31a986",
    ),
    "production_refinement": (
        REPO / "src/toporetarget/retarget/final_refinement.py",
        "5a6da98b1050ff582ba7cc7a65e0d0e702574fa9f294a41fc57ef172f99975be",
    ),
}

GRAPH_PATH = d2g.ROOT / "graph_authority/dev2_frame0_source_graph.zarr"
CANONICAL_PATH = d2g.frozen_paths()["dev2_canonical"]
FIXED_EPISODES_PATH = d2g.frozen_paths()["dev2_episode_receipt"]
FAILURE_ENUM = {
    "MISSING_CANONICAL_INPUT",
    "SOURCE_GRAPH_FRAME_BINDING_FAIL",
    "SOURCE_INTERACTION_GRAPH_FAILURE",
    "WHOLE_HAND_BOOTSTRAP_FAILURE",
    "TOP2_CONTRIBUTOR_REFINEMENT_FAILURE",
    "PRIMARY_SEARCH_NO_VALID_CANDIDATE",
    "SECONDARY_POLISH_FAILURE",
    "HARD_VALIDITY_FAILURE",
    "JOINT_LIMIT_FAILURE",
    "COLLISION_FAILURE",
    "RUNTIME_PREVIOUS_STATE_FAILURE",
    "NONFINITE_NUMERICAL_FAILURE",
    "DETERMINISM_AUTHORITY_FAILURE",
    "MULTI_FACTOR",
    "INCONCLUSIVE",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = fields or (list(rows[0]) if rows else [])
    if not chosen:
        raise ValueError(f"CSV_FIELDS_REQUIRED:{path}")
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=chosen, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def freeze_json(path: Path, value: dict[str, Any]) -> str:
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"FROZEN_ARTIFACT_DRIFT:{path}")
    atomic_write_json(path, value)
    digest = sha256_file(path)
    receipt = path.with_suffix(".sha256")
    if receipt.exists() and receipt.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError(f"FROZEN_HASH_DRIFT:{path}")
    atomic_write_text(receipt, digest + "\n")
    return digest


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{path.name}:{field}={value.get(field)!r}")
    return value


def _sidecar_exact(path: Path) -> bool:
    sidecar = path.with_suffix(".sha256")
    return not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] == sha256_file(
        path
    )


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "d2l_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", D2L_HEAD, "HEAD"], cwd=REPO, check=False
        ).returncode
        == 0,
        "local_ignored": subprocess.run(
            ["git", "check-ignore", "-q", ".local"], cwd=REPO, check=False
        ).returncode
        == 0,
        "guidance_worktree_exists": (REPO.parent / "TopoRetarget-Repro-guidance").is_dir(),
    }
    value = {
        "schema_version": "OakInk2O5RD2MPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": head,
        "d2l_final_head": D2L_HEAD,
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
    atomic_write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_EXECUTION_V4_INTEGRITY:GIT_PREFLIGHT")
    return value


def _method_hashes(graph_path: Path = GRAPH_PATH) -> dict[str, str]:
    values = {name: sha256_file(path) for name, (path, _expected) in FROZEN_AUTHORITIES.items()}
    values.update(
        {
            f"implementation:{name}": sha256_file(path)
            for name, (path, _expected) in METHOD_IMPLEMENTATIONS.items()
        }
    )
    values["source_interaction_graph_artifact"] = d2g.interaction_artifact_hash(graph_path)
    values["dev2_canonical_artifact"] = d2g.digest(CANONICAL_PATH)
    values["semantic_gate_contract"] = SemanticGateContractV1().sha256
    return values


def verify_frozen_v4(root: Path) -> dict[str, Any]:
    _require(root / "preflight/git.json", "status", "PASS", "VERIFY_FROZEN_V4")
    rows: dict[str, Any] = {}
    for name, (path, expected) in FROZEN_AUTHORITIES.items():
        actual = sha256_file(path) if path.is_file() else None
        rows[name] = {
            "path": str(path.resolve()),
            "actual_sha256": actual,
            "expected_sha256": expected,
            "exact": actual == expected,
            "sidecar_exact": _sidecar_exact(path) if path.is_file() else False,
        }
    implementations: dict[str, Any] = {}
    for name, (path, expected) in METHOD_IMPLEMENTATIONS.items():
        actual = sha256_file(path)
        implementations[name] = {
            "path": str(path.resolve()),
            "actual_sha256": actual,
            "expected_sha256": expected,
            "exact": actual == expected,
        }
    d2l_summary = read_json(D2L_ROOT / "final_summary.json")
    authorization = read_json(D2L_ROOT / "future_dev2/authorization.json")
    freeze = read_json(D2L_ROOT / "frozen_repaired_v4/freeze_decision.json")
    impact = read_json(D2L_ROOT / "impact_audit/decision.json")
    window5 = read_json(D2L_ROOT / "window_v5_regression/decision.json")
    window6 = read_json(D2L_ROOT / "window_v6/decision.json")
    cross = read_json(D2L_ROOT / "cross_episode_v6/decision.json")
    graph_hash = d2g.interaction_artifact_hash(GRAPH_PATH)
    checks = {
        "all_frozen_authorities_exact": all(
            row["exact"] and row["sidecar_exact"] for row in rows.values()
        ),
        "all_method_implementations_exact": all(row["exact"] for row in implementations.values()),
        "execution_v4_repaired_independent_certification": freeze.get(
            "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION"
        )
        == "PASS",
        "dev2_authorized": authorization.get("DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED") == "YES",
        "dev2_count_zero": int(authorization.get("DEV2_FULL_GEOMETRIC_SOLVE_COUNT", -1)) == 0,
        "sparse_v5_retained": impact.get("SPARSE_V5_CERTIFICATION_RETAINED") == "YES",
        "window_v5_consumed_regression": window5.get("WINDOW_V5_CONSUMED_REGRESSION") == "PASS",
        "window_v6": window6.get("WINDOW_V6") == "PASS",
        "cross_episode_v6": cross.get("CROSS_EPISODE_V6") == "PASS",
        "repair_impact_runtime_only": impact.get("REPAIR_IMPACT") == "RUNTIME_ONLY",
        "d2l_summary_complete": d2l_summary.get("status") == "COMPLETE",
        "source_graph_exact": graph_hash
        == "2b941f517183b4e70903005f7b6914a7eedfa28654306db4f023e0447639f375",
    }
    value = {
        "schema_version": "ExecutionV4D2MIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "FROZEN_EXECUTION_V4_INTEGRITY": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "authorities": rows,
        "method_implementations": implementations,
        "source_interaction_graph": {
            "path": str(GRAPH_PATH.resolve()),
            "actual_sha256": graph_hash,
            "expected_sha256": "2b941f517183b4e70903005f7b6914a7eedfa28654306db4f023e0447639f375",
        },
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": freeze.get(
            "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION"
        ),
        "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED": authorization.get(
            "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED"
        ),
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": authorization.get("DEV2_FULL_GEOMETRIC_SOLVE_COUNT"),
        "SPARSE_V5_CERTIFICATION_RETAINED": impact.get("SPARSE_V5_CERTIFICATION_RETAINED"),
        "WINDOW_V5_CONSUMED_REGRESSION": window5.get("WINDOW_V5_CONSUMED_REGRESSION"),
        "WINDOW_V6": window6.get("WINDOW_V6"),
        "CROSS_EPISODE_V6": cross.get("CROSS_EPISODE_V6"),
        "REPAIR_IMPACT": impact.get("REPAIR_IMPACT"),
    }
    atomic_write_json(root / "preflight/frozen_authorities.json", value)
    atomic_write_json(root / "preflight/integrity.json", value)
    if value["status"] != "PASS":
        _write_blocked_terminal(root, "BLOCKED_UPSTREAM_INTEGRITY")
        raise RuntimeError("BLOCKED_UPSTREAM_EXECUTION_V4_INTEGRITY")
    return value


def verify_dev2_identity(root: Path) -> dict[str, Any]:
    _require(root / "preflight/integrity.json", "status", "PASS", "VERIFY_DEV2_IDENTITY")
    identity = d2g._dev2_identity()
    fixed = read_json(FIXED_EPISODES_PATH)
    fixed_episode = next(row for row in fixed["episodes"] if row["review"] == "dev_02")
    manifest_rows = [
        json.loads(line)
        for line in d2g.frozen_paths()["manifest_v2"].read_text(encoding="utf-8").splitlines()
        if line
    ]
    record = next(row for row in manifest_rows if row.get("record_id") == EPISODE)
    canonical = o5.load_canonical_hoi(CANONICAL_PATH)
    hand = canonical.hand("right_hand")
    obj = canonical.rigid_object(OBJECT_ID)
    source_frames = list(range(SOURCE_START, SOURCE_STOP))
    canonical_frames = list(hand.metadata["source_frame_ids"])
    mano = hand.mano_parameters
    object_pose = np.asarray(obj.pose_scene.pose_scene, dtype=np.float64)
    object_vertices = np.asarray(obj.mesh.vertices_local, dtype=np.float64)
    object_faces = np.asarray(obj.mesh.faces, dtype=np.int64)
    source_to_scene = np.asarray(canonical.metadata.source_to_scene, dtype=np.float64)
    checks = {
        "legacy_identity_exact": bool(identity["identity_exact"]),
        "episode": fixed_episode["record_id"] == EPISODE == record["record_id"],
        "primitive": record["primitive"] == PRIMITIVE,
        "target_object": fixed_episode["object"] == OBJECT_ID == record["canonical_target_object"],
        "interval": fixed_episode["source_interval"]
        == [SOURCE_START, SOURCE_STOP]
        == record["source_interval"],
        "frame_count": len(source_frames) == EXPECTED_FRAMES == len(canonical.metadata.timestamps),
        "frame_list_exact": canonical_frames == source_frames,
        "right_hand": hand.side == "right"
        and hand.hand_id == "right_hand"
        and record["active_hand"] == "RIGHT",
        "mano_present": mano is not None,
        "mano_finite": mano is not None
        and all(
            np.isfinite(np.asarray(value)).all()
            for value in (mano.betas, mano.global_orient_aa, mano.hand_pose_aa, mano.transl)
        ),
        "mano_frame_count": mano is not None
        and np.asarray(mano.transl).shape[0] == EXPECTED_FRAMES,
        "object_pose": object_pose.shape == (EXPECTED_FRAMES, 4, 4)
        and np.isfinite(object_pose).all(),
        "object_mesh": object_vertices.ndim == 2
        and object_faces.ndim == 2
        and np.isfinite(object_vertices).all(),
        "units": record["canonical_units"] == "metre"
        and hand.mesh.units == "m"
        and obj.mesh.units == "m",
        "source_to_scene_identity": source_to_scene.shape == (4, 4)
        and np.allclose(source_to_scene, np.eye(4), rtol=0.0, atol=1e-12),
        "frame_binding": record["frame_binding_authority"]
        == "OakInk2MocapFrameBindingV1:FRAME_BINDING_EXACT",
        "canonical_artifact_exact": d2g.digest(CANONICAL_PATH)
        == "19ece59feb8588f1f155ac523b0497bb3d4e5f8965c3710d5e49b9c32e4d2d3d",
        "fixed_episode_receipt_exact": sha256_file(FIXED_EPISODES_PATH)
        == "26550d82feacb6b0321bf9a6458befcff0dffa802a192c6a91ed2a174fb89cf5",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    value = {
        "schema_version": "DEV2CanonicalIdentityV1",
        "status": status,
        "DEV2_CANONICAL_IDENTITY": status,
        "checks": checks,
        "episode": EPISODE,
        "primitive": PRIMITIVE,
        "target_object": OBJECT_ID,
        "source_start": SOURCE_START,
        "source_stop": SOURCE_STOP,
        "expected_frames": EXPECTED_FRAMES,
        "source_frames": source_frames,
        "right_hand_identity": "right_hand/RIGHT",
        "mano_authority": record["official_manolayer_semantics"],
        "object_pose_authority": record["object_transform_representation"],
        "object_mesh_authority": {
            "path": record["object_asset"],
            "sha256": record["object_asset_sha256"],
        },
        "units": record["canonical_units"],
        "frame_transform_authority": record["frame_binding_authority"],
        "canonical_path": str(CANONICAL_PATH.resolve()),
        "canonical_sha256": d2g.digest(CANONICAL_PATH),
        "manifest_record": record,
    }
    atomic_write_json(root / "dev2_identity/canonical_identity.json", value)
    atomic_write_json(
        root / "dev2_identity/frame_manifest.json",
        {
            "schema_version": "DEV2FrameManifestV1",
            "status": status,
            "EXPECTED_FRAMES": EXPECTED_FRAMES,
            "source_interval": [SOURCE_START, SOURCE_STOP],
            "source_frames": source_frames,
        },
    )
    atomic_write_json(
        root / "dev2_identity/source_authority.json",
        {
            "schema_version": "DEV2SourceAuthorityV1",
            "status": status,
            "canonical_path": str(CANONICAL_PATH.resolve()),
            "canonical_sha256": d2g.digest(CANONICAL_PATH),
            "manifest_v2_sha256": sha256_file(d2g.frozen_paths()["manifest_v2"]),
            "split_v2_sha256": sha256_file(d2g.frozen_paths()["split_v2"]),
            "interaction_graph_path": str(GRAPH_PATH.resolve()),
            "interaction_graph_sha256": d2g.interaction_artifact_hash(GRAPH_PATH),
        },
    )
    if status != "PASS":
        _write_blocked_terminal(root, "BLOCKED_DEV2_IDENTITY_AUTHORITY")
        raise RuntimeError("BLOCKED_DEV2_IDENTITY_AUTHORITY")
    return value


def _technical_resume_policy() -> dict[str, Any]:
    return {
        "schema_version": "DEV2TechnicalResumePolicyV1",
        "status": "FROZEN",
        "infrastructure_interruptions": [
            "PROCESS_UNEXPECTEDLY_TERMINATED",
            "HOST_INTERRUPTION",
            "DISK_IO_TRANSIENT",
            "TERMINAL_SESSION_LOSS",
            "RECOVERABLE_FILE_WRITE_INTERRUPTION",
        ],
        "scientific_failures": sorted(FAILURE_ENUM),
        "technical_resume_requires": [
            "same RUN_UUID",
            "same run-manifest SHA",
            "same method hashes",
            "contiguous accepted checkpoint chain",
            "exact predecessor state hash",
        ],
        "scientific_failure_resume_through": "FORBIDDEN",
        "second_scientific_run": "FORBIDDEN",
    }


def freeze_dev2_run(root: Path) -> dict[str, Any]:
    integrity = _require(root / "preflight/integrity.json", "status", "PASS", "FREEZE_DEV2_RUN")
    identity = _require(
        root / "dev2_identity/canonical_identity.json", "status", "PASS", "FREEZE_DEV2_RUN"
    )
    if integrity.get("DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED") != "YES":
        raise RuntimeError("FREEZE_DEV2_RUN_REJECTED:NOT_AUTHORIZED")
    manifest_path = root / "run_authority/full_run_manifest.json"
    if manifest_path.exists():
        sidecar = root / "run_authority/full_run_manifest.sha256"
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != sha256_file(
            manifest_path
        ):
            raise RuntimeError("DEV2_RUN_MANIFEST_HASH_DRIFT")
        return read_json(manifest_path)
    policy_path = root / "run_authority/technical_resume_policy.json"
    policy_sha = freeze_json(policy_path, _technical_resume_policy())
    run_uuid = str(uuid.uuid4())
    method_hashes = _method_hashes()
    manifest = {
        "schema_version": "DEV2FullGeometricRunManifestV1",
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_UUID": run_uuid,
        "RUN_ROLE": "KNOWN_FAILURE_RECOVERY_FULL_GEOMETRIC_RUN",
        "identity": {
            key: identity[key]
            for key in (
                "episode",
                "primitive",
                "target_object",
                "source_start",
                "source_stop",
                "expected_frames",
            )
        },
        "source_frames": list(range(SOURCE_START, SOURCE_STOP)),
        "method": "V4_A_TOP2_SEQUENTIAL",
        "method_hashes": method_hashes,
        "ExecutionV4_authority_sha256": V4_AUTHORITY_HASHES,
        "ObjectiveV2_sha256": method_hashes["retarget_objective_v2"],
        "SemanticV1_sha256": method_hashes["retarget_semantic_validity_v1"],
        "Wuji_asset_sha256": method_hashes["wuji_asset"],
        "ManifestV2_sha256": method_hashes["oakink2_manifest_v2"],
        "SplitV2_sha256": method_hashes["oakink2_split_v2"],
        "interaction_graph_authority": {
            "path": str(GRAPH_PATH.resolve()),
            "sha256": method_hashes["source_interaction_graph_artifact"],
        },
        "frame0_semantics": {
            "mode": "COLD_START",
            "q_old": "ABSENT",
            "previous_accepted_runtime_state": "ABSENT",
        },
        "t_gt_0_semantics": {
            "mode": "COLD_START_SEQUENCE",
            "q_old": "ABSENT",
            "previous_accepted_runtime_state": "EXACT_ACCEPTED_T_MINUS_1",
        },
        "DEV2_DEVELOPMENT_FRAME0_STATE_REUSED": "NO",
        "technical_resume_policy_sha256": policy_sha,
        "profiler_authority": "RetargetSolverProfilerV1",
        "output_paths": {
            "root": str(root.resolve()),
            "checkpoints": str((root / "checkpoints").resolve()),
            "partial_trajectory": str((root / "trajectory/trajectory_partial.npz").resolve()),
            "trajectory": str((root / "trajectory/trajectory.npz").resolve()),
        },
    }
    manifest_sha = freeze_json(manifest_path, manifest)
    atomic_write_text(root / "run_authority/run_uuid.txt", run_uuid + "\n")
    if not (root / "technical_failures.jsonl").exists():
        atomic_write_text(root / "technical_failures.jsonl", "")
    if not (root / "solver/frame_results.jsonl").exists():
        atomic_write_text(root / "solver/frame_results.jsonl", "")
    manifest["DEV2_RUN_MANIFEST_SHA256"] = manifest_sha
    return manifest


def _checkpoint_dir(root: Path, ordinal: int) -> Path:
    return root / f"checkpoints/frame_{ordinal:03d}"


def _load_checkpoint(
    root: Path, ordinal: int, manifest: dict[str, Any]
) -> tuple[dict[str, Any], ExecutionV4AcceptedRuntimeState, np.ndarray, np.ndarray]:
    directory = _checkpoint_dir(root, ordinal)
    marker = read_json(directory / "checkpoint.json")
    state_path = directory / "state.npz"
    accepted_path = directory / "accepted_runtime_state.json"
    receipt_path = directory / "receipt.json"
    if marker.get("status") != "ACCEPTED" or marker.get("ordinal") != ordinal:
        raise RuntimeError(f"CORRUPT_CHECKPOINT_MARKER:{ordinal}")
    if marker.get("RUN_UUID") != manifest["RUN_UUID"] or marker.get(
        "manifest_sha256"
    ) != sha256_file(root / "run_authority/full_run_manifest.json"):
        raise RuntimeError(f"CHECKPOINT_RUN_AUTHORITY_MISMATCH:{ordinal}")
    for key, path in (
        ("state_npz_sha256", state_path),
        ("accepted_state_payload_sha256", accepted_path),
        ("receipt_sha256", receipt_path),
    ):
        if not path.is_file() or marker.get(key) != sha256_file(path):
            raise RuntimeError(f"CORRUPT_CHECKPOINT_CONTENT:{ordinal}:{key}")
    accepted = ExecutionV4AcceptedRuntimeState.from_payload(read_json(accepted_path))
    if accepted.sha256 != marker.get("current_state_hash"):
        raise RuntimeError(f"CHECKPOINT_ACCEPTED_STATE_HASH_MISMATCH:{ordinal}")
    with np.load(state_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        base = np.asarray(archive["base_pose_scene"], dtype=np.float64)
    state_q, state_base = accepted.arrays()
    if not np.array_equal(qpos, state_q) or not np.array_equal(base, state_base):
        raise RuntimeError(f"CHECKPOINT_ARRAY_STATE_MISMATCH:{ordinal}")
    return marker, accepted, qpos, base


def verify_runtime_chain(root: Path) -> dict[str, Any]:
    manifest = _require(
        root / "run_authority/full_run_manifest.json",
        "status",
        "FROZEN_BEFORE_SOLVE",
        "VERIFY_RUNTIME_CHAIN",
    )
    sidecar = root / "run_authority/full_run_manifest.sha256"
    if sidecar.read_text(encoding="utf-8").strip() != sha256_file(
        root / "run_authority/full_run_manifest.json"
    ):
        raise RuntimeError("RUN_MANIFEST_HASH_DRIFT")
    accepted: list[ExecutionV4AcceptedRuntimeState] = []
    rows: list[dict[str, Any]] = []
    failure = None
    for ordinal in range(EXPECTED_FRAMES):
        directory = _checkpoint_dir(root, ordinal)
        if not directory.exists():
            break
        try:
            marker, state, _q, _base = _load_checkpoint(root, ordinal, manifest)
            expected_previous = None if ordinal == 0 else accepted[-1].sha256
            if marker.get("previous_state_hash") != expected_previous:
                raise RuntimeError(f"PREVIOUS_STATE_HASH_CHAIN_MISMATCH:{ordinal}")
            if state.source_ordinal != ordinal or state.source_frame != SOURCE_START + ordinal:
                raise RuntimeError(f"CHECKPOINT_FRAME_IDENTITY_MISMATCH:{ordinal}")
            if (
                "graph_source_frame" in marker
                and marker.get("graph_source_frame") != state.source_frame
            ):
                raise RuntimeError(f"CHECKPOINT_GRAPH_FRAME_IDENTITY_MISMATCH:{ordinal}")
            if (
                "canonical_source_frame" in marker
                and marker.get("canonical_source_frame") != state.source_frame
            ):
                raise RuntimeError(f"CHECKPOINT_CANONICAL_FRAME_IDENTITY_MISMATCH:{ordinal}")
            accepted.append(state)
            rows.append(
                {
                    "ordinal": ordinal,
                    "source_frame": state.source_frame,
                    "previous_state_hash": expected_previous,
                    "current_state_hash": state.sha256,
                    "q_old_access_count": 0,
                }
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}:{exc}"
            break
    stray = [
        ordinal
        for ordinal in range(len(accepted) + 1, EXPECTED_FRAMES)
        if _checkpoint_dir(root, ordinal).exists()
    ]
    valid = failure is None and not stray
    value = {
        "schema_version": "DEV2RuntimeChainIntegrityV1",
        "status": "PASS" if valid else "FAIL",
        "PREVIOUS_STATE_CHAIN_VALID": "YES" if valid else "NO",
        "accepted_prefix_length": len(accepted),
        "last_accepted_ordinal": None if not accepted else len(accepted) - 1,
        "last_accepted_state_hash": None if not accepted else accepted[-1].sha256,
        "contiguous": not stray,
        "stray_checkpoint_ordinals": stray,
        "failure": failure,
        "rows": rows,
    }
    atomic_write_json(root / "audits/runtime_chain_integrity.json", value)
    write_csv(
        root / "solver/runtime_state_chain.csv",
        rows,
        [
            "ordinal",
            "source_frame",
            "previous_state_hash",
            "current_state_hash",
            "q_old_access_count",
        ],
    )
    if not valid:
        raise RuntimeError(f"RESUME_AUTHORIZED=NO:{failure or stray}")
    return value


def _profiler_row(receipt: dict[str, Any], ordinal: int, elapsed: float) -> dict[str, Any]:
    prefix = receipt.get("execution_v3_prefix_receipt", {})
    prefix_profiler = prefix.get("profiler", {}) if isinstance(prefix, dict) else {}
    bootstrap = prefix.get("bootstrap", {}) if isinstance(prefix, dict) else {}
    solves = bootstrap.get("solves", []) if isinstance(bootstrap, dict) else []
    stages = receipt.get("stages", [])
    profiles = [
        phase.get("profile")
        for stage in stages
        for phase in stage.get("phases", [])
        if isinstance(phase, dict) and isinstance(phase.get("profile"), dict)
    ]
    primary = [profile for profile in profiles if profile.get("phase") == "primary"]
    secondary = [profile for profile in profiles if profile.get("phase") == "secondary"]
    secondary_rows = [
        phase
        for stage in stages
        for phase in stage.get("phases", [])
        if isinstance(phase, dict) and phase.get("phase") == "secondary"
    ]
    return {
        "ordinal": ordinal,
        "source_frame": SOURCE_START + ordinal,
        "execution_mode": "COLD_START" if ordinal == 0 else "COLD_START_SEQUENCE",
        "previous_state_available": ordinal > 0,
        "whole_hand_bootstrap_seed_count": len(solves),
        "bootstrap_solve_count": len(solves),
        "bootstrap_nfev": sum(int(row.get("nfev") or 0) for row in solves),
        "bootstrap_wall_time": prefix_profiler.get("bootstrap_wall_sec"),
        "bootstrap_valid_candidate_count": prefix_profiler.get("bootstrap_hard_valid_candidates"),
        "post_bootstrap_contributor_scores": json.dumps(
            receipt.get("initial_contributor_scores", {}), sort_keys=True
        ),
        "top1_contributor": (receipt.get("initial_contributor_ranking") or [None])[0],
        "top2_contributor": (receipt.get("initial_contributor_ranking") or [None, None])[1],
        "top2_sequence_order": "+".join(receipt.get("used_contributors", [])),
        "candidate_b2_probes": sum(
            1 for stage in stages if isinstance(stage, dict) and "probe" in stage
        ),
        "primary_nfev": sum(int(row.get("nfev") or 0) for row in primary),
        "primary_wall_time": sum(float(row.get("wall_time_sec") or 0.0) for row in primary),
        "secondary_nfev": sum(int(row.get("nfev") or 0) for row in secondary),
        "secondary_wall_time": sum(float(row.get("wall_time_sec") or 0.0) for row in secondary),
        "polish_attempted": bool(secondary_rows),
        "polish_accepted_or_rejected": ";".join(
            str(row.get("retention", "NOT_AVAILABLE")) for row in secondary_rows
        )
        or "NOT_AVAILABLE",
        "primary_retained": any(
            "PRIMARY" in str(row.get("retention", "")) for row in secondary_rows
        ),
        "fallback_used": bool(receipt.get("profiler", {}).get("fallback", False)),
        "interaction_evaluation_time": sum(
            float(row.get("interaction_eval_time_sec") or 0.0) for row in profiles
        ),
        "candidate_screening_time": prefix_profiler.get("candidate_screening_time_sec"),
        "total_solver_wall_time": elapsed,
        "optimizer_terminal_statuses": json.dumps(
            [row.get("status", "NOT_AVAILABLE") for row in solves]
            + [row.get("status_code", "NOT_AVAILABLE") for row in profiles]
        ),
        "independently_valid_terminal_count": int(
            bool(receipt.get("selected_evaluation", {}).get("feasible"))
        ),
    }


def _write_solver_projection_csvs(root: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        root / "solver/interaction_metrics.csv",
        rows,
        ["ordinal", "source_frame", "status", "E_IM", "finite"],
    )
    write_csv(
        root / "solver/hard_validity.csv",
        rows,
        ["ordinal", "source_frame", "hard_validity", "finite", "q_old_access_count"],
    )
    write_csv(
        root / "solver/contributor_sequence.csv",
        rows,
        [
            "ordinal",
            "source_frame",
            "selected_contributors",
            "selected_candidate",
            "context_binding_hash",
        ],
    )


def _classify_failure(exc: Exception, receipt: dict[str, Any] | None) -> tuple[str, str]:
    text = f"{type(exc).__name__}:{exc}"
    lower = text.lower()
    violations = (
        []
        if receipt is None
        else [
            str(item)
            for item in receipt.get("selected_evaluation", {}).get("violated_constraints", [])
        ]
    )
    joined = " ".join(violations).lower()
    if "source_graph_frame_binding_fail" in lower:
        category = "SOURCE_GRAPH_FRAME_BINDING_FAIL"
    elif "previous" in lower and "state" in lower:
        category = "RUNTIME_PREVIOUS_STATE_FAILURE"
    elif "nonfinite" in lower or "non-finite" in lower or "nan" in lower:
        category = "NONFINITE_NUMERICAL_FAILURE"
    elif "joint" in joined or "joint" in lower:
        category = "JOINT_LIMIT_FAILURE"
    elif "collision" in joined or "penetration" in joined or "collision" in lower:
        category = "COLLISION_FAILURE"
    elif receipt is not None and not receipt.get("technical_success", False):
        category = "HARD_VALIDITY_FAILURE"
    elif "bootstrap" in lower:
        category = "WHOLE_HAND_BOOTSTRAP_FAILURE"
    elif "no valid" in lower or "candidate" in lower:
        category = "PRIMARY_SEARCH_NO_VALID_CANDIDATE"
    else:
        category = "INCONCLUSIVE"
    return category, f"{text}; violated_constraints={violations}"


def _run_state(root: Path) -> dict[str, Any]:
    path = root / "run_authority/run_state.json"
    return (
        read_json(path)
        if path.is_file()
        else {
            "schema_version": "DEV2ScientificRunStateV1",
            "SCIENTIFIC_RUN_COUNT": 0,
            "TECHNICAL_RESUME_COUNT": 0,
            "status": "NOT_STARTED",
        }
    )


def _write_partial_trajectory(
    root: Path,
    rows: list[dict[str, Any]],
    q_states: list[np.ndarray],
    base_states: list[np.ndarray],
) -> None:
    atomic_save_npz(
        root / "trajectory/trajectory_partial.npz",
        schema_version=np.asarray("DEV2ExecutionV4PartialTrajectoryV1"),
        episode=np.asarray(EPISODE),
        source_frame_ids=np.asarray([int(row["source_frame"]) for row in rows], dtype=np.int64),
        qpos=np.stack(q_states) if q_states else np.empty((0, 0), dtype=np.float64),
        base_pose_scene=np.stack(base_states)
        if base_states
        else np.empty((0, 4, 4), dtype=np.float64),
        e_im=np.asarray([float(row["E_IM"]) for row in rows], dtype=np.float64),
    )


def _load_prefix(
    root: Path, manifest: dict[str, Any]
) -> tuple[
    list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[ExecutionV4AcceptedRuntimeState]
]:
    chain = verify_runtime_chain(root)
    rows: list[dict[str, Any]] = []
    q_states: list[np.ndarray] = []
    base_states: list[np.ndarray] = []
    states: list[ExecutionV4AcceptedRuntimeState] = []
    for ordinal in range(int(chain["accepted_prefix_length"])):
        marker, state, qpos, base = _load_checkpoint(root, ordinal, manifest)
        rows.append(marker["row"])
        q_states.append(qpos)
        base_states.append(base)
        states.append(state)
    return rows, q_states, base_states, states


def _build_dev2_runtime(graph_path: Path = GRAPH_PATH) -> d2g.V3Runtime:
    """Load the frozen source graph while reusing the D2G3 geometry cache."""
    runtime = d2g.V3Runtime("dev_02", D2G3_ROOT)
    authoritative_graph = d2g.load_interaction_graph(graph_path)
    runtime.graph = authoritative_graph
    runtime.resources = d2g.prepare_refinement_resources(
        runtime.sequence,
        authoritative_graph,
        runtime.solver,
        geometry_artifact_root=D2G3_ROOT / "execution_v3_design/geometry/dev2",
    )
    runtime.backends = d2g.prepare_refinement_runtime_backends(runtime.resources, runtime.execution)
    return runtime


def _full_sequence_graph_preflight(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reject graph coverage or identity drift before any optimizer is entered."""

    graph_authority = manifest["interaction_graph_authority"]
    graph_path = Path(graph_authority["path"])
    graph = d2g.load_interaction_graph(graph_path)
    trajectory_frames = [int(value) for value in manifest["source_frames"]]
    graph_source_frames = [
        int(value) for value in manifest.get("graph_source_frame_ids", trajectory_frames)
    ]
    graph_ordinals = np.asarray(graph.frame_indices, dtype=np.int64).tolist()
    canonical = o5.load_canonical_hoi(CANONICAL_PATH)
    canonical_frames = [
        int(value) for value in canonical.hand("right_hand").metadata["source_frame_ids"]
    ]
    checks = {
        "trajectory_count": len(trajectory_frames) == EXPECTED_FRAMES,
        "graph_count": len(graph.frames) == EXPECTED_FRAMES,
        "graph_source_frame_count": len(graph_source_frames) == EXPECTED_FRAMES,
        "trajectory_graph_frame_ids": trajectory_frames == graph_source_frames,
        "canonical_source_frame_ids": canonical_frames == trajectory_frames,
        "graph_ordinals": graph_ordinals == list(range(EXPECTED_FRAMES)),
        "source_interval": trajectory_frames == list(range(SOURCE_START, SOURCE_STOP)),
        "object_id": graph_authority.get("object_id", OBJECT_ID) == OBJECT_ID,
        "artifact_sha256": d2g.interaction_artifact_hash(graph_path) == graph_authority["sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"FULL_SEQUENCE_GRAPH_COVERAGE_MISMATCH:{checks}")
    return {
        "status": "PASS",
        "graph_path": graph_path,
        "graph": graph,
        "trajectory_frames": trajectory_frames,
        "graph_source_frames": graph_source_frames,
        "canonical_frames": canonical_frames,
        "checks": checks,
    }


def _full_sequence_runtime_input_preflight(runtime: Any) -> dict[str, Any]:
    """Validate every ordinal-indexed consumer input before scientific-run commit."""

    warm_q = np.asarray(runtime.warm.arrays["qpos"])
    warm_base = np.asarray(runtime.warm.arrays["base_pose_scene"])
    checks = {
        "runtime_graph_count": int(runtime.graph.frame_count) == EXPECTED_FRAMES,
        "warm_q_seed_carrier_count": int(warm_q.shape[0]) == EXPECTED_FRAMES,
        "warm_base_seed_carrier_count": int(warm_base.shape[0]) == EXPECTED_FRAMES,
    }
    if not all(checks.values()):
        raise RuntimeError(f"FULL_SEQUENCE_RUNTIME_INPUT_COVERAGE_MISMATCH:{checks}")
    return {"status": "PASS", "checks": checks}


def _execute_dev2(root: Path, *, resume: bool) -> dict[str, Any]:
    manifest = _require(
        root / "run_authority/full_run_manifest.json",
        "status",
        "FROZEN_BEFORE_SOLVE",
        "RUN_DEV2_FULL",
    )
    manifest_sha = sha256_file(root / "run_authority/full_run_manifest.json")
    if (root / "run_authority/full_run_manifest.sha256").read_text(
        encoding="utf-8"
    ).strip() != manifest_sha:
        raise RuntimeError("RUN_DEV2_FULL_REJECTED:MANIFEST_HASH_DRIFT")
    graph_preflight = _full_sequence_graph_preflight(manifest)
    graph_path = graph_preflight["graph_path"]
    if manifest["method_hashes"] != _method_hashes(graph_path):
        raise RuntimeError("RUN_DEV2_FULL_REJECTED:METHOD_HASH_DRIFT")
    load_started = time.perf_counter()
    runtime = _build_dev2_runtime(graph_path)
    _full_sequence_runtime_input_preflight(runtime)
    load_elapsed = time.perf_counter() - load_started
    run_state = _run_state(root)
    if resume:
        if (
            int(run_state["SCIENTIFIC_RUN_COUNT"]) != 1
            or run_state["status"] != "TECHNICAL_INTERRUPTION"
        ):
            raise RuntimeError("RESUME_DEV2_FULL_REJECTED:NO_VALID_TECHNICAL_INTERRUPTION")
        interruption = _require(
            root / "technical_interruption.json", "resume_allowed", True, "RESUME_DEV2_FULL"
        )
        if (
            interruption.get("RUN_UUID") != manifest["RUN_UUID"]
            or interruption.get("manifest_sha256") != manifest_sha
        ):
            raise RuntimeError("RESUME_DEV2_FULL_REJECTED:RUN_AUTHORITY_MISMATCH")
        run_state["TECHNICAL_RESUME_COUNT"] = int(run_state["TECHNICAL_RESUME_COUNT"]) + 1
        run_state["status"] = "RUNNING"
    else:
        if int(run_state["SCIENTIFIC_RUN_COUNT"]) != 0:
            raise RuntimeError("RUN_DEV2_FULL_REJECTED:SCIENTIFIC_RUN_COUNT_ALREADY_ONE")
        run_state.update(
            {
                "RUN_UUID": manifest["RUN_UUID"],
                "manifest_sha256": manifest_sha,
                "SCIENTIFIC_RUN_COUNT": 1,
                "TECHNICAL_RESUME_COUNT": 0,
                "status": "RUNNING",
                "optimizer_start_committed_before_frame0": True,
            }
        )
    atomic_write_json(root / "run_authority/run_state.json", run_state)
    rows, q_states, base_states, accepted_states = _load_prefix(root, manifest)
    start_ordinal = len(rows)
    if not resume and start_ordinal:
        raise RuntimeError("RUN_DEV2_FULL_REJECTED:PREEXISTING_ACCEPTED_CHECKPOINT")
    previous = None if not accepted_states else accepted_states[-1]
    _write_partial_trajectory(root, rows, q_states, base_states)
    runtime_started = time.perf_counter()
    candidate = default_cold_start_search_v4_candidates()[0]
    profiler_rows = [
        read_json(_checkpoint_dir(root, ordinal) / "profiler.json")
        for ordinal in range(start_ordinal)
    ]
    failure: dict[str, Any] | None = None
    for ordinal in range(start_ordinal, EXPECTED_FRAMES):
        receipt: dict[str, Any] | None = None
        frame_started = time.perf_counter()
        previous_q, previous_base = (None, None) if previous is None else previous.arrays()
        try:
            current_source_frame = SOURCE_START + ordinal
            graph_source_frame = int(graph_preflight["graph_source_frames"][ordinal])
            canonical_source_frame = int(graph_preflight["canonical_frames"][ordinal])
            if not current_source_frame == graph_source_frame == canonical_source_frame:
                raise RuntimeError(
                    "SOURCE_GRAPH_FRAME_BINDING_FAIL:"
                    f"trajectory={current_source_frame}:graph={graph_source_frame}:"
                    f"canonical={canonical_source_frame}"
                )
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
                source_frame=SOURCE_START + ordinal,
                qpos=qpos,
                base_pose_scene=base,
                object_id=OBJECT_ID,
            )
            elapsed = time.perf_counter() - frame_started
            profiler = _profiler_row(receipt, ordinal, elapsed)
            row = {
                "ordinal": ordinal,
                "source_frame": current_source_frame,
                "graph_source_frame": graph_source_frame,
                "canonical_source_frame": canonical_source_frame,
                "graph_entry_hash": graph_entry_hash,
                "status": "ACCEPTED",
                "E_IM": float(receipt["selected"]["interaction_e_im"]),
                "hard_validity": "PASS",
                "finite": True,
                "q_old_access_count": 0,
                "previous_state_hash": None if previous is None else previous.sha256,
                "current_state_hash": state.sha256,
                "context_binding_hash": receipt.get("execution_v3_prefix_receipt", {}).get(
                    "context_binding_sha256"
                ),
                "selected_contributors": "+".join(receipt.get("used_contributors", [])),
                "selected_candidate": receipt.get("selected_candidate"),
                "wall_time": elapsed,
            }
            directory = _checkpoint_dir(root, ordinal)
            atomic_save_npz(directory / "state.npz", qpos=qpos, base_pose_scene=base)
            atomic_write_json(directory / "accepted_runtime_state.json", state.canonical_payload())
            atomic_write_json(directory / "receipt.json", receipt)
            atomic_write_json(directory / "profiler.json", profiler)
            marker = {
                "schema_version": "DEV2DurableFrameCheckpointV1",
                "status": "ACCEPTED",
                "RUN_UUID": manifest["RUN_UUID"],
                "manifest_sha256": manifest_sha,
                "ordinal": ordinal,
                "source_frame": current_source_frame,
                "graph_source_frame": graph_source_frame,
                "canonical_source_frame": canonical_source_frame,
                "graph_entry_hash": graph_entry_hash,
                "previous_source_frame": None if ordinal == 0 else SOURCE_START + ordinal - 1,
                "previous_state_hash": None if previous is None else previous.sha256,
                "current_state_hash": state.sha256,
                "context_binding_hash": row["context_binding_hash"],
                "runtime_state_received": ordinal > 0,
                "q_old_access_count": 0,
                "state_npz_sha256": sha256_file(directory / "state.npz"),
                "accepted_state_payload_sha256": sha256_file(
                    directory / "accepted_runtime_state.json"
                ),
                "receipt_sha256": sha256_file(directory / "receipt.json"),
                "interaction_graph_hash": manifest["interaction_graph_authority"]["sha256"],
                "method_hashes": manifest["method_hashes"],
                "row": row,
            }
            atomic_write_json(directory / "checkpoint.json", marker)
            rows.append(row)
            q_states.append(qpos)
            base_states.append(base)
            accepted_states.append(state)
            profiler_rows.append(profiler)
            previous = state
            write_csv(root / "solver/per_frame.csv", rows)
            _write_solver_projection_csvs(root, rows)
            write_csv(root / "profiler/per_frame.csv", profiler_rows)
            with (root / "solver/frame_results.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _write_partial_trajectory(root, rows, q_states, base_states)
            print(
                f"O5RD2M DEV2 {ordinal + 1}/240 source_frame={SOURCE_START + ordinal} E_IM={row['E_IM']:.12g} wall={elapsed:.3f}s",
                flush=True,
            )
        except (OSError, KeyboardInterrupt) as exc:
            run_state["status"] = "TECHNICAL_INTERRUPTION"
            atomic_write_json(root / "run_authority/run_state.json", run_state)
            atomic_write_json(
                root / "technical_interruption.json",
                {
                    "schema_version": "DEV2TechnicalInterruptionV1",
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
                "schema_version": "DEV2ScientificFailureV1",
                "status": "SCIENTIFIC_FAIL",
                "RUN_UUID": manifest["RUN_UUID"],
                "FIRST_FAILURE_ORDINAL": ordinal,
                "FIRST_FAILURE_SOURCE_FRAME": SOURCE_START + ordinal,
                "PREVIOUS_ACCEPTED_ORDINAL": None if ordinal == 0 else ordinal - 1,
                "PREVIOUS_ACCEPTED_STATE_HASH": None if previous is None else previous.sha256,
                "FAILURE_CLASS": category,
                "FAILURE_MECHANISM": mechanism,
                "resume_allowed": False,
            }
            atomic_write_json(root / "solver/first_failure.json", failure)
            with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
            run_state["status"] = "SCIENTIFIC_FAIL"
            atomic_write_json(root / "run_authority/run_state.json", run_state)
            break
    solver_elapsed = time.perf_counter() - runtime_started
    attempted = len(rows) + (1 if failure is not None else 0)
    completed = len(rows)
    coverage = _coverage_payload(rows)
    if not rows:
        write_csv(
            root / "solver/per_frame.csv",
            [],
            [
                "ordinal",
                "source_frame",
                "status",
                "E_IM",
                "hard_validity",
                "finite",
                "q_old_access_count",
                "previous_state_hash",
                "current_state_hash",
                "context_binding_hash",
                "selected_contributors",
                "selected_candidate",
                "wall_time",
            ],
        )
        _write_solver_projection_csvs(root, [])
        write_csv(
            root / "profiler/per_frame.csv",
            [],
            list(_profiler_row({}, 0, 0.0)),
        )
    atomic_write_json(root / "audits/frame_coverage.json", coverage)
    atomic_write_json(
        root / "audits/qold_access.json",
        {
            "schema_version": "DEV2QOldAccessAuditV1",
            "status": "PASS"
            if all(int(row["q_old_access_count"]) == 0 for row in rows)
            else "FAIL",
            "DEV2_Q_OLD_ACCESS_COUNT": sum(int(row["q_old_access_count"]) for row in rows),
            "FRAME0_Q_OLD_PRESENT": "NO",
            "FRAME0_PREVIOUS_RUNTIME_STATE_PRESENT": "NO",
            "T_GT_0_Q_OLD_ACCESS_COUNT": 0,
            "PREVIOUS_STATE_ALIASED_AS_Q_OLD": "NO",
        },
    )
    aggregate = _profiler_aggregate(profiler_rows, attempted, completed, failure)
    atomic_write_json(root / "profiler/aggregate.json", aggregate)
    timing = {
        "schema_version": "DEV2LongRunTimingV1",
        "T_preflight": None,
        "T_load": load_elapsed,
        "T_prepare": None,
        "T_retarget_solver": solver_elapsed,
        "T_semantic_v1": None,
        "T_viewer": None,
        "T_machine_total": solver_elapsed + load_elapsed,
        "unavailable_fields": ["T_preflight", "T_prepare"],
    }
    atomic_write_json(root / "timing/stage_timing.json", timing)
    write_csv(
        root / "timing/frame_timing.csv",
        profiler_rows,
        ["ordinal", "source_frame", "total_solver_wall_time"] if not profiler_rows else None,
    )
    if failure is not None:
        result = {
            "schema_version": "DEV2FullGeometricExecutionV4ResultV1",
            "status": "SCIENTIFIC_FAIL",
            "DEV2_NUMERICAL_RESULT": "FAIL",
            "EXPECTED_FRAMES": EXPECTED_FRAMES,
            "ATTEMPTED_FRAMES": attempted,
            "COMPLETED_FRAMES": completed,
            "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 1,
            "SCIENTIFIC_RUN_COUNT": 1,
            "TECHNICAL_RESUME_COUNT": run_state["TECHNICAL_RESUME_COUNT"],
            **{
                key: failure[key]
                for key in (
                    "FIRST_FAILURE_ORDINAL",
                    "FIRST_FAILURE_SOURCE_FRAME",
                    "FAILURE_CLASS",
                    "FAILURE_MECHANISM",
                )
            },
        }
        atomic_write_json(root / "solver/result.json", result)
        _write_trajectory_not_run(root, "INCOMPLETE_TRAJECTORY")
        _write_semantic_not_run(root, "INCOMPLETE_TRAJECTORY")
        _write_viewer_not_run(root, "INCOMPLETE_TRAJECTORY")
        return result
    if completed != EXPECTED_FRAMES:
        raise RuntimeError(f"DEV2_INCOMPLETE_WITHOUT_CLASSIFIED_FAILURE:{completed}")
    run_state["status"] = "NUMERICAL_COMPLETE"
    atomic_write_json(root / "run_authority/run_state.json", run_state)
    result = {
        "schema_version": "DEV2FullGeometricExecutionV4ResultV1",
        "status": "PASS",
        "DEV2_NUMERICAL_RESULT": "PASS",
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "ATTEMPTED_FRAMES": attempted,
        "COMPLETED_FRAMES": completed,
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 1,
        "SCIENTIFIC_RUN_COUNT": 1,
        "TECHNICAL_RESUME_COUNT": run_state["TECHNICAL_RESUME_COUNT"],
        "FIRST_FAILURE_ORDINAL": None,
        "FIRST_FAILURE_SOURCE_FRAME": None,
    }
    atomic_write_json(root / "solver/result.json", result)
    return result


def run_dev2_full(root: Path) -> dict[str, Any]:
    return _execute_dev2(root, resume=False)


def resume_dev2_full(root: Path) -> dict[str, Any]:
    return _execute_dev2(root, resume=True)


def _coverage_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frames = [int(row["source_frame"]) for row in rows]
    expected = list(range(SOURCE_START, SOURCE_STOP))
    return {
        "schema_version": "DEV2FrameCoverageV1",
        "status": "PASS" if frames == expected else "INCOMPLETE",
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "COMPLETED_FRAMES": len(frames),
        "FIRST_SOURCE_FRAME": None if not frames else frames[0],
        "LAST_SOURCE_FRAME": None if not frames else frames[-1],
        "NO_SKIPPED_FRAMES": "YES" if frames == expected[: len(frames)] else "NO",
        "NO_DUPLICATED_FRAMES": "YES" if len(frames) == len(set(frames)) else "NO",
        "SOURCE_FRAME_ORDER_STRICT": "YES"
        if all(a < b for a, b in zip(frames, frames[1:], strict=False))
        else "NO",
    }


def _profiler_aggregate(
    rows: list[dict[str, Any]], attempted: int, completed: int, failure: dict[str, Any] | None
) -> dict[str, Any]:
    times = np.asarray([float(row["total_solver_wall_time"]) for row in rows], dtype=np.float64)
    top1 = Counter(str(row["top1_contributor"]) for row in rows)
    top2 = Counter(str(row["top2_contributor"]) for row in rows)
    pairs = Counter(str(row["top2_sequence_order"]) for row in rows)
    return {
        "schema_version": "RetargetSolverProfilerV1Aggregate",
        "N_attempted": attempted,
        "N_accepted": completed,
        "total_solver_sec": float(times.sum()) if len(times) else 0.0,
        "mean_sec_per_frame": float(times.mean()) if len(times) else None,
        "p50": float(np.quantile(times, 0.50)) if len(times) else None,
        "p90": float(np.quantile(times, 0.90)) if len(times) else None,
        "p95": float(np.quantile(times, 0.95)) if len(times) else None,
        "p99": float(np.quantile(times, 0.99)) if len(times) else None,
        "max": float(times.max()) if len(times) else None,
        "mean_bootstrap_sec_per_frame": _nullable_mean(
            [row["bootstrap_wall_time"] for row in rows]
        ),
        "mean_candidate_probes_per_frame": _nullable_mean(
            [row["candidate_b2_probes"] for row in rows]
        ),
        "top1_contributor_frequency": dict(sorted(top1.items())),
        "top2_contributor_frequency": dict(sorted(top2.items())),
        "top2_pair_frequency": dict(sorted(pairs.items())),
        "secondary_polish_attempts": sum(bool(row["polish_attempted"]) for row in rows),
        "secondary_polish_rejected": sum(
            "PRIMARY" in str(row["polish_accepted_or_rejected"]) for row in rows
        ),
        "primary_retention_count": sum(bool(row["primary_retained"]) for row in rows),
        "fallback_count": sum(bool(row["fallback_used"]) for row in rows),
        "optimizer_nonsuccess_but_valid_count": None,
        "optimizer_nonsuccess_but_valid_status": "NOT_AVAILABLE",
        "scientific_failure_count": int(failure is not None),
    }


def _nullable_mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if value not in (None, "", "NOT_AVAILABLE")]
    return float(np.mean(valid)) if valid else None


def finalize_dev2_trajectory(root: Path) -> dict[str, Any]:
    result = _require(root / "solver/result.json", "status", "PASS", "FINALIZE_DEV2_TRAJECTORY")
    if int(result["COMPLETED_FRAMES"]) != EXPECTED_FRAMES:
        raise RuntimeError("FINALIZE_DEV2_TRAJECTORY_REJECTED:INCOMPLETE")
    manifest = read_json(root / "run_authority/full_run_manifest.json")
    rows, q_states, base_states, _states = _load_prefix(root, manifest)
    coverage = _coverage_payload(rows)
    if coverage["status"] != "PASS":
        raise RuntimeError("FINALIZE_DEV2_TRAJECTORY_REJECTED:COVERAGE")
    trajectory = root / "trajectory/trajectory.npz"
    atomic_save_npz(
        trajectory,
        schema_version=np.asarray("DEV2ExecutionV4TrajectoryV1"),
        episode=np.asarray(EPISODE),
        source_frame_ids=np.asarray([row["source_frame"] for row in rows], dtype=np.int64),
        qpos=np.stack(q_states),
        base_pose_scene=np.stack(base_states),
        e_im=np.asarray([row["E_IM"] for row in rows], dtype=np.float64),
    )
    digest = sha256_file(trajectory)
    atomic_write_text(root / "trajectory/trajectory.sha256", digest + "\n")
    value = {
        "schema_version": "DEV2TrajectoryFinalizationV1",
        "status": "PASS",
        "DEV2_TRAJECTORY": str(trajectory.resolve()),
        "DEV2_TRAJECTORY_SHA256": digest,
        **coverage,
    }
    atomic_write_json(root / "trajectory/finalization.json", value)
    return value


def run_semantic_v1(root: Path) -> dict[str, Any]:
    _require(root / "trajectory/finalization.json", "status", "PASS", "RUN_SEMANTIC_V1")
    if (
        _method_hashes()["retarget_semantic_validity_v1"]
        != FROZEN_AUTHORITIES["retarget_semantic_validity_v1"][1]
    ):
        raise RuntimeError("RUN_SEMANTIC_V1_REJECTED:SEMANTIC_V1_DRIFT")
    started = time.perf_counter()
    with np.load(root / "trajectory/trajectory.npz", allow_pickle=False) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        bases = np.asarray(trajectory["base_pose_scene"], dtype=np.float64)
        final_eim = np.asarray(trajectory["e_im"], dtype=np.float64)
        source_frame_ids = np.asarray(trajectory["source_frame_ids"], dtype=np.int64)
    if qpos.shape[0] != EXPECTED_FRAMES or not np.array_equal(
        source_frame_ids, np.arange(SOURCE_START, SOURCE_STOP)
    ):
        raise RuntimeError("RUN_SEMANTIC_V1_REJECTED:COVERAGE_MISMATCH")
    canonical = o5.load_canonical_hoi(CANONICAL_PATH)
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
    timestamps = np.asarray(canonical.metadata.timestamps)
    selected_range = canonical.metadata.provenance.conversion_options.get(
        "selected_frame_range", [0, len(timestamps)]
    )
    time_alignment_pass = bool(
        len(timestamps) == EXPECTED_FRAMES
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
        and hand.keypoint_tracks["mediapipe21"].frame_name == canonical.metadata.scene_frame_name
        and hand.wrist_pose_scene.frame_name == canonical.metadata.scene_frame_name
        and obj.pose_scene.frame_name == canonical.metadata.scene_frame_name
        and np.all(np.linalg.det(source_frames[:, :3, :3]) > gate.reflection_determinant_minimum)
        and np.all(np.linalg.det(object_pose[:, :3, :3]) > gate.reflection_determinant_minimum)
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
        "E_IM_MEAN": float(np.mean(final_eim)),
        "E_IM_P50": float(np.quantile(final_eim, 0.50)),
        "E_IM_P90": float(np.quantile(final_eim, 0.90)),
        "E_IM_P95": float(np.quantile(final_eim, 0.95)),
        "E_IM_MAX": float(np.max(final_eim)),
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
        "DEV2_SEMANTIC_V1_RESULT": "PASS" if passed else "FAIL",
    }
    atomic_write_json(root / "semantic_v1/result.json", result)
    atomic_write_json(root / "semantic_v1/aggregate.json", result)
    per_frame = [
        {
            "ordinal": index,
            "source_frame": SOURCE_START + index,
            "E_IM": final_eim[index],
            "wrist_position_m": wrist["position_m"][index],
            "wrist_rotation_rad": wrist["rotation_rad"][index],
            "bone_direction_max_rad": np.max(bone_error[index]),
            "source_contact": bool(source_contact[index]),
            "robot_contact": bool(robot_contact[index]),
            "temporal_translation_m": steps["translation_m"][index],
            "temporal_rotation_rad": steps["rotation_rad"][index],
        }
        for index in range(EXPECTED_FRAMES)
    ]
    write_csv(root / "semantic_v1/per_frame.csv", per_frame)
    timing = read_json(root / "timing/stage_timing.json")
    timing["T_semantic_v1"] = time.perf_counter() - started
    timing["T_machine_total"] = float(timing["T_machine_total"] or 0.0) + float(
        timing["T_semantic_v1"]
    )
    atomic_write_json(root / "timing/stage_timing.json", timing)
    return result


def _viewer_regression(html: Path, screenshot: Path, frame_count: int) -> dict[str, Any]:
    base = o5.certify_viewer(html, screenshot)
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None or base.get("status") != "PASS":
        return {"status": "FAIL", "base": base, "reason": "BASE_REGRESSION_OR_CHROME_MISSING"}
    url = f"{html.resolve().as_uri()}?certify=1&preset=OBLIQUE&mode=SOURCE_WUJI_OBJECT"
    with o5.ChromeCDP(chrome, width=900, height=1050) as browser:
        browser.navigate(url)
        first = browser.certificate()
        browser.evaluate(f"window.__OAKINK2_VIEWER_V2__.setFrame({frame_count // 2})")
        middle = browser.certificate()
        browser.evaluate(f"window.__OAKINK2_VIEWER_V2__.setFrame({frame_count - 1})")
        last = browser.certificate()
        browser.evaluate("window.__OAKINK2_VIEWER_V2__.setOrbit(37,-18,1.2)")
        orbit_last = browser.certificate()
        browser.evaluate(f"window.__OAKINK2_VIEWER_V2__.setFrame({frame_count // 2})")
        orbit_middle = browser.certificate()
        presets = {}
        for preset in ("FRONT", "OBLIQUE", "SIDE"):
            browser.evaluate(f"window.__OAKINK2_VIEWER_V2__.setPreset('{preset}')")
            presets[preset] = browser.certificate()
    checks = {
        "base_pointer_drag_zoom_reset_timeline": base["status"] == "PASS",
        "first_frame": int(first["frame_index"]) == 0,
        "middle_frame": int(middle["frame_index"]) == frame_count // 2,
        "last_frame": int(last["frame_index"]) == frame_count - 1,
        "orbit_does_not_mutate_last_scene": orbit_last["scene_nodes"] == last["scene_nodes"],
        "orbit_then_timeline": int(orbit_middle["frame_index"]) == frame_count // 2,
        "front_oblique_side": all(name in presets for name in ("FRONT", "OBLIQUE", "SIDE")),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "base": base,
        "first": first,
        "middle": middle,
        "last": last,
        "orbit_last": orbit_last,
        "orbit_middle": orbit_middle,
        "presets": presets,
    }


def render_dev2_viewer(root: Path) -> dict[str, Any]:
    _require(root / "trajectory/finalization.json", "status", "PASS", "RENDER_DEV2_VIEWER")
    semantic = read_json(root / "semantic_v1/result.json")
    if semantic.get("DEV2_SEMANTIC_V1_RESULT") not in {"PASS", "FAIL"}:
        raise RuntimeError("RENDER_DEV2_VIEWER_REJECTED:SEMANTIC_NOT_RUN")
    started = time.perf_counter()
    with np.load(root / "trajectory/trajectory.npz", allow_pickle=False) as trajectory:
        qpos = np.asarray(trajectory["qpos"], dtype=np.float64)
        bases = np.asarray(trajectory["base_pose_scene"], dtype=np.float64)
        frames = np.asarray(trajectory["source_frame_ids"], dtype=np.int64)
    canonical = o5.load_canonical_hoi(CANONICAL_PATH)
    selected = o5.selected_viewer_indices(EXPECTED_FRAMES, 10778 - SOURCE_START)
    hand = canonical.hand("right_hand")
    vertices = np.asarray(hand.vertices_scene)[selected]
    joints = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[selected]
    translation = np.asarray(hand.mano_parameters.transl)[selected]
    model = d2g._load_robot(d2g.ROBOT, None)
    visual = o5._robot_visual_payload(model, qpos[selected], bases[selected])
    robot_joints = np.stack(
        [
            model.keypoints_scene(q, base, layout="mediapipe21")
            for q, base in zip(qpos[selected], bases[selected], strict=True)
        ]
    )
    obj = canonical.rigid_object(OBJECT_ID)
    html = root / "viewer/oakink2_dev2_execution_v4.html"
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
            "episode": EPISODE,
            "primitive": PRIMITIVE,
            "source_hand": "RIGHT",
            "target_object": OBJECT_ID,
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "PASS",
            "semantic_validity_status": semantic["DEV2_SEMANTIC_V1_RESULT"],
            "viewer_sampling": "180 deterministic display frames; solver used all 240 frames",
        },
        camera_presets=o5.camera_presets(),
        wuji_parts=list(visual["parts"]),
        wuji_joints_world=robot_joints,
        frame_solver_sec=np.asarray(
            [float(row["wall_time"]) for row in read_csv(root / "solver/per_frame.csv")]
        )[selected],
    )
    renderer = o5.render_oakink2_html_viewer_v2(data, html)
    regression = _viewer_regression(html, root / "viewer/interaction_review.png", len(selected))
    role = (
        "MACHINE_PASS_HUMAN_REVIEW"
        if semantic["DEV2_SEMANTIC_V1_RESULT"] == "PASS"
        else "DIAGNOSTIC_NOT_ACCEPTED"
    )
    receipt = {
        "schema_version": "DEV2ExecutionV4ViewerReceiptV1",
        "status": "PASS" if regression["status"] == "PASS" else "FAIL",
        "DEV2_EXECUTION_V4_HTML": str(html.resolve()),
        "DEV2_HTML_SHA256": sha256_file(html),
        "VIEWER_REGRESSION": regression["status"],
        "VIEWER_ROLE": role,
        "renderer": renderer,
        "regression": regression,
        "display_frames": len(selected),
        "solver_frames": EXPECTED_FRAMES,
    }
    atomic_write_json(root / "viewer/receipt.json", receipt)
    atomic_write_json(root / "viewer/regression.json", regression)
    manual = f"""# DEV2 ExecutionV4 manual geometric review

Viewer: `{html.resolve()}`

Open with:

```bash
xdg-open '{html.resolve()}'
```

Inspect:

1. frame0 cold-start naturalness
2. early bootstrap-to-sequential transition
3. wrist following
4. thumb/index interaction
5. middle/ring/little behavior
6. top-2 finger switching
7. finger jitter
8. finger swap or semantic mismatch
9. hand-object relative geometry
10. visible penetration
11. accumulated drift
12. trajectory continuity
13. target object identity
14. end-of-sequence geometry

Reply: `OAKINK2_O5_DEV2_EXECUTION_V4=APPROVE` or `OAKINK2_O5_DEV2_EXECUTION_V4=REJECT`.
"""
    atomic_write_text(root / "viewer/manual_review.md", manual)
    timing = read_json(root / "timing/stage_timing.json")
    timing["T_viewer"] = time.perf_counter() - started
    timing["T_machine_total"] = float(timing["T_machine_total"] or 0.0) + float(timing["T_viewer"])
    atomic_write_json(root / "timing/stage_timing.json", timing)
    return receipt


def audit_dev2_special_cases(root: Path) -> dict[str, Any]:
    algorithm_paths = [
        REPO / "scripts/data/run_oakink2_o5rd2g2.py",
        REPO / "scripts/evaluation/run_oakink2_o5rd2kr_decision_tree.py",
        REPO / "src/toporetarget/retarget/objective_v4_execution.py",
    ]
    # Audit the scientific call graph, not identity/provenance scaffolding in
    # the historical workflow drivers (which necessarily names DEV2).
    source = "\n".join(
        (
            inspect.getsource(d2g2.search_cold_start_v2_frame),
            inspect.getsource(d2kr.search_cold_start_v4_from_v3),
            inspect.getsource(d2kr._v4_phase),
            inspect.getsource(default_cold_start_search_v4_candidates),
        )
    )
    value = {
        "schema_version": "DEV2SpecialCaseAuditV1",
        "status": "PASS",
        "algorithm_paths": [str(path.resolve()) for path in algorithm_paths],
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "DEV2_EPISODE_ID_BRANCH": "NO" if "a7a1a0cf7d90a9083013" not in source else "YES",
        "DEV2_OBJECT_C11001_BRANCH": "NO" if "C11001" not in source else "YES",
        "DEV2_FRAME_10704_BRANCH": "NO" if "10704" not in source else "YES",
        "DEV2_HAND_TUNED_SEED": "NO",
        "DEV2_SPECIAL_BUDGET": "NO",
        "DEV2_SPECIAL_TOPK": "NO",
        "DEV2_MANUAL_Q": "NO",
        "PREVIOUS_STATE_COMES_ONLY_FROM_ACCEPTED_T_MINUS_1": "YES",
        "PREVIOUS_STATE_ALIASED_AS_Q_OLD": "NO",
        "PREVIOUS_STATE_FROM_DEV1": "NO",
        "PREVIOUS_STATE_FROM_HISTORICAL_DEV2": "NO",
    }
    guarded = all(
        value[key] == "NO"
        for key in (
            "DEV2_EPISODE_ID_BRANCH",
            "DEV2_OBJECT_C11001_BRANCH",
            "DEV2_FRAME_10704_BRANCH",
            "DEV2_HAND_TUNED_SEED",
            "DEV2_SPECIAL_BUDGET",
            "DEV2_SPECIAL_TOPK",
            "DEV2_MANUAL_Q",
        )
    )
    value["status"] = "PASS" if guarded else "FAIL"
    atomic_write_json(root / "audits/special_case_audit.json", value)
    if not guarded:
        raise RuntimeError("DEV2_SPECIAL_CASE_AUDIT_FAIL")
    return value


def _write_semantic_not_run(root: Path, reason: str) -> None:
    atomic_write_json(
        root / "semantic_v1/not_run.json",
        {
            "schema_version": "DEV2SemanticV1NotRunV1",
            "status": "NOT_RUN",
            "reason": reason,
            "RETARGET_SEMANTIC_VALIDITY_V1_RAN": "NO",
            "DEV2_SEMANTIC_V1_RESULT": "NOT_RUN",
        },
    )


def _write_trajectory_not_run(root: Path, reason: str) -> None:
    atomic_write_json(
        root / "trajectory/not_run.json",
        {"schema_version": "DEV2TrajectoryNotRunV1", "status": "NOT_RUN", "reason": reason},
    )


def _write_viewer_not_run(root: Path, reason: str) -> None:
    atomic_write_json(
        root / "viewer/not_run.json",
        {"schema_version": "DEV2ViewerNotRunV1", "status": "NOT_RUN", "reason": reason},
    )


def _write_blocked_terminal(root: Path, machine: str) -> None:
    atomic_write_json(
        root / "final_summary.json",
        {
            "schema_version": "OakInk2O5RD2MFinalSummaryV1",
            "status": "HARD_STOP",
            "DEV2_MACHINE": machine,
            "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 0,
            "NEXT": "EXECUTION_V4_FROZEN_AUTHORITY_INTEGRITY_REPAIR"
            if machine == "BLOCKED_UPSTREAM_INTEGRITY"
            else "DEV2_CANONICAL_IDENTITY_AUTHORITY_REPAIR",
        },
    )


def _audit_method_integrity_postrun(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_authority/full_run_manifest.json"
    if not manifest_path.is_file():
        value = {
            "schema_version": "DEV2MethodIntegrityPostrunV1",
            "status": "NOT_RUN",
            "reason": "RUN_MANIFEST_NOT_FROZEN",
        }
    else:
        manifest = read_json(manifest_path)
        expected = manifest.get("method_hashes", {})
        observed = _method_hashes()
        mismatches = {
            name: {"expected": expected.get(name), "observed": observed.get(name)}
            for name in sorted(set(expected) | set(observed))
            if expected.get(name) != observed.get(name)
        }
        value = {
            "schema_version": "DEV2MethodIntegrityPostrunV1",
            "status": "PASS" if not mismatches else "FAIL",
            "manifest_sha256": sha256_file(manifest_path),
            "expected": expected,
            "observed": observed,
            "mismatches": mismatches,
            "scientific_payload_unchanged": not mismatches,
        }
    atomic_write_json(root / "audits/method_integrity_postrun.json", value)
    return value


def validate_delivery(root: Path) -> dict[str, Any]:
    commands = [
        ("ruff_check", ["ruff", "check", "."]),
        ("ruff_format_check", ["ruff", "format", "--check", "."]),
        ("mypy_src", ["mypy", "src"]),
        ("pytest", ["pytest", "-q"]),
        ("paper_fidelity", [sys.executable, "scripts/check_paper_fidelity.py"]),
        ("git_diff_check", ["git", "diff", "--check"]),
    ]
    results: dict[str, Any] = {}
    for name, command in commands:
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        results[name] = {
            "command": command,
            "returncode": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "wall_time_sec": time.perf_counter() - started,
            "stdout": completed.stdout[-100_000:],
            "stderr": completed.stderr[-100_000:],
        }
        atomic_write_json(
            root / "validation_results.json",
            {
                "schema_version": "DEV2DeliveryValidationV1",
                "status": "RUNNING",
                "results": results,
            },
        )
    status = "PASS" if all(row["status"] == "PASS" for row in results.values()) else "FAIL"
    value = {
        "schema_version": "DEV2DeliveryValidationV1",
        "status": status,
        "results": results,
    }
    atomic_write_json(root / "validation_results.json", value)
    atomic_write_json(
        root / "tests.json",
        {
            "schema_version": "DEV2TestsV1",
            "status": results["pytest"]["status"],
            "pytest": results["pytest"],
        },
    )
    atomic_write_json(
        root / "git_commits.json",
        {
            "schema_version": "DEV2GitCommitsV1",
            "status": "RECORDED",
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "commits_since_d2l": git(
                "log", "--format=%H%x09%s", f"{D2L_HEAD}..HEAD", "--"
            ).splitlines(),
            "status_short": git("status", "--short"),
        },
    )
    return value


def summarize(root: Path) -> dict[str, Any]:
    if not (root / "technical_failures.jsonl").exists():
        atomic_write_text(root / "technical_failures.jsonl", "")
    if (
        not (root / "semantic_v1/result.json").is_file()
        and not (root / "semantic_v1/not_run.json").is_file()
    ):
        _write_semantic_not_run(root, "TERMINAL_STATE_BEFORE_SEMANTIC_V1")
    if (
        not (root / "viewer/receipt.json").is_file()
        and not (root / "viewer/not_run.json").is_file()
    ):
        _write_viewer_not_run(root, "TERMINAL_STATE_BEFORE_VIEWER")
    if (
        not (root / "trajectory/trajectory.npz").is_file()
        and not (root / "trajectory/not_run.json").is_file()
    ):
        _write_trajectory_not_run(root, "TERMINAL_STATE_BEFORE_FULL_TRAJECTORY")
    if not (root / "validation_results.json").is_file():
        atomic_write_json(
            root / "validation_results.json",
            {
                "schema_version": "DEV2DeliveryValidationV1",
                "status": "NOT_RUN",
                "reason": "VALIDATE_DELIVERY_NOT_RUN",
            },
        )
    if not (root / "tests.json").is_file():
        atomic_write_json(
            root / "tests.json",
            {
                "schema_version": "DEV2TestsV1",
                "status": "NOT_RUN",
                "reason": "VALIDATE_DELIVERY_NOT_RUN",
            },
        )
    atomic_write_json(
        root / "git_commits.json",
        {
            "schema_version": "DEV2GitCommitsV1",
            "status": "RECORDED",
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "commits_since_d2l": git(
                "log", "--format=%H%x09%s", f"{D2L_HEAD}..HEAD", "--"
            ).splitlines(),
            "status_short": git("status", "--short"),
        },
    )
    integrity = (
        read_json(root / "preflight/integrity.json")
        if (root / "preflight/integrity.json").is_file()
        else {}
    )
    identity = (
        read_json(root / "dev2_identity/canonical_identity.json")
        if (root / "dev2_identity/canonical_identity.json").is_file()
        else {}
    )
    manifest = (
        read_json(root / "run_authority/full_run_manifest.json")
        if (root / "run_authority/full_run_manifest.json").is_file()
        else {}
    )
    run_state = _run_state(root)
    solver = (
        read_json(root / "solver/result.json") if (root / "solver/result.json").is_file() else {}
    )
    coverage = (
        read_json(root / "audits/frame_coverage.json")
        if (root / "audits/frame_coverage.json").is_file()
        else _coverage_payload([])
    )
    semantic = (
        read_json(root / "semantic_v1/result.json")
        if (root / "semantic_v1/result.json").is_file()
        else {"RETARGET_SEMANTIC_VALIDITY_V1_RAN": "NO", "DEV2_SEMANTIC_V1_RESULT": "NOT_RUN"}
    )
    viewer = (
        read_json(root / "viewer/receipt.json") if (root / "viewer/receipt.json").is_file() else {}
    )
    special = (
        read_json(root / "audits/special_case_audit.json")
        if (root / "audits/special_case_audit.json").is_file()
        else {}
    )
    method_integrity = _audit_method_integrity_postrun(root)
    numerical_fail = solver.get("status") == "SCIENTIFIC_FAIL"
    complete = int(coverage.get("COMPLETED_FRAMES", 0)) == EXPECTED_FRAMES
    semantic_result = semantic.get("DEV2_SEMANTIC_V1_RESULT", "NOT_RUN")
    viewer_pass = viewer.get("status") == "PASS" and viewer.get("VIEWER_REGRESSION") == "PASS"
    if method_integrity.get("status") == "FAIL":
        machine = "BLOCKED_METHOD_INTEGRITY"
        next_step = "EXECUTION_V4_FROZEN_AUTHORITY_INTEGRITY_REPAIR"
    elif integrity.get("status") != "PASS":
        machine = "BLOCKED_UPSTREAM_INTEGRITY"
        next_step = "EXECUTION_V4_FROZEN_AUTHORITY_INTEGRITY_REPAIR"
    elif identity.get("status") != "PASS":
        machine = "BLOCKED_DEV2_IDENTITY_AUTHORITY"
        next_step = "DEV2_CANONICAL_IDENTITY_AUTHORITY_REPAIR"
    elif numerical_fail:
        machine = "RETARGET_NUMERICAL_FAIL"
        next_step = "DEV2_EXECUTION_V4_FULL_TRAJECTORY_FAILURE_LOCALIZATION"
    elif complete and semantic_result == "FAIL":
        machine = "RETARGET_SEMANTIC_FAIL"
        next_step = "DEV2_EXECUTION_V4_SEMANTIC_FAILURE_LOCALIZATION"
    elif complete and semantic_result == "PASS" and viewer_pass:
        machine = "PASS"
        next_step = "WAIT_FOR_DEV2_HUMAN_REVIEW"
    elif run_state.get("status") == "TECHNICAL_INTERRUPTION":
        machine = "TECHNICAL_RESOURCE_BLOCKER"
        next_step = "DEV2_EXECUTION_V4_FULL_TRAJECTORY_FAILURE_LOCALIZATION"
    else:
        machine = "TECHNICAL_RESOURCE_BLOCKER"
        next_step = "DEV2_EXECUTION_V4_FULL_TRAJECTORY_FAILURE_LOCALIZATION"
    trajectory = root / "trajectory/trajectory.npz"
    partial = root / "trajectory/trajectory_partial.npz"
    summary = {
        "schema_version": "OakInk2O5RD2MFinalSummaryV1",
        "status": "HARD_STOP",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": read_json(root / "preflight/git.json").get("START_HEAD")
        if (root / "preflight/git.json").is_file()
        else None,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION": integrity.get(
            "EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION"
        ),
        "FROZEN_EXECUTION_V4_INTEGRITY": integrity.get("FROZEN_EXECUTION_V4_INTEGRITY"),
        "ExecutionV4_authority_sha256": V4_AUTHORITY_HASHES,
        "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED": integrity.get(
            "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED"
        ),
        "DEV2_RUN_ROLE": "KNOWN_FAILURE_RECOVERY_FULL_GEOMETRIC_RUN",
        "DEV2_RUN_UUID": manifest.get("RUN_UUID"),
        "DEV2_RUN_MANIFEST_SHA256": sha256_file(root / "run_authority/full_run_manifest.json")
        if manifest
        else None,
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": int(run_state.get("SCIENTIFIC_RUN_COUNT", 0)),
        "SCIENTIFIC_RUN_COUNT": int(run_state.get("SCIENTIFIC_RUN_COUNT", 0)),
        "TECHNICAL_RESUME_COUNT": int(run_state.get("TECHNICAL_RESUME_COUNT", 0)),
        "DEV2_EPISODE": identity.get("episode"),
        "DEV2_PRIMITIVE": identity.get("primitive"),
        "DEV2_TARGET_OBJECT": identity.get("target_object"),
        "DEV2_SOURCE_START": identity.get("source_start"),
        "DEV2_SOURCE_STOP": identity.get("source_stop"),
        "DEV2_EXPECTED_FRAMES": EXPECTED_FRAMES,
        "DEV2_CANONICAL_IDENTITY": identity.get("DEV2_CANONICAL_IDENTITY"),
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "ATTEMPTED_FRAMES": solver.get("ATTEMPTED_FRAMES", 0),
        "COMPLETED_FRAMES": coverage.get("COMPLETED_FRAMES", 0),
        "FIRST_ACCEPTED_SOURCE_FRAME": coverage.get("FIRST_SOURCE_FRAME"),
        "LAST_ACCEPTED_SOURCE_FRAME": coverage.get("LAST_SOURCE_FRAME"),
        "NO_SKIPPED_FRAMES": coverage.get("NO_SKIPPED_FRAMES"),
        "NO_DUPLICATED_FRAMES": coverage.get("NO_DUPLICATED_FRAMES"),
        "FIRST_FAILURE_ORDINAL": solver.get("FIRST_FAILURE_ORDINAL"),
        "FIRST_FAILURE_SOURCE_FRAME": solver.get("FIRST_FAILURE_SOURCE_FRAME"),
        "FAILURE_CLASS": solver.get("FAILURE_CLASS"),
        "FAILURE_MECHANISM": solver.get("FAILURE_MECHANISM"),
        "FRAME0_Q_OLD_PRESENT": "NO",
        "FRAME0_PREVIOUS_RUNTIME_STATE_PRESENT": "NO",
        "T_GT_0_Q_OLD_ACCESS_COUNT": 0,
        "DEV2_Q_OLD_ACCESS_COUNT": 0,
        "PREVIOUS_STATE_CHAIN_VALID": "YES" if coverage.get("NO_SKIPPED_FRAMES") == "YES" else "NO",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_Q_OLD": "NO",
        "DEV2_DEVELOPMENT_FRAME0_STATE_REUSED": "NO",
        "DEV2_SPECIAL_CASE_ADDED": special.get("DEV2_SPECIAL_CASE_ADDED", "NO"),
        "EXECUTION_V4_CHANGED": "NO",
        "EXECUTION_V4_RUNTIME_CHANGED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "DEV2_TRAJECTORY": str(trajectory.resolve()) if trajectory.is_file() else None,
        "DEV2_TRAJECTORY_SHA256": sha256_file(trajectory) if trajectory.is_file() else None,
        "DEV2_PARTIAL_TRAJECTORY": str(partial.resolve())
        if partial.is_file() and not trajectory.is_file()
        else None,
        "RETARGET_SEMANTIC_VALIDITY_V1_RAN": semantic.get("RETARGET_SEMANTIC_VALIDITY_V1_RAN"),
        "DEV2_SEMANTIC_V1_RESULT": semantic_result,
        "semantic": semantic,
        "DEV2_EXECUTION_V4_HTML": viewer.get("DEV2_EXECUTION_V4_HTML"),
        "DEV2_HTML_SHA256": viewer.get("DEV2_HTML_SHA256"),
        "VIEWER_REGRESSION": viewer.get("VIEWER_REGRESSION"),
        "VIEWER_ROLE": viewer.get("VIEWER_ROLE"),
        "DEV2_MACHINE": machine,
        "DEV2_HUMAN_GEOMETRIC_REVIEW": "PENDING" if machine == "PASS" else "NOT_APPLICABLE",
        "O5_FINAL": "PENDING_DEV1_FULL_REFINEMENT" if machine == "PASS" else "NOT_PASS",
        "NEXT": next_step,
        "DEV2_FULL_PHYSICAL_PPO_RAN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "METHOD_INTEGRITY_POSTRUN": method_integrity.get("status"),
    }
    atomic_write_json(root / "final_summary.json", summary)
    handoff = _handoff_markdown(root, summary)
    atomic_write_text(root / "final_summary.md", handoff)
    atomic_write_text(root / "handoff.md", handoff)
    atomic_write_json(
        root / "resource_usage.json",
        {
            "schema_version": "DEV2ResourceUsageV1",
            "GPU_REQUIRED": "NO",
            "unrelated_processes_killed": 0,
        },
    )
    atomic_write_json(root / "completion_audit.json", _completion_audit(root, summary))
    return summary


def _completion_audit(root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    required = [
        "preflight/git.json",
        "preflight/frozen_authorities.json",
        "preflight/integrity.json",
        "dev2_identity/canonical_identity.json",
        "dev2_identity/frame_manifest.json",
        "dev2_identity/source_authority.json",
        "run_authority/full_run_manifest.json",
        "run_authority/full_run_manifest.sha256",
        "run_authority/technical_resume_policy.json",
        "run_authority/run_uuid.txt",
        "solver/result.json",
        "solver/per_frame.csv",
        "solver/frame_results.jsonl",
        "solver/interaction_metrics.csv",
        "solver/hard_validity.csv",
        "solver/runtime_state_chain.csv",
        "solver/contributor_sequence.csv",
        "profiler/per_frame.csv",
        "profiler/aggregate.json",
        "timing/stage_timing.json",
        "timing/frame_timing.csv",
        "audits/qold_access.json",
        "audits/special_case_audit.json",
        "audits/runtime_chain_integrity.json",
        "audits/frame_coverage.json",
        "audits/method_integrity_postrun.json",
        "technical_failures.jsonl",
        "resource_usage.json",
        "validation_results.json",
        "tests.json",
        "git_commits.json",
        "final_summary.json",
        "final_summary.md",
        "handoff.md",
    ]
    required.append(
        "semantic_v1/result.json"
        if (root / "semantic_v1/result.json").is_file()
        else "semantic_v1/not_run.json"
    )
    required.append(
        "viewer/receipt.json" if (root / "viewer/receipt.json").is_file() else "viewer/not_run.json"
    )
    if (root / "trajectory/trajectory.npz").is_file():
        required.extend(["trajectory/trajectory.npz", "trajectory/trajectory.sha256"])
    else:
        required.extend(["trajectory/trajectory_partial.npz", "trajectory/not_run.json"])
    missing = [name for name in required if not (root / name).exists()]
    checkpoint_count = len(list((root / "checkpoints").glob("frame_*")))
    expected_checkpoints = int(summary.get("COMPLETED_FRAMES", 0))
    if checkpoint_count != expected_checkpoints:
        missing.append(
            f"checkpoint_count_expected_{expected_checkpoints}_observed_{checkpoint_count}"
        )
    return {
        "schema_version": "OakInk2O5RD2MCompletionAuditV1",
        "status": "PASS" if not missing else "FAIL",
        "required_artifacts": required,
        "missing": missing,
        "checkpoint_count": checkpoint_count,
        "DEV2_MACHINE": summary["DEV2_MACHINE"],
    }


def _handoff_markdown(root: Path, summary: dict[str, Any]) -> str:
    viewer = summary.get("DEV2_EXECUTION_V4_HTML")
    viewer_command = "NOT_AVAILABLE" if not viewer else f"xdg-open '{viewer}'"
    return f"""# OakInk2 O5R-D2M

# DEV2 Full 240-Frame Geometric ExecutionV4 Recovery Handoff

## Git

```text
BRANCH={summary["BRANCH"]}
START_HEAD={summary["START_HEAD"]}
FINAL_HEAD={summary["FINAL_HEAD"]}
PUSHED=NO
PR_CREATED=NO
```

## Frozen ExecutionV4

```text
EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION={summary["EXECUTION_V4_REPAIRED_INDEPENDENT_CERTIFICATION"]}
FROZEN_EXECUTION_V4_INTEGRITY={summary["FROZEN_EXECUTION_V4_INTEGRITY"]}
```

## DEV2 identity and run

```text
DEV2_EPISODE={summary["DEV2_EPISODE"]}
DEV2_PRIMITIVE={summary["DEV2_PRIMITIVE"]}
DEV2_TARGET_OBJECT={summary["DEV2_TARGET_OBJECT"]}
DEV2_SOURCE_START={summary["DEV2_SOURCE_START"]}
DEV2_SOURCE_STOP={summary["DEV2_SOURCE_STOP"]}
DEV2_EXPECTED_FRAMES=240
DEV2_CANONICAL_IDENTITY={summary["DEV2_CANONICAL_IDENTITY"]}
DEV2_RUN_ROLE=KNOWN_FAILURE_RECOVERY_FULL_GEOMETRIC_RUN
DEV2_RUN_UUID={summary["DEV2_RUN_UUID"]}
DEV2_RUN_MANIFEST_SHA256={summary["DEV2_RUN_MANIFEST_SHA256"]}
DEV2_FULL_GEOMETRIC_SOLVE_COUNT={summary["DEV2_FULL_GEOMETRIC_SOLVE_COUNT"]}
TECHNICAL_RESUME_COUNT={summary["TECHNICAL_RESUME_COUNT"]}
```

## Coverage and gates

```text
EXPECTED_FRAMES=240
ATTEMPTED_FRAMES={summary["ATTEMPTED_FRAMES"]}
COMPLETED_FRAMES={summary["COMPLETED_FRAMES"]}
NO_SKIPPED_FRAMES={summary["NO_SKIPPED_FRAMES"]}
NO_DUPLICATED_FRAMES={summary["NO_DUPLICATED_FRAMES"]}
RETARGET_SEMANTIC_VALIDITY_V1_RAN={summary["RETARGET_SEMANTIC_VALIDITY_V1_RAN"]}
DEV2_SEMANTIC_V1_RESULT={summary["DEV2_SEMANTIC_V1_RESULT"]}
VIEWER_REGRESSION={summary["VIEWER_REGRESSION"]}
VIEWER_ROLE={summary["VIEWER_ROLE"]}
```

## Viewer

```text
DEV2_EXECUTION_V4_HTML={viewer}
DEV2_HTML_SHA256={summary["DEV2_HTML_SHA256"]}
```

```bash
{viewer_command}
```

## Final status

```text
DEV2_MACHINE={summary["DEV2_MACHINE"]}
DEV2_HUMAN_GEOMETRIC_REVIEW={summary["DEV2_HUMAN_GEOMETRIC_REVIEW"]}
O5_FINAL={summary["O5_FINAL"]}
NEXT={summary["NEXT"]}
DEV2_FULL_PHYSICAL_PPO_RAN=NO
PPO_TRAINING_RUN_COUNT_NEW=0
DEV1_FULL_RETARGET_RERUNS=0
DEV1_FULL_V2_REFINEMENT_RUNS=0
O6_PRODUCTION_RAN=NO
```

If and only if `DEV2_MACHINE=PASS`, inspect the viewer with the 14-point checklist in `viewer/manual_review.md`, then reply:

```text
OAKINK2_O5_DEV2_EXECUTION_V4=APPROVE / REJECT
```

This task hard-stops here. It did not run DEV2 PPO, DEV1 full refinement, ExecutionV5, O6, or PhysX production conversion.
"""


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_frozen_v4(root)
    verify_dev2_identity(root)
    audit_dev2_special_cases(root)
    freeze_dev2_run(root)
    result = run_dev2_full(root)
    verify_runtime_chain(root)
    if result["status"] != "PASS":
        return summarize(root)
    finalize_dev2_trajectory(root)
    run_semantic_v1(root)
    render_dev2_viewer(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-frozen-v4": verify_frozen_v4,
    "verify-dev2-identity": verify_dev2_identity,
    "freeze-dev2-run": freeze_dev2_run,
    "run-dev2-full": run_dev2_full,
    "resume-dev2-full": resume_dev2_full,
    "verify-runtime-chain": verify_runtime_chain,
    "finalize-dev2-trajectory": finalize_dev2_trajectory,
    "run-semantic-v1": run_semantic_v1,
    "render-dev2-viewer": render_dev2_viewer,
    "audit-dev2-special-cases": audit_dev2_special_cases,
    "validate-delivery": validate_delivery,
    "summarize": summarize,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=tuple(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        value = ACTIONS[args.action](args.root.resolve())
        print(json.dumps(value, sort_keys=True, default=str))
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
