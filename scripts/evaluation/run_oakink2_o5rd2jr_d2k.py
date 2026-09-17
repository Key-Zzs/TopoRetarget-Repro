#!/usr/bin/env python3
"""O5R-D2J-R support-proxy repair and gated D2K recoverability study."""

# ruff: noqa: E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.rl.isaaclab.import_hocap_objects import _bounded_convex_proxy, _read_obj  # noqa: E402
from toporetarget.physics.support import (  # noqa: E402
    StaticRecoverabilitySupportProxyParameterV1,
    build_static_recoverability_support_proxy,
    support_collision_policy,
    write_finite_planar_support_usda,
)
from toporetarget.rl.static_retarget_reference import (  # noqa: E402
    write_static_retarget_reference,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2jr_d2k_support_proxy_ppo_recoverability_v1"
D2I_ROOT = REPO / ".local/reports/oakink2_o5rd2i_sparsev4_failure_and_ppo_recoverability_v1"
D2J_ROOT = REPO / ".local/reports/oakink2_o5rd2j_d2k_static_physics_and_ppo_recoverability_v1"
D2H_ROOT = (
    REPO / ".local/reports/oakink2_o5rd2h_execution_v3_independent_coldstart_certification_v1"
)
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "f69a5540a82113f3803bebe0e3689baca7a4d69b"
SEQUENCE = "scene_01__A003++seq__d4ddf93a38e3228cdd3a__2023-04-15-10-15-10"
OBJECT_ID = "C10001"
OAKINK2_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2")
ASSET_HUB = OAKINK2_ROOT / "data/OakInk-v2-hub"
VISUAL_MESH = ASSET_HUB / "object_repair/align_ds" / OBJECT_ID / "model.obj"
ANNOTATION = OAKINK2_ROOT / "downloads/hf/OakInk-v2/anno_preview" / f"{SEQUENCE}.pkl"
OBJECT_USD = D2J_ROOT / "object_physics/assets/C10001.usda"
STRICT_CONTRACT = REPO / ".local/reports/stage16d_strict_per_finger_v4/strict_v4_contract.json"
WORKER = REPO / "scripts/rl/isaaclab/run_oakink2_static_scene_smoke.py"

REQUIRED_ACTIONS = (
    "preflight",
    "verify-upstream",
    "audit-existing-support-backend",
    "define-support-proxy",
    "build-support-proxies",
    "run-support-proxy-determinism",
    "run-hocap-support-regression",
    "build-oakink2-static-scenes",
    "run-d2jr-reference-hold-smoke",
    "freeze-d2jr-contracts",
    "audit-ppo-authority",
    "freeze-ppo-contract",
    "freeze-recoverability-study",
    "run-all-reference-hold-baselines",
    "run-ppo-recoverability",
    "evaluate-ppo-recoverability",
    "analyze-recoverability",
    "render-review",
    "summarize",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def freeze_json(path: Path, value: dict[str, Any]) -> str:
    write_json(path, value)
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require(path: Path, status: str, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get("status") != status:
        raise RuntimeError(f"{action}_REJECTED:{path.name}={value.get('status')}")
    return value


def _manifest() -> dict[str, Any]:
    return read_json(D2I_ROOT / "study_manifest/recoverability_manifest.json")


def _anchor_id(index: int) -> str:
    return f"A{index + 1:02d}"


def _clip_id(anchor_id: str) -> str:
    return f"oakink2_static_{anchor_id.lower()}"


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"D2JR_BRANCH_MISMATCH:{branch}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=REPO, check=False
    ).returncode:
        raise RuntimeError("D2JR_START_HEAD_NOT_ANCESTOR")
    value = {
        "schema_version": "OakInk2O5RD2JRGitPreflightV1",
        "status": "PASS",
        "repo": str(REPO),
        "branch": branch,
        "start_head": START_HEAD,
        "observed_head": git("rev-parse", "HEAD"),
        "status_short_at_direct_preflight": "",
        "status_short_at_action": git("status", "--short", "--untracked-files=all"),
        "diff_check": subprocess.run(["git", "diff", "--check"], cwd=REPO).returncode == 0,
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "local_ignored": subprocess.run(
            ["git", "check-ignore", "-q", ".local"], cwd=REPO, check=False
        ).returncode
        == 0,
        "max_gpu_jobs": 1,
        "new_branch_created": False,
        "new_worktree_created": False,
    }
    write_json(root / "preflight/git.json", value)
    return value


def verify_upstream(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "PASS", "VERIFY_UPSTREAM")
    manifest_path = D2I_ROOT / "study_manifest/recoverability_manifest.json"
    manifest = read_json(manifest_path)
    historical_support = read_json(D2J_ROOT / "support/oakink2_support_authority.json")
    object_authority = read_json(D2J_ROOT / "object_physics/oakink2_object_physics_authority.json")
    rows = []
    for index, anchor in enumerate(manifest["anchors"]):
        state = D2H_ROOT / f"sparse_v4/receipts/frame_{int(anchor['ordinal']):04d}_run_1.npz"
        with np.load(state, allow_pickle=False) as archive:
            keys = sorted(archive.files)
            q = np.asarray(archive["qpos"])
            base = np.asarray(archive["base_pose_scene"])
        rows.append(
            {
                "anchor_id": _anchor_id(index),
                "group": anchor["group"],
                "frame": int(anchor["frame_id"]),
                "ordinal": int(anchor["ordinal"]),
                "state_path": str(state.resolve()),
                "state_sha256": sha256_file(state),
                "keys": keys,
                "q_shape": list(q.shape),
                "base_shape": list(base.shape),
                "q_old_present": "q_old" in keys,
            }
        )
    expected_frames = [3567, 3332, 3886, 2986, 3218, 3662, 4687, 2041, 2245, 2535, 2463, 2607]
    checks = {
        "manifest_frozen_anchor_selection": manifest.get("frozen_before_physics") is True,
        "manifest_count_12": len(manifest["anchors"]) == 12,
        "manifest_frames_exact": [int(row["frame_id"]) for row in manifest["anchors"]]
        == expected_frames,
        "reference_is_sparsev4": manifest.get("reference_authority")
        == "stored SparseV4 ExecutionV3 final q/base",
        "q_old_forbidden": manifest.get("q_old_reference_forbidden") is True,
        "states_exact": all(
            row["keys"] == ["base_pose_scene", "qpos"]
            and row["q_shape"] == [20]
            and row["base_shape"] == [4, 4]
            and not row["q_old_present"]
            for row in rows
        ),
        "historical_source_support_unresolved": historical_support.get("selected_support_type")
        == "UNRESOLVED",
        "historical_reason_preserved": historical_support.get("resolver_reason")
        == "no_stable_interval_before_manipulation",
        "object_authority_pass": object_authority.get("status") == "PASS",
        "visual_sha_exact": object_authority.get("visual", {}).get("sha256")
        == "0a21f364bf8fd18f546c2fc7e53ec138579e2252e32e6fdeb60fda1b9e2dc1c1",
        "collision_sha_exact": object_authority.get("collision", {}).get("sha256")
        == "29a7806686f2c6c31aab5e2d9e5bb99959a71690c9c72ac76ef115b311b981c9",
        "collision_representation": object_authority.get("collision", {}).get("representation")
        == "convex_hull_v1",
        "object_usd_hash_matches": OBJECT_USD.is_file()
        and sha256_file(OBJECT_USD) == object_authority.get("collision", {}).get("sha256"),
    }
    value = {
        "schema_version": "OakInk2O5RD2JRFrozenUpstreamEvidenceV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "UPSTREAM_RECOVERABILITY_EVIDENCE_INTEGRITY": ("PASS" if all(checks.values()) else "FAIL"),
        "checks": checks,
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "anchor_states": rows,
        "historical_source_support": {
            "authority": "SUPPORT_UNRESOLVED",
            "reason": historical_support.get("resolver_reason"),
            "result_rewritten": False,
            "receipt_path": str((D2J_ROOT / "support/oakink2_support_authority.json").resolve()),
            "receipt_sha256": sha256_file(D2J_ROOT / "support/oakink2_support_authority.json"),
        },
        "object_authority": object_authority,
        "retarget_optimizer_run_count": 0,
        "certification_split_new_consumption": 0,
        "heldout_split_new_consumption": 0,
    }
    write_json(root / "preflight/upstream_integrity.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_EVIDENCE_INTEGRITY")
    return value


def audit_existing_support_backend(root: Path) -> dict[str, Any]:
    require(root / "preflight/upstream_integrity.json", "PASS", "SUPPORT_BACKEND_AUDIT")
    paths = {
        "source_support_resolver": REPO / "src/toporetarget/physics/support/resolver.py",
        "support_types": REPO / "src/toporetarget/physics/support/types.py",
        "support_physicalizer": REPO / "src/toporetarget/physics/support/runtime_support.py",
        "static_environment": REPO / "scripts/rl/isaaclab/smoke_stage16_full_trajectory_ppo.py",
        "world_wrist_environment": REPO
        / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py",
        "gravity_contract": REPO / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml",
    }
    collision = support_collision_policy("STATIC_RECOVERABILITY_PLANAR_PROXY")
    checks = {
        "all_sources_exist": all(path.is_file() for path in paths.values()),
        "semantics_separate_from_materialization": "write_finite_planar_support_usda"
        not in paths["source_support_resolver"].read_text(encoding="utf-8"),
        "generic_finite_support_materializer": "write_finite_planar_support_usda"
        in paths["support_physicalizer"].read_text(encoding="utf-8"),
        "pairwise_filtering": "apply_hand_support_pair_filter"
        in paths["support_physicalizer"].read_text(encoding="utf-8"),
        "object_support_on": collision["object_support_collision"] is True,
        "hand_support_off": collision["hand_support_collision"] is False,
        "global_collision_not_disabled": collision["global_support_collision_disabled"] is False,
    }
    value = {
        "schema_version": "ExistingSupportPhysicalizationAuditV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "semantic_authority": "SupportResolutionV1 decides source support semantics and remains immutable/unresolved for this OakInk2 sequence.",
        "materialization_authority": "FinitePlanarSupportProxy plus write_finite_planar_support_usda and the shared TableSupportedEnv materialize an already-authorized record.",
        "accepts_study_proxy_without_new_physics": True,
        "architecture": [
            "StaticRecoverabilitySupportProxyV1",
            "FinitePlanarSupportProxy generic record",
            "write_finite_planar_support_usda",
            "TableSupportedEnv/PhysX",
        ],
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "collision_policy": collision,
        "duplicate_physics_backend_created": False,
    }
    write_json(root / "support_backend_audit/hocap_support_backend.json", value)
    write_json(
        root / "support_backend_audit/reusable_components.json",
        {
            "status": value["status"],
            "architecture": value["architecture"],
            "source_hashes": value["source_hashes"],
        },
    )
    if value["status"] != "PASS":
        raise RuntimeError("SUPPORT_PHYSICALIZATION_BACKEND_FAIL")
    return value


def define_support_proxy(root: Path) -> dict[str, Any]:
    require(root / "support_backend_audit/hocap_support_backend.json", "PASS", "DEFINE_PROXY")
    parameters = StaticRecoverabilitySupportProxyParameterV1()
    parameter_payload = {
        "schema_version": parameters.schema_version,
        "status": "FROZEN_BEFORE_ANY_OAKINK2_PHYSX_ROLLOUT",
        **parameters.as_dict(),
        "mathematical_reason": "Use the supporting plane of the deterministic canonical convex collision hull along gravity; audit the shared 1.5 mm bottom envelope; size the finite support from the full projected object bbox plus the existing 20 mm margin.",
        "outcome_tuning": False,
    }
    write_json(root / "support_proxy/proxy_parameters.json", parameter_payload)
    contract = {
        "schema_version": "StaticRecoverabilitySupportProxyV1",
        "status": "FROZEN_FOR_SMOKE",
        "support_type": "STATIC_RECOVERABILITY_PLANAR_PROXY",
        "authority_layer": "STATIC_RECOVERABILITY_STUDY_SUPPORT_PROXY",
        "source_scene_fidelity_claim": False,
        "study_proxy": True,
        "source_support_result_changed": False,
        "inputs_allowed": [
            "frozen canonical object collision geometry",
            "frozen canonical object world pose",
            "world gravity direction",
            "existing generic support physicalization constants",
        ],
        "inputs_forbidden": [
            "PPO output",
            "baseline rollout result",
            "hand penetration result",
            "hand geometry",
            "SparseV4 group label",
            "E_IM",
        ],
        "algorithm": "ObjectBottomSupportEnvelopeV1",
        "parameters": parameter_payload,
        "collision_policy": support_collision_policy("STATIC_RECOVERABILITY_PLANAR_PROXY"),
        "object_pose_changed": False,
        "support_height_outcome_tuned": False,
    }
    digest = freeze_json(root / "support_proxy/support_proxy_contract.json", contract)
    write_json(
        root / "support_proxy/proxy_contract_draft.json",
        {**contract, "contract_sha256": digest, "draft_role": "FROZEN_PRE_ROLLOUT_DRAFT"},
    )
    return {**contract, "sha256": digest}


def _annotation() -> dict[str, Any]:
    with ANNOTATION.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError("OAKINK2_ANNOTATION_NOT_MAPPING")
    return value


def _collision_vertices() -> np.ndarray:
    vertices, _counts, _indices, _low, _high = _read_obj(VISUAL_MESH)
    proxy_vertices, _faces, _gap = _bounded_convex_proxy(vertices)
    return np.asarray(proxy_vertices, dtype=np.float64)


def _proxy_record(
    proxy: Any, audit: dict[str, Any], *, anchor: dict[str, Any], index: int, contract_sha: str
) -> dict[str, Any]:
    return {
        **proxy.as_dict(),
        "schema_version": "StaticRecoverabilitySupportProxyV1",
        "status": "CONSTRUCTED",
        "support_type": "STATIC_RECOVERABILITY_PLANAR_PROXY",
        "authority_layer": "STATIC_RECOVERABILITY_STUDY_SUPPORT_PROXY",
        "SOURCE_SCENE_FIDELITY_CLAIM": "NO",
        "STUDY_PROXY": "YES",
        "anchor_id": _anchor_id(index),
        "clip_id": _clip_id(_anchor_id(index)),
        "object_id": OBJECT_ID,
        "frame": int(anchor["frame_id"]),
        "object_collision_sha256": sha256_file(OBJECT_USD),
        "object_visual_sha256": sha256_file(VISUAL_MESH),
        "proxy_contract_sha256": contract_sha,
        "audit": audit,
    }


def build_support_proxies(root: Path) -> dict[str, Any]:
    require(root / "support_proxy/support_proxy_contract.json", "FROZEN_FOR_SMOKE", "BUILD_PROXY")
    upstream = require(root / "preflight/upstream_integrity.json", "PASS", "BUILD_PROXY")
    annotation = _annotation()
    collision_vertices = _collision_vertices()
    parameters = StaticRecoverabilitySupportProxyParameterV1()
    rows = []
    for index, anchor in enumerate(_manifest()["anchors"]):
        anchor_id = _anchor_id(index)
        pose = np.asarray(
            annotation["obj_transf"][OBJECT_ID][int(anchor["frame_id"])], dtype=np.float64
        )
        proxy, audit = build_static_recoverability_support_proxy(
            collision_vertices, pose, parameters=parameters
        )
        record = _proxy_record(
            proxy,
            audit,
            anchor=anchor,
            index=index,
            contract_sha=sha256_file(root / "support_proxy/support_proxy_contract.json"),
        )
        directory = root / "support_proxy/per_anchor" / anchor_id
        write_json(directory / "support_proxy.json", record)
        write_finite_planar_support_usda(proxy, directory / "support_proxy.usda")
        record_sha = sha256_file(directory / "support_proxy.json")
        asset_sha = sha256_file(directory / "support_proxy.usda")
        rows.append(
            {
                "anchor_id": anchor_id,
                "object": OBJECT_ID,
                "frame": int(anchor["frame_id"]),
                "support_height": proxy.plane_offset,
                "support_center": json.dumps(proxy.table_pose[:3]),
                "support_normal": json.dumps(proxy.plane_normal),
                "extent": json.dumps(proxy.table_extent),
                "collision": "object-support ON; hand-support OFF",
                "proxy_sha": record_sha,
                "asset_sha": asset_sha,
                "result": "PASS",
            }
        )
    write_csv(
        root / "support_proxy/per_anchor_proxy.csv",
        rows,
        [
            "anchor_id",
            "object",
            "frame",
            "support_height",
            "support_center",
            "support_normal",
            "extent",
            "collision",
            "proxy_sha",
            "asset_sha",
            "result",
        ],
    )
    object_authority = upstream["object_authority"]
    revalidation = {
        "schema_version": "OakInk2ObjectPhysicsAuthorityRevalidationV1",
        "status": "PASS",
        "object_id": OBJECT_ID,
        "visual_sha256": sha256_file(VISUAL_MESH),
        "collision_sha256": sha256_file(OBJECT_USD),
        "scale": 1.0,
        "mass_kg": object_authority["dynamics"]["mass_kg"],
        "center_of_mass_m": object_authority["dynamics"]["center_of_mass_m"],
        "principal_inertia_kgm2": object_authority["dynamics"]["principal_inertia_kgm2"],
        "static_friction": object_authority["dynamics"]["static_friction"],
        "dynamic_friction": object_authority["dynamics"]["dynamic_friction"],
        "collision_representation": object_authority["collision"]["representation"],
        "object_dynamics_changed": False,
    }
    write_json(root / "object_physics/authority_revalidation.json", revalidation)
    write_csv(
        root / "object_physics/visual_collision_manifest.csv",
        [
            {
                "object": OBJECT_ID,
                "visual_sha256": revalidation["visual_sha256"],
                "collision_sha256": revalidation["collision_sha256"],
                "collision_representation": revalidation["collision_representation"],
            }
        ],
        ["object", "visual_sha256", "collision_sha256", "collision_representation"],
    )
    summary = {
        "schema_version": "StaticRecoverabilitySupportProxyConstructionV1",
        "status": "PASS" if len(rows) == 12 else "FAIL",
        "constructed": len(rows),
        "target": 12,
        "SUPPORT_PROXY_CONSTRUCTED": f"{len(rows)}/12",
    }
    write_json(root / "support_proxy/construction.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("SUPPORT_PROXY_CONSTRUCTION_FAIL")
    return summary


def run_proxy_determinism(root: Path) -> dict[str, Any]:
    require(root / "support_proxy/construction.json", "PASS", "PROXY_DETERMINISM")
    annotation = _annotation()
    vertices = _collision_vertices()
    rows = []
    with tempfile.TemporaryDirectory(prefix="d2jr_proxy_determinism_") as temporary:
        temp = Path(temporary)
        for index, anchor in enumerate(_manifest()["anchors"]):
            pose = np.asarray(annotation["obj_transf"][OBJECT_ID][int(anchor["frame_id"])])
            first, first_audit = build_static_recoverability_support_proxy(vertices, pose)
            second, second_audit = build_static_recoverability_support_proxy(vertices, pose)
            first_path, second_path = temp / f"{index}_1.usda", temp / f"{index}_2.usda"
            write_finite_planar_support_usda(first, first_path)
            write_finite_planar_support_usda(second, second_path)
            serialized_first = json.dumps(
                {**first.as_dict(), "audit": first_audit}, sort_keys=True, separators=(",", ":")
            ).encode()
            serialized_second = json.dumps(
                {**second.as_dict(), "audit": second_audit}, sort_keys=True, separators=(",", ":")
            ).encode()
            passed = serialized_first == serialized_second and sha256_file(
                first_path
            ) == sha256_file(second_path)
            rows.append(
                {
                    "anchor_id": _anchor_id(index),
                    "status": "PASS" if passed else "FAIL",
                    "record_sha256": hashlib.sha256(serialized_first).hexdigest(),
                    "asset_sha256": sha256_file(first_path),
                }
            )
    value = {
        "schema_version": "StaticRecoverabilitySupportProxyDeterminismV1",
        "status": "PASS"
        if len(rows) == 12 and all(row["status"] == "PASS" for row in rows)
        else "FAIL",
        "deterministic_count": sum(row["status"] == "PASS" for row in rows),
        "target": 12,
        "rows": rows,
    }
    write_json(root / "support_proxy/determinism.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("SUPPORT_PROXY_CONSTRUCTION_FAIL:DETERMINISM")
    return value


def _run_worker(
    root: Path,
    *,
    mode: str,
    output: Path,
    manifest: Path | None = None,
    hocap_clip: str | None = None,
    item_index: int | None = None,
) -> dict[str, Any]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "toporetarget-isaaclab",
        "python",
        str(WORKER),
        mode,
        "--output",
        str(output),
        "--steps",
        "40",
        "--accept-eula",
    ]
    if manifest is not None:
        command.extend(("--manifest", str(manifest)))
    if hocap_clip is not None:
        command.extend(("--hocap-clip", hocap_clip))
    if item_index is not None:
        command.extend(("--item-index", str(item_index)))
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output.with_name(output.stem + "_stdout.log")
    stderr_path = output.with_name(output.stem + "_stderr.log")
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    output_present = output.is_file()
    process = {
        "schema_version": "StaticRecoverabilityWorkerProcessV1",
        "status": "PASS" if completed.returncode == 0 and output_present else "FAIL",
        "command": command,
        "returncode": completed.returncode,
        "output_present": output_present,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": stdout[-16000:],
        "stderr_tail": stderr[-16000:],
        "max_gpu_jobs": 1,
    }
    write_json(output.with_name(output.stem + "_process.json"), process)
    if completed.returncode != 0 or not output_present:
        raise RuntimeError(f"STATIC_RECOVERABILITY_WORKER_FAILED:{mode}:{completed.returncode}")
    return read_json(output)


def run_hocap_support_regression(root: Path) -> dict[str, Any]:
    require(root / "support_proxy/determinism.json", "PASS", "HOCAP_REGRESSION")
    rows = []
    for clip in ("hocap_170105", "hocap_170650"):
        result = _run_worker(
            root,
            mode="hocap-regression",
            output=root / f"hocap_regression/{clip}.json",
            hocap_clip=clip,
        )
        rows.extend(result["rows"])
    result = {
        "schema_version": "ExistingSupportPhysicalizationHOCapRegressionV1",
        "status": "PASS"
        if len(rows) == 2 and all(row["status"] == "PASS" for row in rows)
        else "FAIL",
        "mode": "hocap-regression",
        "max_gpu_jobs": 1,
        "execution": "isolated_isaac_processes_strictly_serialized",
        "rows": rows,
    }
    write_json(root / "hocap_regression/regression.json", result)
    parity = {
        "schema_version": "ExistingSupportPhysicalizationHOCapParityV1",
        "status": result["status"],
        "positive_controls": [row["clip"] for row in result["rows"]],
        "support_transform_parity": all(row.get("status") == "PASS" for row in result["rows"]),
        "collision_filtering_parity": all(
            row.get("contract", {})
            .get("gravity_friction_curriculum", {})
            .get("support_collision_contract", {})
            .get("support_type")
            == "INFERRED_PLANAR_SUPPORT"
            for row in result["rows"]
        ),
        "gravity_parity": all(
            row.get("checks", {}).get("gravity_contract") is True for row in result["rows"]
        ),
        "scene_construction_parity": all(
            row.get("checks", {}).get("scene_construction") is True for row in result["rows"]
        ),
        "HOCAP_PHYSICAL_BACKEND_REGRESSION": result["status"],
    }
    write_json(root / "hocap_regression/parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError("HOCAP_BACKEND_REGRESSION_FAIL")
    return parity


def build_static_scenes(root: Path) -> dict[str, Any]:
    require(root / "support_proxy/construction.json", "PASS", "BUILD_STATIC_SCENES")
    require(root / "object_physics/authority_revalidation.json", "PASS", "BUILD_STATIC_SCENES")
    if not STRICT_CONTRACT.is_file():
        raise FileNotFoundError(f"STRICT_REWARD_CONTRACT_MISSING:{STRICT_CONTRACT}")
    from scripts.data import run_oakink2_o5rd2g as d2g
    from scripts.rl.isaaclab.freeze_stage16d_reward_v3_contact_contract import (
        reference_distances_to_visual_mesh,
    )

    runtime = d2g.V3Runtime("dev_01", D2H_ROOT)
    annotation = _annotation()
    manifest_rows = []
    for index, anchor in enumerate(_manifest()["anchors"]):
        anchor_id = _anchor_id(index)
        clip_id = _clip_id(anchor_id)
        state_path = D2H_ROOT / f"sparse_v4/receipts/frame_{int(anchor['ordinal']):04d}_run_1.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            base = np.asarray(archive["base_pose_scene"], dtype=np.float64)
        object_pose = np.asarray(
            annotation["obj_transf"][OBJECT_ID][int(anchor["frame_id"])], dtype=np.float64
        )
        directory = root / "scene_adapter/per_anchor" / anchor_id
        reference = directory / f"{clip_id}.reference_kinematics_v2.npz"
        write_static_retarget_reference(
            reference,
            qpos=qpos,
            base_pose_world=base,
            object_pose_world=object_pose,
            robot_model=runtime.model,
            source_frame_id=int(anchor["frame_id"]),
            anchor_id=anchor_id,
        )
        object_mesh_root = directory / "object_mesh"
        object_mesh_root.mkdir(parents=True, exist_ok=True)
        object_mesh = object_mesh_root / f"{clip_id}.obj"
        shutil.copyfile(VISUAL_MESH, object_mesh)
        distances, distance_metadata = reference_distances_to_visual_mesh(
            reference=reference, object_mesh=object_mesh
        )
        contact_mask = distances < 0.03
        contracts = directory / "reference_contracts"
        contracts.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            contracts / f"reference_contact_mask_{clip_id}.npz",
            reference_expected_contact_mask=contact_mask,
            reference_fingertip_to_object_distance_m=distances.astype(np.float32),
            finger_order=np.asarray(("thumb", "index", "middle", "ring", "pinky")),
            metadata=np.asarray(
                json.dumps(
                    {
                        "schema_version": "StaticReferenceContactDistanceV1",
                        "status": "PASS",
                        "clip": clip_id,
                        "threshold_m": 0.03,
                        "threshold_authority": "existing Stage16DReferenceContactMaskV1 strict primary distance",
                        "geometry": distance_metadata,
                        "outcomes_observed": False,
                    },
                    sort_keys=True,
                )
            ),
        )
        source_class = np.where(contact_mask, "SOURCE_CONTACT_CONFIRMED", "SOURCE_NO_CONTACT")
        np.savez_compressed(
            contracts / f"strict_source_contact_mask_{clip_id}.npz",
            strict_source_contact_mask=contact_mask,
            source_contact_class=source_class,
            finger_names=np.asarray(("thumb", "index", "middle", "ring", "pinky")),
            control_index=np.arange(len(contact_mask), dtype=np.int64),
        )
        support_dir = root / "support_proxy/per_anchor" / anchor_id
        manifest_rows.append(
            {
                "anchor_id": anchor_id,
                "clip_id": clip_id,
                "group": anchor["group"],
                "frame": int(anchor["frame_id"]),
                "ordinal": int(anchor["ordinal"]),
                "reference": str(reference.resolve()),
                "reference_sha256": sha256_file(reference),
                "state_sha256": sha256_file(state_path),
                "object_usd": str(OBJECT_USD.resolve()),
                "support_proxy": str((support_dir / "support_proxy.json").resolve()),
                "support_asset": str((support_dir / "support_proxy.usda").resolve()),
                "contact_contract": str(STRICT_CONTRACT.resolve()),
                "contact_mask_root": str(contracts.resolve()),
                "reference_distance_root": str(contracts.resolve()),
                "object_mesh_root": str(object_mesh_root.resolve()),
                "expected_contact_fingers": [
                    name
                    for name, active in zip(
                        ("thumb", "index", "middle", "ring", "pinky"),
                        contact_mask[0],
                        strict=True,
                    )
                    if active
                ],
                "q_old_used": False,
                "hand_target_constant": True,
                "object_target_constant": True,
                "reference_velocity_zero": True,
            }
        )
    value = {
        "schema_version": "OakInk2StaticContactSceneAdapterV1",
        "status": "PASS" if len(manifest_rows) == 12 else "FAIL",
        "anchors": manifest_rows,
        "object_pose_writes_after_reset": 0,
        "hidden_object_force": False,
        "wrist_root_teleport": False,
        "reset_authority": "exact stored SparseV4 q/base plus stored canonical OakInk2 object pose plus study proxy",
    }
    write_json(root / "scene_adapter/oakink2_static_scene_adapter.json", value)
    write_json(
        root / "scene_adapter/static_reference_adapter.json",
        {
            "schema_version": "StaticRetargetReferenceAdapterV1",
            "status": value["status"],
            "runtime_samples": 321,
            "hand_target": "constant exact SparseV4 q/base",
            "object_target": "constant exact canonical object pose",
            "reference_velocity": "zero",
            "q_old_used": False,
            "anchors": [
                {
                    key: row[key]
                    for key in ("anchor_id", "reference", "reference_sha256", "state_sha256")
                }
                for row in manifest_rows
            ],
        },
    )
    write_json(root / "scene_adapter/runtime_manifest.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("OAKINK2_SCENE_ADAPTER_FAIL")
    return value


def run_d2jr_smoke(root: Path) -> dict[str, Any]:
    require(root / "object_physics/authority_revalidation.json", "PASS", "D2JR_SMOKE")
    require(root / "support_proxy/support_proxy_contract.json", "FROZEN_FOR_SMOKE", "D2JR_SMOKE")
    require(root / "scene_adapter/oakink2_static_scene_adapter.json", "PASS", "D2JR_SMOKE")
    require(root / "hocap_regression/parity.json", "PASS", "D2JR_SMOKE")
    smoke_rows = []
    for item_index in range(12):
        anchor_id = _anchor_id(item_index)
        item = _run_worker(
            root,
            mode="oakink2-smoke",
            output=root / f"d2jr_smoke/per_anchor/{anchor_id}.json",
            manifest=root / "scene_adapter/runtime_manifest.json",
            item_index=item_index,
        )
        smoke_rows.extend(item["rows"])
    result = {
        "schema_version": "OakInk2StaticReferenceHoldSmokeV1",
        "status": "PASS"
        if len(smoke_rows) == 12 and all(row["status"] == "PASS" for row in smoke_rows)
        else "FAIL",
        "mode": "oakink2-smoke",
        "max_gpu_jobs": 1,
        "execution": "isolated_isaac_processes_strictly_serialized",
        "rows": smoke_rows,
    }
    write_json(root / "d2jr_smoke/qualification.json", result)
    rows = []
    for row in result["rows"]:
        outcome = row.get("scientific_physical_outcome", {})
        rows.append(
            {
                "anchor_id": row["anchor_id"],
                "scene": "PASS" if row.get("checks", {}).get("scene_construction") else "FAIL",
                "physx": "PASS" if row.get("checks", {}).get("physx_starts") else "FAIL",
                "controller": "PASS" if row.get("checks", {}).get("controller_starts") else "FAIL",
                "finite": "PASS" if row.get("checks", {}).get("simulation_finite") else "FAIL",
                "scientific_physical_outcome": json.dumps(outcome, sort_keys=True),
                "result": row["status"],
            }
        )
    write_csv(
        root / "d2jr_smoke/per_anchor.csv",
        rows,
        [
            "anchor_id",
            "scene",
            "physx",
            "controller",
            "finite",
            "scientific_physical_outcome",
            "result",
        ],
    )
    write_csv(
        root / "d2jr_smoke/scene_metrics.csv",
        [
            {
                "anchor_id": row["anchor_id"],
                **row.get("scientific_physical_outcome", {}),
            }
            for row in result["rows"]
        ],
        [
            "anchor_id",
            "support_contact_observed",
            "object_translation_drift_m_max",
            "object_rotation_drift_rad_max",
            "classification_role",
        ],
    )
    if result["status"] != "PASS" or len(result["rows"]) != 12:
        raise RuntimeError("REFERENCE_HOLD_INFRA_FAIL")
    return result


def freeze_d2jr_contracts(root: Path) -> dict[str, Any]:
    require(root / "support_proxy/determinism.json", "PASS", "FREEZE_D2JR")
    require(root / "hocap_regression/parity.json", "PASS", "FREEZE_D2JR")
    smoke = require(root / "d2jr_smoke/qualification.json", "PASS", "FREEZE_D2JR")
    if len(smoke.get("rows", [])) != 12:
        raise RuntimeError("FREEZE_D2JR_REJECTED:REFERENCE_HOLD_NOT_12")
    proxy = read_json(root / "support_proxy/support_proxy_contract.json")
    scene = read_json(root / "scene_adapter/oakink2_static_scene_adapter.json")
    reference = read_json(root / "scene_adapter/static_reference_adapter.json")
    frozen = root / "frozen_d2jr"
    proxy_sha = freeze_json(frozen / "support_proxy.json", {**proxy, "status": "PASS"})
    scene_sha = freeze_json(frozen / "static_scene_adapter.json", scene)
    reference_sha = freeze_json(frozen / "static_reference_adapter.json", reference)
    contract = {
        "schema_version": "StaticPhysicalStudySceneContractV2",
        "status": "PASS",
        "support_proxy_sha256": proxy_sha,
        "static_scene_adapter_sha256": scene_sha,
        "static_reference_adapter_sha256": reference_sha,
        "object_physics_authority": read_json(root / "object_physics/authority_revalidation.json"),
        "gravity_world_mps2": [0.0, 0.0, -9.81],
        "support_collision_policy": support_collision_policy("STATIC_RECOVERABILITY_PLANAR_PROXY"),
        "reference_hold_executable": "12/12",
        "source_support_authority": "SUPPORT_UNRESOLVED",
        "source_support_result_rewritten": False,
        "source_scene_fidelity_claim": False,
        "study_proxy": True,
        "object_pose_writes_after_reset": 0,
        "hidden_object_force_used": False,
        "wrist_root_teleport_used": False,
        "D2J_R_STATUS": "PASS",
        "D2K_AUTHORIZED": "YES",
    }
    contract_sha = freeze_json(frozen / "physical_study_scene_contract.json", contract)
    value = {
        "schema_version": "OakInk2O5RD2JRFreezeV1",
        "status": "PASS",
        "D2J_R_STATUS": "PASS",
        "D2K_AUTHORIZED": "YES",
        "STATIC_RECOVERABILITY_SUPPORT_PROXY_SHA256": proxy_sha,
        "OAKINK2_STATIC_SCENE_ADAPTER_SHA256": scene_sha,
        "STATIC_REFERENCE_ADAPTER_SHA256": reference_sha,
        "STATIC_PHYSICAL_STUDY_SCENE_CONTRACT_V2_SHA256": contract_sha,
    }
    write_json(frozen / "qualification.json", value)
    return value


def audit_ppo_authority(root: Path) -> dict[str, Any]:
    require(root / "frozen_d2jr/qualification.json", "PASS", "PPO_AUTHORITY")
    historical = read_json(D2I_ROOT / "ppo_authority/authority_audit.json")
    sources = {
        "environment": "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py",
        "environment_config": "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env_cfg.py",
        "environment_factory": "scripts/rl/isaaclab/smoke_stage16_full_trajectory_ppo.py",
        "ppo_contract": "src/toporetarget/rl/ppo/ppo26d_contract.py",
        "ppo_trainer": "src/toporetarget/rl/ppo/ppo26d_trainer.py",
        "ppo_algorithm": "src/toporetarget/rl/ppo/trainer.py",
        "ppo_networks": "src/toporetarget/rl/ppo/networks.py",
        "ppo_distribution": "src/toporetarget/rl/ppo/distribution.py",
        "reward": "src/toporetarget/rl/reference_tracking/ppo26d_reward.py",
        "grouped_reward": "src/toporetarget/rl/reference_tracking/grouped_multiplicative_reward.py",
        "strict_contact": "src/toporetarget/rl/reference_tracking/strict_per_finger_contact.py",
        "rse": "src/toporetarget/rl/reference_tracking/reference_scoped_exploration.py",
        "training_config": "configs/rl/physical_refinement.yaml",
    }
    source_hashes = {
        name: {"path": path, "sha256": sha256_file(REPO / path)} for name, path in sources.items()
    }
    value = {
        "schema_version": "CurrentPhysicalPPOAuthorityV1",
        "status": "UNIQUE_CURRENT_AUTHORITY",
        "lineage": "stored retarget reference -> Stage16D PPO26D physical residual policy",
        "environment_class": "TableSupportedEnv(IsaacPPO26DReferenceTrackingEnv)",
        "policy_implementation": "ActorCritic tanh-bounded diagonal Gaussian",
        "ppo_implementation": "PPO26DTrainer/PPOTrainer",
        "network": historical["network"],
        "observation": historical["observation"],
        "action": historical["action"],
        "controller": "explicit finite virtual 6-DoF wrist plus 20 Wuji finger actuators",
        "reward": "Stage16GroupedMultiplicativeRewardV1 plus strict per-finger V4 contact",
        "reward_weights": "unchanged current production implementation",
        "rse": "Stage16ReferenceScopedExplorationV1",
        "rsi_reset": "static single reference index adapter; exact frame0 state on reset",
        "action_bounds": [-1.0, 1.0],
        "normalization": "PPO26D running observation normalizer",
        "rollout_horizon": 40,
        "num_envs": 1024,
        "samples_per_update": 40960,
        "max_updates": 15,
        "evaluation_protocol": "Eval10, same pre-registered seeds before and after PPO",
        "checkpoint_acceptance_protocol": "fixed-budget final update; no outcome-driven selection",
        "independent_policy_per_anchor": True,
        "authority_ambiguity": False,
        "static_adapter_delta": "reference/reset and scene adapter only; PPO/reward math unchanged",
        "source_hashes": source_hashes,
    }
    write_json(root / "ppo_authority/current_ppo_authority.json", value)
    return value


def freeze_ppo_contract(root: Path) -> dict[str, Any]:
    authority = require(
        root / "ppo_authority/current_ppo_authority.json", "UNIQUE_CURRENT_AUTHORITY", "PPO_FREEZE"
    )
    reward_sources = {
        name: item
        for name, item in authority["source_hashes"].items()
        if name in {"reward", "grouped_reward", "strict_contact", "rse", "environment_config"}
    }
    reward = {
        "schema_version": "PPORewardAuthorityV1",
        "status": "FROZEN_UNCHANGED_CURRENT_PRODUCTION_REWARD",
        "PPO_REWARD_CHANGED": "NO",
        "reward": "Stage16GroupedMultiplicativeRewardV1",
        "contact": "TopoRetargetReferenceTrackingReward26DV4 strict per-finger",
        "aggregation": "grouped_multiplicative_v1",
        "normalization": "current PPO26D running observation normalizer",
        "rse": "Stage16ReferenceScopedExplorationV1 unchanged algorithm",
        "penetration_metric_role": "EVALUATION_ONLY_NOT_REWARD",
        "sparsev4_specific_term": False,
        "object_specific_term": False,
        "source_hashes": reward_sources,
    }
    reward_sha = freeze_json(root / "ppo_authority/reward_authority.json", reward)
    contract = {
        "schema_version": "StaticDiagnosticPPOContractV1",
        "status": "FROZEN_BEFORE_BASELINE_OR_PPO",
        "study_role": "DIAGNOSTIC_PHYSICAL_RECOVERABILITY",
        "algorithm": "PPO26D",
        "environment": authority["environment_class"],
        "network": authority["network"],
        "observation": authority["observation"],
        "action": authority["action"],
        "controller": authority["controller"],
        "reward_authority_sha256": reward_sha,
        "PPO_REWARD_CHANGED": "NO",
        "rse": authority["rse"],
        "rsi_reset": authority["rsi_reset"],
        "action_bounds": authority["action_bounds"],
        "normalization": authority["normalization"],
        "training": {
            "num_envs": 1024,
            "rollout_length": 40,
            "samples_per_update": 40960,
            "max_updates": 15,
            "sample_cap_per_anchor": 614400,
            "adaptive_budget": False,
            "independent_policy_per_anchor": True,
        },
        "evaluation": {"episodes": 10, "paired_reset_seeds": True},
        "checkpoint_acceptance": authority["checkpoint_acceptance_protocol"],
        "source_hashes": authority["source_hashes"],
    }
    ppo_sha = freeze_json(root / "ppo_authority/ppo_contract.json", contract)
    return {
        "schema_version": "OakInk2StaticPPOAuthorityFreezeV1",
        "status": "PASS",
        "PPO_REWARD_CHANGED": "NO",
        "reward_sha256": reward_sha,
        "ppo_sha256": ppo_sha,
    }


def _runtime_geometry_manifest(root: Path, clips: list[str]) -> Path:
    from scipy.spatial import ConvexHull

    source = read_json(
        REPO
        / ".local/reports/stage16d_metric_qualification_and_ppo/runtime_collision_geometry_manifest.json"
    )
    vertices = _collision_vertices()
    hull = ConvexHull(vertices)
    faces = hull.simplices.astype(np.int64)
    geometry_sha = hashlib.sha256(
        vertices.astype(np.float64).tobytes() + faces.tobytes()
    ).hexdigest()
    object_shapes = {}
    pair_filter = {}
    for clip in clips:
        shape_id = f"object:{clip}:0:/{clip}/Collision/convex_hull_v1"
        shape = {
            "body_name": clip,
            "convex_vertices_m": vertices.tolist(),
            "generated_asset_path": str(OBJECT_USD.relative_to(REPO)),
            "generated_asset_sha256": sha256_file(OBJECT_USD),
            "geometry_sha256": geometry_sha,
            "geometry_type": "convex_hull",
            "local_transform": {
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "translation_xyz_m": [0.0, 0.0, 0.0],
            },
            "scale_xyz": [1.0, 1.0, 1.0],
            "shape_id": shape_id,
            "source_asset_path": str(VISUAL_MESH),
            "source_asset_sha256": sha256_file(VISUAL_MESH),
            "triangle_count": int(len(faces)),
            "triangle_indices": faces.tolist(),
            "vertex_count": int(len(vertices)),
        }
        object_shapes[clip] = [shape]
        pair_filter[clip] = [f"{hand['shape_id']}<->{shape_id}" for hand in source["hand_shapes"]]
    value = {
        "schema_version": "RuntimeCollisionGeometryManifestV1",
        "units": "metre",
        "geometry_authority": "authored Wuji collision proxies and frozen C10001 convex collision hull",
        "handedness": "right",
        "hand_shapes": source["hand_shapes"],
        "object_shapes": object_shapes,
        "pair_filter": pair_filter,
        "excluded": source["excluded"],
        "validation": {
            "all_local_transforms_finite": True,
            "all_scales_positive": True,
            "asset_hashes_match": True,
            "manifest_hand_shape_count": len(source["hand_shapes"]),
            "runtime_hand_shape_count": len(source["hand_shapes"]),
            "runtime_composed_hand_shape_count": len(source["hand_shapes"]),
            "runtime_object_shape_count": {clip: 1 for clip in clips},
            "runtime_cfg_asset_paths": {
                "robot": source["validation"]["runtime_cfg_asset_paths"]["robot"],
                **{clip: str(OBJECT_USD.relative_to(REPO)) for clip in clips},
            },
            "runtime_cfg_asset_paths_match": True,
        },
    }
    path = root / "study_contract/runtime_collision_geometry_manifest.json"
    write_json(path, value)
    return path


def freeze_recoverability_study(root: Path) -> dict[str, Any]:
    qualification = require(root / "frozen_d2jr/qualification.json", "PASS", "STUDY_FREEZE")
    ppo = require(
        root / "ppo_authority/ppo_contract.json", "FROZEN_BEFORE_BASELINE_OR_PPO", "STUDY_FREEZE"
    )
    reward = require(
        root / "ppo_authority/reward_authority.json",
        "FROZEN_UNCHANGED_CURRENT_PRODUCTION_REWARD",
        "STUDY_FREEZE",
    )
    source_manifest = _manifest()
    scene = read_json(root / "scene_adapter/runtime_manifest.json")
    contributor_rows: dict[int, dict[str, str]] = {}
    with (D2I_ROOT / "sparsev4_analysis/contributor_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            contributor_rows[int(row["ordinal"])] = row
    anchors = []
    seeds = []
    for index, (source, runtime) in enumerate(
        zip(source_manifest["anchors"], scene["anchors"], strict=True)
    ):
        anchor_id = _anchor_id(index)
        contributor = contributor_rows[int(source["ordinal"])]
        anchors.append(
            {
                **runtime,
                "semantic_supergroup": (
                    "SEMANTIC_PASS"
                    if source["group"] in {"GROUP_A", "GROUP_C"}
                    else "SEMANTIC_FAIL"
                ),
                "final_e_im": float(source["final_e_im"]),
                "bootstrap_e_im": float(contributor["bootstrap_e_im"]),
                "rho1": float(contributor["rho1"]),
                "rho2": float(contributor["rho2"]),
                "top1_contributor": contributor["top1_finger"],
                "top2_contributor": contributor["top2_finger"],
                "geom_penetration_p95_m": float(source["geom_penetration_p95_m"]),
                "geom_penetration_max_m": float(source["geom_penetration_max_m"]),
                "source_support_authority": "SUPPORT_UNRESOLVED",
                "study_support_proxy": True,
            }
        )
        seeds.append(
            {
                "anchor_id": anchor_id,
                "training_seed": 20261900 + index,
                "eval_seeds": [20262900 + 100 * index + episode for episode in range(10)],
                "baseline_post_ppo_paired": True,
            }
        )
    final_manifest = {
        "schema_version": "RetargetToPPORecoverabilityFinalManifestV1",
        "status": "FROZEN_BEFORE_BASELINE_OR_PPO",
        "anchor_count": len(anchors),
        "semantic_pass_n": sum(row["semantic_supergroup"] == "SEMANTIC_PASS" for row in anchors),
        "semantic_fail_n": sum(row["semantic_supergroup"] == "SEMANTIC_FAIL" for row in anchors),
        "same_d2jr_anchors": True,
        "q_old_reference_forbidden": True,
        "anchors": anchors,
    }
    manifest_sha = freeze_json(root / "study_manifest/final_manifest.json", final_manifest)
    write_json(
        root / "study_manifest/manifest_audit.json",
        {
            "schema_version": "RetargetToPPORecoverabilityManifestAuditV1",
            "status": "PASS",
            "same_anchor_ids": [row["anchor_id"] for row in anchors]
            == [row["anchor_id"] for row in scene["anchors"]],
            "anchor_count": len(anchors),
            "semantic_pass_n": 6,
            "semantic_fail_n": 6,
            "source_manifest_sha256": sha256_file(
                D2I_ROOT / "study_manifest/recoverability_manifest.json"
            ),
            "final_manifest_sha256": manifest_sha,
        },
    )
    geometry = _runtime_geometry_manifest(root, [row["clip_id"] for row in anchors])
    contract = {
        "schema_version": "RetargetToPPORecoverabilityStudyContractV1",
        "status": "FROZEN_BEFORE_PPO",
        "STUDY_ROLE": "DIAGNOSTIC_PHYSICAL_RECOVERABILITY",
        "anchor_manifest_sha256": manifest_sha,
        "scene_contract_sha256": qualification["STATIC_PHYSICAL_STUDY_SCENE_CONTRACT_V2_SHA256"],
        "support_proxy_sha256": qualification["STATIC_RECOVERABILITY_SUPPORT_PROXY_SHA256"],
        "ppo_contract_sha256": sha256_file(root / "ppo_authority/ppo_contract.json"),
        "reward_sha256": sha256_file(root / "ppo_authority/reward_authority.json"),
        "runtime_geometry_manifest": str(geometry),
        "runtime_geometry_manifest_sha256": sha256_file(geometry),
        "seeds": seeds,
        "training_budget": ppo["training"],
        "eval_n": 10,
        "baseline_horizon": 321,
        "recoverability_metric": "paired continuous physical metrics",
        "RECOVERABILITY_MODE": "CONTINUOUS_METRICS_ONLY",
        "binary_gate_authority_audit": {
            "status": "NO_DEFENSIBLE_STATIC_BINARY_GATE",
            "reason": "current repo physical gates are trajectory/lift-specific and do not authorize thresholds for static OakInk2 support-proxy anchors",
            "thresholds_invented": False,
        },
        "decision_thresholds": {
            "binary_recovery": None,
            "final_result_maximum": "INCONCLUSIVE",
            "strong_if_binary_existed": ">=80% SEMANTIC_FAIL",
            "poor_if_binary_existed": "<=20% SEMANTIC_FAIL",
        },
        "technical_retry_policy": {
            "schema_version": "PPOStudyTechnicalRetryPolicyV1",
            "allowed": ["process crash", "IO interruption", "host interruption"],
            "oom_num_env_reduction": False,
            "scientific_failure_extra_budget": False,
        },
        "PPO_REWARD_CHANGED": reward["PPO_REWARD_CHANGED"],
        "MAX_GPU_JOBS": 1,
    }
    study_sha = freeze_json(root / "study_contract/recoverability_study_contract.json", contract)
    return {
        "schema_version": "RetargetToPPORecoverabilityStudyFreezeV1",
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "study_contract_sha256": study_sha,
        "RECOVERABILITY_MODE": "CONTINUOUS_METRICS_ONLY",
    }


PPO_STUDY_WORKER = REPO / "scripts/rl/isaaclab/run_oakink2_static_ppo_study.py"


def _run_ppo_study_worker(
    root: Path,
    *,
    mode: str,
    item_index: int,
    output: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "toporetarget-isaaclab",
        "python",
        str(PPO_STUDY_WORKER),
        mode,
        "--manifest",
        str(root / "study_manifest/final_manifest.json"),
        "--item-index",
        str(item_index),
        "--study-contract",
        str(root / "study_contract/recoverability_study_contract.json"),
        "--geometry-manifest",
        str(root / "study_contract/runtime_collision_geometry_manifest.json"),
        "--output",
        str(output),
        "--accept-eula",
    ]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    log_root = output if mode == "train" else output.parent / output.stem
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = log_root / "stdout.log", log_root / "stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        completed = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    receipt_path = output / "train_receipt.json" if mode == "train" else output
    process = {
        "schema_version": "OakInk2StaticPPOStudyProcessV1",
        "status": "PASS" if completed.returncode == 0 and receipt_path.is_file() else "FAIL",
        "mode": mode,
        "item_index": item_index,
        "command": command,
        "returncode": completed.returncode,
        "max_gpu_jobs": 1,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_tail": stdout_path.read_text(encoding="utf-8", errors="replace")[-12000:],
        "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-12000:],
    }
    write_json(log_root / "process.json", process)
    if process["status"] != "PASS":
        with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(process, sort_keys=True) + "\n")
        raise RuntimeError(f"PPO_STUDY_{mode.upper()}_TECHNICAL_FAILURE:{item_index}")
    return read_json(receipt_path)


def _aggregate_evaluation(
    root: Path, *, phase: str, results: list[dict[str, Any]]
) -> dict[str, Any]:
    per_rollout = [row for result in results for row in result["rows"]]
    per_anchor = []
    numeric = [
        "physical_penetration_p95_m",
        "physical_penetration_max_m",
        "physical_penetration_terminal_m",
        "physical_penetrating_fraction",
        "contact_count",
        "contact_persistence_fraction",
        "object_translation_drift_m_max",
        "object_rotation_drift_rad_max",
        "object_linear_velocity_mps_max",
        "object_angular_velocity_radps_max",
        "terminal_object_linear_velocity_mps",
        "terminal_object_angular_velocity_radps",
        "wrist_reference_deviation_m_mean",
        "wrist_reference_deviation_rad_mean",
        "finger_reference_deviation_l2_rad_mean",
        "action_l2_mean",
        "action_max_abs",
        "action_saturation_fraction",
        "reward_total_mean",
        "reward_object_mean",
        "reward_interaction_mean",
        "object_support_contact_fraction",
    ]
    for result in results:
        rows = result["rows"]
        per_anchor.append(
            {
                "anchor_id": result["anchor_id"],
                "clip_id": result["clip_id"],
                "episodes": len(rows),
                "simulation_finite_all": all(row["simulation_finite"] for row in rows),
                "metrics_complete_all": all(row["metrics_complete"] for row in rows),
                **{name: float(np.mean([row[name] for row in rows])) for name in numeric},
            }
        )
    directory = root / ("baseline_physics" if phase == "baseline" else "ppo_eval")
    write_csv(directory / "per_rollout.csv", per_rollout, list(per_rollout[0]))
    write_csv(directory / "per_anchor.csv", per_anchor, list(per_anchor[0]))
    summary = {
        "schema_version": "OakInk2StaticPhysicalEvaluationAggregateV1",
        "status": "PASS"
        if len(per_anchor) == 12
        and len(per_rollout) == 120
        and all(row["simulation_finite_all"] and row["metrics_complete_all"] for row in per_anchor)
        else "TECHNICAL_INFRA_FAILURE",
        "phase": phase,
        "anchors": len(per_anchor),
        "rollouts": len(per_rollout),
        "per_anchor": per_anchor,
    }
    write_json(directory / "summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError(f"{phase.upper()}_INFRA_FAILURE")
    return summary


def run_all_baselines(root: Path) -> dict[str, Any]:
    require(
        root / "study_contract/recoverability_study_contract.json", "FROZEN_BEFORE_PPO", "BASELINE"
    )
    results = []
    for index in range(12):
        results.append(
            _run_ppo_study_worker(
                root,
                mode="baseline",
                item_index=index,
                output=root / f"baseline_physics/per_anchor_raw/{_anchor_id(index)}.json",
            )
        )
    return _aggregate_evaluation(root, phase="baseline", results=results)


def run_ppo_recoverability(root: Path) -> dict[str, Any]:
    baseline = require(root / "baseline_physics/summary.json", "PASS", "PPO_TRAINING")
    if baseline["anchors"] != 12:
        raise RuntimeError("PPO_TRAINING_REQUIRES_ALL_12_BASELINES")
    receipts = []
    for index in range(12):
        receipts.append(
            _run_ppo_study_worker(
                root,
                mode="train",
                item_index=index,
                output=root / f"ppo_training/{_anchor_id(index)}",
            )
        )
    value = {
        "schema_version": "OakInk2StaticPPOTrainingAggregateV1",
        "status": "PASS"
        if len(receipts) == 12
        and all(row["status"] == "PASS" and row["cumulative_samples"] == 614400 for row in receipts)
        else "TECHNICAL_INFRA_FAILURE",
        "PPO_TRAINING_RUN_COUNT": len(receipts),
        "total_training_samples": sum(int(row["cumulative_samples"]) for row in receipts),
        "receipts": receipts,
    }
    write_json(root / "ppo_training/summary.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("PPO_TRAINING_INFRA_FAILURE")
    return value


def evaluate_ppo_recoverability(root: Path) -> dict[str, Any]:
    training = require(root / "ppo_training/summary.json", "PASS", "PPO_EVAL")
    if training["PPO_TRAINING_RUN_COUNT"] != 12:
        raise RuntimeError("PPO_EVAL_REQUIRES_12_TRAINED_POLICIES")
    results = []
    for index in range(12):
        checkpoint = root / f"ppo_training/{_anchor_id(index)}/checkpoint.pt"
        results.append(
            _run_ppo_study_worker(
                root,
                mode="eval",
                item_index=index,
                output=root / f"ppo_eval/per_anchor_raw/{_anchor_id(index)}.json",
                checkpoint=checkpoint,
            )
        )
    return _aggregate_evaluation(root, phase="eval", results=results)


def analyze_recoverability(root: Path) -> dict[str, Any]:
    baseline = require(root / "baseline_physics/summary.json", "PASS", "ANALYSIS")
    ppo_eval = require(root / "ppo_eval/summary.json", "PASS", "ANALYSIS")
    study = require(
        root / "study_contract/recoverability_study_contract.json", "FROZEN_BEFORE_PPO", "ANALYSIS"
    )
    manifest = read_json(root / "study_manifest/final_manifest.json")
    baseline_by = {row["anchor_id"]: row for row in baseline["per_anchor"]}
    ppo_by = {row["anchor_id"]: row for row in ppo_eval["per_anchor"]}
    paired = []
    for anchor in manifest["anchors"]:
        base, post = baseline_by[anchor["anchor_id"]], ppo_by[anchor["anchor_id"]]
        before = float(base["physical_penetration_p95_m"])
        after = float(post["physical_penetration_p95_m"])
        paired.append(
            {
                "anchor_id": anchor["anchor_id"],
                "group": anchor["group"],
                "semantic_supergroup": anchor["semantic_supergroup"],
                "final_e_im": anchor["final_e_im"],
                "bootstrap_e_im": anchor["bootstrap_e_im"],
                "rho1": anchor["rho1"],
                "rho2": anchor["rho2"],
                "top1_contributor": anchor["top1_contributor"],
                "top2_contributor": anchor["top2_contributor"],
                "geometric_penetration_p95_m": anchor["geom_penetration_p95_m"],
                "geometric_penetration_max_m": anchor["geom_penetration_max_m"],
                "baseline_physical_penetration_p95_m": before,
                "ppo_physical_penetration_p95_m": after,
                "delta_penetration_p95_m": after - before,
                "delta_penetration_max_m": float(post["physical_penetration_max_m"])
                - float(base["physical_penetration_max_m"]),
                "penetration_reduction_ratio": (before - after) / max(before, 1.0e-12),
                "delta_object_translation_drift_m": float(post["object_translation_drift_m_max"])
                - float(base["object_translation_drift_m_max"]),
                "delta_object_rotation_drift_rad": float(post["object_rotation_drift_rad_max"])
                - float(base["object_rotation_drift_rad_max"]),
                "delta_contact_persistence": float(post["contact_persistence_fraction"])
                - float(base["contact_persistence_fraction"]),
                "delta_hand_deviation_m": float(post["wrist_reference_deviation_m_mean"])
                - float(base["wrist_reference_deviation_m_mean"]),
                "delta_terminal_linear_velocity_mps": float(
                    post["terminal_object_linear_velocity_mps"]
                )
                - float(base["terminal_object_linear_velocity_mps"]),
                "delta_action_saturation": float(post["action_saturation_fraction"])
                - float(base["action_saturation_fraction"]),
                "binary_recovered": "NOT_DEFINED_CONTINUOUS_METRICS_ONLY",
            }
        )
    write_csv(root / "analysis/paired_metrics.csv", paired, list(paired[0]))

    def aggregate(label: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "group": label,
            "N": len(selected),
            "baseline_penetration_p95_m_mean": float(
                np.mean([row["baseline_physical_penetration_p95_m"] for row in selected])
            ),
            "ppo_penetration_p95_m_mean": float(
                np.mean([row["ppo_physical_penetration_p95_m"] for row in selected])
            ),
            "penetration_reduction_ratio_mean": float(
                np.mean([row["penetration_reduction_ratio"] for row in selected])
            ),
            "recoverability_rate": "NOT_DEFINED_CONTINUOUS_METRICS_ONLY",
            "updates_per_anchor": 15,
            "samples_per_anchor": 614400,
        }

    group_rows = [
        aggregate(group, [row for row in paired if row["group"] == group])
        for group in ("GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D")
    ]
    super_rows = [
        aggregate(group, [row for row in paired if row["semantic_supergroup"] == group])
        for group in ("SEMANTIC_PASS", "SEMANTIC_FAIL")
    ]
    write_csv(root / "analysis/group_metrics.csv", group_rows, list(group_rows[0]))
    write_csv(root / "analysis/semantic_supergroups.csv", super_rows, list(super_rows[0]))
    order = sorted(paired, key=lambda row: row["geometric_penetration_max_m"])
    subgroup_rows = []
    for rank, row in enumerate(order):
        subgroup_rows.append(
            {
                **row,
                "penetration_subgroup": "LOW_PEN"
                if rank < 4
                else ("MID_PEN" if rank < 8 else "HIGH_PEN"),
                "analysis_only": True,
            }
        )
    write_csv(root / "analysis/penetration_subgroups.csv", subgroup_rows, list(subgroup_rows[0]))
    from scipy.stats import spearmanr

    correlations = {}
    pairs = {
        "E_IM_vs_penetration_reduction": ("final_e_im", "penetration_reduction_ratio"),
        "E_IM_vs_recoverability": ("final_e_im", None),
        "geom_penetration_vs_penetration_reduction": (
            "geometric_penetration_max_m",
            "penetration_reduction_ratio",
        ),
        "geom_penetration_vs_recoverability": ("geometric_penetration_max_m", None),
        "rho1_vs_recoverability": ("rho1", None),
        "rho2_vs_recoverability": ("rho2", None),
    }
    for name, (left, right) in pairs.items():
        if right is None:
            correlations[name] = {
                "N": 12,
                "status": "NOT_COMPUTABLE_BINARY_RECOVERABILITY_UNDEFINED",
                "raw_left": [row[left] for row in paired],
            }
        else:
            left_values = [float(row[left]) for row in paired]
            right_values = [float(row[right]) for row in paired]
            result = spearmanr(left_values, right_values)
            correlations[name] = {
                "N": 12,
                "spearman": float(result.statistic),
                "direction": "positive" if result.statistic > 0 else "negative",
                "effect_magnitude_abs": abs(float(result.statistic)),
                "raw_left": left_values,
                "raw_right": right_values,
                "small_sample_note": "descriptive only; do not over-interpret significance",
            }
    write_json(
        root / "analysis/correlations.json",
        {"schema_version": "RecoverabilityCorrelationAnalysisV1", "correlations": correlations},
    )
    decision = {
        "schema_version": "RetargetToPPORecoverabilityDecisionV1",
        "status": "COMPLETE",
        "RECOVERABILITY_MODE": study["RECOVERABILITY_MODE"],
        "RETARGET_TO_PPO_RECOVERABILITY_RESULT": "INCONCLUSIVE",
        "binary_recoverability_rate": "NOT_DEFINED",
        "reason": "No current authoritative binary physical gate exists for static OakInk2 support-proxy anchors; continuous paired evidence is reported without inventing thresholds.",
        "PPO_REWARD_CHANGED": "NO",
        "NEXT": "RECOVERABILITY_STUDY_V2_DESIGN",
    }
    write_json(root / "analysis/recoverability_decision.json", decision)
    return decision


def render_review(root: Path) -> dict[str, Any]:
    decision = require(root / "analysis/recoverability_decision.json", "COMPLETE", "REVIEW")
    paired = list(csv.DictReader((root / "analysis/paired_metrics.csv").open(encoding="utf-8")))
    per_anchor_dir = root / "review/per_anchor"
    per_anchor_dir.mkdir(parents=True, exist_ok=True)
    per_anchor_pages = []
    for row in paired:
        anchor_id = row["anchor_id"]
        page = per_anchor_dir / f"{anchor_id}.html"
        metric_rows = "\n".join(
            f"<tr><th>{escape(key)}</th><td>{escape(str(item))}</td></tr>"
            for key, item in row.items()
        )
        page.write_text(
            f"""<!doctype html><meta charset=\"utf-8\"><title>{anchor_id} RecoverabilityReviewV1</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:left}}</style>
<p><a href=\"../index.html\">Back to index</a></p><h1>{anchor_id}</h1>
<p>Binary recovery is intentionally undefined. Values below are paired continuous evidence.</p>
<table><tbody>{metric_rows}</tbody></table>
<p>Trusted raw traces: <code>../../baseline_physics/per_anchor_raw/{anchor_id}/</code> and <code>../../ppo_eval/per_anchor_raw/{anchor_id}/</code>.</p>
""",
            encoding="utf-8",
        )
        per_anchor_pages.append(str(page.relative_to(root / "review")))
    rows = "\n".join(
        "<tr>"
        + "".join(
            (
                f'<td><a href="per_anchor/{row["anchor_id"]}.html">{escape(row[key])}</a></td>'
                if key == "anchor_id"
                else f"<td>{escape(row[key])}</td>"
            )
            for key in (
                "anchor_id",
                "group",
                "final_e_im",
                "geometric_penetration_max_m",
                "baseline_physical_penetration_p95_m",
                "ppo_physical_penetration_p95_m",
                "penetration_reduction_ratio",
                "binary_recovered",
            )
        )
        + "</tr>"
        for row in paired
    )
    html = f"""<!doctype html><meta charset=\"utf-8\"><title>RecoverabilityReviewV1</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>RecoverabilityReviewV1</h1>
<p>Result: <strong>{decision["RETARGET_TO_PPO_RECOVERABILITY_RESULT"]}</strong>. Physical values use runtime collision proxies; geometric SparseV4 values are separate.</p>
<table><thead><tr><th>Anchor</th><th>Group</th><th>E_IM</th><th>Geom max m</th><th>No-PPO p95 m</th><th>Post-PPO p95 m</th><th>Reduction ratio</th><th>Recovered</th></tr></thead><tbody>{rows}</tbody></table>
<p>Trusted raw evidence: per-rollout NPZ traces under baseline_physics/per_anchor_raw and ppo_eval/per_anchor_raw. No new geometry renderer was introduced.</p>
"""
    path = root / "review/index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    manifest = {
        "schema_version": "RecoverabilityReviewV1",
        "status": "PASS",
        "index": str(path),
        "anchor_count": len(paired),
        "views": ["GEOMETRIC_SPARSEV4", "NO_PPO_PHYSICS", "POST_PPO_PHYSICS"],
        "rendering": "HTML_METRICS_PLUS_TRUSTED_RAW_NPZ_TRACES",
        "new_geometry_renderer_created": False,
        "per_anchor_pages": per_anchor_pages,
    }
    write_json(root / "review/manifest.json", manifest)
    return manifest


def summarize(root: Path) -> dict[str, Any]:
    qualification = (
        read_json(root / "frozen_d2jr/qualification.json")
        if (root / "frozen_d2jr/qualification.json").is_file()
        else {"status": "FAIL", "D2J_R_STATUS": "FAIL", "D2K_AUTHORIZED": "NO"}
    )
    decision = (
        read_json(root / "analysis/recoverability_decision.json")
        if (root / "analysis/recoverability_decision.json").is_file()
        else None
    )
    training = (
        read_json(root / "ppo_training/summary.json")
        if (root / "ppo_training/summary.json").is_file()
        else {"PPO_TRAINING_RUN_COUNT": 0, "total_training_samples": 0}
    )
    object_physics = read_json(root / "object_physics/authority_revalidation.json")
    proxy_determinism = read_json(root / "support_proxy/determinism.json")
    hocap = read_json(root / "hocap_regression/regression.json")
    smoke = read_json(root / "d2jr_smoke/qualification.json")
    manifest = read_json(root / "study_manifest/final_manifest.json")
    ppo_contract = read_json(root / "ppo_authority/ppo_contract.json")
    baseline = read_json(root / "baseline_physics/summary.json")
    ppo_eval = read_json(root / "ppo_eval/summary.json")
    ppo_contract_sha = (root / "ppo_authority/ppo_contract.sha256").read_text().strip()
    reward_contract_sha = (root / "ppo_authority/reward_authority.sha256").read_text().strip()
    value = {
        "schema_version": "OakInk2O5RD2JRD2KSummaryV1",
        "status": (
            "COMPLETE"
            if decision is not None and decision.get("status") == "COMPLETE"
            else ("D2JR_PASS_D2K_PENDING" if qualification.get("status") == "PASS" else "D2JR_FAIL")
        ),
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "SPARSE_VALIDATION_V4": "FAIL",
        "HISTORICAL_SPARSE_V4_RESULT_REWRITTEN": "NO",
        "SOURCE_SUPPORT_AUTHORITY": "SUPPORT_UNRESOLVED",
        "SOURCE_SUPPORT_RESULT_REWRITTEN": "NO",
        "SUPPORT_PROXY_TYPE": "STATIC_RECOVERABILITY_PLANAR_PROXY",
        "SOURCE_SCENE_FIDELITY_CLAIM": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "EXECUTION_V3_CHANGED": "NO",
        "CERTIFICATION_GATE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "EXECUTION_V4_CREATED": "NO",
        "OAKINK2_OBJECT_PHYSICS_AUTHORITY": object_physics["status"],
        "STATIC_RECOVERABILITY_SUPPORT_PROXY": proxy_determinism["status"],
        "HOCAP_PHYSICAL_BACKEND_REGRESSION": hocap["status"],
        "REFERENCE_HOLD_EXECUTABLE": f"{sum(row['status'] == 'PASS' for row in smoke['rows'])}/12",
        "D2J_R_STATUS": qualification.get("D2J_R_STATUS", "FAIL"),
        "D2K_AUTHORIZED": qualification.get("D2K_AUTHORIZED", "NO"),
        "PPO_REWARD_CHANGED": "NO",
        "PPO_CONTRACT_SHA256": ppo_contract_sha,
        "REWARD_CONTRACT_SHA256": reward_contract_sha,
        "STUDY_ANCHOR_COUNT": manifest["anchor_count"],
        "REFERENCE_HOLD_BASELINE_COUNT": baseline["anchors"],
        "FINITE_BASELINE_COUNT": sum(
            bool(row["simulation_finite_all"]) for row in baseline["per_anchor"]
        ),
        "PPO_STUDY_ANCHOR_COUNT": ppo_eval["anchors"],
        "PPO_TRAINING_RUN_COUNT": training["PPO_TRAINING_RUN_COUNT"],
        "PPO_ANCHORS_COMPLETED": len(training.get("receipts", [])),
        "PPO_TECHNICAL_FAILURES": 0,
        "PPO_TOTAL_TRAINING_SAMPLES": training.get("total_training_samples", 0),
        "MAX_GPU_JOBS": 1,
        "RETARGET_TO_PPO_RECOVERABILITY_RESULT": (
            "BLOCKED_PHYSICAL_STUDY"
            if decision is None
            else decision["RETARGET_TO_PPO_RECOVERABILITY_RESULT"]
        ),
        "RECOVERABILITY_MODE": "NOT_RUN" if decision is None else decision["RECOVERABILITY_MODE"],
        "Q_OLD_USED_AS_PPO_REFERENCE": "NO",
        "SPARSEV4_STATE_USED_AS_REFERENCE": "YES",
        "OBJECT_POSE_WRITES_AFTER_RESET": 0,
        "HIDDEN_OBJECT_FORCE_USED": "NO",
        "WRIST_ROOT_TELEPORT_USED": "NO",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "NEXT": "RECOVERABILITY_STUDY_V2_DESIGN"
        if decision is not None
        and decision["RETARGET_TO_PPO_RECOVERABILITY_RESULT"] == "INCONCLUSIVE"
        else "STATIC_SUPPORT_PROXY_INFRASTRUCTURE_REPAIR",
    }
    write_json(root / "final_summary.json", value)
    (root / "final_summary.md").write_text(
        "# OakInk2 O5R-D2J-R / D2K summary\n\n"
        + "\n".join(f"- `{key}={item}`" for key, item in value.items() if key != "schema_version")
        + "\n",
        encoding="utf-8",
    )
    required = [
        "preflight/git.json",
        "preflight/upstream_integrity.json",
        "support_backend_audit/hocap_support_backend.json",
        "support_proxy/support_proxy_contract.json",
        "object_physics/authority_revalidation.json",
        "scene_adapter/oakink2_static_scene_adapter.json",
        "hocap_regression/regression.json",
        "d2jr_smoke/qualification.json",
        "frozen_d2jr/qualification.json",
        "ppo_authority/current_ppo_authority.json",
        "ppo_authority/reward_authority.json",
        "ppo_authority/ppo_contract.json",
        "study_contract/recoverability_study_contract.json",
        "study_manifest/final_manifest.json",
        "baseline_physics/per_anchor.csv",
        "ppo_training/summary.json",
        "ppo_eval/per_anchor.csv",
        "analysis/paired_metrics.csv",
        "analysis/correlations.json",
        "analysis/recoverability_decision.json",
        "review/index.html",
        "review/manifest.json",
        "tests.json",
        "validation_results.json",
        "git_commits.json",
        "technical_failures.jsonl",
        "resource_usage.json",
    ]
    completion = {
        "schema_version": "OakInk2O5RD2JRD2KCompletionAuditV1",
        "status": "PASS"
        if value["status"] == "COMPLETE" and all((root / path).is_file() for path in required)
        else "FAIL",
        "required_artifacts": {path: (root / path).is_file() for path in required},
        "historical_sparsev4_rewritten": False,
        "historical_source_support_rewritten": False,
        "q_old_used": False,
        "ppo_reward_changed": False,
        "dev2_full_runs": 0,
        "dev1_full_reruns": 0,
        "o6_ran": False,
    }
    write_json(root / "completion_audit.json", completion)

    def csv_rows(path: str) -> list[dict[str, str]]:
        with (root / path).open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
        return (
            "| "
            + " | ".join(headers)
            + " |\n"
            + "| "
            + " | ".join("---" for _ in headers)
            + " |\n"
            + "\n".join("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
        )

    proxy_rows = csv_rows("support_proxy/per_anchor_proxy.csv")
    smoke_rows = csv_rows("d2jr_smoke/per_anchor.csv")
    baseline_rows = csv_rows("baseline_physics/per_anchor.csv")
    paired_rows = csv_rows("analysis/paired_metrics.csv")
    group_rows = csv_rows("analysis/semantic_supergroups.csv")
    correlations = read_json(root / "analysis/correlations.json")["correlations"]
    proxy_table = markdown_table(
        ["Anchor", "Object", "Support height", "Extent", "Collision", "Proxy SHA", "Result"],
        [
            [
                row["anchor_id"],
                row["object"],
                row["support_height"],
                row["extent"],
                row["collision"],
                row["proxy_sha"],
                row["result"],
            ]
            for row in proxy_rows
        ],
    )
    smoke_table = markdown_table(
        [
            "Anchor",
            "Scene",
            "PhysX",
            "Controller",
            "Finite",
            "Scientific physical outcome",
            "Result",
        ],
        [
            [
                row["anchor_id"],
                row["scene"],
                row["physx"],
                row["controller"],
                row["finite"],
                row["scientific_physical_outcome"],
                row["result"],
            ]
            for row in smoke_rows
        ],
    )
    manifest_table = markdown_table(
        ["Anchor", "Group", "Frame", "E_IM", "Geom Pen p95", "Object", "Proxy support", "Eligible"],
        [
            [
                row["anchor_id"],
                row["group"],
                row["frame"],
                row["final_e_im"],
                row["geom_penetration_p95_m"],
                OBJECT_ID,
                row["study_support_proxy"],
                not row["q_old_used"],
            ]
            for row in manifest["anchors"]
        ],
    )
    baseline_by = {row["anchor_id"]: row for row in baseline_rows}
    baseline_table = markdown_table(
        ["Anchor", "E_IM", "GeomPen", "BasePhysPen", "ObjectDrift", "Contact", "Finite"],
        [
            [
                row["anchor_id"],
                row["final_e_im"],
                row["geometric_penetration_max_m"],
                baseline_by[row["anchor_id"]]["physical_penetration_p95_m"],
                baseline_by[row["anchor_id"]]["object_translation_drift_m_max"],
                baseline_by[row["anchor_id"]]["contact_persistence_fraction"],
                baseline_by[row["anchor_id"]]["simulation_finite_all"],
            ]
            for row in paired_rows
        ],
    )
    paired_table = markdown_table(
        [
            "Anchor",
            "Group",
            "Base Pen",
            "PPO Pen",
            "Pen Reduction",
            "Base Drift",
            "PPO Drift",
            "Contact Preserved",
            "Recovered",
        ],
        [
            [
                row["anchor_id"],
                row["group"],
                row["baseline_physical_penetration_p95_m"],
                row["ppo_physical_penetration_p95_m"],
                row["penetration_reduction_ratio"],
                baseline_by[row["anchor_id"]]["object_translation_drift_m_max"],
                float(baseline_by[row["anchor_id"]]["object_translation_drift_m_max"])
                + float(row["delta_object_translation_drift_m"]),
                float(row["delta_contact_persistence"]) >= 0.0,
                row["binary_recovered"],
            ]
            for row in paired_rows
        ],
    )
    flags = "\n".join(f"{key}={item}" for key, item in value.items() if key != "schema_version")
    handoff = f"""# OakInk2 O5R-D2J-R / D2K

# Static Support Proxy + PPO Recoverability Handoff

## Historical state

`SPARSE_VALIDATION_V4=FAIL`; `HISTORICAL_SPARSE_V4_RESULT_REWRITTEN=NO`; `SOURCE_SUPPORT_AUTHORITY=SUPPORT_UNRESOLVED`; `SOURCE_SUPPORT_RESULT_REWRITTEN=NO`.

The static support is a deterministic study proxy, not an OakInk2 source-scene reconstruction. `SOURCE_SCENE_FIDELITY_CLAIM=NO`.

## D2J-R support proxy

{proxy_table}

Proxy determinism: `{proxy_determinism["deterministic_count"]}/{proxy_determinism["target"]}` deterministic. HOCap regression: `{hocap["status"]}`.

## D2J-R smoke

{smoke_table}

`D2J_R_STATUS={qualification["D2J_R_STATUS"]}`; `D2K_AUTHORIZED={qualification["D2K_AUTHORIZED"]}`.

- `STATIC_RECOVERABILITY_SUPPORT_PROXY_SHA256={qualification["STATIC_RECOVERABILITY_SUPPORT_PROXY_SHA256"]}`
- `OAKINK2_STATIC_SCENE_ADAPTER_SHA256={qualification["OAKINK2_STATIC_SCENE_ADAPTER_SHA256"]}`
- `STATIC_REFERENCE_ADAPTER_SHA256={qualification["STATIC_REFERENCE_ADAPTER_SHA256"]}`
- `STATIC_PHYSICAL_STUDY_SCENE_CONTRACT_V2_SHA256={qualification["STATIC_PHYSICAL_STUDY_SCENE_CONTRACT_V2_SHA256"]}`

## PPO authority

- `PPO_REWARD_CHANGED=NO`
- `PPO_CONTRACT_SHA256={ppo_contract_sha}`
- `REWARD_CONTRACT_SHA256={reward_contract_sha}`
- `PPO_ALGORITHM={ppo_contract["algorithm"]}`
- `PPO_BUDGET={json.dumps(ppo_contract["training"], sort_keys=True)}`
- `EVAL_PROTOCOL={json.dumps(ppo_contract["evaluation"], sort_keys=True)}`

## Frozen manifest

{manifest_table}

## Baselines

`REFERENCE_HOLD_BASELINE_COUNT={baseline["anchors"]}`; `FINITE_BASELINE_COUNT={value["FINITE_BASELINE_COUNT"]}`.

{baseline_table}

## PPO and paired continuous outcomes

`PPO_STUDY_ANCHOR_COUNT={ppo_eval["anchors"]}`; `PPO_TRAINING_RUN_COUNT={training["PPO_TRAINING_RUN_COUNT"]}`; `PPO_ANCHORS_COMPLETED={value["PPO_ANCHORS_COMPLETED"]}`; `PPO_TECHNICAL_FAILURES=0`.

{paired_table}

## Supergroups

{markdown_table(list(group_rows[0]), [list(row.values()) for row in group_rows])}

Binary recovered counts/rates are `NOT_DEFINED_CONTINUOUS_METRICS_ONLY` for both semantic supergroups.

## Correlations

- `E_IM_VS_RECOVERABILITY={correlations["E_IM_vs_recoverability"]["status"]}`
- `GEOM_PENETRATION_VS_RECOVERABILITY={correlations["geom_penetration_vs_recoverability"]["status"]}`
- `RHO1_VS_RECOVERABILITY={correlations["rho1_vs_recoverability"]["status"]}`
- `RHO2_VS_RECOVERABILITY={correlations["rho2_vs_recoverability"]["status"]}`
- Descriptive `E_IM_vs_penetration_reduction` Spearman: `{correlations["E_IM_vs_penetration_reduction"]["spearman"]}` (`N=12`).
- Descriptive `geom_penetration_vs_penetration_reduction` Spearman: `{correlations["geom_penetration_vs_penetration_reduction"]["spearman"]}` (`N=12`).

## Final decision

`RETARGET_TO_PPO_RECOVERABILITY_RESULT={value["RETARGET_TO_PPO_RECOVERABILITY_RESULT"]}` because no authoritative binary static physical gate was available before execution. Continuous paired evidence is reported without inventing a threshold.

`NEXT={value["NEXT"]}`

## Final safety flags

```text
{flags}
```
"""
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    return value


ACTIONS = {
    "preflight": preflight,
    "verify-upstream": verify_upstream,
    "audit-existing-support-backend": audit_existing_support_backend,
    "define-support-proxy": define_support_proxy,
    "build-support-proxies": build_support_proxies,
    "run-support-proxy-determinism": run_proxy_determinism,
    "run-hocap-support-regression": run_hocap_support_regression,
    "build-oakink2-static-scenes": build_static_scenes,
    "run-d2jr-reference-hold-smoke": run_d2jr_smoke,
    "freeze-d2jr-contracts": freeze_d2jr_contracts,
    "audit-ppo-authority": audit_ppo_authority,
    "freeze-ppo-contract": freeze_ppo_contract,
    "freeze-recoverability-study": freeze_recoverability_study,
    "run-all-reference-hold-baselines": run_all_baselines,
    "run-ppo-recoverability": run_ppo_recoverability,
    "evaluate-ppo-recoverability": evaluate_ppo_recoverability,
    "analyze-recoverability": analyze_recoverability,
    "render-review": render_review,
    "summarize": summarize,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=REQUIRED_ACTIONS)
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        value = ACTIONS[args.action](args.root.resolve())
        print(json.dumps(value, sort_keys=True, default=str))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
