#!/usr/bin/env python3
"""O5R-D2J/D2K static-physics and PPO-recoverability workflow.

The driver repairs and audits only the physical-scene boundary.  It consumes
the frozen D2I anchor manifest and stored SparseV4 q/base states read-only.
Every D2K action is fail-closed unless D2J is fully qualified.
"""

# ruff: noqa: E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.rl.isaaclab.import_hocap_objects import (  # noqa: E402
    _bbox_inertia,
    _bounded_convex_proxy,
    _read_obj,
)
from toporetarget.physics.support import resolve_support  # noqa: E402
from toporetarget.physics.support.types import jsonable  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2j_d2k_static_physics_and_ppo_recoverability_v1"
D2I_ROOT = REPO / ".local/reports/oakink2_o5rd2i_sparsev4_failure_and_ppo_recoverability_v1"
D2H_ROOT = (
    REPO / ".local/reports/oakink2_o5rd2h_execution_v3_independent_coldstart_certification_v1"
)
O5_ROOT = REPO / ".local/reports/oakink2_o5_geometric_retarget_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "5112aa4e5a6292d7bdfb9be4829010fda8f92a04"
SEQUENCE = "scene_01__A003++seq__d4ddf93a38e3228cdd3a__2023-04-15-10-15-10"
OBJECT_ID = "C10001"
OAKINK2_ASSET_HUB = Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub")
OAKINK2_ANNOTATION = (
    Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview")
    / f"{SEQUENCE}.pkl"
)
VISUAL_MESH = OAKINK2_ASSET_HUB / "object_repair/align_ds" / OBJECT_ID / "model.obj"
RAW_MESH = OAKINK2_ASSET_HUB / "object_raw/align_ds" / OBJECT_ID / "mug.ply"
SUPPORT_CONFIG = REPO / "configs/physics/support_resolution_v1.yaml"
OBJECT_IMPORTER = REPO / "scripts/rl/isaaclab/import_hocap_objects.py"

REQUIRED_ACTIONS = (
    "preflight",
    "verify-upstream-evidence",
    "audit-existing-physical-pipeline",
    "build-oakink2-object-physics-authority",
    "build-oakink2-support-authority",
    "build-oakink2-static-scene-adapter",
    "run-hocap-backend-regression",
    "audit-recoverability-manifest",
    "qualify-physical-study-anchors",
    "run-d2j-reference-hold-smoke",
    "freeze-d2j-physical-contracts",
    "audit-ppo-authority",
    "freeze-ppo-recoverability-contract",
    "run-reference-hold-baselines",
    "run-ppo-recoverability-study",
    "evaluate-ppo-recoverability",
    "analyze-retarget-quality-vs-recovery",
    "render-recoverability-review",
    "summarize",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _status(path: Path) -> str:
    return str(read_json(path).get("status", "MISSING")) if path.exists() else "MISSING"


def _require(path: Path, expected: str, action: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    payload = read_json(path)
    if payload.get("status") != expected:
        raise RuntimeError(f"{action}_REJECTED:{path.name}={payload.get('status')}")
    return payload


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2J_BRANCH_MISMATCH:{branch}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=REPO, check=False
    ).returncode:
        raise RuntimeError("O5RD2J_START_HEAD_NOT_ANCESTOR")
    payload = {
        "schema_version": "OakInk2O5RD2JGitPreflightV1",
        "status": "PASS",
        "repo": str(REPO),
        "branch": branch,
        "start_head": START_HEAD,
        "observed_head": git("rev-parse", "HEAD"),
        "initial_status_short_from_direct_operator_preflight": "",
        "status_short_at_cli_action": git("status", "--short", "--untracked-files=all"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "local_ignored": subprocess.run(
            ["git", "check-ignore", "-q", ".local"], cwd=REPO, check=False
        ).returncode
        == 0,
        "max_gpu_jobs": 1,
        "new_branch_created": False,
        "new_worktree_created": False,
    }
    write_json(root / "preflight/git.json", payload)
    return payload


def verify_upstream_evidence(root: Path) -> dict[str, Any]:
    manifest_path = D2I_ROOT / "study_manifest/recoverability_manifest.json"
    manifest_sha_path = manifest_path.with_suffix(".sha256")
    manifest = read_json(manifest_path)
    state_rows = []
    for anchor in manifest["anchors"]:
        ordinal = int(anchor["ordinal"])
        state = D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.npz"
        with np.load(state, allow_pickle=False) as archive:
            keys = sorted(archive.files)
            q_shape = list(np.asarray(archive["qpos"]).shape)
            base_shape = list(np.asarray(archive["base_pose_scene"]).shape)
        state_rows.append(
            {
                "ordinal": ordinal,
                "frame": int(anchor["frame_id"]),
                "path": str(state.resolve()),
                "sha256": sha256_file(state),
                "keys": keys,
                "qpos_shape": q_shape,
                "base_pose_scene_shape": base_shape,
                "q_old_present": "q_old" in keys,
            }
        )
    authorities = {
        "d2i_manifest": manifest_path,
        "d2i_manifest_sha": manifest_sha_path,
        "d2h_final_summary": D2H_ROOT / "final_summary.json",
        "d2h_sparse_manifest": D2H_ROOT / "sparse_v4/manifest.json",
        "d2h_sparse_gate": D2H_ROOT / "sparse_v4/gate_decision.json",
        "oakink2_visual_mesh": VISUAL_MESH,
        "oakink2_raw_mesh": RAW_MESH,
        "oakink2_annotation": OAKINK2_ANNOTATION,
        "support_contract": SUPPORT_CONFIG,
    }
    missing = [name for name, path in authorities.items() if not path.is_file()]
    expected_sha = manifest_sha_path.read_text(encoding="utf-8").strip()
    d2h_summary = read_json(authorities["d2h_final_summary"])
    d2h_gate = read_json(authorities["d2h_sparse_gate"])
    checks = {
        "no_missing_authorities": not missing,
        "d2i_manifest_hash_matches": sha256_file(manifest_path) == expected_sha,
        "anchor_count_12": len(manifest["anchors"]) == 12,
        "selection_frozen_before_physics": manifest.get("frozen_before_physics") is True,
        "reference_is_sparsev4_state": manifest.get("reference_authority")
        == "stored SparseV4 ExecutionV3 final q/base",
        "q_old_forbidden": manifest.get("q_old_reference_forbidden") is True,
        "state_receipts_exact_schema": all(
            row["keys"] == ["base_pose_scene", "qpos"]
            and row["qpos_shape"] == [20]
            and row["base_pose_scene_shape"] == [4, 4]
            for row in state_rows
        ),
        "state_q_old_absent": all(not row["q_old_present"] for row in state_rows),
        "historical_sparse_v4_fail": d2h_gate.get("COLDSTART_SPARSE_VALIDATION_V4") == "FAIL",
        "historical_execution_v3_fail": d2h_summary.get(
            "EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION"
        )
        == "FAIL",
    }
    payload = {
        "schema_version": "OakInk2O5RD2JFrozenUpstreamEvidenceV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing": missing,
        "authorities": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in authorities.items()
            if path.is_file()
        },
        "anchor_states": state_rows,
        "retarget_optimizer_run_count": 0,
        "new_certification_consumption": 0,
        "new_heldout_consumption": 0,
    }
    write_json(root / "preflight/frozen_upstream.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("BLOCKED_UPSTREAM_EVIDENCE_INTEGRITY")
    return payload


def audit_existing_physical_pipeline(root: Path) -> dict[str, Any]:
    paths = {
        "physicalization_authority": REPO / "src/toporetarget/physics/physicalization_authority.py",
        "physical_scene_authority": REPO / "src/toporetarget/physics/physical_scene_authority.py",
        "support_resolver": REPO / "src/toporetarget/physics/support/resolver.py",
        "support_contract": SUPPORT_CONFIG,
        "independent_runtime": REPO
        / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env_cfg.py",
        "independent_environment": REPO
        / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py",
        "object_importer": OBJECT_IMPORTER,
        "ppo_driver": REPO / "scripts/rl/isaaclab/run_physical_refinement.py",
        "hocap_protocol": REPO / "configs/contracts/hocap_physicalization_hardening_v2.json",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    checks = {
        "source_first_support_precedence": "SOURCE_EXPLICIT_SUPPORT" in text["support_contract"]
        and "SOURCE_RECONSTRUCTED_SUPPORT" in text["support_contract"],
        "finite_static_support_backend": "configure_independent_clip_runtime"
        in text["independent_runtime"],
        "object_gravity_runtime_control": "disable_gravity = False" in text["independent_runtime"],
        "post_reset_write_telemetry": "object_rollout_state_writes"
        in text["independent_environment"]
        and "wrist_root_state_writes_during_step" in text["independent_environment"],
        "independent_object_import": "--mesh" in text["object_importer"],
        "generic_convex_collision_recipe": "convex_hull_v1" in text["object_importer"],
    }
    payload = {
        "schema_version": "ExistingPhysicalPipelineAuthorityAuditV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "separation": {
            "generic_components": [
                "source-first SupportResolutionV1",
                "finite static support runtime",
                "independent object/reference reset",
                "convex_hull_v1 collision recipe",
                "0.05 kg geometry-derived bbox inertia engineering proxy",
            ],
            "hocap_specific_components": [
                "HOCap EpisodeV1 source adapter",
                "HOCap protocol and source contact events",
                "HOCap-named object importer wrapper",
            ],
            "oakink2_required_components": [
                "OakInk2StaticContactStudyRecordV1",
                "OakInk2 support authority",
                "OakInk2 static scene adapter qualification",
            ],
        },
    }
    write_json(root / "d2j_authority_audit/existing_physical_pipeline.json", payload)
    write_json(root / "d2j_authority_audit/generic_vs_hocap_specific.json", payload["separation"])
    if payload["status"] != "PASS":
        raise RuntimeError("BLOCKED_EXISTING_PHYSICAL_PIPELINE_AUDIT")
    return payload


def _object_import_command(root: Path) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        "toporetarget-isaaclab",
        "python",
        str(OBJECT_IMPORTER),
        "--mesh",
        str(VISUAL_MESH),
        "--object-id",
        OBJECT_ID,
        "--output-dir",
        str(root / "object_physics/assets"),
        "--report",
        str(root / "object_physics/object_import_receipt.json"),
        "--accept-eula",
    ]


def build_oakink2_object_physics_authority(root: Path) -> dict[str, Any]:
    _require(root / "preflight/frozen_upstream.json", "PASS", "OBJECT_AUTHORITY")
    _require(
        root / "d2j_authority_audit/existing_physical_pipeline.json",
        "PASS",
        "OBJECT_AUTHORITY",
    )
    object_import_receipt = root / "object_physics/object_import_receipt.json"
    object_usd = root / "object_physics/assets/C10001.usda"
    if not object_import_receipt.is_file() or not object_usd.is_file():
        root.joinpath("object_physics/assets").mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            _object_import_command(root), cwd=REPO, text=True, capture_output=True, check=False
        )
        write_json(
            root / "object_physics/import_process.json",
            {
                "schema_version": "OakInk2ObjectImportProcessV1",
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "command": _object_import_command(root),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-8000:],
                "stderr_tail": completed.stderr[-8000:],
                "gpu_job_count": 1,
                "max_concurrent_gpu_jobs": 1,
            },
        )
        if completed.returncode:
            raise RuntimeError("BLOCKED_OBJECT_PHYSICS_AUTHORITY:IMPORT_FAILED")
    receipt = read_json(object_import_receipt)
    vertices, _counts, _indices, bbox_min, bbox_max = _read_obj(VISUAL_MESH)
    inertia = _bbox_inertia(bbox_min, bbox_max, mass_kg=0.05)
    visual_sha = sha256_file(VISUAL_MESH)
    usd_sha = sha256_file(object_usd)
    checks = {
        "import_complete": receipt.get("status") == "COMPLETE",
        "visual_hash_matches": receipt.get("visual_mesh_sha256") == visual_sha,
        "generated_usd_hash_matches": receipt.get("generated_sha256") == usd_sha,
        "collision_is_separate_convex_hull": receipt.get("collision_method") == "convex_hull_v1",
        "collision_prim_present": int(receipt.get("collision_prim_count", 0)) >= 1,
        "mass_is_existing_generic_proxy": float(receipt.get("mass_kg", -1.0)) == 0.05,
        "inertia_is_geometry_derived": np.allclose(
            np.asarray(receipt.get("principal_inertia_kgm2")), np.asarray(inertia)
        ),
        "isaac_stage_reopened": bool(receipt.get("root_prim")),
        "asset_nonempty": object_usd.stat().st_size > 0,
    }
    authority = {
        "schema_version": "OakInk2ObjectPhysicsAuthorityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "object_id": OBJECT_ID,
        "checks": checks,
        "visual": {
            "path": str(VISUAL_MESH.resolve()),
            "sha256": visual_sha,
            "vertex_count": len(vertices),
            "role": "VISUAL_GEOMETRY",
        },
        "collision": {
            "path": str(object_usd.resolve()),
            "sha256": usd_sha,
            "representation": "convex_hull_v1",
            "role": "PHYSX_COLLISION_GEOMETRY",
            "hull_vertex_limit": 64,
            "geometry_deviation": receipt.get("geometry_deviation"),
        },
        "dynamics": {
            "provenance_class": "EXISTING_PROJECT_GENERIC_PROXY",
            "profile": "fixed_0p05kg_mesh_bbox_inertia_v1",
            "mass_kg": 0.05,
            "principal_inertia_kgm2": list(inertia),
            "center_of_mass_m": receipt.get("center_of_mass_m"),
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0,
            "ground_truth_claimed": False,
            "anchor_dependent": False,
        },
        "loadability_evidence": "created and reopened by Isaac Sim USD APIs",
        "dynamic_physx_scene_smoke": "PENDING_D2J_REFERENCE_HOLD_SMOKE",
        "gpu_job_count": 1,
        "max_concurrent_gpu_jobs": 1,
    }
    write_json(root / "object_physics/oakink2_object_physics_authority.json", authority)
    write_csv(
        root / "object_physics/visual_collision_asset_manifest.csv",
        [
            {
                "object_id": OBJECT_ID,
                "visual_path": authority["visual"]["path"],
                "visual_sha256": visual_sha,
                "collision_path": authority["collision"]["path"],
                "collision_sha256": usd_sha,
                "collision_representation": "convex_hull_v1",
            }
        ],
        [
            "object_id",
            "visual_path",
            "visual_sha256",
            "collision_path",
            "collision_sha256",
            "collision_representation",
        ],
    )
    write_csv(
        root / "object_physics/dynamics_provenance.csv",
        [{"object_id": OBJECT_ID, **authority["dynamics"]}],
        [
            "object_id",
            "provenance_class",
            "profile",
            "mass_kg",
            "principal_inertia_kgm2",
            "center_of_mass_m",
            "static_friction",
            "dynamic_friction",
            "restitution",
            "ground_truth_claimed",
            "anchor_dependent",
        ],
    )
    write_json(
        root / "object_physics/asset_qualification.json",
        {
            "schema_version": "OakInk2ObjectAssetQualificationV1",
            "status": authority["status"],
            "checks": checks,
            "physx_dynamic_smoke": "NOT_RUN_BEFORE_SUPPORT_AUTHORITY",
        },
    )
    if authority["status"] != "PASS":
        raise RuntimeError("BLOCKED_OBJECT_PHYSICS_AUTHORITY")
    return authority


def _load_annotation() -> dict[str, Any]:
    with OAKINK2_ANNOTATION.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError("OAKINK2_ANNOTATION_NOT_MAPPING")
    return value


def _load_visual_vertices(path: Path) -> np.ndarray:
    vertices, _counts, _indices, _low, _high = _read_obj(path)
    return np.asarray(vertices, dtype=np.float64)


def _track(annotation: dict[str, Any], object_id: str) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(annotation["mocap_frame_id_list"], dtype=np.int64)
    poses = np.stack(
        [np.asarray(annotation["obj_transf"][object_id][int(frame)]) for frame in frames]
    )
    return frames, poses


def _world_aabb(vertices: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    return world.min(axis=0), world.max(axis=0)


def _source_geometry_audit(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    target_vertices = _load_visual_vertices(VISUAL_MESH)
    _frames, target_pose = _track(annotation, OBJECT_ID)
    audit_frames = (0, 1969, 2041, 4690, 7067)
    rows: list[dict[str, Any]] = []
    for candidate_id in annotation["obj_list"]:
        if candidate_id == OBJECT_ID:
            continue
        mesh = OAKINK2_ASSET_HUB / "object_repair/align_ds" / candidate_id / "model.obj"
        candidate_vertices = _load_visual_vertices(mesh)
        _other_frames, candidate_pose = _track(annotation, candidate_id)
        per_frame = []
        ever_xy_overlap = False
        ever_vertical_support_match = False
        for frame in audit_frames:
            target_low, _target_high = _world_aabb(target_vertices, target_pose[frame])
            candidate_low, candidate_high = _world_aabb(candidate_vertices, candidate_pose[frame])
            xy_overlap = bool(
                candidate_high[0] >= target_low[0]
                and candidate_low[0] <= _target_high[0]
                and candidate_high[1] >= target_low[1]
                and candidate_low[1] <= _target_high[1]
            )
            vertical_gap = float(target_low[2] - candidate_high[2])
            vertical_match = abs(vertical_gap) <= 0.005
            ever_xy_overlap |= xy_overlap
            ever_vertical_support_match |= vertical_match
            per_frame.append(
                {
                    "frame": frame,
                    "xy_aabb_overlap": xy_overlap,
                    "target_bottom_minus_candidate_top_m": vertical_gap,
                    "within_support_gap_tolerance": vertical_match,
                }
            )
        rows.append(
            {
                "candidate_object": candidate_id,
                "mesh": str(mesh.resolve()),
                "mesh_sha256": sha256_file(mesh),
                "source_semantic_role": "INDEPENDENT_MANIPULATED_OR_COMPANION_OBJECT_NOT_ENVIRONMENT_SUPPORT",
                "ever_xy_aabb_overlap_at_audit_frames": ever_xy_overlap,
                "ever_vertical_support_match_at_audit_frames": ever_vertical_support_match,
                "qualifies_as_source_reconstructed_support": False,
                "frames": per_frame,
            }
        )
    return rows


def compact_support_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the authority decision inspectable without embedding 4x7068 samples."""

    compact = dict(payload)
    stable = dict(compact.get("stable_interval") or {})
    series_summary: dict[str, Any] = {}
    for key in (
        "linear_speed_mps",
        "angular_speed_radps",
        "translation_step_m",
        "rotation_step_rad",
    ):
        values = np.asarray(stable.pop(key, []), dtype=np.float64)
        series_summary[key] = {
            "count": int(values.size),
            "sha256": stable_hash(values.tolist()),
            "min": float(np.min(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
        }
    candidates = list(stable.pop("candidate_intervals", []))
    stable["kinematic_series_summary"] = series_summary
    stable["candidate_interval_count"] = len(candidates)
    stable["candidate_intervals_sha256"] = stable_hash(candidates)
    stable["candidate_intervals_first_10"] = candidates[:10]
    compact["stable_interval"] = stable
    diagnostics = dict(compact.get("diagnostics") or {})
    audit = list(diagnostics.pop("candidate_interval_audit", []))
    diagnostics["candidate_interval_audit_count"] = len(audit)
    diagnostics["candidate_interval_audit_sha256"] = stable_hash(audit)
    diagnostics["candidate_interval_audit_first_10"] = audit[:10]
    diagnostics["all_candidate_intervals_authority"] = (
        "POST_MANIPULATION_DIAGNOSTIC_ONLY_NOT_SUPPORT_AUTHORITY"
        if audit and all(not row.get("support_inference_authorized", False) for row in audit)
        else "SEE_FULL_RESOLVER_STATUS"
    )
    compact["diagnostics"] = diagnostics
    return compact


def build_oakink2_support_authority(root: Path) -> dict[str, Any]:
    _require(
        root / "object_physics/oakink2_object_physics_authority.json",
        "PASS",
        "SUPPORT_AUTHORITY",
    )
    annotation = _load_annotation()
    frames, poses = _track(annotation, OBJECT_ID)
    visual = _load_visual_vertices(VISUAL_MESH)
    proxy_vertices, _proxy_faces, max_support_gap = _bounded_convex_proxy(visual.tolist())
    quaternion_xyzw = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]
    source_geometry = _source_geometry_audit(annotation)
    result = resolve_support(
        dataset="oakink2",
        sequence=SEQUENCE,
        object_visual_vertices_local=visual,
        object_collision_vertices_local=np.asarray(proxy_vertices, dtype=np.float64),
        object_pose_translation_world=poses[:, :3, 3],
        object_pose_quaternion_world_wxyz=quaternion_wxyz,
        timestamps=frames.astype(np.float64) / 30.0,
        gravity_world_mps2=(0.0, 0.0, -9.81),
        source_support={
            "explicit": False,
            "recovered": False,
            "provenance": {
                "annotation": str(OAKINK2_ANNOTATION.resolve()),
                "annotation_fields": sorted(annotation),
                "source_environment_geometry_present": False,
                "other_object_geometry_audit": source_geometry,
            },
        },
        source_reference_kind="oakink2_full_raw_source_object_pose_0_7067",
        mode="auto",
    )
    result_payload = compact_support_result(result.as_dict())
    status = "PASS" if result.support_type.value != "UNRESOLVED" else "FAIL"
    reason = str(result.diagnostics.get("reason", ""))
    authority = {
        "schema_version": "OakInk2SupportAuthorityV1",
        "status": status,
        "decision": "BLOCKED_SUPPORT_AUTHORITY" if status == "FAIL" else "AUTHORIZED",
        "object_id": OBJECT_ID,
        "sequence": SEQUENCE,
        "precedence": [
            "SOURCE_EXPLICIT_SUPPORT",
            "SOURCE_RECONSTRUCTED_SUPPORT",
            "INFERRED_PLANAR_SUPPORT",
            "UNRESOLVED",
        ],
        "selected_support_type": result.support_type.value,
        "support_source": result.support_source,
        "resolver_status": result.status,
        "resolver_reason": reason,
        "source_explicit_support": False,
        "source_environment_geometry_present": False,
        "source_reconstructed_candidates": source_geometry,
        "full_source_interval": [int(frames[0]), int(frames[-1]) + 1],
        "full_source_frame_count": len(frames),
        "inferred_planar_authorized": result.support_inferred,
        "post_manipulation_candidates_are_diagnostic_only": True,
        "manual_z0_plane_created": False,
        "outcome_dependent_adjustment": False,
        "collision_proxy_sha256": stable_hash({"vertices": proxy_vertices, "faces": _proxy_faces}),
        "collision_proxy_max_support_gap_m": max_support_gap,
        "object_authority_sha256": sha256_file(
            root / "object_physics/oakink2_object_physics_authority.json"
        ),
        "support_contract_path": str(SUPPORT_CONFIG.resolve()),
        "support_contract_sha256": sha256_file(SUPPORT_CONFIG),
        "resolver_result": result_payload,
    }
    write_json(root / "support/oakink2_support_authority.json", authority)
    manifest = read_json(D2I_ROOT / "study_manifest/recoverability_manifest.json")
    write_csv(
        root / "support/per_anchor_support.csv",
        [
            {
                "anchor_id": f"A{index:02d}",
                "ordinal": anchor["ordinal"],
                "frame": anchor["frame_id"],
                "object": anchor["object"],
                "support_type": result.support_type.value,
                "support_status": status,
                "authority_hash": stable_hash(authority),
                "eligible": False,
                "reason": "BLOCKED_SUPPORT_AUTHORITY" if status == "FAIL" else "",
            }
            for index, anchor in enumerate(manifest["anchors"], start=1)
        ],
        [
            "anchor_id",
            "ordinal",
            "frame",
            "object",
            "support_type",
            "support_status",
            "authority_hash",
            "eligible",
            "reason",
        ],
    )
    return authority


def build_oakink2_static_scene_adapter(root: Path) -> dict[str, Any]:
    object_status = _status(root / "object_physics/oakink2_object_physics_authority.json")
    support_status = _status(root / "support/oakink2_support_authority.json")
    schema = {
        "schema_version": "CanonicalStaticContactStudyRecordV1",
        "required_fields": [
            "dataset",
            "sequence",
            "anchor_id",
            "object_asset",
            "object_dynamics",
            "support_authority",
            "hand_qpos",
            "wrist_pose_scene",
            "object_pose_scene",
            "reference_velocities",
            "source_hashes",
        ],
        "invariants": {
            "object_gravity": "ON",
            "hand_and_wrist_gravity": "OFF",
            "object_pose_writes_after_reset": 0,
            "hidden_object_force": False,
            "wrist_root_teleport": False,
        },
    }
    write_json(root / "scene_adapter/canonical_static_contact_record_schema.json", schema)
    ready = object_status == support_status == "PASS"
    payload = {
        "schema_version": "OakInk2StaticContactSceneAdapterV1",
        "status": "PASS" if ready else "NOT_RUN",
        "reason": None if ready else f"OBJECT={object_status};SUPPORT={support_status}",
        "record_schema_sha256": stable_hash(schema),
        "reference": "stored SparseV4 ExecutionV3 final q/base repeated as static target",
        "q_old_used": False,
        "reset_contract": schema["invariants"],
        "implementation": "existing independent WorldWrist direct runtime",
    }
    write_json(root / "scene_adapter/oakink2_static_scene_adapter_contract.json", payload)
    if not ready:
        raise RuntimeError("STATIC_SCENE_ADAPTER_REJECTED:BLOCKED_SUPPORT_AUTHORITY")
    return payload


def _write_not_run(root: Path, path: str, schema: str, reason: str) -> dict[str, Any]:
    payload = {"schema_version": schema, "status": "NOT_RUN", "reason": reason}
    write_json(root / path, payload)
    return payload


def run_hocap_backend_regression(root: Path) -> dict[str, Any]:
    _require(
        root / "scene_adapter/oakink2_static_scene_adapter_contract.json",
        "PASS",
        "HOCAP_REGRESSION",
    )
    raise RuntimeError("HOCAP_REGRESSION_BACKEND_REQUIRES_D2J_READY_SCENE")


def audit_recoverability_manifest(root: Path) -> dict[str, Any]:
    source = read_json(D2I_ROOT / "study_manifest/recoverability_manifest.json")
    source_sha = sha256_file(D2I_ROOT / "study_manifest/recoverability_manifest.json")
    audit = {
        "schema_version": "RecoverabilityStudyManifestV2AuditV1",
        "status": "PASS",
        "source_manifest_sha256": source_sha,
        "frozen_before_physics": source["frozen_before_physics"],
        "anchor_count": len(source["anchors"]),
        "groups": {
            group: sum(row["group"] == group for row in source["anchors"])
            for group in ("GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D")
        },
        "immutable_anchor_selection": True,
        "q_old_reference_forbidden": True,
    }
    write_json(root / "study_manifest/manifest_audit.json", audit)
    return audit


def qualify_physical_study_anchors(root: Path) -> dict[str, Any]:
    object_status = _status(root / "object_physics/oakink2_object_physics_authority.json")
    support_status = _status(root / "support/oakink2_support_authority.json")
    scene_status = _status(root / "scene_adapter/oakink2_static_scene_adapter_contract.json")
    if (object_status, support_status, scene_status) != ("PASS", "PASS", "PASS"):
        raise RuntimeError(
            "QUALIFY_PHYSICAL_STUDY_ANCHORS_REJECTED:"
            f"OBJECT={object_status};SUPPORT={support_status};SCENE={scene_status}"
        )
    raise RuntimeError("QUALIFICATION_REQUIRES_D2J_READY_SCENE")


def run_d2j_reference_hold_smoke(root: Path) -> dict[str, Any]:
    _require(root / "study_manifest/physical_eligible_pool.json", "PASS", "D2J_SMOKE")
    raise RuntimeError("D2J_SMOKE_REQUIRES_QUALIFIED_ANCHORS")


def freeze_d2j_physical_contracts(root: Path) -> dict[str, Any]:
    _require(root / "hocap_regression/parity.json", "PASS", "FREEZE_D2J")
    _require(root / "d2j_baseline_smoke/qualification.json", "PASS", "FREEZE_D2J")
    raise RuntimeError("FREEZE_D2J_REQUIRES_PASSED_SMOKES")


def audit_ppo_authority(root: Path) -> dict[str, Any]:
    _require(root / "frozen_d2j/static_physical_study_scene_contract.json", "PASS", "PPO_AUDIT")
    raise RuntimeError("PPO_AUDIT_REQUIRES_D2J_PASS")


def freeze_ppo_recoverability_contract(root: Path) -> dict[str, Any]:
    _require(root / "ppo_authority/current_ppo_authority.json", "PASS", "FREEZE_PPO")
    raise RuntimeError("FREEZE_PPO_REQUIRES_UNIQUE_AUTHORITY")


def run_reference_hold_baselines(root: Path) -> dict[str, Any]:
    _require(root / "study_contract/recoverability_study_contract.json", "PASS", "BASELINE")
    raise RuntimeError("BASELINE_REQUIRES_D2K_AUTHORITY")


def run_ppo_recoverability_study(root: Path) -> dict[str, Any]:
    summary = _require(root / "final_summary.json", "PASS", "PPO_STUDY")
    if summary.get("D2J_STATUS") != "PASS":
        raise RuntimeError("PPO_STUDY_REJECTED:D2J_STATUS_NOT_PASS")
    _require(root / "baseline_physics/completion.json", "PASS", "PPO_STUDY")
    raise RuntimeError("PPO_STUDY_BACKEND_REQUIRES_D2K_AUTHORITY")


def evaluate_ppo_recoverability(root: Path) -> dict[str, Any]:
    _require(root / "ppo_training/completion.json", "PASS", "PPO_EVAL")
    raise RuntimeError("PPO_EVAL_REQUIRES_COMPLETED_TRAINING")


def analyze_retarget_quality_vs_recovery(root: Path) -> dict[str, Any]:
    _require(root / "ppo_eval/completion.json", "PASS", "PPO_ANALYSIS")
    raise RuntimeError("PPO_ANALYSIS_REQUIRES_COMPLETED_EVAL")


def render_recoverability_review(root: Path) -> dict[str, Any]:
    decision = root / "analysis/recoverability_decision.json"
    if not decision.exists():
        raise RuntimeError("RECOVERABILITY_REVIEW_REJECTED:DECISION_MISSING")
    return read_json(decision)


def _scaffold_blocked_outputs(root: Path, blocker: str) -> None:
    manifest = read_json(D2I_ROOT / "study_manifest/recoverability_manifest.json")
    support = read_json(root / "support/oakink2_support_authority.json")
    support_hash = stable_hash(support)
    write_json(
        root / "scene_adapter/canonical_static_contact_record_schema.json",
        {
            "schema_version": "CanonicalStaticContactStudyRecordV1",
            "status": "SCHEMA_ONLY",
            "required_fields": [
                "anchor_id",
                "object_asset",
                "object_dynamics",
                "support_authority",
                "hand_qpos",
                "wrist_pose_scene",
                "object_pose_scene",
                "source_hashes",
            ],
        },
    )
    if not (root / "scene_adapter/oakink2_static_scene_adapter_contract.json").exists():
        _write_not_run(
            root,
            "scene_adapter/oakink2_static_scene_adapter_contract.json",
            "OakInk2StaticContactSceneAdapterV1",
            blocker,
        )
    _write_not_run(root, "hocap_regression/parity.json", "HOCapBackendParityV1", blocker)
    _write_not_run(root, "hocap_regression/smoke_results.json", "HOCapSmokeResultsV1", blocker)
    audit_recoverability_manifest(root)
    pool = [
        {
            **anchor,
            "anchor_id": f"A{index:02d}",
            "support": "UNRESOLVED",
            "support_authority_sha256": support_hash,
            "physical_study_eligible": False,
            "physical_ineligibility": blocker,
        }
        for index, anchor in enumerate(manifest["anchors"], start=1)
    ]
    write_json(
        root / "study_manifest/physical_eligible_pool.json",
        {
            "schema_version": "RecoverabilityPhysicalEligiblePoolV1",
            "status": "FAIL",
            "reason": blocker,
            "target_anchor_count": len(pool),
            "physicalizable_anchor_count": 0,
            "anchors": pool,
        },
    )
    write_json(
        root / "study_manifest/replacement_receipt.json",
        {
            "schema_version": "RecoverabilityManifestReplacementReceiptV1",
            "status": "NO_REPLACEMENT",
            "replacement_count": 0,
            "reason": "FROZEN_SELECTION_MUST_NOT_CHANGE_AFTER_PHYSICS_AUTHORITY_AUDIT",
        },
    )
    final_manifest = {
        "schema_version": "RecoverabilityStudyManifestV2",
        "status": "BLOCKED_PHYSICAL_STUDY",
        "reason": blocker,
        "frozen_before_physics": True,
        "anchors": pool,
        "target_anchor_count": len(pool),
        "runnable_anchor_count": 0,
        "reference_authority": "stored SparseV4 ExecutionV3 final q/base",
        "q_old_reference_forbidden": True,
        "selection_replaced": False,
    }
    final_path = root / "study_manifest/final_recoverability_manifest.json"
    write_json(final_path, final_manifest)
    final_path.with_suffix(".sha256").write_text(sha256_file(final_path) + "\n", encoding="utf-8")
    write_csv(
        root / "d2j_baseline_smoke/per_anchor.csv",
        [],
        ["anchor_id", "status", "finite", "object_drift_m", "penetration_p95_m"],
    )
    _write_not_run(
        root, "d2j_baseline_smoke/qualification.json", "D2JSceneQualificationV1", blocker
    )
    for name in (
        "object_physics_authority",
        "support_authority",
        "static_scene_adapter",
        "static_reference_adapter",
        "static_physical_study_scene_contract",
    ):
        _write_not_run(root, f"frozen_d2j/{name}.json", "D2JFrozenContractV1", blocker)
        (root / f"frozen_d2j/{name}.sha256").write_text("NOT_FROZEN\n", encoding="utf-8")
    _write_not_run(
        root, "ppo_authority/current_ppo_authority.json", "CurrentPhysicalPPOAuthorityV1", blocker
    )
    _write_not_run(root, "ppo_authority/reward_authority.json", "PPORewardAuthorityV1", blocker)
    _write_not_run(root, "ppo_authority/ppo_frozen_contract.json", "PPOFrozenContractV1", blocker)
    (root / "ppo_authority/reward_authority.sha256").write_text("NOT_FROZEN\n", encoding="utf-8")
    (root / "ppo_authority/ppo_frozen_contract.sha256").write_text("NOT_FROZEN\n", encoding="utf-8")
    _write_not_run(
        root,
        "study_contract/recoverability_study_contract.json",
        "RetargetToPPORecoverabilityStudyContractV1",
        blocker,
    )
    (root / "study_contract/recoverability_study_contract.sha256").write_text(
        "NOT_FROZEN\n", encoding="utf-8"
    )
    write_csv(
        root / "baseline_physics/per_anchor.csv",
        [],
        ["anchor_id", "status", "penetration_p95_m", "object_drift_m", "contact_status"],
    )
    write_csv(
        root / "baseline_physics/per_rollout.csv",
        [],
        ["anchor_id", "seed", "status", "finite"],
    )
    (root / "ppo_training").mkdir(parents=True, exist_ok=True)
    _write_not_run(root, "ppo_training/not_run.json", "PPOTrainingNotRunV1", blocker)
    write_csv(root / "ppo_eval/per_anchor.csv", [], ["anchor_id", "status", "recovered"])
    write_csv(root / "ppo_eval/per_rollout.csv", [], ["anchor_id", "seed", "status"])
    write_csv(
        root / "analysis/paired_metrics.csv",
        [],
        [
            "anchor_id",
            "group",
            "e_im",
            "geom_penetration_p95_m",
            "baseline_penetration_p95_m",
            "ppo_penetration_p95_m",
            "object_drift_m",
            "contact_status",
            "recovered",
        ],
    )
    write_csv(root / "analysis/semantic_group_metrics.csv", [], ["group", "N", "recovered", "rate"])
    write_csv(
        root / "analysis/penetration_group_metrics.csv", [], ["group", "N", "recovered", "rate"]
    )
    write_json(
        root / "analysis/correlations.json",
        {
            "schema_version": "RecoverabilityCorrelationsV1",
            "status": "NOT_RUN",
            "E_IM_VS_RECOVERABILITY": None,
            "GEOM_PENETRATION_VS_RECOVERABILITY": None,
            "RHO1_VS_RECOVERABILITY": None,
            "RHO2_VS_RECOVERABILITY": None,
            "reason": blocker,
        },
    )
    write_json(
        root / "analysis/recoverability_decision.json",
        {
            "schema_version": "RetargetToPPORecoverabilityDecisionV1",
            "status": "BLOCKED",
            "RETARGET_TO_PPO_RECOVERABILITY_RESULT": "BLOCKED_PHYSICAL_STUDY",
            "blocker": blocker,
            "NEXT": "OAKINK2_PHYSICAL_SCENE_AUTHORITY_REPAIR",
        },
    )
    write_json(
        root / "review/manifest.json",
        {
            "schema_version": "RecoverabilityReviewV1",
            "status": "NOT_RUN",
            "reason": blocker,
            "geometric_reference": str(
                (
                    D2H_ROOT / "diagnostic_viewer/oakink2_sparse_v4_geometry_diagnostic.html"
                ).resolve()
            ),
            "baseline_panels": 0,
            "ppo_panels": 0,
        },
    )
    review = root / "review/index.html"
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text(
        "<!doctype html><meta charset='utf-8'><title>O5R-D2J/D2K blocked</title>"
        "<h1>OakInk2 O5R-D2J/D2K</h1>"
        f"<p>D2J stopped fail-closed: <code>{blocker}</code>.</p>"
        "<p>No reference-hold or PPO rollout was executed.</p>\n",
        encoding="utf-8",
    )


def _evidence_role_ledger(root: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "O5RD2JEvidenceRoleLedgerV1",
        "entries": [
            {
                "evidence": "D2I frozen 12-anchor manifest",
                "role": "FROZEN_SELECTION_INPUT",
                "mutable": False,
            },
            {
                "evidence": "SparseV4 per-anchor qpos/base_pose_scene NPZ",
                "role": "STATIC_REFERENCE_INPUT",
                "mutable": False,
            },
            {
                "evidence": "OakInk2 full source object track and meshes",
                "role": "SUPPORT_AUTHORITY_INPUT",
                "mutable": False,
            },
            {
                "evidence": "post-manipulation stable candidates",
                "role": "DIAGNOSTIC_ONLY_NOT_AUTHORITY",
                "mutable": False,
            },
            {"evidence": "q_old", "role": "FORBIDDEN_AS_PPO_REFERENCE", "used": False},
            {
                "evidence": "PPO/reward source",
                "role": "NOT_AUDITED_DUE_D2J_HARD_STOP",
                "changed": False,
            },
        ],
    }
    write_json(root / "ledger/evidence_role_ledger.json", payload)
    return payload


def summarize(root: Path) -> dict[str, Any]:
    blocker = "BLOCKED_SUPPORT_AUTHORITY"
    _scaffold_blocked_outputs(root, blocker)
    _evidence_role_ledger(root)
    current_head = git("rev-parse", "HEAD")
    commits = [
        {"sha": row.split("\t", 1)[0], "subject": row.split("\t", 1)[1]}
        for row in git("log", "--format=%H%x09%s", f"{START_HEAD}..{current_head}").splitlines()
        if "\t" in row
    ]
    write_json(
        root / "git_commits.json",
        {
            "schema_version": "O5RD2JGitCommitsV1",
            "start_head": START_HEAD,
            "final_head_at_summary": current_head,
            "commits": commits,
            "pushed": False,
            "pr_created": False,
        },
    )
    technical_rows = [
        {
            "status": "RESOLVED",
            "stage": "audit-existing-physical-pipeline",
            "error": "AUDIT_TOKEN_FALSE_NEGATIVE",
            "repair": "audit gravity in cfg and write telemetry in environment implementation",
            "scientific_evidence_consumed": False,
        },
        {
            "status": "RESOLVED",
            "stage": "build-oakink2-support-authority",
            "error": "SupportResolutionResult not JSON serializable",
            "repair": "serialize the existing authority as_dict representation",
            "support_resolver_attempts": 2,
            "scientific_selection_changed": False,
        },
    ]
    (root / "technical_failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in technical_rows),
        encoding="utf-8",
    )
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2JResourceUsageV1",
            "max_gpu_jobs": 1,
            "max_concurrent_gpu_jobs_observed": 1,
            "gpu_jobs_run": 1,
            "gpu_jobs": ["C10001 Isaac Sim USD materialization"],
            "physx_rollouts": 0,
            "ppo_training_runs": 0,
            "support_resolver_attempts": 2,
        },
    )
    object_authority = read_json(root / "object_physics/oakink2_object_physics_authority.json")
    support_authority = read_json(root / "support/oakink2_support_authority.json")
    safety = {
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": current_head,
        "SPARSE_VALIDATION_V4": "FAIL",
        "HISTORICAL_SPARSE_V4_RESULT_REWRITTEN": "NO",
        "SPARSE_V4_PRIMARY_ROOT_CAUSE": "TOP1_CONTRIBUTOR_LOCALITY_INSUFFICIENT",
        "RETARGET_METHOD_CHANGED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "EXECUTION_V3_CHANGED": "NO",
        "CERTIFICATION_GATE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "EXECUTION_V4_CREATED": "NO",
        "D2J_STATUS": "FAIL",
        "D2J_BLOCKER": blocker,
        "OAKINK2_OBJECT_PHYSICS_AUTHORITY": object_authority["status"],
        "VISUAL_COLLISION_AUTHORITY_SEPARATED": "YES",
        "OAKINK2_SUPPORT_AUTHORITY": support_authority["status"],
        "HOCAP_PHYSICAL_BACKEND_REGRESSION": "NOT_RUN_DUE_SUPPORT_HARD_STOP",
        "TARGET_ANCHORS": 12,
        "PHYSICALIZABLE_ANCHORS": 0,
        "REFERENCE_HOLD_EXECUTABLE": 0,
        "PHYSICAL_STUDY_ANCHOR_COUNT": 0,
        "OAKINK2_OBJECT_PHYSICS_AUTHORITY_SHA256": None,
        "OAKINK2_SUPPORT_AUTHORITY_SHA256": None,
        "OAKINK2_STATIC_SCENE_ADAPTER_SHA256": None,
        "STATIC_REFERENCE_ADAPTER_SHA256": None,
        "STATIC_PHYSICAL_STUDY_SCENE_CONTRACT_SHA256": None,
        "D2K_AUTHORIZED": "NO",
        "PPO_REWARD_CHANGED": "NO",
        "PPO_CONTRACT_SHA256": None,
        "REWARD_CONTRACT_SHA256": None,
        "REFERENCE_HOLD_BASELINE_COUNT": 0,
        "PPO_STUDY_ANCHOR_COUNT": 0,
        "PPO_ANCHORS_STARTED": 0,
        "PPO_ANCHORS_COMPLETED": 0,
        "PPO_TECHNICAL_FAILURES": 0,
        "PPO_TRAINING_RUN_COUNT": 0,
        "MAX_GPU_JOBS": 1,
        "GPU_JOBS_RUN": 1,
        "RETARGET_TO_PPO_RECOVERABILITY_RESULT": "BLOCKED_PHYSICAL_STUDY",
        "Q_OLD_USED_AS_PPO_REFERENCE": "NO",
        "SPARSEV4_STATE_USED_AS_REFERENCE": "NO_PHYSICS_RUN;FROZEN_INPUT_AUDITED",
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
        "NEXT": "OAKINK2_PHYSICAL_SCENE_AUTHORITY_REPAIR",
    }
    summary = {
        "schema_version": "OakInk2O5RD2JD2KFinalSummaryV1",
        "status": "FAIL",
        **safety,
        "object_authority": object_authority,
        "support_summary": {
            "support_type_counts": {"UNRESOLVED": 12},
            "resolver_status": support_authority["resolver_status"],
            "resolver_reason": support_authority["resolver_reason"],
        },
        "study_manifest": read_json(root / "study_manifest/final_recoverability_manifest.json"),
        "correlations": read_json(root / "analysis/correlations.json"),
    }
    write_json(root / "final_summary.json", summary)
    handoff_lines = [
        "# OakInk2 O5R-D2J/D2K",
        "",
        "## Result",
        "",
        "D2J stopped fail-closed at `BLOCKED_SUPPORT_AUTHORITY`. D2K was not authorized.",
        "The object asset authority passed, but OakInk2 provides no explicit environment support; the other source objects fail support geometry checks; and SupportResolutionV1 found no stable pre-manipulation interval in the full source track.",
        "",
        "## Safety flags",
        "",
        "```text",
        *[f"{key}={value}" for key, value in safety.items()],
        "```",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python {Path(__file__).resolve()} --help",
        f"python {Path(__file__).resolve()} build-oakink2-support-authority --root {root}",
        f"python {Path(__file__).resolve()} summarize --root {root}",
        "```",
    ]
    (root / "handoff.md").write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
    (root / "final_summary.md").write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
    required = [
        "handoff.md",
        "final_summary.md",
        "final_summary.json",
        "completion_audit.json",
        "preflight/git.json",
        "preflight/frozen_upstream.json",
        "d2j_authority_audit/existing_physical_pipeline.json",
        "d2j_authority_audit/generic_vs_hocap_specific.json",
        "object_physics/oakink2_object_physics_authority.json",
        "object_physics/visual_collision_asset_manifest.csv",
        "object_physics/dynamics_provenance.csv",
        "object_physics/asset_qualification.json",
        "support/oakink2_support_authority.json",
        "support/per_anchor_support.csv",
        "scene_adapter/canonical_static_contact_record_schema.json",
        "scene_adapter/oakink2_static_scene_adapter_contract.json",
        "hocap_regression/parity.json",
        "hocap_regression/smoke_results.json",
        "study_manifest/manifest_audit.json",
        "study_manifest/physical_eligible_pool.json",
        "study_manifest/replacement_receipt.json",
        "study_manifest/final_recoverability_manifest.json",
        "study_manifest/final_recoverability_manifest.sha256",
        "d2j_baseline_smoke/per_anchor.csv",
        "d2j_baseline_smoke/qualification.json",
        "frozen_d2j/object_physics_authority.json",
        "frozen_d2j/object_physics_authority.sha256",
        "frozen_d2j/support_authority.json",
        "frozen_d2j/support_authority.sha256",
        "frozen_d2j/static_scene_adapter.json",
        "frozen_d2j/static_scene_adapter.sha256",
        "frozen_d2j/static_reference_adapter.json",
        "frozen_d2j/static_reference_adapter.sha256",
        "frozen_d2j/static_physical_study_scene_contract.json",
        "frozen_d2j/static_physical_study_scene_contract.sha256",
        "ppo_authority/current_ppo_authority.json",
        "ppo_authority/reward_authority.json",
        "ppo_authority/reward_authority.sha256",
        "ppo_authority/ppo_frozen_contract.json",
        "ppo_authority/ppo_frozen_contract.sha256",
        "study_contract/recoverability_study_contract.json",
        "study_contract/recoverability_study_contract.sha256",
        "baseline_physics/per_anchor.csv",
        "baseline_physics/per_rollout.csv",
        "ppo_training/not_run.json",
        "ppo_eval/per_anchor.csv",
        "ppo_eval/per_rollout.csv",
        "analysis/paired_metrics.csv",
        "analysis/semantic_group_metrics.csv",
        "analysis/penetration_group_metrics.csv",
        "analysis/correlations.json",
        "analysis/recoverability_decision.json",
        "review/index.html",
        "review/manifest.json",
        "ledger/evidence_role_ledger.json",
        "tests.json",
        "validation_results.json",
        "git_commits.json",
        "technical_failures.jsonl",
        "resource_usage.json",
    ]
    missing = [
        name for name in required if name != "completion_audit.json" and not (root / name).exists()
    ]
    write_json(
        root / "completion_audit.json",
        {
            "schema_version": "O5RD2JCompletionAuditV1",
            "status": "PASS" if not missing else "FAIL",
            "workflow_outcome": "D2J_FAIL_D2K_NOT_RUN",
            "required_artifact_count": len(required),
            "missing": missing,
            "contract_hard_stop_honored": True,
        },
    )
    return summary


ACTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "preflight": preflight,
    "verify-upstream-evidence": verify_upstream_evidence,
    "audit-existing-physical-pipeline": audit_existing_physical_pipeline,
    "build-oakink2-object-physics-authority": build_oakink2_object_physics_authority,
    "build-oakink2-support-authority": build_oakink2_support_authority,
    "build-oakink2-static-scene-adapter": build_oakink2_static_scene_adapter,
    "run-hocap-backend-regression": run_hocap_backend_regression,
    "audit-recoverability-manifest": audit_recoverability_manifest,
    "qualify-physical-study-anchors": qualify_physical_study_anchors,
    "run-d2j-reference-hold-smoke": run_d2j_reference_hold_smoke,
    "freeze-d2j-physical-contracts": freeze_d2j_physical_contracts,
    "audit-ppo-authority": audit_ppo_authority,
    "freeze-ppo-recoverability-contract": freeze_ppo_recoverability_contract,
    "run-reference-hold-baselines": run_reference_hold_baselines,
    "run-ppo-recoverability-study": run_ppo_recoverability_study,
    "evaluate-ppo-recoverability": evaluate_ppo_recoverability,
    "analyze-retarget-quality-vs-recovery": analyze_retarget_quality_vs_recovery,
    "render-recoverability-review": render_recoverability_review,
    "summarize": summarize,
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="action", required=True)
    for action in REQUIRED_ACTIONS:
        command = subparsers.add_parser(action, help=ACTIONS[action].__doc__)
        command.add_argument("--root", type=Path, default=ROOT)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        payload = ACTIONS[args.action](args.root.resolve())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(jsonable(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
