#!/usr/bin/env python3
"""Freeze and qualify the exact Stage 16-D runtime convex collision geometry."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.contracts import (  # noqa: E402
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
)
from toporetarget.rl.geometry_audit.convex_query import (  # noqa: E402
    PythonFCLConvexQueryBackend,
)
from toporetarget.rl.geometry_audit.metrics import (  # noqa: E402
    aggregate_penetration,
    qualify_source_corrected,
)
from toporetarget.rl.geometry_audit.runtime_geometry import (  # noqa: E402
    ConvexProxyGeometry,
    load_runtime_geometry_manifest,
    sha256_file,
)
from toporetarget.rl.geometry_audit.transforms import compose_poses, pose_from_matrix  # noqa: E402
from toporetarget.rl.geometry_audit.validation import (  # noqa: E402
    run_geometry_query_analytic_tests,
)
from toporetarget.rl.geometry_audit.visual_diagnostics import (  # noqa: E402
    unsigned_surface_diagnostics,
)
from toporetarget.robots.registry import get_robot_registry  # noqa: E402

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo"
OLD_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting"
ASSET_REPORT_ROOT = REPO_ROOT / ".local/reports/stage16c1_asset_migration"
ARTIFACT_ROOT = REPO_ROOT / ".local/physics_consistent_retargeting"
HAND_BODY_NAMES = (
    "r_wrist",
    "r_index_finger_proximal",
    "r_index_finger_proximal_abd",
    "r_index_finger_middle",
    "r_index_finger_distal",
    "r_middle_finger_proximal",
    "r_middle_finger_proximal_abd",
    "r_middle_finger_middle",
    "r_middle_finger_distal",
    "r_pinky_proximal",
    "r_pinky_proximal_abd",
    "r_pinky_middle",
    "r_pinky_distal",
    "r_ring_finger_proximal",
    "r_ring_finger_proximal_abd",
    "r_ring_finger_middle",
    "r_ring_finger_distal",
    "r_thumb_proximal",
    "r_thumb_proximal_abd",
    "r_thumb_middle",
    "r_thumb_distal",
)
CLIPS = ("hocap_170105", "hocap_170650")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _inventory_files(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"STAGE16D_GEOMETRY_INPUT_MISSING:{path}")
        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def freeze_inputs() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if branch != "feature/reference-tracking-isaaclab":
        raise RuntimeError(f"STAGE16D_BRANCH_LINEAGE_FAILURE:{branch}")
    for ancestor in ("e613a4e", "363f2c8"):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, "HEAD"],
            cwd=REPO_ROOT,
            check=True,
        )
    report_inputs = [
        OLD_REPORT_ROOT / "final_summary.json",
        OLD_REPORT_ROOT / "penetration_audit.json",
        OLD_REPORT_ROOT / "trajectory_qualification_170105_v3.json",
        OLD_REPORT_ROOT / "trajectory_qualification_170650_v3.json",
        OLD_REPORT_ROOT / "optimizer_170105_s3.json",
        OLD_REPORT_ROOT / "optimizer_170650_s3.json",
        OLD_REPORT_ROOT / "failure_transitions.jsonl",
        OLD_REPORT_ROOT / "handoff.md",
        ASSET_REPORT_ROOT / "wuji_asset_manifest.json",
        ASSET_REPORT_ROOT / "hocap_170105_asset_manifest.json",
        ASSET_REPORT_ROOT / "hocap_170650_asset_manifest.json",
    ]
    trace_inputs = [
        OLD_REPORT_ROOT / f"trajectory_trace_{clip.removeprefix('hocap_')}_v3.npz" for clip in CLIPS
    ] + [
        OLD_REPORT_ROOT / f"optimizer_{clip.removeprefix('hocap_')}_s3.actions.npy"
        for clip in CLIPS
    ]
    artifact_inputs = [ARTIFACT_ROOT / clip / "manifest.json" for clip in CLIPS]
    config_inputs = [
        REPO_ROOT / "configs/rl/stage16/stage16d_trajectory_gate.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_trajectory_optimizer.yaml",
        REPO_ROOT / "configs/rl/stage16/stage16d_ppo.yaml",
        REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml",
        REPO_ROOT / "configs/rl/stage16/isaaclab_physx_contract_selected.yaml",
    ]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = (
        REPO_ROOT / ".local/archive" / f"stage16d_geometry_block_baseline_{timestamp}_{head[:7]}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    for path in report_inputs + artifact_inputs + config_inputs:
        destination = archive / path.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    manifest = {
        "schema_version": "Stage16DMetricQualificationFrozenInputsV1",
        "generated_at": datetime.now(UTC).isoformat(),
        "branch": branch,
        "head": head,
        "lineage": {"e613a4e": True, "363f2c8": True},
        "archive": str(archive.relative_to(REPO_ROOT)),
        "existing_reports": _inventory_files(report_inputs),
        "trajectory_inputs": _inventory_files(trace_inputs),
        "artifact_manifests": _inventory_files(artifact_inputs),
        "configs": _inventory_files(config_inputs),
        "source_and_generated_assets": _inventory_files(
            [
                REPO_ROOT
                / _read_json(ASSET_REPORT_ROOT / "wuji_asset_manifest.json")["generated_usd"],
                *[
                    REPO_ROOT
                    / _read_json(ASSET_REPORT_ROOT / f"{clip}_asset_manifest.json")["generated_usd"]
                    for clip in CLIPS
                ],
            ]
        ),
        "formal_results_computed": False,
    }
    _write_json(REPORT_ROOT / "frozen_inputs.json", manifest)
    _write_json(archive / "frozen_manifest.json", manifest)
    _write_json(
        archive / "existing_qualification.json",
        _read_json(OLD_REPORT_ROOT / "final_summary.json"),
    )
    _write_json(
        archive / "current_geometry_metrics.json",
        _read_json(OLD_REPORT_ROOT / "penetration_audit.json"),
    )
    (archive / "README.md").write_text(
        "# Frozen Stage 16-D geometry-block baseline\n\n"
        "This immutable snapshot predates RuntimeCollisionProxyPenetrationV1 formal results.\n",
        encoding="utf-8",
    )
    return manifest


def _matrix_rows(value: Any) -> np.ndarray:
    return np.asarray([[float(value[row][column]) for column in range(4)] for row in range(4)])


def _extract_mesh_proxy(
    *,
    stage: Any,
    usd_geom: Any,
    prim_path: str,
    shape_id: str,
    body_name: str,
    source_path: Path,
    generated_path: Path,
) -> ConvexProxyGeometry:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"STAGE16D_RUNTIME_COLLISION_PRIM_MISSING:{prim_path}")
    mesh = usd_geom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    if not np.all(counts == 3):
        raise RuntimeError(f"STAGE16D_NONTRIANGULAR_RUNTIME_PROXY:{prim_path}")
    xformable = usd_geom.Xformable(prim)
    local_matrix = xformable.GetLocalTransformation()
    usd_matrix = _matrix_rows(local_matrix)
    # USD stores row-vector transforms. Transpose into the column-vector convention.
    local_pose = pose_from_matrix(usd_matrix.T)
    scale = np.linalg.norm(usd_matrix[:3, :3], axis=1)
    if not np.allclose(scale, 1.0, atol=1.0e-12):
        raise RuntimeError(f"STAGE16D_UNEXPECTED_RUNTIME_PROXY_SCALE:{prim_path}:{scale}")
    return ConvexProxyGeometry(
        shape_id=shape_id,
        body_name=body_name,
        geometry_type="convex_hull",
        vertices=vertices,
        faces=indices.reshape(-1, 3),
        local_pose_xyz_wxyz=local_pose,
        scale_xyz=scale,
        source_asset_path=str(source_path.relative_to(REPO_ROOT)),
        source_asset_sha256=sha256_file(source_path),
        generated_asset_path=str(generated_path.relative_to(REPO_ROOT)),
        generated_asset_sha256=sha256_file(generated_path),
    )


def build_runtime_manifest(*, accept_eula: bool) -> dict[str, Any]:
    if not accept_eula:
        raise RuntimeError("explicit --accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    # Kit consumes process arguments itself. Keep this script's ``--phase`` and
    # ``--accept-eula`` out of Kit's parser after argparse has handled them.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        app = AppLauncher(headless=True).app
    finally:
        sys.argv = original_argv
    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        hand_asset_manifest = _read_json(ASSET_REPORT_ROOT / "wuji_asset_manifest.json")
        hand_usd = REPO_ROOT / hand_asset_manifest["generated_usd"]
        if sha256_file(hand_usd) != hand_asset_manifest["generated_sha256"]:
            raise RuntimeError("STAGE16D_GEOMETRY_INPUT_HASH_DRIFT:wuji_generated_usd")
        hand_stage = Usd.Stage.Open(str(hand_usd))
        if hand_stage is None:
            raise RuntimeError("STAGE16D_RUNTIME_HAND_USD_OPEN_FAILURE")
        hand_paths = hand_asset_manifest["collision_geoms"]
        if tuple(path.rsplit("/", 3)[-3] for path in hand_paths) != HAND_BODY_NAMES:
            raise RuntimeError("STAGE16D_RUNTIME_HAND_BODY_ORDER_MISMATCH")
        hand_shapes = []
        for body_index, (body_name, prim_path) in enumerate(
            zip(HAND_BODY_NAMES, hand_paths, strict=True)
        ):
            source_path = (
                REPO_ROOT / hand_asset_manifest["collision_proxy_inventory"][body_name]["source"]
            )
            proxy = _extract_mesh_proxy(
                stage=hand_stage,
                usd_geom=UsdGeom,
                prim_path=prim_path,
                shape_id=f"hand:{body_index}:{prim_path}",
                body_name=body_name,
                source_path=source_path,
                generated_path=hand_usd,
            )
            expected = hand_asset_manifest["collision_proxy_inventory"][body_name]
            if (
                len(proxy.vertices) != expected["vertices"]
                or len(proxy.faces) != expected["triangles"]
            ):
                raise RuntimeError(f"STAGE16D_RUNTIME_HAND_PROXY_COUNT_DRIFT:{body_name}")
            hand_shapes.append(proxy)
        object_shapes: dict[str, list[ConvexProxyGeometry]] = {}
        for clip in CLIPS:
            manifest = _read_json(ASSET_REPORT_ROOT / f"{clip}_asset_manifest.json")
            generated = REPO_ROOT / manifest["generated_usd"]
            if sha256_file(generated) != manifest["generated_sha256"]:
                raise RuntimeError(f"STAGE16D_GEOMETRY_INPUT_HASH_DRIFT:{clip}_generated_usd")
            stage = Usd.Stage.Open(str(generated))
            if stage is None:
                raise RuntimeError(f"STAGE16D_RUNTIME_OBJECT_USD_OPEN_FAILURE:{clip}")
            collision_paths = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if len(collision_paths) != manifest["collision_prim_count"]:
                raise RuntimeError(f"STAGE16D_RUNTIME_OBJECT_PROXY_COUNT_DRIFT:{clip}")
            rows = []
            for piece_index, prim_path in enumerate(collision_paths):
                rows.append(
                    _extract_mesh_proxy(
                        stage=stage,
                        usd_geom=UsdGeom,
                        prim_path=prim_path,
                        shape_id=f"object:{clip}:{piece_index}:{prim_path}",
                        body_name=clip,
                        source_path=REPO_ROOT / manifest["source_file"],
                        generated_path=generated,
                    )
                )
            object_shapes[clip] = rows
        from toporetarget.rl.environments.isaaclab_backend.physics_consistent_retargeting_env_cfg import (  # noqa: E501
            IsaacPhysicsConsistentRetargetingEnvCfg,
            configure_stage16d_nominal,
        )

        runtime_cfg = IsaacPhysicsConsistentRetargetingEnvCfg()
        configure_stage16d_nominal(runtime_cfg, num_envs=1, clip="hocap_170105")
        runtime_paths = {
            "robot": Path(runtime_cfg.robot.spawn.usd_path).resolve(),
            "hocap_170105": Path(runtime_cfg.object_170105.spawn.usd_path).resolve(),
            "hocap_170650": Path(runtime_cfg.object_170650.spawn.usd_path).resolve(),
        }
        expected_object_paths = {
            clip: (
                REPO_ROOT
                / _read_json(ASSET_REPORT_ROOT / f"{clip}_asset_manifest.json")["generated_usd"]
            ).resolve()
            for clip in CLIPS
        }
        if any(runtime_paths[clip] != expected_object_paths[clip] for clip in CLIPS):
            raise RuntimeError("STAGE16D_RUNTIME_CFG_OBJECT_ASSET_PATH_DRIFT")
        runtime_hand_stage = Usd.Stage.Open(str(runtime_paths["robot"]))
        if runtime_hand_stage is None:
            raise RuntimeError("STAGE16D_RUNTIME_CFG_HAND_USD_OPEN_FAILURE")
        composed_hand_meshes = [
            prim
            for prim in runtime_hand_stage.Traverse()
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.GetTypeName() == "Mesh"
            and "stage16c1_collision" in str(prim.GetPath())
        ]
        if len(composed_hand_meshes) != len(hand_shapes):
            raise RuntimeError("STAGE16D_RUNTIME_CFG_HAND_PROXY_COUNT_DRIFT")
        composed_by_body = {
            str(prim.GetPath()).split("/stage16c1_collision", 1)[0].rsplit("/", 1)[-1]: prim
            for prim in composed_hand_meshes
        }
        if set(composed_by_body) != set(HAND_BODY_NAMES):
            raise RuntimeError("STAGE16D_RUNTIME_CFG_HAND_PROXY_ORDER_DRIFT")
        for proxy in hand_shapes:
            mesh = UsdGeom.Mesh(composed_by_body[proxy.body_name])
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            if not np.array_equal(vertices, proxy.vertices) or not np.array_equal(
                indices.reshape(-1, 3), proxy.faces
            ):
                raise RuntimeError(
                    f"STAGE16D_RUNTIME_CFG_HAND_PROXY_GEOMETRY_DRIFT:{proxy.body_name}"
                )
        payload = {
            "schema_version": "RuntimeCollisionGeometryManifestV1",
            "geometry_authority": "authored collision meshes in exact C.1 USDs used by runtime cfg",
            "units": "metre",
            "handedness": "right",
            "hand_shapes": [
                {**row.as_dict(), "body_index": index} for index, row in enumerate(hand_shapes)
            ],
            "object_shapes": {
                clip: [row.as_dict() for row in rows] for clip, rows in object_shapes.items()
            },
            "pair_filter": {
                clip: [f"{row.shape_id}<->{object_shapes[clip][0].shape_id}" for row in hand_shapes]
                for clip in CLIPS
            },
            "excluded": list(GEOMETRY_METRIC_CONTRACT.excludes),
            "validation": {
                "runtime_hand_shape_count": len(hand_shapes),
                "manifest_hand_shape_count": len(hand_asset_manifest["collision_geoms"]),
                "runtime_object_shape_count": {clip: len(object_shapes[clip]) for clip in CLIPS},
                "runtime_composed_hand_shape_count": len(composed_hand_meshes),
                "all_local_transforms_finite": True,
                "all_scales_positive": True,
                "asset_hashes_match": True,
                "runtime_cfg_asset_paths_match": True,
                "runtime_cfg_asset_paths": {
                    key: str(value.relative_to(REPO_ROOT)) for key, value in runtime_paths.items()
                },
            },
        }
        _write_json(REPORT_ROOT / "runtime_collision_geometry_manifest.json", payload)
        _write_json(
            REPORT_ROOT / "hand_collision_geometry.json",
            {"schema_version": "RuntimeHandCollisionGeometryV1", "shapes": payload["hand_shapes"]},
        )
        for clip in CLIPS:
            _write_json(
                REPORT_ROOT / f"{clip}_collision_geometry.json",
                {
                    "schema_version": "RuntimeObjectCollisionGeometryV1",
                    "clip": clip,
                    "shapes": payload["object_shapes"][clip],
                },
            )
        (REPORT_ROOT / "runtime_geometry_inventory.md").write_text(
            "# Runtime collision geometry inventory\n\n"
            f"- Hand convex shapes: {len(hand_shapes)}\n"
            "- hocap_170105 convex shapes: 1\n"
            "- hocap_170650 convex shapes: 1\n"
            "- Visual-only meshes, ghosts, inactive objects, self collision, ground, and support "
            "are excluded.\n",
            encoding="utf-8",
        )
        return payload
    except Exception:
        # SimulationApp.close may terminate Kit before Python prints a pending
        # exception, so flush the actionable failure first.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close(wait_for_replicator=False)


def qualify_backend() -> dict[str, Any]:
    backend = PythonFCLConvexQueryBackend()
    analytic = run_geometry_query_analytic_tests(backend)
    _write_json(REPORT_ROOT / "geometry_query_backend_contract.json", backend.contract.as_dict())
    _write_json(REPORT_ROOT / "geometry_query_analytic_tests.json", analytic)
    _write_json(
        REPORT_ROOT / "geometry_query_numerical_tolerance.json",
        {
            "schema_version": "Stage16DGeometryNumericalToleranceV1",
            "numerical_tolerance_m": backend.contract.numerical_tolerance_m,
            "metric_epsilon_m": backend.contract.metric_epsilon_m,
            "derivation": (
                "5e-7 m is above the 1.35e-7 m sphere overlap signed-distance/MTD "
                "disagreement measured by G2 and was frozen before trajectory queries"
            ),
        },
    )
    _write_json(
        REPORT_ROOT / "geometry_metric_contract.json",
        GEOMETRY_METRIC_CONTRACT.as_dict(query_contract=backend.contract),
    )
    if not analytic["all_pass"]:
        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_FAILED")
    return analytic


def build_source_collision_state(clip: str) -> Path:
    suffix = clip.removeprefix("hocap_")
    source_trace = OLD_REPORT_ROOT / f"source_trace_{suffix}.npz"
    with np.load(source_trace, allow_pickle=False) as payload:
        wrist_pose = np.asarray(payload["wrist_pose"], dtype=np.float64)
        finger_q = np.asarray(payload["finger_q"], dtype=np.float64)
        object_pose = np.asarray(payload["object_pose"], dtype=np.float64)
    if wrist_pose.shape != (321, 7) or finger_q.shape != (321, 20):
        raise RuntimeError(f"STAGE16D_SOURCE_COLLISION_STATE_SHAPE_FAILURE:{clip}")
    model = get_robot_registry(repo_root=REPO_ROOT).load("wuji_hand2_beta1_rh")
    hand_pose = np.empty((321, 1, len(HAND_BODY_NAMES), 7), dtype=np.float64)
    for frame in range(321):
        transforms = model.forward_kinematics_reference(finger_q[frame])
        wrist_matrix = np.eye(4, dtype=np.float64)
        from toporetarget.rl.geometry_audit.transforms import pose_matrix

        wrist_matrix[:] = pose_matrix(wrist_pose[frame])
        for body_index, body_name in enumerate(HAND_BODY_NAMES):
            hand_pose[frame, 0, body_index] = pose_from_matrix(wrist_matrix @ transforms[body_name])
    destination = REPORT_ROOT / f"source_collision_state_{suffix}.npz"
    np.savez_compressed(
        destination,
        object_pose=object_pose[:, None, :],
        hand_collision_body_pose=hand_pose,
        hand_collision_body_names=np.asarray(HAND_BODY_NAMES),
        source_trace_sha256=np.asarray(sha256_file(source_trace)),
    )
    return destination


def _audit_state(
    *,
    clip: str,
    state_path: Path,
    state_kind: str,
    manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    hand_proxies, object_proxies_by_clip = load_runtime_geometry_manifest(manifest_path)
    object_proxies = object_proxies_by_clip[clip]
    backend = PythonFCLConvexQueryBackend()
    hand_shapes = [backend.proxy_shape(proxy) for proxy in hand_proxies]
    object_shapes = [backend.proxy_shape(proxy) for proxy in object_proxies]
    with np.load(state_path, allow_pickle=False) as payload:
        object_pose = np.asarray(payload["object_pose"], dtype=np.float64)
        hand_pose = np.asarray(payload["hand_collision_body_pose"], dtype=np.float64)
        names = tuple(str(value) for value in payload["hand_collision_body_names"])
    if names != HAND_BODY_NAMES or hand_pose.shape[:3] != (
        object_pose.shape[0],
        object_pose.shape[1],
        len(HAND_BODY_NAMES),
    ):
        raise RuntimeError(f"STAGE16D_COLLISION_STATE_CONTRACT_FAILURE:{clip}:{state_kind}")
    if object_pose.shape[0] != 321 or object_pose.shape[2] != 7 or hand_pose.shape[3] != 7:
        raise RuntimeError(f"STAGE16D_COLLISION_STATE_INCOMPLETE:{clip}:{state_kind}")
    frames, replicas = object_pose.shape[:2]
    pair_ids = [
        f"{hand.shape_id}<->{object.shape_id}" for hand in hand_proxies for object in object_proxies
    ]
    signed = np.empty((frames, replicas, len(pair_ids)), dtype=np.float64)
    penetration = np.empty_like(signed)
    direction = np.empty((*signed.shape, 3), dtype=np.float64)
    for frame in range(frames):
        for replica in range(replicas):
            pair_index = 0
            for hand_index, (hand_proxy, hand_shape) in enumerate(
                zip(hand_proxies, hand_shapes, strict=True)
            ):
                hand_world = compose_poses(
                    hand_pose[frame, replica, hand_index], hand_proxy.local_pose_xyz_wxyz
                )
                for object_proxy, object_shape in zip(object_proxies, object_shapes, strict=True):
                    object_world = compose_poses(
                        object_pose[frame, replica], object_proxy.local_pose_xyz_wxyz
                    )
                    query = backend.query(hand_shape, hand_world, object_shape, object_world)
                    if not query.converged:
                        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_NONCONVERGENCE")
                    signed[frame, replica, pair_index] = query.signed_separation_m
                    penetration[frame, replica, pair_index] = query.penetration_depth_m
                    direction[frame, replica, pair_index] = query.depenetration_direction_for_second
                    pair_index += 1
    worst_pair = np.argmax(penetration, axis=2)
    worst = np.take_along_axis(penetration, worst_pair[..., None], axis=2)[..., 0]
    aggregate = aggregate_penetration(worst, worst_pair, pair_ids)
    aggregate.update(
        {
            "schema_version": "RuntimeCollisionProxyPenetrationResultV1",
            "metric_contract": GEOMETRY_METRIC_CONTRACT.schema_version,
            "query_backend": GEOMETRY_QUERY_CONTRACT.backend,
            "query_backend_version": GEOMETRY_QUERY_CONTRACT.backend_version,
            "clip": clip,
            "state_kind": state_kind,
            "state_path": str(state_path.relative_to(REPO_ROOT)),
            "state_sha256": sha256_file(state_path),
            "runtime_geometry_manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "runtime_geometry_manifest_sha256": sha256_file(manifest_path),
            "complete_321_steps": frames == 321,
            "all_queries_converged": True,
            "pair_ids": pair_ids,
        }
    )
    raw_path = REPORT_ROOT / f"{state_kind}_penetration_pairs_{clip.removeprefix('hocap_')}.npz"
    np.savez_compressed(
        raw_path,
        signed_separation_m=signed,
        penetration_depth_m=penetration,
        depenetration_direction_for_object=direction,
        frame_worst_penetration_m=worst,
        frame_worst_pair_index=worst_pair,
        pair_ids=np.asarray(pair_ids),
    )
    return aggregate, raw_path


def _runtime_physx_crosscheck(
    *, clip: str, corrected_raw: Path, corrected_trace: Path
) -> dict[str, Any]:
    suffix = clip.removeprefix("hocap_")
    with np.load(corrected_raw, allow_pickle=False) as query_payload:
        signed = np.asarray(query_payload["signed_separation_m"], dtype=np.float64)[:, 0]
        depth = np.asarray(query_payload["penetration_depth_m"], dtype=np.float64)[:, 0]
        pair_ids = np.asarray(query_payload["pair_ids"], dtype=str)
    with np.load(corrected_trace, allow_pickle=False) as trace:
        runtime_contact = np.asarray(trace["contact_pair_presence"], dtype=bool)
        wrist_pose = np.asarray(trace["wrist_pose"], dtype=np.float64)
        finger_q = np.asarray(trace["finger_q"], dtype=np.float64)
        runtime_hand_pose = np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)
    if signed.shape != runtime_contact.shape:
        raise RuntimeError(f"RUNTIME_GEOMETRY_QUERY_MISMATCH:{clip}:pair_shape")
    offline_contact_offset = signed <= 0.002
    overlap = signed < 0.0
    overall_agreement = float(np.mean(offline_contact_offset == runtime_contact))
    overlap_runtime_recall = (
        float(np.mean(runtime_contact[overlap])) if bool(overlap.any()) else 1.0
    )
    runtime_contact_separation = signed[runtime_contact]
    maximum_runtime_contact_separation = (
        float(runtime_contact_separation.max(initial=0.0))
        if runtime_contact_separation.size
        else 0.0
    )

    model = get_robot_registry(repo_root=REPO_ROOT).load("wuji_hand2_beta1_rh")
    from toporetarget.rl.geometry_audit.transforms import (
        pose_matrix,
        quaternion_matrix_wxyz,
    )

    position_errors = []
    orientation_errors = []
    for frame in range(321):
        transforms = model.forward_kinematics_reference(finger_q[frame])
        wrist_matrix = pose_matrix(wrist_pose[frame])
        for body_index, body_name in enumerate(HAND_BODY_NAMES):
            expected = pose_from_matrix(wrist_matrix @ transforms[body_name])
            actual = runtime_hand_pose[frame, body_index]
            position_errors.append(float(np.linalg.norm(expected[:3] - actual[:3])))
            relative_rotation = quaternion_matrix_wxyz(expected[3:]).T
            relative_rotation @= quaternion_matrix_wxyz(actual[3:])
            orientation_errors.append(
                float(
                    np.arccos(np.clip((float(np.trace(relative_rotation)) - 1.0) / 2.0, -1.0, 1.0))
                )
            )
    flat_deepest = int(np.argmax(depth))
    deepest_frame, deepest_pair = np.unravel_index(flat_deepest, depth.shape)
    contact_frames = np.flatnonzero(runtime_contact.any(axis=1))
    onset_frame = int(contact_frames[0]) if contact_frames.size else 0
    positive = np.where(signed > 0.0, signed, np.inf)
    near_flat = int(np.argmin(positive))
    near_frame, near_pair = np.unravel_index(near_flat, positive.shape)
    no_contact_candidates = np.flatnonzero(~runtime_contact.any(axis=1))
    no_contact_frame = int(no_contact_candidates[0]) if no_contact_candidates.size else 0
    no_contact_pair = int(np.argmin(signed[no_contact_frame]))

    def selected_row(label: str, frame: int, pair: int) -> dict[str, Any]:
        return {
            "label": label,
            "frame": frame,
            "pair_index": pair,
            "pair_id": str(pair_ids[pair]),
            "offline_signed_separation_m": float(signed[frame, pair]),
            "offline_penetration_m": float(depth[frame, pair]),
            "physx_runtime_contact_present": bool(runtime_contact[frame, pair]),
        }

    selected = [
        selected_row("deepest_penetration", int(deepest_frame), int(deepest_pair)),
        selected_row("contact_onset", onset_frame, int(np.argmax(runtime_contact[onset_frame]))),
        selected_row("near_touch", int(near_frame), int(near_pair)),
        selected_row("no_contact", no_contact_frame, no_contact_pair),
    ]
    passed = (
        overall_agreement >= 0.90
        and maximum_runtime_contact_separation < 0.010
        and max(position_errors) < 1.0e-5
        and max(orientation_errors) < 1.0e-4
    )
    return {
        "schema_version": "Stage16DRuntimePhysXGeometryCrosscheckV1",
        "clip": clip,
        "offline_contact_definition": "signed separation <= frozen 2mm PhysX contact offset",
        "overall_pair_classification_agreement": overall_agreement,
        "overlap_runtime_contact_recall": overlap_runtime_recall,
        "maximum_offline_separation_for_runtime_contact_m": maximum_runtime_contact_separation,
        "runtime_contact_pair_count": int(runtime_contact.sum()),
        "offline_overlap_pair_count": int(overlap.sum()),
        "offline_contact_offset_pair_count": int(offline_contact_offset.sum()),
        "fk_runtime_transform_position_error_max_m": max(position_errors),
        "fk_runtime_transform_position_error_rms_m": float(
            np.sqrt(np.mean(np.square(position_errors)))
        ),
        "fk_runtime_transform_orientation_error_max_rad": max(orientation_errors),
        "fk_runtime_transform_orientation_error_rms_rad": float(
            np.sqrt(np.mean(np.square(orientation_errors)))
        ),
        "selected_frames": selected,
        "physx_numeric_depth_comparison": (
            "not available from object-centric force-matrix telemetry; sign/contact only"
        ),
        "explanation": (
            "PhysX contact generation uses offsets and solver manifolds, so exact pairwise "
            "numeric equality is neither available nor required"
        ),
        "pass": passed,
        "status": "RUNTIME_GEOMETRY_QUERY_CROSSCHECK_PASS"
        if passed
        else "RUNTIME_GEOMETRY_QUERY_MISMATCH",
        "suffix": suffix,
    }


def _visual_proxy_report(manifest_path: Path) -> dict[str, Any]:
    import trimesh

    _, objects = load_runtime_geometry_manifest(manifest_path)
    result: dict[str, Any] = {
        "schema_version": "Stage16DVisualProxyDiagnosticsInventoryV1",
        "formal_gate_authority": False,
        "clips": {},
    }
    for clip in CLIPS:
        asset_manifest = _read_json(ASSET_REPORT_ROOT / f"{clip}_asset_manifest.json")
        visual = trimesh.load_mesh(REPO_ROOT / asset_manifest["source_file"], process=False)
        if not isinstance(visual, trimesh.Trimesh):
            raise RuntimeError(f"STAGE16D_VISUAL_MESH_LOAD_FAILURE:{clip}")
        proxy = objects[clip][0]
        row = unsigned_surface_diagnostics(
            visual_vertices=np.asarray(visual.vertices),
            visual_faces=np.asarray(visual.faces),
            proxy_vertices=proxy.scaled_vertices,
            proxy_faces=proxy.faces,
        )
        row.update(
            {
                "clip": clip,
                "visual_source": asset_manifest["source_file"],
                "visual_source_sha256": asset_manifest["visual_mesh_sha256"],
                "visual_watertight": bool(asset_manifest["watertight"]),
                "proxy_geometry_sha256": proxy.as_dict()["geometry_sha256"],
            }
        )
        result["clips"][clip] = row
    return result


def audit_trajectories() -> dict[str, Any]:
    manifest_path = REPORT_ROOT / "runtime_collision_geometry_manifest.json"
    if not _read_json(REPORT_ROOT / "geometry_query_analytic_tests.json")["all_pass"]:
        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_NOT_QUALIFIED")
    source_results: dict[str, dict[str, Any]] = {}
    corrected_results: dict[str, dict[str, Any]] = {}
    qualifications: dict[str, dict[str, Any]] = {}
    raw_paths: dict[str, dict[str, str]] = {}
    crosschecks: dict[str, dict[str, Any]] = {}
    for clip in CLIPS:
        suffix = clip.removeprefix("hocap_")
        source_state = build_source_collision_state(clip)
        source, source_raw = _audit_state(
            clip=clip,
            state_path=source_state,
            state_kind="source_runtime",
            manifest_path=manifest_path,
        )
        trace = OLD_REPORT_ROOT / f"trajectory_trace_{suffix}_v3.npz"
        # Normalize the existing trace keys without changing the frozen trace.
        with np.load(trace, allow_pickle=False) as payload:
            corrected_state = REPORT_ROOT / f"corrected_collision_state_{suffix}.npz"
            np.savez_compressed(
                corrected_state,
                object_pose=np.asarray(payload["replica_object_pose"], dtype=np.float64),
                hand_collision_body_pose=np.asarray(
                    payload["replica_hand_collision_body_pose"], dtype=np.float64
                ),
                hand_collision_body_names=np.asarray(payload["hand_collision_body_names"]),
                corrected_trace_sha256=np.asarray(sha256_file(trace)),
            )
        corrected, corrected_raw = _audit_state(
            clip=clip,
            state_path=corrected_state,
            state_kind="corrected_runtime",
            manifest_path=manifest_path,
        )
        qualification = qualify_source_corrected(source, corrected)
        crosscheck = _runtime_physx_crosscheck(
            clip=clip, corrected_raw=corrected_raw, corrected_trace=trace
        )
        qualification["runtime_physx_crosscheck"] = crosscheck
        qualification["formal_pass"] = bool(qualification["formal_pass"] and crosscheck["pass"])
        suffix_status = clip.removeprefix("hocap_")
        qualification["status"] = (
            f"STAGE16D_{suffix_status}_GEOMETRY_VALIDATED"
            if qualification["formal_pass"]
            else f"STAGE16D_{suffix_status}_GEOMETRY_BLOCKED"
        )
        qualification.update({"clip": clip, "source": source, "corrected": corrected})
        source_results[clip] = source
        corrected_results[clip] = corrected
        qualifications[clip] = qualification
        crosschecks[clip] = crosscheck
        raw_paths[clip] = {
            "source": str(source_raw.relative_to(REPO_ROOT)),
            "corrected": str(corrected_raw.relative_to(REPO_ROOT)),
        }
        _write_json(REPORT_ROOT / f"source_runtime_penetration_{suffix}.json", source)
        _write_json(REPORT_ROOT / f"corrected_runtime_penetration_{suffix}.json", corrected)
        _write_json(REPORT_ROOT / f"geometry_qualification_{suffix}.json", qualification)
    comparison = {
        "schema_version": "Stage16DSourceCorrectedPenetrationComparisonV1",
        "metric_contract": GEOMETRY_METRIC_CONTRACT.as_dict(query_contract=GEOMETRY_QUERY_CONTRACT),
        "clips": qualifications,
        "raw_pair_timelines": raw_paths,
        "runtime_physx_crosschecks": crosschecks,
    }
    _write_json(REPORT_ROOT / "source_vs_corrected_penetration.json", comparison)
    _write_json(
        REPORT_ROOT / "runtime_physx_geometry_crosscheck.json",
        {
            "schema_version": "Stage16DRuntimePhysXGeometryCrosscheckInventoryV1",
            "clips": crosschecks,
            "all_pass": all(row["pass"] for row in crosschecks.values()),
        },
    )
    _write_json(REPORT_ROOT / "visual_proxy_diagnostics.json", _visual_proxy_report(manifest_path))
    _write_json(
        REPORT_ROOT / "penetration_windows.json",
        {
            "schema_version": "Stage16DPenetrationWindowsV1",
            "source": {clip: source_results[clip]["contiguous_windows"] for clip in CLIPS},
            "corrected": {clip: corrected_results[clip]["contiguous_windows"] for clip in CLIPS},
        },
    )
    return comparison


def _collision_state_from_trace(*, trace: Path, clip: str, label: str) -> Path:
    """Normalize a qualification trace into the formal geometry state contract."""
    if not trace.is_file():
        raise FileNotFoundError(f"STAGE16D_CANDIDATE_TRACE_MISSING:{trace}")
    if not re.fullmatch(r"[a-z0-9_]+", label):
        raise ValueError(f"STAGE16D_CANDIDATE_LABEL_INVALID:{label}")
    suffix = clip.removeprefix("hocap_")
    state_path = REPORT_ROOT / f"corrected_collision_state_{suffix}_{label}.npz"
    with np.load(trace, allow_pickle=False) as payload:
        required = {
            "replica_object_pose",
            "replica_hand_collision_body_pose",
            "hand_collision_body_names",
        }
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(
                f"STAGE16D_CANDIDATE_TRACE_CONTRACT_FAILURE:{clip}:{sorted(missing)}"
            )
        np.savez_compressed(
            state_path,
            object_pose=np.asarray(payload["replica_object_pose"], dtype=np.float64),
            hand_collision_body_pose=np.asarray(
                payload["replica_hand_collision_body_pose"], dtype=np.float64
            ),
            hand_collision_body_names=np.asarray(payload["hand_collision_body_names"]),
            corrected_trace_sha256=np.asarray(sha256_file(trace)),
            candidate_label=np.asarray(label),
        )
    return state_path


def audit_corrected_candidate(*, clip: str, trace: Path, label: str) -> dict[str, Any]:
    """Audit one bounded correction candidate without overwriting baseline evidence."""
    if clip not in CLIPS:
        raise ValueError(f"STAGE16D_CANDIDATE_CLIP_INVALID:{clip}")
    manifest_path = REPORT_ROOT / "runtime_collision_geometry_manifest.json"
    if not _read_json(REPORT_ROOT / "geometry_query_analytic_tests.json")["all_pass"]:
        raise RuntimeError("STAGE16D_FORMAL_CONVEX_QUERY_BACKEND_NOT_QUALIFIED")
    suffix = clip.removeprefix("hocap_")
    source_path = REPORT_ROOT / f"source_runtime_penetration_{suffix}.json"
    source = _read_json(source_path)
    state_path = _collision_state_from_trace(trace=trace, clip=clip, label=label)
    corrected, corrected_raw = _audit_state(
        clip=clip,
        state_path=state_path,
        state_kind=f"corrected_runtime_{label}",
        manifest_path=manifest_path,
    )
    qualification = qualify_source_corrected(source, corrected)
    crosscheck = _runtime_physx_crosscheck(
        clip=clip,
        corrected_raw=corrected_raw,
        corrected_trace=trace,
    )
    qualification["runtime_physx_crosscheck"] = crosscheck
    qualification["formal_pass"] = bool(qualification["formal_pass"] and crosscheck["pass"])
    qualification["status"] = (
        f"STAGE16D_{suffix}_{label.upper()}_GEOMETRY_VALIDATED"
        if qualification["formal_pass"]
        else f"STAGE16D_{suffix}_{label.upper()}_GEOMETRY_BLOCKED"
    )
    qualification.update(
        {
            "clip": clip,
            "candidate_label": label,
            "candidate_trace": str(trace.relative_to(REPO_ROOT)),
            "candidate_trace_sha256": sha256_file(trace),
            "source": source,
            "corrected": corrected,
            "raw_pair_timeline": str(corrected_raw.relative_to(REPO_ROOT)),
        }
    )
    _write_json(REPORT_ROOT / f"corrected_runtime_penetration_{suffix}_{label}.json", corrected)
    _write_json(REPORT_ROOT / f"geometry_qualification_{suffix}_{label}.json", qualification)
    return qualification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("freeze", "inventory", "backend", "audit", "candidate", "all"),
        default="all",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=CLIPS)
    parser.add_argument("--corrected-trace", type=Path)
    parser.add_argument("--candidate-label")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"phase": args.phase}
    if args.phase in {"freeze", "all"}:
        result["freeze"] = freeze_inputs()
    if args.phase in {"inventory", "all"}:
        result["inventory"] = build_runtime_manifest(accept_eula=args.accept_eula)
    if args.phase in {"backend", "all"}:
        result["backend"] = qualify_backend()
    if args.phase in {"audit", "all"}:
        result["audit"] = audit_trajectories()
    if args.phase == "candidate":
        if args.clip is None or args.corrected_trace is None or args.candidate_label is None:
            raise RuntimeError(
                "--phase candidate requires --clip, --corrected-trace, and --candidate-label"
            )
        result["candidate"] = audit_corrected_candidate(
            clip=args.clip,
            trace=args.corrected_trace.resolve(),
            label=args.candidate_label,
        )
    print(json.dumps({"phase": args.phase, "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
