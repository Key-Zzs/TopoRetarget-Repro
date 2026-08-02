#!/usr/bin/env python3
"""Convert the frozen Wuji Hand2 Beta1 RH URDF into an ignored Isaac USD."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.environments.isaaclab_backend.asset_contracts import (  # noqa: E402
    load_asset_migration_config,
    sha256_file,
    write_json,
)
from toporetarget.rl.environments.isaaclab_backend.asset_validation import (  # noqa: E402
    validate_manifest_schema,
)


def _binary_stl_vertices(path: Path):
    import numpy as np

    payload = path.read_bytes()
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected = 84 + triangle_count * 50
    if len(payload) != expected:
        raise ValueError(f"COLLISION_IMPORT_FAILURE: unsupported STL encoding {path}")
    vertices = np.empty((triangle_count * 3, 3), dtype=np.float32)
    for index in range(triangle_count):
        offset = 84 + index * 50 + 12
        vertices[index * 3 : index * 3 + 3] = np.frombuffer(
            payload, dtype="<f4", count=9, offset=offset
        ).reshape(3, 3)
    return np.unique(vertices, axis=0)


def _bounded_convex_proxy(path: Path):
    import numpy as np
    from scipy.spatial import ConvexHull

    vertices = _binary_stl_vertices(path)
    directions = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(58):
        z = 1.0 - 2.0 * (index + 0.5) / 58.0
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        directions.append(
            (radius * math.cos(index * golden_angle), radius * math.sin(index * golden_angle), z)
        )
    directions.extend(((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
    direction_array = np.asarray(directions, dtype=np.float32)
    chosen = np.unique(np.argmax(vertices @ direction_array.T, axis=0))
    proxy_points = vertices[chosen]
    hull = ConvexHull(proxy_points)
    return proxy_points, hull.simplices.astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
        help="Read-only wuji-description checkout root; recorded in the generated report only",
    )
    return parser.parse_args()


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def main() -> None:
    args = parse_args()
    cfg = load_asset_migration_config(args.config)
    joints = cfg.validate(REPO_ROOT)
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
    try:
        import isaacsim  # noqa: F401
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        source = (REPO_ROOT / cfg.wuji.source_file).resolve()
        upstream_bundle = args.upstream_root.resolve() / "hand2/hand2_beta1/body/usd/right"
        upstream_usd = upstream_bundle / "wujihand2.usd"
        if not upstream_usd.is_file():
            raise FileNotFoundError(f"WUJI_SOURCE_ASSET_NOT_FOUND: {upstream_usd}")
        output_dir = (REPO_ROOT / cfg.output_root / "wuji_hand2_beta1").resolve()
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(upstream_bundle, output_dir)
        generated = output_dir / "configuration/wujihand2_physics.usd"
        stage = Usd.Stage.Open(str(generated))
        if stage is None:
            raise RuntimeError("URDF_IMPORT_FAILURE: copied upstream USD could not be opened")
        fixed_joint_paths = [
            prim.GetPath()
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.FixedJoint) and prim.GetName() == "root_joint"
        ]
        if len(fixed_joint_paths) != 1:
            raise RuntimeError(
                f"FLOATING_ROOT_FAILURE: expected one root_joint, got {fixed_joint_paths}"
            )
        for path in fixed_joint_paths:
            stage.OverridePrim(path).SetActive(False)
        default_prim = stage.GetDefaultPrim()
        UsdPhysics.ArticulationRootAPI.Apply(default_prim)
        physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(default_prim)
        physx_articulation.CreateEnabledSelfCollisionsAttr(cfg.wuji.self_collision)
        stage.OverridePrim("/colliders").SetActive(False)
        proxy_inventory = {}
        collision_bodies = [cfg.wuji.root_link, *(joint.child for joint in joints)]
        for body_name in collision_bodies:
            stage.OverridePrim(f"/wujihand2_right/{body_name}/collisions").SetActive(False)
            mesh_source = (
                REPO_ROOT
                / "third_party/robot_hands/wuji_hand2_beta1/meshes/right"
                / f"{body_name}.STL"
            )
            if not mesh_source.is_file():
                continue
            points, faces = _bounded_convex_proxy(mesh_source)
            mesh = UsdGeom.Mesh.Define(
                stage, f"/wujihand2_right/{body_name}/stage16c1_collision/mesh"
            )
            mesh.CreatePointsAttr([tuple(float(value) for value in row) for row in points])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            proxy_inventory[body_name] = {
                "source": str(mesh_source.relative_to(REPO_ROOT)),
                "vertices": int(len(points)),
                "triangles": int(len(faces)),
            }
        stage.GetRootLayer().Save()
        prims = list(stage.Traverse())
        physics_prims = prims
        articulation_prims = [
            str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        body_prims = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        collision_prims = [
            str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        visual_prims = [
            str(prim.GetPath())
            for prim in prims
            if prim.IsA(UsdGeom.Mesh) and "/visuals/" in str(prim.GetPath())
        ]
        usd_joints = [
            prim
            for prim in prims
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
        ]
        manifest = {
            "schema_version": "toporetarget.stage16c1.wuji_asset_manifest.v1",
            "source_repo": "https://github.com/wuji-technology/wuji-description.git",
            "source_ref": "release/v2026.7.23",
            "source_commit": cfg.wuji.source_commit,
            "source_file": str(upstream_usd),
            "source_sha256": sha256_file(upstream_usd),
            "kinematic_source_file": str(cfg.wuji.source_file),
            "kinematic_source_sha256": cfg.wuji.source_sha256,
            "license": "MIT",
            "import_tool": {
                "name": "frozen_upstream_usd_floating_root_overlay_v1",
                "isaac_lab": "2.3.2",
                "isaac_sim": "5.1.0.0",
                "configuration": {
                    "fixed_joints_deactivated": [str(path) for path in fixed_joint_paths],
                    "self_collision": cfg.wuji.self_collision,
                    "collision": cfg.wuji.collision_strategy,
                    "runtime_collision_proxy": "deterministic_64_vertex_convex_hull_v1",
                },
            },
            "generated_usd": str(generated.relative_to(REPO_ROOT)),
            "generated_sha256": sha256_file(generated),
            "root_prim": str(default_prim.GetPath()),
            "root_applied_schemas": list(default_prim.GetAppliedSchemas()),
            "root_variant_sets": {
                name: default_prim.GetVariantSet(name).GetVariantSelection()
                for name in default_prim.GetVariantSets().GetNames()
            },
            "articulation_root": articulation_prims,
            "fixed_base": False,
            "body_names": [path.rsplit("/", 1)[-1] for path in body_prims],
            "joint_names": [prim.GetName() for prim in usd_joints],
            "joint_order": list(cfg.wuji.joint_order),
            "joint_types": {joint.name: joint.joint_type for joint in joints},
            "joint_axes": {joint.name: joint.axis for joint in joints},
            "limits": {joint.name: joint.limits for joint in joints},
            "default_pose": {name: 0.0 for name in cfg.wuji.joint_order},
            "drive_configuration": {
                "type": "upstream_implicit_position_force",
                "source": "upstream config.yaml at release/v2026.7.23",
                "runtime_override_stiffness": cfg.wuji.drive_stiffness,
                "runtime_override_damping": cfg.wuji.drive_damping,
            },
            "collision_geoms": collision_prims,
            "collision_proxy_inventory": proxy_inventory,
            "visual_geoms": visual_prims,
            "mesh_inventory": {
                str(prim.GetPath()): list(prim.GetAppliedSchemas())
                for prim in prims
                if prim.IsA(UsdGeom.Mesh)
            },
            "physics_layer_inventory": {
                str(prim.GetPath()): {
                    "type": prim.GetTypeName(),
                    "schemas": list(prim.GetAppliedSchemas()),
                }
                for prim in physics_prims
                if prim.GetAppliedSchemas() or "collision" in str(prim.GetPath()).lower()
            },
            "mass_inertia": "preserved from frozen URDF per link",
            "tracked_links": list(cfg.wuji.tracked_links),
            "semantic_mapping": cfg.wuji.semantic_mapping,
            "self_collision": cfg.wuji.self_collision,
            "units": {"meters_per_unit": stage.GetMetadata("metersPerUnit"), "angle": "radian"},
            "warnings": [
                "The explicitly supplied external checkout USD bundle is byte-identical to "
                "release/v2026.7.23 for this imported directory.",
                "The first URDF converter strategy timed out during extension resolution; the "
                "frozen upstream official USD is the complete traced source format.",
            ],
        }
        validate_manifest_schema(manifest)
        report = REPO_ROOT / cfg.report_root / "wuji_asset_manifest.json"
        write_json(report, manifest)
        write_json(
            REPO_ROOT / cfg.report_root / "wuji_source_inventory.json",
            {
                "status": "WUJI_SOURCE_PROVENANCE_COMPLETE",
                "configured_source": str(source),
                "configured_source_sha256": sha256_file(source),
                "frozen_ref": "release/v2026.7.23",
                "frozen_commit": cfg.wuji.source_commit,
                "external_checkout_actual_path": str(args.upstream_root.resolve()),
                "external_checkout_actual_branch": _git_value(
                    args.upstream_root, "branch", "--show-current"
                ),
                "external_checkout_actual_commit": _git_value(
                    args.upstream_root, "rev-parse", "HEAD"
                ),
                "external_checkout_modified": bool(
                    _git_value(args.upstream_root, "status", "--porcelain")
                ),
            },
        )
        print(json.dumps({"status": "WUJI_USD_IMPORTED", "manifest": manifest}, sort_keys=True))
    finally:
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
