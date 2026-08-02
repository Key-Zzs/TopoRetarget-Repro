#!/usr/bin/env python3
"""Convert the two frozen non-watertight HO-Cap OBJs into ignored rigid USDs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.environments.isaaclab_backend.asset_contracts import (  # noqa: E402
    load_asset_migration_config,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/rl/stage16/isaaclab_asset_validation.yaml",
    )
    parser.add_argument("--accept-eula", action="store_true")
    return parser.parse_args()


def _read_obj(
    path: Path,
) -> tuple[list[tuple[float, float, float]], list[int], list[int], list[float], list[float]]:
    vertices: list[tuple[float, float, float]] = []
    face_counts: list[int] = []
    face_indices: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("v "):
                values = line.split()
                vertices.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("f "):
                indices = [int(value.split("/", 1)[0]) - 1 for value in line.split()[1:]]
                face_counts.append(len(indices))
                face_indices.extend(indices)
    if not vertices:
        raise ValueError(f"OBJECT_MESH_IMPORT_FAILURE: no vertices in {path}")
    return (
        vertices,
        face_counts,
        face_indices,
        [min(row[index] for row in vertices) for index in range(3)],
        [max(row[index] for row in vertices) for index in range(3)],
    )


def _define_mesh(usd_geom, gf, stage, path, vertices, face_counts, face_indices):
    mesh = usd_geom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([gf.Vec3f(*vertex) for vertex in vertices])
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr(usd_geom.Tokens.none)
    return mesh


def _bounded_convex_proxy(vertices):
    import numpy as np
    from scipy.spatial import ConvexHull

    values = np.asarray(vertices, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    def directions(count):
        result = []
        for index in range(count):
            z = 1.0 - 2.0 * (index + 0.5) / count
            radius = math.sqrt(max(0.0, 1.0 - z * z))
            result.append(
                (
                    radius * math.cos(index * golden_angle),
                    radius * math.sin(index * golden_angle),
                    z,
                )
            )
        return np.asarray(result, dtype=np.float64)

    build_directions = directions(58)
    build_directions = np.vstack((build_directions, np.eye(3), -np.eye(3)))
    chosen = np.unique(np.argmax(values @ build_directions.T, axis=0))
    points = values[chosen]
    hull = ConvexHull(points)
    audit_directions = directions(256)
    original_support = np.max(values @ audit_directions.T, axis=0)
    proxy_support = np.max(points @ audit_directions.T, axis=0)
    max_support_gap = float(np.max(original_support - proxy_support))
    return points.tolist(), hull.simplices.astype(np.int32).tolist(), max_support_gap


def main() -> None:
    args = parse_args()
    cfg = load_asset_migration_config(args.config)
    cfg.validate(REPO_ROOT)
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
    try:
        import isaacsim  # noqa: F401
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        summaries = []
        for item in cfg.objects:
            source = (REPO_ROOT / item.source_file).resolve()
            output_dir = (REPO_ROOT / cfg.output_root / item.object_id).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            generated = output_dir / f"{item.object_id}.usda"
            vertices, face_counts, face_indices, bbox_min, bbox_max = _read_obj(source)
            proxy_vertices, proxy_faces, max_support_gap = _bounded_convex_proxy(vertices)
            proxy_bbox_min = [min(row[index] for row in proxy_vertices) for index in range(3)]
            proxy_bbox_max = [max(row[index] for row in proxy_vertices) for index in range(3)]
            stage = Usd.Stage.CreateNew(str(generated))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            root = UsdGeom.Xform.Define(stage, f"/{item.object_id}").GetPrim()
            stage.SetDefaultPrim(root)
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(root)
            rigid_api.CreateRigidBodyEnabledAttr(True)
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
            physx_rigid.CreateDisableGravityAttr(True)
            physx_rigid.CreateLinearDampingAttr(0.0)
            physx_rigid.CreateAngularDampingAttr(0.0)
            mass_api = UsdPhysics.MassAPI.Apply(root)
            mass_api.CreateMassAttr(item.mass_kg)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*item.principal_inertia_kgm2))
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*item.center_of_mass_m))
            mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

            _define_mesh(
                UsdGeom,
                Gf,
                stage,
                f"/{item.object_id}/Visual/original_obj",
                vertices,
                face_counts,
                face_indices,
            )
            collision = _define_mesh(
                UsdGeom,
                Gf,
                stage,
                f"/{item.object_id}/Collision/convex_hull_v1",
                proxy_vertices,
                [3] * len(proxy_faces),
                [index for face in proxy_faces for index in face],
            )
            UsdGeom.Imageable(collision.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr(
                "convexHull"
            )
            convex_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(collision.GetPrim())
            convex_api.CreateHullVertexLimitAttr(64)
            material = UsdShade.Material.Define(stage, f"/{item.object_id}/PhysicsMaterial")
            material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            material_api.CreateStaticFrictionAttr(item.friction[0])
            material_api.CreateDynamicFrictionAttr(item.friction[0])
            material_api.CreateRestitutionAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
                material, materialPurpose="physics"
            )
            stage.GetRootLayer().Save()
            # Re-open the serialized file so the manifest describes bytes on disk.
            stage = Usd.Stage.Open(str(generated))
            root = stage.GetDefaultPrim()
            mesh_prims = [prim for prim in stage.Traverse() if prim.GetTypeName() == "Mesh"]
            collision_prims = [
                str(prim.GetPath()) for prim in mesh_prims if prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            manifest = {
                "schema_version": "toporetarget.stage16c1.hocap_asset_manifest.v1",
                "object_id": item.object_id,
                "source_file": str(item.source_file),
                "visual_mesh_sha256": item.source_sha256,
                "visual_mesh_original_obj": True,
                "generated_usd": str(generated.relative_to(REPO_ROOT)),
                "generated_sha256": sha256_file(generated),
                "root_prim": str(root.GetPath()),
                "collision_method": "convex_hull_v1",
                "collision_parameters": {
                    "hull_vertex_limit": 64,
                    "fallback_reason": "convex decomposition cooking exceeded 240 seconds",
                },
                "convex_part_count": 1,
                "collision_prim_count": len(collision_prims),
                "visual_bbox_m": {"min": bbox_min, "max": bbox_max},
                "collision_bbox_m": {"min": proxy_bbox_min, "max": proxy_bbox_max},
                "geometry_deviation": {
                    "metric": "max_support_gap_over_256_deterministic_directions_m",
                    "value": max_support_gap,
                },
                "collision_proxy_vertex_count": len(proxy_vertices),
                "collision_proxy_triangle_count": len(proxy_faces),
                "vertex_count": len(vertices),
                "face_count": len(face_counts),
                "scale": item.scale,
                "translation": item.translation,
                "rotation_wxyz": item.rotation_wxyz,
                "mass_kg": item.mass_kg,
                "principal_inertia_kgm2": item.principal_inertia_kgm2,
                "center_of_mass_m": item.center_of_mass_m,
                "friction": item.friction,
                "rigid_body": {
                    "free": True,
                    "gravity_enabled": False,
                    "ground": False,
                    "support": "none",
                },
                "physical_classification": item.physical_classification,
                "watertight": False,
                "warnings": [
                    "Mass, inertia and COM are cross-backend engineering nominal values, "
                    "not ground truth."
                ],
            }
            write_json(
                REPO_ROOT / cfg.report_root / f"{item.object_id}_asset_manifest.json", manifest
            )
            summaries.append(manifest)
        write_json(
            REPO_ROOT / cfg.report_root / "object_collision_validation.json",
            {
                "status": "STAGE16C1_HOCAP_COLLISION_ASSETS_IMPORTED",
                "uniform_strategy": "convex_hull_v1",
                "objects": summaries,
            },
        )
        print(json.dumps({"status": "HOCAP_USD_IMPORTED", "objects": summaries}, sort_keys=True))
    finally:
        simulation_app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
