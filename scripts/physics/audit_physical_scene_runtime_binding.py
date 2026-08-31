#!/usr/bin/env python3
"""Enumerate collision authority from the composed Wuji/object USD stages.

This is intentionally a runtime-asset audit, not a visualization audit.  It
records the collision prim, owning rigid body/link, shape, transforms, and the
authored/default PhysX flags that the scene builder can actually bind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument(
        "--object-usd", action="append", nargs=2, metavar=("ID", "PATH"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-eula", action="store_true")
    return parser.parse_args()


def _value(prim: Any, names: tuple[str, ...]) -> tuple[Any, str]:
    for name in names:
        attribute = prim.GetAttribute(name)
        if attribute and attribute.IsValid() and attribute.HasAuthoredValueOpinion():
            return attribute.Get(), f"authored:{name}"
    return None, "not_authored"


def _matrix(value: Any) -> list[list[float]]:
    return [[float(value[row][column]) for column in range(4)] for row in range(4)]


def _nearest_rigid_body(prim: Any, usd_physics: Any) -> Any | None:
    current = prim
    while current and current.IsValid():
        if current.HasAPI(usd_physics.RigidBodyAPI):
            return current
        current = current.GetParent()
    return None


def _shape_row(
    prim: Any, *, stage: Any, usd_geom: Any, usd_physics: Any, physx_schema: Any, role: str
) -> dict[str, object]:
    body = _nearest_rigid_body(prim, usd_physics)
    xform = usd_geom.XformCache().GetLocalToWorldTransform(prim)
    local = usd_geom.Xformable(prim).GetLocalTransformation()
    collision_enabled, collision_source = _value(prim, ("physics:collisionEnabled",))
    if collision_enabled is None:
        collision_enabled, collision_source = True, "default_collision_api_enabled"
    body_enabled, body_source = (
        _value(body, ("physics:rigidBodyEnabled",)) if body else (False, "missing_body")
    )
    report_enabled = False
    report_source = "not_applied"
    if body is not None and body.HasAPI(physx_schema.PhysxContactReportAPI):
        report_enabled = True
        report_source = "PhysxContactReportAPI"
    group, group_source = _value(prim, ("physxCollision:collisionGroup", "physics:collisionGroup"))
    mask, mask_source = _value(
        prim, ("physxCollision:collisionFilterMask", "physics:collisionFilterMask")
    )
    approximation, approximation_source = _value(
        prim, ("physics:approximation", "physxCollision:approximation")
    )
    mesh = usd_geom.Mesh(prim) if prim.GetTypeName() == "Mesh" else None
    shape_type = str(approximation or ("mesh" if mesh is not None else prim.GetTypeName()))
    if shape_type in {"convexHull", "convexDecomposition", "convexMesh"}:
        shape_type = "convex_hull"
    return {
        "role": role,
        "articulation_path": str(body.GetPath()) if body is not None else "",
        "link_name": body.GetPath().name if body is not None else "",
        "collision_prim": str(prim.GetPath()),
        "shape_type": shape_type,
        "source_asset": str(stage.GetRootLayer().realPath or stage.GetRootLayer().identifier),
        "local_transform": _matrix(local),
        "world_transform": _matrix(xform),
        "collision_enabled": bool(collision_enabled),
        "collision_enabled_source": collision_source,
        "rigid_body": bool(body is not None and body_enabled is not False),
        "rigid_body_enabled_source": body_source,
        "contact_report_enabled": report_enabled,
        "contact_report_source": report_source,
        "collision_group": group,
        "collision_group_source": group_source,
        "collision_mask": mask,
        "collision_mask_source": mask_source,
        "approximation_source": approximation_source,
    }


def _audit_stage(
    path: Path, *, role: str, usd: Any, usd_geom: Any, usd_physics: Any, physx_schema: Any
) -> dict[str, object]:
    stage = usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"RUNTIME_BINDING_STAGE_OPEN_FAILURE:{path}")
    prims = [prim for prim in stage.Traverse() if prim.HasAPI(usd_physics.CollisionAPI)]
    rows = [
        _shape_row(
            prim,
            stage=stage,
            usd_geom=usd_geom,
            usd_physics=usd_physics,
            physx_schema=physx_schema,
            role=role,
        )
        for prim in prims
    ]
    return {
        "role": role,
        "stage": str(path.resolve()),
        "root_layer_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "shape_count": len(rows),
        "shapes": rows,
    }


def main() -> int:
    args = _parse_args()
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        app = AppLauncher(headless=True).app
    finally:
        sys.argv = original_argv
    try:
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        robot = _audit_stage(
            args.robot_usd,
            role="robot",
            usd=Usd,
            usd_geom=UsdGeom,
            usd_physics=UsdPhysics,
            physx_schema=PhysxSchema,
        )
        objects = {
            object_id: _audit_stage(
                Path(object_path),
                role="object",
                usd=Usd,
                usd_geom=UsdGeom,
                usd_physics=UsdPhysics,
                physx_schema=PhysxSchema,
            )
            for object_id, object_path in args.object_usd
        }
        payload = {
            "schema_version": "PhysicalSceneRuntimeBindingV1",
            "authority": "composed_runtime_usd_collision_shapes",
            "robot": robot,
            "objects": objects,
            "validation": {
                "robot_shape_count": robot["shape_count"],
                "object_shape_counts": {
                    key: value["shape_count"] for key, value in objects.items()
                },
                "all_collision_prims_have_body": all(
                    bool(row["articulation_path"]) for row in robot["shapes"]
                ),
                "all_transforms_finite": all(
                    all(
                        abs(float(component)) < float("inf")
                        for line in row["world_transform"]
                        for component in line
                    )
                    for row in robot["shapes"]
                    for _ in (0,)
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "robot_shapes": robot["shape_count"],
                    "objects": payload["validation"]["object_shape_counts"],
                }
            ),
            flush=True,
        )
        return 0
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
