"""Runtime finite-box support asset helpers.

The generated asset is a static/kinematic rigid box with finite dimensions.
It is an explicit collision actor, never an object attachment or force source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .planar_inference import quaternion_to_rotation_matrix
from .types import (
    FinitePlanarSupportProxy,
    SupportCollisionContractV1,
    SupportType,
)


def support_collision_policy(support_type: SupportType | str) -> dict[str, object]:
    """Return the frozen pairwise collision matrix for one support type."""

    return SupportCollisionContractV1().policy(support_type).as_dict()


def apply_hand_support_pair_filter(
    stage: object,
    *,
    hand_prim_paths: tuple[str, ...],
    support_prim_paths: tuple[str, ...],
    support_type: SupportType | str,
) -> dict[str, object]:
    """Author USD filtered-pair relationships without disabling support globally.

    Object/support collision is never touched.  For inferred proxies only, the
    hand articulation root is filtered against the proxy rigid-body root in
    each environment.
    """

    policy = SupportCollisionContractV1().policy(support_type)
    if len(hand_prim_paths) != len(support_prim_paths) or not hand_prim_paths:
        raise ValueError("SUPPORT_COLLISION_PAIR_PATH_CARDINALITY_INVALID")
    if policy.hand_support_collision:
        return {
            **policy.as_dict(),
            "status": "NO_FILTER_REQUIRED_REAL_SUPPORT_COLLISION_ACTIVE",
            "filtered_pairs": [],
        }
    if not policy.object_support_collision:
        raise ValueError("SUPPORT_COLLISION_UNRESOLVED_CANNOT_ENTER_RUNTIME")
    from pxr import Sdf, UsdPhysics

    filtered_pairs: list[dict[str, str]] = []
    for hand_path, support_path in zip(hand_prim_paths, support_prim_paths, strict=True):
        hand = stage.GetPrimAtPath(hand_path)  # type: ignore[attr-defined]
        support = stage.GetPrimAtPath(support_path)  # type: ignore[attr-defined]
        if not hand.IsValid() or not support.IsValid():
            raise RuntimeError(f"SUPPORT_COLLISION_FILTER_PRIM_MISSING:{hand_path}:{support_path}")
        relationship = UsdPhysics.FilteredPairsAPI.Apply(hand).CreateFilteredPairsRel()
        relationship.AddTarget(Sdf.Path(support_path))
        targets = {str(path) for path in relationship.GetTargets()}
        if support_path not in targets:
            raise RuntimeError("SUPPORT_COLLISION_FILTER_RELATIONSHIP_NOT_AUTHORED")
        filtered_pairs.append({"hand": hand_path, "support": support_path})
    return {
        **policy.as_dict(),
        "status": "PAIRWISE_HAND_SUPPORT_FILTER_AUTHORED",
        "filtered_pairs": filtered_pairs,
    }


def write_finite_planar_support_usda(
    proxy: FinitePlanarSupportProxy,
    destination: Path,
    *,
    prim_name: str = "FinitePlanarSupportProxy",
) -> Path:
    """Write a minimal Isaac-compatible USD box collision proxy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    # The asset is authored in a local support frame.  Isaac Lab applies the
    # audited world pose through ``RigidObjectCfg.init_state`` at spawn time;
    # this avoids relying on a nested reference transform being preserved as
    # the actor root pose.
    x, y, z = 0.0, 0.0, 0.0
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    half_x, half_y = proxy.table_extent[0] / 2.0, proxy.table_extent[1] / 2.0
    half_z = proxy.table_thickness / 2.0
    text = f'''#usda 1.0
(
    defaultPrim = "{prim_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{prim_name}" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
)
{{
    float3 xformOp:scale = ({half_x}, {half_y}, {half_z})
    quatf xformOp:orient = ({qw}, {qx}, {qy}, {qz})
    double3 xformOp:translate = ({x}, {y}, {z})
    uniform token[] xformOpOrder = ["xformOp:scale", "xformOp:orient", "xformOp:translate"]
    bool physics:rigidBodyEnabled = 1
    bool physics:kinematicEnabled = 1
    float physxRigidBody:linearDamping = 0
    float physxRigidBody:angularDamping = 0

    def Cube "Collision" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsCubeCollisionAPI",
            "MaterialBindingAPI"]
    )
    {{
        double size = 2
        bool physics:collisionEnabled = 1
    }}
    def Cube "Visual"
    {{
        double size = 2
        color3f[] primvars:displayColor = [(0.15, 0.55, 0.85)]
    }}
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:staticFriction = {proxy.material.static_friction}
        float physics:dynamicFriction = {proxy.material.dynamic_friction}
        float physics:restitution = {proxy.material.restitution}
    }}
}}
'''
    destination.write_text(text, encoding="utf-8")
    return destination


def table_top_corners(proxy: FinitePlanarSupportProxy) -> np.ndarray:
    """Return top-plane corners for visualization and geometry overlays."""

    pose = np.asarray(proxy.table_pose, dtype=np.float64)
    rotation = quaternion_to_rotation_matrix(pose[None, 3:])[0]
    tangent_u, tangent_v, normal = rotation.T
    declared_normal = np.asarray(proxy.plane_normal, dtype=np.float64)
    declared_normal /= np.linalg.norm(declared_normal)
    if float(np.dot(normal, declared_normal)) < 1.0 - 1.0e-6:
        raise ValueError("SUPPORT_TABLE_POSE_NORMAL_MISMATCH")
    center = pose[:3]
    if abs(float(np.dot(center, declared_normal)) - proxy.plane_offset) > 1.0e-6:
        raise ValueError("SUPPORT_TABLE_TOP_PLANE_MISMATCH")
    return np.asarray(
        [
            center
            + sign_u * proxy.table_extent[0] / 2.0 * tangent_u
            + sign_v * proxy.table_extent[1] / 2.0 * tangent_v
            for sign_u, sign_v in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        ]
    )


def isaac_rigid_object_config_kwargs(proxy_asset: Path, prim_path: str) -> dict[str, object]:
    """Return explicit config values for a caller that owns Isaac imports."""

    return {
        "prim_path": prim_path,
        "usd_path": str(proxy_asset),
        "static_actor": True,
        "kinematic_enabled": True,
        "external_guidance": False,
        "object_attachment": False,
        "support_force_injection": False,
    }


__all__ = [
    "apply_hand_support_pair_filter",
    "isaac_rigid_object_config_kwargs",
    "support_collision_policy",
    "table_top_corners",
    "write_finite_planar_support_usda",
]
