"""Runtime finite-box support asset helpers.

The generated asset is a static/kinematic rigid box with finite dimensions.
It is an explicit collision actor, never an object attachment or force source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import FinitePlanarSupportProxy


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

    center = np.asarray(proxy.table_pose[:3], dtype=np.float64)
    normal = np.asarray(proxy.plane_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent_u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(tangent_u) < 1.0e-8:
        tangent_u = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(normal, tangent_u)
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
    "isaac_rigid_object_config_kwargs",
    "table_top_corners",
    "write_finite_planar_support_usda",
]
