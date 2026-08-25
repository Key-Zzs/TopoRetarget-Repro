"""Ignored USD wrapper for the explicit virtual 3P+3R wrist articulation.

This is an engineering world-to-wrist actuator, not a real arm.  It composes
the frozen C.1 hand asset without changing that source file.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .asset_contracts import sha256_file
from .d6_wrist_asset import D6_WRIST_PROFILES
from .explicit_virtual_wrist import EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER

_TRANSLATION_LIMIT_M = 0.40
_ROTATION_LIMIT_DEG = 179.0
_INTERMEDIATE_LINK_MASS_KG = 1.0e-3
_INTERMEDIATE_LINK_INERTIA_KGM2 = 1.0e-6


def explicit_virtual_wrist_recipe(
    profile_identifier: str = "nominal", *, continuous_virtual_wrist_angles: bool = False
) -> dict[str, Any]:
    """Return the no-Isaac recipe for the six serial articulation joints."""

    profile = next(
        (
            candidate
            for candidate in D6_WRIST_PROFILES
            if candidate.identifier == profile_identifier
        ),
        None,
    )
    if profile is None:
        valid = ", ".join(candidate.identifier for candidate in D6_WRIST_PROFILES)
        raise ValueError(
            f"unknown explicit virtual wrist profile {profile_identifier!r}; expected {valid}"
        )
    return {
        "schema_version": "toporetarget.stage16c3.explicit_virtual_wrist_wrapper.v1",
        "implementation": "finite_virtual_6d_wrist_actuator_v1",
        "articulation_model": "explicit_serial_3p3r",
        "engineering_model": "abstract_6dof_wrist_not_real_arm",
        "labels": [
            "ENGINEERING_WRIST_ACTUATION",
            "ABSTRACT_6DOF_WRIST_ACTUATOR",
            "NOT_A_REAL_ARM_MODEL",
            "NOT_PAPER_MINIMAL_CONTROLLER",
        ],
        "base_asset_policy": "reference_frozen_c1_asset_without_mutation",
        "anchor": {"kind": "fixed_articulation_root", "path": "VirtualWristAnchor"},
        "joint_order": list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER),
        "joint_types": {
            **{name: "PrismaticJoint" for name in EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[:3]},
            **{name: "RevoluteJoint" for name in EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[3:]},
        },
        "joint_axes": {
            name: axis
            for name, axis in zip(
                EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER,
                ("X", "Y", "Z", "X", "Y", "Z"),
                strict=True,
            )
        },
        "joint_position_limits_enforced": True,
        "finger_joint_position_limits_enforced": True,
        "virtual_wrist_translation_limits_enforced": True,
        "virtual_wrist_rotation_limits_enforced": not continuous_virtual_wrist_angles,
        "continuous_virtual_wrist_angles": bool(continuous_virtual_wrist_angles),
        "translation_limits_m": [-_TRANSLATION_LIMIT_M, _TRANSLATION_LIMIT_M],
        "rotation_limits_deg": (
            None if continuous_virtual_wrist_angles else [-_ROTATION_LIMIT_DEG, _ROTATION_LIMIT_DEG]
        ),
        "target_conversion": "SE3_target_to_serial_xyz_inverse_kinematics",
        "policy_rotation_residual": "rotation_vector_then_quaternion_not_euler",
        "rotation_singularity": "serial_xyz_pitch_at_plus_or_minus_90_deg",
        "profile": asdict(profile),
        "intermediate_link_dynamics": {
            "mass_kg": _INTERMEDIATE_LINK_MASS_KG,
            "diagonal_inertia_kgm2": _INTERMEDIATE_LINK_INERTIA_KGM2,
            "collision": "none",
        },
        "rollout_prohibitions": [
            "no_wrist_root_pose_or_velocity_state_writes",
            "no_direct_object_state_writes",
            "no_external_wrench_fallback",
            "no_hidden_attachment_to_object",
            "no_real_arm",
        ],
    }


def _apply_rigid_body(stage: Any, path: str, *, mass_kg: float) -> Any:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(
        Gf.Vec3f(
            _INTERMEDIATE_LINK_INERTIA_KGM2,
            _INTERMEDIATE_LINK_INERTIA_KGM2,
            _INTERMEDIATE_LINK_INERTIA_KGM2,
        )
    )
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateDisableGravityAttr(True)
    physx.CreateLinearDampingAttr(0.0)
    physx.CreateAngularDampingAttr(0.0)
    return prim


def _define_joint(
    stage: Any,
    *,
    root_path: str,
    name: str,
    axis: str,
    body0: Any,
    body1: Any,
    profile: dict[str, Any],
    continuous_virtual_wrist_angles: bool,
) -> Any:
    from pxr import Gf, PhysxSchema, UsdPhysics

    is_translation = name.startswith("virtual_prismatic")
    joint_cls = UsdPhysics.PrismaticJoint if is_translation else UsdPhysics.RevoluteJoint
    joint = joint_cls.Define(stage, f"{root_path}/Joints/{name}")
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(identity)
    joint.CreateLocalRot1Attr(identity)
    joint.CreateAxisAttr(axis)
    if is_translation:
        joint.CreateLowerLimitAttr(-_TRANSLATION_LIMIT_M)
        joint.CreateUpperLimitAttr(_TRANSLATION_LIMIT_M)
        drive_name = "linear"
        stiffness = profile["translation_stiffness_npm"]
        damping = profile["translation_damping_ns_per_m"]
        effort = profile["translation_effort_limit_n"]
        velocity = profile["translation_velocity_limit_mps"]
    else:
        if not continuous_virtual_wrist_angles:
            joint.CreateLowerLimitAttr(-_ROTATION_LIMIT_DEG)
            joint.CreateUpperLimitAttr(_ROTATION_LIMIT_DEG)
        drive_name = "angular"
        # USD angular drive gains are stored per degree; Isaac Lab's runtime
        # actuator configuration converts the same SI/radian values again.
        stiffness = profile["rotation_stiffness_nm_per_rad"] * math.pi / 180.0
        damping = profile["rotation_damping_nm_s_per_rad"] * math.pi / 180.0
        effort = profile["rotation_effort_limit_nm"]
        velocity = profile["rotation_velocity_limit_radps"] * 180.0 / math.pi
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), drive_name)
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(effort)
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateMaxJointVelocityAttr(velocity)
    return joint.GetPrim()


def write_explicit_virtual_wrist_wrapper(
    *,
    base_asset: Path,
    output_usda: Path,
    profile_identifier: str = "nominal",
    continuous_virtual_wrist_angles: bool = False,
) -> dict[str, Any]:
    """Compose the frozen hand below six explicit articulation joints."""

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    recipe = explicit_virtual_wrist_recipe(
        profile_identifier,
        continuous_virtual_wrist_angles=continuous_virtual_wrist_angles,
    )
    if not base_asset.is_file():
        raise FileNotFoundError(f"C3_EXPLICIT_WRIST_BASE_ASSET_MISSING: {base_asset}")
    output_usda.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usda))
    root_path = "/stage16c3_explicit_virtual_wrist"
    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
    stage.SetDefaultPrim(root_prim)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    articulation_api.CreateEnabledSelfCollisionsAttr(False)

    hand = UsdGeom.Xform.Define(stage, f"{root_path}/Hand")
    relative_base = os.path.relpath(base_asset, output_usda.parent)
    hand.GetPrim().GetReferences().AddReference(relative_base)
    # The frozen source carries its floating articulation root on the default
    # prim.  Suppress that composed API only in this wrapper so the upstream
    # virtual joints and all hand joints form one reduced-coordinate tree.
    hand_prim = hand.GetPrim()
    hand_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    hand_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    anchor = _apply_rigid_body(
        stage,
        f"{root_path}/VirtualLinks/VirtualWristAnchor",
        mass_kg=_INTERMEDIATE_LINK_MASS_KG,
    )
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/Joints/VirtualWristAnchorFixed")
    fixed.CreateBody1Rel().SetTargets([anchor.GetPath()])

    child_paths = [
        f"{root_path}/VirtualLinks/virtual_prismatic_x_link",
        f"{root_path}/VirtualLinks/virtual_prismatic_y_link",
        f"{root_path}/VirtualLinks/virtual_prismatic_z_link",
        f"{root_path}/VirtualLinks/virtual_revolute_x_link",
        f"{root_path}/VirtualLinks/virtual_revolute_y_link",
    ]
    children = [
        _apply_rigid_body(stage, path, mass_kg=_INTERMEDIATE_LINK_MASS_KG) for path in child_paths
    ]
    wrist = stage.GetPrimAtPath(f"{root_path}/Hand/r_wrist")
    if not wrist.IsValid() or not wrist.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("C3_EXPLICIT_WRIST_REFERENCED_WRIST_RIGID_BODY_MISSING")
    body_pairs = list(zip([anchor, *children], [*children, wrist], strict=True))
    joint_prims = []
    for name, axis, (body0, body1) in zip(
        EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER,
        ("X", "Y", "Z", "X", "Y", "Z"),
        body_pairs,
        strict=True,
    ):
        joint_prims.append(
            _define_joint(
                stage,
                root_path=root_path,
                name=name,
                axis=axis,
                body0=body0,
                body1=body1,
                profile=recipe["profile"],
                continuous_virtual_wrist_angles=continuous_virtual_wrist_angles,
            )
        )
    stage.GetRootLayer().Save()
    return {
        **recipe,
        "base_asset": str(base_asset),
        "base_asset_sha256": sha256_file(base_asset),
        "generated_usda": str(output_usda),
        "generated_usda_sha256": sha256_file(output_usda),
        "articulation_root_path": root_path,
        "hand_source_articulation_root_suppressed": True,
        "joint_paths": [str(prim.GetPath()) for prim in joint_prims],
        "status": "EXPLICIT_VIRTUAL_WRIST_WRAPPER_AUTHORED_PENDING_GPU_TENSOR_CONTRACT",
    }


__all__ = [
    "explicit_virtual_wrist_recipe",
    "write_explicit_virtual_wrist_wrapper",
]
