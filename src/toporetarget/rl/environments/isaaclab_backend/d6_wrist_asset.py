"""Recipe for the optional finite Stage 16-C.3 D6 wrist wrapper.

The C.1 hand asset stays byte-for-byte untouched.  This module only creates a
small ignored USD composition layer which references that asset, adds one
kinematic anchor, and connects the hand wrist to it through an explicit D6
joint.  Isaac modules are deliberately imported inside the writer so this
recipe remains safe to import in the normal test environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .asset_contracts import sha256_file

D6_WRIST_AXES = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")


@dataclass(frozen=True)
class FiniteD6WristProfile:
    """One global, non-clip-specific D6 drive profile."""

    identifier: str
    translation_stiffness_npm: float
    translation_damping_ns_per_m: float
    translation_effort_limit_n: float
    translation_velocity_limit_mps: float
    rotation_stiffness_nm_per_rad: float
    rotation_damping_nm_s_per_rad: float
    rotation_effort_limit_nm: float
    rotation_velocity_limit_radps: float


D6_WRIST_PROFILES = (
    FiniteD6WristProfile(
        identifier="conservative",
        translation_stiffness_npm=400.0,
        translation_damping_ns_per_m=60.0,
        translation_effort_limit_n=30.0,
        translation_velocity_limit_mps=1.0,
        rotation_stiffness_nm_per_rad=3.0,
        rotation_damping_nm_s_per_rad=0.35,
        rotation_effort_limit_nm=3.0,
        rotation_velocity_limit_radps=4.0,
    ),
    FiniteD6WristProfile(
        identifier="nominal",
        translation_stiffness_npm=800.0,
        translation_damping_ns_per_m=100.0,
        translation_effort_limit_n=50.0,
        translation_velocity_limit_mps=1.5,
        rotation_stiffness_nm_per_rad=6.0,
        rotation_damping_nm_s_per_rad=0.60,
        rotation_effort_limit_nm=5.0,
        rotation_velocity_limit_radps=5.0,
    ),
    FiniteD6WristProfile(
        identifier="high_authority_bounded",
        # Reference-envelope qualification: m=5.6207 kg and a_max=4.102
        # m/s^2 imply about 23.1 N feed-forward demand.  These gains retain
        # a finite 500 N ceiling while approaching critical damping for the
        # explicit serial translation chain.
        translation_stiffness_npm=10000.0,
        translation_damping_ns_per_m=500.0,
        translation_effort_limit_n=500.0,
        translation_velocity_limit_mps=2.0,
        # I_max=0.14401 kg m^2 and alpha_max=20.818 rad/s^2 imply about
        # 3.0 Nm.  The finite 500 Nm cap covers serial-coordinate coupling and
        # bounded substep transients without using a kinematic drive.
        rotation_stiffness_nm_per_rad=3000.0,
        rotation_damping_nm_s_per_rad=45.0,
        rotation_effort_limit_nm=500.0,
        rotation_velocity_limit_radps=6.0,
    ),
)


def d6_wrist_recipe(profile_identifier: str = "nominal") -> dict[str, Any]:
    """Return the fully explicit wrapper recipe without importing Isaac."""

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
            f"unknown D6 wrist profile {profile_identifier!r}; expected one of {valid}"
        )
    return {
        "schema_version": "toporetarget.stage16c3.finite_d6_wrist_wrapper.v1",
        "implementation": "finite_d6_wrist_actuator_v1",
        "engineering_model": "abstract_6dof_wrist_not_real_arm",
        "base_asset_policy": "reference_frozen_c1_asset_without_mutation",
        "anchor": {"kind": "static", "path": "WristAnchor"},
        "joint": {
            "type": "PhysicsJoint",
            "constraint_model": "D6_via_six_LimitAPI_and_DriveAPI_axes",
            "path": "WristD6",
            "body0": "WristAnchor",
            "body1": "Hand/r_wrist",
            "axes": list(D6_WRIST_AXES),
            "translation_limits_m": [-0.30, 0.30],
            "rotation_limits_deg": [-100.0, 100.0],
            "target_orientation_representation": "quaternion_then_rotation_log",
            "target_orientation_residual_representation": "not_euler",
        },
        "profile": asdict(profile),
        "rollout_prohibitions": [
            "no_wrist_pose_or_velocity_state_writes",
            "no_kinematic_root_or_object_attachment",
            "no_direct_object_state_writes",
        ],
    }


def write_d6_wrist_wrapper(
    *,
    base_asset: Path,
    output_usda: Path,
    profile_identifier: str = "nominal",
) -> dict[str, Any]:
    """Create an ignored USD layer plus a provenance manifest.

    Call only after ``AppLauncher`` has started.  The authored joint is kept
    inside the wrapper hierarchy and *not* excluded from the articulation; a
    later runtime probe determines whether PhysX publishes it as tensor DoFs.
    """

    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    recipe = d6_wrist_recipe(profile_identifier)
    if not base_asset.is_file():
        raise FileNotFoundError(f"C3_D6_BASE_ASSET_MISSING: {base_asset}")
    output_usda.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usda))
    root = UsdGeom.Xform.Define(stage, "/stage16c3_d6_wrist")
    stage.SetDefaultPrim(root.GetPrim())
    hand = UsdGeom.Xform.Define(stage, "/stage16c3_d6_wrist/Hand")
    # The relative reference makes the wrapper relocatable within the ignored
    # generated-assets directory while retaining the exact source relationship.
    hand.GetPrim().GetReferences().AddReference(
        "../wuji_hand2_beta1/configuration/wujihand2_physics.usd"
    )
    anchor = UsdGeom.Xform.Define(stage, "/stage16c3_d6_wrist/WristAnchor")
    anchor_prim = anchor.GetPrim()
    # Isaac Sim 5.1 represents D6 constraints as a generic PhysicsJoint with
    # six named LimitAPI/DriveAPI axes; ``UsdPhysics.D6Joint`` is not exported
    # by this installed pxr module.
    d6 = UsdPhysics.Joint.Define(stage, "/stage16c3_d6_wrist/WristD6")
    d6_prim = d6.GetPrim()
    d6.CreateBody0Rel().SetTargets([anchor_prim.GetPath()])
    d6.CreateBody1Rel().SetTargets([hand.GetPrim().GetPath().AppendPath("r_wrist")])
    d6.CreateLocalPos0Attr((0.0, 0.0, 0.0))
    d6.CreateLocalPos1Attr((0.0, 0.0, 0.0))
    identity_quaternion = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    d6.CreateLocalRot0Attr(identity_quaternion)
    d6.CreateLocalRot1Attr(identity_quaternion)
    d6.CreateJointEnabledAttr(True)
    profile = recipe["profile"]
    for axis in D6_WRIST_AXES:
        limit = UsdPhysics.LimitAPI.Apply(d6_prim, axis)
        if axis.startswith("trans"):
            lower, upper = recipe["joint"]["translation_limits_m"]
            stiffness = profile["translation_stiffness_npm"]
            damping = profile["translation_damping_ns_per_m"]
            effort = profile["translation_effort_limit_n"]
        else:
            lower, upper = recipe["joint"]["rotation_limits_deg"]
            stiffness = profile["rotation_stiffness_nm_per_rad"]
            damping = profile["rotation_damping_nm_s_per_rad"]
            effort = profile["rotation_effort_limit_nm"]
        limit.CreateLowAttr(lower)
        limit.CreateHighAttr(upper)
        drive = UsdPhysics.DriveAPI.Apply(d6_prim, axis)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(effort)
    # PhysX exposes maxJointVelocity once per generic D6 joint (rather than
    # per DriveAPI axis).  Keep it finite at the strictest common cap authored
    # by the selected profile.
    PhysxSchema.PhysxJointAPI.Apply(d6_prim).CreateMaxJointVelocityAttr(
        max(
            profile["translation_velocity_limit_mps"],
            profile["rotation_velocity_limit_radps"],
        )
    )
    stage.GetRootLayer().Save()
    return {
        **recipe,
        "base_asset": str(base_asset),
        "base_asset_sha256": sha256_file(base_asset),
        "generated_usda": str(output_usda),
        "generated_usda_sha256": sha256_file(output_usda),
        "status": "D6_WRAPPER_AUTHORED_PENDING_GPU_TENSOR_CONTRACT",
    }
