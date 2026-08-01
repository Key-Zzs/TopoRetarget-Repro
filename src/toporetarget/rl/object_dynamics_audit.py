"""Reproducible object mass, inertia, support, and contact-demand audit helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .environments.world_wrist_backend import WorldWristFingerBackend

OBJECT_DYNAMICS_AUDIT_ID = "stage16b_object_dynamics_audit_v1"


def reference_accelerations(timestamps: np.ndarray, twists_world: np.ndarray) -> np.ndarray:
    """Finite-difference world twist with one-sided endpoint derivatives."""

    time = np.asarray(timestamps, dtype=np.float64)
    twist = np.asarray(twists_world, dtype=np.float64)
    if time.ndim != 1 or twist.shape != (len(time), 6) or len(time) < 2:
        raise ValueError("timestamps/twists must have shapes [T] and [T,6], T >= 2")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return np.gradient(twist, time, axis=0, edge_order=1)


def inertial_wrench_demand(
    backend: WorldWristFingerBackend,
) -> dict[str, Any]:
    """Compute the no-gravity net wrench implied by the geometric reference."""

    reference = backend.reference
    acceleration = reference_accelerations(reference.timestamps, reference.object_twist_world_ref)
    mass = float(backend.model.body_mass[backend.object_body_id])
    inertia_body = np.diag(backend.model.body_inertia[backend.object_body_id])
    forces: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    for pose, twist, accel in zip(
        reference.object_pose_world_ref,
        reference.object_twist_world_ref,
        acceleration,
        strict=True,
    ):
        rotation = pose[:3, :3]
        inertia_world = rotation @ inertia_body @ rotation.T
        angular_velocity = twist[3:]
        forces.append(mass * accel[:3])
        torques.append(
            inertia_world @ accel[3:] + np.cross(angular_velocity, inertia_world @ angular_velocity)
        )
    force = np.asarray(forces)
    torque = np.asarray(torques)
    return {
        "method": "finite_difference_reference_twist_rigid_body_inertial_wrench_zero_gravity",
        "force_norm_n": np.linalg.norm(force, axis=1).tolist(),
        "torque_norm_nm": np.linalg.norm(torque, axis=1).tolist(),
        "max_force_n": float(np.max(np.linalg.norm(force, axis=1))),
        "max_torque_nm": float(np.max(np.linalg.norm(torque, axis=1))),
        "frames_with_nontrivial_demand": int(
            np.count_nonzero(
                (np.linalg.norm(force, axis=1) > 1e-3) | (np.linalg.norm(torque, axis=1) > 1e-5)
            )
        ),
    }


def static_reference_contact_proxy(
    backend: WorldWristFingerBackend,
) -> dict[str, Any]:
    """Audit exact-reference geometry contacts using reset-only state writes."""

    counts: list[int] = []
    penetrations: list[float] = []
    pairs: set[tuple[str, str]] = set()
    for index in range(backend.reference.frame_count):
        backend.reset(reference_index=index)
        contact = backend.contact_summary()
        counts.append(int(contact["hand_object_contact_count"]))
        penetrations.append(float(contact["hand_object_max_penetration_m"]))
        for row in contact["contacts"]:
            pairs.add((str(row["geom1"]), str(row["geom2"])))
    return {
        "semantics": "reference_geometry_proxy_not_ground_truth_contact_or_force",
        "reset_only_diagnostic": True,
        "counts": counts,
        "contact_frame_count": int(np.count_nonzero(counts)),
        "contact_frame_fraction": float(np.mean(np.asarray(counts) > 0)),
        "first_contact_frame": next((i for i, count in enumerate(counts) if count), None),
        "max_penetration_m": max(penetrations, default=0.0),
        "geom_pairs": [list(pair) for pair in sorted(pairs)],
    }


def support_model_audit(backend: WorldWristFingerBackend) -> dict[str, Any]:
    """Describe modeled support without inferring unprovided dataset semantics."""

    gravity = np.asarray(backend.model.opt.gravity, dtype=np.float64)
    damping = backend.model.dof_damping[backend.object_dof_address : backend.object_dof_address + 6]
    no_gravity = bool(np.allclose(gravity, 0.0))
    no_damping = bool(np.allclose(damping, 0.0))
    return {
        "gravity_mps2": gravity.tolist(),
        "object_joint": "freejoint_6dof",
        "object_dof_damping": damping.tolist(),
        "ground_geom_present": False,
        "support_constraint_present": False,
        "modeled_support": "none",
        "classification": (
            "UNSUPPORTED_FREE_BODY_ZERO_GRAVITY_NO_DAMPING"
            if no_gravity and no_damping
            else "FREE_BODY_WITH_EXTERNAL_FIELD_OR_DAMPING"
        ),
        "dataset_support_provenance": "unresolved",
        "formal_support_inference_allowed": False,
    }


def impulse_sensitivity_candidates(
    *, mass_kg: float, principal_inertia_kgm2: np.ndarray, impulse_ns: float
) -> list[dict[str, Any]]:
    """Report bounded shared scale counterfactuals; do not select by tracking score."""

    inertia = np.asarray(principal_inertia_kgm2, dtype=np.float64)
    if mass_kg <= 0.0 or inertia.shape != (3,) or np.any(inertia <= 0.0):
        raise ValueError("mass and principal inertia must be positive")
    if impulse_ns < 0.0:
        raise ValueError("impulse must be non-negative")
    rows = []
    for scale in (0.5, 1.0, 2.0, 5.0):
        rows.append(
            {
                "shared_mass_inertia_scale": scale,
                "mass_kg": mass_kg * scale,
                "principal_inertia_kgm2": (inertia * scale).tolist(),
                "linear_delta_v_upper_proxy_mps": impulse_ns / (mass_kg * scale),
                "angular_delta_omega_unit_lever_1m_upper_proxy_radps": (
                    impulse_ns / float(np.min(inertia * scale))
                ),
                "physical_provenance_eligible": False,
            }
        )
    return rows


__all__ = [
    "OBJECT_DYNAMICS_AUDIT_ID",
    "impulse_sensitivity_candidates",
    "inertial_wrench_demand",
    "reference_accelerations",
    "static_reference_contact_proxy",
    "support_model_audit",
]
