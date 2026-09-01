"""Read-only metrics used by Physical Policy Failure Localization V1.

These helpers deliberately consume recorded rollout arrays.  They neither own an
environment nor mutate a policy, so they are safe to use for frozen-trace audits.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def action_saturation(actions: np.ndarray, *, near: float = 0.9) -> dict[str, np.ndarray | float]:
    """Return exact/near bound fractions for normalized 26-D actions."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 26:
        raise ValueError("expected [frames, 26] normalized actions")
    return {
        "near_per_dim": np.mean(np.abs(values) > near, axis=0),
        "exact_per_dim": np.mean(np.abs(values) >= 1.0 - 1e-6, axis=0),
        "near_all": float(np.mean(np.abs(values) > near)),
        "exact_all": float(np.mean(np.abs(values) >= 1.0 - 1e-6)),
    }


def tracking_errors(
    wrist_pose: np.ndarray,
    wrist_target: np.ndarray,
    finger_q: np.ndarray,
    finger_target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute command-to-actual position and joint error without a convention guess."""
    wrist_pose = np.asarray(wrist_pose, dtype=np.float64)
    wrist_target = np.asarray(wrist_target, dtype=np.float64)
    finger_q = np.asarray(finger_q, dtype=np.float64)
    finger_target = np.asarray(finger_target, dtype=np.float64)
    if wrist_pose.shape != wrist_target.shape or wrist_pose.ndim != 2 or wrist_pose.shape[1] < 3:
        raise ValueError("wrist actual/target pose shapes do not match")
    if finger_q.shape != finger_target.shape:
        raise ValueError("finger actual/target shapes do not match")
    return {
        "wrist_translation_m": np.linalg.norm(wrist_pose[:, :3] - wrist_target[:, :3], axis=1),
        "finger_joint_abs_rad": np.abs(finger_q - finger_target),
    }


def force_feasibility(
    points: np.ndarray,
    forces: np.ndarray,
    object_com: np.ndarray,
    gravity_force: np.ndarray,
    *,
    min_contacts: int = 2,
) -> tuple[str, int, int, float]:
    """A conservative recorded-contact wrench proxy, not a Ferrari--Canny claim."""
    points = np.asarray(points, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    if points.shape != forces.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points and forces must be [contacts, 3]")
    active = np.linalg.norm(forces, axis=1) > 1e-8
    points, forces = points[active], forces[active]
    count = len(points)
    if count < min_contacts:
        return "INSUFFICIENT_CONTACTS", count, 0, float("nan")
    normals = forces / np.maximum(np.linalg.norm(forces, axis=1, keepdims=True), 1e-12)
    normal_rank = int(np.linalg.matrix_rank(normals, tol=1e-5))
    if normal_rank < 2:
        return "CONTACT_GEOMETRY_DEGENERATE", count, normal_rank, float("nan")
    torque = np.cross(points - np.asarray(object_com, dtype=np.float64), forces).sum(axis=0)
    residual = np.linalg.norm(
        forces.sum(axis=0) + np.asarray(gravity_force, dtype=np.float64)
    ) + np.linalg.norm(torque)
    status = (
        "GRASP_FEASIBLE"
        if residual < max(0.25 * np.linalg.norm(gravity_force), 1e-6)
        else "GRASP_MARGINAL"
    )
    return status, count, normal_rank, float(residual)


def viability_probability(
    labels: Iterable[str], probabilities: Iterable[float]
) -> dict[str, float]:
    """Aggregate an explicitly supplied RSI distribution without changing it."""
    labels = list(labels)
    probabilities = np.asarray(list(probabilities), dtype=np.float64)
    if (
        len(labels) != len(probabilities)
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0)
    ):
        raise ValueError("labels/probabilities must be aligned and sum to one")
    return {
        "viable": float(probabilities[[x == "REFERENCE_STATE_VIABLE" for x in labels]].sum()),
        "marginal": float(probabilities[[x == "REFERENCE_STATE_MARGINAL" for x in labels]].sum()),
        "nonviable": float(probabilities[[x == "REFERENCE_STATE_NONVIABLE" for x in labels]].sum()),
    }


def reward_product_error(groups: np.ndarray, total: np.ndarray) -> float:
    """Maximum product-reconstruction error for [R_obj, R_hand, R_int, R_reg]."""
    groups, total = np.asarray(groups, dtype=np.float64), np.asarray(total, dtype=np.float64)
    if groups.ndim != 2 or groups.shape[1] != 4 or groups.shape[0] != total.shape[0]:
        raise ValueError("invalid reward shapes")
    return float(np.max(np.abs(np.prod(groups, axis=1) - total)))


def forgetting(best: float, final: float, *, tolerance: float = 1e-9) -> bool:
    """A result is forgotten only after a genuine earlier improvement."""
    return bool(best > tolerance and final + tolerance < best)
