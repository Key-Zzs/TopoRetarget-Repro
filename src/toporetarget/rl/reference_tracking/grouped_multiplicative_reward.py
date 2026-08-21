"""Grouped multiplicative Stage16 reward with reference-scoped hand tracking.

This is an Isaac-free contract.  The Stage16 runtime supplies immutable
reference distances, exact visual-mesh distances for the actual fingertips,
and the existing strict V4 pair-force inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from toporetarget.rl.environments.isaaclab_backend.reward_terms import (
    Stage16WorldWristRewardProfileV1,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV4,
    ppo26d_reward_v4_strict_per_finger_contact_terms,
)

LEGACY_ADDITIVE = "legacy_additive"
GROUPED_MULTIPLICATIVE_V1 = "grouped_multiplicative_v1"
REWARD_MODES = (LEGACY_ADDITIVE, GROUPED_MULTIPLICATIVE_V1)


@dataclass(frozen=True)
class GroupedMultiplicativeRewardV1:
    """Global, non-object-tuned grouped reward parameters."""

    identifier: str = "Stage16GroupedMultiplicativeRewardV1"
    distance_scope_m: float = 0.20
    proximity_tolerance_m: float = 0.03
    proximity_scale_per_m: float = 1.0 / 0.03
    epsilon: float = 1.0e-12
    object_exponent: float = 1.0
    hand_exponent: float = 1.0
    interaction_exponent: float = 1.0
    regularization_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.distance_scope_m != 0.20:
            raise ValueError("GROUPED_V1_DISTANCE_SCOPE_DRIFT")
        if self.proximity_tolerance_m != 0.03:
            raise ValueError("GROUPED_V1_PROXIMITY_TOLERANCE_DRIFT")
        if self.proximity_scale_per_m != 1.0 / self.proximity_tolerance_m:
            raise ValueError("GROUPED_V1_PROXIMITY_SCALE_DRIFT")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("GROUPED_V1_EPSILON_INVALID")
        if any(
            value != 1.0
            for value in (
                self.object_exponent,
                self.hand_exponent,
                self.interaction_exponent,
                self.regularization_exponent,
            )
        ):
            raise ValueError("GROUPED_V1_EXPONENTS_ARE_FROZEN_AT_ONE")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_obj_triangles(path: str | Path) -> torch.Tensor:
    """Load deterministic triangular OBJ geometry without optional trimesh deps."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    source = Path(path)
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            values = line.split()
            vertices.append((float(values[1]), float(values[2]), float(values[3])))
        elif line.startswith("f "):
            values = line.split()[1:]
            if len(values) != 3:
                raise ValueError(f"GROUPED_V1_OBJECT_MESH_NOT_TRIANGULATED:{source}")
            indices = tuple(int(value.split("/", maxsplit=1)[0]) - 1 for value in values)
            faces.append((indices[0], indices[1], indices[2]))
    if not vertices or not faces:
        raise ValueError(f"GROUPED_V1_OBJECT_MESH_EMPTY:{source}")
    vertex_tensor = torch.tensor(vertices, dtype=torch.float32)
    face_tensor = torch.tensor(faces, dtype=torch.long)
    triangles = vertex_tensor[face_tensor]
    if not bool(torch.isfinite(triangles).all()):
        raise ValueError(f"GROUPED_V1_OBJECT_MESH_NONFINITE:{source}")
    return triangles


def point_to_triangle_surface_distance(
    points: torch.Tensor,
    triangles: torch.Tensor,
    *,
    face_chunk_size: int = 512,
) -> torch.Tensor:
    """Return exact unsigned point-to-triangle distance on the input device."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("GROUPED_V1_POINTS_MUST_BE_[N,3]")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or triangles.shape[0] == 0:
        raise ValueError("GROUPED_V1_TRIANGLES_MUST_BE_[F,3,3]")
    if face_chunk_size <= 0:
        raise ValueError("GROUPED_V1_FACE_CHUNK_INVALID")
    if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(triangles).all()):
        raise FloatingPointError("GROUPED_V1_SURFACE_INPUT_NONFINITE")
    if points.shape[0] == 0:
        return torch.empty(0, dtype=points.dtype, device=points.device)

    best_distance2 = torch.full(
        (points.shape[0],), float("inf"), dtype=points.dtype, device=points.device
    )
    tiny = torch.finfo(points.dtype).tiny
    for start in range(0, triangles.shape[0], face_chunk_size):
        tri = triangles[start : start + face_chunk_size].to(
            device=points.device, dtype=points.dtype
        )
        a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
        point = points[:, None, :]
        ab, ac = b - a, c - a
        ap, bp, cp = point - a, point - b, point - c
        d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
        d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
        d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
        bary = torch.zeros((len(points), len(tri), 3), dtype=points.dtype, device=points.device)

        mask_a = (d1 <= 0) & (d2 <= 0)
        bary[..., 0] = torch.where(mask_a, 1.0, bary[..., 0])
        mask_b = (d3 >= 0) & (d4 <= d3)
        bary[..., 1] = torch.where(mask_b, 1.0, bary[..., 1])
        vc = d1 * d4 - d3 * d2
        mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        v_ab = d1 / (d1 - d3).clamp_min(tiny)
        bary[..., 0] = torch.where(mask_ab, 1.0 - v_ab, bary[..., 0])
        bary[..., 1] = torch.where(mask_ab, v_ab, bary[..., 1])
        mask_c = (d6 >= 0) & (d5 <= d6)
        bary[..., 2] = torch.where(mask_c, 1.0, bary[..., 2])
        vb = d5 * d2 - d1 * d6
        mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        w_ac = d2 / (d2 - d6).clamp_min(tiny)
        bary[..., 0] = torch.where(mask_ac, 1.0 - w_ac, bary[..., 0])
        bary[..., 2] = torch.where(mask_ac, w_ac, bary[..., 2])
        va = d3 * d6 - d5 * d4
        mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(tiny)
        bary[..., 1] = torch.where(mask_bc, 1.0 - w_bc, bary[..., 1])
        bary[..., 2] = torch.where(mask_bc, w_bc, bary[..., 2])
        interior = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)
        denominator = (va + vb + vc).clamp_min(tiny)
        bary[..., 0] = torch.where(interior, 1.0 - (vb + vc) / denominator, bary[..., 0])
        bary[..., 1] = torch.where(interior, vb / denominator, bary[..., 1])
        bary[..., 2] = torch.where(interior, vc / denominator, bary[..., 2])
        closest = bary[..., 0, None] * a + bary[..., 1, None] * b + bary[..., 2, None] * c
        distance2 = (point - closest).square().sum(-1)
        best_distance2 = torch.minimum(best_distance2, distance2.amin(dim=1))
    result = best_distance2.clamp_min(0.0).sqrt()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("GROUPED_V1_SURFACE_DISTANCE_NONFINITE")
    return result


def reference_scope_weight(
    reference_fingertip_surface_distance_m: torch.Tensor, scope_m: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return minimum reference hand distance and monotonic RSE weight."""

    if (
        reference_fingertip_surface_distance_m.ndim != 2
        or reference_fingertip_surface_distance_m.shape[-1] != 5
    ):
        raise ValueError("GROUPED_V1_REFERENCE_DISTANCE_MUST_BE_[N,5]")
    if scope_m <= 0.0 or not bool(torch.isfinite(reference_fingertip_surface_distance_m).all()):
        raise ValueError("GROUPED_V1_REFERENCE_DISTANCE_INVALID")
    distance = reference_fingertip_surface_distance_m.amin(dim=-1)
    return distance, torch.clamp(distance / float(scope_m), 0.0, 1.0)


def grouped_multiplicative_reward_v1_terms(
    *,
    reference_fingertip_surface_distance_m: torch.Tensor,
    actual_fingertip_surface_distance_m: torch.Tensor,
    source_contact_mask: torch.Tensor,
    fingertip_object_pair_force_world: torch.Tensor,
    fingertip_object_pair_presence: torch.Tensor,
    object_twist_world: torch.Tensor,
    object_twist_world_ref: torch.Tensor,
    object_axis_points: torch.Tensor,
    object_axis_points_ref: torch.Tensor,
    tracked_links: torch.Tensor,
    tracked_links_ref: torch.Tensor,
    finger_q: torch.Tensor,
    finger_q_ref: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    wrist_position: torch.Tensor,
    wrist_quaternion_wxyz: torch.Tensor,
    wrist_position_ref: torch.Tensor,
    wrist_quaternion_ref_wxyz: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
    second_previous_action: torch.Tensor,
    profile: TopoRetargetReferenceTrackingReward26DV4,
    contract: GroupedMultiplicativeRewardV1 | None = None,
) -> dict[str, torch.Tensor]:
    """Compute semantic group errors, bounded rewards, and their log-space product."""

    frozen = contract or GroupedMultiplicativeRewardV1()
    legacy = ppo26d_reward_v4_strict_per_finger_contact_terms(
        object_twist_world=object_twist_world,
        object_twist_world_ref=object_twist_world_ref,
        source_contact_mask=source_contact_mask,
        fingertip_object_pair_force_world=fingertip_object_pair_force_world,
        fingertip_object_pair_presence=fingertip_object_pair_presence,
        object_axis_points=object_axis_points,
        object_axis_points_ref=object_axis_points_ref,
        tracked_links=tracked_links,
        tracked_links_ref=tracked_links_ref,
        finger_q=finger_q,
        finger_q_ref=finger_q_ref,
        joint_lower=joint_lower,
        joint_upper=joint_upper,
        wrist_position=wrist_position,
        wrist_quaternion_wxyz=wrist_quaternion_wxyz,
        wrist_position_ref=wrist_position_ref,
        wrist_quaternion_ref_wxyz=wrist_quaternion_ref_wxyz,
        action=action,
        previous_action=previous_action,
        second_previous_action=second_previous_action,
        profile=profile,
    )
    base_profile: Stage16WorldWristRewardProfileV1 = profile.profile()
    axis_error = torch.linalg.vector_norm(object_axis_points - object_axis_points_ref, dim=-1).mean(
        -1
    )
    linear_error = torch.linalg.vector_norm(
        object_twist_world[:, :3] - object_twist_world_ref[:, :3], dim=-1
    )
    angular_error = torch.linalg.vector_norm(
        object_twist_world[:, 3:] - object_twist_world_ref[:, 3:], dim=-1
    )
    object_costs = torch.stack(
        (
            (axis_error / profile.object_sigma_m).square(),
            (linear_error / profile.object_velocity_sigma_mps).square(),
            (angular_error / profile.object_angular_velocity_sigma_radps).square(),
        ),
        dim=-1,
    )
    object_weights = object_costs.new_tensor(
        (
            profile.object_weight,
            profile.object_velocity_weight,
            profile.object_angular_velocity_weight,
        )
    )
    e_object = (object_costs * object_weights).sum(-1) / object_weights.sum()

    link_error = torch.linalg.vector_norm(tracked_links - tracked_links_ref, dim=-1)
    normalized_finger_error = (finger_q - finger_q_ref) / (joint_upper - joint_lower)
    wrist_position_error = torch.linalg.vector_norm(wrist_position - wrist_position_ref, dim=-1)
    quaternion_dot = (
        torch.sum(wrist_quaternion_wxyz * wrist_quaternion_ref_wxyz, dim=-1).abs().clamp(0.0, 1.0)
    )
    wrist_rotation_error = 2.0 * torch.acos(quaternion_dot)
    hand_costs = torch.stack(
        (
            (link_error / profile.link_sigma_m).square().mean(-1),
            (normalized_finger_error / profile.finger_sigma_normalized).square().mean(-1),
            (wrist_position_error / profile.wrist_position_sigma_m).square(),
            (wrist_rotation_error / profile.wrist_rotation_sigma_rad).square(),
        ),
        dim=-1,
    )
    hand_weights = hand_costs.new_tensor(
        (
            profile.link_weight,
            profile.finger_weight,
            profile.wrist_position_weight,
            profile.wrist_rotation_weight,
        )
    )
    e_hand = (hand_costs * hand_weights).sum(-1) / hand_weights.sum()
    d_ref, w_scope = reference_scope_weight(
        reference_fingertip_surface_distance_m, frozen.distance_scope_m
    )

    mask = source_contact_mask.to(torch.bool)
    expected_count = mask.sum(-1)
    proximity_excess = torch.relu(
        actual_fingertip_surface_distance_m - frozen.proximity_tolerance_m
    )
    mean_excess = torch.where(
        expected_count > 0,
        (proximity_excess * mask.to(proximity_excess.dtype)).sum(-1)
        / expected_count.clamp_min(1).to(proximity_excess.dtype),
        torch.zeros_like(proximity_excess[:, 0]),
    )
    r_proximity = torch.where(
        expected_count > 0,
        torch.exp(-frozen.proximity_scale_per_m * mean_excess),
        torch.ones_like(mean_excess),
    )
    r_contact = torch.where(
        expected_count > 0,
        legacy["r_contact_v4"],
        torch.ones_like(legacy["r_contact_v4"]),
    )
    r_interaction = 0.5 * (r_contact + r_proximity)
    e_regularization = abs(base_profile.smoothness_weight) * legacy["smoothness"]

    groups = {
        "R_obj": torch.exp(-e_object),
        "R_hand": torch.exp(-w_scope * e_hand),
        "R_int": r_interaction,
        "R_reg": torch.exp(-e_regularization),
    }
    clamped = {name: value.clamp(min=frozen.epsilon, max=1.0) for name, value in groups.items()}
    log_total = (
        frozen.object_exponent * torch.log(clamped["R_obj"])
        + frozen.hand_exponent * torch.log(clamped["R_hand"])
        + frozen.interaction_exponent * torch.log(clamped["R_int"])
        + frozen.regularization_exponent * torch.log(clamped["R_reg"])
    )
    total = torch.exp(log_total)
    result = {
        **legacy,
        **clamped,
        "total_legacy_additive": legacy["total"],
        "total": total,
        "log_R_total": log_total,
        "E_obj": e_object,
        "E_hand": e_hand,
        "E_reg": e_regularization,
        "D_ref": d_ref,
        "w_scope": w_scope,
        "actual_fingertip_surface_distance_m": actual_fingertip_surface_distance_m,
        "proximity_excess_m": proximity_excess,
        "R_contact": r_contact.clamp(min=frozen.epsilon, max=1.0),
        "R_prox": r_proximity.clamp(min=frozen.epsilon, max=1.0),
    }
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("GROUPED_V1_REWARD_NONFINITE")
    return result


__all__ = [
    "GROUPED_MULTIPLICATIVE_V1",
    "LEGACY_ADDITIVE",
    "REWARD_MODES",
    "GroupedMultiplicativeRewardV1",
    "grouped_multiplicative_reward_v1_terms",
    "parse_obj_triangles",
    "point_to_triangle_surface_distance",
    "reference_scope_weight",
]
