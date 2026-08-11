"""Pure contracts for the Stage 16-D reference-gated contact reward.

The run-time environment owns the Isaac Lab contact sensor.  This module is
deliberately Isaac-free so the frozen mask, force mapping, and reward formula
can be audited in ordinary CPU tests before any GPU PPO launch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch

CONTACT_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
EVALUATION_FINGERTIP_LINKS = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)


@dataclass(frozen=True)
class ReferenceGatedContactRewardContractV1:
    """The immutable non-tuned parameters for Reward V3."""

    identifier: str = "ReferenceGatedContactRewardV1"
    finger_order: tuple[str, ...] = CONTACT_FINGER_ORDER
    fingertip_links: tuple[str, ...] = EVALUATION_FINGERTIP_LINKS
    distance_source: str = "reference_robot_distal_root_to_visual_object_surface_unsigned"
    force_source: str = "active_object_filtered_pair_force_matrix_world"
    reference_kinematics_version: int = 2
    xi_c_m: float = 0.03
    diagnostic_xi_c_m: float = 0.02
    contact_weight: float = 1.0
    epsilon_n: float = 1.0e-5
    persistent_window_control_steps: int = 3

    def __post_init__(self) -> None:
        if self.finger_order != CONTACT_FINGER_ORDER or len(self.fingertip_links) != 5:
            raise ValueError("CONTACT_REWARD_REQUIRES_FIVE_SHARED_FINGERTIPS")
        if self.reference_kinematics_version != 2:
            raise ValueError("CONTACT_REWARD_REQUIRES_REFERENCE_KINEMATICS_V2")
        if self.xi_c_m != 0.03 or self.diagnostic_xi_c_m != 0.02:
            raise ValueError("CONTACT_REWARD_DISTANCE_THRESHOLDS_ARE_FROZEN")
        if self.contact_weight != 1.0 or self.epsilon_n != 1.0e-5:
            raise ValueError("CONTACT_REWARD_WEIGHT_OR_EPSILON_DRIFT")
        if self.persistent_window_control_steps != 3:
            raise ValueError("CONTACT_REWARD_PERSISTENT_WINDOW_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def fingertip_force_indices(collision_body_names: Sequence[str]) -> tuple[int, ...]:
    """Map the shared five tips to filtered active-object sensor columns."""

    available = {str(name): index for index, name in enumerate(collision_body_names)}
    missing = [name for name in EVALUATION_FINGERTIP_LINKS if name not in available]
    if missing:
        raise ValueError(f"CONTACT_REWARD_FINGERTIP_FORCE_MAPPING_MISSING:{missing}")
    result = tuple(available[name] for name in EVALUATION_FINGERTIP_LINKS)
    if len(set(result)) != 5:
        raise ValueError("CONTACT_REWARD_FINGERTIP_FORCE_MAPPING_NOT_UNIQUE")
    return result


def reference_expected_contact_mask(
    distances_m: np.ndarray | torch.Tensor,
    *,
    contract: ReferenceGatedContactRewardContractV1 | None = None,
) -> np.ndarray | torch.Tensor:
    """Apply the frozen strict `< 3 cm` reference-only mask."""

    frozen = contract or ReferenceGatedContactRewardContractV1()
    if distances_m.shape[-1] != 5:
        raise ValueError("CONTACT_REWARD_DISTANCE_SHAPE_MUST_END_IN_FIVE")
    if isinstance(distances_m, torch.Tensor):
        if not bool(torch.isfinite(distances_m).all()):
            raise ValueError("CONTACT_REWARD_DISTANCE_NONFINITE")
        return distances_m < frozen.xi_c_m
    values = np.asarray(distances_m, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("CONTACT_REWARD_DISTANCE_NONFINITE")
    return values < frozen.xi_c_m


def reference_mask_summary(mask: np.ndarray, *, clip: str) -> dict[str, object]:
    """Summarize an immutable [T,5] reference contact mask."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("CONTACT_REWARD_MASK_MUST_BE_[T,5]")
    any_mask = values.any(axis=1)
    longest = 0
    current = 0
    for active in any_mask:
        current = current + 1 if bool(active) else 0
        longest = max(longest, current)
    onset = np.flatnonzero(any_mask)
    return {
        "clip": clip,
        "total_frames": int(values.shape[0]),
        "expected_contact_frames_per_fingertip": {
            finger: int(values[:, index].sum()) for index, finger in enumerate(CONTACT_FINGER_ORDER)
        },
        "any_finger_expected_contact_frames": int(any_mask.sum()),
        "longest_expected_contact_window_control_steps": int(longest),
        "expected_contact_onset": None if onset.size == 0 else int(onset[0]),
        "expected_contact_final_frame": bool(any_mask[-1]),
        "mask_fraction": float(any_mask.mean()),
    }


def reference_gated_contact_reward(
    *,
    reference_expected_mask: torch.Tensor,
    fingertip_object_pair_force_world: torch.Tensor,
    lambda_c_n: float,
    contract: ReferenceGatedContactRewardContractV1 | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the V3 additive term from exact fingertip-object force pairs.

    ``reference_expected_mask`` is intentionally an input independent of all
    actual contact data.  The explicit ``where`` keeps no-reference steps
    exactly zero rather than relying on numerical underflow.
    """

    frozen = contract or ReferenceGatedContactRewardContractV1()
    if lambda_c_n <= frozen.epsilon_n or not np.isfinite(lambda_c_n):
        raise ValueError("CONTACT_REWARD_LAMBDA_MUST_BE_FINITE_AND_GREATER_THAN_EPSILON")
    if reference_expected_mask.ndim != 2 or reference_expected_mask.shape[-1] != 5:
        raise ValueError("CONTACT_REWARD_REFERENCE_MASK_MUST_BE_[N,5]")
    expected_shape = (*reference_expected_mask.shape, 3)
    if tuple(fingertip_object_pair_force_world.shape) != expected_shape:
        raise ValueError("CONTACT_REWARD_PAIR_FORCE_MUST_MATCH_REFERENCE_MASK_[N,5,3]")
    if not bool(torch.isfinite(fingertip_object_pair_force_world).all()):
        raise FloatingPointError("CONTACT_REWARD_PAIR_FORCE_NONFINITE")
    mask = reference_expected_mask.to(dtype=torch.bool)
    magnitudes = torch.linalg.vector_norm(fingertip_object_pair_force_world, dim=-1)
    scale = (magnitudes * mask.to(magnitudes.dtype)).sum(dim=-1)
    has_expected_contact = mask.any(dim=-1)
    reward_when_expected = frozen.contact_weight * torch.exp(
        -float(lambda_c_n) / (scale + frozen.epsilon_n)
    )
    reward = torch.where(has_expected_contact, reward_when_expected, torch.zeros_like(scale))
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("CONTACT_REWARD_NONFINITE")
    if bool((reward < 0.0).any()) or bool((reward > frozen.contact_weight + 1.0e-6).any()):
        raise FloatingPointError("CONTACT_REWARD_OUT_OF_BOUNDS")
    return {
        "r_contact": reward,
        "contact_force_scale_n": scale,
        "fingertip_object_force_magnitude_n": magnitudes,
        "reference_expected_contact_mask": mask,
        "actual_fingertip_object_contact_mask": magnitudes > 1.0e-4,
    }


def exact_pair_force_trace_status(keys: Sequence[str]) -> dict[str, object]:
    """Fail closed when a historical trace has only aggregate contact force."""

    available = {str(key) for key in keys}
    exact_fields = {
        "fingertip_object_pair_force_world",
        "replica_fingertip_object_pair_force_world",
    }
    matches = sorted(exact_fields & available)
    if matches:
        return {
            "status": "PAIR_FORCE_AVAILABLE",
            "exact_pair_force_fields": matches,
            "aggregate_force_is_not_used": True,
        }
    return {
        "status": "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED",
        "exact_pair_force_fields": [],
        "aggregate_force_is_not_used": True,
        "aggregate_fields_present": sorted(
            {"contact_force_world", "replica_contact_force_world"} & available
        ),
        "reason": (
            "Historical trace lacks per-fingertip active-object force vectors; "
            "aggregate force and pair presence cannot be decomposed safely."
        ),
    }


__all__ = [
    "CONTACT_FINGER_ORDER",
    "EVALUATION_FINGERTIP_LINKS",
    "ReferenceGatedContactRewardContractV1",
    "exact_pair_force_trace_status",
    "fingertip_force_indices",
    "reference_expected_contact_mask",
    "reference_gated_contact_reward",
    "reference_mask_summary",
]
