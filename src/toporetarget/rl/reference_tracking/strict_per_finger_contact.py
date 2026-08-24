"""Isaac-free Strict Per-Finger V4 source-contact reward contract.

Unlike Reward V3, V4 never aggregates force across fingertips.  A source
requirement for one finger can receive credit only from that same named distal
body's filtered active-object PhysX pair force.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .reference_gated_contact import CONTACT_FINGER_ORDER, EVALUATION_FINGERTIP_LINKS

SOURCE_CONTACT_CONFIRMED = "SOURCE_CONTACT_CONFIRMED"
SOURCE_CONTACT_PERSISTENT = "SOURCE_CONTACT_PERSISTENT"
SOURCE_CONTACT_REQUIRED_CLASSES = (
    SOURCE_CONTACT_CONFIRMED,
    SOURCE_CONTACT_PERSISTENT,
)


@dataclass(frozen=True)
class StrictPerFingerContactRewardV4:
    """Frozen V4 contact-only contract; all non-contact terms remain V2."""

    identifier: str = "StrictPerFingerContactRewardV4"
    source_contact_semantics_identifier: str = "SourcePerFingerContactEvidenceV1"
    finger_order: tuple[str, ...] = CONTACT_FINGER_ORDER
    fingertip_links: tuple[str, ...] = EVALUATION_FINGERTIP_LINKS
    source_required_classes: tuple[str, ...] = SOURCE_CONTACT_REQUIRED_CLASSES
    force_source: str = "active_object_filtered_named_distal_pair_force_world"
    contact_weight: float = 1.0
    epsilon_n: float = 1.0e-5
    numerical_floor_n: float = 1.0e-4
    expected_runtime_frames: int = 321
    aggregation: str = "mean_over_source_required_fingers_only"
    pair_contact_required: bool = True

    def __post_init__(self) -> None:
        if (
            self.finger_order != CONTACT_FINGER_ORDER
            or self.fingertip_links != EVALUATION_FINGERTIP_LINKS
        ):
            raise ValueError("STRICT_V4_FINGERTIP_MAPPING_DRIFT")
        if self.source_contact_semantics_identifier != "SourcePerFingerContactEvidenceV1":
            raise ValueError("STRICT_V4_SOURCE_SEMANTICS_DRIFT")
        if self.source_required_classes != SOURCE_CONTACT_REQUIRED_CLASSES:
            raise ValueError("STRICT_V4_SOURCE_CLASS_POLICY_DRIFT")
        if self.contact_weight != 1.0 or self.epsilon_n != 1.0e-5:
            raise ValueError("STRICT_V4_WEIGHT_OR_EPSILON_DRIFT")
        if self.numerical_floor_n <= 0.0:
            raise ValueError("STRICT_V4_NUMERICAL_FLOOR_INVALID")
        if self.expected_runtime_frames < 17 or (self.expected_runtime_frames - 1) % 8 != 0:
            raise ValueError("STRICT_V4_RUNTIME_FRAME_DOMAIN_INVALID")
        if self.aggregation != "mean_over_source_required_fingers_only":
            raise ValueError("STRICT_V4_NORMALIZATION_DRIFT")
        if not self.pair_contact_required:
            raise ValueError("STRICT_V4_PAIR_PRESENCE_REQUIRED")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def strict_source_contact_mask(class_label: np.ndarray) -> np.ndarray:
    """Make the immutable V4 source mask from a source-audit label array."""

    labels = np.asarray(class_label)
    if (
        labels.ndim != 2
        or labels.shape[1] != 5
        or labels.shape[0] < 17
        or (labels.shape[0] - 1) % 8 != 0
    ):
        raise ValueError(f"STRICT_V4_SOURCE_CLASS_SHAPE_INVALID:{labels.shape}")
    if labels.dtype.kind not in {"U", "S", "O"}:
        raise ValueError("STRICT_V4_SOURCE_CLASS_DTYPE_INVALID")
    result = np.isin(labels.astype(str), SOURCE_CONTACT_REQUIRED_CLASSES)
    if result.shape != labels.shape:  # Defensive invariant for future NumPy changes.
        raise AssertionError("STRICT_V4_SOURCE_MASK_SHAPE_INVALID")
    return result


def strict_per_finger_contact_reward(
    *,
    source_contact_mask: torch.Tensor,
    fingertip_object_pair_force_world: torch.Tensor,
    lambda_tip_n: float,
    pair_presence: torch.Tensor | None = None,
    contract: StrictPerFingerContactRewardV4 | None = None,
) -> dict[str, torch.Tensor]:
    """Return V4's normalized independent-finger contact reward.

    ``pair_presence`` is deliberately a separate gate: a nonzero numerical
    force alone must not manufacture reward from a sensor-noise sample.
    """

    frozen = contract or StrictPerFingerContactRewardV4()
    if not np.isfinite(lambda_tip_n) or lambda_tip_n <= frozen.epsilon_n:
        raise ValueError("STRICT_V4_LAMBDA_TIP_INVALID")
    if source_contact_mask.ndim != 2 or source_contact_mask.shape[-1] != 5:
        raise ValueError("STRICT_V4_SOURCE_MASK_MUST_BE_[N,5]")
    expected_force_shape = (*source_contact_mask.shape, 3)
    if tuple(fingertip_object_pair_force_world.shape) != expected_force_shape:
        raise ValueError("STRICT_V4_PAIR_FORCE_MUST_MATCH_SOURCE_MASK_[N,5,3]")
    if not bool(torch.isfinite(fingertip_object_pair_force_world).all()):
        raise FloatingPointError("STRICT_V4_PAIR_FORCE_NONFINITE")
    if pair_presence is None:
        presence = (
            torch.linalg.vector_norm(fingertip_object_pair_force_world, dim=-1)
            > frozen.numerical_floor_n
        )
    else:
        if tuple(pair_presence.shape) != tuple(source_contact_mask.shape):
            raise ValueError("STRICT_V4_PAIR_PRESENCE_MUST_MATCH_SOURCE_MASK_[N,5]")
        presence = pair_presence.to(dtype=torch.bool)
    mask = source_contact_mask.to(dtype=torch.bool)
    magnitudes = torch.linalg.vector_norm(fingertip_object_pair_force_world, dim=-1)
    valid_force = presence & (magnitudes > frozen.numerical_floor_n)
    raw_per_finger = torch.exp(-float(lambda_tip_n) / (magnitudes + frozen.epsilon_n))
    per_finger = torch.where(valid_force, raw_per_finger, torch.zeros_like(magnitudes))
    required_per_finger = per_finger * mask.to(dtype=magnitudes.dtype)
    expected_count = mask.sum(dim=-1)
    contact_reward = torch.where(
        expected_count > 0,
        frozen.contact_weight
        * required_per_finger.sum(dim=-1)
        / expected_count.to(magnitudes.dtype),
        torch.zeros_like(expected_count, dtype=magnitudes.dtype),
    )
    if not bool(torch.isfinite(contact_reward).all()):
        raise FloatingPointError("STRICT_V4_CONTACT_REWARD_NONFINITE")
    if bool((contact_reward < 0.0).any()) or bool(
        (contact_reward > frozen.contact_weight + 1.0e-6).any()
    ):
        raise FloatingPointError("STRICT_V4_CONTACT_REWARD_OUT_OF_BOUNDS")
    return {
        "r_contact_v4": contact_reward,
        "source_contact_mask": mask,
        "tip_pair_presence": presence,
        "tip_pair_force_norm_n": magnitudes,
        "per_finger_contact_reward": required_per_finger,
        "source_expected_finger_count": expected_count,
        "source_satisfied_tip_count": (mask & valid_force).sum(dim=-1),
        "source_tip_coverage_ratio": torch.where(
            expected_count > 0,
            (mask & valid_force).sum(dim=-1).to(magnitudes.dtype)
            / expected_count.to(magnitudes.dtype),
            torch.zeros_like(expected_count, dtype=magnitudes.dtype),
        ),
    }


__all__ = [
    "SOURCE_CONTACT_CONFIRMED",
    "SOURCE_CONTACT_PERSISTENT",
    "SOURCE_CONTACT_REQUIRED_CLASSES",
    "StrictPerFingerContactRewardV4",
    "strict_per_finger_contact_reward",
    "strict_source_contact_mask",
]
