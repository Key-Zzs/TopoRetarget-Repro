"""Offline contracts for sequence-aware PPO generalization audits.

The helpers in this module are deliberately Isaac-free.  They define global,
metadata-derived budget and RSI semantics without authorizing training or
changing the frozen reward, RSE, PF, or DF implementations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final

PHASES: Final[tuple[str, ...]] = (
    "PRE/IDLE",
    "APPROACH",
    "CONTACT",
    "PICKUP",
    "TRANSPORT",
    "PLACE",
    "RELEASE",
    "RETREAT",
)
INTERACTION_PHASES: Final[frozenset[str]] = frozenset(
    {"CONTACT", "PICKUP", "TRANSPORT", "PLACE", "RELEASE"}
)


@dataclass(frozen=True)
class EpisodeV1RuntimeEvents:
    """Complete EpisodeV1 event indices in a runtime reference domain."""

    reference_length: int
    approach: int
    contact: int
    pickup: int
    place: int
    release: int
    retreat: int

    def __post_init__(self) -> None:
        ordered = (
            0,
            self.approach,
            self.contact,
            self.pickup,
            self.place,
            self.release,
            self.retreat,
            self.reference_length - 1,
        )
        if self.reference_length <= 0 or any(left > right for left, right in pairwise(ordered)):
            raise ValueError("PPO_GENERALIZATION_EPISODE_EVENTS_INVALID")

    def phase_labels(self) -> tuple[str, ...]:
        """Return one non-overlapping lifecycle label per reference index."""

        labels = ["PRE/IDLE"] * self.reference_length
        labels[self.approach : self.contact] = ["APPROACH"] * (self.contact - self.approach)
        labels[self.contact : self.pickup] = ["CONTACT"] * (self.pickup - self.contact)
        labels[self.pickup] = "PICKUP"
        labels[self.pickup + 1 : self.place] = ["TRANSPORT"] * (self.place - self.pickup - 1)
        labels[self.place : self.release] = ["PLACE"] * (self.release - self.place)
        labels[self.release : self.retreat] = ["RELEASE"] * (self.retreat - self.release)
        labels[self.retreat :] = ["RETREAT"] * (self.reference_length - self.retreat)
        return tuple(labels)


def map_source_event_to_runtime(
    event_frame: int,
    *,
    source_start_frame: int,
    source_end_frame: int,
    runtime_reference_length: int,
) -> int:
    """Map an inclusive source-frame event into an inclusive runtime domain."""

    source_span = source_end_frame - source_start_frame
    if source_span <= 0 or runtime_reference_length <= 1:
        raise ValueError("PPO_GENERALIZATION_EVENT_DOMAIN_INVALID")
    if not source_start_frame <= event_frame <= source_end_frame:
        raise ValueError("PPO_GENERALIZATION_EVENT_OUTSIDE_SOURCE_DOMAIN")
    scale = (runtime_reference_length - 1) / source_span
    return int(round((event_frame - source_start_frame) * scale))


@dataclass(frozen=True)
class CappedSequenceBudgetV1:
    """A deterministic candidate budget with a mandatory global sample cap."""

    samples_per_valid_index: float
    samples_per_interaction_index: float
    global_sample_cap: int
    samples_per_update: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.samples_per_valid_index)
            or self.samples_per_valid_index <= 0.0
            or not math.isfinite(self.samples_per_interaction_index)
            or self.samples_per_interaction_index <= 0.0
            or self.global_sample_cap <= 0
            or self.samples_per_update <= 0
        ):
            raise ValueError("PPO_GENERALIZATION_BUDGET_INVALID")

    def derive(self, *, valid_indices: int, interaction_indices: int) -> dict[str, int]:
        if valid_indices <= 0 or not 0 < interaction_indices <= valid_indices:
            raise ValueError("PPO_GENERALIZATION_REFERENCE_COUNTS_INVALID")
        raw = max(
            valid_indices * self.samples_per_valid_index,
            interaction_indices * self.samples_per_interaction_index,
        )
        updates = max(1, math.ceil(raw / self.samples_per_update))
        capped_updates = min(updates, self.global_sample_cap // self.samples_per_update)
        if capped_updates <= 0:
            raise ValueError("PPO_GENERALIZATION_CAP_BELOW_ONE_UPDATE")
        return {
            "updates": capped_updates,
            "total_samples": capped_updates * self.samples_per_update,
            "capped": int(capped_updates < updates),
        }

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class UniformEventBalancedRSIV1:
    """Global RSI mixture retaining a nonzero uniform valid-domain component."""

    uniform_alpha: float
    interaction_source: str = "EpisodeV1_events_CONTACT_through_RELEASE"

    def __post_init__(self) -> None:
        if not 0.0 < self.uniform_alpha <= 1.0:
            raise ValueError("PPO_GENERALIZATION_UNIFORM_COMPONENT_REQUIRED")
        if self.interaction_source != "EpisodeV1_events_CONTACT_through_RELEASE":
            raise ValueError("PPO_GENERALIZATION_INTERACTION_SOURCE_DRIFT")

    @property
    def event_balanced_alpha(self) -> float:
        return 1.0 - self.uniform_alpha

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "uniform_alpha": self.uniform_alpha,
            "event_balanced_alpha": self.event_balanced_alpha,
            "interaction_source": self.interaction_source,
            "uniform_component_preserved": True,
            "frame0_only": False,
        }


def object_bbox_diagonal_m(path: str | Path) -> float:
    """Compute a characteristic length from finite OBJ vertex bounds."""

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    count = 0
    with Path(path).open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            values = tuple(float(value) for value in line.split()[1:4])
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError("PPO_GENERALIZATION_OBJECT_VERTICES_INVALID")
            for axis, value in enumerate(values):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            count += 1
    if count == 0:
        raise ValueError("PPO_GENERALIZATION_OBJECT_VERTICES_EMPTY")
    diagonal = math.sqrt(
        sum((upper - lower) ** 2 for lower, upper in zip(minimum, maximum, strict=True))
    )
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("PPO_GENERALIZATION_OBJECT_SCALE_INVALID")
    return diagonal


@dataclass(frozen=True)
class DimensionlessObjectScaleV1:
    """Globally anchored object-scale conversion with no bespoke object knobs."""

    anchor_bbox_diagonal_m: float
    proximity_tolerance_at_anchor_m: float = 0.03
    distance_scope_at_anchor_m: float = 0.20
    object_tracking_sigma_at_anchor_m: float = 0.04
    object_velocity_sigma_at_anchor_mps: float = 0.075
    object_position_at_anchor_m: float = 0.05
    object_axis_at_anchor_m: float = 0.05

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("PPO_GENERALIZATION_SCALE_CONTRACT_INVALID")

    def thresholds(self, object_bbox_diagonal: float) -> dict[str, float]:
        if not math.isfinite(object_bbox_diagonal) or object_bbox_diagonal <= 0.0:
            raise ValueError("PPO_GENERALIZATION_OBJECT_SCALE_INVALID")
        ratio = object_bbox_diagonal / self.anchor_bbox_diagonal_m
        return {
            "proximity_tolerance_m": self.proximity_tolerance_at_anchor_m * ratio,
            "distance_scope_m": self.distance_scope_at_anchor_m * ratio,
            "object_tracking_sigma_m": self.object_tracking_sigma_at_anchor_m * ratio,
            "object_velocity_sigma_mps": self.object_velocity_sigma_at_anchor_mps * ratio,
            "object_position_base_m": self.object_position_at_anchor_m * ratio,
            "object_axis_base_m": self.object_axis_at_anchor_m * ratio,
        }

    def dimensionless_ratios(self) -> dict[str, float]:
        return {
            "proximity_tolerance_over_s_o": self.proximity_tolerance_at_anchor_m
            / self.anchor_bbox_diagonal_m,
            "distance_scope_over_s_o": self.distance_scope_at_anchor_m
            / self.anchor_bbox_diagonal_m,
            "object_tracking_sigma_over_s_o": self.object_tracking_sigma_at_anchor_m
            / self.anchor_bbox_diagonal_m,
            "object_velocity_sigma_over_s_o_per_s": self.object_velocity_sigma_at_anchor_mps
            / self.anchor_bbox_diagonal_m,
            "object_position_base_over_s_o": self.object_position_at_anchor_m
            / self.anchor_bbox_diagonal_m,
            "object_axis_base_over_s_o": self.object_axis_at_anchor_m / self.anchor_bbox_diagonal_m,
        }


@dataclass(frozen=True)
class DimensionlessScaledGroupedRewardV1:
    """Grouped Multiplicative Reward V1 with P3's automatic object scale."""

    identifier: str
    distance_scope_m: float
    proximity_tolerance_m: float
    object_bbox_diagonal_m: float
    anchor_bbox_diagonal_m: float
    proximity_scale_per_m: float
    epsilon: float = 1.0e-12
    object_exponent: float = 1.0
    hand_exponent: float = 1.0
    interaction_exponent: float = 1.0
    regularization_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.identifier != "Stage16GroupedMultiplicativeRewardV1+P3ObjectScaleV1":
            raise ValueError("PPO_GENERALIZATION_GROUPED_SCALE_IDENTIFIER_DRIFT")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.distance_scope_m,
                self.proximity_tolerance_m,
                self.object_bbox_diagonal_m,
                self.anchor_bbox_diagonal_m,
                self.proximity_scale_per_m,
            )
        ):
            raise ValueError("PPO_GENERALIZATION_GROUPED_SCALE_INVALID")
        if not math.isclose(
            self.proximity_scale_per_m,
            1.0 / self.proximity_tolerance_m,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("PPO_GENERALIZATION_PROXIMITY_SCALE_DRIFT")
        if any(
            value != 1.0
            for value in (
                self.object_exponent,
                self.hand_exponent,
                self.interaction_exponent,
                self.regularization_exponent,
            )
        ):
            raise ValueError("PPO_GENERALIZATION_GROUPED_STRUCTURE_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DimensionlessScaledReferenceScopedExplorationV1:
    """RSE V1 with only P3-approved object-relative metric scales changed."""

    identifier: str
    enabled: bool
    distance_relaxation: bool
    adaptive_termination: bool
    distance_scope_m: float
    object_position_base_m: float
    object_axis_base_m: float
    proximity_tolerance_m: float
    object_bbox_diagonal_m: float
    anchor_bbox_diagonal_m: float
    kappa_min: float = 0.50
    initial_fail_count: int = 1
    initial_total_count: int = 1
    object_orientation_base_rad: float = 0.7853981633974483
    hand_position_base_m: float = 0.20
    hand_orientation_base_rad: float = 1.5707963267948966
    interaction_excess_base_normalized: float = 1.0

    def __post_init__(self) -> None:
        if self.identifier != "Stage16ReferenceScopedExplorationV1+P3ObjectScaleV1":
            raise ValueError("PPO_GENERALIZATION_RSE_SCALE_IDENTIFIER_DRIFT")
        if self.kappa_min != 0.50 or self.initial_fail_count != 1 or self.initial_total_count != 1:
            raise ValueError("PPO_GENERALIZATION_RSE_COUNTER_SEMANTICS_DRIFT")
        if self.hand_position_base_m != 0.20:
            raise ValueError("PPO_GENERALIZATION_RSE_HAND_SCALE_DRIFT")
        values = (
            self.distance_scope_m,
            self.object_position_base_m,
            self.object_axis_base_m,
            self.proximity_tolerance_m,
            self.object_bbox_diagonal_m,
            self.anchor_bbox_diagonal_m,
            self.object_orientation_base_rad,
            self.hand_orientation_base_rad,
            self.interaction_excess_base_normalized,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("PPO_GENERALIZATION_RSE_SCALE_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "CappedSequenceBudgetV1",
    "DimensionlessScaledGroupedRewardV1",
    "DimensionlessScaledReferenceScopedExplorationV1",
    "DimensionlessObjectScaleV1",
    "EpisodeV1RuntimeEvents",
    "INTERACTION_PHASES",
    "PHASES",
    "UniformEventBalancedRSIV1",
    "map_source_event_to_runtime",
    "object_bbox_diagonal_m",
]
