"""Reference-scoped exploration contracts compatible with uniform Stage16 RSI."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ReferenceScopedExplorationV1:
    identifier: str = "Stage16ReferenceScopedExplorationV1"
    enabled: bool = True
    distance_relaxation: bool = True
    adaptive_termination: bool = True
    distance_scope_m: float = 0.20
    kappa_min: float = 0.50
    initial_fail_count: int = 1
    initial_total_count: int = 1
    object_position_base_m: float = 0.05
    object_axis_base_m: float = 0.05
    object_orientation_base_rad: float = 0.7853981633974483
    hand_position_base_m: float = 0.20
    hand_orientation_base_rad: float = 1.5707963267948966
    interaction_excess_base_normalized: float = 1.0
    proximity_tolerance_m: float = 0.03

    def __post_init__(self) -> None:
        if self.distance_scope_m != 0.20 or self.kappa_min != 0.50:
            raise ValueError("RSE_V1_GLOBAL_SCOPE_DRIFT")
        if self.initial_fail_count != 1 or self.initial_total_count != 1:
            raise ValueError("RSE_V1_INITIAL_COUNTS_DRIFT")
        if not 0.0 < self.kappa_min <= 1.0:
            raise ValueError("RSE_V1_KAPPA_MIN_INVALID")
        if any(
            value <= 0.0
            for value in (
                self.object_position_base_m,
                self.object_axis_base_m,
                self.object_orientation_base_rad,
                self.hand_position_base_m,
                self.hand_orientation_base_rad,
                self.interaction_excess_base_normalized,
                self.proximity_tolerance_m,
            )
        ):
            raise ValueError("RSE_V1_BASE_THRESHOLD_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AdaptiveScopeStateV1:
    fail_count: int = 1
    total_count: int = 1
    kappa_min: float = 0.50

    @property
    def kappa(self) -> float:
        return min(1.0, max(self.kappa_min, self.fail_count / self.total_count))

    def record(self, *, rse_failures: int, normal_completions: int) -> None:
        if rse_failures < 0 or normal_completions < 0:
            raise ValueError("RSE_V1_COUNTER_INCREMENT_INVALID")
        self.fail_count += int(rse_failures)
        self.total_count += int(rse_failures + normal_completions)
        if self.fail_count > self.total_count:
            raise RuntimeError("RSE_V1_COUNTER_STATE_INVALID")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "fail_count": self.fail_count,
            "total_count": self.total_count,
            "kappa_min": self.kappa_min,
            "kappa": self.kappa,
        }


def adaptive_kappa(
    fail_count: int | torch.Tensor,
    total_count: int | torch.Tensor,
    *,
    kappa_min: float = 0.50,
) -> torch.Tensor:
    fail = torch.as_tensor(fail_count, dtype=torch.float64)
    total = torch.as_tensor(total_count, dtype=torch.float64)
    if bool((fail < 0).any()) or bool((total <= 0).any()) or bool((fail > total).any()):
        raise ValueError("RSE_V1_COUNTERS_INVALID")
    return torch.clamp(fail / total, min=kappa_min, max=1.0)


def rse_deviation_termination(
    *,
    object_position_error_m: torch.Tensor,
    object_axis_error_m: torch.Tensor,
    object_orientation_error_rad: torch.Tensor,
    hand_position_error_m: torch.Tensor,
    hand_orientation_error_rad: torch.Tensor,
    actual_fingertip_surface_distance_m: torch.Tensor,
    source_contact_mask: torch.Tensor,
    kappa: float,
    contract: ReferenceScopedExplorationV1 | None = None,
) -> dict[str, torch.Tensor]:
    """Return dimensionless group deviations and the adaptive fail mask."""

    frozen = contract or ReferenceScopedExplorationV1()
    if not frozen.kappa_min <= kappa <= 1.0:
        raise ValueError("RSE_V1_KAPPA_OUT_OF_RANGE")
    mask = source_contact_mask.to(torch.bool)
    expected_count = mask.sum(-1)
    interaction_excess = torch.relu(
        actual_fingertip_surface_distance_m / frozen.proximity_tolerance_m - 1.0
    )
    interaction_deviation = torch.where(
        expected_count > 0,
        (interaction_excess * mask.to(interaction_excess.dtype)).sum(-1)
        / expected_count.clamp_min(1).to(interaction_excess.dtype),
        torch.zeros_like(interaction_excess[:, 0]),
    )
    object_deviation = torch.stack(
        (
            object_position_error_m / frozen.object_position_base_m,
            object_axis_error_m / frozen.object_axis_base_m,
            object_orientation_error_rad / frozen.object_orientation_base_rad,
        ),
        dim=-1,
    ).amax(-1)
    hand_deviation = torch.stack(
        (
            hand_position_error_m / frozen.hand_position_base_m,
            hand_orientation_error_rad / frozen.hand_orientation_base_rad,
        ),
        dim=-1,
    ).amax(-1)
    interaction_deviation = interaction_deviation / frozen.interaction_excess_base_normalized
    threshold = torch.full_like(object_deviation, float(kappa))
    failure = (
        (object_deviation > threshold)
        | (hand_deviation > threshold)
        | ((expected_count > 0) & (interaction_deviation > threshold))
    )
    result = {
        "object_deviation_normalized": object_deviation,
        "hand_deviation_normalized": hand_deviation,
        "interaction_deviation_normalized": interaction_deviation,
        "adaptive_threshold_normalized": threshold,
        "rse_deviation_failure": failure,
    }
    if not all(
        bool(torch.isfinite(value).all()) for value in result.values() if value.dtype != torch.bool
    ):
        raise FloatingPointError("RSE_V1_DEVIATION_NONFINITE")
    return result


__all__ = [
    "AdaptiveScopeStateV1",
    "ReferenceScopedExplorationV1",
    "adaptive_kappa",
    "rse_deviation_termination",
]
