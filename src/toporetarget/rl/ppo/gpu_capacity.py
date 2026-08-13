"""Measured-GPU capacity selection for Stage 16-D.5 PPO-26D."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GpuCapacityMeasurement:
    num_envs: int
    samples_per_s: float | None
    total_vram_mib: float
    peak_vram_mib: float | None
    free_vram_mib: float | None
    ppo_update_ok: bool
    clean_exit: bool
    oom: bool = False
    cuda_allocation_failure: bool = False
    nan_or_inf: bool = False
    physx_fatal_warning: bool = False
    contact_buffer_overflow: bool = False

    @property
    def required_headroom_mib(self) -> float:
        return max(2048.0, self.total_vram_mib * 0.15)

    @property
    def eligible(self) -> bool:
        return (
            not self.oom
            and not self.cuda_allocation_failure
            and not self.nan_or_inf
            and not self.physx_fatal_warning
            and not self.contact_buffer_overflow
            and self.ppo_update_ok
            and self.clean_exit
            and self.free_vram_mib is not None
            and self.free_vram_mib >= self.required_headroom_mib
            and self.samples_per_s is not None
            and self.samples_per_s > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "required_headroom_mib": self.required_headroom_mib,
            "eligible": self.eligible,
        }


def select_ppo26d_environment_capacity(
    measurements: list[GpuCapacityMeasurement],
) -> dict[str, Any]:
    eligible = [measurement for measurement in measurements if measurement.eligible]
    if not eligible:
        raise RuntimeError("ALL_GPU_CAPACITY_CANDIDATES_FAIL")
    eligible_with_metrics: list[tuple[GpuCapacityMeasurement, float, float]] = []
    for measurement in eligible:
        if measurement.samples_per_s is None or measurement.free_vram_mib is None:
            raise AssertionError("eligible capacity row is missing a measured metric")
        eligible_with_metrics.append(
            (measurement, measurement.samples_per_s, measurement.free_vram_mib)
        )
    peak = max(samples_per_s for _, samples_per_s, _ in eligible_with_metrics)
    near_peak = [item for item in eligible_with_metrics if item[1] >= 0.95 * peak]
    selected = sorted(
        near_peak,
        key=lambda item: (-item[2], item[0].num_envs),
    )[0][0]
    return {
        "selector": "Stage16DPPOEnvCapacitySelectorV1",
        "selected_num_envs": selected.num_envs,
        "peak_samples_per_s": peak,
        "selection_reason": (
            "eligible, within 95 percent of measured peak throughput, then maximum VRAM "
            "headroom and smaller env count tie-break"
        ),
        "selected": selected.as_dict(),
        "measurements": [measurement.as_dict() for measurement in measurements],
    }


__all__ = ["GpuCapacityMeasurement", "select_ppo26d_environment_capacity"]
