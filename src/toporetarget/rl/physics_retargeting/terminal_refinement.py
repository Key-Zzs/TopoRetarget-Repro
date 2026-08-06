"""Bounded terminal-tail action parameterization for Stage 16-D recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TerminalTailRefinementConfigV1:
    frame_count: int = 321
    terminal_window_control_steps: int = 20
    transition_window_control_steps: int = 8
    knot_count: int = 8
    population: int = 96
    replicas: int = 4
    iterations: int = 8
    elites: int = 12
    initial_std: float = 0.25
    minimum_std: float = 0.03
    seed: int = 20260806

    def __post_init__(self) -> None:
        signature = (
            self.frame_count,
            self.terminal_window_control_steps,
            self.knot_count,
            self.population,
            self.replicas,
            self.iterations,
            self.elites,
        )
        if signature != (321, 20, 8, 96, 4, 8, 12):
            raise ValueError("terminal refinement violates the frozen 8/96/4/8/12 budget")
        if self.transition_window_control_steps != 8:
            raise ValueError("terminal transition window is frozen at 8 control steps")
        if self.initial_std != 0.25 or self.minimum_std != 0.03:
            raise ValueError("terminal refinement std schedule is frozen")

    @property
    def terminal_start(self) -> int:
        return self.frame_count - self.terminal_window_control_steps

    @property
    def tail_start(self) -> int:
        return self.terminal_start - self.transition_window_control_steps

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "TerminalTailRefinementConfigV1",
            **asdict(self),
            "terminal_start": self.terminal_start,
            "tail_start": self.tail_start,
        }


def materialize_terminal_tail(
    baseline_actions: torch.Tensor,
    knots: torch.Tensor,
    config: TerminalTailRefinementConfigV1 = TerminalTailRefinementConfigV1(),
) -> torch.Tensor:
    """Interpolate a bounded tail while keeping the prefix byte-equivalent."""

    if baseline_actions.shape != (config.frame_count, 26):
        raise ValueError("terminal refinement baseline must be [321,26]")
    if knots.shape[-2:] != (config.knot_count, 26):
        raise ValueError("terminal refinement knots must end in [8,26]")
    if not bool(torch.isfinite(baseline_actions).all()) or not bool(torch.isfinite(knots).all()):
        raise ValueError("terminal refinement inputs must be finite")
    bounded = knots.clamp(-1.0, 1.0).clone()
    bounded[..., 0, :] = baseline_actions[config.tail_start]
    tail_frames = config.frame_count - config.tail_start
    coordinate = torch.linspace(
        0.0,
        config.knot_count - 1,
        tail_frames,
        dtype=bounded.dtype,
        device=bounded.device,
    )
    left = torch.floor(coordinate).long()
    right = (left + 1).clamp_max(config.knot_count - 1)
    alpha = coordinate - left.to(coordinate.dtype)
    prefix_shape = (1,) * (bounded.ndim - 2)
    weight = alpha.reshape(prefix_shape + (tail_frames, 1))
    tail = (1.0 - weight) * bounded[..., left, :] + weight * bounded[..., right, :]
    prefix = baseline_actions[: config.tail_start]
    prefix = prefix.reshape((1,) * (bounded.ndim - 2) + prefix.shape)
    prefix = prefix.expand(bounded.shape[:-2] + prefix.shape[-2:])
    return torch.cat((prefix, tail.clamp(-1.0, 1.0)), dim=-2)


__all__ = ["TerminalTailRefinementConfigV1", "materialize_terminal_tail"]
