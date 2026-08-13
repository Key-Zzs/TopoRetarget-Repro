"""Adaptive no-padding horizon contraction for Stage 16-C.5B."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveHorizonSelectorV1:
    horizons: tuple[int, ...] = (1, 5, 10)

    def __post_init__(self) -> None:
        if self.horizons != (1, 5, 10):
            raise ValueError("Stage16-C.5B freezes horizons at [1, 5, 10]")

    def select(self, remaining_steps: int) -> tuple[int, ...]:
        if remaining_steps < 0:
            raise ValueError("remaining steps cannot be negative")
        if remaining_steps >= 10:
            return (1, 5, 10)
        if remaining_steps >= 5:
            return (1, 5)
        if remaining_steps > 1:
            return (1,)
        if remaining_steps == 1:
            return (1,)
        return ()

    def as_dict(self) -> dict[str, object]:
        return {
            "version": "AdaptiveHorizonSelectorV1",
            "horizons": list(self.horizons),
            "rules": {
                "remaining>=10": [1, 5, 10],
                "5<=remaining<10": [1, 5],
                "1<=remaining<5": [1],
                "remaining==0": [],
            },
            "padding": False,
        }


__all__ = ["AdaptiveHorizonSelectorV1"]
