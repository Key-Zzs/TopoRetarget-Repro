"""Exact control-step action history for deterministic C.5A replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CandidateActionHistoryV1:
    """Actions from frame zero through the current planning boundary.

    Actions are stored as detached clones on the source device.  They are
    control-step actions only; no synthetic substep actions are introduced.
    """

    actions: list[torch.Tensor] = field(default_factory=list)
    start_reference_index: int = 0

    def append(self, action: torch.Tensor) -> None:
        if action.ndim != 2 or action.shape[-1] != 26:
            raise ValueError("CandidateActionHistoryV1 requires [num_envs, 26] actions")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("CandidateActionHistoryV1 rejects non-finite actions")
        self.actions.append(action.detach().clone())

    @property
    def length(self) -> int:
        return len(self.actions)

    def stack(self) -> torch.Tensor:
        if not self.actions:
            return torch.empty((0, 26), dtype=torch.float32)
        if any(value.shape != self.actions[0].shape for value in self.actions):
            raise ValueError("action history has inconsistent action shapes")
        return torch.stack(self.actions, dim=0)

    def validate(self, *, expected_boundary_index: int | None = None) -> None:
        if self.start_reference_index != 0:
            raise ValueError("C.5A history must begin at runtime reference frame zero")
        if expected_boundary_index is not None and self.length != expected_boundary_index:
            raise ValueError(
                "C.5A action history length/index mismatch: "
                f"length={self.length}, boundary={expected_boundary_index}"
            )

    def as_dict(self) -> dict[str, object]:
        tensor = self.stack()
        return {
            "version": "CandidateActionHistoryV1",
            "start_reference_index": self.start_reference_index,
            "length": self.length,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }


__all__ = ["CandidateActionHistoryV1"]
