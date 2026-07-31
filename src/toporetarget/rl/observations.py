"""Frozen Table-4 observation ordering, lookahead, noise, and delay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .contracts import Stage16ReferenceClip

PAPER_OBSERVATION_ID = "paper_reference_observation_v1"
PAPER_LOOKAHEAD_OFFSETS = (0, 1, 3, 5)


@dataclass(frozen=True)
class ObservationContract:
    dof_count: int
    link_count: int
    lookahead_offsets: tuple[int, ...] = PAPER_LOOKAHEAD_OFFSETS

    @property
    def dimension(self) -> int:
        proprio = self.dof_count * 3
        current_object = 6 * 3
        per_reference = self.dof_count + 6 * 3 + self.link_count * 3
        return proprio + current_object + len(self.lookahead_offsets) * per_reference

    def as_dict(self) -> dict[str, object]:
        return {
            "id": PAPER_OBSERVATION_ID,
            "order": [
                "q",
                "qdot",
                "previous_action",
                "current_object_axis_points",
                "reference[offset]:q_finger",
                "reference[offset]:object_axis_points",
                "reference[offset]:tracked_link_positions",
            ],
            "lookahead_offsets": list(self.lookahead_offsets),
            "dimension": self.dimension,
            "dof_count": self.dof_count,
            "link_count": self.link_count,
        }


class ObservationDelayBuffer:
    """Latest-only bounded delay; references are deliberately never delayed."""

    def __init__(self, delay_steps: int) -> None:
        if delay_steps < 0 or delay_steps > 2:
            raise ValueError("paper observation delay must be in [0,2]")
        self.delay_steps = delay_steps
        self._values: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=delay_steps + 1
        )

    def push(
        self, q: np.ndarray, qdot: np.ndarray, axes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current = (np.asarray(q).copy(), np.asarray(qdot).copy(), np.asarray(axes).copy())
        self._values.append(current)
        return self._values[0] if len(self._values) > self.delay_steps else current


def build_observation(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    previous_action: np.ndarray,
    current_object_axis_points: np.ndarray,
    reference: Stage16ReferenceClip,
    reference_index: int,
    contract: ObservationContract | None = None,
) -> np.ndarray:
    """Build exactly the documented observation without leaking hidden simulation state."""

    spec = contract or ObservationContract(reference.dof_count, len(reference.tracked_link_names))
    if spec.dof_count != reference.dof_count or spec.link_count != len(
        reference.tracked_link_names
    ):
        raise ValueError("observation contract does not match reference clip")
    proprio = [
        np.asarray(q).reshape(-1),
        np.asarray(qdot).reshape(-1),
        np.asarray(previous_action).reshape(-1),
    ]
    axes = np.asarray(current_object_axis_points, dtype=np.float64)
    if axes.shape != (6, 3):
        raise ValueError("current object axis points must have shape [6,3]")
    chunks = [*proprio, axes.reshape(-1)]
    for offset in spec.lookahead_offsets:
        index = min(max(reference_index + offset, 0), reference.frame_count - 1)
        chunks.extend(
            [
                reference.q_finger_ref[index],
                reference.object_axis_points_base_ref[index].reshape(-1),
                reference.tracked_link_positions_base_ref[index].reshape(-1),
            ]
        )
    result = np.concatenate(chunks).astype(np.float32, copy=False)
    if result.shape != (spec.dimension,) or not np.isfinite(result).all():
        raise ValueError("observation violates dimension or finite-value contract")
    return result


__all__ = [
    "ObservationContract",
    "ObservationDelayBuffer",
    "PAPER_LOOKAHEAD_OFFSETS",
    "PAPER_OBSERVATION_ID",
    "build_observation",
]
