"""Complete, append-only Stage 16-D rollout trace recorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PhysicsConsistentTrajectoryRecorderV1:
    clip: str
    expected_frames: int = 321
    rows: list[dict[str, np.ndarray]] = field(default_factory=list)

    REQUIRED = (
        "wrist_pose",
        "wrist_twist",
        "virtual_wrist_q",
        "virtual_wrist_qd",
        "finger_q",
        "finger_qd",
        "actions",
        "targets",
        "efforts",
        "object_pose",
        "object_twist",
        "object_axis_points",
        "contact_force",
        "contact_impulse",
        "penetration",
        "semantic_progress",
    )

    def append(self, row: dict[str, Any]) -> None:
        missing = set(self.REQUIRED) - set(row)
        if missing:
            raise ValueError(f"trajectory row misses fields: {sorted(missing)}")
        if len(self.rows) >= self.expected_frames:
            raise RuntimeError("trajectory recorder refuses samples beyond the frozen episode")
        converted = {name: np.asarray(row[name]).copy() for name in self.REQUIRED}
        if any(not np.isfinite(value).all() for value in converted.values()):
            raise ValueError("trajectory row contains non-finite values")
        if converted["actions"].shape != (26,) or np.max(np.abs(converted["actions"])) > 1.0:
            raise ValueError("trajectory actions must be bounded 26D vectors")
        self.rows.append(converted)

    def finalize(self) -> dict[str, np.ndarray]:
        if len(self.rows) != self.expected_frames:
            raise RuntimeError(
                f"incomplete trajectory: {len(self.rows)} != {self.expected_frames} frames"
            )
        return {name: np.stack([row[name] for row in self.rows], axis=0) for name in self.REQUIRED}


__all__ = ["PhysicsConsistentTrajectoryRecorderV1"]
