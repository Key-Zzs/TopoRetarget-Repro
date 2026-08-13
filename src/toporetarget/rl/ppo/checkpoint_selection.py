"""Frame-zero semantic checkpoint selection for Stage 16-D."""

from __future__ import annotations

from typing import Any


def select_physics_correction_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("checkpoint selection requires at least one record")
    if any(not bool(row.get("frame_zero_full_episode")) for row in records):
        raise ValueError("Stage16D checkpoint selection requires frame-zero full episodes")

    def key(row: dict[str, Any]) -> tuple[float | str, ...]:
        return (
            float(row["success_rate"]),
            float(row["semantic_reach_rate"]),
            float(row["contact_pass_rate"]),
            float(row["penetration_pass_rate"]),
            -float(row["robot_deviation"]),
            -float(row["action_smoothness"]),
            str(row["checkpoint"]),
        )

    return max(records, key=key)


__all__ = ["select_physics_correction_checkpoint"]
