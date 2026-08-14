"""Versioned selection of one legal full-trajectory PhysX episode start.

This deliberately answers a smaller question than the historical RSI gates:
whether a clip has one individually valid, table-supported PRE_CONTACT reset.
It neither creates a random-state pool nor changes the immutable reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CLIPS = ("hocap_170105", "hocap_170650")
FULL_TRAJECTORY_EPISODE_START_SCHEMA = "Stage16FullTrajectoryEpisodeStartV1"
TABLE_RESTING_RESET_SCHEMA = "TABLE_RESTING_RESET_SEMANTICS_V1"


@dataclass(frozen=True)
class FullTrajectoryEpisodeStartV1:
    """A single start; no continuous-window or mid-trajectory RSI requirement."""

    clip: str
    start_index: int
    semantic_class: str
    support_state: str
    reference_hash: str
    support_contract_hash: str
    object_linear_velocity_mps: tuple[float, float, float]
    object_angular_velocity_radps: tuple[float, float, float]
    reference_modified: bool = False
    random_state_init: bool = False
    schema_version: str = FULL_TRAJECTORY_EPISODE_START_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_TRAJECTORY_EPISODE_START_SCHEMA:
            raise ValueError("FULL_TRAJECTORY_START_SCHEMA_DRIFT")
        if self.clip not in CLIPS or self.start_index < 0:
            raise ValueError("FULL_TRAJECTORY_START_IDENTITY_INVALID")
        if self.semantic_class != "PRE_CONTACT":
            raise ValueError("FULL_TRAJECTORY_START_NOT_PRE_CONTACT")
        if self.support_state not in {"TABLE_SUPPORTED", "SHARED_SUPPORT"}:
            raise ValueError("FULL_TRAJECTORY_START_NOT_TABLE_SUPPORTED")
        if not self.reference_hash or not self.support_contract_hash:
            raise ValueError("FULL_TRAJECTORY_START_PROVENANCE_MISSING")
        if self.reference_modified or self.random_state_init:
            raise ValueError("FULL_TRAJECTORY_START_SEMANTICS_DRIFT")
        values = (*self.object_linear_velocity_mps, *self.object_angular_velocity_radps)
        if len(values) != 6 or not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError("FULL_TRAJECTORY_START_VELOCITY_INVALID")

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["object_linear_velocity_mps"] = list(self.object_linear_velocity_mps)
        payload["object_angular_velocity_radps"] = list(self.object_angular_velocity_radps)
        payload["episode_horizon"] = "reference[start_index:terminal]"
        payload["mid_trajectory_rsi"] = "disabled"
        payload["table_actor_active"] = True
        return payload


def _row(rows: Mapping[str, np.ndarray], field: str) -> np.ndarray:
    if field not in rows:
        raise ValueError(f"FULL_TRAJECTORY_START_FIELD_MISSING:{field}")
    return np.asarray(rows[field])


def select_full_trajectory_episode_start(
    *,
    clip: str,
    validity_rows: Mapping[str, np.ndarray],
    stable_indices: Sequence[int],
    reference_hash: str,
    support_contract_hash: str,
) -> FullTrajectoryEpisodeStartV1:
    """Prefer frame zero, otherwise choose the earliest individually valid row.

    ``stable_indices`` may be a source-derived early support interval.  It is
    an eligibility boundary only: unlike ``EarlyTableResetCoverageGateV1`` it
    never imposes an 8-frame continuity threshold.
    """

    if clip not in CLIPS:
        raise ValueError("FULL_TRAJECTORY_START_UNKNOWN_CLIP")
    index = _row(validity_rows, "runtime_index").astype(np.int64)
    semantic = _row(validity_rows, "semantic_class").astype("U32")
    support = _row(validity_rows, "support_state").astype("U32")
    geometry = _row(validity_rows, "overall_reference_geometry_valid").astype(bool)
    twist = _row(validity_rows, "reference_object_twist").astype(np.float64)
    if not (index.ndim == semantic.ndim == support.ndim == geometry.ndim == 1):
        raise ValueError("FULL_TRAJECTORY_START_ROWS_SHAPE_INVALID")
    if twist.shape != (len(index), 6):
        raise ValueError("FULL_TRAJECTORY_START_TWIST_SHAPE_INVALID")
    eligible = (
        (semantic == "PRE_CONTACT")
        & np.isin(support, ("TABLE_SUPPORTED", "SHARED_SUPPORT"))
        & geometry
    )
    stable = {int(value) for value in stable_indices}
    if not stable:
        raise ValueError("FULL_TRAJECTORY_START_STABLE_INTERVAL_EMPTY")
    candidates = [int(value) for value in index[eligible] if int(value) in stable]
    if not candidates:
        raise ValueError("P3_RESTART_BLOCKED_EPISODE_START")
    selected = 0 if 0 in candidates else min(candidates)
    position = int(np.flatnonzero(index == selected)[0])
    # Stable table support proves a resting reset; annotation differentiation
    # is not injected as artificial initial object momentum.
    return FullTrajectoryEpisodeStartV1(
        clip=clip,
        start_index=selected,
        semantic_class=str(semantic[position]),
        support_state=str(support[position]),
        reference_hash=reference_hash,
        support_contract_hash=support_contract_hash,
        object_linear_velocity_mps=(0.0, 0.0, 0.0),
        object_angular_velocity_radps=(0.0, 0.0, 0.0),
    )


def validate_full_trajectory_start(payload: Mapping[str, Any], *, clip: str) -> dict[str, object]:
    """Validate a persisted receipt before an Isaac process starts."""

    if payload.get("schema_version") != FULL_TRAJECTORY_EPISODE_START_SCHEMA:
        raise ValueError("FULL_TRAJECTORY_START_RECEIPT_SCHEMA_INVALID")
    if payload.get("clip") != clip:
        raise ValueError("FULL_TRAJECTORY_START_RECEIPT_CLIP_MISMATCH")
    if (
        payload.get("mid_trajectory_rsi") != "disabled"
        or payload.get("table_actor_active") is not True
    ):
        raise ValueError("FULL_TRAJECTORY_START_RECEIPT_SEMANTICS_INVALID")
    return dict(payload)


__all__ = [
    "CLIPS",
    "FULL_TRAJECTORY_EPISODE_START_SCHEMA",
    "TABLE_RESTING_RESET_SCHEMA",
    "FullTrajectoryEpisodeStartV1",
    "select_full_trajectory_episode_start",
    "validate_full_trajectory_start",
]
