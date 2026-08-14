"""P3-B.7 restart-gate contracts.

The geometric reference remains an immutable soft target.  This module makes
the only relevant entry boundary explicit: a reset state and the actual PhysX
rollout must be physically safe.  In particular, a full-reference geometry
diagnostic is deliberately absent from the hard-gate decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CLIPS = ("hocap_170105", "hocap_170650")
EARLY_TABLE_RESET_POOL_SCHEMA = "Stage16EarlyTableResetPoolV1"
P3_RESTART_GATE_V2_SCHEMA = "Stage16P3RestartGateV2"


@dataclass(frozen=True)
class EarlyTableResetCoverageGateV1:
    """Pre-registered reset-pool sufficiency; never tune it per clip."""

    identifier: str = "EarlyTableResetCoverageGateV1"
    minimum_continuous_frames: int = 8
    required_semantic_class: str = "PRE_CONTACT"
    required_support_states: tuple[str, ...] = ("TABLE_SUPPORTED", "SHARED_SUPPORT")
    required_dynamic_gravity_world_mps2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    required_dynamic_replicas_per_state: int = 4

    def __post_init__(self) -> None:
        if self.minimum_continuous_frames < 1:
            raise ValueError("EARLY_TABLE_RESET_MINIMUM_WINDOW_INVALID")
        if self.required_semantic_class != "PRE_CONTACT":
            raise ValueError("EARLY_TABLE_RESET_SEMANTIC_DRIFT")
        if not self.required_support_states:
            raise ValueError("EARLY_TABLE_RESET_SUPPORT_STATES_MISSING")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def contiguous_windows(indices: Sequence[int]) -> list[dict[str, int]]:
    """Return maximal continuous windows in a sorted reset-index set."""

    values = sorted({int(value) for value in indices})
    windows: list[dict[str, int]] = []
    if not values:
        return windows
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
            continue
        windows.append({"start": start, "end": end, "length": end - start + 1})
        start = end = value
    windows.append({"start": start, "end": end, "length": end - start + 1})
    return windows


def _field(rows: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return np.asarray(rows[name])
    except KeyError as error:
        raise ValueError(f"EARLY_TABLE_RESET_FIELD_MISSING:{name}") from error


def build_early_table_reset_pool(
    *,
    clip: str,
    validity_rows: Mapping[str, np.ndarray],
    dynamic_safe_indices: Sequence[int],
    dynamic_summary: Mapping[str, Any],
    coverage_gate: EarlyTableResetCoverageGateV1 | None = None,
) -> dict[str, object]:
    """Intersect exact reset geometry with actual 1g dynamic qualification.

    ``dynamic_safe_indices`` are receipts from a run with the finite inferred
    table actor and no rollout writes.  They are evidence, not a blacklist.
    """

    if clip not in CLIPS:
        raise ValueError(f"EARLY_TABLE_RESET_UNKNOWN_CLIP:{clip}")
    gate = coverage_gate or EarlyTableResetCoverageGateV1()
    index = _field(validity_rows, "runtime_index").astype(np.int64)
    semantic = _field(validity_rows, "semantic_class").astype("U32")
    support = _field(validity_rows, "support_state").astype("U32")
    geometry = _field(validity_rows, "overall_reference_geometry_valid").astype(bool)
    if not (index.ndim == semantic.ndim == support.ndim == geometry.ndim == 1):
        raise ValueError("EARLY_TABLE_RESET_ROWS_SHAPE_INVALID")
    if not (len(index) == len(semantic) == len(support) == len(geometry)):
        raise ValueError("EARLY_TABLE_RESET_ROWS_LENGTH_MISMATCH")
    if (
        tuple(dynamic_summary.get("gravity_world_mps2", ()))
        != gate.required_dynamic_gravity_world_mps2
    ):
        raise ValueError("EARLY_TABLE_RESET_DYNAMIC_GRAVITY_DRIFT")
    if dynamic_summary.get("support_mode") != "finite_inferred_table_proxy_v1":
        raise ValueError("EARLY_TABLE_RESET_SUPPORT_MODE_INVALID")
    if dynamic_summary.get("external_guidance") is not False:
        raise ValueError("EARLY_TABLE_RESET_GUIDANCE_INVALID")
    if dynamic_summary.get("all_replicas_write_gate_pass") is not True:
        raise ValueError("EARLY_TABLE_RESET_CAUSAL_WRITE_GATE_FAILED")

    static_mask = (
        (semantic == gate.required_semantic_class)
        & np.isin(support, gate.required_support_states)
        & geometry
    )
    static_indices = index[static_mask]
    dynamic = {int(value) for value in dynamic_safe_indices}
    selected = [int(value) for value in static_indices if int(value) in dynamic]
    windows = contiguous_windows(selected)
    qualifying = [row for row in windows if row["length"] >= gate.minimum_continuous_frames]
    longest = max(windows, key=lambda row: row["length"], default=None)
    return {
        "schema_version": EARLY_TABLE_RESET_POOL_SCHEMA,
        "clip": clip,
        "coverage_gate": gate.as_dict(),
        "primary_reset_mode": "early_table_supported",
        "auxiliary_physical_rsi": "OFF",
        "static_early_table_geometry_safe_count": int(len(static_indices)),
        "dynamic_qualified_count": int(len(dynamic)),
        "frames": selected,
        "windows": windows,
        "qualifying_windows": qualifying,
        "longest_window": longest,
        "status": "EARLY_TABLE_SUPPORTED_HARD_RESET_SAFE" if qualifying else "NO_EARLY_HARD_RESET",
        "hard_requirements": {
            "joint_limits": "PASS_IN_EXACT_RESET_GEOMETRY",
            "hand_object": "PASS_IN_EXACT_RESET_GEOMETRY",
            "hand_table": "PASS_IN_EXACT_RESET_GEOMETRY",
            "object_table": "PASS_IN_EXACT_RESET_GEOMETRY",
            "inter_finger": "PASS_IN_EXACT_RESET_GEOMETRY",
            "one_g_dynamic": "PASS_FROM_DYNAMIC_QUALIFICATION",
        },
    }


def stage16_p3_restart_gate_v2(
    *,
    provenance_valid: bool,
    support_valid: bool,
    pools: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Evaluate B.7 in order, never letting diagnostics silently block PPO."""

    if set(pools) != set(CLIPS):
        raise ValueError("P3_RESTART_GATE_V2_CLIP_SET_INVALID")
    r1 = all(bool(pools[clip].get("qualifying_windows")) for clip in CLIPS)
    gates: dict[str, dict[str, object]] = {
        "R0_provenance": {"status": "PASS" if provenance_valid else "FAIL"},
        "R1_hard_reset_validity": {"status": "PASS" if r1 else "FAIL"},
        "R2_support_validity": {"status": "PASS" if support_valid else "FAIL"},
        "R3_residual_recoverability": {"status": "NOT_RUN"},
        "R4_controller_safety": {"status": "NOT_RUN"},
        "R5_ppo_smoke": {"status": "NOT_RUN"},
        "R6_causality": {
            "status": "NOT_RUN",
            "required": {
                "guidance": 0,
                "object_rollout_write": 0,
                "wrist_root_rollout_write": 0,
                "hidden_attachment": 0,
            },
        },
    }
    if not provenance_valid:
        decision = "P3B7_BLOCKED_TECHNICAL"
        next_action = "NEXT_REPAIR_PROVENANCE_INPUT_DRIFT"
    elif not r1:
        decision = "P3B7_BLOCKED_HARD_RESET_POOL"
        next_action = "NEXT_DEBUG_RESET_PIPELINE"
    elif not support_valid:
        decision = "P3B7_BLOCKED_TECHNICAL"
        next_action = "NEXT_DEBUG_INFERRED_TABLE_SUPPORT"
    else:
        decision = "P3B7_READY_FOR_R3"
        next_action = "RUN_RESIDUAL_RECOVERABILITY_AUDIT"
    return {
        "schema_version": P3_RESTART_GATE_V2_SCHEMA,
        "reference_geometry": "DIAGNOSTIC_ONLY",
        "hard_reset_geometry": "HARD_GATE",
        "actual_ppo_rollout_geometry": "HARD_GATE",
        "historical_reference_geometry_blocker_retained": True,
        "gates": gates,
        "decision": decision,
        "next_action": next_action,
    }


__all__ = [
    "CLIPS",
    "EARLY_TABLE_RESET_POOL_SCHEMA",
    "P3_RESTART_GATE_V2_SCHEMA",
    "EarlyTableResetCoverageGateV1",
    "build_early_table_reset_pool",
    "contiguous_windows",
    "stage16_p3_restart_gate_v2",
]
