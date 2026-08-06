"""Qualification and anti-degenerate classification for Stage 16-D."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from statistics import fmean
from typing import Any

import numpy as np

from .contracts import PhysicsConsistentTaskGateV1


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("percentile requires values and q in [0,1]")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="linear"))


def qualify_physics_consistent_replicas(
    episodes: Sequence[dict[str, Any]],
    gate: PhysicsConsistentTaskGateV1,
    *,
    formal_replicas: int = 20,
) -> dict[str, Any]:
    if len(episodes) != formal_replicas:
        raise ValueError(f"qualification requires exactly {formal_replicas} replicas")
    for row in episodes:
        if row.get("clip") != gate.clip:
            raise ValueError("qualification clip mismatch")
    success_rate = fmean(float(bool(row["success"])) for row in episodes)
    semantic_rate = fmean(
        float(float(row["semantic_progress"]) >= gate.minimum_semantic_progress) for row in episodes
    )
    contact_rate = fmean(
        float(float(row["contact_recall"]) >= gate.minimum_contact_recall) for row in episodes
    )
    penetration = [float(row["max_penetration_m"]) for row in episodes]
    p95_penetration = [float(row["p95_penetration_m"]) for row in episodes]
    hard_contract = all(
        bool(row["no_hidden_control"])
        and int(row["formal_object_state_writes"]) == 0
        and int(row["formal_wrist_state_writes"]) == 0
        and bool(row["action_bounds_pass"])
        and bool(row["contact_causality_pass"])
        and bool(row["complete_trajectory"])
        for row in episodes
    )
    penetration_pass = (
        max(penetration) <= gate.catastrophic_penetration_m
        and percentile(p95_penetration, 0.95) <= gate.p95_penetration_m
    )
    seed_pass = (
        success_rate >= gate.seed_success_rate
        and semantic_rate >= 0.80
        and contact_rate >= 0.80
        and penetration_pass
        and hard_contract
        and all(bool(row["terminal_stability_pass"]) for row in episodes)
    )
    partial = (
        success_rate >= 0.30
        and fmean(float(row["contact_recall"]) for row in episodes) >= 0.50
        and max(penetration) <= gate.catastrophic_penetration_m
        and all(bool(row["no_hidden_control"]) for row in episodes)
        and all(bool(row["numerical_pass"]) for row in episodes)
        and max(float(row["semantic_progress"]) for row in episodes) > 0.0
    )
    classification = (
        "D3_SEED_VALIDATED"
        if seed_pass
        else "NONDEGENERATE_PARTIAL_SEED"
        if partial
        else "DEGENERATE_SEED"
    )
    return {
        "schema_version": "Stage16DTrajectoryQualificationV1",
        "clip": gate.clip,
        "episodes": formal_replicas,
        "success_rate": success_rate,
        "semantic_reach_rate": semantic_rate,
        "contact_topology_pass_rate": contact_rate,
        "penetration": {
            "max_m": max(penetration),
            "p95_of_episode_p95_m": percentile(p95_penetration, 0.95),
            "passes": penetration_pass,
        },
        "hard_contract_pass": hard_contract,
        "termination_distribution": dict(
            sorted(Counter(str(row["termination"]) for row in episodes).items())
        ),
        "classification": classification,
        "ppo_entry": (
            "STAGE16D_SINGLE_CLIP_PPO_AUTHORIZED"
            if seed_pass
            else "STAGE16D_EXPLORATORY_PPO_AUTHORIZED"
            if partial
            else "PPO_NOT_AUTHORIZED_FOR_CLIP"
        ),
    }


def independent_penetration_audit(
    corrected_penetration_m: np.ndarray,
    source_penetration_m: np.ndarray,
) -> dict[str, Any]:
    corrected = np.asarray(corrected_penetration_m, dtype=np.float64).reshape(-1)
    source = np.asarray(source_penetration_m, dtype=np.float64).reshape(-1)
    if (
        corrected.size == 0
        or source.size == 0
        or not np.isfinite(corrected).all()
        or not np.isfinite(source).all()
    ):
        raise ValueError("penetration audit needs finite nonempty arrays")
    corrected = np.maximum(corrected, 0.0)
    source = np.maximum(source, 0.0)
    max_corrected = float(corrected.max())
    max_source = float(source.max())
    return {
        "schema_version": "Stage16DIndependentPenetrationAuditV1",
        "corrected_max_m": max_corrected,
        "corrected_p95_m": percentile(corrected.tolist(), 0.95),
        "source_stage12_max_m": max_source,
        "source_stage12_p95_m": percentile(source.tolist(), 0.95),
        "relative_max_change": (
            math.inf
            if max_source == 0.0 and max_corrected > 0.0
            else 0.0
            if max_source == 0.0
            else (max_corrected - max_source) / max_source
        ),
        "passes_catastrophic_10mm": max_corrected <= 0.010,
        "passes_p95_3mm": percentile(corrected.tolist(), 0.95) <= 0.003,
        "passes_no_more_than_10pct_degradation": max_corrected <= 1.10 * max_source,
    }


__all__ = [
    "independent_penetration_audit",
    "percentile",
    "qualify_physics_consistent_replicas",
]
