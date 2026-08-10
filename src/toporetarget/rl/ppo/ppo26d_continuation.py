"""Bounded decisions and summaries for the Stage 16-D PPO-26D continuation.

The module deliberately has no Isaac imports.  This makes the frozen decision
rules testable in the base package before an Isaac Lab process is launched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Stage16DPPO26DContinuationStateMachine(str, Enum):
    """The only ordered states allowed by the Stage 16-D continuation plan."""

    INPUT_FREEZE = "INPUT_FREEZE"
    DOCS_UPDATE = "DOCS_UPDATE"
    L0_REBASELINE = "L0_REBASELINE"
    R6A_RESUME_4M = "R6A_RESUME_4M"
    LEARNING_DIAGNOSIS = "LEARNING_DIAGNOSIS"
    PPO_UPDATE_DIAGNOSIS = "PPO_UPDATE_DIAGNOSIS"
    R6B_CONTINUE = "R6B_CONTINUE"
    R6C_RSI_CURRICULUM = "R6C_RSI_CURRICULUM"
    R6D_REWARD_V2 = "R6D_REWARD_V2"
    CHECKPOINT_SELECTION = "CHECKPOINT_SELECTION"
    R7_POST_PPO_QUALIFICATION = "R7_POST_PPO_QUALIFICATION"
    R8_170105 = "R8_170105"
    R8_QUALIFICATION = "R8_QUALIFICATION"
    D6_MULTICLIP = "D6_MULTICLIP"
    D6_QUALIFICATION = "D6_QUALIFICATION"
    D7_EXPORT = "D7_EXPORT"
    CLOSEOUT = "CLOSEOUT"


@dataclass(frozen=True)
class FrozenSeedSet:
    """A named, reproducible evaluation seed set with no hidden sampling."""

    identifier: str
    seeds: tuple[int, ...]
    purpose: str

    def as_dict(self) -> dict[str, object]:
        return {"identifier": self.identifier, "seeds": list(self.seeds), "purpose": self.purpose}


def generate_frozen_seed_set(
    identifier: str, *, base_seed: int, count: int, purpose: str
) -> FrozenSeedSet:
    """Generate unique integer seeds without consuming process-global RNG state."""

    if count <= 0:
        raise ValueError("frozen seed-set count must be positive")
    generator = np.random.default_rng(base_seed)
    values = tuple(int(value) for value in generator.choice(2**31 - 1, size=count, replace=False))
    return FrozenSeedSet(identifier=identifier, seeds=values, purpose=purpose)


def quantile(values: Iterable[float], percentile: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary quantile requires finite non-empty values")
    return float(np.quantile(array, percentile, method="linear"))


def summarize_episodes(rows: Iterable[dict[str, Any]]) -> dict[str, object]:
    """Summarize the common development/formal evaluator episode schema."""

    episodes = list(rows)
    if not episodes:
        raise ValueError("at least one episode is required")
    for row in episodes:
        required = {"contact", "terminal_contact", "contact_step_count", "object_tracking_error_m"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"episode metric missing fields: {sorted(missing)}")
    contact_steps = [float(row["contact_step_count"]) for row in episodes]
    final_object_errors = [float(row["object_tracking_error_m"]["final"]) for row in episodes]
    first_contact = [
        float(row["first_contact_index"])
        for row in episodes
        if row.get("first_contact_index") is not None
    ]
    last_contact = [
        float(row["last_contact_index"])
        for row in episodes
        if row.get("last_contact_index") is not None
    ]
    longest_window = [float(row.get("longest_continuous_contact_window", 0)) for row in episodes]
    terminal_stable = [bool(row.get("terminal_stable", False)) for row in episodes]
    rotation = [float(row.get("final_object_rotation_error_rad", float("nan"))) for row in episodes]
    axis = [float(row.get("final_object_axis_error_m", float("nan"))) for row in episodes]
    linear_speed = [
        float(row.get("terminal_object_linear_speed_mps", float("nan"))) for row in episodes
    ]
    angular_speed = [
        float(row.get("terminal_object_angular_speed_radps", float("nan"))) for row in episodes
    ]

    def finite_summary(values: list[float]) -> dict[str, float | None]:
        valid = [value for value in values if np.isfinite(value)]
        if not valid:
            return {"mean": None, "median": None, "p95": None}
        return {
            "mean": float(np.mean(valid)),
            "median": quantile(valid, 0.5),
            "p95": quantile(valid, 0.95),
        }

    return {
        "episodes": len(episodes),
        "ever_contact_rate": float(np.mean([bool(row["contact"]) for row in episodes])),
        "terminal_contact_rate": float(
            np.mean([bool(row["terminal_contact"]) for row in episodes])
        ),
        "terminal_stability_rate": float(np.mean(terminal_stable)),
        "contact_steps": finite_summary(contact_steps),
        "first_contact_index": finite_summary(first_contact),
        "last_contact_index": {
            "median": quantile(last_contact, 0.5) if last_contact else None,
            "p75": quantile(last_contact, 0.75) if last_contact else None,
            "max": max(last_contact) if last_contact else None,
        },
        "longest_continuous_contact_window": finite_summary(longest_window),
        "final_object_position_error_m": finite_summary(final_object_errors),
        "final_object_rotation_error_rad": finite_summary(rotation),
        "final_object_axis_error_m": finite_summary(axis),
        "terminal_object_linear_speed_mps": finite_summary(linear_speed),
        "terminal_object_angular_speed_radps": finite_summary(angular_speed),
    }


class R6ADecision(str, Enum):
    IMPROVING = "IMPROVING"
    RSI_GOOD_FRAME_ZERO_BAD = "RSI_GOOD_FRAME_ZERO_BAD"
    PLATEAU = "PLATEAU"
    AMBIGUOUS_ONE_TIME_EXTENSION = "AMBIGUOUS_ONE_TIME_EXTENSION"


class RSICurriculumPhase(str, Enum):
    """The three bounded, telemetry-derived RSI curriculum phases."""

    C0 = "C0"
    C1 = "C1"
    C2 = "C2"


@dataclass(frozen=True)
class R6ADecisionResult:
    decision: R6ADecision
    indicators: tuple[str, ...]
    deltas: dict[str, float]
    extension_allowed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision.value,
            "indicators": list(self.indicators),
            "deltas": self.deltas,
            "extension_allowed": self.extension_allowed,
        }


def _metric(summary: dict[str, object], *keys: str) -> float:
    value: object = summary
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing metric: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, (int, float)):
        raise ValueError(f"metric is unavailable: {'.'.join(keys)}")
    return float(value)


def classify_r6a(
    *,
    baseline_frame_zero: dict[str, object],
    baseline_rsi: dict[str, object],
    four_m_frame_zero: dict[str, object],
    four_m_rsi: dict[str, object],
    late_object_reward_baseline: float,
    late_object_reward_four_m: float,
    no_new_safety_failure: bool,
    no_reward_exploit: bool,
    extension_already_used: bool = False,
) -> R6ADecisionResult:
    """Apply the pre-frozen R6A decision tree with branch-B precedence."""

    fz_terminal_before = _metric(baseline_frame_zero, "terminal_contact_rate")
    fz_terminal_after = _metric(four_m_frame_zero, "terminal_contact_rate")
    rsi_terminal_before = _metric(baseline_rsi, "terminal_contact_rate")
    rsi_terminal_after = _metric(four_m_rsi, "terminal_contact_rate")
    contact_before = _metric(baseline_frame_zero, "contact_steps", "median")
    contact_after = _metric(four_m_frame_zero, "contact_steps", "median")
    last_before = _metric(baseline_frame_zero, "last_contact_index", "p75")
    last_after = _metric(four_m_frame_zero, "last_contact_index", "p75")
    error_before = _metric(baseline_frame_zero, "final_object_position_error_m", "median")
    error_after = _metric(four_m_frame_zero, "final_object_position_error_m", "median")
    reward_change = (late_object_reward_four_m - late_object_reward_baseline) / max(
        abs(late_object_reward_baseline), 1.0e-12
    )
    error_reduction = (error_before - error_after) / max(error_before, 1.0e-12)
    deltas = {
        "frame_zero_terminal_contact_absolute": fz_terminal_after - fz_terminal_before,
        "rsi_terminal_contact_absolute": rsi_terminal_after - rsi_terminal_before,
        "median_contact_steps": contact_after - contact_before,
        "p75_last_contact_index": last_after - last_before,
        "median_object_error_relative_reduction": error_reduction,
        "late_object_reward_relative_change": reward_change,
    }
    indicators = []
    if fz_terminal_after - fz_terminal_before >= 0.15 or fz_terminal_after >= 3.0 / 20.0:
        indicators.append("I1_FRAME_ZERO_TERMINAL_CONTACT")
    if contact_after - contact_before >= 10.0:
        indicators.append("I2_MEDIAN_CONTACT_STEPS")
    if last_after - last_before >= 30.0:
        indicators.append("I3_P75_LAST_CONTACT_INDEX")
    if error_reduction >= 0.20:
        indicators.append("I4_MEDIAN_OBJECT_ERROR")
    if rsi_terminal_after - rsi_terminal_before >= 0.20:
        indicators.append("I5_RSI_TERMINAL_CONTACT")
    if reward_change >= 0.20:
        indicators.append("I6_LATE_OBJECT_REWARD")
    if (
        rsi_terminal_after >= 0.60
        and fz_terminal_after <= 0.20
        and rsi_terminal_after - fz_terminal_after >= 0.40
    ):
        decision = R6ADecision.RSI_GOOD_FRAME_ZERO_BAD
    elif len(indicators) >= 2 and no_new_safety_failure and no_reward_exploit:
        decision = R6ADecision.IMPROVING
    elif (
        abs(fz_terminal_after - fz_terminal_before) < 0.10
        and abs(rsi_terminal_after - rsi_terminal_before) < 0.10
        and (contact_after - contact_before) / max(abs(contact_before), 1.0) < 0.25
        and error_reduction < 0.10
        and reward_change < 0.10
    ):
        decision = R6ADecision.PLATEAU
    else:
        decision = R6ADecision.AMBIGUOUS_ONE_TIME_EXTENSION
    return R6ADecisionResult(
        decision=decision,
        indicators=tuple(indicators),
        deltas=deltas,
        extension_allowed=decision is R6ADecision.AMBIGUOUS_ONE_TIME_EXTENSION
        and not extension_already_used,
    )


def classify_ppo_update_bottleneck(iterations: Iterable[dict[str, Any]]) -> dict[str, object]:
    """Determine whether the one allowed LR-only fallback is authorized."""

    rows = list(iterations)
    if not rows:
        raise ValueError("PPO update diagnosis needs at least one training iteration")
    early = [row for row in rows if bool(row.get("ppo", {}).get("kl_early_stop", False))]
    actual_epochs = [float(row["ppo"]["actual_epochs_executed"]) for row in rows]
    first_epoch_kl = [float(row["ppo"]["kl_per_epoch"][0]) for row in rows]
    target_kl = _metric(rows[0], "ppo", "target_kl")
    early_ratio = len(early) / len(rows)
    median_epochs = quantile(actual_epochs, 0.5)
    first_epoch_excess = float(np.mean([value > 1.5 * target_kl for value in first_epoch_kl]))
    possible = early_ratio > 0.80 and median_epochs <= 1.5 and first_epoch_excess > 0.0
    return {
        "iterations": len(rows),
        "kl_early_stop_ratio": early_ratio,
        "median_actual_epochs_executed": median_epochs,
        "first_epoch_kl_exceeds_1_5_target_ratio": first_epoch_excess,
        "target_kl": target_kl,
        "classification": "POSSIBLE_PPO_UPDATE_BOTTLENECK" if possible else "NOT_UPDATE_BOTTLENECK",
        "lr_fallback": {
            "authorized_if_task_metrics_plateau": possible,
            "from": 1.0e-4,
            "to": 5.0e-5,
            "probe_budget_samples": 1_000_000,
            "other_contract_changes": "FORBIDDEN",
        },
    }


def extract_rsi_curriculum_regions(
    rows: Iterable[dict[str, Any]], *, frame_count: int
) -> dict[str, tuple[int, ...]]:
    """Derive contact and approach regions from evaluated RSI telemetry.

    The evaluator emits reference-relative first/last contact indices for each
    independently seeded episode.  This function projects them into the
    canonical reference timeline and derives the pre-contact span from the
    observed contact-window width.  No clip ID or fixed frame boundary enters
    the procedure.
    """

    if frame_count <= 1:
        raise ValueError("curriculum needs a reference with at least two frames")
    contact: set[int] = set()
    pre_contact: set[int] = set()
    for row in rows:
        first = row.get("first_contact_index")
        last = row.get("last_contact_index")
        start = row.get("start_reference_index")
        if first is None or last is None or start is None:
            continue
        first_index = int(start) + int(first)
        last_index = int(start) + int(last)
        if last_index < first_index:
            raise ValueError("RSI contact telemetry has inverted contact bounds")
        first_index = min(max(first_index, 0), frame_count - 1)
        last_index = min(max(last_index, 0), frame_count - 1)
        if last_index < first_index:
            continue
        contact.update(range(first_index, last_index + 1))
        width = max(1, last_index - first_index + 1)
        pre_contact.update(range(max(0, first_index - width), first_index))
    if not contact:
        raise ValueError("RSI curriculum needs at least one observed contact window")
    if not pre_contact:
        pre_contact.add(0)
    return {
        "contact_persistent": tuple(sorted(contact)),
        "pre_contact_late_approach": tuple(sorted(pre_contact)),
        "frame_zero": (0,),
        "uniform_full_reference": tuple(range(frame_count)),
    }


def build_rsi_curriculum_distribution(
    rows: Iterable[dict[str, Any]], *, frame_count: int, phase: RSICurriculumPhase
) -> dict[str, object]:
    """Build C0/C1/C2 reset probabilities from actual evaluated contact spans."""

    regions = extract_rsi_curriculum_regions(rows, frame_count=frame_count)
    component_weights: dict[RSICurriculumPhase, tuple[tuple[str, float], ...]] = {
        RSICurriculumPhase.C0: (
            ("contact_persistent", 0.60),
            ("pre_contact_late_approach", 0.30),
            ("frame_zero", 0.10),
        ),
        RSICurriculumPhase.C1: (
            ("contact_persistent", 0.40),
            ("uniform_full_reference", 0.30),
            ("frame_zero", 0.30),
        ),
        RSICurriculumPhase.C2: (
            ("contact_persistent", 0.20),
            ("uniform_full_reference", 0.30),
            ("frame_zero", 0.50),
        ),
    }
    probabilities = np.zeros(frame_count, dtype=np.float64)
    components = component_weights[phase]
    for region_name, total_weight in components:
        indices = np.asarray(regions[region_name], dtype=np.int64)
        probabilities[indices] += total_weight / len(indices)
    probabilities /= probabilities.sum()
    supported = np.flatnonzero(probabilities > 0.0)
    return {
        "contract": "Stage16DPPO26DRSICurriculumV1",
        "phase": phase.value,
        "frame_count": frame_count,
        "components": [
            {
                "region": region_name,
                "weight": total_weight,
                "derived_frame_count": len(regions[region_name]),
            }
            for region_name, total_weight in components
        ],
        "regions": {name: list(indices) for name, indices in regions.items()},
        "reference_indices": supported.tolist(),
        "probabilities": probabilities[supported].tolist(),
        "frame_zero_probability": float(probabilities[0]),
    }


def rank_development_checkpoints(candidates: Iterable[dict[str, Any]]) -> list[dict[str, object]]:
    """Rank development checkpoints by the frozen R6 lexicographic rule."""

    rows = list(candidates)
    if not rows:
        raise ValueError("checkpoint selection needs at least one development evaluation")

    def mean(rows: list[dict[str, Any]], field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    scored: list[dict[str, object]] = []
    for candidate in rows:
        frame_zero = candidate.get("frame_zero")
        rsi = candidate.get("rsi")
        if (
            not isinstance(frame_zero, list)
            or not isinstance(rsi, list)
            or not frame_zero
            or not rsi
        ):
            raise ValueError("checkpoint evaluation must include non-empty frame-zero and RSI rows")
        frame_summary = summarize_episodes(frame_zero)
        rsi_summary = summarize_episodes(rsi)
        completion = float(np.mean([bool(row["reached_final_reference"]) for row in frame_zero]))
        total_reward = mean(frame_zero, "total_reward")
        saturation = float(candidate.get("action_saturation", 0.0))
        sample_count = int(candidate["cumulative_training_samples"])
        key = (
            -completion,
            -_metric(frame_summary, "terminal_contact_rate"),
            -_metric(frame_summary, "terminal_stability_rate"),
            -_metric(frame_summary, "longest_continuous_contact_window", "median"),
            _metric(frame_summary, "final_object_position_error_m", "median"),
            _metric(frame_summary, "final_object_rotation_error_rad", "median"),
            -_metric(rsi_summary, "terminal_contact_rate"),
            -total_reward,
            saturation,
            sample_count,
        )
        scored.append(
            {
                "checkpoint": candidate.get("checkpoint"),
                "checkpoint_sha256": candidate.get("checkpoint_sha256"),
                "cumulative_training_samples": sample_count,
                "selection_key": list(key),
                "selection_metrics": {
                    "frame_zero_reference_completion_rate": completion,
                    "frame_zero_terminal_contact_rate": _metric(
                        frame_summary, "terminal_contact_rate"
                    ),
                    "frame_zero_terminal_stability_rate": _metric(
                        frame_summary, "terminal_stability_rate"
                    ),
                    "frame_zero_longest_contact_median": _metric(
                        frame_summary, "longest_continuous_contact_window", "median"
                    ),
                    "frame_zero_final_object_position_error_median": _metric(
                        frame_summary, "final_object_position_error_m", "median"
                    ),
                    "frame_zero_final_object_rotation_error_median": _metric(
                        frame_summary, "final_object_rotation_error_rad", "median"
                    ),
                    "rsi_terminal_contact_rate": _metric(rsi_summary, "terminal_contact_rate"),
                    "frame_zero_total_reward_mean": total_reward,
                    "action_saturation": saturation,
                },
            }
        )

    def selection_key(row: dict[str, object]) -> tuple[float, ...]:
        value = row["selection_key"]
        if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
            raise ValueError("checkpoint selection key is invalid")
        return tuple(float(item) for item in value)

    return sorted(scored, key=selection_key)


@dataclass
class ContinuationTransitions:
    """Records state-machine progression for ``failure_transitions.jsonl``."""

    state: Stage16DPPO26DContinuationStateMachine = (
        Stage16DPPO26DContinuationStateMachine.INPUT_FREEZE
    )
    transitions: list[dict[str, str]] = field(default_factory=list)

    def transition(self, target: Stage16DPPO26DContinuationStateMachine, *, reason: str) -> None:
        if target is self.state:
            raise ValueError("continuation state must advance")
        self.transitions.append({"from": self.state.value, "to": target.value, "reason": reason})
        self.state = target


__all__ = [
    "ContinuationTransitions",
    "FrozenSeedSet",
    "R6ADecision",
    "R6ADecisionResult",
    "RSICurriculumPhase",
    "Stage16DPPO26DContinuationStateMachine",
    "build_rsi_curriculum_distribution",
    "classify_ppo_update_bottleneck",
    "classify_r6a",
    "extract_rsi_curriculum_regions",
    "generate_frozen_seed_set",
    "rank_development_checkpoints",
    "summarize_episodes",
]
