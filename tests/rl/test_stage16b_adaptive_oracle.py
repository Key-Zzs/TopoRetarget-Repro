from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from toporetarget.rl.stage16b_recovery import (
    AdaptiveOracleFailure,
    Stage16BAdaptiveOracleStateMachine,
    Stage16BSingleClipPPOStateMachine,
)
from toporetarget.rl.world_wrist_oracle import (
    AdaptiveMultiHorizonContactOracle,
    GateFirstHorizonSelector,
    effective_horizon_portfolio,
    formal_gate_barrier,
)


def test_adaptive_oracle_uses_bounded_full_clip_viability_window() -> None:
    assert AdaptiveMultiHorizonContactOracle().selection_lookahead == 10
    with pytest.raises(ValueError, match="lookahead"):
        AdaptiveMultiHorizonContactOracle(selection_lookahead=41)


@pytest.mark.parametrize(
    ("remaining", "expected"),
    [
        (12, (1, 5, 10)),
        (9, (1, 5, 9)),
        (5, (1, 5)),
        (4, (1, 4)),
        (2, (1, 2)),
        (1, (1,)),
        (0, ()),
    ],
)
def test_effective_horizon_portfolio_has_no_terminal_padding(
    remaining: int, expected: tuple[int, ...]
) -> None:
    assert effective_horizon_portfolio(remaining) == expected
    assert len(expected) == len(set(expected))
    assert all(horizon <= remaining for horizon in expected)


def test_formal_gate_barrier_is_monotonic_and_penalizes_violation() -> None:
    values = formal_gate_barrier(np.asarray([0.0, 0.5, 0.9, 0.99, 1.0, 1.1]))
    assert np.diff(values).min() > 0.0
    assert values[4] >= 1000.0
    assert values[5] > values[4]
    with pytest.raises(ValueError, match="non-negative"):
        formal_gate_barrier(-0.1)


def _candidate(
    *,
    horizon: int,
    termination: str | None = None,
    gate: float = 0.5,
    contact_loss: float = 0.0,
    impulse: float = 0.0,
) -> SimpleNamespace:
    result = SimpleNamespace(
        predicted_termination=termination,
        gate_violation=gate,
        minimum_gate_margin=1.0 - gate,
        near_axis_error_m=0.01,
        near_object_position_error_m=0.01,
        near_object_rotation_error_rad=0.1,
        reference_complete=False,
        contact_loss=contact_loss,
        excessive_impulse=impulse,
        max_penetration_m=0.0,
        wrist_error=0.1,
        first_difference=0.1,
        second_difference=0.1,
        effort=0.1,
    )
    return SimpleNamespace(result=result, effective_horizon=horizon)


def test_gate_first_selector_prioritizes_hard_gate_then_contact_then_short_tie() -> None:
    selector = GateFirstHorizonSelector()
    safe = _candidate(horizon=10, gate=0.9, contact_loss=1.0)
    unsafe = _candidate(horizon=1, termination="FAILURE_OBJECT_POSITION", gate=0.2)
    selected, reason = selector.select([unsafe, safe])
    assert selected is safe
    assert "lexicographic_gate_first" in reason

    contact = _candidate(horizon=5, contact_loss=0.0, impulse=1.0)
    no_contact = _candidate(horizon=1, contact_loss=1.0, impulse=0.0)
    assert selector.select([no_contact, contact])[0] is contact

    long_tie = _candidate(horizon=10)
    short_tie = _candidate(horizon=1)
    assert selector.select([long_tie, short_tie])[0] is short_tie
    assert "clip" not in inspect.signature(selector.select).parameters


def test_stage16b_state_machines_are_bounded() -> None:
    oracle = Stage16BAdaptiveOracleStateMachine()
    for _ in range(4):
        transition = oracle.record(
            failure=AdaptiveOracleFailure.CEM_CONTACT_MODE_MISS,
            evidence={},
            fallback="cross_horizon_seed",
            repair="bounded_repair",
            rerun="both_clips",
            result="RETRY",
        )
    assert transition.result == "BLOCKED_CLASS_REPAIR_BUDGET_EXHAUSTED"
    assert not oracle.as_dict()["bounded"]

    ppo = Stage16BSingleClipPPOStateMachine()
    assert ppo.as_dict()["formal_reruns"] == 5
    assert ppo.as_dict()["major_transitions"] == 16
