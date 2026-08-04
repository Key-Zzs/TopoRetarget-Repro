"""C.5A recovery limits and fallback-only state-machine tests."""

from __future__ import annotations

from toporetarget.rl.isaaclab_oracle.recovery import Stage16C5ARecoveryStateMachine


def test_recovery_allows_one_history_replay_switch_only() -> None:
    machine = Stage16C5ARecoveryStateMachine()
    machine.transition("TENSOR_CLONE", reason="start")
    machine.record_failure("TENSOR_CLONE_CONTACT_MISMATCH")
    machine.switch_to_history_replay(reason="PhysX cache unavailable")
    assert machine.phase == "HISTORY_REPLAY"
    assert machine.replication_method_switches == 1
    try:
        machine.switch_to_history_replay(reason="second switch")
    except RuntimeError as error:
        assert "SWITCH_BUDGET" in str(error)
    else:
        raise AssertionError("second replication switch was accepted")


def test_recovery_limits_repairs_per_failure_class() -> None:
    machine = Stage16C5ARecoveryStateMachine()
    for _ in range(3):
        machine.record_failure("CANDIDATE_STATE_FIELD_MISSING")
    try:
        machine.record_failure("CANDIDATE_STATE_FIELD_MISSING")
    except RuntimeError as error:
        assert "REPAIR_BUDGET" in str(error)
    else:
        raise AssertionError("fourth repair was accepted")


def test_natural_baseline_failure_closes_without_history_replay() -> None:
    machine = Stage16C5ARecoveryStateMachine()
    machine.transition("NOISE_FLOOR", reason="baseline starts")
    machine.fail_closed(
        "PHYSX_REPLICATION_BASELINE_NONDETERMINISM",
        reason="hard cap exceeded before clone",
    )
    assert machine.phase == "CLOSEOUT"
    assert machine.terminal_failure == "PHYSX_REPLICATION_BASELINE_NONDETERMINISM"
    try:
        machine.switch_to_history_replay(reason="not eligible")
    except RuntimeError as error:
        assert "FAIL_CLOSED" in str(error)
    else:
        raise AssertionError("history replay was allowed after natural baseline failure")
