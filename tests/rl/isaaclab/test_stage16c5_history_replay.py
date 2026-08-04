"""C.5A recovery limits and fallback-only state-machine tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from toporetarget.rl.isaaclab_oracle.history_replay import (
    raw_control_step,
    synchronize_reset_boundary,
)
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


def test_reset_boundary_sync_matches_direct_env_reset_order() -> None:
    calls: list[str] = []

    class _Scene:
        def write_data_to_sim(self) -> None:
            calls.append("write")

        def update(self, dt: float) -> None:
            assert dt == 1.0 / 120.0
            calls.append("update")

    class _Sim:
        def forward(self) -> None:
            calls.append("forward")

    env = SimpleNamespace(
        device=torch.device("cpu"), scene=_Scene(), sim=_Sim(), physics_dt=1.0 / 120.0
    )
    synchronize_reset_boundary(env)
    assert calls == ["write", "forward", "update"]


def test_raw_control_step_keeps_terminal_state_but_preserves_step_order() -> None:
    calls: list[str] = []

    class _Scene:
        def write_data_to_sim(self) -> None:
            calls.append("write")

        def update(self, _dt: float) -> None:
            calls.append("update")

    class _Sim:
        def step(self, *, render: bool) -> None:
            assert not render
            calls.append("step")

    terminated = torch.tensor([True])
    timed_out = torch.tensor([False])
    env = SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(decimation=2),
        scene=_Scene(),
        sim=_Sim(),
        physics_dt=1.0 / 120.0,
        _sim_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        common_step_counter=0,
        reset_terminated=torch.zeros(1, dtype=torch.bool),
        reset_time_outs=torch.zeros(1, dtype=torch.bool),
        reset_buf=torch.zeros(1, dtype=torch.bool),
        reward_buf=torch.zeros(1),
    )

    def _pre(actions: torch.Tensor) -> None:
        assert tuple(actions.shape) == (1, 26)
        calls.append("pre")

    def _apply() -> None:
        calls.append("apply")

    def _dones() -> tuple[torch.Tensor, torch.Tensor]:
        calls.append("dones")
        return terminated, timed_out

    def _rewards() -> torch.Tensor:
        calls.append("rewards")
        return torch.tensor([1.25])

    env._pre_physics_step = _pre
    env._apply_action = _apply
    env._get_dones = _dones
    env._get_rewards = _rewards

    actual_terminated, actual_timed_out = raw_control_step(env, torch.zeros((1, 26)))
    assert torch.equal(actual_terminated, terminated)
    assert torch.equal(actual_timed_out, timed_out)
    assert env._sim_step_counter == 2
    assert env.common_step_counter == 1
    assert env.episode_length_buf.tolist() == [1]
    assert env.reset_buf.tolist() == [True]
    assert env.reward_buf.tolist() == [1.25]
    assert calls == [
        "pre",
        "apply",
        "write",
        "step",
        "update",
        "apply",
        "write",
        "step",
        "update",
        "dones",
        "rewards",
    ]


def test_raw_control_step_allocates_a_missing_framework_reward_cache() -> None:
    env = SimpleNamespace(
        device=torch.device("cpu"),
        cfg=SimpleNamespace(decimation=0),
        scene=SimpleNamespace(),
        sim=SimpleNamespace(),
        physics_dt=1.0 / 120.0,
        _sim_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        common_step_counter=0,
        reset_terminated=torch.zeros(1, dtype=torch.bool),
        reset_time_outs=torch.zeros(1, dtype=torch.bool),
        reset_buf=torch.zeros(1, dtype=torch.bool),
    )
    env._pre_physics_step = lambda _actions: None
    env._apply_action = lambda: None
    env._get_dones = lambda: (torch.tensor([False]), torch.tensor([False]))
    env._get_rewards = lambda: torch.tensor([2.5])

    raw_control_step(env, torch.zeros((1, 26)))

    assert env.reward_buf.tolist() == [2.5]
