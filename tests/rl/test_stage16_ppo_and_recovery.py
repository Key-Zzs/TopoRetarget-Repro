from __future__ import annotations

import numpy as np
import pytest
import torch

from toporetarget.rl.failure_classifier import FailureClass
from toporetarget.rl.ppo.distribution import SoftplusGaussian
from toporetarget.rl.ppo.gae import generalized_advantage_estimate
from toporetarget.rl.ppo.networks import ACTOR_HIDDEN, CRITIC_HIDDEN, ActorCritic
from toporetarget.rl.ppo.storage import RolloutStorage
from toporetarget.rl.randomization import (
    DomainRandomizationConfig,
    RandomizationSwitches,
    sample_randomization,
)
from toporetarget.rl.state_machine import RecoveryBudget, Stage16RecoveryStateMachine


def test_table_six_networks_and_softplus_distribution() -> None:
    model = ActorCritic(7, 3)
    assert ACTOR_HIDDEN == (512, 256, 128)
    assert CRITIC_HIDDEN == (512, 512, 256, 128)
    distribution = SoftplusGaussian(model.mean(torch.zeros(2, 7)), model.log_std_parameter)
    assert torch.all(distribution.std > 0.0)
    assert distribution.log_prob(torch.zeros(2, 3)).shape == (2,)


def test_gae_terminal_bootstrap_and_paper_sample_accounting() -> None:
    rewards = torch.ones(2, 1)
    values = torch.zeros(2, 1)
    dones = torch.tensor([[False], [True]])
    advantages, returns = generalized_advantage_estimate(rewards, values, dones, torch.ones(1))
    assert advantages.shape == returns.shape == (2, 1)
    assert returns[-1, 0] == pytest.approx(1.0)
    storage = RolloutStorage(
        observations=torch.zeros(40, 4096, 1),
        actions=torch.zeros(40, 4096, 1),
        log_probs=torch.zeros(40, 4096),
        rewards=torch.zeros(40, 4096),
        dones=torch.zeros(40, 4096, dtype=torch.bool),
        values=torch.zeros(40, 4096),
    )
    assert storage.sample_count == 163840


def test_randomization_ranges_and_reproducibility() -> None:
    first = sample_randomization(np.random.default_rng(3), DomainRandomizationConfig())
    second = sample_randomization(np.random.default_rng(3), DomainRandomizationConfig())
    assert first == second
    assert 0 <= first["observation_delay_steps"] <= 2
    assert 0.75 <= first["pd_stiffness_scale"] <= 1.5
    assert 0.5 <= first["joint_friction_loss_scale"] <= 2.0


def test_randomization_switches_are_independent_and_nominal_when_disabled() -> None:
    disabled = RandomizationSwitches(
        observation_noise=False,
        observation_delay=False,
        reference_reset=False,
        object_com=False,
        robot_friction_and_geometry=False,
        object_mass_and_inertia=False,
        pd=False,
        joint_dynamics=False,
        encoder_bias=False,
        robot_link_mass_and_inertia=False,
        external_disturbance=False,
    )
    sample = sample_randomization(
        np.random.default_rng(7), DomainRandomizationConfig(switches=disabled)
    )
    assert not any(sample["active_switches"].values())
    assert sample["observation_delay_steps"] == 0
    assert sample["reset_joint_noise_rad"] == 0.0
    assert sample["robot_friction_scale"] == 1.0
    assert sample["pd_stiffness_scale"] == 1.0
    assert sample["next_disturbance_s"] == float("inf")


def test_randomization_ranges_have_non_degenerate_statistical_coverage() -> None:
    samples = [sample_randomization(np.random.default_rng(seed)) for seed in range(128)]
    friction = np.asarray([sample["robot_friction_scale"] for sample in samples])
    stiffness = np.asarray([sample["pd_stiffness_scale"] for sample in samples])
    assert friction.min() >= 0.7 and friction.max() <= 1.3
    assert stiffness.min() >= 0.75 and stiffness.max() <= 1.5
    assert friction.std() > 0.05
    assert stiffness.std() > 0.05


def test_recovery_state_machine_enforces_budgets(tmp_path) -> None:
    machine = Stage16RecoveryStateMachine(
        RecoveryBudget(repairs_per_class=1, reruns_per_phase=1, major_repairs=2)
    )
    initial = machine.record(
        phase="T0",
        failure_class=FailureClass.PPO_NUMERICAL_FAILURE,
        evidence={"nan": True},
        repair="validate",
        rerun_scope="smoke",
    )
    exhausted = machine.record(
        phase="T0",
        failure_class=FailureClass.PPO_NUMERICAL_FAILURE,
        evidence={"nan": True},
        repair="retry",
        rerun_scope="smoke",
    )
    assert initial.result == "RECORDED"
    assert exhausted.result.startswith("ESCALATED")
    path = machine.write_jsonl(tmp_path / "transitions.jsonl")
    assert path.read_text().count("\n") == 2
    resumed = Stage16RecoveryStateMachine.from_jsonl(path, machine.budget)
    follow_up = resumed.record(
        phase="T0",
        failure_class=FailureClass.PPO_NUMERICAL_FAILURE,
        evidence={"nan": True},
        repair="retry_again",
        rerun_scope="smoke",
    )
    assert follow_up.attempt == 3
    resumed.write_jsonl(path)
    assert path.read_text().count("\n") == 3
