from __future__ import annotations

import json

import pytest
import torch

from toporetarget.rl.instrumentation.saturation import SaturationRecorder


def _environment(*, steps: int, envs: int) -> dict[str, torch.Tensor]:
    # Recorder accepts a single production step at a time; the helper returns
    # one correctly shaped detached snapshot.
    del steps
    return {
        "scaled_residual": torch.zeros((envs, 26)),
        "pre_safety_command": torch.zeros((envs, 26)),
        "post_safety_command": torch.zeros((envs, 26)),
        "actuator_target": torch.zeros((envs, 26)),
        "actual_joint_q": torch.zeros((envs, 26)),
        "actual_joint_qdot": torch.zeros((envs, 26)),
        "wrist_state": torch.zeros((envs, 13)),
        "phase_code": torch.arange(envs) % 5,
        "hand_object_contact": torch.ones(envs, dtype=torch.bool),
        "hand_object_force": torch.ones(envs),
        "table_object_contact": torch.zeros(envs, dtype=torch.bool),
        "object_tracking_error": torch.zeros(envs),
    }


def test_metric_sign_split_phase_and_pre_gate_persistence(tmp_path) -> None:
    recorder = SaturationRecorder(tmp_path)
    actor = torch.zeros((3, 26))
    actor[0, 0] = 0.99
    actor[1, 6] = -0.99
    actor[2, 25] = 0.99
    for _ in range(2):
        recorder.record_step(
            actor_location=torch.atanh(actor.clamp(-0.999, 0.999)),
            actor_mean=actor,
            actor_log_std=torch.zeros(26),
            sampled_action=actor.clone(),
            environment=_environment(steps=1, envs=3),
        )
    summary, full = recorder.persist_pre_gate(samples_before=100, samples_after=106)
    assert summary["rollout_length"] == 2
    assert summary["num_envs"] == 3
    assert summary["global_saturation"] == pytest.approx(6 / (2 * 3 * 26))
    assert summary["per_dimension_positive"][0] == pytest.approx(1 / 3)
    assert summary["per_dimension_negative"][6] == pytest.approx(1 / 3)
    assert full.is_file()
    payload = torch.load(full, weights_only=True)
    assert payload["actor_mean_tanh"].shape == (2, 3, 26)
    contract = json.loads((tmp_path / "instrumentation_contract.json").read_text())
    assert contract["persistence_order"].endswith("evaluate_gate->update_or_stop")


def test_recorder_does_not_mutate_actor_or_actions(tmp_path) -> None:
    recorder = SaturationRecorder(tmp_path)
    actor = torch.randn((2, 26), requires_grad=True)
    sampled = torch.tanh(actor)
    actor_before = actor.detach().clone()
    sampled_before = sampled.detach().clone()
    recorder.record_step(
        actor_location=actor,
        actor_mean=sampled,
        actor_log_std=torch.zeros(26),
        sampled_action=sampled,
        environment=_environment(steps=1, envs=2),
    )
    assert torch.equal(actor.detach(), actor_before)
    assert torch.equal(sampled.detach(), sampled_before)
    assert actor.grad is None


def test_failure_window_keeps_trigger_and_previous_full_rollout(tmp_path) -> None:
    recorder = SaturationRecorder(tmp_path)
    paths = []
    for value in (0.0, 0.99):
        actor = torch.full((1, 26), value)
        recorder.record_step(
            actor_location=actor,
            actor_mean=actor,
            actor_log_std=torch.zeros(26),
            sampled_action=actor,
            environment=_environment(steps=1, envs=1),
        )
        _, path = recorder.persist_pre_gate(samples_before=0, samples_after=1)
        paths.append(path)
    recorder.preserve_failure_window(triggering=paths[-1])
    assert (tmp_path / "failure/failure_rollout.pt").is_file()
    assert (tmp_path / "failure/previous_full_rollout.pt").is_file()
