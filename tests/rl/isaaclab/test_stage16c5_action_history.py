"""Control-step action-history contracts for C.5A replay."""

from __future__ import annotations

import torch

from toporetarget.rl.isaaclab_oracle.action_history import CandidateActionHistoryV1


def test_action_history_preserves_exact_control_step_count() -> None:
    history = CandidateActionHistoryV1()
    for index in range(7):
        history.append(torch.full((1, 26), float(index)))
    history.validate(expected_boundary_index=7)
    assert history.stack().shape == (7, 1, 26)
    assert history.stack()[0, 0, 0].item() == 0.0
    assert history.stack()[-1, 0, 0].item() == 6.0


def test_action_history_rejects_action_dimension_and_boundary_errors() -> None:
    history = CandidateActionHistoryV1()
    try:
        history.append(torch.zeros((1, 25)))
    except ValueError as error:
        assert "[num_envs, 26]" in str(error)
    else:
        raise AssertionError("invalid action dimension accepted")
    history.append(torch.zeros((1, 26)))
    try:
        history.validate(expected_boundary_index=2)
    except ValueError as error:
        assert "length/index mismatch" in str(error)
    else:
        raise AssertionError("omitted action was accepted")
