from __future__ import annotations

import pytest
import torch

from toporetarget.rl.physics_retargeting.terminal_refinement import (
    TerminalTailRefinementConfigV1,
    materialize_terminal_tail,
)


def test_terminal_tail_freezes_prefix_and_continuity() -> None:
    config = TerminalTailRefinementConfigV1()
    baseline = torch.linspace(-0.5, 0.5, 321 * 26).reshape(321, 26)
    knots = torch.randn((3, 8, 26))
    actions = materialize_terminal_tail(baseline, knots, config)
    assert actions.shape == (3, 321, 26)
    torch.testing.assert_close(
        actions[:, : config.tail_start],
        baseline[: config.tail_start].expand(3, -1, -1),
    )
    torch.testing.assert_close(
        actions[:, config.tail_start],
        baseline[config.tail_start].expand(3, -1),
    )
    assert bool((actions.abs() <= 1.0).all())


def test_terminal_refinement_budget_is_exact() -> None:
    with pytest.raises(ValueError, match="frozen"):
        TerminalTailRefinementConfigV1(population=128)
