"""Unit checks for the Isaac-free C4 Formal20 postprocessor."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module() -> object:
    path = Path(__file__).resolve().parents[1] / "scripts/evaluation/analyze_stage16_c4_formal20.py"
    spec = importlib.util.spec_from_file_location("stage16_c4_formal20_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_episode_penetration_uses_active_frames_for_p95() -> None:
    module = _module()
    summary = module._episode_penetration(np.asarray([0.0, 0.001, 0.003, 0.0]))
    assert summary == {
        "max_penetration_m": 0.003,
        "p95_penetration_m": np.quantile(np.asarray([0.001, 0.003]), 0.95),
        "active_p95_penetration_m": np.quantile(np.asarray([0.001, 0.003]), 0.95),
    }


def test_terminal_stability_uses_contact_specific_limits() -> None:
    module = _module()
    gate = {
        "terminal_window_control_steps": 2,
        "terminal_linear_speed_mps": 0.1,
        "terminal_angular_speed_radps": 0.2,
        "terminal_free_object_linear_speed_mps": 0.3,
        "terminal_free_object_angular_speed_radps": 0.4,
    }
    twist = np.asarray([[0.0] * 6, [0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05] * 6])
    assert module._terminal_stability(
        twist=twist,
        hand_contact=np.asarray([False, False, True]),
        valid=np.asarray([False, True, True]),
        gate=gate,
    )
