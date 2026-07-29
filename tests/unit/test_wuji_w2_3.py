from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from toporetarget.retarget.continuous import (
    CONTINUOUS_PROFILE_ID,
    SEQUENTIAL_PROFILE_ID,
    is_continuous_profile,
)
from toporetarget.retarget.wuji_w2_3 import (
    CLIPS,
    REPLAY_BASE_ROTATION_TOLERANCE,
    REPLAY_BASE_TRANSLATION_TOLERANCE,
    REPLAY_QPOS_TOLERANCE,
    _threshold_key,
    profile_structural_diff,
    selected_replay_frames,
)


def test_sequential_profile_is_continuous_but_distinct() -> None:
    assert is_continuous_profile(CONTINUOUS_PROFILE_ID)
    assert is_continuous_profile(SEQUENTIAL_PROFILE_ID)
    assert not is_continuous_profile("faithful_current_baseline")


def test_profile_diff_allows_only_window_fallback_semantic_change() -> None:
    result = profile_structural_diff(Path(__file__).parents[2])
    assert result["passed"]
    assert result["only_allowed_window_difference"]
    assert result["forbidden_difference_count"] == 0


def test_selected_replay_frames_include_required_retry_evidence() -> None:
    retry = np.asarray(
        [
            "continuous_propagated",
            "propagated_trust_region",
            "deterministic_multi_start",
            "deterministic_multi_start",
            "deterministic_multi_start",
            "deterministic_multi_start",
            "deterministic_multi_start",
        ],
        dtype="S96",
    )
    artifact = SimpleNamespace(arrays={"retry_profile": retry})
    assert selected_replay_frames(artifact, "W1") == [0, 1, 2, 29, 59]
    assert selected_replay_frames(artifact, "W2") == [0, 23, 24, 25, 30, 59]
    assert selected_replay_frames(artifact, "W3") == [0, 1, 2, 3, 4, 5, 6, 30, 59]


def test_replay_tolerances_are_below_continuity_acceptance_scale() -> None:
    assert REPLAY_QPOS_TOLERANCE < 0.05
    assert REPLAY_BASE_TRANSLATION_TOLERANCE < 0.01
    assert REPLAY_BASE_ROTATION_TOLERANCE < 0.08726646259971647


def test_penetration_threshold_keys_are_stable() -> None:
    assert [_threshold_key(value) for value in (0.0, 0.25, 0.5, 1.0, 2.0)] == [
        "R_pen_0p0_mm",
        "R_pen_0p25_mm",
        "R_pen_0p5_mm",
        "R_pen_1p0_mm",
        "R_pen_2p0_mm",
    ]
    assert [clip["unit"] for clip in CLIPS] == ["W1", "W2", "W3"]
