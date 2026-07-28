from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.continuous import (
    BASE_CORRECTION_CONVENTION,
    CONTINUOUS_PROFILE_ID,
    S_Q_RAD,
    S_ROT_RAD,
    ContinuousRetargetProfile,
    RecedingHorizonWindow,
    continuity_metrics,
    correction_temporal_energy,
    correction_temporal_residual,
    decode_base_correction,
    encode_base_correction,
    transport_previous_final_to_current_warm,
)


def _pose(translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[:3, 3] = translation
    return value


def test_base_correction_round_trip_and_root_convention() -> None:
    warm = _pose((0.1, -0.2, 0.3))
    correction = np.array([0.01, -0.02, 0.03, 0.1, -0.05, 0.02])
    final = decode_base_correction(warm, correction)
    encoded = encode_base_correction(warm, final)
    assert BASE_CORRECTION_CONVENTION == "scene_local_seed_delta_exp_left"
    np.testing.assert_allclose(encoded, correction, atol=1.0e-10)
    np.testing.assert_allclose(decode_base_correction(warm, encoded), final, atol=1.0e-10)


def test_transport_static_warm_and_zero_correction() -> None:
    warm = _pose((0.2, 0.0, 0.0))
    q = np.linspace(-0.2, 0.2, 20)
    state = transport_previous_final_to_current_warm(
        warm,
        warm,
        warm,
        q,
        q,
        q,
        np.full(20, -1.0),
        np.full(20, 1.0),
        previous_frame=3,
        current_frame=4,
    )
    np.testing.assert_allclose(state.predicted_base_scene, warm)
    np.testing.assert_allclose(state.predicted_qpos, q)
    assert state.q_clamp_count == 0


def test_transport_moving_warm_constant_correction_and_clamp() -> None:
    previous_warm = _pose()
    previous_final = _pose((0.01, 0.0, 0.0))
    current_warm = _pose((0.5, 0.0, 0.0))
    previous_q = np.zeros(20)
    previous_final_q = np.full(20, 0.2)
    current_q = np.zeros(20)
    state = transport_previous_final_to_current_warm(
        previous_warm,
        previous_final,
        current_warm,
        previous_q,
        previous_final_q,
        current_q,
        np.full(20, -0.1),
        np.full(20, 0.1),
        previous_frame=0,
        current_frame=1,
    )
    np.testing.assert_allclose(state.predicted_base_scene[:3, 3], [0.51, 0.0, 0.0])
    np.testing.assert_allclose(state.predicted_qpos, 0.1)
    assert state.q_clamp_count == 20


def test_temporal_residual_scales_and_zero() -> None:
    predicted = _pose()
    final = _pose((0.01, 0.0, 0.0))
    q0 = np.zeros(20)
    q1 = np.full(20, 0.05)
    residual = correction_temporal_residual(predicted, final, q0, q1)
    np.testing.assert_allclose(residual[0], 1.0)
    np.testing.assert_allclose(residual[3], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(residual[6:], 1.0)
    assert correction_temporal_energy(predicted, predicted, q0, q0) == 0.0
    assert S_Q_RAD == 0.05
    assert S_ROT_RAD == pytest.approx(np.deg2rad(5.0))


def test_continuity_gate_and_window_shape() -> None:
    predicted = _pose()
    final = _pose((0.011, 0.0, 0.0))
    metrics = continuity_metrics(predicted, final, np.zeros(20), np.zeros(20), frame=9)
    assert metrics["trajectory_continuous"] is False
    assert "base_translation" in metrics["continuity_failure_reasons"]
    assert RecedingHorizonWindow.for_target(10, 60).variable_frames == (10, 11, 12, 13)
    assert RecedingHorizonWindow.for_target(59, 60).variable_frames == (59,)
    with pytest.raises(ValueError):
        RecedingHorizonWindow.for_target(0, 1, window_size=7)


def test_profile_metadata_is_explicit() -> None:
    profile = ContinuousRetargetProfile.load()
    assert profile.values["profile_id"] == CONTINUOUS_PROFILE_ID
    assert profile.values["paper_core_frame_objective"] == "unchanged"
    assert profile.values["paper_method"] is False
    assert profile.values["author_exact"] == "unresolved"
