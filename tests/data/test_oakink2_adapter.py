from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from toporetarget.adapters.datasets.oakink2 import OakInk2AdapterError, OakInk2CanonicalAdapterV1


def _adapter(tmp_path: Path) -> OakInk2CanonicalAdapterV1:
    hub = tmp_path / "data" / "OakInk-v2-hub"
    program = hub / "program" / "program_info"
    program.mkdir(parents=True)
    (hub / "anno_preview").mkdir()
    (program / "scene_01__A001++seq__one.json").write_text(
        json.dumps(
            {
                "((0, 2), (0, 2))": {
                    "primitive": "take",
                    "interaction_mode": "rh_main",
                    "obj_list": ["target"],
                    "obj_list_lh": [],
                    "obj_list_rh": ["target"],
                },
                "((2, 2), (2, 2))": {
                    "primitive": "invalid_interval",
                    "interaction_mode": "rh_main",
                    "obj_list": ["target"],
                    "obj_list_lh": [],
                    "obj_list_rh": ["target"],
                },
            }
        ),
        encoding="utf-8",
    )
    return OakInk2CanonicalAdapterV1(tmp_path)


def _annotation() -> dict[str, object]:
    pose = np.zeros((1, 16, 4), dtype=np.float64)
    pose[..., 0] = 1.0
    hand = {
        "rh__pose_coeffs": pose,
        "rh__tsl": np.zeros((1, 3), dtype=np.float64),
        "rh__betas": np.zeros((1, 10), dtype=np.float64),
    }
    return {
        "raw_mano": {0: hand, 1: hand, 2: hand},
        "obj_transf": {"target": {0: np.eye(4), 1: np.eye(4), 2: np.eye(4)}},
        "obj_list": ["target"],
        "mocap_frame_id_list": [0, 1, 2],
    }


def test_oakink2_adapter_uses_official_primitive_intervals(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)

    tasks = adapter.primitives()

    assert tasks[0].rh_interval == (0, 2)
    assert tasks[0].record_id.endswith(":00000")
    assert tasks[1].rh_interval is None


def test_oakink2_adapter_extracts_canonical_right_hand_and_object_tracks(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    annotation = _annotation()
    frames = adapter.select_interval((0, 2), adapter.available_frames(annotation))

    hand = adapter.hand_track(annotation, "right", frames)
    object_track = adapter.object_track(annotation, "target", frames)

    assert frames.tolist() == [0, 1]
    assert hand["pose_quat_wxyz"].shape == (2, 16, 4)
    assert hand["translation_world"].shape == (2, 3)
    assert object_track.shape == (2, 4, 4)
    assert np.allclose(adapter.quaternion_matrices_wxyz(hand["pose_quat_wxyz"][:, 0]), np.eye(3))


def test_oakink2_exact_frame_lookup_and_half_open_interval_fail_closed(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    annotation = _annotation()

    assert adapter.select_interval((0, 2), adapter.available_frames(annotation)).tolist() == [0, 1]
    with pytest.raises(OakInk2AdapterError, match="OAKINK2_MANO_FRAME_MISSING:3"):
        adapter.hand_track(annotation, "right", np.asarray([3]))


def test_oakink2_wxyz_quaternion_known_rotation_and_xyzw_negative_control(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    half = np.sqrt(0.5)
    wxyz = np.asarray([[half, 0.0, 0.0, half]])
    expected = np.asarray([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    actual = adapter.quaternion_matrices_wxyz(wxyz)
    wrong_xyzw_interpretation = adapter.quaternion_matrices_wxyz(wxyz[:, [1, 2, 3, 0]])

    assert np.allclose(actual, expected, atol=1e-7)
    assert not np.allclose(wrong_xyzw_interpretation, expected, atol=1e-3)
