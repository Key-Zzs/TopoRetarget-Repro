from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from toporetarget.keypoints.registry import get_layout
from toporetarget.workflows.warm_start_audit import (
    _build_html,
    _jsonable,
    _kabsch,
    _read_contact_rows,
    _root_cause,
)


def test_kabsch_uses_repository_column_vector_convention() -> None:
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([0.2, 0.3, 0.4])
    target = source @ rotation.T + translation
    transform = _kabsch(source, target)
    aligned = source @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(aligned, target, atol=1e-12)


def test_jsonable_turns_nonfinite_values_into_null() -> None:
    assert _jsonable(float("nan")) is None
    assert _jsonable(np.array([1.0, np.inf, -np.inf])) == [1.0, None, None]


def test_contact_reader_preserves_string_provenance(tmp_path: Path) -> None:
    path = tmp_path / "canonical_per_finger_retention.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame",
                "finger",
                "warm_contact_proxy_8mm",
                "canonical_backend_id",
                "distance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame": 2,
                "finger": "thumb",
                "warm_contact_proxy_8mm": "False",
                "canonical_backend_id": "reference_triangle_winding",
                "distance": "0.0125",
            }
        )
    rows = _read_contact_rows(tmp_path)
    assert rows[(2, "thumb")] == {
        "warm_contact_proxy_8mm": False,
        "canonical_backend_id": "reference_triangle_winding",
        "distance": 0.0125,
    }


def test_root_cause_requires_robot_length_evidence_for_morphology_label() -> None:
    attribution = {
        "per_finger": [
            {
                "region": "thumb",
                "warm_contact_proxy": 0.0,
                "warm_ebone": 0.3,
                "warm_keypoint_rmse_m": 0.01,
            }
        ],
        "whole_hand": {
            "warm_keypoint_rmse_m": 0.006,
            "final_keypoint_rmse_m": 0.008,
            "warm_eim_total": 0.001,
            "final_eim_total": 0.002,
            "warm_to_final_change_m": 0.002,
        },
    }
    diagnostic = {
        "results": [],
        "workspace": [
            {
                "frame": 0,
                "nearest_sample_distance_m": 0.012,
                "robot_length_nearest_sample_distance_m": 0.003,
            },
            {
                "frame": 1,
                "nearest_sample_distance_m": 0.011,
                "robot_length_nearest_sample_distance_m": 0.004,
            },
        ],
    }
    per_frame = [
        {
            "frame": 0,
            "warm_to_final_keypoint_change_m": 0.002,
            "warm_contact_proxy": True,
            "final_contact_proxy": False,
        },
        {
            "frame": 1,
            "warm_to_final_keypoint_change_m": 0.001,
            "warm_contact_proxy": False,
            "final_contact_proxy": False,
        },
    ]
    result = _root_cause(
        {"gates": {"replay": True}},
        {"mapping_error_detected": False},
        {"mapping_error_detected": False},
        attribution,
        {},
        {},
        diagnostic,
        per_frame,
    )
    assert result["readiness"] == "WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3"
    assert result["CONTINUE_STAGE9_3_3"] == "YES"
    assert result["ranked_causes"][0]["cause"] == "MORPHOLOGY_LENGTH_MISMATCH_DOMINATES"


def test_html_has_fixed_scale_states_and_actual_units(tmp_path: Path) -> None:
    layout = get_layout("mediapipe21")
    points = np.zeros((1, 21, 3), dtype=np.float64)
    per_frame = [
        {
            "frame": 0,
            "finger": "thumb",
            "warm_keypoint_rmse_m": 0.01,
            "warm_eim_contribution": 0.001,
        }
    ]
    summary = {
        "per_finger": [
            {
                "region": "thumb",
                "warm_ebone": 0.1,
                "warm_keypoint_rmse_m": 0.01,
                "warm_eim_contribution": 0.001,
                "warm_contact_proxy": 0.0,
                "warm_to_final_change": 0.0,
                "joint_limit_min_margin_rad": 0.1,
            }
        ],
        "global_frame_start": 240,
    }
    report = _build_html(
        tmp_path / "audit.html", summary, per_frame, points, points, points, layout
    )
    html = (tmp_path / "audit.html").read_text(encoding="utf-8")
    assert report["required_tokens"]["global fixed scale"]
    assert "source+warm" in html
    assert "actual mm" in html
