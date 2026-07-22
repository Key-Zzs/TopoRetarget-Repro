from pathlib import Path

import numpy as np
import pytest

from toporetarget.workflows.contact_audit import (
    _anchor_provenance,
    _dense_samples,
    _grouped_stats,
    _html,
    _json,
    _object_local_direction,
    _region_indices,
    _shadow_frame_selection,
    _shadow_score_diagnostic,
    _slerp,
    _threshold_record,
)


def test_dense_samples_are_deterministic_and_keep_original_vertices() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.asarray([[0, 1, 2]])

    first = _dense_samples(vertices, faces, count=32, mesh_id="triangle", seed=7)
    second = _dense_samples(vertices, faces, count=32, mesh_id="triangle", seed=7)

    assert first.shape == (32, 3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:3], vertices)


def test_threshold_record_preserves_positive_outside_sign_convention() -> None:
    signed = np.asarray([-0.002, 0.001, 0.010])
    unsigned = np.abs(signed)

    record = _threshold_record(signed, unsigned, [0.002, 0.005])

    assert record["2mm"]["penetration_count"] == 1
    assert record["2mm"]["positive_gap_count"] == 1
    assert record["5mm"]["near_surface_count"] == 2


def test_object_local_direction_applies_inverse_pose_rotation() -> None:
    pose = np.eye(4)
    pose[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    direction, distance = _object_local_direction(np.asarray([0.0, 1.0, 0.0]), np.zeros(3), pose)

    assert distance == pytest.approx(1.0)
    np.testing.assert_allclose(direction, np.asarray([1.0, 0.0, 0.0]))


def test_anchor_provenance_keeps_face_and_object_local_vector() -> None:
    class Query:
        closest_points = np.asarray([[1.0, 2.0, 3.0]])
        closest_face_indices = np.asarray([17])
        signed_distance = np.asarray([0.002])
        unsigned_distance = np.asarray([0.002])

    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    record = _anchor_provenance(np.asarray([[1.0, 2.0, 4.0]]), Query(), pose)[0]

    assert record["closest_face_index"] == 17
    assert record["point_object_local_m"] == pytest.approx([0.0, 0.0, 1.0])
    assert record["nearest_vector_object_local_m"] == pytest.approx([0.0, 0.0, 1.0])


def test_grouped_stats_exposes_overall_and_per_link_values() -> None:
    result = _grouped_stats(
        np.asarray([0.001, 0.002, 0.003]), np.asarray(["palm", "index", "index"])
    )

    assert result["overall"]["count"] == 3
    assert result["by_label"]["index"]["count"] == 2


def test_slerp_interpolates_rotation_and_translation() -> None:
    first = np.eye(4)
    second = np.eye(4)
    second[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    second[:3, 3] = [2.0, 0.0, 0.0]

    midpoint = _slerp(first, second, 0.5)

    np.testing.assert_allclose(midpoint[:3, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(midpoint[:3, :3] @ midpoint[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(midpoint[:3, :3]) == pytest.approx(1.0)


def test_region_indices_map_finger_anchor_chains() -> None:
    anchors = np.zeros((21, 3))
    anchors[8] = [1.0, 0.0, 0.0]
    anchors[12] = [0.0, 1.0, 0.0]
    points = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    assert _region_indices(points, anchors).tolist() == ["index", "middle", "palm"]


def test_json_normalizes_nonfinite_values() -> None:
    value = _json({"nan": float("nan"), "inf": float("inf"), "finite": np.float64(2.0)})

    assert value == {"nan": None, "inf": None, "finite": 2.0}


def test_shadow_diagnostic_is_score_only_and_does_not_claim_solver_evidence() -> None:
    rows = [
        {
            "frame": 0,
            "final_eval_stage9_total": 10.0,
            "final_eval_stage9_slack_reg": 2.0,
            "final_eval_stage9_temporal_reg": 1.0,
            "final_eval_stage9_base_reg": 3.0,
        },
        {
            "frame": 2,
            "final_eval_stage9_total": 8.0,
            "final_eval_stage9_slack_reg": 1.0,
            "final_eval_stage9_temporal_reg": 0.5,
            "final_eval_stage9_base_reg": 2.0,
        },
    ]

    assert _shadow_frame_selection("auto", [0, 1, 2]) == [0, 1, 2]
    report = _shadow_score_diagnostic(rows, [0, 2], "0,2")

    assert report["ran"] is True
    assert report["solver_invocation_count"] == 0
    assert report["ablation_type"] == "score_only_counterfactual"
    assert all(item["score_only"] for item in report["variants"])


def test_html_contains_layer_and_link_controls(tmp_path: Path) -> None:
    payload = {
        "worst_frame": 0,
        "link_options": ["all", "index1z"],
        "frames": [
            {
                "metrics": {"frame": 0, "global_frame": 240, "final_visual_min_m": 0.0},
                "object_points": [[0.0, 0.0, 0.0]],
                "source_points": [[0.0, 0.0, 0.0]],
                "source_regions": ["index"],
                "source_links": ["index"],
                "warm_points": [],
                "final_points": [],
                "collision_points": [],
                "query_points": [],
                "anchor_points": [],
                "segments": [],
            }
        ],
    }
    destination = tmp_path / "audit.html"

    _html(payload, destination)

    text = destination.read_text()
    assert "visual surface" in text
    assert "collision geometry" in text
    assert "full 512 audit" in text
    assert "timelineMetric" in text
    assert "formalViewer" in text
    assert "index1z" in text
    assert 'name+"_links"' in text
