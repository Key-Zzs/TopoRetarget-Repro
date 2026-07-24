"""Synthetic contracts for Stage 9.3.5 diagnostics."""

from html.parser import HTMLParser
from types import SimpleNamespace

import numpy as np

from toporetarget.workflows.stage9_3_5 import (
    _base_from_path,
    _bisect_boundary,
    _compare_official_artifact_snapshots,
    _finger_from_link,
    _html_path_traces,
    _html_state,
    _path_intervals,
    _pose_valid,
    _projection_objective,
    _write_html,
)


def test_path_intervals_keeps_non_monotonic_components() -> None:
    assert _path_intervals(np.asarray([False, True, True, False, True, False])) == [
        (0.2, 0.4),
        (0.8, 0.8),
    ]


def test_bisection_refines_transition() -> None:
    boundary = _bisect_boundary(lambda value: value >= 0.375, 0.25, 0.5, False)
    assert abs(boundary - 0.375) < 1e-8


def test_so3_path_has_geodesic_endpoints_and_valid_rotation() -> None:
    start = np.eye(4)
    end = np.eye(4)
    angle = np.pi / 2.0
    end[:3, :3] = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    end[:3, 3] = [0.2, -0.1, 0.3]
    assert np.allclose(_base_from_path(start, end, 0.0), start)
    assert np.allclose(_base_from_path(start, end, 1.0), end)
    midpoint = _base_from_path(start, end, 0.5)
    assert np.allclose(midpoint[:3, :3].T @ midpoint[:3, :3], np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(midpoint[:3, :3]), 1.0, atol=1e-10)
    assert np.allclose(midpoint[:3, 3], [0.1, -0.05, 0.15])


def test_projection_state_metric_is_warm_centered_and_unit_explicit() -> None:
    paper = SimpleNamespace(
        lambda_reg=2.5,
        lambda_base_pos=100.0,
        lambda_base_rot=1.0,
        w_s=100000.0,
    )
    bundle = SimpleNamespace(paper=paper, projection_warm_q=np.zeros(22))
    warm = np.zeros(28)
    total, gradient = _projection_objective(
        bundle, warm, "minimal_soft_safe_projection_from_warm_v2"
    )
    assert total == 0.0
    assert np.allclose(gradient, 0.0)
    moved = warm.copy()
    moved[0] = 0.01
    moved[6] = 0.2
    total, gradient = _projection_objective(
        bundle, moved, "minimal_soft_safe_projection_from_warm_v2"
    )
    assert np.isclose(total, 0.5 * 100.0 * 0.01**2 + 0.5 * 2.5 * 0.2**2)
    assert gradient[0] == 100.0 * 0.01
    assert gradient[6] == 2.5 * 0.2


def test_official_slack_penalty_and_link_mapping_are_diagnostic_only() -> None:
    paper = SimpleNamespace(
        lambda_reg=2.5, lambda_base_pos=100.0, lambda_base_rot=1.0, w_s=100000.0
    )
    bundle = SimpleNamespace(paper=paper, projection_warm_q=np.zeros(22))
    value = np.zeros(540)
    value[28] = 1e-3
    total, gradient = _projection_objective(bundle, value, "official_slack_projection_from_warm_v2")
    assert np.isclose(total, 0.5 * paper.w_s * 1e-6)
    assert gradient[28] == paper.w_s * 1e-3
    assert _finger_from_link("index_tip") == "index"
    assert _finger_from_link("palm") == "palm"


def test_html_headless_smoke_contains_required_interaction_surfaces(tmp_path) -> None:
    class IdParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.ids: set[str] = set()

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            del tag
            self.ids.update(value for key, value in attrs if key == "id" and value)

    html_path = tmp_path / "stage9_3_5.html"
    _write_html(
        html_path,
        {
            "frames": [0],
            "states": [{"frame": 0, "state": "warm", "label": "warm", "terms": {}}],
            "path": {"0": {"soft_safe_intervals": []}},
            "path_traces": {"0": []},
            "objective_endpoints": [],
            "objective_directional": [],
            "variable_group": [],
            "pressure": [],
            "pressure_aggregates": [],
            "projection_results": [],
            "readiness": {"ENTER_STAGE9_4": "NO"},
            "root_cause": {},
            "branch": {},
            "global_scale": {"rmse_m": 1, "pressure_score": 1, "objective": 1},
        },
    )
    parser = IdParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    required = {
        "frame",
        "view",
        "state",
        "alpha",
        "pathPlot",
        "fingerTimeline",
        "objectiveTable",
        "directionalTable",
        "variableTable",
        "pressureLink",
        "pressureFinger",
        "pressureTable",
        "projectionTable",
        "branch",
        "readiness",
        "motionSummary",
    }
    assert required <= parser.ids
    document = html_path.read_text(encoding="utf-8")
    assert "const SCALE=DATA.global_scale" in document
    assert "warm/projection/final" in document
    assert "StateFraction" in document


def test_pose_contract_rejects_corrupted_rotation() -> None:
    valid = np.eye(4)
    corrupted = valid.copy()
    corrupted[0, 0] = 2.0
    assert _pose_valid(valid)
    assert not _pose_valid(corrupted)


def test_official_artifact_snapshot_detects_hash_or_mtime_change() -> None:
    before = {
        "entries": [
            {
                "path": "/tmp/a",
                "label": "a",
                "exists": True,
                "sha256": "old",
                "mtime_ns": 1,
            }
        ]
    }
    after = {
        "entries": [
            {
                "path": "/tmp/a",
                "label": "a",
                "exists": True,
                "sha256": "new",
                "mtime_ns": 2,
            }
        ]
    }
    report = _compare_official_artifact_snapshots(before, after)
    assert report["official_artifacts_changed"]
    assert report["changed"][0]["unchanged"] is False


def test_cached_path_trace_preserves_units_and_finger_timeline(tmp_path) -> None:
    path = tmp_path / "path.csv"
    path.write_text(
        "frame,alpha,min_sdf_m,long_finger_rmse_m,index_rmse_m,hard_feasible,soft_safe_feasible,zero_penetration_feasible\n"
        "10,0.0,0.001,0.002,0.003,true,true,false\n",
        encoding="utf-8",
    )
    traces = _html_path_traces(path)
    assert traces["10"][0]["alpha"] == 0.0
    assert traces["10"][0]["index_rmse_m"] == 0.003
    assert traces["10"][0]["soft_safe_feasible"] is True


def test_html_state_is_compact_and_excludes_full512_residual_arrays() -> None:
    compact = _html_state(
        {
            "frame": 10,
            "label": "warm",
            "long_finger_rmse_m": 0.01,
            "full_signed_distance": np.zeros(512),
            "full_hard_residual": np.zeros(512),
        }
    )
    assert compact["label"] == "warm"
    assert "full_signed_distance" not in compact
    assert "full_hard_residual" not in compact
