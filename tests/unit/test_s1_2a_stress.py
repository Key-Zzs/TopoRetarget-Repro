from __future__ import annotations

import json
from pathlib import Path

import pytest

import toporetarget.workflows.s1_2a_stress as stress
from toporetarget.workflows.s1_2a_stress import (
    FAILURE_CLASSES,
    _comparison_delta,
    _failure_class,
    _rank_key,
    _strict_summary,
)


def test_s1_2a_failure_classification_isolated() -> None:
    assert _failure_class("qhull graph failed") == FAILURE_CLASSES["graph"]
    assert _failure_class("SLSQP solver status 9") == FAILURE_CLASSES["solver"]
    assert _failure_class("reference winding SDF failed") == FAILURE_CLASSES["sdf"]
    assert _failure_class("timeout exceeded wall time") == FAILURE_CLASSES["timeout"]


def test_e0_ranking_is_lexicographic_and_deterministic() -> None:
    base = {
        "sequence": "s2/a",
        "e0_metrics": {
            "frames_gt_1mm": 4,
            "mean_excess_penetration_m": 0.002,
            "max_penetration_m": 0.004,
            "active_link_count": 2,
        },
    }
    stronger = {
        **base,
        "sequence": "s1/z",
        "e0_metrics": {**base["e0_metrics"], "frames_gt_1mm": 5},
    }
    tie = {**base, "sequence": "s1/a"}
    assert _rank_key(stronger) < _rank_key(base)
    assert _rank_key(tie) < _rank_key(base)


def test_frozen_source_exclusions_are_not_result_based() -> None:
    assert stress.EXCLUDED == {
        "s1/airplane_lift",
        "s1/apple_eat_1",
        "s1/banana_lift",
        "s1/alarmclock_lift",
    }
    assert stress.PROFILE == "dense_squared_hinge_deadzone1mm_v2"
    assert stress.LAMBDA_SDF == pytest.approx(0.1)


def test_e0_failure_does_not_stop_probe_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = tmp_path / "experiment"
    selection = experiment / "selection"
    selection.mkdir(parents=True)
    rows = []
    for index in range(2):
        rows.append(
            {
                "candidate_id": f"W{index + 1:04d}",
                "warm_pass": True,
                "sequence": f"s1/clip_{index}",
                "subject": "s1",
                "object": "cup",
                "source_row": {
                    "sequence": f"s1/clip_{index}",
                    "start_frame": 0,
                    "end_frame": 60,
                    "source_file": f"clip_{index}.npz",
                },
            }
        )
    (selection / "warm_candidates.json").write_text(json.dumps({"rows": rows}))

    monkeypatch.setattr(stress, "_prepare_robot_surface", lambda *args: None)
    monkeypatch.setattr(
        stress,
        "_prepare_inputs",
        lambda _repo, _experiment, _cfg, _row, candidate_id, probe: {
            "root": experiment / "e0_probe" / candidate_id,
            "canonical": tmp_path / "canonical.zarr",
            "warm": tmp_path / "warm.npz",
            "graph": tmp_path / "graph.npz",
            "samples": tmp_path / "samples.npz",
        },
    )

    def fake_refine(_repo, _experiment, paths, _cfg, **_kwargs):
        if paths["root"].name == "W0001":
            raise RuntimeError("solver failed")
        return {
            "status": "pass",
            "strict_accepted": True,
            "metrics": {
                "frame_count": 3,
                "full_sample_count": 512,
                "strict_accepted_count": 3,
                "status_9_count": 0,
                "finite": True,
                "frames_gt_1mm": 1,
                "mean_excess_penetration_m": 1e-5,
                "max_penetration_m": 0.0011,
                "active_link_count": 1,
            },
        }

    monkeypatch.setattr(stress, "_refine", fake_refine)
    result = stress.run_e0_probes(Path.cwd(), stress.DEFAULT_CONFIG, experiment)
    assert result["count"] == 2
    assert result["probe_pass"] == 1
    assert result["failure_counts"]["SOLVER_FAILURE"] == 1


def test_strict_summary_requires_full_512_and_no_status9() -> None:
    metrics = {
        "frame_count": 60,
        "full_sample_count": 512,
        "strict_accepted_count": 60,
        "status_9_count": 0,
        "finite": True,
    }
    assert _strict_summary(metrics, 60)
    assert not _strict_summary({**metrics, "status_9_count": 1}, 60)
    assert not _strict_summary({**metrics, "full_sample_count": 32}, 60)


def test_comparison_is_observation_only_and_does_not_tune_clips() -> None:
    e0 = {"penetration_energy": 2.0, "max_penetration_m": 0.004}
    s1 = {"penetration_energy": 1.0, "max_penetration_m": 0.003}
    delta = _comparison_delta(e0, s1)
    assert delta["penetration_energy"] == {"e0": 2.0, "s1": 1.0, "delta": -1.0}
    assert delta["max_penetration_m"]["delta"] == pytest.approx(-0.001)
