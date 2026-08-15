"""Checks for the Isaac-free C4 comparison and replay finalizer."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


def _module() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/evaluation/finalize_stage16_causal_physical_c4.py"
    )
    spec = importlib.util.spec_from_file_location("stage16_c4_formal20_finalize", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualification() -> dict[str, object]:
    return {
        "contact_mode": "aggregate_v3",
        "evaluation_suite_v2": {
            "aggregate": {
                "E_r_mean_deg": {"mean": 1.0},
                "E_t_mean_cm": {"mean": 2.0},
                "E_j_mean_cm": {"mean": 3.0},
                "E_ft_mean_cm": {"mean": 4.0},
                "kinematic_success": {"rate": 0.5},
                "physics_success": {"rate": 0.6},
                "qualified_success": {"rate": 0.4},
            }
        },
        "interaction": {
            "aggregate": {
                "source_tip_recall": 0.1,
                "source_persistent_tip_recall": 0.2,
                "cross_finger_compensation": 0.3,
                "persistent_cross_finger_compensation": 0.4,
                "fully_missing_source_contact": 0.5,
            }
        },
        "flight": {"no_hand_object_contact_fraction": 0.6, "longest_flight_gap": 7},
        "twist": {
            "Delta_v_mean_mps": 0.1,
            "Delta_v_p95_mps": 0.2,
            "Delta_v_terminal_mps": 0.3,
            "Delta_omega_mean_radps": 0.4,
            "Delta_omega_p95_radps": 0.5,
            "Delta_omega_terminal_radps": 0.6,
            "terminal_stability_rate": 0.7,
        },
        "penetration": {
            "aggregate": {
                "max_penetration_m": 0.001,
                "p95_penetration_m": 0.0005,
                "active_p95_penetration_m": 0.0004,
            },
            "interfinger_max_penetration_m": 0.0003,
            "absolute_geometry_pass": True,
            "hand_table": {"max_penetration_m": 0.0002, "absolute_geometry_pass": True},
        },
    }


def test_c4_row_has_every_required_wide_comparison_field() -> None:
    module = _module()
    row = module._c4_row(_qualification())
    assert tuple(row) == module.TABLE_COLUMNS
    assert row["Reward"] == "V3"
    assert row["H-O max mm"] == 1.0
    assert row["H-T max mm"] == 0.2


def test_selection_keeps_a_real_unsuccessful_episode_for_failure_replay() -> None:
    module = _module()
    selection = module._selection(
        [
            {"episode": 0, "qualified_success": True, "E_r_mean_deg": 1.0, "E_t_mean_cm": 1.0},
            {"episode": 1, "qualified_success": False, "E_r_mean_deg": 3.0, "E_t_mean_cm": 3.0},
            {"episode": 2, "qualified_success": False, "E_r_mean_deg": 2.0, "E_t_mean_cm": 2.0},
        ]
    )
    assert selection == {
        "representative_best": 0,
        "median_typical": 2,
        "representative_failure_or_worst": 1,
    }


def test_legacy_v4_row_only_populates_historical_comparable_contact_fields() -> None:
    module = _module()
    row = module._legacy_zero_row(
        evidence={
            "reward": "V4",
            "captured_metrics": {
                "source_tip_recall": 0.1,
                "persistent_source_tip_recall": 0.2,
                "no_hand_object_contact_flight_fraction": 0.3,
                "longest_no_hand_flight_gap": 4,
            },
        }
    )
    assert tuple(row) == module.TABLE_COLUMNS
    assert row["Tip recall"] == 0.1
    assert row["Persistent recall"] == 0.2
    assert row["No-hand fraction"] == 0.3
    assert row["Longest gap"] == 4
    assert row["Er°"] == "N/A"


def test_mode_conclusions_report_mixed_without_a_cross_category_scalar() -> None:
    module = _module()
    v3 = _qualification()
    v4 = copy.deepcopy(v3)
    v4["interaction"]["aggregate"]["source_tip_recall"] = 0.2
    v4["twist"]["Delta_v_mean_mps"] = 0.2
    conclusion = module._mode_conclusions(v3, v4)
    assert conclusion["interaction"]["verdict"] == "V4 better"
    assert conclusion["twist"]["verdict"] == "V3 better"
    assert set(conclusion) == {
        "interaction",
        "twist",
        "penetration",
        "evaluation_suite_v2",
    }
