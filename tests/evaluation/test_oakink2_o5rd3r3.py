from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluation import run_oakink2_o5rd3r3 as r3


def test_frozen_r2_design_and_gate_hashes_are_exact() -> None:
    design = r3.R2_ROOT / "design/refinement_v2_design.json"
    gate = r3.R2_ROOT / "design/refinement_v2_development_gate.json"
    assert r3.sha256_file(design) == r3.DESIGN_SHA256
    assert r3.sha256_file(gate) == r3.GATE_SHA256
    assert r3._sidecar_value(design.with_suffix(".sha256")) == r3.DESIGN_SHA256
    assert r3._sidecar_value(gate.with_suffix(".sha256")) == r3.GATE_SHA256


def test_failure_frame_identity_is_exact_and_frozen() -> None:
    rows = r3._failure_rows()
    assert len(rows) == 145
    assert len({int(row["ordinal"]) for row in rows}) == 145
    assert len({int(row["source_frame"]) for row in rows}) == 145
    assert {int(row["cluster_id"]) for row in rows} == {1, 2, 3, 4}
    assert all(row["group"] == "GROUP_B" for row in rows)
    assert [row["source_frame"] for row in rows[:2]] == [3919, 3920]
    assert rows[-1]["source_frame"] == 4213


def test_recovery_count_rounds_up_for_frozen_population() -> None:
    assert math.ceil(0.8 * 145) == 116
    assert 145 - math.ceil(0.8 * 145) == 29


def test_action_surface_has_no_fresh_or_downstream_execution() -> None:
    actions = set(r3.ACTIONS)
    assert "run-fresh-refinement-sparse" not in actions
    assert "run-fresh-refinement-window" not in actions
    assert "run-d3v2" not in actions
    assert "run-dev2" not in actions
    assert "run-ppo" not in actions
    assert "run-physx" not in actions
    assert "run-o6" not in actions


def test_normal_path_short_circuits_expanded_path(monkeypatch: pytest.MonkeyPatch) -> None:
    q = np.arange(20, dtype=np.float64)
    base = np.eye(4)
    normal = {
        "hard_valid": True,
        "interaction_valid": True,
        "E_IM": 9.0e-5,
    }
    monkeypatch.setattr(r3, "_normal_receipt", lambda *_args: (q, base, normal))

    def forbidden(*_args: object) -> None:
        raise AssertionError("expanded path must not execute")

    monkeypatch.setattr(r3.r2, "_oracle_c_frame", forbidden)
    selected_q, selected_base, receipt = r3._conditional_frame(object(), object(), 10, q, base)
    assert np.array_equal(selected_q, q)
    assert np.array_equal(selected_base, base)
    assert receipt["selected_path"] == "NORMAL_PATH"
    assert receipt["expanded"]["triggered"] is False


def test_frozen_trigger_invokes_exact_twenty_dof_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = np.zeros(20, dtype=np.float64)
    base = np.eye(4)
    normal = {
        "hard_valid": True,
        "interaction_valid": False,
        "E_IM": 1.2e-4,
    }
    monkeypatch.setattr(r3, "_normal_receipt", lambda *_args: (q, base, normal))
    expanded_q = np.ones(20, dtype=np.float64)
    expanded = {
        "active_dofs": list(range(20)),
        "selected_candidate": "expanded_primary:old_production",
        "selected_E_IM": 8.0e-5,
        "selected_hard_valid": True,
        "selected_interaction_valid": True,
        "primary_profile": {"status": 9, "message": "limit", "nfev": 12},
        "secondary_profile": {"status": 9, "message": "limit", "nfev": 19},
        "wall_time_sec": 2.0,
        "retention": "PRIMARY_RETAINED",
        "candidate_ids": ["d3_v1_baseline", "expanded_primary:old_production"],
    }
    monkeypatch.setattr(
        r3.r2,
        "_oracle_c_frame",
        lambda *_args: (expanded_q, base, expanded),
    )
    selected_q, _selected_base, receipt = r3._conditional_frame(object(), object(), 10, q, base)
    assert np.array_equal(selected_q, expanded_q)
    assert receipt["expanded"]["triggered"] is True
    assert receipt["expanded"]["active_dofs"] == list(range(20))
    assert receipt["expanded"]["nfev"] == 31
    assert receipt["selected_path"] == "EXPANDED_PATH"


def test_hard_invalid_terminal_does_not_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    q = np.zeros(20, dtype=np.float64)
    base = np.eye(4)
    normal = {
        "hard_valid": False,
        "interaction_valid": False,
        "E_IM": 1.2e-4,
    }
    monkeypatch.setattr(r3, "_normal_receipt", lambda *_args: (q, base, normal))
    monkeypatch.setattr(
        r3.r2,
        "_oracle_c_frame",
        lambda *_args: pytest.fail("frozen trigger requires a hard-valid normal terminal"),
    )
    _q, _base, receipt = r3._conditional_frame(object(), object(), 10, q, base)
    assert receipt["expanded"]["triggered"] is False
    assert receipt["final_hard_valid"] is False


def test_failure_summary_uses_preregistered_mathematical_early_stop(tmp_path: Path) -> None:
    rows = [
        {
            "cluster_id": 1 + index // 8,
            "recovery": False,
            "hard_valid": True,
            "expanded_triggered": True,
            "selected_path": "NORMAL_PATH_RETAINED_AFTER_EXPANDED_SCREENING",
            "D3_V1_E_IM": 1.2e-4,
            "final_E_IM": 1.1e-4,
            "relative_reduction": 1.0 / 12.0,
        }
        for index in range(30)
    ]
    result = r3._summarize_failures(tmp_path, rows, True)
    assert result["unrecovered"] == 30
    assert result["maximum_possible_final_recovered"] == 115
    assert result["maximum_possible_recovery_fraction"] == pytest.approx(115 / 145)
    assert result["RECOVERY_GATE_MATHEMATICALLY_IMPOSSIBLE"] == "YES"


def test_qold_and_runtime_state_authorities_are_separate() -> None:
    d3_manifest = r3.read_json(r3.D3_ROOT / "run_authority/run_manifest.json")
    assert d3_manifest["q_old_array_sha256"] != r3.sha256_file(
        r3.D3_ROOT / "trajectory/trajectory.npz"
    )
    design = r3.read_json(r3.R2_ROOT / "design/refinement_v2_design.json")
    assert "q_old[t] remains historical and separate" in design["previous_state_lifecycle"]


def test_expanded_dofs_are_asset_derived_and_wrist_base_locked() -> None:
    design = r3.read_json(r3.R2_ROOT / "design/refinement_v2_design.json")
    assert design["active_dofs"] == list(range(20))
    assert design["bounds_envelope"] == "Wuji asset joint limits; wrist/base remain fixed"


def test_authorization_fails_closed_on_development_failure(tmp_path: Path) -> None:
    r3.write_json(
        tmp_path / "gate/decision.json",
        {
            "D3_R3_STATUS": "FAIL_DEVELOPMENT_GATE",
            "REFINEMENT_V2_DEVELOPMENT_VALIDATION": "FAIL",
        },
    )
    with pytest.raises(RuntimeError, match="D3_R3_NOT_PASS"):
        r3.authorize_fresh_certification(tmp_path)
    assert (
        r3.read_json(tmp_path / "future/not_authorized.json")[
            "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED"
        ]
        == "NO"
    )


def test_authorization_only_emits_not_run_future_receipt_on_pass(tmp_path: Path) -> None:
    r3.write_json(
        tmp_path / "gate/decision.json",
        {
            "D3_R3_STATUS": "PASS",
            "REFINEMENT_V2_DEVELOPMENT_VALIDATION": "PASS",
        },
    )
    result = r3.authorize_fresh_certification(tmp_path)
    assert result["FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED"] == "YES"
    assert result["REFINEMENT_V2_INDEPENDENT_CERTIFICATION"] == "NOT_RUN"
    assert result["status"] == "AUTHORIZED_NOT_RUN"


def test_failure_row_uses_historical_qold_not_d3v1(monkeypatch: pytest.MonkeyPatch) -> None:
    q_old = np.zeros(20, dtype=np.float64)
    q_d3 = np.ones(20, dtype=np.float64)
    q_final = np.full(20, 2.0, dtype=np.float64)
    runtime = SimpleNamespace(
        final=SimpleNamespace(arrays={"qpos": np.stack([q_old])}),
        model=SimpleNamespace(
            dof_names=tuple(
                [f"thumb_{i}" for i in range(4)]
                + [f"index_{i}" for i in range(4)]
                + [f"middle_{i}" for i in range(4)]
                + [f"ring_{i}" for i in range(4)]
                + [f"pinky_{i}" for i in range(4)]
            )
        ),
    )
    monkeypatch.setattr(r3.r2, "_d3_state", lambda _ordinal: (q_d3, np.eye(4)))
    receipt = {
        "normal": {
            "active_dofs": [0, 1, 2, 3],
            "seed_provenance": "stored",
            "objective": "ObjectiveV2",
            "E_IM": 1.2e-4,
            "hard_valid": True,
            "solver_status": 9,
            "nfev": 10,
            "wall_time_sec": 1.0,
        },
        "expanded": {
            "triggered": True,
            "trigger_reason": "frozen",
            "active_dofs": list(range(20)),
            "E_IM": 8.0e-5,
            "hard_valid": True,
            "solver_status": 9,
            "nfev": 20,
            "wall_time_sec": 2.0,
        },
        "selected_path": "EXPANDED_PATH",
        "final_E_IM": 8.0e-5,
        "final_interaction_valid": True,
        "final_hard_valid": True,
        "wall_time_sec": 3.0,
    }
    row = r3._failure_row(
        {
            "ordinal": 0,
            "source_frame": 1,
            "cluster_id": 1,
            "D3_V1_E_IM": 1.2e-4,
            "old_historical_E_IM": 1.5e-4,
        },
        q_final,
        np.eye(4),
        receipt,
        runtime,
    )
    assert row["q_deviation_from_q_old_l2"] == pytest.approx(np.linalg.norm(q_final - q_old))
    assert row["q_deviation_from_D3V1_l2"] == pytest.approx(np.linalg.norm(q_final - q_d3))
    assert row["q_deviation_from_q_old_l2"] != row["q_deviation_from_D3V1_l2"]


def test_cli_help_is_real() -> None:
    parser = r3.parser()
    help_text = parser.format_help()
    for action in (
        "preflight",
        "verify-r2-design-authority",
        "verify-r3-development-gate",
        "freeze-r3-validation-manifest",
        "run-r3-failure-frame-validation",
        "resume-r3-failure-frame-validation",
        "run-r3-valid-controls",
        "run-r3-sequential-windows",
        "run-r3-determinism",
        "evaluate-r3-development-gate",
        "profile-r3-cost",
        "render-r3-development-viewer",
        "audit-r3-special-cases",
        "authorize-fresh-certification",
        "validate-repository",
        "summarize",
    ):
        assert action in help_text


def test_frozen_design_has_no_episode_or_frame_special_cases() -> None:
    design = r3.read_json(r3.R2_ROOT / "design/refinement_v2_design.json")
    assert design["frame_episode_object_special_cases"] is False
    assert design["seed_authority"] == [
        "old_production",
        "wuji_canonical_rest",
        "joint_range_midpoint",
    ]
