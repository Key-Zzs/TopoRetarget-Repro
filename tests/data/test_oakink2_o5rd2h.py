from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.data import run_oakink2_o5rd2h as d2h


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_plan_freezes_all_counts_and_fail_closed_order() -> None:
    plan = d2h.certification_plan()
    assert plan["sparse_v4"]["composition"] == {"HIGH": 10, "MID": 10, "LOW": 10}
    assert plan["window_v4"]["windows"] == ["HIGH_1", "HIGH_2", "MID", "LOW"]
    assert plan["cross_episode"]["N"] == 3
    assert plan["cross_episode"]["runs_per_control"] == 3
    assert plan["dev2_full"]["scientific_run_limit"] == 1
    assert plan["outcome_driven_mutation"] is False


def test_execution_v3_hashes_match_d2g3_frozen_authority() -> None:
    for name, path in d2h.FROZEN_CONTRACT_PATHS.items():
        assert d2h.sha256_file(path) == d2h.EXPECTED_HASHES[name]


def test_window_freeze_rejects_sparse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(d2h, "verify_frozen_execution_v3", lambda _root: {"status": "PASS"})
    _write(
        tmp_path / "sparse_v4/gate_decision.json",
        {"COLDSTART_SPARSE_VALIDATION_V4": "FAIL"},
    )
    with pytest.raises(RuntimeError, match="O5RD2H_STAGE_BLOCKED"):
        d2h.freeze_window_v4(tmp_path)


def test_cross_episode_freeze_rejects_window_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(d2h, "verify_frozen_execution_v3", lambda _root: {"status": "PASS"})
    _write(
        tmp_path / "window_v4/gate_decision.json",
        {"COLDSTART_WINDOW_VALIDATION_V4": "FAIL"},
    )
    with pytest.raises(RuntimeError, match="O5RD2H_STAGE_BLOCKED"):
        d2h.freeze_cross_episode_controls(tmp_path)


def test_dev2_authorization_rejects_cross_episode_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(d2h, "verify_frozen_execution_v3", lambda _root: {"status": "PASS"})
    _write(
        tmp_path / "sparse_v4/gate_decision.json",
        {"COLDSTART_SPARSE_VALIDATION_V4": "PASS"},
    )
    _write(
        tmp_path / "window_v4/gate_decision.json",
        {"COLDSTART_WINDOW_VALIDATION_V4": "PASS"},
    )
    _write(
        tmp_path / "cross_episode_controls/decision.json",
        {"FRESH_CROSS_EPISODE_CONTROLS": "FAIL"},
    )
    with pytest.raises(RuntimeError, match="O5RD2H_STAGE_BLOCKED"):
        d2h.authorize_dev2_full(tmp_path)


def test_cli_exposes_contract_actions() -> None:
    expected = {
        "preflight",
        "verify-frozen-execution-v3",
        "freeze-certification-plan",
        "build-exclusion-ledgers",
        "freeze-sparse-v4",
        "run-sparse-v4",
        "freeze-window-v4",
        "run-window-v4",
        "freeze-cross-episode-controls",
        "run-cross-episode-controls",
        "authorize-dev2-full",
        "run-dev2-full",
        "resume-dev2-full-technical-only",
        "run-dev2-semantic-v1",
        "render-dev2-viewer",
        "summarize",
    }
    assert expected <= d2h.ACTIONS.keys()
