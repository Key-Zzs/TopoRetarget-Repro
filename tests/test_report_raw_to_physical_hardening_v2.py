from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.report_raw_to_physical_hardening_v2 import (
    EPISODES,
    MAIN_COLUMNS,
    TIMING_PHASES,
    aggregate,
)


def _result(episode: str) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "HardeningV2P5EpisodeResultV1",
        "dataset_role": "PIPELINE_HARDENING_SET_V1",
        "held_out": False,
        "episode": episode,
        "input_quality": "PASS",
        "retarget": "PASS",
        "source_controller": "FAIL",
        "l0_updates": 25,
        "support": "NOT_REACHED",
        "frozen_pf_v2": "NOT_REACHED",
        "ppo_updates": 0,
        "pf_pick": "NOT_REACHED",
        "pf_transport": "NOT_REACHED",
        "pf_place": "NOT_REACHED",
        "pf_release": "NOT_REACHED",
        "pf_retreat": "NOT_REACHED",
        "pf_full": "FAIL",
        "df_pose": "NOT_REACHED",
        "df_linear": "NOT_REACHED",
        "df_angular": "NOT_REACHED",
        "status": "SOURCE_CONTROLLER_FAILED",
        "timing_seconds": dict.fromkeys(TIMING_PHASES),
        "replay_commands": [],
    }
    assert set(MAIN_COLUMNS).issubset(row)
    return row


def _write_results(root: Path) -> None:
    for episode in EPISODES:
        path = root / "p5_hardening_regression" / "per_episode" / episode / "final_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_result(episode)), encoding="utf-8")


def test_aggregate_requires_all_five_and_never_reports_held_out_rate(tmp_path: Path) -> None:
    _write_results(tmp_path)
    summary = aggregate(tmp_path)
    assert summary["held_out_benchmark"] is False
    assert summary["held_out_success_rate_computed"] is False
    with (tmp_path / "p5_hardening_regression/main_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [row["episode"] for row in rows] == list(EPISODES)


def test_aggregate_fails_closed_on_missing_terminal_receipt(tmp_path: Path) -> None:
    _write_results(tmp_path)
    missing = (
        tmp_path / "p5_hardening_regression" / "per_episode" / EPISODES[-1] / "final_status.json"
    )
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="P5_TERMINAL_RECEIPT_MISSING"):
        aggregate(tmp_path)


def test_aggregate_rejects_held_out_relabeling(tmp_path: Path) -> None:
    _write_results(tmp_path)
    path = tmp_path / "p5_hardening_regression" / "per_episode" / EPISODES[0] / "final_status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["held_out"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="P5_EPISODE_RESULT_INVALID"):
        aggregate(tmp_path)
