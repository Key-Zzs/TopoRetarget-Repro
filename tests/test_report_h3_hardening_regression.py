from __future__ import annotations

import json
from pathlib import Path

from scripts.report_h3_hardening_regression import EPISODES, aggregate


def _row(episode: str, *, ready: bool = True) -> dict[str, object]:
    return {
        "schema_version": "H3HardeningRegressionEpisodeResultV1",
        "dataset_role": "PIPELINE_HARDENING_SET_V1",
        "held_out": False,
        "episode": episode,
        "retarget": "PASS",
        "source_route": "ZERO_RESIDUAL",
        "source_executable": "PASS",
        "source_fidelity": "DEGRADED",
        "support": "PASS",
        "frozen_pf": "0/10",
        "ppo_updates": 15,
        "pf_pick": "0/10",
        "pf_transport": "NOT_REACHED",
        "pf_place": "NOT_REACHED",
        "pf_release": "NOT_REACHED",
        "pf_retreat": "NOT_REACHED",
        "pf_full": "0/10",
        "df_pose": "0/10",
        "df_linear": "10/10",
        "df_angular": "10/10",
        "status": "PPO_BUDGET_EXHAUSTED",
        "method_contract_hash": "a" * 64,
        "timing_seconds": {"retarget": 1.0, "ppo": 2.0},
        "failure_taxonomy": [],
        "pipeline_readiness": {
            "exact_retarget_terminal": ready,
            "entered_frozen_full_gravity_evaluation": ready,
            "blocked_by_task_fidelity_only_gate": False,
            "unresolved_generic_technical_blocker": False,
            "initial_physical_failure": True,
            "ppo_terminal": True,
            "explicit_physical_invalid_state": False,
            "per_episode_tuning": False,
        },
    }


def _populate(root: Path, *, first_ready: bool = True) -> None:
    for index, episode in enumerate(EPISODES):
        path = root / "per_episode" / episode / "final_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_row(episode, ready=first_ready if index == 0 else True)),
            encoding="utf-8",
        )


def test_aggregate_marks_ready_only_when_all_gates_pass(tmp_path: Path) -> None:
    _populate(tmp_path)
    decision = aggregate(tmp_path)
    assert decision["pipeline_readiness"]["H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION"] == "YES"
    assert (tmp_path / "main_metrics.csv").is_file()
    assert (tmp_path / "failure_taxonomy.csv").is_file()


def test_aggregate_fails_readiness_without_cancelling_report(tmp_path: Path) -> None:
    _populate(tmp_path, first_ready=False)
    decision = aggregate(tmp_path)
    assert decision["pipeline_readiness"]["H3C_READY_FOR_UNSEEN_OBJECT_EXECUTION"] == "NO"
    assert decision["status"] == "COMPLETE"
