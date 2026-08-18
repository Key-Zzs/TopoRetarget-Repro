from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from toporetarget.rl.fixed_wrist_causal_queue import (
    QUEUE_LINEAGES,
    all_four_c4_complete,
    atomic_write_json,
    completed_stage_count,
    initial_lineage_state,
    stage_sequence,
)


def test_queue_order_and_fail_closed_c0_sequence() -> None:
    assert [lineage["id"] for lineage in QUEUE_LINEAGES] == [
        "v3_hocap_170105",
        "v4_hocap_170105",
        "v3_hocap_170650",
        "v4_hocap_170650",
    ]
    assert stage_sequence(c0_reusable=False) == ("C0", "C1", "C2", "C3", "C4")
    assert stage_sequence(c0_reusable=True) == ("C1", "C2", "C3", "C4")


def test_state_requires_all_real_c4_endpoints() -> None:
    state = {lineage["id"]: initial_lineage_state(c0_reusable=False) for lineage in QUEUE_LINEAGES}
    assert completed_stage_count(state) == 0
    for lineage in state.values():
        lineage["c4"] = "COMPLETE"
    assert all_four_c4_complete(state)
    state["v3_hocap_170105"]["c4"] = "TECHNICALLY_INCOMPLETE"
    assert not all_four_c4_complete(state)


def test_atomic_json_write_never_leaves_a_partial_document(tmp_path) -> None:
    target = tmp_path / "queue_state.json"
    atomic_write_json(target, {"status": "RUNNING", "samples": 123})
    assert json.loads(target.read_text(encoding="utf-8")) == {"samples": 123, "status": "RUNNING"}


def test_monitor_is_read_only(tmp_path) -> None:
    run_dir = tmp_path / "run"
    state_path = run_dir / "queue_state.json"
    state = {
        "status": "RUNNING",
        "active_lineage": "v3_hocap_170105",
        "active_stage": "C0",
        "stage_samples_done": 1,
        "stage_samples_total": 10,
        "lineage_index": 1,
        "lineage_total": 4,
        "latest_checkpoint": None,
        "last_update_time": "2026-08-17T10:00:00+00:00",
        "technical_retries": {},
        "ALL_FOUR_C4_COMPLETE": "NO",
        "per_lineage": {
            lineage["id"]: initial_lineage_state(c0_reusable=False) for lineage in QUEUE_LINEAGES
        },
    }
    atomic_write_json(state_path, state)
    before = state_path.read_bytes()
    monitor = (
        Path(__file__).resolve().parents[2]
        / "scripts/rl/monitor_stage16_fixed_wrist_causal_queue.py"
    )
    completed = subprocess.run(
        [sys.executable, str(monitor), "--run-dir", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "QUEUE_STATUS=RUNNING" in completed.stdout
    assert "ALL_FOUR_C4_COMPLETE=NO" in completed.stdout
    assert state_path.read_bytes() == before


def test_recovered_queue_publishes_running_before_resuming() -> None:
    runner = (
        Path(__file__).resolve().parents[2] / "scripts/rl/run_stage16_fixed_wrist_causal_queue.py"
    )
    source = runner.read_text(encoding="utf-8")
    resume = source[source.index("state = _load_or_initialize_state") :]
    resume = resume[: resume.index("try:")]
    assert 'state["status"] = "RUNNING"' in resume
    assert 'state["process_alive"] = True' in resume
