from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.data import run_oakink2_o5rd2g2 as workflow
from toporetarget.utils.hashing import sha256_file


def test_frozen_upstream_hashes_are_unchanged() -> None:
    paths = workflow.d2g.frozen_paths()
    assert sha256_file(paths["objective_v2"]) == workflow.OBJECTIVE_SHA
    assert sha256_file(paths["execution_v2"]) == workflow.EXECUTION_V2_SHA
    assert sha256_file(paths["gate_v2"]) == workflow.GATE_V2_SHA


def test_search_v2_has_no_qold_or_episode_lookup() -> None:
    source = inspect.getsource(workflow.search_cold_start_v2_frame)
    assert "runtime.final" not in source
    assert "old_production_q=None" in source
    for forbidden in ("C11001", "10704", "scene_01__A003", '"thumb"'):
        assert forbidden not in source
    assert all(value == "NO" for value in workflow._special_case_audit().values())


def test_candidate_preregistration_is_outcome_independent(tmp_path: Path) -> None:
    workflow._write_json(tmp_path / "failure_localization/root_cause.json", {"status": "PASS"})
    workflow._write_json(
        tmp_path / "failure_localization/source_geometric_multistart_contract_audit.json",
        {"status": "PASS"},
    )
    result = workflow.freeze_search_v2_candidates(tmp_path)
    assert result["frozen_before_development"]
    assert result["N_COLDSTART_SEARCH_V2_CANDIDATES"] == 1
    assert workflow._read_json(tmp_path / "search_v2_candidates/cs2_a.json")[
        "predeclared_before_outcomes"
    ]
    assert (
        workflow._read_json(tmp_path / "search_v2_candidates/cs2_b.json")["status"]
        == "NOT_IMPLEMENTED_NOT_AUTHORIZED"
    )
    assert (
        workflow._read_json(tmp_path / "search_v2_candidates/cs2_c.json")["status"]
        == "NOT_IMPLEMENTED_NOT_AUTHORIZED"
    )


def test_fail_closed_cli_ordering(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        workflow.run_cs2_a_development(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.select_coldstart_search_v2(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.run_refinement_regression(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.lock_selected_candidate(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.run_dev2_frame0_development(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.freeze_execution_v3(tmp_path)


def test_cli_has_every_contract_action_and_help_executes() -> None:
    required = {
        "preflight",
        "verify-frozen-upstream",
        "localize-coldstart-failures",
        "audit-source-geometric-multistart",
        "freeze-search-v2-candidates",
        "run-cs2-a-development",
        "run-cs2-b-development",
        "run-cs2-c-development",
        "select-coldstart-search-v2",
        "run-refinement-regression",
        "lock-selected-candidate",
        "run-dev2-frame0-development",
        "freeze-execution-input-authority",
        "freeze-source-interaction-graph-authority",
        "freeze-coldstart-bootstrap-contract",
        "freeze-coldstart-seed-authority-v2",
        "freeze-execution-v3",
        "generate-future-certification-plan",
        "summarize",
        "completion-audit",
    }
    assert required <= set(workflow.ACTIONS)
    script = Path(workflow.__file__)
    commands = ([sys.executable, str(script), "--help"],) + tuple(
        [sys.executable, str(script), action, "--help"] for action in required
    )
    for command in commands:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith("usage:")


def test_no_forbidden_downstream_action_exists() -> None:
    forbidden = (
        "sparse-v4",
        "window-v4",
        "cross-episode-control",
        "dev2-full",
        "dev1-full",
        "physx",
        "ppo",
        "support",
    )
    assert not any(any(token in action for token in forbidden) for action in workflow.ACTIONS)


def test_bootstrap_profiler_records_unavailable_internal_timing_without_inference() -> None:
    fields = workflow._bootstrap_profiler_fields(
        {
            "seed_count": 2,
            "solve_count": 2,
            "nfev": 101,
            "njev": 99,
            "wall_sec": 12.5,
            "residual_evals": 101,
        }
    )
    assert fields["bootstrap_fk_calls"] is None
    assert fields["bootstrap_fk_time_sec"] is None
    assert fields["bootstrap_residual_evals"] == 101
    assert fields["bootstrap_residual_eval_time_sec"] is None
    assert (
        fields["bootstrap_internal_timing_status"]
        == "NOT_EMITTED_BY_EXISTING_GEOMETRIC_SOLVER_PRIMITIVE"
    )


def test_locked_candidate_cannot_be_switched_after_dev2_failure(tmp_path: Path) -> None:
    workflow._write_json(
        tmp_path / "development/selected_candidate_lock.json",
        {
            "status": "LOCKED_BEFORE_DEV2",
            "candidate_name": "CS2_C_GENERIC_MULTIBLOCK_INTERACTION_RESCUE",
        },
    )
    (tmp_path / "development/selected_candidate_lock.sha256").write_text(
        sha256_file(tmp_path / "development/selected_candidate_lock.json") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="LOCK_DRIFT"):
        workflow.run_dev2_frame0_development(tmp_path)
