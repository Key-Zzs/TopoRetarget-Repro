from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.data import run_oakink2_o5rd2g3 as workflow
from toporetarget.utils.hashing import sha256_file


def test_frozen_scientific_authorities_are_unchanged() -> None:
    paths = workflow.d2g.frozen_paths()
    assert sha256_file(paths["objective_v2"]) == workflow.OBJECTIVE_SHA
    assert sha256_file(paths["execution_v2"]) == workflow.EXECUTION_V2_SHA
    assert sha256_file(paths["gate_v2"]) == workflow.GATE_V2_SHA


def test_refinement_isolation_and_seed_delegation() -> None:
    source = inspect.getsource(workflow.d2g._run_refinement_v3)
    assert "search_frame(" in source
    assert "default_search_contracts()[0]" in source
    assert "search_cold_start_v2_frame" not in source
    assert "_whole_hand_geometric_bootstrap" not in source
    contract = workflow.default_search_contracts()[0]
    assert contract.name == "S1_DETERMINISTIC_MULTI_START"
    assert contract.seed_sources == (
        "old_production",
        "wuji_canonical_rest",
        "joint_range_midpoint",
    )


def test_post_bootstrap_ranking_is_coldstart_only() -> None:
    coldstart = inspect.getsource(workflow.d2g2.search_cold_start_v2_frame)
    refinement = inspect.getsource(workflow.d2g._run_refinement_v3)
    assert "bootstrap" in coldstart
    assert "post_bootstrap" in coldstart
    assert "bootstrap" not in refinement
    assert "post_bootstrap" not in refinement


def test_authoritative_sampled_predecessor_uses_frozen_s1_state() -> None:
    runtime = workflow.d2g.D2ARuntime(workflow.ROOT)
    window = workflow._windows_by_id()["high_around_consumed_median_anchor"]
    first_q, first_base, first_authority = workflow._authoritative_previous_state(
        runtime, window, 2355
    )
    assert first_authority == "OLD_PRODUCTION_WINDOW_PREDECESSOR"
    np.testing.assert_array_equal(first_q, runtime.final.arrays["qpos"][2354])
    np.testing.assert_array_equal(first_base, runtime.final.arrays["base_pose_scene"][2354])

    sampled_q, sampled_base, sampled_authority = workflow._authoritative_previous_state(
        runtime, window, 2359
    )
    assert sampled_authority == "FROZEN_S1_PREVIOUS_ACCEPTED_STATE"
    expected_path, _ = workflow._historical_paths("high_around_consumed_median_anchor", 2358)
    expected_q, expected_base = workflow._load_state(expected_path)
    np.testing.assert_array_equal(sampled_q, expected_q)
    np.testing.assert_array_equal(sampled_base, expected_base)
    assert not np.array_equal(sampled_q, runtime.final.arrays["qpos"][2358])


def test_metadata_comparator_normalizes_tuple_and_json_list() -> None:
    observed = {
        "selected_block": "thumb",
        "contributor_ranking": ("thumb", "index"),
        "seed_pool": ("old_production", "wuji_canonical_rest"),
        "retained_primary_id": "probe:old_production",
        "retention_decision": "PRIMARY_RETAINED_LEXICOGRAPHIC_REGRESSION",
        "baseline_fallback": False,
        "selected_candidate": "probe:old_production",
    }
    expected = {
        **observed,
        "contributor_ranking": ["thumb", "index"],
        "seed_pool": ["old_production", "wuji_canonical_rest"],
    }
    assert all(workflow._metadata_parity(observed, expected).values())


def test_coldstart_has_no_qold_or_dev2_special_case() -> None:
    source = inspect.getsource(workflow.d2g2.search_cold_start_v2_frame)
    assert "old_production_q=None" in source
    assert "runtime.final" not in source
    for forbidden in ("scene_01__A003", "C11001", "10704", '"thumb"'):
        assert forbidden not in source


def test_fail_closed_cli_ordering(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        workflow.repair_refinement_parity(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.run_refinement_regression(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.lock_cs2_a(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.run_dev2_frame0_development(tmp_path)
    with pytest.raises(FileNotFoundError):
        workflow.freeze_execution_v3(tmp_path)


def test_cli_has_all_contract_actions_and_help_executes() -> None:
    required = {
        "preflight",
        "verify-frozen-upstream",
        "audit-refinement-parity-contract",
        "localize-refinement-divergence",
        "repair-refinement-parity",
        "run-refinement-regression",
        "audit-coldstart-payload-impact",
        "verify-d2g2-coldstart-evidence",
        "rerun-coldstart-development-if-required",
        "lock-cs2-a",
        "run-dev2-frame0-development",
        "audit-dev2-special-cases",
        "freeze-execution-input-authority",
        "freeze-source-interaction-graph-authority",
        "freeze-coldstart-bootstrap-contract",
        "freeze-coldstart-seed-authority-v2",
        "freeze-execution-v3",
        "generate-future-certification-plan",
        "summarize",
    }
    assert required <= set(workflow.ACTIONS)
    script = Path(workflow.__file__)
    for command in ([sys.executable, str(script), "--help"],) + tuple(
        [sys.executable, str(script), action, "--help"] for action in required
    ):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.startswith("usage:")


def test_no_forbidden_downstream_action_exists() -> None:
    forbidden = (
        "sparse-v4",
        "window-v4",
        "fresh-cross-episode",
        "dev2-full",
        "dev1-full",
        "physx",
        "ppo",
        "support",
    )
    assert not any(any(token in action for token in forbidden) for action in workflow.ACTIONS)
