from __future__ import annotations

import inspect

import numpy as np

from scripts.evaluation import run_oakink2_o5rd3r2 as study


def test_cli_stops_at_d3r3_authorization() -> None:
    required = {
        "preflight",
        "verify-d3r-authority",
        "build-failure-cluster-manifest",
        "freeze-representative-development-set",
        "freeze-search-space-oracle-plan",
        "replay-baseline-sentinels",
        "run-oracle-source-conditioned-seed",
        "run-oracle-expanded-active-set",
        "run-oracle-envelope-expansion",
        "run-compositional-oracle-if-authorized",
        "compare-search-space-oracles",
        "run-sequential-hysteresis-diagnostic",
        "decide-search-space-mechanism",
        "design-refinement-v2",
        "run-refinement-v2-design-sentinels",
        "freeze-refinement-v2-design",
        "freeze-d3r3-development-gate",
        "authorize-d3r3",
        "summarize",
    }
    assert required <= set(study.ACTIONS)
    forbidden = {
        "run-d3r3",
        "run-fresh-refinement-sparse",
        "run-fresh-refinement-window",
        "run-cross-episode-refinement",
        "run-d3v2-full",
        "run-dev2",
        "run-ppo",
        "run-o6",
    }
    assert not (forbidden & set(study.ACTIONS))


def test_scientific_authorities_are_frozen() -> None:
    assert study.TAU == 1.0e-4
    assert study.OBJECTIVE_V2_SHA256 == study.d3r.OBJECTIVE_V2_SHA256
    assert study.SEMANTIC_V1_SHA256 == study.d3r.SEMANTIC_V1_SHA256
    source = inspect.getsource(study.freeze_search_space_oracle_plan)
    assert '"hard_validity": "UNCHANGED"' in source
    assert '"q_old": "historical q_old[t] unchanged"' in source
    assert '"RETARGET_MODE": "REFINEMENT"' in source


def test_representative_selection_is_outcome_independent_and_distinct() -> None:
    partition = [
        {"source_frame": str(index + 100), "new_E_IM": str(1.0e-4 + index * 1.0e-7)}
        for index in range(20)
    ]
    segment = {"start_ordinal": 4, "stop_ordinal": 7}
    selected = study._four_distinct(segment, partition)
    assert len(selected) == 4
    assert len({row["ordinal"] for row in selected}) == 4
    assert {row["ordinal"] for row in selected} == {4, 5, 6, 7}


def test_oracle_count_and_composition_are_bounded() -> None:
    assert study.PRIMARY_ORACLES == (
        "B_SOURCE_CONDITIONED_ALTERNATE_SEED",
        "C_EXPANDED_ACTIVE_SET",
        "D_BOUNDED_SEARCH_ENVELOPE_EXPANSION",
    )
    source = inspect.getsource(study.freeze_search_space_oracle_plan)
    assert '"max_compositional_oracles": 1' in source
    assert '"dynamic_escalation": False' in source
    composition = inspect.getsource(study.run_compositional_oracle_if_authorized)
    assert "FROZEN_AMBIGUITY_RULE_NOT_SATISFIED" in composition
    assert "authorized = bool" in composition


def test_source_seed_oracle_remains_refinement() -> None:
    source = inspect.getsource(study._oracle_b_frame)
    assert '"RETARGET_MODE": "REFINEMENT"' in source
    assert '"q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED"' in source
    assert "search_cold_start_v2_frame" not in source


def test_expanded_active_set_does_not_release_wrist_or_base() -> None:
    source = inspect.getsource(study._oracle_c_frame)
    assert '"wrist_base_search_expansion": "NO"' in source
    assert "block = tuple(range(len(old_q)))" in source
    phase = inspect.getsource(study._run_primary)
    assert "block=block" in phase


def test_envelope_oracle_uses_domain_equivalence_not_budget_sweep() -> None:
    source = inspect.getsource(study.run_oracle_envelope_expansion)
    assert "domain_equivalent_to_r0" in source
    assert "STORED_R0_REUSED_BY_PREREGISTERED_DOMAIN_EQUIVALENCE_PROOF" in source
    assert "maxiter" not in source
    assert "max_nfev" not in source


def test_sequential_windows_keep_qold_separate_from_previous_accepted() -> None:
    source = inspect.getsource(study.run_sequential_hysteresis_diagnostic)
    assert '"q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED"' in source
    assert '"previous_accepted_authority": "CURRENT_ORACLE_TRAJECTORY_T_MINUS_1"' in source
    assert "previous_q, previous_base = q, base" in source


def test_zero_recovery_cannot_freeze_design() -> None:
    sentinel = inspect.getsource(study.run_refinement_v2_design_sentinels)
    freeze = inspect.getsource(study.freeze_refinement_v2_design)
    assert "recovered > 0" in sentinel
    assert 'require(root / "sentinels/decision.json", "status", "PASS"' in freeze


def test_no_dataset_or_frame_special_case_in_design() -> None:
    source = inspect.getsource(study.design_refinement_v2)
    assert '"frame_episode_object_special_cases": False' in source
    assert "C10001" not in source
    assert "source_frame" not in source


def test_frozen_hash_is_canonical_and_repeatable() -> None:
    value = {"b": [2, 1], "a": {"x": np.float64(1.0)}}
    first = study.canonical_sha(value)
    second = study.canonical_sha({"a": {"x": 1.0}, "b": [2, 1]})
    assert first == second


def test_summary_keeps_forbidden_runs_zero() -> None:
    source = inspect.getsource(study.summarize)
    for token in (
        '"D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT": 0',
        '"FRESH_REFINEMENT_SPARSE": "NOT_RUN"',
        '"FRESH_REFINEMENT_WINDOW": "NOT_RUN"',
        '"CROSS_EPISODE_REFINEMENT": "NOT_RUN"',
        '"D3_V2_SCIENTIFIC_RUN_COUNT": 0',
        '"DEV2_RERUN": "NO"',
        '"PPO_TRAINING_RUN_COUNT_NEW": 0',
        '"O6_PRODUCTION_RAN": "NO"',
    ):
        assert token in source
