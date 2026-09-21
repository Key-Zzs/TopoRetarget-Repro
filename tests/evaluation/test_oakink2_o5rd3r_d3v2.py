from __future__ import annotations

import inspect

from scripts.evaluation import run_oakink2_o5rd3r_d3v2 as study


def test_localization_cli_is_optimizer_free() -> None:
    actions = {
        "preflight",
        "verify-d3-v1-history",
        "verify-frozen-authorities",
        "partition-d3-semantic-tail",
        "analyze-tail-excess",
        "analyze-failure-segments",
        "analyze-candidate-availability",
        "analyze-contributor-locality",
        "analyze-temporal-context",
        "audit-polish-retention",
        "decide-d3r-root-cause",
        "freeze-targeted-repair-plan",
    }
    assert actions <= set(study.ACTIONS)
    source = "\n".join(inspect.getsource(study.ACTIONS[name]) for name in actions)
    assert "search_frame(" not in source
    assert "_run_refinement" not in source


def test_partition_is_exhaustive_and_threshold_is_frozen() -> None:
    assert study.TAU == 1.0e-4
    assert study._group(study.TAU + 1e-8, study.TAU) == "A"
    assert study._group(study.TAU + 1e-8, study.TAU + 1e-8) == "B"
    assert study._group(study.TAU, study.TAU) == "C"
    assert study._group(study.TAU, study.TAU + 1e-8) == "D"


def test_repair_plan_is_single_candidate_and_preserves_scientific_authorities() -> None:
    source = inspect.getsource(study.freeze_targeted_repair_plan)
    assert "REFINEMENT_V2_A_SEQUENTIAL_MULTI_START" in source
    assert '"candidate_count": 1' in source
    assert '"ObjectiveV2_sha256"' in source
    assert '"SemanticV1_sha256"' in source
    assert '"E_IM_threshold"' in source
    assert '"cross_episode_refinement_required": True' in source


def test_historical_artifacts_are_read_only_inputs() -> None:
    source = inspect.getsource(study)
    assert 'D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO' in source
    assert 'D3_V1_TRAJECTORY_MODIFIED": "NO' in source
    assert 'D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0' in source


def test_development_is_gated_and_uses_existing_s2_only() -> None:
    source = inspect.getsource(study.run_targeted_repair_development)
    assert "targeted_refinement_repair_plan.json" in source
    assert "default_search_contracts()[1]" not in source
    frame_source = inspect.getsource(study._run_development_frame)
    assert "default_search_contracts()[1]" in frame_source
    assert "previous_q=previous_q" in frame_source
    assert "previous_base=previous_base" in frame_source
    assert "D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD" in frame_source


def test_scientific_payload_change_requires_fresh_cross_episode_certification() -> None:
    source = inspect.getsource(study.audit_repair_impact)
    assert "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED" in source
    assert '"fresh_certification_required": True' in source
    assert '"cross_episode_refinement_required": True' in source
    assert '"RETARGET_OBJECTIVE_V2_CHANGED": "NO"' in source


def test_full_contract_cli_is_exposed_and_downstream_is_fail_closed() -> None:
    required = {
        "preflight",
        "verify-d3-v1-history",
        "partition-d3-semantic-tail",
        "analyze-tail-excess",
        "analyze-failure-segments",
        "analyze-candidate-availability",
        "analyze-contributor-locality",
        "analyze-temporal-context",
        "audit-polish-retention",
        "decide-d3r-root-cause",
        "freeze-targeted-repair-plan",
        "run-targeted-repair-development",
        "select-refinement-v2",
        "audit-repair-impact",
        "audit-fresh-refinement-pool",
        "freeze-fresh-refinement-sparse",
        "run-fresh-refinement-sparse",
        "freeze-fresh-refinement-window",
        "run-fresh-refinement-window",
        "run-cross-episode-refinement-if-required",
        "freeze-refinement-v2",
        "freeze-d3v2-run",
        "run-d3v2-full",
        "resume-d3v2-full",
        "compare-old-v1-v2",
        "run-d3v2-semantic-v1",
        "render-d3v2-viewer",
        "summarize",
    }
    assert required <= set(study.ACTIONS)
    for name in required - {
        "preflight",
        "verify-d3-v1-history",
        "partition-d3-semantic-tail",
        "analyze-tail-excess",
        "analyze-failure-segments",
        "analyze-candidate-availability",
        "analyze-contributor-locality",
        "analyze-temporal-context",
        "audit-polish-retention",
        "decide-d3r-root-cause",
        "freeze-targeted-repair-plan",
        "run-targeted-repair-development",
        "select-refinement-v2",
        "audit-repair-impact",
        "summarize",
    }:
        assert "_require_development_pass" in inspect.getsource(study.ACTIONS[name])


def test_summary_preserves_hard_stop_and_human_review_boundary() -> None:
    source = inspect.getsource(study.summarize)
    assert "BLOCKED_REPAIR_DEVELOPMENT" in source
    assert '"D3_V2_SCIENTIFIC_RUN_COUNT": 0' in source
    assert '"DEV1_D3_V2_HUMAN_GEOMETRIC_REVIEW": "NOT_OPEN"' in source
    assert '"O5_FINAL": "NOT_PASS"' in source
