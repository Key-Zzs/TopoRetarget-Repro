from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd3certv3 as certv3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(function: object) -> str:
    return inspect.getsource(function)


def test_frozen_authority_hashes_are_exact() -> None:
    assert _sha(certv3.R2_ROOT / "design/refinement_v2_design.json") == certv3.DESIGN_SHA256
    assert _sha(certv3.R4_ROOT / "gate_v2/development_gate_v2.json") == certv3.GATE_V2_SHA256
    assert (
        _sha(certv3.CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
        == certv3.PROTOCOL_V2_SHA256
    )
    assert (
        _sha(certv3.CERT_QOLD_R_ROOT / "eligibility/qold_baseline_eligibility.json")
        == certv3.QOLD_ELIGIBILITY_SHA256
    )
    assert (
        _sha(certv3.CERT_QOLD_R_ROOT / "freshness/cert_v3_exclusion_ledger.json")
        == certv3.CERT_V3_EXCLUSION_LEDGER_SHA256
    )


def test_historical_attempts_and_qold_r_are_immutable() -> None:
    cert_r = certv3.read_json(certv3.CERT_R_ROOT / "final_summary.json")
    cert_v2 = certv3.read_json(certv3.CERT_V2_ROOT / "final_summary.json")
    qold_r = certv3.read_json(certv3.CERT_QOLD_R_ROOT / "final_summary.json")
    assert cert_r["D3_CERT_V1_HISTORICAL_RESULT"] == "FAIL"
    assert cert_r["D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN"] == "NO"
    assert cert_v2["D3_CERT_V2_STATUS"] == "BLOCKED_QOLD_BASELINE_GENERATION"
    assert qold_r["D3_CERT_V2_HISTORICAL_RESULT_REWRITTEN"] == "NO"
    assert qold_r["CERT_QOLD_R_STATUS"] == "PASS_QOLD_AUTHORITY_REPAIR"
    assert qold_r["REUSABLE_BASELINE_COUNT"] == 33


def test_partial_failed_baseline_is_excluded_without_special_branch() -> None:
    ledger = certv3.read_json(certv3.CERT_QOLD_R_ROOT / "freshness/cert_v3_exclusion_ledger.json")
    partial = certv3.read_json(certv3.CERT_QOLD_R_ROOT / "existing_qold/partial_records.json")
    assert len(partial["records"]) == 1
    assert partial["records"][0]["sequence_id"] in ledger["excluded_sequence_ids"]
    source = Path(certv3.__file__).read_text(encoding="utf-8")
    assert "5884" not in source
    assert "window_03" not in source


def test_cli_exposes_full_v3_contract_and_no_forbidden_action() -> None:
    required = {
        "preflight",
        "verify-cert-qold-r-authority",
        "verify-cert-v3-exclusion-ledger",
        "build-cert-v3-fresh-pool",
        "audit-reusable-qold",
        "evaluate-cert-v3-qold-sufficiency",
        "freeze-cert-v3-qold-augmentation-plan",
        "generate-cert-v3-qold-if-required",
        "freeze-cert-v3-qold-authority",
        "select-cert-v3-sparse",
        "select-cert-v3-windows",
        "select-cert-v3-cross-episode",
        "freeze-cert-v3-stage-manifests",
        "freeze-cert-v3-run-plan",
        "run-cert-v3-pre-scientific-audit",
        "freeze-cert-v3-code-head",
        "run-fresh-sparse-v3",
        "resume-fresh-sparse-v3",
        "run-fresh-sparse-v3-determinism",
        "evaluate-fresh-sparse-v3",
        "run-fresh-window-v3",
        "resume-fresh-window-v3",
        "run-fresh-window-v3-determinism",
        "evaluate-fresh-window-v3",
        "run-fresh-cross-episode-v3",
        "resume-fresh-cross-episode-v3",
        "run-fresh-cross-episode-v3-determinism",
        "evaluate-fresh-cross-episode-v3",
        "audit-cert-v3-postrun-integrity",
        "decide-cert-v3",
        "freeze-certified-refinement-v2-v3",
        "authorize-d3v2",
        "generate-d3v2-plan",
        "summarize",
    }
    assert required <= set(certv3.ACTIONS)
    assert certv3.FORBIDDEN_ACTION_NAMES.isdisjoint(certv3.ACTIONS)


def test_qold_sufficiency_is_evaluated_before_generation(tmp_path: Path) -> None:
    records = [
        {
            "status": "PASS",
            "role": "SPARSE_CANDIDATE",
            "frame_count": 40,
            "sequence_id": f"s{index:02d}",
        }
        for index in range(certv3.SPARSE_N)
    ]
    records.extend(
        {
            "status": "PASS",
            "role": "WINDOW_CANDIDATE",
            "frame_count": 64,
            "sequence_id": f"w{index:02d}",
        }
        for index in range(3)
    )
    certv3.write_json(tmp_path / "qold/reusable_audit.json", {"status": "PASS", "records": records})
    value = certv3.evaluate_cert_v3_qold_sufficiency(tmp_path)
    assert value["QOLD_POOL_SUFFICIENT_FOR_SPARSE"] == "YES"
    assert value["QOLD_POOL_SUFFICIENT_FOR_WINDOWS"] == "NO"
    assert value["QOLD_POOL_SUFFICIENT_FOR_CROSS_EPISODE"] == "NO"
    assert value["deficiency"]["window"] == 1
    assert value["deficiency"]["cross_episode"] == 3


def test_augmentation_is_frozen_and_structurally_preflighted_before_solver() -> None:
    block = _source(certv3.freeze_cert_v3_qold_augmentation_plan)
    generation = _source(certv3.generate_cert_v3_qold_if_required)
    assert "CertV3QoldAugmentationPlanV1" in block
    assert "STRUCTURAL_PREFLIGHT_BEFORE_QOLD_OPTIMIZER" in block
    assert "refinement_v2_outcome_used" in block
    assert "final_E_IM" not in block
    assert generation.index("augmentation_preflight.json") < generation.index(
        "cert._generate_one_baseline"
    )
    assert "_run_refinement_frame" not in generation


def test_valid_reusable_qold_is_materialized_not_recomputed() -> None:
    block = _source(certv3.generate_cert_v3_qold_if_required)
    assert "_materialize_reused_authority" in block
    assert 'item["reuse_source_authority_path"] is not None' in block
    assert block.index("_materialize_reused_authority") < block.index("cert._generate_one_baseline")


def test_qold_and_all_stage_identities_freeze_before_run_plan() -> None:
    sparse = _source(certv3.freeze_cert_v3_sparse_manifest)
    run_plan = _source(certv3.freeze_cert_v3_run_plan)
    assert "qold/qold_authority_manifest.json" in sparse
    assert "QOLD_NOT_FROZEN_BEFORE_TARGET_SELECTION" in sparse
    for path in (
        "selection/sparse_manifest.json",
        "selection/window_manifest.json",
        "selection/cross_episode_manifest.json",
    ):
        assert path in run_plan
    assert "ALL_STAGE_IDENTITIES_FROZEN_BEFORE_FIRST_FRESH_RUN" in run_plan


def test_pre_scientific_audit_never_fabricates_runtime_values() -> None:
    block = _source(certv3.run_cert_v3_pre_scientific_audit)
    assert "cert._run_refinement_frame" not in block
    assert '"runtime_values_fabricated": False' in block
    assert "RUNTIME_VALUE_NOT_YET_PRODUCED_EXPECTED" in block
    assert "REFINEMENT_V2_FRESH_RUN_COUNT_BEFORE_PREFLIGHT" in block


def test_prefix_execution_is_exact_and_context_precedes_optimizer() -> None:
    block = _source(certv3._execute_prefix_unit)
    assert "for ordinal in range(stop_ordinal + 1):" in block
    assert block.index("runtime.bind_context") < block.index("cert._run_refinement_frame")
    assert "PREVIOUS_ACCEPTED_REFINED_RUNTIME" in Path(certv3.__file__).read_text()
    assert "q_old_not_previous_refined" in block


def test_sparse_metrics_count_targets_only_and_median_is_diagnostic() -> None:
    block = _source(certv3.evaluate_fresh_sparse_v3)
    assert 'rows = read_csv(root / "sparse_v3/target_results.csv")' in block
    assert 'context = read_csv(root / "sparse_v3/context_frames.csv")' in block
    assert '"SPARSE_TARGET_COUNT": len(rows)' in block
    assert "MEDIAN_RELATIVE_REDUCTION_DIAGNOSTIC" in block


def test_window_and_cross_episode_require_full_prefix() -> None:
    source = Path(certv3.__file__).read_text(encoding="utf-8")
    assert '"prefix_ordinals": list(range(start))' in source
    assert "stop_ordinal = max(target_ordinals)" in source
    assert "for ordinal in range(stop_ordinal + 1):" in source
    assert "never reset at window start" in source


def test_conditional_twenty_dof_path_remains_conditional() -> None:
    source = Path(certv3.cert.__file__).read_text(encoding="utf-8")
    block = source[source.index("def _run_refinement_frame") : source.index("def _begin_stage")]
    assert "normal_hard and normal_eim > TAU" in block
    assert "if trigger:" in block
    design = certv3.read_json(certv3.R2_ROOT / "design/refinement_v2_design.json")
    assert design["active_dofs"] == list(range(20))
    assert design["bounds_envelope"] == "Wuji asset joint limits; wrist/base remain fixed"


def test_scientific_retry_is_forbidden_and_resume_is_authority_bound() -> None:
    block = _source(certv3._begin_or_resume_stage)
    assert "SCIENTIFIC_RERUN_FORBIDDEN" in block
    assert "TECHNICAL_RESUME_AUTHORITY_MISMATCH" in block
    assert 'plan["CERT_V3_RUN_UUID"]' in block
    assert 'state.get("code_head")' in block


def test_stage_order_and_certification_are_fail_closed() -> None:
    window = _source(certv3.run_fresh_window_v3)
    cross = _source(certv3.run_fresh_cross_episode_v3)
    decision = _source(certv3.decide_cert_v3)
    assert '"FRESH_SPARSE_V3", "PASS"' in window
    assert '"FRESH_WINDOW_V3", "PASS"' in cross
    for criterion in (
        "C1_FRESH_SPARSE_V3",
        "C2_FRESH_WINDOW_V3",
        "C3_FRESH_CROSS_EPISODE_V3",
        "C4_METHOD_INTEGRITY",
        "C5_GATE_INTEGRITY",
        "C6_PROTOCOL_INTEGRITY",
        "C7_QOLD_AUTHORITY",
        "C8_DATA_HYGIENE",
    ):
        assert criterion in decision
    assert "passed = all(criteria.values())" in decision


def test_code_change_after_freeze_blocks_execution() -> None:
    block = _source(certv3._plan)
    assert "CODE_HEAD_OR_TREE_DRIFT" in block
    assert 'git("status", "--short", "--untracked-files=all")' in block


@pytest.mark.parametrize(
    "forbidden",
    [
        "run-d3v2",
        "run-dev2",
        "run-ppo",
        "run-physx",
        "run-o6",
        "consume-certification-split",
        "consume-heldout-split",
        "replace-target",
    ],
)
def test_forbidden_actions_are_absent(forbidden: str) -> None:
    assert forbidden not in certv3.ACTIONS


def test_split_consumption_is_always_zero() -> None:
    source = Path(certv3.__file__).read_text(encoding="utf-8")
    assert source.count('"CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0') >= 8
    assert source.count('"HELDOUT_SPLIT_NEW_CONSUMPTION": 0') >= 8


def test_authorization_never_executes_d3v2() -> None:
    block = _source(certv3.authorize_d3v2)
    assert '"D3_V2_SCIENTIFIC_RUN_COUNT": 0' in block
    assert "run_d3v2" not in block
