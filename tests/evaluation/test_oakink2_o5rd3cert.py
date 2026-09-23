from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd3cert as cert


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_refinement_v2_design_hash_exact() -> None:
    path = cert.R2_ROOT / "design/refinement_v2_design.json"
    assert _sha(path) == cert.DESIGN_SHA256
    assert path.with_suffix(".sha256").read_text().split()[0] == cert.DESIGN_SHA256


def test_frozen_gate_v2_hash_and_semantics_exact() -> None:
    path = cert.R4_ROOT / "gate_v2/development_gate_v2.json"
    gate = cert.read_json(path)
    assert _sha(path) == cert.GATE_V2_SHA256
    assert path.with_suffix(".sha256").read_text().split()[0] == cert.GATE_V2_SHA256
    assert gate["old_invalid_recovery_rate_minimum"] == 0.8
    assert gate["threshold_aware_nonregression_required_fraction"] == 1.0
    assert gate["valid_preservation_required_fraction"] == 1.0
    assert gate["median_invalid_relative_E_IM_reduction"] == "DIAGNOSTIC_ONLY"


def test_historical_r3_failure_is_immutable_posthoc_compatibility_only() -> None:
    r3 = cert.read_json(cert.R3_ROOT / "final_summary.json")
    r4 = cert.read_json(cert.R4_ROOT / "final_summary.json")
    assert r3["D3_R3_STATUS"] == "FAIL_DEVELOPMENT_GATE"
    assert r4["HISTORICAL_GATE_V1_RESULT"] == "FAIL"
    assert r4["HISTORICAL_R3_RESULT_REWRITTEN"] == "NO"
    assert r4["EVIDENCE_ROLE"] == "POST_HOC_COMPATIBILITY_ONLY"
    assert r4["R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY"] == "PASS"


def test_cross_episode_requirement_cannot_be_disabled() -> None:
    source = Path(cert.__file__).read_text()
    stub = cert.read_json(cert.R4_ROOT / "future/fresh_certification_plan_stub.json")
    assert stub["CROSS_EPISODE_REQUIREMENT"] == "REQUIRED"
    assert '"cross_episode_requirement": "REQUIRED"' in source
    assert '"CROSS_EPISODE_REQUIREMENT": "REQUIRED"' in source


def test_cli_exposes_real_required_actions_and_no_forbidden_action() -> None:
    required = {
        "preflight",
        "verify-r4-authority",
        "build-certification-exclusion-ledger",
        "audit-fresh-refinement-pool",
        "audit-qold-authority",
        "freeze-baseline-generation-plan",
        "generate-qold-baselines-if-required",
        "freeze-qold-authority",
        "freeze-certification-plan",
        "select-fresh-sparse",
        "freeze-fresh-sparse",
        "run-fresh-sparse",
        "run-fresh-sparse-determinism",
        "evaluate-fresh-sparse",
        "select-fresh-windows",
        "freeze-fresh-windows",
        "run-fresh-windows",
        "run-fresh-window-determinism",
        "evaluate-fresh-windows",
        "select-fresh-cross-episode-controls",
        "freeze-fresh-cross-episode",
        "run-fresh-cross-episode",
        "evaluate-fresh-cross-episode",
        "audit-method-integrity",
        "decide-refinement-v2-certification",
        "freeze-certified-refinement-v2",
        "authorize-d3v2",
        "generate-d3v2-plan",
        "summarize",
    }
    assert required <= set(cert.ACTIONS)
    assert cert.FORBIDDEN_ACTION_NAMES.isdisjoint(cert.ACTIONS)


def test_refinement_v2_cannot_generate_its_own_qold() -> None:
    source = Path(cert.__file__).read_text()
    assert '"RefinementV2_generates_q_old": False' in source
    assert '"generator": "FROZEN_HISTORICAL_O5_PRODUCTION_SOLVER"' in source
    assert "generate_qold_baselines_if_required" in source
    assert (
        "_run_refinement_frame"
        not in source[
            source.index("def generate_qold_baselines_if_required") : source.index(
                "def freeze_qold_authority"
            )
        ]
    )


def test_generic_runtime_loads_warm_start_from_artifact_authority() -> None:
    source = Path(cert.__file__).read_text()
    block = source[
        source.index("class FreshRefinementRuntime") : source.index("def _baseline_metrics")
    ]
    assert "self.warm = load_warm_start" in block
    assert "o5.load_warm_start" not in block


def test_plan_is_required_before_every_fresh_manifest() -> None:
    source = Path(cert.__file__).read_text()
    for function in (
        "select_fresh_sparse",
        "select_fresh_windows",
        "select_fresh_cross_episode_controls",
    ):
        block = source[source.index(f"def {function}") :]
        block = block[: block.index("\ndef ", 5)]
        assert "_manifest_authority" in block


def test_sparse_strata_use_only_frozen_baseline_metrics() -> None:
    source = Path(cert.__file__).read_text()
    block = source[
        source.index("def freeze_certification_plan") : source.index("def _manifest_authority")
    ]
    assert "baseline_E_IM" in block
    assert "final_E_IM" not in block
    assert "_run_refinement_frame" not in block
    assert cert.SPARSE_N == 30
    assert cert.SPARSE_PER_STRATUM == 10


def test_conditional_expanded_path_is_not_unconditional() -> None:
    source = Path(cert.__file__).read_text()
    block = source[source.index("def _run_refinement_frame") : source.index("def _begin_stage")]
    assert "normal_hard and normal_eim > TAU" in block
    assert "if trigger:" in block
    assert '"triggered": False' in block
    assert '"active_dofs": []' in block


def test_expanded_active_set_is_exact_asset_derived_twenty_dofs() -> None:
    design = cert.read_json(cert.R2_ROOT / "design/refinement_v2_design.json")
    source = Path(cert.__file__).read_text()
    assert design["active_dofs"] == list(range(20))
    assert "asset_derived_dof_blocks(runtime.model.dof_names)" in source
    assert "EXPANDED_ACTIVE_SET_ASSET_INVENTORY_MISMATCH" in source


def test_expanded_wrist_and_base_are_locked_by_frozen_phase() -> None:
    source = Path(cert.d2c.__file__).read_text()
    assert "context.trust_region_limits = (" in source
    assert "1.0e-12," in source
    design = cert.read_json(cert.R2_ROOT / "design/refinement_v2_design.json")
    assert design["bounds_envelope"] == "Wuji asset joint limits; wrist/base remain fixed"


def test_threshold_aware_nonregression_is_exact() -> None:
    source = Path(cert.__file__).read_text()
    assert "new <= max(old, TAU) + EPS" in source
    gate = cert.read_json(cert.R4_ROOT / "gate_v2/development_gate_v2.json")
    assert gate["threshold_aware_nonregression"] == "E_new <= max(E_old, tau) + epsilon_num"
    assert cert.EPS == gate["epsilon_num"]


def test_sparse_failure_blocks_windows_and_cross_episode() -> None:
    source = Path(cert.__file__).read_text()
    block = source[
        source.index("def evaluate_fresh_sparse") : source.index("def select_fresh_windows")
    ]
    assert '"FRESH_REFINEMENT_WINDOW": "NOT_RUN"' in block
    assert '"CROSS_EPISODE_REFINEMENT": "NOT_RUN"' in block
    assert "DEV1_REFINEMENT_V2_FRESH_SPARSE_FAILURE_ANALYSIS" in block


def test_consumed_sparse_technical_failure_is_fail_closed() -> None:
    source = Path(cert.__file__).read_text()
    block = source[
        source.index("def finalize_sparse_technical_failure") : source.index(
            "def select_fresh_windows"
        )
    ]
    assert '"FAILED_TECHNICAL_AFTER_OPTIMIZER_START"' in block
    assert '"TECHNICAL": f"{len(rows)}/{SPARSE_N}"' in block
    assert '"SCIENTIFIC_RERUN_ALLOWED": "NO"' in block
    assert '"FRESH_REFINEMENT_WINDOW": "NOT_RUN"' in block
    assert '"CROSS_EPISODE_REFINEMENT": "NOT_RUN"' in block
    assert '"RECOVERED_COUNT": "UNKNOWN_NOT_MEASURED"' in block


def test_handoff_contains_contract_required_safety_and_runtime_fields() -> None:
    source = Path(cert.__file__).read_text()
    block = source[source.index("def summarize") : source.index("def validate_repository")]
    for field in (
        "HIGH_N",
        "MID_N",
        "LOW_N",
        "WINDOW_MANIFEST_SHA256",
        "CROSS_EPISODE_CONTROL_COUNT",
        "NORMAL_PATH_FRAME_COUNT",
        "VALID_NORMAL_PATHS_UNNECESSARILY_EXPANDED",
        "ESTIMATED_D3_V2_2722_RUNTIME_SEC",
        "REFINEMENT_V2_DESIGN_CHANGED_DURING_CERTIFICATION",
        "CERTIFICATION_GATE_V2_CHANGED",
        "GUIDANCE_WORKTREE_MODIFIED",
        "BASELINE_GENERATION_RUN_COUNT",
        "REFINEMENT_V2_FRESH_CERTIFICATION_PLAN_SHA256",
        "ENGINEERING_DELIVERY_STATUS",
        "missing_terminal_artifacts",
    ):
        assert field in block


def test_window_failure_blocks_cross_episode() -> None:
    source = Path(cert.__file__).read_text()
    block = source[
        source.index("def evaluate_fresh_windows") : source.index(
            "def select_fresh_cross_episode_controls"
        )
    ]
    assert '"CROSS_EPISODE_REFINEMENT": "NOT_RUN"' in block
    assert "DEV1_REFINEMENT_V2_FRESH_SEQUENCE_FAILURE_ANALYSIS" in block


def test_window_qold_is_distinct_authority_from_previous_runtime_state() -> None:
    source = Path(cert.__file__).read_text()
    assert '"q_old_semantics": "historical q_old[current frame] immutable"' in source
    assert (
        '"previous_runtime_semantics": "ABSENT at local frame0; '
        'accepted refined state[t-1] thereafter"' in source
    )
    assert '"q_old_distinct_from_previous_runtime"' in source


def test_sparse_window_overlap_is_rejected() -> None:
    source = Path(cert.__file__).read_text()
    assert "if set(keys) & sparse_keys:" in source
    assert 'raise RuntimeError("WINDOW_SPARSE_OVERLAP")' in source


def test_cross_episode_source_sequence_overlap_is_rejected() -> None:
    source = Path(cert.__file__).read_text()
    assert "if sequences & prior_sequences:" in source
    assert 'raise RuntimeError("CROSS_SOURCE_SEQUENCE_NOT_DISJOINT")' in source


def test_scientific_stage_cannot_be_retried() -> None:
    source = Path(cert.__file__).read_text()
    block = source[source.index("def _begin_stage") : source.index("def _mark_completed")]
    assert "if path.exists():" in block
    assert "SCIENTIFIC_RERUN_FORBIDDEN" in block
    assert '"retry_allowed": False' in block


def test_determinism_repeats_are_separate_from_primary_results() -> None:
    source = Path(cert.__file__).read_text()
    assert "sparse/determinism_run_state.json" in source
    assert "window/determinism_run_state.json" in source
    assert '"scientific_primary_count"' in source
    assert cert.DETERMINISM_RUNS == 3


def test_method_hash_change_invalidates_certification() -> None:
    source = Path(cert.__file__).read_text()
    block = source[
        source.index("def audit_method_integrity") : source.index("def decide_refinement")
    ]
    assert "observed[name] == digest" in block
    assert '"METHOD_INTEGRITY_POSTRUN": "PASS" if all(checks.values()) else "FAIL"' in block


def test_authorize_d3v2_requires_certified_authority() -> None:
    source = Path(cert.__file__).read_text()
    block = source[source.index("def authorize_d3v2") : source.index("def generate_d3v2_plan")]
    assert "certified_refinement_v2_authority.json" in block
    assert 'certified.get("status") != "CERTIFIED"' in block
    assert '"D3_V2_SCIENTIFIC_RUN_COUNT": 0' in block


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
    ],
)
def test_forbidden_downstream_action_is_absent(forbidden: str) -> None:
    assert forbidden not in cert.ACTIONS


def test_split_consumption_stays_zero_in_every_generated_authority() -> None:
    source = Path(cert.__file__).read_text()
    assert source.count('"CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0') >= 8
    assert source.count('"HELDOUT_SPLIT_NEW_CONSUMPTION": 0') >= 8


def test_freeze_json_detects_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    cert.freeze_json(path, {"status": "FROZEN", "value": 1})
    path.write_text(json.dumps({"status": "FROZEN", "value": 2}))
    with pytest.raises(RuntimeError, match="FROZEN_ARTIFACT_HASH_DRIFT"):
        cert.freeze_json(path, {"status": "FROZEN", "value": 1})


def test_begin_stage_refuses_second_scientific_run(tmp_path: Path) -> None:
    root = tmp_path
    cert._begin_stage(root, "sparse/run_state.json", "a" * 64, 30)
    with pytest.raises(RuntimeError, match="SCIENTIFIC_RERUN_FORBIDDEN"):
        cert._begin_stage(root, "sparse/run_state.json", "a" * 64, 30)
