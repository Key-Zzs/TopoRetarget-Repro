from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd3certv2 as certv2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_upstream_hashes_are_exact() -> None:
    assert _sha(certv2.R2_ROOT / "design/refinement_v2_design.json") == certv2.DESIGN_SHA256
    assert _sha(certv2.R4_ROOT / "gate_v2/development_gate_v2.json") == certv2.GATE_V2_SHA256
    assert (
        _sha(certv2.CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
        == certv2.PROTOCOL_V2_SHA256
    )


def test_cert_v1_failure_and_cert_r_pass_are_immutable() -> None:
    cert_v1 = certv2.read_json(certv2.CERT_V1_ROOT / "final_summary.json")
    cert_r = certv2.read_json(certv2.CERT_R_ROOT / "final_summary.json")
    assert cert_v1["REFINEMENT_V2_INDEPENDENT_CERTIFICATION"] == "FAIL"
    assert cert_r["D3_CERT_V1_HISTORICAL_RESULT"] == "FAIL"
    assert cert_r["D3_CERT_V1_HISTORICAL_RESULT_REWRITTEN"] == "NO"
    assert cert_r["CERT_R_STATUS"] == "PASS_PROTOCOL_REPAIR"
    assert cert_r["SPARSE_V1_CONSUMED_REGRESSION"] == "PASS"


def test_protocol_v2_requires_prefix_and_cross_episode() -> None:
    protocol = certv2.read_json(certv2.CERT_R_ROOT / "protocol_v2/certification_protocol_v2.json")
    assert protocol["SPARSE_UNIT"] == "PREFIX_ANCHORED_TARGET"
    assert protocol["previous_runtime_authority"] == "PREVIOUS_ACCEPTED_REFINED_RUNTIME"
    assert protocol["gate_counting_rule"] == "exactly 30 target frames"
    assert protocol["CROSS_EPISODE_REQUIREMENT"] == "REQUIRED"
    assert protocol["scientific_retry_policy"] == "NO_RETRY_AFTER_OPTIMIZER_START"


def test_cli_exposes_full_contract_and_no_forbidden_action() -> None:
    required = {
        "preflight",
        "verify-cert-r-authority",
        "verify-certification-protocol-v2",
        "verify-cert-v2-exclusion-ledger",
        "build-cert-v2-fresh-pool",
        "audit-reusable-qold",
        "freeze-cert-v2-qold-generation-plan",
        "generate-cert-v2-qold-if-required",
        "freeze-cert-v2-qold-authority",
        "freeze-cert-v2-run-plan",
        "freeze-cert-v2-sparse-manifest",
        "freeze-cert-v2-window-manifest",
        "freeze-cert-v2-cross-episode-manifest",
        "run-fresh-sparse-v2",
        "resume-fresh-sparse-v2",
        "run-fresh-sparse-v2-determinism",
        "evaluate-fresh-sparse-v2",
        "run-fresh-window-v2",
        "resume-fresh-window-v2",
        "run-fresh-window-v2-determinism",
        "evaluate-fresh-window-v2",
        "run-fresh-cross-episode-v2",
        "resume-fresh-cross-episode-v2",
        "run-fresh-cross-episode-v2-determinism",
        "evaluate-fresh-cross-episode-v2",
        "audit-cert-v2-method-integrity",
        "audit-cert-v2-protocol-integrity",
        "decide-cert-v2",
        "freeze-certified-refinement-v2",
        "authorize-d3v2",
        "generate-d3v2-plan",
        "summarize",
    }
    assert required <= set(certv2.ACTIONS)
    assert certv2.FORBIDDEN_ACTION_NAMES.isdisjoint(certv2.ACTIONS)


def test_qold_generation_has_no_refinement_execution() -> None:
    source = Path(certv2.__file__).read_text()
    block = source[
        source.index("def generate_cert_v2_qold_if_required") : source.index(
            "def freeze_cert_v2_qold_authority"
        )
    ]
    assert "_run_refinement_frame" not in block
    assert "_generate_one_baseline" in block


def test_qold_is_frozen_before_target_selection() -> None:
    source = Path(certv2.__file__).read_text()
    block = source[source.index("def freeze_cert_v2_run_plan") : source.index("def _plan")]
    assert 'frozen(root / "qold/qold_authority_manifest.json"' in block
    assert "QOLD_NOT_FROZEN_BEFORE_TARGET_SELECTION" in block
    assert "_select_sparse_targets" in block
    assert "final_E_IM" not in block


def test_sparse_selection_uses_baseline_only_and_distinct_sequences() -> None:
    rows = []
    for baseline in range(certv2.SPARSE_N):
        for ordinal, value in enumerate(
            (1.0e-5 + baseline * 1.0e-7, 1.0e-4, 2.0e-4 + baseline * 1.0e-6)
        ):
            rows.append(
                {
                    "baseline_id": f"b{baseline:02d}",
                    "record_id": f"r{baseline:02d}",
                    "sequence_id": f"s{baseline:02d}",
                    "primitive": "p",
                    "object_id": "o",
                    "ordinal": ordinal,
                    "source_frame": ordinal,
                    "baseline_E_IM": value,
                    "hard_valid": True,
                }
            )
    selected = certv2._select_sparse_targets(rows)
    assert len(selected) == certv2.SPARSE_N
    assert len({item["sequence_id"] for item in selected}) == certv2.SPARSE_N
    assert {
        name: sum(item["stratum"] == name for item in selected) for name in ("HIGH", "MID", "LOW")
    } == {
        "HIGH": 10,
        "MID": 10,
        "LOW": 10,
    }


def test_prefix_execution_is_exact_zero_through_target() -> None:
    source = Path(certv2.__file__).read_text()
    block = source[source.index("def _execute_prefix_unit") : source.index("def _append_consumed")]
    assert "for ordinal in range(stop_ordinal + 1):" in block
    assert "runtime.bind_context" in block
    assert "cert._run_refinement_frame" in block
    assert block.index("runtime.bind_context") < block.index("cert._run_refinement_frame")
    assert "PREVIOUS_ACCEPTED_REFINED_RUNTIME" in source


def test_context_frames_are_not_counted_as_sparse_targets() -> None:
    source = Path(certv2.__file__).read_text()
    block = source[
        source.index("def evaluate_fresh_sparse_v2") : source.index("def _run_sequence_stage")
    ]
    assert 'rows = read_csv(root / "sparse_v2/target_results.csv")' in block
    assert 'context = read_csv(root / "sparse_v2/context_frames.csv")' in block
    assert '"SPARSE_TARGET_COUNT": len(rows)' in block


def test_window_nonzero_start_executes_full_prefix() -> None:
    source = Path(certv2.__file__).read_text()
    assert '"prefix_ordinals": list(range(start))' in source
    assert "stop_ordinal = max(target_ordinals)" in source
    assert "for ordinal in range(stop_ordinal + 1):" in source
    assert "never reset at window start" in source


def test_determinism_repeats_complete_prefix() -> None:
    source = Path(certv2.__file__).read_text()
    assert source.count("_repeat_prefix(") >= 3
    assert source.count('"full_prefix_repeated": True') >= 2
    assert certv2.DETERMINISM_RUNS == 3


def test_conditional_twenty_dof_path_remains_conditional() -> None:
    source = Path(certv2.cert.__file__).read_text()
    block = source[source.index("def _run_refinement_frame") : source.index("def _begin_stage")]
    assert "normal_hard and normal_eim > TAU" in block
    assert "if trigger:" in block
    design = certv2.read_json(certv2.R2_ROOT / "design/refinement_v2_design.json")
    assert design["active_dofs"] == list(range(20))
    assert design["bounds_envelope"] == "Wuji asset joint limits; wrist/base remain fixed"


def test_scientific_rerun_is_forbidden_and_resume_is_authority_bound(tmp_path: Path) -> None:
    root = tmp_path
    (root / "run_plan").mkdir()
    (root / "run_plan/certification_run_plan_v2.json").write_text("{}")
    path, state = certv2._begin_or_resume_stage(
        root, "sparse_v2/run_state.json", "a" * 64, 30, resume=False
    )
    assert state["scientific_retry_allowed"] is False
    with pytest.raises(RuntimeError, match="SCIENTIFIC_RERUN_FORBIDDEN"):
        certv2._begin_or_resume_stage(root, "sparse_v2/run_state.json", "a" * 64, 30, resume=False)
    stored = json.loads(path.read_text())
    assert stored["run_uuid"]
    assert stored["manifest_sha256"] == "a" * 64


def test_stage_order_is_fail_closed() -> None:
    source = Path(certv2.__file__).read_text()
    window = source[
        source.index("def run_fresh_window_v2") : source.index("def resume_fresh_window_v2")
    ]
    cross = source[
        source.index("def run_fresh_cross_episode_v2") : source.index(
            "def resume_fresh_cross_episode_v2"
        )
    ]
    assert '"FRESH_SPARSE_V2", "PASS"' in window
    assert '"FRESH_WINDOW_V2", "PASS"' in cross


def test_certification_pass_requires_all_six_criteria() -> None:
    source = Path(certv2.__file__).read_text()
    block = source[source.index("def decide_cert_v2") : source.index("def freeze_certified")]
    for name in (
        "C1_FRESH_SPARSE_V2",
        "C2_FRESH_WINDOW_V2",
        "C3_FRESH_CROSS_EPISODE_V2",
        "C4_METHOD_INTEGRITY",
        "C5_PROTOCOL_INTEGRITY",
        "C6_DATA_HYGIENE",
    ):
        assert name in block
    assert "passed = all(criteria.values())" in block


def test_authorize_d3v2_never_executes_d3v2() -> None:
    source = Path(certv2.__file__).read_text()
    block = source[source.index("def authorize_d3v2") : source.index("def generate_d3v2_plan")]
    assert 'certified.get("status") != "CERTIFIED"' in block
    assert '"D3_V2_SCIENTIFIC_RUN_COUNT": 0' in block
    assert "run_d3v2" not in block


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
    assert forbidden not in certv2.ACTIONS


def test_split_consumption_is_always_zero() -> None:
    source = Path(certv2.__file__).read_text()
    assert source.count('"CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0') >= 8
    assert source.count('"HELDOUT_SPLIT_NEW_CONSUMPTION": 0') >= 8
