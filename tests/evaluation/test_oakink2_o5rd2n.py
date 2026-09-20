from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation import run_oakink2_o5rd2n as study


def test_cli_exposes_d2n_contract() -> None:
    required = {
        "preflight",
        "verify-d2mr2-preconditions",
        "verify-v1-v2-history",
        "verify-frozen-v4",
        "verify-consumer-authorities",
        "verify-dev2-identity",
        "run-full-consumer-preflight",
        "freeze-v3-run",
        "run-v3-full",
        "resume-v3-full",
        "verify-frame0-parity",
        "verify-frame10705-milestone",
        "verify-input-authority-clearance",
        "verify-runtime-chain",
        "verify-source-bindings",
        "finalize-v3-trajectory",
        "run-v3-semantic-v1",
        "render-v3-viewer",
        "audit-v3-special-cases",
        "summarize",
    }
    assert required <= study.ACTIONS.keys()


def test_frozen_v3_uuid_comes_from_plan_and_is_new() -> None:
    plan = study._frozen_plan()
    assert plan["RUN_UUID"] == study.EXPECTED_V3_UUID
    assert plan["RUN_UUID"] not in {study.V1_UUID, study.V2_UUID}
    assert plan["DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT"] == 0


def test_frozen_full_sequence_authorities_are_exact() -> None:
    assert (
        study.sha256_file(
            study.D2MR2_ROOT / "frozen_authority/full_sequence_consumer_input_authority.json"
        )
        == study.CONSUMER_AUTHORITY_SHA
    )
    assert (
        study.sha256_file(
            study.D2MR2_ROOT / "frozen_authority/full_sequence_preflight_contract.json"
        )
        == study.PREFLIGHT_CONTRACT_SHA
    )
    assert (
        study.sha256_file(study.d2mr.ROOT / "graph_authority/sequence_authority.json")
        == study.GRAPH_AUTHORITY_SHA
    )


def test_v1_v2_histories_remain_immutable() -> None:
    upstream = study.read_json(study.D2MR2_ROOT / "history/run_history.json")
    assert upstream["V1"]["RUN_UUID"] == study.V1_UUID
    assert upstream["V1"]["result"] == "RETARGET_NUMERICAL_FAIL"
    assert upstream["V1"]["completed"] == 1
    assert upstream["V2"]["RUN_UUID"] == study.V2_UUID
    assert upstream["V2"]["result"] == "BLOCKED_INPUT_AUTHORITY"
    assert upstream["V2"]["completed"] == 1


def test_runtime_never_reads_warm_after_frame0_or_qold() -> None:
    binding = inspect.getsource(study.d2mr2.FullSequenceV3Runtime.bind_context)
    execution = inspect.getsource(study._execute_v3)
    assert "self.warm" not in binding
    assert "q_old" not in binding
    assert '"q_old_access_count": 0' in execution
    assert '"warm_access_count_t_gt_0": 0' in execution


def test_previous_runtime_state_is_not_qold() -> None:
    source = inspect.getsource(study._execute_v3)
    assert "previous_q, previous_base =" in source
    assert "previous = state" in source
    assert "q_old=" not in source


def test_full_preflight_is_required_before_manifest_and_run() -> None:
    freeze = inspect.getsource(study.freeze_v3_run)
    execute = inspect.getsource(study._execute_v3)
    assert '"preflight/consumer_preflight.json"' in freeze
    assert '"preflight/consumer_preflight.json"' in execute
    assert '"status", "PASS"' in freeze
    assert '"status", "PASS"' in execute


def test_frame10705_barrier_requires_context_and_optimizer_start() -> None:
    source = inspect.getsource(study._execute_v3)
    marker = inspect.getsource(study._mark_optimizer_start)
    assert "_record_context_milestone" in source
    assert "_mark_optimizer_start" in source
    assert source.index("_record_context_milestone") < source.index("_mark_optimizer_start")
    assert '"FRAME10705_CONTEXT_BUILT": "YES"' in marker
    assert '"FRAME10705_OPTIMIZER_STARTED": "YES"' in marker
    assert '"FULL_SEQUENCE_INPUT_AUTHORITY_BARRIERS_CLEARED": "YES"' in marker


def test_input_failure_before_optimizer_is_not_scientific(tmp_path: Path) -> None:
    value = study._input_failure(
        tmp_path,
        {"RUN_UUID": study.EXPECTED_V3_UUID},
        1,
        RuntimeError("binding mismatch"),
        [{"ordinal": 0}],
    )
    assert value["status"] == "BLOCKED_INPUT_AUTHORITY"
    assert value["FIRST_INPUT_AUTHORITY_FAILURE_ORDINAL"] == 1
    assert value["FIRST_REAL_FULL_TRAJECTORY_FAILURE_ORDINAL"] is None
    assert value["COMPLETED_FRAMES"] == 1


def test_scientific_count_changes_only_at_frame0_start(tmp_path: Path) -> None:
    manifest = {"RUN_UUID": study.EXPECTED_V3_UUID}
    run_state = {
        "SCIENTIFIC_RUN_COUNT": 0,
        "TECHNICAL_RESUME_COUNT": 0,
        "status": "FROZEN_NOT_STARTED",
    }
    study.write_json(
        tmp_path / "history/run_history.json",
        {
            "V1": {},
            "V2": {},
            "V3": {"scientific_run_count": 0},
            "DEV2_FULL_GEOMETRIC_RUN_HISTORY_COUNT": 2,
        },
    )
    study._mark_optimizer_start(tmp_path, manifest, run_state, 0)
    observed = study.read_json(tmp_path / "run_authority/run_state.json")
    history = study.read_json(tmp_path / "history/run_history.json")
    assert observed["SCIENTIFIC_RUN_COUNT"] == 1
    assert history["V3"]["scientific_run_count"] == 1
    assert history["DEV2_FULL_GEOMETRIC_RUN_HISTORY_COUNT"] == 3


def test_second_v3_scientific_run_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = {
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_UUID": study.EXPECTED_V3_UUID,
        "method_hashes": {},
    }
    study.write_json(tmp_path / "run_authority/full_run_manifest.json", manifest)
    study.write_text(
        tmp_path / "run_authority/full_run_manifest.sha256",
        study.sha256_file(tmp_path / "run_authority/full_run_manifest.json") + "\n",
    )
    study.write_json(tmp_path / "preflight/consumer_preflight.json", {"status": "PASS"})
    study.write_json(
        tmp_path / "run_authority/run_state.json",
        {"SCIENTIFIC_RUN_COUNT": 1, "TECHNICAL_RESUME_COUNT": 0, "status": "SCIENTIFIC_FAIL"},
    )
    monkeypatch.setattr(
        study.d2m,
        "_full_sequence_graph_preflight",
        lambda _manifest: {"graph_source_frames": [], "canonical_frames": []},
    )
    monkeypatch.setattr(study.d2m, "_method_hashes", lambda _path: {})
    monkeypatch.setattr(study.d2mr2, "build_runtime", lambda: SimpleNamespace())
    with pytest.raises(RuntimeError, match="SCIENTIFIC_RUN_COUNT_ALREADY_ONE"):
        study.run_v3_full(tmp_path)


def test_resume_policy_is_same_uuid_and_fail_closed() -> None:
    policy = study._technical_resume_policy()
    assert "same RUN_UUID" in policy["requires"]
    assert "same manifest SHA" in policy["requires"]
    assert policy["scientific_failure_resume_through"] == "FORBIDDEN"
    assert policy["second_v3_scientific_attempt"] == "FORBIDDEN"


def test_no_runner_fallback_or_budget_escalation() -> None:
    source = inspect.getsource(study._execute_v3)
    assert "d2g2.CS2_A" in source
    assert "default_cold_start_search_v4_candidates()[0]" in source
    assert "qpos = previous_q" not in source
    assert "top_k =" not in source
    assert "max_nfev =" not in source


def test_semantic_and_viewer_require_complete_trajectory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="RUN_SEMANTIC_V1_REJECTED:MISSING"):
        study.run_v3_semantic_v1(tmp_path)
    with pytest.raises(RuntimeError, match="RENDER_V3_VIEWER_REJECTED:MISSING"):
        study.render_v3_viewer(tmp_path)


def test_special_case_audit_uses_only_frozen_scientific_callgraph(tmp_path: Path) -> None:
    value = study.audit_v3_special_cases(tmp_path)
    assert value["status"] == "PASS"
    assert value["DEV2_EPISODE_SPECIFIC_SEARCH_BRANCH"] == "NO"
    assert value["DEV2_C11001_SEARCH_BRANCH"] == "NO"
    assert value["DEV2_FRAME10705_SEARCH_BRANCH"] == "NO"


def test_method_hashes_remain_exact() -> None:
    for path, expected in study.d2m.FROZEN_AUTHORITIES.values():
        assert study.sha256_file(path) == expected
    for path, expected in study.d2m.METHOD_IMPLEMENTATIONS.values():
        assert study.sha256_file(path) == expected
