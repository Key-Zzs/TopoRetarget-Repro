from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.evaluation import run_oakink2_o5rd2m as study


def test_frozen_authorities_match_d2l_and_upstream() -> None:
    for path, expected in study.FROZEN_AUTHORITIES.values():
        assert study.sha256_file(path) == expected
        sidecar = path.with_suffix(".sha256")
        if sidecar.is_file():
            assert sidecar.read_text(encoding="utf-8").split()[0] == expected
    for path, expected in study.METHOD_IMPLEMENTATIONS.values():
        assert study.sha256_file(path) == expected
    assert study.GRAPH_PATH == (study.d2g.ROOT / "graph_authority/dev2_frame0_source_graph.zarr")
    assert study.d2g.interaction_artifact_hash(study.GRAPH_PATH) == (
        "2b941f517183b4e70903005f7b6914a7eedfa28654306db4f023e0447639f375"
    )
    assert study.d2g.digest(study.CANONICAL_PATH) == (
        "19ece59feb8588f1f155ac523b0497bb3d4e5f8965c3710d5e49b9c32e4d2d3d"
    )


def test_cli_exposes_full_fail_closed_contract() -> None:
    expected = {
        "preflight",
        "verify-frozen-v4",
        "verify-dev2-identity",
        "freeze-dev2-run",
        "run-dev2-full",
        "resume-dev2-full",
        "verify-runtime-chain",
        "finalize-dev2-trajectory",
        "run-semantic-v1",
        "render-dev2-viewer",
        "audit-dev2-special-cases",
        "summarize",
    }
    assert expected <= study.ACTIONS.keys()


def test_exact_240_frame_coverage_contract() -> None:
    rows = [
        {"ordinal": ordinal, "source_frame": study.SOURCE_START + ordinal}
        for ordinal in range(study.EXPECTED_FRAMES)
    ]
    coverage = study._coverage_payload(rows)
    assert coverage["status"] == "PASS"
    assert coverage["COMPLETED_FRAMES"] == 240
    assert coverage["FIRST_SOURCE_FRAME"] == 10704
    assert coverage["LAST_SOURCE_FRAME"] == 10943
    assert coverage["NO_SKIPPED_FRAMES"] == "YES"
    assert coverage["NO_DUPLICATED_FRAMES"] == "YES"
    assert coverage["SOURCE_FRAME_ORDER_STRICT"] == "YES"


def test_skip_and_duplicate_fail_coverage() -> None:
    skipped = [
        {"ordinal": ordinal, "source_frame": study.SOURCE_START + ordinal}
        for ordinal in range(10)
        if ordinal != 5
    ]
    duplicated = skipped + [skipped[-1]]
    assert study._coverage_payload(skipped)["NO_SKIPPED_FRAMES"] == "NO"
    assert study._coverage_payload(duplicated)["NO_DUPLICATED_FRAMES"] == "NO"


def test_technical_resume_policy_separates_scientific_failure() -> None:
    policy = study._technical_resume_policy()
    assert policy["scientific_failure_resume_through"] == "FORBIDDEN"
    assert policy["second_scientific_run"] == "FORBIDDEN"
    assert set(policy["scientific_failures"]) == study.FAILURE_ENUM
    assert "same RUN_UUID" in policy["technical_resume_requires"]


def test_freeze_rejects_missing_upstream_integrity(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="FREEZE_DEV2_RUN_REJECTED:MISSING"):
        study.freeze_dev2_run(tmp_path)


def test_corrupt_checkpoint_rejected(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "DEV2FullGeometricRunManifestV1",
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_UUID": "run",
    }
    study.atomic_write_json(tmp_path / "run_authority/full_run_manifest.json", manifest)
    study.atomic_write_text(
        tmp_path / "run_authority/full_run_manifest.sha256",
        study.sha256_file(tmp_path / "run_authority/full_run_manifest.json") + "\n",
    )
    study.atomic_write_json(
        tmp_path / "checkpoints/frame_000/checkpoint.json",
        {
            "status": "ACCEPTED",
            "ordinal": 0,
            "RUN_UUID": "run",
            "manifest_sha256": study.sha256_file(tmp_path / "run_authority/full_run_manifest.json"),
        },
    )
    with pytest.raises(RuntimeError, match="CORRUPT_CHECKPOINT_CONTENT"):
        study.verify_runtime_chain(tmp_path)


def test_scientific_run_count_cannot_exceed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method_hashes = {"frozen": "exact"}
    manifest = {
        "schema_version": "DEV2FullGeometricRunManifestV1",
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_UUID": "one-run",
        "method_hashes": method_hashes,
    }
    study.atomic_write_json(tmp_path / "run_authority/full_run_manifest.json", manifest)
    study.atomic_write_text(
        tmp_path / "run_authority/full_run_manifest.sha256",
        study.sha256_file(tmp_path / "run_authority/full_run_manifest.json") + "\n",
    )
    monkeypatch.setattr(study, "_method_hashes", lambda *_args: method_hashes)
    monkeypatch.setattr(
        study,
        "_full_sequence_graph_preflight",
        lambda _manifest: {
            "graph_path": study.GRAPH_PATH,
            "graph_source_frames": list(range(study.SOURCE_START, study.SOURCE_STOP)),
            "canonical_frames": list(range(study.SOURCE_START, study.SOURCE_STOP)),
        },
    )
    monkeypatch.setattr(
        study,
        "_build_dev2_runtime",
        lambda *_args: SimpleNamespace(
            graph=SimpleNamespace(
                frame_count=study.EXPECTED_FRAMES,
                graph_hashes=["graph"] * study.EXPECTED_FRAMES,
            ),
            warm=SimpleNamespace(
                arrays={
                    "qpos": [[0.0]] * study.EXPECTED_FRAMES,
                    "base_pose_scene": [[[0.0]]] * study.EXPECTED_FRAMES,
                }
            ),
        ),
    )

    def fail_scientifically(*_args, **_kwargs):
        raise RuntimeError("no valid candidate")

    monkeypatch.setattr(study.d2g2, "search_cold_start_v2_frame", fail_scientifically)
    result = study.run_dev2_full(tmp_path)
    assert result["status"] == "SCIENTIFIC_FAIL"
    assert result["DEV2_FULL_GEOMETRIC_SOLVE_COUNT"] == 1
    with pytest.raises(RuntimeError, match="SCIENTIFIC_RUN_COUNT_ALREADY_ONE"):
        study.run_dev2_full(tmp_path)


def test_semantic_v1_requires_complete_finalized_trajectory(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="RUN_SEMANTIC_V1_REJECTED:MISSING"):
        study.run_semantic_v1(tmp_path)


def test_no_budget_escalation_and_qold_absence_are_static_contracts() -> None:
    candidate = study.default_cold_start_search_v4_candidates()[0]
    assert candidate.name == "V4_A_TOP2_SEQUENTIAL"
    assert candidate.top_k == 2
    assert candidate.contributor_probe_max_nfev == 24
    assert candidate.selected_primary_maxiter == 8
    assert candidate.secondary_polish_maxiter == 8
    source = inspect.getsource(study._execute_dev2)
    assert "d2g2.CS2_A" in source
    assert "q_old_access_count" in source
    assert "DEV2_DEVELOPMENT_FRAME0_STATE_REUSED" not in source


def test_special_case_audit_scans_only_scientific_call_graph(tmp_path: Path) -> None:
    result = study.audit_dev2_special_cases(tmp_path)
    assert result["status"] == "PASS"
    assert result["DEV2_EPISODE_ID_BRANCH"] == "NO"
    assert result["DEV2_OBJECT_C11001_BRANCH"] == "NO"
    assert result["DEV2_FRAME_10704_BRANCH"] == "NO"


def test_resume_rejects_different_run_uuid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    method_hashes = {"frozen": "exact"}
    manifest = {
        "schema_version": "DEV2FullGeometricRunManifestV1",
        "status": "FROZEN_BEFORE_SOLVE",
        "RUN_UUID": "original-run",
        "method_hashes": method_hashes,
    }
    study.atomic_write_json(tmp_path / "run_authority/full_run_manifest.json", manifest)
    study.atomic_write_text(
        tmp_path / "run_authority/full_run_manifest.sha256",
        study.sha256_file(tmp_path / "run_authority/full_run_manifest.json") + "\n",
    )
    study.atomic_write_json(
        tmp_path / "run_authority/run_state.json",
        {
            "SCIENTIFIC_RUN_COUNT": 1,
            "TECHNICAL_RESUME_COUNT": 0,
            "status": "TECHNICAL_INTERRUPTION",
        },
    )
    study.atomic_write_json(
        tmp_path / "technical_interruption.json",
        {
            "resume_allowed": True,
            "RUN_UUID": "different-run",
            "manifest_sha256": study.sha256_file(tmp_path / "run_authority/full_run_manifest.json"),
        },
    )
    monkeypatch.setattr(study, "_method_hashes", lambda *_args: method_hashes)
    monkeypatch.setattr(
        study,
        "_full_sequence_graph_preflight",
        lambda _manifest: {"graph_path": study.GRAPH_PATH},
    )
    monkeypatch.setattr(
        study,
        "_build_dev2_runtime",
        lambda *_args: SimpleNamespace(
            graph=SimpleNamespace(frame_count=study.EXPECTED_FRAMES),
            warm=SimpleNamespace(
                arrays={
                    "qpos": [[0.0]] * study.EXPECTED_FRAMES,
                    "base_pose_scene": [[[0.0]]] * study.EXPECTED_FRAMES,
                }
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="RUN_AUTHORITY_MISMATCH"):
        study.resume_dev2_full(tmp_path)


def test_full_trajectory_rejects_239_frames(tmp_path: Path) -> None:
    study.atomic_write_json(
        tmp_path / "solver/result.json", {"status": "PASS", "COMPLETED_FRAMES": 239}
    )
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        study.finalize_dev2_trajectory(tmp_path)


def test_runtime_seed_carrier_coverage_rejected_before_optimizer() -> None:
    runtime = SimpleNamespace(
        graph=SimpleNamespace(frame_count=study.EXPECTED_FRAMES),
        warm=SimpleNamespace(
            arrays={"qpos": [[0.0]], "base_pose_scene": [[[0.0]]]},
        ),
    )
    with pytest.raises(RuntimeError, match="FULL_SEQUENCE_RUNTIME_INPUT_COVERAGE_MISMATCH"):
        study._full_sequence_runtime_input_preflight(runtime)


def test_sequential_state_and_viewer_role_are_fail_closed() -> None:
    execute_source = inspect.getsource(study._execute_dev2)
    assert "previous = None if not accepted_states else accepted_states[-1]" in execute_source
    assert 'previous_accepted_state"] = "ABSENT" if ordinal == 0 else "PRESENT"' in execute_source
    assert "previous_q, previous_base" in execute_source
    viewer_source = inspect.getsource(study.render_dev2_viewer)
    assert 'if semantic["DEV2_SEMANTIC_V1_RESULT"] == "PASS"' in viewer_source
    assert "MACHINE_PASS_HUMAN_REVIEW" in viewer_source


def test_required_solver_projection_artifacts_exist_when_empty(tmp_path: Path) -> None:
    study._write_solver_projection_csvs(tmp_path, [])
    assert (tmp_path / "solver/interaction_metrics.csv").is_file()
    assert (tmp_path / "solver/hard_validity.csv").is_file()
    assert (tmp_path / "solver/contributor_sequence.csv").is_file()
