from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd2m as v1
from scripts.evaluation import run_oakink2_o5rd2mr as study


def test_cli_exposes_required_d2mr_and_v2_actions() -> None:
    expected = {
        "preflight",
        "verify-d2m-v1-history",
        "audit-dev2-graph-coverage",
        "audit-graph-construction-authority",
        "build-dev2-graph-sequence",
        "verify-frame0-graph-parity",
        "verify-graph-sequence-coverage",
        "verify-graph-determinism",
        "freeze-dev2-graph-authority",
        "audit-repair-impact",
        "run-full-sequence-preflight-regression",
        "freeze-d2mr",
        "freeze-dev2-v2-run",
        "run-dev2-v2-full",
        "resume-dev2-v2",
        "verify-dev2-v2-runtime-chain",
        "verify-dev2-v2-graph-binding",
        "finalize-dev2-v2-trajectory",
        "run-dev2-v2-semantic-v1",
        "render-dev2-v2-viewer",
        "summarize",
    }
    assert expected <= study.ACTIONS.keys()


@pytest.mark.parametrize("count", [1, 239, 241])
def test_wrong_graph_count_rejected_pre_optimizer(count: int) -> None:
    with pytest.raises(RuntimeError, match="FULL_SEQUENCE_GRAPH_COVERAGE_MISMATCH"):
        study._validate_graph_binding(
            study.SOURCE_FRAMES,
            study.SOURCE_FRAMES[:count]
            if count <= study.EXPECTED_FRAMES
            else [*study.SOURCE_FRAMES, study.SOURCE_STOP],
            study.OBJECT_ID,
            study.OBJECT_ID,
        )


def test_shift_duplicate_missing_and_wrong_object_rejected() -> None:
    shifted = [study.SOURCE_START + 1, *study.SOURCE_FRAMES[1:]]
    duplicate = [study.SOURCE_START, study.SOURCE_START, *study.SOURCE_FRAMES[2:]]
    missing = [*study.SOURCE_FRAMES[:50], *study.SOURCE_FRAMES[51:], study.SOURCE_STOP]
    for values in (shifted, duplicate, missing):
        with pytest.raises(RuntimeError, match="FULL_SEQUENCE_GRAPH_"):
            study._validate_graph_binding(
                study.SOURCE_FRAMES, values, study.OBJECT_ID, study.OBJECT_ID
            )
    with pytest.raises(RuntimeError, match="FULL_SEQUENCE_GRAPH_OBJECT_MISMATCH"):
        study._validate_graph_binding(
            study.SOURCE_FRAMES, study.SOURCE_FRAMES, study.OBJECT_ID, "WRONG"
        )


def test_exact_240_frame_binding_passes() -> None:
    study._validate_graph_binding(
        study.SOURCE_FRAMES, study.SOURCE_FRAMES, study.OBJECT_ID, study.OBJECT_ID
    )


def test_determinism_subset_is_preregistered_from_interval() -> None:
    assert study._determinism_ordinals() == [0, 60, 120, 180, 239]
    assert [study.SOURCE_FRAMES[index] for index in study._determinism_ordinals()] == [
        10704,
        10764,
        10824,
        10884,
        10943,
    ]


def test_graph_builder_is_source_only_and_does_not_read_retarget_outputs() -> None:
    source = inspect.getsource(study.build_dev2_graph_sequence)
    assert "build_source_interaction_graph" in source
    assert "frame_indices=list(range(EXPECTED_FRAMES))" in source
    assert "search_cold_start" not in source
    assert "qpos" not in source
    assert "E_IM" not in source


def test_v1_uuid_cannot_be_configured_as_v2(tmp_path: Path) -> None:
    v1.atomic_write_json(
        tmp_path / "d2mv2/run_authority/full_run_manifest.json",
        {"RUN_UUID": study.V1_RUN_UUID},
    )
    with pytest.raises(RuntimeError, match="V1_RUN_UUID_CANNOT_RESUME_AS_V2"):
        study._configure_v2(tmp_path)


def test_frozen_execution_v4_hashes_are_still_exact() -> None:
    for name in v1.V4_AUTHORITY_HASHES:
        path, expected = v1.FROZEN_AUTHORITIES[name]
        assert v1.sha256_file(path) == expected == v1.V4_AUTHORITY_HASHES[name]


def test_v2_execution_preserves_qold_absence_and_sequential_previous_state() -> None:
    source = inspect.getsource(v1._execute_dev2)
    assert '"q_old_access_count": 0' in source
    assert "previous_q, previous_base" in source
    assert "SOURCE_GRAPH_FRAME_BINDING_FAIL" in source
    assert "graph_entry_hash" in source


def test_semantic_v1_still_requires_240_complete_frames(tmp_path: Path) -> None:
    v1.atomic_write_json(
        tmp_path / "solver/result.json", {"status": "PASS", "COMPLETED_FRAMES": 239}
    )
    with pytest.raises(RuntimeError):
        v1.run_semantic_v1(tmp_path)
