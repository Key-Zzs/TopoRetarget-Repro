from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd2mr2 as study


def test_cli_exposes_complete_fail_closed_contract() -> None:
    required = {
        "preflight",
        "verify-historical-runs",
        "verify-frozen-v4",
        "verify-full-graph-authority",
        "audit-consumer-callgraph",
        "inventory-ordinal-indexed-inputs",
        "classify-consumer-input-authorities",
        "audit-warm-authority",
        "repair-consumer-input-binding",
        "build-consumer-input-authority",
        "run-offline-240-preflight",
        "run-frame1-context-canary",
        "run-multi-ordinal-preflight",
        "run-input-regression-tests",
        "audit-repair-impact",
        "freeze-consumer-input-authority",
        "authorize-dev2-v3",
        "generate-d2n-plan",
        "summarize",
    }
    assert required <= study.ACTIONS.keys()


def test_unknown_ordinal_input_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="UNKNOWN_ORDINAL_INDEXED_INPUT"):
        study.validate_registered_inputs(
            [{"field": "new_carrier", "role": "UNKNOWN", "ordinal_indexed": True}]
        )


def test_trajectory_static_cannot_be_ordinal_indexed() -> None:
    with pytest.raises(RuntimeError, match="TRAJECTORY_STATIC_CANNOT_BE_ORDINAL_INDEXED"):
        study.validate_registered_inputs(
            [{"field": "mesh", "role": "TRAJECTORY_STATIC", "ordinal_indexed": True, "length": 1}]
        )


@pytest.mark.parametrize("count", [1, 239, 241])
def test_source_frame_indexed_requires_exact_240_coverage(count: int) -> None:
    ids = study.SOURCE_FRAMES[:count]
    if count > 240:
        ids = study.SOURCE_FRAMES + [study.SOURCE_STOP]
    with pytest.raises(RuntimeError, match="FAIL_FULL_SEQUENCE_INPUT_COVERAGE"):
        study.validate_registered_inputs(
            [
                {
                    "field": "graph",
                    "role": "SOURCE_FRAME_INDEXED",
                    "ordinal_indexed": True,
                    "length": count,
                    "frame_ids": ids,
                }
            ]
        )


def test_shifted_frame_ids_are_rejected() -> None:
    with pytest.raises(RuntimeError, match="FAIL_FULL_SEQUENCE_INPUT_COVERAGE"):
        study.validate_registered_inputs(
            [
                {
                    "field": "pose",
                    "role": "SOURCE_FRAME_INDEXED",
                    "ordinal_indexed": True,
                    "length": 240,
                    "frame_ids": study.SOURCE_FRAMES[1:] + [study.SOURCE_STOP],
                }
            ]
        )


def test_frame0_initializer_cannot_be_read_at_t_gt_0() -> None:
    with pytest.raises(RuntimeError, match="FRAME0_ONLY_INITIALIZER_ACCESSED_AT_T_GT_0"):
        study.validate_registered_inputs(
            [
                {
                    "field": "warm.qpos",
                    "role": "FRAME0_ONLY_INITIALIZER",
                    "ordinal_indexed": False,
                    "length": 1,
                    "max_accessed_ordinal": 1,
                }
            ]
        )


def test_previous_runtime_absence_fails_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="PREVIOUS_RUNTIME_STATE_REQUIRED"):
        study.require_previous_runtime(1, None)


def test_repaired_binding_never_reads_warm_or_q_old() -> None:
    source = inspect.getsource(study.FullSequenceV3Runtime.bind_context)
    assert "self.warm" not in source
    assert "q_old" not in source
    assert "PREVIOUS_RUNTIME_STATE_REQUIRED" in source
    assert "seed_base=seed_base" in source
    assert "seed_qpos=neutral" in source


def test_canary_implementation_cannot_launch_optimizer() -> None:
    source = inspect.getsource(study.run_frame1_context_canary)
    assert "search_cold_start_v2_frame(" not in source
    assert "search_cold_start_v4_from_v3(" not in source
    assert ".minimize(" not in source


def test_execution_v4_four_frozen_hashes_are_exact() -> None:
    assert {
        key: study.sha256_file(path)
        for key, (path, _expected) in study.d2m.FROZEN_AUTHORITIES.items()
        if key in study.d2m.V4_AUTHORITY_HASHES
    } == study.d2m.V4_AUTHORITY_HASHES


def test_graph_authority_is_still_exact_240() -> None:
    authority = study.read_json(study.d2mr.ROOT / "graph_authority/sequence_authority.json")
    assert (
        study.sha256_file(study.d2mr.ROOT / "graph_authority/sequence_authority.json")
        == study.GRAPH_AUTHORITY_SHA
    )
    assert authority["source_frame_ids"] == study.SOURCE_FRAMES


def test_v1_v2_history_remains_immutable() -> None:
    assert study.read_json(study.V1_ROOT / "final_summary.json")["DEV2_RUN_UUID"] == study.V1_UUID
    assert (
        study.read_json(study.d2mr.ROOT / "final_summary.json")["DEV2_V2_RUN_UUID"] == study.V2_UUID
    )
    assert {
        rel: study.sha256_file(study.d2mr.ROOT / rel) for rel in study.V2_IMMUTABLE_HASHES
    } == study.V2_IMMUTABLE_HASHES


def test_v3_authorization_code_does_not_run_v3() -> None:
    source = inspect.getsource(study.authorize_dev2_v3)
    assert "search_cold_start" not in source
    assert '"DEV2_FULL_RECOVERY_V3_SCIENTIFIC_RUN_COUNT": 0' in source


def test_no_blind_warm_repeat_or_qold_synthesis() -> None:
    source = Path(study.__file__).read_text(encoding="utf-8")
    assert "np.repeat" not in source
    assert 'Q_OLD_SYNTHESIZED_FROM_WARM": "NO' in source


def test_future_canaries_are_schema_only() -> None:
    source = inspect.getsource(study.run_multi_ordinal_preflight)
    assert "NO_FUTURE_STATE_INVENTED" in source
    assert "build_runtime()" not in source
