from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts.evaluation import run_oakink2_o5rd3 as study
from toporetarget.retarget.objective_v3_execution import (
    ExecutionFrameInputsV3,
    ExecutionInputAuthorityError,
    RetargetMode,
)


def test_cli_exposes_required_d3_actions() -> None:
    required = {
        "preflight",
        "record-dev2-approval",
        "verify-dev2-acceptance",
        "verify-frozen-refinement-authorities",
        "verify-dev1-identity",
        "verify-old-production-trajectory",
        "audit-refinement-consumer-inputs",
        "build-refinement-input-authority",
        "run-dev1-refinement-preflight",
        "freeze-d3-run",
        "run-dev1-full-refinement",
        "resume-dev1-full-refinement",
        "verify-dev1-runtime-chain",
        "finalize-dev1-trajectory",
        "compare-old-vs-new",
        "run-dev1-semantic-v1",
        "render-dev1-viewer",
        "audit-dev1-special-cases",
        "summarize",
    }
    assert required <= set(study.ACTIONS)


def test_dev1_identity_and_frame_count_are_frozen() -> None:
    assert study.EPISODE.endswith(":00001")
    assert study.PRIMITIVE == "pour"
    assert study.OBJECT_ID == "C10001"
    assert (study.SOURCE_START, study.SOURCE_STOP) == (1969, 4691)
    assert study.EXPECTED_FRAMES == 2722
    assert study.SOURCE_FRAMES == list(range(1969, 4691))


def test_refinement_requires_q_old_and_separate_previous_state() -> None:
    q = np.zeros(20)
    base = np.eye(4)
    frame0 = ExecutionFrameInputsV3(
        mode=RetargetMode.REFINEMENT,
        runtime_step_index=0,
        old_production_q=q,
        previous_accepted_q=None,
        previous_accepted_base=None,
    ).validate()
    assert frame0.old_production_q is q
    continuation = ExecutionFrameInputsV3(
        mode=RetargetMode.REFINEMENT,
        runtime_step_index=1,
        old_production_q=q + 1.0,
        previous_accepted_q=q,
        previous_accepted_base=base,
    ).validate()
    assert not np.array_equal(continuation.old_production_q, continuation.previous_accepted_q)
    with pytest.raises(ExecutionInputAuthorityError):
        ExecutionFrameInputsV3(
            mode=RetargetMode.REFINEMENT,
            runtime_step_index=1,
            old_production_q=None,
            previous_accepted_q=q,
            previous_accepted_base=base,
        ).validate()


def test_cold_start_is_not_reachable_from_d3_executor() -> None:
    source = inspect.getsource(study._execute)
    assert "search_cold_start" not in source
    assert "_run_refinement_v3" in source
    assert "RetargetMode.REFINEMENT" in source


def test_inventory_is_complete_and_has_no_singleton_indexed_carrier() -> None:
    inventory = study._refinement_inventory()
    roles = {row["authority_role"] for row in inventory}
    assert "OLD_PRODUCTION_FRAME_INDEXED" in roles
    assert "PREVIOUS_ACCEPTED_RUNTIME" in roles
    assert all(row["authority_role"] != "UNKNOWN" for row in inventory)
    assert not [
        row
        for row in inventory
        if row["authority_role"] in {"SOURCE_FRAME_INDEXED", "OLD_PRODUCTION_FRAME_INDEXED"}
        and row["actual_length"] == 1
    ]


def test_scientific_failure_and_second_run_are_fail_closed() -> None:
    source = inspect.getsource(study._execute)
    assert "SCIENTIFIC_RUN_COUNT_ALREADY_ONE" in source
    assert '"resume_allowed": False' in source
    assert "HARD_VALIDITY_FAILURE" in source


def test_semantic_and_viewer_require_complete_trajectory() -> None:
    semantic = inspect.getsource(study.run_dev1_semantic_v1)
    viewer = inspect.getsource(study.render_dev1_viewer)
    assert "trajectory/finalization.json" in semantic
    assert "COVERAGE_MISMATCH" in semantic
    assert "trajectory/finalization.json" in viewer
    assert "old_wuji_parts" in viewer
    assert '"old_wuji": True' in viewer
    assert '"new_wuji": True' in viewer


def test_forbidden_downstream_actions_are_absent() -> None:
    forbidden = {"ppo", "physx", "o6", "certification", "heldout", "dev2-rerun"}
    assert not (forbidden & {name.lower() for name in study.ACTIONS})
