from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts.data import run_oakink2_o5rd2g as workflow
from toporetarget.retarget.objective_v3_execution import (
    ExecutionFrameInputsV3,
    ExecutionInputAuthorityError,
    RetargetMode,
)
from toporetarget.utils.hashing import sha256_file


def test_frozen_scientific_contract_hashes_remain_exact() -> None:
    paths = workflow.frozen_paths()
    assert sha256_file(paths["objective_v2"]) == workflow.OBJECTIVE_SHA
    assert sha256_file(paths["execution_v2"]) == workflow.EXECUTION_V2_SHA
    assert sha256_file(paths["gate_v2"]) == workflow.GATE_V2_SHA


def test_dev2_literals_are_identity_only_not_algorithm_branches() -> None:
    source = inspect.getsource(workflow.search_cold_start_frame)
    for forbidden in ("C11001", "10704", "scene_01__A003"):
        assert forbidden not in source
    assert workflow._special_case_audit()["DEV2_SPECIAL_CASE_ADDED"] == "NO"


def test_cold_start_rejects_synthetic_qold_before_optimizer() -> None:
    with pytest.raises(ExecutionInputAuthorityError, match="COLD_START_FORBIDS"):
        ExecutionFrameInputsV3(
            mode=RetargetMode.COLD_START,
            runtime_step_index=0,
            old_production_q=np.zeros(20),
            previous_accepted_q=None,
            previous_accepted_base=None,
        ).validate()


def test_fail_closed_stage_order_rejects_missing_authority(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        workflow.run_refinement_regression(tmp_path)


def test_cli_has_every_contract_action() -> None:
    required = {
        "preflight",
        "verify-frozen-upstream",
        "audit-execution-input-authority",
        "audit-qold-role",
        "audit-interaction-graph-authority",
        "audit-dev2-input-completeness",
        "run-graph-parity",
        "develop-execution-v3-candidates",
        "run-masked-qold-development",
        "run-refinement-regression",
        "select-execution-v3",
        "run-dev2-frame0-development",
        "freeze-execution-input-authority",
        "freeze-coldstart-seed-authority",
        "freeze-interaction-graph-authority",
        "freeze-execution-v3",
        "generate-future-certification-plan",
        "summarize",
    }
    assert required <= set(workflow.ACTIONS)


def test_no_downstream_execution_action_exists() -> None:
    forbidden = ("sparse-v4", "window-v4", "dev2-full", "physx", "ppo", "support")
    assert not any(any(token in action for token in forbidden) for action in workflow.ACTIONS)
