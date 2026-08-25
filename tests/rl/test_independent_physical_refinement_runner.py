from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/rl/isaaclab/run_independent_physical_refinement.py"
)


def _module() -> object:
    spec = importlib.util.spec_from_file_location("independent_physical_refinement_runner", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finalize_existing_validates_a_complete_nonresumable_u15_lineage() -> None:
    runner = _module()
    complete = {
        "schema_version": "PhysicalRefinementTrainingV1",
        "clip": "hocap_example",
        "max_new_updates": 15,
        "actual_new_updates": 2,
        "best_checkpoint": {"checkpoint": "checkpoint.pt"},
    }
    progression = [{"new_update": "1"}, {"new_update": "2"}]

    assert runner._validate_complete(complete, progression, clip_id="hocap_example") == 2


def test_finalize_existing_rejects_incomplete_or_misaligned_progression() -> None:
    runner = _module()
    complete = {
        "schema_version": "PhysicalRefinementTrainingV1",
        "clip": "hocap_example",
        "max_new_updates": 15,
        "actual_new_updates": 2,
        "best_checkpoint": {"checkpoint": "checkpoint.pt"},
    }

    with pytest.raises(RuntimeError, match="RESULT_CONTRACT_INVALID"):
        runner._validate_complete(complete, [{"new_update": "2"}], clip_id="hocap_example")
