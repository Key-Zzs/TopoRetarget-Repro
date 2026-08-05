from __future__ import annotations

from dataclasses import replace

import pytest

from toporetarget.rl.environments.isaaclab_backend.physx_contract import candidate_matrix


def test_r2_candidate_matrix_is_closed_and_has_one_cpu_diagnostic() -> None:
    matrix = candidate_matrix()
    assert tuple(matrix) == ("G0", "G1", "G2", "G3", "G4", "G5", "C0")
    assert sum(item.device_kind == "gpu" for item in matrix.values()) == 6
    assert sum(item.device_kind == "cpu" for item in matrix.values()) == 1
    assert matrix["G0"].config_hash != matrix["G1"].config_hash


def test_r2_contract_rejects_invalid_solver_or_scene_construction() -> None:
    baseline = candidate_matrix()["G0"]
    with pytest.raises(ValueError, match="solver type"):
        replace(baseline, solver_type=7)
    with pytest.raises(ValueError, match="scene construction"):
        replace(baseline, scene_construction="runtime_rebuild")
