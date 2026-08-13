"""CPU-only candidate-pool allocation and isolation contracts."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from toporetarget.rl.isaaclab_oracle.candidate_pool import PhysXOracleCandidatePoolV1


def _env(count: int) -> SimpleNamespace:
    origins = torch.stack((torch.arange(count), torch.zeros(count), torch.zeros(count)), dim=-1)
    return SimpleNamespace(
        num_envs=count, device=torch.device("cpu"), scene=SimpleNamespace(env_origins=origins)
    )


def test_candidate_pool_allocates_disjoint_96_and_144_capacity() -> None:
    pool_96 = PhysXOracleCandidatePoolV1(_env(97), candidate_count=96)
    pool_144 = PhysXOracleCandidatePoolV1(_env(145), candidate_count=144)
    assert pool_96.validate_layout()["unique_origins"]
    assert pool_96.layout.population_per_horizon == 32
    assert pool_144.validate_layout()["candidate_execution_disjoint"]
    assert pool_144.layout.population_per_horizon == 48


def test_candidate_pool_refuses_default_4096_candidate_misuse() -> None:
    try:
        PhysXOracleCandidatePoolV1(_env(4097), candidate_count=4096)
    except ValueError as error:
        assert "1, 32, 96, or 144" in str(error)
    else:
        raise AssertionError("C.5A accepted 4096 as candidate capacity")
