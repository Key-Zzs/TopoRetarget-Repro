"""P0 configuration loads the frozen parent interfaces and rejects drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from toporetarget.rl.physical_stage import (
    EARTH_NOMINAL_GRAVITY,
    Stage16PhysicalBootstrapContractV1,
    load_p1_rsi_acceptance_contract,
    load_p3_entry_gate,
    load_physical_bootstrap_contract,
)


def test_physical_bootstrap_and_p3_gate_load() -> None:
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/rl/stage16/stage16_physical_bootstrap.yaml"
    gate = root / "configs/rl/stage16/stage16_p3_entry_gate_v1.yaml"

    bootstrap = load_physical_bootstrap_contract(config)
    loaded_gate = load_p3_entry_gate(gate)
    acceptance = load_p1_rsi_acceptance_contract(gate)

    assert bootstrap.target_gravity_world_mps2 == EARTH_NOMINAL_GRAVITY
    assert bootstrap.external_guidance is False
    assert loaded_gate["gates"]["G5_causality"]["rollout_object_state_writes"] == 0
    assert acceptance.replicas_per_state == 4


def test_invalid_physical_target_fails_fast() -> None:
    with pytest.raises(ValueError, match="TARGET_GRAVITY_INVALID"):
        Stage16PhysicalBootstrapContractV1(target_gravity_world_mps2=(0.0, 0.0, 0.0))
