from __future__ import annotations

import pytest

from toporetarget.rl.isaaclab_oracle.recovery import Stage16C5AR2RecoveryStateMachine


def test_r2_recovery_bounds_candidates_retries_and_transitions() -> None:
    machine = Stage16C5AR2RecoveryStateMachine()
    machine.transition("P0_CURRENT_CONTRACT_AUDIT", reason="audit")
    for candidate_id in ("G0", "G1", "G2", "G3", "G4", "G5"):
        machine.start_candidate(candidate_id, device_kind="gpu")
    with pytest.raises(RuntimeError, match="GPU_CANDIDATE_BUDGET"):
        machine.start_candidate("G6", device_kind="gpu")
    machine.transition("S2_CONTACT_ONSET", reason="candidate")
    machine.record_failure("PHYSX_NONDETERMINISM")
    machine.record_failure("PHYSX_NONDETERMINISM")
    with pytest.raises(RuntimeError, match="STAGE_RETRY_BUDGET"):
        machine.record_failure("PHYSX_NONDETERMINISM")


def test_r2_recovery_fail_closes_on_runtime_fallback() -> None:
    machine = Stage16C5AR2RecoveryStateMachine()
    machine.fail_closed("RUNTIME_CONFIG_FALLBACK", reason="USD attr mismatch")
    assert machine.phase == "CLOSEOUT"
    assert machine.terminal_failure == "RUNTIME_CONFIG_FALLBACK"
    with pytest.raises(RuntimeError, match="FAIL_CLOSED"):
        machine.transition("S0_SMOKE", reason="forbidden")
