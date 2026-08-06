from __future__ import annotations

import pytest

from toporetarget.rl.geometry_audit.attainability import (
    ContactPreservingFeasibilityV1,
    StableDynamicContactCalibrationV1,
    decide_penetration_gate_contract,
)
from toporetarget.rl.geometry_audit.contracts import (
    GEOMETRY_METRIC_CONTRACT,
    GEOMETRY_QUERY_CONTRACT,
)


def _stable(
    provenance: str = "generic stable runtime contact calibration",
) -> StableDynamicContactCalibrationV1:
    return StableDynamicContactCalibrationV1(
        experiment_id="stable_contact_v1",
        provenance=provenance,
        replicas=20,
        control_steps=100,
        free_object=True,
        formal_state_writes=0,
        required_contact_present_rate=1.0,
        replica_p99_max_penetration_m=0.0008,
        pooled_active_p95_penetration_m=0.0004,
        max_penetration_m=0.001,
        bounded_normal_load=True,
    )


def _local(clip: str, *, attainable: bool) -> ContactPreservingFeasibilityV1:
    return ContactPreservingFeasibilityV1(
        clip=clip,
        free_object=True,
        formal_state_writes=0,
        required_topology_pass=True,
        semantic_intent_pass=True,
        v1_max_limit_m=0.0002,
        v1_active_p95_limit_m=0.0002,
        best_contact_preserving_max_m=0.0008,
        best_contact_preserving_active_p95_m=0.0004,
        v1_pass_with_required_contact=attainable,
        lower_penetration_only_by_contact_or_task_degeneracy=not attainable,
    )


def test_v2_decision_preserves_absolute_gates_and_shared_floor() -> None:
    result = decide_penetration_gate_contract(
        query_contract=GEOMETRY_QUERY_CONTRACT,
        v1_contract=GEOMETRY_METRIC_CONTRACT,
        numerical_query_p99_m=1.0e-7,
        no_contact_max_penetration_m=0.0,
        stable_contact=_stable(),
        local_feasibility=(
            _local("hocap_170105", attainable=False),
            _local("hocap_170650", attainable=False),
        ),
    )
    assert result["status"] == "STAGE16D_GEOMETRY_V2_VALIDATED"
    contract = result["metric_contract"]
    assert contract["strict_catastrophic_max_m"] == 0.010
    assert contract["maximum_p95_m"] == 0.003
    assert not contract["clip_specific_thresholds"]


def test_v1_is_kept_when_contact_preserving_feasibility_passes() -> None:
    result = decide_penetration_gate_contract(
        query_contract=GEOMETRY_QUERY_CONTRACT,
        v1_contract=GEOMETRY_METRIC_CONTRACT,
        numerical_query_p99_m=1.0e-7,
        no_contact_max_penetration_m=0.0,
        stable_contact=_stable(),
        local_feasibility=(
            _local("hocap_170105", attainable=True),
            _local("hocap_170650", attainable=True),
        ),
    )
    assert result["status"] == "STAGE16D_GEOMETRY_V1_ATTAINABLE"
    assert not result["v2_created"]


def test_corrected_candidate_cannot_define_dynamic_floor() -> None:
    with pytest.raises(ValueError, match="cannot use corrected"):
        _stable("corrected candidate formal20 result")
