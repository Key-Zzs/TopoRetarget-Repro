from __future__ import annotations

from copy import deepcopy

import numpy as np

from toporetarget.evaluation.physical_functionality_full_cycle_v1 import (
    PhysicalFunctionalityFullCycleV1Contract,
    evaluate_physical_functionality_full_cycle_v1,
)
from toporetarget.evaluation.stage16_pf_v2_causal_lift import (
    Stage16PhysicalFunctionalityV2Contract,
    evaluate_stage16_physical_functionality_v2,
)


def _successful_trace() -> dict[str, object]:
    count = 30
    object_pose = np.zeros((count, 7), dtype=np.float64)
    object_pose[:, 3] = 1.0
    object_pose[:, 2] = np.asarray(
        [0.0, 0.0, 0.0, 0.01, 0.03, 0.05, 0.07, 0.09] + [0.09] * (count - 8)
    )
    object_pose[9:14, 0] = np.linspace(0.02, 0.12, 5)
    object_pose[14:, 0] = 0.12

    wrist_pose = object_pose.copy()
    wrist_pose[:, 2] += 0.02
    wrist_pose[18:, 0] = object_pose[18:, 0] + np.linspace(0.02, 0.20, count - 18)

    contact = np.zeros(count, dtype=bool)
    contact[2:18] = True
    tips = np.zeros((count, 5), dtype=bool)
    tips[:, :2] = contact[:, None]
    hand = np.zeros((count, 21), dtype=bool)
    hand[:, 0] = contact
    table = np.zeros(count, dtype=bool)
    table[:4] = True
    destination_region = np.zeros(count, dtype=bool)
    destination_region[13:] = True
    destination_support = np.zeros(count, dtype=bool)
    destination_support[14:] = True
    return {
        "object_pose_wxyz": object_pose,
        "wrist_pose_wxyz": wrist_pose,
        "tip_pair_presence": tips,
        "hand_object_pair_presence": hand,
        "table_object_contact": table,
        "destination_region": destination_region,
        "destination_support_contact": destination_support,
        "interaction_valid": np.ones(count, dtype=bool),
        "support_valid": np.ones(count, dtype=bool),
        "destination_region_valid": np.ones(count, dtype=bool),
        "destination_support_valid": np.ones(count, dtype=bool),
        "reference_lift_onset": 4,
        "reference_events": {
            "source_contact": 2,
            "persistent_contact": 2,
            "pickup": 4,
            "place": 12,
            "release": 17,
        },
        "causal_execution": True,
        "geometry_safe": True,
        "action_bounds_safe": True,
        "no_hidden_control": True,
    }


def _evaluate(trace: dict[str, object]) -> dict[str, object]:
    return evaluate_physical_functionality_full_cycle_v1(**trace)


def test_complete_pick_place_release_retreat_passes() -> None:
    result = _evaluate(_successful_trace())
    assert result["pf_full_cycle"] is True
    assert [
        result[name]["status"]
        for name in ("PF_pick", "PF_transport", "PF_place", "PF_release", "PF_retreat")
    ] == ["PASS"] * 5
    assert result["PF_place"]["diagnostics"]["is_exact_support_force"] is False


def test_pick_succeeds_but_transport_loses_object() -> None:
    trace = _successful_trace()
    trace["hand_object_pair_presence"][10:13] = False
    result = _evaluate(trace)
    assert result["PF_pick"]["status"] == "PASS"
    assert result["PF_transport"]["status"] == "FAIL"
    assert "hand_object_coupling_lost" in result["PF_transport"]["failure_reasons"]
    assert result["PF_place"]["status"] == "NOT_REACHED"
    assert result["PF_release"]["status"] == "NOT_REACHED"
    assert result["PF_retreat"]["status"] == "NOT_REACHED"


def test_transport_succeeds_but_place_fails() -> None:
    trace = _successful_trace()
    trace["destination_support_contact"][:] = False
    result = _evaluate(trace)
    assert result["PF_transport"]["status"] == "PASS"
    assert result["PF_place"]["status"] == "FAIL"
    assert "destination_support_never_acquired" in result["PF_place"]["failure_reasons"]
    assert result["PF_release"]["status"] == "NOT_REACHED"


def test_place_succeeds_but_release_fails() -> None:
    trace = _successful_trace()
    trace["hand_object_pair_presence"][18:, 0] = True
    trace["tip_pair_presence"][18:, :2] = True
    result = _evaluate(trace)
    assert result["PF_place"]["status"] == "PASS"
    assert result["PF_release"]["status"] == "FAIL"
    assert "hand_object_contact_did_not_release" in result["PF_release"]["failure_reasons"]
    assert result["PF_retreat"]["status"] == "NOT_REACHED"


def test_release_succeeds_but_retreat_disturbs_object() -> None:
    trace = _successful_trace()
    trace["object_pose_wxyz"][23:, 1] += 0.04
    result = _evaluate(trace)
    assert result["PF_release"]["status"] == "PASS"
    assert result["PF_retreat"]["status"] == "FAIL"
    assert "retreat_disturbed_object_translation" in result["PF_retreat"]["failure_reasons"]


def test_ballistic_object_motion_is_rejected() -> None:
    trace = _successful_trace()
    trace["tip_pair_presence"][7:] = False
    trace["hand_object_pair_presence"][7:] = False
    result = _evaluate(trace)
    assert result["PF_pick"]["status"] == "FAIL"
    assert result["PF_transport"]["status"] == "NOT_REACHED"
    assert result["pf_full_cycle"] is False


def test_teleported_object_is_rejected_during_transport() -> None:
    trace = _successful_trace()
    trace["object_pose_wxyz"][11:, 0] += 0.10
    trace["wrist_pose_wxyz"][11:, 0] += 0.10
    result = _evaluate(trace)
    assert result["PF_pick"]["status"] == "PASS"
    assert result["PF_transport"]["status"] == "FAIL"
    assert "object_translation_discontinuity_teleport" in result["PF_transport"]["failure_reasons"]


def test_support_never_transfers_is_a_place_failure() -> None:
    trace = _successful_trace()
    trace["destination_support_contact"][:] = False
    result = _evaluate(trace)
    assert result["PF_transport"]["status"] == "PASS"
    assert result["PF_place"]["status"] == "FAIL"
    assert result["PF_release"]["status"] == "NOT_REACHED"


def test_unavailable_support_is_identifiable_separately_from_not_reached() -> None:
    trace = _successful_trace()
    trace["destination_support_valid"][:] = False
    result = _evaluate(trace)
    assert result["PF_place"]["status"] == "NOT_IDENTIFIABLE"
    assert result["PF_release"]["status"] == "NOT_REACHED"
    assert result["PF_retreat"]["status"] == "NOT_REACHED"


def test_interaction_timing_is_diagnostic_only() -> None:
    trace = _successful_trace()
    trace["reference_lift_onset"] = 25
    trace["reference_events"] = {
        "source_contact": 25,
        "persistent_contact": 25,
        "pickup": 25,
        "place": 25,
        "release": 25,
    }
    result = _evaluate(trace)
    assert result["pf_full_cycle"] is True
    assert result["DF_interaction_timing"]["included_in_pf_hard_gate"] is False
    assert result["DF_interaction_timing"]["pickup_timing"]["delta_control_steps"] != 0


def test_pf_pick_is_exact_frozen_v2_parity() -> None:
    trace = _successful_trace()
    standalone = evaluate_stage16_physical_functionality_v2(
        object_pose_wxyz=trace["object_pose_wxyz"],
        wrist_pose_wxyz=trace["wrist_pose_wxyz"],
        tip_pair_presence=trace["tip_pair_presence"],
        hand_object_pair_presence=trace["hand_object_pair_presence"],
        table_object_contact=trace["table_object_contact"],
        interaction_valid=trace["interaction_valid"],
        support_valid=trace["support_valid"],
        reference_lift_onset=trace["reference_lift_onset"],
        causal_execution=True,
        geometry_safe=True,
        action_bounds_safe=True,
        no_hidden_control=True,
    )
    full_cycle = _evaluate(trace)
    assert full_cycle["PF_pick"]["passed"] == standalone["pf_v2"]
    assert full_cycle["PF_pick"]["failure_reasons"] == standalone["pf_v2_failure_reasons"]
    assert full_cycle["PF_pick"]["diagnostics"]["detail"] == standalone


def test_contract_construction_does_not_mutate_pf_v2_contract() -> None:
    before = deepcopy(Stage16PhysicalFunctionalityV2Contract().as_dict())
    full_cycle = PhysicalFunctionalityFullCycleV1Contract()
    after = Stage16PhysicalFunctionalityV2Contract().as_dict()
    assert before == after
    assert full_cycle.pick_authority == before["schema_version"]
