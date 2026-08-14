from __future__ import annotations

import numpy as np
import pytest

from toporetarget.rl.physical_scene_rsi import (
    HO_ACTIVE_P95_M,
    HO_MAX_M,
    INTER_FINGER_MAX_M,
    TABLE_HAND_MAX_PENETRATION_M,
    TABLE_OBJECT_MAX_PENETRATION_M,
    _failure_reasons,
    _safe_bank,
    _support_state,
    evaluate_physical_pose_geometry,
)


def test_physical_gates_are_frozen_and_named() -> None:
    assert HO_ACTIVE_P95_M == 0.003
    assert HO_MAX_M == 0.010
    assert TABLE_HAND_MAX_PENETRATION_M == 0.002
    assert TABLE_OBJECT_MAX_PENETRATION_M == 0.002
    assert INTER_FINGER_MAX_M == 0.003


def test_support_state_preserves_semantic_and_finite_table_causality() -> None:
    result = _support_state(
        semantic=np.asarray(["PRE_CONTACT", "CONTACT_READY", "CONTACT_READY", "AMBIGUOUS"]),
        object_table_penetration=np.asarray([0.0, 0.0, 0.0, 0.0]),
        object_table_signed_distance=np.asarray([0.001, 0.001, 0.010, 0.001]),
        hand_table_penetration=np.asarray([0.0, 0.0, 0.0, 0.0]),
        hand_table_signed_distance=np.asarray([0.010, 0.001, 0.010, 0.001]),
    )
    assert result.tolist() == [
        "TABLE_SUPPORTED",
        "SHARED_SUPPORT",
        "AIRBORNE_OR_TRANSITION",
        "TABLE_SUPPORTED",
    ]


def test_failure_reasons_report_each_frozen_gate_without_blacklist() -> None:
    result = _failure_reasons(
        hand_object=np.asarray([0.0, 0.004]),
        hand_table=np.asarray([0.003, 0.003]),
        object_table=np.asarray([0.0, 0.003]),
        inter_finger=np.asarray([0.0, 0.004]),
    )
    assert result[0] == "H_T_GT_2MM"
    assert result[1] == ("H_O_ACTIVE_P95_GT_3MM|H_T_GT_2MM|O_T_GT_2MM|INTER_FINGER_GT_3MM")


def test_safe_bank_selects_physical_states_and_does_not_encode_blacklist() -> None:
    bank = {
        "runtime_index": np.arange(4),
        "source_index_or_interval": np.arange(4),
        "semantic_class": np.asarray(["PRE_CONTACT", "NEAR_CONTACT", "AMBIGUOUS", "TERMINAL_HOLD"]),
        "source_expected_contact": np.asarray([False, True, True, True]),
        "classification_confidence": np.asarray(["MEDIUM"] * 4),
        "classification_evidence": np.asarray(["{}"] * 4),
        "retargeted_geometry_gap_m": np.zeros(4),
    }
    validity = {
        "overall_reference_geometry_valid": np.asarray([True, True, True, False]),
        "support_state": np.asarray(
            ["TABLE_SUPPORTED", "SHARED_SUPPORT", "TABLE_SUPPORTED", "TABLE_SUPPORTED"]
        ),
        "reference_geometry_failure_reason": np.asarray(["", "", "", "H_O_MAX_GE_10MM"]),
    }
    result = _safe_bank(bank=bank, validity=validity, coverage_minimum_pre_contact=1)
    assert result["runtime_index"].tolist() == [0, 1]
    assert result["physical_safe_bank"].tolist() == [
        "PRE_CONTACT_TABLE_SUPPORTED_SAFE",
        "NEAR_CONTACT_PHYSICAL_SAFE",
    ]
    assert not any("blacklist" in key.lower() for key in result)
    assert '"pre_contact_table_supported_pass": true' in str(result["coverage_gate_json"].item())


def test_trace_geometry_rejects_ambiguous_shapes_before_runtime_assets() -> None:
    with pytest.raises(ValueError, match="PHYSICAL_TRACE_HAND_POSE_SHAPE_INVALID"):
        evaluate_physical_pose_geometry(
            clip="hocap_170105",
            wrist_pose=np.zeros((2, 7)),
            finger_q=np.zeros((3, 20)),
            object_pose=np.zeros((2, 7)),
            geometry_manifest_path=None,  # type: ignore[arg-type]
            table_proxy_path=None,  # type: ignore[arg-type]
            repo_root=None,  # type: ignore[arg-type]
        )
