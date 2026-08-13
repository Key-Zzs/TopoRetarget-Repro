"""CPU-only contracts for generic contact-ready RSI V2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from toporetarget.rl.physical_stage import load_p1_rsi_acceptance_contract
from toporetarget.rl.rsi.contact_ready_v2 import (
    ContactReadySamplerV2,
    GravitySafetyLabel,
    RSIStateSemanticClass,
    build_safe_bank,
    classify_contact_ready_states,
)


def _evidence(frames: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels = np.full((frames, 5), "SOURCE_NO_CONTACT", dtype="U32")
    expected = np.zeros((frames, 5), dtype=bool)
    labels[6:8, 0] = "SOURCE_CONTACT_TRANSITION"
    labels[8:, 0] = "SOURCE_CONTACT_PERSISTENT"
    expected[8:, 0] = True
    gap = np.linspace(0.05, 0.001, frames)
    twist = np.zeros((frames, 6), dtype=np.float64)
    twist[16:20, 0] = 0.1
    return labels, expected, gap, twist


def test_semantics_use_source_onset_motion_and_terminal_hold_not_a_clip_index() -> None:
    labels, expected, gap, twist = _evidence()
    result = classify_contact_ready_states(
        source_class_label=labels,
        source_expected_contact=expected,
        retargeted_geometry_gap_m=gap,
        reference_object_twist=twist,
    )

    semantic = result["semantic_class"].tolist()
    assert semantic[0] == RSIStateSemanticClass.PRE_CONTACT.value
    assert semantic[6] == RSIStateSemanticClass.NEAR_CONTACT.value
    assert semantic[8] == RSIStateSemanticClass.CONTACT_READY.value
    assert semantic[16] == RSIStateSemanticClass.MANIPULATION.value
    assert semantic[-1] == RSIStateSemanticClass.TERMINAL_HOLD.value
    assert "3cm" not in " ".join(result["classification_evidence"].tolist())


def test_conflicting_expected_contact_is_ambiguous_and_cannot_enter_a_safe_bank() -> None:
    labels, expected, gap, twist = _evidence()
    labels[10] = "SOURCE_CONTACT_TRANSITION"
    result = classify_contact_ready_states(
        source_class_label=labels,
        source_expected_contact=expected,
        retargeted_geometry_gap_m=gap,
        reference_object_twist=twist,
    )

    assert result["semantic_class"][10] == RSIStateSemanticClass.AMBIGUOUS.value


def test_safe_bank_requires_every_replica_and_sampler_excludes_risk_and_ambiguous() -> None:
    root = Path(__file__).resolve().parents[3]
    acceptance = load_p1_rsi_acceptance_contract(
        root / "configs/rl/stage16/stage16_p3_entry_gate_v1.yaml"
    )
    state_bank = {
        "runtime_index": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "semantic_class": np.asarray(
            ["CONTACT_READY", "PERSISTENT_CONTACT", "AMBIGUOUS", "MANIPULATION"], dtype="U24"
        ),
    }
    good = {
        "object_displacement_before_contact_m": 0.001,
        "object_vertical_displacement_before_contact_m": 0.0,
        "object_speed_before_contact_mps": 0.01,
        "object_angular_speed_before_contact_radps": 0.01,
        "contact_achieved": True,
        "contact_persistence_control_steps": 3,
        "joint_limit_failure": False,
        "object_drop": False,
        "catastrophic_failure": False,
        "nonfinite": False,
    }
    rows: list[dict[str, object]] = []
    for index, semantic in zip(
        state_bank["runtime_index"], state_bank["semantic_class"], strict=True
    ):
        for replica in range(4):
            row = {
                **good,
                "runtime_index": int(index),
                "replica": replica,
                "semantic_class": semantic,
            }
            if int(index) == 4 and replica == 3:
                row["object_drop"] = True
            rows.append(row)
    safe = build_safe_bank(state_bank=state_bank, diagnostic_rows=rows, acceptance=acceptance)

    assert safe["all_gravity_label"].tolist() == [
        GravitySafetyLabel.GRAVITY_SAFE.value,
        GravitySafetyLabel.GRAVITY_SAFE.value,
        GravitySafetyLabel.INVALID_RESET.value,
        GravitySafetyLabel.GRAVITY_RISK.value,
    ]
    sampler = ContactReadySamplerV2(
        runtime_index=safe["runtime_index"], safe_bank=safe["safe_bank"]
    )
    selected = sampler.sample(np.random.default_rng(11), count=12)
    assert set(selected) <= {1, 2}
    np.testing.assert_array_equal(
        selected,
        sampler.sample(np.random.default_rng(11), count=12),
    )
    with pytest.raises(ValueError, match="UNKNOWN_BANK"):
        sampler.indices(("GRAVITY_RISK",))


def test_safe_bank_marks_undiagnosed_pre_contact_as_invalid_not_missing() -> None:
    root = Path(__file__).resolve().parents[3]
    acceptance = load_p1_rsi_acceptance_contract(
        root / "configs/rl/stage16/stage16_p3_entry_gate_v1.yaml"
    )
    state_bank = {
        "runtime_index": np.asarray([0, 1], dtype=np.int64),
        "semantic_class": np.asarray(["PRE_CONTACT", "CONTACT_READY"], dtype="U24"),
    }
    diagnostic_rows = [
        {
            "runtime_index": 1,
            "semantic_class": "CONTACT_READY",
            "object_displacement_before_contact_m": 0.0,
            "object_vertical_displacement_before_contact_m": 0.0,
            "object_speed_before_contact_mps": 0.0,
            "object_angular_speed_before_contact_radps": 0.0,
            "contact_achieved": True,
            "contact_persistence_control_steps": 3,
            "joint_limit_failure": False,
            "object_drop": False,
            "catastrophic_failure": False,
            "nonfinite": False,
        }
        for _ in range(4)
    ]
    safe = build_safe_bank(
        state_bank=state_bank, diagnostic_rows=diagnostic_rows, acceptance=acceptance
    )
    assert safe["all_gravity_label"].tolist() == ["INVALID_RESET", "GRAVITY_SAFE"]
    assert safe["runtime_index"].tolist() == [1]
