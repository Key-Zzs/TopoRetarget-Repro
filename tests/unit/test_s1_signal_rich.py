from __future__ import annotations

import numpy as np

from toporetarget.data.adapters import GrabDatasetAdapter
from toporetarget.data.readers import GrabSequenceRecord
from toporetarget.workflows.s1_signal_rich import (
    _active_gate,
    _backend_metrics,
    _continuity,
    _stratified,
)


def test_signal_rich_reader_and_adapter_exports_are_lazy_import_safe() -> None:
    assert GrabDatasetAdapter is not None
    assert GrabSequenceRecord is not None


def test_continuity_is_finite_and_deterministic() -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    first = _continuity(values, 120.0)
    second = _continuity(values, 120.0)
    assert first == second
    assert first["finite"] is True
    assert first["velocity"]["max"] > 0.0


def test_fast_reference_gate_accepts_identical_finite_samples() -> None:
    values = np.asarray([-0.002, -0.0011, 0.0, 0.002], dtype=np.float64)
    result = _backend_metrics(values, values.copy(), None, None)
    assert result["gate_pass"] is True
    assert result["reference_gt_1mm_recall"] == 1.0
    assert result["absolute_error_max_m"] == 0.0


def test_fast_reference_gate_rejects_active_signal_loss() -> None:
    fast = np.asarray([0.002, 0.002, 0.002, 0.002], dtype=np.float64)
    reference = np.asarray([-0.002, -0.002, -0.002, -0.002], dtype=np.float64)
    result = _backend_metrics(fast, reference, None, None)
    assert result["gate_pass"] is False
    assert result["false_negative_count"] == 4


def test_source_stratification_is_deterministic_and_caps_repetition() -> None:
    rows = [
        {
            "sequence": f"s{index}/object_{index}",
            "subject": "s1",
            "object": f"o{index}",
            "source_score": 10 - index,
        }
        for index in range(8)
    ]
    rows.extend(
        {
            "sequence": f"s1/repeat_{index}",
            "subject": "s1",
            "object": "o0",
            "source_score": 20 - index,
        }
        for index in range(4)
    )
    first = _stratified(rows, 8)
    second = _stratified(rows, 8)
    assert [row["sequence"] for row in first] == [row["sequence"] for row in second]
    assert len({row["object"] for row in first}) >= 3
    assert sum(row["subject"] == "s1" for row in first) <= 4


def test_active_gate_requires_two_configured_conditions() -> None:
    source = {"source_penetration_gt_1mm_frames": 0}
    e0_rows = [
        {
            "max_penetration_m": 0.002,
            "full_negative_sample_fraction": 0.01,
            "e_sdf": 0.0002,
        }
        for _ in range(12)
    ]
    cfg = {
        "penetration_active_gate": {
            "required_conditions": 2,
            "min_frames_gt_1mm": 5,
            "min_frames_max_gt_1_5mm": 3,
            "min_frames_fraction_gt_0_005": 5,
        }
    }
    result = _active_gate(source, e0_rows, cfg)
    assert result["penetration_active"] is True
    assert result["satisfied_count"] >= 2
