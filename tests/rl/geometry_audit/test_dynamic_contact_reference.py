from __future__ import annotations

import numpy as np

from toporetarget.rl.geometry_audit.dynamic_contact_reference import (
    SelectedStableCalibrationV1,
    bootstrap_upper_confidence_quantile,
    decide_geometry_v1_v2,
    freeze_empirical_dynamic_contact_reference,
)


def _calibration(
    object_id: str,
    family_id: str,
    *,
    value_m: float,
    v1_limit_m: float,
) -> SelectedStableCalibrationV1:
    return SelectedStableCalibrationV1(
        object_id=object_id,
        family_id=family_id,
        candidate_id=f"candidate_{object_id}",
        qualification_sha256="a" * 64,
        replica_max_penetration_m=(value_m,) * 20,
        replica_active_p95_penetration_m=(0.5 * value_m,) * 20,
        v1_max_limit_m=v1_limit_m,
        v1_active_p95_limit_m=v1_limit_m,
    )


def test_bootstrap_ucb_is_deterministic_and_uses_twenty_replicas() -> None:
    values = np.linspace(0.0005, 0.0015, 20)
    first = bootstrap_upper_confidence_quantile(values)
    second = bootstrap_upper_confidence_quantile(values)
    assert first == second
    assert values.min() <= first <= values.max()


def test_shared_reference_uses_worst_object_family_ucb() -> None:
    rows = (
        _calibration("hocap_170105", "thumb_opposition", value_m=0.001, v1_limit_m=0.0001),
        _calibration("hocap_170650", "bilateral_fingertip", value_m=0.002, v1_limit_m=0.0002),
    )
    required = (
        ("hocap_170105", "thumb_opposition"),
        ("hocap_170650", "bilateral_fingertip"),
    )
    reference = freeze_empirical_dynamic_contact_reference(
        rows, required_object_family_pairs=required
    )
    assert reference["dynamic_reference_max_m"] == 0.002
    assert reference["dynamic_reference_active_p95_m"] == 0.001
    assert reference["shared_across_clips"]
    assert not reference["corrected_trajectory_used"]


def test_v1_is_retained_when_every_stable_replica_meets_v1() -> None:
    rows = (
        _calibration("hocap_170105", "thumb_opposition", value_m=0.0001, v1_limit_m=0.0002),
        _calibration("hocap_170650", "bilateral_fingertip", value_m=0.0001, v1_limit_m=0.0002),
    )
    required = tuple((row.object_id, row.family_id) for row in rows)
    decision = decide_geometry_v1_v2(rows, required_object_family_pairs=required)
    assert decision["status"] == "STAGE16D_GEOMETRY_V1_ATTAINABLE"
    assert not decision["v2_created"]


def test_v2_is_shared_and_preserves_absolute_gate_when_v1_is_too_low() -> None:
    rows = (
        _calibration("hocap_170105", "thumb_opposition", value_m=0.001, v1_limit_m=0.0001),
        _calibration("hocap_170650", "bilateral_fingertip", value_m=0.002, v1_limit_m=0.0002),
    )
    required = tuple((row.object_id, row.family_id) for row in rows)
    decision = decide_geometry_v1_v2(rows, required_object_family_pairs=required)
    assert decision["status"] == "STAGE16D_GEOMETRY_V2_VALIDATED"
    assert decision["v2_created"]
    contract = decision["v2_contract"]
    assert contract["absolute_gates_unchanged"]
    assert contract["clip_specific_thresholds"] is False
    assert contract["dynamic_contact_reference_max_m"] == 0.002
