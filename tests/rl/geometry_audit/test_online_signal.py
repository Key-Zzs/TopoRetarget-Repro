from __future__ import annotations

import numpy as np

from toporetarget.rl.geometry_audit.online_signal import qualify_online_geometry_signal


def test_exact_signal_passes_online_qualification() -> None:
    exact = np.asarray([0.0, 0.001, 0.004, 0.011, 0.0, 0.002, 0.006, 0.012])
    result = qualify_online_geometry_signal(
        estimated_penetration_m=exact,
        exact_penetration_m=exact,
        contact_active=exact > 0.0,
        split=np.asarray(["calibration"] * 4 + ["holdout"] * 4),
        exact_top_k=24,
        elite_count=12,
        exact_for_all_elites=True,
        exact_for_all_final_candidates=True,
        exact_for_all_formal_replicas=True,
    ).as_dict()
    assert result["status"] == "STAGE16D_ONLINE_GEOMETRY_SIGNAL_VALIDATED"
    assert not result["formal_gate_authority"]


def test_failed_fast_signal_can_only_use_exact_topk_fallback() -> None:
    exact = np.asarray([0.0, 0.004, 0.011, 0.0])
    result = qualify_online_geometry_signal(
        estimated_penetration_m=np.zeros(4),
        exact_penetration_m=exact,
        contact_active=exact > 0.0,
        split=np.asarray(["calibration", "calibration", "holdout", "holdout"]),
        exact_top_k=24,
        elite_count=12,
        exact_for_all_elites=True,
        exact_for_all_final_candidates=True,
        exact_for_all_formal_replicas=True,
    ).as_dict()
    assert result["status"] == "STAGE16D_EXACT_TOPK_GEOMETRY_FALLBACK_VALIDATED"
