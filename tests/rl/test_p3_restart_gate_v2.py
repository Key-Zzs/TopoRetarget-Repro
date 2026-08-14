from __future__ import annotations

import numpy as np

from toporetarget.rl.p3_restart_gate_v2 import (
    EarlyTableResetCoverageGateV1,
    build_early_table_reset_pool,
    contiguous_windows,
    stage16_p3_restart_gate_v2,
)


def _rows() -> dict[str, np.ndarray]:
    return {
        "runtime_index": np.arange(12),
        "semantic_class": np.asarray(["PRE_CONTACT"] * 12),
        "support_state": np.asarray(["TABLE_SUPPORTED"] * 12),
        "overall_reference_geometry_valid": np.asarray([True] * 12),
    }


def _dynamic() -> dict[str, object]:
    return {
        "gravity_world_mps2": [0.0, 0.0, -9.81],
        "support_mode": "finite_inferred_table_proxy_v1",
        "external_guidance": False,
        "all_replicas_write_gate_pass": True,
    }


def test_contiguous_windows_retains_all_discrete_pool_members() -> None:
    assert contiguous_windows([2, 3, 4, 8, 10, 11]) == [
        {"start": 2, "end": 4, "length": 3},
        {"start": 8, "end": 8, "length": 1},
        {"start": 10, "end": 11, "length": 2},
    ]


def test_early_pool_requires_exact_geometry_and_actual_dynamic_receipt() -> None:
    pool = build_early_table_reset_pool(
        clip="hocap_170105",
        validity_rows=_rows(),
        dynamic_safe_indices=range(2, 11),
        dynamic_summary=_dynamic(),
    )
    assert pool["frames"] == list(range(2, 11))
    assert pool["longest_window"] == {"start": 2, "end": 10, "length": 9}
    assert pool["status"] == "EARLY_TABLE_SUPPORTED_HARD_RESET_SAFE"


def test_reference_wide_failure_is_not_a_restart_gate_but_missing_pool_is() -> None:
    failed = {"qualifying_windows": []}
    passed = {"qualifying_windows": [{"start": 2, "end": 10, "length": 9}]}
    result = stage16_p3_restart_gate_v2(
        provenance_valid=True,
        support_valid=True,
        pools={"hocap_170105": passed, "hocap_170650": failed},
    )
    assert result["reference_geometry"] == "DIAGNOSTIC_ONLY"
    assert result["decision"] == "P3B7_BLOCKED_HARD_RESET_POOL"
    assert result["gates"]["R3_residual_recoverability"]["status"] == "NOT_RUN"


def test_coverage_gate_is_pre_registered() -> None:
    assert EarlyTableResetCoverageGateV1().minimum_continuous_frames == 8
