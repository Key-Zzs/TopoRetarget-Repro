from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.report_h3_retarget_throughput import (
    RUNTIME_ONLY_ARRAYS,
    _arrays_equal,
    _before_after,
    _compare_validation,
    _selected_attempt_profile,
)


def _timing_row(*, total: float, solver_ms: float) -> dict[str, str]:
    return {
        "case_id": "A",
        "repeat": "1",
        "frames": "10",
        "solver_ms_per_frame": str(solver_ms),
        "checkpoint_payload_seconds": "0.1",
        "checkpoint_serialization_seconds": "0.2",
        "append_write_seconds": "0.3",
        "durable_checkpoint_seconds": "0.4",
        "assembly_validation_seconds": "1.0",
        "full_frame_validation_seconds": "2.0",
        "html_generation_seconds": "3.0",
        "orchestration_seconds": "4.0",
        "total_seconds": str(total),
    }


def test_array_parity_is_bitwise_with_nan_equivalence() -> None:
    left = np.asarray([1.0, np.nan], dtype=np.float64)
    assert _arrays_equal(left, left.copy())
    right = left.copy()
    right[0] = np.nextafter(right[0], np.inf)
    assert not _arrays_equal(left, right)
    assert RUNTIME_ONLY_ARRAYS == {"solve_time_s"}


def test_before_after_uses_same_case_and_frame_denominator() -> None:
    rows = _before_after(
        [_timing_row(total=20.0, solver_ms=100.0)],
        [_timing_row(total=15.0, solver_ms=100.0)],
    )
    assert len(rows) == 1
    assert rows[0]["frames"] == 10
    assert rows[0]["total_improvement_fraction"] == 0.25
    assert rows[0]["baseline_io_ms_per_frame"] == 100.0
    assert rows[0]["optimized_io_ms_per_frame"] == 100.0


def test_validation_parity_excludes_only_artifact_path(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    payload = {"status": "pass", "pass": True, "frames": [{"accepted": True}]}
    baseline.write_text(json.dumps({**payload, "final": "/baseline/final.zarr"}), encoding="utf-8")
    optimized.write_text(
        json.dumps({**payload, "final": "/optimized/final.zarr"}), encoding="utf-8"
    )
    assert _compare_validation(baseline, optimized)["status"] == "PASS"
    optimized.write_text(
        json.dumps(
            {
                **payload,
                "frames": [{"accepted": False}],
                "final": "/optimized/final.zarr",
            }
        ),
        encoding="utf-8",
    )
    assert _compare_validation(baseline, optimized)["status"] == "FAIL"


def test_selected_attempt_profile_separates_slsqp_and_final_audit(tmp_path: Path) -> None:
    case = {"case_id": "A", "prepared_root": "unused"}
    frames = tmp_path / "A/repeat_01/retarget/continuous_checkpoints/frames"
    frames.mkdir(parents=True)
    metadata = {
        "solve_time_s": 4.0,
        "diagnostics": {
            "timers": {
                "elapsed_s": {
                    "slsqp_total": 2.0,
                    "active_set_discovery": 1.0,
                    "final_full_audit": 0.5,
                }
            },
            "final_audit_query_reused": True,
            "physical_reference_final_audit_query_count": 0,
        },
    }
    np.savez(frames / "frame_000000.npz", metadata_json=json.dumps(metadata))
    result = _selected_attempt_profile(tmp_path, case, 1)
    assert result["selected_attempt_slsqp_ms_per_frame"] == 2000.0
    assert result["final_full_audit_ms_per_frame"] == 500.0
    assert result["final_audit_query_reuse_frames"] == 1
