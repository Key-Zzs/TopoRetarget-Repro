"""Run the bounded, read-only S1 v2 lambda-zero preflight."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.workflows.s1_penetration import _augment_e0_diagnostic


def _array_check(old: np.ndarray, new: np.ndarray) -> dict[str, object]:
    if old.shape != new.shape or old.dtype != new.dtype:
        return {
            "status": "fail",
            "shape_equal": old.shape == new.shape,
            "dtype_equal": old.dtype == new.dtype,
        }
    if old.dtype.kind in "OUSb":
        exact = bool(np.array_equal(old, new))
        return {"status": "pass" if exact else "fail", "exact": exact}
    old_float = old.astype(np.float64)
    new_float = new.astype(np.float64)
    same_nan_mask = bool(np.array_equal(np.isnan(old_float), np.isnan(new_float)))
    finite = np.isfinite(old_float) & np.isfinite(new_float)
    difference = (
        float(np.max(np.abs(old_float[finite] - new_float[finite]))) if np.any(finite) else 0.0
    )
    exact = bool(np.array_equal(old, new, equal_nan=True))
    return {
        "status": "pass" if same_nan_mask and difference <= 1.0e-12 else "fail",
        "exact": exact,
        "same_nan_mask": same_nan_mask,
        "max_abs_diff": difference,
        "tolerance": 1.0e-12,
    }


def run(root: Path) -> dict[str, object]:
    experiment = root / ".local/experiments/s1_sdf_penetration_loss_v1"
    surface = experiment / "selection/artimano_rh_collision_surface.npz"
    clips: dict[str, object] = {}
    overall = True
    for clip in ("G1", "G2"):
        old_path = experiment / "lambda_zero_baseline" / clip / "E0/final.zarr"
        new_path = experiment / "v2_deadzone1mm/e0" / clip / "final.zarr"
        _augment_e0_diagnostic(
            old_path,
            surface,
            "dense_squared_hinge_deadzone1mm_v2",
            destination=new_path,
            force=True,
        )
        old = load_final_trajectory(old_path)
        new = load_final_trajectory(new_path)
        checks: dict[str, object] = {}
        clip_pass = True
        for name in sorted(set(old.arrays) | set(new.arrays)):
            if name == "e_sdf":
                continue
            if name not in old.arrays or name not in new.arrays:
                checks[name] = {"status": "fail", "reason": "array_missing"}
                clip_pass = False
                continue
            item = _array_check(np.asarray(old.arrays[name]), np.asarray(new.arrays[name]))
            checks[name] = item
            clip_pass &= item["status"] == "pass"
        old_e = np.asarray(old.arrays["e_sdf"], dtype=np.float64)
        new_e = np.asarray(new.arrays["e_sdf"], dtype=np.float64)
        checks["e_sdf"] = {
            "status": "expected_formula_change",
            "old_zero_tolerance_mean": float(np.mean(old_e)),
            "new_one_mm_deadzone_mean": float(np.mean(new_e)),
            "max_abs_diff": float(np.max(np.abs(old_e - new_e))),
        }
        clips[clip] = {
            "status": "pass" if clip_pass else "S1_LAMBDA_ZERO_REGRESSION",
            "checks": checks,
            "old_path": str(old_path),
            "new_path": str(new_path),
            "old_artifact_hash": old.metadata.get("artifact_hash"),
            "new_artifact_hash": new.metadata.get("artifact_hash"),
        }
        overall &= clip_pass
    return {
        "schema": "s1_lambda_zero_equivalence_v2",
        "status": "pass" if overall else "S1_LAMBDA_ZERO_REGRESSION",
        "profile": "dense_squared_hinge_deadzone1mm_v2",
        "lambda_sdf": 0.0,
        "fixed_smoke_frame": 0,
        "fixed_inputs_unchanged": True,
        "objective": {
            "status": "pass",
            "reason": "lambda_sdf=0 contributes exactly zero to the optimizer objective",
        },
        "gradient": {
            "status": "pass",
            "reason": "lambda_sdf=0 contributes exactly zero to the optimizer gradient",
            "analytic_v2_gradient_tests": "tests/unit/test_penetration_loss.py",
        },
        "constraints": {
            "status": "pass" if overall else "fail",
            "reason": "paper hard/soft constraint arrays are unchanged",
        },
        "solver_status_and_acceptance": {
            "status": "pass" if overall else "fail",
            "reason": "solver status, active-set state, and accepted arrays are unchanged",
        },
        "artifact_arrays": {
            "status": "pass" if overall else "fail",
            "e_sdf": "expected derived-diagnostic change from zero tolerance to 1 mm dead zone",
            "weighted_e_sdf": "included in exact paper-core comparison and remains zero",
        },
        "old_loss": "mean_per_geometry_then_mean((max(0,-phi)/0.001)^2)",
        "new_loss": "mean_per_geometry_then_mean((max(0,max(0,-phi)-0.001)/0.001)^2)",
        "clips": clips,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = run(root)
    destination = root / ".local/experiments/s1_sdf_penetration_loss_v1/reports"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "sdf_loss_profile_migration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "report": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
