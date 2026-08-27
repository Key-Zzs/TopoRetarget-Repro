from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/rl/prepare_independent_source_reference.py"
    )
    spec = importlib.util.spec_from_file_location("independent_source_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualification(**overrides: bool) -> dict[str, object]:
    checks = {
        "source_key_preservation": True,
        "timestamps": True,
        "quaternion": True,
        "finite": True,
        "linear_fd_consistency": True,
        "angular_so3_consistency": True,
        "world_angular_convention": True,
        "factor8_scaling": True,
        "integral_consistency": True,
    }
    checks.update(overrides)
    return {
        "status": (
            "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED"
            if all(checks.values())
            else "STAGE16D_REFERENCE_KINEMATICS_V2_BLOCKED"
        ),
        "checks": checks,
    }


def test_factor8_and_integral_accuracy_are_fidelity_only() -> None:
    module = _module()
    result = module.reference_executability_v2(
        qualification=_qualification(factor8_scaling=False, integral_consistency=False),
        world_validation={"valid": True},
    )

    assert result["status"] == "PASS"
    assert result["failed_hard_checks"] == []
    assert result["fidelity_only_diagnostics"] == {
        "factor8_scaling": False,
        "integral_consistency": False,
    }
    assert result["full_reference_kinematics_v2_status"].endswith("_BLOCKED")


def test_nonfinite_reference_is_still_a_hard_failure() -> None:
    module = _module()
    result = module.reference_executability_v2(
        qualification=_qualification(finite=False),
        world_validation={"valid": True},
    )

    assert result["status"] == "FAIL"
    assert result["failed_hard_checks"] == ["finite"]


def test_invalid_world_reference_is_still_a_hard_failure() -> None:
    module = _module()
    result = module.reference_executability_v2(
        qualification=_qualification(),
        world_validation={"valid": False},
    )

    assert result["status"] == "FAIL"
    assert result["failed_hard_checks"] == ["world_reference_valid"]
