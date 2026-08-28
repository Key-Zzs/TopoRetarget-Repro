from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


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


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        value = hashlib.sha256(item.read_bytes()).hexdigest()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _semantic_receipt(
    tmp_path: Path, *, status: str = "RETARGET_SEMANTIC_PASS"
) -> tuple[Path, Path, Path]:
    canonical = tmp_path / "canonical.zarr"
    final = tmp_path / "final.zarr"
    canonical.mkdir()
    final.mkdir()
    (canonical / "zarr.json").write_text('{"kind":"canonical"}', encoding="utf-8")
    (final / "zarr.json").write_text('{"kind":"final"}', encoding="utf-8")
    gate = {"schema_version": "RetargetSemanticValidityV1", "frozen_limit": 1.0}
    gate_hash = hashlib.sha256(
        json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt = tmp_path / "semantic_qualification.json"
    receipt.write_text(
        json.dumps(
            {
                "identifier": "clip",
                "artifacts": {
                    "canonical": {"path": str(canonical), "sha256": _tree_hash(canonical)},
                    "final": {"path": str(final), "sha256": _tree_hash(final)},
                },
                "final": {
                    "qualification": {
                        "schema_version": "RetargetSemanticValidityV1",
                        "status": status,
                        "gate_contract_sha256": gate_hash,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "semantic_gate_contract.json").write_text(json.dumps(gate), encoding="utf-8")
    (tmp_path / "semantic_gate_contract_sha256.txt").write_text(gate_hash + "\n", encoding="utf-8")
    return receipt, canonical, final


def _geometric_receipt(
    tmp_path: Path,
    canonical: Path,
    final: Path,
    *,
    status: str = "PASS",
) -> tuple[Path, Path]:
    checkpoint = tmp_path / "manifest.json"
    checkpoint.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "geometric_retarget_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": status,
                "clip_id": "clip",
                "artifacts": {
                    "canonical": {"path": str(canonical)},
                    "final": {"path": str(final)},
                    "checkpoint_manifest": {"path": str(checkpoint)},
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt, checkpoint


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


def test_semantic_pass_is_bound_to_identifier_and_artifacts(tmp_path: Path) -> None:
    module = _module()
    receipt, canonical, final = _semantic_receipt(tmp_path)

    result = module.require_semantic_admission(
        receipt, identifier="clip", canonical=canonical, final=final
    )

    assert result["status"] == "RETARGET_SEMANTIC_PASS"
    assert result["gate_contract_sha256"]


@pytest.mark.parametrize("status", ["RETARGET_SEMANTIC_FAIL", "RETARGET_SEMANTIC_INCONCLUSIVE"])
def test_semantic_nonpass_stops_reference_preparation(tmp_path: Path, status: str) -> None:
    module = _module()
    receipt, canonical, final = _semantic_receipt(tmp_path, status=status)

    with pytest.raises(ValueError, match="RETARGET_SEMANTIC_ADMISSION_NONPASS"):
        module.require_semantic_admission(
            receipt, identifier="clip", canonical=canonical, final=final
        )


def test_semantic_receipt_fails_closed_on_artifact_drift(tmp_path: Path) -> None:
    module = _module()
    receipt, canonical, final = _semantic_receipt(tmp_path)
    (final / "zarr.json").write_text('{"kind":"drifted"}', encoding="utf-8")

    with pytest.raises(ValueError, match="RETARGET_SEMANTIC_ADMISSION_ARTIFACT_DRIFT:final"):
        module.require_semantic_admission(
            receipt, identifier="clip", canonical=canonical, final=final
        )


def test_numerical_pass_is_bound_to_the_same_artifacts(tmp_path: Path) -> None:
    module = _module()
    _, canonical, final = _semantic_receipt(tmp_path)
    receipt, checkpoint = _geometric_receipt(tmp_path, canonical, final)

    result = module.require_numerical_solver_success(
        receipt,
        clip_id="clip",
        canonical=canonical,
        final=final,
        checkpoint_manifest=checkpoint,
    )

    assert result["status"] == "PASS"


def test_numerical_nonpass_stops_reference_preparation(tmp_path: Path) -> None:
    module = _module()
    _, canonical, final = _semantic_receipt(tmp_path)
    receipt, checkpoint = _geometric_receipt(tmp_path, canonical, final, status="FAIL")

    with pytest.raises(ValueError, match="NUMERICAL_SOLVER_NONPASS"):
        module.require_numerical_solver_success(
            receipt,
            clip_id="clip",
            canonical=canonical,
            final=final,
            checkpoint_manifest=checkpoint,
        )
