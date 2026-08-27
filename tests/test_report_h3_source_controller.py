from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.report_h3_source_controller import build_report


def _write(path: Path, value: object) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _entry(root: Path, index: int, *, l0: bool = False) -> dict[str, object]:
    episode = f"episode_{index}"
    zero = {
        "schema_version": "SourceControllerQualificationV2",
        "clip_id": episode,
        "mode": "ZERO_RESIDUAL_DETERMINISTIC",
        "source_controller_executability_v2": "FAIL" if l0 else "PASS",
        "source_controller_fidelity_v2": "FAIL" if l0 else "DEGRADED",
    }
    selected = (
        {
            "schema_version": "SourceControllerQualificationV2",
            "clip_id": episode,
            "mode": "CORRECTED_L0",
            "source_controller_executability_v2": "PASS",
            "source_controller_fidelity_v2": "DEGRADED",
        }
        if l0
        else zero
    )
    zero_artifact = _write(root / episode / "zero.json", zero)
    selected_artifact = _write(root / episode / "l0.json", selected) if l0 else zero_artifact
    receipt = {
        "schema_version": "IndependentSourcePolicyReceiptV4",
        "status": "PASS",
        "clip_id": episode,
        "selected_route": "CORRECTED_L0" if l0 else "ZERO_RESIDUAL",
        "source_controller_executability_v2": "PASS",
        "source_controller_fidelity_v2": "DEGRADED",
        "l0_samples": 1_024_000 if l0 else 0,
        "source_qualification": selected_artifact,
        "artifacts": {"zero_residual_qualification": zero_artifact},
    }
    receipt_path = root / episode / "receipt.json"
    _write(receipt_path, receipt)
    return {
        "episode_id": episode,
        "dataset_role": ("DEVELOPMENT_AUDIT_SET_V1" if index < 2 else "PIPELINE_HARDENING_SET_V1"),
        "source_policy_receipt": str(receipt_path),
        "old_status": "SOURCE_CONTROLLER_FAILED",
    }


def _direct_dev_entry(root: Path, index: int) -> dict[str, object]:
    episode = f"episode_{index}"
    artifact = _write(
        root / episode / "direct_zero.json",
        {
            "schema_version": "SourceControllerQualificationV2",
            "clip_id": episode,
            "mode": "ZERO_RESIDUAL_DETERMINISTIC",
            "training_samples": 0,
            "source_controller_executability_v2": "PASS",
            "source_controller_fidelity_v2": "DEGRADED",
        },
    )
    return {
        "episode_id": episode,
        "dataset_role": "DEVELOPMENT_AUDIT_SET_V1",
        "zero_qualification": artifact,
        "old_status": "DEVELOPMENT_BASELINE",
    }


def test_build_report_accepts_seven_terminal_audits(tmp_path: Path) -> None:
    entries = [_direct_dev_entry(tmp_path, index) for index in range(2)]
    entries.extend(_entry(tmp_path, index, l0=index == 4) for index in range(2, 7))
    index = {
        "schema_version": "H3SourceControllerAuditIndexV1",
        "episodes": entries,
    }
    rows, decision = build_report(index)
    assert len(rows) == 7
    assert rows[4]["selected_route"] == "CORRECTED_L0"
    assert decision["decision"] == "H3A_SOURCE_ADMISSION_V2_VALIDATED"
    assert decision["H3A_SELECTED_SOURCE_CONTROLLER_CONTRACT"].startswith("SourceControllerAutoV2")
    assert decision["hardening_executable_count"] == 5


def test_build_report_rejects_nonterminal_selected_receipt(tmp_path: Path) -> None:
    entries = [_direct_dev_entry(tmp_path, index) for index in range(2)]
    entries.extend(_entry(tmp_path, index) for index in range(2, 7))
    receipt_path = Path(str(entries[-1]["source_policy_receipt"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_controller_executability_v2"] = "FAIL"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="H3A_SELECTED_SOURCE_NOT_EXECUTABLE"):
        build_report({"schema_version": "H3SourceControllerAuditIndexV1", "episodes": entries})


def test_build_report_requires_two_dev_and_five_hardening(tmp_path: Path) -> None:
    entries = [_direct_dev_entry(tmp_path, index) for index in range(2)]
    entries.extend(_entry(tmp_path, index) for index in range(2, 7))
    entries[0]["dataset_role"] = "PIPELINE_HARDENING_SET_V1"
    with pytest.raises(ValueError, match="H3A_AUDIT_ROLE_COUNTS_INVALID"):
        build_report({"schema_version": "H3SourceControllerAuditIndexV1", "episodes": entries})
