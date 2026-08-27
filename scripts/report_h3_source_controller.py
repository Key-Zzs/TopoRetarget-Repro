#!/usr/bin/env python3
"""Summarize the seven-episode H3 source-controller admission audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_DECISIONS = {
    "H3A_SOURCE_ADMISSION_V2_VALIDATED",
    "H3A_SOURCE_ADMISSION_V2_PARTIAL",
    "H3A_TRUE_SOURCE_CONTROLLER_HARD_FAILURES_IDENTIFIED",
    "H3A_INCONCLUSIVE",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"H3A_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"H3A_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3A_REPORT_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _qualification_from_artifact(row: object) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(row, dict):
        raise ValueError("H3A_QUALIFICATION_ARTIFACT_REQUIRED")
    path = Path(str(row.get("path", ""))).resolve()
    artifact = _artifact(path)
    if row.get("sha256") != artifact["sha256"]:
        raise ValueError(f"H3A_QUALIFICATION_HASH_DRIFT:{path}")
    qualification = _json(path)
    if qualification.get("schema_version") != "SourceControllerQualificationV2":
        raise ValueError(f"H3A_QUALIFICATION_SCHEMA_INVALID:{path}")
    return qualification, artifact


def _old_status(entry: dict[str, Any]) -> tuple[str, dict[str, str] | None]:
    value = entry.get("old_status")
    path_value = entry.get("old_status_path")
    if path_value is None:
        return str(value or "NOT_AVAILABLE"), None
    path = Path(str(path_value)).resolve()
    payload = _json(path)
    return str(payload.get("source_controller", payload.get("status", value))), _artifact(path)


def _summarize(entry: dict[str, Any]) -> dict[str, Any]:
    episode_id = str(entry.get("episode_id", ""))
    if not episode_id:
        raise ValueError("H3A_EPISODE_ID_REQUIRED")
    if entry.get("source_policy_receipt") is None:
        return _summarize_direct_qualification(entry, episode_id)
    receipt_path = Path(str(entry["source_policy_receipt"])).resolve()
    receipt = _json(receipt_path)
    if not (
        receipt.get("schema_version") == "IndependentSourcePolicyReceiptV4"
        and receipt.get("status") == "PASS"
        and receipt.get("clip_id") == episode_id
    ):
        raise ValueError(f"H3A_SOURCE_POLICY_RECEIPT_INVALID:{episode_id}")
    selected = str(receipt.get("selected_route", ""))
    if selected not in {"ZERO_RESIDUAL", "CORRECTED_L0"}:
        raise ValueError(f"H3A_SELECTED_ROUTE_INVALID:{episode_id}")
    zero, zero_artifact = _qualification_from_artifact(
        receipt.get("artifacts", {}).get("zero_residual_qualification")
    )
    selected_qualification, selected_artifact = _qualification_from_artifact(
        receipt.get("source_qualification")
    )
    if selected == "ZERO_RESIDUAL" and selected_qualification.get("mode") not in {
        "ZERO_RESIDUAL_DETERMINISTIC",
        "ZERO_RESIDUAL_NETWORK",
    }:
        raise ValueError(f"H3A_ZERO_SELECTED_QUALIFICATION_MODE_INVALID:{episode_id}")
    if selected == "CORRECTED_L0" and selected_qualification.get("mode") != "CORRECTED_L0":
        raise ValueError(f"H3A_L0_SELECTED_QUALIFICATION_MODE_INVALID:{episode_id}")
    if receipt.get("source_controller_executability_v2") != "PASS":
        raise ValueError(f"H3A_SELECTED_SOURCE_NOT_EXECUTABLE:{episode_id}")
    l0_run = selected == "CORRECTED_L0"
    old, old_artifact = _old_status(entry)
    return {
        "episode_id": episode_id,
        "label": str(entry.get("label", episode_id)),
        "dataset_role": str(entry.get("dataset_role", "SOURCE_CONTROLLER_AUDIT_SET_V1")),
        "zero_residual_executable": str(
            zero.get("source_controller_executability_v2", "INCONCLUSIVE")
        ),
        "zero_residual_fidelity": str(zero.get("source_controller_fidelity_v2", "INCONCLUSIVE")),
        "l0_run": l0_run,
        "l0_executable": (
            str(selected_qualification.get("source_controller_executability_v2"))
            if l0_run
            else "NOT_RUN_ZERO_RESIDUAL_EXECUTABLE"
        ),
        "l0_fidelity": (
            str(selected_qualification.get("source_controller_fidelity_v2"))
            if l0_run
            else "NOT_RUN_ZERO_RESIDUAL_EXECUTABLE"
        ),
        "selected_route": selected,
        "selected_executable": str(receipt["source_controller_executability_v2"]),
        "selected_fidelity": str(receipt.get("source_controller_fidelity_v2", "INCONCLUSIVE")),
        "l0_samples": int(receipt.get("l0_samples", -1)),
        "old_status": old,
        "new_admission": "AUTHORIZED_FOR_FROZEN_FULL_GRAVITY_EVALUATION",
        "source_policy_receipt": _artifact(receipt_path),
        "zero_qualification": zero_artifact,
        "selected_qualification": selected_artifact,
        "old_status_artifact": old_artifact,
    }


def _summarize_direct_qualification(entry: dict[str, Any], episode_id: str) -> dict[str, Any]:
    zero, zero_artifact = _qualification_from_artifact(entry.get("zero_qualification"))
    if zero.get("clip_id") != episode_id:
        raise ValueError(f"H3A_DIRECT_QUALIFICATION_CLIP_MISMATCH:{episode_id}")
    zero_executable = str(zero.get("source_controller_executability_v2", "INCONCLUSIVE"))
    l0_value = entry.get("l0_qualification")
    l0: dict[str, Any] | None = None
    l0_artifact: dict[str, str] | None = None
    if l0_value is not None:
        l0, l0_artifact = _qualification_from_artifact(l0_value)
        if l0.get("clip_id") != episode_id or l0.get("mode") != "CORRECTED_L0":
            raise ValueError(f"H3A_DIRECT_L0_QUALIFICATION_INVALID:{episode_id}")
    if zero_executable == "PASS":
        selected = "ZERO_RESIDUAL"
        selected_qualification = zero
        selected_artifact = zero_artifact
    elif l0 is not None and l0.get("source_controller_executability_v2") == "PASS":
        selected = "CORRECTED_L0"
        selected_qualification = l0
        assert l0_artifact is not None
        selected_artifact = l0_artifact
    else:
        raise ValueError(f"H3A_DIRECT_SOURCE_CONTROLLER_HARD_FAILURE:{episode_id}")
    old, old_artifact = _old_status(entry)
    return {
        "episode_id": episode_id,
        "label": str(entry.get("label", episode_id)),
        "dataset_role": str(entry.get("dataset_role", "DEVELOPMENT_AUDIT_SET_V1")),
        "zero_residual_executable": zero_executable,
        "zero_residual_fidelity": str(zero.get("source_controller_fidelity_v2", "INCONCLUSIVE")),
        "l0_run": l0 is not None,
        "l0_executable": (
            str(l0.get("source_controller_executability_v2"))
            if l0 is not None
            else "NOT_RUN_ZERO_RESIDUAL_EXECUTABLE"
        ),
        "l0_fidelity": (
            str(l0.get("source_controller_fidelity_v2"))
            if l0 is not None
            else "NOT_RUN_ZERO_RESIDUAL_EXECUTABLE"
        ),
        "selected_route": selected,
        "selected_executable": str(selected_qualification["source_controller_executability_v2"]),
        "selected_fidelity": str(
            selected_qualification.get("source_controller_fidelity_v2", "INCONCLUSIVE")
        ),
        "l0_samples": int(selected_qualification.get("training_samples", 0)),
        "old_status": old,
        "new_admission": "AUTHORIZED_FOR_FROZEN_FULL_GRAVITY_EVALUATION",
        "source_policy_receipt": None,
        "zero_qualification": zero_artifact,
        "selected_qualification": selected_artifact,
        "old_status_artifact": old_artifact,
    }


def build_report(index: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if index.get("schema_version") != "H3SourceControllerAuditIndexV1":
        raise ValueError("H3A_AUDIT_INDEX_SCHEMA_INVALID")
    entries = index.get("episodes")
    if not isinstance(entries, list) or len(entries) != 7:
        raise ValueError("H3A_AUDIT_REQUIRES_SEVEN_EPISODES")
    rows = [_summarize(dict(entry)) for entry in entries if isinstance(entry, dict)]
    if len(rows) != 7 or len({row["episode_id"] for row in rows}) != 7:
        raise ValueError("H3A_AUDIT_EPISODES_NOT_UNIQUE")
    hardening = [row for row in rows if row["dataset_role"] == "PIPELINE_HARDENING_SET_V1"]
    development = [row for row in rows if row["dataset_role"] == "DEVELOPMENT_AUDIT_SET_V1"]
    if len(hardening) != 5 or len(development) != 2:
        raise ValueError("H3A_AUDIT_ROLE_COUNTS_INVALID")
    selected_pass = sum(row["selected_executable"] == "PASS" for row in rows)
    hardening_pass = sum(row["selected_executable"] == "PASS" for row in hardening)
    decision = (
        "H3A_SOURCE_ADMISSION_V2_VALIDATED"
        if selected_pass == 7 and hardening_pass == 5
        else "H3A_SOURCE_ADMISSION_V2_PARTIAL"
    )
    if decision not in EXPECTED_DECISIONS:
        raise RuntimeError("H3A_DECISION_INVALID")
    return rows, {
        "schema_version": "H3SourceControllerDecisionV1",
        "status": "PASS" if selected_pass == 7 else "PARTIAL",
        "decision": decision,
        "H3A_SELECTED_SOURCE_CONTROLLER_CONTRACT": (
            "SourceControllerAutoV2_ZERO_RESIDUAL_THEN_BOUNDED_L0_V1"
        ),
        "audited_episode_count": len(rows),
        "selected_executable_count": selected_pass,
        "hardening_executable_count": hardening_pass,
        "hardening_episode_count": len(hardening),
        "all_hardening_authorized_for_frozen_full_gravity_evaluation": hardening_pass == 5,
        "downstream_outcomes_used_for_route_selection": False,
        "l0_runs": sum(bool(row["l0_run"]) for row in rows),
    }


def main() -> int:
    args = _parser().parse_args()
    index_path = args.audit_index.resolve()
    index = _json(index_path)
    rows, decision = build_report(index)
    output = args.output_root.resolve()
    fields = [
        "episode_id",
        "label",
        "dataset_role",
        "zero_residual_executable",
        "zero_residual_fidelity",
        "l0_run",
        "l0_executable",
        "l0_fidelity",
        "selected_route",
        "old_status",
        "new_admission",
        "l0_samples",
    ]
    csv_path = output / "zero_residual_vs_l0.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        raise FileExistsError(f"H3A_REPORT_OUTPUT_EXISTS:{csv_path}")
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    os.replace(temporary, csv_path)
    per_episode = output / "per_episode"
    for row in rows:
        _write_new(
            per_episode / str(row["episode_id"]) / "source_controller_audit.json",
            json.dumps(row, indent=2, sort_keys=True) + "\n",
        )
    common = {
        "audit_index": _artifact(index_path),
        "episode_count": len(rows),
        "per_episode": [
            {
                "episode_id": row["episode_id"],
                "selected_route": row["selected_route"],
                "source_policy_receipt": row["source_policy_receipt"],
            }
            for row in rows
        ],
    }
    _write_new(
        output / "executable_contract.json",
        json.dumps(
            {
                "schema_version": "H3SourceControllerExecutableContractV2Receipt",
                "status": "PASS" if decision["selected_executable_count"] == 7 else "PARTIAL",
                "contract": "SourceControllerExecutableContractV2",
                "hard_gate_scope": "finite execution and true physical safety only",
                **common,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_new(
        output / "fidelity_contract.json",
        json.dumps(
            {
                "schema_version": "H3SourceControllerFidelityContractV2Receipt",
                "status": "DIAGNOSTIC_NON_GATING",
                "contract": "SourceControllerFidelityV2",
                "downstream_admission_gate": False,
                **common,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _write_new(
        output / "route_contract.json",
        json.dumps(
            {
                "schema_version": "H3SelectedSourceControllerContractV1",
                "status": "H3A_SELECTED_SOURCE_CONTROLLER_CONTRACT",
                "route_policy": "SourceControllerMode.AUTO_V2",
                "downstream_outcomes_used": False,
                **common,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    decision["artifacts"] = {
        "audit_index": _artifact(index_path),
        "comparison_csv": _artifact(csv_path),
    }
    _write_new(
        output / "final_decision.json",
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
