#!/usr/bin/env python3
"""Certify two manually approved semantic canaries without rerunning retarget."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SEED = 20260831


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--p4-root", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P6_MAPPING_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_contains(commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=False
        ).returncode
        == 0
    )


def _artifact_hash(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(_sha256(child).encode())
            digest.update(b"\n")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def main() -> int:
    args = _parser().parse_args()
    report_root = args.report_root.resolve()
    p5 = report_root / "p5_two_canary_retarget"
    p6 = report_root / "p6_semantic_certification"
    if p6.exists():
        partial = {path.name for path in p6.iterdir() if path.is_file()}
        if partial != {"manual_acceptance.json"}:
            raise FileExistsError(f"P6_OUTPUT_EXISTS:{p6}")

    manifest = _read_json(args.manifest.resolve())
    acceptance = _read_json(args.acceptance.resolve())
    if acceptance.get("schema_version") != "ManualRetargetAcceptanceV1":
        raise ValueError("P6_MANUAL_ACCEPTANCE_SCHEMA_INVALID")
    if acceptance.get("status") != "APPROVED_FOR_P6_P8":
        raise ValueError("P6_MANUAL_ACCEPTANCE_STATUS_INVALID")
    approved = {str(row["episode_id"]): row for row in acceptance.get("canaries", [])}
    if len(approved) != 2 or any(
        row.get("reviewer_decision") != "APPROVE" for row in approved.values()
    ):
        raise ValueError("P6_TWO_CANARY_APPROVAL_REQUIRED")

    expected_manifest_hash = str(manifest.get("manifest_sha256", ""))
    manifest_core = dict(manifest)
    manifest_core.pop("manifest_sha256", None)
    actual_manifest_hash = hashlib.sha256(
        json.dumps(manifest_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_manifest_hash != actual_manifest_hash:
        raise ValueError("P6_P5_MANIFEST_EMBEDDED_HASH_INVALID")
    manifest_hash_file = p5 / "manifest_sha256.txt"
    if manifest_hash_file.read_text(encoding="utf-8").strip() != expected_manifest_hash:
        raise ValueError("P6_P5_MANIFEST_HASH_RECEIPT_DRIFT")

    p5_final = _read_json(p5 / "final_summary.json")
    p5_head = str(p5_final["p5_execution_head"])
    current_head = _git_head()
    if not _git_contains(p5_head):
        raise ValueError("P6_CURRENT_HEAD_DOES_NOT_CONTAIN_P5_EXECUTION_HEAD")

    p4_results = list(csv.DictReader((args.p4_root / "semantic_preflight_results.csv").open()))
    p4_candidates = list(csv.DictReader((args.p4_root / "all_episode_candidates.csv").open()))
    canonical_records: dict[str, dict[str, Any]] = {}
    with (args.p4_root / "canonical_hoi_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                canonical_records[str(record["episode_id"])] = record

    canary_checks: list[dict[str, Any]] = []
    integrity_ok = True
    for row in p5_final["canaries"]:
        episode_id = str(row["episode_id"])
        approval = approved.get(episode_id)
        if approval is None or approval.get("machine_semantic_status") != row["semantic_status"]:
            raise ValueError(f"P6_APPROVAL_BINDING_INVALID:{episode_id}")
        wrapper = p5 / f"canary_{row['rank']}" / "visualization.html"
        geometric_dir = p5 / f"canary_{row['rank']}" / "report" / "episodes" / episode_id
        receipt = _read_json(geometric_dir / "geometric_retarget_receipt.json")
        semantic = _read_json(geometric_dir / "retarget" / "semantic_qualification.json")
        gate_hash = (
            (geometric_dir / "retarget" / "semantic_gate_contract_sha256.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        expected_record_hash = next(
            item["canonical_record_sha256"]
            for item in manifest["clips"]
            if item["episode_id"] == episode_id
        )
        checks = {
            "manifest_hash_unchanged": manifest_hash_file.read_text(encoding="utf-8").strip()
            == expected_manifest_hash,
            "canonical_record_hash_unchanged": canonical_records[episode_id][
                "canonical_record_sha256"
            ]
            == expected_record_hash,
            "retarget_output_hash_unchanged": _artifact_hash(Path(row["retarget_output"]))
            == row["retarget_output_sha256"],
            "html_hash_unchanged": _sha256(wrapper) == row["visualization_sha256"],
            "semantic_gate_hash_unchanged": gate_hash == row["semantic_gate_sha256"],
            "geometric_receipt_pass": receipt.get("status") == "PASS",
            "semantic_pass": row.get("semantic_status") == "RETARGET_SEMANTIC_PASS",
            "human_approve": approval.get("reviewer_decision") == "APPROVE",
            "approval_html_hash_bound": approval.get("reviewed_html_sha256")
            == row["visualization_sha256"],
            "approval_retarget_hash_bound": approval.get("reviewed_retarget_sha256")
            == row["retarget_output_sha256"],
            "semantic_qualification_receipt_pass": semantic.get("final", {})
            .get("qualification", {})
            .get("status")
            == "RETARGET_SEMANTIC_PASS",
        }
        integrity_ok &= all(checks.values())
        canary_checks.append({"episode_id": episode_id, "checks": checks})

    binding_conflicts = [row for row in p4_candidates if row.get("binding_status") != "PASS"]
    official_conflicts = [
        row
        for row in p4_candidates
        if row.get("target_authority_status") == "OFFICIAL_VS_GEOMETRY_CONFLICT"
    ]
    quarantined = list(csv.DictReader((args.p4_root / "quarantine.csv").open()))
    ambiguous_cases = list(csv.DictReader((args.p4_root / "ambiguous_cases.csv").open()))
    semantic_pass = [row for row in p4_results if row.get("status") == "SEMANTIC_PREFLIGHT_PASS"]
    pass_keys = sorted(semantic_pass, key=lambda row: _stable_key(SEED, row["episode_id"]))
    random_sample = pass_keys[:20]
    multi_object_pass: list[dict[str, Any]] = []
    for row in semantic_pass:
        record = canonical_records.get(row["episode_id"], {})
        if record.get("other_object_ids"):
            multi_object_pass.append(row)
    multi_object_pass.sort(key=lambda row: _stable_key(SEED, row["episode_id"]))
    multi_object_sample = multi_object_pass[:20]

    sample_rows: list[dict[str, Any]] = []
    for sample_kind, rows in (
        ("DETERMINISTIC_RANDOM_SEMANTIC_PASS", random_sample),
        ("MULTI_OBJECT_CONTEXT_SEMANTIC_PASS", multi_object_sample),
    ):
        for rank, row in enumerate(rows, start=1):
            sample_rows.append(
                {
                    "sample_kind": sample_kind,
                    "sample_rank": rank,
                    "episode_id": row["episode_id"],
                    "active_hand": row["active_hand"],
                    "target_object": row.get("target_object_id", ""),
                    "semantic_status": row["status"],
                    "canonical_record_sha256": row["canonical_record_sha256"],
                    "downstream_outcome_used": False,
                }
            )

    now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    _write_json(
        p6 / "manual_acceptance.json",
        {**acceptance, "copied_utc": now, "p5_manifest_sha256": expected_manifest_hash},
    )
    _write_json(
        p6 / "canary_role_update.json",
        {
            "schema_version": "SemanticPipelineCanaryRoleUpdateV1",
            "status": "APPLIED",
            "role": "SEMANTIC_PIPELINE_CANARY_SET_V1",
            "development_exclusion_required": True,
            "canaries": [
                {
                    "episode_id": row["episode_id"],
                    "target_object": row["target_object"],
                    "raw_sequence": row["raw_sequence"],
                    "subject": row["subject"],
                    "machine_semantic_status": "RETARGET_SEMANTIC_PASS",
                    "human_decision": approved[row["episode_id"]]["reviewer_decision"],
                }
                for row in manifest["clips"]
            ],
            "outcome_used": False,
        },
    )
    _write_json(
        p6 / "corpus_qa_sample.json",
        {
            "schema_version": "CorpusSemanticQASampleV1",
            "status": "PASS",
            "selection_seed": SEED,
            "selection_basis": "deterministic metadata-only hash ordering",
            "random_semantic_pass_count": len(random_sample),
            "multi_object_context_pass_count": len(multi_object_pass),
            "multi_object_context_reported_count": len(multi_object_sample),
            "downstream_outcomes_used": False,
            "samples_csv": "corpus_qa_results.csv",
        },
    )
    _write_csv(
        p6 / "corpus_qa_results.csv",
        sample_rows,
        [
            "sample_kind",
            "sample_rank",
            "episode_id",
            "active_hand",
            "target_object",
            "semantic_status",
            "canonical_record_sha256",
            "downstream_outcome_used",
        ],
    )
    _write_json(
        p6 / "semantic_false_positive_analysis.json",
        {
            "schema_version": "SemanticFalsePositiveAnalysisV1",
            "status": "NOT_RUN",
            "reason": (
                "P6 corpus QA is outcome-independent; no held-out or physical outcome was consumed."
            ),
            "false_positive_count": None,
            "downstream_outcomes_used": False,
        },
    )
    _write_json(
        p6 / "final_authority_decision.json",
        {
            "schema_version": "PostApprovalSemanticCertificationV1",
            "status": "PASS" if integrity_ok else "FAIL",
            "gate": "P6_SEMANTIC_CERTIFICATION",
            "created_utc": now,
            "current_head": current_head,
            "p5_execution_head": p5_head,
            "p5_execution_head_contained": True,
            "p5_manifest_sha256": expected_manifest_hash,
            "canary_checks": canary_checks,
            "corpus_qa": {
                "semantic_pass": len(semantic_pass),
                "ambiguous_cases": len(ambiguous_cases),
                "quarantined_cases": len(quarantined),
                "binding_conflicts": len(binding_conflicts),
                "official_vs_geometry_conflicts": len(official_conflicts),
                "deterministic_random_semantic_pass_sample": len(random_sample),
                "multi_object_context_semantic_pass_sample": len(multi_object_sample),
                "downstream_outcomes_used": False,
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "P6_SEMANTIC_CERTIFICATION_PASS"
                if integrity_ok
                else "P6_SEMANTIC_CERTIFICATION_FAIL",
                "output": str(p6),
                "binding_conflicts": len(binding_conflicts),
                "official_vs_geometry_conflicts": len(official_conflicts),
                "quarantined": len(quarantined),
                "random_sample": len(random_sample),
                "multi_object_sample": len(multi_object_sample),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if integrity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
