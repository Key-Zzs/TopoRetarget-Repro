#!/usr/bin/env python3
"""Freeze Stage 16-C.5A-R2 inputs, archived G0 evidence, and all candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-audit", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--r1-summary", type=Path, required=True)
    parser.add_argument("--r1-noise", type=Path, required=True)
    parser.add_argument("--r1-tolerances", type=Path, required=True)
    parser.add_argument("--r1-qualification", type=Path, required=True)
    parser.add_argument("--r1-transitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage16 C5A R2 input is not a JSON object: {path}")
    return payload


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R2_FREEZE_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _input_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"C5A R2 input must be inside repository: {resolved}") from error
    return {"path": str(relative), "sha256": _sha256(resolved)}


def main() -> int:
    from toporetarget.rl.environments.isaaclab_backend.physx_contract import candidate_matrix

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_matrix = output_dir / "physx_contract_candidate_matrix.json"
    expected_outputs = {
        output_dir / "frozen_baseline_contract.json",
        output_dir / "frozen_nondeterminism_evidence.json",
        output_dir / "asset_hashes.json",
        output_dir / "reference_hashes.json",
        output_dir / "controller_hashes.json",
        output_dir / "physics_hashes.json",
        output_matrix,
    }
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("STAGE16C5A_R2_FREEZE_OUTPUT_ALREADY_EXISTS")
    audit = _load(args.api_audit)
    if audit.get("status") != "STAGE16C5A_PHYSX_API_AUDITED":
        raise RuntimeError("STAGE16C5A_R2_API_AUDIT_NOT_VALIDATED")
    frozen = _load(args.frozen_inputs)
    if frozen.get("status") != "STAGE16C5A_INPUT_HASHES_MATCH":
        raise RuntimeError("STAGE16C5A_R2_INPUT_HASH_DRIFT")
    r1_summary = _load(args.r1_summary)
    if r1_summary.get("reason") != "TRUE_FROZEN_PHYSX_BASELINE_NONDETERMINISM":
        raise RuntimeError("STAGE16C5A_R2_R1_FAILURE_EVIDENCE_MISMATCH")
    candidates = candidate_matrix()
    gpu_ids = [identifier for identifier, value in candidates.items() if value.device_kind == "gpu"]
    cpu_ids = [identifier for identifier, value in candidates.items() if value.device_kind == "cpu"]
    if len(gpu_ids) != 6 or len(cpu_ids) != 1:
        raise RuntimeError("STAGE16C5A_R2_CANDIDATE_BUDGET_VIOLATION")
    candidate_payload = {
        identifier: contract.as_dict() for identifier, contract in candidates.items()
    }
    matrix_without_hash = {
        "schema_version": "stage16c5a_r2_physx_contract_matrix_v1",
        "matrix_frozen": True,
        "frozen_before_candidate_execution": True,
        "gpu_candidate_count": len(gpu_ids),
        "cpu_diagnostic_count": len(cpu_ids),
        "candidate_order": ["G0", "G1", "G2", "G3", "G4", "G5", "C0"],
        "shared_contract_constraints": {
            "clips": ["hocap_170105", "hocap_170650"],
            "per_clip_or_per_env_variants": False,
            "reference_time_scale": 8,
            "physics_dt_hz": 120,
            "control_dt_hz": 20,
            "decimation": 6,
            "action_dimension": 26,
            "observation_dimension": 764,
            "hard_caps_or_tolerance_formula_changed": False,
        },
        "candidates": candidate_payload,
    }
    matrix_hash = hashlib.sha256(
        json.dumps(matrix_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    matrix = {**matrix_without_hash, "matrix_sha256": matrix_hash}
    inputs = {
        "api_audit": _input_record(args.api_audit),
        "frozen_inputs": _input_record(args.frozen_inputs),
        "r1_summary": _input_record(args.r1_summary),
        "r1_noise": _input_record(args.r1_noise),
        "r1_tolerances": _input_record(args.r1_tolerances),
        "r1_qualification": _input_record(args.r1_qualification),
        "r1_transitions": _input_record(args.r1_transitions),
    }
    legacy = _load(Path(args.frozen_inputs).resolve())
    asset_hashes = legacy.get("verification", {})
    archive_name = (
        "stage16c5a_r2_baseline_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{_sha256(Path(args.r1_summary))[:8]}"
    )
    archive = args.archive_root.resolve() / archive_name
    if archive.exists():
        raise FileExistsError(f"STAGE16C5A_R2_ARCHIVE_ALREADY_EXISTS: {archive}")
    baseline = {
        "identifier": "physx_factor8_baseline_v1",
        "status": "HISTORIC_C3_C4_VALIDATED_C5A_NATURAL_BASELINE_FAILED",
        "contract": candidates["G0"].as_dict(),
        "source_inputs": inputs,
        "parent_contract_hash": candidates["G0"].config_hash,
    }
    nondeterminism = {
        "status": r1_summary["status"],
        "reason": r1_summary["reason"],
        "natural_baseline": r1_summary["natural_baseline"],
        "same_process_33_environment_divergence": r1_summary["e2_vector_same_process_raw_diverged"],
        "controls": {
            "single_environment": r1_summary["e1_single_env_same_process_zero_error"],
            "origin": r1_summary["e3_origin_status"],
            "one_env_cross_process": r1_summary["e4_cross_process_status"],
            "vector_cross_process": r1_summary["e5_cross_process_status"],
            "telemetry": r1_summary["e6_telemetry_status"],
        },
        "evidence_inputs": inputs,
    }
    _write(output_dir / "frozen_baseline_contract.json", baseline)
    _write(output_dir / "frozen_nondeterminism_evidence.json", nondeterminism)
    _write(output_dir / "asset_hashes.json", asset_hashes)
    _write(output_dir / "reference_hashes.json", legacy.get("runtime", {}))
    _write(output_dir / "controller_hashes.json", legacy.get("runtime", {}))
    _write(output_dir / "physics_hashes.json", {"g0": candidates["G0"].as_dict()})
    _write(output_matrix, matrix)
    _write(archive / "frozen_baseline_contract.json", baseline)
    _write(archive / "frozen_nondeterminism_evidence.json", nondeterminism)
    _write(archive / "asset_hashes.json", asset_hashes)
    _write(archive / "reference_hashes.json", legacy.get("runtime", {}))
    _write(archive / "controller_hashes.json", legacy.get("runtime", {}))
    _write(archive / "physics_hashes.json", {"g0": candidates["G0"].as_dict()})
    _write(
        archive / "README.md",
        "# Stage 16-C.5A-R2 frozen baseline archive\n\n"
        "G0 remains `physx_factor8_baseline_v1`: historical C3/C4 evidence is retained, "
        "while the same-scene C5A natural baseline failure is preserved. R2 candidates are "
        "a versioned child experiment and never overwrite this archive.\n",
    )
    print(
        json.dumps(
            {
                "status": "STAGE16C5A_R2_INPUTS_AND_CANDIDATE_MATRIX_FROZEN",
                "matrix_sha256": matrix_hash,
                "archive": str(archive),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
