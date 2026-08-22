#!/usr/bin/env python3
"""Run frozen Eval10/Confirm20 before any independent physical PPO update."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    assert_frozen_manifest,
    atomic_write_json,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--source-policy-receipt", type=Path, required=True)
    parser.add_argument("--support-receipt", type=Path, required=True)
    parser.add_argument("--strict-v4-contract", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_FROZEN_EVALUATION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_FROZEN_EVALUATION_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _receipt_path(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file() or row.get("sha256") != sha256_file(path):
        raise ValueError("INDEPENDENT_FROZEN_EVALUATION_RECEIPT_ARTIFACT_DRIFT")
    return path


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("INDEPENDENT_FROZEN_EVALUATION_REQUIRES_EULA")
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_FROZEN_EVALUATION_CLIP_ID_INVALID")
    manifest_path = args.manifest.resolve()
    manifest = _json(manifest_path)
    assert_frozen_manifest(manifest)
    if len([row for row in manifest["clips"] if row.get("clip_id") == args.clip_id]) != 1:
        raise ValueError("INDEPENDENT_FROZEN_EVALUATION_CLIP_NOT_FROZEN")
    source_path = args.source_policy_receipt.resolve()
    support_path = args.support_receipt.resolve()
    source = _json(source_path)
    support = _json(support_path)
    if (
        source.get("status") != "PASS"
        or source.get("clip_id") != args.clip_id
        or support.get("status") != "PASS"
        or support.get("clip_id") != args.clip_id
        or source.get("selection_manifest_sha256") != manifest["manifest_sha256"]
        or support.get("selection_manifest_sha256") != manifest["manifest_sha256"]
    ):
        raise ValueError("INDEPENDENT_FROZEN_EVALUATION_UPSTREAM_RECEIPT_INVALID")

    reference = _receipt_path(source["artifacts"]["reference_v2"])
    object_usd = _receipt_path(source["artifacts"]["object_usd"])
    object_mesh = _receipt_path(source["artifacts"]["object_mesh"])
    source_training_result = _receipt_path(source["source_training_result"])
    source_contact_receipt = _json(_receipt_path(source["artifacts"]["source_contact"]))
    contact_mask = _receipt_path(source_contact_receipt["artifacts"]["strict_mask"])
    reference_distance = _receipt_path(source_contact_receipt["artifacts"]["reference_distance"])
    if contact_mask.parent != reference_distance.parent:
        raise ValueError("INDEPENDENT_FROZEN_EVALUATION_CONTACT_ROOT_MISMATCH")
    support_proxy = _receipt_path(support["artifacts"]["support_proxy"])
    support_asset = _receipt_path(support["artifacts"]["support_asset"])
    physical_contract_receipt = _json(_receipt_path(support["artifacts"]["physical_contracts"]))
    contract_artifacts = physical_contract_receipt["artifacts"]
    runtime_geometry = _receipt_path(contract_artifacts["runtime_geometry"])
    evaluation_gates = _receipt_path(contract_artifacts["evaluation_gates"])
    seed_manifest = _receipt_path(contract_artifacts["seed_manifest"])
    strict_contract = args.strict_v4_contract.resolve()
    _artifact(strict_contract)

    report_root = args.report_root.resolve() / "clips" / args.clip_id / "physical_refinement"
    run_root = args.run_root.resolve() / args.clip_id / "physical_refinement"
    final_receipt = report_root / "frozen_evaluation_receipt.json"
    if final_receipt.exists():
        raise FileExistsError(f"INDEPENDENT_FROZEN_EVALUATION_REFUSES_OVERWRITE:{final_receipt}")
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "toporetarget-isaaclab",
        "python",
        "scripts/rl/isaaclab/run_physical_refinement.py",
        "evaluate-first",
        "--clip",
        args.clip_id,
        "--source-training-result",
        str(source_training_result),
        "--reference",
        str(reference),
        "--object-usd",
        str(object_usd),
        "--support-proxy",
        str(support_proxy),
        "--support-asset",
        str(support_asset),
        "--contact-contract",
        str(strict_contract),
        "--contact-mask-root",
        str(contact_mask.parent),
        "--reference-distance-root",
        str(reference_distance.parent),
        "--object-mesh-root",
        str(object_mesh.parent),
        "--runtime-geometry-manifest",
        str(runtime_geometry),
        "--frozen-evaluation-gates",
        str(evaluation_gates),
        "--seed-manifest",
        str(seed_manifest),
        "--max-new-updates",
        "15",
        "--report-root",
        str(report_root),
        "--run-root",
        str(run_root),
        "--accept-eula",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    tick = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = report_root / "frozen_evaluation.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    decision_path = report_root / "evaluate_first" / args.clip_id / "decision.json"
    if result.returncode != 0 or not decision_path.is_file():
        failure = {
            "schema_version": "IndependentFrozenPhysicalEvaluationFailureV1",
            "status": "FAIL",
            "clip_id": args.clip_id,
            "returncode": result.returncode,
            "command": command,
            "log": _artifact(log),
        }
        atomic_write_json(report_root / "frozen_evaluation_failure.json", failure)
        raise RuntimeError("INDEPENDENT_FROZEN_PHYSICAL_EVALUATION_FAILED")
    decision = _json(decision_path)
    receipt = {
        "schema_version": "IndependentFrozenPhysicalEvaluationReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "accepted": bool(decision.get("accepted") is True),
        "ppo_required": bool(decision.get("ppo_required") is True),
        "ppo_updates": 0,
        "decision": _artifact(decision_path),
        "source_policy_receipt": _artifact(source_path),
        "support_receipt": _artifact(support_path),
        "command": command,
        "log": _artifact(log),
        "productive_run_seconds": time.perf_counter() - tick,
        "technical_retry_seconds": 0.0,
        "retry_count": 0,
        "cache_hit": False,
    }
    atomic_write_json(final_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
