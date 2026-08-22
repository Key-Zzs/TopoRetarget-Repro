#!/usr/bin/env python3
"""Validate and publish replay commands for one terminal independent trace."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--source-policy-receipt", type=Path, required=True)
    parser.add_argument("--support-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--accept-eula", action="store_true")
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"INDEPENDENT_TRACE_EXPORT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"INDEPENDENT_TRACE_EXPORT_ARTIFACT_MISSING:{resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verified(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if _artifact(path)["sha256"] != row.get("sha256"):
        raise RuntimeError("INDEPENDENT_TRACE_EXPORT_UPSTREAM_HASH_DRIFT")
    return path


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("INDEPENDENT_TRACE_EXPORT_REQUIRES_EULA")
    if not args.clip_id or any(token in args.clip_id for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_TRACE_EXPORT_CLIP_ID_INVALID")
    final_path = args.final_result.resolve()
    source_path = args.source_policy_receipt.resolve()
    support_path = args.support_receipt.resolve()
    final = _json(final_path)
    source = _json(source_path)
    support = _json(support_path)
    if any(
        row.get("status") != "PASS" or row.get("clip_id") != args.clip_id
        for row in (final, source, support)
    ):
        raise ValueError("INDEPENDENT_TRACE_EXPORT_UPSTREAM_RECEIPT_INVALID")
    trace = _verified(final["selected_trace"])
    reference = _verified(source["artifacts"]["world_reference"])
    support_proxy = _verified(support["artifacts"]["support_proxy"])
    physical_contract = _json(_verified(support["artifacts"]["physical_contracts"]))
    geometry = _verified(physical_contract["artifacts"]["runtime_geometry"])
    output = args.output_root.resolve()
    receipt_path = output / "trace_export_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"INDEPENDENT_TRACE_EXPORT_REFUSES_OVERWRITE:{receipt_path}")
    validation = output / "headless_replay_validation.json"
    similarity = output / "raw_mocap_similarity.json"
    base = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "toporetarget-isaaclab",
        "python",
        "scripts/rl/isaaclab/replay_physical_hoi_trace.py",
        "--trace",
        str(trace),
        "--manifest",
        str(geometry),
        "--object",
        args.clip_id,
        "--reference",
        str(reference),
        "--support-proxy",
        str(support_proxy),
        "--mocap-similarity-output",
        str(similarity),
        "--accept-eula",
    ]
    headless = [
        *base,
        "--headless",
        "--max-loops",
        "1",
        "--validation-output",
        str(validation),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    tick = time.perf_counter()
    result = subprocess.run(
        headless,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    log = output / "headless_replay.log"
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0 or not validation.is_file() or not similarity.is_file():
        failure = {
            "schema_version": "IndependentPhysicalTraceExportFailureV1",
            "status": "FAIL",
            "clip_id": args.clip_id,
            "returncode": result.returncode,
            "command": headless,
            "log": _artifact(log),
        }
        atomic_write_json(output / "trace_export_failure.json", failure)
        raise RuntimeError("INDEPENDENT_PHYSICAL_TRACE_REPLAY_VALIDATION_FAILED")
    validation_payload = _json(validation)
    if (
        validation_payload.get("status") != "STAGE16D_PPO26D_REPLAY_VALIDATED"
        or validation_payload.get("finite") is not True
    ):
        raise RuntimeError("INDEPENDENT_PHYSICAL_TRACE_REPLAY_RECEIPT_INVALID")
    gui_command = " ".join(shlex.quote(value) for value in base)
    atomic_write_text(
        output / "visualization_commands.md",
        "# Independent physical trace replay\n\n"
        f"Classification: `{final['classification']}`.\n\n"
        "```bash\n"
        f"{gui_command}\n"
        "```\n",
    )
    receipt = {
        "schema_version": "IndependentPhysicalTraceExportReceiptV1",
        "status": "PASS",
        "clip_id": args.clip_id,
        "classification": final["classification"],
        "accepted": final["accepted"],
        "trace": _artifact(trace),
        "headless_validation": _artifact(validation),
        "raw_mocap_similarity": _artifact(similarity),
        "visualization_commands": _artifact(output / "visualization_commands.md"),
        "final_result": _artifact(final_path),
        "headless_command": headless,
        "gui_command": base,
        "log": _artifact(log),
        "productive_run_seconds": time.perf_counter() - tick,
        "technical_retry_seconds": 0.0,
        "retry_count": 0,
        "cache_hit": False,
    }
    atomic_write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
