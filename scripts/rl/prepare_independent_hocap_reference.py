#!/usr/bin/env python3
"""Materialize one raw-derived world-wrist reference with a durable receipt.

This is a deliberately thin production boundary around the existing Stage-12
to world-wrist exporter.  It does not alter retarget data or reuse a source
actor: an independent source-policy lineage consumes the output explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import atomic_write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--final-trajectory", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--world-reference-output", type=Path, required=True)
    parser.add_argument("--object-mesh-output", type=Path, required=True)
    parser.add_argument("--wuji-mjcf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.clip or any(token in args.clip for token in ("/", "\\", "..")):
        raise ValueError("INDEPENDENT_REFERENCE_CLIP_ID_INVALID")
    inputs = {
        "final_trajectory": args.final_trajectory.resolve(),
        "canonical": args.canonical.resolve(),
        "checkpoint_manifest": args.checkpoint_manifest.resolve(),
        "wuji_mjcf": args.wuji_mjcf.resolve(),
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(f"INDEPENDENT_REFERENCE_INPUT_MISSING:{name}:{path}")
    exporter_report = args.report.with_name(f"{args.report.stem}.exporter.json")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/rl/export_stage16_world_wrist_reference.py"),
        "--final-trajectory",
        str(inputs["final_trajectory"]),
        "--canonical",
        str(inputs["canonical"]),
        "--checkpoint-manifest",
        str(inputs["checkpoint_manifest"]),
        "--output",
        str(args.world_reference_output.resolve()),
        "--object-mesh-output",
        str(args.object_mesh_output.resolve()),
        "--wuji-mjcf",
        str(inputs["wuji_mjcf"]),
        "--report",
        str(exporter_report.resolve()),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            "INDEPENDENT_REFERENCE_EXPORT_FAILED:"
            f"exit={result.returncode}:stderr={result.stderr[-4000:]}"
        )
    output = args.world_reference_output.resolve()
    object_mesh = args.object_mesh_output.resolve()
    if not output.is_file() or not object_mesh.is_file() or not exporter_report.is_file():
        raise RuntimeError("INDEPENDENT_REFERENCE_EXPORT_OUTPUT_MISSING")
    exporter = json.loads(exporter_report.read_text(encoding="utf-8"))
    if exporter.get("status") != "STAGE16B_WORLD_REFERENCE_VALIDATED":
        raise RuntimeError("INDEPENDENT_REFERENCE_EXPORT_NOT_VALIDATED")
    receipt = {
        "schema_version": "IndependentHOCapReferencePreparationV1",
        "status": "COMPLETE",
        "clip_id": args.clip,
        "authority": "scripts/rl/export_stage16_world_wrist_reference.py",
        "input_hashes": {
            name: _sha256(path / "zarr.json") if path.is_dir() else _sha256(path)
            for name, path in inputs.items()
            if name != "wuji_mjcf"
        }
        | {"wuji_mjcf": _sha256(inputs["wuji_mjcf"])},
        "outputs": {
            "world_reference": str(output),
            "world_reference_sha256": _sha256(output),
            "object_mesh": str(object_mesh),
            "object_mesh_sha256": _sha256(object_mesh),
            "exporter_report": str(exporter_report.resolve()),
            "exporter_report_sha256": _sha256(exporter_report),
        },
        "productive_run_seconds": elapsed,
        "technical_retry_seconds": 0.0,
        "cache_hit": False,
        "retry_count": 0,
        "exit_code": result.returncode,
        "exporter_status": exporter["status"],
    }
    atomic_write_json(args.report.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
