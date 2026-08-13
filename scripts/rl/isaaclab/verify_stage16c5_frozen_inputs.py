#!/usr/bin/env python3
"""Verify the immutable C.3/C.4 inputs used by the C.5A-R1 repair run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_FROZEN_INPUTS_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(baseline.get("inputs"), dict):
        raise ValueError("C5A frozen baseline is malformed")
    verified: dict[str, Any] = {}
    for name, entry in baseline["inputs"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"C5A frozen input entry is malformed: {name}")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"C5A frozen input hash is malformed: {name}")
        actual = _sha256(REPO_ROOT / relative)
        verified[name] = {
            "path": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected,
        }
    matches = all(entry["matches"] for entry in verified.values())
    report = {
        "schema_version": "stage16c5a_repair_frozen_inputs_v1",
        "baseline_manifest": str(args.baseline),
        "verification": verified,
        "reference_timing": baseline.get("reference_timing"),
        "runtime": baseline.get("runtime"),
        "status": "STAGE16C5A_INPUT_HASHES_MATCH" if matches else "STAGE16C5A_INPUT_HASH_DRIFT",
    }
    _write(args.output, report)
    print(json.dumps({"status": report["status"]}, sort_keys=True))
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
