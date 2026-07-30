#!/usr/bin/env python3
"""Freeze the inherited v2/v3 evidence without modifying its worktree."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    base = root.parent / "TopoRetarget-Repro-compiled-kernel"
    v2_path = base / ".local/reports/compiled_sdf_cpu_v1/five_frame_qualification.json"
    v2_v3 = json.loads(v2_path.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    profiles = {
        name: {
            "sha256": _sha256(root / "configs/retarget/refinement_execution" / f"{name}.yaml"),
            "path": str(root / "configs/retarget/refinement_execution" / f"{name}.yaml"),
        }
        for name in (
            "wuji_continuous_sequential_fast_exact_v2",
            "wuji_continuous_sequential_fast_exact_v3_compiled_cpu",
            "wuji_continuous_sequential_fast_exact_v4_compiled_sign",
        )
    }
    rows = list(v2_v3.get("five_frame", []))
    payload = {
        "schema_version": "toporetarget.compiled_exact_sign.frozen_baseline.v1",
        "base_commit": commit,
        "source_artifact_read_only": str(v2_path),
        "fixed_frames": [0, 12, 29, 45, 59],
        "profiles": profiles,
        "inherited_v2_v3_qualification": rows,
        "source_status": v2_v3.get("status"),
    }
    output = root / ".local/reports/compiled_exact_sign_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_baseline_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "frozen_baseline_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["frame"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
