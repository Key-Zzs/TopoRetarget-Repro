#!/usr/bin/env python3
"""Materialize the bounded Stage 16-D R6C reset curriculum from RSI telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_continuation import (  # noqa: E402
    RSICurriculumPhase,
    build_rsi_curriculum_distribution,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evaluation must be a JSON object")
    if not isinstance(payload.get("rsi"), list):
        raise ValueError("evaluation has no RSI episode rows")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=[phase.value for phase in RSICurriculumPhase], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation_path = args.evaluation.resolve()
    payload = _read(evaluation_path)
    phase = RSICurriculumPhase(args.phase)
    rows = payload["rsi"]
    frame_count = int(payload["frame_count"])
    distribution = build_rsi_curriculum_distribution(rows, frame_count=frame_count, phase=phase)
    result = {
        "schema_version": "Stage16DPPO26DRSICurriculumManifestV1",
        "clip": payload["requested_clip"],
        "source_evaluation": str(evaluation_path),
        "source_evaluation_sha256": _sha256(evaluation_path),
        "telemetry_episode_count": len(rows),
        **distribution,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"phase": phase.value, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
