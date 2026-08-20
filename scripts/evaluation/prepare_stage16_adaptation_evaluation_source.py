#!/usr/bin/env python3
"""Prepare one immutable adapted-checkpoint source for frozen physical evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ADAPTATION_EVALUATION_SOURCE_JSON_REQUIRED:{path}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-source-receipt", type=Path, required=True)
    parser.add_argument("--frozen-source-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adaptation-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"ADAPTATION_EVALUATION_SOURCE_OUTPUT_EXISTS:{output}")
    if not args.adaptation_label.replace("_", "").isalnum():
        raise ValueError("ADAPTATION_EVALUATION_SOURCE_LABEL_INVALID")
    source_receipt = args.frozen_source_receipt.resolve()
    frozen = _read(source_receipt)
    sources = frozen.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get(args.frozen_source_id), dict):
        raise ValueError("ADAPTATION_EVALUATION_SOURCE_FROZEN_SOURCE_UNKNOWN")
    original = dict(sources[args.frozen_source_id])
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ADAPTATION_EVALUATION_SOURCE_CHECKPOINT_MISSING:{checkpoint}")

    # These helpers import Torch-only evaluation code; Isaac is never started.
    from scripts.rl.isaaclab.evaluate_stage16d_ppo26d import model_from_checkpoint
    from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
        _component_hashes,
    )

    trainer, payload = model_from_checkpoint(checkpoint, "cpu", expected_clip=str(original["clip"]))
    hashes = _component_hashes(trainer)
    source_id = f"{args.frozen_source_id}_{args.adaptation_label}"
    source = {
        "id": source_id,
        "reward": original["reward"],
        "contact_mode": original["contact_mode"],
        "clip": original["clip"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "actor_hash": hashes["actor"],
        "normalizer_hash": hashes["normalizer"],
        "schema_version": payload["schema_version"],
        "frozen_source": {
            "receipt": {"path": str(source_receipt), "sha256": _sha256(source_receipt)},
            "id": original["id"],
            "checkpoint": original["checkpoint"],
            "checkpoint_sha256": original["checkpoint_sha256"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "Stage16AdaptationEvaluationSourceV1",
                "sources": {source_id: source},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ADAPTATION_EVALUATION_SOURCE_COMPLETE", "source": source_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
