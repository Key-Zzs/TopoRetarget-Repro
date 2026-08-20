#!/usr/bin/env python3
"""Write immutable provenance for a frozen-source C4 Formal20 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"FROZEN_SOURCE_FORMAL20_JSON_OBJECT_REQUIRED:{path}")
    return value


def _receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"FROZEN_SOURCE_FORMAL20_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--support-contract", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"FROZEN_SOURCE_FORMAL20_OUTPUT_EXISTS:{output}")
    source_receipt = args.source_receipt.resolve()
    source_payload = _read(source_receipt)
    sources = source_payload.get("sources")
    if not isinstance(sources, dict) or not isinstance(sources.get(args.source), dict):
        raise ValueError(f"FROZEN_SOURCE_FORMAL20_SOURCE_UNKNOWN:{args.source}")
    source = dict(sources[args.source])
    clip = str(source.get("clip"))
    mode = str(source.get("contact_mode"))
    if mode not in {"aggregate_v3", "strict_per_finger_v4"}:
        raise ValueError("FROZEN_SOURCE_FORMAL20_CONTACT_MODE_INVALID")
    checkpoint = Path(str(source.get("checkpoint"))).resolve()
    if _sha256(checkpoint) != source.get("checkpoint_sha256"):
        raise ValueError("FROZEN_SOURCE_FORMAL20_CHECKPOINT_HASH_DRIFT")
    selection = source.get("selection_receipt")
    if not isinstance(selection, dict):
        raise ValueError("FROZEN_SOURCE_FORMAL20_SELECTION_RECEIPT_MISSING")
    selection_path = Path(str(selection.get("path"))).resolve()
    if _sha256(selection_path) != selection.get("sha256"):
        raise ValueError("FROZEN_SOURCE_FORMAL20_SELECTION_RECEIPT_HASH_DRIFT")
    reference_hashes = source.get("reference_hash")
    if not isinstance(reference_hashes, dict) or not isinstance(reference_hashes.get(clip), str):
        raise ValueError("FROZEN_SOURCE_FORMAL20_REFERENCE_HASH_MISSING")
    reference = args.reference_root.resolve() / f"{clip}.reference_kinematics_v2.npz"
    if _sha256(reference) != reference_hashes[clip]:
        raise ValueError("FROZEN_SOURCE_FORMAL20_REFERENCE_HASH_DRIFT")
    support = args.support_contract.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "Stage16FrozenSourceC4Formal20ConfigV1",
        "artifact_role": "FROZEN_SOURCE_C4_EVALUATION_PROVENANCE",
        "PPO_TRAINING_RUN": False,
        "PPO_OPTIMIZER_STEP": 0,
        "clip": clip,
        "contact_mode": mode,
        "curriculum_stage": "C4",
        "evaluation_reset": "FRAME0_ONLY_FULL_TRAJECTORY",
        "reference_hash": reference_hashes[clip],
        "support_contract_hash": _sha256(support),
        "source_authority": {
            "source_receipt": _receipt(source_receipt),
            "source_id": source["id"],
            "checkpoint": _receipt(checkpoint),
            "actor_hash": source["actor_hash"],
            "normalizer_hash": source["normalizer_hash"],
            "selection_receipt": _receipt(selection_path),
        },
        "reference": _receipt(reference),
        "support_contract": _receipt(support),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN_SOURCE_FORMAL20_CONFIG_COMPLETE", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
