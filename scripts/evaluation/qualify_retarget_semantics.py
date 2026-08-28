#!/usr/bin/env python3
"""Qualify one receipt-bound retarget without starting physics or a solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.audit_retarget_semantic_validity import (  # noqa: E402
    _case_metrics,
    _write_csv,
    _write_json,
)
from toporetarget.evaluation.retarget_semantic_validity import (  # noqa: E402
    SemanticGateContractV1,
    SemanticStatus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-frame-csv", type=Path, required=True)
    parser.add_argument(
        "--allow-nonpass",
        action="store_true",
        help=(
            "Diagnostic batch use only; production defaults to a nonzero exit on FAIL/INCONCLUSIVE."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = {
        "canonical": args.canonical.resolve(),
        "warm": args.warm_start.resolve(),
        "final": args.final.resolve(),
        "graph": args.graph.resolve(),
        "evaluation": args.evaluation.resolve(),
    }
    if args.viewer is not None:
        paths["viewer"] = args.viewer.resolve()
    if args.receipt is not None:
        paths["receipt"] = args.receipt.resolve()
    gate = SemanticGateContractV1()
    result = _case_metrics(args.episode_id, paths, gate)
    _write_json(args.output.resolve(), result)
    _write_csv(args.per_frame_csv.resolve(), list(result["per_frame"]))
    gate_path = args.output.resolve().with_name("semantic_gate_contract.json")
    gate_hash_path = args.output.resolve().with_name("semantic_gate_contract_sha256.txt")
    _write_json(gate_path, gate.as_dict())
    gate_hash_path.write_text(gate.sha256 + "\n", encoding="utf-8")
    status = str(result["final"]["qualification"]["status"])
    print(
        json.dumps(
            {
                "episode_id": args.episode_id,
                "status": status,
                "report": str(args.output.resolve()),
                "gate_sha256": gate.sha256,
            },
            indent=2,
        )
    )
    if status != SemanticStatus.PASS.value and not args.allow_nonpass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
