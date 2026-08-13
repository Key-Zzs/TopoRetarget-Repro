#!/usr/bin/env python3
"""Freeze the existing V3 Formal20 physics gates for V4-only comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CLIPS = ("hocap_170105", "hocap_170650")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_GATE_FREEZE_JSON_OBJECT_REQUIRED:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v3-qualification",
        type=Path,
        nargs=2,
        action="append",
        required=True,
        metavar=("CLIP", "QUALIFICATION"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {str(clip): path.resolve() for clip, path in args.v3_qualification}
    if set(inputs) != set(CLIPS) or len(inputs) != len(args.v3_qualification):
        raise ValueError("STRICT_V4_GATE_FREEZE_REQUIRES_TWO_UNIQUE_CLIPS")
    gates: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    for clip, path in sorted(inputs.items()):
        value = _read(path)
        if (
            value.get("clip") != clip
            or value.get("kind") != "formal"
            or not isinstance(value.get("task_gate"), dict)
            or not isinstance(value.get("contact_topology"), dict)
        ):
            raise ValueError(f"STRICT_V4_GATE_FREEZE_INPUT_INVALID:{clip}")
        gates[clip] = value["task_gate"]
        topology[clip] = value["contact_topology"]
        receipts[clip] = {"path": str(path), "sha256": _hash(path)}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "Stage16DStrictPerFingerV4EvaluationGateFreezeV1",
                "status": "STRICT_V4_EVALUATION_GATES_FROZEN",
                "source": "frozen_V3_Formal20_task_and_topology_gates_only",
                "inputs": receipts,
                "task_gates": {"clips": gates},
                "contact_topology": {"clips": topology},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "STRICT_V4_EVALUATION_GATES_FROZEN", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
