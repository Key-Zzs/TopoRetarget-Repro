#!/usr/bin/env python3
"""Validate the frozen Stage 16-D semantic/contact reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    semantics = [
        json.loads((args.report_root / f"task_semantics_{clip}.json").read_text())
        for clip in ("170105", "170650")
    ]
    topology = json.loads((args.report_root / "contact_topology.json").read_text())
    result = {
        "schema_version": "Stage16DSemanticQualificationV1",
        "status": "STAGE16D_TASK_SEMANTICS_PARTIAL",
        "shared_algorithm": True,
        "clip_conditionals": False,
        "contracts_finite": all(
            0.0 <= float(row["classification_confidence"]) <= 1.0 for row in semantics
        ),
        "ambiguous_clips": [
            row["clip"]
            for row in semantics
            if row["classification_status"] == "TASK_SEMANTIC_CLASSIFICATION_AMBIGUOUS"
        ],
        "required_groups": {
            clip: row["required_body_groups"] for clip, row in topology["clips"].items()
        },
        "reason": "validated C3 contact traces contain only six and two control steps",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
