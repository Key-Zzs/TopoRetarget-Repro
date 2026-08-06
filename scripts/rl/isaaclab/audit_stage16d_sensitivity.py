#!/usr/bin/env python3
"""Guard nominal-first Stage 16-D sensitivity execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluations = [json.loads(path.read_text()) for path in args.ppo_evaluation]
    nominal_pass = bool(evaluations) and all(
        row["status"] == "STAGE16D_SINGLE_CLIP_PPO_VALIDATED" for row in evaluations
    )
    result = {
        "schema_version": "Stage16DSensitivityAuditV1",
        "status": "STAGE16D_SENSITIVITY_NOT_RUN",
        "nominal_ppo_pass": nominal_pass,
        "one_factor_at_a_time": True,
        "frozen_factors": {
            "mass": [0.5, 1.0, 2.0],
            "inertia": [0.5, 1.0, 2.0],
            "friction": [0.5, 1.0, 1.5],
            "controller_effort": [0.8, 1.0, 1.2],
        },
        "reason": "nominal PPO must pass before sensitivity",
        "sim_to_real_claim": False,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
