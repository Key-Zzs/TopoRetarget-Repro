#!/usr/bin/env python3
"""Select one shared bounded Stage-16B MPC budget from frozen two-clip reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _score(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
    clips = candidate["h10"]
    return (
        min(float(row["success_rate"]) for row in clips),
        min(float(row["final_reach_rate"]) for row in clips),
        min(float(row["progress"]) for row in clips),
        -max(
            max(
                float(row["object_position_error_cm"]) / 2.0,
                float(row["object_rotation_error_deg"]) / 10.0,
                float(row["max_axis_error_cm"]) / 3.0,
            )
            for row in clips
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 2 <= len(args.candidate_report) <= 4:
        raise ValueError("selection requires a bounded set of two to four candidates")
    candidates = []
    for path in args.candidate_report:
        report = json.loads(path.read_text(encoding="utf-8"))
        h10 = []
        for clip in report["horizons"]["H10"]:
            summary = clip["summary"]
            h10.append(
                {
                    "clip": clip["clip"],
                    **{
                        key: summary[key]
                        for key in (
                            "success_rate",
                            "final_reach_rate",
                            "progress",
                            "object_position_error_cm",
                            "object_rotation_error_deg",
                            "max_axis_error_cm",
                            "termination_distribution",
                        )
                    },
                }
            )
        if len(h10) != 2:
            raise ValueError("every candidate must contain the frozen two H10 clips")
        candidates.append(
            {
                "source_report": str(path.resolve()),
                "config": report["mpc_config"],
                "h10": h10,
            }
        )
    selected = max(candidates, key=_score)
    result = {
        "id": "stage16b_contact_aware_mpc_candidate_selection_v1",
        "status": "SHARED_MPC_CANDIDATE_SELECTED_ORACLE_GATE_STILL_EVALUATED_SEPARATELY",
        "selection_rule": (
            "lexicographic worst-clip H10 success, final reach, progress, then normalized "
            "terminal gate error; never clip-specific"
        ),
        "candidates": candidates,
        "selected": selected,
        "selected_score": list(_score(selected)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected": selected["config"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
