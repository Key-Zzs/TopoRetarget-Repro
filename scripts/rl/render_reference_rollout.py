#!/usr/bin/env python3
"""Write a numerical dashboard when a rendered reference rollout is unavailable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.visualization import write_dashboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".local/reports/stage16_reference_tracking_ppo/dashboard.html"),
    )
    parser.add_argument("--input-json", type=Path, default=None)
    args = parser.parse_args()
    payload = (
        json.loads(args.input_json.read_text(encoding="utf-8"))
        if args.input_json is not None
        else {
            "status": "VISUALIZATION_BACKEND_UNAVAILABLE",
            "fallback": "numerical_dashboard",
            "reason": "No eligible dynamic HO-Cap reference exists for rollout rendering.",
        }
    )
    output = write_dashboard(args.output, payload)
    review = {
        "status": "VISUALIZATION_BACKEND_UNAVAILABLE",
        "fallback": "dashboard",
        "dashboard": str(output.resolve()),
        "rendered_physics_claim": False,
    }
    review_path = output.with_name("visual_review.json")
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
