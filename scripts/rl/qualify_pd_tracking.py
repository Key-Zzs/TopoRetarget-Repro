#!/usr/bin/env python3
"""Record global non-learning action-scale qualification candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.actuators import (
    ACTION_SCALE_CANDIDATES,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    # A missing mapped object/reference makes the physical qualification fail closed.
    report = {
        "status": "PD_QUALIFICATION_BLOCKED_REFERENCE_OR_OBJECT_ASSET",
        "candidates": list(ACTION_SCALE_CANDIDATES),
        "selection": None,
        "reason": (
            "No provenance-complete Stage16 HO-Cap robot reference and per-object "
            "collision asset are available."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
