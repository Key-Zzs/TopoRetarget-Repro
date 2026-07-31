#!/usr/bin/env python3
"""Summarize all provided policy evaluation episodes without success-only filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.evaluation import EpisodeMetrics, summarize_episodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="JSON list of episode metric mappings")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    values = json.loads(args.episodes.read_text(encoding="utf-8"))
    report = summarize_episodes([EpisodeMetrics(**item) for item in values])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
