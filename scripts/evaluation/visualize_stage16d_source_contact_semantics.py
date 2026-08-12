#!/usr/bin/env python3
"""Print source-contact diagnostic PLY scenes for manual offline inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    args = parser.parse_args()
    path = args.report / args.clip / "source_geometry_visualization_windows.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload["viewer_command"])
    for window in payload["windows"]:
        if window["status"] == "AVAILABLE":
            print(f"{window['kind']}: {window['scene_ply']}")
        else:
            print(f"{window['kind']}: NO_SUCH_WINDOW")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
