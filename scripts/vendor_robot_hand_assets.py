#!/usr/bin/env python3
"""Deterministically vendor tracked robot-hand assets with a license gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("third_party/robot_hands/artimano"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--imported-at")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from toporetarget.paths.assets import vendor_artimano

    try:
        result = vendor_artimano(
            args.source_root,
            args.destination,
            dry_run=args.dry_run,
            force=args.force,
            imported_at=args.imported_at,
        )
    except Exception as exc:  # noqa: BLE001 - standalone tool reports a concise failure.
        parser.error(str(exc))
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
