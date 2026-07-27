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
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--asset-id", default="artimano", choices=["artimano", "wuji_hand2_beta1"])
    parser.add_argument("--upstream-ref", default="release/v2026.7.23")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--imported-at")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    try:
        if args.asset_id == "wuji_hand2_beta1":
            from toporetarget.robots.vendor import vendor_wuji_hand2_beta1

            destination = args.destination or Path("third_party/robot_hands/wuji_hand2_beta1")
            result = vendor_wuji_hand2_beta1(
                args.source_root,
                destination,
                upstream_ref=args.upstream_ref,
                dry_run=args.dry_run,
                force=args.force,
                imported_at=args.imported_at,
            )
        else:
            from toporetarget.paths.assets import vendor_artimano

            destination = args.destination or Path("third_party/robot_hands/artimano")
            result = vendor_artimano(
                args.source_root,
                destination,
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
