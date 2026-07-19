#!/usr/bin/env python3
"""Standalone wrapper for the local Arti-MANO importer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_importer():
    try:
        from toporetarget.paths.assets import import_artimano
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "src"))
        from toporetarget.paths.assets import import_artimano
    return import_artimano


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    importer = _load_importer()
    try:
        result = importer(
            args.source_root, args.destination, dry_run=args.dry_run, force=args.force
        )
    except Exception as exc:  # noqa: BLE001 - standalone CLI reports a concise failure.
        parser.error(str(exc))
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
