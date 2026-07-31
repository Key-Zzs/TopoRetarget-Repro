#!/usr/bin/env python3
"""Fail-closed validation for a generated Stage16ReferenceClip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.contracts import Stage16ReferenceClip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            Stage16ReferenceClip.from_npz(args.reference).validate(expected_hz=20.0), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
