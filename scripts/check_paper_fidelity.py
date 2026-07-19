#!/usr/bin/env python3
"""Validate the paper-to-repository traceability manifest."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        from toporetarget.paper.fidelity import validate_paper_fidelity
    except ModuleNotFoundError:
        sys.path.insert(0, str(repo_root / "src"))
        from toporetarget.paper.fidelity import validate_paper_fidelity
    errors = validate_paper_fidelity(repo_root)
    if errors:
        print("paper fidelity: FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("paper fidelity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
