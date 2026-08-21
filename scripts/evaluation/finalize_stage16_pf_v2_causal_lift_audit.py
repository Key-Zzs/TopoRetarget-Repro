#!/usr/bin/env python3
"""Compatibility entrypoint for the corrected PF V2 final report.

The original fail-closed handoff encoded a now-repaired mask-authority bug: it
applied hand-pair force validity to the independent table ContactSensor. Keep
the filename so existing local commands remain useful, but delegate to the
frozen-contract/symmetric-PPO report generator rather than recreating the
obsolete `PF_V2_SEMANTICS_INVALID` stop receipt.
"""

# ruff: noqa: E402, I001

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.finalize_stage16_pf_v2_symmetric_report import main


if __name__ == "__main__":
    raise SystemExit(main())
