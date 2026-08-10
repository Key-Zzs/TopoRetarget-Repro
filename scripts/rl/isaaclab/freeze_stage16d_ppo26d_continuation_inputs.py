#!/usr/bin/env python3
"""Freeze the reusable development and formal evaluation seeds for PPO-26D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_continuation import (  # noqa: E402
    ContinuationTransitions,
    Stage16DPPO26DContinuationStateMachine,
    generate_frozen_seed_set,
)

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d_continuation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.output_root.resolve()
    development = generate_frozen_seed_set(
        "development_eval_seed_set_v1",
        base_seed=1_706_500,
        count=20,
        purpose="R6A/R6B/R6C/R6D checkpoint trend evaluation only",
    )
    formal = generate_frozen_seed_set(
        "formal_holdout_seed_set_v1",
        base_seed=1_706_501,
        count=20,
        purpose="unseen R7 frame-zero formal qualification only",
    )
    if set(development.seeds).intersection(formal.seeds):
        raise AssertionError("development and formal PPO seed sets must be disjoint")
    transitions = ContinuationTransitions()
    transitions.transition(
        Stage16DPPO26DContinuationStateMachine.DOCS_UPDATE,
        reason="continuation seed contracts frozen before any resumed PPO training",
    )
    transitions.transition(
        Stage16DPPO26DContinuationStateMachine.L0_REBASELINE,
        reason="development seed set ready for the fair L0 rebaseline",
    )
    payload = {
        "schema_version": "Stage16DPPO26DContinuationInputsV1",
        "development_eval_seed_set_v1": development.as_dict(),
        "formal_holdout_seed_set_v1": formal.as_dict(),
        "formal_holdout_use_before_r7": "FORBIDDEN",
        "transitions": transitions.transitions,
    }
    destination = root / "frozen_seed_sets.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "status": "INPUT_FREEZE_COMPLETE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
