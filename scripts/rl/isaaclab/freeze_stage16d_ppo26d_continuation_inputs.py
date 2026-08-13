#!/usr/bin/env python3
"""Freeze clip-isolated development and formal evaluation seeds for PPO-26D."""

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


def seed_set_names(clip: str) -> tuple[str, str]:
    """Return the stable, clip-qualified names used by the evaluator and R7."""

    # Keep the already-used 170650 names byte-for-byte stable: they identify
    # the L0/R6 development history and its reserved formal holdout.  A second
    # object may never inherit either set, especially not its formal seeds.
    if clip == "hocap_170650":
        return "development_eval_seed_set_v1", "formal_holdout_seed_set_v1"
    short = clip.removeprefix("hocap_")
    return f"development_eval_seed_set_{short}_v1", f"formal_holdout_seed_set_{short}_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), default="hocap_170650")
    args = parser.parse_args()
    root = args.output_root.resolve()
    development_name, formal_name = seed_set_names(args.clip)
    base_seed = int(args.clip.removeprefix("hocap_"))
    development = generate_frozen_seed_set(
        development_name,
        base_seed=base_seed,
        count=20,
        purpose=f"{args.clip} development checkpoint trend evaluation only",
    )
    formal = generate_frozen_seed_set(
        formal_name,
        base_seed=base_seed + 1,
        count=20,
        purpose=f"unseen {args.clip} R7 frame-zero formal qualification only",
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
        "schema_version": "Stage16DPPO26DContinuationInputsV2",
        "clip": args.clip,
        development_name: development.as_dict(),
        formal_name: formal.as_dict(),
        "formal_holdout_use_before_r7": "FORBIDDEN",
        "cross_clip_reuse": "FORBIDDEN",
        "transitions": transitions.transitions,
    }
    suffix = "" if args.clip == "hocap_170650" else f"_{args.clip.removeprefix('hocap_')}"
    destination = root / f"frozen_seed_sets{suffix}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "status": "INPUT_FREEZE_COMPLETE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
