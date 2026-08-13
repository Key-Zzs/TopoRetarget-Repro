#!/usr/bin/env python3
"""Append one fail-closed, evidence-backed Stage 16-D Phase 3 transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.phase3_state import (  # noqa: E402
    Stage16DReferenceKinematicsPhase3StateMachine,
)

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_state(path: Path) -> Stage16DReferenceKinematicsPhase3StateMachine:
    if not path.is_file():
        raise FileNotFoundError(f"PHASE3_TRANSITION_LEDGER_MISSING:{path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("PHASE3_TRANSITION_LEDGER_EMPTY")
    states = tuple(Stage16DReferenceKinematicsPhase3StateMachine)
    last = Stage16DReferenceKinematicsPhase3StateMachine.INPUT_FREEZE
    try:
        for row in rows:
            current = Stage16DReferenceKinematicsPhase3StateMachine(row["state"])
            if states.index(current) < states.index(last):
                raise ValueError("PHASE3_TRANSITION_LEDGER_NONMONOTONIC")
            last = current
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("PHASE3_TRANSITION_LEDGER_LAST_STATE_INVALID") from error
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--state",
        choices=[state.value for state in Stage16DReferenceKinematicsPhase3StateMachine],
        required=True,
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    ledger = root / "failure_transitions.jsonl"
    source = _last_state(ledger)
    target = Stage16DReferenceKinematicsPhase3StateMachine(args.state)
    states = tuple(Stage16DReferenceKinematicsPhase3StateMachine)
    if states.index(target) <= states.index(source):
        raise ValueError("PHASE3_TRANSITION_MUST_ADVANCE_MONOTONICALLY")
    if not args.reason.strip():
        raise ValueError("PHASE3_TRANSITION_REASON_REQUIRED")
    evidence = []
    for supplied in args.evidence:
        path = supplied.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PHASE3_TRANSITION_EVIDENCE_MISSING:{path}")
        evidence.append({"path": str(path), "sha256": _sha256(path)})
    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "from": source.value,
        "state": target.value,
        "reason": args.reason.strip(),
        "evidence": evidence,
    }
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"status": "PHASE3_TRANSITION_RECORDED", "state": target.value}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
