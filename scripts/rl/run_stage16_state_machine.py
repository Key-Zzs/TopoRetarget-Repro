#!/usr/bin/env python3
"""Write a bounded Stage-16 recovery log for actual failed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toporetarget.rl.failure_classifier import FailureClass
from toporetarget.rl.state_machine import Stage16RecoveryStateMachine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    machine = Stage16RecoveryStateMachine.from_jsonl(args.log)
    machine.record(
        phase="reference_inventory",
        failure_class=FailureClass.DATA_UNAVAILABLE,
        evidence={
            "qualified_hocap_robot_reference": False,
            "reason": (
                "Post-source-contract Stage-12 retarget reference has not been "
                "regenerated; old formal artifact is invalidated."
            ),
        },
        repair="do_not_use_invalidated_artifact; require approved Stage16-local regeneration",
        rerun_scope="reference_export_gate",
        result="BLOCKED_PRESERVED_EVIDENCE",
    )
    machine.record(
        phase="penspin_inventory",
        failure_class=FailureClass.DATA_UNAVAILABLE,
        evidence={"dataset_found": False, "paper_dataset": "self-collected"},
        repair="record_unavailable_without_substitution",
        rerun_scope="none",
        result="STAGE16_PENSPIN_DATA_UNAVAILABLE",
    )
    machine.write_jsonl(args.log)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(machine.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(machine.summary(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
