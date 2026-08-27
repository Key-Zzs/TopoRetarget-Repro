#!/usr/bin/env python3
"""Verify an H3-B interrupted/resumed run against an uninterrupted run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.report_h3_retarget_throughput import (  # noqa: E402
    _compare_artifacts,
    _compare_validation,
    _write_json,
)
from toporetarget.retarget.refinement_checkpoint import CheckpointStore  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted-final", type=Path, required=True)
    parser.add_argument("--resumed-final", type=Path, required=True)
    parser.add_argument("--uninterrupted-validation", type=Path, required=True)
    parser.add_argument("--resumed-validation", type=Path, required=True)
    parser.add_argument("--interrupted-progress", type=Path, required=True)
    parser.add_argument("--resumed-checkpoint-root", type=Path, required=True)
    parser.add_argument("--interruption-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    interrupted = json.loads(args.interrupted_progress.resolve().read_text(encoding="utf-8"))
    store = CheckpointStore(args.resumed_checkpoint_root.resolve())
    chain = store.validate_chain()
    manifest = store.manifest or {}
    trajectory = _compare_artifacts(
        args.uninterrupted_final.resolve(), args.resumed_final.resolve()
    )
    validation = _compare_validation(
        args.uninterrupted_validation.resolve(), args.resumed_validation.resolve()
    )
    interruption_checks = {
        "paused": interrupted.get("status") == "paused",
        "inclusive_stop_frame_recorded": int(interrupted.get("stop_after_frame", -1))
        == args.interruption_frame,
        "next_frame_is_k_plus_one": int(interrupted.get("next_frame", -1))
        == args.interruption_frame + 1,
        "remaining_frames_positive": int(interrupted.get("remaining_frames", 0)) > 0,
    }
    resumed_checks = {
        "checkpoint_chain_complete": bool(chain.get("complete")),
        "checkpoint_chain_pass": bool(chain.get("chain_pass")),
        "no_invalid_frames": not chain.get("invalid_frames"),
        "no_orphan_frames": not chain.get("orphan_frames"),
        "multiple_elapsed_sessions": int(manifest.get("elapsed_sessions", 0)) >= 2,
    }
    passed = bool(
        all(interruption_checks.values())
        and all(resumed_checks.values())
        and trajectory["status"] == "PASS"
        and validation["status"] == "PASS"
    )
    receipt = {
        "schema_version": "H3RetargetResumeParityV1",
        "status": "PASS" if passed else "FAIL",
        "RESUME_PARITY": "PASS" if passed else "FAIL",
        "interruption_frame_inclusive": args.interruption_frame,
        "interruption_checks": interruption_checks,
        "resumed_checks": resumed_checks,
        "checkpoint_chain": chain,
        "trajectory_parity": trajectory,
        "validation_receipt_parity": validation,
        "uninterrupted_final": str(args.uninterrupted_final.resolve()),
        "resumed_final": str(args.resumed_final.resolve()),
        "interrupted_progress": str(args.interrupted_progress.resolve()),
        "resumed_checkpoint_root": str(args.resumed_checkpoint_root.resolve()),
    }
    _write_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
