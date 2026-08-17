#!/usr/bin/env python3
"""Read-only status monitor for the autonomous fixed-wrist Stage16 PPO queue."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / ".local/runs/stage16_fixed_wrist_causal_ppo_rerun"


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _age(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    try:
        then = datetime.fromisoformat(value)
    except ValueError:
        return "invalid"
    return f"{max(0.0, (datetime.now(UTC) - then).total_seconds()):.0f}s"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--json", action="store_true", help="Print the raw state only.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "queue_state.json"
    if not state_path.is_file():
        print(f"QUEUE_STATUS=NOT_STARTED\nRUN_DIR={run_dir}")
        return 1
    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    pid_path = run_dir / "runner.pid"
    pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.is_file() else 0
    done = sum(
        int(value.get(stage) in {"COMPLETE", "REUSED"})
        for value in state.get("per_lineage", {}).values()
        for stage in ("c0", "c1", "c2", "c3", "c4")
    )
    total = 20
    active = state.get("active_lineage") or "none"
    stage = state.get("active_stage") or "none"
    samples_done = int(state.get("stage_samples_done", 0))
    samples_total = int(state.get("stage_samples_total", 0))
    percent = 0.0 if samples_total <= 0 else 100.0 * samples_done / samples_total
    print("QUEUE STATUS")
    print(f"QUEUE_STATUS={state.get('status', 'UNKNOWN')}")
    print(f"RUN_DIR={run_dir}")
    print(f"ACTIVE_LINEAGE={active}")
    print(f"ACTIVE_STAGE={stage}")
    print(f"STAGE_PROGRESS={samples_done}/{samples_total} ({percent:.1f}%)")
    print(f"OVERALL_STAGES={done}/{total}")
    print(f"LINEAGE_PROGRESS={state.get('lineage_index', 0)}/{state.get('lineage_total', 4)}")
    print(f"LATEST_CHECKPOINT={state.get('latest_checkpoint') or 'none'}")
    print(f"LAST_UPDATE_AGE={_age(state.get('last_update_time'))}")
    print(f"RUNNER_PID={pid}")
    print(f"PROCESS_ALIVE={'YES' if _pid_alive(pid) else 'NO'}")
    print(f"TECHNICAL_RETRIES={json.dumps(state.get('technical_retries', {}), sort_keys=True)}")
    for lineage, value in state.get("per_lineage", {}).items():
        stages = ",".join(
            f"{stage}={value.get(stage.lower(), 'UNKNOWN')}"
            for stage in ("C0", "C1", "C2", "C3", "C4")
        )
        print(f"C0_REUSE_{lineage}={'YES' if value.get('c0_reuse') else 'NO'} STAGES={stages}")
    print(f"ALL_FOUR_C4_COMPLETE={state.get('ALL_FOUR_C4_COMPLETE', 'NO')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
