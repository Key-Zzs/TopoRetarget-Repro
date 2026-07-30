#!/usr/bin/env python3
"""Create the required atomic host-window request without touching Stage-12."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    control = root.parent / ".toporetarget_host_control"
    control.mkdir(parents=True, exist_ok=True)
    request_id = f"compiled-exact-sign-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    payload = {
        "request_id": request_id,
        "requesting_branch": "feature/compiled-exact-sign",
        "requesting_worktree": str(root),
        "pid": os.getpid(),
        "benchmark_type": "five_frame_sixty_frame_and_microbenchmark",
        "requested_timestamp": datetime.now(UTC).isoformat(),
        "expected_duration": "30 minutes",
        "required_active_final_workers": 0,
    }
    destination = control / "compiled_exact_sign_benchmark.request.json"
    temporary = destination.with_suffix(".request.json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
