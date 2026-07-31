#!/usr/bin/env python3
"""Close a granted exact-sign host window without touching its grant history."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    control = root.parent / ".toporetarget_host_control"
    request = control / "compiled_exact_sign_benchmark.request.json"
    granted = control / "compiled_exact_sign_benchmark.granted.json"
    if not request.is_file() or not granted.is_file():
        raise RuntimeError("request and grant must exist before completion")
    request_payload = json.loads(request.read_text(encoding="utf-8"))
    grant_payload = json.loads(granted.read_text(encoding="utf-8"))
    if request_payload["request_id"] != grant_payload.get("request_id"):
        raise RuntimeError("grant request_id does not match request")
    payload = {
        "request_id": request_payload["request_id"],
        "completed_timestamp": datetime.now(UTC).isoformat(),
        "requesting_branch": request_payload["requesting_branch"],
        "requesting_worktree": request_payload["requesting_worktree"],
        "active_final_workers_at_grant": grant_payload.get("active_final_workers"),
        "status": "completed",
    }
    done = control / "compiled_exact_sign_benchmark.done.json"
    temporary = done.with_suffix(".done.json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, done)
    request.unlink()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
