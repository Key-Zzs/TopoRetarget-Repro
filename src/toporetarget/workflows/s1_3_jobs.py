"""Scope-isolated, file-backed controls for S1.3 bounded jobs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

S1_3_SCOPE = "s1_3"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _root(repo_root: Path, scope: str) -> Path:
    if scope != S1_3_SCOPE:
        raise ValueError(f"unsupported jobs scope {scope!r}; expected {S1_3_SCOPE!r}")
    return repo_root / ".local" / "control" / "s1_3_jobs"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def initialize(repo_root: Path, *, scope: str = S1_3_SCOPE) -> dict[str, Any]:
    root = _root(repo_root, scope)
    root.mkdir(parents=True, exist_ok=True)
    scheduler = root / "scheduler_state.json"
    if not scheduler.exists():
        _atomic_json(
            scheduler,
            {
                "schema_version": "toporetarget.s1_3.jobs.v1",
                "scope": scope,
                "state": "ready",
                "updated_at": _now(),
                "max_workers": 1,
                "no_silent_restart": True,
            },
        )
    active = root / "active_jobs.json"
    if not active.exists():
        _atomic_json(active, {"schema_version": "toporetarget.s1_3.jobs.v1", "jobs": []})
    return status(repo_root, scope=scope)


def pause(repo_root: Path, *, scope: str = S1_3_SCOPE) -> dict[str, Any]:
    initialize(repo_root, scope=scope)
    root = _root(repo_root, scope)
    payload = {"scope": scope, "requested_at": _now(), "stop_after_current_frame": True}
    _atomic_json(root / "pause_manifest.json", payload)
    (root / "PAUSED").touch(exist_ok=True)
    _atomic_json(
        root / "scheduler_state.json",
        {
            "schema_version": "toporetarget.s1_3.jobs.v1",
            "scope": scope,
            "state": "pause_requested",
            "updated_at": _now(),
            "max_workers": 1,
            "no_silent_restart": True,
        },
    )
    return status(repo_root, scope=scope)


def resume(repo_root: Path, *, scope: str = S1_3_SCOPE) -> dict[str, Any]:
    initialize(repo_root, scope=scope)
    root = _root(repo_root, scope)
    (root / "PAUSED").unlink(missing_ok=True)
    _atomic_json(
        root / "scheduler_state.json",
        {
            "schema_version": "toporetarget.s1_3.jobs.v1",
            "scope": scope,
            "state": "ready",
            "updated_at": _now(),
            "max_workers": 1,
            "no_silent_restart": True,
        },
    )
    return status(repo_root, scope=scope)


def status(repo_root: Path, *, scope: str = S1_3_SCOPE) -> dict[str, Any]:
    root = _root(repo_root, scope)
    scheduler_path = root / "scheduler_state.json"
    active_path = root / "active_jobs.json"
    scheduler = json.loads(scheduler_path.read_text()) if scheduler_path.exists() else None
    active = json.loads(active_path.read_text()) if active_path.exists() else {"jobs": []}
    return {
        "scope": scope,
        "control_root": str(root),
        "paused": (root / "PAUSED").exists(),
        "scheduler": scheduler,
        "active_jobs": active.get("jobs", []),
    }


def drain(repo_root: Path, *, scope: str = S1_3_SCOPE) -> dict[str, Any]:
    value = pause(repo_root, scope=scope)
    jobs = value["active_jobs"]
    value["drain_safe"] = all(bool(job.get("checkpoint_path")) for job in jobs)
    value["action"] = "stop_after_current_frame"
    return value


__all__ = ["S1_3_SCOPE", "drain", "initialize", "pause", "resume", "status"]
