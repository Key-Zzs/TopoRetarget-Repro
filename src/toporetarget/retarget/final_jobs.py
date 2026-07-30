"""Operator controls and bounded runtime policy for final-refinement jobs.

This module deliberately does not decide what a final job is allowed to solve.
It only provides a fail-closed pause sentinel, atomic state/heartbeat writes,
and an explicit CPU-thread budget which callers apply *before* NumPy/SciPy or
Torch are imported.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

PAUSE_STATE = "PAUSED_BY_OPERATOR_CONTROL"
CONTROL_SCHEMA = "toporetarget.final_jobs.control.v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def control_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".local" / "control" / "final_jobs"


def runtime_root(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".local" / "runtime" / "final_jobs"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def pause_final_jobs(root: Path | None = None, *, reason: str) -> dict[str, Any]:
    """Install the persistent fail-closed pause sentinel atomically."""

    destination = control_root(root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "PAUSED").write_text(f"{PAUSE_STATE}\n", encoding="utf-8")
    payload = {
        "schema_version": CONTROL_SCHEMA,
        "state": PAUSE_STATE,
        "reason": reason,
        "timestamp_unix_s": time.time(),
        "new_final_tasks_allowed": False,
        "full_stage12_resumed": False,
    }
    _atomic_json(destination / "pause_manifest.json", payload)
    _atomic_json(
        destination / "scheduler_state.json",
        {
            "schema_version": CONTROL_SCHEMA,
            "state": PAUSE_STATE,
            "new_final_tasks_allowed": False,
            "max_final_workers": 0,
        },
    )
    return payload


def paused(root: Path | None = None) -> bool:
    return (control_root(root) / "PAUSED").is_file()


def assert_final_jobs_allowed(root: Path | None = None) -> None:
    if paused(root):
        raise FinalJobPaused(PAUSE_STATE)


class FinalJobPaused(RuntimeError):
    """Raised before a final task consumes a queue item or starts a worker."""


@dataclass(frozen=True)
class FinalRefinementCPUConfig:
    """Versioned thread/process policy; physical core count is authoritative."""

    max_workers: int = 1
    blas_threads: int = 1
    torch_threads: int = 1
    torch_interop_threads: int = 1
    cpu_affinity: str = "none"
    heartbeat_interval: float = 30.0
    stop_after_frame: bool = True
    pause_control_path: str = ".local/control/final_jobs"

    @classmethod
    def load(cls, path: Path | None = None) -> FinalRefinementCPUConfig:
        source = path or (repo_root() / "configs" / "runtime" / "final_refinement_cpu_v1.yaml")
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        values = {key: raw[key] for key in cls.__dataclass_fields__ if key in raw}
        result = cls(**values)
        if result.max_workers < 1 or result.blas_threads < 1:
            raise ValueError("final-refinement worker and BLAS limits must be positive")
        physical = physical_cpu_cores()
        if result.max_workers > max(1, physical // 4):
            raise ValueError("max_final_workers exceeds physical-core safety cap")
        if result.cpu_affinity not in {"none", "auto-disjoint"}:
            raise ValueError("cpu_affinity must be none or auto-disjoint")
        return result


def physical_cpu_cores() -> int:
    """Best-effort physical-core count, never the logical CPU count by default."""

    pairs: set[tuple[str, str]] = set()
    cpu_root = Path("/sys/devices/system/cpu")
    for topology in cpu_root.glob("cpu[0-9]*/topology"):
        try:
            pairs.add(
                (
                    (topology / "physical_package_id").read_text().strip(),
                    (topology / "core_id").read_text().strip(),
                )
            )
        except OSError:
            continue
    return len(pairs) or max(1, (os.cpu_count() or 1) // 2)


def configure_cpu_runtime(config: FinalRefinementCPUConfig) -> dict[str, Any]:
    """Set BLAS env before imports; configure Torch only if it is already usable."""

    value = str(config.blas_threads)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = value
    torch_state: dict[str, Any] = {"available": False}
    try:
        import torch

        torch.set_num_threads(config.torch_threads)
        torch.set_num_interop_threads(config.torch_interop_threads)
        torch_state = {
            "available": True,
            "threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "physical_cpu_cores": physical_cpu_cores(),
        "config": asdict(config),
        "blas_environment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "torch": torch_state,
    }


def append_heartbeat(
    job_id: str, event: str, payload: dict[str, Any], *, root: Path | None = None
) -> None:
    """Append one JSONL event; a heartbeat never rewrites formal artifacts."""

    destination = runtime_root(root) / job_id / "heartbeat.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": "toporetarget.final_jobs.heartbeat.v1",
        "event": event,
        "timestamp_unix_s": time.time(),
        "job_id": job_id,
        **payload,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


__all__ = [
    "FinalJobPaused",
    "FinalRefinementCPUConfig",
    "PAUSE_STATE",
    "append_heartbeat",
    "assert_final_jobs_allowed",
    "configure_cpu_runtime",
    "control_root",
    "pause_final_jobs",
    "paused",
    "physical_cpu_cores",
    "runtime_root",
]
