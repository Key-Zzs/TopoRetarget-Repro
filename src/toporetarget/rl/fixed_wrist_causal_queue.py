"""Durable state helpers for the fixed-wrist causal PPO queue.

The queue runner itself lives under :mod:`scripts` because it owns Isaac Sim
processes.  These helpers intentionally have no Isaac or torch dependency so
the state-machine rules stay unit-testable.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .physical_p3 import PHYSICAL_STAGE_BUDGETS

QUEUE_LINEAGES: tuple[dict[str, str], ...] = (
    {"id": "v3_hocap_170105", "directory": "v3", "mode": "aggregate_v3", "clip": "hocap_170105"},
    {
        "id": "v4_hocap_170105",
        "directory": "v4",
        "mode": "strict_per_finger_v4",
        "clip": "hocap_170105",
    },
    {"id": "v3_hocap_170650", "directory": "v3", "mode": "aggregate_v3", "clip": "hocap_170650"},
    {
        "id": "v4_hocap_170650",
        "directory": "v4",
        "mode": "strict_per_finger_v4",
        "clip": "hocap_170650",
    },
)

STAGES: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with a UTF-8 JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_sequence(*, c0_reusable: bool) -> tuple[str, ...]:
    """Return the only legal stage sequence for one lineage."""

    return STAGES[1:] if c0_reusable else STAGES


def initial_lineage_state(*, c0_reusable: bool) -> dict[str, Any]:
    """Build a fail-closed state entry for one queued lineage."""

    stages = {stage.lower(): "NOT_STARTED" for stage in STAGES}
    if c0_reusable:
        stages["c0"] = "REUSED"
    return {"c0_reuse": c0_reusable, **stages, "latest_checkpoint": None, "warnings": []}


def completed_stage_count(per_lineage: Mapping[str, Mapping[str, Any]]) -> int:
    """Count completed/reused stages for the monitor without inferring progress."""

    complete = {"COMPLETE", "REUSED"}
    return sum(
        int(lineage.get(stage.lower()) in complete)
        for lineage in per_lineage.values()
        for stage in STAGES
    )


def stage_budget_samples(stage: str) -> int:
    """Expose the existing immutable stage budget to queue consumers."""

    return PHYSICAL_STAGE_BUDGETS[stage].additional_samples


def all_four_c4_complete(per_lineage: Mapping[str, Mapping[str, Any]]) -> bool:
    """Require every named lineage to have a real C4 endpoint."""

    return len(per_lineage) == len(QUEUE_LINEAGES) and all(
        lineage.get("c4") == "COMPLETE" for lineage in per_lineage.values()
    )


__all__ = [
    "QUEUE_LINEAGES",
    "STAGES",
    "all_four_c4_complete",
    "atomic_write_json",
    "completed_stage_count",
    "initial_lineage_state",
    "stage_budget_samples",
    "stage_sequence",
]
