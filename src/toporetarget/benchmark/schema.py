"""Versioned benchmark records and deterministic artifact helpers."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_SCHEMA_VERSION = "toporetarget.hoi_benchmark.v1"
METRIC_REGISTRY_VERSION = "toporetarget.metric_registry.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass
class BenchmarkUnit:
    benchmark_id: str
    dataset: str
    native_sample_id: str
    subject: str
    object_name: str
    action: str
    hand: str
    side: str
    frame_range: list[int]
    dynamic: bool
    native_static_grasp: bool
    temporal_metrics_applicable: bool
    native_fps: float | None
    source_path: str
    source_hash: str | None
    object_mesh_path: str | None
    object_mesh_hash: str | None
    contact_annotation_type: str
    contact_annotation_hash: str | None
    contact_regions: list[str] = field(default_factory=list)
    contact_mode: str = "unknown"
    canonical_validity: str = "unchecked"
    sdf_validity: str = "unchecked"
    source_identity: str = ""
    selection_score: dict[str, float] = field(default_factory=dict)
    selection_reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    frozen_selection_rank: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def start_frame(self) -> int:
        return int(self.frame_range[0])

    @property
    def end_frame(self) -> int:
        return int(self.frame_range[1])

    @property
    def contact_class(self) -> str:
        names = {item.lower() for item in self.contact_regions}
        if any("palm" in item or "hand" == item for item in names):
            return "palm/non-tip"
        if any("1" in item or "2" in item for item in names):
            return "non-tip"
        if len(names) <= 2 and any("thumb" in item for item in names):
            return "precision"
        if len(names) >= 3:
            return "power"
        return "transition"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contact_mode"] = self.contact_mode
        value["contact_class"] = self.contact_class
        return value


def write_json(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_units(units: Iterable[BenchmarkUnit], root: str | Path) -> None:
    destination = Path(root)
    rows = [unit.as_dict() for unit in units]
    write_json(rows, destination / "benchmark_units.json")
    fields = [
        "benchmark_id",
        "dataset",
        "native_sample_id",
        "subject",
        "object_name",
        "action",
        "hand",
        "side",
        "dynamic",
        "native_static_grasp",
        "temporal_metrics_applicable",
        "native_fps",
        "source_path",
        "source_hash",
        "object_mesh_path",
        "object_mesh_hash",
        "contact_annotation_type",
        "contact_annotation_hash",
        "contact_class",
        "contact_mode",
        "canonical_validity",
        "sdf_validity",
        "selection_rank",
    ]
    csv_path = destination / "benchmark_units.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "selection_rank": row.get("frozen_selection_rank"),
                    **{key: row.get(key, "") for key in fields if key != "selection_rank"},
                }
            )


def flatten_rows(rows: Iterable[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        result.append({key: row.get(key, "") for key in fields})
    return result


def write_rows_csv(rows: Iterable[dict[str, Any]], path: str | Path, fields: list[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flatten_rows(rows, fields))
