"""Stable schemas and provenance helpers for the A--E quality experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "grab_artimano_quality_v1"
QUALITY_SCHEMA_VERSION = "toporetarget.quality_experiment.v1"
CONTACTPOSE_STATUS = "deferred"


class QualityExperimentError(RuntimeError):
    """Raised when a quality experiment input or hard gate is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: str | Path) -> str | None:
    root = Path(path)
    if not root.exists():
        return None
    digest = hashlib.sha256()
    if root.is_file():
        return file_hash(root)
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash(item).encode("ascii"))  # type: ignore[union-attr]
        digest.update(b"\n")
    return digest.hexdigest()


def git_commit(repo_root: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_json(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ClipSpec:
    """One frozen native-frame GRAB development unit."""

    unit_id: str
    sequence: str
    subject: str
    object_name: str
    start_frame: int
    end_frame: int
    hand: str = "right"
    robot: str = "artimano_rh"
    native_fps: float = 120.0
    contact_regions: tuple[str, ...] = ()

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame

    def validate(self) -> None:
        if self.subject != "s1":
            raise QualityExperimentError(f"{self.unit_id}: subject must be frozen to s1")
        if self.hand != "right" or self.robot != "artimano_rh":
            raise QualityExperimentError(f"{self.unit_id}: expected right/artimano_rh")
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise QualityExperimentError(f"{self.unit_id}: invalid half-open frame range")
        if self.length != 60:
            raise QualityExperimentError(f"{self.unit_id}: quality clips must contain 60 frames")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contact_regions"] = list(self.contact_regions)
        value["frame_range"] = [self.start_frame, self.end_frame]
        value["length"] = self.length
        return value


CLIPS: tuple[ClipSpec, ...] = (
    ClipSpec("G1", "s1/airplane_lift", "s1", "airplane", 240, 300),
    ClipSpec("G2", "s1/apple_eat_1", "s1", "apple", 212, 272),
    ClipSpec("G3", "s1/banana_lift", "s1", "banana", 1658, 1718),
    ClipSpec("G4", "s1/alarmclock_lift", "s1", "alarmclock", 407, 467),
)


@dataclass
class ArtifactRecord:
    path: str
    sha256: str | None = None
    status: str = "unverified"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_label(
    *, paper_method: bool, paper_external_extension: bool, engineering: bool = False
) -> dict[str, Any]:
    return {
        "paper_method": bool(paper_method),
        "paper_external_extension": bool(paper_external_extension),
        "engineering_extension": bool(engineering),
    }


__all__ = [
    "CLIPS",
    "CONTACTPOSE_STATUS",
    "EXPERIMENT_ID",
    "QUALITY_SCHEMA_VERSION",
    "ArtifactRecord",
    "ClipSpec",
    "QualityExperimentError",
    "file_hash",
    "git_commit",
    "profile_label",
    "read_json",
    "stable_hash",
    "tree_hash",
    "utc_now",
    "write_json",
]
