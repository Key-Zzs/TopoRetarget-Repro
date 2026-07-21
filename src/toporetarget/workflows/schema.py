"""Serializable contracts for Stage 10 plans, runs, nodes, and exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

WORKFLOW_SCHEMA_VERSION = "toporetarget.workflow_run.v1"
REFERENCE_SCHEMA_VERSION = "toporetarget.robot_reference.v1"
WORKFLOW_ID = "grab_to_artimano"
WORKFLOW_VERSION = "1.0.0"

NodeStatus = Literal[
    "planned",
    "running",
    "reused",
    "passed",
    "failed",
    "blocked",
    "pending_human_acceptance",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkflowRequest:
    sequence: str
    index: Path
    hand: str
    robot: str
    start_frame: int | None = None
    end_frame: int | None = None
    auto_contact_window: bool = False
    window_length: int = 60
    minimum_contact_frame_ratio: float = 0.5
    maximum_source_contact_median_distance_m: float = 0.02
    final_contact_sanity_max_distance_m: float = 0.05
    mano_model_root: Path | None = None
    asset_root: Path | None = None
    run_root: Path = Path(".local/runs/grab")
    repo_root: Path = Path(".")
    refinement_solver_profile: str = "scipy_slsqp_active_set_v1"

    def validate(self) -> None:
        if not self.sequence.strip():
            raise ValueError("sequence must be explicit; full-dataset mode is not supported")
        if self.hand not in {"right", "left"}:
            raise ValueError("hand must be right or left")
        expected = "artimano_rh" if self.hand == "right" else "artimano_lh"
        if self.robot != expected:
            raise ValueError(f"robot {self.robot!r} does not match source hand {self.hand!r}")
        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        if not 0.0 <= self.minimum_contact_frame_ratio <= 1.0:
            raise ValueError("minimum_contact_frame_ratio must be in [0,1]")
        if self.maximum_source_contact_median_distance_m <= 0:
            raise ValueError("source contact sanity threshold must be positive")
        if self.final_contact_sanity_max_distance_m <= 0:
            raise ValueError("final contact sanity threshold must be positive")
        if (self.start_frame is None) != (self.end_frame is None):
            raise ValueError("start_frame and end_frame must be supplied together")
        if self.start_frame is not None and self.end_frame is not None:
            if self.start_frame < 0 or self.end_frame <= self.start_frame:
                raise ValueError("frame range is [start,end), with end > start")
            if self.end_frame - self.start_frame != self.window_length:
                raise ValueError("explicit frame range must equal window_length")
        if not self.auto_contact_window and self.start_frame is None:
            raise ValueError("provide an explicit frame range or --auto-contact-window")
        if not self.refinement_solver_profile.strip():
            raise ValueError("refinement_solver_profile must be a registered profile ID")

    @property
    def side(self) -> str:
        return "rh" if self.hand == "right" else "lh"


@dataclass
class NodeState:
    node_id: str
    implementation_version: str
    dependencies: list[str] = field(default_factory=list)
    input_hashes: dict[str, str] = field(default_factory=dict)
    config_hashes: dict[str, str] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    output_hashes: dict[str, str] = field(default_factory=dict)
    expected_signature: str = ""
    actual_signature: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float | None = None
    status: NodeStatus = "planned"
    reused: bool = False
    skipped: bool = False
    invalidation_reason: str | None = None
    validation_status: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class WorkflowPlan:
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    workflow_id: str = WORKFLOW_ID
    workflow_version: str = WORKFLOW_VERSION
    run_id: str = ""
    run_root: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    nodes: list[NodeState] = field(default_factory=list)
    selected_window: dict[str, Any] | None = None
    assumptions: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["nodes"] = [node.as_dict() for node in self.nodes]
        return _json_value(value)


@dataclass
class WorkflowRunManifest:
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    run_id: str = ""
    workflow_id: str = WORKFLOW_ID
    workflow_version: str = WORKFLOW_VERSION
    run_root: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    git_commit: str = "unknown"
    dirty_worktree: bool = False
    repo_root: str = ""
    environment: dict[str, Any] = field(default_factory=dict)
    source_dataset: str = "grab"
    index_path: str = ""
    index_hash: str | None = None
    source_sequence: str = ""
    source_path: str = ""
    source_hash: str | None = None
    mano_model_root: str | None = None
    asset_root: str | None = None
    subject: str | None = None
    object_id: str | None = None
    action: str | None = None
    hand: str = ""
    robot: str = ""
    selected_frame_range: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    native_fps: float | None = None
    contact_window_selection: dict[str, Any] = field(default_factory=dict)
    stage9_window_geometry_audit: dict[str, Any] = field(default_factory=dict)
    semantic_contact_statistics: dict[str, Any] = field(default_factory=dict)
    source_contact_geometry_sanity: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, Any] = field(default_factory=dict)
    config_hashes: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    validations: dict[str, Any] = field(default_factory=dict)
    source_integrity: dict[str, Any] = field(default_factory=dict)
    review_bundle: dict[str, Any] = field(default_factory=dict)
    manual_acceptance: dict[str, Any] = field(default_factory=dict)
    nodes: list[NodeState] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    run_status: str = "planned"
    final_artifact_path: str | None = None
    final_visualization_command: str | None = None
    export_paths: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["nodes"] = [node.as_dict() for node in self.nodes]
        return _json_value(value)


def write_json(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = value.as_dict() if hasattr(value, "as_dict") else _json_value(value)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run_id_for(request: WorkflowRequest) -> str:
    if request.start_frame is None:
        window = "auto"
    else:
        window = f"f{request.start_frame:06d}_f{request.end_frame:06d}"
    sequence = request.sequence.replace("/", "__").replace("\\", "__")
    return f"{sequence}__{request.hand}__{request.robot}__{window}"


__all__ = [
    "NodeState",
    "REFERENCE_SCHEMA_VERSION",
    "WORKFLOW_ID",
    "WORKFLOW_SCHEMA_VERSION",
    "WORKFLOW_VERSION",
    "WorkflowPlan",
    "WorkflowRequest",
    "WorkflowRunManifest",
    "canonical_json",
    "read_json",
    "run_id_for",
    "stable_hash",
    "utc_now",
    "write_json",
]
