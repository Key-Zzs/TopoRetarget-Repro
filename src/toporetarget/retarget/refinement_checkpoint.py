"""Atomic frame checkpoints, resume validation, and final assembly for Stage 9.2."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from toporetarget.retarget.final_refinement import (
    FinalFrameResult,
    FinalRetargetTrajectory,
    _FrameContext,
    dynamic_collision_points_numpy,
    final_artifact_hash,
    load_robot_surface_samples,
    save_final_trajectory,
)

CHECKPOINT_SCHEMA_VERSION = "toporetarget.final_retarget_checkpoint.v1"


class CheckpointError(RuntimeError):
    """Raised when checkpoint integrity, identity, or continuity is invalid."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _hash_arrays(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()
        for name, value in sorted(arrays.items())
    }


def _checkpoint_hash(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    clean = dict(metadata)
    clean.pop("per_frame_checkpoint_hash", None)
    return hashlib.sha256(
        _json_bytes({"metadata": clean, "arrays": _hash_arrays(arrays)})
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        values = {name: np.asarray(value) for name, value in arrays.items()}
        cast(Any, np.savez)(temporary, **values)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_fsync_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one crash-durable journal event without rewriting history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _decode(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    return value


def frame_checkpoint_payload(
    local_index: int,
    frame: FinalFrameResult,
    context: _FrameContext,
    *,
    global_frame: int,
    source_frame: int | None = None,
    timestamp: float,
    input_signature: str,
    solver_profile: dict[str, Any],
    execution_profile: dict[str, Any],
    previous_checkpoint_hash: str | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Materialize the complete immutable payload for one accepted frame."""

    model = context.robot_model
    qpos = np.asarray(frame.qpos, dtype=np.float64)
    base = np.asarray(frame.base_pose_scene, dtype=np.float64)
    collision = dynamic_collision_points_numpy(model, context.surface, qpos, base)
    keypoints_base = np.asarray(model.keypoints_base(qpos), dtype=np.float64)
    keypoints_scene = np.asarray(model.keypoints_scene(qpos, base), dtype=np.float64)
    fk = model.forward_kinematics_reference(qpos)
    link_poses = np.stack([base @ fk[name] for name in model.link_names])
    query_ids = np.asarray(frame.query_set.sample_ids, dtype=np.int64)
    reasons = np.asarray(frame.query_set.inclusion_reasons, dtype="S96")
    full_phi = np.asarray(frame.full_signed_distance, dtype=np.float64)
    paper = context.paper
    unqueried = np.setdiff1d(np.arange(len(full_phi)), query_ids, assume_unique=True)
    components = np.asarray(
        [
            frame.breakdown.e_im,
            frame.breakdown.e_bone,
            frame.breakdown.e_temporal,
            frame.breakdown.e_base_pos,
            frame.breakdown.e_base_rot,
            frame.breakdown.e_slack,
            frame.breakdown.weighted_e_im,
            frame.breakdown.weighted_e_bone,
            frame.breakdown.total,
            frame.breakdown.e_morph,
            frame.breakdown.weighted_e_morph,
            frame.breakdown.e_contact_pos,
            frame.breakdown.weighted_e_contact_pos,
            frame.breakdown.e_contact_dir,
            frame.breakdown.weighted_e_contact_dir,
            frame.warm_breakdown.e_im,
            frame.warm_breakdown.e_bone,
            frame.warm_breakdown.total,
        ],
        dtype=np.float64,
    )
    arrays = {
        "qpos": qpos,
        "base_pose_scene": base,
        "base_correction": np.asarray(frame.base_correction, dtype=np.float64),
        "joint_limit_margins": np.minimum(
            qpos - np.asarray(model.joint_lower, dtype=np.float64),
            np.asarray(model.joint_upper, dtype=np.float64) - qpos,
        ),
        "robot_keypoints_base": keypoints_base,
        "robot_keypoints_scene": keypoints_scene,
        "robot_link_poses": np.asarray(link_poses, dtype=np.float64),
        "collision_points_scene": np.asarray(collision, dtype=np.float64),
        "query_ids": query_ids,
        "query_active_round": np.asarray(frame.query_set.active_round, dtype=np.int64),
        "query_inclusion_reason": reasons,
        "slack": np.asarray(frame.slack, dtype=np.float64),
        "signed_distance": np.asarray(frame.signed_distance, dtype=np.float64),
        "hard_residual": np.asarray(frame.hard_residual, dtype=np.float64),
        "soft_residual": np.asarray(frame.soft_residual, dtype=np.float64),
        "full_signed_distance": full_phi,
        "full_closest_points": np.asarray(frame.full_closest_points, dtype=np.float64),
        "full_surface_normals": np.asarray(frame.full_surface_normals, dtype=np.float64),
        "full_hard_residual": full_phi + paper.b,
        "full_soft_violation_count": np.asarray(
            np.count_nonzero(full_phi < -paper.tau - 1e-6), dtype=np.int64
        ),
        "unqueried_soft_violation_count": np.asarray(
            np.count_nonzero(full_phi[unqueried] < -paper.tau - 1e-6), dtype=np.int64
        ),
        "objective_components": components,
    }
    metadata: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "local_frame_index": int(local_index),
        "global_frame_index": int(global_frame),
        "source_global_frame_index": int(global_frame if source_frame is None else source_frame),
        "timestamp": float(timestamp),
        "input_signature": input_signature,
        "solver_profile": solver_profile,
        "execution_profile": execution_profile,
        "query_set_hash": frame.query_set.query_hash,
        "query_count": int(len(query_ids)),
        "active_set_rounds": int(frame.active_set_rounds),
        "optimizer_status_code": int(frame.optimizer_status_code),
        "optimizer_message": str(frame.optimizer_message),
        "optimizer_iterations": int(frame.optimizer_iterations),
        "optimizer_converged": bool(frame.optimizer_converged),
        "strict_accepted": bool(frame.accepted),
        "single_frame_feasible": bool(frame.single_frame_feasible),
        "trajectory_continuous": bool(frame.trajectory_continuous),
        "final_accepted": bool(frame.accepted),
        "continuity_failure_reasons": list(
            frame.continuity_metrics.get("continuity_failure_reasons", [])
        ),
        "initialization_source": frame.initialization_source,
        "retry_attempt": int(frame.retry_attempt),
        "retry_profile": frame.retry_profile,
        "window_used": bool(frame.window_used),
        "continuity_metrics": frame.continuity_metrics,
        "q_clamp_count": int(frame.jacobian_diagnostics.get("q_clamp_count", 0)),
        "solver_success": bool(frame.solver_success),
        "qpos_bounds_pass": bool(frame.qpos_bounds_pass),
        "slack_bounds_pass": bool(frame.slack_bounds_pass),
        "active_constraints_feasible": bool(frame.active_constraints_feasible),
        "full_surface_hard_audit_pass": bool(frame.full_surface_hard_audit_pass),
        "full_surface_soft_audit_pass": bool(frame.full_surface_soft_audit_pass),
        "full_soft_violation_count": int(np.count_nonzero(full_phi < -paper.tau - 1e-6)),
        "unqueried_soft_violation_count": int(
            np.count_nonzero(full_phi[unqueried] < -paper.tau - 1e-6)
        ),
        "active_set_converged": bool(frame.active_set_converged),
        "all_values_finite": bool(frame.all_values_finite),
        "acceptance_policy_id": frame.acceptance_policy_id,
        "acceptance_reason": frame.acceptance_reason,
        "solve_time_s": float(frame.solve_time_s),
        "iterations": int(frame.iterations),
        "function_evaluations": int(frame.function_evaluations),
        "jacobian_evaluations": int(frame.jacobian_evaluations),
        "optimizer_function_evaluations": int(frame.optimizer_function_evaluations),
        "optimizer_jacobian_evaluations": int(frame.optimizer_jacobian_evaluations),
        "initial_objective": float(frame.initial_objective),
        "final_objective": float(frame.final_objective),
        "final_objective_change": float(frame.final_objective_change),
        "final_step_norm": float(frame.final_step_norm),
        "stationarity_checked": bool(frame.stationarity_checked),
        "stationarity_residual": float(frame.stationarity_residual),
        "previous_checkpoint_hash": previous_checkpoint_hash,
        "cache": frame.jacobian_diagnostics.get("cache", {}),
        "timers": frame.jacobian_diagnostics.get("timers", {}),
        "diagnostics": frame.jacobian_diagnostics,
    }
    metadata["per_frame_checkpoint_hash"] = _checkpoint_hash(metadata, arrays)
    return metadata, arrays


def _strict_metadata_pass(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> bool:
    """Recompute the strict frame gate from checkpoint content, never its pass flag."""

    status = int(metadata.get("optimizer_status_code", -1))
    full_phi = np.asarray(arrays["full_signed_distance"], dtype=np.float64)
    hard = np.asarray(arrays["hard_residual"], dtype=np.float64)
    soft = np.asarray(arrays["soft_residual"], dtype=np.float64)
    finite_arrays = True
    for value in arrays.values():
        array = np.asarray(value)
        if array.dtype.kind not in "OUS":
            finite_arrays = finite_arrays and bool(np.all(np.isfinite(array)))
    return bool(
        metadata.get("optimizer_converged", False)
        and status != 9
        and metadata.get("qpos_bounds_pass", False)
        and metadata.get("slack_bounds_pass", False)
        and metadata.get("active_constraints_feasible", False)
        and metadata.get("full_surface_hard_audit_pass", False)
        and metadata.get("full_surface_soft_audit_pass", False)
        and metadata.get("active_set_converged", False)
        and metadata.get("all_values_finite", False)
        and finite_arrays
        and len(hard) == len(soft)
        and np.all(np.isfinite(full_phi))
    )


@dataclass
class CheckpointStore:
    root: Path
    manifest: dict[str, Any] | None = None
    _scan_cache: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()
        if self.manifest is None:
            manifest_path = self.root / "manifest.json"
            if not manifest_path.is_file():
                raise CheckpointError(f"checkpoint manifest not found: {manifest_path}")
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert self.manifest is not None
        self.manifest.setdefault(
            "frame_range",
            [int(self.manifest.get("start_frame", 0)), int(self.manifest.get("end_frame", 0))],
        )
        self.manifest.setdefault(
            "artifact_metadata", dict(self.manifest.get("final_artifact_metadata", {}))
        )

    def initialize(self, manifest: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
        expected = dict(manifest)
        expected.setdefault(
            "frame_range",
            [int(expected.get("start_frame", 0)), int(expected.get("end_frame", 0))],
        )
        expected.setdefault("artifact_metadata", dict(expected.get("final_artifact_metadata", {})))
        opened = self.open(self.root, manifest=expected, resume=resume)
        assert opened.manifest is not None
        self.manifest = opened.manifest
        return dict(self.manifest)

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        manifest: dict[str, Any] | None = None,
        resume: bool = False,
        force: bool = False,
    ) -> CheckpointStore:
        path = Path(root).expanduser()
        existing_path = path / "manifest.json"
        if existing_path.is_file():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if not resume and not force:
                raise CheckpointError("checkpoint run exists; pass --resume to continue")
            if manifest is not None:
                expected = dict(manifest)
                expected.setdefault(
                    "frame_range",
                    [
                        int(expected.get("start_frame", 0)),
                        int(expected.get("end_frame", 0)),
                    ],
                )
                expected.setdefault(
                    "artifact_metadata",
                    dict(expected.get("final_artifact_metadata", {})),
                )
                for key in (
                    "input_signature",
                    "solver_profile_hash",
                    "execution_profile_hash",
                    "query_profile_hash",
                    "frame_range",
                ):
                    if existing.get(key) != expected.get(key):
                        raise CheckpointError(f"checkpoint identity mismatch for {key}")
            existing["elapsed_sessions"] = int(existing.get("elapsed_sessions", 0)) + 1
            _atomic_json(existing_path, existing)
            return cls(path, existing)
        if path.exists() and any(path.iterdir()) and not force:
            raise CheckpointError(f"checkpoint root is non-empty without a manifest: {path}")
        if resume and manifest is None:
            raise CheckpointError("resume requires an expected checkpoint manifest")
        if manifest is None:
            raise CheckpointError("new checkpoint run requires a manifest")
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(exist_ok=True)
        (path / "temporary").mkdir(exist_ok=True)
        (path / "logs").mkdir(exist_ok=True)
        clean = dict(manifest)
        clean.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
        clean.setdefault("created_at", time_now())
        clean.setdefault("elapsed_sessions", 1)
        clean.setdefault(
            "frame_range",
            [int(clean.get("start_frame", 0)), int(clean.get("end_frame", 0))],
        )
        clean.setdefault("artifact_metadata", dict(clean.get("final_artifact_metadata", {})))
        _atomic_json(path / "manifest.json", clean)
        (path / "append_events.jsonl").touch(exist_ok=True)
        _fsync_directory(path)
        store = cls(path, clean)
        store.update_progress(status="created")
        return store

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    def update_progress(
        self, *, status: str = "paused", elapsed_s: float | None = None
    ) -> dict[str, Any]:
        assert self.manifest is not None
        manifest = self.manifest
        scan = self.scan(refresh=False)
        progress = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": manifest.get("run_id"),
            "status": status,
            "accepted_frames": scan["contiguous_frames"],
            "orphan_frames": scan["orphan_frames"],
            "invalid_frames": scan["invalid_frames"],
            "last_accepted_frame": scan["last_contiguous_frame"],
            "next_frame": scan["next_frame"],
            "remaining_frames": max(
                0,
                int(manifest.get("frame_range", [0, 0])[1]) - int(scan["next_frame"]),
            ),
            "frame_range": manifest.get("frame_range"),
            "elapsed_s": elapsed_s,
            "elapsed_sessions": int(manifest.get("elapsed_sessions", 1)),
            "input_signature": manifest.get("input_signature"),
            "resume_command": manifest.get("resume_command"),
        }
        _atomic_json(self.root / "progress.json", progress)
        return progress

    def write_progress(
        self, *, status: str = "paused", manifest: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if manifest is not None:
            self.manifest = dict(manifest)
            self.manifest.setdefault(
                "frame_range",
                [
                    int(self.manifest.get("start_frame", 0)),
                    int(self.manifest.get("end_frame", 0)),
                ],
            )
            self.manifest.setdefault(
                "artifact_metadata", dict(self.manifest.get("final_artifact_metadata", {}))
            )
        return self.update_progress(status=status)

    def save_frame(self, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
        if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("unsupported checkpoint schema")
        if not _strict_metadata_pass(metadata, arrays):
            raise CheckpointError("only strict-accepted frames may enter frames/")
        local = int(metadata["local_frame_index"])
        expected = _checkpoint_hash(metadata, arrays)
        if metadata.get("per_frame_checkpoint_hash") != expected:
            raise CheckpointError("frame checkpoint hash does not match payload")
        destination = self.frames_dir / f"frame_{local:06d}.npz"
        if destination.exists():
            existing = self.load_frame(local)
            if existing[0].get("per_frame_checkpoint_hash") != expected:
                raise CheckpointError(f"refusing to overwrite hash-mismatched frame {local}")
            return expected
        stored = dict(arrays)
        stored["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, default=str))
        _atomic_npz(destination, stored)
        _fsync_directory(self.frames_dir)
        _append_fsync_jsonl(
            self.root / "append_events.jsonl",
            {
                "event": "FRAME_APPENDED",
                "local_frame_index": local,
                "per_frame_checkpoint_hash": expected,
                "previous_checkpoint_hash": metadata.get("previous_checkpoint_hash"),
            },
        )
        self._record_saved_frame(local)
        assert self.manifest is not None
        interval = int(self.manifest.get("durable_checkpoint_interval_frames", 1))
        start = int(self.manifest.get("frame_range", [0, 0])[0])
        if (local - start + 1) % interval == 0:
            self.commit_durable_checkpoint(status="running")
        return expected

    def _record_saved_frame(self, local: int) -> None:
        scan = self.scan(refresh=False)
        found = sorted(set(scan["found_frames"]) | {int(local)})
        assert self.manifest is not None
        start, end = [int(value) for value in self.manifest.get("frame_range", [0, 0])]
        contiguous: list[int] = []
        for value in range(start, end):
            if value not in found:
                break
            contiguous.append(value)
        self._scan_cache = {
            "found_frames": found,
            "contiguous_frames": contiguous,
            "orphan_frames": sorted(set(found) - set(contiguous)),
            "invalid_frames": list(scan["invalid_frames"]),
            "last_contiguous_frame": contiguous[-1] if contiguous else None,
            "next_frame": contiguous[-1] + 1 if contiguous else start,
            "complete": len(contiguous) == end - start,
        }

    def commit_durable_checkpoint(self, *, status: str) -> dict[str, Any]:
        """Atomically publish the latest full-chain resume point every K frames."""

        assert self.manifest is not None
        scan = self.scan(refresh=False)
        last = scan["last_contiguous_frame"]
        last_hash: str | None = None
        if last is not None:
            metadata, _ = self.load_frame(int(last))
            last_hash = str(metadata["per_frame_checkpoint_hash"])
        marker = {
            "schema_version": "toporetarget.durable_retarget_checkpoint.v1",
            "status": status,
            "run_id": self.manifest.get("run_id"),
            "input_signature": self.manifest.get("input_signature"),
            "frame_range": self.manifest.get("frame_range"),
            "durable_checkpoint_interval_frames": int(
                self.manifest.get("durable_checkpoint_interval_frames", 1)
            ),
            "accepted_frame_count": len(scan["contiguous_frames"]),
            "last_accepted_frame": last,
            "last_checkpoint_hash": last_hash,
            "next_frame": scan["next_frame"],
            "append_only": True,
            "historical_sequence_rewrite": False,
        }
        _atomic_json(self.root / "durable_checkpoint.json", marker)
        _fsync_directory(self.root)
        _append_fsync_jsonl(
            self.root / "append_events.jsonl",
            {
                "event": "DURABLE_CHECKPOINT_COMMITTED",
                "status": status,
                "last_accepted_frame": last,
                "last_checkpoint_hash": last_hash,
            },
        )
        return marker

    def write_frame(
        self, metadata: dict[str, Any], arrays: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        checkpoint_hash = self.save_frame(metadata, arrays)
        return {"checkpoint_hash": checkpoint_hash, "metadata": metadata}

    def load_frame(self, local_index: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        path = self.frames_dir / f"frame_{int(local_index):06d}.npz"
        if not path.is_file():
            raise CheckpointError(f"missing checkpoint frame: {path}")
        with np.load(path, allow_pickle=False) as data:
            if "metadata_json" not in data:
                raise CheckpointError(f"checkpoint frame missing metadata: {path}")
            metadata = json.loads(str(_decode(data["metadata_json"])))
            arrays = {
                name: np.asarray(data[name]) for name in data.files if name != "metadata_json"
            }
        if metadata.get("per_frame_checkpoint_hash") != _checkpoint_hash(metadata, arrays):
            raise CheckpointError(f"checkpoint hash mismatch: {path}")
        if not _strict_metadata_pass(metadata, arrays):
            raise CheckpointError(f"checkpoint strict gate failed: {path}")
        return metadata, arrays

    def read_frame(self, local_index: int) -> dict[str, Any]:
        metadata, arrays = self.load_frame(local_index)
        return {"metadata": metadata, "arrays": arrays}

    def scan(self, *, refresh: bool = True) -> dict[str, Any]:
        if self._scan_cache is not None and not refresh:
            return dict(self._scan_cache)
        assert self.manifest is not None
        manifest = self.manifest
        found: list[int] = []
        invalid: list[int] = []
        for path in sorted(self.frames_dir.glob("frame_*.npz")):
            try:
                local = int(path.stem.split("_")[-1])
                self.load_frame(local)
                found.append(local)
            except (CheckpointError, ValueError, OSError):
                try:
                    invalid.append(int(path.stem.split("_")[-1]))
                except ValueError:
                    pass
        start = int(manifest.get("frame_range", [0, 0])[0])
        end = int(manifest.get("frame_range", [0, 0])[1])
        expected_local = list(range(start, end))
        contiguous: list[int] = []
        for local in expected_local:
            if local not in found:
                break
            contiguous.append(local)
        result = {
            "found_frames": sorted(found),
            "contiguous_frames": contiguous,
            "orphan_frames": sorted(set(found) - set(contiguous)),
            "invalid_frames": sorted(invalid),
            "last_contiguous_frame": contiguous[-1] if contiguous else None,
            "next_frame": contiguous[-1] + 1 if contiguous else start,
            "complete": len(contiguous) == len(expected_local),
        }
        self._scan_cache = result
        return dict(result)

    def validate_chain(self, allow_incomplete: bool = False) -> dict[str, Any]:
        scan = self.scan(refresh=True)
        errors: list[str] = []
        previous: str | None = None
        for local in scan["contiguous_frames"]:
            metadata, _ = self.load_frame(local)
            if metadata.get("previous_checkpoint_hash") != previous:
                errors.append(f"frame {local} previous checkpoint hash mismatch")
            previous = str(metadata["per_frame_checkpoint_hash"])
        result = {
            **scan,
            "chain_pass": not errors and not scan["invalid_frames"],
            "errors": errors,
            "last_checkpoint_hash": previous,
        }
        if not allow_incomplete and not result["complete"]:
            result["chain_pass"] = False
        return result

    def status(self) -> dict[str, Any]:
        assert self.manifest is not None
        manifest = self.manifest
        chain = self.validate_chain(allow_incomplete=True)
        return {
            **chain,
            "run_id": manifest.get("run_id"),
            "input_signature": manifest.get("input_signature"),
            "solver_profile_id": manifest.get("solver_profile_id"),
            "execution_profile_id": manifest.get("execution_profile_id"),
            "elapsed_sessions": manifest.get("elapsed_sessions", 0),
            "resume_command": manifest.get("resume_command"),
        }

    def assemble(self, output: str | Path, *, force: bool = False) -> Path:
        assert self.manifest is not None
        manifest = self.manifest
        chain = self.validate_chain()
        if not chain["complete"] or not chain["chain_pass"]:
            raise CheckpointError(f"cannot assemble incomplete checkpoint chain: {chain}")
        rows = [self.load_frame(local) for local in chain["contiguous_frames"]]
        metadata_rows = [item[0] for item in rows]
        payloads = [item[1] for item in rows]
        independent_validation = _independent_source_validation(
            self.manifest, metadata_rows, payloads
        )
        arrays = _assemble_arrays(metadata_rows, payloads)
        artifact_metadata = dict(manifest.get("artifact_metadata", {}))
        execution_profile_id = manifest.get("execution_profile_id")
        execution_profile_hash = manifest.get("execution_profile_hash")
        if execution_profile_id:
            from toporetarget.retarget.refinement_performance import RefinementExecutionProfile

            execution_profile = RefinementExecutionProfile.load(str(execution_profile_id))
            if execution_profile_hash and execution_profile.profile_hash != execution_profile_hash:
                raise CheckpointError("execution profile hash changed before assembly")
            artifact_metadata["execution_profile"] = execution_profile.as_dict()
            artifact_metadata["point_jacobian_backend"] = execution_profile.point_jacobian_backend
            artifact_metadata["strict_recovery"] = execution_profile.strict_recovery
            artifact_metadata["sdf_tree_leaf_size"] = execution_profile.sdf_tree_leaf_size
        artifact_metadata.update(
            {
                "schema_version": manifest.get(
                    "final_artifact_schema", "toporetarget.final_retarget.v2"
                ),
                "artifact_type": "final_interaction_preserving_robot_reference",
                "frame_count": len(rows),
                "frame_range": [
                    int(metadata_rows[0]["global_frame_index"]),
                    int(metadata_rows[-1]["global_frame_index"]) + 1,
                ],
                "timestamps": [float(item["timestamp"]) for item in metadata_rows],
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "checkpoint_root": str(self.root),
                "checkpoint_chain_hash": str(metadata_rows[-1]["per_frame_checkpoint_hash"]),
                "checkpoint_final_independent_validation": independent_validation,
                "artifact_hash": None,
            }
        )
        trajectory = FinalRetargetTrajectory(artifact_metadata, arrays).validate()
        trajectory.metadata["artifact_hash"] = final_artifact_hash(trajectory)
        return save_final_trajectory(trajectory, output, force=force)


def _independent_source_validation(
    manifest: dict[str, Any],
    metadata_rows: list[dict[str, Any]],
    payloads: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Re-query the reference SDF for every assembled frame."""

    from toporetarget.data.storage import load_hoi_sequence
    from toporetarget.geometry.signed_distance.derived_proxy import (
        build_hybrid_signed_distance_backend,
    )
    from toporetarget.retarget.interaction_artifacts import load_interaction_graph
    from toporetarget.robots.registry import get_robot_registry

    canonical = Path(str(manifest["canonical"])).expanduser()
    graph_path = Path(str(manifest["graph"])).expanduser()
    surface_path = Path(str(manifest["collision_samples"])).expanduser()
    sequence = load_hoi_sequence(canonical)
    graph = load_interaction_graph(graph_path)
    object_id = str(graph.metadata["object_id"])
    obj = sequence.rigid_object(object_id)
    backend, geometry = build_hybrid_signed_distance_backend(
        obj.mesh.vertices_local,
        obj.mesh.faces,
        source_path=None,
        artifact_root=None,
    )
    expected_geometry = manifest.get("final_artifact_metadata", {}).get("geometry_policy") or {}
    if expected_geometry.get("cache_signature") not in {None, geometry.cache_signature}:
        raise CheckpointError("final independent geometry policy signature changed")
    robot_name = str(manifest["robot_name"])
    model = get_robot_registry(repo_root=Path(__file__).resolve().parents[3]).load(robot_name)
    surface = load_robot_surface_samples(surface_path)
    max_error = 0.0
    rows: list[dict[str, Any]] = []
    for metadata, payload in zip(metadata_rows, payloads, strict=True):
        global_frame = int(metadata["global_frame_index"])
        points = dynamic_collision_points_numpy(
            model,
            surface,
            np.asarray(payload["qpos"], dtype=np.float64),
            np.asarray(payload["base_pose_scene"], dtype=np.float64),
        )
        result = backend.query_scene(points, obj.pose_scene.pose_scene[global_frame])
        stored = np.asarray(payload["full_signed_distance"], dtype=np.float64)
        error = float(np.max(np.abs(result.signed_distance - stored), initial=0.0))
        max_error = max(max_error, error)
        rows.append(
            {
                "global_frame_index": global_frame,
                "sample_count": int(len(result.signed_distance)),
                "max_abs_signed_distance_error_m": error,
                "sign_valid": bool(np.all(result.sign_valid)),
                "near_original_boundary_count": int(
                    np.count_nonzero(result.near_original_boundary)
                    if result.near_original_boundary is not None
                    else 0
                ),
                "proxy_patch_count": int(
                    np.count_nonzero(result.proxy_closest_is_synthetic_patch)
                    if result.proxy_closest_is_synthetic_patch is not None
                    else 0
                ),
            }
        )
        if error > 1e-9 or not np.all(result.sign_valid):
            raise CheckpointError(
                f"final independent reference audit failed at frame {global_frame}: {error}"
            )
    return {
        "status": "pass",
        "backend": backend.describe(),
        "frame_count": len(rows),
        "max_abs_signed_distance_error_m": max_error,
        "frames": rows,
    }


def assemble_checkpoint_run(
    checkpoint_root: str | Path, output: str | Path, *, force: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Validate and assemble a complete checkpoint chain into one artifact."""

    store = CheckpointStore(Path(checkpoint_root))
    destination = store.assemble(output, force=force)
    return destination, store.status()


def _assemble_arrays(
    metadata: list[dict[str, Any]], rows: list[dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    def stack(name: str) -> np.ndarray:
        return np.stack([np.asarray(row[name]) for row in rows])

    def ragged(name: str, dtype: Any) -> tuple[np.ndarray, np.ndarray]:
        values = [np.asarray(row[name], dtype=dtype) for row in rows]
        offsets = np.zeros(len(values) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(value) for value in values])
        return np.concatenate(values), offsets

    query_ids, query_offsets = ragged("query_ids", np.int64)
    query_round, _ = ragged("query_active_round", np.int64)
    reasons, _ = ragged("query_inclusion_reason", "S96")
    slack, slack_offsets = ragged("slack", np.float64)
    signed, _ = ragged("signed_distance", np.float64)
    hard, _ = ragged("hard_residual", np.float64)
    soft, _ = ragged("soft_residual", np.float64)
    components = stack("objective_components")
    # Legacy checkpoints contain 12 components; expose zero-valued optional
    # quality terms when assembling them into the current artifact schema.
    if components.shape[1] == 12:
        components = np.concatenate(
            [components[:, :9], np.zeros((len(components), 6)), components[:, 9:]],
            axis=1,
        )
    frame_metadata = metadata

    def bool_array(name: str, default: bool = True) -> np.ndarray:
        return np.asarray([bool(item.get(name, default)) for item in frame_metadata], dtype=bool)

    def int_array(name: str, default: int = 0) -> np.ndarray:
        return np.asarray([int(item.get(name, default)) for item in frame_metadata], dtype=np.int64)

    def float_array(name: str, default: float = np.nan) -> np.ndarray:
        return np.asarray(
            [float(item.get(name, default)) for item in frame_metadata], dtype=np.float64
        )

    return {
        "timestamps": np.asarray([item["timestamp"] for item in frame_metadata], dtype=np.float64),
        "frame_indices": np.asarray(
            [item["global_frame_index"] for item in frame_metadata], dtype=np.int64
        ),
        "source_frame_indices": np.asarray(
            [
                item.get("source_global_frame_index", item["global_frame_index"])
                for item in frame_metadata
            ],
            dtype=np.int64,
        ),
        "qpos": stack("qpos"),
        "base_pose_scene": stack("base_pose_scene"),
        "base_corrections": stack("base_correction"),
        "robot_keypoints_base": stack("robot_keypoints_base"),
        "robot_keypoints_scene": stack("robot_keypoints_scene"),
        "robot_link_poses": stack("robot_link_poses"),
        "collision_points_scene": stack("collision_points_scene"),
        "joint_limit_margins": stack("joint_limit_margins"),
        "e_im": components[:, 0],
        "e_bone": components[:, 1],
        "e_temporal": components[:, 2],
        "e_base_pos": components[:, 3],
        "e_base_rot": components[:, 4],
        "e_slack": components[:, 5],
        "weighted_e_im": components[:, 6],
        "weighted_e_bone": components[:, 7],
        "total_objective": components[:, 8],
        "e_morph": components[:, 9],
        "weighted_e_morph": components[:, 10],
        "e_contact_pos": components[:, 11],
        "weighted_e_contact_pos": components[:, 12],
        "e_contact_dir": components[:, 13],
        "weighted_e_contact_dir": components[:, 14],
        "warm_e_im": components[:, 15],
        "warm_e_bone": components[:, 16],
        "warm_total_objective": components[:, 17],
        "query_ids_concat": query_ids,
        "query_offsets": query_offsets,
        "query_active_round_concat": query_round,
        "query_inclusion_reason_concat": reasons,
        "slack_concat": slack,
        "slack_offsets": slack_offsets,
        "signed_distance_concat": signed,
        "hard_residual_concat": hard,
        "soft_residual_concat": soft,
        "full_signed_distance": stack("full_signed_distance"),
        "full_closest_points": stack("full_closest_points"),
        "full_surface_normals": stack("full_surface_normals"),
        "full_hard_residual": stack("full_hard_residual"),
        "full_soft_violation_count": np.asarray(
            [int(row["full_soft_violation_count"]) for row in rows], dtype=np.int64
        ),
        "unqueried_soft_violation_count": np.asarray(
            [int(row["unqueried_soft_violation_count"]) for row in rows], dtype=np.int64
        ),
        "min_full_signed_distance": np.min(stack("full_signed_distance"), axis=1),
        "max_penetration": np.maximum(0.0, -np.min(stack("full_signed_distance"), axis=1)),
        "solver_success": bool_array("solver_success"),
        "valid_mask": bool_array("strict_accepted"),
        "solver_status": int_array("optimizer_status_code", -1),
        "iterations": int_array("iterations"),
        "function_evaluations": int_array("function_evaluations"),
        "jacobian_evaluations": int_array("jacobian_evaluations"),
        "solve_time_s": float_array("solve_time_s"),
        "active_set_rounds": int_array("active_set_rounds"),
        "active_set_converged": bool_array("active_set_converged", False),
        "optimizer_converged": bool_array("optimizer_converged", False),
        "optimizer_status_code": int_array("optimizer_status_code", -1),
        "optimizer_message": np.asarray(
            [item["optimizer_message"] for item in frame_metadata], dtype="S256"
        ),
        "optimizer_iterations": int_array("optimizer_iterations"),
        "optimizer_function_evaluations": int_array("optimizer_function_evaluations"),
        "optimizer_jacobian_evaluations": int_array("optimizer_jacobian_evaluations"),
        "qpos_bounds_pass": bool_array("qpos_bounds_pass", False),
        "slack_bounds_pass": bool_array("slack_bounds_pass", False),
        "active_constraints_feasible": bool_array("active_constraints_feasible", False),
        "full_surface_hard_audit_pass": bool_array("full_surface_hard_audit_pass", False),
        "full_surface_soft_audit_pass": bool_array("full_surface_soft_audit_pass", False),
        "all_values_finite": bool_array("all_values_finite", False),
        "stationarity_checked": bool_array("stationarity_checked", False),
        "stationarity_residual": float_array("stationarity_residual"),
        "accepted": bool_array("strict_accepted", False),
        "single_frame_feasible": bool_array("single_frame_feasible", False),
        "trajectory_continuous": bool_array("trajectory_continuous", False),
        "final_accepted": bool_array("final_accepted", False),
        "continuity_failure_reasons": np.asarray(
            [
                ",".join(str(value) for value in row.get("continuity_failure_reasons", []))
                for row in frame_metadata
            ],
            dtype="S512",
        ),
        "initialization_source": np.asarray(
            [str(row.get("initialization_source", "warm_reset")) for row in frame_metadata],
            dtype="S96",
        ),
        "retry_attempt": int_array("retry_attempt"),
        "retry_profile": np.asarray(
            [str(row.get("retry_profile", "none")) for row in frame_metadata], dtype="S96"
        ),
        "window_used": bool_array("window_used", False),
        "continuity_base_translation_m": np.asarray(
            [
                float(row.get("continuity_metrics", {}).get("delta_base_translation_m", np.nan))
                for row in frame_metadata
            ],
            dtype=np.float64,
        ),
        "continuity_base_rotation_rad": np.asarray(
            [
                float(row.get("continuity_metrics", {}).get("delta_base_rotation_rad", np.nan))
                for row in frame_metadata
            ],
            dtype=np.float64,
        ),
        "continuity_finger_inf_rad": np.asarray(
            [
                float(row.get("continuity_metrics", {}).get("delta_finger_inf_rad", np.nan))
                for row in frame_metadata
            ],
            dtype=np.float64,
        ),
        "continuity_excess_keypoint_m": np.asarray(
            [
                float(row.get("continuity_metrics", {}).get("excess_keypoint_max_m", np.nan))
                for row in frame_metadata
            ],
            dtype=np.float64,
        ),
        "q_clamp_count": int_array("q_clamp_count"),
        "acceptance_reason": np.asarray(
            [item["acceptance_reason"] for item in frame_metadata], dtype="S512"
        ),
        "initial_objective": float_array("initial_objective"),
        "final_objective": float_array("final_objective"),
        "final_objective_change": float_array("final_objective_change"),
        "final_step_norm": float_array("final_step_norm"),
    }


def time_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointStore",
    "assemble_checkpoint_run",
    "frame_checkpoint_payload",
]
