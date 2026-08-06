"""Non-destructive Stage 16-D trajectory and provenance export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_physics_consistent_trajectory(
    *,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    manifest: dict[str, Any],
    quality: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, str]:
    """Export PhysicsConsistentRetargetedTrajectoryV1 without touching source data."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Stage16D refuses to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"actions", "object_pose", "finger_q", "wrist_pose"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"trajectory export misses arrays: {sorted(missing)}")
    frame_counts = {np.asarray(value).shape[0] for value in arrays.values()}
    if frame_counts != {321}:
        raise ValueError("PhysicsConsistentRetargetedTrajectoryV1 requires 321 samples")
    actions = np.asarray(arrays["actions"])
    if (
        actions.shape != (321, 26)
        or not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > 1.0
    ):
        raise ValueError("trajectory actions violate the frozen [321,26] contract")
    if any(not np.isfinite(np.asarray(value)).all() for value in arrays.values()):
        raise ValueError("trajectory arrays must be finite")
    trajectory_path = output_dir / "trajectory.npz"
    np.savez_compressed(trajectory_path, **arrays)  # type: ignore[arg-type]
    action_path = output_dir / "action_trace.npy"
    np.save(action_path, actions)
    enriched = {
        "schema_version": "PhysicsConsistentRetargetedTrajectoryV1",
        "protocol": "physics_consistent_retargeting_v1",
        "object_trajectory_role": "free_physx_rollout_output_not_decision_variable",
        "source_overwritten": False,
        "trajectory_sha256": sha256_file(trajectory_path),
        "action_trace_sha256": sha256_file(action_path),
        **manifest,
    }
    manifest_path = output_dir / "manifest.json"
    quality_path = output_dir / "quality.json"
    manifest_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    try:
        import zarr

        rollout_path = output_dir / "rollout.zarr"
        group = zarr.open_group(str(rollout_path), mode="w")
        for name, value in arrays.items():
            array = np.asarray(value)
            chunks = (min(64, array.shape[0]), *array.shape[1:])
            if hasattr(group, "create_array"):
                group.create_array(name, data=array, chunks=chunks)
            else:  # zarr 2.x compatibility
                group.create_dataset(name, data=array, shape=array.shape, chunks=chunks)
        group.attrs.update(enriched)
    except ImportError:
        rollout_path = output_dir / "rollout.zarr.NOT_WRITTEN.json"
        rollout_path.write_text(
            json.dumps(
                {"status": "ZARR_DEPENDENCY_UNAVAILABLE", "trajectory": str(trajectory_path)}
            )
            + "\n"
        )
    return {
        "trajectory": str(trajectory_path),
        "rollout": str(rollout_path),
        "manifest": str(manifest_path),
        "quality": str(quality_path),
        "action_trace": str(action_path),
    }


__all__ = ["export_physics_consistent_trajectory", "sha256_file"]
