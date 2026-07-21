"""Read-only robot reference export for downstream control tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import REFERENCE_SCHEMA_VERSION, stable_hash, write_json


def _reference_payload(run: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from toporetarget.data.storage import load_hoi_sequence
    from toporetarget.retarget.final_refinement import load_final_trajectory

    final_path = run["artifacts"]["final"]["path"]
    canonical_path = run["artifacts"]["canonical"]["path"]
    final = load_final_trajectory(final_path)
    sequence = load_hoi_sequence(canonical_path)
    object_id = str(final.metadata["object_id"])
    object_track = sequence.rigid_object(object_id)
    frame_indices = np.asarray(final.arrays["frame_indices"], dtype=np.int64)
    arrays = {
        "timestamps": np.asarray(final.arrays["timestamps"], dtype=np.float64),
        "frame_indices": frame_indices,
        "qpos": np.asarray(final.arrays["qpos"], dtype=np.float64),
        "base_pose_scene": np.asarray(final.arrays["base_pose_scene"], dtype=np.float64),
        "robot_keypoints_scene": np.asarray(
            final.arrays["robot_keypoints_scene"], dtype=np.float64
        ),
        "robot_link_poses": np.asarray(final.arrays["robot_link_poses"], dtype=np.float64),
        "object_pose_scene": np.asarray(
            object_track.pose_scene.pose_scene[frame_indices], dtype=np.float64
        ),
    }
    metadata = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "robot": run["robot"],
        "side": run["hand"],
        "native_fps": run["native_fps"],
        "source_sequence": run["source_sequence"],
        "subject": run.get("subject"),
        "object_id": run.get("object_id", object_id),
        "action": run.get("action"),
        "source_hash": run.get("source_hash"),
        "final_artifact_path": final_path,
        "final_artifact_hash": run["artifacts"]["final"].get("hash"),
        "provenance": {
            "workflow_run_id": run["run_id"],
            "canonical": canonical_path,
            "final": final_path,
            "no_source_copy": True,
            "no_solver_run_during_export": True,
        },
    }
    metadata["content_hash"] = stable_hash(
        {"metadata": metadata, "arrays": {name: value.tolist() for name, value in arrays.items()}}
    )
    return metadata, arrays


def export_reference(
    run: dict[str, Any],
    *,
    output: str | Path,
    format: str = "zarr",
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata, arrays = _reference_payload(run)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"reference output exists; choose a new path: {destination}")
    if format == "npz":
        save_npz: Any = np.savez_compressed
        save_npz(destination, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    elif format == "zarr":
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError("Zarr export requires the cache extra") from exc
        group = zarr.open_group(str(destination), mode="w")
        group.attrs.update(metadata)
        for name, value in arrays.items():
            try:
                group.create_array(name, data=value, overwrite=True)
            except AttributeError:
                legacy_group: Any = group
                legacy_group.create_dataset(name, data=value, overwrite=True)
    else:
        raise ValueError("format must be zarr or npz")
    if metadata_path is not None:
        write_json(metadata, metadata_path)
    return {
        "status": "pass",
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "output": str(destination),
        "format": format,
        "metadata": metadata,
    }


__all__ = ["export_reference"]
