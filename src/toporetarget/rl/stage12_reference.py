"""Fail-closed bridge from an accepted Stage-12 final to Stage-16 inputs.

Stage-12 final-retarget artifacts are not ``RobotReferenceV2`` files: they
store the robot trajectory in scene coordinates while the canonical HOI cache
owns the object trajectory and mesh.  This module joins those two immutable
inputs without changing either source artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.contracts.reference import RobotReferenceV2
from toporetarget.data.storage import direct_zarr3_arrays, load_hoi_sequence
from toporetarget.geometry.se3 import invert_transform, relative_transform, transform_points
from toporetarget.robots import get_robot_registry

_FINAL_ARRAYS = (
    "qpos",
    "base_pose_scene",
    "robot_link_poses",
    "timestamps",
    "source_frame_indices",
    "final_accepted",
    "trajectory_continuous",
    "valid_mask",
)


class Stage12ReferenceError(ValueError):
    """Raised when a Stage-12 final is not eligible for Stage-16 use."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_metadata(path: Path) -> dict[str, Any]:
    group = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    raw = group.get("attributes", {}).get("metadata_json", "{}")
    metadata = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(metadata, dict):
        raise Stage12ReferenceError("final-retarget metadata must decode to a mapping")
    return metadata


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage12ReferenceError(message)


def build_robot_reference_from_stage12_final(
    *,
    final_arrays: dict[str, np.ndarray],
    object_pose_scene: np.ndarray,
    canonical_timestamps: np.ndarray,
    final_metadata: dict[str, Any],
    final_artifact: str,
    canonical_artifact: str,
    manifest_artifact: str,
    manifest_sha256: str,
) -> RobotReferenceV2:
    """Build and validate a base-frame reference from already-loaded artifacts."""

    required = set(_FINAL_ARRAYS)
    missing = sorted(required.difference(final_arrays))
    _require(not missing, f"Stage-12 final is missing required arrays: {missing}")
    qpos = np.asarray(final_arrays["qpos"], dtype=np.float64)
    base_pose = np.asarray(final_arrays["base_pose_scene"], dtype=np.float64)
    link_poses = np.asarray(final_arrays["robot_link_poses"], dtype=np.float64)
    timestamps = np.asarray(final_arrays["timestamps"], dtype=np.float64).reshape(-1)
    source_indices = np.asarray(final_arrays["source_frame_indices"], dtype=np.int64).reshape(-1)
    accepted = np.asarray(final_arrays["final_accepted"], dtype=bool).reshape(-1)
    continuous = np.asarray(final_arrays["trajectory_continuous"], dtype=bool).reshape(-1)
    valid = np.asarray(final_arrays["valid_mask"], dtype=bool).reshape(-1)
    frame_count = timestamps.size

    _require(frame_count >= 2, "Stage-12 final must have at least two frames")
    _require(qpos.ndim == 2 and qpos.shape[0] == frame_count, "qpos must have shape [T,D]")
    _require(base_pose.shape == (frame_count, 4, 4), "base_pose_scene must have shape [T,4,4]")
    _require(
        link_poses.ndim == 4
        and link_poses.shape[:2] == (frame_count, link_poses.shape[1])
        and link_poses.shape[2:] == (4, 4),
        "robot_link_poses must have shape [T,L,4,4]",
    )
    _require(
        all(
            values.shape == (frame_count,)
            for values in (source_indices, accepted, continuous, valid)
        ),
        "Stage-12 final masks and source indices must align with timestamps",
    )
    _require(bool(accepted.all()), "Stage-12 final has unaccepted frames")
    _require(bool(continuous.all()), "Stage-12 final failed trajectory continuity")
    _require(bool(valid.all()), "Stage-12 final has invalid frames")
    _require(
        bool(np.all(np.diff(timestamps) > 0.0)), "Stage-12 final timestamps are not increasing"
    )
    _require(
        source_indices.min() >= 0 and source_indices.max() < object_pose_scene.shape[0],
        "source frame indices are outside canonical object poses",
    )
    _require(
        object_pose_scene.shape == (canonical_timestamps.size, 4, 4),
        "canonical object poses must align with canonical timestamps",
    )
    _require(
        np.allclose(timestamps, canonical_timestamps[source_indices], atol=1e-9, rtol=0.0),
        "Stage-12 final timestamps do not align with source canonical timestamps",
    )
    link_names = tuple(str(value) for value in final_metadata.get("robot_link_names", ()))
    _require(
        len(link_names) == link_poses.shape[1],
        "robot_link_names do not align with robot_link_poses",
    )
    _require(
        qpos.shape[1] == int(final_metadata.get("robot_dof_count", -1)),
        "qpos DoF does not match Stage-12 manifest",
    )
    robot_hash = str(final_metadata.get("robot_spec_hash", ""))
    _require(bool(robot_hash), "Stage-12 manifest has no robot_spec_hash")
    robot_name = str(final_metadata.get("robot_name", ""))
    _require(robot_name == "wuji_hand2_beta1_rh", "Stage-12 final is not the supported Wuji RH")
    joint_order = get_robot_registry().get_spec(robot_name).dof_order
    _require(
        len(joint_order) == qpos.shape[1],
        "registered Wuji joint order does not match Stage-12 qpos dimensions",
    )

    base_inverse = invert_transform(base_pose)
    reference = RobotReferenceV2(
        qpos_reference=qpos,
        base_pose=base_pose,
        object_pose_base=relative_transform(base_pose, object_pose_scene[source_indices]),
        tracked_link_positions=transform_points(base_inverse, link_poses[..., :3, 3]),
        timestamps=timestamps,
        fps=float(1.0 / np.median(np.diff(timestamps))),
        joint_order=joint_order,
        robot_hash=robot_hash,
        dataset_provenance={
            "kind": "accepted_stage12_hocap_final",
            "source_final_artifact": final_artifact,
            "source_canonical_artifact": canonical_artifact,
            "source_checkpoint_manifest": manifest_artifact,
            "source_checkpoint_manifest_sha256": manifest_sha256,
            "source_sequence": final_metadata.get("source_sequence_id"),
            "acceptance_policy_id": final_metadata.get("acceptance_policy_id"),
            "continuity_acceptance": True,
        },
        frame_indices=source_indices,
        tracked_link_names=link_names,
        metadata={
            "source_schema_version": "toporetarget.final_retarget.v3",
            "source_stage12_final_accepted": True,
            "source_stage12_trajectory_continuous": True,
            "joint_order_source": "registered_wuji_hand2_beta1_rh_qpos_profile",
            "source_stage12_final_metadata": final_metadata,
        },
    )
    reference.validate()
    return reference


def materialize_accepted_stage12_reference(
    *,
    final_trajectory: str | Path,
    canonical: str | Path,
    checkpoint_manifest: str | Path,
) -> tuple[RobotReferenceV2, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load immutable Stage-12 artifacts and return a reference plus object mesh."""

    final_path = Path(final_trajectory).resolve()
    canonical_path = Path(canonical).resolve()
    manifest_path = Path(checkpoint_manifest).resolve()
    _require(
        (final_path / "zarr.json").is_file(), f"final trajectory is not a Zarr group: {final_path}"
    )
    _require(canonical_path.is_dir(), f"canonical cache is unavailable: {canonical_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = dict(manifest.get("final_artifact_metadata", {}))
    _require(
        metadata.get("acceptance_policy_id")
        == "strict_optimizer_converged_audits_and_continuity_v1",
        "Stage-12 final does not use the accepted strict continuity policy",
    )
    _require(
        bool(metadata.get("continuity_acceptance")),
        "Stage-12 checkpoint lacks continuity acceptance",
    )
    arrays = direct_zarr3_arrays(final_path, _FINAL_ARRAYS, array_prefix="")
    sequence = load_hoi_sequence(canonical_path)
    rigid = sequence.primary_rigid_object()
    if rigid.valid is None or rigid.pose_scene.valid is None:
        raise Stage12ReferenceError("canonical object has no validity mask")
    _require(
        bool(rigid.valid.all()) and bool(rigid.pose_scene.valid.all()),
        "canonical object has invalid frames",
    )
    reference = build_robot_reference_from_stage12_final(
        final_arrays=arrays,
        object_pose_scene=rigid.pose_scene.pose_scene,
        canonical_timestamps=sequence.timestamps,
        final_metadata=metadata,
        final_artifact=str(final_path),
        canonical_artifact=str(canonical_path),
        manifest_artifact=str(manifest_path),
        manifest_sha256=_sha256(manifest_path),
    )
    mesh_metadata = {
        "object_id": rigid.object_id,
        "mesh_id": rigid.mesh.mesh_id,
        "mesh_hash": rigid.mesh.mesh_hash,
        "units": rigid.mesh.units,
        "canonical_artifact": str(canonical_path),
        "canonical_sequence": sequence.metadata.sequence_id,
        "canonical_provenance": asdict(sequence.metadata.provenance),
    }
    return reference, rigid.mesh.vertices_local, rigid.mesh.faces, mesh_metadata


def write_obj_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    """Write a derived local OBJ for MuJoCo; the source dataset remains untouched."""

    destination = Path(path)
    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    _require(values.ndim == 2 and values.shape[1] == 3, "mesh vertices must have shape [V,3]")
    _require(triangles.ndim == 2 and triangles.shape[1] == 3, "mesh faces must have shape [F,3]")
    _require(bool(np.isfinite(values).all()), "mesh vertices contain NaN or Inf")
    _require(
        triangles.min() >= 0 and triangles.max() < values.shape[0], "mesh faces are out of bounds"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for vertex in values:
            handle.write(f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n")
        for face in triangles:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    return destination


__all__ = [
    "Stage12ReferenceError",
    "build_robot_reference_from_stage12_final",
    "materialize_accepted_stage12_reference",
    "write_obj_mesh",
]
