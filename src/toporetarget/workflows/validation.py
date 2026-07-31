"""Cross-stage and semantic sanity validation for one bounded workflow run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .cache import path_hash
from .schema import stable_hash, write_json

FINAL_CONTACT_ASSUMPTION = "A_WORKFLOW_FINAL_CONTACT_SANITY_001"


def validate_manual_acceptance(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"manual acceptance is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manual acceptance must be a JSON object")
    required = {
        "schema_version",
        "status",
        "reviewer",
        "reviewed_frames",
        "current_window_interpretation",
        "contact_rich_clip_validated",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"manual acceptance is missing fields: {', '.join(missing)}")
    if payload["status"] != "pass":
        raise ValueError("manual acceptance status must be pass")
    if payload["reviewer"] != "human":
        raise ValueError("manual acceptance reviewer must be human")
    if not {0, 29, 59}.issubset(set(int(item) for item in payload["reviewed_frames"])):
        raise ValueError("manual acceptance must include frames 0, 29, and 59")
    if payload["current_window_interpretation"] not in {"pre_contact", "contact_rich", "invalid"}:
        raise ValueError("manual acceptance has an invalid current_window_interpretation")
    if payload["current_window_interpretation"] == "invalid":
        raise ValueError("Stage 9 manual acceptance interpretation is invalid")
    return payload


def environment_snapshot(repo_root: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "torch", "zarr", "trimesh", "matplotlib", "PIL"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            packages[name] = "unavailable"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": commit,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _nearest_object_distance(points: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(object_points) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(object_points).query(points, k=1)[0], dtype=np.float64)
    except ImportError:
        values = np.full(len(points), np.inf, dtype=np.float64)
        for start in range(0, len(points), 128):
            chunk = points[start : start + 128]
            delta = chunk[:, None, :] - object_points[None, :, :]
            values[start : start + len(chunk)] = np.sqrt(
                np.min(np.sum(delta * delta, axis=-1), axis=1)
            )
        return values


def build_semantic_sanity_report(
    *,
    canonical: str | Path,
    final: str | Path,
    robot: str,
    collision_samples: str | Path,
    selected_window: dict[str, Any],
    final_contact_sanity_max_distance_m: float,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate semantic identity and report coverage limitations honestly."""

    from toporetarget.data.storage import load_hoi_sequence
    from toporetarget.retarget.final_refinement import (
        load_final_trajectory,
        load_robot_surface_samples,
    )
    from toporetarget.robots.artimano import load_artimano_model

    sequence = load_hoi_sequence(canonical)
    artifact = load_final_trajectory(final)
    model = load_artimano_model("right" if robot.endswith("rh") else "left")
    surface = load_robot_surface_samples(collision_samples)
    hand = sequence.hand(str(artifact.metadata["source_hand_id"]))
    object_track = sequence.rigid_object(str(artifact.metadata["object_id"]))
    object_scene_points: list[np.ndarray] = []
    for frame in range(sequence.num_frames):
        pose = object_track.pose_scene.pose_scene[frame]
        object_scene_points.append(
            np.asarray(object_track.mesh.vertices_local) @ pose[:3, :3].T + pose[:3, 3]
        )
    start = int(selected_window["start_frame"])
    local_contact_frames = [
        int(frame - start)
        for frame in range(int(selected_window["start_frame"]), int(selected_window["end_frame"]))
        if frame in set(selected_window.get("contact_frames", []))
    ]
    if not local_contact_frames:
        ratio = float(selected_window.get("contact_frame_ratio", 0.0))
        count = int(round(ratio * (int(selected_window["end_frame"]) - start)))
        local_contact_frames = list(range(min(count, artifact.frame_count)))
    local_contact_frames = [
        frame for frame in local_contact_frames if 0 <= frame < artifact.frame_count
    ]
    final_min_distance = np.asarray(artifact.arrays["min_full_signed_distance"], dtype=np.float64)
    final_penetration = np.asarray(artifact.arrays["max_penetration"], dtype=np.float64)
    contact_distances = (
        final_min_distance[local_contact_frames] if local_contact_frames else np.asarray([])
    )
    keypoint_distances: list[float] = []
    for frame in local_contact_frames:
        points = np.asarray(artifact.arrays["robot_keypoints_scene"][frame])
        keypoint_distances.extend(
            _nearest_object_distance(points, object_scene_points[frame]).tolist()
        )
    qpos = np.asarray(artifact.arrays["qpos"], dtype=np.float64)
    lower = np.asarray(model.joint_lower, dtype=np.float64)
    upper = np.asarray(model.joint_upper, dtype=np.float64)
    qpos_bounds = bool(
        np.all(qpos >= lower[None, :] - 1e-10) and np.all(qpos <= upper[None, :] + 1e-10)
    )
    base = np.asarray(artifact.arrays["base_pose_scene"], dtype=np.float64)
    base_continuity = (
        float(np.max(np.linalg.norm(np.diff(base[:, :3, 3], axis=0), axis=1)))
        if len(base) > 1
        else 0.0
    )
    contact_max_distance = float(np.max(contact_distances)) if len(contact_distances) else None
    contact_min_distance = float(np.min(contact_distances)) if len(contact_distances) else None
    warning = (
        contact_max_distance is not None
        and contact_max_distance > final_contact_sanity_max_distance_m
    )
    report = {
        "schema_version": "toporetarget.semantic_trajectory_sanity.v1",
        "status": "semantic_warning" if warning else "pass",
        "assumptions": [FINAL_CONTACT_ASSUMPTION],
        "canonical": str(canonical),
        "final_artifact": str(final),
        "robot": robot,
        "source": {
            "sequence": sequence.metadata.sequence_id,
            "hand": hand.side,
            "contact_frame_ratio": selected_window.get("contact_frame_ratio"),
            "contact_frame_count": selected_window.get("contact_frame_count"),
            "source_contact_median_distance_m": selected_window.get(
                "source_contact_median_distance_m"
            ),
            "source_contact_min_distance_m": selected_window.get("source_contact_min_distance_m"),
            "object_coordinate_sanity": True,
        },
        "contact_frames": {
            "local_indices": local_contact_frames,
            "count": len(local_contact_frames),
            "ratio": float(len(local_contact_frames) / artifact.frame_count)
            if artifact.frame_count
            else 0.0,
        },
        "final": {
            "collision_surface_min_distance_m": float(np.min(final_min_distance)),
            "contact_frame_min_distance_m": contact_min_distance,
            "contact_frame_max_distance_m": contact_max_distance,
            "contact_frame_median_distance_m": float(np.median(contact_distances))
            if len(contact_distances)
            else None,
            "max_penetration_m": float(np.max(final_penetration))
            if len(final_penetration)
            else 0.0,
            "contact_frame_max_penetration_m": float(
                np.max(final_penetration[local_contact_frames])
            )
            if local_contact_frames
            else None,
            "robot_keypoint_object_min_distance_m": float(np.min(keypoint_distances))
            if keypoint_distances
            else None,
            "robot_keypoint_object_median_distance_m": float(np.median(keypoint_distances))
            if keypoint_distances
            else None,
            "qpos_bounds_pass": qpos_bounds,
            "base_continuity_max_translation_m": base_continuity,
            "active_set_converged": bool(np.all(artifact.arrays["active_set_converged"])),
            "solver_success": bool(np.all(artifact.arrays["solver_success"])),
            "e_im_max": float(np.max(artifact.arrays["e_im"])),
            "e_bone_max": float(np.max(artifact.arrays["e_bone"])),
            "max_slack_m": float(np.max(artifact.arrays["e_slack"])),
        },
        "thresholds": {
            "final_contact_sanity_max_distance_m": final_contact_sanity_max_distance_m,
            "final_contact_gate_pass": not warning,
        },
        "coverage": {
            "missing_tip_collision": model.describe()
            .get("geometry", {})
            .get("missing_collision_links", []),
            "visual_fallback": bool(surface.profile.visual_fallback),
            "collision_surface_profile": surface.profile.__dict__,
        },
        "warnings": [
            "Final collision-surface distance exceeds the engineering contact sanity threshold."
            if warning
            else ""
        ],
    }
    report["warnings"] = [item for item in report["warnings"] if item]
    report["report_hash"] = stable_hash(report)
    if report_path is not None:
        write_json(report, report_path)
    return report


def cross_stage_identity_report(
    *,
    canonical: str | Path,
    warm_start: str | Path,
    graph: str | Path,
    final: str | Path,
    object_samples: str | Path,
    robot: str,
) -> dict[str, Any]:
    """Check identity fields and artifact provenance without truncating frames."""

    from toporetarget.data.storage import load_hoi_sequence
    from toporetarget.geometry.surface_artifacts import load_surface_artifact
    from toporetarget.retarget.artifacts import load_warm_start
    from toporetarget.retarget.final_refinement import load_final_trajectory
    from toporetarget.retarget.interaction_artifacts import load_interaction_graph

    sequence = load_hoi_sequence(canonical)
    warm = load_warm_start(warm_start)
    graph_value = load_interaction_graph(graph)
    final_value = load_final_trajectory(final)
    samples = load_surface_artifact(object_samples)
    source_provenance = sequence.metadata.provenance
    canonical_source_hash = str(source_provenance.source_hash)
    canonical_object_mesh_hash = str(
        source_provenance.conversion_options.get("object_mesh_hash", "")
    )
    primary_object = sequence.primary_rigid_object()
    canonical_hand = sequence.hands[0]
    canonical_timestamps = np.asarray(sequence.metadata.timestamps, dtype=np.float64)
    warm_timestamps = np.asarray(warm.arrays["timestamps"], dtype=np.float64)
    graph_timestamps = np.asarray(graph_value.timestamps, dtype=np.float64)
    final_timestamps = np.asarray(final_value.arrays["timestamps"], dtype=np.float64)
    timestamp_equal = bool(
        np.allclose(canonical_timestamps, warm_timestamps, rtol=0.0, atol=1e-12)
        and np.allclose(canonical_timestamps, graph_timestamps, rtol=0.0, atol=1e-12)
        and np.allclose(canonical_timestamps, final_timestamps, rtol=0.0, atol=1e-12)
    )
    final_frame_indices = np.asarray(
        final_value.arrays.get("frame_indices", np.arange(final_value.frame_count)), dtype=np.int64
    )
    graph_frame_indices = np.asarray(graph_value.frame_indices, dtype=np.int64)
    expected_local_indices = np.arange(sequence.num_frames, dtype=np.int64)
    frame_indices_equal = bool(
        np.array_equal(graph_frame_indices, expected_local_indices)
        and np.array_equal(final_frame_indices, expected_local_indices)
    )
    hand_identity = {
        "canonical_side": canonical_hand.side,
        "warm_start_side": warm.metadata.get("source_side"),
        "graph_side": graph_value.metadata.get("source_hand_side"),
        "final_side": final_value.metadata.get("source_hand_side"),
    }
    hand_identity_pass = len(set(hand_identity.values())) == 1
    object_identity = {
        "canonical_object_id": primary_object.object_id,
        "final_object_id": final_value.metadata.get("object_id"),
        "sample_source_object_id": samples.source_provenance.get(
            "object_id", samples.source_provenance.get("object_name")
        ),
    }
    object_identity_pass = len(set(object_identity.values())) == 1
    object_mesh_identity = {
        "canonical_object_mesh_hash": canonical_object_mesh_hash,
        "sample_object_mesh_hash": str(
            samples.source_provenance.get("object_mesh_hash", getattr(samples, "mesh_hash", ""))
        ),
        "sample_mesh_array_hash": str(getattr(samples, "mesh_array_hash", "")),
        "graph_object_mesh_hash": str(graph_value.metadata.get("object_mesh_hash", "")),
        "final_object_mesh_hash": str(final_value.metadata.get("object_mesh_hash", "")),
    }
    # Stage 5 preserves the source mesh hash; Stage 6/8/9 may carry the
    # canonicalized in-memory mesh-array hash.  Validate both links explicitly.
    object_mesh_identity_pass = bool(
        object_mesh_identity["canonical_object_mesh_hash"]
        == object_mesh_identity["sample_object_mesh_hash"]
        and object_mesh_identity["graph_object_mesh_hash"]
        == object_mesh_identity["final_object_mesh_hash"]
    )
    robot_identity = {
        "requested_robot": robot,
        "warm_start_robot": warm.metadata.get("robot_name"),
        "final_robot": final_value.metadata.get("robot_name"),
    }
    robot_identity_pass = len(set(robot_identity.values())) == 1
    canonical_artifact_hash = path_hash(canonical)
    source_cache_identity = {
        "canonical_artifact_hash": canonical_artifact_hash,
        "warm_start_source_cache_hash": str(warm.metadata.get("source_cache_hash", "")),
        "graph_source_cache_hash": str(graph_value.metadata.get("source_cache_hash", "")),
        "final_source_cache_hash": str(final_value.metadata.get("source_cache_hash", "")),
    }
    known_source_cache_hashes = {
        value for value in source_cache_identity.values() if value not in {"", "None"}
    }
    source_cache_identity_pass = len(known_source_cache_hashes) <= 1
    checks = {
        "canonical_frame_count": sequence.num_frames,
        "warm_start_frame_count": warm.frame_count,
        "graph_frame_count": graph_value.frame_count,
        "final_frame_count": final_value.frame_count,
        "frame_counts_equal": len(
            {
                sequence.num_frames,
                warm.frame_count,
                graph_value.frame_count,
                final_value.frame_count,
            }
        )
        == 1,
        "timestamps_equal": timestamp_equal,
        "frame_indices_equal": frame_indices_equal,
        "hand_identity": hand_identity,
        "hand_identity_pass": hand_identity_pass,
        "object_identity": object_identity,
        "object_identity_pass": object_identity_pass,
        "object_mesh_identity": object_mesh_identity,
        "object_mesh_identity_pass": object_mesh_identity_pass,
        "robot_identity": robot_identity,
        "robot_identity_pass": robot_identity_pass,
        "source_cache_identity": source_cache_identity,
        "source_cache_identity_pass": source_cache_identity_pass,
        "native_fps": sequence.metadata.native_fps,
        "hand": str(final_value.metadata.get("source_side", "")),
        "robot": robot,
        "object_id": str(final_value.metadata.get("object_id", "")),
        "source_canonical_hash": path_hash(canonical),
        "warm_start_hash": path_hash(warm_start),
        "graph_hash": path_hash(graph),
        "final_hash": path_hash(final),
        "object_samples_hash": path_hash(object_samples),
        "source_hash": canonical_source_hash,
    }
    checks["identity_pass"] = bool(
        checks["frame_counts_equal"]
        and checks["timestamps_equal"]
        and checks["frame_indices_equal"]
        and checks["hand_identity_pass"]
        and checks["object_identity_pass"]
        and checks["object_mesh_identity_pass"]
        and checks["robot_identity_pass"]
        and checks["source_cache_identity_pass"]
    )
    checks["status"] = "pass" if checks["identity_pass"] else "fail"
    return checks


__all__ = [
    "FINAL_CONTACT_ASSUMPTION",
    "build_semantic_sanity_report",
    "cross_stage_identity_report",
    "environment_snapshot",
    "validate_manual_acceptance",
]
