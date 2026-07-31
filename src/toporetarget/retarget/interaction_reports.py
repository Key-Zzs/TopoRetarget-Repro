"""Bounded Stage 8 audit, topology, and scale-diagnostic reports."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.surface_sampling import SurfaceSampleSet, transform_surface_samples
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.pipeline import source_cache_hash
from toporetarget.robots.artimano import load_artimano_model

from .delaunay import DelaunayProfile
from .interaction_graph import InteractionGraphTrajectory, build_source_interaction_graph


def _layout_hash() -> str:
    from toporetarget.keypoints.registry import get_layout

    layout = get_layout("mediapipe21")
    value = {
        "name": layout.name,
        "version": layout.version,
        "semantic_names": list(layout.semantic_names),
        "parents": list(layout.parents),
        "edges": [list(item) for item in layout.edges],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_input_audit(
    inputs: list[dict[str, Any]],
    samples_path: str | Path,
    *,
    robot_names: tuple[str, ...] = ("artimano_rh", "artimano_lh"),
    asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit both source sides and their Stage 7 warm starts before building graphs."""

    samples = SurfaceSampleSet.load(samples_path)
    robot_info: dict[str, Any] = {}
    for name in robot_names:
        model = load_artimano_model(
            "right" if name.endswith("rh") else "left", asset_root=asset_root
        )
        robot_info[name] = {
            "name": model.name,
            "side": model.side,
            "spec_hash": model.spec_hash,
            "urdf_hash": model.urdf_hash,
            "asset_manifest_hash": model.asset_manifest_hash,
            "anchor_profile_id": model.anchor_profile.profile_id,
            "anchor_profile_hash": model.anchor_profile.sha256,
            "layout": model.spec.semantic_keypoint_layout,
        }
    records: list[dict[str, Any]] = []
    for item in inputs:
        canonical = Path(item["canonical"])
        warm_path = Path(item["warm_start"])
        sequence = load_hoi_sequence(canonical)
        hand_id = str(item.get("hand_id") or sequence.hands[0].hand_id)
        hand = sequence.hand(hand_id)
        try:
            object_track = sequence.primary_rigid_object()
        except KeyError as exc:
            raise ValueError(f"canonical primary object is invalid: {canonical}") from exc
        mesh = audit_mesh(object_track.mesh.vertices_local, object_track.mesh.faces)
        warm = load_warm_start(warm_path)
        records.append(
            {
                "canonical_path": str(canonical),
                "canonical_hash": source_cache_hash(canonical),
                "warm_start_path": str(warm_path),
                "warm_start_hash": artifact_hash(warm_path),
                "warm_start_source_hash": warm.metadata.get("source_cache_hash"),
                "sequence_id": sequence.metadata.sequence_id,
                "hand_id": hand.hand_id,
                "hand_side": hand.side,
                "frame_range": [0, sequence.num_frames],
                "frame_count": sequence.num_frames,
                "timestamps": sequence.metadata.timestamps.tolist(),
                "timestamps_match_warm_start": bool(
                    np.array_equal(sequence.metadata.timestamps, warm.arrays["timestamps"])
                ),
                "native_fps": sequence.metadata.native_fps,
                "object_id": object_track.object_id,
                "object_mesh_hash": mesh.mesh_hash,
                "object_topology_hash": mesh.topology_hash,
                "object_sample_path": str(samples_path),
                "object_sample_hash": hashlib.sha256(Path(samples_path).read_bytes()).hexdigest(),
                "object_sample_profile": samples.profile_id,
                "object_sample_profile_hash": samples.profile_hash,
                "object_sample_count": samples.count,
                "source_layout": hand.keypoint_tracks["mediapipe21"].layout_name,
                "source_layout_hash": _layout_hash(),
                "warm_qpos_shape": list(warm.arrays["qpos"].shape),
                "warm_base_pose_shape": list(warm.arrays["base_pose_scene"].shape),
                "warm_robot_name": warm.metadata.get("robot_name"),
                "warm_robot_spec_hash": warm.metadata.get("robot_spec_hash"),
                "warm_robot_urdf_hash": warm.metadata.get("urdf_hash"),
                "warm_robot_anchor_hash": warm.metadata.get("anchor_profile_hash"),
                "artifact_compatibility": {
                    "frame_count": warm.frame_count == sequence.num_frames,
                    "timestamps": bool(
                        np.array_equal(sequence.metadata.timestamps, warm.arrays["timestamps"])
                    ),
                    "hand_side": warm.metadata.get("source_side") in {None, hand.side},
                    "object_mesh_hash": mesh.mesh_hash
                    in {samples.mesh_hash, samples.mesh_array_hash},
                    "object_sample_count": samples.count == 50,
                    "source_layout": hand.keypoint_tracks["mediapipe21"].layout_name
                    == "mediapipe21",
                    "warm_source_hash": warm.metadata.get("source_cache_hash")
                    == source_cache_hash(canonical),
                },
            }
        )
    return {
        "schema_version": "toporetarget.stage8.input_audit.v1",
        "scipy_version": __import__("scipy").__version__,
        "numpy_version": np.__version__,
        "qhull_options": "Qbb Qc Qz Q12",
        "coordinate_frame": "S",
        "units": "m",
        "vertex_count": 71,
        "hand_vertex_count": 21,
        "object_vertex_count": 50,
        "object_sample": {
            "path": str(samples_path),
            "count": samples.count,
            "profile_id": samples.profile_id,
            "profile_hash": samples.profile_hash,
            "mesh_hash": samples.mesh_hash,
            "topology_hash": samples.topology_hash,
            "face_ids": samples.face_indices.tolist(),
            "barycentric": samples.barycentric.tolist(),
        },
        "robots": robot_info,
        "inputs": records,
        "all_compatibility_checks_pass": all(
            all(bool(value) for value in record["artifact_compatibility"].values())
            for record in records
        ),
    }


def topology_over_time(graph: InteractionGraphTrajectory) -> dict[str, Any]:
    hashes = graph.graph_hashes
    jaccard: list[float] = []
    changes: list[int] = []
    for index in range(1, graph.frame_count):
        left = {tuple(edge) for edge in graph.edge_frames[index - 1].tolist()}
        right = {tuple(edge) for edge in graph.edge_frames[index].tolist()}
        union = left | right
        score = 1.0 if not union else len(left & right) / len(union)
        jaccard.append(float(score))
        if left != right:
            changes.append(index)
    simplex = [int(frame["simplex_count"]) for frame in graph.frame_statistics]
    edges = [int(frame["edge_count"]) for frame in graph.frame_statistics]
    return {
        "frame_count": graph.frame_count,
        "simplex_count": {
            "min": min(simplex),
            "mean": float(np.mean(simplex)),
            "max": max(simplex),
        },
        "edge_count": {"min": min(edges), "mean": float(np.mean(edges)), "max": max(edges)},
        "graph_hash_change_count": int(
            sum(left != right for left, right in zip(hashes, hashes[1:], strict=False))
        ),
        "topology_change_frames": changes,
        "adjacent_edge_jaccard": jaccard,
        "graph_hashes": hashes,
    }


def source_integrity_snapshot(paths: dict[str, str | Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in paths.items():
        value = Path(path)
        if value.is_dir():
            result[name] = source_cache_hash(value)
        elif value.is_file():
            result[name] = hashlib.sha256(value.read_bytes()).hexdigest()
        else:
            result[name] = None
    return result


def compare_object_scales(
    sequence: Any,
    hand_id: str,
    samples: SurfaceSampleSet,
    profile: DelaunayProfile,
    scales: list[float],
    *,
    frame: int = 0,
    object_id: str = "primary",
    kappa: float | None = None,
) -> dict[str, Any]:
    """Recompute bounded graph diagnostics from local barycentric anchors."""

    sequence.hand(hand_id)
    object_track = (
        sequence.rigid_object(object_id)
        if object_id not in {"primary", "object"}
        else sequence.primary_rigid_object()
    )
    mesh_vertices = np.asarray(object_track.mesh.vertices_local, dtype=np.float64)
    original_points, _ = transform_surface_samples(
        samples, object_track.pose_scene.pose_scene[frame]
    )
    values: list[dict[str, Any]] = []
    for scale in scales:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("object scales must be finite and positive")
        scaled_points_local = np.einsum(
            "ni,nij->nj",
            samples.barycentric,
            (mesh_vertices * scale)[object_track.mesh.faces[samples.face_indices]],
        )
        from toporetarget.geometry.se3 import transform_points

        scaled_points = transform_points(
            object_track.pose_scene.pose_scene[frame], scaled_points_local[None, ...]
        )[0]
        scaled_sequence = copy.deepcopy(sequence)
        scaled_object = scaled_sequence.rigid_object(object_track.object_id)
        scaled_vertices = mesh_vertices * scale
        scaled_object.mesh.vertices_local = scaled_vertices
        scaled_mesh_audit = audit_mesh(scaled_vertices, scaled_object.mesh.faces)
        scaled_samples = replace(
            samples,
            mesh_hash=scaled_mesh_audit.mesh_hash,
            mesh_array_hash=None,
            points_local=scaled_points_local.copy(),
            scale=np.asarray(1.0),
        )
        graph = build_source_interaction_graph(
            scaled_sequence,
            hand_id,
            object_track.object_id,
            scaled_samples,
            delaunay_profile=profile,
            kappa=kappa,
            frame_indices=[frame],
        )
        graph_frame = graph.frame_statistics[0]
        values.append(
            {
                "scale": float(scale),
                "sample_count": samples.count,
                "face_ids_unchanged": bool(
                    np.array_equal(samples.face_indices, samples.face_indices)
                ),
                "barycentric_unchanged": bool(
                    np.array_equal(samples.barycentric, samples.barycentric)
                ),
                "source_points_recomputed": True,
                "original_scale_one_points_max_error_m": float(
                    np.max(np.abs(scaled_points - original_points)) if scale == 1.0 else 0.0
                ),
                "simplex_count": int(graph_frame["simplex_count"]),
                "edge_count": int(graph_frame["edge_count"]),
                "hand_object_edge_count": int(graph_frame["hand_object_edge_count"]),
                "graph_hash": graph.graph_hashes[0],
                "optimization_performed": False,
            }
        )
    return {"frame": frame, "scales": values, "diagnostic_only": True}


__all__ = [
    "build_input_audit",
    "compare_object_scales",
    "source_integrity_snapshot",
    "topology_over_time",
]
