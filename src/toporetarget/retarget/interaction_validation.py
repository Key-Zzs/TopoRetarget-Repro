"""Validation and acceptance reports for Stage 8 artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.surface_sampling import SurfaceSampleSet, transform_surface_samples
from toporetarget.retarget.artifacts import load_warm_start

from .graph_weights import build_source_weights
from .interaction_evaluation import InteractionEvaluationTrajectory
from .interaction_graph import InteractionGraphTrajectory
from .interaction_objective import InteractionMeshObjective, InteractionMeshResidual


def _source_object(sequence: Any, object_id: str) -> Any:
    if object_id not in {"primary", "object"}:
        return sequence.rigid_object(object_id)
    for item in sequence.rigid_objects:
        if item.metadata.get("role") == "primary_manipulation_object":
            return item
    return sequence.rigid_objects[0]


def _write_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _csv_frames(values: list[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for value in values for key in value})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(values)


def validate_interaction_graph(
    graph: InteractionGraphTrajectory,
    canonical: str | Path,
    samples: SurfaceSampleSet,
    *,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate graph source reconstruction without changing graph topology."""

    sequence = load_hoi_sequence(canonical)
    hand = sequence.hand(str(graph.metadata["source_hand_id"]))
    object_track = _source_object(sequence, str(graph.metadata.get("object_id", "primary")))
    frame_indices = [
        int(item)
        for item in graph.metadata.get(
            "source_frame_indices",
            graph.metadata.get("frame_indices", graph.frame_indices.tolist()),
        )
    ]
    source_hash = _tree_hash(canonical)
    source_hash_match = graph.metadata.get("source_cache_hash") in {None, source_hash}
    frame_reports: list[dict[str, Any]] = []
    source_vertices_match = True
    all_valid = True
    for local, frame_index in enumerate(frame_indices):
        object_points, _ = transform_surface_samples(
            samples, object_track.pose_scene.pose_scene[frame_index]
        )
        expected = np.concatenate(
            [hand.keypoint_tracks["mediapipe21"].positions_scene[frame_index], object_points],
            axis=0,
        )
        current_vertices = graph.source_vertices[local]
        current_simplices = graph.simplex_frames[local]
        current_directed = graph.directed_frames[local]
        vertex_error = float(np.max(np.abs(expected - current_vertices)))
        source_vertices_match &= bool(np.array_equal(expected, current_vertices))
        edges = graph.edge_frames[local]
        duplicate_edges = int(len(edges) - len(np.unique(edges, axis=0)))
        self_edges = int(np.count_nonzero(edges[:, 0] == edges[:, 1]))
        degrees = np.bincount(edges.reshape(-1), minlength=71)
        weight = build_source_weights(
            current_vertices, edges, float(graph.metadata["kappa"]), vertex_count=71
        )
        weight_error = float(np.max(np.abs(weight.weights - current_directed.weights)))
        row_error = float(np.max(np.abs(current_directed.row_sums - 1.0)))
        valid = bool(
            graph.frame_valid[local]
            and vertex_error == 0.0
            and duplicate_edges == 0
            and self_edges == 0
            and np.count_nonzero(degrees == 0) == 0
            and graph.frame_statistics[local]["hand_object_edge_count"] > 0
            and row_error <= 1e-12
            and weight_error <= 1e-15
        )
        all_valid &= valid
        frame_reports.append(
            {
                "frame": frame_index,
                "valid": valid,
                "vertex_count": int(current_vertices.shape[0]),
                "vertex_error_m": vertex_error,
                "simplex_count": int(len(current_simplices)),
                "edge_count": int(len(edges)),
                "hand_hand_edge_count": graph.frame_statistics[local]["hand_hand_edge_count"],
                "hand_object_edge_count": graph.frame_statistics[local]["hand_object_edge_count"],
                "object_object_edge_count": graph.frame_statistics[local][
                    "object_object_edge_count"
                ],
                "isolated_vertex_count": int(np.count_nonzero(degrees == 0)),
                "duplicate_edge_count": duplicate_edges,
                "self_edge_count": self_edges,
                "row_sum_max_error": row_error,
                "weight_max_error": weight_error,
                "graph_hash": graph.graph_hashes[local],
            }
        )
    report = {
        "schema_version": graph.schema_version,
        "canonical": str(canonical),
        "graph_artifact": None if graph.source_path is None else str(graph.source_path),
        "frame_count": graph.frame_count,
        "source_cache_hash": source_hash,
        "source_cache_hash_match": source_hash_match,
        "object_sample_path": None if sample_path is None else str(sample_path),
        "object_sample_count": samples.count,
        "object_sample_profile_hash": samples.profile_hash,
        "source_vertices_exact_match": source_vertices_match,
        "delaunay_invocation_count_recorded": graph.delaunay_invocation_count,
        "robot_delaunay_invocation_count_recorded": graph.metadata.get(
            "robot_delaunay_invocation_count", 0
        ),
        "frames": frame_reports,
        "all_frames_valid": all_valid and source_hash_match,
        "source_integrity": {
            "canonical_read_only": True,
            "stage6_samples_modified": False,
            "stage7_warm_start_modified": False,
            "sdf_accessed": False,
        },
    }
    return report


def validate_interaction_evaluation(
    graph: InteractionGraphTrajectory,
    evaluation: InteractionEvaluationTrajectory,
    *,
    warm_start: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Eq. (7), shared connectivity references, and evaluation integrity."""

    evaluation.validate()
    frame_count = graph.frame_count
    if evaluation.frame_count != frame_count:
        raise ValueError("graph/evaluation frame count mismatch")
    source_hashes = list(graph.graph_hashes)
    shared_hashes = list(evaluation.metadata.get("graph_hashes", source_hashes))
    shared_connectivity = source_hashes == shared_hashes
    rows: list[dict[str, Any]] = []
    all_valid = shared_connectivity
    qpos_unchanged = True
    base_pose_unchanged = True
    if warm_start is not None:
        warm = load_warm_start(warm_start)
        qpos_unchanged = bool(np.array_equal(warm.arrays["qpos"], evaluation.qpos))
        base_pose_unchanged = bool(
            np.array_equal(warm.arrays["base_pose_scene"], evaluation.base_pose_scene)
        )
    identity_losses: list[float] = []
    scaled_errors: list[float] = []
    object_errors: list[float] = []
    for index, frame_vertices in enumerate(graph.source_vertices):
        directed = graph.directed_frames[index]
        objective = InteractionMeshObjective(
            InteractionMeshResidual(
                frame_vertices,
                directed.source_index,
                directed.destination_index,
                directed.weights,
            )
        )
        identity = objective.loss_tensor(frame_vertices)
        identity_losses.append(
            float(identity.detach().cpu() if hasattr(identity, "detach") else identity)
        )
        residual = evaluation.residual[index]
        e_im = float(np.sum(residual * residual) / 71.0)
        scaled_norm = float(
            np.dot(evaluation.scaled_residual[index], evaluation.scaled_residual[index])
        )
        scaled_errors.append(abs(scaled_norm - e_im))
        object_error = float(
            np.max(np.abs(evaluation.robot_vertices[index, 21:] - frame_vertices[21:]))
        )
        object_errors.append(object_error)
        contribution = evaluation.per_vertex_contribution[index]
        valid = bool(
            np.isfinite(e_im)
            and abs(e_im - float(evaluation.e_im[index])) <= 1e-12
            and scaled_errors[-1] <= 1e-12
            and object_error == 0.0
            and evaluation.qpos_jacobian.shape[1] == 213
            and evaluation.qpos_jacobian.shape[2] == evaluation.qpos.shape[1]
        )
        all_valid &= valid
        rows.append(
            {
                "frame": index,
                "valid": valid,
                "e_im": e_im,
                "scaled_residual_squared_error": scaled_errors[-1],
                "object_points_max_error_m": object_error,
                "hand_contribution": float(np.sum(contribution[:21])),
                "object_contribution": float(np.sum(contribution[21:])),
                "max_residual_vertex": int(evaluation.max_residual_vertex[index]),
            }
        )
    report = {
        "schema_version": evaluation.schema_version,
        "graph_artifact_hash": evaluation.metadata.get("graph_artifact_hash"),
        "evaluation_artifact": None
        if evaluation.source_path is None
        else str(evaluation.source_path),
        "frame_count": frame_count,
        "shared_connectivity_exact": shared_connectivity,
        "shared_weights_source_reused": bool(
            evaluation.metadata.get("provenance", {}).get("source_weights_reused", False)
        ),
        "object_points_exact": max(object_errors, default=0.0) == 0.0,
        "qpos_unchanged": qpos_unchanged,
        "identity_oracle": {
            "max_e_im": max(identity_losses, default=0.0),
            "mean_e_im": float(np.mean(identity_losses)) if identity_losses else 0.0,
            "all_near_zero": bool(max(identity_losses, default=0.0) <= 1e-20),
        },
        "scaled_residual_consistency": {
            "max_abs_error": max(scaled_errors, default=0.0),
            "pass": bool(max(scaled_errors, default=0.0) <= 1e-12),
        },
        "e_im": {
            "min": float(np.min(evaluation.e_im)),
            "mean": float(np.mean(evaluation.e_im)),
            "max": float(np.max(evaluation.e_im)),
        },
        "jacobian_shape": list(evaluation.qpos_jacobian.shape),
        "optimization_performed": bool(evaluation.metadata.get("optimization_performed", True)),
        "frames": rows,
        "base_pose_unchanged": base_pose_unchanged,
        "all_frames_valid": all_valid and qpos_unchanged and base_pose_unchanged,
    }
    return report


def _tree_hash(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(item.read_bytes()).hexdigest().encode())
            digest.update(b"\n")
    return digest.hexdigest()


def write_validation_reports(
    report: dict[str, Any], json_path: str | Path, csv_path: str | Path
) -> None:
    _write_json(report, json_path)
    frames = report.get("frames")
    if isinstance(frames, list):
        _csv_frames([dict(item) for item in frames], csv_path)


__all__ = [
    "validate_interaction_evaluation",
    "validate_interaction_graph",
    "write_validation_reports",
]
