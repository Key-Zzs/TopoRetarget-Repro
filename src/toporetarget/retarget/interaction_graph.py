"""Eq. (3)-(6) source interaction graph construction."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.surface_sampling import SurfaceSampleSet, transform_surface_samples
from toporetarget.keypoints.registry import get_layout
from toporetarget.retarget.pipeline import source_cache_hash

from .delaunay import (
    DELAUNAY_PROFILE_ID,
    DelaunayProfile,
    DelaunayResult,
    edge_category,
    extract_unique_edges,
    load_delaunay_profile,
    tetrahedralize,
)
from .graph_weights import DirectedGraphWeights, build_source_weights
from .laplacian import laplacian_numpy

HAND_VERTEX_COUNT = 21
OBJECT_VERTEX_COUNT = 50
GRAPH_VERTEX_COUNT = HAND_VERTEX_COUNT + OBJECT_VERTEX_COUNT
INTERACTION_GRAPH_SCHEMA_VERSION = "toporetarget.interaction_graph.v1"


def load_paper_kappa() -> float:
    path = Path(__file__).resolve().parents[3] / "configs" / "paper" / "retarget.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return float(values["distance_decay_kappa"])


class InteractionGraphError(RuntimeError):
    """Raised for incompatible source graph inputs or invalid graph frames."""


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float | None]:
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(value)),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95)),
        "max": float(np.max(value)),
    }


def vertex_metadata(samples: SurfaceSampleSet | None = None) -> list[dict[str, Any]]:
    layout = get_layout("mediapipe21")
    result: list[dict[str, Any]] = [
        {
            "local_index": index,
            "global_graph_index": index,
            "kind": "hand",
            "semantic_name": name,
            "source": "canonical_mediapipe21" if index < 21 else "robot_fk",
            "coordinate_frame": "S",
            "unit": "m",
        }
        for index, name in enumerate(layout.semantic_names)
    ]
    result.extend(
        {
            "local_index": index - 21,
            "global_graph_index": index,
            "kind": "object",
            "sample_id": index - 21,
            "source": "stage6_surface_sample",
            "face_id": None if samples is None else int(samples.face_indices[index - 21]),
            "barycentric": None
            if samples is None
            else np.asarray(samples.barycentric[index - 21]).tolist(),
            "coordinate_frame": "S",
            "unit": "m",
        }
        for index in range(21, GRAPH_VERTEX_COUNT)
    )
    return result


@dataclass
class InteractionGraphFrame:
    frame_index: int
    source_vertices: np.ndarray
    simplices: np.ndarray
    edges: np.ndarray
    directed: DirectedGraphWeights
    source_laplacian: np.ndarray
    simplex_volumes: np.ndarray
    normalized_simplex_volumes: np.ndarray
    statistics: dict[str, Any]
    graph_hash: str
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def directed_source_index(self) -> np.ndarray:
        return self.directed.source_index

    @property
    def directed_destination_index(self) -> np.ndarray:
        return self.directed.destination_index

    @property
    def weights(self) -> np.ndarray:
        return self.directed.weights

    @property
    def source_distance_squared(self) -> np.ndarray:
        return self.directed.distance_squared


@dataclass
class InteractionGraphTrajectory:
    metadata: dict[str, Any]
    timestamps: np.ndarray
    source_vertices: np.ndarray
    source_laplacian: np.ndarray
    simplex_frames: list[np.ndarray]
    edge_frames: list[np.ndarray]
    directed_frames: list[DirectedGraphWeights]
    frame_statistics: list[dict[str, Any]]
    frame_valid: np.ndarray
    frame_status: list[str]
    frame_indices: np.ndarray
    object_face_indices: np.ndarray
    object_barycentric: np.ndarray
    graph_hashes: list[str]
    source_vertex_metadata: list[dict[str, Any]]
    source_path: Path | None = None
    artifact_hash: str | None = None

    @property
    def schema_version(self) -> str:
        return str(self.metadata.get("schema_version", ""))

    @property
    def frame_count(self) -> int:
        return int(self.source_vertices.shape[0])

    @property
    def delaunay_invocation_count(self) -> int:
        return int(self.metadata.get("delaunay_invocation_count", 0))

    @property
    def frames(self) -> list[InteractionGraphFrame]:
        return [
            InteractionGraphFrame(
                frame_index=int(self.frame_indices[index]),
                source_vertices=self.source_vertices[index],
                simplices=self.simplex_frames[index],
                edges=self.edge_frames[index],
                directed=self.directed_frames[index],
                source_laplacian=self.source_laplacian[index],
                simplex_volumes=np.asarray([], dtype=np.float64),
                normalized_simplex_volumes=np.asarray([], dtype=np.float64),
                statistics=self.frame_statistics[index],
                graph_hash=self.graph_hashes[index],
            )
            for index in range(self.frame_count)
        ]

    def ragged_arrays(self) -> dict[str, np.ndarray]:
        """Compatibility alias used by the artifact writer."""

        values = self.arrays()
        return {
            "timestamps": values["timestamps"],
            "frame_indices": values["frame_indices"],
            "source_vertices": values["source_vertices"],
            "source_laplacian": values["source_laplacian"],
            "object_face_indices": values["object_face_indices"],
            "object_barycentric": values["object_barycentric"],
            "simplices": values["simplices_concat"],
            "simplex_offsets": values["simplex_offsets"],
            "edges": values["edges_concat"],
            "edge_offsets": values["edge_offsets"],
            "directed_source_index": values["directed_source_index"],
            "directed_destination_index": values["directed_destination_index"],
            "weights": values["weights"],
            "log_unnormalized": values["log_unnormalized"],
            "source_distance_squared": values["source_distance_squared"],
            "directed_offsets": values["directed_frame_offsets"],
            "directed_row_offsets": values["directed_row_offsets"],
            "row_sums": values["row_sums"],
            "frame_valid": values["frame_valid"],
            "status_code": values["status_codes"],
        }

    def validate(self) -> InteractionGraphTrajectory:
        if self.schema_version != INTERACTION_GRAPH_SCHEMA_VERSION:
            raise InteractionGraphError(f"unsupported graph schema: {self.schema_version!r}")
        if self.source_vertices.shape != (self.frame_count, GRAPH_VERTEX_COUNT, 3):
            raise InteractionGraphError("source_vertices must have shape [T,71,3]")
        if self.source_laplacian.shape != self.source_vertices.shape:
            raise InteractionGraphError("source_laplacian shape does not match source_vertices")
        if self.timestamps.shape != (self.frame_count,):
            raise InteractionGraphError("timestamps frame count mismatch")
        if (
            len(self.simplex_frames) != self.frame_count
            or len(self.edge_frames) != self.frame_count
        ):
            raise InteractionGraphError("ragged topology frame count mismatch")
        if len(self.directed_frames) != self.frame_count:
            raise InteractionGraphError("ragged directed frame count mismatch")
        if self.object_face_indices.shape != (OBJECT_VERTEX_COUNT,):
            raise InteractionGraphError("object face indices must have shape [50]")
        if self.object_barycentric.shape != (OBJECT_VERTEX_COUNT, 3):
            raise InteractionGraphError("object barycentric coordinates must have shape [50,3]")
        if not np.all(self.frame_valid):
            raise InteractionGraphError(f"invalid graph frames: {self.frame_status}")
        for frame, edges, directed in zip(
            self.simplex_frames, self.edge_frames, self.directed_frames, strict=True
        ):
            if frame.ndim != 2 or frame.shape[1:] != (4,):
                raise InteractionGraphError("invalid ragged simplex shape")
            if edges.ndim != 2 or edges.shape[1:] != (2,):
                raise InteractionGraphError("invalid ragged edge shape")
            directed.validate()
            if directed.directed_count != 2 * len(edges):
                raise InteractionGraphError("directed adjacency count is not two per edge")
        return self

    def arrays(self) -> dict[str, np.ndarray]:
        simplex_offsets = np.zeros(self.frame_count + 1, dtype=np.int64)
        edge_offsets = np.zeros(self.frame_count + 1, dtype=np.int64)
        directed_offsets = np.zeros(self.frame_count + 1, dtype=np.int64)
        for index in range(self.frame_count):
            simplex_offsets[index + 1] = simplex_offsets[index] + len(self.simplex_frames[index])
            edge_offsets[index + 1] = edge_offsets[index] + len(self.edge_frames[index])
            directed_offsets[index + 1] = (
                directed_offsets[index] + self.directed_frames[index].directed_count
            )
        empty_simplex = np.empty((0, 4), dtype=np.int64)
        empty_edge = np.empty((0, 2), dtype=np.int64)
        return {
            "timestamps": np.asarray(self.timestamps, dtype=np.float64),
            "frame_indices": np.asarray(self.frame_indices, dtype=np.int64),
            "source_vertices": np.asarray(self.source_vertices, dtype=np.float64),
            "source_laplacian": np.asarray(self.source_laplacian, dtype=np.float64),
            "object_face_indices": np.asarray(self.object_face_indices, dtype=np.int64),
            "object_barycentric": np.asarray(self.object_barycentric, dtype=np.float64),
            "simplices_concat": np.concatenate(self.simplex_frames, axis=0)
            if self.simplex_frames
            else empty_simplex,
            "simplex_offsets": simplex_offsets,
            "edges_concat": np.concatenate(self.edge_frames, axis=0)
            if self.edge_frames
            else empty_edge,
            "edge_offsets": edge_offsets,
            "directed_source_index": np.concatenate(
                [item.source_index for item in self.directed_frames], axis=0
            ),
            "directed_destination_index": np.concatenate(
                [item.destination_index for item in self.directed_frames], axis=0
            ),
            "weights": np.concatenate([item.weights for item in self.directed_frames], axis=0),
            "log_unnormalized": np.concatenate(
                [item.log_unnormalized for item in self.directed_frames], axis=0
            ),
            "source_distance_squared": np.concatenate(
                [item.distance_squared for item in self.directed_frames], axis=0
            ),
            "directed_row_offsets": np.stack(
                [
                    item.row_offsets + directed_offsets[index]
                    for index, item in enumerate(self.directed_frames)
                ]
            ),
            "directed_frame_offsets": directed_offsets,
            "row_sums": np.concatenate([item.row_sums for item in self.directed_frames], axis=0),
            "frame_valid": np.asarray(self.frame_valid, dtype=bool),
            "status_codes": np.asarray(
                [0 if item == "valid" else 1 for item in self.frame_status], dtype=np.int8
            ),
        }


def _category_counts(edges: np.ndarray) -> dict[str, int]:
    counts = {"hand-hand": 0, "hand-object": 0, "object-object": 0}
    for edge in edges:
        counts[edge_category(edge)] += 1
    return counts


def _frame_statistics(
    vertices: np.ndarray,
    simplices: np.ndarray,
    edges: np.ndarray,
    directed: DirectedGraphWeights,
    result: DelaunayResult,
    profile: DelaunayProfile,
) -> dict[str, Any]:
    categories = _category_counts(edges)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    category_lengths: dict[str, dict[str, float | None]] = {}
    for category in categories:
        mask = np.asarray([edge_category(edge) == category for edge in edges], dtype=bool)
        category_lengths[category] = _stats(lengths[mask])
    degrees = np.bincount(edges.reshape(-1), minlength=GRAPH_VERTEX_COUNT)
    directed_edge_categories = np.asarray(
        [
            edge_category((src, dst))
            for src, dst in zip(directed.source_index, directed.destination_index, strict=True)
        ],
        dtype=object,
    )
    hand_object_weight_mass = float(
        np.sum(directed.weights[directed_edge_categories == "hand-object"])
    )
    return {
        "simplex_count": int(len(simplices)),
        "edge_count": int(len(edges)),
        "directed_adjacency_count": int(directed.directed_count),
        "hand_hand_edge_count": categories["hand-hand"],
        "hand_object_edge_count": categories["hand-object"],
        "object_object_edge_count": categories["object-object"],
        "degree": {
            "min": int(degrees.min()),
            "mean": float(degrees.mean()),
            "max": int(degrees.max()),
        },
        "isolated_vertex_count": int(np.count_nonzero(degrees == 0)),
        "edge_length_m": _stats(lengths),
        "edge_length_by_category_m": category_lengths,
        "tetra_volume_m3": _stats(result.simplex_volumes),
        "normalized_tetra_volume": _stats(result.normalized_simplex_volumes),
        "near_degenerate_simplex_count": int(result.near_degenerate_simplex_count),
        "hand_object_edge_weight_mass": hand_object_weight_mass,
        "row_sum_max_error": float(np.max(np.abs(directed.row_sums - 1.0))),
        "point_diagnostics": result.point_diagnostics.as_dict(),
        "delaunay_profile_id": profile.profile_id,
        "edge_filter_policy": "no filtering",
    }


def _frame_graph(
    source_vertices: np.ndarray,
    *,
    frame_index: int,
    profile: DelaunayProfile,
    kappa: float,
) -> InteractionGraphFrame:
    started = time.perf_counter()
    delaunay = tetrahedralize(
        source_vertices, profile, frame_index=frame_index, expected_count=GRAPH_VERTEX_COUNT
    )
    delaunay_time = time.perf_counter() - started
    started_edges = time.perf_counter()
    edges = extract_unique_edges(
        delaunay.simplices, vertex_count=GRAPH_VERTEX_COUNT, frame_index=frame_index
    )
    edge_time = time.perf_counter() - started_edges
    started_weights = time.perf_counter()
    directed = build_source_weights(source_vertices, edges, kappa, vertex_count=GRAPH_VERTEX_COUNT)
    weight_time = time.perf_counter() - started_weights
    started_laplacian = time.perf_counter()
    source_laplacian = laplacian_numpy(
        source_vertices,
        directed.source_index,
        directed.destination_index,
        directed.weights,
    )
    laplacian_time = time.perf_counter() - started_laplacian
    stats = _frame_statistics(
        source_vertices, delaunay.simplices, edges, directed, delaunay, profile
    )
    graph_hash = _hash_arrays(
        source_vertices,
        delaunay.simplices,
        edges,
        directed.source_index,
        directed.destination_index,
        directed.weights,
        source_laplacian,
    )
    stats["graph_hash"] = graph_hash
    return InteractionGraphFrame(
        frame_index=frame_index,
        source_vertices=np.asarray(source_vertices, dtype=np.float64),
        simplices=delaunay.simplices,
        edges=edges,
        directed=directed,
        source_laplacian=source_laplacian,
        simplex_volumes=delaunay.simplex_volumes,
        normalized_simplex_volumes=delaunay.normalized_simplex_volumes,
        statistics=stats,
        graph_hash=graph_hash,
        timings={
            "delaunay_s": float(delaunay_time),
            "edge_extraction_s": float(edge_time),
            "weight_s": float(weight_time),
            "source_laplacian_s": float(laplacian_time),
        },
    )


def build_source_interaction_graph(
    sequence: HOISequence,
    hand_id: str,
    object_id: str,
    object_samples: SurfaceSampleSet,
    *,
    source_cache: str | Path | None = None,
    object_sample_path: str | Path | None = None,
    delaunay_profile: DelaunayProfile | None = None,
    kappa: float | None = None,
    frame_indices: np.ndarray | list[int] | None = None,
) -> InteractionGraphTrajectory:
    """Build one source graph per selected frame and never load a robot."""

    profile = delaunay_profile or load_delaunay_profile(DELAUNAY_PROFILE_ID)
    profile.validate()
    hand = sequence.hand(hand_id)
    if hand.side not in {"left", "right"}:
        raise InteractionGraphError(f"unsupported hand side: {hand.side!r}")
    track = hand.keypoint_tracks.get("mediapipe21")
    if track is None or track.layout_name != "mediapipe21":
        raise InteractionGraphError("source hand must provide the canonical mediapipe21 track")
    if track.positions_scene.shape != (sequence.num_frames, HAND_VERTEX_COUNT, 3):
        raise InteractionGraphError("canonical hand track must have shape [T,21,3]")
    if object_id in {"primary", "object"}:
        try:
            obj = sequence.primary_rigid_object()
        except KeyError as exc:
            raise InteractionGraphError(str(exc)) from exc
    else:
        obj = sequence.rigid_object(object_id)
    if obj.pose_scene.pose_scene.shape != (sequence.num_frames, 4, 4):
        raise InteractionGraphError("object pose frame count does not match canonical sequence")
    if object_samples.count != OBJECT_VERTEX_COUNT:
        raise InteractionGraphError(
            f"Stage 8 requires exactly 50 object samples, got {object_samples.count}"
        )
    mesh_audit = audit_mesh(obj.mesh.vertices_local, obj.mesh.faces)
    sample_validation = object_samples.validate(obj.mesh.vertices_local, obj.mesh.faces)
    if not sample_validation["mesh_hash_match"] or not sample_validation["topology_hash_match"]:
        raise InteractionGraphError("object sample artifact does not match canonical object mesh")
    if not sample_validation["count_exact"]:
        raise InteractionGraphError("object sample artifact count does not match its profile")
    selected_indices = (
        np.arange(sequence.num_frames, dtype=np.int64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    )
    if (
        selected_indices.size == 0
        or np.any(selected_indices < 0)
        or np.any(selected_indices >= sequence.num_frames)
    ):
        raise InteractionGraphError("selected frame indices are outside the canonical sequence")
    if np.unique(selected_indices).shape[0] != selected_indices.shape[0]:
        raise InteractionGraphError("selected frame indices must be unique")
    selected_kappa = load_paper_kappa() if kappa is None else float(kappa)
    frames: list[InteractionGraphFrame] = []
    source_vertices: list[np.ndarray] = []
    timestamps = np.asarray(sequence.metadata.timestamps)[selected_indices]
    for frame in selected_indices.tolist():
        hand_points = np.asarray(track.positions_scene[frame], dtype=np.float64)
        object_points, _ = transform_surface_samples(
            object_samples, np.asarray(obj.pose_scene.pose_scene[frame], dtype=np.float64)
        )
        vertices = np.concatenate([hand_points, object_points], axis=0)
        built = _frame_graph(
            vertices, frame_index=int(frame), profile=profile, kappa=selected_kappa
        )
        frames.append(built)
        source_vertices.append(vertices)
    frame_valid = np.asarray(
        [
            bool(
                frame.statistics["isolated_vertex_count"] == 0
                and frame.statistics["hand_object_edge_count"] > 0
                and frame.statistics["row_sum_max_error"] <= 1e-12
            )
            for frame in frames
        ],
        dtype=bool,
    )
    if not np.all(frame_valid):
        invalid = selected_indices[~frame_valid].tolist()
        raise InteractionGraphError(
            f"strict interaction graph has no valid hand-object connectivity in frames {invalid}"
        )
    sample_hash = None
    if object_sample_path is not None:
        sample_hash = hashlib.sha256(Path(object_sample_path).read_bytes()).hexdigest()
    layout = get_layout("mediapipe21")
    layout_hash = hashlib.sha256(
        json.dumps(
            {
                "name": layout.name,
                "version": layout.version,
                "semantic_names": layout.semantic_names,
                "parents": layout.parents,
                "edges": layout.edges,
            },
            sort_keys=True,
            default=list,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    metadata = {
        "schema_version": INTERACTION_GRAPH_SCHEMA_VERSION,
        "sequence_id": sequence.metadata.sequence_id,
        "source_hand_id": hand.hand_id,
        "source_hand_side": hand.side,
        "frame_range": [int(selected_indices[0]), int(selected_indices[-1]) + 1],
        "frame_indices": selected_indices.tolist(),
        "timestamps": timestamps.tolist(),
        "native_fps": sequence.metadata.native_fps,
        "source_cache_path": None if source_cache is None else str(source_cache),
        "source_cache_hash": source_cache_hash(source_cache),
        "object_id": obj.object_id,
        "object_mesh_hash": mesh_audit.mesh_hash,
        "object_topology_hash": mesh_audit.topology_hash,
        "object_sample_artifact_path": None
        if object_sample_path is None
        else str(object_sample_path),
        "object_sample_artifact_hash": sample_hash,
        "object_sample_profile_id": object_samples.profile_id,
        "object_sample_profile_hash": object_samples.profile_hash,
        "object_sample_count": object_samples.count,
        "object_face_identity_preserved": True,
        "object_barycentric_identity_preserved": True,
        "hand_layout_name": layout.name,
        "hand_layout_hash": layout_hash,
        "coordinate_frame": "S",
        "units": "m",
        "vertex_count": GRAPH_VERTEX_COUNT,
        "hand_vertex_count": HAND_VERTEX_COUNT,
        "object_vertex_count": OBJECT_VERTEX_COUNT,
        "vertex_index_semantics": "0..20=mediapipe21;21..70=Stage6 sample order",
        "delaunay_profile_id": profile.profile_id,
        "delaunay_profile_hash": profile.sha256,
        "delaunay_backend": profile.backend,
        "delaunay_invocation_count": len(frames),
        "delaunay_robot_invocation_count": 0,
        "robot_delaunay_invocation_count": 0,
        "scipy_version": __import__("scipy").__version__,
        "numpy_version": np.__version__,
        "qhull_options": profile.qhull_options,
        "delaunay_normalization": profile.normalization,
        "kappa": selected_kappa,
        "edge_filter_policy": "no filtering",
        "object_face_indices": np.asarray(object_samples.face_indices, dtype=np.int64).tolist(),
        "object_barycentric": np.asarray(object_samples.barycentric, dtype=np.float64).tolist(),
        "source_vertex_metadata": vertex_metadata(object_samples),
        "graph_hashes": [frame.graph_hash for frame in frames],
        "frame_statistics": [frame.statistics for frame in frames],
        "frame_status": ["valid" if valid else "invalid" for valid in frame_valid],
        "assumptions": [
            "A_DELAUNAY_BACKEND_001",
            "A_DELAUNAY_OPTIONS_001",
            "A_DELAUNAY_DEGENERACY_001",
            "A_INTERACTION_EDGE_FILTERING_001",
            "A_INTERACTION_GRAPH_FRAME_001",
            "A_INTERACTION_GRAPH_REBUILD_001",
            "A_LAPLACIAN_WEIGHT_NUMERICS_001",
            "A_OBJECT_SAMPLING_001",
            "A_OBJECT_SAMPLING_METHOD_001",
            "A_OBJECT_SAMPLE_TEMPORAL_REUSE_001",
        ],
        "provenance": {
            "robot_loaded": False,
            "warm_start_loaded": False,
            "sdf_accessed": False,
            "robot_collision_surface_accessed": False,
            "object_samples_resampled": False,
            "object_sample_order_changed": False,
        },
        "timings": {
            "source_vertex_assembly_s": float(sum(sum(frame.timings.values()) for frame in frames)),
            "delaunay_s": float(sum(frame.timings["delaunay_s"] for frame in frames)),
            "edge_extraction_s": float(sum(frame.timings["edge_extraction_s"] for frame in frames)),
            "weight_s": float(sum(frame.timings["weight_s"] for frame in frames)),
            "source_laplacian_s": float(
                sum(frame.timings["source_laplacian_s"] for frame in frames)
            ),
        },
    }
    result = InteractionGraphTrajectory(
        metadata=metadata,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        source_vertices=np.stack(source_vertices),
        source_laplacian=np.stack([frame.source_laplacian for frame in frames]),
        simplex_frames=[frame.simplices for frame in frames],
        edge_frames=[frame.edges for frame in frames],
        directed_frames=[frame.directed for frame in frames],
        frame_statistics=[frame.statistics for frame in frames],
        frame_valid=frame_valid,
        frame_status=["valid" if valid else "invalid" for valid in frame_valid],
        frame_indices=selected_indices,
        object_face_indices=np.asarray(object_samples.face_indices, dtype=np.int64),
        object_barycentric=np.asarray(object_samples.barycentric, dtype=np.float64),
        graph_hashes=[frame.graph_hash for frame in frames],
        source_vertex_metadata=vertex_metadata(object_samples),
    )
    return result.validate()


__all__ = [
    "GRAPH_VERTEX_COUNT",
    "HAND_VERTEX_COUNT",
    "INTERACTION_GRAPH_SCHEMA_VERSION",
    "InteractionGraphError",
    "InteractionGraphFrame",
    "InteractionGraphTrajectory",
    "OBJECT_VERTEX_COUNT",
    "build_source_interaction_graph",
    "load_paper_kappa",
    "vertex_metadata",
]
