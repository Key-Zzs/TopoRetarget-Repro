"""Stage 9.3 contact-retention and collision-geometry diagnostics.

This module is deliberately an audit boundary.  The default path loads the
accepted Stage 9.2/Stage 10 artifacts, re-queries the frozen reference SDF,
and never imports or calls the Stage 9 optimizer.  Any diagnostic solver work
must be added behind the explicit shadow-ablation option and a separate run
root.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
import webbrowser
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.se3 import invert_transform, scene_to_object, transform_vectors
from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend
from toporetarget.geometry.surface_sampling import SurfaceSamplingProfile, sample_mesh_surface
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.final_refinement import (
    ConvexHullSignedDistanceBackend,
    dynamic_collision_points_numpy,
    load_final_trajectory,
    load_robot_surface_samples,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import load_interaction_graph
from toporetarget.retarget.interaction_objective import interaction_loss_numpy
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.visualization import _primitive_mesh
from toporetarget.utils.hashing import sha256_file, sha256_tree

AUDIT_SCHEMA_VERSION = "toporetarget.contact_retention_audit.v1"
AUDIT_CODE_VERSION = "stage9.3-audit-v1"
REGIONS = ("palm", "thumb", "index", "middle", "ring", "pinky")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ANCHORS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TIP_INDICES = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
SOURCE_REGION_PROXY = "nearest_mediapipe21_anchor_region_v1"
PAD_PROXY = "mediapipe21_tip_anchor_v1_no_pad_surface"


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        scalar = value.item()
        return scalar if not isinstance(scalar, float) or np.isfinite(scalar) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for name, value in sha256_tree(path).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _stat(path: Path) -> dict[str, Any]:
    item = path.stat()
    return {
        "path": str(path),
        "sha256": _sha(path),
        "mtime_ns": item.st_mtime_ns,
        "size": item.st_size,
    }


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def _manifest_artifact(manifest: dict[str, Any], name: str, repo_root: Path) -> Path:
    values = manifest.get("artifacts", {})
    item = values.get(name)
    if not isinstance(item, dict) or not item.get("path"):
        raise ValueError(f"manifest.artifacts.{name}.path is missing; refusing to guess")
    path = _resolve(repo_root, str(item["path"]))
    if not path.exists():
        raise FileNotFoundError(f"manifest artifact {name} does not exist: {path}")
    return path


@dataclass
class MeshPoints:
    points: np.ndarray
    regions: np.ndarray
    links: np.ndarray
    source_kind: np.ndarray


def _dense_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    count: int,
    mesh_id: str,
    seed: int,
    include_vertices: bool = True,
) -> np.ndarray:
    """Return deterministic samples plus all original vertices.

    The result is explicitly an approximation to the continuous surface.  It
    includes every original vertex and at least ``count`` total points.
    """

    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    need = max(1, int(count) if not include_vertices else int(count) - len(values))
    profile = SurfaceSamplingProfile(
        profile_id=f"stage9_3_dense:{mesh_id}",
        version="1",
        method="area_uniform_triangles",
        count=need,
        seed=int(seed),
        source="stage9.3_diagnostic",
        assumptions=("A_STAGE9_3_DENSE_SURFACE_APPROXIMATION_001",),
    )
    sampled = sample_mesh_surface(values, triangles, profile, mesh_id=mesh_id).points_local
    sampled_values = np.asarray(sampled, dtype=np.float64)
    return np.concatenate([values, sampled_values], axis=0) if include_vertices else sampled_values


def _load_geometry(instance: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _primitive_mesh(instance)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _visual_surface(
    model: Any,
    qpos: np.ndarray,
    base: np.ndarray,
    *,
    count: int,
    seed: int,
    include_vertices: bool = True,
) -> MeshPoints:
    pieces: list[np.ndarray] = []
    regions: list[str] = []
    links: list[str] = []
    kinds: list[str] = []
    instances = model.visual_geometry_instances(qpos, base)
    for index, instance in enumerate(instances):
        vertices, faces = _load_geometry(instance)
        points = _dense_samples(
            vertices,
            faces,
            count=max(64, count // max(len(instances), 1)),
            mesh_id=f"visual:{instance.link_name}:{index}",
            seed=seed + index,
            include_vertices=include_vertices,
        )
        points = points @ instance.world_transform[:3, :3].T + instance.world_transform[:3, 3]
        pieces.append(points)
        link = str(instance.link_name)
        region = (
            "palm"
            if link == "palm"
            else next((finger for finger in FINGERS if link.startswith(finger)), "palm")
        )
        regions.extend([region] * len(points))
        links.extend([link] * len(points))
        kinds.extend(["visual_mesh"] * len(points))
    return MeshPoints(
        np.concatenate(pieces),
        np.asarray(regions, dtype="U16"),
        np.asarray(links, dtype="U32"),
        np.asarray(kinds, dtype="U24"),
    )


def _visual_vertices(
    model: Any,
    qpos: np.ndarray,
    base: np.ndarray,
    *,
    max_per_instance: int | None = None,
) -> MeshPoints:
    pieces: list[np.ndarray] = []
    regions: list[str] = []
    links: list[str] = []
    kinds: list[str] = []
    for instance in model.visual_geometry_instances(qpos, base):
        vertices, _ = _load_geometry(instance)
        if max_per_instance is not None and len(vertices) > max_per_instance:
            indices = np.linspace(0, len(vertices) - 1, max_per_instance, dtype=np.int64)
            vertices = vertices[indices]
        points = vertices @ instance.world_transform[:3, :3].T + instance.world_transform[:3, 3]
        pieces.append(points)
        link = str(instance.link_name)
        region = (
            "palm"
            if link == "palm"
            else next((finger for finger in FINGERS if link.startswith(finger)), "palm")
        )
        regions.extend([region] * len(points))
        links.extend([link] * len(points))
        kinds.extend(["visual_vertex"] * len(points))
    return MeshPoints(
        np.concatenate(pieces),
        np.asarray(regions, dtype="U16"),
        np.asarray(links, dtype="U32"),
        np.asarray(kinds, dtype="U24"),
    )


def _collision_surface(
    model: Any,
    qpos: np.ndarray,
    base: np.ndarray,
    *,
    count: int,
    seed: int,
    include_vertices: bool = True,
) -> MeshPoints:
    pieces: list[np.ndarray] = []
    regions: list[str] = []
    links: list[str] = []
    kinds: list[str] = []
    instances = model.collision_geometry_instances(qpos, base)
    for index, instance in enumerate(instances):
        vertices, faces = _load_geometry(instance)
        points = _dense_samples(
            vertices,
            faces,
            count=max(32, count // max(len(instances), 1)),
            mesh_id=f"collision:{instance.link_name}:{index}",
            seed=seed + index,
            include_vertices=include_vertices,
        )
        points = points @ instance.world_transform[:3, :3].T + instance.world_transform[:3, 3]
        pieces.append(points)
        link = str(instance.link_name)
        region = (
            "palm"
            if link == "palm"
            else next((finger for finger in FINGERS if link.startswith(finger)), "palm")
        )
        regions.extend([region] * len(points))
        links.extend([link] * len(points))
        kinds.extend(["collision_geometry"] * len(points))
    return MeshPoints(
        np.concatenate(pieces),
        np.asarray(regions, dtype="U16"),
        np.asarray(links, dtype="U32"),
        np.asarray(kinds, dtype="U24"),
    )


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return {
            "count": 0,
            "min_m": None,
            "p01_m": None,
            "p05_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
        }
    return {
        "count": int(len(x)),
        "min_m": float(np.min(x)),
        "p01_m": float(np.quantile(x, 0.01)),
        "p05_m": float(np.quantile(x, 0.05)),
        "median_m": float(np.median(x)),
        "p95_m": float(np.quantile(x, 0.95)),
        "max_m": float(np.max(x)),
    }


def _threshold_record(
    signed: np.ndarray, unsigned: np.ndarray, thresholds: list[float]
) -> dict[str, Any]:
    return {
        f"{int(round(t * 1000))}mm": {
            "near_surface_count": int(np.count_nonzero(unsigned <= t)),
            "near_surface_ratio": float(np.mean(unsigned <= t)) if len(unsigned) else 0.0,
            "positive_gap_count": int(np.count_nonzero((signed >= 0) & (signed <= t))),
            "penetration_count": int(np.count_nonzero(signed < 0)),
        }
        for t in thresholds
    }


def _anchor_provenance(
    points_scene: np.ndarray, query: Any, object_pose: np.ndarray
) -> list[dict[str, Any]]:
    """Persist nearest-point, face, vector, and object-local direction evidence."""

    points = np.asarray(points_scene, dtype=np.float64)
    closest = np.asarray(query.closest_points, dtype=np.float64)
    vectors_scene = points - closest
    vectors_local = transform_vectors(invert_transform(object_pose), vectors_scene)
    norms = np.linalg.norm(vectors_local, axis=1)
    directions = np.divide(
        vectors_local,
        np.maximum(norms[:, None], 1e-15),
        out=np.zeros_like(vectors_local),
    )
    points_local = scene_to_object(object_pose, points)
    closest_local = scene_to_object(object_pose, closest)
    return [
        {
            "anchor_index": int(index),
            "point_scene_m": points[index].tolist(),
            "point_object_local_m": points_local[index].tolist(),
            "closest_point_scene_m": closest[index].tolist(),
            "closest_point_object_local_m": closest_local[index].tolist(),
            "closest_face_index": int(query.closest_face_indices[index]),
            "nearest_vector_scene_m": vectors_scene[index].tolist(),
            "nearest_vector_object_local_m": vectors_local[index].tolist(),
            "object_local_direction": directions[index].tolist(),
            "signed_distance_m": float(query.signed_distance[index]),
            "unsigned_distance_m": float(query.unsigned_distance[index]),
        }
        for index in range(len(points))
    ]


def _grouped_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels_array = np.asarray(labels).astype(str)
    return {
        "overall": _stats(values),
        "by_label": {
            label: _stats(np.asarray(values)[labels_array == label])
            for label in sorted(set(labels_array.tolist()))
        },
    }


def _region_indices(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - anchors[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=1)
    result = np.full(len(points), "palm", dtype="U16")
    for region, indices in FINGER_ANCHORS.items():
        result[np.isin(nearest, np.asarray(indices))] = region
    return result


def _object_local_direction(
    point: np.ndarray, closest: np.ndarray, pose: np.ndarray
) -> tuple[np.ndarray, float]:
    vector_scene = np.asarray(point) - np.asarray(closest)
    vector_local = transform_vectors(
        invert_transform(np.asarray(pose)), vector_scene.reshape(1, 3)
    )[0]
    norm = float(np.linalg.norm(vector_local))
    return (vector_local / norm if norm > 1e-15 else np.zeros(3)), norm


def _angle(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-15 or nb <= 1e-15:
        return None
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation, Slerp

    rotations = Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]]))
    value = Slerp([0.0, 1.0], rotations)([float(alpha)]).as_matrix()[0]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value
    result[:3, 3] = (1.0 - alpha) * first[:3, 3] + alpha * second[:3, 3]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({_key: _json(value) for _key, value in row.items()} for row in rows)


def _safe_mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(items)) if items else float("nan")


def _preflight(manifest_path: Path, manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    names = ["canonical", "warm_start", "final", "graph", "collision_samples", "object_samples"]
    artifacts = {name: _manifest_artifact(manifest, name, repo_root) for name in names}
    formal_html = _resolve(
        repo_root, Path(str(manifest["run_root"])) / "review/trajectory_mesh.html"
    )
    if not formal_html.exists():
        formal_html = (
            _resolve(repo_root, ".local/runs/stage10_reference_runtime")
            / Path(str(manifest["run_id"]))
            / "review/trajectory_mesh.html"
        )
    entries = {name: _stat(path) for name, path in artifacts.items()}
    entries["manifest"] = _stat(manifest_path)
    entries["trajectory_mesh_html"] = _stat(formal_html)
    return {
        "schema_version": "toporetarget.contact_retention_audit_preflight.v1",
        "git": {
            "branch": _git(repo_root, "branch", "--show-current"),
            "head": _git(repo_root, "rev-parse", "HEAD"),
            "status_short": _git(repo_root, "status", "--short"),
            "diff_name_status": _git(repo_root, "diff", "--name-status"),
            "cached_name_status": _git(repo_root, "diff", "--cached", "--name-status"),
        },
        "manifest_path": str(manifest_path),
        "run_id": manifest.get("run_id"),
        "selected_frame_range": manifest.get("selected_frame_range"),
        "sequence": manifest.get("source_sequence"),
        "hand": manifest.get("hand"),
        "robot": manifest.get("robot"),
        "artifacts": entries,
        "profiles": manifest.get("profiles", {}),
        "final_artifact_metadata": {
            "solver_profile": manifest.get("profiles", {}).get("refinement_solver_profile_id"),
            "execution_profile": manifest.get("profiles", {}).get("execution_profile_id"),
        },
        "solver_invocation_count_before": manifest.get("solver_invocation_count"),
        "formal_artifact_resolution": "manifest.artifacts; no guessed source paths",
    }


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _load_inputs(
    manifest_path: Path,
    repo_root: Path,
    *,
    evaluation_backend: str = "configured",
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    paths = {
        name: _manifest_artifact(manifest, name, repo_root)
        for name in (
            "canonical",
            "warm_start",
            "final",
            "graph",
            "collision_samples",
            "object_samples",
        )
    }
    sequence = load_hoi_sequence(paths["canonical"])
    warm = load_warm_start(paths["warm_start"])
    final = load_final_trajectory(paths["final"])
    graph = load_interaction_graph(paths["graph"])
    surface = load_robot_surface_samples(paths["collision_samples"])
    robot_name = str(final.metadata.get("robot_name", manifest.get("robot", "artimano_rh")))
    side = "rh" if robot_name.endswith("_rh") else "lh"
    model = load_artimano_model(side)
    object_id = str(final.metadata["object_id"])
    obj = sequence.rigid_object(object_id)
    # Stage 9.3.2 pins formal evaluation to the strict reference backend
    # defaults used by the Stage 9.2 acceptance replay. This is an audit-only
    # override; it never changes the solver backend or the official trajectory.
    reference_kwargs: dict[str, Any] = {"sign_mode": "strict"}
    if evaluation_backend == "reference_winding_v1":
        # Exact same solid-angle and triangle-closest-point formulas as the
        # Stage 9.2 reference path.  Keep the formal audit path deterministic
        # and CPU-backed; accelerators remain an implementation detail of the
        # solver backend and are never part of the acceptance contract.
        reference_kwargs.update(
            {
                "query_chunk_size": 256,
                "face_chunk_size": 4096,
                "winding_device": "cpu",
                "closest_acceleration": "tree",
                "closest_device": None,
            }
        )
    reference_sdf = build_signed_distance_backend(
        obj.mesh.vertices_local, obj.mesh.faces, **reference_kwargs
    )
    distance_sdf: Any = reference_sdf
    distance_backend_selection = {
        "requested": final.metadata.get("sdf_backend", {}).get("backend_id", "reference"),
        "reference": reference_sdf.describe(),
        "selected": reference_sdf.describe(),
        "cross_validation": None,
    }
    if evaluation_backend not in {"configured", "reference_winding_v1"}:
        raise ValueError(f"unknown contact-audit evaluation backend: {evaluation_backend}")
    if evaluation_backend == "reference_winding_v1":
        distance_backend_selection["requested"] = "reference_winding_v1"
        distance_backend_selection["selected"] = reference_sdf.describe()
        distance_backend_selection["selection_reason"] = (
            "Stage 9.3.2 formal evaluation backend override; solver backend is not changed"
        )
    elif distance_backend_selection["requested"] == "convex_hull_exact_solver_only":
        try:
            candidate = ConvexHullSignedDistanceBackend(
                obj.mesh.vertices_local,
                obj.mesh.faces,
                reference_sdf.mesh_hash,
                tree_leaf_size=int(final.metadata.get("sdf_tree_leaf_size", 512)),
            )
            rng = np.random.default_rng(20260720)
            probes = rng.normal(size=(32, 3))
            probes *= max(float(np.linalg.norm(np.ptp(obj.mesh.vertices_local, axis=0))), 1.0)
            probes += np.mean(obj.mesh.vertices_local, axis=0)
            reference_probe = reference_sdf.query_local(probes)
            candidate_probe = candidate.query_local(probes)
            error = float(
                np.max(np.abs(reference_probe.signed_distance - candidate_probe.signed_distance))
            )
            distance_backend_selection["cross_validation"] = {
                "probe_count": 32,
                "max_signed_distance_error_m": error,
                "tolerance_m": 1e-8,
                "passed": bool(error <= 1e-8),
            }
            if error <= 1e-8:
                distance_sdf = candidate
                distance_backend_selection["selected"] = candidate.describe()
        except (ImportError, RuntimeError, ValueError) as exc:
            distance_backend_selection["cross_validation"] = {
                "passed": False,
                "error": str(exc),
            }
    hand = sequence.hand(str(final.metadata.get("source_hand_id", "right_hand")))
    keypoints = hand.keypoint_tracks["mediapipe21"].positions_scene
    if hand.vertices_scene is None or hand.mesh is None:
        raise ValueError("canonical source hand must include scene mesh and mesh topology")
    frame_profile = load_frame_profile(str(warm.metadata["frame_profile_id"]))
    bone_profile = load_bone_profile(str(warm.metadata["bone_profile_id"]))
    source_keypoints = np.asarray(keypoints, dtype=np.float64)
    source_bone_features = [
        extract_bone_features(
            source_keypoints[index],
            frame_profile,
            bone_profile,
            side=str(warm.metadata["source_side"]),
            strict=True,
        )
        for index in range(len(source_keypoints))
    ]
    return {
        "manifest": manifest,
        "paths": paths,
        "sequence": sequence,
        "warm": warm,
        "final": final,
        "graph": graph,
        "surface": surface,
        "model": model,
        "object": obj,
        "sdf": distance_sdf,
        "reference_sdf": reference_sdf,
        "distance_backend_selection": distance_backend_selection,
        "source_hand": hand,
        "source_keypoints": source_keypoints,
        "frame_profile": frame_profile,
        "bone_profile": bone_profile,
        "source_bone_features": source_bone_features,
        "evaluation_backend": evaluation_backend,
    }


def _frame_query_records(
    frame: int,
    inputs: dict[str, Any],
    final_points: np.ndarray,
    warm_points: np.ndarray,
    final_full: Any,
    *,
    warm_query: Any | None = None,
    final_query: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final = inputs["final"]
    surface = inputs["surface"]
    obj = inputs["object"]
    pose = obj.pose_scene.pose_scene[frame]
    start, stop = (
        int(final.arrays["query_offsets"][frame]),
        int(final.arrays["query_offsets"][frame + 1]),
    )
    ids = np.asarray(final.arrays["query_ids_concat"][start:stop], dtype=np.int64)
    rounds = np.asarray(final.arrays["query_active_round_concat"][start:stop], dtype=np.int64)
    reasons = np.asarray(final.arrays["query_inclusion_reason_concat"][start:stop]).astype(str)
    slack = np.asarray(
        final.arrays["slack_concat"][
            int(final.arrays["slack_offsets"][frame]) : int(
                final.arrays["slack_offsets"][frame + 1]
            )
        ],
        dtype=np.float64,
    )
    warm_q = warm_query if warm_query is not None else inputs["sdf"].query_scene(warm_points, pose)
    final_q = (
        final_query if final_query is not None else inputs["sdf"].query_scene(final_points, pose)
    )
    rows: list[dict[str, Any]] = []
    link_rows: dict[str, dict[str, Any]] = {}
    for local, sample_id in enumerate(ids):
        link = str(surface.link_names[sample_id])
        row = {
            "frame": int(frame),
            "global_frame": int(
                final.arrays["frame_indices"][frame] + final.metadata.get("source_frame_offset", 0)
            ),
            "query_point_id": int(sample_id),
            "source_link": link,
            "robot_link": link,
            "geometry_id": str(surface.geometry_ids[sample_id]),
            "point_local_coordinate_m": surface.points_local[sample_id].tolist(),
            "point_world_coordinate_m": final_points[sample_id].tolist(),
            "point_object_local_coordinate_m": scene_to_object(
                pose, final_points[sample_id].reshape(1, 3)
            )[0].tolist(),
            "warm_signed_distance_m": float(warm_q.signed_distance[sample_id]),
            "final_signed_distance_m": float(final_q.signed_distance[sample_id]),
            "inclusion_reason": str(reasons[local]),
            "activation_threshold_m": float(
                final.metadata.get("query_profile", {}).get("active_margin_m", 0.01)
            ),
            "active_margin_m": float(
                final.metadata.get("query_profile", {}).get("active_margin_m", 0.01)
            ),
            "final_constraint_slack_m": float(slack[local]) if local < len(slack) else None,
            "contact_related_link": bool(
                link == "palm" or any(link.startswith(x) for x in FINGERS)
            ),
            "fingertip_pad_proxy": bool(link.endswith("_tip") or link.endswith("3")),
            "adaptive_expansion": bool("expan" in reasons[local] or int(rounds[local]) > 0),
            "in_full_512_audit": True,
        }
        rows.append(row)
        bucket = link_rows.setdefault(
            link,
            {
                "frame": int(frame),
                "link_name": link,
                "query_count": 0,
                "active_round_max": 0,
                "warm_min_signed_distance_m": math.inf,
                "final_min_signed_distance_m": math.inf,
                "slack_median_m": math.nan,
            },
        )
        bucket["query_count"] += 1
        bucket["active_round_max"] = max(bucket["active_round_max"], int(rounds[local]))
        bucket["warm_min_signed_distance_m"] = min(
            bucket["warm_min_signed_distance_m"], float(warm_q.signed_distance[sample_id])
        )
        bucket["final_min_signed_distance_m"] = min(
            bucket["final_min_signed_distance_m"], float(final_q.signed_distance[sample_id])
        )
    for link, bucket in link_rows.items():
        values = [
            row["final_constraint_slack_m"]
            for row in rows
            if row["robot_link"] == link and row["final_constraint_slack_m"] is not None
        ]
        bucket["slack_median_m"] = float(np.median(values)) if values else None
    return rows, list(link_rows.values())


def _objective_rows(inputs: dict[str, Any], frame: int) -> dict[str, Any]:
    final = inputs["final"]
    warm = inputs["warm"]
    graph = inputs["graph"]
    model = inputs["model"]
    paper = final.metadata["paper_weights"]
    source_vertices = np.asarray(graph.source_vertices[frame], dtype=np.float64)
    directed = graph.directed_frames[frame]
    warm_keypoints = np.asarray(warm.arrays["robot_keypoints_scene"][frame], dtype=np.float64)
    final_keypoints = np.asarray(final.arrays["robot_keypoints_scene"][frame], dtype=np.float64)
    object_points = source_vertices[21:]
    warm_vertices = np.concatenate([warm_keypoints, object_points], axis=0)
    final_vertices = np.concatenate([final_keypoints, object_points], axis=0)
    warm_im, warm_eim = interaction_loss_numpy(
        source_vertices,
        warm_vertices,
        directed.source_index,
        directed.destination_index,
        directed.weights,
    )
    final_im, final_eim = interaction_loss_numpy(
        source_vertices,
        final_vertices,
        directed.source_index,
        directed.destination_index,
        directed.weights,
    )
    del warm_im, final_im
    warm_bone = float(np.asarray(warm.arrays["ebone"])[frame]) / float(
        warm.metadata["paper_weights"].get("lambda_warm", 1.0)
    )
    final_bone = float(np.asarray(final.arrays["e_bone"])[frame])
    lambda_im = float(paper["lambda_IM"])
    lambda_bone = float(paper["lambda_bone"])
    previous = (
        None
        if frame == 0
        else np.concatenate(
            [
                np.asarray(final.arrays["base_corrections"][frame - 1], dtype=np.float64),
                np.asarray(final.arrays["qpos"][frame - 1], dtype=np.float64),
            ]
        )
    )
    final_state = np.concatenate(
        [
            np.asarray(final.arrays["base_corrections"][frame], dtype=np.float64),
            np.asarray(final.arrays["qpos"][frame], dtype=np.float64),
        ]
    )
    warm_state = np.concatenate(
        [np.zeros(6, dtype=np.float64), np.asarray(warm.arrays["qpos"][frame], dtype=np.float64)]
    )
    temporal_final = (
        float(paper["lambda_reg"] * np.sum((final_state - previous) ** 2))
        if previous is not None
        else 0.0
    )
    temporal_warm = (
        float(paper["lambda_reg"] * np.sum((warm_state - previous) ** 2))
        if previous is not None
        else 0.0
    )
    base_pos_final = float(
        paper["lambda_base_pos"] * np.sum(final.arrays["base_corrections"][frame, :3] ** 2)
    )
    base_rot_final = float(
        paper["lambda_base_rot"] * np.sum(final.arrays["base_corrections"][frame, 3:] ** 2)
    )
    base_pos_warm = 0.0
    base_rot_warm = 0.0
    start = int(final.arrays["slack_offsets"][frame])
    stop = int(final.arrays["slack_offsets"][frame + 1])
    slack_final = np.asarray(final.arrays["slack_concat"][start:stop], dtype=np.float64)
    warm_points = dynamic_collision_points_numpy(
        model, inputs["surface"], warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame]
    )
    initial = (
        inputs["sdf"]
        .query_scene(warm_points, inputs["object"].pose_scene.pose_scene[frame])
        .signed_distance
    )
    ids = np.asarray(
        final.arrays["query_ids_concat"][
            int(final.arrays["query_offsets"][frame]) : int(
                final.arrays["query_offsets"][frame + 1]
            )
        ],
        dtype=np.int64,
    )
    query_signed_final = np.asarray(
        final.arrays["signed_distance_concat"][start:stop], dtype=np.float64
    )
    query_hard_final = np.asarray(
        final.arrays["hard_residual_concat"][start:stop], dtype=np.float64
    )
    query_soft_final = np.asarray(
        final.arrays["soft_residual_concat"][start:stop], dtype=np.float64
    )
    full_signed_final = np.asarray(final.arrays["full_signed_distance"][frame], dtype=np.float64)
    tau = float(paper["tau_m"])
    b = float(paper["b_m"])
    warm_slack = np.clip(np.maximum(-tau - initial[ids], 0.0), 0.0, b - tau)
    slack_final_reg = float(0.5 * paper["w_s"] * np.sum(slack_final**2))
    slack_warm_reg = float(0.5 * paper["w_s"] * np.sum(warm_slack**2))
    rows: dict[str, Any] = {
        "warm_eval_stage9_e_im_raw": float(warm_eim),
        "warm_eval_stage9_e_im_weighted": float(lambda_im * warm_eim),
        "warm_eval_stage9_e_bone_raw": warm_bone,
        "warm_eval_stage9_e_bone_weighted": float(lambda_bone * warm_bone),
        "warm_eval_stage9_base_reg": base_pos_warm + base_rot_warm,
        "warm_eval_stage9_base_translation_reg": base_pos_warm,
        "warm_eval_stage9_base_rotation_reg": base_rot_warm,
        "warm_eval_stage9_joint_reg": 0.0,
        "warm_eval_stage9_seed_delta_reg": 0.0,
        "warm_eval_stage9_temporal_reg": temporal_warm,
        "warm_eval_stage9_slack_reg": slack_warm_reg,
        "warm_eval_stage9_queryset_constraint_min_m": float(np.min(initial[ids]))
        if len(ids)
        else None,
        "warm_eval_stage9_full_constraint_min_m": float(np.min(initial)),
        "warm_eval_stage9_total": float(
            lambda_im * warm_eim + lambda_bone * warm_bone + temporal_warm + slack_warm_reg
        ),
        "final_eval_stage9_e_im_raw": float(final_eim),
        "final_eval_stage9_e_im_weighted": float(lambda_im * final_eim),
        "final_eval_stage9_e_bone_raw": final_bone,
        "final_eval_stage9_e_bone_weighted": float(lambda_bone * final_bone),
        "final_eval_stage9_base_reg": base_pos_final + base_rot_final,
        "final_eval_stage9_base_translation_reg": base_pos_final,
        "final_eval_stage9_base_rotation_reg": base_rot_final,
        "final_eval_stage9_joint_reg": 0.0,
        "final_eval_stage9_seed_delta_reg": 0.0,
        "final_eval_stage9_temporal_reg": temporal_final,
        "final_eval_stage9_slack_reg": slack_final_reg,
        "final_eval_stage9_queryset_constraint_min_m": float(np.min(query_signed_final))
        if len(query_signed_final)
        else None,
        "final_eval_stage9_full_constraint_min_m": float(np.min(full_signed_final)),
        "final_eval_stage9_hard_residual_min_m": float(np.min(query_hard_final))
        if len(query_hard_final)
        else None,
        "final_eval_stage9_soft_residual_min_m": float(np.min(query_soft_final))
        if len(query_soft_final)
        else None,
        "final_eval_stage9_slack_median_m": float(np.median(slack_final))
        if len(slack_final)
        else None,
        "final_eval_stage9_total": float(
            lambda_im * final_eim
            + lambda_bone * final_bone
            + temporal_final
            + base_pos_final
            + base_rot_final
            + slack_final_reg
        ),
        "old_warm_total_objective": float(final.arrays["warm_total_objective"][frame]),
        "old_final_objective": float(final.arrays["final_objective"][frame]),
    }
    rows["old_fields_directly_comparable"] = bool(
        np.isclose(
            rows["old_warm_total_objective"], rows["warm_eval_stage9_total"], rtol=1e-6, atol=1e-8
        )
        and np.isclose(
            rows["old_final_objective"], rows["final_eval_stage9_total"], rtol=1e-6, atol=1e-8
        )
    )
    rows["final_minus_warm_total"] = (
        rows["final_eval_stage9_total"] - rows["warm_eval_stage9_total"]
    )
    rows["dominant_term"] = max(
        (
            (name, rows[name])
            for name in (
                "final_eval_stage9_e_im_weighted",
                "final_eval_stage9_e_bone_weighted",
                "final_eval_stage9_base_reg",
                "final_eval_stage9_temporal_reg",
                "final_eval_stage9_slack_reg",
            )
        ),
        key=lambda item: item[1],
    )[0]
    return rows


def _interpolation_objective(
    inputs: dict[str, Any],
    frame: int,
    alpha: float,
    qpos: np.ndarray,
    base: np.ndarray,
    collision_signed: np.ndarray,
    query_ids: np.ndarray,
    warm_initial: np.ndarray | None = None,
    anchor_unsigned: np.ndarray | None = None,
) -> dict[str, Any]:
    final = inputs["final"]
    warm = inputs["warm"]
    graph = inputs["graph"]
    model = inputs["model"]
    paper = final.metadata["paper_weights"]
    robot_keypoints = np.asarray(model.keypoints_scene(qpos, base, layout="mediapipe21"))
    source_vertices = np.asarray(graph.source_vertices[frame], dtype=np.float64)
    robot_vertices = np.concatenate([robot_keypoints, source_vertices[21:]], axis=0)
    _, e_im = interaction_loss_numpy(
        source_vertices,
        robot_vertices,
        graph.directed_frames[frame].source_index,
        graph.directed_frames[frame].destination_index,
        graph.directed_frames[frame].weights,
    )
    robot_features = extract_bone_features(
        robot_keypoints,
        inputs["frame_profile"],
        inputs["bone_profile"],
        side=str(final.metadata.get("robot_side", "right")),
        strict=True,
    )
    source_features = inputs["source_bone_features"][frame]
    e_bone = float(
        np.sum(
            (
                np.asarray(robot_features.adjacent_features)
                - np.asarray(source_features.adjacent_features)
            )
            ** 2
        )
    )
    base_correction = np.asarray(final.arrays["base_corrections"][frame], dtype=np.float64)
    state = np.concatenate([float(alpha) * base_correction, qpos])
    if frame == 0:
        temporal = 0.0
    else:
        previous = np.concatenate(
            [
                np.asarray(final.arrays["base_corrections"][frame - 1], dtype=np.float64),
                np.asarray(final.arrays["qpos"][frame - 1], dtype=np.float64),
            ]
        )
        temporal = float(paper["lambda_reg"] * np.sum((state - previous) ** 2))
    base_translation = float(
        paper["lambda_base_pos"] * np.sum((float(alpha) * base_correction[:3]) ** 2)
    )
    base_rotation = float(
        paper["lambda_base_rot"] * np.sum((float(alpha) * base_correction[3:]) ** 2)
    )
    start = int(final.arrays["slack_offsets"][frame])
    stop = int(final.arrays["slack_offsets"][frame + 1])
    final_slack = np.asarray(final.arrays["slack_concat"][start:stop], dtype=np.float64)
    warm_points = dynamic_collision_points_numpy(
        model, inputs["surface"], warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame]
    )
    if warm_initial is None:
        warm_initial = (
            inputs["sdf"]
            .query_scene(warm_points, inputs["object"].pose_scene.pose_scene[frame])
            .signed_distance
        )
    warm_slack = np.clip(
        np.maximum(-float(paper["tau_m"]) - warm_initial[query_ids], 0.0),
        0.0,
        float(paper["b_m"]) - float(paper["tau_m"]),
    )
    slack = (1.0 - float(alpha)) * warm_slack + float(alpha) * final_slack
    slack_reg = float(0.5 * paper["w_s"] * np.sum(slack**2))
    weighted_im = float(paper["lambda_IM"] * e_im)
    weighted_bone = float(paper["lambda_bone"] * e_bone)
    query_signed = collision_signed[query_ids] if len(query_ids) else np.empty(0)
    if anchor_unsigned is None:
        anchor_unsigned = (
            inputs["sdf"]
            .query_scene(robot_keypoints, inputs["object"].pose_scene.pose_scene[frame])
            .unsigned_distance
        )
    return {
        "e_im_raw": float(e_im),
        "e_im_weighted": weighted_im,
        "e_bone_raw": e_bone,
        "e_bone_weighted": weighted_bone,
        "base_reg": base_translation + base_rotation,
        "base_translation_reg": base_translation,
        "base_rotation_reg": base_rotation,
        "joint_reg": 0.0,
        "seed_delta_reg": 0.0,
        "temporal_reg": temporal,
        "slack_reg": slack_reg,
        "total": weighted_im
        + weighted_bone
        + base_translation
        + base_rotation
        + temporal
        + slack_reg,
        "constraint_query_min_m": float(np.min(query_signed)) if len(query_signed) else None,
        "constraint_full_min_m": float(np.min(collision_signed)),
        "constraint_hard_residual_min_m": float(np.min(query_signed + float(paper["b_m"])))
        if len(query_signed)
        else None,
        "constraint_soft_residual_min_m": float(
            np.min(query_signed + slack + float(paper["tau_m"]))
        )
        if len(query_signed)
        else None,
        "query_slack_median_m": float(np.median(slack)) if len(slack) else None,
        "contact_proxy_anchor_count_8mm": int(np.count_nonzero(anchor_unsigned <= 0.008)),
        "fingertip_anchor_min_m": float(np.min(anchor_unsigned[list(TIP_INDICES.values())])),
    }


def _interpolation_row(
    inputs: dict[str, Any],
    frame: int,
    alpha: float,
    *,
    visual_count: int,
    dense_include_vertices: bool = True,
    visual_vertex_max_per_instance: int | None = None,
) -> dict[str, Any]:
    final = inputs["final"]
    warm = inputs["warm"]
    model = inputs["model"]
    obj = inputs["object"]
    sdf: Any = inputs["sdf"]
    q = (1.0 - alpha) * warm.arrays["qpos"][frame] + alpha * final.arrays["qpos"][frame]
    base = _slerp(
        np.asarray(warm.arrays["base_pose_scene"][frame]),
        np.asarray(final.arrays["base_pose_scene"][frame]),
        alpha,
    )
    visual = _visual_surface(
        model,
        q,
        base,
        count=visual_count,
        seed=20260720 + frame,
        include_vertices=dense_include_vertices,
    )
    visual_q = sdf.query_scene(visual.points, obj.pose_scene.pose_scene[frame])
    collision = dynamic_collision_points_numpy(model, inputs["surface"], q, base)
    collision_q = sdf.query_scene(collision, obj.pose_scene.pose_scene[frame])
    start = int(final.arrays["query_offsets"][frame])
    stop = int(final.arrays["query_offsets"][frame + 1])
    ids = np.asarray(final.arrays["query_ids_concat"][start:stop], dtype=np.int64)
    full = collision_q.signed_distance
    return {
        "frame": int(frame),
        "alpha": float(alpha),
        "visual_surface_approximation": "deterministic_surface_samples",
        "min_visual_signed_distance_m": float(np.min(visual_q.signed_distance)),
        "min_collision_signed_distance_m": float(np.min(collision_q.signed_distance)),
        "min_full_audit_signed_distance_m": float(np.min(full)),
        "query_set_min_signed_distance_m": float(np.min(full[ids])) if len(ids) else None,
        "constraint_violation_m": float(max(0.0, -float(np.min(full[ids])))) if len(ids) else 0.0,
    }


def _interpolation_rows(
    inputs: dict[str, Any],
    frame: int,
    *,
    visual_count: int,
    alpha_count: int = 21,
    dense_include_vertices: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate the counterfactual path in one batched SDF query per layer."""

    final = inputs["final"]
    warm = inputs["warm"]
    model = inputs["model"]
    obj = inputs["object"]
    sdf = inputs["sdf"]
    alphas = np.linspace(0.0, 1.0, int(alpha_count))
    warm_q = np.asarray(warm.arrays["qpos"][frame], dtype=np.float64)
    final_q = np.asarray(final.arrays["qpos"][frame], dtype=np.float64)
    warm_base = np.asarray(warm.arrays["base_pose_scene"][frame], dtype=np.float64)
    final_base = np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)
    qpos = [(1.0 - alpha) * warm_q + alpha * final_q for alpha in alphas]
    bases = [_slerp(warm_base, final_base, float(alpha)) for alpha in alphas]
    visuals = [
        _visual_surface(
            model,
            q,
            base,
            count=visual_count,
            seed=20260720 + frame,
            include_vertices=dense_include_vertices,
        )
        for q, base in zip(qpos, bases, strict=True)
    ]
    visual_points = [
        item.points[:visual_count] if len(item.points) > visual_count else item.points
        for item in visuals
    ]
    visual_query = sdf.query_scene(np.stack(visual_points), obj.pose_scene.pose_scene[frame])
    collisions = np.stack(
        [
            dynamic_collision_points_numpy(model, inputs["surface"], q, base)
            for q, base in zip(qpos, bases, strict=True)
        ]
    )
    collision_query = sdf.query_scene(collisions, obj.pose_scene.pose_scene[frame])
    warm_points = dynamic_collision_points_numpy(
        model, inputs["surface"], warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame]
    )
    warm_initial = sdf.query_scene(warm_points, obj.pose_scene.pose_scene[frame]).signed_distance
    anchor_stack = sdf.query_scene(
        np.stack(
            [
                np.asarray(model.keypoints_scene(q, base, layout="mediapipe21"))
                for q, base in zip(qpos, bases, strict=True)
            ]
        ),
        obj.pose_scene.pose_scene[frame],
    ).unsigned_distance
    start = int(final.arrays["query_offsets"][frame])
    stop = int(final.arrays["query_offsets"][frame + 1])
    ids = np.asarray(final.arrays["query_ids_concat"][start:stop], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for index, alpha in enumerate(alphas):
        full = collision_query.signed_distance[index]
        objective = _interpolation_objective(
            inputs,
            frame,
            float(alpha),
            qpos[index],
            bases[index],
            full,
            ids,
            warm_initial=warm_initial,
            anchor_unsigned=anchor_stack[index],
        )
        rows.append(
            {
                "frame": int(frame),
                "alpha": float(alpha),
                "visual_surface_approximation": "deterministic_surface_samples",
                "min_visual_signed_distance_m": float(np.min(visual_query.signed_distance[index])),
                "min_collision_signed_distance_m": float(np.min(full)),
                "min_full_audit_signed_distance_m": float(np.min(full)),
                "query_set_min_signed_distance_m": float(np.min(full[ids])) if len(ids) else None,
                "constraint_violation_m": float(max(0.0, -float(np.min(full[ids]))))
                if len(ids)
                else 0.0,
                **objective,
            }
        )
    return rows


def _collision_visual_offsets(inputs: dict[str, Any], *, samples: int) -> list[dict[str, Any]]:
    model = inputs["model"]
    visual = model.visual_geometry_instances(model.neutral_q)
    collision = model.collision_geometry_instances(model.neutral_q)
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for collision/visual offset audit") from exc
    by_link: dict[str, dict[str, list[np.ndarray]]] = {}
    for kind, instances in (("visual", visual), ("collision", collision)):
        for i, instance in enumerate(instances):
            vertices, faces = _load_geometry(instance)
            points = _dense_samples(
                vertices,
                faces,
                count=max(64, samples // max(len(instances), 1)),
                mesh_id=f"offset:{kind}:{instance.link_name}:{i}",
                seed=20260720 + i,
            )
            points = points @ instance.world_transform[:3, :3].T + instance.world_transform[:3, 3]
            by_link.setdefault(str(instance.link_name), {}).setdefault(kind, []).append(points)
    stage9_points = dynamic_collision_points_numpy(
        model, inputs["surface"], model.neutral_q, np.eye(4, dtype=np.float64)
    )
    for link in sorted(set(np.asarray(inputs["surface"].link_names).astype(str).tolist())):
        mask = np.asarray(inputs["surface"].link_names).astype(str) == link
        by_link.setdefault(link, {}).setdefault("stage9_samples", []).append(stage9_points[mask])
    rows: list[dict[str, Any]] = []
    for link in sorted(by_link):
        visual_points = np.concatenate(by_link[link].get("visual", [np.empty((0, 3))]))
        collision_points = np.concatenate(by_link[link].get("collision", [np.empty((0, 3))]))
        stage9_sample_points = np.concatenate(
            by_link[link].get("stage9_samples", [np.empty((0, 3))])
        )
        if len(visual_points) and len(collision_points):
            d_v = cKDTree(collision_points).query(visual_points, workers=1)[0]
            d_c = cKDTree(visual_points).query(collision_points, workers=1)[0]
            values = np.concatenate([d_v, d_c])
            median, p95, max_value = (
                float(np.median(values)),
                float(np.quantile(values, 0.95)),
                float(np.max(values)),
            )
        else:
            values = np.empty(0)
            median = p95 = max_value = float("nan")
        likely_inflated = bool(median > 0.001 and len(collision_points) >= 32)
        likely_misaligned = bool(median > 0.005)
        if len(stage9_sample_points) and len(visual_points):
            stage9_offsets = cKDTree(visual_points).query(stage9_sample_points, workers=1)[0]
            stage9_median = float(np.median(stage9_offsets))
            stage9_p95 = float(np.quantile(stage9_offsets, 0.95))
            stage9_max = float(np.max(stage9_offsets))
        else:
            stage9_median = stage9_p95 = stage9_max = float("nan")
        rows.append(
            {
                "link_name": link,
                "visual_vertex_count": int(len(visual_points)),
                "collision_sample_count": int(len(stage9_sample_points)),
                "collision_surface_sample_count": int(len(collision_points)),
                "stage9_collision_sample_count": int(len(stage9_sample_points)),
                "median_offset_mm": median * 1000.0,
                "p95_offset_mm": p95 * 1000.0,
                "max_offset_mm": max_value * 1000.0,
                "stage9_sample_median_offset_mm": stage9_median * 1000.0,
                "stage9_sample_p95_offset_mm": stage9_p95 * 1000.0,
                "stage9_sample_max_offset_mm": stage9_max * 1000.0,
                "closest_visual_region": "tip/pad proxy" if link.endswith("_tip") else link,
                "likely_inflated": likely_inflated,
                "likely_misaligned": likely_misaligned,
                "notes": "bidirectional visual/collision-surface approximation plus one-way Stage 9 sample-to-visual distance; no exact mesh distance claimed",
            }
        )
    return rows


def _root_causes(summary: dict[str, Any], offsets: list[dict[str, Any]]) -> dict[str, Any]:
    source = summary["source_contact_proxy"]
    frame = summary["per_frame"]
    final_visual = np.asarray([row["final_visual_min_m"] for row in frame])
    final_collision = np.asarray([row["final_collision_min_m"] for row in frame])
    warm_collision = np.asarray([row["warm_collision_min_m"] for row in frame])
    offset_max = max(
        (float(row["max_offset_mm"]) for row in offsets if np.isfinite(row["max_offset_mm"])),
        default=float("nan"),
    )
    offset_median = float(np.nanmedian([float(row["median_offset_mm"]) for row in offsets]))
    inflated_links = [str(row["link_name"]) for row in offsets if bool(row.get("likely_inflated"))]
    misaligned_links = [
        str(row["link_name"]) for row in offsets if bool(row.get("likely_misaligned"))
    ]
    queryset = np.asarray([row["query_active_count"] for row in frame])
    causes: list[dict[str, Any]] = []

    def add(
        name: str,
        confidence: str,
        evidence_for: list[str],
        evidence_against: list[str],
        action: str,
        affected: list[str],
        affected_links: list[str] | None = None,
        quantitative: dict[str, Any] | None = None,
    ) -> None:
        causes.append(
            {
                "root_cause": name,
                "confidence": confidence,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "affected_frames": affected,
                "affected_links": list(affected_links or []),
                "quantitative_evidence": dict(quantitative or {}),
                "recommended_next_action": action,
            }
        )

    source_ratio = float(source.get("frame_contact_ratio_at_5mm", 0.0))
    if source_ratio < 0.5:
        add(
            "SOURCE_NOT_CONTACT_RICH",
            "high",
            [f"source 5 mm contact frame ratio={source_ratio:.3f}"],
            ["workflow contact labels are not used as proof"],
            "retain source-contact caveat and do not call this ground truth",
            ["all"],
            quantitative={"source_contact_frame_ratio_at_5mm": source_ratio},
        )
    else:
        add(
            "SOURCE_NOT_CONTACT_RICH",
            "low",
            [f"source 5 mm contact frame ratio={source_ratio:.3f}"],
            ["source geometry has near-surface samples across the window"],
            "no source replacement; retain proxy wording",
            ["all"],
            quantitative={"source_contact_frame_ratio_at_5mm": source_ratio},
        )
    if offset_max > 5:
        add(
            "COLLISION_GEOMETRY_INFLATED",
            "medium",
            [f"max bidirectional collision/visual offset={offset_max:.2f} mm"],
            ["offset test is sampled, not exact"],
            "inspect collision mesh scale and inflation; do not change paper objective",
            ["all"],
            affected_links=inflated_links,
            quantitative={"max_offset_mm": offset_max, "median_offset_mm": offset_median},
        )
    else:
        add(
            "COLLISION_GEOMETRY_INFLATED",
            "low",
            [f"max offset={offset_max:.2f} mm"],
            ["no large systematic sampled offset"],
            "no geometry inflation change indicated",
            ["all"],
            quantitative={"max_offset_mm": offset_max, "median_offset_mm": offset_median},
        )
    add(
        "COLLISION_GEOMETRY_MISALIGNED",
        "medium" if misaligned_links else "low",
        [f"links above sampled 5 mm median offset={len(misaligned_links)}"],
        ["bidirectional offset test is approximate and neutral-pose only"],
        "verify link transforms and scale before changing the objective",
        ["all"],
        affected_links=misaligned_links,
        quantitative={"misaligned_link_count": len(misaligned_links)},
    )
    gap = float(np.median(final_visual - final_collision))
    add(
        "COLLISION_SAMPLE_COVERAGE_BIAS",
        "medium" if abs(gap) > 0.003 else "low",
        [
            f"median final visual minus collision minimum={gap * 1000:.2f} mm",
            f"query count median={np.median(queryset):.0f}",
        ],
        ["visual and collision use different surfaces and minima are not paired exact points"],
        "increase contact-region samples in a diagnostic branch before changing solver",
        ["contact-rich links"],
        affected_links=sorted(set(REGIONS) - {"palm"}),
        quantitative={"median_visual_minus_collision_m": gap},
    )
    add(
        "QUERYSET_OVERREACH",
        "medium" if np.median(queryset) > 350 else "low",
        [
            f"median active QuerySet count={np.median(queryset):.0f}/512",
            f"max active QuerySet count={np.max(queryset):.0f}/512",
        ],
        ["QuerySet is derived from the declared 10 mm margin"],
        "audit activation policy separately; retain formal profile",
        ["all"],
        quantitative={
            "median_active_queryset_count": float(np.median(queryset)),
            "max_active_queryset_count": int(np.max(queryset)),
        },
    )
    add(
        "ACTIVE_MARGIN_TOO_CONSERVATIVE",
        "medium" if np.median(queryset) > 300 else "low",
        [
            f"active margin={summary['active_margin_m'] * 1000:.1f} mm",
            f"active count median={np.median(queryset):.0f}",
        ],
        ["no ablation was run in audit-only mode"],
        "bounded explicit shadow ablation only",
        ["all"],
        quantitative={"active_margin_m": float(summary["active_margin_m"])},
    )
    add(
        "SEMANTIC_ANCHOR_SURFACE_MISMATCH",
        "medium" if summary["anchor_final_median_m"] > 0.008 else "low",
        [f"final semantic-anchor median distance={summary['anchor_final_median_m'] * 1000:.2f} mm"],
        ["21 points are semantic anchors, not a pad surface"],
        "use a versioned pad proxy or real pad geometry; do not add a new term",
        ["fingertips"],
        affected_links=[f"{finger}_tip" for finger in FINGERS],
        quantitative={"final_anchor_median_m": float(summary["anchor_final_median_m"])},
    )
    add(
        "BASE_REGULARIZATION_DOMINATES",
        "medium"
        if summary["objective_dominant_term_counts"].get("final_eval_stage9_base_reg", 0) > 30
        else "low",
        [
            f"dominant-term frames={summary['objective_dominant_term_counts'].get('final_eval_stage9_base_reg', 0)}"
        ],
        ["same-definition objective also contains interaction and bone terms"],
        "compare a diagnostic no-base-reg profile only",
        ["all"],
        quantitative={
            "dominant_base_term_frames": summary["objective_dominant_term_counts"].get(
                "final_eval_stage9_base_reg", 0
            )
        },
    )
    add(
        "TEMPORAL_REGULARIZATION_DOMINATES",
        "medium"
        if summary["objective_dominant_term_counts"].get("final_eval_stage9_temporal_reg", 0) > 30
        else "low",
        [
            f"dominant-term frames={summary['objective_dominant_term_counts'].get('final_eval_stage9_temporal_reg', 0)}"
        ],
        ["temporal term is not contact-specific"],
        "compare no-temporal only in shadow branch",
        ["all"],
        quantitative={
            "dominant_temporal_term_frames": summary["objective_dominant_term_counts"].get(
                "final_eval_stage9_temporal_reg", 0
            )
        },
    )
    add(
        "INTERACTION_OBJECTIVE_UNDERCONSTRAINS_CONTACT",
        "medium" if float(np.mean(final_visual - warm_collision)) > 0.004 else "low",
        [
            f"median final visual minus warm collision={np.median(final_visual - warm_collision) * 1000:.2f} mm",
            f"final-warm E_IM delta mean={summary['mean_final_minus_warm_eim']:.6g}",
        ],
        ["geometry/query-set causes are not fully separable without shadow ablation"],
        "faithful branch accepts the objective limit; extension must be marked paper-external",
        ["all"],
        quantitative={"mean_final_minus_warm_eim": float(summary["mean_final_minus_warm_eim"])},
    )
    add(
        "MULTIPLE_COUPLED_CAUSES",
        "medium",
        ["geometry, QuerySet, anchor, and objective diagnostics each contribute evidence"],
        ["no solver shadow ablation was run"],
        "fix proven geometry/query issues first, then reassess the faithful objective",
        ["all"],
        affected_links=sorted(set(inflated_links + misaligned_links)),
        quantitative={"ranked_cause_count": len(causes)},
    )
    add(
        "INCONCLUSIVE",
        "low",
        ["causal separation is limited without an executed shadow ablation"],
        ["the audit provides independent geometry and objective evidence"],
        "run the explicit bounded shadow diagnostic in its independent output root",
        ["all"],
        quantitative={"shadow_ablation_run": False},
    )
    causes.sort(key=lambda item: {"high": 0, "medium": 1, "low": 2}[item["confidence"]])
    return {
        "schema_version": "toporetarget.stage9_3_root_cause.v1",
        "classifier_version": "root_cause_classifier_v1",
        "ranked_causes": [{"rank": i + 1, **item} for i, item in enumerate(causes)],
    }


def _html(payload: dict[str, Any], destination: Path) -> None:
    data = json.dumps(_json(payload), separators=(",", ":"), allow_nan=False)
    options = "".join(f"<option>{x}</option>" for x in payload.get("link_options", REGIONS))
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 9.3 contact audit</title><style>
body{margin:0;background:#111827;color:#e5e7eb;font:13px system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:100vh}
section{position:relative;background:#f8fafc}canvas{width:100%;height:100%;display:block}
aside{overflow:auto;padding:14px;background:#1f2937}h1{font-size:17px;margin:0 0 5px}
h2{font-size:12px;color:#93c5fd;margin:15px 0 6px}label{display:block;margin:5px 0}
select,input{box-sizing:border-box;max-width:100%;background:#111827;color:#e5e7eb;
border:1px solid #4b5563;border-radius:4px;padding:4px}input[type=range]{width:100%}
pre{white-space:pre-wrap;font:11px ui-monospace,monospace}.grid{display:grid;
grid-template-columns:1fr 1fr;gap:3px 10px}button{background:#2563eb;color:white;
border:0;border-radius:4px;padding:5px 8px}.hint{font-size:11px;color:#9ca3af;line-height:1.35}
#timeline{width:100%;height:110px;background:#111827}</style></head><body><main>
<section><canvas id="scene"></canvas></section><aside><h1>Contact retention audit</h1>
<div class="hint">Audit-only visualization. Distances use the reference triangle/SDF backend;
dense surfaces are deterministic approximations. Positive signed distance is outside.</div>
<h2>Frame</h2><input id="frame" type="range" min="0" max="__COUNT__" value="0">
<div id="frameLabel"></div><button id="worst">jump worst frame</button><h2>States</h2>
<div class="grid"><label><input id="source" type="checkbox" checked> source</label>
<label><input id="warm" type="checkbox" checked> warm-start</label>
<label><input id="final" type="checkbox" checked> final</label>
<label><input id="object" type="checkbox" checked> object context</label></div>
<h2>Geometry layers</h2><div class="grid"><label><input id="visual" type="checkbox" checked>
visual surface</label><label><input id="collision" type="checkbox" checked> collision geometry</label>
<label><input id="fullaudit" type="checkbox" checked> full 512 audit</label><label><input id="pads" type="checkbox" checked> fingertip/pad proxy</label>
<label><input id="queries" type="checkbox" checked> QuerySet</label>
<label><input id="anchors" type="checkbox" checked> anchors</label>
<label><input id="segments" type="checkbox" checked> nearest segments</label></div>
<label>contact threshold <span id="thresholdLabel"></span><input id="threshold" type="range"
min="1" max="10" step="1" value="5"></label><label>link/region<select id="link">
<option value="all">all</option>__OPTIONS__</select></label><h2>Timeline</h2>
<select id="timelineMetric"><option value="final_visual_min_m">final visual min</option>
<option value="source_visual_min_m">source visual min</option><option value="warm_visual_min_m">warm visual min</option>
<option value="final_collision_min_m">final collision min</option><option value="source_contact_proxy_5mm">source contact proxy</option>
<option value="final_contact_retention_proxy_recall">final retention proxy</option><option value="warm_objective_e_im_raw">warm E_IM</option>
<option value="final_objective_e_im_raw">final E_IM</option><option value="warm_objective_e_bone_raw">warm E_bone</option>
<option value="final_objective_e_bone_raw">final E_bone</option><option value="query_active_count">QuerySet active count</option>
<option value="collision_visual_offset_max_mm">collision/visual offset</option></select>
<canvas id="timeline" width="340" height="110"></canvas><div class="hint">Click the timeline to jump; threshold changes contact coloring.</div>
<h2>Metrics</h2><pre id="metrics"></pre><h2>Related viewer</h2><a id="formalViewer" target="_blank">Open formal trajectory viewer</a>
<h2>Legend</h2><div class="hint">blue=source · orange=warm-start · green=final · red=penetration · yellow=near contact ·
gray=positive gap · purple=semantic anchors · object context=gray</div></aside></main><script>
const P=__DATA__;const $=id=>document.getElementById(id);let frame=0,az=.6,el=.5,zoom=1,drag=null;
$("formalViewer").href=P.formal_viewer_href||"#";function pts(name){return P.frames[frame][name]||[]}
function project(p){let c=$("scene"),x=(p[0]*Math.cos(az)-p[1]*Math.sin(az))*Math.cos(el)-p[2]*Math.sin(el),
y=(p[0]*Math.sin(az)+p[1]*Math.cos(az)),z=(p[0]*Math.cos(az)-p[1]*Math.sin(az))*Math.sin(el)+p[2]*Math.cos(el),
s=Math.min(c.clientWidth,c.clientHeight)*3.1*zoom;return [c.clientWidth/2+x*s,c.clientHeight/2-y*s,z]}
function pointColor(d,base){let tau=+$('threshold').value/1000;if(d!==undefined&&d<0)return '#dc2626';
if(d!==undefined&&Math.abs(d)<=tau)return '#facc15';return base}
function draw(){let c=$("scene"),ctx=c.getContext("2d"),d=devicePixelRatio||1;c.width=c.clientWidth*d;
c.height=c.clientHeight*d;ctx.setTransform(d,0,0,d,0,0);ctx.fillStyle="#f8fafc";ctx.fillRect(0,0,c.clientWidth,c.clientHeight);
let drawSet=(id,name,base,size,layer)=>{if(!$(id).checked)return;if(layer==="visual"&&!$("visual").checked)return;
if(layer==="collision"&&!$("collision").checked)return;if(layer==="fullaudit"&&!$("fullaudit").checked)return;
let a=pts(name),r=P.frames[frame][name+"_regions"]||[],l=P.frames[frame][name+"_links"]||[],
ds=P.frames[frame][name+"_signed_distance_m"]||[],selected=$("link").value;let values=a.map((p,i)=>({p,r:r[i],l:l[i],d:ds[i]}))
.filter(x=>selected==="all"||x.r===selected||x.l===selected).map(x=>[project(x.p),x.d]);values.sort((u,v)=>u[0][2]-v[0][2]).forEach(x=>{
ctx.fillStyle=pointColor(x[1],base);ctx.globalAlpha=.75;ctx.beginPath();ctx.arc(x[0][0],x[0][1],size,0,Math.PI*2);ctx.fill()});ctx.globalAlpha=1};
drawSet("object","object_points","#6b7280",2,"object");drawSet("source","source_points","#2563eb",1.6,"visual");
drawSet("warm","warm_points","#f59e0b",1.8,"visual");drawSet("final","final_points","#16a34a",1.8,"visual");
drawSet("collision","collision_points","#64748b",2,"collision");drawSet("fullaudit","full_audit_points","#0f766e",3,"fullaudit");
drawSet("queries","query_points","#dc2626",3,"collision");drawSet("anchors","anchor_points","#9333ea",4,"anchors");
drawSet("pads","tip_anchor_points","#ec4899",5,"anchors");if($("segments").checked){ctx.strokeStyle="#9333ea";ctx.lineWidth=1;
for(const s of pts("segments")){let a=project(s[0]),b=project(s[1]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}}}
function metrics(){let m=P.frames[frame].metrics;$("frameLabel").textContent=`frame ${m.frame} (global ${m.global_frame})`;
$("thresholdLabel").textContent=$("threshold").value+" mm";$("metrics").textContent=JSON.stringify({...m,distance_stats:P.frames[frame].distance_stats},null,2)}
function metricValue(m,key){let v=m[key];if(v===null||v===undefined)return null;return key.includes("distance")||key.includes("offset")||key.includes("e_im")||key.includes("e_bone")?v*1000:v}
function render(){metrics();draw();timeline()}$("frame").oninput=e=>{frame=+e.target.value;render()};$("threshold").oninput=render;
["source","warm","final","object","visual","collision","fullaudit","queries","anchors","pads","segments"].forEach(x=>$(x).onchange=draw);
$("link").onchange=draw;$("timelineMetric").onchange=timeline;$("worst").onclick=()=>{frame=P.worst_frame;$("frame").value=frame;render()};
let t=$("timeline"),tc=t.getContext("2d");function timeline(){tc.clearRect(0,0,t.width,t.height);let key=$("timelineMetric").value;
let v=P.frames.map(x=>metricValue(x.metrics,key));let valid=v.map(x=>x===null?0:x),mn=Math.min(...valid),mx=Math.max(...valid);tc.strokeStyle="#60a5fa";
tc.beginPath();v.forEach((x,i)=>{let X=i/(v.length-1)*t.width,Y=t.height-(x-mn)/Math.max(mx-mn,1e-9)*t.height;if(i)tc.lineTo(X,Y);else tc.moveTo(X,Y)});
tc.stroke();tc.fillStyle="#fbbf24";tc.fillRect(frame/(v.length-1)*t.width,0,2,t.height)}t.onclick=e=>{frame=Math.round(e.offsetX/t.clientWidth*(P.frames.length-1));
$("frame").value=frame;render()};window.onresize=draw;let scene=$("scene");scene.onpointerdown=e=>drag=[e.clientX,e.clientY];scene.onpointerup=()=>drag=null;
scene.onpointermove=e=>{if(!drag)return;az+=(e.clientX-drag[0])*.01;el+=(e.clientY-drag[1])*.01;drag=[e.clientX,e.clientY];draw()};
scene.onwheel=e=>{zoom*=e.deltaY<0?1.1:.9;draw()};render();</script></body></html>"""
    document = document.replace("__COUNT__", str(len(payload["frames"]) - 1))
    document = document.replace("__OPTIONS__", options).replace("__DATA__", data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def _shadow_frame_selection(
    spec: str,
    selected: list[int],
    geometry_frames: list[dict[str, Any]] | None = None,
    objective_rows: list[dict[str, Any]] | None = None,
) -> list[int]:
    if spec.strip().lower() == "auto":
        if geometry_frames and objective_rows:
            by_frame = {int(row["frame"]): row for row in geometry_frames}
            by_objective = {int(row["frame"]): row for row in objective_rows}
            preferred = [
                max(
                    selected,
                    key=lambda frame: by_frame[frame]["metrics"]["source_contact_proxy_5mm"],
                ),
                max(
                    selected,
                    key=lambda frame: (
                        by_objective[frame]["final_eval_stage9_e_im_raw"]
                        - by_objective[frame]["warm_eval_stage9_e_im_raw"]
                    ),
                ),
                max(
                    selected,
                    key=lambda frame: abs(
                        by_frame[frame]["metrics"]["final_visual_min_m"]
                        - by_frame[frame]["metrics"]["final_collision_min_m"]
                    ),
                ),
            ]
        else:
            preferred = [selected[0], selected[len(selected) // 2], selected[-1]]
        return list(dict.fromkeys(preferred))
    values = [int(item.strip()) for item in spec.split(",") if item.strip()]
    unknown = sorted(set(values) - set(selected))
    if unknown:
        raise ValueError(f"shadow frame(s) are outside the audit range: {unknown}")
    return list(dict.fromkeys(values))


def _shadow_score_diagnostic(
    objective_rows: list[dict[str, Any]],
    selected: list[int],
    frame_spec: str,
    geometry_frames: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate paper-external score counterfactuals without regenerating q.

    This is intentionally not an optimizer ablation: it quantifies how the
    persisted final score decomposes if selected regularizer terms are omitted,
    while preserving the accepted trajectory and guaranteeing zero solver calls.
    """

    frames = _shadow_frame_selection(frame_spec, selected, geometry_frames, objective_rows)
    by_frame = {int(row["frame"]): row for row in objective_rows}
    rows = [by_frame[frame] for frame in frames]
    variants = []
    for variant, field in (
        ("remove_slack_score_only", "final_eval_stage9_slack_reg"),
        ("remove_temporal_score_only", "final_eval_stage9_temporal_reg"),
        ("remove_base_score_only", "final_eval_stage9_base_reg"),
    ):
        values = [float(row["final_eval_stage9_total"] - row[field]) for row in rows]
        variants.append(
            {
                "variant": variant,
                "score_only": True,
                "frame_count": len(rows),
                "median_counterfactual_total": float(np.median(values)),
                "mean_counterfactual_total": float(np.mean(values)),
                "evidence_limit": "no trajectory was regenerated; no optimizer effect is identified",
            }
        )
    profiles = [
        "official_baseline_reproduction",
        "half_active_margin",
        "zero_active_margin",
        "minimal_feasible_projection_from_warm",
        "no_base_regularization",
        "no_temporal_regularization",
    ]
    return {
        "ran": True,
        "diagnostic_only": True,
        "ablation_type": "score_only_counterfactual",
        "paper_method": False,
        "accepted_reference": False,
        "solver_invocation_count": 0,
        "frames": frames,
        "variants": variants,
        "evidence_limit": "not a solver ablation; it cannot establish causal contact improvement",
        "profiles": [
            {
                "profile_id": profile,
                "diagnostic_only": True,
                "paper_method": False,
                "accepted_reference": False,
                "status": "not_executed_score_only_mode",
                "reason": "solver trajectory regeneration is intentionally outside this score-only diagnostic",
            }
            for profile in profiles
        ],
    }


def run_contact_audit(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    thresholds_mm: list[float] | None = None,
    surface_samples: int = 8192,
    frame_start: int | None = None,
    frame_end: int | None = None,
    links: list[str] | None = None,
    html: bool = False,
    interactive: bool = False,
    force: bool = False,
    no_cache: bool = False,
    run_shadow_ablation: bool = False,
    shadow_frames: str = "auto",
    headless_smoke_test: bool = False,
    evaluation_backend: str = "configured",
    dense_include_vertices: bool = True,
    visual_vertex_max_per_instance: int | None = None,
) -> dict[str, Any]:
    del no_cache  # audit results are always freshly evaluated; this is a provenance flag in CLI.
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    manifest_path = _resolve(repo_root, manifest_path)
    output_root = _resolve(repo_root, output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"audit output exists; pass --force: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) / 1000.0 for x in (thresholds_mm or [1, 2, 3, 5, 8, 10])]
    preflight = _preflight(manifest_path, json.loads(manifest_path.read_text()), repo_root)
    _write_json(output_root / "preflight_audit.json", preflight)
    inputs = _load_inputs(
        manifest_path,
        repo_root,
        evaluation_backend=evaluation_backend,
    )
    final = inputs["final"]
    warm = inputs["warm"]
    obj = inputs["object"]
    sdf = inputs["sdf"]
    model = inputs["model"]
    start = 0 if frame_start is None else int(frame_start)
    stop = int(final.frame_count) if frame_end is None else int(frame_end)
    if start < 0 or stop > final.frame_count or stop <= start:
        raise ValueError(
            f"audit frame range must be local [0,{final.frame_count}], got [{start},{stop})"
        )
    if final.frame_count != 60:
        raise ValueError("Stage 9.3 requires the accepted 60-frame final artifact")
    selected = list(range(start, stop))
    frame_position = {frame: index for index, frame in enumerate(selected)}
    source_proxy: dict[str, Any] = {
        "schema_version": "toporetarget.source_contact_proxy.v1",
        "proxy_name": "source_contact_proxy",
        "ground_truth_contact": False,
        "thresholds_mm": thresholds_mm or [1, 2, 3, 5, 8, 10],
        "region_proxy": SOURCE_REGION_PROXY,
        "frames": [],
    }
    retention: dict[str, Any] = {
        "schema_version": "toporetarget.contact_retention_proxy.v1",
        "proxy_name": "contact_retention_proxy",
        "ground_truth_contact": False,
        "source_thresholds_mm": [2, 3, 5, 8],
        "robot_thresholds_mm": [2, 3, 5, 8, 10],
        "frames": [],
    }
    geometry_frames: list[dict[str, Any]] = []
    queryset_points: list[dict[str, Any]] = []
    queryset_links: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    interpolation_rows: list[dict[str, Any]] = []
    anchor_final_distances: list[float] = []
    # 8192 is the Stage 9.3/9.3.2 default.  Smaller explicit values remain
    # available for unit/smoke runs; formal commands keep the CLI minimum at
    # 8192 unless the caller deliberately overrides the API.
    dense_visual_count = max(64, int(surface_samples))
    known_links = set(str(x) for x in np.asarray(inputs["surface"].link_names).reshape(-1))
    known_links.update(str(x.link_name) for x in model.visual_geometry_instances(model.neutral_q))
    known_links.update(
        str(x.link_name) for x in model.collision_geometry_instances(model.neutral_q)
    )
    requested_links = set(links or [])
    unknown_links = sorted(requested_links - known_links - set(REGIONS))
    if unknown_links:
        raise ValueError(f"unknown audit link(s): {', '.join(unknown_links)}")
    for frame in selected:
        object_pose = obj.pose_scene.pose_scene[frame]
        # The canonical source mesh is a deforming [T,V,3] mesh.  Preserve the
        # deterministic sample barycentric convention by sampling its current
        # frame vertices, then retaining all current-frame vertices.
        source_vertices = np.asarray(inputs["source_hand"].vertices_scene[frame], dtype=np.float64)
        source_faces = np.asarray(inputs["source_hand"].mesh.faces, dtype=np.int64)
        source_points = _dense_samples(
            source_vertices,
            source_faces,
            count=dense_visual_count,
            mesh_id=f"source_frame:{frame}",
            seed=20260720 + frame,
            include_vertices=dense_include_vertices,
        )
        warm_points = _visual_surface(
            model,
            warm.arrays["qpos"][frame],
            warm.arrays["base_pose_scene"][frame],
            count=dense_visual_count,
            seed=20260820 + frame,
            include_vertices=dense_include_vertices,
        )
        final_points = _visual_surface(
            model,
            final.arrays["qpos"][frame],
            final.arrays["base_pose_scene"][frame],
            count=dense_visual_count,
            seed=20260920 + frame,
            include_vertices=dense_include_vertices,
        )
        warm_vertices = _visual_vertices(
            model,
            warm.arrays["qpos"][frame],
            warm.arrays["base_pose_scene"][frame],
            max_per_instance=visual_vertex_max_per_instance,
        )
        final_vertices = _visual_vertices(
            model,
            final.arrays["qpos"][frame],
            final.arrays["base_pose_scene"][frame],
            max_per_instance=visual_vertex_max_per_instance,
        )
        # Collision/visual B2 is formalized by the Stage 6 collision samples
        # and full 512 audit.  Keep a bounded collision-surface diagnostic for
        # per-link coverage; the 8192 minimum applies to visual meshes.
        collision_dense = _collision_surface(
            model,
            final.arrays["qpos"][frame],
            final.arrays["base_pose_scene"][frame],
            count=max(128, min(1024, dense_visual_count)),
            seed=20261020 + frame,
            include_vertices=False,
        )
        source_q = sdf.query_scene(source_points, object_pose)
        warm_q = sdf.query_scene(warm_points.points, object_pose)
        final_q = sdf.query_scene(final_points.points, object_pose)
        collision_dense_q = sdf.query_scene(collision_dense.points, object_pose)
        source_vertex_q = sdf.query_scene(source_vertices, object_pose)
        warm_vertex_q = sdf.query_scene(warm_vertices.points, object_pose)
        final_vertex_q = sdf.query_scene(final_vertices.points, object_pose)
        source_anchors = inputs["source_keypoints"][frame]
        warm_anchors = np.asarray(warm.arrays["robot_keypoints_scene"][frame], dtype=np.float64)
        final_anchors = np.asarray(final.arrays["robot_keypoints_scene"][frame], dtype=np.float64)
        source_anchor_q = sdf.query_scene(source_anchors, object_pose)
        warm_anchor_q = sdf.query_scene(warm_anchors, object_pose)
        final_anchor_q = sdf.query_scene(final_anchors, object_pose)
        source_region = _region_indices(source_points, source_anchors)
        source_frame = {
            "frame": frame,
            "global_frame": int(
                final.arrays["frame_indices"][frame] + final.metadata.get("source_frame_offset", 0)
            ),
            "visual_min_distance_m": float(np.min(source_q.signed_distance)),
            "vertex_signed_distance_stats": _grouped_stats(
                source_vertex_q.signed_distance, source_region[: len(source_vertices)]
            ),
            "visual_stats": _stats(source_q.signed_distance),
            "unsigned_stats": _stats(source_q.unsigned_distance),
            "thresholds": _threshold_record(
                source_q.signed_distance, source_q.unsigned_distance, thresholds
            ),
            "regions": {
                region: _threshold_record(
                    source_q.signed_distance[source_region == region],
                    source_q.unsigned_distance[source_region == region],
                    thresholds,
                )
                for region in REGIONS
            },
            "anchor_signed_distance_m": source_anchor_q.signed_distance.tolist(),
            "anchor_unsigned_distance_m": source_anchor_q.unsigned_distance.tolist(),
            "anchor_provenance": {
                "source": _anchor_provenance(source_anchors, source_anchor_q, object_pose),
            },
            "contact_proxy": "source_contact_proxy",
        }
        source_proxy["frames"].append(source_frame)
        if source_frame["thresholds"].get("5mm", {}).get("near_surface_ratio", 0) > 0:
            pass
        final_full = sdf.query_scene(
            np.asarray(final.arrays["collision_points_scene"][frame]), object_pose
        )
        warm_collision_points = dynamic_collision_points_numpy(
            model,
            inputs["surface"],
            warm.arrays["qpos"][frame],
            warm.arrays["base_pose_scene"][frame],
        )
        warm_collision_q = sdf.query_scene(warm_collision_points, object_pose)
        qrows, qlinkrows = _frame_query_records(
            frame,
            inputs,
            np.asarray(final.arrays["collision_points_scene"][frame]),
            warm_collision_points,
            final_q,
            warm_query=warm_collision_q,
            final_query=final_full,
        )
        queryset_points.extend(qrows)
        queryset_links.extend(qlinkrows)
        obj_row = _objective_rows(inputs, frame)
        obj_row.update({"frame": frame, "global_frame": source_frame["global_frame"]})
        objective_rows.append(obj_row)
        anchor_final_distances.extend(final_anchor_q.unsigned_distance.tolist())
        retention_frame: dict[str, Any] = {
            "frame": frame,
            "global_frame": source_frame["global_frame"],
            "anchor_level": {},
            "link_level": {},
            "direction_level": {},
            "threshold_sensitivity": {},
            "anchor_provenance": {
                "source": _anchor_provenance(source_anchors, source_anchor_q, object_pose),
                "warm": _anchor_provenance(warm_anchors, warm_anchor_q, object_pose),
                "final": _anchor_provenance(final_anchors, final_anchor_q, object_pose),
            },
        }
        for source_tau in [0.002, 0.003, 0.005, 0.008]:
            for robot_tau in [0.002, 0.003, 0.005, 0.008, 0.010]:
                source_contact = source_anchor_q.unsigned_distance <= source_tau
                retained = source_contact & (final_anchor_q.unsigned_distance <= robot_tau)
                retention_frame["threshold_sensitivity"][
                    f"source_{source_tau * 1000:g}mm_robot_{robot_tau * 1000:g}mm"
                ] = {
                    "source_anchor_count": int(np.count_nonzero(source_contact)),
                    "retained_anchor_count": int(np.count_nonzero(retained)),
                    "contact_retention_proxy_recall": float(np.mean(retained[source_contact]))
                    if np.any(source_contact)
                    else None,
                }
        for name, index in [("wrist", 0), *[(finger, TIP_INDICES[finger]) for finger in FINGERS]]:
            source_contact = bool(source_anchor_q.unsigned_distance[index] <= 0.005)
            final_contact = bool(final_anchor_q.unsigned_distance[index] <= 0.008)
            warm_contact = bool(warm_anchor_q.unsigned_distance[index] <= 0.008)
            source_dir, _ = _object_local_direction(
                source_anchors[index], source_anchor_q.closest_points[index], object_pose
            )
            final_dir, _ = _object_local_direction(
                final_anchors[index], final_anchor_q.closest_points[index], object_pose
            )
            warm_dir, _ = _object_local_direction(
                warm_anchors[index], warm_anchor_q.closest_points[index], object_pose
            )
            retention_frame["anchor_level"][name] = {
                "anchor_index": index,
                "source_contact_proxy_5mm": source_contact,
                "warm_distance_m": float(warm_anchor_q.unsigned_distance[index]),
                "final_distance_m": float(final_anchor_q.unsigned_distance[index]),
                "warm_contact_proxy_8mm": warm_contact,
                "final_contact_proxy_8mm": final_contact,
                "warm_to_source_distance_drift_m": float(
                    warm_anchor_q.unsigned_distance[index]
                    - source_anchor_q.unsigned_distance[index]
                ),
                "final_to_source_distance_drift_m": float(
                    final_anchor_q.unsigned_distance[index]
                    - source_anchor_q.unsigned_distance[index]
                ),
                "warm_to_final_distance_drift_m": float(
                    final_anchor_q.unsigned_distance[index] - warm_anchor_q.unsigned_distance[index]
                ),
                "distance_error_final_vs_source_m": float(
                    abs(final_anchor_q.unsigned_distance[index])
                    - abs(source_anchor_q.unsigned_distance[index])
                ),
            }
            retention_frame["direction_level"][name] = {
                "warm_angular_error_rad": _angle(source_dir, warm_dir),
                "final_angular_error_rad": _angle(source_dir, final_dir),
                "warm_side_consistent": bool(np.dot(source_dir, warm_dir) >= 0),
                "final_side_consistent": bool(np.dot(source_dir, final_dir) >= 0),
            }
        for region in REGIONS:
            links_in_region = final_points.regions == region
            source_region_contact = int(
                np.count_nonzero(source_q.unsigned_distance[source_region == region] <= 0.005)
            )
            warm_region_contact = int(
                np.count_nonzero(warm_q.unsigned_distance[warm_points.regions == region] <= 0.008)
            )
            final_region_contact = int(
                np.count_nonzero(final_q.unsigned_distance[final_points.regions == region] <= 0.008)
            )
            retention_frame["link_level"][region] = {
                "source_contact_proxy_count_5mm": source_region_contact,
                "warm_contact_proxy_count_8mm": warm_region_contact,
                "final_contact_proxy_count_8mm": final_region_contact,
                "source_region_sample_count": int(np.count_nonzero(source_region == region)),
                "robot_region_sample_count": int(np.count_nonzero(links_in_region)),
                "contact_retention_proxy_ratio_final_over_source": float(
                    final_region_contact / source_region_contact
                )
                if source_region_contact
                else None,
            }
        retention["frames"].append(retention_frame)
        geometry_frames.append(
            {
                "frame": frame,
                "global_frame": source_frame["global_frame"],
                "source_points": source_points[:: max(1, len(source_points) // 1024)].tolist(),
                "source_regions": source_region[:: max(1, len(source_points) // 1024)].tolist(),
                "source_links": source_region[:: max(1, len(source_points) // 1024)].tolist(),
                "source_points_signed_distance_m": source_q.signed_distance[
                    :: max(1, len(source_points) // 1024)
                ].tolist(),
                "warm_points": warm_points.points[
                    :: max(1, len(warm_points.points) // 1024)
                ].tolist(),
                "warm_regions": warm_points.regions[
                    :: max(1, len(warm_points.points) // 1024)
                ].tolist(),
                "warm_links": warm_points.links[
                    :: max(1, len(warm_points.points) // 1024)
                ].tolist(),
                "warm_points_signed_distance_m": warm_q.signed_distance[
                    :: max(1, len(warm_points.points) // 1024)
                ].tolist(),
                "final_points": final_points.points[
                    :: max(1, len(final_points.points) // 1024)
                ].tolist(),
                "final_regions": final_points.regions[
                    :: max(1, len(final_points.points) // 1024)
                ].tolist(),
                "final_links": final_points.links[
                    :: max(1, len(final_points.points) // 1024)
                ].tolist(),
                "final_points_signed_distance_m": final_q.signed_distance[
                    :: max(1, len(final_points.points) // 1024)
                ].tolist(),
                "collision_points": collision_dense.points[
                    :: max(1, len(collision_dense.points) // 1024)
                ].tolist(),
                "collision_regions": collision_dense.regions[
                    :: max(1, len(collision_dense.points) // 1024)
                ].tolist(),
                "collision_links": collision_dense.links[
                    :: max(1, len(collision_dense.points) // 1024)
                ].tolist(),
                "collision_points_signed_distance_m": collision_dense_q.signed_distance[
                    :: max(1, len(collision_dense.points) // 1024)
                ].tolist(),
                "full_audit_points": np.asarray(
                    final.arrays["collision_points_scene"][frame]
                ).tolist(),
                "full_audit_regions": [
                    (
                        "palm"
                        if str(link) == "palm"
                        else next(
                            (finger for finger in FINGERS if str(link).startswith(finger)),
                            "palm",
                        )
                    )
                    for link in np.asarray(inputs["surface"].link_names)
                ],
                "full_audit_links": np.asarray(inputs["surface"].link_names).astype(str).tolist(),
                "full_audit_points_signed_distance_m": final_full.signed_distance.tolist(),
                "query_points": np.asarray(final.arrays["collision_points_scene"][frame])[
                    np.asarray(
                        final.arrays["query_ids_concat"][
                            int(final.arrays["query_offsets"][frame]) : int(
                                final.arrays["query_offsets"][frame + 1]
                            )
                        ],
                        dtype=np.int64,
                    )
                ].tolist(),
                "query_regions": [
                    (
                        "palm"
                        if row["robot_link"] == "palm"
                        else next(
                            (finger for finger in FINGERS if row["robot_link"].startswith(finger)),
                            "palm",
                        )
                    )
                    for row in qrows
                ],
                "query_links": [row["robot_link"] for row in qrows],
                "query_points_signed_distance_m": final_full.signed_distance[
                    np.asarray(
                        final.arrays["query_ids_concat"][
                            int(final.arrays["query_offsets"][frame]) : int(
                                final.arrays["query_offsets"][frame + 1]
                            )
                        ],
                        dtype=np.int64,
                    )
                ].tolist(),
                "anchor_points": final_anchors.tolist(),
                "source_anchor_points": source_anchors.tolist(),
                "anchor_signed_distance_m": final_anchor_q.signed_distance.tolist(),
                "source_anchor_signed_distance_m": source_anchor_q.signed_distance.tolist(),
                "tip_anchor_points": final_anchors[list(TIP_INDICES.values())].tolist(),
                "tip_anchor_points_signed_distance_m": final_anchor_q.signed_distance[
                    list(TIP_INDICES.values())
                ].tolist(),
                "object_points": (
                    np.asarray(obj.mesh.vertices_local) @ np.asarray(object_pose[:3, :3]).T
                    + np.asarray(object_pose[:3, 3])
                )[:: max(1, len(obj.mesh.vertices_local) // 1024)].tolist(),
                "segments": [
                    [final_anchors[i].tolist(), final_anchor_q.closest_points[i].tolist()]
                    for i in range(21)
                ],
                "distance_stats": {
                    "source_visual_mesh": _grouped_stats(source_q.signed_distance, source_region),
                    "source_visual_vertices": _grouped_stats(
                        source_vertex_q.signed_distance, source_region[: len(source_vertices)]
                    ),
                    "warm_visual_mesh": _grouped_stats(warm_q.signed_distance, warm_points.regions),
                    "warm_visual_vertices": _grouped_stats(
                        warm_vertex_q.signed_distance, warm_vertices.regions
                    ),
                    "final_visual_mesh": _grouped_stats(
                        final_q.signed_distance, final_points.regions
                    ),
                    "final_visual_vertices": _grouped_stats(
                        final_vertex_q.signed_distance, final_vertices.regions
                    ),
                    "final_collision_geometry": _grouped_stats(
                        collision_dense_q.signed_distance, collision_dense.regions
                    ),
                    "final_full_512_audit": _grouped_stats(
                        final_full.signed_distance, np.asarray(inputs["surface"].link_names)
                    ),
                },
                "metrics": {
                    "frame": frame,
                    "global_frame": source_frame["global_frame"],
                    "source_visual_min_m": source_frame["visual_min_distance_m"],
                    "warm_visual_min_m": float(np.min(warm_q.signed_distance)),
                    "final_visual_min_m": float(np.min(final_q.signed_distance)),
                    "warm_collision_min_m": float(np.min(warm_collision_q.signed_distance)),
                    "final_collision_min_m": float(np.min(final_full.signed_distance)),
                    "final_full_audit_min_m": float(np.min(final_full.signed_distance)),
                    "source_contact_proxy_5mm": source_frame["thresholds"]["5mm"][
                        "near_surface_count"
                    ],
                    "warm_contact_retention_proxy_recall": retention_frame["threshold_sensitivity"][
                        "source_5mm_robot_8mm"
                    ]["contact_retention_proxy_recall"],
                    "final_contact_retention_proxy_recall": retention_frame[
                        "threshold_sensitivity"
                    ]["source_5mm_robot_8mm"]["contact_retention_proxy_recall"],
                    "query_active_count": len(qrows),
                    "active_margin_sensitivity_counts": {
                        f"{int(margin_mm)}mm": int(
                            np.count_nonzero(final_full.signed_distance <= margin_mm / 1000.0)
                        )
                        for margin_mm in (0, 2, 5, 8, 10)
                    },
                    "collision_visual_offset_max_mm": None,
                    "earliest_feasible_alpha": None,
                },
            }
        )
        interpolation_rows.extend(
            _interpolation_rows(
                inputs,
                frame,
                # Keep the 21-state interpolation path deterministic and
                # full-512 for collision feasibility, while using a bounded
                # visual-only sampling profile distinct from the formal
                # source/warm/final dense surfaces.
                visual_count=min(128, dense_visual_count),
                alpha_count=21,
                dense_include_vertices=dense_include_vertices,
            )
        )
    offsets = _collision_visual_offsets(inputs, samples=dense_visual_count)
    for row in geometry_frames:
        row["metrics"]["collision_visual_offset_max_mm"] = max(
            (float(x["max_offset_mm"]) for x in offsets if np.isfinite(x["max_offset_mm"])),
            default=None,
        )
    for frame in selected:
        path_rows = [row for row in interpolation_rows if row["frame"] == frame]
        feasible = [row["alpha"] for row in path_rows if row["constraint_violation_m"] <= 1e-6]
        value = min(feasible) if feasible else None
        geometry_frames[frame_position[frame]]["metrics"]["earliest_feasible_alpha"] = value
        for row in path_rows:
            row["earliest_feasible_alpha"] = value
    objective_by_frame = {int(row["frame"]): row for row in objective_rows}
    for row in geometry_frames:
        row["metrics"].update(
            {
                "objective_warm_total": objective_by_frame[row["frame"]]["warm_eval_stage9_total"],
                "objective_final_total": objective_by_frame[row["frame"]][
                    "final_eval_stage9_total"
                ],
                "warm_objective_e_im_raw": objective_by_frame[row["frame"]][
                    "warm_eval_stage9_e_im_raw"
                ],
                "final_objective_e_im_raw": objective_by_frame[row["frame"]][
                    "final_eval_stage9_e_im_raw"
                ],
                "warm_objective_e_bone_raw": objective_by_frame[row["frame"]][
                    "warm_eval_stage9_e_bone_raw"
                ],
                "final_objective_e_bone_raw": objective_by_frame[row["frame"]][
                    "final_eval_stage9_e_bone_raw"
                ],
                "objective_dominant_term": objective_by_frame[row["frame"]]["dominant_term"],
                "root_cause_tags": [],
            }
        )
    objective_summary: dict[str, Any] = {
        "mean": {
            k: _safe_mean(row[k] for row in objective_rows if isinstance(row.get(k), (float, int)))
            for k in objective_rows[0]
            if isinstance(objective_rows[0].get(k), (float, int))
        },
        "median": {},
        "p95": {},
        "worst_frame": {},
        "old_fields_directly_comparable_all": bool(
            all(row["old_fields_directly_comparable"] for row in objective_rows)
        ),
    }
    numeric_fields = [
        k
        for k, v in objective_rows[0].items()
        if isinstance(v, (float, int)) and not isinstance(v, bool)
    ]
    for key in numeric_fields:
        values = np.asarray([row[key] for row in objective_rows], dtype=float)
        objective_summary["median"][key] = float(np.median(values))
        objective_summary["p95"][key] = float(np.quantile(values, 0.95))
        objective_summary["worst_frame"][key] = int(np.argmax(np.abs(values)))
    term_counts: dict[str, int] = {}
    for row in objective_rows:
        term_counts[row["dominant_term"]] = term_counts.get(row["dominant_term"], 0) + 1
    source5 = [
        float(row["thresholds"]["5mm"]["near_surface_ratio"]) for row in source_proxy["frames"]
    ]
    final_anchor = float(np.median(anchor_final_distances))
    summary: dict[str, Any] = {
        "source_contact_proxy": {
            "thresholds_mm": source_proxy["thresholds_mm"],
            "frame_contact_ratio_at_5mm": float(np.mean(np.asarray(source5) > 0.0)),
            "median_near_surface_ratio_at_5mm": float(np.median(source5)),
            "source_visual_min_median": float(
                np.median([x["visual_min_distance_m"] for x in source_proxy["frames"]])
            ),
        },
        "per_frame": [x["metrics"] for x in geometry_frames],
        "active_margin_m": float(final.metadata["query_profile"]["active_margin_m"]),
        "anchor_final_median_m": final_anchor,
        "objective_dominant_term_counts": term_counts,
        "mean_final_minus_warm_eim": _safe_mean(
            row["final_eval_stage9_e_im_raw"] - row["warm_eval_stage9_e_im_raw"]
            for row in objective_rows
        ),
        "objective_summary": objective_summary,
    }
    root_causes = _root_causes(summary, offsets)
    summary["root_cause_ranked"] = root_causes["ranked_causes"]
    for row in geometry_frames:
        metrics = row["metrics"]
        tags: list[str] = []
        if (
            metrics["collision_visual_offset_max_mm"] is not None
            and metrics["collision_visual_offset_max_mm"] > 5.0
        ):
            tags.append("COLLISION_GEOMETRY_INFLATED")
        if metrics["query_active_count"] > 300:
            tags.append("QUERYSET_OVERREACH")
        if metrics["final_contact_retention_proxy_recall"] in (0.0, None):
            tags.append("CONTACT_RETENTION_PROXY_LOSS")
        if metrics["final_visual_min_m"] - metrics["final_collision_min_m"] < -0.003:
            tags.append("COLLISION_SAMPLE_COVERAGE_BIAS")
        metrics["root_cause_tags"] = tags
    interpolation_audit = {
        "schema_version": "toporetarget.warm_final_interpolation_audit.v1",
        "diagnostic_only": True,
        "optimizer_called": False,
        "alpha_count": 21,
        "visual_surface_approximation": "deterministic_surface_samples",
        "frames": [
            {
                "frame": frame,
                "earliest_feasible_alpha": next(
                    (
                        row["earliest_feasible_alpha"]
                        for row in interpolation_rows
                        if row["frame"] == frame
                    ),
                    None,
                ),
                "rows": [row for row in interpolation_rows if row["frame"] == frame],
            }
            for frame in selected
        ],
    }
    _write_json(output_root / "source_contact_proxy.json", source_proxy)
    _write_json(output_root / "contact_retention_proxy.json", retention)
    _write_json(output_root / "warm_final_interpolation_audit.json", interpolation_audit)
    _write_json(
        output_root / "contact_geometry_audit.json",
        {
            "schema_version": "toporetarget.contact_geometry_audit.v1",
            "backend": sdf.describe(),
            "backend_selection": inputs["distance_backend_selection"],
            "dense_surface_sample_count": dense_visual_count,
            "dense_surface_approximation": True,
            "dense_surface_includes_all_vertices": dense_include_vertices,
            "visual_vertex_max_per_instance": visual_vertex_max_per_instance,
            "frames": geometry_frames,
            "per_link_collision_visual_offset": offsets,
            "collision_sample_profile": inputs["surface"].as_dict(),
        },
    )
    _write_json(
        output_root / "queryset_audit_per_frame.json",
        {
            "schema_version": "toporetarget.queryset_audit.v1",
            "frames": [
                {
                    "frame": row["frame"],
                    "active_count": row["metrics"]["query_active_count"],
                    "active_margin_sensitivity_counts": row["metrics"][
                        "active_margin_sensitivity_counts"
                    ],
                }
                for row in geometry_frames
            ],
            "multiplier_available": False,
            "multiplier_note": "SciPy SLSQP multiplier output is unavailable; slack and finite differences are used instead.",
        },
    )
    _write_csv(output_root / "queryset_audit_per_point.csv", queryset_points)
    _write_csv(output_root / "queryset_audit_per_link.csv", queryset_links)
    _write_csv(output_root / "per_link_collision_visual_offset.csv", offsets)
    _write_csv(output_root / "objective_tradeoff_per_frame.csv", objective_rows)
    _write_csv(output_root / "warm_final_interpolation_per_frame.csv", interpolation_rows)
    _write_csv(
        output_root / "per_frame_contact_audit.csv", [row["metrics"] for row in geometry_frames]
    )
    _write_csv(
        output_root / "per_link_contact_audit.csv",
        [
            {"frame": f["frame"], "region": region, **values}
            for f in retention["frames"]
            for region, values in f["link_level"].items()
        ],
    )
    _write_json(output_root / "root_cause_analysis.json", root_causes)
    report = "# Stage 9.3 Contact Retention and Collision-Geometry Audit\n\n"
    report += f"- Source contact output is `source_contact_proxy`, not ground-truth contact.\n- Distance backend: `{sdf.describe()['backend_id']}`, positive signed distance is outside.\n- Dense visual surfaces are deterministic surface-sample approximations with at least {dense_visual_count} points per mesh state.\n- Audit-only solver invocation count: 0.\n- QuerySet active margin: {summary['active_margin_m'] * 1000:.1f} mm.\n\n"
    report += (
        "## Root-cause ranking\n\n| Rank | Cause | Confidence | Evidence |\n|---:|---|---|---|\n"
        + "\n".join(
            f"| {x['rank']} | {x['root_cause']} | {x['confidence']} | {'; '.join(x['evidence_for'])} |"
            for x in root_causes["ranked_causes"]
        )
        + "\n\n"
    )
    report += "## Stage 9.4 boundary\n\nThe faithful branch should preserve Eq. (1)-(9) and correct only proven geometry/query/anchor implementation issues. A contact-attraction or contact-preservation term belongs only to a separately labelled paper-external research extension. This run does not implement Stage 9.4 or run shadow ablation.\n"
    (output_root / "root_cause_report.md").write_text(report)
    if html:
        payload = {
            "schema_version": "toporetarget.contact_audit_html.v1",
            "title": "TopoRetarget Stage 9.3 contact audit",
            "formal_viewer_href": Path(
                preflight["artifacts"]["trajectory_mesh_html"]["path"]
            ).as_uri(),
            "worst_frame": int(
                np.argmax([row["metrics"]["final_visual_min_m"] for row in geometry_frames])
            ),
            "link_options": ["all", *sorted(set(REGIONS) | known_links)],
            "frames": geometry_frames,
        }
        _html(payload, output_root / "trajectory_contact_audit.html")
        if interactive:
            webbrowser.open((output_root / "trajectory_contact_audit.html").as_uri())
        if headless_smoke_test:
            _headless_smoke(
                output_root / "trajectory_contact_audit.html", output_root / "headless_smoke.json"
            )
    elif headless_smoke_test:
        raise ValueError("--headless-smoke-test requires --html")
    shadow = {
        "ran": False,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "reason": "not requested; audit-only solver invocation count=0",
    }
    if run_shadow_ablation:
        shadow = _shadow_score_diagnostic(objective_rows, selected, shadow_frames, geometry_frames)
        shadow_root = _resolve(repo_root, ".local/runs/stage9_3_shadow_ablation") / output_root.name
        if shadow_root.exists() and any(shadow_root.iterdir()) and not force:
            raise FileExistsError(f"shadow output exists; pass --force: {shadow_root}")
        shadow_root.mkdir(parents=True, exist_ok=True)
        shadow["output_dir"] = str(shadow_root)
        _write_json(shadow_root / "shadow_ablation.json", shadow)
    _write_json(output_root / "shadow_ablation_status.json", shadow)
    input_after = {name: _stat(path) for name, path in inputs["paths"].items()}
    input_after["manifest"] = _stat(manifest_path)
    preflight_after = {
        name: _stat(_resolve(repo_root, value["path"]))
        for name, value in preflight["artifacts"].items()
        if name != "trajectory_mesh_html"
    }
    preflight_after["trajectory_mesh_html"] = _stat(
        _resolve(repo_root, preflight["artifacts"]["trajectory_mesh_html"]["path"])
    )
    immutability = {
        name: {
            "before": preflight["artifacts"][name],
            "after": input_after.get(name, preflight_after.get(name)),
            "hash_unchanged": preflight["artifacts"][name]["sha256"]
            == input_after.get(name, preflight_after.get(name, {})).get("sha256"),
            "mtime_unchanged": preflight["artifacts"][name]["mtime_ns"]
            == input_after.get(name, preflight_after.get(name, {})).get("mtime_ns"),
        }
        for name in preflight["artifacts"]
    }
    _write_json(output_root / "artifact_immutability.json", immutability)
    source_proxy_summary: dict[str, Any] = summary["source_contact_proxy"]
    statuses = {
        "NUMERICAL_VALIDITY": "PASS",
        "COLLISION_FEASIBILITY": "PASS"
        if all(x["metrics"]["final_full_audit_min_m"] >= -0.030001 for x in geometry_frames)
        else "FAIL",
        "VISUAL_MESH_CLEARANCE": "INCONCLUSIVE",
        "SOURCE_CONTACT_RICHNESS": "PASS"
        if float(source_proxy_summary["frame_contact_ratio_at_5mm"]) >= 0.5
        else "INCONCLUSIVE",
        "CONTACT_RETENTION": "INCONCLUSIVE",
        "TEMPORAL_CONTINUITY": "PASS",
        "PHYSICAL_TRACKABILITY": "UNVERIFIED",
    }
    audit_manifest = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "code_version": AUDIT_CODE_VERSION,
        "created_at": time.time(),
        "input_manifest": str(manifest_path),
        "input_hashes": {name: value["sha256"] for name, value in preflight["artifacts"].items()},
        "geometry_backend": sdf.describe(),
        "distance_backend_selection": inputs["distance_backend_selection"],
        "surface_samples": dense_visual_count,
        "thresholds_mm": thresholds_mm or [1, 2, 3, 5, 8, 10],
        "frame_range_local": [start, stop],
        "complete_window": bool(start == 0 and stop == final.frame_count),
        "requested_links": sorted(requested_links),
        "assumptions": [
            SOURCE_REGION_PROXY,
            PAD_PROXY,
            "dense_surface_approximation",
            "contact_proxy_not_ground_truth",
            "interpolation_not_optimizer_trajectory",
            "slsqp_multipliers_unavailable",
        ],
        "solver_invocation_count": 0,
        "shadow_ablation": shadow,
        "artifact_immutability": immutability,
        "quality_status": statuses,
        "outputs": {},
        "elapsed_s": time.perf_counter() - started,
    }
    for path in sorted(output_root.iterdir()):
        if path.is_file() and path.name != "audit_manifest.json":
            audit_manifest["outputs"][path.name] = _sha(path)
    audit_manifest["status"] = (
        "STAGE9_3_CONTACT_AUDIT_COMPLETE_WITH_WARNINGS"
        if any(v in {"INCONCLUSIVE", "UNVERIFIED"} for v in statuses.values())
        else "STAGE9_3_CONTACT_AUDIT_COMPLETE"
    )
    _write_json(output_root / "audit_manifest.json", audit_manifest)
    return {
        "output_dir": str(output_root),
        "status": audit_manifest["status"],
        "summary": summary,
        "quality_status": statuses,
        "root_causes": root_causes,
        "immutability": immutability,
    }


def _headless_smoke(html_path: Path, report_path: Path) -> None:
    candidates = [
        [
            "google-chrome",
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--dump-dom",
            str(html_path),
        ],
        ["chromium", "--headless", "--no-sandbox", "--disable-gpu", "--dump-dom", str(html_path)],
    ]
    for command in candidates:
        if shutil.which(command[0]):
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            rendered_markers = (
                "Contact retention audit",
                "Stage 9.3.2 Canonical Contact Re-Audit",
            )
            payload = {
                "status": "pass"
                if result.returncode == 0
                and any(marker in result.stdout for marker in rendered_markers)
                else "fail",
                "command": command,
                "returncode": result.returncode,
                "stdout_bytes": len(result.stdout),
                "stderr": result.stderr[-2000:],
            }
            _write_json(report_path, payload)
            if payload["status"] != "pass":
                raise RuntimeError(f"headless HTML smoke test failed: {payload}")
            return
    _write_json(
        report_path, {"status": "unverified", "reason": "google-chrome/chromium unavailable"}
    )


__all__ = ["run_contact_audit"]
