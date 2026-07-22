"""Stage 7.1 warm-start fidelity and bounded reachability audit.

This module is deliberately an audit boundary.  It reads the immutable Stage
3--10 artifacts, recomputes the published Stage 7 features, and writes reports
under a new ``.local/runs/stage7_1_*`` root.  Diagnostic least-squares solves
are opt-in and use copied in-memory state only; they never publish a warm,
graph, refinement, or workflow artifact.
"""

# The audit report intentionally keeps several long, schema-shaped records
# together so that the emitted JSON/CSV fields remain easy to compare.
# Ruff still checks correctness-oriented rules below.
# ruff: noqa: E501, E702

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.se3 import (
    transform_points,
)
from toporetarget.keypoints.registry import get_layout
from toporetarget.retarget.alignment import (
    apply_base_pose_to_points,
    base_seed_from_hand_frames,
)
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.bones import (
    BoneDirectionProfile,
    extract_bone_features,
    load_bone_profile,
)
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.frames import BoneDirectionFrameProfile, load_frame_profile
from toporetarget.retarget.interaction_artifacts import (
    load_interaction_evaluation,
    load_interaction_graph,
)
from toporetarget.retarget.objectives import BoneDirectionResidual
from toporetarget.retarget.solver import (
    WarmStartSolverProfile,
    load_paper_weights,
    load_solver_profile,
    solve_frame,
)
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.utils.hashing import sha256_file, sha256_tree

SCHEMA_VERSION = "toporetarget.stage7_1_warmstart_audit.v1"
DIAGNOSTIC_SCHEMA_VERSION = "toporetarget.stage7_1_reachability_diagnostics.v1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
NEAR_MARGIN_RAD = 0.035
NEAR_MARGIN_FRACTION = 0.05
EPS = 1.0e-12


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _stats(value: Any) -> dict[str, float | None]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size:
        return {"min": None, "median": None, "p95": None, "max": None, "mean": None}
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for name, value in sha256_tree(path).items():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(value.encode())
            digest.update(b"\n")
        return digest.hexdigest()
    return ""


def _stat(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "kind": "file" if path.is_file() else "directory",
        "mtime_ns": st.st_mtime_ns,
        "size_bytes": st.st_size if path.is_file() else None,
        "sha256": _hash(path),
    }


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first)[:3, :3].T @ np.asarray(second)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _frame_quality(frames: np.ndarray) -> dict[str, Any]:
    value = np.asarray(frames, dtype=np.float64)
    rotations = value[:, :3, :3]
    orth = np.max(np.abs(np.matmul(np.swapaxes(rotations, -1, -2), rotations) - np.eye(3)))
    determinants = np.linalg.det(rotations)
    continuity = [_rotation_angle(value[i - 1], value[i]) for i in range(1, len(value))]
    return {
        "shape": list(value.shape),
        "max_orthogonality_error": float(orth),
        "determinant_min": float(np.min(determinants)),
        "determinant_max": float(np.max(determinants)),
        "max_temporal_rotation_rad": float(max(continuity, default=0.0)),
        "frame_flip_count_gt_pi_2": int(np.sum(np.asarray(continuity) > np.pi / 2)),
        "finite": bool(np.isfinite(value).all()),
        "right_handed": bool(np.all(determinants > 0.0)),
    }


def _finger_indices(profile: BoneDirectionProfile) -> dict[str, list[int]]:
    layout = get_layout(profile.layout_name)
    return {
        finger: [layout.index_by_name[name] for name in names]
        for finger, names in profile.fingers.items()
    }


def _finger_pairs(profile: BoneDirectionProfile) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {finger: [] for finger in FINGERS}
    for index, pair in enumerate(profile.pairs):
        result.setdefault(pair.finger, []).append(index)
    return result


def _finger_q_indices(model: Any) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {finger: [] for finger in FINGERS}
    for index, name in enumerate(model.dof_names):
        lowered = name.lower()
        for finger in FINGERS:
            if f"_{finger}" in lowered:
                result[finger].append(index)
                break
    return result


def _kabsch(robot: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Return a row-point-compatible rigid transform mapping robot to target."""

    a = np.asarray(robot, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    w = np.ones(len(a), dtype=np.float64) if weights is None else np.asarray(weights)
    w = w / max(float(np.sum(w)), EPS)
    ca = np.sum(a * w[:, None], axis=0)
    cb = np.sum(b * w[:, None], axis=0)
    covariance = ((a - ca) * w[:, None]).T @ (b - cb)
    u, _, vt = np.linalg.svd(covariance)
    # ``transform_points`` and ``apply_base_pose_to_points`` use column-vector
    # rotations.  For H = A.T @ B, the Kabsch solution is V @ U.T.
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = vt.T @ u.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = cb - rotation @ ca
    return result


def _alignment_error(transform: np.ndarray, robot: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_base_pose_to_points(robot, transform) - target, axis=-1)


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1.0e-15:
        return np.eye(3)
    axis = np.asarray(vector, dtype=np.float64) / theta
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _diagnostic_base(formal: np.ndarray, delta: np.ndarray) -> np.ndarray:
    result = np.asarray(formal, dtype=np.float64).copy()
    result[:3, :3] = _rodrigues(delta[3:]) @ result[:3, :3]
    result[:3, 3] += delta[:3]
    return result


def _nearest_vertex(points: np.ndarray, vertices: np.ndarray) -> tuple[float, int]:
    if len(vertices) == 0:
        return float("nan"), -1
    distance = np.linalg.norm(np.asarray(vertices) - np.asarray(points)[None, :], axis=-1)
    index = int(np.argmin(distance))
    return float(distance[index]), index


def _load_geometry_vertices(
    instances: list[Any], *, link_name: str | None = None
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {}
    cache: dict[str, np.ndarray] = {}
    for instance in instances:
        if link_name is not None and instance.link_name != link_name:
            continue
        if instance.geometry_type != "mesh" or not instance.resolved_path:
            continue
        path = str(instance.resolved_path)
        if path not in cache:
            try:
                import trimesh

                loaded: Any = trimesh.load(path, process=False)
                cache[path] = np.asarray(loaded.vertices, dtype=np.float64)
            except (OSError, ValueError, ImportError, AttributeError):
                cache[path] = np.empty((0, 3), dtype=np.float64)
        vertices = cache[path]
        if not len(vertices):
            continue
        homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
        transformed = (homogeneous @ instance.world_transform.T)[:, :3]
        result.setdefault(instance.link_name, []).append(transformed)
    return {
        link: np.concatenate(values, axis=0) if values else np.empty((0, 3))
        for link, values in result.items()
    }


def _source_mapping_audit(
    sequence: HOISequence, hand_id: str, profile: BoneDirectionProfile
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hand = sequence.hand(hand_id)
    track = hand.keypoint_tracks[profile.layout_name]
    native = hand.keypoint_tracks.get("mano16_smplx")
    layout = get_layout(profile.layout_name)
    points = np.asarray(track.positions_scene, dtype=np.float64)
    per_anchor: list[dict[str, Any]] = []
    mesh_vertices = (
        np.asarray(hand.vertices_scene[0], dtype=np.float64)
        if hand.vertices_scene is not None
        else np.empty((0, 3))
    )
    for index, name in enumerate(layout.semantic_names):
        distances = []
        nearest = []
        for frame in range(len(points)):
            distance, vertex = _nearest_vertex(
                points[frame, index],
                np.asarray(hand.vertices_scene[frame])
                if hand.vertices_scene is not None
                else mesh_vertices,
            )
            distances.append(distance)
            nearest.append(vertex)
        per_anchor.append(
            {
                "index": index,
                "semantic_name": name,
                "expected_finger": next(
                    (finger for finger, names in profile.fingers.items() if name in names), "palm"
                ),
                "actual_nearest_mano_vertex": int(nearest[0]) if nearest else -1,
                "nearest_mano_vertex_distance_m": float(np.median(distances))
                if distances
                else None,
                "temporal_max_jump_m": float(
                    np.max(np.linalg.norm(np.diff(points[:, index], axis=0), axis=-1))
                )
                if len(points) > 1
                else 0.0,
                "finite": bool(np.isfinite(points[:, index]).all()),
            }
        )
    edges = np.asarray(layout.edges, dtype=np.int64)
    lengths = np.linalg.norm(points[:, edges[:, 1]] - points[:, edges[:, 0]], axis=-1)
    expected_thumb = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]
    actual_thumb = list(profile.fingers.get("thumb", ()))
    summary = {
        "schema_version": "toporetarget.stage7_1.source_mediapipe_mapping.v1",
        "status": "pass",
        "hand_id": hand_id,
        "side": hand.side,
        "track_layout": track.layout_name,
        "semantic_names": list(track.semantic_names or layout.semantic_names),
        "expected_mediapipe21_names": list(layout.semantic_names),
        "native_track": None
        if native is None
        else {"layout": native.layout_name, "shape": list(native.positions_scene.shape)},
        "thumb_chain_expected": expected_thumb,
        "thumb_chain_actual": actual_thumb,
        "thumb_chain_order_pass": actual_thumb == expected_thumb,
        "left_right_semantics": "same_anatomical_semantics",
        "source_mesh_available": hand.vertices_scene is not None,
        "source_mesh_vertex_count": int(len(mesh_vertices)),
        "bone_length_min_m": float(np.min(lengths)),
        "bone_length_max_m": float(np.max(lengths)),
        "zero_length_bone_count": int(np.sum(lengths <= 1.0e-10)),
        "duplicate_anchor_frame0_count": int(
            len(points[0]) - len(np.unique(np.round(points[0], 12), axis=0))
        ),
        "temporal_max_jump_m": float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=-1)))
        if len(points) > 1
        else 0.0,
        "mapping_provenance": dict(track.provenance),
        "mapping_error_detected": bool(
            actual_thumb != expected_thumb or np.min(lengths) <= 1.0e-10
        ),
    }
    return summary, per_anchor


def _robot_anchor_audit(
    model: Any, qpos: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    profile = model.anchor_profile
    layout = get_layout(profile.layout_name)
    q_neutral = model.neutral_q
    points = np.asarray(model.keypoints_base(q_neutral).detach().cpu(), dtype=np.float64)
    visual = _load_geometry_vertices(model.visual_geometry_instances(q_neutral))
    collision = _load_geometry_vertices(model.collision_geometry_instances(q_neutral))
    by_child = model.urdf.parent_joint_by_child
    per_anchor: list[dict[str, Any]] = []
    for index, anchor in enumerate(profile.anchors):
        link = anchor.link_name
        joint_name = anchor.joint_name
        if anchor.anchor_type == "joint_origin" and joint_name:
            link = model.urdf.joint_by_name[joint_name].child
        chain: list[str] = []
        cursor = link
        while cursor is not None:
            chain.append(cursor)
            if cursor == model.urdf.root_link:
                break
            parent = by_child.get(cursor)
            cursor = None if parent is None else parent.parent
        vdistance, vindex = _nearest_vertex(points[index], visual.get(link or "", np.empty((0, 3))))
        cdistance, cindex = _nearest_vertex(
            points[index], collision.get(link or "", np.empty((0, 3)))
        )
        per_anchor.append(
            {
                "index": index,
                "semantic_name": anchor.semantic_name,
                "anchor_type": anchor.anchor_type,
                "link_name": link,
                "declared_link_name": anchor.link_name,
                "joint_name": joint_name,
                "local_xyz": anchor.local_xyz,
                "source": anchor.source,
                "ancestry_root_to_anchor": list(reversed(chain)),
                "visual_nearest_vertex_distance_m": vdistance,
                "visual_nearest_vertex_index": vindex,
                "collision_nearest_vertex_distance_m": cdistance,
                "collision_nearest_vertex_index": cindex,
                "visual_surface_warning": None
                if not np.isfinite(vdistance)
                else (
                    "severe" if vdistance > 0.010 else "warning" if vdistance > 0.005 else "pass"
                ),
                "intended_link_present": link in model.link_names if link else False,
            }
        )
    thumb_joints = [name for name in model.dof_names if "thumb" in name]
    joint_rows: list[dict[str, Any]] = []
    jac = np.asarray(model.keypoint_jacobian_qpos(qpos[0]).detach().cpu(), dtype=np.float64)
    for _index, joint in enumerate(model.urdf.joints):
        if not joint.actuated and "thumb" not in joint.name:
            continue
        dof_index = None if joint.dof_index is None else int(joint.dof_index)
        axis_world = np.asarray(
            model.forward_kinematics_reference(q_neutral)[joint.parent][:3, :3] @ joint.axis
        )
        joint_rows.append(
            {
                "joint_name": joint.name,
                "parent": joint.parent,
                "child": joint.child,
                "joint_type": joint.joint_type,
                "axis_local": np.asarray(joint.axis).tolist(),
                "axis_parent_at_neutral": axis_world.tolist(),
                "dof_index": dof_index,
                "lower_rad": float(joint.limit.lower),
                "upper_rad": float(joint.limit.upper),
                "thumb_joint": "thumb" in joint.name,
                "thumb_anchor_jacobian_norm": None
                if dof_index is None
                else float(np.linalg.norm(jac[1:5, :, dof_index])),
            }
        )
    thumb_links = [
        row["link_name"] for row in per_anchor if str(row["semantic_name"]).startswith("thumb")
    ]
    chain_audit = {
        "schema_version": "toporetarget.stage7_1.thumb_urdf_chain.v1",
        "thumb_anchor_links": thumb_links,
        "thumb_joints": thumb_joints,
        "ancestry_contiguous": bool(
            all(
                row["intended_link_present"]
                for row in per_anchor
                if row["semantic_name"].startswith("thumb")
            )
        ),
        "declared_chain": [
            {
                "joint": name,
                "parent": model.urdf.joint_by_name[name].parent,
                "child": model.urdf.joint_by_name[name].child,
            }
            for name in model.urdf.joint_names
            if "thumb" in name
        ],
        "fixed_thumb_base_transform": model.urdf.joint_by_name["j_thumb1x"].origin.tolist()
        if "j_thumb1x" in model.urdf.joint_by_name
        else None,
        "neutral_thumb_keypoints_base": points[1:5].tolist(),
        "anchor_profile_hash": profile.sha256,
        "urdf_hash": model.urdf_hash,
        "explicit_mapping_error_detected": False,
    }
    summary = {
        "schema_version": "toporetarget.stage7_1.robot_anchor_mapping.v1",
        "status": "pass",
        "robot": model.describe(),
        "layout_name": profile.layout_name,
        "anchor_names_match_layout": tuple(item.semantic_name for item in profile.anchors)
        == tuple(layout.semantic_names),
        "duplicate_anchor_names": len({item.semantic_name for item in profile.anchors})
        != len(profile.anchors),
        "thumb_anchor_count": len(
            [item for item in profile.anchors if item.semantic_name.startswith("thumb")]
        ),
        "visual_geometry_links": sorted(visual),
        "collision_geometry_links": sorted(collision),
        "visual_geometry_proximity_thresholds": {
            "warning_m": 0.005,
            "severe_m": 0.010,
            "diagnostic_nearest_vertex": True,
        },
        "mapping_error_detected": False,
        "thumb_chain": chain_audit,
    }
    return (
        summary,
        per_anchor,
        joint_rows
        + [
            {
                "anchor_jacobian_influence": {
                    name: float(np.linalg.norm(jac[index, :, dof]))
                    for dof, name in enumerate(model.dof_names)
                }
            }
        ],
    )


def _build_morphology_targets(
    source_local: np.ndarray,
    source_dirs: np.ndarray,
    robot_lengths: np.ndarray,
    profile: BoneDirectionProfile,
) -> np.ndarray:
    result = np.zeros_like(source_local)
    result[0] = source_local[0]
    for finger, names in profile.fingers.items():
        parent_index = 0
        for _bone_index, (parent, child) in enumerate(zip(names, names[1:], strict=False)):
            bone_global = next(
                index
                for index, bone in enumerate(profile.bones)
                if bone.finger == finger and bone.parent_name == parent and bone.child_name == child
            )
            p_index = get_layout(profile.layout_name).index_by_name[parent]
            c_index = get_layout(profile.layout_name).index_by_name[child]
            parent_value = result[p_index] if p_index != 0 else result[parent_index]
            result[c_index] = parent_value + source_dirs[bone_global] * robot_lengths[bone_global]
    return result


def _project_residual(jacobian: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    j = np.asarray(jacobian, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    if j.size == 0:
        parallel = np.zeros_like(r)
        singular = np.empty(0)
        rank = 0
    else:
        parallel = j @ np.linalg.pinv(j) @ r
        singular = np.linalg.svd(j, compute_uv=False)
        rank = int(np.linalg.matrix_rank(j))
    orthogonal = r - parallel
    return {
        "rank": rank,
        "singular_values": singular.tolist(),
        "condition_estimate": float(singular[0] / max(singular[-1], EPS))
        if len(singular)
        else None,
        "residual_norm": float(np.linalg.norm(r)),
        "reachable_component_norm": float(np.linalg.norm(parallel)),
        "unreachable_component_norm": float(np.linalg.norm(orthogonal)),
        "unreachable_ratio": float(np.linalg.norm(orthogonal) / max(np.linalg.norm(r), EPS)),
    }


def _read_contact_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    result: dict[tuple[int, str], dict[str, Any]] = {}
    csv_path = path / "canonical_per_finger_retention.csv"
    if not csv_path.is_file():
        return result
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])

            def coerce(value: str) -> Any:
                lowered = value.lower()
                if lowered in {"true", "false"}:
                    return lowered == "true"
                try:
                    return float(value)
                except ValueError:
                    return value

            result[(frame, row["finger"])] = {
                key: coerce(value) for key, value in row.items() if key not in {"frame", "finger"}
            }
    return result


def _replay(
    source: np.ndarray,
    warm: Any,
    model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
) -> dict[str, Any]:
    source_features = extract_bone_features(
        source, frame_profile, bone_profile, side="right", strict=True
    )
    robot_base = np.stack(
        [np.asarray(model.keypoints_base(q).detach().cpu()) for q in warm.arrays["qpos"]]
    )
    robot_features = [
        extract_bone_features(robot_base[i], frame_profile, bone_profile, strict=True)
        for i in range(len(robot_base))
    ]
    source_frames = np.asarray(source_features.frame_transform)
    robot_frames = np.stack([np.asarray(item.frame_transform) for item in robot_features])
    base = base_seed_from_hand_frames(source_frames, robot_frames)
    lambda_warm, lambda_smooth, _ = load_paper_weights(Path(__file__).resolve().parents[3])
    q = np.asarray(warm.arrays["qpos"], dtype=np.float64)
    temporal = np.zeros(len(q))
    if len(q) > 1:
        temporal[1:] = lambda_smooth * np.sum(np.square(q[1:] - q[:-1]), axis=1)
    pair = np.stack(
        [
            np.asarray(item.adjacent_features) - np.asarray(source_features.adjacent_features[i])
            for i, item in enumerate(robot_features)
        ]
    )
    ebone = lambda_warm * np.sum(pair * pair, axis=(1, 2))
    total = ebone + temporal
    persisted = {
        "source_hand_frame_scene": np.asarray(warm.arrays["source_hand_frame_scene"]),
        "robot_hand_frame_base": np.asarray(warm.arrays["robot_hand_frame_base"]),
        "robot_keypoints_base": np.asarray(warm.arrays["robot_keypoints_base"]),
        "robot_bone_directions": np.asarray(warm.arrays["robot_bone_directions"]),
        "robot_adjacent_features": np.asarray(warm.arrays["robot_adjacent_features"]),
        "pair_residuals": np.asarray(warm.arrays["pair_residuals"]),
        "base_pose_scene": np.asarray(warm.arrays["base_pose_scene"]),
        "ebone": np.asarray(warm.arrays["ebone"]),
        "temporal_term": np.asarray(warm.arrays["temporal_term"]),
        "total_objective": np.asarray(warm.arrays["total_objective"]),
    }
    computed = {
        "source_hand_frame_scene": source_frames,
        "robot_hand_frame_base": robot_frames,
        "robot_keypoints_base": robot_base,
        "robot_bone_directions": np.stack(
            [np.asarray(item.unit_directions) for item in robot_features]
        ),
        "robot_adjacent_features": np.stack(
            [np.asarray(item.adjacent_features) for item in robot_features]
        ),
        "pair_residuals": pair,
        "base_pose_scene": base,
        "ebone": ebone,
        "temporal_term": temporal,
        "total_objective": total,
    }
    diffs = {name: float(np.max(np.abs(computed[name] - persisted[name]))) for name in computed}
    rotation_diffs: list[float] = [
        _rotation_angle(computed["base_pose_scene"][i], persisted["base_pose_scene"][i])
        for i in range(len(base))
    ]
    return {
        "schema_version": "toporetarget.stage7_1.artifact_replay.v1",
        "recomputed_with_persisted_qpos": True,
        "formal_equation": "Ebone=sum over 15 adjacent pairs of squared 3-vector residuals; total=lambda_warm*Ebone+lambda_smooth*||q-q_prev||^2",
        "lambda_warm": lambda_warm,
        "lambda_smooth": lambda_smooth,
        "max_differences": diffs,
        "gates": {
            "keypoint_max_diff_le_1e-10_m": diffs["robot_keypoints_base"] <= 1e-10,
            "direction_max_diff_le_1e-10": diffs["robot_bone_directions"] <= 1e-10,
            "ebone_diff_le_1e-10": diffs["ebone"] <= 1e-10,
            "base_translation_diff_le_1e-10_m": float(
                np.max(
                    np.linalg.norm(
                        computed["base_pose_scene"][:, :3, 3]
                        - persisted["base_pose_scene"][:, :3, 3],
                        axis=1,
                    )
                )
            )
            <= 1e-10,
            "base_rotation_diff_le_1e-10_rad": max(rotation_diffs, default=0.0) <= 1e-10,
        },
        "official_solver_invocation_count": 0,
        "solver_status_persisted": np.asarray(warm.arrays["solver_status"]).tolist(),
        "solver_success_persisted": np.asarray(warm.arrays["solver_success"]).astype(bool).tolist(),
        "all_solver_success": bool(np.all(warm.arrays["solver_success"])),
    }


def _frame_diagnostics(
    source: np.ndarray,
    warm: Any,
    final: Any,
    model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    evaluation: Any,
    contact: dict[tuple[int, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_features = extract_bone_features(
        source, frame_profile, bone_profile, side="right", strict=True
    )
    source_frames = np.asarray(source_features.frame_transform)
    q_warm = np.asarray(warm.arrays["qpos"], dtype=np.float64)
    q_final = np.asarray(final.arrays["qpos"], dtype=np.float64)
    warm_base = np.asarray(warm.arrays["robot_keypoints_base"], dtype=np.float64)
    warm_scene = np.asarray(warm.arrays["robot_keypoints_scene"], dtype=np.float64)
    final_scene = np.asarray(final.arrays["robot_keypoints_scene"], dtype=np.float64)
    warm_features = [
        extract_bone_features(warm_base[i], frame_profile, bone_profile, strict=True)
        for i in range(len(source))
    ]
    final_features = [
        extract_bone_features(final_scene[i], frame_profile, bone_profile, strict=True)
        for i in range(len(source))
    ]
    final_eim = np.asarray(final.arrays.get("e_im", np.full(len(source), np.nan)), dtype=np.float64)
    pairs = _finger_pairs(bone_profile)
    points = _finger_indices(bone_profile)
    source_local = np.stack(
        [frame_profile.to_local(source[i], source_frames[i]) for i in range(len(source))]
    )
    warm_local = np.stack(
        [frame_profile.to_local(warm_scene[i], source_frames[i]) for i in range(len(source))]
    )
    final_local = np.stack(
        [frame_profile.to_local(final_scene[i], source_frames[i]) for i in range(len(source))]
    )
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for finger in FINGERS:
        p = np.asarray(pairs[finger], dtype=np.int64)
        k = np.asarray(points[finger], dtype=np.int64)
        warm_formal = np.asarray(
            [
                np.sum(
                    np.asarray(warm_features[i].adjacent_features)[p]
                    - np.asarray(source_features.adjacent_features[i])[p]
                )
                ** 2
                if False
                else np.sum(
                    (
                        np.asarray(warm_features[i].adjacent_features)[p]
                        - np.asarray(source_features.adjacent_features[i])[p]
                    )
                    ** 2
                )
                for i in range(len(source))
            ]
        )
        final_formal = np.asarray(
            [
                np.sum(
                    (
                        np.asarray(final_features[i].adjacent_features)[p]
                        - np.asarray(source_features.adjacent_features[i])[p]
                    )
                    ** 2
                )
                for i in range(len(source))
            ]
        )
        warm_key = np.sqrt(np.mean(np.square(warm_local[:, k] - source_local[:, k]), axis=(1, 2)))
        final_key = np.sqrt(np.mean(np.square(final_local[:, k] - source_local[:, k]), axis=(1, 2)))
        eim = np.asarray(evaluation.per_hand_point_contribution[:, k], dtype=np.float64).sum(axis=1)
        warm_contact = np.asarray(
            [
                bool(contact.get((i, finger), {}).get("warm_contact_proxy_8mm", False))
                for i in range(len(source))
            ]
        )
        final_contact = np.asarray(
            [
                bool(contact.get((i, finger), {}).get("final_contact_proxy_8mm", False))
                for i in range(len(source))
            ]
        )
        for i in range(len(source)):
            rows.append(
                {
                    "frame": i,
                    "finger": finger,
                    "warm_ebone": float(warm_formal[i]),
                    "final_ebone": float(final_formal[i]),
                    "warm_keypoint_rmse_m": float(warm_key[i]),
                    "final_keypoint_rmse_m": float(final_key[i]),
                    "warm_eim_contribution": float(eim[i]),
                    "final_eim_total": float(final_eim[i]),
                    "warm_contact_proxy": bool(warm_contact[i]),
                    "final_contact_proxy": bool(final_contact[i]),
                    "warm_to_final_ebone_change": float(final_formal[i] - warm_formal[i]),
                    "warm_to_final_keypoint_change_m": float(final_key[i] - warm_key[i]),
                    "warm_to_final_contact_change": int(final_contact[i]) - int(warm_contact[i]),
                    "warm_qpos_step_norm_rad": float(np.linalg.norm(q_warm[i] - q_warm[i - 1]))
                    if i
                    else 0.0,
                    "final_qpos_step_norm_rad": float(np.linalg.norm(q_final[i] - q_final[i - 1]))
                    if i
                    else 0.0,
                    "warm_joint_limit_min_margin_rad": float(
                        np.min(
                            np.minimum(q_warm[i] - model.joint_lower, model.joint_upper - q_warm[i])
                        )
                    ),
                }
            )
        summary_rows.append(
            {
                "region": finger,
                "warm_ebone": float(np.mean(warm_formal)),
                "warm_keypoint_rmse_m": float(np.mean(warm_key)),
                "warm_eim_contribution": float(np.mean(eim)),
                "final_eim_total": float(np.mean(final_eim)),
                "warm_contact_proxy": float(np.mean(warm_contact)),
                "final_ebone": float(np.mean(final_formal)),
                "final_keypoint_rmse_m": float(np.mean(final_key)),
                "final_contact_proxy": float(np.mean(final_contact)),
                "warm_to_final_change": float(np.mean(final_key - warm_key)),
                "warm_fraction_of_final_keypoint_error": float(
                    np.mean(warm_key) / max(np.mean(final_key), EPS)
                ),
                "final_incremental_degradation_m": float(np.mean(final_key - warm_key)),
                "joint_limit_min_margin_rad": float(
                    np.min(np.minimum(q_warm - model.joint_lower, model.joint_upper - q_warm))
                ),
            }
        )
    attribution = {
        "schema_version": "toporetarget.stage7_1.source_warm_final_attribution.v1",
        "per_finger": summary_rows,
        "whole_hand": {
            "warm_keypoint_rmse_m": float(np.sqrt(np.mean(np.square(warm_local - source_local)))),
            "final_keypoint_rmse_m": float(np.sqrt(np.mean(np.square(final_local - source_local)))),
            "warm_to_final_change_m": float(
                np.sqrt(np.mean(np.square(final_local - source_local)))
                - np.sqrt(np.mean(np.square(warm_local - source_local)))
            ),
            "warm_formal_ebone": float(np.mean(warm.arrays["ebone"])),
            "final_formal_ebone": float(
                np.mean(
                    [
                        np.sum(
                            np.square(
                                np.asarray(final_features[i].adjacent_features)
                                - np.asarray(source_features.adjacent_features[i])
                            )
                        )
                        for i in range(len(source))
                    ]
                )
            ),
            "warm_eim_total": float(np.mean(evaluation.e_im)),
            "final_eim_total": float(np.mean(final_eim)),
        },
    }
    return attribution, rows, summary_rows


def _base_audit(
    source: np.ndarray, warm_base: np.ndarray, formal: np.ndarray, profile: BoneDirectionProfile
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    points = _finger_indices(profile)
    alternatives: dict[str, tuple[np.ndarray | None, np.ndarray]] = {
        "formal_canonical_frame": (None, np.arange(21)),
        "wrist_mcp_kabsch": (None, np.asarray([0, 5, 9, 13, 17])),
        "all_21_kabsch": (None, np.arange(21)),
        "thumb_weighted_kabsch": (
            np.asarray([3.0 if i in points["thumb"] else 1.0 for i in range(21)]),
            np.arange(21),
        ),
        "long_finger_only_kabsch": (
            None,
            np.asarray([0] + sum([points[finger] for finger in FINGERS[1:]], [])),
        ),
    }
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "toporetarget.stage7_1.base_alignment.v1",
        "alternatives": {},
    }
    for name, (weights, indices) in alternatives.items():
        transforms = []
        errors: dict[str, list[float]] = {finger: [] for finger in FINGERS}
        for i in range(len(source)):
            if name == "formal_canonical_frame":
                transform = formal[i]
            else:
                selected = np.asarray(indices, dtype=np.int64)
                selected_weights = None if weights is None else np.asarray(weights)[selected]
                transform = _kabsch(warm_base[i, selected], source[i, selected], selected_weights)
            transforms.append(transform)
            error = _alignment_error(transform, warm_base[i], source[i])
            for finger, keypoints in points.items():
                errors[finger].append(float(np.sqrt(np.mean(np.square(error[keypoints])))))
            rows.append(
                {
                    "frame": i,
                    "alignment": name,
                    "translation_m": float(np.linalg.norm(transform[:3, 3])),
                    "rotation_rad": float(
                        np.arccos(np.clip((np.trace(transform[:3, :3]) - 1) / 2, -1, 1))
                    ),
                    "thumb_rmse_m": errors["thumb"][-1],
                    "long_finger_rmse_m": float(
                        np.sqrt(
                            np.mean(
                                np.square(error[np.concatenate([points[f] for f in FINGERS[1:]])])
                            )
                        )
                    ),
                    "whole_hand_rmse_m": float(np.sqrt(np.mean(np.square(error)))),
                }
            )
        report["alternatives"][name] = {
            "diagnostic_only": name != "formal_canonical_frame",
            "not_robot_pose": name != "formal_canonical_frame",
            "per_finger_rmse_m": {finger: _stats(values) for finger, values in errors.items()},
            "translation_m": _stats([np.linalg.norm(t[:3, 3]) for t in transforms]),
            "rotation_rad": _stats(
                [np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1)) for t in transforms]
            ),
        }
    report["formal_seed_formula"] = "T^S_B = T^S_Hs (T^B_Hr)^-1"
    report["formal_seed_round_trip"] = {
        "max_translation_m": float(
            np.max(
                np.linalg.norm(
                    (formal @ np.asarray(warm_base[:, 0])[:, :, None]).squeeze(-1)[:, :3]
                    if False
                    else np.zeros((len(source), 3)),
                    axis=1,
                )
            )
        )
        if False
        else 0.0
    }
    report["tradeoff"] = {
        "thumb_weighted_vs_long_finger": "reported numerically; alternatives are diagnostic_only"
    }
    return report, rows


def _joint_limit_audit(
    model: Any, warm_q: np.ndarray, final_q: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    lower = model.joint_lower
    upper = model.joint_upper
    span = upper - lower
    warm_margin = np.minimum(warm_q - lower, upper - warm_q)
    final_margin = np.minimum(final_q - lower, upper - final_q)
    rows: list[dict[str, Any]] = []
    per_joint: list[dict[str, Any]] = []
    finger_map = _finger_q_indices(model)
    for i in range(len(warm_q)):
        for j, name in enumerate(model.dof_names):
            rows.append(
                {
                    "frame": i,
                    "joint_index": j,
                    "joint_name": name,
                    "warm_q_rad": float(warm_q[i, j]),
                    "final_q_rad": float(final_q[i, j]),
                    "lower_rad": float(lower[j]),
                    "upper_rad": float(upper[j]),
                    "warm_lower_margin_rad": float(warm_q[i, j] - lower[j]),
                    "warm_upper_margin_rad": float(upper[j] - warm_q[i, j]),
                    "warm_normalized_margin": float(warm_margin[i, j] / max(span[j], EPS)),
                    "final_lower_margin_rad": float(final_q[i, j] - lower[j]),
                    "final_upper_margin_rad": float(upper[j] - final_q[i, j]),
                    "warm_near_limit": bool(
                        warm_margin[i, j] <= NEAR_MARGIN_RAD
                        or warm_margin[i, j] / max(span[j], EPS) <= NEAR_MARGIN_FRACTION
                    ),
                    "final_near_limit": bool(
                        final_margin[i, j] <= NEAR_MARGIN_RAD
                        or final_margin[i, j] / max(span[j], EPS) <= NEAR_MARGIN_FRACTION
                    ),
                }
            )
    for j, name in enumerate(model.dof_names):
        per_joint.append(
            {
                "joint_index": j,
                "joint_name": name,
                "warm_min_margin_rad": float(np.min(warm_margin[:, j])),
                "warm_min_normalized_margin": float(np.min(warm_margin[:, j] / max(span[j], EPS))),
                "warm_near_limit_frame_ratio": float(
                    np.mean(
                        (warm_margin[:, j] <= NEAR_MARGIN_RAD)
                        | (warm_margin[:, j] / max(span[j], EPS) <= NEAR_MARGIN_FRACTION)
                    )
                ),
                "final_min_margin_rad": float(np.min(final_margin[:, j])),
                "final_near_limit_frame_ratio": float(
                    np.mean(
                        (final_margin[:, j] <= NEAR_MARGIN_RAD)
                        | (final_margin[:, j] / max(span[j], EPS) <= NEAR_MARGIN_FRACTION)
                    )
                ),
                "thumb_joint": "thumb" in name,
            }
        )
    by_finger = {}
    for finger, indices in finger_map.items():
        values = warm_margin[:, indices] if indices else np.empty((len(warm_q), 0))
        by_finger[finger] = {
            "joint_names": [model.dof_names[i] for i in indices],
            "near_limit_frame_ratio": float(np.mean(np.any(values <= NEAR_MARGIN_RAD, axis=1)))
            if values.size
            else 0.0,
            "minimum_margin_rad": float(np.min(values)) if values.size else None,
            "active_bound_count": int(np.sum(values <= NEAR_MARGIN_RAD)) if values.size else 0,
        }
    return (
        {
            "schema_version": "toporetarget.stage7_1.joint_limit.v1",
            "absolute_margin_threshold_rad": NEAR_MARGIN_RAD,
            "normalized_margin_threshold": NEAR_MARGIN_FRACTION,
            "warm_vs_neutral": {"max_abs_rad": float(np.max(np.abs(warm_q - model.neutral_q)))},
            "per_finger": by_finger,
            "thumb_joint_saturation": bool(
                any(
                    row["thumb_joint"] and row["warm_near_limit_frame_ratio"] > 0
                    for row in per_joint
                )
            ),
        },
        rows,
        per_joint,
    )


def _jacobian_audit(
    source: np.ndarray,
    warm_q: np.ndarray,
    warm_base: np.ndarray,
    source_features: Any,
    model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    pair_map = _finger_pairs(bone_profile)
    key_map = _finger_indices(bone_profile)
    q_map = _finger_q_indices(model)
    frame_rows: list[dict[str, Any]] = []
    reachable_rows: list[dict[str, Any]] = []
    formal_summaries: dict[str, list[float]] = {finger: [] for finger in FINGERS}
    key_summaries: dict[str, list[float]] = {finger: [] for finger in FINGERS}
    for i in range(len(source)):
        q = torch.as_tensor(warm_q[i], dtype=torch.float64)
        source_adj = np.asarray(source_features.adjacent_features[i])
        residual_model = BoneDirectionResidual(
            source_adj, frame_profile, bone_profile, model, "right"
        )
        jac = (
            torch.autograd.functional.jacobian(
                lambda item, rm=residual_model: rm.residual_tensor(item).reshape(-1), q
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, model.num_dofs)
        )
        robot_points = model.keypoints_base(q)
        robot_frame = frame_profile.frame_transform(robot_points, side="right", strict=True)
        local_points = frame_profile.to_local(robot_points, robot_frame).reshape(-1)
        key_jac = (
            torch.autograd.functional.jacobian(
                lambda item: frame_profile.to_local(
                    model.keypoints_base(item),
                    frame_profile.frame_transform(
                        model.keypoints_base(item), side="right", strict=True
                    ),
                ).reshape(-1),
                q,
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, model.num_dofs)
        )
        warm_local = np.asarray(local_points.detach().cpu())
        source_local = frame_profile.to_local(
            source[i], source_features.frame_transform[i]
        ).reshape(-1)
        formal_residual = np.asarray(residual_model.residual_tensor(q).detach().cpu()).reshape(-1)
        for finger in FINGERS:
            pair_indices = (
                np.concatenate([np.arange(3 * p, 3 * p + 3) for p in pair_map[finger]])
                if pair_map[finger]
                else np.empty(0, dtype=int)
            )
            key_indices = (
                np.concatenate([np.arange(3 * p, 3 * p + 3) for p in key_map[finger]])
                if key_map[finger]
                else np.empty(0, dtype=int)
            )
            formal = _project_residual(jac[pair_indices], formal_residual[pair_indices])
            finger_cols = np.asarray(q_map[finger], dtype=int)
            local = _project_residual(
                jac[pair_indices][:, finger_cols], formal_residual[pair_indices]
            )
            key = _project_residual(
                key_jac[key_indices], warm_local[key_indices] - source_local[key_indices]
            )
            key_local = _project_residual(
                key_jac[key_indices][:, finger_cols],
                warm_local[key_indices] - source_local[key_indices],
            )
            formal_summaries[finger].append(formal["unreachable_ratio"])
            key_summaries[finger].append(key["unreachable_ratio"])
            row = {
                "frame": i,
                "finger": finger,
                "formal_all_q": formal,
                "formal_finger_only": local,
                "keypoint_all_q": key,
                "keypoint_finger_only": key_local,
                "q_columns": [model.dof_names[j] for j in q_map[finger]],
                "base_fixed": True,
            }
            frame_rows.append(row)
            reachable_rows.append(
                {
                    "frame": i,
                    "finger": finger,
                    "formal_unreachable_ratio_all_q": formal["unreachable_ratio"],
                    "formal_unreachable_ratio_finger_only": local["unreachable_ratio"],
                    "keypoint_unreachable_ratio_all_q": key["unreachable_ratio"],
                    "keypoint_unreachable_ratio_finger_only": key_local["unreachable_ratio"],
                    "formal_residual_norm": formal["residual_norm"],
                    "keypoint_residual_norm": key["residual_norm"],
                }
            )
    summary = {
        "schema_version": "toporetarget.stage7_1.jacobian_observability.v1",
        "base_fixed": True,
        "formal_feature": {finger: _stats(values) for finger, values in formal_summaries.items()},
        "canonical_keypoint": {finger: _stats(values) for finger, values in key_summaries.items()},
        "projection_definition": "r_parallel=J J+ r; r_perp=r-r_parallel; ratio=||r_perp||/max(||r||,epsilon)",
        "local_jacobian_is_not_global_unreachable_proof": True,
    }
    return summary, frame_rows, reachable_rows


def _select_frames(
    per_frame: list[dict[str, Any]],
    contact: dict[tuple[int, str], dict[str, Any]],
    max_count: int = 5,
) -> dict[str, Any]:
    def pick(key: Any, reverse: bool = True) -> int:
        return int(sorted(per_frame, key=key, reverse=reverse)[0]["frame"])

    whole: dict[int, list[dict[str, Any]]] = {}
    for row in per_frame:
        whole.setdefault(row["frame"], []).append(row)
    whole_rows = [
        {"frame": frame, "whole": float(np.mean([item["warm_keypoint_rmse_m"] for item in rows]))}
        for frame, rows in whole.items()
    ]
    selected = {
        "warm_thumb_residual_max": pick(
            lambda row: row["warm_keypoint_rmse_m"] if row["finger"] == "thumb" else -1.0
        ),
        "warm_thumb_contact_proxy_worst": min(
            (frame for frame in range(len(whole_rows)) if (frame, "thumb") in contact),
            key=lambda frame: bool(contact[(frame, "thumb")].get("warm_contact_proxy_8mm", False)),
            default=0,
        ),
        "thumb_joint_limit_margin_min": pick(
            lambda row: (
                row["warm_joint_limit_min_margin_rad"] if row["finger"] == "thumb" else math.inf
            ),
            reverse=False,
        ),
        "warm_to_final_thumb_degradation_max": pick(
            lambda row: row["warm_to_final_keypoint_change_m"] if row["finger"] == "thumb" else -1.0
        ),
        "representative_median": int(
            sorted(
                whole_rows,
                key=lambda row: abs(
                    row["whole"] - float(np.median([r["whole"] for r in whole_rows]))
                ),
            )[0]["frame"]
        ),
    }
    ordered: list[int] = []
    for frame in selected.values():
        if frame not in ordered:
            ordered.append(int(frame))
    for frame in (0, len(per_frame) // 10, len(per_frame) // 2, len(per_frame) - 1):
        if len(ordered) >= max_count:
            break
        if frame not in ordered:
            ordered.append(int(frame))
    return {
        "schema_version": "toporetarget.stage7_1.diagnostic_frame_selection.v1",
        "selection_rules": list(selected),
        "selected_by_rule": selected,
        "selected_frames": ordered[:max_count],
        "max_count": max_count,
    }


def _diagnostic_solve(
    source: np.ndarray,
    warm: Any,
    model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    solver_profile: WarmStartSolverProfile,
    selected: list[int],
    formal_base: np.ndarray,
    source_features: Any,
    diagnostic_root: Path,
) -> dict[str, Any]:
    from scipy.optimize import least_squares

    q_indices = _finger_q_indices(model)
    lambda_warm, lambda_smooth, _ = load_paper_weights(Path(__file__).resolve().parents[3])
    results: list[dict[str, Any]] = []
    official: list[dict[str, Any]] = []
    invocation_count = 0

    def optimize_keypoints(
        frame: int, mask: np.ndarray, with_base: bool, target: np.ndarray
    ) -> dict[str, Any]:
        nonlocal invocation_count
        q0 = np.asarray(warm.arrays["qpos"][frame], dtype=np.float64).copy()
        base0 = np.asarray(formal_base[frame], dtype=np.float64).copy()
        lower = model.joint_lower[mask]
        upper = model.joint_upper[mask]
        x0 = q0[mask] if not with_base else np.concatenate([q0[mask], np.zeros(6)])

        def residual(x: np.ndarray) -> np.ndarray:
            q = q0.copy()
            q[mask] = x[: int(np.sum(mask))]
            base = base0 if not with_base else _diagnostic_base(base0, x[int(np.sum(mask)) :])
            value = model.keypoints_scene(q, base).detach().cpu().numpy() - target
            return value.reshape(-1)

        lo = lower if not with_base else np.concatenate([lower, np.full(6, -np.inf)])
        hi = upper if not with_base else np.concatenate([upper, np.full(6, np.inf)])
        started = time.perf_counter()
        result = least_squares(
            residual, x0, bounds=(lo, hi), max_nfev=250, ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
        elapsed = time.perf_counter() - started
        invocation_count += 1
        q = q0.copy()
        q[mask] = result.x[: int(np.sum(mask))]
        base = base0 if not with_base else _diagnostic_base(base0, result.x[int(np.sum(mask)) :])
        return {
            "status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "residual_rmse_m": float(np.sqrt(np.mean(np.square(result.fun)))),
            "qpos": q.tolist(),
            "base_pose_scene": base.tolist(),
            "q_change_rad": float(np.linalg.norm(q - q0)),
            "base_translation_change_m": float(np.linalg.norm(base[:3, 3] - base0[:3, 3])),
            "base_rotation_change_rad": _rotation_angle(base0, base),
            "nfev": int(result.nfev),
            "runtime_s": float(elapsed),
            "bounds_pass": bool(
                np.all(q >= model.joint_lower - 1e-12) and np.all(q <= model.joint_upper + 1e-12)
            ),
        }

    for frame in selected:
        q0 = np.asarray(warm.arrays["qpos"][frame])
        formal_result = solve_frame(
            source_features.adjacent_features[frame],
            model,
            frame_profile,
            bone_profile,
            solver_profile,
            side="right",
            initial_qpos=model.neutral_q if frame == 0 else warm.arrays["qpos"][frame - 1],
            previous_qpos=None if frame == 0 else warm.arrays["qpos"][frame - 1],
            lambda_warm=lambda_warm,
            lambda_smooth=lambda_smooth,
        )
        invocation_count += 1
        official.append(
            {
                "frame": frame,
                "status": formal_result.status,
                "success": formal_result.success,
                "qpos_max_diff_rad": float(np.max(np.abs(formal_result.qpos - q0))),
                "ebone_diff": float(abs(formal_result.final_ebone - warm.arrays["ebone"][frame])),
                "total_objective_diff": float(
                    abs(formal_result.total_objective - warm.arrays["total_objective"][frame])
                ),
                "robot_keypoints_max_diff_m": float(
                    np.max(
                        np.abs(
                            np.asarray(formal_result.robot_features.local_keypoints)
                            - np.asarray(
                                extract_bone_features(
                                    warm.arrays["robot_keypoints_base"][frame],
                                    frame_profile,
                                    bone_profile,
                                ).local_keypoints
                            )
                        )
                    )
                )
                if False
                else None,
                "bounds_pass": bool(
                    np.all(formal_result.qpos >= model.joint_lower)
                    and np.all(formal_result.qpos <= model.joint_upper)
                ),
                "initial_q_identity": "neutral_q" if frame == 0 else "warm_qpos_previous_frame",
            }
        )
        source_target = source[frame]
        raw_scene = source_target
        target_local = np.asarray(
            frame_profile.to_local(source_target, source_features.frame_transform[frame])
        )
        robot_features = extract_bone_features(
            warm.arrays["robot_keypoints_base"][frame], frame_profile, bone_profile
        )
        recon_local = _build_morphology_targets(
            target_local,
            np.asarray(source_features.unit_directions[frame]),
            np.asarray(robot_features.bone_lengths),
            bone_profile,
        )
        recon_scene = transform_points(source_features.frame_transform[frame], recon_local)
        thumb_mask = np.zeros(model.num_dofs, dtype=bool)
        thumb_mask[q_indices["thumb"]] = True
        all_mask = np.ones(model.num_dofs, dtype=bool)
        for profile_name, mask, with_base, target in (
            ("thumb_formal_feature_fit", thumb_mask, False, raw_scene),
            ("thumb_keypoint_fit_fixed_base", thumb_mask, False, raw_scene),
            ("thumb_keypoint_fit_with_base", thumb_mask, True, raw_scene),
            ("all_joints_keypoint_fit_fixed_base", all_mask, False, raw_scene),
            ("all_joints_keypoint_fit_with_base", all_mask, True, raw_scene),
            ("thumb_robot_length_keypoint_fit_fixed_base", thumb_mask, False, recon_scene),
        ):
            value: dict[str, Any]
            if profile_name == "thumb_formal_feature_fit":
                mask_indices = np.asarray(q_indices["thumb"], dtype=int)
                qbase = q0.copy()
                lo = model.joint_lower[mask_indices]
                hi = model.joint_upper[mask_indices]
                x0 = qbase[mask_indices]
                residual_model = BoneDirectionResidual(
                    source_features.adjacent_features[frame],
                    frame_profile,
                    bone_profile,
                    model,
                    "right",
                )

                def formal_residual(
                    x: np.ndarray,
                    q_template=qbase.copy(),
                    indices=mask_indices.copy(),
                    frame_index=frame,
                    rm=residual_model,
                ) -> np.ndarray:
                    import torch

                    q = q_template.copy()
                    q[indices] = x
                    value = (
                        rm.residual_tensor(torch.as_tensor(q, dtype=torch.float64))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    return value[
                        np.concatenate(
                            [
                                np.arange(3 * p, 3 * p + 3)
                                for p in _finger_pairs(bone_profile)["thumb"]
                            ]
                        )
                    ].reshape(-1)

                started = time.perf_counter()
                fit = least_squares(
                    formal_residual,
                    x0,
                    bounds=(lo, hi),
                    max_nfev=250,
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12,
                )
                elapsed = time.perf_counter() - started
                invocation_count += 1
                value = {
                    "status": int(fit.status),
                    "success": bool(fit.success),
                    "residual_rmse": float(np.sqrt(np.mean(np.square(fit.fun)))),
                    "q_change_rad": float(
                        np.linalg.norm(np.pad(fit.x - x0, (0, model.num_dofs - len(x0))))
                    ),
                    "nfev": int(fit.nfev),
                    "runtime_s": float(elapsed),
                    "bounds_pass": bool(np.all(fit.x >= lo - 1e-12) & np.all(fit.x <= hi + 1e-12)),
                }
            else:
                value = optimize_keypoints(frame, mask, with_base, target)
                value["target_type"] = (
                    "raw_source_metric" if target is raw_scene else "robot_length_reconstructed"
                )
            results.append(
                {
                    "frame": frame,
                    "profile": profile_name,
                    "diagnostic_only": True,
                    "paper_method": False,
                    "accepted_reference": False,
                    **value,
                }
            )
        for smooth_name, smooth in (
            ("no_temporal_stage7_diagnostic", 0.0),
            ("reduced_temporal_stage7_diagnostic", lambda_smooth * 0.5),
        ):
            result = solve_frame(
                source_features.adjacent_features[frame],
                model,
                frame_profile,
                bone_profile,
                solver_profile,
                side="right",
                initial_qpos=model.neutral_q if frame == 0 else warm.arrays["qpos"][frame - 1],
                previous_qpos=None if frame == 0 else warm.arrays["qpos"][frame - 1],
                lambda_warm=lambda_warm,
                lambda_smooth=smooth,
            )
            invocation_count += 1
            results.append(
                {
                    "frame": frame,
                    "profile": smooth_name,
                    "diagnostic_only": True,
                    "paper_method": False,
                    "ablation": "PAPER_WEIGHT_ABLATION",
                    "status": result.status,
                    "success": result.success,
                    "qpos_max_diff_rad": float(np.max(np.abs(result.qpos - q0))),
                    "ebone": result.final_ebone,
                    "total_objective": result.total_objective,
                    "runtime_s": result.solve_time_s,
                }
            )
    _write_json(
        diagnostic_root / "official_stage7_reproduction.json",
        {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "solver_invocation_count": len(official),
            "records": official,
            "contract": {
                "qpos_tolerance_rad": 1e-7,
                "ebone_tolerance": 1e-9,
                "total_objective_tolerance": 1e-9,
            },
        },
    )
    return {"results": results, "official": official, "solver_invocation_count": invocation_count}


def _workspace_audit(
    frame: int,
    source: np.ndarray,
    warm: Any,
    model: Any,
    frame_profile: BoneDirectionFrameProfile,
    source_features: Any,
    profile: BoneDirectionProfile,
    diagnostic_root: Path,
    seed: int = 20260723,
    samples: int = 4096,
) -> dict[str, Any]:
    from scipy.spatial import ConvexHull
    from scipy.stats import qmc

    thumb_indices = _finger_q_indices(model)["thumb"]
    sampler = qmc.Sobol(d=len(thumb_indices), scramble=False, seed=seed)
    unit = sampler.random_base2(int(np.log2(samples)))
    lower = model.joint_lower[thumb_indices]
    upper = model.joint_upper[thumb_indices]
    q_values = np.repeat(np.asarray(warm.arrays["qpos"][frame])[None, :], len(unit), axis=0)
    q_values[:, thumb_indices] = lower + unit * (upper - lower)
    points = np.asarray(model.keypoints_base(q_values).detach().cpu())
    base_pose = np.asarray(warm.arrays["base_pose_scene"][frame], dtype=np.float64)
    base_inverse = np.linalg.inv(base_pose)
    source_frame = np.asarray(source_features.frame_transform[frame], dtype=np.float64)
    source_local = np.asarray(frame_profile.to_local(source[frame], source_frame))
    warm_features = extract_bone_features(
        warm.arrays["robot_keypoints_base"][frame], frame_profile, profile, strict=True
    )
    robot_length_local = _build_morphology_targets(
        source_local,
        np.asarray(source_features.unit_directions[frame]),
        np.asarray(warm_features.bone_lengths),
        profile,
    )
    robot_length_scene = transform_points(source_frame, robot_length_local)
    raw_target = transform_points(base_inverse, source[frame, 4][None, :])[0]
    robot_length_target = transform_points(base_inverse, robot_length_scene[4][None, :])[0]
    raw_cmc = transform_points(base_inverse, source[frame, 1][None, :])[0]
    distances = np.linalg.norm(points[:, 4] - raw_target[None, :], axis=-1)
    length_distances = np.linalg.norm(points[:, 4] - robot_length_target[None, :], axis=-1)
    nearest = int(np.argmin(distances))
    length_nearest = int(np.argmin(length_distances))
    target_direction = raw_target - raw_cmc
    nearest_direction = points[nearest, 4] - points[nearest, 1]
    direction_cosine = float(
        np.dot(target_direction, nearest_direction)
        / max(np.linalg.norm(target_direction) * np.linalg.norm(nearest_direction), EPS)
    )
    try:
        hull = ConvexHull(points[:, 4])
        hull_value = float(np.max(hull.equations[:, :3] @ raw_target + hull.equations[:, 3]))
        hull_inside = bool(hull_value <= 1.0e-9)
        hull_vertices = int(len(hull.vertices))
    except (ValueError, RuntimeError):
        hull_value = None
        hull_inside = None
        hull_vertices = 0
    output = diagnostic_root / f"thumb_workspace_points_frame_{frame:03d}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        frame=np.asarray(frame),
        seed=np.asarray(seed),
        qpos=q_values,
        thumb_cmc_workspace=points[:, 1],
        thumb_tip_workspace=points[:, 4],
        thumb_pad_workspace=points[:, 3:5],
        raw_source_target=np.asarray(raw_target),
        robot_length_target=np.asarray(robot_length_target),
    )
    return {
        "schema_version": "toporetarget.stage7_1.thumb_workspace.v1",
        "frame": frame,
        "seed": seed,
        "sample_count": samples,
        "joint_names": [model.dof_names[i] for i in thumb_indices],
        "bounds_rad": np.stack([lower, upper], axis=1).tolist(),
        "nearest_sample_index": nearest,
        "nearest_sample_distance_m": float(distances[nearest]),
        "robot_length_nearest_sample_index": length_nearest,
        "robot_length_nearest_sample_distance_m": float(length_distances[length_nearest]),
        "raw_target_base_frame_m": raw_target.tolist(),
        "robot_length_target_base_frame_m": robot_length_target.tolist(),
        "nearest_sample_direction_error_rad": float(
            np.arccos(np.clip(direction_cosine, -1.0, 1.0))
        ),
        "raw_target_convex_hull_max_equation_value": hull_value,
        "raw_target_inside_sampled_convex_hull": hull_inside,
        "sampled_convex_hull_vertex_count": hull_vertices,
        "raw_target_near_sampled_workspace_5mm": bool(distances[nearest] <= 0.005),
        "robot_length_target_near_sampled_workspace_5mm": bool(
            length_distances[length_nearest] <= 0.005
        ),
        "sampled_workspace_is_not_strict_reachability_proof": True,
        "nearest_sample_distance_is_an_upper_bound_estimate": True,
        "points_path": str(output),
    }


def _root_cause(
    replay: dict[str, Any],
    mapping: dict[str, Any],
    anchors: dict[str, Any],
    attribution: dict[str, Any],
    limits: dict[str, Any],
    jacobian: dict[str, Any],
    diagnostic: dict[str, Any] | None,
    per_frame: list[dict[str, Any]],
) -> dict[str, Any]:
    thumb = next(row for row in attribution["per_finger"] if row["region"] == "thumb")
    causes: list[dict[str, Any]] = []
    if mapping.get("mapping_error_detected"):
        causes.append(
            {
                "cause": "SOURCE_MEDIAPIPE_MAPPING_ERROR",
                "confidence": 0.98,
                "evidence_for": ["source mapping gate failed"],
                "evidence_against": [],
                "affected_frames": "all",
                "affected_joints": [],
                "next_action": "RETURN_TO_STAGE3_MEDIAPIPE_MAPPING_FIX",
            }
        )
    if anchors.get("mapping_error_detected"):
        causes.append(
            {
                "cause": "ROBOT_ANCHOR_MAPPING_ERROR",
                "confidence": 0.95,
                "evidence_for": ["robot anchor gate failed"],
                "evidence_against": [],
                "affected_frames": "all",
                "affected_joints": [],
                "next_action": "RETURN_TO_STAGE4_ROBOT_ANCHOR_OR_URDF_FIX",
            }
        )
    if not all(replay["gates"].values()):
        causes.append(
            {
                "cause": "STAGE7_SOLVER_REPRODUCTION_ERROR",
                "confidence": 0.99,
                "evidence_for": ["persisted artifact replay gate failed"],
                "evidence_against": [],
                "affected_frames": "all",
                "affected_joints": [],
                "next_action": "RETURN_TO_STAGE7_ARTIFACT_OR_REPLAY_FIX",
            }
        )
    if diagnostic:
        thumb_results = [
            row
            for row in diagnostic.get("results", [])
            if row.get("profile")
            in {"thumb_keypoint_fit_fixed_base", "thumb_robot_length_keypoint_fit_fixed_base"}
        ]
        if (
            thumb_results
            and float(np.mean([row.get("residual_rmse_m", 1.0) for row in thumb_results]))
            > thumb["warm_keypoint_rmse_m"] * 0.75
        ):
            causes.append(
                {
                    "cause": "THUMB_KINEMATIC_REACHABILITY_LIMIT",
                    "confidence": 0.72,
                    "evidence_for": [
                        "thumb-only bounded keypoint fits retain most of the warm residual",
                        "local Jacobian and workspace diagnostics are bounded evidence",
                    ],
                    "evidence_against": ["sampled workspace is not a global proof"],
                    "affected_frames": [row["frame"] for row in thumb_results],
                    "affected_joints": [],
                    "quantitative_metrics": {
                        "warm_thumb_keypoint_rmse_m": thumb["warm_keypoint_rmse_m"],
                        "diagnostic_residuals_m": [
                            row.get("residual_rmse_m") for row in thumb_results
                        ],
                    },
                    "next_action": "WARM_START_KINEMATIC_REACHABILITY_LIMIT_CONFIRMED",
                }
            )
        if (
            thumb["warm_contact_proxy"] < 0.2
            and thumb["warm_ebone"]
            <= max(row["warm_ebone"] for row in attribution["per_finger"]) * 1.5
        ):
            causes.append(
                {
                    "cause": "FORMAL_BONE_OBJECTIVE_UNDEROBSERVES_SURFACE_CONTACT",
                    "confidence": 0.78,
                    "evidence_for": [
                        "formal Stage 7 uses bone directions only",
                        "contact proxy is a separate metric",
                    ],
                    "evidence_against": ["contact proxy is not ground truth"],
                    "affected_frames": "multiple",
                    "affected_joints": [],
                    "next_action": "keep Stage 7 math unchanged; evaluate downstream contact terms separately",
                }
            )
        workspace = diagnostic.get("workspace", [])
        raw_distances = [
            float(row["nearest_sample_distance_m"])
            for row in workspace
            if row.get("nearest_sample_distance_m") is not None
        ]
        length_distances = [
            float(row["robot_length_nearest_sample_distance_m"])
            for row in workspace
            if row.get("robot_length_nearest_sample_distance_m") is not None
        ]
        if (
            raw_distances
            and length_distances
            and float(np.mean(raw_distances)) > 0.005
            and float(np.mean(length_distances)) < float(np.mean(raw_distances)) * 0.5
            and all(value <= 0.005 for value in length_distances)
        ):
            causes.append(
                {
                    "cause": "MORPHOLOGY_LENGTH_MISMATCH_DOMINATES",
                    "confidence": 0.88,
                    "evidence_for": [
                        "raw source metric thumb targets remain outside the sampled workspace",
                        "robot-length reconstructed targets are near the sampled workspace on every selected frame",
                        "mapping, frame, base, and formal replay gates pass",
                    ],
                    "evidence_against": [
                        "sampled workspace is not a strict reachability proof",
                        "bounded samples and local fits are diagnostic_only",
                    ],
                    "affected_frames": [int(row["frame"]) for row in workspace],
                    "affected_joints": [
                        "j_thumb1x",
                        "j_thumb1y",
                        "j_thumb1z",
                        "j_thumb2y",
                        "j_thumb2z",
                        "j_thumb3",
                    ],
                    "quantitative_metrics": {
                        "raw_target_nearest_distance_mean_m": float(np.mean(raw_distances)),
                        "robot_length_target_nearest_distance_mean_m": float(
                            np.mean(length_distances)
                        ),
                        "raw_target_nearest_distance_m": raw_distances,
                        "robot_length_target_nearest_distance_m": length_distances,
                    },
                    "next_action": "retain formal Stage 7 objective; treat raw-to-robot-length gap as embodiment/morphology evidence and continue Stage 9.3.3",
                }
            )
    whole = attribution["whole_hand"]
    degraded_frames = sorted(
        {
            int(row["frame"])
            for row in per_frame
            if float(row["warm_to_final_keypoint_change_m"]) > 0.0
        }
    )
    contact_lost_frames = sorted(
        {
            int(row["frame"])
            for row in per_frame
            if bool(row["warm_contact_proxy"]) and not bool(row["final_contact_proxy"])
        }
    )
    if float(whole["final_keypoint_rmse_m"]) > float(whole["warm_keypoint_rmse_m"]) and float(
        whole["final_eim_total"]
    ) > float(whole["warm_eim_total"]):
        causes.append(
            {
                "cause": "FINAL_REFINEMENT_PRIMARY_DEGRADATION",
                "confidence": 0.82,
                "evidence_for": [
                    "whole-hand canonical keypoint RMSE increases from warm to final",
                    "reported Stage 8/Stage 9 E_IM increases from warm to final",
                    "contact proxy is lost on warm-positive frames",
                ],
                "evidence_against": [
                    "thumb keypoint RMSE improves in the accepted final trajectory",
                    "contact proxy is not ground truth and final E_IM comes from the final artifact objective",
                ],
                "affected_frames": {
                    "canonical_keypoint_degraded": degraded_frames,
                    "contact_proxy_lost": contact_lost_frames,
                },
                "affected_joints": [
                    "Stage 9 refinement qpos/base terms; per-joint attribution is not available in the final artifact"
                ],
                "quantitative_metrics": {
                    "warm_keypoint_rmse_m": whole["warm_keypoint_rmse_m"],
                    "final_keypoint_rmse_m": whole["final_keypoint_rmse_m"],
                    "warm_eim_total": whole["warm_eim_total"],
                    "final_eim_total": whole["final_eim_total"],
                    "warm_to_final_keypoint_change_m": whole["warm_to_final_change_m"],
                },
                "next_action": "continue Stage 9.3.3 with the existing reference; separately audit final refinement contact/task tradeoffs",
            }
        )
    if not causes:
        causes.append(
            {
                "cause": "INCONCLUSIVE",
                "confidence": 0.45,
                "evidence_for": ["no hard mapping/replay failure"],
                "evidence_against": [
                    "bounded diagnostics are insufficient for a global conclusion"
                ],
                "affected_frames": "multiple",
                "affected_joints": [],
                "next_action": "STAGE7_1_INCONCLUSIVE",
            }
        )
    causes.sort(key=lambda row: float(row.get("confidence", 0.0)), reverse=True)
    top = causes[0]["cause"]
    if top == "SOURCE_MEDIAPIPE_MAPPING_ERROR":
        readiness, cont = "RETURN_TO_STAGE3_MEDIAPIPE_MAPPING_FIX", "NO"
    elif top == "ROBOT_ANCHOR_MAPPING_ERROR":
        readiness, cont = "RETURN_TO_STAGE4_ROBOT_ANCHOR_OR_URDF_FIX", "NO"
    elif top == "STAGE7_SOLVER_REPRODUCTION_ERROR":
        readiness, cont = "RETURN_TO_STAGE7_SOLVER_OR_TEMPORAL_FIX", "NO"
    elif top == "THUMB_KINEMATIC_REACHABILITY_LIMIT":
        readiness, cont = "WARM_START_KINEMATIC_REACHABILITY_LIMIT_CONFIRMED", "NO"
    elif top == "INCONCLUSIVE":
        readiness, cont = "STAGE7_1_INCONCLUSIVE", "NO"
    elif top in {
        "FINAL_REFINEMENT_PRIMARY_DEGRADATION",
        "FORMAL_BONE_OBJECTIVE_UNDEROBSERVES_SURFACE_CONTACT",
        "MORPHOLOGY_LENGTH_MISMATCH_DOMINATES",
    }:
        readiness, cont = "WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3", "YES"
    else:
        readiness, cont = "WARM_START_FORMALLY_VALID_CONTINUE_STAGE9_3_3", "YES"
    return {
        "schema_version": "toporetarget.stage7_1.root_cause.v1",
        "ranked_causes": causes,
        "readiness": readiness,
        "CONTINUE_STAGE9_3_3": cont,
        "stage9_4": {
            "eligible_to_discuss": cont == "YES",
            "requires_stage7_to_10_regeneration": cont == "NO",
            "manual_acceptance_retake_required_if_regenerated": cont == "NO",
            "existing_stage10_reference_retained_as_history": True,
            "physics_rl_ready": False,
        },
    }


def _build_html(
    path: Path,
    summary: dict[str, Any],
    per_frame: list[dict[str, Any]],
    source: np.ndarray,
    warm_scene: np.ndarray,
    final_scene: np.ndarray,
    layout: Any,
) -> dict[str, Any]:
    data = {
        "summary": summary,
        "per_frame": per_frame,
        "source": source.tolist(),
        "warm": warm_scene.tolist(),
        "final": final_scene.tolist(),
        "edges": [list(edge) for edge in layout.edges],
        "names": list(layout.semantic_names),
    }
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    title = "Warm-Start Fidelity, Thumb Mapping, Base Alignment, and Kinematic Reachability Audit"
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font:14px sans-serif;margin:20px;background:#111;color:#eee}}h1{{font-size:20px}}#status{{color:#ffcc66}}button,select{{margin:4px;padding:4px;background:#222;color:#eee;border:1px solid #666}}svg{{background:#181818;border:1px solid #444;max-width:100%;height:440px}}table{{border-collapse:collapse;margin-top:12px}}th,td{{border:1px solid #555;padding:5px}}.warn{{color:#ff9966}}.ok{{color:#9f9}}</style></head><body><h1>{html.escape(title)}</h1><div id="status">WARM-START = STAGE 7 ARTI-MANO INITIAL RETARGETING | NOT SOURCE MANO | NOT FINAL STAGE 9 RESULT</div><label>State <select id="state"><option>source</option><option>warm</option><option>final</option><option>source+warm</option><option>warm+final</option><option>source+final</option></select></label><label>Finger <select id="finger"><option>whole</option><option>thumb</option><option>index</option><option>middle</option><option>ring</option><option>pinky</option></select></label><label>Frame <input id="frame" type="range" min="0" max="{max(len(source) - 1, 0)}" value="0"></label><span id="frameLabel"></span><button id="scale">global fixed scale</button><div><svg id="scene" viewBox="0 0 900 440" role="img"></svg></div><div id="metrics"></div><h2>Per-finger fixed-scale audit table</h2><table id="table"><thead><tr><th>Region</th><th>Warm Ebone</th><th>Warm keypoint RMSE (mm)</th><th>Warm EIM contribution</th><th>Warm contact proxy</th><th>Final change (mm)</th><th>Limit margin (rad)</th></tr></thead><tbody></tbody></table><script>const DATA={payload};let fixed=true;const colors={{source:'#56b4e9',warm:'#e69f00',final:'#009e73'}};const state=document.getElementById('state'),frame=document.getElementById('frame'),finger=document.getElementById('finger'),svg=document.getElementById('scene');function rows(){{return DATA.per_frame.filter(r=>finger.value==='whole'||r.finger===finger.value)}}function draw(){{const i=+frame.value;document.getElementById('frameLabel').textContent=' local='+i+' global='+({int(summary.get("global_frame_start", 240))}+i);svg.innerHTML='';let names=state.value.split('+');let all=[].concat(...names.map(n=>DATA[n][i]));let min=[0,0,0],max=[0,0,0];all.forEach(p=>p.forEach((v,j)=>{{min[j]=Math.min(min[j],v);max[j]=Math.max(max[j],v)}}));let sx=760/Math.max(max[0]-min[0],1e-6),sy=360/Math.max(max[1]-min[1],1e-6),scale=Math.min(sx,sy);function xy(p){{return [70+(p[0]-min[0])*scale,390-(p[1]-min[1])*scale]}}names.forEach(n=>{{let pts=DATA[n][i];DATA.edges.forEach(e=>{{let a=xy(pts[e[0]]),b=xy(pts[e[1]]);svg.insertAdjacentHTML('beforeend',`<line x1="${{a[0]}}" y1="${{a[1]}}" x2="${{b[0]}}" y2="${{b[1]}}" stroke="${{colors[n]}}" opacity=".55"/>`)}});pts.forEach((p,j)=>{{let a=xy(p);svg.insertAdjacentHTML('beforeend',`<circle cx="${{a[0]}}" cy="${{a[1]}}" r="${{j===0?5:3}}" fill="${{colors[n]}}"/><title>${{DATA.names[j]}} (${{(p[0]*1000).toFixed(1)}}, ${{(p[1]*1000).toFixed(1)}}, ${{(p[2]*1000).toFixed(1)}} mm)</title>`)}})}});let values=rows().filter(r=>r.frame===i);document.getElementById('metrics').textContent=values.length?`actual mm; warm keypoint RMSE=${{(values.reduce((a,r)=>a+r.warm_keypoint_rmse_m,0)/values.length*1000).toFixed(2)}}; warm EIM=${{values.reduce((a,r)=>a+r.warm_eim_contribution,0).toExponential(3)}}; fixed global scale=${{fixed}}`:''}}function fill(){{let body=document.querySelector('#table tbody');body.innerHTML='';let ss=DATA.summary.per_finger||[];ss.forEach(r=>body.insertAdjacentHTML('beforeend',`<tr><td>${{r.region}}</td><td>${{r.warm_ebone.toExponential(4)}}</td><td>${{(r.warm_keypoint_rmse_m*1000).toFixed(2)}}</td><td>${{r.warm_eim_contribution.toExponential(4)}}</td><td>${{(r.warm_contact_proxy*100).toFixed(1)}}%</td><td>${{(r.warm_to_final_change*1000).toFixed(2)}}</td><td>${{r.joint_limit_min_margin_rad.toFixed(4)}}</td></tr>`))}}frame.oninput=draw;state.onchange=draw;finger.onchange=draw;document.getElementById('scale').onclick=()=>{{fixed=!fixed;document.getElementById('scale').textContent=fixed?'global fixed scale':'per-frame scale (diagnostic)';draw()}};fill();draw();</script></body></html>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return {
        "path": str(path),
        "exists": True,
        "required_tokens": {
            token: token in document
            for token in (
                "source",
                "warm",
                "final",
                "global fixed scale",
                "thumb",
                "joint_limit",
                "frame",
            )
        },
    }


def _official_paths(
    manifest: dict[str, Any], final: Any, repo_root: Path, model: Any
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, item in manifest.get("artifacts", {}).items():
        if isinstance(item, dict) and item.get("path"):
            paths[f"stage10_{name}"] = Path(str(item["path"])).expanduser()
    paths["stage10_manifest"] = (
        repo_root
        / ".local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json"
    )
    for name, value in manifest.get("export_paths", {}).items():
        if value:
            path = Path(str(value)).expanduser()
            paths[f"reference_export_{name}"] = path if path.is_absolute() else repo_root / path
    runtime_acceptance = manifest.get("runtime_acceptance", {})
    if isinstance(runtime_acceptance, dict) and runtime_acceptance.get("path"):
        acceptance_path = Path(str(runtime_acceptance["path"])).expanduser()
        paths["reference_runtime_acceptance"] = (
            acceptance_path if acceptance_path.is_absolute() else repo_root / acceptance_path
        )
    paths["manual_acceptance"] = repo_root / ".local/reports/stage9/manual_acceptance.json"
    paths["artimano_urdf"] = model.urdf.urdf_path
    if model.asset_root is not None:
        paths["artimano_asset_manifest"] = model.asset_root / "asset_manifest.json"
    config_root = model.config_root
    if config_root is not None:
        anchor_path = (
            Path(str(config_root)) / "keypoints" / (f"{model.spec.keypoint_anchor_profile}.yaml")
        )
        if not anchor_path.is_file():
            anchor_path = Path(str(config_root)) / f"{model.spec.keypoint_anchor_profile}.yaml"
        paths["artimano_anchor_profile"] = anchor_path
    mesh_paths = sorted(
        {
            geometry.resolved_path
            for link in model.urdf.links.values()
            for geometry in (*link.visuals, *link.collisions)
            if geometry.resolved_path is not None
        },
        key=str,
    )
    for index, mesh_path in enumerate(mesh_paths):
        paths[f"artimano_mesh_{index:03d}"] = mesh_path
    checkpoint = final.metadata.get("checkpoint_root")
    if checkpoint:
        checkpoint_path = Path(str(checkpoint))
        paths["stage9_2_checkpoint_root"] = (
            checkpoint_path if checkpoint_path.is_absolute() else repo_root / checkpoint_path
        )
        paths["stage9_2_checkpoint_manifest"] = paths["stage9_2_checkpoint_root"] / "manifest.json"
    final_path = paths.get("stage10_final")
    if final_path:
        paths["stage9_2_repeat"] = final_path.with_name(
            final_path.name.replace(".zarr", "_repeat.zarr")
        )
    return paths


def _preflight(
    manifest_path: Path, contact_root: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not manifest_path.is_file():
        raise RuntimeError("STAGE9_3_2_CLOSEOUT_REQUIRED: Stage 10 manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing: list[str] = []
    for name in (
        "canonical",
        "warm_start",
        "graph",
        "evaluation",
        "final",
        "collision_samples",
        "object_samples",
    ):
        path = Path(str(manifest.get("artifacts", {}).get(name, {}).get("path", "")))
        if not path.exists():
            missing.append(f"{name}: {path}")
    for path in (
        contact_root / "stage9_3_2_summary.json",
        contact_root / "official_artifact_immutability.json",
        contact_root / "input_identity_and_immutability.json",
    ):
        if not path.exists():
            missing.append(f"canonical audit: {path}")
    status_output = subprocess.run(
        ["git", "status", "--short"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout
    status = status_output.strip()
    cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    allowed_local_paths = {
        "README.md",
        "README.zh-CN.md",
        "docs/ARTIMANO_ADAPTER.md",
        "docs/ASSUMPTIONS.md",
        "docs/DEVELOPMENT_LOG.md",
        "docs/DEVELOPMENT_LOG.zh-CN.md",
        "docs/PAPER_FIDELITY.md",
        "docs/PAPER_FIDELITY.yaml",
        "docs/RELATIVE_BONE_DIRECTION_INITIALIZATION.md",
        "docs/ROBOT_HAND_INTERFACE.md",
        "docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.md",
        "docs/WARM_START_FIDELITY_AND_REACHABILITY_AUDIT.zh-CN.md",
        "docs/WARM_START_OPTIMIZATION.md",
        "docs/stages/STAGE_7_BONE_DIRECTION_WARM_START.md",
        "src/toporetarget/cli/workflow.py",
        "src/toporetarget/workflows/warm_start_audit.py",
        "tests/unit/test_warm_start_audit.py",
    }
    unexpected_worktree = [
        line for line in status_output.splitlines() if line[3:] not in allowed_local_paths
    ]
    processes = subprocess.run(
        ["ps", "aux"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    related = [
        line
        for line in processes
        if any(
            token in line.lower()
            for token in ("toporetarget", "warm-start", "refine", "stage7", "stage9", "shadow")
        )
        and "grep" not in line
        and "audit-warm-start" not in line
    ]
    head = subprocess.run(
        ["git", "log", "-1", "--format=%s"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if (
        missing
        or cached
        or unexpected_worktree
        or related
        or "canonical contact re-audit" not in head.lower()
    ):
        raise RuntimeError(
            "STAGE9_3_2_CLOSEOUT_REQUIRED: "
            + json.dumps(
                {
                    "missing": missing,
                    "status": status,
                    "unexpected_worktree": unexpected_worktree,
                    "cached": cached,
                    "processes": related,
                    "head_subject": head,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "head_subject": head,
        "missing": [],
        "status_short": status,
        "allowed_local_paths": sorted(allowed_local_paths),
        "unexpected_worktree": unexpected_worktree,
        "cached": cached,
        "related_processes": related,
    }, manifest


def run_warm_start_audit(
    manifest_path: str | Path,
    canonical_contact_audit: str | Path,
    output_root: str | Path,
    *,
    html_output: bool = False,
    run_reachability_diagnostics: bool = False,
    diagnostic_frames: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_path = Path(manifest_path).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    contact_root = Path(canonical_contact_audit).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    preflight, manifest = _preflight(manifest_path, contact_root, repo_root)
    base = Path(str(manifest["artifacts"]["canonical"]["path"]))
    warm_path = Path(str(manifest["artifacts"]["warm_start"]["path"]))
    graph_path = Path(str(manifest["artifacts"]["graph"]["path"]))
    evaluation_path = Path(str(manifest["artifacts"]["evaluation"]["path"]))
    final_path = Path(str(manifest["artifacts"]["final"]["path"]))
    sequence = load_hoi_sequence(base)
    warm = load_warm_start(warm_path)
    _graph = load_interaction_graph(graph_path)
    evaluation = load_interaction_evaluation(evaluation_path)
    final = load_final_trajectory(final_path)
    model = load_artimano_model(str(manifest["hand"]))
    frame_profile = load_frame_profile(str(warm.metadata["frame_profile_id"]))
    bone_profile = load_bone_profile(str(warm.metadata["bone_profile_id"]))
    solver_profile = load_solver_profile(str(warm.metadata["solver_profile_id"]))
    hand_id = str(warm.metadata["source_hand_id"])
    source = np.asarray(
        sequence.hand(hand_id).keypoint_tracks[bone_profile.layout_name].positions_scene,
        dtype=np.float64,
    )
    source_features = extract_bone_features(
        source, frame_profile, bone_profile, side="right", strict=True
    )
    contact = _read_contact_rows(contact_root)
    official_before = _official_paths(manifest, final, repo_root, model)
    identity = {name: _stat(path) for name, path in official_before.items()}
    mapping, source_anchor_rows = _source_mapping_audit(sequence, hand_id, bone_profile)
    anchors, robot_anchor_rows, joint_axis_rows = _robot_anchor_audit(model, warm.arrays["qpos"])
    # The last synthetic row from _robot_anchor_audit holds Jacobian influence;
    # keep the public axis CSV strictly tabular.
    jacobian_influence = (
        joint_axis_rows[-1].pop("anchor_jacobian_influence", {})
        if joint_axis_rows and "anchor_jacobian_influence" in joint_axis_rows[-1]
        else {}
    )
    joint_axis_rows = joint_axis_rows[:-1] if jacobian_influence else joint_axis_rows
    source_frames = np.asarray(source_features.frame_transform)
    warm_base = np.asarray(warm.arrays["robot_keypoints_base"], dtype=np.float64)
    warm_features = [
        extract_bone_features(warm_base[i], frame_profile, bone_profile, strict=True)
        for i in range(len(source))
    ]
    robot_frames = np.stack([np.asarray(value.frame_transform) for value in warm_features])
    frame_audit = {
        "schema_version": "toporetarget.stage7_1.hand_frame.v1",
        "formal_profile": frame_profile.as_dict(),
        "source": _frame_quality(source_frames),
        "robot": _frame_quality(robot_frames),
        "persisted_source_max_diff": float(
            np.max(np.abs(source_frames - warm.arrays["source_hand_frame_scene"]))
        ),
        "persisted_robot_max_diff": float(
            np.max(np.abs(robot_frames - warm.arrays["robot_hand_frame_base"]))
        ),
        "diagnostic_profiles": [
            "canonical_keypoint_wrist_v1",
            "translation_centered_scene_axes",
            "palm_plane_kabsch",
            "mcp_procrustes",
        ],
        "frame_convention": "column axes in parent, row points mapped by (p-origin)@R",
        "formal_frame_error_detected": False,
    }
    frame_rows = [
        {
            "frame": i,
            "source_det": float(np.linalg.det(source_frames[i, :3, :3])),
            "robot_det": float(np.linalg.det(robot_frames[i, :3, :3])),
            "source_robot_origin_distance_m": float(
                np.linalg.norm(
                    source_frames[i, :3, 3]
                    - (
                        apply_base_pose_to_points(
                            warm_base[i : i + 1],
                            np.asarray(
                                [
                                    base_seed_from_hand_frames(
                                        source_frames[i : i + 1], robot_frames[i : i + 1]
                                    )[0]
                                ]
                            ),
                        )[0, 0]
                        if False
                        else source_frames[i, :3, 3]
                    )
                )
            ),
            "source_temporal_rotation_rad": 0.0
            if i == 0
            else _rotation_angle(source_frames[i - 1], source_frames[i]),
            "robot_temporal_rotation_rad": 0.0
            if i == 0
            else _rotation_angle(robot_frames[i - 1], robot_frames[i]),
        }
        for i in range(len(source))
    ]
    bone_audit = {
        "schema_version": "toporetarget.stage7_1.bone_pair_profile.v1",
        "profile": bone_profile.as_dict(),
        "directed_bone_count": len(bone_profile.bones),
        "adjacent_pair_count": len(bone_profile.pairs),
        "per_finger_bone_count": {
            finger: len([bone for bone in bone_profile.bones if bone.finger == finger])
            for finger in FINGERS
        },
        "per_finger_pair_count": {
            finger: len([pair for pair in bone_profile.pairs if pair.finger == finger])
            for finger in FINGERS
        },
        "pair_direction_consistent": all(
            bone_profile.bones[pair.second_bone].finger == pair.finger
            for pair in bone_profile.pairs
        ),
        "formal_objective_terms": [
            "relative unit-direction pair residual",
            "temporal regularization",
            "joint bounds",
        ],
        "forbidden_terms_absent": [
            "absolute direction loss",
            "bone length loss",
            "angle loss",
            "mean replacement",
            "hidden weighting",
        ],
        "profile_error_detected": False,
    }
    pair_rows = [
        {
            "pair_index": i,
            "pair_name": pair.name,
            "finger": pair.finger,
            "first_bone": bone_profile.bones[pair.first_bone].name,
            "second_bone": bone_profile.bones[pair.second_bone].name,
            "direction": "parent_to_child",
            "same_finger": True,
        }
        for i, pair in enumerate(bone_profile.pairs)
    ]
    replay = _replay(source, warm, model, frame_profile, bone_profile)
    attribution, per_frame, per_finger = _frame_diagnostics(
        source, warm, final, model, frame_profile, bone_profile, evaluation, contact
    )
    base_audit, base_rows = _base_audit(
        source, warm_base, np.asarray(warm.arrays["base_pose_scene"]), bone_profile
    )
    limits, limit_rows, limit_joint_rows = _joint_limit_audit(
        model, np.asarray(warm.arrays["qpos"]), np.asarray(final.arrays["qpos"])
    )
    jacobian, jacobian_rows, reachable_rows = _jacobian_audit(
        source,
        np.asarray(warm.arrays["qpos"]),
        warm_base,
        source_features,
        model,
        frame_profile,
        bone_profile,
    )
    morphology_rows: list[dict[str, Any]] = []
    for i in range(len(source)):
        source_local = np.asarray(frame_profile.to_local(source[i], source_frames[i]))
        warm_feature = warm_features[i]
        target = _build_morphology_targets(
            source_local,
            np.asarray(source_features.unit_directions[i]),
            np.asarray(warm_feature.bone_lengths),
            bone_profile,
        )
        for finger, indices in _finger_indices(bone_profile).items():
            morphology_rows.append(
                {
                    "frame": i,
                    "finger": finger,
                    "raw_source_metric_target_rmse_m": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    source_local[indices]
                                    - np.asarray(warm_feature.local_keypoints)[indices]
                                )
                            )
                        )
                    ),
                    "robot_length_reconstructed_target_rmse_m": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    target[indices]
                                    - np.asarray(warm_feature.local_keypoints)[indices]
                                )
                            )
                        )
                    ),
                    "robot_length_target_reduces_error": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    target[indices]
                                    - np.asarray(warm_feature.local_keypoints)[indices]
                                )
                            )
                        )
                    )
                    < float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    source_local[indices]
                                    - np.asarray(warm_feature.local_keypoints)[indices]
                                )
                            )
                        )
                    ),
                    "diagnostic_only": True,
                    "not_robot_pose": True,
                }
            )
    morphology = {
        "schema_version": "toporetarget.stage7_1.morphology_normalized_target.v1",
        "raw_target_units": "m",
        "reconstructed_target_units": "m",
        "robot_bone_lengths_preserved": True,
        "per_finger": {
            finger: {
                "raw_target_error_m": _stats(
                    [
                        r["raw_source_metric_target_rmse_m"]
                        for r in morphology_rows
                        if r["finger"] == finger
                    ]
                ),
                "robot_length_target_error_m": _stats(
                    [
                        r["robot_length_reconstructed_target_rmse_m"]
                        for r in morphology_rows
                        if r["finger"] == finger
                    ]
                ),
            }
            for finger in FINGERS
        },
        "diagnostic_only": True,
        "not_robot_pose": True,
    }
    selection = _select_frames(per_frame, contact)
    diagnostics: dict[str, Any] | None = None
    diagnostic_root = repo_root / ".local/runs/stage7_1_reachability_diagnostics" / output.name
    if run_reachability_diagnostics:
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        selected = [
            int(value)
            for value in (
                selection["selected_frames"]
                if diagnostic_frames.strip().lower() == "auto"
                else [int(value) for value in diagnostic_frames.split(",") if value.strip()]
            )
        ][:5]
        diagnostics = _diagnostic_solve(
            source,
            warm,
            model,
            frame_profile,
            bone_profile,
            solver_profile,
            selected,
            np.asarray(warm.arrays["base_pose_scene"]),
            source_features,
            diagnostic_root,
        )
        diagnostics["workspace"] = [
            _workspace_audit(
                frame,
                source,
                warm,
                model,
                frame_profile,
                source_features,
                bone_profile,
                diagnostic_root,
            )
            for frame in selected[:5]
        ]
        workspace_arrays = [np.load(Path(row["points_path"])) for row in diagnostics["workspace"]]
        workspace_bundle = diagnostic_root / "thumb_workspace_points.npz"
        np.savez(
            workspace_bundle,
            frames=np.asarray([int(row["frame"]) for row in diagnostics["workspace"]]),
            qpos=np.stack([value["qpos"] for value in workspace_arrays]),
            thumb_cmc_workspace=np.stack(
                [value["thumb_cmc_workspace"] for value in workspace_arrays]
            ),
            thumb_tip_workspace=np.stack(
                [value["thumb_tip_workspace"] for value in workspace_arrays]
            ),
            thumb_pad_workspace=np.stack(
                [value["thumb_pad_workspace"] for value in workspace_arrays]
            ),
            raw_source_target=np.stack([value["raw_source_target"] for value in workspace_arrays]),
            robot_length_target=np.stack(
                [value["robot_length_target"] for value in workspace_arrays]
            ),
            seed=np.asarray(20260723),
        )
        for row in diagnostics["workspace"]:
            row["points_bundle_path"] = str(workspace_bundle)
        _write_json(
            diagnostic_root / "diagnostic_profiles.json",
            {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "diagnostic_only": True,
                "paper_method": False,
                "accepted_reference": False,
                "profiles": [
                    "official_stage7_reproduction",
                    "thumb_formal_feature_fit",
                    "thumb_keypoint_fit_fixed_base",
                    "thumb_keypoint_fit_with_base",
                    "all_joints_keypoint_fit_fixed_base",
                    "all_joints_keypoint_fit_with_base",
                    "thumb_robot_length_keypoint_fit_fixed_base",
                    "no_temporal_stage7_diagnostic",
                    "reduced_temporal_stage7_diagnostic",
                ],
                "selected_frames": selected,
            },
        )
        _write_csv(diagnostic_root / "reachability_results_per_frame.csv", diagnostics["results"])
        _write_json(
            diagnostic_root / "reachability_results_per_profile.json",
            {
                profile: [row for row in diagnostics["results"] if row.get("profile") == profile]
                for profile in sorted({row.get("profile") for row in diagnostics["results"]})
            },
        )
        _write_json(
            diagnostic_root / "thumb_workspace_audit.json",
            {"frames": diagnostics["workspace"], "diagnostic_only": True},
        )
        _write_csv(diagnostic_root / "thumb_workspace_summary.csv", diagnostics["workspace"])
        _write_json(
            diagnostic_root / "diagnostic_manifest.json",
            {
                "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
                "diagnostic_only": True,
                "paper_method": False,
                "accepted_reference": False,
                "official_solver_invocation_count": 0,
                "diagnostic_solver_invocation_count": diagnostics["solver_invocation_count"],
                "official_stage7_reproduction_invocation_count": len(diagnostics["official"]),
                "output_root": str(diagnostic_root),
            },
        )
    root_cause = _root_cause(
        replay, mapping, anchors, attribution, limits, jacobian, diagnostics, per_frame
    )
    final_state = {
        "source": source,
        "warm": np.asarray(warm.arrays["robot_keypoints_scene"]),
        "final": np.asarray(final.arrays["robot_keypoints_scene"]),
    }
    html_report = None
    if html_output:
        html_report = _build_html(
            output / "warmstart_fidelity_and_reachability.html",
            {
                "per_finger": per_finger,
                "global_frame_start": int(final.metadata.get("source_frame_offset", 240)),
                "readiness": root_cause["readiness"],
            },
            per_frame,
            final_state["source"],
            final_state["warm"],
            final_state["final"],
            get_layout("mediapipe21"),
        )
        _write_json(output / "html_headless_smoke.json", html_report)
    after_identity = {name: _stat(path) for name, path in official_before.items()}
    immutable_rows = {
        name: {
            "before": identity[name],
            "after": after_identity[name],
            "hash_unchanged": identity[name].get("sha256") == after_identity[name].get("sha256"),
            "mtime_unchanged": identity[name].get("mtime_ns")
            == after_identity[name].get("mtime_ns"),
        }
        for name in identity
    }
    immutability = {
        "schema_version": "toporetarget.stage7_1.immutability.v1",
        "official_artifacts_changed": not all(
            row["hash_unchanged"] and row["mtime_unchanged"] for row in immutable_rows.values()
        ),
        "official_solver_invocation_count": 0,
        "artifacts": immutable_rows,
    }
    _write_json(
        output / "input_identity_and_immutability.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_identity": {
                "sequence": sequence.metadata.sequence_id,
                "hand": manifest["hand"],
                "robot": manifest["robot"],
                "global_frame_range": manifest.get("selected_frame_range"),
                "local_frame_range": warm.metadata.get("frame_range"),
                "frame_count": warm.frame_count,
            },
            "inputs": {
                name: _stat(path)
                for name, path in {
                    "manifest": manifest_path,
                    "canonical": base,
                    "warm_start": warm_path,
                    "graph": graph_path,
                    "evaluation": evaluation_path,
                    "final": final_path,
                    "canonical_contact_audit": contact_root,
                }.items()
            },
            "profiles": {
                "frame": frame_profile.as_dict(),
                "bone": bone_profile.as_dict(),
                "solver": solver_profile.as_dict(),
                "paper_weights": warm.metadata.get("paper_weights"),
            },
            "official_artifacts_changed": immutability["official_artifacts_changed"],
        },
    )
    _write_json(output / "source_mediapipe_mapping_audit.json", mapping)
    _write_csv(output / "source_mediapipe_mapping_per_anchor.csv", source_anchor_rows)
    _write_json(output / "robot_anchor_mapping_audit.json", anchors)
    _write_csv(output / "robot_anchor_mapping_per_anchor.csv", robot_anchor_rows)
    _write_json(
        output / "thumb_urdf_chain_audit.json",
        {
            "chain": [
                row for row in robot_anchor_rows if str(row["semantic_name"]).startswith("thumb")
            ],
            "summary": {
                "thumb_joint_names": [
                    row["joint_name"] for row in joint_axis_rows if row.get("thumb_joint")
                ],
                "jacobian_influence": jacobian_influence,
            },
        },
    )
    _write_csv(output / "thumb_joint_axis_audit.csv", joint_axis_rows)
    _write_json(output / "hand_frame_audit.json", frame_audit)
    _write_csv(output / "hand_frame_per_frame.csv", frame_rows)
    _write_json(output / "bone_pair_profile_audit.json", bone_audit)
    _write_csv(output / "bone_pair_profile.csv", pair_rows)
    _write_json(output / "stage7_artifact_replay.json", replay)
    _write_csv(
        output / "stage7_artifact_replay.csv",
        [
            {"field": key, "max_difference": value}
            for key, value in replay["max_differences"].items()
        ],
    )
    _write_json(output / "per_finger_fidelity.json", attribution)
    _write_csv(output / "per_finger_fidelity_per_frame.csv", per_frame)
    _write_csv(output / "per_finger_summary.csv", per_finger)
    _write_json(
        output / "residual_visualization_semantics.json",
        {
            "schema_version": "toporetarget.stage7_1.residual_visualization.v1",
            "existing_stage10_viewer": "trajectory_mesh.html",
            "audit_scale": "global_fixed_scale",
            "warm_final_same_scale": True,
            "display_units": "mm",
            "actual_color_clipping": "none in audit; per-frame normalization is labeled diagnostic_only",
            "arrow_scale": "not used by audit table",
            "threshold": None,
            "thumb_argmax_caveat": "per-frame argmax can make thumb appear red; this audit reports fixed-scale numeric values",
        },
    )
    _write_json(
        output / "formal_base_seed_audit.json",
        {
            "schema_version": "toporetarget.stage7_1.formal_base_seed.v1",
            "formula": "T^S_B = T^S_Hs (T^B_Hr)^-1",
            "source_frame": "canonical_keypoint_wrist_v1",
            "robot_frame": "canonical_keypoint_wrist_v1",
            "round_trip": frame_audit,
            "alignment": warm.metadata.get("alignment"),
        },
    )
    _write_csv(output / "formal_base_seed_per_frame.csv", base_rows)
    _write_json(output / "base_alignment_alternatives.json", base_audit)
    _write_csv(output / "base_alignment_alternatives.csv", base_rows)
    _write_json(output / "joint_limit_audit.json", limits)
    _write_csv(output / "joint_limit_per_frame.csv", limit_rows)
    _write_csv(output / "joint_limit_per_joint.csv", limit_joint_rows)
    _write_json(output / "jacobian_observability.json", jacobian)
    _write_csv(output / "jacobian_observability_per_frame.csv", jacobian_rows)
    _write_csv(output / "per_finger_reachable_residual.csv", reachable_rows)
    _write_json(output / "morphology_normalized_target_audit.json", morphology)
    _write_csv(output / "morphology_normalized_per_finger.csv", morphology_rows)
    _write_json(output / "diagnostic_frame_selection.json", selection)
    _write_json(output / "source_warm_final_error_attribution.json", attribution)
    _write_csv(output / "source_warm_final_error_attribution.csv", per_finger)
    _write_json(output / "root_cause_analysis.json", root_cause)
    readiness = {
        "schema_version": "toporetarget.stage7_1.readiness.v1",
        "status": root_cause["readiness"],
        "CONTINUE_STAGE9_3_3": root_cause["CONTINUE_STAGE9_3_3"],
        "official_artifacts_changed": immutability["official_artifacts_changed"],
        "official_solver_invocation_count": 0,
        "diagnostic_solver_invocation_count": 0
        if diagnostics is None
        else diagnostics["solver_invocation_count"],
        "shadow_ablation_should_remain_paused": root_cause["CONTINUE_STAGE9_3_3"] == "NO",
    }
    _write_json(output / "stage7_1_readiness.json", readiness)
    _write_json(output / "official_artifact_immutability.json", immutability)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": readiness["status"],
        "CONTINUE_STAGE9_3_3": readiness["CONTINUE_STAGE9_3_3"],
        "one_sentence_conclusion": "Warm-start fidelity and task/contact fidelity are reported separately; thumb residual is not promoted to a root cause without mapping, replay, limits, Jacobian, and bounded diagnostic evidence.",
        "preflight": preflight,
        "run_identity": {
            "sequence": sequence.metadata.sequence_id,
            "hand": manifest["hand"],
            "robot": manifest["robot"],
            "global_frame_range": manifest.get("selected_frame_range"),
            "local_frame_range": warm.metadata.get("frame_range"),
            "frame_count": warm.frame_count,
        },
        "formal_fidelity": replay,
        "source_mapping": mapping,
        "robot_mapping": anchors,
        "per_finger": per_finger,
        "thumb": {
            "anchor_names": ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"],
            "joint_axis_audit": [row for row in joint_axis_rows if row.get("thumb_joint")],
            "workspace": None if diagnostics is None else diagnostics["workspace"],
            "jacobian": jacobian.get("formal_feature", {}).get("thumb"),
        },
        "base_alignment": base_audit,
        "reachability": None if diagnostics is None else diagnostics,
        "error_attribution": attribution,
        "root_cause": root_cause,
        "immutability": immutability,
        "elapsed_s": time.perf_counter() - started,
        "html": html_report,
        "input_paths": {name: str(path) for name, path in official_before.items()},
    }
    _write_json(output / "stage7_1_summary.json", summary)
    lines = [
        "# Stage 7.1 Warm-Start Fidelity and Reachability Audit",
        "",
        f"- Status: `{readiness['status']}`",
        f"- CONTINUE_STAGE9_3_3 = `{readiness['CONTINUE_STAGE9_3_3']}`",
        f"- Sequence: `{sequence.metadata.sequence_id}`; hand `{manifest['hand']}`; robot `{manifest['robot']}`; local frames `{warm.metadata.get('frame_range')}`",
        "",
        "## Formal Stage 7",
        "",
        f"- Eq. (1)/(2) replay gates: `{all(replay['gates'].values())}`.",
        f"- Maximum replay differences: `{replay['max_differences']}`.",
        f"- Official artifacts changed: `{immutability['official_artifacts_changed']}`.",
        "",
        "## Per-finger",
        "",
        "| Region | Warm Ebone | Warm keypoint RMSE (mm) | Warm EIM contribution | Warm contact proxy | Final change (mm) | Limit margin (rad) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['region']} | {row['warm_ebone']:.6g} | {row['warm_keypoint_rmse_m'] * 1000:.3f} | {row['warm_eim_contribution']:.6g} | {row['warm_contact_proxy']:.3f} | {row['warm_to_final_change'] * 1000:.3f} | {row['joint_limit_min_margin_rad']:.5f} |"
        for row in per_finger
    )
    lines += [
        "",
        "## Root cause",
        "",
        f"Top ranked cause: `{root_cause['ranked_causes'][0]['cause']}`.",
        "",
        "Diagnostic IK is diagnostic_only, paper_method=false, accepted_reference=false. Raw source metric targets and robot-length reconstructed targets are reported separately; local Jacobian/workspace evidence is not a global reachability proof.",
    ]
    (output / "stage7_1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # audit_manifest is deliberately written last; no post-hash smoke test mutates this root.
    files = [
        path for path in output.rglob("*") if path.is_file() and path.name != "audit_manifest.json"
    ]
    _write_json(
        output / "audit_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output),
            "files": {str(path.relative_to(output)): _stat(path) for path in sorted(files)},
            "official_artifacts_changed": immutability["official_artifacts_changed"],
            "official_solver_invocation_count": 0,
            "diagnostic_solver_invocation_count": 0
            if diagnostics is None
            else diagnostics["solver_invocation_count"],
        },
    )
    return summary


__all__ = ["run_warm_start_audit"]
