"""GRAB contact-proxy targets and fixed paper-external final candidates."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.readers.grab import load_grab_auxiliary
from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.signed_distance.reference import ReferenceSignedDistanceBackend
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory

from .schema import QUALITY_SCHEMA_VERSION, ClipSpec, write_json

CONTACT_GRID = (
    ("P1", 0.25, 0.0),
    ("P2", 1.0, 0.0),
    ("P3", 4.0, 0.0),
    ("PD1", 1.0, 0.1),
    ("PD2", 1.0, 0.5),
)
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_PARENTS = {"thumb": 3, "index": 7, "middle": 11, "ring": 15, "pinky": 19}
FINGER_REGION = {
    "thumb": "thumb_pad_distal",
    "index": "index_pad_distal",
    "middle": "middle_pad_distal",
    "ring": "ring_pad_distal",
    "pinky": "pinky_pad_distal",
}


def _object_track(sequence: Any) -> Any:
    if hasattr(sequence, "rigid_object"):
        try:
            return sequence.rigid_object("primary")
        except (KeyError, ValueError):
            pass
    if not sequence.rigid_objects:
        raise ValueError("canonical sequence has no object track")
    return sequence.rigid_objects[0]


def _finger_label(label_name: str) -> str | None:
    lowered = label_name.lower()
    for name in FINGER_TIPS:
        if name in lowered:
            return name
    return None


def _proxy_metrics(
    sequence: Any,
    clip: ClipSpec,
    final_path: str | Path,
    source_path: str | Path,
    *,
    threshold_m: float = 0.005,
) -> dict[str, Any]:
    try:
        artifact: Any = load_final_trajectory(final_path)
    except (OSError, ValueError, RuntimeError):
        # Warm-start artifacts expose the same robot_keypoints_scene contract.
        artifact = load_warm_start(final_path)
    hand = next(item for item in sequence.hands if item.side == clip.hand)
    source_keypoints = np.asarray(
        hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    object_track = _object_track(sequence)
    object_vertices = np.asarray(object_track.mesh.vertices_local, dtype=np.float64)
    poses = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
    raw = load_grab_auxiliary(
        source_path,
        frame_range=FrameRange(clip.start_frame, clip.end_frame),
        include_table=False,
        contact_mode="semantic",
    )
    labels = np.asarray(raw["contact"]["object"], dtype=np.int64)
    object_points = np.concatenate(
        [
            object_vertices @ poses[index, :3, :3].T + poses[index, :3, 3]
            for index in range(len(poses))
        ],
        axis=0,
    ).reshape(len(poses), len(object_vertices), 3)
    names = {int(item["id"]): str(item["name"]) for item in _mapping_table()}
    robot_keypoints = np.asarray(artifact.arrays["robot_keypoints_scene"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    thresholds = (0.002, 0.003, 0.005, 0.008, 0.010)
    values: dict[float, dict[str, list[float]]] = {
        value: {"precision": [], "recall": [], "f1": []} for value in thresholds
    }
    per_finger_values: dict[float, dict[str, dict[str, list[float]]]] = {
        value: {finger: {"precision": [], "recall": [], "f1": []} for finger in FINGER_TIPS}
        for value in thresholds
    }
    for frame in range(min(len(labels), len(robot_keypoints))):
        source_active: dict[str, bool] = {name: False for name in FINGER_TIPS}
        for label in np.unique(labels[frame]):
            finger = _finger_label(names.get(int(label), ""))
            if finger is not None and int(label) != 0:
                source_active[finger] = True
        source_distances: dict[str, float] = {}
        robot_distances: dict[str, float] = {}
        for finger, tip_index in FINGER_TIPS.items():
            source_distance = _nearest_distance(
                source_keypoints[frame, tip_index], object_points[frame]
            )
            robot_distance = _nearest_distance(
                robot_keypoints[frame, tip_index], object_points[frame]
            )
            source_distances[finger] = source_distance
            robot_distances[finger] = robot_distance
        truth = np.asarray([source_active[name] for name in FINGER_TIPS], dtype=bool)
        threshold_rows: dict[str, Any] = {}
        for threshold in thresholds:
            predicted = np.asarray(
                [robot_distances[name] <= threshold for name in FINGER_TIPS], dtype=bool
            )
            tp = int(np.count_nonzero(truth & predicted))
            precision = tp / max(int(np.count_nonzero(predicted)), 1)
            recall = tp / max(int(np.count_nonzero(truth)), 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            values[threshold]["precision"].append(precision)
            values[threshold]["recall"].append(recall)
            values[threshold]["f1"].append(f1)
            threshold_rows[f"{int(threshold * 1000)}mm"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            for finger_index, finger in enumerate(FINGER_TIPS):
                finger_truth = bool(truth[finger_index])
                finger_predicted = bool(predicted[finger_index])
                finger_tp = int(finger_truth and finger_predicted)
                finger_precision = float(finger_tp / max(int(finger_predicted), 1))
                finger_recall = float(finger_tp / max(int(finger_truth), 1))
                finger_f1 = float(
                    2.0
                    * finger_precision
                    * finger_recall
                    / max(finger_precision + finger_recall, 1e-12)
                )
                per_finger_values[threshold][finger]["precision"].append(finger_precision)
                per_finger_values[threshold][finger]["recall"].append(finger_recall)
                per_finger_values[threshold][finger]["f1"].append(finger_f1)
        alignment = float(
            np.mean(
                [
                    max(0.0, 1.0 - robot_distances[name] / max(threshold_m, 1e-12))
                    for name in FINGER_TIPS
                    if source_active[name]
                ]
            )
            if any(source_active.values())
            else 0.0
        )
        rows.append(
            {
                "local_frame": frame,
                "global_frame": clip.start_frame + frame,
                "source_active": source_active,
                "source_contact_proxy": {
                    name: source_distances[name] <= threshold_m for name in FINGER_TIPS
                },
                "robot_contact_proxy": {
                    name: robot_distances[name] <= threshold_m for name in FINGER_TIPS
                },
                "source_distances_m": source_distances,
                "robot_distances_m": robot_distances,
                "alignment": alignment,
                "retention": threshold_rows,
                "metric_semantics": "DATASET_PROXY",
            }
        )
    aggregate = {
        f"{int(threshold * 1000)}mm": {
            key: float(np.mean(values[threshold][key])) for key in ("precision", "recall", "f1")
        }
        for threshold in thresholds
    }
    per_finger = {
        f"{int(threshold * 1000)}mm": {
            finger: {
                key: float(np.mean(per_finger_values[threshold][finger][key]))
                for key in ("precision", "recall", "f1")
            }
            for finger in FINGER_TIPS
        }
        for threshold in thresholds
    }
    five = aggregate["5mm"]
    return {
        "threshold_m": threshold_m,
        "retention_thresholds_mm": [int(value * 1000) for value in thresholds],
        "retention": aggregate,
        "per_finger_retention": per_finger,
        "metric_semantics": "DATASET_PROXY",
        "ground_truth_contact": False,
        "frames": rows,
        "contact_precision_proxy": five["precision"],
        "contact_recall_proxy": five["recall"],
        "contact_alignment_proxy": float(np.mean([item["alignment"] for item in rows]))
        if rows
        else 0.0,
        "contact_f1": five["f1"],
    }


def _nearest_distance(point: np.ndarray, points: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    return float(cKDTree(points).query(point, k=1)[0])


def _mapping_table() -> list[dict[str, Any]]:
    from toporetarget.data.contacts.grab import load_grab_contact_mapping

    return list(load_grab_contact_mapping().table().values())


def build_source_contact_targets(
    canonical_path: str | Path,
    source_path: str | Path,
    clip: ClipSpec,
    surface_profile_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build deterministic source-MANO/object-local contact targets.

    GRAB's ``contact.object`` is a per-object-vertex semantic label array.  The
    source hand centroid is obtained from the canonical MANO vertices nearest
    to the corresponding MediaPipe fingertip; no frame is hand-selected or
    edited.  The object closest point is queried on the original object mesh,
    never on a visual/contact proxy.
    """

    sequence = load_hoi_sequence(canonical_path)
    hand = next(item for item in sequence.hands if item.side == clip.hand)
    if hand.vertices_scene is None:
        raise ValueError("source canonical artifact does not contain MANO vertices")
    object_track = _object_track(sequence)
    object_vertices = np.asarray(object_track.mesh.vertices_local, dtype=np.float64)
    poses = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
    source_keypoints = np.asarray(
        hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    raw = load_grab_auxiliary(
        source_path,
        frame_range=FrameRange(clip.start_frame, clip.end_frame),
        include_table=False,
        contact_mode="semantic",
    )
    labels = np.asarray(raw["contact"]["object"], dtype=np.int64)
    names = {int(item["id"]): str(item["name"]) for item in _mapping_table()}
    region_ids = tuple(FINGER_REGION.values())
    region_count = len(region_ids)
    target_relative = np.zeros((clip.length, region_count, 3), dtype=np.float64)
    target_direction = np.zeros_like(target_relative)
    active = np.zeros((clip.length, region_count), dtype=bool)
    weights = np.zeros((clip.length, region_count), dtype=np.float64)
    source_centroids = np.zeros_like(target_relative)
    object_points = np.zeros_like(target_relative)
    source_distances = np.zeros((clip.length, region_count), dtype=np.float64)
    source_sample_count = np.zeros((clip.length, region_count), dtype=np.int64)

    distance_backend = ReferenceSignedDistanceBackend(
        object_vertices,
        np.asarray(object_track.mesh.faces, dtype=np.int64),
        sign_mode="unsigned_only",
    )
    for frame in range(clip.length):
        pose = poses[frame]
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        for region_index, (finger, _region_id) in enumerate(FINGER_REGION.items()):
            finger_active = False
            for label in np.unique(labels[frame]):
                label_name = names.get(int(label), "")
                if int(label) != 0 and _finger_label(label_name) == finger:
                    finger_active = True
                    break
            if not finger_active:
                continue
            tip = source_keypoints[frame, FINGER_TIPS[finger]]
            source_vertices = np.asarray(hand.vertices_scene[frame], dtype=np.float64)
            nearest_ids = np.argsort(np.linalg.norm(source_vertices - tip[None, :], axis=1))[:32]
            centroid_scene = np.mean(source_vertices[nearest_ids], axis=0)
            centroid_local = (centroid_scene - translation) @ rotation
            closest = distance_backend.query_local(centroid_local.reshape(1, 3))
            object_point_local = np.asarray(closest.closest_points[0], dtype=np.float64)
            direction_scene = (
                source_keypoints[frame, FINGER_TIPS[finger]]
                - source_keypoints[frame, FINGER_PARENTS[finger]]
            )
            direction_local = direction_scene @ rotation
            direction_local /= max(float(np.linalg.norm(direction_local)), 1e-12)
            target_relative[frame, region_index] = centroid_local - object_point_local
            target_direction[frame, region_index] = direction_local
            active[frame, region_index] = True
            source_centroids[frame, region_index] = centroid_local
            object_points[frame, region_index] = object_point_local
            source_distances[frame, region_index] = float(closest.unsigned_distance[0])
            source_sample_count[frame, region_index] = int(len(nearest_ids))
            weights[frame, region_index] = 1.0 / max(
                1.0, 1.0 + source_distances[frame, region_index] / 0.01
            )

    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    target_path = destination / "source_contact_targets.npz"
    np.savez_compressed(
        target_path,
        region_id=np.asarray(region_ids),
        target_relative=target_relative,
        target_direction=target_direction,
        active=active,
        weights=weights,
        source_centroid_object_local=source_centroids,
        object_point_object_local=object_points,
        source_distance_m=source_distances,
        source_sample_count=source_sample_count,
    )
    profile = json.loads(Path(surface_profile_path).read_text(encoding="utf-8"))
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "pass",
        "profile_id": "grab_source_contact_targets_v1",
        "metric_semantics": "DATASET_PROXY",
        "canonical_path": str(Path(canonical_path).resolve()),
        "source_path": str(Path(source_path).resolve()),
        "surface_profile_hash": profile.get("profile_hash"),
        "region_ids": list(region_ids),
        "frame_count": clip.length,
        "active_region_count": int(np.count_nonzero(active)),
        "active_region_frame_count": int(np.count_nonzero(np.any(active, axis=1))),
        "source_surface_representation": "canonical MANO vertices nearest to semantic fingertip",
        "object_closest_point_representation": "original object mesh reference unsigned distance",
        "virtual_contact_warning": bool(np.any(source_distances[active] > 0.01))
        if np.any(active)
        else False,
        "artifact": str(target_path.resolve()),
        "no_manual_frame_selection": True,
    }
    write_json(payload, destination / "source_contact_targets.json")
    return {**payload, "artifact": str(target_path.resolve())}


def load_contact_objective_extension(
    target_path: str | Path,
    surface_profile_path: str | Path,
    *,
    profile_id: str,
    lambda_contact_pos: float,
    lambda_contact_dir: float,
) -> dict[str, Any]:
    """Load a fixed D candidate into the differentiable final solver contract."""

    if profile_id not in {item[0] for item in CONTACT_GRID}:
        raise ValueError(f"unknown fixed contact profile: {profile_id}")
    target = np.load(target_path, allow_pickle=False)
    profile = json.loads(Path(surface_profile_path).read_text(encoding="utf-8"))
    regions_by_id = {item["semantic_id"]: item for item in profile["regions"]}
    region_ids = [str(item) for item in target["region_id"].tolist()]
    regions: list[dict[str, Any]] = []
    for region_id in region_ids:
        item = regions_by_id[region_id]
        samples = item["samples"]
        regions.append(
            {
                "region_id": region_id,
                "link": item["link"],
                "local_transform": np.asarray(item["link_local_frame"], dtype=np.float64),
                "points_link": np.asarray([sample["point_link"] for sample in samples]),
                "semantic_direction_link": np.asarray(
                    item["semantic_direction_link"], dtype=np.float64
                ),
            }
        )
    return {
        "profile_id": profile_id,
        "paper_method": False,
        "paper_external_extension": True,
        "contact_regions": regions,
        "contact_region_ids": region_ids,
        "contact_target_relative": np.asarray(target["target_relative"], dtype=np.float64),
        "contact_target_direction": np.asarray(target["target_direction"], dtype=np.float64),
        "contact_active": np.asarray(target["active"], dtype=bool),
        "contact_weights": np.asarray(target["weights"], dtype=np.float64),
        "lambda_contact_pos": float(lambda_contact_pos),
        "lambda_contact_dir": float(lambda_contact_dir),
        "contact_position_scale_m": 0.01,
        "huber_delta": 1.0,
        "source_contact_target_artifact": str(Path(target_path).resolve()),
        "surface_profile_hash": profile.get("profile_hash"),
    }


def _huber(value: np.ndarray, delta: float = 1.0) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute <= delta, 0.5 * value * value, delta * (absolute - 0.5 * delta))


def build_contact_candidates(
    canonical_path: str | Path,
    source_path: str | Path,
    final_path: str | Path,
    clip: ClipSpec,
    surface_profile_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Score the predeclared D grid without changing the frozen Eq. (8) solver."""

    sequence = load_hoi_sequence(canonical_path)
    baseline = _proxy_metrics(sequence, clip, final_path, source_path)
    source_frames = baseline["frames"]
    selected_frames = sorted(
        {
            0,
            29,
            59,
            max(
                range(len(source_frames)),
                key=lambda index: sum(source_frames[index]["source_active"].values()),
            ),
            max(
                range(len(source_frames)),
                key=lambda index: sum(
                    source_frames[index]["source_active"][name]
                    != source_frames[max(index - 1, 0)]["source_active"][name]
                    for name in FINGER_TIPS
                ),
            ),
        }
    )
    candidates: list[dict[str, Any]] = []
    for profile_id, lambda_pos, lambda_dir in CONTACT_GRID:
        # Position and direction terms are explicitly evaluated and normalized;
        # the existing Eq. (8) artifact is not mutated by this diagnostic lane.
        contact_pos = float(
            np.mean([sum(frame["robot_distances_m"].values()) / 0.01 for frame in source_frames])
        )
        contact_dir = float(1.0 - baseline["contact_alignment_proxy"])
        loss = (
            lambda_pos * float(np.mean(_huber(np.asarray([contact_pos]))))
            + lambda_dir * contact_dir
        )
        candidates.append(
            {
                "profile_id": profile_id,
                "lambda_contact_pos": lambda_pos,
                "lambda_contact_dir": lambda_dir,
                "huber_delta": 1.0,
                "position_normalization_m": 0.01,
                "active_region_mean": True,
                "contact_position_loss": contact_pos,
                "contact_direction_loss": contact_dir,
                "diagnostic_loss": loss,
                "paper_method": False,
                "paper_external_extension": True,
                "solver_invocation_count": 0,
                "diagnostic_only": True,
                "accepted": False,
                "rejection_reason": "contact-aware objective is not wired into frozen Stage 9 solver in this lane",
            }
        )
    selected = min(candidates, key=lambda item: (item["diagnostic_loss"], item["profile_id"]))
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "surface_profile": str(Path(surface_profile_path).resolve()),
            "selected_source_frames": selected_frames,
            "candidates": candidates,
            "pre_screen": {"status": "complete", "all_fixed_candidates_scored": True},
            "full_trajectory": {
                "status": "diagnostic_only",
                "contact_profile_accepted": False,
                "reason": candidates[0]["rejection_reason"],
            },
            "c_star": {**selected, "not_recommended": True},
        },
        # This is the proxy-only diagnostic lane.  The authoritative
        # solver-backed selection is written by the A--E orchestrator after
        # prescreen/full-trajectory outcomes are known.
        output / "contact_proxy_diagnostic_selection.json",
    )
    write_json(baseline, output / "baseline_contact_proxy.json")
    return {
        "status": "CONTACT_PROFILE_REJECTED",
        "c_star": {**selected, "not_recommended": True},
        "candidates": candidates,
        "proxy": baseline,
    }


__all__ = [
    "CONTACT_GRID",
    "build_contact_candidates",
    "build_source_contact_targets",
    "load_contact_objective_extension",
]
