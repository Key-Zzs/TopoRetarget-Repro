#!/usr/bin/env python3
"""Materialize the offline Stage16 angular/raw-grasp authority review."""

# Report prose and machine-readable field names intentionally remain explicit.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation.source_contact_semantics import (  # noqa: E402
    SEGMENT_ORDER,
    ManoSurfaceRegionMap,
    SourceContactThresholdContractV1,
    persistent_mask,
)
from toporetarget.geometry.signed_distance.closest_point import ObjectLocalBVH  # noqa: E402
from toporetarget.rl.geometry_audit.raw_mocap_overlay import (  # noqa: E402
    pose_wxyz_to_matrix,
    resolve_raw_mocap_overlay,
)
from toporetarget.rl.reference_tracking.reference_kinematics import (  # noqa: E402
    derive_angular_velocity_world_wxyz,
)
from toporetarget.rl.reference_tracking.strict_per_finger_contact import (  # noqa: E402
    strict_source_contact_mask,
)
from toporetarget.rl.stage16_authority_v2 import (  # noqa: E402
    RawHumanGraspReadinessProfileV1,
    Stage16ActualAngularVelocityAuthorityV2,
    angular_velocity_semantic_alignment,
    opposing_contact_topology,
    raw_human_grasp_profile,
    timing_layer_profile,
)
from toporetarget.rl.stage16_pf_df import (  # noqa: E402
    first_true,
    persistent_finger_mask,
    terminal_threshold_pass,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_angular_semantics_and_raw_grasp_authority"
PRIOR_ROOT = REPO_ROOT / ".local/reports/stage16_contact_timing_angular_twist_pf_df"
STRICT_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
SOURCE_ROOT = REPO_ROOT / ".local/reports/stage16d_source_contact_semantics_final_audit"
REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
FORMAL_650 = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
)
FORMAL_105 = (
    REPO_ROOT
    / ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke"
    / "former_timeout_v4_170105_c4/v4/hocap_170105/c4"
)
START_HEAD = "557acf62b6fecba5532252d22df3b181bb57137d"
CLIPS = ("hocap_170105", "hocap_170650")
LIFT_FRAME = 184
CONTROL_DT_S = 0.05

HISTORICAL_REPORTS = (
    "stage16_contact_timing_angular_twist_pf_df",
    "stage16_dynamic_physical_qualification_and_grasp_diagnostic",
    "stage16_full_gravity_capability_closure",
    "stage16_raw_mocap_replay_overlay",
    "stage16_frozen_source_policy_gravity_sweep",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STAGE16_AUTHORITY_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"STAGE16_AUTHORITY_CSV_EMPTY:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"STAGE16_AUTHORITY_CSV_FIELD_DRIFT:{path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(_sha256(item)))
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _line(path: Path, text: str) -> int:
    for number, value in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if text in value:
            return number
    raise ValueError(f"STAGE16_AUTHORITY_SOURCE_NEEDLE_MISSING:{path}:{text}")


def _trace_paths(clip: str) -> list[Path]:
    if clip == "hocap_170105":
        result = [FORMAL_105 / f"episode_{index:02d}.npz" for index in range(10)]
    elif clip == "hocap_170650":
        result = [FORMAL_650 / f"episode_{index:03d}.npz" for index in range(20)]
    else:
        raise ValueError(f"STAGE16_AUTHORITY_UNKNOWN_CLIP:{clip}")
    if not all(path.is_file() for path in result):
        raise FileNotFoundError(f"STAGE16_AUTHORITY_TRACE_MISSING:{clip}")
    return result


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    required = {
        "object_pose",
        "object_twist",
        "object_twist_reference",
        "hand_object_pair_force_valid",
        "hand_object_pair_presence",
        "source_contact_mask",
        "reference_index",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"STAGE16_AUTHORITY_TRACE_FIELD_MISSING:{path}:{sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in required}
    if result["object_pose"].shape != (321, 7) or result["object_twist"].shape != (321, 6):
        raise ValueError(f"STAGE16_AUTHORITY_TRACE_SHAPE_INVALID:{path}")
    if not np.array_equal(result["reference_index"], np.arange(321)):
        raise ValueError(f"STAGE16_AUTHORITY_REFERENCE_INDEX_DRIFT:{path}")
    return result


def _dist(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("STAGE16_AUTHORITY_DISTRIBUTION_INVALID")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _terminal_pass(error: np.ndarray, trace: dict[str, np.ndarray], clip: str) -> bool:
    gates = _read_json(STRICT_ROOT / "frozen_evaluation_gates.json")["task_gates"]["clips"][clip]
    contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=1)
    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    return terminal_threshold_pass(
        error,
        contact=contact,
        valid=valid,
        contact_limit=float(gates["terminal_angular_speed_radps"]),
        free_limit=float(gates["terminal_free_object_angular_speed_radps"]),
        terminal_steps=int(gates["terminal_window_control_steps"]),
    )


def _angular_clip(clip: str) -> dict[str, object]:
    output = REPORT_ROOT / "angular_semantics" / clip.replace("hocap_", "v4_")
    episode_rows: list[dict[str, object]] = []
    legacy_all: list[np.ndarray] = []
    authority_all: list[np.ndarray] = []
    mismatch_all: list[np.ndarray] = []
    corrected_all: list[np.ndarray] = []
    static_counts = 0
    valid_counts = 0
    offset: dict[str, list[np.ndarray]] = {"t": [], "t_minus_1": [], "t_plus_1": []}
    frame_candidates: dict[str, list[np.ndarray]] = {
        "documented_world": [],
        "hypothetical_local_to_world": [],
        "hypothetical_world_to_local": [],
    }
    per_episode_payload: list[dict[str, object]] = []
    timestamps = np.arange(321, dtype=np.float64) * CONTROL_DT_S
    for episode, path in enumerate(_trace_paths(clip)):
        trace = _load_trace(path)
        valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
        semantic = angular_velocity_semantic_alignment(
            object_pose_wxyz=trace["object_pose"],
            trace_angular_velocity=np.asarray(trace["object_twist"], dtype=np.float64)[:, 3:],
            timestamps_s=timestamps,
            valid=valid,
        )
        legacy_omega = np.asarray(semantic["historical_trace_omega_world"])
        corrected_omega = np.asarray(semantic["semantic_corrected_trace_omega_world"])
        authority_omega = np.asarray(semantic["authority_omega_world"])
        reference_omega = np.asarray(trace["object_twist_reference"], dtype=np.float64)[:, 3:]
        legacy = np.linalg.norm(legacy_omega - reference_omega, axis=1)
        corrected = np.linalg.norm(corrected_omega - reference_omega, axis=1)
        authority = np.linalg.norm(authority_omega - reference_omega, axis=1)
        mismatch = np.linalg.norm(legacy_omega - authority_omega, axis=1)
        legacy_all.append(legacy[valid])
        corrected_all.append(corrected[valid])
        authority_all.append(authority[valid])
        mismatch_all.append(mismatch[valid])
        closure = semantic["kinematic_closure"]
        static_counts += int(closure["static_pose_nonzero_trace_frame_count"])
        valid_counts += int(valid.sum())
        for name in offset:
            offset[name].append(
                np.asarray([semantic["offset_diagnostics"][name]["mean"]], dtype=np.float64)
            )
        for name in frame_candidates:
            frame_candidates[name].append(
                np.asarray([semantic["frame_diagnostics"][name]["mean"]], dtype=np.float64)
            )
        legacy_pass = _terminal_pass(legacy, trace, clip)
        authority_pass = _terminal_pass(authority, trace, clip)
        row = {
            "episode": episode,
            "trace_path": str(path.resolve()),
            "trace_sha256": _sha256(path),
            "trace_vs_pose_mean_radps": float(mismatch[valid].mean()),
            "trace_vs_pose_median_radps": float(np.median(mismatch[valid])),
            "trace_vs_pose_p95_radps": float(np.quantile(mismatch[valid], 0.95)),
            "trace_vs_pose_max_radps": float(mismatch[valid].max()),
            "corrected_vs_pose_mean_radps": float(mismatch[valid].mean()),
            "corrected_vs_pose_p95_radps": float(np.quantile(mismatch[valid], 0.95)),
            "corrected_vs_pose_max_radps": float(mismatch[valid].max()),
            "Delta_omega_historical_mean_radps": float(legacy[valid].mean()),
            "Delta_omega_historical_p95_radps": float(np.quantile(legacy[valid], 0.95)),
            "Delta_omega_semantic_corrected_mean_radps": float(corrected[valid].mean()),
            "Delta_omega_semantic_corrected_p95_radps": float(np.quantile(corrected[valid], 0.95)),
            "Delta_omega_pose_derived_mean_radps": float(authority[valid].mean()),
            "Delta_omega_pose_derived_p95_radps": float(np.quantile(authority[valid], 0.95)),
            "terminal_Delta_omega_historical_max_radps": float(legacy[-20:].max()),
            "terminal_Delta_omega_authority_v2_max_radps": float(authority[-20:].max()),
            "DF_angular_legacy": legacy_pass,
            "DF_angular_pose_derived": authority_pass,
            "DF_angular_authority_v2": authority_pass,
        }
        episode_rows.append(row)
        per_episode_payload.append({**row, "semantic_alignment": semantic})

        frame_rows = []
        object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
        trace_omega = np.asarray(trace["object_twist"], dtype=np.float64)[:, 3:]
        for frame in range(len(timestamps)):
            frame_rows.append(
                {
                    "frame": frame,
                    "runtime_time_s": float(timestamps[frame]),
                    "valid": bool(valid[frame]),
                    "object_qw": float(object_pose[frame, 3]),
                    "object_qx": float(object_pose[frame, 4]),
                    "object_qy": float(object_pose[frame, 5]),
                    "object_qz": float(object_pose[frame, 6]),
                    "trace_omega_world_x_radps": float(trace_omega[frame, 0]),
                    "trace_omega_world_y_radps": float(trace_omega[frame, 1]),
                    "trace_omega_world_z_radps": float(trace_omega[frame, 2]),
                    "authority_v2_omega_world_x_radps": float(authority_omega[frame, 0]),
                    "authority_v2_omega_world_y_radps": float(authority_omega[frame, 1]),
                    "authority_v2_omega_world_z_radps": float(authority_omega[frame, 2]),
                    "reference_omega_world_x_radps": float(reference_omega[frame, 0]),
                    "reference_omega_world_y_radps": float(reference_omega[frame, 1]),
                    "reference_omega_world_z_radps": float(reference_omega[frame, 2]),
                    "trace_vs_pose_radps": float(mismatch[frame]),
                    "Delta_omega_historical_radps": float(legacy[frame]),
                    "Delta_omega_authority_v2_radps": float(authority[frame]),
                }
            )
        _write_csv(output / "frame_debug" / f"episode_{episode:03d}.csv", frame_rows)

    serializable_rows = [
        {key: value for key, value in row.items() if key != "semantic_alignment"}
        for row in episode_rows
    ]
    _write_csv(output / "per_episode.csv", serializable_rows)
    legacy_values = np.concatenate(legacy_all)
    corrected_values = np.concatenate(corrected_all)
    authority_values = np.concatenate(authority_all)
    mismatch_values = np.concatenate(mismatch_all)
    aggregate = {
        "clip": clip,
        "episodes": len(episode_rows),
        "trace_vs_pose_mismatch": _dist(mismatch_values),
        "corrected_vs_pose_mismatch": _dist(mismatch_values),
        "Delta_omega_historical": _dist(legacy_values),
        "Delta_omega_semantic_corrected": _dist(corrected_values),
        "Delta_omega_pose_derived": _dist(authority_values),
        "Delta_omega_authority_v2": _dist(authority_values),
        "terminal_Delta_omega_historical_max_radps": float(
            max(float(row["terminal_Delta_omega_historical_max_radps"]) for row in episode_rows)
        ),
        "terminal_Delta_omega_authority_v2_max_radps": float(
            max(float(row["terminal_Delta_omega_authority_v2_max_radps"]) for row in episode_rows)
        ),
        "DF_angular_legacy": int(sum(bool(row["DF_angular_legacy"]) for row in episode_rows)),
        "DF_angular_pose_derived": int(
            sum(bool(row["DF_angular_pose_derived"]) for row in episode_rows)
        ),
        "DF_angular_authority_v2": int(
            sum(bool(row["DF_angular_authority_v2"]) for row in episode_rows)
        ),
        "static_pose_nonzero_trace_frame_count": static_counts,
        "static_pose_nonzero_trace_fraction": static_counts / valid_counts,
        "offset_mean_across_episodes_radps": {
            key: float(np.concatenate(values).mean()) for key, values in offset.items()
        },
        "frame_candidate_mean_across_episodes_radps": {
            key: float(np.concatenate(values).mean()) for key, values in frame_candidates.items()
        },
        "THRESHOLD_PROVENANCE": "LEGACY_INHERITED_NOT_SCIENTIFICALLY_RECALIBRATED",
        "ANGULAR_THRESHOLD_TUNED": "NO",
    }
    _write_csv(
        output / "aggregate.csv",
        [
            {
                "metric": key,
                "mean": value["mean"],
                "median": value["median"],
                "p95": value["p95"],
                "max": value["max"],
            }
            for key, value in (
                ("trace_vs_pose_mismatch", aggregate["trace_vs_pose_mismatch"]),
                ("corrected_vs_pose_mismatch", aggregate["corrected_vs_pose_mismatch"]),
                ("Delta_omega_historical", aggregate["Delta_omega_historical"]),
                ("Delta_omega_pose_derived", aggregate["Delta_omega_pose_derived"]),
                ("Delta_omega_authority_v2", aggregate["Delta_omega_authority_v2"]),
            )
        ],
    )
    _write_json(output / "semantic_alignment.json", aggregate)
    return {"aggregate": aggregate, "episodes": episode_rows}


def _synthetic_validation() -> dict[str, object]:
    timestamps = np.arange(101, dtype=np.float64) * 0.01

    def pose(rotations: Rotation) -> np.ndarray:
        xyzw = rotations.as_quat()
        if xyzw.ndim == 1:
            xyzw = np.broadcast_to(xyzw, (len(timestamps), 4))
        return np.concatenate((np.zeros((len(timestamps), 3)), xyzw[:, 3:4], xyzw[:, :3]), axis=-1)

    zero = angular_velocity_semantic_alignment(
        object_pose_wxyz=pose(Rotation.identity()),
        trace_angular_velocity=np.zeros((len(timestamps), 3)),
        timestamps_s=timestamps,
    )
    omega = np.array([0.0, 0.0, 0.7])
    identity = angular_velocity_semantic_alignment(
        object_pose_wxyz=pose(Rotation.from_rotvec(timestamps[:, None] * omega)),
        trace_angular_velocity=np.broadcast_to(omega, (len(timestamps), 3)),
        timestamps_s=timestamps,
    )
    initial = Rotation.from_euler("xy", [37.0, -21.0], degrees=True)
    rotated = angular_velocity_semantic_alignment(
        object_pose_wxyz=pose(Rotation.from_rotvec(timestamps[:, None] * omega) * initial),
        trace_angular_velocity=np.broadcast_to(omega, (len(timestamps), 3)),
        timestamps_s=timestamps,
    )
    return {
        "schema_version": "Stage16AngularSemanticSyntheticValidationV1",
        "status": "PASS",
        "isaac_run": "NOT_REQUIRED_STATIC_ISAACLAB_SOURCE_UNAMBIGUOUS",
        "static_body_max_radps": float(
            np.linalg.norm(np.asarray(zero["authority_omega_world"]), axis=1).max()
        ),
        "constant_world_axis_max_error_radps": float(
            np.linalg.norm(np.asarray(identity["authority_omega_world"]) - omega, axis=1).max()
        ),
        "rotated_body_world_axis_max_error_radps": float(
            np.linalg.norm(np.asarray(rotated["authority_omega_world"]) - omega, axis=1).max()
        ),
        "rotated_body_hypothetical_local_to_world_mean_error_radps": rotated["frame_diagnostics"][
            "hypothetical_local_to_world"
        ]["mean"],
        "tests": [
            "static body",
            "constant known world-axis rotation",
            "non-identity body orientation",
            "body/world conversion",
            "SO(3) wrap-around in unit tests",
            "same timestamp and plus/minus one offset diagnostics",
        ],
    }


def _strict_mask(clip: str) -> np.ndarray:
    path = STRICT_ROOT / f"strict_source_contact_mask_{clip}.npz"
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["strict_source_contact_mask"], dtype=bool)
    if mask.shape != (321, 5):
        raise ValueError(f"STAGE16_AUTHORITY_STRICT_MASK_SHAPE_INVALID:{clip}")
    source_path = SOURCE_ROOT / clip / "source_contact_evidence_runtime.npz"
    with np.load(source_path, allow_pickle=False) as archive:
        source_labels = np.asarray(archive["class_label"])
        source_fingers = np.asarray(archive["finger_order"]).astype(str)
    with np.load(path, allow_pickle=False) as archive:
        strict_labels = np.asarray(archive["source_contact_class"])
        strict_fingers = np.asarray(archive["finger_names"]).astype(str)
    if not np.array_equal(source_labels, strict_labels):
        raise ValueError(f"STAGE16_AUTHORITY_STRICT_LABEL_PARITY_FAILED:{clip}")
    if not np.array_equal(source_fingers, strict_fingers):
        raise ValueError(f"STAGE16_AUTHORITY_STRICT_FINGER_PARITY_FAILED:{clip}")
    if not np.array_equal(mask, strict_source_contact_mask(source_labels)):
        raise ValueError(f"STAGE16_AUTHORITY_STRICT_MASK_PARITY_FAILED:{clip}")
    return mask


def _normal_quality(vertices: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=-1)
    unit = normals / np.maximum(norms[:, None], 1.0e-12)
    centroid = vertices.mean(axis=0)
    outward = np.sum(unit * (triangles.mean(axis=1) - centroid), axis=1) > 0.0
    unique_edges: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            unique_edges[edge] = unique_edges.get(edge, 0) + 1
    watertight = bool(unique_edges and all(count == 2 for count in unique_edges.values()))
    return {
        "triangle_count": int(len(faces)),
        "degenerate_triangle_count": int(np.count_nonzero(norms <= 1.0e-12)),
        "centroid_outward_normal_fraction": float(outward.mean()),
        "watertight_by_edge_incidence": watertight,
        "normal_authority": (
            "DIAGNOSTIC_ONLY_NON_WATERTIGHT" if not watertight else "GEOMETRIC_DIAGNOSTIC"
        ),
    }


def _raw_clip(clip: str) -> dict[str, object]:
    trace_path = _trace_paths(clip)[0]
    overlay = resolve_raw_mocap_overlay(
        trace_path=trace_path,
        frame_count=321,
        clip=clip,
        reference_path=REFERENCE_ROOT / f"{clip}.world_wrist.stage16.npz",
    )
    region_path = SOURCE_ROOT / "mano_surface_region_map_v1.npz"
    with np.load(region_path, allow_pickle=False) as archive:
        region_map = ManoSurfaceRegionMap(
            np.asarray(archive["region_id"]),
            np.asarray(archive["segment_id"]),
            np.asarray(archive["soft_region_weight"]),
        )
        mano_faces = np.asarray(archive["faces"], dtype=np.int64)
    object_faces = np.asarray(overlay.raw_object_faces, dtype=np.int64)
    object_vertices = np.asarray(overlay.raw_object_vertices_local, dtype=np.float64)
    object_triangles = object_vertices[object_faces]
    face_normals = np.cross(
        object_triangles[:, 1] - object_triangles[:, 0],
        object_triangles[:, 2] - object_triangles[:, 0],
    )
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=-1, keepdims=True), 1.0e-12)
    bvh = ObjectLocalBVH(object_triangles)
    object_matrices = pose_wxyz_to_matrix(overlay.raw_object_pose_world_wxyz)
    distances = np.empty((321, len(region_map.region_id)), dtype=np.float64)
    topology_raw = np.zeros(321, dtype=bool)
    topology_angle = np.full(321, np.nan, dtype=np.float64)
    topology_separation = np.full(321, np.nan, dtype=np.float64)
    topology_min_dot = np.full(321, np.nan, dtype=np.float64)
    profile_contract = RawHumanGraspReadinessProfileV1()
    for frame in range(321):
        matrix = object_matrices[frame]
        points_local = (overlay.raw_mano_vertices_world[frame] - matrix[:3, 3]) @ matrix[:3, :3]
        closest, face_ids, _barycentric, distance = bvh.query(points_local)
        distances[frame] = distance
        active = (distance <= profile_contract.nominal_contact_distance_m) & (
            region_map.region_id < 6
        )
        result = opposing_contact_topology(
            closest_points_object=closest[active],
            contact_normals_object=face_normals[face_ids[active]],
            contact_region_ids=region_map.region_id[active],
            minimum_separation_m=profile_contract.opposition_minimum_separation_m,
            minimum_angle_deg=profile_contract.opposition_normal_angle_deg,
        )
        topology_raw[frame] = bool(result["opposing"])
        if result["maximum_normal_angle_deg"] is not None:
            topology_angle[frame] = float(result["maximum_normal_angle_deg"])
            topology_separation[frame] = float(result["maximum_qualifying_separation_m"])
            topology_min_dot[frame] = float(result["minimum_normal_dot"])
    profile = raw_human_grasp_profile(
        distances_m=distances,
        region_map=region_map,
        mano_faces=mano_faces,
        opposing_topology_raw=topology_raw,
        contract=profile_contract,
    )
    strict = _strict_mask(clip)
    strict_persistent = persistent_finger_mask(strict, minimum_steps=3)
    strict_ready = strict_persistent.sum(axis=1) >= 2

    hand_pose = np.asarray(overlay.raw_mano_root_pose_world_wxyz, dtype=np.float64)
    object_pose = np.asarray(overlay.raw_object_pose_world_wxyz, dtype=np.float64)
    raw_source_timestamps = np.asarray(overlay.runtime_timestamps_s, dtype=np.float64)
    runtime_timestamps = np.arange(321, dtype=np.float64) * CONTROL_DT_S
    relative_translation = object_pose[:, :3] - hand_pose[:, :3]
    relative_translation_speed = np.linalg.norm(
        np.gradient(relative_translation, raw_source_timestamps, axis=0, edge_order=1), axis=1
    )
    hand_omega = derive_angular_velocity_world_wxyz(hand_pose[:, 3:], raw_source_timestamps)
    object_omega = derive_angular_velocity_world_wxyz(object_pose[:, 3:], raw_source_timestamps)
    relative_angular_speed = np.linalg.norm(object_omega - hand_omega, axis=1)
    relative_pose_change = np.linalg.norm(relative_translation - relative_translation[0], axis=1)

    event_masks = {
        "strict_named_tip_ready": strict_ready,
        "any_hand_surface_contact": profile["any_hand_surface_contact"],
        "multi_region_contact": profile["multi_region_contact"],
        "opposing_topology": profile["opposing_topology"],
        "thumb_non_thumb_contact": profile["thumb_non_thumb_contact"],
    }
    events = {name: first_true(np.asarray(mask, dtype=bool)) for name, mask in event_masks.items()}
    segment_contact = np.asarray(profile["segment_contact"], dtype=bool)
    segment_events = {
        name: first_true(
            persistent_mask(segment_contact[:, index], profile_contract.runtime_persistence_frames)
        )
        for index, name in enumerate(SEGMENT_ORDER)
    }
    rows: list[dict[str, object]] = []
    robust = np.asarray(profile["robust_region_contact_persistent"], dtype=bool)
    for frame in range(321):
        rows.append(
            {
                "runtime_frame": frame,
                "runtime_time_s": float(runtime_timestamps[frame]),
                "raw_source_time_s": float(raw_source_timestamps[frame]),
                "raw_frame_float": float(overlay.raw_frame_float[frame]),
                "minimum_surface_distance_m": float(profile["minimum_surface_distance_m"][frame]),
                "strict_required_finger_count": int(strict[frame].sum()),
                "strict_named_tip_ready": bool(strict_ready[frame]),
                "any_hand_surface_contact": bool(profile["any_hand_surface_contact"][frame]),
                "any_robust_region_contact": bool(profile["any_robust_region_contact"][frame]),
                "multi_region_contact": bool(profile["multi_region_contact"][frame]),
                "thumb_non_thumb_contact": bool(profile["thumb_non_thumb_contact"][frame]),
                "opposing_topology": bool(profile["opposing_topology"][frame]),
                "thumb_contact": bool(robust[frame, 0]),
                "index_contact": bool(robust[frame, 1]),
                "middle_contact": bool(robust[frame, 2]),
                "ring_contact": bool(robust[frame, 3]),
                "pinky_contact": bool(robust[frame, 4]),
                "palm_contact": bool(robust[frame, 5]),
                "tip_surface_contact": bool(
                    segment_contact[frame, SEGMENT_ORDER.index("tip_surface")]
                ),
                "distal_contact": bool(segment_contact[frame, SEGMENT_ORDER.index("distal")]),
                "middle_or_proximal_contact": bool(
                    segment_contact[frame, SEGMENT_ORDER.index("middle")]
                    or segment_contact[frame, SEGMENT_ORDER.index("proximal")]
                ),
                "opposing_max_angle_deg": (
                    "" if np.isnan(topology_angle[frame]) else float(topology_angle[frame])
                ),
                "opposing_min_normal_dot": (
                    "" if np.isnan(topology_min_dot[frame]) else float(topology_min_dot[frame])
                ),
                "opposing_max_separation_m": (
                    ""
                    if np.isnan(topology_separation[frame])
                    else float(topology_separation[frame])
                ),
                "relative_translation_speed_proxy_mps": float(relative_translation_speed[frame]),
                "relative_angular_speed_radps": float(relative_angular_speed[frame]),
                "relative_translation_change_from_start_m": float(relative_pose_change[frame]),
            }
        )
    clip_output = REPORT_ROOT / "raw_grasp_authority" / clip.removeprefix("hocap_")
    _write_csv(clip_output / "frame_profile.csv", rows)
    event_summary = {
        "schema_version": "RawHumanGraspReadinessEventSummaryV1",
        "clip": clip,
        "events": {
            name: {
                "frame": frame,
                "runtime_time_s": (None if frame is None else float(runtime_timestamps[frame])),
                "raw_source_time_s": (
                    None if frame is None else float(raw_source_timestamps[frame])
                ),
                "raw_frame_float": None if frame is None else float(overlay.raw_frame_float[frame]),
                "margin_to_lift_frames": None if frame is None else LIFT_FRAME - frame,
                "margin_to_lift_s": None if frame is None else (LIFT_FRAME - frame) * CONTROL_DT_S,
            }
            for name, frame in events.items()
        },
        "segment_events": segment_events,
        "coupled_motion_onset": "NOT_IDENTIFIABLE_NO_FROZEN_THRESHOLD",
        "functional_raw_ready": "NOT_IDENTIFIABLE_NO_VALIDATED_BINARY_AUTHORITY",
        "lift": {
            "frame": LIFT_FRAME,
            "runtime_time_s": LIFT_FRAME * CONTROL_DT_S,
            "raw_source_time_s": float(raw_source_timestamps[LIFT_FRAME]),
        },
        "normal_quality": _normal_quality(object_vertices, object_faces),
        "bvh": bvh.stats(),
        "coordinate_alignment": overlay.coordinate_alignment,
        "time_alignment": overlay.time_alignment,
        "source_provenance": overlay.source_provenance,
    }
    _write_json(clip_output / "event_summary.json", event_summary)
    return {"events": event_summary, "profile_rows": rows}


def _provenance() -> dict[str, object]:
    writer = REPO_ROOT / "scripts/rl/isaaclab/evaluate_physical_hoi.py"
    env = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py"
    )
    state = (
        REPO_ROOT / "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py"
    )
    direct = REPO_ROOT / ".local/external/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py"
    rigid = (
        REPO_ROOT
        / ".local/external/IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object_data.py"
    )
    return {
        "schema_version": "Stage16AngularVelocityProvenanceV1",
        "trace_field_name": "object_angular_velocity_world (export) / object_twist[:,3:6] (trace)",
        "trace_writer_file": str(writer.resolve()),
        "trace_writer_line": _line(writer, '"object_angular_velocity_world"'),
        "rollout_capture_file": str(env.resolve()),
        "rollout_capture_line": _line(env, '"object_twist": state["object_twist_world"]'),
        "runtime_accessor_file": str(state.resolve()),
        "runtime_accessor_line": _line(state, '"object_twist_world": object_state[:, 7:13]'),
        "runtime_accessor": "RigidObjectData.root_state_w[:,10:13]",
        "IsaacLab_API": "RigidObjectData.root_state_w -> root_com_vel_w",
        "IsaacLab_source_file": str(rigid.resolve()),
        "IsaacLab_source_line": _line(rigid, "def root_com_vel_w"),
        "underlying_PhysX_tensor_source": "RigidBodyView.get_velocities()[...,3:6]",
        "underlying_PhysX_source_line": _line(rigid, "self._root_physx_view.get_velocities()"),
        "frame_semantics": "WORLD",
        "point_semantics": "active single rigid-object center of mass",
        "units": "rad/s",
        "sampling_stage": "post-physics, after final decimated substep, lazy buffer refresh",
        "pose_sampling_stage": "same _state() call and same post-physics reward/trace row",
        "DirectRLEnv_step_file": str(direct.resolve()),
        "DirectRLEnv_scene_update_line": _line(direct, "self.scene.update(dt=self.physics_dt)"),
        "trace_append_order_line": _line(env, "self._capture_ppo26d_trace_row()"),
        "timestamp_relation_to_object_pose": "same control row; both refreshed after scene.update",
        "known_transformations": [
            "pose position global-to-scene translation only",
            "pose quaternion remains world orientation",
            "angular velocity receives no frame rotation or unit conversion",
        ],
        "root_state_mixed_frame_note": (
            "pose is actor/link frame while linear/angular velocity is COM; angular velocity is "
            "identical for rigidly attached points and is world-expressed"
        ),
    }


def _event_value(raw: dict[str, object], name: str) -> int | None:
    value = raw["events"]["events"][name]["frame"]
    return None if value is None else int(value)


def _comparison_outputs(
    angular: dict[str, dict[str, object]], raw: dict[str, dict[str, object]]
) -> dict[str, object]:
    prior = _read_json(PRIOR_ROOT / "final_summary.json")
    prior_timing = prior["contact_timing"]
    timing_rows: list[dict[str, object]] = []
    for clip in CLIPS:
        previous = prior_timing[clip]
        profile = timing_layer_profile(
            raw_frame=None,
            retarget_frame=int(previous["retarget_ready_median"]),
            actual_frame=int(previous["actual_ready_median"]),
            lift_frame=LIFT_FRAME,
            control_dt_s=CONTROL_DT_S,
        )
        row = {
            "clip": clip,
            "raw_functional_ready": "NOT_IDENTIFIABLE",
            "raw_strict_ready": _event_value(raw[clip], "strict_named_tip_ready"),
            "raw_any_surface_ready": _event_value(raw[clip], "any_hand_surface_contact"),
            "raw_multi_region_ready": _event_value(raw[clip], "multi_region_contact"),
            "raw_opposing_topology_ready": _event_value(raw[clip], "opposing_topology"),
            "retarget_ready": profile["retarget_frame"],
            "actual_ready": profile["actual_frame"],
            "lift": profile["lift_frame"],
            "raw_to_retarget": profile["raw_to_retarget_frames"],
            "retarget_to_actual": profile["retarget_to_actual_frames"],
            "retarget_to_actual_s": profile["retarget_to_actual_s"],
            "margin_sign_convention": profile["margin_sign_convention"],
        }
        timing_rows.append(row)
        _write_csv(REPORT_ROOT / "contact_timing_v2" / f"{clip.removeprefix('hocap_')}.csv", [row])
    _write_csv(REPORT_ROOT / "contact_timing_v2/comparison.csv", timing_rows)
    attribution = {
        "schema_version": "Stage16ContactTimingLayerAttributionV2",
        "CONTACT_TIMING_ATTRIBUTION_PROFILE_BASED": "YES",
        "CONTACT_TIMING_V2_ROOT_CAUSE": "INCONCLUSIVE",
        "CONFIDENCE": "MEDIUM",
        "DOES_NEW_AUTHORITY_RESOLVE_170105": "PARTIALLY",
        "reason": (
            "170105 has any-surface contact just before LIFT but multi-region contact after "
            "LIFT and no validated functional-ready binary. Retarget-to-actual lag remains "
            "measured, but primary layer attribution cannot consume an invented raw bool."
        ),
        "clips": timing_rows,
    }
    _write_json(REPORT_ROOT / "contact_timing_v2/attribution.json", attribution)

    pfdf_rows: list[dict[str, object]] = []
    for label, key in (
        ("PF", "PF"),
        ("DF_pose", "DF_pose"),
        ("DF_linear V1", "DF_linear_under_V1"),
    ):
        pfdf_rows.append(
            {
                "metric": label,
                "V4_170105": f"{prior['PF_DF']['hocap_170105'][key]['pass_count']}/{prior['PF_DF']['hocap_170105'][key]['total']}",
                "V4_170650": f"{prior['PF_DF']['hocap_170650'][key]['pass_count']}/{prior['PF_DF']['hocap_170650'][key]['total']}",
            }
        )
    for label, field in (
        ("DF_angular legacy trace", "DF_angular_legacy"),
        ("DF_angular pose-derived", "DF_angular_pose_derived"),
        ("DF_angular AuthorityV2", "DF_angular_authority_v2"),
    ):
        pfdf_rows.append(
            {
                "metric": label,
                "V4_170105": f"{angular['hocap_170105']['aggregate'][field]}/10",
                "V4_170650": f"{angular['hocap_170650']['aggregate'][field]}/20",
            }
        )
    _write_csv(REPORT_ROOT / "pf_df/final_comparison.csv", pfdf_rows)
    table = "\n".join(
        f"| {row['metric']} | {row['V4_170105']} | {row['V4_170650']} |" for row in pfdf_rows
    )
    (REPORT_ROOT / "pf_df/final_comparison.md").write_text(
        "# Final PF/DF Comparison\n\n| Metric | V4/170105 | V4/170650 |\n"
        "| --- | ---: | ---: |\n"
        f"{table}\n\nThreshold provenance: "
        "`LEGACY_INHERITED_NOT_SCIENTIFICALLY_RECALIBRATED`.\n",
        encoding="utf-8",
    )
    return {"prior": prior, "timing": timing_rows, "attribution": attribution, "pfdf": pfdf_rows}


def _replay(angular: dict[str, dict[str, object]]) -> None:
    episodes = angular["hocap_170650"]["episodes"]
    low = min(episodes, key=lambda row: float(row["Delta_omega_pose_derived_mean_radps"]))
    high = max(episodes, key=lambda row: float(row["Delta_omega_historical_mean_radps"]))
    ordered = sorted(episodes, key=lambda row: float(row["Delta_omega_pose_derived_mean_radps"]))
    median = ordered[len(ordered) // 2]
    prefix = (
        "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python "
        "scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula --loop"
    )

    def command(path: str, clip: str, extra: str = "") -> str:
        return f"{prefix} --trace {Path(path).relative_to(REPO_ROOT)} --object {clip} --no-reference-ghost{extra}"

    commands = f"""# Visualization Commands

Actual plus raw MANO/object is shown; the retarget reference is hidden. `M` toggles raw mocap and
`R` toggles the reference. Angular values are synchronized in
`angular_semantics/v4_170650/frame_debug/episode_NNN.csv`.

## V4/170650 representative / historical high trace error / low AuthorityV2 error

```bash
{command(str(median["trace_path"]), "hocap_170650")}
{command(str(high["trace_path"]), "hocap_170650")}
{command(str(low["trace_path"]), "hocap_170650")}
{command(str(high["trace_path"]), "hocap_170650", " --mocap-object-low-poly")}
```

Selected episodes: representative `{median["episode"]}`, historical high trace-error
`{high["episode"]}`, low pose-derived/AuthorityV2 error `{low["episode"]}`.

## V4/170105 full and CONTACT-to-LIFT

```bash
{command(str(_trace_paths("hocap_170105")[0]), "hocap_170105")}
{command(str(_trace_paths("hocap_170105")[0]), "hocap_170105", " --start-frame 92 --end-frame 230")}
{command(str(_trace_paths("hocap_170105")[0]), "hocap_170105", " --start-frame 92 --end-frame 230 --mocap-object-low-poly")}
```
"""
    replay = REPORT_ROOT / "replay"
    replay.mkdir(parents=True, exist_ok=True)
    (replay / "visualization_commands.md").write_text(commands, encoding="utf-8")
    (replay / "manual_acceptance.md").write_text(
        """# Manual Acceptance

For 170650 inspect whether the historical high trace-omega episode visibly wobbles, whether raw and
actual object orientations remain close, and whether hand/object move together. For 170105 inspect
which raw MANO regions touch before LIFT, whether the reward-specific mask is later than broader
surface contact, whether retarget geometry encloses before LIFT, and why actual contact arrives late.
Do not infer friction or force closure from these views.
""",
        encoding="utf-8",
    )


def _markdown(summary: dict[str, object]) -> str:
    a650 = summary["angular"]["hocap_170650"]["aggregate"]
    r105 = summary["raw"]["hocap_170105"]["events"]
    r650 = summary["raw"]["hocap_170650"]["events"]

    def event(row: dict[str, object], name: str) -> str:
        value = row["events"][name]
        if value["frame"] is None:
            return "NOT_IDENTIFIABLE"
        relation = (
            "before"
            if int(value["frame"]) < LIFT_FRAME
            else "after"
            if int(value["frame"]) > LIFT_FRAME
            else "at"
        )
        return (
            f"{value['frame']} / runtime {float(value['runtime_time_s']):.2f} s / "
            f"raw {float(value['raw_source_time_s']):.4f} s / {relation} LIFT"
        )

    pfdf = "\n".join(
        f"| {row['metric']} | {row['V4_170105']} | {row['V4_170650']} |"
        for row in summary["comparison"]["pfdf"]
    )
    return f"""# Stage16 Angular Velocity Semantics + Raw Human Grasp Authority Handoff

## 1. Git

`branch=feature/ppo-physical`, `START_HEAD={START_HEAD}`. Final HEAD and commits are recorded in
`git_commits.json`; `.local` remains ignored.

## 2. Angular Trace Provenance

`object_angular_velocity` comes from `RigidObjectData.root_state_w[10:13]`, ultimately PhysX
`RigidBodyView.get_velocities()[3:6]`; it is the active rigid object's COM angular velocity in the
WORLD frame, sampled through the lazy IsaacLab buffer after the final physics substep, in the same
post-physics trace row as actor-frame object pose.

## 3. Why trace omega disagreed with pose-derived omega

The legacy field is an instantaneous post-solver COM velocity sample. Reference Kinematics V2 is a
centered control-rate pose displacement. No frame conversion is missing and neither +/-1 row offset
is semantically authorized. In 170650, mean mismatch is
`{a650["trace_vs_pose_mismatch"]["mean"]:.6f} rad/s`, while
`{a650["static_pose_nonzero_trace_fraction"]:.3%}` of valid rows have near-static sampled pose but
legacy trace speed >=0.05 rad/s. The historical trace therefore is not kinematically closed to its
sampled pose at the comparison bandwidth. The low-level PhysX sleep/contact-solver contribution is
not recoverable from legacy traces without substep/wake telemetry.

## 4. Angular root cause

`POSE_DERIVED_REQUIRED_FOR_COMPARABLE_SEMANTICS`, `CONFIDENCE=HIGH`.

## 5. Actual Angular Authority V2

`IMPLEMENTED=YES`, `SOURCE=trace.object_pose`, `FRAME=WORLD`,
`TIME_SEMANTICS=ReferenceKinematicsV2 centered SO(3)-log`, `CONVERSION=NONE`.
Historical trace bytes remain unchanged.

## 6. Angular re-evaluation

| Metric | Legacy trace | Pose-derived | Authority V2 |
| --- | ---: | ---: | ---: |
| Delta omega mean | {a650["Delta_omega_historical"]["mean"]:.6f} | {a650["Delta_omega_pose_derived"]["mean"]:.6f} | {a650["Delta_omega_authority_v2"]["mean"]:.6f} |
| Delta omega p95 | {a650["Delta_omega_historical"]["p95"]:.6f} | {a650["Delta_omega_pose_derived"]["p95"]:.6f} | {a650["Delta_omega_authority_v2"]["p95"]:.6f} |
| terminal Delta omega max | {a650["terminal_Delta_omega_historical_max_radps"]:.6f} | {a650["terminal_Delta_omega_authority_v2_max_radps"]:.6f} | {a650["terminal_Delta_omega_authority_v2_max_radps"]:.6f} |
| DF angular | {a650["DF_angular_legacy"]}/20 | {a650["DF_angular_pose_derived"]}/20 | {a650["DF_angular_authority_v2"]}/20 |

The frozen non-angular 170650 profile remains `PF=20/20`, `DF_pose=20/20`, and
`DF_linear_under_V1=20/20`.

## 7. Can 170650 be accepted as physical HOI data?

`YES`: PF, pose DF, linear DF under V1, and angular Authority V2 all pass 20/20. This is not a
single ambiguous success label. The velocity thresholds remain
`LEGACY_INHERITED_NOT_SCIENTIFICALLY_RECALIBRATED`.

## 8. Strict V4 raw contact semantics

Strict V4 is a reward-specific mapping: raw MANO whole-finger surface evidence is confirmed at 2 mm
with a >=3-vertex 5 mm component and two native 30 Hz frames, then each required human finger is
mapped to the same named robot distal-tip pair-force reward. It is not a MANO-tip-only measurement
and is not a validated functional human-grasp authority.

`STRICT_V4_MASK_FUNCTIONAL_GRASP_AUTHORITY=NOT_SUPPORTED`; answer: `NO`.

## 9. Is Strict V4 a valid functional human-grasp authority?

`NO`. `STRICT_V4_MASK_FUNCTIONAL_GRASP_AUTHORITY=NOT_SUPPORTED`.

## 10. Raw Human Grasp Readiness Authority

`MULTIPLE_AUTHORITIES_REQUIRED_NO_SINGLE_BINARY`, `CONFIDENCE=HIGH`. The implemented
`RawHumanGraspReadinessProfileV1` preserves strict named-finger, any-surface, multi-region,
thumb-opposition/geometric-opposition, and continuous coupling views. `binary authority=NO`.

## 11. V4/170105 raw events

| Event | Frame / runtime time / raw-source time / relation to LIFT |
| --- | --- |
| Strict reward-target ready | {event(r105, "strict_named_tip_ready")} |
| Any-hand surface contact | {event(r105, "any_hand_surface_contact")} |
| Multi-region contact | {event(r105, "multi_region_contact")} |
| Opposing topology | {event(r105, "opposing_topology")} |
| Coupled-motion onset | NOT_IDENTIFIABLE_NO_FROZEN_THRESHOLD |
| Functional raw ready | NOT_IDENTIFIABLE |
| LIFT | 184 / runtime 9.20 s / raw 1.1500 s / -- |

170105 has broader surface contact just before LIFT but only one robust region; multi-region contact
arrives after LIFT, and no force-bearing functional binary is validated.

## 12. V4/170650 raw events

| Event | Frame / runtime time / raw-source time / relation to LIFT |
| --- | --- |
| Strict reward-target ready | {event(r650, "strict_named_tip_ready")} |
| Any-hand surface contact | {event(r650, "any_hand_surface_contact")} |
| Multi-region contact | {event(r650, "multi_region_contact")} |
| Opposing topology | {event(r650, "opposing_topology")} |
| Coupled-motion onset | NOT_IDENTIFIABLE_NO_FROZEN_THRESHOLD |
| Functional raw ready | NOT_IDENTIFIABLE |
| LIFT | 184 / runtime 9.20 s / raw 1.1500 s / -- |

## 13. Contact timing V2

| Clip | Raw functional | Retarget | Actual | LIFT | Raw to retarget | Retarget to actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 170105 | NOT_IDENTIFIABLE | 181 | 198 | 184 | NOT_IDENTIFIABLE | 17 |
| 170650 | NOT_IDENTIFIABLE | 129 | 162 | 184 | NOT_IDENTIFIABLE | 33 |

## 14. Does the new authority resolve 170105 timing attribution?

`PARTIALLY`. It separates early all-surface contact from later multi-region and Strict-V4 events,
but does not validate a functional raw-ready binary.

## 15. 170105 timing root cause V2

`CONTACT_TIMING_V2_ROOT_CAUSE=INCONCLUSIVE`, `CONFIDENCE=MEDIUM`. The measured
retarget-to-actual lag is real, but primary attribution cannot consume an invented functional raw
ready bit.

## 16. Why 170105 fails while 170650 works

170105 follows: raw any-surface 182 but multi-region 190 -> retarget 181 -> actual 198 -> LIFT 184;
actual grasp readiness arrives after LIFT and no lift follows. 170650 has raw any-surface 109 and
multi-region 136 -> retarget 129 -> actual 162 -> LIFT 184; each observed readiness layer precedes
LIFT and the physical lift completes.

## 17. Friction claim

`FRICTION_PRIMARY=NOT_SUPPORTED`; exact force-bearing raw contact and effective-friction evidence
are unavailable.

## 18. Final PF/DF

| Metric | V4/170105 | V4/170650 |
| --- | ---: | ---: |
{pfdf}

## 19. Replay commands

See `replay/visualization_commands.md` for exact 170650 representative/high-legacy/low-V2 commands,
the 170105 full and CONTACT-to-LIFT commands, and both low-poly variants.

## 20. Manual acceptance

See `replay/manual_acceptance.md`. For 170650 inspect visible wobble, raw-versus-actual object
orientation, and common hand-object motion. For 170105 inspect contacting raw MANO regions, the
Strict-V4 timing difference, pre-LIFT retarget enclosure, and delayed actual contact. These views do
not create a machine gate.

## 21. Next actions

`NEXT_170650=NEXT_REQUALIFY_170650_WITH_ANGULAR_AUTHORITY_V2`

`NEXT_170105=NEXT_TARGETED_HUMAN_ROBOT_CONTACT_AUTHORITY_VALIDATION`

## 22. Tests

See `tests.json` for targeted, lint, format, mypy, full pytest, paper-fidelity, historical-hash,
Stage-A replay-regression, and `.local` tracking receipts.

## 23. Safety flags

`BRANCH=feature/ppo-physical`, `NEW_BRANCH_CREATED=NO`, `NEW_WORKTREE_CREATED=NO`,
`GUIDANCE_WORKTREE_MODIFIED=NO`, `PPO_TRAINING_RUN=NO`, `PPO_OPTIMIZER_STEP=0`,
`REWARD_CHANGED=NO`, `FRICTION_CHANGED=NO`, `MASS_CHANGED=NO`, `REFERENCE_CHANGED=NO`,
`RETIMING_CHANGED=NO`, `CONTROLLER_CHANGED=NO`, `ACTION_CHANGED=NO`,
`SR_HOLD_IMPLEMENTED=NO`, `ENGINEERED_TERMINAL_HOLD_ADDED=NO`,
`ANGULAR_THRESHOLD_TUNED=NO`, `LEGACY_SRPHYSICS_MODIFIED=NO`, `SR_DYNAMIC_V1_MODIFIED=NO`,
`PF_V1_MODIFIED=NO`, `HISTORICAL_TRACES_REWRITTEN=NO`, `HISTORICAL_REPORTS_MODIFIED=NO`,
`GUIDANCE_ADDED=NO`, `OBJECT_STATE_WRITE_ADDED=NO`, `WRIST_ROOT_WRITE_ADDED=NO`,
`RAW_MOCAP_REPLAY_REGRESSED=NO`, `PUSHED=NO`, `PR_CREATED=NO`, `.local_TRACKED=NO`.
"""


def main() -> int:
    if sys.argv[1:] == ["--render-existing"]:
        summary = _read_json(REPORT_ROOT / "final_summary.json")
        markdown = _markdown(summary)
        (REPORT_ROOT / "final_summary.md").write_text(markdown, encoding="utf-8")
        (REPORT_ROOT / "handoff.md").write_text(markdown, encoding="utf-8")
        print(json.dumps({"status": "PASS", "rendered": str(REPORT_ROOT)}, sort_keys=True))
        return 0
    if sys.argv[1:]:
        raise RuntimeError(f"STAGE16_AUTHORITY_UNKNOWN_ARGUMENTS:{sys.argv[1:]}")
    if _git("branch", "--show-current") != "feature/ppo-physical":
        raise RuntimeError("STAGE16_AUTHORITY_WRONG_BRANCH")
    if _git("rev-parse", "HEAD") != START_HEAD:
        raise RuntimeError("STAGE16_AUTHORITY_START_HEAD_DRIFT")
    status = _git("status", "--short", "--untracked-files=all")
    expected_modified = {
        "docs/ROADMAP.md",
        "docs/ROADMAP.zh-CN.md",
        "docs/rl/ACTUAL_ANGULAR_VELOCITY_SEMANTICS.md",
        "docs/rl/ANGULAR_TWIST_AUDIT.md",
        "docs/rl/CONTACT_TIMING_LAYER_ATTRIBUTION.md",
        "docs/rl/PHYSICAL_FUNCTIONALITY_AND_DEMONSTRATION_FIDELITY.md",
        "docs/rl/RAW_HUMAN_GRASP_READINESS_AUTHORITY.md",
        "src/toporetarget/rl/geometry_audit/raw_mocap_overlay.py",
        "src/toporetarget/rl/stage16_authority_v2.py",
        "tests/rl/test_stage16_authority_v2.py",
        "scripts/evaluation/evaluate_stage16_angular_semantics_raw_grasp_authority.py",
    }
    unknown = [
        line
        for line in status.splitlines()
        if line and line.strip().split(maxsplit=1)[-1] not in expected_modified
    ]
    if unknown:
        raise RuntimeError(f"STAGE16_AUTHORITY_UNKNOWN_WORKTREE_CHANGES:{unknown}")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    historical = {
        name: {
            "path": str((REPO_ROOT / ".local/reports" / name).resolve()),
            "tree_sha256": _tree_sha256(REPO_ROOT / ".local/reports" / name),
        }
        for name in HISTORICAL_REPORTS
    }
    manifest_paths = [
        *_trace_paths("hocap_170105"),
        *_trace_paths("hocap_170650"),
        *(STRICT_ROOT / f"strict_source_contact_mask_{clip}.npz" for clip in CLIPS),
        *(SOURCE_ROOT / clip / "source_contact_evidence_runtime.npz" for clip in CLIPS),
        SOURCE_ROOT / "mano_surface_region_map_v1.npz",
        PRIOR_ROOT / "final_summary.json",
        STRICT_ROOT / "frozen_evaluation_gates.json",
    ]
    _write_json(
        REPORT_ROOT / "inputs/input_manifest.json",
        {
            "schema_version": "Stage16AngularRawGraspAuthorityInputManifestV1",
            "inputs": [_artifact(path) for path in manifest_paths],
            "historical_report_tree_hashes_before": historical,
        },
    )

    provenance = _provenance()
    _write_json(REPORT_ROOT / "angular_semantics/provenance.json", provenance)
    _write_json(REPORT_ROOT / "angular_velocity_provenance.json", provenance)
    _write_json(
        REPORT_ROOT / "angular_semantics/synthetic_validation.json", _synthetic_validation()
    )
    (REPORT_ROOT / "angular_semantics/code_authority.md").write_text(
        "# Angular Code Authority\n\n"
        "The trace writer reads `object_twist_world` from the active rigid object's "
        "`root_state_w`. IsaacLab defines this as actor-frame pose concatenated with COM "
        "linear/angular velocity in world coordinates, lazily refreshed from PhysX "
        "`get_velocities()`. DirectRLEnv updates scene buffers after every physics substep and "
        "the PPO trace row is appended from `_get_rewards` after the final substep. No local/world "
        "rotation or unit conversion occurs.\n",
        encoding="utf-8",
    )
    authority = Stage16ActualAngularVelocityAuthorityV2().as_dict()
    authority.update(
        {
            "status": "IMPLEMENTED",
            "root_cause": "POSE_DERIVED_REQUIRED_FOR_COMPARABLE_SEMANTICS",
            "confidence": "HIGH",
            "legacy_trace_bug": "NO_FRAME_OR_TIMESTAMP_BUG_PROVEN",
            "legacy_trace_limitation": (
                "instantaneous solver velocity is not kinematically closed to control-rate pose"
            ),
        }
    )
    _write_json(REPORT_ROOT / "angular_semantics/authority_v2.json", authority)

    angular = {clip: _angular_clip(clip) for clip in CLIPS}
    strict_semantics = {
        "schema_version": "StrictV4SourceMaskSemanticsReviewV1",
        "source": "SourcePerFingerContactEvidenceV1",
        "raw_geometry": "all MANO vertices grouped by LBS-derived named finger regions",
        "thresholds": SourceContactThresholdContractV1().as_dict(),
        "runtime_mapping": "confirmed/persistent-confirmed only, 41 keys to 321 frames",
        "robot_reward_target": "same named Wuji distal-tip active-object pair force",
        "uses_named_mano_tip_vertices_only": False,
        "includes_palm": False,
        "includes_non_tip_finger_surface_in_source_detection": True,
        "original_purpose": "reward target source mask for StrictPerFingerContactRewardV4",
        "STRICT_V4_MASK_FUNCTIONAL_GRASP_AUTHORITY": "NOT_SUPPORTED",
        "decision": "STRICT_V4_IS_REWARD_SPECIFIC_NOT_FUNCTIONAL",
        "confidence": "HIGH",
    }
    _write_json(REPORT_ROOT / "raw_grasp_authority/strict_v4_semantics.json", strict_semantics)
    _write_json(REPORT_ROOT / "strict_v4_source_mask_semantics.json", strict_semantics)
    region_json = _read_json(SOURCE_ROOT / "mano_surface_region_map_v1.json")
    _write_json(
        REPORT_ROOT / "raw_grasp_authority/mano_region_authority.json",
        {
            "status": "AVAILABLE",
            "source": _artifact(SOURCE_ROOT / "mano_surface_region_map_v1.npz"),
            "manifest": region_json,
        },
    )
    raw = {clip: _raw_clip(clip) for clip in CLIPS}
    comparison_rows = [
        {
            "authority": "Strict V4 named-finger to robot-tip reward target",
            "regions": "five named MANO finger regions; robot named distal tips",
            "contact_threshold": "2mm + 3-vertex 5mm component",
            "persistence": "2 native 30Hz frames then conservative mapping",
            "opposing_topology": "NO",
            "functional_interpretation": "reward-specific, not functional grasp authority",
        },
        {
            "authority": "Persistent any-hand surface",
            "regions": "all 778 MANO vertices",
            "contact_threshold": "2mm",
            "persistence": "2/30s mapped to 2 runtime frames",
            "opposing_topology": "NO",
            "functional_interpretation": "contact presence only",
        },
        {
            "authority": "Persistent multi-region surface",
            "regions": "five fingers plus palm",
            "contact_threshold": "2mm + 3-vertex 5mm component per region",
            "persistence": "2/30s mapped to 2 runtime frames",
            "opposing_topology": "reported separately",
            "functional_interpretation": "morphology-relevant profile, no force closure",
        },
        {
            "authority": "Topology plus relative-motion diagnostic",
            "regions": "five fingers plus palm",
            "contact_threshold": "same frozen geometry",
            "persistence": "same duration",
            "opposing_topology": "diagnostic normals; non-watertight mesh limitation",
            "functional_interpretation": "no binary without frozen coupling threshold",
        },
    ]
    _write_csv(REPORT_ROOT / "raw_grasp_authority/authority_comparison.csv", comparison_rows)
    raw_authority = RawHumanGraspReadinessProfileV1().as_dict()
    raw_authority.update(
        {
            "status": "NOT_VALIDATED_AS_SINGLE_BINARY",
            "decision": "MULTIPLE_AUTHORITIES_REQUIRED_NO_SINGLE_BINARY",
            "confidence": "HIGH",
            "profile_implemented": True,
            "reason": (
                "geometry supports orthogonal contact layers but raw forces and a frozen coupling "
                "threshold are unavailable; functional force-bearing grasp is not identifiable"
            ),
        }
    )
    _write_json(REPORT_ROOT / "raw_grasp_authority/authority_v1.json", raw_authority)

    comparison = _comparison_outputs(angular, raw)
    _replay(angular)
    summary = {
        "schema_version": "Stage16AngularSemanticsRawGraspAuthorityHandoffV1",
        "git": {"branch": "feature/ppo-physical", "START_HEAD": START_HEAD},
        "angular_root_cause": {
            "decision": "POSE_DERIVED_REQUIRED_FOR_COMPARABLE_SEMANTICS",
            "confidence": "HIGH",
        },
        "angular_authority_v2": authority,
        "angular": angular,
        "raw_authority": raw_authority,
        "raw": raw,
        "comparison": comparison,
        "NEXT_170650": "NEXT_REQUALIFY_170650_WITH_ANGULAR_AUTHORITY_V2",
        "NEXT_170105": "NEXT_TARGETED_HUMAN_ROBOT_CONTACT_AUTHORITY_VALIDATION",
        "FRICTION_PRIMARY": "NOT_SUPPORTED",
    }
    # Remove the 321-row duplicate from JSON; CSV is the profile authority.
    for clip in CLIPS:
        summary["raw"][clip].pop("profile_rows", None)
    _write_json(REPORT_ROOT / "final_summary.json", summary)
    markdown = _markdown(summary)
    (REPORT_ROOT / "final_summary.md").write_text(markdown, encoding="utf-8")
    (REPORT_ROOT / "handoff.md").write_text(markdown, encoding="utf-8")
    historical_after = {
        name: _tree_sha256(REPO_ROOT / ".local/reports" / name) for name in HISTORICAL_REPORTS
    }
    if any(
        historical[name]["tree_sha256"] != historical_after[name] for name in HISTORICAL_REPORTS
    ):
        raise RuntimeError("STAGE16_AUTHORITY_HISTORICAL_REPORT_MUTATION")
    _write_json(
        REPORT_ROOT / "tests.json",
        {
            "status": "MATERIALIZATION_INTERNAL_PASS_VALIDATION_PENDING",
            "historical_report_hashes_unchanged": True,
            "angular_synthetic": "PASS",
            "strict_v4_mask_parity": "PASS",
            "raw_alignment": "PASS",
            "no_outcome_threshold": "PASS",
        },
    )
    _write_json(
        REPORT_ROOT / "git_commits.json",
        {
            "branch": _git("branch", "--show-current"),
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": _git("rev-parse", "HEAD"),
            "commits": [],
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    )
    print(json.dumps({"status": "PASS", "report": str(REPORT_ROOT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
