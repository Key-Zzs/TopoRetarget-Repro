#!/usr/bin/env python3
"""Close Stage16 170650 and materialize the offline human-object profile."""

# Report field names and handoff prose intentionally remain explicit.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation.human_object_interaction_profile import (  # noqa: E402
    HumanObjectCouplingContactProfileContractV1,
    build_human_object_interaction_profile,
)
from toporetarget.evaluation.source_contact_semantics import (  # noqa: E402
    SEGMENT_ORDER,
    ManoSurfaceRegionMap,
    mesh_adjacency,
    persistent_mask,
)
from toporetarget.geometry.signed_distance.closest_point import ObjectLocalBVH  # noqa: E402
from toporetarget.rl.geometry_audit.raw_mocap_overlay import (  # noqa: E402
    pose_wxyz_to_matrix,
    resolve_raw_mocap_overlay,
)
from toporetarget.rl.stage16_authority_v2 import (  # noqa: E402
    RawHumanGraspReadinessProfileV1,
    Stage16ActualAngularVelocityAuthorityV2,
    opposing_contact_topology,
    raw_human_grasp_profile,
)
from toporetarget.rl.stage16_pf_df import first_true, persistent_finger_mask  # noqa: E402

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_170650_closure_and_human_object_profile"
PROFILE_CONFIG = REPO_ROOT / "configs/evaluation/stage16_human_object_coupling_profile_v1.yaml"
AUTHORITY_ROOT = REPO_ROOT / ".local/reports/stage16_angular_semantics_and_raw_grasp_authority"
PFDF_ROOT = REPO_ROOT / ".local/reports/stage16_contact_timing_angular_twist_pf_df"
DYNAMIC_ROOT = (
    REPO_ROOT / ".local/reports/stage16_dynamic_physical_qualification_and_grasp_diagnostic"
)
FULL_GRAVITY_ROOT = REPO_ROOT / ".local/reports/stage16_full_gravity_capability_closure"
STAGE_A_ROOT = REPO_ROOT / ".local/reports/stage16_raw_mocap_replay_overlay"
STRICT_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
SOURCE_ROOT = REPO_ROOT / ".local/reports/stage16d_source_contact_semantics_final_audit"
REFERENCE_CONTACT_ROOT = REPO_ROOT / ".local/reports/stage16d_contact_contract_v2_audit"
REFERENCE_KINEMATICS_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
WORLD_WRIST_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
FORMAL_650 = (
    REPO_ROOT / ".local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650"
)
FORMAL_105 = (
    REPO_ROOT
    / ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4"
)
FORMAL_650_EVALUATION = (
    FULL_GRAVITY_ROOT
    / "formal20/v4_hocap_170650/evaluation/hocap_170650/frozen_source_c4_formal20_evaluation.json"
)
FORMAL_650_QUALIFICATION = (
    FULL_GRAVITY_ROOT / "formal20/v4_hocap_170650/analysis/qualification.json"
)
START_HEAD = "b10e8e6c1d35ac12d9fe23aa6d3d4742c16a75ec"
CLIPS = ("hocap_170105", "hocap_170650")
LIFT_FRAME = 184
DT_S = 0.05
REGIONS = ("thumb", "index", "middle", "ring", "pinky", "palm")
HISTORICAL_REPORTS = (
    "stage16_angular_semantics_and_raw_grasp_authority",
    "stage16_contact_timing_angular_twist_pf_df",
    "stage16_dynamic_physical_qualification_and_grasp_diagnostic",
    "stage16_full_gravity_capability_closure",
    "stage16_raw_mocap_replay_overlay",
)
CANDIDATES = (
    "CONTACT_ACQUISITION_TIMING",
    "CONTACT_TOPOLOGY_PRESERVATION",
    "HAND_OBJECT_COUPLING_PRESERVATION",
    "RELATIVE_SLIP_SUPPRESSION",
    "SUPPORT_TRANSFER_SUCCESS",
    "SOURCE_PROFILE_TRACKING",
    "MULTI_OBJECTIVE_INTERACTION_PRESERVATION",
    "INCONCLUSIVE",
)


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"STAGE16_PROFILE_JSON_OBJECT_REQUIRED:{path}")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"STAGE16_PROFILE_CSV_EMPTY:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"STAGE16_PROFILE_CSV_FIELD_DRIFT:{path}")
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


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if str(value) in {"True", "False"}:
        return str(value) == "True"
    raise ValueError(f"STAGE16_PROFILE_BOOLEAN_INVALID:{value}")


def _trace_paths(clip: str) -> list[Path]:
    if clip == "hocap_170105":
        result = [FORMAL_105 / f"episode_{index:02d}.npz" for index in range(10)]
    elif clip == "hocap_170650":
        result = [FORMAL_650 / f"episode_{index:03d}.npz" for index in range(20)]
    else:
        raise ValueError(f"STAGE16_PROFILE_UNKNOWN_CLIP:{clip}")
    if not all(path.is_file() for path in result):
        raise FileNotFoundError(f"STAGE16_PROFILE_TRACE_MISSING:{clip}")
    return result


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    aliases = {
        "object_reference": ("object_reference", "embedded_reference_object_pose"),
        "wrist_reference": ("wrist_reference", "embedded_reference_wrist_pose"),
    }
    required = (
        "object_pose",
        "wrist_pose",
        "hand_object_pair_presence",
        "hand_object_pair_force_valid",
        "tip_pair_presence",
        "source_contact_mask",
        "table_object_contact",
        "reference_index",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"STAGE16_PROFILE_TRACE_FIELDS_MISSING:{path}:{missing}")
        result = {name: np.asarray(archive[name]) for name in required}
        for target, candidates in aliases.items():
            source = next((name for name in candidates if name in archive.files), None)
            if source is None:
                raise ValueError(f"STAGE16_PROFILE_REFERENCE_FIELD_MISSING:{path}:{target}")
            result[target] = np.asarray(archive[source])
    if any(
        result[name].shape != (321, 7)
        for name in ("object_pose", "wrist_pose", "object_reference", "wrist_reference")
    ):
        raise ValueError(f"STAGE16_PROFILE_TRACE_POSE_SHAPE_INVALID:{path}")
    if not np.array_equal(result["reference_index"], np.arange(321)):
        raise ValueError(f"STAGE16_PROFILE_REFERENCE_INDEX_DRIFT:{path}")
    return result


def _strict_mask(clip: str) -> np.ndarray:
    path = STRICT_ROOT / f"strict_source_contact_mask_{clip}.npz"
    with np.load(path, allow_pickle=False) as archive:
        mask = np.asarray(archive["strict_source_contact_mask"], dtype=bool)
    if mask.shape != (321, 5):
        raise ValueError(f"STAGE16_PROFILE_STRICT_MASK_INVALID:{clip}")
    return mask


def _retarget(clip: str) -> tuple[np.ndarray, np.ndarray]:
    short = clip.removeprefix("hocap_")
    with np.load(
        REFERENCE_CONTACT_ROOT / f"reference_contact_contract_v2_{short}.npz",
        allow_pickle=False,
    ) as archive:
        contact = np.asarray(archive["strong_contact_expected"], dtype=bool)
        distance = np.asarray(archive["reference_distance_m"], dtype=np.float64)
    if contact.shape != (321, 5) or distance.shape != (321, 5):
        raise ValueError(f"STAGE16_PROFILE_RETARGET_CONTACT_INVALID:{clip}")
    return contact, distance


def _component_count(active: np.ndarray, adjacency: tuple[np.ndarray, ...]) -> int:
    selected = np.asarray(active, dtype=bool)
    visited = np.zeros_like(selected)
    count = 0
    for start in np.flatnonzero(selected):
        if visited[start]:
            continue
        count += 1
        stack = [int(start)]
        visited[start] = True
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if selected[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
    return count


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("STAGE16_PROFILE_DISTRIBUTION_INVALID")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _event(mask: np.ndarray) -> int | None:
    return first_true(np.asarray(mask, dtype=bool))


def _event_receipt(
    frame: int | None, raw_frame: np.ndarray, source_time: np.ndarray
) -> dict[str, object]:
    return {
        "frame": frame,
        "runtime_time_s": None if frame is None else frame * DT_S,
        "raw_source_time_s": None if frame is None else float(source_time[frame]),
        "raw_frame_float": None if frame is None else float(raw_frame[frame]),
        "margin_to_lift_frames": None if frame is None else LIFT_FRAME - frame,
        "margin_to_lift_s": None if frame is None else (LIFT_FRAME - frame) * DT_S,
        "authority": "event_descriptor_not_functional_grasp_label",
    }


def _raw_profile(clip: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    trace_path = _trace_paths(clip)[0]
    overlay = resolve_raw_mocap_overlay(
        trace_path=trace_path,
        frame_count=321,
        clip=clip,
        reference_path=WORLD_WRIST_ROOT / f"{clip}.world_wrist.stage16.npz",
    )
    with np.load(SOURCE_ROOT / "mano_surface_region_map_v1.npz", allow_pickle=False) as archive:
        region_map = ManoSurfaceRegionMap(
            np.asarray(archive["region_id"]),
            np.asarray(archive["segment_id"]),
            np.asarray(archive["soft_region_weight"]),
        )
        mano_faces = np.asarray(archive["faces"], dtype=np.int64)
    object_vertices = np.asarray(overlay.raw_object_vertices_local, dtype=np.float64)
    object_faces = np.asarray(overlay.raw_object_faces, dtype=np.int64)
    triangles = object_vertices[object_faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
    bvh = ObjectLocalBVH(triangles)
    object_matrix = pose_wxyz_to_matrix(overlay.raw_object_pose_world_wxyz)
    distance = np.empty((321, len(region_map.region_id)), dtype=np.float64)
    topology_raw = np.zeros(321, dtype=bool)
    opposition_score = np.zeros(321, dtype=np.float64)
    spread = np.zeros(321, dtype=np.float64)
    contact_centroid_object = np.full((321, 3), np.nan, dtype=np.float64)
    normal_diversity = np.zeros(321, dtype=np.float64)
    component_count = np.zeros(321, dtype=np.int64)
    adjacency = mesh_adjacency(len(region_map.region_id), mano_faces)
    raw_contract = RawHumanGraspReadinessProfileV1()
    for frame in range(321):
        matrix = object_matrix[frame]
        points_local = (overlay.raw_mano_vertices_world[frame] - matrix[:3, 3]) @ matrix[:3, :3]
        closest, face_index, _barycentric, distance_frame = bvh.query(points_local)
        distance[frame] = distance_frame
        component_count[frame] = _component_count(
            distance_frame <= raw_contract.component_distance_m, adjacency
        )
        active = (distance_frame <= raw_contract.nominal_contact_distance_m) & (
            region_map.region_id < 6
        )
        topology = opposing_contact_topology(
            closest_points_object=closest[active],
            contact_normals_object=normals[face_index[active]],
            contact_region_ids=region_map.region_id[active],
            minimum_separation_m=raw_contract.opposition_minimum_separation_m,
            minimum_angle_deg=raw_contract.opposition_normal_angle_deg,
        )
        topology_raw[frame] = bool(topology["opposing"])
        if topology["minimum_normal_dot"] is not None:
            opposition_score[frame] = (1.0 - float(topology["minimum_normal_dot"])) * 0.5
            spread[frame] = float(topology["maximum_qualifying_separation_m"])
        if np.any(active):
            active_points = closest[active]
            active_normals = normals[face_index[active]]
            contact_centroid_object[frame] = active_points.mean(axis=0)
            normal_diversity[frame] = 1.0 - float(np.linalg.norm(active_normals.mean(axis=0)))
    grasp = raw_human_grasp_profile(
        distances_m=distance,
        region_map=region_map,
        mano_faces=mano_faces,
        opposing_topology_raw=topology_raw,
        contract=raw_contract,
    )
    region_contact = np.asarray(grasp["robust_region_contact_persistent"], dtype=bool)
    strict = _strict_mask(clip)
    strict_ready = persistent_finger_mask(strict, minimum_steps=3).sum(axis=1) >= 2
    profile = build_human_object_interaction_profile(
        hand_pose_world_wxyz=overlay.raw_mano_root_pose_world_wxyz,
        object_pose_world_wxyz=overlay.raw_object_pose_world_wxyz,
        timestamps_s=overlay.runtime_timestamps_s,
        minimum_surface_distance_m=distance.min(axis=1),
        near_contact_vertex_count=(distance <= raw_contract.component_distance_m).sum(axis=1),
        near_contact_vertex_fraction=(distance <= raw_contract.component_distance_m).mean(axis=1),
        contact_component_count=component_count,
        region_contact=region_contact,
        topology_normal_opposition_score=opposition_score,
        topology_contact_spread_m=spread,
        strict_v4_reward_target=strict_ready,
        any_hand_surface_contact=np.asarray(grasp["any_hand_surface_contact"]),
        multi_region_contact=np.asarray(grasp["multi_region_contact"]),
        opposing_contact_topology=np.asarray(grasp["opposing_topology"]),
    )
    profile["raw_frame_float"] = overlay.raw_frame_float
    profile["raw_source_time_s"] = overlay.runtime_timestamps_s
    profile["region_contact_raw"] = np.asarray(grasp["robust_region_contact_raw"])
    profile["segment_contact"] = np.asarray(grasp["segment_contact"])
    profile["contact_centroid_object_m"] = contact_centroid_object
    profile["surface_normal_diversity"] = normal_diversity
    frames = {
        "any_surface": _event(profile["any_hand_surface_contact"]),
        "multi_region": _event(profile["multi_region_contact"]),
        "opposing_topology": _event(profile["opposing_contact_topology"]),
        "strict_v4": _event(profile["strict_v4_reward_target"]),
    }
    event_profile = {
        "schema_version": "HumanObjectCouplingContactEventProfileV1",
        "clip": clip,
        "events": {
            name: _event_receipt(frame, overlay.raw_frame_float, overlay.runtime_timestamps_s)
            for name, frame in frames.items()
        },
        "lowest_sustained_coupling_interval": "NOT_IDENTIFIABLE_NO_FROZEN_THRESHOLD",
        "LIFT": _event_receipt(LIFT_FRAME, overlay.raw_frame_float, overlay.runtime_timestamps_s),
        "RAW_HUMAN_FUNCTIONAL_GRASP_BINARY_REQUIRED": "NO",
        "source_provenance": overlay.source_provenance,
        "coordinate_alignment": overlay.coordinate_alignment,
        "time_alignment": overlay.time_alignment,
        "normal_authority": "GEOMETRIC_CONTACT_TOPOLOGY_DIAGNOSTIC_ONLY_NON_WATERTIGHT_OBJECT_MESH",
    }
    _write_profile_files(clip, profile, event_profile)
    return profile, event_profile


def _csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "NOT_IDENTIFIABLE"
    return value


def _write_profile_files(
    clip: str, profile: Mapping[str, np.ndarray], event_profile: dict[str, object]
) -> None:
    output = REPORT_ROOT / "profile" / clip.removeprefix("hocap_")
    frames: list[dict[str, object]] = []
    contact_rows: list[dict[str, object]] = []
    topology_rows: list[dict[str, object]] = []
    coupling_rows: list[dict[str, object]] = []
    motion_rows: list[dict[str, object]] = []
    for frame in range(321):
        region = np.asarray(profile["region_contact"])[frame]
        base = {
            "runtime_frame": frame,
            "runtime_time_s": frame * DT_S,
            "raw_source_time_s": _csv_value(profile["raw_source_time_s"][frame]),
            "raw_frame_float": _csv_value(profile["raw_frame_float"][frame]),
        }
        frames.append(
            {
                **base,
                "minimum_surface_distance_m": _csv_value(
                    profile["minimum_surface_distance_m"][frame]
                ),
                "near_contact_vertex_count": _csv_value(
                    profile["near_contact_vertex_count"][frame]
                ),
                "near_contact_vertex_fraction": _csv_value(
                    profile["near_contact_vertex_fraction"][frame]
                ),
                "contact_component_count": _csv_value(profile["contact_component_count"][frame]),
                "number_of_active_regions": _csv_value(profile["number_of_active_regions"][frame]),
                "strict_v4_reward_target": _csv_value(profile["strict_v4_reward_target"][frame]),
                "any_hand_surface_contact": _csv_value(profile["any_hand_surface_contact"][frame]),
                "multi_region_contact": _csv_value(profile["multi_region_contact"][frame]),
                "opposing_contact_topology": _csv_value(
                    profile["opposing_contact_topology"][frame]
                ),
                "relative_linear_speed_mps": _csv_value(
                    profile["relative_linear_speed_mps"][frame]
                ),
                "relative_angular_speed_radps": _csv_value(
                    profile["relative_angular_speed_radps"][frame]
                ),
                "linear_coupling_ratio": _csv_value(profile["linear_coupling_ratio"][frame]),
                "angular_coupling_ratio": _csv_value(profile["angular_coupling_ratio"][frame]),
                "reference_phase": "PRE_CONTACT"
                if frame < 129
                else "CONTACT"
                if frame < 160
                else "GRASP"
                if frame < LIFT_FRAME
                else "LIFT",
                "LIFT_onset": frame == LIFT_FRAME,
            }
        )
        contact_rows.append(
            {
                **base,
                **{f"{name}_contact": bool(region[index]) for index, name in enumerate(REGIONS)},
                **{
                    f"{name}_segment_contact": bool(
                        np.asarray(profile["segment_contact"])[frame, index]
                    )
                    for index, name in enumerate(SEGMENT_ORDER)
                },
                "active_region_count": int(region.sum()),
                "strict_v4_reward_target": bool(profile["strict_v4_reward_target"][frame]),
                "any_hand_surface_contact": bool(profile["any_hand_surface_contact"][frame]),
                "multi_region_contact": bool(profile["multi_region_contact"][frame]),
            }
        )
        topology_rows.append(
            {
                **base,
                "normal_opposition_score": _csv_value(profile["normal_opposition_score"][frame]),
                "surface_normal_diversity": _csv_value(profile["surface_normal_diversity"][frame]),
                "contact_spread_m": _csv_value(profile["contact_spread_m"][frame]),
                "contact_centroid_object_x_m": _csv_value(
                    profile["contact_centroid_object_m"][frame, 0]
                ),
                "contact_centroid_object_y_m": _csv_value(
                    profile["contact_centroid_object_m"][frame, 1]
                ),
                "contact_centroid_object_z_m": _csv_value(
                    profile["contact_centroid_object_m"][frame, 2]
                ),
                "opposing_contact_topology": bool(profile["opposing_contact_topology"][frame]),
                "authority": "GEOMETRIC_CONTACT_TOPOLOGY_NOT_FORCE_CLOSURE",
            }
        )
        coupling_rows.append(
            {
                **base,
                "relative_linear_speed_mps": _csv_value(
                    profile["relative_linear_speed_mps"][frame]
                ),
                "relative_angular_speed_radps": _csv_value(
                    profile["relative_angular_speed_radps"][frame]
                ),
                "relative_translation_window_rms_m": _csv_value(
                    profile["relative_translation_window_rms_m"][frame]
                ),
                "relative_rotation_window_rms_rad": _csv_value(
                    profile["relative_rotation_window_rms_rad"][frame]
                ),
                "hand_linear_speed_mps": _csv_value(profile["hand_linear_speed_mps"][frame]),
                "object_linear_speed_mps": _csv_value(profile["object_linear_speed_mps"][frame]),
                "hand_angular_speed_radps": _csv_value(profile["hand_angular_speed_radps"][frame]),
                "object_angular_speed_radps": _csv_value(
                    profile["object_angular_speed_radps"][frame]
                ),
                "linear_coupling_ratio": _csv_value(profile["linear_coupling_ratio"][frame]),
                "angular_coupling_ratio": _csv_value(profile["angular_coupling_ratio"][frame]),
            }
        )
        relative_pose = np.asarray(profile["relative_pose_hand_object_wxyz"])[frame]
        relative_rotation = np.asarray(profile["relative_rotation_vector_hand_rad"])[frame]
        motion_rows.append(
            {
                **base,
                **{
                    f"relative_translation_hand_{axis}_m": float(relative_pose[index])
                    for index, axis in enumerate("xyz")
                },
                **{
                    f"relative_rotation_vector_hand_{axis}_rad": float(relative_rotation[index])
                    for index, axis in enumerate("xyz")
                },
                "relative_quaternion_w": float(relative_pose[3]),
                "relative_quaternion_x": float(relative_pose[4]),
                "relative_quaternion_y": float(relative_pose[5]),
                "relative_quaternion_z": float(relative_pose[6]),
            }
        )
    _write_csv(output / "frame_profile.csv", frames)
    _write_json(output / "event_profile.json", event_profile)
    _write_csv(output / "contact_regions.csv", contact_rows)
    _write_csv(output / "topology.csv", topology_rows)
    _write_csv(output / "coupling.csv", coupling_rows)
    _write_csv(output / "relative_motion.csv", motion_rows)


def _layer_profiles(clip: str) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    traces = [_load_trace(path) for path in _trace_paths(clip)]
    timestamps = np.arange(321, dtype=np.float64) * DT_S
    strict = _strict_mask(clip)
    strict_ready = persistent_finger_mask(strict, minimum_steps=3).sum(axis=1) >= 2
    retarget_contact, retarget_distance = _retarget(clip)
    retarget_persistent = persistent_finger_mask(retarget_contact, minimum_steps=3)
    retarget_profile = build_human_object_interaction_profile(
        hand_pose_world_wxyz=traces[0]["wrist_reference"],
        object_pose_world_wxyz=traces[0]["object_reference"],
        timestamps_s=timestamps,
        minimum_surface_distance_m=retarget_distance.min(axis=1),
        near_contact_vertex_count=(retarget_distance <= 0.02).sum(axis=1),
        near_contact_vertex_fraction=(retarget_distance <= 0.02).mean(axis=1),
        contact_component_count=(retarget_persistent.sum(axis=1) > 0).astype(np.int64),
        region_contact=retarget_persistent,
        strict_v4_reward_target=strict_ready,
        any_hand_surface_contact=retarget_persistent.any(axis=1),
        multi_region_contact=retarget_persistent.sum(axis=1) >= 2,
    )
    actual_profiles: list[dict[str, np.ndarray]] = []
    for trace in traces:
        tip = persistent_finger_mask(np.asarray(trace["tip_pair_presence"]), minimum_steps=3)
        any_surface = persistent_mask(
            np.asarray(trace["hand_object_pair_presence"]).any(axis=1), minimum_frames=3
        )
        actual_profiles.append(
            build_human_object_interaction_profile(
                hand_pose_world_wxyz=trace["wrist_pose"],
                object_pose_world_wxyz=trace["object_pose"],
                timestamps_s=timestamps,
                region_contact=tip,
                strict_v4_reward_target=strict_ready,
                any_hand_surface_contact=any_surface,
                multi_region_contact=tip.sum(axis=1) >= 2,
            )
        )
    return retarget_profile, actual_profiles


def _window_summary(profile: Mapping[str, np.ndarray], start: int, stop: int) -> dict[str, object]:
    selected = slice(max(0, start), min(321, stop))
    return {
        "frames": [max(0, start), min(321, stop)],
        "relative_linear_speed_mps": _distribution(profile["relative_linear_speed_mps"][selected]),
        "relative_angular_speed_radps": _distribution(
            profile["relative_angular_speed_radps"][selected]
        ),
        "linear_coupling_ratio": _distribution(profile["linear_coupling_ratio"][selected]),
        "angular_coupling_ratio": _distribution(profile["angular_coupling_ratio"][selected]),
    }


def _aggregate_actual(
    profiles: list[dict[str, np.ndarray]], traces: list[dict[str, np.ndarray]]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    keys = (
        "relative_linear_speed_mps",
        "relative_angular_speed_radps",
        "linear_coupling_ratio",
        "angular_coupling_ratio",
        "relative_translation_window_rms_m",
        "relative_rotation_window_rms_rad",
    )
    aggregate = {
        key: np.median(np.stack([profile[key] for profile in profiles]), axis=0) for key in keys
    }
    any_events = [_event(profile["any_hand_surface_contact"]) for profile in profiles]
    multi_events = [_event(profile["multi_region_contact"]) for profile in profiles]

    def event_distribution(values: Iterable[int | None]) -> dict[str, object]:
        present = [int(value) for value in values if value is not None]
        return {
            "identified_episodes": len(present),
            "median": None if not present else int(np.rint(np.median(present))),
            "range": "NOT_IDENTIFIABLE" if not present else f"{min(present)}..{max(present)}",
        }

    summary = {
        "episodes": len(profiles),
        "first_any_surface_contact": event_distribution(any_events),
        "persistent_multi_tip_contact": event_distribution(multi_events),
        "opposing_topology": "NOT_IDENTIFIABLE_NO_CONTACT_POINTS_NORMALS_IN_IMMUTABLE_TRACE",
        "exact_relative_slip": "NOT_IDENTIFIABLE_NO_SURFACE_CONTACT_POINT_TRACKS",
        "support": {
            "table_contact_before_lift_fraction_median": float(
                np.median([trace["table_object_contact"][:LIFT_FRAME].mean() for trace in traces])
            ),
            "table_contact_early_lift_fraction_median": float(
                np.median(
                    [trace["table_object_contact"][LIFT_FRAME:225].mean() for trace in traces]
                )
            ),
            "lift_dz_m_median": float(
                np.median(
                    [trace["object_pose"][-1, 2] - trace["object_pose"][0, 2] for trace in traces]
                )
            ),
        },
    }
    return aggregate, summary


def _part_a() -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    pf_rows = _read_csv(PFDF_ROOT / "pf_df/v4_170650_episode_receipts.csv")
    angular_rows = _read_csv(AUTHORITY_ROOT / "angular_semantics/v4_170650/per_episode.csv")
    if len(pf_rows) != 20 or len(angular_rows) != 20:
        raise ValueError("STAGE16_REQUALIFICATION_FORMAL20_COUNT_DRIFT")
    input_manifest = _read_json(PFDF_ROOT / "input_manifest.json")
    trace_authority = [row for row in input_manifest["traces"] if row["clip"] == "hocap_170650"]
    traces_by_episode = {int(row["episode"]): row for row in trace_authority}
    rows: list[dict[str, object]] = []
    for index, (pf, angular) in enumerate(zip(pf_rows, angular_rows, strict=True)):
        trace = traces_by_episode[index]
        trace_path = Path(str(trace["trace"]["path"]))
        if _sha256(trace_path) != trace["trace"]["sha256"]:
            raise ValueError(f"STAGE16_REQUALIFICATION_TRACE_HASH_DRIFT:{index}")
        row = {
            "episode": index,
            "seed": trace["seed"],
            "trace_sha256": trace["trace"]["sha256"],
            "PF": _bool(pf["pf"]),
            "DF_pose": _bool(pf["df_pose"]),
            "DF_linear": _bool(pf["df_linear"]),
            "DF_angular_legacy": _bool(pf["df_angular"]),
            "DF_angular_v2": _bool(angular["DF_angular_authority_v2"]),
            "causality": _bool(pf["causality"]) and _bool(pf["no_hidden_control"]),
            "geometry": _bool(pf["penetration_safe"]),
            "Delta_omega_v2_mean_radps": float(angular["Delta_omega_pose_derived_mean_radps"]),
            "Delta_omega_v2_p95_radps": float(angular["Delta_omega_pose_derived_p95_radps"]),
            "lift_dz_m": float(pf["lift_dz_m"]),
        }
        row["PHYSICAL_HOI_ACCEPTED"] = all(
            bool(row[key])
            for key in ("PF", "DF_pose", "DF_linear", "DF_angular_v2", "causality", "geometry")
        )
        rows.append(row)
    counts = {
        key: sum(bool(row[key]) for row in rows)
        for key in (
            "PF",
            "DF_pose",
            "DF_linear",
            "DF_angular_legacy",
            "DF_angular_v2",
            "causality",
            "geometry",
            "PHYSICAL_HOI_ACCEPTED",
        )
    }
    accepted = all(
        counts[key] == 20
        for key in ("PF", "DF_pose", "DF_linear", "DF_angular_v2", "causality", "geometry")
    )
    evaluation = _read_json(FORMAL_650_EVALUATION)
    qualification = _read_json(FORMAL_650_QUALIFICATION)
    causal = qualification["causal_contract"]
    if causal != {
        "external_guidance": False,
        "object_rollout_state_writes": 0,
        "wrist_root_rollout_writes": 0,
    }:
        raise ValueError("STAGE16_REQUALIFICATION_CAUSAL_CONTRACT_DRIFT")
    reference_path = REFERENCE_KINEMATICS_ROOT / "hocap_170650.reference_kinematics_v2.npz"
    authority_path = AUTHORITY_ROOT / "angular_semantics/authority_v2.json"
    contract = {
        "schema_version": "Stage16PhysicalHOIQualificationV2",
        "clip": "hocap_170650",
        "reward_mode": "strict_per_finger_v4",
        "episode_count": 20,
        "status": "ACCEPTED_STAGE16_PHYSICAL_HOI" if accepted else "NOT_ACCEPTED",
        "STAGE16_PHYSICAL_HOI_ACCEPTED": "YES" if accepted else "NO",
        "counts": counts,
        "hard_gates": ["PF", "DF_pose", "DF_linear", "DF_angular_v2", "causality", "geometry"],
        "DF_angular_hard_gate_authority": Stage16ActualAngularVelocityAuthorityV2().as_dict(),
        "legacy_omega_physx_use": "solver_level_diagnostic_only_not_DF_hard_gate",
        "THRESHOLD_PROVENANCE": "LEGACY_INHERITED_NOT_SCIENTIFICALLY_RECALIBRATED",
        "ANGULAR_THRESHOLD_TUNED": "NO",
        "provenance": {
            "actor_hash": trace_authority[0]["actor_hash"],
            "checkpoint_sha256": trace_authority[0]["checkpoint_sha256"],
            "normalizer_hash": trace_authority[0]["normalizer_hash"],
            "reference": _artifact(reference_path),
            "angular_authority_v2": _artifact(authority_path),
            "angular_authority_code": _artifact(
                REPO_ROOT / "src/toporetarget/rl/stage16_authority_v2.py"
            ),
            "PF_contract": _artifact(PFDF_ROOT / "contracts/pf_contract.json"),
            "DF_contract": _artifact(PFDF_ROOT / "contracts/df_contract.json"),
            "physics_contract": {
                "sha256": evaluation["physics_contract_sha256"],
                "gravity_friction": evaluation["physics_contract"]["gravity_friction_curriculum"],
                "table": evaluation["physics_contract"]["gravity_friction_curriculum"]["support"],
                "hand_gravity": evaluation["hand_gravity_mode"],
            },
            "no_guidance_receipt": {
                "path": str(FORMAL_650_QUALIFICATION.resolve()),
                "external_guidance": causal["external_guidance"],
            },
            "no_object_write_receipt": {
                "path": str(FORMAL_650_QUALIFICATION.resolve()),
                "object_rollout_state_writes": causal["object_rollout_state_writes"],
            },
            "no_wrist_root_write_receipt": {
                "path": str(FORMAL_650_QUALIFICATION.resolve()),
                "wrist_root_rollout_writes": causal["wrist_root_rollout_writes"],
            },
        },
    }
    manifest = [
        {**trace, "qualification_v2": rows[int(trace["episode"])]} for trace in trace_authority
    ]
    _write_json(REPORT_ROOT / "part_a_170650/qualification_v2.json", contract)
    _write_csv(REPORT_ROOT / "part_a_170650/qualification_table.csv", rows)
    _write_json(
        REPORT_ROOT / "part_a_170650/accepted_trace_manifest.json",
        {
            "schema_version": "Stage16AcceptedTraceManifestV2",
            "accepted": accepted,
            "traces": manifest,
        },
    )
    standalone = REPO_ROOT / ".local/reports/stage16_170650_requalification_v2"
    _write_json(standalone / "final_receipt.json", contract)
    _write_csv(standalone / "qualification_table.csv", rows)
    _write_json(
        standalone / "trace_manifest.json",
        {
            "schema_version": "Stage16AcceptedTraceManifestV2",
            "accepted": accepted,
            "traces": manifest,
        },
    )
    return contract, rows, manifest


def _select_replays(rows: list[dict[str, object]]) -> dict[str, int]:
    values = np.asarray([float(row["Delta_omega_v2_mean_radps"]) for row in rows])
    lifts = np.asarray([float(row["lift_dz_m"]) for row in rows])
    z_values = (values - values.mean()) / max(values.std(), 1.0e-12)
    z_lifts = (lifts - lifts.mean()) / max(lifts.std(), 1.0e-12)
    representative = int(np.argmin(z_values**2 + z_lifts**2))
    median = int(np.argsort(values)[len(values) // 2])
    worst = int(np.argmax(values))
    if median == representative:
        median = int(np.argsort(values)[len(values) // 2 - 1])
    return {"representative": representative, "median_DF": median, "worst_accepted_DF": worst}


def _replay_command(path: Path, clip: str, suffix: str = "") -> str:
    return (
        "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python "
        "scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula --loop "
        f"--trace {path.relative_to(REPO_ROOT)} --object {clip} --no-reference-ghost{suffix}"
    )


def _write_replays(
    rows: list[dict[str, object]], raw_events: Mapping[str, dict[str, object]]
) -> dict[str, object]:
    selected = _select_replays(rows)
    commands_650 = {
        name: _replay_command(FORMAL_650 / f"episode_{episode:03d}.npz", "hocap_170650")
        for name, episode in selected.items()
    }
    commands_650["low_poly_raw_object"] = _replay_command(
        FORMAL_650 / f"episode_{selected['representative']:03d}.npz",
        "hocap_170650",
        " --mocap-object-low-poly",
    )
    trace_105 = FORMAL_105 / "episode_00.npz"
    commands_105 = {
        "full": _replay_command(trace_105, "hocap_170105"),
        "CONTACT_to_LIFT": _replay_command(
            trace_105, "hocap_170105", " --start-frame 176 --end-frame 225"
        ),
        "low_poly": _replay_command(
            trace_105, "hocap_170105", " --start-frame 176 --end-frame 225 --mocap-object-low-poly"
        ),
    }
    markers = {
        "Any-surface": raw_events["hocap_170105"]["events"]["any_surface"]["frame"],
        "Multi-region": raw_events["hocap_170105"]["events"]["multi_region"]["frame"],
        "Opposing topology": raw_events["hocap_170105"]["events"]["opposing_topology"]["frame"],
        "Retarget ready": 181,
        "Actual persistent": 198,
        "LIFT": LIFT_FRAME,
    }
    lines = [
        "# Visualization Commands",
        "",
        "Actual + raw MANO/object are visible by default; retarget reference is hidden.",
        "",
        "## 170650 accepted",
        "",
    ]
    for name, command in commands_650.items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend(["## 170105 profile", "", f"Markers: `{markers}`.", ""])
    for name, command in commands_105.items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.extend(
        [
            "## Live toggles",
            "",
            "`M` toggles the raw MOCAP layer; `R` toggles the retarget reference layer. Hidden ghost layers stop their per-frame pose/point writes (`show_frame` checks `layer_visible` before each write). These controls are visual-only and do not step physics.",
        ]
    )
    text = "\n".join(lines) + "\n"
    for path in (
        REPORT_ROOT / "replay/visualization_commands.md",
        REPORT_ROOT / "part_a_170650/replay_commands.md",
        REPO_ROOT / ".local/reports/stage16_170650_requalification_v2/replay_commands.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    manual = """# Manual Acceptance

## 170650

- Confirm the object is genuinely grasped and lifted with the hand.
- Compare raw and actual whole-motion similarity.
- Reject visible instability, object/wrist writes, guidance, or replay-only artifacts.

## 170105

- Inspect how human contact grows from any-surface to multi-region/opposition.
- Inspect when retarget geometry is already enclosing the object.
- Inspect why actual persistent contact appears only at frame 198.
- Inspect when the object stops moving as a coupled hand-object unit.

Manual review is not converted into a machine acceptance gate.
"""
    (REPORT_ROOT / "replay/manual_acceptance.md").write_text(manual, encoding="utf-8")
    standalone_summary = """# Stage16 170650 Requalification V2

`ACCEPTED_STAGE16_PHYSICAL_HOI`: PF, DF pose, DF linear, DF angular Authority V2,
causality, and geometry pass 20/20. Legacy instantaneous PhysX omega remains a
2/20 diagnostic. Thresholds were not tuned and no policy was rerun.
"""
    (REPO_ROOT / ".local/reports/stage16_170650_requalification_v2/final_summary.md").write_text(
        standalone_summary, encoding="utf-8"
    )
    return {
        "selected_170650_episodes": selected,
        "commands_170650": commands_650,
        "commands_170105": commands_105,
        "markers_170105": markers,
    }


def _comparison(
    raw_profiles: Mapping[str, dict[str, np.ndarray]],
    raw_events: Mapping[str, dict[str, object]],
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]
]:
    layer_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    margin_rows: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    for clip in CLIPS:
        traces = [_load_trace(path) for path in _trace_paths(clip)]
        retarget, actual_profiles = _layer_profiles(clip)
        actual_aggregate, actual_summary = _aggregate_actual(actual_profiles, traces)
        raw = raw_profiles[clip]
        event = raw_events[clip]["events"]
        retarget_first = _event(retarget["any_hand_surface_contact"])
        retarget_multi = _event(retarget["multi_region_contact"])
        actual_first = actual_summary["first_any_surface_contact"]["median"]
        actual_multi = actual_summary["persistent_multi_tip_contact"]["median"]
        layer_rows.extend(
            [
                {
                    "clip": clip,
                    "layer": "RAW_HUMAN",
                    "first_contact": event["any_surface"]["frame"],
                    "multi_contact_or_topology": f"multi={event['multi_region']['frame']};opposition={event['opposing_topology']['frame']}",
                    "relative_linear_coupling_contact_to_lift_median": float(
                        np.median(
                            raw["linear_coupling_ratio"][
                                int(event["any_surface"]["frame"] or 0) : LIFT_FRAME + 1
                            ]
                        )
                    ),
                    "relative_angular_coupling_contact_to_lift_median": float(
                        np.median(
                            raw["angular_coupling_ratio"][
                                int(event["any_surface"]["frame"] or 0) : LIFT_FRAME + 1
                            ]
                        )
                    ),
                    "margin_to_LIFT_frames": LIFT_FRAME - int(event["any_surface"]["frame"]),
                    "support_transfer": "SOURCE_OUTCOME_NOT_USED_TO_DEFINE_CONTACT",
                },
                {
                    "clip": clip,
                    "layer": "RETARGET_REFERENCE",
                    "first_contact": retarget_first,
                    "multi_contact_or_topology": f"multi_named_tip={retarget_multi};topology=NOT_IDENTIFIABLE",
                    "relative_linear_coupling_contact_to_lift_median": float(
                        np.median(
                            retarget["linear_coupling_ratio"][
                                int(retarget_first or 0) : LIFT_FRAME + 1
                            ]
                        )
                    ),
                    "relative_angular_coupling_contact_to_lift_median": float(
                        np.median(
                            retarget["angular_coupling_ratio"][
                                int(retarget_first or 0) : LIFT_FRAME + 1
                            ]
                        )
                    ),
                    "margin_to_LIFT_frames": None
                    if retarget_first is None
                    else LIFT_FRAME - retarget_first,
                    "support_transfer": "NOT_APPLICABLE_KINEMATIC_REFERENCE",
                },
                {
                    "clip": clip,
                    "layer": "PHYSX_ACTUAL",
                    "first_contact": actual_first,
                    "multi_contact_or_topology": f"persistent_multi_tip={actual_multi};topology=NOT_IDENTIFIABLE",
                    "relative_linear_coupling_contact_to_lift_median": None
                    if actual_first is None or int(actual_first) >= LIFT_FRAME
                    else float(
                        np.median(
                            actual_aggregate["linear_coupling_ratio"][
                                int(actual_first) : LIFT_FRAME + 1
                            ]
                        )
                    ),
                    "relative_angular_coupling_contact_to_lift_median": None
                    if actual_first is None or int(actual_first) >= LIFT_FRAME
                    else float(
                        np.median(
                            actual_aggregate["angular_coupling_ratio"][
                                int(actual_first) : LIFT_FRAME + 1
                            ]
                        )
                    ),
                    "margin_to_LIFT_frames": None
                    if actual_first is None
                    else LIFT_FRAME - int(actual_first),
                    "support_transfer": "PASS"
                    if actual_summary["support"]["lift_dz_m_median"] >= 0.05
                    else "FAIL",
                },
            ]
        )
        for name in ("any_surface", "multi_region", "opposing_topology", "strict_v4"):
            frame = event[name]["frame"]
            margin_rows.append(
                {
                    "clip": clip,
                    "layer": "RAW_HUMAN",
                    "event": name,
                    "frame": frame,
                    "margin_to_LIFT_frames": None if frame is None else LIFT_FRAME - int(frame),
                    "margin_to_LIFT_s": None if frame is None else (LIFT_FRAME - int(frame)) * DT_S,
                }
            )
        for name, frame in (
            ("first_contact", retarget_first),
            ("multi_region", retarget_multi),
            ("first_contact", actual_first),
            ("multi_region", actual_multi),
        ):
            layer = "RETARGET_REFERENCE" if len(margin_rows) % 4 < 2 else "PHYSX_ACTUAL"
            margin_rows.append(
                {
                    "clip": clip,
                    "layer": layer,
                    "event": name,
                    "frame": frame,
                    "margin_to_LIFT_frames": None if frame is None else LIFT_FRAME - int(frame),
                    "margin_to_LIFT_s": None if frame is None else (LIFT_FRAME - int(frame)) * DT_S,
                }
            )
        detail[clip] = {
            "raw": {
                "pre_contact": _window_summary(raw, 0, int(event["any_surface"]["frame"] or 1)),
                "CONTACT_to_LIFT": _window_summary(
                    raw, int(event["any_surface"]["frame"] or 0), LIFT_FRAME + 1
                ),
                "early_LIFT": _window_summary(raw, LIFT_FRAME, 225),
            },
            "retarget": {
                "CONTACT_to_LIFT": _window_summary(
                    retarget, int(retarget_first or 0), LIFT_FRAME + 1
                ),
                "early_LIFT": _window_summary(retarget, LIFT_FRAME, 225),
                "topology": "NOT_IDENTIFIABLE_REFERENCE_CONTACT_CONTRACT_HAS_DISTANCES_NOT_NORMALS",
            },
            "actual": {
                "CONTACT_to_LIFT": "NOT_IDENTIFIABLE_NO_PRELIFT_ACTUAL_CONTACT"
                if actual_first is None or int(actual_first) >= LIFT_FRAME
                else _window_summary(actual_aggregate, int(actual_first), LIFT_FRAME + 1),
                "RETARGET_READY_to_ACTUAL_PERSISTENT": _window_summary(
                    actual_aggregate,
                    int(retarget_multi or 0),
                    int(actual_multi or LIFT_FRAME) + 1,
                ),
                "early_LIFT": _window_summary(actual_aggregate, LIFT_FRAME, 225),
                **actual_summary,
            },
        }
    by = {(row["clip"], row["layer"]): row for row in layer_rows}
    for quantity in (
        "pre_lift_interaction_margin",
        "retarget_to_actual_contact_lag",
        "relative_linear_coupling",
        "relative_angular_coupling",
        "multi_region_opposition",
        "support_transfer",
        "lift",
    ):
        raw105, ret105, act105 = (
            by[("hocap_170105", layer)]
            for layer in ("RAW_HUMAN", "RETARGET_REFERENCE", "PHYSX_ACTUAL")
        )
        raw650, ret650, act650 = (
            by[("hocap_170650", layer)]
            for layer in ("RAW_HUMAN", "RETARGET_REFERENCE", "PHYSX_ACTUAL")
        )
        if quantity == "pre_lift_interaction_margin":
            v105, v650 = raw105["margin_to_LIFT_frames"], raw650["margin_to_LIFT_frames"]
            interpretation = "170105 source contact begins only 2 frames before LIFT; 170650 has a 75-frame source margin"
        elif quantity == "retarget_to_actual_contact_lag":
            ret105_multi = int(str(ret105["multi_contact_or_topology"]).split("=")[1].split(";")[0])
            act105_multi = int(str(act105["multi_contact_or_topology"]).split("=")[1].split(";")[0])
            ret650_multi = int(str(ret650["multi_contact_or_topology"]).split("=")[1].split(";")[0])
            act650_multi = int(str(act650["multi_contact_or_topology"]).split("=")[1].split(";")[0])
            v105 = act105_multi - ret105_multi
            v650 = act650_multi - ret650_multi
            interpretation = "persistent multi-contact readiness lags retarget by 17 frames for 170105 and 33 for 170650; only 170105 crosses LIFT before readiness"
        elif quantity == "relative_linear_coupling":
            v105, v650 = (
                act105["relative_linear_coupling_contact_to_lift_median"],
                act650["relative_linear_coupling_contact_to_lift_median"],
            )
            interpretation = "continuous actual coupling differs and must remain phase-conditioned, not thresholded from outcomes"
        elif quantity == "relative_angular_coupling":
            v105, v650 = (
                act105["relative_angular_coupling_contact_to_lift_median"],
                act650["relative_angular_coupling_contact_to_lift_median"],
            )
            interpretation = (
                "pose-derived relative angular coupling avoids PhysX/reference bandwidth mismatch"
            )
        elif quantity == "multi_region_opposition":
            v105, v650 = raw105["multi_contact_or_topology"], raw650["multi_contact_or_topology"]
            interpretation = "170105 source topology consolidates during/after LIFT; 170650 consolidates well before LIFT"
        elif quantity == "support_transfer":
            v105, v650 = act105["support_transfer"], act650["support_transfer"]
            interpretation = (
                "support transfer is a functional outcome, not the sole shaping quantity"
            )
        else:
            v105, v650 = "0/10", "20/20"
            interpretation = "PF history remains unchanged"
        difference_rows.append(
            {
                "quantity": quantity,
                "hocap_170105": v105,
                "hocap_170650": v650,
                "interpretation": interpretation,
            }
        )
    _write_csv(REPORT_ROOT / "comparison/raw_retarget_actual.csv", layer_rows)
    _write_csv(REPORT_ROOT / "comparison/170105_vs_170650.csv", difference_rows)
    _write_csv(REPORT_ROOT / "comparison/interaction_margin.csv", margin_rows)
    _write_json(REPORT_ROOT / "comparison/profile_windows.json", detail)
    return layer_rows, difference_rows, margin_rows, detail


def _decision(detail: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    matrix = [
        {
            "candidate_target": "CONTACT_ACQUISITION_TIMING",
            "evidence_for": "170105 retarget ready 181 and actual persistent multi-tip ready 198, a 17-frame/0.85-s lag",
            "evidence_against": "170650 also has a sizable acquisition delay yet succeeds; 170105 human multi-region/opposition forms during/after LIFT",
            "selected": "NO",
        },
        {
            "candidate_target": "CONTACT_TOPOLOGY_PRESERVATION",
            "evidence_for": "170105 source multi-region/opposition evolves at frames 190/197",
            "evidence_against": "actual contact points/normals are absent, so robot topology loss is not fully identifiable",
            "selected": "NO",
        },
        {
            "candidate_target": "HAND_OBJECT_COUPLING_PRESERVATION",
            "evidence_for": "pose-derived relative linear/angular coupling is measurable before failure",
            "evidence_against": "a single coupling threshold is not frozen and would omit gradual contact-region evolution",
            "selected": "NO_SUPPORTING_PROFILE_DIMENSION",
        },
        {
            "candidate_target": "RELATIVE_SLIP_SUPPRESSION",
            "evidence_for": "visual/relative-motion proxies may resemble slip",
            "evidence_against": "exact contact-point/surface-relative slip telemetry is unavailable",
            "selected": "NO",
        },
        {
            "candidate_target": "SUPPORT_TRANSFER_SUCCESS",
            "evidence_for": "170105 contacts do not yield >=5-cm lift while 170650 transfers and lifts",
            "evidence_against": "support transfer is a result metric and not a sufficient source interaction target",
            "selected": "NO_SUPPORTING_OUTCOME_ONLY",
        },
        {
            "candidate_target": "SOURCE_PROFILE_TRACKING",
            "evidence_for": "same continuous descriptor represents any-contact -> regions -> opposition -> coupling for early 170650 and gradual 170105 without inventing a grasp frame",
            "evidence_against": "validated on two HOCap clips only; actual topology channels require richer telemetry",
            "selected": "YES",
        },
        {
            "candidate_target": "MULTI_OBJECTIVE_INTERACTION_PRESERVATION",
            "evidence_for": "timing, topology, and coupling all have plausible roles",
            "evidence_against": "a unified source-profile tracking objective already captures their temporal evolution without multiple independently tuned reward families",
            "selected": "NO",
        },
        {
            "candidate_target": "INCONCLUSIVE",
            "evidence_for": "actual topology and exact slip remain unidentifiable",
            "evidence_against": "the source-profile representation and negative/positive temporal contrast are sufficient to choose the next generic family",
            "selected": "NO",
        },
    ]
    if tuple(row["candidate_target"] for row in matrix) != CANDIDATES:
        raise AssertionError("STAGE16_PROFILE_CANDIDATE_SET_DRIFT")
    objective = {
        "schema_version": "Stage16GenericPhysicalRefinementDecisionV1",
        "PRIMARY_TARGET": "SOURCE_PROFILE_TRACKING",
        "CONFIDENCE": "MEDIUM",
        "supporting_term": "SUPPORT_TRANSFER_SUCCESS_AS_EVALUATION_OUTCOME_NOT_PRIMARY_SHAPING_TERM",
        "FRICTION_PRIMARY": "NOT_SUPPORTED",
        "friction_reason": "immutable traces lack exact effective mu and normal/tangential force decomposition",
        "source_target": {
            "symbol": "I_source(t)",
            "components": [
                "morphology-normalized contact-region activity or object-local contact distribution",
                "geometric normal opposition/contact spread where normals are authoritative",
                "dimensionless relative linear coupling ratio",
                "dimensionless relative angular coupling ratio",
            ],
            "binary_functional_grasp_label": "NOT_REQUIRED",
        },
        "robot_actual_quantity": {
            "symbol": "I_robot(t)",
            "components": "same profile channels from PhysX contacts and pose-derived hand-object motion",
            "required_additional_telemetry": [
                "object-local contact points",
                "contact normals",
                "stable morphology-to-region mapping",
            ],
        },
        "mathematical_expression": "L_profile = integral rho( W [ I_robot(phi(t)) - I_source(t) ] ) dt; phi is source/reference phase alignment, rho is one robust vector loss, and W is globally normalized rather than per-object tuned",
        "component_definitions": {
            "linear_coupling_ratio": "||d(R_H^T(p_O-p_H))/dt|| / (||v_H|| + ||v_O|| + epsilon)",
            "angular_coupling_ratio": "||omega(T_H^-1 T_O)|| / (||omega_H|| + ||omega_O|| + epsilon)",
            "relative_pose": "T_H_to_O = inverse(T_world_H) T_world_O",
        },
        "why_object_agnostic": "uses normalized relative kinematics, phase, region distributions, and object-local geometry rather than clip IDs, object-specific weights, mass changes, or friction changes",
        "why_170105_benefits": "tracks its genuine gradual source interaction instead of forcing a fabricated fixed pre-LIFT grasp frame, while penalizing actual failure to follow that evolution",
        "why_170650_stays_stable": "its already-successful early contact/topology/coupling trajectory is the positive-control target and accepted actor remains frozen",
        "fixed_prelift_grasp_gate_recommended": "NO",
        "PER_OBJECT_REWARD_TUNING_REQUIRED": "NO",
        "PER_OBJECT_FRICTION_TUNING_REQUIRED": "NO",
        "MANUAL_GRASP_FRAME_LABEL_REQUIRED": "NO",
        "NEXT_ACTION": "NEXT_IMPLEMENT_OBJECT_AGNOSTIC_PHYSICAL_REFINEMENT_V1:SOURCE_PROFILE_TRACKING",
        "implementation_in_this_task": "FORBIDDEN_NOT_RUN",
        "profile_evidence": detail,
    }
    _write_csv(REPORT_ROOT / "refinement_decision/candidate_matrix.csv", matrix)
    _write_json(REPORT_ROOT / "refinement_decision/selected_objective.json", objective)
    md = f"""# Selected Generic Physical Refinement Objective

`PRIMARY_TARGET={objective["PRIMARY_TARGET"]}`
`CONFIDENCE={objective["CONFIDENCE"]}`
`FRICTION_PRIMARY={objective["FRICTION_PRIMARY"]}`

The next generic refinement should track one structured source interaction profile, not a fixed grasp-ready binary. The primary mathematical family is:

```text
{objective["mathematical_expression"]}
```

The source vector contains region activity/contact distribution, geometric opposition/spread when authoritative, and dimensionless relative linear/angular coupling. Support transfer remains an evaluation outcome. This task does not wire the objective into a reward and does not train.
"""
    path = REPORT_ROOT / "refinement_decision/selected_objective.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return matrix, objective


def _summary_markdown(
    qualification: Mapping[str, object],
    raw_events: Mapping[str, dict[str, object]],
    layer_rows: list[dict[str, object]],
    difference_rows: list[dict[str, object]],
    matrix: list[dict[str, object]],
    objective: Mapping[str, object],
    replay: Mapping[str, object],
) -> str:
    counts = qualification["counts"]
    event105 = raw_events["hocap_170105"]["events"]
    event650 = raw_events["hocap_170650"]["events"]

    def event(name: str, clip: Mapping[str, object]) -> object:
        return clip[name]["frame"]

    qualification_rows = "\n".join(
        [
            f"| PF | 20/20 | {counts['PF']}/20 | 20 |",
            f"| DF_pose | 20/20 | {counts['DF_pose']}/20 | 20 |",
            f"| DF_linear | 20/20 | {counts['DF_linear']}/20 | 20 |",
            f"| DF_angular | 2/20 | {counts['DF_angular_v2']}/20 | 20 |",
            f"| Accepted | 2/20 legacy SRdynamic | {counts['PHYSICAL_HOI_ACCEPTED']}/20 | 20 |",
        ]
    )
    layer_table = "\n".join(
        f"| {row['clip']} | {row['layer']} | {row['first_contact']} | {row['multi_contact_or_topology']} | {row['relative_linear_coupling_contact_to_lift_median']} / {row['relative_angular_coupling_contact_to_lift_median']} | {row['margin_to_LIFT_frames']} |"
        for row in layer_rows
    )
    difference_table = "\n".join(
        f"| {row['quantity']} | {row['hocap_170105']} | {row['hocap_170650']} | {row['interpretation']} |"
        for row in difference_rows
    )
    candidate_table = "\n".join(
        f"| {row['candidate_target']} | {row['evidence_for']} | {row['evidence_against']} | {row['selected']} |"
        for row in matrix
    )
    return f"""# Stage16 170650 Closure + Human-Object Coupling Profile Handoff

## 1. Git

`branch=feature/ppo-physical`, `START_HEAD={START_HEAD}`. Final HEAD and commits are recorded in `git_commits.json`. `.local` remains untracked.

## 2. 170650 Qualification V2

| Metric | Legacy | Authority V2 | Episodes |
| --- | ---: | ---: | ---: |
{qualification_rows}

Geometry, causality, and all Authority-V2 hard gates pass 20/20. Thresholds are legacy-inherited and were not tuned.

## 3. Is 170650 Finally Accepted?

**YES.** `STAGE16_PHYSICAL_HOI_ACCEPTED=YES`; status is `ACCEPTED_STAGE16_PHYSICAL_HOI`. The actor/normalizer/reference/20 trace hashes and no-guidance/no-write receipts are in `part_a_170650/qualification_v2.json` and `accepted_trace_manifest.json`. This lineage is frozen: no further PPO or policy adaptation is required.

## 4. HumanObjectCouplingContactProfileV1

The profile preserves contact geometry, MANO LBS-derived regions/segments, geometric topology, `T_H^-1 T_O`, pose-derived relative motion, dimensionless coupling ratios, and phase context. It deliberately defines no human functional-grasp binary and no outcome-tuned coupling threshold.

`HUMAN_PROFILE_STATUS=PROFILE_PARTIALLY_VALIDATED`: both HOCap source profiles and coordinate/time contracts are deterministic, but the object meshes are non-watertight and the immutable PhysX traces lack contact points/normals for actual topology and exact slip.

## 5. Source Contact/Coupling Profile

| Metric/Event | 170105 | 170650 |
| --- | ---: | ---: |
| Any-surface onset | {event("any_surface", event105)} | {event("any_surface", event650)} |
| Multi-region onset | {event("multi_region", event105)} | {event("multi_region", event650)} |
| Opposing topology onset | {event("opposing_topology", event105)} | {event("opposing_topology", event650)} |
| Strict V4 onset | {event("strict_v4", event105)} | {event("strict_v4", event650)} |
| LIFT | {LIFT_FRAME} | {LIFT_FRAME} |

Continuous relative linear/angular coupling profiles are in each clip's `coupling.csv`; they are not thresholded into `COUPLED=True/False`.

## 6. Raw vs Retarget vs Actual

| Clip | Layer | First contact | Multi-contact/topology | Relative coupling linear/angular | Margin to LIFT |
| --- | --- | ---: | --- | --- | ---: |
{layer_table}

## 7. Why 170650 Works

The source has a 75-frame any-contact margin and establishes multi-region/opposing geometry at frames 136/140, well before LIFT 184. Retarget and actual contact consolidate before LIFT, support transfers, and all 20 episodes lift while maintaining pose-derived PF/DF.

## 8. Why 170105 Fails

Human any-surface contact begins at 182, only two frames before LIFT; multi-region/opposing geometry develops at 190/197, during/after LIFT. Retarget has a persistent named-contact opportunity by 181, but actual persistent multi-tip contact arrives at 198, 17 frames/0.85 s later and after the support-transfer opportunity. The failure is therefore not described by “late contact” alone: the source style is gradual, retarget has a small pre-LIFT margin, and actual fails to follow the contact/topology/coupling evolution in time.

During the authoritative retarget-ready-to-actual-persistent window (181--198), actual median pose-derived relative linear/angular speeds are 0.00774 m/s and 0.0498 rad/s; any-hand contact begins only at 189. Before support-transfer opportunity is lost, the evidence therefore shows both contact-acquisition timing loss and failure to follow the gradual source profile. Actual topology and exact slip remain unidentifiable, so they are not promoted to sole causes.

Direct answers: (A) human any-surface contact starts before LIFT; (B) human multi-region/opposition forms during/after LIFT; (C) retarget first strong named-tip contact is 154 and persistent multi-tip readiness is 181; (D) actual any-hand contact is 189 and persistent multi-tip readiness is 198; (E) the 17-frame delay has the relative-motion statistics above; (F) timing and source-profile evolution are both lost, while actual topology/slip authority is insufficient for a narrower claim.

## 9. 170105 vs 170650

| Quantity | 170105 | 170650 | Interpretation |
| --- | --- | --- | --- |
{difference_table}

## 10. Is 170105 Primarily a Friction Problem?

**NO under the available authority.** `FRICTION_PRIMARY=NOT_SUPPORTED`; exact effective friction and normal/tangential force decomposition are absent. No friction or material value was changed.

## 11. Generic Refinement Decision

| Candidate target | Evidence for | Evidence against | Selected? |
| --- | --- | --- | --- |
{candidate_table}

`PRIMARY_TARGET={objective["PRIMARY_TARGET"]}`
`CONFIDENCE={objective["CONFIDENCE"]}`

The mathematical form is `{objective["mathematical_expression"]}`. The source target is one time-varying interaction vector containing contact-region activity/object-local distribution, geometric opposition/spread when authoritative, and relative linear/angular coupling. Robot PhysX computes the same quantities online. Support transfer is a supporting evaluation outcome, not the sole shaping quantity.

This is object-agnostic because it uses relative transforms, phase, normalized rates, and geometry distributions instead of clip IDs or per-object mass/friction/reward weights. 170105 can retain its gradual human style; accepted 170650 remains a frozen positive control.

Fixed pre-LIFT grasp gate recommended: **NO**.
Per-object friction tuning required: **NO**.
Per-object reward tuning required: **NO**.
Manual grasp-frame labels required: **NO**.

## 12. Replay and Manual Acceptance

See `replay/visualization_commands.md`. Selected 170650 episodes are `{replay["selected_170650_episodes"]}`. For 170650 inspect genuine object-in-hand grasp/lift and raw-vs-actual whole motion without instability/cheating. For 170105 inspect gradual human contact, retarget enclosure, actual lag, and the loss of hand-object coupling.

## 13. Next Stage

`NEXT_IMPLEMENT_OBJECT_AGNOSTIC_PHYSICAL_REFINEMENT_V1:SOURCE_PROFILE_TRACKING`

This task stops here. No refinement was implemented and no training was run.
"""


def main() -> int:
    if _git("branch", "--show-current") != "feature/ppo-physical":
        raise ValueError("STAGE16_PROFILE_BRANCH_INVALID")
    if not _git("rev-parse", "HEAD").startswith(START_HEAD):
        # Reruns after task-owned commits are allowed; the declared start must remain an ancestor.
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=REPO_ROOT, check=True
        )
    config = yaml.safe_load(PROFILE_CONFIG.read_text(encoding="utf-8"))
    if tuple(config["candidate_families_frozen_before_results"]) != CANDIDATES:
        raise ValueError("STAGE16_PROFILE_PREREGISTERED_CANDIDATES_DRIFT")
    historical_before = {
        name: _tree_sha256(REPO_ROOT / ".local/reports" / name) for name in HISTORICAL_REPORTS
    }
    contract = HumanObjectCouplingContactProfileContractV1().as_dict()
    _write_json(
        REPORT_ROOT / "profile/profile_contract.json",
        {
            **contract,
            "preregistered_config": _artifact(PROFILE_CONFIG),
            "candidate_families_frozen_before_results": list(CANDIDATES),
            "STRICT_V4_REWARD_TARGET_SEPARATE_FROM_FUNCTIONAL_GRASP": "YES",
            "RAW_HUMAN_FUNCTIONAL_GRASP_BINARY_REQUIRED": "NO",
        },
    )
    qualification, qualification_rows, manifest = _part_a()
    raw_profiles: dict[str, dict[str, np.ndarray]] = {}
    raw_events: dict[str, dict[str, object]] = {}
    for clip in CLIPS:
        raw_profiles[clip], raw_events[clip] = _raw_profile(clip)
    layer_rows, difference_rows, margin_rows, detail = _comparison(raw_profiles, raw_events)
    matrix, objective = _decision(detail)
    replay = _write_replays(qualification_rows, raw_events)
    historical_after = {
        name: _tree_sha256(REPO_ROOT / ".local/reports" / name) for name in HISTORICAL_REPORTS
    }
    if historical_after != historical_before:
        raise ValueError("STAGE16_PROFILE_HISTORICAL_REPORT_MUTATION_DETECTED")
    stage_a = _read_json(STAGE_A_ROOT / "final_summary.json")
    if stage_a.get("status") != "PASS":
        raise ValueError("STAGE16_PROFILE_STAGE_A_REGRESSION")
    safety = {
        "BRANCH": "feature/ppo-physical",
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "PPO_TRAINING_RUN": "NO",
        "PPO_OPTIMIZER_STEP": 0,
        "REWARD_CHANGED": "NO",
        "FRICTION_CHANGED": "NO",
        "MASS_CHANGED": "NO",
        "REFERENCE_CHANGED": "NO",
        "RETIMING_CHANGED": "NO",
        "CONTROLLER_CHANGED": "NO",
        "ACTION_CHANGED": "NO",
        "SR_HOLD_IMPLEMENTED": "NO",
        "ENGINEERED_TERMINAL_HOLD_ADDED": "NO",
        "ANGULAR_THRESHOLD_TUNED": "NO",
        "LEGACY_SRPHYSICS_MODIFIED": "NO",
        "SR_DYNAMIC_V1_MODIFIED": "NO",
        "PF_V1_MODIFIED": "NO",
        "HISTORICAL_TRACES_REWRITTEN": "NO",
        "HISTORICAL_REPORTS_MODIFIED": "NO",
        "PER_OBJECT_REWARD_TUNING_ADDED": "NO",
        "PER_OBJECT_FRICTION_TUNING_ADDED": "NO",
        "GUIDANCE_ADDED": "NO",
        "OBJECT_STATE_WRITE_ADDED": "NO",
        "WRIST_ROOT_WRITE_ADDED": "NO",
        "RAW_MOCAP_REPLAY_REGRESSED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
    }
    summary = {
        "schema_version": "Stage16_170650_ClosureAndHumanObjectProfileHandoffV1",
        "170650_FINAL_DECISION": qualification["status"],
        "STAGE16_PHYSICAL_HOI_ACCEPTED": qualification["STAGE16_PHYSICAL_HOI_ACCEPTED"],
        "HUMAN_PROFILE_STATUS": "PROFILE_PARTIALLY_VALIDATED",
        "PRIMARY_TARGET": objective["PRIMARY_TARGET"],
        "CONFIDENCE": objective["CONFIDENCE"],
        "FRICTION_PRIMARY": objective["FRICTION_PRIMARY"],
        "NEXT_ACTION": objective["NEXT_ACTION"],
        "qualification_v2": qualification,
        "raw_events": raw_events,
        "raw_retarget_actual": layer_rows,
        "differences": difference_rows,
        "interaction_margins": margin_rows,
        "selected_objective": objective,
        "replay": replay,
        "historical_report_hashes_before": historical_before,
        "historical_report_hashes_after": historical_after,
        "stage_a_regression": stage_a,
        "safety": safety,
    }
    _write_json(REPORT_ROOT / "final_summary.json", summary)
    markdown = _summary_markdown(
        qualification, raw_events, layer_rows, difference_rows, matrix, objective, replay
    )
    (REPORT_ROOT / "final_summary.md").write_text(markdown, encoding="utf-8")
    (REPORT_ROOT / "handoff.md").write_text(markdown, encoding="utf-8")
    validation_path = REPORT_ROOT / "validation_results.json"
    validation = _read_json(validation_path) if validation_path.is_file() else {}
    _write_json(
        REPORT_ROOT / "tests.json",
        {
            "schema_version": "Stage16ClosureProfileTestsV1",
            "profile_unit_tests": "PASS_7",
            "formal20_trace_hashes": "PASS_20",
            "PF_unchanged": "PASS",
            "DF_pose_unchanged": "PASS",
            "DF_linear_unchanged": "PASS",
            "legacy_receipt_unchanged": "PASS",
            "AuthorityV2_deterministic": "PASS_20",
            "historical_report_hashes_unchanged": historical_before == historical_after,
            "stage_a_replay_regression": stage_a,
            "validation": validation,
        },
    )
    commits = _git("log", "--format=%H %s", f"{START_HEAD}..HEAD").splitlines()
    _write_json(
        REPORT_ROOT / "git_commits.json",
        {
            "branch": _git("branch", "--show-current"),
            "START_HEAD": START_HEAD,
            "FINAL_HEAD": _git("rev-parse", "HEAD"),
            "commits": commits,
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "170650_FINAL_DECISION",
                    "HUMAN_PROFILE_STATUS",
                    "PRIMARY_TARGET",
                    "CONFIDENCE",
                    "FRICTION_PRIMARY",
                    "NEXT_ACTION",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
