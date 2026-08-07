"""Validated read-only replay contract for Stage 16-D IsaacLab traces.

The loader accepts both the multi-replica calibration capture schema and the
nominal trace inside ``PhysicsConsistentRetargetedTrajectoryV1`` artifacts.
Both are normalized to one read-only replay contract; no simulator state is
created or modified here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

TRACE_REQUIRED_ARRAYS = (
    "object_pose",
    "hand_collision_body_pose",
    "hand_collision_body_names",
    "contact_force_world",
    "contact_pair_presence",
    "contact_group_presence",
    "contact_group_names",
    "object_twist",
    "mean_absolute_effort",
    "finite",
    "reason_code",
    "actions",
)

CORRECTED_TRACE_REQUIRED_ARRAYS = (
    "object_pose",
    "object_twist",
    "hand_collision_body_pose",
    "hand_collision_body_names",
    "contact_force_world",
    "contact_pair_presence",
    "actuator_effort",
    "reason_code",
    "action",
)

CONTACT_GROUP_NAMES = ("thumb", "index", "middle", "ring", "pinky", "palm")
HOCAP_REFERENCE_FIELDS = (
    "timestamps",
    "object_pose_translation_world_ref",
    "object_pose_quaternion_world_ref_wxyz",
    "object_twist_world_ref",
)


def infer_object_id(path: Path) -> str:
    """Infer a supported object id from an artifact filename."""

    matches = [name for name in ("hocap_170105", "hocap_170650") if name in path.name]
    if len(matches) != 1:
        raise ValueError("cannot infer object id from trace filename; pass --object explicitly")
    return matches[0]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _validate_pose_array(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    quaternion_norm = np.linalg.norm(value[..., 3:7], axis=-1)
    if np.any(quaternion_norm < 1.0e-8):
        raise ValueError(f"{name} contains a zero quaternion")


@dataclass(frozen=True)
class ReplayFrameDiagnostics:
    frame: int
    replica: int
    contact_body_count: int
    contact_groups: tuple[str, ...]
    contact_force_norm_n: float
    object_linear_speed_mps: float
    object_angular_speed_radps: float
    mean_absolute_effort: float
    finite: bool
    reason_code: int
    worst_penetration_m: float | None
    worst_pair_index: int | None
    inter_finger_penetration_m: float | None


@dataclass(frozen=True)
class Stage16DSimulationTraceReplay:
    """Arrays required to reproduce recorded collision-proxy poses in IsaacLab."""

    trace_path: Path
    trace_kind: str
    object_pose: np.ndarray
    hand_collision_body_pose: np.ndarray
    hand_collision_body_names: tuple[str, ...]
    contact_force_world: np.ndarray
    contact_pair_presence: np.ndarray
    contact_group_presence: np.ndarray
    contact_group_names: tuple[str, ...]
    object_twist: np.ndarray
    mean_absolute_effort: np.ndarray
    finite: np.ndarray
    reason_code: np.ndarray
    actions: np.ndarray
    qualification_status: str = "QUALIFICATION_NOT_SUPPLIED"
    qualification_path: Path | None = None
    qualification_metrics: dict[str, float | str] | None = None
    penetration_depth_m: np.ndarray | None = None
    frame_worst_penetration_m: np.ndarray | None = None
    frame_worst_pair_index: np.ndarray | None = None
    inter_finger_penetration_m: np.ndarray | None = None

    @property
    def frame_count(self) -> int:
        return int(self.object_pose.shape[0])

    @property
    def replica_count(self) -> int:
        return int(self.object_pose.shape[1])

    @property
    def body_count(self) -> int:
        return len(self.hand_collision_body_names)

    def validate_replica(self, replica: int) -> None:
        if not 0 <= replica < self.replica_count:
            raise ValueError(f"replica must be in [0, {self.replica_count - 1}], got {replica}")

    def frame_indices(self, start: int, end: int | None) -> range:
        final = self.frame_count if end is None else end
        if not 0 <= start < self.frame_count:
            raise ValueError(f"start frame must be in [0, {self.frame_count - 1}]")
        if not start < final <= self.frame_count:
            raise ValueError(f"end frame must be in [{start + 1}, {self.frame_count}]")
        return range(start, final)

    def diagnostics(self, frame: int, replica: int) -> ReplayFrameDiagnostics:
        self.validate_replica(replica)
        if not 0 <= frame < self.frame_count:
            raise ValueError(f"frame must be in [0, {self.frame_count - 1}]")
        contact_mask = self.contact_pair_presence[frame, replica]
        group_mask = self.contact_group_presence[frame, replica]
        twist = self.object_twist[frame, replica]
        worst_penetration = None
        worst_pair = None
        if self.frame_worst_penetration_m is not None:
            worst_penetration = float(self.frame_worst_penetration_m[frame, replica])
        if self.frame_worst_pair_index is not None:
            worst_pair = int(self.frame_worst_pair_index[frame, replica])
        inter_finger = None
        if self.inter_finger_penetration_m is not None:
            inter_finger = float(self.inter_finger_penetration_m[frame, replica])
        return ReplayFrameDiagnostics(
            frame=frame,
            replica=replica,
            contact_body_count=int(np.count_nonzero(contact_mask)),
            contact_groups=tuple(
                name
                for name, present in zip(self.contact_group_names, group_mask, strict=True)
                if present
            ),
            contact_force_norm_n=float(
                np.asarray(np.linalg.norm(self.contact_force_world[frame, replica], axis=-1)).sum()
            ),
            object_linear_speed_mps=float(np.linalg.norm(twist[:3])),
            object_angular_speed_radps=float(np.linalg.norm(twist[3:])),
            mean_absolute_effort=float(self.mean_absolute_effort[frame, replica]),
            finite=bool(self.finite[frame, replica]),
            reason_code=int(self.reason_code[frame, replica]),
            worst_penetration_m=worst_penetration,
            worst_pair_index=worst_pair,
            inter_finger_penetration_m=inter_finger,
        )

    def camera_bounds(self, replica: int, frames: Sequence[int]) -> tuple[np.ndarray, float]:
        self.validate_replica(replica)
        indices = np.asarray(list(frames), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("camera frame selection is empty")
        positions = np.concatenate(
            (
                self.object_pose[indices, replica, :3].reshape(-1, 3),
                self.hand_collision_body_pose[indices, replica, :, :3].reshape(-1, 3),
            ),
            axis=0,
        )
        lower = positions.min(axis=0)
        upper = positions.max(axis=0)
        center = (lower + upper) * 0.5
        radius = max(float(np.linalg.norm(upper - lower)), 0.12)
        return center, radius


def _contact_group_presence(
    contact_pair_presence: np.ndarray, body_names: Sequence[str]
) -> np.ndarray:
    """Derive body-group presence without inventing contact points or forces."""

    from toporetarget.rl.physics_retargeting.contact_topology import body_contact_group

    pair_presence = np.asarray(contact_pair_presence, dtype=bool)
    result = np.zeros((*pair_presence.shape[:-1], len(CONTACT_GROUP_NAMES)), dtype=bool)
    group_index = {name: index for index, name in enumerate(CONTACT_GROUP_NAMES)}
    for body_index, body_name in enumerate(body_names):
        group = body_contact_group(str(body_name))
        if group is not None:
            result[..., group_index[group]] |= pair_presence[..., body_index]
    return result


def _qualification_path_for_trace(trace_path: Path) -> Path | None:
    name = trace_path.name
    if not name.startswith("trajectory_trace_"):
        return None
    candidate = trace_path.with_name(
        name.replace("trajectory_trace_", "trajectory_qualification_", 1)
    ).with_suffix(".json")
    return candidate if candidate.is_file() else None


def _load_qualification(
    trace_path: Path, qualification_path: Path | None
) -> tuple[str, Path | None, dict[str, float | str] | None]:
    selected = qualification_path or _qualification_path_for_trace(trace_path)
    if selected is None:
        return "QUALIFICATION_NOT_SUPPLIED", None, None
    if not selected.is_file():
        raise FileNotFoundError(selected)
    payload = json.loads(selected.read_text(encoding="utf-8"))
    status = str(payload.get("status", "QUALIFICATION_STATUS_MISSING"))
    metrics: dict[str, float | str] = {}
    for name in (
        "success_rate",
        "semantic_reach_rate",
        "contact_topology_pass_rate",
        "complete_trajectory_rate",
        "empirical_classification",
        "formal_classification",
    ):
        if name in payload:
            value = payload[name]
            metrics[name] = float(value) if isinstance(value, (int, float)) else str(value)
    return status, selected, metrics


def _load_corrected_trace(
    trace_path: Path,
    arrays: dict[str, np.ndarray],
    *,
    expected_body_names: Sequence[str] | None,
    qualification_path: Path | None,
) -> Stage16DSimulationTraceReplay:
    missing = sorted(set(CORRECTED_TRACE_REQUIRED_ARRAYS) - arrays.keys())
    if missing:
        raise ValueError(f"corrected trace is missing required arrays: {missing}")

    object_pose = arrays["object_pose"]
    hand_pose = arrays["hand_collision_body_pose"]
    if object_pose.ndim != 2 or object_pose.shape[-1] != 7:
        raise ValueError("corrected object_pose must be [frames, 7]")
    frames = object_pose.shape[0]
    if hand_pose.ndim != 3 or hand_pose.shape[0] != frames or hand_pose.shape[-1] != 7:
        raise ValueError("corrected hand_collision_body_pose must be [frames, bodies, 7]")
    bodies = hand_pose.shape[1]
    body_names = tuple(str(value) for value in arrays["hand_collision_body_names"].tolist())
    if len(body_names) != bodies or len(set(body_names)) != bodies:
        raise ValueError("corrected hand collision body names must be unique and match poses")
    if expected_body_names is not None and body_names != tuple(expected_body_names):
        raise ValueError("trace body order does not match the runtime geometry manifest")

    _validate_pose_array("object_pose", object_pose, (frames, 7))
    _validate_pose_array("hand_collision_body_pose", hand_pose, (frames, bodies, 7))
    expected_shapes = {
        "object_twist": (frames, 6),
        "contact_force_world": (frames, 3),
        "contact_pair_presence": (frames, bodies),
        "actuator_effort": (frames, 26),
        "reason_code": (frames,),
        "action": (frames, 26),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {arrays[name].shape}")
    for name in ("object_twist", "contact_force_world", "actuator_effort", "action"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")

    pair_presence = arrays["contact_pair_presence"].astype(bool, copy=False)
    group_presence = _contact_group_presence(pair_presence, body_names)
    finite = np.logical_and.reduce(
        (
            np.isfinite(object_pose).reshape(frames, -1).all(axis=1),
            np.isfinite(hand_pose).reshape(frames, -1).all(axis=1),
            np.isfinite(arrays["object_twist"]).all(axis=1),
            np.isfinite(arrays["action"]).all(axis=1),
        )
    )
    status, selected_qualification, metrics = _load_qualification(trace_path, qualification_path)
    inter_finger = arrays.get("inter_finger_penetration_m")
    if inter_finger is not None:
        if inter_finger.shape != (frames,) or not np.isfinite(inter_finger).all():
            raise ValueError("inter_finger_penetration_m must be finite [frames]")
    return Stage16DSimulationTraceReplay(
        trace_path=trace_path,
        trace_kind="physics_consistent_corrected_nominal",
        object_pose=object_pose[:, None, :],
        hand_collision_body_pose=hand_pose[:, None, :, :],
        hand_collision_body_names=body_names,
        contact_force_world=arrays["contact_force_world"][:, None, :],
        contact_pair_presence=pair_presence[:, None, :],
        contact_group_presence=group_presence[:, None, :],
        contact_group_names=CONTACT_GROUP_NAMES,
        object_twist=arrays["object_twist"][:, None, :],
        mean_absolute_effort=np.mean(np.abs(arrays["actuator_effort"]), axis=1)[:, None],
        finite=finite[:, None],
        reason_code=arrays["reason_code"][:, None],
        actions=arrays["action"],
        qualification_status=status,
        qualification_path=selected_qualification,
        qualification_metrics=metrics,
        inter_finger_penetration_m=(inter_finger[:, None] if inter_finger is not None else None),
    )


def load_stage16d_simulation_trace(
    trace_path: Path,
    *,
    geometry_path: Path | None = None,
    expected_body_names: Sequence[str] | None = None,
    qualification_path: Path | None = None,
) -> Stage16DSimulationTraceReplay:
    """Load and fail closed on an incompatible simulation replay artifact."""

    arrays = _load_npz(trace_path)
    if arrays.get("object_pose", np.empty(0)).ndim == 2:
        trace = _load_corrected_trace(
            trace_path,
            arrays,
            expected_body_names=expected_body_names,
            qualification_path=qualification_path,
        )
        if geometry_path is not None:
            geometry = _load_npz(geometry_path)
            required_geometry = {
                "penetration_depth_m",
                "frame_worst_penetration_m",
                "frame_worst_pair_index",
            }
            missing_geometry = sorted(required_geometry - geometry.keys())
            if missing_geometry:
                raise ValueError(f"geometry trace is missing required arrays: {missing_geometry}")
            corrected_geometry_shapes = {
                "penetration_depth_m": (trace.frame_count, trace.body_count),
                "frame_worst_penetration_m": (trace.frame_count,),
                "frame_worst_pair_index": (trace.frame_count,),
            }
            for name, shape in corrected_geometry_shapes.items():
                if geometry[name].shape != shape:
                    raise ValueError(f"{name} must have shape {shape}, got {geometry[name].shape}")
            trace = replace(
                trace,
                penetration_depth_m=geometry["penetration_depth_m"][:, None, :],
                frame_worst_penetration_m=geometry["frame_worst_penetration_m"][:, None],
                frame_worst_pair_index=geometry["frame_worst_pair_index"][:, None],
            )
        return trace

    missing = sorted(set(TRACE_REQUIRED_ARRAYS) - arrays.keys())
    if missing:
        raise ValueError(f"trace is missing required arrays: {missing}")

    object_pose = arrays["object_pose"]
    hand_pose = arrays["hand_collision_body_pose"]
    if object_pose.ndim != 3 or object_pose.shape[-1] != 7:
        raise ValueError("object_pose must be [frames, replicas, 7]")
    frames, replicas = object_pose.shape[:2]
    if hand_pose.ndim != 4 or hand_pose.shape[:2] != (frames, replicas) or hand_pose.shape[-1] != 7:
        raise ValueError("hand_collision_body_pose must be [frames, replicas, bodies, 7]")
    bodies = hand_pose.shape[2]
    body_names = tuple(str(value) for value in arrays["hand_collision_body_names"].tolist())
    group_names = tuple(str(value) for value in arrays["contact_group_names"].tolist())
    if len(body_names) != bodies or len(set(body_names)) != bodies:
        raise ValueError("hand collision body names must be unique and match the pose body axis")
    if expected_body_names is not None and body_names != tuple(expected_body_names):
        raise ValueError("trace body order does not match the runtime geometry manifest")
    if not group_names or len(set(group_names)) != len(group_names):
        raise ValueError("contact group names must be non-empty and unique")

    _validate_pose_array("object_pose", object_pose, (frames, replicas, 7))
    _validate_pose_array("hand_collision_body_pose", hand_pose, (frames, replicas, bodies, 7))
    expected_shapes = {
        "contact_force_world": (frames, replicas, bodies, 3),
        "contact_pair_presence": (frames, replicas, bodies),
        "contact_group_presence": (frames, replicas, len(group_names)),
        "object_twist": (frames, replicas, 6),
        "mean_absolute_effort": (frames, replicas),
        "finite": (frames, replicas),
        "reason_code": (frames, replicas),
        "actions": (frames, 26),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {arrays[name].shape}")
    for name in ("contact_force_world", "object_twist", "mean_absolute_effort", "actions"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains non-finite values")

    penetration = worst = worst_pair = None
    if geometry_path is not None:
        geometry = _load_npz(geometry_path)
        required_geometry = {
            "penetration_depth_m",
            "frame_worst_penetration_m",
            "frame_worst_pair_index",
        }
        missing_geometry = sorted(required_geometry - geometry.keys())
        if missing_geometry:
            raise ValueError(f"geometry trace is missing required arrays: {missing_geometry}")
        penetration = geometry["penetration_depth_m"]
        worst = geometry["frame_worst_penetration_m"]
        worst_pair = geometry["frame_worst_pair_index"]
        for name, value, expected_shape in (
            ("penetration_depth_m", penetration, (frames, replicas, bodies)),
            ("frame_worst_penetration_m", worst, (frames, replicas)),
            ("frame_worst_pair_index", worst_pair, (frames, replicas)),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
        if not np.isfinite(penetration).all() or not np.isfinite(worst).all():
            raise ValueError("geometry trace contains non-finite penetration values")

    return Stage16DSimulationTraceReplay(
        trace_path=trace_path,
        trace_kind="stable_grasp_calibration",
        object_pose=object_pose,
        hand_collision_body_pose=hand_pose,
        hand_collision_body_names=body_names,
        contact_force_world=arrays["contact_force_world"],
        contact_pair_presence=arrays["contact_pair_presence"].astype(bool, copy=False),
        contact_group_presence=arrays["contact_group_presence"].astype(bool, copy=False),
        contact_group_names=group_names,
        object_twist=arrays["object_twist"],
        mean_absolute_effort=arrays["mean_absolute_effort"],
        finite=arrays["finite"].astype(bool, copy=False),
        reason_code=arrays["reason_code"],
        actions=arrays["actions"],
        penetration_depth_m=penetration,
        frame_worst_penetration_m=worst,
        frame_worst_pair_index=worst_pair,
    )


def _hermite_retime_numpy(
    values: np.ndarray,
    derivatives: np.ndarray,
    interval: np.ndarray,
    alpha: np.ndarray,
    *,
    source_dt_s: float,
) -> np.ndarray:
    start = values[interval]
    end = values[interval + 1]
    velocity_start = derivatives[interval]
    velocity_end = derivatives[interval + 1]
    weight = alpha[:, None]
    weight2 = weight * weight
    weight3 = weight2 * weight
    return (
        (2.0 * weight3 - 3.0 * weight2 + 1.0) * start
        + (weight3 - 2.0 * weight2 + weight) * source_dt_s * velocity_start
        + (-2.0 * weight3 + 3.0 * weight2) * end
        + (weight3 - weight2) * source_dt_s * velocity_end
    )


def _quaternion_retime_numpy(
    values: np.ndarray, interval: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    start = values[interval]
    end = values[interval + 1].copy()
    end[np.sum(start * end, axis=-1) < 0.0] *= -1.0
    weight = alpha[:, None]
    result = (1.0 - weight) * start + weight * end
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-12)


def load_factor8_hocap_reference_object_pose(
    path: Path,
    *,
    expected_frames: int,
    time_scale: int = 8,
) -> np.ndarray:
    """Materialize the frozen 41-frame HO-Cap reference as the runtime view.

    This intentionally accepts only the immutable Stage 16 reference schema,
    not a recorded PhysX ``source_trace``.  Its interpolation is the NumPy
    equivalent of :meth:`WorldWristReferenceBank.apply_uniform_time_scale`.
    """

    if isinstance(time_scale, bool) or not isinstance(time_scale, int) or time_scale < 1:
        raise ValueError("reference time_scale must be a positive integer")
    arrays = _load_npz(path)
    missing = sorted(set(HOCAP_REFERENCE_FIELDS) - arrays.keys())
    if missing:
        raise ValueError(
            "HO-Cap reference is missing frozen reference fields; a PhysX source trace "
            f"cannot be used as the ghost: {missing}"
        )
    timestamps = np.asarray(arrays["timestamps"], dtype=np.float64)
    position = np.asarray(arrays["object_pose_translation_world_ref"], dtype=np.float64)
    quaternion = np.asarray(arrays["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64)
    twist = np.asarray(arrays["object_twist_world_ref"], dtype=np.float64)
    if (
        timestamps.shape != (41,)
        or position.shape != (41, 3)
        or quaternion.shape != (41, 4)
        or twist.shape != (41, 6)
    ):
        raise ValueError("HO-Cap reference must be the frozen 41-frame object contract")
    if not all(np.isfinite(value).all() for value in (timestamps, position, quaternion, twist)):
        raise ValueError("HO-Cap reference contains non-finite values")
    source_delta = np.diff(timestamps)
    if not np.all(source_delta > 0.0) or not np.isclose(np.median(source_delta), 0.05, atol=1.0e-8):
        raise ValueError("HO-Cap reference must be the frozen 20 Hz contract")
    quaternion_norm = np.linalg.norm(quaternion, axis=-1)
    if np.any(quaternion_norm < 1.0e-8):
        raise ValueError("HO-Cap reference contains a zero quaternion")
    quaternion = quaternion / quaternion_norm[:, None]

    retimed_frames = (len(timestamps) - 1) * time_scale + 1
    if expected_frames != retimed_frames:
        raise ValueError(
            f"factor-{time_scale} HO-Cap reference produces {retimed_frames} frames, "
            f"but the replay trace has {expected_frames}"
        )
    coordinate = np.arange(retimed_frames, dtype=np.float64) / float(time_scale)
    interval = np.minimum(np.floor(coordinate).astype(np.int64), len(timestamps) - 2)
    alpha = np.clip(coordinate - interval, 0.0, 1.0)
    alpha[-1] = 1.0
    retimed_position = _hermite_retime_numpy(
        position,
        twist[:, :3],
        interval,
        alpha,
        source_dt_s=0.05,
    )
    retimed_quaternion = _quaternion_retime_numpy(quaternion, interval, alpha)
    pose = np.concatenate((retimed_position, retimed_quaternion), axis=-1)
    _validate_pose_array("factor-8 HO-Cap reference object pose", pose, (expected_frames, 7))
    return pose.astype(np.float32)


__all__ = [
    "ReplayFrameDiagnostics",
    "Stage16DSimulationTraceReplay",
    "TRACE_REQUIRED_ARRAYS",
    "CORRECTED_TRACE_REQUIRED_ARRAYS",
    "HOCAP_REFERENCE_FIELDS",
    "infer_object_id",
    "load_factor8_hocap_reference_object_pose",
    "load_stage16d_simulation_trace",
]
