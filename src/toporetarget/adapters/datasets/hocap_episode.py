"""HOCap single-hand/object episode segmentation authority.

The production unit is one active hand, one target object, and one complete
pick-place lifecycle.  Raw-sequence primary-object ranking and fixed temporal
padding are intentionally outside this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from toporetarget.evaluation.source_contact_semantics import (
    FINGER_ORDER,
    REGION_ORDER,
    SEGMENT_ORDER,
    SourceContactThresholdContractV1,
    build_mano_surface_region_map,
)

from .hocap import hocap_mano_storage_index
from .stage12_base import load_mesh, render_mano_pca45, sha256_paths


class HOCapEpisodeError(RuntimeError):
    """Raised when raw episode evidence is absent, malformed, or ambiguous."""


class EpisodeType(str, Enum):
    SINGLE_HAND_PICK_PLACE = "SINGLE_HAND_PICK_PLACE"
    BIMANUAL_SAME_OBJECT = "BIMANUAL_SAME_OBJECT"
    HANDOVER = "HANDOVER"
    INCOMPLETE_INTERACTION = "INCOMPLETE_INTERACTION"


@dataclass(frozen=True)
class HOCapSingleHandObjectEpisodeContractV1:
    """Dataset-wide, outcome-independent segmentation thresholds."""

    schema_version: str = "HOCapSingleHandObjectEpisodeV1"
    distance_authority: str = "MANO_whole_surface_to_exact_object_triangle_mesh"
    contact_surface_distance_m: float = 0.010
    semantic_contact_distance_m: float = 0.005
    semantic_contact_min_vertices: int = 3
    non_interacting_distance_m: float = 0.060
    approach_distance_m: float = 0.100
    min_contact_frames: int = 3
    max_contact_gap_frames: int = 3
    idle_stability_frames: int = 8
    pickup_persistence_frames: int = 3
    pickup_height_m: float = 0.010
    max_stable_linear_speed_mps: float = 0.030
    max_stable_angular_speed_radps: float = 0.150
    manipulation_linear_speed_mps: float = 0.025
    manipulation_angular_speed_radps: float = 0.250
    minimum_object_displacement_m: float = 0.010
    bimanual_overlap_frames: int = 3
    handover_max_gap_frames: int = 15
    returned_near_initial_pose_m: float = 0.050
    frame_range_semantics: str = "start_inclusive_end_exclusive"
    return_semantics: str = "RETURN_TO_NON_INTERACTING_IDLE"

    def __post_init__(self) -> None:
        positive = (
            "contact_surface_distance_m",
            "semantic_contact_distance_m",
            "non_interacting_distance_m",
            "approach_distance_m",
            "pickup_height_m",
            "max_stable_linear_speed_mps",
            "max_stable_angular_speed_radps",
            "manipulation_linear_speed_mps",
            "manipulation_angular_speed_radps",
            "minimum_object_displacement_m",
            "returned_near_initial_pose_m",
        )
        if any(
            not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in positive
        ):
            raise ValueError("HOCAP_EPISODE_THRESHOLD_INVALID")
        if not (
            self.semantic_contact_distance_m
            <= self.contact_surface_distance_m
            < self.non_interacting_distance_m
            < self.approach_distance_m
        ):
            raise ValueError("HOCAP_EPISODE_DISTANCE_ORDER_INVALID")
        if (
            min(
                self.semantic_contact_min_vertices,
                self.min_contact_frames,
                self.idle_stability_frames,
                self.pickup_persistence_frames,
                self.bimanual_overlap_frames,
            )
            <= 0
        ):
            raise ValueError("HOCAP_EPISODE_FRAME_THRESHOLD_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HandObjectSignals:
    """Native-frame interaction evidence for exactly one hand/object pair."""

    subject: str
    raw_sequence: str
    side: str
    object_id: str
    fps: float
    min_surface_distance_m: np.ndarray
    near_surface_vertex_count: np.ndarray
    semantic_contact_region_mask: np.ndarray
    fingertip_distance_m: np.ndarray
    object_translation_world_m: np.ndarray
    object_rotation_world: np.ndarray
    object_linear_speed_mps: np.ndarray
    object_angular_speed_radps: np.ndarray
    object_bottom_height_m: np.ndarray
    relative_translation_rate_mps: np.ndarray
    relative_angular_rate_radps: np.ndarray
    wrist_translation_world_m: np.ndarray
    source_support_metadata: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        frames = len(np.asarray(self.min_surface_distance_m))
        arrays = {
            "near_surface_vertex_count": (frames,),
            "semantic_contact_region_mask": (frames, 6),
            "fingertip_distance_m": (frames, 5),
            "object_translation_world_m": (frames, 3),
            "object_rotation_world": (frames, 3, 3),
            "object_linear_speed_mps": (frames,),
            "object_angular_speed_radps": (frames,),
            "object_bottom_height_m": (frames,),
            "relative_translation_rate_mps": (frames,),
            "relative_angular_rate_radps": (frames,),
            "wrist_translation_world_m": (frames, 3),
        }
        if frames < 2 or not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("HOCAP_EPISODE_SIGNAL_TIMEBASE_INVALID")
        if self.side not in {"left", "right"}:
            raise ValueError("HOCAP_EPISODE_SIGNAL_SIDE_INVALID")
        for name, shape in arrays.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"HOCAP_EPISODE_SIGNAL_INVALID:{name}:{value.shape}:{shape}")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("HOCAP_EPISODE_RUN_MASK_INVALID")
    boundaries = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _persistent(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = np.zeros_like(np.asarray(mask, dtype=bool))
    for start, end in _runs(mask):
        if end - start >= minimum:
            result[start:end] = True
    return result


def _fill_short_false_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for start, end in _runs(~result):
        if start > 0 and end < len(result) and end - start <= maximum_gap:
            result[start:end] = True
    return result


def _first_run(mask: np.ndarray, *, start: int, stop: int, minimum: int) -> tuple[int, int] | None:
    if stop <= start:
        return None
    for run_start, run_end in _runs(np.asarray(mask, dtype=bool)[start:stop]):
        if run_end - run_start >= minimum:
            return start + run_start, start + run_end
    return None


def _last_run(mask: np.ndarray, *, start: int, stop: int, minimum: int) -> tuple[int, int] | None:
    runs = [
        (start + run_start, start + run_end)
        for run_start, run_end in _runs(np.asarray(mask, dtype=bool)[start:stop])
        if run_end - run_start >= minimum
    ]
    return runs[-1] if runs else None


def _event_row(
    signals: HandObjectSignals,
    *,
    episode_type: EpisodeType,
    start_frame: int | None,
    approach_frame: int | None,
    contact_frame: int,
    pickup_frame: int | None,
    transport_frame: int | None,
    place_frame: int | None,
    release_frame: int,
    retreat_frame: int | None,
    end_frame: int | None,
    semantic_regions: Sequence[str],
    exclusion_reason: str,
    contract: HOCapSingleHandObjectEpisodeContractV1,
) -> dict[str, object]:
    complete = all(
        value is not None
        for value in (
            start_frame,
            approach_frame,
            pickup_frame,
            transport_frame,
            place_frame,
            retreat_frame,
            end_frame,
        )
    )
    eligible = complete and episode_type is EpisodeType.SINGLE_HAND_PICK_PLACE
    returned = False
    if start_frame is not None and end_frame is not None and end_frame > start_frame:
        returned = bool(
            np.linalg.norm(
                signals.wrist_translation_world_m[end_frame - 1]
                - signals.wrist_translation_world_m[start_frame]
            )
            <= contract.returned_near_initial_pose_m
        )
    duration = 0 if start_frame is None or end_frame is None else end_frame - start_frame
    return {
        "subject": signals.subject,
        "raw_sequence": signals.raw_sequence,
        "episode_id": "PENDING_DETERMINISTIC_ID",
        "active_hand": signals.side,
        "target_object": signals.object_id,
        "episode_type": episode_type.value,
        "start_frame": start_frame,
        "approach_frame": approach_frame,
        "contact_frame": contact_frame,
        "pickup_frame": pickup_frame,
        "transport_frame": transport_frame,
        "place_frame": place_frame,
        "release_frame": release_frame,
        "retreat_frame": retreat_frame,
        "end_frame": end_frame,
        "duration_frames": duration,
        "duration_seconds": duration / signals.fps,
        "other_hand_same_target": False,
        "overlapping_other_hand_other_object": False,
        "complete": complete,
        "physicalization_v1_eligible": eligible,
        "exclusion_reason": exclusion_reason if not eligible else "",
        "returned_near_initial_pose": returned,
        "return_semantics": contract.return_semantics,
        "semantic_contact_regions": list(semantic_regions),
        "source_support_metadata": dict(signals.source_support_metadata),
        "provenance": dict(signals.provenance),
        "contract_sha256": _stable_hash(contract.as_dict()),
    }


def segment_hand_object_signals(
    signals: HandObjectSignals,
    *,
    contract: HOCapSingleHandObjectEpisodeContractV1 | None = None,
) -> list[dict[str, object]]:
    """Segment complete lifecycles without event-frame padding."""

    frozen = contract or HOCapSingleHandObjectEpisodeContractV1()
    distance = np.asarray(signals.min_surface_distance_m, dtype=np.float64)
    semantic = np.asarray(signals.semantic_contact_region_mask, dtype=bool)
    near_count = np.asarray(signals.near_surface_vertex_count, dtype=np.int64)
    contact = (distance <= frozen.contact_surface_distance_m) & (
        near_count >= frozen.semantic_contact_min_vertices
    )
    contact |= np.any(semantic, axis=1)
    contact = _persistent(
        _fill_short_false_gaps(contact, frozen.max_contact_gap_frames),
        frozen.min_contact_frames,
    )
    stable = (np.asarray(signals.object_linear_speed_mps) <= frozen.max_stable_linear_speed_mps) & (
        np.asarray(signals.object_angular_speed_radps) <= frozen.max_stable_angular_speed_radps
    )
    object_motion = (
        np.asarray(signals.object_linear_speed_mps) >= frozen.manipulation_linear_speed_mps
    ) | (np.asarray(signals.object_angular_speed_radps) >= frozen.manipulation_angular_speed_radps)
    idle = stable & (distance >= frozen.non_interacting_distance_m)
    rows: list[dict[str, object]] = []
    contact_runs = _runs(contact)
    for run_index, (contact_start, contact_end) in enumerate(contact_runs):
        previous_end = contact_runs[run_index - 1][1] if run_index else 0
        next_start = (
            contact_runs[run_index + 1][0] if run_index + 1 < len(contact_runs) else len(contact)
        )
        local_translation = signals.object_translation_world_m[contact_start:contact_end]
        displacement = (
            float(np.max(np.linalg.norm(local_translation - local_translation[0], axis=1)))
            if len(local_translation)
            else 0.0
        )
        if not np.any(object_motion[contact_start:contact_end]) and (
            displacement < frozen.minimum_object_displacement_m
        ):
            continue

        pre_idle = _last_run(
            idle,
            start=previous_end,
            stop=contact_start,
            minimum=frozen.idle_stability_frames,
        )
        start_frame = pre_idle[1] - frozen.idle_stability_frames if pre_idle is not None else None
        approach_frame = pre_idle[1] if pre_idle is not None else None
        if approach_frame is not None:
            approach_candidates = np.flatnonzero(
                distance[approach_frame:contact_start] <= frozen.approach_distance_m
            )
            if len(approach_candidates):
                approach_frame += int(approach_candidates[0])

        baseline_start = pre_idle[0] if pre_idle is not None else previous_end
        baseline_values = signals.object_bottom_height_m[baseline_start:contact_start]
        baseline_stable = stable[baseline_start:contact_start]
        if np.any(baseline_stable):
            baseline_height = float(np.median(baseline_values[baseline_stable]))
        elif len(baseline_values):
            baseline_height = float(np.median(baseline_values))
        else:
            baseline_height = float(signals.object_bottom_height_m[contact_start])
        lifted = np.asarray(signals.object_bottom_height_m) >= (
            baseline_height + frozen.pickup_height_m
        )
        pickup_run = _first_run(
            lifted & contact,
            start=contact_start,
            stop=contact_end,
            minimum=frozen.pickup_persistence_frames,
        )
        pickup_frame = pickup_run[0] if pickup_run is not None else None
        transport_frame = pickup_frame

        release_frame = contact_end
        post_supported = stable & ~lifted
        place_run = (
            _first_run(
                post_supported,
                start=pickup_frame + frozen.pickup_persistence_frames,
                stop=min(next_start, len(contact)),
                minimum=frozen.idle_stability_frames,
            )
            if pickup_frame is not None
            else None
        )
        place_frame = place_run[0] if place_run is not None else None
        if place_frame is not None and place_frame > release_frame:
            # A release preceding stable placement is a drop, not a complete
            # pick-place episode.
            place_frame = None

        post_idle = _first_run(
            idle,
            start=release_frame,
            stop=next_start,
            minimum=frozen.idle_stability_frames,
        )
        retreat_frame = post_idle[0] if post_idle is not None else None
        end_frame = post_idle[0] + frozen.idle_stability_frames if post_idle is not None else None
        missing: list[str] = []
        for name, value in (
            ("IDLE_PRE", start_frame),
            ("APPROACH", approach_frame),
            ("PICKUP", pickup_frame),
            ("TRANSPORT", transport_frame),
            ("PLACE", place_frame),
            ("RETREAT", retreat_frame),
            ("IDLE_POST", end_frame),
        ):
            if value is None:
                missing.append(name)
        active_regions = [
            name
            for index, name in enumerate(REGION_ORDER[:6])
            if bool(np.any(semantic[contact_start:contact_end, index]))
        ]
        rows.append(
            _event_row(
                signals,
                episode_type=(
                    EpisodeType.SINGLE_HAND_PICK_PLACE
                    if not missing
                    else EpisodeType.INCOMPLETE_INTERACTION
                ),
                start_frame=start_frame,
                approach_frame=approach_frame,
                contact_frame=contact_start,
                pickup_frame=pickup_frame,
                transport_frame=transport_frame,
                place_frame=place_frame,
                release_frame=release_frame,
                retreat_frame=retreat_frame,
                end_frame=end_frame,
                semantic_regions=active_regions,
                exclusion_reason=("" if not missing else "MISSING_" + "_".join(missing)),
                contract=frozen,
            )
        )
    return rows


def _interval(row: Mapping[str, object]) -> tuple[int, int]:
    start = row.get("start_frame")
    end = row.get("end_frame")
    contact = int(cast(int, row["contact_frame"]))
    release = int(cast(int, row["release_frame"]))
    return (
        contact if start is None else int(cast(int, start)),
        release if end is None else int(cast(int, end)),
    )


def _interaction_interval(row: Mapping[str, object]) -> tuple[int, int]:
    return int(cast(int, row["contact_frame"])), int(cast(int, row["release_frame"]))


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def classify_sequence_interactions(
    rows: Sequence[Mapping[str, object]],
    *,
    contract: HOCapSingleHandObjectEpisodeContractV1 | None = None,
) -> list[dict[str, object]]:
    """Preserve bimanual/handover semantics and different-object overlap."""

    frozen = contract or HOCapSingleHandObjectEpisodeContractV1()
    result = [dict(row) for row in rows]
    consumed: set[int] = set()
    merged: list[dict[str, object]] = []
    for first_index, first in enumerate(result):
        if first_index in consumed:
            continue
        for second_index in range(first_index + 1, len(result)):
            if second_index in consumed:
                continue
            second = result[second_index]
            if (
                first["target_object"] != second["target_object"]
                or first["active_hand"] == second["active_hand"]
            ):
                continue
            first_interaction = _interaction_interval(first)
            second_interaction = _interaction_interval(second)
            overlap_frames = _overlap(first_interaction, second_interaction)
            gap = max(
                0,
                max(first_interaction[0], second_interaction[0])
                - min(first_interaction[1], second_interaction[1]),
            )
            if overlap_frames < frozen.bimanual_overlap_frames and (
                gap > frozen.handover_max_gap_frames
            ):
                continue
            episode_type = (
                EpisodeType.BIMANUAL_SAME_OBJECT
                if overlap_frames >= frozen.bimanual_overlap_frames
                else EpisodeType.HANDOVER
            )
            base = dict(first)
            base.update(
                {
                    "active_hand": "both",
                    "episode_type": episode_type.value,
                    "start_frame": min(_interval(first)[0], _interval(second)[0]),
                    "approach_frame": min(
                        (
                            int(cast(int, value))
                            for value in (
                                first.get("approach_frame"),
                                second.get("approach_frame"),
                            )
                            if value is not None
                        ),
                        default=None,
                    ),
                    "contact_frame": min(first_interaction[0], second_interaction[0]),
                    "pickup_frame": min(
                        (
                            int(cast(int, value))
                            for value in (
                                first.get("pickup_frame"),
                                second.get("pickup_frame"),
                            )
                            if value is not None
                        ),
                        default=None,
                    ),
                    "transport_frame": min(
                        (
                            int(cast(int, value))
                            for value in (
                                first.get("transport_frame"),
                                second.get("transport_frame"),
                            )
                            if value is not None
                        ),
                        default=None,
                    ),
                    "place_frame": max(
                        (
                            int(cast(int, value))
                            for value in (
                                first.get("place_frame"),
                                second.get("place_frame"),
                            )
                            if value is not None
                        ),
                        default=None,
                    ),
                    "release_frame": max(first_interaction[1], second_interaction[1]),
                    "retreat_frame": max(
                        (
                            int(cast(int, value))
                            for value in (
                                first.get("retreat_frame"),
                                second.get("retreat_frame"),
                            )
                            if value is not None
                        ),
                        default=None,
                    ),
                    "end_frame": max(_interval(first)[1], _interval(second)[1]),
                    "other_hand_same_target": True,
                    "complete": False,
                    "physicalization_v1_eligible": False,
                    "exclusion_reason": episode_type.value,
                    "source_pair_episode_indices": [first_index, second_index],
                }
            )
            base["duration_frames"] = int(cast(int, base["end_frame"])) - int(
                cast(int, base["start_frame"])
            )
            original_duration_frames = int(cast(int, first.get("duration_frames") or 0))
            original_duration_seconds = float(cast(float, first.get("duration_seconds") or 0.0))
            fps = (
                original_duration_frames / original_duration_seconds
                if original_duration_frames > 0 and original_duration_seconds > 0
                else 30.0
            )
            base["duration_seconds"] = float(cast(int, base["duration_frames"])) / max(fps, 1.0e-9)
            consumed.update((first_index, second_index))
            merged.append(base)
            break

    retained = [row for index, row in enumerate(result) if index not in consumed]
    retained.extend(merged)
    for first_index, first in enumerate(retained):
        if first["active_hand"] not in {"left", "right"}:
            continue
        for second_index, second in enumerate(retained):
            if first_index == second_index or second["active_hand"] not in {"left", "right"}:
                continue
            if (
                first["active_hand"] != second["active_hand"]
                and first["target_object"] != second["target_object"]
                and _overlap(_interval(first), _interval(second)) > 0
            ):
                first["overlapping_other_hand_other_object"] = True
                break
    return retained


def assign_episode_ids(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Assign stable IDs after semantic cross-hand merging."""

    output = [dict(row) for row in rows]
    counters: dict[tuple[str, str, str, str], int] = {}
    output.sort(
        key=lambda row: (
            str(row["subject"]),
            str(row["raw_sequence"]),
            int(cast(int, row.get("start_frame") or row["contact_frame"])),
            str(row["active_hand"]),
            str(row["target_object"]),
        )
    )
    for row in output:
        sequence_name = Path(str(row["raw_sequence"])).name
        key = (
            str(row["subject"]),
            sequence_name,
            str(row["active_hand"]),
            str(row["target_object"]),
        )
        index = counters.get(key, 0)
        counters[key] = index + 1
        row["episode_id"] = (
            f"hocap_{row['subject']}_{sequence_name}"
            f"__{row['active_hand']}__{row['target_object']}__ep{index:02d}"
        )
    return output


def _pose_matrices(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise HOCapEpisodeError(f"HOCAP_OBJECT_POSE_INVALID:{values.shape}")
    matrices = np.broadcast_to(np.eye(4), (len(values), 4, 4)).copy()
    matrices[:, :3, :3] = Rotation.from_quat(values[:, :4]).as_matrix()
    matrices[:, :3, 3] = values[:, 4:]
    return matrices


def _pose_rates(poses: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(poses[:, :3, 3], dtype=np.float64)
    rotation = np.asarray(poses[:, :3, :3], dtype=np.float64)
    linear = np.zeros(len(poses), dtype=np.float64)
    angular = np.zeros(len(poses), dtype=np.float64)
    linear[1:] = np.linalg.norm(np.diff(translation, axis=0), axis=1) * fps
    relative = np.einsum("tji,tjk->tik", rotation[:-1], rotation[1:])
    angular[1:] = Rotation.from_matrix(relative).magnitude() * fps
    return linear, angular


def _relative_pose_rates(
    hand_poses: np.ndarray, object_poses: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
    inverse_hand = np.linalg.inv(np.asarray(hand_poses, dtype=np.float64))
    relative = np.einsum("tij,tjk->tik", inverse_hand, object_poses)
    return _pose_rates(relative, fps)


@lru_cache(maxsize=4)
def _mano_region_data(mano_model_root: str, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # MANO v1.2 may import old chumpy aliases.  Keep the compatibility shim as
    # narrow as the existing source-contact authority.
    aliases: dict[str, object] = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    import smplx

    is_right = side == "right"
    path = Path(mano_model_root) / ("MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl")
    model = smplx.create(
        model_path=str(path),
        model_type="mano",
        is_rhand=is_right,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    )
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int64)
    rest_vertices = model.v_template.detach().cpu().numpy().astype(np.float64)
    regressor = model.J_regressor
    rest_joints = (
        regressor.to_dense().detach().cpu().numpy()
        if hasattr(regressor, "to_dense")
        else regressor.detach().cpu().numpy()
    ) @ rest_vertices
    region_map = build_mano_surface_region_map(
        weights,
        faces,
        rest_vertices,
        rest_joints,
        contract=SourceContactThresholdContractV1(),
    )
    return faces, region_map.region_id, region_map.segment_id


def _exact_unsigned_distance(
    vertices: np.ndarray, faces: np.ndarray, points_local: np.ndarray
) -> tuple[np.ndarray, str]:
    mesh_vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    mesh_faces = np.ascontiguousarray(faces, dtype=np.int64)
    points = np.ascontiguousarray(points_local, dtype=np.float64)
    try:
        from toporetarget.geometry.signed_distance.compiled_sdf_cpu import (
            CompiledBVHHandle,
            compiled_available,
        )

        if compiled_available():
            handle = CompiledBVHHandle(mesh_vertices, mesh_faces)
            return np.asarray(handle.query(points)[3]), handle.backend_id
    except (ImportError, OSError, RuntimeError):
        pass
    from toporetarget.geometry.signed_distance.closest_point import ObjectLocalBVH

    tree = ObjectLocalBVH(mesh_vertices[mesh_faces])
    return np.asarray(tree.query(points)[3]), tree.backend_id


def _source_support_metadata(sequence_dir: Path, meta: Mapping[str, object]) -> dict[str, object]:
    terms = ("table", "support", "plane", "floor", "scene_geometry")
    hits = {
        str(key): value
        for key, value in meta.items()
        if any(term in str(key).lower() for term in terms)
    }
    candidates = [
        path
        for path in sorted(sequence_dir.rglob("*"))
        if path.is_file() and any(term in path.name.lower() for term in terms)
    ]
    return {
        "authority": "SOURCE_METADATA_AUDIT_ONLY",
        "metadata_hits": hits,
        "geometry_candidates": [str(path.resolve()) for path in candidates],
        "source_explicit_support_present": bool(hits),
        "source_reconstructed_support_candidate_present": bool(candidates),
    }


def extract_sequence_signals(
    sequence_dir: Path,
    *,
    dataset_root: Path,
    mano_model_root: Path,
    selected_sides: Iterable[str] | None = None,
    contract: HOCapSingleHandObjectEpisodeContractV1 | None = None,
) -> tuple[list[HandObjectSignals], dict[str, object]]:
    """Extract every official hand/object pair from one raw HOCap sequence."""

    frozen = contract or HOCapSingleHandObjectEpisodeContractV1()
    sequence_dir = sequence_dir.resolve()
    meta_path = sequence_dir / "meta.yaml"
    poses_m_path = sequence_dir / "poses_m.npy"
    poses_o_path = sequence_dir / "poses_o.npy"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    subject = str(meta.get("subject_id") or sequence_dir.parent.name)
    sides = [str(side).lower() for side in (meta.get("mano_sides") or [])]
    if (
        not sides
        or len(sides) != len(set(sides))
        or any(side not in {"left", "right"} for side in sides)
    ):
        raise HOCapEpisodeError(f"HOCAP_EPISODE_OFFICIAL_HAND_SIDES_INVALID:{sequence_dir}")
    requested = set(sides if selected_sides is None else selected_sides)
    sides_to_run = [side for side in sides if side in requested]
    object_ids = [str(value) for value in (meta.get("object_ids") or [])]
    poses_m = np.asarray(np.load(poses_m_path, mmap_mode="r"), dtype=np.float64)
    poses_o = np.asarray(np.load(poses_o_path, mmap_mode="r"), dtype=np.float64)
    required_slots = max(hocap_mano_storage_index(side) for side in sides) + 1
    if poses_m.ndim != 3 or poses_m.shape[2] != 51 or poses_m.shape[0] < required_slots:
        raise HOCapEpisodeError(f"HOCAP_EPISODE_MANO_POSE_INVALID:{poses_m.shape}")
    if poses_o.ndim != 3 or poses_o.shape[2] != 7:
        raise HOCapEpisodeError(f"HOCAP_EPISODE_OBJECT_POSE_INVALID:{poses_o.shape}")
    if poses_o.shape[1] == poses_m.shape[1] and poses_o.shape[0] != poses_m.shape[1]:
        poses_o = poses_o.transpose(1, 0, 2)
    frames = min(
        poses_m.shape[1],
        poses_o.shape[0],
        int(meta.get("num_frames") or poses_m.shape[1]),
    )
    if len(object_ids) != poses_o.shape[1] or frames < frozen.idle_stability_frames * 2:
        raise HOCapEpisodeError("HOCAP_EPISODE_OBJECT_CARDINALITY_OR_LENGTH_INVALID")
    poses_m = poses_m[:, :frames]
    poses_o = poses_o[:frames]
    if not all(np.isfinite(poses_m[hocap_mano_storage_index(side)]).all() for side in sides):
        raise HOCapEpisodeError("HOCAP_EPISODE_RAW_MANO_POSE_NONFINITE")
    if not np.isfinite(poses_o).all():
        raise HOCapEpisodeError("HOCAP_EPISODE_RAW_POSE_NONFINITE")
    fps = float(meta.get("fps") or meta.get("frame_rate") or 30.0)
    calibration_path = dataset_root / "data/calibration/mano" / f"{subject}.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8")) or {}
    betas = np.asarray(calibration.get("betas"), dtype=np.float64)
    if betas.shape != (10,) or not np.isfinite(betas).all():
        raise HOCapEpisodeError(f"HOCAP_EPISODE_BETAS_INVALID:{calibration_path}")
    source_hash = sha256_paths([meta_path, poses_m_path, poses_o_path, calibration_path])
    support = _source_support_metadata(sequence_dir, meta)

    object_data: dict[str, dict[str, object]] = {}
    for object_index, object_id in enumerate(object_ids):
        mesh_path = dataset_root / "data/models" / object_id / "textured_mesh.obj"
        vertices, faces = load_mesh(mesh_path)
        matrices = _pose_matrices(poses_o[:, object_index])
        linear, angular = _pose_rates(matrices, fps)
        bottom = np.empty(frames, dtype=np.float64)
        for start in range(0, frames, 128):
            stop = min(start + 128, frames)
            bottom[start:stop] = np.min(
                np.einsum("ti,vi->tv", matrices[start:stop, 2, :3], vertices)
                + matrices[start:stop, None, 2, 3],
                axis=1,
            )
        object_data[object_id] = {
            "mesh_path": mesh_path,
            "vertices": vertices,
            "faces": faces,
            "poses": matrices,
            "linear": linear,
            "angular": angular,
            "bottom": bottom,
        }

    output: list[HandObjectSignals] = []
    backends: set[str] = set()
    for side in sides:
        if side not in sides_to_run:
            continue
        hand_index = hocap_mano_storage_index(side)
        render = render_mano_pca45(
            poses_m[hand_index],
            side=side,
            mano_model_root=mano_model_root,
            betas=betas,
            dataset_name="hocap",
            source_annotation_path=poses_m_path,
            source_annotation_hash=source_hash,
        )
        hand_vertices = np.asarray(render.vertices, dtype=np.float64)
        wrist_poses = np.asarray(render.wrist_pose_scene, dtype=np.float64)
        mano_faces, region_id, segment_id = _mano_region_data(str(mano_model_root.resolve()), side)
        if not np.array_equal(np.asarray(render.faces, dtype=np.int64), mano_faces):
            raise HOCapEpisodeError(f"HOCAP_EPISODE_MANO_TOPOLOGY_MISMATCH:{side}")
        tip_masks = [
            (region_id == REGION_ORDER.index(finger))
            & (segment_id == SEGMENT_ORDER.index("tip_surface"))
            for finger in FINGER_ORDER
        ]
        for object_id in object_ids:
            data = object_data[object_id]
            object_poses = np.asarray(data["poses"], dtype=np.float64)
            local = np.einsum(
                "tvi,tij->tvj",
                hand_vertices - object_poses[:, None, :3, 3],
                object_poses[:, :3, :3],
            ).reshape(-1, 3)
            distance_flat, backend = _exact_unsigned_distance(
                np.asarray(data["vertices"]), np.asarray(data["faces"]), local
            )
            backends.add(backend)
            distance = distance_flat.reshape(frames, hand_vertices.shape[1])
            semantic = np.zeros((frames, 6), dtype=bool)
            for region_index in range(6):
                region_values = distance[:, region_id == region_index]
                semantic[:, region_index] = (
                    np.sum(region_values <= frozen.semantic_contact_distance_m, axis=1)
                    >= frozen.semantic_contact_min_vertices
                )
            tips = np.stack(
                [np.min(distance[:, mask], axis=1) for mask in tip_masks],
                axis=1,
            )
            relative_linear, relative_angular = _relative_pose_rates(wrist_poses, object_poses, fps)
            output.append(
                HandObjectSignals(
                    subject=subject,
                    raw_sequence=str(sequence_dir.relative_to(dataset_root / "data")),
                    side=side,
                    object_id=object_id,
                    fps=fps,
                    min_surface_distance_m=np.min(distance, axis=1),
                    near_surface_vertex_count=np.sum(
                        distance <= frozen.contact_surface_distance_m, axis=1
                    ),
                    semantic_contact_region_mask=semantic,
                    fingertip_distance_m=tips,
                    object_translation_world_m=object_poses[:, :3, 3],
                    object_rotation_world=object_poses[:, :3, :3],
                    object_linear_speed_mps=np.asarray(data["linear"]),
                    object_angular_speed_radps=np.asarray(data["angular"]),
                    object_bottom_height_m=np.asarray(data["bottom"]),
                    relative_translation_rate_mps=relative_linear,
                    relative_angular_rate_radps=relative_angular,
                    wrist_translation_world_m=wrist_poses[:, :3, 3],
                    source_support_metadata=support,
                    provenance={
                        "meta": {"path": str(meta_path), "sha256": _sha256(meta_path)},
                        "raw_mano": {
                            "path": str(poses_m_path),
                            "sha256": _sha256(poses_m_path),
                            "official_hand_index": hand_index,
                        },
                        "raw_object": {
                            "path": str(poses_o_path),
                            "sha256": _sha256(poses_o_path),
                            "official_object_index": object_ids.index(object_id),
                        },
                        "object_mesh": {
                            "path": str(data["mesh_path"]),
                            "sha256": _sha256(Path(str(data["mesh_path"]))),
                        },
                        "mano_calibration": {
                            "path": str(calibration_path),
                            "sha256": _sha256(calibration_path),
                        },
                        "distance_backend": backend,
                        "pose_derived_angular_kinematics": True,
                    },
                )
            )
    receipt = {
        "subject": subject,
        "raw_sequence": str(sequence_dir.relative_to(dataset_root / "data")),
        "frames": frames,
        "fps": fps,
        "official_hands": sides,
        "parsed_hands": sides_to_run,
        "objects": object_ids,
        "candidate_hand_object_pairs": len(sides_to_run) * len(object_ids),
        "distance_backends": sorted(backends),
        "source_support_metadata": support,
        "source_hash": source_hash,
    }
    return output, receipt


def parse_sequence(
    sequence_dir: Path,
    *,
    dataset_root: Path,
    mano_model_root: Path,
    selected_sides: Iterable[str] | None = None,
    contract: HOCapSingleHandObjectEpisodeContractV1 | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Extract, segment, classify, and ID one raw sequence."""

    frozen = contract or HOCapSingleHandObjectEpisodeContractV1()
    signals, receipt = extract_sequence_signals(
        sequence_dir,
        dataset_root=dataset_root,
        mano_model_root=mano_model_root,
        selected_sides=selected_sides,
        contract=frozen,
    )
    rows = [row for pair in signals for row in segment_hand_object_signals(pair, contract=frozen)]
    rows = classify_sequence_interactions(rows, contract=frozen)
    rows = assign_episode_ids(rows)
    receipt["candidate_episode_count"] = len(rows)
    receipt["eligible_episode_count"] = sum(
        bool(row["physicalization_v1_eligible"]) for row in rows
    )
    return rows, receipt


def aggregate_episode_rows(
    rows: Sequence[Mapping[str, object]], sequence_receipts: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Build the required dataset-wide deterministic aggregate."""

    def count(**matches: object) -> int:
        return sum(all(row.get(key) == value for key, value in matches.items()) for row in rows)

    return {
        "schema_version": "HOCapSingleHandObjectEpisodeV1Aggregate",
        "number_sequences": len(sequence_receipts),
        "number_candidate_hand_object_pairs": sum(
            int(cast(int, row["candidate_hand_object_pairs"])) for row in sequence_receipts
        ),
        "left_single_hand_episodes": count(
            active_hand="left", episode_type=EpisodeType.SINGLE_HAND_PICK_PLACE.value
        ),
        "right_single_hand_episodes": count(
            active_hand="right", episode_type=EpisodeType.SINGLE_HAND_PICK_PLACE.value
        ),
        "bimanual_episodes": count(episode_type=EpisodeType.BIMANUAL_SAME_OBJECT.value),
        "handover_episodes": count(episode_type=EpisodeType.HANDOVER.value),
        "complete_pick_place": sum(bool(row.get("complete")) for row in rows),
        "incomplete": sum(not bool(row.get("complete")) for row in rows),
        "eligible_episodes": sum(bool(row.get("physicalization_v1_eligible")) for row in rows),
        "overlapping_different_object_episodes": sum(
            bool(row.get("overlapping_other_hand_other_object")) for row in rows
        ),
    }


__all__ = [
    "EpisodeType",
    "HOCapEpisodeError",
    "HOCapSingleHandObjectEpisodeContractV1",
    "HandObjectSignals",
    "aggregate_episode_rows",
    "assign_episode_ids",
    "classify_sequence_interactions",
    "extract_sequence_signals",
    "parse_sequence",
    "segment_hand_object_signals",
]
