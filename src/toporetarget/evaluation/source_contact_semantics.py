"""Source MANO surface-contact semantics for the Stage 16-D final audit.

The helpers here deliberately do not know about Reward V3 or Isaac Lab.  They
turn MANO LBS weights into repeatable hand regions, reduce exact
point-to-triangle distances to contact evidence, and make the native-to-control
time mapping explicit.  The final audit consumes their output read-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

FINGER_ORDER: Final = ("thumb", "index", "middle", "ring", "pinky")
REGION_ORDER: Final = (*FINGER_ORDER, "palm", "boundary_ambiguous")
SEGMENT_ORDER: Final = (
    "palm",
    "proximal",
    "middle",
    "distal",
    "tip_surface",
    "boundary_ambiguous",
)

# MANO v1.2 has one wrist/root and three articulated joints per digit.  This
# ordering is part of the model's kinematic tree, not a manually curated list
# of vertex IDs.  See MANO_RIGHT.pkl's kintree_table for the parent chain.
MANO_V12_JOINT_CHAINS: Final = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (13, 14, 15),
}

SOURCE_CLASSES: Final = (
    "SOURCE_CONTACT_CONFIRMED",
    "SOURCE_CONTACT_PROBABLE",
    "SOURCE_CONTACT_TRANSITION",
    "SOURCE_PROXIMITY_ONLY",
    "SOURCE_NO_CONTACT",
    "SOURCE_CONTACT_PERSISTENT",
)
SOURCE_CLASS_TO_CODE: Final = {name: index for index, name in enumerate(SOURCE_CLASSES)}


@dataclass(frozen=True)
class SourceContactThresholdContractV1:
    """Frozen source-surface evidence thresholds, all expressed in metres."""

    identifier: str = "SourcePerFingerContactEvidenceV1"
    nominal_min_distance_m: float = 0.002
    sensitivity_min_distance_m: tuple[float, ...] = (0.001, 0.002, 0.005)
    proximity_distance_m: float = 0.010
    component_distance_m: float = 0.005
    minimum_component_vertices: int = 3
    native_persistence_frames: int = 2
    boundary_margin: float = 0.10
    tip_quantile: float = 0.80
    timing_mapping: str = "native_41_keys_to_factor8_control_321"
    training_use: str = "forbidden_diagnostic_only"

    def __post_init__(self) -> None:
        if self.nominal_min_distance_m != 0.002:
            raise ValueError("SOURCE_CONTACT_NOMINAL_THRESHOLD_DRIFT")
        if self.sensitivity_min_distance_m != (0.001, 0.002, 0.005):
            raise ValueError("SOURCE_CONTACT_SENSITIVITY_THRESHOLD_DRIFT")
        if self.minimum_component_vertices != 3 or self.native_persistence_frames != 2:
            raise ValueError("SOURCE_CONTACT_ROBUSTNESS_CONTRACT_DRIFT")
        if not 0.0 < self.boundary_margin < 1.0 or not 0.0 < self.tip_quantile < 1.0:
            raise ValueError("SOURCE_CONTACT_REGION_PARAMETERS_INVALID")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ManoSurfaceRegionMap:
    """Topology-derived MANO region and segment assignment."""

    region_id: np.ndarray
    segment_id: np.ndarray
    soft_region_weight: np.ndarray
    region_names: tuple[str, ...] = REGION_ORDER
    segment_names: tuple[str, ...] = SEGMENT_ORDER
    assignment_method: str = "mano_v12_lbs_chain_soft_weights_v1"

    def __post_init__(self) -> None:
        vertex_count = int(np.asarray(self.region_id).size)
        if np.asarray(self.region_id).shape != (vertex_count,):
            raise ValueError("MANO_REGION_ID_SHAPE_INVALID")
        if np.asarray(self.segment_id).shape != (vertex_count,):
            raise ValueError("MANO_SEGMENT_ID_SHAPE_INVALID")
        if np.asarray(self.soft_region_weight).shape != (vertex_count, 6):
            raise ValueError("MANO_REGION_SOFT_WEIGHT_SHAPE_INVALID")
        if not np.isfinite(self.soft_region_weight).all():
            raise ValueError("MANO_REGION_SOFT_WEIGHT_NONFINITE")

    @property
    def boundary_region_id(self) -> int:
        return self.region_names.index("boundary_ambiguous")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "ManoSurfaceRegionMapV1",
            "assignment_method": self.assignment_method,
            "mano_joint_chains": {key: list(value) for key, value in MANO_V12_JOINT_CHAINS.items()},
            "region_names": list(self.region_names),
            "segment_names": list(self.segment_names),
            "vertex_count": int(self.region_id.size),
            "region_vertex_counts": {
                name: int(np.count_nonzero(self.region_id == index))
                for index, name in enumerate(self.region_names)
            },
            "segment_vertex_counts": {
                name: int(np.count_nonzero(self.segment_id == index))
                for index, name in enumerate(self.segment_names)
            },
            "soft_weight_normalization": "palm_plus_five_finger_chain_weights_sum_to_one",
        }


def _validate_mano_topology(
    weights: np.ndarray, faces: np.ndarray, rest_vertices: np.ndarray, rest_joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    skinning = np.asarray(weights, dtype=np.float64)
    mesh_faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(rest_vertices, dtype=np.float64)
    joints = np.asarray(rest_joints, dtype=np.float64)
    if skinning.ndim != 2 or skinning.shape[1] != 16:
        raise ValueError("MANO_LBS_WEIGHTS_MUST_BE_[V,16]")
    if vertices.shape != (skinning.shape[0], 3) or joints.shape != (16, 3):
        raise ValueError("MANO_REGION_REST_SHAPE_INVALID")
    if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3 or mesh_faces.size == 0:
        raise ValueError("MANO_REGION_FACES_INVALID")
    if mesh_faces.min() < 0 or mesh_faces.max() >= len(vertices):
        raise ValueError("MANO_REGION_FACE_INDEX_INVALID")
    if (
        not np.isfinite(skinning).all()
        or not np.isfinite(vertices).all()
        or not np.isfinite(joints).all()
    ):
        raise ValueError("MANO_REGION_NONFINITE")
    if not np.allclose(skinning.sum(axis=1), 1.0, atol=1.0e-6):
        raise ValueError("MANO_LBS_WEIGHTS_NOT_NORMALIZED")
    return skinning, mesh_faces, vertices, joints


def build_mano_surface_region_map(
    weights: np.ndarray,
    faces: np.ndarray,
    rest_vertices: np.ndarray,
    rest_joints: np.ndarray,
    *,
    contract: SourceContactThresholdContractV1 | None = None,
) -> ManoSurfaceRegionMap:
    """Assign every MANO vertex from LBS influence and kinematic geometry.

    Finger ownership is the sum of its three MANO LBS joint weights.  The
    root-joint weight is the palm candidate.  Vertices without a clear winner
    are retained as ``boundary_ambiguous`` rather than silently assigned to a
    finger.  ``tip_surface`` is the terminal longitudinal quantile of a finger
    and is recomputed from the actual model geometry, so no fixed vertex IDs
    are encoded in the contract.
    """

    frozen = contract or SourceContactThresholdContractV1()
    skinning, _faces, vertices, joints = _validate_mano_topology(
        weights, faces, rest_vertices, rest_joints
    )
    soft = np.empty((len(vertices), 6), dtype=np.float64)
    # Keep the report order (thumb, index, middle, ring, pinky, palm), while
    # using the MANO model's own joint numbering for each chain.
    for column, finger in enumerate(FINGER_ORDER):
        soft[:, column] = skinning[:, MANO_V12_JOINT_CHAINS[finger]].sum(axis=1)
    soft[:, 5] = skinning[:, 0]
    if not np.allclose(soft.sum(axis=1), 1.0, atol=1.0e-6):
        raise AssertionError("MANO_REGION_SOFT_WEIGHTS_LOST_MASS")

    order = np.argsort(soft, axis=1)
    winner = order[:, -1]
    margin = soft[np.arange(len(soft)), winner] - soft[np.arange(len(soft)), order[:, -2]]
    region = winner.astype(np.int16)
    region[margin < frozen.boundary_margin] = len(REGION_ORDER) - 1

    segment = np.full(len(vertices), SEGMENT_ORDER.index("boundary_ambiguous"), dtype=np.int16)
    segment[region == REGION_ORDER.index("palm")] = SEGMENT_ORDER.index("palm")
    for finger_index, finger in enumerate(FINGER_ORDER):
        owned = region == finger_index
        if not np.any(owned):
            raise ValueError(f"MANO_REGION_EMPTY_FINGER:{finger}")
        chain = MANO_V12_JOINT_CHAINS[finger]
        local_joint = np.argmax(skinning[:, chain], axis=1)
        segment[owned] = local_joint[owned] + SEGMENT_ORDER.index("proximal")
        distal = owned & (local_joint == 2)
        axis = joints[chain[2]] - joints[chain[0]]
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1.0e-9:
            raise ValueError(f"MANO_REGION_DEGENERATE_FINGER_AXIS:{finger}")
        longitudinal = (vertices - joints[chain[0]]) @ (axis / axis_norm)
        if np.any(distal):
            tip_cutoff = float(np.quantile(longitudinal[distal], frozen.tip_quantile))
            segment[distal & (longitudinal >= tip_cutoff)] = SEGMENT_ORDER.index("tip_surface")

    result = ManoSurfaceRegionMap(region, segment, soft)
    required = (*FINGER_ORDER, "palm")
    missing = [
        name for name in required if not np.any(result.region_id == result.region_names.index(name))
    ]
    if missing:
        raise AssertionError(f"MANO_REGION_REQUIRED_GROUP_EMPTY:{missing}")
    return result


def mesh_adjacency(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return deterministic face-edge adjacency for component-size checks."""

    if vertex_count <= 0:
        raise ValueError("MANO_ADJACENCY_VERTEX_COUNT_INVALID")
    values = np.asarray(faces, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("MANO_ADJACENCY_FACES_INVALID")
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for a, b, c in values:
        for first, second in ((a, b), (b, c), (c, a)):
            neighbors[int(first)].add(int(second))
            neighbors[int(second)].add(int(first))
    return tuple(np.asarray(sorted(value), dtype=np.int64) for value in neighbors)


def largest_connected_component_size(active: np.ndarray, adjacency: tuple[np.ndarray, ...]) -> int:
    """Find the largest active component using the MANO triangle topology."""

    mask = np.asarray(active, dtype=bool)
    if mask.shape != (len(adjacency),):
        raise ValueError("MANO_COMPONENT_ACTIVE_SHAPE_INVALID")
    visited = np.zeros_like(mask)
    largest = 0
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        stack = [int(start)]
        visited[start] = True
        count = 0
        while stack:
            current = stack.pop()
            count += 1
            for neighbor in adjacency[current]:
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
        largest = max(largest, count)
    return largest


def per_region_surface_statistics(
    distances_m: np.ndarray,
    region_map: ManoSurfaceRegionMap,
    faces: np.ndarray,
    *,
    thresholds_m: tuple[float, ...] = (0.001, 0.002, 0.005, 0.010),
) -> dict[str, np.ndarray]:
    """Reduce exact vertex-to-triangle distances to per-region diagnostics."""

    distance = np.asarray(distances_m, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[1] != region_map.region_id.size:
        raise ValueError("SOURCE_CONTACT_DISTANCE_MUST_BE_[T,V]")
    if not np.isfinite(distance).all() or np.any(distance < 0.0):
        raise ValueError("SOURCE_CONTACT_DISTANCE_INVALID")
    thresholds = np.asarray(thresholds_m, dtype=np.float64)
    if thresholds.ndim != 1 or np.any(thresholds <= 0.0):
        raise ValueError("SOURCE_CONTACT_THRESHOLDS_INVALID")
    adjacency = mesh_adjacency(distance.shape[1], faces)
    # Only six semantic regions enter contact decisions.  Boundary vertices are
    # preserved in the map but never silently promoted to one finger.
    region_count = 6
    shape = (distance.shape[0], region_count)
    minimum = np.full(shape, np.inf, dtype=np.float64)
    p01 = np.full(shape, np.inf, dtype=np.float64)
    p05 = np.full(shape, np.inf, dtype=np.float64)
    counts = np.zeros((*shape, len(thresholds)), dtype=np.int32)
    fractions = np.zeros((*shape, len(thresholds)), dtype=np.float64)
    component5 = np.zeros(shape, dtype=np.int32)
    for region_index in range(region_count):
        vertices = region_map.region_id == region_index
        if not np.any(vertices):
            raise ValueError(f"SOURCE_CONTACT_EMPTY_REGION:{region_index}")
        values = distance[:, vertices]
        minimum[:, region_index] = values.min(axis=1)
        p01[:, region_index] = np.quantile(values, 0.01, axis=1)
        p05[:, region_index] = np.quantile(values, 0.05, axis=1)
        for threshold_index, threshold in enumerate(thresholds):
            hit = values <= threshold
            counts[:, region_index, threshold_index] = hit.sum(axis=1)
            fractions[:, region_index, threshold_index] = hit.mean(axis=1)
        active_threshold = thresholds[np.argmin(np.abs(thresholds - 0.005))]
        for frame in range(distance.shape[0]):
            active = vertices & (distance[frame] <= active_threshold)
            component5[frame, region_index] = largest_connected_component_size(active, adjacency)
    return {
        "minimum_surface_distance_m": minimum,
        "p01_surface_distance_m": p01,
        "p05_surface_distance_m": p05,
        "thresholds_m": thresholds,
        "near_vertex_count": counts,
        "near_vertex_fraction": fractions,
        "largest_component_vertices_at_5mm": component5,
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        result.append((start, end))
        start = end
    return result


def persistent_mask(mask: np.ndarray, minimum_frames: int = 2) -> np.ndarray:
    """Retain runs meeting the declared native-frame persistence criterion."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1 or minimum_frames <= 0:
        raise ValueError("SOURCE_CONTACT_PERSISTENCE_INPUT_INVALID")
    result = np.zeros_like(values)
    for start, end in _runs(values):
        if end - start >= minimum_frames:
            result[start:end] = True
    return result


def classify_source_contact(
    minimum_surface_distance_m: np.ndarray,
    largest_component_vertices_at_5mm: np.ndarray,
    *,
    threshold_m: float | None = None,
    contract: SourceContactThresholdContractV1 | None = None,
) -> dict[str, np.ndarray]:
    """Classify source contact from robust surface evidence at one threshold."""

    frozen = contract or SourceContactThresholdContractV1()
    threshold = frozen.nominal_min_distance_m if threshold_m is None else float(threshold_m)
    minimum = np.asarray(minimum_surface_distance_m, dtype=np.float64)
    component = np.asarray(largest_component_vertices_at_5mm, dtype=np.int32)
    if minimum.ndim != 2 or component.shape != minimum.shape:
        raise ValueError("SOURCE_CONTACT_CLASSIFICATION_SHAPE_INVALID")
    if not np.isfinite(minimum).all() or np.any(minimum < 0.0):
        raise ValueError("SOURCE_CONTACT_CLASSIFICATION_DISTANCE_INVALID")
    raw = (minimum <= threshold) & (component >= frozen.minimum_component_vertices)
    confirmed = np.zeros_like(raw)
    for finger in range(raw.shape[1]):
        confirmed[:, finger] = persistent_mask(raw[:, finger], frozen.native_persistence_frames)
    probable = raw & ~confirmed
    transition = np.zeros_like(raw)
    transition[1:] |= raw[:-1] & ~raw[1:]
    transition[:-1] |= raw[1:] & ~raw[:-1]
    transition &= ~raw
    proximity = (minimum <= frozen.proximity_distance_m) & ~raw & ~transition
    label = np.full(minimum.shape, "SOURCE_NO_CONTACT", dtype="<U32")
    label[proximity] = "SOURCE_PROXIMITY_ONLY"
    label[transition] = "SOURCE_CONTACT_TRANSITION"
    label[probable] = "SOURCE_CONTACT_PROBABLE"
    label[confirmed] = "SOURCE_CONTACT_CONFIRMED"
    confidence = np.full(minimum.shape, "LOW", dtype="<U8")
    confidence[proximity] = "LOW"
    confidence[transition] = "LOW"
    confidence[probable] = "HIGH"
    confidence[confirmed] = "HIGH"
    return {
        "raw_robust_contact": raw,
        "confirmed_contact": confirmed,
        "probable_contact": probable,
        "transition": transition,
        "proximity_only": proximity,
        "class": label,
        "confidence": confidence,
    }


def map_native_contact_to_control(
    native_class: np.ndarray,
    *,
    factor: int = 8,
    control_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Map source keys to factor-8 control with explicit interval semantics.

    Exact source keys retain their direct class.  Only two adjacent confirmed
    source keys create a persistent expected-contact interval.  Two no-contact
    endpoints create no contact; every other interval is a transition and is
    not eligible to manufacture a confirmed expected-contact frame.
    """

    labels = np.asarray(native_class)
    if labels.ndim != 2:
        raise ValueError("SOURCE_CONTACT_NATIVE_CLASS_MUST_BE_[K,F]")
    if factor <= 0:
        raise ValueError("SOURCE_CONTACT_FACTOR_INVALID")
    frames = int(control_frames or (factor * (labels.shape[0] - 1) + 1))
    if frames != factor * (labels.shape[0] - 1) + 1:
        raise ValueError("SOURCE_CONTACT_CONTROL_LENGTH_INVALID")
    result = np.full((frames, labels.shape[1]), "SOURCE_NO_CONTACT", dtype="<U32")
    exact = np.zeros_like(result, dtype=bool)
    for native_index in range(labels.shape[0]):
        control_index = factor * native_index
        result[control_index] = labels[native_index]
        exact[control_index] = True
    confirmed = labels == "SOURCE_CONTACT_CONFIRMED"
    no_contact = labels == "SOURCE_NO_CONTACT"
    for native_index in range(labels.shape[0] - 1):
        start = factor * native_index + 1
        stop = factor * (native_index + 1)
        both_confirmed = confirmed[native_index] & confirmed[native_index + 1]
        both_no_contact = no_contact[native_index] & no_contact[native_index + 1]
        result[start:stop, both_confirmed] = "SOURCE_CONTACT_PERSISTENT"
        result[start:stop, ~both_confirmed & ~both_no_contact] = "SOURCE_CONTACT_TRANSITION"
    expected = np.isin(
        result,
        ("SOURCE_CONTACT_CONFIRMED", "SOURCE_CONTACT_PROBABLE", "SOURCE_CONTACT_PERSISTENT"),
    )
    return {
        "class": result,
        "expected_contact": expected,
        "exact_source_key": exact,
        "native_to_control_index": np.arange(labels.shape[0], dtype=np.int64) * factor,
    }


def source_contact_localization(
    distances_m: np.ndarray,
    region_map: ManoSurfaceRegionMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame/finger nearest semantic segment and its distance."""

    distance = np.asarray(distances_m, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[1] != region_map.region_id.size:
        raise ValueError("SOURCE_CONTACT_LOCALIZATION_SHAPE_INVALID")
    segments = np.full((distance.shape[0], len(FINGER_ORDER)), "boundary_ambiguous", dtype="<U24")
    values = np.full((distance.shape[0], len(FINGER_ORDER)), np.inf, dtype=np.float64)
    for finger_index, _finger in enumerate(FINGER_ORDER):
        owned = region_map.region_id == finger_index
        for segment_index, segment_name in enumerate(SEGMENT_ORDER):
            member = owned & (region_map.segment_id == segment_index)
            if not np.any(member):
                continue
            candidate = distance[:, member].min(axis=1)
            better = candidate < values[:, finger_index]
            values[better, finger_index] = candidate[better]
            segments[better, finger_index] = segment_name
    return segments, values


__all__ = [
    "FINGER_ORDER",
    "MANO_V12_JOINT_CHAINS",
    "REGION_ORDER",
    "SEGMENT_ORDER",
    "SOURCE_CLASSES",
    "SOURCE_CLASS_TO_CODE",
    "ManoSurfaceRegionMap",
    "SourceContactThresholdContractV1",
    "build_mano_surface_region_map",
    "classify_source_contact",
    "largest_connected_component_size",
    "map_native_contact_to_control",
    "mesh_adjacency",
    "per_region_surface_statistics",
    "persistent_mask",
    "source_contact_localization",
]
