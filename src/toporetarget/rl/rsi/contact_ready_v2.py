"""Contact-evidence RSI V2 state banks and deterministic safe-bank sampling.

The classifier intentionally consumes source contact labels, the frozen V2
reference, and retargeted geometry evidence.  It never reads the historic
three-centimetre V3 reward mask and has no clip-name branches.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from ..physical_stage import Stage16P1RSIAcceptanceContractV1


class RSIStateSemanticClass(str, Enum):
    PRE_CONTACT = "PRE_CONTACT"
    NEAR_CONTACT = "NEAR_CONTACT"
    CONTACT_READY = "CONTACT_READY"
    PERSISTENT_CONTACT = "PERSISTENT_CONTACT"
    MANIPULATION = "MANIPULATION"
    TERMINAL_HOLD = "TERMINAL_HOLD"
    AMBIGUOUS = "AMBIGUOUS"


class GravitySafetyLabel(str, Enum):
    GRAVITY_SAFE = "GRAVITY_SAFE"
    GRAVITY_RISK = "GRAVITY_RISK"
    INVALID_RESET = "INVALID_RESET"


SAFE_BANK_BY_CLASS: dict[RSIStateSemanticClass, str] = {
    RSIStateSemanticClass.NEAR_CONTACT: "NEAR_CONTACT_SAFE",
    RSIStateSemanticClass.CONTACT_READY: "CONTACT_READY_SAFE",
    RSIStateSemanticClass.PERSISTENT_CONTACT: "PERSISTENT_SAFE",
    RSIStateSemanticClass.MANIPULATION: "MANIPULATION_SAFE",
    RSIStateSemanticClass.TERMINAL_HOLD: "TERMINAL_SAFE",
}
INITIAL_P3_BANKS = ("CONTACT_READY_SAFE", "PERSISTENT_SAFE", "MANIPULATION_SAFE")
_SOURCE_CONTACT_CLASSES = {"SOURCE_CONTACT_CONFIRMED", "SOURCE_CONTACT_PERSISTENT"}
_SOURCE_NEAR_CLASSES = {"SOURCE_CONTACT_TRANSITION", "SOURCE_PROXIMITY_ONLY"}


@dataclass(frozen=True)
class ContactReadyRSIV2ContractV1:
    """Result-independent semantics for the generic P1 state-bank builder."""

    identifier: str = "Stage16ContactReadyRSIV2"
    source_contact_contract: str = "SourcePerFingerContactEvidenceV1"
    reference_kinematics: str = "Stage16DReferenceKinematicsV2"
    control_dt_s: float = 0.05
    near_contact_window_control_steps: int = 8
    terminal_hold_window_control_steps: int = 16
    contact_ready_window_control_steps: int = 8
    manipulation_linear_speed_mps: float = 0.03
    manipulation_angular_speed_radps: float = 0.5
    geometry_evidence: str = "retargeted_wuji_link_to_object_axis_proxy"
    forbidden_truth_source: str = "ReferenceGatedContactRewardV1_3cm_mask"

    def __post_init__(self) -> None:
        if self.identifier != "Stage16ContactReadyRSIV2":
            raise ValueError("CONTACT_READY_RSI_IDENTIFIER_INVALID")
        if self.source_contact_contract != "SourcePerFingerContactEvidenceV1":
            raise ValueError("CONTACT_READY_RSI_SOURCE_CONTACT_CONTRACT_INVALID")
        if self.reference_kinematics != "Stage16DReferenceKinematicsV2":
            raise ValueError("CONTACT_READY_RSI_REFERENCE_CONTRACT_INVALID")
        if self.control_dt_s != 0.05:
            raise ValueError("CONTACT_READY_RSI_CONTROL_DT_DRIFT")
        if (
            self.near_contact_window_control_steps < 1
            or self.terminal_hold_window_control_steps < 1
        ):
            raise ValueError("CONTACT_READY_RSI_WINDOW_INVALID")
        if self.contact_ready_window_control_steps < 1:
            raise ValueError("CONTACT_READY_RSI_READY_WINDOW_INVALID")
        if (
            self.manipulation_linear_speed_mps <= 0.0
            or self.manipulation_angular_speed_radps <= 0.0
        ):
            raise ValueError("CONTACT_READY_RSI_MANIPULATION_THRESHOLD_INVALID")
        if self.forbidden_truth_source != "ReferenceGatedContactRewardV1_3cm_mask":
            raise ValueError("CONTACT_READY_RSI_V3_MASK_POLICY_DRIFT")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _require_shape(value: np.ndarray, shape: tuple[int | None, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        raise ValueError(f"{name}_SHAPE_INVALID:{array.shape}")
    return array


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    result: list[tuple[int, int]] = []
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and values[stop]:
            stop += 1
        result.append((start, stop))
        start = stop
    return result


def _start_of_active_run(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    starts = np.zeros_like(values, dtype=bool)
    starts[0] = values[0]
    starts[1:] = values[1:] & ~values[:-1]
    return starts


def _next_expected_distance(expected: np.ndarray) -> np.ndarray:
    result = np.full(len(expected), np.iinfo(np.int64).max, dtype=np.int64)
    next_index: int | None = None
    for index in range(len(expected) - 1, -1, -1):
        if expected[index]:
            next_index = index
        if next_index is not None:
            result[index] = next_index - index
    return result


def _terminal_hold_mask(expected: np.ndarray, *, minimum_steps: int) -> np.ndarray:
    """Identify a final source-contact hold, without assuming a fixed index."""

    result = np.zeros_like(expected, dtype=bool)
    runs = _runs(expected)
    if not runs:
        return result
    start, stop = runs[-1]
    if stop != len(expected) or stop - start < minimum_steps:
        return result
    result[max(start, stop - minimum_steps) : stop] = True
    return result


def _json_evidence(
    *,
    source_classes: np.ndarray,
    expected_contact: bool,
    geometry_gap_m: float,
    motion_linear_mps: float,
    motion_angular_radps: float,
    semantic_class: RSIStateSemanticClass,
) -> str:
    return json.dumps(
        {
            "source_per_finger_class": [str(value) for value in source_classes],
            "source_expected_contact": expected_contact,
            "retargeted_geometry": {
                "kind": "retargeted_wuji_link_to_object_axis_proxy",
                "min_distance_m": geometry_gap_m,
                "not_reward_v3_mask": True,
            },
            "reference_motion": {
                "linear_speed_mps": motion_linear_mps,
                "angular_speed_radps": motion_angular_radps,
            },
            "semantic_class": semantic_class.value,
        },
        sort_keys=True,
    )


def classify_contact_ready_states(
    *,
    source_class_label: np.ndarray,
    source_expected_contact: np.ndarray,
    retargeted_geometry_gap_m: np.ndarray,
    reference_object_twist: np.ndarray,
    contract: ContactReadyRSIV2ContractV1 | None = None,
) -> dict[str, np.ndarray]:
    """Classify all reference states from source/geometry/motion evidence.

    Contact evidence is the primary truth.  The retargeted gap is recorded as
    geometry evidence and only admits a pre-onset frame to ``NEAR_CONTACT``
    when source labels already place it in the declared transition/proximity
    interval.  This prevents a distance-only mask from manufacturing contact.
    """

    frozen = contract or ContactReadyRSIV2ContractV1()
    labels = _require_shape(source_class_label, (None, 5), name="RSI_SOURCE_CLASS").astype("U32")
    expected = _require_shape(source_expected_contact, (len(labels), 5), name="RSI_EXPECTED")
    gaps = _require_shape(retargeted_geometry_gap_m, (len(labels),), name="RSI_GEOMETRY_GAP")
    twist = _require_shape(reference_object_twist, (len(labels), 6), name="RSI_OBJECT_TWIST")
    if not np.isfinite(gaps).all() or np.any(gaps < 0.0) or not np.isfinite(twist).all():
        raise ValueError("RSI_CLASSIFICATION_NONFINITE_INPUT")

    source_expected = np.asarray(np.asarray(expected, dtype=bool).any(axis=1), dtype=np.bool_)
    source_near = np.asarray(
        np.isin(labels, tuple(_SOURCE_NEAR_CLASSES)).any(axis=1), dtype=np.bool_
    )
    source_confirmed = np.asarray(
        np.isin(labels, tuple(_SOURCE_CONTACT_CLASSES)).any(axis=1), dtype=np.bool_
    )
    source_no_contact = np.asarray(np.all(labels == "SOURCE_NO_CONTACT", axis=1), dtype=np.bool_)
    next_expected = _next_expected_distance(source_expected)
    near = (
        ~source_expected
        & source_near
        & (next_expected > 0)
        & (next_expected <= frozen.near_contact_window_control_steps)
    )
    active_start = _start_of_active_run(source_expected)
    ready_window = np.zeros(len(labels), dtype=bool)
    for start, stop in _runs(source_expected):
        ready_window[start : min(stop, start + frozen.contact_ready_window_control_steps)] = True
    terminal_window = _terminal_hold_mask(
        source_expected, minimum_steps=frozen.terminal_hold_window_control_steps
    )
    linear_speed = np.linalg.vector_norm(twist[:, :3], axis=1)
    angular_speed = np.linalg.vector_norm(twist[:, 3:], axis=1)
    manipulation = source_expected & (
        (linear_speed >= frozen.manipulation_linear_speed_mps)
        | (angular_speed >= frozen.manipulation_angular_speed_radps)
    )
    # The final source-contact window is a terminal candidate, but a reference
    # state still undergoing material object motion remains manipulation. Tiny
    # terminal twist below the same frozen motion threshold remains a hold.
    terminal = terminal_window & ~manipulation

    semantic = np.full(len(labels), RSIStateSemanticClass.AMBIGUOUS.value, dtype="U24")
    semantic[source_no_contact & ~near] = RSIStateSemanticClass.PRE_CONTACT.value
    semantic[near] = RSIStateSemanticClass.NEAR_CONTACT.value
    semantic[source_expected] = RSIStateSemanticClass.PERSISTENT_CONTACT.value
    semantic[ready_window] = RSIStateSemanticClass.CONTACT_READY.value
    semantic[manipulation] = RSIStateSemanticClass.MANIPULATION.value
    semantic[terminal] = RSIStateSemanticClass.TERMINAL_HOLD.value
    # A claimed expected contact without a confirmed/persistent source label is
    # inconsistent input, never a state we are allowed to sample.
    semantic[source_expected & ~source_confirmed] = RSIStateSemanticClass.AMBIGUOUS.value

    confidence = np.full(len(labels), "LOW", dtype="U8")
    confidence[semantic == RSIStateSemanticClass.PRE_CONTACT.value] = "MEDIUM"
    confidence[semantic == RSIStateSemanticClass.NEAR_CONTACT.value] = "MEDIUM"
    confidence[
        np.isin(
            semantic,
            [
                RSIStateSemanticClass.CONTACT_READY.value,
                RSIStateSemanticClass.PERSISTENT_CONTACT.value,
                RSIStateSemanticClass.MANIPULATION.value,
                RSIStateSemanticClass.TERMINAL_HOLD.value,
            ],
        )
    ] = "HIGH"
    evidence = np.asarray(
        [
            _json_evidence(
                source_classes=labels[index],
                expected_contact=bool(source_expected[index]),
                geometry_gap_m=float(gaps[index]),
                motion_linear_mps=float(linear_speed[index]),
                motion_angular_radps=float(angular_speed[index]),
                semantic_class=RSIStateSemanticClass(semantic[index]),
            )
            for index in range(len(labels))
        ],
        dtype="U1024",
    )
    return {
        "semantic_class": semantic,
        "classification_confidence": confidence,
        "classification_evidence": evidence,
        "source_expected_contact": source_expected,
        "retargeted_geometry_gap_m": gaps.astype(np.float64),
        "reference_linear_speed_mps": linear_speed,
        "reference_angular_speed_radps": angular_speed,
        "source_contact_run_start": active_start,
    }


def _reference_state_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        position = np.asarray(archive["object_pose_translation_world_ref"], dtype=np.float64)
        quaternion = np.asarray(archive["object_pose_quaternion_world_ref_wxyz"], dtype=np.float64)
        twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
        links = np.asarray(archive["tracked_link_positions_world_ref"], dtype=np.float64)
        axes = np.asarray(archive["object_axis_points_world_ref"], dtype=np.float64)
    _require_shape(position, (None, 3), name="RSI_REFERENCE_OBJECT_POSITION")
    _require_shape(quaternion, (len(position), 4), name="RSI_REFERENCE_OBJECT_QUATERNION")
    _require_shape(twist, (len(position), 6), name="RSI_REFERENCE_OBJECT_TWIST")
    _require_shape(links, (len(position), None, 3), name="RSI_REFERENCE_LINKS")
    _require_shape(axes, (len(position), None, 3), name="RSI_REFERENCE_OBJECT_AXES")
    if not all(np.isfinite(value).all() for value in (position, quaternion, twist, links, axes)):
        raise ValueError("RSI_REFERENCE_NONFINITE")
    pose = np.concatenate((position, quaternion), axis=1)
    link_axis_delta = links[:, :, None, :] - axes[:, None, :, :]
    gap = np.linalg.vector_norm(link_axis_delta, axis=-1).min(axis=(1, 2))
    return pose, twist, gap


def build_contact_ready_state_bank(
    *,
    reference_path: Path,
    source_contact_evidence_path: Path,
    contract: ContactReadyRSIV2ContractV1 | None = None,
) -> dict[str, np.ndarray]:
    """Build a full 321-state P1 bank from frozen local evidence."""

    frozen = contract or ContactReadyRSIV2ContractV1()
    with np.load(source_contact_evidence_path, allow_pickle=False) as archive:
        labels = np.asarray(archive["class_label"])
        expected = np.asarray(archive["expected_contact"], dtype=bool)
        runtime_index = np.asarray(archive["control_index"], dtype=np.int64)
        native_index = np.asarray(archive["native_to_control_index"], dtype=np.int64)
    _require_shape(runtime_index, (len(labels),), name="RSI_RUNTIME_INDEX")
    if not np.array_equal(runtime_index, np.arange(len(labels), dtype=np.int64)):
        raise ValueError("RSI_RUNTIME_INDEX_NOT_DENSE")
    pose, twist, gap = _reference_state_arrays(reference_path)
    if len(pose) != len(labels):
        raise ValueError("RSI_SOURCE_REFERENCE_FRAME_COUNT_MISMATCH")
    result = classify_contact_ready_states(
        source_class_label=labels,
        source_expected_contact=expected,
        retargeted_geometry_gap_m=gap,
        reference_object_twist=twist,
        contract=frozen,
    )
    nearest_source = np.searchsorted(native_index, runtime_index, side="right") - 1
    nearest_source = np.clip(nearest_source, 0, len(native_index) - 1).astype(np.int64)
    return {
        "runtime_index": runtime_index,
        "source_index_or_interval": nearest_source,
        "semantic_class": result["semantic_class"],
        "source_expected_contact": result["source_expected_contact"],
        "reference_object_pose": pose,
        "reference_object_twist": twist,
        "classification_evidence": result["classification_evidence"],
        "classification_confidence": result["classification_confidence"],
        "retargeted_geometry_gap_m": result["retargeted_geometry_gap_m"],
        "reference_linear_speed_mps": result["reference_linear_speed_mps"],
        "reference_angular_speed_radps": result["reference_angular_speed_radps"],
        "contract_identifier": np.asarray(frozen.identifier),
    }


def save_state_bank(path: Path, bank: Mapping[str, np.ndarray]) -> None:
    """Write a non-pickled portable RSI bank."""

    required = {
        "runtime_index",
        "source_index_or_interval",
        "semantic_class",
        "source_expected_contact",
        "reference_object_pose",
        "reference_object_twist",
        "classification_evidence",
        "classification_confidence",
    }
    missing = sorted(required - set(bank))
    if missing:
        raise ValueError(f"RSI_STATE_BANK_FIELDS_MISSING:{missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_compressed: Any = np.savez_compressed
    savez_compressed(path, **bank)


def load_state_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        bank = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "runtime_index",
        "semantic_class",
        "source_expected_contact",
        "reference_object_pose",
    }
    if required - set(bank):
        raise ValueError("RSI_STATE_BANK_REQUIRED_FIELDS_MISSING")
    runtime = _require_shape(bank["runtime_index"], (None,), name="RSI_STATE_BANK_RUNTIME_INDEX")
    _require_shape(bank["semantic_class"], (len(runtime),), name="RSI_STATE_BANK_SEMANTIC")
    source_expected = bank["source_expected_contact"]
    _require_shape(source_expected, (len(runtime),), name="RSI_STATE_BANK_EXPECTED")
    _require_shape(
        bank["reference_object_pose"], (len(runtime), 7), name="RSI_STATE_BANK_OBJECT_POSE"
    )
    return bank


def _numeric_row_value(row: Mapping[str, object], key: str, *, default: float) -> float:
    """Read a JSON diagnostic scalar; malformed data fails toward risk."""

    value = row.get(key, default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return default


def _integer_row_value(row: Mapping[str, object], key: str, *, default: int) -> int:
    """Read an integer diagnostic counter; malformed data fails toward risk."""

    value = row.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def classify_gravity_diagnostic_row(
    row: Mapping[str, object], *, acceptance: Stage16P1RSIAcceptanceContractV1
) -> GravitySafetyLabel:
    """Map one real PhysX row to a frozen P1 safety label."""

    semantic = RSIStateSemanticClass(str(row["semantic_class"]))
    if semantic in {RSIStateSemanticClass.PRE_CONTACT, RSIStateSemanticClass.AMBIGUOUS}:
        return GravitySafetyLabel.INVALID_RESET
    nonfinite = bool(row.get("nonfinite", False))
    if (
        nonfinite
        or bool(row.get("joint_limit_failure", False))
        or bool(row.get("catastrophic_failure", False))
    ):
        return GravitySafetyLabel.INVALID_RESET
    if bool(row.get("object_drop", False)):
        return GravitySafetyLabel.GRAVITY_RISK
    linear = _numeric_row_value(row, "object_speed_before_contact_mps", default=np.inf)
    angular = _numeric_row_value(row, "object_angular_speed_before_contact_radps", default=np.inf)
    if (
        linear > acceptance.max_object_linear_speed_mps
        or angular > acceptance.max_object_angular_speed_radps
    ):
        return GravitySafetyLabel.GRAVITY_RISK
    displacement = abs(
        _numeric_row_value(row, "object_displacement_before_contact_m", default=np.inf)
    )
    downward = max(
        0.0,
        -_numeric_row_value(row, "object_vertical_displacement_before_contact_m", default=-np.inf),
    )
    if (
        displacement > acceptance.max_pre_contact_displacement_m
        or downward > acceptance.max_pre_contact_downward_displacement_m
    ):
        return GravitySafetyLabel.GRAVITY_RISK
    contact_persisted = _integer_row_value(row, "contact_persistence_control_steps", default=0)
    if (
        not bool(row.get("contact_achieved", False))
        or contact_persisted < acceptance.min_contact_persistence_control_steps
    ):
        return GravitySafetyLabel.GRAVITY_RISK
    return GravitySafetyLabel.GRAVITY_SAFE


def build_safe_bank(
    *,
    state_bank: Mapping[str, np.ndarray],
    diagnostic_rows: Sequence[Mapping[str, object]],
    acceptance: Stage16P1RSIAcceptanceContractV1,
) -> dict[str, np.ndarray]:
    """Admit only states whose every replica passed the pre-frozen P1 gate."""

    states = np.asarray(state_bank["runtime_index"], dtype=np.int64)
    semantics = np.asarray(state_bank["semantic_class"]).astype("U24")
    rows_by_state: dict[int, list[Mapping[str, object]]] = {int(index): [] for index in states}
    for row in diagnostic_rows:
        runtime_index = row["runtime_index"]
        if not isinstance(runtime_index, int) or isinstance(runtime_index, bool):
            raise ValueError("RSI_DIAGNOSTIC_RUNTIME_INDEX_INVALID")
        index = runtime_index
        if index not in rows_by_state:
            raise ValueError(f"RSI_DIAGNOSTIC_UNKNOWN_RUNTIME_INDEX:{index}")
        rows_by_state[index].append(row)
    selected_indices: list[int] = []
    selected_classes: list[str] = []
    selected_banks: list[str] = []
    state_labels: list[str] = []
    for position, index in enumerate(states):
        rows = rows_by_state[int(index)]
        semantic = RSIStateSemanticClass(semantics[position])
        # P1 deliberately does not run PRE_CONTACT or AMBIGUOUS states: they
        # are forbidden reset classes rather than missing measurements.  Keep
        # them in the full bank for auditability while making their exclusion
        # explicit in the same all-state gravity-label vector.
        if semantic in {RSIStateSemanticClass.PRE_CONTACT, RSIStateSemanticClass.AMBIGUOUS}:
            state_labels.append(GravitySafetyLabel.INVALID_RESET.value)
            continue
        if len(rows) != acceptance.replicas_per_state:
            raise ValueError(f"RSI_DIAGNOSTIC_REPLICA_COUNT_INVALID:{index}:{len(rows)}")
        labels = [classify_gravity_diagnostic_row(row, acceptance=acceptance) for row in rows]
        if all(label is GravitySafetyLabel.GRAVITY_SAFE for label in labels):
            state_label = GravitySafetyLabel.GRAVITY_SAFE
        elif any(label is GravitySafetyLabel.INVALID_RESET for label in labels):
            state_label = GravitySafetyLabel.INVALID_RESET
        else:
            state_label = GravitySafetyLabel.GRAVITY_RISK
        state_labels.append(state_label.value)
        safe_bank = SAFE_BANK_BY_CLASS.get(semantic)
        if state_label is GravitySafetyLabel.GRAVITY_SAFE and safe_bank is not None:
            selected_indices.append(int(index))
            selected_classes.append(semantic.value)
            selected_banks.append(safe_bank)
    return {
        "runtime_index": np.asarray(selected_indices, dtype=np.int64),
        "semantic_class": np.asarray(selected_classes, dtype="U24"),
        "safe_bank": np.asarray(selected_banks, dtype="U24"),
        "all_runtime_index": states,
        "all_gravity_label": np.asarray(state_labels, dtype="U16"),
        "contract_identifier": np.asarray("ContactReadySafeBankV2"),
    }


def save_safe_bank(path: Path, bank: Mapping[str, np.ndarray]) -> None:
    required = {
        "runtime_index",
        "semantic_class",
        "safe_bank",
        "all_runtime_index",
        "all_gravity_label",
    }
    if required - set(bank):
        raise ValueError("RSI_SAFE_BANK_REQUIRED_FIELDS_MISSING")
    path.parent.mkdir(parents=True, exist_ok=True)
    savez_compressed: Any = np.savez_compressed
    savez_compressed(path, **bank)


def load_safe_bank(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"runtime_index", "semantic_class", "safe_bank"}
    if required - set(result):
        raise ValueError("RSI_SAFE_BANK_REQUIRED_FIELDS_MISSING")
    runtime = _require_shape(result["runtime_index"], (None,), name="RSI_SAFE_BANK_RUNTIME")
    _require_shape(result["semantic_class"], (len(runtime),), name="RSI_SAFE_BANK_SEMANTIC")
    _require_shape(result["safe_bank"], (len(runtime),), name="RSI_SAFE_BANK_NAME")
    if not np.all(np.isin(result["safe_bank"], tuple(SAFE_BANK_BY_CLASS.values()))):
        raise ValueError("RSI_SAFE_BANK_UNKNOWN_NAME")
    return result


@dataclass(frozen=True)
class ContactReadySamplerV2:
    """Deterministic generic sampler; only P1-qualified named banks are usable."""

    runtime_index: np.ndarray
    safe_bank: np.ndarray

    @classmethod
    def from_safe_bank(cls, path: Path) -> ContactReadySamplerV2:
        bank = load_safe_bank(path)
        return cls(
            runtime_index=np.asarray(bank["runtime_index"], dtype=np.int64),
            safe_bank=np.asarray(bank["safe_bank"]).astype("U24"),
        )

    def indices(
        self,
        allowed_banks: Sequence[str] = INITIAL_P3_BANKS,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        allowed = tuple(str(item) for item in allowed_banks)
        unknown = sorted(set(allowed) - set(SAFE_BANK_BY_CLASS.values()))
        if unknown:
            raise ValueError(f"RSI_SAMPLER_UNKNOWN_BANK:{unknown}")
        keep = np.isin(self.safe_bank, allowed)
        indices = self.runtime_index[keep]
        names = self.safe_bank[keep]
        if not len(indices):
            raise ValueError("RSI_SAMPLER_NO_ALLOWED_SAFE_STATES")
        if weights is None:
            probability = np.full(len(indices), 1.0 / len(indices), dtype=np.float64)
        else:
            requested = {str(key): float(value) for key, value in weights.items()}
            if any(not np.isfinite(value) or value < 0.0 for value in requested.values()):
                raise ValueError("RSI_SAMPLER_WEIGHT_INVALID")
            probability = np.asarray([requested.get(name, 0.0) for name in names], dtype=np.float64)
            if probability.sum() <= 0.0:
                raise ValueError("RSI_SAMPLER_WEIGHT_ZERO_MASS")
            probability /= probability.sum()
        return indices, probability

    def sample(
        self,
        rng: np.random.Generator,
        *,
        count: int,
        allowed_banks: Sequence[str] = INITIAL_P3_BANKS,
        weights: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        if count < 1:
            raise ValueError("RSI_SAMPLER_COUNT_INVALID")
        indices, probability = self.indices(allowed_banks, weights=weights)
        return rng.choice(indices, size=count, replace=True, p=probability).astype(np.int64)


def summarize_state_bank(bank: Mapping[str, np.ndarray]) -> dict[str, object]:
    classes = [str(item) for item in np.asarray(bank["semantic_class"])]
    counts = Counter(classes)
    return {
        "schema_version": "Stage16ContactReadyRSIV2StateBankSummaryV1",
        "state_count": len(classes),
        "semantic_class_counts": {
            member.value: int(counts[member.value]) for member in RSIStateSemanticClass
        },
    }


__all__ = [
    "ContactReadyRSIV2ContractV1",
    "ContactReadySamplerV2",
    "GravitySafetyLabel",
    "INITIAL_P3_BANKS",
    "RSIStateSemanticClass",
    "SAFE_BANK_BY_CLASS",
    "build_contact_ready_state_bank",
    "build_safe_bank",
    "classify_contact_ready_states",
    "classify_gravity_diagnostic_row",
    "load_safe_bank",
    "load_state_bank",
    "save_safe_bank",
    "save_state_bank",
    "summarize_state_bank",
]
