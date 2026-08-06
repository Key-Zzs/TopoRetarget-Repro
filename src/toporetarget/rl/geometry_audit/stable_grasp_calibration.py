"""Contracts for clip-agnostic free-object stable-grasp calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CALIBRATION_GROUP_ORDER = ("thumb", "index", "middle", "ring", "pinky", "palm")
GROUP_FLEXION_ACTION_INDICES = {
    "thumb": (0, 2, 3),
    "index": (4, 6, 7),
    "middle": (8, 10, 11),
    "ring": (12, 14, 15),
    "pinky": (16, 18, 19),
    "palm": (),
}


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GraspTopologyFamilyV1:
    """A data-derived opposition topology shared by every calibration object."""

    identifier: str
    contact_groups: tuple[str, ...]
    first_opposition_group: str
    second_opposition_group: str
    applicable_clips: tuple[str, ...]
    source_required_groups: tuple[str, ...]
    derivation: str = "TaskSemanticContractV1+ContactTopologyContract+collision body groups"
    schema_version: str = "GraspTopologyFamilyV1"

    def __post_init__(self) -> None:
        if len(self.contact_groups) < 2 or len(set(self.contact_groups)) != len(
            self.contact_groups
        ):
            raise ValueError("stable grasp topology requires at least two unique groups")
        if self.first_opposition_group == self.second_opposition_group:
            raise ValueError("opposition groups must differ")
        if not {self.first_opposition_group, self.second_opposition_group}.issubset(
            self.contact_groups
        ):
            raise ValueError("opposition groups must belong to contact_groups")
        if any(group not in CALIBRATION_GROUP_ORDER for group in self.contact_groups):
            raise ValueError("unknown calibration contact group")
        if not set(self.source_required_groups).issubset(self.contact_groups):
            raise ValueError("calibration family must cover every source-required group")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_grasp_topology_families(
    topology_contract: Mapping[str, Any],
    *,
    available_groups: Sequence[str] = CALIBRATION_GROUP_ORDER,
) -> tuple[GraspTopologyFamilyV1, ...]:
    """Derive opposition families with one algorithm and no clip-name conditions."""

    clips = topology_contract.get("clips")
    if not isinstance(clips, Mapping) or not clips:
        raise ValueError("contact topology contract must contain a nonempty clips mapping")
    available = set(available_groups)
    if not available.issubset(CALIBRATION_GROUP_ORDER):
        raise ValueError("available groups contain an unknown calibration group")
    rows: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    non_thumb_order = ("index", "middle", "ring", "pinky")
    for clip, raw in sorted(clips.items()):
        if not isinstance(raw, Mapping):
            raise ValueError("clip topology row must be a mapping")
        required = tuple(
            group for group in CALIBRATION_GROUP_ORDER if group in raw["required_body_groups"]
        )
        optional = set(raw.get("optional_body_groups", ()))
        if not required or not set(required).issubset(available):
            raise ValueError(f"required contact groups unavailable for {clip}")
        non_thumb = tuple(group for group in non_thumb_order if group in required)
        if "palm" in required:
            opposing = next((group for group in non_thumb_order if group in required), None)
            if opposing is None:
                raise ValueError("palm topology needs at least one finger group")
            identifier = "palm_finger_enclosure"
            contact_groups = tuple(dict.fromkeys((*required, opposing)))
            first, second = "palm", opposing
        elif "thumb" in required and non_thumb:
            identifier = "thumb_opposition"
            contact_groups = required
            first, second = "thumb", non_thumb[0]
        elif len(non_thumb) >= 2:
            # Anatomically adjacent non-thumb fingers are not an opposition
            # pair.  Add the available thumb as the common opposing surface
            # so every source-required finger participates in a mechanically
            # enclosing topology.  Palm is a generic fallback only on hands
            # without a usable thumb collision group.
            opposition = "thumb" if "thumb" in available else "palm"
            if opposition not in available:
                raise ValueError("multi-finger topology needs thumb or palm opposition")
            identifier = "multi_finger_enclosure"
            contact_groups = tuple(
                group for group in CALIBRATION_GROUP_ORDER if group in set(required) | {opposition}
            )
            first, second = opposition, non_thumb[-1]
        else:
            opposing = non_thumb[0] if non_thumb else required[0]
            if "thumb" not in available:
                raise ValueError("single-finger topology needs the available thumb opposition")
            identifier = "thumb_opposition"
            contact_groups = tuple(
                group for group in CALIBRATION_GROUP_ORDER if group in set(required) | {"thumb"}
            )
            first, second = "thumb", opposing
            if "thumb" not in optional and "thumb" not in required:
                derivation_note = "generic opposition added because source has one required finger"
            else:
                derivation_note = "source optional thumb supplies opposition"
            raw = {**raw, "calibration_opposition_note": derivation_note}
        key = (identifier, tuple(contact_groups))
        entry = rows.setdefault(
            key,
            {
                "identifier": identifier,
                "contact_groups": tuple(contact_groups),
                "first": first,
                "second": second,
                "clips": [],
                "required": set(),
            },
        )
        entry["clips"].append(str(clip))
        entry["required"].update(required)
    families = tuple(
        GraspTopologyFamilyV1(
            identifier=row["identifier"],
            contact_groups=row["contact_groups"],
            first_opposition_group=row["first"],
            second_opposition_group=row["second"],
            applicable_clips=tuple(sorted(row["clips"])),
            source_required_groups=tuple(
                group for group in CALIBRATION_GROUP_ORDER if group in row["required"]
            ),
        )
        for _, row in sorted(rows.items())
    )
    covered = {
        clip
        for family in families
        for clip in family.applicable_clips
        if set(family.source_required_groups).issubset(family.contact_groups)
    }
    if covered != set(clips):
        raise RuntimeError("STAGE16D_GRASP_TOPOLOGY_FAMILY_COVERAGE_FAILURE")
    return families


@dataclass(frozen=True)
class StableGraspCandidateV1:
    object_id: str
    family_id: str
    contact_groups: tuple[str, ...]
    first_opposition_group: str
    second_opposition_group: str
    approach_offset_m: float
    closure_amplitude: float
    level: str
    candidate_id: str
    schema_version: str = "StableGraspCandidateV1"

    def __post_init__(self) -> None:
        if self.level not in {"C1", "C2"}:
            raise ValueError("calibration level must be C1 or C2")
        if not -0.015 <= self.approach_offset_m <= 0.015:
            raise ValueError("calibration approach offset is outside the frozen bound")
        if not 0.0 < self.closure_amplitude <= 1.0:
            raise ValueError("closure amplitude must be in (0,1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def freeze_candidate_matrix(
    *,
    object_ids: Sequence[str],
    families: Sequence[GraspTopologyFamilyV1],
    level: str,
) -> dict[str, Any]:
    """Freeze C1/C2 candidates before any calibration result is observed."""

    offsets: tuple[float, ...]
    closures: tuple[float, ...]
    if level == "C1":
        offsets = (-0.006, 0.0, 0.006)
        closures = (0.5, 1.0)
        limit = 24
    elif level == "C2":
        # C2 contains only candidates that were not already evaluated in C1.
        # The sole expanded dimension is approach offset; closure is unchanged.
        offsets = (-0.010, 0.010)
        # The bounded C2 upgrade expands exactly one dimension.  Keeping the
        # C1 closure grid unchanged prevents an accidental offset+closure
        # combinatorial search after observing C1 outcomes.
        closures = (0.5, 1.0)
        limit = 48
    else:
        raise ValueError("calibration level must be C1 or C2")
    rows: dict[str, list[dict[str, Any]]] = {}
    for object_id in sorted(set(object_ids)):
        applicable = [family for family in families if object_id in family.applicable_clips]
        if not applicable:
            raise ValueError(f"no data-derived grasp topology applies to {object_id}")
        object_rows: list[dict[str, Any]] = []
        for family in applicable:
            for offset in offsets:
                for closure in closures:
                    identity = {
                        "object_id": object_id,
                        "family_id": family.identifier,
                        "contact_groups": list(family.contact_groups),
                        "first_opposition_group": family.first_opposition_group,
                        "second_opposition_group": family.second_opposition_group,
                        "approach_offset_m": offset,
                        "closure_amplitude": closure,
                        "level": level,
                    }
                    candidate_id = f"{level.lower()}_{_stable_hash(identity)[:12]}"
                    candidate = StableGraspCandidateV1(
                        object_id=object_id,
                        family_id=family.identifier,
                        contact_groups=family.contact_groups,
                        first_opposition_group=family.first_opposition_group,
                        second_opposition_group=family.second_opposition_group,
                        approach_offset_m=offset,
                        closure_amplitude=closure,
                        level=level,
                        candidate_id=candidate_id,
                    )
                    object_rows.append(candidate.as_dict())
        if len(object_rows) > limit:
            raise RuntimeError(f"STAGE16D_CALIBRATION_CANDIDATE_BUDGET_EXHAUSTED:{object_id}")
        rows[object_id] = object_rows
    payload = {
        "schema_version": "StableGraspCandidateMatrixV1",
        "level": level,
        "candidate_limit_per_object": limit,
        "approach_offsets_m": list(offsets),
        "closure_amplitudes": list(closures),
        "parent_level": "C1" if level == "C2" else None,
        "expanded_dimension": "approach_offset_m" if level == "C2" else None,
        "candidates_are_unique_from_parent": level == "C2",
        "same_generation_rule_for_all_objects": True,
        "result_data_observed_before_freeze": False,
        "objects": rows,
    }
    return {**payload, "contract_sha256": _stable_hash(payload)}


@dataclass(frozen=True)
class StableGraspCalibrationActionScheduleV1:
    """Shared 321-step open/approach/closure/hold action schedule."""

    open_steps: int = 24
    approach_steps: int = 48
    contact_establishment_steps: int = 32
    closure_steps: int = 64
    settle_steps: int = 53
    terminal_hold_steps: int = 100
    wrist_approach_action_amplitude: float = 1.0
    hold_closure_fraction: float = 0.60
    schema_version: str = "StableGraspCalibrationActionScheduleV1"

    def __post_init__(self) -> None:
        if sum(self.phase_lengths) != 321:
            raise ValueError("stable-grasp calibration schedule must have exactly 321 steps")
        if not 0.0 <= self.wrist_approach_action_amplitude <= 1.0:
            raise ValueError("wrist approach action amplitude must be in [0,1]")
        if not 0.0 <= self.hold_closure_fraction <= 1.0:
            raise ValueError("hold closure fraction must be in [0,1]")

    @property
    def phase_lengths(self) -> tuple[int, ...]:
        return (
            self.open_steps,
            self.approach_steps,
            self.contact_establishment_steps,
            self.closure_steps,
            self.settle_steps,
            self.terminal_hold_steps,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "phase_order": [
                "open",
                "approach",
                "contact_establishment",
                "closure",
                "settle",
                "terminal_hold",
            ],
            "action_dimension": 26,
            "action_bounds": [-1.0, 1.0],
            "object_action_fields": 0,
            "shared_across_objects": True,
            "unselected_flexion_during_all_phases": -1.0,
        }

    def actions(
        self,
        *,
        contact_groups: Sequence[str],
        closure_amplitude: float,
        wrist_approach_direction_local: np.ndarray,
    ) -> np.ndarray:
        direction = np.asarray(wrist_approach_direction_local, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("wrist approach direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        direction = np.zeros(3) if norm <= 1.0e-10 else direction / norm
        closure = float(closure_amplitude)
        if not 0.0 < closure <= 1.0:
            raise ValueError("closure amplitude must be in (0,1]")
        finger_indices = sorted(
            {index for group in contact_groups for index in GROUP_FLEXION_ACTION_INDICES[group]}
        )
        opening_indices = sorted(
            {
                index
                for group in ("thumb", "index", "middle", "ring", "pinky")
                for index in GROUP_FLEXION_ACTION_INDICES[group]
            }
        )
        rows: list[np.ndarray] = []

        def append_phase(count: int, wrist: np.ndarray, finger: float) -> None:
            for _ in range(count):
                action = np.zeros(26, dtype=np.float32)
                action[:3] = wrist.astype(np.float32)
                action[6 + np.asarray(opening_indices, dtype=np.int64)] = -1.0
                action[6 + np.asarray(finger_indices, dtype=np.int64)] = np.float32(finger)
                rows.append(action)

        append_phase(self.open_steps, np.zeros(3), -1.0)
        for alpha in np.linspace(0.0, 1.0, self.approach_steps, endpoint=True):
            append_phase(
                1,
                direction * self.wrist_approach_action_amplitude * float(alpha),
                -1.0,
            )
        for alpha in np.linspace(0.0, 1.0, self.contact_establishment_steps, endpoint=True):
            append_phase(
                1,
                direction * self.wrist_approach_action_amplitude,
                -1.0 + 0.5 * closure * float(alpha),
            )
        for alpha in np.linspace(0.0, 1.0, self.closure_steps, endpoint=True):
            append_phase(
                1,
                direction * self.wrist_approach_action_amplitude,
                -1.0 + 0.5 * closure + 0.5 * closure * float(alpha),
            )
        peak_closure = -1.0 + closure
        hold_closure = -1.0 + closure * self.hold_closure_fraction
        for finger in np.linspace(peak_closure, hold_closure, self.settle_steps, endpoint=True):
            append_phase(
                1,
                direction * self.wrist_approach_action_amplitude,
                float(finger),
            )
        append_phase(
            self.terminal_hold_steps,
            direction * self.wrist_approach_action_amplitude,
            hold_closure,
        )
        result = np.stack(rows)
        if result.shape != (321, 26) or np.any(np.abs(result) > 1.0):
            raise RuntimeError("STAGE16D_CALIBRATION_ACTION_CONTRACT_FAILURE")
        return result


@dataclass(frozen=True)
class StableGraspCalibrationGateV1:
    replicas: int = 20
    development_replicas: int = 4
    steps: int = 321
    topology_coverage_min: float = 0.95
    terminal_hold_coverage_min: float = 0.95
    maximum_consecutive_contact_loss_steps: int = 5
    final_window_steps: int = 100
    final_linear_speed_p95_max_mps: float = 0.01
    final_angular_speed_p95_max_radps: float = 0.10
    strict_catastrophic_max_m: float = 0.010
    active_p95_max_m: float = 0.003
    schema_version: str = "StableGraspCalibrationGateV1"

    def __post_init__(self) -> None:
        if self.replicas != 20 or self.steps != 321 or self.final_window_steps != 100:
            raise ValueError("formal stable-grasp calibration is frozen at 20x321, final-100")
        if self.strict_catastrophic_max_m != 0.010 or self.active_p95_max_m != 0.003:
            raise ValueError("stable-grasp calibration cannot change the 10mm/3mm gate")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maximum_false_run(values: np.ndarray) -> int:
    maximum = current = 0
    for value in np.asarray(values, dtype=bool):
        current = 0 if value else current + 1
        maximum = max(maximum, current)
    return maximum


def qualify_stable_grasp(
    *,
    contact_group_presence: np.ndarray,
    object_twist: np.ndarray,
    finite: np.ndarray,
    action_bounds_pass: np.ndarray,
    workspace_pass: np.ndarray,
    exact_replica_max_penetration_m: np.ndarray,
    exact_replica_active_p95_m: np.ndarray,
    gate: StableGraspCalibrationGateV1 = StableGraspCalibrationGateV1(),
) -> dict[str, Any]:
    """Apply the frozen 20-replica stable-grasp hard gates."""

    presence = np.asarray(contact_group_presence, dtype=bool)
    twist = np.asarray(object_twist, dtype=np.float64)
    expected = (gate.steps, gate.replicas)
    if presence.ndim != 3 or presence.shape[:2] != expected or presence.shape[2] < 2:
        raise ValueError("contact group presence must have shape [321,20,G>=2]")
    if twist.shape != (*expected, 6):
        raise ValueError("object twist must have shape [321,20,6]")
    vectors = {
        "finite": np.asarray(finite, dtype=bool),
        "action_bounds": np.asarray(action_bounds_pass, dtype=bool),
        "workspace": np.asarray(workspace_pass, dtype=bool),
        "max": np.asarray(exact_replica_max_penetration_m, dtype=np.float64),
        "p95": np.asarray(exact_replica_active_p95_m, dtype=np.float64),
    }
    if any(values.shape != (gate.replicas,) for values in vectors.values()):
        raise ValueError("replica qualification vectors must have shape [20]")
    any_contact = presence.any(axis=-1)
    group_ever = presence.any(axis=0)
    topology_replica_pass = group_ever.all(axis=-1)
    final_presence = presence[-gate.final_window_steps :].all(axis=-1)
    final_hold_replica_pass = np.asarray(
        [
            _maximum_false_run(final_presence[:, replica])
            <= gate.maximum_consecutive_contact_loss_steps
            for replica in range(gate.replicas)
        ],
        dtype=bool,
    )
    linear = np.linalg.norm(twist[-gate.final_window_steps :, :, :3], axis=-1)
    angular = np.linalg.norm(twist[-gate.final_window_steps :, :, 3:], axis=-1)
    linear_p95 = np.quantile(linear, 0.95, axis=0)
    angular_p95 = np.quantile(angular, 0.95, axis=0)
    twist_pass = (linear_p95 <= gate.final_linear_speed_p95_max_mps) & (
        angular_p95 <= gate.final_angular_speed_p95_max_radps
    )
    absolute_geometry_pass = (vectors["max"] < gate.strict_catastrophic_max_m) & (
        vectors["p95"] <= gate.active_p95_max_m
    )
    hard = {
        "finite_20_of_20": bool(vectors["finite"].all()),
        "no_hidden_control_20_of_20": True,
        "action_bounds_20_of_20": bool(vectors["action_bounds"].all()),
        "real_contact_20_of_20": bool(any_contact.any(axis=0).all()),
        "topology_coverage": float(topology_replica_pass.mean()) >= gate.topology_coverage_min,
        "terminal_hold_coverage": float(final_hold_replica_pass.mean())
        >= gate.terminal_hold_coverage_min,
        "maximum_contact_loss_20_of_20": bool(final_hold_replica_pass.all()),
        "workspace_20_of_20": bool(vectors["workspace"].all()),
        "absolute_geometry_20_of_20": bool(absolute_geometry_pass.all()),
        "terminal_twist_20_of_20": bool(twist_pass.all()),
    }
    passed = all(hard.values())
    return {
        "schema_version": "StableGraspCalibrationQualificationV1",
        "status": (
            "STAGE16D_STABLE_GRASP_CALIBRATION_VALIDATED"
            if passed
            else "STAGE16D_STABLE_GRASP_CALIBRATION_PARTIAL"
        ),
        "passed": passed,
        "hard_gates": hard,
        "topology_coverage": float(topology_replica_pass.mean()),
        "terminal_hold_coverage": float(final_hold_replica_pass.mean()),
        "terminal_linear_speed_p95_max_mps": float(linear_p95.max()),
        "terminal_angular_speed_p95_max_radps": float(angular_p95.max()),
        "replica_max_penetration_m": vectors["max"].tolist(),
        "replica_active_p95_penetration_m": vectors["p95"].tolist(),
        "gate": gate.as_dict(),
    }


__all__ = [
    "CALIBRATION_GROUP_ORDER",
    "GROUP_FLEXION_ACTION_INDICES",
    "GraspTopologyFamilyV1",
    "StableGraspCalibrationActionScheduleV1",
    "StableGraspCalibrationGateV1",
    "StableGraspCandidateV1",
    "extract_grasp_topology_families",
    "freeze_candidate_matrix",
    "qualify_stable_grasp",
]
