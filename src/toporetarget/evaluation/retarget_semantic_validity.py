"""Frame-authority invariants and semantic retarget qualification.

The public functions in this module are outcome-independent.  They use the
repository transform convention ``A_T_B``: a homogeneous transform mapping a
point expressed in frame B into frame A.  Numerical solver success is never an
input to the semantic gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from toporetarget.utils.hashing import sha256_file, sha256_tree


class FrameAuthority(StrEnum):
    RAW_WORLD = "RAW_WORLD"
    EPISODE_WORLD = "EPISODE_WORLD"
    CANONICAL_WORLD = "CANONICAL_WORLD"
    SOURCE_HAND = "SOURCE_HAND"
    SOURCE_OBJECT = "SOURCE_OBJECT"
    ROBOT_BASE = "ROBOT_BASE"
    ROBOT_WRIST = "ROBOT_WRIST"
    OBJECT_LOCAL = "OBJECT_LOCAL"
    VISUALIZATION_WORLD = "VISUALIZATION_WORLD"


class SemanticStatus(StrEnum):
    PASS = "RETARGET_SEMANTIC_PASS"
    FAIL = "RETARGET_SEMANTIC_FAIL"
    INCONCLUSIVE = "RETARGET_SEMANTIC_INCONCLUSIVE"


@dataclass(frozen=True)
class SemanticGateContractV1:
    """Loose fail-safe bounds frozen from invariants and positive controls."""

    schema_version: str = "RetargetSemanticValidityV1"
    calibration_authority: str = (
        "MATHEMATICAL_INVARIANTS_THEN_PHYSICAL_SEMANTICS_THEN_POSITIVE_CONTROLS"
    )
    positive_controls: str = "170105,170650"
    hardening_outcomes_used_to_tune_thresholds: bool = False
    unsupported_threshold_policy: str = "DIAGNOSTIC_ONLY"
    object_relative_wrist_position_limit_m: float = 0.01
    object_relative_wrist_rotation_limit_rad: float = np.deg2rad(10.0)
    bone_direction_p95_limit_rad: float = np.deg2rad(45.0)
    contact_opportunity_distance_m: float = 0.003
    source_contact_recall_minimum: float = 1e-12
    interaction_e_im_p95_limit: float = 1e-4
    temporal_translation_step_limit_m: float = 0.05
    temporal_rotation_step_limit_rad: float = np.deg2rad(90.0)
    rigid_invariant_atol_m: float = 1e-9
    rotation_invariant_atol_rad: float = 1e-9
    reflection_determinant_minimum: float = 0.999999
    unit_scale_ratio_minimum: float = 0.5
    unit_scale_ratio_maximum: float = 2.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def artifact_tree_sha256(path: Path) -> str:
    """Hash one file or directory tree using the semantic-audit convention."""

    resolved = path.resolve()
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise FileNotFoundError(f"RETARGET_SEMANTIC_ARTIFACT_MISSING:{resolved}")
    digest = hashlib.sha256()
    for name, value in sha256_tree(resolved).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def require_semantic_admission(
    receipt_path: Path,
    *,
    identifier: str,
    canonical: Path,
    final: Path,
) -> dict[str, str]:
    """Require a receipt-bound semantic PASS before downstream production use."""

    resolved_receipt = receipt_path.resolve()
    try:
        payload = json.loads(resolved_receipt.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"RETARGET_SEMANTIC_ADMISSION_RECEIPT_INVALID:{resolved_receipt}") from exc
    if not isinstance(payload, dict) or payload.get("identifier") != identifier:
        raise ValueError("RETARGET_SEMANTIC_ADMISSION_IDENTIFIER_MISMATCH")
    final_result = payload.get("final")
    qualification = final_result.get("qualification") if isinstance(final_result, dict) else None
    if (
        not isinstance(qualification, dict)
        or qualification.get("schema_version") != "RetargetSemanticValidityV1"
        or qualification.get("status") != SemanticStatus.PASS.value
    ):
        raise ValueError("RETARGET_SEMANTIC_ADMISSION_NONPASS")

    gate_path = resolved_receipt.with_name("semantic_gate_contract.json")
    gate_hash_path = resolved_receipt.with_name("semantic_gate_contract_sha256.txt")
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        recorded_gate_hash = gate_hash_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError("RETARGET_SEMANTIC_ADMISSION_GATE_RECEIPT_INVALID") from exc
    if not isinstance(gate, dict) or gate.get("schema_version") != "RetargetSemanticValidityV1":
        raise ValueError("RETARGET_SEMANTIC_ADMISSION_GATE_SCHEMA_INVALID")
    computed_gate_hash = hashlib.sha256(
        json.dumps(gate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        recorded_gate_hash != computed_gate_hash
        or qualification.get("gate_contract_sha256") != computed_gate_hash
    ):
        raise ValueError("RETARGET_SEMANTIC_ADMISSION_GATE_HASH_MISMATCH")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("RETARGET_SEMANTIC_ADMISSION_ARTIFACTS_MISSING")
    for name, expected in (("canonical", canonical.resolve()), ("final", final.resolve())):
        row = artifacts.get(name)
        if not isinstance(row, dict):
            raise ValueError(f"RETARGET_SEMANTIC_ADMISSION_ARTIFACT_MISSING:{name}")
        reported = Path(str(row.get("path", ""))).resolve()
        if reported != expected or row.get("sha256") != artifact_tree_sha256(expected):
            raise ValueError(f"RETARGET_SEMANTIC_ADMISSION_ARTIFACT_DRIFT:{name}")
    return {
        "path": str(resolved_receipt),
        "sha256": sha256_file(resolved_receipt),
        "gate_contract_path": str(gate_path),
        "gate_contract_sha256": computed_gate_hash,
        "status": SemanticStatus.PASS.value,
    }


def validate_transform(value: np.ndarray, *, name: str = "transform") -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must end in [4,4], got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} contains non-finite values")
    bottom = transform[..., 3, :]
    expected = np.broadcast_to(np.asarray([0.0, 0.0, 0.0, 1.0]), bottom.shape)
    if not np.allclose(bottom, expected, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    return transform


def invert_transform(value: np.ndarray) -> np.ndarray:
    transform = validate_transform(value)
    result = np.broadcast_to(np.eye(4), transform.shape).copy()
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    result[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    result[..., :3, 3] = -np.einsum("...ji,...j->...i", rotation, translation)
    return result


def compose(a_t_b: np.ndarray, b_t_c: np.ndarray) -> np.ndarray:
    return np.matmul(
        validate_transform(a_t_b, name="A_T_B"),
        validate_transform(b_t_c, name="B_T_C"),
    )


def relative_transform(world_t_reference: np.ndarray, world_t_target: np.ndarray) -> np.ndarray:
    """Return ``reference_T_target`` from two world-parent transforms."""

    return compose(invert_transform(world_t_reference), world_t_target)


def transform_points(a_t_b: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    transform = validate_transform(a_t_b)
    points = np.asarray(points_b, dtype=np.float64)
    return (
        np.einsum("...ij,...nj->...ni", transform[..., :3, :3], points)
        + transform[..., None, :3, 3]
    )


def rotation_distance_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    relative = np.matmul(np.swapaxes(left, -1, -2), right)
    return Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(relative.shape[:-2])


def transform_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, np.ndarray]:
    left = validate_transform(reference, name="reference")
    right = validate_transform(candidate, name="candidate")
    return {
        "position_m": np.linalg.norm(left[..., :3, 3] - right[..., :3, 3], axis=-1),
        "rotation_rad": rotation_distance_rad(left[..., :3, :3], right[..., :3, :3]),
    }


def common_rigid_transform_invariant(
    world_t_hand: np.ndarray, world_t_object: np.ndarray, new_world_t_world: np.ndarray
) -> dict[str, float | bool]:
    before = relative_transform(world_t_object, world_t_hand)
    after = relative_transform(
        compose(new_world_t_world, world_t_object), compose(new_world_t_world, world_t_hand)
    )
    error = transform_error(before, after)
    position = float(np.max(error["position_m"], initial=0.0))
    rotation = float(np.max(error["rotation_rad"], initial=0.0))
    return {
        "pass": position <= 1e-9 and rotation <= 1e-9,
        "max_position_error_m": position,
        "max_rotation_error_rad": rotation,
    }


def detect_transform_misuse(
    expected: np.ndarray,
    candidate: np.ndarray,
    *,
    gate: SemanticGateContractV1 | None = None,
) -> dict[str, bool | float]:
    frozen = gate or SemanticGateContractV1()
    expected_value = validate_transform(expected, name="expected")
    candidate_value = validate_transform(candidate, name="candidate")
    candidate_rotation = candidate_value[..., :3, :3]
    determinants = np.linalg.det(candidate_rotation)
    reflected = bool(np.any(determinants < frozen.reflection_determinant_minimum))
    if reflected:
        position_error = np.linalg.norm(
            expected_value[..., :3, 3] - candidate_value[..., :3, 3], axis=-1
        )
        rotation_error = np.full(position_error.shape, np.pi, dtype=np.float64)
        error = {"position_m": position_error, "rotation_rad": rotation_error}
    else:
        error = transform_error(expected_value, candidate_value)
    expected_steps = np.linalg.norm(np.diff(expected_value[..., :3, 3], axis=0), axis=-1)
    candidate_steps = np.linalg.norm(np.diff(candidate_value[..., :3, 3], axis=0), axis=-1)
    valid = expected_steps > 1e-9
    ratio = (
        np.divide(candidate_steps[valid], expected_steps[valid]) if np.any(valid) else np.ones(1)
    )
    return {
        "inverse_or_composition_error": bool(
            np.max(error["position_m"], initial=0.0) > frozen.rigid_invariant_atol_m
            or np.max(error["rotation_rad"], initial=0.0) > frozen.rotation_invariant_atol_rad
        ),
        "handedness_reflection": reflected,
        "unit_scale_mismatch": bool(
            np.any(ratio < frozen.unit_scale_ratio_minimum)
            or np.any(ratio > frozen.unit_scale_ratio_maximum)
        ),
        "max_position_error_m": float(np.max(error["position_m"], initial=0.0)),
        "max_rotation_error_rad": float(np.max(error["rotation_rad"], initial=0.0)),
    }


def angular_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    left = left / np.maximum(np.linalg.norm(left, axis=-1, keepdims=True), 1e-15)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-15)
    return np.arccos(np.clip(np.sum(left * right, axis=-1), -1.0, 1.0))


def temporal_steps(transforms: np.ndarray) -> dict[str, np.ndarray]:
    values = validate_transform(transforms)
    translation = np.zeros(values.shape[0], dtype=np.float64)
    rotation = np.zeros(values.shape[0], dtype=np.float64)
    if len(values) > 1:
        translation[1:] = np.linalg.norm(np.diff(values[:, :3, 3], axis=0), axis=-1)
        rotation[1:] = rotation_distance_rad(values[:-1, :3, :3], values[1:, :3, :3])
    return {"translation_m": translation, "rotation_rad": rotation}


def summarize(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("semantic metric is empty or non-finite")
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(np.max(data)),
    }


def qualify_semantics(
    *,
    wrist_position_m: np.ndarray,
    wrist_rotation_rad: np.ndarray,
    bone_error_rad: np.ndarray,
    source_contact: np.ndarray,
    robot_contact: np.ndarray,
    robot_wrist_transforms: np.ndarray,
    frame_authority_pass: bool,
    time_alignment_pass: bool,
    interaction_geometry_pass: bool,
    inconclusive_reasons: tuple[str, ...] = (),
    gate: SemanticGateContractV1 | None = None,
) -> dict[str, Any]:
    frozen = gate or SemanticGateContractV1()
    position = summarize(wrist_position_m)
    rotation = summarize(wrist_rotation_rad)
    bone = summarize(bone_error_rad)
    source = np.asarray(source_contact, dtype=bool)
    robot = np.asarray(robot_contact, dtype=bool)
    if source.shape != robot.shape:
        raise ValueError("source and robot contact masks must have identical shape")
    expected = int(np.count_nonzero(source))
    recall = float(np.count_nonzero(source & robot) / expected) if expected else None
    predicted = int(np.count_nonzero(robot))
    precision = float(np.count_nonzero(source & robot) / predicted) if predicted else None
    steps = temporal_steps(robot_wrist_transforms)
    temporal_pass = bool(
        np.max(steps["translation_m"], initial=0.0) <= frozen.temporal_translation_step_limit_m
        and np.max(steps["rotation_rad"], initial=0.0) <= frozen.temporal_rotation_step_limit_rad
    )
    gross_sanity_pass = bool(
        position["max"] <= frozen.object_relative_wrist_position_limit_m
        and rotation["max"] <= frozen.object_relative_wrist_rotation_limit_rad
        and bone["p95"] <= frozen.bone_direction_p95_limit_rad
    )
    contact_pass = bool(recall is None or recall >= frozen.source_contact_recall_minimum)
    required = (
        frame_authority_pass,
        time_alignment_pass,
        gross_sanity_pass,
        contact_pass,
        interaction_geometry_pass,
        temporal_pass,
    )
    status = (
        SemanticStatus.INCONCLUSIVE
        if inconclusive_reasons
        else SemanticStatus.PASS
        if all(required)
        else SemanticStatus.FAIL
    )
    return {
        "schema_version": frozen.schema_version,
        "status": status.value,
        "inconclusive_reasons": list(inconclusive_reasons),
        "frame_authority_pass": bool(frame_authority_pass),
        "time_alignment_pass": bool(time_alignment_pass),
        "gross_sanity_pass": gross_sanity_pass,
        "interaction_geometry_pass": bool(interaction_geometry_pass),
        "bone_direction_status": "PASS"
        if bone["p95"] <= frozen.bone_direction_p95_limit_rad
        else "FAIL",
        "contact_recall_status": "DIAGNOSTIC_ONLY"
        if recall is None
        else "PASS"
        if contact_pass
        else "FAIL",
        "contact_recall_gate_role": "GROSS_TOTAL_CONTACT_LOSS_FAIL_SAFE",
        "interaction_graph_status": "PASS" if interaction_geometry_pass else "FAIL",
        "temporal_continuity_status": "PASS" if temporal_pass else "FAIL",
        "manual_visualization_required": True,
        "metrics": {
            "object_relative_wrist_position_m": position,
            "object_relative_wrist_rotation_rad": rotation,
            "bone_direction_error_rad": bone,
            "source_contact_frames": expected,
            "robot_contact_frames": predicted,
            "source_contact_recall": recall,
            "source_contact_precision": precision,
            "temporal_translation_step_m": summarize(steps["translation_m"]),
            "temporal_rotation_step_rad": summarize(steps["rotation_rad"]),
        },
        "gate_contract_sha256": frozen.sha256,
    }


__all__ = [
    "FrameAuthority",
    "SemanticGateContractV1",
    "SemanticStatus",
    "angular_error",
    "artifact_tree_sha256",
    "common_rigid_transform_invariant",
    "compose",
    "detect_transform_misuse",
    "invert_transform",
    "qualify_semantics",
    "require_semantic_admission",
    "relative_transform",
    "rotation_distance_rad",
    "summarize",
    "temporal_steps",
    "transform_error",
    "transform_points",
    "validate_transform",
]
