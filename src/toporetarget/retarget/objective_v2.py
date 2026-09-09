"""Interaction-first retarget objective contracts and development adapters.

This module adds a new, explicitly versioned objective contract.  It does not
alter :mod:`toporetarget.retarget.final_refinement` or the historical V1 scalar
objective.  The optimizer adapter is deliberately marked development-only;
independent certification must freeze a separate production-search contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np

from toporetarget.evaluation.retarget_semantic_validity import SemanticGateContractV1
from toporetarget.retarget.continuous import S_POS_M, S_Q_RAD, S_ROT_RAD

OBJECTIVE_V2_SCHEMA_VERSION = "RetargetObjectiveV2"
CONTEXT_BINDING_V2_SCHEMA_VERSION = "ProductionObjectiveContextBindingV2"
NONREGRESSION_AUTHORITY_SCHEMA_VERSION = "RetargetNonRegressionBudgetAuthorityV1"
DEVELOPMENT_OPTIMIZER_SCHEMA_VERSION = "ObjectiveV2DevelopmentOptimizerV1"
NUMERICAL_RETENTION_EPSILON = 1.0e-10

CandidateName = Literal[
    "A_INTERACTION_CONSTRAINED_FIDELITY",
    "B_LEXICOGRAPHIC_THRESHOLD",
]
PhaseName = Literal["primary", "secondary"]


def canonical_sha256(value: Any) -> str:
    """Return the canonical JSON SHA256 used by frozen V2 contracts."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def interaction_hinge(interaction_e_im: float, target: float) -> float:
    """Squared, dimensionless Semantic-V1 interaction hinge."""

    if target <= 0.0:
        raise ValueError("interaction target must be positive")
    excess = max(float(interaction_e_im) - float(target), 0.0) / float(target)
    return excess * excess


@dataclass(frozen=True)
class ProductionObjectiveContextBindingV2:
    """Complete frame/runtime binding shared by all ObjectiveV2 operations."""

    active_frame_id: int
    local_ordinal: int
    current_source_frame_id: int
    previous_source_frame_id: int | None
    previous_runtime_base_scene: np.ndarray | None
    previous_robot_qpos: np.ndarray | None
    continuous_predicted_translation_scene: np.ndarray | None
    continuous_predicted_rotation_scene: np.ndarray | None
    continuous_predicted_qpos: np.ndarray | None
    base_correction_reference: np.ndarray | None
    object_pose_scene: np.ndarray
    object_id: str
    robot_name: str
    robot_side: str
    robot_dof_names: tuple[str, ...]
    robot_mapping_authority: str
    source_hand_id: str
    temporal_scope: str = "continuous_full_state"
    schema_version: str = CONTEXT_BINDING_V2_SCHEMA_VERSION

    def validate(self) -> ProductionObjectiveContextBindingV2:
        if self.local_ordinal < 0 or self.active_frame_id != self.current_source_frame_id:
            raise ValueError("active/current frame binding is inconsistent")
        if np.asarray(self.object_pose_scene).shape != (4, 4):
            raise ValueError("object context must contain one homogeneous pose")
        if not self.robot_name or not self.robot_side or not self.robot_dof_names:
            raise ValueError("robot mapping authority is incomplete")
        previous_values = (
            self.previous_source_frame_id,
            self.previous_runtime_base_scene,
            self.previous_robot_qpos,
            self.continuous_predicted_translation_scene,
            self.continuous_predicted_rotation_scene,
            self.continuous_predicted_qpos,
            self.base_correction_reference,
        )
        if self.local_ordinal == 0:
            if any(item is not None for item in previous_values):
                raise ValueError("frame zero must preserve the V1 no-temporal branch")
        elif any(item is None for item in previous_values):
            raise ValueError("nonzero frame omitted a production continuous input")
        if self.continuous_predicted_translation_scene is not None:
            if np.asarray(self.continuous_predicted_translation_scene).shape != (3,):
                raise ValueError("predicted translation must have shape (3,)")
            if np.asarray(self.continuous_predicted_rotation_scene).shape != (3, 3):
                raise ValueError("predicted rotation must have shape (3,3)")
            if np.asarray(self.continuous_predicted_qpos).shape != (len(self.robot_dof_names),):
                raise ValueError("predicted q does not match robot mapping")
        return self

    @property
    def continuous_prediction_base(self) -> np.ndarray | None:
        if self.continuous_predicted_translation_scene is None:
            return None
        value = np.eye(4, dtype=np.float64)
        value[:3, :3] = np.asarray(self.continuous_predicted_rotation_scene, dtype=np.float64)
        value[:3, 3] = np.asarray(self.continuous_predicted_translation_scene, dtype=np.float64)
        return value

    def as_dict(self) -> dict[str, Any]:
        def convert(item: Any) -> Any:
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, tuple):
                return list(item)
            return item

        return {key: convert(item) for key, item in asdict(self).items()}

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class RetargetNonRegressionBudgetAuthorityV1:
    """Outcome-independent limits inherited from existing frozen authorities."""

    interaction_target: float
    wrist_position_limit_m: float
    wrist_rotation_limit_rad: float
    bone_direction_limit_rad: float
    temporal_base_translation_limit_m: float
    temporal_base_rotation_limit_rad: float
    temporal_q_inf_limit_rad: float
    temporal_excess_keypoint_limit_m: float
    collision_hard_bound_m: float
    collision_soft_tolerance_m: float
    reflection_determinant_minimum: float
    unit_scale_ratio_minimum: float
    unit_scale_ratio_maximum: float
    interaction_authority: str
    fidelity_authority: str
    temporal_authority: str
    collision_authority: str
    schema_version: str = NONREGRESSION_AUTHORITY_SCHEMA_VERSION

    @classmethod
    def from_frozen_v1(
        cls, *, collision_hard_bound_m: float, collision_soft_tolerance_m: float
    ) -> RetargetNonRegressionBudgetAuthorityV1:
        gate = SemanticGateContractV1()
        return cls(
            interaction_target=gate.interaction_e_im_p95_limit,
            wrist_position_limit_m=gate.object_relative_wrist_position_limit_m,
            wrist_rotation_limit_rad=gate.object_relative_wrist_rotation_limit_rad,
            bone_direction_limit_rad=gate.bone_direction_p95_limit_rad,
            temporal_base_translation_limit_m=S_POS_M,
            temporal_base_rotation_limit_rad=S_ROT_RAD,
            temporal_q_inf_limit_rad=S_Q_RAD,
            temporal_excess_keypoint_limit_m=0.020,
            collision_hard_bound_m=float(collision_hard_bound_m),
            collision_soft_tolerance_m=float(collision_soft_tolerance_m),
            reflection_determinant_minimum=gate.reflection_determinant_minimum,
            unit_scale_ratio_minimum=gate.unit_scale_ratio_minimum,
            unit_scale_ratio_maximum=gate.unit_scale_ratio_maximum,
            interaction_authority="RetargetSemanticValidityV1.interaction_e_im_p95_limit",
            fidelity_authority="RetargetSemanticValidityV1 explicit wrist/bone/invariant limits",
            temporal_authority="wuji_continuous_full_state_v1 continuity thresholds",
            collision_authority="production PaperRefinementWeights tau/b hard constraints",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveV2Measurements:
    """Physical/semantic quantities required for one structured receipt."""

    interaction_e_im: float
    secondary_objective: float
    bone_direction_p95_rad: float
    wrist_position_m: float
    wrist_rotation_rad: float
    temporal_base_translation_m: float
    temporal_base_rotation_rad: float
    temporal_q_inf_rad: float
    temporal_excess_keypoint_m: float
    collision_min_signed_distance_m: float
    joint_limit_min_margin_rad: float
    rotation_determinant: float
    unit_scale_ratio: float
    per_components: dict[str, float]


@dataclass(frozen=True)
class ObjectiveV2Candidate:
    name: CandidateName
    primary_definition: str
    secondary_definition: str
    retention_definition: str
    low_interaction_behavior: str
    schema_version: str = OBJECTIVE_V2_SCHEMA_VERSION

    @classmethod
    def candidate_a(cls) -> ObjectiveV2Candidate:
        return cls(
            name="A_INTERACTION_CONSTRAINED_FIDELITY",
            primary_definition="minimize E_IM under hard and non-regression constraints",
            secondary_definition="minimize V1 fidelity terms excluding interaction",
            retention_definition="E_IM <= tau if reached, else primary optimum + 1e-10",
            low_interaction_behavior="continues to reduce E_IM below tau in primary phase",
        )

    @classmethod
    def candidate_b(cls) -> ObjectiveV2Candidate:
        return cls(
            name="B_LEXICOGRAPHIC_THRESHOLD",
            primary_definition="minimize max(E_IM/tau - 1, 0)^2",
            secondary_definition="minimize V1 fidelity terms excluding interaction",
            retention_definition="E_IM <= tau if reached, else primary optimum + 1e-10",
            low_interaction_behavior="zero primary pressure at and below Semantic V1 tau",
        )

    def primary_value(self, interaction_e_im: float, target: float) -> float:
        if self.name == "A_INTERACTION_CONSTRAINED_FIDELITY":
            return float(interaction_e_im) / float(target)
        return interaction_hinge(interaction_e_im, target)

    def preference_key(
        self,
        measurement: ObjectiveV2Measurements,
        authority: RetargetNonRegressionBudgetAuthorityV1,
    ) -> tuple[float, float]:
        return (
            self.primary_value(measurement.interaction_e_im, authority.interaction_target),
            float(measurement.secondary_objective),
        )


def constraint_margins(
    values: ObjectiveV2Measurements, authority: RetargetNonRegressionBudgetAuthorityV1
) -> dict[str, float]:
    """Return positive-is-feasible margins for every frozen budget."""

    return {
        "wrist_position_m": authority.wrist_position_limit_m - values.wrist_position_m,
        "wrist_rotation_rad": authority.wrist_rotation_limit_rad - values.wrist_rotation_rad,
        "bone_direction_rad": authority.bone_direction_limit_rad - values.bone_direction_p95_rad,
        "temporal_base_translation_m": authority.temporal_base_translation_limit_m
        - values.temporal_base_translation_m,
        "temporal_base_rotation_rad": authority.temporal_base_rotation_limit_rad
        - values.temporal_base_rotation_rad,
        "temporal_q_inf_rad": authority.temporal_q_inf_limit_rad - values.temporal_q_inf_rad,
        "temporal_excess_keypoint_m": authority.temporal_excess_keypoint_limit_m
        - values.temporal_excess_keypoint_m,
        "collision_hard_m": values.collision_min_signed_distance_m
        + authority.collision_hard_bound_m,
        "joint_limit_rad": values.joint_limit_min_margin_rad,
        "reflection_determinant": values.rotation_determinant
        - authority.reflection_determinant_minimum,
        "unit_scale_lower": values.unit_scale_ratio - authority.unit_scale_ratio_minimum,
        "unit_scale_upper": authority.unit_scale_ratio_maximum - values.unit_scale_ratio,
    }


def evaluate_candidate(
    candidate: ObjectiveV2Candidate,
    values: ObjectiveV2Measurements,
    authority: RetargetNonRegressionBudgetAuthorityV1,
) -> dict[str, Any]:
    """Evaluate one state without collapsing primary and constraints to one scalar."""

    margins = constraint_margins(values, authority)
    violated = sorted(name for name, margin in margins.items() if margin < -1.0e-12)
    return {
        "schema_version": "RetargetObjectiveV2EvaluationReceipt",
        "candidate": candidate.name,
        "primary_objective": candidate.primary_value(
            values.interaction_e_im, authority.interaction_target
        ),
        "primary_interaction_e_im": values.interaction_e_im,
        "per_frame_semantic_target_met": values.interaction_e_im <= authority.interaction_target,
        "per_frame_target_is_not_trajectory_p95_gate": True,
        "secondary_objective": values.secondary_objective,
        "constraint_margins": margins,
        "violated_constraints": violated,
        "feasible": not violated,
        "per_components": dict(values.per_components),
        "semantic_diagnostics": {
            "wrist_position_m": values.wrist_position_m,
            "wrist_rotation_rad": values.wrist_rotation_rad,
            "bone_direction_p95_rad": values.bone_direction_p95_rad,
            "rotation_determinant": values.rotation_determinant,
            "unit_scale_ratio": values.unit_scale_ratio,
        },
    }


def compare_candidate_states(
    candidate: ObjectiveV2Candidate,
    left: ObjectiveV2Measurements,
    right: ObjectiveV2Measurements,
    authority: RetargetNonRegressionBudgetAuthorityV1,
) -> dict[str, Any]:
    """Compare two states using feasibility then the candidate lexicographic order."""

    left_receipt = evaluate_candidate(candidate, left, authority)
    right_receipt = evaluate_candidate(candidate, right, authority)
    if left_receipt["feasible"] != right_receipt["feasible"]:
        preferred = "LEFT" if left_receipt["feasible"] else "RIGHT"
        reason = "FEASIBILITY"
    elif not left_receipt["feasible"]:
        preferred, reason = "NEITHER", "BOTH_INFEASIBLE"
    else:
        left_key = candidate.preference_key(left, authority)
        right_key = candidate.preference_key(right, authority)
        preferred = "LEFT" if left_key < right_key else "RIGHT" if right_key < left_key else "TIE"
        reason = "LEXICOGRAPHIC_PRIMARY_THEN_SECONDARY"
    return {
        "candidate": candidate.name,
        "preferred": preferred,
        "reason": reason,
        "left": left_receipt,
        "right": right_receipt,
    }


class ObjectiveV2DevelopmentContext:
    """Proxy a production frame context with one V2 development objective.

    Collision handling, joint bounds, coordinate normalization, active-set
    updates, and full-surface audits remain delegated to the frozen production
    solver.  Only the objective callback and optional lexicographic retention
    inequality are replaced.
    """

    schema_version = DEVELOPMENT_OPTIMIZER_SCHEMA_VERSION
    development_only = True
    production_certified = False

    def __init__(
        self,
        base_context: Any,
        candidate: ObjectiveV2Candidate,
        authority: RetargetNonRegressionBudgetAuthorityV1,
        *,
        phase: PhaseName,
        interaction_retention_limit: float | None = None,
    ) -> None:
        self.base_context = base_context
        self.candidate = candidate
        self.authority = authority
        self.phase = phase
        self.interaction_retention_limit = interaction_retention_limit

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_context, name)

    def _interaction_tensor(self, value: Any) -> Any:
        robot_keypoints = self.base_context.robot_model.keypoints_scene(
            self.base_context.unpack(value)[2],
            self.base_context.base_pose_torch(value),
            layout="mediapipe21",
        )
        robot_vertices = self.base_context.robot_graph_vertices_torch(value, robot_keypoints)
        residual = self.base_context._residual_model(robot_vertices)
        return residual.square().sum() / 71.0

    def _interaction_value_gradient(self, value: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        variable = torch.as_tensor(
            np.asarray(value, dtype=np.float64), dtype=torch.float64
        ).requires_grad_(True)
        interaction = self._interaction_tensor(variable)
        gradient = torch.autograd.grad(interaction, variable, create_graph=False)[0]
        return float(interaction.detach().cpu()), gradient.detach().cpu().numpy().copy()

    def objective(self, value: np.ndarray, query_hash: str | None = None) -> tuple[Any, ...]:
        interaction, interaction_gradient = self._interaction_value_gradient(value)
        target = self.authority.interaction_target
        _v1_total, v1_gradient, breakdown = self.base_context.objective(value, query_hash)
        if self.phase == "primary":
            if self.candidate.name == "A_INTERACTION_CONSTRAINED_FIDELITY":
                total = interaction / target
                gradient = interaction_gradient / target
            else:
                excess = max(interaction / target - 1.0, 0.0)
                total = excess * excess
                gradient = (
                    np.zeros_like(interaction_gradient)
                    if excess == 0.0
                    else 2.0 * excess * interaction_gradient / target
                )
        else:
            total = float(_v1_total) - self.base_context.paper.lambda_im * interaction
            gradient = np.asarray(v1_gradient) - (
                self.base_context.paper.lambda_im * interaction_gradient
            )
        return (
            float(total),
            np.asarray(gradient, dtype=np.float64),
            replace(breakdown, total=float(total)),
        )

    def constraint_values(
        self, value: np.ndarray, query_ids: np.ndarray, query_hash: str | None = None
    ) -> np.ndarray:
        base = self.base_context.constraint_values(value, query_ids, query_hash)
        if self.interaction_retention_limit is None:
            return base
        interaction, _gradient = self._interaction_value_gradient(value)
        return np.concatenate([base, [float(self.interaction_retention_limit) - interaction]])

    def constraint_jacobian(
        self,
        value: np.ndarray,
        query_ids: np.ndarray,
        eps: float,
        query_hash: str | None = None,
        backend: str = "analytic_urdf_spatial_v2",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        base, diagnostics = self.base_context.constraint_jacobian(
            value, query_ids, eps, query_hash, backend
        )
        if self.interaction_retention_limit is None:
            return base, diagnostics
        _interaction, gradient = self._interaction_value_gradient(value)
        return np.vstack([base, -gradient]), diagnostics


def interaction_retention_limit(primary_e_im: float, interaction_target: float) -> float:
    """Freeze the secondary-phase retention limit without outcome tuning."""

    if primary_e_im <= interaction_target:
        return float(interaction_target)
    return float(primary_e_im) + NUMERICAL_RETENTION_EPSILON


__all__ = [
    "CONTEXT_BINDING_V2_SCHEMA_VERSION",
    "DEVELOPMENT_OPTIMIZER_SCHEMA_VERSION",
    "NONREGRESSION_AUTHORITY_SCHEMA_VERSION",
    "NUMERICAL_RETENTION_EPSILON",
    "OBJECTIVE_V2_SCHEMA_VERSION",
    "ObjectiveV2Candidate",
    "ObjectiveV2DevelopmentContext",
    "ObjectiveV2Measurements",
    "ProductionObjectiveContextBindingV2",
    "RetargetNonRegressionBudgetAuthorityV1",
    "canonical_sha256",
    "compare_candidate_states",
    "constraint_margins",
    "evaluate_candidate",
    "interaction_hinge",
    "interaction_retention_limit",
]
