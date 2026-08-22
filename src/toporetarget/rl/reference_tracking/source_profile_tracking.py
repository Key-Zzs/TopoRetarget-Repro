"""Object-agnostic SourceProfileTrackingV1 reward primitives.

This module deliberately consumes the already-authoritative
``HumanObjectCouplingContactProfileV1`` materialization.  It does not create a
second source-profile definition, infer a grasp label, or change the reference
timeline.  The runtime path is torch-only so it can stay on the simulator
device; the same functions are also used by the offline validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import torch

SOURCE_PROFILE_TRACKING_V1: Final = "Stage16SourceProfileTrackingV1"
SOURCE_PROFILE_CLIPS: Final = ("hocap_170105", "hocap_170650")


@dataclass(frozen=True)
class Stage16SourceProfileTrackingV1:
    """Frozen, additive V1 source-profile objective contract.

    Every included residual is already dimensionless: activity is bounded,
    geometry is divided by the object characteristic span, and each coupling
    ratio is divided by one global source-population scale.  Therefore all
    component weights are fixed global unit weights rather than object or
    outcome-tuned coefficients.
    """

    identifier: str = SOURCE_PROFILE_TRACKING_V1
    source_profile_identifier: str = "HumanObjectCouplingContactProfileV1"
    time_alignment: str = "reference_index_identity_no_dtw_no_learned_or_outcome_shift"
    robust_loss: str = "pseudo_huber_delta_1"
    pseudo_huber_delta: float = 1.0
    profile_reward_weight: float = 1.0
    channel_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    force_activation: str = "one_minus_exp_negative_named_tip_force_over_v4_lambda_tip"
    geometry: str = "source_active_semantic_tip_centroid_in_object_local_frame"
    object_scale: str = "reference_object_axis_pair_span"
    coupling: str = "pose_derived_relative_ratio_reference_kinematics_v2_family"
    opposition_exact_matching: str = "DIAGNOSTIC_ONLY_NO_ACTUAL_CONTACT_NORMALS"
    exact_slip: str = "DIAGNOSTIC_ONLY_NO_ACTUAL_CONTACT_POINT_TRACKS"
    fixed_pre_lift_grasp_gate_added: bool = False
    manual_grasp_frame_added: bool = False
    outcome_tuned: bool = False
    per_object_profile_weight: bool = False

    def __post_init__(self) -> None:
        if self.identifier != SOURCE_PROFILE_TRACKING_V1:
            raise ValueError("SOURCE_PROFILE_TRACKING_IDENTIFIER_DRIFT")
        if self.source_profile_identifier != "HumanObjectCouplingContactProfileV1":
            raise ValueError("SOURCE_PROFILE_TRACKING_SOURCE_AUTHORITY_DRIFT")
        if self.pseudo_huber_delta <= 0.0 or self.profile_reward_weight != 1.0:
            raise ValueError("SOURCE_PROFILE_TRACKING_WEIGHT_OR_LOSS_DRIFT")
        if self.channel_weights != (1.0, 1.0, 1.0, 1.0):
            raise ValueError("SOURCE_PROFILE_TRACKING_GLOBAL_WEIGHT_DRIFT")
        if (
            self.fixed_pre_lift_grasp_gate_added
            or self.manual_grasp_frame_added
            or self.outcome_tuned
            or self.per_object_profile_weight
        ):
            raise ValueError("SOURCE_PROFILE_TRACKING_FORBIDDEN_TUNING_OR_GATE")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceProfileTrackingTargetsV1:
    """GPU-ready source target arrays in the frozen reference-bank order."""

    contact_activity: torch.Tensor  # [2, 321, 5], dimensionless
    geometry_object_normalized: torch.Tensor  # [2, 321, 3], dimensionless
    geometry_valid: torch.Tensor  # [2, 321]
    linear_coupling_normalized: torch.Tensor  # [2, 321], dimensionless
    angular_coupling_normalized: torch.Tensor  # [2, 321], dimensionless
    object_characteristic_length_m: torch.Tensor  # [2]
    linear_coupling_scale: float
    angular_coupling_scale: float
    source_path: str

    @classmethod
    def from_npz(
        cls, path: str | Path, *, device: torch.device | str
    ) -> SourceProfileTrackingTargetsV1:
        """Load the immutable target sidecar and reject schema/order drift."""

        resolved = Path(path).resolve()
        with np.load(resolved, allow_pickle=False) as archive:
            required = {
                "schema_version",
                "clip_ids",
                "contact_activity",
                "geometry_object_normalized",
                "geometry_valid",
                "linear_coupling_normalized",
                "angular_coupling_normalized",
                "object_characteristic_length_m",
                "linear_coupling_scale",
                "angular_coupling_scale",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"SOURCE_PROFILE_TARGET_FIELDS_MISSING:{sorted(missing)}")
            if str(archive["schema_version"].item()) != SOURCE_PROFILE_TRACKING_V1:
                raise ValueError("SOURCE_PROFILE_TARGET_SCHEMA_DRIFT")
            clip_ids = tuple(str(value) for value in archive["clip_ids"].tolist())
            if clip_ids != SOURCE_PROFILE_CLIPS:
                raise ValueError("SOURCE_PROFILE_TARGET_CLIP_ORDER_DRIFT")
            contact = np.asarray(archive["contact_activity"], dtype=np.float32)
            geometry = np.asarray(archive["geometry_object_normalized"], dtype=np.float32)
            valid = np.asarray(archive["geometry_valid"], dtype=bool)
            linear = np.asarray(archive["linear_coupling_normalized"], dtype=np.float32)
            angular = np.asarray(archive["angular_coupling_normalized"], dtype=np.float32)
            scale = np.asarray(archive["object_characteristic_length_m"], dtype=np.float32)
            linear_scale = float(archive["linear_coupling_scale"].item())
            angular_scale = float(archive["angular_coupling_scale"].item())
        if (
            contact.shape != (2, 321, 5)
            or geometry.shape != (2, 321, 3)
            or valid.shape != (2, 321)
            or linear.shape != (2, 321)
            or angular.shape != (2, 321)
            or scale.shape != (2,)
            or not np.isfinite(contact).all()
            or not np.isfinite(geometry[valid]).all()
            or not np.isfinite(linear).all()
            or not np.isfinite(angular).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
            or not np.isfinite([linear_scale, angular_scale]).all()
            or linear_scale <= 0.0
            or angular_scale <= 0.0
        ):
            raise ValueError("SOURCE_PROFILE_TARGET_NUMERICAL_CONTRACT_INVALID")
        # Invalid geometry is masked before use.  Store a finite neutral value
        # so CUDA finite checks cannot be bypassed by a NaN sentinel.
        geometry = np.where(valid[..., None], geometry, 0.0)
        return cls(
            contact_activity=torch.as_tensor(contact, device=device),
            geometry_object_normalized=torch.as_tensor(geometry, device=device),
            geometry_valid=torch.as_tensor(valid, dtype=torch.bool, device=device),
            linear_coupling_normalized=torch.as_tensor(linear, device=device),
            angular_coupling_normalized=torch.as_tensor(angular, device=device),
            object_characteristic_length_m=torch.as_tensor(scale, device=device),
            linear_coupling_scale=linear_scale,
            angular_coupling_scale=angular_scale,
            source_path=str(resolved),
        )

    def gather(
        self, clip_index: torch.Tensor, reference_index: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if clip_index.ndim != 1 or reference_index.shape != clip_index.shape:
            raise ValueError("SOURCE_PROFILE_TARGET_INDEX_SHAPE_INVALID")
        clips = clip_index.to(dtype=torch.long)
        frames = reference_index.to(dtype=torch.long).clamp(0, 320)
        if bool(((clips < 0) | (clips >= 2)).any()):
            raise ValueError("SOURCE_PROFILE_TARGET_CLIP_INDEX_INVALID")
        return {
            "source_contact_activity": self.contact_activity[clips, frames],
            "source_geometry_object_normalized": self.geometry_object_normalized[clips, frames],
            "source_geometry_valid": self.geometry_valid[clips, frames],
            "source_linear_coupling_normalized": self.linear_coupling_normalized[clips, frames],
            "source_angular_coupling_normalized": self.angular_coupling_normalized[clips, frames],
            "object_characteristic_length_m": self.object_characteristic_length_m[clips],
        }


def _quaternion_to_matrix_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return rotation matrices for normalized WXYZ quaternions."""

    if quaternion.shape[-1] != 4:
        raise ValueError("SOURCE_PROFILE_QUATERNION_SHAPE_INVALID")
    normalized = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    w, x, y, z = normalized.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _rotation_vector(rotation: torch.Tensor) -> torch.Tensor:
    """Stable SO(3) logarithm for a batch of rotation matrices."""

    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    skew = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    scale = angle / (2.0 * sine.clamp_min(1.0e-7))
    value = skew * scale[..., None]
    # sin(theta) becomes small near zero; first-order log(R) is exact enough
    # for the 20 Hz pose-derived finite-difference estimator used here.
    return torch.where((angle < 1.0e-5)[..., None], 0.5 * skew, value)


def pose_derived_coupling_ratios(
    *,
    previous_wrist_pose_wxyz: torch.Tensor,
    current_wrist_pose_wxyz: torch.Tensor,
    previous_object_pose_wxyz: torch.Tensor,
    current_object_pose_wxyz: torch.Tensor,
    dt_s: float,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal pose-only coupling counterpart of the V2 estimator family.

    This uses the latest fixed 20-Hz finite difference only.  It neither reads
    simulator angular velocities nor changes the reference phase, so it cannot
    erase a contact delay through a learned or outcome-selected alignment.
    """

    if dt_s <= 0.0 or epsilon <= 0.0:
        raise ValueError("SOURCE_PROFILE_COUPLING_TIME_OR_EPSILON_INVALID")
    expected = previous_wrist_pose_wxyz.shape
    if (
        expected[-1] != 7
        or current_wrist_pose_wxyz.shape != expected
        or previous_object_pose_wxyz.shape != expected
        or current_object_pose_wxyz.shape != expected
    ):
        raise ValueError("SOURCE_PROFILE_POSE_SHAPE_INVALID")
    wrist_previous_rotation = _quaternion_to_matrix_wxyz(previous_wrist_pose_wxyz[:, 3:])
    wrist_current_rotation = _quaternion_to_matrix_wxyz(current_wrist_pose_wxyz[:, 3:])
    object_previous_rotation = _quaternion_to_matrix_wxyz(previous_object_pose_wxyz[:, 3:])
    object_current_rotation = _quaternion_to_matrix_wxyz(current_object_pose_wxyz[:, 3:])
    relative_previous = torch.matmul(
        wrist_previous_rotation.transpose(-1, -2), object_previous_rotation
    )
    relative_current = torch.matmul(
        wrist_current_rotation.transpose(-1, -2), object_current_rotation
    )
    relative_position_previous = torch.matmul(
        wrist_previous_rotation.transpose(-1, -2),
        (previous_object_pose_wxyz[:, :3] - previous_wrist_pose_wxyz[:, :3]).unsqueeze(-1),
    ).squeeze(-1)
    relative_position_current = torch.matmul(
        wrist_current_rotation.transpose(-1, -2),
        (current_object_pose_wxyz[:, :3] - current_wrist_pose_wxyz[:, :3]).unsqueeze(-1),
    ).squeeze(-1)
    hand_linear = torch.linalg.vector_norm(
        (current_wrist_pose_wxyz[:, :3] - previous_wrist_pose_wxyz[:, :3]) / dt_s, dim=-1
    )
    object_linear = torch.linalg.vector_norm(
        (current_object_pose_wxyz[:, :3] - previous_object_pose_wxyz[:, :3]) / dt_s, dim=-1
    )
    relative_linear = torch.linalg.vector_norm(
        (relative_position_current - relative_position_previous) / dt_s, dim=-1
    )
    hand_angular = torch.linalg.vector_norm(
        _rotation_vector(
            torch.matmul(wrist_previous_rotation.transpose(-1, -2), wrist_current_rotation)
        )
        / dt_s,
        dim=-1,
    )
    object_angular = torch.linalg.vector_norm(
        _rotation_vector(
            torch.matmul(object_previous_rotation.transpose(-1, -2), object_current_rotation)
        )
        / dt_s,
        dim=-1,
    )
    relative_angular = torch.linalg.vector_norm(
        _rotation_vector(torch.matmul(relative_previous.transpose(-1, -2), relative_current))
        / dt_s,
        dim=-1,
    )
    linear_ratio = relative_linear / (hand_linear + object_linear + epsilon)
    angular_ratio = relative_angular / (hand_angular + object_angular + epsilon)
    if not bool(torch.isfinite(linear_ratio).all()) or not bool(
        torch.isfinite(angular_ratio).all()
    ):
        raise FloatingPointError("SOURCE_PROFILE_POSE_DERIVED_COUPLING_NONFINITE")
    return linear_ratio, angular_ratio


def _pseudo_huber(residual: torch.Tensor, *, delta: float) -> torch.Tensor:
    return delta * delta * (torch.sqrt(1.0 + (residual / delta).square()) - 1.0)


def source_profile_tracking_terms(
    *,
    source_contact_activity: torch.Tensor,
    source_geometry_object_normalized: torch.Tensor,
    source_geometry_valid: torch.Tensor,
    source_linear_coupling_normalized: torch.Tensor,
    source_angular_coupling_normalized: torch.Tensor,
    object_characteristic_length_m: torch.Tensor,
    robot_tip_positions_world: torch.Tensor,
    robot_tip_pair_force_world: torch.Tensor,
    object_pose_wxyz: torch.Tensor,
    robot_linear_coupling_normalized: torch.Tensor,
    robot_angular_coupling_normalized: torch.Tensor,
    contact_force_scale_n: float,
    contract: Stage16SourceProfileTrackingV1 | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the frozen V1 profile terms for a batch of simulator states."""

    frozen = contract or Stage16SourceProfileTrackingV1()
    if contact_force_scale_n <= 0.0:
        raise ValueError("SOURCE_PROFILE_CONTACT_SCALE_INVALID")
    batch = source_contact_activity.shape[0]
    if (
        source_contact_activity.shape != (batch, 5)
        or source_geometry_object_normalized.shape != (batch, 3)
        or source_geometry_valid.shape != (batch,)
        or source_linear_coupling_normalized.shape != (batch,)
        or source_angular_coupling_normalized.shape != (batch,)
        or object_characteristic_length_m.shape != (batch,)
        or robot_tip_positions_world.shape != (batch, 5, 3)
        or robot_tip_pair_force_world.shape != (batch, 5, 3)
        or object_pose_wxyz.shape != (batch, 7)
        or robot_linear_coupling_normalized.shape != (batch,)
        or robot_angular_coupling_normalized.shape != (batch,)
    ):
        raise ValueError("SOURCE_PROFILE_RUNTIME_SHAPE_INVALID")
    all_values = (
        source_contact_activity,
        source_geometry_object_normalized,
        source_linear_coupling_normalized,
        source_angular_coupling_normalized,
        object_characteristic_length_m,
        robot_tip_positions_world,
        robot_tip_pair_force_world,
        object_pose_wxyz,
        robot_linear_coupling_normalized,
        robot_angular_coupling_normalized,
    )
    if not all(bool(torch.isfinite(value).all()) for value in all_values):
        raise FloatingPointError("SOURCE_PROFILE_RUNTIME_INPUT_NONFINITE")
    if bool((object_characteristic_length_m <= 0.0).any()):
        raise ValueError("SOURCE_PROFILE_OBJECT_SCALE_INVALID")

    force_norm = torch.linalg.vector_norm(robot_tip_pair_force_world, dim=-1)
    robot_contact_activity = 1.0 - torch.exp(-force_norm / float(contact_force_scale_n))
    object_rotation = _quaternion_to_matrix_wxyz(object_pose_wxyz[:, 3:])
    tip_object_local = torch.matmul(
        object_rotation.transpose(-1, -2)[:, None],
        (robot_tip_positions_world - object_pose_wxyz[:, None, :3]).unsqueeze(-1),
    ).squeeze(-1)
    source_activity_sum = source_contact_activity.sum(dim=-1, keepdim=True)
    semantic_weights = torch.where(
        source_activity_sum > 0.0,
        source_contact_activity / source_activity_sum.clamp_min(1.0e-8),
        torch.full_like(source_contact_activity, 1.0 / 5.0),
    )
    robot_geometry_normalized = (semantic_weights[..., None] * tip_object_local).sum(
        dim=1
    ) / object_characteristic_length_m[:, None]
    contact_component = _pseudo_huber(
        robot_contact_activity - source_contact_activity, delta=frozen.pseudo_huber_delta
    ).mean(dim=-1)
    geometry_residual = _pseudo_huber(
        robot_geometry_normalized - source_geometry_object_normalized,
        delta=frozen.pseudo_huber_delta,
    ).mean(dim=-1)
    geometry_component = torch.where(
        source_geometry_valid.to(dtype=torch.bool),
        geometry_residual,
        torch.zeros_like(geometry_residual),
    )
    linear_component = _pseudo_huber(
        robot_linear_coupling_normalized - source_linear_coupling_normalized,
        delta=frozen.pseudo_huber_delta,
    )
    angular_component = _pseudo_huber(
        robot_angular_coupling_normalized - source_angular_coupling_normalized,
        delta=frozen.pseudo_huber_delta,
    )
    total_loss = (
        contact_component + geometry_component + linear_component + angular_component
    ) / 4.0
    profile_reward = torch.exp(-total_loss)
    if (
        not bool(torch.isfinite(total_loss).all())
        or not bool(torch.isfinite(profile_reward).all())
        or bool((profile_reward <= 0.0).any())
        or bool((profile_reward > 1.0 + 1.0e-6).any())
    ):
        raise FloatingPointError("SOURCE_PROFILE_REWARD_NONFINITE_OR_OUT_OF_RANGE")
    return {
        "l_profile": total_loss,
        "l_profile_contact": contact_component,
        "l_profile_geometry": geometry_component,
        "l_profile_linear_coupling": linear_component,
        "l_profile_angular_coupling": angular_component,
        "r_profile": frozen.profile_reward_weight * profile_reward,
        "robot_contact_activity": robot_contact_activity,
        "robot_geometry_object_normalized": robot_geometry_normalized,
        "robot_linear_coupling_normalized": robot_linear_coupling_normalized,
        "robot_angular_coupling_normalized": robot_angular_coupling_normalized,
    }


__all__ = [
    "SOURCE_PROFILE_CLIPS",
    "SOURCE_PROFILE_TRACKING_V1",
    "SourceProfileTrackingTargetsV1",
    "Stage16SourceProfileTrackingV1",
    "pose_derived_coupling_ratios",
    "source_profile_tracking_terms",
]
