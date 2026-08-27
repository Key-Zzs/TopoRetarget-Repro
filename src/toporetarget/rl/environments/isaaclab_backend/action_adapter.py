"""Frozen 26-D action contract and canonical-to-Isaac joint mapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Stage16ActionContractV1:
    identifier: str = "stage16_world_wrist_finger_action_v1"
    action_dimension: int = 26
    wrist_translation_scale_m: float = 0.01
    wrist_rotation_scale_rad: float = 0.08726646259971647
    finger_joint_range_fraction: float = 0.10
    wrist_residual_frame: str = "reference_wrist_local"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Stage16ActionAdapter:
    """Apply residual actions without relying on Isaac's internal joint order."""

    def __init__(
        self,
        *,
        canonical_joint_names: tuple[str, ...],
        isaac_joint_names: tuple[str, ...],
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        contract: Stage16ActionContractV1 = Stage16ActionContractV1(),
    ) -> None:
        if len(canonical_joint_names) != 20 or len(isaac_joint_names) != 20:
            raise ValueError("Stage 16-C requires exactly 20 finger joints")
        if set(canonical_joint_names) != set(isaac_joint_names):
            raise ValueError("canonical and Isaac joint names must be an exact permutation")
        if joint_lower.shape != (20,) or joint_upper.shape != (20,):
            raise ValueError("finger joint bounds must have shape [20]")
        if bool(torch.any(joint_upper <= joint_lower)):
            raise ValueError("finger joint upper bounds must exceed lower bounds")
        self.contract = contract
        self.canonical_joint_names = canonical_joint_names
        self.isaac_joint_names = isaac_joint_names
        self.isaac_from_canonical = torch.tensor(
            [isaac_joint_names.index(name) for name in canonical_joint_names],
            dtype=torch.long,
            device=joint_lower.device,
        )
        self.canonical_from_isaac = torch.argsort(self.isaac_from_canonical)
        self.joint_lower = joint_lower
        self.joint_upper = joint_upper

    def validate_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != self.contract.action_dimension:
            raise ValueError("Stage 16-C action must have shape [num_envs, 26]")
        if not bool(torch.isfinite(action).all()):
            raise ValueError("Stage 16-C action must be finite")
        return action.clamp(-1.0, 1.0)

    def finger_target_canonical(
        self, reference_q: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        values = self.validate_action(action)
        if reference_q.shape != (values.shape[0], 20):
            raise ValueError("reference finger state must have shape [num_envs, 20]")
        return torch.clamp(
            reference_q
            + values[:, 6:]
            * (self.joint_upper - self.joint_lower)
            * self.contract.finger_joint_range_fraction,
            self.joint_lower,
            self.joint_upper,
        )

    def canonical_to_isaac(self, canonical_values: torch.Tensor) -> torch.Tensor:
        if canonical_values.shape[-1] != 20:
            raise ValueError("canonical joint values must end in dimension 20")
        result = torch.empty_like(canonical_values)
        result[..., self.isaac_from_canonical] = canonical_values
        return result

    def isaac_to_canonical(self, isaac_values: torch.Tensor) -> torch.Tensor:
        if isaac_values.shape[-1] != 20:
            raise ValueError("Isaac joint values must end in dimension 20")
        return isaac_values[..., self.isaac_from_canonical]

    def mapping_manifest(self) -> dict[str, object]:
        return {
            "contract": self.contract.as_dict(),
            "canonical_joint_names": list(self.canonical_joint_names),
            "isaac_joint_names": list(self.isaac_joint_names),
            "isaac_index_for_canonical_index": self.isaac_from_canonical.detach().cpu().tolist(),
            "joint_position_target_limits_enforced": True,
        }


__all__ = ["Stage16ActionAdapter", "Stage16ActionContractV1"]
