"""Frozen configuration for the assisted object-reference wrench."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

GuidanceMode = Literal["none", "reference_wrench_v1"]


@dataclass(frozen=True)
class ObjectGuidanceContractV1:
    """Physically normalized, bounded world-frame reference-wrench contract.

    Acceleration limits are invariant to the selected HOCap object.  Runtime
    force/torque limits are consequently derived from that object's measured
    mass and world inertia instead of being hand-tuned per clip.
    """

    identifier: str = "ObjectGuidanceContractV1"
    mode: GuidanceMode = "none"
    reference_kinematics_version: int = 2
    reference_source: str = "Stage16DReferenceKinematicsV2"
    translation_natural_frequency_hz: float = 2.0
    translation_damping_ratio: float = 1.0
    rotation_natural_frequency_hz: float = 2.0
    rotation_damping_ratio: float = 1.0
    translation_acceleration_cap_mps2: float = 3.0
    rotation_acceleration_cap_radps2: float = 12.0
    position_deadband_m: float = 0.001
    rotation_deadband_rad: float = 0.01
    linear_velocity_deadband_mps: float = 0.0
    angular_velocity_deadband_radps: float = 0.0
    external_guidance: bool = True
    assisted_dynamics: bool = True
    causal_physics: bool = False
    engineering_label: str = "ENGINEERING_EXTENSION_ASSISTED_DYNAMICS"

    def __post_init__(self) -> None:
        if self.mode not in {"none", "reference_wrench_v1"}:
            raise ValueError(f"OBJECT_GUIDANCE_MODE_INVALID:{self.mode}")
        if self.mode != "none" and self.reference_kinematics_version != 2:
            raise ValueError("OBJECT_GUIDANCE_REQUIRES_REFERENCE_KINEMATICS_V2")
        for field in (
            "translation_natural_frequency_hz",
            "translation_damping_ratio",
            "rotation_natural_frequency_hz",
            "rotation_damping_ratio",
            "translation_acceleration_cap_mps2",
            "rotation_acceleration_cap_radps2",
        ):
            if getattr(self, field) <= 0.0:
                raise ValueError(f"OBJECT_GUIDANCE_NONPOSITIVE:{field}")
        for field in (
            "position_deadband_m",
            "rotation_deadband_rad",
            "linear_velocity_deadband_mps",
            "angular_velocity_deadband_radps",
        ):
            if getattr(self, field) < 0.0:
                raise ValueError(f"OBJECT_GUIDANCE_NEGATIVE_DEADBAND:{field}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def sha256(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
