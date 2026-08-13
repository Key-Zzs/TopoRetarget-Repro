"""Versioned Stage 16-D contact-reward selection and compatibility helpers.

The stable closeout interface is deliberately small: a user selects a named
contact objective, while the implementation continues to use the frozen V3 or
V4 contracts.  Older artifacts did not carry this field, so only their
unambiguous reward-contract identifiers are migrated.  Nothing without that
provenance is silently assigned a new contact objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV3,
    TopoRetargetReferenceTrackingReward26DV4,
)


class ContactRewardMode(str, Enum):
    """The two frozen Stage 16-D contact objectives."""

    AGGREGATE_V3 = "aggregate_v3"
    STRICT_PER_FINGER_V4 = "strict_per_finger_v4"

    @classmethod
    def parse(cls, value: str | ContactRewardMode) -> ContactRewardMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            available = ", ".join(item.value for item in cls)
            raise ValueError(
                f"STAGE16D_CONTACT_MODE_INVALID:{value!r}; expected one of: {available}"
            ) from exc


CONTACT_MODE_TO_REWARD_CONTRACT: dict[ContactRewardMode, str] = {
    ContactRewardMode.AGGREGATE_V3: "TopoRetargetReferenceTrackingReward26DV3",
    ContactRewardMode.STRICT_PER_FINGER_V4: "TopoRetargetReferenceTrackingReward26DV4",
}
REWARD_CONTRACT_TO_CONTACT_MODE: dict[str, ContactRewardMode] = {
    value: key for key, value in CONTACT_MODE_TO_REWARD_CONTRACT.items()
}


@dataclass(frozen=True)
class Stage16DContactRewardConfigV1:
    """Portable configuration with the frozen stable default.

    This describes newly created Stage 16-D closeout configurations.  It is
    intentionally not applied to missing fields in historical artifacts.
    """

    identifier: str = "Stage16DContactRewardConfigV1"
    mode: ContactRewardMode = ContactRewardMode.AGGREGATE_V3

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Stage16DContactRewardConfigV1:
        reward = payload.get("reward")
        if not isinstance(reward, Mapping):
            raise ValueError("STAGE16D_CONTACT_CONFIG_REWARD_SECTION_MISSING")
        contact = reward.get("contact")
        if not isinstance(contact, Mapping):
            raise ValueError("STAGE16D_CONTACT_CONFIG_SECTION_MISSING")
        mode = contact.get("mode", ContactRewardMode.AGGREGATE_V3.value)
        if not isinstance(mode, str):
            raise ValueError("STAGE16D_CONTACT_MODE_MUST_BE_A_STRING")
        return cls(mode=ContactRewardMode.parse(mode))

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "reward": {"contact": {"mode": self.mode.value}},
        }


def legacy_contact_mode(reward_contract_identifier: str) -> ContactRewardMode | None:
    """Map only a historical V3/V4 identifier; V1/V2 have no contact mode."""

    return REWARD_CONTRACT_TO_CONTACT_MODE.get(reward_contract_identifier)


def resolve_contact_mode(
    *,
    configured_mode: str | ContactRewardMode | None,
    reward_contract_identifier: str,
) -> ContactRewardMode | None:
    """Resolve an explicit mode or an unambiguous historical V3/V4 mapping."""

    migrated = legacy_contact_mode(reward_contract_identifier)
    if configured_mode is None:
        return migrated
    selected = ContactRewardMode.parse(configured_mode)
    if migrated is not None and selected is not migrated:
        raise ValueError(
            "STAGE16D_CONTACT_MODE_CONTRACT_MISMATCH:"
            f"mode={selected.value}:contract={reward_contract_identifier}"
        )
    if migrated is None:
        raise ValueError(
            "STAGE16D_CONTACT_MODE_REQUIRES_V3_OR_V4_CONTRACT:"
            f"mode={selected.value}:contract={reward_contract_identifier}"
        )
    return selected


def validate_frozen_contact_contract(
    mode: str | ContactRewardMode, payload: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate the mode-specific frozen receipt before an environment is built."""

    selected = ContactRewardMode.parse(mode)
    parameters = payload.get("frozen_parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("STAGE16D_CONTACT_FROZEN_PARAMETERS_MISSING")
    if selected is ContactRewardMode.AGGREGATE_V3:
        value = parameters.get("lambda_c_n")
        if payload.get("status") != "CONTACT_REWARD_CONTRACT_FROZEN":
            raise ValueError("PPO26D_REWARD_V3_CONTACT_CONTRACT_NOT_FROZEN")
        if not isinstance(value, (int, float)) or float(value) <= 1.0e-5:
            raise ValueError("PPO26D_REWARD_V3_CONTACT_LAMBDA_INVALID")
    else:
        value = parameters.get("lambda_tip_n")
        if payload.get("status") != "STRICT_V4_CONTACT_CONTRACT_FROZEN":
            raise ValueError("STRICT_V4_CONTACT_CONTRACT_NOT_FROZEN")
        if not isinstance(value, (int, float)) or float(value) <= 1.0e-5:
            raise ValueError("STRICT_V4_CONTACT_LAMBDA_INVALID")
        floor = parameters.get("numerical_floor_n")
        if not isinstance(floor, (int, float)) or float(floor) <= 0.0:
            raise ValueError("STRICT_V4_CONTACT_NUMERICAL_FLOOR_INVALID")
    return parameters


def build_contact_reward(
    mode: str | ContactRewardMode, *, frozen_parameters: Mapping[str, object]
) -> TopoRetargetReferenceTrackingReward26DV3 | TopoRetargetReferenceTrackingReward26DV4:
    """Build the existing frozen implementation selected by the unified mode."""

    selected = ContactRewardMode.parse(mode)
    if selected is ContactRewardMode.AGGREGATE_V3:
        value = frozen_parameters.get("lambda_c_n")
        if not isinstance(value, (int, float)):
            raise ValueError("PPO26D_REWARD_V3_CONTACT_LAMBDA_MISSING")
        return TopoRetargetReferenceTrackingReward26DV3(contact_force_scale_lambda_n=float(value))
    value = frozen_parameters.get("lambda_tip_n")
    floor = frozen_parameters.get("numerical_floor_n")
    if not isinstance(value, (int, float)):
        raise ValueError("STRICT_V4_CONTACT_LAMBDA_MISSING")
    if not isinstance(floor, (int, float)):
        raise ValueError("STRICT_V4_CONTACT_NUMERICAL_FLOOR_MISSING")
    return TopoRetargetReferenceTrackingReward26DV4(
        contact_force_scale_lambda_tip_n=float(value),
        contact_numerical_floor_n=float(floor),
    )


__all__ = [
    "CONTACT_MODE_TO_REWARD_CONTRACT",
    "ContactRewardMode",
    "Stage16DContactRewardConfigV1",
    "build_contact_reward",
    "legacy_contact_mode",
    "resolve_contact_mode",
    "validate_frozen_contact_contract",
]
