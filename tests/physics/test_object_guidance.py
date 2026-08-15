from __future__ import annotations

import math

import pytest
import torch

from toporetarget.physics.guidance import ObjectGuidanceContractV1, ReferenceWrenchGuidance


def _guidance(**overrides: object) -> ReferenceWrenchGuidance:
    values: dict[str, object] = {
        "mode": "reference_wrench_v1",
        "translation_natural_frequency_hz": 1.0,
        "rotation_natural_frequency_hz": 1.0,
        "translation_acceleration_cap_mps2": 100.0,
        "rotation_acceleration_cap_radps2": 100.0,
    }
    values.update(overrides)
    return ReferenceWrenchGuidance(ObjectGuidanceContractV1(**values))


def _compute(guidance: ReferenceWrenchGuidance, **overrides: torch.Tensor):
    values = {
        "reference_position_world": torch.zeros((1, 3)),
        "reference_quaternion_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "reference_twist_world": torch.zeros((1, 6)),
        "object_position_world": torch.zeros((1, 3)),
        "object_quaternion_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "object_twist_world": torch.zeros((1, 6)),
        "mass_kg": torch.tensor([2.0]),
        "inertia_world_kgm2": torch.diag_embed(torch.tensor([[2.0, 3.0, 4.0]])),
    }
    values.update(overrides)
    return guidance.compute(**values)


def test_zero_error_is_exactly_zero_wrench() -> None:
    result = _compute(_guidance())
    assert torch.equal(result.force_world, torch.zeros((1, 3)))
    assert torch.equal(result.torque_world, torch.zeros((1, 3)))
    assert not bool(result.guidance_active.item())


def test_translation_and_velocity_signs_follow_reference_error() -> None:
    guidance = _guidance()
    positive = _compute(guidance, reference_position_world=torch.tensor([[1.0, 0.0, 0.0]]))
    negative = _compute(guidance, reference_position_world=torch.tensor([[-1.0, 0.0, 0.0]]))
    damping = _compute(guidance, object_twist_world=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    assert positive.force_world[0, 0] > 0.0
    assert negative.force_world[0, 0] < 0.0
    assert damping.force_world[0, 0] < 0.0


def test_rotation_and_angular_damping_signs_follow_world_error() -> None:
    guidance = _guidance()
    half = math.pi / 4.0
    desired_positive_z = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
    positive = _compute(guidance, reference_quaternion_wxyz=desired_positive_z)
    damping = _compute(guidance, object_twist_world=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    assert positive.torque_world[0, 2] > 0.0
    assert damping.torque_world[0, 2] < 0.0


def test_force_and_torque_are_vector_norm_bounded() -> None:
    guidance = _guidance(
        translation_acceleration_cap_mps2=1.5,
        rotation_acceleration_cap_radps2=2.0,
    )
    result = _compute(
        guidance,
        reference_position_world=torch.tensor([[100.0, 100.0, 0.0]]),
        reference_quaternion_wxyz=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    assert torch.linalg.vector_norm(result.force_world, dim=-1).item() <= 3.0 + 1.0e-6
    assert torch.linalg.vector_norm(result.torque_world, dim=-1).item() <= 8.0 + 1.0e-6
    assert bool(result.force_clipped.item())
    assert bool(result.torque_clipped.item())


def test_deadbands_suppress_tiny_guidance_without_mutating_inputs() -> None:
    guidance = _guidance(position_deadband_m=0.01, rotation_deadband_rad=0.1)
    reference_position = torch.tensor([[0.009, 0.0, 0.0]])
    reference_quaternion = torch.tensor([[math.cos(0.04), 0.0, 0.0, math.sin(0.04)]])
    before_position = reference_position.clone()
    before_quaternion = reference_quaternion.clone()
    result = _compute(
        guidance,
        reference_position_world=reference_position,
        reference_quaternion_wxyz=reference_quaternion,
    )
    assert torch.equal(result.force_world, torch.zeros((1, 3)))
    assert torch.equal(result.torque_world, torch.zeros((1, 3)))
    assert torch.equal(reference_position, before_position)
    assert torch.equal(reference_quaternion, before_quaternion)


def test_none_mode_is_exact_zero_even_with_large_error() -> None:
    guidance = ReferenceWrenchGuidance(ObjectGuidanceContractV1(mode="none"))
    result = _compute(
        guidance,
        reference_position_world=torch.tensor([[10.0, 0.0, 0.0]]),
        reference_twist_world=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 10.0]]),
    )
    assert torch.equal(result.force_world, torch.zeros((1, 3)))
    assert torch.equal(result.torque_world, torch.zeros((1, 3)))
    assert not bool(result.force_clipped.item())
    assert not bool(result.torque_clipped.item())


def test_nonfinite_input_fails_closed() -> None:
    with pytest.raises(ValueError, match="OBJECT_GUIDANCE_NONFINITE_INPUT"):
        _compute(_guidance(), reference_position_world=torch.tensor([[float("nan"), 0.0, 0.0]]))
