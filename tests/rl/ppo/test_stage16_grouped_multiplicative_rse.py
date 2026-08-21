"""Pure tests for grouped multiplicative reward and reference-scoped exploration."""

from __future__ import annotations

import pytest
import torch

from toporetarget.rl.reference_tracking.grouped_multiplicative_reward import (
    GroupedMultiplicativeRewardV1,
    grouped_multiplicative_reward_v1_terms,
    point_to_triangle_surface_distance,
    reference_scope_weight,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV4,
    ppo26d_reward_v4_strict_per_finger_contact_terms,
)
from toporetarget.rl.reference_tracking.reference_scoped_exploration import (
    AdaptiveScopeStateV1,
    adaptive_kappa,
    rse_deviation_termination,
)


def _inputs(batch: int = 4) -> dict[str, torch.Tensor | TopoRetargetReferenceTrackingReward26DV4]:
    zeros3 = torch.zeros((batch, 3))
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(batch, -1)
    return {
        "reference_fingertip_surface_distance_m": torch.full((batch, 5), 0.20),
        "actual_fingertip_surface_distance_m": torch.full((batch, 5), 0.03),
        "source_contact_mask": torch.tensor([[True, False, False, False, False]]).expand(batch, -1),
        "fingertip_object_pair_force_world": torch.tensor([[[1000.0, 0.0, 0.0]] * 5]).expand(
            batch, -1, -1
        ),
        "fingertip_object_pair_presence": torch.ones((batch, 5), dtype=torch.bool),
        "object_twist_world": torch.zeros((batch, 6)),
        "object_twist_world_ref": torch.zeros((batch, 6)),
        "object_axis_points": torch.zeros((batch, 6, 3)),
        "object_axis_points_ref": torch.zeros((batch, 6, 3)),
        "tracked_links": torch.zeros((batch, 16, 3)),
        "tracked_links_ref": torch.zeros((batch, 16, 3)),
        "finger_q": torch.zeros((batch, 20)),
        "finger_q_ref": torch.zeros((batch, 20)),
        "joint_lower": torch.full((20,), -1.0),
        "joint_upper": torch.full((20,), 1.0),
        "wrist_position": zeros3,
        "wrist_quaternion_wxyz": identity,
        "wrist_position_ref": zeros3,
        "wrist_quaternion_ref_wxyz": identity,
        "action": torch.zeros((batch, 26)),
        "previous_action": torch.zeros((batch, 26)),
        "second_previous_action": torch.zeros((batch, 26)),
        "profile": TopoRetargetReferenceTrackingReward26DV4(
            contact_force_scale_lambda_tip_n=0.5766498904285564
        ),
    }


def test_group_rewards_are_bounded_perfect_near_one_and_soft_and() -> None:
    terms = grouped_multiplicative_reward_v1_terms(**_inputs())
    for name in ("R_obj", "R_hand", "R_int", "R_reg", "total"):
        assert torch.all(terms[name] > 0.0)
        assert torch.all(terms[name] <= 1.0)
    assert torch.allclose(terms["R_obj"], torch.ones(4))
    assert torch.allclose(terms["R_hand"], torch.ones(4))
    assert torch.allclose(terms["R_reg"], torch.ones(4))
    minimum = torch.stack([terms[name] for name in ("R_obj", "R_hand", "R_int", "R_reg")]).amin(0)
    assert torch.all(terms["total"] <= minimum + 1.0e-7)


def test_legacy_additive_total_is_numerically_unchanged() -> None:
    values = _inputs()
    legacy = ppo26d_reward_v4_strict_per_finger_contact_terms(
        **{
            name: value
            for name, value in values.items()
            if name
            not in {
                "reference_fingertip_surface_distance_m",
                "actual_fingertip_surface_distance_m",
            }
        }
    )
    grouped = grouped_multiplicative_reward_v1_terms(**values)
    assert torch.equal(grouped["total_legacy_additive"], legacy["total"])


def test_each_group_degradation_lowers_total_without_compensation() -> None:
    perfect = grouped_multiplicative_reward_v1_terms(**_inputs())["total"]
    hand = _inputs()
    hand["finger_q"] = torch.full((4, 20), 0.5)
    obj = _inputs()
    obj["object_twist_world"] = torch.full((4, 6), 0.1)
    interaction = _inputs()
    interaction["fingertip_object_pair_force_world"] = torch.zeros((4, 5, 3))
    interaction["fingertip_object_pair_presence"] = torch.zeros((4, 5), dtype=torch.bool)
    interaction["actual_fingertip_surface_distance_m"] = torch.full((4, 5), 0.20)
    assert torch.all(grouped_multiplicative_reward_v1_terms(**hand)["total"] < perfect)
    assert torch.all(grouped_multiplicative_reward_v1_terms(**obj)["total"] < perfect)
    degraded = grouped_multiplicative_reward_v1_terms(**interaction)
    assert torch.all(degraded["total"] < perfect)
    assert torch.all(degraded["total"] <= degraded["R_int"] + 1.0e-7)


def test_contact_group_no_contact_contact_near_and_far_are_monotonic() -> None:
    no_reference = _inputs()
    no_reference["source_contact_mask"] = torch.zeros((4, 5), dtype=torch.bool)
    no_reference["actual_fingertip_surface_distance_m"] = torch.full((4, 5), 10.0)
    assert torch.allclose(
        grouped_multiplicative_reward_v1_terms(**no_reference)["R_int"], torch.ones(4)
    )
    contact = grouped_multiplicative_reward_v1_terms(**_inputs())["R_int"]
    near = _inputs()
    near["fingertip_object_pair_force_world"] = torch.zeros((4, 5, 3))
    near["fingertip_object_pair_presence"] = torch.zeros((4, 5), dtype=torch.bool)
    far = dict(near)
    far["actual_fingertip_surface_distance_m"] = torch.full((4, 5), 0.20)
    near_value = grouped_multiplicative_reward_v1_terms(**near)["R_int"]
    far_value = grouped_multiplicative_reward_v1_terms(**far)["R_int"]
    assert torch.all(contact > near_value)
    assert torch.all(near_value > far_value)


def test_reference_scope_is_monotonic_and_near_relaxes_only_hand() -> None:
    distances = torch.tensor([[0.20] * 5, [0.10] * 5, [0.0] * 5])
    _, weight = reference_scope_weight(distances, 0.20)
    assert weight.tolist() == pytest.approx([1.0, 0.5, 0.0])
    far = _inputs(1)
    far["finger_q"] = torch.full((1, 20), 0.5)
    near = dict(far)
    near["reference_fingertip_surface_distance_m"] = torch.zeros((1, 5))
    far_terms = grouped_multiplicative_reward_v1_terms(**far)
    near_terms = grouped_multiplicative_reward_v1_terms(**near)
    assert near_terms["R_hand"].item() > far_terms["R_hand"].item()
    for name in ("R_obj", "R_int", "R_reg"):
        assert torch.equal(near_terms[name], far_terms[name])


def test_log_clamp_prevents_zero_and_nan() -> None:
    values = _inputs(1)
    values["actual_fingertip_surface_distance_m"] = torch.full((1, 5), 1.0e6)
    values["fingertip_object_pair_force_world"] = torch.zeros((1, 5, 3))
    values["fingertip_object_pair_presence"] = torch.zeros((1, 5), dtype=torch.bool)
    terms = grouped_multiplicative_reward_v1_terms(**values)
    assert torch.isfinite(terms["total"]).all()
    assert terms["total"].item() > 0.0


def test_exact_triangle_surface_distance() -> None:
    triangle = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    points = torch.tensor([[0.25, 0.25, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    distance = point_to_triangle_surface_distance(points, triangle, face_chunk_size=1)
    assert distance.tolist() == pytest.approx([2.0, 1.0, 0.0])


def test_kappa_and_adaptive_state_are_clamped_and_directional() -> None:
    assert adaptive_kappa(9, 10).item() == pytest.approx(0.9)
    assert adaptive_kappa(1, 10).item() == pytest.approx(0.5)
    state = AdaptiveScopeStateV1()
    assert state.kappa == 1.0
    state.record(rse_failures=0, normal_completions=9)
    assert state.kappa == 0.5
    state.record(rse_failures=9, normal_completions=0)
    assert state.kappa > 0.5


def test_rse_deviation_tightens_as_failure_ratio_falls() -> None:
    common = {
        "object_position_error_m": torch.tensor([0.04]),
        "object_axis_error_m": torch.tensor([0.0]),
        "object_orientation_error_rad": torch.tensor([0.0]),
        "hand_position_error_m": torch.tensor([0.0]),
        "hand_orientation_error_rad": torch.tensor([0.0]),
        "actual_fingertip_surface_distance_m": torch.full((1, 5), 0.03),
        "source_contact_mask": torch.zeros((1, 5), dtype=torch.bool),
    }
    wide = rse_deviation_termination(kappa=1.0, **common)
    tight = rse_deviation_termination(kappa=0.5, **common)
    assert not wide["rse_deviation_failure"].item()
    assert tight["rse_deviation_failure"].item()


def test_contract_contains_no_phase_gate_and_preserves_uniform_rsi_by_design() -> None:
    contract = GroupedMultiplicativeRewardV1()
    assert contract.object_exponent == contract.interaction_exponent == 1.0
