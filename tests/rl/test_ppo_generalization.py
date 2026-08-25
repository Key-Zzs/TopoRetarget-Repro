from __future__ import annotations

import math

import pytest

from toporetarget.rl.ppo_generalization import (
    INTERACTION_PHASES,
    PHASES,
    CappedSequenceBudgetV1,
    DimensionlessObjectScaleV1,
    DimensionlessScaledGroupedRewardV1,
    DimensionlessScaledReferenceScopedExplorationV1,
    EpisodeV1RuntimeEvents,
    UniformEventBalancedRSIV1,
    map_source_event_to_runtime,
    object_bbox_diagonal_m,
)


def test_length_based_budget_is_deterministic_and_capped() -> None:
    contract = CappedSequenceBudgetV1(
        samples_per_valid_index=500.0,
        samples_per_interaction_index=800.0,
        global_sample_cap=1_024_000,
        samples_per_update=40_960,
    )
    first = contract.derive(valid_indices=1_121, interaction_indices=986)
    second = contract.derive(valid_indices=1_121, interaction_indices=986)
    assert first == second == {"updates": 20, "total_samples": 819_200, "capped": 0}

    capped = contract.derive(valid_indices=4_000, interaction_indices=3_000)
    assert capped == {"updates": 25, "total_samples": 1_024_000, "capped": 1}


def test_episode_events_produce_complete_non_overlapping_phase_accounting() -> None:
    events = EpisodeV1RuntimeEvents(
        reference_length=21,
        approach=2,
        contact=4,
        pickup=7,
        place=15,
        release=17,
        retreat=19,
    )
    labels = events.phase_labels()
    assert len(labels) == 21
    assert set(labels) == set(PHASES)
    assert sum(label in INTERACTION_PHASES for label in labels) == 15
    assert labels[0] == "PRE/IDLE"
    assert labels[7] == "PICKUP"
    assert labels[20] == "RETREAT"


def test_uniform_component_is_mandatory_and_frame0_only_is_forbidden() -> None:
    contract = UniformEventBalancedRSIV1(uniform_alpha=0.5)
    assert contract.as_dict() == {
        "uniform_alpha": 0.5,
        "event_balanced_alpha": 0.5,
        "interaction_source": "EpisodeV1_events_CONTACT_through_RELEASE",
        "uniform_component_preserved": True,
        "frame0_only": False,
    }
    with pytest.raises(ValueError, match="UNIFORM_COMPONENT_REQUIRED"):
        UniformEventBalancedRSIV1(uniform_alpha=0.0)


def test_object_scale_normalization_is_global_not_object_tuned(tmp_path) -> None:
    mesh = tmp_path / "box.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 2 0\nv 0 0 2\nf 1 2 3\n", encoding="utf-8")
    scale = object_bbox_diagonal_m(mesh)
    assert scale == pytest.approx(3.0)

    contract = DimensionlessObjectScaleV1(anchor_bbox_diagonal_m=0.2)
    ratios = contract.dimensionless_ratios()
    assert ratios["proximity_tolerance_over_s_o"] == pytest.approx(0.15)
    small = contract.thresholds(0.1)
    large = contract.thresholds(0.3)
    assert small["proximity_tolerance_m"] == pytest.approx(0.015)
    assert large["proximity_tolerance_m"] == pytest.approx(0.045)
    assert small["object_tracking_sigma_m"] == pytest.approx(0.02)
    assert large["object_velocity_sigma_mps"] == pytest.approx(0.1125)
    assert math.isclose(
        small["proximity_tolerance_m"] / 0.1,
        large["proximity_tolerance_m"] / 0.3,
    )


def test_event_mapping_preserves_inclusive_endpoints() -> None:
    assert (
        map_source_event_to_runtime(
            749, source_start_frame=749, source_end_frame=959, runtime_reference_length=1_121
        )
        == 0
    )
    assert (
        map_source_event_to_runtime(
            959, source_start_frame=749, source_end_frame=959, runtime_reference_length=1_121
        )
        == 1_120
    )


def test_scaled_runtime_contracts_keep_grouped_and_rse_structure() -> None:
    grouped = DimensionlessScaledGroupedRewardV1(
        identifier="Stage16GroupedMultiplicativeRewardV1+P3ObjectScaleV1",
        distance_scope_m=0.24,
        proximity_tolerance_m=0.036,
        object_bbox_diagonal_m=0.2616,
        anchor_bbox_diagonal_m=0.218,
        proximity_scale_per_m=1.0 / 0.036,
    )
    assert grouped.object_exponent == grouped.hand_exponent == 1.0

    rse = DimensionlessScaledReferenceScopedExplorationV1(
        identifier="Stage16ReferenceScopedExplorationV1+P3ObjectScaleV1",
        enabled=True,
        distance_relaxation=True,
        adaptive_termination=True,
        distance_scope_m=0.24,
        object_position_base_m=0.06,
        object_axis_base_m=0.06,
        proximity_tolerance_m=0.036,
        object_bbox_diagonal_m=0.2616,
        anchor_bbox_diagonal_m=0.218,
    )
    assert rse.hand_position_base_m == 0.20
    assert rse.kappa_min == 0.50


def test_scaled_runtime_contracts_fail_closed_on_structure_drift() -> None:
    with pytest.raises(ValueError, match="GROUPED_STRUCTURE_DRIFT"):
        DimensionlessScaledGroupedRewardV1(
            identifier="Stage16GroupedMultiplicativeRewardV1+P3ObjectScaleV1",
            distance_scope_m=0.24,
            proximity_tolerance_m=0.036,
            object_bbox_diagonal_m=0.2616,
            anchor_bbox_diagonal_m=0.218,
            proximity_scale_per_m=1.0 / 0.036,
            object_exponent=2.0,
        )
