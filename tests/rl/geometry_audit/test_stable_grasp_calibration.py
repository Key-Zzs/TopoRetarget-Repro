from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from toporetarget.rl.geometry_audit.calibration_initialization import (
    ObjectCanonicalGraspInitializationV1,
    initialize_object_between_contacts,
    object_canonical_frame,
    refine_balanced_contact_pose,
)
from toporetarget.rl.geometry_audit.stable_grasp_calibration import (
    StableGraspCalibrationActionScheduleV1,
    extract_grasp_topology_families,
    freeze_candidate_matrix,
    qualify_stable_grasp,
)


def _topology() -> dict[str, object]:
    return {
        "clips": {
            "hocap_170105": {
                "required_body_groups": ["index"],
                "optional_body_groups": ["thumb"],
            },
            "hocap_170650": {
                "required_body_groups": ["index", "pinky"],
                "optional_body_groups": [],
            },
        }
    }


def test_topology_families_are_data_derived_and_cover_both_contracts() -> None:
    families = extract_grasp_topology_families(_topology())
    assert [(row.identifier, row.contact_groups) for row in families] == [
        ("multi_finger_enclosure", ("thumb", "index", "pinky")),
        ("thumb_opposition", ("thumb", "index")),
    ]
    assert {clip for row in families for clip in row.applicable_clips} == {
        "hocap_170105",
        "hocap_170650",
    }
    multi = next(row for row in families if row.identifier == "multi_finger_enclosure")
    assert (multi.first_opposition_group, multi.second_opposition_group) == ("thumb", "pinky")


def test_object_canonical_initializer_uses_pca_obb_and_contact_span() -> None:
    vertices = np.array(
        [[x, y, z] for x in (-0.03, 0.03) for y in (-0.02, 0.02) for z in (-0.01, 0.01)],
        dtype=np.float64,
    )
    frame = object_canonical_frame(vertices)
    assert sorted(frame.principal_extents_m) == pytest.approx([0.02, 0.04, 0.06])
    result = initialize_object_between_contacts(
        frame=frame,
        first_contact_center_scene=np.array([-0.03, 0.0, 0.0]),
        second_contact_center_scene=np.array([0.03, 0.0, 0.0]),
        first_contact_vertices_scene=np.array(
            [[-0.035, y, z] for y in (0.0, 0.01) for z in (-0.01, 0.01)]
        ),
        second_contact_vertices_scene=np.array(
            [[0.035, y, z] for y in (0.0, 0.01) for z in (-0.01, 0.01)]
        ),
        palm_center_scene=np.array([0.0, -0.1, 0.0]),
        palm_rotation_scene=np.eye(3),
        approach_offset_m=0.004,
    )
    assert result.opposition_span_m == pytest.approx(0.07)
    assert result.object_opposition_extent_m == pytest.approx(0.06)
    assert result.placement_mode == "inserted_between_contacts"
    assert result.contact_support_plane_scene[1] == pytest.approx(0.01)
    assert result.object_pose_scene_xyz_wxyz[1] > 0.033
    assert result.precontact_clearance_m == pytest.approx(0.003)
    assert not result.corrected_trajectory_used
    assert not result.source_object_pose_used
    assert result.rollout_state_writes == 0


def test_initializer_rejects_corrected_or_source_pose_provenance() -> None:
    with pytest.raises(ValueError, match="cannot use trajectory object poses"):
        ObjectCanonicalGraspInitializationV1(
            object_pose_scene_xyz_wxyz=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            object_pca_axis_for_opposition=0,
            object_pca_axis_for_approach=1,
            opposition_span_m=0.1,
            object_opposition_extent_m=0.1,
            placement_mode="external_approach",
            palm_approach_direction_scene=(0.0, 0.0, 1.0),
            contact_support_plane_scene=(0.0, 0.0, 0.0),
            precontact_clearance_m=0.003,
            approach_offset_m=0.0,
            corrected_trajectory_used=True,
        )


def test_exact_contact_refinement_balances_two_sides_with_bounded_translation() -> None:
    initial = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def query(pose: np.ndarray) -> list[SimpleNamespace]:
        x = float(pose[0])
        return [
            SimpleNamespace(
                signed_separation_m=-0.00005 + x,
                depenetration_direction_for_second=(1.0, 0.0, 0.0),
            ),
            SimpleNamespace(
                signed_separation_m=0.00055 - x,
                depenetration_direction_for_second=(-1.0, 0.0, 0.0),
            ),
        ]

    result = refine_balanced_contact_pose(
        initial_pose_scene_xyz_wxyz=initial,
        selected_groups=("left", "right"),
        group_slots={"left": (0,), "right": (1,)},
        query_pose=query,
    )
    assert result.converged
    assert result.safe_reset
    assert result.translation_correction_m == pytest.approx((0.0003, 0.0, 0.0))
    assert result.selected_signed_separation_after_m == pytest.approx((0.00025, 0.00025))
    assert result.maximum_pair_penetration_after_m == 0.0
    assert not result.corrected_trajectory_used
    assert not result.source_object_pose_used


def test_candidate_matrix_freezes_shared_bounded_c1_rule() -> None:
    families = extract_grasp_topology_families(_topology())
    matrix = freeze_candidate_matrix(
        object_ids=("hocap_170105", "hocap_170650"), families=families, level="C1"
    )
    assert matrix["same_generation_rule_for_all_objects"]
    assert matrix["result_data_observed_before_freeze"] is False
    assert all(len(rows) == 6 for rows in matrix["objects"].values())
    assert len(matrix["contract_sha256"]) == 64


def test_c2_expands_only_the_approach_offset_dimension() -> None:
    families = extract_grasp_topology_families(_topology())
    matrix = freeze_candidate_matrix(
        object_ids=("hocap_170105", "hocap_170650"), families=families, level="C2"
    )
    assert matrix["approach_offsets_m"] == [-0.010, 0.010]
    assert matrix["closure_amplitudes"] == [0.5, 1.0]
    assert matrix["parent_level"] == "C1"
    assert matrix["expanded_dimension"] == "approach_offset_m"
    assert matrix["candidates_are_unique_from_parent"] is True
    assert all(len(rows) == 4 for rows in matrix["objects"].values())


def test_action_schedule_is_321_by_26_bounded_and_has_no_object_field() -> None:
    schedule = StableGraspCalibrationActionScheduleV1()
    actions = schedule.actions(
        contact_groups=("thumb", "index"),
        closure_amplitude=1.0,
        wrist_approach_direction_local=np.array([1.0, 0.0, 0.0]),
    )
    assert actions.shape == (321, 26)
    assert np.max(np.abs(actions)) <= 1.0
    assert np.all(actions[:, 3:6] == 0.0)
    # The selected finger residual is continuous at contact-establishment -> closure.
    assert actions[103, 6] == pytest.approx(-0.5)
    assert actions[104, 6] == pytest.approx(-0.5)
    assert actions[167, 6] == pytest.approx(0.0)
    assert actions[168, 6] == pytest.approx(0.0)
    assert actions[-1, 6] == pytest.approx(-0.4)
    assert np.all(actions[:, 14] == -1.0)
    assert np.all(actions[:, 18] == -1.0)
    assert np.all(actions[:, 22] == -1.0)
    assert schedule.as_dict()["object_action_fields"] == 0


def test_stable_grasp_gate_requires_contact_hold_and_terminal_twist() -> None:
    presence = np.ones((321, 20, 2), dtype=bool)
    twist = np.zeros((321, 20, 6), dtype=np.float64)
    values = np.full(20, 0.001, dtype=np.float64)
    result = qualify_stable_grasp(
        contact_group_presence=presence,
        object_twist=twist,
        finite=np.ones(20, dtype=bool),
        action_bounds_pass=np.ones(20, dtype=bool),
        workspace_pass=np.ones(20, dtype=bool),
        exact_replica_max_penetration_m=values,
        exact_replica_active_p95_m=values,
    )
    assert result["status"] == "STAGE16D_STABLE_GRASP_CALIBRATION_VALIDATED"
    presence[-10:, 0] = False
    failed = qualify_stable_grasp(
        contact_group_presence=presence,
        object_twist=twist,
        finite=np.ones(20, dtype=bool),
        action_bounds_pass=np.ones(20, dtype=bool),
        workspace_pass=np.ones(20, dtype=bool),
        exact_replica_max_penetration_m=values,
        exact_replica_active_p95_m=values,
    )
    assert not failed["passed"]
    assert not failed["hard_gates"]["maximum_contact_loss_20_of_20"]
