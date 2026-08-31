from __future__ import annotations

import numpy as np

from toporetarget.physics.physical_scene_authority import (
    ContactState,
    PhysicalSceneAuthorityContractV1,
    PhysicalSceneStatus,
    SupportAuthority,
    SupportExpectation,
    admit_physical_scene,
    classify_contact_state,
    resolve_support_expectation,
    support_collision_policy,
    validate_runtime_collision_shapes,
    validate_support_geometry,
)


def _shape(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "articulation_path": "/World/Robot/r_wrist",
        "link_name": "r_wrist",
        "collision_prim": "/World/Robot/r_wrist/Collision",
        "shape_type": "convex_hull",
        "source_asset": "robot.usd",
        "local_transform": np.eye(4).tolist(),
        "world_transform": np.eye(4).tolist(),
        "collision_enabled": True,
        "rigid_body": True,
    }
    value.update(overrides)
    return value


def _support(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "plane_normal_world": (0.0, 0.0, 1.0),
        "gravity_world_mps2": (0.0, 0.0, -9.81),
        "support_center_world": (0.0, 0.0, 0.0),
        "support_extent_m": (0.2, 0.2),
        "object_footprint_world": (
            (-0.05, -0.05, 0.0),
            (0.05, -0.05, 0.0),
            (0.05, 0.05, 0.0),
            (-0.05, 0.05, 0.0),
        ),
        "center_of_mass_world": (0.0, 0.0, 0.05),
        "object_min_signed_distance_m": 0.0,
        "object_max_signed_distance_m": 0.05,
    }
    value.update(overrides)
    return value


def test_runtime_collision_authority_requires_enabled_rigid_shapes() -> None:
    assert validate_runtime_collision_shapes([_shape()], role="robot")["status"] == "PASS"
    assert (
        validate_runtime_collision_shapes([_shape(collision_enabled=False)], role="robot")["status"]
        == "FAIL"
    )


def test_contact_state_keeps_intended_touch_distinct_from_severe_penetration() -> None:
    assert (
        classify_contact_state(max_penetration_m=0.0, intended_contact=True)
        is ContactState.INTENDED_CONTACT
    )
    assert (
        classify_contact_state(max_penetration_m=0.05, intended_contact=True)
        is ContactState.SEVERE_PENETRATION
    )
    assert (
        classify_contact_state(max_penetration_m=None, intended_contact=False)
        is ContactState.INCONCLUSIVE
    )


def test_support_geometry_negative_controls_fail_closed() -> None:
    contract = PhysicalSceneAuthorityContractV1()
    assert (
        validate_support_geometry(**_support(object_min_signed_distance_m=0.2), contract=contract)[
            "status"
        ]
        == "FAIL"
    )
    assert (
        validate_support_geometry(
            **_support(plane_normal_world=(0.0, 0.0, -1.0)), contract=contract
        )["status"]
        == "FAIL"
    )
    assert (
        validate_support_geometry(
            **_support(center_of_mass_world=(0.2, 0.0, 0.05)), contract=contract
        )["status"]
        == "FAIL"
    )


def test_support_expectation_does_not_turn_hand_support_into_table_support() -> None:
    result = resolve_support_expectation({"hand_supported": True})
    assert result["expectation"] == SupportExpectation.HAND_SUPPORTED.value
    assert result["expectation"] != SupportExpectation.STATIC_ENVIRONMENT_SUPPORT.value


def test_inferred_support_policy_is_pairwise_and_not_global_disable() -> None:
    policy = support_collision_policy(SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT)
    assert policy["object_support_collision"] is True
    assert policy["hand_support_collision"] is False
    assert policy["global_support_collision_disabled"] is False


def test_admission_is_fail_closed_and_ready_requires_all_checks() -> None:
    kwargs = dict(
        runtime_binding_status="PASS",
        robot_collision_status="PASS",
        object_collision_status="PASS",
        collision_filter_status="PASS",
        reset_contact_state=ContactState.NO_CONTACT,
        support_expectation=SupportExpectation.STATIC_ENVIRONMENT_SUPPORT,
        support_authority=SupportAuthority.INFERRED_ENVIRONMENT_SUPPORT,
        support_dynamics_status="PASS",
    )
    assert (
        admit_physical_scene(**kwargs)["status"] == PhysicalSceneStatus.PHYSICAL_SCENE_READY.value
    )
    failed = admit_physical_scene(**(kwargs | {"support_authority": SupportAuthority.UNRESOLVED}))
    assert failed["status"] == PhysicalSceneStatus.SUPPORT_AUTHORITY_UNRESOLVED.value
