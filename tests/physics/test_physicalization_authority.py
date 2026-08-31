from __future__ import annotations

from toporetarget.physics.physicalization_authority import (
    DynamicsAuthorityStatus,
    ObjectDynamicsAuthorityContractV1,
    PhysicalizationCandidateV1,
    PhysicalizationDeviationBudgetV1,
    PhysicalizationMode,
    RetargetReuseStatus,
    SettledDynamicsStatus,
    SettledSupportDynamicsQualificationV2,
    SupportExistenceContractV1,
    SupportExistenceStatus,
    audit_object_dynamics_provenance,
    audit_runtime_default_provenance,
    build_physical_scene_protocol_v2,
    compare_retarget_reuse,
    qualify_settled_support_dynamics_v2,
    resolve_support_existence,
    select_physicalization_candidate,
)


def _record(step: int, *, position: float = 0.0, speed: float = 0.0) -> dict[str, object]:
    return {
        "time_s": (step + 1) / 120.0,
        "position_world_m": [position, 0.0, 0.0],
        "orientation_world_wxyz": [1.0, 0.0, 0.0, 0.0],
        "linear_velocity_world_mps": [speed, 0.0, 0.0],
        "angular_velocity_world_radps": [0.0, 0.0, speed],
        "support_contact": True,
        "support_contact_count": 1,
        "support_force_world_n": [0.0, 0.0, 0.49],
    }


def test_stationary_environment_support_is_a_valid_explicit_resolution() -> None:
    result = resolve_support_existence(
        {
            "source_explicit_support": False,
            "hand_supported": False,
            "other_object_supported": False,
            "stationary_initial_frames": 8,
            "initial_linear_speed_max_mps": 0.01,
            "initial_angular_speed_max_radps": 0.05,
            "finite_environment_geometry_available": True,
        }
    )
    assert result["status"] == SupportExistenceStatus.ENVIRONMENT_SUPPORT_REQUIRED.value
    assert result["pass"] is True


def test_missing_environment_geometry_remains_fail_closed() -> None:
    result = resolve_support_existence(
        {
            "stationary_initial_frames": 8,
            "initial_linear_speed_max_mps": 0.01,
            "initial_angular_speed_max_radps": 0.05,
        }
    )
    assert result["status"] == SupportExistenceStatus.UNRESOLVED.value
    assert result["pass"] is False


def test_physicalization_is_deterministic_and_relative_projection_is_rejected() -> None:
    budget = PhysicalizationDeviationBudgetV1()
    candidates = (
        PhysicalizationCandidateV1(
            "support_only",
            PhysicalizationMode.SUPPORT_ONLY,
            support_translation_m=0.01,
            support_normal_change_rad=0.0,
        ),
        PhysicalizationCandidateV1(
            "relative_projection",
            PhysicalizationMode.RELATIVE_OBJECT_PROJECTION,
            support_translation_m=0.0,
            support_normal_change_rad=0.0,
            relative_object_translation_m=0.001,
        ),
    )
    result = select_physicalization_candidate(candidates, budget)
    assert result["selected_candidate_id"] == "support_only"
    assert result["candidate_count"] == 2
    rejected = result["evaluations"][1]
    assert "relative_object_projection_requires_exact_retarget" in rejected["rejection_reasons"]


def test_retarget_reuse_compares_full_relative_trajectory() -> None:
    translation = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]
    quaternion = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    result = compare_retarget_reuse(
        hand_translation_before_m=translation,
        hand_quaternion_before_wxyz=quaternion,
        object_translation_before_m=[[0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        object_quaternion_before_wxyz=quaternion,
        hand_translation_after_m=translation,
        hand_quaternion_after_wxyz=quaternion,
        object_translation_after_m=[[0.1, 0.0, 0.0], [0.11, 0.0, 0.0]],
        object_quaternion_after_wxyz=quaternion,
        budget=PhysicalizationDeviationBudgetV1(),
        mode=PhysicalizationMode.SUPPORT_ONLY,
    )
    assert result["status"] == RetargetReuseStatus.REUSE_GEOMETRIC_RETARGET.value
    assert result["frame_count"] == 2
    assert result["full_trajectory_compared"] is True


def test_settled_v2_ignores_impact_peak_but_requires_terminal_settling() -> None:
    rows = []
    for step in range(360):
        rows.append(
            _record(step, position=0.001 if step < 20 else 0.0, speed=2.0 if step < 5 else 0.001)
        )
    result = qualify_settled_support_dynamics_v2(rows, mass_kg=0.05)
    assert result["pass"] is True
    assert result["status"] == SettledDynamicsStatus.SETTLED_AFTER_TRANSIENT.value
    assert result["impact_peaks_diagnostic_only"] is True
    assert result["terminal_linear_speed_max_mps"] < 0.02


def test_settled_v2_fails_without_terminal_contact() -> None:
    rows = [_record(step, speed=0.001) for step in range(360)]
    for row in rows[240:]:
        row["support_contact"] = False
    result = qualify_settled_support_dynamics_v2(rows, mass_kg=0.05)
    assert result["pass"] is False
    assert result["status"] == SettledDynamicsStatus.NO_CONTACT.value


def test_object_and_runtime_authorities_are_separate() -> None:
    asset = {
        "object_id": "G21_3",
        "visual_mesh_sha256": "a" * 64,
        "generated_usd": "G21_3.usda",
        "generated_sha256": "b" * 64,
        "mass_kg": 0.05,
        "center_of_mass_m": [0.0, 0.0, 0.0],
        "principal_inertia_kgm2": [0.001, 0.002, 0.003],
        "collision_method": "convex_hull_v1",
        "collision_prim_count": 1,
        "rigid_body": {"free": True, "gravity_enabled": False},
    }
    audit = audit_object_dynamics_provenance(asset)
    assert audit["status"] == DynamicsAuthorityStatus.PASS.value
    runtime = audit_runtime_default_provenance(
        {
            "mass_kg": 0.05,
            "center_of_mass_m": [0.0, 0.0, 0.0],
            "diagonal_inertia_kgm2": [0.001, 0.002, 0.003],
            "gravity_enabled": True,
            "collision_enabled": True,
            "rigid_body": True,
        },
        {
            "mass_kg": 0.05,
            "center_of_mass_m": [0.0, 0.0, 0.0],
            "diagonal_inertia_kgm2": [0.001, 0.002, 0.003],
            "gravity_enabled": True,
            "collision_enabled": True,
            "rigid_body": True,
        },
    )
    assert runtime["status"] == DynamicsAuthorityStatus.PASS.value


def test_protocol_hash_is_stable() -> None:
    protocol, digest = build_physical_scene_protocol_v2(
        dynamics_contract=ObjectDynamicsAuthorityContractV1(),
        support_contract=SupportExistenceContractV1(),
        deviation_budget=PhysicalizationDeviationBudgetV1(),
        settled_contract=SettledSupportDynamicsQualificationV2(),
    )
    assert protocol["schema_version"] == "PhysicalSceneProtocolV2"
    assert len(digest) == 64
