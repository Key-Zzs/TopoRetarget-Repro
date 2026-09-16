from __future__ import annotations

import inspect
import json

import pytest

from scripts.evaluation import run_oakink2_o5rd2j_d2k as d2jk


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_contains_every_contract_action() -> None:
    assert set(d2jk.REQUIRED_ACTIONS) == set(d2jk.ACTIONS)
    help_text = d2jk.parser().format_help()
    assert "preflight" in help_text
    assert "summarize" in help_text


def test_visual_and_collision_lineages_are_separate() -> None:
    source = inspect.getsource(d2jk.build_oakink2_object_physics_authority)
    assert '"role": "VISUAL_GEOMETRY"' in source
    assert '"role": "PHYSX_COLLISION_GEOMETRY"' in source
    assert '"representation": "convex_hull_v1"' in source


def test_dynamics_contract_is_anchor_independent_existing_proxy() -> None:
    source = inspect.getsource(d2jk.build_oakink2_object_physics_authority)
    assert '"provenance_class": "EXISTING_PROJECT_GENERIC_PROXY"' in source
    assert '"profile": "fixed_0p05kg_mesh_bbox_inertia_v1"' in source
    assert '"anchor_dependent": False' in source


def test_support_hash_is_deterministic() -> None:
    payload = {"support": "UNRESOLVED", "pose": [1.0, 2.0, 3.0]}
    assert d2jk.stable_hash(payload) == d2jk.stable_hash(dict(reversed(list(payload.items()))))


def test_support_receipt_compaction_preserves_decision_and_hashes_series() -> None:
    payload = {
        "status": "SUPPORT_UNRESOLVED",
        "stable_interval": {
            "status": "PLANAR_SUPPORT_INFERENCE_NOT_AUTHORIZED",
            "linear_speed_mps": [0.0, 0.1],
            "angular_speed_radps": [0.0, 0.2],
            "translation_step_m": [0.0, 0.01],
            "rotation_step_rad": [0.0, 0.02],
            "candidate_intervals": [{"start": 4, "authorized": False}],
        },
        "diagnostics": {
            "reason": "no_stable_interval_before_manipulation",
            "candidate_interval_audit": [{"support_inference_authorized": False, "start": 4}],
        },
    }
    compact = d2jk.compact_support_result(payload)
    assert compact["status"] == "SUPPORT_UNRESOLVED"
    stable = compact["stable_interval"]
    assert "linear_speed_mps" not in stable
    assert stable["kinematic_series_summary"]["linear_speed_mps"]["count"] == 2
    assert compact["diagnostics"]["all_candidate_intervals_authority"].startswith(
        "POST_MANIPULATION"
    )


def test_anchor_qualification_rejects_unresolved_support(tmp_path) -> None:
    _write(
        tmp_path / "object_physics/oakink2_object_physics_authority.json",
        {"status": "PASS"},
    )
    _write(tmp_path / "support/oakink2_support_authority.json", {"status": "FAIL"})
    _write(
        tmp_path / "scene_adapter/oakink2_static_scene_adapter_contract.json",
        {"status": "NOT_RUN"},
    )
    with pytest.raises(RuntimeError, match="QUALIFY_PHYSICAL_STUDY_ANCHORS_REJECTED"):
        d2jk.qualify_physical_study_anchors(tmp_path)


def test_d2j_freeze_requires_hocap_and_oakink2_smokes(tmp_path) -> None:
    _write(tmp_path / "hocap_regression/parity.json", {"status": "NOT_RUN"})
    _write(tmp_path / "d2j_baseline_smoke/qualification.json", {"status": "NOT_RUN"})
    with pytest.raises(RuntimeError, match="FREEZE_D2J_REJECTED"):
        d2jk.freeze_d2j_physical_contracts(tmp_path)


def test_ppo_study_rejects_nonpass_d2j(tmp_path) -> None:
    _write(tmp_path / "final_summary.json", {"status": "FAIL", "D2J_STATUS": "FAIL"})
    with pytest.raises(RuntimeError, match="PPO_STUDY_REJECTED"):
        d2jk.run_ppo_recoverability_study(tmp_path)


def test_static_scene_contract_has_no_cheat_invariants(tmp_path) -> None:
    _write(
        tmp_path / "object_physics/oakink2_object_physics_authority.json",
        {"status": "PASS"},
    )
    _write(tmp_path / "support/oakink2_support_authority.json", {"status": "FAIL"})
    with pytest.raises(RuntimeError, match="BLOCKED_SUPPORT_AUTHORITY"):
        d2jk.build_oakink2_static_scene_adapter(tmp_path)
    schema = json.loads(
        (tmp_path / "scene_adapter/canonical_static_contact_record_schema.json").read_text()
    )
    invariants = schema["invariants"]
    assert invariants["object_pose_writes_after_reset"] == 0
    assert invariants["hidden_object_force"] is False
    assert invariants["wrist_root_teleport"] is False


def test_workflow_has_no_retarget_or_reward_mutation_entrypoint() -> None:
    source = inspect.getsource(d2jk)
    forbidden = (
        "search_cold_start_v2_frame(",
        "search_cold_start_frame(",
        "solve_frame(",
        "scipy.optimize.minimize",
        "penetration_reward",
        "git add .",
        "git add -A",
    )
    assert all(token not in source for token in forbidden)


def test_support_authority_forbids_manual_plane_and_post_outcome_adjustment() -> None:
    source = inspect.getsource(d2jk.build_oakink2_support_authority)
    assert '"manual_z0_plane_created": False' in source
    assert '"outcome_dependent_adjustment": False' in source
    assert '"post_manipulation_candidates_are_diagnostic_only": True' in source


def test_gpu_serialization_and_fixed_anchor_manifest_are_explicit() -> None:
    source = inspect.getsource(d2jk)
    assert '"max_concurrent_gpu_jobs": 1' in source
    assert '"immutable_anchor_selection": True' in source
    assert '"selection_replaced": False' in source
