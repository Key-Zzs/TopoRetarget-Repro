from __future__ import annotations

import inspect
import json

import pytest

from scripts.evaluation import run_oakink2_o5rd2jr_d2k as study
from toporetarget.physics.support import resolver
from toporetarget.physics.support.static_recoverability_proxy import (
    build_static_recoverability_support_proxy,
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cli_exposes_the_complete_ordered_contract() -> None:
    assert set(study.REQUIRED_ACTIONS) == set(study.ACTIONS)
    help_text = study.parser().format_help()
    assert "run-all-reference-hold-baselines" in help_text
    assert "run-ppo-recoverability" in help_text
    assert "evaluate-ppo-recoverability" in help_text
    assert "summarize" in help_text


def test_source_support_resolver_cannot_emit_study_proxy() -> None:
    source = inspect.getsource(resolver)
    assert "STATIC_RECOVERABILITY_PLANAR_PROXY" not in source
    assert "SOURCE_RECONSTRUCTED_SUPPORT" in source


def test_proxy_builder_has_no_outcome_or_hand_geometry_inputs() -> None:
    signature = inspect.signature(build_static_recoverability_support_proxy)
    assert set(signature.parameters) == {
        "collision_vertices_object",
        "object_pose_world",
        "gravity_world_mps2",
        "parameters",
    }
    source = inspect.getsource(build_static_recoverability_support_proxy)
    for forbidden in (
        "ppo_results",
        "baseline_metrics",
        "e_im",
        "group_label",
        "hand_penetration",
    ):
        assert forbidden not in source


def test_upstream_revalidation_separates_visual_collision_and_dynamics() -> None:
    source = inspect.getsource(study.build_support_proxies)
    assert '"visual_sha256"' in source
    assert '"collision_sha256"' in source
    assert '"collision_representation"' in source
    assert '== "convex_hull_v1"' in inspect.getsource(study.verify_upstream)
    assert '"object_dynamics_changed": False' in source


def test_ppo_training_rejects_missing_or_incomplete_baselines(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="PPO_TRAINING_REJECTED:MISSING"):
        study.run_ppo_recoverability(tmp_path)
    _write(
        tmp_path / "baseline_physics/summary.json",
        {"status": "PASS", "anchors": 11},
    )
    with pytest.raises(RuntimeError, match="PPO_TRAINING_REQUIRES_ALL_12_BASELINES"):
        study.run_ppo_recoverability(tmp_path)


def test_study_freezes_single_gpu_no_cheat_and_unchanged_reward() -> None:
    source = inspect.getsource(study)
    assert '"MAX_GPU_JOBS": 1' in source
    assert '"object_pose_writes_after_reset": 0' in source
    assert '"hidden_object_force_used": False' in source
    assert '"wrist_root_teleport_used": False' in source
    assert '"PPO_REWARD_CHANGED": "NO"' in source
    assert '"same_d2jr_anchors": True' in source
    assert '"adaptive_budget": False' in source


def test_inconclusive_next_step_matches_handoff_contract() -> None:
    source = inspect.getsource(study.analyze_recoverability)
    assert '"NEXT": "RECOVERABILITY_STUDY_V2_DESIGN"' in source
    assert '"binary_recovered": "NOT_DEFINED_CONTINUOUS_METRICS_ONLY"' in source
