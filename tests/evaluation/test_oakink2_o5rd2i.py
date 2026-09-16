from __future__ import annotations

import inspect
import json

from scripts.evaluation import run_oakink2_o5rd2i as d2i


def test_outcome_groups_are_threshold_and_hard_validity_bound() -> None:
    base = {"semantic_hard_pass": True}
    assert d2i.outcome_group({**base, "old_e_im": 2e-4, "new_e_im": 9e-5}) == "GROUP_A"
    assert d2i.outcome_group({**base, "old_e_im": 2e-4, "new_e_im": 2e-4}) == "GROUP_B"
    assert d2i.outcome_group({**base, "old_e_im": 9e-5, "new_e_im": 9e-5}) == "GROUP_C"
    assert d2i.outcome_group({**base, "old_e_im": 9e-5, "new_e_im": 2e-4}) == "GROUP_D"
    assert (
        d2i.outcome_group({"semantic_hard_pass": False, "old_e_im": 9e-5, "new_e_im": 9e-5})
        == "GROUP_D"
    )


def test_deterministic_anchor_indices_cover_low_median_high() -> None:
    assert d2i._quantile_indices(5) == [1, 2, 3]
    assert d2i._quantile_indices(15) == [4, 7, 11]


def test_candidate_validity_requires_interaction_and_hard_validity() -> None:
    assert d2i._candidate_valid({"whole_e_im": 9e-5, "independent_feasible": True})
    assert not d2i._candidate_valid({"whole_e_im": 9e-5, "independent_feasible": False})
    assert not d2i._candidate_valid({"whole_e_im": 2e-4, "independent_feasible": True})


def test_driver_has_no_retarget_optimizer_entrypoint() -> None:
    source = inspect.getsource(d2i)
    forbidden = (
        "search_cold_start_v2_frame(",
        "search_cold_start_frame(",
        "solve_frame(",
        "scipy.optimize.minimize",
        "scipy.optimize.least_squares",
    )
    assert all(token not in source for token in forbidden)
    assert 'retarget_optimizer_run_count": 0' in source


def test_reward_contract_is_hash_frozen_and_has_no_penetration_reward(tmp_path) -> None:
    result = d2i.freeze_ppo_study_contract(tmp_path)
    reward = json.loads((tmp_path / "ppo_authority/reward_contract.json").read_text())
    assert result["PPO_REWARD_CHANGED"] == "NO"
    assert result["REWARD_CONTRACT_SHA256"]
    assert reward["penetration_metric_role"] == "EVALUATION_ONLY_NOT_REWARD"
    assert reward["sparsev4_specific_term"] is False
    assert reward["object_specific_term"] is False


def test_blocked_physical_manifest_cannot_start_baseline(tmp_path) -> None:
    manifest = {
        "schema_version": "RetargetToPPORecoverabilityStudyV1",
        "status": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
    }
    path = tmp_path / "study_manifest/recoverability_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    result = d2i.run_reference_hold_baselines(tmp_path)
    assert result["status"] == "NOT_RUN"
    assert result["gpu_jobs_started"] == 0
    assert result["ppo_updates"] == 0


def test_cli_contract_contains_every_required_action() -> None:
    required = {
        "preflight",
        "verify-frozen-retarget-evidence",
        "analyze-sparsev4-failures",
        "analyze-penetration",
        "analyze-contributor-locality",
        "analyze-bootstrap-basin",
        "decide-sparsev4-root-cause",
        "audit-ppo-authority",
        "freeze-ppo-study-contract",
        "build-physical-study-eligible-pool",
        "freeze-recoverability-manifest",
        "run-reference-hold-baselines",
        "run-ppo-recoverability-study",
        "evaluate-ppo-recoverability",
        "analyze-retarget-to-ppo",
        "render-recoverability-review",
        "summarize",
    }
    assert required <= d2i.ACTIONS.keys()
