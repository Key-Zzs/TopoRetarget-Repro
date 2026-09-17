from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from scripts.evaluation import run_oakink2_o5rd2kr_decision_tree as study


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_pre_d2k_cutoff_and_post_d2k_contract_are_distinct() -> None:
    assert study.CUTOFF_HEAD == "f69a5540a82113f3803bebe0e3689baca7a4d69b"
    assert study.KNOWN_D2K_HEAD == "7b54d4bc5aa4fd72520bba0cd214bfc24f93df05"
    assert study.CUTOFF_HEAD != study.KNOWN_D2K_HEAD


def test_v2_binary_contract_uses_preexisting_units_and_is_prospective() -> None:
    contract = study._v2_contract()
    gate = contract["rollout_gate"]
    assert contract["status"] == "FROZEN_BEFORE_V2_ANCHOR_SELECTION"
    assert contract["current_d2k_outcome_distribution_used"] is False
    assert gate["object_translation_error_m_mean_lt"] == 0.03
    assert gate["object_rotation_error_rad_mean_lt"] == pytest.approx(0.5235987755982988)
    assert gate["physical_penetration_p95_m_lte"] == 0.003
    assert gate["physical_penetration_max_m_lt"] == 0.010
    assert contract["anchor_gate"]["minimum_passing_rollouts"] == 8


def test_fresh_v2_selection_has_exact_groups_and_no_v1_overlap() -> None:
    selected, counts = study._select_fresh_rows()
    old = study.read_json(study.D2I_ROOT / "study_manifest/recoverability_manifest.json")
    old_ordinals = {int(row["ordinal"]) for row in old["anchors"]}
    selected_ordinals = {int(row["ordinal"]) for row in selected}
    assert counts == {"GROUP_A": 2, "GROUP_B": 12, "GROUP_C": 2, "GROUP_D": 2}
    assert len(selected) == 12
    assert old_ordinals.isdisjoint(selected_ordinals)
    assert {group: sum(row["group"] == group for row in selected) for group in counts} == {
        "GROUP_A": 2,
        "GROUP_B": 6,
        "GROUP_C": 2,
        "GROUP_D": 2,
    }


def test_existing_rescore_rejects_partial_gate(tmp_path: Path) -> None:
    _write(
        tmp_path / "gate_audit/gate_alignment_decision.json",
        {"status": "PASS", "PRE_D2K_BINARY_GATE_STATUS": "PARTIALLY_APPLICABLE"},
    )
    with pytest.raises(RuntimeError, match="NOT_FULLY_APPLICABLE"):
        study.rescore_existing_d2k_rollouts(tmp_path)


def test_v2_freeze_rejects_manifest_selected_too_early(tmp_path: Path) -> None:
    _write(
        tmp_path / "recoverability_v2/binary_contract.draft.json",
        study._v2_contract(),
    )
    _write(tmp_path / "recoverability_v2/manifest.json", {"status": "EARLY"})
    with pytest.raises(RuntimeError, match="MANIFEST_ALREADY_EXISTS"):
        study.freeze_recoverability_v2(tmp_path)


@pytest.mark.parametrize(
    ("selected", "requested"),
    [("STRONG", "MIXED"), ("STRONG", "POOR"), ("MIXED", "STRONG"), ("POOR", "STRONG")],
)
def test_terminal_branch_exclusivity(tmp_path: Path, selected: str, requested: str) -> None:
    _write(
        tmp_path / "recoverability_v2/decision.json",
        {"status": "COMPLETE", "SELECTED_TERMINAL_BRANCH": selected},
    )
    with pytest.raises(RuntimeError, match="SELECTED_BRANCH"):
        study._require_branch(tmp_path, requested, "TEST_BRANCH")


def test_cold_start_authority_has_no_q_old() -> None:
    selected, _counts = study._select_fresh_rows()
    for row in selected:
        receipt = study.D2H_ROOT / f"sparse_v4/receipts/frame_{int(row['ordinal']):04d}_run_1.npz"
        import numpy as np

        with np.load(receipt, allow_pickle=False) as archive:
            assert sorted(archive.files) == ["base_pose_scene", "qpos"]


def test_action_surface_exposes_every_contract_cli() -> None:
    assert tuple(study.ACTIONS) == study.REQUIRED_ACTIONS


def test_execution_v4_candidate_family_is_small_ordered_and_budget_frozen() -> None:
    candidates = study.default_cold_start_search_v4_candidates()
    assert [candidate.name for candidate in candidates] == [
        "V4_A_TOP2_SEQUENTIAL",
        "V4_B_TOP2_JOINT_BLOCK",
        "V4_C_ADAPTIVE_K3",
    ]
    assert [candidate.top_k for candidate in candidates] == [2, 2, 3]
    assert all(candidate.top_k <= 3 for candidate in candidates)
    assert all(candidate.contributor_probe_max_nfev == 24 for candidate in candidates)
    assert all(candidate.selected_primary_maxiter == 8 for candidate in candidates)
    assert all(candidate.secondary_polish_maxiter == 8 for candidate in candidates)


def test_poor_branch_cannot_execute_admission_policy(tmp_path: Path) -> None:
    _write(
        tmp_path / "recoverability_v2/decision.json",
        {"status": "COMPLETE", "SELECTED_TERMINAL_BRANCH": "POOR"},
    )
    with pytest.raises(RuntimeError, match="SELECTED_BRANCH=POOR"):
        study.build_retarget_admission_v2(tmp_path)


def test_v4_receipt_contract_forbids_q_old_and_hidden_carriers() -> None:
    source = inspect.getsource(study.search_cold_start_v4_from_v3)
    assert '"old_production_q": "ABSENT"' in source
    assert '"q_old_synthesized": False' in source
    assert '"failed_stage7_terminal_used_as_q_old": False' in source
    assert "object_pose" not in source
    assert "apply_force" not in source
    assert "wrist_teleport" not in source


def test_sparse_v5_excludes_all_prior_and_v4_development_frames() -> None:
    manifest = study.read_json(study.ROOT / "poor_branch/sparse_v5/manifest.json")
    selected = {int(row["ordinal"]) for row in manifest["frames"]}
    excluded, _counts = study._v4_frame_exclusions()
    assert len(selected) == 30
    assert selected.isdisjoint(excluded)
    assert manifest["overlap_with_all_historical_and_v4_development"] == 0


def test_window_v5_has_zero_overlap_with_every_frozen_predecessor() -> None:
    manifest = study.read_json(study.ROOT / "poor_branch/window_v5/manifest.json")
    selected = {int(ordinal) for window in manifest["windows"] for ordinal in window["ordinals"]}
    sparse = {
        int(row["ordinal"])
        for row in study.read_json(study.ROOT / "poor_branch/sparse_v5/manifest.json")["frames"]
    }
    excluded, _counts = study._v4_frame_exclusions()
    assert len(selected) == 128
    assert selected.isdisjoint(sparse | excluded)
    assert manifest["overlaps"] == {
        "internal": 0,
        "prior_and_v4_development": 0,
        "sparse_v5": 0,
    }


def test_window_failure_fail_closes_cross_episode_and_dev2() -> None:
    summary = study.read_json(study.ROOT / "final_summary.json")
    cross = study.read_json(study.ROOT / "poor_branch/cross_episode_v5/not_run.json")
    authorization = study.read_json(study.ROOT / "poor_branch/dev2_full/authorization.json")
    assert summary["WINDOW_V5"] == "FAIL"
    assert cross["episodes_consumed"] == 0
    assert cross["certification_split_consumed"] == 0
    assert cross["heldout_split_consumed"] == 0
    assert authorization["DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED"] == "NO"
    assert authorization["DEV2_FULL_GEOMETRIC_SOLVE_COUNT"] == 0
