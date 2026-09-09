from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.objective_v2_execution import (
    ScreenedCandidate,
    asset_derived_dof_blocks,
    contributor_scores,
    default_search_contracts,
    deterministic_block_seeds,
    rank_contributors,
    retain_after_polish,
    select_candidate,
)


def candidate(
    name: str,
    *,
    feasible: bool = True,
    finite: bool = True,
    hinge: float = 0.0,
    secondary: float = 0.0,
    e_im: float = 1.0e-4,
    ordinal: int = 0,
    converged: bool | None = None,
) -> ScreenedCandidate:
    return ScreenedCandidate(
        candidate_id=name,
        feasible=feasible,
        finite=finite,
        primary_hinge=hinge,
        secondary_objective=secondary,
        interaction_e_im=e_im,
        ordinal=ordinal,
        optimizer_converged=converged,
    )


def test_asset_derived_blocks_are_complete_without_finger_special_case() -> None:
    names = [
        *(f"r_thumb_{index}" for index in range(4)),
        *(f"r_index_{index}" for index in range(4)),
        *(f"r_middle_{index}" for index in range(4)),
        *(f"r_ring_{index}" for index in range(4)),
        *(f"r_pinky_{index}" for index in range(4)),
    ]
    blocks = asset_derived_dof_blocks(names)
    assert blocks["thumb"] == (0, 1, 2, 3)
    assert blocks["little"] == (16, 17, 18, 19)
    assert sorted(index for block in blocks.values() for index in block) == list(range(20))


def test_contributor_ranking_uses_mass_then_semantic_name() -> None:
    mass = np.zeros(21)
    mass[[4, 8]] = 3.0
    scores = contributor_scores(mass)
    assert scores["thumb"] == 3.0
    assert scores["index"] == 3.0
    assert rank_contributors(scores)[:2] == ("index", "thumb")


def test_seed_generation_is_deterministic_bounded_and_keeps_nonblock_dofs() -> None:
    s1, s2 = default_search_contracts()
    old = np.asarray([0.1, 0.2, 0.3, 0.4])
    neutral = np.zeros(4)
    lower = -np.ones(4)
    upper = np.ones(4)
    transported = np.asarray([0.9, 0.8, 0.7, 0.6])
    kwargs = dict(
        old_q=old,
        neutral_q=neutral,
        lower_q=lower,
        upper_q=upper,
        block=(1, 2),
        transported_q=transported,
    )
    first = deterministic_block_seeds(contract=s2, **kwargs)
    second = deterministic_block_seeds(contract=s2, **kwargs)
    assert [name for name, _value in first] == [name for name, _value in second]
    assert all(
        np.array_equal(left, right) for (_, left), (_, right) in zip(first, second, strict=True)
    )
    assert all(value[0] == old[0] and value[3] == old[3] for _name, value in first)
    assert all(np.all(value >= lower) and np.all(value <= upper) for _name, value in first)
    assert "previous_refined_transported" not in {
        name for name, _value in deterministic_block_seeds(contract=s1, **kwargs)
    }


def test_candidate_selection_rejects_invalid_then_uses_primary_secondary_tie_break() -> None:
    selected = select_candidate(
        [
            candidate("invalid", feasible=False, hinge=-1.0),
            candidate("higher_hinge", hinge=2.0, secondary=0.0),
            candidate("higher_secondary", hinge=1.0, secondary=3.0),
            candidate("winner", hinge=1.0, secondary=2.0),
        ]
    )
    assert selected is not None
    assert selected.candidate_id == "winner"


def test_nonconverged_terminal_state_can_be_independently_screened() -> None:
    terminal = candidate("status_9", converged=False, hinge=0.0, e_im=9.0e-5)
    assert terminal.usable
    assert select_candidate([terminal]) == terminal


def test_interaction_valid_primary_survives_regressive_polish() -> None:
    primary = candidate("primary", e_im=9.0e-5, secondary=4.0)
    polished = candidate("polished", e_im=1.1e-4, secondary=1.0)
    selected, reason = retain_after_polish(primary, polished, interaction_target=1.0e-4)
    assert selected is primary
    assert reason == "PRIMARY_RETAINED_INTERACTION_REGRESSION"


def test_valid_old_q_is_a_technical_completion_when_new_candidates_fail() -> None:
    old = candidate("old_production", e_im=7.0e-5, ordinal=0)
    broken = candidate("new", feasible=False, e_im=6.0e-5, ordinal=1)
    assert select_candidate([broken, old]) == old


def test_sequential_contract_keeps_old_q_and_has_no_stale_state() -> None:
    _s1, s2 = default_search_contracts()
    assert s2.seed_sources[0] == "old_production"
    assert s2.seed_sources.count("previous_refined_transported") == 1
    with pytest.raises(ValueError, match="unavailable"):
        deterministic_block_seeds(
            old_q=np.zeros(4),
            neutral_q=np.zeros(4),
            lower_q=-np.ones(4),
            upper_q=np.ones(4),
            block=(0,),
            transported_q=None,
            contract=s2,
        )
