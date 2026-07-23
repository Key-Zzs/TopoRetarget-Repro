from __future__ import annotations

import numpy as np
import pytest

from toporetarget.retarget.final_refinement import CollisionQueryProfile
from toporetarget.workflows.shadow_equivalence import (
    CONTINUOUS_FLOORS,
    HARD_CAPS,
    PROFILES,
    _checkpoint_path,
    _equivalence_level,
    _numerical_contract,
    _profile_isolation,
    _profile_query,
)


def _equivalent_fields() -> dict[str, object]:
    fields: dict[str, object] = {
        "identity_pass": True,
        "context_mismatch": False,
        "previous_frame_temporal_context_mismatch": False,
        "optimizer_status": 0,
        "qpos_bounds_pass": True,
        "slack_bounds_pass": True,
        "hard_audit_pass": True,
        "soft_audit_pass": True,
        "frame_identity": True,
        "timestamp_identity": True,
        "final_queryset_order_identity": True,
    }
    fields.update({name: 0.0 for name in CONTINUOUS_FLOORS})
    fields["objective_total"] = 0.0
    return fields


def test_numerical_contract_uses_repeat_noise_floor_and_hard_caps() -> None:
    contract = _numerical_contract({"qpos": 2e-9, "objective_absolute": 1e-11})

    assert contract["continuous_fields"]["qpos"]["selected_tolerance"] == 4e-8
    assert contract["continuous_fields"]["objective_absolute"][
        "selected_tolerance"
    ] == pytest.approx(2e-10)
    assert contract["hard_cap_pass"] is True

    capped = _numerical_contract({"qpos": HARD_CAPS["qpos"]})
    assert capped["hard_cap_pass"] is False
    assert capped["continuous_fields"]["qpos"]["selected_tolerance"] == 20 * HARD_CAPS["qpos"]


def test_equivalence_gate_rejects_context_and_status9() -> None:
    contract = _numerical_contract({})

    assert _equivalence_level(_equivalent_fields(), contract) == "EXACT"

    status9 = _equivalent_fields()
    status9["optimizer_status"] = 9
    assert _equivalence_level(status9, contract) == "NOT_EQUIVALENT"

    context_mismatch = _equivalent_fields()
    context_mismatch["context_mismatch"] = True
    assert _equivalence_level(context_mismatch, contract) == "NOT_EQUIVALENT"

    feasibility_only = _equivalent_fields()
    feasibility_only["hard_audit_pass"] = False
    assert _equivalence_level(feasibility_only, contract) == "FEASIBILITY_EQUIVALENT_ONLY"


def test_zero_margin_is_a_valid_diagnostic_query_profile() -> None:
    final = type(
        "Final",
        (),
        {
            "metadata": {
                "query_profile": {
                    "profile_id": "adaptive_active_set_v1",
                    "version": "1.0.0",
                    "mode": "adaptive",
                    "active_margin_m": 0.01,
                    "max_active_set_rounds": 5,
                    "paper_status": "not_paper_specified",
                    "assumptions": [],
                    "profile_hash": "official",
                }
            }
        },
    )()

    query = _profile_query(final, "zero_active_margin")
    assert isinstance(query, CollisionQueryProfile)
    assert query.mode == "adaptive"
    assert query.active_margin_m == 0.0


def test_profile_isolation_declares_only_intended_changes() -> None:
    report = _profile_isolation(object(), PROFILES)
    assert report["formal_artifact_mutation"] is False
    assert all(row["pass"] for row in report["profiles"])
    half = next(row for row in report["profiles"] if row["profile"] == "half_active_margin")
    assert half["expected_changed_fields"] == ["query_profile.active_margin_m"]
    projection = next(
        row for row in report["profiles"] if row["profile"] == "official_slack_projection_from_warm"
    )
    assert "objective" in projection["expected_changed_fields"]
    assert projection["warm_state_unchanged"] is True


def test_shadow_checkpoint_path_isolated_and_profile_keyed(tmp_path) -> None:
    path = _checkpoint_path(tmp_path, 14, "half_active_margin", 2)
    assert path == tmp_path / "frame_000014" / "half_active_margin" / "repeat_002"
    assert not path.exists()


def test_contract_floor_constants_are_predeclared() -> None:
    assert CONTINUOUS_FLOORS["qpos"] == 1e-8
    assert CONTINUOUS_FLOORS["canonical_sdf"] == 1e-9
    assert CONTINUOUS_FLOORS["objective_relative"] == 1e-9
    assert HARD_CAPS["canonical_sdf"] == 1e-7
    assert np.isfinite(list(CONTINUOUS_FLOORS.values())).all()
