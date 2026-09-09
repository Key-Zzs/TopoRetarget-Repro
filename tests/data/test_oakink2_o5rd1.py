from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data.run_oakink2_o5rc import block_map
from scripts.data.run_oakink2_o5rd1 import (
    _component_rows,
    bind_state_quartet,
    central_directional_derivative,
    classify_b1,
    reconstruct_total,
    rollback_ratio,
)


def breakdown(**updates: float) -> SimpleNamespace:
    values = {
        "e_im": 2.0,
        "e_bone": 3.0,
        "e_temporal": 5.0,
        "e_base_pos": 7.0,
        "e_base_rot": 11.0,
        "e_slack": 13.0,
        "weighted_e_im": 17.0,
        "weighted_e_bone": 19.0,
        "weighted_e_morph": 23.0,
        "weighted_e_contact_pos": 29.0,
        "weighted_e_contact_dir": 31.0,
        "total": 155.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_objective_component_sum_reconstructs_scalar() -> None:
    value = breakdown()
    assert reconstruct_total(value) == pytest.approx(value.total)


def test_component_rows_do_not_double_apply_weights() -> None:
    value = breakdown(e_im=0.2, weighted_e_im=100.0, e_bone=4.0, weighted_e_bone=0.4)
    paper = SimpleNamespace(
        lambda_im=500.0,
        lambda_bone=0.1,
        lambda_base_pos=100.0,
        lambda_base_rot=1.0,
        w_s=100000.0,
    )
    rows = _component_rows(
        {"frame_id": 1, "ordinal": 2, "population": "TEST", "stratum": "HIGH"},
        "Q0_OLD_PRODUCTION",
        value,
        np.zeros(6),
        np.zeros(3),
        paper,
    )
    indexed = {row["component"]: row for row in rows}
    assert indexed["interaction_mesh"]["weighted"] == pytest.approx(100.0)
    assert indexed["interaction_mesh"]["normalized"] == pytest.approx(0.2)
    assert indexed["bone_direction"]["weighted"] == pytest.approx(0.4)


def test_state_quartet_binding_preserves_authorities() -> None:
    arrays = {
        "q0": np.asarray([0.0]),
        "base0": np.eye(4),
        "q1": np.asarray([1.0]),
        "q2": np.asarray([2.0]),
        "base2": np.eye(4) * 2.0,
        "q3": np.asarray([3.0]),
        "base3": np.eye(4) * 3.0,
    }
    full = bind_state_quartet(arrays, True)
    assert [row[0] for row in full] == [
        "Q0_OLD_PRODUCTION",
        "Q1_B1_BEST_OBSERVED",
        "Q2_STRUCTURED_BLOCK",
        "Q3_STRUCTURED_FINAL",
    ]
    assert np.array_equal(full[1][1], arrays["q1"])
    assert np.array_equal(full[1][2], arrays["base0"])
    without_q1 = bind_state_quartet(arrays, False)
    assert [row[0] for row in without_q1] == [
        "Q0_OLD_PRODUCTION",
        "Q2_STRUCTURED_BLOCK",
        "Q3_STRUCTURED_FINAL",
    ]


@pytest.mark.parametrize(
    ("gain", "loss", "expected"),
    [
        (0.0, 1.0, None),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 1.0),
        (2.0, 3.0, 1.5),
    ],
)
def test_rollback_ratio_cases(gain: float, loss: float, expected: float | None) -> None:
    assert rollback_ratio(gain, loss) == expected


def test_asset_derived_contributor_mapping_has_no_thumb_special_case() -> None:
    model = SimpleNamespace(
        dof_names=[
            *(f"r_thumb_{index}" for index in range(4)),
            *(f"r_index_{index}" for index in range(4)),
            *(f"r_middle_{index}" for index in range(4)),
            *(f"r_ring_{index}" for index in range(4)),
            *(f"r_pinky_{index}" for index in range(4)),
        ]
    )
    mapping = block_map(model)
    assert mapping["THUMB"] == [0, 1, 2, 3]
    assert mapping["LITTLE"] == [16, 17, 18, 19]
    assert all(
        len(mapping[finger]) == 4 for finger in ("THUMB", "INDEX", "MIDDLE", "RING", "LITTLE")
    )


def test_directional_derivative_matches_synthetic_quadratic() -> None:
    origin = np.asarray([1.0, 2.0])
    direction = np.asarray([3.0, -4.0])
    observed = central_directional_derivative(lambda value: float(value @ value), origin, direction)
    expected = float(2.0 * origin @ direction)
    assert observed == pytest.approx(expected, abs=1e-9)


def test_b1_alignment_classification_uses_frozen_epsilon() -> None:
    assert classify_b1(-1.0, -1.0, 1e-10) == "ALIGNED_BETTER"
    assert classify_b1(-1.0, 1.0, 1e-10) == "SEMANTIC_BETTER_OBJECTIVE_WORSE"
    assert classify_b1(-1e-12, -1.0, 1e-10) == "NO_MEANINGFUL_SEMANTIC_IMPROVEMENT"
