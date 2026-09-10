"""CPU-only contracts for O5R-D2D independent certification."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.data.run_oakink2_o5rd2d import (
    EXECUTION_NAME,
    EXECUTION_SHA,
    OBJECTIVE_SHA,
    decide_sparse_gate,
    select_sparse_rows,
    sparse_gate_contract,
    window_gate_contract,
)


def _row(ordinal: int, eim: float) -> dict[str, float | int]:
    return {"ordinal": ordinal, "frame_id": 1000 + ordinal, "old_e_im": eim}


def _gate_row(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "stratum": "HIGH",
        "old_e_im": 2.0e-4,
        "new_e_im": 8.0e-5,
        "relative_reduction": 0.6,
        "technical_completion": True,
        "semantic_hard_pass": True,
        "wrist_pass": True,
        "bone_pass": True,
        "continuity_pass": True,
        "collision_pass": True,
        "joint_limits_pass": True,
    }
    value.update(changes)
    return value


def test_frozen_hash_constants_and_s1_only() -> None:
    assert OBJECTIVE_SHA == "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
    assert EXECUTION_SHA == "b114b3960c47b641a85eac44d974a6e71921613bee85273e3d7de811ba4e937e"
    assert EXECUTION_NAME == "S1_DETERMINISTIC_MULTI_START"


def test_sparse_selection_is_deterministic_and_excludes_development() -> None:
    rows = []
    for ordinal in range(300):
        if ordinal % 3 == 0:
            eim = 2.0e-4 + ordinal * 1.0e-8
        elif ordinal % 3 == 1:
            eim = 9.0e-5
        else:
            eim = 5.0e-5
        rows.append(_row(ordinal, eim))
    excluded = {3, 4, 5, 90, 120}
    first = select_sparse_rows(rows, excluded)
    second = select_sparse_rows(rows, excluded)
    assert first == second
    assert len(first) == 30
    assert {row["stratum"] for row in first} == {"HIGH", "MID", "LOW"}
    assert all(
        sum(row["stratum"] == name for row in first) == 10 for name in ("HIGH", "MID", "LOW")
    )
    assert not ({int(row["ordinal"]) for row in first} & excluded)


def test_sparse_gate_pass_and_each_hard_failure() -> None:
    rows = [_gate_row() for _ in range(20)]
    rows += [
        _gate_row(stratum="LOW", old_e_im=8.0e-5, new_e_im=8.0e-5, relative_reduction=0.0)
        for _ in range(10)
    ]
    assert decide_sparse_gate(rows, True)["SPARSE_VALIDATION_V2"] == "PASS"
    assert decide_sparse_gate(rows, False)["SPARSE_VALIDATION_V2"] == "FAIL"
    broken = [dict(row) for row in rows]
    broken[0]["technical_completion"] = False
    assert decide_sparse_gate(broken, True)["SPARSE_VALIDATION_V2"] == "FAIL"
    broken = [dict(row) for row in rows]
    broken[0]["new_e_im"] = 3.0e-4
    assert decide_sparse_gate(broken, True)["SPARSE_VALIDATION_V2"] == "FAIL"


def test_gate_contracts_are_preregistered_and_immutable() -> None:
    sparse = sparse_gate_contract()
    window = window_gate_contract()
    assert sparse["mutable_after_first_solve"] is False
    assert sparse["determinism_repeats"] == 3
    assert window["mutable_after_first_solve"] is False
    assert window["determinism_windows"] == ["HIGH_1", "LOW"]


def test_cli_has_real_fail_closed_actions_and_no_forbidden_special_case() -> None:
    source = Path("scripts/data/run_oakink2_o5rd2d.py").read_text(encoding="utf-8")
    for action in (
        "preflight",
        "verify-frozen-method",
        "freeze-sparse-v2",
        "run-sparse-v2",
        "freeze-window-v2",
        "run-window-v2",
        "run-dev2-frame0-hard-control",
        "run-dev2-full-if-authorized",
        "run-dev2-semantic-v1",
        "render-dev2-viewer",
        "summarize",
    ):
        assert f'"{action}"' in source
    assert "S2_SEQUENTIAL_MULTI_START" not in source
    assert "DEV2-specific" not in source
    assert "DEV1 full" not in source


def test_downstream_gate_refusal_is_explicit() -> None:
    source = Path("scripts/data/run_oakink2_o5rd2d.py").read_text(encoding="utf-8")
    assert "O5RD2D_WINDOW_BLOCKED_BY_SPARSE" in source
    assert "O5RD2D_FRAME0_BLOCKED" in source
    assert "O5RD2D_FULL_BLOCKED" in source


def test_invalid_sparse_support_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="HIGH_SELECTION_INSUFFICIENT_SUPPORT"):
        select_sparse_rows([_row(index, 5.0e-5) for index in range(100)], set())
