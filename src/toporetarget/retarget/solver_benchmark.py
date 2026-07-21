"""Auditable Stage 9.1 maxiter benchmark contracts.

The benchmark module does not alter Eq. (8), Eq. (9), bounds, weights, or
acceptance.  It only normalizes per-frame solver observations and selects the
smallest *uniform* budget that passes every fixed case.
"""

from __future__ import annotations

from typing import Any

DEFAULT_BENCHMARK_GRID = (30, 60, 100, 200, 400)
BENCHMARK_SCHEMA_VERSION = "toporetarget.stage9_solver_benchmark.v1"
REQUIRED_FRAME_FIELDS = (
    "case_id",
    "budget",
    "result_success",
    "status_code",
    "message",
    "nit",
    "nfev",
    "njev",
    "initial_objective",
    "final_objective",
    "final_objective_change",
    "final_step_norm",
    "min_hard_residual_m",
    "min_soft_residual_m",
    "full_surface_min_signed_distance_m",
    "active_set_rounds",
    "runtime_s",
    "strict_acceptance",
    "independent_full_surface_audit_pass",
    "deterministic_repeat",
)


def choose_uniform_maxiter(
    records: list[dict[str, Any]],
    grid: tuple[int, ...] = DEFAULT_BENCHMARK_GRID,
    *,
    case_ids: tuple[str, ...] | None = None,
) -> int | None:
    """Choose the smallest budget passing all cases under the strict gate."""

    expected_case_ids = (
        set(case_ids) if case_ids is not None else {str(item.get("case_id")) for item in records}
    )
    for budget in sorted(set(int(item) for item in grid)):
        candidates = [item for item in records if int(item.get("budget", -1)) == budget]
        if not candidates:
            continue
        if (
            all(
                bool(item.get("result_success"))
                and bool(item.get("strict_acceptance"))
                and bool(item.get("independent_full_surface_audit_pass"))
                for item in candidates
            )
            and {str(item.get("case_id")) for item in candidates} == expected_case_ids
        ):
            return budget
    return None


def validate_benchmark_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate the report shape without making a pass/fail claim for results."""

    if payload.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported Stage 9 benchmark report schema")
    grid = tuple(int(item) for item in payload.get("budget_grid", ()))
    if not grid or any(item <= 0 for item in grid):
        raise ValueError("benchmark report has no positive budget grid")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("benchmark report has no frame records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("benchmark frame record is not an object")
        missing = sorted(set(REQUIRED_FRAME_FIELDS) - set(record))
        if missing:
            raise ValueError(f"benchmark frame record is missing fields: {missing}")
        if int(record["budget"]) not in grid:
            raise ValueError("benchmark frame record uses a budget outside the grid")
    selected = payload.get("selected_maxiter")
    if selected is not None and int(selected) != choose_uniform_maxiter(records, grid):
        raise ValueError("benchmark selected_maxiter violates the uniform minimum rule")
    return payload


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_BENCHMARK_GRID",
    "REQUIRED_FRAME_FIELDS",
    "choose_uniform_maxiter",
    "validate_benchmark_report",
]
