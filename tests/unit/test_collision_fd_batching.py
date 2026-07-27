from __future__ import annotations

import numpy as np

from toporetarget.retarget.final_refinement import (
    _batched_constraint_finite_difference,
    _column_sparse_constraint_finite_difference,
    _vectorized_constraint_finite_difference,
    _virtual_closure_query_gate,
)


def test_nonsmooth_constraint_fd_batches_rows_without_changing_values() -> None:
    current = np.asarray([0.2, -0.4, 0.7], dtype=np.float64)
    rows = np.asarray([0, 2, 4], dtype=np.int64)
    matrix = np.asarray(
        [
            [1.0, 2.0, -1.0],
            [0.3, -0.2, 0.8],
            [-1.0, 0.5, 0.2],
            [2.0, -3.0, 0.1],
            [0.7, 1.3, -2.0],
        ],
        dtype=np.float64,
    )
    calls = 0

    def residual(value: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return matrix @ value + np.sin(matrix @ value)

    actual, reported_calls = _batched_constraint_finite_difference(
        residual,
        current,
        variable_count=len(current),
        row_ids=rows,
        epsilon=1e-6,
    )
    expected = np.empty_like(actual)
    for output_row, residual_row in enumerate(rows):
        for column in range(len(current)):
            plus = current.copy()
            minus = current.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            expected[output_row, column] = (
                matrix @ plus + np.sin(matrix @ plus) - matrix @ minus - np.sin(matrix @ minus)
            )[residual_row] / 2e-6
    assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert calls == reported_calls == 2 * len(current)
    assert calls < 2 * len(current) * len(rows)


def test_nonsmooth_constraint_fd_vectorizes_all_perturbations_without_changing_values() -> None:
    current = np.asarray([0.2, -0.4, 0.7], dtype=np.float64)
    rows = np.asarray([0, 2, 4], dtype=np.int64)
    matrix = np.asarray(
        [
            [1.0, 2.0, -1.0],
            [0.3, -0.2, 0.8],
            [-1.0, 0.5, 0.2],
            [2.0, -3.0, 0.1],
            [0.7, 1.3, -2.0],
        ],
        dtype=np.float64,
    )
    calls = 0

    def residual_batch(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        linear = values @ matrix.T
        return linear + np.sin(linear)

    actual, reported_calls = _vectorized_constraint_finite_difference(
        residual_batch,
        current,
        variable_count=len(current),
        row_ids=rows,
        epsilon=1e-6,
    )
    expected, _ = _batched_constraint_finite_difference(
        lambda value: matrix @ value + np.sin(matrix @ value),
        current,
        variable_count=len(current),
        row_ids=rows,
        epsilon=1e-6,
    )
    assert np.array_equal(actual, expected)
    assert calls == reported_calls == 1


def test_nonsmooth_constraint_fd_skips_structurally_zero_dependencies() -> None:
    current = np.asarray([0.2, -0.4, 0.7], dtype=np.float64)
    matrix = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [0.0, -0.2, 0.0],
            [-1.0, 0.0, 0.2],
        ],
        dtype=np.float64,
    )
    dependencies = matrix != 0.0
    calls = 0

    def requested_residuals(values: np.ndarray, row_blocks: list[np.ndarray]) -> list[np.ndarray]:
        nonlocal calls
        calls += 1
        linear = values @ matrix.T
        residuals = linear + np.sin(linear)
        return [residuals[index, rows] for index, rows in enumerate(row_blocks)]

    actual, reported_calls, probe_count = _column_sparse_constraint_finite_difference(
        requested_residuals,
        current,
        dependency_mask=dependencies,
        epsilon=1e-6,
    )
    expected, _ = _batched_constraint_finite_difference(
        lambda value: matrix @ value + np.sin(matrix @ value),
        current,
        variable_count=len(current),
        row_ids=np.arange(len(matrix), dtype=np.int64),
        epsilon=1e-6,
    )
    assert np.array_equal(actual, expected)
    assert calls == reported_calls == 1
    assert probe_count == 2 * int(np.count_nonzero(dependencies))


def test_virtual_closure_gate_counts_only_active_queryset_members() -> None:
    patch_mask = np.zeros(512, dtype=bool)
    patch_mask[:12] = True
    active_ids = np.asarray([0, 1, 2, 100, 200], dtype=np.int64)
    count, limit = _virtual_closure_query_gate(
        patch_mask,
        active_ids,
        full_collision_sample_count=512,
    )
    assert count == 3
    assert limit == 11
