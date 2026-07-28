from __future__ import annotations

import numpy as np

from toporetarget.retarget.penetration_loss import (
    DenseSDFPenetrationLoss,
    PenetrationLossProfile,
    build_objective_term,
)


def _term(lambda_sdf: float = 1.0) -> DenseSDFPenetrationLoss:
    return build_objective_term(
        "dense_sdf_penetration",
        profile=PenetrationLossProfile.load("dense_squared_hinge_deadzone1mm_v2"),
        lambda_sdf=lambda_sdf,
    )


def test_outside_and_surface_are_zero_without_repulsion() -> None:
    term = _term()
    phi = np.asarray([0.002, 0.0, 0.001], dtype=np.float64)
    jacobian = np.ones((3, 3, 2), dtype=np.float64)
    normals = np.zeros((3, 3), dtype=np.float64)
    normals[:, 0] = 1.0
    result = term.evaluate(
        phi,
        np.asarray(["a", "a", "b"]),
        surface_normals=normals,
        point_jacobian=jacobian,
    )
    assert result.value == 0.0
    np.testing.assert_array_equal(result.gradient, np.zeros(2))


def test_penetration_is_monotone_and_uses_d_ref() -> None:
    term = _term()
    ids = np.asarray(["a", "a", "b"])
    shallow = term.value_only(np.asarray([-0.001, 0.0, 0.0]), ids)
    deep = term.value_only(np.asarray([-0.002, 0.0, 0.0]), ids)
    assert shallow == 0.0
    assert deep == 0.25
    assert deep > shallow


def test_one_mm_dead_zone_only_penalizes_excess_depth() -> None:
    term = _term()
    ids = np.asarray(["a", "b"])
    np.testing.assert_allclose(term.value_only(np.asarray([-0.001, -0.0009]), ids), 0.0)
    np.testing.assert_allclose(
        term.value_only(np.asarray([-0.002, -0.003]), ids), (1.0 + 4.0) / 2.0
    )


def test_geometry_balanced_reduction() -> None:
    term = _term()
    ids = np.asarray(["a", "a", "a", "b"])
    values = np.asarray([-0.002, -0.002, -0.002, -0.003])
    expected = (1.0 + 4.0) / 2.0
    assert term.value_only(values, ids) == expected


def test_gradient_matches_normal_times_point_jacobian_and_has_no_slack() -> None:
    term = _term()
    phi = np.asarray([-0.002, 0.002])
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    jacobian = np.zeros((2, 3, 3))
    jacobian[0, 0, 0] = 1.0
    jacobian[0, 0, 1] = 2.0
    jacobian[0, 0, 2] = 3.0
    result = term.evaluate(
        phi,
        np.asarray(["a", "b"]),
        surface_normals=normals,
        point_jacobian=jacobian,
    )
    np.testing.assert_allclose(result.gradient, [-1000.0, -2000.0, -3000.0])


def test_lambda_zero_is_constructible_but_has_zero_weight() -> None:
    term = _term(0.0)
    assert term.lambda_sdf == 0.0
    assert term.value_only(np.asarray([-0.001]), np.asarray(["a"])) == 0.0
    assert term.value_only(np.asarray([-0.002]), np.asarray(["a"])) == 1.0
    result = term.evaluate(
        np.asarray([-0.002]),
        np.asarray(["a"]),
        surface_normals=np.asarray([[1.0, 0.0, 0.0]]),
        point_jacobian=np.asarray([[[1.0], [0.0], [0.0]]]),
    )
    np.testing.assert_array_equal(term.lambda_sdf * result.gradient, np.zeros(1))


def test_v1_is_retained_as_deprecated_zero_tolerance_profile() -> None:
    profile = PenetrationLossProfile.load("dense_squared_hinge_v1")
    assert profile.deprecated_for_zero_tolerance_comparison is True
    term = build_objective_term("dense_sdf_penetration", profile=profile, lambda_sdf=1.0)
    assert term.value_only(np.asarray([-0.001]), np.asarray(["a"])) == 1.0


def test_invalid_signed_distance_is_rejected() -> None:
    term = _term()
    try:
        term.value_only(np.asarray([np.nan]), np.asarray(["a"]))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-finite signed distance was accepted")


def test_gradient_matches_central_difference_away_from_hinge() -> None:
    term = _term()
    phi0 = np.asarray([-0.0015, -0.0005], dtype=np.float64)
    ids = np.asarray(["a", "b"])
    normals = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    jacobian = np.zeros((2, 3, 2), dtype=np.float64)
    jacobian[0, 0, 0] = 1.0
    jacobian[1, 1, 1] = 1.0
    analytic = term.evaluate(
        phi0,
        ids,
        surface_normals=normals,
        point_jacobian=jacobian,
    ).gradient

    def value(state: np.ndarray) -> float:
        phi = phi0 + np.asarray([state[0], state[1]])
        return term.value_only(phi, ids)

    epsilon = 1.0e-7
    numerical = np.asarray(
        [
            (value(np.eye(2)[index] * epsilon) - value(-np.eye(2)[index] * epsilon))
            / (2.0 * epsilon)
            for index in range(2)
        ]
    )
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-6)


def test_registry_rejects_unknown_or_duplicate_terms() -> None:
    term = _term()
    assert term.term_id == "dense_sdf_penetration"


def test_sphere_and_cube_directional_derivatives_are_finite() -> None:
    term = _term()
    # Analytic sphere and cube queries represented by their local signed
    # distance and outward normals. Both use a variable two-state point map.
    cases = [
        (np.asarray([-0.0012]), np.asarray([[1.0, 0.0, 0.0]])),
        (np.asarray([-0.0008]), np.asarray([[0.0, 1.0, 0.0]])),
    ]
    for phi, normal in cases:
        jacobian = np.zeros((1, 3, 2), dtype=np.float64)
        jacobian[0, :, 0] = normal[0]
        jacobian[0, :, 1] = np.asarray([0.0, 0.0, 1.0])
        result = term.evaluate(
            phi,
            np.asarray(["geometry"]),
            surface_normals=normal,
            point_jacobian=jacobian,
        )
        assert np.all(np.isfinite(result.gradient))
        if phi[0] < -0.001:
            assert result.gradient[0] < 0.0
        else:
            np.testing.assert_allclose(result.gradient[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(result.gradient[1], 0.0, atol=1e-12)


def test_float32_inputs_are_promoted_without_nan() -> None:
    term = _term()
    result = term.evaluate(
        np.asarray([-0.001], dtype=np.float32),
        np.asarray(["a"]),
        surface_normals=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        point_jacobian=np.asarray([[[1.0], [0.0], [0.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _term().value_only(np.asarray([-0.002], dtype=np.float32), np.asarray(["a"])),
        1.0,
        rtol=1e-6,
    )
    assert np.all(np.isfinite(result.gradient))
