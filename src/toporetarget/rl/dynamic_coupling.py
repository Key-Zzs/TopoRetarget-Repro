"""Bounded Stage-16.1 hand--object dynamic-coupling diagnostics.

The utilities here are intentionally engineering diagnostics.  They preserve
the formal 20D residual action and never supply a direct object command.  In
particular, the local oracle can only clone a MuJoCo state, predict bounded
finger actions, and select the first residual action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from .contracts import Stage16ReferenceClip
from .environments.mujoco_backend import MujocoReferenceTrackingBackend


class ResetVelocityProfile(StrEnum):
    """The four pre-registered reset profiles in the Stage-16.1 protocol."""

    ZERO = "zero"
    FULL_REFERENCE = "full_reference"
    OBJECT_REFERENCE = "object_reference"
    HAND_REFERENCE = "hand_reference"


def finite_difference(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Differentiate a sampled signal with endpoint one-sided differences.

    Interior points use a central difference exactly as the frozen protocol
    requires.  Timestamps are validated rather than assuming a particular
    cadence; the Stage-16 runner separately requires the supplied 20 Hz clip.
    """

    signal = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if signal.shape[0] != times.size or times.size < 2:
        raise ValueError("finite difference requires at least two matching samples")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("finite difference timestamps must be strictly increasing")
    result = np.empty_like(signal)
    result[0] = (signal[1] - signal[0]) / (times[1] - times[0])
    result[-1] = (signal[-1] - signal[-2]) / (times[-1] - times[-2])
    if times.size > 2:
        denominator = (times[2:] - times[:-2]).reshape((-1,) + (1,) * (signal.ndim - 1))
        result[1:-1] = (signal[2:] - signal[:-2]) / denominator
    return result


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the principal SO(3) logarithm as a three-vector."""

    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    sine = np.linalg.norm(axis) * 0.5
    if sine < 1e-8:
        # At pi, use the symmetric part instead of a numerically unstable skew.
        eigenvalues, eigenvectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    else:
        axis /= 2.0 * sine
    return angle * axis / max(np.linalg.norm(axis), 1e-12)


def reference_velocities(clip: Stage16ReferenceClip) -> tuple[np.ndarray, np.ndarray]:
    """Derive qdot and free-joint world velocity from immutable references."""

    clip.validate(expected_hz=20.0)
    qdot = finite_difference(clip.q_finger_ref, clip.timestamps)
    linear = finite_difference(clip.object_pose_base_ref[:, :3, 3], clip.timestamps)
    angular = np.empty((clip.frame_count, 3), dtype=np.float64)
    for index in range(clip.frame_count):
        if index == 0:
            left, right = 0, 1
        elif index == clip.frame_count - 1:
            left, right = index - 1, index
        else:
            left, right = index - 1, index + 1
        dt = float(clip.timestamps[right] - clip.timestamps[left])
        # World-frame angular velocity: R(right) R(left)^T is the finite motion
        # expressed in world coordinates, matching MuJoCo free-joint qvel.
        angular[index] = (
            _rotation_vector(
                clip.object_pose_base_ref[right, :3, :3] @ clip.object_pose_base_ref[left, :3, :3].T
            )
            / dt
        )
    velocity = np.concatenate([linear, angular], axis=1)
    if not np.isfinite(qdot).all() or not np.isfinite(velocity).all():
        raise ValueError("reference velocity derivation produced a non-finite value")
    return qdot, velocity


def reference_acceleration(clip: Stage16ReferenceClip) -> dict[str, np.ndarray]:
    """Return finite, timestamp-correct reference acceleration diagnostics."""

    qdot, object_velocity = reference_velocities(clip)
    return {
        "qdot": qdot,
        "qddot": finite_difference(qdot, clip.timestamps),
        "object_velocity": object_velocity,
        "object_acceleration": finite_difference(object_velocity, clip.timestamps),
    }


def object_error_vector(
    backend: MujocoReferenceTrackingBackend,
    state: dict[str, np.ndarray],
    reference_index: int,
    *,
    axis_weight: float = 1.0,
    rotation_weight: float = 0.25,
    joint_weight: float = 0.10,
    link_weight: float = 0.50,
) -> np.ndarray:
    """Shared weighted object/hand tracking vector for local diagnostics."""

    reference = backend.reference
    axes = (
        state["object_axis_points"] - reference.object_axis_points_base_ref[reference_index]
    ).reshape(-1)
    rotation = _rotation_vector(
        state["object_pose"][:3, :3].T @ reference.object_pose_base_ref[reference_index, :3, :3]
    )
    joint = (state["q"] - reference.q_finger_ref[reference_index]) / (
        backend.joint_upper - backend.joint_lower
    )
    links = (state["links"] - reference.tracked_link_positions_base_ref[reference_index]).reshape(
        -1
    )
    return np.concatenate(
        [axis_weight * axes, rotation_weight * rotation, joint_weight * joint, link_weight * links]
    )


@dataclass(frozen=True)
class OracleDiagnostics:
    """A compact audit record emitted on every object-aware oracle action."""

    rank: int
    singular_values: tuple[float, ...]
    condition_estimate: float
    action_norm: float
    saturated_dimensions: int
    baseline_error_norm: float
    predicted_error_norm: float
    actual_error_norm: float | None

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectAwareResidualOracle:
    """Central-difference, bounded least-squares diagnostic finger oracle.

    The oracle returns exactly one 20D normalized residual action.  All
    sensitivity simulations are performed through :meth:`predict_step`, which
    restores the live MuJoCo state after each clone rollout.
    """

    finite_difference_delta: float = 0.05
    ridge_lambda: float = 1e-3
    axis_weight: float = 1.0
    rotation_weight: float = 0.25
    joint_weight: float = 0.10
    link_weight: float = 0.50
    last_diagnostics: OracleDiagnostics | None = None

    def __post_init__(self) -> None:
        if self.finite_difference_delta <= 0.0 or self.ridge_lambda <= 0.0:
            raise ValueError("oracle finite-difference delta and ridge lambda must be positive")

    def action(self, backend: MujocoReferenceTrackingBackend) -> np.ndarray:
        """Solve one bounded action with no direct object-state mutation."""

        dimension = backend.reference.dof_count
        if dimension != 20:
            raise ValueError("ObjectAwareResidualOracle is frozen to the formal 20D action")
        target = min(backend.reference_index + 1, backend.reference.frame_count - 1)
        nominal = np.zeros(dimension, dtype=np.float64)
        baseline_state = backend.predict_step(nominal)
        baseline = object_error_vector(
            backend,
            baseline_state,
            target,
            axis_weight=self.axis_weight,
            rotation_weight=self.rotation_weight,
            joint_weight=self.joint_weight,
            link_weight=self.link_weight,
        )
        sensitivity = np.empty((baseline.size, dimension), dtype=np.float64)
        for axis in range(dimension):
            plus = nominal.copy()
            minus = nominal.copy()
            plus[axis] = self.finite_difference_delta
            minus[axis] = -self.finite_difference_delta
            error_plus = object_error_vector(
                backend,
                backend.predict_step(plus),
                target,
                axis_weight=self.axis_weight,
                rotation_weight=self.rotation_weight,
                joint_weight=self.joint_weight,
                link_weight=self.link_weight,
            )
            error_minus = object_error_vector(
                backend,
                backend.predict_step(minus),
                target,
                axis_weight=self.axis_weight,
                rotation_weight=self.rotation_weight,
                joint_weight=self.joint_weight,
                link_weight=self.link_weight,
            )
            sensitivity[:, axis] = (error_plus - error_minus) / (2.0 * self.finite_difference_delta)
        ridge = np.sqrt(self.ridge_lambda) * np.eye(dimension)
        previous = backend.previous_action.copy()
        solve = lsq_linear(
            np.vstack([sensitivity, ridge]),
            np.concatenate([-baseline, np.sqrt(self.ridge_lambda) * previous]),
            bounds=(-np.ones(dimension), np.ones(dimension)),
            method="trf",
            lsmr_tol="auto",
        )
        action = np.asarray(solve.x, dtype=np.float64)
        singular_values = np.linalg.svd(sensitivity, compute_uv=False)
        nonzero = singular_values[singular_values > 1e-10]
        predicted = baseline + sensitivity @ action
        self.last_diagnostics = OracleDiagnostics(
            rank=int(np.linalg.matrix_rank(sensitivity, tol=1e-10)),
            singular_values=tuple(float(value) for value in singular_values),
            condition_estimate=(
                float(nonzero.max() / nonzero.min()) if nonzero.size else float("inf")
            ),
            action_norm=float(np.linalg.norm(action)),
            saturated_dimensions=int(np.count_nonzero(np.isclose(np.abs(action), 1.0))),
            baseline_error_norm=float(np.linalg.norm(baseline)),
            predicted_error_norm=float(np.linalg.norm(predicted)),
            actual_error_norm=None,
        )
        return action

    def __call__(
        self, backend: MujocoReferenceTrackingBackend, _state: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Make the oracle usable through the shared diagnostic policy protocol."""

        return self.action(backend)

    def record_actual_error(
        self, backend: MujocoReferenceTrackingBackend, state: dict[str, np.ndarray]
    ) -> None:
        """Attach the actual post-action error without changing the chosen action."""

        if self.last_diagnostics is None:
            raise RuntimeError("oracle action must be computed before actual error is recorded")
        target = backend.reference_index
        actual = object_error_vector(
            backend,
            state,
            target,
            axis_weight=self.axis_weight,
            rotation_weight=self.rotation_weight,
            joint_weight=self.joint_weight,
            link_weight=self.link_weight,
        )
        self.last_diagnostics = OracleDiagnostics(
            **{**self.last_diagnostics.json(), "actual_error_norm": float(np.linalg.norm(actual))}
        )


@dataclass(frozen=True)
class ShootingResult:
    horizon: int
    candidate_count: int
    baseline_error_norm: float
    best_error_norm: float
    first_action: tuple[float, ...]
    classification: str

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectAwareShootingOracle:
    """Fixed-seed, fixed-budget local shooting fallback for diagnosis only."""

    horizons: tuple[int, ...] = (5, 10)
    candidates: int = 32
    seed: int = 20260801

    def __post_init__(self) -> None:
        if self.horizons not in {(5,), (5, 10)} or not 1 <= self.candidates <= 128:
            raise ValueError("shooting is limited to H=5/H=10 and at most 128 candidates")

    def diagnose(
        self, backend: MujocoReferenceTrackingBackend, nominal_action: np.ndarray, horizon: int
    ) -> ShootingResult:
        """Select a first action from deterministic bounded residual sequences."""

        if horizon not in self.horizons:
            raise ValueError("shooting horizon is not in the fixed protocol")
        dimension = backend.reference.dof_count
        if dimension != 20:
            raise ValueError("shooting uses only the formal 20D action")
        snapshot = backend.snapshot()
        target = min(backend.reference_index + horizon, backend.reference.frame_count - 1)
        baseline_state = snapshot
        try:
            generator = np.random.default_rng(self.seed + horizon + backend.reference_index)
            sequences = [np.zeros((horizon, dimension), dtype=np.float64)]
            sequences.append(np.broadcast_to(nominal_action, (horizon, dimension)).copy())
            for _ in range(self.candidates - len(sequences)):
                sequence = np.broadcast_to(nominal_action, (horizon, dimension)).copy()
                sequence += generator.normal(0.0, 0.20, size=sequence.shape)
                sequences.append(np.clip(sequence, -1.0, 1.0))
            best_loss = float("inf")
            best = sequences[0]
            baseline = float("inf")
            for sequence_index, sequence in enumerate(sequences):
                backend.restore(snapshot)
                state: dict[str, np.ndarray] | None = None
                for action in sequence:
                    state = backend.step(action)
                if state is None:
                    raise RuntimeError("shooting sequence is empty")
                loss = float(np.linalg.norm(object_error_vector(backend, state, target)))
                if sequence_index == 0:
                    baseline = loss
                if loss < best_loss:
                    best_loss = loss
                    best = sequence
            classification = (
                "LOCAL_FEASIBILITY_PASS" if best_loss < baseline else "LOCAL_FEASIBILITY_NO_DESCENT"
            )
            return ShootingResult(
                horizon=horizon,
                candidate_count=len(sequences),
                baseline_error_norm=baseline,
                best_error_norm=best_loss,
                first_action=tuple(float(value) for value in best[0]),
                classification=classification,
            )
        finally:
            backend.restore(baseline_state)


__all__ = [
    "ObjectAwareResidualOracle",
    "ObjectAwareShootingOracle",
    "OracleDiagnostics",
    "ResetVelocityProfile",
    "ShootingResult",
    "finite_difference",
    "object_error_vector",
    "reference_acceleration",
    "reference_velocities",
]
