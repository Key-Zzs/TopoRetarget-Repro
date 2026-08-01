"""Bounded 26-D object-aware oracle for Stage-16B controllability diagnosis.

The oracle only clones/restores :class:`WorldWristFingerBackend` state.  It
never has an object action and never writes the live object's freejoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .environments.world_wrist_backend import WorldWristFingerBackend
from .world_wrist import so3_log


def world_tracking_error(
    backend: WorldWristFingerBackend, state: dict[str, np.ndarray], reference_index: int
) -> np.ndarray:
    """Fixed normalized object/wrist/finger/link error vector for oracle clones."""

    reference = backend.reference
    index = min(max(reference_index, 0), reference.frame_count - 1)
    object_axis = (
        state["object_axis_points"] - reference.object_axis_points_world_ref[index]
    ).reshape(-1) / 0.04
    wrist_position = (
        state["wrist_pose"][:3, 3] - reference.wrist_pose_world_ref[index, :3, 3]
    ) / 0.02
    wrist_rotation = so3_log(
        state["wrist_pose"][:3, :3].T @ reference.wrist_pose_world_ref[index, :3, :3]
    ) / np.deg2rad(10.0)
    fingers = (state["q"] - reference.q_finger_ref[index]) / (
        backend.joint_upper - backend.joint_lower
    )
    links = (state["links"] - reference.tracked_link_positions_world_ref[index]).reshape(-1) / 0.025
    return np.concatenate([object_axis, wrist_position, wrist_rotation, fingers, links])


@dataclass(frozen=True)
class WorldWristOracleDiagnostics:
    horizon: int
    rank: int
    condition_estimate: float
    baseline_error_norm: float
    predicted_error_norm: float
    action_norm: float
    saturated_dimensions: int
    candidate_error_norms: tuple[float, ...]
    clone_only: bool = True
    direct_object_control: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class WorldWristFingerObjectAwareOracle:
    """One-step finite difference plus fixed-budget H=1/5/10 clone shooting."""

    def __init__(self, *, finite_difference_epsilon: float = 0.05, ridge: float = 1e-3) -> None:
        if finite_difference_epsilon <= 0.0 or ridge < 0.0:
            raise ValueError("oracle epsilon must be positive and ridge non-negative")
        self.finite_difference_epsilon = finite_difference_epsilon
        self.ridge = ridge
        self.last_diagnostics: WorldWristOracleDiagnostics | None = None

    def _rollout_error(
        self, backend: WorldWristFingerBackend, action: np.ndarray, horizon: int
    ) -> float:
        snapshot = backend.snapshot()
        try:
            state = backend._state()  # noqa: SLF001 - oracle clone diagnosis
            for _ in range(horizon):
                state = backend.step(action)
            return float(
                np.linalg.norm(world_tracking_error(backend, state, backend.reference_index))
            )
        finally:
            backend.restore(snapshot)

    def action(self, backend: WorldWristFingerBackend, *, horizon: int = 1) -> np.ndarray:
        if horizon not in (1, 5, 10):
            raise ValueError("oracle horizon must be one of [1, 5, 10]")
        snapshot = backend.snapshot()
        try:
            zero = np.zeros(26, dtype=np.float64)
            baseline_state = backend.predict_step(zero)
            target_index = min(backend.reference_index + 1, backend.reference.frame_count - 1)
            baseline = world_tracking_error(backend, baseline_state, target_index)
            jacobian = np.empty((baseline.size, 26), dtype=np.float64)
            for dimension in range(26):
                positive = zero.copy()
                negative = zero.copy()
                positive[dimension] = self.finite_difference_epsilon
                negative[dimension] = -self.finite_difference_epsilon
                plus = world_tracking_error(backend, backend.predict_step(positive), target_index)
                minus = world_tracking_error(backend, backend.predict_step(negative), target_index)
                jacobian[:, dimension] = (plus - minus) / (2.0 * self.finite_difference_epsilon)
            gram = jacobian.T @ jacobian + self.ridge * np.eye(26)
            rhs = -jacobian.T @ baseline
            linear_action = np.clip(np.linalg.solve(gram, rhs), -1.0, 1.0)
            candidates = (zero, 0.5 * linear_action, linear_action)
            candidate_errors = tuple(
                self._rollout_error(backend, candidate, horizon) for candidate in candidates
            )
            selected_index = int(np.argmin(candidate_errors))
            selected = candidates[selected_index].copy()
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            nonzero = singular_values[singular_values > 1e-10]
            self.last_diagnostics = WorldWristOracleDiagnostics(
                horizon=horizon,
                rank=int(np.linalg.matrix_rank(jacobian)),
                condition_estimate=(
                    float(nonzero.max() / nonzero.min()) if nonzero.size else float("inf")
                ),
                baseline_error_norm=float(np.linalg.norm(baseline)),
                predicted_error_norm=float(candidate_errors[selected_index]),
                action_norm=float(np.linalg.norm(selected)),
                saturated_dimensions=int(np.count_nonzero(np.isclose(np.abs(selected), 1.0))),
                candidate_error_norms=candidate_errors,
            )
            return selected
        finally:
            backend.restore(snapshot)


__all__ = [
    "WorldWristFingerObjectAwareOracle",
    "WorldWristOracleDiagnostics",
    "world_tracking_error",
]
