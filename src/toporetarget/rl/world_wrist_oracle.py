"""Bounded contact-aware action-sequence MPC for Stage-16B diagnosis.

The controller has exactly the formal 26-D wrist/finger action.  Candidate
rollouts operate on cloned backend state; there is no object action, wrench,
pose write, or velocity write.
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
class ContactAwareMPCConfig:
    """One globally shared, deterministic and explicitly bounded search budget."""

    population: int = 32
    iterations: int = 3
    elite_count: int = 8
    finite_difference_epsilon: float = 0.05
    ridge: float = 1e-3
    initial_std: float = 0.35
    minimum_std: float = 0.05
    normal_impulse_soft_limit_ns: float = 0.01
    penetration_soft_limit_m: float = 0.002
    seed: int = 20260801

    def validate(self) -> None:
        if not 4 <= self.population <= 32:
            raise ValueError("MPC population must be in [4, 32]")
        if not 1 <= self.iterations <= 3:
            raise ValueError("MPC iterations must be in [1, 3]")
        if not 1 <= self.elite_count < self.population:
            raise ValueError("MPC elite_count must be smaller than population")
        if (
            self.finite_difference_epsilon <= 0.0
            or self.ridge < 0.0
            or self.initial_std <= 0.0
            or self.minimum_std <= 0.0
            or self.normal_impulse_soft_limit_ns <= 0.0
            or self.penetration_soft_limit_m <= 0.0
        ):
            raise ValueError("MPC scales and safety limits must be positive")


@dataclass(frozen=True)
class WorldWristOracleDiagnostics:
    horizon: int
    optimizer: str
    population: int
    iterations: int
    evaluated_sequences: int
    sequence_shape: tuple[int, int]
    rank: int
    condition_estimate: float
    baseline_error_norm: float
    predicted_cost: float
    zero_sequence_cost: float
    action_norm: float
    saturated_dimensions: int
    sequence_variation_norm: float
    predicted_contact_substeps: int
    predicted_normal_impulse_ns: float
    predicted_max_penetration_m: float
    clone_only: bool = True
    direct_object_control: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _SequenceResult:
    cost: float
    final_error_norm: float
    contact_substeps: int
    normal_impulse_ns: float
    max_penetration_m: float


class WorldWristFingerObjectAwareOracle:
    """Receding-horizon contact-aware MPC over an H-by-26 action sequence."""

    def __init__(
        self,
        *,
        config: ContactAwareMPCConfig | None = None,
        finite_difference_epsilon: float | None = None,
        ridge: float | None = None,
    ) -> None:
        # The keyword compatibility is intentional for existing callers, but
        # all formal runs serialize the complete global config.
        base = ContactAwareMPCConfig() if config is None else config
        if finite_difference_epsilon is not None or ridge is not None:
            base = ContactAwareMPCConfig(
                **(
                    asdict(base)
                    | {
                        "finite_difference_epsilon": (
                            base.finite_difference_epsilon
                            if finite_difference_epsilon is None
                            else finite_difference_epsilon
                        ),
                        "ridge": base.ridge if ridge is None else ridge,
                    }
                )
            )
        base.validate()
        self.config = base
        self.last_diagnostics: WorldWristOracleDiagnostics | None = None
        self.last_action_sequence: np.ndarray | None = None

    def _linear_seed(
        self, backend: WorldWristFingerBackend
    ) -> tuple[np.ndarray, int, float, float]:
        zero = np.zeros(26, dtype=np.float64)
        target_index = min(backend.reference_index + 1, backend.reference.frame_count - 1)
        baseline = world_tracking_error(backend, backend.predict_step(zero), target_index)
        jacobian = np.empty((baseline.size, 26), dtype=np.float64)
        for dimension in range(26):
            positive = zero.copy()
            negative = zero.copy()
            positive[dimension] = self.config.finite_difference_epsilon
            negative[dimension] = -self.config.finite_difference_epsilon
            plus = world_tracking_error(backend, backend.predict_step(positive), target_index)
            minus = world_tracking_error(backend, backend.predict_step(negative), target_index)
            jacobian[:, dimension] = (plus - minus) / (2.0 * self.config.finite_difference_epsilon)
        gram = jacobian.T @ jacobian + self.config.ridge * np.eye(26)
        linear_action = np.clip(np.linalg.solve(gram, -jacobian.T @ baseline), -1.0, 1.0)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        nonzero = singular_values[singular_values > 1e-10]
        condition = float(nonzero.max() / nonzero.min()) if nonzero.size else float("inf")
        return (
            linear_action,
            int(np.linalg.matrix_rank(jacobian)),
            condition,
            float(np.linalg.norm(baseline)),
        )

    @staticmethod
    def _reference_dynamic_demand(backend: WorldWristFingerBackend, index: int) -> float:
        reference = backend.reference
        previous = max(index - 1, 0)
        following = min(index + 1, reference.frame_count - 1)
        dt = max(float(reference.timestamps[following] - reference.timestamps[previous]), 1e-9)
        twist_change = (
            reference.object_twist_world_ref[following] - reference.object_twist_world_ref[previous]
        ) / dt
        return float(
            np.linalg.norm(twist_change[:3]) / 2.0 + np.linalg.norm(twist_change[3:]) / 20.0
        )

    def _sequence_rollout(
        self, backend: WorldWristFingerBackend, sequence: np.ndarray
    ) -> _SequenceResult:
        snapshot = backend.snapshot()
        total = 0.0
        contact_substeps = 0
        normal_impulse = 0.0
        maximum_penetration = 0.0
        final_error = float("inf")
        previous = backend.previous_action.copy()
        try:
            for offset, action in enumerate(sequence):
                state = backend.step(action)
                index = backend.reference_index
                error = world_tracking_error(backend, state, index)
                final_error = float(np.linalg.norm(error))
                # Object axis points are the formal 5 cm safety quantity.  A
                # plain mean over the full 92-D error vector lets 74 hand
                # terms dilute them, so preserve the full error while giving
                # the object block an explicit shared weight.
                tracking = float(np.mean(np.square(error)) + 8.0 * np.mean(np.square(error[:18])))
                twist_error = (
                    state["object_twist"] - backend.reference.object_twist_world_ref[index]
                )
                twist_cost = float(
                    np.mean(np.square(twist_error[:3] / 0.10))
                    + np.mean(np.square(twist_error[3:] / 1.0))
                )
                step_contacts = sum(
                    int(row["hand_object_contact_count"] > 0) for row in backend.last_physics_trace
                )
                step_impulse = sum(
                    float(row["hand_object_normal_impulse_ns"])
                    for row in backend.last_physics_trace
                )
                step_penetration = max(
                    (
                        float(row["hand_object_max_penetration_m"])
                        for row in backend.last_physics_trace
                    ),
                    default=0.0,
                )
                contact_substeps += step_contacts
                normal_impulse += step_impulse
                maximum_penetration = max(maximum_penetration, step_penetration)
                demand = self._reference_dynamic_demand(backend, index)
                missing_contact = demand if step_contacts == 0 else 0.0
                impulse_excess = max(
                    step_impulse / self.config.normal_impulse_soft_limit_ns - 1.0, 0.0
                )
                penetration_excess = max(
                    step_penetration / self.config.penetration_soft_limit_m - 1.0, 0.0
                )
                smoothness = float(np.mean(np.square(action - previous)))
                effort = float(np.mean(np.square(action)))
                # Receding-horizon MPC only executes the first action.  A
                # fixed near-term discount prevents H10 from sacrificing that
                # action solely for a distant terminal state.
                discount = 0.85**offset
                total += discount * (tracking + 2.0 * twist_cost)
                total += discount * (
                    0.25 * missing_contact + impulse_excess**2 + penetration_excess**2
                )
                total += discount * (0.01 * effort + 0.03 * smoothness)
                previous = action
            return _SequenceResult(
                cost=total,
                final_error_norm=final_error,
                contact_substeps=contact_substeps,
                normal_impulse_ns=normal_impulse,
                max_penetration_m=maximum_penetration,
            )
        finally:
            backend.restore(snapshot)

    @staticmethod
    def _correlated_noise(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        noise = rng.standard_normal(shape)
        for step in range(1, shape[0]):
            noise[step] = 0.65 * noise[step - 1] + np.sqrt(1.0 - 0.65**2) * noise[step]
        return noise

    def action(self, backend: WorldWristFingerBackend, *, horizon: int = 1) -> np.ndarray:
        if horizon not in (1, 5, 10):
            raise ValueError("oracle horizon must be one of [1, 5, 10]")
        effective_horizon = min(
            horizon, max(backend.reference.frame_count - 1 - backend.reference_index, 1)
        )
        snapshot = backend.snapshot()
        saved_trace = list(backend.last_physics_trace)
        saved_control = backend.last_control
        try:
            linear, rank, condition, baseline_norm = self._linear_seed(backend)
            zero = np.zeros((effective_horizon, 26), dtype=np.float64)
            repeated = np.broadcast_to(linear, zero.shape).copy()
            taper = np.linspace(1.0, 0.35, effective_horizon)[:, None]
            tapered = repeated * taper
            if self.last_action_sequence is not None:
                warm = np.vstack([self.last_action_sequence[1:], self.last_action_sequence[-1:]])
                if warm.shape[0] != effective_horizon:
                    warm = repeated
            else:
                warm = tapered
            bases = [zero, repeated, tapered, np.clip(warm, -1.0, 1.0)]
            base_results = [self._sequence_rollout(backend, candidate) for candidate in bases]
            best_index = int(np.argmin([row.cost for row in base_results]))
            mean = bases[best_index].copy()
            std = np.full_like(mean, self.config.initial_std)
            zero_cost = base_results[0].cost
            evaluated = len(bases)
            best_sequence = mean.copy()
            best_result = base_results[best_index]
            seed = self.config.seed + 1009 * backend.reference_index + 9176 * effective_horizon
            rng = np.random.default_rng(seed)
            for _ in range(self.config.iterations):
                candidates = [mean]
                while len(candidates) < self.config.population:
                    sample = mean + std * self._correlated_noise(rng, mean.shape)
                    candidates.append(np.clip(sample, -1.0, 1.0))
                results = [self._sequence_rollout(backend, candidate) for candidate in candidates]
                evaluated += len(candidates)
                order = np.argsort([row.cost for row in results])
                elites = np.asarray([candidates[int(i)] for i in order[: self.config.elite_count]])
                mean = np.clip(np.mean(elites, axis=0), -1.0, 1.0)
                std = np.maximum(np.std(elites, axis=0), self.config.minimum_std)
                iteration_best = int(order[0])
                if results[iteration_best].cost < best_result.cost:
                    best_sequence = candidates[iteration_best].copy()
                    best_result = results[iteration_best]
            self.last_action_sequence = best_sequence.copy()
            variation = (
                float(np.linalg.norm(np.diff(best_sequence, axis=0)))
                if len(best_sequence) > 1
                else 0.0
            )
            selected = best_sequence[0].copy()
            self.last_diagnostics = WorldWristOracleDiagnostics(
                horizon=horizon,
                optimizer="bounded_deterministic_contact_aware_cem_sequence_mpc_v1",
                population=self.config.population,
                iterations=self.config.iterations,
                evaluated_sequences=evaluated,
                sequence_shape=(effective_horizon, 26),
                rank=rank,
                condition_estimate=condition,
                baseline_error_norm=baseline_norm,
                predicted_cost=best_result.cost,
                zero_sequence_cost=zero_cost,
                action_norm=float(np.linalg.norm(selected)),
                saturated_dimensions=int(np.count_nonzero(np.isclose(np.abs(selected), 1.0))),
                sequence_variation_norm=variation,
                predicted_contact_substeps=best_result.contact_substeps,
                predicted_normal_impulse_ns=best_result.normal_impulse_ns,
                predicted_max_penetration_m=best_result.max_penetration_m,
            )
            return selected
        finally:
            backend.restore(snapshot)
            backend.last_physics_trace = saved_trace
            backend.last_control = saved_control


__all__ = [
    "ContactAwareMPCConfig",
    "WorldWristFingerObjectAwareOracle",
    "WorldWristOracleDiagnostics",
    "world_tracking_error",
]
