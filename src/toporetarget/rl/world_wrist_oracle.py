"""Bounded contact-aware action-sequence MPC for Stage-16B diagnosis.

The controller has exactly the formal 26-D wrist/finger action.  Candidate
rollouts operate on cloned backend state; there is no object action, wrench,
pose write, or velocity write.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

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
    formal_gate_barrier_weight: float = 50.0
    formal_gate_violation_penalty: float = 1000.0

    def validate(self) -> None:
        if not 4 <= self.population <= 48:
            raise ValueError("MPC population must be in [4, 48]")
        if not 1 <= self.iterations <= 4:
            raise ValueError("MPC iterations must be in [1, 4]")
        if not 1 <= self.elite_count < self.population:
            raise ValueError("MPC elite_count must be smaller than population")
        if (
            self.finite_difference_epsilon <= 0.0
            or self.ridge < 0.0
            or self.initial_std <= 0.0
            or self.minimum_std <= 0.0
            or self.normal_impulse_soft_limit_ns <= 0.0
            or self.penetration_soft_limit_m <= 0.0
            or self.formal_gate_barrier_weight <= 0.0
            or self.formal_gate_violation_penalty <= 0.0
        ):
            raise ValueError("MPC scales and safety limits must be positive")


@dataclass(frozen=True)
class FormalGateThresholds:
    """Frozen Stage-16B formal and wrist-safety thresholds."""

    object_position_m: float = 0.05
    object_axis_point_m: float = 0.05
    object_rotation_rad: float = float(np.deg2rad(45.0))
    wrist_position_m: float = 0.20
    wrist_rotation_rad: float = float(np.deg2rad(90.0))


def formal_gate_barrier(
    normalized: float | np.ndarray,
    *,
    epsilon: float = 1e-6,
    violation_penalty: float = 1000.0,
) -> float | np.ndarray:
    """Smooth in-gate barrier with an explicit non-cancellable violation penalty."""

    values = np.asarray(normalized, dtype=np.float64)
    if np.any(values < 0.0) or epsilon <= 0.0 or violation_penalty <= 0.0:
        raise ValueError("normalized gates must be non-negative and barrier scales positive")
    inside = -np.log(np.maximum(1.0 - np.minimum(values, 1.0), epsilon))
    result = np.where(values < 1.0, inside, violation_penalty + values - 1.0)
    return float(result) if result.ndim == 0 else result


def effective_horizon_portfolio(remaining: int) -> tuple[int, ...]:
    """Return the unique terminal-contracted H1/H5/H10 shared portfolio."""

    if remaining < 0:
        raise ValueError("remaining transitions cannot be negative")
    if remaining == 0:
        return ()
    if remaining == 1:
        return (1,)
    if remaining < 5:
        return (1, remaining)
    if remaining < 10:
        return tuple(dict.fromkeys((1, 5, remaining)))
    return (1, 5, 10)


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
    requested_horizon: int | None = None
    effective_horizon: int | None = None
    predicted_termination: str | None = None
    predicted_gate_violation: float = float("inf")
    minimum_gate_margin: float = float("-inf")
    near_axis_error_m: float = float("inf")
    near_object_position_error_m: float = float("inf")
    near_object_rotation_error_rad: float = float("inf")
    predicted_reference_complete: bool = False
    predicted_contact_loss: float = float("inf")
    predicted_excessive_impulse: float = float("inf")
    predicted_wrist_error: float = float("inf")
    action_first_difference: float = float("inf")
    action_second_difference: float = float("inf")
    action_effort: float = float("inf")
    selection_lookahead: int | None = None
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
    predicted_termination: str | None
    gate_violation: float
    minimum_gate_margin: float
    near_axis_error_m: float
    near_object_position_error_m: float
    near_object_rotation_error_rad: float
    reference_complete: bool
    contact_loss: float
    excessive_impulse: float
    wrist_error: float
    first_difference: float
    second_difference: float
    effort: float
    step_metrics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HorizonCandidate:
    """One optimized horizon candidate exposed to the shared selector."""

    requested_horizon: int
    effective_horizon: int
    sequence: np.ndarray = field(repr=False, compare=False)
    result: _SequenceResult = field(repr=False, compare=False)
    diagnostics: WorldWristOracleDiagnostics

    def as_dict(self, *, include_sequence: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "requested_horizon": self.requested_horizon,
            "effective_horizon": self.effective_horizon,
            "diagnostics": self.diagnostics.as_dict(),
            "step_metrics": list(self.result.step_metrics),
        }
        if include_sequence:
            value["sequence"] = self.sequence.tolist()
        return value


class GateFirstHorizonSelector:
    """Clip-agnostic lexicographic selector for predicted formal outcomes."""

    @staticmethod
    def key(candidate: HorizonCandidate) -> tuple[float | int, ...]:
        result = candidate.result
        return (
            int(result.predicted_termination is not None),
            result.gate_violation,
            -result.minimum_gate_margin,
            result.near_axis_error_m,
            result.near_object_position_error_m,
            result.near_object_rotation_error_rad,
            int(not result.reference_complete),
            result.contact_loss,
            result.excessive_impulse,
            result.max_penetration_m,
            result.wrist_error,
            result.first_difference,
            result.second_difference,
            result.effort,
            candidate.effective_horizon,
        )

    def select(self, candidates: Iterable[HorizonCandidate]) -> tuple[HorizonCandidate, str]:
        values = tuple(candidates)
        if not values:
            raise ValueError("at least one horizon candidate is required")
        selected = min(values, key=self.key)
        reason = (
            "lexicographic_gate_first: "
            f"termination={selected.result.predicted_termination}; "
            f"gate={selected.result.gate_violation:.9g}; "
            f"margin={selected.result.minimum_gate_margin:.9g}; "
            f"near_axis={selected.result.near_axis_error_m:.9g}; "
            f"near_position={selected.result.near_object_position_error_m:.9g}; "
            f"near_rotation={selected.result.near_object_rotation_error_rad:.9g}; "
            f"effective_horizon={selected.effective_horizon}"
        )
        return selected, reason


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

    @staticmethod
    def _resize_seed(sequence: np.ndarray, horizon: int) -> np.ndarray:
        """Deterministically truncate or taper/repeat a shared sequence seed."""

        value = np.asarray(sequence, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 26 or value.shape[0] < 1:
            raise ValueError("seed sequence must have shape [H, 26]")
        if value.shape[0] >= horizon:
            return value[:horizon].copy()
        extra = horizon - value.shape[0]
        decay = np.linspace(0.85, 0.35, extra, dtype=np.float64)[:, None]
        return np.vstack([value, value[-1:] * decay])

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
        second_previous = backend.second_previous_action.copy()
        thresholds = FormalGateThresholds()
        gate_values: list[float] = []
        axis_errors: list[float] = []
        position_errors: list[float] = []
        rotation_errors: list[float] = []
        wrist_errors: list[float] = []
        contact_losses: list[float] = []
        impulse_excesses: list[float] = []
        first_differences: list[float] = []
        second_differences: list[float] = []
        efforts: list[float] = []
        step_metrics: list[dict[str, Any]] = []
        predicted_termination: str | None = None
        try:
            for offset, action in enumerate(sequence):
                state, _, reason = backend.transition(action)
                index = backend.reference_index
                error = world_tracking_error(backend, state, index)
                final_error = float(np.linalg.norm(error))
                object_position_error = float(
                    np.linalg.norm(
                        state["object_pose"][:3, 3]
                        - backend.reference.object_pose_world_ref[index, :3, 3]
                    )
                )
                object_rotation_error = float(
                    np.linalg.norm(
                        so3_log(
                            state["object_pose"][:3, :3].T
                            @ backend.reference.object_pose_world_ref[index, :3, :3]
                        )
                    )
                )
                axis_error = float(
                    np.max(
                        np.linalg.norm(
                            state["object_axis_points"]
                            - backend.reference.object_axis_points_world_ref[index],
                            axis=1,
                        )
                    )
                )
                wrist_position_error = float(
                    np.linalg.norm(
                        state["wrist_pose"][:3, 3]
                        - backend.reference.wrist_pose_world_ref[index, :3, 3]
                    )
                )
                wrist_rotation_error = float(
                    np.linalg.norm(
                        so3_log(
                            state["wrist_pose"][:3, :3].T
                            @ backend.reference.wrist_pose_world_ref[index, :3, :3]
                        )
                    )
                )
                normalized_gates = np.asarray(
                    [
                        object_position_error / thresholds.object_position_m,
                        axis_error / thresholds.object_axis_point_m,
                        object_rotation_error / thresholds.object_rotation_rad,
                        wrist_position_error / thresholds.wrist_position_m,
                        wrist_rotation_error / thresholds.wrist_rotation_rad,
                    ]
                )
                gate_value = float(np.max(normalized_gates))
                gate_values.append(gate_value)
                position_errors.append(object_position_error)
                rotation_errors.append(object_rotation_error)
                axis_errors.append(axis_error)
                wrist_errors.append(
                    max(
                        wrist_position_error / thresholds.wrist_position_m,
                        wrist_rotation_error / thresholds.wrist_rotation_rad,
                    )
                )
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
                second_difference = float(
                    np.mean(np.square(action - 2.0 * previous + second_previous))
                )
                effort = float(np.mean(np.square(action)))
                # Receding-horizon MPC only executes the first action.  A
                # fixed near-term discount prevents H10 from sacrificing that
                # action solely for a distant terminal state.
                discount = 0.85**offset
                total += discount * (tracking + 2.0 * twist_cost)
                total += self.config.formal_gate_barrier_weight * float(
                    np.sum(
                        formal_gate_barrier(
                            normalized_gates,
                            violation_penalty=self.config.formal_gate_violation_penalty,
                        )
                    )
                )
                total += discount * (
                    0.25 * missing_contact + impulse_excess**2 + penetration_excess**2
                )
                total += discount * (0.01 * effort + 0.03 * smoothness)
                contact_losses.append(missing_contact)
                impulse_excesses.append(impulse_excess)
                first_differences.append(smoothness)
                second_differences.append(second_difference)
                efforts.append(effort)
                step_metrics.append(
                    {
                        "offset": offset,
                        "reference_index": index,
                        "object_position_error_m": object_position_error,
                        "object_rotation_error_rad": object_rotation_error,
                        "max_axis_error_m": axis_error,
                        "wrist_position_error_m": wrist_position_error,
                        "wrist_rotation_error_rad": wrist_rotation_error,
                        "gate_normalized": normalized_gates.tolist(),
                        "predicted_gate_violation": gate_value,
                        "termination": reason,
                        "contact_substeps": step_contacts,
                        "normal_impulse_ns": step_impulse,
                        "max_penetration_m": step_penetration,
                    }
                )
                second_previous = previous
                previous = action
                if reason is not None:
                    if reason != "SUCCESS_REFERENCE_COMPLETE":
                        predicted_termination = reason
                    break
            near_count = min(3, len(axis_errors))
            maximum_gate = max(gate_values, default=float("inf"))
            return _SequenceResult(
                cost=total,
                final_error_norm=final_error,
                contact_substeps=contact_substeps,
                normal_impulse_ns=normal_impulse,
                max_penetration_m=maximum_penetration,
                predicted_termination=predicted_termination,
                gate_violation=maximum_gate,
                minimum_gate_margin=1.0 - maximum_gate,
                near_axis_error_m=max(axis_errors[:near_count], default=float("inf")),
                near_object_position_error_m=max(
                    position_errors[:near_count], default=float("inf")
                ),
                near_object_rotation_error_rad=max(
                    rotation_errors[:near_count], default=float("inf")
                ),
                reference_complete=backend.reference_index >= backend.reference.frame_count - 1,
                contact_loss=float(np.sum(contact_losses)),
                excessive_impulse=max(impulse_excesses, default=0.0),
                wrist_error=max(wrist_errors, default=float("inf")),
                first_difference=float(np.sum(first_differences)),
                second_difference=float(np.sum(second_differences)),
                effort=float(np.sum(efforts)),
                step_metrics=tuple(step_metrics),
            )
        finally:
            backend.restore(snapshot)

    @staticmethod
    def _correlated_noise(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
        noise = rng.standard_normal(shape)
        for step in range(1, shape[0]):
            noise[step] = 0.65 * noise[step - 1] + np.sqrt(1.0 - 0.65**2) * noise[step]
        return noise

    @staticmethod
    def _result_key(result: _SequenceResult) -> tuple[float | int, ...]:
        return (
            int(result.predicted_termination is not None),
            result.cost,
        )

    def optimize_horizon(
        self,
        backend: WorldWristFingerBackend,
        *,
        requested_horizon: int,
        warm_sequence: np.ndarray | None = None,
        seed_sequences: Iterable[np.ndarray] = (),
    ) -> HorizonCandidate:
        if requested_horizon < 1:
            raise ValueError("requested horizon must be positive")
        remaining = backend.reference.frame_count - 1 - backend.reference_index
        if remaining <= 0:
            raise ValueError("reference is already complete; no extra action is permitted")
        effective_horizon = min(requested_horizon, remaining)
        snapshot = backend.snapshot()
        saved_trace = list(backend.last_physics_trace)
        saved_control = backend.last_control
        try:
            linear, rank, condition, baseline_norm = self._linear_seed(backend)
            zero = np.zeros((effective_horizon, 26), dtype=np.float64)
            repeated = np.broadcast_to(linear, zero.shape).copy()
            taper = np.linspace(1.0, 0.35, effective_horizon)[:, None]
            tapered = repeated * taper
            if warm_sequence is not None:
                warm = self._resize_seed(warm_sequence, effective_horizon)
            elif self.last_action_sequence is not None:
                shifted = np.vstack([self.last_action_sequence[1:], self.last_action_sequence[-1:]])
                warm = self._resize_seed(shifted, effective_horizon)
            else:
                warm = tapered
            bases = [zero, repeated, tapered, np.clip(warm, -1.0, 1.0)]
            shared_seeds = [
                np.clip(self._resize_seed(seed, effective_horizon), -1.0, 1.0)
                for seed in seed_sequences
            ]
            base_results = [self._sequence_rollout(backend, candidate) for candidate in bases]
            best_index = min(
                range(len(base_results)),
                key=lambda index: self._result_key(base_results[index]),
            )
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
                for shared_seed in shared_seeds:
                    if len(candidates) >= self.config.population:
                        break
                    candidates.append(shared_seed)
                while len(candidates) < self.config.population:
                    sample = mean + std * self._correlated_noise(rng, mean.shape)
                    candidates.append(np.clip(sample, -1.0, 1.0))
                results = [self._sequence_rollout(backend, candidate) for candidate in candidates]
                evaluated += len(candidates)
                order = sorted(
                    range(len(results)), key=lambda index: self._result_key(results[index])
                )
                elites = np.asarray([candidates[int(i)] for i in order[: self.config.elite_count]])
                mean = np.clip(np.mean(elites, axis=0), -1.0, 1.0)
                std = np.maximum(np.std(elites, axis=0), self.config.minimum_std)
                iteration_best = int(order[0])
                if self._result_key(results[iteration_best]) < self._result_key(best_result):
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
                horizon=requested_horizon,
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
                requested_horizon=requested_horizon,
                effective_horizon=effective_horizon,
                predicted_termination=best_result.predicted_termination,
                predicted_gate_violation=best_result.gate_violation,
                minimum_gate_margin=best_result.minimum_gate_margin,
                near_axis_error_m=best_result.near_axis_error_m,
                near_object_position_error_m=best_result.near_object_position_error_m,
                near_object_rotation_error_rad=best_result.near_object_rotation_error_rad,
                predicted_reference_complete=best_result.reference_complete,
                predicted_contact_loss=best_result.contact_loss,
                predicted_excessive_impulse=best_result.excessive_impulse,
                predicted_wrist_error=best_result.wrist_error,
                action_first_difference=best_result.first_difference,
                action_second_difference=best_result.second_difference,
                action_effort=best_result.effort,
            )
            return HorizonCandidate(
                requested_horizon=requested_horizon,
                effective_horizon=effective_horizon,
                sequence=best_sequence.copy(),
                result=best_result,
                diagnostics=self.last_diagnostics,
            )
        finally:
            backend.restore(snapshot)
            backend.last_physics_trace = saved_trace
            backend.last_control = saved_control

    def action(self, backend: WorldWristFingerBackend, *, horizon: int = 1) -> np.ndarray:
        if horizon not in (1, 5, 10):
            raise ValueError("oracle horizon must be one of [1, 5, 10]")
        candidate = self.optimize_horizon(backend, requested_horizon=horizon)
        self.last_action_sequence = candidate.sequence.copy()
        return candidate.sequence[0].copy()


class TerminalSafeOneStepController:
    """The remaining==1 branch of the same shared adaptive algorithm."""

    def __init__(self, optimizer: WorldWristFingerObjectAwareOracle) -> None:
        self.optimizer = optimizer

    def candidate(
        self,
        backend: WorldWristFingerBackend,
        *,
        warm_sequence: np.ndarray | None = None,
        seed_sequences: Iterable[np.ndarray] = (),
    ) -> HorizonCandidate:
        remaining = backend.reference.frame_count - 1 - backend.reference_index
        if remaining != 1:
            raise ValueError("terminal-safe one-step control is only valid when remaining == 1")
        return self.optimizer.optimize_horizon(
            backend,
            requested_horizon=1,
            warm_sequence=warm_sequence,
            seed_sequences=seed_sequences,
        )


class AdaptiveMultiHorizonContactOracle:
    """Shared state-adaptive H1/H5/H10 oracle with no clip identity input."""

    def __init__(
        self,
        *,
        config: ContactAwareMPCConfig | None = None,
        selector: GateFirstHorizonSelector | None = None,
        selection_lookahead: int = 10,
    ) -> None:
        if not 1 <= selection_lookahead <= 40:
            raise ValueError("selection lookahead must be in [1, 40]")
        self.optimizer = WorldWristFingerObjectAwareOracle(config=config)
        self.selector = selector or GateFirstHorizonSelector()
        self.terminal_controller = TerminalSafeOneStepController(self.optimizer)
        self.selection_lookahead = selection_lookahead
        self._warm_sequences: dict[int, np.ndarray] = {}
        self.last_candidates: tuple[HorizonCandidate, ...] = ()
        self.last_selected: HorizonCandidate | None = None
        self.last_reason: str | None = None
        self.selection_trace: list[dict[str, Any]] = []

    @staticmethod
    def _shift(sequence: np.ndarray) -> np.ndarray:
        return np.vstack([sequence[1:], sequence[-1:]]) if len(sequence) > 1 else sequence.copy()

    def _closed_loop_viability_candidate(
        self,
        backend: WorldWristFingerBackend,
        initial: HorizonCandidate,
        *,
        lookahead: int,
    ) -> HorizonCandidate:
        """Compare horizons over one common window using true receding replans.

        Every projected step solves a newly terminal-contracted sequence and
        executes only its first action on a clone.  No action/reference padding
        or repeated final-frame cost is introduced.
        """

        snapshot = backend.snapshot()
        saved_trace = list(backend.last_physics_trace)
        saved_control = backend.last_control
        actions: list[np.ndarray] = []
        current = initial
        warm = initial.sequence.copy()
        try:
            for offset in range(lookahead):
                if offset:
                    current = self.optimizer.optimize_horizon(
                        backend,
                        requested_horizon=initial.requested_horizon,
                        warm_sequence=self._shift(warm),
                    )
                    warm = current.sequence.copy()
                action = current.sequence[0].copy()
                actions.append(action)
                _, _, reason = backend.transition(action)
                if reason is not None:
                    break
        finally:
            backend.restore(snapshot)
            backend.last_physics_trace = saved_trace
            backend.last_control = saved_control
        projection = self.optimizer._sequence_rollout(  # noqa: SLF001 - same oracle contract
            backend, np.asarray(actions, dtype=np.float64)
        )
        diagnostics = replace(
            initial.diagnostics,
            predicted_cost=projection.cost,
            predicted_termination=projection.predicted_termination,
            predicted_gate_violation=projection.gate_violation,
            minimum_gate_margin=projection.minimum_gate_margin,
            near_axis_error_m=projection.near_axis_error_m,
            near_object_position_error_m=projection.near_object_position_error_m,
            near_object_rotation_error_rad=projection.near_object_rotation_error_rad,
            predicted_reference_complete=projection.reference_complete,
            predicted_contact_loss=projection.contact_loss,
            predicted_excessive_impulse=projection.excessive_impulse,
            predicted_contact_substeps=projection.contact_substeps,
            predicted_normal_impulse_ns=projection.normal_impulse_ns,
            predicted_max_penetration_m=projection.max_penetration_m,
            predicted_wrist_error=projection.wrist_error,
            action_first_difference=projection.first_difference,
            action_second_difference=projection.second_difference,
            action_effort=projection.effort,
            selection_lookahead=len(actions),
        )
        return replace(initial, result=projection, diagnostics=diagnostics)

    def action(self, backend: WorldWristFingerBackend) -> np.ndarray:
        remaining = backend.reference.frame_count - 1 - backend.reference_index
        portfolio = effective_horizon_portfolio(remaining)
        if not portfolio:
            raise ValueError("reference complete; adaptive oracle must not execute another action")
        candidates: list[HorizonCandidate] = []
        current_sequences: dict[int, np.ndarray] = {}
        for requested_horizon in portfolio:
            cross_horizon: list[np.ndarray] = []
            for source_horizon in sorted(current_sequences):
                cross_horizon.append(current_sequences[source_horizon])
            for source_horizon in sorted(self._warm_sequences):
                if source_horizon != requested_horizon:
                    cross_horizon.append(self._shift(self._warm_sequences[source_horizon]))
            warm = self._warm_sequences.get(requested_horizon)
            shifted_warm = None if warm is None else self._shift(warm)
            if remaining == 1:
                candidate = self.terminal_controller.candidate(
                    backend,
                    warm_sequence=shifted_warm,
                    seed_sequences=cross_horizon,
                )
            else:
                candidate = self.optimizer.optimize_horizon(
                    backend,
                    requested_horizon=requested_horizon,
                    warm_sequence=shifted_warm,
                    seed_sequences=cross_horizon,
                )
            candidates.append(candidate)
            current_sequences[requested_horizon] = candidate.sequence.copy()
        preview_lookahead = min(self.selection_lookahead, remaining)
        comparable_candidates = [
            self._closed_loop_viability_candidate(
                backend,
                candidate,
                lookahead=preview_lookahead,
            )
            for candidate in candidates
        ]
        selected, reason = self.selector.select(comparable_candidates)
        self._warm_sequences = current_sequences
        self.last_candidates = tuple(comparable_candidates)
        self.last_selected = selected
        self.last_reason = reason
        self.selection_trace.append(
            {
                "reference_index": backend.reference_index,
                "remaining": remaining,
                "portfolio": list(portfolio),
                "selector_common_viability_lookahead": preview_lookahead,
                "selector_projection": "closed_loop_receding_replans_without_padding",
                "selected_requested_horizon": selected.requested_horizon,
                "selected_effective_horizon": selected.effective_horizon,
                "reason": reason,
                "candidates": [candidate.as_dict() for candidate in comparable_candidates],
            }
        )
        return selected.sequence[0].copy()


__all__ = [
    "ContactAwareMPCConfig",
    "AdaptiveMultiHorizonContactOracle",
    "FormalGateThresholds",
    "GateFirstHorizonSelector",
    "HorizonCandidate",
    "TerminalSafeOneStepController",
    "WorldWristFingerObjectAwareOracle",
    "WorldWristOracleDiagnostics",
    "effective_horizon_portfolio",
    "formal_gate_barrier",
    "world_tracking_error",
]
