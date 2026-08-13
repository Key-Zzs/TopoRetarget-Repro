"""Bounded GPU batched local joint-space TVLQR for the explicit wrist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch


@dataclass(frozen=True)
class BoundedTVLQRWristProfileV1:
    identifier: str = "bounded_tvlqr_wrist_v1"
    horizon: int = 5
    effort_limit: float = 500.0
    q_translation: float = 250.0
    q_rotation: float = 100.0
    q_velocity: float = 2.0
    r_effort: float = 0.02


@dataclass(frozen=True)
class BoundedMPCWristProfileV1:
    """One fixed box-constrained preview configuration, not a tuning family."""

    identifier: str = "bounded_mpc_wrist_v1"
    horizon: int = 5
    effort_limit: float = 500.0
    q_translation: float = 250.0
    q_rotation: float = 100.0
    q_velocity: float = 2.0
    r_effort: float = 0.02
    projected_gradient_iterations: int = 8
    projected_gradient_step: float = 0.1


@dataclass(frozen=True)
class ExplicitWristLocalDynamicsIdentifierV1:
    """Immutable finite-difference identification contract for Path B."""

    identifier: str = "ExplicitWristLocalDynamicsIdentifierV1"
    state: str = "[q_wrist-q_ref,qdot_wrist-qdot_ref]"
    action: str = "joint_effort_wrist"
    num_envs: int = 12
    state_position_perturbation: float = 1.0e-4
    state_velocity_perturbation: float = 1.0e-3
    effort_perturbation: float = 1.0


@dataclass(frozen=True)
class ExplicitWristLocalDynamicsIdentifierV2:
    """Unit-scaled multi-axis identification at every physics substep."""

    identifier: str = "ExplicitWristLocalDynamicsIdentifierV2"
    state: str = "[q_wrist-q_ref,qdot_wrist-qdot_ref]"
    action: str = "joint_effort_wrist-u_nominal"
    design: str = "32_direction_hadamard_central_difference"
    state_scale_fraction: float = 0.05
    effort_scale_fraction: float = 0.002
    direction_count: int = 32
    num_envs: int = 65


class BoundedTVLQRResult(TypedDict):
    a: torch.Tensor
    b: torch.Tensor
    gain: torch.Tensor
    feedback: torch.Tensor
    command: torch.Tensor
    applied: torch.Tensor
    saturation: torch.Tensor
    model_source: str


def local_double_integrator(
    mass_wrist: torch.Tensor, dt_s: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU local model x=[dq,dqd], u=tau around the live reference."""

    count = mass_wrist.shape[0]
    inverse = torch.linalg.inv(mass_wrist)
    eye = torch.eye(6, device=mass_wrist.device, dtype=mass_wrist.dtype).expand(count, -1, -1)
    zero = torch.zeros_like(eye)
    a = torch.cat((torch.cat((eye, dt_s * eye), dim=-1), torch.cat((zero, eye), dim=-1)), dim=-2)
    b = torch.cat((0.5 * dt_s * dt_s * inverse, dt_s * inverse), dim=-2)
    return a, b


def finite_horizon_tvlqr_gain(
    a: torch.Tensor, b: torch.Tensor, q: torch.Tensor, r: torch.Tensor, horizon: int
) -> torch.Tensor:
    """Batched Riccati recursion returning the first finite-horizon gain."""

    p = q
    gain = torch.zeros((a.shape[0], 6, 12), dtype=a.dtype, device=a.device)
    for _ in range(horizon):
        solve = r + b.mT @ p @ b
        gain = torch.linalg.solve(solve, b.mT @ p @ a)
        p = q + a.mT @ p @ (a - b @ gain)
    return gain


class BoundedTVLQRWristControllerV1:
    """One global cost/horizon TVLQR controller with strict box effort bounds."""

    def __init__(self, profile: BoundedTVLQRWristProfileV1, *, device: torch.device | str):
        self.profile = profile
        q_diag = [profile.q_translation] * 3 + [profile.q_rotation] * 3 + [profile.q_velocity] * 6
        self.q = torch.diag(torch.tensor(q_diag, dtype=torch.float32, device=device))[None]
        self.r = torch.eye(6, dtype=torch.float32, device=device)[None] * profile.r_effort

    def compute(
        self,
        *,
        mass_wrist: torch.Tensor,
        feedforward: torch.Tensor,
        q_wrist: torch.Tensor,
        qd_wrist: torch.Tensor,
        q_wrist_ref: torch.Tensor,
        qd_wrist_ref: torch.Tensor,
        dt_s: float,
        dynamics_a: torch.Tensor | None = None,
        dynamics_b: torch.Tensor | None = None,
    ) -> BoundedTVLQRResult:
        if (dynamics_a is None) != (dynamics_b is None):
            raise ValueError("TVLQR needs both empirical A and B or neither")
        if dynamics_a is None:
            a, b = local_double_integrator(mass_wrist, dt_s)
            model_source = "live_mass_double_integrator"
        else:
            assert dynamics_b is not None
            a, b = dynamics_a, dynamics_b
            if a.shape != (mass_wrist.shape[0], 12, 12) or b.shape != (
                mass_wrist.shape[0],
                12,
                6,
            ):
                raise ValueError(f"TVLQR empirical model shape invalid: {a.shape}, {b.shape}")
            model_source = "gpu_finite_difference_identification"
        gain = finite_horizon_tvlqr_gain(
            a,
            b,
            self.q.expand(a.shape[0], -1, -1),
            self.r.expand(a.shape[0], -1, -1),
            self.profile.horizon,
        )
        state_error = torch.cat((q_wrist - q_wrist_ref, qd_wrist - qd_wrist_ref), dim=-1)
        feedback = -torch.bmm(gain, state_error.unsqueeze(-1)).squeeze(-1)
        command = feedforward + feedback
        limit = torch.full_like(command, self.profile.effort_limit)
        applied = torch.clamp(command, min=-limit, max=limit)
        return {
            "a": a,
            "b": b,
            "gain": gain,
            "feedback": feedback,
            "command": command,
            "applied": applied,
            "saturation": command.abs() > limit,
            "model_source": model_source,
        }


class BoundedMPCWristControllerV1:
    """GPU batched finite-horizon QP with immutable per-joint effort boxes."""

    def __init__(self, profile: BoundedMPCWristProfileV1, *, device: torch.device | str):
        self.profile = profile
        q_diag = [profile.q_translation] * 3 + [profile.q_rotation] * 3 + [profile.q_velocity] * 6
        self.q = torch.diag(torch.tensor(q_diag, dtype=torch.float32, device=device))[None]
        self.r = torch.eye(6, dtype=torch.float32, device=device)[None] * profile.r_effort

    @staticmethod
    def _lifted_dynamics(
        a: torch.Tensor, b: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count, state_size, action_size = a.shape[0], a.shape[1], b.shape[2]
        powers = [torch.eye(state_size, dtype=a.dtype, device=a.device).expand(count, -1, -1)]
        for _ in range(horizon):
            powers.append(torch.bmm(a, powers[-1]))
        f = torch.cat(powers[1:], dim=1)
        g = torch.zeros(
            (count, horizon * state_size, horizon * action_size), dtype=a.dtype, device=a.device
        )
        for time in range(horizon):
            rows = slice(time * state_size, (time + 1) * state_size)
            for control in range(time + 1):
                cols = slice(control * action_size, (control + 1) * action_size)
                g[:, rows, cols] = torch.bmm(powers[time - control], b)
        return f, g

    @staticmethod
    def _lifted_time_varying_dynamics(
        a: torch.Tensor, b: torch.Tensor, affine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lift ``x+ = A_t x + B_t du + c_t`` over one batched horizon."""

        if a.ndim != 4 or b.ndim != 4 or affine.ndim != 3:
            raise ValueError("time-varying MPC model must have [batch,horizon,...] tensors")
        count, horizon, state_size, _ = a.shape
        action_size = b.shape[-1]
        if b.shape != (count, horizon, state_size, action_size) or affine.shape != (
            count,
            horizon,
            state_size,
        ):
            raise ValueError("time-varying MPC model shapes disagree")
        transition = torch.eye(state_size, dtype=a.dtype, device=a.device).expand(count, -1, -1)
        offset = torch.zeros((count, state_size), dtype=a.dtype, device=a.device)
        f = torch.empty((count, horizon * state_size, state_size), dtype=a.dtype, device=a.device)
        g = torch.zeros(
            (count, horizon * state_size, horizon * action_size),
            dtype=a.dtype,
            device=a.device,
        )
        d = torch.empty((count, horizon * state_size), dtype=a.dtype, device=a.device)
        responses: list[torch.Tensor] = []
        for time in range(horizon):
            transition = torch.bmm(a[:, time], transition)
            offset = torch.bmm(a[:, time], offset.unsqueeze(-1)).squeeze(-1) + affine[:, time]
            responses = [torch.bmm(a[:, time], response) for response in responses]
            responses.append(b[:, time])
            rows = slice(time * state_size, (time + 1) * state_size)
            f[:, rows] = transition
            d[:, rows] = offset
            for control, response in enumerate(responses):
                cols = slice(control * action_size, (control + 1) * action_size)
                g[:, rows, cols] = response
        return f, g, d

    def compute(
        self,
        *,
        dynamics_a: torch.Tensor,
        dynamics_b: torch.Tensor,
        feedforward: torch.Tensor | None,
        q_wrist: torch.Tensor,
        qd_wrist: torch.Tensor,
        q_wrist_ref: torch.Tensor,
        qd_wrist_ref: torch.Tensor,
        model_source: str,
        dynamics_affine: torch.Tensor | None = None,
        nominal_effort_sequence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str]:
        count = q_wrist.shape[0]
        horizon = self.profile.horizon
        state = torch.cat((q_wrist - q_wrist_ref, qd_wrist - qd_wrist_ref), dim=-1)
        if dynamics_a.ndim == 3:
            if dynamics_a.shape != (count, 12, 12) or dynamics_b.shape != (count, 12, 6):
                raise ValueError("MPC empirical model shape invalid")
            f, g = self._lifted_dynamics(dynamics_a, dynamics_b, horizon)
            lifted_affine = torch.zeros((count, horizon * 12), device=state.device)
            if feedforward is None or feedforward.shape != (count, 6):
                raise ValueError("stationary MPC requires one feedforward effort")
            nominal_effort = feedforward[:, None].expand(-1, horizon, -1)
        elif dynamics_a.ndim == 4:
            if dynamics_a.shape != (count, horizon, 12, 12) or dynamics_b.shape != (
                count,
                horizon,
                12,
                6,
            ):
                raise ValueError("time-varying MPC empirical model shape invalid")
            if dynamics_affine is None or nominal_effort_sequence is None:
                raise ValueError("time-varying MPC requires affine and nominal-effort sequences")
            if nominal_effort_sequence.shape != (count, horizon, 6):
                raise ValueError("time-varying MPC nominal-effort shape invalid")
            f, g, lifted_affine = self._lifted_time_varying_dynamics(
                dynamics_a, dynamics_b, dynamics_affine
            )
            nominal_effort = nominal_effort_sequence
        else:
            raise ValueError("MPC empirical model rank invalid")
        q_block = torch.block_diag(*([self.q[0]] * horizon)).expand(count, -1, -1)
        r_block = torch.block_diag(*([self.r[0]] * horizon)).expand(count, -1, -1)
        hessian = g.mT @ q_block @ g + r_block
        uncontrolled = torch.bmm(f, state.unsqueeze(-1)).squeeze(-1) + lifted_affine
        linear = torch.bmm(g.mT @ q_block, uncontrolled.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(hessian).all() and torch.isfinite(linear).all()):
            raise RuntimeError("C3_MPC_NUMERIC_UNSTABLE")
        unconstrained = -torch.linalg.solve(hessian, linear.unsqueeze(-1)).squeeze(-1)
        lower_sequence = (-self.profile.effort_limit - nominal_effort).flatten(1)
        upper_sequence = (self.profile.effort_limit - nominal_effort).flatten(1)
        control = torch.clamp(unconstrained, min=lower_sequence, max=upper_sequence)
        # The profile value is an upper bound, not an unconditional step.  The
        # old fixed 0.1 step violated the projected-gradient stability bound at
        # every identified node (alpha * lambda_max ranged from 2.15 to 20.10).
        # Capping it by 1 / lambda_max preserves the frozen objective, horizon,
        # iteration count, and effort box while making each descent step
        # spectrally valid for the live Hessian.
        hessian_lambda_max = torch.linalg.eigvalsh(hessian).amax(dim=-1)
        projected_gradient_step = torch.minimum(
            torch.full_like(hessian_lambda_max, self.profile.projected_gradient_step),
            hessian_lambda_max.reciprocal(),
        )
        for _ in range(self.profile.projected_gradient_iterations):
            gradient = torch.bmm(hessian, control.unsqueeze(-1)).squeeze(-1) + linear
            control = torch.clamp(
                control - projected_gradient_step[:, None] * gradient,
                min=lower_sequence,
                max=upper_sequence,
            )
        feedback = control[:, :6]
        feedforward_first = nominal_effort[:, 0]
        command = feedforward_first + feedback
        applied = torch.clamp(command, -self.profile.effort_limit, self.profile.effort_limit)
        unconstrained_first = unconstrained[:, :6]
        projected_first = control[:, :6]
        lower_first = lower_sequence[:, :6]
        upper_first = upper_sequence[:, :6]
        # Report the constraint that is active on the command actually sent to
        # PhysX.  The old flag inspected the unconstrained solution, which can
        # disagree with the eight-iteration projected solution and therefore
        # under-report the frozen saturation gate.
        boundary_tolerance = 1.0e-4
        saturation = (projected_first <= lower_first + boundary_tolerance) | (
            projected_first >= upper_first - boundary_tolerance
        )
        if not bool(torch.isfinite(applied).all() and torch.isfinite(control).all()):
            raise RuntimeError("C3_MPC_NUMERIC_UNSTABLE")
        return {
            "state_error": state,
            "hessian": hessian,
            "linear": linear,
            "lifted_affine": lifted_affine,
            "nominal_effort_sequence": nominal_effort,
            "feedback": feedback,
            "command": command,
            "applied": applied,
            "saturation": saturation,
            "unconstrained_control": unconstrained_first,
            "unconstrained_control_sequence": unconstrained,
            "projected_control_sequence": control,
            "lower_effort_delta": lower_first,
            "upper_effort_delta": upper_first,
            "hessian_lambda_max": hessian_lambda_max,
            "projected_gradient_step": projected_gradient_step,
            "projected_gradient_stability_product": (projected_gradient_step * hessian_lambda_max),
            "model_source": model_source,
        }


__all__ = [
    "BoundedTVLQRWristControllerV1",
    "BoundedMPCWristControllerV1",
    "BoundedMPCWristProfileV1",
    "BoundedTVLQRWristProfileV1",
    "ExplicitWristLocalDynamicsIdentifierV1",
    "ExplicitWristLocalDynamicsIdentifierV2",
    "finite_horizon_tvlqr_gain",
    "local_double_integrator",
]
