"""Bounded GPU batched local joint-space TVLQR for the explicit wrist."""

from __future__ import annotations

from dataclasses import dataclass

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
    ) -> dict[str, torch.Tensor]:
        if (dynamics_a is None) != (dynamics_b is None):
            raise ValueError("TVLQR needs both empirical A and B or neither")
        if dynamics_a is None:
            a, b = local_double_integrator(mass_wrist, dt_s)
            model_source = "live_mass_double_integrator"
        else:
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

    def compute(
        self,
        *,
        dynamics_a: torch.Tensor,
        dynamics_b: torch.Tensor,
        feedforward: torch.Tensor,
        q_wrist: torch.Tensor,
        qd_wrist: torch.Tensor,
        q_wrist_ref: torch.Tensor,
        qd_wrist_ref: torch.Tensor,
        model_source: str,
    ) -> dict[str, torch.Tensor | str]:
        if dynamics_a.shape != (q_wrist.shape[0], 12, 12) or dynamics_b.shape != (
            q_wrist.shape[0],
            12,
            6,
        ):
            raise ValueError("MPC empirical model shape invalid")
        count = q_wrist.shape[0]
        horizon = self.profile.horizon
        state = torch.cat((q_wrist - q_wrist_ref, qd_wrist - qd_wrist_ref), dim=-1)
        f, g = self._lifted_dynamics(dynamics_a, dynamics_b, horizon)
        q_block = torch.block_diag(*([self.q[0]] * horizon)).expand(count, -1, -1)
        r_block = torch.block_diag(*([self.r[0]] * horizon)).expand(count, -1, -1)
        hessian = g.mT @ q_block @ g + r_block
        linear = torch.bmm(g.mT @ q_block @ f, state.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(hessian).all() and torch.isfinite(linear).all()):
            raise RuntimeError("C3_MPC_NUMERIC_UNSTABLE")
        unconstrained = -torch.linalg.solve(hessian, linear.unsqueeze(-1)).squeeze(-1)
        lower = -self.profile.effort_limit - feedforward
        upper = self.profile.effort_limit - feedforward
        lower_sequence = lower.repeat(1, horizon)
        upper_sequence = upper.repeat(1, horizon)
        control = torch.clamp(unconstrained, min=lower_sequence, max=upper_sequence)
        for _ in range(self.profile.projected_gradient_iterations):
            gradient = torch.bmm(hessian, control.unsqueeze(-1)).squeeze(-1) + linear
            control = torch.clamp(
                control - self.profile.projected_gradient_step * gradient,
                min=lower_sequence,
                max=upper_sequence,
            )
        feedback = control[:, :6]
        command = feedforward + feedback
        applied = torch.clamp(command, -self.profile.effort_limit, self.profile.effort_limit)
        unconstrained_first = unconstrained[:, :6]
        saturation = (unconstrained_first < lower) | (unconstrained_first > upper)
        if not bool(torch.isfinite(applied).all() and torch.isfinite(control).all()):
            raise RuntimeError("C3_MPC_NUMERIC_UNSTABLE")
        return {
            "feedback": feedback,
            "command": command,
            "applied": applied,
            "saturation": saturation,
            "unconstrained_control": unconstrained_first,
            "model_source": model_source,
        }


__all__ = [
    "BoundedTVLQRWristControllerV1",
    "BoundedMPCWristControllerV1",
    "BoundedMPCWristProfileV1",
    "BoundedTVLQRWristProfileV1",
    "ExplicitWristLocalDynamicsIdentifierV1",
    "finite_horizon_tvlqr_gain",
    "local_double_integrator",
]
