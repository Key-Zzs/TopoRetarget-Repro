"""Sequential bound-constrained Eq. (2) warm-start solver."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .bones import BoneDirectionProfile, BoneFeatures, extract_bone_features
from .frames import BoneDirectionFrameProfile
from .objectives import BoneDirectionObjective, BoneDirectionResidual


class WarmStartSolveError(RuntimeError):
    """Raised by the strict solver on the first unsuccessful frame."""


@dataclass(frozen=True)
class WarmStartSolverProfile:
    profile_id: str
    version: str
    backend: str
    method: str
    dtype: str
    max_nfev: int
    ftol: float
    xtol: float
    gtol: float
    jacobian_mode: str
    sequential: bool
    strict_failure_policy: str
    first_frame_profile: str
    paper_weights_source: str
    finite_difference_epsilon: float
    assumptions: tuple[str, ...]
    profile_hash: str
    source_path: Path | None = None

    @property
    def sha256(self) -> str:
        return self.profile_hash

    @classmethod
    def from_mapping(
        cls, values: dict[str, Any], *, source_path: Path | None = None, raw: bytes | None = None
    ) -> WarmStartSolverProfile:
        result = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            backend=str(values["backend"]),
            method=str(values["method"]),
            dtype=str(values.get("dtype", "float64")),
            max_nfev=int(values.get("max_nfev", 250)),
            ftol=float(values.get("ftol", 1e-12)),
            xtol=float(values.get("xtol", 1e-12)),
            gtol=float(values.get("gtol", 1e-12)),
            jacobian_mode=str(values.get("jacobian_mode", "torch_autograd")),
            sequential=bool(values.get("sequential", True)),
            strict_failure_policy=str(values.get("strict_failure_policy", "fail_fast")),
            first_frame_profile=str(values.get("first_frame_profile", "neutral")),
            paper_weights_source=str(
                values.get("paper_weights_source", "configs/paper/retarget.yaml")
            ),
            finite_difference_epsilon=float(values.get("finite_difference_epsilon", 1e-6)),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            profile_hash=hashlib.sha256(raw or b"").hexdigest() if raw is not None else "",
            source_path=source_path,
        )
        result.validate()
        return result

    def validate(self) -> WarmStartSolverProfile:
        if self.backend != "scipy.optimize.least_squares" or self.method != "trf":
            raise ValueError("Stage 7 paper solver profile requires scipy least_squares/trf")
        if self.dtype != "float64" or self.jacobian_mode != "torch_autograd":
            raise ValueError("Stage 7 reference solver requires float64 torch autograd")
        if self.max_nfev <= 0 or min(self.ftol, self.xtol, self.gtol) <= 0:
            raise ValueError("invalid solver tolerances")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "backend": self.backend,
            "method": self.method,
            "dtype": self.dtype,
            "max_nfev": self.max_nfev,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "jacobian_mode": self.jacobian_mode,
            "sequential": self.sequential,
            "strict_failure_policy": self.strict_failure_policy,
            "first_frame_profile": self.first_frame_profile,
            "paper_weights_source": self.paper_weights_source,
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "assumptions": list(self.assumptions),
            "profile_hash": self.profile_hash,
        }


@dataclass
class FrameSolveResult:
    qpos: np.ndarray
    initial_qpos: np.ndarray
    initial_ebone: float
    final_ebone: float
    temporal_term: float
    total_objective: float
    initial_total_objective: float
    pair_residual: np.ndarray
    robot_features: BoneFeatures
    status: int
    success: bool
    message: str
    nfev: int
    njev: int
    solve_time_s: float


@dataclass
class SequenceSolveResult:
    source_features: BoneFeatures
    qpos: np.ndarray
    initial_qpos: np.ndarray
    initial_ebone: np.ndarray
    final_ebone: np.ndarray
    temporal_term: np.ndarray
    total_objective: np.ndarray
    initial_total_objective: np.ndarray
    pair_residuals: np.ndarray
    robot_features: list[BoneFeatures]
    solver_status: np.ndarray
    solver_success: np.ndarray
    solver_messages: list[str]
    nfev: np.ndarray
    njev: np.ndarray
    solve_time_s: np.ndarray


def load_solver_profile(
    profile_id: str, *, config_root: str | Path | None = None
) -> WarmStartSolverProfile:
    root = (
        Path(config_root)
        if config_root is not None
        else Path(__file__).resolve().parents[3] / "configs" / "retarget" / "warm_start"
    )
    path = root / f"{profile_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"solver profile not found: {profile_id}")
    raw = path.read_bytes()
    values = yaml.safe_load(raw) or {}
    if not isinstance(values, dict):
        raise ValueError(f"solver profile must be a mapping: {path}")
    return WarmStartSolverProfile.from_mapping(values, source_path=path, raw=raw)


def load_paper_weights(repo_root: str | Path | None = None) -> tuple[float, float, str]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    path = root / "configs" / "paper" / "retarget.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # The two aliases are resolved from the single paper config, never copied
    # into solver code or a second config file.
    warm = values.get("lambda_warm")
    smooth = values.get("lambda_smooth")
    if warm is None:
        warm = values["lambda_warm_bone"]
    if smooth is None:
        smooth = values["lambda_initial_temporal_smoothness"]
    return float(warm), float(smooth), str(path)


def _torch_jacobian(objective: BoneDirectionObjective, qpos: np.ndarray) -> np.ndarray:
    import torch

    q = torch.as_tensor(qpos, dtype=torch.float64).detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(objective.residual_tensor, q, create_graph=False)
    return jacobian.detach().cpu().numpy()


def solve_frame(
    source_feature: Any,
    robot_model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    solver_profile: WarmStartSolverProfile,
    *,
    side: str,
    initial_qpos: np.ndarray,
    previous_qpos: np.ndarray | None,
    lambda_warm: float,
    lambda_smooth: float,
) -> FrameSolveResult:
    from scipy.optimize import least_squares

    residual_model = BoneDirectionResidual(
        source_feature, frame_profile, bone_profile, robot_model, side
    )
    objective = BoneDirectionObjective(residual_model, lambda_warm, lambda_smooth, previous_qpos)
    q0 = np.asarray(initial_qpos, dtype=np.float64).copy()
    lower = np.asarray(robot_model.joint_lower, dtype=np.float64)
    upper = np.asarray(robot_model.joint_upper, dtype=np.float64)
    q0 = np.minimum(np.maximum(q0, lower), upper)

    def residual_numpy(q: np.ndarray) -> np.ndarray:
        import torch

        value = objective.residual_tensor(torch.as_tensor(q, dtype=torch.float64))
        return value.detach().cpu().numpy()

    def jacobian_numpy(q: np.ndarray) -> np.ndarray:
        return _torch_jacobian(objective, q)

    initial_report = objective.paper_objective(
        __import__("torch").as_tensor(q0, dtype=__import__("torch").float64)
    )
    started = time.perf_counter()
    result = least_squares(
        residual_numpy,
        q0,
        jac=jacobian_numpy,
        bounds=(lower, upper),
        method=solver_profile.method,
        max_nfev=solver_profile.max_nfev,
        ftol=solver_profile.ftol,
        xtol=solver_profile.xtol,
        gtol=solver_profile.gtol,
    )
    elapsed = time.perf_counter() - started
    import torch

    final_report = objective.paper_objective(torch.as_tensor(result.x, dtype=torch.float64))
    robot_features = residual_model.robot_features(torch.as_tensor(result.x, dtype=torch.float64))
    residual = final_report["bone_residual"].detach().cpu().numpy()
    frame_result = FrameSolveResult(
        qpos=np.asarray(result.x, dtype=np.float64),
        initial_qpos=q0,
        initial_ebone=float(initial_report["ebone"].detach().cpu()),
        final_ebone=float(final_report["ebone"].detach().cpu()),
        temporal_term=float(final_report["temporal"].detach().cpu()),
        total_objective=float(final_report["total"].detach().cpu()),
        initial_total_objective=float(initial_report["total"].detach().cpu()),
        pair_residual=residual,
        robot_features=robot_features,
        status=int(result.status),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        njev=int(result.njev or 0),
        solve_time_s=float(elapsed),
    )
    if not frame_result.success and solver_profile.strict_failure_policy == "fail_fast":
        raise WarmStartSolveError(
            "warm-start solver failed: "
            f"frame status={frame_result.status}, message={frame_result.message}"
        )
    return frame_result


def solve_sequence(
    source_keypoints: Any,
    robot_model: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    solver_profile: WarmStartSolverProfile,
    *,
    side: str,
    lambda_warm: float,
    lambda_smooth: float,
) -> SequenceSolveResult:
    source_features = extract_bone_features(
        source_keypoints, frame_profile, bone_profile, side=side, strict=True
    )
    frame_count = int(source_keypoints.shape[0])
    q_values: list[np.ndarray] = []
    initial_values: list[np.ndarray] = []
    initial_ebone: list[float] = []
    final_ebone: list[float] = []
    temporal: list[float] = []
    total: list[float] = []
    initial_total: list[float] = []
    residuals: list[np.ndarray] = []
    robot_features: list[BoneFeatures] = []
    statuses: list[int] = []
    successes: list[bool] = []
    messages: list[str] = []
    nfev: list[int] = []
    njev: list[int] = []
    elapsed: list[float] = []
    previous: np.ndarray | None = None
    for frame in range(frame_count):
        q0 = (
            robot_model.neutral_q if previous is None or not solver_profile.sequential else previous
        )
        result = solve_frame(
            source_features.adjacent_features[frame],
            robot_model,
            frame_profile,
            bone_profile,
            solver_profile,
            side=side,
            initial_qpos=q0,
            previous_qpos=None if previous is None or not solver_profile.sequential else previous,
            lambda_warm=lambda_warm,
            lambda_smooth=lambda_smooth,
        )
        q_values.append(result.qpos)
        initial_values.append(result.initial_qpos)
        initial_ebone.append(result.initial_ebone)
        final_ebone.append(result.final_ebone)
        temporal.append(result.temporal_term)
        total.append(result.total_objective)
        initial_total.append(result.initial_total_objective)
        residuals.append(result.pair_residual)
        robot_features.append(result.robot_features)
        statuses.append(result.status)
        successes.append(result.success)
        messages.append(result.message)
        nfev.append(result.nfev)
        njev.append(result.njev)
        elapsed.append(result.solve_time_s)
        if result.success:
            previous = result.qpos
        elif solver_profile.sequential:
            previous = None
    return SequenceSolveResult(
        source_features=source_features,
        qpos=np.stack(q_values),
        initial_qpos=np.stack(initial_values),
        initial_ebone=np.asarray(initial_ebone),
        final_ebone=np.asarray(final_ebone),
        temporal_term=np.asarray(temporal),
        total_objective=np.asarray(total),
        initial_total_objective=np.asarray(initial_total),
        pair_residuals=np.stack(residuals),
        robot_features=robot_features,
        solver_status=np.asarray(statuses),
        solver_success=np.asarray(successes, dtype=bool),
        solver_messages=messages,
        nfev=np.asarray(nfev),
        njev=np.asarray(njev),
        solve_time_s=np.asarray(elapsed),
    )


__all__ = [
    "FrameSolveResult",
    "SequenceSolveResult",
    "WarmStartSolveError",
    "WarmStartSolverProfile",
    "load_paper_weights",
    "load_solver_profile",
    "solve_frame",
    "solve_sequence",
]
