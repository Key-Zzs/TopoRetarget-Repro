"""Stage 9 Eq. (8)-(9) constrained interaction-preserving refinement.

This module deliberately keeps the paper's optimization terms small and explicit.
The graph, bone features, collision samples, and SDF are supplied by earlier
stages; no graph is rebuilt and no semantic contact labels are consulted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.storage import direct_zarr3_arrays, write_zarr3_group_direct
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.robot_surface import (
    RobotSurfaceSampleSet,
    RobotSurfaceSamplingProfile,
)
from toporetarget.geometry.signed_distance.base import SignedDistanceQueryResult
from toporetarget.geometry.signed_distance.reference import (
    ReferenceSignedDistanceBackend,
)
from toporetarget.retarget.artifacts import WarmStartTrajectory
from toporetarget.retarget.bones import (
    BoneDirectionProfile,
    BoneFeatures,
    extract_bone_features,
)
from toporetarget.retarget.frames import BoneDirectionFrameProfile
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
)
from toporetarget.retarget.interaction_graph import (
    InteractionGraphFrame,
    InteractionGraphTrajectory,
)
from toporetarget.retarget.interaction_objective import InteractionMeshResidual
from toporetarget.retarget.refinement_performance import (
    RefinementEvaluationCache,
    TimerBook,
)

FINAL_REFINEMENT_SCHEMA_VERSION_V1 = "toporetarget.final_retarget.v1"
FINAL_REFINEMENT_SCHEMA_VERSION_V2 = "toporetarget.final_retarget.v2"
# Historical callers and v1 fixtures continue to use this public alias.
FINAL_REFINEMENT_SCHEMA_VERSION = FINAL_REFINEMENT_SCHEMA_VERSION_V1
COORDINATE_PROFILE_ID = "local_seed_delta_v1"
FULL_QUERY_PROFILE_ID = "full_collision_surface_reference_v1"
ACTIVE_QUERY_PROFILE_ID = "adaptive_active_set_v1"
SOLVER_PROFILE_ID = "scipy_slsqp_active_set_v1"
CONTACT_RICH_SOLVER_PROFILE_ID = "scipy_slsqp_active_set_contact_rich_v2"
FULL_SOLVER_PROFILE_ID = "scipy_slsqp_full_surface_reference_v1"
STRICT_ACCEPTANCE_POLICY_ID = "strict_optimizer_converged_and_audits_v1"
DEFERRED_STATIONARITY_POLICY_ID = "feasible_stationary_v1_deferred"


def _as_np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    )


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value, _sha256_bytes(raw)


@dataclass(frozen=True)
class RefinementCoordinateProfile:
    profile_id: str
    version: str
    base_parameterization: str
    base_prior_reference: str
    previous_state_mapping: str
    first_frame_policy: str
    rotation_vector_units: str
    translation_units: str
    assumptions: tuple[str, ...]
    profile_hash: str
    source_path: Path | None = None

    def validate(self) -> RefinementCoordinateProfile:
        if self.base_parameterization != "scene_local_seed_delta_exp_left":
            raise ValueError("unsupported refinement base parameterization")
        if self.base_prior_reference != "seed_delta":
            raise ValueError("unsupported base prior reference")
        if self.previous_state_mapping != "previous_final_remapped_to_current_seed":
            raise ValueError("unsupported previous-state mapping")
        if self.first_frame_policy != "warm_start_zero_delta_no_temporal":
            raise ValueError("unsupported first-frame policy")
        if self.rotation_vector_units != "radians" or self.translation_units != "meters":
            raise ValueError("refinement coordinates must use raw meters and radians")
        return self

    @classmethod
    def load(cls, profile_id: str, root: Path | None = None) -> RefinementCoordinateProfile:
        repo = root or Path(__file__).resolve().parents[3]
        path = repo / "configs" / "retarget" / "refinement" / f"{profile_id}.yaml"
        values, digest = _load_yaml(path)
        result = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            base_parameterization=str(values["base_parameterization"]),
            base_prior_reference=str(values["base_prior_reference"]),
            previous_state_mapping=str(values["previous_state_mapping"]),
            first_frame_policy=str(values["first_frame_policy"]),
            rotation_vector_units=str(values.get("rotation_vector_units", "radians")),
            translation_units=str(values.get("translation_units", "meters")),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            profile_hash=digest,
            source_path=path,
        )
        return result.validate()

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "base_parameterization": self.base_parameterization,
            "base_prior_reference": self.base_prior_reference,
            "previous_state_mapping": self.previous_state_mapping,
            "first_frame_policy": self.first_frame_policy,
            "rotation_vector_units": self.rotation_vector_units,
            "translation_units": self.translation_units,
            "assumptions": list(self.assumptions),
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class CollisionQueryProfile:
    profile_id: str
    version: str
    mode: str
    active_margin_m: float
    max_active_set_rounds: int
    paper_status: str
    assumptions: tuple[str, ...]
    profile_hash: str
    source_path: Path | None = None

    def validate(self) -> CollisionQueryProfile:
        if self.mode not in {"full", "adaptive"}:
            raise ValueError("query profile mode must be full or adaptive")
        if self.mode == "adaptive" and self.active_margin_m <= 0:
            raise ValueError("adaptive active margin must be positive")
        if self.max_active_set_rounds <= 0:
            raise ValueError("max_active_set_rounds must be positive")
        return self

    @classmethod
    def load(cls, profile_id: str, root: Path | None = None) -> CollisionQueryProfile:
        repo = root or Path(__file__).resolve().parents[3]
        path = repo / "configs" / "retarget" / "collision_queries" / f"{profile_id}.yaml"
        values, digest = _load_yaml(path)
        result = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            mode=str(values["mode"]),
            active_margin_m=float(values.get("active_margin_m", 0.010)),
            max_active_set_rounds=int(values.get("max_active_set_rounds", 5)),
            paper_status=str(values.get("paper_status", "not_paper_specified")),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            profile_hash=digest,
            source_path=path,
        )
        return result.validate()

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "mode": self.mode,
            "active_margin_m": self.active_margin_m,
            "max_active_set_rounds": self.max_active_set_rounds,
            "paper_status": self.paper_status,
            "assumptions": list(self.assumptions),
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class RefinementSolverProfile:
    profile_id: str
    version: str
    backend: str
    method: str
    dtype: str
    maxiter: int
    ftol: float
    constraint_tolerance: float
    finite_difference_epsilon: float
    disp: bool
    strict_failure_policy: str
    termination_contract: str
    acceptance_policy_id: str
    active_set_continuation_policy: str
    maxiter_provenance: dict[str, Any]
    stationarity_policy: str
    benchmark_grid: tuple[int, ...]
    sdf_backend: str
    sdf_probe_count: int
    sdf_cross_validation_tolerance_m: float
    assumptions: tuple[str, ...]
    profile_hash: str
    source_path: Path | None = None

    @classmethod
    def load(cls, profile_id: str, root: Path | None = None) -> RefinementSolverProfile:
        repo = root or Path(__file__).resolve().parents[3]
        path = repo / "configs" / "retarget" / "refinement_solvers" / f"{profile_id}.yaml"
        values, digest = _load_yaml(path)
        result = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            backend=str(values["backend"]),
            method=str(values["method"]),
            dtype=str(values.get("dtype", "float64")),
            maxiter=int(values.get("maxiter", 100)),
            ftol=float(values.get("ftol", 1e-10)),
            constraint_tolerance=float(values.get("constraint_tolerance", 1e-6)),
            finite_difference_epsilon=float(values.get("finite_difference_epsilon", 1e-6)),
            disp=bool(values.get("disp", False)),
            strict_failure_policy=str(values.get("strict_failure_policy", "fail_fast")),
            termination_contract=str(
                values.get("termination_contract", "strict_result_success_and_primal_audits_v1")
            ),
            acceptance_policy_id=str(
                values.get("acceptance_policy_id", STRICT_ACCEPTANCE_POLICY_ID)
            ),
            active_set_continuation_policy=str(
                values.get("active_set_continuation_policy", "warm_seed_reinitialized_v1")
            ),
            maxiter_provenance=dict(
                values.get(
                    "maxiter_provenance",
                    {
                        "source": "repository_profile",
                        "status": "not_benchmarked",
                    },
                )
            ),
            stationarity_policy=str(
                values.get("stationarity_policy", DEFERRED_STATIONARITY_POLICY_ID)
            ),
            benchmark_grid=tuple(
                int(item) for item in values.get("benchmark_grid", (30, 60, 100, 200, 400))
            ),
            sdf_backend=str(values.get("sdf_backend", "reference")),
            sdf_probe_count=int(values.get("sdf_probe_count", 32)),
            sdf_cross_validation_tolerance_m=float(
                values.get("sdf_cross_validation_tolerance_m", 1e-8)
            ),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            profile_hash=digest,
            source_path=path,
        )
        if result.backend != "scipy.optimize.minimize" or result.method != "SLSQP":
            raise ValueError("Stage 9 reference profile requires scipy SLSQP")
        if result.dtype != "float64" or result.maxiter <= 0:
            raise ValueError("invalid Stage 9 solver profile")
        if result.acceptance_policy_id == "" or result.termination_contract == "":
            raise ValueError(
                "Stage 9 solver profile must declare acceptance and termination contracts"
            )
        if result.maxiter_provenance.get("source") is None:
            raise ValueError("Stage 9 solver profile must declare maxiter provenance")
        if any(item <= 0 for item in result.benchmark_grid):
            raise ValueError("Stage 9 benchmark grid must contain positive iteration budgets")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "backend": self.backend,
            "method": self.method,
            "dtype": self.dtype,
            "maxiter": self.maxiter,
            "ftol": self.ftol,
            "constraint_tolerance": self.constraint_tolerance,
            "finite_difference_epsilon": self.finite_difference_epsilon,
            "disp": self.disp,
            "strict_failure_policy": self.strict_failure_policy,
            "termination_contract": self.termination_contract,
            "acceptance_policy_id": self.acceptance_policy_id,
            "active_set_continuation_policy": self.active_set_continuation_policy,
            "maxiter_provenance": self.maxiter_provenance,
            "stationarity_policy": self.stationarity_policy,
            "benchmark_grid": list(self.benchmark_grid),
            "sdf_backend": self.sdf_backend,
            "sdf_probe_count": self.sdf_probe_count,
            "sdf_cross_validation_tolerance_m": self.sdf_cross_validation_tolerance_m,
            "assumptions": list(self.assumptions),
            "profile_hash": self.profile_hash,
        }


@dataclass(frozen=True)
class PaperRefinementWeights:
    lambda_im: float
    lambda_bone: float
    lambda_reg: float
    lambda_base_pos: float
    lambda_base_rot: float
    tau: float
    b: float
    w_s: float
    config_path: Path
    config_hash: str

    @classmethod
    def load(cls, repo_root: Path | None = None) -> PaperRefinementWeights:
        root = repo_root or Path(__file__).resolve().parents[3]
        path = root / "configs" / "paper" / "retarget.yaml"
        values, digest = _load_yaml(path)
        result = cls(
            lambda_im=float(values["lambda_interaction_mesh"]),
            lambda_bone=float(values["lambda_refinement_bone"]),
            lambda_reg=float(values["lambda_refinement_temporal_regularization"]),
            lambda_base_pos=float(values["lambda_base_position"]),
            lambda_base_rot=float(values["lambda_base_rotation"]),
            tau=float(values["penetration_soft_tolerance_m"]),
            b=float(values["penetration_hard_bound_m"]),
            w_s=float(values["slack_penalty_weight"]),
            config_path=path,
            config_hash=digest,
        )
        if result.b <= result.tau:
            raise ValueError("paper hard penetration bound must exceed soft tolerance")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "lambda_IM": self.lambda_im,
            "lambda_bone": self.lambda_bone,
            "lambda_reg": self.lambda_reg,
            "lambda_base_pos": self.lambda_base_pos,
            "lambda_base_rot": self.lambda_base_rot,
            "tau_m": self.tau,
            "b_m": self.b,
            "w_s": self.w_s,
            "config_path": str(self.config_path),
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True)
class CollisionQuerySet:
    sample_ids: np.ndarray
    inclusion_reasons: tuple[str, ...]
    active_round: np.ndarray
    initial_signed_distance: np.ndarray
    query_hash: str

    def validate(self, sample_count: int) -> CollisionQuerySet:
        ids = np.asarray(self.sample_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != len(self.inclusion_reasons):
            raise ValueError("query set IDs and inclusion reasons mismatch")
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= sample_count):
            raise ValueError("query set IDs are not unique or in range")
        if np.any(np.diff(ids) < 0):
            raise ValueError("query IDs must be deterministic ascending")
        return self

    @property
    def count(self) -> int:
        return int(len(self.sample_ids))


def load_robot_surface_samples(path: str | Path) -> RobotSurfaceSampleSet:
    """Load the immutable Stage 6 collision-surface artifact."""

    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        profile_values = dict(metadata["profile"])
        profile_values["assumptions"] = tuple(profile_values.get("assumptions", []))
        profile = RobotSurfaceSamplingProfile(**profile_values)
        return RobotSurfaceSampleSet(
            robot_name=str(metadata["robot_name"]),
            side=str(metadata["side"]),
            profile=profile,
            geometry_ids=np.asarray(data["geometry_ids"]),
            link_names=np.asarray(data["link_names"]),
            geometry_types=np.asarray(data["geometry_types"]),
            sample_ids=np.asarray(data["sample_ids"], dtype=np.int64),
            points_local=np.asarray(data["points_local"], dtype=np.float64),
            normals_local=np.asarray(data["normals_local"], dtype=np.float64),
            points_base=np.asarray(data["points_base"], dtype=np.float64),
            normals_base=np.asarray(data["normals_base"], dtype=np.float64),
            points_scene=np.asarray(data["points_scene"], dtype=np.float64),
            normals_scene=np.asarray(data["normals_scene"], dtype=np.float64),
            geometry_metadata=list(metadata.get("geometry_metadata", [])),
            source_provenance=dict(metadata.get("source_provenance", {})),
        )


def _query_hash(ids: np.ndarray, reasons: Iterable[str]) -> str:
    return _stable_hash(
        {"sample_ids": np.asarray(ids, dtype=np.int64).tolist(), "reasons": list(reasons)}
    )


def build_query_set(
    signed_distance: np.ndarray,
    geometry_ids: np.ndarray,
    profile: CollisionQueryProfile,
) -> CollisionQuerySet:
    phi = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
    geometry = np.asarray(geometry_ids).astype(str)
    if profile.mode == "full":
        ids = np.arange(len(phi), dtype=np.int64)
        reasons = ["full_surface_reference"] * len(ids)
    else:
        selected: dict[int, set[str]] = {}
        for idx, value in enumerate(phi):
            if value <= profile.active_margin_m:
                selected.setdefault(idx, set()).add(
                    "initial_penetration" if value < 0 else "initial_active_margin"
                )
        for geometry_id in sorted(set(geometry.tolist())):
            members = np.flatnonzero(geometry == geometry_id)
            if len(members):
                selected.setdefault(int(members[np.argmin(phi[members])]), set()).add(
                    "nearest_per_geometry"
                )
        ids = np.asarray(sorted(selected), dtype=np.int64)
        reasons = ["+".join(sorted(selected[int(idx)])) for idx in ids]
    return CollisionQuerySet(
        sample_ids=ids,
        inclusion_reasons=tuple(reasons),
        active_round=np.zeros(len(ids), dtype=np.int64),
        initial_signed_distance=phi[ids],
        query_hash=_query_hash(ids, reasons),
    ).validate(len(phi))


def expand_query_set(
    query_set: CollisionQuerySet,
    signed_distance: np.ndarray,
    profile: CollisionQueryProfile,
    *,
    active_round: int,
    force_full: bool = False,
) -> tuple[CollisionQuerySet, np.ndarray]:
    phi = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
    existing = set(int(item) for item in query_set.sample_ids.tolist())
    if force_full:
        candidates = np.arange(len(phi), dtype=np.int64)
    else:
        candidates = np.flatnonzero(phi < profile.active_margin_m)
    new_ids = [int(item) for item in candidates if int(item) not in existing]
    if not new_ids:
        return query_set, np.empty(0, dtype=np.int64)
    ids = np.concatenate([query_set.sample_ids, np.asarray(new_ids, dtype=np.int64)])
    order = np.argsort(ids, kind="stable")
    reason_by_id = dict(
        zip(query_set.sample_ids.tolist(), query_set.inclusion_reasons, strict=True)
    )
    for item in new_ids:
        reason_by_id[item] = (
            "post_solve_hard_or_soft_violation"
            if phi[item] < -0.001
            else "post_solve_active_margin"
        )
    reasons = tuple(reason_by_id[int(item)] for item in ids[order])
    rounds = {
        int(item): int(value)
        for item, value in zip(query_set.sample_ids, query_set.active_round, strict=True)
    }
    rounds.update({item: active_round for item in new_ids})
    new_set = CollisionQuerySet(
        sample_ids=ids[order],
        inclusion_reasons=reasons,
        active_round=np.asarray([rounds[int(item)] for item in ids[order]], dtype=np.int64),
        initial_signed_distance=phi[ids[order]],
        query_hash=_query_hash(ids[order], reasons),
    )
    return new_set.validate(len(phi)), np.asarray(new_ids, dtype=np.int64)


def continue_active_set_initial(
    result_x: np.ndarray,
    previous_query_set: CollisionQuerySet,
    expanded_query_set: CollisionQuerySet,
    *,
    new_query_ids: np.ndarray,
    signed_distance: np.ndarray,
    tau: float,
    b: float,
) -> np.ndarray:
    """Build the next SLSQP initial point from the preceding result.

    The non-slack coordinates are copied byte-for-byte from ``result_x``.  Slack
    values are looked up by stable query ID, so a deterministic reorder of the
    expanded set cannot change an existing slack value.  New slacks use the
    smallest bounded value that satisfies the soft constraint at the returned
    candidate.  This helper is intentionally independent of the solver so the
    continuation contract can be tested without importing a robot or mesh.
    """

    previous_ids = np.asarray(previous_query_set.sample_ids, dtype=np.int64)
    expanded_ids = np.asarray(expanded_query_set.sample_ids, dtype=np.int64)
    new_ids = np.asarray(new_query_ids, dtype=np.int64).reshape(-1)
    if len(np.unique(expanded_ids)) != len(expanded_ids):
        raise ValueError("expanded active set contains duplicate query IDs")
    if not np.all(np.isin(previous_ids, expanded_ids)):
        raise ValueError("active-set continuation cannot remove an existing query ID")
    expected_new = np.setdiff1d(expanded_ids, previous_ids, assume_unique=True)
    if not np.array_equal(np.sort(expected_new), np.sort(new_ids)):
        raise ValueError("active-set continuation new query IDs do not match expansion")

    value = np.asarray(result_x, dtype=np.float64).reshape(-1)
    previous_count = len(previous_ids)
    if previous_count:
        if len(value) < previous_count:
            raise ValueError("solver result is shorter than the previous slack vector")
        base_and_qpos = value[:-previous_count].copy()
        previous_slack = value[-previous_count:]
    else:
        base_and_qpos = value.copy()
        previous_slack = np.empty(0, dtype=np.float64)
    previous_slack_by_id = dict(zip(previous_ids.tolist(), previous_slack.tolist(), strict=True))
    phi = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
    if len(phi) == 0 or np.any(new_ids < 0) or np.any(new_ids >= len(phi)):
        raise ValueError("new active-set query ID is outside the signed-distance vector")
    upper = float(b - tau)
    if upper < 0:
        raise ValueError("hard slack bound must exceed soft tolerance")
    new_slack_by_id = {
        int(query_id): float(np.clip(max(-float(tau) - float(phi[query_id]), 0.0), 0.0, upper))
        for query_id in new_ids.tolist()
    }
    slack = np.asarray(
        [
            previous_slack_by_id[int(query_id)]
            if int(query_id) in previous_slack_by_id
            else new_slack_by_id[int(query_id)]
            for query_id in expanded_ids.tolist()
        ],
        dtype=np.float64,
    )
    return np.concatenate([base_and_qpos, slack])


def active_set_is_monotonic(
    previous_query_set: CollisionQuerySet, expanded_query_set: CollisionQuerySet
) -> bool:
    """Return whether an active-set expansion preserves all previous IDs."""

    previous = np.asarray(previous_query_set.sample_ids, dtype=np.int64)
    expanded = np.asarray(expanded_query_set.sample_ids, dtype=np.int64)
    return bool(
        len(np.unique(expanded)) == len(expanded)
        and np.all(np.isin(previous, expanded))
        and len(expanded) >= len(previous)
    )


def _skew(vector: Any) -> Any:
    import torch

    result = torch.zeros((*vector.shape[:-1], 3, 3), dtype=vector.dtype, device=vector.device)
    result[..., 0, 1] = -vector[..., 2]
    result[..., 0, 2] = vector[..., 1]
    result[..., 1, 0] = vector[..., 2]
    result[..., 1, 2] = -vector[..., 0]
    result[..., 2, 0] = -vector[..., 1]
    result[..., 2, 1] = vector[..., 0]
    return result


def so3_exp(vector: Any) -> Any:
    """Differentiable Rodrigues Exp for raw scene-frame rotation vectors."""

    import torch

    value = vector
    theta = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    k = _skew(value)
    theta2 = theta.square()
    safe_theta = torch.clamp(theta, min=1e-8)
    safe_theta2 = torch.clamp(theta2, min=1e-16)
    a = torch.where(theta > 1e-8, torch.sin(theta) / safe_theta, 1.0 - theta2 / 6.0)
    b = torch.where(
        theta > 1e-8,
        (1.0 - torch.cos(theta)) / safe_theta2,
        0.5 - theta2 / 24.0,
    )
    eye = torch.eye(3, dtype=value.dtype, device=value.device).expand(*value.shape[:-1], 3, 3)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def so3_log(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    trace = np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(trace))
    if theta < 1e-10:
        return (
            np.asarray(
                [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]]
            )
            * 0.5
        )
    return (
        theta
        / (2.0 * np.sin(theta))
        * np.asarray(
            [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]]
        )
    )


def map_previous_state_to_seed(
    previous_base: np.ndarray, previous_qpos: np.ndarray, current_seed_base: np.ndarray
) -> np.ndarray:
    prev = np.asarray(previous_base, dtype=np.float64)
    seed = np.asarray(current_seed_base, dtype=np.float64)
    delta_p = prev[:3, 3] - seed[:3, 3]
    delta_w = so3_log(prev[:3, :3] @ seed[:3, :3].T)
    return np.concatenate([delta_p, delta_w, np.asarray(previous_qpos, dtype=np.float64)])


class ConvexHullSignedDistanceBackend:
    """Exact signed distance for a convex, closed triangle mesh.

    It is solver-only.  Construction is rejected unless every source vertex is
    on the convex hull and the reference probe comparison is explicitly passed.
    """

    backend_id = "convex_hull_exact_solver_only"

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        mesh_hash: str,
        *,
        tree_leaf_size: int = 32,
    ):
        from scipy.spatial import ConvexHull

        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.mesh_hash = mesh_hash
        self.hull = ConvexHull(self.vertices)
        distances = self.hull.equations[:, :3] @ self.vertices.T + self.hull.equations[:, 3, None]
        if float(np.max(distances)) > 1e-9:
            raise ValueError("mesh is not convex")
        self.triangles = self.vertices[self.hull.simplices]
        # Keep the exact hull closest-point acceleration alive for the whole
        # refinement run.  The leaf closest-point computation is unchanged;
        # only triangles that cannot beat the current best distance are pruned.
        from toporetarget.geometry.signed_distance.closest_point import TriangleAABBTree

        if tree_leaf_size <= 0:
            raise ValueError("SDF tree leaf size must be positive")
        self._closest_tree = TriangleAABBTree(self.triangles, leaf_size=tree_leaf_size)
        self.tree_leaf_size = int(tree_leaf_size)

    def audit(self) -> dict[str, Any]:
        return {"strict": True, "convex": True, "hull_face_count": int(len(self.triangles))}

    def describe(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "sign_convention": "positive_outside",
            "mesh_hash": self.mesh_hash,
            "hull_face_count": int(len(self.triangles)),
            "solver_only": True,
            "triangle_aabb_leaf_size": self.tree_leaf_size,
        }

    def query_local(self, points_local: np.ndarray) -> SignedDistanceQueryResult:
        from toporetarget.geometry.signed_distance.closest_point import closest_points_on_triangles

        points = np.asarray(points_local, dtype=np.float64)
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        closest, face, bary, unsigned = closest_points_on_triangles(
            flat,
            self.triangles,
            query_chunk_size=4096,
            face_chunk_size=len(self.triangles),
            tree=self._closest_tree,
        )
        equations = self.hull.equations
        inside = np.all(flat @ equations[:, :3].T + equations[:, 3] <= 1e-10, axis=1)
        signed = np.where(inside, -unsigned, unsigned)
        direction = np.where(inside[:, None], closest - flat, flat - closest)
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        normal = direction / np.maximum(norm, 1e-15)
        non_smooth = unsigned <= 1e-10
        return SignedDistanceQueryResult(
            signed_distance=signed.reshape(shape),
            unsigned_distance=unsigned.reshape(shape),
            closest_points=closest.reshape((*shape, 3)),
            closest_face_indices=face.reshape(shape),
            closest_barycentric=bary.reshape((*shape, 3)),
            surface_normals=normal.reshape((*shape, 3)),
            inside=inside.reshape(shape),
            on_surface=(unsigned <= 1e-10).reshape(shape),
            valid=np.ones(shape, dtype=bool),
            sign_valid=np.ones(shape, dtype=bool),
            sign_confidence=np.ones(shape, dtype=np.float64),
            sign_method="convex_hull_halfspace",
            backend_id=self.backend_id,
            mesh_hash=self.mesh_hash,
            non_smooth=non_smooth.reshape(shape),
            gradient_valid=(~non_smooth).reshape(shape),
        )

    def query_scene(
        self, points_scene: np.ndarray, object_pose_scene: np.ndarray
    ) -> SignedDistanceQueryResult:
        from toporetarget.geometry.se3 import invert_transform, transform_points, transform_vectors

        local = transform_points(invert_transform(object_pose_scene), np.asarray(points_scene))
        result = self.query_local(local)
        result.closest_points = transform_points(object_pose_scene, result.closest_points)
        result.surface_normals = transform_vectors(object_pose_scene, result.surface_normals)
        result.surface_normals /= np.maximum(
            np.linalg.norm(result.surface_normals, axis=-1, keepdims=True), 1e-15
        )
        return result


def choose_solver_sdf_backend(
    vertices: np.ndarray,
    faces: np.ndarray,
    reference: ReferenceSignedDistanceBackend,
    profile: RefinementSolverProfile,
    *,
    object_pose_scene: np.ndarray,
    tree_leaf_size: int = 32,
) -> tuple[Any, dict[str, Any]]:
    report: dict[str, Any] = {
        "reference": reference.describe(),
        "requested": profile.sdf_backend,
        "selected": "reference",
        "cross_validation": None,
    }
    if profile.sdf_backend != "convex_hull_exact_solver_only":
        return reference, report
    try:
        candidate = ConvexHullSignedDistanceBackend(
            vertices,
            faces,
            reference.mesh_hash,
            tree_leaf_size=tree_leaf_size,
        )
        rng = np.random.default_rng(20260720)
        local = rng.normal(size=(profile.sdf_probe_count, 3))
        local *= max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0)
        local += np.mean(vertices, axis=0)
        ref = reference.query_local(local)
        fast = candidate.query_local(local)
        error = float(np.max(np.abs(ref.signed_distance - fast.signed_distance)))
        report["cross_validation"] = {
            "probe_count": len(local),
            "max_signed_distance_error_m": error,
            "tolerance_m": profile.sdf_cross_validation_tolerance_m,
            "passed": error <= profile.sdf_cross_validation_tolerance_m,
        }
        if error <= profile.sdf_cross_validation_tolerance_m:
            report["selected"] = candidate.backend_id
            return candidate, report
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        report["cross_validation"] = {"passed": False, "error": str(exc)}
    return reference, report


@dataclass
class FinalObjectiveBreakdown:
    e_im: float
    e_bone: float
    e_temporal: float
    e_base_pos: float
    e_base_rot: float
    e_slack: float
    weighted_e_im: float
    weighted_e_bone: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class _FrameContext:
    robot_model: Any
    graph_frame: InteractionGraphFrame
    source_features: BoneFeatures
    frame_profile: BoneDirectionFrameProfile
    bone_profile: BoneDirectionProfile
    seed_base: np.ndarray
    seed_qpos: np.ndarray
    previous_reference: np.ndarray | None
    paper: PaperRefinementWeights
    sdf: Any
    reference_sdf: Any
    object_pose_scene: np.ndarray
    surface: RobotSurfaceSampleSet
    surface_points_local: np.ndarray
    surface_geometry_indices: np.ndarray
    surface_local_transforms: tuple[np.ndarray, ...]
    surface_link_names: tuple[str, ...]
    geometry_slices: tuple[tuple[int, int, int], ...]
    frame_id: int | str = -1
    context_hash: str = ""
    cache: RefinementEvaluationCache = field(
        default_factory=lambda: RefinementEvaluationCache(-1, "")
    )
    timers: TimerBook = field(default_factory=TimerBook)
    full_audit_call_count: int = 0
    full_audit_call_reasons: list[str] = field(default_factory=list)
    active_query_call_count: int = 0
    _residual_model: Any = field(default=None, init=False, repr=False)
    _surface_joint_paths: tuple[tuple[Any, ...], ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _active_query_hash: str = field(default="__unbound__", init=False, repr=False)

    def __post_init__(self) -> None:
        self._residual_model = InteractionMeshResidual(
            self.graph_frame.source_vertices,
            self.graph_frame.directed_source_index,
            self.graph_frame.directed_destination_index,
            self.graph_frame.weights,
        )
        by_child = self.robot_model.urdf.parent_joint_by_child
        paths: list[tuple[Any, ...]] = []
        for link_name in self.surface_link_names:
            current = link_name
            path: list[Any] = []
            while current != self.robot_model.urdf.root_link:
                joint = by_child[current]
                if joint.actuated:
                    path.append(joint)
                current = joint.parent
            paths.append(tuple(reversed(path)))
        self._surface_joint_paths = tuple(paths)

    @property
    def variable_size_without_slack(self) -> int:
        return 6 + int(self.robot_model.num_dofs)

    def unpack(self, value: Any) -> tuple[Any, Any, Any, Any]:
        with self.timers.measure("candidate_state_decode"):
            delta_p = value[..., :3]
            delta_w = value[..., 3:6]
            qpos = value[..., 6 : 6 + self.robot_model.num_dofs]
            slack = value[..., 6 + self.robot_model.num_dofs :]
        return delta_p, delta_w, qpos, slack

    def base_pose_torch(self, value: Any) -> Any:
        import torch

        delta_p, delta_w, _, _ = self.unpack(value)
        seed = torch.as_tensor(self.seed_base, dtype=value.dtype, device=value.device)
        with self.timers.measure("so3_exp"):
            rotation = so3_exp(delta_w) @ seed[:3, :3]
        base = torch.eye(4, dtype=value.dtype, device=value.device)
        base = base.clone()
        base[:3, :3] = rotation
        base[:3, 3] = seed[:3, 3] + delta_p
        return base

    def collision_points_torch(self, value: Any, fk: dict[str, Any] | None = None) -> Any:
        import torch

        _, _, qpos, _ = self.unpack(value)
        if fk is None:
            with self.timers.measure("robot_fk"):
                fk = self.robot_model.forward_kinematics_base(qpos)
        pieces: list[Any] = []
        start = 0
        for geometry_index, start, stop in self.geometry_slices:
            points = torch.as_tensor(
                self.surface_points_local[start:stop], dtype=value.dtype, device=value.device
            )
            link_transform = fk[self.surface_link_names[geometry_index]]
            local_transform = torch.as_tensor(
                self.surface_local_transforms[geometry_index],
                dtype=value.dtype,
                device=value.device,
            )
            points = points @ local_transform[:3, :3].transpose(-1, -2) + local_transform[:3, 3]
            points = points @ link_transform[:3, :3].transpose(-1, -2) + link_transform[:3, 3]
            pieces.append(points)
        base = self.base_pose_torch(value)
        points = torch.cat(pieces, dim=0)
        return points @ base[:3, :3].transpose(-1, -2) + base[:3, 3]

    def robot_graph_vertices_torch(self, value: Any, robot_keypoints: Any | None = None) -> Any:
        import torch

        _, _, qpos, _ = self.unpack(value)
        base = self.base_pose_torch(value)
        hand = (
            self.robot_model.keypoints_scene(qpos, base, layout="mediapipe21")
            if robot_keypoints is None
            else robot_keypoints
        )
        object_points = torch.as_tensor(
            self.graph_frame.source_vertices[21:], dtype=value.dtype, device=value.device
        )
        return torch.cat([hand, object_points], dim=-2)

    def breakdown_tensor(self, value: Any) -> tuple[Any, FinalObjectiveBreakdown]:
        import torch

        delta_p, delta_w, qpos, slack = self.unpack(value)
        base = self.base_pose_torch(value)
        with self.timers.measure("robot_keypoints"):
            robot_keypoints = self.robot_model.keypoints_scene(qpos, base, layout="mediapipe21")
        robot_vertices = self.robot_graph_vertices_torch(value, robot_keypoints)
        with self.timers.measure("interaction_laplacian"):
            residual = self._residual_model(robot_vertices)
        with self.timers.measure("e_im"):
            e_im = residual.square().sum() / 71.0
        with self.timers.measure("bone_features"):
            robot_features = extract_bone_features(
                robot_keypoints,
                self.frame_profile,
                self.bone_profile,
                side=self.robot_model.side,
                strict=True,
            )
        source_adj = torch.as_tensor(
            _as_np(self.source_features.adjacent_features), dtype=value.dtype, device=value.device
        )
        with self.timers.measure("e_bone"):
            e_bone = (robot_features.adjacent_features - source_adj).square().sum()
        with self.timers.measure("temporal_base_slack_objective"):
            e_temporal = value.new_zeros(())
            if self.previous_reference is not None:
                previous = torch.as_tensor(
                    self.previous_reference, dtype=value.dtype, device=value.device
                )
                e_temporal = (
                    self.paper.lambda_reg
                    * (value[: self.variable_size_without_slack] - previous).square().sum()
                )
            e_base_pos = self.paper.lambda_base_pos * delta_p.square().sum()
            e_base_rot = self.paper.lambda_base_rot * delta_w.square().sum()
            e_slack = 0.5 * self.paper.w_s * slack.square().sum()
        weighted_im = self.paper.lambda_im * e_im
        weighted_bone = self.paper.lambda_bone * e_bone
        total = weighted_im + weighted_bone + e_temporal + e_base_pos + e_base_rot + e_slack
        return total, FinalObjectiveBreakdown(
            e_im=float(e_im.detach().cpu()),
            e_bone=float(e_bone.detach().cpu()),
            e_temporal=float(e_temporal.detach().cpu()),
            e_base_pos=float(e_base_pos.detach().cpu()),
            e_base_rot=float(e_base_rot.detach().cpu()),
            e_slack=float(e_slack.detach().cpu()),
            weighted_e_im=float(weighted_im.detach().cpu()),
            weighted_e_bone=float(weighted_bone.detach().cpu()),
            total=float(total.detach().cpu()),
        )

    def objective(
        self, value: np.ndarray, query_hash: str | None = None
    ) -> tuple[float, np.ndarray, FinalObjectiveBreakdown]:
        import torch

        current = np.asarray(value, dtype=np.float64).reshape(-1)
        active_hash = str(query_hash or getattr(self, "_active_query_hash", "__unbound__"))
        self._active_query_hash = active_hash
        self.cache.prepare(current, active_hash)
        cached = self.cache.get("objective")
        if cached is not None:
            total, gradient, breakdown = cached
            return float(total), np.asarray(gradient, dtype=np.float64).copy(), breakdown
        with self.timers.measure("objective_autograd"):
            with self.timers.measure("numpy_to_torch"):
                variable = torch.as_tensor(current, dtype=torch.float64).requires_grad_(True)
            total, breakdown = self.breakdown_tensor(variable)
            gradient = torch.autograd.grad(total, variable, create_graph=False)[0]
        if self.cache.get("candidate_points") is None:
            with self.timers.measure("collision_point_transform"):
                points = _as_np(self.collision_points_torch(variable))
            self.cache.put("candidate_points", np.asarray(points, dtype=np.float64).copy())
        with self.timers.measure("torch_to_numpy"):
            stored = (float(total.detach().cpu()), _as_np(gradient).copy(), breakdown)
        self.cache.put("objective", stored)
        return float(stored[0]), np.asarray(stored[1]).copy(), stored[2]

    def candidate_points(self, value: np.ndarray, query_hash: str | None = None) -> np.ndarray:
        import torch

        current = np.asarray(value, dtype=np.float64).reshape(-1)
        active_hash = str(query_hash or getattr(self, "_active_query_hash", "__unbound__"))
        self._active_query_hash = active_hash
        self.cache.prepare(current, active_hash)
        cached = self.cache.get("candidate_points")
        if cached is not None:
            return np.asarray(cached, dtype=np.float64).copy()
        with self.timers.measure("collision_point_transform"):
            variable = torch.as_tensor(current, dtype=torch.float64)
            points = _as_np(self.collision_points_torch(variable))
        self.cache.put("candidate_points", np.asarray(points, dtype=np.float64).copy())
        return np.asarray(points, dtype=np.float64).copy()

    def collision_points_jacobian_numpy(self, value: np.ndarray) -> np.ndarray:
        """Return the exact batched point Jacobian without per-point autograd.

        The URDF chain gives each collision point's qpos derivative directly
        from the joint origin/axis spatial Jacobian.  Only the six base
        coordinates use Torch autograd, over a fixed 6-vector and constant
        link points; this keeps the float64 Exp-map derivative identical to
        the objective path while removing the O(points * qpos) functional
        Jacobian cost from the hot callback.
        """

        import torch

        current = np.asarray(value, dtype=np.float64).reshape(-1)
        _, _, qpos, _ = self.unpack(current)
        qpos = np.asarray(qpos, dtype=np.float64)
        fk = self.robot_model.forward_kinematics_reference(qpos)
        points_base = np.empty((self.surface.count, 3), dtype=np.float64)
        qpos_jacobian_base = np.zeros(
            (self.surface.count, 3, self.robot_model.num_dofs), dtype=np.float64
        )
        for geometry_index, start, stop in self.geometry_slices:
            local = np.asarray(self.surface_points_local[start:stop], dtype=np.float64)
            local_transform = np.asarray(
                self.surface_local_transforms[geometry_index], dtype=np.float64
            )
            local_points = local @ local_transform[:3, :3].T + local_transform[:3, 3]
            link_name = self.surface_link_names[geometry_index]
            link = np.asarray(fk[link_name], dtype=np.float64)
            points = local_points @ link[:3, :3].T + link[:3, 3]
            points_base[start:stop] = points
            for joint in self._surface_joint_paths[geometry_index]:
                parent = np.asarray(fk[joint.parent], dtype=np.float64)
                origin = parent @ np.asarray(joint.origin, dtype=np.float64)
                axis = (
                    parent[:3, :3]
                    @ np.asarray(joint.origin[:3, :3], dtype=np.float64)
                    @ np.asarray(joint.axis, dtype=np.float64)
                )
                if joint.joint_type in {"revolute", "continuous"}:
                    derivative = np.cross(axis[None, :], points - origin[:3, 3][None, :])
                elif joint.joint_type == "prismatic":
                    derivative = np.broadcast_to(axis, points.shape)
                else:  # pragma: no cover - fixed joints are excluded above
                    continue
                q_index = int(self.robot_model._dof_index[joint.name])
                qpos_jacobian_base[start:stop, :, q_index] += derivative

        base_delta = torch.as_tensor(current[:6], dtype=torch.float64)
        seed_rotation = torch.as_tensor(self.seed_base[:3, :3], dtype=torch.float64)
        base_points = torch.as_tensor(points_base, dtype=torch.float64)

        def base_points_fn(delta: Any) -> Any:
            rotation = so3_exp(delta[3:]) @ seed_rotation
            translation = torch.as_tensor(self.seed_base[:3, 3], dtype=delta.dtype) + delta[:3]
            return base_points @ rotation.transpose(-1, -2) + translation

        with self.timers.measure("collision_point_jacobian"):
            base_jacobian = torch.autograd.functional.jacobian(
                base_points_fn,
                base_delta,
                create_graph=False,
                vectorize=True,
                strategy="reverse-mode",
            )
        base_jacobian_np = _as_np(base_jacobian)
        rotation_np = _as_np(so3_exp(base_delta)) @ np.asarray(
            self.seed_base[:3, :3], dtype=np.float64
        )
        qpos_jacobian_scene = np.einsum(
            "ab,mbd->mad", rotation_np, qpos_jacobian_base, optimize=True
        )
        result = np.zeros(
            (self.surface.count, 3, self.variable_size_without_slack), dtype=np.float64
        )
        result[:, :, :6] = base_jacobian_np
        result[:, :, 6:] = qpos_jacobian_scene
        return result

    def collision_points_jacobian_reference_numpy(self, value: np.ndarray) -> np.ndarray:
        """Reference batched Torch Jacobian used only for strict recovery."""

        import torch

        current = np.asarray(value, dtype=np.float64).reshape(-1)
        variable = torch.as_tensor(current, dtype=torch.float64)
        with self.timers.measure("collision_point_jacobian_reference"):
            jacobian = torch.autograd.functional.jacobian(
                lambda item: self.collision_points_torch(item),
                variable,
                create_graph=False,
                vectorize=True,
                strategy="reverse-mode",
            )
        return _as_np(jacobian)[:, :, : self.variable_size_without_slack]

    def constraint_query(
        self, value: np.ndarray, query_ids: np.ndarray, query_hash: str | None = None
    ) -> SignedDistanceQueryResult:
        current = np.asarray(value, dtype=np.float64).reshape(-1)
        active_hash = str(query_hash or getattr(self, "_active_query_hash", "__unbound__"))
        self._active_query_hash = active_hash
        self.cache.prepare(current, active_hash)
        cached = self.cache.get("constraint_query")
        if cached is not None:
            return cached
        with self.timers.measure("solver_sdf"):
            result = self.sdf.query_scene(
                self.candidate_points(current)[query_ids], self.object_pose_scene
            )
        self.active_query_call_count += 1
        if not np.all(result.sign_valid) or not np.all(result.valid):
            raise ValueError("invalid signed-distance result entered constrained solve")
        self.cache.put("constraint_query", result)
        return result

    def constraint_values(
        self, value: np.ndarray, query_ids: np.ndarray, query_hash: str | None = None
    ) -> np.ndarray:
        current = np.asarray(value, dtype=np.float64).reshape(-1)
        active_hash = str(query_hash or getattr(self, "_active_query_hash", "__unbound__"))
        self._active_query_hash = active_hash
        self.cache.prepare(current, active_hash)
        cached = self.cache.get("constraint_values")
        if cached is not None:
            return np.asarray(cached, dtype=np.float64).copy()
        result = self.constraint_query(current, query_ids, active_hash)
        _, _, _, slack = self.unpack(np.asarray(value, dtype=np.float64))
        output = np.concatenate(
            [result.signed_distance + self.paper.b, result.signed_distance + slack + self.paper.tau]
        )
        self.cache.put("constraint_values", output.copy())
        return output

    def constraint_jacobian(
        self,
        value: np.ndarray,
        query_ids: np.ndarray,
        eps: float,
        query_hash: str | None = None,
        backend: str = "analytic_urdf_spatial_v2",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        current = np.asarray(value, dtype=np.float64)
        active_hash = str(query_hash or getattr(self, "_active_query_hash", "__unbound__"))
        self._active_query_hash = active_hash
        self.cache.prepare(current, active_hash)
        cached = self.cache.get("constraint_jacobian")
        if cached is not None:
            jacobian, diagnostics = cached
            return np.asarray(jacobian, dtype=np.float64).copy(), dict(diagnostics)
        n = self.variable_size_without_slack
        if backend == "analytic_urdf_spatial_v2":
            # Use the URDF spatial Jacobian for qpos and a fixed six-coordinate
            # autograd derivative for the base Exp-map.
            jac_np = self.collision_points_jacobian_numpy(current)[query_ids]
        elif backend == "reference_batched_torch_v1":
            jac_np = self.collision_points_jacobian_reference_numpy(current)[query_ids]
        else:
            raise ValueError(f"unsupported collision point Jacobian backend: {backend}")
        result = self.constraint_query(current, query_ids, active_hash)
        normals = np.asarray(result.surface_normals, dtype=np.float64).reshape(-1, 3)
        valid = np.asarray(
            result.gradient_valid if result.gradient_valid is not None else result.sign_valid,
            dtype=bool,
        ).reshape(-1)
        values = np.zeros((len(query_ids), len(current)), dtype=np.float64)
        fallback_count = 0
        for row, sample_valid in enumerate(valid):
            if sample_valid:
                values[row, :n] = normals[row] @ jac_np[row, :, :n]
            else:
                fallback_count += 1
                for col in range(n):
                    plus = current.copy()
                    minus = current.copy()
                    plus[col] += eps
                    minus[col] -= eps
                    values[row, col] = (
                        self.constraint_query(plus, query_ids).signed_distance[row]
                        - self.constraint_query(minus, query_ids).signed_distance[row]
                    ) / (2.0 * eps)
        values_soft = values.copy()
        values_soft[:, n:] = 0.0
        for row in range(len(query_ids)):
            values_soft[row, n + row] = 1.0
        output = np.vstack([values, values_soft])
        diagnostics = {
            "gradient_valid_count": int(np.count_nonzero(valid)),
            "finite_difference_fallback_count": fallback_count,
            "normal_frame": "scene",
            "point_jacobian_backend": backend,
        }
        self.cache.put("constraint_jacobian", (output.copy(), dict(diagnostics)))
        return output, diagnostics


@dataclass
class FinalFrameResult:
    qpos: np.ndarray
    base_pose_scene: np.ndarray
    base_correction: np.ndarray
    slack: np.ndarray
    query_set: CollisionQuerySet
    breakdown: FinalObjectiveBreakdown
    warm_breakdown: FinalObjectiveBreakdown
    signed_distance: np.ndarray
    hard_residual: np.ndarray
    soft_residual: np.ndarray
    full_signed_distance: np.ndarray
    full_closest_points: np.ndarray
    full_surface_normals: np.ndarray
    solver_success: bool
    solver_status: int
    solver_message: str
    iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    optimizer_converged: bool
    optimizer_status_code: int
    optimizer_message: str
    optimizer_iterations: int
    optimizer_function_evaluations: int
    optimizer_jacobian_evaluations: int
    qpos_bounds_pass: bool
    slack_bounds_pass: bool
    active_constraints_feasible: bool
    full_surface_hard_audit_pass: bool
    full_surface_soft_audit_pass: bool
    active_set_converged: bool
    all_values_finite: bool
    stationarity_checked: bool
    stationarity_residual: float
    accepted: bool
    acceptance_policy_id: str
    acceptance_reason: str
    initial_objective: float
    final_objective: float
    final_objective_change: float
    final_step_norm: float
    solve_time_s: float
    active_set_rounds: int
    jacobian_diagnostics: dict[str, Any]
    failure: str | None = None


def _surface_layout(
    model: Any, samples: RobotSurfaceSampleSet
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[str, ...], tuple[tuple[int, int, int], ...]]:
    geometry_ids = np.asarray(samples.geometry_ids).astype(str)
    geometry_indices = np.asarray(
        [int(value.split(":", 2)[1]) for value in geometry_ids], dtype=np.int64
    )
    instances = model.collision_geometry_instances(model.neutral_q)
    if len(instances) == 0 or np.max(geometry_indices) >= len(instances):
        raise ValueError("collision surface artifact geometry IDs do not match robot")
    unique = np.unique(geometry_indices)
    if not np.array_equal(unique, np.arange(len(instances))):
        raise ValueError("collision surface artifact must contain all geometries in order")
    starts: list[tuple[int, int, int]] = []
    for geometry_index in unique.tolist():
        members = np.flatnonzero(geometry_indices == geometry_index)
        if not np.array_equal(members, np.arange(members[0], members[-1] + 1)):
            raise ValueError("collision samples are not grouped deterministically by geometry")
        starts.append((int(geometry_index), int(members[0]), int(members[-1] + 1)))
    return (
        geometry_indices,
        tuple(item.local_transform for item in instances),
        tuple(item.link_name for item in instances),
        tuple(starts),
    )


def dynamic_collision_points_numpy(
    model: Any, samples: RobotSurfaceSampleSet, qpos: np.ndarray, base_pose_scene: np.ndarray
) -> np.ndarray:
    """Evaluate Stage 6 collision samples at a final qpos/base pose."""

    _, transforms, links, slices = _surface_layout(model, samples)
    fk = model.forward_kinematics_reference(np.asarray(qpos, dtype=np.float64))
    pieces: list[np.ndarray] = []
    for geometry_index, start, stop in slices:
        local = np.asarray(samples.points_local[start:stop], dtype=np.float64)
        transform = np.asarray(transforms[geometry_index], dtype=np.float64)
        link = np.asarray(fk[links[geometry_index]], dtype=np.float64)
        local_points = local @ transform[:3, :3].T + transform[:3, 3]
        pieces.append(local_points @ link[:3, :3].T + link[:3, 3])
    base = np.asarray(base_pose_scene, dtype=np.float64)
    points = np.concatenate(pieces, axis=0)
    return points @ base[:3, :3].T + base[:3, 3]


def _make_source_features(
    keypoints: np.ndarray,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    side: str,
) -> BoneFeatures:
    return extract_bone_features(keypoints, frame_profile, bone_profile, side=side, strict=True)


def _solver_call(
    context: _FrameContext,
    initial: np.ndarray,
    query_set: CollisionQuerySet,
    solver: RefinementSolverProfile,
    *,
    point_jacobian_backend: str = "analytic_urdf_spatial_v2",
) -> tuple[Any, dict[str, Any]]:
    from scipy.optimize import minimize

    query_ids = query_set.sample_ids
    # SLSQP sees normalized variables, while every objective/constraint and
    # every persisted artifact remains in the paper's raw meters/radians/
    # slack coordinates.  This is an invertible diagonal reparameterization;
    # it changes numerical conditioning only and leaves Eq. (8)-(9), bounds,
    # tolerances, and the fixed maxiter contract unchanged.
    variable_scales = np.concatenate(
        [
            np.full(3, 0.1, dtype=np.float64),
            np.ones(3, dtype=np.float64),
            np.maximum(
                np.asarray(context.robot_model.joint_upper, dtype=np.float64)
                - np.asarray(context.robot_model.joint_lower, dtype=np.float64),
                1e-6,
            ),
            np.full(query_set.count, context.paper.b - context.paper.tau, dtype=np.float64),
        ]
    )

    def physical(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) * variable_scales

    def normalized_gradient(gradient: np.ndarray) -> np.ndarray:
        return np.asarray(gradient, dtype=np.float64) * variable_scales

    objective_calls = 0
    objective_jacobian_calls = 0
    constraint_calls = 0
    jacobian_calls = 0
    fallback_total = 0
    callback_iterates: list[np.ndarray] = []
    context._active_query_hash = query_set.query_hash

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal objective_calls
        objective_calls += 1
        with context.timers.measure("objective_callback"):
            total, grad, _ = context.objective(physical(value), query_set.query_hash)
        return total, normalized_gradient(grad)

    def constraint(value: np.ndarray) -> np.ndarray:
        nonlocal constraint_calls
        constraint_calls += 1
        with context.timers.measure("constraint_callback"):
            return context.constraint_values(physical(value), query_ids, query_set.query_hash)

    def objective_jacobian(value: np.ndarray) -> np.ndarray:
        nonlocal objective_jacobian_calls
        objective_jacobian_calls += 1
        with context.timers.measure("objective_jacobian_callback"):
            return normalized_gradient(
                context.objective(physical(value), query_set.query_hash)[1]
            )

    def constraint_jac(value: np.ndarray) -> np.ndarray:
        nonlocal jacobian_calls, fallback_total
        jacobian_calls += 1
        with context.timers.measure("constraint_jacobian_callback"):
            jac, diagnostics = context.constraint_jacobian(
                physical(value),
                query_ids,
                solver.finite_difference_epsilon,
                query_set.query_hash,
                backend=point_jacobian_backend,
            )
        fallback_total += int(diagnostics["finite_difference_fallback_count"])
        return np.asarray(jac, dtype=np.float64) * variable_scales[None, :]

    def callback(value: np.ndarray) -> None:
        callback_iterates.append(physical(value).copy())

    lower_physical = np.concatenate(
        [
            np.full(6, -np.inf),
            np.asarray(context.robot_model.joint_lower, dtype=np.float64),
            np.zeros(query_set.count, dtype=np.float64),
        ]
    )
    upper_physical = np.concatenate(
        [
            np.full(6, np.inf),
            np.asarray(context.robot_model.joint_upper, dtype=np.float64),
            np.full(query_set.count, context.paper.b - context.paper.tau, dtype=np.float64),
        ]
    )
    lower = lower_physical / variable_scales
    upper = upper_physical / variable_scales
    bounds = list(zip(lower, upper, strict=True))
    with context.timers.measure("slsqp_total"):
        result = minimize(
            lambda value: objective(value)[0],
            np.asarray(initial, dtype=np.float64) / variable_scales,
            jac=objective_jacobian,
            method=solver.method,
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraint, "jac": constraint_jac},
            options={"maxiter": solver.maxiter, "ftol": solver.ftol, "disp": solver.disp},
            callback=callback,
        )
    result.x = physical(np.asarray(result.x, dtype=np.float64))
    initial_objective = float(
        context.objective(np.asarray(initial, dtype=np.float64), query_set.query_hash)[0]
    )
    final_objective = float(
        context.objective(np.asarray(result.x, dtype=np.float64), query_set.query_hash)[0]
    )
    previous_iterate = callback_iterates[-1] if callback_iterates else np.asarray(initial)
    return result, {
        "objective_evaluations": objective_calls,
        "objective_jacobian_evaluations": objective_jacobian_calls,
        "constraint_evaluations": constraint_calls,
        "constraint_jacobian_evaluations": jacobian_calls,
        "point_jacobian_backend": point_jacobian_backend,
        "finite_difference_fallback_count": fallback_total,
        "optimizer_function_evaluations": int(getattr(result, "nfev", objective_calls)),
        "optimizer_jacobian_evaluations": int(getattr(result, "njev", jacobian_calls)),
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "final_objective_change": initial_objective - final_objective,
        "final_step_norm": float(
            np.linalg.norm(np.asarray(result.x, dtype=np.float64) - previous_iterate)
        ),
        "cache": context.cache.as_dict(),
        "timers": context.timers.as_dict(),
    }


def _independent_constraints(
    context: _FrameContext,
    value: np.ndarray,
    query_set: CollisionQuerySet,
    *,
    distance_result: SignedDistanceQueryResult | None = None,
) -> dict[str, Any]:
    result = distance_result or context.constraint_query(
        value, query_set.sample_ids, query_set.query_hash
    )
    _, _, _, slack = context.unpack(np.asarray(value, dtype=np.float64))
    hard = result.signed_distance + context.paper.b
    soft = result.signed_distance + slack + context.paper.tau
    return {
        "signed_distance": result.signed_distance,
        "hard_residual": hard,
        "soft_residual": soft,
        "min_hard_residual": float(np.min(hard)) if len(hard) else math.inf,
        "min_soft_residual": float(np.min(soft)) if len(soft) else math.inf,
        "maximum_violation": float(max(0.0, -np.min(np.concatenate([hard, soft]))))
        if len(hard)
        else 0.0,
        "violating_sample_ids": query_set.sample_ids[(hard < -1e-6) | (soft < -1e-6)].tolist(),
        "slack_required": np.maximum(-context.paper.tau - result.signed_distance, 0.0),
    }


def strict_acceptance_decision(
    *,
    optimizer_converged: bool,
    optimizer_status_code: int,
    qpos_bounds_pass: bool,
    slack_bounds_pass: bool,
    active_constraints_feasible: bool,
    full_surface_hard_audit_pass: bool,
    full_surface_soft_audit_pass: bool,
    active_set_converged: bool,
    all_values_finite: bool,
    acceptance_policy_id: str = STRICT_ACCEPTANCE_POLICY_ID,
) -> dict[str, Any]:
    """Apply strict Stage 9 termination; feasibility never overrides status 9."""

    optimizer_ok = bool(optimizer_converged) and int(optimizer_status_code) != 9
    checks = {
        "optimizer_converged": optimizer_ok,
        "qpos_bounds_pass": bool(qpos_bounds_pass),
        "slack_bounds_pass": bool(slack_bounds_pass),
        "active_constraints_feasible": bool(active_constraints_feasible),
        "full_surface_hard_audit_pass": bool(full_surface_hard_audit_pass),
        "full_surface_soft_audit_pass": bool(full_surface_soft_audit_pass),
        "active_set_converged": bool(active_set_converged),
        "all_values_finite": bool(all_values_finite),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if acceptance_policy_id != STRICT_ACCEPTANCE_POLICY_ID:
        failures.append(f"unregistered_acceptance_policy:{acceptance_policy_id}")
    accepted = not failures
    return {
        "accepted": accepted,
        "acceptance_policy_id": acceptance_policy_id,
        "acceptance_reason": (
            "strict contract passed"
            if accepted
            else "strict contract failed: " + ", ".join(failures)
        ),
        "checks": checks,
    }


def refine_frame(
    context: _FrameContext,
    query_set: CollisionQuerySet,
    solver: RefinementSolverProfile,
    *,
    max_rounds: int,
    active_margin_m: float = 0.010,
    point_jacobian_backend: str = "reference_batched_torch_v1",
    strict_recovery: str = "none",
) -> FinalFrameResult:
    started = time.perf_counter()
    warm_value_without = np.concatenate([np.zeros(6), context.seed_qpos])
    warm_slack = np.clip(
        np.maximum(-context.paper.tau - query_set.initial_signed_distance, 0.0),
        0.0,
        context.paper.b - context.paper.tau,
    )
    initial = np.concatenate([warm_value_without, warm_slack])
    query_rounds = 0
    result: Any = None
    full: SignedDistanceQueryResult | None = None
    diagnostics: dict[str, Any] = {}
    continuation_trace: list[dict[str, Any]] = []
    solver_attempt_trace: list[dict[str, Any]] = []
    active_set_converged = False
    full_audit_call_count = int(context.full_audit_call_count)
    full_audit_call_reasons = list(context.full_audit_call_reasons)
    while True:
        query_rounds += 1
        outer_round_started = time.perf_counter()
        if len(initial) != 6 + context.robot_model.num_dofs + query_set.count:
            raise ValueError(
                "query set changed variable dimension without rebuilding initial state"
            )
        result, solve_diag = _solver_call(
            context,
            initial,
            query_set,
            solver,
            point_jacobian_backend=point_jacobian_backend,
        )
        primary_result = result
        primary_diag = dict(solve_diag)
        attempt = {
            "round": query_rounds,
            "primary_status": int(getattr(primary_result, "status", -1)),
            "primary_success": bool(primary_result.success),
            "primary_message": str(primary_result.message),
            "primary_backend": point_jacobian_backend,
            "recovery_used": False,
        }
        if (
            int(getattr(primary_result, "status", -1)) == 9
            and strict_recovery == "reference_batched_from_primary_result_v1"
        ):
            result, recovery_diag = _solver_call(
                context,
                np.asarray(primary_result.x, dtype=np.float64),
                query_set,
                solver,
                point_jacobian_backend="reference_batched_torch_v1",
            )
            for key in (
                "objective_evaluations",
                "objective_jacobian_evaluations",
                "constraint_evaluations",
                "constraint_jacobian_evaluations",
                "optimizer_function_evaluations",
                "optimizer_jacobian_evaluations",
                "finite_difference_fallback_count",
            ):
                recovery_diag[key] = int(primary_diag.get(key, 0)) + int(
                    recovery_diag.get(key, 0)
                )
            recovery_diag["initial_objective"] = primary_diag["initial_objective"]
            recovery_diag["final_objective_change"] = float(
                primary_diag["initial_objective"] - recovery_diag["final_objective"]
            )
            recovery_diag["primary_solver_status"] = int(getattr(primary_result, "status", -1))
            recovery_diag["primary_solver_message"] = str(primary_result.message)
            recovery_diag["primary_optimizer_iterations"] = int(
                getattr(primary_result, "nit", 0)
            )
            recovery_diag["primary_point_jacobian_backend"] = point_jacobian_backend
            recovery_diag["solver_recovery"] = strict_recovery
            recovery_diag["solver_retry_count"] = 1
            attempt.update(
                {
                    "recovery_used": True,
                    "recovery_status": int(getattr(result, "status", -1)),
                    "recovery_success": bool(result.success),
                    "recovery_message": str(result.message),
                    "recovery_backend": "reference_batched_torch_v1",
                }
            )
            solve_diag = recovery_diag
        else:
            solve_diag["primary_solver_status"] = int(getattr(primary_result, "status", -1))
            solve_diag["primary_solver_message"] = str(primary_result.message)
            solve_diag["primary_optimizer_iterations"] = int(
                getattr(primary_result, "nit", 0)
            )
            solve_diag["primary_point_jacobian_backend"] = point_jacobian_backend
            solve_diag["solver_recovery"] = "none"
            solve_diag["solver_retry_count"] = 0
        solver_attempt_trace.append(attempt)
        diagnostics.update(solve_diag)
        independent = _independent_constraints(context, result.x, query_set)
        # Use the Stage 6 reference backend once per frame for both active-set
        # expansion and the persisted independent full-surface audit. The
        # solver-only backend remains reserved for inner constraint calls.
        with context.timers.measure("full_512_audit"):
            full = context.reference_sdf.query_scene(
                context.candidate_points(result.x, query_set.query_hash),
                context.object_pose_scene,
            )
        full_audit_call_count += 1
        full_audit_call_reasons.append("active_set_round_end")
        full_phi = np.asarray(full.signed_distance, dtype=np.float64)
        if not np.all(full.sign_valid):
            raise ValueError("full-surface audit received invalid signed distance")
        unqueried = np.setdiff1d(np.arange(len(full_phi)), query_set.sample_ids, assume_unique=True)
        unqueried_soft_ok = bool(
            len(unqueried) == 0 or np.all(full_phi[unqueried] >= -context.paper.tau - 1e-6)
        )
        hard_ok = bool(np.min(full_phi) >= -context.paper.b - 1e-6)
        active_constraints_ok = bool(
            independent["min_hard_residual"] >= -1e-6 and independent["min_soft_residual"] >= -1e-6
        )
        no_active_unqueried = not np.any(
            full_phi[unqueried] < (active_margin_m if len(unqueried) else -np.inf)
        )
        active_set_converged = bool(
            active_constraints_ok
            and hard_ok
            and unqueried_soft_ok
            and (query_set.count == len(full_phi) or no_active_unqueried)
        )
        context.timers.add("active_set_outer_loop", time.perf_counter() - outer_round_started)
        if result.success and active_set_converged:
            break
        if query_rounds >= max_rounds or query_set.count == len(full_phi):
            break
        with context.timers.measure("active_set_expansion"):
            expanded, new_ids = expand_query_set(
                query_set,
                full_phi,
                CollisionQueryProfile(
                    profile_id="runtime",
                    version="1",
                    mode="adaptive",
                    active_margin_m=active_margin_m,
                    max_active_set_rounds=max_rounds,
                    paper_status="not_paper_specified",
                    assumptions=(),
                    profile_hash="",
                ),
                active_round=query_rounds,
            )
        if len(new_ids) == 0:
            break
        previous_query_set = query_set
        query_set = expanded
        continuation_is_v2 = (
            solver.active_set_continuation_policy == "result_x_query_id_slack_remap_v2"
        )
        if continuation_is_v2:
            initial = continue_active_set_initial(
                result.x,
                previous_query_set,
                query_set,
                new_query_ids=new_ids,
                signed_distance=full_phi,
                tau=context.paper.tau,
                b=context.paper.b,
            )
        else:
            # Preserve the v1 warm-seed reinitialization behavior exactly. It is
            # retained for regression comparison; v2 is the only profile that
            # opts into result.x continuation.
            warm_slack = np.clip(
                np.maximum(-context.paper.tau - query_set.initial_signed_distance, 0.0),
                0.0,
                context.paper.b - context.paper.tau,
            )
            initial = np.concatenate([warm_value_without, warm_slack])
        continuation_trace.append(
            {
                "round": query_rounds,
                "query_ids_before": previous_query_set.sample_ids.tolist(),
                "query_ids_after": query_set.sample_ids.tolist(),
                "new_query_ids": new_ids.tolist(),
                "active_set_monotonic": active_set_is_monotonic(previous_query_set, query_set),
                "resumed_from_result_x": continuation_is_v2,
                "reinitialized_from_stage7_warm_seed": not continuation_is_v2,
                "result_x_sha256": _sha256_bytes(
                    np.asarray(result.x, dtype=np.float64).tobytes(order="C")
                ),
            }
        )
    if full is None:
        raise RuntimeError("Stage 9 full-surface audit did not run")
    value = np.asarray(result.x, dtype=np.float64)
    with context.timers.measure("full_512_audit"):
        full = context.reference_sdf.query_scene(
            context.candidate_points(value, query_set.query_hash),
            context.object_pose_scene,
        )
    full_audit_call_count += 1
    full_audit_call_reasons.append("frame_final_independent_acceptance")
    if not np.all(full.sign_valid):
        raise ValueError("final independent full-surface audit received invalid signed distance")
    _, _, qpos, slack = context.unpack(value)
    _, _, breakdown = context.objective(value, query_set.query_hash)
    final_warm_slack = np.clip(
        np.maximum(-context.paper.tau - query_set.initial_signed_distance, 0.0),
        0.0,
        context.paper.b - context.paper.tau,
    )
    _, _, warm_breakdown = context.objective(
        np.concatenate([warm_value_without, final_warm_slack]), query_set.query_hash
    )
    selected_full = SignedDistanceQueryResult(
        signed_distance=full.signed_distance[query_set.sample_ids],
        unsigned_distance=full.unsigned_distance[query_set.sample_ids],
        closest_points=full.closest_points[query_set.sample_ids],
        closest_face_indices=full.closest_face_indices[query_set.sample_ids],
        closest_barycentric=full.closest_barycentric[query_set.sample_ids],
        surface_normals=full.surface_normals[query_set.sample_ids],
        inside=None if full.inside is None else full.inside[query_set.sample_ids],
        on_surface=full.on_surface[query_set.sample_ids],
        valid=full.valid[query_set.sample_ids],
        sign_valid=full.sign_valid[query_set.sample_ids],
        sign_confidence=full.sign_confidence[query_set.sample_ids],
        sign_method=full.sign_method,
        backend_id=full.backend_id,
        mesh_hash=full.mesh_hash,
        winding_value=None
        if full.winding_value is None
        else full.winding_value[query_set.sample_ids],
        non_smooth=None if full.non_smooth is None else full.non_smooth[query_set.sample_ids],
        gradient_valid=None
        if full.gradient_valid is None
        else full.gradient_valid[query_set.sample_ids],
    )
    independent = _independent_constraints(context, value, query_set, distance_result=selected_full)
    diagnostics["outer_converged"] = bool(active_set_converged and result.success)
    diagnostics["active_set_converged"] = active_set_converged
    diagnostics["full_audit_call_count"] = full_audit_call_count
    diagnostics["full_audit_call_reasons"] = full_audit_call_reasons
    diagnostics["active_query_call_count"] = int(context.active_query_call_count)
    diagnostics["active_set_continuation"] = continuation_trace
    diagnostics["solver_attempt_trace"] = solver_attempt_trace
    diagnostics["full_surface_backend_id"] = full.backend_id
    diagnostics["evaluation_cache"] = context.cache.as_dict()
    diagnostics["timers"] = context.timers.as_dict()
    optimizer_status_code = int(getattr(result, "status", -1))
    optimizer_converged = bool(result.success) and optimizer_status_code != 9
    qpos_bounds_pass = bool(
        np.all(qpos >= np.asarray(context.robot_model.joint_lower) - 1e-10)
        and np.all(qpos <= np.asarray(context.robot_model.joint_upper) + 1e-10)
    )
    slack_bounds_pass = bool(
        np.all(slack >= -1e-10) and np.all(slack <= context.paper.b - context.paper.tau + 1e-10)
    )
    full_surface_hard_audit_pass = bool(
        np.all(np.asarray(full.signed_distance) >= -context.paper.b - 1e-6)
    )
    unqueried_final = np.setdiff1d(
        np.arange(len(full.signed_distance)), query_set.sample_ids, assume_unique=True
    )
    full_surface_soft_audit_pass = bool(
        len(unqueried_final) == 0
        or np.all(np.asarray(full.signed_distance)[unqueried_final] >= -context.paper.tau - 1e-6)
    )
    all_values_finite = bool(
        np.all(
            np.isfinite(
                np.concatenate(
                    [
                        value.reshape(-1),
                        np.asarray(full.signed_distance).reshape(-1),
                        np.asarray(independent["hard_residual"]).reshape(-1),
                        np.asarray(independent["soft_residual"]).reshape(-1),
                    ]
                )
            )
        )
        and np.isfinite(float(diagnostics.get("initial_objective", math.nan)))
        and np.isfinite(float(diagnostics.get("final_objective", math.nan)))
    )
    decision = strict_acceptance_decision(
        optimizer_converged=optimizer_converged,
        optimizer_status_code=optimizer_status_code,
        qpos_bounds_pass=qpos_bounds_pass,
        slack_bounds_pass=slack_bounds_pass,
        active_constraints_feasible=active_constraints_ok,
        full_surface_hard_audit_pass=full_surface_hard_audit_pass,
        full_surface_soft_audit_pass=full_surface_soft_audit_pass,
        active_set_converged=active_set_converged,
        all_values_finite=all_values_finite,
        acceptance_policy_id=solver.acceptance_policy_id,
    )
    accepted = bool(decision["accepted"])
    acceptance_reason = str(decision["acceptance_reason"])
    # feasible_stationary_v1 is deliberately not implemented in this closeout.
    # Keep the artifact field explicit without treating a placeholder as a
    # stationarity proof or as part of strict acceptance.
    stationarity_checked = False
    stationarity_residual = float("nan")
    solver_success = accepted
    failure = None
    if not solver_success:
        failure = (
            str(result.message)
            if not optimizer_converged
            else f"strict acceptance failed: {acceptance_reason}"
        )
    return FinalFrameResult(
        qpos=np.asarray(qpos, dtype=np.float64),
        base_pose_scene=_as_np(
            context.base_pose_torch(
                __import__("torch").as_tensor(value, dtype=__import__("torch").float64)
            )
        ),
        base_correction=value[:6].copy(),
        slack=np.asarray(slack, dtype=np.float64),
        query_set=query_set,
        breakdown=breakdown,
        warm_breakdown=warm_breakdown,
        signed_distance=np.asarray(independent["signed_distance"], dtype=np.float64),
        hard_residual=np.asarray(independent["hard_residual"], dtype=np.float64),
        soft_residual=np.asarray(independent["soft_residual"], dtype=np.float64),
        full_signed_distance=np.asarray(full.signed_distance, dtype=np.float64),
        full_closest_points=np.asarray(full.closest_points, dtype=np.float64),
        full_surface_normals=np.asarray(full.surface_normals, dtype=np.float64),
        solver_success=solver_success,
        solver_status=optimizer_status_code,
        solver_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(diagnostics.get("objective_evaluations", 0)),
        jacobian_evaluations=int(diagnostics.get("constraint_jacobian_evaluations", 0)),
        optimizer_converged=optimizer_converged,
        optimizer_status_code=optimizer_status_code,
        optimizer_message=str(result.message),
        optimizer_iterations=int(getattr(result, "nit", 0)),
        optimizer_function_evaluations=int(diagnostics.get("optimizer_function_evaluations", 0)),
        optimizer_jacobian_evaluations=int(diagnostics.get("optimizer_jacobian_evaluations", 0)),
        qpos_bounds_pass=qpos_bounds_pass,
        slack_bounds_pass=slack_bounds_pass,
        active_constraints_feasible=active_constraints_ok,
        full_surface_hard_audit_pass=full_surface_hard_audit_pass,
        full_surface_soft_audit_pass=full_surface_soft_audit_pass,
        active_set_converged=active_set_converged,
        all_values_finite=all_values_finite,
        stationarity_checked=stationarity_checked,
        stationarity_residual=stationarity_residual,
        accepted=accepted,
        acceptance_policy_id=solver.acceptance_policy_id,
        acceptance_reason=acceptance_reason,
        initial_objective=float(diagnostics.get("initial_objective", math.nan)),
        final_objective=float(diagnostics.get("final_objective", math.nan)),
        final_objective_change=float(diagnostics.get("final_objective_change", math.nan)),
        final_step_norm=float(diagnostics.get("final_step_norm", math.nan)),
        solve_time_s=float(time.perf_counter() - started),
        active_set_rounds=query_rounds,
        jacobian_diagnostics=diagnostics,
        failure=failure,
    )


def _make_context(
    sequence: Any,
    graph: InteractionGraphTrajectory,
    warm: WarmStartTrajectory,
    robot_model: Any,
    surface: RobotSurfaceSampleSet,
    sdf: Any,
    reference_sdf: Any,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    paper: PaperRefinementWeights,
    local_index: int,
    previous: np.ndarray | None,
) -> _FrameContext:
    frame = graph.frames[local_index]
    global_frame = int(graph.frame_indices[local_index])
    source_track = sequence.hand(str(warm.metadata["source_hand_id"])).keypoint_tracks[
        "mediapipe21"
    ]
    source_features = _make_source_features(
        source_track.positions_scene[global_frame],
        frame_profile,
        bone_profile,
        str(warm.metadata["source_side"]),
    )
    obj = sequence.rigid_object(str(graph.metadata["object_id"]))
    object_pose = np.asarray(obj.pose_scene.pose_scene[global_frame], dtype=np.float64)
    geometry_indices, transforms, links, slices = _surface_layout(robot_model, surface)
    context_hash = _stable_hash(
        {
            "global_frame": global_frame,
            "robot_name": robot_model.name,
            "object_pose": object_pose.tolist(),
            "graph_frame": int(local_index),
            "source_feature_shape": list(np.asarray(source_features.adjacent_features).shape),
            "paper_hash": paper.config_hash,
            "surface_profile_hash": surface.profile.profile_hash,
        }
    )
    return _FrameContext(
        robot_model=robot_model,
        graph_frame=frame,
        source_features=source_features,
        frame_profile=frame_profile,
        bone_profile=bone_profile,
        seed_base=np.asarray(warm.arrays["base_pose_scene"][local_index], dtype=np.float64),
        seed_qpos=np.asarray(warm.arrays["qpos"][local_index], dtype=np.float64),
        previous_reference=previous,
        paper=paper,
        sdf=sdf,
        reference_sdf=reference_sdf,
        object_pose_scene=object_pose,
        surface=surface,
        surface_points_local=np.asarray(surface.points_local, dtype=np.float64),
        surface_geometry_indices=geometry_indices,
        surface_local_transforms=transforms,
        surface_link_names=links,
        geometry_slices=slices,
        frame_id=global_frame,
        context_hash=context_hash,
        cache=RefinementEvaluationCache(global_frame, context_hash),
    )


def _ragged(values: list[np.ndarray], width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.empty((0,) if width is None else (0, width)), np.zeros(1, dtype=np.int64)
    arrays = [np.asarray(value) for value in values]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(item) for item in arrays])
    return np.concatenate(arrays, axis=0), offsets


@dataclass
class FinalRetargetTrajectory:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def schema_version(self) -> str:
        return str(self.metadata.get("schema_version", ""))

    @property
    def frame_count(self) -> int:
        return int(np.asarray(self.arrays["qpos"]).shape[0])

    def validate(self) -> FinalRetargetTrajectory:
        if self.schema_version not in {
            FINAL_REFINEMENT_SCHEMA_VERSION_V1,
            FINAL_REFINEMENT_SCHEMA_VERSION_V2,
        }:
            raise ValueError(f"unsupported final artifact schema: {self.schema_version}")
        t = self.frame_count
        required = {
            "timestamps": (t,),
            "qpos": (t, 22),
            "base_pose_scene": (t, 4, 4),
            "base_corrections": (t, 6),
            "robot_keypoints_base": (t, 21, 3),
            "robot_keypoints_scene": (t, 21, 3),
            "collision_points_scene": (t, 512, 3),
            "slack_concat": (None,),
            "query_offsets": (t + 1,),
            "full_signed_distance": (t, 512),
            "full_closest_points": (t, 512, 3),
            "full_surface_normals": (t, 512, 3),
            "full_hard_residual": (t, 512),
            "full_soft_violation_count": (t,),
            "unqueried_soft_violation_count": (t,),
            "active_set_converged": (t,),
            "robot_link_poses": (None,),
            "valid_mask": (t,),
        }
        for name, shape in required.items():
            if name not in self.arrays:
                raise ValueError(f"final artifact missing array: {name}")
            value = np.asarray(self.arrays[name])
            if shape[0] is not None and tuple(value.shape) != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if shape[0] is None and value.ndim != 1:
                if name != "robot_link_poses" or value.ndim != 4 or value.shape[0] != t:
                    raise ValueError(f"{name} has invalid variable shape {value.shape}")
        optional_frame_arrays = {
            "optimizer_converged": (t,),
            "optimizer_status_code": (t,),
            "optimizer_message": (t,),
            "optimizer_iterations": (t,),
            "optimizer_function_evaluations": (t,),
            "optimizer_jacobian_evaluations": (t,),
            "qpos_bounds_pass": (t,),
            "slack_bounds_pass": (t,),
            "active_constraints_feasible": (t,),
            "full_surface_hard_audit_pass": (t,),
            "full_surface_soft_audit_pass": (t,),
            "all_values_finite": (t,),
            "stationarity_checked": (t,),
            "stationarity_residual": (t,),
            "accepted": (t,),
            "acceptance_reason": (t,),
            "initial_objective": (t,),
            "final_objective": (t,),
            "final_objective_change": (t,),
            "final_step_norm": (t,),
        }
        present = set(optional_frame_arrays).intersection(self.arrays)
        if present and present != set(optional_frame_arrays):
            missing = sorted(set(optional_frame_arrays) - present)
            raise ValueError(f"new Stage 9 termination contract is incomplete: {missing}")
        for name in present:
            if tuple(np.asarray(self.arrays[name]).shape) != optional_frame_arrays[name]:
                raise ValueError(
                    f"{name} has shape {np.asarray(self.arrays[name]).shape}, "
                    f"expected {optional_frame_arrays[name]}"
                )
        return self

    def arrays_for_storage(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.arrays.items()}


@dataclass
class RefinementResources:
    """Immutable per-run resources shared by every refinement frame."""

    paper: PaperRefinementWeights
    object_vertices: np.ndarray
    object_faces: np.ndarray
    mesh_hash: str
    reference_sdf: Any
    sdf: Any
    sdf_report: dict[str, Any]
    build_counts: dict[str, int]


def prepare_refinement_resources(
    sequence: Any,
    graph: InteractionGraphTrajectory,
    solver_profile: RefinementSolverProfile,
    *,
    object_vertices: np.ndarray | None = None,
    object_faces: np.ndarray | None = None,
    sdf_tree_leaf_size: int = 32,
) -> RefinementResources:
    """Build mesh/SDF resources once for a refinement run."""

    from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend

    paper = PaperRefinementWeights.load()
    if object_vertices is None or object_faces is None:
        obj = sequence.rigid_object(str(graph.metadata["object_id"]))
        object_vertices = obj.mesh.vertices_local
        object_faces = obj.mesh.faces
    vertices = np.asarray(object_vertices, dtype=np.float64)
    faces = np.asarray(object_faces, dtype=np.int64)
    mesh_audit = audit_mesh(vertices, faces)
    reference_sdf = build_signed_distance_backend(
        vertices, faces, sign_mode="strict", mesh_hash=mesh_audit.mesh_hash
    )
    obj = sequence.rigid_object(str(graph.metadata["object_id"]))
    sdf, sdf_report = choose_solver_sdf_backend(
        vertices,
        faces,
        reference_sdf,
        solver_profile,
        object_pose_scene=np.asarray(obj.pose_scene.pose_scene[0]),
        tree_leaf_size=sdf_tree_leaf_size,
    )
    return RefinementResources(
        paper=paper,
        object_vertices=vertices,
        object_faces=faces,
        mesh_hash=mesh_audit.mesh_hash,
        reference_sdf=reference_sdf,
        sdf=sdf,
        sdf_report=sdf_report,
        build_counts={
            "mesh_load_count": 1,
            "solver_sdf_build_count": 1,
            "reference_sdf_build_count": 1,
            "convex_hull_build_count": int(
                getattr(sdf, "backend_id", "") == "convex_hull_exact_solver_only"
            ),
            "bvh_build_count": 1,
        },
    )


def final_artifact_hash(trajectory: FinalRetargetTrajectory) -> str:
    """Return a stable content hash independent of Zarr chunk layout."""

    metadata = dict(trajectory.metadata)
    metadata["artifact_hash"] = None
    arrays = {
        name: {
            "dtype": str(np.asarray(value).dtype),
            "shape": list(np.asarray(value).shape),
            "sha256": _sha256_bytes(np.asarray(value).tobytes(order="C")),
        }
        for name, value in sorted(trajectory.arrays.items())
    }
    return _stable_hash({"metadata": metadata, "arrays": arrays})


def save_final_trajectory(
    trajectory: FinalRetargetTrajectory, path: str | Path, *, force: bool = False
) -> Path:
    trajectory.validate()
    destination = Path(path).expanduser()
    if destination.exists() and not force:
        raise FileExistsError(f"artifact exists; pass --force: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    metadata = dict(trajectory.metadata)
    metadata["schema_version"] = str(
        metadata.get("schema_version", FINAL_REFINEMENT_SCHEMA_VERSION_V1)
    )
    metadata["array_manifest"] = sorted(trajectory.arrays)
    try:
        write_zarr3_group_direct(
            temporary,
            {
                "schema_version": metadata["schema_version"],
                "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
            },
            trajectory.arrays_for_storage(),
            array_prefix="",
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_final_trajectory(path: str | Path) -> FinalRetargetTrajectory:
    source = Path(path).expanduser()
    root_metadata = source / "zarr.json"
    group: Any = None
    if root_metadata.is_file():
        root = json.loads(root_metadata.read_text(encoding="utf-8"))
        attributes = root.get("attributes", {})
    else:
        import zarr

        group = zarr.open_group(source, mode="r")
        attributes = group.attrs
    schema_version = attributes.get("schema_version")
    if schema_version not in {
        FINAL_REFINEMENT_SCHEMA_VERSION_V1,
        FINAL_REFINEMENT_SCHEMA_VERSION_V2,
    }:
        raise ValueError("unsupported final artifact schema")
    metadata = json.loads(str(attributes["metadata_json"]))
    if root_metadata.is_file():
        arrays = direct_zarr3_arrays(source, metadata["array_manifest"], array_prefix="")
    else:
        arrays = {name: np.asarray(group[name][:]) for name in metadata["array_manifest"]}
    return FinalRetargetTrajectory(metadata, arrays).validate()


def build_final_trajectory(
    sequence: Any,
    warm: WarmStartTrajectory,
    graph: InteractionGraphTrajectory,
    robot_model: Any,
    surface: RobotSurfaceSampleSet,
    frame_profile: BoneDirectionFrameProfile,
    bone_profile: BoneDirectionProfile,
    coordinate_profile: RefinementCoordinateProfile,
    query_profile: CollisionQueryProfile,
    solver_profile: RefinementSolverProfile,
    *,
    start_frame: int = 0,
    end_frame: int | None = None,
    initial_previous: tuple[np.ndarray, np.ndarray] | None = None,
    object_vertices: np.ndarray | None = None,
    object_faces: np.ndarray | None = None,
    warm_artifact_hash: str | None = None,
    graph_artifact_hash: str | None = None,
    continue_on_failure: bool = False,
    resources: RefinementResources | None = None,
    frame_callback: Callable[[int, FinalFrameResult, _FrameContext], None] | None = None,
    pause_check: Callable[[int], bool] | None = None,
    source_frame_offset: int = 0,
    execution_profile: Any | None = None,
) -> tuple[FinalRetargetTrajectory, dict[str, Any]]:
    warm.validate()
    graph.validate()
    if warm.frame_count != graph.frame_count:
        raise ValueError("warm-start and graph frame counts differ")
    if warm.metadata.get("robot_name") != robot_model.name:
        raise ValueError("warm-start and selected robot differ")
    if not np.array_equal(warm.arrays["timestamps"], graph.timestamps):
        raise ValueError("warm-start and graph timestamps differ")
    point_jacobian_backend = str(
        getattr(execution_profile, "point_jacobian_backend", "reference_batched_torch_v1")
    )
    strict_recovery = str(getattr(execution_profile, "strict_recovery", "none"))
    sdf_tree_leaf_size = int(getattr(execution_profile, "sdf_tree_leaf_size", 32))
    resources = resources or prepare_refinement_resources(
        sequence,
        graph,
        solver_profile,
        object_vertices=object_vertices,
        object_faces=object_faces,
        sdf_tree_leaf_size=sdf_tree_leaf_size,
    )
    paper = resources.paper
    object_vertices = resources.object_vertices
    object_faces = resources.object_faces
    reference_sdf = resources.reference_sdf
    sdf = resources.sdf
    sdf_report = resources.sdf_report
    stop = warm.frame_count if end_frame is None else int(end_frame)
    if start_frame < 0 or stop <= start_frame or stop > warm.frame_count:
        raise ValueError(f"invalid frame range [{start_frame},{stop})")
    frame_indices = list(range(start_frame, stop))
    frames: list[FinalFrameResult] = []
    previous_base: np.ndarray | None = None
    previous_qpos: np.ndarray | None = None
    if initial_previous is not None:
        previous_base = np.asarray(initial_previous[0], dtype=np.float64)
        previous_qpos = np.asarray(initial_previous[1], dtype=np.float64)
    query_summaries: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    paused = False
    for local_index in frame_indices:
        if pause_check is not None and pause_check(local_index):
            paused = True
            break
        previous = None
        if previous_base is not None and previous_qpos is not None:
            previous = map_previous_state_to_seed(
                previous_base, previous_qpos, warm.arrays["base_pose_scene"][local_index]
            )
        context = _make_context(
            sequence,
            graph,
            warm,
            robot_model,
            surface,
            sdf,
            reference_sdf,
            frame_profile,
            bone_profile,
            paper,
            local_index,
            previous,
        )
        initial_points = context.candidate_points(np.concatenate([np.zeros(6), context.seed_qpos]))
        with context.timers.measure("full_512_audit"):
            initial_full = context.reference_sdf.query_scene(
                initial_points, context.object_pose_scene
            )
        if not np.all(initial_full.sign_valid):
            raise ValueError("initial full-surface audit received invalid signed distance")
        context.full_audit_call_count = 1
        context.full_audit_call_reasons = ["frame_query_set_initialization"]
        # The selected solver backend is exact for this audited convex mesh and
        # has passed probe comparison. It is sufficient for initial QuerySet
        # selection; reference_sdf remains the independent persisted audit.
        initial_query = sdf.query_scene(initial_points, context.object_pose_scene)
        query_set = build_query_set(
            initial_query.signed_distance, surface.geometry_ids, query_profile
        )
        if query_profile.mode == "full":
            query_set = CollisionQuerySet(
                query_set.sample_ids,
                query_set.inclusion_reasons,
                query_set.active_round,
                np.asarray(initial_query.signed_distance)[query_set.sample_ids],
                query_set.query_hash,
            )
        frame_result = refine_frame(
            context,
            query_set,
            solver_profile,
            max_rounds=query_profile.max_active_set_rounds,
            active_margin_m=query_profile.active_margin_m,
            point_jacobian_backend=point_jacobian_backend,
            strict_recovery=strict_recovery,
        )
        frames.append(frame_result)
        query_summaries.append(
            {
                "frame": int(graph.frame_indices[local_index]),
                "initial_query_count": query_set.count,
                "final_query_count": frame_result.query_set.count,
                "query_hash": frame_result.query_set.query_hash,
                "active_set_rounds": frame_result.active_set_rounds,
                "active_set_converged": frame_result.active_set_converged,
                "optimizer_converged": frame_result.optimizer_converged,
                "accepted": frame_result.accepted,
                "continuation": frame_result.jacobian_diagnostics.get(
                    "active_set_continuation", []
                ),
                "full_audit_call_count": frame_result.jacobian_diagnostics.get(
                    "full_audit_call_count", 0
                ),
                "cache": frame_result.jacobian_diagnostics.get("cache", {}),
                "timers": frame_result.jacobian_diagnostics.get("timers", {}),
            }
        )
        performance_rows.append(
            {
                "frame": int(graph.frame_indices[local_index]),
                "solve_time_s": float(frame_result.solve_time_s),
                "active_set_rounds": int(frame_result.active_set_rounds),
                "full_audit_call_count": int(
                    frame_result.jacobian_diagnostics.get("full_audit_call_count", 0)
                ),
                "objective_calls": int(
                    frame_result.jacobian_diagnostics.get("objective_evaluations", 0)
                ),
                "objective_jacobian_calls": int(
                    frame_result.jacobian_diagnostics.get("objective_jacobian_evaluations", 0)
                ),
                "constraint_calls": int(
                    frame_result.jacobian_diagnostics.get("constraint_evaluations", 0)
                ),
                "constraint_jacobian_calls": int(
                    frame_result.jacobian_diagnostics.get("constraint_jacobian_evaluations", 0)
                ),
                "cache": frame_result.jacobian_diagnostics.get("cache", {}),
                "timers": frame_result.jacobian_diagnostics.get("timers", {}),
            }
        )
        if frame_callback is not None and frame_result.accepted:
            frame_callback(local_index, frame_result, context)
        if (
            not frame_result.solver_success
            and solver_profile.strict_failure_policy == "fail_fast"
            and not continue_on_failure
        ):
            raise RuntimeError(f"Stage 9 frame {local_index} failed: {frame_result.failure}")
        previous_base, previous_qpos = frame_result.base_pose_scene, frame_result.qpos
    if not frames:
        raise RuntimeError("refinement paused before any frame completed")
    processed_frame_indices = frame_indices[: len(frames)]
    qpos = np.stack([item.qpos for item in frames])
    base = np.stack([item.base_pose_scene for item in frames])
    corrections = np.stack([item.base_correction for item in frames])
    slack_concat, slack_offsets = _ragged([item.slack for item in frames])
    query_ids_concat, query_offsets = _ragged([item.query_set.sample_ids for item in frames])
    query_round_concat, _ = _ragged([item.query_set.active_round for item in frames])
    signed_concat, _ = _ragged([item.signed_distance for item in frames])
    hard_concat, _ = _ragged([item.hard_residual for item in frames])
    soft_concat, _ = _ragged([item.soft_residual for item in frames])
    reason_concat, _ = _ragged(
        [np.asarray(item.query_set.inclusion_reasons, dtype="S96") for item in frames]
    )
    timestamps = np.asarray(warm.arrays["timestamps"])[processed_frame_indices]
    collision_points = np.stack(
        [
            dynamic_collision_points_numpy(robot_model, surface, item.qpos, item.base_pose_scene)
            for item in frames
        ]
    )
    robot_link_names = tuple(robot_model.link_names)
    link_pose_rows: list[np.ndarray] = []
    for item in frames:
        fk = robot_model.forward_kinematics_reference(item.qpos)
        link_pose_rows.append(
            np.stack([item.base_pose_scene @ fk[name] for name in robot_link_names])
        )
    robot_link_poses = np.stack(link_pose_rows)

    def series(name: str) -> np.ndarray:
        return np.asarray([getattr(item.breakdown, name) for item in frames], dtype=np.float64)

    def warm_series(name: str) -> np.ndarray:
        return np.asarray([getattr(item.warm_breakdown, name) for item in frames], dtype=np.float64)

    full_phi = np.stack([item.full_signed_distance for item in frames])
    full_hard = full_phi + paper.b
    full_soft_violation_count = np.asarray(
        [np.count_nonzero(item.full_signed_distance < -paper.tau - 1e-6) for item in frames],
        dtype=np.int64,
    )
    unqueried_soft_violation_count = np.asarray(
        [
            np.count_nonzero(
                item.full_signed_distance[
                    np.setdiff1d(
                        np.arange(surface.count), item.query_set.sample_ids, assume_unique=True
                    )
                ]
                < -paper.tau - 1e-6
            )
            for item in frames
        ],
        dtype=np.int64,
    )
    lower, upper = robot_model.joint_lower, robot_model.joint_upper
    arrays = {
        "timestamps": timestamps,
        "frame_indices": np.asarray(graph.frame_indices[processed_frame_indices], dtype=np.int64),
        "source_frame_indices": np.asarray(
            graph.frame_indices[processed_frame_indices] + int(source_frame_offset), dtype=np.int64
        ),
        "qpos": qpos,
        "base_pose_scene": base,
        "base_corrections": corrections,
        "robot_keypoints_base": np.stack(
            [_as_np(robot_model.keypoints_base(item.qpos)) for item in frames]
        ),
        "robot_keypoints_scene": np.stack(
            [
                _as_np(robot_model.keypoints_scene(item.qpos, item.base_pose_scene))
                for item in frames
            ]
        ),
        "collision_points_scene": collision_points,
        "robot_link_poses": robot_link_poses,
        "joint_limit_margins": np.minimum(qpos - lower[None, :], upper[None, :] - qpos),
        "e_im": series("e_im"),
        "e_bone": series("e_bone"),
        "e_temporal": series("e_temporal"),
        "e_base_pos": series("e_base_pos"),
        "e_base_rot": series("e_base_rot"),
        "e_slack": series("e_slack"),
        "weighted_e_im": series("weighted_e_im"),
        "weighted_e_bone": series("weighted_e_bone"),
        "total_objective": series("total"),
        "warm_e_im": warm_series("e_im"),
        "warm_e_bone": warm_series("e_bone"),
        "warm_total_objective": warm_series("total"),
        "query_ids_concat": query_ids_concat.astype(np.int64),
        "query_offsets": query_offsets,
        "query_active_round_concat": query_round_concat.astype(np.int64),
        "query_inclusion_reason_concat": reason_concat,
        "slack_concat": slack_concat.astype(np.float64),
        "slack_offsets": slack_offsets,
        "signed_distance_concat": signed_concat.astype(np.float64),
        "hard_residual_concat": hard_concat.astype(np.float64),
        "soft_residual_concat": soft_concat.astype(np.float64),
        "full_signed_distance": full_phi,
        "full_closest_points": np.stack([item.full_closest_points for item in frames]),
        "full_surface_normals": np.stack([item.full_surface_normals for item in frames]),
        "full_hard_residual": full_hard,
        "full_soft_violation_count": full_soft_violation_count,
        "unqueried_soft_violation_count": unqueried_soft_violation_count,
        "min_full_signed_distance": np.min(full_phi, axis=1),
        "max_penetration": np.maximum(0.0, -np.min(full_phi, axis=1)),
        "solver_success": np.asarray([item.solver_success for item in frames], dtype=bool),
        "valid_mask": np.asarray([item.solver_success for item in frames], dtype=bool),
        "solver_status": np.asarray([item.solver_status for item in frames], dtype=np.int64),
        "iterations": np.asarray([item.iterations for item in frames], dtype=np.int64),
        "function_evaluations": np.asarray(
            [item.function_evaluations for item in frames], dtype=np.int64
        ),
        "jacobian_evaluations": np.asarray(
            [item.jacobian_evaluations for item in frames], dtype=np.int64
        ),
        "solve_time_s": np.asarray([item.solve_time_s for item in frames], dtype=np.float64),
        "active_set_rounds": np.asarray(
            [item.active_set_rounds for item in frames], dtype=np.int64
        ),
        "active_set_converged": np.asarray(
            [item.active_set_converged for item in frames],
            dtype=bool,
        ),
        "optimizer_converged": np.asarray(
            [item.optimizer_converged for item in frames], dtype=bool
        ),
        "optimizer_status_code": np.asarray(
            [item.optimizer_status_code for item in frames], dtype=np.int64
        ),
        "optimizer_message": np.asarray([item.optimizer_message for item in frames], dtype="S256"),
        "optimizer_iterations": np.asarray(
            [item.optimizer_iterations for item in frames], dtype=np.int64
        ),
        "optimizer_function_evaluations": np.asarray(
            [item.optimizer_function_evaluations for item in frames], dtype=np.int64
        ),
        "optimizer_jacobian_evaluations": np.asarray(
            [item.optimizer_jacobian_evaluations for item in frames], dtype=np.int64
        ),
        "qpos_bounds_pass": np.asarray([item.qpos_bounds_pass for item in frames], dtype=bool),
        "slack_bounds_pass": np.asarray([item.slack_bounds_pass for item in frames], dtype=bool),
        "active_constraints_feasible": np.asarray(
            [item.active_constraints_feasible for item in frames], dtype=bool
        ),
        "full_surface_hard_audit_pass": np.asarray(
            [item.full_surface_hard_audit_pass for item in frames], dtype=bool
        ),
        "full_surface_soft_audit_pass": np.asarray(
            [item.full_surface_soft_audit_pass for item in frames], dtype=bool
        ),
        "all_values_finite": np.asarray([item.all_values_finite for item in frames], dtype=bool),
        "stationarity_checked": np.asarray(
            [item.stationarity_checked for item in frames], dtype=bool
        ),
        "stationarity_residual": np.asarray(
            [item.stationarity_residual for item in frames], dtype=np.float64
        ),
        "accepted": np.asarray([item.accepted for item in frames], dtype=bool),
        "acceptance_reason": np.asarray([item.acceptance_reason for item in frames], dtype="S512"),
        "initial_objective": np.asarray(
            [item.initial_objective for item in frames], dtype=np.float64
        ),
        "final_objective": np.asarray([item.final_objective for item in frames], dtype=np.float64),
        "final_objective_change": np.asarray(
            [item.final_objective_change for item in frames], dtype=np.float64
        ),
        "final_step_norm": np.asarray([item.final_step_norm for item in frames], dtype=np.float64),
    }
    object_mesh_hash = resources.mesh_hash
    metadata = {
        "schema_version": (
            FINAL_REFINEMENT_SCHEMA_VERSION_V2
            if solver_profile.profile_id == CONTACT_RICH_SOLVER_PROFILE_ID
            else FINAL_REFINEMENT_SCHEMA_VERSION_V1
        ),
        "artifact_type": "final_interaction_preserving_robot_reference",
        "source_sequence_id": warm.metadata.get("source_sequence_id"),
        "source_hand_id": warm.metadata.get("source_hand_id"),
        "source_hand_side": warm.metadata.get("source_side"),
        "source_canonical_hash": warm.metadata.get("source_cache_hash"),
        "object_id": graph.metadata.get("object_id"),
        "object_mesh_hash": object_mesh_hash,
        "object_sample_artifact_hash": graph.metadata.get("object_sample_artifact_hash"),
        "graph_artifact_hash": graph_artifact_hash
        or (interaction_artifact_hash(graph.source_path) if graph.source_path else None),
        "warm_start_artifact_hash": warm_artifact_hash,
        "robot_name": robot_model.name,
        "robot_side": robot_model.side,
        "robot_spec_hash": robot_model.spec_hash,
        "robot_urdf_hash": robot_model.urdf_hash,
        "robot_asset_manifest_hash": robot_model.asset_manifest_hash,
        "robot_link_names": list(robot_link_names),
        "collision_surface_profile_hash": surface.profile.profile_hash,
        "collision_surface_sample_count": surface.count,
        "sdf_backend": sdf.describe(),
        "sdf_reference_backend": reference_sdf.describe(),
        "query_profile": query_profile.as_dict(),
        "coordinate_profile": coordinate_profile.as_dict(),
        "solver_profile": solver_profile.as_dict(),
        "solver_profile_id": solver_profile.profile_id,
        "solver_profile_hash": solver_profile.profile_hash,
        "execution_profile": (
            None if execution_profile is None else execution_profile.as_dict()
        ),
        "point_jacobian_backend": point_jacobian_backend,
        "strict_recovery": strict_recovery,
        "sdf_tree_leaf_size": sdf_tree_leaf_size,
        "termination_contract": solver_profile.termination_contract,
        "acceptance_policy_id": solver_profile.acceptance_policy_id,
        "active_set_continuation_policy": solver_profile.active_set_continuation_policy,
        "maxiter_provenance": solver_profile.maxiter_provenance,
        "stationarity_policy": solver_profile.stationarity_policy,
        "paper_weights": paper.as_dict(),
        "sdf_selection_report": sdf_report,
        "frame_range": [
            int(graph.frame_indices[processed_frame_indices[0]]),
            int(graph.frame_indices[processed_frame_indices[-1]]) + 1,
        ],
        "requested_frame_range": [
            int(graph.frame_indices[start_frame]),
            int(graph.frame_indices[stop - 1]) + 1,
        ],
        "native_fps": warm.metadata.get("native_fps"),
        "frame_count": len(frames),
        "paused": paused,
        "initial_previous_frame": None if initial_previous is None else int(frame_indices[0] - 1),
        "timestamps": timestamps.tolist(),
        "assumptions": sorted(
            set(
                coordinate_profile.assumptions
                + query_profile.assumptions
                + solver_profile.assumptions
                + (
                    "A_REFINEMENT_BASE_PARAMETERIZATION_001",
                    "A_REFINEMENT_BASE_PRIOR_REFERENCE_001",
                    "A_REFINEMENT_FIRST_FRAME_001",
                    "A_REFINEMENT_COORDINATE_SCALING_001",
                    "A_REFINEMENT_TIME_DISCRETIZATION_001",
                    "A_REFINEMENT_QUERY_SET_001",
                    "A_COLLISION_ACTIVE_MARGIN_001",
                    "A_COLLISION_ACTIVE_SET_REFINEMENT_001",
                    "A_SDF_CONSTRAINT_JACOBIAN_001",
                    "A_REFINEMENT_SDF_BACKEND_001",
                    "A_REFINEMENT_SOLVER_001",
                    "A_REFINEMENT_SOLVER_TOLERANCES_001",
                    "A_REFINEMENT_HAND_SURFACE_SAMPLES_001",
                    "A_ARTIMANO_COLLISION_COVERAGE_001",
                )
            )
        ),
        "provenance": {
            "stage6_inputs_unchanged": True,
            "stage7_inputs_unchanged": True,
            "stage8_inputs_unchanged": True,
            "graph_rebuilt": False,
            "delaunay_rebuilt": False,
            "semantic_contacts_used": False,
            "visual_collision_fallback": False,
            "physics_simulation": False,
            "rl_used": False,
            "stage10_started": False,
        },
        "query_summaries": query_summaries,
        "performance": {
            "frame_rows": performance_rows,
            "resource_build_counts": {
                "robot_model_load_count": 1,
                "graph_load_count": 1,
                "source_feature_build_count": len(frames),
                **resources.build_counts,
            },
            "full_audit_in_inner_callbacks": False,
            "point_jacobian_backend": point_jacobian_backend,
            "strict_recovery": strict_recovery,
            "sdf_tree_leaf_size": sdf_tree_leaf_size,
            "variable_scaling": "seed_delta_normalized_v1",
        },
        "solver_messages": [item.solver_message for item in frames],
        "termination_contract_summary": {
            "optimizer_converged_required": True,
            "primal_bounds_and_full_audits_required": True,
            "active_set_converged_required": True,
            "status_9_is_not_accepted": True,
            "stationarity_policy": solver_profile.stationarity_policy,
        },
        "artifact_hash": None,
    }
    trajectory = FinalRetargetTrajectory(metadata, arrays).validate()
    return trajectory, {
        "sdf": sdf_report,
        "query_summaries": query_summaries,
        "performance": performance_rows,
        "paused": paused,
        "processed_frame_indices": [int(item) for item in processed_frame_indices],
        "resource_counts": resources.build_counts,
    }


__all__ = [
    "ACTIVE_QUERY_PROFILE_ID",
    "CONTACT_RICH_SOLVER_PROFILE_ID",
    "COORDINATE_PROFILE_ID",
    "CollisionQueryProfile",
    "CollisionQuerySet",
    "DEFERRED_STATIONARITY_POLICY_ID",
    "FinalFrameResult",
    "FinalObjectiveBreakdown",
    "FinalRetargetTrajectory",
    "FULL_QUERY_PROFILE_ID",
    "FULL_SOLVER_PROFILE_ID",
    "FINAL_REFINEMENT_SCHEMA_VERSION_V1",
    "FINAL_REFINEMENT_SCHEMA_VERSION_V2",
    "PaperRefinementWeights",
    "RefinementCoordinateProfile",
    "RefinementResources",
    "RefinementSolverProfile",
    "SOLVER_PROFILE_ID",
    "STRICT_ACCEPTANCE_POLICY_ID",
    "active_set_is_monotonic",
    "continue_active_set_initial",
    "build_final_trajectory",
    "build_query_set",
    "choose_solver_sdf_backend",
    "dynamic_collision_points_numpy",
    "expand_query_set",
    "final_artifact_hash",
    "load_final_trajectory",
    "load_robot_surface_samples",
    "map_previous_state_to_seed",
    "prepare_refinement_resources",
    "save_final_trajectory",
    "so3_exp",
    "so3_log",
    "strict_acceptance_decision",
]
