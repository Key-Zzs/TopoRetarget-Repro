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
from toporetarget.geometry.signed_distance.base import (
    SignedDistanceBackend,
    SignedDistanceQueryResult,
)
from toporetarget.geometry.signed_distance.compiled_sdf_cpu import (
    CompiledSDFUnavailable,
    CompiledSpatialFDBackend,
)
from toporetarget.geometry.signed_distance.derived_proxy import (
    HybridSignedDistanceBackend,
    ObjectSDFGeometryPolicy,
    build_hybrid_signed_distance_backend,
)
from toporetarget.geometry.signed_distance.gradient import (
    SignedDistanceGradientAmbiguityPolicy,
    ambiguity_reason_counts,
    analytic_spatial_gradient,
)
from toporetarget.geometry.signed_distance.sign_cache import LipschitzSignCache
from toporetarget.retarget.artifacts import WarmStartTrajectory
from toporetarget.retarget.bones import (
    BoneDirectionProfile,
    BoneFeatures,
    extract_bone_features,
)
from toporetarget.retarget.continuous import (
    BASE_CORRECTION_CONVENTION,
    CONTINUOUS_PROFILE_IDS,
    LAMBDA_CORR,
    S_POS_M,
    S_Q_RAD,
    S_ROT_RAD,
    ContinuousRetargetProfile,
    PropagatedRetargetState,
    RecedingHorizonWindow,
    continuity_metrics,
    correction_temporal_energy,
    encode_base_correction,
    is_continuous_profile,
    transport_previous_final_to_current_warm,
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
FINAL_REFINEMENT_SCHEMA_VERSION_V3 = "toporetarget.final_retarget.v3"
# Historical callers and v1 fixtures continue to use this public alias.
FINAL_REFINEMENT_SCHEMA_VERSION = FINAL_REFINEMENT_SCHEMA_VERSION_V1
COORDINATE_PROFILE_ID = "local_seed_delta_v1"
FULL_QUERY_PROFILE_ID = "full_collision_surface_reference_v1"
ACTIVE_QUERY_PROFILE_ID = "adaptive_active_set_v1"
SOLVER_PROFILE_ID = "scipy_slsqp_active_set_v1"
CONTACT_RICH_SOLVER_PROFILE_ID = "scipy_slsqp_active_set_contact_rich_v2"
FAITHFUL_CONTACT_RICH_SOLVER_PROFILE_ID = "scipy_slsqp_active_set_contact_rich_v3_fixed"
FULL_SOLVER_PROFILE_ID = "scipy_slsqp_full_surface_reference_v1"
STRICT_ACCEPTANCE_POLICY_ID = "strict_optimizer_converged_and_audits_v1"
DEFERRED_STATIONARITY_POLICY_ID = "feasible_stationary_v1_deferred"
# A newly expanded soft constraint is initialized from a reference-SDF value,
# while the first SLSQP callback may use the solver-SDF backend.  Keep a tiny,
# fixed interior margin so round-off between those two representations cannot
# make an otherwise feasible continuation point start slightly infeasible.
# This is initialization conditioning only; it is not an objective, tolerance,
# active-margin, or acceptance-policy change.
ACTIVE_SET_CONTINUATION_FEASIBILITY_BUFFER_M = 1.0e-9
VIRTUAL_CLOSURE_QUERY_FRACTION_LIMIT = 0.02


def regularization_profile_for_solver(
    solver_profile_id: str, requested_profile: str = "auto"
) -> str:
    """Bind the versioned faithful solver ID to its Eq. (9) temporal semantics."""

    if requested_profile != "auto":
        return requested_profile
    if solver_profile_id == FAITHFUL_CONTACT_RICH_SOLVER_PROFILE_ID:
        return "faithful_regularization_fix_v1"
    if solver_profile_id in CONTINUOUS_PROFILE_IDS:
        return "wuji_continuous_full_state_v1"
    return "faithful_current_baseline"


def _as_np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _finite_difference_inputs(
    current: np.ndarray,
    *,
    variable_count: int,
    row_ids: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    value = np.asarray(current, dtype=np.float64).reshape(-1)
    rows = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    step = float(epsilon)
    if variable_count < 0 or variable_count > len(value):
        raise ValueError("variable_count must select a prefix of current")
    if np.any(rows < 0):
        raise ValueError("row_ids must be non-negative")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return value, rows, step


def _batched_constraint_finite_difference(
    residual: Callable[[np.ndarray], np.ndarray],
    current: np.ndarray,
    *,
    variable_count: int,
    row_ids: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, int]:
    """Evaluate each central-difference column once for every requested row."""

    value, rows, step = _finite_difference_inputs(
        current,
        variable_count=variable_count,
        row_ids=row_ids,
        epsilon=epsilon,
    )
    output = np.empty((len(rows), variable_count), dtype=np.float64)
    if len(rows) == 0 or variable_count == 0:
        return output, 0
    calls = 0
    for column in range(variable_count):
        plus = value.copy()
        minus = value.copy()
        plus[column] += step
        minus[column] -= step
        plus_values = np.asarray(residual(plus), dtype=np.float64).reshape(-1)
        minus_values = np.asarray(residual(minus), dtype=np.float64).reshape(-1)
        calls += 2
        if int(rows.max()) >= len(plus_values) or len(minus_values) != len(plus_values):
            raise ValueError("residual output does not cover row_ids consistently")
        output[:, column] = (plus_values[rows] - minus_values[rows]) / (2.0 * step)
    return output, calls


def _vectorized_constraint_finite_difference(
    residual_batch: Callable[[np.ndarray], np.ndarray],
    current: np.ndarray,
    *,
    variable_count: int,
    row_ids: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, int]:
    """Evaluate all central-difference perturbations in one batched callback."""

    value, rows, step = _finite_difference_inputs(
        current,
        variable_count=variable_count,
        row_ids=row_ids,
        epsilon=epsilon,
    )
    output = np.empty((len(rows), variable_count), dtype=np.float64)
    if len(rows) == 0 or variable_count == 0:
        return output, 0
    probes = np.repeat(value[None, :], 2 * variable_count, axis=0)
    columns = np.arange(variable_count, dtype=np.int64)
    probes[2 * columns, columns] += step
    probes[2 * columns + 1, columns] -= step
    residuals = np.asarray(residual_batch(probes), dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[0] != len(probes):
        raise ValueError("batched residual must return shape [probe_count, residual_count]")
    if int(rows.max()) >= residuals.shape[1]:
        raise ValueError("batched residual output does not cover row_ids")
    plus_values = residuals[0::2][:, rows]
    minus_values = residuals[1::2][:, rows]
    output[:] = ((plus_values - minus_values) / (2.0 * step)).T
    return output, 1


def _column_sparse_constraint_finite_difference(
    requested_residuals: Callable[[np.ndarray, list[np.ndarray]], list[np.ndarray]],
    current: np.ndarray,
    *,
    dependency_mask: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, int, int]:
    """Batch only row/column pairs allowed by the structural dependency mask."""

    dependencies = np.asarray(dependency_mask, dtype=bool)
    if dependencies.ndim != 2:
        raise ValueError("dependency_mask must have shape [row_count, variable_count]")
    row_count, variable_count = dependencies.shape
    value, _, step = _finite_difference_inputs(
        current,
        variable_count=variable_count,
        row_ids=np.empty(0, dtype=np.int64),
        epsilon=epsilon,
    )
    output = np.zeros((row_count, variable_count), dtype=np.float64)
    probes: list[np.ndarray] = []
    row_blocks: list[np.ndarray] = []
    column_blocks: list[tuple[int, np.ndarray, int, int]] = []
    for column in range(variable_count):
        rows = np.flatnonzero(dependencies[:, column])
        if len(rows) == 0:
            continue
        plus = value.copy()
        minus = value.copy()
        plus[column] += step
        minus[column] -= step
        plus_index = len(probes)
        probes.extend((plus, minus))
        row_blocks.extend((rows, rows))
        column_blocks.append((column, rows, plus_index, plus_index + 1))
    if not probes:
        return output, 0, 0
    values = np.stack(probes)
    residual_blocks = requested_residuals(values, row_blocks)
    if len(residual_blocks) != len(row_blocks):
        raise ValueError("requested residual callback returned an inconsistent block count")
    for column, rows, plus_index, minus_index in column_blocks:
        plus_values = np.asarray(residual_blocks[plus_index], dtype=np.float64).reshape(-1)
        minus_values = np.asarray(residual_blocks[minus_index], dtype=np.float64).reshape(-1)
        if len(plus_values) != len(rows) or len(minus_values) != len(rows):
            raise ValueError("requested residual callback returned an inconsistent block shape")
        output[rows, column] = (plus_values - minus_values) / (2.0 * step)
    probe_count = sum(len(rows) for rows in row_blocks)
    return output, 1, int(probe_count)


def _virtual_closure_query_gate(
    synthetic_patch_mask: np.ndarray,
    active_query_ids: np.ndarray,
    *,
    full_collision_sample_count: int,
) -> tuple[int, int]:
    """Count active synthetic-patch queries against a fixed 2% full-surface limit."""

    if full_collision_sample_count <= 0:
        raise ValueError("full_collision_sample_count must be positive")
    patch_mask = np.asarray(synthetic_patch_mask, dtype=bool).reshape(-1)
    query_ids = np.asarray(active_query_ids, dtype=np.int64).reshape(-1)
    if len(patch_mask) != full_collision_sample_count:
        raise ValueError("synthetic_patch_mask must cover the full collision sample set")
    if np.any(query_ids < 0) or np.any(query_ids >= full_collision_sample_count):
        raise ValueError("active_query_ids are outside the full collision sample set")
    active_patch_count = int(np.count_nonzero(patch_mask[query_ids]))
    limit = int(math.ceil(VIRTUAL_CLOSURE_QUERY_FRACTION_LIMIT * full_collision_sample_count))
    return active_patch_count, limit


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
        # Zero is reserved for explicitly versioned diagnostic profiles.  The
        # formal adaptive profile remains at 10 mm; allowing zero here lets
        # Stage 9.3.4 isolate the active-margin selection rule without
        # changing Eq. (8)/(9), paper weights, or the strict acceptance gate.
        if self.mode == "adaptive" and self.active_margin_m < 0:
            raise ValueError("adaptive active margin must be non-negative")
        if (
            self.mode == "adaptive"
            and self.active_margin_m == 0
            and self.profile_id != "zero_active_margin_diagnostic_v1"
        ):
            raise ValueError("zero adaptive margin requires the versioned diagnostic profile")
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
    feasibility_buffer_m: float = 0.0,
) -> np.ndarray:
    """Build the next SLSQP initial point from the preceding result.

    The non-slack coordinates are copied byte-for-byte from ``result_x``.  Slack
    values are looked up by stable query ID, so a deterministic reorder of the
    expanded set cannot change an existing slack value.  New slacks use the
    smallest bounded value that satisfies the soft constraint at the returned
    candidate, optionally with a fixed interior buffer for reference/solver
    signed-distance round-off.  This helper is intentionally independent of the
    solver so the continuation contract can be tested without importing a robot
    or mesh.
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
    if not np.isfinite(feasibility_buffer_m) or feasibility_buffer_m < 0:
        raise ValueError("continuation feasibility buffer must be finite and non-negative")
    new_slack_by_id = {
        int(query_id): float(
            np.clip(
                max(-float(tau) - float(phi[query_id]), 0.0) + float(feasibility_buffer_m),
                0.0,
                upper,
            )
        )
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


def _so3_log_torch(rotation: Any) -> Any:
    """Differentiable SO(3) log used only by the continuous profile."""

    import torch

    cosine = torch.clamp(
        (torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5, -1.0, 1.0
    )
    theta = torch.acos(cosine)
    skew = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    safe = torch.clamp(torch.sin(theta), min=1.0e-8)
    scale = torch.where(theta < 1.0e-6, 0.5 - theta.square() / 12.0, theta / (2.0 * safe))
    return skew * scale[..., None]


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
    reference: SignedDistanceBackend,
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
        reference_mesh_hash = reference.describe().get("mesh_hash")
        if not isinstance(reference_mesh_hash, str):
            reference_mesh_hash = audit_mesh(vertices, faces).mesh_hash
        candidate = ConvexHullSignedDistanceBackend(
            vertices,
            faces,
            reference_mesh_hash,
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
    # Optional paper-external quality terms.  They default to zero so legacy
    # paper-core artifacts retain the exact public breakdown contract.
    e_morph: float = 0.0
    weighted_e_morph: float = 0.0
    e_contact_pos: float = 0.0
    weighted_e_contact_pos: float = 0.0
    e_contact_dir: float = 0.0
    weighted_e_contact_dir: float = 0.0

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
    temporal_scope: str = "base_and_finger"
    fixed_base_to_seed: bool = False
    fixed_qpos_to_seed: bool = False
    quality_extension: dict[str, Any] | None = None
    continuous_prediction_base: np.ndarray | None = None
    continuous_prediction_qpos: np.ndarray | None = None
    trust_region_reference: np.ndarray | None = None
    trust_region_limits: tuple[float, float, float] | None = None
    initialization_source: str = "warm_reset"
    cache: RefinementEvaluationCache = field(
        default_factory=lambda: RefinementEvaluationCache(-1, "")
    )
    timers: TimerBook = field(default_factory=TimerBook)
    full_audit_call_count: int = 0
    full_audit_call_reasons: list[str] = field(default_factory=list)
    active_query_call_count: int = 0
    spatial_gradient_backend: str = "legacy_surface_normal_optimizer_fd_v1"
    sign_cache: LipschitzSignCache | None = None
    compiled_spatial_fd_backend: CompiledSpatialFDBackend | None = None
    gradient_policy: SignedDistanceGradientAmbiguityPolicy = field(
        default_factory=SignedDistanceGradientAmbiguityPolicy
    )
    _residual_model: Any = field(default=None, init=False, repr=False)
    _surface_joint_paths: tuple[tuple[Any, ...], ...] = field(
        default_factory=tuple, init=False, repr=False
    )
    _active_query_hash: str = field(default="__unbound__", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temporal_scope not in {
            "base_and_finger",
            "finger_only",
            "base_only",
            "none",
            "continuous_full_state",
            "continuous_full_state_plus_paper",
        }:
            raise ValueError(f"unsupported temporal scope: {self.temporal_scope}")
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
            if (
                self.temporal_scope in {"continuous_full_state", "continuous_full_state_plus_paper"}
                and self.continuous_prediction_base is not None
                and self.continuous_prediction_qpos is not None
            ):
                predicted = torch.as_tensor(
                    self.continuous_prediction_base, dtype=value.dtype, device=value.device
                )
                predicted_q = torch.as_tensor(
                    self.continuous_prediction_qpos, dtype=value.dtype, device=value.device
                )
                predicted_rotation = predicted[:3, :3]
                predicted_translation = predicted[:3, 3]
                base_error_translation = (
                    predicted_rotation.T @ (base[:3, 3] - predicted_translation)
                ) / S_POS_M
                base_error_rotation = (
                    _so3_log_torch(predicted_rotation.T @ base[:3, :3]) / S_ROT_RAD
                )
                finger_error = (qpos - predicted_q) / S_Q_RAD
                e_temporal = LAMBDA_CORR * (
                    base_error_translation.square().mean()
                    + base_error_rotation.square().mean()
                    + finger_error.square().mean()
                )
                if (
                    self.temporal_scope == "continuous_full_state_plus_paper"
                    and self.previous_reference is not None
                ):
                    previous = torch.as_tensor(
                        self.previous_reference, dtype=value.dtype, device=value.device
                    )
                    current_q_delta = value[6 : 6 + self.robot_model.num_dofs]
                    previous_q_delta = previous[6 : 6 + self.robot_model.num_dofs]
                    e_temporal = (
                        e_temporal
                        + self.paper.lambda_reg
                        * (current_q_delta - previous_q_delta).square().sum()
                    )
            elif self.previous_reference is not None and self.temporal_scope != "none":
                previous = torch.as_tensor(
                    self.previous_reference, dtype=value.dtype, device=value.device
                )
                if self.temporal_scope == "finger_only":
                    current_delta = value[6 : self.variable_size_without_slack]
                    previous_delta = previous[6 : self.variable_size_without_slack]
                elif self.temporal_scope == "base_only":
                    current_delta = value[:6]
                    previous_delta = previous[:6]
                else:
                    current_delta = value[: self.variable_size_without_slack]
                    previous_delta = previous[: self.variable_size_without_slack]
                e_temporal = self.paper.lambda_reg * (current_delta - previous_delta).square().sum()
            e_base_pos = self.paper.lambda_base_pos * delta_p.square().sum()
            e_base_rot = self.paper.lambda_base_rot * delta_w.square().sum()
            e_slack = 0.5 * self.paper.w_s * slack.square().sum()
        e_morph = value.new_zeros(())
        weighted_morph = value.new_zeros(())
        e_contact_pos = value.new_zeros(())
        weighted_contact_pos = value.new_zeros(())
        e_contact_dir = value.new_zeros(())
        weighted_contact_dir = value.new_zeros(())
        extension = self.quality_extension
        if extension is not None:
            target_keypoints = extension.get("morphology_target_keypoints_scene")
            lambda_morph = float(extension.get("lambda_morph", 0.0))
            morph_scale = max(float(extension.get("morphology_scale_m", 1.0)), 1e-12)
            if target_keypoints is not None and lambda_morph > 0.0:
                target = torch.as_tensor(target_keypoints, dtype=value.dtype, device=value.device)
                e_morph = (robot_keypoints - target).square().sum(dim=-1).mean()
                e_morph = e_morph / (morph_scale * morph_scale)
                weighted_morph = lambda_morph * e_morph

            regions = extension.get("contact_regions", ())
            active = np.asarray(extension.get("contact_active", ()), dtype=bool)
            targets = extension.get("contact_target_relative")
            directions = extension.get("contact_target_direction")
            weights = np.asarray(extension.get("contact_weights", ()), dtype=np.float64)
            lambda_pos = float(extension.get("lambda_contact_pos", 0.0))
            lambda_dir = float(extension.get("lambda_contact_dir", 0.0))
            region_ids = tuple(
                str(item)
                for item in extension.get(
                    "contact_region_ids", [item.get("region_id") for item in regions]
                )
            )
            regions_by_id = {str(item["region_id"]): item for item in regions}
            if (
                regions
                and targets is not None
                and len(active) == len(region_ids)
                and len(targets) == len(region_ids)
            ):
                fk = self.robot_model.forward_kinematics_base(qpos)
                object_rotation = torch.as_tensor(
                    self.object_pose_scene[:3, :3], dtype=value.dtype, device=value.device
                )
                object_translation = torch.as_tensor(
                    self.object_pose_scene[:3, 3], dtype=value.dtype, device=value.device
                )
                position_terms: list[Any] = []
                direction_terms: list[Any] = []
                position_weights: list[float] = []
                direction_weights: list[float] = []
                target_relative = torch.as_tensor(targets, dtype=value.dtype, device=value.device)
                target_direction = (
                    None
                    if directions is None
                    else torch.as_tensor(directions, dtype=value.dtype, device=value.device)
                )
                for region_index, region_id in enumerate(region_ids):
                    if not active[region_index]:
                        continue
                    region = regions_by_id[region_id]
                    points = torch.as_tensor(
                        region["points_link"], dtype=value.dtype, device=value.device
                    )
                    local_transform = torch.as_tensor(
                        region["local_transform"], dtype=value.dtype, device=value.device
                    )
                    link_transform = fk[str(region["link"])]
                    points = points @ local_transform[:3, :3].transpose(-1, -2)
                    points = points + local_transform[:3, 3]
                    points = points @ link_transform[:3, :3].transpose(-1, -2)
                    points = points + link_transform[:3, 3]
                    points = points @ base[:3, :3].transpose(-1, -2) + base[:3, 3]
                    centroid = points.mean(dim=0)
                    relative = (centroid - object_translation) @ object_rotation
                    scale = extension.get("contact_position_scale_m", 0.01)
                    position_terms.append(
                        torch.linalg.vector_norm(
                            (relative - target_relative[region_index]) / float(scale)
                        )
                    )
                    position_weights.append(
                        float(weights[region_index]) if len(weights) == len(regions) else 1.0
                    )
                    if target_direction is not None:
                        semantic = torch.as_tensor(
                            region["semantic_direction_link"],
                            dtype=value.dtype,
                            device=value.device,
                        )
                        direction = semantic @ link_transform[:3, :3].transpose(-1, -2)
                        direction = direction @ base[:3, :3].transpose(-1, -2)
                        direction = direction @ object_rotation
                        direction = direction / torch.clamp(
                            torch.linalg.vector_norm(direction), min=1e-12
                        )
                        source_direction = target_direction[region_index]
                        dot = torch.clamp((direction * source_direction).sum(), -1.0, 1.0)
                        direction_terms.append(1.0 - dot)
                        direction_weights.append(
                            float(weights[region_index]) if len(weights) == len(regions) else 1.0
                        )
                if position_terms:
                    position_value = torch.stack(position_terms)
                    # Huber delta is fixed at 1.0 in normalized 10 mm units.
                    huber_value = torch.where(
                        position_value.abs() <= 1.0,
                        0.5 * position_value.square(),
                        position_value.abs() - 0.5,
                    )
                    weight_value = torch.as_tensor(
                        position_weights, dtype=value.dtype, device=value.device
                    )
                    e_contact_pos = (huber_value * weight_value).sum() / torch.clamp(
                        weight_value.sum(), min=1e-12
                    )
                    weighted_contact_pos = lambda_pos * e_contact_pos
                if direction_terms:
                    direction_value = torch.stack(direction_terms)
                    weight_value = torch.as_tensor(
                        direction_weights, dtype=value.dtype, device=value.device
                    )
                    e_contact_dir = (direction_value * weight_value).sum() / torch.clamp(
                        weight_value.sum(), min=1e-12
                    )
                    weighted_contact_dir = lambda_dir * e_contact_dir
        weighted_im = self.paper.lambda_im * e_im
        weighted_bone = self.paper.lambda_bone * e_bone
        total = (
            weighted_im
            + weighted_bone
            + e_temporal
            + e_base_pos
            + e_base_rot
            + e_slack
            + weighted_morph
            + weighted_contact_pos
            + weighted_contact_dir
        )
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
            e_morph=float(e_morph.detach().cpu()),
            weighted_e_morph=float(weighted_morph.detach().cpu()),
            e_contact_pos=float(e_contact_pos.detach().cpu()),
            weighted_e_contact_pos=float(weighted_contact_pos.detach().cpu()),
            e_contact_dir=float(e_contact_dir.detach().cpu()),
            weighted_e_contact_dir=float(weighted_contact_dir.detach().cpu()),
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

    def collision_points_jacobian_numpy(
        self, value: np.ndarray, *, point_ids: np.ndarray | None = None
    ) -> np.ndarray:
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
        if point_ids is None:
            selected = np.arange(self.surface.count, dtype=np.int64)
        else:
            selected = np.asarray(point_ids, dtype=np.int64).reshape(-1)
            if (
                np.any(selected < 0)
                or np.any(selected >= self.surface.count)
                or len(np.unique(selected)) != len(selected)
            ):
                raise ValueError("collision Jacobian point IDs must be unique surface indices")
        _, _, qpos, _ = self.unpack(current)
        qpos = np.asarray(qpos, dtype=np.float64)
        fk = self.robot_model.forward_kinematics_reference(qpos)
        points_base = np.empty((len(selected), 3), dtype=np.float64)
        qpos_jacobian_base = np.zeros(
            (len(selected), 3, self.robot_model.num_dofs), dtype=np.float64
        )
        for geometry_index, start, stop in self.geometry_slices:
            selected_rows = np.flatnonzero((selected >= start) & (selected < stop))
            if not len(selected_rows):
                continue
            local_indices = selected[selected_rows] - start
            local = np.asarray(self.surface_points_local[start:stop], dtype=np.float64)[
                local_indices
            ]
            local_transform = np.asarray(
                self.surface_local_transforms[geometry_index], dtype=np.float64
            )
            local_points = local @ local_transform[:3, :3].T + local_transform[:3, 3]
            link_name = self.surface_link_names[geometry_index]
            link = np.asarray(fk[link_name], dtype=np.float64)
            points = local_points @ link[:3, :3].T + link[:3, 3]
            points_base[selected_rows] = points
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
                qpos_jacobian_base[selected_rows, :, q_index] += derivative

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
        result = np.zeros((len(selected), 3, self.variable_size_without_slack), dtype=np.float64)
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
        points = self.candidate_points(current)[query_ids]
        with self.timers.measure("solver_sdf"):
            if self.sign_cache is not None and isinstance(
                self.sdf, (HybridSignedDistanceBackend, CompiledSpatialFDBackend)
            ):
                result = self.sdf.query_scene(
                    points,
                    self.object_pose_scene,
                    sample_ids=query_ids,
                    sign_cache=self.sign_cache,
                    evaluation_lineage=f"frame={self.frame_id};query={active_hash}",
                )
            else:
                result = self.sdf.query_scene(points, self.object_pose_scene)
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
            jac_np = self.collision_points_jacobian_numpy(current, point_ids=query_ids)
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
        spatial_fd_probe_count = 0
        last_resort_count = 0
        gradient_diagnostics: dict[str, Any] = {}
        if self.spatial_gradient_backend == "spatial_gradient_chain_rule_v1":
            points = self.candidate_points(current, active_hash)[query_ids]
            gradient = analytic_spatial_gradient(points, result, policy=self.gradient_policy)
            valid = gradient.analytic_mask
            values[valid, :n] = np.einsum(
                "ri,rij->rj", gradient.spatial_gradient_scene[valid], jac_np[valid, :, :n]
            )
            invalid_rows = np.flatnonzero(gradient.spatial_fd_mask)
            fallback_count = int(len(invalid_rows))
            gradient_diagnostics = {
                "signed_distance_gradient": "spatial_gradient_chain_rule_v1",
                "analytic_point_count": int(np.count_nonzero(gradient.analytic_mask)),
                "ambiguous_point_count": fallback_count,
                "fallback_point_count": fallback_count,
                "fallback_ratio": float(fallback_count / len(query_ids)) if len(query_ids) else 0.0,
                "ambiguity_reason_counts": ambiguity_reason_counts(gradient),
            }
        else:
            invalid_rows = np.flatnonzero(~valid)
            for row, sample_valid in enumerate(valid):
                if sample_valid:
                    values[row, :n] = normals[row] @ jac_np[row, :, :n]
                else:
                    fallback_count += 1
        if len(invalid_rows):
            if self.spatial_gradient_backend == "spatial_gradient_chain_rule_v1":
                probe_points = self.candidate_points(current, active_hash)[query_ids[invalid_rows]]
                h = self.gradient_policy.spatial_fd_step_m
                with self.timers.measure("spatial_fd_fallback"):
                    if self.compiled_spatial_fd_backend is not None:
                        with self.timers.measure("compiled_kernel"):
                            compiled = self.compiled_spatial_fd_backend.spatial_fd_gradient_scene(
                                np.ascontiguousarray(probe_points, dtype=np.float64),
                                self.object_pose_scene,
                                h,
                            )
                        probe_result = compiled.probe_result
                        spatial = compiled.gradient_scene
                        if self.compiled_spatial_fd_backend.compiled_winding:
                            stats: dict[str, Any] = dict(
                                self.compiled_spatial_fd_backend.probe_sign_stats
                            )
                            stats["reuse_rate"] = (
                                float(stats["certified_probe_reuse"] / stats["total_fd_probes"])
                                if stats["total_fd_probes"]
                                else 0.0
                            )
                            gradient_diagnostics["compiled_exact_sign"] = stats
                    else:
                        axes = np.eye(3, dtype=np.float64)
                        probes = np.concatenate(
                            [probe_points + h * axis for axis in axes]
                            + [probe_points - h * axis for axis in axes],
                            axis=0,
                        )
                        with self.timers.measure("solver_sdf"):
                            if isinstance(self.sdf, HybridSignedDistanceBackend):
                                probe_result = self.sdf.query_scene(
                                    probes,
                                    self.object_pose_scene,
                                    cache_update=False,
                                    evaluation_lineage="spatial_fd_probe",
                                )
                            else:
                                probe_result = self.sdf.query_scene(probes, self.object_pose_scene)
                        probe_phi = np.asarray(
                            probe_result.signed_distance, dtype=np.float64
                        ).reshape(6, -1)
                        spatial = ((probe_phi[:3] - probe_phi[3:]) / (2.0 * h)).T
                if not np.all(probe_result.sign_valid) or not np.all(probe_result.valid):
                    raise ValueError("invalid signed-distance result entered spatial-FD solve")
                spatial_fd_probe_count = int(6 * len(invalid_rows))
                norms = np.linalg.norm(spatial, axis=1)
                bad = (~np.isfinite(spatial).all(axis=1)) | (norms < 0.1) | (norms > 10.0)
                if np.any(bad):
                    # This result remains explicitly fail-closed in the P2
                    # qualification report; never silently use a normal.
                    spatial[bad] = normals[invalid_rows[bad]]
                    last_resort_count = int(np.count_nonzero(bad))
                values[invalid_rows, :n] = np.einsum(
                    "ri,rij->rj", spatial, jac_np[invalid_rows, :, :n]
                )
            else:
                invalid_query_ids = np.asarray(query_ids[invalid_rows], dtype=np.int64)

                def fd_residual_batch(probes: np.ndarray) -> np.ndarray:
                    point_blocks = [
                        self.candidate_points(probe, "__fd_probe__")[invalid_query_ids]
                        for probe in np.asarray(probes, dtype=np.float64)
                    ]
                    points = np.concatenate(point_blocks, axis=0)
                    with self.timers.measure("solver_sdf"):
                        fd_result = self.sdf.query_scene(points, self.object_pose_scene)
                    if not np.all(fd_result.sign_valid) or not np.all(fd_result.valid):
                        raise ValueError(
                            "invalid signed-distance result entered finite-difference solve"
                        )
                    return np.asarray(fd_result.signed_distance, dtype=np.float64).reshape(
                        len(point_blocks), len(invalid_query_ids)
                    )

                values[invalid_rows, :n], _ = _vectorized_constraint_finite_difference(
                    fd_residual_batch,
                    current,
                    variable_count=n,
                    row_ids=np.arange(len(invalid_rows), dtype=np.int64),
                    epsilon=eps,
                )
                self.cache.prepare(current, active_hash)
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
            "spatial_fd_probe_count": spatial_fd_probe_count,
            "sdf_batches_for_jacobian": int(1 + (1 if spatial_fd_probe_count else 0)),
            "surface_normal_last_resort_count": last_resort_count,
            "ambiguity_fd_backend": (
                "compiled_spatial_central_fd_v1"
                if self.compiled_spatial_fd_backend is not None
                else "fast_exact_v2_python"
            ),
            **gradient_diagnostics,
        }
        if self.sign_cache is not None:
            diagnostics["sign_cache"] = self.sign_cache.as_dict()
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
    single_frame_feasible: bool = False
    trajectory_continuous: bool = True
    continuity_metrics: dict[str, Any] = field(default_factory=dict)
    initialization_source: str = "warm_reset"
    retry_attempt: int = 0
    retry_profile: str = "none"
    window_used: bool = False


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
    analytic_total = 0
    ambiguous_total = 0
    spatial_fd_probes_total = 0
    spatial_sdf_batches_total = 0
    surface_normal_last_resort_total = 0
    ambiguity_totals: dict[str, int] = {}
    sign_cache_stats: dict[str, Any] = {}
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
            return normalized_gradient(context.objective(physical(value), query_set.query_hash)[1])

    def constraint_jac(value: np.ndarray) -> np.ndarray:
        nonlocal jacobian_calls, fallback_total, analytic_total, ambiguous_total
        nonlocal spatial_fd_probes_total, spatial_sdf_batches_total
        nonlocal surface_normal_last_resort_total
        nonlocal sign_cache_stats
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
        analytic_total += int(diagnostics.get("analytic_point_count", 0))
        ambiguous_total += int(diagnostics.get("ambiguous_point_count", 0))
        spatial_fd_probes_total += int(diagnostics.get("spatial_fd_probe_count", 0))
        spatial_sdf_batches_total += int(diagnostics.get("sdf_batches_for_jacobian", 0))
        surface_normal_last_resort_total += int(
            diagnostics.get("surface_normal_last_resort_count", 0)
        )
        for key, count in diagnostics.get("ambiguity_reason_counts", {}).items():
            ambiguity_totals[str(key)] = ambiguity_totals.get(str(key), 0) + int(count)
        sign_cache_stats = dict(diagnostics.get("sign_cache", sign_cache_stats))
        return np.asarray(jac, dtype=np.float64) * variable_scales[None, :]

    def callback(value: np.ndarray) -> None:
        callback_iterates.append(physical(value).copy())

    constraints: list[dict[str, Any]] = [{"type": "ineq", "fun": constraint, "jac": constraint_jac}]
    if context.trust_region_reference is not None:
        if context.trust_region_limits is None:
            raise ValueError("trust-region reference requires trust-region limits")
        reference = np.asarray(context.trust_region_reference, dtype=np.float64)
        position_limit, rotation_limit, q_limit = context.trust_region_limits

        def trust_region_constraint(value: np.ndarray) -> np.ndarray:
            physical_value = physical(value)
            delta = physical_value[: 6 + context.robot_model.num_dofs]
            return np.concatenate(
                [
                    position_limit - np.abs(delta[:3] - reference[:3]),
                    rotation_limit - np.abs(delta[3:6] - reference[3:6]),
                    q_limit
                    - np.abs(
                        delta[6 : 6 + context.robot_model.num_dofs]
                        - reference[6 : 6 + context.robot_model.num_dofs]
                    ),
                ]
            )

        constraints.append({"type": "ineq", "fun": trust_region_constraint})

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
    if context.fixed_base_to_seed:
        lower_physical[:6] = 0.0
        upper_physical[:6] = 0.0
    if context.fixed_qpos_to_seed:
        start = 6
        stop = start + context.robot_model.num_dofs
        lower_physical[start:stop] = context.seed_qpos
        upper_physical[start:stop] = context.seed_qpos
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
            constraints=constraints,
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
        "analytic_point_count": analytic_total,
        "ambiguous_point_count": ambiguous_total,
        "spatial_fd_probe_count": spatial_fd_probes_total,
        "sdf_batches_for_jacobian": spatial_sdf_batches_total,
        "surface_normal_last_resort_count": surface_normal_last_resort_total,
        "ambiguity_reason_counts": dict(sorted(ambiguity_totals.items())),
        "sign_cache": sign_cache_stats,
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
    if acceptance_policy_id not in {
        STRICT_ACCEPTANCE_POLICY_ID,
        "strict_optimizer_converged_audits_and_continuity_v1",
    }:
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
    initial_state_without_slack: np.ndarray | None = None,
    initialization_source: str = "warm_reset",
    retry_attempt: int = 0,
    retry_profile: str = "none",
) -> FinalFrameResult:
    started = time.perf_counter()
    warm_value_without = np.concatenate([np.zeros(6), context.seed_qpos])
    warm_slack = np.clip(
        np.maximum(-context.paper.tau - query_set.initial_signed_distance, 0.0),
        0.0,
        context.paper.b - context.paper.tau,
    )
    initial_value_without = warm_value_without
    if initial_state_without_slack is not None:
        initial_value_without = np.asarray(initial_state_without_slack, dtype=np.float64).reshape(
            -1
        )
        if len(initial_value_without) != context.variable_size_without_slack:
            raise ValueError("initial continuous state has the wrong dimension")
    context.initialization_source = initialization_source
    initial = np.concatenate([initial_value_without, warm_slack])
    query_rounds = 0
    result: Any = None
    full: SignedDistanceQueryResult | None = None
    diagnostics: dict[str, Any] = {}
    continuation_trace: list[dict[str, Any]] = []
    solver_attempt_trace: list[dict[str, Any]] = []
    active_set_converged = False
    discovery_audit_count = 0
    final_full_audit_count = 0
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
                recovery_diag[key] = int(primary_diag.get(key, 0)) + int(recovery_diag.get(key, 0))
            recovery_diag["initial_objective"] = primary_diag["initial_objective"]
            recovery_diag["final_objective_change"] = float(
                primary_diag["initial_objective"] - recovery_diag["final_objective"]
            )
            recovery_diag["primary_solver_status"] = int(getattr(primary_result, "status", -1))
            recovery_diag["primary_solver_message"] = str(primary_result.message)
            recovery_diag["primary_optimizer_iterations"] = int(getattr(primary_result, "nit", 0))
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
            solve_diag["primary_optimizer_iterations"] = int(getattr(primary_result, "nit", 0))
            solve_diag["primary_point_jacobian_backend"] = point_jacobian_backend
            solve_diag["solver_recovery"] = "none"
            solve_diag["solver_retry_count"] = 0
        solver_attempt_trace.append(attempt)
        diagnostics.update(solve_diag)
        independent = _independent_constraints(context, result.x, query_set)
        # Active-set discovery is distinct from final acceptance.  For the
        # compiled exact spatial backend, it may use the solver's exact
        # closest-point/winding query to discover candidates; final acceptance
        # below still performs exactly one independent reference-SDF audit.
        # This avoids paying the expensive reference full-surface scan twice
        # per frame without weakening the fail-closed acceptance gate.
        discovery_sdf = (
            context.sdf
            if isinstance(context.sdf, CompiledSpatialFDBackend)
            else context.reference_sdf
        )
        discovery_backend_id = str(discovery_sdf.backend_id)
        with context.timers.measure("active_set_discovery"):
            full = discovery_sdf.query_scene(
                context.candidate_points(result.x, query_set.query_hash),
                context.object_pose_scene,
            )
        discovery_audit_count += 1
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
                feasibility_buffer_m=ACTIVE_SET_CONTINUATION_FEASIBILITY_BUFFER_M,
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
    with context.timers.measure("final_full_audit"):
        with context.timers.measure("full_512_audit"):
            full = context.reference_sdf.query_scene(
                context.candidate_points(value, query_set.query_hash),
                context.object_pose_scene,
            )
    final_full_audit_count += 1
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
    diagnostics["full_audit_call_count"] = final_full_audit_count
    diagnostics["full_audit_call_reasons"] = ["frame_final_independent_acceptance"]
    diagnostics["active_set_discovery_audit_count"] = discovery_audit_count
    diagnostics["active_set_discovery_backend_id"] = discovery_backend_id
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
        single_frame_feasible=accepted,
        trajectory_continuous=True,
        initialization_source=initialization_source,
        retry_attempt=int(retry_attempt),
        retry_profile=retry_profile,
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
    *,
    temporal_scope: str = "base_and_finger",
    fixed_base_to_seed: bool = False,
    fixed_qpos_to_seed: bool = False,
    quality_extension: dict[str, Any] | None = None,
    continuous_prediction_base: np.ndarray | None = None,
    continuous_prediction_qpos: np.ndarray | None = None,
    spatial_gradient_backend: str = "legacy_surface_normal_optimizer_fd_v1",
    sign_cache: LipschitzSignCache | None = None,
    compiled_spatial_fd_backend: CompiledSpatialFDBackend | None = None,
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
            "quality_extension": None
            if quality_extension is None
            else {
                "profile_id": quality_extension.get("profile_id"),
                "lambda_morph": quality_extension.get("lambda_morph", 0.0),
                "lambda_contact_pos": quality_extension.get("lambda_contact_pos", 0.0),
                "lambda_contact_dir": quality_extension.get("lambda_contact_dir", 0.0),
            },
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
        temporal_scope=temporal_scope,
        fixed_base_to_seed=fixed_base_to_seed,
        fixed_qpos_to_seed=fixed_qpos_to_seed,
        quality_extension=quality_extension,
        continuous_prediction_base=continuous_prediction_base,
        continuous_prediction_qpos=continuous_prediction_qpos,
        cache=RefinementEvaluationCache(global_frame, context_hash),
        spatial_gradient_backend=spatial_gradient_backend,
        sign_cache=sign_cache,
        compiled_spatial_fd_backend=compiled_spatial_fd_backend,
    )


def _quality_extension_for_frame(
    quality_extension: dict[str, Any] | None, local_index: int
) -> dict[str, Any] | None:
    """Slice frame-varying quality targets without mutating the shared spec."""

    if quality_extension is None:
        return None
    result = dict(quality_extension)
    for key in (
        "morphology_target_keypoints_scene",
        "contact_target_relative",
        "contact_target_direction",
        "contact_active",
        "contact_weights",
    ):
        value = result.get(key)
        if value is not None:
            array = np.asarray(value)
            if array.ndim > 0 and array.shape[0] > local_index:
                result[key] = array[local_index]
    return result


def _apply_continuity_gate(
    frame: FinalFrameResult,
    context: _FrameContext,
    *,
    frame_id: int,
    retry_attempt: int,
    retry_profile: str,
    window_used: bool = False,
) -> FinalFrameResult:
    """Separate frame feasibility from trajectory continuity for v3 artifacts."""

    frame.single_frame_feasible = bool(frame.accepted)
    metrics: dict[str, Any]
    if context.continuous_prediction_base is None or context.continuous_prediction_qpos is None:
        metrics = {
            "schema_version": "toporetarget.trajectory_continuity.v1",
            "frame": int(frame_id),
            "trajectory_continuous": True,
            "continuity_failure_reasons": [],
            "frame_zero_gate": True,
        }
        continuous = True
    else:
        predicted_keypoints = np.asarray(
            context.robot_model.keypoints_scene(
                context.continuous_prediction_qpos,
                context.continuous_prediction_base,
            ),
            dtype=np.float64,
        )
        final_keypoints = np.asarray(
            context.robot_model.keypoints_scene(frame.qpos, frame.base_pose_scene),
            dtype=np.float64,
        )
        metrics = continuity_metrics(
            context.continuous_prediction_base,
            frame.base_pose_scene,
            context.continuous_prediction_qpos,
            frame.qpos,
            predicted_keypoints_scene=predicted_keypoints,
            final_keypoints_scene=final_keypoints,
            frame=frame_id,
        )
        continuous = bool(metrics["trajectory_continuous"])
    frame.trajectory_continuous = continuous
    frame.continuity_metrics = metrics
    frame.retry_attempt = int(retry_attempt)
    frame.retry_profile = retry_profile
    frame.window_used = bool(window_used)
    frame.accepted = bool(frame.single_frame_feasible and continuous)
    frame.solver_success = bool(frame.accepted)
    if not frame.accepted:
        frame.failure = frame.failure or "continuity gate failed: " + ",".join(
            str(item) for item in metrics.get("continuity_failure_reasons", [])
        )
    frame.jacobian_diagnostics["continuity"] = metrics
    return frame


def _initial_query_set_for_context(
    context: _FrameContext,
    state_without_slack: np.ndarray,
    query_profile: CollisionQueryProfile,
) -> CollisionQuerySet:
    """Build and independently audit one window frame's initial QuerySet."""

    initial_value = np.concatenate(
        [np.asarray(state_without_slack, dtype=np.float64), np.zeros(0, dtype=np.float64)]
    )
    initial_points = context.candidate_points(initial_value)
    with context.timers.measure("full_512_audit"):
        initial_full = context.reference_sdf.query_scene(initial_points, context.object_pose_scene)
    if not np.all(initial_full.sign_valid):
        raise ValueError("window initial full-surface audit received invalid signed distance")
    context.full_audit_call_count = 1
    context.full_audit_call_reasons = ["window_frame_query_set_initialization"]
    initial_query = context.sdf.query_scene(initial_points, context.object_pose_scene)
    return build_query_set(
        initial_query.signed_distance,
        context.surface.geometry_ids,
        query_profile,
    )


def _window_joint_objective(
    entries: list[tuple[_FrameContext, CollisionQuerySet, np.ndarray]],
    solver: RefinementSolverProfile,
    *,
    point_jacobian_backend: str,
    left_anchor: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Solve a bounded five-frame correction window in one SLSQP problem.

    The formal single-frame solver remains the source of the persisted center
    result.  This helper is only the deterministic Attempt-3 fallback: it
    solves the center and future variables jointly, validates every frame's
    active constraints through its own QuerySet, and returns future states as
    hints.  No future final artifact is treated as ground truth.
    """

    from scipy.optimize import minimize

    if not entries or len(entries) > 4:
        raise ValueError("a five-frame window has one fixed left anchor and at most four variables")

    def scales(context: _FrameContext, query_set: CollisionQuerySet) -> np.ndarray:
        return np.concatenate(
            [
                np.full(3, 0.010, dtype=np.float64),
                np.full(3, 0.1, dtype=np.float64),
                np.full(context.robot_model.num_dofs, 0.050, dtype=np.float64),
                np.full(query_set.count, 0.001, dtype=np.float64),
            ]
        )

    offsets = [0]
    scale_rows: list[np.ndarray] = []
    initial_rows: list[np.ndarray] = []
    bounds: list[tuple[float, float]] = []
    for context, query_set, state_without_slack in entries:
        expected = context.variable_size_without_slack
        state = np.asarray(state_without_slack, dtype=np.float64).reshape(-1)
        if len(state) != expected:
            raise ValueError("window initialization has the wrong state dimension")
        warm_slack = np.clip(
            np.maximum(-context.paper.tau - query_set.initial_signed_distance, 0.0),
            0.0,
            context.paper.b - context.paper.tau,
        )
        scale = scales(context, query_set)
        scale_rows.append(scale)
        initial_rows.append(np.concatenate([state, warm_slack]))
        offsets.append(offsets[-1] + len(scale))
        bounds.extend(
            list(
                zip(
                    np.concatenate(
                        [
                            np.full(6, -np.inf),
                            np.asarray(context.robot_model.joint_lower),
                            np.zeros(query_set.count),
                        ]
                    )
                    / scale,
                    np.concatenate(
                        [
                            np.full(6, np.inf),
                            np.asarray(context.robot_model.joint_upper),
                            np.full(query_set.count, context.paper.b - context.paper.tau),
                        ]
                    )
                    / scale,
                    strict=True,
                )
            )
        )

    initial = np.concatenate(
        [row / scale for row, scale in zip(initial_rows, scale_rows, strict=True)]
    )

    def split(value: np.ndarray) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for index, (_context, _query_set, _) in enumerate(entries):
            start, stop = offsets[index], offsets[index + 1]
            result.append(np.asarray(value[start:stop], dtype=np.float64) * scale_rows[index])
        return result

    def pose_from_state(context: _FrameContext, state: np.ndarray) -> np.ndarray:
        import torch

        return _as_np(context.base_pose_torch(torch.as_tensor(state, dtype=torch.float64)))

    def objective(value: np.ndarray) -> float:
        rows = split(value)
        total = 0.0
        poses: list[np.ndarray] = []
        qposes: list[np.ndarray] = []
        for (context, query_set, _), row in zip(entries, rows, strict=True):
            total += float(context.objective(row, query_set.query_hash)[0])
            poses.append(pose_from_state(context, row))
            qposes.append(np.asarray(row[6 : 6 + context.robot_model.num_dofs]))
        if left_anchor is not None:
            total += correction_temporal_energy(
                np.asarray(left_anchor[0], dtype=np.float64),
                poses[0],
                np.asarray(left_anchor[1], dtype=np.float64),
                qposes[0],
            )
        for previous, current in zip(range(len(entries) - 1), range(1, len(entries)), strict=True):
            total += correction_temporal_energy(
                poses[previous], poses[current], qposes[previous], qposes[current]
            )
        return float(total)

    def objective_jac(value: np.ndarray) -> np.ndarray:
        """Return the exact joint objective gradient in normalized coordinates.

        A joint SLSQP problem has one objective over all window frames.  With
        no callback supplied, SciPy finite-differences that objective across
        every state variable, which is needlessly expensive because each
        frame objective already exposes an autograd gradient.  Reuse those
        gradients and differentiate only the cheap inter-frame correction
        energy with Torch.  The objective value itself is unchanged.
        """

        import torch

        rows = split(value)
        normalized_gradient = np.zeros(len(value), dtype=np.float64)
        row_tensors: list[Any] = []
        poses: list[Any] = []
        for index, ((context, query_set, _), row) in enumerate(zip(entries, rows, strict=True)):
            _total, frame_gradient, _breakdown = context.objective(row, query_set.query_hash)
            start, stop = offsets[index], offsets[index + 1]
            normalized_gradient[start:stop] += np.asarray(frame_gradient) * scale_rows[index]
            tensor = torch.as_tensor(row, dtype=torch.float64).requires_grad_(True)
            row_tensors.append(tensor)
            poses.append(context.base_pose_torch(tensor))

        temporal_total = None
        temporal_pairs: list[tuple[Any, Any, Any, Any]] = []
        if left_anchor is not None:
            anchor_pose = torch.as_tensor(
                np.asarray(left_anchor[0], dtype=np.float64), dtype=torch.float64
            )
            anchor_q = torch.as_tensor(
                np.asarray(left_anchor[1], dtype=np.float64), dtype=torch.float64
            )
            temporal_pairs.append(
                (
                    anchor_pose,
                    anchor_q,
                    poses[0],
                    row_tensors[0][6 : 6 + entries[0][0].robot_model.num_dofs],
                )
            )
        temporal_pairs.extend(
            (
                poses[previous],
                row_tensors[previous][6 : 6 + entries[previous][0].robot_model.num_dofs],
                poses[current],
                row_tensors[current][6 : 6 + entries[current][0].robot_model.num_dofs],
            )
            for previous, current in zip(range(len(rows) - 1), range(1, len(rows)), strict=True)
        )
        for previous_pose, previous_q, current_pose, current_q_tensor in temporal_pairs:
            previous_rotation = previous_pose[:3, :3]
            current_rotation = current_pose[:3, :3]
            relative_rotation = previous_rotation.T @ current_rotation
            relative_translation = previous_rotation.T @ (
                current_pose[:3, 3] - previous_pose[:3, 3]
            )
            base_translation = relative_translation / S_POS_M
            base_rotation = _so3_log_torch(relative_rotation) / S_ROT_RAD
            finger = (current_q_tensor - previous_q) / S_Q_RAD
            pair = LAMBDA_CORR * (
                base_translation.square().mean()
                + base_rotation.square().mean()
                + finger.square().mean()
            )
            temporal_total = pair if temporal_total is None else temporal_total + pair
        if temporal_total is not None:
            temporal_gradients = torch.autograd.grad(temporal_total, row_tensors)
            for index, gradient in enumerate(temporal_gradients):
                start, stop = offsets[index], offsets[index + 1]
                normalized_gradient[start:stop] += (
                    gradient.detach().cpu().numpy() * scale_rows[index]
                )
        return normalized_gradient

    constraints: list[dict[str, Any]] = []
    for index, (context, query_set, _) in enumerate(entries):
        start, stop = offsets[index], offsets[index + 1]

        def constraint(
            value: np.ndarray,
            *,
            start=start,
            stop=stop,
            context=context,
            query_set=query_set,
            scale=scale_rows[index],
        ) -> np.ndarray:
            row = np.asarray(value[start:stop], dtype=np.float64) * scale
            return context.constraint_values(row, query_set.sample_ids, query_set.query_hash)

        def constraint_jac(
            value: np.ndarray,
            *,
            start=start,
            stop=stop,
            context=context,
            query_set=query_set,
            scale=scale_rows[index],
        ) -> np.ndarray:
            """Analytic per-frame Jacobian in the joint normalized space.

            The original window helper left SLSQP to finite-difference every
            collision constraint across the concatenated four-frame state.
            That made one shadow solve spend hours evaluating the triangle
            tree.  The single-frame solver already exposes the exact same
            audited Jacobian; embed its block and apply the diagonal
            normalization here.  This changes no objective, constraints,
            QuerySet, or solver budget.
            """

            row = np.asarray(value[start:stop], dtype=np.float64) * scale
            local_jac, _diagnostics = context.constraint_jacobian(
                row,
                query_set.sample_ids,
                solver.finite_difference_epsilon,
                query_set.query_hash,
                backend=point_jacobian_backend,
            )
            block = np.asarray(local_jac, dtype=np.float64) * scale[None, :]
            global_jac = np.zeros((block.shape[0], len(value)), dtype=np.float64)
            global_jac[:, start:stop] = block
            return global_jac

        constraints.append({"type": "ineq", "fun": constraint, "jac": constraint_jac})

    slsqp_result = minimize(
        objective,
        initial,
        method=solver.method,
        jac=objective_jac,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": solver.maxiter, "ftol": solver.ftol, "disp": solver.disp},
    )
    result = slsqp_result
    backend = "SLSQP"
    trust_result: Any | None = None
    trust_error: str | None = None
    if not bool(slsqp_result.success):
        # This fallback is strictly window-local.  It receives the same
        # normalized objective, bounds, QuerySets, and constraint functions;
        # it does not alter the formal single-frame mathematics.
        from scipy.optimize import Bounds, NonlinearConstraint

        lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
        upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
        nonlinear = [
            NonlinearConstraint(
                item["fun"],
                0.0,
                np.inf,
                jac=item["jac"],
            )
            for item in constraints
        ]
        try:
            trust_result = minimize(
                objective,
                np.asarray(slsqp_result.x, dtype=np.float64),
                method="trust-constr",
                jac=objective_jac,
                bounds=Bounds(lower, upper),
                constraints=nonlinear,
                options={
                    "maxiter": int(solver.maxiter),
                    "gtol": 1.0e-8,
                    "xtol": 1.0e-8,
                    "verbose": 0,
                },
            )
            result = trust_result
            backend = "trust-constr"
        except Exception as exc:  # pragma: no cover - SciPy backend dependent
            trust_error = f"{type(exc).__name__}: {exc}"
    rows = split(np.asarray(result.x, dtype=np.float64))
    full_surface_audits: list[dict[str, Any]] = []
    for (context, _, _), row in zip(entries, rows, strict=True):
        full = context.reference_sdf.query_scene(
            context.candidate_points(row), context.object_pose_scene
        )
        hard = np.asarray(full.signed_distance, dtype=np.float64) + context.paper.b
        soft = np.asarray(full.signed_distance, dtype=np.float64) + context.paper.tau
        full_surface_audits.append(
            {
                "frame": int(context.frame_id),
                "sign_valid": bool(np.all(full.sign_valid)),
                "hard_pass": bool(np.all(hard >= -1e-6)),
                "soft_pass": bool(np.all(soft >= -1e-6)),
                "min_hard_residual": float(np.min(hard)),
                "min_soft_residual": float(np.min(soft)),
            }
        )
    return {
        "success": bool(result.success)
        and all(
            item["sign_valid"] and item["hard_pass"] and item["soft_pass"]
            for item in full_surface_audits
        ),
        "status": int(getattr(result, "status", -1)),
        "message": str(result.message),
        "backend": backend,
        "scaled_coordinates": True,
        "coordinate_scales": [scale.tolist() for scale in scale_rows],
        "left_anchor_used": left_anchor is not None,
        "slsqp": {
            "success": bool(slsqp_result.success),
            "status": int(getattr(slsqp_result, "status", -1)),
            "message": str(slsqp_result.message),
        },
        "trust_constr": None
        if trust_result is None
        else {
            "success": bool(trust_result.success),
            "status": int(getattr(trust_result, "status", -1)),
            "message": str(trust_result.message),
        },
        "trust_constr_error": trust_error,
        "iterations": int(getattr(result, "nit", 0)),
        "objective": float(objective(np.asarray(result.x, dtype=np.float64))),
        "states": [
            row[: context.variable_size_without_slack].copy()
            for row, (context, _, _) in zip(rows, entries, strict=True)
        ],
        "full_surface_audits": full_surface_audits,
        "slacks": [
            row[context.variable_size_without_slack :].copy()
            for row, (context, _, _) in zip(rows, entries, strict=True)
        ],
    }


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
            FINAL_REFINEMENT_SCHEMA_VERSION_V3,
        }:
            raise ValueError(f"unsupported final artifact schema: {self.schema_version}")
        t = self.frame_count
        qpos_value = np.asarray(self.arrays.get("qpos"))
        if qpos_value.ndim != 2 or qpos_value.shape[0] != t or qpos_value.shape[1] <= 0:
            raise ValueError(f"qpos has invalid shape {qpos_value.shape}")
        dof_value = self.metadata.get("robot_dof_count")
        robot_dof_count = qpos_value.shape[1] if dof_value is None else int(dof_value)
        if robot_dof_count != qpos_value.shape[1]:
            raise ValueError(
                "qpos width does not match robot_dof_count: "
                f"{qpos_value.shape[1]} != {robot_dof_count}"
            )
        collision_points_value = np.asarray(self.arrays.get("collision_points_scene"))
        if (
            collision_points_value.ndim != 3
            or collision_points_value.shape[0] != t
            or collision_points_value.shape[2] != 3
            or collision_points_value.shape[1] <= 0
        ):
            raise ValueError(
                f"collision_points_scene has invalid shape {collision_points_value.shape}"
            )
        sample_value = self.metadata.get("collision_surface_sample_count")
        collision_sample_count = (
            collision_points_value.shape[1] if sample_value is None else int(sample_value)
        )
        if collision_sample_count != collision_points_value.shape[1]:
            raise ValueError(
                "collision_points_scene width does not match collision_surface_sample_count: "
                f"{collision_points_value.shape[1]} != {collision_sample_count}"
            )
        required = {
            "timestamps": (t,),
            "qpos": (t, robot_dof_count),
            "base_pose_scene": (t, 4, 4),
            "base_corrections": (t, 6),
            "robot_keypoints_base": (t, 21, 3),
            "robot_keypoints_scene": (t, 21, 3),
            "collision_points_scene": (t, collision_sample_count, 3),
            "slack_concat": (None,),
            "query_offsets": (t + 1,),
            "full_signed_distance": (t, collision_sample_count),
            "full_closest_points": (t, collision_sample_count, 3),
            "full_surface_normals": (t, collision_sample_count, 3),
            "full_hard_residual": (t, collision_sample_count),
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
        continuous_arrays = {
            "single_frame_feasible": (t,),
            "trajectory_continuous": (t,),
            "final_accepted": (t,),
            "continuity_failure_reasons": (t,),
            "initialization_source": (t,),
            "retry_attempt": (t,),
            "retry_profile": (t,),
            "window_used": (t,),
            "continuity_base_translation_m": (t,),
            "continuity_base_rotation_rad": (t,),
            "continuity_finger_inf_rad": (t,),
            "continuity_excess_keypoint_m": (t,),
            "q_clamp_count": (t,),
        }
        continuous_present = set(continuous_arrays).intersection(self.arrays)
        if self.schema_version == FINAL_REFINEMENT_SCHEMA_VERSION_V3 and continuous_present != set(
            continuous_arrays
        ):
            raise ValueError(
                "continuous final artifact is missing fields: "
                + ", ".join(sorted(set(continuous_arrays) - continuous_present))
            )
        for name in continuous_present:
            if tuple(np.asarray(self.arrays[name]).shape) != continuous_arrays[name]:
                raise ValueError(
                    f"{name} has shape {np.asarray(self.arrays[name]).shape}, "
                    f"expected {continuous_arrays[name]}"
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
    geometry_policy: dict[str, Any]
    build_counts: dict[str, int]


@dataclass
class RefinementRuntimeBackends:
    """Persistent exact-sign helpers shared by every frame of one job."""

    solver_sdf: Any
    compiled_spatial_fd_backend: CompiledSpatialFDBackend | None
    sign_cache: LipschitzSignCache | None
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
    geometry_artifact_root: str | Path | None = None,
) -> RefinementResources:
    """Build mesh/SDF resources once for a refinement run."""

    paper = PaperRefinementWeights.load()
    if object_vertices is None or object_faces is None:
        obj = sequence.rigid_object(str(graph.metadata["object_id"]))
        object_vertices = obj.mesh.vertices_local
        object_faces = obj.mesh.faces
    vertices = np.asarray(object_vertices, dtype=np.float64)
    faces = np.asarray(object_faces, dtype=np.int64)
    mesh_audit = audit_mesh(vertices, faces)
    source_path: Path | None = None
    source_file = getattr(getattr(sequence, "metadata", None), "provenance", None)
    source_file_value = getattr(source_file, "source_file", None)
    object_mesh_relative = str(
        getattr(sequence.rigid_object(str(graph.metadata["object_id"])), "metadata", {}).get(
            "source_mesh", ""
        )
    )
    if source_file_value and object_mesh_relative:
        source_path = Path(str(source_file_value)).resolve().parents[2] / object_mesh_relative
    geometry_policy = ObjectSDFGeometryPolicy.load()
    geometry_output = None
    if geometry_artifact_root is not None:
        geometry_output = Path(geometry_artifact_root).expanduser() / mesh_audit.mesh_hash
    reference_sdf, geometry = build_hybrid_signed_distance_backend(
        vertices,
        faces,
        policy=geometry_policy,
        source_path=source_path,
        artifact_root=geometry_output,
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
        geometry_policy=geometry.compact_dict(
            artifact_root=None
            if geometry_output is None
            else str(Path(geometry_output).expanduser().resolve())
        ),
        build_counts={
            "mesh_load_count": 1,
            "solver_sdf_build_count": 1,
            "reference_sdf_build_count": 1,
            "convex_hull_build_count": int(
                getattr(sdf, "backend_id", "") == "convex_hull_exact_solver_only"
            ),
            "derived_proxy_build_count": 1,
            "bvh_build_count": 1,
        },
    )


def prepare_refinement_runtime_backends(
    resources: RefinementResources, execution_profile: Any | None
) -> RefinementRuntimeBackends:
    """Create immutable compiled handles and certified cache once per job."""

    ambiguity_fd_backend = str(
        getattr(execution_profile, "ambiguity_fd_backend", "fast_exact_v2_python")
    )
    sign_backend = str(getattr(execution_profile, "sign_backend", "exact_winding_per_query_v1"))
    leaf_size = int(getattr(execution_profile, "sdf_tree_leaf_size", 32))
    report = dict(resources.sdf_report)
    compiled: CompiledSpatialFDBackend | None = None
    if ambiguity_fd_backend in {
        "compiled_spatial_central_fd_v1",
        "compiled_spatial_central_fd_winding_v1",
    }:
        if not isinstance(resources.reference_sdf, HybridSignedDistanceBackend):
            raise ValueError("compiled spatial-FD backend requires the hybrid reference SDF")
        try:
            compiled = CompiledSpatialFDBackend(
                resources.reference_sdf,
                leaf_size=leaf_size,
                compiled_winding=(ambiguity_fd_backend == "compiled_spatial_central_fd_winding_v1"),
            )
        except (CompiledSDFUnavailable, ImportError, OSError) as exc:
            report["compiled_spatial_fd_fallback"] = str(exc)
    sign_cache = (
        LipschitzSignCache(resources.mesh_hash, _stable_hash(resources.geometry_policy))
        if sign_backend == "lipschitz_certified_cache_with_exact_fallback_v1"
        else None
    )
    solver_sdf: Any = resources.sdf
    if (
        getattr(execution_profile, "exact_closest_point_backend", "")
        == "compiled_object_local_bvh_v1"
    ):
        if compiled is None:
            report["compiled_solver_sdf_fallback"] = "compiled backend unavailable"
        else:
            # Values remain exact hybrid SDF values; this only moves the
            # object-local closest-point and generalized-winding evaluation to
            # the persistent compiled handles.  Full-surface audits continue
            # to use resources.reference_sdf independently.
            solver_sdf = compiled
            report["selected"] = "compiled_hybrid_exact_solver_only_v1"
            report["compiled_solver_sdf"] = compiled.describe()
    return RefinementRuntimeBackends(
        solver_sdf=solver_sdf,
        compiled_spatial_fd_backend=compiled,
        sign_cache=sign_cache,
        sdf_report=report,
        build_counts={
            "compiled_bvh_build_count": int(compiled is not None),
            "compiled_winding_handle_build_count": int(
                compiled is not None and compiled.winding_handle is not None
            ),
            "sign_cache_build_count": int(sign_cache is not None),
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
        FINAL_REFINEMENT_SCHEMA_VERSION_V3,
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
    initial_query_sets: dict[int, CollisionQuerySet] | None = None,
    object_vertices: np.ndarray | None = None,
    object_faces: np.ndarray | None = None,
    warm_artifact_hash: str | None = None,
    graph_artifact_hash: str | None = None,
    continue_on_failure: bool = False,
    resources: RefinementResources | None = None,
    runtime_backends: RefinementRuntimeBackends | None = None,
    frame_callback: Callable[[int, FinalFrameResult, _FrameContext], None] | None = None,
    pause_check: Callable[[int], bool] | None = None,
    source_frame_offset: int = 0,
    execution_profile: Any | None = None,
    regularization_profile: str = "auto",
    transport_previous_final: bool = False,
    enable_continuity_recovery: bool = True,
    diagnostic_force_window: bool = False,
    fixed_base_to_seed: bool = False,
    fixed_qpos_to_seed: bool = False,
    quality_extension: dict[str, Any] | None = None,
) -> tuple[FinalRetargetTrajectory, dict[str, Any]]:
    warm.validate()
    graph.validate()
    if warm.frame_count != graph.frame_count:
        raise ValueError("warm-start and graph frame counts differ")
    if warm.metadata.get("robot_name") != robot_model.name:
        raise ValueError("warm-start and selected robot differ")
    if not np.array_equal(warm.arrays["timestamps"], graph.timestamps):
        raise ValueError("warm-start and graph timestamps differ")
    temporal_scope_by_profile = {
        "faithful_current_baseline": "base_and_finger",
        "faithful_regularization_fix_v1": "finger_only",
        "temporal_finger_only": "finger_only",
        "temporal_base_only": "base_only",
        "no_temporal": "none",
        "wuji_continuous_full_state_v1": "continuous_full_state",
        "continuous_full_state_plus_paper": "continuous_full_state_plus_paper",
    }
    regularization_profile = regularization_profile_for_solver(
        solver_profile.profile_id, regularization_profile
    )
    if regularization_profile not in temporal_scope_by_profile:
        raise ValueError(f"unsupported regularization profile: {regularization_profile}")
    temporal_scope = temporal_scope_by_profile[regularization_profile]
    continuous_profile = (
        ContinuousRetargetProfile.load(profile_id=solver_profile.profile_id)
        if is_continuous_profile(solver_profile.profile_id)
        else None
    )
    transport_enabled = continuous_profile is not None or bool(transport_previous_final)
    continuity_gate_enabled = transport_enabled
    continuity_recovery_enabled = bool(enable_continuity_recovery and transport_enabled)
    window_fallback_enabled = bool(
        continuous_profile is None
        or continuous_profile.values.get("window", {}).get("fallback_enabled", True)
    )
    point_jacobian_backend = str(
        getattr(execution_profile, "point_jacobian_backend", "reference_batched_torch_v1")
    )
    strict_recovery = str(getattr(execution_profile, "strict_recovery", "none"))
    sdf_tree_leaf_size = int(getattr(execution_profile, "sdf_tree_leaf_size", 32))
    spatial_gradient_backend = str(
        getattr(
            execution_profile, "signed_distance_gradient", "legacy_surface_normal_optimizer_fd_v1"
        )
    )
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
    runtime_backends = runtime_backends or prepare_refinement_runtime_backends(
        resources, execution_profile
    )
    sdf = runtime_backends.solver_sdf
    sdf_report = runtime_backends.sdf_report
    compiled_spatial_fd_backend = runtime_backends.compiled_spatial_fd_backend
    sign_cache = runtime_backends.sign_cache
    stop = warm.frame_count if end_frame is None else int(end_frame)
    if start_frame < 0 or stop <= start_frame or stop > warm.frame_count:
        raise ValueError(f"invalid frame range [{start_frame},{stop})")
    frame_indices = list(range(start_frame, stop))
    frames: list[FinalFrameResult] = []
    future_hints: dict[int, np.ndarray] = {}
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
        propagated: PropagatedRetargetState | None = None
        if previous_base is not None and previous_qpos is not None:
            previous = map_previous_state_to_seed(
                previous_base, previous_qpos, warm.arrays["base_pose_scene"][local_index]
            )
        if transport_enabled:
            if previous_base is None or previous_qpos is None:
                predicted_base = np.asarray(
                    warm.arrays["base_pose_scene"][local_index], dtype=np.float64
                )
                predicted_qpos = np.asarray(warm.arrays["qpos"][local_index], dtype=np.float64)
                propagated = PropagatedRetargetState(
                    predicted_base_scene=predicted_base,
                    predicted_qpos=predicted_qpos,
                    base_correction=np.zeros(6, dtype=np.float64),
                    q_correction=np.zeros_like(predicted_qpos),
                    q_clamp_count=0,
                    previous_frame=None,
                    current_frame=int(graph.frame_indices[local_index]),
                    initialization_source="warm_first_frame",
                )
            else:
                previous_local = local_index - 1
                if previous_local < 0:
                    raise ValueError("continuous transport requires the previous warm frame")
                propagated = transport_previous_final_to_current_warm(
                    warm.arrays["base_pose_scene"][previous_local],
                    previous_base,
                    warm.arrays["base_pose_scene"][local_index],
                    warm.arrays["qpos"][previous_local],
                    previous_qpos,
                    warm.arrays["qpos"][local_index],
                    robot_model.joint_lower,
                    robot_model.joint_upper,
                    previous_frame=int(graph.frame_indices[previous_local]),
                    current_frame=int(graph.frame_indices[local_index]),
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
            temporal_scope=temporal_scope,
            fixed_base_to_seed=fixed_base_to_seed,
            fixed_qpos_to_seed=fixed_qpos_to_seed,
            quality_extension=_quality_extension_for_frame(quality_extension, local_index),
            continuous_prediction_base=(
                None
                if propagated is None or propagated.previous_frame is None
                else propagated.predicted_base_scene
            ),
            continuous_prediction_qpos=(
                None
                if propagated is None or propagated.previous_frame is None
                else propagated.predicted_qpos
            ),
            spatial_gradient_backend=spatial_gradient_backend,
            sign_cache=sign_cache,
            compiled_spatial_fd_backend=compiled_spatial_fd_backend,
        )
        # The selected solver backend preserves the original closest-point
        # magnitude and uses the audited sign policy. It is sufficient for
        # initial QuerySet selection; reference_sdf remains the independent
        # persisted audit.
        initial_points = context.candidate_points(np.concatenate([np.zeros(6), context.seed_qpos]))
        initial_query = sdf.query_scene(initial_points, context.object_pose_scene)
        if getattr(initial_query, "near_original_boundary", None) is not None:
            boundary_mask = np.asarray(initial_query.near_original_boundary, dtype=bool)
            patch_mask = (
                np.zeros_like(boundary_mask, dtype=bool)
                if initial_query.proxy_closest_is_synthetic_patch is None
                else np.asarray(initial_query.proxy_closest_is_synthetic_patch, dtype=bool)
            )
            if query_profile.mode == "full":
                active_boundary_mask = boundary_mask
            else:
                active_boundary_mask = boundary_mask & (
                    np.asarray(initial_query.signed_distance) < query_profile.active_margin_m
                )
            if np.any(active_boundary_mask):
                raise ValueError(
                    "SIGN_PROXY_CONTACT_REGION_CONFLICT: active collision QuerySet contains "
                    "samples in the original boundary exclusion zone; "
                    "active_queryset_near_boundary_count="
                    f"{int(np.count_nonzero(active_boundary_mask))}; "
                    "active_queryset_proxy_patch_count="
                    f"{int(np.count_nonzero(active_boundary_mask & patch_mask))}"
                )
        native_query_set = build_query_set(
            initial_query.signed_distance, surface.geometry_ids, query_profile
        )
        query_set = native_query_set
        if initial_query_sets is not None and local_index in initial_query_sets:
            # Diagnostic callers may freeze the IDs/reasons selected by an
            # official initial pass while retaining the current seed's signed
            # distances for slack initialization.  The default formal path is
            # unchanged when this mapping is omitted.
            frozen = initial_query_sets[local_index].validate(surface.count)
            query_set = CollisionQuerySet(
                sample_ids=np.asarray(frozen.sample_ids, dtype=np.int64),
                inclusion_reasons=tuple(frozen.inclusion_reasons),
                active_round=np.asarray(frozen.active_round, dtype=np.int64),
                initial_signed_distance=np.asarray(initial_query.signed_distance)[
                    np.asarray(frozen.sample_ids, dtype=np.int64)
                ],
                query_hash=frozen.query_hash,
            ).validate(surface.count)
        if query_profile.mode == "full":
            query_set = CollisionQuerySet(
                query_set.sample_ids,
                query_set.inclusion_reasons,
                query_set.active_round,
                np.asarray(initial_query.signed_distance)[query_set.sample_ids],
                query_set.query_hash,
            )
        initial_state_without_slack = None
        initialization_source = "warm_reset"
        if propagated is not None:
            initial_state_without_slack = np.concatenate(
                [
                    propagated.base_correction,
                    propagated.predicted_qpos,
                ]
            )
            initialization_source = propagated.initialization_source
        frame_result: FinalFrameResult | None = None
        if not diagnostic_force_window:
            frame_result = refine_frame(
                context,
                query_set,
                solver_profile,
                max_rounds=query_profile.max_active_set_rounds,
                active_margin_m=query_profile.active_margin_m,
                point_jacobian_backend=point_jacobian_backend,
                strict_recovery=strict_recovery,
                initial_state_without_slack=initial_state_without_slack,
                initialization_source=initialization_source,
            )
        if continuity_gate_enabled:
            attempt_candidates: list[FinalFrameResult] = []
            if frame_result is not None and not diagnostic_force_window:
                frame_result = _apply_continuity_gate(
                    frame_result,
                    context,
                    frame_id=int(graph.frame_indices[local_index]),
                    retry_attempt=0,
                    retry_profile="continuous_propagated",
                )
                attempt_candidates = [frame_result]
            if (
                not diagnostic_force_window
                and continuity_recovery_enabled
                and frame_result is not None
                and not frame_result.accepted
            ):
                if initial_state_without_slack is None:
                    raise RuntimeError("continuous profile did not construct an initial state")
                context.trust_region_reference = initial_state_without_slack
                context.trust_region_limits = (S_POS_M, S_ROT_RAD, S_Q_RAD)
                trust_result = refine_frame(
                    context,
                    query_set,
                    solver_profile,
                    max_rounds=query_profile.max_active_set_rounds,
                    active_margin_m=query_profile.active_margin_m,
                    point_jacobian_backend=point_jacobian_backend,
                    strict_recovery=strict_recovery,
                    initial_state_without_slack=initial_state_without_slack,
                    initialization_source="propagated_previous_final",
                    retry_attempt=1,
                    retry_profile="propagated_trust_region",
                )
                trust_result = _apply_continuity_gate(
                    trust_result,
                    context,
                    frame_id=int(graph.frame_indices[local_index]),
                    retry_attempt=1,
                    retry_profile="propagated_trust_region",
                )
                attempt_candidates.append(trust_result)
                if trust_result.accepted:
                    frame_result = trust_result
                else:
                    candidate_states: list[tuple[str, np.ndarray]] = [
                        ("propagated_previous_final", initial_state_without_slack),
                        (
                            "current_warm",
                            np.concatenate(
                                [
                                    np.zeros(6, dtype=np.float64),
                                    np.asarray(warm.arrays["qpos"][local_index], dtype=np.float64),
                                ]
                            ),
                        ),
                    ]
                    if local_index in future_hints:
                        candidate_states.append(
                            ("previous_window_future_hint", future_hints[local_index])
                        )
                    if previous_base is not None and previous_qpos is not None:
                        candidate_states.append(
                            (
                                "last_active_set_feasible",
                                np.concatenate(
                                    [
                                        encode_base_correction(
                                            warm.arrays["base_pose_scene"][local_index],
                                            frame_result.base_pose_scene,
                                        ),
                                        frame_result.qpos,
                                    ]
                                ),
                            )
                        )
                        candidate_states.append(
                            (
                                "previous_absolute_final_reencoded",
                                np.concatenate(
                                    [
                                        encode_base_correction(
                                            warm.arrays["base_pose_scene"][local_index],
                                            previous_base,
                                        ),
                                        previous_qpos,
                                    ]
                                ),
                            )
                        )
                    multi_results: list[FinalFrameResult] = []
                    seen: set[bytes] = set()
                    for source, candidate_state in candidate_states:
                        key = np.asarray(candidate_state, dtype=np.float64).tobytes()
                        if key in seen:
                            continue
                        seen.add(key)
                        multi = refine_frame(
                            context,
                            query_set,
                            solver_profile,
                            max_rounds=query_profile.max_active_set_rounds,
                            active_margin_m=query_profile.active_margin_m,
                            point_jacobian_backend=point_jacobian_backend,
                            strict_recovery=strict_recovery,
                            initial_state_without_slack=candidate_state,
                            initialization_source=source,
                            retry_attempt=2,
                            retry_profile="deterministic_multi_start",
                        )
                        multi_results.append(
                            _apply_continuity_gate(
                                multi,
                                context,
                                frame_id=int(graph.frame_indices[local_index]),
                                retry_attempt=2,
                                retry_profile="deterministic_multi_start",
                            )
                        )
                    attempt_candidates.extend(multi_results)
                    accepted_candidates = [item for item in attempt_candidates if item.accepted]
                    if accepted_candidates:
                        frame_result = min(
                            accepted_candidates,
                            key=lambda item: (item.final_objective, item.retry_attempt),
                        )
                    elif window_fallback_enabled:
                        window = RecedingHorizonWindow.for_target(
                            local_index, warm.frame_count, window_size=5
                        )
                        # Attempt 3 is a genuine joint offline solve over
                        # [t, t+1, t+2, t+3].  The center result is still
                        # revalidated through the formal single-frame path;
                        # future solutions become hints only.
                        window_entries: list[
                            tuple[_FrameContext, CollisionQuerySet, np.ndarray]
                        ] = [(context, query_set, initial_state_without_slack)]
                        window_previous_base = _as_np(
                            context.base_pose_torch(
                                __import__("torch").as_tensor(
                                    np.concatenate(
                                        [
                                            initial_state_without_slack,
                                            np.zeros(query_set.count),
                                        ]
                                    ),
                                    dtype=__import__("torch").float64,
                                )
                            )
                        )
                        window_previous_qpos = np.asarray(
                            initial_state_without_slack[6 : 6 + robot_model.num_dofs],
                            dtype=np.float64,
                        )
                        try:
                            for future_local in window.variable_frames[1:]:
                                propagated_future = transport_previous_final_to_current_warm(
                                    warm.arrays["base_pose_scene"][future_local - 1],
                                    window_previous_base,
                                    warm.arrays["base_pose_scene"][future_local],
                                    warm.arrays["qpos"][future_local - 1],
                                    window_previous_qpos,
                                    warm.arrays["qpos"][future_local],
                                    robot_model.joint_lower,
                                    robot_model.joint_upper,
                                    previous_frame=int(graph.frame_indices[future_local - 1]),
                                    current_frame=int(graph.frame_indices[future_local]),
                                )
                                future_context = _make_context(
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
                                    future_local,
                                    None,
                                    temporal_scope=temporal_scope,
                                    quality_extension=_quality_extension_for_frame(
                                        quality_extension, future_local
                                    ),
                                    continuous_prediction_base=propagated_future.predicted_base_scene,
                                    continuous_prediction_qpos=propagated_future.predicted_qpos,
                                )
                                future_state = np.concatenate(
                                    [
                                        propagated_future.base_correction,
                                        propagated_future.predicted_qpos,
                                    ]
                                )
                                future_query = _initial_query_set_for_context(
                                    future_context, future_state, query_profile
                                )
                                window_entries.append((future_context, future_query, future_state))
                                window_previous_base = propagated_future.predicted_base_scene
                                window_previous_qpos = propagated_future.predicted_qpos

                            joint = _window_joint_objective(
                                window_entries,
                                solver_profile,
                                point_jacobian_backend=point_jacobian_backend,
                                left_anchor=(previous_base, previous_qpos)
                                if previous_base is not None and previous_qpos is not None
                                else None,
                            )
                            for hint_local, hint_state in zip(
                                window.variable_frames[1:], joint["states"][1:], strict=True
                            ):
                                future_hints[int(hint_local)] = np.asarray(
                                    hint_state, dtype=np.float64
                                )
                            window_result = refine_frame(
                                context,
                                query_set,
                                solver_profile,
                                max_rounds=query_profile.max_active_set_rounds,
                                active_margin_m=query_profile.active_margin_m,
                                point_jacobian_backend=point_jacobian_backend,
                                strict_recovery=strict_recovery,
                                initial_state_without_slack=joint["states"][0],
                                initialization_source="five_frame_window_joint_center",
                                retry_attempt=3,
                                retry_profile="five_frame_window",
                            )
                            window_result.jacobian_diagnostics["window_joint"] = {
                                "success": bool(joint["success"]),
                                "status": int(joint["status"]),
                                "message": str(joint["message"]),
                                "backend": joint.get("backend"),
                                "scaled_coordinates": bool(joint.get("scaled_coordinates", False)),
                                "coordinate_scales": joint.get("coordinate_scales"),
                                "left_anchor_used": bool(joint.get("left_anchor_used", False)),
                                "slsqp": joint.get("slsqp"),
                                "trust_constr": joint.get("trust_constr"),
                                "trust_constr_error": joint.get("trust_constr_error"),
                                "iterations": int(joint["iterations"]),
                                "objective": float(joint["objective"]),
                                "full_surface_audits": joint.get("full_surface_audits", []),
                                "variable_frames": list(window.variable_frames),
                                "future_hint_frames": list(window.variable_frames[1:]),
                                "per_frame_query_sets": [
                                    {
                                        "frame": int(graph.frame_indices[local]),
                                        "query_count": int(entry[1].count),
                                        "query_hash": entry[1].query_hash,
                                        "active_set_rounds": int(
                                            entry[1].active_round.max(initial=0)
                                        ),
                                    }
                                    for local, entry in zip(
                                        window.variable_frames, window_entries, strict=True
                                    )
                                ],
                            }
                        except Exception as exc:
                            window_result = refine_frame(
                                context,
                                query_set,
                                solver_profile,
                                max_rounds=query_profile.max_active_set_rounds,
                                active_margin_m=query_profile.active_margin_m,
                                point_jacobian_backend=point_jacobian_backend,
                                strict_recovery=strict_recovery,
                                initial_state_without_slack=initial_state_without_slack,
                                initialization_source="five_frame_window_failed_fallback",
                                retry_attempt=3,
                                retry_profile="five_frame_window",
                            )
                            window_result.jacobian_diagnostics["window_joint"] = {
                                "success": False,
                                "failure": type(exc).__name__ + ": " + str(exc),
                                "variable_frames": list(window.variable_frames),
                            }
                        window_result = _apply_continuity_gate(
                            window_result,
                            context,
                            frame_id=int(graph.frame_indices[local_index]),
                            retry_attempt=3,
                            retry_profile="five_frame_window",
                            window_used=True,
                        )
                        window_result.jacobian_diagnostics["window"] = window.as_dict()
                        frame_result = window_result
                    else:
                        # The sequential profile deliberately stops after the
                        # formal propagated/trust/multi-start attempts.  In
                        # particular, it must never silently enter the
                        # experimental five-frame path.
                        frame_result = min(
                            attempt_candidates,
                            key=lambda item: (not item.accepted, item.final_objective),
                        )
            elif diagnostic_force_window:
                window = RecedingHorizonWindow.for_target(
                    local_index, warm.frame_count, window_size=5
                )
                if initial_state_without_slack is None:
                    raise RuntimeError("forced five-frame window requires an initial state")
                forced_window_entries: list[tuple[_FrameContext, CollisionQuerySet, np.ndarray]] = [
                    (context, query_set, initial_state_without_slack)
                ]
                window_previous_base = _as_np(
                    context.base_pose_torch(
                        __import__("torch").as_tensor(
                            np.concatenate(
                                [initial_state_without_slack, np.zeros(query_set.count)]
                            ),
                            dtype=__import__("torch").float64,
                        )
                    )
                )
                window_previous_qpos = np.asarray(
                    initial_state_without_slack[6 : 6 + robot_model.num_dofs], dtype=np.float64
                )
                try:
                    for future_local in window.variable_frames[1:]:
                        propagated_future = transport_previous_final_to_current_warm(
                            warm.arrays["base_pose_scene"][future_local - 1],
                            window_previous_base,
                            warm.arrays["base_pose_scene"][future_local],
                            warm.arrays["qpos"][future_local - 1],
                            window_previous_qpos,
                            warm.arrays["qpos"][future_local],
                            robot_model.joint_lower,
                            robot_model.joint_upper,
                            previous_frame=int(graph.frame_indices[future_local - 1]),
                            current_frame=int(graph.frame_indices[future_local]),
                        )
                        future_context = _make_context(
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
                            future_local,
                            None,
                            temporal_scope=temporal_scope,
                            quality_extension=_quality_extension_for_frame(
                                quality_extension, future_local
                            ),
                            continuous_prediction_base=propagated_future.predicted_base_scene,
                            continuous_prediction_qpos=propagated_future.predicted_qpos,
                        )
                        future_state = np.concatenate(
                            [propagated_future.base_correction, propagated_future.predicted_qpos]
                        )
                        future_query = _initial_query_set_for_context(
                            future_context, future_state, query_profile
                        )
                        forced_window_entries.append((future_context, future_query, future_state))
                        window_previous_base = propagated_future.predicted_base_scene
                        window_previous_qpos = propagated_future.predicted_qpos
                    joint = _window_joint_objective(
                        forced_window_entries,
                        solver_profile,
                        point_jacobian_backend=point_jacobian_backend,
                        left_anchor=(previous_base, previous_qpos)
                        if previous_base is not None and previous_qpos is not None
                        else None,
                    )
                    for hint_local, hint_state in zip(
                        window.variable_frames[1:], joint["states"][1:], strict=True
                    ):
                        future_hints[int(hint_local)] = np.asarray(hint_state, dtype=np.float64)
                    window_result = refine_frame(
                        context,
                        query_set,
                        solver_profile,
                        max_rounds=query_profile.max_active_set_rounds,
                        active_margin_m=query_profile.active_margin_m,
                        point_jacobian_backend=point_jacobian_backend,
                        strict_recovery=strict_recovery,
                        initial_state_without_slack=joint["states"][0],
                        initialization_source="five_frame_window_joint_center",
                        retry_attempt=3,
                        retry_profile="five_frame_window",
                    )
                    window_result.jacobian_diagnostics["window_joint"] = {
                        "success": bool(joint["success"]),
                        "status": int(joint["status"]),
                        "message": str(joint["message"]),
                        "backend": joint.get("backend"),
                        "scaled_coordinates": bool(joint.get("scaled_coordinates", False)),
                        "coordinate_scales": joint.get("coordinate_scales"),
                        "left_anchor_used": bool(joint.get("left_anchor_used", False)),
                        "slsqp": joint.get("slsqp"),
                        "trust_constr": joint.get("trust_constr"),
                        "trust_constr_error": joint.get("trust_constr_error"),
                        "iterations": int(joint["iterations"]),
                        "objective": float(joint["objective"]),
                        "full_surface_audits": joint.get("full_surface_audits", []),
                        "variable_frames": list(window.variable_frames),
                        "future_hint_frames": list(window.variable_frames[1:]),
                        "per_frame_query_sets": [
                            {
                                "frame": int(graph.frame_indices[local]),
                                "query_count": int(entry[1].count),
                                "query_hash": entry[1].query_hash,
                                "active_set_rounds": int(entry[1].active_round.max(initial=0)),
                            }
                            for local, entry in zip(
                                window.variable_frames, forced_window_entries, strict=True
                            )
                        ],
                    }
                except Exception as exc:
                    window_result = refine_frame(
                        context,
                        query_set,
                        solver_profile,
                        max_rounds=query_profile.max_active_set_rounds,
                        active_margin_m=query_profile.active_margin_m,
                        point_jacobian_backend=point_jacobian_backend,
                        strict_recovery=strict_recovery,
                        initial_state_without_slack=initial_state_without_slack,
                        initialization_source="five_frame_window_failed_fallback",
                        retry_attempt=3,
                        retry_profile="five_frame_window",
                    )
                    window_result.jacobian_diagnostics["window_joint"] = {
                        "success": False,
                        "failure": type(exc).__name__ + ": " + str(exc),
                        "variable_frames": list(window.variable_frames),
                    }
                frame_result = _apply_continuity_gate(
                    window_result,
                    context,
                    frame_id=int(graph.frame_indices[local_index]),
                    retry_attempt=3,
                    retry_profile="five_frame_window",
                    window_used=True,
                )
                frame_result.jacobian_diagnostics["window"] = window.as_dict()
            elif frame_result is None:
                raise RuntimeError("continuous refinement produced no frame result")
            context.trust_region_reference = None
            context.trust_region_limits = None
            frame_result.jacobian_diagnostics["q_clamp_count"] = int(
                0 if propagated is None else propagated.q_clamp_count
            )
        if frame_result is None:
            raise RuntimeError("refinement produced no frame result")
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
                "gradient": {
                    key: frame_result.jacobian_diagnostics.get(key)
                    for key in (
                        "analytic_point_count",
                        "ambiguous_point_count",
                        "spatial_fd_probe_count",
                        "sdf_batches_for_jacobian",
                        "surface_normal_last_resort_count",
                        "ambiguity_reason_counts",
                        "compiled_exact_sign",
                    )
                },
                "compiled_exact_sign": (
                    {}
                    if context.compiled_spatial_fd_backend is None
                    else {
                        **context.compiled_spatial_fd_backend.probe_sign_stats,
                        "reuse_rate": (
                            float(
                                context.compiled_spatial_fd_backend.probe_sign_stats[
                                    "certified_probe_reuse"
                                ]
                                / context.compiled_spatial_fd_backend.probe_sign_stats[
                                    "total_fd_probes"
                                ]
                            )
                            if context.compiled_spatial_fd_backend.probe_sign_stats[
                                "total_fd_probes"
                            ]
                            else 0.0
                        ),
                    }
                ),
                "sign_cache": frame_result.jacobian_diagnostics.get("sign_cache", {}),
                "timers": frame_result.jacobian_diagnostics.get("timers", {}),
                "window_joint": frame_result.jacobian_diagnostics.get("window_joint"),
                "window": frame_result.jacobian_diagnostics.get("window"),
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
                "gradient": {
                    key: frame_result.jacobian_diagnostics.get(key)
                    for key in (
                        "analytic_point_count",
                        "ambiguous_point_count",
                        "spatial_fd_probe_count",
                        "sdf_batches_for_jacobian",
                        "surface_normal_last_resort_count",
                        "ambiguity_reason_counts",
                        "compiled_exact_sign",
                    )
                },
                "sign_cache": frame_result.jacobian_diagnostics.get("sign_cache", {}),
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

    def continuity_series(key: str, default: float = math.nan) -> np.ndarray:
        return np.asarray(
            [float(item.continuity_metrics.get(key, default)) for item in frames],
            dtype=np.float64,
        )

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
        "e_morph": series("e_morph"),
        "weighted_e_morph": series("weighted_e_morph"),
        "e_contact_pos": series("e_contact_pos"),
        "weighted_e_contact_pos": series("weighted_e_contact_pos"),
        "e_contact_dir": series("e_contact_dir"),
        "weighted_e_contact_dir": series("weighted_e_contact_dir"),
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
        "single_frame_feasible": np.asarray(
            [item.single_frame_feasible for item in frames], dtype=bool
        ),
        "trajectory_continuous": np.asarray(
            [item.trajectory_continuous for item in frames], dtype=bool
        ),
        "final_accepted": np.asarray([item.accepted for item in frames], dtype=bool),
        "continuity_failure_reasons": np.asarray(
            [
                ",".join(
                    str(value)
                    for value in item.continuity_metrics.get("continuity_failure_reasons", [])
                )
                for item in frames
            ],
            dtype="S512",
        ),
        "initialization_source": np.asarray(
            [item.initialization_source for item in frames], dtype="S96"
        ),
        "retry_attempt": np.asarray([item.retry_attempt for item in frames], dtype=np.int64),
        "retry_profile": np.asarray([item.retry_profile for item in frames], dtype="S96"),
        "window_used": np.asarray([item.window_used for item in frames], dtype=bool),
        "continuity_base_translation_m": continuity_series("delta_base_translation_m"),
        "continuity_base_rotation_rad": continuity_series("delta_base_rotation_rad"),
        "continuity_finger_inf_rad": continuity_series("delta_finger_inf_rad"),
        "continuity_excess_keypoint_m": continuity_series("excess_keypoint_max_m"),
        "q_clamp_count": np.asarray(
            [
                int(
                    item.jacobian_diagnostics.get(
                        "q_clamp_count",
                        0,
                    )
                )
                for item in frames
            ],
            dtype=np.int64,
        ),
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
            FINAL_REFINEMENT_SCHEMA_VERSION_V3
            if continuous_profile is not None
            else (
                FINAL_REFINEMENT_SCHEMA_VERSION_V2
                if solver_profile.profile_id
                in {
                    CONTACT_RICH_SOLVER_PROFILE_ID,
                    FAITHFUL_CONTACT_RICH_SOLVER_PROFILE_ID,
                }
                else FINAL_REFINEMENT_SCHEMA_VERSION_V1
            )
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
        "robot_dof_count": int(robot_model.num_dofs),
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
        "execution_profile": (None if execution_profile is None else execution_profile.as_dict()),
        "point_jacobian_backend": point_jacobian_backend,
        "strict_recovery": strict_recovery,
        "sdf_tree_leaf_size": sdf_tree_leaf_size,
        "termination_contract": solver_profile.termination_contract,
        "acceptance_policy_id": solver_profile.acceptance_policy_id,
        "active_set_continuation_policy": solver_profile.active_set_continuation_policy,
        "maxiter_provenance": solver_profile.maxiter_provenance,
        "stationarity_policy": solver_profile.stationarity_policy,
        "paper_weights": paper.as_dict(),
        "regularization_profile": regularization_profile,
        "temporal_scope": temporal_scope,
        "base_correction_convention": BASE_CORRECTION_CONVENTION,
        "continuous_profile": (
            None if continuous_profile is None else continuous_profile.as_dict()
        ),
        "continuity_schema_version": (
            None if continuous_profile is None else "toporetarget.trajectory_continuity.v1"
        ),
        "continuity_thresholds": (
            None
            if continuous_profile is None
            else dict(continuous_profile.values.get("continuity", {}))
        ),
        "retry_profile": (
            None if continuous_profile is None else dict(continuous_profile.values.get("retry", {}))
        ),
        "window_profile": (
            None
            if continuous_profile is None
            else dict(continuous_profile.values.get("window", {}))
        ),
        "fixed_base_to_seed": bool(fixed_base_to_seed),
        "fixed_qpos_to_seed": bool(fixed_qpos_to_seed),
        "quality_extension": None
        if quality_extension is None
        else {
            key: value
            for key, value in quality_extension.items()
            if key
            not in {
                "morphology_target_keypoints_scene",
                "contact_target_relative",
                "contact_target_direction",
                "contact_active",
                "contact_weights",
                "contact_regions",
            }
        },
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
                **runtime_backends.build_counts,
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
        "future_hints": {
            str(frame): np.asarray(state, dtype=np.float64).tolist()
            for frame, state in sorted(future_hints.items())
        },
    }


__all__ = [
    "ACTIVE_QUERY_PROFILE_ID",
    "CONTACT_RICH_SOLVER_PROFILE_ID",
    "FAITHFUL_CONTACT_RICH_SOLVER_PROFILE_ID",
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
    "FINAL_REFINEMENT_SCHEMA_VERSION_V3",
    "PaperRefinementWeights",
    "RefinementCoordinateProfile",
    "RefinementResources",
    "RefinementRuntimeBackends",
    "RefinementSolverProfile",
    "regularization_profile_for_solver",
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
    "_apply_continuity_gate",
    "prepare_refinement_resources",
    "prepare_refinement_runtime_backends",
    "save_final_trajectory",
    "so3_exp",
    "so3_log",
    "strict_acceptance_decision",
]
