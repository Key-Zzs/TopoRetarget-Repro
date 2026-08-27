"""Bounded performance instrumentation for Stage 9 constrained refinement.

The helpers in this module are deliberately numerical-policy agnostic.  They
only account for exact callback reuse and wall-clock cost; they do not change
the objective, constraints, solver profile, or acceptance contract.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml


@dataclass
class TimerBook:
    """Inclusive wall timers with explicit event counts."""

    elapsed_s: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.elapsed_s[name] += perf_counter() - started
            self.counts[name] += 1

    def add(self, name: str, elapsed_s: float) -> None:
        self.elapsed_s[name] += float(elapsed_s)
        self.counts[name] += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "elapsed_s": {key: float(value) for key, value in sorted(self.elapsed_s.items())},
            "counts": {key: int(value) for key, value in sorted(self.counts.items())},
        }


@dataclass
class RefinementEvaluationCache:
    """Exact-x, single-frame callback cache.

    ``np.array_equal`` is the only identity test.  No rounded keys or
    cryptographic hash is computed for candidate vectors.  A query-set change
    invalidates every layer so an active-set expansion cannot reuse a result
    produced under a different constraint dimension.
    """

    frame_id: int | str
    context_hash: str
    _x: np.ndarray | None = field(default=None, init=False, repr=False)
    _query_hash: str | None = field(default=None, init=False, repr=False)
    _layers: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _stats: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False, repr=False)

    def prepare(self, value: np.ndarray, query_hash: str) -> None:
        current = np.asarray(value, dtype=np.float64).reshape(-1)
        query = str(query_hash)
        if self._x is None:
            self._x = current.copy()
            self._query_hash = query
            self._stats["unique_x"] += 1
            return
        same_x = np.array_equal(self._x, current)
        same_query = self._query_hash == query
        if not same_x:
            self._x = current.copy()
            self._query_hash = query
            self._layers.clear()
            self._stats["unique_x"] += 1
            self._stats["new_x_invalidations"] += 1
        elif not same_query:
            self._query_hash = query
            self._layers.clear()
            self._stats["query_set_invalidations"] += 1
        else:
            self._stats["repeated_same_x_callbacks"] += 1

    def get(self, layer: str) -> Any | None:
        if layer in self._layers:
            self._stats["hits"] += 1
            self._stats[f"{layer}_hits"] += 1
            if layer == "candidate_points":
                self._stats["avoided_fk"] += 1
            elif layer == "constraint_query":
                self._stats["avoided_sdf"] += 1
            elif layer == "constraint_jacobian":
                self._stats["avoided_jacobian"] += 1
            return self._layers[layer]
        self._stats["misses"] += 1
        self._stats[f"{layer}_misses"] += 1
        return None

    def put(self, layer: str, value: Any) -> Any:
        self._layers[layer] = value
        return value

    def invalidate(self) -> None:
        self._layers.clear()
        self._stats["manual_invalidations"] += 1

    def as_dict(self) -> dict[str, Any]:
        hits = int(self._stats.get("hits", 0))
        misses = int(self._stats.get("misses", 0))
        lookups = hits + misses
        values: dict[str, Any] = {key: int(value) for key, value in sorted(self._stats.items())}
        values.update(
            {
                "frame_id": self.frame_id,
                "context_hash": self.context_hash,
                "query_hash": self._query_hash,
                "hits": hits,
                "misses": misses,
                "unique_x": int(self._stats.get("unique_x", 0)),
                "reuse_ratio": float(hits / lookups) if lookups else 0.0,
            }
        )
        return values


@dataclass(frozen=True)
class RefinementExecutionProfile:
    """Engineering execution policy kept separate from the SLSQP profile."""

    profile_id: str
    version: str
    device: str
    dtype: str
    cache_mode: str
    checkpoint_schema: str
    durable_checkpoint_interval_frames: int
    intermediate_checkpoint_mode: str
    historical_sequence_rewrite: bool
    full_audit_mode: str
    initialization_profile: str
    variable_scaling: str
    point_jacobian_backend: str
    strict_recovery: str
    sdf_tree_leaf_size: int
    role: str
    math_equivalent: bool
    final_full_surface_audit: bool
    final_audit_scheduling: str
    recommended: bool
    stage12_default: bool
    paper_objective_unchanged: bool
    paper_constraints_unchanged: bool
    continuity_contract_unchanged: bool
    author_exact: str
    profile_hash: str
    source_path: Path | None = None
    signed_distance_gradient: str = "legacy_surface_normal_optimizer_fd_v1"
    sign_backend: str = "exact_winding_per_query_v1"
    ambiguity_fd_backend: str = "fast_exact_v2_python"
    exact_closest_point_backend: str = "exact_object_local_bvh_v1"

    @classmethod
    def load(
        cls, profile_id: str = "cached_checkpoint_cpu_float64_v1", root: Path | None = None
    ) -> RefinementExecutionProfile:
        repo = root or Path(__file__).resolve().parents[3]
        path = repo / "configs" / "retarget" / "refinement_execution" / f"{profile_id}.yaml"
        raw = path.read_bytes()
        values = yaml.safe_load(raw) or {}
        if not isinstance(values, dict):
            raise ValueError(f"execution profile must be a mapping: {path}")
        result = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            device=str(values.get("device", "cpu")),
            dtype=str(values.get("dtype", "float64")),
            cache_mode=str(
                values.get("cache_mode", values.get("cache", "exact_x_common_forward"))
            ).replace("_v1", ""),
            checkpoint_schema=str(
                values.get("checkpoint_schema", "toporetarget.final_retarget_checkpoint.v1")
            ),
            durable_checkpoint_interval_frames=int(
                values.get("durable_checkpoint_interval_frames", 1)
            ),
            intermediate_checkpoint_mode=str(
                values.get("intermediate_checkpoint_mode", "atomic_per_frame")
            ),
            historical_sequence_rewrite=bool(values.get("historical_sequence_rewrite", False)),
            full_audit_mode=str(
                values.get(
                    "full_audit_mode",
                    values.get("full_audit_policy", "outer_round_and_final"),
                )
            )
            .replace("_independent_v1", "")
            .replace("_v1", ""),
            initialization_profile=str(values.get("initialization_profile", "stage7_seed_v1")),
            variable_scaling=str(values.get("variable_scaling", "seed_delta_normalized_v1")),
            point_jacobian_backend=str(
                values.get("point_jacobian_backend", "reference_batched_torch_v1")
            ),
            strict_recovery=str(values.get("strict_recovery", "none")),
            sdf_tree_leaf_size=int(values.get("sdf_tree_leaf_size", 32)),
            role=str(values.get("role", "engineering_execution")),
            math_equivalent=bool(values.get("math_equivalent", False)),
            final_full_surface_audit=bool(values.get("final_full_surface_audit", True)),
            final_audit_scheduling=str(
                values.get("final_audit_scheduling", "independent_reference_query_v1")
            ),
            recommended=bool(values.get("recommended", False)),
            stage12_default=bool(values.get("stage12_default", False)),
            paper_objective_unchanged=bool(values.get("paper_objective_unchanged", True)),
            paper_constraints_unchanged=bool(values.get("paper_constraints_unchanged", True)),
            continuity_contract_unchanged=bool(values.get("continuity_contract_unchanged", True)),
            author_exact=str(values.get("author_exact", "unresolved")),
            profile_hash=hashlib.sha256(raw).hexdigest(),
            source_path=path,
            signed_distance_gradient=str(
                values.get("signed_distance_gradient", "legacy_surface_normal_optimizer_fd_v1")
            ),
            sign_backend=str(values.get("sign_backend", "exact_winding_per_query_v1")),
            ambiguity_fd_backend=str(values.get("ambiguity_fd_backend", "fast_exact_v2_python")),
            exact_closest_point_backend=str(
                values.get("exact_closest_point_backend", "exact_object_local_bvh_v1")
            ),
        )
        if result.device != "cpu" or result.dtype != "float64":
            raise ValueError("the validated Stage 9.2 execution profile must be CPU float64")
        if result.cache_mode != "exact_x_common_forward":
            raise ValueError("unsupported refinement cache mode")
        if result.variable_scaling != "seed_delta_normalized_v1":
            raise ValueError("unsupported refinement variable scaling")
        if result.point_jacobian_backend not in {
            "reference_batched_torch_v1",
            "analytic_urdf_spatial_v2",
        }:
            raise ValueError("unsupported refinement point Jacobian backend")
        if result.signed_distance_gradient not in {
            "legacy_surface_normal_optimizer_fd_v1",
            "spatial_gradient_chain_rule_v1",
        }:
            raise ValueError("unsupported signed-distance gradient backend")
        if result.sign_backend not in {
            "exact_winding_per_query_v1",
            "lipschitz_certified_cache_with_exact_fallback_v1",
        }:
            raise ValueError("unsupported signed-distance sign backend")
        if result.strict_recovery not in {
            "none",
            "reference_batched_from_primary_result_v1",
        }:
            raise ValueError("unsupported refinement strict recovery policy")
        if result.ambiguity_fd_backend not in {
            "fast_exact_v2_python",
            "compiled_spatial_central_fd_v1",
            "compiled_spatial_central_fd_winding_v1",
        }:
            raise ValueError("unsupported ambiguity spatial-FD backend")
        if result.exact_closest_point_backend not in {
            "exact_object_local_bvh_v1",
            "compiled_object_local_bvh_v1",
        }:
            raise ValueError("unsupported exact closest-point backend")
        if result.sdf_tree_leaf_size <= 0:
            raise ValueError("refinement SDF tree leaf size must be positive")
        if result.durable_checkpoint_interval_frames <= 0:
            raise ValueError("durable checkpoint interval must be positive")
        if result.historical_sequence_rewrite:
            raise ValueError("refinement checkpoints may not rewrite historical sequence output")
        if result.role == "performance_candidate" and (
            result.recommended or result.stage12_default
        ):
            raise ValueError("an unvalidated performance candidate cannot be a Stage 12 default")
        if not result.final_full_surface_audit:
            raise ValueError("all final-refinement execution profiles require a full final audit")
        if result.final_audit_scheduling not in {
            "independent_reference_query_v1",
            "reuse_exact_reference_discovery_if_identical_v1",
        }:
            raise ValueError("unsupported final full-surface audit scheduling")
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "device": self.device,
            "dtype": self.dtype,
            "cache_mode": self.cache_mode,
            "cache": f"{self.cache_mode}_v1",
            "checkpoint_schema": self.checkpoint_schema,
            "durable_checkpoint_interval_frames": self.durable_checkpoint_interval_frames,
            "intermediate_checkpoint_mode": self.intermediate_checkpoint_mode,
            "historical_sequence_rewrite": self.historical_sequence_rewrite,
            "full_audit_mode": self.full_audit_mode,
            "full_audit_policy": f"{self.full_audit_mode}_independent_v1",
            "initialization_profile": self.initialization_profile,
            "variable_scaling": self.variable_scaling,
            "point_jacobian_backend": self.point_jacobian_backend,
            "strict_recovery": self.strict_recovery,
            "sdf_tree_leaf_size": self.sdf_tree_leaf_size,
            "role": self.role,
            "math_equivalent": self.math_equivalent,
            "final_full_surface_audit": self.final_full_surface_audit,
            "final_audit_scheduling": self.final_audit_scheduling,
            "recommended": self.recommended,
            "stage12_default": self.stage12_default,
            "paper_objective_unchanged": self.paper_objective_unchanged,
            "paper_constraints_unchanged": self.paper_constraints_unchanged,
            "continuity_contract_unchanged": self.continuity_contract_unchanged,
            "author_exact": self.author_exact,
            "signed_distance_gradient": self.signed_distance_gradient,
            "sign_backend": self.sign_backend,
            "ambiguity_fd_backend": self.ambiguity_fd_backend,
            "exact_closest_point_backend": self.exact_closest_point_backend,
            "profile_hash": self.profile_hash,
            "source_path": None if self.source_path is None else str(self.source_path),
        }


def safe_percentile(values: list[float] | np.ndarray, percentile: float) -> float | None:
    """Return a JSON-friendly percentile for possibly empty timing lists."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        return None
    return float(np.percentile(array, percentile))


__all__ = [
    "RefinementEvaluationCache",
    "RefinementExecutionProfile",
    "TimerBook",
    "safe_percentile",
]
