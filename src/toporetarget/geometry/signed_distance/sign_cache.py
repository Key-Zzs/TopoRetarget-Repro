"""Certified, object-local signed-distance sign reuse.

The cache deliberately stores *only* sign provenance.  Every query still
computes an exact closest point and unsigned distance.  A reuse is permitted
only by the 1-Lipschitz signed-distance bound, which proves that the sign
cannot change between the two object-local points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

LIPSCHITZ_SIGN_CACHE_PROFILE_ID = "lipschitz_certified_cache_with_exact_fallback_v1"


@dataclass(frozen=True)
class SignCachePolicy:
    """Global numerical policy; it is never tuned per dataset or frame."""

    surface_epsilon_m: float = 1e-7
    sign_safety_margin_m: float = 1e-9
    profile_id: str = LIPSCHITZ_SIGN_CACHE_PROFILE_ID


@dataclass(frozen=True)
class _SignRecord:
    point_local: np.ndarray
    signed_distance: float
    sign: int
    reliable: bool
    mesh_hash: str
    sign_profile_hash: str
    lineage: str


@dataclass
class LipschitzSignCache:
    """Per stable QuerySet sample cache with proof-carrying hit decisions."""

    mesh_hash: str
    sign_profile_hash: str
    policy: SignCachePolicy = field(default_factory=SignCachePolicy)
    _entries: dict[int, _SignRecord] = field(default_factory=dict, init=False, repr=False)
    _stats: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def _count(self, name: str, amount: int = 1) -> None:
        self._stats[name] = int(self._stats.get(name, 0)) + amount

    def lookup(
        self,
        points_local: np.ndarray,
        sample_ids: np.ndarray | None,
        *,
        lineage: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return certified signs and a hit mask without querying a sign backend."""

        points = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        signs = np.zeros(len(points), dtype=np.int8)
        hits = np.zeros(len(points), dtype=bool)
        if sample_ids is None:
            self._count("invalidation_count")
            self._count("sign_cache_misses", len(points))
            return signs, hits
        ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
        if len(ids) != len(points) or len(np.unique(ids)) != len(ids):
            self._count("invalidation_count")
            self._count("sign_cache_misses", len(points))
            return signs, hits
        self._count("sign_cache_queries", len(points))
        for index, (point, sample_id) in enumerate(zip(points, ids, strict=True)):
            record = self._entries.get(int(sample_id))
            if (
                record is None
                or record.mesh_hash != self.mesh_hash
                or record.sign_profile_hash != self.sign_profile_hash
                or not record.reliable
            ):
                self._count("sign_cache_misses")
                continue
            displacement = float(np.linalg.norm(point - record.point_local))
            # If this strict inequality holds, |phi(new)| remains above the
            # surface band and the old sign is a mathematical certificate.
            safe_bound = (
                displacement + self.policy.sign_safety_margin_m + self.policy.surface_epsilon_m
            )
            if safe_bound < abs(record.signed_distance):
                signs[index] = record.sign
                hits[index] = True
                self._count("sign_cache_hits")
                self._count("certified_reuse_count")
            else:
                self._count("sign_cache_misses")
        return signs, hits

    def record_exact(
        self,
        points_local: np.ndarray,
        sample_ids: np.ndarray | None,
        signed_distance: np.ndarray,
        sign_reliable: np.ndarray,
        *,
        lineage: str,
    ) -> None:
        """Update entries only from exact sign evaluations, never FD probes."""

        if sample_ids is None:
            return
        points = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        ids = np.asarray(sample_ids, dtype=np.int64).reshape(-1)
        phi = np.asarray(signed_distance, dtype=np.float64).reshape(-1)
        reliable = np.asarray(sign_reliable, dtype=bool).reshape(-1)
        if not (len(points) == len(ids) == len(phi) == len(reliable)):
            raise ValueError("sign-cache record arrays must have equal lengths")
        for point, sample_id, value, is_reliable in zip(points, ids, phi, reliable, strict=True):
            if not np.isfinite(value) or value == 0.0 or not is_reliable:
                self._count("invalidation_count")
                continue
            self._entries[int(sample_id)] = _SignRecord(
                point_local=point.copy(),
                signed_distance=float(value),
                sign=1 if value > 0.0 else -1,
                reliable=True,
                mesh_hash=self.mesh_hash,
                sign_profile_hash=self.sign_profile_hash,
                lineage=str(lineage),
            )

    def note_exact_winding(self, count: int) -> None:
        self._count("exact_winding_count", int(count))

    def note_ambiguous_bypass(self, count: int) -> None:
        self._count("ambiguous_bypass_count", int(count))

    def as_dict(self) -> dict[str, Any]:
        queries = int(self._stats.get("sign_cache_queries", 0))
        hits = int(self._stats.get("sign_cache_hits", 0))
        return {
            "profile_id": self.policy.profile_id,
            "mesh_hash": self.mesh_hash,
            "sign_profile_hash": self.sign_profile_hash,
            "sign_cache_queries": queries,
            "sign_cache_hits": hits,
            "sign_cache_misses": int(self._stats.get("sign_cache_misses", 0)),
            "certified_reuse_count": int(self._stats.get("certified_reuse_count", 0)),
            "exact_winding_count": int(self._stats.get("exact_winding_count", 0)),
            "ambiguous_bypass_count": int(self._stats.get("ambiguous_bypass_count", 0)),
            "invalidation_count": int(self._stats.get("invalidation_count", 0)),
            "hit_rate": float(hits / queries) if queries else 0.0,
        }


__all__ = ["LIPSCHITZ_SIGN_CACHE_PROFILE_ID", "LipschitzSignCache", "SignCachePolicy"]
