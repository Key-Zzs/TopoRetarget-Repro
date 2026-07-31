"""Signed-distance backends with the repository's positive-outside convention."""

from .base import SignedDistanceBackend, SignedDistanceQueryResult, local_linearization
from .closest_point import ObjectLocalBVH
from .derived_proxy import (
    DERIVED_SDF_PROXY_SCHEMA_VERSION,
    HYBRID_SIGNED_DISTANCE_PROFILE_ID,
    DerivedWatertightSignProxy,
    HybridSignedDistanceBackend,
    ObjectSDFGeometryPolicy,
    build_derived_sign_proxy,
    build_hybrid_signed_distance_backend,
)
from .gradient import (
    AMBIGUITY_POLICY_ID,
    ANALYTIC_CLOSEST_FEATURE_GRADIENT_ID,
    SPATIAL_FD_FALLBACK_ID,
    SignedDistanceGradientAmbiguityPolicy,
    SignedDistanceGradientResult,
    ambiguity_reason_counts,
    analytic_spatial_gradient,
)
from .reference import (
    ReferenceSignedDistanceBackend,
    SignedDistanceError,
    build_signed_distance_backend,
)
from .sign_cache import LIPSCHITZ_SIGN_CACHE_PROFILE_ID, LipschitzSignCache, SignCachePolicy

__all__ = [
    "ReferenceSignedDistanceBackend",
    "DERIVED_SDF_PROXY_SCHEMA_VERSION",
    "HYBRID_SIGNED_DISTANCE_PROFILE_ID",
    "DerivedWatertightSignProxy",
    "HybridSignedDistanceBackend",
    "ObjectSDFGeometryPolicy",
    "ObjectLocalBVH",
    "SignedDistanceBackend",
    "SignedDistanceError",
    "SignedDistanceQueryResult",
    "SignedDistanceGradientAmbiguityPolicy",
    "SignedDistanceGradientResult",
    "LipschitzSignCache",
    "SignCachePolicy",
    "ANALYTIC_CLOSEST_FEATURE_GRADIENT_ID",
    "AMBIGUITY_POLICY_ID",
    "SPATIAL_FD_FALLBACK_ID",
    "LIPSCHITZ_SIGN_CACHE_PROFILE_ID",
    "analytic_spatial_gradient",
    "ambiguity_reason_counts",
    "build_signed_distance_backend",
    "build_derived_sign_proxy",
    "build_hybrid_signed_distance_backend",
    "local_linearization",
]
