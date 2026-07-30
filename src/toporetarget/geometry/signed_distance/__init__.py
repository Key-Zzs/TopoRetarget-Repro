"""Signed-distance backends with the repository's positive-outside convention."""

from .base import SignedDistanceBackend, SignedDistanceQueryResult, local_linearization
from .batched_exact import (
    BatchedOriginalMeshProximityBackend,
    ObjectLocalProximityContext,
    ReferenceFaithfulSignedDistanceBackend,
)
from .derived_proxy import (
    DERIVED_SDF_PROXY_SCHEMA_VERSION,
    HYBRID_SIGNED_DISTANCE_PROFILE_ID,
    DerivedWatertightSignProxy,
    HybridSignedDistanceBackend,
    ObjectSDFGeometryPolicy,
    build_derived_sign_proxy,
    build_hybrid_signed_distance_backend,
)
from .reference import (
    ReferenceSignedDistanceBackend,
    SignedDistanceError,
    build_signed_distance_backend,
)
from .signed_grid import (
    GRID_SCHEMA_VERSION,
    OriginalMeshSignedGridSDFBackend,
    grid_resolution_from_profile,
)

__all__ = [
    "ReferenceSignedDistanceBackend",
    "BatchedOriginalMeshProximityBackend",
    "ObjectLocalProximityContext",
    "ReferenceFaithfulSignedDistanceBackend",
    "OriginalMeshSignedGridSDFBackend",
    "GRID_SCHEMA_VERSION",
    "DERIVED_SDF_PROXY_SCHEMA_VERSION",
    "HYBRID_SIGNED_DISTANCE_PROFILE_ID",
    "DerivedWatertightSignProxy",
    "HybridSignedDistanceBackend",
    "ObjectSDFGeometryPolicy",
    "SignedDistanceBackend",
    "SignedDistanceError",
    "SignedDistanceQueryResult",
    "build_signed_distance_backend",
    "build_derived_sign_proxy",
    "build_hybrid_signed_distance_backend",
    "local_linearization",
    "grid_resolution_from_profile",
]
