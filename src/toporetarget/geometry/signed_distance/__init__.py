"""Signed-distance backends with the repository's positive-outside convention."""

from .base import SignedDistanceBackend, SignedDistanceQueryResult, local_linearization
from .reference import (
    ReferenceSignedDistanceBackend,
    SignedDistanceError,
    build_signed_distance_backend,
)

__all__ = [
    "ReferenceSignedDistanceBackend",
    "SignedDistanceBackend",
    "SignedDistanceError",
    "SignedDistanceQueryResult",
    "build_signed_distance_backend",
    "local_linearization",
]
