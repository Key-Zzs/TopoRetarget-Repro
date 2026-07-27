"""GRAB/Arti-MANO quality experiment orchestration.

The quality experiment is deliberately layered on top of the frozen Stage 5--10
artifacts.  Paper-core solver profiles remain untouched; quality extensions are
versioned and labelled as engineering or paper-external diagnostics.
"""

from .schema import (
    CONTACTPOSE_STATUS,
    EXPERIMENT_ID,
    QUALITY_SCHEMA_VERSION,
    ClipSpec,
    QualityExperimentError,
)

__all__ = [
    "CONTACTPOSE_STATUS",
    "EXPERIMENT_ID",
    "QUALITY_SCHEMA_VERSION",
    "ClipSpec",
    "QualityExperimentError",
]
