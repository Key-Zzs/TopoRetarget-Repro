"""Reusable support detection, inference, geometry, and validation APIs."""

from .geometry_validation import (
    qualify_geometry,
    validate_hand_table_geometry,
    validate_object_table_geometry,
)
from .physics_validation import (
    build_physics_validation,
    compare_support_counterfactuals,
    summarize_static_support_test,
)
from .planar_inference import (
    detect_stable_pre_contact_interval,
    infer_planar_support,
    normalize_gravity,
    support_normal_from_gravity,
    transform_mesh_trajectory,
)
from .resolver import resolve_support, validate_and_finalize_resolution
from .runtime_support import (
    isaac_rigid_object_config_kwargs,
    table_top_corners,
    write_finite_planar_support_usda,
)
from .source_evidence import (
    NormalizedSourceEvidence,
    SupportEvidenceAdapter,
    evidence_from_sequence_directory,
    normalize_source_evidence,
)
from .types import (
    FinitePlanarSupportProxy,
    GeometryValidation,
    NominalSupportMaterialV1,
    PhysicsValidation,
    StableIntervalResult,
    StablePreContactDetectionContractV1,
    SupportExtentContractV1,
    SupportInterval,
    SupportPatchType,
    SupportPlaneConsistencyGateV1,
    SupportPlaneFit,
    SupportResolutionMode,
    SupportResolutionResult,
    SupportResolutionStatus,
    SupportType,
)

__all__ = [
    "FinitePlanarSupportProxy",
    "GeometryValidation",
    "NominalSupportMaterialV1",
    "NormalizedSourceEvidence",
    "PhysicsValidation",
    "StableIntervalResult",
    "StablePreContactDetectionContractV1",
    "SupportEvidenceAdapter",
    "SupportExtentContractV1",
    "SupportInterval",
    "SupportPatchType",
    "SupportPlaneConsistencyGateV1",
    "SupportPlaneFit",
    "SupportResolutionMode",
    "SupportResolutionResult",
    "SupportResolutionStatus",
    "SupportType",
    "build_physics_validation",
    "compare_support_counterfactuals",
    "detect_stable_pre_contact_interval",
    "evidence_from_sequence_directory",
    "infer_planar_support",
    "isaac_rigid_object_config_kwargs",
    "normalize_gravity",
    "normalize_source_evidence",
    "qualify_geometry",
    "resolve_support",
    "summarize_static_support_test",
    "support_normal_from_gravity",
    "table_top_corners",
    "transform_mesh_trajectory",
    "validate_and_finalize_resolution",
    "validate_hand_table_geometry",
    "validate_object_table_geometry",
    "write_finite_planar_support_usda",
]
