"""Physics-stage support provenance, feasibility, and reconstruction APIs."""

from .physical_scene_authority import (
    ContactState,
    PhysicalSceneAuthorityContractV1,
    PhysicalSceneStatus,
    SupportAuthority,
    SupportExpectation,
    admit_physical_scene,
    classify_contact_state,
    resolve_support_expectation,
    validate_runtime_collision_shapes,
    validate_support_geometry,
)
from .physical_scene_authority import (
    support_collision_policy as physical_support_collision_policy,
)
from .support import resolve_support, validate_and_finalize_resolution
from .support_contract import (
    SourceSupportContractV1,
    SupportClassification,
    SupportMode,
    discover_source_support_evidence,
)
from .support_feasibility import build_support_timeline, decide_support_mode

__all__ = [
    "SourceSupportContractV1",
    "SupportClassification",
    "SupportMode",
    "build_support_timeline",
    "decide_support_mode",
    "discover_source_support_evidence",
    "resolve_support",
    "validate_and_finalize_resolution",
    "ContactState",
    "PhysicalSceneAuthorityContractV1",
    "PhysicalSceneStatus",
    "SupportAuthority",
    "SupportExpectation",
    "admit_physical_scene",
    "classify_contact_state",
    "physical_support_collision_policy",
    "resolve_support_expectation",
    "validate_runtime_collision_shapes",
    "validate_support_geometry",
]
