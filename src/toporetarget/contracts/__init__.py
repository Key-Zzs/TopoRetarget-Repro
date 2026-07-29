"""Stable Stage 11 contracts for future datasets, robot hands, and playback."""

from .canonical import (
    CanonicalHOI,
    CanonicalHOIv2,
    CanonicalHOIValidationError,
    load_canonical_hoi,
    migrate_v1_to_v2,
    save_canonical_hoi,
)
from .dataset import (
    DatasetAdapter,
    DatasetAdapterRegistry,
    DatasetCapabilities,
    DatasetDescriptor,
)
from .metrics import MetricRegistry, MetricSpec, MetricType, get_metric_registry
from .migration import generate_stage11_migration_report
from .reference import (
    RobotReferenceV2,
    RobotReferenceValidationError,
    load_robot_reference,
    migrate_reference_v1_to_v2,
    save_robot_reference,
    validate_reference,
)
from .robot import (
    RobotCapabilities,
    RobotHandPlugin,
    RobotHandPluginRegistry,
    RobotReferenceExportProfile,
    get_robot_plugin_registry,
)
from .version import (
    CANONICAL_HOI_V1,
    CANONICAL_HOI_V2,
    DATASET_ADAPTER_V1,
    METRIC_REGISTRY_V1,
    ROBOT_HAND_PLUGIN_V1,
    ROBOT_REFERENCE_V1,
    ROBOT_REFERENCE_V2,
)

__all__ = [
    "CanonicalHOI",
    "CanonicalHOIValidationError",
    "CanonicalHOIv2",
    "load_canonical_hoi",
    "migrate_v1_to_v2",
    "save_canonical_hoi",
    "DatasetAdapter",
    "DatasetAdapterRegistry",
    "DatasetCapabilities",
    "DatasetDescriptor",
    "MetricRegistry",
    "MetricSpec",
    "MetricType",
    "get_metric_registry",
    "generate_stage11_migration_report",
    "RobotCapabilities",
    "RobotHandPlugin",
    "RobotHandPluginRegistry",
    "RobotReferenceExportProfile",
    "get_robot_plugin_registry",
    "RobotReferenceV2",
    "RobotReferenceValidationError",
    "load_robot_reference",
    "migrate_reference_v1_to_v2",
    "save_robot_reference",
    "validate_reference",
    "CANONICAL_HOI_V1",
    "CANONICAL_HOI_V2",
    "DATASET_ADAPTER_V1",
    "METRIC_REGISTRY_V1",
    "ROBOT_HAND_PLUGIN_V1",
    "ROBOT_REFERENCE_V1",
    "ROBOT_REFERENCE_V2",
]
