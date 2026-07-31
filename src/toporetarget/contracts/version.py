"""Version identifiers for the Stage 11 frozen contracts.

These identifiers are deliberately independent from implementation versions.  A
solver or an adapter may evolve internally without changing the serialized
contract it consumes or produces.
"""

from __future__ import annotations

CANONICAL_HOI_V1 = "toporetarget.hoi.v1"
CANONICAL_HOI_V2 = "toporetarget.hoi.v2"
DATASET_ADAPTER_V1 = "toporetarget.dataset_adapter.v1"
ROBOT_HAND_PLUGIN_V1 = "toporetarget.robot_hand_plugin.v1"
ROBOT_REFERENCE_V1 = "toporetarget.robot_reference.v1"
ROBOT_REFERENCE_V2 = "toporetarget.robot_reference.v2"
METRIC_REGISTRY_V1 = "toporetarget.metric_registry.v1"

SUPPORTED_CANONICAL_HOI_VERSIONS = (CANONICAL_HOI_V1, CANONICAL_HOI_V2)

__all__ = [
    "CANONICAL_HOI_V1",
    "CANONICAL_HOI_V2",
    "DATASET_ADAPTER_V1",
    "METRIC_REGISTRY_V1",
    "ROBOT_HAND_PLUGIN_V1",
    "ROBOT_REFERENCE_V1",
    "ROBOT_REFERENCE_V2",
    "SUPPORTED_CANONICAL_HOI_VERSIONS",
]
