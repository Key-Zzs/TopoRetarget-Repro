"""Dataset adapter registry for the frozen Stage 11/12 dataset contract."""

from __future__ import annotations

from toporetarget.contracts.dataset import DatasetAdapterRegistry

from .contactpose import ContactPoseAdapterV1
from .dexycb import DexYCBAdapterV1
from .grab import GrabDatasetAdapterV1
from .hocap import HOCapAdapterV1
from .oakink import OakInkAdapterV1


def get_dataset_adapter_registry() -> DatasetAdapterRegistry:
    registry = DatasetAdapterRegistry()
    registry.register("grab", GrabDatasetAdapterV1)
    registry.register("dexycb", DexYCBAdapterV1)
    registry.register("oakink", OakInkAdapterV1)
    registry.register("hocap", HOCapAdapterV1)
    registry.register("contactpose", ContactPoseAdapterV1)
    return registry


__all__ = [
    "ContactPoseAdapterV1",
    "DexYCBAdapterV1",
    "HOCapAdapterV1",
    "OakInkAdapterV1",
    "get_dataset_adapter_registry",
]
