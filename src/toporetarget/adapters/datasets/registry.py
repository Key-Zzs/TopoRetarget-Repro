"""Dataset adapter registry with GRAB registered as the first instance."""

from __future__ import annotations

from toporetarget.contracts.dataset import DatasetAdapterRegistry

from .grab import GrabDatasetAdapterV1


def get_dataset_adapter_registry() -> DatasetAdapterRegistry:
    registry = DatasetAdapterRegistry()
    registry.register("grab", GrabDatasetAdapterV1)
    return registry


__all__ = ["get_dataset_adapter_registry"]
