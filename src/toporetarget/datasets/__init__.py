"""Compatibility import path for dataset adapters.

New code should use ``toporetarget.adapters.datasets``.  The historical
``toporetarget.data.adapters`` path remains supported as well.
"""

from toporetarget.adapters.datasets import GrabDatasetAdapterV1, get_dataset_adapter_registry
from toporetarget.data.adapters import (
    FrameRange,
    GrabDatasetAdapter,
    GrabLoadOptions,
    HOIDatasetAdapter,
)

__all__ = [
    "FrameRange",
    "GrabDatasetAdapter",
    "GrabDatasetAdapterV1",
    "GrabLoadOptions",
    "HOIDatasetAdapter",
    "get_dataset_adapter_registry",
]
