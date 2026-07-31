"""DatasetAdapter v1 protocol and capability declaration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from toporetarget.data.schema import HOISequence

from .canonical import CanonicalHOIv2
from .version import DATASET_ADAPTER_V1


@dataclass(frozen=True)
class DatasetCapabilities:
    """Capabilities are declarations, not evidence that a run succeeded."""

    canonical_hoi: bool = False
    contact_annotation: bool = False
    articulated_object: bool = False
    bimanual: bool = False
    body_model: bool = False
    rgb: bool = False
    depth: bool = False

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    version: str = DATASET_ADAPTER_V1
    capabilities: DatasetCapabilities = field(default_factory=DatasetCapabilities)
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": {"name": self.name, "version": self.version},
            "capabilities": self.capabilities.as_dict(),
            "provenance": dict(self.provenance),
        }


class DatasetAdapter(ABC):
    """The only dataset-facing surface used by future pipeline stages."""

    contract_version = DATASET_ADAPTER_V1
    descriptor: DatasetDescriptor

    @abstractmethod
    def discover(self, **kwargs: Any) -> Any:
        """Discover candidate sequences without loading frame geometry."""

    @abstractmethod
    def index(self, **kwargs: Any) -> Any:
        """Build or refresh a disposable dataset index."""

    @abstractmethod
    def describe(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        """Describe one sequence without converting it."""

    @abstractmethod
    def load_sequence(self, sequence: str = "", **kwargs: Any) -> HOISequence:
        """Load one explicit sequence or contiguous clip."""

    @abstractmethod
    def convert_to_canonical(self, sequence: HOISequence, **kwargs: Any) -> CanonicalHOIv2:
        """Convert the loaded sequence to Canonical HOI v2."""

    @abstractmethod
    def validate(self, sequence: HOISequence | CanonicalHOIv2, **kwargs: Any) -> Any:
        """Validate source and canonical semantics, failing closed."""

    @abstractmethod
    def visualize(self, sequence: HOISequence | CanonicalHOIv2, **kwargs: Any) -> Any:
        """Return or write a visualization handle without changing data."""

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def capabilities(self) -> DatasetCapabilities:
        return self.descriptor.capabilities


class DatasetAdapterRegistry:
    """Explicit name-to-factory registry; no dataset-name conditionals."""

    def __init__(self) -> None:
        self._factories: dict[str, Any] = {}

    def register(self, name: str, factory: Any, *, replace: bool = False) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("dataset adapter name must not be empty")
        if key in self._factories and not replace:
            raise ValueError(f"dataset adapter already registered: {key}")
        self._factories[key] = factory

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def create(self, name: str, **kwargs: Any) -> DatasetAdapter:
        key = str(name).strip().lower()
        try:
            adapter = self._factories[key](**kwargs)
        except KeyError as exc:
            raise KeyError(f"unknown dataset adapter {name!r}; choose from {self.names()}") from exc
        if not isinstance(adapter, DatasetAdapter):
            raise TypeError(f"factory {key!r} did not return a DatasetAdapter")
        return adapter

    def describe(self) -> list[dict[str, Any]]:
        return [self._factories[name]().descriptor.as_dict() for name in self.names()]


__all__ = [
    "DATASET_ADAPTER_V1",
    "DatasetAdapter",
    "DatasetAdapterRegistry",
    "DatasetCapabilities",
    "DatasetDescriptor",
]
