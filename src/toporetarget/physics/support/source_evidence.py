"""Generic source-support evidence and adapter boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..support_contract import discover_source_support_evidence
from .types import SupportType


class SupportEvidenceAdapter(Protocol):
    """Dataset adapter hook used by the resolver.

    Adapters may return a mapping or an object with the same named attributes.
    They must not create fallback geometry when source evidence is absent.
    """

    def get_support_evidence(self, sequence: str) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class NormalizedSourceEvidence:
    explicit: bool = False
    recovered: bool = False
    explicit_validated: bool = False
    recovered_validated: bool = False
    explicit_metadata: dict[str, object] = field(default_factory=dict)
    recovered_assets: tuple[dict[str, object], ...] = ()
    validation: dict[str, object] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def has_source_support(self) -> bool:
        return self.explicit_validated or self.recovered_validated

    @property
    def support_type(self) -> SupportType | None:
        if self.explicit_validated:
            return SupportType.SOURCE_EXPLICIT_SUPPORT
        if self.recovered_validated:
            return SupportType.SOURCE_RECOVERED_SUPPORT
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "explicit": self.explicit,
            "recovered": self.recovered,
            "explicit_validated": self.explicit_validated,
            "recovered_validated": self.recovered_validated,
            "explicit_metadata": self.explicit_metadata,
            "recovered_assets": list(self.recovered_assets),
            "validation": self.validation,
            "provenance": self.provenance,
        }


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {
        name: getattr(value, name)
        for name in (
            "explicit",
            "recovered",
            "explicit_validated",
            "recovered_validated",
            "explicit_metadata",
            "recovered_assets",
            "validation",
            "provenance",
        )
        if hasattr(value, name)
    }


def normalize_source_evidence(value: object | None) -> NormalizedSourceEvidence:
    """Normalize adapter output without interpreting generic metadata as proof."""

    if value is None:
        return NormalizedSourceEvidence(
            provenance={"source": "no_adapter_evidence", "source_support": "not_provided"}
        )
    raw = _mapping(value)
    explicit_metadata = raw.get("explicit_metadata", {})
    recovered_assets = raw.get("recovered_assets", raw.get("assets", ()))
    validation = raw.get("validation", {})
    provenance = raw.get("provenance", {})
    if not isinstance(explicit_metadata, Mapping):
        raise TypeError("SUPPORT_EVIDENCE_EXPLICIT_METADATA_MUST_BE_MAPPING")
    if not isinstance(validation, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError("SUPPORT_EVIDENCE_VALIDATION_PROVENANCE_MUST_BE_MAPPING")
    if not isinstance(recovered_assets, (list, tuple)):
        raise TypeError("SUPPORT_EVIDENCE_ASSETS_MUST_BE_SEQUENCE")
    explicit = bool(raw.get("explicit", raw.get("source_explicit", False)))
    recovered = bool(raw.get("recovered", raw.get("source_recovered", False)))
    explicit_validated = bool(
        raw.get("explicit_validated", explicit and validation.get("explicit", False))
    )
    recovered_validated = bool(
        raw.get("recovered_validated", recovered and validation.get("recovered", False))
    )
    if explicit_validated and not explicit:
        raise ValueError("SUPPORT_EVIDENCE_EXPLICIT_VALIDATION_WITHOUT_EXPLICIT_SOURCE")
    if recovered_validated and not recovered:
        raise ValueError("SUPPORT_EVIDENCE_RECOVERED_VALIDATION_WITHOUT_RECOVERED_SOURCE")
    return NormalizedSourceEvidence(
        explicit=explicit,
        recovered=recovered,
        explicit_validated=explicit_validated,
        recovered_validated=recovered_validated,
        explicit_metadata=dict(explicit_metadata),
        recovered_assets=tuple(
            dict(item) if isinstance(item, Mapping) else {"value": item}
            for item in recovered_assets
        ),
        validation=dict(validation),
        provenance=dict(provenance),
    )


def evidence_from_sequence_directory(sequence_dir: Any) -> NormalizedSourceEvidence:
    """Audit an existing source directory; no asset is downloaded or synthesized."""

    discovered = discover_source_support_evidence(sequence_dir)
    candidates = discovered.get("source_scene_geometry_candidates", [])
    metadata_hits = discovered.get("metadata_support_hits", [])
    assets = tuple(item for item in candidates if isinstance(item, Mapping))
    return normalize_source_evidence(
        {
            "explicit": bool(metadata_hits),
            "recovered": bool(assets),
            "explicit_validated": False,
            "recovered_validated": False,
            "explicit_metadata": {"hits": metadata_hits},
            "recovered_assets": list(assets),
            "validation": {
                "explicit": False,
                "recovered": False,
                "reason": "discovery_is_not_geometry_validation",
            },
            "provenance": discovered,
        }
    )


def call_support_evidence_adapter(adapter: object, sequence: str) -> object:
    """Call the canonical adapter hook, failing loudly on unsupported adapters."""

    method = getattr(adapter, "get_support_evidence", None)
    if not callable(method):
        raise TypeError("SUPPORT_ADAPTER_MISSING_GET_SUPPORT_EVIDENCE")
    return method(sequence)


__all__ = [
    "NormalizedSourceEvidence",
    "SupportEvidenceAdapter",
    "call_support_evidence_adapter",
    "evidence_from_sequence_directory",
    "normalize_source_evidence",
]
