"""Canonical HOI v2 and the lossless Stage 2--10 compatibility bridge.

``HOISequence`` remains the storage model for existing v1 caches.  The v2
class is a versioned facade over the same fields, so migration never rewrites
or mutates a historical cache.  Articulated objects are represented by the
existing ``articulated_objects`` field and are intentionally optional for the
rigid GRAB lane.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from toporetarget.data.schema import (
    HOISequence,
    HOIValidationError,
)
from toporetarget.data.storage import load_hoi_sequence, save_hoi_sequence

from .version import CANONICAL_HOI_V1, CANONICAL_HOI_V2


class CanonicalHOIValidationError(ValueError):
    """Raised when a Canonical HOI v2 payload is invalid."""


@dataclass
class CanonicalHOIv2(HOISequence):
    """Versioned canonical hand-object sequence.

    The inherited field layout is intentionally the same as the established
    v1 model.  This makes all rigid Stage 5--10 data available without a
    lossy reshape while leaving room for ``articulated_objects`` in v2.
    """

    def __post_init__(self) -> None:
        self.metadata = replace(self.metadata, schema_version=CANONICAL_HOI_V2)

    @classmethod
    def from_v1(cls, sequence: HOISequence, *, copy_arrays: bool = False) -> CanonicalHOIv2:
        """Wrap a v1 sequence without changing its values or source cache."""

        sequence.validate()
        if copy_arrays:
            value = copy.deepcopy(sequence)
        else:
            value = sequence
        return cls(
            metadata=replace(value.metadata, schema_version=CANONICAL_HOI_V2),
            hands=value.hands,
            rigid_objects=value.rigid_objects,
            articulated_objects=value.articulated_objects,
            contacts=value.contacts,
        )

    def to_v1(self, *, copy_arrays: bool = False) -> HOISequence:
        """Return the legacy shape for existing readers and writers."""

        value: CanonicalHOIv2 = copy.deepcopy(self) if copy_arrays else self
        return HOISequence(
            metadata=replace(value.metadata, schema_version=CANONICAL_HOI_V1),
            hands=value.hands,
            rigid_objects=value.rigid_objects,
            articulated_objects=value.articulated_objects,
            contacts=value.contacts,
        )

    def validate(self, *, raise_on_error: bool = True) -> list[str]:
        """Validate v2 metadata using the unchanged v1 field-level rules."""

        try:
            errors = self.to_v1().validate(raise_on_error=False)
        except (HOIValidationError, ValueError) as exc:
            errors = [str(exc)]
        if errors and raise_on_error:
            raise CanonicalHOIValidationError("; ".join(errors))
        return errors

    def as_dict(self) -> dict[str, Any]:
        """Return a small schema-level summary suitable for reports."""

        return {
            "schema_version": CANONICAL_HOI_V2,
            "dataset": self.metadata.dataset_name,
            "sequence_id": self.metadata.sequence_id,
            "num_frames": self.num_frames,
            "fps": self.metadata.native_fps,
            "hands": [{"hand_id": hand.hand_id, "side": hand.side} for hand in self.hands],
            "objects": [{"object_id": obj.object_id, "type": "rigid"} for obj in self.rigid_objects]
            + [
                {
                    "object_id": obj.object_id,
                    "type": "articulated",
                    "links": [part.part_id for part in obj.parts],
                }
                for obj in self.articulated_objects
            ],
            "contacts": len(self.contacts),
            "provenance": {
                "source_dataset": self.metadata.provenance.source_dataset,
                "source_sequence": self.metadata.provenance.source_sequence,
                "source_hash": self.metadata.provenance.source_hash,
                "adapter_name": self.metadata.provenance.adapter_name,
                "adapter_version": self.metadata.provenance.adapter_version,
            },
        }


CanonicalHOI = CanonicalHOIv2


def migrate_v1_to_v2(
    source: HOISequence | CanonicalHOIv2 | str | Path,
    *,
    copy_arrays: bool = False,
) -> CanonicalHOIv2:
    """Migrate an in-memory sequence or an existing v1 cache to v2."""

    if isinstance(source, CanonicalHOIv2):
        source.validate()
        return CanonicalHOIv2.from_v1(source.to_v1(), copy_arrays=copy_arrays)
    if isinstance(source, HOISequence):
        return CanonicalHOIv2.from_v1(source, copy_arrays=copy_arrays)
    return CanonicalHOIv2.from_v1(load_hoi_sequence(source), copy_arrays=copy_arrays)


def save_canonical_hoi(sequence: CanonicalHOIv2, path: str | Path) -> Path:
    """Write a v2 marker plus the lossless legacy payload.

    The underlying field encoding is intentionally delegated to the existing
    v1 writer.  Old readers can still open the payload, while the marker makes
    the new contract explicit to v2 readers.
    """

    sequence.validate()
    destination = save_hoi_sequence(sequence.to_v1(copy_arrays=False), path)
    (destination / "canonical_v2.json").write_text(
        json.dumps(
            {
                "schema_version": CANONICAL_HOI_V2,
                "storage_schema_version": CANONICAL_HOI_V1,
                "migration": "lossless_wrapper",
                "summary": sequence.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_canonical_hoi(path: str | Path) -> CanonicalHOIv2:
    """Load either a v2-marked cache or an unchanged historical v1 cache."""

    source = Path(path)
    sequence = load_hoi_sequence(source)
    marker = source / "canonical_v2.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CANONICAL_HOI_V2:
            raise CanonicalHOIValidationError(f"unsupported canonical marker: {source}")
    return CanonicalHOIv2.from_v1(sequence)


__all__ = [
    "CANONICAL_HOI_V1",
    "CANONICAL_HOI_V2",
    "CanonicalHOI",
    "CanonicalHOIValidationError",
    "CanonicalHOIv2",
    "load_canonical_hoi",
    "migrate_v1_to_v2",
    "save_canonical_hoi",
]
