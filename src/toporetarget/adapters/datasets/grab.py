"""GRAB's first DatasetAdapter v1 instance.

The Stage 5 adapter remains the implementation of loading and conversion.  A
small facade supplies the frozen discovery/index/validation/visualization
surface without changing any GRAB parsing or MANO numerics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from toporetarget.contracts.canonical import CanonicalHOIv2
from toporetarget.contracts.dataset import (
    DatasetAdapter,
    DatasetCapabilities,
    DatasetDescriptor,
)
from toporetarget.data.adapters.grab import (
    GrabAdapterError,
    GrabDatasetAdapter,
    GrabLoadOptions,
)
from toporetarget.data.indexes.grab import build_grab_index, load_grab_index
from toporetarget.data.schema import HOISequence


class GrabDatasetAdapterV1(GrabDatasetAdapter, DatasetAdapter):
    """Contract facade for the existing generic GRAB adapter."""

    contract_version = "toporetarget.dataset_adapter.v1"
    descriptor = DatasetDescriptor(
        name="grab",
        capabilities=DatasetCapabilities(
            canonical_hoi=True,
            contact_annotation=True,
            articulated_object=False,
            bimanual=True,
            body_model=True,
            rgb=False,
            depth=False,
        ),
        provenance={
            "source_contract": "toporetarget.data.adapters.grab.GrabDatasetAdapter",
            "native_time": True,
            "no_temporal_resampling": True,
        },
    )

    def discover(self, **kwargs: Any) -> list[dict[str, Any]]:
        index = kwargs.get("index", self.index_path)
        if index is None:
            return []
        return load_grab_index(index)

    def index(self, **kwargs: Any) -> dict[str, Any]:
        return build_grab_index(
            grab_root=kwargs.get("grab_root", self.grab_root_override),
            output=kwargs.get("output", self.index_path or ".local/index/grab"),
            hash_files=bool(kwargs.get("hash_files", False)),
            discovery_report=kwargs.get("discovery_report"),
        )

    def describe(self, sequence: str = "", **kwargs: Any) -> dict[str, Any]:
        return self.describe_sequence(sequence, **kwargs)

    def convert_to_canonical(
        self, sequence: HOISequence | CanonicalHOIv2, **kwargs: Any
    ) -> CanonicalHOIv2:
        if isinstance(sequence, CanonicalHOIv2):
            sequence.validate()
            return sequence
        return CanonicalHOIv2.from_v1(sequence, copy_arrays=bool(kwargs.get("copy_arrays", False)))

    def validate(self, sequence: HOISequence | CanonicalHOIv2, **kwargs: Any) -> Any:
        if isinstance(sequence, CanonicalHOIv2):
            return {
                "schema_version": sequence.metadata.schema_version,
                "errors": sequence.validate(),
            }
        return self.validate_sequence(sequence, **kwargs)

    def visualize(
        self,
        sequence: HOISequence | CanonicalHOIv2,
        *,
        output: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a viewer-neutral handle; existing HTML viewers stay unchanged."""

        canonical = self.convert_to_canonical(sequence)
        handle = {
            "status": "ready",
            "schema_version": canonical.metadata.schema_version,
            "dataset": self.descriptor.name,
            "sequence_id": canonical.metadata.sequence_id,
            "num_frames": canonical.num_frames,
            "viewer": "use existing toporetarget data visualize command",
        }
        if output is not None:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(handle, indent=2, sort_keys=True) + "\n")
            handle["output"] = str(destination)
        return handle


__all__ = ["GrabAdapterError", "GrabDatasetAdapterV1", "GrabLoadOptions"]
