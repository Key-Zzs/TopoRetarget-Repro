"""Optional Zarr cache for lossless semantic HOI round-trips."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import (
    ArticulatedObjectTrack,
    ArticulatedPartTrack,
    ContactTrack,
    HandTrack,
    HOISequence,
    KeypointTrack,
    ManoParameterTrack,
    MeshDefinition,
    PoseTrack,
    ProvenanceRecord,
    RigidObjectTrack,
    SequenceMetadata,
)


class StorageError(RuntimeError):
    """Raised for unavailable optional cache support or invalid cache data."""


_TYPE_REGISTRY = {
    cls.__name__: cls
    for cls in (
        ArticulatedObjectTrack,
        ArticulatedPartTrack,
        ContactTrack,
        HandTrack,
        HOISequence,
        KeypointTrack,
        ManoParameterTrack,
        MeshDefinition,
        PoseTrack,
        ProvenanceRecord,
        RigidObjectTrack,
        SequenceMetadata,
    )
}


def _require_zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise StorageError(
            "Zarr cache support is optional; install with `pip install -e '.[cache]'`."
        ) from exc
    return zarr


def _local_store(zarr: Any, path: Path, *, read_only: bool) -> Any:
    """Return a local store whose async methods do not spawn file threads.

    Zarr 3's default ``LocalStore`` delegates every read to
    ``asyncio.to_thread``.  That is normally fine, but it can deadlock under
    managed filesystems whose thread wrappers do not complete.  The cache
    format itself is unchanged; this adapter only performs the same local
    file operations synchronously inside Zarr's already-synchronous bridge.
    """

    base = getattr(getattr(zarr, "storage", None), "LocalStore", None)
    if base is None:  # Zarr 2.x accepts a path directly.
        return str(path)

    class DirectLocalStore(base):  # type: ignore[misc, valid-type]
        async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
            return self.get_sync(key, prototype=prototype, byte_range=byte_range)

        async def get_partial_values(self, prototype: Any, key_ranges: Any) -> list[Any]:
            return [
                self.get_sync(key, prototype=prototype, byte_range=byte_range)
                for key, byte_range in key_ranges
            ]

        async def set(self, key: str, value: Any) -> None:
            self.set_sync(key, value)

        async def set_if_not_exists(self, key: str, value: Any) -> None:
            self._ensure_open_sync()
            self._check_writable()
            path = self.root / key
            if not path.exists():
                self.set_sync(key, value)

        async def delete(self, key: str) -> None:
            self.delete_sync(key)

        async def exists(self, key: str) -> bool:
            self._ensure_open_sync()
            return (self.root / key).is_file()

    return DirectLocalStore(path, read_only=read_only)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported metadata value for cache: {type(value).__name__}")


def _encode(value: Any, arrays: dict[str, np.ndarray], path: str) -> Any:
    if isinstance(value, np.ndarray):
        arrays[path] = value
        return {"__array__": path}
    if dataclasses.is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            "fields": {
                item.name: _encode(getattr(value, item.name), arrays, f"{path}/{item.name}")
                for item in dataclasses.fields(value)
            },
        }
    if isinstance(value, dict):
        return {str(key): _encode(item, arrays, f"{path}/{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, arrays, f"{path}/{index}") for index, item in enumerate(value)]
    return _json_safe(value)


def _decode(value: Any, group: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item, group) for item in value]
    if not isinstance(value, dict):
        return value
    if "__array__" in value:
        return np.asarray(group[f"arrays/{value['__array__']}"], dtype=None)
    if "__path__" in value:
        return Path(value["__path__"])
    if "__type__" in value:
        cls = _TYPE_REGISTRY.get(value["__type__"])
        if cls is None:
            raise StorageError(f"unsupported cached dataclass: {value['__type__']}")
        fields = {key: _decode(item, group) for key, item in value.get("fields", {}).items()}
        return cls(**fields)
    return {key: _decode(item, group) for key, item in value.items()}


def _chunks(array: np.ndarray) -> tuple[int, ...] | None:
    if array.ndim == 0:
        return None
    return (min(64, array.shape[0]),) + tuple(array.shape[1:])


def save_hoi_sequence(sequence: HOISequence, path: str | Path) -> Path:
    """Write one explicitly requested sequence to a Zarr directory."""

    sequence.validate()
    zarr = _require_zarr()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = _encode(sequence, arrays, "sequence")
    group = zarr.open_group(_local_store(zarr, destination, read_only=False), mode="w")
    group.attrs["schema_version"] = sequence.metadata.schema_version
    group.attrs["metadata_json"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    for name, array in arrays.items():
        data = np.asarray(array)
        dataset_name = f"arrays/{name}"
        chunks = _chunks(data)
        try:
            group.create_array(dataset_name, data=data, chunks=chunks, overwrite=True)
        except AttributeError:  # zarr 2.x
            group.create_dataset(dataset_name, data=data, chunks=chunks, overwrite=True)
    (destination / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": sequence.metadata.schema_version,
                "manifest": manifest,
                "array_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def load_hoi_sequence(path: str | Path) -> HOISequence:
    """Load one cached sequence and validate its schema version and contents."""

    zarr = _require_zarr()
    source = Path(path)
    if not source.exists():
        raise StorageError(f"HOI cache does not exist: {source}")
    group = zarr.open_group(_local_store(zarr, source, read_only=True), mode="r")
    schema_version = group.attrs.get("schema_version")
    if schema_version != "toporetarget.hoi.v1":
        raise StorageError(f"unsupported cache schema version: {schema_version!r}")
    manifest_json = group.attrs.get("metadata_json")
    if manifest_json is None:
        metadata_file = source / "metadata.json"
        if not metadata_file.is_file():
            raise StorageError(f"cache has no metadata manifest: {source}")
        manifest_json = json.loads(metadata_file.read_text(encoding="utf-8"))["manifest"]
    manifest = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json
    result = _decode(manifest, group)
    if not isinstance(result, HOISequence):
        raise StorageError("cache manifest root is not HOISequence")
    result.validate()
    return result


def open_hoi_sequence(path: str | Path) -> HOISequence:
    """Open a cached sequence; this function is intentionally eager at sequence scope."""

    return load_hoi_sequence(path)


__all__ = [
    "StorageError",
    "load_hoi_sequence",
    "open_hoi_sequence",
    "save_hoi_sequence",
]
