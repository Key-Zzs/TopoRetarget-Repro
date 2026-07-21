"""Optional Zarr cache for lossless semantic HOI round-trips."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

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


ZarrMode = Literal["r", "r+", "a", "w", "w-"]


async def _async_group_async(zarr: Any, path: Path, *, mode: ZarrMode) -> Any:
    """Open Zarr 3 through its async API.

    The synchronous Zarr 3 bridge can stall on this Python 3.13 workstation
    even for a small local store.  The repository's local store already
    provides synchronous file methods, so running the async API directly is
    deterministic and keeps the cache format unchanged.  Zarr 2 falls back
    to its regular synchronous API.
    """

    if not hasattr(getattr(zarr, "api", None), "asynchronous"):
        return zarr.open_group(_local_store(zarr, path, read_only=mode == "r"), mode=mode)
    from zarr.api.asynchronous import open_group as async_open_group

    return await async_open_group(_local_store(zarr, path, read_only=mode == "r"), mode=mode)


def _async_group(zarr: Any, path: Path, *, mode: ZarrMode) -> Any:
    return asyncio.run(_async_group_async(zarr, path, mode=mode))


class _AsyncReadableGroup:
    """Minimal synchronous adapter for eager manifest decoding."""

    def __init__(self, group: Any, arrays: dict[str, np.ndarray] | None = None) -> None:
        self.group = group
        self.arrays = arrays

    def __getitem__(self, key: str) -> np.ndarray:
        if self.arrays is not None:
            try:
                return self.arrays[key]
            except KeyError as exc:
                raise KeyError(key) from exc

        async def read_once() -> np.ndarray:
            array = await self.group.getitem(key)
            return np.asarray(await array.getitem(slice(None)))

        return asyncio.run(read_once())


def _read_zarr3_array_direct(
    root: Path, relative: str, *, array_prefix: str = "arrays"
) -> np.ndarray:
    """Read a repository-written Zarr v3 array without the async codec bridge.

    The managed filesystem can stall inside Zarr's asynchronous chunk decoder
    even though the local chunk and metadata files are readable.  Stage 5
    caches are written with the small ``bytes`` + ``zstd`` codec pipeline, so
    decoding those chunks directly preserves the cache bytes and schema while
    avoiding the stalled bridge.
    """

    array_root = root / array_prefix / relative if array_prefix else root / relative
    metadata_path = array_root / "zarr.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("zarr_format") != 3 or metadata.get("node_type") != "array":
        raise StorageError(f"unsupported direct Zarr array: {array_root}")
    shape = tuple(int(item) for item in metadata["shape"])
    chunk_shape = tuple(
        int(item) for item in metadata["chunk_grid"]["configuration"]["chunk_shape"]
    )
    dtype = np.dtype(str(metadata["data_type"]))
    fill_value = metadata.get("fill_value", 0)
    result = np.full(shape, fill_value, dtype=dtype)
    chunk_counts = tuple(
        (size + chunk - 1) // chunk for size, chunk in zip(shape, chunk_shape, strict=True)
    )
    codecs = tuple(item.get("name") for item in metadata.get("codecs", ()))
    for chunk_index in itertools.product(*(range(count) for count in chunk_counts)):
        starts = tuple(index * chunk for index, chunk in zip(chunk_index, chunk_shape, strict=True))
        stops = tuple(
            min(start + chunk, size)
            for start, chunk, size in zip(starts, chunk_shape, shape, strict=True)
        )
        chunk_file = array_root / "c" / "/".join(str(item) for item in chunk_index)
        if not chunk_file.is_file():
            continue
        encoded = chunk_file.read_bytes()
        for codec in reversed(codecs):
            if codec == "zstd":
                from numcodecs import Zstd

                encoded = Zstd().decode(encoded)
            elif codec != "bytes":
                raise StorageError(f"unsupported direct Zarr codec {codec!r}: {array_root}")
        actual_shape = tuple(stop - start for start, stop in zip(starts, stops, strict=True))
        expected_count = int(np.prod(actual_shape, dtype=np.int64))
        values = np.frombuffer(encoded, dtype=dtype, count=expected_count)
        if values.size != expected_count:
            raise StorageError(f"truncated Zarr chunk: {chunk_file}")
        result[tuple(slice(start, stop) for start, stop in zip(starts, stops, strict=True))] = (
            values.reshape(actual_shape)
        )
    return result


class _DirectReadableGroup:
    """Manifest decoder backed by direct local Zarr v3 chunk reads."""

    def __init__(self, root: Path, *, array_prefix: str = "arrays") -> None:
        self.root = root
        self.array_prefix = array_prefix

    def __getitem__(self, key: str) -> np.ndarray:
        relative = key.removeprefix(f"{self.array_prefix}/") if self.array_prefix else key
        return _read_zarr3_array_direct(self.root, relative, array_prefix=self.array_prefix)


def direct_zarr3_arrays(
    root: str | Path, names: Iterable[str], *, array_prefix: str = "arrays"
) -> dict[str, np.ndarray]:
    """Read named repository-written Zarr v3 arrays through local chunks."""

    source = Path(root)
    return {
        str(name): _read_zarr3_array_direct(source, str(name), array_prefix=array_prefix)
        for name in names
    }


def write_zarr3_group_direct(
    root: str | Path,
    attributes: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    array_prefix: str = "arrays",
) -> None:
    """Write the repository's small local Zarr v3 format without the async bridge."""

    from numcodecs import Zstd

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "zarr.json").write_text(
        json.dumps(
            {
                "attributes": attributes,
                "zarr_format": 3,
                "node_type": "group",
                "consolidated_metadata": None,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    compressor = Zstd(level=0)
    for name, value in arrays.items():
        data = np.asarray(value)
        if data.ndim == 0 or any(size == 0 for size in data.shape):
            raise StorageError(f"direct Zarr writer requires non-empty arrays: {name}")
        chunk_shape = data.shape
        array_root = (
            destination / array_prefix / str(name) if array_prefix else destination / str(name)
        )
        chunk_root = array_root / "c"
        chunk_root.mkdir(parents=True, exist_ok=True)
        encoded = compressor.encode(np.ascontiguousarray(data).tobytes(order="C"))
        chunk_index = tuple(0 for _ in data.shape)
        chunk_path = chunk_root / "/".join(str(index) for index in chunk_index)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_bytes(encoded)
        (array_root / "zarr.json").write_text(
            json.dumps(
                {
                    "shape": list(data.shape),
                    "data_type": data.dtype.str,
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": list(chunk_shape)},
                    },
                    "chunk_key_encoding": {
                        "name": "default",
                        "configuration": {"separator": "/"},
                    },
                    "fill_value": 0,
                    "codecs": [
                        {"name": "bytes", "configuration": {"endian": "little"}},
                        {
                            "name": "zstd",
                            "configuration": {"level": 0, "checksum": False},
                        },
                    ],
                    "attributes": {},
                    "zarr_format": 3,
                    "node_type": "array",
                    "storage_transformers": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )


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
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = _encode(sequence, arrays, "sequence")

    write_zarr3_group_direct(
        destination,
        {
            "schema_version": sequence.metadata.schema_version,
            "metadata_json": json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        },
        arrays,
    )
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
    metadata_file = source / "metadata.json"
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        schema_version = metadata.get("schema_version")
        manifest = metadata.get("manifest")
        if schema_version != "toporetarget.hoi.v1" or not isinstance(manifest, dict):
            raise StorageError(f"unsupported cache schema or manifest: {source}")
        group: Any = _DirectReadableGroup(source)
    else:
        group = _async_group(zarr, source, mode="r")
        schema_version = group.attrs.get("schema_version")
        if schema_version != "toporetarget.hoi.v1":
            raise StorageError(f"unsupported cache schema version: {schema_version!r}")
        manifest_json = group.attrs.get("metadata_json")
        if manifest_json is None:
            raise StorageError(f"cache has no metadata manifest: {source}")
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
