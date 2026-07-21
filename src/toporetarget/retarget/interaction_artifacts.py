"""Atomic Zarr storage for source graphs and interaction evaluations."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import (
    _AsyncReadableGroup,
    direct_zarr3_arrays,
    write_zarr3_group_direct,
)
from toporetarget.utils.hashing import sha256_tree

from .graph_weights import DirectedGraphWeights
from .interaction_graph import (
    INTERACTION_GRAPH_SCHEMA_VERSION,
    InteractionGraphTrajectory,
)

INTERACTION_EVALUATION_SCHEMA_VERSION = "toporetarget.interaction_evaluation.v1"


class InteractionArtifactError(RuntimeError):
    """Raised for invalid, incompatible, or unsafe Stage 8 artifacts."""


def interaction_artifact_hash(path: str | Path) -> str:
    root = Path(path).expanduser()
    if root.is_file():
        return hashlib.sha256(root.read_bytes()).hexdigest()
    if not root.is_dir():
        raise InteractionArtifactError(f"artifact does not exist: {root}")
    digest = hashlib.sha256()
    for name, value in sha256_tree(root).items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _metadata_text(metadata: dict[str, Any]) -> str:
    # Zarr 3 parses root attributes through its async synchronizer.  Keeping
    # the per-frame statistics in a 100+ KiB JSON attribute makes local-store
    # open latency unreliable, so retain the metadata in the same attribute
    # but compress the serialized payload.  The prefix keeps the format
    # self-describing and the reader remains compatible with older plain JSON.
    raw = json.dumps(metadata, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
    if len(raw) < 65536:
        return raw.decode("utf-8")
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    return "zlib+base64:" + encoded


def _parse_metadata(raw: Any) -> dict[str, Any]:
    text = raw if isinstance(raw, str) else str(raw)
    if text.startswith("zlib+base64:"):
        compressed = base64.b64decode(text.removeprefix("zlib+base64:"))
        text = zlib.decompress(compressed).decode("utf-8")
    metadata = json.loads(text)
    if not isinstance(metadata, dict):
        raise InteractionArtifactError("artifact metadata is not a mapping")
    return metadata


def _create_array(group: Any, name: str, data: np.ndarray) -> None:
    value = np.asarray(data)
    chunks = None
    if value.ndim:
        # Large ragged arrays must not be written as thousands of 32-element
        # Zarr chunks: that makes artifact open/read latency dominate graph
        # evaluation on local filesystems.
        # Keep graph ragged arrays (including ``[N, 2]`` edges and
        # ``[N, 3]`` coordinates) in a small number of filesystem objects.
        # The frame-major tensors remain bounded at 32 frames per chunk.
        first_chunk = min(4096, max(1, int(value.shape[0])))
        if value.ndim >= 3:
            first_chunk = min(32, max(1, int(value.shape[0])))
        chunks = (first_chunk,) + tuple(int(size) for size in value.shape[1:])
    try:
        if chunks is None:
            group.create_array(name, data=value, overwrite=True)
        else:
            group.create_array(name, data=value, chunks=chunks, overwrite=True)
    except AttributeError:  # zarr 2.x
        if chunks is None:
            group.create_dataset(name, data=value, overwrite=True)
        else:
            group.create_dataset(name, data=value, chunks=chunks, overwrite=True)


def _write_group(
    zarr: Any,
    temporary: Path,
    schema_version: str,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    del zarr
    write_zarr3_group_direct(
        temporary,
        {"schema_version": schema_version, "metadata_json": _metadata_text(metadata)},
        arrays,
        array_prefix="",
    )


def _atomic_destination(path: str | Path, force: bool) -> tuple[Path, Path]:
    destination = Path(path).expanduser()
    if destination.exists() and not force:
        raise InteractionArtifactError(f"artifact exists; pass --force: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    return destination, temporary


def save_interaction_graph(
    trajectory: InteractionGraphTrajectory, path: str | Path, *, force: bool = False
) -> Path:
    """Publish a graph without a root-level unrecognized metadata sidecar."""

    trajectory.validate()
    destination, temporary = _atomic_destination(path, force)
    metadata = dict(trajectory.metadata)
    metadata["schema_version"] = INTERACTION_GRAPH_SCHEMA_VERSION
    metadata["artifact_type"] = "source_only"
    metadata["array_manifest"] = sorted(trajectory.arrays())
    try:
        import zarr

        _write_group(
            zarr,
            temporary,
            INTERACTION_GRAPH_SCHEMA_VERSION,
            metadata,
            trajectory.arrays(),
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(exc, InteractionArtifactError):
            raise
        raise InteractionArtifactError(f"could not publish graph artifact: {exc}") from exc
    trajectory.source_path = destination
    trajectory.artifact_hash = interaction_artifact_hash(destination)
    return destination


def _read_group(path: str | Path, expected_schema: str) -> tuple[Any, dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.is_dir():
        raise InteractionArtifactError(f"artifact does not exist: {source}")
    try:
        root_metadata = source / "zarr.json"
        root = json.loads(root_metadata.read_text(encoding="utf-8"))
        attributes = root.get("attributes", {})
    except ImportError as exc:  # pragma: no cover
        raise InteractionArtifactError("interaction artifacts require zarr") from exc
    if attributes.get("schema_version") != expected_schema:
        raise InteractionArtifactError(
            f"unsupported artifact schema: {attributes.get('schema_version')!r}"
        )
    raw = attributes.get("metadata_json")
    if raw is None:
        raise InteractionArtifactError("artifact has no metadata_json Zarr attribute")
    metadata = _parse_metadata(raw)

    values = direct_zarr3_arrays(source, metadata.get("array_manifest", []), array_prefix="")
    return _AsyncReadableGroup(None, values), metadata


def _array(group: Any, name: str) -> np.ndarray:
    try:
        return np.asarray(group[name])
    except (KeyError, FileNotFoundError):
        raise InteractionArtifactError(f"artifact missing array: {name}") from None


def load_interaction_graph(path: str | Path) -> InteractionGraphTrajectory:
    group, metadata = _read_group(path, INTERACTION_GRAPH_SCHEMA_VERSION)
    arrays = {name: _array(group, name) for name in metadata.get("array_manifest", [])}
    required = {
        "timestamps",
        "frame_indices",
        "source_vertices",
        "source_laplacian",
        "object_face_indices",
        "object_barycentric",
        "simplices_concat",
        "simplex_offsets",
        "edges_concat",
        "edge_offsets",
        "directed_source_index",
        "directed_destination_index",
        "weights",
        "log_unnormalized",
        "source_distance_squared",
        "directed_row_offsets",
        "directed_frame_offsets",
        "row_sums",
        "frame_valid",
        "status_codes",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise InteractionArtifactError(f"graph artifact missing arrays: {missing}")
    frame_count = int(arrays["source_vertices"].shape[0])
    simplex_offsets = arrays["simplex_offsets"]
    edge_offsets = arrays["edge_offsets"]
    directed_offsets = arrays["directed_frame_offsets"]
    status_names = list(metadata.get("frame_status", ["valid"] * frame_count))
    simplex_frames = [
        arrays["simplices_concat"][simplex_offsets[i] : simplex_offsets[i + 1]]
        for i in range(frame_count)
    ]
    edge_frames = [
        arrays["edges_concat"][edge_offsets[i] : edge_offsets[i + 1]] for i in range(frame_count)
    ]
    directed_frames: list[DirectedGraphWeights] = []
    for index in range(frame_count):
        start, stop = (int(directed_offsets[index]), int(directed_offsets[index + 1]))
        row_offsets = arrays["directed_row_offsets"][index] - start
        directed_frames.append(
            DirectedGraphWeights(
                source_index=arrays["directed_source_index"][start:stop],
                destination_index=arrays["directed_destination_index"][start:stop],
                weights=arrays["weights"][start:stop],
                log_unnormalized=arrays["log_unnormalized"][start:stop],
                distance_squared=arrays["source_distance_squared"][start:stop],
                row_offsets=row_offsets,
                row_sums=arrays["row_sums"][index * 71 : (index + 1) * 71],
            ).validate()
        )
    result = InteractionGraphTrajectory(
        metadata=metadata,
        timestamps=arrays["timestamps"],
        source_vertices=arrays["source_vertices"],
        source_laplacian=arrays["source_laplacian"],
        simplex_frames=simplex_frames,
        edge_frames=edge_frames,
        directed_frames=directed_frames,
        frame_statistics=list(metadata.get("frame_statistics", [])),
        frame_valid=arrays["frame_valid"].astype(bool),
        frame_status=status_names,
        frame_indices=arrays["frame_indices"],
        object_face_indices=arrays["object_face_indices"],
        object_barycentric=arrays["object_barycentric"],
        graph_hashes=list(metadata.get("graph_hashes", [])),
        source_vertex_metadata=list(metadata.get("source_vertex_metadata", [])),
        source_path=Path(path),
        artifact_hash=interaction_artifact_hash(path),
    )
    return result.validate()


def save_interaction_evaluation(evaluation: Any, path: str | Path, *, force: bool = False) -> Path:
    evaluation.validate()
    destination, temporary = _atomic_destination(path, force)
    metadata = dict(evaluation.metadata)
    metadata["schema_version"] = INTERACTION_EVALUATION_SCHEMA_VERSION
    metadata["array_manifest"] = sorted(evaluation.arrays())
    try:
        import zarr

        _write_group(
            zarr,
            temporary,
            INTERACTION_EVALUATION_SCHEMA_VERSION,
            metadata,
            evaluation.arrays(),
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise InteractionArtifactError(f"could not publish evaluation artifact: {exc}") from exc
    evaluation.source_path = destination
    evaluation.artifact_hash = interaction_artifact_hash(destination)
    return destination


def load_interaction_evaluation(path: str | Path) -> Any:
    # Import lazily to keep graph-only commands independent of robot/Torch code.
    from .interaction_evaluation import InteractionEvaluationTrajectory

    group, metadata = _read_group(path, INTERACTION_EVALUATION_SCHEMA_VERSION)
    arrays = {name: _array(group, name) for name in metadata.get("array_manifest", [])}
    return InteractionEvaluationTrajectory.from_arrays(
        metadata, arrays, source_path=Path(path), artifact_hash=interaction_artifact_hash(path)
    )


__all__ = [
    "INTERACTION_EVALUATION_SCHEMA_VERSION",
    "InteractionArtifactError",
    "interaction_artifact_hash",
    "load_interaction_evaluation",
    "load_interaction_graph",
    "save_interaction_evaluation",
    "save_interaction_graph",
]
