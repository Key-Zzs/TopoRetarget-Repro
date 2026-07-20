"""Independent ``toporetarget.warm_start.v1`` artifact storage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import _async_group_async
from toporetarget.utils.hashing import sha256_tree

WARM_START_SCHEMA_VERSION = "toporetarget.warm_start.v1"


class WarmStartArtifactError(RuntimeError):
    """Raised for invalid, incompatible, or unsafe warm-start artifacts."""


@dataclass
class WarmStartTrajectory:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def schema_version(self) -> str:
        return str(self.metadata.get("schema_version", ""))

    @property
    def frame_count(self) -> int:
        qpos = self.arrays.get("qpos")
        return 0 if qpos is None else int(qpos.shape[0])

    def validate(self) -> WarmStartTrajectory:
        if self.schema_version != WARM_START_SCHEMA_VERSION:
            raise WarmStartArtifactError(f"unsupported warm-start schema: {self.schema_version!r}")
        required = {
            "qpos": (2, 22),
            "base_pose_scene": (3, 4, 4),
            "robot_keypoints_base": (3, 21, 3),
            "robot_keypoints_scene": (3, 21, 3),
            "source_hand_frame_scene": (3, 4, 4),
            "robot_hand_frame_base": (3, 4, 4),
            "source_bone_directions": (3, 20, 3),
            "robot_bone_directions": (3, 20, 3),
            "source_adjacent_features": (3, 15, 3),
            "robot_adjacent_features": (3, 15, 3),
            "pair_residuals": (3, 15, 3),
            "ebone": (1,),
            "temporal_term": (1,),
            "total_objective": (1,),
            "valid_mask": (1,),
        }
        frame_count = self.frame_count
        if frame_count == 0:
            raise WarmStartArtifactError("warm-start qpos is empty")
        for name, shape_tail in required.items():
            if name not in self.arrays:
                raise WarmStartArtifactError(f"warm-start artifact missing array: {name}")
            array = np.asarray(self.arrays[name])
            if array.ndim != len(shape_tail) or tuple(array.shape[1:]) != shape_tail[1:]:
                raise WarmStartArtifactError(f"{name} has invalid shape {array.shape}")
            if array.shape[0] != frame_count:
                raise WarmStartArtifactError(f"{name} frame count mismatch")
        if self.arrays["qpos"].shape[1] != 22:
            raise WarmStartArtifactError("Stage 7 artifact qpos must have 22 columns")
        if not np.all(np.isfinite(self.arrays["qpos"])):
            raise WarmStartArtifactError("qpos contains NaN or Inf")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "arrays": {key: list(value.shape) for key, value in self.arrays.items()},
        }


def _json_metadata(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def artifact_hash(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for name, value in sha256_tree(root).items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def save_warm_start(
    trajectory: WarmStartTrajectory, path: str | Path, *, force: bool = False
) -> Path:
    trajectory.validate()
    destination = Path(path).expanduser()
    if destination.exists() and not force:
        raise WarmStartArtifactError(f"warm-start artifact exists; pass --force: {destination}")
    try:
        import zarr

        from toporetarget.data.storage import _local_store
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise WarmStartArtifactError("warm-start artifacts require the cache extra (zarr)") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        group = zarr.open_group(_local_store(zarr, temporary, read_only=False), mode="w")
        group.attrs["schema_version"] = WARM_START_SCHEMA_VERSION
        group.attrs["metadata_json"] = _json_metadata(trajectory.metadata)
        for name, array in trajectory.arrays.items():
            data = np.asarray(array)
            chunks: tuple[int, ...] | None = None
            if data.ndim > 0:
                chunks = (min(32, int(data.shape[0])),) + tuple(
                    int(size) for size in data.shape[1:]
                )
            try:
                if chunks is None:
                    group.create_array(name, data=data, overwrite=True)
                else:
                    group.create_array(name, data=data, chunks=chunks, overwrite=True)
            except AttributeError:  # zarr 2.x
                legacy_group: Any = group
                if chunks is None:
                    legacy_group.create_dataset(name, data=data, overwrite=True)
                else:
                    legacy_group.create_dataset(name, data=data, chunks=chunks, overwrite=True)
        (temporary / "metadata.json").write_text(
            _json_metadata(trajectory.metadata), encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if isinstance(exc, WarmStartArtifactError):
            raise
        raise WarmStartArtifactError(f"could not publish warm-start artifact: {exc}") from exc
    return destination


def load_warm_start(path: str | Path) -> WarmStartTrajectory:
    source = Path(path).expanduser()
    if not source.is_dir():
        raise WarmStartArtifactError(f"warm-start artifact does not exist: {source}")
    try:
        import zarr

        group = asyncio.run(_async_group_async(zarr, source, mode="r"))
    except ImportError as exc:  # pragma: no cover
        raise WarmStartArtifactError("warm-start artifacts require zarr") from exc
    version = group.attrs.get("schema_version")
    if version != WARM_START_SCHEMA_VERSION:
        raise WarmStartArtifactError(f"unsupported warm-start schema: {version!r}")
    raw = group.attrs.get("metadata_json")
    if raw is None:
        metadata_path = source / "metadata.json"
        if not metadata_path.is_file():
            raise WarmStartArtifactError("warm-start artifact has no metadata")
        raw = metadata_path.read_text(encoding="utf-8")
    metadata = json.loads(raw if isinstance(raw, str) else str(raw))
    if not isinstance(metadata, dict):
        raise WarmStartArtifactError("warm-start metadata is not a mapping")

    async def read_arrays() -> dict[str, np.ndarray]:
        names = [name async for name in group.array_keys()]
        result: dict[str, np.ndarray] = {}
        for name in names:
            array = await group.getitem(name)
            result[name] = np.asarray(await array.getitem(slice(None)))
        return result

    arrays = asyncio.run(read_arrays())
    result = WarmStartTrajectory(metadata, arrays)
    return result.validate()


__all__ = [
    "WARM_START_SCHEMA_VERSION",
    "WarmStartArtifactError",
    "WarmStartTrajectory",
    "artifact_hash",
    "load_warm_start",
    "save_warm_start",
]
