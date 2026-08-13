"""Validated serialization of the authored C.1 runtime convex proxies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    return hashlib.sha256(points.tobytes() + triangles.tobytes()).hexdigest()


@dataclass(frozen=True)
class ConvexProxyGeometry:
    shape_id: str
    body_name: str
    geometry_type: str
    vertices: np.ndarray
    faces: np.ndarray
    local_pose_xyz_wxyz: np.ndarray
    scale_xyz: np.ndarray
    source_asset_path: str
    source_asset_sha256: str
    generated_asset_path: str
    generated_asset_sha256: str

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        pose = np.asarray(self.local_pose_xyz_wxyz, dtype=np.float64)
        scale = np.asarray(self.scale_xyz, dtype=np.float64)
        if self.geometry_type != "convex_hull":
            raise ValueError(f"unsupported formal runtime geometry: {self.geometry_type}")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError(f"invalid convex vertices for {self.shape_id}")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
            raise ValueError(f"invalid convex faces for {self.shape_id}")
        if int(faces.min(initial=0)) < 0 or int(faces.max(initial=0)) >= len(vertices):
            raise ValueError(f"invalid convex face index for {self.shape_id}")
        if pose.shape != (7,) or scale.shape != (3,) or np.any(scale <= 0.0):
            raise ValueError(f"invalid local transform or scale for {self.shape_id}")
        if not all(np.isfinite(value).all() for value in (vertices, pose, scale)):
            raise ValueError(f"non-finite runtime geometry for {self.shape_id}")

    @property
    def scaled_vertices(self) -> np.ndarray:
        return np.asarray(self.vertices, dtype=np.float64) * np.asarray(
            self.scale_xyz, dtype=np.float64
        )

    def as_dict(self) -> dict[str, Any]:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        geometry_hash = _geometry_sha256(vertices, faces)
        return {
            "shape_id": self.shape_id,
            "body_name": self.body_name,
            "geometry_type": self.geometry_type,
            "local_transform": {
                "translation_xyz_m": self.local_pose_xyz_wxyz[:3].tolist(),
                "rotation_wxyz": self.local_pose_xyz_wxyz[3:].tolist(),
            },
            "scale_xyz": self.scale_xyz.tolist(),
            "convex_vertices_m": vertices.tolist(),
            "triangle_indices": faces.tolist(),
            "vertex_count": len(vertices),
            "triangle_count": len(faces),
            "geometry_sha256": geometry_hash,
            "source_asset_path": self.source_asset_path,
            "source_asset_sha256": self.source_asset_sha256,
            "generated_asset_path": self.generated_asset_path,
            "generated_asset_sha256": self.generated_asset_sha256,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> ConvexProxyGeometry:
        local = row["local_transform"]
        result = cls(
            shape_id=str(row["shape_id"]),
            body_name=str(row["body_name"]),
            geometry_type=str(row["geometry_type"]),
            vertices=np.asarray(row["convex_vertices_m"], dtype=np.float64),
            faces=np.asarray(row["triangle_indices"], dtype=np.int64),
            local_pose_xyz_wxyz=np.asarray(
                [*local["translation_xyz_m"], *local["rotation_wxyz"]], dtype=np.float64
            ),
            scale_xyz=np.asarray(row["scale_xyz"], dtype=np.float64),
            source_asset_path=str(row["source_asset_path"]),
            source_asset_sha256=str(row["source_asset_sha256"]),
            generated_asset_path=str(row["generated_asset_path"]),
            generated_asset_sha256=str(row["generated_asset_sha256"]),
        )
        if str(row["geometry_sha256"]) != _geometry_sha256(result.vertices, result.faces):
            raise ValueError(f"runtime geometry hash mismatch: {result.shape_id}")
        return result


def load_runtime_geometry_manifest(
    path: Path,
) -> tuple[list[ConvexProxyGeometry], dict[str, list[ConvexProxyGeometry]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "RuntimeCollisionGeometryManifestV1":
        raise ValueError("runtime collision geometry manifest schema mismatch")
    hand = [ConvexProxyGeometry.from_dict(row) for row in payload["hand_shapes"]]
    objects = {
        clip: [ConvexProxyGeometry.from_dict(row) for row in rows]
        for clip, rows in payload["object_shapes"].items()
    }
    if len(hand) != int(payload["validation"]["runtime_hand_shape_count"]):
        raise ValueError("runtime hand shape count does not match manifest")
    for clip, rows in objects.items():
        if len(rows) != int(payload["validation"]["runtime_object_shape_count"][clip]):
            raise ValueError(f"runtime object shape count does not match manifest: {clip}")
    return hand, objects


__all__ = ["ConvexProxyGeometry", "load_runtime_geometry_manifest", "sha256_file"]
