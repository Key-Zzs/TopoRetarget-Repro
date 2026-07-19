"""Read-only mesh integrity audits used by the Stage 6 geometry foundation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _topology_hash(faces: np.ndarray) -> str:
    value = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    return _array_hash(np.sort(value, axis=0)) if value.size else _array_hash(value)


def _edge_data(
    faces: np.ndarray,
) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    counts: Counter[tuple[int, int]] = Counter()
    directed: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for _face_index, face in enumerate(faces):
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = (min(start, end), max(start, end))
            counts[key] += 1
            directed[key].append((start, end))
    return counts, directed


def _winding_consistency(faces: np.ndarray) -> tuple[bool | None, bool | None]:
    if faces.size == 0:
        return None, None
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for index, face in enumerate(faces):
        for _edge_index, (start, end) in enumerate(
            ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        ):
            key = (min(int(start), int(end)), max(int(start), int(end)))
            sign = 1 if (int(start), int(end)) == key else -1
            adjacency[key].append((index, sign))
    graph: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for values in adjacency.values():
        if len(values) != 2:
            continue
        (first, first_sign), (second, second_sign) = values
        relation = first_sign == second_sign
        graph[first].append((second, relation))
        graph[second].append((first, relation))
    colors: dict[int, bool] = {}
    orientable = True
    for root in range(len(faces)):
        if root in colors:
            continue
        colors[root] = False
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for other, same_color in graph[current]:
                # Oppositely directed shared edges are already compatible;
                # equally directed edges require flipping the neighbour.
                expected = colors[current] if not same_color else not colors[current]
                if other in colors:
                    orientable &= colors[other] == expected
                else:
                    colors[other] = expected
                    queue.append(other)
    if not graph:
        return None, orientable
    consistent = True
    for values in adjacency.values():
        if len(values) == 2:
            consistent &= values[0][1] != values[1][1]
    return consistent, orientable


def _connected_components(vertex_count: int, faces: np.ndarray) -> int:
    if vertex_count == 0:
        return 0
    neighbours: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in faces:
        a, b, c = (int(item) for item in face)
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    seen: set[int] = set()
    components = 0
    for root in range(vertex_count):
        if root in seen:
            continue
        components += 1
        queue = [root]
        seen.add(root)
        while queue:
            current = queue.pop()
            for other in neighbours[current]:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
    return components


@dataclass(frozen=True)
class MeshAuditReport:
    vertex_count: int
    face_count: int
    vertex_dtype: str
    face_dtype: str
    units: str
    finite_vertices: bool
    valid_face_indices: bool
    triangular_faces: bool
    zero_area_faces: int
    near_zero_area_faces: int
    degenerate_area_threshold: float
    duplicate_vertices: int
    duplicate_faces: int
    unreferenced_vertices: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    connected_component_count: int
    watertight: bool
    winding_consistent: bool | None
    orientable: bool | None
    signed_volume: float | None
    euler_number: int | None
    bounding_box_min: list[float] | None
    bounding_box_max: list[float] | None
    bounding_box_diagonal: float | None
    surface_area: float | None
    center_of_mass: list[float] | None
    mesh_hash: str
    topology_hash: str
    source_path: str | None
    source_provenance: dict[str, Any] = field(default_factory=dict)
    derived_valid_face_count: int = 0
    derived_invalid_face_count: int = 0
    sign_reliability: str = "unknown"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items()}

    def summary(self) -> str:
        return (
            f"vertices={self.vertex_count} faces={self.face_count} "
            f"watertight={self.watertight} winding={self.winding_consistent} "
            f"components={self.connected_component_count} sign={self.sign_reliability}"
        )

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("field", "value"))
            for key, value in self.as_dict().items():
                writer.writerow((key, json.dumps(value, sort_keys=True)))


def audit_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    units: str = "m",
    source_path: str | Path | None = None,
    source_provenance: dict[str, Any] | None = None,
    degenerate_area_threshold: float = 1e-12,
    mesh_id: str | None = None,
) -> MeshAuditReport:
    """Audit without changing the input arrays or repairing the source mesh."""

    vertex_array = np.asarray(vertices)
    face_array = np.asarray(faces)
    if vertex_array.ndim != 2 or (
        vertex_array.shape[1:] != (3,) if vertex_array.ndim >= 1 else True
    ):
        raise ValueError(f"vertices must have shape [V,3], got {vertex_array.shape}")
    triangular = face_array.ndim == 2 and face_array.shape[1:] == (3,)
    if face_array.ndim != 2:
        face_array = np.empty((0, 3), dtype=np.int64)
    elif face_array.shape[1] != 3:
        face_array = np.empty((0, 3), dtype=np.int64)
    face_int = np.asarray(face_array, dtype=np.int64)
    finite = bool(np.all(np.isfinite(vertex_array)))
    valid_indices = bool(
        face_int.size == 0 or (np.all(face_int >= 0) and np.all(face_int < len(vertex_array)))
    )
    valid_mask = np.ones(len(face_int), dtype=bool)
    if not valid_indices and len(face_int):
        valid_mask &= np.all(face_int >= 0, axis=1) & np.all(face_int < len(vertex_array), axis=1)
    valid_faces = face_int[valid_mask]
    areas = np.zeros(len(valid_faces), dtype=np.float64)
    if len(valid_faces):
        a = np.asarray(vertex_array[valid_faces[:, 0]], dtype=np.float64)
        b = np.asarray(vertex_array[valid_faces[:, 1]], dtype=np.float64)
        c = np.asarray(vertex_array[valid_faces[:, 2]], dtype=np.float64)
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    zero = int(np.count_nonzero(areas == 0.0))
    near_zero = int(np.count_nonzero(areas <= degenerate_area_threshold))
    derived_mask = areas > degenerate_area_threshold
    derived_faces = valid_faces[derived_mask]
    derived_areas = areas[derived_mask]
    unique_vertices = len({tuple(row.tolist()) for row in vertex_array})
    duplicate_vertices = max(0, len(vertex_array) - unique_vertices)
    duplicate_faces = max(
        0, len(valid_faces) - len({tuple(sorted(row.tolist())) for row in valid_faces})
    )
    referenced = set(derived_faces.reshape(-1).tolist()) if len(derived_faces) else set()
    unreferenced = int(len(vertex_array) - len(referenced))
    edge_counts, _ = _edge_data(derived_faces)
    boundary = int(sum(count == 1 for count in edge_counts.values()))
    non_manifold = int(sum(count > 2 for count in edge_counts.values()))
    components = _connected_components(len(vertex_array), derived_faces)
    winding, orientable = _winding_consistency(derived_faces)
    watertight = bool(len(derived_faces) > 0 and boundary == 0 and non_manifold == 0)
    signed_volume: float | None = None
    center_of_mass: list[float] | None = None
    if watertight and len(derived_faces):
        a = np.asarray(vertex_array[derived_faces[:, 0]], dtype=np.float64)
        b = np.asarray(vertex_array[derived_faces[:, 1]], dtype=np.float64)
        c = np.asarray(vertex_array[derived_faces[:, 2]], dtype=np.float64)
        signed_volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
        tetra_volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
        if abs(float(np.sum(tetra_volumes))) > 1e-15:
            center = np.sum((a + b + c) * tetra_volumes[:, None], axis=0) / (
                4.0 * np.sum(tetra_volumes)
            )
            center_of_mass = center.tolist()
    euler: int | None = None
    if len(derived_faces):
        euler = int(len(referenced) - len(edge_counts) + len(derived_faces))
    bounds_min = np.min(vertex_array, axis=0).astype(float).tolist() if len(vertex_array) else None
    bounds_max = np.max(vertex_array, axis=0).astype(float).tolist() if len(vertex_array) else None
    diagonal = (
        float(np.linalg.norm(np.asarray(bounds_max) - np.asarray(bounds_min)))
        if bounds_min and bounds_max
        else None
    )
    area = float(np.sum(derived_areas)) if len(derived_areas) else 0.0
    if not finite or not valid_indices or not triangular or not len(derived_faces):
        reliability = "degenerate" if not len(derived_faces) or not finite else "unknown"
    elif not watertight:
        reliability = "non_manifold" if non_manifold else "open_surface"
    elif winding is False or orientable is False:
        reliability = "watertight_inconsistent_winding"
    else:
        reliability = "reliable_watertight"
    notes: list[str] = []
    if near_zero:
        notes.append(
            "degenerate faces are excluded only from derived calculations; source is unchanged"
        )
    if mesh_id:
        notes.append(f"mesh_id={mesh_id}")
    return MeshAuditReport(
        vertex_count=int(len(vertex_array)),
        face_count=int(len(faces)),
        vertex_dtype=str(vertex_array.dtype),
        face_dtype=str(np.asarray(faces).dtype),
        units=units,
        finite_vertices=finite,
        valid_face_indices=valid_indices,
        triangular_faces=triangular,
        zero_area_faces=zero,
        near_zero_area_faces=near_zero,
        degenerate_area_threshold=float(degenerate_area_threshold),
        duplicate_vertices=duplicate_vertices,
        duplicate_faces=duplicate_faces,
        unreferenced_vertices=unreferenced,
        boundary_edge_count=boundary,
        non_manifold_edge_count=non_manifold,
        connected_component_count=components,
        watertight=watertight,
        winding_consistent=winding,
        orientable=orientable,
        signed_volume=signed_volume,
        euler_number=euler,
        bounding_box_min=bounds_min,
        bounding_box_max=bounds_max,
        bounding_box_diagonal=diagonal,
        surface_area=area,
        center_of_mass=center_of_mass,
        mesh_hash=_array_hash(np.asarray(vertex_array, dtype=np.float64)) + _array_hash(face_int),
        topology_hash=_topology_hash(face_int),
        source_path=None if source_path is None else str(Path(source_path).expanduser()),
        source_provenance=dict(source_provenance or {}),
        derived_valid_face_count=int(len(derived_faces)),
        derived_invalid_face_count=int(len(face_int) - len(derived_faces)),
        sign_reliability=reliability,
        notes=notes,
    )


__all__ = ["MeshAuditReport", "audit_mesh"]
