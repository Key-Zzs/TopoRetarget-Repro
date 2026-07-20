"""Strict, source-side Delaunay tetrahedralization for Stage 8.

The paper does not identify a tetrahedralization implementation.  The default
profile therefore makes the SciPy/Qhull choice explicit and records the
numerical normalization used for the Qhull input.  The normalization is only
an origin translation and a uniform scale; source vertices and every
downstream metric remain in the canonical meter scene frame.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DELAUNAY_PROFILE_ID = "strict_scipy_qhull_v1"
DIAGNOSTIC_PROFILE_ID = "deterministic_jitter_diagnostic"
DEFAULT_QHULL_OPTIONS = "Qbb Qc Qz Q12"


class DelaunayValidationError(ValueError):
    """Raised when a source point cloud cannot satisfy the strict contract."""


class DelaunayConstructionError(RuntimeError):
    """Raised when SciPy/Qhull cannot construct a valid tetrahedralization."""


@dataclass(frozen=True)
class DelaunayProfile:
    profile_id: str
    backend: str
    dimension: int
    incremental: bool
    qhull_options: str
    normalization: str
    volume_tolerance: float
    near_degenerate_tolerance: float
    near_duplicate_relative: float
    jitter: bool = False
    jitter_seed: int | None = None
    jitter_relative: float = 0.0
    source_path: Path | None = None
    profile_hash: str = ""

    @property
    def sha256(self) -> str:
        return self.profile_hash

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "backend": self.backend,
            "dimension": self.dimension,
            "incremental": self.incremental,
            "qhull_options": self.qhull_options,
            "normalization": self.normalization,
            "volume_tolerance": self.volume_tolerance,
            "near_degenerate_tolerance": self.near_degenerate_tolerance,
            "near_duplicate_relative": self.near_duplicate_relative,
            "jitter": self.jitter,
            "jitter_seed": self.jitter_seed,
            "jitter_relative": self.jitter_relative,
        }

    def validate(self) -> DelaunayProfile:
        if self.backend != "scipy.spatial.Delaunay":
            raise ValueError(f"unsupported Delaunay backend: {self.backend}")
        if self.dimension != 3 or self.incremental:
            raise ValueError("Stage 8 requires non-incremental three-dimensional Delaunay")
        if not self.qhull_options or "QJ" in self.qhull_options.split():
            if not self.jitter:
                raise ValueError("strict Delaunay profile must use explicit options without QJ")
        if self.volume_tolerance < 0 or self.near_degenerate_tolerance < 0:
            raise ValueError("Delaunay tolerances must be non-negative")
        if self.jitter and self.jitter_seed is None:
            raise ValueError("diagnostic jitter requires a deterministic seed")
        return self


@dataclass(frozen=True)
class PointCloudDiagnostics:
    shape: tuple[int, ...]
    finite: bool
    unique_vertices: int
    duplicate_count: int
    near_duplicate_count: int
    affine_rank: int
    bounding_box_diagonal: float
    minimum_intervertex_distance: float
    coordinate_frame: str
    units: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "finite": self.finite,
            "unique_vertices": self.unique_vertices,
            "duplicate_count": self.duplicate_count,
            "near_duplicate_count": self.near_duplicate_count,
            "affine_rank": self.affine_rank,
            "bounding_box_diagonal_m": self.bounding_box_diagonal,
            "minimum_intervertex_distance_m": self.minimum_intervertex_distance,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
        }


@dataclass(frozen=True)
class DelaunayResult:
    points: np.ndarray
    normalized_points: np.ndarray
    simplices: np.ndarray
    simplex_volumes: np.ndarray
    normalized_simplex_volumes: np.ndarray
    point_diagnostics: PointCloudDiagnostics
    near_degenerate_simplex_count: int


def _profile_hash(values: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _profile_from_values(
    values: dict[str, Any], *, source_path: Path | None = None, raw: bytes | None = None
) -> DelaunayProfile:
    profile_id = str(values["profile_id"])
    normalized: dict[str, Any] = {
        "profile_id": profile_id,
        "backend": str(values.get("backend", "scipy.spatial.Delaunay")),
        "dimension": int(values.get("dimension", 3)),
        "incremental": bool(values.get("incremental", False)),
        "qhull_options": str(values.get("qhull_options", DEFAULT_QHULL_OPTIONS)),
        "normalization": str(values.get("normalization", "centroid_bbox_diagonal")),
        "volume_tolerance": float(values.get("volume_tolerance", 1e-24)),
        "near_degenerate_tolerance": float(values.get("near_degenerate_tolerance", 1e-18)),
        "near_duplicate_relative": float(values.get("near_duplicate_relative", 1e-10)),
        "jitter": bool(values.get("jitter", False)),
        "jitter_seed": None if values.get("jitter_seed") is None else int(values["jitter_seed"]),
        "jitter_relative": float(values.get("jitter_relative", 0.0)),
    }
    profile_hash = hashlib.sha256(raw).hexdigest() if raw is not None else _profile_hash(normalized)
    return DelaunayProfile(
        **normalized, source_path=source_path, profile_hash=profile_hash
    ).validate()


def load_delaunay_profile(
    profile_id: str = DELAUNAY_PROFILE_ID, *, config_root: str | Path | None = None
) -> DelaunayProfile:
    """Load one tracked Stage 8 Delaunay profile."""

    root = (
        Path(config_root).expanduser()
        if config_root is not None
        else Path(__file__).resolve().parents[3] / "configs" / "retarget" / "interaction"
    )
    path = root / f"{profile_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Delaunay profile not found: {profile_id}")
    raw = path.read_bytes()
    values = yaml.safe_load(raw) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Delaunay profile must be a mapping: {path}")
    return _profile_from_values(values, source_path=path, raw=raw)


def _frame_context(frame_index: int | None) -> str:
    return "frame unknown" if frame_index is None else f"frame {frame_index}"


def validate_point_cloud(
    points: np.ndarray,
    *,
    expected_count: int | None = None,
    frame_index: int | None = None,
    coordinate_frame: str = "S",
    units: str = "m",
    near_duplicate_relative: float = 1e-10,
) -> PointCloudDiagnostics:
    """Validate a point cloud without deleting, merging, or modifying points."""

    value = np.asarray(points, dtype=np.float64)
    context = _frame_context(frame_index)
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise DelaunayValidationError(f"{context}: points must have shape [N,3], got {value.shape}")
    if expected_count is not None and value.shape[0] != expected_count:
        raise DelaunayValidationError(
            f"{context}: expected {expected_count} vertices, got {value.shape[0]}"
        )
    finite = bool(np.all(np.isfinite(value)))
    if not finite:
        raise DelaunayValidationError(f"{context}: point cloud contains NaN or Inf")
    if value.shape[0] < 4:
        raise DelaunayValidationError(f"{context}: at least four 3D points are required")
    unique_count = int(np.unique(value, axis=0).shape[0])
    duplicate_count = int(value.shape[0] - unique_count)
    if duplicate_count:
        raise DelaunayValidationError(f"{context}: exact duplicate vertices: {duplicate_count}")
    bounds = np.ptp(value, axis=0)
    diagonal = float(np.linalg.norm(bounds))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise DelaunayValidationError(f"{context}: bounding-box diagonal is not positive")
    distances = np.linalg.norm(value[:, None, :] - value[None, :, :], axis=-1)
    distances[np.diag_indices(value.shape[0])] = np.inf
    minimum_distance = float(np.min(distances))
    near_threshold = near_duplicate_relative * diagonal
    near_count = int(np.count_nonzero(np.triu(distances < near_threshold, k=1)))
    if near_count:
        raise DelaunayValidationError(
            f"{context}: near-duplicate vertex pairs: {near_count} "
            f"(threshold={near_threshold:.3e} m)"
        )
    centered = value - value.mean(axis=0, keepdims=True)
    affine_rank = int(np.linalg.matrix_rank(centered, tol=max(diagonal * 1e-12, 1e-15)))
    if affine_rank < 3:
        raise DelaunayValidationError(f"{context}: affine rank is {affine_rank}, expected 3")
    return PointCloudDiagnostics(
        shape=tuple(int(item) for item in value.shape),
        finite=finite,
        unique_vertices=unique_count,
        duplicate_count=duplicate_count,
        near_duplicate_count=near_count,
        affine_rank=affine_rank,
        bounding_box_diagonal=diagonal,
        minimum_intervertex_distance=minimum_distance,
        coordinate_frame=coordinate_frame,
        units=units,
    )


def _normalize_for_qhull(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.asarray(points, dtype=np.float64).copy()
    if mode != "centroid_bbox_diagonal":
        raise DelaunayValidationError(f"unsupported Delaunay normalization: {mode}")
    value = np.asarray(points, dtype=np.float64)
    diagonal = float(np.linalg.norm(np.ptp(value, axis=0)))
    if diagonal <= 0 or not np.isfinite(diagonal):
        raise DelaunayValidationError("cannot normalize a point cloud with zero scale")
    return (value - value.mean(axis=0, keepdims=True)) / diagonal


def _simplex_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    tetra = np.asarray(points, dtype=np.float64)[np.asarray(simplices, dtype=np.int64)]
    a, b, c, d = (tetra[:, index] for index in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def tetrahedralize(
    points: np.ndarray,
    profile: DelaunayProfile | None = None,
    *,
    frame_index: int | None = None,
    expected_count: int | None = 71,
) -> DelaunayResult:
    """Run exactly one strict source-side SciPy Delaunay call."""

    selected = profile or _profile_from_values({"profile_id": DELAUNAY_PROFILE_ID})
    value = np.asarray(points, dtype=np.float64)
    diagnostics = validate_point_cloud(
        value,
        expected_count=expected_count,
        frame_index=frame_index,
        near_duplicate_relative=selected.near_duplicate_relative,
    )
    normalized = _normalize_for_qhull(value, selected.normalization)
    if selected.jitter:
        rng = np.random.default_rng(selected.jitter_seed)
        amplitude = selected.jitter_relative * max(
            float(np.linalg.norm(np.ptp(normalized, axis=0))), 1.0
        )
        normalized = normalized + rng.normal(0.0, amplitude, size=normalized.shape)
    try:
        from scipy.spatial import Delaunay, QhullError

        triangulation = Delaunay(
            normalized,
            incremental=selected.incremental,
            qhull_options=selected.qhull_options,
        )
    except Exception as exc:
        context = _frame_context(frame_index)
        try:
            from scipy.spatial import QhullError

            if isinstance(exc, QhullError):
                raise DelaunayConstructionError(
                    f"{context}: SciPy/Qhull failure with options {selected.qhull_options!r}: {exc}"
                ) from exc
        except ImportError:  # pragma: no cover - scipy is the selected backend
            pass
        raise DelaunayConstructionError(f"{context}: Delaunay failure: {exc}") from exc
    simplices = np.asarray(triangulation.simplices, dtype=np.int64)
    context = _frame_context(frame_index)
    if simplices.ndim != 2 or simplices.shape[1:] != (4,):
        raise DelaunayConstructionError(
            f"{context}: Qhull returned invalid simplex shape {simplices.shape}"
        )
    if np.any(simplices < 0) or np.any(simplices >= value.shape[0]):
        raise DelaunayConstructionError(f"{context}: simplex index is out of range")
    if np.any(np.sort(simplices, axis=1)[:, 1:] == np.sort(simplices, axis=1)[:, :-1]):
        raise DelaunayConstructionError(f"{context}: simplex contains repeated vertices")
    canonical = np.sort(simplices, axis=1)
    if np.unique(canonical, axis=0).shape[0] != canonical.shape[0]:
        raise DelaunayConstructionError(f"{context}: duplicate tetrahedra returned by Qhull")
    normalized_volumes = _simplex_volumes(normalized, simplices)
    if not np.all(np.isfinite(normalized_volumes)) or np.any(
        normalized_volumes <= selected.volume_tolerance
    ):
        count = int(np.count_nonzero(normalized_volumes <= selected.volume_tolerance))
        raise DelaunayConstructionError(
            f"{context}: {count} simplex volumes are at or below strict tolerance "
            f"{selected.volume_tolerance:.3e}"
        )
    volumes = _simplex_volumes(value, simplices)
    return DelaunayResult(
        points=value.copy(),
        normalized_points=normalized,
        simplices=simplices,
        simplex_volumes=volumes,
        normalized_simplex_volumes=normalized_volumes,
        point_diagnostics=diagnostics,
        near_degenerate_simplex_count=int(
            np.count_nonzero(normalized_volumes <= selected.near_degenerate_tolerance)
        ),
    )


def extract_unique_edges(
    simplices: np.ndarray, *, vertex_count: int = 71, frame_index: int | None = None
) -> np.ndarray:
    """Extract the six sorted edges of every tetrahedron, then globally sort."""

    value = np.asarray(simplices, dtype=np.int64)
    context = _frame_context(frame_index)
    if value.ndim != 2 or value.shape[1:] != (4,):
        raise DelaunayValidationError(f"{context}: simplices must have shape [S,4]")
    if np.any(value < 0) or np.any(value >= vertex_count):
        raise DelaunayValidationError(f"{context}: simplex index outside [0,{vertex_count})")
    pairs = np.concatenate(
        [
            value[:, [0, 1]],
            value[:, [0, 2]],
            value[:, [0, 3]],
            value[:, [1, 2]],
            value[:, [1, 3]],
            value[:, [2, 3]],
        ],
        axis=0,
    )
    edges = np.sort(pairs, axis=1)
    edges = np.unique(edges, axis=0)
    if len(edges):
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    if len(edges) and (
        np.any(edges[:, 0] == edges[:, 1])
        or (len(edges) > 1 and np.any(np.all(edges[1:] == edges[:-1], axis=1)))
    ):
        raise DelaunayValidationError(f"{context}: self or duplicate edge detected")
    return edges.astype(np.int64, copy=False)


def edge_category(edge: tuple[int, int] | np.ndarray) -> str:
    first, second = (int(item) for item in edge)
    if first < 21 and second < 21:
        return "hand-hand"
    if (first < 21) != (second < 21):
        return "hand-object"
    return "object-object"


__all__ = [
    "DEFAULT_QHULL_OPTIONS",
    "DELAUNAY_PROFILE_ID",
    "DIAGNOSTIC_PROFILE_ID",
    "DelaunayConstructionError",
    "DelaunayProfile",
    "DelaunayResult",
    "DelaunayValidationError",
    "PointCloudDiagnostics",
    "edge_category",
    "extract_unique_edges",
    "load_delaunay_profile",
    "tetrahedralize",
    "validate_point_cloud",
]
