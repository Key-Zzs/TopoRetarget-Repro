"""Explicit, versioned MANO reconstruction contracts.

The Stage 12 source adapters must never infer whether a 45-value hand pose is
PCA or axis-angle from its field name or dimension.  This module makes that
choice, the MANO mean, shape coefficients, and model provenance explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class ManoContractError(RuntimeError):
    """Base class for explicit MANO contract violations."""


class AmbiguousManoPoseRepresentationError(ManoContractError):
    """Raised when the pose representation is absent or not explicit."""


class InvalidManoPoseDimensionError(ManoContractError):
    """Raised when a pose array does not match its declared representation."""


class MissingRequiredManoBetasError(ManoContractError):
    """Raised when a dataset that requires calibrated betas omits them."""


class InvalidManoSideError(ManoContractError):
    """Raised when side is missing or outside the two MANO model sides."""


class ManoModelProvenanceError(ManoContractError):
    """Raised when model, annotation, unit, or dtype provenance is incomplete."""


class ManoPoseRepresentation(str, Enum):
    """The only accepted native MANO hand-pose encodings."""

    AXIS_ANGLE = "axis_angle"
    PCA = "pca"


class ManoJointSource(str, Enum):
    """Declared provenance for the canonical joint source."""

    BACKEND_POSED = "backend_posed"
    DATASET_NATIVE = "dataset_native"
    CONTACTPOSE_OFFICIAL = "contactpose_official"
    EXPLICIT_REGRESSED_FALLBACK = "explicit_regressed_fallback"


def _float64_array(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise ManoModelProvenanceError(
            f"{field_name} must be explicitly float64, got {array.dtype}"
        )
    return np.ascontiguousarray(array)


@dataclass
class ManoModelProvenance:
    """Resolved model and annotation provenance retained with each result."""

    model_path: str
    model_hash: str
    model_version: str
    dataset_name: str
    source_annotation_path: str
    source_annotation_hash: str
    pca_basis_hash: str | None = None
    hand_mean_hash: str | None = None


@dataclass
class ManoReconstructionRequest:
    """A fully declared MANO reconstruction request.

    ``betas_required`` exists solely for legacy datasets such as GRAB that
    explicitly carry no shape coefficients.  Dataset adapters in Stage 12 use
    its default ``True`` and therefore fail closed when calibration is absent.
    """

    side: str
    pose_representation: ManoPoseRepresentation | str | None
    global_orient: np.ndarray
    hand_pose: np.ndarray
    num_pca_components: int | None
    translation: np.ndarray
    betas: np.ndarray | None
    flat_hand_mean: bool | None
    units: str
    dtype: np.dtype[Any] | str | type[np.float64]
    model_path: str | Path
    model_hash: str
    model_version: str
    dataset_name: str
    source_annotation_path: str | Path
    source_annotation_hash: str
    betas_required: bool = True

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise InvalidManoSideError(f"MANO side must be 'left' or 'right', got {self.side!r}")
        try:
            self.pose_representation = ManoPoseRepresentation(self.pose_representation)
        except (TypeError, ValueError) as exc:
            raise AmbiguousManoPoseRepresentationError(
                "pose_representation must explicitly be 'axis_angle' or 'pca'"
            ) from exc
        if type(self.flat_hand_mean) is not bool:
            raise ManoModelProvenanceError("flat_hand_mean must be an explicit bool")
        if self.units != "metre":
            raise ManoModelProvenanceError(
                f"MANO units must explicitly be 'metre', got {self.units!r}"
            )
        if np.dtype(self.dtype) != np.dtype(np.float64):
            raise ManoModelProvenanceError(
                f"MANO dtype must explicitly be float64, got {np.dtype(self.dtype)}"
            )
        for field_name in (
            "model_path",
            "model_hash",
            "model_version",
            "dataset_name",
            "source_annotation_path",
            "source_annotation_hash",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ManoModelProvenanceError(f"{field_name} is required for MANO provenance")
        self.model_path = str(Path(self.model_path))
        self.source_annotation_path = str(Path(self.source_annotation_path))
        self.global_orient = _float64_array(self.global_orient, field_name="global_orient")
        self.hand_pose = _float64_array(self.hand_pose, field_name="hand_pose")
        self.translation = _float64_array(self.translation, field_name="translation")
        if self.global_orient.ndim != 2 or self.global_orient.shape[1:] != (3,):
            raise InvalidManoPoseDimensionError(
                f"global_orient must have shape [T,3], got {self.global_orient.shape}"
            )
        frame_count = self.global_orient.shape[0]
        if self.translation.shape != (frame_count, 3):
            raise InvalidManoPoseDimensionError(
                f"translation must have shape [{frame_count},3], got {self.translation.shape}"
            )
        if self.pose_representation is ManoPoseRepresentation.AXIS_ANGLE:
            if self.num_pca_components is not None:
                raise InvalidManoPoseDimensionError(
                    "axis-angle requests must declare num_pca_components=None"
                )
            if self.hand_pose.shape != (frame_count, 45):
                raise InvalidManoPoseDimensionError(
                    "axis-angle hand_pose must have shape "
                    f"[{frame_count},45], got {self.hand_pose.shape}"
                )
        else:
            if not isinstance(self.num_pca_components, int) or not (
                1 <= self.num_pca_components <= 45
            ):
                raise InvalidManoPoseDimensionError(
                    "PCA requests must explicitly declare num_pca_components in [1,45]"
                )
            if self.hand_pose.shape != (frame_count, self.num_pca_components):
                raise InvalidManoPoseDimensionError(
                    "PCA hand_pose must match its declared component count: "
                    f"expected [{frame_count},{self.num_pca_components}], "
                    f"got {self.hand_pose.shape}"
                )
        if self.betas is None:
            if self.betas_required:
                raise MissingRequiredManoBetasError("required MANO betas are missing")
            return
        self.betas = _float64_array(self.betas, field_name="betas")
        if self.betas.shape not in {(10,), (1, 10), (frame_count, 10)}:
            raise MissingRequiredManoBetasError(
                f"betas must have shape [10], [1,10], or [{frame_count},10], got {self.betas.shape}"
            )
        if not np.isfinite(self.betas).all():
            raise MissingRequiredManoBetasError("betas contain non-finite values")

    @property
    def frame_count(self) -> int:
        return int(self.global_orient.shape[0])

    def broadcast_betas(self) -> np.ndarray | None:
        """Return documented [T,10] beta broadcasting without hidden defaults."""

        if self.betas is None:
            return None
        return np.broadcast_to(self.betas.reshape(-1, 10), (self.frame_count, 10)).copy()


@dataclass
class ManoReconstructionResult:
    """Native mesh/joints plus derived full axis-angle and provenance."""

    vertices: np.ndarray
    posed_joints_native: np.ndarray
    posed_joint_layout: str
    full_pose_axis_angle: np.ndarray
    global_orient_axis_angle: np.ndarray
    hand_pose_axis_angle: np.ndarray
    faces: np.ndarray
    side: str
    betas: np.ndarray | None
    translation: np.ndarray
    wrist_pose_scene: np.ndarray
    model_provenance: ManoModelProvenance
    reconstruction_manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "vertices",
            "posed_joints_native",
            "full_pose_axis_angle",
            "global_orient_axis_angle",
            "hand_pose_axis_angle",
            "translation",
            "wrist_pose_scene",
        ):
            setattr(self, field_name, np.asarray(getattr(self, field_name), dtype=np.float64))
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.betas is not None:
            self.betas = np.asarray(self.betas, dtype=np.float64)


__all__ = [
    "AmbiguousManoPoseRepresentationError",
    "InvalidManoPoseDimensionError",
    "InvalidManoSideError",
    "ManoContractError",
    "ManoJointSource",
    "ManoModelProvenance",
    "ManoModelProvenanceError",
    "ManoPoseRepresentation",
    "ManoReconstructionRequest",
    "ManoReconstructionResult",
    "MissingRequiredManoBetasError",
]
