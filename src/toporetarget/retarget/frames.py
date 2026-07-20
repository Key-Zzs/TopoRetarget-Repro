"""Explicit wrist-centered frame profiles used by Stage 7.

The implementation uses row-major point arrays while transforms retain the
repository convention: transform columns are the local axes expressed in the
parent frame.  Thus ``(p - origin) @ R`` maps scene/base row points to local
coordinates.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.keypoints.registry import get_layout


class FrameDegeneracyError(ValueError):
    """Raised when a strict frame cannot be constructed from keypoints."""


@dataclass(frozen=True)
class BoneDirectionFrameProfile:
    profile_id: str
    version: str
    strategy: str
    layout_name: str
    zero_length_threshold_m: float
    strict: bool
    axes: dict[str, Any]
    side_rule: dict[str, Any]
    assumptions: tuple[str, ...]
    notes: str
    source_path: Path | None = None
    profile_hash: str = ""

    @property
    def sha256(self) -> str:
        return self.profile_hash

    @classmethod
    def from_mapping(
        cls, values: dict[str, Any], *, source_path: Path | None = None, raw: bytes | None = None
    ) -> BoneDirectionFrameProfile:
        profile = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            strategy=str(values["strategy"]),
            layout_name=str(values.get("layout_name", "mediapipe21")),
            zero_length_threshold_m=float(values.get("zero_length_threshold_m", 1e-10)),
            strict=bool(values.get("strict", True)),
            axes=dict(values.get("axes", {})),
            side_rule=dict(values.get("side_rule", {})),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            notes=str(values.get("notes", "")),
            source_path=source_path,
            profile_hash=hashlib.sha256(raw or b"").hexdigest() if raw is not None else "",
        )
        profile.validate()
        return profile

    def validate(self) -> BoneDirectionFrameProfile:
        if self.strategy not in {"canonical_keypoint_wrist", "translation_centered_scene_axes"}:
            raise ValueError(f"unsupported frame strategy: {self.strategy}")
        if self.zero_length_threshold_m <= 0:
            raise ValueError("frame zero_length_threshold_m must be positive")
        layout = get_layout(self.layout_name)
        if layout.point_count != 21 or layout.wrist_index != 0:
            raise ValueError("Stage 7 frame profile requires the canonical MediaPipe-21 layout")
        return self

    def _indices(self) -> tuple[int, int, int]:
        names = get_layout(self.layout_name).index_by_name
        return names["wrist"], names["middle_mcp"], names["index_mcp"]

    def frame_transform(
        self, keypoints: Any, *, side: str = "right", strict: bool | None = None
    ) -> Any:
        """Return ``T_parent_hand`` with shape ``[...,4,4]``.

        Strict mode reports every frame whose wrist-to-middle or index-to-pinky
        construction is below the explicit engineering threshold.
        """

        value = keypoints
        if getattr(value, "shape", None) is None or value.shape[-2:] != (21, 3):
            raise ValueError(
                f"keypoints must have shape [...,21,3], got {getattr(value, 'shape', None)}"
            )
        use_strict = self.strict if strict is None else strict
        wrist, middle, index = self._indices()
        layout = get_layout(self.layout_name)
        pinky = layout.index_by_name["pinky_mcp"]
        threshold = self.zero_length_threshold_m
        is_torch = type(value).__module__.split(".")[0] == "torch"
        if is_torch:
            import torch

            origin = value[..., wrist, :]
            longitudinal_raw = value[..., middle, :] - origin
            lateral_raw = value[..., index, :] - value[..., pinky, :]
            long_norm = torch.linalg.vector_norm(longitudinal_raw, dim=-1)
            lateral_norm = torch.linalg.vector_norm(lateral_raw, dim=-1)
            bad = (long_norm <= threshold) | (lateral_norm <= threshold)
            if use_strict and bool(torch.any(bad).detach().cpu()):
                indices = torch.nonzero(bad.detach().cpu(), as_tuple=False).tolist()
                raise FrameDegeneracyError(
                    f"{self.profile_id}: degenerate frame indices {indices}; "
                    f"longitudinal/lateral threshold={threshold:g} m"
                )
            safe_long = torch.clamp(long_norm, min=threshold)
            longitudinal = longitudinal_raw / safe_long[..., None]
            lateral_orth = (
                lateral_raw
                - torch.sum(lateral_raw * longitudinal, dim=-1, keepdim=True) * longitudinal
            )
            lateral_orth_norm = torch.linalg.vector_norm(lateral_orth, dim=-1)
            bad_cross = lateral_orth_norm <= threshold
            if use_strict and bool(torch.any(bad_cross).detach().cpu()):
                indices = torch.nonzero(bad_cross.detach().cpu(), as_tuple=False).tolist()
                raise FrameDegeneracyError(f"{self.profile_id}: near-collinear axes at {indices}")
            lateral = lateral_orth / torch.clamp(lateral_orth_norm, min=threshold)[..., None]
            third = torch.linalg.cross(lateral, longitudinal, dim=-1)
            third = (
                third
                / torch.clamp(torch.linalg.vector_norm(third, dim=-1), min=threshold)[..., None]
            )
            # Re-orthogonalize the lateral axis after the cross product.
            lateral = torch.linalg.cross(longitudinal, third, dim=-1)
            lateral = lateral / torch.linalg.vector_norm(lateral, dim=-1, keepdim=True)
            rotation = torch.stack((lateral, longitudinal, third), dim=-1)
            if self.strategy == "translation_centered_scene_axes":
                rotation = torch.eye(3, dtype=value.dtype, device=value.device).expand(
                    *value.shape[:-2], 3, 3
                )
            transform = (
                torch.eye(4, dtype=value.dtype, device=value.device)
                .expand(*value.shape[:-2], 4, 4)
                .clone()
            )
            transform[..., :3, :3] = rotation
            transform[..., :3, 3] = origin
            return transform
        points = np.asarray(value)
        origin = points[..., wrist, :]
        longitudinal_raw = points[..., middle, :] - origin
        lateral_raw = points[..., index, :] - points[..., pinky, :]
        long_norm = np.linalg.norm(longitudinal_raw, axis=-1)
        lateral_norm = np.linalg.norm(lateral_raw, axis=-1)
        bad = (long_norm <= threshold) | (lateral_norm <= threshold)
        if use_strict and np.any(bad):
            raise FrameDegeneracyError(
                f"{self.profile_id}: degenerate frame indices {np.argwhere(bad).tolist()}; "
                f"longitudinal/lateral threshold={threshold:g} m"
            )
        longitudinal = longitudinal_raw / np.maximum(long_norm, threshold)[..., None]
        lateral_orth = (
            lateral_raw - np.sum(lateral_raw * longitudinal, axis=-1, keepdims=True) * longitudinal
        )
        lateral_orth_norm = np.linalg.norm(lateral_orth, axis=-1)
        if use_strict and np.any(lateral_orth_norm <= threshold):
            raise FrameDegeneracyError(f"{self.profile_id}: near-collinear axes")
        lateral = lateral_orth / np.maximum(lateral_orth_norm, threshold)[..., None]
        third = np.cross(lateral, longitudinal)
        third = third / np.maximum(np.linalg.norm(third, axis=-1), threshold)[..., None]
        lateral = np.cross(longitudinal, third)
        lateral = lateral / np.linalg.norm(lateral, axis=-1, keepdims=True)
        rotation_np = np.stack((lateral, longitudinal, third), axis=-1)
        if self.strategy == "translation_centered_scene_axes":
            rotation_np = np.broadcast_to(np.eye(3), points.shape[:-2] + (3, 3)).copy()
        transform_np = np.broadcast_to(np.eye(4), points.shape[:-2] + (4, 4)).copy()
        transform_np[..., :3, :3] = rotation_np
        transform_np[..., :3, 3] = origin
        return transform_np

    def to_local(self, keypoints: Any, transform: Any) -> Any:
        origin = transform[..., :3, 3]
        rotation = transform[..., :3, :3]
        if type(keypoints).__module__.split(".")[0] == "torch":
            return torch_matmul_points(keypoints - origin[..., None, :], rotation)
        return np.einsum(
            "...ni,...ij->...nj", np.asarray(keypoints) - origin[..., None, :], rotation
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "strategy": self.strategy,
            "layout_name": self.layout_name,
            "zero_length_threshold_m": self.zero_length_threshold_m,
            "strict": self.strict,
            "axes": self.axes,
            "side_rule": self.side_rule,
            "assumptions": list(self.assumptions),
            "notes": self.notes,
            "profile_hash": self.profile_hash,
        }


def torch_matmul_points(points: Any, rotation: Any) -> Any:
    import torch

    return torch.matmul(points, rotation)


def load_frame_profile(
    profile_id: str, *, config_root: str | Path | None = None
) -> BoneDirectionFrameProfile:
    root = (
        Path(config_root)
        if config_root is not None
        else Path(__file__).resolve().parents[3] / "configs" / "retarget" / "frames"
    )
    path = root / f"{profile_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"frame profile not found: {profile_id}")
    raw = path.read_bytes()
    values = yaml.safe_load(raw) or {}
    if not isinstance(values, dict):
        raise ValueError(f"frame profile must be a mapping: {path}")
    return BoneDirectionFrameProfile.from_mapping(values, source_path=path, raw=raw)


__all__ = ["BoneDirectionFrameProfile", "FrameDegeneracyError", "load_frame_profile"]
