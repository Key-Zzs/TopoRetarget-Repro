"""Semantic directed bones and differentiable relative-direction features."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.keypoints.registry import get_layout


class ZeroLengthBoneError(ValueError):
    """Raised when strict direction extraction sees a zero-length directed bone."""


@dataclass(frozen=True)
class DirectedBone:
    name: str
    finger: str
    parent_name: str
    child_name: str
    parent_index: int
    child_index: int


@dataclass(frozen=True)
class BonePair:
    name: str
    finger: str
    first_bone: int
    second_bone: int


@dataclass(frozen=True)
class BoneDirectionProfile:
    profile_id: str
    version: str
    layout_name: str
    include_wrist_to_mcp: bool
    fingers: dict[str, tuple[str, ...]]
    bones: tuple[DirectedBone, ...]
    pairs: tuple[BonePair, ...]
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
    ) -> BoneDirectionProfile:
        layout = get_layout(str(values.get("layout_name", "mediapipe21")))
        names = layout.index_by_name
        fingers = {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(values["fingers"]).items()
        }
        bones: list[DirectedBone] = []
        pairs: list[BonePair] = []
        for finger, path in fingers.items():
            if len(path) < 2:
                raise ValueError(f"{finger}: a bone profile needs at least two semantic points")
            finger_bones: list[int] = []
            for _index, (parent, child) in enumerate(zip(path, path[1:], strict=False)):
                if parent not in names or child not in names:
                    raise ValueError(f"{finger}: unknown semantic point in {parent}->{child}")
                bone_index = len(bones)
                bones.append(
                    DirectedBone(
                        f"{finger}:{parent}->{child}",
                        finger,
                        parent,
                        child,
                        names[parent],
                        names[child],
                    )
                )
                finger_bones.append(bone_index)
            for first, second in zip(finger_bones, finger_bones[1:], strict=False):
                pair_suffix = (
                    f"{bones[first].name.split(':', 1)[1]}|{bones[second].name.split(':', 1)[1]}"
                )
                pairs.append(
                    BonePair(
                        f"{finger}:{pair_suffix}",
                        finger,
                        first,
                        second,
                    )
                )
        profile = cls(
            profile_id=str(values["profile_id"]),
            version=str(values.get("version", "1.0.0")),
            layout_name=str(values.get("layout_name", "mediapipe21")),
            include_wrist_to_mcp=bool(values.get("include_wrist_to_mcp", False)),
            fingers=fingers,
            bones=tuple(bones),
            pairs=tuple(pairs),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            notes=str(values.get("notes", "")),
            source_path=source_path,
            profile_hash=hashlib.sha256(raw or b"").hexdigest() if raw is not None else "",
        )
        profile.validate()
        return profile

    def validate(self) -> BoneDirectionProfile:
        layout = get_layout(self.layout_name)
        if len(self.bones) != sum(len(path) - 1 for path in self.fingers.values()):
            raise ValueError(f"{self.profile_id}: bone count does not match semantic paths")
        if len(set(item.name for item in self.bones)) != len(self.bones):
            raise ValueError(f"{self.profile_id}: bone names must be unique")
        if len(set(item.name for item in self.pairs)) != len(self.pairs):
            raise ValueError(f"{self.profile_id}: pair names must be unique")
        for pair in self.pairs:
            first, second = self.bones[pair.first_bone], self.bones[pair.second_bone]
            if first.finger != pair.finger or second.finger != pair.finger:
                raise ValueError(f"{self.profile_id}: cross-finger pair {pair.name}")
            if pair.second_bone != pair.first_bone + 1:
                raise ValueError(f"{self.profile_id}: pair is not consecutive")
        if layout.point_count != 21:
            raise ValueError("Stage 7 requires MediaPipe-21")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "layout_name": self.layout_name,
            "include_wrist_to_mcp": self.include_wrist_to_mcp,
            "fingers": {key: list(value) for key, value in self.fingers.items()},
            "bones": [item.__dict__ for item in self.bones],
            "pairs": [item.__dict__ for item in self.pairs],
            "assumptions": list(self.assumptions),
            "notes": self.notes,
            "profile_hash": self.profile_hash,
        }


@dataclass
class BoneFeatures:
    local_keypoints: Any
    bone_vectors: Any
    bone_lengths: Any
    unit_directions: Any
    adjacent_features: Any
    valid_bones: Any
    valid_pairs: Any
    frame_transform: Any
    bone_names: tuple[str, ...]
    pair_names: tuple[str, ...]
    pair_fingers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if hasattr(value, "detach"):
                return value.detach().cpu().tolist()
            if hasattr(value, "tolist"):
                return value.tolist()
            return value

        return {
            key: convert(value)
            for key, value in self.__dict__.items()
            if key not in {"bone_names", "pair_names", "pair_fingers"}
        } | {
            "bone_names": list(self.bone_names),
            "pair_names": list(self.pair_names),
            "pair_fingers": list(self.pair_fingers),
        }


def extract_bone_features(
    keypoints: Any,
    frame_profile: Any,
    bone_profile: BoneDirectionProfile,
    *,
    side: str = "right",
    frame_transform: Any | None = None,
    strict: bool = True,
    zero_length_threshold_m: float = 1e-10,
) -> BoneFeatures:
    if frame_transform is None:
        frame_transform = frame_profile.frame_transform(keypoints, side=side, strict=strict)
    local = frame_profile.to_local(keypoints, frame_transform)
    parents = np.asarray([item.parent_index for item in bone_profile.bones], dtype=np.int64)
    children = np.asarray([item.child_index for item in bone_profile.bones], dtype=np.int64)
    is_torch = type(local).__module__.split(".")[0] == "torch"
    if is_torch:
        import torch

        p = torch.as_tensor(parents, dtype=torch.long, device=local.device)
        c = torch.as_tensor(children, dtype=torch.long, device=local.device)
        vectors = local[..., c, :] - local[..., p, :]
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        valid_bones = lengths > zero_length_threshold_m
        if strict and bool(torch.any(~valid_bones).detach().cpu()):
            bad = torch.nonzero((~valid_bones).detach().cpu(), as_tuple=False).tolist()
            raise ZeroLengthBoneError(f"{bone_profile.profile_id}: zero-length bones at {bad}")
        directions = vectors / torch.clamp(lengths, min=zero_length_threshold_m)[..., None]
        first_t = torch.as_tensor(
            [item.first_bone for item in bone_profile.pairs], dtype=torch.long, device=local.device
        )
        second_t = torch.as_tensor(
            [item.second_bone for item in bone_profile.pairs], dtype=torch.long, device=local.device
        )
        features = directions[..., first_t, :] - directions[..., second_t, :]
        valid_pairs = valid_bones[..., first_t] & valid_bones[..., second_t]
    else:
        points = np.asarray(local)
        vectors = points[..., children, :] - points[..., parents, :]
        lengths = np.linalg.norm(vectors, axis=-1)
        valid_bones = lengths > zero_length_threshold_m
        if strict and np.any(~valid_bones):
            raise ZeroLengthBoneError(
                f"{bone_profile.profile_id}: zero-length bones at "
                f"{np.argwhere(~valid_bones).tolist()}"
            )
        directions = vectors / np.maximum(lengths, zero_length_threshold_m)[..., None]
        first_np = np.asarray([item.first_bone for item in bone_profile.pairs], dtype=np.int64)
        second_np = np.asarray([item.second_bone for item in bone_profile.pairs], dtype=np.int64)
        features = directions[..., first_np, :] - directions[..., second_np, :]
        valid_pairs = valid_bones[..., first_np] & valid_bones[..., second_np]
    return BoneFeatures(
        local_keypoints=local,
        bone_vectors=vectors,
        bone_lengths=lengths,
        unit_directions=directions,
        adjacent_features=features,
        valid_bones=valid_bones,
        valid_pairs=valid_pairs,
        frame_transform=frame_transform,
        bone_names=tuple(item.name for item in bone_profile.bones),
        pair_names=tuple(item.name for item in bone_profile.pairs),
        pair_fingers=tuple(item.finger for item in bone_profile.pairs),
    )


def load_bone_profile(
    profile_id: str, *, config_root: str | Path | None = None
) -> BoneDirectionProfile:
    root = (
        Path(config_root)
        if config_root is not None
        else Path(__file__).resolve().parents[3] / "configs" / "retarget" / "bones"
    )
    path = root / f"{profile_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"bone profile not found: {profile_id}")
    raw = path.read_bytes()
    values = yaml.safe_load(raw) or {}
    if not isinstance(values, dict):
        raise ValueError(f"bone profile must be a mapping: {path}")
    return BoneDirectionProfile.from_mapping(values, source_path=path, raw=raw)


__all__ = [
    "BoneDirectionProfile",
    "BoneFeatures",
    "BonePair",
    "DirectedBone",
    "ZeroLengthBoneError",
    "extract_bone_features",
    "load_bone_profile",
]
