"""Versioned Stage 16-D self-collision contract and GPU capsule metric."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from toporetarget.rl.geometry_audit.runtime_geometry import load_runtime_geometry_manifest
from toporetarget.rl.geometry_audit.transforms import transform_points

from .contact_topology import body_contact_group

SELF_COLLISION_SCHEMA = "toporetarget.stage16d.self_collision_physics_contract.v1"
SELF_COLLISION_IDENTIFIER = "stage16d_inter_finger_self_collision_v1"
INTER_FINGER_METRIC = "pca_capsule_inter_finger_penetration_v1"
_FINGER_GROUPS = frozenset(("thumb", "index", "middle", "ring", "pinky"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SelfCollisionPhysicsContractV1:
    identifier: str
    enabled_self_collisions: bool
    source_asset_config_path: str
    source_asset_config_sha256: str
    generated_hand_usd_path: str
    generated_hand_usd_sha256: str
    runtime_collision_manifest_path: str
    runtime_collision_manifest_sha256: str
    physx_collision_filtering: str
    body_pair_scope: str
    excluded_pair_classes: tuple[str, ...]
    inter_finger_metric: str
    capsule_radius_scale: float
    maximum_inter_finger_penetration_m: float
    penalty_normalization_m: float
    schema_version: str = SELF_COLLISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SELF_COLLISION_SCHEMA:
            raise ValueError("unknown Stage16D self-collision contract schema")
        if self.identifier != SELF_COLLISION_IDENTIFIER:
            raise ValueError("unknown Stage16D self-collision contract identifier")
        if not self.enabled_self_collisions:
            raise ValueError("Stage16D self-collision contract must enable PhysX self collision")
        if self.physx_collision_filtering != "articulation_default_v1":
            raise ValueError("unsupported self-collision filtering contract")
        if self.body_pair_scope != "distinct_anatomical_fingers":
            raise ValueError("unsupported inter-finger pair scope")
        if set(self.excluded_pair_classes) != {"same_body", "same_finger", "wrist_or_palm"}:
            raise ValueError("self-collision exclusions must be complete and exact")
        if self.inter_finger_metric != INTER_FINGER_METRIC:
            raise ValueError("unsupported inter-finger penetration metric")
        if not 0.0 < self.capsule_radius_scale <= 1.0:
            raise ValueError("capsule radius scale must be in (0,1]")
        for name in (
            "maximum_inter_finger_penetration_m",
            "penalty_normalization_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 0.010:
                raise ValueError(f"{name} must be finite and in (0, 0.010]")

    @property
    def config_sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "config_sha256": self.config_sha256}


def _artifact_row(payload: dict[str, Any], name: str) -> tuple[str, str]:
    row = payload.get(name)
    if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
        raise ValueError(f"self-collision contract has malformed {name}")
    return str(row["path"]), str(row["sha256"])


def load_self_collision_contract(
    path: Path,
    *,
    repo_root: Path,
    validate_artifacts: bool = True,
) -> SelfCollisionPhysicsContractV1:
    """Load the exact Stage16D contract and optionally verify all authorities."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("self-collision contract must be a YAML mapping")
    source_path, source_hash = _artifact_row(payload, "source_asset_config")
    usd_path, usd_hash = _artifact_row(payload, "generated_hand_usd")
    manifest_path, manifest_hash = _artifact_row(payload, "runtime_collision_manifest")
    contract = SelfCollisionPhysicsContractV1(
        schema_version=str(payload.get("schema_version", "")),
        identifier=str(payload.get("identifier", "")),
        enabled_self_collisions=bool(payload.get("enabled_self_collisions", False)),
        source_asset_config_path=source_path,
        source_asset_config_sha256=source_hash,
        generated_hand_usd_path=usd_path,
        generated_hand_usd_sha256=usd_hash,
        runtime_collision_manifest_path=manifest_path,
        runtime_collision_manifest_sha256=manifest_hash,
        physx_collision_filtering=str(payload.get("physx_collision_filtering", "")),
        body_pair_scope=str(payload.get("body_pair_scope", "")),
        excluded_pair_classes=tuple(payload.get("excluded_pair_classes", ())),
        inter_finger_metric=str(payload.get("inter_finger_metric", "")),
        capsule_radius_scale=float(payload.get("capsule_radius_scale", math.nan)),
        maximum_inter_finger_penetration_m=float(
            payload.get("maximum_inter_finger_penetration_m", math.nan)
        ),
        penalty_normalization_m=float(payload.get("penalty_normalization_m", math.nan)),
    )
    if validate_artifacts:
        for name, relative, expected in (
            ("source_asset_config", source_path, source_hash),
            ("generated_hand_usd", usd_path, usd_hash),
            ("runtime_collision_manifest", manifest_path, manifest_hash),
        ):
            authority = (repo_root / relative).resolve()
            if not authority.is_file():
                raise FileNotFoundError(
                    f"SELF_COLLISION_CONTRACT_AUTHORITY_MISSING:{name}:{authority}"
                )
            actual = _sha256(authority)
            if actual != expected:
                raise ValueError(
                    f"SELF_COLLISION_CONTRACT_HASH_DRIFT:{name}:expected={expected}:actual={actual}"
                )
    return contract


def _quaternion_rotate_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        1.0e-12
    )
    xyz = quaternion[..., 1:]
    uv = torch.linalg.cross(xyz, vector, dim=-1)
    uuv = torch.linalg.cross(xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def capsule_segment_distance(
    first_start: torch.Tensor,
    first_end: torch.Tensor,
    second_start: torch.Tensor,
    second_end: torch.Tensor,
) -> torch.Tensor:
    """Return a stable closest distance for matching batches of line segments."""

    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa = (first * first).sum(dim=-1).clamp_min(1.0e-12)
    bb = (first * second).sum(dim=-1)
    cc = (second * second).sum(dim=-1).clamp_min(1.0e-12)
    dd = (first * offset).sum(dim=-1)
    ee = (second * offset).sum(dim=-1)
    determinant = aa * cc - bb * bb
    first_parameter = torch.where(
        determinant > 1.0e-12,
        ((bb * ee - cc * dd) / determinant.clamp_min(1.0e-12)).clamp(0.0, 1.0),
        torch.zeros_like(determinant),
    )
    second_parameter = ((bb * first_parameter + ee) / cc).clamp(0.0, 1.0)
    first_parameter = ((bb * second_parameter - dd) / aa).clamp(0.0, 1.0)
    second_parameter = ((bb * first_parameter + ee) / cc).clamp(0.0, 1.0)
    first_point = first_start + first_parameter[..., None] * first
    second_point = second_start + second_parameter[..., None] * second
    return torch.linalg.vector_norm(first_point - second_point, dim=-1)


class InterFingerCapsulePenetrationV1:
    """Cheap GPU inter-finger overlap proxy derived from C.1 convex bodies."""

    def __init__(
        self,
        *,
        body_names: tuple[str, ...],
        endpoint_start_local: torch.Tensor,
        endpoint_end_local: torch.Tensor,
        radii_m: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_names: tuple[str, ...],
    ) -> None:
        body_count = len(body_names)
        if endpoint_start_local.shape != (body_count, 3):
            raise ValueError("capsule start points do not match body order")
        if endpoint_end_local.shape != (body_count, 3) or radii_m.shape != (body_count,):
            raise ValueError("capsule geometry does not match body order")
        if (
            pair_indices.ndim != 2
            or pair_indices.shape[1] != 2
            or len(pair_names) != len(pair_indices)
        ):
            raise ValueError("inter-finger pair contract is malformed")
        self.body_names = body_names
        self.endpoint_start_local = endpoint_start_local
        self.endpoint_end_local = endpoint_end_local
        self.radii_m = radii_m
        self.pair_indices = pair_indices.long()
        self.pair_names = pair_names

    @classmethod
    def from_runtime_manifest(
        cls,
        path: Path,
        *,
        expected_body_names: tuple[str, ...],
        radius_scale: float,
        device: torch.device | str,
    ) -> InterFingerCapsulePenetrationV1:
        hand_proxies, _ = load_runtime_geometry_manifest(path)
        body_names = tuple(proxy.body_name for proxy in hand_proxies)
        if body_names != expected_body_names:
            raise ValueError("self-collision manifest body order differs from runtime articulation")
        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        radii: list[float] = []
        for proxy in hand_proxies:
            points = transform_points(proxy.scaled_vertices, proxy.local_pose_xyz_wxyz)
            center = points.mean(axis=0)
            covariance = np.cov((points - center).T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            projection = (points - center) @ axis
            radial = np.linalg.norm((points - center) - projection[:, None] * axis, axis=-1)
            radius = max(float(np.max(radial)) * radius_scale, 1.0e-5)
            midpoint = 0.5 * (float(projection.min()) + float(projection.max()))
            half_segment = max(
                0.5 * float(projection.max() - projection.min()) - radius,
                0.0,
            )
            starts.append(center + (midpoint - half_segment) * axis)
            ends.append(center + (midpoint + half_segment) * axis)
            radii.append(radius)
        pairs: list[tuple[int, int]] = []
        pair_names: list[str] = []
        groups = [body_contact_group(name) for name in body_names]
        for first in range(len(body_names)):
            for second in range(first + 1, len(body_names)):
                if (
                    groups[first] not in _FINGER_GROUPS
                    or groups[second] not in _FINGER_GROUPS
                    or groups[first] == groups[second]
                ):
                    continue
                pairs.append((first, second))
                pair_names.append(f"{body_names[first]}<->{body_names[second]}")
        if not pairs:
            raise ValueError("self-collision contract produced no inter-finger pairs")
        torch_device = torch.device(device)
        return cls(
            body_names=body_names,
            endpoint_start_local=torch.as_tensor(
                np.asarray(starts), dtype=torch.float32, device=torch_device
            ),
            endpoint_end_local=torch.as_tensor(
                np.asarray(ends), dtype=torch.float32, device=torch_device
            ),
            radii_m=torch.as_tensor(radii, dtype=torch.float32, device=torch_device),
            pair_indices=torch.as_tensor(pairs, dtype=torch.long, device=torch_device),
            pair_names=tuple(pair_names),
        )

    def evaluate(self, body_pose_world: torch.Tensor) -> dict[str, torch.Tensor]:
        if body_pose_world.ndim != 3 or body_pose_world.shape[1:] != (
            len(self.body_names),
            7,
        ):
            raise ValueError("inter-finger metric needs [envs,bodies,xyz+wxyz]")
        if not bool(torch.isfinite(body_pose_world).all()):
            raise ValueError("inter-finger body poses must be finite")
        position = body_pose_world[..., :3]
        quaternion = body_pose_world[..., 3:7]
        start = position + _quaternion_rotate_wxyz(
            quaternion, self.endpoint_start_local[None].expand(position.shape[0], -1, -1)
        )
        end = position + _quaternion_rotate_wxyz(
            quaternion, self.endpoint_end_local[None].expand(position.shape[0], -1, -1)
        )
        first = self.pair_indices[:, 0]
        second = self.pair_indices[:, 1]
        distance = capsule_segment_distance(
            start[:, first], end[:, first], start[:, second], end[:, second]
        )
        pair_penetration = (
            self.radii_m[first][None] + self.radii_m[second][None] - distance
        ).clamp_min(0.0)
        return {
            "pair_penetration_m": pair_penetration,
            "maximum_penetration_m": pair_penetration.max(dim=-1).values,
            "mean_squared_penetration_m2": pair_penetration.square().mean(dim=-1),
            "worst_pair_index": pair_penetration.argmax(dim=-1),
        }


__all__ = [
    "INTER_FINGER_METRIC",
    "SELF_COLLISION_IDENTIFIER",
    "SELF_COLLISION_SCHEMA",
    "InterFingerCapsulePenetrationV1",
    "SelfCollisionPhysicsContractV1",
    "capsule_segment_distance",
    "load_self_collision_contract",
]
