"""Versioned generic target-hand profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _profile_hash(values: dict[str, Any]) -> str:
    payload = {key: value for key, value in values.items() if key != "qpos_order_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_qpos_order_profile(
    profile_id: str, *, config_root: str | Path, expected_dof_order: tuple[str, ...]
) -> dict[str, Any]:
    path = Path(config_root).expanduser() / profile_id
    if not path.is_file():
        raise FileNotFoundError(f"qpos order profile not found: {path}")
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError(f"qpos order profile must be a mapping: {path}")
    order = tuple(str(item) for item in values.get("dof_order", ()))
    if order != expected_dof_order:
        raise ValueError(f"qpos order profile {path} does not match RobotHandSpec.dof_order")
    if tuple(str(item) for item in values.get("urdf_joint_order", ())) != order:
        raise ValueError(f"qpos order profile {path} has an incomplete URDF order")
    if tuple(str(item) for item in values.get("mjcf_joint_order", ())) != order:
        raise ValueError(f"qpos order profile {path} has an incomplete MJCF order")
    limits = dict(values.get("limits", {}))
    if set(limits) != set(order):
        raise ValueError(f"qpos order profile {path} must declare limits for every DoF")
    declared_hash = str(values.get("qpos_order_hash", ""))
    computed_hash = _profile_hash(values)
    if declared_hash != computed_hash:
        raise ValueError(
            f"qpos order profile {path} hash mismatch: {declared_hash} != {computed_hash}"
        )
    values["path"] = str(path)
    values["computed_hash"] = computed_hash
    return values


__all__ = ["load_qpos_order_profile"]
