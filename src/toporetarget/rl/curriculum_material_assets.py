"""Materialize immutable, friction-only HOCap USD variants for P3/P4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .gravity_friction_curriculum import Stage16GravityFrictionCurriculumV1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format_usda_float(value: float) -> str:
    if value <= 0.0:
        raise ValueError("CURRICULUM_OBJECT_FRICTION_MUST_BE_POSITIVE")
    return f"{value:.8g}"


def materialize_curriculum_object_assets(
    *,
    repo_root: Path,
    contract: Stage16GravityFrictionCurriculumV1,
    stage: str,
) -> dict[str, dict[str, object]]:
    """Create cacheable USD variants whose only authored differences are friction.

    The active HOCap material is authored inside a referenced USD.  Editing the
    instance proxy during Isaac scene construction is unsupported, so the
    curriculum instead materializes one provenance-recorded input USD per
    stage.  Mass, inertia, restitution, damping, collision geometry and every
    other source byte remain unchanged.
    """

    physics = contract.physics(stage)
    roles = physics["material_roles"]
    if not isinstance(roles, dict):
        raise RuntimeError("CURRICULUM_ASSET_MATERIAL_ROLES_INVALID")
    object_role = roles.get("hocap_bound_object_material")
    if not isinstance(object_role, dict):
        raise RuntimeError("CURRICULUM_ASSET_OBJECT_ROLE_MISSING")
    static = _format_usda_float(float(object_role["static_friction"]))
    dynamic = _format_usda_float(float(object_role["dynamic_friction"]))
    if float(object_role["restitution"]) != 0.0:
        raise RuntimeError("CURRICULUM_ASSET_RESTITUTION_DRIFT")
    asset_root = repo_root / ".local/generated_assets/isaaclab"
    output_root = asset_root / "stage16_gravity_friction_curriculum_v1" / stage
    result: dict[str, dict[str, object]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        source = asset_root / clip / f"{clip}.usda"
        if not source.is_file():
            raise FileNotFoundError(f"CURRICULUM_OBJECT_SOURCE_USD_MISSING:{source}")
        original = source.read_text(encoding="utf-8")
        expected_static = "float physics:staticFriction = 1"
        expected_dynamic = "float physics:dynamicFriction = 1"
        if original.count(expected_static) != 1 or original.count(expected_dynamic) != 1:
            raise RuntimeError(f"CURRICULUM_OBJECT_SOURCE_FRICTION_CONTRACT_DRIFT:{clip}")
        derived = original.replace(expected_static, f"float physics:staticFriction = {static}")
        derived = derived.replace(expected_dynamic, f"float physics:dynamicFriction = {dynamic}")
        target = output_root / clip / f"{clip}.usda"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text(encoding="utf-8") != derived:
            target.write_text(derived, encoding="utf-8")
        payload = {
            "source_usd": str(source.resolve()),
            "source_sha256": _sha256(source),
            "derived_usd": str(target.resolve()),
            "derived_sha256": _sha256(target),
            "curriculum_stage": stage,
            "static_friction": float(object_role["static_friction"]),
            "dynamic_friction": float(object_role["dynamic_friction"]),
            "restitution": 0.0,
            "allowed_source_text_changes": [
                "physics:staticFriction",
                "physics:dynamicFriction",
            ],
        }
        receipt = target.with_suffix(".material_receipt.json")
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[clip] = payload
    return result


__all__ = ["materialize_curriculum_object_assets"]
