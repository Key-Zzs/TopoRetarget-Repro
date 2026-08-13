"""Pure validation helpers shared by Stage 16-C.1 runtime scripts and tests."""

from __future__ import annotations

from typing import Any


def classify_c1_status(gates: dict[str, bool]) -> str:
    if gates and all(gates.values()):
        return "STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED"
    if any(gates.values()):
        return "STAGE16C1_ISAACLAB_ASSET_MIGRATION_PARTIAL"
    return "STAGE16C1_ISAACLAB_ASSET_MIGRATION_BLOCKED"


def classify_c2_entry(c1_status: str, *, entry_authorized: bool) -> str:
    """Authorize only C.2 entry; never imply that C.2 or later work ran."""
    if c1_status == "STAGE16C1_ISAACLAB_ASSET_MIGRATION_VALIDATED" and entry_authorized:
        return "STAGE16C2_DIRECT_RL_ENV_AUTHORIZED"
    return "STAGE16C2_DIRECT_RL_ENV_BLOCKED"


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    required = {
        "source_repo",
        "source_commit",
        "source_file",
        "license",
        "import_tool",
        "generated_usd",
        "generated_sha256",
        "root_prim",
        "articulation_root",
        "fixed_base",
        "body_names",
        "joint_names",
        "joint_order",
        "joint_types",
        "joint_axes",
        "limits",
        "default_pose",
        "drive_configuration",
        "collision_geoms",
        "visual_geoms",
        "mass_inertia",
        "warnings",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f"asset manifest missing fields: {missing}")
