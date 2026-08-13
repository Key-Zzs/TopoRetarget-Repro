"""Named full-hand contact telemetry contracts for Stage 16-D R2."""

from __future__ import annotations

from pathlib import Path
from typing import Final

HAND_BODY_GROUPS: Final = (
    "palm_or_wrist",
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "other",
)
FINGERTIP_NAMES: Final = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)


def hand_body_group(name: str) -> str:
    """Derive one stable semantic group from the actual Wuji body name."""

    if name == "r_wrist":
        return "palm_or_wrist"
    for group in ("thumb", "index", "middle", "ring", "pinky"):
        if f"r_{group}" in name or f"r_{group}_finger" in name:
            return group
    return "other"


def hand_body_manifest(body_names: tuple[str, ...], *, repo_root: Path) -> dict[str, object]:
    """Return a collision manifest without treating a root link as a palm."""

    if len(body_names) != 21 or len(set(body_names)) != len(body_names):
        raise ValueError("FULL_HAND_CONTACT_BODY_MANIFEST_INVALID")
    asset = repo_root / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
    asset_text = asset.read_text(encoding="utf-8")
    wrist_has_mesh = '<body name="r_wrist">' in asset_text and 'mesh="r_wrist"' in asset_text
    return {
        "schema_version": "FullHandObjectPairTelemetryV1",
        "hand_body_names": list(body_names),
        "hand_body_indices": list(range(len(body_names))),
        "hand_body_groups": [hand_body_group(name) for name in body_names],
        "fingertip_body_names": list(FINGERTIP_NAMES),
        "collision_shape_mapping": {
            "source": str(asset),
            "collision_profile": "configs/robots/collision/wuji_hand2_beta1_mjcf_rh.yaml",
            "body_to_mesh_name": {name: name for name in body_names},
            "collision_source": "official_convex_hull_geoms",
        },
        "palm_mapping": {
            "palm_body_name": None,
            "palm_mapping_status": "PALM_CONTACT_BODY_UNAVAILABLE",
            "wrist_base_contact_body": "r_wrist" if wrist_has_mesh else None,
            "r_wrist_interpretation": "WRIST_BASE_CONTACT_BODY"
            if wrist_has_mesh
            else "UNAVAILABLE",
            "reason": (
                "The asset declares r_wrist as base_link/root_link with a mesh, but no distinct "
                "palm collision body is named in the 21-body filtered manifest."
            ),
        },
        "force_frame": "world",
        "force_units": "N",
        "force_semantics": "force on active object from named hand collision body",
    }
