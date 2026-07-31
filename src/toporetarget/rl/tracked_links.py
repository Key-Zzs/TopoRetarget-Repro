"""Frozen, non-clip-specific tracked-link profile."""

import numpy as np

TRACKED_LINK_PROFILE_ID = "tracked_link_profile_v1"
TRACKED_LINKS_WUJI_RH = (
    "r_wrist",
    "r_thumb_proximal",
    "r_thumb_middle",
    "r_thumb_distal",
    "r_index_finger_proximal",
    "r_index_finger_middle",
    "r_index_finger_distal",
    "r_middle_finger_proximal",
    "r_middle_finger_middle",
    "r_middle_finger_distal",
    "r_ring_finger_proximal",
    "r_ring_finger_middle",
    "r_ring_finger_distal",
    "r_pinky_proximal",
    "r_pinky_middle",
    "r_pinky_distal",
)


def select_tracked_links(
    positions: np.ndarray,
    source_names: tuple[str, ...],
    *,
    profile: tuple[str, ...] = TRACKED_LINKS_WUJI_RH,
) -> np.ndarray:
    """Select the globally frozen profile, failing closed on a missing link."""

    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != len(source_names) or values.shape[2] != 3:
        raise ValueError("source link positions must have shape [T,L,3] matching source_names")
    lookup = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in profile if name not in lookup]
    if missing:
        raise ValueError(f"tracked_link_profile_v1 missing source links: {missing}")
    return values[:, [lookup[name] for name in profile], :]


__all__ = ["TRACKED_LINK_PROFILE_ID", "TRACKED_LINKS_WUJI_RH", "select_tracked_links"]
