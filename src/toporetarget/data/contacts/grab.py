"""Preserve GRAB's numeric contact labels without inventing semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import ContactTrack


class ContactLoadError(RuntimeError):
    """Raised when source contact data cannot be aligned or semantically mapped."""


def build_grab_contacts(
    auxiliary: dict[str, Any],
    *,
    hand_ids: list[str],
    object_id: str,
    object_vertex_count: int,
    frame_count: int,
    mode: str,
    strict: bool = True,
    mapping_config: str | Path | None = None,
) -> list[ContactTrack]:
    """Create one lossless contact track per selected hand.

    The GRAB object contact array is shared by the source hands.  It is kept
    as raw numeric labels and duplicated by value (never by mutable array
    reference) for each selected hand.  ``binary`` is explicitly derived.
    """

    if mode == "none":
        return []
    if mode not in {"source", "binary", "semantic"}:
        raise ContactLoadError(f"unsupported contact mode: {mode}")
    raw = auxiliary.get("contact")
    if not isinstance(raw, dict) or "object" not in raw:
        if strict:
            raise ContactLoadError(
                "GRAB contact mode requested but source contact.object is absent"
            )
        return []
    labels = np.asarray(raw["object"])
    if labels.ndim != 2 or labels.shape[0] != frame_count:
        raise ContactLoadError(
            f"GRAB contact.object must have shape [{frame_count},V], got {labels.shape}"
        )
    if labels.shape[1] != object_vertex_count:
        raise ContactLoadError(
            "GRAB contact/object mesh vertex mismatch: "
            f"labels have {labels.shape[1]}, mesh has {object_vertex_count}"
        )
    semantic_labels: np.ndarray | None = None
    semantic_status = "not_requested"
    if mode == "semantic":
        semantic_status = "unavailable_no_verified_official_mapping"
        message = (
            "semantic GRAB contact mapping is unavailable; numeric labels are preserved, "
            "but no label names are guessed"
        )
        if strict:
            raise ContactLoadError(message)
    binary = labels != 0
    body = np.asarray(raw["body"]) if "body" in raw else None
    threshold = raw.get("threshold")
    tracks: list[ContactTrack] = []
    for hand_id in hand_ids:
        tracks.append(
            ContactTrack(
                hand_id=hand_id,
                object_id=object_id,
                source_contact_representation="grab_numeric_per_object_vertex",
                valid=np.ones(frame_count, dtype=bool),
                labels=labels.copy(),
                vertex_associations=np.arange(object_vertex_count, dtype=np.int64),
                binary=binary.copy(),
                semantic_labels=None if semantic_labels is None else semantic_labels.copy(),
                metadata={
                    "mode": mode,
                    "raw_labels_preserved": True,
                    "binary_is_derived": True,
                    "source_body_labels": None if body is None else body.copy(),
                    "threshold": None if threshold is None else np.asarray(threshold).item(),
                    "semantic_mapping_status": semantic_status,
                    "mapping_config": None if mapping_config is None else str(mapping_config),
                    "unique_numeric_labels": np.unique(labels).astype(int).tolist(),
                },
            )
        )
    return tracks


__all__ = ["ContactLoadError", "build_grab_contacts"]
