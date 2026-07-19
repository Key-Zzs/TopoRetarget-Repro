"""Lossless GRAB contact loading and official semantic label conversion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.schema import ContactTrack


class ContactLoadError(RuntimeError):
    """Raised when source contact data cannot be aligned or semantically mapped."""


def _default_mapping_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "datasets" / "grab_contact_parts.yaml"


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GrabContactMapping:
    """Versioned mapping extracted from the official GRAB tools source."""

    config_path: Path
    mapping_id: str
    mapping_version: str
    source: dict[str, Any]
    no_contact_label: int
    unknown_semantic_id: int
    expected_label_range: tuple[int, int]
    labels: dict[int, dict[str, Any]]
    config_sha256: str

    @property
    def source_sha256(self) -> str | None:
        value = self.source.get("file_sha256")
        return None if value is None else str(value)

    @property
    def label_min(self) -> int:
        return self.expected_label_range[0]

    @property
    def label_max(self) -> int:
        return self.expected_label_range[1]

    def table(self, *, include_unknown: bool = False) -> dict[int, dict[str, Any]]:
        result = {key: dict(value) for key, value in self.labels.items()}
        if include_unknown and self.unknown_semantic_id not in result:
            result[self.unknown_semantic_id] = {
                "id": self.unknown_semantic_id,
                "name": "unknown",
                "category": "unknown",
                "side": None,
                "is_hand": False,
                "is_body": False,
                "is_contact": True,
                "source_name": None,
                "notes": "Non-strict fallback for a label outside the verified official range",
            }
        return result


def load_grab_contact_mapping(path: str | Path | None = None) -> GrabContactMapping:
    """Load and validate the tracked official GRAB contact mapping."""

    config_path = _default_mapping_path() if path is None else Path(path).expanduser()
    if not config_path.is_file():
        raise ContactLoadError(f"GRAB contact mapping config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ContactLoadError(
            f"could not read GRAB contact mapping: {config_path}: {exc}"
        ) from exc
    if payload.get("mapping_status") == "unavailable":
        raise ContactLoadError(f"GRAB contact mapping is unavailable in {config_path}")
    try:
        expected = tuple(int(item) for item in payload["expected_label_range"])
        raw_labels = payload["labels"]
        labels: dict[int, dict[str, Any]] = {}
        required = {
            "id",
            "name",
            "category",
            "side",
            "is_hand",
            "is_body",
            "is_contact",
            "source_name",
            "notes",
        }
        for item in raw_labels:
            if not required.issubset(item):
                raise ValueError(f"label entry is missing fields: {item}")
            label_id = int(item["id"])
            if label_id in labels:
                raise ValueError(f"duplicate GRAB contact label: {label_id}")
            if label_id < expected[0] or label_id > expected[1]:
                raise ValueError(f"label {label_id} is outside expected range {expected}")
            labels[label_id] = {key: item[key] for key in required}
        expected_labels = set(range(expected[0], expected[1] + 1))
        if set(labels) != expected_labels:
            raise ValueError(
                f"mapping does not cover expected labels: {sorted(expected_labels - set(labels))}"
            )
        no_contact = int(payload["no_contact_label"])
        if labels[no_contact]["category"] != "none" or labels[no_contact]["is_contact"]:
            raise ValueError("no_contact_label must be a non-contact entry")
        unknown_id = int(payload["unknown_semantic_id"])
        if unknown_id in labels:
            raise ValueError("unknown_semantic_id must not collide with official labels")
        source = dict(payload["source"])
        mapping_id = str(payload["mapping_id"])
        mapping_version = str(payload["mapping_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContactLoadError(f"invalid GRAB contact mapping {config_path}: {exc}") from exc
    return GrabContactMapping(
        config_path=config_path,
        mapping_id=mapping_id,
        mapping_version=mapping_version,
        source=source,
        no_contact_label=no_contact,
        unknown_semantic_id=unknown_id,
        expected_label_range=(expected[0], expected[1]),
        labels=labels,
        config_sha256=_config_hash(config_path),
    )


def _integer_labels(value: Any) -> np.ndarray:
    labels = np.asarray(value)
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.issubdtype(labels.dtype, np.number) or not np.all(labels == np.floor(labels)):
            raise ContactLoadError("GRAB contact labels must be integer-valued")
    return labels.astype(np.int64, copy=False)


def _semantic_labels(
    labels: np.ndarray,
    mapping: GrabContactMapping,
    *,
    strict: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    observed = sorted(int(item) for item in np.unique(labels))
    mapped = sorted(item for item in observed if item in mapping.labels)
    unmapped = sorted(set(observed) - set(mapped))
    if unmapped and strict:
        raise ContactLoadError(
            "GRAB semantic contact mapping has unmapped labels: "
            + ", ".join(str(item) for item in unmapped)
        )
    semantic = labels.astype(np.int64, copy=True)
    if unmapped:
        semantic[np.isin(labels, unmapped)] = mapping.unknown_semantic_id
    return semantic, {
        "observed_labels": observed,
        "mapped_labels": mapped,
        "unmapped_labels": unmapped,
        "unmapped_vertex_count": int(np.count_nonzero(np.isin(labels, unmapped))),
        "unmapped_frame_count": int(np.count_nonzero(np.any(np.isin(labels, unmapped), axis=-1)))
        if labels.ndim > 1
        else int(bool(unmapped)),
        "fully_mapped": not unmapped,
    }


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
    """Create lossless contact tracks for the selected hands.

    The source object contact array is shared by source hands.  Each selected
    hand receives an independent copy so callers cannot accidentally mutate a
    second hand's contact data.  Semantic IDs intentionally reuse the official
    numeric IDs; the mapping table carries the names and categories.
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
    labels = _integer_labels(raw["object"])
    if labels.ndim != 2 or labels.shape[0] != frame_count:
        raise ContactLoadError(
            f"GRAB contact.object must have shape [{frame_count},V], got {labels.shape}"
        )
    if labels.shape[1] != object_vertex_count:
        raise ContactLoadError(
            "GRAB contact/object mesh vertex mismatch: "
            f"labels have {labels.shape[1]}, mesh has {object_vertex_count}"
        )
    mapping: GrabContactMapping | None = None
    semantic_labels: np.ndarray | None = None
    semantic_details: dict[str, Any] = {}
    if mode == "semantic":
        mapping = load_grab_contact_mapping(mapping_config)
        semantic_labels, semantic_details = _semantic_labels(labels, mapping, strict=strict)
    no_contact_label = mapping.no_contact_label if mapping is not None else 0
    binary = labels != no_contact_label
    body = np.asarray(raw["body"]) if "body" in raw else None
    threshold = raw.get("threshold")
    tracks: list[ContactTrack] = []
    for hand_id in hand_ids:
        metadata: dict[str, Any] = {
            "mode": mode,
            "raw_labels_preserved": True,
            "binary_is_derived": True,
            "source_body_labels": None if body is None else body.copy(),
            "threshold": None if threshold is None else np.asarray(threshold).item(),
            "unique_numeric_labels": np.unique(labels).astype(int).tolist(),
            "no_contact_label": no_contact_label,
        }
        if mapping is None:
            metadata.update(
                {
                    "semantic_mapping_status": "not_requested",
                    "mapping_config": None,
                }
            )
        else:
            metadata.update(
                {
                    "semantic_mapping_status": (
                        "fully_mapped" if semantic_details["fully_mapped"] else "non_strict_unknown"
                    ),
                    "mapping_config": str(mapping.config_path),
                    "mapping_id": mapping.mapping_id,
                    "mapping_version": mapping.mapping_version,
                    "mapping_config_sha256": mapping.config_sha256,
                    "source_mapping_hash": mapping.source_sha256,
                    "expected_label_range": list(mapping.expected_label_range),
                    "source_mapping": mapping.source,
                    **semantic_details,
                }
            )
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
                semantic_mapping=None
                if mapping is None
                else mapping.table(include_unknown=not semantic_details.get("fully_mapped", True)),
                metadata=metadata,
            )
        )
    return tracks


__all__ = [
    "ContactLoadError",
    "GrabContactMapping",
    "build_grab_contacts",
    "load_grab_contact_mapping",
]
