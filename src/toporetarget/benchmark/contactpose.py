"""Lazy ContactPose adapter with explicit unavailable-field reporting."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import file_hash


class ContactPoseAdapterError(RuntimeError):
    """Raised when a selected ContactPose unit cannot be converted safely."""


class ContactPoseDatasetAdapter:
    """Index one local ContactPose installation without modifying its files."""

    excluded_names = {"mug", "scissors", "utah_teapot", "utah teapot", "utah-teapot"}

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.excluded_objects: set[str] = set()
        self.scan_truncated = False

    def _files(self) -> Iterator[Path]:
        """Bounded discovery that avoids recursing into media/cache trees."""

        base = (
            self.root / "contactpose_data"
            if (self.root / "contactpose_data").is_dir()
            else self.root
        )
        if not base.is_dir():
            return
        interesting = {
            "annotations.json",
            "mano_fits_15.json",
            "mano_fits_10.json",
            "license.txt",
            "readme.md",
        }
        preferred_collections = (
            "full47_use",
            "full28_use",
            "full5_use",
            "full10_use",
            "full11_use",
        )
        collections = [base / name for name in preferred_collections if (base / name).is_dir()]
        for collection in collections:
            try:
                objects = sorted(os.scandir(collection), key=lambda item: item.name)
            except OSError:
                continue
            if len(objects) > 128:
                self.scan_truncated = True
                objects = objects[:128]
            for object_entry in objects:
                try:
                    if not object_entry.is_dir(follow_symlinks=False):
                        continue
                    entries = sorted(os.scandir(object_entry.path), key=lambda item: item.name)
                except OSError:
                    continue
                for entry in entries:
                    path = Path(entry.path)
                    if entry.name.lower() in interesting or path.suffix.lower() in {
                        ".ply",
                        ".obj",
                        ".stl",
                    }:
                        yield path

    @staticmethod
    def _object_from_path(path: Path) -> str:
        tokens = [item for item in path.parts if item not in {"contactpose_data", "data"}]
        if path.parent.name and path.parent.name not in {"annotations", "grasps", "data"}:
            return path.parent.name
        return tokens[-2] if len(tokens) >= 2 else "unknown"

    @staticmethod
    def _json_contacts(path: Path) -> tuple[str, list[str], dict[str, Any]]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return "unavailable", [], {"error": f"json_read:{type(exc).__name__}:{exc}"}
        text = json.dumps(value, sort_keys=True).lower()
        regions = []
        for name in ("thumb", "index", "middle", "ring", "pinky", "palm", "hand"):
            if name in text:
                regions.append(name)
        return (
            "native_json",
            sorted(set(regions)),
            {
                "json_keys": sorted(value) if isinstance(value, dict) else [],
                "native_contact_field_present": any(
                    token in text for token in ("contact", "in_contact", "bone")
                ),
            },
        )

    def inspect(self) -> dict[str, Any]:
        files = list(self._files())
        licenses = [
            str(path)
            for path in files
            if path.name.lower() in {"license", "license.txt", "license.md", "readme", "readme.md"}
        ]
        annotation_files = [
            path for path in files if path.name.lower() in {"annotations.json", "mano_fits_15.json"}
        ]
        mesh_files = [path for path in files if path.suffix.lower() in {".ply", ".obj", ".stl"}]
        object_names = sorted({self._object_from_path(path).lower() for path in mesh_files})
        self.excluded_objects = {
            name
            for name in object_names
            if name in self.excluded_names or any(ex in name for ex in self.excluded_names)
        }
        return {
            "schema_version": "toporetarget.contactpose.audit.v1",
            "root": str(self.root),
            "data_root": str(self.root / "contactpose_data"),
            "exists": self.root.is_dir(),
            "file_count_bounded_scan": len(files),
            "license_files": licenses[:50],
            "annotation_file_count": len(annotation_files),
            "mesh_file_count": len(mesh_files),
            "scan_truncated": self.scan_truncated,
            "object_names": object_names,
            "excluded_deep_concave_diagnostic_set": sorted(self.excluded_objects),
            "source_modification": False,
            "notes": "Directory and schema are discovered locally; absent official fields remain unavailable.",
        }

    def index(self) -> list[dict[str, Any]]:
        files = list(self._files())
        annotations = [path for path in files if path.name.lower() == "annotations.json"]
        if not annotations:
            annotations = [path for path in files if path.name.lower() == "mano_fits_15.json"]
        mesh_by_object: dict[str, Path] = {}
        for path in files:
            if path.suffix.lower() not in {".ply", ".obj", ".stl"}:
                continue
            mesh_by_object.setdefault(self._object_from_path(path).lower(), path)
        candidates: list[dict[str, Any]] = []
        for path in sorted(annotations):
            object_name = self._object_from_path(path).lower()
            contact_type, regions, details = self._json_contacts(path)
            mesh = mesh_by_object.get(object_name)
            excluded = object_name in self.excluded_objects or any(
                ex in object_name for ex in self.excluded_names
            )
            reasons: list[str] = []
            if excluded:
                reasons.append("excluded_deep_concave_diagnostic_set")
            if mesh is None:
                reasons.append("object_mesh_unresolved")
            if contact_type == "unavailable" or not details.get("native_contact_field_present"):
                reasons.append("official_contact_annotation_unavailable_or_unrecognized")
            if not regions or not ("thumb" in regions or len(regions) >= 2):
                reasons.append("thumb_or_two_finger_regions_unavailable")
            grasp_id = path.parent.name
            candidates.append(
                {
                    "native_sample_id": str(path.relative_to(self.root)),
                    "source_path": str(path),
                    "source_hash": file_hash(path),
                    "subject": path.parts[-3] if len(path.parts) >= 3 else "unknown",
                    "object_name": object_name,
                    "grasp_name": grasp_id,
                    "hand": "right",
                    "side": "right",
                    "dynamic": False,
                    "native_static_grasp": True,
                    "temporal_metrics_applicable": False,
                    "start": 0,
                    "end": 1,
                    "native_fps": None,
                    "object_mesh_path": None if mesh is None else str(mesh),
                    "object_mesh_hash": None if mesh is None else file_hash(mesh),
                    "contact_annotation_type": contact_type,
                    "contact_annotation_hash": file_hash(path),
                    "contact_regions": regions,
                    "contact_mode": "native_annotation" if not reasons else "unavailable",
                    "canonical_validity": "pending",
                    "sdf_validity": "pending" if mesh else "invalid",
                    "selection_score": {"contact_region_diversity": float(len(regions))},
                    "selection_reasons": [],
                    "rejection_reasons": reasons,
                    "details": details,
                }
            )
        return candidates

    def select(self, candidates: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
        valid = [item for item in candidates if not item.get("rejection_reasons")]
        valid.sort(
            key=lambda item: (
                -len(item.get("contact_regions", [])),
                -int("thumb" in item.get("contact_regions", [])),
                str(item.get("object_name", "")),
                str(item.get("native_sample_id", "")),
            )
        )
        selected: list[dict[str, Any]] = []
        objects: set[str] = set()
        for item in valid:
            if len(selected) >= target:
                break
            if (
                item.get("object_name") in objects
                and len({i.get("object_name") for i in selected}) < 3
            ):
                continue
            item["benchmark_id"] = (
                f"contactpose_{Path(str(item['native_sample_id'])).stem}_{len(selected) + 1:02d}"
            )
            item["frozen_selection_rank"] = len(selected) + 1
            item["selection_reasons"] = [
                "native contact annotation parsed",
                "static T=1 semantics",
                "object identity resolved",
            ]
            selected.append(item)
            objects.add(str(item.get("object_name")))
        return selected


__all__ = ["ContactPoseAdapterError", "ContactPoseDatasetAdapter"]
