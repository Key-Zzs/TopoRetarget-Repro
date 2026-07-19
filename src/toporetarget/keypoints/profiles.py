"""Versioned MANO-to-target mapping profiles."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toporetarget.keypoints.layouts import KeypointLayoutDefinition


@dataclass(frozen=True)
class FingertipAnchor:
    target_semantic: str
    source_type: str
    vertex_index: int
    source_evidence: str
    compatible_topology: str
    side_applicability: str
    verification_status: str

    @classmethod
    def from_mapping(cls, target_semantic: str, values: dict[str, Any]) -> FingertipAnchor:
        return cls(
            target_semantic=target_semantic,
            source_type=str(values["source_type"]),
            vertex_index=int(values["vertex_index"]),
            source_evidence=str(values.get("source_evidence", "")),
            compatible_topology=str(values.get("compatible_topology", "")),
            side_applicability=str(values.get("side_applicability", "left and right")),
            verification_status=str(values.get("verification_status", "unverified")),
        )


@dataclass(frozen=True)
class MappingProfile:
    profile_id: str
    version: str
    source_model: str
    source_model_version: str
    expected_vertex_count: int
    source_joint_layout: str
    target_layout: str
    mapping_mode: str
    joint_mapping: dict[str, str]
    fingertip_mapping: dict[str, FingertipAnchor]
    semantic_notes: tuple[str, ...]
    assumptions: tuple[str, ...]
    provenance: dict[str, Any]
    verification: dict[str, Any]
    path: Path
    sha256: str

    @classmethod
    def from_mapping(cls, values: dict[str, Any], *, path: Path) -> MappingProfile:
        tips = {
            name: FingertipAnchor.from_mapping(name, item)
            for name, item in dict(values.get("fingertip_mapping", {})).items()
        }
        result = cls(
            profile_id=str(values["profile_id"]),
            version=str(values["version"]),
            source_model=str(values["source_model"]),
            source_model_version=str(values["source_model_version"]),
            expected_vertex_count=int(values["expected_vertex_count"]),
            source_joint_layout=str(values["source_joint_layout"]),
            target_layout=str(values["target_layout"]),
            mapping_mode=str(values["mapping_mode"]),
            joint_mapping={
                str(key): str(value) for key, value in dict(values["joint_mapping"]).items()
            },
            fingertip_mapping=tips,
            semantic_notes=tuple(str(item) for item in values.get("semantic_notes", [])),
            assumptions=tuple(str(item) for item in values.get("assumptions", [])),
            provenance=dict(values.get("provenance", {})),
            verification=dict(values.get("verification", {})),
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        return result

    def validate(self, layouts: dict[str, KeypointLayoutDefinition]) -> MappingProfile:
        if self.mapping_mode not in {
            "joint_map_plus_tip_vertices",
            "validated_mano21_reorder",
            "vertices_with_joint_regressor",
        }:
            raise ValueError(f"unsupported mapping mode: {self.mapping_mode}")
        source_layout = layouts.get(self.source_joint_layout)
        target_layout = layouts.get(self.target_layout)
        if source_layout is None:
            raise ValueError(
                f"profile {self.profile_id}: unknown source layout {self.source_joint_layout}"
            )
        if target_layout is None:
            raise ValueError(
                f"profile {self.profile_id}: unknown target layout {self.target_layout}"
            )
        target_names = set(target_layout.semantic_names)
        if not set(self.joint_mapping).issubset(target_names):
            raise ValueError(
                f"profile {self.profile_id}: joint mapping has unknown target semantic"
            )
        for _target, source in self.joint_mapping.items():
            if source not in source_layout.semantic_names:
                raise ValueError(f"profile {self.profile_id}: unknown source semantic {source}")
        if self.mapping_mode == "validated_mano21_reorder":
            if len(source_layout.semantic_names) != 21 or target_layout.point_count != 21:
                raise ValueError(
                    f"profile {self.profile_id}: validated MANO-21 reorder requires two "
                    "21-point layouts"
                )
            if set(self.joint_mapping) != target_names:
                raise ValueError(
                    f"profile {self.profile_id}: validated MANO-21 mapping must cover every target"
                )
        else:
            tips = {
                target_layout.semantic_names[index] for index in target_layout.fingertip_indices
            }
            if target_names - tips != set(self.joint_mapping):
                raise ValueError(f"profile {self.profile_id}: non-tip mapping is incomplete")
        for target, anchor in self.fingertip_mapping.items():
            if target not in target_names:
                raise ValueError(f"profile {self.profile_id}: unknown tip semantic {target}")
            if anchor.source_type != "vertex":
                raise ValueError(
                    f"profile {self.profile_id}: only vertex tip anchors are supported"
                )
            if not 0 <= anchor.vertex_index < self.expected_vertex_count:
                raise ValueError(f"profile {self.profile_id}: tip vertex is out of range")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "source_model": self.source_model,
            "source_model_version": self.source_model_version,
            "expected_vertex_count": self.expected_vertex_count,
            "source_joint_layout": self.source_joint_layout,
            "target_layout": self.target_layout,
            "mapping_mode": self.mapping_mode,
            "joint_mapping": dict(self.joint_mapping),
            "fingertip_mapping": {
                key: {
                    "source_type": value.source_type,
                    "vertex_index": value.vertex_index,
                    "source_evidence": value.source_evidence,
                    "compatible_topology": value.compatible_topology,
                    "side_applicability": value.side_applicability,
                    "verification_status": value.verification_status,
                }
                for key, value in self.fingertip_mapping.items()
            },
            "semantic_notes": list(self.semantic_notes),
            "assumptions": list(self.assumptions),
            "provenance": dict(self.provenance),
            "verification": dict(self.verification),
            "profile_hash": self.sha256,
        }


__all__ = ["FingertipAnchor", "MappingProfile"]
