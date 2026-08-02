"""Simulator-independent contracts for Stage 16-C.1 asset migration."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_C1_FAILURES = frozenset(
    {
        "WUJI_SOURCE_ASSET_NOT_FOUND",
        "URDF_IMPORT_FAILURE",
        "ARTICULATION_ROOT_FAILURE",
        "FLOATING_ROOT_FAILURE",
        "JOINT_NAME_MAPPING_FAILURE",
        "JOINT_AXIS_OR_LIMIT_FAILURE",
        "DRIVE_CONFIGURATION_FAILURE",
        "COLLISION_IMPORT_FAILURE",
        "SELF_COLLISION_INSTABILITY",
        "OBJECT_MESH_IMPORT_FAILURE",
        "OBJECT_SCALE_OR_ORIGIN_FAILURE",
        "CONVEX_DECOMPOSITION_FAILURE",
        "OBJECT_MASS_INERTIA_FAILURE",
        "CONTACT_RESPONSE_FAILURE",
        "VECTOR_SPAWN_FAILURE",
        "GPU_TENSOR_FAILURE",
        "HEADLESS_RENDER_FAILURE",
        "ASSET_HASH_DRIFT",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WujiJoint:
    name: str
    joint_type: str
    axis: tuple[float, float, float]
    limits: tuple[float, float]
    parent: str
    child: str


@dataclass(frozen=True)
class WujiAssetSpec:
    source_file: Path
    source_sha256: str
    source_commit: str
    root_link: str
    fixed_base: bool
    joint_order: tuple[str, ...]
    tracked_links: tuple[str, ...]
    semantic_mapping: dict[str, str]
    collision_strategy: str
    self_collision: bool
    drive_stiffness: float
    drive_damping: float

    def validate(self, repo_root: Path) -> tuple[WujiJoint, ...]:
        source = (repo_root / self.source_file).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"WUJI_SOURCE_ASSET_NOT_FOUND: {source}")
        if sha256_file(source) != self.source_sha256:
            raise ValueError("ASSET_HASH_DRIFT: Wuji source hash differs from frozen config")
        root = ET.parse(source).getroot()
        links = {element.attrib["name"] for element in root.findall("link")}
        if self.root_link not in links:
            raise ValueError(f"ARTICULATION_ROOT_FAILURE: missing {self.root_link}")
        joints: list[WujiJoint] = []
        for joint in root.findall("joint"):
            if joint.attrib.get("type") not in {"revolute", "continuous"}:
                continue
            axis_element = joint.find("axis")
            limit_element = joint.find("limit")
            if axis_element is None or limit_element is None:
                raise ValueError(f"JOINT_AXIS_OR_LIMIT_FAILURE: {joint.attrib['name']}")
            axis_values = [float(value) for value in axis_element.attrib["xyz"].split()]
            if len(axis_values) != 3:
                raise ValueError(f"JOINT_AXIS_OR_LIMIT_FAILURE: {joint.attrib['name']}")
            joints.append(
                WujiJoint(
                    name=joint.attrib["name"],
                    joint_type=joint.attrib["type"],
                    axis=(axis_values[0], axis_values[1], axis_values[2]),
                    limits=(
                        float(limit_element.attrib["lower"]),
                        float(limit_element.attrib["upper"]),
                    ),
                    parent=joint.find("parent").attrib["link"],  # type: ignore[union-attr]
                    child=joint.find("child").attrib["link"],  # type: ignore[union-attr]
                )
            )
        names = tuple(joint.name for joint in joints)
        if len(joints) != 20 or names != self.joint_order:
            raise ValueError(
                f"JOINT_NAME_MAPPING_FAILURE: expected exact 20-joint order, got {names}"
            )
        if set(self.semantic_mapping) != set(self.joint_order):
            raise ValueError("JOINT_NAME_MAPPING_FAILURE: semantic mapping is not 20/20")
        missing_links = [name for name in self.tracked_links if name not in links]
        if len(self.tracked_links) != 16 or missing_links:
            raise ValueError(f"JOINT_NAME_MAPPING_FAILURE: tracked links missing {missing_links}")
        if self.fixed_base:
            raise ValueError("FLOATING_ROOT_FAILURE: Stage 16-C.1 requires fixed_base=false")
        if (
            self.collision_strategy
            != "deterministic_support_hull_proxies_with_floating_root_overlay_v1"
        ):
            raise ValueError("COLLISION_IMPORT_FAILURE: unsupported Wuji collision strategy")
        if self.drive_stiffness <= 0.0 or self.drive_damping <= 0.0:
            raise ValueError("DRIVE_CONFIGURATION_FAILURE: gains must be finite positive values")
        return tuple(joints)


@dataclass(frozen=True)
class HOCapObjectSpec:
    object_id: str
    source_file: Path
    source_sha256: str
    collision_strategy: str
    scale: tuple[float, float, float]
    translation: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    mass_kg: float
    principal_inertia_kgm2: tuple[float, float, float]
    center_of_mass_m: tuple[float, float, float]
    friction: tuple[float, float, float]
    gravity_enabled: bool
    ground_enabled: bool
    support: str
    physical_classification: str

    def validate(self, repo_root: Path) -> None:
        source = (repo_root / self.source_file).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"OBJECT_MESH_IMPORT_FAILURE: {source}")
        if sha256_file(source) != self.source_sha256:
            raise ValueError(f"ASSET_HASH_DRIFT: {self.object_id}")
        if self.collision_strategy != "convex_hull_v1":
            raise ValueError("CONVEX_DECOMPOSITION_FAILURE: fallback strategy must be uniform")
        if self.scale != (1.0, 1.0, 1.0) or self.rotation_wxyz != (1.0, 0.0, 0.0, 0.0):
            raise ValueError("OBJECT_SCALE_OR_ORIGIN_FAILURE: unexpected transform")
        if self.mass_kg != 0.05 or min(self.principal_inertia_kgm2) <= 0.0:
            raise ValueError("OBJECT_MASS_INERTIA_FAILURE: explicit nominal values required")
        if self.gravity_enabled or self.ground_enabled or self.support != "none":
            raise ValueError(
                "OBJECT_MASS_INERTIA_FAILURE: object must be unsupported in zero gravity"
            )
        if self.physical_classification != "ENGINEERING_NOMINAL_PHYSICAL_PROVENANCE_UNRESOLVED":
            raise ValueError("OBJECT_MASS_INERTIA_FAILURE: physical provenance label missing")


@dataclass(frozen=True)
class AssetMigrationConfig:
    wuji: WujiAssetSpec
    objects: tuple[HOCapObjectSpec, ...]
    output_root: Path
    report_root: Path
    c0_required_statuses: tuple[str, ...]
    allow_c2: bool

    def validate(self, repo_root: Path) -> tuple[WujiJoint, ...]:
        joints = self.wuji.validate(repo_root)
        if len(self.objects) != 2 or {item.object_id for item in self.objects} != {
            "hocap_170105",
            "hocap_170650",
        }:
            raise ValueError("OBJECT_MESH_IMPORT_FAILURE: exact frozen two-object set required")
        for item in self.objects:
            item.validate(repo_root)
        if len({item.collision_strategy for item in self.objects}) != 1:
            raise ValueError("CONVEX_DECOMPOSITION_FAILURE: per-object strategy drift")
        if self.allow_c2:
            raise ValueError("Stage 16-C.2 execution is outside this asset migration")
        for value in (self.output_root, self.report_root):
            if value.is_absolute():
                raise ValueError("generated/report roots must be repository-relative")
        return joints


@dataclass
class BoundedAssetRecovery:
    repairs_per_class: int = 3
    reruns_per_phase: int = 5
    strategy_switches: int = 2
    major_transitions: int = 20

    def __post_init__(self) -> None:
        self._repairs: dict[str, int] = {}
        self._reruns: dict[str, int] = {}
        self._switches = 0
        self._transitions = 0

    def record(self, failure_class: str, *, phase: str, strategy_switch: bool = False) -> None:
        if failure_class not in EXPECTED_C1_FAILURES:
            raise ValueError(f"unknown Stage 16-C.1 failure class: {failure_class}")
        self._repairs[failure_class] = self._repairs.get(failure_class, 0) + 1
        self._reruns[phase] = self._reruns.get(phase, 0) + 1
        self._transitions += 1
        if strategy_switch:
            self._switches += 1
        if self._repairs[failure_class] > self.repairs_per_class:
            raise RuntimeError("repairs_per_class budget exceeded")
        if self._reruns[phase] > self.reruns_per_phase:
            raise RuntimeError("reruns_per_phase budget exceeded")
        if self._switches > self.strategy_switches:
            raise RuntimeError("strategy_switches budget exceeded")
        if self._transitions > self.major_transitions:
            raise RuntimeError("major_transitions budget exceeded")

    def as_dict(self) -> dict[str, Any]:
        return {
            "repairs_by_class": dict(self._repairs),
            "reruns_by_phase": dict(self._reruns),
            "strategy_switches": self._switches,
            "major_transitions": self._transitions,
            "budgets": {
                "repairs_per_class": self.repairs_per_class,
                "reruns_per_phase": self.reruns_per_phase,
                "strategy_switches": self.strategy_switches,
                "major_transitions": self.major_transitions,
            },
        }


def _tuple(values: list[Any], length: int) -> tuple[Any, ...]:
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {len(values)}")
    return tuple(values)


def load_asset_migration_config(path: Path) -> AssetMigrationConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    wuji_raw = raw["wuji"]
    wuji = WujiAssetSpec(
        source_file=Path(wuji_raw["source_file"]),
        source_sha256=wuji_raw["source_sha256"],
        source_commit=wuji_raw["source_commit"],
        root_link=wuji_raw["root_link"],
        fixed_base=bool(wuji_raw["fixed_base"]),
        joint_order=tuple(wuji_raw["joint_order"]),
        tracked_links=tuple(wuji_raw["tracked_links"]),
        semantic_mapping=dict(wuji_raw["semantic_mapping"]),
        collision_strategy=wuji_raw["collision_strategy"],
        self_collision=bool(wuji_raw["self_collision"]),
        drive_stiffness=float(wuji_raw["drive"]["stiffness"]),
        drive_damping=float(wuji_raw["drive"]["damping"]),
    )
    objects = tuple(
        HOCapObjectSpec(
            object_id=item["id"],
            source_file=Path(item["source_file"]),
            source_sha256=item["source_sha256"],
            collision_strategy=item["collision_strategy"],
            scale=_tuple(item["scale"], 3),
            translation=_tuple(item["translation"], 3),
            rotation_wxyz=_tuple(item["rotation_wxyz"], 4),
            mass_kg=float(item["mass_kg"]),
            principal_inertia_kgm2=_tuple(item["principal_inertia_kgm2"], 3),
            center_of_mass_m=_tuple(item["center_of_mass_m"], 3),
            friction=_tuple(item["friction"], 3),
            gravity_enabled=bool(item["gravity_enabled"]),
            ground_enabled=bool(item["ground_enabled"]),
            support=item["support"],
            physical_classification=item["physical_classification"],
        )
        for item in raw["objects"]
    )
    return AssetMigrationConfig(
        wuji=wuji,
        objects=objects,
        output_root=Path(raw["output_root"]),
        report_root=Path(raw["report_root"]),
        c0_required_statuses=tuple(raw["c0_required_statuses"]),
        allow_c2=bool(raw["scope"]["allow_c2"]),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
