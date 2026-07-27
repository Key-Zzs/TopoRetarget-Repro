"""Backend-free URDF/MJCF kinematic and asset consistency checks."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .urdf.kinematics import joint_origins_numpy


def _vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"MJCF vector must contain three values: {value!r}")
    return values


def _quat_matrix(value: str | None) -> np.ndarray:
    q = np.asarray([float(item) for item in (value or "1 0 0 0").split()], dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"MJCF quaternion must contain four values: {value!r}")
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("MJCF quaternion cannot be zero")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose(element: ET.Element) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _quat_matrix(element.attrib.get("quat"))
    result[:3, 3] = _vector(element.attrib.get("pos"), (0.0, 0.0, 0.0))
    return result


def _motion(axis: np.ndarray, q: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    skew = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    result[:3, :3] = np.eye(3) + math.sin(q) * skew + (1.0 - math.cos(q)) * (skew @ skew)
    return result


@dataclass(frozen=True)
class MjcfJoint:
    name: str
    axis: np.ndarray
    lower: float
    upper: float
    body_name: str


@dataclass(frozen=True)
class MjcfSite:
    name: str
    body_name: str
    local_pose: np.ndarray


@dataclass
class MjcfModel:
    path: Path
    root_body: str
    body_parent: dict[str, str | None] = field(default_factory=dict)
    body_pose: dict[str, np.ndarray] = field(default_factory=dict)
    joints: list[MjcfJoint] = field(default_factory=list)
    sites: dict[str, MjcfSite] = field(default_factory=dict)
    mesh_files: dict[str, Path] = field(default_factory=dict)
    collision_geom_count: int = 0
    excluded_collision_pairs: list[tuple[str, str]] = field(default_factory=list)
    actuators: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> MjcfModel:
        source = Path(path).expanduser().resolve()
        root = ET.parse(source).getroot()
        if root.tag != "mujoco":
            raise ValueError(f"MJCF root element must be mujoco: {source}")
        result = cls(source, "")
        compiler = root.find("compiler")
        meshdir = Path("" if compiler is None else compiler.attrib.get("meshdir", ""))
        asset = root.find("asset")
        if asset is not None:
            for mesh in asset.findall("mesh"):
                name = mesh.attrib.get("name")
                file_name = mesh.attrib.get("file")
                if name and file_name:
                    result.mesh_files[name] = (source.parent / meshdir / file_name).resolve()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF has no worldbody")
        bodies = worldbody.findall("body")
        if len(bodies) != 1:
            raise ValueError(f"MJCF expected one root hand body, found {len(bodies)}")

        def visit(body: ET.Element, parent: str | None) -> None:
            name = body.attrib.get("name")
            if not name:
                raise ValueError("MJCF body is missing name")
            if name in result.body_parent:
                raise ValueError(f"duplicate MJCF body: {name}")
            result.body_parent[name] = parent
            result.body_pose[name] = _pose(body)
            for joint in body.findall("joint"):
                joint_name = joint.attrib.get("name")
                if not joint_name:
                    raise ValueError(f"MJCF joint in {name} is missing name")
                axis = _vector(joint.attrib.get("axis"), (1.0, 0.0, 0.0))
                axis /= np.linalg.norm(axis)
                lower, upper = (
                    float(item) for item in joint.attrib.get("range", "-inf inf").split()
                )
                result.joints.append(MjcfJoint(joint_name, axis, lower, upper, name))
            for site in body.findall("site"):
                site_name = site.attrib.get("name")
                if site_name:
                    result.sites[site_name] = MjcfSite(site_name, name, _pose(site))
            result.collision_geom_count += sum(
                1
                for geom in body.findall("geom")
                if geom.attrib.get("contype", "1") != "0"
                or geom.attrib.get("conaffinity", "1") != "0"
            )
            for child in body.findall("body"):
                visit(child, name)

        visit(bodies[0], None)
        result.root_body = bodies[0].attrib["name"]
        contact = root.find("contact")
        if contact is not None:
            result.excluded_collision_pairs = [
                (item.attrib["body1"], item.attrib["body2"])
                for item in contact.findall("exclude")
                if "body1" in item.attrib and "body2" in item.attrib
            ]
        actuator = root.find("actuator")
        if actuator is not None:
            result.actuators = [
                (item.attrib.get("name", ""), item.attrib["joint"])
                for item in actuator
                if "joint" in item.attrib
            ]
        return result

    def forward_kinematics(self, qpos: np.ndarray, order: tuple[str, ...]) -> dict[str, np.ndarray]:
        q = np.asarray(qpos, dtype=np.float64)
        if q.shape != (len(order),):
            raise ValueError(f"MJCF qpos must have shape [{len(order)}], got {q.shape}")
        values = dict(zip(order, q, strict=True))
        transforms: dict[str, np.ndarray] = {}

        def visit(name: str) -> None:
            parent = self.body_parent[name]
            parent_transform = np.eye(4) if parent is None else transforms[parent]
            transform = parent_transform @ self.body_pose[name]
            joints = [item for item in self.joints if item.body_name == name]
            for joint in joints:
                transform = transform @ _motion(joint.axis, values[joint.name])
            transforms[name] = transform
            for child, child_parent in self.body_parent.items():
                if child_parent == name:
                    visit(child)

        visit(self.root_body)
        return transforms

    def joint_origins(self, qpos: np.ndarray, order: tuple[str, ...]) -> dict[str, np.ndarray]:
        q = np.asarray(qpos, dtype=np.float64)
        values = dict(zip(order, q, strict=True))
        result: dict[str, np.ndarray] = {}

        def visit(name: str, parent_transform: np.ndarray) -> None:
            frame = parent_transform @ self.body_pose[name]
            joints = [item for item in self.joints if item.body_name == name]
            for joint in joints:
                result[joint.name] = frame.copy()
            transform = frame
            for joint in joints:
                transform = transform @ _motion(joint.axis, values[joint.name])
            for child, child_parent in self.body_parent.items():
                if child_parent == name:
                    visit(child, transform)

        visit(self.root_body, np.eye(4))
        return result

    def site_positions(self, qpos: np.ndarray, order: tuple[str, ...]) -> dict[str, np.ndarray]:
        transforms = self.forward_kinematics(qpos, order)
        return {
            name: transforms[site.body_name] @ site.local_pose for name, site in self.sites.items()
        }


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    sine = 0.5 * np.linalg.norm(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arctan2(sine, cosine))


def validate_urdf_mjcf(model: Any, *, seed: int = 4, random_count: int = 10) -> dict[str, Any]:
    if model.spec.optional_mjcf_relative_path is None or model.asset_root is None:
        return {"status": "not_applicable", "reason": "robot has no MJCF asset"}
    mjcf_path = model.asset_root / model.spec.optional_mjcf_relative_path
    mjcf = MjcfModel.load(mjcf_path)
    urdf_order = tuple(model.dof_names)
    mjcf_order = tuple(item.name for item in mjcf.joints)
    actuator_order = tuple(item[1] for item in mjcf.actuators)
    order_match = urdf_order == mjcf_order == actuator_order
    urdf_by_name = model.urdf.joint_by_name
    joint_rows = []
    joint_pass = True
    for joint in mjcf.joints:
        urdf = urdf_by_name.get(joint.name)
        if urdf is None:
            joint_pass = False
            joint_rows.append({"joint": joint.name, "present": False})
            continue
        axis_error = float(np.max(np.abs(urdf.axis - joint.axis)))
        limit_error = max(abs(urdf.limit.lower - joint.lower), abs(urdf.limit.upper - joint.upper))
        passed = axis_error <= 1e-6 and limit_error <= 1e-6
        joint_pass &= passed
        joint_rows.append(
            {
                "joint": joint.name,
                "axis_error": axis_error,
                "limit_error": limit_error,
                "passed": passed,
            }
        )
    lower, upper = model.joint_lower, model.joint_upper
    midpoint = (lower + upper) / 2.0
    rng = np.random.default_rng(seed)
    samples = [model.neutral_q, midpoint]
    samples.extend(
        lower + rng.uniform(0.1, 0.9, size=(random_count, model.num_dofs)) * (upper - lower)
    )
    max_translation = 0.0
    max_rotation = 0.0
    max_anchor = 0.0
    all_finite = True
    anchor_rows = model.anchor_profile.anchors
    for q in samples:
        urdf_fk = model.forward_kinematics_reference(q)
        mjcf_fk = mjcf.forward_kinematics(q, urdf_order)
        for link in model.link_names:
            if link not in mjcf_fk:
                if link not in model.spec.expected_tip_links:
                    all_finite = False
                continue
            max_translation = max(
                max_translation, float(np.max(np.abs(urdf_fk[link][:3, 3] - mjcf_fk[link][:3, 3])))
            )
            max_rotation = max(max_rotation, _rotation_error(urdf_fk[link], mjcf_fk[link]))
        urdf_joint = joint_origins_numpy(model.urdf, q)
        mjcf_joint = mjcf.joint_origins(q, urdf_order)
        sites = mjcf.site_positions(q, urdf_order)
        for anchor in anchor_rows:
            if anchor.anchor_type == "joint_origin":
                expected = urdf_joint[anchor.joint_name]
            elif anchor.anchor_type == "link_origin":
                expected = urdf_fk[anchor.link_name]
            else:
                raise ValueError(
                    "MJCF consistency does not support link_local_point anchor "
                    f"{anchor.semantic_name}"
                )
            site_name = (anchor.provenance or {}).get("site")
            if site_name:
                if site_name not in sites:
                    all_finite = False
                    continue
                actual_point = sites[site_name]
                max_anchor = max(
                    max_anchor, float(np.max(np.abs(expected[:3, 3] - actual_point[:3, 3])))
                )
            if anchor.anchor_type == "joint_origin":
                max_anchor = max(
                    max_anchor, float(np.max(np.abs(expected - mjcf_joint[anchor.joint_name])))
                )
        all_finite &= bool(np.isfinite(list(urdf_fk.values())[0]).all())
    mesh_missing = sorted(name for name, path in mjcf.mesh_files.items() if not path.is_file())
    determinants = [
        float(np.linalg.det(value[:3, :3]))
        for value in mjcf.forward_kinematics(model.neutral_q, urdf_order).values()
    ]
    expected_sites = {
        (anchor.provenance or {}).get("site")
        for anchor in anchor_rows
        if (anchor.provenance or {}).get("site")
    }
    site_match = expected_sites == set(mjcf.sites)
    return {
        "status": "pass"
        if order_match
        and joint_pass
        and site_match
        and not mesh_missing
        and all_finite
        and max_translation <= 1e-6
        and max_rotation <= 2e-6
        and max_anchor <= 1e-6
        else "fail",
        "mjcf_path": str(mjcf_path),
        "mjcf_hash": model.spec.simulation.source_hash,
        "root": {
            "urdf": model.urdf.root_link,
            "mjcf": mjcf.root_body,
            "match": model.urdf.root_link == mjcf.root_body,
        },
        "joint_order_match": order_match,
        "joint_rows": joint_rows,
        "actuator_order": list(actuator_order),
        "tip_sites": sorted(mjcf.sites),
        "tip_site_match": site_match,
        "expected_tip_site_count": 5,
        "collision_geometry_count": mjcf.collision_geom_count,
        "excluded_collision_pairs": [list(pair) for pair in mjcf.excluded_collision_pairs],
        "mesh_missing": mesh_missing,
        "sample_count": len(samples),
        "max_link_translation_error_m": max_translation,
        "max_link_rotation_error_rad": max_rotation,
        "max_anchor_translation_error_m": max_anchor,
        "rotation_determinant_min": min(determinants, default=1.0),
        "rotation_determinant_max": max(determinants, default=1.0),
        "all_finite": all_finite,
        "tolerance": {"translation_m": 1e-6, "rotation_rad": 2e-6, "anchor_m": 1e-6},
    }


__all__ = ["MjcfModel", "validate_urdf_mjcf"]
