"""Small, strict URDF parser used by the robot-hand interface."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.utils.hashing import sha256_file

from .model import GeometrySpec, JointLimit, JointSpec, LinkSpec, UrdfModel


class UrdfParseError(ValueError):
    """Raised when a URDF violates the supported tree/geometry contract."""


def _vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    try:
        result = np.asarray([float(item) for item in value.split()], dtype=np.float64)
    except ValueError as exc:
        raise UrdfParseError(f"invalid numeric vector: {value!r}") from exc
    if result.shape != (3,):
        raise UrdfParseError(f"expected three numbers, got {value!r}")
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def pose_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def _parse_pose(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    return pose_transform(
        _vector(element.attrib.get("xyz"), (0.0, 0.0, 0.0)),
        _vector(element.attrib.get("rpy"), (0.0, 0.0, 0.0)),
    )


def _resolve_mesh(
    filename: str, urdf_path: Path, asset_root: Path | None
) -> tuple[Path | None, str]:
    reference = filename
    cleaned = filename
    if cleaned.startswith("package://"):
        cleaned = cleaned.removeprefix("package://").split("/", 1)[-1]
    candidate = Path(cleaned)
    candidates = [candidate] if candidate.is_absolute() else [urdf_path.parent / candidate]
    if asset_root is not None and not candidate.is_absolute():
        candidates.append(asset_root / candidate)
    resolved = next((path.resolve() for path in candidates if path.is_file()), None)
    return resolved, reference


def _mesh_scale(value: str | None) -> tuple[float, float, float]:
    if value is None:
        return (1.0, 1.0, 1.0)
    items = tuple(float(item) for item in value.split())
    if len(items) == 1:
        items *= 3
    if len(items) != 3 or any(item == 0.0 for item in items):
        raise UrdfParseError(f"mesh scale must contain one or three non-zero values: {value!r}")
    return items


def _parse_geometry(
    element: ET.Element,
    *,
    urdf_path: Path,
    asset_root: Path | None,
) -> GeometrySpec:
    origin = _parse_pose(element.find("origin"))
    geometry = element.find("geometry")
    if geometry is None:
        raise UrdfParseError("visual/collision is missing geometry")
    child = next(iter(geometry), None)
    if child is None or child.tag not in {"mesh", "sphere", "box", "cylinder"}:
        name = None if child is None else child.tag
        raise UrdfParseError(f"unsupported or empty geometry: {name!r}")
    kind = child.tag
    parameters: dict[str, Any] = {}
    source_file: str | None = None
    resolved_path: Path | None = None
    source_hash: str | None = None
    if kind == "mesh":
        filename = child.attrib.get("filename")
        if not filename:
            raise UrdfParseError("mesh geometry is missing filename")
        resolved_path, source_file = _resolve_mesh(filename, urdf_path, asset_root)
        if resolved_path is not None:
            source_hash = sha256_file(resolved_path)
        parameters["scale"] = _mesh_scale(child.attrib.get("scale"))
    elif kind == "sphere":
        if "radius" not in child.attrib:
            raise UrdfParseError("sphere geometry is missing radius")
        parameters["radius"] = float(child.attrib["radius"])
    elif kind == "box":
        parameters["size"] = tuple(
            float(item) for item in _vector(child.attrib.get("size"), (0, 0, 0))
        )
        if any(item <= 0 for item in parameters["size"]):
            raise UrdfParseError("box dimensions must be positive")
    elif kind == "cylinder":
        if "radius" not in child.attrib or "length" not in child.attrib:
            raise UrdfParseError("cylinder geometry needs radius and length")
        parameters["radius"] = float(child.attrib["radius"])
        parameters["length"] = float(child.attrib["length"])
    return GeometrySpec(kind, origin, parameters, source_file, resolved_path, source_hash)


def _parse_limit(element: ET.Element | None, joint_type: str) -> JointLimit:
    if joint_type == "continuous":
        lower, upper = -float("inf"), float("inf")
    elif element is None:
        lower, upper = -float("inf"), float("inf")
    else:
        lower = float(element.attrib.get("lower", "-inf"))
        upper = float(element.attrib.get("upper", "inf"))
    if lower > upper:
        raise UrdfParseError(f"joint limit lower exceeds upper: {lower} > {upper}")
    effort = (
        None
        if element is None or "effort" not in element.attrib
        else float(element.attrib["effort"])
    )
    velocity = (
        None
        if element is None or "velocity" not in element.attrib
        else float(element.attrib["velocity"])
    )
    return JointLimit(lower, upper, effort, velocity)


def parse_urdf(path: str | Path, *, asset_root: str | Path | None = None) -> UrdfModel:
    """Parse a supported URDF without importing any simulator backend."""

    urdf_path = Path(path).expanduser().resolve()
    if not urdf_path.is_file():
        raise UrdfParseError(f"URDF does not exist: {urdf_path}")
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise UrdfParseError(f"invalid URDF XML: {urdf_path}") from exc
    if root.tag != "robot":
        raise UrdfParseError("URDF root element must be <robot>")
    links: dict[str, LinkSpec] = {}
    for element in root.findall("link"):
        name = element.attrib.get("name")
        if not name or name in links:
            raise UrdfParseError(f"missing or duplicate link name: {name!r}")
        visuals = tuple(
            _parse_geometry(
                item,
                urdf_path=urdf_path,
                asset_root=Path(asset_root).resolve() if asset_root else None,
            )
            for item in element.findall("visual")
        )
        collisions = tuple(
            _parse_geometry(
                item,
                urdf_path=urdf_path,
                asset_root=Path(asset_root).resolve() if asset_root else None,
            )
            for item in element.findall("collision")
        )
        links[name] = LinkSpec(name, visuals, collisions)
    if not links:
        raise UrdfParseError("URDF contains no links")
    joints: list[JointSpec] = []
    joint_names: set[str] = set()
    child_links: set[str] = set()
    allowed = {"fixed", "revolute", "continuous", "prismatic"}
    for element in root.findall("joint"):
        name = element.attrib.get("name")
        joint_type = element.attrib.get("type")
        if not name or name in joint_names:
            raise UrdfParseError(f"missing or duplicate joint name: {name!r}")
        if joint_type not in allowed:
            raise UrdfParseError(f"unsupported joint type for {name}: {joint_type!r}")
        if element.find("mimic") is not None:
            raise UrdfParseError(f"mimic joints are not supported: {name}")
        parent_element = element.find("parent")
        child_element = element.find("child")
        parent = None if parent_element is None else parent_element.attrib.get("link")
        child = None if child_element is None else child_element.attrib.get("link")
        if not parent or not child or parent not in links or child not in links:
            raise UrdfParseError(f"joint {name} has a missing parent or child link")
        if child in child_links:
            raise UrdfParseError(f"link has multiple parent joints: {child}")
        child_links.add(child)
        axis_element = element.find("axis")
        axis = _vector(
            None if axis_element is None else axis_element.attrib.get("xyz"), (1.0, 0.0, 0.0)
        )
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise UrdfParseError(f"joint axis is zero: {name}")
        axis = axis / norm
        joints.append(
            JointSpec(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=_parse_pose(element.find("origin")),
                axis=axis,
                limit=_parse_limit(element.find("limit"), joint_type),
            )
        )
        joint_names.add(name)
    roots = sorted(set(links) - child_links)
    if len(roots) != 1:
        raise UrdfParseError(f"URDF must have exactly one root link, found {roots}")
    root_link = roots[0]
    children: dict[str, list[JointSpec]] = {name: [] for name in links}
    for joint in joints:
        children[joint.parent].append(joint)
    link_order: list[str] = []
    visiting: set[str] = set()

    def visit(link: str) -> None:
        if link in visiting:
            raise UrdfParseError(f"URDF joint graph contains a cycle at {link}")
        visiting.add(link)
        link_order.append(link)
        for joint in children[link]:
            visit(joint.child)
        visiting.remove(link)

    visit(root_link)
    if len(link_order) != len(links):
        raise UrdfParseError("URDF contains disconnected links")
    dof = 0
    indexed_joints = []
    for joint in joints:
        indexed_joints.append(
            JointSpec(
                joint.name,
                joint.joint_type,
                joint.parent,
                joint.child,
                joint.origin,
                joint.axis,
                joint.limit,
                dof if joint.actuated else None,
            )
        )
        if joint.actuated:
            dof += 1
    return UrdfModel(
        root.attrib.get("name", "robot"),
        links,
        tuple(indexed_joints),
        root_link,
        tuple(link_order),
        urdf_path,
    )


__all__ = ["UrdfParseError", "parse_urdf", "pose_transform"]
