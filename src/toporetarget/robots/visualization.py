"""Static robot-hand geometry, anchor, and frame visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.keypoints.registry import get_layout

from .base import RobotHandModel
from .urdf.geometry import RobotGeometryInstance


def _primitive_mesh(instance: RobotGeometryInstance) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    params = instance.geometry
    if instance.geometry_type == "mesh":
        if instance.resolved_path is None:
            raise FileNotFoundError(f"unresolved mesh reference: {instance.source_file}")
        mesh = trimesh.load_mesh(instance.resolved_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        vertices *= np.asarray(params.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
        return vertices, np.asarray(mesh.faces, dtype=np.int64)
    if instance.geometry_type == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(params["radius"]))
    elif instance.geometry_type == "box":
        mesh = trimesh.creation.box(extents=np.asarray(params["size"], dtype=np.float64))
    elif instance.geometry_type == "cylinder":
        mesh = trimesh.creation.cylinder(
            radius=float(params["radius"]), height=float(params["length"])
        )
    else:
        raise ValueError(f"unsupported geometry type: {instance.geometry_type}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _transform_vertices(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def render_robot_hand(
    model: RobotHandModel,
    qpos: Any,
    *,
    geometry: str = "visual",
    output: str | Path | None = None,
    show: bool = False,
    show_keypoints: bool = True,
    show_skeleton: bool = True,
    show_labels: bool = False,
    show_base_frame: bool = True,
    show_link_frames: bool = False,
    show_joint_axes: bool = False,
    alpha: float = 0.75,
    title_suffix: str = "",
) -> Path | None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if geometry not in {"visual", "collision", "both"}:
        raise ValueError("geometry must be visual, collision, or both")
    q = np.asarray(qpos, dtype=np.float64)
    instances: list[RobotGeometryInstance] = []
    if geometry in {"visual", "both"}:
        instances.extend(model.visual_geometry_instances(q))
    if geometry in {"collision", "both"}:
        instances.extend(model.collision_geometry_instances(q))
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    colors = {"visual": "#5b8ff9", "collision": "#f08a5d"}
    all_points: list[np.ndarray] = []
    for instance in instances:
        vertices, faces = _primitive_mesh(instance)
        transformed = _transform_vertices(vertices, instance.world_transform)
        all_points.append(transformed)
        polygons = transformed[faces]
        axis.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolor=colors[instance.kind],
                edgecolor="none",
                alpha=alpha if instance.kind == "collision" else min(alpha, 0.85),
            )
        )

    points = model.keypoints_base(q).detach().cpu().numpy()
    layout = get_layout(model.spec.semantic_keypoint_layout)
    if show_keypoints:
        axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], color="#222222", s=18, depthshade=False
        )
        if show_labels:
            for index, name in enumerate(layout.semantic_names):
                axis.text(*points[index], f"{index}:{name}", fontsize=7)
    if show_skeleton:
        for parent, child in layout.edges:
            axis.plot(
                points[[parent, child], 0],
                points[[parent, child], 1],
                points[[parent, child], 2],
                color="#222222",
                linewidth=1.2,
            )
    if show_base_frame:
        _draw_frame(axis, np.eye(4), scale=0.03, label="base")
    if show_joint_axes or show_link_frames:
        joint_frames = model.forward_kinematics_reference(q)
        from .urdf.kinematics import joint_origins_numpy

        origins = joint_origins_numpy(model.urdf, model._reorder_to_urdf_numpy(q))
        if show_joint_axes:
            for name, transform in origins.items():
                _draw_frame(axis, transform, scale=0.012, label=name)
        if show_link_frames:
            for name, transform in joint_frames.items():
                _draw_frame(axis, transform, scale=0.009, label=name)
    if all_points:
        bounds = np.concatenate(all_points + [points], axis=0)
    else:
        bounds = points
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.05) * 1.25
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.set_title(
        f"{model.name} | side={model.side} | geometry={geometry} | "
        f"profile={model.anchor_profile.profile_id}{title_suffix}"
    )
    figure.tight_layout()
    result = None
    if output is not None:
        result = Path(output)
        result.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(result, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return result


def _draw_frame(axis: Any, transform: np.ndarray, *, scale: float, label: str) -> None:
    origin = transform[:3, 3]
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    for index, color in enumerate(colors):
        endpoint = origin + scale * transform[:3, index]
        axis.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            linewidth=0.7,
        )
    if label:
        axis.text(*origin, label, fontsize=5)


def render_robot_pair(
    left: RobotHandModel, right: RobotHandModel, *, output: str | Path, show_keypoints: bool = True
) -> Path:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(14, 7))
    for subplot, model in enumerate((left, right), start=1):
        axis = figure.add_subplot(1, 2, subplot, projection="3d")
        instances = model.visual_geometry_instances(model.neutral_q)
        all_points: list[np.ndarray] = []
        for instance in instances:
            vertices, faces = _primitive_mesh(instance)
            transformed = _transform_vertices(vertices, instance.world_transform)
            all_points.append(transformed)
            axis.add_collection3d(
                Poly3DCollection(
                    transformed[faces], facecolor="#5b8ff9", edgecolor="none", alpha=0.82
                )
            )
        points = model.keypoints_base(model.neutral_q).detach().cpu().numpy()
        if show_keypoints:
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], color="#222222", s=12)
        bounds = np.concatenate(all_points + [points], axis=0)
        low, high = bounds.min(axis=0), bounds.max(axis=0)
        center = (low + high) / 2.0
        radius = max(float(np.max(high - low)) / 2.0, 0.05) * 1.25
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_title(f"{model.side} | {model.name}")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
    figure.tight_layout()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


__all__ = ["render_robot_hand", "render_robot_pair"]
