"""Headless-friendly Stage 6 geometry visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.viz.responsive_fonts import install_responsive_font_scaling

from .object_geometry import scene_samples_for_frame
from .robot_surface import RobotSurfaceSampleSet
from .signed_distance.reference import ReferenceSignedDistanceBackend
from .surface_sampling import SurfaceSampleSet


def _plt(output: str | Path | None = None) -> Any:
    try:
        import matplotlib

        if output is not None:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "geometry visualization needs `pip install -e '.[viz,geometry]'`"
        ) from exc
    return plt


def _set_equal(axis: Any, points: np.ndarray) -> None:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = (low + high) / 2.0
    half = max(float(np.max(high - low)) / 2.0, 1e-3) * 1.15
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)


def render_object_samples(
    track: Any,
    samples: SurfaceSampleSet,
    *,
    frame: int = 0,
    output: str | Path | None = None,
    show_normals: bool = False,
    show_ids: bool = False,
    show_object_frame: bool = False,
    show_scene_frame: bool = False,
) -> Path | None:
    plt = _plt(output)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    points, normals = scene_samples_for_frame(track, samples, frame)
    vertices = track.mesh.vertices_local
    pose = track.pose_scene.pose_scene[frame]
    mesh_scene = vertices @ pose[:3, :3].T + pose[:3, 3]
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(
        Poly3DCollection(
            mesh_scene[track.mesh.faces],
            alpha=0.18,
            facecolor="tab:blue",
            edgecolor="gray",
            linewidth=0.25,
        )
    )
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color="tab:red",
        s=28,
        label=f"object samples ({samples.count})",
    )
    if show_normals:
        axis.quiver(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            normals[:, 0],
            normals[:, 1],
            normals[:, 2],
            length=0.01,
            color="tab:orange",
            linewidth=0.7,
        )
    if show_ids:
        for index, point in enumerate(points):
            axis.text(*point, str(index), fontsize=6)
    if show_object_frame or show_scene_frame:
        origin = pose[:3, 3] if show_object_frame else np.zeros(3)
        transform = pose if show_object_frame else np.eye(4)
        for axis_index, color in enumerate(("r", "g", "b")):
            endpoint = origin + transform[:3, axis_index] * 0.05
            axis.plot(*zip(origin, endpoint, strict=True), color=color, linewidth=1.5)
    _set_equal(axis, np.concatenate((mesh_scene, points), axis=0))
    axis.set_title(
        f"Object samples | frame={frame} | profile={samples.profile_id} | positive outside"
    )
    axis.legend(loc="upper right")
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(figure)
        return destination
    connection, _ = install_responsive_font_scaling(figure)
    try:
        plt.show()
    finally:
        figure.canvas.mpl_disconnect(connection)
    return None


def render_sdf_slice(
    backend: ReferenceSignedDistanceBackend,
    *,
    axis_name: str = "z",
    slice_value: float = 0.0,
    extent: float = 2.5,
    resolution: int = 160,
    output: str | Path | None = None,
    signed: bool = True,
) -> Path | None:
    plt = _plt(output)
    axis_index = {"x": 0, "y": 1, "z": 2}.get(axis_name)
    if axis_index is None:
        raise ValueError("slice axis must be x, y, or z")
    values = np.linspace(-extent, extent, resolution)
    first, second = np.meshgrid(values, values, indexing="xy")
    points = np.zeros((resolution * resolution, 3), dtype=np.float64)
    other = [index for index in range(3) if index != axis_index]
    points[:, other[0]] = first.reshape(-1)
    points[:, other[1]] = second.reshape(-1)
    points[:, axis_index] = slice_value
    result = backend.query_local(points)
    field = (result.signed_distance if signed else result.unsigned_distance).reshape(
        resolution, resolution
    )
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        field,
        extent=(-extent, extent, -extent, extent),
        origin="lower",
        cmap="coolwarm",
        aspect="equal",
    )
    if signed and np.all(np.isfinite(field)):
        axis.contour(first, second, field, levels=[0.0], colors="black", linewidths=1.0)
    figure.colorbar(
        image,
        ax=axis,
        label="signed distance (positive outside)" if signed else "unsigned distance",
    )
    axis.set_xlabel("xyz".replace(axis_name, "")[0] if False else f"axis {other[0]}")
    axis.set_ylabel(f"axis {other[1]}")
    axis.set_title(f"{backend.sign_mode} SDF slice: {axis_name}={slice_value:g}")
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(figure)
        return destination
    connection, _ = install_responsive_font_scaling(figure)
    try:
        plt.show()
    finally:
        figure.canvas.mpl_disconnect(connection)
    return None


def render_robot_surface(
    samples: RobotSurfaceSampleSet,
    *,
    output: str | Path | None = None,
    show_normals: bool = False,
    show_collision_mesh: bool = False,
) -> Path | None:
    plt = _plt(output)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    colors = {
        link: f"C{index % 10}"
        for index, link in enumerate(sorted(set(samples.link_names.tolist())))
    }
    for link in sorted(set(samples.link_names.tolist())):
        mask = samples.link_names == link
        axis.scatter(
            samples.points_scene[mask, 0],
            samples.points_scene[mask, 1],
            samples.points_scene[mask, 2],
            s=16,
            color=colors[link],
            label=link,
        )
        if show_normals:
            axis.quiver(
                samples.points_scene[mask, 0],
                samples.points_scene[mask, 1],
                samples.points_scene[mask, 2],
                samples.normals_scene[mask, 0],
                samples.normals_scene[mask, 1],
                samples.normals_scene[mask, 2],
                length=0.006,
                color=colors[link],
                linewidth=0.5,
            )
    _set_equal(axis, samples.points_scene)
    axis.set_title(
        f"{samples.robot_name} collision surface samples | {samples.count} | visual fallback=false"
    )
    axis.legend(loc="upper left", fontsize=6, ncol=2)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=140, bbox_inches="tight")
        plt.close(figure)
        return destination
    connection, _ = install_responsive_font_scaling(figure)
    try:
        plt.show()
    finally:
        figure.canvas.mpl_disconnect(connection)
    return None


__all__ = ["render_object_samples", "render_robot_surface", "render_sdf_slice"]
