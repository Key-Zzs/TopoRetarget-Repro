"""Matplotlib comparison renderer; imports Matplotlib only when called."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.se3 import object_to_scene
from toporetarget.viz.errors import ComparisonResult


@dataclass
class ViewerOptions:
    layout: str = "side-by-side"
    frame: int = 0
    raw_color: str = "tab:blue"
    canonical_color: str = "tab:orange"
    show_scene_frame: bool = False
    show_wrist_frame: bool = False
    show_object_frame: bool = False
    show_keypoints: bool = True
    show_mesh: bool = True


def _optional_matplotlib(*, show: bool, output: Path | None) -> Any:
    try:
        import matplotlib

        if output is not None and not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "Visualization support is optional; install with `pip install -e '.[viz]'`."
        ) from exc
    return plt


def _frame_geometries(sequence: HOISequence, frame: int) -> list[np.ndarray]:
    geometries: list[np.ndarray] = []
    for hand in sequence.hands:
        if hand.vertices_scene is not None and frame < hand.vertices_scene.shape[0]:
            geometries.append(hand.vertices_scene[frame])
        for track in hand.keypoint_tracks.values():
            if frame < track.positions_scene.shape[0]:
                geometries.append(track.positions_scene[frame])
    for obj in sequence.rigid_objects:
        if frame < obj.pose_scene.pose_scene.shape[0]:
            local = np.asarray(obj.mesh.vertices_local)
            geometries.append(
                object_to_scene(obj.pose_scene.pose_scene[frame], local[None, ...])[0]
            )
    return geometries


def _limits(raw: HOISequence, canonical: HOISequence, frame: int) -> tuple[np.ndarray, np.ndarray]:
    geometries = _frame_geometries(raw, frame) + _frame_geometries(canonical, frame)
    if not geometries:
        return np.full(3, -0.1), np.full(3, 0.1)
    values = np.concatenate(geometries, axis=0)
    low = values.min(axis=0)
    high = values.max(axis=0)
    extent = max(float(np.max(high - low)), 1e-3)
    pad = max(0.1 * extent, 1e-3)
    center = (low + high) / 2.0
    half = extent / 2.0 + pad
    return center - half, center + half


def _draw_frame(ax: Any, pose: np.ndarray, *, label: str, length: float = 0.03) -> None:
    origin = pose[:3, 3]
    colors = ("r", "g", "b")
    for index, color in enumerate(colors):
        endpoint = origin + pose[:3, index] * length
        ax.plot(
            *zip(origin, endpoint, strict=True),
            color=color,
            linewidth=1.3,
            label=label if index == 0 else None,
        )


def _draw_sequence(
    ax: Any,
    sequence: HOISequence,
    frame: int,
    *,
    color: str,
    object_color: str,
    options: ViewerOptions,
    prefix: str,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for hand in sequence.hands:
        if hand.vertices_scene is not None and frame < hand.vertices_scene.shape[0]:
            vertices = hand.vertices_scene[frame]
            ax.plot(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                color=color,
                linewidth=1.4,
                label=f"{prefix} hand",
            )
            if options.show_mesh and hand.mesh is not None:
                polygons = [vertices[face] for face in hand.mesh.faces]
                ax.add_collection3d(
                    Poly3DCollection(polygons, alpha=0.20, facecolor=color, edgecolor=color)
                )
        if options.show_keypoints:
            for layout, track in hand.keypoint_tracks.items():
                if frame < track.positions_scene.shape[0]:
                    points = track.positions_scene[frame]
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        color=color,
                        s=12,
                        label=f"{prefix} {layout}",
                    )
        if options.show_wrist_frame and frame < hand.wrist_pose_scene.pose_scene.shape[0]:
            _draw_frame(ax, hand.wrist_pose_scene.pose_scene[frame], label=f"{prefix} wrist")

    for obj in sequence.rigid_objects:
        if frame >= obj.pose_scene.pose_scene.shape[0]:
            continue
        pose = obj.pose_scene.pose_scene[frame]
        local = np.asarray(obj.mesh.vertices_local)
        vertices = object_to_scene(pose, local[None, ...])[0]
        if options.show_mesh:
            polygons = [vertices[face] for face in obj.mesh.faces]
            ax.add_collection3d(
                Poly3DCollection(
                    polygons, alpha=0.25, facecolor=object_color, edgecolor=object_color
                )
            )
        ax.scatter(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            color=object_color,
            s=10,
            label=f"{prefix} object",
        )
        if options.show_object_frame:
            _draw_frame(ax, pose, label=f"{prefix} object frame")


def _error_text(result: ComparisonResult, frame: int) -> str:
    payload = result.as_dict()
    lines: list[str] = []
    for name in (
        "hand_vertex_rmse_m",
        "wrist_translation_error_m",
        "wrist_rotation_geodesic_deg",
        "object_pose_translation_error_m",
        "object_pose_rotation_geodesic_deg",
    ):
        values = payload.get("per_frame", {}).get(name)
        if values and frame < len(values):
            lines.append(f"{name}: {values[frame]:.3g}")
    return "\n".join(lines) or "error metrics unavailable"


def render_comparison(
    raw: HOISequence,
    canonical: HOISequence,
    result: ComparisonResult,
    *,
    options: ViewerOptions | None = None,
    output: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Render one frame; side-by-side uses identical limits and camera settings."""

    opts = options or ViewerOptions()
    if opts.layout not in {"side-by-side", "overlay"}:
        raise ValueError("layout must be side-by-side or overlay")
    if opts.frame < 0 or opts.frame >= min(raw.num_frames, canonical.num_frames):
        raise ValueError(f"frame {opts.frame} is outside the common range")
    destination = None if output is None else Path(output)
    plt = _optional_matplotlib(show=show, output=destination)
    figure = plt.figure(figsize=(13, 6) if opts.layout == "side-by-side" else (7, 6))
    if opts.layout == "side-by-side":
        axes = [
            figure.add_subplot(1, 2, 1, projection="3d"),
            figure.add_subplot(1, 2, 2, projection="3d"),
        ]
    else:
        axes = [figure.add_subplot(111, projection="3d")]
    low, high = _limits(raw, canonical, opts.frame)
    if opts.layout == "side-by-side":
        _draw_sequence(
            axes[0],
            raw,
            opts.frame,
            color=opts.raw_color,
            object_color="tab:green",
            options=opts,
            prefix="raw",
        )
        _draw_sequence(
            axes[1],
            canonical,
            opts.frame,
            color=opts.canonical_color,
            object_color="tab:red",
            options=opts,
            prefix="canonical",
        )
        axes[0].set_title("raw")
        axes[1].set_title("canonical")
    else:
        _draw_sequence(
            axes[0],
            raw,
            opts.frame,
            color=opts.raw_color,
            object_color="tab:green",
            options=opts,
            prefix="raw",
        )
        _draw_sequence(
            axes[0],
            canonical,
            opts.frame,
            color=opts.canonical_color,
            object_color="tab:red",
            options=opts,
            prefix="canonical",
        )
        axes[0].set_title("raw / canonical overlay")
    for axis in axes:
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_zlim(low[2], high[2])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.view_init(elev=22, azim=-65)
        try:
            axis.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass
    if opts.show_scene_frame:
        for axis in axes:
            _draw_frame(axis, np.eye(4), label="scene frame", length=max(high - low) * 0.15)
    timestamp = raw.timestamps[opts.frame]
    figure.suptitle(
        f"frame {opts.frame} | timestamp {timestamp:.6f} s\n{_error_text(result, opts.frame)}"
    )
    if opts.layout == "overlay":
        axes[0].legend(loc="upper right", fontsize="small")
    figure.tight_layout()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = destination.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            raise ValueError("single-frame output must be a PNG, JPG, SVG, or PDF")
        figure.savefig(destination, dpi=160)
    if show:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "--show requires a local GUI display; use --output for headless rendering"
            )
        plt.show()
    return figure


def show_comparison(
    raw: HOISequence,
    canonical: HOISequence,
    result: ComparisonResult,
    *,
    options: ViewerOptions | None = None,
) -> Any:
    """Show a comparison with a frame slider when a local GUI is available."""

    opts = options or ViewerOptions()
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError(
            "--show requires a local GUI display; use --output for headless rendering"
        )
    plt = _optional_matplotlib(show=True, output=None)
    figure = render_comparison(raw, canonical, result, options=opts, show=False)
    from matplotlib.widgets import Slider

    slider_axis = figure.add_axes((0.18, 0.02, 0.64, 0.03))
    slider = Slider(
        slider_axis,
        "frame",
        0,
        min(raw.num_frames, canonical.num_frames) - 1,
        valinit=opts.frame,
        valstep=1,
    )

    def update(value: float) -> None:
        figure.clear()
        opts.frame = int(value)
        # Re-rendering in place keeps the slider callback simple and explicit.
        new_figure = render_comparison(raw, canonical, result, options=opts, show=False)
        figure.canvas.draw_idle()
        del new_figure

    slider.on_changed(update)
    plt.show()
    return figure


__all__ = ["ViewerOptions", "render_comparison", "show_comparison"]
