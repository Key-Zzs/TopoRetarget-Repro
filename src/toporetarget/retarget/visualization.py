"""Static and interactive warm-start diagnostics built on Matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.se3 import object_to_scene
from toporetarget.keypoints.registry import get_layout

from .artifacts import WarmStartTrajectory


def _plt(output: str | Path | None, show: bool) -> Any:
    try:
        import matplotlib

        if output is not None and not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("visualization requires the viz extra") from exc
    return plt


def _draw_frame(axis: Any, transform: np.ndarray, *, length: float, label: str) -> None:
    origin = transform[:3, 3]
    for index, color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
        endpoint = origin + length * transform[:3, index]
        axis.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            linewidth=1.1,
            label=label if index == 0 else None,
        )


def _local(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return (points - frame[None, :3, 3]) @ frame[:3, :3]


def _set_equal(axis: Any, values: np.ndarray) -> None:
    low, high = values.min(axis=0), values.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-3) * 1.25
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _set_line3d(line: Any, points: np.ndarray) -> None:
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def _frame_points(
    sequence: HOISequence,
    trajectory: WarmStartTrajectory,
    *,
    hand_id: str,
    frame: int,
    view: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(
        sequence.hand(hand_id).keypoint_tracks["mediapipe21"].positions_scene[frame]
    )
    robot = np.asarray(trajectory.arrays["robot_keypoints_scene"][frame])
    source_frame = np.asarray(trajectory.arrays["source_hand_frame_scene"][frame])
    robot_frame = np.asarray(trajectory.arrays["robot_hand_frame_base"][frame])
    if view == "local-hand":
        source = _local(source, source_frame)
        robot = _local(np.asarray(trajectory.arrays["robot_keypoints_base"][frame]), robot_frame)
        source_frame = np.eye(4)
        robot_frame = np.eye(4)
    return source, robot, source_frame, robot_frame


def _set_visibility(artists: list[Any], visible: bool) -> None:
    for artist in artists:
        artist.set_visible(visible)


def _responsive_font_scale(figure: Any) -> float:
    """Return a bounded font scale derived from the current canvas size."""

    width, height = figure.canvas.get_width_height()
    return float(np.clip(min(width / 900.0, height / 800.0), 0.65, 2.5))


def _set_vector_segments(
    artists: list[Any], origins: np.ndarray, vectors: np.ndarray, *, scale: float, normalize: bool
) -> None:
    for artist, origin, vector in zip(artists, origins, vectors, strict=True):
        value = np.asarray(vector, dtype=np.float64)
        if normalize:
            norm = float(np.linalg.norm(value))
            value = value / norm if norm > 1e-15 else np.zeros(3)
        endpoint = origin + scale * value
        _set_line3d(artist, np.stack([origin, endpoint]))


_BONE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_PAIR_PARENTS = (0, 1, 2, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)


def render_warm_start_frame(
    sequence: HOISequence,
    trajectory: WarmStartTrajectory,
    *,
    hand_id: str,
    frame: int,
    view: str = "scene",
    output: str | Path | None = None,
    show_source_hand: bool = True,
    show_robot_hand: bool = True,
    show_source_skeleton: bool = True,
    show_robot_skeleton: bool = True,
    show_object_context: bool = False,
    show_hand_frames: bool = True,
    show_directions: bool = False,
    show_adjacent_features: bool = False,
    show_residuals: bool = False,
    show_labels: bool = False,
    robot_model: Any | None = None,
    show: bool = False,
) -> Path | None:
    if view not in {"scene", "local-hand"}:
        raise ValueError("view must be scene or local-hand")
    if frame < 0 or frame >= trajectory.frame_count:
        raise ValueError(f"frame {frame} outside [0,{trajectory.frame_count})")
    plt = _plt(output, show)
    hand = sequence.hand(hand_id)
    source = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene[frame])
    robot = trajectory.arrays["robot_keypoints_scene"][frame]
    source_frame = trajectory.arrays["source_hand_frame_scene"][frame]
    robot_frame = trajectory.arrays["robot_hand_frame_base"][frame]
    if view == "local-hand":
        source = _local(source, source_frame)
        robot = _local(trajectory.arrays["robot_keypoints_base"][frame], robot_frame)
        source_frame = np.eye(4)
        robot_frame = np.eye(4)
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    layout = get_layout("mediapipe21")
    if show_source_skeleton:
        for parent, child in layout.edges:
            axis.plot(
                source[[parent, child], 0],
                source[[parent, child], 1],
                source[[parent, child], 2],
                color="#1f77b4",
                linewidth=1.5,
            )
    if show_robot_skeleton:
        for parent, child in layout.edges:
            axis.plot(
                robot[[parent, child], 0],
                robot[[parent, child], 1],
                robot[[parent, child], 2],
                color="#ff7f0e",
                linewidth=1.5,
            )
    if show_source_hand:
        axis.scatter(
            source[:, 0],
            source[:, 1],
            source[:, 2],
            color="#1f77b4",
            s=14,
            label="source MediaPipe-21",
        )
    if show_robot_hand:
        axis.scatter(
            robot[:, 0], robot[:, 1], robot[:, 2], color="#ff7f0e", s=14, label="optimized robot"
        )
    if show_labels:
        for index, name in enumerate(layout.semantic_names):
            axis.text(*source[index], name, fontsize=6)
    if show_hand_frames:
        _draw_frame(axis, source_frame, length=0.035, label="source canonical frame")
        _draw_frame(axis, robot_frame, length=0.035, label="robot canonical frame")
    if show_directions or show_adjacent_features or show_residuals:
        source_dirs = trajectory.arrays["source_bone_directions"][frame]
        robot_dirs = trajectory.arrays["robot_bone_directions"][frame]
        parents = [
            item[0]
            for item in (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            )
        ]
        if view == "scene":
            source_dirs = source_dirs @ source_frame[:3, :3].T
            robot_dirs = robot_dirs @ robot_frame[:3, :3].T
        for index, parent in enumerate(parents):
            if show_directions:
                origin = source[parent] if view == "local-hand" else source[parent]
                axis.quiver(
                    *origin,
                    *source_dirs[index],
                    color="#2ca02c",
                    length=0.025,
                    normalize=True,
                    alpha=0.75,
                )
                origin = robot[parent]
                axis.quiver(
                    *origin,
                    *robot_dirs[index],
                    color="#d62728",
                    length=0.025,
                    normalize=True,
                    alpha=0.75,
                )
        if show_residuals:
            residual = trajectory.arrays["pair_residuals"][frame]
            for index, value in enumerate(residual):
                origin = source[index % 20]
                axis.quiver(
                    *origin, *value, color="#9467bd", length=0.04, normalize=False, alpha=0.7
                )
    if show_object_context and view == "scene":
        for obj in sequence.rigid_objects:
            pose = obj.pose_scene.pose_scene[frame]
            vertices = object_to_scene(pose, obj.mesh.vertices_local[None, ...])[0]
            axis.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                color="#888888",
                s=2,
                alpha=0.25,
                label=f"object context: {obj.object_id}",
            )
    values = np.concatenate([source, robot], axis=0)
    _set_equal(axis, values)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    metadata = trajectory.metadata
    axis.set_title(
        f"{view} | frame {frame} | t={trajectory.arrays['timestamps'][frame]:.6f}s\n"
        f"{metadata.get('robot_name')} | Ebone={trajectory.arrays['ebone'][frame]:.5g} | "
        f"total={trajectory.arrays['total_objective'][frame]:.5g}"
    )
    axis.legend(loc="upper right", fontsize="small")
    figure.tight_layout()
    result = None if output is None else Path(output)
    if result is not None:
        result.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(result, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return result


def render_warm_start_plots(
    trajectory: WarmStartTrajectory, *, output_dir: str | Path, prefix: str
) -> list[Path]:
    plt = _plt(output_dir, False)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    t = trajectory.arrays["timestamps"]
    outputs: list[Path] = []
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(t, trajectory.arrays["initial_ebone"], label="initial Ebone")
    axis.plot(t, trajectory.arrays["ebone"], label="final Ebone")
    axis.plot(t, trajectory.arrays["temporal_term"], label="temporal term")
    axis.plot(t, trajectory.arrays["total_objective"], label="total objective")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("objective")
    axis.legend()
    figure.tight_layout()
    target = root / f"{prefix}_loss_curves.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    outputs.append(target)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(t, trajectory.arrays["qpos"])
    axis.set_title("raw qpos (radians)")
    axis.set_xlabel("time (s)")
    figure.tight_layout()
    target = root / f"{prefix}_qpos_curves.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    outputs.append(target)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(t, trajectory.arrays["joint_limit_margins"])
    axis.set_title("joint-limit margins (radians)")
    axis.set_xlabel("time (s)")
    figure.tight_layout()
    target = root / f"{prefix}_joint_limit_margins.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    outputs.append(target)
    return outputs


def launch_warm_start_viewer(
    trajectory: WarmStartTrajectory,
    sequence: HOISequence,
    *,
    hand_id: str,
    view: str = "scene",
    start_frame: int = 0,
    end_frame: int | None = None,
    show_source_hand: bool = True,
    show_robot_hand: bool = True,
    show_source_skeleton: bool = True,
    show_robot_skeleton: bool = True,
    show_object_context: bool = False,
    show_hand_frames: bool = True,
    show_directions: bool = False,
    show_adjacent_features: bool = False,
    show_labels: bool = False,
    show_residuals: bool = False,
) -> dict[str, Any]:
    """Interactive viewer sharing the static scene/local-hand diagnostic layers."""
    if view not in {"scene", "local-hand"}:
        raise ValueError("view must be scene or local-hand")
    if start_frame < 0 or start_frame >= trajectory.frame_count:
        raise ValueError(f"start_frame {start_frame} outside [0,{trajectory.frame_count})")
    plt = _plt(None, True)
    from matplotlib.widgets import Slider

    end = trajectory.frame_count if end_frame is None else min(end_frame, trajectory.frame_count)
    start = start_frame
    if end <= start:
        raise ValueError(f"invalid frame range [{start},{end})")
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    layout = get_layout("mediapipe21")
    source, robot, source_frame, robot_frame = _frame_points(
        sequence, trajectory, hand_id=hand_id, frame=start, view=view
    )
    stable_artists: list[Any] = []

    source_skeleton = [
        axis.plot([], [], [], color="#1f77b4", linewidth=1.5)[0] for _ in layout.edges
    ]
    robot_skeleton = [
        axis.plot([], [], [], color="#ff7f0e", linewidth=1.5)[0] for _ in layout.edges
    ]
    stable_artists.extend(source_skeleton + robot_skeleton)
    source_scatter = axis.scatter(
        source[:, 0], source[:, 1], source[:, 2], color="#1f77b4", s=14, label="source MediaPipe-21"
    )
    robot_scatter = axis.scatter(
        robot[:, 0], robot[:, 1], robot[:, 2], color="#ff7f0e", s=14, label="optimized robot"
    )
    stable_artists.extend([source_scatter, robot_scatter])

    labels = [
        axis.text(*source[index], name, fontsize=6)
        for index, name in enumerate(layout.semantic_names)
    ]
    stable_artists.extend(labels)

    source_frame_lines = [
        axis.plot([], [], [], color=color, linewidth=1.1)[0]
        for color in ("#d62728", "#2ca02c", "#1f77b4")
    ]
    robot_frame_lines = [
        axis.plot([], [], [], color=color, linewidth=1.1)[0]
        for color in ("#d62728", "#2ca02c", "#1f77b4")
    ]
    stable_artists.extend(source_frame_lines + robot_frame_lines)

    source_direction_lines = [
        axis.plot([], [], [], color="#2ca02c", linewidth=1.0, alpha=0.75)[0] for _ in _BONE_EDGES
    ]
    robot_direction_lines = [
        axis.plot([], [], [], color="#d62728", linewidth=1.0, alpha=0.75)[0] for _ in _BONE_EDGES
    ]
    stable_artists.extend(source_direction_lines + robot_direction_lines)

    source_feature_lines = [
        axis.plot([], [], [], color="#17becf", linewidth=1.0, alpha=0.75)[0] for _ in _PAIR_PARENTS
    ]
    robot_feature_lines = [
        axis.plot([], [], [], color="#bcbd22", linewidth=1.0, alpha=0.75)[0] for _ in _PAIR_PARENTS
    ]
    stable_artists.extend(source_feature_lines + robot_feature_lines)

    residual_lines = [
        axis.plot([], [], [], color="#9467bd", linewidth=1.0, alpha=0.75)[0] for _ in _PAIR_PARENTS
    ]
    stable_artists.extend(residual_lines)

    object_scatters: list[Any] = []
    if view == "scene":
        for obj in sequence.rigid_objects:
            pose = obj.pose_scene.pose_scene[start]
            vertices = object_to_scene(pose, obj.mesh.vertices_local[None, ...])[0]
            object_scatters.append(
                axis.scatter(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    color="#888888",
                    s=2,
                    alpha=0.25,
                    label=f"object context: {obj.object_id}",
                )
            )
    stable_artists.extend(object_scatters)

    slider_axis = figure.add_axes([0.18, 0.04, 0.62, 0.03])
    slider = Slider(slider_axis, "frame", start, max(start, end - 1), valinit=start, valstep=1)

    def set_scatter(scatter: Any, points: np.ndarray) -> None:
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    def set_text(text: Any, point: np.ndarray) -> None:
        text.set_position((point[0], point[1]))
        text.set_3d_properties(point[2], zdir="z")

    def set_frame_lines(lines: list[Any], transform: np.ndarray, length: float = 0.035) -> None:
        origin = transform[:3, 3]
        for index, line in enumerate(lines):
            _set_line3d(line, np.stack([origin, origin + length * transform[:3, index]]))

    def set_object_context(frame: int) -> None:
        if view != "scene":
            return
        for scatter, obj in zip(object_scatters, sequence.rigid_objects, strict=True):
            pose = obj.pose_scene.pose_scene[frame]
            vertices = object_to_scene(pose, obj.mesh.vertices_local[None, ...])[0]
            set_scatter(scatter, vertices)

    def update(frame_value: float) -> None:
        frame = int(frame_value)
        source_points, robot_points, source_transform, robot_transform = _frame_points(
            sequence, trajectory, hand_id=hand_id, frame=frame, view=view
        )
        for line, (parent, child) in zip(source_skeleton, layout.edges, strict=True):
            _set_line3d(line, source_points[[parent, child]])
        for line, (parent, child) in zip(robot_skeleton, layout.edges, strict=True):
            _set_line3d(line, robot_points[[parent, child]])
        set_scatter(source_scatter, source_points)
        set_scatter(robot_scatter, robot_points)
        for text, point in zip(labels, source_points, strict=True):
            set_text(text, point)

        set_frame_lines(source_frame_lines, source_transform)
        set_frame_lines(robot_frame_lines, robot_transform)

        source_dirs = np.asarray(trajectory.arrays["source_bone_directions"][frame])
        robot_dirs = np.asarray(trajectory.arrays["robot_bone_directions"][frame])
        source_features = np.asarray(trajectory.arrays["source_adjacent_features"][frame])
        robot_features = np.asarray(trajectory.arrays["robot_adjacent_features"][frame])
        residual = np.asarray(trajectory.arrays["pair_residuals"][frame])
        if view == "scene":
            source_dirs = source_dirs @ source_transform[:3, :3].T
            robot_dirs = robot_dirs @ robot_transform[:3, :3].T
            source_features = source_features @ source_transform[:3, :3].T
            robot_features = robot_features @ robot_transform[:3, :3].T
            residual = residual @ source_transform[:3, :3].T
        source_bone_origins = source_points[np.asarray([edge[0] for edge in _BONE_EDGES])]
        robot_bone_origins = robot_points[np.asarray([edge[0] for edge in _BONE_EDGES])]
        _set_vector_segments(
            source_direction_lines, source_bone_origins, source_dirs, scale=0.025, normalize=True
        )
        _set_vector_segments(
            robot_direction_lines, robot_bone_origins, robot_dirs, scale=0.025, normalize=True
        )
        pair_origins_source = source_points[np.asarray(_PAIR_PARENTS)]
        pair_origins_robot = robot_points[np.asarray(_PAIR_PARENTS)]
        _set_vector_segments(
            source_feature_lines, pair_origins_source, source_features, scale=0.025, normalize=False
        )
        _set_vector_segments(
            robot_feature_lines, pair_origins_robot, robot_features, scale=0.025, normalize=False
        )
        _set_vector_segments(
            residual_lines, pair_origins_source, residual, scale=0.04, normalize=False
        )

        set_object_context(frame)
        values = [source_points, robot_points]
        if show_object_context and view == "scene":
            values.extend(
                [
                    object_to_scene(
                        obj.pose_scene.pose_scene[frame], obj.mesh.vertices_local[None, ...]
                    )[0]
                    for obj in sequence.rigid_objects
                ]
            )
        _set_equal(axis, np.concatenate(values, axis=0))
        _set_visibility(source_skeleton, show_source_skeleton)
        _set_visibility(robot_skeleton, show_robot_skeleton)
        source_scatter.set_visible(show_source_hand)
        robot_scatter.set_visible(show_robot_hand)
        _set_visibility(labels, show_labels)
        _set_visibility(source_frame_lines + robot_frame_lines, show_hand_frames)
        _set_visibility(source_direction_lines + robot_direction_lines, show_directions)
        _set_visibility(source_feature_lines + robot_feature_lines, show_adjacent_features)
        _set_visibility(residual_lines, show_residuals)
        _set_visibility(object_scatters, show_object_context and view == "scene")
        timestamp = trajectory.arrays["timestamps"][frame]
        axis.set_title(
            f"{view} | frame {frame} | t={timestamp:.6f}s\n"
            f"{trajectory.metadata.get('robot_name')} | "
            f"Ebone={trajectory.arrays['ebone'][frame]:.5g} | "
            f"total={trajectory.arrays['total_objective'][frame]:.5g}"
        )
        figure.canvas.draw_idle()

    slider.on_changed(update)
    update(start)
    legend = axis.legend()

    def apply_font_scale() -> None:
        scale = _responsive_font_scale(figure)
        axis.title.set_fontsize(11.0 * scale)
        axis.xaxis.label.set_fontsize(9.0 * scale)
        axis.yaxis.label.set_fontsize(9.0 * scale)
        axis.zaxis.label.set_fontsize(9.0 * scale)
        for tick in axis.get_xticklabels() + axis.get_yticklabels() + axis.get_zticklabels():
            tick.set_fontsize(7.0 * scale)
        for text in labels:
            text.set_fontsize(6.0 * scale)
        for text in legend.get_texts():
            text.set_fontsize(8.0 * scale)
        slider.label.set_fontsize(8.0 * scale)
        slider.valtext.set_fontsize(8.0 * scale)
        for tick in slider.ax.get_xticklabels() + slider.ax.get_yticklabels():
            tick.set_fontsize(7.0 * scale)

    def on_resize(_event: Any) -> None:
        apply_font_scale()
        figure.canvas.draw_idle()

    resize_connection = figure.canvas.mpl_connect("resize_event", on_resize)
    apply_font_scale()
    try:
        plt.show()
    finally:
        figure.canvas.mpl_disconnect(resize_connection)
    return {
        "artist_count": len(stable_artists),
        "stable_artist_count": len(stable_artists),
        "slider": True,
        "frame_range": [start, end],
        "layout_edges": len(layout.edges),
        "view": view,
        "responsive_fonts": True,
        "layers": {
            "source_skeleton": show_source_skeleton,
            "robot_skeleton": show_robot_skeleton,
            "hand_frames": show_hand_frames,
            "directions": show_directions,
            "adjacent_features": show_adjacent_features,
            "residuals": show_residuals,
            "labels": show_labels,
            "object_context": show_object_context and view == "scene",
        },
    }


__all__ = ["launch_warm_start_viewer", "render_warm_start_frame", "render_warm_start_plots"]
