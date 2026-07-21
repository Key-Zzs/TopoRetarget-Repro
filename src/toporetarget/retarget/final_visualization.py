"""Stage 9 scene and penetration views.

The renderer is intentionally artifact-only: it never invokes the optimizer or
changes any source artifact.  It follows the existing Matplotlib viewer's
scene-frame conventions and is suitable for headless smoke tests.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.geometry.se3 import transform_points
from toporetarget.keypoints.registry import get_layout
from toporetarget.viz.responsive_fonts import install_responsive_font_scaling

from .artifacts import WarmStartTrajectory
from .final_refinement import (
    FinalRetargetTrajectory,
    dynamic_collision_points_numpy,
)
from .interaction_graph import InteractionGraphTrajectory


def _pyplot(output: str | Path | None, show: bool) -> Any:
    try:
        import matplotlib

        if output is not None and not show:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("final visualization requires the viz extra") from exc
    return plt


def _limits(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.min(values, axis=0)
    high = np.max(values, axis=0)
    extent = max(float(np.max(high - low)), 1e-3)
    center = (low + high) * 0.5
    half = extent * 0.6
    return center - half, center + half


def _view_transform(points: np.ndarray, object_pose: np.ndarray, view: str) -> np.ndarray:
    if view == "scene":
        return points
    if view == "object":
        return transform_points(np.linalg.inv(object_pose), points)
    raise ValueError("view must be scene or object")


def _view_directions(vectors: np.ndarray, object_pose: np.ndarray, view: str) -> np.ndarray:
    if view == "scene":
        return vectors
    if view == "object":
        return vectors @ np.linalg.inv(object_pose)[:3, :3]
    raise ValueError("view must be scene or object")


def render_refinement_frame(
    sequence: Any,
    warm: WarmStartTrajectory,
    graph: InteractionGraphTrajectory,
    final: FinalRetargetTrajectory,
    robot_model: Any,
    surface: Any,
    *,
    frame: int,
    output: str | Path | None = None,
    show_source_hand: bool = True,
    show_warm_start: bool = True,
    show_final: bool = True,
    show_object: bool = True,
    show_interaction_edges: bool = True,
    show_collision_samples: bool = True,
    show_query_set: bool = True,
    show_penetrations: bool = True,
    show_slack: bool = True,
    show_labels: bool = False,
    show_frames: bool = False,
    show_objective: bool = True,
    show_closest: bool = False,
    view: str = "scene",
    show: bool = False,
) -> dict[str, Any]:
    if view not in {"scene", "object"}:
        raise ValueError("view must be scene or object")
    if frame < 0 or frame >= final.frame_count:
        raise ValueError(f"frame {frame} outside final artifact")
    plt = _pyplot(output, show)
    graph_index = int(
        np.flatnonzero(graph.frame_indices == final.arrays["frame_indices"][frame])[0]
    )
    hand_id = str(final.metadata["source_hand_id"])
    source = np.asarray(
        sequence.hand(hand_id)
        .keypoint_tracks["mediapipe21"]
        .positions_scene[final.arrays["frame_indices"][frame]]
    )
    warm_points = np.asarray(
        warm.arrays["robot_keypoints_scene"][final.arrays["frame_indices"][frame]]
    )
    final_points = np.asarray(final.arrays["robot_keypoints_scene"][frame])
    obj = sequence.rigid_object(str(final.metadata["object_id"]))
    object_pose = np.asarray(obj.pose_scene.pose_scene[int(final.arrays["frame_indices"][frame])])
    object_vertices = transform_points(object_pose, np.asarray(obj.mesh.vertices_local))
    collision = dynamic_collision_points_numpy(
        robot_model,
        surface,
        final.arrays["qpos"][frame],
        final.arrays["base_pose_scene"][frame],
    )
    query_start = int(final.arrays["query_offsets"][frame])
    query_stop = int(final.arrays["query_offsets"][frame + 1])
    query_ids = final.arrays["query_ids_concat"][query_start:query_stop]
    phi = np.asarray(final.arrays["full_signed_distance"][frame])
    source = _view_transform(source, object_pose, view)
    warm_points = _view_transform(warm_points, object_pose, view)
    final_points = _view_transform(final_points, object_pose, view)
    object_vertices = _view_transform(object_vertices, object_pose, view)
    collision = _view_transform(collision, object_pose, view)
    values = np.concatenate([source, warm_points, final_points, object_vertices, collision], axis=0)
    low, high = _limits(values)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    layout = get_layout("mediapipe21")
    for parent, child in layout.edges:
        if show_source_hand:
            axis.plot(*source[[parent, child]].T, color="#1f77b4", linewidth=1.3)
        if show_warm_start:
            axis.plot(*warm_points[[parent, child]].T, color="#ff7f0e", linewidth=1.2)
        if show_final:
            axis.plot(*final_points[[parent, child]].T, color="#2ca02c", linewidth=1.4)
    if show_source_hand:
        axis.scatter(*source.T, color="#1f77b4", s=15, label="source")
    if show_warm_start:
        axis.scatter(*warm_points.T, color="#ff7f0e", s=14, label="warm start")
    if show_final:
        axis.scatter(*final_points.T, color="#2ca02c", s=16, label="final")
    if show_labels:
        for index, name in enumerate(layout.semantic_names):
            axis.text(*final_points[index], name, fontsize=6)
    if show_object:
        axis.scatter(
            *object_vertices[:: max(1, len(object_vertices) // 1500)].T,
            color="#777777",
            s=2,
            alpha=0.25,
            label="object mesh",
        )
    if show_collision_samples:
        colors = np.full(len(collision), "#aaaaaa", dtype="U8")
        colors[phi < 0] = "#d62728"
        if show_query_set:
            colors[query_ids] = "#17becf"
        if show_slack and len(query_ids):
            slack_start = int(final.arrays["slack_offsets"][frame])
            slack_stop = int(final.arrays["slack_offsets"][frame + 1])
            slack = final.arrays["slack_concat"][slack_start:slack_stop]
            colors[query_ids[slack > 1e-10]] = "#e377c2"
        axis.scatter(*collision.T, color=colors.tolist(), s=6, alpha=0.8, label="collision samples")
    if show_penetrations and np.any(phi < 0):
        for point in collision[phi < 0]:
            axis.scatter(*point, color="#d62728", s=25)
    if show_closest:
        selected = np.flatnonzero(phi < 0)
        if len(selected):
            closest = np.asarray(final.arrays["full_closest_points"][frame])[selected]
            normals = np.asarray(final.arrays["full_surface_normals"][frame])[selected]
            closest = _view_transform(closest, object_pose, view)
            normals = _view_directions(normals, object_pose, view)
            axis.scatter(*closest.T, color="#8c564b", s=12, label="closest object point")
            axis.quiver(*closest.T, *normals.T, color="#8c564b", length=0.01, alpha=0.7)
    if show_interaction_edges:
        for edge in graph.frames[graph_index].edges:
            if np.all(edge < 21):
                pair = final_points[np.asarray(edge, dtype=np.int64)]
                axis.plot(*pair.T, color="#9467bd", linewidth=0.45, alpha=0.35)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    if show_frames:
        base_pose = np.asarray(final.arrays["base_pose_scene"][frame])
        if view == "object":
            base_pose = np.linalg.inv(object_pose) @ base_pose
        origin = base_pose[:3, 3]
        for index, color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
            endpoint = origin + 0.035 * base_pose[:3, index]
            axis.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=color,
                linewidth=1.2,
                label="final base frame" if index == 0 else None,
            )
    axis.view_init(elev=22, azim=-65)
    axis.legend(loc="upper right", fontsize="small")
    rounds = int(final.arrays["active_set_rounds"][frame])
    slack_start = int(final.arrays["slack_offsets"][frame])
    slack_stop = int(final.arrays["slack_offsets"][frame + 1])
    max_slack = float(np.max(final.arrays["slack_concat"][slack_start:slack_stop], initial=0))
    frame_id = int(final.arrays["frame_indices"][frame])
    success = bool(final.arrays["solver_success"][frame])
    if show_objective:
        title = (
            f"{view} | frame {frame_id} | status {success} | "
            f"rounds {rounds}\nEIM={final.arrays['e_im'][frame]:.4g} | "
            f"min phi={np.min(phi):.4g} m | max slack={max_slack:.4g} m"
        )
        figure.suptitle(title)
    figure.tight_layout()
    destination = None if output is None else Path(output)
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    if show:
        connection, _ = install_responsive_font_scaling(figure)
        try:
            plt.show()
        finally:
            figure.canvas.mpl_disconnect(connection)
    else:
        plt.close(figure)
    return {
        "frame": frame,
        "output": None if destination is None else str(destination),
        "query_count": int(len(query_ids)),
        "min_signed_distance": float(np.min(phi)),
        "solver_success": bool(final.arrays["solver_success"][frame]),
    }


def launch_refinement_viewer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Launch a read-only Stage 9 slider viewer.

    The control pattern follows the existing Stage 8 Matplotlib viewer: one
    figure, a frame slider, navigation buttons, and a timer. No refinement is
    invoked and the saved artifact is never changed.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, Slider

    if len(args) < 6:
        raise TypeError("viewer requires sequence, warm, graph, final, robot_model, and surface")
    sequence, warm, graph, final, robot_model, surface = args[:6]
    start = int(kwargs.pop("start_frame", 0))
    end_value = kwargs.pop("end_frame", None)
    stop = final.frame_count if end_value is None else int(end_value)
    show = bool(kwargs.pop("show", True))
    if start < 0 or stop <= start or stop > final.frame_count:
        raise ValueError("invalid interactive frame range")
    figure = plt.figure(figsize=(11, 8))
    figure.subplots_adjust(bottom=0.28)
    axis = figure.add_subplot(111, projection="3d")
    state = {
        "source": bool(kwargs.pop("show_source_hand", True)),
        "warm": bool(kwargs.pop("show_warm_start", True)),
        "final": bool(kwargs.pop("show_final", True)),
        "object": bool(kwargs.pop("show_object", True)),
        "edges": bool(kwargs.pop("show_interaction_edges", True)),
        "collision": bool(kwargs.pop("show_collision_samples", True)),
        "query": bool(kwargs.pop("show_query_set", True)),
        "penetration": bool(kwargs.pop("show_penetrations", True)),
        "slack": bool(kwargs.pop("show_slack", True)),
        "labels": bool(kwargs.pop("show_labels", False)),
        "frames": bool(kwargs.pop("show_frames", False)),
        "objective": bool(kwargs.pop("show_objective", True)),
        "closest": bool(kwargs.pop("show_closest", False)),
        "playing": False,
    }
    responsive_apply: Any = None
    current = {"frame": start}
    view = str(kwargs.pop("view", "scene"))
    if view not in {"scene", "object"}:
        raise ValueError("view must be scene or object")
    layout = get_layout("mediapipe21")

    def draw(frame: int) -> None:
        current["frame"] = int(frame)
        axis.cla()
        global_frame = int(final.arrays["frame_indices"][frame])
        graph_index = int(np.flatnonzero(graph.frame_indices == global_frame)[0])
        hand_id = str(final.metadata["source_hand_id"])
        source = np.asarray(
            sequence.hand(hand_id).keypoint_tracks["mediapipe21"].positions_scene[global_frame]
        )
        warm_points = np.asarray(warm.arrays["robot_keypoints_scene"][global_frame])
        final_points = np.asarray(final.arrays["robot_keypoints_scene"][frame])
        obj = sequence.rigid_object(str(final.metadata["object_id"]))
        object_pose = np.asarray(obj.pose_scene.pose_scene[global_frame])
        object_vertices = transform_points(object_pose, np.asarray(obj.mesh.vertices_local))
        collision = np.asarray(final.arrays["collision_points_scene"][frame])
        phi = np.asarray(final.arrays["full_signed_distance"][frame])
        query_start = int(final.arrays["query_offsets"][frame])
        query_stop = int(final.arrays["query_offsets"][frame + 1])
        query_ids = np.asarray(
            final.arrays["query_ids_concat"][query_start:query_stop], dtype=np.int64
        )
        source = _view_transform(source, object_pose, view)
        warm_points = _view_transform(warm_points, object_pose, view)
        final_points = _view_transform(final_points, object_pose, view)
        object_vertices = _view_transform(object_vertices, object_pose, view)
        collision = _view_transform(collision, object_pose, view)
        values = np.concatenate(
            [
                source,
                warm_points,
                final_points,
                object_vertices[:: max(1, len(object_vertices) // 1500)],
                collision,
            ],
            axis=0,
        )
        low, high = _limits(values)
        for parent, child in layout.edges:
            if state["source"]:
                axis.plot(*source[[parent, child]].T, color="#1f77b4", linewidth=1.2)
            if state["warm"]:
                axis.plot(*warm_points[[parent, child]].T, color="#ff7f0e", linewidth=1.1)
            if state["final"]:
                axis.plot(*final_points[[parent, child]].T, color="#2ca02c", linewidth=1.3)
        if state["source"]:
            axis.scatter(*source.T, color="#1f77b4", s=15, label="source")
        if state["warm"]:
            axis.scatter(*warm_points.T, color="#ff7f0e", s=14, label="warm start")
        if state["final"]:
            axis.scatter(*final_points.T, color="#2ca02c", s=16, label="final")
        if state["labels"]:
            for index, name in enumerate(layout.semantic_names):
                axis.text(*final_points[index], name, fontsize=6)
        if state["object"]:
            axis.scatter(
                *object_vertices[:: max(1, len(object_vertices) // 1500)].T,
                color="#777777",
                s=2,
                alpha=0.2,
                label="object",
            )
        if state["collision"]:
            colors = np.full(len(collision), "#aaaaaa", dtype="U8")
            colors[phi < 0] = "#d62728"
            if state["query"]:
                colors[query_ids] = "#17becf"
            if state["slack"] and len(query_ids):
                s0 = int(final.arrays["slack_offsets"][frame])
                s1 = int(final.arrays["slack_offsets"][frame + 1])
                slack = final.arrays["slack_concat"][s0:s1]
                colors[query_ids[slack > 1e-10]] = "#e377c2"
            axis.scatter(*collision.T, color=colors.tolist(), s=6, alpha=0.8, label="collision")
        if state["penetration"]:
            for point in collision[phi < 0]:
                axis.scatter(*point, color="#d62728", s=24)
        if state["closest"]:
            closest = np.asarray(final.arrays["full_closest_points"][frame])
            normals = np.asarray(final.arrays["full_surface_normals"][frame])
            closest = _view_transform(closest, object_pose, view)
            normals = _view_directions(normals, object_pose, view)
            selected = np.flatnonzero(phi < 0)
            if len(selected):
                axis.scatter(
                    *closest[selected].T, color="#8c564b", s=12, label="closest object point"
                )
                axis.quiver(
                    *closest[selected].T,
                    *normals[selected].T,
                    color="#8c564b",
                    length=0.01,
                    alpha=0.7,
                )
        if state["edges"]:
            for edge in graph.frames[graph_index].edges:
                if np.all(edge < 21):
                    pair = final_points[np.asarray(edge, dtype=np.int64)]
                    axis.plot(*pair.T, color="#9467bd", linewidth=0.45, alpha=0.35)
        if state["frames"]:
            base_pose = np.asarray(final.arrays["base_pose_scene"][frame])
            if view == "object":
                base_pose = np.linalg.inv(object_pose) @ base_pose
            origin = base_pose[:3, 3]
            for index, color in enumerate(("#d62728", "#2ca02c", "#1f77b4")):
                endpoint = origin + 0.035 * base_pose[:3, index]
                axis.plot(
                    [origin[0], endpoint[0]],
                    [origin[1], endpoint[1]],
                    [origin[2], endpoint[2]],
                    color=color,
                    linewidth=1.2,
                    label="final base frame" if index == 0 else None,
                )
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_zlim(low[2], high[2])
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        title = (
            f"{view} | frame {global_frame} | solver={bool(final.arrays['solver_success'][frame])}"
        )
        if state["objective"]:
            title += (
                f" | total={final.arrays['total_objective'][frame]:.4g}"
                f" | min phi={np.min(phi):.4g} m"
            )
        axis.set_title(title)
        axis.view_init(elev=22, azim=-65)
        axis.legend(loc="upper right", fontsize="small")
        if responsive_apply is not None:
            responsive_apply()
        figure.canvas.draw_idle()

    slider_axis = figure.add_axes((0.18, 0.08, 0.64, 0.035))
    slider = Slider(slider_axis, "frame", start, stop - 1, valinit=start, valstep=1)
    slider.on_changed(lambda value: draw(int(value)))
    button_specs: tuple[tuple[float, str, Callable[[], None]], ...] = (
        (0.02, "first", lambda: draw(start)),
        (0.08, "prev", lambda: draw(max(start, current["frame"] - 1))),
        (0.14, "play", lambda: state.__setitem__("playing", not state["playing"])),
        (0.88, "next", lambda: draw(min(stop - 1, current["frame"] + 1))),
        (0.94, "last", lambda: draw(stop - 1)),
    )

    def button_handler(action: Callable[[], None]) -> Callable[[Any], None]:
        def handler(_event: Any) -> None:
            action()

        return handler

    buttons: list[Any] = []
    for left, label, callback in button_specs:
        button_axis = figure.add_axes((left, 0.13, 0.055, 0.04))
        button = Button(button_axis, label)
        button.on_clicked(button_handler(callback))
        buttons.append(button)
    toggle_names = (
        "source",
        "warm",
        "final",
        "object",
        "edges",
        "collision",
        "query",
        "slack",
        "labels",
        "frames",
        "objective",
        "closest",
    )
    toggle_axis = figure.add_axes((0.66, 0.005, 0.32, 0.20))
    toggle = CheckButtons(toggle_axis, toggle_names, [state[name] for name in toggle_names])

    def toggle_visibility(label: str | None) -> None:
        if label is None:
            return
        state[label] = not state[label]
        draw(current["frame"])

    toggle.on_clicked(toggle_visibility)
    timer = figure.canvas.new_timer(interval=80)

    def tick() -> None:
        if state["playing"]:
            next_frame = current["frame"] + 1
            if next_frame >= stop:
                next_frame = start
            slider.set_val(next_frame)

    timer.add_callback(tick)
    timer.start()
    figure.canvas.mpl_connect("close_event", lambda _event: timer.stop())
    figure._stage9_widgets = (buttons, toggle, slider, timer)  # type: ignore[attr-defined]
    resize_connection, responsive_apply = install_responsive_font_scaling(figure)
    responsive_apply()
    draw(start)
    try:
        if show:
            plt.show()
        else:
            plt.close(figure)
    finally:
        figure.canvas.mpl_disconnect(resize_connection)
    return {
        "interactive": True,
        "frame_range": [start, stop],
        "controls": ["slider", "first", "prev", "next", "last", "timer"],
        "toggles": sorted(key for key in state if key not in {"playing"}),
        "solver_invoked": False,
        "artifact_modified": False,
    }


__all__ = ["launch_refinement_viewer", "render_refinement_frame"]
