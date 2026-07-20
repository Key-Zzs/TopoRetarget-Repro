"""Matplotlib source/shared-graph and Laplacian diagnostics for Stage 8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .interaction_evaluation import InteractionEvaluationTrajectory
from .interaction_graph import InteractionGraphTrajectory

EDGE_COLORS = {
    "hand-hand": "#286090",
    "hand-object": "#d95f02",
    "object-object": "#5aae61",
}


def _category(edge: np.ndarray) -> str:
    i, j = (int(x) for x in edge)
    if i < 21 and j < 21:
        return "hand-hand"
    if i < 21 <= j:
        return "hand-object"
    return "object-object"


def _setup_axis(ax: Any, points: np.ndarray, *, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    bounds = np.ptp(points, axis=0)
    center = np.mean(points, axis=0)
    radius = max(float(np.max(bounds)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _draw_edges(ax: Any, points: np.ndarray, edges: np.ndarray, *, show: dict[str, bool]) -> None:
    for edge_id, edge in enumerate(edges):
        category = _category(edge)
        if not show.get(category, True):
            continue
        segment = points[np.asarray(edge)]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=EDGE_COLORS[category],
            linewidth=0.55 if category == "object-object" else 1.0,
            alpha=0.60,
            label=category
            if edge_id
            == next((i for i, item in enumerate(edges) if _category(item) == category), edge_id)
            else None,
        )


def _draw_points(
    ax: Any, points: np.ndarray, *, title: str, contributions: np.ndarray | None = None
) -> None:
    colors: Any = "#333333"
    if contributions is None:
        colors = np.asarray(["#1f77b4"] * 21 + ["#e377c2"] * (len(points) - 21))
    else:
        colors = contributions
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=14, depthshade=False)
    _setup_axis(ax, points, title=title)


def render_interaction_frame(
    graph: InteractionGraphTrajectory,
    *,
    evaluation: InteractionEvaluationTrajectory | None = None,
    frame: int = 0,
    mode: str = "source",
    layout: str = "single",
    output: str | Path | None = None,
    show_hand_hand_edges: bool = True,
    show_hand_object_edges: bool = True,
    show_object_object_edges: bool = True,
    show_laplacian: bool = False,
    show_residuals: bool = False,
    show_contributions: bool = False,
    show_labels: bool = False,
) -> dict[str, Any]:
    """Render a frozen graph/evaluation; no topology is recomputed."""

    if frame < 0 or frame >= graph.frame_count:
        raise ValueError(f"frame {frame} is outside graph range [0,{graph.frame_count})")
    if mode not in {"source", "compare", "laplacian"}:
        raise ValueError(f"unsupported interaction visualization mode: {mode}")
    import matplotlib.pyplot as plt

    source = graph.source_vertices[frame]
    edges = graph.edge_frames[frame]
    source_laplacian = graph.source_laplacian[frame]
    stats = graph.frame_statistics[frame]
    graph_hash = graph.graph_hashes[frame]
    edge_show = {
        "hand-hand": show_hand_hand_edges,
        "hand-object": show_hand_object_edges,
        "object-object": show_object_object_edges,
    }
    if mode == "compare" or layout == "side-by-side":
        if evaluation is None:
            raise ValueError("compare visualization requires an evaluation artifact")
        figure = plt.figure(figsize=(13, 6))
        axes = [figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d")]
        robot = evaluation.robot_vertices[frame]
        combined = np.concatenate([source, robot], axis=0)
        _draw_points(axes[0], source, title="source graph")
        _draw_points(axes[1], robot, title="robot graph — shared connectivity")
        for axis, points in zip(axes, (source, robot), strict=True):
            _draw_edges(axis, points, edges, show=edge_show)
            if show_labels:
                for index, point in enumerate(points):
                    axis.text(*point, str(index), fontsize=5)
        if show_laplacian:
            src_delta = source_laplacian
            rob_delta = evaluation.robot_laplacian[frame]
            for axis, points, delta, color in zip(
                axes, (source, robot), (src_delta, rob_delta), ("#111111", "#9467bd"), strict=True
            ):
                axis.quiver(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    delta[:, 0],
                    delta[:, 1],
                    delta[:, 2],
                    color=color,
                    length=0.03,
                    normalize=False,
                    alpha=0.65,
                )
        if show_residuals:
            residual = evaluation.residual[frame]
            axes[1].quiver(
                robot[:, 0],
                robot[:, 1],
                robot[:, 2],
                residual[:, 0],
                residual[:, 1],
                residual[:, 2],
                color="#e41a1c",
                length=0.03,
                normalize=False,
                alpha=0.7,
            )
        figure.suptitle(f"shared connectivity | frame {frame} | E_IM={evaluation.e_im[frame]:.6e}")
        _setup_axis(axes[0], combined, title="source graph")
        _setup_axis(axes[1], combined, title="robot graph — shared connectivity")
    else:
        figure = plt.figure(figsize=(8, 7))
        axis = figure.add_subplot(111, projection="3d")
        points = source
        contribution = None
        if show_contributions and evaluation is not None:
            values = evaluation.per_vertex_contribution[frame]
            contribution = np.asarray(values, dtype=np.float64)
        _draw_points(
            axis,
            points,
            title=f"source interaction graph | frame {frame}",
            contributions=contribution,
        )
        _draw_edges(axis, points, edges, show=edge_show)
        if show_labels:
            for index, point in enumerate(points):
                axis.text(*point, str(index), fontsize=5)
        if show_laplacian:
            delta = source_laplacian
            axis.quiver(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                delta[:, 0],
                delta[:, 1],
                delta[:, 2],
                color="#111111",
                length=0.03,
                normalize=False,
                alpha=0.65,
            )
        if show_residuals:
            if evaluation is None:
                raise ValueError("residual visualization requires an evaluation artifact")
            delta = evaluation.residual[frame]
            axis.quiver(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                delta[:, 0],
                delta[:, 1],
                delta[:, 2],
                color="#e41a1c",
                length=0.03,
                normalize=False,
                alpha=0.7,
            )
        axis.text2D(
            0.02,
            0.02,
            f"simplices={stats['simplex_count']} edges={stats['edge_count']} "
            f"hand-object={stats['hand_object_edge_count']} hash={graph_hash[:12]}",
            transform=axis.transAxes,
        )
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return {
        "frame": frame,
        "mode": mode,
        "layout": layout,
        "output": None if output is None else str(output),
        "graph_hash": graph_hash,
        "shared_connectivity": mode == "compare",
        "artist_stability": True,
    }


def launch_interaction_viewer(
    graph: InteractionGraphTrajectory,
    *,
    evaluation: InteractionEvaluationTrajectory | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    mode: str = "compare",
    show_laplacian: bool = False,
    show_residuals: bool = False,
    show_contributions: bool = False,
) -> dict[str, Any]:
    """Launch a bounded slider viewer using the already-built graph arrays."""

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    stop = graph.frame_count if end_frame is None else min(end_frame, graph.frame_count)
    if start_frame < 0 or stop <= start_frame:
        raise ValueError("invalid interactive frame range")
    if mode == "compare" and evaluation is None:
        raise ValueError("compare viewer requires evaluation")
    figure = plt.figure(figsize=(13, 7) if mode == "compare" else (8, 7))
    axes = (
        [figure.add_subplot(121, projection="3d"), figure.add_subplot(122, projection="3d")]
        if mode == "compare"
        else [figure.add_subplot(111, projection="3d")]
    )
    axis = axes[0]
    frame = start_frame
    state = {
        "source": True,
        "robot": True,
        "hand-hand": True,
        "hand-object": True,
        "object-object": True,
        "laplacian": show_laplacian,
        "residuals": show_residuals,
        "contributions": show_contributions,
        "playing": False,
    }

    def draw(current_frame: int) -> None:
        for item in axes:
            item.cla()
        graph_vertices = graph.source_vertices[current_frame]
        graph_edges = graph.edge_frames[current_frame]
        edge_show = {key: state[key] for key in EDGE_COLORS}
        if mode == "compare":
            assert evaluation is not None
            source = graph_vertices
            robot = evaluation.robot_vertices[current_frame]
            combined = np.concatenate([source, robot], axis=0)
            for item, points, title in zip(
                axes,
                (source, robot),
                ("source graph", "robot graph — shared connectivity"),
                strict=True,
            ):
                visible = state["source"] if item is axes[0] else state["robot"]
                if visible:
                    contribution = (
                        evaluation.per_vertex_contribution[current_frame]
                        if state["contributions"] and item is axes[1]
                        else None
                    )
                    _draw_points(item, points, title=title, contributions=contribution)
                    _draw_edges(item, points, graph_edges, show=edge_show)
                _setup_axis(item, combined, title=title)
                if state["laplacian"] and visible:
                    delta = (
                        graph.source_laplacian[current_frame]
                        if item is axes[0]
                        else evaluation.robot_laplacian[current_frame]
                    )
                    item.quiver(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        delta[:, 0],
                        delta[:, 1],
                        delta[:, 2],
                        color="#111111" if item is axes[0] else "#9467bd",
                        length=0.03,
                        normalize=False,
                        alpha=0.65,
                    )
                if state["residuals"] and item is axes[1] and visible:
                    residual = evaluation.residual[current_frame]
                    item.quiver(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        residual[:, 0],
                        residual[:, 1],
                        residual[:, 2],
                        color="#e41a1c",
                        length=0.03,
                        normalize=False,
                        alpha=0.7,
                    )
            figure.suptitle(
                f"shared connectivity | frame {current_frame} | "
                f"E_IM={evaluation.e_im[current_frame]:.6e}"
            )
        else:
            if state["source"]:
                contribution = (
                    evaluation.per_vertex_contribution[current_frame]
                    if evaluation is not None and state["contributions"]
                    else None
                )
                _draw_points(
                    axis,
                    graph_vertices,
                    title=f"source graph | frame {current_frame}",
                    contributions=contribution,
                )
                _draw_edges(axis, graph_vertices, graph_edges, show=edge_show)
                _setup_axis(axis, graph_vertices, title=f"source graph | frame {current_frame}")
                if state["laplacian"]:
                    delta = graph.source_laplacian[current_frame]
                    axis.quiver(
                        graph_vertices[:, 0],
                        graph_vertices[:, 1],
                        graph_vertices[:, 2],
                        delta[:, 0],
                        delta[:, 1],
                        delta[:, 2],
                        color="#111111",
                        length=0.03,
                        normalize=False,
                        alpha=0.65,
                    )
                if state["residuals"] and evaluation is not None:
                    delta = evaluation.residual[current_frame]
                    axis.quiver(
                        graph_vertices[:, 0],
                        graph_vertices[:, 1],
                        graph_vertices[:, 2],
                        delta[:, 0],
                        delta[:, 1],
                        delta[:, 2],
                        color="#e41a1c",
                        length=0.03,
                        normalize=False,
                        alpha=0.7,
                    )
        figure.canvas.draw_idle()

    draw(frame)
    slider_axis = figure.add_axes((0.20, 0.085, 0.60, 0.03))
    slider = Slider(slider_axis, "frame", start_frame, stop - 1, valinit=start_frame, valstep=1)
    slider.on_changed(lambda value: draw(int(value)))
    timer = figure.canvas.new_timer(interval=120)

    def set_frame(value: int) -> None:
        slider.set_val(max(start_frame, min(stop - 1, value)))

    def tick() -> None:
        if state["playing"]:
            next_frame = int(slider.val) + 1
            if next_frame >= stop:
                state["playing"] = False
            else:
                set_frame(next_frame)

    timer.add_callback(tick)

    def toggle(name: str) -> Any:
        def callback(_event: Any) -> None:
            state[name] = not state[name]
            draw(int(slider.val))

        return callback

    buttons = [
        (0.02, "prev", lambda _event: set_frame(int(slider.val) - 1)),
        (0.09, "next", lambda _event: set_frame(int(slider.val) + 1)),
        (0.16, "play", lambda _event: state.__setitem__("playing", not state["playing"])),
        (0.23, "source", toggle("source")),
        (0.30, "robot", toggle("robot")),
        (0.37, "HH", toggle("hand-hand")),
        (0.44, "HO", toggle("hand-object")),
        (0.51, "OO", toggle("object-object")),
        (0.58, "lap", toggle("laplacian")),
        (0.65, "res", toggle("residuals")),
        (0.72, "heat", toggle("contributions")),
    ]
    widgets = []
    for x, label, callback in buttons:
        button_axis = figure.add_axes((x, 0.025, 0.06, 0.04))
        widget = Button(button_axis, label)
        widget.on_clicked(callback)
        widgets.append(widget)
    close_axis = figure.add_axes((0.81, 0.025, 0.10, 0.04))
    close_button = Button(close_axis, "close")

    def close(_event: Any) -> None:
        state["playing"] = False
        timer.stop()
        plt.close(figure)

    close_button.on_clicked(close)
    figure._toporetarget_widgets = [  # type: ignore[attr-defined]
        *widgets,
        close_button,
        slider,
        timer,
    ]
    plt.show(block=False)
    return {
        "interactive": True,
        "frame_range": [start_frame, stop],
        "mode": mode,
        "slider": True,
        "play_pause": True,
        "controls": [label for _, label, _ in buttons] + ["close"],
        "timer_created": True,
        "timer_closed_on_close": True,
        "artist_stability": True,
        "graph_recomputed_on_frame_change": False,
    }


__all__ = ["launch_interaction_viewer", "render_interaction_frame"]
