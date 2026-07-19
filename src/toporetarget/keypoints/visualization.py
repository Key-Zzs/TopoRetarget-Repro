"""Matplotlib validation views for source MANO and MediaPipe21 geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.se3 import object_to_scene, scene_to_wrist
from toporetarget.keypoints.registry import get_layout


def render_keypoint_view(
    sequence: HOISequence,
    *,
    hand_id: str,
    frame: int,
    view: str = "scene",
    show_source_layout: bool = False,
    show_mesh: bool = True,
    show_labels: bool = False,
    output: str | Path,
) -> Path:
    """Render one explicit frame; this function imports matplotlib only when called."""

    if view not in {"scene", "wrist"}:
        raise ValueError("view must be scene or wrist")
    if frame < 0 or frame >= sequence.num_frames:
        raise ValueError(f"frame {frame} is outside [0, {sequence.num_frames})")
    hand = sequence.hand(hand_id)
    track = hand.keypoint_tracks["mediapipe21"]
    target_layout = get_layout("mediapipe21")
    target = track.positions_scene[frame]
    source_track = next(
        (value for key, value in hand.keypoint_tracks.items() if key != "mediapipe21"), None
    )
    source = None if source_track is None else source_track.positions_scene[frame]
    mesh = None if hand.vertices_scene is None else hand.vertices_scene[frame]
    wrist_pose = hand.wrist_pose_scene.pose_scene[frame : frame + 1]
    if view == "wrist":
        target = scene_to_wrist(wrist_pose, target[None, ...])[0]
        if source is not None:
            source = scene_to_wrist(wrist_pose, source[None, ...])[0]
        if mesh is not None:
            mesh = scene_to_wrist(wrist_pose, mesh[None, ...])[0]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    if show_mesh and mesh is not None and hand.mesh is not None:
        faces = hand.mesh.faces.astype(np.int64)
        polygons = mesh[faces]
        collection = Poly3DCollection(polygons, alpha=0.18, facecolor="lightgray", edgecolor="none")
        axes.add_collection3d(collection)
    if show_source_layout and source is not None:
        axes.scatter(
            source[:, 0], source[:, 1], source[:, 2], c="tab:blue", s=18, label="MANO source"
        )
    axes.scatter(
        target[:, 0], target[:, 1], target[:, 2], c="tab:orange", s=28, label="mediapipe21 joints"
    )
    tips = set(target_layout.fingertip_indices)
    if tips:
        tip_array = target[list(sorted(tips))]
        axes.scatter(
            tip_array[:, 0],
            tip_array[:, 1],
            tip_array[:, 2],
            c="tab:red",
            s=38,
            label="tip anchors",
        )
    for parent, child in target_layout.edges:
        if np.isfinite(target[[parent, child]]).all():
            axes.plot(*target[[parent, child]].T, color="tab:orange", linewidth=1.5)
    if show_labels:
        for index, name in enumerate(target_layout.semantic_names):
            if np.isfinite(target[index]).all():
                axes.text(*target[index], str(index) + ":" + name, fontsize=7)
    wrist = target[target_layout.wrist_index]
    if np.isfinite(wrist).all():
        extent = max(float(np.ptp(target[np.isfinite(target).all(axis=1)], axis=0).max()), 0.05)
        axes.quiver(*wrist, extent, 0, 0, color="r", length=extent, normalize=False)
        axes.quiver(*wrist, 0, extent, 0, color="g", length=extent, normalize=False)
        axes.quiver(*wrist, 0, 0, extent, color="b", length=extent, normalize=False)
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_zlabel("z [m]")
    axes.set_title(
        f"{view} frame | hand={hand.side} | frame={frame} | t={sequence.timestamps[frame]:.6f}s\n"
        f"profile={track.provenance.get('mapping_profile_id', 'unknown')}"
    )
    axes.legend(loc="best")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def launch_interactive_keypoint_viewer(
    sequence: HOISequence,
    *,
    hand_id: str,
    start_frame: int = 0,
    end_frame: int | None = None,
    view: str = "scene",
    show_source_layout: bool = False,
    show_mesh: bool = True,
    show_target: bool = True,
    show_skeleton: bool = True,
    show_labels: bool = False,
    show_object_mesh: bool = False,
    show_axes: bool = True,
) -> None:
    """Launch a local, read-only interactive sequence viewer.

    All displayed wrist-relative geometry is derived into temporary arrays.  The
    canonical scene-frame sequence, including its keypoint coordinates, is never
    modified by callbacks or by the renderer.
    """

    if view not in {"scene", "wrist"}:
        raise ValueError("view must be scene or wrist")
    frame_end = sequence.num_frames if end_frame is None else end_frame
    if start_frame < 0 or frame_end > sequence.num_frames or start_frame >= frame_end:
        raise ValueError(
            f"frame range [{start_frame}, {frame_end}) is outside [0, {sequence.num_frames})"
        )
    hand = sequence.hand(hand_id)
    target_track = hand.keypoint_tracks.get("mediapipe21")
    if target_track is None:
        raise ValueError(f"hand {hand_id!r} has no mediapipe21 track")
    target_layout = get_layout("mediapipe21")
    source_track = next(
        (value for key, value in hand.keypoint_tracks.items() if key != "mediapipe21"), None
    )

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    figure = plt.figure(figsize=(12, 9))
    axes = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.04, right=0.79, bottom=0.25, top=0.95)
    state: dict[str, Any] = {
        "frame": int(start_frame),
        "view": view,
        "show_source_layout": bool(show_source_layout),
        "show_mesh": bool(show_mesh),
        "show_target": bool(show_target),
        "show_skeleton": bool(show_skeleton),
        "show_labels": bool(show_labels),
        "show_object_mesh": bool(show_object_mesh),
        "show_axes": bool(show_axes),
    }

    def _display_points(points_scene: np.ndarray, frame: int) -> np.ndarray:
        values = np.asarray(points_scene[frame], dtype=np.float64)
        if state["view"] == "wrist":
            pose = hand.wrist_pose_scene.pose_scene[frame]
            return scene_to_wrist(pose, values)
        return values.copy()

    def _display_mesh(frame: int) -> np.ndarray | None:
        if hand.vertices_scene is None:
            return None
        values = np.asarray(hand.vertices_scene[frame], dtype=np.float64)
        if state["view"] == "wrist":
            return scene_to_wrist(hand.wrist_pose_scene.pose_scene[frame], values)
        return values.copy()

    def _display_object_mesh(frame: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        if not state["show_object_mesh"]:
            return []
        meshes: list[tuple[np.ndarray, np.ndarray, str]] = []
        for object_track in sequence.rigid_objects:
            vertices = object_to_scene(
                object_track.pose_scene.pose_scene[frame], object_track.mesh.vertices_local
            )
            if state["view"] == "wrist":
                vertices = scene_to_wrist(hand.wrist_pose_scene.pose_scene[frame], vertices)
            meshes.append(
                (vertices, object_track.mesh.faces.astype(np.int64), object_track.object_id)
            )
        return meshes

    def _draw_axes(origin: np.ndarray, rotation: np.ndarray, label: str) -> None:
        if not state["show_axes"]:
            return
        extent = 0.08
        basis = np.asarray(rotation, dtype=np.float64)[:3, :3]
        colors = ("tab:red", "tab:green", "tab:blue")
        for index, color in enumerate(colors):
            direction = basis[:, index] * extent
            axes.quiver(*origin, *direction, color=color, length=extent, normalize=False)
        axes.text(*origin, label, fontsize=8)

    def _draw(frame: int) -> None:
        axes.clear()
        target = _display_points(target_track.positions_scene, frame)
        source = (
            None if source_track is None else _display_points(source_track.positions_scene, frame)
        )
        mesh = _display_mesh(frame)
        if state["show_mesh"] and mesh is not None and hand.mesh is not None:
            faces = hand.mesh.faces.astype(np.int64)
            axes.add_collection3d(
                Poly3DCollection(mesh[faces], alpha=0.18, facecolor="lightgray", edgecolor="none")
            )
        for vertices, faces, object_id in _display_object_mesh(frame):
            if faces.size:
                axes.add_collection3d(
                    Poly3DCollection(
                        vertices[faces],
                        alpha=0.2,
                        facecolor="tab:purple",
                        edgecolor="none",
                        label=f"object mesh: {object_id}",
                    )
                )
        if state["show_source_layout"] and source is not None:
            axes.scatter(
                source[:, 0],
                source[:, 1],
                source[:, 2],
                c="tab:blue",
                s=18,
                label="source MANO joints",
            )
        if state["show_target"]:
            axes.scatter(
                target[:, 0],
                target[:, 1],
                target[:, 2],
                c="tab:orange",
                s=28,
                label="MediaPipe-21",
            )
            tips = np.asarray(target[list(target_layout.fingertip_indices)], dtype=np.float64)
            if tips.size:
                axes.scatter(
                    tips[:, 0], tips[:, 1], tips[:, 2], c="tab:red", s=38, label="tip anchors"
                )
        if state["show_skeleton"] and state["show_target"]:
            for parent, child in target_layout.edges:
                if np.isfinite(target[[parent, child]]).all():
                    axes.plot(*target[[parent, child]].T, color="tab:orange", linewidth=1.5)
        if state["show_labels"] and state["show_target"]:
            for index, name in enumerate(target_layout.semantic_names):
                if np.isfinite(target[index]).all():
                    axes.text(*target[index], f"{index}:{name}", fontsize=7)

        wrist_pose = hand.wrist_pose_scene.pose_scene[frame]
        if state["view"] == "scene":
            _draw_axes(np.zeros(3), np.eye(3), "scene")
            _draw_axes(wrist_pose[:3, 3], wrist_pose[:3, :3], "wrist")
        else:
            scene_rotation = wrist_pose[:3, :3].T
            scene_origin = -scene_rotation @ wrist_pose[:3, 3]
            _draw_axes(scene_origin, scene_rotation, "scene")
            _draw_axes(np.zeros(3), np.eye(3), "wrist")

        displayed = [target]
        if state["show_source_layout"] and source is not None:
            displayed.append(source)
        if mesh is not None and state["show_mesh"]:
            displayed.append(mesh)
        for vertices, _faces, _object_id in _display_object_mesh(frame):
            displayed.append(vertices)
        finite = np.concatenate(
            [value[np.isfinite(value).all(axis=1)] for value in displayed if value.size], axis=0
        )
        if finite.size:
            minimum = finite.min(axis=0)
            maximum = finite.max(axis=0)
            centre = (minimum + maximum) / 2.0
            half_extent = max(float(np.max(maximum - minimum)) / 2.0, 0.05)
            axes.set_xlim(centre[0] - half_extent, centre[0] + half_extent)
            axes.set_ylim(centre[1] - half_extent, centre[1] + half_extent)
            axes.set_zlim(centre[2] - half_extent, centre[2] + half_extent)
        axes.set_xlabel("x [m]")
        axes.set_ylabel("y [m]")
        axes.set_zlabel("z [m]")
        axes.set_title(
            f"{state['view']} view | hand={hand.side} | frame={frame} | "
            f"t={sequence.timestamps[frame]:.6f}s\n"
            f"mapping profile={target_track.provenance.get('mapping_profile_id', 'unknown')}"
        )
        handles, labels = axes.get_legend_handles_labels()
        if handles:
            axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
        figure.canvas.draw_idle()

    slider_axes = figure.add_axes((0.16, 0.13, 0.56, 0.035))
    frame_slider = Slider(
        slider_axes,
        "frame",
        valmin=start_frame,
        valmax=frame_end - 1,
        valinit=start_frame,
        valstep=1,
    )
    previous_axes = figure.add_axes((0.16, 0.065, 0.12, 0.045))
    next_axes = figure.add_axes((0.30, 0.065, 0.12, 0.045))
    previous_button = Button(previous_axes, "previous")
    next_button = Button(next_axes, "next")
    view_axes = figure.add_axes((0.81, 0.67, 0.16, 0.16))
    view_buttons = RadioButtons(view_axes, ("scene", "wrist"), active=(0 if view == "scene" else 1))
    toggle_axes = figure.add_axes((0.81, 0.30, 0.17, 0.32))
    toggle_labels = (
        "MANO mesh",
        "source MANO joints",
        "MediaPipe-21",
        "skeleton edges",
        "semantic labels",
        "object mesh",
        "scene/wrist axes",
    )
    toggle_buttons = CheckButtons(
        toggle_axes,
        toggle_labels,
        (
            state["show_mesh"],
            state["show_source_layout"],
            state["show_target"],
            state["show_skeleton"],
            state["show_labels"],
            state["show_object_mesh"],
            state["show_axes"],
        ),
    )

    def _set_frame(value: float) -> None:
        state["frame"] = int(round(value))
        _draw(state["frame"])

    def _step(delta: int) -> None:
        new_frame = min(max(state["frame"] + delta, start_frame), frame_end - 1)
        if new_frame != state["frame"]:
            frame_slider.set_val(new_frame)
        else:
            _draw(new_frame)

    def _set_view(label: str | None) -> None:
        if label is None:
            return
        state["view"] = label
        _draw(state["frame"])

    toggle_keys = (
        "show_mesh",
        "show_source_layout",
        "show_target",
        "show_skeleton",
        "show_labels",
        "show_object_mesh",
        "show_axes",
    )

    def _toggle(label: str | None) -> None:
        if label is None:
            return
        key = toggle_keys[toggle_labels.index(label)]
        state[key] = not bool(state[key])
        _draw(state["frame"])

    frame_slider.on_changed(_set_frame)
    previous_button.on_clicked(lambda _event: _step(-1))
    next_button.on_clicked(lambda _event: _step(1))
    view_buttons.on_clicked(_set_view)
    toggle_buttons.on_clicked(_toggle)
    _draw(start_frame)
    plt.show()


__all__ = ["launch_interactive_keypoint_viewer", "render_keypoint_view"]
