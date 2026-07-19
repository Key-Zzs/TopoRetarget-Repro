"""Interactive raw/canonical GRAB HOI viewer with stable Matplotlib artists."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.geometry.se3 import object_to_scene, scene_to_object, scene_to_wrist


@dataclass
class GrabViewerOptions:
    mode: str = "canonical"
    layout: str = "overlay"
    reference_frame: str = "scene"
    frame: int = 0
    display_stride: int = 1
    show_mesh: bool = True
    show_mediapipe21: bool = True
    show_native_joints: bool = False
    show_skeleton: bool = True
    show_labels: bool = False
    show_object: bool = True
    show_table: bool = True
    show_contacts: bool = False
    show_axes: bool = False
    show_errors: bool = True
    playback_speed: float = 1.0
    raw_color: str = "tab:blue"
    canonical_color: str = "tab:orange"
    loop: bool = False


def _mpl(output: Path | None = None) -> Any:
    try:
        import matplotlib

        if output is not None:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("GRAB visualization needs `pip install -e '.[viz]'`.") from exc
    return plt


def _sequence_frame_count(groups: list[tuple[str, HOISequence, str]]) -> int:
    if not groups:
        raise ValueError("viewer has no sequence")
    return min(sequence.num_frames for _, sequence, _ in groups)


def _pairs() -> list[tuple[int, int]]:
    # The canonical Stage 3 layout is the single source of truth for ordering.
    return [
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
    ]


def _set_scatter(scatter: Any, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])


def _set_line(line: Any, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    line.set_data_3d(points[:, 0], points[:, 1], points[:, 2])


_MAX_RENDER_FACES = 20_000
_MAX_RENDER_POINTS = 2_000


class InteractiveHOIViewer:
    """A callback-testable viewer that updates existing artists in place."""

    def __init__(
        self,
        *,
        canonical: HOISequence | None = None,
        raw: HOISequence | None = None,
        options: GrabViewerOptions | None = None,
    ) -> None:
        self.options = options or GrabViewerOptions()
        if self.options.mode not in {"raw", "canonical", "compare"}:
            raise ValueError("mode must be raw, canonical, or compare")
        if self.options.layout not in {"overlay", "side-by-side"}:
            raise ValueError("layout must be overlay or side-by-side")
        self.canonical = canonical
        self.raw = raw
        if self.options.mode == "raw" and raw is None:
            raise ValueError("raw mode requires raw sequence")
        if self.options.mode == "canonical" and canonical is None:
            raise ValueError("canonical mode requires canonical sequence")
        if self.options.mode == "compare" and (raw is None or canonical is None):
            raise ValueError("compare mode requires raw and canonical sequences")
        self.groups: list[tuple[str, HOISequence, str]] = []
        if self.options.mode in {"raw", "compare"} and raw is not None:
            self.groups.append(("raw", raw, self.options.raw_color))
        if self.options.mode in {"canonical", "compare"} and canonical is not None:
            self.groups.append(("canonical", canonical, self.options.canonical_color))
        self.num_frames = _sequence_frame_count(self.groups)
        if self.options.display_stride < 1:
            raise ValueError("display_stride must be positive")
        if self.options.playback_speed <= 0:
            raise ValueError("playback_speed must be positive")
        self.display_frames = list(range(0, self.num_frames, self.options.display_stride))
        if self.display_frames[-1] != self.num_frames - 1:
            self.display_frames.append(self.num_frames - 1)
        if not 0 <= self.options.frame < self.num_frames:
            self.options.frame = 0
        self.frame = self.options.frame
        self.reference_frame = self.options.reference_frame
        self.visibility: dict[str, bool] = {
            "raw": True,
            "canonical": True,
            "mesh": self.options.show_mesh,
            "right_hand": True,
            "left_hand": True,
            "mediapipe21": self.options.show_mediapipe21,
            "native_joints": self.options.show_native_joints,
            "skeleton": self.options.show_skeleton,
            "labels": self.options.show_labels,
            "object": self.options.show_object,
            "table": self.options.show_table,
            "contacts": self.options.show_contacts,
            "axes": self.options.show_axes,
            "errors": self.options.show_errors,
        }
        self.control_enabled = {
            "right_hand": all(
                any(hand.side == "right" for hand in seq.hands) for _, seq, _ in self.groups
            ),
            "left_hand": all(
                any(hand.side == "left" for hand in seq.hands) for _, seq, _ in self.groups
            ),
        }
        self.figure: Any = None
        self.axes: list[Any] = []
        self.slider: Any = None
        self.timer: Any = None
        self.is_playing = False
        self.artists: list[Any] = []
        self._items: list[dict[str, Any]] = []
        self._key_connection: Any = None
        self.speed_slider: Any = None
        self._visibility_buttons: dict[str, Any] = {}
        self._build_figure()

    def _build_figure(self) -> None:
        plt = _mpl()
        self.figure = plt.figure(
            figsize=(13, 7) if self.options.layout == "side-by-side" else (8, 7)
        )
        if self.options.layout == "side-by-side" and len(self.groups) == 2:
            self.axes = [
                self.figure.add_subplot(1, 2, 1, projection="3d"),
                self.figure.add_subplot(1, 2, 2, projection="3d"),
            ]
        else:
            self.axes = [self.figure.add_subplot(111, projection="3d")]
        self._build_artists()
        self._build_controls()
        self._key_connection = self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self._set_limits()
        self.update(self.frame)

    def _axis_for_group(self, group_index: int) -> Any:
        return self.axes[group_index] if len(self.axes) == len(self.groups) else self.axes[0]

    def _reference_points(
        self, sequence: HOISequence, points: np.ndarray, frame: int, hand_id: str | None = None
    ) -> np.ndarray:
        if self.reference_frame == "scene":
            return np.asarray(points)
        if self.reference_frame == "object":
            if not sequence.rigid_objects:
                raise ValueError("object reference frame requires a primary object")
            return scene_to_object(
                sequence.rigid_objects[0].pose_scene.pose_scene[frame],
                np.asarray(points)[None, ...],
            )[0]
        if self.reference_frame == "right-wrist":
            hand = next((item for item in sequence.hands if item.side == "right"), None)
        else:
            hand = next((item for item in sequence.hands if item.side == "left"), None)
        if hand is None:
            raise ValueError(f"{self.reference_frame} reference frame requires that hand")
        return scene_to_wrist(
            hand.wrist_pose_scene.pose_scene[frame], np.asarray(points)[None, ...]
        )[0]

    def _scene_mesh(self, sequence: HOISequence, object_track: Any, frame: int) -> np.ndarray:
        return object_to_scene(
            object_track.pose_scene.pose_scene[frame], object_track.mesh.vertices_local[None, ...]
        )[0]

    def _add_poly(self, ax: Any, *, color: str, alpha: float) -> Any:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        poly = Poly3DCollection([], alpha=alpha, facecolor=color, edgecolor=color)
        ax.add_collection3d(poly)
        self.artists.append(poly)
        return poly

    def _build_artists(self) -> None:
        for group_index, (_name, sequence, color) in enumerate(self.groups):
            ax = self._axis_for_group(group_index)
            for hand in sequence.hands:
                item: dict[str, Any] = {
                    "group": group_index,
                    "source": _name,
                    "side": hand.side,
                    "hand": hand,
                }
                item["mesh"] = self._add_poly(ax, color=color, alpha=0.22)
                item["vertices"] = ax.plot(
                    [], [], [], color=color, linewidth=1.0, label=f"{_name} {hand.side} mesh"
                )[0]
                self.artists.append(item["vertices"])
                item["mp"] = ax.scatter(
                    [], [], [], color="crimson", s=18, label=f"{_name} mediapipe21"
                )
                item["native"] = ax.scatter(
                    [], [], [], color=color, s=10, label=f"{_name} native joints"
                )
                self.artists.extend([item["mp"], item["native"]])
                item["skeleton"] = [
                    ax.plot([], [], [], color="crimson", linewidth=1.1)[0] for _ in _pairs()
                ]
                self.artists.extend(item["skeleton"])
                item["labels"] = [ax.text(0, 0, 0, str(index), fontsize=7) for index in range(21)]
                self.artists.extend(item["labels"])
                item["axes"] = [
                    ax.plot([], [], [], color=axis_color, linewidth=1.2)[0]
                    for axis_color in ("r", "g", "b")
                ]
                self.artists.extend(item["axes"])
                self._items.append(item)
            for object_track in sequence.rigid_objects:
                is_table = (
                    object_track.object_id == "table"
                    or object_track.metadata.get("role") == "support_surface"
                )
                object_item = {
                    "group": group_index,
                    "source": _name,
                    "object": object_track,
                    "table": is_table,
                    "render_mesh": object_track.mesh.faces.shape[0] <= _MAX_RENDER_FACES,
                    "point_indices": np.linspace(
                        0,
                        object_track.mesh.vertices_local.shape[0] - 1,
                        min(object_track.mesh.vertices_local.shape[0], _MAX_RENDER_POINTS),
                        dtype=np.int64,
                    ),
                }
                object_color = "lightgray" if is_table else "tab:green"
                object_item["mesh"] = self._add_poly(ax, color=object_color, alpha=0.28)
                object_item["points"] = ax.scatter([], [], [], color=object_color, s=7)
                object_item["frame"] = [
                    ax.plot([], [], [], color=axis_color, linewidth=1.1)[0]
                    for axis_color in ("r", "g", "b")
                ]
                self.artists.append(object_item["points"])
                self.artists.extend(object_item["frame"])
                if not is_table:
                    object_item["contacts"] = ax.scatter(
                        [], [], [], color="magenta", s=18, label="contacts"
                    )
                    self.artists.append(object_item["contacts"])
                self._items.append(object_item)

    def _set_limits(self) -> None:
        points: list[np.ndarray] = []
        for _name, sequence, _color in self.groups:
            frame = min(self.frame, sequence.num_frames - 1)
            for hand in sequence.hands:
                if hand.vertices_scene is not None:
                    points.append(
                        self._reference_points(
                            sequence, hand.vertices_scene[frame], frame, hand.hand_id
                        )
                    )
            for object_track in sequence.rigid_objects:
                points.append(
                    self._reference_points(
                        sequence, self._scene_mesh(sequence, object_track, frame), frame
                    )
                )
        if not points:
            low, high = np.full(3, -0.1), np.full(3, 0.1)
        else:
            values = np.concatenate(points, axis=0)
            low = np.nanmin(values, axis=0)
            high = np.nanmax(values, axis=0)
            extent = max(float(np.max(high - low)), 1e-3)
            pad = max(0.12 * extent, 1e-3)
            center = (low + high) / 2.0
            half = extent / 2.0 + pad
            low, high = center - half, center + half
        for axis in self.axes:
            axis.set_xlim(float(low[0]), float(high[0]))
            axis.set_ylim(float(low[1]), float(high[1]))
            axis.set_zlim(float(low[2]), float(high[2]))
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_zlabel("z (m)")
            try:
                axis.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass

    def _build_controls(self) -> None:
        from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

        slider_ax = self.figure.add_axes((0.16, 0.035, 0.54, 0.025))
        self.slider = Slider(
            slider_ax, "frame", 0, max(self.num_frames - 1, 1), valinit=self.frame, valstep=1
        )
        self.slider.on_changed(self.on_slider_changed)
        buttons: list[tuple[str, tuple[float, float, float, float], Callable[[], Any]]] = [
            ("first", (0.02, 0.035, 0.06, 0.035), self.first_frame),
            ("prev", (0.09, 0.035, 0.06, 0.035), self.previous_frame),
            ("next", (0.72, 0.035, 0.06, 0.035), self.next_frame),
            ("last", (0.79, 0.035, 0.06, 0.035), self.last_frame),
            ("play", (0.87, 0.035, 0.08, 0.035), self.toggle_play),
        ]
        self.buttons: dict[str, Any] = {}
        for label, rect, callback in buttons:
            axis = self.figure.add_axes(rect)
            button = Button(axis, label)

            def _button_callback(_event: Any, cb: Callable[[], Any] = callback) -> None:
                cb()

            button.on_clicked(_button_callback)
            self.buttons[label] = button
        radio_ax = self.figure.add_axes((0.87, 0.55, 0.11, 0.18))
        self.reference_radio = RadioButtons(
            radio_ax, ("scene", "object", "right-wrist", "left-wrist"), active=0
        )
        self.reference_radio.on_clicked(self.set_reference_frame)
        checks_ax = self.figure.add_axes((0.86, 0.11, 0.13, 0.31))
        check_labels = (
            "mesh",
            "right hand",
            "left hand",
            "mediapipe21",
            "native",
            "skeleton",
            "labels",
            "object",
            "table",
            "contacts",
            "axes",
            "errors",
        )
        self.check_buttons = CheckButtons(
            checks_ax,
            check_labels,
            (
                self.visibility["mesh"],
                self.visibility["right_hand"],
                self.visibility["left_hand"],
                self.visibility["mediapipe21"],
                self.visibility["native_joints"],
                self.visibility["skeleton"],
                self.visibility["labels"],
                self.visibility["object"],
                self.visibility["table"],
                self.visibility["contacts"],
                self.visibility["axes"],
                self.visibility["errors"],
            ),
        )
        self.check_buttons.on_clicked(self._check_toggle)
        self._visibility_buttons = {
            label: key
            for label, key in zip(
                check_labels,
                (
                    "mesh",
                    "right_hand",
                    "left_hand",
                    "mediapipe21",
                    "native_joints",
                    "skeleton",
                    "labels",
                    "object",
                    "table",
                    "contacts",
                    "axes",
                    "errors",
                ),
                strict=True,
            )
        }
        for label, key in self._visibility_buttons.items():
            if key in self.control_enabled and not self.control_enabled[key]:
                index = check_labels.index(label)
                self.check_buttons.labels[index].set_color("0.55")
        source_ax = self.figure.add_axes((0.86, 0.44, 0.13, 0.12))
        self.source_buttons = CheckButtons(
            source_ax,
            ("raw", "canonical"),
            (self.visibility["raw"], self.visibility["canonical"]),
        )
        self.source_buttons.on_clicked(self._source_toggle)
        speed_ax = self.figure.add_axes((0.16, 0.072, 0.54, 0.018))
        self.speed_slider = Slider(
            speed_ax,
            "speed",
            0.25,
            4.0,
            valinit=self.options.playback_speed,
            valstep=0.25,
        )
        self.speed_slider.on_changed(self.set_playback_speed)
        interval = 1000.0 / max(
            float(next((seq.metadata.native_fps or 120 for _, seq, _ in self.groups), 120))
            * self.options.playback_speed,
            1.0,
        )
        self.timer = self.figure.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._timer_tick)

    def _check_toggle(self, label: str | None) -> None:
        if label is None:
            return
        key = self._visibility_buttons[label]
        if key in self.control_enabled and not self.control_enabled[key]:
            return
        self.visibility[key] = not self.visibility[key]
        self.update(self.frame)

    def _source_toggle(self, label: str | None) -> None:
        if label is None:
            return
        key = label
        self.visibility[key] = not self.visibility[key]
        self.update(self.frame)

    def set_playback_speed(self, value: float) -> None:
        speed = max(float(value), 0.01)
        self.options.playback_speed = speed
        if self.timer is not None:
            fps = float(next((seq.metadata.native_fps or 120 for _, seq, _ in self.groups), 120))
            self.timer.interval = 1000.0 / max(fps * speed, 1.0)

    def update(self, frame: int) -> None:
        frame = int(np.clip(frame, 0, self.num_frames - 1))
        frame = min(self.display_frames, key=lambda candidate: abs(candidate - frame))
        self.frame = frame
        for item in self._items:
            sequence = self.groups[item["group"]][1]
            if "hand" in item:
                hand = item["hand"]
                visible_side = self.visibility.get(hand.side + "_hand", True)
                if hand.vertices_scene is not None and frame < hand.vertices_scene.shape[0]:
                    vertices = self._reference_points(
                        sequence, hand.vertices_scene[frame], frame, hand.hand_id
                    )
                    _set_line(item["vertices"], vertices)
                    faces = (
                        [vertices[face] for face in hand.mesh.faces]
                        if hand.mesh is not None
                        else []
                    )
                    item["mesh"].set_verts(faces)
                item["vertices"].set_visible(visible_side and self.visibility[item["source"]])
                item["mesh"].set_visible(
                    visible_side and self.visibility["mesh"] and self.visibility[item["source"]]
                )
                native = next(
                    (
                        track
                        for name, track in hand.keypoint_tracks.items()
                        if name != "mediapipe21"
                    ),
                    None,
                )
                mp = hand.keypoint_tracks.get("mediapipe21")
                if mp is not None and frame < mp.positions_scene.shape[0]:
                    points = self._reference_points(
                        sequence, mp.positions_scene[frame], frame, hand.hand_id
                    )
                    _set_scatter(item["mp"], points)
                    item["mp"].set_visible(visible_side and self.visibility["mediapipe21"])
                    for line, (a, b) in zip(item["skeleton"], _pairs(), strict=True):
                        _set_line(line, points[[a, b]])
                        line.set_visible(
                            visible_side
                            and self.visibility["mediapipe21"]
                            and self.visibility["skeleton"]
                        )
                    for label, point, _index in zip(
                        item["labels"], points, range(min(21, len(points))), strict=False
                    ):
                        label.set_position((point[0], point[1]))
                        label.set_3d_properties(point[2], zdir="z")
                        label.set_visible(visible_side and self.visibility["labels"])
                else:
                    item["mp"].set_visible(False)
                    for artist in item["skeleton"] + item["labels"]:
                        artist.set_visible(False)
                if native is not None and frame < native.positions_scene.shape[0]:
                    _set_scatter(
                        item["native"],
                        self._reference_points(
                            sequence, native.positions_scene[frame], frame, hand.hand_id
                        ),
                    )
                item["native"].set_visible(visible_side and self.visibility["native_joints"])
                pose = hand.wrist_pose_scene.pose_scene[frame]
                if self.reference_frame == "object":
                    pose = (
                        np.linalg.inv(sequence.rigid_objects[0].pose_scene.pose_scene[frame]) @ pose
                    )
                elif self.reference_frame in {"right-wrist", "left-wrist"}:
                    selected = next(
                        hand_item
                        for hand_item in sequence.hands
                        if hand_item.side == self.reference_frame.split("-")[0]
                    )
                    pose = np.linalg.inv(selected.wrist_pose_scene.pose_scene[frame]) @ pose
                for index, line in enumerate(item["axes"]):
                    origin = pose[:3, 3]
                    _set_line(line, np.stack([origin, origin + pose[:3, index] * 0.03]))
                    line.set_visible(visible_side and self.visibility["axes"])
            else:
                track = item["object"]
                is_table = item["table"]
                points_scene = self._scene_mesh(sequence, track, frame)
                points = self._reference_points(sequence, points_scene, frame)
                if item["render_mesh"]:
                    faces = [points[face] for face in track.mesh.faces]
                    item["mesh"].set_verts(faces)
                else:
                    item["mesh"].set_verts([])
                key = "table" if is_table else "object"
                item["mesh"].set_visible(
                    item["render_mesh"]
                    and self.visibility[key]
                    and self.visibility.get("mesh", self.options.show_mesh)
                    and self.visibility[item["source"]]
                )
                _set_scatter(item["points"], points[item["point_indices"]])
                item["points"].set_visible(self.visibility[key] and self.visibility[item["source"]])
                pose = track.pose_scene.pose_scene[frame]
                if self.reference_frame == "object" and not is_table:
                    pose = np.eye(4)
                elif self.reference_frame == "right-wrist" or self.reference_frame == "left-wrist":
                    side = self.reference_frame.split("-")[0]
                    selected = next((hand for hand in sequence.hands if hand.side == side), None)
                    if selected is not None:
                        pose = np.linalg.inv(selected.wrist_pose_scene.pose_scene[frame]) @ pose
                for index, line in enumerate(item["frame"]):
                    origin = pose[:3, 3]
                    _set_line(line, np.stack([origin, origin + pose[:3, index] * 0.03]))
                    line.set_visible(self.visibility[key] and self.visibility["axes"])
                if "contacts" in item:
                    contact = next(
                        (
                            track_item
                            for track_item in sequence.contacts
                            if track_item.object_id == track.object_id
                        ),
                        None,
                    )
                    if contact is not None:
                        active = (
                            np.asarray(contact.binary[frame], dtype=bool)
                            if contact.binary is not None
                            else np.asarray(contact.labels[frame]) != 0
                        )
                        _set_scatter(item["contacts"], points[active])
                        item["contacts"].set_visible(
                            self.visibility["contacts"]
                            and self.visibility["object"]
                            and self.visibility[item["source"]]
                        )
        if self.slider is not None and int(self.slider.val) != frame:
            self.slider.set_val(frame)
        self._set_title()

    def _set_title(self) -> None:
        values = [f"frame {self.frame} | t={self._timestamp(self.frame):.6f}s"]
        values.append(
            f"reference={self.reference_frame} | display_stride={self.options.display_stride}"
        )
        self.figure.suptitle(" | ".join(values))

    def _timestamp(self, frame: int) -> float:
        return float(self.groups[0][1].timestamps[min(frame, self.groups[0][1].num_frames - 1)])

    def on_slider_changed(self, value: float) -> None:
        self.update(int(value))

    def first_frame(self) -> None:
        self.update(self.display_frames[0])

    def last_frame(self) -> None:
        self.update(self.display_frames[-1])

    def next_frame(self) -> None:
        index = self.display_frames.index(self.frame)
        if index + 1 >= len(self.display_frames):
            self.update(self.display_frames[0] if self.options.loop else self.display_frames[-1])
        else:
            self.update(self.display_frames[index + 1])

    def previous_frame(self) -> None:
        index = self.display_frames.index(self.frame)
        self.update(self.display_frames[-1] if index <= 0 else self.display_frames[index - 1])

    def toggle_play(self) -> bool:
        self.is_playing = not self.is_playing
        if self.timer is not None:
            self.timer.start() if self.is_playing else self.timer.stop()
        return self.is_playing

    def _timer_tick(self) -> None:
        if self.is_playing:
            self.next_frame()

    def set_reference_frame(self, value: str | None) -> None:
        if value is None:
            return
        if value not in {"scene", "object", "right-wrist", "left-wrist"}:
            raise ValueError(f"unsupported reference frame: {value}")
        if value == "right-wrist" and not all(
            any(hand.side == "right" for hand in seq.hands) for _, seq, _ in self.groups
        ):
            raise ValueError("right-wrist reference frame requires right hand in every view")
        if value == "left-wrist" and not all(
            any(hand.side == "left" for hand in seq.hands) for _, seq, _ in self.groups
        ):
            raise ValueError("left-wrist reference frame requires left hand")
        self.reference_frame = value
        self._set_limits()
        self.update(self.frame)

    def set_visibility(self, name: str, visible: bool) -> None:
        if name not in self.visibility:
            raise KeyError(name)
        self.visibility[name] = bool(visible)
        self.update(self.frame)

    def on_key(self, event: Any) -> None:
        key = getattr(event, "key", None)
        if key == "left":
            self.previous_frame()
        elif key == "right":
            self.next_frame()
        elif key == "home":
            self.first_frame()
        elif key == "end":
            self.last_frame()
        elif key == "space":
            self.toggle_play()

    def render_headless(
        self,
        output: str | Path,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> Path:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.suffix.lower() in {".gif", ".mp4"}:
            return self.render_animation(destination, start_frame=start_frame, end_frame=end_frame)
        if destination.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            raise ValueError("headless output must be PNG/JPG/SVG/PDF/GIF/MP4")
        self.figure.canvas.draw()
        self.figure.savefig(destination, dpi=160)
        return destination

    def render_animation(
        self,
        output: str | Path,
        *,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> Path:
        """Render optional GIF/MP4 output using only display frames."""

        destination = Path(output)
        start = 0 if start_frame is None else max(int(start_frame), 0)
        end = self.num_frames if end_frame is None else min(int(end_frame), self.num_frames)
        frames = [frame for frame in self.display_frames if start <= frame < end]
        if not frames:
            raise ValueError(f"animation range [{start}, {end}) contains no display frames")
        previous = self.frame
        self.is_playing = False
        try:
            if destination.suffix.lower() == ".gif":
                try:
                    from PIL import Image
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError("GIF output needs Pillow; install the viz extra") from exc
                images = []
                for frame in frames:
                    self.update(frame)
                    self.figure.canvas.draw()
                    rgba = np.asarray(self.figure.canvas.buffer_rgba()).copy()
                    images.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))
                images[0].save(
                    destination,
                    save_all=True,
                    append_images=images[1:],
                    duration=max(int(1000 / max(self._native_fps(), 1.0)), 1),
                    loop=0,
                )
                return destination
            if destination.suffix.lower() == ".mp4":
                try:
                    from matplotlib.animation import FFMpegWriter
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError("MP4 output needs Matplotlib's ffmpeg writer") from exc
                writer = FFMpegWriter(fps=max(int(round(self._native_fps())), 1))
                try:
                    with writer.saving(self.figure, str(destination), dpi=160):
                        for frame in frames:
                            self.update(frame)
                            self.figure.canvas.draw()
                            writer.grab_frame()
                except (FileNotFoundError, RuntimeError) as exc:
                    raise RuntimeError(
                        "MP4 output requires an available ffmpeg executable"
                    ) from exc
                return destination
            raise ValueError("animation output must be GIF or MP4")
        finally:
            self.update(previous)

    def _native_fps(self) -> float:
        return float(next((seq.metadata.native_fps or 120 for _, seq, _ in self.groups), 120))

    def show(self) -> None:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError(
                "interactive viewer needs a GUI backend/display; use --output for headless PNG"
            )
        _mpl().show()

    def close(self) -> None:
        self.is_playing = False
        if self.timer is not None:
            self.timer.stop()
        if self.figure is not None:
            self.figure.canvas.mpl_disconnect(self._key_connection)
            _mpl().close(self.figure)


GrabInteractiveViewer = InteractiveHOIViewer


def render_grab_view(
    *,
    canonical: HOISequence | None = None,
    raw: HOISequence | None = None,
    options: GrabViewerOptions | None = None,
    output: str | Path | None = None,
    interactive: bool = False,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> InteractiveHOIViewer:
    viewer = InteractiveHOIViewer(canonical=canonical, raw=raw, options=options)
    if output is not None:
        viewer.render_headless(output, start_frame=start_frame, end_frame=end_frame)
    if interactive:
        viewer.show()
    return viewer


__all__ = ["GrabInteractiveViewer", "GrabViewerOptions", "InteractiveHOIViewer", "render_grab_view"]
