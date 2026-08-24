#!/usr/bin/env python3
"""Render an interactive HOCap EpisodeV1 MANO/object lifecycle inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.adapters.datasets.hocap import hocap_mano_storage_index  # noqa: E402
from toporetarget.adapters.datasets.stage12_base import (  # noqa: E402
    DEFAULT_MANO_ROOT,
    DEFAULT_STORAGE_ROOT,
    load_mesh,
    render_mano_pca45,
    sha256_paths,
)

EVENT_FIELDS = (
    "start_frame",
    "approach_frame",
    "contact_frame",
    "pickup_frame",
    "transport_frame",
    "place_frame",
    "release_frame",
    "retreat_frame",
    "end_frame",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--mano-model-root", type=Path, default=DEFAULT_MANO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sanity-output", type=Path)
    parser.add_argument(
        "--include-other-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--max-object-faces", type=int, default=2000)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _episode(path: Path, episode_id: str) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("HOCAP_EPISODE_VISUALIZATION_INDEX_LIST_REQUIRED")
    selected = [
        row for row in rows if isinstance(row, dict) and row.get("episode_id") == episode_id
    ]
    if len(selected) != 1:
        raise ValueError(f"HOCAP_EPISODE_VISUALIZATION_ID_CARDINALITY:{len(selected)}")
    return selected[0]


def _mesh_subset(
    vertices: np.ndarray, faces: np.ndarray, maximum_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) <= maximum_faces:
        return vertices, faces
    selected = np.linspace(0, len(faces) - 1, maximum_faces, dtype=np.int64)
    selected_faces = faces[selected]
    used = np.unique(selected_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[selected_faces]


def _object_poses(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    result = np.broadcast_to(np.eye(4), (len(raw), 4, 4)).copy()
    result[:, :3, :3] = Rotation.from_quat(raw[:, :4]).as_matrix()
    result[:, :3, 3] = raw[:, 4:]
    return result


def _transform(vertices: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum("tij,vj->tvi", poses[:, :3, :3], vertices) + poses[:, None, :3, 3]


def _selected_frames(row: dict[str, Any], maximum: int) -> np.ndarray:
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    count = min(maximum, end - start)
    frames = set(np.linspace(start, end - 1, count, dtype=np.int64).tolist())
    frames.update(
        int(row[name])
        for name in EVENT_FIELDS
        if isinstance(row.get(name), int) and start <= int(row[name]) < end
    )
    return np.asarray(sorted(frames), dtype=np.int64)


def _mesh_trace(go: Any, vertices: np.ndarray, faces: np.ndarray, **kwargs: object) -> Any:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        **kwargs,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.max_frames < 2 or args.max_object_faces < 4:
        raise ValueError("HOCAP_EPISODE_VISUALIZATION_BUDGET_INVALID")
    index_path = args.episode_index.resolve()
    row = _episode(index_path, args.episode_id)
    active_side = str(row["active_hand"])
    if active_side not in {"left", "right"}:
        raise ValueError("HOCAP_EPISODE_VISUALIZATION_SINGLE_HAND_REQUIRED")
    requested_data_root = args.data_root.resolve()
    dataset_root = (
        requested_data_root
        if (requested_data_root / "data").is_dir()
        else requested_data_root / "HOCap"
    )
    sequence_dir = dataset_root / "data" / str(row["raw_sequence"])
    meta_path = sequence_dir / "meta.yaml"
    poses_m_path = sequence_dir / "poses_m.npy"
    poses_o_path = sequence_dir / "poses_o.npy"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    sides = [str(value).lower() for value in (meta.get("mano_sides") or [])]
    object_ids = [str(value) for value in (meta.get("object_ids") or [])]
    if active_side not in sides or str(row["target_object"]) not in object_ids:
        raise ValueError("HOCAP_EPISODE_VISUALIZATION_TARGET_DRIFT")
    start = int(row["start_frame"])
    end = int(row["end_frame"])
    selected_global = _selected_frames(row, args.max_frames)
    selected_local = selected_global - start
    poses_m = np.asarray(np.load(poses_m_path, mmap_mode="r"), dtype=np.float64)
    poses_o = np.asarray(np.load(poses_o_path, mmap_mode="r"), dtype=np.float64)
    if poses_o.shape[1] == poses_m.shape[1] and poses_o.shape[0] != poses_m.shape[1]:
        poses_o = poses_o.transpose(1, 0, 2)
    calibration_path = dataset_root / "data/calibration/mano" / f"{row['subject']}.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8")) or {}
    betas = np.asarray(calibration.get("betas"), dtype=np.float64)
    source_hash = sha256_paths([meta_path, poses_m_path, calibration_path])
    rendered: dict[str, np.ndarray] = {}
    hand_faces: np.ndarray | None = None
    render_sides = [active_side]
    if args.include_other_hand:
        render_sides.extend(side for side in sides if side != active_side)
    for side in render_sides:
        hand_index = hocap_mano_storage_index(side)
        hand = render_mano_pca45(
            poses_m[hand_index, start:end],
            side=side,
            mano_model_root=args.mano_model_root.resolve(),
            betas=betas,
            dataset_name="hocap",
            source_annotation_path=poses_m_path,
            source_annotation_hash=source_hash,
        )
        rendered[side] = np.asarray(hand.vertices, dtype=np.float64)[selected_local]
        if hand_faces is None:
            hand_faces = np.asarray(hand.faces, dtype=np.int64)
    assert hand_faces is not None
    object_index = object_ids.index(str(row["target_object"]))
    object_mesh_path = (
        dataset_root / "data/models" / str(row["target_object"]) / "textured_mesh.obj"
    )
    object_vertices, object_faces = load_mesh(object_mesh_path)
    object_vertices, object_faces = _mesh_subset(
        object_vertices, object_faces, args.max_object_faces
    )
    object_pose = _object_poses(poses_o[start:end, object_index])[selected_local]
    object_world = _transform(object_vertices, object_pose)

    import plotly.graph_objects as go

    active_color = "#1f77b4" if active_side == "left" else "#d62728"
    traces = [
        _mesh_trace(
            go,
            rendered[active_side][0],
            hand_faces,
            name=f"active MANO {active_side}",
            color=active_color,
            opacity=0.82,
            flatshading=True,
        ),
        _mesh_trace(
            go,
            object_world[0],
            object_faces,
            name=f"target {row['target_object']}",
            color="#8c8c8c",
            opacity=0.72,
            flatshading=True,
        ),
    ]
    other_sides = [side for side in render_sides if side != active_side]
    for side in other_sides:
        traces.append(
            _mesh_trace(
                go,
                rendered[side][0],
                hand_faces,
                name=f"other MANO {side}",
                color="#2ca02c",
                opacity=0.28,
                flatshading=True,
            )
        )
    event_names: list[str] = []
    event_positions: list[np.ndarray] = []
    for name in EVENT_FIELDS:
        frame = row.get(name)
        if isinstance(frame, int) and start <= frame < end:
            nearest = int(np.argmin(np.abs(selected_global - frame)))
            event_names.append(f"{name.removesuffix('_frame')} @ {frame}")
            event_positions.append(np.mean(rendered[active_side][nearest], axis=0))
    if event_positions:
        positions = np.asarray(event_positions)
        traces.append(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers+text",
                text=event_names,
                textposition="top center",
                marker={"size": 4, "color": "#ffbf00"},
                name="episode event markers",
            )
        )
    figure_frames = []
    animated_trace_count = 2 + len(other_sides)
    for local_index, global_frame in enumerate(selected_global):
        event_at_frame = [
            name.removesuffix("_frame")
            for name in EVENT_FIELDS
            if row.get(name) == int(global_frame)
        ]
        frame_data = [
            _mesh_trace(
                go,
                rendered[active_side][local_index],
                hand_faces,
                color=active_color,
                opacity=0.82,
                flatshading=True,
            ),
            _mesh_trace(
                go,
                object_world[local_index],
                object_faces,
                color="#8c8c8c",
                opacity=0.72,
                flatshading=True,
            ),
        ]
        frame_data.extend(
            _mesh_trace(
                go,
                rendered[side][local_index],
                hand_faces,
                color="#2ca02c",
                opacity=0.28,
                flatshading=True,
            )
            for side in other_sides
        )
        label = str(global_frame)
        if event_at_frame:
            label += " · " + "/".join(event_at_frame)
        figure_frames.append(
            go.Frame(
                data=frame_data,
                traces=list(range(animated_trace_count)),
                name=str(global_frame),
                layout={"title": f"{row['episode_id']} — raw frame {label}"},
            )
        )
    figure = go.Figure(data=traces, frames=figure_frames)
    figure.update_layout(
        title=f"{row['episode_id']} — raw frame {selected_global[0]}",
        scene={
            "aspectmode": "data",
            "xaxis_title": "world x (m)",
            "yaxis_title": "world y (m)",
            "zaxis_title": "world z (m)",
        },
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "raw frame: "},
                "steps": [
                    {
                        "label": frame.name,
                        "method": "animate",
                        "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0}}],
                    }
                    for frame in figure_frames
                ],
            }
        ],
        annotations=[
            {
                "text": "Segmentation lifecycle markers are not PPO reward phase gates.",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.0,
                "showarrow": False,
            }
        ],
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"HOCAP_EPISODE_VISUALIZATION_REFUSES_OVERWRITE:{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output, include_plotlyjs=True, full_html=True)
    sanity_path = (
        args.sanity_output.resolve()
        if args.sanity_output is not None
        else output.with_suffix(".sanity.json")
    )
    checks = {
        "target_object_correct": str(row["target_object"]) in object_ids,
        "hand_side_correct": active_side in sides,
        "episode_contains_pickup": isinstance(row.get("pickup_frame"), int),
        "episode_contains_place_release": isinstance(row.get("place_frame"), int)
        and isinstance(row.get("release_frame"), int),
        "episode_contains_retreat": isinstance(row.get("retreat_frame"), int),
    }
    sanity = {
        "schema_version": "HOCapEpisodeVisualizationSanityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "episode_id": row["episode_id"],
        "active_hand": active_side,
        "target_object": row["target_object"],
        "source_frame_range": [start, end],
        "rendered_frame_count": len(selected_global),
        "rendered_global_frames": selected_global.tolist(),
        "include_other_hand": args.include_other_hand,
        "event_markers": {name: row.get(name) for name in EVENT_FIELDS},
        "checks": checks,
        "html": {"path": str(output), "sha256": _sha256(output)},
        "episode_index": {"path": str(index_path), "sha256": _sha256(index_path)},
    }
    _atomic_json(sanity_path, sanity)
    print(json.dumps(sanity, indent=2, sort_keys=True))
    return 0 if sanity["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
