#!/usr/bin/env python3
"""Render small static P3-B.5 visual/proxy diagnostics from frozen traces.

The images are explanatory only.  Runtime convex-proxy metrics remain the
formal geometry authority; non-watertight object visual meshes cannot provide
signed visual penetration evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = REPO_ROOT / ".local/reports/stage16_p3b5_geometry_attribution"
MANIFEST = (
    REPO_ROOT
    / ".local/reports/stage16d_metric_qualification_and_ppo"
    / "runtime_collision_geometry_manifest.json"
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P3B5_JSON_OBJECT_REQUIRED:{path}")
    return value


def rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_wxyz / np.linalg.norm(quaternion_wxyz)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return vertices @ rotation(pose[3:7]).T + pose[:3]


def add_mesh(
    axis: object, vertices: np.ndarray, faces: np.ndarray, *, color: str, alpha: float, label: str
) -> None:
    triangles = vertices[faces]
    collection = Poly3DCollection(triangles, facecolor=color, edgecolor="none", alpha=alpha)
    collection.set_label(label)
    axis.add_collection3d(collection)


def render(clip: str, case_id: str) -> dict[str, object]:
    import trimesh

    trace_path = OUTPUT / "counterfactuals" / clip / case_id / "open_loop/A" / "trace.npz"
    manifest = read_json(MANIFEST)
    hand = manifest["hand_shapes"][4]
    object_shape = manifest["object_shapes"][clip][0]
    with np.load(trace_path, allow_pickle=False) as trace:
        hand_pose = np.asarray(trace["hand_collision_body_pose"], dtype=np.float64)[0, 4]
        object_pose = np.asarray(trace["object_pose"], dtype=np.float64)[0]
        reference_pose = np.asarray(trace["object_reference"], dtype=np.float64)[0]
    hand_proxy_v = world(np.asarray(hand["convex_vertices_m"]), hand_pose)
    hand_proxy_f = np.asarray(hand["triangle_indices"], dtype=np.int64)
    object_proxy_v = world(np.asarray(object_shape["convex_vertices_m"]), object_pose)
    object_proxy_f = np.asarray(object_shape["triangle_indices"], dtype=np.int64)
    object_visual = trimesh.load_mesh(
        REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_objects" / f"{clip}.obj",
        process=False,
    )
    hand_visual = trimesh.load_mesh(REPO_ROOT / hand["source_asset_path"], process=False)
    if not isinstance(object_visual, trimesh.Trimesh) or not isinstance(
        hand_visual, trimesh.Trimesh
    ):
        raise RuntimeError("P3B5_VISUAL_MESH_LOAD_FAILURE")
    figure = plt.figure(figsize=(8, 7), dpi=160)
    axis = figure.add_subplot(projection="3d")
    add_mesh(
        axis,
        world(np.asarray(object_visual.vertices), object_pose),
        np.asarray(object_visual.faces),
        color="#b0b8c2",
        alpha=0.32,
        label="actual object visual mesh",
    )
    add_mesh(
        axis,
        object_proxy_v,
        object_proxy_f,
        color="#d62728",
        alpha=0.18,
        label="formal object convex proxy",
    )
    add_mesh(
        axis,
        world(np.asarray(object_visual.vertices), reference_pose),
        np.asarray(object_visual.faces),
        color="#2ca02c",
        alpha=0.10,
        label="reference object ghost",
    )
    add_mesh(
        axis,
        world(np.asarray(hand_visual.vertices), hand_pose),
        np.asarray(hand_visual.faces),
        color="#1f77b4",
        alpha=0.65,
        label="actual index-distal visual mesh",
    )
    add_mesh(
        axis,
        hand_proxy_v,
        hand_proxy_f,
        color="#ff7f0e",
        alpha=0.28,
        label="highlighted formal hand proxy",
    )
    points = np.concatenate((object_proxy_v, hand_proxy_v), axis=0)
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low + high) * 0.5, max(float((high - low).max()) * 0.7, 0.04)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=-58)
    axis.set_title(f"{clip} frame 0: index-distal / object convex-pair violation")
    axis.legend(loc="upper left", fontsize=7)
    figure.tight_layout()
    target = OUTPUT / "proxy_audit/screenshots" / f"{clip}_frame000_proxy_visual.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return {
        "clip": clip,
        "case_id": case_id,
        "frame": 0,
        "screenshot": str(target.resolve()),
        "visual_object_watertight": bool(object_visual.is_watertight),
        "visual_hand_watertight": bool(hand_visual.is_watertight),
        "formal_authority": "runtime collision proxy only",
    }


def main() -> int:
    rows = [
        render("hocap_170105", "C2_170105_MAX_FAILURE"),
        render("hocap_170650", "PRIMARY_COMMON_MODE_FAILURE_CASE_170650"),
    ]
    payload = {
        "schema_version": "Stage16P3B5ProxyVisualFramesV1",
        "rows": rows,
        "automated_review": (
            "VISUAL_MESH_NONWATERTIGHT: no signed visual-overlap conclusion; human review required."
        ),
    }
    (OUTPUT / "proxy_audit/screenshots/receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "P3B5_PROXY_FRAMES_RENDERED", "count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
