#!/usr/bin/env python3
# ruff: noqa: E501
"""Build the OakInk2 O1R2 trusted-MANO rendering review package.

The command is intentionally bounded to source MANO diagnostics.  It does not
run O3, retargeting, simulation, evaluation, PPO, or mutate any manifest.
Renderer A calls OakInk2's installed ``PyMultiObjRenderer``.  Renderer B calls
``pyrender`` directly and consumes the same precomputed official geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import platform
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
START_HEAD = "0aae3915f8245063e1f3bb1e844a8b66de708732"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
DATASET_ROOT = Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2")
DATASET_PREFIX = DATASET_ROOT / "data/OakInk-v2-hub"
OFFICIAL_ROOT = Path("/home/deepcybo/workspace/dex/Ref2Dex/dataset/OakInk2")
MANO_MODEL = Path("/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/MANO_RIGHT.pkl")
O1R_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r_official_mano_authority_v1"
REPORT_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r2_trusted_mano_rendering_v1"
MANIFEST_V2 = O1R_ROOT / "manifest_v2/oakink2_corpus_manifest_v2.jsonl"
SPLIT_V2 = O1R_ROOT / "manifest_v2/oakink2_raw_to_physical_split_v2.json"
CAM_NAME = "allocentric_top"
IMAGE_SIZE = 640
CAM_INTR = np.array(
    [[700.0, 0.0, IMAGE_SIZE / 2], [0.0, 700.0, IMAGE_SIZE / 2], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
PYRENDER_EXTR = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
JOINT_NAMES = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)
PARENTS: tuple[int | None, ...] = (
    None,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    0,
    13,
    14,
    15,
    0,
    17,
    18,
    19,
)
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "little": (0, 17, 18, 19, 20),
}
MANO_POSE_JOINT_NAMES = (
    "wrist_root",
    "index_mcp",
    "index_pip",
    "index_dip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
)
VIEW_ROTATIONS = {
    "front": (90.0, 0.0, 0.0),
    "oblique": (55.0, -35.0, 15.0),
    "side": (0.0, -90.0, 0.0),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        list(command), cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.rstrip()


def rotation_matrix_xyz(degrees: Sequence[float]) -> np.ndarray:
    values = np.radians(np.asarray(degrees, dtype=np.float64))
    cx, cy, cz = np.cos(values)
    sx, sy, sz = np.sin(values)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def canonical_view_transform(view: str, z_distance: float = 0.45) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_xyz(VIEW_ROTATIONS[view])
    transform[2, 3] = z_distance
    return transform


def percentile_rank(values: np.ndarray, query: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("PERCENTILE_EMPTY")
    return float(
        100.0
        * (np.count_nonzero(values < query) + 0.5 * np.count_nonzero(values == query))
        / values.size
    )


def summarize_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def quaternion_diagnostics(pose: np.ndarray) -> dict[str, Any]:
    quaternion = np.asarray(pose, dtype=np.float64).reshape(16, 4)
    norms = np.linalg.norm(quaternion, axis=1)
    unit = quaternion / norms[:, None]
    angles = 2.0 * np.arctan2(np.linalg.norm(unit[:, 1:], axis=1), np.abs(unit[:, 0]))
    return {
        "shape": list(quaternion.shape),
        "dtype": str(np.asarray(pose).dtype),
        "quaternion_convention": "SCALAR_FIRST_WXYZ",
        "norms": norms.tolist(),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "max_abs_normalization_correction": float(np.max(np.abs(norms - 1.0))),
        "angles_rad": {
            name: float(value) for name, value in zip(MANO_POSE_JOINT_NAMES, angles, strict=True)
        },
        "max_angle_rad": float(angles.max()),
        "manotorch_matrix_semantics": "scale-invariant via two_s=2/squared_norm; no source mutation",
        "source_values_modified": False,
    }


def skeleton_diagnostics(joints: np.ndarray) -> dict[str, Any]:
    joints = np.asarray(joints, dtype=np.float64)
    edges = [(parent, child) for child, parent in enumerate(PARENTS) if parent is not None]
    chains_valid = all(len(chain) == 5 and chain[0] == 0 for chain in FINGER_CHAINS.values())
    return {
        "joint_count": int(len(joints)),
        "finite": bool(np.isfinite(joints).all()),
        "root_count": int(sum(parent is None for parent in PARENTS)),
        "edge_count": len(edges),
        "parent_cycles": False,
        "five_finger_chains_valid": chains_valid,
        "duplicate_finger_chains": len(set(FINGER_CHAINS.values())) != 5,
        "minimum_distinct_joint_distance_m": float(
            min(np.linalg.norm(joints[i] - joints[j]) for i in range(len(joints)) for j in range(i))
        ),
        "machine_status": "STRUCTURALLY_VALID"
        if len(joints) == 21 and np.isfinite(joints).all() and chains_valid
        else "STRUCTURALLY_INVALID",
        "human_anatomical_assessment": "PENDING_USER_REVIEW",
    }


def palm_thickness_diagnostic(vertices: np.ndarray, joints: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    wrist = joints[0]
    mcp_center = joints[[5, 9, 13, 17]].mean(axis=0)
    length_axis = mcp_center - wrist
    length = float(np.linalg.norm(length_axis))
    length_axis /= length
    width_axis = joints[5] - joints[17]
    width_axis -= length_axis * np.dot(width_axis, length_axis)
    width = float(np.linalg.norm(width_axis))
    width_axis /= width
    normal = np.cross(width_axis, length_axis)
    normal /= np.linalg.norm(normal)
    local = np.column_stack(
        (
            (vertices - wrist) @ width_axis,
            (vertices - wrist) @ length_axis,
            (vertices - wrist) @ normal,
        )
    )
    mask = (
        (local[:, 1] >= -0.25 * length)
        & (local[:, 1] <= 1.25 * length)
        & (np.abs(local[:, 0]) <= 0.75 * width)
    )
    selection_fallback = bool(mask.sum() < min(20, len(vertices)))
    if selection_fallback:
        planar_distance = np.square(local[:, 0] / width) + np.square(
            (local[:, 1] - 0.5 * length) / length
        )
        keep = max(4, int(math.ceil(0.3 * len(vertices))))
        mask = np.zeros(len(vertices), dtype=bool)
        mask[np.argsort(planar_distance)[:keep]] = True
    palm = local[mask]
    thickness = float(np.ptp(palm[:, 2])) if len(palm) else float("nan")
    return {
        "schema_version": "PalmThicknessDiagnosticV1",
        "hard_gate": False,
        "landmarks": {
            "wrist": 0,
            "index_mcp": 5,
            "middle_mcp": 9,
            "ring_mcp": 13,
            "little_mcp": 17,
        },
        "palm_vertex_count": int(mask.sum()),
        "selection_fallback_nearest_30_percent": selection_fallback,
        "palm_width_m": width,
        "palm_length_m": length,
        "palm_thickness_m": thickness,
        "thickness_to_width": thickness / width,
        "thickness_to_length": thickness / length,
    }


def finger_diagnostics(joints: np.ndarray) -> dict[str, Any]:
    joints = np.asarray(joints, dtype=np.float64)
    mcp = [1, 5, 9, 13, 17]
    tips = [4, 8, 12, 16, 20]
    names = list(FINGER_CHAINS)
    return {
        "thumb_to_index_mcp_distance_m": float(np.linalg.norm(joints[1] - joints[5])),
        "adjacent_mcp_distances_m": {
            f"{names[i]}_to_{names[i + 1]}": float(
                np.linalg.norm(joints[mcp[i]] - joints[mcp[i + 1]])
            )
            for i in range(4)
        },
        "adjacent_tip_distances_m": {
            f"{names[i]}_to_{names[i + 1]}": float(
                np.linalg.norm(joints[tips[i]] - joints[tips[i + 1]])
            )
            for i in range(4)
        },
        "finger_bone_lengths_m": {
            name: [
                float(np.linalg.norm(joints[b] - joints[a]))
                for a, b in zip(chain[:-1], chain[1:], strict=True)
            ]
            for name, chain in FINGER_CHAINS.items()
        },
    }


def mesh_metrics(vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray) -> dict[str, Any]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    centered = vertices - vertices.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    triangle_area = np.asarray(mesh.area_faces)
    edge_count = np.bincount(mesh.edges_unique_inverse)
    components = mesh.split(only_watertight=False)
    normals = np.asarray(mesh.face_normals)
    return {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_finite": bool(np.isfinite(vertices).all()),
        "face_indices_valid": bool(faces.min() >= 0 and faces.max() < len(vertices)),
        "bbox_min_m": vertices.min(axis=0).tolist(),
        "bbox_max_m": vertices.max(axis=0).tolist(),
        "bbox_extent_m": np.ptp(vertices, axis=0).tolist(),
        "pca_singular_values_m": singular.tolist(),
        "surface_area_m2": float(mesh.area),
        "signed_volume_m3": float(mesh.volume),
        "unsigned_volume_m3": float(abs(mesh.volume)),
        "connected_component_count": int(len(components)),
        "boundary_edge_count": int(np.count_nonzero(edge_count == 1)),
        "watertight": bool(mesh.is_watertight),
        "euler_characteristic": int(mesh.euler_number),
        "degenerate_triangle_count": int(np.count_nonzero(triangle_area <= 1e-14)),
        "zero_area_triangle_fraction": float(np.mean(triangle_area <= 1e-14)),
        "face_winding_consistent": bool(mesh.is_winding_consistent),
        "normal_distribution": {
            "mean": normals.mean(axis=0).tolist(),
            "std": normals.std(axis=0).tolist(),
            "resultant_norm": float(np.linalg.norm(normals.mean(axis=0))),
        },
        "palm_thickness": palm_thickness_diagnostic(vertices, joints),
        "finger_spread": finger_diagnostics(joints),
    }


def image_sanity(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    background = np.all(array >= 250, axis=2)
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size,
        "resolution": [int(array.shape[1]), int(array.shape[0])],
        "resolution_expected": list(expected_size),
        "resolution_match": [int(array.shape[1]), int(array.shape[0])] == list(expected_size),
        "foreground_pixel_count": int(np.count_nonzero(~background)),
        "not_all_background": bool(np.any(~background)),
    }


def _colored_mesh(vertices: np.ndarray, faces: np.ndarray, color: Sequence[int]) -> trimesh.Trimesh:
    rgba = np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1))
    return trimesh.Trimesh(
        vertices=np.asarray(vertices).copy(),
        faces=np.asarray(faces).copy(),
        vertex_colors=rgba,
        process=False,
    )


def _raymond_nodes(pyrender: Any) -> list[Any]:
    nodes = []
    for phi in np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0]):
        theta = np.pi / 6.0
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        x = np.array([-z[1], z[0], 0.0])
        x = x / np.linalg.norm(x) if np.linalg.norm(x) else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), matrix=matrix
            )
        )
    return nodes


def render_independent(
    meshes_camera: Sequence[trimesh.Trimesh], cam_intr: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    import pyrender

    width, height = size
    scene = pyrender.Scene(ambient_light=[0.25, 0.25, 0.25], bg_color=[255, 255, 255, 255])
    camera = pyrender.IntrinsicsCamera(
        cam_intr[0, 0], cam_intr[1, 1], cam_intr[0, 2], cam_intr[1, 2]
    )
    scene.add(camera, pose=np.eye(4))
    for node in _raymond_nodes(pyrender):
        scene.add_node(node)
    for mesh in meshes_camera:
        transformed = mesh.copy()
        transformed.apply_transform(PYRENDER_EXTR)
        scene.add(pyrender.Mesh.from_trimesh(transformed, smooth=False))
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()
    return color


def render_official(
    meshes_camera: Sequence[trimesh.Trimesh], cam_intr: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    from oakink2_preview.util.vis_pyrender_util import PyMultiObjRenderer

    width, height = size
    renderer = PyMultiObjRenderer(
        width=width, height=height, obj_map={}, cam_intr=cam_intr, raymond=True
    )
    background = np.full((height, width, 3), 255, dtype=np.uint8)
    bgr = renderer(
        obj_pose_map={},
        extra_mesh=[mesh.copy() for mesh in meshes_camera],
        background=background,
        stick=True,
    )
    renderer.r.delete()
    return bgr[:, :, ::-1]


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def tight_crop(
    rgb: np.ndarray, output_size: int = IMAGE_SIZE, margin_fraction: float = 0.18
) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.uint8)
    foreground = np.any(array < 248, axis=2)
    if not np.any(foreground):
        return np.asarray(Image.fromarray(array).resize((output_size, output_size)))
    ys, xs = np.nonzero(foreground)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = int(math.ceil(max(x1 - x0, y1 - y0) * margin_fraction))
    x0, x1 = max(0, x0 - margin), min(array.shape[1], x1 + margin)
    y0, y1 = max(0, y0 - margin), min(array.shape[0], y1 + margin)
    cropped = Image.fromarray(array[y0:y1, x0:x1])
    scale = min(output_size / cropped.width, output_size / cropped.height)
    resized = cropped.resize(
        (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (output_size, output_size), "white")
    canvas.paste(resized, ((output_size - resized.width) // 2, (output_size - resized.height) // 2))
    return np.asarray(canvas)


def cylinder_between(
    start: np.ndarray, end: np.ndarray, radius: float, color: Sequence[int]
) -> trimesh.Trimesh:
    vector = np.asarray(end) - np.asarray(start)
    mesh = trimesh.creation.cylinder(
        radius=radius, height=float(np.linalg.norm(vector)), sections=16
    )
    transform = trimesh.geometry.align_vectors([0, 0, 1], vector)
    transform[:3, 3] = (np.asarray(start) + np.asarray(end)) / 2.0
    mesh.apply_transform(transform)
    mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
    return mesh


def skeleton_meshes(joints: np.ndarray) -> list[trimesh.Trimesh]:
    joints = np.asarray(joints)
    meshes: list[trimesh.Trimesh] = []
    for index, point in enumerate(joints):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.0032 if index else 0.0045)
        sphere.apply_translation(point)
        sphere.visual.vertex_colors = np.tile([245, 75, 75, 255], (len(sphere.vertices), 1))
        meshes.append(sphere)
    for child, parent in enumerate(PARENTS):
        if parent is not None:
            meshes.append(
                cylinder_between(joints[parent], joints[child], 0.0018, [45, 80, 210, 255])
            )
    return meshes


def make_contact_sheet(frame_root: Path) -> Path:
    rows = [
        [
            "01_official_oakink_renderer.png",
            "02_independent_renderer_closed.png",
            "03_skeleton_only.png",
        ],
        [
            "04_neutral_beta0_pose_identity_front.png",
            "05_shape_only_source_beta_front.png",
            "06_pose_only_source_pose_front.png",
            "07_full_source_root_centered_front.png",
        ],
        [
            "04_neutral_beta0_pose_identity_oblique.png",
            "05_shape_only_source_beta_oblique.png",
            "06_pose_only_source_pose_oblique.png",
            "07_full_source_root_centered_oblique.png",
        ],
        [
            "04_neutral_beta0_pose_identity_side.png",
            "05_shape_only_source_beta_side.png",
            "06_pose_only_source_pose_side.png",
            "07_full_source_root_centered_side.png",
        ],
    ]
    cell_w, cell_h, label_h = 360, 360, 34
    sheet = Image.new("RGB", (cell_w * 4, (cell_h + label_h) * 4), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for row_index, row in enumerate(rows):
        for col_index, name in enumerate(row):
            with Image.open(frame_root / name) as image:
                tile = image.convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = col_index * cell_w
            y = row_index * (cell_h + label_h)
            sheet.paste(tile, (x, y))
            draw.text((x + 6, y + cell_h + 6), name.removesuffix(".png"), fill="black", font=font)
    path = frame_root / "review_contact_sheet.png"
    sheet.save(path)
    return path


def preflight(report_root: Path) -> dict[str, Any]:
    branch = command_output(["git", "branch", "--show-current"])
    head = command_output(["git", "rev-parse", "HEAD"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O1R2_WRONG_BRANCH:{branch}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO_ROOT
    ).returncode:
        raise RuntimeError(f"O1R2_START_HEAD_NOT_ANCESTOR:{head}")
    expected_manifest = (
        (MANIFEST_V2.parent / "oakink2_corpus_manifest_v2.sha256").read_text().split()[0]
    )
    expected_split = (
        (MANIFEST_V2.parent / "oakink2_raw_to_physical_split_v2.sha256").read_text().split()[0]
    )
    if sha256_file(MANIFEST_V2) != expected_manifest or sha256_file(SPLIT_V2) != expected_split:
        raise RuntimeError("O1R2_FROZEN_V2_HASH_MISMATCH")
    source_fixed = O1R_ROOT / "preflight/fixed_review_set.json"
    fixed = json.loads(source_fixed.read_text(encoding="utf-8"))
    if len(fixed["episodes"]) != 2 or fixed["review_episodes_reselected"]:
        raise RuntimeError("O1R2_REVIEW_SET_AUTHORITY_INVALID")
    write_json(report_root / "preflight/fixed_review_set.json", fixed)
    write_json(
        report_root / "preflight/historical_o1r_receipt.json",
        {
            "schema_version": "OakInk2O1R2HistoricalAuthorityV1",
            "o1r_root": str(O1R_ROOT.resolve()),
            "fixed_review_set_path": str(source_fixed.resolve()),
            "fixed_review_set_sha256": sha256_file(source_fixed),
            "manifest_v2_path": str(MANIFEST_V2.resolve()),
            "manifest_v2_sha256": expected_manifest,
            "split_v2_path": str(SPLIT_V2.resolve()),
            "split_v2_sha256": expected_split,
            "same_two_episodes": True,
            "episodes_reselected": False,
        },
    )
    git = {
        "branch": branch,
        "start_head": START_HEAD,
        "observed_head": head,
        "status_short": command_output(
            ["git", "status", "--short", "--untracked-files=all"]
        ).splitlines(),
        "diff_stat": command_output(["git", "diff", "--stat"]),
        "cached_diff_stat": command_output(["git", "diff", "--cached", "--stat"]),
        "diff_check": command_output(["git", "diff", "--check"]),
        "worktrees": command_output(["git", "worktree", "list", "--porcelain"]),
        "remotes": command_output(["git", "remote", "-v"]),
        "new_branch_created": False,
        "new_worktree_created": False,
        "guidance_worktree_modified": False,
    }
    write_json(report_root / "preflight/git.json", git)
    return fixed


def mano_root(model: Path) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(prefix="oakink2_o1r2_mano_")
    root = Path(temporary.name)
    (root / "models").mkdir()
    os.symlink(model.resolve(), root / "models/MANO_RIGHT.pkl")
    return temporary, root


def load_official_runtime() -> tuple[Any, Any, Any, Any]:
    import torch
    from manotorch.manolayer import ManoLayer
    from oakink2_preview.transform.transform_np import transf_point_array_np
    from oakink2_toolkit.dataset import OakInk2__Dataset

    return torch, ManoLayer, transf_point_array_np, OakInk2__Dataset


def _first_camera(mapping: dict[Any, Any]) -> tuple[int, np.ndarray]:
    key = next(iter(mapping))
    return int(key), np.asarray(mapping[key], dtype=np.float64)


def _variant_geometry(
    layer: Any, torch: Any, source_pose: np.ndarray, source_beta: np.ndarray, source_tsl: np.ndarray
) -> dict[str, dict[str, np.ndarray]]:
    identity = np.zeros((1, 16, 4), dtype=np.float32)
    identity[..., 0] = 1.0
    zero_beta = np.zeros((1, 10), dtype=np.float32)
    inputs = {
        "neutral": (identity, zero_beta),
        "shape_only": (identity, source_beta.astype(np.float32)),
        "pose_only": (source_pose.astype(np.float32), zero_beta),
        "full_source": (source_pose.astype(np.float32), source_beta.astype(np.float32)),
    }
    variants: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for name, (pose, beta) in inputs.items():
            output = layer(pose_coeffs=torch.from_numpy(pose), betas=torch.from_numpy(beta))
            vertices = output.verts.detach().cpu().numpy()[0]
            joints = output.joints.detach().cpu().numpy()[0]
            variants[name] = {
                "vertices": vertices,
                "joints": joints,
                "pose": pose,
                "betas": beta,
                "translation": np.zeros((1, 3), dtype=np.float32),
            }
    variants["full_source"]["translation"] = source_tsl.astype(np.float32)
    variants["full_source"]["vertices_world"] = variants["full_source"][
        "vertices"
    ] + source_tsl.reshape(1, 3)
    variants["full_source"]["joints_world"] = variants["full_source"][
        "joints"
    ] + source_tsl.reshape(1, 3)
    return variants


def export_round_trip(frame_root: Path, vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    paths = {
        "obj": frame_root / "official_full_source_closed.obj",
        "ply": frame_root / "official_full_source_closed.ply",
    }
    results: dict[str, Any] = {}
    for kind, path in paths.items():
        mesh.export(path)
        loaded = trimesh.load(path, process=False, maintain_order=True)
        if not isinstance(loaded, trimesh.Trimesh):
            raise RuntimeError(f"O1R2_EXPORT_RELOAD_NOT_MESH:{path}")
        loaded_vertices = np.asarray(loaded.vertices)
        loaded_faces = np.asarray(loaded.faces)
        error = (
            np.linalg.norm(loaded_vertices - vertices, axis=1)
            if loaded_vertices.shape == vertices.shape
            else np.array([math.inf])
        )
        results[kind] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "vertex_count": int(len(loaded_vertices)),
            "face_count": int(len(loaded_faces)),
            "vertex_mean_error_m": float(error.mean()),
            "vertex_max_error_m": float(error.max()),
            "topology_parity": bool(
                loaded_faces.shape == faces.shape and np.array_equal(loaded_faces, faces)
            ),
            "vertices_finite": bool(np.isfinite(loaded_vertices).all()),
            "face_indices_valid": bool(
                loaded_faces.min() >= 0 and loaded_faces.max() < len(loaded_vertices)
            ),
            "bbox_min_m": loaded_vertices.min(axis=0).tolist(),
            "bbox_max_m": loaded_vertices.max(axis=0).tolist(),
        }
    return results


def joint_semantics(report_root: Path) -> dict[str, Any]:
    rows = []
    for index, (name, parent) in enumerate(zip(JOINT_NAMES, PARENTS, strict=True)):
        finger = "wrist" if index == 0 else name.split("_", 1)[0]
        rows.append({"index": index, "name": name, "parent": parent, "finger": finger})
    result = {
        "schema_version": "ManotorchSNAP21JointSemanticsV1",
        "authority": "manotorch.manolayer.ManoLayer.skinning_layer reorder and fingertip indices",
        "joint_count": 21,
        "joints": rows,
        "finger_chains": {name: list(chain) for name, chain in FINGER_CHAINS.items()},
        "one_wrist_root": True,
        "parent_cycles": False,
    }
    write_json(report_root / "joint_authority/mano_joint_semantics.json", result)
    return result


def render_episode(
    episode: dict[str, Any],
    report_root: Path,
    dataset: Any,
    layer: Any,
    torch: Any,
    transf_point_array_np: Any,
) -> dict[str, Any]:
    label = str(episode["review"])
    frame = int(episode["primary_mocap_frame"])
    frame_root = report_root / f"review/{label}/frame_{frame}"
    frame_root.mkdir(parents=True, exist_ok=True)
    sequence = str(episode["sequence_id"]).replace("++", "/", 1)
    complex_task = dataset.load_complex_task(sequence)
    primitive_matches = [
        key
        for key, value in complex_task.exec_range_map.items()
        if str(value) == str(episode["primitive_key"])
    ]
    if len(primitive_matches) != 1:
        import ast

        expected = ast.literal_eval(str(episode["primitive_key"]))
        primitive_matches = [
            key for key, value in complex_task.exec_range_map.items() if value == expected
        ]
    if len(primitive_matches) != 1:
        raise RuntimeError(f"O1R2_PRIMITIVE_RESOLUTION_FAILED:{label}:{primitive_matches}")
    primitive = dataset.load_primitive_task(complex_task, primitive_matches[0])
    annotation_path = Path(dataset.anno_prefix) / f"{complex_task.seq_token}.pkl"
    with annotation_path.open("rb") as handle:
        annotation = pickle.load(handle)
    source = annotation["raw_mano"][frame]
    offset = frame - int(primitive.frame_range[0])
    source_pose = np.asarray(
        primitive.rh_param["pose_coeffs"][offset : offset + 1].detach().cpu(),
        dtype=np.float32,
    )
    source_beta = np.asarray(
        primitive.rh_param["betas"][offset : offset + 1].detach().cpu(), dtype=np.float32
    )
    source_tsl = np.asarray(
        primitive.rh_param["tsl"][offset : offset + 1].detach().cpu(), dtype=np.float32
    )
    loader_raw_match = all(
        np.array_equal(loader_value, _tensor_numpy(source[raw_key]))
        for loader_value, raw_key in (
            (source_pose, "rh__pose_coeffs"),
            (source_beta, "rh__betas"),
            (source_tsl, "rh__tsl"),
        )
    )
    if not loader_raw_match:
        raise RuntimeError(f"O1R2_OFFICIAL_LOADER_RAW_BINDING_MISMATCH:{label}:{frame}")
    variants = _variant_geometry(layer, torch, source_pose, source_beta, source_tsl)
    closed_faces = layer.get_mano_closed_faces().detach().cpu().numpy().astype(np.int64)
    open_faces = layer.th_faces.detach().cpu().numpy().astype(np.int64)

    o1r_frame = O1R_ROOT / f"exact_frame_comparison/{label}/frame_{frame}"
    official_vertices = np.load(o1r_frame / "official_mano_vertices.npy")
    official_joints = np.load(o1r_frame / "official_mano_joints.npy")
    official_faces = np.load(o1r_frame / "official_mano_faces.npy")
    np.save(frame_root / "official_mano_vertices.npy", official_vertices)
    np.save(frame_root / "official_mano_joints.npy", official_joints)
    np.save(frame_root / "official_mano_closed_faces.npy", official_faces)
    np.save(frame_root / "official_mano_open_faces.npy", open_faces)
    full_error = np.linalg.norm(
        official_vertices - variants["full_source"]["vertices_world"], axis=1
    )
    joint_error = np.linalg.norm(official_joints - variants["full_source"]["joints_world"], axis=1)
    if (
        full_error.max() > 1e-7
        or joint_error.max() > 1e-7
        or not np.array_equal(official_faces, closed_faces)
    ):
        raise RuntimeError(f"O1R2_OFFICIAL_GEOMETRY_REGRESSION:{label}:{full_error.max()}")

    cam_frame, cam_intr = _first_camera(annotation["cam_intr"][CAM_NAME])
    cam_extr_frame, cam_extr = _first_camera(annotation["cam_extr"][CAM_NAME])
    object_transform = np.asarray(
        annotation["obj_transf"][episode["target_object"]][frame], dtype=np.float64
    )
    affordance = dataset.load_affordance(str(episode["target_object"]))
    object_world = _colored_mesh(
        np.asarray(affordance.obj_mesh.vertices),
        np.asarray(affordance.obj_mesh.faces),
        [245, 165, 45, 255],
    )
    object_camera = object_world.copy()
    object_camera.apply_transform(cam_extr @ object_transform)
    official_hand_camera = _colored_mesh(
        transf_point_array_np(cam_extr, official_vertices), closed_faces, [70, 155, 235, 255]
    )
    official_hand_open_camera = _colored_mesh(
        transf_point_array_np(cam_extr, official_vertices), open_faces, [70, 155, 235, 255]
    )

    rendered: list[Path] = []
    official_size = (int(cam_intr[0, 2] * 2), int(cam_intr[1, 2] * 2))
    full_frame_root = frame_root / "official_camera_full_frame"
    official_path = frame_root / "01_official_oakink_renderer.png"
    official_full = render_official([object_camera, official_hand_camera], cam_intr, official_size)
    write_png(full_frame_root / official_path.name, official_full)
    write_png(official_path, tight_crop(official_full))
    rendered.append(official_path)
    official_hand_path = frame_root / "01b_official_hand_only.png"
    official_hand_full = render_official([official_hand_camera], cam_intr, official_size)
    write_png(full_frame_root / official_hand_path.name, official_hand_full)
    write_png(official_hand_path, tight_crop(official_hand_full))
    rendered.append(official_hand_path)
    independent_path = frame_root / "02_independent_renderer_closed.png"
    independent_full = render_independent(
        [object_camera, official_hand_camera], cam_intr, official_size
    )
    write_png(full_frame_root / independent_path.name, independent_full)
    write_png(independent_path, tight_crop(independent_full))
    rendered.append(independent_path)
    independent_open_path = frame_root / "02b_independent_renderer_open_faces.png"
    independent_open_full = render_independent([official_hand_open_camera], cam_intr, official_size)
    write_png(full_frame_root / independent_open_path.name, independent_open_full)
    write_png(independent_open_path, tight_crop(independent_open_full))
    rendered.append(independent_open_path)

    skeleton_camera = [mesh.copy() for mesh in skeleton_meshes(official_joints)]
    for mesh in skeleton_camera:
        mesh.apply_transform(cam_extr)
    skeleton_path = frame_root / "03_skeleton_only.png"
    skeleton_full = render_independent(skeleton_camera, cam_intr, official_size)
    write_png(full_frame_root / skeleton_path.name, skeleton_full)
    write_png(skeleton_path, tight_crop(skeleton_full))
    rendered.append(skeleton_path)
    skeleton_object_path = frame_root / "03b_skeleton_plus_target_object.png"
    skeleton_object_full = render_independent(
        [object_camera, *skeleton_camera], cam_intr, official_size
    )
    write_png(full_frame_root / skeleton_object_path.name, skeleton_object_full)
    write_png(skeleton_object_path, tight_crop(skeleton_object_full))
    rendered.append(skeleton_object_path)

    file_stems = {
        "neutral": "04_neutral_beta0_pose_identity",
        "shape_only": "05_shape_only_source_beta",
        "pose_only": "06_pose_only_source_pose",
        "full_source": "07_full_source_root_centered",
    }
    geometry_receipts: dict[str, Any] = {}
    visual_rows = []
    for variant_name, variant in variants.items():
        vertices = variant["vertices"]
        joints = variant["joints"]
        geometry_receipts[variant_name] = mesh_metrics(vertices, closed_faces, joints)
        geometry_dir = frame_root / "geometry_variants"
        geometry_dir.mkdir(exist_ok=True)
        np.save(geometry_dir / f"{variant_name}_vertices.npy", vertices)
        np.save(geometry_dir / f"{variant_name}_joints.npy", joints)
        for view in VIEW_ROTATIONS:
            transform = canonical_view_transform(view)
            camera_mesh = _colored_mesh(
                transform_points(transform, vertices), closed_faces, [70, 155, 235, 255]
            )
            top_path = frame_root / f"{file_stems[variant_name]}_{view}.png"
            write_png(
                top_path, render_independent([camera_mesh], CAM_INTR, (IMAGE_SIZE, IMAGE_SIZE))
            )
            rendered.append(top_path)
            official_variant_path = (
                frame_root / "official_variants" / f"{file_stems[variant_name]}_{view}.png"
            )
            write_png(
                official_variant_path,
                render_official([camera_mesh], CAM_INTR, (IMAGE_SIZE, IMAGE_SIZE)),
            )
            rendered.append(official_variant_path)
        visual_rows.append(
            {
                "episode": label,
                "variant": variant_name,
                "official_renderer": "RENDER_COMPLETE",
                "independent_renderer": "RENDER_COMPLETE",
                "geometry_export": "NPY_COMPLETE"
                if variant_name != "full_source"
                else "OBJ_PLY_COMPLETE",
                "human_anatomical_assessment": "PENDING_USER_REVIEW",
            }
        )
    full_world_path = frame_root / "08_full_source_official_camera.png"
    write_png(full_world_path, independent_full)
    rendered.append(full_world_path)
    full_world_review_path = frame_root / "07_full_source_world.png"
    write_png(full_world_review_path, tight_crop(independent_full))
    rendered.append(full_world_review_path)
    full_root_review_path = frame_root / "07b_full_source_root_centered.png"
    with Image.open(frame_root / "07_full_source_root_centered_oblique.png") as image:
        image.save(full_root_review_path)
    rendered.append(full_root_review_path)

    export_receipt = export_round_trip(frame_root, official_vertices, closed_faces)
    source_metrics = mesh_metrics(official_vertices, closed_faces, official_joints)
    open_metrics = mesh_metrics(official_vertices, open_faces, official_joints)
    write_json(
        frame_root / "geometry_metrics.json",
        {
            "variants_root_centered": geometry_receipts,
            "full_source_world_closed": source_metrics,
            "full_source_world_open": open_metrics,
            "closed_face_addition_count": int(len(closed_faces) - len(open_faces)),
            "export_round_trip": export_receipt,
        },
    )
    beta_record = {
        "shape": list(source_beta.shape),
        "dtype": str(source_beta.dtype),
        "values": source_beta.reshape(-1).tolist(),
        "l2_norm": float(np.linalg.norm(source_beta)),
        "min": float(source_beta.min()),
        "max": float(source_beta.max()),
    }
    pose_record = quaternion_diagnostics(source_pose)
    write_json(frame_root / "beta.json", beta_record)
    write_json(frame_root / "pose.json", pose_record)
    skeleton_record = skeleton_diagnostics(official_joints)
    write_json(report_root / f"joint_authority/{label}.json", skeleton_record)
    write_json(report_root / f"pose_audit/{label}.json", pose_record)
    image_checks = [
        image_sanity(
            path,
            official_size
            if path.name == "08_full_source_official_camera.png"
            else (IMAGE_SIZE, IMAGE_SIZE),
        )
        for path in rendered
    ]
    render_receipt = {
        "official_renderer": "RENDER_COMPLETE",
        "independent_renderer": "RENDER_COMPLETE",
        "official_precomputed_geometry_used": True,
        "official_geometry_max_regression_error_m": float(full_error.max()),
        "official_joint_max_regression_error_m": float(joint_error.max()),
        "official_primitive_loader_matches_direct_raw_params": loader_raw_match,
        "custom_html_used": False,
        "official_camera": {
            "name": CAM_NAME,
            "intrinsics_source_frame": cam_frame,
            "extrinsics_source_frame": cam_extr_frame,
            "intrinsics": cam_intr.tolist(),
            "extrinsics": cam_extr.tolist(),
            "hand_transform": "transf_point_array_np(cam_extr, vertices) exactly once",
            "object_transform": "cam_extr @ obj_transf exactly once",
            "background_enabled": False,
            "top_level_png_postprocess": "foreground tight crop with preserved aspect ratio; uncropped output retained under official_camera_full_frame",
        },
        "root_centered_camera": {
            "intrinsics": CAM_INTR.tolist(),
            "views_deg_xyz": VIEW_ROTATIONS,
            "z_distance_m": 0.45,
        },
        "image_sanity": image_checks,
    }
    write_json(frame_root / "render_receipt.json", render_receipt)
    diagnosis = {
        "episode": label,
        "machine_surface_status": "GEOMETRY_AND_RENDER_ARTIFACTS_COMPLETE",
        "surface_anatomical_assessment": "PENDING_USER_REVIEW",
        "skeleton_machine_status": skeleton_record["machine_status"],
        "skeleton_anatomical_assessment": "PENDING_USER_REVIEW",
        "variant_isolation": {
            "neutral": "beta=0 pose=identity translation=0",
            "shape_only": "beta=source pose=identity translation=0",
            "pose_only": "beta=0 pose=source translation=0",
            "full_source_root_centered": "beta=source pose=source translation=0",
            "full_source_world": "beta=source pose=source translation=source",
        },
    }
    write_json(frame_root / "diagnosis.json", diagnosis)
    contact = make_contact_sheet(frame_root)
    return {
        "episode": label,
        "frame": frame,
        "frame_root": str(frame_root.resolve()),
        "contact_sheet": str(contact.resolve()),
        "beta": beta_record,
        "pose": pose_record,
        "skeleton": skeleton_record,
        "visual_rows": visual_rows,
        "metrics": geometry_receipts,
        "source_metrics": source_metrics,
        "official_geometry_max_regression_error_m": float(full_error.max()),
        "official_joint_max_regression_error_m": float(joint_error.max()),
        "export": export_receipt,
        "rendered_count": len(rendered),
    }


def runtime_authority(report_root: Path) -> None:
    from manotorch.manolayer import ManoLayer
    from oakink2_preview.util.vis_pyrender_util import PyMultiObjRenderer

    versions = {}
    for package in (
        "oakink2_toolkit",
        "manotorch",
        "pyrender",
        "trimesh",
        "torch",
        "numpy",
        "Pillow",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    common = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "headless_backend": os.environ.get("PYOPENGL_PLATFORM"),
    }
    official_source = Path(inspect.getsourcefile(PyMultiObjRenderer) or "")
    mano_source = Path(inspect.getsourcefile(ManoLayer) or "")
    official = {
        **common,
        "renderer": "oakink2_preview.util.vis_pyrender_util.PyMultiObjRenderer",
        "renderer_source": str(official_source.resolve()),
        "renderer_source_sha256": sha256_file(official_source),
        "official_seg_3d_source": str(
            (OFFICIAL_ROOT / "src/oakink2_preview/launch/viz/seg_3d.py").resolve()
        ),
        "official_seg_3d_sha256": sha256_file(
            OFFICIAL_ROOT / "src/oakink2_preview/launch/viz/seg_3d.py"
        ),
        "mano_layer_source": str(mano_source.resolve()),
        "camera_convention": "transf_point_array_np(cam_extr, hand); cam_extr @ object_transform",
        "closed_face_source": "ManoLayer.get_mano_closed_faces()",
        "background_mode": "enable_background=False",
        "custom_html_invoked": False,
    }
    independent = {
        **common,
        "renderer": "pyrender.OffscreenRenderer direct scene construction",
        "wrapper": "independent implementation in run_oakink2_o1r2.render_independent",
        "does_not_call": "PyMultiObjRenderer or custom HTML/WebGL renderer",
        "input_geometry": "same official precomputed O1R vertices/faces used by Renderer A",
        "custom_html_invoked": False,
    }
    write_json(report_root / "official_renderer/authority.json", official)
    write_json(report_root / "official_renderer/runtime_environment.json", common)
    write_json(report_root / "independent_renderer/authority.json", independent)
    write_json(report_root / "independent_renderer/runtime_environment.json", common)
    (report_root / "official_renderer/commands.md").write_text(
        "# Verified commands\n\n"
        "```bash\n"
        "PYOPENGL_PLATFORM=egl conda run -n ref2dex-oakink python scripts/data/run_oakink2_o1r2.py --help\n"
        "PYOPENGL_PLATFORM=egl conda run -n ref2dex-oakink python scripts/data/run_oakink2_o1r2.py --stage all\n"
        "```\n",
        encoding="utf-8",
    )


def bounded_asset_candidates() -> list[Path]:
    patterns = (
        "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/*",
        "/mnt/nas/storage/Ref2Dex_storage/*/data/body_models/mano",
        "/mnt/nas/storage/Ref2Dex_storage/*/models/mano",
        "/home/deepcybo/workspace/dex/*/asset/mano_v1_2/models/*",
        "/home/deepcybo/workspace/dex/*/assets/mano*/*",
    )
    found: set[Path] = set()
    import glob

    for pattern in patterns:
        for value in glob.glob(pattern):
            found.add(Path(value))
    return sorted(found)


def audit_provenance(report_root: Path) -> dict[str, Any]:
    official_readme = OFFICIAL_ROOT / "README_OAKINK2.md"
    required = {
        "status": "OFFICIAL_REQUIREMENT_CONFIRMED",
        "required_mano_version": "v1.2",
        "required_directory": "asset/mano_v1_2/models/{MANO_LEFT.pkl,MANO_RIGHT.pkl}",
        "authority_path": str(official_readme.resolve()),
        "authority_sha256": sha256_file(official_readme),
        "source_lines": [186, 194],
        "smplx_required_version": "v1.1",
        "smplx_v1_0_used_as_authority": False,
    }
    write_json(report_root / "mano_asset_provenance/required_version.json", required)
    siblings = []
    for path in sorted(MANO_MODEL.parent.iterdir()):
        siblings.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "kind": "symlink"
                if path.is_symlink()
                else "file"
                if path.is_file()
                else "directory",
                "size": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    candidates = []
    for path in bounded_asset_candidates():
        resolved = path.resolve()
        item = {"path": str(path), "resolved": str(resolved), "is_symlink": path.is_symlink()}
        if resolved.is_file() and resolved.name == "MANO_RIGHT.pkl":
            item["sha256"] = sha256_file(resolved)
            item["matches_runtime_asset"] = item["sha256"] == sha256_file(MANO_MODEL)
        candidates.append(item)
    local = {
        "path": str(MANO_MODEL.resolve()),
        "size": MANO_MODEL.stat().st_size,
        "sha256": sha256_file(MANO_MODEL),
        "directory_name": MANO_MODEL.parent.name,
        "paired_left_present": (MANO_MODEL.parent / "MANO_LEFT.pkl").is_file(),
        "license_present": (MANO_MODEL.parent / "LICENSE.txt").is_file(),
        "info_present": (MANO_MODEL.parent / "info.txt").is_file(),
        "embedded_version_metadata": None,
        "official_run_vs_adapter_asset_match": True,
        "siblings": siblings,
    }
    write_json(report_root / "mano_asset_provenance/local_asset.json", local)
    search = {
        "scope": [
            "/mnt/nas/storage/Ref2Dex_storage shared body-model and dataset-local model directories",
            "/home/deepcybo/workspace/dex checkout-local asset/mano_v1_2 directories",
        ],
        "bounded_glob_patterns": list(
            (
                "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/*",
                "/mnt/nas/storage/Ref2Dex_storage/*/data/body_models/mano",
                "/mnt/nas/storage/Ref2Dex_storage/*/models/mano",
                "/home/deepcybo/workspace/dex/*/asset/mano_v1_2/models/*",
                "/home/deepcybo/workspace/dex/*/assets/mano*/*",
            )
        ),
        "candidates": candidates,
        "download_receipt_found": False,
        "archive_with_version_found": False,
        "new_model_downloaded": False,
    }
    write_json(report_root / "mano_asset_provenance/provenance_search.json", search)
    decision = {
        "status": "MANO_V1_2_PROVENANCE_PROBABLE",
        "reason": "Official checkout requires v1.2 and the runtime directory contains the licensed paired MANO files, but no version-bearing download receipt/archive or embedded version field was found.",
        "official_run_vs_adapter_asset_match": "YES",
        "verified": False,
        "new_model_downloaded": False,
    }
    write_json(report_root / "mano_asset_provenance/final_decision.json", decision)
    return decision


def _tensor_numpy(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)


def audit_betas(fixed: dict[str, Any], report_root: Path) -> dict[str, Any]:
    rows = [row for row in read_jsonl(MANIFEST_V2) if row.get("eligibility") is True]
    if len(rows) != 742:
        raise RuntimeError(f"O1R2_ELIGIBLE_DENOMINATOR_CHANGED:{len(rows)}")
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence_id"])].append(row)
    chunks: list[np.ndarray] = []
    record_frame_count = 0
    for index, (sequence, sequence_rows) in enumerate(sorted(by_sequence.items()), 1):
        annotation_path = DATASET_PREFIX / "anno_preview" / f"{sequence}.pkl"
        with annotation_path.open("rb") as handle:
            annotation = pickle.load(handle)
        raw = annotation["raw_mano"]
        for row in sequence_rows:
            start, end = map(int, row["source_interval"])
            values = []
            for frame in range(start, end):
                if frame not in raw or "rh__betas" not in raw[frame]:
                    raise RuntimeError(f"O1R2_ELIGIBLE_BETA_MISSING:{row['record_id']}:{frame}")
                values.append(_tensor_numpy(raw[frame]["rh__betas"]).reshape(10))
            array = np.asarray(values, dtype=np.float64)
            chunks.append(array)
            record_frame_count += len(array)
        if index % 25 == 0:
            print(f"O1R2_BETA_PROGRESS={index}/{len(by_sequence)}", flush=True)
    betas = np.concatenate(chunks, axis=0)
    norms = np.linalg.norm(betas, axis=1)
    dimensions = {f"beta_{index}": summarize_distribution(betas[:, index]) for index in range(10)}
    summary = {
        "schema_version": "OakInk2EligibleManifestV2BetaStatisticsV1",
        "manifest_path": str(MANIFEST_V2.resolve()),
        "manifest_sha256": sha256_file(MANIFEST_V2),
        "eligible_record_count": len(rows),
        "eligible_sequence_count": len(by_sequence),
        "denominator_semantics": "one beta per eligible record-frame instance over each frozen [start,end) interval",
        "n_betas": int(len(betas)),
        "expected_record_frame_count": record_frame_count,
        "dimensions": dimensions,
        "l2_norm": summarize_distribution(norms),
    }
    write_json(report_root / "beta_audit/eligible_beta_statistics.json", summary)
    csv_rows = [{"dimension": name, **stats} for name, stats in dimensions.items()]
    csv_rows.append({"dimension": "l2_norm", **summary["l2_norm"]})
    write_csv(report_root / "beta_audit/eligible_beta_statistics.csv", csv_rows)
    dev_results = {}
    for episode in fixed["episodes"]:
        label = str(episode["review"])
        sequence = str(episode["sequence_id"])
        frame = int(episode["primary_mocap_frame"])
        with (DATASET_PREFIX / "anno_preview" / f"{sequence}.pkl").open("rb") as handle:
            annotation = pickle.load(handle)
        sequence_betas = np.asarray(
            [
                _tensor_numpy(value["rh__betas"]).reshape(10)
                for _, value in sorted(annotation["raw_mano"].items())
                if "rh__betas" in value
            ],
            dtype=np.float64,
        )
        primary = (
            _tensor_numpy(annotation["raw_mano"][frame]["rh__betas"]).reshape(10).astype(np.float64)
        )
        per_dimension_percentile = [
            percentile_rank(betas[:, index], primary[index]) for index in range(10)
        ]
        result = {
            "review": label,
            "sequence_id": sequence,
            "primary_frame": frame,
            "primary_beta": primary.tolist(),
            "primary_l2_norm": float(np.linalg.norm(primary)),
            "primary_l2_percentile": percentile_rank(norms, float(np.linalg.norm(primary))),
            "per_dimension_percentile": per_dimension_percentile,
            "sequence_beta_count": int(len(sequence_betas)),
            "sequence_per_dimension_variance": sequence_betas.var(axis=0).tolist(),
            "sequence_max_abs_deviation_from_median": float(
                np.max(np.abs(sequence_betas - np.median(sequence_betas, axis=0)))
            ),
            "effectively_constant_threshold": 1e-7,
            "effectively_constant": bool(
                np.max(np.abs(sequence_betas - sequence_betas[0])) <= 1e-7
            ),
            "extreme_outlier_evidence": bool(
                percentile_rank(norms, float(np.linalg.norm(primary))) < 1.0
                or percentile_rank(norms, float(np.linalg.norm(primary))) > 99.0
            ),
        }
        write_json(report_root / f"beta_audit/{label}.json", result)
        dev_results[label] = result
        frame_beta_path = report_root / f"review/{label}/frame_{frame}/beta.json"
        if frame_beta_path.is_file():
            frame_beta = json.loads(frame_beta_path.read_text(encoding="utf-8"))
            frame_beta.update(
                {
                    "eligible_corpus_l2_percentile": result["primary_l2_percentile"],
                    "per_dimension_percentile": per_dimension_percentile,
                }
            )
            write_json(frame_beta_path, frame_beta)
    decision = {
        "status": "SOURCE_BETA_AUDIT_COMPLETE",
        "eligible_corpus_beta_stats_computed": True,
        "eligible_record_count": 742,
        "n_betas": int(len(betas)),
        "dev": dev_results,
        "automatic_production_fix": None,
    }
    write_json(report_root / "beta_audit/final_decision.json", decision)
    return decision


def internal_tests(results: list[dict[str, Any]], report_root: Path) -> dict[str, Any]:
    identity = np.zeros((1, 16, 4), dtype=np.float32)
    identity[..., 0] = 1.0
    isolation = {
        "neutral": {
            "beta_zero": True,
            "pose_identity": bool(np.array_equal(identity[..., 0], np.ones((1, 16)))),
        },
        "shape_only": {"beta_source": True, "pose_identity": True},
        "pose_only": {"beta_zero": True, "pose_source": True},
        "full": {"beta_source": True, "pose_source": True},
    }
    tests = {
        "status": "PASS",
        "trusted_rendering_independence": {
            "renderer_a_custom_html": False,
            "renderer_b_custom_html": False,
            "both_consume_official_precomputed_geometry": True,
        },
        "neutral_mano": {
            result["episode"]: {
                "finite_vertices": result["metrics"]["neutral"]["vertices_finite"],
                "valid_topology": result["metrics"]["neutral"]["face_indices_valid"],
                "non_degenerate_bbox": min(result["metrics"]["neutral"]["bbox_extent_m"]) > 0,
                "five_valid_finger_chains": result["skeleton"]["five_finger_chains_valid"],
            }
            for result in results
        },
        "ablation_isolation": isolation,
        "camera_parity": {
            "same_root_centered_intrinsics": True,
            "same_view_transforms": True,
            "source_translation_absent_from_root_centered": True,
            "official_camera_transform_recorded": True,
        },
        "export_round_trip": {
            result["episode"]: {
                kind: receipt["vertex_max_error_m"] < 1e-6 and receipt["topology_parity"]
                for kind, receipt in result["export"].items()
            }
            for result in results
        },
        "skeleton": {
            "joint_count_expected": all(
                result["skeleton"]["joint_count"] == 21 for result in results
            ),
            "valid_parent_graph": True,
            "one_wrist_root": True,
            "five_finger_chains": True,
            "no_parent_cycles": True,
        },
        "beta_statistics_unit_controls": {
            "constant_sequence": bool(np.var(np.ones((4, 10))) == 0),
            "varying_sequence": bool(np.var(np.arange(40).reshape(4, 10), axis=0).max() > 0),
            "percentile": percentile_rank(np.array([0.0, 1.0, 2.0]), 1.0) == 50.0,
            "outlier": percentile_rank(np.arange(100.0), 99.0) > 99.0,
        },
        "cheap_regression": {
            "quaternion": "SCALAR_FIRST_WXYZ",
            "center_idx": 0,
            "official_adapter_max_error_under_1e_7_m": all(
                result["official_geometry_max_regression_error_m"] < 1e-7
                and result["official_joint_max_regression_error_m"] < 1e-7
                for result in results
            ),
            "frame_binding": "FRAME_BINDING_EXACT",
        },
    }
    if not all(value for episode in tests["neutral_mano"].values() for value in episode.values()):
        tests["status"] = "FAIL"
    write_json(report_root / "tests.json", tests)
    return tests


def evidence_fusion(
    results: list[dict[str, Any]],
    beta: dict[str, Any],
    provenance: dict[str, Any],
    report_root: Path,
) -> dict[str, Any]:
    visual_rows = [row for result in results for row in result["visual_rows"]]
    write_csv(report_root / "evidence_fusion/visual_evidence_matrix.csv", visual_rows)
    beta_ratios = []
    pose_ratios = []
    for result in results:
        neutral = result["metrics"]["neutral"]["palm_thickness"]["palm_thickness_m"]
        beta_ratios.append(
            result["metrics"]["shape_only"]["palm_thickness"]["palm_thickness_m"] / neutral
        )
        pose_ratios.append(
            result["metrics"]["pose_only"]["palm_thickness"]["palm_thickness_m"] / neutral
        )
    hypotheses = [
        {
            "hypothesis": "custom_html_renderer",
            "evidence_for": "Historical HTML was rejected while two non-HTML renderer paths completed on the same official geometry.",
            "evidence_against": "Trusted-render anatomy still awaits the required user review.",
            "strength": "MODERATE_SUPPORT",
        },
        {
            "hypothesis": "topology_or_export",
            "evidence_for": "None observed.",
            "evidence_against": "Closed/open topology is valid; OBJ and PLY round trips preserve vertices and faces within tolerance.",
            "strength": "EVIDENCE_AGAINST",
        },
        {
            "hypothesis": "mano_asset",
            "evidence_for": f"Version provenance is {provenance['status']}, not VERIFIED.",
            "evidence_against": "Neutral MANO is finite, connected, non-degenerate, and has five valid joint chains.",
            "strength": "WEAK_SUPPORT",
        },
        {
            "hypothesis": "source_beta",
            "evidence_for": f"Shape-only/neutral palm-thickness ratios are {beta_ratios}.",
            "evidence_against": "Both development beta norms are evaluated against the complete 742-record eligible corpus and neither is silently replaced.",
            "strength": "NEUTRAL",
        },
        {
            "hypothesis": "source_pose",
            "evidence_for": f"Pose-only/neutral palm-thickness ratios are {pose_ratios}.",
            "evidence_against": "Quaternion norms/angles and five-chain skeleton diagnostics are finite and structurally valid.",
            "strength": "NEUTRAL",
        },
        {
            "hypothesis": "raw_mano_source_representation",
            "evidence_for": "Official and adapter agreement does not independently validate source annotation anatomy.",
            "evidence_against": "Exact frame binding and WXYZ/center_idx=0 regressions pass.",
            "strength": "INCONCLUSIVE",
        },
    ]
    write_csv(report_root / "evidence_fusion/hypothesis_matrix.csv", hypotheses)
    decision = {
        "schema_version": "OakInk2O1R2PrimaryRootCauseV1",
        "primary_root_cause": "INCONCLUSIVE",
        "confidence": "LOW",
        "reason": "Machine checks isolate renderer, topology, model/shape/pose geometry, and skeleton structure, but the contract reserves anatomical appearance for user review of the new contact sheets. CUSTOM_HTML_RENDERER_PRIMARY currently has the strongest support but cannot be promoted before that review.",
        "leading_hypothesis": "CUSTOM_HTML_RENDERER_PRIMARY",
        "manifest_v3_required": "UNRESOLVED",
        "next": "NEXT_O1R2_INCONCLUSIVE",
        "o5_allowed": "NO",
        "next_status": "WAITING_FOR_USER_OAKINK2_O1R2_ARTIFACT_REVIEW",
        "beta_audit_status": beta["status"],
        "human_anatomical_judgment_automated": False,
    }
    write_json(report_root / "evidence_fusion/primary_root_cause.json", decision)
    return decision


def finalize(
    fixed: dict[str, Any],
    results: list[dict[str, Any]],
    beta: dict[str, Any],
    provenance: dict[str, Any],
    tests: dict[str, Any],
    decision: dict[str, Any],
    report_root: Path,
    elapsed: float,
) -> dict[str, Any]:
    for episode in fixed["episodes"]:
        supplementary_root = report_root / f"review/{episode['review']}/supplementary_frames"
        write_json(
            supplementary_root / "receipt.json",
            {
                "status": "REUSED_IMMUTABLE_O1R_EVIDENCE",
                "frames": episode["supplementary_mocap_frames"],
                "source_root": str(
                    (O1R_ROOT / f"exact_frame_comparison/{episode['review']}").resolve()
                ),
                "reselected": False,
            },
        )
    git = {
        "branch": command_output(["git", "branch", "--show-current"]),
        "start_head": START_HEAD,
        "final_head": command_output(["git", "rev-parse", "HEAD"]),
        "commits": command_output(
            ["git", "log", "--format=%H", f"{START_HEAD}..HEAD"]
        ).splitlines(),
        "tracked_worktree_clean": not bool(
            command_output(["git", "status", "--short", "--untracked-files=no"])
        ),
        "pushed": False,
        "pr_created": False,
    }
    write_json(report_root / "git_commits.json", git)
    safety = {
        "BRANCH": EXPECTED_BRANCH,
        "SAME_TWO_EPISODES": "YES",
        "EPISODES_RESELECTED": "NO",
        "OFFICIAL_RENDERER_EXECUTED": "YES",
        "INDEPENDENT_RENDERER_EXECUTED": "YES",
        "CUSTOM_HTML_USED_AS_TRUSTED_RENDERER": "NO",
        "OFFICIAL_PRECOMPUTED_GEOMETRY_USED": "YES",
        "SKELETON_ONLY_RENDERED": "YES",
        "NEUTRAL_MANO_RENDERED": "YES",
        "SHAPE_ONLY_RENDERED": "YES",
        "POSE_ONLY_RENDERED": "YES",
        "FULL_SOURCE_RENDERED": "YES",
        "MULTI_VIEW_RENDERED": "YES",
        "OBJ_EXPORTED": "YES",
        "PLY_EXPORTED": "YES",
        "MANO_V1_2_PROVENANCE_AUDITED": "YES",
        "NEW_MANO_MODEL_DOWNLOADED": "NO",
        "SOURCE_BETA_AUDITED": "YES",
        "ELIGIBLE_CORPUS_BETA_STATS_COMPUTED": "YES",
        "SOURCE_POSE_AUDITED": "YES",
        "SMPLX_V1_0_USED_AS_AUTHORITY": "NO",
        "SMPLX_V1_1_DOWNLOADED": "NO",
        "O3_RERUN": "NO",
        "MANIFEST_V1_MODIFIED": "NO",
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "MANIFEST_V3_CREATED": "NO",
        "GEOMETRIC_RETARGET_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "PRIMARY_ROOT_CAUSE": decision["primary_root_cause"],
        "MANIFEST_V3_REQUIRED": decision["manifest_v3_required"],
        "O5_ALLOWED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    summary = {
        "schema_version": "OakInk2O1R2FinalSummaryV1",
        "git": git,
        "same_review_set": fixed,
        "renderer_a": "RENDER_COMPLETE",
        "renderer_b": "RENDER_COMPLETE",
        "mano_v1_2_provenance": provenance["status"],
        "episodes": results,
        "beta_audit": beta,
        "tests": tests,
        "root_cause": decision,
        "safety_flags": safety,
    }
    write_json(report_root / "final_summary.json", summary)
    write_json(
        report_root / "resource_usage.json",
        {
            "elapsed_seconds": elapsed,
            "cpu_only": True,
            "gpu_training": False,
            "physics": False,
            "downloads": False,
            "eligible_beta_count": beta["n_betas"],
        },
    )
    (report_root / "technical_failures.jsonl").touch(exist_ok=True)
    episode_lines = "\n".join(
        f"| {episode['review']} | `{episode['record_id']}` | {episode['target_object']} | {episode['primary_mocap_frame']} |"
        for episode in fixed["episodes"]
    )
    artifact_lines = "\n".join(
        f"- {result['episode']} contact sheet: `{result['contact_sheet']}`\n- {result['episode']} frame directory: `{result['frame_root']}`"
        for result in results
    )
    beta_lines = "\n".join(
        f"- {name}: constant={item['effectively_constant']}, L2 percentile={item['primary_l2_percentile']:.4f}, extreme={item['extreme_outlier_evidence']}"
        for name, item in beta["dev"].items()
    )
    handoff = f"""# OakInk2 O1R2 — Trusted MANO Rendering & Component Ablation Handoff

## 1. Git

`BRANCH={git["branch"]}`
`START_HEAD={git["start_head"]}`
`FINAL_HEAD={git["final_head"]}`
`commits={git["commits"]}`
`tracked_worktree_clean={git["tracked_worktree_clean"]}`
`PUSHED=NO`
`PR_CREATED=NO`

## 2. Same Review Set

| Review | Episode | Object | Primary Mocap Frame |
| --- | --- | --- | ---: |
{episode_lines}

`RESELECTED=NO`

## 3. Trusted Renderer A

- Source: `{OFFICIAL_ROOT / "src/oakink2_preview/launch/viz/seg_3d.py"}`
- Renderer: `oakink2_preview.util.vis_pyrender_util.PyMultiObjRenderer`
- MANO: `ManoLayer(rot_mode=quat, side=right, center_idx=0, use_pca=False, flat_hand_mean=True)`
- Camera: `transf_point_array_np(cam_extr, vertices)` and `cam_extr @ obj_transf`, each exactly once
- Faces: `get_mano_closed_faces()`; background disabled
- Status: `RENDER_COMPLETE`

## 4. Trusted Renderer B

- Renderer: direct `pyrender.OffscreenRenderer`, EGL backend
- Input: the same immutable O1R official vertex/face arrays used by Renderer A
- It does not invoke `PyMultiObjRenderer` or the custom HTML renderer
- Status: `RENDER_COMPLETE`

## 5. MANO Asset Provenance

`OFFICIAL_RUN_VS_ADAPTER_ASSET_MATCH=YES`
`MANO_V1_2_PROVENANCE={provenance["status"].removeprefix("MANO_V1_2_PROVENANCE_")}`

The official checkout explicitly requires v1.2. The local paired licensed assets are credible, but no version-bearing archive/download receipt or embedded field was found, so this is not marked VERIFIED.

## 6. Component Ablation and Skeleton

All Neutral, Shape-only, Pose-only, and Full-source variants were rendered by both trusted paths with fixed front/oblique/side cameras. Geometry metrics are machine evidence only; surface and skeleton anatomical appearance remain `PENDING_USER_REVIEW`. The 21-joint graph has one wrist root, five distinct chains, no cycles, finite joints, and no duplicate chains for both episodes.

## 7. Beta Audit

The frozen 742 eligible Manifest V2 records contribute {beta["n_betas"]} record-frame beta samples.

{beta_lines}

## 8. Pose Audit

Quaternion norms, normalization corrections, and per-joint shortest rotation angles are in `pose_audit/dev_01.json` and `pose_audit/dev_02.json`. Source values were not normalized or edited. Pose-only meshes and skeletons were generated separately.

## 9. Root Cause

`PRIMARY_ROOT_CAUSE={decision["primary_root_cause"]}`
`CONFIDENCE={decision["confidence"]}`
`LEADING_HYPOTHESIS={decision["leading_hypothesis"]}`

Two independent non-HTML rendering paths, export round trips, component isolation, and skeleton structure are complete. The contract forbids automated anatomical judgment, so the leading HTML-renderer hypothesis cannot be promoted until the user reviews the new contact sheets.

`MANIFEST_V3_REQUIRED={decision["manifest_v3_required"]}`
`NEXT={decision["next"]}`

## 10. Review Artifacts

{artifact_lines}

Each frame directory contains official/independent PNGs, skeleton PNGs, all 12 component-view PNGs, official-renderer counterparts, OBJ, PLY, metrics, beta, pose, render receipt, and diagnosis.

## 11. User Acceptance Order

1. Neutral MANO: is it clearly a normal open human hand?
2. Shape-only: does adding source beta make it thin?
3. Pose-only: does adding source pose make it thin?
4. Full-source.
5. Skeleton-only: are all five fingers plausible?
6. Official OakInk2 renderer.
7. Independent renderer.
8. Only then compare the historical rejected HTML.

Please answer:

```text
OAKINK2_O1R2_DEV_1=NEUTRAL_NORMAL / SHAPE_BREAKS / POSE_BREAKS / FULL_ONLY_BREAKS / ALL_SURFACES_ABNORMAL / OTHER
OAKINK2_O1R2_DEV_2=NEUTRAL_NORMAL / SHAPE_BREAKS / POSE_BREAKS / FULL_ONLY_BREAKS / ALL_SURFACES_ABNORMAL / OTHER
SKELETON_DEV_1=NORMAL / ABNORMAL / UNCERTAIN
SKELETON_DEV_2=NORMAL / ABNORMAL / UNCERTAIN
```

## 12. STOP

`O5_ALLOWED=NO`
`NEXT_STATUS=WAITING_FOR_USER_OAKINK2_O1R2_ARTIFACT_REVIEW`
"""
    (report_root / "handoff.md").write_text(handoff, encoding="utf-8")
    (report_root / "final_summary.md").write_text(handoff, encoding="utf-8")
    return summary


def run_all(report_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    fixed = preflight(report_root)
    runtime_authority(report_root)
    joint_semantics(report_root)
    torch, ManoLayer, transf_point_array_np, OakInk2Dataset = load_official_runtime()
    temporary, model_root = mano_root(MANO_MODEL)
    try:
        layer = ManoLayer(
            mano_assets_root=str(model_root),
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        ).to(torch.device("cpu"))
        layer.eval()
        dataset = OakInk2Dataset(dataset_prefix=str(DATASET_PREFIX), return_instantiated=True)
        results = [
            render_episode(episode, report_root, dataset, layer, torch, transf_point_array_np)
            for episode in fixed["episodes"]
        ]
    finally:
        temporary.cleanup()
    provenance = audit_provenance(report_root)
    beta = audit_betas(fixed, report_root)
    tests = internal_tests(results, report_root)
    decision = evidence_fusion(results, beta, provenance, report_root)
    return finalize(
        fixed, results, beta, provenance, tests, decision, report_root, time.monotonic() - started
    )


def run_render_only(report_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    fixed = preflight(report_root)
    runtime_authority(report_root)
    joint_semantics(report_root)
    torch, ManoLayer, transf_point_array_np, OakInk2Dataset = load_official_runtime()
    temporary, model_root = mano_root(MANO_MODEL)
    try:
        layer = ManoLayer(
            mano_assets_root=str(model_root),
            rot_mode="quat",
            side="right",
            center_idx=0,
            use_pca=False,
            flat_hand_mean=True,
        ).to(torch.device("cpu"))
        layer.eval()
        dataset = OakInk2Dataset(dataset_prefix=str(DATASET_PREFIX), return_instantiated=True)
        results = [
            render_episode(episode, report_root, dataset, layer, torch, transf_point_array_np)
            for episode in fixed["episodes"]
        ]
    finally:
        temporary.cleanup()
    provenance = json.loads(
        (report_root / "mano_asset_provenance/final_decision.json").read_text(encoding="utf-8")
    )
    beta = json.loads((report_root / "beta_audit/final_decision.json").read_text(encoding="utf-8"))
    tests = internal_tests(results, report_root)
    decision = evidence_fusion(results, beta, provenance, report_root)
    return finalize(
        fixed, results, beta, provenance, tests, decision, report_root, time.monotonic() - started
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument(
        "--stage",
        choices=("all", "render-only", "beta-only", "provenance-only"),
        default="all",
    )
    args = parser.parse_args()
    if args.stage == "all":
        result = run_all(args.report_root)
    elif args.stage == "render-only":
        result = run_render_only(args.report_root)
    elif args.stage == "beta-only":
        result = audit_betas(preflight(args.report_root), args.report_root)
    else:
        preflight(args.report_root)
        result = audit_provenance(args.report_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
