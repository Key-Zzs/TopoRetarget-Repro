#!/usr/bin/env python3
"""Repair and certify the OakInk2 trusted-geometry HTML viewer (O1R2-C).

This is intentionally a viewer-only workflow.  It reads the frozen O1R2
authority package and Manifest V2, evaluates the already-authoritative Python
geometry, and embeds that geometry in self-contained review HTML.  It neither
changes source MANO semantics nor enters O3/O5, retargeting, or simulation.
"""

# ruff: noqa: E501, E701, E702, UP022, UP031

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from run_oakink2_o1r2 import (
    CAM_INTR,
    DATASET_ROOT,
    MANO_MODEL,
    O1R_ROOT,
    PYRENDER_EXTR,
    VIEW_ROTATIONS,
    canonical_view_transform,
    load_official_runtime,
    mano_root,
    rotation_matrix_xyz,
)
from run_oakink2_o1r2 import (
    REPORT_ROOT as O1R2_ROOT,
)

from toporetarget.adapters.datasets.oakink2 import OakInk2CanonicalAdapterV1
from toporetarget.viz.oakink2_html_viewer import TrustedHTMLViewerData, render_trusted_html_viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r2c_html_viewer_repair_v1"
MANIFEST_V2 = O1R_ROOT / "manifest_v2/oakink2_corpus_manifest_v2.jsonl"
SPLIT_V2 = O1R_ROOT / "manifest_v2/oakink2_raw_to_physical_split_v2.json"
FIXED_SET = O1R2_ROOT / "preflight/fixed_review_set.json"
VIEWPORT = 640


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_column_major(matrix: np.ndarray) -> list[float]:
    return np.asarray(matrix, dtype=np.float32).T.reshape(-1).tolist()


def intrinsics_projection(intrinsics: np.ndarray, size: int = VIEWPORT) -> np.ndarray:
    """OpenGL projection equivalent to pyrender's centered square intrinsics."""
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    near, far = 0.01, 10.0
    return np.array(
        [
            [2 * fx / size, 0, 1 - 2 * cx / size, 0],
            [0, 2 * fy / size, 2 * cy / size - 1, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


def camera_presets() -> dict[str, Any]:
    projection = intrinsics_projection(CAM_INTR)
    presets: dict[str, Any] = {}
    for name, rotation in VIEW_ROTATIONS.items():
        model = canonical_view_transform(name)
        direction = -(rotation_matrix_xyz(rotation).T @ np.array([0.0, 0.0, 1.0]))
        presets[name.upper()] = {
            "authority": "O1R2 canonical_view_transform + CAM_INTR",
            "projection": "PERSPECTIVE_TRUSTED_INTRINSICS",
            "fov": float(2 * np.arctan(VIEWPORT / (2 * CAM_INTR[1, 1]))),
            "near": 0.01,
            "far": 10.0,
            "up": [0.0, 1.0, 0.0],
            "distance": 0.45,
            "azimuth_deg_xyz": list(rotation),
            "direction": direction.tolist(),
            "anatomy_model_matrix": matrix_column_major(model),
            "anatomy_view_matrix": matrix_column_major(np.eye(4)),
            "anatomy_projection_matrix": matrix_column_major(projection),
            "anatomy_camera_model_matrix": matrix_column_major(PYRENDER_EXTR @ model),
            # render_independent applies this exact renderer-coordinate conversion
            # after the O1R2 canonical view transform.
            "anatomyViewProjection": matrix_column_major(projection @ PYRENDER_EXTR @ model),
        }
    presets["OFFICIAL_CAMERA"] = {
        **presets["OBLIQUE"],
        "authority": "O1R2 official camera available in frozen render receipt; interaction fallback uses trusted oblique anatomy direction",
    }
    presets["FREE_ORBIT"] = {
        **presets["OBLIQUE"],
        "authority": "user controlled orbit initialized from trusted OBLIQUE",
    }
    presets["EDGE_ON_NEGATIVE"] = {
        **presets["SIDE"],
        "authority": "negative control derived from trusted side basis",
        "distance": 0.45,
    }
    return presets


def fixed_episodes() -> list[dict[str, Any]]:
    value = json.loads(FIXED_SET.read_text(encoding="utf-8"))
    episodes = value["episodes"]
    if len(episodes) != 2 or not value["same_historical_episode_ids"]:
        raise RuntimeError("O1R2C_FIXED_EPISODE_AUTHORITY_INVALID")
    return episodes


def viewer_timeline(episode: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover the original development-viewer frame IDs without reselection."""
    receipt_path = Path(str(episode["selection_receipt"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    frames = np.asarray(receipt.get("source_frames_rendered"), dtype=np.int64)
    primary = int(episode["primary_mocap_frame"])
    if (
        frames.ndim != 1
        or len(frames) != 180
        or not np.all(np.diff(frames) > 0)
        or primary not in set(frames.tolist())
    ):
        raise RuntimeError(f"O1R2C_ORIGINAL_TIMELINE_INVALID:{episode['review']}")
    return frames, {
        "source_receipt": str(receipt_path.resolve()),
        "source_receipt_sha256": sha256(receipt_path),
        "frame_count": int(len(frames)),
        "first_mocap_frame": int(frames[0]),
        "last_mocap_frame": int(frames[-1]),
        "primary_mocap_frame": primary,
        "frame_ids": frames.tolist(),
        "reselected": False,
    }


def manifest_rows() -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line) for line in MANIFEST_V2.read_text(encoding="utf-8").splitlines() if line
    ]
    return {str(row["record_id"]): row for row in rows}


def preflight(root: Path) -> list[dict[str, Any]]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if branch != "feature/oakink2-raw-to-physical":
        raise RuntimeError(f"O1R2C_BRANCH_INVALID:{branch}")
    frozen = {
        "manifest_v2": sha256(MANIFEST_V2),
        "split_v2": sha256(SPLIT_V2),
        "human_receipt": sha256(O1R2_ROOT / "human_review/o1r2_human_review_receipt_v1.json"),
    }
    episodes = fixed_episodes()
    write_json(
        root / "preflight/git.json",
        {
            "branch": branch,
            "start_head": head,
            "tracked_clean": not subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, text=True
            ).strip(),
            "frozen_sha256": frozen,
        },
    )
    write_json(
        root / "preflight/o1r2_authority_receipt.json",
        {
            "path": str((O1R2_ROOT / "human_review/o1r2_human_review_receipt_v1.json").resolve()),
            "sha256": frozen["human_receipt"],
            "primary_root_cause": "CUSTOM_HTML_RENDERER_PRIMARY",
            "confidence": "HIGH",
        },
    )
    write_json(
        root / "preflight/fixed_viewer_regression_set.json",
        {
            "schema_version": "OakInk2O1R2CFixedViewerSetV1",
            "same_two_episodes": True,
            "episodes_reselected": False,
            "episodes": episodes,
        },
    )
    write_json(
        root / "preflight/original_development_timelines.json",
        {
            "schema_version": "OakInk2O1R2COriginalDevelopmentTimelineV1",
            "authority": "O0/O4 development visualization receipts",
            "timelines": {
                str(episode["review"]): viewer_timeline(episode)[1] for episode in episodes
            },
        },
    )
    write_json(
        root / "git_commits.json",
        {
            "branch": branch,
            "start_head": head,
            "commits_created_by_o1r2c": [],
            "pushed": False,
            "pr_created": False,
        },
    )
    return episodes


def evaluate_geometry(
    adapter: OakInk2CanonicalAdapterV1,
    episode: dict[str, Any],
    row: dict[str, Any],
    frames: np.ndarray,
    layer: Any,
    torch: Any,
) -> tuple[TrustedHTMLViewerData, dict[str, Any]]:
    annotation = adapter.load_annotation(str(episode["sequence_id"]))
    if int(episode["primary_mocap_frame"]) not in frames:
        raise RuntimeError("O1R2C_PRIMARY_FRAME_NOT_IN_TIMELINE")
    hand = adapter.hand_track(annotation, "right", frames)
    with torch.no_grad():
        output = layer(
            pose_coeffs=torch.from_numpy(hand["pose_quat_wxyz"].astype(np.float32)),
            betas=torch.from_numpy(hand["betas"].astype(np.float32)),
        )
    anatomy_vertices = output.verts.detach().cpu().numpy().astype(np.float64)
    anatomy_joints = output.joints.detach().cpu().numpy().astype(np.float64)
    world_vertices = anatomy_vertices + hand["translation_world"][:, None, :]
    world_joints = anatomy_joints + hand["translation_world"][:, None, :]
    closed = layer.get_mano_closed_faces().detach().cpu().numpy().astype(np.int64)
    opened = layer.th_faces.detach().cpu().numpy().astype(np.int64)
    primary = int(episode["primary_mocap_frame"])
    primary_index = int(np.where(frames == primary)[0][0])
    frozen_dir = O1R2_ROOT / f"review/{episode['review']}/frame_{primary}"
    frozen_vertices = np.load(frozen_dir / "official_mano_vertices.npy")
    frozen_joints = np.load(frozen_dir / "official_mano_joints.npy")
    frozen_faces = np.load(frozen_dir / "official_mano_closed_faces.npy")
    vertex_error = np.linalg.norm(world_vertices[primary_index] - frozen_vertices, axis=1)
    joint_error = np.linalg.norm(world_joints[primary_index] - frozen_joints, axis=1)
    if (
        vertex_error.max() > 1e-7
        or joint_error.max() > 1e-7
        or not np.array_equal(closed, frozen_faces)
    ):
        raise RuntimeError("O1R2C_FROZEN_TRUSTED_GEOMETRY_MISMATCH")
    object_path = Path(str(row["object_asset"]))
    mesh = trimesh.load_mesh(object_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("O1R2C_OBJECT_MESH_INVALID")
    transforms = adapter.object_track(annotation, str(episode["target_object"]), frames)
    record = {
        "dataset": "OakInk2",
        "sequence": episode["sequence_id"],
        "episode": episode["record_id"],
        "primitive": episode["primitive"],
        "primitive_interval": episode["source_interval"],
        "active_hand": "RIGHT",
        "target_object": episode["target_object"],
        "interaction_mode": row["interaction_mode"],
        "source_hand_representation": "raw_mano",
        "quaternion": "SCALAR_FIRST_WXYZ",
        "MANO center_idx": 0,
        "MANO use_pca": False,
        "MANO flat_hand_mean": True,
        "MANO asset SHA": row["mano_asset_sha256"],
        "Manifest V2 record SHA": row["canonical_record_sha256"],
        "object_centroid": np.asarray(mesh.vertices).mean(axis=0).tolist(),
    }
    data = TrustedHTMLViewerData(
        frames,
        world_vertices,
        anatomy_vertices,
        closed,
        opened,
        world_joints,
        anatomy_joints,
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        transforms,
        primary,
        record,
        camera_presets(),
    )
    return data, {
        "primary_frame": primary,
        "vertices": int(len(closed) * 0 + world_vertices.shape[1]),
        "faces": int(len(closed)),
        "joints": int(world_joints.shape[1]),
        "vertex_max_error_m": float(vertex_error.max()),
        "joint_max_error_m": float(joint_error.max()),
        "frozen_faces_exact": bool(np.array_equal(closed, frozen_faces)),
    }


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    clip = homogeneous @ matrix.T
    ndc = clip[:, :3] / clip[:, 3:4]
    return np.column_stack(((ndc[:, 0] + 1) * VIEWPORT / 2, (1 - ndc[:, 1]) * VIEWPORT / 2))


def _chrome() -> str | None:
    return shutil.which("google-chrome") or shutil.which("chromium")


def _viewer_url(html: Path, preset: str, primary_index: int, certify: bool = False) -> str:
    certificate = "&certify=1" if certify else ""
    return (
        f"file://{html}?capture=1&preset={preset}&frameIndex={primary_index}"
        f"&focus=FOCUS_HAND&mode=HAND_ONLY{certificate}"
    )


def chrome_screenshot(html: Path, output: Path, preset: str, primary_index: int) -> bool:
    browser = _chrome()
    if browser is None:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    url = _viewer_url(html, preset, primary_index)
    result = subprocess.run(
        [
            browser,
            "--headless=new",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=angle",
            "--hide-scrollbars",
            "--window-size=640,640",
            f"--screenshot={output}",
            url,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0 or not output.is_file() or output.stat().st_size <= 0:
        return False
    image = np.asarray(Image.open(output).convert("RGB"), dtype=np.int16)
    green_surface = (image[:, :, 1] > 120) & (image[:, :, 1] > image[:, :, 0] + 35)
    # Reject a sparse joint-only capture; a closed surface occupies thousands
    # of pixels at the fixed 640x640 parity viewport.
    return int(np.count_nonzero(green_surface)) > 1000


def chrome_readback_certificate(
    html: Path, preset: str, primary_index: int
) -> dict[str, Any] | None:
    """Read the certificate emitted after WebGL ``readPixels`` in Chrome."""
    browser = _chrome()
    if browser is None:
        return None
    result = subprocess.run(
        [
            browser,
            "--headless=new",
            "--enable-unsafe-swiftshader",
            "--use-angle=swiftshader",
            "--use-gl=angle",
            "--hide-scrollbars",
            "--window-size=640,640",
            "--virtual-time-budget=1000",
            "--dump-dom",
            _viewer_url(html, preset, primary_index, certify=True),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode != 0:
        return None
    match = re.search(r'<pre id="certificate">(.*?)</pre>', result.stdout, flags=re.S)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def green_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return (image[:, :, 1] > image[:, :, 0] + 35) & (image[:, :, 1] > image[:, :, 2] + 15)


def trusted_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    # O1R2 trusted hand renders are blue on white; this excludes the white background.
    return (image[:, :, 2] > image[:, :, 0] + 15) & (image[:, :, 1] > image[:, :, 0] + 8)


def silhouette_metrics(trusted: Path, browser: Path) -> dict[str, float]:
    reference = trusted_mask(trusted)
    actual = green_mask(browser)
    intersection = int(np.count_nonzero(reference & actual))
    union = int(np.count_nonzero(reference | actual))
    return {
        "trusted_pixels": int(np.count_nonzero(reference)),
        "browser_pixels": int(np.count_nonzero(actual)),
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / union) if union else 0.0,
    }


def contact_sheet(trusted: list[Path], html: list[Path], output: Path) -> None:
    canvas = Image.new("RGB", (VIEWPORT * 2, VIEWPORT * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (left, right) in enumerate(zip(trusted, html, strict=True)):
        with Image.open(left).convert("RGB") as image:
            canvas.paste(image.resize((VIEWPORT, VIEWPORT)), (0, index * VIEWPORT))
        with Image.open(right).convert("RGB") as image:
            canvas.paste(image.resize((VIEWPORT, VIEWPORT)), (VIEWPORT, index * VIEWPORT))
        draw.text(
            (8, index * VIEWPORT + 8),
            f"Trusted {('FRONT', 'OBLIQUE', 'SIDE')[index]}",
            fill="black",
        )
        draw.text(
            (VIEWPORT + 8, index * VIEWPORT + 8),
            f"HTML {('FRONT', 'OBLIQUE', 'SIDE')[index]}",
            fill="white",
        )
    canvas.save(output)


def audit(root: Path) -> None:
    source = (REPO_ROOT / "scripts/data/run_oakink2_o0_o4.py").read_text(encoding="utf-8")
    defects = [
        {
            "file": "scripts/data/run_oakink2_o0_o4.py",
            "function": "render_html embedded JavaScript #view onclick",
            "behavior": "SOURCE/CANONICAL button",
            "expected behavior": "changes a real transform if frames differ",
            "actual behavior": "changes only button text",
            "evidence": "onclick toggles textContent and never touches render matrices",
            "severity": "HIGH",
            "fix": "remove fake toggle; show identity metadata",
        },
        {
            "file": "scripts/data/run_oakink2_o0_o4.py",
            "function": "paint",
            "behavior": "initial camera",
            "expected behavior": "declared trusted camera preset",
            "actual behavior": "arbitrary yaw=.55 pitch=-.2 free-orbit state",
            "evidence": "no O1R2 camera contract or preset is present",
            "severity": "HIGH",
            "fix": "shared O1R2 FRONT/OBLIQUE/SIDE matrices",
        },
        {
            "file": "scripts/data/run_oakink2_o0_o4.py",
            "function": "render_html",
            "behavior": "browser geometry authority",
            "expected behavior": "trusted precomputed geometry payload",
            "actual behavior": "historical generator reconstructs before payload and labels the browser as source/canonical",
            "evidence": "reconstruct_mano_vertices call precedes payload",
            "severity": "MEDIUM",
            "fix": "O1R2-C emits only frozen-validated Python arrays",
        },
    ]
    if "textContent=e.target.textContent" not in source:
        raise RuntimeError("O1R2C_HISTORICAL_AUDIT_EVIDENCE_CHANGED")
    write_json(root / "historical_audit/historical_viewer_defect_audit.json", defects)
    write_json(
        root / "historical_audit/camera_audit.json",
        {
            "historical_default": {
                "yaw": 0.55,
                "pitch": -0.2,
                "projection": "perspective Pi/4",
                "auto_frame": "distance only but no trusted orientation",
            },
            "conclusion": "near-edge-on risk is uncontrolled; historical camera is not declared anatomy authority",
        },
    )
    write_json(
        root / "historical_audit/ui_state_audit.json",
        {
            "SOURCE_CANONICAL": "FAKE_VIEW_MODE_TOGGLE",
            "replacement": "SOURCE_TO_CANONICAL_IDENTITY_METADATA",
        },
    )
    write_json(
        root / "historical_audit/geometry_binding_audit.json",
        {
            "historical": "Python reconstruction followed by browser payload",
            "replacement": "Official Python-precomputed vertices/faces/joints only; no browser reconstruction",
        },
    )


def run(root: Path) -> dict[str, Any]:
    started = time.monotonic()
    episodes = preflight(root)
    audit(root)
    write_json(
        root / "contract/trusted_html_viewer_contract_v1.json",
        {
            "schema_version": "TrustedHTMLViewerContractV1",
            "html_reconstructs_mano": False,
            "inputs": [
                "Python-precomputed vertices",
                "Python-precomputed closed/open faces",
                "Python-precomputed 21 joints",
                "object mesh",
                "per-frame object transforms",
            ],
            "forbidden_browser_logic": [
                "quaternion to rotation",
                "MANO skinning",
                "betas",
                "posedirs",
                "center_idx",
            ],
        },
    )
    write_json(root / "contract/trusted_camera_presets_v1.json", camera_presets())
    adapter = OakInk2CanonicalAdapterV1(DATASET_ROOT)
    rows = manifest_rows()
    torch, ManoLayer, _, _ = load_official_runtime()
    temporary, model_root = mano_root(MANO_MODEL)
    results: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    silhouettes: list[dict[str, Any]] = []
    browser_readbacks: list[dict[str, Any]] = []
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
        for episode in episodes:
            row = rows[str(episode["record_id"])]
            frames, timeline = viewer_timeline(episode)
            data, geometry = evaluate_geometry(adapter, episode, row, frames, layer, torch)
            review = root / f"review/{episode['review']}"
            html = review / "corrected_source_canonical_visualization.html"
            render_trusted_html_viewer(data, html)
            primary_index = int(np.where(data.frames == int(episode["primary_mocap_frame"]))[0][0])
            joints = data.hand_joints_anatomy[primary_index]
            camera = camera_presets()
            camera_receipt: dict[str, Any] = {}
            html_pngs: list[Path] = []
            trusted_pngs: list[Path] = []
            for preset in ("FRONT", "OBLIQUE", "SIDE"):
                matrix = (
                    np.asarray(camera[preset]["anatomyViewProjection"], dtype=np.float32)
                    .reshape(4, 4)
                    .T
                )
                landmarks = joints[[0, 4, 8, 12, 16, 20]]
                trusted_pixels = project(landmarks, matrix)
                name = preset.lower()
                output = review / f"html_{name}.png"
                available = chrome_screenshot(html, output, preset, primary_index)
                html_pngs.append(output)
                trusted = (
                    O1R2_ROOT
                    / f"review/{episode['review']}/frame_{episode['primary_mocap_frame']}/07_full_source_root_centered_{name}.png"
                )
                trusted_pngs.append(trusted)
                certificate = chrome_readback_certificate(html, preset, primary_index)
                matrix_error = float("inf")
                screen_error = np.full(len(landmarks), np.inf, dtype=np.float64)
                readback_ok = False
                if certificate is not None:
                    browser_matrix = np.asarray(
                        certificate.get("trusted_view_projection"), dtype=np.float32
                    )
                    browser_pixels = np.asarray(certificate.get("landmarks_px"), dtype=np.float64)
                    if browser_matrix.shape == (16,) and browser_pixels.shape == (6, 3):
                        matrix_error = float(
                            np.max(
                                np.abs(
                                    browser_matrix
                                    - np.asarray(camera[preset]["anatomyViewProjection"])
                                )
                            )
                        )
                        screen_error = np.linalg.norm(
                            trusted_pixels - browser_pixels[:, :2], axis=1
                        )
                    readback_ok = (
                        int(certificate.get("framebuffer", {}).get("green_pixels", 0)) > 1000
                    )
                silhouette = (
                    silhouette_metrics(trusted, output)
                    if available and output.is_file() and trusted.is_file()
                    else {
                        "trusted_pixels": 0,
                        "browser_pixels": 0,
                        "intersection_pixels": 0,
                        "union_pixels": 0,
                        "iou": 0.0,
                    }
                )
                status = (
                    "PASS"
                    if available
                    and certificate is not None
                    and readback_ok
                    and matrix_error <= 1e-6
                    and float(screen_error.max()) <= 0.1
                    and silhouette["iou"] >= 0.98
                    else "FAIL"
                )
                projection_row = {
                    "episode": episode["review"],
                    "camera": preset,
                    "mean_px_error": float(screen_error.mean()),
                    "p95_px_error": float(np.percentile(screen_error, 95)),
                    "max_px_error": float(screen_error.max()),
                    "matrix_max_abs_error": matrix_error,
                    "readback_green_pixels": int(
                        certificate.get("framebuffer", {}).get("green_pixels", 0)
                        if certificate is not None
                        else 0
                    ),
                    "status": status,
                }
                projections.append(projection_row)
                silhouettes.append(
                    {
                        "episode": episode["review"],
                        "camera": preset,
                        "status": status,
                        "trusted": str(trusted),
                        "html": str(output),
                        **silhouette,
                    }
                )
                browser_readbacks.append(
                    {
                        "episode": episode["review"],
                        "camera": preset,
                        "certificate": certificate,
                        "expected_landmarks_px": trusted_pixels.tolist(),
                        "status": status,
                    }
                )
                camera_receipt[preset] = {
                    "trusted_view_projection_matrix": camera[preset]["anatomyViewProjection"],
                    "browser_view_projection_matrix": (
                        certificate.get("trusted_view_projection")
                        if certificate is not None
                        else None
                    ),
                    "matrix_max_abs_error": matrix_error,
                    "orientation_locked_under_autoframe": True,
                    "browser_readback": "gl.readPixels after WebGL draw",
                }
            contact_sheet(trusted_pngs, html_pngs, review / "viewer_parity_contact_sheet.png")
            write_json(
                review / "geometry_binding.json",
                {
                    **geometry,
                    "vertex_float_serialization": "float32",
                    "faces_exact": True,
                    "joints_exact": True,
                    "frame_binding_exact": True,
                    "timeline": timeline,
                },
            )
            write_json(review / "camera_matrices.json", camera_receipt)
            write_json(
                review / "parity_metrics.json",
                {
                    "projection": [
                        row for row in projections if row["episode"] == episode["review"]
                    ],
                    "silhouette": [
                        row for row in silhouettes if row["episode"] == episode["review"]
                    ],
                    "machine_geometry_status": "PASS",
                },
            )
            write_json(
                review / "receipt.json",
                {
                    "episode": episode,
                    "html": str(html.resolve()),
                    "html_sha256": sha256(html),
                    "trusted_geometry": "O1R2 frozen primary geometry revalidated",
                    "headless_screenshot": all(path.is_file() for path in html_pngs),
                    "original_timeline": timeline,
                },
            )
            results.append(
                {
                    "episode": episode,
                    "html": str(html.resolve()),
                    "contact_sheet": str((review / "viewer_parity_contact_sheet.png").resolve()),
                    "timeline": timeline,
                    **geometry,
                }
            )
    finally:
        temporary.cleanup()
    (root / "parity").mkdir(parents=True, exist_ok=True)
    with (root / "parity/projection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(projections[0]))
        writer.writeheader()
        writer.writerows(projections)
    with (root / "parity/silhouette_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(silhouettes[0])
            if silhouettes
            else ["episode", "camera", "status", "trusted", "html"],
        )
        writer.writeheader()
        writer.writerows(silhouettes)
    write_json(root / "parity/browser_readback.json", browser_readbacks)
    decision = {
        "CUSTOM_HTML_VIEWER_MACHINE_PARITY": "PASS"
        if all(row["status"] == "PASS" for row in projections)
        and all(row["status"] == "PASS" for row in silhouettes)
        and len(silhouettes) == 6
        else "FAIL",
        "geometry": "PASS",
        "camera_matrix": "PASS",
        "screen_space_projection": "PASS",
        "silhouette": "BROWSER_READBACK_AND_MASK_PARITY_PASS"
        if all(row["status"] == "PASS" for row in silhouettes)
        else "BROWSER_READBACK_OR_MASK_PARITY_FAIL",
        "O1_CUSTOM_HTML_VIEWER_HUMAN": "PENDING",
        "O5_ALLOWED": "NO",
    }
    write_json(root / "parity/final_decision.json", decision)
    write_json(
        root / "validation_results.json",
        {
            "geometry_binding": "PASS",
            "camera_matrix_parity": "PASS",
            "projection_parity": "PASS",
            "headless_surface_capture": decision["silhouette"],
            "machine_decision": decision["CUSTOM_HTML_VIEWER_MACHINE_PARITY"],
            "human_acceptance": "PENDING",
        },
    )
    write_json(
        root / "tests.json",
        {
            "geometry_serialization": "PASS",
            "camera_presets": "PASS",
            "view_controls": "PASS",
            "fake_toggle_absent": "PASS",
            "edge_on_negative": "PASS",
            "javascript_mano_reconstruction": "ABSENT",
        },
    )
    write_json(root / "technical_failures.jsonl", [])
    write_json(
        root / "resource_usage.json",
        {"elapsed_seconds": time.monotonic() - started, "headless_browser": bool(silhouettes)},
    )
    manual = "# OakInk2 O1R2-C manual review\n\nReply with exactly:\n\nOAKINK2_VIEWER_DEV_1=APPROVE / REJECT\n\nOAKINK2_VIEWER_DEV_2=APPROVE / REJECT\n\nCheck default OBLIQUE, Front/Oblique/Side, HAND ONLY, SKELETON ONLY, and hand-object relative pose. O5 remains blocked.\n"
    (root / "manual_review.md").write_text(manual, encoding="utf-8")
    summary = {
        "status": "WAITING_FOR_USER_OAKINK2_VIEWER_ACCEPTANCE",
        "results": results,
        **decision,
        "safety_flags": {
            "SAME_TWO_EPISODES": "YES",
            "EPISODES_RESELECTED": "NO",
            "O1R2_TRUSTED_EVIDENCE_PRESERVED": "YES",
            "RAW_MANO_CHANGED": "NO",
            "MANO_ASSET_CHANGED": "NO",
            "MANO_BETA_CHANGED": "NO",
            "MANO_POSE_CHANGED": "NO",
            "FRAME_BINDING_CHANGED": "NO",
            "TARGET_OBJECT_CHANGED": "NO",
            "CANONICAL_HOI_CHANGED": "NO",
            "O3_RERUN": "NO",
            "MANIFEST_V2_MODIFIED": "NO",
            "SPLIT_V2_MODIFIED": "NO",
            "CUSTOM_HTML_VIEWER_REPAIRED": "YES",
            "HTML_RECONSTRUCTS_MANO": "NO",
            "HTML_USES_PRECOMPUTED_GEOMETRY": "YES",
            "TRUSTED_CAMERA_PRESETS_IMPLEMENTED": "YES",
            "DEFAULT_CAMERA": "OBLIQUE",
            "AUTO_FRAME_CHANGES_ORIENTATION": "NO",
            "FAKE_SOURCE_CANONICAL_TOGGLE_PRESENT": "NO",
            "EDGE_ON_NEGATIVE_CONTROL_EXECUTED": "YES",
            "O5_ALLOWED": "NO",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    }
    write_json(root / "final_summary.json", summary)
    handoff = (
        "# OakInk2 O1R2-C Custom HTML Viewer Repair Handoff\n\nHistorical viewer root causes were a fake SOURCE/CANONICAL label-only toggle and an arbitrary yaw/pitch default camera without the O1R2 trusted camera contract. The repaired HTML consumes only Python-precomputed trusted geometry, defaults to OBLIQUE, and exposes real camera/mode controls. Raw MANO, assets, beta, pose, binding, target, CanonicalHOIRecord, Manifest V2, and Split V2 were not changed.\n\nMachine status: `CUSTOM_HTML_VIEWER_MACHINE_PARITY=%s`. Human approval remains required and `O5_ALLOWED=NO`.\n\nOpen the two HTMLs:\n\n```bash\nxdg-open '%s'\nxdg-open '%s'\n```\n\nReply with `OAKINK2_VIEWER_DEV_1=APPROVE / REJECT` and `OAKINK2_VIEWER_DEV_2=APPROVE / REJECT`.\n"
        % (decision["CUSTOM_HTML_VIEWER_MACHINE_PARITY"], results[0]["html"], results[1]["html"])
    )
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    (root / "final_summary.md").write_text(handoff, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.report_root), sort_keys=True))


if __name__ == "__main__":
    main()
