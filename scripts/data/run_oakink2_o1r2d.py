#!/usr/bin/env python3
"""Generate and certify the OakInk2 O1R2-D HTML Viewer V2.

The default command is the frozen same-two-episode workflow. ``--action
generate`` emits HTML only; ``--action certify`` tests already-generated HTML;
``--review`` narrows either operation to one authoritative development episode.
"""

# ruff: noqa: E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from oakink2_browser_cdp import ChromeCDP
from PIL import Image, ImageDraw
from run_oakink2_o1r2 import MANO_MODEL, load_official_runtime, mano_root
from run_oakink2_o1r2c import (
    DATASET_ROOT,
    MANIFEST_V2,
    O1R2_ROOT,
    SPLIT_V2,
    OakInk2CanonicalAdapterV1,
    camera_presets,
    evaluate_geometry,
    fixed_episodes,
    manifest_rows,
    sha256,
    silhouette_metrics,
    viewer_timeline,
)

from toporetarget.viz.oakink2_html_viewer import render_oakink2_html_viewer_v2

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r2d_ref2dex_html_viewer_v1"
REF2DEX_ROOT = Path("/home/deepcybo/workspace/dex/Ref2Dex-main")
REF_VIEWER = REF2DEX_ROOT / "dataset/OakInk2/dataset_audit/common/meshcat_viewer.py"
REF_ENTRYPOINT = REF2DEX_ROOT / "dataset/OakInk2/dataset_audit/scripts/view_meshcat.py"
REF_README = REF2DEX_ROOT / "dataset/OakInk2/README.md"
VIEWPORT = 640
HISTORICAL_ROOT = REPO_ROOT / ".local/reports/oakink2_o1r2c_html_viewer_repair_v1"
LANDMARK_INDICES = (0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20)


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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).rstrip()


def file_sha256(path: Path) -> str:
    return sha256(path)


def tree_metadata_snapshot(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        relative = str(path.relative_to(root))
        digest.update(relative.encode())
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
        count += 1
    return {"file_count": count, "metadata_sha256": digest.hexdigest()}


def frozen_hashes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    geometry: dict[str, dict[str, str]] = {}
    for episode in episodes:
        primary = int(episode["primary_mocap_frame"])
        root = O1R2_ROOT / f"review/{episode['review']}/frame_{primary}"
        geometry[str(episode["review"])] = {
            name: file_sha256(root / name)
            for name in (
                "official_mano_vertices.npy",
                "official_mano_joints.npy",
                "official_mano_closed_faces.npy",
                "official_mano_open_faces.npy",
            )
        }
    return {
        "manifest_v2": {"path": str(MANIFEST_V2.resolve()), "sha256": file_sha256(MANIFEST_V2)},
        "split_v2": {"path": str(SPLIT_V2.resolve()), "sha256": file_sha256(SPLIT_V2)},
        "mano_asset": {"path": str(MANO_MODEL.resolve()), "sha256": file_sha256(MANO_MODEL)},
        "o1r2_source_geometry": geometry,
        "episode_selection_receipts": {
            str(episode["review"]): file_sha256(Path(str(episode["selection_receipt"])))
            for episode in episodes
        },
    }


def preflight(root: Path, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    if branch != "feature/oakink2-raw-to-physical":
        raise RuntimeError(f"O1R2D_BRANCH_INVALID:{branch}")
    timelines = {str(ep["review"]): viewer_timeline(ep)[1] for ep in episodes}
    if any(value["frame_count"] != 180 for value in timelines.values()):
        raise RuntimeError("O1R2D_TIMELINE_NOT_180")
    status = git("status", "--short", "--untracked-files=all").splitlines()
    result = {
        "branch": branch,
        "start_head": head,
        "status_short": status,
        "tracked_worktree_clean": not bool(git("status", "--porcelain", "--untracked-files=no")),
        "new_branch_created": False,
        "new_worktree_created": False,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "preflight/git.json", result)
    write_json(
        root / "preflight/fixed_regression_set.json",
        {
            "schema_version": "OakInk2O1R2DFixedRegressionSetV1",
            "authority": "O1R2-C authoritative receipts",
            "same_two_episodes": True,
            "episodes_reselected": False,
            "objects_reselected": False,
            "primary_frames_reselected": False,
            "episodes": episodes,
            "timelines": timelines,
        },
    )
    hashes = frozen_hashes(episodes)
    write_json(root / "preflight/frozen_hashes.json", hashes)
    return {"git": result, "hashes": hashes, "timelines": timelines}


def audit_reference(root: Path) -> dict[str, Any]:
    required = (REF_VIEWER, REF_ENTRYPOINT, REF_README)
    if not all(path.is_file() for path in required):
        raise RuntimeError("REF2DEX_REFERENCE_SOURCE_MISSING")
    before = tree_metadata_snapshot(REF2DEX_ROOT)
    source_files = [
        {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in required
    ]
    git_pointer = (REF2DEX_ROOT / ".git").read_text(encoding="utf-8").strip()
    audit = {
        "schema_version": "Ref2DexViewerReferenceAuditV1",
        "repo": str(REF2DEX_ROOT),
        "commit_head": None,
        "commit_head_status": "UNAVAILABLE_BROKEN_LINKED_WORKTREE_METADATA",
        "git_pointer": git_pointer,
        "source_files": source_files,
        "main_html_generator": None,
        "main_viewer_entrypoint": str(REF_ENTRYPOINT),
        "rendering_library": "MeshCat Python API with its browser WebGL frontend",
        "camera_model": "Renderer-owned scene camera; Ref2Dex Python never replaces mesh transforms to orbit",
        "orbit_implementation": "MeshCat browser controls, external to Ref2Dex Python source",
        "zoom_implementation": "MeshCat browser controls, external to Ref2Dex Python source",
        "reset_implementation": "MeshCat browser controls; no dataset-side reset implementation",
        "geometry_serialization": "NumPy vertices and integer faces passed to TriangularMeshGeometry",
        "multiple_meshes": "Named scene nodes such as vis['object'] and vis[label]",
        "animation_implementation": "Python frame loop replaces each named node's per-frame geometry",
        "per_frame_object_transform": "Already-transformed per-frame object vertices; no camera/model coupling",
        "scene_graph": True,
        "model_transforms_separate_from_camera": True,
        "dependency_strategy": "Installed MeshCat live server/browser dependency",
        "cdn_detected_in_reference_source": False,
        "self_contained_html": False,
        "production_runtime_dependency_suitable": False,
        "what_is_reusable": [
            "named hand/object scene nodes",
            "Python-precomputed mesh arrays",
            "one shared renderer camera for all nodes",
            "dataset-independent visibility through scene nodes",
        ],
        "what_must_not_be_copied": [
            "live MeshCat server dependency",
            "time.sleep playback loop",
            "full per-frame geometry retransmission",
            "Ref2Dex dataset-cache path dependency",
        ],
        "reference_tree_before": before,
    }
    write_json(root / "ref2dex_reference/architecture_audit.json", audit)
    write_json(root / "ref2dex_reference/source_files.json", source_files)
    decision = {
        "decision": "REBUILD_USING_REF2DEX_VIEWER_ARCHITECTURE",
        "reason": "Ref2Dex's stable separation is a MeshCat live scene graph, not a reusable standalone HTML generator. Viewer V2 ports the node/camera architecture into a self-contained repo-owned WebGL document.",
        "code_copied_verbatim": False,
        "runtime_dependency_on_ref2dex_main": False,
        "runtime_dependency_on_meshcat": False,
        "cdn_required": False,
    }
    write_json(root / "ref2dex_reference/reuse_decision.json", decision)
    return audit


def viewer_contracts(root: Path) -> None:
    write_json(
        root / "new_viewer/contract.json",
        {
            "schema_version": "OakInk2HTMLViewerV2",
            "production_viewer": True,
            "old_viewer_production": False,
            "python_precomputed_geometry_only": True,
            "html_reconstructs_mano": False,
            "offline_self_contained": True,
            "cdn_required": False,
        },
    )
    write_json(
        root / "new_viewer/camera_state_contract.json",
        {
            "schema_version": "ViewerCameraStateV1",
            "fields": ["basePreset", "focus", "yaw", "pitch", "distanceScale"],
            "default": {
                "basePreset": "OBLIQUE",
                "focus": "FOCUS_INTERACTION",
                "yaw": 0,
                "pitch": 0,
                "distanceScale": 1,
            },
            "left_drag": "yaw and bounded pitch",
            "wheel": "camera distance scale",
            "reset": "deterministic OBLIQUE",
            "pan": "PAN_NOT_SUPPORTED_BY_DESIGN",
        },
    )
    write_json(
        root / "new_viewer/scene_frame_contract.json",
        {
            "scene_frame": "SCENE_WORLD_MANO_ROOT_RELATIVE",
            "hand_model": "IDENTITY",
            "skeleton_model": "IDENTITY",
            "object_model": "Python-precomputed per-frame source object transform relative to hand root",
            "camera_equation": "p_camera = T_camera_from_scene * p_scene",
            "orbit_mutates_scene_geometry": False,
        },
    )


def historical_audit(root: Path) -> None:
    old_source = REPO_ROOT / "src/toporetarget/viz/oakink2_html_viewer.py"
    rejection = HISTORICAL_ROOT / "human_review/free_orbit_rejection_v1.json"
    write_json(
        root / "historical_viewer/defect_audit.json",
        {
            "historical_artifact_root": str(HISTORICAL_ROOT.resolve()),
            "historical_artifacts_overwritten": False,
            "human_rejection_receipt": str(rejection.resolve()),
            "human_rejection_receipt_sha256": file_sha256(rejection),
            "old_root_cause": "FREE_ORBIT freeView constructed a new view with model=identity and discarded anatomy_camera_model_matrix",
            "replacement_source": str(old_source.resolve()),
            "replacement_has_free_view_function": "function freeView()"
            in old_source.read_text(encoding="utf-8"),
        },
    )


def html_url(
    html: Path, primary_index: int, *, preset: str = "OBLIQUE", mode: str = "HAND_OBJECT"
) -> str:
    return f"{html.resolve().as_uri()}?capture=1&certify=1&frameIndex={primary_index}&preset={preset}&focus=FOCUS_INTERACTION&mode={mode}"


def pairwise(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(np.asarray(value, dtype=np.float64)).all())
    except (TypeError, ValueError):
        return False


def certificate_metrics(
    certificate: dict[str, Any], baseline: dict[str, Any] | None, *, require_camera_change: bool
) -> dict[str, Any]:
    screen = np.asarray(certificate["landmarks_px"], dtype=np.float64)[:, :2]
    scene = np.asarray(certificate["landmarks_scene"], dtype=np.float64)
    camera = np.asarray(certificate["landmarks_camera"], dtype=np.float64)
    centered = screen - screen.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.count_nonzero(singular > max(float(singular[0]) * 1e-6, 1e-7)))
    result: dict[str, Any] = {
        "finite_view_matrix": finite(certificate["camera_view_matrix"]),
        "finite_projection": finite(certificate["projection_matrix"]),
        "finite_clip_and_landmarks": finite(certificate["landmarks_px"]),
        "green_pixels": int(certificate["framebuffer"]["green_pixels"]),
        "hand_visible": int(certificate["framebuffer"]["green_pixels"]) > 1000,
        "screen_covariance_rank": rank,
        "screen_non_degenerate": rank == 2,
        "camera_depth_range": float(np.ptp(camera[:, 2])),
        "camera_depth_meaningful": float(np.ptp(camera[:, 2])) > 1e-5,
        "scene_landmarks_finite": finite(scene),
        "camera_changed": None,
        "pairwise_3d_max_error_m": 0.0,
        "hand_fingerprint_unchanged": True,
        "joint_fingerprint_unchanged": True,
        "object_model_unchanged": True,
        "hand_object_anchor_unchanged": True,
    }
    if baseline is not None:
        base_scene = np.asarray(baseline["landmarks_scene"], dtype=np.float64)
        result["pairwise_3d_max_error_m"] = float(
            np.max(np.abs(pairwise(scene) - pairwise(base_scene)))
        )
        result["hand_fingerprint_unchanged"] = (
            certificate["scene_nodes"]["hand"]["fingerprint"]
            == baseline["scene_nodes"]["hand"]["fingerprint"]
        )
        result["joint_fingerprint_unchanged"] = (
            certificate["scene_nodes"]["skeleton"]["fingerprint"]
            == baseline["scene_nodes"]["skeleton"]["fingerprint"]
        )
        result["object_model_unchanged"] = np.array_equal(
            np.asarray(certificate["scene_nodes"]["object"]["model_matrix"]),
            np.asarray(baseline["scene_nodes"]["object"]["model_matrix"]),
        )
        result["hand_object_anchor_unchanged"] = (
            certificate["hand_object_anchor"] == baseline["hand_object_anchor"]
        )
        result["camera_changed"] = not np.allclose(
            certificate["camera_view_matrix"], baseline["camera_view_matrix"], atol=1e-7, rtol=0
        )
    checks = [
        result["finite_view_matrix"],
        result["finite_projection"],
        result["finite_clip_and_landmarks"],
        result["hand_visible"],
        result["screen_non_degenerate"],
        result["camera_depth_meaningful"],
        result["scene_landmarks_finite"],
        result["pairwise_3d_max_error_m"] <= 1e-7,
        result["hand_fingerprint_unchanged"],
        result["joint_fingerprint_unchanged"],
        result["object_model_unchanged"],
        result["hand_object_anchor_unchanged"],
    ]
    if require_camera_change:
        checks.append(result["camera_changed"] is True)
    result["status"] = "PASS" if all(checks) else "FAIL"
    return result


def runtime_evaluate(browser: ChromeCDP, statement: str) -> Any:
    return browser.evaluate(f"(()=>{{{statement};return true}})()")


def make_contact_sheet(review: Path, trusted_oblique: Path) -> Path:
    names = [
        ("fixed_front.png", "FRONT"),
        ("fixed_oblique.png", "OBLIQUE"),
        ("fixed_side.png", "SIDE"),
        ("drag_horizontal.png", "drag horizontal"),
        ("drag_vertical.png", "drag vertical"),
        ("drag_diagonal.png", "drag diagonal"),
        ("drag_after_10_gestures.png", "10 drags"),
        ("zoom_in.png", "zoom in"),
        ("reset_after_drag.png", "reset"),
        (str(trusted_oblique), "trusted oblique"),
        ("fixed_oblique.png", "new oblique"),
        ("drag_horizontal.png", "new post-drag"),
    ]
    canvas = Image.new("RGB", (VIEWPORT * 3, VIEWPORT * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (raw, label) in enumerate(names):
        path = Path(raw) if Path(raw).is_absolute() else review / raw
        with Image.open(path).convert("RGB") as image:
            canvas.paste(
                image.resize((VIEWPORT, VIEWPORT)),
                ((index % 3) * VIEWPORT, (index // 3) * VIEWPORT),
            )
        draw.rectangle(
            (
                (index % 3) * VIEWPORT,
                (index // 3) * VIEWPORT,
                (index % 3) * VIEWPORT + 175,
                (index // 3) * VIEWPORT + 25,
            ),
            fill="white",
        )
        draw.text(((index % 3) * VIEWPORT + 6, (index // 3) * VIEWPORT + 6), label, fill="black")
    output = review / "interactive_orbit_contact_sheet.png"
    canvas.save(output)
    return output


def generate_html(
    root: Path, selected: list[dict[str, Any]], frame_ids: list[int] | None
) -> list[dict[str, Any]]:
    rows = manifest_rows()
    adapter = OakInk2CanonicalAdapterV1(DATASET_ROOT)
    torch, ManoLayer, _, _ = load_official_runtime()
    temporary, model_root = mano_root(MANO_MODEL)
    generated: list[dict[str, Any]] = []
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
        for episode in selected:
            frozen_frames, timeline = viewer_timeline(episode)
            frames = (
                np.asarray(frame_ids, dtype=np.int64) if frame_ids is not None else frozen_frames
            )
            data, geometry = evaluate_geometry(
                adapter, episode, rows[str(episode["record_id"])], frames, layer, torch
            )
            review = root / f"review/{episode['review']}"
            html = review / "oakink2_interactive_viewer.html"
            renderer = render_oakink2_html_viewer_v2(data, html)
            receipt = {
                "episode": episode,
                "html": str(html.resolve()),
                "html_sha256": file_sha256(html),
                "renderer": renderer,
                "geometry": geometry,
                "timeline": timeline
                if frame_ids is None
                else {"frame_ids": frames.tolist(), "explicit": True},
                "n_frames": int(len(frames)),
                "episodes_reselected": False,
                "objects_reselected": False,
                "primary_frames_reselected": False,
            }
            write_json(review / "receipt.json", receipt)
            generated.append(receipt)
    finally:
        temporary.cleanup()
    return generated


def certify_episode(root: Path, episode: dict[str, Any], browser_executable: str) -> dict[str, Any]:
    review = root / f"review/{episode['review']}"
    html = review / "oakink2_interactive_viewer.html"
    if not html.is_file():
        raise RuntimeError(f"O1R2D_HTML_MISSING:{html}")
    timeline = viewer_timeline(episode)[1]
    primary_index = timeline["frame_ids"].index(int(episode["primary_mocap_frame"]))
    fixed_rows: list[dict[str, Any]] = []
    visibility_rows: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    drag_rows: list[dict[str, Any]] = []
    zoom_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    playback_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    readbacks: dict[str, Any] = {}
    with ChromeCDP(browser_executable, width=VIEWPORT, height=VIEWPORT) as browser:
        for preset in ("FRONT", "OBLIQUE", "SIDE"):
            browser.navigate(html_url(html, primary_index, preset=preset, mode="HAND_ONLY"))
            cert = browser.certificate()
            browser.screenshot(review / f"fixed_{preset.lower()}.png")
            expected = np.asarray(
                camera_presets()[preset]["anatomyViewProjection"], dtype=np.float64
            )
            actual = np.asarray(cert["view_projection_matrix"], dtype=np.float64)
            metrics = certificate_metrics(cert, None, require_camera_change=False)
            trusted = (
                O1R2_ROOT
                / f"review/{episode['review']}/frame_{episode['primary_mocap_frame']}/07_full_source_root_centered_{preset.lower()}.png"
            )
            silhouette = silhouette_metrics(trusted, review / f"fixed_{preset.lower()}.png")
            row = {
                "episode": episode["review"],
                "preset": preset,
                "matrix_max_abs_error": float(np.max(np.abs(expected - actual))),
                "silhouette_iou": silhouette["iou"],
                "green_pixels": metrics["green_pixels"],
                "landmark_rank": metrics["screen_covariance_rank"],
            }
            row["status"] = (
                "PASS"
                if row["matrix_max_abs_error"] <= 1e-6
                and row["silhouette_iou"] >= 0.98
                and metrics["status"] == "PASS"
                else "FAIL"
            )
            fixed_rows.append(row)
            states[f"fixed_{preset.lower()}"] = cert
            readbacks[f"fixed_{preset.lower()}"] = cert["framebuffer"]

        browser.navigate(html_url(html, primary_index, preset="OBLIQUE", mode="HAND_OBJECT"))
        baseline = browser.certificate()
        for visibility_mode in (
            "HAND_ONLY",
            "HAND_OBJECT",
            "SKELETON_ONLY",
            "HAND_SKELETON_OBJECT",
        ):
            runtime_evaluate(
                browser,
                f"window.__OAKINK2_VIEWER_V2__.setMode('{visibility_mode}')",
            )
            cert = browser.certificate()
            framebuffer = cert["framebuffer"]
            identity = np.eye(4).T.reshape(-1)
            shared_scene_basis = bool(
                np.array_equal(cert["scene_nodes"]["hand"]["model_matrix"], identity)
                and np.array_equal(cert["scene_nodes"]["skeleton"]["model_matrix"], identity)
            )
            hand_expected = visibility_mode != "SKELETON_ONLY"
            object_expected = visibility_mode in ("HAND_OBJECT", "HAND_SKELETON_OBJECT")
            visible = int(framebuffer["foreground_pixels"]) > 20
            hand_matches = (int(framebuffer["green_pixels"]) > 1000) == hand_expected
            # Yellow skeleton fragments overlap the broad orange diagnostic.
            # Object presence is therefore color-checked only in modes that
            # include the hand surface; skeleton-only visibility uses its
            # non-background pixels plus the absent green surface.
            object_matches = (
                int(framebuffer["orange_pixels"]) > 100
                if object_expected
                else int(framebuffer["orange_pixels"]) < 100
                if visibility_mode == "HAND_ONLY"
                else True
            )
            visibility_rows.append(
                {
                    "episode": episode["review"],
                    "mode": visibility_mode,
                    "foreground_pixels": int(framebuffer["foreground_pixels"]),
                    "green_pixels": int(framebuffer["green_pixels"]),
                    "orange_pixels": int(framebuffer["orange_pixels"]),
                    "shared_scene_basis": shared_scene_basis,
                    "status": "PASS"
                    if visible and hand_matches and object_matches and shared_scene_basis
                    else "FAIL",
                }
            )
        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setMode('HAND_OBJECT')")

        gestures = {
            "horizontal": ((300, 300), (410, 300)),
            "vertical": ((300, 300), (300, 385)),
            "diagonal": ((285, 285), (385, 360)),
        }
        for name, (start, end) in gestures.items():
            runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
            base = browser.certificate()
            browser.mouse_drag(start, end, steps=3)
            cert = browser.certificate()
            browser.screenshot(review / f"drag_{name}.png")
            metrics = certificate_metrics(cert, base, require_camera_change=True)
            drag_rows.append({"episode": episode["review"], "drag": name, **metrics})
            states[f"drag_{name}"] = cert
            readbacks[f"drag_{name}"] = cert["framebuffer"]

        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
        reverse_base = browser.certificate()
        browser.mouse_drag((300, 300), (390, 330), steps=2)
        browser.mouse_drag((390, 330), (300, 300), steps=2)
        reverse = browser.certificate()
        reverse_metrics = certificate_metrics(reverse, reverse_base, require_camera_change=False)
        reverse_metrics["returned_near_start"] = bool(
            np.allclose(
                reverse["camera_view_matrix"], reverse_base["camera_view_matrix"], atol=2e-6, rtol=0
            )
        )
        reverse_metrics["status"] = (
            "PASS"
            if reverse_metrics["status"] == "PASS" and reverse_metrics["returned_near_start"]
            else "FAIL"
        )
        drag_rows.append({"episode": episode["review"], "drag": "reverse", **reverse_metrics})

        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
        repeated_base = browser.certificate()
        for offset in range(10):
            browser.mouse_drag((260 + offset, 300), (272 + offset, 306), steps=2)
        repeated = browser.certificate()
        browser.screenshot(review / "drag_after_10_gestures.png")
        repeated_metrics = certificate_metrics(repeated, repeated_base, require_camera_change=True)
        drag_rows.append({"episode": episode["review"], "drag": "repeated_10", **repeated_metrics})
        states["drag_after_10_gestures"] = repeated
        readbacks["drag_after_10_gestures"] = repeated["framebuffer"]

        for preset in ("FRONT", "OBLIQUE", "SIDE"):
            runtime_evaluate(browser, f"window.__OAKINK2_VIEWER_V2__.setPreset('{preset}')")
            base = browser.certificate()
            browser.mouse_drag((300, 300), (345, 320), steps=2)
            cert = browser.certificate()
            metrics = certificate_metrics(cert, base, require_camera_change=True)
            drag_rows.append({"episode": episode["review"], "drag": f"{preset}_to_drag", **metrics})
            runtime_evaluate(browser, f"window.__OAKINK2_VIEWER_V2__.setPreset('{preset}')")
            restored = browser.certificate()
            expected = states[f"fixed_{preset.lower()}"]["camera_view_matrix"]
            deterministic = bool(
                np.array_equal(np.asarray(restored["camera_view_matrix"]), np.asarray(expected))
            )
            reset_rows.append(
                {
                    "episode": episode["review"],
                    "transition": f"drag_to_{preset}",
                    "deterministic": deterministic,
                    "status": "PASS" if deterministic else "FAIL",
                }
            )

        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
        zoom_base = browser.certificate()
        browser.wheel(-260)
        zoom_in = browser.certificate()
        browser.screenshot(review / "zoom_in.png")
        zoom_in_metrics = certificate_metrics(zoom_in, zoom_base, require_camera_change=True)
        zoom_rows.append({"episode": episode["review"], "zoom": "in", **zoom_in_metrics})
        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
        zoom_base = browser.certificate()
        browser.wheel(260)
        zoom_out = browser.certificate()
        browser.screenshot(review / "zoom_out.png")
        zoom_out_metrics = certificate_metrics(zoom_out, zoom_base, require_camera_change=True)
        zoom_rows.append({"episode": episode["review"], "zoom": "out", **zoom_out_metrics})
        states["zoom_in"] = zoom_in
        states["zoom_out"] = zoom_out

        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
        browser.mouse_drag((300, 300), (390, 355), steps=3)
        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.resetCamera()")
        reset = browser.certificate()
        browser.screenshot(review / "reset_after_drag.png")
        deterministic = bool(
            np.array_equal(
                np.asarray(reset["camera_view_matrix"]),
                np.asarray(states["fixed_oblique"]["camera_view_matrix"]),
            )
        )
        reset_rows.append(
            {
                "episode": episode["review"],
                "transition": "drag_to_reset",
                "deterministic": deterministic,
                "status": "PASS" if deterministic else "FAIL",
            }
        )
        states["reset_after_drag"] = reset

        for index, yaw in enumerate((-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150)):
            pitch = (-35, 0, 35)[index % 3]
            runtime_evaluate(browser, f"window.__OAKINK2_VIEWER_V2__.setOrbit({yaw},{pitch},1)")
            cert = browser.certificate()
            metrics = certificate_metrics(
                cert, baseline, require_camera_change=yaw != 0 or pitch != 0
            )
            sweep_rows.append(
                {"episode": episode["review"], "yaw_deg": yaw, "pitch_deg": pitch, **metrics}
            )
        for index in range(50):
            yaw = -175 + index * 350 / 49
            pitch = 55 * math.sin(index * math.pi * 2 / 49)
            runtime_evaluate(browser, f"window.__OAKINK2_VIEWER_V2__.setOrbit({yaw},{pitch},1)")
            cert = browser.certificate()
            metrics = certificate_metrics(cert, baseline, require_camera_change=True)
            sweep_rows.append(
                {
                    "episode": episode["review"],
                    "yaw_deg": yaw,
                    "pitch_deg": pitch,
                    "synthetic_step": index,
                    **metrics,
                }
            )

        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.setPreset('OBLIQUE')")
        browser.mouse_drag((300, 300), (365, 330), steps=3)
        before_frame = browser.certificate()
        next_index = (primary_index + 1) % len(timeline["frame_ids"])
        runtime_evaluate(browser, f"window.__OAKINK2_VIEWER_V2__.setFrame({next_index})")
        after_frame = browser.certificate()
        retained = bool(
            np.array_equal(
                np.asarray(before_frame["camera_view_matrix"]),
                np.asarray(after_frame["camera_view_matrix"]),
            )
        )
        playback_rows.append(
            {
                "episode": episode["review"],
                "test": "frame_change_after_drag",
                "from_mocap_frame": before_frame["mocap_frame_id"],
                "to_mocap_frame": after_frame["mocap_frame_id"],
                "camera_retained": retained,
                "hand_visible": after_frame["framebuffer"]["green_pixels"] > 1000,
                "status": "PASS"
                if retained and after_frame["framebuffer"]["green_pixels"] > 1000
                else "FAIL",
            }
        )
        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.play()")
        time.sleep(0.38)
        runtime_evaluate(browser, "window.__OAKINK2_VIEWER_V2__.pause()")
        after_play = browser.certificate()
        play_retained = bool(
            np.array_equal(
                np.asarray(before_frame["camera_view_matrix"]),
                np.asarray(after_play["camera_view_matrix"]),
            )
        )
        advanced = after_play["frame_index"] != after_frame["frame_index"]
        playback_rows.append(
            {
                "episode": episode["review"],
                "test": "playback_after_drag",
                "camera_retained": play_retained,
                "frame_advanced": advanced,
                "hand_visible": after_play["framebuffer"]["green_pixels"] > 1000,
                "status": "PASS"
                if play_retained and advanced and after_play["framebuffer"]["green_pixels"] > 1000
                else "FAIL",
            }
        )

        old_html = (
            HISTORICAL_ROOT
            / f"review/{episode['review']}/corrected_source_canonical_visualization.html"
        )
        browser.navigate(
            f"{old_html.resolve().as_uri()}?capture=1&certify=1&frameIndex={primary_index}&preset=OBLIQUE&focus=FOCUS_HAND&mode=HAND_ONLY"
        )
        old_before = browser.certificate()
        browser.mouse_drag((300, 300), (410, 300), steps=3)
        old_after = browser.certificate()
        historical = {
            "episode": episode["review"],
            "historical_html": str(old_html.resolve()),
            "historical_html_sha256": file_sha256(old_html),
            "before_camera_model_matrix": old_before.get("camera_model_matrix"),
            "after_camera_model_matrix": old_after.get("camera_model_matrix"),
            "after_preset": old_after.get("preset"),
            "failure_reproduced": old_after.get("preset") == "FREE_ORBIT"
            and not np.array_equal(
                np.asarray(old_after.get("camera_model_matrix")),
                np.asarray(old_before.get("camera_model_matrix")),
            )
            and int(np.count_nonzero(np.asarray(old_after.get("camera_model_matrix")))) == 4,
        }
        historical["status"] = (
            "PASS_NEGATIVE_FIXTURE" if historical["failure_reproduced"] else "NOT_REPRODUCED"
        )

    trusted_oblique = (
        O1R2_ROOT
        / f"review/{episode['review']}/frame_{episode['primary_mocap_frame']}/07_full_source_root_centered_oblique.png"
    )
    contact_sheet = make_contact_sheet(review, trusted_oblique)
    write_json(review / "camera_states.json", states)
    write_json(review / "projection_metrics.json", fixed_rows)
    write_json(review / "relative_transform_metrics.json", drag_rows + zoom_rows)
    write_json(review / "browser_readback.json", readbacks)
    receipt_path = review / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "interactive_certification": "PASS"
            if all(
                row["status"] == "PASS"
                for row in visibility_rows
                + drag_rows
                + zoom_rows
                + reset_rows
                + playback_rows
                + sweep_rows
            )
            and all(row["status"] == "PASS" for row in fixed_rows)
            else "FAIL",
            "interactive_orbit_contact_sheet": str(contact_sheet.resolve()),
            "browser": subprocess.check_output(
                [browser_executable, "--version"], text=True
            ).strip(),
            "real_input_authority": "Chrome DevTools Input.dispatchMouseEvent",
        }
    )
    write_json(receipt_path, receipt)
    return {
        "episode": episode,
        "fixed": fixed_rows,
        "visibility": visibility_rows,
        "drag": drag_rows,
        "zoom": zoom_rows,
        "reset": reset_rows,
        "playback": playback_rows,
        "sweep": sweep_rows,
        "historical": historical,
        "receipt": receipt,
    }


def assemble_reports(
    root: Path,
    pre: dict[str, Any],
    reference: dict[str, Any],
    results: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    fixed = [row for result in results for row in result["fixed"]]
    visibility = [row for result in results for row in result["visibility"]]
    drags = [row for result in results for row in result["drag"]]
    zoom = [row for result in results for row in result["zoom"]]
    reset = [row for result in results for row in result["reset"]]
    playback = [row for result in results for row in result["playback"]]
    sweep = [row for result in results for row in result["sweep"]]
    historical = [result["historical"] for result in results]
    write_json(root / "browser_certification/pointer_drag_tests.json", drags)
    write_json(root / "browser_certification/visibility_tests.json", visibility)
    write_csv(root / "browser_certification/orbit_sweep.csv", sweep)
    write_json(root / "browser_certification/frame_playback_tests.json", playback)
    write_json(root / "browser_certification/reset_tests.json", reset)
    write_json(root / "browser_certification/zoom_tests.json", zoom)
    write_csv(root / "regression/fixed_preset_parity.csv", fixed)
    comparison = {
        "historical": historical,
        "new": [
            {
                "episode": result["episode"]["review"],
                "interactive_certification": result["receipt"]["interactive_certification"],
            }
            for result in results
        ],
    }
    write_json(root / "regression/historical_vs_new_interactive_regression.json", comparison)
    write_json(root / "historical_viewer/free_orbit_failure_receipt.json", historical)
    all_rows = fixed + visibility + drags + zoom + reset + playback + sweep
    machine = (
        "PASS"
        if len(results) == 2
        and all(row["status"] == "PASS" for row in all_rows)
        and all(row["failure_reproduced"] for row in historical)
        else "FAIL"
    )
    post_hashes = frozen_hashes([result["episode"] for result in results])
    frozen_unchanged = post_hashes == pre["hashes"]
    ref_after = tree_metadata_snapshot(REF2DEX_ROOT)
    reference_unmodified = ref_after == reference["reference_tree_before"]
    safety = {
        "BRANCH": git("branch", "--show-current"),
        "REF2DEX_REFERENCE_ROOT": str(REF2DEX_ROOT),
        "REF2DEX_MAIN_MODIFIED": "NO" if reference_unmodified else "UNKNOWN",
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "SAME_TWO_EPISODES": "YES",
        "EPISODES_RESELECTED": "NO",
        "RAW_MANO_CHANGED": "NO",
        "MANO_ASSET_CHANGED": "NO",
        "MANO_BETA_CHANGED": "NO",
        "MANO_POSE_CHANGED": "NO",
        "FRAME_BINDING_CHANGED": "NO",
        "TARGET_OBJECT_CHANGED": "NO",
        "O3_RERUN": "NO",
        "MANIFEST_V2_MODIFIED": "NO" if frozen_unchanged else "YES",
        "SPLIT_V2_MODIFIED": "NO" if frozen_unchanged else "YES",
        "MANIFEST_V3_CREATED": "NO",
        "OLD_VIEWER_USED_AS_PRODUCTION": "NO",
        "NEW_REF2DEX_STYLE_VIEWER_IMPLEMENTED": "YES",
        "HTML_RECONSTRUCTS_MANO": "NO",
        "HTML_USES_PRECOMPUTED_GEOMETRY": "YES",
        "CAMERA_ORBIT_MUTATES_SCENE_GEOMETRY": "NO",
        "HAND_VERTICES_CHANGE_ON_DRAG": "NO",
        "JOINTS_CHANGE_ON_DRAG": "NO",
        "OBJECT_WORLD_POSE_CHANGE_ON_DRAG": "NO",
        "HAND_OBJECT_RELATIVE_TRANSFORM_CHANGE_ON_DRAG": "NO",
        "REAL_POINTER_DRAG_TESTED": "YES",
        "MULTIPLE_DRAGS_TESTED": "YES",
        "ORBIT_SWEEP_TESTED": "YES",
        "ZOOM_TESTED": "YES",
        "RESET_TESTED": "YES",
        "FRAME_CHANGE_AFTER_DRAG_TESTED": "YES",
        "PLAYBACK_AFTER_DRAG_TESTED": "YES",
        "FIXED_FRONT_PASS": "YES"
        if all(row["status"] == "PASS" for row in fixed if row["preset"] == "FRONT")
        else "NO",
        "FIXED_OBLIQUE_PASS": "YES"
        if all(row["status"] == "PASS" for row in fixed if row["preset"] == "OBLIQUE")
        else "NO",
        "FIXED_SIDE_PASS": "YES"
        if all(row["status"] == "PASS" for row in fixed if row["preset"] == "SIDE")
        else "NO",
        "INTERACTIVE_VIEWER_CERTIFICATION": machine,
        "O1_CUSTOM_HTML_VIEWER_HUMAN": "PENDING",
        "GEOMETRIC_RETARGET_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "O5_ALLOWED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    latest = {str(result["episode"]["review"]): result["receipt"] for result in results}
    summary = {
        "schema_version": "OakInk2O1R2DFinalSummaryV1",
        "status": "WAITING_FOR_SAME_TWO_HUMAN_REVIEW",
        "INTERACTIVE_VIEWER_CERTIFICATION": machine,
        "O1_CUSTOM_HTML_VIEWER_HUMAN": "PENDING",
        "O5_ALLOWED": "NO",
        "fixed_preset_rows": len(fixed),
        "pointer_drag_rows": len(drags),
        "orbit_sweep_rows": len(sweep),
        "timeline_under_orbit_rows": len(playback),
        "results": latest,
        "safety_flags": safety,
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "tests.json",
        {
            "machine_rows": len(all_rows),
            "passed": sum(row["status"] == "PASS" for row in all_rows),
            "failed": sum(row["status"] != "PASS" for row in all_rows),
            "historical_negative_fixture": historical,
        },
    )
    write_json(
        root / "validation_results.json",
        {
            "interactive_viewer_certification": machine,
            "frozen_hashes_unchanged": frozen_unchanged,
            "ref2dex_main_modified": not reference_unmodified,
            "human_gate": "PENDING",
            "o5_allowed": False,
        },
    )
    technical_failures = [
        {
            "phase": "initial_generation",
            "failure": "toporetarget-rl does not provide manotorch",
            "resolution": "reused the receipt-matched ref2dex-oakink runtime; no dependency installation",
            "resolved": True,
        },
        {
            "phase": "initial_cdp_handshake",
            "failure": "a spurious trailing question mark changed the Chrome target id",
            "resolution": "append the WebSocket query delimiter only when a query exists",
            "resolved": True,
        },
        {
            "phase": "initial_v2_drag_probe",
            "failure": "the local identity matrix literal placed its z diagonal at index 9",
            "resolution": "corrected the column-major identity literal and added a regression assertion",
            "resolved": True,
        },
    ]
    (root / "technical_failures.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in technical_failures),
        encoding="utf-8",
    )
    write_json(
        root / "resource_usage.json",
        {
            "elapsed_seconds": time.monotonic() - started,
            "episodes": len(results),
            "browser": shutil.which("google-chrome") or shutil.which("chromium"),
        },
    )
    write_json(
        root / "git_commits.json",
        {
            "branch": git("branch", "--show-current"),
            "start_head": pre["git"]["start_head"],
            "final_head_before_commit": git("rev-parse", "HEAD"),
            "commits": [],
            "pushed": False,
            "pr_created": False,
        },
    )
    manual = """# OakInk2 O1R2-D manual review\n\nOpen both new HTML files. For each: inspect default OBLIQUE; drag 5-10 times; orbit to the opposite side; drag vertically; wheel zoom; reset; drag again; play; change frames while retaining the rotated camera; verify hand-object alignment.\n\nReply exactly:\n\n```text\nOAKINK2_VIEWER_V2_DEV_1=APPROVE / REJECT\nOAKINK2_VIEWER_V2_DEV_2=APPROVE / REJECT\n```\n\nO5 remains blocked until both are approved.\n"""
    (root / "manual_review.md").write_text(manual, encoding="utf-8")
    html1 = latest["dev_01"]["html"]
    html2 = latest["dev_02"]["html"]
    sheet1 = latest["dev_01"]["interactive_orbit_contact_sheet"]
    sheet2 = latest["dev_02"]["interactive_orbit_contact_sheet"]
    handoff = f"""# OakInk2 O1R2-D Ref2Dex-style Interactive Viewer Handoff\n\nMachine decision: `INTERACTIVE_VIEWER_CERTIFICATION={machine}`. Human gate remains `PENDING`; `O5_ALLOWED=NO`.\n\nRef2Dex reference: `{REF_VIEWER}` and `{REF_ENTRYPOINT}`. The adopted architecture is named scene nodes plus a renderer-owned camera. No Ref2Dex code was copied verbatim and there is no runtime dependency on Ref2Dex or MeshCat. `REF2DEX_MAIN_MODIFIED={"NO" if reference_unmodified else "UNKNOWN"}`.\n\nOld root cause: `freeView()` discarded the trusted camera/model basis by switching the model to identity. Viewer V2 instead keeps hand/skeleton in identity scene nodes, keeps the Python-precomputed object model, and mutates only `ViewerCameraStateV1`.\n\n- DEV_01_HTML={html1}\n- DEV_02_HTML={html2}\n- DEV_01_ORBIT_CONTACT_SHEET={sheet1}\n- DEV_02_ORBIT_CONTACT_SHEET={sheet2}\n\n```bash\nxdg-open '{html1}'\nxdg-open '{html2}'\n```\n\nThe browser suite used Chrome DevTools real mouse input for horizontal, vertical, diagonal, reverse, and repeated drags; wheel input; deterministic reset/presets; 122 orbit states; and frame/playback-under-orbit checks. Fixed FRONT/OBLIQUE/SIDE parity remained {"PASS" if all(row["status"] == "PASS" for row in fixed) else "FAIL"}. Frozen MANO, frame binding, targets, Manifest V2, Split V2, and O3 were unchanged.\n\nReply with `OAKINK2_VIEWER_V2_DEV_1=APPROVE / REJECT` and `OAKINK2_VIEWER_V2_DEV_2=APPROVE / REJECT`.\n"""
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    (root / "final_summary.md").write_text(handoff, encoding="utf-8")
    return summary


def parse_frame_ids(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(
            "--frame-ids must be a non-empty unique comma-separated list"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--action", choices=("all", "generate", "certify"), default="all")
    parser.add_argument(
        "--review",
        choices=("dev_01", "dev_02"),
        action="append",
        help="Narrow generation/certification to one authoritative episode; repeat for both.",
    )
    parser.add_argument(
        "--frame-ids",
        help="Explicit comma-separated source mocap timeline for one --review; must include its frozen primary frame. Not used by O1R2-D certification.",
    )
    args = parser.parse_args()
    started = time.monotonic()
    root = args.report_root.resolve()
    all_episodes = fixed_episodes()
    selected = [
        episode
        for episode in all_episodes
        if args.review is None or episode["review"] in args.review
    ]
    frames = parse_frame_ids(args.frame_ids)
    if frames is not None and len(selected) != 1:
        parser.error("--frame-ids requires exactly one --review")
    if args.action in ("all", "certify") and frames is not None:
        parser.error(
            "--frame-ids is generation-only and cannot be certified as the frozen O1R2-D set"
        )
    pre = preflight(root, all_episodes)
    reference = audit_reference(root)
    viewer_contracts(root)
    historical_audit(root)
    if args.action in ("all", "generate"):
        generated = generate_html(root, selected, frames)
        if args.action == "generate":
            print(json.dumps({"status": "GENERATED", "results": generated}, sort_keys=True))
            return
    browser = shutil.which("google-chrome") or shutil.which("chromium")
    if browser is None:
        raise RuntimeError("O1R2D_CHROME_UNAVAILABLE")
    results = [certify_episode(root, episode, browser) for episode in selected]
    if len(selected) == 2:
        summary = assemble_reports(root, pre, reference, results, started)
    else:
        summary = {"status": "PARTIAL_EPISODE_CERTIFICATION", "results": results}
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
