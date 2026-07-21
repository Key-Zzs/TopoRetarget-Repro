"""Review-frame selection and no-solver visual bundle generation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .schema import stable_hash, write_json


def choose_review_frames(
    *, final: str | Path, selected_window: dict[str, Any], warm_start: str | Path | None = None
) -> dict[str, Any]:
    from toporetarget.retarget.final_refinement import load_final_trajectory

    artifact = load_final_trajectory(final)
    count = artifact.frame_count
    values: dict[str, int] = {
        "first": 0,
        "middle": max(0, count // 2),
        "last": max(0, count - 1),
        "sdf_min": int(np.argmin(artifact.arrays["min_full_signed_distance"])),
        "penetration_max": int(np.argmax(artifact.arrays["max_penetration"])),
        "slack_max": int(np.argmax(artifact.arrays["e_slack"])),
        "e_im_max": int(np.argmax(artifact.arrays["e_im"])),
        "e_bone_max": int(np.argmax(artifact.arrays["e_bone"])),
        "base_correction_max": int(
            np.argmax(np.linalg.norm(artifact.arrays["base_corrections"][:, :3], axis=1))
        ),
        "qpos_correction_max": 0,
        "slowest_solve": int(np.argmax(artifact.arrays["solve_time_s"])),
    }
    if warm_start is not None:
        from toporetarget.retarget.artifacts import load_warm_start

        warm = load_warm_start(warm_start)
        count = min(artifact.frame_count, warm.frame_count)
        if count:
            correction = np.linalg.norm(
                np.asarray(artifact.arrays["qpos"][:count])
                - np.asarray(warm.arrays["qpos"][:count]),
                axis=1,
            )
            values["qpos_correction_max"] = int(np.argmax(correction))
    start = int(selected_window.get("start_frame", 0))
    contact_frames = selected_window.get("contact_frames", [])
    if contact_frames:
        local = [int(frame) - start for frame in contact_frames]
        local = [frame for frame in local if 0 <= frame < count]
        if local:
            values["contact_max"] = int(
                local[
                    np.argmax(
                        [
                            selected_window.get("contact_counts", {}).get(str(frame + start), 0)
                            for frame in local
                        ]
                    )
                ]
            )
    deduplicated: list[int] = []
    for frame in values.values():
        if frame not in deduplicated and 0 <= frame < count:
            deduplicated.append(frame)
    return {
        "schema_version": "toporetarget.review_frames.v1",
        "named_frames": values,
        "unique_frames": deduplicated,
        "frame_count": count,
        "hash": stable_hash({"named_frames": values, "unique_frames": deduplicated}),
    }


def _command(
    manifest: dict[str, Any], *, frame: int | None = None, output: Path | None = None
) -> list[str]:
    artifacts = manifest["artifacts"]
    command = [
        sys.executable,
        "-m",
        "toporetarget",
        "retarget",
        "visualize-refinement",
        "--canonical",
        artifacts["canonical"]["path"],
        "--warm-start",
        artifacts["warm_start"]["path"],
        "--graph",
        artifacts["graph"]["path"],
        "--final",
        artifacts["final"]["path"],
        "--collision-samples",
        artifacts["collision_samples"]["path"],
        "--robot",
        manifest["robot"],
        "--show-source-hand",
        "--show-warm-start",
        "--show-final",
        "--show-object",
        "--show-interaction-edges",
        "--show-collision-samples",
        "--show-query-set",
        "--show-penetrations",
        "--show-slack",
    ]
    if manifest.get("asset_root"):
        command += ["--asset-root", str(manifest["asset_root"])]
    if frame is None:
        command += [
            "--interactive",
            "--start-frame",
            "0",
            "--end-frame",
            str(manifest["selected_frame_range"][1] - manifest["selected_frame_range"][0]),
        ]
    else:
        command += ["--frame", str(frame), "--output", str(output)]
    return command


def generate_review_bundle(
    *,
    manifest: dict[str, Any],
    final: str | Path,
    selected_window: dict[str, Any],
    review_root: str | Path,
) -> dict[str, Any]:
    root = Path(review_root)
    root.mkdir(parents=True, exist_ok=True)
    frames = choose_review_frames(
        final=final,
        selected_window=selected_window,
        warm_start=manifest["artifacts"]["warm_start"]["path"],
    )
    named = frames["named_frames"]
    frame_files: dict[str, str] = {}
    for name, frame in named.items():
        target = root / f"{name}.png"
        command = _command(manifest, frame=frame, output=target)
        result = subprocess.run(
            command, cwd=Path(manifest["repo_root"]), text=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"review render failed for {name}: {result.stderr[-1000:]}")
        frame_files[name] = str(target)
    gif_path = root / "trajectory.gif"
    try:
        from PIL import Image

        frame_count = int(manifest["selected_frame_range"][1]) - int(
            manifest["selected_frame_range"][0]
        )
        with tempfile.TemporaryDirectory(prefix="toporetarget_stage10_review_") as temp_root:
            images: list[Any] = []
            for frame in range(frame_count):
                target = Path(temp_root) / f"frame_{frame:06d}.png"
                command = _command(manifest, frame=frame, output=target)
                result = subprocess.run(
                    command, cwd=Path(manifest["repo_root"]), text=True, capture_output=True
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"review GIF render failed for frame {frame}: {result.stderr[-1000:]}"
                    )
                with Image.open(target) as image:
                    images.append(image.convert("RGB"))
            if images:
                images[0].save(
                    gif_path, save_all=True, append_images=images[1:], duration=120, loop=0
                )
            for image in images:
                image.close()
    except ImportError:
        gif_path = root / "trajectory.gif.unavailable"
        gif_path.write_text("Pillow is required for GIF export.\n", encoding="utf-8")
    manual_template = {
        "schema_version": "toporetarget.manual_acceptance.v1",
        "status": "pending_human_review",
        "reviewer": "",
        "reviewed_frames": [
            0,
            max(
                0,
                int(manifest["selected_frame_range"][1] - manifest["selected_frame_range"][0]) // 2,
            ),
            max(
                0,
                int(manifest["selected_frame_range"][1] - manifest["selected_frame_range"][0]) - 1,
            ),
        ],
        "current_window_interpretation": None,
        "source_object_alignment": None,
        "warm_start_object_alignment": None,
        "final_object_alignment": None,
        "sdf_visual_consistency": None,
        "right_left_semantics": None,
        "no_visible_discontinuity": None,
        "no_visible_mirroring": None,
        "no_unexplained_scale_error": None,
        "contact_rich_clip_validated": False,
        "notes": ["Codex does not fill human acceptance."],
    }
    write_json(manual_template, root / "manual_acceptance.template.json")
    interactive = " ".join(_command(manifest))
    (root / "visualize_command.txt").write_text(interactive + "\n", encoding="utf-8")
    review = {
        "root": str(root),
        "frames": frames,
        "frame_files": frame_files,
        "gif": str(gif_path),
        "visualize_command": interactive,
        "manual_acceptance_template": str(root / "manual_acceptance.template.json"),
    }
    write_json(review, root / "review_frames.json")
    return review


__all__ = ["choose_review_frames", "generate_review_bundle"]
