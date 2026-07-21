"""Manifest-driven wrapper around the existing Stage 9 viewer."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .schema import read_json, write_json


def viewer_command(
    manifest: dict[str, Any],
    *,
    frame: int | None = None,
    output: str | Path | None = None,
    interactive: bool = False,
    view: str = "scene",
    start_frame: int | None = None,
    end_frame: int | None = None,
    show_source_hand: bool = True,
    show_warm_start: bool = True,
    show_final: bool = True,
    show_object: bool = True,
    show_interaction_edges: bool = True,
    show_collision_samples: bool = True,
    show_query_set: bool = True,
    show_penetrations: bool = True,
    show_slack: bool = True,
) -> list[str]:
    if view not in {"scene", "object"}:
        raise ValueError("view must be scene or object")
    artifacts = manifest["artifacts"]
    selected_start, selected_end = manifest["selected_frame_range"]
    frame_count = int(selected_end - selected_start)
    local_start = 0 if start_frame is None else int(start_frame)
    local_end = frame_count if end_frame is None else int(end_frame)
    if local_start < 0 or local_end <= local_start or local_end > frame_count:
        raise ValueError(f"viewer frame range must be within [0,{frame_count})")
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
        "--view",
        view,
        "--show-source-hand" if show_source_hand else "--hide-source-hand",
        "--show-warm-start" if show_warm_start else "--hide-warm-start",
        "--show-final" if show_final else "--hide-final",
        "--show-object" if show_object else "--hide-object",
        "--show-interaction-edges" if show_interaction_edges else "--hide-interaction-edges",
        "--show-collision-samples" if show_collision_samples else "--hide-collision-samples",
        "--show-query-set" if show_query_set else "--hide-query-set",
        "--show-penetrations" if show_penetrations else "--hide-penetrations",
        "--show-slack" if show_slack else "--hide-slack",
    ]
    asset_root = manifest.get("asset_root")
    if asset_root:
        command[command.index("--robot") + 2 : command.index("--robot") + 2] = [
            "--asset-root",
            str(asset_root),
        ]
    if interactive:
        if output is not None:
            raise ValueError("interactive visualization does not accept --output")
        command += [
            "--interactive",
            "--start-frame",
            str(local_start),
            "--end-frame",
            str(local_end),
        ]
    else:
        selected_frame = local_start if frame is None else int(frame)
        if not 0 <= selected_frame < frame_count:
            raise ValueError(f"viewer frame must be within [0,{frame_count})")
        command += ["--frame", str(selected_frame)]
        if output is None:
            raise ValueError("static visualization requires output")
        command += ["--output", str(output)]
    return command


def run_visualization(
    manifest_path: str | Path,
    *,
    frame: int | None = None,
    view: str = "scene",
    start_frame: int | None = None,
    end_frame: int | None = None,
    display_stride: int = 1,
    output: str | Path | None = None,
    interactive: bool = False,
    flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    if display_stride <= 0:
        raise ValueError("display_stride must be positive")
    manifest = read_json(manifest_path)
    options = flags or {}
    if interactive:
        command = viewer_command(
            manifest,
            frame=frame,
            output=output,
            interactive=True,
            view=view,
            start_frame=start_frame,
            end_frame=end_frame,
            **{key: value for key, value in options.items() if value is not None},
        )
        result = subprocess.run(
            command, cwd=Path(manifest["repo_root"]), text=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"viewer failed: {result.stderr[-2000:]}")
        commands = [" ".join(command)]
        stdout = result.stdout[-4000:]
        rendered_frames: list[int] = []
    elif output is not None and Path(output).suffix.lower() == ".gif":
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        selected_start, selected_end = manifest["selected_frame_range"]
        frame_count = int(selected_end - selected_start)
        local_start = 0 if start_frame is None else int(start_frame)
        local_end = frame_count if end_frame is None else int(end_frame)
        if frame is not None:
            local_start, local_end = int(frame), int(frame) + 1
        if local_start < 0 or local_end <= local_start or local_end > frame_count:
            raise ValueError(f"GIF frame range must be within [0,{frame_count})")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("GIF visualization requires Pillow") from exc
        commands = []
        rendered_frames = list(range(local_start, local_end, display_stride))
        if not rendered_frames:
            raise ValueError("GIF frame range produced no frames")
        with tempfile.TemporaryDirectory(prefix="toporetarget_stage10_gif_") as temp_root:
            images: list[Any] = []
            for item in rendered_frames:
                frame_path = Path(temp_root) / f"frame_{item:06d}.png"
                command = viewer_command(
                    manifest,
                    frame=item,
                    output=frame_path,
                    interactive=False,
                    view=view,
                    start_frame=local_start,
                    end_frame=local_end,
                    **{key: value for key, value in options.items() if value is not None},
                )
                result = subprocess.run(
                    command, cwd=Path(manifest["repo_root"]), text=True, capture_output=True
                )
                if result.returncode != 0:
                    raise RuntimeError(f"viewer failed for frame {item}: {result.stderr[-2000:]}")
                commands.append(" ".join(command))
                with Image.open(frame_path) as image:
                    images.append(image.convert("RGB"))
            images[0].save(
                destination,
                save_all=True,
                append_images=images[1:],
                duration=120,
                loop=0,
            )
            for image in images:
                image.close()
        stdout = ""
    else:
        selected_frame = 0 if frame is None else frame
        command = viewer_command(
            manifest,
            frame=selected_frame,
            output=output,
            interactive=False,
            view=view,
            start_frame=start_frame,
            end_frame=end_frame,
            **{key: value for key, value in options.items() if value is not None},
        )
        result = subprocess.run(
            command, cwd=Path(manifest["repo_root"]), text=True, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"viewer failed: {result.stderr[-2000:]}")
        commands = [" ".join(command)]
        stdout = result.stdout[-4000:]
        rendered_frames = [selected_frame]
    return {
        "status": "pass",
        "manifest": str(manifest_path),
        "command": commands[0] if len(commands) == 1 else commands,
        "interactive": interactive,
        "frame": frame,
        "view": view,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "display_stride": display_stride,
        "rendered_frames": rendered_frames,
        "output": None if output is None else str(output),
        "stdout": stdout,
    }


def write_visualization_report(result: dict[str, Any], path: str | Path) -> Path:
    return write_json(result, path)


__all__ = ["run_visualization", "viewer_command", "write_visualization_report"]
