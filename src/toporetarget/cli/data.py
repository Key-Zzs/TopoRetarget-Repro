"""Explicit, sequence-scoped data commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.storage import StorageError, load_hoi_sequence, save_hoi_sequence
from toporetarget.data.synthetic import SyntheticAdapter
from toporetarget.viz.comparison import ComparisonMetrics
from toporetarget.viz.matplotlib_viewer import ViewerOptions, render_comparison, show_comparison

app = typer.Typer(help="Inspect and convert one explicitly selected HOI sequence.")


def _json_print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _range(start_frame: int, end_frame: int | None) -> FrameRange | None:
    if start_frame == 0 and end_frame is None:
        return None
    return FrameRange(start=start_frame, end=end_frame)


def _synthetic(sequence: str, frame_range: FrameRange | None = None):
    return SyntheticAdapter().load_raw_renderable(sequence, frame_range=frame_range)


def _load_raw(
    dataset: str,
    sequence: str,
    *,
    sequence_path: Path | None,
    hand: str,
    grab_root: Path | None,
    mano_model_root: Path | None,
    frame_range: FrameRange | None,
):
    if dataset == "synthetic":
        return _synthetic(sequence, frame_range)
    if dataset == "grab":
        try:
            from toporetarget.data.adapters.grab_inspect import GrabInspectionAdapter
        except ImportError as exc:  # pragma: no cover - Stage 2B optional module
            raise typer.BadParameter(str(exc)) from exc
        if sequence_path is None:
            raise typer.BadParameter("--sequence-path is required for --dataset grab")
        adapter = GrabInspectionAdapter(
            sequence_path=sequence_path,
            mano_model_root=mano_model_root,
            grab_root=grab_root,
            hand=hand,
        )
        return adapter.load_raw_renderable(sequence_path.name, frame_range=frame_range)
    raise typer.BadParameter(f"unsupported dataset: {dataset}; use synthetic or grab")


@app.command("make-synthetic")
def make_synthetic(
    output: Path = typer.Option(..., "--output", help="Explicit Zarr cache destination."),
    num_frames: int = typer.Option(8, "--num-frames", min=1),
    irregular_timestamps: bool = typer.Option(False, "--irregular-timestamps"),
) -> None:
    """Create one deterministic synthetic sequence cache."""

    sequence = SyntheticAdapter().load_sequence(
        "demo", irregular_timestamps=irregular_timestamps, num_frames=num_frames
    )
    try:
        save_hoi_sequence(sequence, output)
    except (StorageError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    _json_print(
        {
            "output": str(output),
            "sequence": sequence.metadata.sequence_id,
            "num_frames": sequence.num_frames,
        }
    )


@app.command("describe")
def describe(
    dataset: str = typer.Option("synthetic", "--dataset"),
    sequence: str = typer.Option("demo", "--sequence"),
    sequence_path: Path | None = typer.Option(None, "--sequence-path"),
    hand: str = typer.Option("right", "--hand"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
) -> None:
    """Describe one sequence without scanning a dataset."""

    if dataset == "synthetic":
        _json_print(SyntheticAdapter().describe_sequence(sequence))
        return
    if dataset == "grab":
        from toporetarget.data.adapters.grab_inspect import GrabInspectionAdapter

        adapter = GrabInspectionAdapter(
            sequence_path=sequence_path or "",
            hand=hand,
            grab_root=grab_root,
            mano_model_root=mano_model_root,
        )
        try:
            _json_print(adapter.describe_sequence(sequence))
        except (RuntimeError, ValueError, OSError) as exc:
            typer.echo(f"description failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        return
    raw = _load_raw(
        dataset,
        sequence,
        sequence_path=sequence_path,
        hand=hand,
        grab_root=grab_root,
        mano_model_root=mano_model_root,
        frame_range=None,
    )
    _json_print(
        {
            "dataset_name": raw.metadata.dataset_name,
            "sequence_id": raw.metadata.sequence_id,
            "num_frames": raw.num_frames,
            "native_fps": raw.metadata.native_fps,
            "timestamps": raw.timestamps.tolist(),
        }
    )


@app.command("convert")
def convert(
    dataset: str = typer.Option("synthetic", "--dataset"),
    sequence: str = typer.Option("demo", "--sequence"),
    sequence_path: Path | None = typer.Option(None, "--sequence-path"),
    hand: str = typer.Option("right", "--hand"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    start_frame: int = typer.Option(0, "--start-frame", min=0, help="Inclusive clip start."),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1, help="Exclusive clip end."),
    output: Path | None = typer.Option(
        None, "--output", help="Optional explicit Zarr cache destination."
    ),
) -> None:
    """Convert exactly one sequence or contiguous clip; no resampling is performed."""

    try:
        raw = _load_raw(
            dataset,
            sequence,
            sequence_path=sequence_path,
            hand=hand,
            grab_root=grab_root,
            mano_model_root=mano_model_root,
            frame_range=_range(start_frame, end_frame),
        )
        canonical = raw
        if output is not None:
            save_hoi_sequence(canonical, output)
        _json_print(
            {
                "dataset": dataset,
                "sequence_id": canonical.metadata.sequence_id,
                "num_frames": canonical.num_frames,
                "native_fps": canonical.metadata.native_fps,
                "timestamps": canonical.timestamps.tolist(),
                "output": None if output is None else str(output),
                "no_temporal_resampling": True,
                "no_spatial_sampling": True,
            }
        )
    except (StorageError, RuntimeError, ValueError, OSError, typer.BadParameter) as exc:
        typer.echo(f"conversion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("inspect")
def inspect(
    input: Path = typer.Option(..., "--input"),
    frame: int = typer.Option(0, "--frame", min=0),
    output: Path | None = typer.Option(None, "--output"),
    show: bool = typer.Option(False, "--show"),
) -> None:
    """Inspect one canonical cache without loading any other data."""

    try:
        sequence = load_hoi_sequence(input)
        if frame >= sequence.num_frames:
            raise ValueError(f"frame {frame} is outside [0, {sequence.num_frames})")
        _json_print(
            {
                "schema_version": sequence.metadata.schema_version,
                "dataset_name": sequence.metadata.dataset_name,
                "sequence_id": sequence.metadata.sequence_id,
                "num_frames": sequence.num_frames,
                "native_fps": sequence.metadata.native_fps,
                "frame": frame,
                "timestamp": float(sequence.timestamps[frame]),
                "hands": [hand.hand_id for hand in sequence.hands],
                "rigid_objects": [obj.object_id for obj in sequence.rigid_objects],
            }
        )
        if output is not None or show:
            result = ComparisonMetrics.compute(sequence, sequence)
            options = ViewerOptions(layout="overlay", frame=frame)
            if show:
                show_comparison(sequence, sequence, result, options=options)
            elif output is not None:
                render_comparison(sequence, sequence, result, options=options, output=output)
    except (StorageError, RuntimeError, ValueError) as exc:
        typer.echo(f"inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare")
def compare(
    dataset: str = typer.Option("synthetic", "--dataset"),
    sequence: str = typer.Option("demo", "--sequence"),
    sequence_path: Path | None = typer.Option(None, "--sequence-path"),
    hand: str = typer.Option("right", "--hand"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    canonical: Path = typer.Option(..., "--canonical"),
    layout: str = typer.Option("side-by-side", "--layout"),
    frame: int = typer.Option(0, "--frame", min=0),
    start_frame: int | None = typer.Option(
        None, "--start-frame", min=0, help="Display range start."
    ),
    end_frame: int | None = typer.Option(
        None, "--end-frame", min=1, help="Exclusive display range end."
    ),
    display_stride: int = typer.Option(
        1, "--display-stride", min=1, help="Display-only stride; data is unchanged."
    ),
    show: bool = typer.Option(False, "--show"),
    output: Path | None = typer.Option(None, "--output"),
    raw_color: str = typer.Option("tab:blue", "--raw-color"),
    canonical_color: str = typer.Option("tab:orange", "--canonical-color"),
    show_scene_frame: bool = typer.Option(False, "--show-scene-frame"),
    show_wrist_frame: bool = typer.Option(False, "--show-wrist-frame"),
    show_object_frame: bool = typer.Option(False, "--show-object-frame"),
    show_keypoints: bool = typer.Option(False, "--show-keypoints"),
    show_mesh: bool = typer.Option(True, "--show-mesh/--hide-mesh"),
    error_json: Path | None = typer.Option(None, "--error-json"),
    error_csv: Path | None = typer.Option(None, "--error-csv"),
) -> None:
    """Compare a raw sequence with a separately loaded canonical cache."""

    try:
        canonical_sequence = load_hoi_sequence(canonical)
        source_start = 0 if start_frame is None else start_frame
        source_end = canonical_sequence.num_frames if end_frame is None else end_frame
        raw = _load_raw(
            dataset,
            sequence,
            sequence_path=sequence_path,
            hand=hand,
            grab_root=grab_root,
            mano_model_root=mano_model_root,
            frame_range=FrameRange(source_start, source_end),
        )
        result = ComparisonMetrics.compute(raw, canonical_sequence)
        if error_json is not None:
            result.write_json(error_json)
        if error_csv is not None:
            result.write_csv(error_csv)
        selected_frame = frame
        if start_frame is not None:
            selected_frame = start_frame
        if display_stride > 1:
            typer.echo(
                f"display_stride={display_stride} affects display only; canonical data is unchanged"
            )
        options = ViewerOptions(
            layout=layout,
            frame=selected_frame,
            raw_color=raw_color,
            canonical_color=canonical_color,
            show_scene_frame=show_scene_frame,
            show_wrist_frame=show_wrist_frame,
            show_object_frame=show_object_frame,
            show_keypoints=show_keypoints,
            show_mesh=show_mesh,
        )
        if show:
            show_comparison(raw, canonical_sequence, result, options=options)
        elif output is not None:
            render_comparison(raw, canonical_sequence, result, options=options, output=output)
        _json_print(
            {
                "summary": result.as_dict()["metrics"],
                "output": None if output is None else str(output),
            }
        )
    except (StorageError, RuntimeError, ValueError, OSError, typer.BadParameter) as exc:
        typer.echo(f"comparison failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
