"""Explicit, sequence-scoped data commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.adapters.grab import GrabAdapterError, GrabDatasetAdapter, GrabLoadOptions
from toporetarget.data.indexes.grab import GrabIndexError, build_grab_index, load_grab_index
from toporetarget.data.storage import StorageError, load_hoi_sequence, save_hoi_sequence
from toporetarget.data.synthetic import SyntheticAdapter
from toporetarget.data.validation.grab import validate_grab_sequence
from toporetarget.viz.comparison import ComparisonMetrics
from toporetarget.viz.grab_viewer import GrabViewerOptions, render_grab_view
from toporetarget.viz.matplotlib_viewer import ViewerOptions, render_comparison, show_comparison

app = typer.Typer(help="Inspect and convert one explicitly selected HOI sequence.")


def _json_print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _range(start_frame: int, end_frame: int | None) -> FrameRange | None:
    if start_frame == 0 and end_frame is None:
        return None
    return FrameRange(start=start_frame, end=end_frame)


def _grab_adapter(
    *,
    sequence_path: Path | None = None,
    grab_root: Path | None = None,
    index: Path | None = None,
    mano_model_root: Path | None = None,
    hands: str = "auto",
    include_table: bool = True,
    contact_mode: str = "source",
    include_mediapipe21: bool = True,
) -> GrabDatasetAdapter:
    return GrabDatasetAdapter(
        sequence_path=sequence_path,
        grab_root=grab_root,
        index=index,
        mano_model_root=mano_model_root,
        options=GrabLoadOptions(
            hands=hands,
            include_table=include_table,
            contact_mode=contact_mode,
            include_mediapipe21=include_mediapipe21,
        ),
    )


def _synthetic(sequence: str, frame_range: FrameRange | None = None):
    return SyntheticAdapter().load_raw_renderable(sequence, frame_range=frame_range)


def _hocap_adapter(*, data_root: Path | None, mano_model_root: Path | None):
    """Construct the existing one-sequence HOCap adapter for public conversion.

    This deliberately exposes no dataset-wide operation.  The adapter already
    validates right-hand MANO calibration and every declared object mesh; the
    CLI merely makes that authoritative conversion reachable from the same
    ``data convert`` boundary used by raw-to-retarget production callers.
    """

    from toporetarget.adapters.datasets.hocap import HOCapAdapterV1

    return HOCapAdapterV1(data_root=data_root, mano_model_root=mano_model_root)


def _load_raw(
    dataset: str,
    sequence: str,
    *,
    sequence_path: Path | None,
    hand: str,
    grab_root: Path | None,
    mano_model_root: Path | None,
    frame_range: FrameRange | None,
    data_root: Path | None = None,
    primary_object: str | None = None,
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
    if dataset == "hocap":
        if primary_object is None:
            raise typer.BadParameter("--primary-object is required for --dataset hocap")
        return _hocap_adapter(data_root=data_root, mano_model_root=mano_model_root).load_sequence(
            sequence,
            frame_range=frame_range,
            hand=hand,
            primary_object_id=primary_object,
        )
    raise typer.BadParameter(f"unsupported dataset: {dataset}; use synthetic, grab, or hocap")


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
    index: Path | None = typer.Option(None, "--index"),
    hand: str = typer.Option("right", "--hand"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    data_root: Path | None = typer.Option(None, "--data-root"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
) -> None:
    """Describe one sequence without scanning a dataset."""

    if dataset == "synthetic":
        _json_print(SyntheticAdapter().describe_sequence(sequence))
        return
    if dataset == "grab":
        adapter = _grab_adapter(
            sequence_path=sequence_path,
            index=index,
            grab_root=grab_root,
            mano_model_root=mano_model_root,
            hands=hand,
        )
        try:
            _json_print(adapter.describe_sequence(sequence if sequence_path is None else ""))
        except (GrabAdapterError, RuntimeError, ValueError, OSError) as exc:
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
        data_root=data_root,
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
    index: Path | None = typer.Option(None, "--index"),
    hand: str = typer.Option("right", "--hand"),
    hands: str | None = typer.Option(None, "--hands", help="auto, right, left, or both."),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    data_root: Path | None = typer.Option(
        None, "--data-root", help="Dataset storage root; HOCap is under <root>/HOCap."
    ),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    primary_object: str | None = typer.Option(
        None, "--primary-object", help="Required manipulation object when converting HOCap."
    ),
    include_table: bool = typer.Option(True, "--include-table/--no-table"),
    contact_mode: str = typer.Option("source", "--contact-mode"),
    include_mediapipe21: bool = typer.Option(True, "--include-mediapipe21/--no-mediapipe21"),
    start_frame: int = typer.Option(0, "--start-frame", min=0, help="Inclusive clip start."),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1, help="Exclusive clip end."),
    output: Path | None = typer.Option(
        None, "--output", help="Optional explicit Zarr cache destination."
    ),
    force: bool = typer.Option(False, "--force", help="Replace an existing explicit cache."),
) -> None:
    """Convert exactly one sequence or contiguous clip; no resampling is performed."""

    try:
        selected_hands = hands or hand
        if dataset == "grab":
            adapter = _grab_adapter(
                sequence_path=sequence_path,
                index=index,
                grab_root=grab_root,
                mano_model_root=mano_model_root,
                hands=selected_hands,
            )
            if output is None:
                canonical = adapter.load_sequence(
                    sequence,
                    frame_range=_range(start_frame, end_frame),
                    options=GrabLoadOptions(
                        hands=selected_hands,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        include_table=include_table,
                        contact_mode=contact_mode,
                        include_mediapipe21=include_mediapipe21,
                    ),
                )
            else:
                destination = adapter.create_cache(
                    sequence,
                    output=output,
                    force=force,
                    options=GrabLoadOptions(
                        hands=selected_hands,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        include_table=include_table,
                        contact_mode=contact_mode,
                        include_mediapipe21=include_mediapipe21,
                    ),
                )
                canonical = load_hoi_sequence(destination)
            _json_print(
                {
                    "dataset": dataset,
                    "sequence_id": canonical.metadata.sequence_id,
                    "num_frames": canonical.num_frames,
                    "native_fps": canonical.metadata.native_fps,
                    "hands": [item.side for item in canonical.hands],
                    "timestamps": canonical.timestamps.tolist(),
                    "output": None if output is None else str(output),
                    "no_temporal_resampling": True,
                    "no_spatial_sampling": True,
                }
            )
            return
        raw = _load_raw(
            dataset,
            sequence,
            sequence_path=sequence_path,
            hand=hand,
            grab_root=grab_root,
            mano_model_root=mano_model_root,
            frame_range=_range(start_frame, end_frame),
            data_root=data_root,
            primary_object=primary_object,
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


@app.command("index")
def index_dataset(
    dataset: str = typer.Option("grab", "--dataset"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    output: Path = typer.Option(Path(".local/index/grab"), "--output"),
    hash_files: bool = typer.Option(
        False, "--hash-files", help="Opt in to source SHA-256 hashing."
    ),
) -> None:
    """Build a lightweight GRAB index without MANO or frame-array loading."""

    if dataset != "grab":
        raise typer.BadParameter("only --dataset grab has an indexer")
    try:
        result = build_grab_index(grab_root=grab_root, output=output, hash_files=hash_files)
    except (GrabIndexError, OSError, ValueError) as exc:
        typer.echo(f"index failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _json_print(
        {
            "index": result["index"],
            "manifest": result["manifest"],
            "file_count": len(result["entries"]),
        }
    )


@app.command("list")
def list_dataset(
    dataset: str = typer.Option("grab", "--dataset"),
    index: Path = typer.Option(Path(".local/index/grab"), "--index"),
    subject: str | None = typer.Option(None, "--subject"),
    object_name: str | None = typer.Option(None, "--object"),
    action: str | None = typer.Option(None, "--action"),
    sequence: str | None = typer.Option(None, "--sequence"),
    contains: str | None = typer.Option(None, "--contains"),
    limit: int = typer.Option(20, "--limit", min=1),
    as_json: bool = typer.Option(False, "--json"),
    as_csv: bool = typer.Option(False, "--csv"),
) -> None:
    """List indexed sequences with bounded output; no MANO is loaded."""

    if dataset != "grab":
        raise typer.BadParameter("only --dataset grab is indexed")
    try:
        entries = load_grab_index(index)
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(
            f"index is unavailable: {index}; run `toporetarget data index --dataset grab` ({exc})",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    def matches(item: dict[str, object]) -> bool:
        values = {
            "subject": item.get("subject_id", ""),
            "object": item.get("object_token", ""),
            "action": item.get("action_token", ""),
            "sequence": item.get("sequence_id", ""),
        }
        return all(
            value is None or str(values[key]) == value
            for key, value in (
                ("subject", subject),
                ("object", object_name),
                ("action", action),
                ("sequence", sequence),
            )
        ) and (contains is None or contains in str(item.get("sequence_id", "")))

    selected = [item for item in entries if matches(item)][:limit]
    if as_json:
        _json_print(selected)
    elif as_csv:
        import csv
        import sys

        fields = [
            "sequence_id",
            "subject_id",
            "object_token",
            "action_token",
            "repetition_token",
            "relative_path",
            "file_size",
            "mtime_ns",
            "metadata_quality",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field) for field in fields} for item in selected)
    else:
        typer.echo("sequence_id\tsubject\tobject\taction\tframes metadata")
        for item in selected:
            typer.echo(
                f"{item.get('sequence_id')}\t{item.get('subject_id')}\t{item.get('object_token')}\t{item.get('action_token')}\t{item.get('metadata_quality')}"
            )
        typer.echo(
            f"showing {len(selected)} indexed sequence(s); use --json/--csv for machine output"
        )


@app.command("validate")
def validate_dataset(
    dataset: str = typer.Option("grab", "--dataset"),
    sequence: str = typer.Option("", "--sequence"),
    sequence_path: Path | None = typer.Option(None, "--sequence-path"),
    index: Path | None = typer.Option(None, "--index"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    hands: str = typer.Option("both", "--hands"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    contact_mode: str = typer.Option("source", "--contact-mode"),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    report: Path | None = typer.Option(None, "--report"),
    csv_output: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Validate one selected source sequence and optional canonical cache."""

    if dataset != "grab":
        raise typer.BadParameter("only --dataset grab has formal validation")
    try:
        adapter = _grab_adapter(
            sequence_path=sequence_path,
            grab_root=grab_root,
            index=index,
            mano_model_root=mano_model_root,
            hands=hands,
            contact_mode=contact_mode,
        )
        if canonical is not None and end_frame is None:
            end_frame = load_hoi_sequence(canonical).num_frames
        loaded = adapter.load_sequence(
            sequence,
            options=GrabLoadOptions(
                hands=hands,
                contact_mode=contact_mode,
                start_frame=start_frame,
                end_frame=end_frame,
            ),
        )
        payload = validate_grab_sequence(
            loaded, canonical=canonical, source_path=loaded.metadata.provenance.source_file
        )
        if report is not None:
            Path(report).parent.mkdir(parents=True, exist_ok=True)
            Path(report).write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
            )
        if csv_output is not None:
            from toporetarget.data.validation.grab import GrabValidationReport

            GrabValidationReport(
                payload["status"],
                payload["errors"],
                payload["warnings"],
                payload["checks"],
                payload["metrics"],
            ).write_csv(csv_output)
        _json_print(payload)
        if payload["status"] != "pass":
            raise typer.Exit(code=1)
    except (GrabAdapterError, StorageError, RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize")
def visualize_dataset(
    dataset: str = typer.Option("grab", "--dataset"),
    sequence: str = typer.Option("", "--sequence"),
    sequence_path: Path | None = typer.Option(None, "--sequence-path"),
    index: Path | None = typer.Option(None, "--index"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    mode: str = typer.Option("canonical", "--mode", help="raw, canonical, or compare"),
    layout: str = typer.Option("overlay", "--layout"),
    reference_frame: str | None = typer.Option(
        None,
        "--reference-frame",
        help="Canonical reference flag: scene, object, right-wrist, or left-wrist.",
    ),
    reference_alias: str | None = typer.Option(
        None,
        "--reference",
        help="Deprecated compatibility alias for --reference-frame.",
    ),
    frame: int = typer.Option(0, "--frame", min=0),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    display_stride: int = typer.Option(1, "--display-stride", min=1),
    hands: str = typer.Option("both", "--hands"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    grab_root: Path | None = typer.Option(None, "--grab-root"),
    contact_mode: str = typer.Option("source", "--contact-mode"),
    show_mediapipe21: bool = typer.Option(False, "--show-mediapipe21"),
    show_native_joints: bool = typer.Option(False, "--show-native-joints"),
    show_mesh: bool = typer.Option(True, "--show-mesh/--hide-mesh"),
    show_table: bool = typer.Option(False, "--show-table"),
    show_contacts: bool = typer.Option(False, "--show-contacts"),
    contact_color_mode: str = typer.Option(
        "binary", "--contact-color-mode", help="source, binary, or semantic contact colors."
    ),
    show_axes: bool = typer.Option(False, "--show-axes"),
    interactive: bool = typer.Option(
        False, "--interactive", "--show", help="Open the interactive slider/buttons viewer."
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Render raw/canonical/compare GRAB scenes; interaction never changes data."""

    if dataset != "grab":
        raise typer.BadParameter("only --dataset grab has formal visualization")
    try:
        if reference_alias is not None:
            typer.echo(
                "Warning: --reference is deprecated; use --reference-frame instead.", err=True
            )
            if reference_frame is not None and reference_frame != reference_alias:
                raise typer.BadParameter(
                    "--reference and --reference-frame must have the same value "
                    "when both are provided"
                )
            reference_frame = reference_alias
        resolved_reference_frame = reference_frame or "scene"
        canonical_sequence = load_hoi_sequence(canonical) if canonical is not None else None
        raw_sequence = None
        if mode in {"raw", "compare"}:
            adapter = _grab_adapter(
                sequence_path=sequence_path,
                grab_root=grab_root,
                index=index,
                mano_model_root=mano_model_root,
                hands=hands,
                contact_mode=contact_mode,
                include_mediapipe21=show_mediapipe21,
            )
            raw_sequence = adapter.load_sequence(
                sequence,
                options=GrabLoadOptions(
                    hands=hands,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    contact_mode=contact_mode,
                    include_mediapipe21=show_mediapipe21,
                ),
            )
        if mode == "canonical" and canonical_sequence is None:
            raise typer.BadParameter("canonical mode requires --canonical")
        if mode == "compare" and canonical_sequence is None:
            raise typer.BadParameter("compare mode requires --canonical")
        opts = GrabViewerOptions(
            mode=mode,
            layout=layout,
            reference_frame=resolved_reference_frame,
            frame=frame,
            display_stride=display_stride,
            show_mesh=show_mesh,
            show_mediapipe21=show_mediapipe21,
            show_native_joints=show_native_joints,
            show_table=show_table,
            show_contacts=show_contacts,
            show_axes=show_axes,
            contact_color_mode=contact_color_mode,
        )
        viewer = render_grab_view(
            canonical=canonical_sequence,
            raw=raw_sequence,
            options=opts,
            output=output,
            interactive=interactive,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        if not interactive:
            viewer.close()
        _json_print(
            {
                "mode": mode,
                "layout": layout,
                "reference_frame": resolved_reference_frame,
                "frame": viewer.frame,
                "output": None if output is None else str(output),
                "interactive": interactive,
            }
        )
    except (GrabAdapterError, StorageError, RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
