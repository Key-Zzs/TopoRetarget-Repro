"""Explicit MANO keypoint layout, conversion, validation, and visualization commands."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import typer

from toporetarget.data.storage import StorageError, load_hoi_sequence, save_hoi_sequence
from toporetarget.keypoints.mano_to_mediapipe import ManoToMediaPipe21Converter, MappingError
from toporetarget.keypoints.registry import load_layouts, load_profiles
from toporetarget.keypoints.reports import validate_mapping
from toporetarget.keypoints.visualization import (
    launch_interactive_keypoint_viewer,
    render_keypoint_view,
)

app = typer.Typer(help="Explicit MANO/source-hand keypoint layouts and MediaPipe21 adapter tools.")


def _json_print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _profile(profile_id: str):
    profiles = load_profiles()
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise typer.BadParameter(
            f"unknown profile {profile_id!r}; choose one of {', '.join(sorted(profiles))}"
        ) from exc


@app.command("layouts")
def layouts() -> None:
    """List registered semantic layouts and source-layout aliases."""

    registry = load_layouts()
    seen: set[str] = set()
    output = []
    for _name, layout in sorted(registry.items()):
        if layout.name in seen:
            continue
        seen.add(layout.name)
        output.append(
            {
                "name": layout.name,
                "version": layout.version,
                "points": layout.point_count,
                "wrist_index": layout.wrist_index,
                "fingertip_indices": list(layout.fingertip_indices),
                "aliases": list(layout.aliases),
                "coordinate_frame": layout.coordinate_frame,
                "units": layout.units,
            }
        )
    _json_print(output)


@app.command("profiles")
def profiles() -> None:
    """List registered MANO mapping profiles."""

    _json_print(
        [
            {
                "profile_id": profile.profile_id,
                "version": profile.version,
                "source_layout": profile.source_joint_layout,
                "target_layout": profile.target_layout,
                "expected_vertex_count": profile.expected_vertex_count,
                "mapping_mode": profile.mapping_mode,
                "verification_status": profile.verification.get("status", "unknown"),
                "profile_hash": profile.sha256,
            }
            for profile in sorted(load_profiles().values(), key=lambda item: item.profile_id)
        ]
    )


@app.command("describe-profile")
def describe_profile(
    profile: str = typer.Option("mano_v1_2_smplx_to_mediapipe21", "--profile"),
) -> None:
    """Print the complete selected mapping profile."""

    _json_print(_profile(profile).as_dict())


@app.command("convert")
def convert(
    input: Path = typer.Option(..., "--input", help="Existing canonical Zarr cache."),
    output: Path = typer.Option(..., "--output", help="New canonical Zarr cache."),
    hand: str = typer.Option(
        "right", "--hand", help="Hand id or side; hand_r/hand_l are accepted."
    ),
    profile: str = typer.Option("mano_v1_2_smplx_to_mediapipe21", "--profile"),
    mano_model_root: Path | None = typer.Option(None, "--mano-model-root"),
    force: bool = typer.Option(False, "--force", help="Allow replacing an existing output cache."),
) -> None:
    """Copy one cache and add an explicit mediapipe21 track without resampling."""

    try:
        if input.resolve() == output.resolve():
            raise MappingError("input and output must be different paths")
        if not input.is_dir():
            raise MappingError(f"input cache does not exist: {input}")
        if output.exists() and not force:
            raise MappingError(f"output already exists: {output}; pass --force to replace it")
        sequence = load_hoi_sequence(input)
        hand_id = hand
        if hand in {"left", "right"}:
            hand_id = next((item.hand_id for item in sequence.hands if item.side == hand), hand)
        converter = ManoToMediaPipe21Converter(_profile(profile))
        converted = converter.convert_sequence(
            sequence,
            hand_id=hand_id,
            mano_model_root=mano_model_root,
            overwrite=force,
        )
        save_hoi_sequence(converted, output)
        track = converted.hand(hand_id).keypoint_tracks["mediapipe21"]
        _json_print(
            {
                "input": str(input),
                "output": str(output),
                "hand_id": hand_id,
                "profile_id": converter.profile.profile_id,
                "profile_hash": converter.profile.sha256,
                "source_layout": track.provenance.get("source_layout"),
                "target_layout": track.layout_name,
                "shape": list(track.positions_scene.shape),
                "frame": track.frame_name,
                "units": track.units,
                "native_fps": converted.metadata.native_fps,
                "num_frames": converted.num_frames,
                "overwrite": force,
            }
        )
    except (StorageError, MappingError, RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"keypoint conversion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate")
def validate(
    input: Path = typer.Option(..., "--input"),
    hand: str = typer.Option("right", "--hand"),
    layout: str = typer.Option("mediapipe21", "--layout"),
    profile: str = typer.Option("mano_v1_2_smplx_to_mediapipe21", "--profile"),
    report: Path | None = typer.Option(None, "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Validate layout, source preservation, frame conversion, and geometry consistency."""

    try:
        sequence = load_hoi_sequence(input)
        hand_id = hand
        if hand in {"left", "right"}:
            hand_id = next((item.hand_id for item in sequence.hands if item.side == hand), hand)
        selected = sequence.hand(hand_id).keypoint_tracks.get(layout)
        if selected is None:
            raise MappingError(f"hand {hand_id!r} has no layout {layout!r}")
        selected_profile = _profile(profile)
        original = copy.deepcopy(sequence)
        original.hand(hand_id).keypoint_tracks.pop(layout, None)
        result = validate_mapping(original, sequence, hand_id=hand_id, profile=selected_profile)
        if report is not None:
            result.write_json(report)
        if csv_report is not None:
            result.write_csv(csv_report)
        _json_print(result.as_dict())
    except (StorageError, MappingError, RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"keypoint validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize")
def visualize(
    input: Path = typer.Option(..., "--input"),
    hand: str = typer.Option("right", "--hand"),
    layout: str = typer.Option("mediapipe21", "--layout"),
    view: str = typer.Option("scene", "--view", help="scene or wrist"),
    frame: int = typer.Option(0, "--frame", min=0),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    show: bool = typer.Option(False, "--show", help="Launch the local interactive viewer."),
    show_source_layout: bool = typer.Option(False, "--show-source-layout"),
    show_mesh: bool = typer.Option(True, "--show-mesh/--hide-mesh"),
    show_target: bool = typer.Option(
        True, "--show-mediapipe21/--hide-mediapipe21", help="Show MediaPipe-21 joints."
    ),
    show_skeleton: bool = typer.Option(True, "--show-skeleton/--hide-skeleton"),
    show_labels: bool = typer.Option(False, "--show-labels"),
    show_object_mesh: bool = typer.Option(False, "--show-object-mesh"),
    show_axes: bool = typer.Option(True, "--show-axes/--hide-axes"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Render a static frame or launch the local interactive sequence viewer."""

    try:
        sequence = load_hoi_sequence(input)
        hand_id = hand
        if hand in {"left", "right"}:
            hand_id = next((item.hand_id for item in sequence.hands if item.side == hand), hand)
        if layout != "mediapipe21":
            raise MappingError("visualize currently renders the mediapipe21 target layout")
        if not show and output is None:
            raise MappingError(
                "provide --output for a static PNG or pass --show for interactive mode"
            )
        if show:
            launch_interactive_keypoint_viewer(
                sequence,
                hand_id=hand_id,
                start_frame=start_frame,
                end_frame=end_frame,
                view=view,
                show_source_layout=show_source_layout,
                show_mesh=show_mesh,
                show_target=show_target,
                show_skeleton=show_skeleton,
                show_labels=show_labels,
                show_object_mesh=show_object_mesh,
                show_axes=show_axes,
            )
        if output is not None:
            render_keypoint_view(
                sequence,
                hand_id=hand_id,
                frame=frame,
                view=view,
                show_source_layout=show_source_layout,
                show_mesh=show_mesh,
                show_labels=show_labels,
                output=output,
            )
        _json_print(
            {
                "input": str(input),
                "output": None if output is None else str(output),
                "hand_id": hand_id,
                "view": view,
                "frame": frame,
                "interactive": show,
                "start_frame": start_frame,
                "end_frame": sequence.num_frames if end_frame is None else end_frame,
            }
        )
    except (StorageError, MappingError, RuntimeError, ValueError, OSError) as exc:
        typer.echo(f"keypoint visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
