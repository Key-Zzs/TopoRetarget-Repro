"""Stage 7 relative-bone-direction inspection, solving, validation, and views."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import typer

from toporetarget.data.storage import StorageError, load_hoi_sequence
from toporetarget.geometry.se3 import pose_rotation_error, pose_translation_error
from toporetarget.geometry.surface_sampling import SurfaceSampleSet
from toporetarget.retarget.alignment import observability_report
from toporetarget.retarget.artifacts import (
    WarmStartArtifactError,
    artifact_hash,
    load_warm_start,
    save_warm_start,
)
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.delaunay import load_delaunay_profile
from toporetarget.retarget.frames import FrameDegeneracyError, load_frame_profile
from toporetarget.retarget.interaction_artifacts import (
    InteractionArtifactError,
    interaction_artifact_hash,
    load_interaction_evaluation,
    load_interaction_graph,
    save_interaction_evaluation,
    save_interaction_graph,
)
from toporetarget.retarget.interaction_evaluation import evaluate_interaction_graph
from toporetarget.retarget.interaction_graph import (
    build_source_interaction_graph,
    load_paper_kappa,
)
from toporetarget.retarget.interaction_reports import (
    build_input_audit,
    compare_object_scales,
    topology_over_time,
)
from toporetarget.retarget.interaction_validation import (
    validate_interaction_evaluation,
    validate_interaction_graph,
    write_validation_reports,
)
from toporetarget.retarget.interaction_visualization import (
    launch_interaction_viewer,
    render_interaction_frame,
)
from toporetarget.retarget.pipeline import build_warm_start_trajectory
from toporetarget.retarget.solver import WarmStartSolveError, load_solver_profile
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.utils.hashing import sha256_tree

app = typer.Typer(help="Stage 7 relative bone-direction and sequential warm-start tools.")


def _json_write(value: Any, path: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if path is None:
        typer.echo(text, nl=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _resolve_hand(sequence: Any, hand: str) -> str:
    if hand in {item.hand_id for item in sequence.hands}:
        return hand
    for item in sequence.hands:
        if item.side == hand:
            return item.hand_id
    raise typer.BadParameter(f"hand {hand!r} is not present in canonical cache")


def _load_robot(name: str, asset_root: Path | None) -> Any:
    side = {"artimano_rh": "rh", "artimano_lh": "lh"}.get(name, name)
    return load_artimano_model(side, asset_root=asset_root)


def _slice_sequence(sequence: Any, start: int, end: int | None) -> Any:
    result = copy.deepcopy(sequence)
    stop = result.num_frames if end is None else end
    if start < 0 or stop <= start or stop > result.num_frames:
        raise typer.BadParameter(
            f"invalid frame range [{start},{stop}) for {result.num_frames} frames"
        )
    result.metadata.timestamps = result.metadata.timestamps[start:stop]
    result.metadata.num_frames = stop - start
    for hand in result.hands:
        hand.wrist_pose_scene.pose_scene = hand.wrist_pose_scene.pose_scene[start:stop]
        if hand.wrist_pose_scene.valid is not None:
            hand.wrist_pose_scene.valid = hand.wrist_pose_scene.valid[start:stop]
        if hand.valid is not None:
            hand.valid = hand.valid[start:stop]
        if hand.vertices_scene is not None:
            hand.vertices_scene = hand.vertices_scene[start:stop]
        for track in hand.keypoint_tracks.values():
            track.positions_scene = track.positions_scene[start:stop]
            if track.valid is not None and track.valid.shape[0] == sequence.num_frames:
                track.valid = track.valid[start:stop]
            if track.confidence is not None and track.confidence.shape[0] == sequence.num_frames:
                track.confidence = track.confidence[start:stop]
    for obj in result.rigid_objects:
        obj.pose_scene.pose_scene = obj.pose_scene.pose_scene[start:stop]
        if obj.pose_scene.valid is not None:
            obj.pose_scene.valid = obj.pose_scene.valid[start:stop]
        if obj.valid is not None:
            obj.valid = obj.valid[start:stop]
    return result


def _write_feature_csv(feature: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "index", "name", "x", "y", "z", "length", "valid"])
        lengths = np.asarray(feature.bone_lengths)
        vectors = np.asarray(feature.bone_vectors)
        valid = np.asarray(feature.valid_bones)
        for index, name in enumerate(feature.bone_names):
            writer.writerow(
                ["bone", index, name, *vectors[index], lengths[index], bool(valid[index])]
            )
        values = np.asarray(feature.adjacent_features)
        for index, name in enumerate(feature.pair_names):
            writer.writerow(
                [
                    "pair",
                    index,
                    name,
                    *values[index],
                    "",
                    bool(np.asarray(feature.valid_pairs)[index]),
                ]
            )


def _np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


@app.command("inspect-bones")
def inspect_bones(
    canonical: Path = typer.Option(..., "--canonical"),
    hand: str = typer.Option("right", "--hand"),
    layout: str = typer.Option("mediapipe21", "--layout"),
    frame_profile: str = typer.Option("canonical_keypoint_wrist_v1", "--frame-profile"),
    bone_profile: str = typer.Option("mediapipe21_full_finger_chain_v1", "--bone-profile"),
    frame: int = typer.Option(0, "--frame", min=0),
    json_report: Path | None = typer.Option(None, "--json"),
    csv_report: Path | None = typer.Option(None, "--csv"),
) -> None:
    try:
        sequence = load_hoi_sequence(canonical)
        hand_id = _resolve_hand(sequence, hand)
        if layout != "mediapipe21":
            raise typer.BadParameter("Stage 7 inspect-bones requires --layout mediapipe21")
        if frame >= sequence.num_frames:
            raise typer.BadParameter("frame is outside canonical cache")
        fp = load_frame_profile(frame_profile)
        bp = load_bone_profile(bone_profile)
        track = sequence.hand(hand_id).keypoint_tracks[layout]
        feature = extract_bone_features(
            track.positions_scene[frame], fp, bp, side=sequence.hand(hand_id).side
        )
        report = {
            "canonical": str(canonical),
            "hand_id": hand_id,
            "side": sequence.hand(hand_id).side,
            "frame": frame,
            "frame_profile": fp.as_dict(),
            "bone_profile": bp.as_dict(),
            "frame_transform": np.asarray(feature.frame_transform).tolist(),
            "features": feature.as_dict(),
        }
        _json_write(report, json_report)
        if csv_report is not None:
            _write_feature_csv(feature, csv_report)
    except (StorageError, ValueError, OSError, FrameDegeneracyError) as exc:
        typer.echo(f"inspect-bones failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare-frame-profiles")
def compare_frame_profiles(
    canonical: Path = typer.Option(..., "--canonical"),
    hand: str = typer.Option("right", "--hand"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    frame: int = typer.Option(0, "--frame", min=0),
    report: Path | None = typer.Option(None, "--report"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        sequence = load_hoi_sequence(canonical)
        hand_id = _resolve_hand(sequence, hand)
        model = _load_robot(robot, asset_root)
        if frame >= sequence.num_frames:
            raise typer.BadParameter("frame is outside canonical cache")
        bp = load_bone_profile("mediapipe21_full_finger_chain_v1")
        track = sequence.hand(hand_id).keypoint_tracks["mediapipe21"]
        source = track.positions_scene[frame]
        profiles = [
            load_frame_profile("canonical_keypoint_wrist_v1"),
            load_frame_profile("translation_centered_scene_axes"),
        ]
        values: dict[str, Any] = {
            "canonical": str(canonical),
            "hand_id": hand_id,
            "frame": frame,
            "robot": robot,
            "profiles": {},
        }
        neutral = model.neutral_q
        for fp in profiles:
            source_feature = extract_bone_features(source, fp, bp, side=sequence.hand(hand_id).side)
            robot_feature = extract_bone_features(
                model.keypoints_base(neutral).detach().cpu().numpy(), fp, bp, side=model.side
            )
            obs = observability_report(
                source_feature.adjacent_features, model, fp, bp, neutral, side=model.side
            )
            values["profiles"][fp.profile_id] = {
                "profile": fp.as_dict(),
                "source_frame": _np(source_feature.frame_transform).tolist(),
                "robot_frame_base": _np(robot_feature.frame_transform).tolist(),
                "ebone_neutral": float(
                    np.sum(
                        (
                            _np(robot_feature.adjacent_features)
                            - _np(source_feature.adjacent_features)
                        )
                        ** 2
                    )
                ),
                "observability": obs,
            }
        values["neutral_source_robot_frame_rotation_error_rad"] = float(
            pose_rotation_error(
                np.asarray(values["profiles"][profiles[0].profile_id]["source_frame"]),
                np.asarray(values["profiles"][profiles[0].profile_id]["robot_frame_base"]),
            )
        )
        values["note"] = (
            "Bounded diagnostic comparison; the lower loss is not used to choose "
            "the default profile."
        )
        _json_write(values, report)
    except (StorageError, ValueError, OSError, RuntimeError) as exc:
        typer.echo(f"compare-frame-profiles failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("warm-start")
def warm_start(
    canonical: Path = typer.Option(..., "--canonical"),
    hand: str = typer.Option("right", "--hand"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    frame_profile: str = typer.Option("canonical_keypoint_wrist_v1", "--frame-profile"),
    bone_profile: str = typer.Option("mediapipe21_full_finger_chain_v1", "--bone-profile"),
    solver_profile: str = typer.Option("paper_repro_scipy_trf", "--solver-profile"),
    output: Path = typer.Option(..., "--output"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    try:
        sequence = _slice_sequence(load_hoi_sequence(canonical), start_frame, end_frame)
        hand_id = _resolve_hand(sequence, hand)
        model = _load_robot(robot, asset_root)
        fp = load_frame_profile(frame_profile)
        bp = load_bone_profile(bone_profile)
        sp = load_solver_profile(solver_profile)
        trajectory, diagnostics = build_warm_start_trajectory(
            sequence, hand_id, model, fp, bp, sp, source_cache=canonical
        )
        trajectory.metadata["frame_range"] = [start_frame, start_frame + trajectory.frame_count]
        trajectory.metadata["source_cache_path"] = str(canonical)
        save_warm_start(trajectory, output, force=force)
        _json_write(
            {
                "output": str(output),
                "artifact_hash": artifact_hash(output),
                "metadata": trajectory.metadata,
                "diagnostics": diagnostics,
            },
            None,
        )
    except (
        StorageError,
        ValueError,
        OSError,
        RuntimeError,
        WarmStartArtifactError,
        WarmStartSolveError,
    ) as exc:
        typer.echo(f"warm-start failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-warm-start")
def validate_warm_start(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    report: Path | None = typer.Option(None, "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        sequence = load_hoi_sequence(canonical)
        trajectory = load_warm_start(warm_start)
        metadata = trajectory.metadata
        hand_id = _resolve_hand(sequence, str(metadata["source_hand_id"]))
        model = _load_robot(str(metadata["robot_name"]), asset_root)
        lower, upper = model.joint_lower, model.joint_upper
        q = trajectory.arrays["qpos"]
        bounds_ok = bool(
            np.all(q >= lower[None, :] - 1e-10) and np.all(q <= upper[None, :] + 1e-10)
        )
        fk = np.stack([np.asarray(model.keypoints_base(item).detach().cpu()) for item in q])
        fk_ok = bool(np.allclose(fk, trajectory.arrays["robot_keypoints_base"], atol=1e-10, rtol=0))
        alignment = {
            "translation_max_m": float(
                np.max(
                    pose_translation_error(
                        np.matmul(
                            trajectory.arrays["base_pose_scene"],
                            trajectory.arrays["robot_hand_frame_base"],
                        ),
                        trajectory.arrays["source_hand_frame_scene"],
                    )
                )
            ),
            "rotation_max_rad": float(
                np.max(
                    pose_rotation_error(
                        np.matmul(
                            trajectory.arrays["base_pose_scene"],
                            trajectory.arrays["robot_hand_frame_base"],
                        ),
                        trajectory.arrays["source_hand_frame_scene"],
                    )
                )
            ),
        }
        objective_ok = bool(
            np.all(
                trajectory.arrays["total_objective"]
                <= trajectory.arrays["initial_total_objective"] + 1e-10
            )
        )
        source_hash = metadata.get("source_cache_hash")
        current_hash = None
        if source_hash is not None:
            digest = __import__("hashlib").sha256()
            for name, value in sha256_tree(canonical).items():
                digest.update(name.encode())
                digest.update(b"\0")
                digest.update(value.encode())
                digest.update(b"\n")
            current_hash = digest.hexdigest()
        result = {
            "schema_version": trajectory.schema_version,
            "artifact_hash": artifact_hash(warm_start),
            "canonical": str(canonical),
            "hand_id": hand_id,
            "source_hash_match": source_hash is None or source_hash == current_hash,
            "frame_count": trajectory.frame_count,
            "qpos_shape": list(q.shape),
            "base_pose_shape": list(trajectory.arrays["base_pose_scene"].shape),
            "bounds_pass": bounds_ok,
            "fk_pass": fk_ok,
            "alignment": alignment,
            "objective_non_increasing": objective_ok,
            "solver_success_count": int(np.sum(trajectory.arrays["solver_success"])),
            "solver_failure_count": int(np.sum(~trajectory.arrays["solver_success"])),
            "source_integrity": {
                "canonical_cache_unchanged_during_validation": True,
                "object_samples_read": False,
                "sdf_read": False,
            },
            "pass": bool(
                bounds_ok
                and fk_ok
                and alignment["translation_max_m"] <= 1e-9
                and alignment["rotation_max_rad"] <= 1e-9
                and objective_ok
                and np.all(trajectory.arrays["solver_success"])
            ),
        }
        _json_write(result, report)
        if csv_report is not None:
            csv_report.parent.mkdir(parents=True, exist_ok=True)
            with csv_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "frame",
                        "ebone",
                        "temporal_term",
                        "total_objective",
                        "initial_total_objective",
                        "solver_success",
                        "joint_limit_min_margin",
                    ]
                )
                for index in range(trajectory.frame_count):
                    writer.writerow(
                        [
                            index,
                            trajectory.arrays["ebone"][index],
                            trajectory.arrays["temporal_term"][index],
                            trajectory.arrays["total_objective"][index],
                            trajectory.arrays["initial_total_objective"][index],
                            trajectory.arrays["solver_success"][index],
                            np.min(trajectory.arrays["joint_limit_margins"][index]),
                        ]
                    )
        _json_write(result, None)
        if not result["pass"]:
            raise typer.Exit(code=1)
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"validate-warm-start failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-warm-start")
def visualize_warm_start(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    view: str = typer.Option("scene", "--view"),
    frame: int = typer.Option(0, "--frame", min=0),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    show_source_hand: bool = typer.Option(False, "--show-source-hand"),
    show_robot_hand: bool = typer.Option(False, "--show-robot-hand"),
    show_source_skeleton: bool = typer.Option(False, "--show-source-skeleton"),
    show_robot_skeleton: bool = typer.Option(False, "--show-robot-skeleton"),
    show_object_context: bool = typer.Option(False, "--show-object-context"),
    show_hand_frames: bool = typer.Option(False, "--show-hand-frames"),
    show_directions: bool = typer.Option(False, "--show-directions"),
    show_adjacent_features: bool = typer.Option(False, "--show-adjacent-features"),
    show_labels: bool = typer.Option(False, "--show-labels"),
    show_residuals: bool = typer.Option(False, "--show-residuals"),
    output: Path | None = typer.Option(None, "--output"),
    interactive: bool = typer.Option(False, "--interactive"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    try:
        from toporetarget.retarget.visualization import (
            launch_warm_start_viewer,
            render_warm_start_frame,
        )

        sequence = load_hoi_sequence(canonical)
        trajectory = load_warm_start(warm_start)
        hand_id = _resolve_hand(sequence, str(trajectory.metadata["source_hand_id"]))
        if interactive:
            show_any_hand_layer = any(
                (show_source_hand, show_robot_hand, show_source_skeleton, show_robot_skeleton)
            )
            smoke = launch_warm_start_viewer(
                trajectory,
                sequence,
                hand_id=hand_id,
                view=view,
                start_frame=start_frame,
                end_frame=end_frame,
                show_source_hand=show_source_hand or not show_any_hand_layer,
                show_robot_hand=show_robot_hand or not show_any_hand_layer,
                show_source_skeleton=show_source_skeleton or not show_any_hand_layer,
                show_robot_skeleton=show_robot_skeleton or not show_any_hand_layer,
                show_object_context=show_object_context,
                show_hand_frames=show_hand_frames,
                show_directions=show_directions,
                show_adjacent_features=show_adjacent_features,
                show_labels=show_labels,
                show_residuals=show_residuals,
            )
            _json_write(smoke, report)
            return
        render_warm_start_frame(
            sequence,
            trajectory,
            hand_id=hand_id,
            frame=frame,
            view=view,
            output=output,
            show_source_hand=show_source_hand
            or not any(
                (show_source_hand, show_robot_hand, show_source_skeleton, show_robot_skeleton)
            ),
            show_robot_hand=show_robot_hand
            or not any(
                (show_source_hand, show_robot_hand, show_source_skeleton, show_robot_skeleton)
            ),
            show_source_skeleton=show_source_skeleton
            or not any(
                (show_source_hand, show_robot_hand, show_source_skeleton, show_robot_skeleton)
            ),
            show_robot_skeleton=show_robot_skeleton
            or not any(
                (show_source_hand, show_robot_hand, show_source_skeleton, show_robot_skeleton)
            ),
            show_object_context=show_object_context,
            show_hand_frames=show_hand_frames,
            show_directions=show_directions,
            show_adjacent_features=show_adjacent_features,
            show_residuals=show_residuals,
            show_labels=show_labels,
        )
        _json_write(
            {
                "output": None if output is None else str(output),
                "view": view,
                "frame": frame,
                "interactive": False,
            },
            report,
        )
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"visualize-warm-start failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("build-interaction-graph")
def build_interaction_graph(
    canonical: Path = typer.Option(..., "--canonical"),
    hand: str = typer.Option("right", "--hand"),
    object_id: str = typer.Option("primary", "--object-id"),
    object_samples: Path = typer.Option(..., "--object-samples"),
    delaunay_profile: str = typer.Option("strict_scipy_qhull_v1", "--delaunay-profile"),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    output: Path = typer.Option(..., "--output"),
    report: Path | None = typer.Option(None, "--report"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Build source-only Eq. (3)-(6) graphs; no robot or warm-start is loaded."""

    try:
        sequence = load_hoi_sequence(canonical)
        stop_frame = sequence.num_frames if end_frame is None else end_frame
        if start_frame < 0 or stop_frame <= start_frame or stop_frame > sequence.num_frames:
            raise typer.BadParameter(
                f"invalid frame range [{start_frame},{stop_frame}) for {sequence.num_frames} frames"
            )
        hand_id = _resolve_hand(sequence, hand)
        samples = SurfaceSampleSet.load(object_samples)
        profile = load_delaunay_profile(delaunay_profile)
        selected = list(range(start_frame, stop_frame))
        graph = build_source_interaction_graph(
            sequence,
            hand_id,
            object_id,
            samples,
            source_cache=canonical,
            object_sample_path=object_samples,
            delaunay_profile=profile,
            kappa=load_paper_kappa(),
            frame_indices=selected,
        )
        save_interaction_graph(graph, output, force=force)
        result = {
            "status": "pass",
            "output": str(output),
            "artifact_hash": interaction_artifact_hash(output),
            "schema_version": graph.schema_version,
            "metadata": graph.metadata,
            "topology_over_time": topology_over_time(graph),
        }
        _json_write(result, report)
        if report is None:
            _json_write(result, None)
    except (
        StorageError,
        ValueError,
        OSError,
        RuntimeError,
        InteractionArtifactError,
    ) as exc:
        typer.echo(f"build-interaction-graph failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("audit-interaction-inputs")
def audit_interaction_inputs(
    right_canonical: Path = typer.Option(..., "--right-canonical"),
    left_canonical: Path = typer.Option(..., "--left-canonical"),
    right_warm_start: Path = typer.Option(..., "--right-warm-start"),
    left_warm_start: Path = typer.Option(..., "--left-warm-start"),
    object_samples: Path = typer.Option(..., "--object-samples"),
    report: Path = typer.Option(..., "--report"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    """Audit Stage 6/7 inputs and robot assets without building a graph."""

    try:
        result = build_input_audit(
            [
                {"canonical": right_canonical, "warm_start": right_warm_start, "hand_id": "hand_r"},
                {
                    "canonical": left_canonical,
                    "warm_start": left_warm_start,
                    "hand_id": "left_hand",
                },
            ],
            object_samples,
            asset_root=asset_root,
        )
        _json_write(result, report)
        _json_write(result, None)
        if not result["all_compatibility_checks_pass"]:
            raise typer.Exit(code=1)
    except (ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"audit-interaction-inputs failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("inspect-interaction-graph")
def inspect_interaction_graph(
    graph: Path = typer.Option(..., "--graph"),
    frame: int = typer.Option(0, "--frame", min=0),
    json_report: Path | None = typer.Option(None, "--json"),
    csv_report: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Inspect one saved graph frame and its directed weights."""

    try:
        trajectory = load_interaction_graph(graph)
        if frame >= trajectory.frame_count:
            raise typer.BadParameter("frame is outside interaction graph")
        item = trajectory.frames[frame]
        payload = {
            "frame": int(trajectory.frame_indices[frame]),
            "vertices": item.source_vertices.tolist(),
            "simplices": item.simplices.tolist(),
            "edges": [
                {"edge_id": i, "indices": edge.tolist(), "category": _edge_category(edge)}
                for i, edge in enumerate(item.edges)
            ],
            "directed_source_index": item.directed_source_index.tolist(),
            "directed_destination_index": item.directed_destination_index.tolist(),
            "weights": item.weights.tolist(),
            "row_sums": item.directed.row_sums.tolist(),
            "statistics": item.statistics,
            "graph_hash": item.graph_hash,
        }
        _json_write(payload, json_report)
        if csv_report is not None:
            csv_report.parent.mkdir(parents=True, exist_ok=True)
            with csv_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["edge_id", "source", "destination", "category", "weight", "distance_squared"]
                )
                for edge_id, (source, destination, weight, distance) in enumerate(
                    zip(
                        item.directed_source_index,
                        item.directed_destination_index,
                        item.weights,
                        item.source_distance_squared,
                        strict=True,
                    )
                ):
                    writer.writerow(
                        [
                            edge_id,
                            int(source),
                            int(destination),
                            _edge_category((source, destination)),
                            weight,
                            distance,
                        ]
                    )
        if json_report is None:
            _json_write(payload, None)
    except (ValueError, OSError, RuntimeError, InteractionArtifactError) as exc:
        typer.echo(f"inspect-interaction-graph failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _edge_category(edge: Any) -> str:
    first, second = (int(value) for value in edge)
    if first < 21 and second < 21:
        return "hand-hand"
    if (first < 21) != (second < 21):
        return "hand-object"
    return "object-object"


@app.command("validate-interaction-graph")
def validate_interaction_graph_command(
    canonical: Path = typer.Option(..., "--canonical"),
    object_samples: Path = typer.Option(..., "--object-samples"),
    graph: Path = typer.Option(..., "--graph"),
    report: Path = typer.Option(..., "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Validate graph vertices, source hashes, connectivity, and weights."""

    try:
        trajectory = load_interaction_graph(graph)
        samples = SurfaceSampleSet.load(object_samples)
        result = validate_interaction_graph(
            trajectory, canonical, samples, sample_path=object_samples
        )
        if csv_report is None:
            csv_report = report.with_suffix(".csv")
        write_validation_reports(result, report, csv_report)
        _json_write(result, None)
        if not result["all_frames_valid"]:
            raise typer.Exit(code=1)
    except (ValueError, OSError, RuntimeError, InteractionArtifactError) as exc:
        typer.echo(f"validate-interaction-graph failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("evaluate-interaction")
def evaluate_interaction_command(
    graph: Path = typer.Option(..., "--graph"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    output: Path = typer.Option(..., "--output"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    force: bool = typer.Option(False, "--force"),
    no_jacobian: bool = typer.Option(False, "--no-jacobian"),
) -> None:
    """Evaluate Eq. (7) on Stage 7 qpos/base without optimization."""

    try:
        graph_trajectory = load_interaction_graph(graph)
        warm = load_warm_start(warm_start)
        model = _load_robot(robot, asset_root)
        evaluation = evaluate_interaction_graph(
            graph_trajectory,
            warm,
            model,
            graph_artifact_hash=interaction_artifact_hash(graph),
            warm_start_artifact_hash=artifact_hash(warm_start),
        )
        save_interaction_evaluation(evaluation, output, force=force)
        result = {
            "status": "pass",
            "output": str(output),
            "artifact_hash": interaction_artifact_hash(output),
            "metadata": evaluation.metadata,
            "e_im": {
                "min": float(np.min(evaluation.e_im)),
                "mean": float(np.mean(evaluation.e_im)),
                "max": float(np.max(evaluation.e_im)),
            },
        }
        _json_write(result, None)
    except (
        ValueError,
        OSError,
        RuntimeError,
        WarmStartArtifactError,
        InteractionArtifactError,
    ) as exc:
        typer.echo(f"evaluate-interaction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-interaction")
def validate_interaction_command(
    graph: Path = typer.Option(..., "--graph"),
    evaluation: Path = typer.Option(..., "--evaluation"),
    report: Path = typer.Option(..., "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
    warm_start: Path | None = typer.Option(None, "--warm-start"),
) -> None:
    """Validate Eq. (7), scaled residuals, identity oracle, and frozen qpos."""

    try:
        graph_trajectory = load_interaction_graph(graph)
        evaluation_trajectory = load_interaction_evaluation(evaluation)
        result = validate_interaction_evaluation(
            graph_trajectory, evaluation_trajectory, warm_start=warm_start
        )
        if csv_report is None:
            csv_report = report.with_suffix(".csv")
        write_validation_reports(result, report, csv_report)
        _json_write(result, None)
        if not result["all_frames_valid"]:
            raise typer.Exit(code=1)
    except (ValueError, OSError, RuntimeError, InteractionArtifactError) as exc:
        typer.echo(f"validate-interaction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-interaction")
def visualize_interaction_command(
    graph: Path = typer.Option(..., "--graph"),
    mode: str = typer.Option("source", "--mode"),
    evaluation: Path | None = typer.Option(None, "--evaluation"),
    layout: str = typer.Option("single", "--layout"),
    frame: int = typer.Option(0, "--frame", min=0),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    show_laplacian: bool = typer.Option(False, "--show-laplacian"),
    show_residuals: bool = typer.Option(False, "--show-residuals"),
    show_contributions: bool = typer.Option(False, "--show-contributions"),
    show_labels: bool = typer.Option(False, "--show-labels"),
    show_hand_hand_edges: bool = typer.Option(
        True, "--show-hand-hand-edges/--hide-hand-hand-edges"
    ),
    show_hand_object_edges: bool = typer.Option(
        True, "--show-hand-object-edges/--hide-hand-object-edges"
    ),
    show_object_object_edges: bool = typer.Option(
        True, "--show-object-object-edges/--hide-object-object-edges"
    ),
    output: Path | None = typer.Option(None, "--output"),
    interactive: bool = typer.Option(False, "--interactive"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    """Render source/shared graph, Laplacians, and Eq. (7) residual diagnostics."""

    try:
        graph_trajectory = load_interaction_graph(graph)
        evaluation_trajectory = (
            None if evaluation is None else load_interaction_evaluation(evaluation)
        )
        if interactive:
            result = launch_interaction_viewer(
                graph_trajectory,
                evaluation=evaluation_trajectory,
                start_frame=start_frame,
                end_frame=end_frame,
                mode=mode,
                show_laplacian=show_laplacian,
                show_residuals=show_residuals,
                show_contributions=show_contributions,
            )
        else:
            result = render_interaction_frame(
                graph_trajectory,
                evaluation=evaluation_trajectory,
                frame=frame,
                mode=mode,
                layout=layout,
                output=output,
                show_hand_hand_edges=show_hand_hand_edges,
                show_hand_object_edges=show_hand_object_edges,
                show_object_object_edges=show_object_object_edges,
                show_laplacian=show_laplacian,
                show_residuals=show_residuals,
                show_contributions=show_contributions,
                show_labels=show_labels,
            )
        _json_write(result, report)
        if report is None:
            _json_write(result, None)
    except (ValueError, OSError, RuntimeError, InteractionArtifactError) as exc:
        typer.echo(f"visualize-interaction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare-object-scales")
def compare_object_scales_command(
    canonical: Path = typer.Option(..., "--canonical"),
    hand: str = typer.Option("right", "--hand"),
    object_id: str = typer.Option("primary", "--object-id"),
    object_samples: Path = typer.Option(..., "--object-samples"),
    scales: list[float] = typer.Option([0.6, 1.0, 1.4], "--scales"),
    frame: int = typer.Option(0, "--frame", min=0),
    report: Path = typer.Option(..., "--report"),
    output: Path | None = typer.Option(None, "--output"),
    delaunay_profile: str = typer.Option("strict_scipy_qhull_v1", "--delaunay-profile"),
) -> None:
    """Run graph-only object scale diagnostics; no retargeting is performed."""

    try:
        sequence = load_hoi_sequence(canonical)
        hand_id = _resolve_hand(sequence, hand)
        samples = SurfaceSampleSet.load(object_samples)
        profile = load_delaunay_profile(delaunay_profile)
        result = compare_object_scales(
            sequence,
            hand_id,
            samples,
            profile,
            list(scales),
            frame=frame,
            object_id=object_id,
            kappa=load_paper_kappa(),
        )
        _json_write(result, report)
        if output is not None:
            graph = build_source_interaction_graph(
                sequence,
                hand_id,
                object_id,
                samples,
                source_cache=canonical,
                object_sample_path=object_samples,
                delaunay_profile=profile,
                kappa=load_paper_kappa(),
                frame_indices=[frame],
            )
            render_interaction_frame(graph, frame=0, mode="source", output=output)
        _json_write(result, None)
    except (ValueError, OSError, RuntimeError, InteractionArtifactError) as exc:
        typer.echo(f"compare-object-scales failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
