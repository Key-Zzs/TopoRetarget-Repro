"""Stage 7/8/9 retargeting inspection, solving, validation, and views."""

from __future__ import annotations

import copy
import cProfile
import csv
import json
import pstats
import re
import shutil
import time
from collections.abc import Callable
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
from toporetarget.retarget.continuous import is_continuous_profile
from toporetarget.retarget.delaunay import load_delaunay_profile
from toporetarget.retarget.final_jobs import PAUSE_STATE, paused
from toporetarget.retarget.final_refinement import (
    ACTIVE_QUERY_PROFILE_ID,
    FINAL_REFINEMENT_SCHEMA_VERSION_V3,
    FULL_QUERY_PROFILE_ID,
    FULL_SOLVER_PROFILE_ID,
    SOLVER_PROFILE_ID,
    CollisionQueryProfile,
    PaperRefinementWeights,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    build_final_trajectory,
    build_query_set,
    dynamic_collision_points_numpy,
    final_artifact_hash,
    load_final_trajectory,
    load_robot_surface_samples,
    prepare_refinement_resources,
    prepare_refinement_runtime_backends,
    save_final_trajectory,
)
from toporetarget.retarget.final_visualization import (
    launch_refinement_viewer,
    render_refinement_frame,
)
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
from toporetarget.retarget.refinement_checkpoint import (
    CheckpointError,
    CheckpointStore,
    frame_checkpoint_payload,
)
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.retarget.solver import WarmStartSolveError, load_solver_profile
from toporetarget.robots.registry import get_robot_registry
from toporetarget.utils.hashing import sha256_file, sha256_tree

app = typer.Typer(help="Stage 7-9 retargeting tools.")


def _json_write(value: Any, path: Path | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    if path is None:
        typer.echo(text, nl=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _decode_bytes(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _load_quality_extension(path: Path | None) -> dict[str, Any] | None:
    """Load a versioned paper-external quality objective specification."""

    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = str(payload.get("kind", ""))
    if kind == "morphology_position_prior":
        from toporetarget.quality.morphology import load_morphology_objective_extension

        return load_morphology_objective_extension(
            payload["target_artifact"],
            profile_id=str(payload["profile_id"]),
            lambda_morph=float(payload["lambda_morph"]),
        )
    if kind == "contact_final":
        from toporetarget.quality.contact import load_contact_objective_extension

        return load_contact_objective_extension(
            payload["target_artifact"],
            payload["surface_profile"],
            profile_id=str(payload["profile_id"]),
            lambda_contact_pos=float(payload["lambda_contact_pos"]),
            lambda_contact_dir=float(payload["lambda_contact_dir"]),
        )
    raise ValueError(f"unsupported quality extension specification: {path}")


def _resolve_hand(sequence: Any, hand: str) -> str:
    if hand in {item.hand_id for item in sequence.hands}:
        return hand
    for item in sequence.hands:
        if item.side == hand:
            return item.hand_id
    raise typer.BadParameter(f"hand {hand!r} is not present in canonical cache")


def _load_robot(name: str, asset_root: Path | None) -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    return get_robot_registry(repo_root=repo_root).load(name, asset_root=asset_root)


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
    resume_from: Path | None = typer.Option(None, "--resume-from"),
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


def _default_collision_samples(robot: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / ".local" / "cache" / "geometry" / "robot_surface" / f"{robot}_neutral.npz"


def _source_frame_offset(path: Path) -> int:
    """Recover the source-frame offset recorded in cropped Stage 9 artifacts."""

    match = re.search(r"f(\d{6})_f\d{6}", str(path))
    return 0 if match is None else int(match.group(1))


def _object_for_graph(sequence: Any, object_id: str) -> Any:
    if object_id in {"primary", "object"}:
        if not sequence.rigid_objects:
            raise ValueError("canonical sequence has no rigid object")
        return sequence.rigid_objects[0]
    return sequence.rigid_object(object_id)


def _refinement_components(
    canonical: Path,
    warm_start: Path,
    graph: Path,
    robot: str,
    collision_samples: Path | None,
    asset_root: Path | None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    sequence = load_hoi_sequence(canonical)
    warm = load_warm_start(warm_start)
    graph_trajectory = load_interaction_graph(graph)
    model = _load_robot(robot, asset_root)
    surface = load_robot_surface_samples(collision_samples or _default_collision_samples(robot))
    return sequence, warm, graph_trajectory, model, surface, collision_samples


def _refinement_input_signature(
    canonical: Path,
    warm_start: Path,
    graph: Path,
    collision_samples: Path,
    robot: str,
    start_frame: int,
    end_frame: int,
    geometry_signature: str | None = None,
) -> str:
    import hashlib

    payload = {
        "canonical": str(canonical.resolve()),
        "canonical_hash": sha256_tree(canonical),
        "warm_start": str(warm_start.resolve()),
        "warm_start_hash": artifact_hash(warm_start),
        "graph": str(graph.resolve()),
        "graph_hash": interaction_artifact_hash(graph),
        "collision_samples": str(collision_samples.resolve()),
        "collision_samples_hash": sha256_file(collision_samples),
        "robot": robot,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "frame_range": [int(start_frame), int(end_frame)],
        "geometry_signature": geometry_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _checkpoint_manifest(
    *,
    sequence: Any,
    warm: Any,
    graph: Any,
    model: Any,
    paper: Any,
    solver: RefinementSolverProfile,
    execution: RefinementExecutionProfile,
    query: CollisionQueryProfile,
    coordinate: RefinementCoordinateProfile,
    surface: Any,
    resources: Any,
    input_signature: str,
    start_frame: int,
    end_frame: int,
    canonical: Path,
    warm_start: Path,
    graph_path: Path,
    collision_samples: Path,
    checkpoint_root: Path,
    source_frame_offset: int,
) -> dict[str, Any]:
    continuous = is_continuous_profile(solver.profile_id)
    result = {
        "schema_version": "toporetarget.final_retarget_checkpoint.v1",
        "run_id": checkpoint_root.name,
        "input_signature": input_signature,
        "source_sequence_id": warm.metadata.get("source_sequence_id"),
        "source_hand_id": warm.metadata.get("source_hand_id"),
        "source_hand_side": warm.metadata.get("source_side"),
        "canonical": str(canonical),
        "warm_start": str(warm_start),
        "graph": str(graph_path),
        "collision_samples": str(collision_samples),
        "robot_name": model.name,
        "robot_side": model.side,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "native_fps": warm.metadata.get("native_fps"),
        "solver_profile_id": solver.profile_id,
        "solver_profile_hash": solver.profile_hash,
        "execution_profile_id": execution.profile_id,
        "execution_profile_hash": execution.profile_hash,
        "query_profile_id": query.profile_id,
        "query_profile_hash": query.profile_hash,
        "frame_range": [int(start_frame), int(end_frame)],
        "source_frame_offset": int(source_frame_offset),
        "source_frame_range": [
            int(source_frame_offset + start_frame),
            int(source_frame_offset + end_frame),
        ],
        "final_artifact_schema": (
            FINAL_REFINEMENT_SCHEMA_VERSION_V3 if continuous else "toporetarget.final_retarget.v2"
        ),
        "paper_weights": paper.as_dict(),
        "resume_command": "toporetarget retarget refine --resume --checkpoint-root "
        + str(checkpoint_root),
        "final_artifact_metadata": {
            "source_sequence_id": warm.metadata.get("source_sequence_id"),
            "source_hand_id": warm.metadata.get("source_hand_id"),
            "source_hand_side": warm.metadata.get("source_side"),
            "source_canonical_hash": warm.metadata.get("source_cache_hash"),
            "object_id": graph.metadata.get("object_id"),
            "object_mesh_hash": resources.mesh_hash,
            "geometry_policy": resources.geometry_policy,
            "graph_artifact_hash": interaction_artifact_hash(graph.source_path)
            if graph.source_path
            else None,
            "warm_start_artifact_hash": artifact_hash(warm_start),
            "robot_name": model.name,
            "robot_side": model.side,
            "robot_dof_count": int(model.num_dofs),
            "robot_spec_hash": model.spec_hash,
            "robot_urdf_hash": model.urdf_hash,
            "robot_asset_manifest_hash": model.asset_manifest_hash,
            "robot_link_names": list(model.link_names),
            "collision_surface_profile_hash": surface.profile.profile_hash,
            "collision_surface_sample_count": surface.count,
            "sdf_backend": resources.sdf.describe(),
            "sdf_reference_backend": resources.reference_sdf.describe(),
            "sdf_selection_report": resources.sdf_report,
            "solver_profile_id": solver.profile_id,
            "solver_profile_hash": solver.profile_hash,
            "execution_profile_id": execution.profile_id,
            "execution_profile_hash": execution.profile_hash,
            "execution_profile": execution.as_dict(),
            "point_jacobian_backend": execution.point_jacobian_backend,
            "strict_recovery": execution.strict_recovery,
            "sdf_tree_leaf_size": execution.sdf_tree_leaf_size,
            "query_profile": query.as_dict(),
            "coordinate_profile": coordinate.as_dict(),
            "solver_profile": solver.as_dict(),
            "termination_contract": solver.termination_contract,
            "acceptance_policy_id": solver.acceptance_policy_id,
            "active_set_continuation_policy": solver.active_set_continuation_policy,
            "maxiter_provenance": solver.maxiter_provenance,
            "stationarity_policy": solver.stationarity_policy,
            "paper_weights": paper.as_dict(),
            "native_fps": warm.metadata.get("native_fps"),
            "input_signature": input_signature,
            "source_frame_offset": int(source_frame_offset),
            "source_frame_range": [
                int(source_frame_offset + start_frame),
                int(source_frame_offset + end_frame),
            ],
        },
    }
    if continuous:
        result["final_artifact_metadata"].update(
            {
                "base_correction_convention": "scene_local_seed_delta_exp_left",
                "continuous_profile_id": solver.profile_id,
                "continuity_schema_version": "toporetarget.trajectory_continuity.v1",
                "continuity_acceptance": True,
                "previous_final_correction_transport": True,
                "receding_horizon_window": 5,
                "paper_method": False,
                "author_exact": "unresolved",
                "engineering_extension": True,
            }
        )
    return result


def _checkpoint_status_payload(store: CheckpointStore) -> dict[str, Any]:
    assert store.manifest is not None
    manifest = store.manifest
    value = store.validate_chain(allow_incomplete=True)
    value["remaining_frames"] = max(
        0,
        int(manifest.get("frame_range", [0, 0])[1]) - int(value["next_frame"]),
    )
    value["elapsed_sessions"] = int(manifest.get("elapsed_sessions", 0))
    value.update(
        {
            "run_id": manifest.get("run_id"),
            "checkpoint_root": str(store.root),
            "input_signature": manifest.get("input_signature"),
            "solver_profile_hash": manifest.get("solver_profile_hash"),
            "execution_profile_hash": manifest.get("execution_profile_hash"),
            "resume_command": manifest.get("resume_command"),
        }
    )
    return value


def _run_checkpoint_refinement(
    *,
    canonical: Path,
    warm_start: Path,
    graph_path: Path,
    robot: str,
    collision_samples: Path | None,
    query_profile_id: str,
    coordinate_profile_id: str,
    solver_profile_id: str,
    execution_profile_id: str,
    start_frame: int,
    end_frame: int | None,
    checkpoint_root: Path,
    output: Path | None,
    asset_root: Path | None,
    resume: bool,
    max_wall_time: float | None,
    stop_after_frame: int | None,
    progress_json: Path | None,
    progress_log: Path | None,
    force: bool,
    quality_extension_path: Path | None = None,
    pause_check: Any | None = None,
    allow_shadow_while_queue_paused: bool = False,
    frame_health_gate: Callable[[dict[str, Any], list[dict[str, Any]]], str | None] | None = None,
    ready_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    sequence, warm, graph, model, surface, selected_samples = _refinement_components(
        canonical, warm_start, graph_path, robot, collision_samples, asset_root
    )
    sample_path = selected_samples or _default_collision_samples(robot)
    stop = warm.frame_count if end_frame is None else int(end_frame)
    if stop <= start_frame or stop > warm.frame_count:
        raise ValueError(f"invalid frame range [{start_frame},{stop})")
    if stop_after_frame is not None and not start_frame <= stop_after_frame < stop:
        raise ValueError("--stop-after-frame must be within the requested half-open range")
    solver = RefinementSolverProfile.load(solver_profile_id)
    execution = RefinementExecutionProfile.load(execution_profile_id)
    query = CollisionQueryProfile.load(query_profile_id)
    coordinate = RefinementCoordinateProfile.load(coordinate_profile_id)
    paper = PaperRefinementWeights.load()
    quality_extension = _load_quality_extension(quality_extension_path)
    resources = prepare_refinement_resources(
        sequence,
        graph,
        solver,
        sdf_tree_leaf_size=execution.sdf_tree_leaf_size,
        geometry_artifact_root=checkpoint_root.parents[3] / "geometry",
    )
    runtime_backends = prepare_refinement_runtime_backends(resources, execution)
    source_frame_offset = _source_frame_offset(canonical)
    signature = _refinement_input_signature(
        canonical,
        warm_start,
        graph_path,
        sample_path,
        robot,
        start_frame,
        stop,
        geometry_signature=str(resources.geometry_policy["cache_signature"]),
    )
    if quality_extension_path is not None:
        signature = f"{signature}:{sha256_file(quality_extension_path)}"
    manifest = _checkpoint_manifest(
        sequence=sequence,
        warm=warm,
        graph=graph,
        model=model,
        paper=paper,
        solver=solver,
        execution=execution,
        query=query,
        coordinate=coordinate,
        surface=surface,
        resources=resources,
        input_signature=signature,
        start_frame=start_frame,
        end_frame=stop,
        canonical=canonical,
        warm_start=warm_start,
        graph_path=graph_path,
        collision_samples=sample_path,
        checkpoint_root=checkpoint_root,
        source_frame_offset=source_frame_offset,
    )
    manifest["quality_extension"] = (
        None
        if quality_extension is None
        else {
            key: value
            for key, value in quality_extension.items()
            if key
            not in {
                "morphology_target_keypoints_scene",
                "contact_target_relative",
                "contact_target_direction",
                "contact_active",
                "contact_weights",
                "contact_regions",
            }
        }
    )
    manifest["final_artifact_metadata"]["quality_extension"] = manifest["quality_extension"]
    store = CheckpointStore.open(checkpoint_root, manifest=manifest, resume=resume)
    assert store.manifest is not None
    manifest = dict(store.manifest)
    chain = store.validate_chain()
    next_frame = int(chain["next_frame"])
    if next_frame < start_frame or next_frame > stop:
        raise CheckpointError(f"invalid next frame from checkpoint chain: {next_frame}")
    previous: tuple[np.ndarray, np.ndarray] | None = None
    previous_hash: str | None = None
    if next_frame > start_frame:
        previous_metadata, previous_arrays = store.load_frame(next_frame - 1)
        previous_hash = str(previous_metadata["per_frame_checkpoint_hash"])
        previous = (
            np.asarray(previous_arrays["base_pose_scene"], dtype=np.float64),
            np.asarray(previous_arrays["qpos"], dtype=np.float64),
        )
    started = time.perf_counter()
    frame_rows: list[dict[str, Any]] = []
    if next_frame > start_frame:
        # A recovered health gate must include already committed frames; this
        # preserves the original five-frame window instead of silently moving
        # it forward after an interruption.
        for index in range(start_frame, next_frame):
            metadata, _ = store.load_frame(index)
            frame_rows.append(metadata)
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    if ready_callback is not None:
        ready_callback()

    while next_frame < stop:
        queue_paused = paused(Path(__file__).resolve().parents[3])
        if (pause_check is not None and bool(pause_check())) or (
            queue_paused and not allow_shadow_while_queue_paused
        ):
            status = store.update_progress(
                status=PAUSE_STATE, elapsed_s=time.perf_counter() - started
            )
            status.update(_checkpoint_status_payload(store))
            status["pause_reason"] = PAUSE_STATE
            if progress_json is not None:
                _json_write(status, progress_json)
            if progress_log is not None:
                progress_log.parent.mkdir(parents=True, exist_ok=True)
                progress_log.open("a", encoding="utf-8").write(json.dumps(status) + "\n")
            return status
        if max_wall_time is not None and time.perf_counter() - started >= max_wall_time:
            status = store.update_progress(status="paused", elapsed_s=time.perf_counter() - started)
            status.update(_checkpoint_status_payload(store))
            if progress_json is not None:
                _json_write(status, progress_json)
            if progress_log is not None:
                progress_log.parent.mkdir(parents=True, exist_ok=True)
                progress_log.open("a", encoding="utf-8").write(json.dumps(status) + "\n")
            return status
        current_frame = next_frame
        callback_hash = previous_hash
        latest_metadata: dict[str, Any] | None = None

        def on_frame(
            local_index: int,
            frame_result: Any,
            context: Any,
        ) -> None:
            nonlocal callback_hash, latest_metadata
            metadata, arrays = frame_checkpoint_payload(
                local_index,
                frame_result,
                context,
                global_frame=int(graph.frame_indices[local_index]),
                source_frame=source_frame_offset + int(graph.frame_indices[local_index]),
                timestamp=float(warm.arrays["timestamps"][local_index]),
                input_signature=manifest["input_signature"],
                solver_profile=solver.as_dict(),
                execution_profile=execution.as_dict(),
                previous_checkpoint_hash=callback_hash,
            )
            callback_hash = store.save_frame(metadata, arrays)
            frame_rows.append(metadata)
            latest_metadata = metadata

        trajectory, _ = build_final_trajectory(
            sequence,
            warm,
            graph,
            model,
            surface,
            frame_profile,
            bone_profile,
            coordinate,
            query,
            solver,
            start_frame=current_frame,
            end_frame=current_frame + 1,
            initial_previous=previous,
            warm_artifact_hash=artifact_hash(warm_start),
            graph_artifact_hash=interaction_artifact_hash(graph_path),
            resources=resources,
            runtime_backends=runtime_backends,
            frame_callback=on_frame,
            source_frame_offset=source_frame_offset,
            execution_profile=execution,
            quality_extension=quality_extension,
        )
        previous = (
            np.asarray(trajectory.arrays["base_pose_scene"][-1], dtype=np.float64),
            np.asarray(trajectory.arrays["qpos"][-1], dtype=np.float64),
        )
        previous_hash = callback_hash
        next_frame += 1
        store.update_progress(status="paused", elapsed_s=time.perf_counter() - started)
        if frame_health_gate is not None:
            if latest_metadata is None:
                raise RuntimeError("final refinement emitted no checkpoint metadata")
            failure_reason = frame_health_gate(latest_metadata, frame_rows)
            if failure_reason is not None:
                status = store.update_progress(
                    status="PAUSED_BY_STAGE12_HEALTH_GATE",
                    elapsed_s=time.perf_counter() - started,
                )
                status.update(_checkpoint_status_payload(store))
                status["pause_reason"] = failure_reason
                if progress_json is not None:
                    _json_write(status, progress_json)
                if progress_log is not None:
                    progress_log.parent.mkdir(parents=True, exist_ok=True)
                    progress_log.open("a", encoding="utf-8").write(json.dumps(status) + "\n")
                return status
        if stop_after_frame is not None and current_frame >= stop_after_frame:
            status = store.update_progress(status="paused", elapsed_s=time.perf_counter() - started)
            status.update(_checkpoint_status_payload(store))
            status["stop_after_frame"] = stop_after_frame
            if progress_json is not None:
                _json_write(status, progress_json)
            return status

    status = store.update_progress(status="complete", elapsed_s=time.perf_counter() - started)
    status.update(_checkpoint_status_payload(store))
    status["frame_rows"] = frame_rows
    if output is not None:
        destination = store.assemble(output, force=force)
        status["final_artifact"] = str(destination)
    if progress_json is not None:
        _json_write(status, progress_json)
    if progress_log is not None:
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        progress_log.open("a", encoding="utf-8").write(json.dumps(status) + "\n")
    return status


@app.command("inspect-query-set")
def inspect_query_set_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    object_id: str = typer.Option("primary", "--object-id"),
    frame: int = typer.Option(0, "--frame", min=0),
    query_profile: str = typer.Option(ACTIVE_QUERY_PROFILE_ID, "--query-profile"),
    json_report: Path | None = typer.Option(None, "--json"),
    csv_report: Path | None = typer.Option(None, "--csv"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        sequence = load_hoi_sequence(canonical)
        warm = load_warm_start(warm_start)
        model = _load_robot(robot, asset_root)
        surface = load_robot_surface_samples(collision_samples or _default_collision_samples(robot))
        if frame >= warm.frame_count:
            raise typer.BadParameter("frame is outside warm-start artifact")
        obj = _object_for_graph(sequence, object_id)
        from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend

        backend = build_signed_distance_backend(
            obj.mesh.vertices_local, obj.mesh.faces, sign_mode="strict"
        )
        points = dynamic_collision_points_numpy(
            model, surface, warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame]
        )
        result = backend.query_scene(points, obj.pose_scene.pose_scene[frame])
        profile = CollisionQueryProfile.load(query_profile)
        query = build_query_set(result.signed_distance, surface.geometry_ids, profile)
        report = {
            "status": "pass",
            "canonical": str(canonical),
            "warm_start": str(warm_start),
            "robot": robot,
            "object_id": obj.object_id,
            "frame": frame,
            "total_collision_samples": int(len(points)),
            "initial_query_count": query.count,
            "per_link_counts": {
                str(name): int(
                    np.count_nonzero(
                        np.isin(query.sample_ids, np.flatnonzero(surface.link_names == name))
                    )
                )
                for name in sorted(set(surface.link_names.tolist()))
            },
            "queries": [
                {
                    "sample_id": int(sample_id),
                    "link_name": str(surface.link_names[sample_id]),
                    "geometry_id": str(surface.geometry_ids[sample_id]),
                    "initial_signed_distance_m": float(result.signed_distance[sample_id]),
                    "active_set_round": int(round_value),
                    "inclusion_reason": reason,
                }
                for sample_id, round_value, reason in zip(
                    query.sample_ids, query.active_round, query.inclusion_reasons, strict=True
                )
            ],
            "min_distance_m": float(np.min(result.signed_distance)),
            "max_distance_m": float(np.max(result.signed_distance)),
            "sign_valid": bool(np.all(result.sign_valid)),
            "query_hash": query.query_hash,
            "profile": profile.as_dict(),
        }
        _json_write(report, json_report)
        if csv_report is not None:
            csv_report.parent.mkdir(parents=True, exist_ok=True)
            with csv_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["sample_id", "link_name", "geometry_id", "signed_distance_m", "reason"]
                )
                for sample_id, reason in zip(
                    query.sample_ids, query.inclusion_reasons, strict=True
                ):
                    writer.writerow(
                        [
                            sample_id,
                            surface.link_names[sample_id],
                            surface.geometry_ids[sample_id],
                            result.signed_distance[sample_id],
                            reason,
                        ]
                    )
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"inspect-query-set failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare-query-profiles")
def compare_query_profiles_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    graph: Path = typer.Option(..., "--graph"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    frames: list[int] = typer.Option([0, 29, 59], "--frames"),
    report: Path = typer.Option(..., "--report"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        sequence, warm, graph_trajectory, model, surface, _ = _refinement_components(
            canonical, warm_start, graph, robot, collision_samples, asset_root
        )
        from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend

        obj = _object_for_graph(sequence, str(graph_trajectory.metadata["object_id"]))
        sdf = build_signed_distance_backend(
            obj.mesh.vertices_local, obj.mesh.faces, sign_mode="strict"
        )
        profiles = [
            CollisionQueryProfile.load(FULL_QUERY_PROFILE_ID),
            CollisionQueryProfile.load(ACTIVE_QUERY_PROFILE_ID),
        ]
        output: dict[str, Any] = {"canonical": str(canonical), "frames": {}}
        for frame in frames:
            if frame < 0 or frame >= warm.frame_count:
                continue
            points = dynamic_collision_points_numpy(
                model, surface, warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame]
            )
            distances = sdf.query_scene(points, obj.pose_scene.pose_scene[frame]).signed_distance
            output["frames"][str(frame)] = {
                profile.profile_id: {
                    "query_count": build_query_set(distances, surface.geometry_ids, profile).count,
                    "min_distance_m": float(np.min(distances)),
                }
                for profile in profiles
            }
        _json_write(output, report)
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"compare-query-profiles failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("profile-refinement")
def profile_refinement_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    graph: Path = typer.Option(..., "--graph"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    frames: list[int] = typer.Option([0, 12, 29], "--frames", min=0),
    query_profile: str = typer.Option(ACTIVE_QUERY_PROFILE_ID, "--query-profile"),
    coordinate_profile: str = typer.Option("local_seed_delta_v1", "--coordinate-profile"),
    solver_profile: str = typer.Option(
        "scipy_slsqp_active_set_contact_rich_v2", "--solver-profile"
    ),
    execution_profile: str = typer.Option(
        "cached_checkpoint_cpu_float64_v1", "--execution-profile"
    ),
    classification_override: str | None = typer.Option(None, "--classification"),
    output_root: Path = typer.Option(..., "--output-root"),
    torch_profiler_enabled: bool = typer.Option(True, "--torch-profiler/--no-torch-profiler"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    """Profile selected real frames without changing the refinement contract."""

    try:
        sequence, warm, graph_trajectory, model, surface, _ = _refinement_components(
            canonical, warm_start, graph, robot, collision_samples, asset_root
        )
        solver = RefinementSolverProfile.load(solver_profile)
        execution = RefinementExecutionProfile.load(execution_profile)
        query = CollisionQueryProfile.load(query_profile)
        coordinate = RefinementCoordinateProfile.load(coordinate_profile)
        frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
        bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
        resources = prepare_refinement_resources(
            sequence,
            graph_trajectory,
            solver,
            sdf_tree_leaf_size=execution.sdf_tree_leaf_size,
        )
        source_frame_offset = _source_frame_offset(canonical)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "cprofile").mkdir(exist_ok=True)
        (output_root / "torch_profiler").mkdir(exist_ok=True)
        rows: list[dict[str, Any]] = []
        callback_rows: list[dict[str, Any]] = []
        cache_rows: list[dict[str, Any]] = []
        torch_status: dict[str, Any] = {"available": False, "frames": {}}
        for requested_frame in sorted(set(int(frame) for frame in frames)):
            matches = np.flatnonzero(graph_trajectory.frame_indices == requested_frame)
            if len(matches):
                local_frame = int(matches[0])
                graph_frame = int(graph_trajectory.frame_indices[local_frame])
                global_frame = (
                    graph_frame
                    if graph_frame >= source_frame_offset
                    else source_frame_offset + graph_frame
                )
            elif 0 <= requested_frame - source_frame_offset < warm.frame_count:
                local_frame = requested_frame - source_frame_offset
                global_frame = requested_frame
            elif 0 <= requested_frame < warm.frame_count:
                local_frame = requested_frame
                global_frame = source_frame_offset + int(
                    graph_trajectory.frame_indices[local_frame]
                )
            else:
                raise ValueError(f"frame is outside warm-start/graph artifact: {requested_frame}")
            classification = classification_override or (
                "status9_frame"
                if global_frame == 240
                else "approach"
                if global_frame in {238, 239}
                else "pre_contact"
                if global_frame < 240
                else "contact_rich"
            )
            cprofile_path = output_root / "cprofile" / f"frame_{local_frame:06d}.prof"
            profile = cProfile.Profile()
            torch_profiler = None
            torch_module: Any = None
            started = time.perf_counter()
            profile.enable()
            try:
                try:
                    import torch

                    torch_module = torch
                    if torch_profiler_enabled:
                        torch_status["available"] = True
                    else:
                        torch_status["frames"][str(global_frame)] = {"status": "disabled_by_cli"}
                except (ImportError, RuntimeError, AttributeError) as exc:
                    torch_status["frames"][str(global_frame)] = {
                        "status": "unavailable",
                        "reason": str(exc),
                    }
                trajectory, diagnostics = build_final_trajectory(
                    sequence,
                    warm,
                    graph_trajectory,
                    model,
                    surface,
                    frame_profile,
                    bone_profile,
                    coordinate,
                    query,
                    solver,
                    start_frame=local_frame,
                    end_frame=local_frame + 1,
                    warm_artifact_hash=artifact_hash(warm_start),
                    graph_artifact_hash=interaction_artifact_hash(graph),
                    resources=resources,
                    continue_on_failure=True,
                    source_frame_offset=source_frame_offset,
                    execution_profile=execution,
                )
            finally:
                if torch_profiler is not None:
                    torch_profiler.__exit__(None, None, None)
                profile.disable()
                profile.dump_stats(str(cprofile_path))
                with (output_root / "cprofile" / f"frame_{local_frame:06d}.txt").open(
                    "w", encoding="utf-8"
                ) as handle:
                    stats = pstats.Stats(profile, stream=handle)
                    stats.strip_dirs().sort_stats("cumtime").print_stats(50)
            elapsed = time.perf_counter() - started
            if torch_module is not None and torch_profiler_enabled:
                trace_path = output_root / "torch_profiler" / f"frame_{local_frame:06d}.json"
                # Profiling the whole SciPy/Torch callback stream creates an
                # unbounded multi-GB event tree. Record a bounded Torch smoke
                # trace here; the real solve's detailed timings are provided by
                # the internal timers and cProfile above.
                with torch_module.profiler.profile(
                    activities=[torch_module.profiler.ProfilerActivity.CPU],
                    record_shapes=False,
                    profile_memory=False,
                ) as smoke_profiler:
                    smoke_value = torch_module.zeros(8, dtype=torch_module.float64)
                    smoke_value.add_(1.0)
                _json_write(
                    {
                        "schema_version": "toporetarget.torch_profiler_smoke.v1",
                        "format": "key_averages_json",
                        "events": [
                            {
                                "key": item.key,
                                "cpu_time_total_us": float(item.cpu_time_total),
                                "cpu_time_avg_us": float(item.cpu_time_total / max(item.count, 1)),
                                "count": int(item.count),
                            }
                            for item in smoke_profiler.key_averages()
                        ],
                        "real_solve_trace_omitted_resource_guard": True,
                    },
                    trace_path,
                )
                torch_status["frames"][str(global_frame)] = {
                    "status": "bounded_smoke",
                    "trace": str(trace_path),
                    "format": "key_averages_json",
                }
            frame_row = dict(diagnostics["performance"][0])
            final_arrays = trajectory.arrays
            frame_row.update(
                {
                    "classification": classification,
                    "global_frame": global_frame,
                    "local_frame": local_frame,
                    "wall_time_s": elapsed,
                    "accepted": bool(trajectory.arrays["accepted"][0]),
                    "optimizer_status_code": int(trajectory.arrays["optimizer_status_code"][0]),
                    "profile_id": solver.profile_id,
                    "profile_hash": solver.profile_hash,
                    "execution_profile": execution.as_dict(),
                    "cprofile": str(cprofile_path),
                    "result_fingerprint": {
                        "final_artifact_hash": final_artifact_hash(trajectory),
                        "qpos": np.asarray(final_arrays["qpos"][0], dtype=np.float64).tolist(),
                        "base_pose_scene": np.asarray(
                            final_arrays["base_pose_scene"][0], dtype=np.float64
                        ).tolist(),
                        "final_objective": float(final_arrays["final_objective"][0]),
                        "min_signed_distance": float(
                            np.min(final_arrays["full_signed_distance"][0])
                        ),
                    },
                    "sdf_backend": trajectory.metadata.get("sdf_backend", {}),
                    "query_summaries": trajectory.metadata.get("query_summaries", []),
                }
            )
            rows.append(frame_row)
            callback_rows.append(
                {
                    "global_frame": global_frame,
                    "classification": classification,
                    "wall_time_s": elapsed,
                    "objective_calls": frame_row["objective_calls"],
                    "objective_jacobian_calls": frame_row["objective_jacobian_calls"],
                    "constraint_calls": frame_row["constraint_calls"],
                    "constraint_jacobian_calls": frame_row["constraint_jacobian_calls"],
                    "full_audit_call_count": frame_row["full_audit_call_count"],
                }
            )
            cache_rows.append(
                {
                    "global_frame": global_frame,
                    **frame_row.get("cache", {}),
                }
            )
            summary_path = output_root / f"profile_{classification}.json"
            _json_write(
                {
                    "status": "pass",
                    "frame": frame_row,
                    "timers": frame_row.get("timers", {}),
                    "resource_counts": diagnostics["resource_counts"],
                },
                summary_path,
            )
        profiled_labels = {str(row["classification"]) for row in rows}
        for label in ("pre_contact", "approach", "contact_rich", "status9_frame"):
            missing_path = output_root / f"profile_{label}.json"
            if label not in profiled_labels and not missing_path.exists():
                _json_write(
                    {
                        "status": "not_profiled",
                        "classification": label,
                        "reason": "no selected frame mapped to this benchmark class",
                    },
                    missing_path,
                )
        timer_totals: dict[str, float] = {}
        for row in rows:
            for name, value in row.get("timers", {}).get("elapsed_s", {}).items():
                timer_totals[name] = timer_totals.get(name, 0.0) + float(value)
        bottlenecks = sorted(timer_totals.items(), key=lambda item: item[1], reverse=True)
        _json_write(
            {
                "status": "pass",
                "frames": rows,
                "timer_totals_s": timer_totals,
                "top_three": [
                    {"name": name, "elapsed_s": elapsed} for name, elapsed in bottlenecks[:3]
                ],
                "resource_counts": resources.build_counts,
            },
            output_root / "bottleneck_summary.json",
        )
        _json_write({"frames": callback_rows}, output_root / "callback_counts.json")
        _json_write({"frames": cache_rows}, output_root / "evaluation_cache_stats.json")
        _json_write(
            {
                "status": "pass",
                "execution_profile": execution.as_dict(),
                "solver_profile": solver.as_dict(),
                "resource_counts": resources.build_counts,
                "full_audit_in_inner_callbacks": False,
                "report_or_artifact_io_in_inner_loop": False,
                "active_set_initialization": "stage7_seed_v1",
                "cache_identity": ["frame_id", "exact_x", "query_set_hash", "context_hash"],
                "numpy_torch_boundary": (
                    "SciPy callback arrays are float64; Torch tensors are created per candidate"
                ),
                "device_synchronization": "CPU float64 execution profile; no CUDA synchronization",
                "checkpoint_present_before_profile": False,
                "final_artifact_write": (
                    "after all accepted frames, or explicit checkpoint assembly"
                ),
            },
            output_root / "execution_path_audit.json",
        )
        py_spy = shutil.which("py-spy")
        _json_write(
            {
                "py_spy": {
                    "available": py_spy is not None,
                    "path": py_spy,
                    "status": "available_but_external_attach_not_started"
                    if py_spy
                    else "unavailable",
                },
                "torch_profiler": torch_status,
            },
            output_root / "profiler_availability.json",
        )
        benchmark_rows: list[dict[str, Any]] = []
        for row in rows:
            try:
                benchmark_rows.append(
                    {
                        "global_frame": row["global_frame"],
                        "local_frame": row["local_frame"],
                        "classification": row["classification"],
                        "canonical": str(canonical),
                        "warm_start": str(warm_start),
                        "graph": str(graph),
                        "collision_samples": str(
                            collision_samples or _default_collision_samples(robot)
                        ),
                        "robot": robot,
                        "input_signature": _refinement_input_signature(
                            canonical,
                            warm_start,
                            graph,
                            collision_samples or _default_collision_samples(robot),
                            robot,
                            row["local_frame"],
                            row["local_frame"] + 1,
                        ),
                        "solver_profile_id": solver.profile_id,
                        "expected_strict_result": bool(row["accepted"]),
                    }
                )
            except (OSError, ValueError) as exc:
                benchmark_rows.append(
                    {
                        "global_frame": row["global_frame"],
                        "local_frame": row["local_frame"],
                        "classification": row["classification"],
                        "status": "provenance_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        _json_write(
            {
                "status": "pass",
                "source": "user-selected Stage 9.1 benchmark frames",
                "frames": benchmark_rows,
            },
            output_root / "benchmark_frames.json",
        )
        _json_write({"status": "pass", "frames": rows}, None)
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"profile-refinement failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("refine")
def refine_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    graph: Path = typer.Option(..., "--graph"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    query_profile: str = typer.Option(ACTIVE_QUERY_PROFILE_ID, "--query-profile"),
    coordinate_profile: str = typer.Option("local_seed_delta_v1", "--coordinate-profile"),
    solver_profile: str = typer.Option(SOLVER_PROFILE_ID, "--solver-profile"),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    resume_from: Path | None = typer.Option(None, "--resume-from"),
    checkpoint_root: Path | None = typer.Option(None, "--checkpoint-root"),
    resume: bool = typer.Option(False, "--resume"),
    max_wall_time: float | None = typer.Option(None, "--max-wall-time", min=0.0),
    stop_after_frame: int | None = typer.Option(None, "--stop-after-frame", min=0),
    progress_json: Path | None = typer.Option(None, "--progress-json"),
    progress_log: Path | None = typer.Option(None, "--progress-log"),
    execution_profile: str = typer.Option(
        "cached_checkpoint_cpu_float64_v1", "--execution-profile"
    ),
    output: Path = typer.Option(..., "--output"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    force: bool = typer.Option(False, "--force"),
    quality_extension: Path | None = typer.Option(None, "--quality-extension"),
) -> None:
    try:
        if checkpoint_root is not None:
            value = _run_checkpoint_refinement(
                canonical=canonical,
                warm_start=warm_start,
                graph_path=graph,
                robot=robot,
                collision_samples=collision_samples,
                query_profile_id=query_profile,
                coordinate_profile_id=coordinate_profile,
                solver_profile_id=solver_profile,
                execution_profile_id=execution_profile,
                start_frame=start_frame,
                end_frame=end_frame,
                checkpoint_root=checkpoint_root,
                output=output,
                asset_root=asset_root,
                resume=resume,
                max_wall_time=max_wall_time,
                stop_after_frame=stop_after_frame,
                progress_json=progress_json,
                progress_log=progress_log,
                force=force,
                quality_extension_path=quality_extension,
            )
            _json_write(value, None)
            return
        sequence, warm, graph_trajectory, model, surface, _ = _refinement_components(
            canonical, warm_start, graph, robot, collision_samples, asset_root
        )
        execution = RefinementExecutionProfile.load(execution_profile)
        quality_extension_spec = _load_quality_extension(quality_extension)
        initial_previous = None
        if resume_from is not None:
            previous = load_final_trajectory(resume_from)
            if start_frame <= 0 or int(previous.arrays["frame_indices"][-1]) != start_frame - 1:
                raise ValueError("--resume-from must end at start_frame - 1")
            initial_previous = (
                previous.arrays["base_pose_scene"][-1],
                previous.arrays["qpos"][-1],
            )
        trajectory, diagnostics = build_final_trajectory(
            sequence,
            warm,
            graph_trajectory,
            model,
            surface,
            load_frame_profile("canonical_keypoint_wrist_v1"),
            load_bone_profile("mediapipe21_full_finger_chain_v1"),
            RefinementCoordinateProfile.load(coordinate_profile),
            CollisionQueryProfile.load(query_profile),
            RefinementSolverProfile.load(solver_profile),
            start_frame=start_frame,
            end_frame=end_frame,
            initial_previous=initial_previous,
            warm_artifact_hash=artifact_hash(warm_start),
            graph_artifact_hash=interaction_artifact_hash(graph),
            source_frame_offset=_source_frame_offset(canonical),
            execution_profile=execution,
            quality_extension=quality_extension_spec,
        )
        trajectory.metadata["artifact_hash"] = final_artifact_hash(trajectory)
        save_final_trajectory(trajectory, output, force=force)
        _json_write(
            {
                "status": "pass",
                "output": str(output),
                "artifact_hash": trajectory.metadata["artifact_hash"],
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
        InteractionArtifactError,
        CheckpointError,
    ) as exc:
        typer.echo(f"refine failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("checkpoint-status")
def checkpoint_status_command(
    checkpoint_root: Path = typer.Option(..., "--checkpoint-root"),
) -> None:
    try:
        _json_write(_checkpoint_status_payload(CheckpointStore(checkpoint_root)), None)
    except (CheckpointError, OSError, ValueError) as exc:
        typer.echo(f"checkpoint-status failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-checkpoints")
def validate_checkpoints_command(
    checkpoint_root: Path = typer.Option(..., "--checkpoint-root"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    try:
        value = CheckpointStore(checkpoint_root).validate_chain()
        _json_write(value, report)
        if report is None:
            _json_write(value, None)
        if not value["chain_pass"]:
            raise typer.Exit(code=1)
    except (CheckpointError, OSError, ValueError) as exc:
        typer.echo(f"validate-checkpoints failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("assemble-refinement")
def assemble_refinement_command(
    checkpoint_root: Path = typer.Option(..., "--checkpoint-root"),
    output: Path = typer.Option(..., "--output"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    try:
        store = CheckpointStore(checkpoint_root)
        destination = store.assemble(output, force=force)
        _json_write({"status": "pass", "output": str(destination)}, None)
    except (CheckpointError, OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"assemble-refinement failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare-refinement-runs")
def compare_refinement_runs_command(
    left: Path = typer.Option(..., "--left"),
    right: Path = typer.Option(..., "--right"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    try:
        left_artifact = load_final_trajectory(left)
        right_artifact = load_final_trajectory(right)
        names = sorted(set(left_artifact.arrays) | set(right_artifact.arrays))
        rows: dict[str, Any] = {}
        equal = True
        for name in names:
            if name == "solve_time_s":
                rows[name] = {"status": "excluded_nondeterministic_runtime_field"}
                continue
            if name not in left_artifact.arrays or name not in right_artifact.arrays:
                rows[name] = {"status": "missing"}
                equal = False
                continue
            lhs = np.asarray(left_artifact.arrays[name])
            rhs = np.asarray(right_artifact.arrays[name])
            row: dict[str, Any]
            if lhs.shape != rhs.shape:
                row = {"shape_equal": False, "array_equal": False}
            elif lhs.dtype.kind in "OUS" or rhs.dtype.kind in "OUS":
                row = {"shape_equal": True, "array_equal": bool(np.array_equal(lhs, rhs))}
            else:
                left_float = lhs.astype(np.float64)
                right_float = rhs.astype(np.float64)
                difference = np.abs(left_float - right_float)
                difference = np.where(np.isnan(left_float) & np.isnan(right_float), 0.0, difference)
                row = {
                    "shape_equal": lhs.shape == rhs.shape,
                    "array_equal": bool(np.array_equal(lhs, rhs, equal_nan=True)),
                    "max_abs_difference": float(np.max(difference, initial=0.0)),
                }
            rows[name] = row
            equal = equal and bool(row.get("array_equal", False))
        value = {
            "status": "pass" if equal else "fail",
            "left": str(left),
            "right": str(right),
            "metadata_fields_excluded": ["artifact_hash", "created_at", "checkpoint_root"],
            "array_fields_excluded": ["solve_time_s"],
            "array_comparison": rows,
        }
        _json_write(value, report)
        if report is None:
            _json_write(value, None)
        if not equal:
            raise typer.Exit(code=1)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"compare-refinement-runs failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _validation_payload(
    canonical: Path,
    warm_start: Path,
    graph_path: Path | None,
    final_path: Path,
    robot: str,
    collision_samples: Path | None,
    asset_root: Path | None,
) -> dict[str, Any]:
    sequence = load_hoi_sequence(canonical)
    warm = load_warm_start(warm_start)
    final = load_final_trajectory(final_path)
    graph = None if graph_path is None else load_interaction_graph(graph_path)
    model = _load_robot(robot, asset_root)
    surface = load_robot_surface_samples(collision_samples or _default_collision_samples(robot))
    obj = _object_for_graph(sequence, str(final.metadata["object_id"]))
    from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend

    sdf = build_signed_distance_backend(obj.mesh.vertices_local, obj.mesh.faces, sign_mode="strict")
    reference_mesh_hash = sdf.mesh_hash
    lower, upper = model.joint_lower, model.joint_upper
    frame_results: list[dict[str, Any]] = []
    for index, global_frame in enumerate(final.arrays["frame_indices"].tolist()):
        points = dynamic_collision_points_numpy(
            model, surface, final.arrays["qpos"][index], final.arrays["base_pose_scene"][index]
        )
        result = sdf.query_scene(points, obj.pose_scene.pose_scene[int(global_frame)])
        q0 = int(final.arrays["query_offsets"][index])
        q1 = int(final.arrays["query_offsets"][index + 1])
        ids = final.arrays["query_ids_concat"][q0:q1]
        s0 = int(final.arrays["slack_offsets"][index])
        s1 = int(final.arrays["slack_offsets"][index + 1])
        slack = final.arrays["slack_concat"][s0:s1]
        hard = result.signed_distance[ids] + 0.030
        soft = result.signed_distance[ids] + slack + 0.001
        unqueried = np.setdiff1d(np.arange(len(points)), ids, assume_unique=True)
        optimizer_converged = bool(
            final.arrays.get("optimizer_converged", final.arrays["solver_success"])[index]
        )
        qpos_bounds_pass = bool(
            np.all(final.arrays["qpos"][index] >= lower - 1e-10)
            and np.all(final.arrays["qpos"][index] <= upper + 1e-10)
        )
        slack_bounds_pass = bool(np.all(slack >= -1e-10) and np.all(slack <= 0.029 + 1e-10))
        active_constraints_feasible = bool(
            np.min(hard, initial=np.inf) >= -1e-6 and np.min(soft, initial=np.inf) >= -1e-6
        )
        full_surface_hard_audit_pass = bool(np.min(result.signed_distance) >= -0.030 - 1e-6)
        full_surface_soft_audit_pass = bool(
            len(unqueried) == 0 or np.all(result.signed_distance[unqueried] >= -0.001 - 1e-6)
        )
        active_set_converged = bool(final.arrays["active_set_converged"][index])
        all_values_finite = bool(
            np.all(
                np.isfinite(
                    np.concatenate(
                        [
                            final.arrays["qpos"][index].reshape(-1),
                            slack.reshape(-1),
                            result.signed_distance.reshape(-1),
                        ]
                    )
                )
            )
        )
        accepted = bool(final.arrays.get("accepted", final.arrays["solver_success"])[index])
        frame_results.append(
            {
                "frame": int(global_frame),
                "solver_success": bool(final.arrays["solver_success"][index]),
                "optimizer_converged": optimizer_converged,
                "optimizer_status_code": int(
                    final.arrays.get("optimizer_status_code", final.arrays["solver_status"])[index]
                ),
                "optimizer_message": _decode_bytes(
                    final.arrays.get(
                        "optimizer_message", np.full(final.frame_count, b"", dtype="S256")
                    )[index]
                ),
                "optimizer_iterations": int(
                    final.arrays.get("optimizer_iterations", final.arrays["iterations"])[index]
                ),
                "optimizer_function_evaluations": int(
                    final.arrays.get(
                        "optimizer_function_evaluations", final.arrays["function_evaluations"]
                    )[index]
                ),
                "optimizer_jacobian_evaluations": int(
                    final.arrays.get(
                        "optimizer_jacobian_evaluations", final.arrays["jacobian_evaluations"]
                    )[index]
                ),
                "qpos_bounds_pass": qpos_bounds_pass,
                "slack_bounds_pass": slack_bounds_pass,
                "active_constraints_feasible": active_constraints_feasible,
                "full_surface_hard_audit_pass": full_surface_hard_audit_pass,
                "full_surface_soft_audit_pass": full_surface_soft_audit_pass,
                "active_set_converged": active_set_converged,
                "all_values_finite": all_values_finite,
                "stationarity_checked": bool(
                    final.arrays.get(
                        "stationarity_checked", np.zeros(final.frame_count, dtype=bool)
                    )[index]
                ),
                "stationarity_residual": None
                if "stationarity_residual" not in final.arrays
                or not np.isfinite(final.arrays["stationarity_residual"][index])
                else float(final.arrays["stationarity_residual"][index]),
                "accepted": accepted,
                "acceptance_policy_id": final.metadata.get(
                    "acceptance_policy_id", "legacy_solver_success"
                ),
                "acceptance_reason": _decode_bytes(
                    final.arrays.get(
                        "acceptance_reason", np.full(final.frame_count, b"legacy", dtype="S512")
                    )[index]
                ),
                "min_hard_residual_m": float(np.min(hard)),
                "min_soft_residual_m": float(np.min(soft)),
                "full_min_signed_distance_m": float(np.min(result.signed_distance)),
                "full_max_penetration_m": float(max(0.0, -np.min(result.signed_distance))),
                "unqueried_soft_violation_count": int(
                    np.count_nonzero(result.signed_distance[unqueried] < -0.001 - 1e-6)
                ),
                "query_count": int(len(ids)),
                "max_slack_m": float(np.max(slack, initial=0.0)),
            }
        )
    graph_hash = interaction_artifact_hash(graph_path) if graph_path is not None else None
    passed = bool(
        len(frame_results) == final.frame_count
        and all(
            item["accepted"]
            and item["optimizer_converged"]
            and item["qpos_bounds_pass"]
            and item["slack_bounds_pass"]
            and item["active_constraints_feasible"]
            and item["full_surface_hard_audit_pass"]
            and item["full_surface_soft_audit_pass"]
            and item["active_set_converged"]
            and item["all_values_finite"]
            for item in frame_results
        )
        and all(
            item["min_hard_residual_m"] >= -1e-6 and item["min_soft_residual_m"] >= -1e-6
            for item in frame_results
        )
        and all(item["full_min_signed_distance_m"] >= -0.030 - 1e-6 for item in frame_results)
        and all(item["unqueried_soft_violation_count"] == 0 for item in frame_results)
        and final.metadata.get("source_canonical_hash")
        in {None, warm.metadata.get("source_cache_hash")}
        and final.metadata.get("warm_start_artifact_hash") in {None, artifact_hash(warm_start)}
        and (graph is None or final.metadata.get("graph_artifact_hash") in {None, graph_hash})
        and final.metadata.get("object_mesh_hash") == reference_mesh_hash
        and final.metadata.get("collision_surface_profile_hash") == surface.profile.profile_hash
    )
    source_hash_match = final.metadata.get("source_canonical_hash") in {
        None,
        warm.metadata.get("source_cache_hash"),
    }
    warm_hash_match = final.metadata.get("warm_start_artifact_hash") in {
        None,
        artifact_hash(warm_start),
    }
    graph_hash_match = graph is None or final.metadata.get("graph_artifact_hash") in {
        None,
        graph_hash,
    }
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "final": str(final_path),
        "frame_count": final.frame_count,
        "frames": frame_results,
        "source_hash_match": source_hash_match,
        "warm_start_hash_match": warm_hash_match,
        "graph_hash_match": graph_hash_match,
        "object_mesh_hash_match": final.metadata.get("object_mesh_hash") == reference_mesh_hash,
        "collision_surface_profile_hash_match": final.metadata.get("collision_surface_profile_hash")
        == surface.profile.profile_hash,
        "artifact_schema": final.schema_version,
        "solver_profile_id": final.metadata.get("solver_profile_id"),
        "solver_profile_hash": final.metadata.get("solver_profile_hash"),
        "acceptance_policy_id": final.metadata.get("acceptance_policy_id"),
        "termination_contract": final.metadata.get("termination_contract"),
        "source_integrity_pass": bool(
            source_hash_match
            and warm_hash_match
            and graph_hash_match
            and final.metadata.get("object_mesh_hash") == reference_mesh_hash
            and final.metadata.get("collision_surface_profile_hash") == surface.profile.profile_hash
        ),
    }


@app.command("validate-refinement")
def validate_refinement_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    graph: Path = typer.Option(..., "--graph"),
    final: Path = typer.Option(..., "--final"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    report: Path = typer.Option(..., "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        value = _validation_payload(
            canonical, warm_start, graph, final, robot, collision_samples, asset_root
        )
        _json_write(value, report)
        if csv_report is not None:
            csv_report.parent.mkdir(parents=True, exist_ok=True)
            with csv_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(value["frames"][0]))
                writer.writeheader()
                writer.writerows(value["frames"])
        _json_write(value, None)
        if not value["pass"]:
            raise typer.Exit(code=1)
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"validate-refinement failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("audit-penetration")
def audit_penetration_command(
    canonical: Path = typer.Option(..., "--canonical"),
    final: Path = typer.Option(..., "--final"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    report: Path = typer.Option(..., "--report"),
    csv_report: Path | None = typer.Option(None, "--csv"),
    warm_start: Path | None = typer.Option(None, "--warm-start"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        if warm_start is None:
            raise typer.BadParameter(
                "audit-penetration requires --warm-start for source provenance"
            )
        value = _validation_payload(
            canonical, warm_start, None, final, robot, collision_samples, asset_root
        )
        _json_write(value, report)
        if csv_report is not None:
            csv_report.parent.mkdir(parents=True, exist_ok=True)
            with csv_report.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(value["frames"][0]))
                writer.writeheader()
                writer.writerows(value["frames"])
        _json_write(value, None)
        if not value["pass"]:
            raise typer.Exit(code=1)
    except (StorageError, ValueError, OSError, RuntimeError, WarmStartArtifactError) as exc:
        typer.echo(f"audit-penetration failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("compare-solvers")
def compare_solvers_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    graph: Path = typer.Option(..., "--graph"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    frames: list[int] = typer.Option([0, 29, 59], "--frames"),
    report: Path = typer.Option(..., "--report"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
) -> None:
    try:
        sequence, warm, graph_trajectory, model, surface, _ = _refinement_components(
            canonical, warm_start, graph, robot, collision_samples, asset_root
        )
        values: list[dict[str, Any]] = []
        for frame in frames:
            if frame >= warm.frame_count:
                continue
            row: dict[str, Any] = {"frame": frame}
            for query_id, solver_id in (
                (ACTIVE_QUERY_PROFILE_ID, SOLVER_PROFILE_ID),
                (FULL_QUERY_PROFILE_ID, FULL_SOLVER_PROFILE_ID),
            ):
                started = __import__("time").perf_counter()
                trajectory, _ = build_final_trajectory(
                    sequence,
                    warm,
                    graph_trajectory,
                    model,
                    surface,
                    load_frame_profile("canonical_keypoint_wrist_v1"),
                    load_bone_profile("mediapipe21_full_finger_chain_v1"),
                    RefinementCoordinateProfile.load("local_seed_delta_v1"),
                    CollisionQueryProfile.load(query_id),
                    RefinementSolverProfile.load(solver_id),
                    start_frame=frame,
                    end_frame=frame + 1,
                    warm_artifact_hash=artifact_hash(warm_start),
                    graph_artifact_hash=interaction_artifact_hash(graph),
                )
                row[query_id] = {
                    "query_count": int(trajectory.arrays["query_offsets"][1]),
                    "total_objective": float(trajectory.arrays["total_objective"][0]),
                    "runtime_s": __import__("time").perf_counter() - started,
                    "min_full_signed_distance_m": float(
                        trajectory.arrays["min_full_signed_distance"][0]
                    ),
                }
            values.append(row)
        _json_write({"frames": values}, report)
    except (
        StorageError,
        ValueError,
        OSError,
        RuntimeError,
        WarmStartArtifactError,
        InteractionArtifactError,
    ) as exc:
        typer.echo(f"compare-solvers failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-refinement")
def visualize_refinement_command(
    canonical: Path = typer.Option(..., "--canonical"),
    warm_start: Path = typer.Option(..., "--warm-start"),
    graph: Path = typer.Option(..., "--graph"),
    final: Path = typer.Option(..., "--final"),
    robot: str = typer.Option("artimano_rh", "--robot"),
    view: str = typer.Option("scene", "--view", help="scene or object"),
    frame: int = typer.Option(0, "--frame", min=0),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int | None = typer.Option(None, "--end-frame", min=1),
    output: Path | None = typer.Option(None, "--output"),
    interactive: bool = typer.Option(False, "--interactive"),
    report: Path | None = typer.Option(None, "--report"),
    collision_samples: Path | None = typer.Option(None, "--collision-samples"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    show_source_hand: bool = typer.Option(True, "--show-source-hand/--hide-source-hand"),
    show_warm_start: bool = typer.Option(True, "--show-warm-start/--hide-warm-start"),
    show_final: bool = typer.Option(True, "--show-final/--hide-final"),
    show_object: bool = typer.Option(True, "--show-object/--hide-object"),
    show_interaction_edges: bool = typer.Option(
        True, "--show-interaction-edges/--hide-interaction-edges"
    ),
    show_collision_samples: bool = typer.Option(
        True, "--show-collision-samples/--hide-collision-samples"
    ),
    show_query_set: bool = typer.Option(True, "--show-query-set/--hide-query-set"),
    show_penetrations: bool = typer.Option(True, "--show-penetrations/--hide-penetrations"),
    show_slack: bool = typer.Option(True, "--show-slack/--hide-slack"),
    show_labels: bool = typer.Option(False, "--show-labels"),
    show_frames: bool = typer.Option(False, "--show-frames"),
    show_objective: bool = typer.Option(True, "--show-objective/--hide-objective"),
    show_closest: bool = typer.Option(False, "--show-closest"),
) -> None:
    try:
        sequence, warm, graph_trajectory, model, surface, _ = _refinement_components(
            canonical, warm_start, graph, robot, collision_samples, asset_root
        )
        artifact = load_final_trajectory(final)
        display_options = {
            "show_source_hand": show_source_hand,
            "show_warm_start": show_warm_start,
            "show_final": show_final,
            "show_object": show_object,
            "show_interaction_edges": show_interaction_edges,
            "show_collision_samples": show_collision_samples,
            "show_query_set": show_query_set,
            "show_penetrations": show_penetrations,
            "show_slack": show_slack,
            "show_labels": show_labels,
            "show_frames": show_frames,
            "show_objective": show_objective,
            "show_closest": show_closest,
            "view": view,
        }
        if interactive:
            value = launch_refinement_viewer(
                sequence,
                warm,
                graph_trajectory,
                artifact,
                model,
                surface,
                start_frame=start_frame,
                end_frame=end_frame,
                show=True,
                **display_options,
            )
        else:
            value = render_refinement_frame(
                sequence,
                warm,
                graph_trajectory,
                artifact,
                model,
                surface,
                frame=frame,
                output=output,
                show=False,
                show_source_hand=show_source_hand,
                show_warm_start=show_warm_start,
                show_final=show_final,
                show_object=show_object,
                show_interaction_edges=show_interaction_edges,
                show_collision_samples=show_collision_samples,
                show_query_set=show_query_set,
                show_penetrations=show_penetrations,
                show_slack=show_slack,
                show_labels=show_labels,
                show_frames=show_frames,
                show_objective=show_objective,
                show_closest=show_closest,
                view=view,
            )
        _json_write(value, report)
        if report is None:
            _json_write(value, None)
    except (
        StorageError,
        ValueError,
        OSError,
        RuntimeError,
        WarmStartArtifactError,
        InteractionArtifactError,
    ) as exc:
        typer.echo(f"visualize-refinement failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
