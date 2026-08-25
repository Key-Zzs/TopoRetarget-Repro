"""Stage 7 sequence boundary: canonical MediaPipe-21 to warm-start artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.schema import HOISequence
from toporetarget.utils.hashing import sha256_tree

from .alignment import alignment_errors, apply_base_pose_to_points, base_seed_from_hand_frames
from .artifacts import WARM_START_SCHEMA_VERSION, WarmStartTrajectory
from .solver import SequenceSolveResult, WarmStartSolverProfile, load_paper_weights, solve_sequence


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()


def source_cache_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists():
        return None
    digest = hashlib.sha256()
    for name, value in sha256_tree(root).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _stack_feature_field(features: list[Any], name: str) -> np.ndarray:
    values = []
    for feature in features:
        value = getattr(feature, name)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        values.append(np.asarray(value))
    return np.stack(values)


def build_warm_start_trajectory(
    sequence: HOISequence,
    hand_id: str,
    robot_model: Any,
    frame_profile: Any,
    bone_profile: Any,
    solver_profile: WarmStartSolverProfile,
    *,
    source_cache: str | Path | None = None,
    lambda_warm: float | None = None,
    lambda_smooth: float | None = None,
) -> tuple[WarmStartTrajectory, dict[str, Any]]:
    hand = sequence.hand(hand_id)
    track = hand.keypoint_tracks.get(bone_profile.layout_name)
    if track is None:
        raise ValueError(f"hand {hand_id!r} has no {bone_profile.layout_name!r} track")
    if track.positions_scene.shape[1:] != (21, 3):
        raise ValueError(f"MediaPipe-21 track has invalid shape: {track.positions_scene.shape}")
    quality = hand.metadata.get("retarget_input_quality")
    mano_primary = (
        isinstance(quality, dict)
        and quality.get("wrist_orientation_authority") == "MANO_GLOBAL_WRIST_ORIENTATION"
    )
    source_frame_transforms: np.ndarray | None = None
    if mano_primary:
        wrist = hand.wrist_pose_scene
        if not wrist.orientation_available:
            raise ValueError("RETARGET_MANO_PRIMARY_WRIST_ORIENTATION_REQUIRED")
        source_frame_transforms = np.asarray(wrist.pose_scene, dtype=np.float64)
        if (
            source_frame_transforms.shape != (sequence.num_frames, 4, 4)
            or not np.isfinite(source_frame_transforms).all()
        ):
            raise ValueError("RETARGET_MANO_PRIMARY_WRIST_FRAME_INVALID")
    root = Path(__file__).resolve().parents[3]
    paper_warm, paper_smooth, paper_path = load_paper_weights(root)
    warm = paper_warm if lambda_warm is None else float(lambda_warm)
    smooth = paper_smooth if lambda_smooth is None else float(lambda_smooth)
    solved: SequenceSolveResult = solve_sequence(
        np.asarray(track.positions_scene, dtype=np.float64),
        robot_model,
        frame_profile,
        bone_profile,
        solver_profile,
        side=hand.side,
        lambda_warm=warm,
        lambda_smooth=smooth,
        source_frame_transforms=source_frame_transforms,
    )
    source_frames = np.asarray(solved.source_features.frame_transform, dtype=np.float64)
    robot_frames = np.stack(
        [
            np.asarray(
                feature.frame_transform.detach().cpu()
                if hasattr(feature.frame_transform, "detach")
                else feature.frame_transform,
                dtype=np.float64,
            )
            for feature in solved.robot_features
        ]
    )
    base_pose = base_seed_from_hand_frames(source_frames, robot_frames)
    robot_base = np.stack(
        [
            np.asarray(robot_model.keypoints_base(q).detach().cpu(), dtype=np.float64)
            for q in solved.qpos
        ]
    )
    robot_scene = apply_base_pose_to_points(robot_base, base_pose)
    align = alignment_errors(source_frames, robot_frames, base_pose)
    lower = np.asarray(robot_model.joint_lower)
    upper = np.asarray(robot_model.joint_upper)
    margins = np.minimum(solved.qpos - lower[None, :], upper[None, :] - solved.qpos)
    metadata = {
        "schema_version": WARM_START_SCHEMA_VERSION,
        "source_sequence_id": sequence.metadata.sequence_id,
        "source_cache_hash": source_cache_hash(source_cache),
        "source_hand_id": hand_id,
        "source_side": hand.side,
        "frame_range": [0, int(sequence.num_frames)],
        "timestamps": np.asarray(sequence.metadata.timestamps).tolist(),
        "native_fps": sequence.metadata.native_fps,
        "robot_name": robot_model.name,
        "robot_side": robot_model.side,
        "robot_dof_count": robot_model.num_dofs,
        "robot_spec_hash": robot_model.spec_hash,
        "urdf_hash": robot_model.urdf_hash,
        "asset_manifest_hash": robot_model.asset_manifest_hash,
        "anchor_profile_id": robot_model.anchor_profile.profile_id,
        "anchor_profile_hash": robot_model.anchor_profile.sha256,
        "frame_profile_id": frame_profile.profile_id,
        "frame_profile_hash": frame_profile.sha256,
        "source_wrist_orientation_authority": (
            "MANO_GLOBAL_WRIST_ORIENTATION"
            if mano_primary
            else "CANONICAL_KEYPOINT_WRIST_LEGACY_OR_NON_HOCAP"
        ),
        "source_keypoint_wrist_frame_production_authority": not mano_primary,
        "bone_profile_id": bone_profile.profile_id,
        "bone_profile_hash": bone_profile.sha256,
        "solver_profile_id": solver_profile.profile_id,
        "solver_profile_hash": solver_profile.sha256,
        "paper_config": str(paper_path),
        "paper_config_hash": hashlib.sha256(Path(paper_path).read_bytes()).hexdigest(),
        "paper_weights": {"lambda_warm": warm, "lambda_smooth": smooth},
        "assumptions": sorted(
            set(
                frame_profile.assumptions
                + bone_profile.assumptions
                + solver_profile.assumptions
                + (
                    "A_WARMSTART_BASE_OBSERVABILITY_001",
                    "A_BASE_SEED_ALIGNMENT_001",
                    "A_WARMSTART_COORDINATES_001",
                    "A_WARMSTART_JOINT_LIMITS_001",
                    "A_WARMSTART_TIME_DISCRETIZATION_001",
                )
            )
        ),
        "provenance": {
            "no_temporal_resampling": True,
            "temporal_weight_dt_normalized": False,
            "source_layout": bone_profile.layout_name,
            "source_frame_name": track.frame_name,
            "base_seed_strategy": (
                "source canonical hand frame times inverse robot canonical hand frame"
            ),
            "object_data_accessed": False,
            "stage6_object_samples_accessed": False,
            "sdf_accessed": False,
            "final_retargeting": False,
            "stage8_started": False,
        },
        "alignment": align,
        "solver": solver_profile.as_dict(),
    }
    arrays = {
        "timestamps": np.asarray(sequence.metadata.timestamps, dtype=np.float64),
        "qpos": solved.qpos,
        "initial_qpos": solved.initial_qpos,
        "base_pose_scene": base_pose,
        "robot_keypoints_base": robot_base,
        "robot_keypoints_scene": robot_scene,
        "source_hand_frame_scene": source_frames,
        "robot_hand_frame_base": robot_frames,
        "source_bone_vectors": np.asarray(solved.source_features.bone_vectors),
        "source_bone_lengths": np.asarray(solved.source_features.bone_lengths),
        "source_bone_directions": np.asarray(solved.source_features.unit_directions),
        "robot_bone_directions": _stack_feature_field(solved.robot_features, "unit_directions"),
        "source_adjacent_features": np.asarray(solved.source_features.adjacent_features),
        "robot_adjacent_features": _stack_feature_field(solved.robot_features, "adjacent_features"),
        "pair_residuals": solved.pair_residuals,
        "ebone": solved.final_ebone,
        "initial_ebone": solved.initial_ebone,
        "temporal_term": solved.temporal_term,
        "total_objective": solved.total_objective,
        "initial_total_objective": solved.initial_total_objective,
        "solver_status": solved.solver_status,
        "solver_success": solved.solver_success,
        "valid_mask": solved.solver_success,
        "nfev": solved.nfev,
        "njev": solved.njev,
        "solve_time_s": solved.solve_time_s,
        "joint_limit_margins": margins,
    }
    trajectory = WarmStartTrajectory(
        metadata, {key: np.asarray(value) for key, value in arrays.items()}
    )
    trajectory.validate()
    diagnostics = {
        "alignment": align,
        "frame_count": int(sequence.num_frames),
        "solver_success_count": int(np.sum(solved.solver_success)),
        "solver_failure_count": int(np.sum(~solved.solver_success)),
        "initial_ebone_mean": float(np.mean(solved.initial_ebone)),
        "final_ebone_mean": float(np.mean(solved.final_ebone)),
        "initial_total_objective": float(np.sum(solved.initial_total_objective)),
        "final_total_objective": float(np.sum(solved.total_objective)),
        "joint_limit_min_margin": float(np.min(margins)),
    }
    return trajectory, diagnostics


__all__ = ["build_warm_start_trajectory", "source_cache_hash"]
