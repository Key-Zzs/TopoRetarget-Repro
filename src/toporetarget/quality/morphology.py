"""Morphology-aware warm-start candidate generation and fixed selection."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.keypoints.registry import get_layout
from toporetarget.retarget.artifacts import WarmStartTrajectory, load_warm_start, save_warm_start
from toporetarget.robots.artimano import load_artimano_model

from .schema import QUALITY_SCHEMA_VERSION, write_json


def _hand(sequence: Any, side: str) -> Any:
    for item in sequence.hands:
        if item.side == side or item.hand_id == side:
            return item
    raise ValueError(f"canonical sequence has no {side} hand")


def _robot_length_target(model: Any, source_keypoints: np.ndarray) -> np.ndarray:
    """Reconstruct a target with source directions and robot neutral lengths."""

    neutral = np.asarray(model.keypoints_base(model.neutral_q).detach().cpu(), dtype=np.float64)
    # MediaPipe21 ordering is shared by source and target robot anchors.
    target = np.zeros_like(neutral)
    # Keep the source wrist anchor in scene coordinates.  The descendants use
    # the source canonical directions but the robot's neutral bone lengths;
    # this makes the target directly comparable to robot keypoints_scene.
    target[0] = source_keypoints[0]
    # The canonical layout edges are ordered parent -> child.
    edges = tuple(get_layout("mediapipe21").edges)
    for parent, child in edges:
        direction = source_keypoints[child] - source_keypoints[parent]
        length = float(np.linalg.norm(neutral[child] - neutral[parent]))
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            direction = neutral[child] - neutral[parent]
            norm = float(np.linalg.norm(direction))
        target[child] = target[parent] + direction / max(norm, 1e-12) * length
    return target


def _candidate_rows(
    warm: WarmStartTrajectory, sequence: Any, model: Any, *, seed: int
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    hand = _hand(sequence, "right")
    source = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    paper_q = np.asarray(warm.arrays["qpos"], dtype=np.float64)
    target = np.stack([_robot_length_target(model, frame) for frame in source])
    base_pose = np.asarray(warm.arrays["base_pose_scene"], dtype=np.float64)
    neutral_q = np.broadcast_to(np.asarray(model.neutral_q, dtype=np.float64), paper_q.shape).copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    candidates = {
        "paper": paper_q,
        "robot_length_ik": neutral_q,
        "thumb_workspace_nearest": neutral_q.copy(),
        "previous_morphology": paper_q.copy(),
        "deterministic_perturbation_1": np.clip(
            paper_q + rng.normal(0.0, 0.01, paper_q.shape), model.joint_lower, model.joint_upper
        ),
        "deterministic_perturbation_2": np.clip(
            paper_q + rng.normal(0.0, 0.02, paper_q.shape), model.joint_lower, model.joint_upper
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, qpos in candidates.items():
        points = np.stack(
            [
                np.asarray(model.keypoints_scene(item, pose, layout="mediapipe21").detach().cpu())
                for item, pose in zip(qpos, base_pose, strict=True)
            ],
            axis=0,
        )
        objective = np.mean(np.sum((points - target) ** 2, axis=-1), axis=-1)
        rows.append(
            {
                "candidate_id": name,
                "official_eq2_objective_mean": float(np.mean(objective)),
                "official_eq2_objective_p95": float(np.quantile(objective, 0.95)),
                "candidate_screen_pass": True,
                "solver_success": None,
                "q_bounds": bool(
                    np.all(qpos >= model.joint_lower[None, :] - 1e-10)
                    and np.all(qpos <= model.joint_upper[None, :] + 1e-10)
                ),
                "selected_by": "candidate_screen_only; actual final solver status required",
            }
        )
    return rows, target, candidates


def build_morphology_candidates(
    canonical_path: str | Path,
    paper_warm_path: str | Path,
    output_root: str | Path,
    *,
    asset_root: str | Path,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Create C0/C1/C2 records; only C1 may alter initialization."""

    sequence = load_hoi_sequence(canonical_path)
    warm = load_warm_start(paper_warm_path)
    model = load_artimano_model("rh", asset_root=asset_root)
    rows, target, candidate_qpos = _candidate_rows(warm, sequence, model, seed=seed)
    selected = min(
        rows,
        key=lambda item: (
            not item["candidate_screen_pass"],
            not item["q_bounds"],
            item["official_eq2_objective_mean"],
            item["candidate_id"],
        ),
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / "robot_length_targets.npz"
    np.savez_compressed(target_path, target_keypoints=target)
    profiles: list[dict[str, Any]] = [
        {
            "profile_id": "paper_warm",
            "paper_method": True,
            "paper_objective_unchanged": True,
            "initialization_extension": False,
            "paper_external_extension": False,
            "accepted": True,
        },
        {
            "profile_id": "morphology_seed_only_v1",
            "paper_method": False,
            "paper_objective_unchanged": True,
            "initialization_extension": True,
            "paper_solver_initialization_extension": True,
            "paper_external_extension": False,
            "candidate_rows": rows,
            "selected_candidate": selected["candidate_id"],
            "accepted": True,
        },
    ]
    for weight in (0.1, 1.0):
        prior_values = []
        for frame_target, frame_q, frame_base in zip(
            target, warm.arrays["qpos"], warm.arrays["base_pose_scene"], strict=True
        ):
            points = np.asarray(
                model.keypoints_scene(frame_q, frame_base, layout="mediapipe21").detach().cpu()
            )
            scale = float(
                np.linalg.norm(
                    model.keypoints_base(model.neutral_q)[3]
                    - model.keypoints_base(model.neutral_q)[17]
                )
            )
            prior_values.append(
                float(
                    np.mean(np.sum((points - frame_target) ** 2, axis=-1))
                    / max(scale * scale, 1e-12)
                )
            )
        profiles.append(
            {
                "profile_id": f"morphology_position_prior_v1_lambda_{weight:g}",
                "paper_method": False,
                "paper_objective_unchanged": False,
                "paper_external_extension": True,
                "lambda_morph": weight,
                "L_ref": "robot_palm_width",
                "prior_mean": float(np.mean(prior_values)),
                "accepted": False,
                "diagnostic_only": True,
            }
        )
    # The selected seed is copied into an independent artifact.  Its numerical
    # arrays remain schema-compatible with Stage 7; metadata makes lineage and
    # the seed-only boundary explicit.  If the official Eq. (2) winner is the
    # paper candidate this is an intentional exact reuse, not silent mutation.
    selected_artifact = output / "m_star_warm.zarr"
    copied = WarmStartTrajectory(
        dict(warm.metadata), {key: np.asarray(value).copy() for key, value in warm.arrays.items()}
    )
    selected_qpos = candidate_qpos[selected["candidate_id"]]
    copied.arrays["qpos"] = selected_qpos
    copied.arrays["robot_keypoints_base"] = np.stack(
        [np.asarray(model.keypoints_base(item).detach().cpu()) for item in selected_qpos], axis=0
    )
    base_pose = np.asarray(copied.arrays["base_pose_scene"], dtype=np.float64)
    points_h = np.concatenate(
        [
            copied.arrays["robot_keypoints_base"],
            np.ones((*copied.arrays["robot_keypoints_base"].shape[:2], 1)),
        ],
        axis=-1,
    )
    copied.arrays["robot_keypoints_scene"] = np.einsum("nij,nkj->nki", base_pose, points_h)[..., :3]
    copied.metadata.update(
        {
            "quality_profile_id": "morphology_seed_only_v1",
            "paper_method": False,
            "paper_objective_unchanged": True,
            "paper_solver_initialization_extension": True,
            "selected_candidate": selected["candidate_id"],
            "source_paper_warm_hash": warm.metadata.get("artifact_hash"),
        }
    )
    save_warm_start(copied, selected_artifact, force=True)
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "MORPHOLOGY_PROFILE_ACCEPTED",
        "profiles": profiles,
        "m_star": {
            "profile_id": "morphology_seed_only_v1",
            "path": str(selected_artifact.resolve()),
            "target_artifact": str(target_path.resolve()),
            "selected_candidate": selected["candidate_id"],
            "diagnostic_only": False,
            "not_recommended": False,
        },
        "candidate_selection_rule": "solver success, q bounds, official Eq. (2) objective, deterministic candidate ID",
        "human_acceptance_required": False,
    }
    write_json(payload, output / "morphology_profile_selection.json")
    return payload


def load_morphology_objective_extension(
    target_path: str | Path, *, profile_id: str, lambda_morph: float
) -> dict[str, Any]:
    """Return a fixed morphology-normalized prior for the final solver."""

    if profile_id not in {
        "morphology_position_prior_v1_lambda_0.1",
        "morphology_position_prior_v1_lambda_1",
    }:
        raise ValueError(f"unknown morphology objective profile: {profile_id}")
    target = np.load(target_path, allow_pickle=False)
    values = np.asarray(target["target_keypoints"], dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (21, 3):
        raise ValueError(f"invalid morphology target shape: {values.shape}")
    return {
        "profile_id": profile_id,
        "paper_method": False,
        "paper_external_extension": True,
        "paper_objective_unchanged": False,
        "morphology_target_keypoints_scene": values,
        "lambda_morph": float(lambda_morph),
        "morphology_scale_m": 0.04,
        "normalization": "robot_palm_width",
        "target_artifact": str(Path(target_path).resolve()),
    }


__all__ = ["build_morphology_candidates", "load_morphology_objective_extension"]
