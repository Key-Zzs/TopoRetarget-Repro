#!/usr/bin/env python3
"""Evaluate every bounded HOCap PPO rollout without success-only filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.environments.mujoco_backend import (
    MujocoBackendConfig,
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.evaluation import EpisodeMetrics, summarize_episodes
from toporetarget.rl.ppo.checkpoint import load_checkpoint
from toporetarget.rl.ppo.trainer import PPOTrainer
from toporetarget.rl.randomization import DomainRandomizationConfig
from toporetarget.rl.termination import BASE_RELATIVE_HOCAP_TERMINATION

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"


def rotation_error_deg(actual: np.ndarray, reference: np.ndarray) -> float:
    relative = actual[:3, :3].T @ reference[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--episodes-per-clip", type=int, default=2)
    parser.add_argument("--action-scale-fraction", type=float, default=0.05)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--episodes-output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if len(args.reference) != len(args.object_mesh):
        raise ValueError("--reference and --object-mesh must have equal counts")
    if not 1 <= args.episodes_per_clip <= 16:
        raise ValueError("--episodes-per-clip must be in 1..16")
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = model.jnt_range[: model.njnt].copy()
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in joint_order):
        raise RuntimeError("Wuji MJCF has unnamed joints")
    joint_order = tuple(name for name in joint_order if name is not None)
    backends: list[MujocoReferenceTrackingBackend] = []
    inventory: list[dict[str, object]] = []
    for index, (reference_path, mesh_path) in enumerate(
        zip(args.reference, args.object_mesh, strict=True)
    ):
        reference = Stage16ReferenceClip.from_npz(reference_path)
        if reference.joint_order != joint_order:
            raise ValueError(f"reference joint order does not match Wuji MJCF: {reference_path}")
        scene = materialize_free_object_scene(
            WUJI_MJCF,
            args.scene_root / f"clip_{index:02d}",
            object_mesh=mesh_path,
            include_ground=False,
            gravity_mps2=(0.0, 0.0, 0.0),
        )
        backends.append(
            MujocoReferenceTrackingBackend(
                scene_path=scene,
                reference=reference,
                joint_lower=bounds[:, 0],
                joint_upper=bounds[:, 1],
                config=MujocoBackendConfig(
                    action_scale_fraction=args.action_scale_fraction,
                    termination_profile=BASE_RELATIVE_HOCAP_TERMINATION,
                ),
                randomization=DomainRandomizationConfig(enabled=args.domain_randomization),
                seed=args.seed + index,
            )
        )
        inventory.append(
            {
                "reference": str(reference_path.resolve()),
                "reference_hash": reference.content_hash(),
                "object_mesh": str(mesh_path.resolve()),
                "source_sequence": reference.provenance["dataset_provenance"]["source_sequence"],
            }
        )
    observation_dim = int(backends[0].observation(backends[0].reset(reference_index=0)).size)
    trainer = PPOTrainer(observation_dim, model.njnt, device=args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=trainer.device)
    trainer.model.load_state_dict(checkpoint["model"])
    trainer.normalizer.load_state_dict(checkpoint["normalizer"])
    trainer.freeze_observation_normalizer()
    episodes: list[dict[str, object]] = []
    for backend, source in zip(backends, inventory, strict=True):
        for episode_index in range(args.episodes_per_clip):
            state = backend.reset(reference_index=0)
            observation = backend.observation(state)
            total_return = 0.0
            actions: list[np.ndarray] = []
            reason: str | None = None
            for _ in range(backend.reference.frame_count + 4):
                observation_tensor = torch.as_tensor(
                    observation[None], dtype=torch.float32, device=trainer.device
                )
                with torch.no_grad():
                    action, _, _ = trainer.act(observation_tensor, deterministic=True)
                action_value = torch.clamp(action[0], -1.0, 1.0).cpu().numpy()
                state, reward, reason = backend.transition(action_value)
                actions.append(action_value)
                total_return += float(reward["total"])
                if reason is not None:
                    break
                observation = backend.observation(state)
            if reason is None:
                reason = "FAILURE_EVALUATION_STEP_BOUND"
            reference_index = backend.reference_index
            reference = backend.reference
            object_reference = reference.object_pose_base_ref[reference_index]
            axis_error = np.linalg.norm(
                state["object_axis_points"]
                - reference.object_axis_points_base_ref[reference_index],
                axis=1,
            )
            link_error = state["links"] - reference.tracked_link_positions_base_ref[reference_index]
            action_array = np.asarray(actions)
            first = np.diff(action_array, axis=0)
            second = np.diff(action_array, n=2, axis=0)
            metric = EpisodeMetrics(
                termination=reason,
                success=reason == "SUCCESS_REFERENCE_COMPLETE",
                final_frame_reached=reference_index >= reference.frame_count - 1,
                object_position_error_m=float(
                    np.linalg.norm(state["object_pose"][:3, 3] - object_reference[:3, 3])
                ),
                object_rotation_error_deg=rotation_error_deg(
                    state["object_pose"], object_reference
                ),
                max_axis_point_error_m=float(axis_error.max()),
                link_rmse_m=float(np.sqrt(np.mean(np.square(link_error)))),
                normalized_joint_error=float(
                    np.mean(
                        np.abs(state["q"] - reference.q_finger_ref[reference_index])
                        / (bounds[:, 1] - bounds[:, 0])
                    )
                ),
                progress_ratio=float(reference_index / (reference.frame_count - 1)),
                return_value=total_return,
                action_magnitude=float(np.mean(np.linalg.norm(action_array, axis=1))),
                action_first_difference=float(np.mean(np.linalg.norm(first, axis=1)))
                if first.size
                else 0.0,
                action_second_difference=float(np.mean(np.linalg.norm(second, axis=1)))
                if second.size
                else 0.0,
            )
            episodes.append(
                {
                    **metric.__dict__,
                    "source_sequence": source["source_sequence"],
                    "episode_index": episode_index,
                    "reference_hash": source["reference_hash"],
                }
            )
    summary = summarize_episodes(
        [
            EpisodeMetrics(
                **{
                    key: value
                    for key, value in row.items()
                    if key in EpisodeMetrics.__dataclass_fields__
                }
            )
            for row in episodes
        ]
    )
    args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
    args.episodes_output.write_text(
        json.dumps(episodes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "status": "HOCAP_REFERENCE_POLICY_EVALUATION_COMPLETE",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_stage": checkpoint["stage"],
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "references": inventory,
        "episodes_per_clip": args.episodes_per_clip,
        "domain_randomization": args.domain_randomization,
        "summary": summary,
        "episodes_path": str(args.episodes_output.resolve()),
        "physics_profile": {
            "backend": "mujoco_cpu_reference",
            "per_object_collision_mesh": True,
            "synthetic_ground_enabled": False,
            "gravity_mps2": [0.0, 0.0, 0.0],
            "height_termination_disabled": True,
        },
        "paper_comparable": False,
        "non_claim": (
            "all bounded functional episodes are reported; this is not HOCap-32 paper evaluation"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
