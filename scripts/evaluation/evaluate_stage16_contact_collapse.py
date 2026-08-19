#!/usr/bin/env python3
"""Evaluate Stage16 contact skill at every frozen C0 PPO update."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.evaluation.audit_stage16_zero_g_frozen_actor_contact import (
    _full_start,
    _load_pairs,
    _run_episode,
    _selection,
    _trace_metrics,
)
from scripts.rl.isaaclab.evaluate_stage16d_ppo26d import model_from_checkpoint
from toporetarget.rl.contact_skill_collapse import (
    FINGER_NAMES,
    command_tracking_metrics,
    detect_contact_milestones,
    lift_timing,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)

DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16_contact_skill_collapse"
OBJECT_MESH = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_objects"
TIP_LINK_INDICES = (3, 6, 9, 12, 15)
JOINT_NAMES = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("CONTACT_COLLAPSE_EVALUATION_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    value = value / np.linalg.norm(value, axis=-1, keepdims=True).clip(min=1.0e-12)
    w, x, y, z = np.moveaxis(value, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*value.shape[:-1], 3, 3)


def _tip_object_distance(trace: dict[str, np.ndarray], *, clip: str) -> np.ndarray:
    """Deterministic visual-mesh vertex distance proxy for five distal roots."""

    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load_mesh(OBJECT_MESH / f"{clip}.obj", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise ValueError("CONTACT_COLLAPSE_OBJECT_VISUAL_MESH_INVALID")
    tips_world = np.asarray(trace["tracked_link_positions"], dtype=np.float64)[:, TIP_LINK_INDICES]
    object_pose = np.asarray(trace["object_pose"], dtype=np.float64)
    rotation = _quaternion_matrix_wxyz(object_pose[:, 3:])
    relative = tips_world - object_pose[:, None, :3]
    tips_local = np.einsum("tji,tfj->tfi", rotation, relative)
    distance = cKDTree(np.asarray(mesh.vertices, dtype=np.float64)).query(
        tips_local.reshape(-1, 3), workers=1
    )[0]
    return np.asarray(distance, dtype=np.float64).reshape(tips_local.shape[:2])


def _mean(values: list[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def _snapshot_specs(
    snapshot_root: Path | None, *, source_only: bool, stage: str
) -> list[dict[str, object]]:
    selection = _selection("aggregate_v3", "hocap_170105")
    specs: list[dict[str, object]] = [
        {
            "label": "SOURCE",
            "update": 0,
            "samples": 0,
            "checkpoint": selection["checkpoint"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
        }
    ]
    if source_only:
        return specs
    if snapshot_root is None:
        raise ValueError("CONTACT_COLLAPSE_SNAPSHOT_ROOT_REQUIRED")
    paths = sorted(snapshot_root.resolve().glob("update_*.pt"))
    if not paths:
        raise FileNotFoundError("CONTACT_COLLAPSE_UPDATE_SNAPSHOTS_MISSING")
    import torch

    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw_update = payload.get("contact_preservation_update_index")
        raw_samples = payload.get("contact_preservation_stage_samples")
        update = int(payload["contact_collapse_update_index"] if raw_update is None else raw_update)
        samples = int(
            payload["contact_collapse_stage_samples"] if raw_samples is None else raw_samples
        )
        specs.append(
            {
                "label": f"{stage}_U{update}",
                "update": update,
                "samples": samples,
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
            }
        )
    return specs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stage", choices=("C0", "C1"), default="C0")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--reset-index",
        type=int,
        help="Frozen deterministic reference index; defaults to the full frame-zero start.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Evaluate exactly this one frozen actor instead of a snapshot directory.",
    )
    parser.add_argument("--label", help="Required label when --checkpoint is used.")
    parser.add_argument("--update", type=int, help="Required update authority with --checkpoint.")
    parser.add_argument(
        "--samples", type=int, help="Required stage sample count with --checkpoint."
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.episodes != 10:
        raise ValueError("CONTACT_COLLAPSE_EVALUATION_FROZEN_AT_10_EPISODES")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is not None:
        if args.source_only or args.snapshot_root is not None:
            raise ValueError("CONTACT_COLLAPSE_SINGLE_CHECKPOINT_INPUT_CONFLICT")
        if args.label is None or args.update is None or args.samples is None:
            raise ValueError("CONTACT_COLLAPSE_SINGLE_CHECKPOINT_AUTHORITY_REQUIRED")
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("CONTACT_COLLAPSE_SINGLE_CHECKPOINT_MISSING")
        specs = [
            {
                "label": args.label,
                "update": args.update,
                "samples": args.samples,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
            }
        ]
    else:
        specs = _snapshot_specs(args.snapshot_root, source_only=args.source_only, stage=args.stage)
    pairs = _load_pairs("hocap_170105", args.episodes)
    seeds = [int(pair["seed"]) for pair in pairs]
    frame0_start = _full_start("hocap_170105")
    start_index = frame0_start if args.reset_index is None else args.reset_index
    if start_index < 0 or start_index >= 321:
        raise ValueError("CONTACT_COLLAPSE_RESET_INDEX_OUT_OF_RANGE")
    _write_json(
        output / "evaluation_contract.json",
        {
            "schema_version": "Stage16ContactCollapseEvaluationContractV1",
            "clip": "hocap_170105",
            "reward": "aggregate_v3",
            "episodes_per_snapshot": args.episodes,
            "seeds": seeds,
            "evaluation_reset": (
                "frame0_full_start" if args.reset_index is None else "fixed_reference_index"
            ),
            "start_index": start_index,
            "frame0_start_index": frame0_start,
            "optimizer_steps": 0,
            "deterministic_actor": True,
            "persistent_contact_frames": 3,
            "actual_wrist_up_rule": "world_z_velocity_gt_0.02_mps_for_3_frames",
            "object_lift_onset_rule": "z_displacement_gt_0.005_m_for_3_frames",
            "lift_success_threshold_m": 0.05,
            "tip_object_distance": "distal_root_to_visual_mesh_vertex_unsigned_proxy",
        },
    )
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
        from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

        env = _make_table_env(
            clip="hocap_170105",
            num_envs=1,
            start_index=start_index,
            mode=ContactRewardMode.AGGREGATE_V3,
            stage=args.stage,
        )
        snapshot_rows: list[dict[str, object]] = []
        per_finger_rows: list[dict[str, object]] = []
        per_joint_rows: list[dict[str, object]] = []
        action_rows: list[dict[str, object]] = []
        timing_rows: list[dict[str, object]] = []
        geometry_rows: list[dict[str, object]] = []
        for spec in specs:
            checkpoint = Path(str(spec["checkpoint"])).resolve()
            trainer, _ = model_from_checkpoint(
                checkpoint, str(env.device), expected_clip="hocap_170105"
            )
            episode_metrics: list[dict[str, object]] = []
            command_metrics: list[dict[str, Any]] = []
            for episode, seed in enumerate(seeds):
                rollout, trace = _run_episode(
                    env=env,
                    trainer=trainer,
                    clip="hocap_170105",
                    seed=seed,
                    reset_index=start_index,
                )
                distance = _tip_object_distance(trace, clip="hocap_170105")
                trace["tip_object_distance_m"] = distance.astype(np.float32)
                trace["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
                trace["hand_collision_body_pose"] = reconstruct_hand_collision_body_pose(
                    trace["wrist_pose"], trace["finger_q"], repo_root=REPO_ROOT
                ).astype(np.float32)
                metric = _trace_metrics(trace, mode="aggregate_v3")
                command = command_tracking_metrics(trace)
                timing = lift_timing(trace)
                phase = np.asarray(trace["phase"])
                focus = np.isin(phase, ("CONTACT", "GRASP"))
                trace_path = (
                    output / "contact_eval" / str(spec["label"]) / f"episode_{episode:02d}.npz"
                )
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trace_path, **trace)
                record = {
                    **spec,
                    "episode": episode,
                    "seed": seed,
                    "trace": str(trace_path),
                    **rollout,
                    **metric,
                    **timing,
                    "tip_object_distance_min_m": float(distance.min()),
                    "tip_object_distance_grasp_mean_m": (
                        None if not np.any(focus) else float(distance[focus].mean())
                    ),
                }
                _write_json(trace_path.with_suffix(".json"), record)
                episode_metrics.append(record)
                command_metrics.append(command)
                timing_rows.append(
                    {
                        "snapshot": spec["label"],
                        "update": spec["update"],
                        "episode": episode,
                        **timing,
                    }
                )
                for finger, name in enumerate(FINGER_NAMES):
                    geometry_rows.append(
                        {
                            "snapshot": spec["label"],
                            "update": spec["update"],
                            "episode": episode,
                            "finger": name,
                            "minimum_distance_m": float(distance[:, finger].min()),
                            "contact_grasp_mean_distance_m": float(distance[focus, finger].mean()),
                        }
                    )
                action = np.asarray(trace["action"], dtype=np.float64)[focus]
                for dimension in range(26):
                    action_rows.append(
                        {
                            "snapshot": spec["label"],
                            "update": spec["update"],
                            "dimension": dimension,
                            "semantic": (
                                ("wrist_translation_xyz" if dimension < 3 else "wrist_rotation_xyz")
                                if dimension < 6
                                else JOINT_NAMES[dimension - 6]
                            ),
                            "signed_mean": float(action[:, dimension].mean()),
                            "mean_abs": float(np.abs(action[:, dimension]).mean()),
                            "p95_abs": float(np.quantile(np.abs(action[:, dimension]), 0.95)),
                        }
                    )
            contacts = [bool(row["any_hand_object_contact"]) for row in episode_metrics]
            snapshot_rows.append(
                {
                    **spec,
                    "episodes": args.episodes,
                    "contact_episodes": int(sum(contacts)),
                    "contact_episode_rate": float(np.mean(contacts)),
                    "any_hand_object_contact_fraction": _mean(
                        [row["hand_object_contact_fraction"] for row in episode_metrics]
                    ),
                    "tip_contact_fraction": _mean(
                        [row["tip_contact_fraction"] for row in episode_metrics]
                    ),
                    "source_tip_recall": _mean(
                        [row["source_tip_recall"] for row in episode_metrics]
                    ),
                    "persistent_tip_recall": _mean(
                        [row["persistent_tip_recall"] for row in episode_metrics]
                    ),
                    "first_contact_step": _mean(
                        [row["first_contact_step"] for row in episode_metrics]
                    ),
                    "max_contact_force_n": max(
                        float(row["max_contact_force_n"]) for row in episode_metrics
                    ),
                    "contact_reward_activation_fraction": float(
                        np.mean(
                            [
                                bool(row["contact_reward_activates_when_actual_contact"])
                                for row in episode_metrics
                            ]
                        )
                    ),
                    "object_lift_dz_m": _mean([row["object_lift_dz_m"] for row in episode_metrics]),
                    "lift_success_rate": float(
                        np.mean(
                            [
                                bool(row["any_hand_object_contact"])
                                and float(row["object_lift_dz_m"]) >= 0.05
                                for row in episode_metrics
                            ]
                        )
                    ),
                    "drop_rate": float(
                        np.mean([bool(row["object_drop"]) for row in episode_metrics])
                    ),
                    "premature_lift_rate": float(
                        np.mean([bool(row["premature_lift"]) for row in episode_metrics])
                    ),
                    "wrist_ref_command_mean_m": _mean(
                        [
                            item["wrist_position_ref_to_command_m"]["mean"]
                            for item in command_metrics
                        ]
                    ),
                    "wrist_command_actual_mean_m": _mean(
                        [
                            item["wrist_position_command_to_actual_m"]["mean"]
                            for item in command_metrics
                        ]
                    ),
                    "wrist_ref_command_rotation_mean_rad": _mean(
                        [
                            item["wrist_rotation_ref_to_command_rad"]["mean"]
                            for item in command_metrics
                        ]
                    ),
                    "wrist_command_actual_rotation_mean_rad": _mean(
                        [
                            item["wrist_rotation_command_to_actual_rad"]["mean"]
                            for item in command_metrics
                        ]
                    ),
                    "finger_ref_command_mean_rad": _mean(
                        [item["finger_ref_to_command_rad"]["mean"] for item in command_metrics]
                    ),
                    "finger_command_actual_mean_rad": _mean(
                        [item["finger_command_to_actual_rad"]["mean"] for item in command_metrics]
                    ),
                    "minimum_tip_object_distance_m": min(
                        float(row["tip_object_distance_min_m"]) for row in episode_metrics
                    ),
                }
            )
            for finger, name in enumerate(FINGER_NAMES):
                per_finger_rows.append(
                    {
                        "snapshot": spec["label"],
                        "update": spec["update"],
                        "finger": name,
                        "mean_abs_rad": _mean(
                            [item["per_finger"][finger]["mean_abs_rad"] for item in command_metrics]
                        ),
                        "p95_abs_rad": _mean(
                            [item["per_finger"][finger]["p95_abs_rad"] for item in command_metrics]
                        ),
                        "signed_mean_rad": _mean(
                            [
                                item["per_finger"][finger]["signed_mean_rad"]
                                for item in command_metrics
                            ]
                        ),
                    }
                )
            for joint, name in enumerate(JOINT_NAMES):
                per_joint_rows.append(
                    {
                        "snapshot": spec["label"],
                        "update": spec["update"],
                        "joint": name,
                        "mean_abs_rad": _mean(
                            [item["per_joint"][joint]["mean_abs_rad"] for item in command_metrics]
                        ),
                        "p95_abs_rad": _mean(
                            [item["per_joint"][joint]["p95_abs_rad"] for item in command_metrics]
                        ),
                        "signed_mean_rad": _mean(
                            [
                                item["per_joint"][joint]["signed_mean_rad"]
                                for item in command_metrics
                            ]
                        ),
                    }
                )
            _write_json(
                output / "progress.json", {"completed": [row["label"] for row in snapshot_rows]}
            )
        source = snapshot_rows[0]
        if source["label"] == "SOURCE" and int(source["contact_episodes"]) != args.episodes:
            raise RuntimeError("SOURCE_POLICY_CONTACT_REGRESSION")
        update_rows = [row for row in snapshot_rows if int(row["update"]) > 0]
        milestones = (
            {}
            if args.source_only
            else detect_contact_milestones(
                update_rows, baseline_contact_episodes=int(source["contact_episodes"])
            )
        )
        _write_csv(output / "contact_vs_update.csv", snapshot_rows)
        _write_csv(
            output / "contact_vs_samples.csv",
            sorted(snapshot_rows, key=lambda row: int(row["samples"])),
        )
        _write_csv(output / "command_drift" / "per_finger.csv", per_finger_rows)
        _write_csv(output / "command_drift" / "finger.csv", per_joint_rows)
        _write_csv(output / "command_drift" / "top_action_dims_source.csv", action_rows)
        _write_csv(output / "lift_timing" / "timing.csv", timing_rows)
        _write_csv(output / "command_drift" / "tip_object_geometry.csv", geometry_rows)
        _write_json(
            output / "evaluation_summary.json",
            {"snapshots": snapshot_rows, "milestones": milestones},
        )
        print(
            json.dumps(
                {"status": "PASS", "snapshots": len(snapshot_rows), "milestones": milestones}
            )
        )
        return 0
    except BaseException as error:
        _write_json(
            output / "technical_failure.json",
            {
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
