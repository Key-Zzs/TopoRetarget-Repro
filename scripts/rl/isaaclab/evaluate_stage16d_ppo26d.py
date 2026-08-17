#!/usr/bin/env python3
"""Reload, evaluate, and export a Stage 16-D.5 PPO-26D L0 checkpoint."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation.full_hand_contact import hand_body_manifest
from toporetarget.rl.full_trajectory_p3 import FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES as FK_HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.physical_p3 import PHYSICAL_PPO_CHECKPOINT_SCHEMA
from toporetarget.rl.ppo.checkpoint import load_checkpoint
from toporetarget.rl.ppo.ppo26d_continuation import summarize_episodes
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode
from toporetarget.rl.reference_tracking.reference_gated_contact import (
    EVALUATION_FINGERTIP_LINKS,
    fingertip_force_indices,
)

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"
PHASE3_CHECKPOINT_SCHEMA = "Stage16DPhase3RewardV2CheckpointV1"
REWARD_V3_CHECKPOINT_SCHEMA = "Stage16DRewardV3CheckpointV1"
STRICT_V4_CHECKPOINT_SCHEMA = "Stage16DStrictPerFingerV4CheckpointV1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), default="hocap_170650")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--full-trajectory-table",
        action="store_true",
        help="Evaluate a C4 full-trajectory checkpoint with inferred table support active.",
    )
    parser.add_argument("--curriculum-stage", choices=("C4",), default="C4")
    parser.add_argument(
        "--reference-kinematics-v2-root",
        type=Path,
        help="Use the explicitly materialized V2 references for diagnostic or Phase 3 evaluation.",
    )
    parser.add_argument(
        "--object-twist-reward-v2",
        action="store_true",
        help="Evaluate a Phase 3 Reward V2 checkpoint; requires --reference-kinematics-v2-root.",
    )
    parser.add_argument(
        "--contact-mode",
        choices=tuple(mode.value for mode in ContactRewardMode),
        help=(
            "Unified contact objective. The historical V3/V4 flags remain compatibility aliases; "
            "checkpoint provenance still determines the required evaluation contract."
        ),
    )
    parser.add_argument(
        "--reference-gated-contact-reward-v3",
        action="store_true",
        help=(
            "Evaluate a frozen Reward V3 checkpoint; requires V2 references, a frozen "
            "pair-force contact contract, and the frozen reference-contact masks."
        ),
    )
    parser.add_argument(
        "--strict-per-finger-contact-reward-v4",
        action="store_true",
        help="Evaluate a frozen Strict Per-Finger V4 checkpoint and source mask.",
    )
    parser.add_argument(
        "--strict-contact-diagnostic-allow-non-v4-checkpoint",
        action="store_true",
        help=(
            "Explicitly permit a V1/V3 policy checkpoint to run once in the frozen V4 "
            "environment only to capture strict source-contact telemetry. The result is "
            "diagnostic-only and cannot be used as a V4 qualification or export."
        ),
    )
    parser.add_argument("--contact-reward-contract", type=Path)
    parser.add_argument("--contact-mask-root", type=Path)
    parser.add_argument("--selected-capacity", type=Path)
    parser.add_argument("--frame-zero-replicas", type=int, default=4)
    parser.add_argument("--rsi-replicas", type=int, default=20)
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        help="Frozen seed-set JSON created by freeze_stage16d_ppo26d_continuation_inputs.py",
    )
    parser.add_argument(
        "--seed-set",
        default="development_eval_seed_set_v1",
        help="Named set in --seed-manifest; formal seeds are reserved for R7",
    )
    parser.add_argument(
        "--artifact-label",
        default="l0",
        help="Distinct label such as l0_rebaseline, r6a_2m, r6a_4m, or formal_r7",
    )
    parser.add_argument(
        "--capture-all-frame-zero-replicas",
        action="store_true",
        help="Export all frame-zero physical states for R7 exact-geometry qualification.",
    )
    parser.add_argument(
        "--capture-exact-fingertip-object-pair-force",
        action="store_true",
        help=(
            "Record the current object-side filtered PhysX force vector from each of the "
            "five named fingertips. This is trace-only telemetry and does not change the "
            "active reward, policy, physics, or actions."
        ),
    )
    parser.add_argument(
        "--capture-full-hand-object-pair-telemetry",
        action="store_true",
        help=(
            "Export the named 21-body active-object filtered pair matrix for diagnostics; "
            "this never changes policy, reward, action, observation, or physics."
        ),
    )
    parser.add_argument(
        "--trace-name",
        help="Optional trace filename under the selected clip output directory.",
    )
    parser.add_argument(
        "--hand-gravity-mode",
        choices=("current_off", "ablation_on"),
        default="current_off",
        help="Diagnostic-only controlled-hand gravity setting for full-trajectory evaluation.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_progress(output: Path, phase: str) -> None:
    write_json(
        output / "evaluation_progress.json",
        {"schema_version": "Stage16DPPO26DEvaluationProgressV1", "phase": phase},
    )


def checkpoint_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def evaluation_seeds(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """Return exact frozen seeds; never silently substitute new random draws."""

    if args.seed_manifest is None:
        return list(range(args.frame_zero_replicas)), list(range(args.rsi_replicas))
    payload = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    row = payload.get(args.seed_set)
    if not isinstance(row, dict) or not isinstance(row.get("seeds"), list):
        raise ValueError(f"PPO26D_EVALUATION_SEED_SET_MISSING:{args.seed_set}")
    seeds = [int(value) for value in row["seeds"]]
    if len(seeds) < max(args.frame_zero_replicas, args.rsi_replicas):
        raise ValueError("PPO26D_EVALUATION_SEED_SET_TOO_SMALL")
    return seeds[: args.frame_zero_replicas], seeds[: args.rsi_replicas]


def apply_episode_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rsi_start_indices(seeds: list[int], *, frame_count: int) -> list[int]:
    """Map each named episode seed to one RSI start index without global RNG drift."""

    return [int(np.random.default_rng(seed).integers(frame_count)) for seed in seeds]


def pad_parallel_seeds(seeds: list[int], *, count: int) -> list[int]:
    if not seeds or count < len(seeds):
        raise ValueError("parallel seed padding requires a non-empty non-truncated seed list")
    return seeds + [seeds[-1]] * (count - len(seeds))


def model_from_checkpoint(
    path: Path, device: str, *, expected_clip: str
) -> tuple[PPO26DTrainer, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=device)
    schema = payload.get("schema_version")
    if schema not in {
        "Stage16DPPO26DCheckpointV1",
        PHASE3_CHECKPOINT_SCHEMA,
        REWARD_V3_CHECKPOINT_SCHEMA,
        STRICT_V4_CHECKPOINT_SCHEMA,
        PHYSICAL_PPO_CHECKPOINT_SCHEMA,
        FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA,
    }:
        raise ValueError("CHECKPOINT_ROUNDTRIP_FAILURE: unexpected checkpoint schema")
    checkpoint_clip = payload.get("clip")
    # The frozen V1 L0 artifact was created before clip metadata became part
    # of the checkpoint contract.  Its filename/location are clip-specific;
    # allow only that absent legacy field, never a conflicting value.
    if checkpoint_clip not in {None, expected_clip}:
        raise ValueError(
            "PPO26D_CHECKPOINT_CLIP_MISMATCH: "
            f"checkpoint={checkpoint_clip!r} requested={expected_clip!r}"
        )
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.cumulative_samples = int(
        payload["reward_v2_samples"]
        if schema == PHASE3_CHECKPOINT_SCHEMA
        else (
            payload["reward_v3_samples"]
            if schema == REWARD_V3_CHECKPOINT_SCHEMA
            else (
                payload["policy_training_samples"]
                if schema == PHYSICAL_PPO_CHECKPOINT_SCHEMA
                else (
                    payload["reward_v4_samples"]
                    if schema == STRICT_V4_CHECKPOINT_SCHEMA
                    else payload["cumulative_samples"]
                )
            )
        )
    )
    trainer.trainer.freeze_observation_normalizer()
    return trainer, payload


def checkpoint_sample_counter(payload: dict[str, Any]) -> tuple[str, int]:
    """Keep Reward V2 sample accounting distinct from V1 cumulative samples."""

    if payload.get("schema_version") == PHASE3_CHECKPOINT_SCHEMA:
        return "reward_v2_samples", int(payload["reward_v2_samples"])
    if payload.get("schema_version") == REWARD_V3_CHECKPOINT_SCHEMA:
        return "reward_v3_samples", int(payload["reward_v3_samples"])
    if payload.get("schema_version") == STRICT_V4_CHECKPOINT_SCHEMA:
        return "reward_v4_samples", int(payload["reward_v4_samples"])
    if payload.get("schema_version") == PHYSICAL_PPO_CHECKPOINT_SCHEMA:
        return "policy_training_samples", int(payload["policy_training_samples"])
    if payload.get("schema_version") == FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA:
        return "policy_training_samples", int(payload["policy_training_samples"])
    return "cumulative_training_samples", int(payload["cumulative_samples"])


def _to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _device_trace_to_numpy(
    trace: dict[str, torch.Tensor], *, all_replicas: bool = False
) -> dict[str, np.ndarray]:
    """Make the sole host transfer after a completed simulator-thread capture."""

    torch.cuda.synchronize()
    return {
        name: (value if all_replicas else value[:, 0]).detach().cpu().numpy().copy()
        for name, value in trace.items()
    }


def _initial_trace_snapshot(
    env: Any,
    *,
    capture_exact_fingertip_object_pair_force: bool,
    capture_full_hand_object_pair_telemetry: bool,
) -> dict[str, np.ndarray]:
    """Capture the physical reset state as trace frame zero without collision reads.

    The simulator callback can safely collect post-physics control rows, but a
    321-key factor-8 reference has 320 control intervals.  This one host-side
    physical-state snapshot supplies the missing reset key so formal geometry
    sees the complete 321-state trajectory rather than a synthetic pad.
    """

    state = env._state()
    clips = env._clip_index
    indices = env._reference_index
    count = env.num_envs
    device = env.device
    object_reference = torch.cat(
        (
            env.reference_bank.gather("object_pose_translation_world_ref", clips, indices),
            env.reference_bank.gather("object_pose_quaternion_world_ref_wxyz", clips, indices),
        ),
        dim=-1,
    )
    wrist_reference = torch.cat(
        (
            env.reference_bank.gather("wrist_pose_translation_world_ref", clips, indices),
            env.reference_bank.gather("wrist_pose_quaternion_world_ref_wxyz", clips, indices),
        ),
        dim=-1,
    )
    virtual_wrist_ids = env._virtual_wrist_joint_ids
    if len(virtual_wrist_ids) != 6:
        raise RuntimeError("PPO26D trace export requires the finite six-DoF virtual wrist")
    values = {
        "object_pose": torch.cat(
            (state["object_position_scene"], state["object_quaternion_wxyz"]), dim=-1
        ),
        "object_twist": state["object_twist_world"],
        "wrist_pose": torch.cat(
            (state["wrist_position_scene"], state["wrist_quaternion_wxyz"]), dim=-1
        ),
        "finger_q": state["finger_q"],
        "finger_qdot": state["finger_qdot"],
        "wrist_twist_world": state["wrist_twist_world"],
        "virtual_wrist_q": env._robot.data.joint_pos[:, virtual_wrist_ids],
        "virtual_wrist_qdot": env._robot.data.joint_vel[:, virtual_wrist_ids],
        "virtual_wrist_target_q": env._explicit_wrist_joint_target,
        "virtual_wrist_target_qdot": env._explicit_wrist_joint_velocity_target,
        "object_axis_points": state["object_axis_points_scene"],
        "tracked_link_positions": state["tracked_links_scene"],
        "contact_force_world": torch.zeros((count, 3), device=device),
        "contact_pair_presence": torch.zeros((count, 21), dtype=torch.bool, device=device),
        "actuator_effort": torch.zeros((count, 26), device=device),
        "reason_code": torch.zeros(count, dtype=torch.long, device=device),
        "terminated": torch.zeros(count, dtype=torch.bool, device=device),
        "timed_out": torch.zeros(count, dtype=torch.bool, device=device),
        "action": torch.zeros((count, 26), device=device),
        "clip_index": clips,
        "reward_total": torch.zeros(count, device=device),
        "reward_object": torch.zeros(count, device=device),
        "reward_link": torch.zeros(count, device=device),
        "reward_finger": torch.zeros(count, device=device),
        "reward_wrist_translation": torch.zeros(count, device=device),
        "reward_wrist_rotation": torch.zeros(count, device=device),
        "reward_smoothness": torch.zeros(count, device=device),
        "wrist_residual": torch.zeros((count, 6), device=device),
        "finger_residual": torch.zeros((count, 20), device=device),
        "wrist_target": wrist_reference,
        "finger_target": env.reference_bank.gather("q_finger_ref", clips, indices),
        "object_reference": object_reference,
        "wrist_reference": wrist_reference,
        "finger_reference": env.reference_bank.gather("q_finger_ref", clips, indices),
        "tracked_link_reference": env.reference_bank.gather(
            "tracked_link_positions_world_ref", clips, indices
        ),
        "reference_index": indices,
    }
    if env.cfg.ppo26d_reward_contract in {
        "TopoRetargetReferenceTrackingReward26DV2",
        "TopoRetargetReferenceTrackingReward26DV3",
        "TopoRetargetReferenceTrackingReward26DV4",
    }:
        values.update(
            {
                "object_twist_reference": env.reference_bank.gather(
                    "object_twist_world_ref", clips, indices
                ),
                "reward_obj_vel": torch.zeros(count, device=device),
                "reward_obj_ang_vel": torch.zeros(count, device=device),
                "error_obj_vel": torch.zeros(count, device=device),
                "error_obj_ang_vel": torch.zeros(count, device=device),
            }
        )
    if env.cfg.ppo26d_reward_contract == "TopoRetargetReferenceTrackingReward26DV3":
        # Like pair force, these are a reset snapshot rather than a sensor/reward
        # callback row.  Keep their dimensions aligned with V3 post-physics trace
        # capture; frame zero's force-derived terms are deliberately all zero.
        values.update(
            {
                "reference_contact_mask": env._reference_expected_contact_mask(indices),
                "actual_contact_mask": torch.zeros((count, 5), dtype=torch.bool, device=device),
                "fingertip_object_pair_force_world": torch.zeros((count, 5, 3), device=device),
                "fingertip_object_force_magnitude": torch.zeros((count, 5), device=device),
                "contact_reward": torch.zeros(count, device=device),
                "contact_force_scale": torch.zeros(count, device=device),
            }
        )
    elif env.cfg.ppo26d_reward_contract == "TopoRetargetReferenceTrackingReward26DV4":
        values.update(
            {
                "source_contact_mask": env._reference_expected_contact_mask(indices),
                "tip_pair_presence": torch.zeros((count, 5), dtype=torch.bool, device=device),
                "tip_pair_force_world": torch.zeros((count, 5, 3), device=device),
                "tip_pair_force_norm": torch.zeros((count, 5), device=device),
                "per_finger_contact_reward": torch.zeros((count, 5), device=device),
                "source_expected_finger_count": torch.zeros(count, dtype=torch.long, device=device),
                "source_satisfied_tip_count": torch.zeros(count, dtype=torch.long, device=device),
                "source_tip_coverage_ratio": torch.zeros(count, device=device),
                "r_contact_v4": torch.zeros(count, device=device),
                "reference_contact_mask": env._reference_expected_contact_mask(indices),
                "actual_contact_mask": torch.zeros((count, 5), dtype=torch.bool, device=device),
                "fingertip_object_pair_force_world": torch.zeros((count, 5, 3), device=device),
                "fingertip_object_force_magnitude": torch.zeros((count, 5), device=device),
                "contact_reward": torch.zeros(count, device=device),
            }
        )
    if capture_exact_fingertip_object_pair_force:
        # Frame zero is a physical reset state, not a post-physics sensor read.
        # Preserve the 321-key state trace while preventing this placeholder from
        # entering any contact-force calibration.
        values.update(
            {
                "fingertip_object_pair_force_world": torch.zeros((count, 5, 3), device=device),
                "fingertip_object_pair_force_valid": torch.zeros(
                    count, dtype=torch.bool, device=device
                ),
            }
        )
    if capture_full_hand_object_pair_telemetry:
        # Frame zero has no post-physics sensor sample.  Keep it explicitly
        # invalid rather than manufacturing no-contact evidence.
        values.update(
            {
                "hand_object_pair_force_world": torch.zeros((count, 21, 3), device=device),
                "hand_object_pair_presence": torch.zeros(
                    (count, 21), dtype=torch.bool, device=device
                ),
                "hand_object_pair_force_valid": torch.zeros(count, dtype=torch.bool, device=device),
            }
        )
    if getattr(env.cfg, "stage16_support_mode", None) == "finite_inferred_table_proxy_v1":
        sensor_name = (
            "object_170105_support_contact"
            if env.cfg.stage16d_fixed_clip == "hocap_170105"
            else "object_170650_support_contact"
        )
        force = env.scene[sensor_name].data.force_matrix_w
        values["table_object_contact"] = (
            torch.linalg.vector_norm(force, dim=-1).amax(dim=(1, 2)) > 1.0e-4
        )
    return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}


def _prepend_initial_trace(
    trace: dict[str, np.ndarray], initial: dict[str, np.ndarray], *, all_replicas: bool
) -> dict[str, np.ndarray]:
    """Prepend the reset state to a post-physics trace with strict shape checks."""

    result = {}
    for name, values in trace.items():
        initial_values = initial[name] if all_replicas else initial[name][0]
        if values.shape[1:] != initial_values.shape:
            raise ValueError(f"initial trace shape mismatch for {name}")
        result[name] = np.concatenate((initial_values[None], values), axis=0)
    return result


def run_episode(
    env: Any,
    trainer: PPO26DTrainer,
    *,
    capture: bool,
    capture_exact_fingertip_object_pair_force: bool,
    capture_full_hand_object_pair_telemetry: bool,
    expected_clip: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run one physical rollout without reading collision articulation tensors.

    ``start_trace_capture`` records post-physics wrist/finger/object state in
    the environment's reward callback.  Collision-body poses are reconstructed
    after the rollout from that physical state, avoiding an unsafe articulation
    tensor read from the Python evaluation loop.
    """

    if seed is not None:
        apply_episode_seed(seed)
    observation, _ = env.reset(seed=seed)
    active_clip_indices = sorted(set(env._clip_index.detach().cpu().tolist()))
    expected_clip_index = env.reference_bank.clip_ids.index(expected_clip)
    if active_clip_indices != [expected_clip_index]:
        raise RuntimeError(
            "PPO26D_FIXED_CLIP_MISMATCH_AFTER_RESET: "
            f"expected={expected_clip_index} active={active_clip_indices}"
        )
    initial_trace = (
        _initial_trace_snapshot(
            env,
            capture_exact_fingertip_object_pair_force=capture_exact_fingertip_object_pair_force,
            capture_full_hand_object_pair_telemetry=capture_full_hand_object_pair_telemetry,
        )
        if capture
        else None
    )
    if capture:
        env.start_trace_capture(
            capacity=env.reference_bank.frame_count,
            capture_exact_fingertip_object_pair_force=capture_exact_fingertip_object_pair_force,
            capture_full_hand_object_pair_telemetry=capture_full_hand_object_pair_telemetry,
        )
    start_reference_index = int(env._reference_index[0].item())
    total_reward = 0.0
    contact_seen = False
    terminal_contact = False
    contact_step_count = 0
    final_extras: dict[str, Any] = {}
    object_tracking_errors: list[float] = []
    object_rotation_errors: list[float] = []
    object_axis_errors: list[float] = []
    wrist_position_errors: list[float] = []
    wrist_rotation_errors: list[float] = []
    finger_errors: list[float] = []
    contact_indices: list[int] = []
    for _ in range(env.reference_bank.frame_count):
        state = env._state()
        index = int(env._reference_index[0].item())
        with torch.no_grad():
            distribution = trainer.trainer.distribution(observation["policy"])
            action = distribution.mean
        object_reference_position = env.reference_bank.object_pose_translation_world_ref[
            expected_clip_index, index
        ]
        object_reference_quaternion = env.reference_bank.object_pose_quaternion_world_ref_wxyz[
            expected_clip_index, index
        ]
        object_reference_axis_points = env.reference_bank.object_axis_points_world_ref[
            expected_clip_index, index
        ]
        wrist_reference_position = env.reference_bank.wrist_pose_translation_world_ref[
            expected_clip_index, index
        ]
        wrist_reference_quaternion = env.reference_bank.wrist_pose_quaternion_world_ref_wxyz[
            expected_clip_index, index
        ]
        finger_reference = env.reference_bank.q_finger_ref[expected_clip_index, index]
        object_tracking_errors.append(
            _to_float(
                torch.linalg.vector_norm(
                    state["object_position_scene"][0] - object_reference_position
                )
            )
        )
        object_rotation_errors.append(
            _to_float(
                torch.acos(
                    torch.clamp(
                        torch.abs(
                            (state["object_quaternion_wxyz"][0] * object_reference_quaternion).sum()
                        ),
                        max=1.0,
                    )
                )
                * 2.0
            )
        )
        object_axis_errors.append(
            _to_float(
                torch.linalg.vector_norm(
                    state["object_axis_points_scene"][0] - object_reference_axis_points,
                    dim=-1,
                ).mean()
            )
        )
        wrist_position_errors.append(
            _to_float(
                torch.linalg.vector_norm(
                    state["wrist_position_scene"][0] - wrist_reference_position
                )
            )
        )
        wrist_rotation_errors.append(
            _to_float(
                torch.acos(
                    torch.clamp(
                        torch.abs(
                            (state["wrist_quaternion_wxyz"][0] * wrist_reference_quaternion).sum()
                        ),
                        max=1.0,
                    )
                )
                * 2.0
            )
        )
        finger_errors.append(
            _to_float(torch.mean(torch.abs(state["finger_q"][0] - finger_reference)))
        )
        observation, reward, terminated, timed_out, extras = env.step(action)
        total_reward += _to_float(reward[0])
        final_extras = extras["ppo26d"]
        terminal_contact = bool(final_extras["contact_any"][0].detach().cpu())
        contact_seen |= terminal_contact
        contact_step_count += int(terminal_contact)
        if terminal_contact:
            contact_indices.append(len(object_tracking_errors) - 1)
        if bool(terminated[0] | timed_out[0]):
            break
    trace = (
        _prepend_initial_trace(
            _device_trace_to_numpy(env.finish_trace_capture()),
            initial_trace,
            all_replicas=False,
        )
        if capture
        else None
    )
    linear_speed = _to_float(final_extras["object_linear_speed_mps"][0])
    angular_speed = _to_float(final_extras["object_angular_speed_radps"][0])
    termination_reason = int(final_extras["primary_reason_code"][0].detach().cpu())
    steps = len(object_tracking_errors)
    longest_window = 0
    running_window = 0
    previous_contact = None
    for contact_index in contact_indices:
        running_window = running_window + 1 if previous_contact == contact_index - 1 else 1
        longest_window = max(longest_window, running_window)
        previous_contact = contact_index
    return {
        "steps": steps,
        "start_reference_index": start_reference_index,
        "final_reference_index": min(
            start_reference_index + steps, env.reference_bank.frame_count - 1
        ),
        "clip": expected_clip,
        "clip_index": expected_clip_index,
        "reached_final_reference": termination_reason == 7,
        "total_reward": total_reward,
        "contact": contact_seen,
        "contact_step_count": contact_step_count,
        "contact_fraction": contact_step_count / max(steps, 1),
        "first_contact_index": contact_indices[0] if contact_indices else None,
        "last_contact_index": contact_indices[-1] if contact_indices else None,
        "longest_continuous_contact_window": longest_window,
        "terminal_contact": terminal_contact,
        "terminal_object_linear_speed_mps": linear_speed,
        "terminal_object_angular_speed_radps": angular_speed,
        "termination_reason": termination_reason,
        "object_tracking_error_m": {
            "mean": float(np.mean(object_tracking_errors)),
            "final": float(object_tracking_errors[-1]),
        },
        "final_object_rotation_error_rad": float(object_rotation_errors[-1]),
        "final_object_axis_error_m": float(object_axis_errors[-1]),
        "wrist_tracking": {
            "position_error_m_mean": float(np.mean(wrist_position_errors)),
            "rotation_error_rad_mean": float(np.mean(wrist_rotation_errors)),
        },
        "finger_tracking_error_rad_mean": float(np.mean(finger_errors)),
        "self_collision_enabled": bool(
            env.cfg.robot.spawn.articulation_props.enabled_self_collisions
        ),
        "inter_finger_penetration_m": "NOT_CAPTURED_IN_PPO_L0_REPLAY_TRACE",
        "terminal_stability": "POST_PPO_QUALIFICATION_NOT_RUN",
        "trace": trace,
    }


def run_parallel_episodes(
    env: Any,
    trainer: PPO26DTrainer,
    *,
    capture: bool,
    capture_all_replicas: bool,
    capture_exact_fingertip_object_pair_force: bool,
    capture_full_hand_object_pair_telemetry: bool,
    expected_clip: str,
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Evaluate named episodes in one physical vector rollout.

    The evaluator deliberately records each environment only until its first
    termination.  Later automatic vector resets are ignored, so batching does
    not convert one formal episode into multiple attempts.
    """

    if len(seeds) != env.num_envs:
        raise ValueError("parallel evaluation seeds must match num_envs")
    observation, _ = env.reset(seed=seeds[0])
    expected_clip_index = env.reference_bank.clip_ids.index(expected_clip)
    active_clip_indices = sorted(set(env._clip_index.detach().cpu().tolist()))
    if active_clip_indices != [expected_clip_index]:
        raise RuntimeError("PPO26D_FIXED_CLIP_MISMATCH_AFTER_RESET")
    initial_trace = (
        _initial_trace_snapshot(
            env,
            capture_exact_fingertip_object_pair_force=capture_exact_fingertip_object_pair_force,
            capture_full_hand_object_pair_telemetry=capture_full_hand_object_pair_telemetry,
        )
        if capture
        else None
    )
    if capture:
        env.start_trace_capture(
            capacity=env.reference_bank.frame_count,
            capture_exact_fingertip_object_pair_force=capture_exact_fingertip_object_pair_force,
            capture_full_hand_object_pair_telemetry=capture_full_hand_object_pair_telemetry,
        )
    device = env.device
    count = env.num_envs
    active = torch.ones(count, dtype=torch.bool, device=device)
    steps = torch.zeros(count, dtype=torch.long, device=device)
    contact_seen = torch.zeros(count, dtype=torch.bool, device=device)
    contact_steps = torch.zeros(count, dtype=torch.long, device=device)
    first_contact = torch.full((count,), -1, dtype=torch.long, device=device)
    last_contact = torch.full((count,), -1, dtype=torch.long, device=device)
    current_window = torch.zeros(count, dtype=torch.long, device=device)
    longest_window = torch.zeros(count, dtype=torch.long, device=device)
    total_reward = torch.zeros(count, dtype=torch.float64, device=device)
    final_error = torch.full((count,), float("nan"), device=device)
    error_sum = torch.zeros(count, device=device)
    final_rotation = torch.full((count,), float("nan"), device=device)
    final_axis = torch.full((count,), float("nan"), device=device)
    final_linear_speed = torch.full((count,), float("nan"), device=device)
    final_angular_speed = torch.full((count,), float("nan"), device=device)
    final_reason = torch.full((count,), -1, dtype=torch.long, device=device)
    terminal_contact = torch.zeros(count, dtype=torch.bool, device=device)
    start_indices = env._reference_index.clone()

    for _ in range(env.reference_bank.frame_count):
        state = env._state()
        index = env._reference_index
        object_reference_position = env.reference_bank.gather(
            "object_pose_translation_world_ref", env._clip_index, index
        )
        object_reference_quaternion = env.reference_bank.gather(
            "object_pose_quaternion_world_ref_wxyz", env._clip_index, index
        )
        object_reference_axis = env.reference_bank.gather(
            "object_axis_points_world_ref", env._clip_index, index
        )
        position_error = torch.linalg.vector_norm(
            state["object_position_scene"] - object_reference_position, dim=-1
        )
        rotation_error = (
            torch.acos(
                torch.clamp(
                    torch.abs(
                        (state["object_quaternion_wxyz"] * object_reference_quaternion).sum(dim=-1)
                    ),
                    max=1.0,
                )
            )
            * 2.0
        )
        axis_error = torch.linalg.vector_norm(
            state["object_axis_points_scene"] - object_reference_axis, dim=-1
        ).mean(dim=-1)
        error_sum += torch.where(active, position_error, 0.0)
        final_error = torch.where(active, position_error, final_error)
        final_rotation = torch.where(active, rotation_error, final_rotation)
        final_axis = torch.where(active, axis_error, final_axis)
        with torch.no_grad():
            action = trainer.trainer.distribution(observation["policy"]).mean
        observation, reward, terminated, timed_out, extras = env.step(action)
        step_active = active.clone()
        total_reward += torch.where(step_active, reward.double(), 0.0)
        steps += step_active.long()
        diagnostic = extras["ppo26d"]
        contact = diagnostic["contact_any"].bool() & step_active
        contact_seen |= contact
        contact_steps += contact.long()
        first_contact = torch.where(contact & (first_contact < 0), steps - 1, first_contact)
        last_contact = torch.where(contact, steps - 1, last_contact)
        current_window = torch.where(contact, current_window + 1, torch.zeros_like(current_window))
        longest_window = torch.maximum(longest_window, current_window)
        done = (terminated | timed_out) & step_active
        terminal_contact = torch.where(done, diagnostic["contact_any"].bool(), terminal_contact)
        final_linear_speed = torch.where(
            done, diagnostic["object_linear_speed_mps"], final_linear_speed
        )
        final_angular_speed = torch.where(
            done, diagnostic["object_angular_speed_radps"], final_angular_speed
        )
        final_reason = torch.where(done, diagnostic["primary_reason_code"], final_reason)
        active &= ~done
        if not bool(active.any()):
            break
    if capture:
        device_trace = env.finish_trace_capture()
        assert initial_trace is not None
        trace = _prepend_initial_trace(
            _device_trace_to_numpy(device_trace), initial_trace, all_replicas=False
        )
        all_replica_trace = (
            _prepend_initial_trace(
                _device_trace_to_numpy(device_trace, all_replicas=True),
                initial_trace,
                all_replicas=True,
            )
            if capture_all_replicas
            else None
        )
    else:
        trace = None
        all_replica_trace = None
    result = []
    for env_index, seed in enumerate(seeds):
        step_count = int(steps[env_index].item())
        if step_count <= 0 or int(final_reason[env_index].item()) < 0:
            raise RuntimeError("PPO26D_PARALLEL_EVALUATION_DID_NOT_TERMINATE")
        result.append(
            {
                "seed": seed,
                "steps": step_count,
                "start_reference_index": int(start_indices[env_index].item()),
                "final_reference_index": min(
                    int(start_indices[env_index].item()) + step_count,
                    env.reference_bank.frame_count - 1,
                ),
                "clip": expected_clip,
                "clip_index": expected_clip_index,
                "reached_final_reference": int(final_reason[env_index].item()) == 7,
                "total_reward": float(total_reward[env_index].item()),
                "contact": bool(contact_seen[env_index].item()),
                "contact_step_count": int(contact_steps[env_index].item()),
                "contact_fraction": float(contact_steps[env_index].item()) / step_count,
                "first_contact_index": (
                    None
                    if int(first_contact[env_index].item()) < 0
                    else int(first_contact[env_index].item())
                ),
                "last_contact_index": (
                    None
                    if int(last_contact[env_index].item()) < 0
                    else int(last_contact[env_index].item())
                ),
                "longest_continuous_contact_window": int(longest_window[env_index].item()),
                "terminal_contact": bool(terminal_contact[env_index].item()),
                "terminal_object_linear_speed_mps": float(final_linear_speed[env_index].item()),
                "terminal_object_angular_speed_radps": float(final_angular_speed[env_index].item()),
                "termination_reason": int(final_reason[env_index].item()),
                "object_tracking_error_m": {
                    "mean": float((error_sum[env_index] / step_count).item()),
                    "final": float(final_error[env_index].item()),
                },
                "final_object_rotation_error_rad": float(final_rotation[env_index].item()),
                "final_object_axis_error_m": float(final_axis[env_index].item()),
                "self_collision_enabled": bool(
                    env.cfg.robot.spawn.articulation_props.enabled_self_collisions
                ),
                "inter_finger_penetration_m": "NOT_CAPTURED_IN_PPO_DEVELOPMENT_TRACE",
                "terminal_stability": "POST_PPO_QUALIFICATION_NOT_RUN",
                "trace": trace if env_index == 0 else None,
                "all_replica_trace": all_replica_trace if env_index == 0 else None,
            }
        )
    return result


def validate_trace_rows(trace: dict[str, np.ndarray]) -> None:
    required = {
        "object_pose": (7,),
        "object_twist": (6,),
        "hand_collision_body_pose": (21, 7),
        "contact_force_world": (3,),
        "contact_pair_presence": (21,),
        "actuator_effort": (26,),
        "reason_code": (),
        "terminated": (),
        "timed_out": (),
        "action": (26,),
        "clip_index": (),
        "object_reference": (7,),
        "wrist_reference": (7,),
        "finger_reference": (20,),
        "tracked_link_reference": (16, 3),
        "wrist_residual": (6,),
        "wrist_target": (7,),
        "finger_target": (20,),
    }
    for name, suffix in required.items():
        value = trace[name]
        if value.ndim != len(suffix) + 1 or value.shape[1:] != suffix:
            raise ValueError(f"PPO26D trace {name} has shape {value.shape}, expected [T, {suffix}]")
        if name != "contact_pair_presence" and not np.isfinite(value).all():
            raise ValueError(f"PPO26D trace {name} contains non-finite values")
    quaternion_norm = np.linalg.norm(trace["hand_collision_body_pose"][..., 3:7], axis=-1)
    if np.any(quaternion_norm < 1.0e-8):
        raise ValueError("PPO26D trace contains a zero hand collision-body quaternion")


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    legacy_mode = (
        ContactRewardMode.AGGREGATE_V3
        if args.reference_gated_contact_reward_v3
        else (
            ContactRewardMode.STRICT_PER_FINGER_V4
            if args.strict_per_finger_contact_reward_v4
            else None
        )
    )
    configured_mode = (
        ContactRewardMode.parse(args.contact_mode) if args.contact_mode is not None else None
    )
    if (
        configured_mode is not None
        and legacy_mode is not None
        and configured_mode is not legacy_mode
    ):
        raise ValueError("STAGE16D_CONTACT_MODE_LEGACY_FLAG_CONFLICT")
    contact_mode = configured_mode or legacy_mode
    if contact_mode is not None:
        args.reference_gated_contact_reward_v3 = contact_mode is ContactRewardMode.AGGREGATE_V3
        args.strict_per_finger_contact_reward_v4 = (
            contact_mode is ContactRewardMode.STRICT_PER_FINGER_V4
        )
    if (
        args.object_twist_reward_v2
        or args.reference_gated_contact_reward_v3
        or args.strict_per_finger_contact_reward_v4
    ) and args.reference_kinematics_v2_root is None:
        raise ValueError("PPO26D_REWARD_V2_REQUIRES_REFERENCE_KINEMATICS_V2")
    if (
        sum(
            bool(value)
            for value in (
                args.object_twist_reward_v2,
                args.reference_gated_contact_reward_v3,
                args.strict_per_finger_contact_reward_v4,
            )
        )
        > 1
    ):
        raise ValueError("PPO26D_REWARD_EVALUATION_MODES_ARE_MUTUALLY_EXCLUSIVE")
    if args.reference_gated_contact_reward_v3 and (
        args.contact_reward_contract is None or args.contact_mask_root is None
    ):
        raise ValueError("PPO26D_REWARD_V3_REQUIRES_FROZEN_CONTACT_INPUTS")
    if args.strict_per_finger_contact_reward_v4 and (
        args.contact_reward_contract is None or args.contact_mask_root is None
    ):
        raise ValueError("STRICT_V4_REQUIRES_FROZEN_CONTACT_INPUTS")
    if (
        args.strict_contact_diagnostic_allow_non_v4_checkpoint
        and not args.strict_per_finger_contact_reward_v4
    ):
        raise ValueError("STRICT_CONTACT_DIAGNOSTIC_REQUIRES_V4_RUNTIME_MODE")
    if args.frame_zero_replicas <= 0 or args.rsi_replicas < 0:
        raise ValueError("--frame-zero-replicas must be positive and --rsi-replicas non-negative")
    if args.full_trajectory_table and (
        contact_mode is None or args.rsi_replicas != 0 or args.curriculum_stage != "C4"
    ):
        raise ValueError("FULL_TRAJECTORY_C4_EVALUATION_REQUIRES_CONTACT_MODE_AND_NO_RSI")
    if args.hand_gravity_mode != "current_off" and not args.full_trajectory_table:
        raise ValueError("HAND_GRAVITY_ABLATION_REQUIRES_FULL_TRAJECTORY_TABLE")
    if not args.artifact_label.replace("_", "").isalnum():
        raise ValueError("--artifact-label must be alphanumeric/underscore")
    if args.capture_exact_fingertip_object_pair_force and not args.capture_all_frame_zero_replicas:
        raise ValueError("PPO26D_PAIR_FORCE_CAPTURE_REQUIRES_ALL_FRAME_ZERO_REPLICAS")
    if args.capture_full_hand_object_pair_telemetry and not args.capture_all_frame_zero_replicas:
        raise ValueError("PPO26D_FULL_HAND_PAIR_CAPTURE_REQUIRES_ALL_FRAME_ZERO_REPLICAS")
    if args.trace_name is not None and (
        Path(args.trace_name).name != args.trace_name or not args.trace_name.endswith(".npz")
    ):
        raise ValueError("PPO26D_TRACE_NAME_MUST_BE_A_LOCAL_NPZ_FILENAME")
    frame_zero_seeds, rsi_seeds = evaluation_seeds(args)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    root = args.output_root.resolve()
    output = root / args.clip
    checkpoint = (
        args.checkpoint or output / f"stage16d_ppo26d_{args.clip.removeprefix('hocap_')}_l0.pt"
    )
    # Keep the launcher alive for the complete evaluation.  Constructing only
    # ``AppLauncher(...).app`` lets the temporary launcher be finalized and
    # closes Isaac before the table-supported environment is built.
    app_launcher = AppLauncher(headless=True)
    app = app_launcher.app
    env = None
    output.mkdir(parents=True, exist_ok=True)
    try:
        # Import optional Isaac modules only after this process owns the app.
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            HAND_COLLISION_BODY_NAMES,
        )

        evaluation_num_envs = max(args.frame_zero_replicas, args.rsi_replicas)
        if args.full_trajectory_table:
            full_trajectory_helper = REPO_ROOT / "scripts/rl/isaaclab"
            if str(full_trajectory_helper) not in sys.path:
                sys.path.insert(0, str(full_trajectory_helper))
            from smoke_stage16_full_trajectory_ppo import _load_start, _make_table_env

            assert contact_mode is not None
            start = _load_start(args.clip)
            robot_usd_path = None
            if args.hand_gravity_mode == "ablation_on":
                from scripts.rl.isaaclab.inspect_stage16_hand_gravity import (
                    _materialize_gravity_on_ablation,
                )

                robot_usd_path = _materialize_gravity_on_ablation()
            env = _make_table_env(
                clip=args.clip,
                num_envs=evaluation_num_envs,
                start_index=int(start["start_index"]),
                mode=contact_mode,
                stage=args.curriculum_stage,
                robot_usd_path=robot_usd_path,
            )
            env.cfg.ppo26d_full_horizon_evaluation = True
        else:
            from toporetarget.rl.environments.isaaclab_backend import (
                ppo26d_reference_tracking_env,
            )
            from toporetarget.rl.environments.isaaclab_backend import (
                ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
            )

            cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
            ppo26d_cfg.configure_stage16d_ppo26d(
                cfg, num_envs=evaluation_num_envs, clip=args.clip, rsi=False, critical_dr=False
            )
            if contact_mode is not None:
                assert args.reference_kinematics_v2_root is not None
                assert args.contact_reward_contract is not None
                assert args.contact_mask_root is not None
                ppo26d_cfg.configure_stage16d_contact_reward(
                    cfg,
                    mode=contact_mode,
                    reference_root=args.reference_kinematics_v2_root,
                    contact_reward_contract=args.contact_reward_contract,
                    contact_mask_root=args.contact_mask_root,
                )
            elif args.object_twist_reward_v2:
                assert args.reference_kinematics_v2_root is not None
                ppo26d_cfg.configure_stage16d_phase3_object_twist_reward(
                    cfg, reference_root=args.reference_kinematics_v2_root
                )
            elif args.reference_kinematics_v2_root is not None:
                ppo26d_cfg.configure_stage16d_reference_kinematics_v2(
                    cfg, reference_root=args.reference_kinematics_v2_root
                )
            env = ppo26d_reference_tracking_env.IsaacPPO26DReferenceTrackingEnv(cfg)
        reference_kinematics_version = int(env.cfg.reference_kinematics_version)
        trainer, payload = model_from_checkpoint(
            checkpoint, str(env.device), expected_clip=args.clip
        )
        strict_contact_diagnostic = bool(args.strict_contact_diagnostic_allow_non_v4_checkpoint)
        if args.strict_per_finger_contact_reward_v4:
            full_v4 = (
                payload.get("schema_version") == FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA
                and payload.get("contact_mode") == ContactRewardMode.STRICT_PER_FINGER_V4.value
            )
            if payload.get("schema_version") != STRICT_V4_CHECKPOINT_SCHEMA and not full_v4:
                if not strict_contact_diagnostic or payload.get("schema_version") not in {
                    "Stage16DPPO26DCheckpointV1",
                    REWARD_V3_CHECKPOINT_SCHEMA,
                }:
                    raise ValueError("STRICT_V4_EVALUATION_REQUIRES_V4_CHECKPOINT")
            elif strict_contact_diagnostic:
                raise ValueError("STRICT_CONTACT_DIAGNOSTIC_REQUIRES_V1_OR_V3_CHECKPOINT")
            assert args.contact_reward_contract is not None
            checkpoint_entry = payload.get("strict_v4_contact_entry", {}).get("contract", {})
            if (
                not full_v4
                and not strict_contact_diagnostic
                and checkpoint_entry.get("sha256") != checkpoint_hash(args.contact_reward_contract)
            ):
                raise ValueError("STRICT_V4_CONTACT_CONTRACT_HASH_MISMATCH")
        elif args.reference_gated_contact_reward_v3:
            full_v3 = (
                payload.get("schema_version") == FULL_TRAJECTORY_P3_CHECKPOINT_SCHEMA
                and payload.get("contact_mode") == ContactRewardMode.AGGREGATE_V3.value
            )
            if payload.get("schema_version") != REWARD_V3_CHECKPOINT_SCHEMA and not full_v3:
                raise ValueError("PPO26D_REWARD_V3_EVALUATION_REQUIRES_V3_CHECKPOINT")
            assert args.contact_reward_contract is not None
            checkpoint_entry = payload.get("reward_v3_contact_entry", {}).get("contract", {})
            if not full_v3 and checkpoint_entry.get("sha256") != checkpoint_hash(
                args.contact_reward_contract
            ):
                raise ValueError("PPO26D_REWARD_V3_CONTACT_CONTRACT_HASH_MISMATCH")
        elif payload.get("schema_version") in {
            REWARD_V3_CHECKPOINT_SCHEMA,
            STRICT_V4_CHECKPOINT_SCHEMA,
        }:
            raise ValueError("PPO26D_V3_CHECKPOINT_REQUIRES_V3_EVALUATION_MODE")
        sample_counter_name, sample_counter = checkpoint_sample_counter(payload)
        selected = int(payload["selected_num_envs"])
        write_progress(output, "checkpoint_loaded")
        write_progress(output, "frame_zero_evaluation_started")
        frame_zero = run_parallel_episodes(
            env,
            trainer,
            capture=True,
            capture_all_replicas=args.capture_all_frame_zero_replicas,
            capture_exact_fingertip_object_pair_force=args.capture_exact_fingertip_object_pair_force,
            capture_full_hand_object_pair_telemetry=args.capture_full_hand_object_pair_telemetry,
            expected_clip=args.clip,
            seeds=pad_parallel_seeds(frame_zero_seeds, count=evaluation_num_envs),
        )[: args.frame_zero_replicas]
        trace_rows = frame_zero[0]["trace"]
        assert trace_rows is not None
        all_replica_trace = frame_zero[0]["all_replica_trace"]
        if tuple(HAND_COLLISION_BODY_NAMES) != FK_HAND_COLLISION_BODY_NAMES:
            raise RuntimeError("PPO26D hand collision-body order disagreement")
        trace_rows["hand_collision_body_pose"] = reconstruct_hand_collision_body_pose(
            trace_rows["wrist_pose"], trace_rows["finger_q"], repo_root=REPO_ROOT
        ).astype(np.float32)
        if all_replica_trace is not None:
            replica_wrist = np.asarray(all_replica_trace["wrist_pose"], dtype=np.float32)
            replica_finger = np.asarray(all_replica_trace["finger_q"], dtype=np.float32)
            replica_hand = np.stack(
                [
                    reconstruct_hand_collision_body_pose(
                        replica_wrist[:, replica], replica_finger[:, replica], repo_root=REPO_ROOT
                    )
                    for replica in range(replica_wrist.shape[1])
                ],
                axis=1,
            ).astype(np.float32)
        else:
            replica_hand = None
        validate_trace_rows(trace_rows)
        clip_indices = np.unique(trace_rows["clip_index"])
        expected_clip_index = env.reference_bank.clip_ids.index(args.clip)
        if not np.array_equal(clip_indices, np.asarray([expected_clip_index])):
            raise RuntimeError(
                "PPO26D_TRACE_CLIP_MISMATCH: "
                f"expected={expected_clip_index} captured={clip_indices.tolist()}"
            )
        actual_clip = env.reference_bank.clip_ids[expected_clip_index]
        write_progress(output, "trace_contract_validated")
        env.cfg.reset_reference_index = "uniform"
        if rsi_seeds:
            rsi_parallel_seeds = pad_parallel_seeds(rsi_seeds, count=evaluation_num_envs)
            env.cfg.evaluation_reset_reference_indices = tuple(
                rsi_start_indices(rsi_parallel_seeds, frame_count=env.reference_bank.frame_count)
            )
            rsi = run_parallel_episodes(
                env,
                trainer,
                capture=False,
                capture_all_replicas=False,
                capture_exact_fingertip_object_pair_force=False,
                capture_full_hand_object_pair_telemetry=False,
                expected_clip=args.clip,
                seeds=rsi_parallel_seeds,
            )[: args.rsi_replicas]
        else:
            rsi = []
        trace_path = output / (args.trace_name or f"ppo_{args.artifact_label}_trace_replica0.npz")
        environment_contract = env.contract_report()
        full_hand_manifest = (
            hand_body_manifest(tuple(HAND_COLLISION_BODY_NAMES), repo_root=REPO_ROOT)
            if args.capture_full_hand_object_pair_telemetry
            else None
        )
        np.savez_compressed(
            trace_path,
            object_pose=trace_rows["object_pose"].astype(np.float32),
            object_twist=trace_rows["object_twist"].astype(np.float32),
            hand_collision_body_pose=trace_rows["hand_collision_body_pose"].astype(np.float32),
            hand_collision_body_names=np.asarray(HAND_COLLISION_BODY_NAMES),
            wrist_pose=trace_rows["wrist_pose"].astype(np.float32),
            finger_q=trace_rows["finger_q"].astype(np.float32),
            finger_qdot=trace_rows["finger_qdot"].astype(np.float32),
            wrist_twist_world=trace_rows["wrist_twist_world"].astype(np.float32),
            virtual_wrist_q=trace_rows["virtual_wrist_q"].astype(np.float32),
            virtual_wrist_qdot=trace_rows["virtual_wrist_qdot"].astype(np.float32),
            virtual_wrist_target_q=trace_rows["virtual_wrist_target_q"].astype(np.float32),
            virtual_wrist_target_qdot=trace_rows["virtual_wrist_target_qdot"].astype(np.float32),
            object_axis_points=trace_rows["object_axis_points"].astype(np.float32),
            tracked_link_positions=trace_rows["tracked_link_positions"].astype(np.float32),
            contact_force_world=trace_rows["contact_force_world"].astype(np.float32),
            contact_pair_presence=trace_rows["contact_pair_presence"].astype(bool),
            actuator_effort=trace_rows["actuator_effort"].astype(np.float32),
            reason_code=trace_rows["reason_code"].astype(np.int64),
            terminated=trace_rows["terminated"].astype(bool),
            timed_out=trace_rows["timed_out"].astype(bool),
            action=trace_rows["action"].astype(np.float32),
            clip_index=trace_rows["clip_index"].astype(np.int64),
            wrist_residual=trace_rows["wrist_residual"].astype(np.float32),
            finger_residual=trace_rows["finger_residual"].astype(np.float32),
            wrist_target_pose=trace_rows["wrist_target"].astype(np.float32),
            finger_target_q=trace_rows["finger_target"].astype(np.float32),
            reward_total=trace_rows["reward_total"].astype(np.float32),
            reward_object=trace_rows["reward_object"].astype(np.float32),
            reward_link=trace_rows["reward_link"].astype(np.float32),
            reward_finger=trace_rows["reward_finger"].astype(np.float32),
            reward_wrist_translation=trace_rows["reward_wrist_translation"].astype(np.float32),
            reward_wrist_rotation=trace_rows["reward_wrist_rotation"].astype(np.float32),
            reward_smoothness=trace_rows["reward_smoothness"].astype(np.float32),
            reference_index=trace_rows["reference_index"].astype(np.int64),
            embedded_reference_object_pose=trace_rows["object_reference"].astype(np.float32),
            embedded_reference_wrist_pose=trace_rows["wrist_reference"].astype(np.float32),
            embedded_reference_finger_q=trace_rows["finger_reference"].astype(np.float32),
            embedded_reference_tracked_links=trace_rows["tracked_link_reference"].astype(
                np.float32
            ),
            trace_type=np.asarray("stage16d_ppo26d"),
            clip=np.asarray(actual_clip),
            requested_clip=np.asarray(args.clip),
            action_contract=np.asarray("26D_reference_residual"),
            checkpoint_path=np.asarray(str(checkpoint.resolve())),
            checkpoint_sha256=np.asarray(checkpoint_hash(checkpoint)),
            **{sample_counter_name: np.asarray(sample_counter)},
            selected_num_envs=np.asarray(selected),
            reference_hash=np.asarray(json.dumps(payload["reference_hash"], sort_keys=True)),
            reference_kinematics_version=np.asarray(reference_kinematics_version),
            simulation_data_capture_version=np.asarray(
                "Stage16DStrictPerFingerV4DiagnosticTraceV1"
                if strict_contact_diagnostic
                else (
                    "Stage16DStrictPerFingerV4SimulationDataV1"
                    if args.strict_per_finger_contact_reward_v4
                    else (
                        "Stage16DRewardV3SimulationDataV1"
                        if args.reference_gated_contact_reward_v3
                        else "Stage16DPPO26DTraceV2"
                    )
                )
            ),
            strict_contact_diagnostic=np.asarray(strict_contact_diagnostic),
            **(
                {}
                if not (
                    args.object_twist_reward_v2
                    or args.reference_gated_contact_reward_v3
                    or args.strict_per_finger_contact_reward_v4
                )
                else {
                    "object_twist_reference": trace_rows["object_twist_reference"].astype(
                        np.float32
                    ),
                    "object_linear_velocity_world": trace_rows["object_twist"][:, :3].astype(
                        np.float32
                    ),
                    "object_angular_velocity_world": trace_rows["object_twist"][:, 3:].astype(
                        np.float32
                    ),
                    "object_linear_velocity_reference_world": trace_rows["object_twist_reference"][
                        :, :3
                    ].astype(np.float32),
                    "object_angular_velocity_reference_world": trace_rows["object_twist_reference"][
                        :, 3:
                    ].astype(np.float32),
                    "delta_object_linear_velocity_world": (
                        trace_rows["object_twist"][:, :3]
                        - trace_rows["object_twist_reference"][:, :3]
                    ).astype(np.float32),
                    "delta_object_angular_velocity_world": (
                        trace_rows["object_twist"][:, 3:]
                        - trace_rows["object_twist_reference"][:, 3:]
                    ).astype(np.float32),
                    "reward_obj_vel": trace_rows["reward_obj_vel"].astype(np.float32),
                    "reward_obj_ang_vel": trace_rows["reward_obj_ang_vel"].astype(np.float32),
                    "error_obj_vel": trace_rows["error_obj_vel"].astype(np.float32),
                    "error_obj_ang_vel": trace_rows["error_obj_ang_vel"].astype(np.float32),
                }
            ),
            **(
                {}
                if not args.reference_gated_contact_reward_v3
                else {
                    "reference_contact_mask": trace_rows["reference_contact_mask"].astype(bool),
                    "actual_contact_mask": trace_rows["actual_contact_mask"].astype(bool),
                    "fingertip_object_force_magnitude": trace_rows[
                        "fingertip_object_force_magnitude"
                    ].astype(np.float32),
                    "contact_reward": trace_rows["contact_reward"].astype(np.float32),
                    "contact_force_scale": trace_rows["contact_force_scale"].astype(np.float32),
                    **(
                        {}
                        if args.capture_exact_fingertip_object_pair_force
                        else {
                            "fingertip_object_pair_force_world": trace_rows[
                                "fingertip_object_pair_force_world"
                            ].astype(np.float32)
                        }
                    ),
                }
            ),
            **(
                {}
                if not args.strict_per_finger_contact_reward_v4
                else {
                    "source_contact_mask": trace_rows["source_contact_mask"].astype(bool),
                    "tip_pair_presence": trace_rows["tip_pair_presence"].astype(bool),
                    "tip_pair_force_world": trace_rows["tip_pair_force_world"].astype(np.float32),
                    "tip_pair_force_norm": trace_rows["tip_pair_force_norm"].astype(np.float32),
                    "per_finger_contact_reward": trace_rows["per_finger_contact_reward"].astype(
                        np.float32
                    ),
                    "source_expected_finger_count": trace_rows[
                        "source_expected_finger_count"
                    ].astype(np.int64),
                    "source_satisfied_tip_count": trace_rows["source_satisfied_tip_count"].astype(
                        np.int64
                    ),
                    "source_tip_coverage_ratio": trace_rows["source_tip_coverage_ratio"].astype(
                        np.float32
                    ),
                    "r_contact_v4": trace_rows["r_contact_v4"].astype(np.float32),
                }
            ),
            **(
                {}
                if all_replica_trace is None or replica_hand is None
                else {
                    "replica_object_pose": all_replica_trace["object_pose"].astype(np.float32),
                    "replica_hand_collision_body_pose": replica_hand,
                    "replica_wrist_pose": all_replica_trace["wrist_pose"].astype(np.float32),
                    "replica_wrist_twist_world": all_replica_trace["wrist_twist_world"].astype(
                        np.float32
                    ),
                    "replica_virtual_wrist_q": all_replica_trace["virtual_wrist_q"].astype(
                        np.float32
                    ),
                    "replica_virtual_wrist_qdot": all_replica_trace["virtual_wrist_qdot"].astype(
                        np.float32
                    ),
                    "replica_virtual_wrist_target_q": all_replica_trace[
                        "virtual_wrist_target_q"
                    ].astype(np.float32),
                    "replica_virtual_wrist_target_qdot": all_replica_trace[
                        "virtual_wrist_target_qdot"
                    ].astype(np.float32),
                    "replica_finger_q": all_replica_trace["finger_q"].astype(np.float32),
                    "replica_finger_qdot": all_replica_trace["finger_qdot"].astype(np.float32),
                    "replica_object_axis_points": all_replica_trace["object_axis_points"].astype(
                        np.float32
                    ),
                    "replica_tracked_link_positions": all_replica_trace[
                        "tracked_link_positions"
                    ].astype(np.float32),
                    "replica_contact_force_world": all_replica_trace["contact_force_world"].astype(
                        np.float32
                    ),
                    "replica_contact_pair_presence": all_replica_trace[
                        "contact_pair_presence"
                    ].astype(bool),
                    "replica_object_twist": all_replica_trace["object_twist"].astype(np.float32),
                    "replica_actuator_effort": all_replica_trace["actuator_effort"].astype(
                        np.float32
                    ),
                    "replica_reason_code": all_replica_trace["reason_code"].astype(np.int64),
                    "replica_terminated": all_replica_trace["terminated"].astype(bool),
                    "replica_timed_out": all_replica_trace["timed_out"].astype(bool),
                    "replica_action": all_replica_trace["action"].astype(np.float32),
                    "replica_clip_index": all_replica_trace["clip_index"].astype(np.int64),
                    "replica_wrist_residual": all_replica_trace["wrist_residual"].astype(
                        np.float32
                    ),
                    "replica_finger_residual": all_replica_trace["finger_residual"].astype(
                        np.float32
                    ),
                    "replica_wrist_target_pose": all_replica_trace["wrist_target"].astype(
                        np.float32
                    ),
                    "replica_finger_target_q": all_replica_trace["finger_target"].astype(
                        np.float32
                    ),
                    "replica_reference_index": all_replica_trace["reference_index"].astype(
                        np.int64
                    ),
                    "replica_embedded_reference_object_pose": all_replica_trace[
                        "object_reference"
                    ].astype(np.float32),
                    "replica_embedded_reference_wrist_pose": all_replica_trace[
                        "wrist_reference"
                    ].astype(np.float32),
                    "replica_embedded_reference_finger_q": all_replica_trace[
                        "finger_reference"
                    ].astype(np.float32),
                    "replica_embedded_reference_tracked_links": all_replica_trace[
                        "tracked_link_reference"
                    ].astype(np.float32),
                    **(
                        {}
                        if "table_object_contact" not in all_replica_trace
                        else {
                            "replica_table_object_contact": all_replica_trace[
                                "table_object_contact"
                            ].astype(bool)
                        }
                    ),
                    "replica_reward_total": all_replica_trace["reward_total"].astype(np.float32),
                    "replica_reward_object": all_replica_trace["reward_object"].astype(np.float32),
                    "replica_reward_link": all_replica_trace["reward_link"].astype(np.float32),
                    "replica_reward_finger": all_replica_trace["reward_finger"].astype(np.float32),
                    "replica_reward_wrist_translation": all_replica_trace[
                        "reward_wrist_translation"
                    ].astype(np.float32),
                    "replica_reward_wrist_rotation": all_replica_trace[
                        "reward_wrist_rotation"
                    ].astype(np.float32),
                    "replica_reward_smoothness": all_replica_trace["reward_smoothness"].astype(
                        np.float32
                    ),
                    **(
                        {}
                        if not args.reference_gated_contact_reward_v3
                        else {
                            "replica_reference_contact_mask": all_replica_trace[
                                "reference_contact_mask"
                            ].astype(bool),
                            "replica_actual_contact_mask": all_replica_trace[
                                "actual_contact_mask"
                            ].astype(bool),
                            "replica_fingertip_object_force_magnitude": all_replica_trace[
                                "fingertip_object_force_magnitude"
                            ].astype(np.float32),
                            "replica_contact_reward": all_replica_trace["contact_reward"].astype(
                                np.float32
                            ),
                            "replica_contact_force_scale": all_replica_trace[
                                "contact_force_scale"
                            ].astype(np.float32),
                            "replica_object_twist_reference": all_replica_trace[
                                "object_twist_reference"
                            ].astype(np.float32),
                            "replica_reward_obj_vel": all_replica_trace["reward_obj_vel"].astype(
                                np.float32
                            ),
                            "replica_reward_obj_ang_vel": all_replica_trace[
                                "reward_obj_ang_vel"
                            ].astype(np.float32),
                            "replica_error_obj_vel": all_replica_trace["error_obj_vel"].astype(
                                np.float32
                            ),
                            "replica_error_obj_ang_vel": all_replica_trace[
                                "error_obj_ang_vel"
                            ].astype(np.float32),
                        }
                    ),
                    **(
                        {}
                        if not args.strict_per_finger_contact_reward_v4
                        else {
                            "replica_source_contact_mask": all_replica_trace[
                                "source_contact_mask"
                            ].astype(bool),
                            "replica_tip_pair_presence": all_replica_trace[
                                "tip_pair_presence"
                            ].astype(bool),
                            "replica_tip_pair_force_world": all_replica_trace[
                                "tip_pair_force_world"
                            ].astype(np.float32),
                            "replica_tip_pair_force_norm": all_replica_trace[
                                "tip_pair_force_norm"
                            ].astype(np.float32),
                            "replica_per_finger_contact_reward": all_replica_trace[
                                "per_finger_contact_reward"
                            ].astype(np.float32),
                            "replica_source_expected_finger_count": all_replica_trace[
                                "source_expected_finger_count"
                            ].astype(np.int64),
                            "replica_source_satisfied_tip_count": all_replica_trace[
                                "source_satisfied_tip_count"
                            ].astype(np.int64),
                            "replica_source_tip_coverage_ratio": all_replica_trace[
                                "source_tip_coverage_ratio"
                            ].astype(np.float32),
                            "replica_r_contact_v4": all_replica_trace["r_contact_v4"].astype(
                                np.float32
                            ),
                            "replica_object_twist_reference": all_replica_trace[
                                "object_twist_reference"
                            ].astype(np.float32),
                        }
                    ),
                    **(
                        {}
                        if not args.capture_exact_fingertip_object_pair_force
                        else {
                            "replica_fingertip_object_pair_force_world": all_replica_trace[
                                "fingertip_object_pair_force_world"
                            ].astype(np.float32),
                            "replica_fingertip_object_pair_force_valid": all_replica_trace[
                                "fingertip_object_pair_force_valid"
                            ].astype(bool),
                        }
                    ),
                    **(
                        {}
                        if not args.capture_full_hand_object_pair_telemetry
                        else {
                            "replica_hand_object_pair_force_world": all_replica_trace[
                                "hand_object_pair_force_world"
                            ].astype(np.float32),
                            "replica_hand_object_pair_presence": all_replica_trace[
                                "hand_object_pair_presence"
                            ].astype(bool),
                            "replica_hand_object_pair_force_valid": all_replica_trace[
                                "hand_object_pair_force_valid"
                            ].astype(bool),
                        }
                    ),
                }
            ),
            **(
                {}
                if "table_object_contact" not in trace_rows
                else {"table_object_contact": trace_rows["table_object_contact"].astype(bool)}
            ),
            **(
                {}
                if not args.capture_exact_fingertip_object_pair_force
                else {
                    "fingertip_object_pair_force_world": trace_rows[
                        "fingertip_object_pair_force_world"
                    ].astype(np.float32),
                    "fingertip_object_pair_force_valid": trace_rows[
                        "fingertip_object_pair_force_valid"
                    ].astype(bool),
                    "fingertip_link_names": np.asarray(EVALUATION_FINGERTIP_LINKS),
                    "fingertip_force_sensor_indices": np.asarray(
                        fingertip_force_indices(HAND_COLLISION_BODY_NAMES), dtype=np.int64
                    ),
                    "pair_force_frame": np.asarray("world"),
                    "pair_force_units": np.asarray("N"),
                    "pair_force_semantics": np.asarray(
                        "force on active object from named filtered hand collision body"
                    ),
                    "pair_force_capture": np.asarray(
                        "STRICT_V4_CONTACT_DIAGNOSTIC"
                        if strict_contact_diagnostic
                        else (
                            "STRICT_V4_PPO_EVALUATION"
                            if args.strict_per_finger_contact_reward_v4
                            else (
                                "V3_PPO_EVALUATION"
                                if args.reference_gated_contact_reward_v3
                                else "V1_PAIRFORCE_REEXPORT_DIAGNOSTIC"
                            )
                        )
                    ),
                }
            ),
            **(
                {}
                if not args.capture_full_hand_object_pair_telemetry or full_hand_manifest is None
                else {
                    "hand_object_pair_force_world": trace_rows[
                        "hand_object_pair_force_world"
                    ].astype(np.float32),
                    "hand_object_pair_presence": trace_rows["hand_object_pair_presence"].astype(
                        bool
                    ),
                    "hand_object_pair_force_valid": trace_rows[
                        "hand_object_pair_force_valid"
                    ].astype(bool),
                    "hand_body_names": np.asarray(full_hand_manifest["hand_body_names"]),
                    "hand_body_indices": np.asarray(
                        full_hand_manifest["hand_body_indices"], dtype=np.int64
                    ),
                    "hand_body_groups": np.asarray(full_hand_manifest["hand_body_groups"]),
                    "hand_collision_shape_mapping": np.asarray(
                        json.dumps(full_hand_manifest["collision_shape_mapping"], sort_keys=True)
                    ),
                    "hand_palm_mapping": np.asarray(
                        json.dumps(full_hand_manifest["palm_mapping"], sort_keys=True)
                    ),
                    "full_hand_pair_force_frame": np.asarray(full_hand_manifest["force_frame"]),
                    "full_hand_pair_force_units": np.asarray(full_hand_manifest["force_units"]),
                    "full_hand_pair_force_semantics": np.asarray(
                        full_hand_manifest["force_semantics"]
                    ),
                }
            ),
        )
        qualification = {
            "schema_version": "Stage16DPPO26DEvaluationV2",
            "status": "PPO_DEVELOPMENT_EVALUATION_COMPLETE",
            "artifact_label": args.artifact_label,
            "contact_mode": None if contact_mode is None else contact_mode.value,
            "full_trajectory_table": bool(args.full_trajectory_table),
            "curriculum_stage": args.curriculum_stage if args.full_trajectory_table else None,
            "hand_gravity_mode": args.hand_gravity_mode,
            "reference_kinematics_version": reference_kinematics_version,
            "reward_contract": environment_contract["ppo26d"]["reward"],
            "physics_contract": environment_contract,
            "physics_contract_sha256": canonical_json_hash(environment_contract),
            "clip": actual_clip,
            "requested_clip": args.clip,
            "clip_index": expected_clip_index,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "checkpoint_schema_version": payload["schema_version"],
            "strict_contact_diagnostic_only": strict_contact_diagnostic,
            "strict_contact_diagnostic_policy_effect": (
                "none; reward is post-action and the policy checkpoint is read-only"
                if strict_contact_diagnostic
                else None
            ),
            sample_counter_name: sample_counter,
            "selected_num_envs": selected,
            "frame_zero": [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"trace", "all_replica_trace"}
                }
                for row in frame_zero
            ],
            "rsi": rsi,
            "frame_zero_summary": summarize_episodes(frame_zero),
            "rsi_summary": summarize_episodes(rsi) if rsi else None,
            "seed_set": {
                "manifest": (
                    str(args.seed_manifest.resolve()) if args.seed_manifest else "legacy_default"
                ),
                "identifier": args.seed_set if args.seed_manifest else "legacy_default",
                "frame_zero": frame_zero_seeds,
                "rsi": rsi_seeds,
                "rsi_start_indices": rsi_start_indices(
                    rsi_seeds, frame_count=env.reference_bank.frame_count
                ),
                "application": (
                    "frame-zero is deterministic without evaluation reset noise; RSI start indices "
                    "are derived independently from each named seed and injected only at reset"
                ),
            },
            "trace": str(trace_path.resolve()),
            "trace_capture": "post_physics_gpu_buffer_then_single_host_export",
            "all_frame_zero_replica_trace": all_replica_trace is not None,
            "hand_collision_body_pose": "offline_fk_from_captured_physical_wrist_and_finger_state",
            "full_hand_object_pair_telemetry": (
                None
                if full_hand_manifest is None
                else {
                    **full_hand_manifest,
                    "capture": "post_physics_gpu_buffer_then_single_host_export",
                    "reward_or_policy_effect": "none",
                }
            ),
            "self_collision": "ENABLED; post-PPO diagnostic pending formal audit",
            "inter_finger_penetration": "POST_PPO_GEOMETRY_DIAGNOSTIC_NOT_RUN",
        }
        write_json(output / f"{args.artifact_label}_evaluation.json", qualification)
        write_json(output / f"ppo_{args.artifact_label}_evaluation.json", qualification)
        write_progress(output, "evaluation_artifacts_written")
        print(json.dumps({"status": qualification["status"], "trace": str(trace_path.resolve())}))
        return 0
    except BaseException as error:
        # Isaac's close path may terminate the process before Python can render
        # an uncaught traceback.  Persist the technical receipt first so formal
        # recovery can distinguish an app-startup failure from a bad result.
        write_json(
            output / "evaluation_failure.json",
            {
                "schema_version": "Stage16CausalPhysicalEvaluationFailureV1",
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
