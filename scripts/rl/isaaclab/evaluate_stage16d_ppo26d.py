#!/usr/bin/env python3
"""Reload, evaluate, and export a Stage 16-D.5 PPO-26D L0 checkpoint."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES as FK_HAND_COLLISION_BODY_NAMES,
)
from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    reconstruct_hand_collision_body_pose,
)
from toporetarget.rl.ppo.checkpoint import load_checkpoint
from toporetarget.rl.ppo.ppo26d_continuation import summarize_episodes
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), default="hocap_170650")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
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
    if payload.get("schema_version") != "Stage16DPPO26DCheckpointV1":
        raise ValueError("CHECKPOINT_ROUNDTRIP_FAILURE: unexpected checkpoint schema")
    if payload.get("clip") != expected_clip:
        raise ValueError(
            "PPO26D_CHECKPOINT_CLIP_MISMATCH: "
            f"checkpoint={payload.get('clip')!r} requested={expected_clip!r}"
        )
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.cumulative_samples = int(payload["cumulative_samples"])
    trainer.trainer.freeze_observation_normalizer()
    return trainer, payload


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


def _initial_trace_snapshot(env: Any) -> dict[str, np.ndarray]:
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
    values = {
        "object_pose": torch.cat(
            (state["object_position_scene"], state["object_quaternion_wxyz"]), dim=-1
        ),
        "object_twist": state["object_twist_world"],
        "wrist_pose": torch.cat(
            (state["wrist_position_scene"], state["wrist_quaternion_wxyz"]), dim=-1
        ),
        "finger_q": state["finger_q"],
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
    env: Any, trainer: PPO26DTrainer, *, capture: bool, expected_clip: str
) -> dict[str, Any]:
    """Run one physical rollout without reading collision articulation tensors.

    ``start_trace_capture`` records post-physics wrist/finger/object state in
    the environment's reward callback.  Collision-body poses are reconstructed
    after the rollout from that physical state, avoiding an unsafe articulation
    tensor read from the Python evaluation loop.
    """

    observation, _ = env.reset()
    active_clip_indices = sorted(set(env._clip_index.detach().cpu().tolist()))
    expected_clip_index = env.reference_bank.clip_ids.index(expected_clip)
    if active_clip_indices != [expected_clip_index]:
        raise RuntimeError(
            "PPO26D_FIXED_CLIP_MISMATCH_AFTER_RESET: "
            f"expected={expected_clip_index} active={active_clip_indices}"
        )
    if capture:
        env.start_trace_capture(capacity=env.reference_bank.frame_count)
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
    trace = _device_trace_to_numpy(env.finish_trace_capture()) if capture else None
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
    initial_trace = _initial_trace_snapshot(env) if capture else None
    if capture:
        env.start_trace_capture(capacity=env.reference_bank.frame_count)
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
    if args.frame_zero_replicas <= 0 or args.rsi_replicas < 0:
        raise ValueError("--frame-zero-replicas must be positive and --rsi-replicas non-negative")
    if not args.artifact_label.replace("_", "").isalnum():
        raise ValueError("--artifact-label must be alphanumeric/underscore")
    frame_zero_seeds, rsi_seeds = evaluation_seeds(args)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    root = args.output_root.resolve()
    output = root / args.clip
    checkpoint = (
        args.checkpoint or output / f"stage16d_ppo26d_{args.clip.removeprefix('hocap_')}_l0.pt"
    )
    app = AppLauncher(headless=True).app
    env = None
    try:
        # Import optional Isaac modules only after this process owns the app.
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            HAND_COLLISION_BODY_NAMES,
        )

        evaluation_num_envs = max(args.frame_zero_replicas, args.rsi_replicas)
        cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo26d_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=evaluation_num_envs, clip=args.clip, rsi=False, critical_dr=False
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        trainer, payload = model_from_checkpoint(
            checkpoint, str(env.device), expected_clip=args.clip
        )
        selected = int(payload["selected_num_envs"])
        write_progress(output, "checkpoint_loaded")
        write_progress(output, "frame_zero_evaluation_started")
        frame_zero = run_parallel_episodes(
            env,
            trainer,
            capture=True,
            capture_all_replicas=args.capture_all_frame_zero_replicas,
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
                expected_clip=args.clip,
                seeds=rsi_parallel_seeds,
            )[: args.rsi_replicas]
        else:
            rsi = []
        trace_path = output / f"ppo_{args.artifact_label}_trace_replica0.npz"
        np.savez_compressed(
            trace_path,
            object_pose=trace_rows["object_pose"].astype(np.float32),
            object_twist=trace_rows["object_twist"].astype(np.float32),
            hand_collision_body_pose=trace_rows["hand_collision_body_pose"].astype(np.float32),
            hand_collision_body_names=np.asarray(HAND_COLLISION_BODY_NAMES),
            wrist_pose=trace_rows["wrist_pose"].astype(np.float32),
            finger_q=trace_rows["finger_q"].astype(np.float32),
            contact_force_world=trace_rows["contact_force_world"].astype(np.float32),
            contact_pair_presence=trace_rows["contact_pair_presence"].astype(bool),
            actuator_effort=trace_rows["actuator_effort"].astype(np.float32),
            reason_code=trace_rows["reason_code"].astype(np.int64),
            terminated=trace_rows["terminated"].astype(bool),
            timed_out=trace_rows["timed_out"].astype(bool),
            action=trace_rows["action"].astype(np.float32),
            clip_index=trace_rows["clip_index"].astype(np.int64),
            wrist_residual=trace_rows["wrist_residual"].astype(np.float32),
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
            cumulative_training_samples=np.asarray(int(payload["cumulative_samples"])),
            selected_num_envs=np.asarray(selected),
            reference_hash=np.asarray(json.dumps(payload["reference_hash"], sort_keys=True)),
            **(
                {}
                if all_replica_trace is None or replica_hand is None
                else {
                    "replica_object_pose": all_replica_trace["object_pose"].astype(np.float32),
                    "replica_hand_collision_body_pose": replica_hand,
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
                }
            ),
        )
        qualification = {
            "schema_version": "Stage16DPPO26DEvaluationV2",
            "status": "PPO_DEVELOPMENT_EVALUATION_COMPLETE",
            "artifact_label": args.artifact_label,
            "clip": actual_clip,
            "requested_clip": args.clip,
            "clip_index": expected_clip_index,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "cumulative_training_samples": int(payload["cumulative_samples"]),
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
            "self_collision": "ENABLED; post-PPO diagnostic pending formal audit",
            "inter_finger_penetration": "POST_PPO_GEOMETRY_DIAGNOSTIC_NOT_RUN",
        }
        write_json(output / f"{args.artifact_label}_evaluation.json", qualification)
        write_json(output / f"ppo_{args.artifact_label}_evaluation.json", qualification)
        write_progress(output, "evaluation_artifacts_written")
        print(json.dumps({"status": qualification["status"], "trace": str(trace_path.resolve())}))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
