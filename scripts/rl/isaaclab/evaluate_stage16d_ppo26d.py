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


def _device_trace_to_numpy(trace: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Make the sole host transfer after a completed simulator-thread capture."""

    torch.cuda.synchronize()
    return {name: value[:, 0].detach().cpu().numpy().copy() for name, value in trace.items()}


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
    wrist_position_errors: list[float] = []
    wrist_rotation_errors: list[float] = []
    finger_errors: list[float] = []
    for _ in range(env.reference_bank.frame_count):
        state = env._state()
        index = int(env._reference_index[0].item())
        with torch.no_grad():
            distribution = trainer.trainer.distribution(observation["policy"])
            action = distribution.mean
        object_reference_position = env.reference_bank.object_pose_translation_world_ref[
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
        if bool(terminated[0] | timed_out[0]):
            break
    trace = _device_trace_to_numpy(env.finish_trace_capture()) if capture else None
    linear_speed = _to_float(final_extras["object_linear_speed_mps"][0])
    angular_speed = _to_float(final_extras["object_angular_speed_radps"][0])
    termination_reason = int(final_extras["primary_reason_code"][0].detach().cpu())
    steps = len(object_tracking_errors)
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
        "terminal_contact": terminal_contact,
        "terminal_object_linear_speed_mps": linear_speed,
        "terminal_object_angular_speed_radps": angular_speed,
        "termination_reason": termination_reason,
        "object_tracking_error_m": {
            "mean": float(np.mean(object_tracking_errors)),
            "final": float(object_tracking_errors[-1]),
        },
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

        cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo26d_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=1, clip=args.clip, rsi=False, critical_dr=False
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        trainer, payload = model_from_checkpoint(
            checkpoint, str(env.device), expected_clip=args.clip
        )
        selected = int(payload["selected_num_envs"])
        write_progress(output, "checkpoint_loaded")
        write_progress(output, "frame_zero_evaluation_started")
        frame_zero = [
            run_episode(env, trainer, capture=index == 0, expected_clip=args.clip)
            for index in range(args.frame_zero_replicas)
        ]
        trace_rows = frame_zero[0]["trace"]
        assert trace_rows is not None
        if tuple(HAND_COLLISION_BODY_NAMES) != FK_HAND_COLLISION_BODY_NAMES:
            raise RuntimeError("PPO26D hand collision-body order disagreement")
        trace_rows["hand_collision_body_pose"] = reconstruct_hand_collision_body_pose(
            trace_rows["wrist_pose"], trace_rows["finger_q"], repo_root=REPO_ROOT
        ).astype(np.float32)
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
        rsi = [
            run_episode(env, trainer, capture=False, expected_clip=args.clip)
            for _ in range(args.rsi_replicas)
        ]
        trace_path = output / "ppo_l0_eval_trace_replica0.npz"
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
        )
        qualification = {
            "schema_version": "Stage16DPPO26DL0EvaluationV1",
            "status": "PPO_L0_COMPLETE_NOT_YET_QUALIFIED",
            "clip": actual_clip,
            "requested_clip": args.clip,
            "clip_index": expected_clip_index,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "cumulative_training_samples": int(payload["cumulative_samples"]),
            "selected_num_envs": selected,
            "frame_zero": [
                {key: value for key, value in row.items() if key != "trace"} for row in frame_zero
            ],
            "rsi": rsi,
            "trace": str(trace_path.resolve()),
            "trace_capture": "post_physics_gpu_buffer_then_single_host_export",
            "hand_collision_body_pose": "offline_fk_from_captured_physical_wrist_and_finger_state",
            "self_collision": "ENABLED; post-PPO diagnostic pending formal audit",
            "inter_finger_penetration": "POST_PPO_GEOMETRY_DIAGNOSTIC_NOT_RUN",
        }
        write_json(output / "l0_evaluation.json", qualification)
        write_json(output / "ppo_l0_eval_qualification.json", qualification)
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
