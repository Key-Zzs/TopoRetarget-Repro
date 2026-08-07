#!/usr/bin/env python3
"""Reload, evaluate, and export a Stage 16-D.5 PPO-26D L0 checkpoint."""

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

from toporetarget.rl.ppo.checkpoint import load_checkpoint  # noqa: E402
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), default="hocap_170650")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selected-capacity", type=Path)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_summary(env: Any) -> tuple[float, np.ndarray]:
    sensor = env._object_contact_sensors["Object170650"]
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return 0.0, np.zeros(21, dtype=bool)
    values = matrix[0, 0]
    norm = torch.linalg.vector_norm(values, dim=-1)
    return float(norm.sum().detach().cpu()), (norm > 1.0e-4).detach().cpu().numpy()


def model_from_checkpoint(path: Path, device: str) -> tuple[PPO26DTrainer, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=device)
    if payload.get("schema_version") != "Stage16DPPO26DCheckpointV1":
        raise ValueError("CHECKPOINT_ROUNDTRIP_FAILURE: unexpected checkpoint schema")
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.cumulative_samples = int(payload["cumulative_samples"])
    trainer.trainer.freeze_observation_normalizer()
    return trainer, payload


def run_episode(env: Any, trainer: PPO26DTrainer, *, capture: bool) -> dict[str, Any]:
    observation, _ = env.reset()
    total_reward = 0.0
    contact_seen = False
    terminal_contact = False
    final_extras: dict[str, Any] = {}
    rows: dict[str, list[np.ndarray]] = {
        "wrist_pose": [],
        "wrist_twist": [],
        "virtual_wrist_q": [],
        "virtual_wrist_qdot": [],
        "finger_q": [],
        "finger_qdot": [],
        "object_pose": [],
        "object_twist": [],
        "hand_collision_body_pose": [],
        "contact_force_world": [],
        "contact_pair_presence": [],
        "actuator_effort": [],
        "action": [],
        "action_mean": [],
        "action_std": [],
        "wrist_target": [],
        "finger_target": [],
        "reference_index": [],
        "wrist_reference": [],
        "finger_reference": [],
        "object_reference": [],
        "tracked_links_reference": [],
        "reward_total": [],
        "reward_object": [],
        "reward_links": [],
        "reward_finger": [],
        "reward_wrist": [],
        "reward_wrist_translation": [],
        "reward_wrist_rotation": [],
        "reward_smoothness": [],
        "reward_smoothness_wrist": [],
        "reward_smoothness_finger": [],
        "reason_code": [],
        "terminated": [],
        "timed_out": [],
        "inter_finger_penetration_m": [],
    }
    object_tracking_errors: list[float] = []
    wrist_position_errors: list[float] = []
    wrist_rotation_errors: list[float] = []
    finger_errors: list[float] = []
    inter_finger_values: list[float] = []
    for _ in range(env.reference_bank.frame_count):
        state = env._state()
        with torch.no_grad():
            distribution = trainer.trainer.distribution(observation["policy"])
            action = distribution.mean.clamp(-1.0, 1.0)
        index = int(env._reference_index[0].item())
        object_reference_position = env.reference_bank.object_pose_translation_world_ref[1, index]
        wrist_reference_position = env.reference_bank.wrist_pose_translation_world_ref[1, index]
        wrist_reference_quaternion = env.reference_bank.wrist_pose_quaternion_world_ref_wxyz[
            1, index
        ]
        finger_reference = env.reference_bank.q_finger_ref[1, index]
        object_tracking_errors.append(
            float(
                torch.linalg.vector_norm(
                    state["object_position_scene"][0] - object_reference_position
                )
            )
        )
        wrist_position_errors.append(
            float(
                torch.linalg.vector_norm(
                    state["wrist_position_scene"][0] - wrist_reference_position
                )
            )
        )
        wrist_rotation_errors.append(
            float(
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
        finger_errors.append(float(torch.mean(torch.abs(state["finger_q"][0] - finger_reference))))
        body_ids = torch.as_tensor(env._hand_collision_body_ids, device=env.device)
        body_pose = torch.cat(
            (
                env._robot.data.body_link_pos_w[0, body_ids],
                env._robot.data.body_link_quat_w[0, body_ids],
            ),
            dim=-1,
        )
        inter_finger = env._inter_finger_metric.evaluate(body_pose.unsqueeze(0))[
            "maximum_penetration_m"
        ]
        inter_finger_values.append(float(inter_finger[0].detach().cpu()))
        if capture:
            rows["wrist_pose"].append(
                torch.cat((state["wrist_position_scene"][0], state["wrist_quaternion_wxyz"][0]))
                .detach()
                .cpu()
                .numpy()
            )
            rows["wrist_twist"].append(state["wrist_twist_world"][0].detach().cpu().numpy())
            rows["virtual_wrist_q"].append(
                env._robot.data.joint_pos[0, env._virtual_wrist_joint_ids].detach().cpu().numpy()
            )
            rows["virtual_wrist_qdot"].append(
                env._robot.data.joint_vel[0, env._virtual_wrist_joint_ids].detach().cpu().numpy()
            )
            rows["finger_q"].append(state["finger_q"][0].detach().cpu().numpy())
            rows["finger_qdot"].append(state["finger_qdot"][0].detach().cpu().numpy())
            rows["object_pose"].append(
                torch.cat((state["object_position_scene"][0], state["object_quaternion_wxyz"][0]))
                .detach()
                .cpu()
                .numpy()
            )
            rows["object_twist"].append(state["object_twist_world"][0].detach().cpu().numpy())
            rows["hand_collision_body_pose"].append(body_pose.detach().cpu().numpy())
            rows["action"].append(action[0].detach().cpu().numpy())
            rows["action_mean"].append(distribution.mean[0].detach().cpu().numpy())
            rows["action_std"].append(distribution.std[0].detach().cpu().numpy())
            rows["reference_index"].append(np.asarray(index, dtype=np.int64))
            rows["wrist_reference"].append(
                np.concatenate(
                    (
                        env.reference_bank.wrist_pose_translation_world_ref[1, index]
                        .detach()
                        .cpu()
                        .numpy(),
                        env.reference_bank.wrist_pose_quaternion_world_ref_wxyz[1, index]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            )
            rows["finger_reference"].append(
                env.reference_bank.q_finger_ref[1, index].detach().cpu().numpy()
            )
            rows["object_reference"].append(
                np.concatenate(
                    (
                        env.reference_bank.object_pose_translation_world_ref[1, index]
                        .detach()
                        .cpu()
                        .numpy(),
                        env.reference_bank.object_pose_quaternion_world_ref_wxyz[1, index]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            )
            rows["tracked_links_reference"].append(
                env.reference_bank.tracked_link_positions_world_ref[1, index].detach().cpu().numpy()
            )
        observation, reward, terminated, timed_out, extras = env.step(action)
        total_reward += float(reward[0].detach().cpu())
        force, pairs = contact_summary(env)
        contact_seen |= bool(pairs.any())
        terminal_contact = bool(pairs.any())
        final_extras = extras.get("ppo26d", {})
        if capture:
            torque = env._robot.data.applied_torque[0].detach().cpu().numpy()
            if torque.shape != (26,):
                torque = np.zeros(26, dtype=np.float32)
            rows["contact_force_world"].append(np.asarray([force, 0.0, 0.0], dtype=np.float32))
            rows["contact_pair_presence"].append(pairs)
            rows["actuator_effort"].append(torque)
            rows["wrist_target"].append(
                torch.cat((env._wrist_target_position[0], env._wrist_target_quaternion[0]))
                .detach()
                .cpu()
                .numpy()
            )
            rows["finger_target"].append(env._joint_target_isaac[0].detach().cpu().numpy())
            terms = env._last_reward_terms
            rows["reward_total"].append(terms["total"][0].detach().cpu().numpy())
            rows["reward_object"].append(terms["r_object"][0].detach().cpu().numpy())
            rows["reward_links"].append(terms["r_link"][0].detach().cpu().numpy())
            rows["reward_finger"].append(terms["r_finger"][0].detach().cpu().numpy())
            rows["reward_wrist"].append(terms["r_wrist"][0].detach().cpu().numpy())
            rows["reward_wrist_translation"].append(
                terms["r_wrist_translation"][0].detach().cpu().numpy()
            )
            rows["reward_wrist_rotation"].append(
                terms["r_wrist_rotation"][0].detach().cpu().numpy()
            )
            rows["reward_smoothness"].append(terms["smoothness"][0].detach().cpu().numpy())
            rows["reward_smoothness_wrist"].append(
                terms["smoothness_wrist"][0].detach().cpu().numpy()
            )
            rows["reward_smoothness_finger"].append(
                terms["smoothness_finger"][0].detach().cpu().numpy()
            )
            rows["reason_code"].append(
                final_extras["primary_reason_code"][0].detach().cpu().numpy()
            )
            rows["terminated"].append(terminated[0].detach().cpu().numpy())
            rows["timed_out"].append(timed_out[0].detach().cpu().numpy())
            rows["inter_finger_penetration_m"].append(inter_finger[0].detach().cpu().numpy())
        if bool(terminated[0] | timed_out[0]):
            break
    linear_speed = float(final_extras.get("object_linear_speed_mps", torch.zeros(1))[0].cpu())
    angular_speed = float(final_extras.get("object_angular_speed_radps", torch.zeros(1))[0].cpu())
    return {
        "steps": len(object_tracking_errors),
        "reached_final_reference": bool(
            len(object_tracking_errors) == env.reference_bank.frame_count
        ),
        "total_reward": total_reward,
        "contact": contact_seen,
        "terminal_contact": terminal_contact,
        "terminal_object_linear_speed_mps": linear_speed,
        "terminal_object_angular_speed_radps": angular_speed,
        "termination_reason": int(final_extras.get("primary_reason_code", torch.zeros(1))[0].cpu()),
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
        "inter_finger_penetration_m": {
            "mean": float(np.mean(inter_finger_values)),
            "max": float(np.max(inter_finger_values)),
        },
        "terminal_stability": "POST_PPO_QUALIFICATION_NOT_RUN",
        "trace": rows if capture else None,
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )
    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        HAND_COLLISION_BODY_NAMES,
    )

    root = args.output_root.resolve()
    output = root / args.clip
    checkpoint = (
        args.checkpoint or output / f"stage16d_ppo26d_{args.clip.removeprefix('hocap_')}_l0.pt"
    )
    payload = load_checkpoint(checkpoint, map_location="cpu")
    selected = int(payload["selected_num_envs"])
    app = AppLauncher(headless=True).app
    env = None
    try:
        cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo26d_cfg.configure_stage16d_ppo26d(cfg, num_envs=1, clip=args.clip, rsi=False)
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        trainer, payload = model_from_checkpoint(checkpoint, str(env.device))
        frame_zero = [run_episode(env, trainer, capture=index == 0) for index in range(4)]
        trace_rows = frame_zero[0]["trace"]
        env.cfg.reset_reference_index = "uniform"
        rsi = [run_episode(env, trainer, capture=False) for _ in range(20)]
        assert trace_rows is not None
        trace_path = output / "ppo_l0_eval_trace_replica0.npz"
        reference_pose = np.asarray(trace_rows["object_reference"], dtype=np.float32)
        np.savez_compressed(
            trace_path,
            wrist_pose=np.asarray(trace_rows["wrist_pose"], dtype=np.float32),
            wrist_twist=np.asarray(trace_rows["wrist_twist"], dtype=np.float32),
            virtual_wrist_q=np.asarray(trace_rows["virtual_wrist_q"], dtype=np.float32),
            virtual_wrist_qdot=np.asarray(trace_rows["virtual_wrist_qdot"], dtype=np.float32),
            finger_q=np.asarray(trace_rows["finger_q"], dtype=np.float32),
            finger_qdot=np.asarray(trace_rows["finger_qdot"], dtype=np.float32),
            object_pose=np.asarray(trace_rows["object_pose"], dtype=np.float32),
            object_twist=np.asarray(trace_rows["object_twist"], dtype=np.float32),
            hand_collision_body_pose=np.asarray(
                trace_rows["hand_collision_body_pose"], dtype=np.float32
            ),
            hand_collision_body_names=np.asarray(HAND_COLLISION_BODY_NAMES),
            contact_force_world=np.asarray(trace_rows["contact_force_world"], dtype=np.float32),
            contact_pair_presence=np.asarray(trace_rows["contact_pair_presence"], dtype=bool),
            actuator_effort=np.asarray(trace_rows["actuator_effort"], dtype=np.float32),
            reason_code=np.asarray(trace_rows["reason_code"], dtype=np.int64),
            terminated=np.asarray(trace_rows["terminated"], dtype=bool),
            timed_out=np.asarray(trace_rows["timed_out"], dtype=bool),
            inter_finger_penetration_m=np.asarray(
                trace_rows["inter_finger_penetration_m"], dtype=np.float32
            ),
            action=np.asarray(trace_rows["action"], dtype=np.float32),
            action_mean=np.asarray(trace_rows["action_mean"], dtype=np.float32),
            action_std=np.asarray(trace_rows["action_std"], dtype=np.float32),
            wrist_target=np.asarray(trace_rows["wrist_target"], dtype=np.float32),
            finger_target=np.asarray(trace_rows["finger_target"], dtype=np.float32),
            reference_index=np.asarray(trace_rows["reference_index"], dtype=np.int64),
            wrist_reference=np.asarray(trace_rows["wrist_reference"], dtype=np.float32),
            finger_reference=np.asarray(trace_rows["finger_reference"], dtype=np.float32),
            object_reference=np.asarray(trace_rows["object_reference"], dtype=np.float32),
            tracked_links_reference=np.asarray(
                trace_rows["tracked_links_reference"], dtype=np.float32
            ),
            embedded_reference_object_pose=reference_pose.astype(np.float32),
            reward_total=np.asarray(trace_rows["reward_total"], dtype=np.float32),
            reward_object=np.asarray(trace_rows["reward_object"], dtype=np.float32),
            reward_links=np.asarray(trace_rows["reward_links"], dtype=np.float32),
            reward_finger=np.asarray(trace_rows["reward_finger"], dtype=np.float32),
            reward_wrist=np.asarray(trace_rows["reward_wrist"], dtype=np.float32),
            reward_wrist_translation=np.asarray(
                trace_rows["reward_wrist_translation"], dtype=np.float32
            ),
            reward_wrist_rotation=np.asarray(trace_rows["reward_wrist_rotation"], dtype=np.float32),
            reward_smoothness=np.asarray(trace_rows["reward_smoothness"], dtype=np.float32),
            reward_smoothness_wrist=np.asarray(
                trace_rows["reward_smoothness_wrist"], dtype=np.float32
            ),
            reward_smoothness_finger=np.asarray(
                trace_rows["reward_smoothness_finger"], dtype=np.float32
            ),
            trace_type=np.asarray("stage16d_ppo26d"),
            clip=np.asarray(args.clip),
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
            "clip": args.clip,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash(checkpoint),
            "cumulative_training_samples": int(payload["cumulative_samples"]),
            "selected_num_envs": selected,
            "frame_zero": [
                {key: value for key, value in row.items() if key != "trace"} for row in frame_zero
            ],
            "rsi": rsi,
            "trace": str(trace_path.resolve()),
            "self_collision": "ENABLED; post-PPO diagnostic pending formal audit",
            "inter_finger_penetration": "POST_PPO_GEOMETRY_DIAGNOSTIC_NOT_RUN",
        }
        write_json(output / "l0_evaluation.json", qualification)
        write_json(output / "ppo_l0_eval_qualification.json", qualification)
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
