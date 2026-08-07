#!/usr/bin/env python3
"""Run Gate A for the Stage 16-D.5 PPO-26D environment and optimizer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selected-capacity", type=Path)
    parser.add_argument("--num-envs", type=int)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_num_envs(args: argparse.Namespace) -> int:
    if args.num_envs is not None:
        return args.num_envs
    path = args.selected_capacity or args.output_root / "gpu" / "selected_capacity.json"
    return int(json.loads(path.read_text(encoding="utf-8"))["selected_num_envs"])


def close_env(env: Any) -> None:
    env.close()
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


def contact_force(env: Any) -> tuple[bool, float]:
    sensor = env._object_contact_sensors["Object170650"]
    matrix = sensor.data.force_matrix_w
    if matrix is None:
        return False, 0.0
    values = matrix[:, 0]
    norm = torch.linalg.vector_norm(values, dim=-1)
    return bool((norm > 1.0e-4).any()), float(norm.sum().detach().cpu())


def rollout_gate(
    *,
    num_envs: int,
    steps: int,
    rsi: bool,
    critical_dr: bool,
    measurement: bool,
    warmup_steps: int = 0,
    contact_probe: bool = False,
) -> dict[str, Any]:
    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )

    cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
    ppo26d_cfg.configure_stage16d_ppo26d(cfg, num_envs=num_envs, rsi=rsi, critical_dr=critical_dr)
    env = IsaacPPO26DReferenceTrackingEnv(cfg)
    try:
        observation, _ = env.reset(seed=20260808)
        action = torch.zeros((num_envs, 26), device=env.device)
        if contact_probe:
            # A bounded policy-space closing residual verifies that physical
            # contact, rather than a kinematic write, can affect the free object.
            action[:, 6:26] = 1.0
        initial_object = env._state()["object_position_scene"].clone()
        for _ in range(warmup_steps):
            env.step(action)
        reward_values = []
        seen_contact = False
        maximum_force = 0.0
        started = time.perf_counter()
        for _ in range(steps):
            observation, reward, terminated, timed_out, _ = env.step(action)
            reward_values.append(reward.detach())
            contact, force = contact_force(env)
            seen_contact |= contact
            maximum_force = max(maximum_force, force)
            if not bool(torch.isfinite(observation["policy"]).all()):
                raise FloatingPointError("PPO26D observation became non-finite")
            if not bool(torch.isfinite(reward).all()):
                raise FloatingPointError("PPO26D reward became non-finite")
            if not bool(
                torch.isfinite(terminated.float()).all() & torch.isfinite(timed_out.float()).all()
            ):
                raise FloatingPointError("PPO26D done flags became non-finite")
        elapsed = time.perf_counter() - started
        rewards = torch.stack(reward_values)
        object_motion = torch.linalg.vector_norm(
            env._state()["object_position_scene"] - initial_object, dim=-1
        )
        contract = env.contract_report()
        return {
            "num_envs": num_envs,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "rsi": rsi,
            "critical_dr": critical_dr,
            "measurement": measurement,
            "contact_probe": contact_probe,
            "observation_shape": list(observation["policy"].shape),
            "reference_index_min": int(env._reference_index.min().item()),
            "reference_index_max": int(env._reference_index.max().item()),
            "wrist_target_finite": bool(torch.isfinite(env._wrist_target_position).all()),
            "finger_target_finite": bool(torch.isfinite(env._joint_target_isaac).all()),
            "action_bounds": bool((action.abs() <= 1.0).all()),
            "joint_limits_safe": bool(
                (
                    (env._state()["finger_q"] >= env.joint_lower)
                    & (env._state()["finger_q"] <= env.joint_upper)
                ).all()
            ),
            "finite": bool(torch.isfinite(rewards).all()),
            "reward_non_constant": bool(torch.std(rewards) > 0.0),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "contact_seen": seen_contact,
            "maximum_contact_force_n": maximum_force,
            "object_motion_m": float(object_motion.max()),
            "contact_can_change_object_state": seen_contact and bool(object_motion.max() > 1.0e-6),
            "samples_per_s": num_envs * steps / max(elapsed, 1.0e-12),
            "contract": contract,
            "rsi_report": env.rsi_report(),
        }
    finally:
        close_env(env)


def ppo_gate(*, num_envs: int, checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )

    cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
    ppo26d_cfg.configure_stage16d_ppo26d(cfg, num_envs=num_envs, rsi=True, critical_dr=True)
    env = IsaacPPO26DReferenceTrackingEnv(cfg)
    try:
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        update = trainer.collect_and_update(env)
        path = trainer.save(
            checkpoint,
            environment_contract=env.contract_report(),
            selected_num_envs=num_envs,
        )
        reload_result = trainer.reload_deterministic_action(path, update["last_policy_observation"])
        update.pop("last_policy_observation")
        return update, {**reload_result, "checkpoint": str(path.resolve())}
    finally:
        close_env(env)


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    root = args.output_root.resolve()
    selected = selected_num_envs(args)
    app = AppLauncher(headless=True).app
    try:
        t0 = rollout_gate(
            num_envs=1,
            steps=100,
            rsi=False,
            critical_dr=False,
            measurement=False,
            contact_probe=True,
        )
        write_json(root / "trainability" / "t0_single_env.json", t0)
        t1 = rollout_gate(
            num_envs=128,
            steps=500,
            rsi=True,
            critical_dr=True,
            measurement=False,
            contact_probe=True,
        )
        write_json(root / "trainability" / "t1_rsi_128env.json", t1)
        t2 = rollout_gate(
            num_envs=selected,
            steps=500,
            warmup_steps=100,
            rsi=True,
            critical_dr=False,
            measurement=True,
        )
        write_json(root / "trainability" / "t2_selected_env.json", t2)
        checkpoint = root / "trainability" / "t4_checkpoint_roundtrip.pt"
        t3, t4 = ppo_gate(num_envs=selected, checkpoint=checkpoint)
        write_json(root / "trainability" / "t3_ppo_update.json", t3)
        write_json(root / "trainability" / "t4_checkpoint_reload.json", t4)
    finally:
        app.close(wait_for_replicator=False)
    checks = {
        "reference_loaded_correctly": t0["contract"]["reference_bank"]["frame_count"] == 321,
        "action_contract_26d": t0["contract"]["ppo26d"]["action_semantic"]
        == "Stage16DReferenceResidualAction26DV1",
        "wrist_se3_adapter": t0["wrist_target_finite"],
        "finger_target": t0["finger_target_finite"],
        "observation_finite": t0["finite"],
        "reward_finite": t0["finite"],
        "reward_non_constant": t0["reward_non_constant"],
        "rsi_effective": t1["rsi_report"]["sample_count"] >= 128,
        "object_free_rigid_body": not t0["contract"]["diagnostic_kinematic_object"],
        "contact_can_change_object_state": t0["contact_can_change_object_state"],
        "self_collision_enabled": t0["contract"]["ppo26d"]["self_collision_enabled"],
        "no_hidden_force": not t0["contract"]["ppo26d"]["hidden_force_or_attachment"],
        "no_rollout_object_state_write": t0["contract"]["object_rollout_state_writes"] == 0,
        "no_rollout_wrist_root_write": t0["contract"]["wrist_root_state_writes_during_step"] == 0,
        "action_bounds": t0["action_bounds"],
        "joint_limits": t0["joint_limits_safe"],
        "physx_finite": t2["finite"],
        "vector_env_stable": t2["finite"],
        "rollout_storage": t3["samples"] == selected * 40,
        "gae_finite": all(t3["finite"].values()),
        "ppo_losses_finite": all(
            torch.isfinite(torch.tensor(value)) for value in t3["ppo"].values()
        ),
        "actor_optimizer_update": t3["actor_parameter_changed"],
        "critic_optimizer_update": t3["critic_parameter_changed"],
        "checkpoint_roundtrip": t4["deterministic_action_identical"],
    }
    status = (
        "STAGE16D_PPO26D_TRAINABILITY_VALIDATED"
        if all(checks.values())
        else "STAGE16D_PPO26D_TRAINABILITY_FAILED"
    )
    result = {
        "schema_version": "Stage16DPPO26DTrainabilityGateV1",
        "status": status,
        "ppo_training_authorized": status == "STAGE16D_PPO26D_TRAINABILITY_VALIDATED",
        "selected_num_envs": selected,
        "checks": checks,
        "post_ppo_only": [
            "terminal_contact",
            "terminal_stability",
            "final_success",
            "exact_hand_object_penetration",
            "inter_finger_penetration",
        ],
    }
    write_json(root / "trainability" / "trainability_gate.json", result)
    print(json.dumps({"status": status, "selected_num_envs": selected}, sort_keys=True))
    return 0 if result["ppo_training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
