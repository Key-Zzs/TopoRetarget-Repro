#!/usr/bin/env python3
"""Run frozen C1/C0 saturation diagnostics without an optimizer step."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
from toporetarget.rl.instrumentation.saturation import SaturationRecorder
from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--failure-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument(
        "--variant",
        choices=("c1_deterministic", "c1_stochastic", "c0_same_actor"),
        required=True,
    )
    return parser


def _load_trainer(path: Path, *, device: str) -> tuple[PPO26DTrainer, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=device)
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.freeze_observation_normalizer()
    trainer.cumulative_samples = int(payload["cumulative_samples"])
    return trainer, payload


@torch.no_grad()
def _run(
    *,
    trainer: PPO26DTrainer,
    payload: dict[str, Any],
    stage: str,
    deterministic: bool,
    num_envs: int,
    horizon: int,
    output: Path,
) -> dict[str, Any]:
    env = _make_table_env(
        clip="hocap_170105",
        num_envs=num_envs,
        start_index=0,
        mode=ContactRewardMode.AGGREGATE_V3,
        stage=stage,
    )
    try:
        observation, _ = env.reset(seed=20260814)
        if not deterministic:
            restore_rng_state(payload["rng"])
        actor_before = parameter_hash(trainer.model, "actor")
        critic_before = parameter_hash(trainer.model, "critic")
        recorder = SaturationRecorder(output / "telemetry")
        for _ in range(horizon):
            distribution = trainer.trainer.distribution(observation["policy"])
            action = distribution.mean if deterministic else distribution.sample()
            observation, _, terminated, timed_out, _ = env.step(action)
            recorder.record_step(
                actor_location=distribution.location,
                actor_mean=distribution.mean,
                actor_log_std=torch.log(distribution.std),
                sampled_action=action,
                environment=env.stage16_saturation_telemetry(),
            )
            if bool((terminated | timed_out).all()):
                break
        summary, full_rollout = recorder.persist_pre_gate(
            samples_before=trainer.cumulative_samples,
            samples_after=trainer.cumulative_samples,
        )
        result = {
            "schema_version": "Stage16P3C1FrozenDiagnosticV1",
            "stage": stage,
            "policy_mode": "deterministic" if deterministic else "stochastic",
            "optimizer_steps": 0,
            "actor_hash_before": actor_before,
            "actor_hash_after": parameter_hash(trainer.model, "actor"),
            "critic_hash_before": critic_before,
            "critic_hash_after": parameter_hash(trainer.model, "critic"),
            "normalizer_training": bool(trainer.trainer.normalizer.training),
            "summary": summary,
            "full_rollout": str(full_rollout),
        }
        _write(output / "diagnostic.json", result)
        return result
    finally:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.num_envs <= 0 or args.horizon != 40:
        raise ValueError("FROZEN_C1_DIAGNOSTIC_CONTRACT_INVALID")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        root = args.output_root.resolve()
        trainer, payload = _load_trainer(args.failure_checkpoint, device="cuda:0")
        stage, deterministic = {
            "c1_deterministic": ("C1", True),
            "c1_stochastic": ("C1", False),
            "c0_same_actor": ("C0", True),
        }[args.variant]
        result = _run(
            trainer=trainer,
            payload=payload,
            stage=stage,
            deterministic=deterministic,
            num_envs=args.num_envs,
            horizon=args.horizon,
            output=root / args.variant,
        )
        _write(
            root / f"{args.variant}_summary.json",
            {
                "schema_version": "Stage16P3C1FrozenDiagnosticSummaryV1",
                "optimizer_steps": 0,
                "variant": args.variant,
                "saturation": result["summary"]["global_saturation"],
                "actor_unchanged": result["actor_hash_before"] == result["actor_hash_after"],
                "critic_unchanged": result["critic_hash_before"] == result["critic_hash_after"],
            },
        )
    finally:
        app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
