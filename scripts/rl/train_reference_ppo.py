#!/usr/bin/env python3
"""Run bounded PPO numerical smoke checks; physical training requires qualified assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toporetarget.rl.ppo.checkpoint import load_checkpoint, rng_state, save_checkpoint
from toporetarget.rl.ppo.gae import generalized_advantage_estimate
from toporetarget.rl.ppo.storage import RolloutStorage
from toporetarget.rl.ppo.trainer import PPOConfig, PPOTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        help="Optional directory for T0 numerical-smoke checkpoints only.",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 20:
        raise ValueError("T0 numerical smoke is bounded to 1..20 iterations")
    device = torch.device(args.device)
    # T0 deliberately uses a small synthetic numerical tensor, not an environment roll-out.
    trainer = PPOTrainer(64, 20, device=str(device))
    finite = True
    traces = []
    for iteration in range(args.iterations):
        observations = torch.randn(40, 32, 64, device=device)
        trainer.update_observation_normalizer(observations)
        with torch.no_grad():
            actions, log_probs, values = trainer.act(observations.reshape(-1, 64))
        actions = actions.reshape(40, 32, 20)
        log_probs = log_probs.reshape(40, 32)
        values = values.reshape(40, 32)
        rewards = torch.randn(40, 32, device=device)
        dones = torch.zeros(40, 32, device=device, dtype=torch.bool)
        advantages, returns = generalized_advantage_estimate(
            rewards, values, dones, torch.zeros(32, device=device)
        )
        storage = RolloutStorage(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
        )
        update = trainer.update(storage, torch.zeros(32, device=device))
        finite &= bool(
            torch.isfinite(advantages).all()
            and torch.isfinite(returns).all()
            and storage.sample_count == 1280
            and all(np.isfinite(value) for value in update.values())
        )
        traces.append(
            {
                "iteration": iteration + 1,
                "samples": storage.sample_count,
                "advantage_mean": float(advantages.mean()),
                "return_mean": float(returns.mean()),
                "update": update,
            }
        )
    checkpoint_validation: dict[str, object] = {
        "status": "NOT_REQUESTED",
        "physical_policy_checkpoint": False,
    }
    if args.checkpoint_directory is not None:
        payload = {
            "kind": "stage16_t0_numerical_smoke",
            "iteration": args.iterations,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "normalizer": trainer.normalizer.state_dict(),
            "rng": rng_state(),
        }
        paths = {
            "last": args.checkpoint_directory / "last.pt",
            "best": args.checkpoint_directory / "best.pt",
            "iteration": args.checkpoint_directory / f"checkpoint_{args.iterations:06d}.pt",
        }
        for path in paths.values():
            save_checkpoint(path, payload)
        reloaded = load_checkpoint(paths["last"], map_location=device)
        checkpoint_validation = {
            "status": "T0_CHECKPOINT_RELOAD_PASS",
            "physical_policy_checkpoint": False,
            "paths": {key: str(value) for key, value in paths.items()},
            "reloaded_iteration": int(reloaded["iteration"]),
            "reloaded_kind": str(reloaded["kind"]),
            "normalizer_count": float(reloaded["normalizer"]["count"]),
        }
        finite &= checkpoint_validation["reloaded_iteration"] == args.iterations
    report = {
        "status": "T0_PPO_NUMERICAL_SMOKE_PASS" if finite else "T0_PPO_NUMERICAL_SMOKE_FAIL",
        "iterations": args.iterations,
        "device": str(device),
        "architecture": {
            "actor": [512, 256, 128],
            "critic": [512, 512, 256, 128],
            "activation": "ELU",
            "distribution": "SoftplusGaussian",
        },
        "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "paper_sample_target": 163840,
        "actual_t0_samples_per_iteration": 1280,
        "physical_training_status": "BLOCKED_PENDING_QUALIFIED_HOCAP_REFERENCE_AND_OBJECT_ASSET",
        "checkpoint_validation": checkpoint_validation,
        "trace": traces,
        "ppo_engineering_assumptions": PPOConfig().as_dict(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({key: report[key] for key in ("status", "iterations", "device")}, sort_keys=True)
    )
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
