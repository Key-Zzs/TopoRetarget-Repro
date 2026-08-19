#!/usr/bin/env python3
# ruff: noqa: E402
"""Replay one frozen Stage16 PPO update in an isolated shadow state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from toporetarget.rl.ppo.checkpoint import load_checkpoint
from toporetarget.rl.ppo.policy_preservation import replay_exact_update, sha256_file
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u25-checkpoint", type=Path, required=True)
    parser.add_argument("--u26-checkpoint", type=Path, required=True)
    parser.add_argument("--exact-batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actor-lr-scale", type=float, default=1.0)
    parser.add_argument("--actor-epochs", type=int)
    parser.add_argument("--critic-baseline-checkpoint", type=Path)
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--kl-gradient-ratio", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in (args.u25_checkpoint, args.u26_checkpoint, args.exact_batch):
        if not path.is_file():
            raise FileNotFoundError(f"POLICY_PRESERVATION_FROZEN_INPUT_MISSING:{path}")
    shadow_checkpoint = output / "shadow_post_update.pt"
    receipt = replay_exact_update(
        checkpoint_path=args.u25_checkpoint.resolve(),
        batch_path=args.exact_batch.resolve(),
        device=args.device,
        output_checkpoint=shadow_checkpoint,
        actor_lr_scale=args.actor_lr_scale,
        actor_epochs=args.actor_epochs,
        critic_baseline_checkpoint=args.critic_baseline_checkpoint,
        anchor_checkpoint=args.anchor_checkpoint,
        kl_gradient_ratio=args.kl_gradient_ratio,
    )
    canonical = load_checkpoint(args.u26_checkpoint.resolve(), map_location="cpu")
    canonical_trainer = PPO26DTrainer(observation_dim=764, device="cpu")
    canonical_trainer.model.load_state_dict(canonical["actor_critic"])
    canonical_actor_hash = parameter_hash(canonical_trainer.model, "actor")
    canonical_critic_hash = parameter_hash(canonical_trainer.model, "critic")
    actor_exact = receipt["actor_parameter_hash_after"] == canonical_actor_hash
    critic_exact = receipt["critic_parameter_hash_after"] == canonical_critic_hash
    classification = (
        "EXACT_UPDATE_REPRODUCED" if actor_exact and critic_exact else "UPDATE_NOT_REPRODUCED"
    )
    payload = {
        **receipt,
        "canonical_u26_checkpoint": str(args.u26_checkpoint.resolve()),
        "canonical_u26_checkpoint_sha256": sha256_file(args.u26_checkpoint.resolve()),
        "canonical_u26_actor_hash": canonical_actor_hash,
        "canonical_u26_critic_hash": canonical_critic_hash,
        "actor_hash_exact": actor_exact,
        "critic_hash_exact": critic_exact,
        "a0_classification": classification,
        "next_action": (
            "A1_ACTOR_LR_ABLATION"
            if classification == "EXACT_UPDATE_REPRODUCED"
            else "NEXT_FIX_EXACT_PPO_UPDATE_REPLAY"
        ),
        "actor_lr_scale": args.actor_lr_scale,
        "actor_epochs": args.actor_epochs,
        "kl_gradient_ratio": args.kl_gradient_ratio,
    }
    (output / "update_receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
