#!/usr/bin/env python3
"""Recover missing C0 PPO-health receipts from a frozen exact update batch.

This utility is intentionally not a trainer: it collects no environment steps,
never writes a canonical checkpoint, and can only enrich a receipt after the
original update checkpoint and exact pre-optimizer batch already exist.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state
from toporetarget.rl.ppo.gae import generalized_advantage_estimate
from toporetarget.rl.ppo.policy_preservation import (
    _configure_actor_lr_scale,
    sha256_file,
    validate_exact_batch,
)
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
from toporetarget.rl.ppo.storage import RolloutStorage

RUN_ROOT = REPO_ROOT / ".local/runs/stage16_contact_preserving_full_c0_validation"
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_contact_preserving_full_c0_validation"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--accept-exact-batch-reconstruction",
        action="store_true",
        help="Required acknowledgement that this is a deterministic receipt reconstruction.",
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"CONTACT_PRESERVING_C0_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _actor_state(trainer: PPO26DTrainer) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in trainer.model.state_dict().items()
        if name.startswith("actor") or name == "log_std_parameter"
    }


def _actor_delta(before: dict[str, torch.Tensor], trainer: PPO26DTrainer) -> float:
    current = trainer.model.state_dict()
    squared = sum(
        (current[name].detach().cpu() - value).double().square().sum()
        for name, value in before.items()
    )
    return float(torch.sqrt(squared))


def _state_error(
    current: dict[str, torch.Tensor], canonical: dict[str, torch.Tensor], prefix: str
) -> dict[str, float]:
    differences = [
        (current[name].detach().cpu() - value.detach().cpu()).double().reshape(-1)
        for name, value in canonical.items()
        if name.startswith(prefix) or (prefix == "actor" and name == "log_std_parameter")
    ]
    flattened = torch.cat(differences)
    return {
        "max_abs": float(flattened.abs().max()),
        "rms": float(flattened.square().mean().sqrt()),
    }


def _checkpoint_cumulative_samples(payload: dict[str, Any]) -> int:
    """Read the source-V3 and C0 checkpoint counters without changing lineage."""

    for field in (
        "cumulative_samples",
        "contact_preservation_stage_samples",
        "reward_v3_samples",
    ):
        value = payload.get(field)
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError("CONTACT_PRESERVING_C0_PREDECESSOR_SAMPLE_COUNTER_MISSING")


def _candidate_trainer_from_predecessor(
    checkpoint: Path, *, device: str
) -> tuple[PPO26DTrainer, dict[str, object], dict[str, Any]]:
    """Restore the actual candidate optimizer layout before loading its state."""

    payload = load_checkpoint(checkpoint, map_location=device)
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("param_groups"), list):
        raise ValueError("CONTACT_PRESERVING_C0_PREDECESSOR_OPTIMIZER_INVALID")
    group_count = len(optimizer["param_groups"])
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(payload["actor_critic"])
    if group_count == 1:
        trainer.trainer.optimizer.load_state_dict(optimizer)
        contract = _configure_actor_lr_scale(trainer, 0.5)
    elif group_count == 2:
        contract = _configure_actor_lr_scale(trainer, 0.5)
        trainer.trainer.optimizer.load_state_dict(optimizer)
    else:
        raise ValueError(f"CONTACT_PRESERVING_C0_PREDECESSOR_GROUP_COUNT_INVALID:{group_count}")
    if (
        len(trainer.trainer.optimizer.param_groups) != 2
        or float(contract["effective_actor_lr"]) != 5.0e-5
        or float(contract["critic_lr"]) != 1.0e-4
    ):
        raise RuntimeError("CONTACT_PRESERVING_C0_RECONSTRUCTION_LR_CONTRACT_INVALID")
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = _checkpoint_cumulative_samples(payload)
    return trainer, contract, payload


def _predecessor_for_update(update: int, frozen: dict[str, Any]) -> Path:
    if update == 1:
        source = frozen.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("checkpoint"), str):
            raise ValueError("CONTACT_PRESERVING_C0_SOURCE_AUTHORITY_MISSING")
        predecessor = Path(source["checkpoint"])
    else:
        predecessor = RUN_ROOT / f"training/updates/U{update - 1:04d}/checkpoint/checkpoint.pt"
    if not predecessor.is_file():
        raise FileNotFoundError(f"CONTACT_PRESERVING_C0_PREDECESSOR_MISSING:{predecessor}")
    return predecessor.resolve()


def reconstruct_update(update: int, *, device: str) -> dict[str, Any]:
    """Replay one exact saved update and compare only against its canonical result."""

    if not 1 <= update <= 26:
        raise ValueError("CONTACT_PRESERVING_C0_UPDATE_OUT_OF_RANGE")
    frozen = _read_json(REPORT_ROOT / "frozen_inputs.json")
    canonical = RUN_ROOT / f"training/updates/U{update:04d}/checkpoint/checkpoint.pt"
    batch_path = RUN_ROOT / f"training/updates/U{update:04d}/exact_batch/exact_batch.pt"
    for path in (canonical, batch_path):
        if not path.is_file():
            raise FileNotFoundError(f"CONTACT_PRESERVING_C0_CANONICAL_ARTIFACT_MISSING:{path}")
    predecessor = _predecessor_for_update(update, frozen)
    trainer, optimizer_contract, predecessor_payload = _candidate_trainer_from_predecessor(
        predecessor, device=device
    )
    source_trainer, _, _ = _candidate_trainer_from_predecessor(
        Path(str(frozen["source"]["checkpoint"])), device=device
    )
    source_actor = _actor_state(source_trainer)
    actor_before = _actor_state(trainer)
    batch = load_checkpoint(batch_path, map_location="cpu")
    integrity = validate_exact_batch(batch)
    storage = RolloutStorage(
        observations=batch["observations"].to(trainer.trainer.device),
        actions=batch["actions"].to(trainer.trainer.device),
        log_probs=batch["old_log_probs"].to(trainer.trainer.device),
        rewards=batch["rewards"].to(trainer.trainer.device),
        dones=batch["dones"].to(trainer.trainer.device),
        values=batch["values"].to(trainer.trainer.device),
    )
    last_value = batch["last_value"].to(trainer.trainer.device)
    advantages, returns = generalized_advantage_estimate(
        storage.rewards,
        storage.values,
        storage.dones,
        last_value,
        gamma=trainer.training_contract.gamma,
        gae_lambda=trainer.training_contract.gae_lambda,
    )
    if not torch.equal(advantages.detach().cpu(), batch["advantages"]) or not torch.equal(
        returns.detach().cpu(), batch["returns"]
    ):
        raise RuntimeError("CONTACT_PRESERVING_C0_RECONSTRUCTION_GAE_DRIFT")
    restore_rng_state(batch["rng_before_optimizer_update"])
    ppo = trainer.trainer.update(storage, last_value)
    trainer.trainer.update_observation_normalizer(storage.observations)
    canonical_payload = load_checkpoint(canonical, map_location="cpu")
    canonical_trainer = PPO26DTrainer(observation_dim=764, device="cpu")
    canonical_trainer.model.load_state_dict(canonical_payload["actor_critic"])
    actor_error = _state_error(
        trainer.model.state_dict(), canonical_payload["actor_critic"], "actor"
    )
    critic_error = _state_error(
        trainer.model.state_dict(), canonical_payload["actor_critic"], "critic"
    )
    normalizer_state = trainer.trainer.normalizer.state_dict()
    canonical_normalizer = canonical_payload["observation_normalization"]
    normalizer_equal = all(
        torch.equal(normalizer_state[name].detach().cpu(), value.detach().cpu())
        if isinstance(value, torch.Tensor)
        else normalizer_state[name] == value
        for name, value in canonical_normalizer.items()
    )
    return {
        "schema_version": "Stage16ContactPreservingC0ExactPPOHealthReconstructionV1",
        "update": update,
        "samples": int(canonical_payload["contact_preservation_stage_samples"]),
        "reconstructed_from_exact_batch": True,
        "predecessor_checkpoint": str(predecessor),
        "predecessor_checkpoint_sha256": sha256_file(predecessor),
        "canonical_checkpoint": str(canonical.resolve()),
        "canonical_checkpoint_sha256": sha256_file(canonical),
        "exact_batch": str(batch_path.resolve()),
        "exact_batch_sha256": sha256_file(batch_path),
        "exact_batch_integrity": integrity,
        "predecessor_optimizer_groups": len(predecessor_payload["optimizer"]["param_groups"]),
        "optimizer_contract": optimizer_contract,
        "stored_gae_exact": True,
        "ppo": ppo,
        "actor_delta_previous": _actor_delta(actor_before, trainer),
        "actor_delta_source": _actor_delta(source_actor, trainer),
        "canonical_actor_error": actor_error,
        "canonical_critic_error": critic_error,
        "canonical_normalizer_exact": normalizer_equal,
        "canonical_actor_hash": parameter_hash(canonical_trainer.model, "actor"),
        "canonical_critic_hash": parameter_hash(canonical_trainer.model, "critic"),
        "reconstruction_note": (
            "Exact saved rollout/GAE/RNG was replayed without environment collection; "
            "CUDA numeric parity is reported as an error, not assumed bitwise."
        ),
    }


def _update_progression(result: dict[str, Any]) -> None:
    path = REPORT_ROOT / "training/progression.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    update = str(result["update"])
    row = next((item for item in rows if item["update"] == update), None)
    if row is None:
        raise ValueError(f"CONTACT_PRESERVING_C0_PROGRESSION_UPDATE_MISSING:{update}")
    ppo = result["ppo"]
    row.update(
        {
            "actor_lr": str(result["optimizer_contract"]["effective_actor_lr"]),
            "critic_lr": str(result["optimizer_contract"]["critic_lr"]),
            "actor_delta_previous": str(result["actor_delta_previous"]),
            "actor_delta_source": str(result["actor_delta_source"]),
            "kl_previous_to_current": str(ppo["kl"]),
            "actor_loss": str(ppo["actor_loss"]),
            "critic_loss": str(ppo["value_loss"]),
            "entropy": str(ppo["entropy"]),
            "clip_fraction": str(ppo["clip_fraction"]),
            "actor_grad_norm": str(ppo["actor_grad_norm"]),
            "critic_grad_norm": str(ppo["critic_grad_norm"]),
        }
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _persist(result: dict[str, Any]) -> None:
    update = int(result["update"])
    root = REPORT_ROOT / f"training/updates/U{update:04d}"
    _write_json(root / "exact_batch_ppo_health_reconstruction.json", result)
    receipt_path = root / "train_receipt.json"
    receipt = _read_json(receipt_path)
    receipt["training_health"] = "RECOVERED_FROM_EXACT_BATCH"
    receipt["exact_batch_ppo_health_reconstruction"] = {
        "path": str((root / "exact_batch_ppo_health_reconstruction.json").resolve()),
        "canonical_actor_error": result["canonical_actor_error"],
        "canonical_critic_error": result["canonical_critic_error"],
        "canonical_normalizer_exact": result["canonical_normalizer_exact"],
    }
    receipt["training"] = {
        "actor_parameter_update_norm": result["actor_delta_previous"],
        "ppo": result["ppo"],
        "reconstructed_from_exact_batch": True,
    }
    _write_json(receipt_path, receipt)
    _update_progression(result)


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_exact_batch_reconstruction:
        raise ValueError("--accept-exact-batch-reconstruction is required")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CONTACT_PRESERVING_C0_RECONSTRUCTION_GPU_UNAVAILABLE")
    for update in sorted(set(args.updates)):
        result = reconstruct_update(update, device=args.device)
        _persist(result)
        print(json.dumps({"update": update, "status": "EXACT_BATCH_HEALTH_RECOVERED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
