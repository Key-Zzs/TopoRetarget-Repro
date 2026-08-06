"""Actor-only behavior cloning warm start for Stage 16-D."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .networks import ActorCritic


@dataclass(frozen=True)
class BehaviorCloningConfigV1:
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 1024
    learning_rate: float = 1.0e-4

    def __post_init__(self) -> None:
        if not 1 <= self.max_epochs <= 50 or self.patience < 1:
            raise ValueError("BC epoch and patience budgets are invalid")
        if self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("BC batch size and learning rate must be positive")


def train_actor_behavior_cloning(
    *,
    model: ActorCritic,
    train_observations: np.ndarray,
    train_actions: np.ndarray,
    validation_observations: np.ndarray,
    validation_actions: np.ndarray,
    output_dir: Path,
    config: BehaviorCloningConfigV1 = BehaviorCloningConfigV1(),
    device: str = "cpu",
) -> dict[str, Any]:
    tensors = tuple(
        torch.as_tensor(value, dtype=torch.float32, device=device)
        for value in (
            train_observations,
            train_actions,
            validation_observations,
            validation_actions,
        )
    )
    train_obs, train_action, validation_obs, validation_action = tensors
    if (
        train_obs.ndim != 2
        or train_obs.shape[1] != 764
        or train_action.shape != (len(train_obs), 26)
    ):
        raise ValueError("BC training tensors violate 764D/26D contract")
    if (
        validation_obs.ndim != 2
        or validation_obs.shape[1] != 764
        or validation_action.shape != (len(validation_obs), 26)
    ):
        raise ValueError("BC validation tensors violate 764D/26D contract")
    model = model.to(device)
    critic_before = {
        name: value.detach().clone() for name, value in model.critic.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device=device).manual_seed(20260806)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, float | int]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config.max_epochs):
        order = torch.randperm(len(train_obs), generator=generator, device=device)
        losses: list[float] = []
        model.train()
        for indices in order.split(config.batch_size):
            prediction = torch.tanh(model.actor(train_obs[indices]))
            loss = torch.nn.functional.mse_loss(prediction, train_action[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                torch.nn.functional.mse_loss(
                    torch.tanh(model.actor(validation_obs)), validation_action
                )
            )
        history.append(
            {
                "epoch": epoch,
                "train_action_loss": sum(losses) / len(losses),
                "validation_action_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1.0e-8:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            save_checkpoint(
                output_dir / "bc_best.pt",
                {
                    "schema_version": "PhysicsCorrectionBCV1",
                    "actor": model.actor.state_dict(),
                    "epoch": epoch,
                    "validation_action_loss": validation_loss,
                },
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    save_checkpoint(
        output_dir / "bc_last.pt",
        {
            "schema_version": "PhysicsCorrectionBCV1",
            "actor": model.actor.state_dict(),
            "epoch": history[-1]["epoch"],
            "validation_action_loss": history[-1]["validation_action_loss"],
        },
    )
    if any(
        not torch.equal(value, critic_before[name])
        for name, value in model.critic.state_dict().items()
    ):
        raise RuntimeError("BC modified critic parameters")
    return {
        "schema_version": "PhysicsCorrectionBehaviorCloningV1",
        "config": asdict(config),
        "actor_only": True,
        "critic_pseudo_labels": False,
        "best_epoch": best_epoch,
        "best_validation_action_loss": best_loss,
        "epochs_executed": len(history),
        "early_stopped": len(history) < config.max_epochs,
        "history": history,
        "best_checkpoint": str(output_dir / "bc_best.pt"),
        "last_checkpoint": str(output_dir / "bc_last.pt"),
    }


__all__ = ["BehaviorCloningConfigV1", "train_actor_behavior_cloning"]
