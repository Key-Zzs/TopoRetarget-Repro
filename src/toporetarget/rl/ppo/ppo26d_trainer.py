"""Real rollout, PPO update, and checkpoint helpers for Stage 16-D.5."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, rng_state, save_checkpoint
from .gae import generalized_advantage_estimate
from .ppo26d_contract import (
    Stage16DPPO26DObservationV2,
    Stage16DPPO26DTrainingConfigV1,
    Stage16DReferenceResidualAction26DV1,
)
from .storage import RolloutStorage
from .trainer import PPOConfig, PPOTrainer


def parameter_hash(model: torch.nn.Module, prefix: str) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(prefix) or (prefix == "actor" and name == "log_std_parameter"):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _policy_observation(observation: dict[str, torch.Tensor]) -> torch.Tensor:
    value = observation["policy"]
    if value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise RuntimeError("PPO26D_OBSERVATION_INVALID")
    return value


class PPO26DTrainer:
    """A minimal actual PPO runner over a Gym-like IsaacLab vector environment."""

    def __init__(self, *, observation_dim: int, device: str) -> None:
        self.action_contract = Stage16DReferenceResidualAction26DV1()
        self.observation_contract = Stage16DPPO26DObservationV2()
        self.training_contract = Stage16DPPO26DTrainingConfigV1()
        if observation_dim != self.observation_contract.dimension:
            raise ValueError("PPO26D observation dimension is frozen at 764")
        config = PPOConfig(
            learning_rate=self.training_contract.learning_rate,
            gamma=self.training_contract.gamma,
            gae_lambda=self.training_contract.gae_lambda,
            epochs=self.training_contract.epochs,
            minibatches=self.training_contract.minibatches,
            entropy_coefficient=self.training_contract.entropy_coefficient,
            clip_epsilon=self.training_contract.ppo_clip,
            max_grad_norm=self.training_contract.max_grad_norm,
        )
        self.trainer = PPOTrainer(
            observation_dim=observation_dim,
            action_dim=self.action_contract.action_dimension,
            config=config,
            device=device,
        )
        self.cumulative_samples = 0

    @property
    def model(self) -> torch.nn.Module:
        return self.trainer.model

    def collect_and_update(self, env: Any) -> dict[str, Any]:
        observation, _ = env.reset()
        policy_observation = _policy_observation(observation)
        self.trainer.update_observation_normalizer(policy_observation)
        rows: dict[str, list[torch.Tensor]] = {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
        reward_terms: dict[str, list[float]] = {}
        started = time.perf_counter()
        for _ in range(self.training_contract.rollout_length):
            policy_observation = _policy_observation(observation)
            action, _, value = self.trainer.act(policy_observation)
            # The contract is bounded in policy space.  The environment validates
            # the same bound before using the existing SE3/finger adapter.
            action = action.clamp(-1.0, 1.0)
            log_prob = self.trainer.distribution(policy_observation).log_prob(action)
            next_observation, reward, terminated, timed_out, _ = env.step(action)
            done = terminated | timed_out
            for key, value_tensor in (
                ("observations", policy_observation),
                ("actions", action),
                ("log_probs", log_prob),
                ("rewards", reward),
                ("dones", done),
                ("values", value),
            ):
                rows[key].append(value_tensor.detach())
            for name, term in getattr(env, "_last_reward_terms", {}).items():
                if isinstance(term, torch.Tensor):
                    reward_terms.setdefault(name, []).append(float(term.mean().detach().cpu()))
            observation = next_observation
        policy_observation = _policy_observation(observation)
        with torch.no_grad():
            last_value = self.trainer.model.value(
                self.trainer.normalizer.normalize(policy_observation)
            )
        storage = RolloutStorage(
            observations=torch.stack(rows["observations"]),
            actions=torch.stack(rows["actions"]),
            log_probs=torch.stack(rows["log_probs"]),
            rewards=torch.stack(rows["rewards"]),
            dones=torch.stack(rows["dones"]),
            values=torch.stack(rows["values"]),
        )
        advantages, returns = generalized_advantage_estimate(
            storage.rewards,
            storage.values,
            storage.dones,
            last_value,
            gamma=self.training_contract.gamma,
            gae_lambda=self.training_contract.gae_lambda,
        )
        finite = {
            "reward": bool(torch.isfinite(storage.rewards).all()),
            "return": bool(torch.isfinite(returns).all()),
            "advantage": bool(torch.isfinite(advantages).all()),
            "value": bool(torch.isfinite(storage.values).all()),
            "logprob": bool(torch.isfinite(storage.log_probs).all()),
        }
        if not all(finite.values()):
            raise FloatingPointError("PPO26D rollout or GAE contains NaN/Inf")
        actor_before = parameter_hash(self.model, "actor")
        critic_before = parameter_hash(self.model, "critic")
        update = self.trainer.update(storage, last_value)
        actor_after = parameter_hash(self.model, "actor")
        critic_after = parameter_hash(self.model, "critic")
        self.cumulative_samples += storage.sample_count
        elapsed = time.perf_counter() - started
        if not all(torch.isfinite(torch.tensor(value)) for value in update.values()):
            raise FloatingPointError("PPO26D losses contain NaN/Inf")
        return {
            "rollout_length": storage.rollout_steps,
            "num_envs": storage.num_envs,
            "samples": storage.sample_count,
            "cumulative_samples": self.cumulative_samples,
            "wall_time_s": elapsed,
            "samples_per_s": storage.sample_count / max(elapsed, 1.0e-12),
            "finite": finite,
            "ppo": update,
            "actor_parameter_hash_before": actor_before,
            "actor_parameter_hash_after": actor_after,
            "critic_parameter_hash_before": critic_before,
            "critic_parameter_hash_after": critic_after,
            "actor_parameter_changed": actor_before != actor_after,
            "critic_parameter_changed": critic_before != critic_after,
            "reward": {name: sum(values) / len(values) for name, values in reward_terms.items()},
            "last_policy_observation": policy_observation.detach().clone(),
        }

    def checkpoint_payload(
        self, *, environment_contract: dict[str, Any], selected_num_envs: int
    ) -> dict[str, Any]:
        return {
            "schema_version": "Stage16DPPO26DCheckpointV1",
            "actor_critic": self.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "observation_normalization": self.trainer.normalizer.state_dict(),
            "ppo_config": asdict(self.training_contract),
            "action_contract": self.action_contract.as_dict(),
            "observation_contract": self.observation_contract.as_dict(),
            "environment_contract": environment_contract,
            "reference_hash": environment_contract["reference_bank"]["hashes"],
            "physics_contract_hash": hashlib.sha256(
                repr(environment_contract.get("joint_mapping", {})).encode("utf-8")
            ).hexdigest(),
            "selected_num_envs": selected_num_envs,
            "cumulative_samples": self.cumulative_samples,
            "rng": rng_state(),
        }

    def save(
        self, path: Path, *, environment_contract: dict[str, Any], selected_num_envs: int
    ) -> Path:
        return save_checkpoint(
            path,
            self.checkpoint_payload(
                environment_contract=environment_contract, selected_num_envs=selected_num_envs
            ),
        )

    def reload_deterministic_action(self, path: Path, observation: torch.Tensor) -> dict[str, Any]:
        payload = load_checkpoint(path, map_location=self.trainer.device)
        restored = PPO26DTrainer(
            observation_dim=observation.shape[-1], device=str(self.trainer.device)
        )
        restored.model.load_state_dict(payload["actor_critic"])
        restored.trainer.optimizer.load_state_dict(payload["optimizer"])
        restored.trainer.normalizer.load_state_dict(payload["observation_normalization"])
        restored.cumulative_samples = int(payload["cumulative_samples"])
        with torch.no_grad():
            before = self.trainer.act(observation, deterministic=True)[0]
            after = restored.trainer.act(observation, deterministic=True)[0]
        return {
            "checkpoint_schema": payload["schema_version"],
            "deterministic_action_identical": bool(torch.equal(before, after)),
            "cumulative_samples": restored.cumulative_samples,
            "action_contract": payload["action_contract"],
        }


__all__ = ["PPO26DTrainer", "parameter_hash"]
