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
            target_kl=self.training_contract.target_kl,
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

    @torch.no_grad()
    def _policy_safety_metrics(
        self,
        observations: torch.Tensor,
        *,
        sampled_actions: torch.Tensor | None = None,
        phase: str,
    ) -> dict[str, float | bool | str]:
        normalized_abs_max = 0.0
        normalized_abs_sum = 0.0
        normalized_count = 0
        deterministic_saturated = 0
        action_count = 0
        finite = True
        for observation in observations:
            normalized = self.trainer.normalizer.normalize(observation)
            finite = finite and bool(torch.isfinite(normalized).all())
            normalized_abs = normalized.abs()
            normalized_abs_max = max(normalized_abs_max, float(normalized_abs.max().detach().cpu()))
            normalized_abs_sum += float(normalized_abs.sum().detach().cpu())
            normalized_count += normalized.numel()
            deterministic_action = self.trainer.distribution(observation).mean
            deterministic_saturated += int(
                (
                    deterministic_action.abs()
                    >= self.training_contract.action_saturation_absolute_threshold
                )
                .sum()
                .detach()
                .cpu()
            )
            action_count += deterministic_action.numel()
        deterministic_fraction = deterministic_saturated / max(action_count, 1)
        sampled_fraction = 0.0
        if sampled_actions is not None:
            sampled_fraction = float(
                (
                    sampled_actions.abs()
                    >= self.training_contract.action_saturation_absolute_threshold
                )
                .float()
                .mean()
                .detach()
                .cpu()
            )
        metrics: dict[str, float | bool | str] = {
            "phase": phase,
            "normalized_observation_finite": finite,
            "normalized_observation_abs_max": normalized_abs_max,
            "normalized_observation_abs_mean": normalized_abs_sum / max(normalized_count, 1),
            "normalized_observation_abs_limit": (
                self.training_contract.normalized_observation_abs_limit
            ),
            "deterministic_action_saturation_fraction": deterministic_fraction,
            "sampled_action_saturation_fraction": sampled_fraction,
            "action_saturation_absolute_threshold": (
                self.training_contract.action_saturation_absolute_threshold
            ),
            "action_saturation_fraction_limit": (
                self.training_contract.action_saturation_fraction_limit
            ),
        }
        if (
            not finite
            or normalized_abs_max > self.training_contract.normalized_observation_abs_limit
        ):
            raise FloatingPointError(
                "PPO26D_NORMALIZED_OBSERVATION_FAIL_FAST: "
                f"phase={phase} finite={finite} abs_max={normalized_abs_max:.6g}"
            )
        if deterministic_fraction > self.training_contract.action_saturation_fraction_limit:
            raise FloatingPointError(
                "PPO26D_ACTION_SATURATION_FAIL_FAST: "
                f"phase={phase} fraction={deterministic_fraction:.6f}"
            )
        return metrics

    def collect_and_update(self, env: Any) -> dict[str, Any]:
        observation, _ = env.reset()
        policy_observation = _policy_observation(observation)
        normalizer_count_before = float(self.trainer.normalizer.count)
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
        reference_index_sum = 0.0
        reference_index_count = 0
        for _ in range(self.training_contract.rollout_length):
            policy_observation = _policy_observation(observation)
            action, log_prob, value = self.trainer.act(policy_observation)
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
            reference_index_sum += float(env._reference_index.float().mean().detach().cpu())
            reference_index_count += 1
            observation = next_observation
        rollout_collection_s = time.perf_counter() - started
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
        safety_before_update = self._policy_safety_metrics(
            storage.observations,
            sampled_actions=storage.actions,
            phase="before_update",
        )
        rollout_storage_mib = (
            sum(
                value.numel() * value.element_size()
                for value in (
                    storage.observations,
                    storage.actions,
                    storage.log_probs,
                    storage.rewards,
                    storage.dones,
                    storage.values,
                )
            )
            / 2**20
        )
        actor_before = parameter_hash(self.model, "actor")
        critic_before = parameter_hash(self.model, "critic")
        update_started = time.perf_counter()
        update = self.trainer.update(storage, last_value)
        ppo_update_s = time.perf_counter() - update_started
        normalizer_count_after_ppo = float(self.trainer.normalizer.count)
        if normalizer_count_after_ppo != normalizer_count_before:
            raise RuntimeError("PPO26D observation statistics changed during rollout/update")
        safety_after_update = self._policy_safety_metrics(
            storage.observations,
            phase="after_update_before_normalizer_refresh",
        )
        self.trainer.update_observation_normalizer(storage.observations)
        normalizer_count_after_refresh = float(self.trainer.normalizer.count)
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
            "rollout_collection_s": rollout_collection_s,
            "ppo_update_s": ppo_update_s,
            "samples_per_s": storage.sample_count / max(elapsed, 1.0e-12),
            "rollout_storage_mib": rollout_storage_mib,
            "finite": finite,
            "safety": {
                "before_update": safety_before_update,
                "after_update": safety_after_update,
                "normalizer_count_before": normalizer_count_before,
                "normalizer_count_after_ppo": normalizer_count_after_ppo,
                "normalizer_count_after_refresh": normalizer_count_after_refresh,
                "normalizer_samples_added": (
                    normalizer_count_after_refresh - normalizer_count_after_ppo
                ),
                "normalizer_frozen_during_rollout_and_update": True,
            },
            "ppo": update,
            "actor_parameter_hash_before": actor_before,
            "actor_parameter_hash_after": actor_after,
            "critic_parameter_hash_before": critic_before,
            "critic_parameter_hash_after": critic_after,
            "actor_parameter_changed": actor_before != actor_after,
            "critic_parameter_changed": critic_before != critic_after,
            "reward": {name: sum(values) / len(values) for name, values in reward_terms.items()},
            "reference": {
                "mean_reference_index": reference_index_sum / max(reference_index_count, 1),
                "reference_progress": reference_index_sum
                / max(reference_index_count * (env.reference_bank.frame_count - 1), 1),
                "rsi": env.rsi_report(),
            },
            "last_policy_observation": policy_observation.detach().clone(),
        }

    def checkpoint_payload(
        self, *, environment_contract: dict[str, Any], selected_num_envs: int
    ) -> dict[str, Any]:
        clip = environment_contract.get("ppo26d", {}).get("fixed_clip")
        active_clips = environment_contract.get("ppo26d", {}).get("active_clip_ids")
        if clip not in {"hocap_170105", "hocap_170650"} or active_clips != [clip]:
            raise ValueError(f"PPO26D_FIXED_CLIP_MISMATCH: fixed={clip!r} active={active_clips!r}")
        return {
            "schema_version": "Stage16DPPO26DCheckpointV1",
            "clip": clip,
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
