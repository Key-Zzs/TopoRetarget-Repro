"""Isolated exact-batch PPO replay helpers for Stage16 policy preservation."""

from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

from .checkpoint import load_checkpoint, restore_rng_state, rng_state, save_checkpoint
from .gae import generalized_advantage_estimate
from .networks import ActorCritic
from .ppo26d_trainer import PPO26DTrainer, parameter_hash
from .storage import RolloutStorage

_BATCH_TENSORS = (
    "observations",
    "actions",
    "old_log_probs",
    "rewards",
    "dones",
    "values",
    "returns",
    "advantages",
    "last_value",
    "reference_indices",
)


def sha256_file(path: Path) -> str:
    """Return the content hash without changing the protected input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(value: object) -> str:
    """Hash a serialized state dictionary using the repository's torch format."""

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _parameter_state(model: torch.nn.Module, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith(prefix) or (prefix == "actor" and name == "log_std_parameter")
    }


def _delta_norm(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    current = model.state_dict()
    squared = sum((current[name] - value).double().square().sum() for name, value in before.items())
    return float(torch.sqrt(squared).detach().cpu())


def _to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.detach().to(device=device)


def _configure_actor_lr_scale(trainer: PPO26DTrainer, scale: float) -> dict[str, object]:
    """Split the shadow optimizer into actor/critic groups without changing state."""

    if not 0.0 < scale <= 1.0:
        raise ValueError("POLICY_PRESERVATION_ACTOR_LR_SCALE_INVALID")
    optimizer = trainer.trainer.optimizer
    if len(optimizer.param_groups) != 1:
        raise ValueError("POLICY_PRESERVATION_EXPECTED_ONE_CANONICAL_OPTIMIZER_GROUP")
    original = optimizer.param_groups[0]
    model = cast(ActorCritic, trainer.model)
    actor_parameters = list(model.actor.parameters()) + [model.log_std_parameter]
    critic_parameters = list(model.critic.parameters())
    if set(map(id, actor_parameters)).intersection(map(id, critic_parameters)):
        raise RuntimeError("POLICY_PRESERVATION_ACTOR_CRITIC_PARAMETERS_SHARED")
    if set(map(id, actor_parameters + critic_parameters)) != set(map(id, original["params"])):
        raise RuntimeError("POLICY_PRESERVATION_PARAMETER_PARTITION_INCOMPLETE")
    baseline_lr = float(original["lr"])
    if scale == 1.0:
        return {
            "actor_critic_shared_parameters": False,
            "canonical_optimizer_groups": 1,
            "shadow_optimizer_groups": 1,
            "actor_lr_scale": scale,
            "baseline_actor_lr": baseline_lr,
            "effective_actor_lr": baseline_lr,
            "critic_lr": baseline_lr,
        }
    hyperparameters = {
        key: copy.deepcopy(value) for key, value in original.items() if key != "params"
    }
    hyperparameters.pop("lr")
    actor_hyperparameters = {**hyperparameters, "lr": baseline_lr * scale}
    critic_hyperparameters = {**hyperparameters, "lr": baseline_lr}
    # Preserve the loaded Adam instance and its parameter-keyed moment state.
    # Reconstructing Adam can alter optimizer state handling even when the
    # numerical hyperparameters look identical.
    optimizer.param_groups = [
        {"params": actor_parameters, **actor_hyperparameters},
        {"params": critic_parameters, **critic_hyperparameters},
    ]
    return {
        "actor_critic_shared_parameters": False,
        "canonical_optimizer_groups": 1,
        "shadow_optimizer_groups": 2,
        "actor_lr_scale": scale,
        "baseline_actor_lr": baseline_lr,
        "effective_actor_lr": baseline_lr * scale,
        "critic_lr": baseline_lr,
    }


def _copy_critic_state(source: ActorCritic, target: ActorCritic) -> None:
    """Copy only the non-shared critic state between shadow models."""

    state = target.state_dict()
    for name, value in source.state_dict().items():
        if name.startswith("critic"):
            state[name] = value.detach().clone()
    target.load_state_dict(state)


def _copy_critic_optimizer_state(source: PPO26DTrainer, target: PPO26DTrainer) -> None:
    """Keep candidate critic Adam moments on the baseline trajectory."""

    source_model = cast(ActorCritic, source.model)
    target_model = cast(ActorCritic, target.model)
    source_parameters = dict(source_model.named_parameters())
    target_parameters = dict(target_model.named_parameters())
    for name, target_parameter in target_parameters.items():
        if name.startswith("critic"):
            source_parameter = source_parameters[name]
            target.trainer.optimizer.state[target_parameter] = copy.deepcopy(
                source.trainer.optimizer.state[source_parameter]
            )


def _minibatch_update(
    trainer: PPO26DTrainer,
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> dict[str, float]:
    """One exact baseline PPO minibatch update, factored for paired shadows."""

    distribution = trainer.trainer.distribution(observations)
    new_log_probs = distribution.log_prob(actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate = torch.minimum(
        ratio * advantages,
        torch.clamp(
            ratio,
            1 - trainer.trainer.config.clip_epsilon,
            1 + trainer.trainer.config.clip_epsilon,
        )
        * advantages,
    )
    actor_loss = -surrogate.mean()
    model = cast(ActorCritic, trainer.model)
    value_prediction = model.value(trainer.trainer.normalizer.normalize(observations))
    value_loss = torch.nn.functional.mse_loss(value_prediction, returns)
    entropy = distribution.entropy().mean()
    loss = (
        actor_loss
        + trainer.trainer.config.value_loss_coefficient * value_loss
        - trainer.trainer.config.entropy_coefficient * entropy
    )
    trainer.trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_parameters = list(model.actor.parameters()) + [model.log_std_parameter]
    critic_parameters = list(model.critic.parameters())
    if any(parameter.grad is None for parameter in actor_parameters + critic_parameters):
        raise RuntimeError("POLICY_PRESERVATION_GRADIENT_MISSING")
    actor_grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [cast(torch.Tensor, parameter.grad).detach().norm() for parameter in actor_parameters]
        )
    )
    critic_grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [cast(torch.Tensor, parameter.grad).detach().norm() for parameter in critic_parameters]
        )
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), trainer.trainer.config.max_grad_norm
    )
    trainer.trainer.optimizer.step()
    with torch.no_grad():
        post_distribution = trainer.trainer.distribution(observations)
        log_ratio = post_distribution.log_prob(actions) - old_log_probs
        approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
    return {
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "kl": float(approximate_kl.detach()),
        "clip_fraction": float(
            ((ratio - 1.0).abs() > trainer.trainer.config.clip_epsilon).float().mean().detach()
        ),
        "grad_norm": float(grad_norm.detach()),
        "actor_grad_norm": float(actor_grad_norm.detach()),
        "critic_grad_norm": float(critic_grad_norm.detach()),
        "ratio": float(ratio.mean().detach()),
        "action_std": float(distribution.std.mean().detach()),
    }


def _actor_lr_update_with_baseline_critic(
    trainer: PPO26DTrainer,
    *,
    checkpoint: dict[str, Any],
    storage: RolloutStorage,
    last_value: torch.Tensor,
    rng_before: dict[str, Any],
) -> dict[str, Any]:
    """Run a paired A1 update with a baseline critic at every minibatch.

    Candidate and baseline use their own saved RNG streams. The candidate's
    critic parameters and Adam moments are replaced with the baseline state
    after each minibatch, so actor LR is the sole evolving policy-side change.
    """

    baseline = PPO26DTrainer(observation_dim=764, device=str(trainer.trainer.device))
    baseline.model.load_state_dict(checkpoint["actor_critic"])
    baseline.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    baseline.trainer.normalizer.load_state_dict(checkpoint["observation_normalization"])
    baseline.trainer.normalizer.training = True
    advantages, returns = generalized_advantage_estimate(
        storage.rewards,
        storage.values,
        storage.dones,
        last_value,
        gamma=trainer.trainer.config.gamma,
        gae_lambda=trainer.trainer.config.gae_lambda,
    )
    flat = storage.flatten(advantages, returns)
    flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (
        flat["advantages"].std(unbiased=False) + 1.0e-8
    )
    generator = torch.Generator(device=trainer.trainer.device)
    generator.manual_seed(0)
    baseline_rng = copy.deepcopy(rng_before)
    candidate_rng = copy.deepcopy(rng_before)
    accumulators = {
        key: 0.0
        for key in (
            "actor_loss",
            "value_loss",
            "entropy",
            "kl",
            "clip_fraction",
            "grad_norm",
            "actor_grad_norm",
            "critic_grad_norm",
            "ratio",
            "action_std",
        )
    }
    updates = 0
    kl_per_minibatch: list[float] = []
    kl_per_epoch: list[float] = []
    minibatches_per_epoch: list[int] = []
    kl_early_stop = False
    for _ in range(trainer.trainer.config.epochs):
        epoch_kl: list[float] = []
        order = torch.randperm(
            storage.sample_count, generator=generator, device=trainer.trainer.device
        )
        for indices in order.chunk(trainer.trainer.config.minibatches):
            tensors = {
                key: flat[key][indices].to(trainer.trainer.device)
                for key in ("observations", "actions", "log_probs", "advantages", "returns")
            }
            _copy_critic_state(cast(ActorCritic, baseline.model), cast(ActorCritic, trainer.model))
            _copy_critic_optimizer_state(baseline, trainer)
            restore_rng_state(baseline_rng)
            _minibatch_update(
                baseline,
                observations=tensors["observations"],
                actions=tensors["actions"],
                old_log_probs=tensors["log_probs"],
                advantages=tensors["advantages"],
                returns=tensors["returns"],
            )
            baseline_rng = rng_state()
            restore_rng_state(candidate_rng)
            metric = _minibatch_update(
                trainer,
                observations=tensors["observations"],
                actions=tensors["actions"],
                old_log_probs=tensors["log_probs"],
                advantages=tensors["advantages"],
                returns=tensors["returns"],
            )
            candidate_rng = rng_state()
            _copy_critic_state(cast(ActorCritic, baseline.model), cast(ActorCritic, trainer.model))
            _copy_critic_optimizer_state(baseline, trainer)
            for key, value in metric.items():
                accumulators[key] += value
            updates += 1
            epoch_kl.append(metric["kl"])
            kl_per_minibatch.append(metric["kl"])
            if metric["kl"] > trainer.trainer.config.target_kl:
                kl_early_stop = True
                break
        kl_per_epoch.append(sum(epoch_kl) / len(epoch_kl))
        minibatches_per_epoch.append(len(epoch_kl))
        if kl_early_stop:
            break
    restore_rng_state(candidate_rng)
    return {key: value / updates for key, value in accumulators.items()} | {
        "sample_count": float(storage.sample_count),
        "updates": float(updates),
        "requested_epochs": float(trainer.trainer.config.epochs),
        "actual_epochs_executed": float(len(kl_per_epoch)),
        "requested_minibatches": float(
            trainer.trainer.config.epochs * trainer.trainer.config.minibatches
        ),
        "actual_minibatches_updated": float(updates),
        "minibatches_per_epoch": minibatches_per_epoch,
        "kl_per_epoch": kl_per_epoch,
        "kl_per_minibatch": kl_per_minibatch,
        "kl_early_stop": kl_early_stop,
        "target_kl": trainer.trainer.config.target_kl,
        "critic_trajectory": "baseline_per_minibatch",
    }


def _anchor_gradient_calibration(
    trainer: PPO26DTrainer,
    storage: RolloutStorage,
    advantages: torch.Tensor,
    anchor: PPO26DTrainer,
) -> dict[str, float]:
    """Calibrate KL beta on the first historical PPO minibatch."""

    flat = storage.flatten(advantages, advantages)
    count = storage.sample_count
    generator = torch.Generator(device=trainer.trainer.device)
    generator.manual_seed(0)
    indices = torch.randperm(count, generator=generator, device=trainer.trainer.device).chunk(
        trainer.trainer.config.minibatches
    )[0]
    observations = flat["observations"][indices].to(trainer.trainer.device)
    actions = flat["actions"][indices].to(trainer.trainer.device)
    old_log_probs = flat["log_probs"][indices].to(trainer.trainer.device)
    advantages_batch = flat["advantages"][indices].to(trainer.trainer.device)
    distribution = trainer.trainer.distribution(observations)
    ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
    surrogate = torch.minimum(
        ratio * advantages_batch,
        torch.clamp(
            ratio,
            1 - trainer.trainer.config.clip_epsilon,
            1 + trainer.trainer.config.clip_epsilon,
        )
        * advantages_batch,
    )
    ppo_objective = (
        -surrogate.mean()
        - trainer.trainer.config.entropy_coefficient * distribution.entropy().mean()
    )
    with torch.no_grad():
        anchor_distribution = anchor.trainer.distribution(observations)
    kl = (
        (
            torch.log(anchor_distribution.std / distribution.std)
            + (
                distribution.std.square()
                + (distribution.location - anchor_distribution.location).square()
            )
            / (2.0 * anchor_distribution.std.square())
            - 0.5
        )
        .sum(dim=-1)
        .mean()
    )
    model = cast(ActorCritic, trainer.model)
    parameters = list(model.actor.parameters()) + [model.log_std_parameter]
    ppo_gradient = torch.autograd.grad(ppo_objective, parameters, retain_graph=True)
    kl_gradient = torch.autograd.grad(kl, parameters)
    ppo_norm = float(
        torch.linalg.vector_norm(torch.stack([value.norm() for value in ppo_gradient]))
    )
    kl_norm = float(torch.linalg.vector_norm(torch.stack([value.norm() for value in kl_gradient])))
    # At the stage-start policy the anchor KL has a zero first-order gradient.
    # Any tiny residual is floating-point noise, not a calibration denominator.
    if ppo_norm <= 0.0 or kl_norm <= 1.0e-6:
        raise RuntimeError("POLICY_PRESERVATION_KL_CALIBRATION_BLOCKED")
    return {"ppo_gradient_norm": ppo_norm, "kl_gradient_norm": kl_norm}


def validate_exact_batch(batch: dict[str, Any]) -> dict[str, object]:
    """Validate the persisted U26 update authority before a shadow step."""

    if batch.get("schema_version") != "Stage16ContactCollapseExactPPOBatchV1":
        raise ValueError("POLICY_PRESERVATION_EXACT_BATCH_SCHEMA_INVALID")
    missing = [key for key in _BATCH_TENSORS if key not in batch]
    if missing:
        raise ValueError(f"POLICY_PRESERVATION_EXACT_BATCH_FIELDS_MISSING:{','.join(missing)}")
    observations = batch["observations"]
    actions = batch["actions"]
    if not isinstance(observations, torch.Tensor) or observations.ndim != 3:
        raise ValueError("POLICY_PRESERVATION_OBSERVATIONS_INVALID")
    if not isinstance(actions, torch.Tensor) or tuple(actions.shape[:2]) != tuple(
        observations.shape[:2]
    ):
        raise ValueError("POLICY_PRESERVATION_ACTIONS_INVALID")
    if observations.shape[-1] != 764 or actions.shape[-1] != 26:
        raise ValueError("POLICY_PRESERVATION_CONTRACT_DIMENSION_INVALID")
    steps, environments, _ = observations.shape
    if batch.get("rollout_steps") != steps or batch.get("num_envs") != environments:
        raise ValueError("POLICY_PRESERVATION_BATCH_METADATA_INVALID")
    if "rng_before_optimizer_update" not in batch:
        raise ValueError("POLICY_PRESERVATION_RNG_MISSING")
    for key in _BATCH_TENSORS:
        value = batch[key]
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"POLICY_PRESERVATION_BATCH_NONFINITE:{key}")
    expected_2d = (steps, environments)
    for key in ("old_log_probs", "rewards", "dones", "values", "returns", "advantages"):
        if tuple(batch[key].shape) != expected_2d:
            raise ValueError(f"POLICY_PRESERVATION_BATCH_SHAPE_INVALID:{key}")
    if tuple(batch["last_value"].shape) != (environments,):
        raise ValueError("POLICY_PRESERVATION_LAST_VALUE_INVALID")
    if tuple(batch["reference_indices"].shape) != expected_2d:
        raise ValueError("POLICY_PRESERVATION_REFERENCE_INDICES_INVALID")
    return {
        "schema_version": batch["schema_version"],
        "rollout_steps": steps,
        "num_envs": environments,
        "sample_count": steps * environments,
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "advantage_normalization": "inside PPOTrainer.update, after stored GAE validation",
        "minibatch_order": "torch.Generator(device).manual_seed(0); randperm per epoch",
    }


def replay_exact_update(
    *,
    checkpoint_path: Path,
    batch_path: Path,
    device: str,
    output_checkpoint: Path | None = None,
    actor_lr_scale: float = 1.0,
    actor_epochs: int | None = None,
    critic_baseline_checkpoint: Path | None = None,
    anchor_checkpoint: Path | None = None,
    kl_gradient_ratio: float | None = None,
) -> dict[str, object]:
    """Replay one historical PPO update on a deep-copied U25 state.

    The canonical checkpoint and exact batch are read only.  Any optional output
    is a shadow checkpoint and is intentionally outside a training lineage.
    """

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    batch = load_checkpoint(batch_path, map_location="cpu")
    integrity = validate_exact_batch(batch)
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(checkpoint["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    trainer.trainer.normalizer.load_state_dict(checkpoint["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(checkpoint["cumulative_samples"])
    optimizer_contract = _configure_actor_lr_scale(trainer, actor_lr_scale)
    baseline_epochs = trainer.trainer.config.epochs
    if actor_epochs is not None:
        if not 1 <= actor_epochs <= baseline_epochs:
            raise ValueError("POLICY_PRESERVATION_ACTOR_EPOCHS_INVALID")
        trainer.trainer.config = replace(trainer.trainer.config, epochs=actor_epochs)
    model = cast(ActorCritic, trainer.model)
    actor_before = _parameter_state(model, "actor")
    critic_before = _parameter_state(model, "critic")
    actor_hash_before = parameter_hash(model, "actor")
    critic_hash_before = parameter_hash(model, "critic")
    storage = RolloutStorage(
        observations=_to_device(batch["observations"], trainer.trainer.device),
        actions=_to_device(batch["actions"], trainer.trainer.device),
        log_probs=_to_device(batch["old_log_probs"], trainer.trainer.device),
        rewards=_to_device(batch["rewards"], trainer.trainer.device),
        dones=_to_device(batch["dones"], trainer.trainer.device),
        values=_to_device(batch["values"], trainer.trainer.device),
    )
    last_value = _to_device(batch["last_value"], trainer.trainer.device)
    advantages, returns = generalized_advantage_estimate(
        storage.rewards,
        storage.values,
        storage.dones,
        last_value,
        gamma=trainer.training_contract.gamma,
        gae_lambda=trainer.training_contract.gae_lambda,
    )
    gae_exact = bool(
        torch.equal(advantages.detach().cpu(), batch["advantages"])
        and torch.equal(returns.detach().cpu(), batch["returns"])
    )
    if not gae_exact:
        raise RuntimeError("POLICY_PRESERVATION_STORED_GAE_MISMATCH")
    restore_rng_state(batch["rng_before_optimizer_update"])
    anchor: PPO26DTrainer | None = None
    calibration: dict[str, float] | None = None
    coefficient = 0.0
    if anchor_checkpoint is not None or kl_gradient_ratio is not None:
        if anchor_checkpoint is None or kl_gradient_ratio is None:
            raise ValueError("POLICY_PRESERVATION_KL_ARGUMENTS_INCOMPLETE")
        anchor_payload = load_checkpoint(anchor_checkpoint, map_location="cpu")
        anchor = PPO26DTrainer(observation_dim=764, device=device)
        anchor.model.load_state_dict(anchor_payload["actor_critic"])
        anchor.trainer.normalizer.load_state_dict(anchor_payload["observation_normalization"])
        anchor.trainer.freeze_observation_normalizer()
        for parameter in anchor.model.parameters():
            parameter.requires_grad_(False)
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1.0e-8
        )
        calibration = _anchor_gradient_calibration(trainer, storage, normalized_advantages, anchor)
        coefficient = (
            float(kl_gradient_ratio)
            * calibration["ppo_gradient_norm"]
            / calibration["kl_gradient_norm"]
        )
    normalizer_count_before = float(trainer.trainer.normalizer.count)
    if actor_lr_scale < 1.0 and anchor is None and actor_epochs is None:
        update = _actor_lr_update_with_baseline_critic(
            trainer,
            checkpoint=checkpoint,
            storage=storage,
            last_value=last_value,
            rng_before=batch["rng_before_optimizer_update"],
        )
    else:
        update = trainer.trainer.update(
            storage,
            last_value,
            anchor_model=None if anchor is None else cast(ActorCritic, anchor.model),
            anchor_normalizer=None if anchor is None else anchor.trainer.normalizer,
            anchor_kl_coefficient=coefficient,
        )
    trainer.trainer.update_observation_normalizer(storage.observations)
    critic_baseline_applied = False
    if critic_baseline_checkpoint is not None:
        baseline = load_checkpoint(critic_baseline_checkpoint, map_location="cpu")
        state = model.state_dict()
        for name, value in baseline["actor_critic"].items():
            if name.startswith("critic"):
                state[name] = value.to(device=trainer.trainer.device)
        model.load_state_dict(state)
        critic_baseline_applied = True
    normalizer_count_after = float(trainer.trainer.normalizer.count)
    actor_hash_after = parameter_hash(model, "actor")
    critic_hash_after = parameter_hash(model, "critic")
    result: dict[str, object] = {
        "schema_version": "Stage16ContactSkillPolicyPreservationShadowReplayV1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "exact_batch": str(batch_path.resolve()),
        "exact_batch_sha256": sha256_file(batch_path),
        "exact_batch_integrity": integrity,
        "stored_gae_exact": gae_exact,
        "actor_parameter_hash_before": actor_hash_before,
        "actor_parameter_hash_after": actor_hash_after,
        "critic_parameter_hash_before": critic_hash_before,
        "critic_parameter_hash_after": critic_hash_after,
        "actor_parameter_delta_norm": _delta_norm(actor_before, model),
        "critic_parameter_delta_norm": _delta_norm(critic_before, model),
        "normalizer_count_before": normalizer_count_before,
        "normalizer_count_after": normalizer_count_after,
        "ppo": update,
        "optimizer_contract": optimizer_contract,
        "actor_epochs": actor_epochs if actor_epochs is not None else baseline_epochs,
        "baseline_actor_epochs": baseline_epochs,
        "critic_baseline_checkpoint": (
            None
            if critic_baseline_checkpoint is None
            else str(critic_baseline_checkpoint.resolve())
        ),
        "critic_baseline_applied": critic_baseline_applied,
        "anchor_checkpoint": None
        if anchor_checkpoint is None
        else str(anchor_checkpoint.resolve()),
        "kl_gradient_ratio": kl_gradient_ratio,
        "effective_kl_coefficient": coefficient,
        "kl_calibration": calibration,
        "rng_after_shadow_step": state_hash(rng_state()),
        "authoritative_ppo_training_run": False,
        "shadow_diagnostic_optimizer_step": True,
    }
    if output_checkpoint is not None:
        shadow_payload = {
            "schema_version": checkpoint["schema_version"],
            "clip": checkpoint["clip"],
            "cumulative_samples": checkpoint["cumulative_samples"] + storage.sample_count,
            "actor_critic": {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            },
            # The frozen evaluator reconstructs the canonical one-group Adam
            # even though it never steps it.  Preserve that loader-compatible
            # state; the experimental two-group contract is retained in the
            # receipt instead of making optimizer-free evaluation impossible.
            "optimizer": checkpoint["optimizer"],
            "observation_normalization": trainer.trainer.normalizer.state_dict(),
        }
        # Shadow evaluation never restores an RNG.  Do not serialize one here:
        # re-pickling a NumPy RNG object in the lightweight RL environment can
        # make an otherwise valid checkpoint unreadable by pinned IsaacLab.
        shadow_payload["shadow_policy_preservation"] = {
            "kind": "EXACT_BATCH_SHADOW_REPLAY",
            "canonical_predecessor": str(checkpoint_path.resolve()),
            "canonical_predecessor_sha256": sha256_file(checkpoint_path),
            "exact_batch": str(batch_path.resolve()),
            "exact_batch_sha256": sha256_file(batch_path),
            "never_canonical_training_lineage": True,
        }
        save_checkpoint(output_checkpoint, shadow_payload)
        result["shadow_checkpoint"] = str(output_checkpoint.resolve())
        result["shadow_checkpoint_sha256"] = sha256_file(output_checkpoint)
    return result


__all__ = ["replay_exact_update", "sha256_file", "state_hash", "validate_exact_batch"]
