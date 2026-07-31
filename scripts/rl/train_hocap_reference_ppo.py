#!/usr/bin/env python3
"""Run bounded, actual MuJoCo PPO updates on accepted HOCap references.

This is a functional CPU protocol, not a claim of the undisclosed paper-scale
4096-environment simulator.  The accepted trajectories use an instantaneous
wrist-relative frame, so this runtime deliberately removes the synthetic floor
and gravity rather than misinterpreting that local frame as world-up.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch

from toporetarget.rl.contracts import Stage16ReferenceClip
from toporetarget.rl.environments.mujoco_backend import (
    MujocoBackendConfig,
    MujocoReferenceTrackingBackend,
    materialize_free_object_scene,
)
from toporetarget.rl.ppo.checkpoint import load_checkpoint, rng_state, save_checkpoint
from toporetarget.rl.ppo.storage import RolloutStorage
from toporetarget.rl.ppo.trainer import PPOConfig, PPOTrainer
from toporetarget.rl.randomization import DomainRandomizationConfig
from toporetarget.rl.termination import BASE_RELATIVE_HOCAP_TERMINATION

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"


def make_backends(
    references: list[Path],
    object_meshes: list[Path],
    scene_root: Path,
    *,
    action_scale_fraction: float,
    domain_randomization: bool,
    seed: int,
) -> tuple[list[MujocoReferenceTrackingBackend], list[dict[str, object]]]:
    """Build one actual per-object mesh environment for every accepted clip."""

    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = model.jnt_range[: model.njnt].copy()
    expected_joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in expected_joint_order):
        raise RuntimeError("Wuji MJCF has unnamed joints")
    config = MujocoBackendConfig(
        action_scale_fraction=action_scale_fraction,
        termination_profile=BASE_RELATIVE_HOCAP_TERMINATION,
    )
    backends: list[MujocoReferenceTrackingBackend] = []
    inventory: list[dict[str, object]] = []
    for index, (reference_path, mesh_path) in enumerate(
        zip(references, object_meshes, strict=True)
    ):
        reference = Stage16ReferenceClip.from_npz(reference_path)
        if reference.joint_order != expected_joint_order:
            raise ValueError(f"reference joint order does not match Wuji MJCF: {reference_path}")
        scene = materialize_free_object_scene(
            WUJI_MJCF,
            scene_root / f"clip_{index:02d}",
            object_mesh=mesh_path,
            include_ground=False,
            gravity_mps2=(0.0, 0.0, 0.0),
        )
        backend = MujocoReferenceTrackingBackend(
            scene_path=scene,
            reference=reference,
            joint_lower=bounds[:, 0],
            joint_upper=bounds[:, 1],
            config=config,
            randomization=DomainRandomizationConfig(enabled=domain_randomization),
            seed=seed + index,
        )
        backends.append(backend)
        inventory.append(
            {
                "reference": str(reference_path.resolve()),
                "reference_hash": reference.content_hash(),
                "frames": reference.frame_count,
                "object_mesh": str(mesh_path.resolve()),
                "scene": str(scene.resolve()),
                "source_sequence": reference.provenance["dataset_provenance"]["source_sequence"],
            }
        )
    return backends, inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--stage", choices=("T1", "T2", "T3"), required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--action-scale-fraction", type=float, default=0.05)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if len(args.reference) != len(args.object_mesh):
        raise ValueError("--reference and --object-mesh must have equal counts")
    if not 1 <= args.iterations <= 64:
        raise ValueError("functional PPO iterations must be in 1..64")
    if args.rollout_steps < 8 or args.rollout_steps % 8:
        raise ValueError("--rollout-steps must be a multiple of 8 and at least 8")
    if args.stage == "T3" and len(args.reference) != 2:
        raise ValueError("T3 functional protocol requires exactly the two accepted HOCap clips")
    if args.stage != "T3" and len(args.reference) != 1:
        raise ValueError("T1/T2 functional protocol requires exactly one HOCap clip")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    backends, inventory = make_backends(
        args.reference,
        args.object_mesh,
        args.scene_root,
        action_scale_fraction=args.action_scale_fraction,
        domain_randomization=args.domain_randomization,
        seed=args.seed,
    )
    observations = [backend.observation(backend.reset()) for backend in backends]
    observation_dim = int(observations[0].size)
    action_dim = backends[0].reference.dof_count
    if any(value.size != observation_dim for value in observations):
        raise RuntimeError("mixed clips have incompatible observation dimensions")
    # The published PPO table does not disclose minibatch settings.  This
    # bounded one-update profile exercises real gradients on CPU and records it
    # explicitly instead of presenting it as paper-scale optimization.
    ppo_config = PPOConfig(epochs=1, minibatches=1)
    trainer = PPOTrainer(observation_dim, action_dim, config=ppo_config, device=args.device)
    traces: list[dict[str, object]] = []
    finite = True
    terminations: Counter[str] = Counter()
    best_return = -float("inf")
    best_payload: dict[str, object] | None = None
    for iteration in range(1, args.iterations + 1):
        initial_observations = torch.as_tensor(
            np.stack(observations), dtype=torch.float32, device=trainer.device
        )
        trainer.update_observation_normalizer(initial_observations)
        rollout_observations: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        rollout_log_probs: list[torch.Tensor] = []
        rollout_rewards: list[torch.Tensor] = []
        rollout_dones: list[torch.Tensor] = []
        rollout_values: list[torch.Tensor] = []
        return_total = np.zeros(len(backends), dtype=np.float64)
        for _ in range(args.rollout_steps):
            observation_tensor = torch.as_tensor(
                np.stack(observations), dtype=torch.float32, device=trainer.device
            )
            with torch.no_grad():
                sampled_actions, _, values = trainer.act(observation_tensor)
                actions = torch.clamp(sampled_actions, -1.0, 1.0)
                log_probs = trainer.distribution(observation_tensor).log_prob(actions)
            next_observations: list[np.ndarray] = []
            rewards: list[float] = []
            dones: list[bool] = []
            for index, backend in enumerate(backends):
                state, reward, reason = backend.transition(actions[index].cpu().numpy())
                finite &= bool(
                    all(np.isfinite(value) for value in reward.values())
                    and all(np.isfinite(value).all() for value in state.values())
                )
                rewards.append(float(reward["total"]))
                return_total[index] += rewards[-1]
                done = reason is not None
                dones.append(done)
                if done:
                    terminations[str(reason)] += 1
                    state = backend.reset()
                next_observations.append(backend.observation(state))
            rollout_observations.append(observation_tensor.cpu())
            rollout_actions.append(actions.cpu())
            rollout_log_probs.append(log_probs.cpu())
            rollout_rewards.append(torch.tensor(rewards, dtype=torch.float32))
            rollout_dones.append(torch.tensor(dones, dtype=torch.bool))
            rollout_values.append(values.cpu())
            observations = next_observations
        storage = RolloutStorage(
            observations=torch.stack(rollout_observations),
            actions=torch.stack(rollout_actions),
            log_probs=torch.stack(rollout_log_probs),
            rewards=torch.stack(rollout_rewards),
            dones=torch.stack(rollout_dones),
            values=torch.stack(rollout_values),
        )
        with torch.no_grad():
            last_tensor = torch.as_tensor(
                np.stack(observations), dtype=torch.float32, device=trainer.device
            )
            _, _, last_value = trainer.act(last_tensor, deterministic=True)
        update = trainer.update(storage, last_value.cpu())
        finite &= all(np.isfinite(value) for value in update.values())
        mean_return = float(return_total.mean())
        payload = {
            "kind": "stage16_hocap_bounded_functional_ppo",
            "stage": args.stage,
            "iteration": iteration,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "normalizer": trainer.normalizer.state_dict(),
            "rng": rng_state(),
        }
        save_checkpoint(args.checkpoint_directory / "last.pt", payload)
        if mean_return > best_return:
            best_return = mean_return
            best_payload = payload
            save_checkpoint(args.checkpoint_directory / "best.pt", payload)
        traces.append(
            {
                "iteration": iteration,
                "sample_count": storage.sample_count,
                "mean_rollout_return": mean_return,
                "terminations": dict(sorted(terminations.items())),
                "update": update,
            }
        )
    if best_payload is None:  # pragma: no cover - iterations is validated above
        raise RuntimeError("PPO did not produce a checkpoint")
    reloaded = load_checkpoint(args.checkpoint_directory / "last.pt", map_location=trainer.device)
    report = {
        "status": "HOCAP_REFERENCE_PPO_BOUNDED_FUNCTIONAL_PASS"
        if finite
        else "HOCAP_REFERENCE_PPO_BOUNDED_FUNCTIONAL_FAIL",
        "stage": args.stage,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "actual_samples_per_iteration": args.rollout_steps * len(backends),
        "device": str(trainer.device),
        "references": inventory,
        "domain_randomization": args.domain_randomization,
        "action_scale_fraction": args.action_scale_fraction,
        "ppo_config": ppo_config.as_dict(),
        "physics_profile": {
            "backend": "mujoco_cpu_reference",
            "per_object_collision_mesh": True,
            "synthetic_ground_enabled": False,
            "gravity_mps2": [0.0, 0.0, 0.0],
            "height_termination_disabled": True,
            "reason": "accepted HOCap reference is wrist-relative; world-up is not preserved",
        },
        "traces": traces,
        "termination_counts": dict(sorted(terminations.items())),
        "checkpoint_validation": {
            "status": "CHECKPOINT_RELOAD_PASS",
            "last": str((args.checkpoint_directory / "last.pt").resolve()),
            "best": str((args.checkpoint_directory / "best.pt").resolve()),
            "reloaded_stage": reloaded["stage"],
            "reloaded_iteration": int(reloaded["iteration"]),
        },
        "paper_comparable": False,
        "non_claim": (
            "bounded functional CPU PPO; not the authors' undisclosed simulator or 4096-env scale"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({key: report[key] for key in ("status", "stage", "iterations")}, sort_keys=True)
    )
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
