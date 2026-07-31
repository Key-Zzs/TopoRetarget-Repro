#!/usr/bin/env python3
"""Bounded Stage-16.2/16.3 PPO trainer with explicit sample accounting.

This trainer is intentionally separate from the earlier functional T1/T2/T3
runner.  It uses the paper optimizer settings (4 epochs, 32 minibatches),
balanced clip collection, random-k0 rollouts, and atomic lineage-carrying
checkpoints.  The caller must provide a completed Stage-16.1 gate before
starting a qualification run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
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
SAMPLE_LADDER = (32_768, 131_072, 524_288, 2_097_152, 8_388_608, 16_777_216)


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_backends(
    references: list[Path],
    meshes: list[Path],
    scene_root: Path,
    *,
    action_scale: float,
    domain_randomization: bool,
    seed: int,
) -> tuple[list[MujocoReferenceTrackingBackend], list[dict[str, object]]]:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    bounds = model.jnt_range[: model.njnt].copy()
    joint_order = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    if any(name is None for name in joint_order):
        raise RuntimeError("Wuji MJCF contains unnamed joints")
    names = tuple(name for name in joint_order if name is not None)
    backends: list[MujocoReferenceTrackingBackend] = []
    inventory: list[dict[str, object]] = []
    for index, (reference_path, mesh_path) in enumerate(zip(references, meshes, strict=True)):
        reference = Stage16ReferenceClip.from_npz(reference_path)
        reference.validate(expected_hz=20.0)
        if reference.joint_order != names:
            raise ValueError(f"reference joint order does not match Wuji MJCF: {reference_path}")
        scene = materialize_free_object_scene(
            WUJI_MJCF,
            scene_root / f"clip_{index:02d}",
            object_mesh=mesh_path,
            include_ground=False,
            gravity_mps2=(0.0, 0.0, 0.0),
        )
        backends.append(
            MujocoReferenceTrackingBackend(
                scene_path=scene,
                reference=reference,
                joint_lower=bounds[:, 0],
                joint_upper=bounds[:, 1],
                config=MujocoBackendConfig(
                    action_scale_fraction=action_scale,
                    termination_profile=BASE_RELATIVE_HOCAP_TERMINATION,
                ),
                randomization=DomainRandomizationConfig(enabled=domain_randomization),
                seed=seed + index,
            )
        )
        inventory.append(
            {
                "reference": str(reference_path.resolve()),
                "reference_hash": reference.content_hash(),
                "object_mesh": str(mesh_path.resolve()),
                "object_mesh_sha256": _sha256(mesh_path),
                "frames": reference.frame_count,
                "source_sequence": reference.provenance["dataset_provenance"]["source_sequence"],
            }
        )
    return backends, inventory


def _as_tensor(values: list[np.ndarray], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.stack(values), dtype=torch.float32, device=device)


def train(
    *,
    backends: list[MujocoReferenceTrackingBackend],
    checkpoint_directory: Path,
    budget: int,
    rollout_steps: int,
    ppo_config: PPOConfig,
    device: str,
    stage: str,
    inventory: list[dict[str, object]],
    seed: int,
    init_checkpoint: Path | None = None,
) -> dict[str, object]:
    if budget not in SAMPLE_LADDER:
        raise ValueError(f"budget must be one of the bounded sample ladder values: {SAMPLE_LADDER}")
    if rollout_steps * len(backends) % ppo_config.minibatches:
        raise ValueError("rollout sample count must divide evenly into 32 minibatches")
    torch.manual_seed(seed)
    np.random.seed(seed)
    initial_states = [backend.reset() for backend in backends]
    observations = [
        backend.observation(state) for backend, state in zip(backends, initial_states, strict=True)
    ]
    trainer = PPOTrainer(
        observations[0].size,
        backends[0].reference.dof_count,
        config=ppo_config,
        device=device,
    )
    if init_checkpoint is not None:
        payload = load_checkpoint(init_checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(payload["model"])
        trainer.normalizer.load_state_dict(payload["normalizer"])
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    cumulative = 0
    iteration = 0
    traces: list[dict[str, object]] = []
    start = time.monotonic()
    while cumulative < budget:
        iteration += 1
        initial_observation = _as_tensor(observations, trainer.device)
        trainer.update_observation_normalizer(initial_observation)
        rollout_observations: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        rollout_log_probs: list[torch.Tensor] = []
        rollout_rewards: list[torch.Tensor] = []
        rollout_dones: list[torch.Tensor] = []
        rollout_values: list[torch.Tensor] = []
        returns = np.zeros(len(backends), dtype=np.float64)
        for _ in range(rollout_steps):
            observation_tensor = _as_tensor(observations, trainer.device)
            with torch.no_grad():
                sampled, log_probs, values = trainer.act(observation_tensor)
                actions = torch.clamp(sampled, -1.0, 1.0)
            next_observations: list[np.ndarray] = []
            rewards: list[float] = []
            dones: list[bool] = []
            for index, backend in enumerate(backends):
                state, reward, reason = backend.transition(actions[index].cpu().numpy())
                rewards.append(float(reward["total"]))
                returns[index] += rewards[-1]
                done = reason is not None
                dones.append(done)
                if done:
                    state = backend.reset()
                next_observations.append(backend.observation(state))
            rollout_observations.append(observation_tensor.cpu())
            rollout_actions.append(actions.cpu())
            rollout_log_probs.append(log_probs.cpu())
            rollout_rewards.append(torch.as_tensor(rewards, dtype=torch.float32))
            rollout_dones.append(torch.as_tensor(dones, dtype=torch.bool))
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
            _, _, last_value = trainer.act(
                _as_tensor(observations, trainer.device), deterministic=True
            )
        update = trainer.update(storage, last_value.cpu())
        cumulative += storage.sample_count
        payload = {
            "kind": "stage16_reference_tracking_qualification_ppo",
            "stage": stage,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "normalizer": trainer.normalizer.state_dict(),
            "rng": rng_state(),
            "iteration": iteration,
            "cumulative_samples": cumulative,
            "env_count": len(backends),
            "rollout_steps": rollout_steps,
            "reference_selection": inventory,
            "config": ppo_config.as_dict(),
            "seed": seed,
            "code_head": _git_head(),
            "paper_protocol": {
                "domain_randomization": False,
                "observation_noise": False,
                "random_k0_training": True,
                "frame0_evaluation_required": True,
            },
        }
        save_checkpoint(checkpoint_directory / "last.pt", payload)
        best_path = save_checkpoint(
            checkpoint_directory / f"checkpoint_{cumulative:09d}.pt", payload
        )
        traces.append(
            {
                "iteration": iteration,
                "samples": storage.sample_count,
                "cumulative_samples": cumulative,
                "mean_rollout_return": float(np.mean(returns)),
                "update": update,
                "checkpoint": str(best_path.resolve()),
            }
        )
    elapsed = time.monotonic() - start
    return {
        "status": "STAGE16_TRAINING_BUDGET_COMPLETE",
        "stage": stage,
        "budget": budget,
        "cumulative_samples": cumulative,
        "iterations": iteration,
        "env_count": len(backends),
        "rollout_steps": rollout_steps,
        "samples_per_iteration": rollout_steps * len(backends),
        "samples_per_second": cumulative / max(elapsed, 1e-9),
        "wall_seconds": elapsed,
        "ppo_config": ppo_config.as_dict(),
        "references": inventory,
        "init_checkpoint": str(init_checkpoint.resolve()) if init_checkpoint else None,
        "checkpoints": {
            "last": str((checkpoint_directory / "last.pt").resolve()),
            "last_reload": load_checkpoint(checkpoint_directory / "last.pt")["cumulative_samples"],
        },
        "traces": traces,
        "paper_comparable": False,
        "non_claim": "bounded qualification training; it is not HOCap-32 paper evaluation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--controllability-report", required=True, type=Path)
    parser.add_argument("--stage", choices=("single", "two_clip"), required=True)
    parser.add_argument("--budget", type=int, required=True, choices=SAMPLE_LADDER)
    parser.add_argument("--rollout-steps", type=int, default=320)
    parser.add_argument("--action-scale-fraction", type=float, default=0.05)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    if len(args.reference) != len(args.object_mesh):
        raise ValueError("reference and object-mesh counts must match")
    if args.stage == "single" and len(args.reference) != 1:
        raise ValueError("single stage requires exactly one clip")
    if args.stage == "two_clip" and len(args.reference) != 2:
        raise ValueError("two_clip stage requires exactly two balanced clips")
    if not args.controllability_report.is_file():
        raise ValueError("Stage16.1 controllability report is required before PPO training")
    gate = json.loads(args.controllability_report.read_text(encoding="utf-8"))
    if gate.get("status") != "STAGE16_1_CONTROLLABILITY_COMPLETE":
        raise ValueError(
            f"Stage16.2/16.3 is gate-blocked: Stage16.1 status is {gate.get('status', 'MISSING')}"
        )
    if args.domain_randomization:
        raise ValueError("Stage16.2/16.3 qualification is nominal no-DR; full DR is Stage16.4")
    backends, inventory = build_backends(
        args.reference,
        args.object_mesh,
        args.scene_root,
        action_scale=args.action_scale_fraction,
        domain_randomization=False,
        seed=args.seed,
    )
    config = PPOConfig(epochs=4, minibatches=32)
    report = train(
        backends=backends,
        checkpoint_directory=args.checkpoint_directory,
        budget=args.budget,
        rollout_steps=args.rollout_steps,
        ppo_config=config,
        device=args.device,
        stage=args.stage,
        inventory=inventory,
        seed=args.seed,
        init_checkpoint=args.init_checkpoint,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "stage", "budget", "cumulative_samples")},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
