#!/usr/bin/env python3
"""Gate-controlled 26-D PPO training for Stage-16B world-wrist tracking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from toporetarget.rl.environments.world_wrist_backend import (
    WorldWristFingerBackend,
    WristFingerActionScaleV1,
    WristImpedanceProfileV1,
    materialize_world_wrist_free_object_scene,
)
from toporetarget.rl.ppo.checkpoint import load_checkpoint, rng_state, save_checkpoint
from toporetarget.rl.ppo.storage import RolloutStorage
from toporetarget.rl.ppo.trainer import PPOConfig, PPOTrainer
from toporetarget.rl.world_wrist import WorldWristFingerReferenceV1

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
SINGLE_SAMPLE_LADDER = (32_768, 131_072, 524_288, 2_097_152, 8_388_608)
TWO_CLIP_SAMPLE_LADDER = (131_072, 524_288, 2_097_152, 8_388_608, 16_777_216)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _profile_from_report(
    report: dict[str, Any],
) -> tuple[WristImpedanceProfileV1, WristFingerActionScaleV1]:
    controller = report["selected"]["profile"]
    scale = (
        json.loads(
            (
                Path(report["action_scale_report"])
                if "action_scale_report" in report
                else Path("/dev/null")
            ).read_text(encoding="utf-8")
        )
        if "action_scale_report" in report
        else None
    )
    if scale is None:
        raise ValueError("qualification report must record selected global action scale")
    return (
        WristImpedanceProfileV1(
            translation_stiffness_npm=float(controller["translation_stiffness_npm"]),
            translation_damping_ratio=float(controller["translation_damping_ratio"]),
            rotation_stiffness_nmprad=float(controller["rotation_stiffness_nmprad"]),
            rotation_damping_ratio=float(controller["rotation_damping_ratio"]),
            force_limit_n=float(controller["force_limit_n"]),
            torque_limit_nm=float(controller["torque_limit_nm"]),
            feedforward_twist_gain=float(controller["feedforward_twist_gain"]),
        ),
        WristFingerActionScaleV1(
            translation_m=float(scale["selected"]["scale"]["translation_m"]),
            rotation_rad=float(scale["selected"]["scale"]["rotation_rad"]),
            finger_joint_range_fraction=float(
                scale["selected"]["scale"]["finger_joint_range_fraction"]
            ),
        ),
    )


def build_backends(
    references: list[Path],
    meshes: list[Path],
    scene_root: Path,
    *,
    impedance: WristImpedanceProfileV1,
    action_scale: WristFingerActionScaleV1,
    seed: int,
) -> list[WorldWristFingerBackend]:
    model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    result: list[WorldWristFingerBackend] = []
    for index, (reference_path, mesh_path) in enumerate(zip(references, meshes, strict=True)):
        reference = WorldWristFingerReferenceV1.from_npz(reference_path)
        scene = materialize_world_wrist_free_object_scene(
            WUJI_MJCF, scene_root / reference_path.stem, object_mesh=mesh_path
        )
        result.append(
            WorldWristFingerBackend(
                scene_path=scene,
                reference=reference,
                joint_lower=model.jnt_range[: model.njnt, 0],
                joint_upper=model.jnt_range[: model.njnt, 1],
                impedance_profile=impedance,
                action_scale=action_scale,
                seed=seed + index,
            )
        )
    return result


def train(
    *,
    backends: list[WorldWristFingerBackend],
    checkpoint_directory: Path,
    budget: int,
    rollout_steps: int,
    device: str,
    seed: int,
    stage: str,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if not backends:
        raise ValueError("at least one backend is required")
    if rollout_steps * len(backends) % 32:
        raise ValueError("rollout sample count must divide exactly into 32 PPO minibatches")
    torch.manual_seed(seed)
    np.random.seed(seed)
    observations = [backend.observation(backend.reset()) for backend in backends]
    trainer = PPOTrainer(
        observations[0].size, 26, config=PPOConfig(epochs=4, minibatches=32), device=device
    )
    if init_checkpoint is not None:
        payload = load_checkpoint(init_checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(payload["model"])
        trainer.normalizer.load_state_dict(payload["normalizer"])
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    total_samples = 0
    iterations = 0
    traces: list[dict[str, Any]] = []
    while total_samples < budget:
        remaining_per_backend = (budget - total_samples) // len(backends)
        current_rollout_steps = min(rollout_steps, remaining_per_backend)
        if current_rollout_steps < 1 or current_rollout_steps * len(backends) % 32:
            raise ValueError("frozen PPO budget cannot be divided into 32 minibatches")
        iterations += 1
        trainer.update_observation_normalizer(torch.as_tensor(np.asarray(observations)))
        rollout = {
            key: []
            for key in ("observations", "actions", "log_probs", "rewards", "dones", "values")
        }
        returns = np.zeros(len(backends), dtype=np.float64)
        for _ in range(current_rollout_steps):
            observation_tensor = torch.as_tensor(
                np.asarray(observations), dtype=torch.float32, device=trainer.device
            )
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
                dones.append(reason is not None)
                if reason is not None:
                    state = backend.reset()
                next_observations.append(backend.observation(state))
            rollout["observations"].append(observation_tensor.cpu())
            rollout["actions"].append(actions.cpu())
            rollout["log_probs"].append(log_probs.cpu())
            rollout["rewards"].append(torch.as_tensor(rewards, dtype=torch.float32))
            rollout["dones"].append(torch.as_tensor(dones, dtype=torch.bool))
            rollout["values"].append(values.cpu())
            observations = next_observations
        storage = RolloutStorage(**{key: torch.stack(value) for key, value in rollout.items()})
        with torch.no_grad():
            _, _, last_value = trainer.act(
                torch.as_tensor(
                    np.asarray(observations), dtype=torch.float32, device=trainer.device
                ),
                deterministic=True,
            )
        update = trainer.update(storage, last_value.cpu())
        total_samples += storage.sample_count
        payload = {
            "kind": "stage16b_world_wrist_finger_ppo",
            "stage": stage,
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "normalizer": trainer.normalizer.state_dict(),
            "rng": rng_state(),
            "cumulative_samples": total_samples,
            "iteration": iterations,
            "action_dim": 26,
            "references": [str(backend.reference.content_hash()) for backend in backends],
            "world_wrist_profile": "world_wrist_finger_residual_v1",
            "ppo_config": PPOConfig(epochs=4, minibatches=32).as_dict(),
            "nominal_no_dr": True,
            "random_k0_training": True,
        }
        checkpoint = save_checkpoint(
            checkpoint_directory / f"checkpoint_{total_samples:09d}.pt", payload
        )
        save_checkpoint(checkpoint_directory / "last.pt", payload)
        traces.append(
            {
                "iteration": iterations,
                "samples": storage.sample_count,
                "cumulative_samples": total_samples,
                "mean_return": float(np.mean(returns)),
                "update": update,
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    elapsed = time.monotonic() - started
    return {
        "status": "STAGE16B_TRAINING_BUDGET_COMPLETE",
        "stage": stage,
        "cumulative_samples": total_samples,
        "iterations": iterations,
        "samples_per_second": total_samples / max(elapsed, 1e-9),
        "wall_seconds": elapsed,
        "checkpoint_last": str((checkpoint_directory / "last.pt").resolve()),
        "checkpoint_reload_samples": load_checkpoint(checkpoint_directory / "last.pt")[
            "cumulative_samples"
        ],
        "traces": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--checkpoint-directory", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--oracle-report", required=True, type=Path)
    parser.add_argument("--controller-report", required=True, type=Path)
    parser.add_argument("--action-scale-report", required=True, type=Path)
    parser.add_argument("--stage", choices=("single", "two_clip"), required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--rollout-steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", type=Path)
    args = parser.parse_args()
    if len(args.reference) != len(args.object_mesh):
        raise ValueError("reference/object-mesh count must match")
    expected_count = 1 if args.stage == "single" else 2
    if len(args.reference) != expected_count:
        raise ValueError(f"{args.stage} stage requires exactly {expected_count} clip(s)")
    ladder = SINGLE_SAMPLE_LADDER if args.stage == "single" else TWO_CLIP_SAMPLE_LADDER
    if args.budget not in ladder:
        raise ValueError(f"budget must be in frozen {args.stage} ladder: {ladder}")
    oracle_report = json.loads(args.oracle_report.read_text(encoding="utf-8"))
    if oracle_report.get("status") != "STAGE16B_26D_ORACLE_VALIDATED":
        raise ValueError(
            "Stage16B single/two-clip PPO is gate-blocked: "
            f"26-D oracle status is {oracle_report.get('status', 'MISSING')}"
        )
    controller_report = json.loads(args.controller_report.read_text(encoding="utf-8"))
    scale_report = json.loads(args.action_scale_report.read_text(encoding="utf-8"))
    if controller_report.get("status") != "STAGE16B_WRIST_CONTROL_VALIDATED":
        raise ValueError("Stage16B PPO requires a validated wrist controller profile")
    profile = {**controller_report, "action_scale_report": str(args.action_scale_report.resolve())}
    impedance, action_scale = _profile_from_report(profile)
    # Field-based comparison deliberately permits report metadata to grow.
    selected = scale_report.get("selected", {}).get("scale", {})
    if any(
        selected.get(key) != value for key, value in action_scale.as_dict().items() if key != "id"
    ):
        raise ValueError("selected action-scale report is inconsistent")
    backends = build_backends(
        args.reference,
        args.object_mesh,
        args.scene_root,
        impedance=impedance,
        action_scale=action_scale,
        seed=args.seed,
    )
    report = train(
        backends=backends,
        checkpoint_directory=args.checkpoint_directory,
        budget=args.budget,
        rollout_steps=args.rollout_steps,
        device=args.device,
        seed=args.seed,
        stage=args.stage,
        init_checkpoint=args.init_checkpoint,
    )
    _write_json(args.report, report)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "stage", "cumulative_samples")}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
