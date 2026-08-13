#!/usr/bin/env python3
"""Resume one bounded Stage 16-D PPO-26D checkpoint without changing its contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state  # noqa: E402
from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DTrainingConfigV1  # noqa: E402
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d_continuation"
MAX_V1_CUMULATIVE_SAMPLES = 67_108_864


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--target-cumulative-samples", type=int, required=True)
    parser.add_argument("--checkpoint-at", type=int, action="append", default=[])
    parser.add_argument("--stage", default="r6a")
    parser.add_argument("--curriculum-manifest", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--no-critical-dr", action="store_false", dest="critical_dr")
    parser.set_defaults(critical_dr=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_curriculum(path: Path | None, *, clip: str) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PPO26D_R6C_MANIFEST_INVALID")
    if payload.get("contract") != "Stage16DPPO26DRSICurriculumV1" or payload.get("clip") != clip:
        raise ValueError("PPO26D_R6C_MANIFEST_CONTRACT_DRIFT")
    phase = payload.get("phase")
    indices = payload.get("reference_indices")
    probabilities = payload.get("probabilities")
    if (
        phase not in {"C0", "C1", "C2"}
        or not isinstance(indices, list)
        or not isinstance(probabilities, list)
    ):
        raise ValueError("PPO26D_R6C_MANIFEST_FIELDS_INVALID")
    if len(indices) == 0 or len(indices) != len(probabilities):
        raise ValueError("PPO26D_R6C_MANIFEST_DISTRIBUTION_INVALID")
    weights = [float(value) for value in probabilities]
    if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1.0e-6:
        raise ValueError("PPO26D_R6C_MANIFEST_PROBABILITIES_INVALID")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "phase": phase,
        "reference_indices": tuple(int(value) for value in indices),
        "probabilities": tuple(weights),
    }


def gpu_probe() -> dict[str, object]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(query, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"HOST_GPU_PROBE_BLOCKED:{result.stderr.strip()}")
    pmon = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1"], capture_output=True, text=True, check=False
    )
    fields = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 11:
        raise RuntimeError("PPO26D_GPU_PROBE_INVALID")
    return {"query": result.stdout.strip(), "pmon": pmon.stdout.strip(), "fields": fields}


def restore_trainer(payload: dict[str, Any], *, device: str) -> PPO26DTrainer:
    contract = Stage16DPPO26DTrainingConfigV1().as_dict()
    if payload.get("schema_version") != "Stage16DPPO26DCheckpointV1":
        raise ValueError("PPO26D_RESUME_CHECKPOINT_SCHEMA_INVALID")
    if payload.get("ppo_config") != contract:
        raise ValueError("PPO26D_RESUME_CONTRACT_DRIFT")
    trainer = PPO26DTrainer(observation_dim=764, device=device)
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.cumulative_samples = int(payload["cumulative_samples"])
    restore_rng_state(payload["rng"])
    return trainer


def checkpoint_name(*, clip: str, cumulative_samples: int) -> str:
    short = clip.removeprefix("hocap_")
    return f"stage16d_ppo26d_{short}_samples_{cumulative_samples}.pt"


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.target_cumulative_samples > MAX_V1_CUMULATIVE_SAMPLES:
        raise ValueError("PPO26D cumulative V1 budget exceeds 67,108,864 samples")
    checkpoint_path = args.checkpoint.resolve()
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    if payload.get("clip") != args.clip:
        raise ValueError("PPO26D_RESUME_CHECKPOINT_CLIP_MISMATCH")
    checkpoint_samples = int(payload["cumulative_samples"])
    if args.target_cumulative_samples <= checkpoint_samples:
        raise ValueError("target cumulative samples must exceed the resumed checkpoint")
    selected_num_envs = int(args.num_envs or payload["selected_num_envs"])
    if selected_num_envs != int(payload["selected_num_envs"]):
        raise ValueError("PPO26D_RESUME_NUM_ENVS_CONTRACT_DRIFT")
    contract = Stage16DPPO26DTrainingConfigV1()
    samples_per_iteration = selected_num_envs * contract.rollout_length
    if not args.stage.replace("_", "").isalnum():
        raise ValueError("PPO26D_STAGE_IDENTIFIER_INVALID")
    curriculum = load_curriculum(args.curriculum_manifest, clip=args.clip)
    output = args.output_root.resolve() / args.clip / args.stage
    probe = gpu_probe()
    free_vram_mib = float(probe["fields"][6])
    if free_vram_mib < 2048.0:
        raise RuntimeError("PPO26D_GPU_HEADROOM_INSUFFICIENT")
    if not args.critical_dr:
        raise ValueError("PPO26D_R6A_CRITICAL_DR_CONTRACT_DRIFT")
    requested_checkpoints = sorted(
        {
            value
            for value in [*args.checkpoint_at, args.target_cumulative_samples]
            if checkpoint_samples < value <= args.target_cumulative_samples
        }
    )
    write_json(
        output / "resume_contract.json",
        {
            "schema_version": "Stage16DPPO26DResumeContractV1",
            "clip": args.clip,
            "source_checkpoint": str(checkpoint_path),
            "source_cumulative_samples": checkpoint_samples,
            "target_cumulative_samples": args.target_cumulative_samples,
            "selected_num_envs": selected_num_envs,
            "samples_per_iteration": samples_per_iteration,
            "critical_dr": args.critical_dr,
            "stage": args.stage,
            "curriculum": curriculum,
            "ppo_contract": contract.as_dict(),
            "gpu_before_training": probe,
        },
    )
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    checkpoints: list[dict[str, object]] = []
    try:
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo26d_cfg.configure_stage16d_ppo26d(
            cfg,
            num_envs=selected_num_envs,
            clip=args.clip,
            rsi=True,
            critical_dr=True,
            curriculum_reference_indices=(
                None if curriculum is None else curriculum["reference_indices"]
            ),
            curriculum_reference_probabilities=(
                None if curriculum is None else curriculum["probabilities"]
            ),
            curriculum_phase=None if curriculum is None else str(curriculum["phase"]),
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        trainer = restore_trainer(payload, device=str(env.device))
        expected_clip_index = env.reference_bank.clip_ids.index(args.clip)
        active = sorted(set(env._clip_index.detach().cpu().tolist()))
        if active != [expected_clip_index]:
            raise RuntimeError("PPO26D_FIXED_CLIP_MISMATCH")
        metrics_path = output / "training_metrics.jsonl"
        prior_iterations = 0
        if metrics_path.is_file():
            prior_iterations = sum(
                1 for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
            )
        next_checkpoint = 0
        with metrics_path.open("a", encoding="utf-8") as stream:
            while trainer.cumulative_samples < args.target_cumulative_samples:
                metric = trainer.collect_and_update(env)
                last_observation = metric.pop("last_policy_observation")
                metric.update(
                    {
                        "iteration": prior_iterations + 1,
                        "clip": args.clip,
                        "resumed_from_samples": checkpoint_samples,
                    }
                )
                prior_iterations += 1
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
                stream.flush()
                while (
                    next_checkpoint < len(requested_checkpoints)
                    and trainer.cumulative_samples >= requested_checkpoints[next_checkpoint]
                ):
                    requested = requested_checkpoints[next_checkpoint]
                    path = output / checkpoint_name(
                        clip=args.clip, cumulative_samples=trainer.cumulative_samples
                    )
                    saved = trainer.save(
                        path,
                        environment_contract=env.contract_report(),
                        selected_num_envs=selected_num_envs,
                    )
                    checkpoints.append(
                        {
                            "requested_cumulative_samples": requested,
                            "actual_cumulative_samples": trainer.cumulative_samples,
                            "checkpoint": str(saved.resolve()),
                            "reload": trainer.reload_deterministic_action(saved, last_observation),
                        }
                    )
                    next_checkpoint += 1
        result = {
            "schema_version": "Stage16DPPO26DResumeTrainingV1",
            "status": f"{args.stage.upper()}_TARGET_REACHED",
            "stage": args.stage,
            "clip": args.clip,
            "source_checkpoint": str(checkpoint_path),
            "source_cumulative_samples": checkpoint_samples,
            "target_cumulative_samples": args.target_cumulative_samples,
            "actual_cumulative_samples": trainer.cumulative_samples,
            "selected_num_envs": selected_num_envs,
            "samples_per_iteration": samples_per_iteration,
            "metrics": str(metrics_path.resolve()),
            "checkpoints": checkpoints,
            "curriculum": curriculum,
            "environment": env.contract_report(),
        }
        write_json(output / "training.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
