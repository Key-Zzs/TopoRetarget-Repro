#!/usr/bin/env python3
"""Train the first real single-clip Stage 16-D.5 PPO-26D policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DTrainingConfigV1  # noqa: E402
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), default="hocap_170650")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--selected-capacity", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--no-critical-dr", action="store_false", dest="critical_dr")
    parser.set_defaults(critical_dr=True)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_num_envs(args: argparse.Namespace) -> int:
    if args.num_envs is not None:
        return args.num_envs
    path = args.selected_capacity or args.output_root / "gpu" / "selected_capacity.json"
    return int(json.loads(path.read_text(encoding="utf-8"))["selected_num_envs"])


def pretraining_gpu_probe() -> dict[str, float | str]:
    """Require the same real host headroom rule immediately before PPO starts."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"PPO26D_GPU_PROBE_FAILED: {result.stderr.strip()}")
    values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
    if len(values) != 5:
        raise RuntimeError("PPO26D_GPU_PROBE_INVALID")
    name, total, used, free, utilization = values
    try:
        payload: dict[str, float | str] = {
            "name": name,
            "total_vram_mib": float(total),
            "used_vram_mib": float(used),
            "free_vram_mib": float(free),
            "gpu_utilization_percent": float(utilization),
        }
    except ValueError as error:
        raise RuntimeError("PPO26D_GPU_PROBE_INVALID") from error
    required = max(2048.0, float(payload["total_vram_mib"]) * 0.15)
    payload["required_headroom_mib"] = required
    payload["headroom_pass"] = float(payload["free_vram_mib"]) >= required
    if not payload["headroom_pass"]:
        raise RuntimeError("PPO26D_GPU_HEADROOM_INSUFFICIENT")
    return payload


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )

    root = args.output_root.resolve()
    output = root / args.clip
    gpu_probe = pretraining_gpu_probe()
    write_json(output / "gpu_before_training.json", gpu_probe)
    num_envs = selected_num_envs(args)
    contract = Stage16DPPO26DTrainingConfigV1()
    samples_per_iteration = num_envs * contract.rollout_length
    required_iterations = math.ceil(1_000_000 / samples_per_iteration)
    iterations = max(2, required_iterations if args.iterations is None else args.iterations)
    if iterations * samples_per_iteration < 1_000_000:
        raise ValueError("requested iterations do not satisfy the L0 minimum of 1,000,000 samples")
    app = AppLauncher(headless=True).app
    env = None
    try:
        cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo26d_cfg.configure_stage16d_ppo26d(
            cfg,
            num_envs=num_envs,
            clip=args.clip,
            rsi=True,
            critical_dr=args.critical_dr,
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        write_json(
            output / "training_config.json",
            {
                "clip": args.clip,
                "selected_num_envs": num_envs,
                "samples_per_iteration": samples_per_iteration,
                "required_iterations": required_iterations,
                "iterations": iterations,
                "critical_dr": args.critical_dr,
                "contract": contract.as_dict(),
                "environment": env.contract_report(),
            },
        )
        metrics_path = output / "training_metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as stream:
            for iteration in range(1, iterations + 1):
                metric = trainer.collect_and_update(env)
                last_observation = metric.pop("last_policy_observation")
                metric.update({"iteration": iteration, "clip": args.clip})
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
                stream.flush()
                if iteration == 2:
                    smoke_checkpoint = trainer.save(
                        output / "smoke_checkpoint.pt",
                        environment_contract=env.contract_report(),
                        selected_num_envs=num_envs,
                    )
                    write_json(
                        output / "smoke_checkpoint_reload.json",
                        trainer.reload_deterministic_action(smoke_checkpoint, last_observation),
                    )
                if iteration == iterations:
                    checkpoint = trainer.save(
                        output / f"stage16d_ppo26d_{args.clip.removeprefix('hocap_')}_l0.pt",
                        environment_contract=env.contract_report(),
                        selected_num_envs=num_envs,
                    )
                    reload_result = trainer.reload_deterministic_action(
                        checkpoint, last_observation
                    )
        result = {
            "schema_version": "Stage16DPPO26DL0TrainingV1",
            "status": "STAGE16D_PPO26D_L0_COMPLETE_NOT_YET_QUALIFIED",
            "clip": args.clip,
            "iterations": iterations,
            "cumulative_samples": trainer.cumulative_samples,
            "samples_per_iteration": samples_per_iteration,
            "smoke_checkpoint": str((output / "smoke_checkpoint.pt").resolve()),
            "smoke_checkpoint_reload": str((output / "smoke_checkpoint_reload.json").resolve()),
            "l0_checkpoint": str(checkpoint.resolve()),
            "checkpoint_reload": reload_result,
            "metrics": str(metrics_path.resolve()),
        }
        write_json(output / "l0_training.json", result)
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
