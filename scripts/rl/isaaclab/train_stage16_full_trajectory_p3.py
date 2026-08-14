#!/usr/bin/env python3
"""Train one formal table-supported Stage16 P3 C0/C1/C2 lineage segment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
from scripts.rl.isaaclab.train_stage16_p3_physical_curriculum import (
    _gpu_probe,
    _restore_zero_g_checkpoint,
    _selected_zero_g_checkpoint,
)
from toporetarget.rl.full_trajectory_episode_start import validate_full_trajectory_start
from toporetarget.rl.full_trajectory_p3 import (
    FULL_TRAJECTORY_P3_RESULT_SCHEMA,
    checkpoint_metadata,
    validate_resume_metadata,
)
from toporetarget.rl.physical_p3 import physical_stage_budget
from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DTrainingConfigV1
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

START_ROOT = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart/episode_start"
SUPPORT_ROOT = REPO_ROOT / ".local/reports/stage16_support_reconstruction/inference"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart/formal"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument(
        "--contact-mode", choices=tuple(mode.value for mode in ContactRewardMode), required=True
    )
    parser.add_argument("--stage", choices=("C0", "C1", "C2"), required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser


def _start_and_hashes(clip: str) -> tuple[dict[str, Any], str, str]:
    start = validate_full_trajectory_start(
        json.loads((START_ROOT / f"{clip}.json").read_text(encoding="utf-8")), clip=clip
    )
    reference_hash = _sha256(REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz")
    support_hash = _sha256(SUPPORT_ROOT / clip / "table_proxy.json")
    if start["reference_hash"] != reference_hash:
        raise RuntimeError("FULL_TRAJECTORY_P3_REFERENCE_HASH_MISMATCH")
    if start["support_contract_hash"] != support_hash:
        raise RuntimeError("FULL_TRAJECTORY_P3_SUPPORT_HASH_MISMATCH")
    return start, support_hash, reference_hash


def _output_dir(root: Path, mode: ContactRewardMode, clip: str, stage: str) -> Path:
    directory = "v3" if mode is ContactRewardMode.AGGREGATE_V3 else "v4"
    return root / directory / clip / stage.lower()


def _validate_environment(contract: dict[str, Any], *, clip: str, stage: str) -> None:
    active = contract.get("ppo26d")
    physics = contract.get("gravity_friction_curriculum")
    if not isinstance(active, dict) or not isinstance(physics, dict):
        raise RuntimeError("FULL_TRAJECTORY_P3_ENVIRONMENT_CONTRACT_INVALID")
    if (
        active.get("fixed_clip") != clip
        or active.get("active_clip_ids") != [clip]
        or physics.get("stage") != stage
        or physics.get("support") != "finite_inferred_table_proxy_v1"
        or physics.get("table_actor_active") is not True
        or physics.get("mid_trajectory_rsi") != "disabled"
        or active.get("object_rollout_state_writes") != 0
        or active.get("wrist_root_state_writes_during_step") != 0
    ):
        raise RuntimeError("FULL_TRAJECTORY_P3_ENVIRONMENT_CONTRACT_INVALID")


def _restore_resume(
    trainer: PPO26DTrainer,
    *,
    checkpoint: Path,
    clip: str,
    mode: ContactRewardMode,
    stage: str,
    num_envs: int,
    start: dict[str, Any],
    support_hash: str,
    reference_hash: str,
) -> tuple[dict[str, Any], int]:
    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    metadata = validate_resume_metadata(
        payload,
        clip=clip,
        mode=mode,
        stage=stage,
        num_envs=num_envs,
        episode_start=start,
        support_contract_hash=support_hash,
        reference_hash=reference_hash,
    )
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(metadata["policy_training_samples"])
    restore_rng_state(payload["rng"])
    return (
        {
            "kind": "FULL_TRAJECTORY_DIRECT_PREDECESSOR_RESUME",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_stage": metadata["source_stage"],
            "critic_restored": True,
            "optimizer_restored": True,
            "normalizer_restored": True,
            "rng_restored": True,
        },
        int(metadata["cumulative_samples"]),
    )


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.num_envs <= 0:
        raise ValueError("FULL_TRAJECTORY_P3_NUM_ENVS_INVALID")
    mode = ContactRewardMode.parse(args.contact_mode)
    if args.stage == "C0" and args.resume_checkpoint is not None:
        raise ValueError("FULL_TRAJECTORY_P3_C0_RESUME_FORBIDDEN")
    if args.stage != "C0" and args.resume_checkpoint is None:
        raise ValueError("FULL_TRAJECTORY_P3_PREDECESSOR_REQUIRED")
    budget = physical_stage_budget(args.stage)
    if budget.additional_samples % args.num_envs:
        raise ValueError("FULL_TRAJECTORY_P3_BUDGET_ALIGNMENT_INVALID")
    start, support_hash, reference_hash = _start_and_hashes(args.clip)
    output = _output_dir(args.output_root.resolve(), mode, args.clip, args.stage)
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "gpu_preflight.json", {"preflight": _gpu_probe()})
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        env = _make_table_env(
            clip=args.clip,
            num_envs=args.num_envs,
            start_index=int(start["start_index"]),
            mode=mode,
            stage=args.stage,
        )
        env.reset(seed=20260814)
        environment = env.contract_report()
        _validate_environment(environment, clip=args.clip, stage=args.stage)
        trainer = PPO26DTrainer(observation_dim=764, device=str(env.device))
        if args.stage == "C0":
            selected = _selected_zero_g_checkpoint(mode, args.clip)
            initialization = _restore_zero_g_checkpoint(
                trainer,
                checkpoint=Path(str(selected["selected_checkpoint"])),
                clip=args.clip,
                mode=mode,
            )
            initialization["selection"] = selected
            initialization["fresh_physical_restart"] = True
            cumulative_samples = 0
        else:
            assert args.resume_checkpoint is not None
            initialization, cumulative_samples = _restore_resume(
                trainer,
                checkpoint=args.resume_checkpoint.resolve(),
                clip=args.clip,
                mode=mode,
                stage=args.stage,
                num_envs=args.num_envs,
                start=start,
                support_hash=support_hash,
                reference_hash=reference_hash,
            )
        config = {
            "schema_version": "Stage16FullTrajectoryP3TrainingConfigV1",
            "clip": args.clip,
            "contact_mode": mode.value,
            "curriculum_stage": args.stage,
            "stage_budget_samples": budget.additional_samples,
            "selected_num_envs": args.num_envs,
            "episode_start": start,
            "support_contract_hash": support_hash,
            "reference_hash": reference_hash,
            "environment": environment,
            "initialization": initialization,
            "old_c2_checkpoint_resumed": False,
        }
        _write(output / "training_config.json", config)
        metrics_path = output / "training_metrics.jsonl"
        stage_samples = 0
        checkpoint: Path | None = None
        finite = True
        with metrics_path.open("w", encoding="utf-8") as stream:
            while stage_samples < budget.additional_samples:
                remaining = budget.additional_samples - stage_samples
                rollout_length = min(
                    Stage16DPPO26DTrainingConfigV1().rollout_length,
                    remaining // args.num_envs,
                )
                metric = trainer.collect_and_update(env, rollout_length=rollout_length)
                metric.pop("last_policy_observation")
                stage_samples += int(metric["samples"])
                cumulative_samples += int(metric["samples"])
                finite &= bool(metric["finite"])
                metric.update(
                    {
                        "curriculum_stage": args.stage,
                        "stage_samples": stage_samples,
                        "cumulative_samples": cumulative_samples,
                        "policy_training_samples": trainer.cumulative_samples,
                    }
                )
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
                stream.flush()
        metadata = checkpoint_metadata(
            stage=args.stage,
            stage_samples=stage_samples,
            cumulative_samples=cumulative_samples,
            policy_training_samples=trainer.cumulative_samples,
            mode=mode,
            episode_start=start,
            support_contract_hash=support_hash,
            reference_hash=reference_hash,
        )
        payload = trainer.checkpoint_payload(
            environment_contract=env.contract_report(), selected_num_envs=args.num_envs
        )
        payload.pop("cumulative_samples", None)
        payload.update(metadata)
        payload["clip"] = args.clip
        payload["selected_num_envs"] = args.num_envs
        checkpoint = (
            output / "checkpoints" / f"stage16_full_trajectory_{mode.value}_{args.stage.lower()}.pt"
        )
        save_checkpoint(checkpoint, payload)
        writes = env.rollout_state_write_report()
        passed = (
            finite
            and stage_samples == budget.additional_samples
            and int(writes["object_rollout_state_writes"]) == 0
            and int(writes["wrist_root_state_writes_during_step"]) == 0
        )
        result = {
            "schema_version": FULL_TRAJECTORY_P3_RESULT_SCHEMA,
            "status": "P3_FULL_TRAJECTORY_STAGE_COMPLETE" if passed else "FAIL",
            "clip": args.clip,
            "contact_mode": mode.value,
            "curriculum_stage": args.stage,
            "stage_samples": stage_samples,
            "cumulative_samples": cumulative_samples,
            "policy_training_samples": trainer.cumulative_samples,
            "finite": finite,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "episode_start": start,
            "support_contract_hash": support_hash,
            "reference_hash": reference_hash,
            "environment": env.contract_report(),
            "rollout_writes": writes,
            "initialization": initialization,
        }
        _write(output / "training_result.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0 if passed else 2
    except BaseException as error:
        _write(
            output / "training_failure.json",
            {
                "schema_version": "Stage16FullTrajectoryP3FailureV1",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
