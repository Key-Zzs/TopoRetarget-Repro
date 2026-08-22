#!/usr/bin/env python3
"""Train a bounded Stage 16-D Phase 3 signed-object-twist reward experiment."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DTrainingConfigV1
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer, parameter_hash
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV2,
)

PHASE3_CHECKPOINT_SCHEMA = "Stage16DPhase3RewardV2CheckpointV1"
REWARD_V3_CHECKPOINT_SCHEMA = "Stage16DRewardV3CheckpointV1"
STRICT_V4_CHECKPOINT_SCHEMA = "Stage16DStrictPerFingerV4CheckpointV1"
L0_SAMPLES = 1_024_000
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"
DEFAULT_REFERENCE_ROOT = DEFAULT_ROOT / "references"
DEFAULT_V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
DEFAULT_V3_CONTACT_CONTRACT = DEFAULT_V3_ROOT / "reward_v3_contract.json"
DEFAULT_V3_CONTACT_MASK_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_contact"
DEFAULT_V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
DEFAULT_V4_CONTACT_CONTRACT = DEFAULT_V4_ROOT / "strict_v4_contract.json"
DEFAULT_L0_CHECKPOINT_170650 = (
    REPO_ROOT
    / ".local/reports/stage16d_ppo26d_clip_repair/hocap_170650/stage16d_ppo26d_170650_l0.pt"
)
DEFAULT_L0_CHECKPOINT_170105 = (
    REPO_ROOT
    / ".local/reports/stage16d_ppo26d_continuation/hocap_170105/stage16d_ppo26d_170105_l0.pt"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_probe() -> dict[str, str]:
    commands = {
        "nvidia_smi": ["nvidia-smi"],
        "query": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        "pmon": ["nvidia-smi", "pmon", "-c", "1"],
    }
    outputs: dict[str, str] = {}
    for name, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"PHASE3_GPU_PROBE_FAILED:{name}:{result.stderr.strip()}")
        outputs[name] = result.stdout.strip()
    return outputs


def _require_phase3_entry(root: Path, *, clip: str) -> dict[str, Any]:
    decision_path = root / "phase3_entry_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    authorization = decision.get("authorization", {})
    if (
        decision.get("status") != "PHASE3_OBJECT_TWIST_REWARD_RECOMMENDED"
        or not authorization.get("phase3_implementation_and_training_authorized")
        or authorization.get("only_clip_authorized") != clip
    ):
        raise RuntimeError("PHASE3_NOT_AUTHORIZED")
    return {
        "path": str(decision_path),
        "sha256": _sha256(decision_path),
        "status": decision["status"],
    }


def _require_reward_v3_entry(contract_path: Path) -> dict[str, Any]:
    """Refuse V3 PPO unless its exact-pair-force preflight is frozen and authorized."""

    resolved_contract = contract_path.resolve()
    contract = json.loads(resolved_contract.read_text(encoding="utf-8"))
    parameters = contract.get("frozen_parameters")
    if (
        contract.get("status") != "CONTACT_REWARD_CONTRACT_FROZEN"
        or not isinstance(parameters, dict)
        or not isinstance(parameters.get("lambda_c_n"), (int, float))
        or float(parameters["lambda_c_n"]) <= 1.0e-5
    ):
        raise RuntimeError("REWARD_V3_CONTACT_CONTRACT_NOT_FROZEN")
    preflight_path = resolved_contract.parent / "reward_v3_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "CONTACT_REWARD_CONTRACT_FROZEN"
        or preflight.get("training_authorized") is not True
    ):
        raise RuntimeError("REWARD_V3_CONTACT_PREFLIGHT_NOT_AUTHORIZED")
    return {
        "contract": {
            "path": str(resolved_contract),
            "sha256": _sha256(resolved_contract),
            "lambda_c_n": float(parameters["lambda_c_n"]),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": _sha256(preflight_path),
            "training_authorized": True,
        },
    }


def _require_strict_v4_entry(contract_path: Path) -> dict[str, Any]:
    """Refuse V4 PPO unless source masks and V1-only calibration are frozen."""

    resolved = contract_path.resolve()
    contract = json.loads(resolved.read_text(encoding="utf-8"))
    parameters = contract.get("frozen_parameters")
    frozen_inputs = resolved.parent / "frozen_inputs.json"
    calibration = resolved.parent / "strict_v4_force_scale_calibration.json"
    if (
        contract.get("status") != "STRICT_V4_CONTACT_CONTRACT_FROZEN"
        or not isinstance(parameters, dict)
        or not isinstance(parameters.get("lambda_tip_n"), (int, float))
        or float(parameters["lambda_tip_n"]) <= 1.0e-5
        or not frozen_inputs.is_file()
        or not calibration.is_file()
    ):
        raise RuntimeError("STRICT_V4_CONTACT_PREFLIGHT_NOT_AUTHORIZED")
    if json.loads(calibration.read_text(encoding="utf-8")).get("status") != (
        "STRICT_V4_CONTACT_CONTRACT_FROZEN"
    ):
        raise RuntimeError("STRICT_V4_CALIBRATION_NOT_FROZEN")
    return {
        "contract": {
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "lambda_tip_n": float(parameters["lambda_tip_n"]),
        },
        "frozen_inputs": {"path": str(frozen_inputs), "sha256": _sha256(frozen_inputs)},
        "calibration": {"path": str(calibration), "sha256": _sha256(calibration)},
    }


def _freeze_scales(reference_root: Path) -> dict[str, Any]:
    """Verify the committed V2 scale constants against pooled V2 dynamics."""

    twists: list[np.ndarray] = []
    sources: dict[str, dict[str, str]] = {}
    for clip in ("hocap_170105", "hocap_170650"):
        path = reference_root / f"{clip}.reference_kinematics_v2.npz"
        if not path.is_file():
            raise FileNotFoundError(f"PHASE3_REFERENCE_V2_MISSING:{path}")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if int(metadata.get("reference_kinematics_version", -1)) != 2:
                raise ValueError("PHASE3_REWARD_V2_REJECTS_V1_REFERENCE")
            twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
        if twist.shape != (321, 6) or not np.isfinite(twist).all():
            raise ValueError(f"PHASE3_REFERENCE_V2_TWIST_INVALID:{path}")
        twists.append(twist)
        sources[clip] = {"path": str(path), "sha256": _sha256(path)}
    pooled = np.concatenate(twists, axis=0)
    linear_p95 = float(np.percentile(np.linalg.norm(pooled[:, :3], axis=-1), 95))
    angular_p95 = float(np.percentile(np.linalg.norm(pooled[:, 3:], axis=-1), 95))
    profile = TopoRetargetReferenceTrackingReward26DV2()
    expected_linear = 0.075
    expected_angular = 0.125
    if (
        profile.object_velocity_sigma_mps != expected_linear
        or profile.object_angular_velocity_sigma_radps != expected_angular
    ):
        raise RuntimeError("PHASE3_REWARD_V2_SCALE_CODE_CONFIG_DRIFT")
    return {
        "schema_version": "Stage16DPhase3RewardV2ScaleFreezeV1",
        "status": "FROZEN_BEFORE_PPO",
        "reference_kinematics_version": 2,
        "sources": sources,
        "pooled_reference_statistics": {
            "linear_speed_p95_mps": linear_p95,
            "angular_speed_p95_radps": angular_p95,
        },
        "existing_terminal_stability_scales": {
            "free_object_linear_mps": 0.01,
            "free_object_angular_radps": 0.25,
        },
        "selection_rule": {
            "sigma_v_mps": "round_up(max(pooled_p95_linear, 5 * free_object_linear), 0.005)",
            "sigma_omega_radps": "max(pooled_p95_angular, 0.5 * free_object_angular)",
        },
        "reward_profile": profile.as_dict(),
        "weight_bound": {
            "combined_twist_maximum": (
                profile.object_velocity_weight + profile.object_angular_velocity_weight
            ),
            "required_at_most": 2.0,
            "current_object_pose_maximum": 8.0,
            "at_most_25_percent_of_object_pose": True,
        },
        "forbidden_terms": [
            "contact_reward",
            "terminal_reward",
            "penetration_reward",
            "guidance_reward",
            "gravity_reward",
        ],
    }


def _load_l0_actor_and_normalizer(
    trainer: PPO26DTrainer, checkpoint: Path, *, expected_clip: str
) -> dict[str, Any]:
    """Load only V1 actor/log-std and normalizer; preserve fresh critic/optimizer."""

    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    if payload.get("schema_version") != "Stage16DPPO26DCheckpointV1":
        raise ValueError("PHASE3_L0_CHECKPOINT_SCHEMA_INVALID")
    if int(payload.get("cumulative_samples", -1)) != L0_SAMPLES:
        raise ValueError("PHASE3_L0_CHECKPOINT_SAMPLE_COUNT_INVALID")
    source_clip = payload.get("clip")
    if source_clip not in {None, expected_clip}:
        raise ValueError("PHASE3_L0_CHECKPOINT_CLIP_MISMATCH")
    source_reward = payload.get("environment_contract", {}).get("ppo26d", {}).get("reward", {})
    if source_reward.get("identifier") != "TopoRetargetReferenceTrackingReward26DV1":
        raise ValueError("PHASE3_L0_CHECKPOINT_MUST_USE_V1_REWARD")
    source_state = payload["actor_critic"]
    target_state = trainer.model.state_dict()
    actor_keys = [
        name for name in target_state if name.startswith("actor.") or name == "log_std_parameter"
    ]
    if set(actor_keys).difference(source_state):
        raise ValueError("PHASE3_L0_ACTOR_KEYS_MISSING")
    critic_before = parameter_hash(trainer.model, "critic")
    target_state.update({name: source_state[name] for name in actor_keys})
    trainer.model.load_state_dict(target_state, strict=True)
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    actor_after = parameter_hash(trainer.model, "actor")
    source_actor_digest = hashlib.sha256()
    for name in sorted(actor_keys):
        source_actor_digest.update(name.encode("utf-8"))
        source_actor_digest.update(source_state[name].detach().cpu().contiguous().numpy().tobytes())
    if actor_after != source_actor_digest.hexdigest():
        raise RuntimeError("PHASE3_ACTOR_INITIALIZATION_HASH_MISMATCH")
    if trainer.trainer.optimizer.state:
        raise RuntimeError("PHASE3_OPTIMIZER_WAS_NOT_FRESH")
    return {
        "initialization": "V1_L0_ACTOR_AND_NORMALIZER_ONLY",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_schema": payload["schema_version"],
        "source_clip": expected_clip if source_clip is None else source_clip,
        "source_clip_metadata_absent": source_clip is None,
        "v1_l0_cumulative_samples": L0_SAMPLES,
        "actor_log_std_hash": actor_after,
        "critic_hash_fresh": critic_before,
        "critic_loaded_from_v1": False,
        "optimizer_loaded_from_v1": False,
        "observation_normalization_loaded_from_v1": True,
        "comparison_initialization_deviation": None,
    }


def _resume_phase3(
    trainer: PPO26DTrainer, checkpoint: Path, *, expected_clip: str, expected_num_envs: int
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    if payload.get("schema_version") != PHASE3_CHECKPOINT_SCHEMA:
        raise ValueError("PHASE3_RESUME_CHECKPOINT_SCHEMA_INVALID")
    if payload.get("clip") != expected_clip:
        raise ValueError("PHASE3_RESUME_CHECKPOINT_CLIP_MISMATCH")
    if int(payload.get("selected_num_envs", -1)) != expected_num_envs:
        raise ValueError("PHASE3_RESUME_CHECKPOINT_ENV_COUNT_MISMATCH")
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(payload["reward_v2_samples"])
    restore_rng_state(payload["rng"])
    return {
        "initialization": "RESUME_REWARD_V2",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "reward_v2_samples_before": trainer.cumulative_samples,
    }


def _phase3_checkpoint_payload(
    trainer: PPO26DTrainer,
    *,
    environment_contract: dict[str, Any],
    selected_num_envs: int,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload(
        environment_contract=environment_contract,
        selected_num_envs=selected_num_envs,
    )
    payload["schema_version"] = PHASE3_CHECKPOINT_SCHEMA
    payload.pop("cumulative_samples")
    payload["reward_v2_samples"] = trainer.cumulative_samples
    payload["phase3_initialization"] = initialization
    payload["reference_kinematics_version"] = 2
    payload["reward_contract"] = environment_contract["ppo26d"]["reward"]
    return payload


def _resume_reward_v3(
    trainer: PPO26DTrainer, checkpoint: Path, *, expected_clip: str, expected_num_envs: int
) -> dict[str, Any]:
    """Resume only an explicitly V3-labelled checkpoint, never a V1/V2 one."""

    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    if payload.get("schema_version") != REWARD_V3_CHECKPOINT_SCHEMA:
        raise ValueError("REWARD_V3_RESUME_CHECKPOINT_SCHEMA_INVALID")
    if payload.get("clip") != expected_clip:
        raise ValueError("REWARD_V3_RESUME_CHECKPOINT_CLIP_MISMATCH")
    if int(payload.get("selected_num_envs", -1)) != expected_num_envs:
        raise ValueError("REWARD_V3_RESUME_CHECKPOINT_ENV_COUNT_MISMATCH")
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(payload["reward_v3_samples"])
    restore_rng_state(payload["rng"])
    return {
        "initialization": "RESUME_REWARD_V3",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "reward_v3_samples_before": trainer.cumulative_samples,
    }


def _reward_v3_checkpoint_payload(
    trainer: PPO26DTrainer,
    *,
    environment_contract: dict[str, Any],
    selected_num_envs: int,
    initialization: dict[str, Any],
    contact_entry: dict[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload(
        environment_contract=environment_contract,
        selected_num_envs=selected_num_envs,
    )
    payload["schema_version"] = REWARD_V3_CHECKPOINT_SCHEMA
    payload.pop("cumulative_samples")
    payload["reward_v3_samples"] = trainer.cumulative_samples
    payload["reward_v3_initialization"] = initialization
    payload["reward_v3_contact_entry"] = contact_entry
    payload["reference_kinematics_version"] = 2
    payload["reward_contract"] = environment_contract["ppo26d"]["reward"]
    return payload


def _resume_strict_v4(
    trainer: PPO26DTrainer, checkpoint: Path, *, expected_clip: str, expected_num_envs: int
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, map_location=trainer.trainer.device)
    if payload.get("schema_version") != STRICT_V4_CHECKPOINT_SCHEMA:
        raise ValueError("STRICT_V4_RESUME_CHECKPOINT_SCHEMA_INVALID")
    if payload.get("clip") != expected_clip:
        raise ValueError("STRICT_V4_RESUME_CHECKPOINT_CLIP_MISMATCH")
    if int(payload.get("selected_num_envs", -1)) != expected_num_envs:
        raise ValueError("STRICT_V4_RESUME_CHECKPOINT_ENV_COUNT_MISMATCH")
    trainer.model.load_state_dict(payload["actor_critic"])
    trainer.trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    trainer.trainer.normalizer.training = True
    trainer.cumulative_samples = int(payload["reward_v4_samples"])
    restore_rng_state(payload["rng"])
    return {
        "initialization": "RESUME_STRICT_V4",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "reward_v4_samples_before": trainer.cumulative_samples,
    }


def _strict_v4_checkpoint_payload(
    trainer: PPO26DTrainer,
    *,
    environment_contract: dict[str, Any],
    selected_num_envs: int,
    initialization: dict[str, Any],
    contact_entry: dict[str, Any],
) -> dict[str, Any]:
    payload = trainer.checkpoint_payload(
        environment_contract=environment_contract,
        selected_num_envs=selected_num_envs,
    )
    payload["schema_version"] = STRICT_V4_CHECKPOINT_SCHEMA
    payload.pop("cumulative_samples")
    payload["reward_v4_samples"] = trainer.cumulative_samples
    payload["strict_v4_initialization"] = initialization
    payload["strict_v4_contact_entry"] = contact_entry
    payload["reference_kinematics_version"] = 2
    payload["reward_contract"] = environment_contract["ppo26d"]["reward"]
    return payload


def _run_reward_smoke(env: Any, trainer: PPO26DTrainer, *, steps: int) -> dict[str, Any]:
    """Exercise a V3 reward for an exact control-state count without PPO storage.

    The frozen PPO rollout is capped at 320 transitions, while a Stage16D
    reference has 321 keys. This direct environment path therefore validates
    all 321 reward callbacks without changing the training rollout contract.
    """

    observation, _ = env.reset()
    reward_sum = 0.0
    finite = True
    terminated_or_timed_out = 0
    for _ in range(steps):
        with torch.no_grad():
            action = trainer.trainer.distribution(observation["policy"]).mean
        observation, reward, terminated, timed_out, _ = env.step(action)
        finite &= bool(torch.isfinite(reward).all())
        finite &= bool(torch.isfinite(observation["policy"]).all())
        reward_sum += float(reward.sum().detach().cpu())
        terminated_or_timed_out += int((terminated | timed_out).sum().detach().cpu())
    if not finite:
        raise FloatingPointError("REWARD_V3_DIRECT_REWARD_SMOKE_NONFINITE")
    return {
        "control_steps": steps,
        "finite": finite,
        "reward_sum": reward_sum,
        "terminated_or_timed_out_events": terminated_or_timed_out,
        "ppo_update_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", default="hocap_170650")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--object-usd", type=Path)
    parser.add_argument(
        "--run-label",
        help=(
            "Optional alphanumeric receipt subdirectory under phase3/<clip>/runs/. "
            "Use this when preserving an interrupted uncheckpointed attempt."
        ),
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--method-scale-reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Frozen development references used only to verify shared Reward V2 scales.",
    )
    parser.add_argument(
        "--capacity-selection",
        type=Path,
        help="Required when live VRAM required a fresh Stage16DPPOEnvCapacitySelectorV1 run.",
    )
    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        help=(
            "Optional V1 L0 actor/normalizer source. Reward V3 otherwise selects the frozen "
            "per-clip V1 L0, never a V2/P1 checkpoint."
        ),
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--target-reward-v2-samples", type=int)
    parser.add_argument(
        "--contact-mode",
        choices=tuple(mode.value for mode in ContactRewardMode),
        help=(
            "Unified Stage16-D contact objective. New invocations default to aggregate_v3; "
            "the older V3/V4 switches below remain deterministic compatibility aliases."
        ),
    )
    parser.add_argument(
        "--reward-v3-contact",
        action="store_true",
        help=(
            "Train the frozen reference-gated pair-force Reward V3. This selects a distinct "
            "checkpoint schema and never resumes a V1 or V2 PPO state."
        ),
    )
    parser.add_argument(
        "--target-reward-v3-samples",
        type=int,
        help="Exact Reward V3 sample budget; required with --reward-v3-contact.",
    )
    parser.add_argument(
        "--strict-per-finger-contact-reward-v4",
        action="store_true",
        help="Train V4 from the frozen source-confirmed strict per-finger contract.",
    )
    parser.add_argument(
        "--target-reward-v4-samples",
        type=int,
        help="Exact Strict V4 sample budget; required with --strict-per-finger-contact-reward-v4.",
    )
    parser.add_argument("--contact-reward-contract", type=Path, default=DEFAULT_V3_CONTACT_CONTRACT)
    parser.add_argument("--contact-mask-root", type=Path, default=DEFAULT_V3_CONTACT_MASK_ROOT)
    parser.add_argument("--strict-v4-contract", type=Path, default=DEFAULT_V4_CONTACT_CONTRACT)
    parser.add_argument(
        "--strict-v4-source-mask-root",
        type=Path,
        default=DEFAULT_V4_ROOT,
    )
    parser.add_argument(
        "--checkpoint-targets",
        type=int,
        nargs="+",
        default=(1_048_576, 2_097_152, 3_145_728, 4_194_304),
        help=(
            "Cumulative Reward V2 sample milestones to preserve.  A checkpoint is written "
            "at the first complete PPO iteration at or above each requested milestone."
        ),
    )
    parser.add_argument(
        "--no-critical-dr",
        action="store_false",
        dest="critical_dr",
        help="Disable the frozen critical domain-randomization setting used by the V1 baseline.",
    )
    parser.set_defaults(critical_dr=True)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help=(
            "Run one fresh V1-initialized rollout/GAE/PPO update, discard it, and write a receipt."
        ),
    )
    parser.add_argument(
        "--smoke-rollout-steps",
        type=int,
        help="Optional explicit PPO-update rollout length, bounded by the frozen 320 steps.",
    )
    parser.add_argument(
        "--smoke-reward-steps",
        type=int,
        help="Optional direct finite-reward smoke length; use 321 for the complete V3 reference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    independent_inputs = (args.reference, args.object_usd)
    if any(value is not None for value in independent_inputs) and not all(
        value is not None for value in independent_inputs
    ):
        raise ValueError(
            "independent reward training requires --reference and --object-usd together"
        )
    if args.reward_v3_contact and args.strict_per_finger_contact_reward_v4:
        raise ValueError("REWARD_V3_AND_STRICT_V4_ARE_MUTUALLY_EXCLUSIVE")
    legacy_mode = (
        ContactRewardMode.AGGREGATE_V3
        if args.reward_v3_contact
        else (
            ContactRewardMode.STRICT_PER_FINGER_V4
            if args.strict_per_finger_contact_reward_v4
            else None
        )
    )
    configured_mode = (
        ContactRewardMode.parse(args.contact_mode) if args.contact_mode is not None else None
    )
    if (
        configured_mode is not None
        and legacy_mode is not None
        and configured_mode is not legacy_mode
    ):
        raise ValueError("STAGE16D_CONTACT_MODE_LEGACY_FLAG_CONFLICT")
    # Explicit historical budget families are unambiguous migration evidence.
    # All new contact invocations otherwise start from the frozen aggregate V3
    # global default.
    contact_mode = configured_mode or legacy_mode
    if contact_mode is None:
        if args.target_reward_v2_samples is not None:
            contact_mode = None
        elif args.target_reward_v4_samples is not None:
            contact_mode = ContactRewardMode.STRICT_PER_FINGER_V4
        else:
            contact_mode = ContactRewardMode.AGGREGATE_V3
    is_reward_v3 = contact_mode is ContactRewardMode.AGGREGATE_V3
    is_strict_v4 = contact_mode is ContactRewardMode.STRICT_PER_FINGER_V4
    is_contact_reward = is_reward_v3 or is_strict_v4
    if not is_contact_reward and args.clip != "hocap_170650":
        raise ValueError("PHASE3_ONLY_HOCAP_170650_IS_AUTHORIZED")
    if is_strict_v4:
        if (
            any(
                value is not None
                for value in (args.target_reward_v2_samples, args.target_reward_v3_samples)
            )
            or args.target_reward_v4_samples is None
        ):
            raise ValueError("STRICT_V4_REQUIRES_ONLY_TARGET_REWARD_V4_SAMPLES")
        target_samples = args.target_reward_v4_samples
        sample_key = "reward_v4_samples"
        output_group = "ppo_v4"
    elif is_reward_v3:
        if (
            args.target_reward_v3_samples is None
            or args.target_reward_v2_samples is not None
            or args.target_reward_v4_samples is not None
        ):
            raise ValueError("REWARD_V3_REQUIRES_ONLY_TARGET_REWARD_V3_SAMPLES")
        target_samples = args.target_reward_v3_samples
        sample_key = "reward_v3_samples"
        output_group = "ppo_v3"
    else:
        if (
            args.target_reward_v2_samples is None
            or args.target_reward_v3_samples is not None
            or args.target_reward_v4_samples is not None
        ):
            raise ValueError("PHASE3_REQUIRES_ONLY_TARGET_REWARD_V2_SAMPLES")
        target_samples = args.target_reward_v2_samples
        sample_key = "reward_v2_samples"
        output_group = "phase3"
    if target_samples > 16_777_216:
        raise ValueError("REWARD_TRAINING_SAMPLE_CAP_EXCEEDED")
    default_initialization_checkpoint = {
        "hocap_170650": DEFAULT_L0_CHECKPOINT_170650,
        "hocap_170105": DEFAULT_L0_CHECKPOINT_170105,
    }.get(args.clip)
    if args.initialization_checkpoint is None and default_initialization_checkpoint is None:
        raise ValueError("INDEPENDENT_REWARD_TRAINING_INITIALIZATION_REQUIRED")
    assert (
        args.initialization_checkpoint is not None
        or default_initialization_checkpoint is not None
    )
    initialization_checkpoint = (
        args.initialization_checkpoint or default_initialization_checkpoint
    ).resolve()
    checkpoint_targets = tuple(sorted(set(args.checkpoint_targets)))
    if not checkpoint_targets or checkpoint_targets[0] <= 0:
        raise ValueError("PHASE3_CHECKPOINT_TARGETS_INVALID")
    if checkpoint_targets[-1] > 16_777_216:
        raise ValueError("PHASE3_CHECKPOINT_TARGETS_EXCEED_SAMPLE_CAP")
    capacity_selection: dict[str, Any] | None = None
    if args.capacity_selection is not None:
        capacity_path = args.capacity_selection.resolve()
        capacity_selection = json.loads(capacity_path.read_text(encoding="utf-8"))
        if capacity_selection.get("selector") != "Stage16DPPOEnvCapacitySelectorV1":
            raise ValueError("PHASE3_CAPACITY_SELECTOR_INVALID")
        if int(capacity_selection.get("selected_num_envs", -1)) != args.num_envs:
            raise ValueError("PHASE3_SELECTED_ENV_COUNT_MISMATCH")
        capacity_selection = {
            "path": str(capacity_path),
            "sha256": _sha256(capacity_path),
            "selected_num_envs": args.num_envs,
            "selector": "Stage16DPPOEnvCapacitySelectorV1",
        }
    root = args.output_root.resolve()
    if args.run_label is not None and not args.run_label.replace("_", "").isalnum():
        raise ValueError("PHASE3_RUN_LABEL_INVALID")
    output = root / output_group / args.clip
    if args.run_label is not None:
        output = output / "runs" / args.run_label
    output.mkdir(parents=True, exist_ok=True)
    entry = (
        _require_strict_v4_entry(args.strict_v4_contract)
        if is_strict_v4
        else (
            _require_reward_v3_entry(args.contact_reward_contract)
            if is_reward_v3
            else _require_phase3_entry(root, clip=args.clip)
        )
    )
    scale_freeze = _freeze_scales(args.method_scale_reference_root.resolve())
    _write_json(root / output_group / "reward_v2_base_scale_freeze.json", scale_freeze)
    gpu_probe = _gpu_probe()
    _write_json(root / output_group / "gpu_probe.json", gpu_probe)
    _write_json(output / "launch_progress.json", {"phase": "gpu_probe_complete"})
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    _write_json(output / "launch_progress.json", {"phase": "app_launcher_ready"})
    env: Any | None = None
    try:
        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            configure_independent_clip_runtime,
        )

        _write_json(output / "launch_progress.json", {"phase": "isaac_imports_ready"})

        cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo_cfg.configure_stage16d_ppo26d(
            cfg,
            num_envs=args.num_envs,
            clip=args.clip,
            rsi=True,
            critical_dr=args.critical_dr,
        )
        if args.reference is not None:
            assert args.object_usd is not None
            configure_independent_clip_runtime(
                cfg,
                clip_id=args.clip,
                reference_path=args.reference,
                object_usd_path=args.object_usd,
            )
        if contact_mode is not None:
            ppo_cfg.configure_stage16d_contact_reward(
                cfg,
                mode=contact_mode,
                reference_root=args.reference_root.resolve(),
                contact_reward_contract=(
                    args.strict_v4_contract.resolve()
                    if is_strict_v4
                    else args.contact_reward_contract.resolve()
                ),
                contact_mask_root=(
                    args.strict_v4_source_mask_root.resolve()
                    if is_strict_v4
                    else args.contact_mask_root.resolve()
                ),
            )
        else:
            ppo_cfg.configure_stage16d_phase3_object_twist_reward(
                cfg, reference_root=args.reference_root.resolve()
            )
        _write_json(
            output / "launch_progress.json",
            {
                "phase": (
                    "strict_v4_configured"
                    if is_strict_v4
                    else "v3_configured"
                    if is_reward_v3
                    else "v2_configured"
                )
            },
        )
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        _write_json(output / "launch_progress.json", {"phase": "environment_ready"})
        if cfg.reference_kinematics_version != 2:
            raise RuntimeError("REWARD_TRAINING_REQUIRES_REFERENCE_KINEMATICS_V2")
        trainer = PPO26DTrainer(
            observation_dim=764,
            device=str(env.device),
            runtime_reference_samples=env.reference_bank.frame_count,
        )
        _write_json(output / "launch_progress.json", {"phase": "trainer_fresh_ready"})
        initialization = (
            (
                _resume_strict_v4(
                    trainer,
                    args.resume_checkpoint.resolve(),
                    expected_clip=args.clip,
                    expected_num_envs=args.num_envs,
                )
                if is_strict_v4
                else (
                    _resume_reward_v3(
                        trainer,
                        args.resume_checkpoint.resolve(),
                        expected_clip=args.clip,
                        expected_num_envs=args.num_envs,
                    )
                    if is_reward_v3
                    else _resume_phase3(
                        trainer,
                        args.resume_checkpoint.resolve(),
                        expected_clip=args.clip,
                        expected_num_envs=args.num_envs,
                    )
                )
            )
            if args.resume_checkpoint is not None
            else _load_l0_actor_and_normalizer(
                trainer, initialization_checkpoint, expected_clip=args.clip
            )
        )
        _write_json(output / "launch_progress.json", {"phase": "initialization_ready"})
        samples_per_iteration = args.num_envs * Stage16DPPO26DTrainingConfigV1().rollout_length
        if trainer.cumulative_samples % args.num_envs != 0:
            raise ValueError("REWARD_TRAINING_RESUME_SAMPLE_COUNTER_MUST_ALIGN_WITH_NUM_ENVS")
        if target_samples <= trainer.cumulative_samples:
            raise ValueError("REWARD_TRAINING_TARGET_MUST_EXCEED_CURRENT_SAMPLES")
        if target_samples % args.num_envs != 0:
            raise ValueError("REWARD_TRAINING_TARGET_MUST_ALIGN_WITH_NUM_ENVS")
        if args.smoke_rollout_steps is not None and args.smoke_rollout_steps <= 0:
            raise ValueError("REWARD_TRAINING_SMOKE_ROLLOUT_STEPS_INVALID")
        if args.smoke_reward_steps is not None and args.smoke_reward_steps <= 0:
            raise ValueError("REWARD_TRAINING_SMOKE_REWARD_STEPS_INVALID")
        if args.smoke_reward_steps is not None and not (is_contact_reward and args.smoke_only):
            raise ValueError("CONTACT_REWARD_DIRECT_REWARD_SMOKE_REQUIRES_CONTACT_SMOKE_ONLY")
        config = {
            "schema_version": (
                "Stage16DStrictPerFingerV4TrainingConfigV1"
                if is_strict_v4
                else (
                    "Stage16DRewardV3TrainingConfigV1"
                    if is_reward_v3
                    else "Stage16DPhase3RewardV2TrainingConfigV1"
                )
            ),
            "clip": args.clip,
            "run_label": args.run_label,
            "reference_kinematics_version": cfg.reference_kinematics_version,
            f"{sample_key}_start": trainer.cumulative_samples,
            f"target_{sample_key}": target_samples,
            f"maximum_{sample_key}": 16_777_216,
            "checkpoint_targets": list(checkpoint_targets),
            "selected_num_envs": args.num_envs,
            "capacity_selection": capacity_selection,
            "samples_per_iteration": samples_per_iteration,
            "critical_dr": args.critical_dr,
            "reward_scale_freeze": scale_freeze,
            "entry": entry,
            "initialization": initialization,
            "environment": env.contract_report(),
        }
        _write_json(output / "training_config.json", config)
        _write_json(output / f"training_segment_target_{target_samples}.json", config)
        if args.smoke_only:
            if not is_contact_reward and args.num_envs != 1024 and capacity_selection is None:
                raise ValueError("PHASE3_RESUME_SMOKE_REQUIRES_1024_ENVS_OR_SELECTION")
            direct_reward_smoke = args.smoke_reward_steps is not None
            if direct_reward_smoke:
                metric = _run_reward_smoke(env, trainer, steps=args.smoke_reward_steps)
            else:
                metric = trainer.collect_and_update(
                    env,
                    rollout_length=(
                        args.smoke_rollout_steps or Stage16DPPO26DTrainingConfigV1().rollout_length
                    ),
                )
            _write_json(output / "launch_progress.json", {"phase": "smoke_update_complete"})
            metric.pop("last_policy_observation", None)
            receipt = {
                "schema_version": (
                    "Stage16DStrictPerFingerV4SmokeV1"
                    if is_strict_v4
                    else (
                        "Stage16DRewardV3SmokeV1" if is_reward_v3 else "Stage16DPhase3ResumeSmokeV1"
                    )
                ),
                "status": (
                    (
                        (
                            "STRICT_V4_321_STEP_REWARD_SMOKE_PASS"
                            if direct_reward_smoke
                            else "STRICT_V4_PPO_SMOKE_PASS"
                        )
                        if is_strict_v4
                        else (
                            "REWARD_V3_321_STEP_REWARD_SMOKE_PASS"
                            if direct_reward_smoke
                            else "REWARD_V3_PPO_SMOKE_PASS"
                        )
                    )
                    if is_contact_reward
                    else (
                        "PHASE3_1024_ENV_RESUME_SMOKE_PASS"
                        if args.num_envs == 1024
                        else "PHASE3_SELECTED_CAPACITY_RESUME_SMOKE_PASS"
                    )
                ),
                "selected_num_envs": args.num_envs,
                "rollout_steps": (
                    args.smoke_reward_steps
                    if direct_reward_smoke
                    else args.smoke_rollout_steps or Stage16DPPO26DTrainingConfigV1().rollout_length
                ),
                f"{sample_key}_exercised_then_discarded": (
                    None if direct_reward_smoke else metric["samples"]
                ),
                "finite": metric["finite"],
                "ppo": None if direct_reward_smoke else metric["ppo"],
                "direct_reward_smoke": metric if direct_reward_smoke else None,
                "initialization": initialization,
                "environment": env.contract_report(),
            }
            _write_json(output / "resume_smoke.json", receipt)
            print(json.dumps(receipt, sort_keys=True))
            return 0
        metrics_path = output / "training_metrics.jsonl"
        checkpoint: Path | None = None
        checkpoint_records: list[dict[str, Any]] = []
        pending_milestones = [
            milestone
            for milestone in checkpoint_targets
            if trainer.cumulative_samples < milestone <= target_samples
        ]
        checkpoint_prefix = (
            "strict_v4" if is_strict_v4 else "reward_v3" if is_reward_v3 else "phase3_reward_v2"
        )

        def save_milestone(*, milestone: int | None, reason: str) -> Path:
            path = (
                output
                / "checkpoints"
                / f"stage16d_{checkpoint_prefix}_samples_{trainer.cumulative_samples}.pt"
            )
            save_checkpoint(
                path,
                (
                    _strict_v4_checkpoint_payload(
                        trainer,
                        environment_contract=env.contract_report(),
                        selected_num_envs=args.num_envs,
                        initialization=initialization,
                        contact_entry=entry,
                    )
                    if is_strict_v4
                    else (
                        _reward_v3_checkpoint_payload(
                            trainer,
                            environment_contract=env.contract_report(),
                            selected_num_envs=args.num_envs,
                            initialization=initialization,
                            contact_entry=entry,
                        )
                        if is_reward_v3
                        else _phase3_checkpoint_payload(
                            trainer,
                            environment_contract=env.contract_report(),
                            selected_num_envs=args.num_envs,
                            initialization=initialization,
                        )
                    )
                ),
            )
            checkpoint_records.append(
                {
                    f"milestone_target_{sample_key}": milestone,
                    f"actual_{sample_key}": trainer.cumulative_samples,
                    "reason": reason,
                    "checkpoint": str(path.resolve()),
                    "checkpoint_sha256": _sha256(path),
                }
            )
            return path

        with metrics_path.open("a", encoding="utf-8") as stream:
            while trainer.cumulative_samples < target_samples:
                remaining_samples = target_samples - trainer.cumulative_samples
                remaining_steps = remaining_samples // args.num_envs
                rollout_steps = min(
                    Stage16DPPO26DTrainingConfigV1().rollout_length, remaining_steps
                )
                if rollout_steps <= 0:
                    raise RuntimeError("REWARD_TRAINING_EXACT_BUDGET_ROLLOUT_STEPS_INVALID")
                metric = trainer.collect_and_update(env, rollout_length=rollout_steps)
                metric.pop("last_policy_observation")
                metric[sample_key] = trainer.cumulative_samples
                metric["clip"] = args.clip
                stream.write(json.dumps(metric, sort_keys=True) + "\n")
                stream.flush()
                crossed = [
                    milestone
                    for milestone in pending_milestones
                    if trainer.cumulative_samples >= milestone
                ]
                for milestone in crossed:
                    checkpoint = save_milestone(milestone=milestone, reason="milestone_crossed")
                    pending_milestones.remove(milestone)
                if trainer.cumulative_samples >= target_samples:
                    if checkpoint is None or checkpoint.name != (
                        f"stage16d_{checkpoint_prefix}_samples_{trainer.cumulative_samples}.pt"
                    ):
                        checkpoint = save_milestone(milestone=None, reason="segment_target_reached")
        if trainer.cumulative_samples != target_samples:
            raise RuntimeError("REWARD_TRAINING_EXACT_SAMPLE_BUDGET_NOT_REACHED")
        if checkpoint is None:
            raise RuntimeError("PHASE3_CHECKPOINT_NOT_WRITTEN")
        _write_json(output / "checkpoint_milestones.json", checkpoint_records)
        result = {
            "schema_version": (
                "Stage16DStrictPerFingerV4TrainingResultV1"
                if is_strict_v4
                else (
                    "Stage16DRewardV3TrainingResultV1"
                    if is_reward_v3
                    else "Stage16DPhase3RewardV2TrainingResultV1"
                )
            ),
            "status": (
                "STRICT_V4_TRAINING_SEGMENT_COMPLETE"
                if is_strict_v4
                else (
                    "REWARD_V3_TRAINING_SEGMENT_COMPLETE"
                    if is_reward_v3
                    else "PHASE3_REWARD_V2_TRAINING_SEGMENT_COMPLETE"
                )
            ),
            "clip": args.clip,
            f"{sample_key}_start": config[f"{sample_key}_start"],
            sample_key: trainer.cumulative_samples,
            f"target_{sample_key}": target_samples,
            "iterations": math.ceil(
                (trainer.cumulative_samples - int(config[f"{sample_key}_start"]))
                / samples_per_iteration
            ),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_milestones": checkpoint_records,
            "metrics": str(metrics_path.resolve()),
            "initialization": initialization,
        }
        _write_json(output / "training_result.json", result)
        _write_json(output / f"training_result_{trainer.cumulative_samples}.json", result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        _write_json(
            output / "training_failure.json",
            {
                "schema_version": (
                    "Stage16DStrictPerFingerV4TrainingFailureV1"
                    if "is_strict_v4" in locals() and is_strict_v4
                    else (
                        "Stage16DRewardV3TrainingFailureV1"
                        if "is_reward_v3" in locals() and is_reward_v3
                        else "Stage16DPhase3TrainingFailureV1"
                    )
                ),
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
