"""IsaacLab configuration for the Stage 16-D.5 PPO-26D environment."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.utils import configclass

from .physics_consistent_retargeting_env_cfg import (
    IsaacPhysicsConsistentRetargetingEnvCfg,
    configure_stage16d_nominal,
)


@configclass
class IsaacPPO26DReferenceTrackingEnvCfg(IsaacPhysicsConsistentRetargetingEnvCfg):
    """Current Stage16-D physics with PPO-specific reset and safety semantics."""

    ppo26d_environment_contract = "Stage16DPPO26DReferenceTrackingEnvV1"
    reset_reference_index = "uniform"
    ppo26d_rsi_enabled = True
    ppo26d_observation_contract = "Stage16DPPO26DObservationV2"
    ppo26d_action_contract = "Stage16DReferenceResidualAction26DV1"
    ppo26d_reward_contract = "TopoRetargetReferenceTrackingReward26DV1"
    ppo26d_reference_contact_mask_paths: dict[str, str] | None = None
    ppo26d_contact_reward_contract_path: str | None = None
    reference_kinematics_version = 1
    ppo26d_workspace_radius_m = 0.75
    ppo26d_object_linear_speed_max_mps = 10.0
    ppo26d_object_angular_speed_max_radps = 500.0
    ppo26d_enable_critical_dr = False
    ppo26d_observation_joint_position_noise_std_rad = 0.02
    ppo26d_observation_joint_velocity_noise_std_radps = 0.05
    ppo26d_observation_axis_position_noise_std_m = 0.002
    ppo26d_reset_joint_noise_rad = 0.02
    ppo26d_reset_object_position_noise_m = 0.005
    ppo26d_reset_object_orientation_noise_rad = 0.03


def configure_stage16d_ppo26d(
    cfg: IsaacPPO26DReferenceTrackingEnvCfg,
    *,
    num_envs: int,
    clip: str = "hocap_170650",
    rsi: bool = True,
    critical_dr: bool = False,
    curriculum_reference_indices: tuple[int, ...] | None = None,
    curriculum_reference_probabilities: tuple[float, ...] | None = None,
    curriculum_phase: str | None = None,
) -> None:
    """Apply the immutable Stage16-D physics contract without clip-specific code."""

    configure_stage16d_nominal(cfg, num_envs=num_envs, clip=clip)
    if (curriculum_reference_indices is None) != (curriculum_reference_probabilities is None):
        raise ValueError("R6C needs both curriculum reset indices and probabilities")
    if curriculum_reference_indices is not None:
        if not rsi or curriculum_phase not in {"C0", "C1", "C2"}:
            raise ValueError("R6C curriculum requires RSI and an explicit C0/C1/C2 phase")
        cfg.reset_reference_index = "curriculum"
        cfg.curriculum_reference_indices = curriculum_reference_indices
        cfg.curriculum_reference_probabilities = curriculum_reference_probabilities
        cfg.curriculum_phase = curriculum_phase
    else:
        cfg.reset_reference_index = "uniform" if rsi else "frame0"
        cfg.curriculum_reference_indices = None
        cfg.curriculum_reference_probabilities = None
        cfg.curriculum_phase = None
    cfg.evaluation_reset_reference_indices = None
    cfg.ppo26d_rsi_enabled = rsi
    cfg.ppo26d_enable_critical_dr = critical_dr
    cfg.contact_telemetry = "aggregate"
    cfg.scene.lazy_sensor_update = True


def configure_stage16d_reference_kinematics_v2(
    cfg: IsaacPPO26DReferenceTrackingEnvCfg, *, reference_root: Path
) -> None:
    """Bind an explicitly materialized V2 reference; never rewrite V1 inputs."""

    root = reference_root.resolve()
    paths = {
        clip: root / f"{clip}.reference_kinematics_v2.npz"
        for clip in ("hocap_170105", "hocap_170650")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"STAGE16D_REFERENCE_KINEMATICS_V2_MISSING: {missing}")
    cfg.reference_paths = {clip: str(path) for clip, path in paths.items()}
    cfg.reference_time_scale = 8
    cfg.reference_kinematics_version = 2
    cfg.episode_length_s = 321 / 20.0


def configure_stage16d_phase3_object_twist_reward(
    cfg: IsaacPPO26DReferenceTrackingEnvCfg, *, reference_root: Path
) -> None:
    """Enable the gated Phase 3 reward without changing actor observations."""

    configure_stage16d_reference_kinematics_v2(cfg, reference_root=reference_root)
    cfg.ppo26d_reward_contract = "TopoRetargetReferenceTrackingReward26DV2"


def configure_stage16d_reference_gated_contact_reward(
    cfg: IsaacPPO26DReferenceTrackingEnvCfg,
    *,
    reference_root: Path,
    contact_reward_contract: Path,
    contact_mask_root: Path,
) -> None:
    """Bind a fully frozen V3 reward without changing physics or observations."""

    configure_stage16d_phase3_object_twist_reward(cfg, reference_root=reference_root)
    receipt_path = contact_reward_contract.resolve()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("status") != "CONTACT_REWARD_CONTRACT_FROZEN":
        raise ValueError("PPO26D_REWARD_V3_CONTACT_CONTRACT_NOT_FROZEN")
    parameters = payload.get("frozen_parameters")
    if not isinstance(parameters, dict) or not isinstance(
        parameters.get("lambda_c_n"), (int, float)
    ):
        raise ValueError("PPO26D_REWARD_V3_CONTACT_LAMBDA_MISSING")
    if float(parameters["lambda_c_n"]) <= 1.0e-5:
        raise ValueError("PPO26D_REWARD_V3_CONTACT_LAMBDA_INVALID")
    paths = {
        clip: contact_mask_root.resolve()
        / f"reference_contact_mask_{clip.removeprefix('hocap_')}.npz"
        for clip in ("hocap_170105", "hocap_170650")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"PPO26D_REWARD_V3_CONTACT_MASK_MISSING:{missing}")
    cfg.ppo26d_reward_contract = "TopoRetargetReferenceTrackingReward26DV3"
    cfg.ppo26d_contact_reward_contract_path = str(receipt_path)
    cfg.ppo26d_reference_contact_mask_paths = {clip: str(path) for clip, path in paths.items()}


__all__ = [
    "IsaacPPO26DReferenceTrackingEnvCfg",
    "configure_stage16d_ppo26d",
    "configure_stage16d_reference_kinematics_v2",
    "configure_stage16d_phase3_object_twist_reward",
    "configure_stage16d_reference_gated_contact_reward",
]
