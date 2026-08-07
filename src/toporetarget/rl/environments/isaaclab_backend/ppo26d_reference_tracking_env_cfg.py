"""IsaacLab configuration for the Stage 16-D.5 PPO-26D environment."""

from __future__ import annotations

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
) -> None:
    """Apply the immutable Stage16-D physics contract without clip-specific code."""

    configure_stage16d_nominal(cfg, num_envs=num_envs, clip=clip)
    cfg.reset_reference_index = "uniform" if rsi else "frame0"
    cfg.ppo26d_rsi_enabled = rsi
    cfg.ppo26d_enable_critical_dr = critical_dr
    cfg.contact_telemetry = "aggregate"
    cfg.scene.lazy_sensor_update = True


__all__ = ["IsaacPPO26DReferenceTrackingEnvCfg", "configure_stage16d_ppo26d"]
