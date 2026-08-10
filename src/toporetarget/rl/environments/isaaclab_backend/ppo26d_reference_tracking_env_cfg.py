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


__all__ = ["IsaacPPO26DReferenceTrackingEnvCfg", "configure_stage16d_ppo26d"]
