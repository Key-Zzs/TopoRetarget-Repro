"""Isaac Lab runtime configuration for ``IsaacWorldWristFingerDirectRLEnv``.

This module is intentionally optional: import it only after ``AppLauncher``.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .d6_wrist_asset import D6_WRIST_PROFILES
from .explicit_virtual_wrist import EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER

REPO_ROOT = Path(__file__).resolve().parents[5]
_ASSET_ROOT = REPO_ROOT / ".local/generated_assets/isaaclab"
_REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
_EXPLICIT_VIRTUAL_WRIST_USD = (
    _ASSET_ROOT
    / "wuji_hand2_beta1_explicit_virtual_wrist"
    / "wujihand2_explicit_virtual_wrist.usda"
)

_JOINT_ORDER = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)


@configclass
class IsaacWorldWristFingerDirectRLEnvCfg(DirectRLEnvCfg):
    """Frozen nominal zero-gravity direct environment, not a PPO config."""

    decimation = 6
    episode_length_s = 2.05
    action_space = 26
    observation_space = 764
    state_space = 0
    is_finite_horizon = True
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=6,
        device="cuda:0",
        gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=8,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=2,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**20,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.75,
        replicate_physics=True,
        # Isaac Sim 5.1 ContactSensor views fail to resolve replicated bodies
        # under Fabric at 128 envs.  USD cloning preserves the GPU PhysX
        # profile and is the qualified contact-enabled execution profile.
        clone_in_fabric=False,
        lazy_sensor_update=False,
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_ROOT / "wuji_hand2_beta1/configuration/wujihand2_physics.usd"),
            copy_from_source=False,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos={".*": 0.0}),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(_JOINT_ORDER),
                stiffness=4.0,
                damping=0.2,
                effort_limit_sim=0.6,
                velocity_limit_sim=12.0,
            )
        },
    )
    object_170105: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object170105",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_ROOT / "hocap_170105/hocap_170105.usda"),
            copy_from_source=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
    )
    object_170650: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object170650",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_ROOT / "hocap_170650/hocap_170650.usda"),
            copy_from_source=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
    )
    reference_paths = {
        "hocap_170105": str(_REFERENCE_ROOT / "hocap_170105.world_wrist.stage16.npz"),
        "hocap_170650": str(_REFERENCE_ROOT / "hocap_170650.world_wrist.stage16.npz"),
    }
    balanced_clip_assignment = True
    alternate_clip_on_reset = False
    reset_reference_index = "frame0"
    inactive_object_scene_offset = (5.0, 5.0, -5.0)
    diagnostic_kinematic_object = False
    # Telemetry has no control/reward effect.  C.3R2 uses one object-centric
    # cached contact view; it never recreates the unstable 21-view design.
    contact_telemetry = "off"
    collect_wrist_diagnostics = False
    wrist_controller_mode = "wrist_impedance_v1"
    # Path B fallback selected only after the authored D6 wrapper is proven
    # unavailable through Isaac Lab's live GPU tensor articulation interface.
    # The profiles are global engineering-wrist bounds, never clip-specific.
    finite_virtual_wrist_profile = "nominal"
    finite_virtual_wrist_authority_enabled = True
    # Path A is a one-implementation, two-run bounded inverse-wrench attempt.
    # The read-only map is generated by an isolated PhysX probe and must be
    # supplied explicitly; it is never a learned or clip-tuned controller.
    identified_wrench_map_path: str | None = None
    identified_wrench_regularization = 0.10
    identified_wrench_translation_position_gain_s2 = 100.0
    identified_wrench_translation_damping_ratio = 1.0
    identified_wrench_rotation_position_gain_s2 = 36.0
    identified_wrench_rotation_damping_ratio = 1.0
    identified_wrench_force_limit_n = 50.0
    identified_wrench_torque_limit_nm = 6.0
    identified_wrench_condition_number_max = 4000.0
    identified_wrench_singular_value_relative_cutoff = 1.0e-3
    wrist_v1_translation_stiffness_npm = 800.0
    wrist_v1_translation_damping_ratio = 0.8
    wrist_v1_rotation_stiffness_nmprad = 6.0
    wrist_v1_rotation_damping_ratio = 0.8
    wrist_v1_force_limit_n = 50.0
    wrist_v1_torque_limit_nm = 4.0
    wrist_translation_position_gain_s2 = 100.0
    wrist_translation_damping_ratio = 1.0
    wrist_rotation_position_gain_s2 = 36.0
    wrist_rotation_damping_ratio = 1.0
    wrist_force_limit_n = 50.0
    wrist_torque_limit_nm = 4.0
    contact_max_data_per_body = 64
    # Diagnostic consumers retain only the latest records.  The sensor force
    # matrix stays GPU-resident; this prevents high-env C.1/C.4 probes from
    # accumulating unbounded Python telemetry.
    contact_record_capacity = 4096
    headless_debug = False


def configure_explicit_virtual_wrist(
    cfg: IsaacWorldWristFingerDirectRLEnvCfg,
    *,
    profile_identifier: str,
    authority_enabled: bool = True,
) -> None:
    """Select the explicit engineering 3P+3R wrist, never a real arm."""

    profile = next(
        (
            candidate
            for candidate in D6_WRIST_PROFILES
            if candidate.identifier == profile_identifier
        ),
        None,
    )
    if profile is None:
        valid = ", ".join(candidate.identifier for candidate in D6_WRIST_PROFILES)
        raise ValueError(f"unknown explicit wrist profile {profile_identifier!r}; expected {valid}")
    if not _EXPLICIT_VIRTUAL_WRIST_USD.is_file():
        raise FileNotFoundError(
            f"C3_EXPLICIT_VIRTUAL_WRIST_ASSET_MISSING: {_EXPLICIT_VIRTUAL_WRIST_USD}"
        )
    cfg.wrist_controller_mode = "finite_virtual_6d_wrist_actuator_v1"
    cfg.finite_virtual_wrist_profile = profile.identifier
    cfg.finite_virtual_wrist_authority_enabled = authority_enabled
    cfg.robot.spawn.usd_path = str(_EXPLICIT_VIRTUAL_WRIST_USD)
    cfg.robot.spawn.articulation_props.fix_root_link = True
    gain_scale = 1.0 if authority_enabled else 0.0
    cfg.robot.actuators = {
        "virtual_translation": ImplicitActuatorCfg(
            joint_names_expr=list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[:3]),
            stiffness=gain_scale * profile.translation_stiffness_npm,
            damping=gain_scale * profile.translation_damping_ns_per_m,
            effort_limit_sim=gain_scale * profile.translation_effort_limit_n,
            velocity_limit_sim=profile.translation_velocity_limit_mps,
        ),
        "virtual_rotation": ImplicitActuatorCfg(
            joint_names_expr=list(EXPLICIT_VIRTUAL_WRIST_JOINT_ORDER[3:]),
            stiffness=gain_scale * profile.rotation_stiffness_nm_per_rad,
            damping=gain_scale * profile.rotation_damping_nm_s_per_rad,
            effort_limit_sim=gain_scale * profile.rotation_effort_limit_nm,
            velocity_limit_sim=profile.rotation_velocity_limit_radps,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=list(_JOINT_ORDER),
            stiffness=4.0,
            damping=0.2,
            effort_limit_sim=0.6,
            velocity_limit_sim=12.0,
        ),
    }


__all__ = ["IsaacWorldWristFingerDirectRLEnvCfg", "configure_explicit_virtual_wrist"]
