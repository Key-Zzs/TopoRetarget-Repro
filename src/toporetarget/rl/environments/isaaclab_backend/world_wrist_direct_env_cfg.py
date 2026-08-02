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

REPO_ROOT = Path(__file__).resolve().parents[5]
_ASSET_ROOT = REPO_ROOT / ".local/generated_assets/isaaclab"
_REFERENCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"

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
        clone_in_fabric=True,
        lazy_sensor_update=False,
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_ASSET_ROOT / "wuji_hand2_beta1/configuration/wujihand2_physics.usd"),
            copy_from_source=False,
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
    headless_debug = False


__all__ = ["IsaacWorldWristFingerDirectRLEnvCfg"]
