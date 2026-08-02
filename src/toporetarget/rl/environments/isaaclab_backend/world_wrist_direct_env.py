"""GPU-vectorized Stage 16-C.2 Isaac Lab DirectRLEnv.

Only this runtime module imports Isaac Lab.  The implementation never writes
the active wrist or object state during a rollout: PhysX receives a bounded
wrist wrench and finger position targets, while object motion comes solely
from contact with the free rigid body.
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .action_adapter import Stage16ActionAdapter
from .reference_bank import WorldWristReferenceBank, quaternion_to_matrix_wxyz
from .reward_terms import Stage16WorldWristRewardProfileV1, world_wrist_reward_terms
from .scene_frame import Stage16CSceneFrameContractV1, global_to_scene, scene_to_global
from .tensor_math import apply_local_residual, relative_rotation_log_local
from .termination_terms import TERMINATION_REASONS, Stage16TerminationProfileV1, stage16_termination
from .world_wrist_direct_env_cfg import IsaacWorldWristFingerDirectRLEnvCfg
from .wrist_controller import IsaacCartesianWristImpedanceController


class IsaacWorldWristFingerDirectRLEnv(DirectRLEnv):
    """C.2 environment with exact 26-D action and 764-D observation contracts."""

    cfg: IsaacWorldWristFingerDirectRLEnvCfg

    def __init__(
        self, cfg: IsaacWorldWristFingerDirectRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.scene_frame_contract = Stage16CSceneFrameContractV1()
        self.reference_bank = WorldWristReferenceBank(cfg.reference_paths, device=self.device)
        self._joint_ids = [
            self._robot.joint_names.index(name) for name in self.reference_bank.joint_order
        ]
        if len(self._robot.joint_names) != 20 or set(self._robot.joint_names) != set(
            self.reference_bank.joint_order
        ):
            raise RuntimeError(
                f"C2_ACTION_MAPPING_FAILURE: unexpected Isaac joints {self._robot.joint_names}"
            )
        self._tracked_link_ids = [
            self._robot.body_names.index(name) for name in self.reference_bank.tracked_link_names
        ]
        self._wrist_body_id = self._robot.body_names.index("r_wrist")
        lower = self._robot.data.joint_pos_limits[0, self._joint_ids, 0].clone()
        upper = self._robot.data.joint_pos_limits[0, self._joint_ids, 1].clone()
        self.action_adapter = Stage16ActionAdapter(
            canonical_joint_names=self.reference_bank.joint_order,
            isaac_joint_names=tuple(self._robot.joint_names),
            joint_lower=lower,
            joint_upper=upper,
        )
        self.joint_lower, self.joint_upper = lower, upper
        masses = self._robot.data.default_mass.to(self.device)
        inertia = self._robot.data.default_inertia.to(self.device)
        wrist_mass = masses.sum(dim=1)
        wrist_inertia = torch.stack(
            (inertia[..., 0].sum(dim=1), inertia[..., 4].sum(dim=1), inertia[..., 8].sum(dim=1)),
            dim=-1,
        )
        self.wrist_controller = IsaacCartesianWristImpedanceController(
            mass_kg=wrist_mass, inertia_kgm2=wrist_inertia
        )
        self.reward_profile = Stage16WorldWristRewardProfileV1()
        self.termination_profile = Stage16TerminationProfileV1()
        self._reference_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._clip_index = self.reference_bank.assignment(
            self.num_envs, balanced=cfg.balanced_clip_assignment
        )
        self._target_reference_index = torch.zeros_like(self._reference_index)
        self._actions = torch.zeros((self.num_envs, 26), dtype=torch.float32, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._second_previous_actions = torch.zeros_like(self._actions)
        self._joint_target_isaac = torch.zeros(
            (self.num_envs, 20), dtype=torch.float32, device=self.device
        )
        self._wrist_target_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._wrist_target_quaternion = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._wrist_target_twist = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self._wrist_target_quaternion[:, 0] = 1.0
        self._force_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._torque_saturated = torch.zeros_like(self._force_saturated)
        self._success = torch.zeros_like(self._force_saturated)
        self._reason_codes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._object_state_write_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._diagnostic_object_state_write_count = torch.zeros_like(self._object_state_write_count)
        self._wrist_step_state_write_count = torch.zeros_like(self._object_state_write_count)
        self._last_reward_terms: dict[str, torch.Tensor] = {}

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._object_170105 = RigidObject(self.cfg.object_170105)
        self._object_170650 = RigidObject(self.cfg.object_170650)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object_170105"] = self._object_170105
        self.scene.rigid_objects["object_170650"] = self._object_170650
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = self.action_adapter.validate_action(actions)
        self._target_reference_index = torch.minimum(
            self._reference_index + 1,
            torch.full_like(self._reference_index, self.reference_bank.frame_count - 1),
        )
        reference_q = self.reference_bank.gather(
            "q_finger_ref", self._clip_index, self._target_reference_index
        )
        target_canonical = self.action_adapter.finger_target_canonical(reference_q, self._actions)
        self._joint_target_isaac = self.action_adapter.canonical_to_isaac(target_canonical)
        reference_position = self.reference_bank.gather(
            "wrist_pose_translation_world_ref", self._clip_index, self._target_reference_index
        )
        reference_quaternion = self.reference_bank.gather(
            "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, self._target_reference_index
        )
        translation_residual = (
            self._actions[:, :3] * self.action_adapter.contract.wrist_translation_scale_m
        )
        rotation_residual = (
            self._actions[:, 3:6] * self.action_adapter.contract.wrist_rotation_scale_rad
        )
        local_position, self._wrist_target_quaternion = apply_local_residual(
            reference_position, reference_quaternion, translation_residual, rotation_residual
        )
        self._wrist_target_position = scene_to_global(local_position, self.scene.env_origins)
        self._wrist_target_twist = self.reference_bank.gather(
            "wrist_twist_world_ref", self._clip_index, self._target_reference_index
        )
        self._force_saturated.zero_()
        self._torque_saturated.zero_()

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_target_isaac)
        wrench = self.wrist_controller.compute(
            target_position=self._wrist_target_position,
            target_quaternion_wxyz=self._wrist_target_quaternion,
            target_twist_world=self._wrist_target_twist,
            current_position=self._robot.data.root_pos_w,
            current_quaternion_wxyz=self._robot.data.root_quat_w,
            current_linear_velocity_world=self._robot.data.root_lin_vel_w,
            current_angular_velocity_world=self._robot.data.root_ang_vel_w,
        )
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=wrench["force_world"].unsqueeze(1),
            torques=wrench["torque_world"].unsqueeze(1),
            body_ids=torch.tensor([self._wrist_body_id], device=self.device),
            is_global=True,
        )
        self._force_saturated |= wrench["force_saturated"]
        self._torque_saturated |= wrench["torque_saturated"]
        if self.cfg.diagnostic_kinematic_object:
            self._write_kinematic_object_diagnostic(self._target_reference_index)

    def _write_kinematic_object_diagnostic(self, reference_index: torch.Tensor) -> None:
        """C3-1-only object playback; formal free-object rollouts never call this."""

        env_ids = self._robot._ALL_INDICES
        active_position = scene_to_global(
            self.reference_bank.gather(
                "object_pose_translation_world_ref", self._clip_index, reference_index
            ),
            self.scene.env_origins,
        )
        active_quaternion = self.reference_bank.gather(
            "object_pose_quaternion_world_ref_wxyz", self._clip_index, reference_index
        )
        active_twist = self.reference_bank.gather(
            "object_twist_world_ref", self._clip_index, reference_index
        )
        active_state = torch.cat((active_position, active_quaternion, active_twist), dim=-1)
        inactive_position = self.scene.env_origins + torch.tensor(
            self.cfg.inactive_object_scene_offset, device=self.device
        )
        inactive_state = torch.cat(
            (
                inactive_position,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, -1),
                torch.zeros((self.num_envs, 6), device=self.device),
            ),
            dim=-1,
        )
        active_first = self._clip_index == 0
        self._object_170105.write_root_state_to_sim(
            torch.where(active_first[:, None], active_state, inactive_state), env_ids=env_ids
        )
        self._object_170650.write_root_state_to_sim(
            torch.where(active_first[:, None], inactive_state, active_state), env_ids=env_ids
        )
        self._diagnostic_object_state_write_count += 1

    def _active_object_state(self) -> torch.Tensor:
        select_first = self._clip_index == 0
        return torch.where(
            select_first[:, None],
            self._object_170105.data.root_state_w,
            self._object_170650.data.root_state_w,
        )

    def _object_axis_points_scene(self, object_state: torch.Tensor) -> torch.Tensor:
        rotation = quaternion_to_matrix_wxyz(object_state[:, 3:7])
        axis_local = torch.where(
            (self._clip_index == 0)[:, None, None],
            self.reference_bank.object_axis_points_local[0].expand(self.num_envs, -1, -1),
            self.reference_bank.object_axis_points_local[1].expand(self.num_envs, -1, -1),
        )
        global_points = object_state[:, None, :3] + torch.matmul(
            rotation[:, None], axis_local.unsqueeze(-1)
        ).squeeze(-1)
        return global_to_scene(global_points, self.scene.env_origins[:, None, :])

    def _state(self) -> dict[str, torch.Tensor]:
        object_state = self._active_object_state()
        wrist_position_scene = global_to_scene(self._robot.data.root_pos_w, self.scene.env_origins)
        wrist_quaternion = self._robot.data.root_quat_w
        object_position_scene = global_to_scene(object_state[:, :3], self.scene.env_origins)
        tracked_links_scene = global_to_scene(
            self._robot.data.body_pos_w[:, self._tracked_link_ids],
            self.scene.env_origins[:, None, :],
        )
        return {
            "wrist_position_scene": wrist_position_scene,
            "wrist_quaternion_wxyz": wrist_quaternion,
            "wrist_twist_world": torch.cat(
                (self._robot.data.root_lin_vel_w, self._robot.data.root_ang_vel_w), dim=-1
            ),
            "finger_q": self._robot.data.joint_pos[:, self._joint_ids],
            "finger_qdot": self._robot.data.joint_vel[:, self._joint_ids],
            "object_position_scene": object_position_scene,
            "object_quaternion_wxyz": object_state[:, 3:7],
            "object_twist_world": object_state[:, 7:13],
            "object_axis_points_scene": self._object_axis_points_scene(object_state),
            "tracked_links_scene": tracked_links_scene,
        }

    def _get_observations(self) -> dict[str, torch.Tensor]:
        state = self._state()
        index = self._reference_index
        wrist_position_ref = self.reference_bank.gather(
            "wrist_pose_translation_world_ref", self._clip_index, index
        )
        wrist_quaternion_ref = self.reference_bank.gather(
            "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, index
        )
        wrist_error = torch.cat(
            (
                wrist_position_ref - state["wrist_position_scene"],
                relative_rotation_log_local(state["wrist_quaternion_wxyz"], wrist_quaternion_ref),
            ),
            dim=-1,
        )
        wrist_rotation = quaternion_to_matrix_wxyz(state["wrist_quaternion_wxyz"])
        object_relative_translation = torch.matmul(
            wrist_rotation.transpose(-1, -2),
            (state["object_position_scene"] - state["wrist_position_scene"]).unsqueeze(-1),
        ).squeeze(-1)
        object_relative_rotation = relative_rotation_log_local(
            state["wrist_quaternion_wxyz"], state["object_quaternion_wxyz"]
        )
        chunks = [
            wrist_error,
            state["wrist_twist_world"],
            state["finger_q"],
            state["finger_qdot"],
            self._previous_actions,
            state["object_axis_points_scene"].flatten(1),
            torch.cat((object_relative_translation, object_relative_rotation), dim=-1),
            state["object_twist_world"],
        ]
        for offset in (0, 1, 3, 5):
            reference_index = torch.clamp(index + offset, max=self.reference_bank.frame_count - 1)
            reference_wrist_position = self.reference_bank.gather(
                "wrist_pose_translation_world_ref", self._clip_index, reference_index
            )
            reference_wrist_quaternion = self.reference_bank.gather(
                "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, reference_index
            )
            chunks.extend(
                (
                    torch.cat(
                        (
                            reference_wrist_position,
                            relative_rotation_log_local(
                                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand_as(
                                    reference_wrist_quaternion
                                ),
                                reference_wrist_quaternion,
                            ),
                        ),
                        dim=-1,
                    ),
                    self.reference_bank.gather(
                        "wrist_twist_world_ref", self._clip_index, reference_index
                    ),
                    self.reference_bank.gather("q_finger_ref", self._clip_index, reference_index),
                    self.reference_bank.gather(
                        "object_axis_points_world_ref", self._clip_index, reference_index
                    ).flatten(1),
                    self.reference_bank.gather(
                        "object_axis_points_wrist_ref", self._clip_index, reference_index
                    ).flatten(1),
                    self.reference_bank.gather(
                        "tracked_link_positions_world_ref", self._clip_index, reference_index
                    ).flatten(1),
                    self.reference_bank.gather(
                        "tracked_link_positions_wrist_ref", self._clip_index, reference_index
                    ).flatten(1),
                )
            )
        observation = torch.cat(chunks, dim=-1)
        if observation.shape != (self.num_envs, self.cfg.observation_space) or not bool(
            torch.isfinite(observation).all()
        ):
            raise RuntimeError(f"C2_OBSERVATION_FAILURE: shape={tuple(observation.shape)}")
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        state = self._state()
        index = self._target_reference_index
        terms = world_wrist_reward_terms(
            object_axis_points=state["object_axis_points_scene"],
            object_axis_points_ref=self.reference_bank.gather(
                "object_axis_points_world_ref", self._clip_index, index
            ),
            tracked_links=state["tracked_links_scene"],
            tracked_links_ref=self.reference_bank.gather(
                "tracked_link_positions_world_ref", self._clip_index, index
            ),
            finger_q=state["finger_q"],
            finger_q_ref=self.reference_bank.gather("q_finger_ref", self._clip_index, index),
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            wrist_position=state["wrist_position_scene"],
            wrist_quaternion_wxyz=state["wrist_quaternion_wxyz"],
            wrist_position_ref=self.reference_bank.gather(
                "wrist_pose_translation_world_ref", self._clip_index, index
            ),
            wrist_quaternion_ref_wxyz=self.reference_bank.gather(
                "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, index
            ),
            action=self._actions,
            previous_action=self._previous_actions,
            second_previous_action=self._second_previous_actions,
            profile=self.reward_profile,
        )
        self._last_reward_terms = terms
        self._second_previous_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._reference_index.copy_(self._target_reference_index)
        return terms["total"]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._state()
        index = self._target_reference_index
        termination = stage16_termination(
            object_position=state["object_position_scene"],
            object_quaternion_wxyz=state["object_quaternion_wxyz"],
            object_axis_points=state["object_axis_points_scene"],
            object_position_ref=self.reference_bank.gather(
                "object_pose_translation_world_ref", self._clip_index, index
            ),
            object_quaternion_ref_wxyz=self.reference_bank.gather(
                "object_pose_quaternion_world_ref_wxyz", self._clip_index, index
            ),
            object_axis_points_ref=self.reference_bank.gather(
                "object_axis_points_world_ref", self._clip_index, index
            ),
            wrist_position=state["wrist_position_scene"],
            wrist_quaternion_wxyz=state["wrist_quaternion_wxyz"],
            wrist_position_ref=self.reference_bank.gather(
                "wrist_pose_translation_world_ref", self._clip_index, index
            ),
            wrist_quaternion_ref_wxyz=self.reference_bank.gather(
                "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, index
            ),
            reference_index=index,
            final_reference_index=self.reference_bank.frame_count - 1,
            profile=self.termination_profile,
        )
        self._success.copy_(termination["success"])
        self._reason_codes.copy_(termination["primary_reason_code"])
        self.extras["stage16"] = {
            # DirectRLEnv resets done environments before returning.  Preserve
            # terminal evidence rather than exposing reset-mutated buffers.
            "success": self._success.clone(),
            "primary_reason_code": self._reason_codes.clone(),
            "clip_index": self._clip_index.clone(),
            "termination_reasons": TERMINATION_REASONS,
            "object_position_error_m": termination["object_position_error_m"].clone(),
            "object_axis_error_m": termination["object_axis_error_m"].clone(),
            "object_orientation_error_rad": termination["object_orientation_error_rad"].clone(),
            "wrist_position_error_m": termination["wrist_position_error_m"].clone(),
            "wrist_orientation_error_rad": termination["wrist_orientation_error_rad"].clone(),
            "force_saturated": self._force_saturated.clone(),
            "torque_saturated": self._torque_saturated.clone(),
        }
        return termination["terminated"], termination["success"]

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._robot.reset(env_ids)
        self._object_170105.reset(env_ids)
        self._object_170650.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.cfg.alternate_clip_on_reset:
            self._clip_index[env_ids] = 1 - self._clip_index[env_ids]
        elif self.cfg.balanced_clip_assignment:
            self._clip_index[env_ids] = env_ids % 2
        if self.cfg.reset_reference_index == "frame0":
            self._reference_index[env_ids] = 0
        elif self.cfg.reset_reference_index == "uniform":
            self._reference_index[env_ids] = torch.randint(
                self.reference_bank.frame_count, (len(env_ids),), device=self.device
            )
        else:
            raise ValueError("reset_reference_index must be frame0 or uniform")
        self._target_reference_index[env_ids] = self._reference_index[env_ids]
        self._previous_actions[env_ids] = 0.0
        self._second_previous_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._force_saturated[env_ids] = False
        self._torque_saturated[env_ids] = False
        self._success[env_ids] = False
        self._reason_codes[env_ids] = 0
        clips = self._clip_index[env_ids]
        frames = self._reference_index[env_ids]
        wrist_position = scene_to_global(
            self.reference_bank.wrist_pose_translation_world_ref[clips, frames],
            self.scene.env_origins[env_ids],
        )
        wrist_quaternion = self.reference_bank.wrist_pose_quaternion_world_ref_wxyz[clips, frames]
        wrist_twist = self.reference_bank.wrist_twist_world_ref[clips, frames]
        wrist_state = torch.cat((wrist_position, wrist_quaternion, wrist_twist), dim=-1)
        self._robot.write_root_state_to_sim(wrist_state, env_ids=env_ids)
        q_canonical = self.reference_bank.q_finger_ref[clips, frames]
        qdot_canonical = self.reference_bank.qdot_finger_ref[clips, frames]
        self._robot.write_joint_state_to_sim(
            self.action_adapter.canonical_to_isaac(q_canonical),
            self.action_adapter.canonical_to_isaac(qdot_canonical),
            env_ids=env_ids,
        )
        active_position = scene_to_global(
            self.reference_bank.object_pose_translation_world_ref[clips, frames],
            self.scene.env_origins[env_ids],
        )
        active_quaternion = self.reference_bank.object_pose_quaternion_world_ref_wxyz[clips, frames]
        active_twist = self.reference_bank.object_twist_world_ref[clips, frames]
        active_state = torch.cat((active_position, active_quaternion, active_twist), dim=-1)
        inactive_position = self.scene.env_origins[env_ids] + torch.tensor(
            self.cfg.inactive_object_scene_offset, device=self.device
        )
        inactive_state = torch.cat(
            (
                inactive_position,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(env_ids), -1),
                torch.zeros((len(env_ids), 6), device=self.device),
            ),
            dim=-1,
        )
        first_active = clips == 0
        self._object_170105.write_root_state_to_sim(
            torch.where(first_active[:, None], active_state, inactive_state), env_ids=env_ids
        )
        self._object_170650.write_root_state_to_sim(
            torch.where(first_active[:, None], inactive_state, active_state), env_ids=env_ids
        )
        self._object_state_write_count[env_ids] += 1

    def contract_report(self) -> dict[str, object]:
        """Static/runtime contract evidence consumed by qualification scripts."""

        return {
            "environment": "IsaacWorldWristFingerDirectRLEnv",
            "action": self.action_adapter.contract.as_dict(),
            "observation_dimension": self.cfg.observation_space,
            "scene_frame": self.scene_frame_contract.as_dict(),
            "reference_bank": self.reference_bank.manifest.as_dict(),
            "joint_mapping": self.action_adapter.mapping_manifest(),
            "termination": self.termination_profile.as_dict(),
            "reward": self.reward_profile.as_dict(),
            "wrist_root_state_writes_during_step": int(
                self._wrist_step_state_write_count.sum().item()
            ),
            "object_state_writes": int(self._object_state_write_count.sum().item()),
            "object_rollout_state_writes": 0,
            "diagnostic_kinematic_object": bool(self.cfg.diagnostic_kinematic_object),
            "diagnostic_object_state_writes": int(
                self._diagnostic_object_state_write_count.sum().item()
            ),
            "root_quaternion_order": "wxyz",
            "root_linear_angular_velocity_frame": "world/world",
        }


__all__ = ["IsaacWorldWristFingerDirectRLEnv"]
