"""GPU-vectorized Stage 16-C.2 Isaac Lab DirectRLEnv.

Only this runtime module imports Isaac Lab.  The implementation never writes
the active wrist or object state during a rollout: PhysX receives a bounded
wrist wrench and finger position targets, while object motion comes solely
from contact with the free rigid body.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensorCfg

from .action_adapter import Stage16ActionAdapter
from .finite_virtual_wrist_actuator import FiniteVirtual6DWristActuator
from .inverse_wrench_controller import (
    DampedSVDInverseWrenchController,
    EffectiveWrenchMap,
    IdentifiedInverseWrenchProfileV1,
)
from .reference_bank import WorldWristReferenceBank, quaternion_to_matrix_wxyz
from .reward_terms import Stage16WorldWristRewardProfileV1, world_wrist_reward_terms
from .scene_frame import Stage16CSceneFrameContractV1, global_to_scene, scene_to_global
from .tensor_math import apply_local_residual, relative_rotation_log_local
from .termination_terms import TERMINATION_REASONS, Stage16TerminationProfileV1, stage16_termination
from .world_wrist_direct_env_cfg import IsaacWorldWristFingerDirectRLEnvCfg
from .wrist_controller import (
    ArticulatedHandCompositeInertiaEstimator,
    IsaacCartesianWristImpedanceController,
    IsaacComputedWrenchWristControllerV2,
    IsaacComputedWrenchWristProfileV2,
    IsaacEffectiveDynamicsWristControllerV3,
    IsaacEffectiveDynamicsWristProfileV3,
    IsaacWristImpedanceProfileV1,
    PhysicsSubstepWristTargetInterpolator,
)

# These are the C.1-generated collision-bearing links.  Runtime construction
# verifies every configured sensor resolves to exactly one body; virtual tip
# links are intentionally absent because they have no generated collision mesh.
HAND_COLLISION_BODY_NAMES = (
    "r_wrist",
    "r_index_finger_proximal",
    "r_index_finger_proximal_abd",
    "r_index_finger_middle",
    "r_index_finger_distal",
    "r_middle_finger_proximal",
    "r_middle_finger_proximal_abd",
    "r_middle_finger_middle",
    "r_middle_finger_distal",
    "r_pinky_proximal",
    "r_pinky_proximal_abd",
    "r_pinky_middle",
    "r_pinky_distal",
    "r_ring_finger_proximal",
    "r_ring_finger_proximal_abd",
    "r_ring_finger_middle",
    "r_ring_finger_distal",
    "r_thumb_proximal",
    "r_thumb_proximal_abd",
    "r_thumb_middle",
    "r_thumb_distal",
)


class IsaacWorldWristFingerDirectRLEnv(DirectRLEnv):
    """C.2 environment with exact 26-D action and 764-D observation contracts."""

    cfg: IsaacWorldWristFingerDirectRLEnvCfg

    def __init__(
        self, cfg: IsaacWorldWristFingerDirectRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        self._install_object_centric_contact_sensor(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        self.scene_frame_contract = Stage16CSceneFrameContractV1()
        self.reference_bank = WorldWristReferenceBank(cfg.reference_paths, device=self.device)
        self._object_contact_sensors = {
            "Object170105": self.scene["object_170105_hand_contact"],
            "Object170650": self.scene["object_170650_hand_contact"],
        }
        for object_name, sensor in self._object_contact_sensors.items():
            if sensor.num_bodies != 1 or sensor.body_names != [object_name]:
                raise RuntimeError(
                    "C3_OBJECT_CENTRIC_CONTACT_SENSOR_COVERAGE_FAILURE: "
                    f"object={object_name} bodies={sensor.body_names}"
                )
        self._validate_contact_telemetry_mode()
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
        self._link_masses_kg = self._robot.data.default_mass.to(self.device).clone()
        self._link_inertia_kgm2 = (
            self._robot.data.default_inertia.to(self.device)
            .reshape(self.num_envs, self._robot.num_bodies, 3, 3)
            .clone()
        )
        wrist_mass = self._link_masses_kg.sum(dim=1)
        wrist_inertia = torch.diagonal(self._link_inertia_kgm2.sum(dim=1), dim1=-2, dim2=-1)
        self.wrist_controller = IsaacCartesianWristImpedanceController(
            mass_kg=wrist_mass,
            inertia_kgm2=wrist_inertia,
            profile=IsaacWristImpedanceProfileV1(
                translation_stiffness_npm=cfg.wrist_v1_translation_stiffness_npm,
                translation_damping_ratio=cfg.wrist_v1_translation_damping_ratio,
                rotation_stiffness_nmprad=cfg.wrist_v1_rotation_stiffness_nmprad,
                rotation_damping_ratio=cfg.wrist_v1_rotation_damping_ratio,
                force_limit_n=cfg.wrist_v1_force_limit_n,
                torque_limit_nm=cfg.wrist_v1_torque_limit_nm,
            ),
        )
        self.wrist_controller_v2 = IsaacComputedWrenchWristControllerV2(
            IsaacComputedWrenchWristProfileV2(
                translation_position_gain_s2=cfg.wrist_translation_position_gain_s2,
                translation_damping_ratio=cfg.wrist_translation_damping_ratio,
                rotation_position_gain_s2=cfg.wrist_rotation_position_gain_s2,
                rotation_damping_ratio=cfg.wrist_rotation_damping_ratio,
                force_limit_n=cfg.wrist_force_limit_n,
                torque_limit_nm=cfg.wrist_torque_limit_nm,
            )
        )
        self.wrist_controller_v3 = IsaacEffectiveDynamicsWristControllerV3(
            IsaacEffectiveDynamicsWristProfileV3(
                translation_position_gain_s2=cfg.wrist_translation_position_gain_s2,
                translation_damping_ratio=cfg.wrist_translation_damping_ratio,
                rotation_position_gain_s2=cfg.wrist_rotation_position_gain_s2,
                rotation_damping_ratio=cfg.wrist_rotation_damping_ratio,
                force_limit_n=cfg.wrist_force_limit_n,
                torque_limit_nm=cfg.wrist_torque_limit_nm,
            )
        )
        self.wrist_controller_inverse: DampedSVDInverseWrenchController | None = None
        self.finite_virtual_wrist_actuator: FiniteVirtual6DWristActuator | None = None
        if cfg.wrist_controller_mode == "identified_inverse_wrench_v1":
            if cfg.identified_wrench_map_path is None:
                raise RuntimeError("C3_PATH_A_MAP_REQUIRED: identified_wrench_map_path is required")
            effective_map = EffectiveWrenchMap.from_json(
                cfg.identified_wrench_map_path, device=self.device
            )
            self.wrist_controller_inverse = DampedSVDInverseWrenchController(
                effective_map=effective_map,
                regularization=cfg.identified_wrench_regularization,
                profile=IdentifiedInverseWrenchProfileV1(
                    translation_position_gain_s2=cfg.identified_wrench_translation_position_gain_s2,
                    translation_damping_ratio=cfg.identified_wrench_translation_damping_ratio,
                    rotation_position_gain_s2=cfg.identified_wrench_rotation_position_gain_s2,
                    rotation_damping_ratio=cfg.identified_wrench_rotation_damping_ratio,
                    force_limit_n=cfg.identified_wrench_force_limit_n,
                    torque_limit_nm=cfg.identified_wrench_torque_limit_nm,
                    condition_number_max=cfg.identified_wrench_condition_number_max,
                    singular_value_relative_cutoff=(
                        cfg.identified_wrench_singular_value_relative_cutoff
                    ),
                ),
            )
        elif cfg.wrist_controller_mode == "finite_virtual_6d_wrist_actuator_v1":
            controller = FiniteVirtual6DWristActuator
            self.finite_virtual_wrist_actuator = controller.from_profile_identifier(
                cfg.finite_virtual_wrist_profile,
            )
        self._wrist_interpolator = PhysicsSubstepWristTargetInterpolator(
            decimation=cfg.decimation, control_dt_s=self.step_dt
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
        self._wrist_target_acceleration = torch.zeros_like(self._wrist_target_twist)
        self._wrist_interval_start_position = torch.zeros_like(self._wrist_target_position)
        self._wrist_interval_end_position = torch.zeros_like(self._wrist_target_position)
        self._wrist_interval_start_quaternion = torch.zeros_like(self._wrist_target_quaternion)
        self._wrist_interval_end_quaternion = torch.zeros_like(self._wrist_target_quaternion)
        self._wrist_interval_start_twist = torch.zeros_like(self._wrist_target_twist)
        self._wrist_interval_end_twist = torch.zeros_like(self._wrist_target_twist)
        self._wrist_translation_residual = torch.zeros_like(self._wrist_target_position)
        self._wrist_rotation_residual = torch.zeros_like(self._wrist_target_position)
        self._wrist_target_quaternion[:, 0] = 1.0
        self._wrist_interval_start_quaternion[:, 0] = 1.0
        self._wrist_interval_end_quaternion[:, 0] = 1.0
        self._physics_substep = 0
        self._force_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._torque_saturated = torch.zeros_like(self._force_saturated)
        self._force_saturation_substeps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._torque_saturation_substeps = torch.zeros_like(self._force_saturation_substeps)
        self._wrist_substeps = torch.zeros_like(self._force_saturation_substeps)
        self._wrist_diagnostic_records: list[dict[str, object]] = []
        self._contact_substep_records: deque[dict[str, object]] = deque(
            maxlen=cfg.contact_record_capacity
        )
        self._contact_substep_record_total = 0
        self._pending_contact_sample: dict[str, torch.Tensor | int] | None = None
        self._success = torch.zeros_like(self._force_saturated)
        self._reason_codes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._object_state_write_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._diagnostic_object_state_write_count = torch.zeros_like(self._object_state_write_count)
        self._wrist_step_state_write_count = torch.zeros_like(self._object_state_write_count)
        self._identified_map_condition_number = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._identified_map_condition_gate_pass = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._identified_map_selected_reference_frame = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._last_reward_terms: dict[str, torch.Tensor] = {}

    @staticmethod
    def _install_object_centric_contact_sensor(cfg: IsaacWorldWristFingerDirectRLEnvCfg) -> None:
        """Install one object-side filtered view for all hand collision bodies.

        Stage 16-C.3R1 found that reading 21 independent filtered sensor views
        could terminate Isaac.  This one cached view preserves an object-net
        contact force and pair presence without fabricating point data.
        """

        filter_prim_paths = [
            f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in HAND_COLLISION_BODY_NAMES
        ]
        for object_name, sensor_name in (
            ("Object170105", "object_170105_hand_contact"),
            ("Object170650", "object_170650_hand_contact"),
        ):
            if getattr(cfg.scene, sensor_name, None) is not None:
                continue
            # Isaac Lab's ContactSensorCfg explicitly supports filtered force
            # matrices only when the sensor primitive resolves to one body per
            # environment.  Thus these are two object-centric views, not the
            # unstable 21 hand-centric views from C.3R1.
            setattr(
                cfg.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
                    update_period=0.0,
                    history_length=1,
                    track_pose=True,
                    track_contact_points=False,
                    track_friction_forces=False,
                    force_threshold=1.0e-4,
                    max_contact_data_count_per_prim=cfg.contact_max_data_per_body,
                    filter_prim_paths_expr=filter_prim_paths,
                ),
            )

    def _validate_contact_telemetry_mode(self) -> None:
        if self.cfg.contact_telemetry not in {"off", "aggregate", "diagnostic"}:
            raise ValueError(
                "contact_telemetry must be off, aggregate, or diagnostic; "
                f"got {self.cfg.contact_telemetry!r}"
            )

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
        self._wrist_interval_start_position = self.reference_bank.gather(
            "wrist_pose_translation_world_ref", self._clip_index, self._reference_index
        )
        self._wrist_interval_end_position = reference_position
        self._wrist_interval_start_quaternion = self.reference_bank.gather(
            "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, self._reference_index
        )
        self._wrist_interval_end_quaternion = reference_quaternion
        self._wrist_interval_start_twist = self.reference_bank.gather(
            "wrist_twist_world_ref", self._clip_index, self._reference_index
        )
        self._wrist_interval_end_twist = self.reference_bank.gather(
            "wrist_twist_world_ref", self._clip_index, self._target_reference_index
        )
        # Keep the endpoint target observable for static action-map tests.  The
        # actual controller target is recomputed from the exact substep sample.
        local_position, self._wrist_target_quaternion = apply_local_residual(
            self._wrist_interval_end_position,
            self._wrist_interval_end_quaternion,
            translation_residual,
            rotation_residual,
        )
        self._wrist_target_position = scene_to_global(local_position, self.scene.env_origins)
        self._wrist_target_twist = self._wrist_interval_end_twist
        self._wrist_translation_residual = translation_residual
        self._wrist_rotation_residual = rotation_residual
        self._physics_substep = 0
        self._force_saturated.zero_()
        self._torque_saturated.zero_()

    def _apply_action(self) -> None:
        self._record_completed_contact_substep()
        self._robot.set_joint_position_target(self._joint_target_isaac)
        target = self._wrist_interpolator.sample(
            position_k=self._wrist_interval_start_position,
            quaternion_k_wxyz=self._wrist_interval_start_quaternion,
            twist_k_world=self._wrist_interval_start_twist,
            position_k1=self._wrist_interval_end_position,
            quaternion_k1_wxyz=self._wrist_interval_end_quaternion,
            twist_k1_world=self._wrist_interval_end_twist,
            substep=self._physics_substep,
        )
        target_scene_position, target_quaternion = apply_local_residual(
            target.position_world,
            target.quaternion_wxyz,
            self._wrist_translation_residual,
            self._wrist_rotation_residual,
        )
        self._wrist_target_position = scene_to_global(target_scene_position, self.scene.env_origins)
        self._wrist_target_quaternion = target_quaternion
        self._wrist_target_twist = target.twist_world
        self._wrist_target_acceleration = target.acceleration_world
        if self.cfg.wrist_controller_mode == "wrist_impedance_v1":
            wrench = self.wrist_controller.compute(
                target_position=self._wrist_target_position,
                target_quaternion_wxyz=self._wrist_target_quaternion,
                target_twist_world=self._wrist_target_twist,
                current_position=self._robot.data.root_pos_w,
                current_quaternion_wxyz=self._robot.data.root_quat_w,
                current_linear_velocity_world=self._robot.data.root_lin_vel_w,
                current_angular_velocity_world=self._robot.data.root_ang_vel_w,
            )
            composite = self._composite_inertia()
        elif self.cfg.wrist_controller_mode == "computed_wrench_v2":
            composite = self._composite_inertia()
            wrench = self.wrist_controller_v2.compute(
                mass_kg=composite.mass_kg,
                inertia_world_kgm2=composite.inertia_world_kgm2,
                target_position_world=self._wrist_target_position,
                target_quaternion_wxyz=self._wrist_target_quaternion,
                target_twist_world=self._wrist_target_twist,
                target_acceleration_world=self._wrist_target_acceleration,
                current_position_world=self._robot.data.root_pos_w,
                current_quaternion_wxyz=self._robot.data.root_quat_w,
                current_linear_velocity_world=self._robot.data.root_lin_vel_w,
                current_angular_velocity_world=self._robot.data.root_ang_vel_w,
            )
        elif self.cfg.wrist_controller_mode == "effective_dynamics_v3":
            composite = self._composite_inertia()
            wrench = self.wrist_controller_v3.compute(
                target_position_world=self._wrist_target_position,
                target_quaternion_wxyz=self._wrist_target_quaternion,
                target_twist_world=self._wrist_target_twist,
                target_acceleration_world=self._wrist_target_acceleration,
                current_position_world=self._robot.data.root_pos_w,
                current_quaternion_wxyz=self._robot.data.root_quat_w,
                current_linear_velocity_world=self._robot.data.root_lin_vel_w,
                current_angular_velocity_world=self._robot.data.root_ang_vel_w,
            )
        elif self.cfg.wrist_controller_mode == "identified_inverse_wrench_v1":
            if self.wrist_controller_inverse is None:
                raise RuntimeError("C3_PATH_A_CONTROLLER_UNINITIALIZED")
            composite = self._composite_inertia()
            wrench = self.wrist_controller_inverse.compute(
                clip_index=self._clip_index,
                reference_index=self._target_reference_index,
                target_position_world=self._wrist_target_position,
                target_quaternion_wxyz=self._wrist_target_quaternion,
                target_twist_world=self._wrist_target_twist,
                target_acceleration_world=self._wrist_target_acceleration,
                current_position_world=self._robot.data.root_pos_w,
                current_quaternion_wxyz=self._robot.data.root_quat_w,
                current_linear_velocity_world=self._robot.data.root_lin_vel_w,
                current_angular_velocity_world=self._robot.data.root_ang_vel_w,
            )
            self._identified_map_condition_number.copy_(wrench["map_condition_number"])
            self._identified_map_condition_gate_pass.copy_(wrench["map_condition_gate_pass"])
            self._identified_map_selected_reference_frame.copy_(
                wrench["map_selected_reference_frame"]
            )
        elif self.cfg.wrist_controller_mode == "finite_virtual_6d_wrist_actuator_v1":
            if self.finite_virtual_wrist_actuator is None:
                raise RuntimeError("C3_PATH_B_VIRTUAL_WRIST_UNINITIALIZED")
            composite = self._composite_inertia()
            wrist_position = self._robot.data.body_link_pos_w[:, self._wrist_body_id]
            wrist_quaternion = self._robot.data.body_link_quat_w[:, self._wrist_body_id]
            wrist_linear_velocity = self._robot.data.body_lin_vel_w[:, self._wrist_body_id]
            wrist_angular_velocity = self._robot.data.body_ang_vel_w[:, self._wrist_body_id]
            virtual_wrench = self.finite_virtual_wrist_actuator.compute(
                target_position_world=self._wrist_target_position,
                target_quaternion_wxyz=self._wrist_target_quaternion,
                target_twist_world=self._wrist_target_twist,
                current_position_world=wrist_position,
                current_quaternion_wxyz=wrist_quaternion,
                current_linear_velocity_world=wrist_linear_velocity,
                current_angular_velocity_world=wrist_angular_velocity,
                dt_s=self.physics_dt,
            )
            wrench = {
                key: virtual_wrench[key]
                for key in (
                    "force_world",
                    "torque_world",
                    "force_saturated",
                    "torque_saturated",
                    "position_deflection_limited",
                    "rotation_deflection_limited",
                )
            }
            if not self.cfg.finite_virtual_wrist_authority_enabled:
                # Explicit Path-B ablation only: retain target generation and
                # physical stepping while removing all virtual wrist authority.
                wrench["force_world"] = torch.zeros_like(wrench["force_world"])
                wrench["torque_world"] = torch.zeros_like(wrench["torque_world"])
                wrench["force_saturated"] = torch.zeros_like(wrench["force_saturated"])
                wrench["torque_saturated"] = torch.zeros_like(wrench["torque_saturated"])
        else:
            raise RuntimeError(f"C3_WRIST_CONTROLLER_FAILURE: {self.cfg.wrist_controller_mode}")
        # ``instantaneous`` is reset by Isaac Lab after every write.  Replacing
        # it every substep refreshes the world-to-link conversion at the live
        # wrist pose and prevents a stale global wrench frame.
        self._robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=wrench["force_world"].unsqueeze(1),
            torques=wrench["torque_world"].unsqueeze(1),
            body_ids=torch.tensor([self._wrist_body_id], device=self.device),
            is_global=True,
        )
        self._force_saturated |= wrench["force_saturated"]
        self._torque_saturated |= wrench["torque_saturated"]
        self._force_saturation_substeps += wrench["force_saturated"].to(torch.long)
        self._torque_saturation_substeps += wrench["torque_saturated"].to(torch.long)
        self._wrist_substeps += 1
        self._pending_contact_sample = {
            "control_step": self._reference_index.clone(),
            "physics_substep": self._physics_substep,
            "reference_index": self._target_reference_index.clone(),
            "object_state_before": self._active_object_state().clone(),
            "force_world": wrench["force_world"].clone(),
            "torque_world": wrench["torque_world"].clone(),
        }
        if self.cfg.collect_wrist_diagnostics:
            self._append_wrist_diagnostic(composite, wrench, target.alpha)
        self._physics_substep += 1
        if self.cfg.diagnostic_kinematic_object:
            self._write_kinematic_object_diagnostic(self._target_reference_index)

    def _composite_inertia(self):
        return ArticulatedHandCompositeInertiaEstimator.estimate(
            masses_kg=self._link_masses_kg,
            inertia_link_kgm2=self._link_inertia_kgm2,
            link_quaternion_world_wxyz=self._robot.data.body_link_quat_w,
            center_of_mass_world=self._robot.data.body_com_pos_w,
            root_origin_world=self._robot.data.root_pos_w,
        )

    def _append_wrist_diagnostic(
        self, composite: Any, wrench: dict[str, torch.Tensor], alpha: torch.Tensor
    ) -> None:
        """Persist compact, finite substep evidence only for explicit diagnostics."""

        for env_id in range(self.num_envs):
            self._wrist_diagnostic_records.append(
                {
                    "env_id": env_id,
                    "control_step": int(self._reference_index[env_id].item()),
                    "physics_substep": self._physics_substep,
                    "reference_alpha": float(alpha[env_id].item()),
                    "target_position_world_m": self._wrist_target_position[env_id]
                    .detach()
                    .cpu()
                    .tolist(),
                    "target_twist_world": self._wrist_target_twist[env_id].detach().cpu().tolist(),
                    "target_acceleration_world": self._wrist_target_acceleration[env_id]
                    .detach()
                    .cpu()
                    .tolist(),
                    "applied_force_world_n": wrench["force_world"][env_id].detach().cpu().tolist(),
                    "applied_torque_world_nm": wrench["torque_world"][env_id]
                    .detach()
                    .cpu()
                    .tolist(),
                    "force_saturated": bool(wrench["force_saturated"][env_id].item()),
                    "torque_saturated": bool(wrench["torque_saturated"][env_id].item()),
                    "composite_mass_kg": float(composite.mass_kg[env_id].item()),
                    "inertia_eigenvalues_kgm2": composite.eigenvalues_kgm2[env_id]
                    .detach()
                    .cpu()
                    .tolist(),
                    "inertia_condition": float(composite.condition[env_id].item()),
                }
            )

    def _record_completed_contact_substep(self) -> None:
        """Capture filtered hand-object contacts after the preceding PhysX step.

        DirectRLEnv updates sensor buffers after each simulation step.  The next
        call to ``_apply_action`` therefore observes exactly the previous
        substep's contact data and its before/after object velocity boundary.
        ``_get_dones`` flushes the final pending sample.
        """

        pending = self._pending_contact_sample
        if pending is None:
            return
        self._pending_contact_sample = None
        if self.cfg.contact_telemetry == "off":
            return
        object_before = pending["object_state_before"]
        assert isinstance(object_before, torch.Tensor)
        object_after = self._active_object_state().clone()
        object_names = ("Object170105", "Object170650")
        object_masses = torch.where(
            self._clip_index == 0,
            self._object_170105.data.default_mass[:, 0].to(self.device),
            self._object_170650.data.default_mass[:, 0].to(self.device),
        )
        first_force_matrix = self._object_contact_sensors["Object170105"].data.force_matrix_w
        second_force_matrix = self._object_contact_sensors["Object170650"].data.force_matrix_w
        if (
            first_force_matrix is None
            or second_force_matrix is None
            or first_force_matrix.ndim != 4
            or second_force_matrix.ndim != 4
        ):
            raise RuntimeError(
                "C3_OBJECT_CENTRIC_CONTACT_DATA_UNAVAILABLE: "
                "shapes="
                f"{None if first_force_matrix is None else tuple(first_force_matrix.shape)},"
                f"{None if second_force_matrix is None else tuple(second_force_matrix.shape)}"
            )
        # force_matrix is the raw force on the object sensor body from each
        # filter body.  Sum it only for the object-net signal; do not split the
        # resultant across fingers or construct point forces.
        pair_force_on_object = torch.where(
            (self._clip_index == 0)[:, None, None],
            first_force_matrix[:, 0],
            second_force_matrix[:, 0],
        )
        pair_presence = torch.linalg.vector_norm(pair_force_on_object, dim=-1) > (
            self._object_contact_sensors["Object170105"].cfg.force_threshold
        )
        aggregate_force_on_object = pair_force_on_object.sum(dim=1)
        for env_id in range(self.num_envs):
            self._contact_substep_records.append(
                self._contact_record(
                    pending=pending,
                    env_id=env_id,
                    object_name=object_names[int(self._clip_index[env_id].item())],
                    pair_presence=pair_presence[env_id],
                    aggregate_force_on_object=aggregate_force_on_object[env_id],
                    pair_force_on_object=(
                        pair_force_on_object[env_id]
                        if self.cfg.contact_telemetry == "diagnostic"
                        else None
                    ),
                    object_before=object_before[env_id],
                    object_after=object_after[env_id],
                    object_mass_kg=object_masses[env_id],
                )
            )
            self._contact_substep_record_total += 1

    def _contact_record(
        self,
        *,
        pending: dict[str, torch.Tensor | int],
        env_id: int,
        object_name: str,
        pair_presence: torch.Tensor,
        aggregate_force_on_object: torch.Tensor,
        pair_force_on_object: torch.Tensor | None,
        object_before: torch.Tensor,
        object_after: torch.Tensor,
        object_mass_kg: torch.Tensor,
    ) -> dict[str, object]:
        control_step = pending["control_step"]
        reference_index = pending["reference_index"]
        applied_force_world = pending["force_world"]
        applied_torque_world = pending["torque_world"]
        assert isinstance(applied_force_world, torch.Tensor)
        assert isinstance(applied_torque_world, torch.Tensor)
        force = aggregate_force_on_object.detach()
        impulse_on_object = force * self.physics_dt
        delta_v = object_after[7:10] - object_before[7:10]
        delta_omega = object_after[10:13] - object_before[10:13]
        record: dict[str, object] = {
            "env_id": env_id,
            "control_step": int(control_step[env_id].item())
            if isinstance(control_step, torch.Tensor)
            else -1,
            "physics_substep": int(pending["physics_substep"]),
            "reference_index": int(reference_index[env_id].item())
            if isinstance(reference_index, torch.Tensor)
            else -1,
            "object_body_name": object_name,
            "contact_count": int(pair_presence.sum().item()),
            "present_hand_body_names": [
                name
                for name, present in zip(
                    HAND_COLLISION_BODY_NAMES, pair_presence.tolist(), strict=True
                )
                if present
            ],
            "contact_detection": "object-side filtered force-matrix norm exceeds sensor threshold",
            "force_frame": "world; aggregate force on object from filtered hand bodies",
            "net_contact_force_world_on_object_n": force.cpu().tolist(),
            "impulse_world_on_object_ns": impulse_on_object.cpu().tolist(),
            "raw_point_fields": "UNAVAILABLE; no point force is inferred from aggregate telemetry",
            "object_linear_velocity_before_world_mps": object_before[7:10].detach().cpu().tolist(),
            "object_linear_velocity_after_world_mps": object_after[7:10].detach().cpu().tolist(),
            "object_angular_velocity_before_world_radps": object_before[10:13]
            .detach()
            .cpu()
            .tolist(),
            "object_angular_velocity_after_world_radps": object_after[10:13]
            .detach()
            .cpu()
            .tolist(),
            "object_delta_v_world_mps": delta_v.detach().cpu().tolist(),
            "object_delta_omega_world_radps": delta_omega.detach().cpu().tolist(),
            "object_delta_momentum_world_ns": (object_mass_kg * delta_v).detach().cpu().tolist(),
            "object_pose_world": object_after[:7].detach().cpu().tolist(),
            "applied_wrist_force_world_n": applied_force_world[env_id].detach().cpu().tolist(),
            "applied_wrist_torque_world_nm": applied_torque_world[env_id].detach().cpu().tolist(),
            "aggregation_level": "object-centric filtered aggregate",
        }
        if pair_force_on_object is not None:
            record["pair_force_world_on_object_n"] = pair_force_on_object.detach().cpu().tolist()
            record["filter_body_order"] = list(HAND_COLLISION_BODY_NAMES)
            record["aggregation_level"] = "object-centric body-pair force matrix"
        return record

    def hand_collision_inventory(self) -> dict[str, object]:
        """Return the live sensor/body coverage contract without proxy-count assumptions."""

        return {
            "collision_bodies_configured": list(HAND_COLLISION_BODY_NAMES),
            "runtime_robot_body_count": len(self._robot.body_names),
            "sensor_coverage_complete": all(
                sensor.contact_physx_view.filter_count == len(HAND_COLLISION_BODY_NAMES)
                for sensor in self._object_contact_sensors.values()
            ),
            "sensor_count": len(self._object_contact_sensors),
            "object_sensor_bodies": {
                object_name: list(sensor.body_names)
                for object_name, sensor in self._object_contact_sensors.items()
            },
            "filter_count_per_sensor": {
                object_name: sensor.contact_physx_view.filter_count
                for object_name, sensor in self._object_contact_sensors.items()
            },
            "filter_prim_paths": list(
                self._object_contact_sensors["Object170105"].cfg.filter_prim_paths_expr
            ),
        }

    def contact_sensor_contract(self) -> dict[str, object]:
        return {
            "api": "isaaclab.sensors.ContactSensorCfg one-object-per-view force_matrix_w",
            "update_period_s": self._object_contact_sensors["Object170105"].cfg.update_period,
            "force_frame": "world; force_matrix_w is force on object sensor body",
            "force_matrix_shapes": {
                object_name: list(sensor.data.force_matrix_w.shape)
                for object_name, sensor in self._object_contact_sensors.items()
            },
            "telemetry_mode": self.cfg.contact_telemetry,
            "optional_point_and_tangent_fields": "UNAVAILABLE; no fake point force",
            "body_sensor_count": len(self._object_contact_sensors),
            "max_contact_data_per_body": self._object_contact_sensors[
                "Object170105"
            ].cfg.max_contact_data_count_per_prim,
            "pair_filtering": (
                "each HOCap object uses one object-side sensor filtered to all "
                "21 hand collision bodies"
            ),
            "record_transport": {
                "policy": "bounded_latest_only",
                "capacity": self.cfg.contact_record_capacity,
                "total_samples": self._contact_substep_record_total,
                "retained_samples": len(self._contact_substep_records),
                "dropped_samples": max(
                    0, self._contact_substep_record_total - len(self._contact_substep_records)
                ),
            },
        }

    @property
    def contact_substep_records(self) -> list[dict[str, object]]:
        return list(self._contact_substep_records)

    @property
    def wrist_diagnostic_records(self) -> list[dict[str, object]]:
        return list(self._wrist_diagnostic_records)

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
        self._record_completed_contact_substep()
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
            "force_saturation_ratio": (
                self._force_saturation_substeps.to(torch.float32)
                / self._wrist_substeps.clamp_min(1).to(torch.float32)
            ),
            "torque_saturation_ratio": (
                self._torque_saturation_substeps.to(torch.float32)
                / self._wrist_substeps.clamp_min(1).to(torch.float32)
            ),
            "wrist_substeps": self._wrist_substeps.clone(),
            "identified_map_condition_number": self._identified_map_condition_number.clone(),
            "identified_map_condition_gate_pass": self._identified_map_condition_gate_pass.clone(),
            "identified_map_selected_reference_frame": (
                self._identified_map_selected_reference_frame.clone()
            ),
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
        self._force_saturation_substeps[env_ids] = 0
        self._torque_saturation_substeps[env_ids] = 0
        self._wrist_substeps[env_ids] = 0
        self._identified_map_condition_number[env_ids] = 0.0
        self._identified_map_condition_gate_pass[env_ids] = True
        self._identified_map_selected_reference_frame[env_ids] = 0
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
        finite_virtual_actuator = getattr(self, "finite_virtual_wrist_actuator", None)
        if finite_virtual_actuator is not None:
            # Controller-only virtual-coordinate reset, paired with normal
            # episode setup.  It does not modify the physical wrist state.
            finite_virtual_actuator.reset(
                position_world=wrist_position,
                quaternion_wxyz=wrist_quaternion,
                env_ids=env_ids,
                num_envs=self.num_envs,
            )
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
            "identified_inverse_wrench": (
                None
                if self.wrist_controller_inverse is None
                else {
                    "profile": self.wrist_controller_inverse.profile.identifier,
                    "map_path": self.wrist_controller_inverse.effective_map.source_path,
                    "regularization": self.wrist_controller_inverse.regularization,
                }
            ),
            "finite_virtual_6d_wrist_actuator": (
                None
                if self.finite_virtual_wrist_actuator is None
                else {
                    "identifier": "finite_virtual_6d_wrist_actuator_v1",
                    "profile": self.finite_virtual_wrist_actuator.profile.identifier,
                    "orientation_target": "quaternion_then_rotation_log",
                    "virtual_joint_order": [
                        "virtual_prismatic_x",
                        "virtual_prismatic_y",
                        "virtual_prismatic_z",
                        "virtual_revolute_x",
                        "virtual_revolute_y",
                        "virtual_revolute_z",
                    ],
                    "rotation_singularity_margin_deg": 80.0,
                    "authority_enabled": bool(self.cfg.finite_virtual_wrist_authority_enabled),
                    "state_writes_during_step": 0,
                }
            ),
        }


__all__ = ["IsaacWorldWristFingerDirectRLEnv"]
