"""Stage 16-D.5 PPO environment with no terminal-success entry dependency.

Only this optional runtime module imports IsaacLab. It retains the validated
Stage16-D physics stack and base 26-D action adapter, but replaces old
terminal/contact/geometry PPO blocking with training-time numerical safety.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import HAND_COLLISION_BODY_NAMES
from toporetarget.rl.ppo.ppo26d_contract import Stage16DPPO26DObservationV2
from toporetarget.rl.reference_tracking.contact_reward_mode import (
    build_contact_reward,
    resolve_contact_mode,
    validate_frozen_contact_contract,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV1,
    TopoRetargetReferenceTrackingReward26DV2,
    TopoRetargetReferenceTrackingReward26DV3,
    TopoRetargetReferenceTrackingReward26DV4,
    ppo26d_reward_v2_object_twist_terms,
    ppo26d_reward_v3_reference_gated_contact_terms,
    ppo26d_reward_v4_strict_per_finger_contact_terms,
)
from toporetarget.rl.reference_tracking.reference_gated_contact import (
    fingertip_force_indices,
)

from .ppo26d_reference_tracking_env_cfg import IsaacPPO26DReferenceTrackingEnvCfg
from .reference_bank import quaternion_to_matrix_wxyz
from .scene_frame import global_to_scene, scene_to_global
from .tensor_math import relative_rotation_log_local
from .world_wrist_direct_env import IsaacWorldWristFingerDirectRLEnv


class IsaacPPO26DReferenceTrackingEnv(IsaacWorldWristFingerDirectRLEnv):
    """Reference-residual PPO environment using current Stage16-D physics only."""

    cfg: IsaacPPO26DReferenceTrackingEnvCfg

    def __init__(
        self,
        cfg: IsaacPPO26DReferenceTrackingEnvCfg,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._observation_contract = Stage16DPPO26DObservationV2()
        self._reward_contract: (
            TopoRetargetReferenceTrackingReward26DV1
            | TopoRetargetReferenceTrackingReward26DV2
            | TopoRetargetReferenceTrackingReward26DV3
            | TopoRetargetReferenceTrackingReward26DV4
        )
        self._reference_contact_mask_by_clip: torch.Tensor | None = None
        self._fingertip_force_sensor_indices: torch.Tensor | None = None
        self._ppo26d_trace_capture_exact_pair_force = False
        self._contact_reward_receipt: dict[str, object] | None = None
        contact_mode = resolve_contact_mode(
            configured_mode=cfg.ppo26d_contact_reward_mode,
            reward_contract_identifier=cfg.ppo26d_reward_contract,
        )
        if contact_mode is not None:
            if cfg.reference_kinematics_version != 2:
                raise RuntimeError("PPO26D_CONTACT_REWARD_REQUIRES_REFERENCE_KINEMATICS_V2")
            if cfg.ppo26d_contact_reward_contract_path is None:
                raise RuntimeError("PPO26D_CONTACT_REWARD_CONTRACT_PATH_MISSING")
            receipt = json.loads(
                Path(cfg.ppo26d_contact_reward_contract_path).read_text(encoding="utf-8")
            )
            parameters = validate_frozen_contact_contract(contact_mode, receipt)
            self._contact_reward_receipt = receipt
            self._reward_contract = build_contact_reward(contact_mode, frozen_parameters=parameters)
        elif cfg.ppo26d_reward_contract == "TopoRetargetReferenceTrackingReward26DV2":
            if cfg.reference_kinematics_version != 2:
                raise RuntimeError("PPO26D_REWARD_V2_REQUIRES_REFERENCE_KINEMATICS_V2")
            self._reward_contract = TopoRetargetReferenceTrackingReward26DV2()
        elif cfg.ppo26d_reward_contract == "TopoRetargetReferenceTrackingReward26DV1":
            self._reward_contract = TopoRetargetReferenceTrackingReward26DV1()
        else:
            raise ValueError(f"unknown PPO26D reward contract: {cfg.ppo26d_reward_contract}")
        self._rsi_start_counts: torch.Tensor | None = None
        self._ppo26d_safety_counts: dict[str, int] = {}
        self._ppo26d_object_write_baseline: torch.Tensor | None = None
        self._ppo26d_wrist_write_baseline: torch.Tensor | None = None
        self._ppo26d_trace_capture: dict[str, torch.Tensor] | None = None
        self._ppo26d_trace_enabled = False
        self._ppo26d_trace_capacity = 0
        self._ppo26d_trace_length = 0
        self._ppo26d_last_terminated: torch.Tensor | None = None
        self._ppo26d_last_timed_out: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)
        if isinstance(
            self._reward_contract,
            (
                TopoRetargetReferenceTrackingReward26DV2,
                TopoRetargetReferenceTrackingReward26DV3,
                TopoRetargetReferenceTrackingReward26DV4,
            ),
        ) and (
            self.reference_bank.manifest.identifier != "world_wrist_reference_bank_kinematics_v2"
            or self.reference_bank.manifest.reference_time_scale != 8
            or self.reference_bank.frame_count != 321
        ):
            raise RuntimeError("PPO26D_REWARD_V2_REQUIRES_MATERIALIZED_REFERENCE_KINEMATICS_V2")
        # The base implementation already supplies the required physical-state
        # reward inputs.  Install the explicit PPO contract so its paper terms
        # and the added controllable-wrist terms are the single source of truth.
        self.reward_profile = self._reward_contract.profile()
        self._rsi_start_counts = torch.zeros(
            self.reference_bank.frame_count, dtype=torch.long, device=self.device
        )
        self._ppo26d_object_write_baseline = self._object_state_write_count.clone()
        self._ppo26d_wrist_write_baseline = self._wrist_step_state_write_count.clone()
        # This name-derived mapping is telemetry infrastructure, not a V3 reward
        # input.  Keeping it available to all reward versions lets a frozen V1
        # policy be re-evaluated with an exact, read-only contact trace.
        self._fingertip_force_sensor_indices = torch.as_tensor(
            fingertip_force_indices(HAND_COLLISION_BODY_NAMES),
            dtype=torch.long,
            device=self.device,
        )
        if isinstance(
            self._reward_contract,
            (TopoRetargetReferenceTrackingReward26DV3, TopoRetargetReferenceTrackingReward26DV4),
        ):
            paths = cfg.ppo26d_reference_contact_mask_paths
            if paths is None or set(paths) != set(self.reference_bank.clip_ids):
                raise RuntimeError("PPO26D_REWARD_V3_CONTACT_MASK_PATHS_INVALID")
            masks: list[np.ndarray] = []
            for clip in self.reference_bank.clip_ids:
                with np.load(Path(paths[clip]), allow_pickle=False) as archive:
                    field = (
                        "reference_expected_contact_mask"
                        if isinstance(
                            self._reward_contract, TopoRetargetReferenceTrackingReward26DV3
                        )
                        else "strict_source_contact_mask"
                    )
                    if field not in archive.files:
                        raise RuntimeError("STRICT_V4_SOURCE_CONTACT_MASK_FIELD_MISSING")
                    mask = np.asarray(archive[field], dtype=bool)
                if mask.shape != (self.reference_bank.frame_count, 5):
                    raise RuntimeError(
                        "PPO26D_REWARD_V3_CONTACT_MASK_SHAPE_INVALID:"
                        f"clip={clip}:shape={mask.shape}"
                    )
                masks.append(mask)
            self._reference_contact_mask_by_clip = torch.as_tensor(
                np.stack(masks), dtype=torch.bool, device=self.device
            )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        super()._reset_idx(env_ids)
        ids = (
            self._robot._ALL_INDICES
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        if self.cfg.ppo26d_enable_critical_dr and ids.numel():
            self._apply_critical_reset_noise(ids)
        if self._ppo26d_object_write_baseline is not None:
            self._ppo26d_object_write_baseline[ids] = self._object_state_write_count[ids]
        if self._ppo26d_wrist_write_baseline is not None:
            self._ppo26d_wrist_write_baseline[ids] = self._wrist_step_state_write_count[ids]
        if self._rsi_start_counts is None:
            return
        if env_ids is None:
            indices = self._reference_index
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            indices = self._reference_index[ids]
        self._rsi_start_counts += torch.bincount(indices, minlength=self.reference_bank.frame_count)

    @staticmethod
    def _quaternion_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        fw, fx, fy, fz = first.unbind(dim=-1)
        sw, sx, sy, sz = second.unbind(dim=-1)
        return torch.stack(
            (
                fw * sw - fx * sx - fy * sy - fz * sz,
                fw * sx + fx * sw + fy * sz - fz * sy,
                fw * sy - fx * sz + fy * sw + fz * sx,
                fw * sz + fx * sy - fy * sx + fz * sw,
            ),
            dim=-1,
        )

    def _apply_critical_reset_noise(self, env_ids: torch.Tensor) -> None:
        """Apply only paper reset perturbations, while reset state writes are allowed."""

        clips = self._clip_index[env_ids]
        frames = self._reference_index[env_ids]
        joint_noise = torch.empty((env_ids.numel(), 20), device=self.device).uniform_(
            -self.cfg.ppo26d_reset_joint_noise_rad,
            self.cfg.ppo26d_reset_joint_noise_rad,
        )
        finger_target = torch.clamp(
            self.reference_bank.q_finger_ref[clips, frames] + joint_noise,
            self.joint_lower,
            self.joint_upper,
        )
        self._robot.write_joint_state_to_sim(
            self.action_adapter.canonical_to_isaac(finger_target),
            self.action_adapter.canonical_to_isaac(
                self.reference_bank.qdot_finger_ref[clips, frames]
            ),
            joint_ids=self._finger_target_joint_ids,
            env_ids=env_ids,
        )
        active_position = scene_to_global(
            self.reference_bank.object_pose_translation_world_ref[clips, frames],
            self.scene.env_origins[env_ids],
        )
        active_position += torch.empty_like(active_position).uniform_(
            -self.cfg.ppo26d_reset_object_position_noise_m,
            self.cfg.ppo26d_reset_object_position_noise_m,
        )
        axis = torch.randn((env_ids.numel(), 3), device=self.device)
        axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-12)
        angle = torch.empty((env_ids.numel(), 1), device=self.device).uniform_(
            -self.cfg.ppo26d_reset_object_orientation_noise_rad,
            self.cfg.ppo26d_reset_object_orientation_noise_rad,
        )
        delta_quaternion = torch.cat(
            (torch.cos(angle * 0.5), axis * torch.sin(angle * 0.5)), dim=-1
        )
        active_quaternion = self._quaternion_multiply(
            self.reference_bank.object_pose_quaternion_world_ref_wxyz[clips, frames],
            delta_quaternion,
        )
        active_state = torch.cat(
            (
                active_position,
                active_quaternion,
                self.reference_bank.object_twist_world_ref[clips, frames],
            ),
            dim=-1,
        )
        first_object = clips == 0
        if bool(first_object.any()):
            self._object_170105.write_root_state_to_sim(
                active_state[first_object], env_ids=env_ids[first_object]
            )
        if bool((~first_object).any()):
            self._object_170650.write_root_state_to_sim(
                active_state[~first_object], env_ids=env_ids[~first_object]
            )
        self._object_state_write_count[env_ids] += 1

    def _pose6(self, position: torch.Tensor, quaternion_wxyz: torch.Tensor) -> torch.Tensor:
        identity = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=quaternion_wxyz.dtype
        ).expand_as(quaternion_wxyz)
        return torch.cat((position, relative_rotation_log_local(identity, quaternion_wxyz)), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Freeze the V2 semantic map at 764 dimensions before PPO training."""

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
        object_ref_position = self.reference_bank.gather(
            "object_pose_translation_world_ref", self._clip_index, index
        )
        object_ref_quaternion = self.reference_bank.gather(
            "object_pose_quaternion_world_ref_wxyz", self._clip_index, index
        )
        chunks = [
            self._pose6(state["wrist_position_scene"], state["wrist_quaternion_wxyz"]),
            wrist_error,
            state["wrist_twist_world"],
            state["finger_q"],
            state["finger_qdot"],
            self._previous_actions,
            state["object_axis_points_scene"].flatten(1),
            torch.cat((object_relative_translation, object_relative_rotation), dim=-1),
            state["object_twist_world"],
            self._pose6(state["object_position_scene"], state["object_quaternion_wxyz"]),
            self._pose6(object_ref_position, object_ref_quaternion),
        ]
        for offset in self._observation_contract.lookahead_offsets:
            reference_index = torch.clamp(index + offset, max=self.reference_bank.frame_count - 1)
            reference_wrist_position = self.reference_bank.gather(
                "wrist_pose_translation_world_ref", self._clip_index, reference_index
            )
            reference_wrist_quaternion = self.reference_bank.gather(
                "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, reference_index
            )
            chunks.extend(
                (
                    self._pose6(reference_wrist_position, reference_wrist_quaternion),
                    self.reference_bank.gather(
                        "wrist_twist_world_ref", self._clip_index, reference_index
                    ),
                    self.reference_bank.gather("q_finger_ref", self._clip_index, reference_index),
                    self.reference_bank.gather(
                        "object_axis_points_world_ref", self._clip_index, reference_index
                    ).flatten(1),
                )
            )
            if offset != 0:
                chunks.append(
                    self.reference_bank.gather(
                        "object_axis_points_wrist_ref", self._clip_index, reference_index
                    ).flatten(1)
                )
            chunks.extend(
                (
                    self.reference_bank.gather(
                        "tracked_link_positions_world_ref", self._clip_index, reference_index
                    ).flatten(1),
                    self.reference_bank.gather(
                        "tracked_link_positions_wrist_ref", self._clip_index, reference_index
                    ).flatten(1),
                )
            )
        observation = torch.cat(chunks, dim=-1)
        expected = (self.num_envs, self._observation_contract.dimension)
        if observation.shape != expected or not bool(torch.isfinite(observation).all()):
            raise RuntimeError(f"PPO26D_OBSERVATION_INVALID: shape={tuple(observation.shape)}")
        if self.cfg.ppo26d_enable_critical_dr:
            observation = observation.clone()
            observation[:, 18:38] += torch.randn_like(observation[:, 18:38]) * (
                self.cfg.ppo26d_observation_joint_position_noise_std_rad
            )
            observation[:, 38:58] += torch.randn_like(observation[:, 38:58]) * (
                self.cfg.ppo26d_observation_joint_velocity_noise_std_radps
            )
            observation[:, 84:102] += torch.randn_like(observation[:, 84:102]) * (
                self.cfg.ppo26d_observation_axis_position_noise_std_m
            )
        return {"policy": observation}

    def _active_object_pair_force_matrix(self) -> torch.Tensor:
        """Return `[N,21,3]` forces on the active object from each hand body.

        The object-side filtered sensor excludes self-collision, inactive object,
        palm-only aggregation, and scene/support contacts by construction.
        """

        first_force = self._object_contact_sensors["Object170105"].data.force_matrix_w
        second_force = self._object_contact_sensors["Object170650"].data.force_matrix_w
        if first_force is None or second_force is None:
            raise RuntimeError("PPO26D reward requires object contact force matrices")
        expected = (self.num_envs, 1, len(HAND_COLLISION_BODY_NAMES), 3)
        if tuple(first_force.shape) != expected or tuple(second_force.shape) != expected:
            raise RuntimeError(
                "PPO26D_CONTACT_PAIR_FORCE_SHAPE_INVALID:"
                f"first={tuple(first_force.shape)} second={tuple(second_force.shape)}"
            )
        pair_force = torch.where(
            (self._clip_index == 0)[:, None, None], first_force[:, 0], second_force[:, 0]
        )
        if not bool(torch.isfinite(pair_force).all()):
            raise FloatingPointError("PPO26D_CONTACT_PAIR_FORCE_NONFINITE")
        return pair_force

    def _reference_expected_contact_mask(self, index: torch.Tensor) -> torch.Tensor:
        masks = self._reference_contact_mask_by_clip
        if masks is None:
            raise RuntimeError("PPO26D_REWARD_V3_CONTACT_MASK_NOT_INITIALIZED")
        return masks[self._clip_index, index]

    def _get_rewards(self) -> torch.Tensor:
        first_difference = self._actions - self._previous_actions
        second_difference = (
            self._actions - 2.0 * self._previous_actions + self._second_previous_actions
        )
        if isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV4):
            state = self._state()
            index = self._target_reference_index
            force_indices = self._fingertip_force_sensor_indices
            if force_indices is None:
                raise RuntimeError("STRICT_V4_FINGERTIP_FORCE_MAPPING_NOT_INITIALIZED")
            pair_force = self._active_object_pair_force_matrix()
            tip_force = pair_force.index_select(1, force_indices)
            terms = ppo26d_reward_v4_strict_per_finger_contact_terms(
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
                object_twist_world=state["object_twist_world"],
                object_twist_world_ref=self.reference_bank.gather(
                    "object_twist_world_ref", self._clip_index, index
                ),
                source_contact_mask=self._reference_expected_contact_mask(index),
                fingertip_object_pair_force_world=tip_force,
                fingertip_object_pair_presence=torch.linalg.vector_norm(tip_force, dim=-1)
                > self._reward_contract.contact_numerical_floor_n,
                profile=self._reward_contract,
            )
            self._last_reward_terms = terms
            self._second_previous_actions.copy_(self._previous_actions)
            self._previous_actions.copy_(self._actions)
            self._reference_index.copy_(self._target_reference_index)
            reward = terms["total"]
        elif isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV3):
            state = self._state()
            index = self._target_reference_index
            force_indices = self._fingertip_force_sensor_indices
            if force_indices is None:
                raise RuntimeError("PPO26D_REWARD_V3_FINGERTIP_FORCE_MAPPING_NOT_INITIALIZED")
            pair_force = self._active_object_pair_force_matrix()
            terms = ppo26d_reward_v3_reference_gated_contact_terms(
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
                object_twist_world=state["object_twist_world"],
                object_twist_world_ref=self.reference_bank.gather(
                    "object_twist_world_ref", self._clip_index, index
                ),
                reference_expected_contact_mask=self._reference_expected_contact_mask(index),
                fingertip_object_pair_force_world=pair_force.index_select(1, force_indices),
                profile=self._reward_contract,
            )
            self._last_reward_terms = terms
            self._second_previous_actions.copy_(self._previous_actions)
            self._previous_actions.copy_(self._actions)
            self._reference_index.copy_(self._target_reference_index)
            reward = terms["total"]
        elif isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV2):
            state = self._state()
            index = self._target_reference_index
            terms = ppo26d_reward_v2_object_twist_terms(
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
                object_twist_world=state["object_twist_world"],
                object_twist_world_ref=self.reference_bank.gather(
                    "object_twist_world_ref", self._clip_index, index
                ),
                profile=self._reward_contract,
            )
            self._last_reward_terms = terms
            self._second_previous_actions.copy_(self._previous_actions)
            self._previous_actions.copy_(self._actions)
            self._reference_index.copy_(self._target_reference_index)
            reward = terms["total"]
        else:
            reward = super()._get_rewards()
        self._last_reward_terms = {
            **self._last_reward_terms,
            "r_object": self._last_reward_terms["object"],
            "r_link": self._last_reward_terms["tracked_links"],
            "r_finger": self._last_reward_terms["finger_joints"],
            "r_wrist_translation": self._last_reward_terms["wrist_position"],
            "r_wrist_rotation": self._last_reward_terms["wrist_rotation"],
            "r_wrist": self._last_reward_terms["wrist_position"]
            + self._last_reward_terms["wrist_rotation"],
            "smoothness_wrist": first_difference[:, :6].square().sum(dim=-1)
            + second_difference[:, :6].square().sum(dim=-1),
            "smoothness_finger": first_difference[:, 6:26].square().sum(dim=-1)
            + second_difference[:, 6:26].square().sum(dim=-1),
        }
        pair_force = self._active_object_pair_force_matrix()
        self.extras["ppo26d"].update(
            {
                "contact_any": (torch.linalg.vector_norm(pair_force, dim=-1) > 1.0e-4)
                .any(dim=-1)
                .clone(),
                "actuator_effort": self._robot.data.applied_torque.clone(),
            }
        )
        if isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV3):
            self.extras["ppo26d"].update(
                {
                    "reference_expected_contact_mask": self._last_reward_terms[
                        "reference_expected_contact_mask"
                    ].clone(),
                    "actual_fingertip_object_contact_mask": self._last_reward_terms[
                        "actual_fingertip_object_contact_mask"
                    ].clone(),
                    "contact_reward": self._last_reward_terms["r_contact"].clone(),
                }
            )
        elif isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV4):
            force_indices = self._fingertip_force_sensor_indices
            if force_indices is None:
                raise RuntimeError("STRICT_V4_FINGERTIP_FORCE_MAPPING_NOT_INITIALIZED")
            self.extras["ppo26d"].update(
                {
                    "source_contact_mask": self._last_reward_terms["source_contact_mask"].clone(),
                    "tip_pair_presence": self._last_reward_terms["tip_pair_presence"].clone(),
                    "tip_pair_force_world": pair_force.index_select(1, force_indices).clone(),
                    "tip_pair_force_norm": self._last_reward_terms["tip_pair_force_norm_n"].clone(),
                    "per_finger_contact_reward": self._last_reward_terms[
                        "per_finger_contact_reward"
                    ].clone(),
                    "source_expected_finger_count": self._last_reward_terms[
                        "source_expected_finger_count"
                    ].clone(),
                    "source_satisfied_tip_count": self._last_reward_terms[
                        "source_satisfied_tip_count"
                    ].clone(),
                    "source_tip_coverage_ratio": self._last_reward_terms[
                        "source_tip_coverage_ratio"
                    ].clone(),
                    "r_contact_v4": self._last_reward_terms["r_contact_v4"].clone(),
                }
            )
        self._capture_ppo26d_trace_row()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate B safety only; task/contact/geometry are post-PPO diagnostics."""

        self._record_completed_contact_substep()
        write_report = self.rollout_state_write_report()
        if (
            write_report["object_rollout_state_writes"]
            or write_report["wrist_root_state_writes_during_step"]
        ):
            raise RuntimeError("PPO26D_ROLLOUT_STATE_WRITE_FORBIDDEN")
        state = self._state()
        object_twist = state["object_twist_world"]
        object_speed = torch.linalg.vector_norm(object_twist[:, :3], dim=-1)
        object_angular_speed = torch.linalg.vector_norm(object_twist[:, 3:], dim=-1)
        wrist_reference = self.reference_bank.gather(
            "wrist_pose_translation_world_ref", self._clip_index, self._target_reference_index
        )
        wrist_workspace = (
            torch.linalg.vector_norm(state["wrist_position_scene"] - wrist_reference, dim=-1)
            > self.cfg.ppo26d_workspace_radius_m
        )
        finite = torch.stack(
            [torch.isfinite(value).flatten(1).all(dim=-1) for value in state.values()], dim=-1
        ).all(dim=-1)
        joint_safe = (
            (state["finger_q"] >= self.joint_lower) & (state["finger_q"] <= self.joint_upper)
        ).all(dim=-1)
        action_valid = torch.isfinite(self._actions).all(dim=-1) & (self._actions.abs() <= 1.0).all(
            dim=-1
        )
        object_linear_failure = object_speed > self.cfg.ppo26d_object_linear_speed_max_mps
        object_angular_failure = (
            object_angular_speed > self.cfg.ppo26d_object_angular_speed_max_radps
        )
        numerical_failure = ~finite
        joint_failure = ~joint_safe
        action_failure = ~action_valid
        diagnostic_terminated = (
            numerical_failure
            | joint_failure
            | action_failure
            | object_linear_failure
            | object_angular_failure
            | wrist_workspace
        )
        # C4 Formal20 must export the actual continuation of every finite
        # trajectory, including physically bad outcomes.  Do not reset merely
        # for diagnostic speed/joint/workspace failures; non-finite state or
        # command remains a genuine technical runtime failure.
        terminated = (
            numerical_failure | action_failure
            if self.cfg.ppo26d_full_horizon_evaluation
            else diagnostic_terminated
        )
        timeout = self._target_reference_index >= self.reference_bank.frame_count - 1
        timed_out = timeout & ~terminated
        reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for mask, code in (
            (timed_out, 7),
            (wrist_workspace, 6),
            (object_angular_failure, 5),
            (object_linear_failure, 4),
            (action_failure, 3),
            (joint_failure, 2),
            (numerical_failure, 1),
        ):
            reason = torch.where(mask, code, reason)
        labels = (
            "NONE",
            "FAILURE_NUMERICAL",
            "FAILURE_JOINT_LIMIT",
            "FAILURE_ACTION_INVALID",
            "FAILURE_OBJECT_LINEAR_VELOCITY",
            "FAILURE_OBJECT_ANGULAR_VELOCITY",
            "FAILURE_WRIST_WORKSPACE",
            "TIMEOUT_REFERENCE_END",
        )
        self._ppo26d_safety_counts = {
            label: int((reason == code).sum().item())
            for code, label in enumerate(labels)
            if code and bool((reason == code).any())
        }
        extras = {
            "primary_reason_code": reason.clone(),
            "termination_reasons": labels,
            "object_linear_speed_mps": object_speed.clone(),
            "object_angular_speed_radps": object_angular_speed.clone(),
            "wrist_workspace_violation": wrist_workspace.clone(),
            "joint_safe": joint_safe.clone(),
            "action_valid": action_valid.clone(),
            "finite": finite.clone(),
            "diagnostic_terminated": diagnostic_terminated.clone(),
            "full_horizon_evaluation": bool(self.cfg.ppo26d_full_horizon_evaluation),
            "terminal_contact_is_post_ppo_diagnostic": True,
            "terminal_stability_is_post_ppo_diagnostic": True,
            **write_report,
        }
        self.extras["ppo26d"] = extras
        self._ppo26d_last_terminated = terminated.clone()
        self._ppo26d_last_timed_out = timed_out.clone()
        return terminated, timed_out

    def start_trace_capture(
        self,
        *,
        capacity: int,
        capture_exact_fingertip_object_pair_force: bool = False,
        capture_full_hand_object_pair_telemetry: bool = False,
    ) -> None:
        """Capture one post-physics row per PPO step without host transfers.

        DirectRLEnv invokes ``_get_dones`` and ``_get_rewards`` on the
        simulator thread.  Isaac Sim can terminate without a Python exception
        when articulation collision-body tensors are accessed across the done
        boundary or copied to CPU there.  Therefore this capture contains only
        stable post-physics state (physical wrist, canonical fingers, object,
        contacts, effort, and action) on the device.  The evaluator rebuilds
        collision-body poses with offline FK after the rollout has ended.
        """

        if capacity <= 0:
            raise ValueError("PPO26D trace capacity must be positive")
        if self._ppo26d_trace_enabled:
            raise RuntimeError("PPO26D trace capture is already active")
        # Allocate lazily inside _get_rewards, after Isaac has completed its
        # post-physics tensor update.  Device allocation immediately after
        # reset can itself close an Isaac Sim 5 app without an exception.
        self._ppo26d_trace_capture = None
        self._ppo26d_trace_enabled = True
        self._ppo26d_trace_capture_exact_pair_force = capture_exact_fingertip_object_pair_force
        self._ppo26d_trace_capture_full_hand_object_pair = capture_full_hand_object_pair_telemetry
        self._ppo26d_trace_capacity = capacity
        self._ppo26d_trace_length = 0

    def finish_trace_capture(self) -> dict[str, torch.Tensor]:
        """Return device-resident rows captured since ``start_trace_capture``."""

        capture = self._ppo26d_trace_capture
        if not self._ppo26d_trace_enabled or capture is None:
            raise RuntimeError("PPO26D trace capture was not started")
        length = self._ppo26d_trace_length
        self._ppo26d_trace_capture = None
        self._ppo26d_trace_enabled = False
        self._ppo26d_trace_capture_exact_pair_force = False
        self._ppo26d_trace_capture_full_hand_object_pair = False
        self._ppo26d_trace_capacity = 0
        self._ppo26d_trace_length = 0
        if length == 0:
            raise RuntimeError("PPO26D trace capture contains no rollout rows")
        return {name: rows[:length] for name, rows in capture.items()}

    def _capture_ppo26d_trace_row(self) -> None:
        if not self._ppo26d_trace_enabled:
            return
        step = self._ppo26d_trace_length
        if step >= self._ppo26d_trace_capacity:
            raise RuntimeError("PPO26D trace capacity exhausted")
        state = self._state()
        pair_force = self._active_object_pair_force_matrix()
        effort = self._robot.data.applied_torque
        terminated = self._ppo26d_last_terminated
        timed_out = self._ppo26d_last_timed_out
        if terminated is None or timed_out is None:
            raise RuntimeError("PPO26D trace capture requires done flags")
        if effort.shape != (self.num_envs, 26):
            raise RuntimeError(f"PPO26D actuator effort has unexpected shape {tuple(effort.shape)}")
        virtual_wrist_ids = self._virtual_wrist_joint_ids
        if len(virtual_wrist_ids) != 6:
            raise RuntimeError(
                "PPO26D trace capture requires the finite six-DoF virtual wrist actuator"
            )
        values = {
            "object_pose": torch.cat(
                (state["object_position_scene"], state["object_quaternion_wxyz"]), dim=-1
            ),
            "object_twist": state["object_twist_world"],
            "wrist_pose": torch.cat(
                (state["wrist_position_scene"], state["wrist_quaternion_wxyz"]), dim=-1
            ),
            # _state() already gathers _joint_ids in canonical reference order.
            # Re-applying isaac_to_canonical here silently permutes trace joints twice.
            "finger_q": state["finger_q"],
            "finger_qdot": state["finger_qdot"],
            "wrist_twist_world": state["wrist_twist_world"],
            "virtual_wrist_q": self._robot.data.joint_pos[:, virtual_wrist_ids],
            "virtual_wrist_qdot": self._robot.data.joint_vel[:, virtual_wrist_ids],
            "virtual_wrist_target_q": self._explicit_wrist_joint_target,
            "virtual_wrist_target_qdot": self._explicit_wrist_joint_velocity_target,
            "object_axis_points": state["object_axis_points_scene"],
            "tracked_link_positions": state["tracked_links_scene"],
            "contact_force_world": pair_force.sum(dim=1),
            "contact_pair_presence": torch.linalg.vector_norm(pair_force, dim=-1) > 1.0e-4,
            "actuator_effort": effort,
            "reason_code": self.extras["ppo26d"]["primary_reason_code"],
            "terminated": terminated,
            "timed_out": timed_out,
            "action": self._actions,
            "clip_index": self._clip_index,
            "reward_total": self._last_reward_terms["total"],
            "reward_object": self._last_reward_terms["object"],
            "reward_link": self._last_reward_terms["tracked_links"],
            "reward_finger": self._last_reward_terms["finger_joints"],
            "reward_wrist_translation": self._last_reward_terms["wrist_position"],
            "reward_wrist_rotation": self._last_reward_terms["wrist_rotation"],
            "reward_smoothness": self._last_reward_terms["smoothness"],
            "wrist_residual": torch.cat(
                (self._wrist_translation_residual, self._wrist_rotation_residual), dim=-1
            ),
            "finger_residual": self._actions[:, 6:],
            "wrist_target": torch.cat(
                (
                    global_to_scene(self._wrist_target_position, self.scene.env_origins),
                    self._wrist_target_quaternion,
                ),
                dim=-1,
            ),
            "finger_target": self.action_adapter.isaac_to_canonical(self._joint_target_isaac),
            "object_reference": torch.cat(
                (
                    self.reference_bank.gather(
                        "object_pose_translation_world_ref",
                        self._clip_index,
                        self._reference_index,
                    ),
                    self.reference_bank.gather(
                        "object_pose_quaternion_world_ref_wxyz",
                        self._clip_index,
                        self._reference_index,
                    ),
                ),
                dim=-1,
            ),
            "wrist_reference": torch.cat(
                (
                    self.reference_bank.gather(
                        "wrist_pose_translation_world_ref",
                        self._clip_index,
                        self._reference_index,
                    ),
                    self.reference_bank.gather(
                        "wrist_pose_quaternion_world_ref_wxyz",
                        self._clip_index,
                        self._reference_index,
                    ),
                ),
                dim=-1,
            ),
            "finger_reference": self.reference_bank.gather(
                "q_finger_ref", self._clip_index, self._reference_index
            ),
            "tracked_link_reference": self.reference_bank.gather(
                "tracked_link_positions_world_ref", self._clip_index, self._reference_index
            ),
            "reference_index": self._reference_index,
        }
        if isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV2):
            values.update(
                {
                    "object_twist_reference": self.reference_bank.gather(
                        "object_twist_world_ref", self._clip_index, self._reference_index
                    ),
                    "reward_obj_vel": self._last_reward_terms["r_obj_vel"],
                    "reward_obj_ang_vel": self._last_reward_terms["r_obj_ang_vel"],
                    "error_obj_vel": self._last_reward_terms["e_obj_vel"],
                    "error_obj_ang_vel": self._last_reward_terms["e_obj_ang_vel"],
                }
            )
        if isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV3):
            force_indices = self._fingertip_force_sensor_indices
            if force_indices is None:
                raise RuntimeError("PPO26D_REWARD_V3_FINGERTIP_FORCE_MAPPING_NOT_INITIALIZED")
            values.update(
                {
                    "reference_contact_mask": self._last_reward_terms[
                        "reference_expected_contact_mask"
                    ],
                    "actual_contact_mask": self._last_reward_terms[
                        "actual_fingertip_object_contact_mask"
                    ],
                    "fingertip_object_pair_force_world": pair_force.index_select(1, force_indices),
                    "fingertip_object_force_magnitude": self._last_reward_terms[
                        "fingertip_object_force_magnitude_n"
                    ],
                    "contact_reward": self._last_reward_terms["r_contact"],
                    "contact_force_scale": self._last_reward_terms["contact_force_scale_n"],
                }
            )
        elif isinstance(self._reward_contract, TopoRetargetReferenceTrackingReward26DV4):
            force_indices = self._fingertip_force_sensor_indices
            if force_indices is None:
                raise RuntimeError("STRICT_V4_FINGERTIP_FORCE_MAPPING_NOT_INITIALIZED")
            values.update(
                {
                    "source_contact_mask": self._last_reward_terms["source_contact_mask"],
                    "tip_pair_presence": self._last_reward_terms["tip_pair_presence"],
                    "tip_pair_force_world": pair_force.index_select(1, force_indices),
                    "tip_pair_force_norm": self._last_reward_terms["tip_pair_force_norm_n"],
                    "per_finger_contact_reward": self._last_reward_terms[
                        "per_finger_contact_reward"
                    ],
                    "source_expected_finger_count": self._last_reward_terms[
                        "source_expected_finger_count"
                    ],
                    "source_satisfied_tip_count": self._last_reward_terms[
                        "source_satisfied_tip_count"
                    ],
                    "source_tip_coverage_ratio": self._last_reward_terms[
                        "source_tip_coverage_ratio"
                    ],
                    "r_contact_v4": self._last_reward_terms["r_contact_v4"],
                    # Common historical evaluator aliases, now sourced from the
                    # strict V4 term rather than a V3 aggregate term.
                    "reference_contact_mask": self._last_reward_terms["source_contact_mask"],
                    "actual_contact_mask": self._last_reward_terms["tip_pair_presence"],
                    "fingertip_object_pair_force_world": pair_force.index_select(1, force_indices),
                    "fingertip_object_force_magnitude": self._last_reward_terms[
                        "tip_pair_force_norm_n"
                    ],
                    "contact_reward": self._last_reward_terms["r_contact_v4"],
                }
            )
        if self._ppo26d_trace_capture_exact_pair_force:
            force_indices = self._fingertip_force_sensor_indices
            if force_indices is None:
                raise RuntimeError("PPO26D_TRACE_PAIR_FORCE_MAPPING_NOT_INITIALIZED")
            values.update(
                {
                    "fingertip_object_pair_force_world": pair_force.index_select(1, force_indices),
                    # Every post-physics row is a direct finite read from the
                    # active object-side filtered force matrix.  The evaluator
                    # supplies an explicitly-invalid reset row when it needs a
                    # 321-key trace.
                    "fingertip_object_pair_force_valid": torch.ones(
                        self.num_envs, dtype=torch.bool, device=self.device
                    ),
                }
            )
        if self._ppo26d_trace_capture_full_hand_object_pair:
            values.update(
                {
                    # This is the raw named 21-body filtered matrix.  It is
                    # observational and deliberately independent of reward version.
                    "hand_object_pair_force_world": pair_force,
                    "hand_object_pair_presence": torch.linalg.vector_norm(pair_force, dim=-1)
                    > 1.0e-4,
                    "hand_object_pair_force_valid": torch.ones(
                        self.num_envs, dtype=torch.bool, device=self.device
                    ),
                }
            )
        capture = self._ppo26d_trace_capture
        if capture is None:
            capture = {
                name: torch.empty(
                    (self._ppo26d_trace_capacity, *value.shape),
                    device=self.device,
                    dtype=value.dtype,
                )
                for name, value in values.items()
            }
            self._ppo26d_trace_capture = capture
        for name, value in values.items():
            capture[name][step].copy_(value.detach())
        self._ppo26d_trace_length += 1

    def rollout_state_write_report(self) -> dict[str, int]:
        """Count state writes since each environment's most recent reset.

        Reset/RSI writes are explicitly allowed.  Any post-reset object or wrist
        root write would be an impermissible shortcut and fails the rollout.
        """

        object_baseline = self._ppo26d_object_write_baseline
        wrist_baseline = self._ppo26d_wrist_write_baseline
        object_delta = (
            torch.zeros_like(self._object_state_write_count)
            if object_baseline is None
            else self._object_state_write_count - object_baseline
        )
        wrist_delta = (
            torch.zeros_like(self._wrist_step_state_write_count)
            if wrist_baseline is None
            else self._wrist_step_state_write_count - wrist_baseline
        )
        return {
            "object_rollout_state_writes": int(object_delta.sum().item()),
            "wrist_root_state_writes_during_step": int(wrist_delta.sum().item()),
        }

    def rsi_report(self) -> dict[str, Any]:
        counts = (
            (
                torch.zeros(self.reference_bank.frame_count, dtype=torch.long, device=self.device)
                if self._rsi_start_counts is None
                else self._rsi_start_counts
            )
            .detach()
            .cpu()
        )
        phase_frames = torch.tensor_split(torch.arange(counts.numel()), 4)
        return {
            "contract": "Stage16DPPO26DRSIV1",
            "frame_count": self.reference_bank.frame_count,
            "counts": counts.tolist(),
            "sample_count": int(counts.sum()),
            "phase_counts": {
                name: int(counts[frames].sum())
                for name, frames in zip(
                    ("approach", "first_contact", "persistent_contact", "terminal"),
                    phase_frames,
                    strict=True,
                )
            },
            "rollout_object_state_writes": self.rollout_state_write_report()[
                "object_rollout_state_writes"
            ],
            "rollout_wrist_root_state_writes": self.rollout_state_write_report()[
                "wrist_root_state_writes_during_step"
            ],
        }

    def contract_report(self) -> dict[str, object]:
        report = super().contract_report()
        clip_assignment = report.get("clip_assignment")
        if not isinstance(clip_assignment, dict):
            raise RuntimeError("PPO26D contract report is missing clip assignment evidence")
        report["ppo26d"] = {
            "action": self.action_adapter.contract.as_dict(),
            "action_semantic": "Stage16DReferenceResidualAction26DV1",
            "observation": self._observation_contract.as_dict(),
            "reward": self._reward_contract.as_dict(),
            "rsi": self.rsi_report(),
            "rsi_curriculum": {
                "phase": getattr(self.cfg, "curriculum_phase", None),
                "reset_reference_index": self.cfg.reset_reference_index,
                "support_count": len(getattr(self.cfg, "curriculum_reference_indices", ()) or ()),
            },
            "self_collision_enabled": bool(
                self.cfg.robot.spawn.articulation_props.enabled_self_collisions
            ),
            "fixed_clip": getattr(self.cfg, "stage16d_fixed_clip", None),
            "active_clip_indices": clip_assignment["active_clip_indices"],
            "active_clip_ids": clip_assignment["active_clip_ids"],
            **self.rollout_state_write_report(),
            "policy_direct_articulation_action": False,
            "policy_direct_object_action": False,
            "hidden_force_or_attachment": False,
            "domain_randomization": {
                "status": "PAPER_FIDELITY_PARTIAL",
                "critical_reset_noise": bool(self.cfg.ppo26d_enable_critical_dr),
                "critical_observation_noise": bool(self.cfg.ppo26d_enable_critical_dr),
                "deferred": [
                    "object_dynamics_scale",
                    "robot_dynamics_scale",
                    "external_disturbance",
                    "observation_delay",
                ],
            },
        }
        return report


__all__ = ["IsaacPPO26DReferenceTrackingEnv"]
