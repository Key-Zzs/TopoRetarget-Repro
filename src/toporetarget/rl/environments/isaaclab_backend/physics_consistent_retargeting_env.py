"""Stage 16-D free-object physics-correction DirectRLEnv.

The class subclasses the validated C.2 backend without mutating its behavior.
Only reward, semantic success, and termination are replaced. Object and wrist
state writes remain reset-only; rollout actions still flow through the 26-D
active wrist/finger controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from toporetarget.rl.physics_retargeting.contact_topology import body_contact_group
from toporetarget.rl.physics_retargeting.contracts import PhysicsConsistentTaskGateV1
from toporetarget.rl.physics_retargeting.rewards import (
    PhysicsConsistentRewardProfileV1,
    physics_consistent_reward_terms,
)
from toporetarget.rl.physics_retargeting.termination import (
    STAGE16D_TERMINATION_REASONS,
    physics_consistent_termination,
)

from .physics_consistent_retargeting_env_cfg import IsaacPhysicsConsistentRetargetingEnvCfg
from .tensor_math import quaternion_geodesic
from .world_wrist_direct_env import (
    HAND_COLLISION_BODY_NAMES,
    IsaacWorldWristFingerDirectRLEnv,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"STAGE16D_CONTRACT_MISSING: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Stage16D contract must be an object: {source}")
    return payload


class IsaacPhysicsConsistentRetargetingEnv(IsaacWorldWristFingerDirectRLEnv):
    cfg: IsaacPhysicsConsistentRetargetingEnvCfg

    def __init__(
        self, cfg: IsaacPhysicsConsistentRetargetingEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        self._semantic_contracts = {
            clip: _read_json(path) for clip, path in cfg.semantic_contract_paths.items()
        }
        topology_payload = _read_json(cfg.contact_topology_path)
        gate_payload = _read_json(cfg.task_gate_path)
        reward_payload = _read_json(cfg.reward_contract_path)
        topology_rows = topology_payload.get("clips")
        gate_rows = gate_payload.get("clips")
        if not isinstance(topology_rows, dict) or not isinstance(gate_rows, dict):
            raise ValueError("Stage16D topology/gate reports need clip mappings")
        self._topology_contracts = topology_rows
        self._task_gates = {
            clip: PhysicsConsistentTaskGateV1(
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"schema_version", "hard_gates"}
                },
                hard_gates=tuple(row["hard_gates"]),
            )
            for clip, row in gate_rows.items()
        }
        profile_row = reward_payload.get("profile")
        if not isinstance(profile_row, dict):
            raise ValueError("Stage16D reward report has no frozen profile")
        self.stage16d_reward_profile = PhysicsConsistentRewardProfileV1(
            **{key: value for key, value in profile_row.items() if key != "schema_version"}
        )
        self._group_slots: dict[str, list[int]] = {}
        for index, name in enumerate(HAND_COLLISION_BODY_NAMES):
            group = body_contact_group(name)
            if group is not None:
                self._group_slots.setdefault(group, []).append(index)
        self._initial_object_position = self._state()["object_position_scene"].clone()
        self._contact_seen = torch.zeros((self.num_envs, 6), device=self.device)
        self._contact_run = torch.zeros_like(self._contact_seen)
        self._contact_run_max = torch.zeros_like(self._contact_seen)
        self._contact_event_count = torch.zeros(self.num_envs, device=self.device)
        self._contact_driven_momentum = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_object_twist = self._state()["object_twist_world"].clone()
        self._last_stage16d_metrics: dict[str, torch.Tensor] = {}
        fixed = getattr(cfg, "stage16d_fixed_clip", None)
        if fixed is not None:
            self._clip_index.fill_(self.reference_bank.clip_ids.index(fixed))
            self._reset_idx(torch.arange(self.num_envs, device=self.device))

    def _contact_metrics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = self._object_contact_sensors["Object170105"].data.force_matrix_w
        second = self._object_contact_sensors["Object170650"].data.force_matrix_w
        if first is None or second is None:
            raise RuntimeError("STAGE16D_CONTACT_FORCE_MATRIX_UNAVAILABLE")
        forces = torch.where((self._clip_index == 0)[:, None, None], first[:, 0], second[:, 0])
        presence = torch.linalg.vector_norm(forces, dim=-1) > 1.0e-4
        group_order = ("thumb", "index", "middle", "ring", "pinky", "palm")
        grouped = torch.stack(
            tuple(
                presence[:, self._group_slots[group]].any(dim=-1)
                if self._group_slots.get(group)
                else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for group in group_order
            ),
            dim=-1,
        )
        self._contact_seen += grouped.float()
        self._contact_run = torch.where(grouped, self._contact_run + 1.0, 0.0)
        self._contact_run_max = torch.maximum(self._contact_run_max, self._contact_run)
        any_contact = grouped.any(dim=-1)
        self._contact_event_count += any_contact.float()
        aggregate = forces.sum(dim=1)
        return grouped, torch.linalg.vector_norm(aggregate, dim=-1), any_contact

    def _stage16d_metrics(self) -> dict[str, torch.Tensor]:
        state = self._state()
        index = self._target_reference_index
        grouped, force_magnitude, any_contact = self._contact_metrics()
        wrist_position_ref = self.reference_bank.gather(
            "wrist_pose_translation_world_ref", self._clip_index, index
        )
        wrist_quaternion_ref = self.reference_bank.gather(
            "wrist_pose_quaternion_world_ref_wxyz", self._clip_index, index
        )
        finger_ref = self.reference_bank.gather("q_finger_ref", self._clip_index, index)
        link_ref = self.reference_bank.gather(
            "tracked_link_positions_world_ref", self._clip_index, index
        )
        object_ref = self.reference_bank.gather(
            "object_pose_translation_world_ref", self._clip_index, index
        )
        wrist_error = torch.linalg.vector_norm(
            state["wrist_position_scene"] - wrist_position_ref, dim=-1
        ) + quaternion_geodesic(state["wrist_quaternion_wxyz"], wrist_quaternion_ref)
        finger_error = (
            ((state["finger_q"] - finger_ref) / (self.joint_upper - self.joint_lower))
            .abs()
            .mean(dim=-1)
        )
        link_error = torch.linalg.vector_norm(state["tracked_links_scene"] - link_ref, dim=-1).mean(
            dim=-1
        )
        displacement = state["object_position_scene"] - self._initial_object_position
        object_motion = torch.linalg.vector_norm(displacement, dim=-1)
        source_start = self.reference_bank.object_pose_translation_world_ref[
            self._clip_index, torch.zeros_like(index)
        ]
        source_end = self.reference_bank.object_pose_translation_world_ref[
            self._clip_index, torch.full_like(index, self.reference_bank.frame_count - 1)
        ]
        direction = source_end - source_start
        direction_norm = torch.linalg.vector_norm(direction, dim=-1).clamp_min(1.0e-6)
        projected = (displacement * direction).sum(dim=-1) / direction_norm
        semantic_progress = (projected / (0.30 * direction_norm).clamp_min(0.005)).clamp(0.0, 1.0)
        required_mask = torch.zeros_like(grouped)
        minimum_persistence = torch.ones(self.num_envs, device=self.device)
        minimum_recall = torch.zeros(self.num_envs, device=self.device)
        minimum_motion = torch.zeros(self.num_envs, device=self.device)
        for env_id in range(self.num_envs):
            clip = self.reference_bank.clip_ids[int(self._clip_index[env_id])]
            topology = self._topology_contracts[clip]
            for group in topology["required_body_groups"]:
                required_mask[
                    env_id, ("thumb", "index", "middle", "ring", "pinky", "palm").index(group)
                ] = True
            minimum_persistence[env_id] = float(topology["minimum_persistence_control_steps"])
            gate = self._task_gates[clip]
            minimum_recall[env_id] = gate.minimum_contact_recall
            minimum_motion[env_id] = gate.minimum_object_motion_m
        required_count = required_mask.sum(dim=-1).clamp_min(1)
        coverage = ((self._contact_seen > 0) & required_mask).sum(dim=-1) / required_count
        persistence = ((self._contact_run_max >= minimum_persistence[:, None]) & required_mask).sum(
            dim=-1
        ) / required_count
        object_twist = state["object_twist_world"]
        delta_twist = object_twist - self._previous_object_twist
        self._contact_driven_momentum |= any_contact & (
            torch.linalg.vector_norm(delta_twist, dim=-1) > 1.0e-7
        )
        self._previous_object_twist.copy_(object_twist)
        final_step = index >= self.reference_bank.frame_count - 1
        terminal_stable = (torch.linalg.vector_norm(object_twist[:, :3], dim=-1) <= 0.10) & (
            torch.linalg.vector_norm(object_twist[:, 3:], dim=-1) <= 1.0
        )
        action_first = torch.linalg.vector_norm(self._actions - self._previous_actions, dim=-1)
        action_second = torch.linalg.vector_norm(
            self._actions - 2.0 * self._previous_actions + self._second_previous_actions, dim=-1
        )
        workspace_distance = torch.linalg.vector_norm(
            state["object_position_scene"] - source_start, dim=-1
        )
        finite = torch.stack(
            [torch.isfinite(value).flatten(1).all(dim=-1) for value in state.values()], dim=-1
        ).all(dim=-1)
        metrics = {
            "wrist_fidelity": torch.exp(-wrist_error / 0.05),
            "finger_fidelity": torch.exp(-finger_error / 0.10),
            "link_fidelity": torch.exp(-link_error / 0.025),
            "contact_coverage": coverage,
            "contact_persistence": persistence,
            "contact_onset_alignment": any_contact.float(),
            "final_topology": (coverage * final_step.float()),
            "forbidden_contact": torch.zeros(self.num_envs, device=self.device),
            "penetration_m": torch.zeros(self.num_envs, device=self.device),
            "impulse_outlier": (force_magnitude > 100.0).float(),
            "object_instability": torch.linalg.vector_norm(object_twist, dim=-1),
            "action_effort": self._robot.data.applied_torque.abs().mean(dim=-1),
            "action_first_difference": action_first,
            "action_second_difference": action_second,
            "semantic_progress": semantic_progress,
            "relative_pose_progress": semantic_progress,
            "source_object_soft_prior": torch.exp(
                -torch.linalg.vector_norm(state["object_position_scene"] - object_ref, dim=-1)
                / 0.10
            ),
            "terminal_success": torch.zeros(self.num_envs, device=self.device),
            "catastrophic_failure": torch.zeros(self.num_envs, device=self.device),
            "finite": finite,
            "workspace_distance_m": workspace_distance,
            "wrist_safe": wrist_error < math.radians(90.0) + 0.20,
            "joint_limits_safe": (
                (state["finger_q"] >= self.joint_lower) & (state["finger_q"] <= self.joint_upper)
            ).all(dim=-1),
            "action_valid": torch.isfinite(self._actions).all(dim=-1)
            & (self._actions.abs() <= 1.0).all(dim=-1),
            "object_speed_mps": torch.linalg.vector_norm(object_twist[:, :3], dim=-1),
            "contact_recall": coverage,
            "contact_causality": self._contact_driven_momentum,
            "terminal_stable": terminal_stable,
            "object_motion_m": object_motion,
            "final_step": final_step,
            "minimum_contact_recall": minimum_recall,
            "minimum_object_motion_m": minimum_motion,
        }
        return metrics

    def _get_rewards(self) -> torch.Tensor:
        metrics = self._last_stage16d_metrics or self._stage16d_metrics()
        rewards = physics_consistent_reward_terms(metrics, self.stage16d_reward_profile)
        self._last_stage16d_metrics = metrics
        self._last_reward_terms = rewards
        self._second_previous_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._reference_index.copy_(self._target_reference_index)
        return rewards["total"]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._record_completed_contact_substep()
        metrics = self._stage16d_metrics()
        self._last_stage16d_metrics = metrics
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timed_out = torch.zeros_like(terminated)
        success = torch.zeros_like(terminated)
        reasons = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        for clip_index, clip in enumerate(self.reference_bank.clip_ids):
            env_mask = self._clip_index == clip_index
            if not bool(env_mask.any()):
                continue
            local = {
                name: value[env_mask]
                for name, value in metrics.items()
                if isinstance(value, torch.Tensor)
            }
            result = physics_consistent_termination(
                local, self._task_gates[clip], final_step=local["final_step"]
            )
            terminated[env_mask] = result["terminated"]
            timed_out[env_mask] = result["timed_out"]
            success[env_mask] = result["success"]
            reasons[env_mask] = result["primary_reason_code"]
        self._success.copy_(success)
        self._reason_codes.copy_(reasons)
        self.extras["stage16d"] = {
            "success": success.clone(),
            "primary_reason_code": reasons.clone(),
            "termination_reasons": STAGE16D_TERMINATION_REASONS,
            "semantic_progress": metrics["semantic_progress"].clone(),
            "contact_recall": metrics["contact_recall"].clone(),
            "contact_persistence": metrics["contact_persistence"].clone(),
            "contact_causality": metrics["contact_causality"].clone(),
            "terminal_stable": metrics["terminal_stable"].clone(),
            "object_motion_m": metrics["object_motion_m"].clone(),
            "source_object_deviation_m": (
                -torch.log(metrics["source_object_soft_prior"].clamp_min(1e-12)) * 0.10
            ),
            "penetration_metric": "UNAVAILABLE_REQUIRES_INDEPENDENT_GEOMETRY_AUDIT",
            "object_rollout_state_writes": 0,
            "wrist_rollout_state_writes": int(self._wrist_step_state_write_count.sum().item()),
        }
        return terminated, timed_out | success

    def _reset_idx(self, env_ids: Any) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "_contact_seen"):
            return
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._initial_object_position[ids] = self._state()["object_position_scene"][ids]
        self._contact_seen[ids] = 0.0
        self._contact_run[ids] = 0.0
        self._contact_run_max[ids] = 0.0
        self._contact_event_count[ids] = 0.0
        self._contact_driven_momentum[ids] = False
        self._previous_object_twist[ids] = self._state()["object_twist_world"][ids]
        self._last_stage16d_metrics = {}

    def contract_report(self) -> dict[str, Any]:
        base = super().contract_report()
        return {
            **base,
            "environment": "IsaacPhysicsConsistentRetargetingEnv",
            "protocol": self.cfg.physics_consistent_protocol,
            "strict_source_object_world_tracking_hard_gate": False,
            "source_object_world_trajectory_role": "low_weight_soft_prior_and_diagnostic",
            "object_trajectory": "free_physx_rollout_output",
            "formal_object_rollout_state_writes": 0,
            "formal_wrist_rollout_state_writes": int(
                self._wrist_step_state_write_count.sum().item()
            ),
            "hidden_force": False,
            "hidden_attachment": False,
            "reward": self.stage16d_reward_profile.as_dict(),
        }


__all__ = ["IsaacPhysicsConsistentRetargetingEnv"]
