"""Runtime-only helpers shared by the Stage 16-C.5A qualification scripts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .action_history import CandidateActionHistoryV1
from .candidate_state import capture_candidate_state
from .history_replay import raw_control_step


def make_stage16c5_env(*, num_envs: int, seed: int = 20260804) -> Any:
    """Instantiate the frozen retimed C.3/C.4 DirectRLEnv after AppLauncher."""

    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
        IsaacWorldWristFingerDirectRLEnv,
    )
    from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
        IsaacWorldWristFingerDirectRLEnvCfg,
        configure_explicit_virtual_wrist,
        configure_uniform_reference_retiming,
    )

    cfg = IsaacWorldWristFingerDirectRLEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.lazy_sensor_update = True
    cfg.contact_telemetry = "aggregate"
    cfg.collect_wrist_diagnostics = False
    cfg.balanced_clip_assignment = True
    cfg.alternate_clip_on_reset = False
    cfg.reset_reference_index = "frame0"
    configure_uniform_reference_retiming(cfg, time_scale=8)
    configure_explicit_virtual_wrist(cfg, profile_identifier="high_authority_bounded")
    env = IsaacWorldWristFingerDirectRLEnv(cfg)
    env.reset(seed=seed)
    return env


def state_view(env: Any, env_ids: Sequence[int] | torch.Tensor) -> dict[str, torch.Tensor]:
    """Canonical local-frame comparison view for a set of environments."""

    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    state = capture_candidate_state(env, ids)
    origins = state.tensors["source_env_origins"]
    robot = state.tensors["robot_root_state"].clone()
    object_105 = state.tensors["object_170105_root_state"].clone()
    object_650 = state.tensors["object_170650_root_state"].clone()
    robot[:, :3] -= origins
    object_105[:, :3] -= origins
    object_650[:, :3] -= origins
    active = torch.where(
        state.tensors["clip_index"][:, None] == 0,
        object_105,
        object_650,
    )
    return {
        "robot_joint_pos": state.tensors["robot_joint_pos"],
        "robot_joint_vel": state.tensors["robot_joint_vel"],
        "robot_root_state": robot,
        "active_object_root_state": active,
        "reference_index": state.tensors["reference_index"],
        "reason_codes": state.tensors["reason_codes"],
    }


def action_history_to_frame(
    env: Any,
    *,
    frame: int,
    execution_env_ids: Sequence[int] | torch.Tensor,
    action: torch.Tensor | None = None,
) -> CandidateActionHistoryV1:
    """Advance normal control steps to ``frame`` and retain exact past actions."""

    if frame < 0 or frame >= env.reference_bank.frame_count:
        raise ValueError("replication test frame outside retimed reference")
    ids = torch.as_tensor(execution_env_ids, dtype=torch.long, device=env.device)
    history = CandidateActionHistoryV1()
    if frame == 0:
        return history
    base_action = (
        torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
        if action is None
        else action
    )
    if base_action.shape != (env.num_envs, 26):
        raise ValueError("execution action must have full DirectRLEnv batch shape")
    for _ in range(frame):
        raw_control_step(env, base_action)
        history.append(base_action.index_select(0, ids))
    history.validate(expected_boundary_index=frame)
    return history


def choose_replication_frames(contact_report: Mapping[str, object]) -> dict[str, object]:
    """Select phase frames solely from C.3 contact traces with generic fallbacks."""

    rows = contact_report.get("clips")
    if not isinstance(rows, list):
        raise ValueError("contact report has no clips list")
    selected: list[dict[str, object]] = []
    for clip_row in rows:
        if not isinstance(clip_row, Mapping) or not isinstance(clip_row.get("clip"), str):
            raise ValueError("contact report clip row is malformed")
        first = clip_row.get("first_contact")
        if not isinstance(first, Mapping):
            raise ValueError(f"contact report has no first contact for {clip_row['clip']}")
        onset = int(first["reference_index"])
        # Aggregated summary has no complete trace; strongest/last are resolved
        # by the qualifier from its runtime sensor trace.  These base values are
        # retained as deterministic generic fallbacks.
        selected.append(
            {
                "clip": clip_row["clip"],
                "F0": 0,
                "Fpre": max(0, onset - 1),
                "Fon": onset,
                "Fcontact": onset,
                "Fpost": min(320, onset + 1),
                "fallbacks": {
                    "Fcontact": "runtime trace peak-force selection required",
                    "Fpost": "runtime trace last-contact-plus-one selection required",
                },
            }
        )
    return {
        "version": "stage16c5_replication_test_frames_v1",
        "selection_source": "C3 object-centric contact trace; no clip-specific code",
        "clips": selected,
    }


def force_norm(record: Mapping[str, object]) -> float:
    vector = record.get("net_contact_force_world_on_object_n", [])
    if not isinstance(vector, list):
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in vector))


__all__ = [
    "action_history_to_frame",
    "choose_replication_frames",
    "force_norm",
    "make_stage16c5_env",
    "raw_control_step",
    "state_view",
]
