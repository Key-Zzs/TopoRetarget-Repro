"""The sole C.5A fallback: deterministic replay from a frame-zero reset.

These helpers deliberately mirror the non-reset portion of
``DirectRLEnv.step``.  C.5A must inspect the state at a control boundary
without letting DirectRLEnv auto-reset a done candidate, but it may not skip
the framework's reset synchronization, counters, or done evaluation.
"""

from __future__ import annotations

from typing import Any

import torch

from .action_history import CandidateActionHistoryV1
from .contracts import Stage16C5WriteAuditV1


def synchronize_reset_boundary(env: Any) -> None:
    """Materialize pending reset writes before any C.5A measurement step.

    Calling ``_reset_idx`` alone only writes buffers into Isaac Lab assets.
    The public ``DirectRLEnv.reset`` immediately follows it with this sequence
    so the next action reads current PhysX tensors rather than a prior trial's
    cached state.  The former C.5A harness omitted it.
    """

    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(env.physics_dt)
    if torch.cuda.is_available() and str(env.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(env.device))


def raw_control_step(env: Any, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one exact control interval while explicitly suppressing auto-reset.

    This is the public DirectRLEnv step order through rewards, except that a
    caller retains a terminal state for comparison instead of resetting it.
    It returns ``(terminated, timed_out)`` at the same control boundary.
    """

    env._pre_physics_step(actions.to(env.device))
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    terminated, timed_out = env._get_dones()
    env.reset_terminated.copy_(terminated)
    env.reset_time_outs.copy_(timed_out)
    env.reset_buf.copy_(terminated | timed_out)
    rewards = env._get_rewards()
    # Isaac Lab versions differ on whether ``reward_buf`` is allocated before
    # the first public ``step``.  It is an output cache, not causal simulator
    # state, so allocate it only when the framework has not materialized it.
    if isinstance(getattr(env, "reward_buf", None), torch.Tensor):
        env.reward_buf.copy_(rewards)
    else:
        env.reward_buf = rewards.clone()
    return terminated.clone(), timed_out.clone()


def reset_candidates_to_frame_zero(
    env: Any,
    candidate_ids: torch.Tensor,
    *,
    clip_index: int,
    write_audit: Stage16C5WriteAuditV1 | None = None,
) -> None:
    """Reset only candidates while preserving the frozen clip and frame-zero semantics."""

    original_balanced = env.cfg.balanced_clip_assignment
    original_alternate = env.cfg.alternate_clip_on_reset
    original_reference_reset = env.cfg.reset_reference_index
    try:
        env.cfg.balanced_clip_assignment = False
        env.cfg.alternate_clip_on_reset = False
        env.cfg.reset_reference_index = "frame0"
        env._clip_index[candidate_ids] = int(clip_index)
        env._reset_idx(candidate_ids)
    finally:
        env.cfg.balanced_clip_assignment = original_balanced
        env.cfg.alternate_clip_on_reset = original_alternate
        env.cfg.reset_reference_index = original_reference_reset
    synchronize_reset_boundary(env)
    if write_audit is not None:
        write_audit.record(
            category="reset",
            operation="history_replay_frame0_reset",
            env_ids=candidate_ids,
            tensor_names=["clip_index", "reference_index", "robot", "objects"],
        )


def deterministic_history_replay(
    env: Any,
    candidate_ids: torch.Tensor,
    history: CandidateActionHistoryV1,
    *,
    clip_index: int,
    write_audit: Stage16C5WriteAuditV1 | None = None,
) -> None:
    """Rebuild candidate contact state using only frame-zero reset and past actions.

    ``raw_control_step`` operates a vector environment, so callers must provide
    a separate guard/execution population with the same historical actions.  No
    mid-trajectory object state write occurs in this function.
    """

    history.validate()
    reset_candidates_to_frame_zero(
        env, candidate_ids, clip_index=clip_index, write_audit=write_audit
    )
    for action in history.actions:
        if action.shape[0] == env.num_envs:
            replay_actions = action
        elif action.shape[0] == 1:
            replay_actions = action.expand(env.num_envs, -1).clone()
        else:
            raise ValueError("history action batch must be one or all DirectRLEnv environments")
        raw_control_step(env, replay_actions)
    actual = env._reference_index.index_select(0, candidate_ids)
    if not bool(torch.eq(actual, history.length).all()):
        raise RuntimeError(
            "C5A_HISTORY_REPLAY_REFERENCE_INDEX_FAILURE: "
            f"expected={history.length}, actual={actual.detach().cpu().tolist()}"
        )


__all__ = [
    "deterministic_history_replay",
    "raw_control_step",
    "reset_candidates_to_frame_zero",
    "synchronize_reset_boundary",
]
