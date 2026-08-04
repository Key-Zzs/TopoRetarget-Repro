#!/usr/bin/env python3
"""Qualify C.5A tensor-clone and deterministic-history-replay state replication."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--candidate-envs", type=int, default=32)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write(path: Path, report: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_O1_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _set_partition_clip(
    env: Any, *, clip_index: int, candidate_ids: Any, candidate_clip: int
) -> None:
    import torch

    all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    original_balanced = env.cfg.balanced_clip_assignment
    original_alternate = env.cfg.alternate_clip_on_reset
    try:
        env.cfg.balanced_clip_assignment = False
        env.cfg.alternate_clip_on_reset = False
        env._clip_index[all_ids] = candidate_clip
        env._clip_index[0] = clip_index
        env._reset_idx(all_ids)
    finally:
        env.cfg.balanced_clip_assignment = original_balanced
        env.cfg.alternate_clip_on_reset = original_alternate


def _phase_errors(env: Any, candidate_ids: Any) -> tuple[dict[str, float], bool, bool]:
    import torch

    from toporetarget.rl.isaaclab_oracle.metrics import state_differences
    from toporetarget.rl.isaaclab_oracle.runtime import state_view

    view = state_view(
        env, torch.cat((torch.zeros(1, dtype=torch.long, device=env.device), candidate_ids))
    )
    source = {
        name: value[:1].expand_as(value[1:])
        for name, value in view.items()
        if name not in {"reference_index", "reason_codes"}
    }
    candidates = {
        name: value[1:]
        for name, value in view.items()
        if name not in {"reference_index", "reason_codes"}
    }
    errors = state_differences(source, candidates)
    reward = env._last_reward_terms["total"]
    errors["reward"] = float((reward[candidate_ids] - reward[0]).abs().amax().detach().cpu())
    reason_exact = bool(torch.eq(view["reason_codes"][1:], view["reason_codes"][0]).all())
    finite = bool(torch.isfinite(env._get_observations()["policy"]).all())
    return errors, reason_exact, finite


def _passes(
    errors: dict[str, float], frozen: dict[str, Any], reason_exact: bool, finite: bool
) -> bool:
    metrics = frozen["metrics"]
    return bool(
        reason_exact
        and finite
        and all(errors[name] <= float(metrics[name]["frozen_tolerance"]) for name in errors)
    )


def _prepare_tensor_clone(env: Any, pool: Any, *, clip_index: int, frame: int) -> Any:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step

    # Candidate environments deliberately follow the other clip before clone.
    # This makes a contact cache mismatch observable instead of prewarming every
    # candidate with the execution trajectory.
    _set_partition_clip(
        env,
        clip_index=clip_index,
        candidate_ids=pool.candidate_ids,
        candidate_clip=1 - clip_index,
    )
    zero = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    for _ in range(frame):
        raw_control_step(env, zero)
    snapshot = pool.capture_execution_state()
    pool.replicate_execution_state(snapshot)
    return snapshot


def _same_action_case(
    env: Any,
    pool: Any,
    *,
    clip_index: int,
    frame: int,
    action_name: str,
) -> dict[str, object]:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step

    _prepare_tensor_clone(env, pool, clip_index=clip_index, frame=frame)
    if action_name == "zero":
        action = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    else:
        vector = torch.linspace(-0.5, 0.5, 26, device=env.device)
        action = vector.expand(env.num_envs, -1).clone()
    raw_control_step(env, action)
    env._get_dones()
    errors, reason_exact, finite = _phase_errors(env, pool.candidate_ids)
    return {
        "action": action_name,
        "errors": errors,
        "termination_exact": reason_exact,
        "finite": finite,
    }


def _branch_case(env: Any, pool: Any, *, clip_index: int, frame: int) -> dict[str, object]:
    import torch

    from toporetarget.rl.isaaclab_oracle.candidate_state import (
        capture_candidate_state,
        hash_candidate_state,
    )
    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import state_view

    source_snapshot = _prepare_tensor_clone(env, pool, clip_index=clip_index, frame=frame)
    actions = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    actions[pool.candidate_ids, 6] = torch.linspace(
        -0.9, 0.9, pool.candidate_ids.numel(), device=env.device
    )
    source_after_setup = capture_candidate_state(env, [0])
    raw_control_step(env, actions)
    env._get_dones()
    source_after = capture_candidate_state(env, [0])
    view = state_view(env, pool.candidate_ids)
    joint_span = float(
        (view["robot_joint_pos"][:, 0] - view["robot_joint_pos"][0, 0]).abs().amax().detach().cpu()
    )
    target_span = float(
        (
            env._joint_target_isaac[pool.candidate_ids, 0]
            - env._joint_target_isaac[pool.candidate_ids[0], 0]
        )
        .abs()
        .amax()
        .detach()
        .cpu()
    )
    source_unchanged_by_setup = all(
        torch.equal(source_snapshot.tensors[name], source_after_setup.tensors[name])
        for name in source_snapshot.tensors
    )
    return {
        "candidate_action_span": float(torch.ptp(actions[pool.candidate_ids, 6]).detach().cpu()),
        "joint_position_span_rad": joint_span,
        "joint_target_span_rad": target_span,
        "branch_divergence": joint_span > 1.0e-10 or target_span > 1.0e-10,
        "execution_snapshot_captured_before_branch": source_unchanged_by_setup,
        "execution_state_after_branch_hash": hash_candidate_state(source_after),
    }


def _history_replay_case(env: Any, pool: Any, *, clip_index: int, frame: int) -> dict[str, object]:
    import torch

    from toporetarget.rl.isaaclab_oracle.action_history import CandidateActionHistoryV1
    from toporetarget.rl.isaaclab_oracle.history_replay import (
        deterministic_history_replay,
        raw_control_step,
    )

    _set_partition_clip(
        env,
        clip_index=clip_index,
        candidate_ids=pool.candidate_ids,
        candidate_clip=clip_index,
    )
    history = CandidateActionHistoryV1()
    one_action = torch.zeros((1, 26), dtype=torch.float32, device=env.device)
    for _ in range(frame):
        history.append(one_action)
    deterministic_history_replay(
        env,
        pool.candidate_ids,
        history,
        clip_index=clip_index,
        write_audit=pool.write_audit,
    )
    action = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    raw_control_step(env, action)
    env._get_dones()
    errors, reason_exact, finite = _phase_errors(env, pool.candidate_ids)
    return {
        "method": "deterministic_history_replay_v1",
        "history": history.as_dict(),
        "errors": errors,
        "termination_exact": reason_exact,
        "finite": finite,
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.candidate_envs < 32 or args.output.exists():
        raise SystemExit("C.5A O1 requires at least 32 candidates and a fresh output path")
    frames = _load(args.frames)
    tolerance_report = _load(args.tolerances)
    frozen = tolerance_report.get("global_tolerances")
    if not isinstance(frozen, dict) or frozen.get("status") != "REPLICATION_TOLERANCES_FROZEN":
        raise RuntimeError("C5A_O1_TOLERANCES_NOT_FROZEN")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from toporetarget.rl.isaaclab_oracle.candidate_pool import PhysXOracleCandidatePoolV1
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env

        env = make_stage16c5_env(num_envs=1 + args.candidate_envs)
        pool = PhysXOracleCandidatePoolV1(env, candidate_count=args.candidate_envs)
        tensor_rows: list[dict[str, object]] = []
        history_rows: list[dict[str, object]] = []
        for clip_index, clip_row in enumerate(frames["clips"]):
            assert isinstance(clip_row, dict)
            phase_frames = clip_row["frames"]
            assert isinstance(phase_frames, dict)
            for phase, frame_value in phase_frames.items():
                frame = int(frame_value)
                zero = _same_action_case(
                    env, pool, clip_index=clip_index, frame=frame, action_name="zero"
                )
                nonzero = _same_action_case(
                    env, pool, clip_index=clip_index, frame=frame, action_name="nonzero"
                )
                branch = _branch_case(env, pool, clip_index=clip_index, frame=frame)
                passed = bool(
                    _passes(
                        zero["errors"],
                        frozen,
                        bool(zero["termination_exact"]),
                        bool(zero["finite"]),
                    )
                    and _passes(
                        nonzero["errors"],
                        frozen,
                        bool(nonzero["termination_exact"]),
                        bool(nonzero["finite"]),
                    )
                    and branch["branch_divergence"]
                )
                tensor_rows.append(
                    {
                        "clip": clip_row["clip"],
                        "phase": phase,
                        "frame": frame,
                        "method": "tensor_clone_replication_v1",
                        "candidates": args.candidate_envs,
                        "zero_action": zero,
                        "same_nonzero_action": nonzero,
                        "branch_group": branch,
                        "result": "PASS" if passed else "FAIL",
                    }
                )
                # History replay is a strictly conditional fallback.  A passing
                # tensor clone must not consume it, because that would hide a
                # later cache-dependent tensor-clone regression.
                if passed:
                    history = {
                        "method": "deterministic_history_replay_v1",
                        "status": "NOT_RUN_TENSOR_CLONE_PASS",
                    }
                    history_pass = True
                else:
                    history = _history_replay_case(env, pool, clip_index=clip_index, frame=frame)
                    history_pass = _passes(
                        history["errors"],
                        frozen,
                        bool(history["termination_exact"]),
                        bool(history["finite"]),
                    )
                history_rows.append(
                    {
                        "clip": clip_row["clip"],
                        "phase": phase,
                        "frame": frame,
                        **history,
                        "result": "PASS" if history_pass else "FAIL",
                    }
                )
        selected = []
        for tensor_row, history_row in zip(tensor_rows, history_rows, strict=True):
            selected.append(
                {
                    "clip": tensor_row["clip"],
                    "phase": tensor_row["phase"],
                    "method": (
                        "tensor_clone_replication_v1"
                        if tensor_row["result"] == "PASS"
                        else "deterministic_history_replay_v1"
                    ),
                    "result": (
                        tensor_row["result"]
                        if tensor_row["result"] == "PASS"
                        else history_row["result"]
                    ),
                }
            )
        contract = env.contract_report()
        all_selected_pass = all(row["result"] == "PASS" for row in selected)
        report = {
            "status": (
                "STAGE16C5A_STATE_REPLICATION_VALIDATED"
                if all_selected_pass
                else "STAGE16C5A_STATE_REPLICATION_PARTIAL"
            ),
            "tensor_clone_status": (
                "STAGE16C5A_TENSOR_CLONE_VALIDATED"
                if all(row["result"] == "PASS" for row in tensor_rows)
                else "STAGE16C5A_TENSOR_CLONE_PRECONTACT_ONLY"
            ),
            "history_replay_status": (
                "STAGE16C5A_HISTORY_REPLAY_VALIDATED"
                if all(row["result"] == "PASS" for row in history_rows)
                else "STAGE16C5A_HISTORY_REPLAY_PARTIAL"
            ),
            "tensor_clone": tensor_rows,
            "history_replay": history_rows,
            "selected_methods": selected,
            "write_audit": pool.write_audit.as_dict(),
            "execution_rollout_writes": {
                "wrist": contract["wrist_root_state_writes_during_step"],
                "object": contract["object_rollout_state_writes"],
                "candidate_audit_formal_execution": (
                    pool.write_audit.formal_execution_rollout_writes
                ),
            },
            "cuda_device": str(env.device),
            "clean_exit": True,
        }
        _write(args.output, report)
        print(json.dumps({"status": report["status"], "tensor": report["tensor_clone_status"]}))
        return 0 if all_selected_pass else 2
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
