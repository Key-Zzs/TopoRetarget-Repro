#!/usr/bin/env python3
"""Run staged persistent-GPU robust CEM and restore-free C5C qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_CLIPS = ("hocap_170105", "hocap_170650")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--mode", choices=("b1", "b2", "b3", "formal"), required=True)
    parser.add_argument("--clip", choices=_CLIPS, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-report", type=Path)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=4)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5B_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace(path: Path, clip: str) -> tuple[Path, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("traces") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("C5B trace manifest has no traces")
    for row in rows:
        if isinstance(row, Mapping) and row.get("clip") == clip:
            source = row.get("action_trace")
            expected = row.get("action_trace_sha256")
            if not isinstance(source, str) or not isinstance(expected, str):
                break
            source_path = Path(source)
            if not source_path.is_file() or _sha256(source_path) != expected:
                raise ValueError("C5B source action hash mismatch")
            with np.load(source_path, allow_pickle=False) as handle:
                actions = np.asarray(handle["actions"], dtype=np.float32)
            if actions.shape != (40, 26) or not np.isfinite(actions).all():
                raise ValueError("C5B source action trace must be [40,26]")
            return source_path, np.repeat(actions, 8, axis=0)
    raise ValueError(f"C5B trace manifest lacks {clip}")


def _phase_frames(path: Path, clip: str) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("clips") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("C5B frame manifest has no clips")
    for row in rows:
        if isinstance(row, Mapping) and row.get("clip") == clip:
            frames = row.get("frames")
            if isinstance(frames, Mapping):
                return {str(name): int(value) for name, value in frames.items()}
    raise ValueError(f"C5B frame manifest lacks {clip}")


def _gpu_memory_mib() -> float | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 2 and values[0] == str(os.getpid()):
            try:
                return float(values[1])
            except ValueError:
                return None
    return None


def _reason_names(codes: Any) -> list[str]:
    from toporetarget.rl.environments.isaaclab_backend.termination_terms import (
        TERMINATION_REASONS,
    )

    result = []
    for code in codes.detach().cpu().tolist():
        value = int(code)
        result.append(
            TERMINATION_REASONS[value] if 0 <= value < len(TERMINATION_REASONS) else f"CODE_{value}"
        )
    return result


def _contact_force_by_env(env: Any) -> Any:
    import torch

    required = env.num_envs * env.cfg.decimation
    rows = env.contact_substep_records[-required:]
    forces = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    counts = torch.zeros_like(forces)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        env_id = int(row.get("env_id", -1))
        vector = row.get("net_contact_force_world_on_object_n")
        if env_id < 0 or env_id >= env.num_envs or not isinstance(vector, list):
            continue
        forces[env_id] += math.sqrt(sum(float(value) ** 2 for value in vector))
        counts[env_id] += 1.0
    return forces / counts.clamp_min(1.0)


def _tracking_error(env: Any) -> Any:
    import torch

    from toporetarget.rl.environments.isaaclab_backend.tensor_math import quaternion_geodesic

    state = env._state()
    index = env._target_reference_index
    link_ref = env.reference_bank.gather("tracked_link_positions_world_ref", env._clip_index, index)
    finger_ref = env.reference_bank.gather("q_finger_ref", env._clip_index, index)
    wrist_position_ref = env.reference_bank.gather(
        "wrist_pose_translation_world_ref", env._clip_index, index
    )
    wrist_quaternion_ref = env.reference_bank.gather(
        "wrist_pose_quaternion_world_ref_wxyz", env._clip_index, index
    )
    link = (
        torch.linalg.vector_norm(state["tracked_links_scene"] - link_ref, dim=-1).mean(dim=-1)
        / 0.025
    )
    finger = ((state["finger_q"] - finger_ref) / (env.joint_upper - env.joint_lower)).abs().mean(
        dim=-1
    ) / 0.10
    wrist_position = (
        torch.linalg.vector_norm(state["wrist_position_scene"] - wrist_position_ref, dim=-1) / 0.02
    )
    wrist_rotation = quaternion_geodesic(
        state["wrist_quaternion_wxyz"], wrist_quaternion_ref
    ) / math.radians(10.0)
    return (link + finger + wrist_position + wrist_rotation) / 4.0


def _slot_env_ids(pool: Any, permutation: Any, horizon: int) -> Any:
    import torch

    from toporetarget.rl.isaaclab_oracle.replica_manager import LogicalCandidateSlotV1

    return torch.tensor(
        [
            permutation.logical_to_env[LogicalCandidateSlotV1(candidate, horizon, replica)]
            for candidate in range(pool.manager.population)
            for replica in range(pool.manager.replicas)
        ],
        dtype=torch.long,
        device=pool.env.device,
    )


def _evaluate_horizon(
    env: Any,
    pool: Any,
    permutation: Any,
    *,
    horizon: int,
    samples: Any,
    force_history: list[Any],
    effort_sum: Any,
    terminal_required: bool,
) -> list[Any]:
    import torch

    from toporetarget.rl.isaaclab_oracle.robust_oracle import (
        RobustCandidateEvaluatorV2,
        RobustCandidateReplicaV2,
    )

    ids = _slot_env_ids(pool, permutation, horizon)
    stage = env.extras["stage16"]
    position = stage["object_position_error_m"].index_select(0, ids)
    rotation = torch.rad2deg(stage["object_orientation_error_rad"].index_select(0, ids))
    axis = stage["object_axis_error_m"].index_select(0, ids)
    tracking = _tracking_error(env).index_select(0, ids)
    forces = torch.stack(force_history, dim=0).index_select(1, ids)
    contact_stability = forces.std(dim=0, correction=0)
    effort = effort_sum.index_select(0, ids) / horizon
    reasons = _reason_names(env._reason_codes.index_select(0, ids))
    successes = env._success.index_select(0, ids).detach().cpu().tolist()
    reach = (
        (env._reference_index.index_select(0, ids) >= env.reference_bank.frame_count - 1)
        .detach()
        .cpu()
        .tolist()
    )
    position_values = position.detach().cpu().tolist()
    rotation_values = rotation.detach().cpu().tolist()
    axis_values = axis.detach().cpu().tolist()
    tracking_values = tracking.detach().cpu().tolist()
    stability_values = contact_stability.detach().cpu().tolist()
    effort_values = effort.detach().cpu().tolist()
    action_delta = torch.diff(
        torch.cat(
            (
                env._previous_actions.index_select(0, ids).reshape(
                    pool.manager.population, pool.manager.replicas, 1, 26
                )[:, 0],
                samples[:, None].expand(-1, pool.manager.replicas, -1, -1)[:, 0],
            ),
            dim=1,
        ),
        dim=1,
    )
    smoothness = torch.linalg.vector_norm(action_delta, dim=-1).mean(dim=-1)
    smoothness_values = smoothness.detach().cpu().tolist()
    evaluator = RobustCandidateEvaluatorV2(replica_count=pool.manager.replicas)
    evaluations = []
    for candidate in range(pool.manager.population):
        rows = []
        for replica in range(pool.manager.replicas):
            flat = candidate * pool.manager.replicas + replica
            rows.append(
                RobustCandidateReplicaV2(
                    object_position_error_m=float(position_values[flat]),
                    object_rotation_error_deg=float(rotation_values[flat]),
                    object_axis_error_m=float(axis_values[flat]),
                    tracking_error=float(tracking_values[flat]),
                    contact_stability=float(stability_values[flat]),
                    smoothness=float(smoothness_values[candidate]),
                    effort=float(effort_values[flat]),
                    termination_reason=reasons[flat],
                    success=bool(successes[flat]),
                    final_reach=bool(reach[flat]),
                    terminal_required=terminal_required,
                )
            )
        evaluations.append(
            evaluator.evaluate(candidate_id=candidate, horizon=horizon, replicas=rows)
        )
    return evaluations


def _plan_one_step(
    env: Any,
    pool: Any,
    *,
    proposal_actions: np.ndarray,
    control_step: int,
    remaining: int,
) -> tuple[Any, dict[str, object]]:
    import torch

    from toporetarget.rl.isaaclab_oracle.candidate_state import restore_candidate_state
    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.horizon_selector import AdaptiveHorizonSelectorV1
    from toporetarget.rl.isaaclab_oracle.robust_cem import (
        RobustCEMConfigV1,
        RobustMultiHorizonCEMV1,
    )
    from toporetarget.rl.isaaclab_oracle.robust_oracle import RobustLexicographicSelectorV1

    active_horizons = AdaptiveHorizonSelectorV1().select(remaining)
    if not active_horizons:
        raise RuntimeError("C5B planner called at terminal remaining=0 boundary")
    config = RobustCEMConfigV1(
        population=pool.manager.population,
        replicas=pool.manager.replicas,
        seed=20260806 + control_step * 97,
    )
    initial_means = {
        horizon: torch.as_tensor(
            proposal_actions[control_step : control_step + horizon],
            dtype=torch.float32,
            device=env.device,
        )
        for horizon in (1, 5, 10)
    }
    # In contracted tails, inactive means use valid slices only by constructing
    # a temporary repeated final action; they are never asked, stepped, or scored.
    for horizon in (1, 5, 10):
        if initial_means[horizon].shape[0] != horizon:
            final = (
                initial_means[horizon][-1:]
                if initial_means[horizon].numel()
                else torch.zeros((1, 26), device=env.device)
            )
            initial_means[horizon] = torch.cat(
                (
                    initial_means[horizon],
                    final.expand(horizon - initial_means[horizon].shape[0], -1),
                ),
                dim=0,
            )
    cem = RobustMultiHorizonCEMV1(config, device=env.device, initial_means=initial_means)
    execution_snapshot = pool.capture_execution_state()
    last_samples: dict[int, Any] = {}
    last_evaluations: list[Any] = []
    mapping_rows: list[dict[str, object]] = []
    rollout_latency = 0.0
    aggregation_latency = 0.0
    for iteration in range(config.iterations):
        samples = cem.ask(active_horizons)
        last_samples = samples
        pool.dispatch_execution_state(execution_snapshot)
        permutation = pool.manager.permutation(control_step * config.iterations + iteration)
        mapping_rows.append(
            {
                "iteration": iteration,
                "seed": permutation.seed,
                "mapping_sha256": hashlib.sha256(
                    json.dumps(permutation.as_dict(), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            }
        )
        force_history: list[Any] = []
        effort_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        evaluations_by_horizon: dict[int, Sequence[Any]] = {}
        iteration_started = time.perf_counter()
        for offset in range(max(active_horizons)):
            actions = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
            for horizon in active_horizons:
                if offset >= horizon:
                    continue
                ids = _slot_env_ids(pool, permutation, horizon)
                values = samples[horizon][:, offset].repeat_interleave(pool.manager.replicas, dim=0)
                actions.index_copy_(0, ids, values)
            raw_control_step(env, actions)
            force_history.append(_contact_force_by_env(env))
            effort_sum += env._robot.data.applied_torque.abs().mean(dim=-1)
            boundary = offset + 1
            if boundary in active_horizons:
                evaluations_by_horizon[boundary] = _evaluate_horizon(
                    env,
                    pool,
                    permutation,
                    horizon=boundary,
                    samples=samples[boundary],
                    force_history=force_history,
                    effort_sum=effort_sum,
                    terminal_required=remaining <= boundary,
                )
        torch.cuda.synchronize(torch.device(env.device))
        rollout_latency += time.perf_counter() - iteration_started
        aggregate_started = time.perf_counter()
        cem.tell(iteration, evaluations_by_horizon)
        aggregation_latency += time.perf_counter() - aggregate_started
        last_evaluations = [
            row for horizon in active_horizons for row in evaluations_by_horizon[horizon]
        ]
    selected = RobustLexicographicSelectorV1().select(last_evaluations)
    selected_action = last_samples[selected.horizon][selected.candidate_id, 0].clone()
    # Candidate simulations advance every scene.  Rewind only the execution
    # scene as an audited planning-setup write, then apply the selected action
    # through the normal controller/PhysX path.  Formal C5C uses no restoration.
    restore_candidate_state(
        env,
        execution_snapshot,
        pool.execution_ids,
        write_audit=pool.write_audit,
    )
    formal_writes_before = pool.write_audit.formal_execution_rollout_writes
    execution_actions = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
    execution_actions[pool.execution_env_id] = selected_action
    raw_control_step(env, execution_actions)
    if pool.write_audit.formal_execution_rollout_writes != formal_writes_before:
        raise RuntimeError("C5B_SELECTED_ACTION_HIDDEN_STATE_WRITE")
    selected_replica_margins = [row.normalized_gate_margin for row in selected.replicas]
    report = {
        "control_step": control_step,
        "remaining_before_step": remaining,
        "active_horizons": list(active_horizons),
        "selected_horizon": selected.horizon,
        "selected_candidate_id": selected.candidate_id,
        "selected_action": selected_action.detach().cpu().tolist(),
        "selected_evaluation": selected.as_dict(),
        "replica_gate_margin_std": float(np.std(selected_replica_margins)),
        "cem_convergence": cem.convergence_report(),
        "mapping_records": mapping_rows,
        "candidate_rollout_latency_s": rollout_latency,
        "aggregation_latency_s": aggregation_latency,
        "planning_rewind": {
            "used": True,
            "classification": "candidate_setup_not_formal_execution",
            "selected_action_applied_via_normal_controller_and_physx": True,
        },
    }
    return selected_action, report


def _advance_execution_with_source(env: Any, pool: Any, actions: np.ndarray, target: int) -> None:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step

    for step in range(target):
        batch = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
        batch[pool.execution_env_id] = torch.as_tensor(actions[step], device=env.device)
        raw_control_step(env, batch)


def _run_planning(args: argparse.Namespace) -> int:
    if not args.accept_eula:
        raise SystemExit("C5B requires --accept-eula")
    source_path, proposal = _trace(args.trace_manifest, args.clip)
    frames = _phase_frames(args.frames, args.clip)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    started = time.perf_counter()
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.replica_manager import (
            PersistentRobustCandidatePoolV1,
        )
        from toporetarget.rl.isaaclab_oracle.runtime import (
            make_stage16c5_env,
            reset_frozen_clip_frame_zero,
        )

        candidate_count = args.population * 3 * args.replicas
        env = make_stage16c5_env(num_envs=candidate_count + 1, contact_telemetry="aggregate")
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        pool = PersistentRobustCandidatePoolV1(
            env, population=args.population, replicas=args.replicas
        )
        all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        records: list[dict[str, object]] = []
        selected_actions: list[list[float]] = []
        if args.mode == "b1":
            phase_targets = (
                ("pre-contact", max(0, frames["contact_onset"] - 1)),
                ("contact", frames["contact_onset"]),
                ("post-contact", frames["post_contact"]),
            )
            for phase, target in phase_targets:
                reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=all_ids)
                _advance_execution_with_source(env, pool, proposal, target)
                action, record = _plan_one_step(
                    env,
                    pool,
                    proposal_actions=proposal,
                    control_step=target,
                    remaining=320 - target,
                )
                record["phase"] = phase
                records.append(record)
                selected_actions.append(action.detach().cpu().tolist())
            start_step = None
        else:
            start_step = max(0, frames["contact_onset"] - 5) if args.mode == "b2" else 0
            step_count = 30 if args.mode == "b2" else 320
            reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=all_ids)
            _advance_execution_with_source(env, pool, proposal, start_step)
            for control_step in range(start_step, min(320, start_step + step_count)):
                action, record = _plan_one_step(
                    env,
                    pool,
                    proposal_actions=proposal,
                    control_step=control_step,
                    remaining=320 - control_step,
                )
                records.append(record)
                selected_actions.append(action.detach().cpu().tolist())
                if len(records) % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "mode": args.mode,
                                "clip": args.clip,
                                "completed_planning_steps": len(records),
                                "latest_selected_horizon": record["selected_horizon"],
                            }
                        ),
                        flush=True,
                    )
        env_contract = env.contract_report()
        payload = {
            "schema_version": f"stage16c5b_{args.mode}_robust_cem_v1",
            "mode": args.mode,
            "clip": args.clip,
            "source_proposal_trace": str(source_path.resolve()),
            "source_proposal_trace_sha256": _sha256(source_path),
            "new_optimized_action_trajectory": selected_actions,
            "start_step": start_step,
            "planning_step_count": len(records),
            "records": records,
            "config": {
                "horizons": [1, 5, 10],
                "population": args.population,
                "iterations": 3,
                "elites": 8,
                "replicas": args.replicas,
                "initial_std": 0.35,
                "minimum_std": 0.05,
            },
            "persistent_candidate_environment": True,
            "candidate_setup_write_audit": pool.write_audit.as_dict(),
            "formal_execution_rollout_writes": pool.write_audit.formal_execution_rollout_writes,
            "no_hidden_control": bool(
                env_contract["object_rollout_state_writes"] == 0
                and env_contract["wrist_root_state_writes_during_step"] == 0
            ),
            "gpu_memory_mib": _gpu_memory_mib(),
            "wall_time_s": time.perf_counter() - started,
        }
        _write(args.output, payload)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "clip": args.clip,
                    "planning_steps": len(records),
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def _formal_row(env: Any, env_id: int, action_trace: np.ndarray, steps: int) -> dict[str, object]:
    stage = env.extras["stage16"]
    position = float(stage["object_position_error_m"][env_id].detach().cpu())
    rotation = math.degrees(float(stage["object_orientation_error_rad"][env_id].detach().cpu()))
    axis = float(stage["object_axis_error_m"][env_id].detach().cpu())
    reason = _reason_names(env._reason_codes[env_id : env_id + 1])[0]
    success = bool(env._success[env_id].detach().cpu())
    reach = int(env._reference_index[env_id].detach().cpu()) >= env.reference_bank.frame_count - 1
    diffs = np.linalg.norm(np.diff(action_trace[:steps], axis=0), axis=-1)
    return {
        "object_position_error_m": position,
        "object_rotation_error_deg": rotation,
        "object_axis_error_m": axis,
        "tracking_error": float(_tracking_error(env)[env_id].detach().cpu()),
        "contact_stability": 0.0,
        "smoothness": float(diffs.mean()) if diffs.size else 0.0,
        "effort": float(env._robot.data.applied_torque[env_id].abs().mean().detach().cpu()),
        "termination_reason": reason,
        "success": success,
        "final_reach": reach,
        "terminal_required": True,
        "steps": steps,
    }


def _run_formal(args: argparse.Namespace) -> int:
    if not args.accept_eula or args.action_report is None:
        raise SystemExit("formal C5C needs --accept-eula and --action-report")
    payload = json.loads(args.action_report.read_text(encoding="utf-8"))
    if payload.get("mode") != "b3" or payload.get("clip") != args.clip:
        raise ValueError("formal C5C action report must be the matching B3 output")
    actions = np.asarray(payload.get("new_optimized_action_trajectory"), dtype=np.float32)
    if actions.shape != (320, 26) or not np.isfinite(actions).all():
        raise ValueError("formal C5C requires a finite new [320,26] B3 action trace")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
        from toporetarget.rl.isaaclab_oracle.robust_oracle import (
            RobustCandidateReplicaV2,
            qualify_two_clip_c5c,
        )
        from toporetarget.rl.isaaclab_oracle.runtime import (
            make_stage16c5_env,
            reset_frozen_clip_frame_zero,
        )

        env = make_stage16c5_env(num_envs=20, contact_telemetry="aggregate")
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        ids = torch.arange(20, dtype=torch.long, device=env.device)
        reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=ids)
        finished: dict[int, dict[str, object]] = {}
        active = torch.ones(20, dtype=torch.bool, device=env.device)
        for step, action in enumerate(actions, start=1):
            batch = torch.as_tensor(action, device=env.device).expand(20, -1).clone()
            batch[~active] = 0.0
            terminated, timed_out = raw_control_step(env, batch)
            newly_finished = active & (terminated | timed_out)
            for env_id in torch.nonzero(newly_finished, as_tuple=False).flatten().tolist():
                finished[int(env_id)] = _formal_row(env, int(env_id), actions, step)
            active &= ~newly_finished
        for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
            finished[int(env_id)] = _formal_row(env, int(env_id), actions, 320)
        rows = [finished[index] for index in range(20)]
        replicas = [
            RobustCandidateReplicaV2(**{key: value for key, value in row.items() if key != "steps"})
            for row in rows
        ]
        other_clip = _CLIPS[1 - _CLIPS.index(args.clip)]
        # The single-worker report uses the exact shared aggregator by pairing
        # a placeholder copy; the parent closeout recomputes it with both clips.
        aggregate = qualify_two_clip_c5c({args.clip: replicas, other_clip: replicas})
        env_contract = env.contract_report()
        result = {
            "schema_version": "stage16c5c_formal_clip_worker_v1",
            "clip": args.clip,
            "action_report": str(args.action_report.resolve()),
            "action_report_sha256": _sha256(args.action_report),
            "episodes": rows,
            "clip_aggregate": aggregate["clips"][args.clip],
            "fresh_reset_frame0": True,
            "independent_physx_rollouts": True,
            "candidate_state_restore_used": False,
            "formal_execution_rollout_writes": 0,
            "no_hidden_control": bool(
                env_contract["object_rollout_state_writes"] == 0
                and env_contract["wrist_root_state_writes_during_step"] == 0
            ),
            "gpu_memory_mib": _gpu_memory_mib(),
        }
        _write(args.output, result)
        print(
            json.dumps(
                {
                    "clip": args.clip,
                    "passes": result["clip_aggregate"]["passes"],
                    "output": str(args.output),
                }
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def main() -> int:
    args = parse_args()
    return _run_formal(args) if args.mode == "formal" else _run_planning(args)


if __name__ == "__main__":
    raise SystemExit(main())
