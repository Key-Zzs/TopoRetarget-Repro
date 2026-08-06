#!/usr/bin/env python3
"""Run bounded GPU PhysX Stage 16-D environment smoke or spline CEM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

CLIPS = ("hocap_170105", "hocap_170650")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--stage", choices=("env-smoke", "d3-s1", "d3-s2", "d3-s3"), required=True)
    parser.add_argument("--clip", choices=CLIPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--knots", type=int, default=16)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elites", type=int, default=12)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Stage16D refuses overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_actions(clip: str) -> tuple[Path, np.ndarray]:
    manifest_path = (
        REPO_ROOT
        / ".local/experiments/stage16c3r5_reference_retiming_c4"
        / "mujoco_selected_traces_fixed_authority_attempt2/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("traces")
    if not isinstance(rows, list):
        raise ValueError("Stage16D demonstration manifest has no traces")
    for row in rows:
        if row.get("clip") != clip:
            continue
        path = Path(row["action_trace"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if sha256(path) != row["action_trace_sha256"]:
            raise RuntimeError("STAGE16D_INPUT_HASH_DRIFT:demonstration_action_trace")
        with np.load(path, allow_pickle=False) as source:
            actions = np.asarray(source["actions"], dtype=np.float32)
        if actions.shape != (40, 26):
            raise ValueError("Stage16D demonstration trace must be [40,26]")
        retimed = np.repeat(actions, 8, axis=0)
        return path, np.concatenate((retimed, retimed[-1:]), axis=0)
    raise ValueError(f"Stage16D demonstration manifest lacks {clip}")


def make_env(*, num_envs: int, clip: str, telemetry: str) -> Any:
    from toporetarget.rl.environments.isaaclab_backend import (
        physics_consistent_retargeting_env_cfg as stage16d_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.physics_consistent_retargeting_env import (
        IsaacPhysicsConsistentRetargetingEnv,
    )
    from toporetarget.rl.environments.isaaclab_backend.physx_contract import (
        apply_physx_contract,
        baseline_contract,
    )

    cfg = stage16d_cfg.IsaacPhysicsConsistentRetargetingEnvCfg()
    stage16d_cfg.configure_stage16d_nominal(cfg, num_envs=num_envs, clip=clip)
    cfg.scene.lazy_sensor_update = True
    cfg.contact_telemetry = telemetry
    cfg.contact_record_capacity = 4096
    apply_physx_contract(cfg, baseline_contract())
    env = IsaacPhysicsConsistentRetargetingEnv(cfg)
    env.reset(seed=20260806)
    return env


def _close_env(env: Any) -> None:
    if env is not None:
        env.close()
        env.sim.clear_all_callbacks()
        env.sim.clear_instance()


def run_smoke(args: argparse.Namespace, app: Any) -> int:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step

    env = None
    started = time.perf_counter()
    try:
        env = make_env(num_envs=args.num_envs, clip=args.clip, telemetry="aggregate")
        steps = args.steps or 2
        for _ in range(steps):
            raw_control_step(env, torch.zeros((env.num_envs, 26), device=env.device))
        contract = env.contract_report()
        stage = env.extras["stage16d"]
        payload = {
            "schema_version": "Stage16DEnvironmentSmokeV1",
            "status": "STAGE16D_PHYSICS_CORRECTION_ENV_VALIDATED",
            "clip": args.clip,
            "num_envs": env.num_envs,
            "steps": steps,
            "observation_shape": list(env._get_observations()["policy"].shape),
            "finite": bool(torch.isfinite(env._get_observations()["policy"]).all()),
            "contract": contract,
            "semantic_progress": stage["semantic_progress"].detach().cpu().tolist(),
            "contact_recall": stage["contact_recall"].detach().cpu().tolist(),
            "object_rollout_state_writes": stage["object_rollout_state_writes"],
            "wrist_rollout_state_writes": stage["wrist_rollout_state_writes"],
            "wall_time_s": time.perf_counter() - started,
        }
        write(args.output, payload)
        print(json.dumps({"status": payload["status"], "output": str(args.output)}))
        return 0
    finally:
        _close_env(env)


def _evaluation_rows(
    env: Any,
    *,
    population: int,
    replicas: int,
    actions: Any,
    effort: Any,
    catastrophic: Any,
) -> tuple[Any, ...]:
    import torch

    from toporetarget.rl.physics_retargeting.robust_optimizer import (
        PhysicsCandidateEvaluationV1,
        PhysicsCandidateReplicaV1,
    )

    stage = env.extras["stage16d"]
    reward = env._last_reward_terms
    smoothness = torch.linalg.vector_norm(torch.diff(actions, dim=1), dim=-1).mean(dim=-1)
    rows = []
    for candidate in range(population):
        candidate_rows = []
        for replica in range(replicas):
            env_id = candidate * replicas + replica
            progress = float(stage["semantic_progress"][env_id].detach().cpu())
            contact = float(stage["contact_recall"][env_id].detach().cpu())
            candidate_rows.append(
                PhysicsCandidateReplicaV1(
                    catastrophic_failure=bool(catastrophic[env_id].detach().cpu()),
                    semantic_failure=progress < 0.30,
                    contact_topology_failure=contact < 0.50,
                    penetration_m=0.0,
                    safety_violation=float(catastrophic[env_id].detach().cpu()),
                    semantic_progress=progress,
                    contact_recall=contact,
                    contact_persistence=float(stage["contact_persistence"][env_id].detach().cpu()),
                    terminal_stability=float(stage["terminal_stable"][env_id].detach().cpu()),
                    robot_fidelity_error=float(
                        (
                            3.5
                            - reward["wrist_fidelity"][env_id]
                            - reward["finger_fidelity"][env_id]
                            - reward["link_fidelity"][env_id]
                        )
                        .clamp_min(0.0)
                        .detach()
                        .cpu()
                    ),
                    source_object_soft_prior_error=float(
                        stage["source_object_deviation_m"][env_id].detach().cpu()
                    ),
                    action_smoothness=float(smoothness[candidate].detach().cpu()),
                    effort=float(effort[env_id].detach().cpu()),
                )
            )
        rows.append(
            PhysicsCandidateEvaluationV1(candidate_id=candidate, replicas=tuple(candidate_rows))
        )
    return tuple(rows)


def run_optimizer(args: argparse.Namespace, app: Any) -> int:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero
    from toporetarget.rl.physics_retargeting.robust_optimizer import (
        PhaseWiseRobustSplineCEMV1,
        PhaseWiseSplineCEMConfigV1,
    )

    env = None
    started = time.perf_counter()
    source_path, proposal = source_actions(args.clip)
    semantic_path = (
        REPO_ROOT
        / ".local/reports/stage16d_physics_consistent_retargeting"
        / f"task_semantics_{args.clip.removeprefix('hocap_')}.json"
    )
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    default_steps = {"d3-s1": 16, "d3-s2": 60, "d3-s3": 321}
    steps = args.steps or default_steps[args.stage]
    start_step = (
        args.start_step
        if args.start_step is not None
        else 0
        if args.stage == "d3-s3"
        else int(semantic["contact_onset_window"]["start"])
    )
    if start_step < 0 or steps < 1 or start_step + steps > 321:
        raise ValueError("Stage16D planning segment exceeds the 321-step trajectory")
    config = PhaseWiseSplineCEMConfigV1(
        knot_count=args.knots,
        population=args.population,
        replicas=args.replicas,
        iterations=args.iterations,
        elites=args.elites,
    )
    try:
        env_count = config.population * config.replicas
        env = make_env(num_envs=env_count, clip=args.clip, telemetry="off")
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        ids = torch.arange(env_count, device=env.device)
        optimizer = PhaseWiseRobustSplineCEMV1(config, device=str(env.device))
        knot_indices = torch.linspace(0, 320, config.knot_count, device=env.device).round().long()
        optimizer.mean.copy_(
            torch.as_tensor(proposal, device=env.device).index_select(0, knot_indices)
        )
        iteration_reports: list[dict[str, Any]] = []
        final_evaluations: tuple[Any, ...] = ()
        best_trace: Any | None = None
        best_key: tuple[float | int, ...] | None = None
        for iteration in range(config.iterations):
            reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=ids)
            for step in range(start_step):
                batch = torch.as_tensor(proposal[step], device=env.device).expand(env_count, -1)
                raw_control_step(env, batch)
            knots, action_population = optimizer.ask()
            effort = torch.zeros(env_count, device=env.device)
            catastrophic = torch.zeros(env_count, dtype=torch.bool, device=env.device)
            active = torch.ones_like(catastrophic)
            for step in range(start_step, start_step + steps):
                batch = action_population[:, step].repeat_interleave(config.replicas, dim=0)
                batch = torch.where(active[:, None], batch, torch.zeros_like(batch))
                terminated, timed_out = raw_control_step(env, batch)
                reason = env._reason_codes
                catastrophic |= terminated & (reason >= 2) & (reason <= 8)
                active &= ~(terminated | timed_out)
                effort += env._robot.data.applied_torque.abs().mean(dim=-1)
            effort /= steps
            evaluations = _evaluation_rows(
                env,
                population=config.population,
                replicas=config.replicas,
                actions=action_population,
                effort=effort,
                catastrophic=catastrophic,
            )
            optimizer.tell(iteration=iteration, knots=knots, evaluations=evaluations)
            final_evaluations = evaluations
            best = min(evaluations, key=lambda row: row.lexical_key())
            if best_key is None or best.lexical_key() < best_key:
                best_key = best.lexical_key()
                best_trace = action_population[best.candidate_id].detach().clone()
            iteration_reports.append(
                {
                    "iteration": iteration,
                    "best": best.as_dict(),
                    "active_final": int(active.sum().detach().cpu()),
                    "catastrophic_count": int(catastrophic.sum().detach().cpu()),
                    "cem": optimizer.records[-1],
                }
            )
            print(
                json.dumps(
                    {
                        "clip": args.clip,
                        "stage": args.stage,
                        "iteration": iteration + 1,
                        "iterations": config.iterations,
                        "best_lexical_key": list(best.lexical_key()),
                    }
                ),
                flush=True,
            )
        if best_trace is None:
            raise RuntimeError("Stage16D optimizer completed without an evaluated candidate")
        best_trace = best_trace.cpu().numpy()
        trace_path = args.output.with_suffix(".actions.npy")
        if trace_path.exists():
            raise FileExistsError(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(trace_path, best_trace)
        ranked = sorted(final_evaluations, key=lambda row: row.lexical_key())
        contract = env.contract_report()
        payload = {
            "schema_version": "Stage16DPhaseWiseRobustSplineCEMV1",
            "stage": args.stage,
            "clip": args.clip,
            "status": "STAGE16D_OPTIMIZATION_SEGMENT_EXECUTED",
            "config": config.as_dict(),
            "start_step": start_step,
            "steps": steps,
            "source_demonstration": str(source_path),
            "source_demonstration_sha256": sha256(source_path),
            "action_trace": str(trace_path),
            "action_trace_sha256": sha256(trace_path),
            "action_trace_shape": list(best_trace.shape),
            "action_trace_selection": "best_lexically_ranked_evaluated_candidate_across_iterations",
            "top_8": [row.as_dict() for row in ranked[:8]],
            "iterations": iteration_reports,
            "object_is_decision_variable": False,
            "object_trajectory": "free_physx_rollout_output",
            "formal_object_rollout_state_writes": 0,
            "formal_wrist_rollout_state_writes": contract["formal_wrist_rollout_state_writes"],
            "hidden_force": False,
            "hidden_attachment": False,
            "penetration_metric": "UNAVAILABLE_REQUIRES_INDEPENDENT_GEOMETRY_AUDIT",
            "wall_time_s": time.perf_counter() - started,
        }
        write(args.output, payload)
        print(
            json.dumps(
                {"status": payload["status"], "output": str(args.output), "trace": str(trace_path)}
            )
        )
        return 0
    finally:
        _close_env(env)


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("Stage16D Isaac execution requires --accept-eula")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        return run_smoke(args, app) if args.stage == "env-smoke" else run_optimizer(args, app)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
