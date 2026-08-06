#!/usr/bin/env python3
"""Run the single authorized Stage 16-D terminal-tail refinement profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.physics_retargeting.terminal_refinement import (  # noqa: E402
    TerminalTailRefinementConfigV1,
    materialize_terminal_tail,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--failure-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-eula", action="store_true")
    return parser.parse_args()


def _make_env(*, num_envs: int, clip: str) -> Any:
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
    cfg.contact_telemetry = "off"
    apply_physx_contract(cfg, baseline_contract())
    env = IsaacPhysicsConsistentRetargetingEnv(cfg)
    env.reset(seed=20260806)
    return env


def _upper_cvar(values: np.ndarray, alpha: float = 0.75) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = max(1, math.ceil((1.0 - alpha) * len(ordered)))
    return float(ordered[-count:].mean())


def run(args: argparse.Namespace) -> int:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    analysis = json.loads(args.failure_analysis.read_text(encoding="utf-8"))
    if analysis.get("clip") != args.clip or not analysis.get("terminal_only_refinement_authorized"):
        raise RuntimeError("STAGE16D_TERMINAL_REFINEMENT_NOT_AUTHORIZED")
    baseline_np = np.asarray(np.load(args.actions, allow_pickle=False), dtype=np.float32)
    if baseline_np.shape != (321, 26) or not np.isfinite(baseline_np).all():
        raise ValueError("terminal refinement actions must be finite [321,26]")
    config = TerminalTailRefinementConfigV1()
    env = None
    started = time.perf_counter()
    try:
        env_count = config.population * config.replicas
        env = _make_env(num_envs=env_count, clip=args.clip)
        device = env.device
        ids = torch.arange(env_count, device=device)
        clip_index = env.reference_bank.clip_ids.index(args.clip)
        baseline = torch.as_tensor(baseline_np, device=device)
        knot_frames = (
            torch.linspace(
                config.tail_start,
                config.frame_count - 1,
                config.knot_count,
                device=device,
            )
            .round()
            .long()
        )
        mean = baseline.index_select(0, knot_frames).clone()
        std = torch.full_like(mean, config.initial_std)
        std[0].zero_()
        generator = torch.Generator(device=device).manual_seed(config.seed)
        reports: list[dict[str, Any]] = []
        best_key: tuple[float | int, ...] | None = None
        best_actions: torch.Tensor | None = None
        best_candidate_report: dict[str, Any] | None = None
        for iteration in range(config.iterations):
            noise = torch.randn(
                (config.population, config.knot_count, 26),
                generator=generator,
                device=device,
            )
            knots = (mean[None] + std[None] * noise).clamp(-1.0, 1.0)
            actions = materialize_terminal_tail(baseline, knots, config)
            reset_frozen_clip_frame_zero(env, clip_index=clip_index, env_ids=ids)
            active = torch.ones(env_count, dtype=torch.bool, device=device)
            catastrophic = torch.zeros_like(active)
            ever_success = torch.zeros_like(active)
            ever_causality = torch.zeros_like(active)
            max_progress = torch.zeros(env_count, device=device)
            max_contact = torch.zeros_like(max_progress)
            effort = torch.zeros_like(max_progress)
            for step in range(config.frame_count):
                population_batch = actions[:, step].repeat_interleave(config.replicas, dim=0)
                batch = torch.where(
                    active[:, None], population_batch, torch.zeros_like(population_batch)
                )
                terminated, timed_out = raw_control_step(env, batch)
                stage = env.extras["stage16d"]
                max_progress = torch.maximum(max_progress, stage["semantic_progress"])
                max_contact = torch.maximum(max_contact, stage["contact_recall"])
                ever_success |= stage["success"]
                ever_causality |= stage["contact_causality"]
                if step >= config.tail_start:
                    effort += env._robot.data.applied_torque.abs().mean(dim=-1)
                reason = stage["primary_reason_code"]
                catastrophic |= terminated & (reason >= 2) & (reason <= 8)
                active &= ~(terminated | timed_out)
            stage = env.extras["stage16d"]
            terminal_stable = stage["terminal_stable"]
            angular_speed = torch.linalg.vector_norm(
                env._state()["object_twist_world"][:, 3:], dim=-1
            )
            tail_smoothness = torch.linalg.vector_norm(
                torch.diff(actions[:, config.tail_start :], dim=1), dim=-1
            ).mean(dim=-1)
            effort /= config.frame_count - config.tail_start
            candidate_rows = []
            keys = []
            for candidate in range(config.population):
                selection = slice(candidate * config.replicas, (candidate + 1) * config.replicas)
                candidate_catastrophic = catastrophic[selection].detach().cpu().numpy()
                candidate_success = ever_success[selection].detach().cpu().numpy()
                candidate_progress = max_progress[selection].detach().cpu().numpy()
                candidate_contact = max_contact[selection].detach().cpu().numpy()
                candidate_causality = ever_causality[selection].detach().cpu().numpy()
                candidate_terminal = terminal_stable[selection].detach().cpu().numpy()
                candidate_angular = angular_speed[selection].detach().cpu().numpy()
                key: tuple[float | int, ...] = (
                    float(candidate_catastrophic.mean()),
                    float(np.mean(candidate_progress < 0.30)),
                    float(np.mean(candidate_contact < 0.50)),
                    -float(candidate_success.mean()),
                    -float(candidate_terminal.mean()),
                    _upper_cvar(candidate_angular),
                    -float(candidate_causality.mean()),
                    float(tail_smoothness[candidate].detach().cpu()),
                    float(effort[selection].mean().detach().cpu()),
                    candidate,
                )
                row = {
                    "candidate_id": candidate,
                    "lexical_key": list(key),
                    "catastrophic_failure_rate": float(candidate_catastrophic.mean()),
                    "success_rate": float(candidate_success.mean()),
                    "semantic_reach_rate": float(np.mean(candidate_progress >= 0.30)),
                    "contact_pass_rate": float(np.mean(candidate_contact >= 0.50)),
                    "contact_causality_rate": float(candidate_causality.mean()),
                    "terminal_stability_rate": float(candidate_terminal.mean()),
                    "angular_speed_radps": candidate_angular.tolist(),
                    "action_smoothness": float(tail_smoothness[candidate].detach().cpu()),
                    "mean_effort": float(effort[selection].mean().detach().cpu()),
                }
                candidate_rows.append(row)
                keys.append(key)
            ranked = sorted(range(config.population), key=lambda index: keys[index])
            elite_ids = torch.as_tensor(ranked[: config.elites], device=device)
            elite = knots.index_select(0, elite_ids)
            previous_mean = mean.clone()
            mean = elite.mean(dim=0)
            std = elite.std(dim=0, correction=0).clamp_min(config.minimum_std)
            mean[0] = baseline[config.tail_start]
            std[0].zero_()
            best_id = ranked[0]
            if best_key is None or keys[best_id] < best_key:
                best_key = keys[best_id]
                best_actions = actions[best_id].detach().clone()
                best_candidate_report = candidate_rows[best_id]
            iteration_row = {
                "iteration": iteration,
                "best": candidate_rows[best_id],
                "elite_candidate_ids": ranked[: config.elites],
                "mean_shift_l2": float(torch.linalg.vector_norm(mean - previous_mean).cpu()),
                "mean_std": float(std[1:].mean().cpu()),
            }
            reports.append(iteration_row)
            print(
                json.dumps(
                    {
                        "iteration": iteration + 1,
                        "iterations": config.iterations,
                        "best_success_rate": candidate_rows[best_id]["success_rate"],
                        "best_terminal_stability_rate": candidate_rows[best_id][
                            "terminal_stability_rate"
                        ],
                        "best_upper_angular_speed_radps": keys[best_id][5],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if best_actions is None or best_candidate_report is None:
            raise RuntimeError("STAGE16D_TERMINAL_REFINEMENT_NO_CANDIDATE")
        action_path = args.output.with_suffix(".actions.npy")
        if args.output.exists() or action_path.exists():
            raise FileExistsError(f"terminal refinement refuses overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(action_path, best_actions.cpu().numpy())
        prefix_equal = bool(
            np.array_equal(
                best_actions[: config.tail_start].cpu().numpy(), baseline_np[: config.tail_start]
            )
        )
        payload = {
            "schema_version": "Stage16DTerminalTailRefinementV1",
            "status": "STAGE16D_TERMINAL_TAIL_REFINEMENT_EXECUTED",
            "clip": args.clip,
            "config": config.as_dict(),
            "source_actions": str(args.actions),
            "source_actions_sha256": _sha256(args.actions),
            "failure_analysis": str(args.failure_analysis),
            "failure_analysis_sha256": _sha256(args.failure_analysis),
            "action_trace": str(action_path),
            "action_trace_sha256": _sha256(action_path),
            "prefix_byte_equivalent": prefix_equal,
            "boundary_continuity_exact": bool(
                torch.equal(best_actions[config.tail_start], baseline[config.tail_start])
            ),
            "action_bounds_pass": bool((best_actions.abs() <= 1.0).all()),
            "object_is_decision_variable": False,
            "object_state_writes": 0,
            "wrist_state_writes": 0,
            "hidden_force": False,
            "hidden_attachment": False,
            "best_four_replica_evaluation": best_candidate_report,
            "iterations": reports,
            "wall_time_s": time.perf_counter() - started,
        }
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": payload["status"], "output": str(args.output)}))
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("Stage16D terminal refinement requires --accept-eula")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    try:
        return run(args)
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
