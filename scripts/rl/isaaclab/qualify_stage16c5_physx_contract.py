#!/usr/bin/env python3
"""Run one fail-fast Stage 16-C.5A-R2 GPU/CPU contract qualification stage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
_STAGE_PHASE = {
    "S1": "pre_contact",
    "S2": "contact_onset",
    "S3": "sustained_contact",
    "S4": "post_contact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--stage", choices=_STAGES, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_R2_CONTRACT_STAGE_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_frames(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
        raise ValueError("Stage16 C5A R2 phase table is malformed")
    rows = payload["clips"]
    if len(rows) != 2 or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Stage16 C5A R2 requires exactly two clip phase rows")
    return rows


def _scalar_max(value: Any) -> float:
    return float(value.abs().amax().detach().cpu())


def _raw_maxima(env: Any, ids: Any) -> dict[str, float]:
    import torch

    from toporetarget.rl.isaaclab_oracle.candidate_state import capture_candidate_state

    tensors = capture_candidate_state(env, ids).tensors
    result: dict[str, float] = {}
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] < 2:
            continue
        if value.dtype in {torch.bool, torch.long}:
            result[name] = float((value[1:] != value[:1]).to(torch.float32).amax().detach().cpu())
        else:
            result[name] = _scalar_max(value[1:] - value[:1])
    return result


def _trial_measurement(env: Any, ids: Any) -> dict[str, object]:
    import torch

    from toporetarget.rl.isaaclab_oracle.metrics import state_differences
    from toporetarget.rl.isaaclab_oracle.runtime import state_view

    if str(env.device).startswith("cuda"):
        torch.cuda.synchronize(env.device)
    state = state_view(env, ids)
    source = {
        name: value[:1].expand_as(value[1:])
        for name, value in state.items()
        if name not in {"reference_index", "reason_codes"}
    }
    peers = {
        name: value[1:]
        for name, value in state.items()
        if name not in {"reference_index", "reason_codes"}
    }
    metrics = state_differences(source, peers)
    rewards = {
        name: float((value[1:] - value[:1]).abs().amax().detach().cpu())
        for name, value in env._last_reward_terms.items()
    }
    metrics["reward"] = rewards["total"]
    termination_exact = bool(
        torch.eq(env.reset_terminated[1:], env.reset_terminated[:1]).all()
        and torch.eq(env.reset_time_outs[1:], env.reset_time_outs[:1]).all()
        and torch.eq(env._reason_codes[1:], env._reason_codes[:1]).all()
    )
    return {
        "metrics": metrics,
        "reward_components": rewards,
        "termination_exact": termination_exact,
        "raw_state_max_abs": _raw_maxima(env, ids),
        "observation_finite": bool(torch.isfinite(env._get_observations()["policy"]).all()),
    }


def _runtime_values(env: Any, contract: Any) -> dict[str, object]:
    """Capture requested config and check the live USD attributes without fallbacks."""

    import omni.usd

    expected = contract.as_dict()
    cfg = env.cfg
    requested = {
        "device": str(cfg.sim.device),
        "enhanced_determinism": cfg.sim.physx.enable_enhanced_determinism,
        "solve_articulation_contact_last": cfg.sim.physx.solve_articulation_contact_last,
        "scene_max_position_iterations": cfg.sim.physx.max_position_iteration_count,
        "scene_max_velocity_iterations": cfg.sim.physx.max_velocity_iteration_count,
        "actor_position_iterations": (
            cfg.robot.spawn.articulation_props.solver_position_iteration_count
        ),
        "actor_velocity_iterations": (
            cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
        ),
        "object_position_iterations": (
            cfg.object_170105.spawn.rigid_props.solver_position_iteration_count
        ),
        "object_velocity_iterations": (
            cfg.object_170105.spawn.rigid_props.solver_velocity_iteration_count
        ),
    }
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("STAGE16C5A_R2_RUNTIME_USD_STAGE_UNAVAILABLE")
    prims = {str(prim.GetPath()): prim for prim in stage.Traverse()}
    scene = prims.get("/physicsScene")
    robot = prims.get("/World/envs/env_0/Robot")
    object_105 = prims.get("/World/envs/env_0/Object170105")
    object_650 = prims.get("/World/envs/env_0/Object170650")
    if any(prim is None for prim in (scene, robot, object_105, object_650)):
        raise RuntimeError("STAGE16C5A_R2_RUNTIME_USD_PRIM_MISSING")

    def attribute(prim: Any, name: str) -> object:
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            return "MISSING"
        return attr.Get()

    actual = {
        "enhanced_determinism": attribute(scene, "physxScene:enableEnhancedDeterminism"),
        "solve_articulation_contact_last": attribute(
            scene, "physxScene:solveArticulationContactLast"
        ),
        "scene_max_position_iterations": attribute(scene, "physxScene:maxPositionIterationCount"),
        "scene_max_velocity_iterations": attribute(scene, "physxScene:maxVelocityIterationCount"),
        "actor_position_iterations": attribute(
            robot, "physxArticulation:solverPositionIterationCount"
        ),
        "actor_velocity_iterations": attribute(
            robot, "physxArticulation:solverVelocityIterationCount"
        ),
        "object_170105_position_iterations": attribute(
            object_105, "physxRigidBody:solverPositionIterationCount"
        ),
        "object_170105_velocity_iterations": attribute(
            object_105, "physxRigidBody:solverVelocityIterationCount"
        ),
        "object_170650_position_iterations": attribute(
            object_650, "physxRigidBody:solverPositionIterationCount"
        ),
        "object_170650_velocity_iterations": attribute(
            object_650, "physxRigidBody:solverVelocityIterationCount"
        ),
    }
    expected_actual = {
        "enhanced_determinism": expected["enhanced_determinism"],
        "solve_articulation_contact_last": expected["solve_articulation_contact_last"],
        "scene_max_position_iterations": expected["scene_max_position_iterations"],
        "scene_max_velocity_iterations": expected["scene_max_velocity_iterations"],
        "actor_position_iterations": expected["actor_position_iterations"],
        "actor_velocity_iterations": expected["actor_velocity_iterations"],
        "object_170105_position_iterations": expected["actor_position_iterations"],
        "object_170105_velocity_iterations": expected["actor_velocity_iterations"],
        "object_170650_position_iterations": expected["actor_position_iterations"],
        "object_170650_velocity_iterations": expected["actor_velocity_iterations"],
    }
    return {
        "contract": contract.as_dict(),
        "cfg": requested,
        "usd": actual,
        "expected_usd": expected_actual,
        "runtime_values_match_contract": actual == expected_actual,
    }


def _aggregate(
    rows: list[dict[str, object]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    metrics: dict[str, list[float]] = {}
    rewards: dict[str, list[float]] = {}
    for row in rows:
        values = row["metrics"]
        components = row["reward_components"]
        assert isinstance(values, Mapping) and isinstance(components, Mapping)
        for name, value in values.items():
            metrics.setdefault(str(name), []).append(float(value))
        for name, value in components.items():
            rewards.setdefault(str(name), []).append(float(value))
    return metrics, rewards


def _phase_rows(
    args: argparse.Namespace, env: Any, frames: list[dict[str, Any]]
) -> list[dict[str, object]]:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero
    from toporetarget.rl.isaaclab_oracle.tolerance import freeze_tolerances, summarize_noise_floor

    phase_names = list(_STAGE_PHASE.values()) if args.stage == "S5" else [_STAGE_PHASE[args.stage]]
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    rows: list[dict[str, object]] = []
    for clip_index, clip_row in enumerate(frames):
        frame_map = clip_row.get("frames")
        if not isinstance(frame_map, Mapping):
            raise ValueError("Stage16 C5A R2 phase map is malformed")
        for phase in phase_names:
            frame = int(frame_map[phase])
            trials: list[dict[str, object]] = []
            for trial_index in range(args.trials):
                reset_frozen_clip_frame_zero(env, clip_index=clip_index)
                actions = torch.zeros((env.num_envs, 26), dtype=torch.float32, device=env.device)
                for step in range(frame + 1):
                    terminated, timed_out = raw_control_step(env, actions)
                    if bool((terminated | timed_out).any()) and step < frame:
                        raise RuntimeError(
                            "STAGE16C5A_R2_NATURAL_BASELINE_EARLY_TERMINATION: "
                            f"clip={clip_row['clip']} phase={phase} trial={trial_index} step={step}"
                        )
                trials.append(_trial_measurement(env, ids))
            metrics, rewards = _aggregate(trials)
            tolerance = freeze_tolerances(metrics)
            rows.append(
                {
                    "clip": clip_row["clip"],
                    "phase": phase,
                    "frame": frame,
                    "trials": trials,
                    "tolerances": tolerance,
                    "reward_component_summary": summarize_noise_floor(rewards),
                    "termination_exact_all_trials": all(
                        bool(trial["termination_exact"]) for trial in trials
                    ),
                    "finite_all_trials": all(bool(trial["observation_finite"]) for trial in trials),
                }
            )
    return rows


def _s0(args: argparse.Namespace, env: Any) -> dict[str, object]:
    import torch

    from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
    from toporetarget.rl.isaaclab_oracle.runtime import reset_frozen_clip_frame_zero

    reset_frozen_clip_frame_zero(env, clip_index=0)
    actions = torch.zeros((1, 26), dtype=torch.float32, device=env.device)
    finite = True
    terminated_early = False
    for _ in range(100):
        terminated, timed_out = raw_control_step(env, actions)
        finite = finite and bool(torch.isfinite(env._get_observations()["policy"]).all())
        terminated_early = terminated_early or bool((terminated | timed_out).any())
    return {
        "one_environment_no_contact_smoke_steps": 100,
        "finite": finite,
        "terminated_or_timed_out": terminated_early,
        "contact_record_total": int(getattr(env, "_contact_substep_record_total", 0)),
        "pass": finite and not terminated_early,
    }


def main() -> int:
    args = parse_args()
    if not args.accept_eula or args.trials != 20:
        raise SystemExit(
            "Stage16 C5A R2 contract qualification requires --accept-eula and 20 trials"
        )
    if args.output.exists():
        raise FileExistsError(f"STAGE16C5A_R2_OUTPUT_ALREADY_EXISTS: {args.output}")
    from toporetarget.rl.environments.isaaclab_backend.physx_contract import load_contract

    contract = load_contract(args.matrix, args.candidate_id)
    frames = _load_frames(args.frames)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env
        from toporetarget.rl.isaaclab_oracle.tolerance import freeze_tolerances

        env_count = 1 if args.stage == "S0" else 33
        env = make_stage16c5_env(
            num_envs=env_count,
            contact_telemetry="aggregate",
            physx_contract=contract,
        )
        runtime = _runtime_values(env, contract)
        if args.stage == "S0":
            stage_result: dict[str, object] = _s0(args, env)
            passed = bool(stage_result["pass"] and runtime["runtime_values_match_contract"])
        else:
            rows = _phase_rows(args, env, frames)
            all_metrics, _rewards = _aggregate(
                [trial for row in rows for trial in row["trials"]]  # type: ignore[index]
            )
            global_tolerance = freeze_tolerances(all_metrics)
            passed = bool(
                global_tolerance["status"] == "REPLICATION_TOLERANCES_FROZEN"
                and runtime["runtime_values_match_contract"]
                and all(bool(row["termination_exact_all_trials"]) for row in rows)
                and all(bool(row["finite_all_trials"]) for row in rows)
            )
            stage_result = {"phase_rows": rows, "global_tolerances": global_tolerance}
        execution_contract = env.contract_report()
        no_hidden_writes = (
            execution_contract["object_rollout_state_writes"] == 0
            and execution_contract["wrist_root_state_writes_during_step"] == 0
        )
        passed = passed and no_hidden_writes
        report = {
            "schema_version": "stage16c5a_r2_physx_contract_stage_v1",
            "candidate_id": args.candidate_id,
            "candidate_identifier": contract.identifier,
            "stage": args.stage,
            "trials": args.trials,
            "num_envs": env_count,
            "device": str(env.device),
            "process": {"pid": os.getpid(), "pgid": os.getpgid(0)},
            "runtime_config": runtime,
            "result": "PASS" if passed else "FAIL",
            "config_fallback_detected": not runtime["runtime_values_match_contract"],
            "no_hidden_execution_state_writes": no_hidden_writes,
            "execution_write_audit": {
                "wrist": execution_contract["wrist_root_state_writes_during_step"],
                "object": execution_contract["object_rollout_state_writes"],
            },
            "stage_result": stage_result,
        }
        _write(args.output, report)
        print(
            json.dumps(
                {"candidate": args.candidate_id, "stage": args.stage, "result": report["result"]}
            )
        )
        return 0
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
