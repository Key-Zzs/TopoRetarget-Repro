#!/usr/bin/env python3
"""Run a non-training full-horizon reference-index/phase wiring check."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _load_start, _make_table_env
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode

PHASES = (
    "PRE_CONTACT",
    "APPROACH",
    "CONTACT",
    "GRASP",
    "LIFT",
    "MANIPULATION",
    "TERMINAL",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula or args.num_envs <= 0:
        raise ValueError("FULL_TRAJECTORY_PROGRESSION_ARGUMENT_INVALID")
    start = _load_start(args.clip)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        env = _make_table_env(
            clip=args.clip,
            num_envs=args.num_envs,
            start_index=int(start["start_index"]),
            mode=ContactRewardMode.AGGREGATE_V3,
            stage="C0",
        )
        observation, _ = env.reset(seed=20260815)
        del observation
        indices = [int(env._reference_index[0].item())]
        phase_codes = [0]
        terminal_step = env.reference_bank.frame_count - 2
        terminated_before_terminal = False
        for step in range(env.reference_bank.frame_count - 1):
            action = torch.zeros((args.num_envs, 26), dtype=torch.float32, device=env.device)
            _, _, terminated, timed_out, _ = env.step(action)
            indices.append(int(env._reference_index[0].item()))
            phase_codes.append(int(env.stage16_saturation_telemetry()["phase_code"][0].item()))
            terminated_before_terminal |= (
                bool((terminated | timed_out)[0].item()) and step < terminal_step
            )
        reached = set(phase_codes)
        # DirectRLEnv resets a completed environment in the same step, so the
        # final visible index returns to zero after the terminal target.  The
        # penultimate target (319 for a 321-frame reference) plus the terminal
        # phase is the observable full-horizon completion receipt.
        full_horizon_completed = max(indices) >= env.reference_bank.frame_count - 2 and 6 in reached
        result = {
            "schema_version": "Stage16FullTrajectoryProgressionCheckV1",
            "clip": args.clip,
            "training_updates": 0,
            "reference_index_initial": indices[0],
            "reference_index_final": indices[-1],
            "reference_index_max": max(indices),
            "reference_frame_count": env.reference_bank.frame_count,
            "reference_index_progressing": full_horizon_completed,
            "phase_codes_reached": phase_codes,
            "phases_reached": [PHASES[index] for index in sorted(reached)],
            "all_full_trajectory_phases_reached": reached == set(range(len(PHASES))),
            "terminated_before_terminal": terminated_before_terminal,
            "environment": env.contract_report(),
        }
        result["status"] = (
            "PASS"
            if result["reference_index_progressing"]
            and result["all_full_trajectory_phases_reached"]
            and not result["terminated_before_terminal"]
            else "FAIL"
        )
        _write(args.output.resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    except BaseException as error:
        _write(
            args.output.resolve(),
            {
                "schema_version": "Stage16FullTrajectoryProgressionCheckFailureV1",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
