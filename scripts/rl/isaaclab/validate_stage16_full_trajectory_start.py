#!/usr/bin/env python3
"""Verify the receipt-selected full-trajectory reset under nominal 1g.

This is a reset/support sanity check, not a trajectory-following qualification:
it uses a finite inferred table, applies the documented resting-velocity reset,
and advances only a few zero-residual-action control steps.
"""

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

from scripts.physics.validate_physical_scene_rsi import _make_physical_env, _support_contact
from toporetarget.rl.full_trajectory_episode_start import validate_full_trajectory_start

START_ROOT = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart/episode_start"
OUTPUT_ROOT = START_ROOT.parent / "nominal_1g_start_sanity"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--control-steps", type=int, default=3)
    return parser


def _selected_state(env: Any, clip: str) -> Any:
    if clip == "hocap_170105":
        return env._object_170105.data.root_state_w
    return env._object_170650.data.root_state_w


def main() -> int:
    args = _parser().parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.num_envs <= 0 or args.control_steps <= 0:
        raise ValueError("FULL_TRAJECTORY_START_SANITY_BUDGET_INVALID")
    start = validate_full_trajectory_start(
        json.loads((START_ROOT / f"{args.clip}.json").read_text(encoding="utf-8")),
        clip=args.clip,
    )
    output = OUTPUT_ROOT / args.clip / "sanity.json"
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        # The YAML curriculum's nominal roles are global 0.5/0.5 and the
        # authored HOCap object material is 1.0/1.0.  This is intentionally
        # outside C0--C2: no training-stage promotion is claimed here.
        env, _ = _make_physical_env(
            clip=args.clip,
            count=args.num_envs,
            start_indices=(int(start["start_index"]),) * args.num_envs,
            global_static_friction=0.5,
            global_dynamic_friction=0.5,
        )
        env.reset(seed=20260814)
        reset_state = _selected_state(env, args.clip).clone()
        reset_state[:, 7:] = 0.0
        if args.clip == "hocap_170105":
            env._object_170105.write_root_state_to_sim(reset_state)
        else:
            env._object_170650.write_root_state_to_sim(reset_state)
        initial = _selected_state(env, args.clip).clone()
        terminated_any = False
        timed_out_any = False
        reasons: set[str] = set()
        support_steps = 0
        for _ in range(args.control_steps):
            _, _, terminated, timed_out, extras = env.step(
                torch.zeros((args.num_envs, 26), device=env.device)
            )
            terminated_any |= bool(terminated.any().item())
            timed_out_any |= bool(timed_out.any().item())
            labels = tuple(extras["ppo26d"]["termination_reasons"])
            codes = extras["ppo26d"]["primary_reason_code"]
            reasons.update(labels[int(code)] for code in codes[terminated | timed_out].tolist())
            sensor = env.scene[
                "object_170105_support_contact"
                if args.clip == "hocap_170105"
                else "object_170650_support_contact"
            ]
            support_steps += int(_support_contact(sensor).sum())
        final = _selected_state(env, args.clip)
        displacement = torch.linalg.vector_norm(final[:, :3] - initial[:, :3], dim=-1)
        finite = bool(torch.isfinite(final).all().item())
        max_displacement_m = float(displacement.max().item())
        support_observed = support_steps > 0
        # This deliberately does not make a controller/reference termination
        # a reset failure: it only rejects an immediate physical launch/sink or
        # non-finite solver state.  It is not a full-reference replay.
        table_prevents_immediate_fall = finite and max_displacement_m <= 0.01
        passed = table_prevents_immediate_fall
        receipt = {
            "schema_version": "Stage16FullTrajectoryNominal1gStartSanityV1",
            "status": "PASS" if passed else "FAIL",
            "clip": args.clip,
            "purpose": "reset_and_table_support_sanity_only",
            "start": start,
            "physics": {
                "gravity_world_mps2": [0.0, 0.0, -9.81],
                "global_default_static_friction": 0.5,
                "global_default_dynamic_friction": 0.5,
                "hocap_object_material": "authored_nominal_1.0/1.0",
                "support": "finite_inferred_table_proxy_v1",
                "table_actor_active": True,
            },
            "reset_semantics": "TABLE_RESTING_RESET_SEMANTICS_V1",
            "mid_trajectory_rsi": "disabled",
            "control_steps": args.control_steps,
            "zero_residual_action": True,
            "support_contact_observed": support_observed,
            "support_contact_env_step_count": support_steps,
            "finite": finite,
            "max_object_displacement_m": max_displacement_m,
            "table_prevents_immediate_fall": table_prevents_immediate_fall,
            "terminated_any": terminated_any,
            "timed_out_any": timed_out_any,
            "termination_reasons_observed": sorted(reasons),
            "full_reference_following_claimed": False,
        }
        _write(output, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0 if passed else 2
    except BaseException as error:
        _write(
            output.with_name("sanity_failure.json"),
            {
                "schema_version": "Stage16FullTrajectoryNominal1gStartSanityFailureV1",
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
