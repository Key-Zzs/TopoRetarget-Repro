#!/usr/bin/env python3
"""Check that a bounded uniform-RSI reset set initializes finite C1 states."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.rl.isaaclab.smoke_stage16_full_trajectory_ppo import _make_table_env
from toporetarget.rl.reference_tracking.contact_reward_mode import ContactRewardMode


def _finite(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / ".local/reports/stage16_contact_stable_physical_continuation/c1/reset_sanity.json",
    )
    args = parser.parse_args()
    if not args.accept_eula or args.samples not in {32, 64}:
        raise ValueError("CONTACT_STABLE_C1_RSI_SANITY_CONTRACT_INVALID")

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        env = _make_table_env(
            clip="hocap_170105",
            num_envs=args.samples,
            start_index=0,
            mode=ContactRewardMode.AGGREGATE_V3,
            stage="C1",
            training_rsi=True,
        )
        observation, _ = env.reset(seed=20260819)
        action = torch.zeros((args.samples, 26), device=env.device)
        next_observation, _, _, _, _ = env.step(action)
        report = env.contract_report()
        rsi = env.rsi_report()
        payload = {
            "schema_version": "Stage16ContactStableC1RSIResetSanityV1",
            "status": "PASS"
            if _finite(observation)
            and _finite(next_observation)
            and _finite(env._robot.data.joint_pos)
            and _finite(env._object_170105.data.root_state_w)
            else "FAIL",
            "samples_frozen": args.samples,
            "training_reset": "uniform[0,320]",
            "curriculum_stage": "C1",
            "physics": report["gravity_friction_curriculum"],
            "reset_reference_index": report["ppo26d"]["rsi_curriculum"],
            "rsi": rsi,
            "checks": {
                "reset_observation_finite": _finite(observation),
                "one_step_observation_finite": _finite(next_observation),
                "controller_state_finite": _finite(env._robot.data.joint_pos),
                "object_state_finite": _finite(env._object_170105.data.root_state_w),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}))
        return 0 if payload["status"] == "PASS" else 2
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
