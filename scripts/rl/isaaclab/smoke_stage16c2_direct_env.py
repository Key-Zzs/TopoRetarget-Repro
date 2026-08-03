#!/usr/bin/env python3
"""Run the Stage 16-C.2 DirectRLEnv smoke on real Isaac GPU PhysX."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, choices=(1, 128), default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--clip-mode", choices=("one", "alternating", "balanced"), default="balanced"
    )
    parser.add_argument("--reset-mode", choices=("frame0", "uniform"), default="frame0")
    parser.add_argument(
        "--wrist-controller-mode",
        choices=(
            "wrist_impedance_v1",
            "computed_wrench_v2",
            "effective_dynamics_v3",
            "identified_inverse_wrench_v1",
            "finite_virtual_6d_wrist_actuator_v1",
        ),
        default="wrist_impedance_v1",
    )
    parser.add_argument(
        "--finite-virtual-profile",
        choices=("conservative", "nominal", "high_authority_bounded"),
        default="nominal",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16c2_c5_isaaclab/c2_smoke.json",
    )
    return parser.parse_args()


def gpu_snapshot() -> dict[str, int | None]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        util, memory = (int(value.strip()) for value in completed.stdout.splitlines()[0].split(","))
    except (IndexError, ValueError):
        return {"utilization_percent": None, "memory_used_mib": None}
    return {"utilization_percent": util, "memory_used_mib": memory}


def serialize(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    if os.environ.get("TOPORETARGET_DEBUG_FAULT_HANDLER") == "1":
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if not args.accept_eula:
        raise SystemExit("explicit --accept-eula is required for this licensed runtime process")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env import (
            IsaacWorldWristFingerDirectRLEnv,
        )
        from toporetarget.rl.environments.isaaclab_backend.world_wrist_direct_env_cfg import (
            IsaacWorldWristFingerDirectRLEnvCfg,
        )

        cfg = IsaacWorldWristFingerDirectRLEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.balanced_clip_assignment = args.clip_mode == "balanced"
        cfg.alternate_clip_on_reset = args.clip_mode == "alternating"
        cfg.reset_reference_index = args.reset_mode
        cfg.wrist_controller_mode = args.wrist_controller_mode
        if args.wrist_controller_mode == "finite_virtual_6d_wrist_actuator_v1":
            cfg.finite_virtual_wrist_profile = args.finite_virtual_profile
            cfg.finite_virtual_wrist_authority_enabled = True
        env = IsaacWorldWristFingerDirectRLEnv(cfg)
        observation, _ = env.reset(seed=20260802)
        policy = observation["policy"]
        if policy.shape != (args.num_envs, 764):
            raise RuntimeError(f"unexpected observation shape {tuple(policy.shape)}")
        if policy.device.type != "cuda":
            raise RuntimeError(f"C2 requires CUDA tensors, got {policy.device}")
        generator = torch.Generator(device=env.device)
        generator.manual_seed(20260802)
        gpu_before = gpu_snapshot()
        started = time.monotonic()
        reset_count = 0
        finite = True
        unique_action_rows = True
        primary_reason_counts: dict[str, int] = {}
        clip_step_counts = {clip_id: 0 for clip_id in env.reference_bank.clip_ids}
        for _ in range(args.steps):
            action = 0.05 * torch.rand((args.num_envs, 26), device=env.device, generator=generator)
            if args.num_envs == 1:
                action.zero_()
            else:
                unique_action_rows = (
                    unique_action_rows and torch.unique(action, dim=0).shape[0] == args.num_envs
                )
            observation, reward, terminated, timed_out, extras = env.step(action)
            policy = observation["policy"]
            finite = finite and bool(
                torch.isfinite(policy).all()
                and torch.isfinite(reward).all()
                and torch.isfinite(terminated.float()).all()
                and torch.isfinite(timed_out.float()).all()
            )
            reset_count += int((terminated | timed_out).sum().item())
            reasons = extras["stage16"]["primary_reason_code"]
            clips = extras["stage16"]["clip_index"]
            for clip_index, clip_id in enumerate(env.reference_bank.clip_ids):
                clip_step_counts[clip_id] += int((clips == clip_index).sum().item())
            for code in torch.unique(reasons).tolist():
                label = extras["stage16"]["termination_reasons"][int(code)]
                primary_reason_counts[label] = primary_reason_counts.get(label, 0) + int(
                    (reasons == code).sum().item()
                )
        elapsed = time.monotonic() - started
        contract = env.contract_report()
        passed = (
            finite
            and contract["object_rollout_state_writes"] == 0
            and contract["wrist_root_state_writes_during_step"] == 0
        )
        if passed:
            status = (
                "STAGE16C2_D6_PROFILE_REGRESSION_VALIDATED"
                if args.wrist_controller_mode == "finite_virtual_6d_wrist_actuator_v1"
                else "STAGE16C2_DIRECT_RL_ENV_VALIDATED"
            )
        else:
            status = "STAGE16C2_DIRECT_RL_ENV_PARTIAL"
        result = {
            "status": status,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "clip_mode": args.clip_mode,
            "reset_mode": args.reset_mode,
            "wrist_controller_mode": args.wrist_controller_mode,
            "finite_virtual_profile": (
                args.finite_virtual_profile
                if args.wrist_controller_mode == "finite_virtual_6d_wrist_actuator_v1"
                else None
            ),
            "qualification_scope": (
                "C.2 runtime contract only; does not select or validate a C.3 wrist profile."
                if args.wrist_controller_mode == "finite_virtual_6d_wrist_actuator_v1"
                else "C.2 DirectRLEnv runtime contract"
            ),
            "action_shape": [args.num_envs, 26],
            "observation_shape": list(policy.shape),
            "observation_device": str(policy.device),
            "reward_device": str(reward.device),
            "finite": finite,
            "resets": reset_count,
            "unique_action_rows": unique_action_rows,
            "primary_reason_counts": primary_reason_counts,
            "clip_step_counts": clip_step_counts,
            "elapsed_s": elapsed,
            "environment_steps_per_s": args.num_envs * args.steps / max(elapsed, 1.0e-9),
            "gpu_before": gpu_before,
            "gpu_after": gpu_snapshot(),
            "contract": serialize(contract),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
        return (
            0
            if result["status"]
            in {
                "STAGE16C2_DIRECT_RL_ENV_VALIDATED",
                "STAGE16C2_D6_PROFILE_REGRESSION_VALIDATED",
            }
            else 1
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
