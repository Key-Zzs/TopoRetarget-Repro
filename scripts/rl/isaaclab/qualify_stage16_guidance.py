#!/usr/bin/env python3
"""Run the bounded G2 counterfactual for explicit object guidance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.physics.guidance import ObjectGuidanceContractV1  # noqa: E402

DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16_guidance_g0_g5/g2"
REFERENCE_ROOT = REPO_ROOT / ".local/frozen_baselines/reference_kinematics_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--steps", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def close_env(env: Any) -> None:
    env.close()
    env.sim.clear_all_callbacks()
    env.sim.clear_instance()


def contract(mode: str) -> ObjectGuidanceContractV1:
    config = yaml.safe_load((REPO_ROOT / "configs/physics/object_guidance_v1.yaml").read_text())
    values = dict(config["guidance"])
    values["mode"] = mode
    return ObjectGuidanceContractV1(**values)


def run_case(
    *, clip: str, frame: int, mode: str, steps: int, reference_root: Path
) -> dict[str, Any]:
    from toporetarget.rl.environments.isaaclab_backend import (
        ppo26d_reference_tracking_env_cfg as ppo26d_cfg,
    )
    from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
        IsaacPPO26DReferenceTrackingEnv,
    )

    cfg = ppo26d_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
    ppo26d_cfg.configure_stage16d_ppo26d(
        cfg,
        num_envs=1,
        clip=clip,
        rsi=True,
        critical_dr=False,
        curriculum_reference_indices=(frame,),
        curriculum_reference_probabilities=(1.0,),
        curriculum_phase="C0",
    )
    ppo26d_cfg.configure_stage16d_reference_kinematics_v2(cfg, reference_root=reference_root)
    selected = contract(mode)
    ppo26d_cfg.configure_stage16d_object_guidance(cfg, contract=selected)
    env = IsaacPPO26DReferenceTrackingEnv(cfg)
    try:
        env.reset(seed=20260816)
        action = torch.zeros((1, 26), device=env.device)
        before = env._state()["object_twist_world"].clone()
        force_rows: list[torch.Tensor] = []
        error_rows: list[torch.Tensor] = []
        active_rows: list[torch.Tensor] = []
        for _ in range(steps):
            observation, reward, _, _, _ = env.step(action)
            wrench = env._last_object_guidance
            if wrench is None:
                raise RuntimeError("GUIDANCE_G2_WRENCH_MISSING")
            if not bool(
                torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all()
            ):
                raise FloatingPointError("GUIDANCE_G2_NONFINITE_ENVIRONMENT")
            force_rows.append(wrench.force_world.detach().clone())
            error_rows.append(wrench.position_error_world.detach().clone())
            active_rows.append(wrench.guidance_active.detach().clone())
        after = env._state()["object_twist_world"].clone()
        forces = torch.cat(force_rows)
        errors = torch.cat(error_rows)
        wrench = env._last_object_guidance
        assert wrench is not None
        force_limits = wrench.force_limit_n.reshape(-1)
        torque_limits = wrench.torque_limit_nm.reshape(-1)
        force_norm = torch.linalg.vector_norm(forces, dim=-1)
        torque_norm = torch.linalg.vector_norm(wrench.torque_world, dim=-1)
        direction = (forces * errors).sum(dim=-1)
        writes = env.rollout_state_write_report()
        return {
            "clip": clip,
            "reference_start_index": frame,
            "mode": mode,
            "steps": steps,
            "finite": True,
            "guidance_active_any": bool(torch.cat(active_rows).any()),
            "force_max_n": float(force_norm.max()),
            "force_limit_n": float(force_limits.max()),
            "torque_max_nm": float(torque_norm.max()),
            "torque_limit_nm": float(torque_limits.max()),
            "force_bound_pass": bool((force_norm <= force_limits.max() + 1.0e-6).all()),
            "torque_bound_pass": bool((torque_norm <= torque_limits.max() + 1.0e-6).all()),
            "force_alignment_min": float(direction.min()),
            "object_velocity_delta_mps": float(
                torch.linalg.vector_norm(after[:, :3] - before[:, :3])
            ),
            "object_angular_velocity_delta_radps": float(
                torch.linalg.vector_norm(after[:, 3:] - before[:, 3:])
            ),
            "rollout_state_writes": writes,
            "contract": env.contract_report()["object_guidance"],
        }
    finally:
        close_env(env)


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.steps < 1:
        raise ValueError("GUIDANCE_G2_STEPS_INVALID")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    reference_root = args.reference_root.resolve()
    launcher = AppLauncher(headless=True)
    app = launcher.app
    try:
        rows: list[dict[str, Any]] = []
        for clip in ("hocap_170105", "hocap_170650"):
            for phase, frame in (("early", 0), ("contact_rich", 160), ("late", 300)):
                none = run_case(
                    clip=clip,
                    frame=frame,
                    mode="none",
                    steps=args.steps,
                    reference_root=reference_root,
                )
                guided = run_case(
                    clip=clip,
                    frame=frame,
                    mode="reference_wrench_v1",
                    steps=args.steps,
                    reference_root=reference_root,
                )
                pair = {"phase": phase, "none": none, "guided": guided}
                rows.append(pair)
                write_json(args.output_root / "counterfactuals" / f"{clip}_{phase}.json", pair)
        guided_rows = [row["guided"] for row in rows]
        qualified = all(
            row["finite"]
            and row["force_bound_pass"]
            and row["torque_bound_pass"]
            and row["rollout_state_writes"]["object_rollout_state_writes"] == 0
            and row["rollout_state_writes"]["wrist_root_state_writes_during_step"] == 0
            for row in guided_rows
        )
        qualification: dict[str, Any] = {
            "schema_version": "Stage16GuidanceG2QualificationV1",
            "status": "GUIDANCE_G2_QUALIFIED" if qualified else "GUIDANCE_G2_FAILED",
            "counterfactual_count": len(rows),
            "physics_application": "instantaneous_world_wrench_before_physx_step",
            "no_hidden_attachment": True,
            "results": rows,
        }
        write_json(args.output_root / "qualification.json", qualification)
        print(json.dumps({"status": qualification["status"], "output_root": str(args.output_root)}))
    except Exception as error:
        failure = {
            "schema_version": "Stage16GuidanceG2TechnicalFailureV1",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.output_root / "technical_failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        raise
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
