#!/usr/bin/env python3
"""Run one isolated Stage 16-C.5A O0 candidate-pool capacity smoke."""

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
    parser.add_argument("--candidate-envs", type=int, choices=(1, 32, 96, 144), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _write(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_O0_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_state(state: Any) -> bool:
    return all(
        (not value.is_floating_point()) or bool(value.isfinite().all())
        for value in state.tensors.values()
    )


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.output.exists():
        raise FileExistsError(f"STAGE16C5A_O0_REFUSES_OVERWRITE: {args.output}")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.candidate_pool import PhysXOracleCandidatePoolV1
        from toporetarget.rl.isaaclab_oracle.candidate_state import capture_candidate_state
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env, state_view

        env = make_stage16c5_env(num_envs=1 + args.candidate_envs)
        population = (
            32
            if args.candidate_envs == 96
            else 48
            if args.candidate_envs == 144
            else args.candidate_envs
        )
        pool = PhysXOracleCandidatePoolV1(
            env,
            candidate_count=args.candidate_envs,
            population_per_horizon=population,
        )
        layout = pool.validate_layout()
        before_execution = capture_candidate_state(env, [0])
        state = pool.replicate_execution_state(before_execution)
        candidate_state = capture_candidate_state(env, pool.candidate_ids)
        candidate_view = state_view(env, pool.candidate_ids)
        execution_view = state_view(env, [0])
        equal_joint_state = bool(
            torch.allclose(
                candidate_view["robot_joint_pos"],
                execution_view["robot_joint_pos"].expand_as(candidate_view["robot_joint_pos"]),
                atol=0.0,
                rtol=0.0,
            )
        )
        equal_clip_and_index = bool(
            torch.eq(candidate_view["reference_index"], execution_view["reference_index"][0]).all()
            and torch.eq(
                candidate_state.tensors["clip_index"], state.tensors["clip_index"][0]
            ).all()
        )
        isolated_candidate = int(pool.candidate_ids[0].item())
        peer_candidate = int(pool.candidate_ids[-1].item())
        peer_before = (
            None if args.candidate_envs == 1 else capture_candidate_state(env, [peer_candidate])
        )
        execution_before_subset_reset = capture_candidate_state(env, [0])
        env._reset_idx(torch.tensor([isolated_candidate], device=env.device))
        pool.write_audit.record(
            category="reset",
            operation="o0_subset_reset",
            env_ids=torch.tensor([isolated_candidate], device=env.device),
            tensor_names=["candidate_reset"],
        )
        peer_after = (
            None if args.candidate_envs == 1 else capture_candidate_state(env, [peer_candidate])
        )
        execution_after_subset_reset = capture_candidate_state(env, [0])
        peer_unchanged = (
            True
            if peer_before is None or peer_after is None
            else all(
                torch.equal(peer_before.tensors[name], peer_after.tensors[name])
                for name in peer_before.tensors
            )
        )
        execution_unchanged = all(
            torch.equal(
                execution_before_subset_reset.tensors[name],
                execution_after_subset_reset.tensors[name],
            )
            for name in execution_before_subset_reset.tensors
        )
        contract = env.contract_report()
        passes = {
            "allocation": bool(layout["unique_env_ids"]),
            "unique_origins": bool(layout["unique_origins"]),
            "cuda": str(env.device).startswith("cuda"),
            "state_tensor_shapes": candidate_state.env_count == args.candidate_envs,
            "finite": _finite_state(candidate_state),
            "clip_and_reference_preserved": equal_clip_and_index,
            "no_env_aliasing": equal_joint_state
            and peer_unchanged
            and bool(layout["candidate_execution_disjoint"]),
            "subset_reset_isolated": peer_unchanged and execution_unchanged,
            "candidate_setup_writes_recorded": pool.write_audit.candidate_setup_writes > 0,
            "execution_rollout_writes_zero": (
                contract["wrist_root_state_writes_during_step"] == 0
                and contract["object_rollout_state_writes"] == 0
                and pool.write_audit.formal_execution_rollout_writes == 0
            ),
        }
        report: dict[str, object] = {
            "status": (
                "STAGE16C5A_O0_CANDIDATE_POOL_VALIDATED"
                if all(passes.values())
                else "STAGE16C5A_O0_CANDIDATE_POOL_PARTIAL"
            ),
            "candidate_envs": args.candidate_envs,
            "horizons_represented": list(pool.layout.horizons),
            "pool": layout,
            "state": state.as_dict(),
            "passes": passes,
            "write_audit": pool.write_audit.as_dict(),
            "direct_env_write_contract": {
                "wrist_root_state_writes_during_step": contract[
                    "wrist_root_state_writes_during_step"
                ],
                "object_rollout_state_writes": contract["object_rollout_state_writes"],
            },
            "clean_exit": True,
        }
        _write(args.output, report)
        print(json.dumps({"status": report["status"], "candidate_envs": args.candidate_envs}))
        return 0 if all(passes.values()) else 2
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
