#!/usr/bin/env python3
"""Run bounded, batched true-PhysX P1 gravity diagnostics; never train PPO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.physical_stage import load_p1_rsi_acceptance_contract  # noqa: E402
from toporetarget.rl.rsi.contact_ready_v2 import (  # noqa: E402
    RSIStateSemanticClass,
    load_state_bank,
)

_CANDIDATE_CLASSES = {
    RSIStateSemanticClass.NEAR_CONTACT.value,
    RSIStateSemanticClass.CONTACT_READY.value,
    RSIStateSemanticClass.PERSISTENT_CONTACT.value,
    RSIStateSemanticClass.MANIPULATION.value,
    RSIStateSemanticClass.TERMINAL_HOLD.value,
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _chunks(values: Sequence[int], size: int) -> list[list[int]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _single_batch(
    *,
    clip: str,
    state_rows: list[dict[str, object]],
    reference_root: Path,
    control_steps: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Run up to sixteen state groups (64 PhysX environments) in one app."""

    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    _app = AppLauncher(headless=True).app
    env: Any = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        reset_indices = tuple(int(row["runtime_index"]) for row in state_rows for _ in range(4))
        cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=len(reset_indices), clip=clip, rsi=False, critical_dr=False
        )
        ppo_cfg.configure_stage16d_reference_kinematics_v2(cfg, reference_root=reference_root)
        cfg.sim.gravity = (0.0, 0.0, -9.81)
        cfg.object_170105.spawn.rigid_props.disable_gravity = False
        cfg.object_170650.spawn.rigid_props.disable_gravity = False
        cfg.evaluation_reset_reference_indices = reset_indices
        cfg.scene.lazy_sensor_update = True
        try:
            env = IsaacPPO26DReferenceTrackingEnv(cfg)
        except SystemExit as exc:
            raise RuntimeError(f"P1_RSI_V2_ENV_CONSTRUCTION_SYSTEM_EXIT:{exc.code}") from exc
        env.reset(seed=seed)
        state = env._state()
        initial_position = state["object_position_scene"].detach().cpu().numpy()
        initial_force = env._active_object_pair_force_matrix()
        already = (
            torch.linalg.vector_norm(initial_force, dim=-1)
            .any(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )
        position_before_contact = initial_position.copy()
        max_linear_before_contact = np.zeros(len(reset_indices), dtype=np.float64)
        max_angular_before_contact = np.zeros(len(reset_indices), dtype=np.float64)
        first_contact = np.full(len(reset_indices), -1, dtype=np.int64)
        contact_steps = np.zeros(len(reset_indices), dtype=np.int64)
        max_consecutive_contact = np.zeros(len(reset_indices), dtype=np.int64)
        consecutive_contact = np.zeros(len(reset_indices), dtype=np.int64)
        joint_limit_failure = np.zeros(len(reset_indices), dtype=bool)
        nonfinite = np.zeros(len(reset_indices), dtype=bool)
        catastrophic = np.zeros(len(reset_indices), dtype=bool)
        zero = torch.zeros((len(reset_indices), 26), device=env.device)
        for step in range(control_steps):
            _, _, terminated, _, extras = env.step(zero)
            report = extras["ppo26d"]
            contact = report["contact_any"].detach().cpu().numpy().astype(bool)
            current = env._state()
            position = current["object_position_scene"].detach().cpu().numpy()
            twist = current["object_twist_world"].detach().cpu().numpy()
            linear = np.linalg.norm(twist[:, :3], axis=-1)
            angular = np.linalg.norm(twist[:, 3:], axis=-1)
            no_contact_yet = first_contact < 0
            still_uncontacted = no_contact_yet & ~contact
            max_linear_before_contact[still_uncontacted] = np.maximum(
                max_linear_before_contact[still_uncontacted], linear[still_uncontacted]
            )
            max_angular_before_contact[still_uncontacted] = np.maximum(
                max_angular_before_contact[still_uncontacted], angular[still_uncontacted]
            )
            position_before_contact[still_uncontacted] = position[still_uncontacted]
            first_now = no_contact_yet & contact
            first_contact[first_now] = step
            contact_steps += contact.astype(np.int64)
            consecutive_contact = np.where(contact, consecutive_contact + 1, 0)
            max_consecutive_contact = np.maximum(max_consecutive_contact, consecutive_contact)
            joint_limit_failure |= ~report["joint_safe"].detach().cpu().numpy().astype(bool)
            nonfinite |= ~report["finite"].detach().cpu().numpy().astype(bool)
            catastrophic |= terminated.detach().cpu().numpy().astype(bool) & ~joint_limit_failure
        write_report = env.rollout_state_write_report()
        if (
            write_report["object_rollout_state_writes"]
            or write_report["wrist_root_state_writes_during_step"]
        ):
            raise RuntimeError("P1_RSI_V2_ROLLOUT_STATE_WRITE_FORBIDDEN")
        rows: list[dict[str, object]] = []
        for group, source in enumerate(state_rows):
            start, stop = group * 4, (group + 1) * 4
            for local, replica in enumerate(range(start, stop)):
                before = position_before_contact[replica] - initial_position[replica]
                rows.append(
                    {
                        "runtime_index": int(source["runtime_index"]),
                        "replica": local,
                        "semantic_class": str(source["semantic_class"]),
                        "source_expected_contact": bool(source["source_expected_contact"]),
                        "already_in_contact_at_reset": bool(already[replica]),
                        "expected_contact_at_reset": bool(source["source_expected_contact"]),
                        "first_actual_contact_time_s": None
                        if first_contact[replica] < 0
                        else float((first_contact[replica] + 1) * 0.05),
                        "object_displacement_before_contact_m": float(np.linalg.norm(before)),
                        "object_vertical_displacement_before_contact_m": float(before[2]),
                        "object_speed_before_contact_mps": float(
                            max_linear_before_contact[replica]
                        ),
                        "object_angular_speed_before_contact_radps": float(
                            max_angular_before_contact[replica]
                        ),
                        "contact_achieved": bool(contact_steps[replica] > 0),
                        "contact_persisted": bool(max_consecutive_contact[replica] >= 3),
                        "contact_persistence_control_steps": int(max_consecutive_contact[replica]),
                        "joint_limit_failure": bool(joint_limit_failure[replica]),
                        "object_drop": bool(before[2] < -0.01 and first_contact[replica] < 0),
                        "catastrophic_failure": bool(catastrophic[replica]),
                        "nonfinite": bool(nonfinite[replica]),
                        "hand_object_penetration": None,
                        "inter_finger_penetration": None,
                        "penetration_capture": "delegated_to_current_absolute_geometry_gates",
                        "guidance": 0,
                        "hidden_support": False,
                        "rollout_object_state_writes": 0,
                        "rollout_wrist_root_state_writes": 0,
                    }
                )
        return rows, {
            "object_rollout_state_writes": int(write_report["object_rollout_state_writes"]),
            "wrist_root_state_writes_during_step": int(
                write_report["wrist_root_state_writes_during_step"]
            ),
        }
    finally:
        # Each worker is a fresh short-lived process.  Isaac Sim 5.1 may exit
        # Python from teardown before main writes its durable receipt, so normal
        # process exit owns teardown after that receipt is committed.
        _ = env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--entry-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--state-groups-per-batch", type=int, default=16)
    args = parser.parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if args.batch_index < 0 or not 1 <= args.state_groups_per_batch <= 16:
        raise ValueError("P1_RSI_V2_BATCH_ARGUMENT_INVALID")
    acceptance = load_p1_rsi_acceptance_contract(args.entry_gate)
    bank = load_state_bank(args.state_bank)
    selected = [
        {
            "runtime_index": int(index),
            "semantic_class": str(semantic),
            "source_expected_contact": bool(expected),
        }
        for index, semantic, expected in zip(
            bank["runtime_index"],
            bank["semantic_class"],
            bank["source_expected_contact"],
            strict=True,
        )
        if str(semantic) in _CANDIDATE_CLASSES
    ]
    if not selected:
        raise ValueError("P1_RSI_V2_NO_DIAGNOSTIC_CANDIDATES")
    batches = _chunks(selected, args.state_groups_per_batch)
    if args.batch_index >= len(batches):
        raise ValueError("P1_RSI_V2_BATCH_INDEX_OUT_OF_RANGE")
    batch = batches[args.batch_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output,
        {
            "schema_version": "Stage16ContactReadyRSIV2GravityDiagnosticV1",
            "status": "INCOMPLETE_ISAAC_RUNTIME_EXIT_BEFORE_RESULT",
            "clip": args.clip,
            "target_gravity_world_mps2": [0.0, 0.0, -9.81],
            "candidate_state_count": len(selected),
            "batch_index": args.batch_index,
            "batch_count": len(batches),
            "batch_state_count": len(batch),
            "replicas_per_state": acceptance.replicas_per_state,
            "control_steps": acceptance.control_steps,
            "controller": "zero_policy_residual_plus_reference_following",
            "ppo_training": False,
            "guidance": 0,
        },
    )
    rows, write_counts = _single_batch(
        clip=args.clip,
        state_rows=batch,
        reference_root=args.reference_root,
        control_steps=acceptance.control_steps,
        seed=20260813 + args.batch_index,
    )
    _write_json(
        args.output,
        {
            "schema_version": "Stage16ContactReadyRSIV2GravityDiagnosticV1",
            "status": "COMPLETE",
            "clip": args.clip,
            "target_gravity_world_mps2": [0.0, 0.0, -9.81],
            "friction": "current_stage16d_nominal",
            "candidate_state_count": len(selected),
            "batch_index": args.batch_index,
            "batch_count": len(batches),
            "batch_state_count": len(batch),
            "replicas_per_state": acceptance.replicas_per_state,
            "control_steps": acceptance.control_steps,
            "controller": "zero_policy_residual_plus_reference_following",
            "ppo_training": False,
            "guidance": 0,
            "hidden_support": False,
            "reset_only_state_writes": write_counts,
            "rows": rows,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
