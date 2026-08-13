#!/usr/bin/env python3
"""Run bounded zero-action RSI state-quality diagnostics in a fresh Isaac process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _indices(topology: dict[str, Any], *, frame_count: int = 321) -> list[int]:
    onset = topology["source_onset_window"]
    hold = topology["final_hold_window"]
    ranges = (
        (0, max(int(onset["start"]) - 1, 0), 12),
        (max(int(onset["start"]) - 16, 0), int(onset["start"]), 10),
        (int(onset["start"]), int(onset["end"]), 10),
        (int(onset["end"]) + 1, max(int(hold["start"]), int(onset["end"]) + 1), 10),
        (int(hold["end"]) + 1, 260, 10),
        (280, frame_count - 1, 12),
    )
    result: list[int] = []
    for start, end, count in ranges:
        if end < start:
            continue
        result.extend(np.linspace(start, min(end, frame_count - 1), count, dtype=np.int64).tolist())
    return sorted(set(result))[:64]


def _phase(index: int, topology: dict[str, Any]) -> str:
    onset = topology["source_onset_window"]
    hold = topology["final_hold_window"]
    if index >= 280:
        return "terminal"
    if index < int(onset["start"]) - 16:
        return "pre_contact"
    if index < int(onset["start"]):
        return "near_contact"
    if index <= int(onset["end"]):
        return "contact_onset"
    if index <= int(hold["end"]):
        return "persistent_contact"
    return "manipulation"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _reset_only_write_report(object_writes: int, wrist_writes: int) -> dict[str, object]:
    """Reject a diagnostic that wrote object or wrist-root state during rollout."""

    if object_writes != 0 or wrist_writes != 0:
        raise RuntimeError(
            "RSI state-quality diagnostic observed prohibited rollout state writes: "
            f"object={object_writes}, wrist_root={wrist_writes}"
        )
    return {
        "object_rollout": object_writes,
        "wrist_root_rollout": wrist_writes,
        "pass": True,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env: Any = None
    try:
        import torch

        from toporetarget.rl.environments.isaaclab_backend import (
            ppo26d_reference_tracking_env_cfg as ppo_cfg,
        )
        from toporetarget.rl.environments.isaaclab_backend.ppo26d_reference_tracking_env import (
            IsaacPPO26DReferenceTrackingEnv,
        )

        topology = _read_json(args.qualification)["contact_topology"]
        selected = _indices(topology)[: args.max_states]
        reset_indices = tuple(index for index in selected for _ in range(4))
        cfg = ppo_cfg.IsaacPPO26DReferenceTrackingEnvCfg()
        ppo_cfg.configure_stage16d_ppo26d(
            cfg, num_envs=len(reset_indices), clip=args.clip, rsi=False, critical_dr=False
        )
        if args.gravity:
            cfg.sim.gravity = (0.0, 0.0, -9.81)
            cfg.object_170105.spawn.rigid_props.disable_gravity = False
            cfg.object_170650.spawn.rigid_props.disable_gravity = False
        cfg.evaluation_reset_reference_indices = reset_indices
        cfg.scene.lazy_sensor_update = True
        env = IsaacPPO26DReferenceTrackingEnv(cfg)
        env.reset(seed=20260811)
        state = env._state()
        link = state["tracked_links_scene"]
        obj = state["object_axis_points_scene"]
        initial_gap = torch.cdist(link, obj).amin(dim=(1, 2)).detach().cpu().numpy()
        initial_position = state["object_position_scene"].detach().cpu().numpy()
        first_contact = np.full(len(reset_indices), -1, dtype=np.int64)
        any_contact = np.zeros(len(reset_indices), dtype=bool)
        terminal_position = initial_position.copy()
        terminal_twist = np.zeros((len(reset_indices), 6), dtype=np.float64)
        position_before_first_contact = initial_position.copy()
        zero = torch.zeros((len(reset_indices), 26), device=env.device)
        for step in range(20):
            _, _, _, _, extras = env.step(zero)
            contact = extras["ppo26d"]["contact_any"].detach().cpu().numpy().astype(bool)
            first_contact_now = (first_contact < 0) & contact
            first_contact[first_contact_now] = step
            any_contact |= contact
            measured = env._state()
            terminal_position = measured["object_position_scene"].detach().cpu().numpy()
            terminal_twist = measured["object_twist_world"].detach().cpu().numpy()
            position_before_first_contact[first_contact_now] = terminal_position[first_contact_now]
            position_before_first_contact[first_contact < 0] = terminal_position[first_contact < 0]
        object_writes = env.rollout_state_write_report()["object_rollout_state_writes"]
        wrist_writes = env.rollout_state_write_report()["wrist_root_state_writes_during_step"]
        rows: list[dict[str, object]] = []
        for group, frame in enumerate(selected):
            start, end = group * 4, (group + 1) * 4
            for replica in range(start, end):
                vertical = float(
                    position_before_first_contact[replica, 2] - initial_position[replica, 2]
                )
                displacement = float(
                    np.linalg.norm(
                        position_before_first_contact[replica] - initial_position[replica]
                    )
                )
                phase = _phase(frame, topology)
                if phase == "pre_contact":
                    if displacement > 0.005:
                        label = "PRE_CONTACT_UNSUPPORTED"
                    elif args.gravity:
                        label = "AMBIGUOUS"
                    else:
                        label = "PRE_CONTACT_STABLE_UNDER_ZERO_G"
                elif phase == "near_contact":
                    label = "NEAR_OBJECT"
                elif phase == "contact_onset" and any_contact[replica]:
                    label = "CONTACT_READY"
                elif phase in {"persistent_contact", "manipulation"} and any_contact[replica]:
                    label = "PERSISTENT_CONTACT"
                elif phase == "terminal":
                    label = "TERMINAL_HOLD"
                else:
                    label = "AMBIGUOUS"
                rows.append(
                    {
                        "reference_index": frame,
                        "replica": replica - start,
                        "reference_phase": phase,
                        "classification": label,
                        "initial_hand_object_gap_m": float(initial_gap[replica]),
                        "first_actual_contact_step": None
                        if first_contact[replica] < 0
                        else int(first_contact[replica]),
                        "object_vertical_displacement_before_contact_m": vertical,
                        "object_total_displacement_before_contact_m": displacement,
                        "contact_achieved": bool(any_contact[replica]),
                        "reference_expected_contact": phase
                        in {"contact_onset", "persistent_contact", "manipulation", "terminal"},
                        "object_linear_speed_mps": float(
                            np.linalg.norm(terminal_twist[replica, :3])
                        ),
                        "object_angular_speed_radps": float(
                            np.linalg.norm(terminal_twist[replica, 3:])
                        ),
                        "catastrophic_failure": bool(
                            not np.isfinite(terminal_twist[replica]).all()
                        ),
                        "penetration": "NOT_CAPTURED_IN_BOUNDED_RSI_DIAGNOSTIC",
                    }
                )
        result: dict[str, object] = {
            "schema_version": "RSIStateQualityAuditV1",
            "clip": args.clip,
            "gravity": [0.0, 0.0, -9.81] if args.gravity else [0.0, 0.0, 0.0],
            "replicas_per_state": 4,
            "control_steps": 20,
            "zero_residual_action": True,
            "reset_only_state_writes": _reset_only_write_report(object_writes, wrist_writes),
            "rows": rows,
        }
        result["status"] = "COMPLETE"
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({"output": str(args.output.resolve()), "row_count": len(rows)}), flush=True
        )
        return result
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--gravity", action="store_true")
    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="Bounded debug override; the formal audit uses the default stratified set.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.accept_eula:
        raise ValueError("--accept-eula is required")
    if not 1 <= args.max_states <= 64:
        raise ValueError("--max-states must be in [1, 64]")
    topology = _read_json(args.qualification)["contact_topology"]
    selected = _indices(topology)[: args.max_states]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Isaac can terminate a process during scene construction without returning
    # a Python exception.  Persist the requested bounded experiment first so a
    # missing completed report cannot be mistaken for a passing dynamic audit.
    preflight = {
        "schema_version": "RSIStateQualityAuditV1",
        "status": "INCOMPLETE_ISAAC_RUNTIME_EXIT_BEFORE_RESULT",
        "clip": args.clip,
        "gravity": [0.0, 0.0, -9.81] if args.gravity else [0.0, 0.0, 0.0],
        "selected_reference_indices": selected,
        "replicas_per_state": 4,
        "requested_rows": len(selected) * 4,
        "control_steps": 20,
        "zero_residual_action": True,
        "note": "Replaced only if the fresh Isaac process returns a completed rows payload.",
    }
    args.output.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
