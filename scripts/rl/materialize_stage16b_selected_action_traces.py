#!/usr/bin/env python3
"""Reproduce and persist the frozen MuJoCo H5/H10 traces required by C3-4."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/rl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"C3_MUJOCO_TRACE_REFUSES_OVERWRITE: {args.output_root}")
    args.output_root.mkdir(parents=True)

    import qualify_stage16_world_wrist as qualification

    from toporetarget.rl.environments.world_wrist_backend import (
        WristFingerActionScaleV1,
        WristImpedanceProfileV1,
    )

    legacy_commit = "8405ed8^"
    legacy_source = subprocess.run(
        ["git", "show", f"{legacy_commit}:src/toporetarget/rl/world_wrist_oracle.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    legacy_module = types.ModuleType("toporetarget.rl._stage16b_fixed_horizon_oracle_authority")
    legacy_module.__package__ = "toporetarget.rl"
    sys.modules[legacy_module.__name__] = legacy_module
    exec(
        compile(legacy_source, f"git:{legacy_commit}:world_wrist_oracle.py", "exec"),
        legacy_module.__dict__,
    )
    qualification.WorldWristFingerObjectAwareOracle = (
        legacy_module.WorldWristFingerObjectAwareOracle
    )
    impedance = WristImpedanceProfileV1(
        translation_stiffness_npm=250.0,
        translation_damping_ratio=1.0,
        rotation_stiffness_nmprad=2.0,
        rotation_damping_ratio=0.5,
        force_limit_n=25.0,
        torque_limit_nm=1.5,
    )
    action_scale = WristFingerActionScaleV1(0.005, float(np.deg2rad(2.5)), 0.05)
    oracle_config = legacy_module.ContactAwareMPCConfig(
        population=32,
        iterations=3,
        elite_count=8,
        initial_std=0.35,
        minimum_std=0.05,
        seed=20260801,
    )
    oracle_config.validate()
    specifications = (
        ("hocap_170105", 5, 0),
        ("hocap_170650", 10, 1),
    )
    records = []
    for clip, horizon, clip_index in specifications:
        reference = (
            REPO_ROOT
            / ".local/stage16_reference_tracking_ppo/world_wrist_references"
            / f"{clip}.world_wrist.stage16.npz"
        )
        mesh = (
            REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_objects" / f"{clip}.obj"
        )
        backend = qualification._make_backend(
            reference_path=reference,
            mesh_path=mesh,
            scene_root=args.output_root / "scenes" / f"{clip}_h{horizon}",
            impedance=impedance,
            action_scale=action_scale,
            seed=20260801 + 400 + horizon + clip_index,
        )
        actions: list[np.ndarray] = []
        episode = qualification._episode(
            backend,
            policy="oracle",
            horizon=horizon,
            action_log=actions,
            oracle_config=oracle_config,
        )
        action_array = np.asarray(actions, dtype=np.float64)
        if action_array.shape != (40, 26):
            raise RuntimeError(
                "C3_MUJOCO_TRACE_INCOMPLETE: "
                f"clip={clip} H={horizon} shape={action_array.shape} episode={asdict(episode)}"
            )
        output = args.output_root / f"{clip}_h{horizon}_actions.npz"
        np.savez_compressed(output, actions=action_array)
        records.append(
            {
                "clip": clip,
                "horizon": horizon,
                "action_trace": str(output.resolve()),
                "action_trace_sha256": _sha256(output),
                "actions_shape": list(action_array.shape),
                "episode": asdict(episode),
            }
        )
    report = {
        "status": "C3_MUJOCO_SELECTED_ACTION_TRACES_MATERIALIZED",
        "source_protocol": (
            "contact_mpc_formal_selected_20260801; shared 32x3 CEM and selected global profiles"
        ),
        "fixed_horizon_oracle_authority": {
            "git_revision": legacy_commit,
            "source_sha256": hashlib.sha256(legacy_source.encode()).hexdigest(),
            "reason": "pre-adaptive H5/H10 implementation that produced the frozen formal report",
        },
        "impedance": impedance.as_dict(),
        "action_scale": action_scale.as_dict(),
        "oracle_config": asdict(oracle_config),
        "traces": records,
    }
    report_path = args.output_root / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
