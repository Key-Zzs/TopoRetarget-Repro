#!/usr/bin/env python3
"""Measure C.5A GPU PhysX baseline noise before any snapshot/restore test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
CONTACT_WORKERS = (
    REPO_ROOT / ".local/reports/stage16c3r5_reference_retiming_c4/.contact_causality_scale8_workers"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=33)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-output", type=Path, required=True)
    return parser.parse_args()


def _write(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"STAGE16C5A_NOISE_REFUSES_OVERWRITE: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _records(clip: str) -> list[dict[str, Any]]:
    path = CONTACT_WORKERS / f"{clip}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("contact_records")
    if not isinstance(rows, list):
        raise ValueError(f"C3 contact worker malformed: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _norm(record: dict[str, Any]) -> float:
    values = record.get("net_contact_force_world_on_object_n", [])
    return sum(float(value) ** 2 for value in values) ** 0.5 if isinstance(values, list) else 0.0


def _frame_table() -> dict[str, object]:
    clips = []
    for clip in ("hocap_170105", "hocap_170650"):
        contacts = [row for row in _records(clip) if int(row.get("contact_count", 0)) > 0]
        if not contacts:
            raise RuntimeError(f"STAGE16C5A_CONTACT_FRAME_MISSING: {clip}")
        first = contacts[0]
        strongest = max(contacts, key=_norm)
        last = contacts[-1]
        onset = int(first["reference_index"])
        clips.append(
            {
                "clip": clip,
                "frames": {
                    "pre_contact": 0,
                    "contact_onset": onset,
                    "sustained_contact": int(strongest["reference_index"]),
                    "post_contact": min(320, int(last["reference_index"]) + 1),
                },
                "selection": {
                    "Fpre": "frame 0 is required initial pre-contact state",
                    "Fon": "first object-centric contact trace record",
                    "Fcontact": "maximum aggregate contact-force trace record",
                    "Fpost": "last contact trace record plus one, clamped to runtime boundary",
                },
            }
        )
    return {
        "version": "stage16c5_replication_test_frames_v1",
        "source": str(CONTACT_WORKERS.relative_to(REPO_ROOT)),
        "clips": clips,
        "runtime_index_range": [0, 320],
    }


def _set_clip_frame_zero(env: Any, clip_index: int) -> None:
    import torch

    ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    original = env.cfg.balanced_clip_assignment
    env.cfg.balanced_clip_assignment = False
    try:
        env._clip_index[ids] = clip_index
        env._reset_idx(ids)
    finally:
        env.cfg.balanced_clip_assignment = original


def _metrics(env: Any) -> dict[str, list[float]]:
    import torch

    from toporetarget.rl.isaaclab_oracle.metrics import state_differences
    from toporetarget.rl.isaaclab_oracle.runtime import state_view

    view = state_view(env, torch.arange(env.num_envs, device=env.device))
    first = {
        name: value[:1].expand_as(value[1:])
        for name, value in view.items()
        if name not in {"reference_index", "reason_codes"}
    }
    rest = {
        name: value[1:]
        for name, value in view.items()
        if name not in {"reference_index", "reason_codes"}
    }
    differences = state_differences(first, rest)
    reward = env._last_reward_terms["total"]
    differences["reward"] = float((reward[1:] - reward[:1]).abs().amax().detach().cpu())
    return {name: [value] for name, value in differences.items()}


def main() -> int:
    args = parse_args()
    if not args.accept_eula:
        raise SystemExit("--accept-eula is required")
    if args.trials < 20 or args.num_envs < 32:
        raise SystemExit("C.5A baseline requires at least 20 trials and 32 environments")
    if args.output.exists() or args.frames_output.exists():
        raise FileExistsError("STAGE16C5A_NOISE_REFUSES_OVERWRITE")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    frames = _frame_table()
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    env = None
    try:
        import torch

        from toporetarget.rl.isaaclab_oracle.history_replay import raw_control_step
        from toporetarget.rl.isaaclab_oracle.runtime import make_stage16c5_env
        from toporetarget.rl.isaaclab_oracle.tolerance import freeze_tolerances

        env = make_stage16c5_env(num_envs=args.num_envs)
        results: list[dict[str, object]] = []
        aggregate: dict[str, list[float]] = {}
        for clip_index, row in enumerate(frames["clips"]):
            assert isinstance(row, dict)
            frame_map = row["frames"]
            assert isinstance(frame_map, dict)
            for phase, frame_value in frame_map.items():
                frame = int(frame_value)
                samples: dict[str, list[float]] = {}
                for _ in range(args.trials):
                    _set_clip_frame_zero(env, clip_index)
                    zero = torch.zeros((env.num_envs, 26), device=env.device)
                    for _step in range(frame + 1):
                        raw_control_step(env, zero)
                    trial = _metrics(env)
                    for name, values in trial.items():
                        samples.setdefault(name, []).extend(values)
                        aggregate.setdefault(name, []).extend(values)
                results.append(
                    {
                        "clip": row["clip"],
                        "phase": phase,
                        "frame": frame,
                        "samples": samples,
                        "tolerances": freeze_tolerances(samples),
                    }
                )
        frozen = freeze_tolerances(aggregate)
        report = {
            "status": frozen["status"],
            "version": "replication_noise_floor_v1",
            "trials": args.trials,
            "num_envs": args.num_envs,
            "cuda_device": str(env.device),
            "phases": results,
            "global_tolerances": frozen,
            "no_snapshot_or_restore": True,
        }
        _write(args.frames_output, frames)
        _write(args.output, report)
        print(json.dumps({"status": report["status"], "trials": args.trials}))
        return 0 if report["status"] == "REPLICATION_TOLERANCES_FROZEN" else 2
    finally:
        if env is not None:
            env.close()
            env.sim.clear_all_callbacks()
            env.sim.clear_instance()
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
