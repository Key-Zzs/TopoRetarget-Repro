#!/usr/bin/env python3
"""Assemble evidence-backed Stage 16-D.5 PPO-26D local handoff artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--clip", default="hocap_170650")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def host_gpu_fields(payload: dict[str, Any]) -> dict[str, str]:
    query = str(payload["query"]).strip().splitlines()
    if not query:
        raise ValueError("host GPU query is empty")
    values = [value.strip() for value in query[0].split(",")]
    if len(values) != 11:
        raise ValueError("host GPU query field count is invalid")
    names = (
        "index",
        "name",
        "uuid",
        "driver_version",
        "total_vram_mib",
        "used_vram_mib",
        "free_vram_mib",
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "temperature_c",
        "power_w",
    )
    return dict(zip(names, values, strict=True))


def trace_evidence(trace_path: Path) -> dict[str, Any]:
    with np.load(trace_path, allow_pickle=False) as archive:
        contact_force = np.asarray(archive["contact_force_world"], dtype=np.float64)
        contact_pairs = np.asarray(archive["contact_pair_presence"], dtype=bool)
        object_twist = np.asarray(archive["object_twist"], dtype=np.float64)
        hand_pose = np.asarray(archive["hand_collision_body_pose"], dtype=np.float64)
        terminated = np.asarray(archive["terminated"], dtype=bool)
        timed_out = np.asarray(archive["timed_out"], dtype=bool)
        frames = int(contact_force.shape[0])
        expected_hand_shape = (frames, contact_pairs.shape[-1], 7)
        if (
            contact_force.shape != (frames, 3)
            or contact_pairs.shape[0] != frames
            or hand_pose.shape != expected_hand_shape
        ):
            raise ValueError("PPO trace contact shapes are invalid")
        hand_quaternion_norm = np.linalg.vector_norm(hand_pose[..., 3:7], axis=-1)
        if not np.isfinite(hand_pose).all() or np.any(hand_quaternion_norm < 1.0e-8):
            raise ValueError("PPO trace contains an invalid hand collision-body pose")
        force_norm = np.linalg.vector_norm(contact_force, axis=-1)
        contact_mask = contact_pairs.any(axis=-1) | (force_norm > 1.0e-6)
        contact_frames = np.flatnonzero(contact_mask)
        if len(contact_frames):
            breaks = np.flatnonzero(np.diff(contact_frames) > 1)
            onset = int(contact_frames[breaks[-1] + 1] if len(breaks) else contact_frames[0])
            end = int(contact_frames[-1])
        else:
            onset = None
            end = None
        terminal_start = max(0, frames - 20)
        peak_force_frame = int(np.argmax(force_norm))
        terminal_twist = np.linalg.vector_norm(object_twist[terminal_start:, :3], axis=-1)
        return {
            "trace": str(trace_path.resolve()),
            "frames": frames,
            "action_dimension": int(np.asarray(archive["action"]).shape[-1]),
            "hand_collision_proxy_count": int(contact_pairs.shape[-1]),
            "hand_collision_body_quaternion_min_norm": float(hand_quaternion_norm.min()),
            "finite": bool(
                all(
                    np.isfinite(np.asarray(archive[name])).all()
                    for name in ("object_pose", "object_twist", "action", "reward_total")
                )
            ),
            "last_meaningful_contact_onset_frame": onset,
            "last_meaningful_contact_end_frame": end,
            "terminal_window_start_frame": terminal_start,
            "terminal_window_end_exclusive": frames,
            "peak_contact_force_frame": peak_force_frame,
            "peak_contact_force_n": float(force_norm[peak_force_frame]),
            "lowest_terminal_linear_twist_frame": int(terminal_start + np.argmin(terminal_twist)),
            "terminated_frames": [int(value) for value in np.flatnonzero(terminated)],
            "timed_out_frames": [int(value) for value in np.flatnonzero(timed_out)],
        }


def git_commits() -> list[dict[str, str]]:
    result = subprocess.run(
        ["git", "log", "--format=%H%x1f%s", "-12"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [
        {"commit": row.split("\x1f", maxsplit=1)[0], "subject": row.split("\x1f", maxsplit=1)[1]}
        for row in result.stdout.splitlines()
        if "\x1f" in row
    ]


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    clip_root = root / args.clip
    host_probe = read_json(root / "gpu" / "host_gpu_probe.json")
    capacity = read_json(root / "gpu" / "ppo_gpu_capacity_benchmark.json")
    selected = read_json(root / "gpu" / "selected_capacity.json")
    trainability = read_json(root / "trainability" / "trainability_gate.json")
    training = read_json(clip_root / "l0_training.json")
    evaluation = read_json(clip_root / "ppo_l0_eval_qualification.json")
    replay = read_json(root / "replay_validation.json")
    tests = read_json(root / "tests.json")
    trace = trace_evidence(Path(str(evaluation["trace"])))
    commits = git_commits()
    gpu = host_gpu_fields(host_probe)
    resource_usage = {
        "schema_version": "Stage16DPPO26DResourceUsageV1",
        "host_gpu": gpu,
        "capacity_rows": capacity["rows"],
        "selected_capacity": selected,
        "training": {
            "selected_num_envs": training["samples_per_iteration"] // 40,
            "iterations": training["iterations"],
            "cumulative_samples": training["cumulative_samples"],
        },
    }
    failure_transitions = [
        {
            "failure_class": "PPO26D_BENCHMARK_JSON_SERIALIZATION",
            "attempt": 1,
            "resolved": True,
            "evidence": (
                "transient CUDA last_policy_observation removed before benchmark JSON output"
            ),
        },
        {
            "failure_class": "CAPACITY_METRIC_COMPLETENESS",
            "attempt": 1,
            "resolved": True,
            "evidence": (
                "final six-size sweep records B0/B1/B2/B3 timings, memory, and NaN/Inf counts"
            ),
        },
        {
            "failure_class": "PPO26D_TRACE_ARTICULATION_CALLBACK_LIFECYCLE",
            "attempt": 1,
            "resolved": True,
            "evidence": (
                "post-physics GPU wrist/finger capture plus one post-rollout host export and "
                "offline FK reconstruction produced a finite 21-body replay trace with no "
                "zero quaternion"
            ),
        },
    ]
    summary = {
        "schema_version": "Stage16DPPO26DFinalSummaryV1",
        "status": training["status"],
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "clip": args.clip,
        "gpu": gpu,
        "selected_num_envs": selected["selected_num_envs"],
        "trainability": trainability,
        "training": training,
        "evaluation": evaluation,
        "trace": trace,
        "replay": replay,
        "tests": tests,
        "git_commits": commits,
        "git_delivery": {
            "NEW_BRANCH_CREATED": "NO",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            "MAIN_MERGED": "NO",
            "TAG_CREATED": "NO",
            "RELEASE_CREATED": "NO",
        },
    }
    write_json(root / "resource_usage.json", resource_usage)
    write_json(root / "git_commits.json", {"commits": commits})
    write_text(
        root / "failure_transitions.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failure_transitions),
    )
    write_json(root / "final_summary.json", summary)
    table_format = (
        "| {num_envs} | {samples_per_s:.1f} | {ppo_update_ok} | {peak_vram_mib:.1f} | "
        "{free_vram_mib:.1f} | {clean_exit} |"
    )
    table = "\n".join(table_format.format(**row) for row in capacity["rows"])
    contact_summary = (
        f"Last contact: `{trace['last_meaningful_contact_onset_frame']}` to "
        f"`{trace['last_meaningful_contact_end_frame']}`; terminal window: "
        f"`{trace['terminal_window_start_frame']}:{trace['terminal_window_end_exclusive']}`; "
        f"peak force frame: `{trace['peak_contact_force_frame']}`."
    )
    markdown = f"""# Stage 16-D PPO-26D L0 Handoff

Status: `{training["status"]}`

Selected environments: `{selected["selected_num_envs"]}`

Trace: `{trace["trace"]}` ({trace["frames"]} frames, action dim {trace["action_dimension"]})

Collision-body pose source: `{evaluation["hand_collision_body_pose"]}`
(`min quaternion norm = {trace["hand_collision_body_quaternion_min_norm"]:.8f}`)

{contact_summary}

| envs | samples/s | PPO update | peak VRAM MiB | free VRAM MiB | clean exit |
|---:|---:|---|---:|---:|---|
{table}

Checkpoint: `{training["l0_checkpoint"]}`

Replay receipt: `{root / "replay_validation.json"}`
"""
    write_text(root / "final_summary.md", markdown)
    write_text(root / "handoff.md", markdown)
    print(json.dumps({"status": summary["status"], "trace": trace["trace"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
