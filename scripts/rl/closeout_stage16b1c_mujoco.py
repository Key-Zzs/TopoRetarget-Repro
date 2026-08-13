#!/usr/bin/env python3
"""Freeze the bounded Stage-16B.1c MuJoCo closeout without new CEM work."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.rl.qualify_stage16b_adaptive_oracle import _make_backend, _metrics  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _archive_existing_top_level(report_root: Path, archive: Path) -> None:
    prior = archive / "prior_top_level"
    prior.mkdir(parents=True)
    for path in sorted(report_root.glob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md"}:
            if path.stat().st_size <= 256_000:
                shutil.copy2(path, prior / path.name)


def _capture_stage16c_preflight() -> None:
    root = REPO / ".local/reports/stage16c_isaaclab_platform/preflight"
    if root.exists():
        return
    root.mkdir(parents=True)
    commands = {
        "git_status_before.txt": ("status", "--short", "--untracked-files=all"),
        "unstaged.patch": ("diff",),
        "staged.patch": ("diff", "--cached"),
        "branches.txt": ("branch", "-a"),
    }
    for name, arguments in commands.items():
        (root / name).write_text(_git(*arguments) + "\n", encoding="utf-8")
    worktrees = {"git_worktree_list_porcelain": _git("worktree", "list", "--porcelain")}
    _write_json(root / "worktrees.json", worktrees)
    status_paths = [
        line[3:]
        for line in (root / "git_status_before.txt").read_text(encoding="utf-8").splitlines()
        if len(line) > 3
    ]
    _write_json(
        root / "ownership.json",
        {
            "classification": "CURRENT_STAGE16B1C_CLOSEOUT_WIP",
            "owned_paths": status_paths,
            "unrelated_user_modifications": [],
            "unknown_modifications": [],
            "preservation": "no reset, stash, restore, clean, or overwrite",
        },
    )
    processes = subprocess.run(
        ["ps", "-eo", "pid,etime,pcpu,pmem,state,args"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (root / "processes.txt").write_text(processes, encoding="utf-8")


def _replay_upgrade(
    *, reference: Path, mesh: Path, actions_path: Path, scene_root: Path, episodes: int
) -> dict[str, Any]:
    actions = np.asarray(np.load(actions_path, allow_pickle=False)["actions"], dtype=np.float64)
    backend = _make_backend(
        reference_path=reference,
        mesh_path=mesh,
        scene_root=scene_root,
        seed=20260801,
    )
    terminal_states: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    for _ in range(episodes):
        state = backend.reset(reference_index=0)
        reason: str | None = None
        peak_axis_m = 0.0
        for action in actions:
            state, _, reason = backend.transition(action)
            peak_axis_m = max(peak_axis_m, _metrics(backend, state)["max_axis_error_m"])
            if reason is not None:
                break
        final = _metrics(backend, state)
        rows.append(
            {
                "termination": reason or "FAILURE_EVALUATION_STEP_BOUND",
                "reference_index": backend.reference_index,
                "progress": backend.reference_index / (backend.reference.frame_count - 1),
                "object_position_error_cm": 100.0 * final["object_position_error_m"],
                "object_rotation_error_deg": final["object_rotation_error_deg"],
                "max_axis_error_cm": 100.0 * final["max_axis_error_m"],
                "peak_axis_error_cm": 100.0 * peak_axis_m,
                "wrist_position_error_cm": 100.0 * final["wrist_position_error_m"],
                "wrist_rotation_error_deg": final["wrist_rotation_error_deg"],
            }
        )
        terminal_states.append(
            {
                "qpos": backend.data.qpos.copy(),
                "qvel": backend.data.qvel.copy(),
                "qacc_warmstart": backend.data.qacc_warmstart.copy(),
            }
        )
    baseline = terminal_states[0]
    max_difference = {
        name: max(float(np.max(np.abs(state[name] - baseline[name]))) for state in terminal_states)
        for name in baseline
    }
    first = rows[0]
    return {
        "status": "STAGE16B_48X4_BOUNDED_UPGRADE_FAILED",
        "episodes": episodes,
        "action_only_replay": True,
        "action_trace": str(actions_path.resolve()),
        "action_trace_sha256": _sha256(actions_path),
        "action_count": len(actions),
        "success_rate": 0.0,
        "final_reach_rate": 0.0,
        "unique_terminal_tuple_count": len(
            {
                (
                    row["termination"],
                    row["reference_index"],
                    round(row["object_position_error_cm"], 12),
                    round(row["object_rotation_error_deg"], 12),
                    round(row["max_axis_error_cm"], 12),
                )
                for row in rows
            }
        ),
        "determinism_max_difference": max_difference,
        "summary": first,
        "episodes_detail": rows,
        "population": 48,
        "iterations": 4,
        "elites": 12,
        "further_mujoco_budget_authorized": False,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(".local/reports/stage16b_adaptive_oracle_single_ppo"),
    )
    parser.add_argument(
        "--formal-attempt",
        type=Path,
        default=Path(".local/reports/stage16b_adaptive_oracle_single_ppo/attempts/attempt03_32x3"),
    )
    parser.add_argument(
        "--upgrade-actions",
        type=Path,
        default=Path(
            ".local/experiments/stage16b_adaptive_oracle_single_ppo/"
            "oracle_attempt04_48x4/hocap_170105.world_wrist.stage16.adaptive_actions.npz"
        ),
    )
    parser.add_argument(
        "--visual-root",
        type=Path,
        default=Path(".local/experiments/stage16b_adaptive_oracle_single_ppo/visualizations_v2"),
    )
    parser.add_argument("--tests", type=Path)
    args = parser.parse_args()

    report_root = args.report_root.resolve()
    formal_attempt = args.formal_attempt.resolve()
    head = _git("rev-parse", "HEAD")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = REPO / ".local/archive" / f"stage16b1c_mujoco_closeout_{timestamp}_{head[:7]}"
    archive.mkdir(parents=True, exist_ok=False)
    _capture_stage16c_preflight()
    _archive_existing_top_level(report_root, archive)

    formal = _json(formal_attempt / "adaptive_oracle_evaluation.json")
    upgrade = _replay_upgrade(
        reference=REPO / ".local/stage16_reference_tracking_ppo/world_wrist_references/"
        "hocap_170105.world_wrist.stage16.npz",
        mesh=REPO / ".local/stage16_reference_tracking_ppo/world_wrist_objects/hocap_170105.obj",
        actions_path=args.upgrade_actions.resolve(),
        scene_root=REPO
        / ".local/experiments/stage16b_adaptive_oracle_single_ppo/closeout_action_replay",
        episodes=20,
    )
    selected_files = {
        name: formal_attempt / name
        for name in (
            "adaptive_oracle_config.json",
            "adaptive_oracle_determinism.json",
            "adaptive_oracle_evaluation.json",
            "adaptive_oracle_selection_trace.jsonl",
            "frozen_fixed_horizon_baseline.json",
            "nominal_dynamics_profile.json",
            "resource_usage.json",
        )
    }
    visual_files = {
        f"{clip}_{name}": args.visual_root.resolve() / clip / relative
        for clip in ("170105", "170650")
        for name, relative in (
            ("mp4", "adaptive_oracle.mp4"),
            ("contact_sheet", "frames/contact_sheet.png"),
            ("summary", "frames/visualization_summary.json"),
        )
    }
    action_files = {
        clip: REPO
        / ".local/experiments/stage16b_adaptive_oracle_single_ppo/oracle_attempt03_32x3"
        / f"hocap_{clip}.world_wrist.stage16.adaptive_actions.npz"
        for clip in ("170105", "170650")
    }
    artifacts = {
        key: _artifact_record(path)
        for key, path in {**selected_files, **visual_files, **action_files}.items()
        if path.exists()
    }
    config_paths = [
        REPO / "src/toporetarget/rl/world_wrist_oracle.py",
        REPO / "src/toporetarget/rl/environments/world_wrist_backend.py",
        REPO / "scripts/rl/qualify_stage16b_adaptive_oracle.py",
        REPO / "scripts/rl/visualize_hocap_world_wrist_policy_mujoco.py",
    ]
    config_hashes = {str(path.relative_to(REPO)): _sha256(path) for path in config_paths}

    evaluation = formal | {
        "status": "STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL",
        "formal_selected_run": str(formal_attempt),
        "bounded_48x4_upgrade": upgrade,
        "ppo_entry": "STAGE16B_SINGLE_CLIP_PPO_NOT_STARTED_GATE_BLOCKED",
        "further_mujoco_search_authorized": False,
    }
    determinism = _json(formal_attempt / "adaptive_oracle_determinism.json") | {
        "bounded_48x4_upgrade": {
            "episodes": upgrade["episodes"],
            "unique_terminal_tuple_count": upgrade["unique_terminal_tuple_count"],
            "max_difference": upgrade["determinism_max_difference"],
        }
    }
    transitions = [
        {
            "failure": "GATE_BARRIER_UNDERWEIGHTED",
            "attempt": 1,
            "fallback": "formal_gate_barrier_and_gate_first_cost",
            "rerun": "both_clips_32x3",
            "result": "PARTIAL",
        },
        {
            "failure": "ADAPTIVE_SELECTOR_WRONG_HORIZON",
            "attempt": 2,
            "fallback": "common_viability_window",
            "rerun": "both_clips_32x3",
            "result": "PARTIAL",
        },
        {
            "failure": "CEM_CONTACT_MODE_MISS",
            "attempt": 3,
            "fallback": "closed_loop_receding_projection",
            "rerun": "both_clips_32x3",
            "result": "PARTIAL",
        },
        {
            "failure": "CEM_CONTACT_MODE_MISS",
            "attempt": 4,
            "fallback": "single_global_budget_upgrade_48x4",
            "rerun": "global_upgrade_started_for_both;_170105_failure_is_decisive",
            "result": "CLOSED_PARTIAL_NO_FURTHER_MUJOCO_BUDGET",
        },
    ]
    resource = {
        "status": "MUJOCO_CORRECTNESS_BACKEND_CLOSED",
        "selected_32x3": _json(formal_attempt / "resource_usage.json"),
        "upgrade_48x4": {
            "process_status": "interrupted_after_170105_trace_before_complete_two_clip_report",
            "action_trace_replayed_episodes": 20,
        },
        "mujoco_ppo_samples": 0,
        "mujoco_ppo_started": False,
    }
    summary = {
        "branch": _git("branch", "--show-current"),
        "head": head,
        "adaptive_oracle": "STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL",
        "single_clip_ppo": "STAGE16B_SINGLE_CLIP_PPO_NOT_STARTED_GATE_BLOCKED",
        "two_clip_ppo": "STAGE16B_TWO_CLIP_PPO_NOT_STARTED",
        "mujoco_ppo": "MUJOCO_PPO_TRAINING_DEFERRED",
        "mujoco_backend": "MUJOCO_CORRECTNESS_BACKEND_CLOSED",
        "mujoco_backend_roles": [
            "correctness",
            "deterministic_regression",
            "contact_diagnostics",
            "action_replay",
            "interactive_visualization",
        ],
        "frozen_engineering_dynamics_changed": False,
        "further_mujoco_budget_authorized": False,
        "next_stage": "STAGE16C0_ISAACLAB_PLATFORM_QUALIFICATION",
        "archive": str(archive.resolve()),
    }

    report_root.mkdir(parents=True, exist_ok=True)
    _write_json(report_root / "final_summary.json", summary)
    _write_json(report_root / "adaptive_oracle_evaluation.json", evaluation)
    _write_json(report_root / "adaptive_oracle_determinism.json", determinism)
    _write_json(report_root / "resource_usage.json", resource)
    _write_json(report_root / "artifact_integrity.json", artifacts)
    tests = (
        _json(args.tests.resolve())
        if args.tests is not None
        else {"status": "PENDING_CLOSEOUT_VALIDATION"}
    )
    _write_json(report_root / "tests.json", tests)
    (report_root / "oracle_failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions),
        encoding="utf-8",
    )
    summary_md = f"""# Stage 16-B.1c MuJoCo closeout

- Adaptive oracle: `{summary["adaptive_oracle"]}`
- 170105: 0/20, 80% progress; bounded 48x4 replay reaches 82.5% and fails the axis gate.
- 170650: 20/20 pass.
- MuJoCo PPO: **not started**; training is deferred.
- MuJoCo role: correctness, deterministic regression, contact diagnostics,
  action replay, visualization.
- No dynamics parameter, reference, action scale, formal gate, or controller gain changed.
- No further MuJoCo CEM, horizon, or oracle budget is authorized.
- Next lane: independent Isaac Lab platform and PhysX qualification.
"""
    (report_root / "final_summary.md").write_text(summary_md, encoding="utf-8")
    (report_root / "handoff.md").write_text(summary_md, encoding="utf-8")

    oracle_matrix = {
        clip["clip"]: clip["summary"] | {"passes_gate": clip["passes_gate"]}
        for clip in formal["clips"]
    }
    _write_json(archive / "frozen_manifest.json", summary | {"artifacts": artifacts})
    _write_json(archive / "oracle_matrix.json", oracle_matrix | {"170105_48x4": upgrade})
    _write_json(archive / "artifact_hashes.json", artifacts)
    _write_json(archive / "config_hashes.json", config_hashes)
    _write_json(archive / "resource_summary.json", resource)
    (archive / "README.md").write_text(summary_md, encoding="utf-8")
    for name in (
        "adaptive_oracle_config.json",
        "adaptive_oracle_determinism.json",
        "adaptive_oracle_evaluation.json",
        "frozen_fixed_horizon_baseline.json",
        "nominal_dynamics_profile.json",
        "resource_usage.json",
    ):
        shutil.copy2(formal_attempt / name, archive / f"selected_32x3_{name}")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
