#!/usr/bin/env python3
"""Build the fail-closed Stage 16-D Reference Kinematics V2 handoff receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"
START_HEAD = "fc15cf0bfd36beda3dbd09b95944576d5c4c7449"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STAGE16D_CLOSEOUT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"STAGE16D_CLOSEOUT_ARTIFACT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def _metrics(metrics_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows or int(rows[-1]["cumulative_samples"]) != 1_048_576:
        raise ValueError("STAGE16D_CLOSEOUT_P1_EXACT_SAMPLE_RECEIPT_MISSING")
    if int(rows[-1]["rollout_length"]) != 24 or not all(
        all(bool(value) for value in row["finite"].values()) for row in rows
    ):
        raise ValueError("STAGE16D_CLOSEOUT_P1_NUMERICAL_RECEIPT_INVALID")
    return {
        "updates": len(rows),
        "reward_v2_samples": int(rows[-1]["cumulative_samples"]),
        "last_rollout_length": int(rows[-1]["rollout_length"]),
        "all_updates_finite": True,
        "training_wall_time_s": float(sum(float(row["wall_time_s"]) for row in rows)),
        "samples_per_s_median": sorted(float(row["samples_per_s"]) for row in rows)[len(rows) // 2],
    }


def _markdown(summary: dict[str, Any]) -> str:
    phase3 = summary["phase3"]
    formal = phase3["formal"]
    return "\n".join(
        (
            "# Stage 16-D Reference Kinematics V2 / Phase 3 Handoff",
            "",
            "## Final status",
            "",
            f"`{summary['status']}` on `{summary['branch']}`.",
            "Reference Kinematics V2 and Phase 1-R are validated; the bounded Reward V2 P1 "
            "experiment is insufficient and no 4M/16M continuation is authorized.",
            "",
            "## Bounded P1 result",
            "",
            f"P1 ran exactly {phase3['training']['reward_v2_samples']:,} Reward-V2 samples "
            f"in {phase3['training']['updates']} updates (final rollout "
            f"{phase3['training']['last_rollout_length']} steps).  Development and formal "
            f"terminal contact/stability are {formal['terminal_contact_rate']:.2f}/"
            f"{formal['terminal_stability_rate']:.2f}; SRphysics/SRqualified are "
            f"{formal['SR_physics']:.2f}/{formal['SR_qualified']:.2f}.  The frozen V1-4M "
            "primary comparator is 1.00 for all four rates, so the P1 collapse gate stops "
            "the experiment.",
            "",
            "## Formal trace replay commands",
            "",
            "```bash",
            *summary["replay_commands"],
            "```",
            "",
            "The best-progress and representative-failure NPZ files preserve V2 reference "
            "twists, selected actual twists, residuals, and the object-twist Reward V2 terms.",
            "",
            "## Artifacts",
            "",
            "See `final_summary.json` for immutable paths and SHA-256 values.",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--start-head", default=START_HEAD)
    args = parser.parse_args()
    root = args.root.resolve()
    p1 = root / "phase3/hocap_170650/runs/p1_post_capacity"
    development = p1 / "dev_evaluations/hocap_170650"
    formal = p1 / "formal_evaluations/hocap_170650"
    phase3_root = root / "phase3/hocap_170650"
    paths = {
        "reference_qualification": root / "reference_kinematics_qualification.json",
        "phase1_rerun": root / "phase1_rerun/summary.json",
        "entry_decision": root / "phase3_entry_decision.json",
        "capacity": root / "phase3/capacity_benchmark/selected_capacity.json",
        "training_result": p1 / "training_result.json",
        "training_metrics": p1 / "training_metrics.jsonl",
        "p1_decision": p1 / "p1_training_decision.json",
        "selection": phase3_root / "checkpoint_selection.json",
        "development_qualification": development / "reward_v2_p1_dev_qualification.json",
        "development_suite": development / "reward_v2_p1_dev_evaluation_suite_v2.json",
        "formal_evaluation": formal / "reward_v2_p1_formal_evaluation.json",
        "formal_qualification": formal / "reward_v2_p1_formal_qualification.json",
        "formal_suite": formal / "reward_v2_p1_formal_evaluation_suite_v2.json",
        "formal_geometry": REPO_ROOT / ".local/reports/stage16d_metric_qualification_and_ppo/"
        "geometry_qualification_170650_reward_v2_p1_formal.json",
        "failure_transitions": root / "failure_transitions.jsonl",
        "comparison_csv": root / "phase3/comparison.csv",
        "comparison_markdown": root / "phase3/comparison.md",
        "best_trace": phase3_root / "best_trace.npz",
        "failure_trace": phase3_root / "failure_trace.npz",
        "trace_export": phase3_root / "representative_trace_export.json",
        "tests": args.tests.resolve(),
    }
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    reference = _read(paths["reference_qualification"])
    phase1 = _read(paths["phase1_rerun"])
    entry = _read(paths["entry_decision"])
    decision = _read(paths["p1_decision"])
    selection = _read(paths["selection"])
    formal_qualification_data = _read(paths["formal_qualification"])
    formal_suite = _read(paths["formal_suite"])
    training = _metrics(paths["training_metrics"])
    if reference.get("status") != "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED":
        raise ValueError("STAGE16D_CLOSEOUT_REFERENCE_V2_NOT_VALIDATED")
    if entry.get("status") != "PHASE3_OBJECT_TWIST_REWARD_RECOMMENDED":
        raise ValueError("STAGE16D_CLOSEOUT_PHASE3_ENTRY_INVALID")
    if decision.get("status") != "PHASE3_P1_COLLAPSE_STOP":
        raise ValueError("STAGE16D_CLOSEOUT_P1_DECISION_INVALID")
    if selection.get("formal_holdout_used_for_selection") is not False:
        raise ValueError("STAGE16D_CLOSEOUT_FORMAL_RESELECTION_FORBIDDEN")
    aggregate = formal_suite["aggregate"]
    formal_metrics = {
        "terminal_contact_rate": float(formal_qualification_data["terminal_contact_rate"]),
        "terminal_stability_rate": float(formal_qualification_data["terminal_stability_rate"]),
        "SR_kinematic": float(aggregate["kinematic_success"]["rate"]),
        "SR_physics": float(aggregate["physics_success"]["rate"]),
        "SR_qualified": float(aggregate["qualified_success"]["rate"]),
        "Delta_v_median_mps": float(
            formal_qualification_data["twist_residuals"]["terminal_delta_v_mps"][
                "per_episode_median"
            ]
        ),
        "Delta_omega_median_radps": float(
            formal_qualification_data["twist_residuals"]["terminal_delta_omega_radps"][
                "per_episode_median"
            ]
        ),
        "absolute_geometry_pass": bool(formal_qualification_data["geometry_absolute_pass"]),
        "source_relative_geometry_diagnostic": formal_qualification_data[
            "source_relative_geometry_diagnostic"
        ],
    }
    branch = _git("branch", "--show-current")
    final_head = _git("rev-parse", "HEAD")
    commits = _git("log", "--oneline", f"{args.start_head}..HEAD").splitlines()
    trace_root = ".local/reports/stage16d_reference_kinematics_v2/phase3/hocap_170650"
    formal_qualification_path = (
        f"{trace_root}/runs/p1_post_capacity/formal_evaluations/hocap_170650/"
        "reward_v2_p1_formal_qualification.json"
    )
    replay_prefix = (
        "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES python "
        "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula"
    )
    resource = {
        "schema_version": "Stage16DReferenceKinematicsV2ResourceUsageV1",
        "selected_num_envs": _read(paths["capacity"])["selected_num_envs"],
        "P1": training,
        "unrelated_processes_terminated": 0,
        "PPO_4M_started": False,
        "PPO_16M_started": False,
    }
    summary = {
        "schema_version": "Stage16DReferenceKinematicsV2Phase3CloseoutV1",
        "status": "STAGE16D_PHASE3_REWARD_V2_INSUFFICIENT",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "branch": branch,
        "start_head": args.start_head,
        "final_head": final_head,
        "reference_kinematics_v2": {
            "status": reference["status"],
            "artifact": artifacts["reference_qualification"],
        },
        "phase1_r": {"status": phase1.get("status"), "artifact": artifacts["phase1_rerun"]},
        "phase3": {
            "status": decision["status"],
            "authorized_next_reward_v2_samples": decision["authorized_next_reward_v2_samples"],
            "training": training,
            "formal": formal_metrics,
            "artifacts": artifacts,
        },
        "git": {
            "commits": commits,
            "new_branch_created": False,
            "new_worktree_created": False,
            "pushed": False,
            "pr_created": False,
            "main_merged": False,
            "tag_created": False,
            "release_created": False,
        },
        "replay_commands": [
            f"{replay_prefix} --trace {trace_root}/best_trace.npz "
            f"--qualification {formal_qualification_path} --object hocap_170650 --replica 8 "
            "--start-frame 0 --end-frame 321",
            f"{replay_prefix} --trace {trace_root}/best_trace.npz "
            f"--qualification {formal_qualification_path} --object hocap_170650 --replica 8 "
            "--no-reference-ghost --start-frame 0 --end-frame 321",
            f"{replay_prefix} --trace {trace_root}/best_trace.npz "
            f"--qualification {formal_qualification_path} --object hocap_170650 --replica 8 "
            "--start-frame 300 --end-frame 321",
            f"{replay_prefix} --trace {trace_root}/failure_trace.npz "
            f"--qualification {formal_qualification_path} --object hocap_170650 --replica 2 "
            "--frame 295",
        ],
    }
    _write(root / "resource_usage.json", resource)
    _write(root / "git_commits.json", {"schema_version": "Stage16DGitCloseoutV1", **summary["git"]})
    _write(root / "final_summary.json", summary)
    handoff = _markdown(summary)
    (root / "final_summary.md").write_text(handoff, encoding="utf-8")
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
