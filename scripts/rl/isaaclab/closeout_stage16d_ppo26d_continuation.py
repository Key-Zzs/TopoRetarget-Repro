#!/usr/bin/env python3
"""Assemble the evidence-backed Stage 16-D PPO-26D continuation closeout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d_continuation"
START_HEAD = "bc1d066829cd2e7d0baeb031debefc12e71a03dc"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def training_health(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty training metrics: {path}")
    return {
        "iterations": len(rows),
        "all_finite": all(all(bool(value) for value in row["finite"].values()) for row in rows),
        "max_object_rollout_state_writes": max(
            int(row["reference"]["rsi"]["rollout_object_state_writes"]) for row in rows
        ),
        "max_wrist_rollout_state_writes": max(
            int(row["reference"]["rsi"]["rollout_wrist_root_state_writes"]) for row in rows
        ),
        "samples_per_s_mean": sum(float(row["samples_per_s"]) for row in rows) / len(rows),
        "samples_per_s_last": float(rows[-1]["samples_per_s"]),
        "kl_early_stop_iterations": sum(bool(row["ppo"]["kl_early_stop"]) for row in rows),
    }


def clip_summary(
    *, root: Path, clip: str, qualification: dict[str, Any], selection: dict[str, Any]
) -> dict[str, object]:
    selected = selection["selected"]
    return {
        "clip": clip,
        "r7_status": qualification["status"],
        "physics_qualified": qualification["physics_qualified"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_checkpoint_samples": selected["cumulative_training_samples"],
        "formal_checkpoint": qualification["checkpoint"],
        "formal_checkpoint_sha256": qualification["checkpoint_sha256"],
        "formal_samples": qualification["cumulative_training_samples"],
        "formal_seed_set": qualification["formal_seed_set"]["identifier"],
        "ppo_task_success_rate": qualification["ppo_task_success_rate"],
        "reference_completion_rate": qualification["reference_completion_rate"],
        "terminal_contact_rate": qualification["terminal_contact_rate"],
        "terminal_stability_rate": qualification["terminal_stability_rate"],
        "geometry_formal_pass": qualification["geometry_formal_pass"],
        "qualification": str((root / clip / "r7_formal_qualification.json").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--ruff", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--format", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--mypy", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--pytest", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    parser.add_argument("--paper-fidelity", choices=("PASS", "FAIL", "PENDING"), default="PENDING")
    args = parser.parse_args()
    root = args.output_root.resolve()
    qualification_650 = read_json(root / "hocap_170650/r7_formal_qualification.json")
    qualification_105 = read_json(root / "hocap_170105/r7_formal_qualification.json")
    selection_650 = read_json(root / "hocap_170650/checkpoint_selection.json")
    selection_105 = read_json(root / "hocap_170105/checkpoint_selection.json")
    decision_650 = read_json(root / "hocap_170650/r6b_16m_decision.json")
    decision_105 = read_json(root / "hocap_170105/r6b_16m_decision.json")
    d6 = read_json(root / "d6_authorization.json")
    d7 = read_json(root / "d7_export_gate.json")
    training_105 = read_json(root / "hocap_170105/r6b/training.json")
    training_650 = read_json(root / "hocap_170650/r6b/training.json")
    health_105 = training_health(root / "hocap_170105/r6b/training_metrics.jsonl")
    health_650 = training_health(root / "hocap_170650/r6b/training_metrics.jsonl")
    if training_105["clip"] != "hocap_170105" or training_650["clip"] != "hocap_170650":
        raise ValueError("closeout clip identity mismatch")
    if d6["multiclip_training_authorized"] or d7["d7_export_authorized"]:
        raise ValueError("this closeout is only for the observed D6/D7-denied result")

    clips = {
        "hocap_170650": clip_summary(
            root=root, clip="hocap_170650", qualification=qualification_650, selection=selection_650
        ),
        "hocap_170105": clip_summary(
            root=root, clip="hocap_170105", qualification=qualification_105, selection=selection_105
        ),
    }
    global_contract = {
        "schema_version": "Stage16DPPO26DGlobalContractV1",
        "reward": "TopoRetargetReferenceTrackingReward26DV1",
        "rsi": "Stage16DPPO26DRSIV1_uniform",
        "action": "Stage16DReferenceResidualAction26DV1",
        "action_dimension": 26,
        "observation": "Stage16DPPO26DObservationV2",
        "observation_dimension": 764,
        "wrist_adapter": "existing_se3_to_explicit_serial_3p3r",
        "control": "20Hz_control__120Hz_physics__decimation6",
        "reference": "factor8_321_samples",
        "physics": "free_zero_gravity_object__0.05kg__unit_friction__self_collision",
        "ppo_learning_rate": 1.0e-4,
        "target_kl": 0.03,
        "r6c_or_reward_v2_authorized": False,
        "cross_clip_actor_transfer": False,
    }
    roadmap = {
        "schema_version": "Stage16DPPO26DRoadmapDecisionTreeV1",
        "actual_path_170650": ["R5", "R6A", "R6B", "R7"],
        "actual_path_170105": ["R8_L0", "R6A_4M", "R6A_5M_EXTENSION", "R6B", "R7"],
        "not_run_gate_condition": ["R6C", "R6D", "D6_MULTI_CLIP", "D7_EXPORT"],
        "r6b_decisions": {
            "hocap_170650": decision_650["decision"],
            "hocap_170105": decision_105["decision"],
        },
        "d6_status": d6["status"],
        "d7_status": d7["status"],
    }
    gpu = read_json(root / "hocap_170105/gpu_before_training.json")
    resource_usage = {
        "schema_version": "Stage16DPPO26DResourceUsageV2",
        "host_gpu_before_170105": gpu,
        "training": {
            "hocap_170650": {"training": training_650, "health": health_650},
            "hocap_170105": {"training": training_105, "health": health_105},
        },
    }
    tests = {
        "schema_version": "Stage16DPPO26DContinuationTestReceiptV1",
        "ruff_check": args.ruff,
        "ruff_format_check": args.format,
        "mypy": args.mypy,
        "pytest": args.pytest,
        "paper_fidelity": args.paper_fidelity,
        "local_tracked_files": len(git("ls-files", ".local").splitlines()),
    }
    git_commits = {
        "schema_version": "Stage16DPPO26DGitCloseoutV1",
        "branch": git("branch", "--show-current"),
        "start_head": START_HEAD,
        "current_head": git("rev-parse", "HEAD"),
        "start_to_current": git("log", "--oneline", f"{START_HEAD}..HEAD").splitlines(),
        "NEW_BRANCH_CREATED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "MAIN_MERGED": "NO",
        "TAG_CREATED": "NO",
        "RELEASE_CREATED": "NO",
    }
    summary = {
        "schema_version": "Stage16DPPO26DContinuationFinalSummaryV1",
        "status": "STAGE16D_PPO26D_TRAINED_WITH_PARTIAL_QUALIFICATION",
        "branch": git_commits["branch"],
        "global_contract": global_contract,
        "clips": clips,
        "r6b_decisions": roadmap["r6b_decisions"],
        "multi_clip": d6,
        "export": d7,
        "replay": {
            "hocap_170650": {
                "success": str((root / "hocap_170650/r7_replay_success_receipt.json").resolve()),
                "failure": str((root / "hocap_170650/r7_replay_failure_receipt.json").resolve()),
            },
            "hocap_170105": {
                "best_progress": str(
                    (root / "hocap_170105/r7_replay_best_progress_receipt.json").resolve()
                ),
                "typical_failure": str(
                    (root / "hocap_170105/r7_replay_typical_failure_receipt.json").resolve()
                ),
            },
        },
        "visualization": {
            "hocap_170650": {"replica": 1, "contact_window": [147, 321], "key_frame": 320},
            "hocap_170105": {"replica": 10, "contact_window": [194, 321], "key_frame": 247},
        },
        "tests": tests,
        "git": git_commits,
    }
    table_rows = []
    for clip, training in (("hocap_170650", training_650), ("hocap_170105", training_105)):
        result = clips[clip]
        table_rows.append(
            "| {clip} | {trained:,} | {selected:,} | {task:.2f} | {contact:.2f} | "
            "{stability:.2f} | relative fail | `{status}` |".format(
                clip=clip,
                trained=int(training["actual_cumulative_samples"]),
                selected=cast(int, result["selected_checkpoint_samples"]),
                task=cast(float, result["ppo_task_success_rate"]),
                contact=cast(float, result["terminal_contact_rate"]),
                stability=cast(float, result["terminal_stability_rate"]),
                status=cast(str, result["r7_status"]),
            )
        )
    table = "\n".join(table_rows)
    table_header = (
        "| Clip | Samples trained | Best checkpoint | Formal task success | Terminal contact | "
        "Terminal stability | Geometry | R7 status |\n"
        "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |"
    )
    markdown = f"""# Stage 16-D PPO-26D Continuation and Qualification Handoff

Status: `{summary["status"]}`

{table_header}
{table}

Actual route: `170650: R5 -> R6A -> R6B -> R7`; `170105: R8 L0 -> R6A 4M
-> 5M extension -> R6B -> R7`.
Both R6B branches ended at their frozen 16M decision. R6C, R6D, multi-clip PPO,
and qualified export were not run because their gates did not authorize them.

`D6={d6["status"]}`; `D7={d7["status"]}`.

The global contract remained `Reward V1 + uniform RSI`, 26-D action, 764-D
observation, explicit 3P+3R wrist adapter, factor-8 reference, and free-object
IsaacLab physics. No actor transfer, object action/state write, hidden force, or
attachment was introduced.

Visualization evidence is recorded in `final_summary.json`: 170650 uses replica
1, contact window `147:321`, key frame 320; 170105 uses replica 10, contact
window `194:321`, key frame 247.

Limitations: virtual wrist is not a real arm; factor-8 changes timing; the
zero-gravity/no-support physics contract and material parameters are not real
world calibration; collision proxies are not visual truth; and simulation
qualification is not sim-to-real validation.
"""
    write_json(root / "global_ppo_contract.json", global_contract)
    write_json(root / "roadmap_decision_tree.json", roadmap)
    write_json(root / "resource_usage.json", resource_usage)
    write_json(root / "tests.json", tests)
    write_json(root / "git_commits.json", git_commits)
    write_json(root / "final_summary.json", summary)
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
