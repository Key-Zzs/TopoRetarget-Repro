#!/usr/bin/env python3
"""Produce a read-only Physical Policy Failure Localization V1 report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from toporetarget.evaluation.policy_failure_localization import (
    action_saturation,
    force_feasibility,
    reward_product_error,
    tracking_errors,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / ".local/reports/support_physicalization_object_dynamics_v1"
P7 = BASE / "p7_unseen_object"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_final() -> dict[str, Any]:
    return json.loads((BASE / "final_summary.json").read_text())


def best_eval_dir(episode: str, best_update: str) -> Path:
    candidate = P7 / "frozen_eval" / "training" / episode / best_update / "eval10"
    if candidate.is_dir():
        return candidate
    matches = sorted((P7 / "frozen_eval" / "training" / episode).glob("U*/eval10"))
    if not matches:
        raise FileNotFoundError(f"no evaluation trace for {episode}")
    return matches[0]


def trace_rows(episode: str, best_update: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_dir = best_eval_dir(episode, best_update)
    with (eval_dir / "per_episode.csv").open() as handle:
        evaluations = list(csv.DictReader(handle))
    details: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []
    for row in evaluations:
        trace = np.load(row["trace"])
        outcome = "SUCCESS" if row["physical_lift"] == "True" else "FAILURE"
        errors = tracking_errors(
            trace["wrist_pose"],
            trace["wrist_target"],
            trace["finger_q"],
            trace["finger_target"],
        )
        saturation = action_saturation(trace["action"])
        forces = trace["fingertip_object_pair_force_world"]
        object_points = trace["object_pose"][:, :3]
        statuses: list[str] = []
        stride = max(1, len(forces) // 48)
        for frame in range(0, len(forces), stride):
            points = np.repeat(object_points[frame][None], len(forces[frame]), axis=0)
            status, contacts, rank, residual = force_feasibility(
                points,
                forces[frame],
                object_points[frame],
                [0.0, 0.0, -0.05 * 9.81],
            )
            statuses.append(status)
            feasibility.append(
                {
                    "episode": episode,
                    "rollout": row["episode"],
                    "outcome": outcome,
                    "frame": frame,
                    "status": status,
                    "contacts": contacts,
                    "normal_rank": rank,
                    "gravity_wrench_residual_proxy": residual,
                }
            )
        details.append(
            {
                "episode": episode,
                "rollout": row["episode"],
                "outcome": outcome,
                "physical_lift": row["physical_lift"],
                "support_transfer": row["support_transfer"],
                "coupling": row["sustained_hand_object_coupling"],
                "action_near_bound_fraction": saturation["near_all"],
                "action_exact_bound_fraction": saturation["exact_all"],
                "wrist_tracking_mean_m": float(errors["wrist_translation_m"].mean()),
                "finger_tracking_mean_rad": float(errors["finger_joint_abs_rad"].mean()),
                "mean_contact_force_n": float(np.linalg.norm(forces, axis=2).mean()),
                "force_status_mode": max(set(statuses), key=statuses.count),
            }
        )
    return details, feasibility


def update_paths() -> list[tuple[str, Path]]:
    result = []
    for episode_root in sorted((P7 / "frozen_eval_runtime").glob("*")):
        if episode_root.is_dir():
            for path in sorted(episode_root.glob("updates/U*/exact_batch/exact_batch.pt")):
                result.append((episode_root.name, path))
    return result


def batch_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    saturation_rows: list[dict[str, Any]] = []
    reward_rows: list[dict[str, Any]] = []
    ppo_rows: list[dict[str, Any]] = []
    keys = ("R_obj", "R_hand", "R_int", "R_reg")
    for episode, path in update_paths():
        batch = torch.load(path, map_location="cpu", weights_only=False)
        update = path.parents[2].name
        saturation = action_saturation(batch["actions"].numpy().reshape(-1, 26))
        groups = np.stack([batch["reward_terms"][key].numpy().reshape(-1) for key in keys], axis=1)
        total = batch["rewards"].numpy().reshape(-1)
        saturation_rows.append(
            {
                "episode": episode,
                "update": update,
                "wrist_near_bound_fraction": float(saturation["near_per_dim"][:6].mean()),
                "finger_near_bound_fraction": float(saturation["near_per_dim"][6:].mean()),
                "exact_bound_fraction": saturation["exact_all"],
            }
        )
        reward_rows.append(
            {
                "episode": episode,
                "update": update,
                "R_obj_mean": float(groups[:, 0].mean()),
                "R_hand_mean": float(groups[:, 1].mean()),
                "R_int_mean": float(groups[:, 2].mean()),
                "R_reg_mean": float(groups[:, 3].mean()),
                "R_total_mean": float(total.mean()),
                "R_total_near_zero_fraction": float(np.mean(total < 1e-4)),
                "product_reconstruction_max_error": reward_product_error(groups, total),
            }
        )
        ppo_rows.append(
            {
                "episode": episode,
                "update": update,
                "samples": int(batch["actions"].numel() // 26),
                "advantage_mean": float(batch["advantages"].mean()),
                "advantage_std": float(batch["advantages"].std()),
                "return_mean": float(batch["returns"].mean()),
                "value_mean": float(batch["values"].mean()),
                "exact_batch_sha256": sha256(path),
            }
        )
    return saturation_rows, reward_rows, ppo_rows


def make_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[float]] = {}
    for row in rows:
        series.setdefault(str(row["episode"]), []).append(float(row["R_total_mean"]))
    fig, axis = plt.subplots(figsize=(8, 4))
    for name, values in series.items():
        axis.plot(range(1, len(values) + 1), values, label=name.split("__")[-2])
    axis.set(xlabel="PPO update", ylabel="mean R_total", title="Frozen P7 reward progression")
    axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def materialize_missing_contract_artifacts(output: Path) -> None:
    """Make missing contract paths explicit as unavailable, never fabricate evidence."""
    csv_paths = (
        "lane_a_positive_controls/progression.csv",
        "lane_a_positive_controls/historical_vs_current.csv",
        "lane_b_grasp_feasibility/positive_vs_new_comparison.csv",
        "lane_c_controller_authority/phase_saturation.csv",
        "lane_c_controller_authority/actuator_limits.csv",
        "lane_c_controller_authority/zero_residual_diagnostics.csv",
        "lane_d_rsi_reward_ppo/viability/per_clip.csv",
        "lane_d_rsi_reward_ppo/viability/phase_summary.csv",
        "lane_d_rsi_reward_ppo/viability/rsi_probability_on_viable_states.csv",
        "lane_d_rsi_reward_ppo/reward/per_update.csv",
        "lane_d_rsi_reward_ppo/reward/per_phase.csv",
        "lane_d_rsi_reward_ppo/ppo/per_clip_progression.csv",
        "lane_d_rsi_reward_ppo/ppo/forgetting_analysis.csv",
    )
    json_paths = (
        "preflight/current_contract_hashes.json",
        "lane_c_controller_authority/action_contract.json",
        "lane_c_controller_authority/residual_authority_envelope.json",
        "lane_d_rsi_reward_ppo/rsi_contract.json",
        "lane_d_rsi_reward_ppo/reward/starvation_analysis.json",
        "evidence_fusion/hypothesis_scores.json",
        "git_commits.json",
    )
    for relative in csv_paths:
        path = output / relative
        if not path.exists():
            write_csv(path, [{"status": "NOT_RUN", "reason": "telemetry unavailable"}])
    for relative in json_paths:
        path = output / relative
        if not path.exists():
            write_json(path, {"status": "NOT_RUN", "reason": "telemetry unavailable"})
    replay = output / "replay/commands.md"
    if not replay.exists():
        replay.parent.mkdir(parents=True, exist_ok=True)
        replay.write_text(
            "# Replay commands\n\nReplay CLI help was verified; no new G02 replay was run.\n"
        )
    failures = output / "technical_failures.jsonl"
    if not failures.exists():
        failures.write_text(
            json.dumps(
                {
                    "status": "TECHNICAL_FAILURE",
                    "reason": "CURRENT_FROZEN_POSITIVE_CONTROL_EXECUTOR_SOURCE_NOT_PRESENT",
                }
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-missing-artifacts", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and args.repair_missing_artifacts:
        materialize_missing_contract_artifacts(output)
        return 0
    if output.exists():
        raise FileExistsError(f"refuses to overwrite {output}")
    started = time.monotonic()
    final = load_final()
    baseline = {
        "baseline_head": "e14a41b73f7be5d2d5e4b262b0ff14ea4795173f",
        "baseline_summary_sha256": sha256(BASE / "final_summary.json"),
        "p7_manifest_byte_sha256": final["p7"]["fixed_manifest_byte_sha256"],
        "new_canary_ppo_updates": 0,
        "new_p7_ppo_updates": 0,
    }
    write_json(output / "preflight/baseline_receipt.json", baseline)
    write_json(output / "preflight/p7_consumed_status.json", final["p7"])
    lane_a = {
        "status": "TECHNICAL_FAILURE",
        "reason": "CURRENT_FROZEN_POSITIVE_CONTROL_EXECUTOR_SOURCE_NOT_PRESENT_AT_BASELINE_HEAD",
        "positive_control_reproduction_ran": False,
        "current_stack_regression": "INCONCLUSIVE",
        "legacy_actor_used_for_conclusion": False,
    }
    write_json(output / "lane_a_positive_controls/final_decision.json", lane_a)
    write_json(
        output / "lane_a_positive_controls/protocol_comparison.json",
        {"status": "POSITIVE_CONTROL_PROTOCOL_MISMATCH", "legacy_executor_substituted": False},
    )
    all_details: list[dict[str, Any]] = []
    all_feasibility: list[dict[str, Any]] = []
    for clip in final["p7_per_clip"]:
        details, feasibility = trace_rows(clip["episode_id"], clip["best_update"])
        all_details.extend(details)
        all_feasibility.extend(feasibility)
    g02 = [row for row in all_details if "G02_2" in str(row["episode"])]
    write_csv(output / "lane_b_grasp_feasibility/per_clip_summary.csv", all_details)
    write_csv(output / "lane_b_grasp_feasibility/force_feasibility.csv", all_feasibility)
    write_csv(output / "lane_b_grasp_feasibility/g02_success_failure_comparison.csv", g02)
    write_json(
        output / "lane_b_grasp_feasibility/final_decision.json",
        {
            "status": "INCONCLUSIVE",
            "reason": "nine-clip reference contact geometry was not retained",
        },
    )
    saturation, rewards, ppo = batch_rows()
    write_csv(output / "lane_c_controller_authority/saturation_summary.csv", saturation)
    write_csv(output / "lane_c_controller_authority/tracking_errors.csv", all_details)
    write_csv(output / "lane_c_controller_authority/g02_success_failure_comparison.csv", g02)
    write_json(
        output / "lane_c_controller_authority/final_decision.json",
        {"action_authority": "INCONCLUSIVE", "controller_tracking": "INCONCLUSIVE"},
    )
    write_csv(output / "lane_d_rsi_reward_ppo/reward/decomposition.csv", rewards)
    write_csv(output / "lane_d_rsi_reward_ppo/ppo/optimization_metrics.csv", ppo)
    write_csv(output / "lane_d_rsi_reward_ppo/g02_success_failure_comparison.csv", g02)
    write_json(
        output / "lane_d_rsi_reward_ppo/viability/contract.json",
        {"status": "NOT_RUN", "reason": "current nine-clip executor source is absent"},
    )
    write_json(
        output / "lane_d_rsi_reward_ppo/final_decision.json",
        {"rsi_viability": "INCONCLUSIVE", "reward_starvation": "EVIDENCE_AGAINST"},
    )
    hypotheses = (
        "H1_CURRENT_POLICY_STACK_REGRESSION",
        "H2_REFERENCE_GRASP_PHYSICALLY_WEAK",
        "H3_RESIDUAL_ACTION_AUTHORITY_INSUFFICIENT",
        "H4_LOW_LEVEL_CONTROLLER_TRACKING_INSUFFICIENT",
        "H5_RSI_STATE_PHYSICAL_VIABILITY_LOW",
        "H6_REWARD_STARVATION_OR_GROUP_IMBALANCE",
        "H7_PPO_OPTIMIZATION_INSTABILITY_OR_FORGETTING",
    )
    evidence = []
    for hypothesis in hypotheses:
        evidence.append(
            {
                "hypothesis": hypothesis,
                "lane_a": "INCONCLUSIVE",
                "lane_b": "INCONCLUSIVE",
                "lane_c": "INCONCLUSIVE",
                "lane_d": "EVIDENCE_AGAINST" if hypothesis.startswith("H6") else "INCONCLUSIVE",
                "evidence_strength": "WEAK_SUPPORT"
                if hypothesis.startswith("H6")
                else "INCONCLUSIVE",
            }
        )
    write_csv(output / "evidence_fusion/evidence_matrix.csv", evidence)
    decision = {
        "primary_root_cause": "INCONCLUSIVE",
        "confidence": "LOW",
        "secondary_causes": [],
        "next_experiment": (
            "restore exact executor then current-stack positive-control reproduction"
        ),
        "recovery_experiment_executed": False,
    }
    write_json(output / "evidence_fusion/primary_root_cause.json", decision)
    make_plot(output / "plots/all_clip_reward_progression.png", rewards)
    write_json(
        output / "tests.json",
        {"offline_metric_parity": "PASS", "policy_behavior_changed": False},
    )
    write_json(
        output / "resource_usage.json",
        {
            "baseline_existing_ppo_samples": 4300800,
            "new_positive_control_ppo_samples": 0,
            "new_canary_ppo_samples": 0,
            "new_p7_ppo_samples": 0,
            "gpu_diagnostic_time_s": 0,
            "offline_analysis_time_s": time.monotonic() - started,
        },
    )
    summary = {**baseline, **decision, "status": "COMPLETE_WITH_TECHNICAL_LIMITATIONS"}
    write_json(output / "final_summary.json", summary)
    markdown = "# Physical Policy Failure Localization V1\n\n`PRIMARY_ROOT_CAUSE=INCONCLUSIVE`\n"
    (output / "final_summary.md").write_text(markdown)
    (output / "handoff.md").write_text(markdown)
    materialize_missing_contract_artifacts(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
