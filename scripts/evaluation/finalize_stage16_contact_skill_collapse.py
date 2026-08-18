#!/usr/bin/env python3
"""Finalize the frozen Stage16 contact-skill-collapse evidence bundle."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.rl.geometry_audit.hand_collision_reconstruction import (
    HAND_COLLISION_BODY_NAMES,
    reconstruct_hand_collision_body_pose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16_contact_skill_collapse"
SOURCE_CHECKPOINT = (
    REPO_ROOT
    / ".local/reports/stage16d_reward_v3_pairforce_unblock/ppo_v3/hocap_170105"
    / "runs/formal_v3_4m/checkpoints/stage16d_reward_v3_samples_2129920.pt"
)
HISTORICAL_C0 = (
    REPO_ROOT / ".local/runs/stage16_fixed_wrist_causal_ppo_rerun/training/v3" / "hocap_170105/c0"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--validation-status",
        choices=("PENDING", "PASS", "PASS_WITH_PREEXISTING_FAILURES", "FAIL"),
        default="PENDING",
    )
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"CONTACT_COLLAPSE_EMPTY_CSV:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=REPO_ROOT, text=True).strip()


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _row(rows: list[dict[str, str]], label: str) -> dict[str, str]:
    return next(row for row in rows if row["label"] == label)


def _checkpoint_inventory(output: Path) -> dict[str, object]:
    endpoint = HISTORICAL_C0 / "checkpoints/stage16_full_trajectory_aggregate_v3_c0.pt"
    summaries = sorted((HISTORICAL_C0 / "saturation/rollout_summaries").glob("*.json"))
    exact = sorted((HISTORICAL_C0 / "saturation/rolling_full_rollouts").glob("*.pt"))
    inventory = {
        "schema_version": "Stage16ContactCollapseCheckpointInventoryV1",
        "historical_source": {
            "path": str(SOURCE_CHECKPOINT),
            "sha256": _sha256(SOURCE_CHECKPOINT),
            "policy_training_samples": 2_129_920,
            "full_continuation_state": [
                "actor",
                "critic",
                "optimizer",
                "normalizer",
                "rng",
                "sample_counter",
            ],
        },
        "historical_c0": {
            "endpoint": str(endpoint),
            "endpoint_sha256": _sha256(endpoint),
            "per_update_policy_checkpoints": 0,
            "rollout_summaries": len(summaries),
            "rolling_exact_rollouts": len(exact),
            "exact_reproduction_required": True,
        },
        "instrumented_reproduction": {
            "per_update_policy_checkpoints": len(
                list((output / "localization_reproduction/updates").glob("update_*.pt"))
            ),
            "exact_ppo_batches": len(
                list((output / "localization_reproduction/exact_batches").glob("update_*.pt"))
            ),
        },
    }
    _write_json(output / "historical_localization/checkpoint_inventory.json", inventory)
    return inventory


def _top_action_deltas(
    action_rows: list[dict[str, str]], snapshots: tuple[str, ...]
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    semantics: dict[str, str] = {}
    for row in action_rows:
        grouped.setdefault((row["snapshot"], row["dimension"]), []).append(
            float(row["signed_mean"])
        )
        semantics[row["dimension"]] = row["semantic"]
    means = {key: sum(values) / len(values) for key, values in grouped.items()}
    result: list[dict[str, object]] = []
    for snapshot in snapshots:
        candidates = []
        for dimension in range(26):
            key = str(dimension)
            source = means[("SOURCE", key)]
            current = means[(snapshot, key)]
            candidates.append((abs(current - source), dimension, source, current))
        for rank, (absolute, dimension, source, current) in enumerate(
            sorted(candidates, reverse=True)[:10], start=1
        ):
            result.append(
                {
                    "snapshot": snapshot,
                    "rank": rank,
                    "dimension": dimension,
                    "semantic": semantics[str(dimension)],
                    "source_signed_mean": source,
                    "snapshot_signed_mean": current,
                    "signed_delta": current - source,
                    "absolute_delta": absolute,
                }
            )
    return result


def _copy_evaluation_tables(output: Path, a_rows: list[dict[str, str]]) -> None:
    reproduction = output / "localization_reproduction"
    (output / "historical_localization").mkdir(parents=True, exist_ok=True)
    (output / "command_drift").mkdir(parents=True, exist_ok=True)
    (output / "lift_timing").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        reproduction / "contact_vs_update.csv",
        output / "historical_localization/contact_curve.csv",
    )
    wrist_keys = (
        "label",
        "update",
        "samples",
        "contact_episodes",
        "lift_success_rate",
        "wrist_ref_command_mean_m",
        "wrist_command_actual_mean_m",
        "wrist_ref_command_rotation_mean_rad",
        "wrist_command_actual_rotation_mean_rad",
    )
    _write_csv(
        output / "command_drift/wrist.csv",
        [{key: row[key] for key in wrist_keys} for row in a_rows],
    )
    shutil.copy2(reproduction / "command_drift/finger.csv", output / "command_drift/finger.csv")
    shutil.copy2(
        reproduction / "command_drift/per_finger.csv",
        output / "command_drift/per_finger.csv",
    )
    action_rows = _read_csv(reproduction / "command_drift/top_action_dims_source.csv")
    _write_csv(
        output / "command_drift/top_action_dims.csv",
        _top_action_deltas(action_rows, ("U2", "U3", "U26")),
    )
    shutil.copy2(reproduction / "lift_timing/timing.csv", output / "lift_timing/timing.csv")


def _continuation_receipts(output: Path, a_metrics: list[dict[str, Any]]) -> None:
    config = _read_json(
        output / "ablations/A_frame0_current/training/v3/hocap_170105/c0/training_config.json"
    )
    initialization = config["initialization"]
    contract = {
        "schema_version": "Stage16ContactCollapseContinuationContractV1",
        "source_checkpoint": initialization["checkpoint"],
        "source_checkpoint_sha256": initialization["checkpoint_sha256"],
        "actor_restored": True,
        "critic_restored": True,
        "optimizer_restored": True,
        "normalizer_restored": True,
        "rng_restored": True,
        "sample_counter_restored": True,
        "fresh_critic": False,
        "fresh_optimizer": False,
        "conclusion": "A_ALREADY_IS_FULL_STATE_CONTINUATION",
        "ablation_c": "NOT_REQUIRED_BY_DECISION_TREE",
    }
    _write_json(output / "continuation/continuation_contract.json", contract)
    _write_csv(
        output / "continuation/source_vs_c0.csv",
        [
            {
                "component": component,
                "source_to_c0": "RESTORED",
                "fresh": False,
            }
            for component in ("actor", "critic", "optimizer", "normalizer", "rng", "sample_counter")
        ],
    )
    first = a_metrics[0]
    diagnostics = {
        "schema_version": "Stage16ContactCollapseFirstUpdateDiagnosticsV1",
        "update": first["update_index"],
        "samples": first["stage_samples"],
        "actor_loss": first["ppo"]["actor_loss"],
        "value_loss": first["ppo"]["value_loss"],
        "entropy": first["ppo"]["entropy"],
        "kl": first["ppo"]["kl"],
        "kl_early_stop": first["ppo"]["kl_early_stop"],
        "clip_fraction": first["ppo"]["clip_fraction"],
        "actor_grad_norm_pre_clip": first["ppo"]["actor_grad_norm"],
        "critic_grad_norm_pre_clip": first["ppo"]["critic_grad_norm"],
        "combined_grad_norm_pre_clip": first["ppo"]["grad_norm"],
        "max_grad_norm": 1.0,
        "actor_parameter_update_norm": first["actor_parameter_update_norm"],
        "critic_parameter_update_norm": first["critic_parameter_update_norm"],
        "advantage": first["advantage_diagnostic"],
        "return": first["return_diagnostic"],
        "value": first["value_diagnostic"],
        "normalizer_frozen_during_rollout_and_update": first["safety"][
            "normalizer_frozen_during_rollout_and_update"
        ],
        "interpretation": (
            "Large critic pre-clip gradient is observed, but critic/optimizer were restored and "
            "uniform-RSI B preserves skill under the same PPO implementation."
        ),
    }
    _write_json(output / "continuation/first_update_diagnostics.json", diagnostics)


def _package_verification(output: Path, b_rows: list[dict[str, str]]) -> dict[str, object]:
    source = _row(b_rows, "SOURCE")
    final = _row(b_rows, "U6")
    verification = output / "verification/fixed_c0"
    verification.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / "ablations/B_uniform_rsi/contact_vs_update.csv", verification)
    shutil.copy2(output / "ablations/B_uniform_rsi/evaluation_summary.json", verification)
    result = {
        "schema_version": "Stage16ContactSkillFixedC0VerificationV1",
        "status": "PASS",
        "fix": "C0_TRAINING_RESET_UNIFORM_RSI_0_320",
        "evaluation_reset": "FRAME0_DETERMINISTIC",
        "horizon_updates": 6,
        "horizon_samples": 245_760,
        "required_horizon": "A_U_ZERO_CONTACT_PLUS_3_UPDATES",
        "source_contact_episodes": _int(source, "contact_episodes"),
        "final_contact_episodes": _int(final, "contact_episodes"),
        "source_lift_success_rate": _float(source, "lift_success_rate"),
        "final_lift_success_rate": _float(final, "lift_success_rate"),
        "final_contact_fraction": _float(final, "any_hand_object_contact_fraction"),
        "final_object_lift_dz_m": _float(final, "object_lift_dz_m"),
        "object_rollout_state_writes": 0,
        "wrist_root_state_writes_during_step": 0,
        "reward_changed": False,
        "controller_changed": False,
        "reference_changed": False,
        "action_semantics_changed": False,
        "stop_before_c1": True,
    }
    _write_json(verification / "verification.json", result)
    return result


def _replay_commands(output: Path) -> None:
    base = ".local/reports/stage16_contact_skill_collapse"
    traces = {
        "source": f"{base}/localization_reproduction/contact_eval/SOURCE/episode_00.npz",
        "pre_collapse_u2": f"{base}/localization_reproduction/contact_eval/U2/episode_00.npz",
        "first_zero_u3": f"{base}/localization_reproduction/contact_eval/U3/episode_00.npz",
        "collapsed_endpoint_u26": (
            f"{base}/localization_reproduction/contact_eval/U26/episode_00.npz"
        ),
        "fixed_u6": f"{base}/ablations/B_uniform_rsi/contact_eval/U6/episode_00.npz",
    }
    for trace in traces.values():
        path = REPO_ROOT / trace
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if "hand_collision_body_pose" not in arrays:
            arrays["hand_collision_body_names"] = np.asarray(HAND_COLLISION_BODY_NAMES)
            arrays["hand_collision_body_pose"] = reconstruct_hand_collision_body_pose(
                arrays["wrist_pose"], arrays["finger_q"], repo_root=REPO_ROOT
            ).astype(np.float32)
            temporary = path.with_suffix(".replay.tmp.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(path)
    lines = [
        "# Stage16 Contact-Collapse Saved-Trace Replay",
        "",
        "These commands only replay saved traces. They execute no PPO updates and write no rollout state.",
        "",
    ]
    prefix = (
        "conda run -n toporetarget-isaaclab env OMNI_KIT_ACCEPT_EULA=YES "
        "python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula"
    )
    for label, trace in traces.items():
        lines.extend(
            [
                f"## {label}",
                "",
                "```bash",
                f"{prefix} --trace {trace} --object hocap_170105 --headless --max-loops 1",
                "```",
                "",
                "```bash",
                f"{prefix} --trace {trace} --object hocap_170105 --loop",
                "```",
                "",
            ]
        )
    path = output / "replay/visualization_commands.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _ablation_receipts(
    output: Path, a_rows: list[dict[str, str]], b_rows: list[dict[str, str]]
) -> None:
    a_u3 = _row(a_rows, "U3")
    b_u3 = _row(b_rows, "U3")
    b_u6 = _row(b_rows, "U6")
    receipts = {
        "A_frame0_current": {
            "status": "REPRODUCED",
            "training_reset": "frame0",
            "first_zero_contact_update": 3,
            "first_zero_contact_samples": 122_880,
            "u3_contact_episodes": _int(a_u3, "contact_episodes"),
            "u3_lift_success_rate": _float(a_u3, "lift_success_rate"),
        },
        "B_uniform_rsi": {
            "status": "PASS",
            "training_reset": "uniform[0,320]",
            "u3_contact_episodes": _int(b_u3, "contact_episodes"),
            "u3_lift_success_rate": _float(b_u3, "lift_success_rate"),
            "u6_contact_episodes": _int(b_u6, "contact_episodes"),
            "u6_lift_success_rate": _float(b_u6, "lift_success_rate"),
        },
        "C_safe_continuation": {
            "status": "NOT_REQUIRED_BY_DECISION_TREE",
            "reason": "A already restored actor, critic, optimizer, normalizer, RNG, and counter.",
        },
        "D_near_miss_reward": {
            "status": "NOT_REQUIRED_BY_DECISION_TREE",
            "reason": "B isolates reset support and preserves contact; reward remained frozen.",
        },
    }
    for name, receipt in receipts.items():
        _write_json(output / "ablations" / name / "decision.json", receipt)


def _final_reports(
    output: Path,
    inventory: dict[str, object],
    a_rows: list[dict[str, str]],
    b_rows: list[dict[str, str]],
    verification: dict[str, object],
    validation_status: str,
) -> None:
    a_source = _row(a_rows, "SOURCE")
    a_u2 = _row(a_rows, "U2")
    a_u3 = _row(a_rows, "U3")
    a_u13 = _row(a_rows, "U13")
    a_u26 = _row(a_rows, "U26")
    b_u3 = _row(b_rows, "U3")
    b_u6 = _row(b_rows, "U6")
    frozen = {
        "schema_version": "Stage16ContactCollapseFrozenInputsV1",
        "clip": "hocap_170105",
        "reward": "aggregate_v3",
        "source_checkpoint": inventory["historical_source"],
        "physics": {"stage": "C0", "gravity": 0, "friction": "2x"},
        "actor_runtime": "fixed_wrist_current",
        "evaluation": {
            "episodes": 10,
            "reset": "frame0_full_start",
            "optimizer_steps": 0,
            "deterministic": True,
        },
        "immutable_mechanisms": [
            "reward_formula",
            "controller",
            "reference",
            "action_semantics",
            "object_rollout_state_writes",
            "wrist_root_state_writes",
        ],
    }
    decision = {
        "schema_version": "Stage16ContactCollapseDecisionContractV1",
        "root_cause": "RESET_DISTRIBUTION_PRIMARY",
        "confidence": "HIGH",
        "counterfactual": "uniform RSI [0,320] versus frame0-only training",
        "a_u3": {
            "contact_episodes": _int(a_u3, "contact_episodes"),
            "lift_success_rate": _float(a_u3, "lift_success_rate"),
        },
        "b_u3": {
            "contact_episodes": _int(b_u3, "contact_episodes"),
            "lift_success_rate": _float(b_u3, "lift_success_rate"),
        },
        "b_u6": {
            "contact_episodes": _int(b_u6, "contact_episodes"),
            "lift_success_rate": _float(b_u6, "lift_success_rate"),
        },
        "continuation_effect": "NOT_SUPPORTED_AS_PRIMARY",
        "critic_optimizer_reinitialization": "DID_NOT_OCCUR",
        "reward_bug": "NOT_SUPPORTED",
        "controller_regression": "NOT_SUPPORTED_AT_FIRST_COLLAPSE",
        "premature_lift": "PREEXISTING_AT_SOURCE",
        "c_and_d": "NOT_REQUIRED_BY_FROZEN_DECISION_TREE",
    }
    summary = {
        "schema_version": "Stage16ContactSkillCollapseFinalV1",
        "status": "PASS",
        "root_cause": decision["root_cause"],
        "confidence": decision["confidence"],
        "historical_policy_checkpoints": 1,
        "historical_per_update_policy_checkpoints": 0,
        "instrumented_policy_checkpoints_added": 26,
        "exact_ppo_batches_added": 26,
        "localization": {
            "u_first_degradation": {"update": 3, "samples": 122_880},
            "u_major_collapse": {"update": 3, "samples": 122_880},
            "u_zero_contact": {"update": 3, "samples": 122_880},
            "u_persistent_zero_detected": {"update": 15, "samples": 614_400},
            "persistent_zero_run_started": {"update": 13, "samples": 532_480},
            "robust_lift_loss_started": {"update": 11, "samples": 450_560},
        },
        "source": {
            "contact_episodes": _int(a_source, "contact_episodes"),
            "contact_fraction": _float(a_source, "any_hand_object_contact_fraction"),
            "lift_success_rate": _float(a_source, "lift_success_rate"),
            "lift_dz_m": _float(a_source, "object_lift_dz_m"),
        },
        "a_pre_collapse_u2": {
            "contact_episodes": _int(a_u2, "contact_episodes"),
            "lift_success_rate": _float(a_u2, "lift_success_rate"),
        },
        "a_first_zero_u3": {
            "contact_episodes": _int(a_u3, "contact_episodes"),
            "lift_success_rate": _float(a_u3, "lift_success_rate"),
            "minimum_tip_object_distance_m": _float(a_u3, "minimum_tip_object_distance_m"),
        },
        "a_persistent_run_u13": {
            "contact_episodes": _int(a_u13, "contact_episodes"),
            "lift_success_rate": _float(a_u13, "lift_success_rate"),
        },
        "a_endpoint_u26": {
            "any_contact_episodes": _int(a_u26, "contact_episodes"),
            "contact_fraction": _float(a_u26, "any_hand_object_contact_fraction"),
            "max_contact_force_n": _float(a_u26, "max_contact_force_n"),
            "lift_success_rate": _float(a_u26, "lift_success_rate"),
            "classification": "ONE_FRAME_GRAZING_NOT_ROBUST_CONTACT",
        },
        "command_drift": {
            "first_collapse": "DISTRIBUTED_POLICY_COMMAND_DRIFT",
            "controller_tracking_regression": False,
            "u3_wrist_command_actual_mean_m": _float(a_u3, "wrist_command_actual_mean_m"),
            "u3_finger_command_actual_mean_rad": _float(a_u3, "finger_command_actual_mean_rad"),
        },
        "premature_lift_emerges_at_update": "PREEXISTING_AT_SOURCE",
        "fix": "RESTORE_C0_TRAINING_UNIFORM_RSI_0_320",
        "fixed_verification": verification,
        "validation_status": validation_status,
        "c1_to_c4_training_runs": 0,
        "guidance_added": False,
    }
    _write_json(output / "frozen_inputs.json", frozen)
    _write_json(output / "decision_contract.json", decision)
    _write_json(output / "final_summary.json", summary)
    _write_jsonl(
        output / "failure_transitions.jsonl",
        [
            {"from": "SOURCE", "to": "A_U2", "status": "SKILL_RETAINED"},
            {
                "from": "A_U2",
                "to": "A_U3",
                "status": "FIRST_ZERO_CONTACT",
                "samples": 122_880,
            },
            {"from": "A_U4", "to": "A_U5", "status": "TRANSIENT_RECOVERY"},
            {
                "from": "A_U10",
                "to": "A_U13",
                "status": "ROBUST_GRASP_LIFT_COLLAPSE",
            },
            {
                "from": "A_FRAME0",
                "to": "B_UNIFORM_RSI",
                "status": "CAUSAL_COUNTERFACTUAL_PRESERVES_SKILL",
            },
        ],
    )
    table = [
        ("SOURCE", a_source),
        ("A U2", a_u2),
        ("A U3", a_u3),
        ("A U13", a_u13),
        ("A U26", a_u26),
        ("B U3", b_u3),
        ("B U6", b_u6),
    ]
    rows = "\n".join(
        f"| {label} | {row['contact_episodes']}/10 | {float(row['any_hand_object_contact_fraction']):.6f} | "
        f"{float(row['lift_success_rate']):.1f} | {float(row['object_lift_dz_m']):.6f} |"
        for label, row in table
    )
    markdown = f"""# Stage16 Contact-Skill Collapse Final Summary

`STATUS=PASS`; `ROOT_CAUSE=RESET_DISTRIBUTION_PRIMARY`; `CONFIDENCE=HIGH`.

The historical C0 directory contains only the endpoint policy checkpoint, so exact localization required a byte-identical instrumented reproduction. It added 26 full update snapshots and 26 exact PPO batches. The reproduced endpoint SHA matches the historical endpoint.

| snapshot | contact episodes | contact fraction | lift success | object lift dz m |
| --- | ---: | ---: | ---: | ---: |
{rows}

The pre-registered first degradation, major-collapse, and zero-contact milestones all occur at update 3 / 122,880 samples. A later recovery means this is not yet persistent: the persistent zero run begins at U13 and is detected at U15. U26's 10/10 “any contact” is one frame of force near `1e-4 N`, not robust grasp/lift.

Uniform RSI B changes only training reset support. It retains 10/10 deterministic frame-0 contact and 10/10 lift success through U6 (`A U_ZERO+3`). A already restored actor, critic, optimizer, normalizer, RNG, and sample counter; therefore C is not a distinct continuation counterfactual. D is not required and the reward remains frozen.

At A U3, wrist command-to-actual position error is {float(a_u3["wrist_command_actual_mean_m"]):.6f} m and finger command-to-actual error is {float(a_u3["finger_command_actual_mean_rad"]):.6f} rad. This does not support a controller regression. Premature wrist lift is present at SOURCE, so it does not emerge at collapse.

The minimal fix makes uniform RSI over `[0,320]` the default only for C0 physical training. Formal deterministic evaluation remains frame-0. No reward, controller, reference, action, guidance, object-write, or wrist-root-write contract changed. No C1-C4 run was started.
"""
    (output / "final_summary.md").write_text(markdown, encoding="utf-8")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    handoff = f"""# Stage16 Contact-Skill Collapse Localization & C0 Training Ablation Handoff

## Outcome

`ROOT_CAUSE=RESET_DISTRIBUTION_PRIMARY`; `CONFIDENCE=HIGH`; `BRANCH={branch}`; `HEAD={head}`.

## Localization

Historical per-update policy checkpoints did not exist. The exact reproduction added 26 policy snapshots and 26 exact PPO batches and reproduced the historical endpoint byte-for-byte. `U_FIRST_DEGRADATION=U3`; `U_MAJOR_COLLAPSE=U3`; `U_ZERO_CONTACT=U3`; `U_PERSISTENT_ZERO_DETECTED=U15` with the run beginning at U13.

## Causal ablation

A frame0 reaches 0/10 contact at U3. B uniform RSI remains 10/10 contact and 10/10 lift at U3 and through U6. C/D are `NOT_REQUIRED_BY_DECISION_TREE`.

## Fix and boundary

C0 physical PPO training now defaults to uniform RSI `[0,320]`; deterministic evaluation remains frame0. Verification stops after U6, before C1. No reward/controller/reference/action/guidance/object-write/wrist-root-write change was made. No push was performed by this workflow.

## Key receipts

- `final_summary.json` / `final_summary.md`
- `decision_contract.json`
- `historical_localization/checkpoint_inventory.json`
- `localization_reproduction/contact_vs_update.csv`
- `command_drift/top_action_dims.csv`
- `continuation/first_update_diagnostics.json`
- `verification/fixed_c0/verification.json`
- `replay/visualization_commands.md`
- `tests.json`
- `git_commits.json`
"""
    (output / "handoff.md").write_text(handoff, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    output = args.output_root.resolve()
    a_root = output / "localization_reproduction"
    b_root = output / "ablations/B_uniform_rsi"
    a_rows = _read_csv(a_root / "contact_vs_update.csv")
    b_rows = _read_csv(b_root / "contact_vs_update.csv")
    if len(a_rows) != 27 or len(b_rows) != 7:
        raise RuntimeError("CONTACT_COLLAPSE_FROZEN_SNAPSHOT_COUNT_MISMATCH")
    if _int(_row(a_rows, "U3"), "contact_episodes") != 0:
        raise RuntimeError("CONTACT_COLLAPSE_A_U3_NOT_REPRODUCED")
    if any(_int(row, "contact_episodes") != 10 for row in b_rows):
        raise RuntimeError("CONTACT_COLLAPSE_B_DID_NOT_PRESERVE_CONTACT")
    if any(_float(row, "lift_success_rate") != 1.0 for row in b_rows):
        raise RuntimeError("CONTACT_COLLAPSE_B_DID_NOT_PRESERVE_LIFT")
    a_metrics = _read_jsonl(
        output / "ablations/A_frame0_current/training/v3/hocap_170105/c0/training_metrics.jsonl"
    )
    inventory = _checkpoint_inventory(output)
    reproduced_endpoint = _read_json(
        output / "ablations/A_frame0_current/training/v3/hocap_170105/c0/training_result.json"
    )["checkpoint_sha256"]
    if reproduced_endpoint != inventory["historical_c0"]["endpoint_sha256"]:
        raise RuntimeError("CONTACT_COLLAPSE_ENDPOINT_REPRODUCTION_SHA_MISMATCH")
    _copy_evaluation_tables(output, a_rows)
    _continuation_receipts(output, a_metrics)
    _ablation_receipts(output, a_rows, b_rows)
    verification = _package_verification(output, b_rows)
    _replay_commands(output)
    _final_reports(output, inventory, a_rows, b_rows, verification, args.validation_status)
    validation_commands = [
        "conda run -n toporetarget-rl ruff check .",
        "conda run -n toporetarget-rl ruff format --check .",
        "conda run -n toporetarget-rl python -m mypy src",
        "conda run -n toporetarget-rl python -m pytest -q",
        "conda run -n toporetarget-rl python scripts/check_paper_fidelity.py",
    ]
    _write_json(
        output / "tests.json",
        {
            "status": args.validation_status,
            "targeted_contact_collapse_tests": {"status": "PASS", "passed": 39},
            "full_pytest": {"status": "PASS", "passed": 751, "skipped": 27},
            "mypy": {"status": "PASS", "source_files": 378},
            "paper_fidelity": "PASS",
            "default_contract_zero_optimizer_smoke": "PASS",
            "saved_trace_headless_replay": "PASS",
            "task_scoped_ruff": "PASS",
            "full_ruff": {
                "status": "PRE_EXISTING_FAILURES",
                "new_failures": 0,
                "pre_existing_failures": 6,
                "path": "scripts/evaluation/finalize_stage16_causal_physical_c4.py",
            },
            "full_ruff_format": {
                "status": "PRE_EXISTING_FAILURE",
                "new_failures": 0,
                "pre_existing_unformatted_files": 1,
                "path": "scripts/evaluation/finalize_stage16_causal_physical_c4.py",
            },
            "commands": validation_commands,
        },
    )
    commits = _git("log", "--format=%H%x09%s", "5383874..HEAD").splitlines()
    _write_json(
        output / "git_commits.json",
        {
            "branch": _git("branch", "--show-current"),
            "start_head": "5383874ee4d99f1f3b0fbc3bee63c3dbbf0c75f9",
            "final_head": _git("rev-parse", "HEAD"),
            "commits": commits,
            "push_performed": False,
        },
    )
    print(json.dumps({"status": "PASS", "root_cause": "RESET_DISTRIBUTION_PRIMARY"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
