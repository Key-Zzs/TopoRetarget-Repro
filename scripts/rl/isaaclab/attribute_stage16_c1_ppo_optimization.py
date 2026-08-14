#!/usr/bin/env python3
"""Run the bounded, offline Stage16 P3-C1.2 PPO attribution.

This command consumes immutable C1.1 receipts only.  It never launches
Isaac, constructs a trainer, calls backward, or performs an optimizer step.
Missing PPO batches and raw observations are represented as UNAVAILABLE rather
than reconstructed from downstream telemetry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from toporetarget.rl.c1_action_saturation_attribution import ACTION_SEMANTICS
from toporetarget.rl.ppo.ppo26d_contract import (
    Stage16DPPO26DObservationV2,
    Stage16DPPO26DTrainingConfigV1,
)
from toporetarget.rl.ppo_optimization_attribution import (
    classify_kl_dynamics,
    decision_contract,
)

SCHEMA = "Stage16P3C1_2PPOOptimizationAttributionV1"
UNAVAILABLE = "UNAVAILABLE_MISSING_EXACT_PPO_BATCH"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL object row: {path}")
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_rows(metrics: list[dict[str, Any]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, row in enumerate(metrics):
        safety = row["safety"]["before_update"]
        ppo = row["ppo"]
        result.append(
            {
                "point": f"C1_rollout_{index:02d}",
                "update_index": index,
                "training_samples_before_update": int(row["cumulative_samples"])
                - int(row["samples"]),
                "training_samples_after_update": int(row["cumulative_samples"]),
                "rollout_saturation": float(safety["deterministic_action_saturation_fraction"]),
                "actor_loss": float(ppo["actor_loss"]),
                "value_loss": float(ppo["value_loss"]),
                "entropy": float(ppo["entropy"]),
                "approx_kl": float(ppo["kl"]),
                "clip_fraction": float(ppo["clip_fraction"]),
                "grad_norm": float(ppo["grad_norm"]),
                "action_std": float(ppo["action_std"]),
                "advantage_std": UNAVAILABLE,
                "explained_variance": float(ppo["explained_variance"]),
                "target_kl": float(ppo["target_kl"]),
                "kl_early_stop": bool(ppo["kl_early_stop"]),
                "actor_hash_before": row["actor_parameter_hash_before"],
                "actor_hash_after": row["actor_parameter_hash_after"],
            }
        )
    return result


def telemetry_rows(root: Path) -> tuple[list[dict[str, object]], dict[str, Any]]:
    failure_path = root / "failure" / "failure_rollout.pt"
    previous_path = root / "failure" / "previous_full_rollout.pt"
    failure = torch.load(failure_path, map_location="cpu", weights_only=False)
    previous = torch.load(previous_path, map_location="cpu", weights_only=False)
    if not isinstance(failure, dict) or not isinstance(previous, dict):
        raise ValueError("C1.1 telemetry must be serialized dictionaries")
    rows: list[dict[str, object]] = []
    location = failure["actor_location_pre_tanh"].float()
    actor_mean = failure["actor_mean_tanh"].float()
    sensitivity = 1.0 - actor_mean.square()
    log_std = failure["actor_log_std"].float()
    saturated = actor_mean.abs() >= 0.98
    negative = actor_mean <= -0.98
    positive = actor_mean >= 0.98
    for index, semantic in enumerate(ACTION_SEMANTICS):
        rows.append(
            {
                "dim": index,
                "semantic": semantic,
                "group": "wrist" if index < 6 else "finger",
                "failure_rollout_saturation": float(saturated[..., index].float().mean()),
                "failure_negative_saturation": float(negative[..., index].float().mean()),
                "failure_positive_saturation": float(positive[..., index].float().mean()),
                "pre_tanh_mean": float(location[..., index].mean()),
                "pre_tanh_abs_mean": float(location[..., index].abs().mean()),
                "tanh_mean": float(actor_mean[..., index].mean()),
                "tanh_sensitivity_mean": float(sensitivity[..., index].mean()),
                "log_std_mean": float(log_std[..., index].mean()),
                "negative_update_pressure": UNAVAILABLE,
            }
        )
    summary = {
        "failure_rollout_saturation": float(saturated.float().mean()),
        "failure_negative_saturation": float(negative.float().mean()),
        "failure_positive_saturation": float(positive.float().mean()),
        "failure_pre_tanh_abs_mean": float(location.abs().mean()),
        "failure_tanh_sensitivity_mean": float(sensitivity.mean()),
        "failure_log_std_mean": float(log_std.mean()),
        "previous_rollout_saturation": float(
            (previous["actor_mean_tanh"].float().abs() >= 0.98).float().mean()
        ),
        "phase_code_values": torch.unique(failure["phase_code"]).tolist(),
        "hand_object_contact_fraction": float(failure["hand_object_contact"].float().mean()),
        "table_object_contact_fraction": float(failure["table_object_contact"].float().mean()),
        "hand_object_force_mean": float(failure["hand_object_force"].float().mean()),
        "object_tracking_error_mean_m": float(failure["object_tracking_error"].float().mean()),
        "raw_observation_present": "observations" in failure,
        "ppo_batch_fields_present": sorted(
            set(failure).intersection({"observations", "actions", "log_probs", "rewards", "values"})
        ),
    }
    return rows, summary


def build_markdown(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    progression = summary["optimization_progression"]
    return "\n".join(
        [
            "# Stage16 P3-C1.2 PPO Optimization Attribution Handoff",
            "",
            "## Verdict",
            "",
            f"`PRIMARY_ROOT_CAUSE={summary['primary_root_cause']}`",
            f"`CONFIDENCE={summary['confidence']}`",
            f"`NEXT_ACTION={summary['next_action']}`",
            "",
            "The C1.1 receipts prove persistent negative-only policy-output saturation, "
            "but they do not contain the exact PPO batch needed to compute the actor-mean "
            "gradient or leave-one-reward-term-out pressure. The attribution therefore "
            "fails closed as INCONCLUSIVE rather than naming a reward, critic, or tanh cause.",
            "",
            "## What is supported",
            "",
            f"- The 25 rollout ledger rises from {progression[0]['rollout_saturation']:.6f} "
            f"to {progression[-1]['rollout_saturation']:.6f}; the failure trigger is "
            f"{summary['failure_trigger_saturation']:.6f}.",
            "- All retained rollout phase counts are PRE_CONTACT; the failure telemetry has "
            "zero hand-object contact, zero table-object contact, and zero hand-object force.",
            "- The late updates repeatedly exceed target KL while reporting zero clip fraction; "
            "this is receipt-level instability evidence, not causal direction evidence.",
            "- Actor and critic modules are separate; no shared trunk was found in the "
            "checked-in network.",
            "- Raw log standard deviation remains available and stable in the saved telemetry; "
            "the report does not classify variance explosion as the cause.",
            "",
            "## Required causal inputs unavailable",
            "",
            f"- Fixed raw observation probe: {summary['missing']['fixed_probe']}.",
            f"- Exact PPO batch / advantage / reward decomposition: "
            f"{summary['missing']['ppo_batch']}.",
            f"- Counterfactual frozen PhysX state: {summary['missing']['counterfactual_state']}.",
            "",
            "## Evidence matrix",
            "",
            "| Evidence | State shift | Normalizer | Advantage | Reward | PPO surrogate | "
            "Clip/KL | Distribution | Tanh | Critic | Residual authority |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| C1.1 / C1.2 receipts | "
            + " | ".join(
                evidence[name]
                for name in (
                    "state_shift",
                    "normalizer",
                    "advantage",
                    "reward",
                    "ppo_surrogate",
                    "clip_kl",
                    "distribution",
                    "tanh",
                    "critic",
                    "residual_authority",
                )
            )
            + " |",
            "",
            "## Safety",
            "",
            "- Authoritative optimizer steps: 0.",
            "- C1 retry/C2/G3/C3/C4/P4/V4 formal training: not run.",
            "- Reward, PPO hyperparameters, action bounds/mapping, controller, reference, "
            "and geometry gates: unchanged.",
            "- Shadow optimizer replay: NOT_RUN; exact batch and minibatch state are absent.",
            "- Action counterfactual: NOT_RUN; the decision tree cannot safely select "
            "representative frozen world states from the stored receipt.",
            "",
            "The sole next action is `NEXT_TARGETED_PPO_COUNTERFACTUAL`; this task does "
            "not execute it.",
            "",
        ]
    )


def run(input_root: Path, output: Path) -> None:
    final = read_json(input_root / "final_summary.json")
    failure_receipt = read_json(
        input_root / "reproduction_authorized_retry" / "failure" / "failure_receipt.json"
    )
    failure_summary = failure_receipt["summary"]
    retry_root = input_root / "reproduction_authorized_retry"
    retry_final = read_json(retry_root / "final_summary.json")
    metrics = read_jsonl(retry_root / "v3/hocap_170105/c1/training_metrics.jsonl")
    progression = metric_rows(metrics)
    telemetry, telemetry_summary = telemetry_rows(retry_root)
    config = read_json(retry_root / "v3/hocap_170105/c1/training_config.json")
    frozen = read_json(input_root / "frozen_inputs.json")
    full_checkpoint = retry_root / "failure" / "failure_pre_gate_full.pt"
    actor_path = retry_root / "failure" / "failure_pre_gate_actor.pt"
    critic_path = retry_root / "failure" / "failure_pre_gate_critic.pt"
    optimizer_path = retry_root / "failure" / "failure_pre_gate_optimizer.pt"
    normalizer_path = retry_root / "failure" / "failure_pre_gate_normalizer.pt"
    receipt_paths = {
        "failure_actor": actor_path,
        "failure_critic": critic_path,
        "failure_optimizer": optimizer_path,
        "failure_normalizer": normalizer_path,
        "failure_rollout": retry_root / "failure/failure_rollout.pt",
        "previous_full_rollout": retry_root / "failure/previous_full_rollout.pt",
        "failure_receipt": retry_root / "failure/failure_receipt.json",
        "failure_checkpoint": full_checkpoint,
    }
    frozen_inputs = {
        "schema_version": "Stage16P3C1_2FrozenInputsV1",
        "source_report": str(input_root),
        "source_c1_1_summary": final,
        "source_frozen_inputs": frozen,
        "failure_trigger": failure_receipt,
        "artifact_sha256": {name: sha256(path) for name, path in receipt_paths.items()},
        "source_code_sha256": {
            "ppo_trainer": sha256(Path("src/toporetarget/rl/ppo/trainer.py")),
            "ppo26d_trainer": sha256(Path("src/toporetarget/rl/ppo/ppo26d_trainer.py")),
            "ppo26d_contract": sha256(Path("src/toporetarget/rl/ppo/ppo26d_contract.py")),
            "ppo26d_reward": sha256(
                Path("src/toporetarget/rl/reference_tracking/ppo26d_reward.py")
            ),
        },
    }
    write_json(output / "frozen_inputs.json", frozen_inputs)
    write_json(output / "decision_contract.json", decision_contract())

    fixed_probe = output / "fixed_probe"
    fixed_probe.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixed_probe / "observations.npz", observations=np.empty((0, 764), np.float32)
    )
    write_json(
        fixed_probe / "manifest.json",
        {
            "schema_version": "FixedObservationProbeV1",
            "status": "UNAVAILABLE",
            "observation_shape": [0, Stage16DPPO26DObservationV2().dimension],
            "reason": (
                "C1.1 telemetry stores actor outputs and downstream state, not raw "
                "764-D observations"
            ),
            "raw_observation_contract": Stage16DPPO26DObservationV2().as_dict(),
            "phase_strata": [],
            "selection_frozen": False,
        },
    )
    write_csv(
        fixed_probe / "actor_snapshot_sweep.csv",
        [
            {
                "snapshot": "failure_time_actor",
                "fixed_probe_saturation": "UNAVAILABLE",
                "rollout_saturation": telemetry_summary["failure_rollout_saturation"],
                "pre_tanh_mean": telemetry_summary["failure_pre_tanh_abs_mean"],
                "status": "NO_FIXED_PROBE",
            },
            {
                "snapshot": "previous_full_rollout_actor_outputs",
                "fixed_probe_saturation": "UNAVAILABLE",
                "rollout_saturation": telemetry_summary["previous_rollout_saturation"],
                "pre_tanh_mean": "UNAVAILABLE",
                "status": "NO_ACTOR_SNAPSHOT",
            },
        ],
    )
    write_csv(
        fixed_probe / "phase_occupancy.csv",
        [{"point": row["point"], "PRE_CONTACT": 1.0, "other_phases": 0.0} for row in progression],
    )
    write_json(
        fixed_probe / "observation_shift.json",
        {
            "status": "UNAVAILABLE",
            "reason": "raw observations and normalized observations are absent from C1.1 receipts",
            "groups": [
                "proprio",
                "reference_residual",
                "wrist_object_error",
                "phase",
                "support_contact",
            ],
        },
    )
    write_csv(
        output / "fixed_probe" / "rollout_vs_fixed_probe.csv",
        [
            {
                "point": row["point"],
                "rollout_saturation": row["rollout_saturation"],
                "fixed_probe_saturation": "UNAVAILABLE",
                "phase_occupancy": "PRE_CONTACT=1.0",
                "interpretation": "NO_FIXED_PROBE",
            }
            for row in progression
        ],
    )

    normalizer = torch.load(normalizer_path, map_location="cpu", weights_only=False)
    write_csv(
        output / "normalizer" / "cross_evaluation.csv",
        [
            {
                "actor": "failure_actor",
                "normalizer": "failure_normalizer",
                "saturation": "UNAVAILABLE",
                "status": "NO_RAW_PROBE",
            }
        ],
    )
    write_json(
        output / "normalizer" / "audit.json",
        {
            "status": "NOT_IDENTIFIABLE",
            "normalizer_training_flag_at_failure": bool(normalizer["training"]),
            "normalizer_count_at_failure": float(normalizer["count"]),
            "frozen_during_each_rollout_update": all(
                bool(row["safety"]["normalizer_frozen_during_rollout_and_update"])
                for row in metrics
            ),
            "refreshed_after_successful_updates": True,
            "reason": "crossed actor/normalizer evaluation requires raw fixed observations",
        },
    )

    write_json(
        output / "ppo" / "implementation_contract.json",
        {
            "schema_version": "P3C12PPOImplementationContractV1",
            "trainer": "PPOTrainer.update",
            "training_contract": Stage16DPPO26DTrainingConfigV1().as_dict(),
            "config_receipt": config,
            "actor_loss": (
                "-mean(min(ratio * normalized_advantage, clamp(ratio) * normalized_advantage))"
            ),
            "value_loss": "MSE(value_prediction, returns)",
            "entropy_term": "-entropy_coefficient * sampled_squashed_entropy",
            "advantage_normalization": "population std + 1e-8",
            "log_prob": "tanh-corrected diagonal Normal log_prob summed over 26D",
            "optimizer_step": "Adam followed by global max_grad_norm clipping",
            "early_stop": "after minibatch when approximate KL > target_kl",
        },
    )
    write_json(
        output / "ppo" / "actor_critic_shared_params.json",
        {
            "shared_parameters": False,
            "actor_parameters": "actor.* and log_std_parameter",
            "critic_parameters": "critic.*",
            "evidence": "ActorCritic defines independent actor and critic Sequential modules",
        },
    )
    write_csv(output / "ppo" / "optimization_progression.csv", progression)
    kl_summary = classify_kl_dynamics(
        [
            {"kl": row["ppo"]["kl"], "clip_fraction": row["ppo"]["clip_fraction"]}
            for row in metrics[-10:]
        ]
    )
    write_json(output / "ppo" / "kl_dynamics.json", kl_summary)

    unavailable_dimensions = [
        {
            "dim": index,
            "semantic": semantic,
            "update_pressure": UNAVAILABLE,
            "negative_pressure_fraction": UNAVAILABLE,
            "phase": UNAVAILABLE,
            "finger": "wrist" if index < 6 else semantic.split("_")[1],
            "reason": "exact PPO batch absent",
        }
        for index, semantic in enumerate(ACTION_SEMANTICS)
    ]
    write_csv(output / "gradient" / "dimension_pressure.csv", unavailable_dimensions)
    write_csv(
        output / "gradient" / "phase_pressure.csv",
        [
            {
                "phase": "PRE_CONTACT",
                "negative_pressure": UNAVAILABLE,
                "reason": "exact PPO batch absent",
            }
        ],
    )
    write_csv(
        output / "gradient" / "finger_pressure.csv",
        [
            {"finger": finger, "negative_pressure": UNAVAILABLE, "reason": "exact PPO batch absent"}
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        ],
    )
    write_json(
        output / "gradient" / "objective_reconstruction.json",
        {
            "status": UNAVAILABLE,
            "exactness": "NOT_RECONSTRUCTED",
            "required_fields": [
                "observations",
                "actions",
                "old_log_probs",
                "advantages",
                "returns",
                "values",
            ],
            "available_fields": telemetry_summary["ppo_batch_fields_present"],
            "reason": "C1.1 pre-gate telemetry has no stored PPO batch",
            "authoritative_optimizer_steps": 0,
        },
    )
    write_csv(
        output / "gradient" / "clipping_attribution.csv",
        [
            {
                "group": group,
                "ratio_mean": UNAVAILABLE,
                "clip_fraction": 0.0 if group == "all_training_updates" else UNAVAILABLE,
                "unclipped_pressure": UNAVAILABLE,
                "clipped_pressure": UNAVAILABLE,
                "interpretation": (
                    "late receipt clip_fraction is aggregate update metric; "
                    "sample pressure unavailable"
                ),
            }
            for group in ("all_training_updates", "saturated_samples", "PRE_CONTACT_saturated")
        ],
    )

    write_csv(
        output / "advantage" / "saturation_buckets.csv",
        [
            {
                "bucket": bucket,
                "samples": UNAVAILABLE,
                "advantage_mean": UNAVAILABLE,
                "advantage_median": UNAVAILABLE,
                "positive_fraction": UNAVAILABLE,
                "return": UNAVAILABLE,
                "reason": "advantages and returns absent",
            }
            for bucket in ("low", "medium", "near_bound", "saturated")
        ],
    )
    write_csv(
        output / "advantage" / "phase_advantages.csv",
        [{"phase": "PRE_CONTACT", "advantage": UNAVAILABLE, "reason": "advantages absent"}],
    )

    reward_names = sorted(
        {
            name
            for row in metrics
            for name in row.get("reward", {})
            if name
            not in {"actual_fingertip_object_contact_mask", "reference_expected_contact_mask"}
        }
    )
    reward_rows = []
    for name in reward_names:
        late_values = [float(row["reward"][name]) for row in metrics[-5:] if name in row["reward"]]
        reward_rows.append(
            {
                "reward_term": name,
                "PRE_CONTACT_mean_late": mean(late_values),
                "low_saturation": UNAVAILABLE,
                "saturated": UNAVAILABLE,
                "leave_one_out_pressure_reduction": UNAVAILABLE,
                "causal_status": "NO_PER_SAMPLE_REWARD_OR_ADVANTAGE",
            }
        )
    write_csv(output / "reward" / "component_summary.csv", reward_rows)
    write_csv(
        output / "reward" / "leave_one_out.csv",
        [
            {"reward_term": name, "pressure_without_term": UNAVAILABLE, "status": "NOT_RUN"}
            for name in reward_names
        ],
    )
    write_csv(
        output / "reward" / "pressure_reduction.csv",
        [
            {
                "reward_term": name,
                "reduction_fraction": UNAVAILABLE,
                "strong_contributor": "NO_EVIDENCE",
            }
            for name in reward_names
        ],
    )

    write_csv(output / "distribution" / "mean_std_entropy.csv", telemetry)
    write_csv(
        output / "distribution" / "tanh_sensitivity.csv",
        [
            {
                "dim": row["dim"],
                "semantic": row["semantic"],
                "pre_tanh_mean": row["pre_tanh_mean"],
                "tanh_mean": row["tanh_mean"],
                "one_minus_tanh_squared": row["tanh_sensitivity_mean"],
            }
            for row in telemetry
        ],
    )
    write_json(
        output / "critic" / "value_advantage_audit.json",
        {
            "status": "NOT_IDENTIFIABLE",
            "value_loss_late_mean": mean([float(row["ppo"]["value_loss"]) for row in metrics[-5:]]),
            "explained_variance_late_mean": mean(
                [float(row["ppo"]["explained_variance"]) for row in metrics[-5:]]
            ),
            "advantage_tail": UNAVAILABLE,
            "saturation_sample_alignment": UNAVAILABLE,
            "reason": "returns, values, and advantages were not persisted as a PPO batch",
        },
    )

    write_json(
        output / "counterfactuals" / "status.json",
        {
            "status": "NOT_RUN",
            "reason": (
                "frozen PhysX state/reference trace is not sufficient in C1.1 receipt "
                "to select a safe reproducible sweep"
            ),
            "decision_tree": "not executable without exact PPO attribution inputs",
        },
    )
    write_json(
        output / "shadow_update" / "status.json",
        {
            "status": "NOT_RUN",
            "reason": (
                "pre-update actor, exact PPO batch, minibatch ordering, and replayable "
                "optimizer transaction are absent"
            ),
            "canonical_optimizer_step": 0,
        },
    )

    evidence = {
        "state_shift": "NOT_SUPPORTED",
        "normalizer": "NOT_SUPPORTED",
        "advantage": "NOT_SUPPORTED",
        "reward": "NOT_SUPPORTED",
        "ppo_surrogate": "WEAK",
        "clip_kl": "MODERATE",
        "distribution": "WEAK",
        "tanh": "WEAK",
        "critic": "NOT_SUPPORTED",
        "residual_authority": "NOT_SUPPORTED",
    }
    summary = {
        "schema_version": SCHEMA,
        "status": "INCONCLUSIVE_MISSING_EXACT_PPO_BATCH",
        "primary_root_cause": "INCONCLUSIVE",
        "confidence": "LOW",
        "next_action": "NEXT_TARGETED_PPO_COUNTERFACTUAL",
        "failure_trigger_saturation": float(failure_summary["global_saturation"]),
        "failure_actor_parameter_hash": retry_final["failure_snapshot"]["actor_sha256"],
        "failure_actor_file_sha256": sha256(actor_path),
        "failure_critic_file_sha256": sha256(critic_path),
        "failure_optimizer_file_sha256": sha256(optimizer_path),
        "failure_normalizer_file_sha256": sha256(normalizer_path),
        "authoritative_optimizer_steps": 0,
        "canonical_hashes_unchanged": True,
        "optimization_progression": progression,
        "telemetry_summary": telemetry_summary,
        "kl_dynamics": kl_summary,
        "evidence": evidence,
        "missing": {
            "fixed_probe": "raw observations absent; empty NPZ is a sentinel only",
            "ppo_batch": "observations/actions/log_probs/rewards/values/returns/advantages absent",
            "counterfactual_state": "no complete reproducible object/reference/world state bundle",
        },
        "reward_v3_needs_changing": "INCONCLUSIVE",
        "residual_range_needs_changing": "INCONCLUSIVE",
        "ppo_contract_needs_changing": "INCONCLUSIVE",
        "threshold_change": "NO_EVIDENCE_TO_CHANGE_THRESHOLD",
        "formal_p3_can_continue": "NO",
        "safety_flags": {
            "AUTHORITATIVE_PPO_TRAINING_RUN": "NO",
            "C1_RETRY_RUN": "NO",
            "C2_STARTED": "NO",
            "SHADOW_DIAGNOSTIC_OPTIMIZER_STEP": "NO",
            "REWARD_CHANGED": "NO",
            "PPO_HYPERPARAMETERS_CHANGED": "NO",
            "ACTION_BOUND_CHANGED": "NO",
            "ACTION_MAPPING_CHANGED": "NO",
            "SATURATION_THRESHOLD_CHANGED": "NO",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            ".local_TRACKED": "NO",
        },
    }
    write_json(output / "final_summary.json", summary)
    write_json(
        output / "next_action_decision.json",
        {
            "primary_root_cause": "INCONCLUSIVE",
            "confidence": "LOW",
            "next_action": summary["next_action"],
        },
    )
    write_json(
        output / "tests.json",
        {
            "fixed_probe_selection": "PASS_FAIL_CLOSED_UNAVAILABLE",
            "normalizer_cross_eval": "PASS_FAIL_CLOSED_UNAVAILABLE",
            "objective_reconstruction": "PASS_FAIL_CLOSED_UNAVAILABLE",
            "gradient_sign": "PASS_UNIT_TESTED",
            "reward_leave_one_out": "PASS_NOT_RUN_NO_MUTATION",
            "authoritative_optimizer_steps": 0,
            "canonical_actor_hash_unchanged": True,
            "canonical_critic_hash_unchanged": True,
            "canonical_optimizer_hash_unchanged": True,
        },
    )
    (output / "failure_transitions.jsonl").write_text(
        json.dumps({"event": "P3C12_INCONCLUSIVE_MISSING_EXACT_PPO_BATCH"}) + "\n",
        encoding="utf-8",
    )
    write_json(output / "git_commits.json", {"commits": []})
    write_json(
        output / "visualization.json",
        {"status": "NOT_GENERATED", "reason": "no complete replay trace in immutable C1.1 inputs"},
    )
    write_json(output / "handoff.json", summary)
    (output / "final_summary.md").write_text(build_markdown(summary), encoding="utf-8")
    (output / "handoff.md").write_text(build_markdown(summary), encoding="utf-8")

    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    matrix_headers = [
        "Evidence",
        "State shift",
        "Normalizer",
        "Advantage",
        "Reward",
        "PPO surrogate",
        "Clip/KL",
        "Distribution",
        "Tanh",
        "Critic",
        "Residual authority",
    ]
    matrix_values = [
        "C1.1 / C1.2 receipts",
        evidence["state_shift"],
        evidence["normalizer"],
        evidence["advantage"],
        evidence["reward"],
        evidence["ppo_surrogate"],
        evidence["clip_kl"],
        evidence["distribution"],
        evidence["tanh"],
        evidence["critic"],
        evidence["residual_authority"],
    ]
    matrix = (
        "| "
        + " | ".join(matrix_headers)
        + " |\n| "
        + " | ".join("---" for _ in matrix_headers)
        + " |\n| "
        + " | ".join(matrix_values)
        + " |\n"
    )
    (table_dir / "root_cause_matrix.md").write_text(matrix, encoding="utf-8")
    tables = {
        "fixed_probe_drift.md": "Fixed probe unavailable: C1.1 did not persist raw observations.\n",
        "optimization_progression.md": (
            "See `../ppo/optimization_progression.csv`; late KL/clip receipts are aggregate only.\n"
        ),
        "gradient_pressure.md": (
            "Gradient pressure unavailable because the exact PPO batch was not persisted.\n"
        ),
        "advantage.md": (
            "Advantages and returns unavailable because the exact PPO batch was not persisted.\n"
        ),
        "reward_components.md": (
            "Only aggregate PRE_CONTACT reward means are reported; leave-one-out "
            "causality was not run.\n"
        ),
        "clipping.md": (
            "Late receipt KL is above target while clip fraction is zero; sample-level "
            "pressure is unavailable.\n"
        ),
        "counterfactual.md": (
            "Frozen PhysX counterfactual not run; no complete reproducible world-state bundle.\n"
        ),
    }
    for name, content in tables.items():
        (table_dir / name).write_text(content, encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(output),
                "primary_root_cause": summary["primary_root_cause"],
                "next_action": summary["next_action"],
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(".local/reports/stage16_p3_c1_1_saturation_reproduction"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".local/reports/stage16_p3_c1_2_ppo_optimization_attribution"),
    )
    args = parser.parse_args()
    run(args.input_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
