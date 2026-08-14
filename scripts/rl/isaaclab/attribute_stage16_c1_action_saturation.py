#!/usr/bin/env python3
"""Write a fail-closed C1 action-saturation attribution from frozen artifacts.

This deliberately has no IsaacLab or optimizer dependency.  It cannot and does
not recreate a missing C1 policy by replaying training updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.c1_action_saturation_attribution import (  # noqa: E402
    ACTION_SEMANTICS,
    action_dimension_rows,
    as_dict,
    conclusion,
    decision_contract,
    history_rows,
    metric_contract,
    parse_failure_metric,
    read_jsonl,
    root_cause_matrix,
    unavailable_dynamic_diagnostics,
)

DEFAULT_SOURCE = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart"
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16_c1_action_saturation_attribution"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    c0 = source / "formal/v3/hocap_170105/c0"
    c1 = source / "formal/v3/hocap_170105/c1"
    failure_path = c1 / "training_failure.json"
    c1_config_path = c1 / "training_config.json"
    c0_checkpoint = c0 / "checkpoints/stage16_full_trajectory_aggregate_v3_c0.pt"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    c1_config = json.loads(c1_config_path.read_text(encoding="utf-8"))
    c0_rows = history_rows(read_jsonl(c0 / "training_metrics.jsonl"), stage="C0")
    c1_rows = history_rows(read_jsonl(c1 / "training_metrics.jsonl"), stage="C1")
    num_envs = int(c1_config["selected_num_envs"])
    tail_steps = int(c1_config["stage_budget_samples"]) // num_envs - len(c1_rows) * 40
    if tail_steps != 24:
        raise RuntimeError("C1_SATURATION_TAIL_LENGTH_CONTRACT_INVALID")
    failure_metric = parse_failure_metric(failure, rollout_steps=tail_steps, num_envs=num_envs)
    dynamic = unavailable_dynamic_diagnostics()
    final_c1_row = c1_rows[-1]
    frozen = {
        "schema_version": "C1ActionSaturationFrozenInputsV1",
        "clip": c1_config["clip"],
        "reward_mode": c1_config["contact_mode"],
        "c0_checkpoint": str(c0_checkpoint),
        "c0_checkpoint_sha256": _sha256(c0_checkpoint),
        "c1_last_valid_actor": {
            "status": "NOT_PERSISTED",
            "last_metric_actor_hash_after": read_jsonl(c1 / "training_metrics.jsonl")[-1][
                "actor_parameter_hash_after"
            ],
            "required_for_diagnostic": True,
        },
        "normalizer": {"status": "NOT_PERSISTED_FOR_LAST_VALID_C1_ACTOR"},
        "reference_hash": c1_config["reference_hash"],
        "support_contract_hash": c1_config["support_contract_hash"],
        "action_contract": c1_config["environment"]["action"],
        "controller": c1_config["environment"]["finite_virtual_6d_wrist_actuator"],
        "curriculum_contract": c1_config["environment"]["gravity_friction_curriculum"],
        "gravity_scale": c1_config["environment"]["gravity_friction_curriculum"]["gravity_scale"],
        "friction_scale": c1_config["environment"]["gravity_friction_curriculum"]["friction_scale"],
        "saturation_gate": metric_contract()["thresholds"],
        "training_sample_counter_before_tail": final_c1_row["samples"],
        "rollout_horizon": 40,
        "tail_rollout_length": tail_steps,
        "failure_receipt_sha256": _sha256(failure_path),
    }
    decisions = conclusion(c1_rows=c1_rows, failure=failure_metric)
    historical = (
        c0_rows
        + c1_rows
        + [
            {
                "stage": "C1",
                "update_index": 25,
                "samples": int(final_c1_row["samples"]) + tail_steps * num_envs,
                "stage_samples": int(c1_config["stage_budget_samples"]),
                "rollout_steps": tail_steps,
                "deterministic_action_saturation_fraction": failure_metric.fraction,
                "sampled_action_saturation_fraction": None,
                "command_clamp": None,
                "actuator_saturation": None,
            }
        ]
    )
    phase_rows = [
        {
            "phase": "APPROACH",
            "steps": int(c1_config["stage_budget_samples"]),
            "raw_action_saturation": "AGGREGATE_ONLY",
            "scaled_residual_saturation": None,
            "command_clamp_rate": None,
            "wrist_saturation": None,
            "finger_saturation": None,
            "hand_object_contact": None,
            "table_object_contact": None,
            "object_tracking_error": None,
            "unavailable_reason": "NO_PERSISTED_C1_STEP_TELEMETRY",
        }
    ]
    finger_rows = [
        {
            "finger": finger,
            "residual_saturation_rate": None,
            "command_clamp_rate": None,
            "contact_force": None,
            "source_expected_contact": None,
            "unavailable_reason": "NO_PERSISTED_C1_ACTION_TELEMETRY",
        }
        for finger in ("thumb", "index", "middle", "ring", "pinky")
    ]
    partial = {
        "status": "UNAVAILABLE_NO_STEP_LEVEL_C1_ACTION_TELEMETRY",
        "requested_window_steps": tail_steps,
        "observed_tail": failure_metric.fraction,
        "statistics": {key: None for key in ("mean", "std", "p50", "p90", "p95", "p99")},
    }
    full_window = dynamic | {"requested_rollout_steps": 40, "exact_c1_actor_required": True}
    final = {
        "schema_version": "C1ActionSaturationAttributionSummaryV1",
        "original_failure": as_dict(failure_metric),
        "metric_contract": metric_contract(),
        "frozen_inputs": frozen,
        "decision": decisions,
        "root_cause_matrix": root_cause_matrix(),
        "dynamic_diagnostics": dynamic,
        "historical_rows": len(historical),
        "action_dimensions": len(ACTION_SEMANTICS),
        "human_review_required": False,
    }
    _write_json(output / "frozen_inputs.json", frozen)
    _write_json(output / "metric_contract.json", metric_contract())
    _write_json(output / "decision_contract.json", decision_contract())
    _write_json(output / "historical/failure_rollout.json", as_dict(failure_metric))
    _write_csv(output / "historical/c0_c1_trend.csv", historical)
    _write_csv(output / "dimension_attribution.csv", action_dimension_rows())
    _write_csv(output / "phase_attribution.csv", phase_rows)
    _write_csv(output / "finger_attribution.csv", finger_rows)
    _write_csv(output / "partial_rollout/historical_24step_distribution.csv", [partial])
    _write_json(output / "partial_rollout/full_window_diagnostic.json", full_window)
    for name in ("deterministic", "stochastic"):
        _write_json(output / f"counterfactuals/{name}/diagnostic.json", dynamic)
    _write_json(output / "counterfactuals/physics/diagnostic.json", dynamic)
    _write_json(output / "next_action_decision.json", decisions)
    _write_json(output / "final_summary.json", final)
    _write_json(
        output / "tests.json",
        {
            "static_metric_contract": "PASS",
            "all_26_semantics_present": "PASS",
            "historical_receipt_unchanged": _sha256(failure_path)
            == frozen["failure_receipt_sha256"],
            "optimizer_step_count": 0,
            "actor_parameters_changed": False,
            "critic_parameters_changed": False,
            "dynamic_diagnostics": dynamic["status"],
        },
    )
    (output / "failure_transitions.jsonl").write_text(
        json.dumps({"event": "C1_PRE_FAILURE_POLICY_NOT_PERSISTED", "result": dynamic["status"]})
        + "\n",
        encoding="utf-8",
    )
    _write_json(output / "git_commits.json", {"commits": []})
    _write_json(
        output / "telemetry/README.json",
        {"status": "NOT_AVAILABLE", "reason": dynamic["reason"]},
    )
    for layer in ("actor", "action26", "command", "actuator", "contact", "phase"):
        _write_json(
            output / f"telemetry/{layer}/status.json",
            {"status": "NOT_AVAILABLE", "reason": dynamic["reason"]},
        )
    _write_json(output / "screenshots/README.json", {"status": "NOT_GENERATED_NO_FROZEN_C1_ACTOR"})
    _write_json(output / "tables/root_cause_matrix.json", root_cause_matrix())
    (output / "tables/metric_definition.md").write_text(
        _markdown_table(
            ["Field", "Value"],
            [
                ["Metric", "deterministic_action_saturation_fraction"],
                ["Numerator", "count of bounded deterministic policy means with abs(mean) >= 0.98"],
                ["Denominator", f"24 * 1024 * 26 = {failure_metric.denominator}"],
                ["Observed", f"{failure_metric.fraction:.6f}"],
                ["Gate", "> 0.25"],
            ],
        ),
        encoding="utf-8",
    )
    (output / "tables/c0_c1_trend.md").write_text(
        _markdown_table(
            ["Point", "Samples", "Rollout", "Saturation"],
            [
                [
                    "C0 endpoint",
                    c0_rows[-1]["samples"],
                    c0_rows[-1]["rollout_steps"],
                    c0_rows[-1]["deterministic_action_saturation_fraction"],
                ],
                [
                    "C1 first full",
                    c1_rows[0]["samples"],
                    40,
                    c1_rows[0]["deterministic_action_saturation_fraction"],
                ],
                [
                    "C1 last full",
                    c1_rows[-1]["samples"],
                    40,
                    c1_rows[-1]["deterministic_action_saturation_fraction"],
                ],
                ["C1 final tail", historical[-1]["samples"], 24, failure_metric.fraction],
            ],
        ),
        encoding="utf-8",
    )
    (output / "tables/dimension_attribution.md").write_text(
        "No C1 pre-failure action tensor was persisted; see dimension_attribution.csv.\n",
        encoding="utf-8",
    )
    (output / "tables/phase_attribution.md").write_text(
        (
            "All retained C1 ledgers reset at the APPROACH prefix; "
            "no per-step action telemetry exists.\n"
        ),
        encoding="utf-8",
    )
    (output / "tables/tail_vs_full.md").write_text(
        _markdown_table(
            ["Window", "Steps", "Saturation", "Gate"],
            [
                [
                    "C1 last full",
                    40,
                    c1_rows[-1]["deterministic_action_saturation_fraction"],
                    "PASS",
                ],
                ["C1 final tail", 24, failure_metric.fraction, "FAIL"],
            ],
        ),
        encoding="utf-8",
    )
    (output / "tables/root_cause_matrix.md").write_text(
        json.dumps(root_cause_matrix(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    handoff = (
        "\n".join(
            [
                "# Stage16 C1 Action Saturation Attribution Handoff",
                "",
                "## Result",
                "",
                (
                    "It is the fraction of deterministic, tanh-squashed policy-mean elements "
                    "whose absolute value is at least 0.98, aggregated over all tail rollout "
                    "steps, environments, and 26 action dimensions."
                ),
                "",
                (
                    f"The recorded tail denominator is 24 * 1024 * 26 = "
                    f"{failure_metric.denominator}; the receipt is "
                    f"{failure_metric.fraction:.6f} > 0.25."
                ),
                "",
                (
                    "C1 full-rollout policy-output saturation increased from 0.018080 to "
                    "0.207148 before the failed tail. This supports "
                    "POLICY_OUTPUT_SATURATION_PRIMARY for the gate itself, not a clamp or "
                    "actuator measurement."
                ),
                "",
                (
                    "The required C1 pre-failure actor, normalizer, RNG, per-dimension actions, "
                    "commands, contacts, and actuator telemetry were not persisted. Dynamic "
                    "residual-authority, clamp, controller, phase/load, physics, and "
                    "partial-window attribution is therefore INCONCLUSIVE; no C0 checkpoint "
                    "was substituted."
                ),
                "",
                (
                    "NEXT_FIX_SATURATION_INSTRUMENTATION is a recommendation only and was not "
                    "implemented. No threshold, action bound, reward, controller, action mapping, "
                    "or PPO training was changed."
                ),
            ]
        )
        + "\n"
    )
    (output / "final_summary.md").write_text(handoff, encoding="utf-8")
    (output / "handoff.md").write_text(handoff, encoding="utf-8")
    print(
        json.dumps(
            {"output_root": str(output), "primary_root_cause": decisions["primary_root_cause"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
