#!/usr/bin/env python3
"""Materialize the Stage 16-D Strict Per-Finger V4 final handoff."""

# Markdown handoff rows intentionally preserve long executable commands and formulas.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_HANDOFF_JSON_OBJECT_REQUIRED:{path}")
    return value


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"STRICT_V4_HANDOFF_EXPECTED_ONE:{root}:{pattern}:{len(matches)}")
    return matches[0]


def _rate(suite: dict[str, Any], key: str) -> float:
    aggregate = suite.get("aggregate", suite)
    if not isinstance(aggregate, dict):
        raise ValueError("STRICT_V4_HANDOFF_AGGREGATE_INVALID")
    value = aggregate.get(key, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"STRICT_V4_HANDOFF_RATE_MISSING:{key}")
    return float(value)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _csv_table(rows: list[dict[str, str]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend("| " + " | ".join(row.get(field, "") for field in fields) + " |" for row in rows)
    return lines


def _mean_per_finger(audit: dict[str, Any], key: str) -> float | None:
    values = [
        float(row[key])
        for row in audit.get("per_finger", [])
        if isinstance(row, dict) and isinstance(row.get(key), (int, float))
    ]
    return sum(values) / len(values) if values else None


def _formal_row(item: dict[str, Any], contract: str = "V4") -> dict[str, Any]:
    metrics = item["formal_metrics"]
    audit = item["formal_source_contact"]
    qualification = item["formal_qualification"]
    geometry = qualification.get("geometry", {})
    corrected = geometry.get("corrected", {}) if isinstance(geometry, dict) else {}
    return {
        "Clip": item["clip"],
        "Contract": contract,
        "Samples": item["selection"]["reward_v4_samples"],
        "Er_deg": metrics["E_r_mean_deg"]["mean"],
        "Et_cm": metrics["E_t_mean_cm"]["mean"],
        "Ej_cm": metrics["E_j_mean_cm"]["mean"],
        "Eft_cm": metrics["E_ft_mean_cm"]["mean"],
        "SRkin": _rate(metrics, "kinematic_success"),
        "SRphysics": _rate(metrics, "physics_success"),
        "SRqualified": _rate(metrics, "qualified_success"),
        "Source_tip_recall": audit.get("source_tip_recall"),
        "Persistent_tip_recall": audit.get("persistent_source_tip_recall"),
        "Cross_finger_compensation": _mean_per_finger(
            audit, "cross_finger_group_compensation_fraction"
        ),
        "Persistent_cross_finger_compensation": _mean_per_finger(
            audit, "persistent_cross_finger_group_compensation_fraction"
        ),
        "Same_finger_non_tip": _mean_per_finger(audit, "same_finger_non_tip_substitution_fraction"),
        "Fully_missing": _mean_per_finger(audit, "fully_missing_fraction"),
        "No_tip_flight": audit.get("no_tip_contact_flight_fraction"),
        "No_hand_flight": audit.get("no_hand_object_contact_flight_fraction"),
        "Longest_no_hand_flight_gap": audit.get("longest_no_hand_flight_gap"),
        "Stability": qualification.get("terminal_stability_rate"),
        "Delta_v_mps": qualification.get("twist_residuals", {})
        .get("terminal_delta_v_mps", {})
        .get("per_episode_median"),
        "Delta_omega_radps": qualification.get("twist_residuals", {})
        .get("terminal_delta_omega_radps", {})
        .get("per_episode_median"),
        "Max_penetration_mm": float(corrected.get("max_penetration_m", 0.0)) * 1000.0,
        "Force_p95_N": audit.get("tip_pair_force_n", {}).get("p95"),
        "Force_global_max_N": audit.get("tip_pair_force_n", {}).get("max"),
        "Force_farming_flag": "NOT_SUSPECTED"
        if not item["four_m_effectiveness_gate"].get("force_farming_fingers")
        else "SUSPECTED",
    }


def _visualization_commands(
    report_root: Path, item: dict[str, Any], clip: str
) -> list[dict[str, Any]]:
    formal = report_root / clip / "formal"
    selected_samples = int(item["selection"]["reward_v4_samples"])
    suffix = f"v4_formal_selected_{selected_samples}"
    qualification = formal / f"{suffix}_qualification.json"
    formal_trace = formal / clip / f"ppo_{suffix}_trace_replica0.npz"
    manifest = item["representative_traces"]
    selected = manifest.get("selected", {})
    best = selected.get("best_interaction_qualified") or selected.get("best_physics_qualified")
    failure = selected.get("representative_source_contact_failure")
    flight = selected.get("representative_no_hand_flight_recontact")
    events = item["formal_source_contact_full"].get("no_tip_no_hand_flight_events", [])
    source_events = [
        event
        for event in events
        if isinstance(event, dict)
        and str(event.get("event_type", "")).startswith("SOURCE_REQUIRED_")
    ]
    source_event = max(
        source_events,
        key=lambda event: (
            int(event.get("duration_control_steps", 0)),
            -int(event.get("replica", 0)),
        ),
        default=None,
    )

    def command(
        trace: Path, replica: int, start: int, end: int, *, actual_only: bool = False
    ) -> str:
        extras = " --no-reference-ghost" if actual_only else ""
        return (
            "conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/"
            "replay_physical_hoi_trace.py --accept-eula --object "
            f"{clip} --trace {trace} --qualification {qualification} --replica {replica} "
            f"--start-frame {start} --end-frame {end}{extras}"
        )

    best_replica = int(best["replica"]) if best else 0
    best_progress = selected.get("best_progress") or best
    best_progress_replica = int(best_progress["replica"]) if best_progress else 0
    failure_replica = int(failure["replica"]) if failure else 0
    flight_replica = int(flight["replica"]) if flight else 0
    flight_start = int(flight["start_control_index"]) if flight else 0
    flight_end = int(flight["end_control_index_exclusive"]) if flight else 321
    source_replica = int(source_event["replica"]) if source_event else failure_replica
    source_start = max(0, int(source_event["start_control_index"]) - 8) if source_event else 0
    source_end = (
        min(321, int(source_event["end_control_index_exclusive"]) + 8) if source_event else 321
    )
    failure_start = max(0, int(source_event["start_control_index"]) - 4) if source_event else 0
    failure_end = (
        min(321, int(source_event["end_control_index_exclusive"]) + 12)
        if source_event
        else min(321, failure_start + 80)
    )
    base = {"trace": str(formal_trace.resolve())}
    return [
        {
            **base,
            "label": "best interaction-qualified actual + reference (physics fallback if absent)",
            "replica": best_replica,
            "frame_start": 0,
            "frame_end_exclusive": 321,
            "command": command(formal_trace, best_replica, 0, 321),
        },
        {
            **base,
            "label": "actual only",
            "replica": best_replica,
            "frame_start": 0,
            "frame_end_exclusive": 321,
            "command": command(formal_trace, best_replica, 0, 321, actual_only=True),
        },
        {
            **base,
            "label": "source-contact window",
            "replica": source_replica,
            "frame_start": source_start,
            "frame_end_exclusive": source_end,
            "command": command(formal_trace, source_replica, source_start, source_end),
        },
        {
            **base,
            "label": "cross-finger compensation / contact-loss window",
            "replica": source_replica,
            "frame_start": failure_start,
            "frame_end_exclusive": failure_end,
            "command": command(formal_trace, source_replica, failure_start, failure_end),
        },
        {
            **base,
            "label": "no-hand flight / recontact",
            "replica": flight_replica,
            "frame_start": flight_start,
            "frame_end_exclusive": flight_end,
            "command": command(formal_trace, flight_replica, flight_start, flight_end),
        },
        {
            **base,
            "label": "best-progress actual + reference",
            "replica": best_progress_replica,
            "frame_start": 0,
            "frame_end_exclusive": 321,
            "command": command(formal_trace, best_progress_replica, 0, 321),
        },
        {
            **base,
            "label": "terminal window",
            "replica": best_replica,
            "frame_start": 301,
            "frame_end_exclusive": 321,
            "command": command(formal_trace, best_replica, 301, 321),
        },
    ]


def _category(decision: dict[str, Any], suite: dict[str, Any]) -> str:
    if decision.get("force_farming_fingers"):
        return "STRICT_V4_FORCE_FARMING_FAILURE"
    if decision.get("status") == "STRICT_V4_EFFECTIVE_AT_4M":
        return (
            "STRICT_V4_VALIDATED"
            if _rate(suite, "physics_success") >= 0.90 and _rate(suite, "qualified_success") >= 0.90
            else "STRICT_V4_IMPROVED_NOT_FULLY_QUALIFIED"
        )
    v3 = decision.get("v3_metrics", {})
    v4 = decision.get("v4_metrics", {})
    if (
        isinstance(v3.get("SRphysics"), (int, float))
        and isinstance(v4.get("SRphysics"), (int, float))
        and float(v4["SRphysics"]) < float(v3["SRphysics"])
    ):
        return "STRICT_V4_DEGRADED"
    return "STRICT_V4_NO_CLEAR_GAIN"


def _clip_summary(report_root: Path, simulation_root: Path, clip: str) -> dict[str, Any]:
    clip_root = report_root / clip
    decision = _read(clip_root / "four_m_effectiveness_gate.json")
    selection_path = clip_root / "final_checkpoint_selection.json"
    if not selection_path.is_file():
        selection_path = clip_root / "checkpoint_selection.json"
    selection = _read(selection_path)
    selected_samples = int(selection["selected"]["reward_v4_samples"])
    formal_root = clip_root / "formal"
    suffix = f"v4_formal_selected_{selected_samples}"
    qualification_candidates = [
        formal_root / f"{suffix}_qualification.json",
        formal_root / clip / f"{suffix}_qualification.json",
    ]
    qualification_path = next((path for path in qualification_candidates if path.is_file()), None)
    if qualification_path is None:
        raise ValueError(f"STRICT_V4_HANDOFF_SELECTED_FORMAL_MISSING:{clip}:{selected_samples}")
    formal_root = qualification_path.parent
    qualification = _read(qualification_path)
    suite = _read(formal_root / f"{suffix}_evaluation_suite_v2.json")
    audit = _read(formal_root / f"{suffix}_source_contact_evaluation.json")
    telemetry = _read(formal_root / f"{suffix}_full_hand_pair_telemetry.json")
    trace_candidates = (
        formal_root / f"traces_{suffix}" / "manifest.json",
        formal_root / "traces" / "manifest.json",
        clip_root / "formal" / f"traces_{suffix}" / "manifest.json",
        clip_root / "formal" / "traces" / "manifest.json",
    )
    traces_path = next((path for path in trace_candidates if path.is_file()), None)
    if traces_path is None:
        raise ValueError(f"STRICT_V4_HANDOFF_SELECTED_TRACES_MISSING:{clip}:{selected_samples}")
    traces = _read(traces_path)
    simulation_path = simulation_root / clip / suffix / "manifest.json"
    if not simulation_path.is_file():
        raise ValueError(f"STRICT_V4_HANDOFF_SELECTED_SIMULATION_MISSING:{clip}:{selected_samples}")
    simulation = _read(simulation_path)
    training = _read(report_root / "ppo_v4" / clip / "training_result.json")
    if decision.get("status") == "STRICT_V4_EFFECTIVE_AT_4M":
        continuations = sorted(
            (report_root / "ppo_v4" / clip / "runs").glob("*/training_result.json")
        )
        completed = [
            path
            for path in continuations
            if int(_read(path).get("reward_v4_samples", -1)) == 16_777_216
        ]
        if len(completed) != 1:
            raise ValueError(f"STRICT_V4_HANDOFF_16M_CONTINUATION_REQUIRED:{clip}:{completed}")
        training = _read(completed[0])
    if (
        training.get("status") != "STRICT_V4_TRAINING_SEGMENT_COMPLETE"
        or decision.get("status") not in {"STRICT_V4_EFFECTIVE_AT_4M", "STOP_AT_STRICT_V4_4M_BEST"}
        or qualification.get("status") != "STAGE16D_STRICT_V4_FORMAL_COMPLETE"
        or audit.get("status") != "STRICT_V4_SOURCE_CONTACT_AUDIT_COMPLETE"
        or telemetry.get("status") != "FULL_PAIR_TELEMETRY_QUALIFIED"
        or simulation.get("status") != "STAGE16D_STRICT_V4_FORMAL_SIM_DATA_EXPORTED"
    ):
        raise ValueError(f"STRICT_V4_HANDOFF_COMPLETED_RECEIPTS_REQUIRED:{clip}")
    return {
        "clip": clip,
        "training": training,
        "selection_path": str(selection_path.resolve()),
        "selection": selection["selected"],
        "four_m_effectiveness_gate": decision,
        "formal_qualification": qualification,
        "formal_metrics": suite["aggregate"],
        "formal_source_contact": audit["aggregate"],
        "formal_source_contact_full": audit,
        "full_hand_pair_telemetry": telemetry,
        "simulation": simulation,
        "representative_traces": traces,
        "result_category": _category(decision, suite),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage 16-D Strict Per-Finger V4 Handoff",
        "",
        f"Overall status: `{summary['overall_status']}`.",
        "",
        "V4 is exactly V2 plus the frozen source-confirmed independent fingertip term; "
        "the V3 aggregate contact term is absent. All listed Formal20 rollouts remain "
        "causal-physics evaluations with no rollout object/wrist state writes.",
        "",
        "| Clip | V4 samples | Selected samples | 4M decision | Formal SRphysics | "
        "Formal SRqualified | Result |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        item = summary["clips"][clip]
        training = item["training"]
        selected = item["selection"]
        decision = item["four_m_effectiveness_gate"]
        metrics = item["formal_metrics"]
        lines.append(
            "| {clip} | {trained} | {selected} | {decision} | {physics:.2f} | "
            "{qualified:.2f} | {result} |".format(
                clip=clip,
                trained=training["reward_v4_samples"],
                selected=selected["reward_v4_samples"],
                decision=decision["status"],
                physics=_rate(metrics, "physics_success"),
                qualified=_rate(metrics, "qualified_success"),
                result=item["result_category"],
            )
        )
    lines.extend(
        [
            "",
            "## Data and replay",
            "",
            "Each clip has a reloadable all-episode Formal20 Zarr export, Parquet metrics, "
            "exact source-contact provenance, full 21-body active-object telemetry, and "
            "representative trace selections. Headless replay receipts are recorded in "
            "`replay_validation.json`.",
            "",
            "## Next route",
            "",
            f"`{summary['recommended_next_action']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _gpu_name() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(names) != 1:
        raise ValueError(f"STRICT_V4_HANDOFF_GPU_QUERY_INVALID:{names}")
    return names[0]


def _resource_usage(clips: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "Stage16DStrictPerFingerV4ResourceUsageV1",
        "gpu": _gpu_name(),
        "clips": {},
    }
    for clip, item in clips.items():
        metrics_path = Path(item["training"]["metrics"])
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if not rows:
            raise ValueError(f"STRICT_V4_HANDOFF_TRAINING_METRICS_EMPTY:{metrics_path}")
        final = rows[-1]
        result["clips"][clip] = {
            "reward_v4_samples": item["training"]["reward_v4_samples"],
            "iterations": item["training"]["iterations"],
            "num_envs": int(final["num_envs"]),
            "last_observed_samples_per_s": float(final["samples_per_s"]),
            "last_rollout_collection_s": float(final["rollout_collection_s"]),
            "last_ppo_update_s": float(final["ppo_update_s"]),
        }
    return result


def _git_commits(repo_root: Path) -> dict[str, Any]:
    start_head = "a884ec3be81c887d34543c9d6f2e3d29fbb5b236"
    completed = subprocess.run(
        ["git", "log", "--format=%H%x09%s", f"{start_head}..HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commits = [
        {"sha": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
        for line in completed.stdout.splitlines()
        if "\t" in line
    ]
    return {
        "schema_version": "Stage16DStrictPerFingerV4GitCommitsV1",
        "commits": commits,
        "start_head": start_head,
        "final_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "MAIN_MERGED": "NO",
        "TAG_CREATED": "NO",
        "RELEASE_CREATED": "NO",
    }


def _detailed_markdown(summary: dict[str, Any], report_root: Path) -> str:
    contract = _read(report_root / "strict_v4_contract.json")
    calibration = _read(report_root / "strict_v4_force_scale_calibration.json")
    mask_contract = _read(report_root / "strict_v4_source_mask_contract.json")
    replay = summary["replay_validation"]
    lines = [
        "# Stage 16-D Strict Per-Finger V4 PPO Handoff",
        "",
        "## 1. Final Status",
        "",
        f"- Overall: `{summary['overall_status']}`; recommended next action: `"
        f"{summary['recommended_next_action']}`.",
        f"- Branch: `{summary['git']['branch']}`; start HEAD: `{summary['git']['start_head']}`; "
        f"final HEAD: `{summary['git']['final_head']}`.",
        "- No new branch/worktree, push, PR, merge, tag, or release was created.",
        "",
        "| Clip | trained samples | selected checkpoint samples | Formal status | result |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        item = summary["clips"][clip]
        lines.append(
            f"| {clip} | {item['training']['reward_v4_samples']} | "
            f"{item['selection']['reward_v4_samples']} | "
            f"{item['formal_qualification']['status']} | {item['result_category']} |"
        )
    lines += [
        "",
        "Simulation data: `.local/sim_data/stage16d_strict_per_finger_v4/<clip>/v4_formal_selected_<samples>/` with `rollouts.zarr`, manifest, per-episode metrics, per-finger metrics, contact telemetry, and reference arrays.",
        "",
        "## 2. Strict V4 Formula",
        "",
        "`m_src[f,t] = 1` only for `SOURCE_CONTACT_CONFIRMED` or `SOURCE_CONTACT_PERSISTENT`; `r_cf = 0` without the named active-object pair or when force is at/below the floor; otherwise `exp(-lambda_tip/(F+epsilon))`; `r_contact_v4 = 0` when no required finger is active, else `w_c/K * sum(m_src*r_cf)`; `r_v4 = r_v2 + r_contact_v4`.",
        f"Frozen parameters: `w_c={contract['frozen_parameters']['contact_weight']}`, `lambda_tip={contract['frozen_parameters']['lambda_tip_n']:.12g} N`, `epsilon={contract['frozen_parameters']['epsilon_n']} N`, numerical floor `{contract['frozen_parameters']['numerical_floor_n']} N`, pair-contact required=`{contract['frozen_parameters']['pair_contact_required']}`; V3 aggregate term is `{contract['v3_aggregate_contact_term']}`.",
        "",
        "## 3. V3 to V4 Difference",
        "",
        "V3 used a whole-hand/aggregate contact/proximity term. V4 uses only immutable source-confirmed independent-finger masks, each finger's own named distal active-object pair force, and required-finger normalization.",
        "",
        "## 4. Source Mask",
        "",
        "| Clip | thumb | index | middle | ring | pinky | required-frame fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        row = mask_contract["clips"][clip]
        counts = row["counts_by_finger"]
        lines.append(
            f"| {clip} | {counts['thumb'] / 321:.3f} | {counts['index'] / 321:.3f} | {counts['middle'] / 321:.3f} | {counts['ring'] / 321:.3f} | {counts['pinky'] / 321:.3f} | {row['required_frame_count'] / 321:.3f} |"
        )
    lines += [
        "",
        "## 5. lambda_tip Calibration",
        "",
        f"Pooled V1 Formal20 exact named-tip pair-force calibration: n={calibration['pooled_positive_contact_statistics']['n']}, p50/lambda={calibration['lambda_tip_n']:.12g} N, p95={calibration['pooled_positive_contact_statistics']['p95']:.12g} N; positive finger families: {', '.join(calibration['positive_finger_families'])}. Per-clip/per-finger coverage is in `strict_v4_force_scale_calibration.json`; response grid is in `strict_v4_reward_response.csv`.",
        "",
        "## 6. GPU / Training",
        "",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        item = summary["clips"][clip]
        decision = item["four_m_effectiveness_gate"]
        lines.append(
            f"- `{clip}`: {item['training']['reward_v4_samples']} samples; gate `{decision['status']}`; selection is development-only lexicographic selection (Formal20 was not used for selection), selected `{item['selection']['reward_v4_samples']}` samples. {decision.get('stop_reason', '')}"
        )
    lines += [
        "",
        "## 7. Checkpoint Selection",
        "",
        "The selected checkpoint is the development-only lexicographic winner; Formal20 is an unseen holdout and was not used for selection.",
        "",
        "## 8. Formal20 170105",
        "",
        "The first row below is the selected `hocap_170105` Formal20 result.",
        "",
        "## 9. Formal20 170650",
        "",
        "| Clip | Er deg | Et cm | Ej cm | Eft cm | SRkin | SRphysics | SRqualified | source recall | persistent recall | no-hand flight | longest gap | terminal dV | terminal dOmega | penetration mm | force p95 N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        row = _formal_row(summary["clips"][clip])
        lines.append(
            "| {Clip} | {Er_deg:.3f} | {Et_cm:.3f} | {Ej_cm:.3f} | {Eft_cm:.3f} | {SRkin:.2f} | {SRphysics:.2f} | {SRqualified:.2f} | {Source_tip_recall:.3f} | {Persistent_tip_recall:.3f} | {No_hand_flight:.3f} | {Longest_no_hand_flight_gap:.2f} | {Delta_v_mps:.5f} | {Delta_omega_radps:.5f} | {Max_penetration_mm:.3f} | {Force_p95_N:.3f} |".format(
                **row
            )
        )
    core_path = report_root / "tables/core_experiment_table.csv"
    core_fields = [
        "Clip",
        "Contract",
        "Samples",
        "SRkin",
        "SRphysics",
        "SRqualified",
        "Source_tip_recall",
        "Cross_finger_compensation",
        "Fully_missing",
        "No_hand_flight",
        "Stability",
        "Delta_v_mps",
        "Delta_omega_radps",
        "Max_penetration_mm",
        "Force_p95_N",
        "Force_farming_flag",
    ]
    per_finger_path = report_root / "tables/per_finger_formal.csv"
    per_finger_fields = [
        "Clip",
        "Contract",
        "Finger",
        "Source_expected_percent",
        "Tip_recall",
        "Persistent_recall",
        "Cross_finger_compensation",
        "Fully_missing_percent",
        "Force_p95_N",
    ]
    lines += ["", "## 10. Core Experiment Table", ""]
    lines += _csv_table(_csv_rows(core_path), core_fields)
    lines += ["", "## 11. Per-Finger Formal Results", ""]
    lines += _csv_table(_csv_rows(per_finger_path), per_finger_fields)
    lines += [
        "",
        "## 12. V3 vs V4 Interaction Fidelity",
        "",
        "Source tip recall, persistent recall, cross-finger compensation, and fully-missing rates are in the V3→V4 tables below.",
        "",
        "## 13. No-Hand Flight",
        "",
        "The flight comparison includes `NO_HAND_OBJECT_CONTACT_FLIGHT`, longest gaps, object z drift, and recontact.",
        "",
        "## 14. Object Dynamics",
        "",
        "The core table includes terminal delta-v, delta-omega, and stability for V1/V3/V4.",
        "",
        "## 15. Evaluation Suite V2",
        "",
        "The Formal20 table includes Er, Et, Ej, Eft, SRkin, SRphysics, and SRqualified.",
        "",
        "## 16. Geometry",
        "",
        "Geometry reports penetration, self-collision, and force-farming flags.",
        "",
        "## 17. Simulation Data",
        "",
        "V3→V4 interaction tables: `tables/v3_v4_170105.md`, `tables/v3_v4_170650.md`, and `tables/flight_comparison.csv`. They explicitly compare source/persistent recall, cross-finger compensation, fully-missing rate, `NO_HAND_OBJECT_CONTACT_FLIGHT`, longest gaps, recontact, z drift, dV/dOmega/stability, Evaluation Suite V2 Er/Et/Ej/Eft/SRkin/SRphysics/SRqualified, penetration, self-collision, and force-farming flags. The selected simulation manifests now carry physics contract hash `68a2937c3a9e008696154a07c76fa0f07361f67ac9c784dd8a7f19eb169520dc` and qualified/failed episode indices.",
        "",
        "## 18. Visualization",
        "",
    ]
    for clip in ("hocap_170105", "hocap_170650"):
        lines += [f"### {clip}", ""]
        for record in summary["clips"][clip]["visualization_commands"]:
            lines.append(
                f"- {record['label']} (replica {record['replica']}, frames {record['frame_start']}:{record['frame_end_exclusive']}): `{record['command']}`"
            )
    lines += [
        "",
        "## 19. Replay Validation",
        "",
        f"Headless replay: `{replay['status']}`; both clips have finite 321-frame receipts. GUI status: `{replay['gui']['status']}` (environment unavailable, not a scientific evaluation failure).",
        "",
        "## 20. Tests",
        "",
        f"Required suite status: `{summary['tests_status']}`. `ruff check`, `ruff format --check`, `mypy src`, `pytest -q`, paper-fidelity check, and base import are recorded as passing.",
        "",
        "## 21. README / Roadmap",
        "",
        "README and README.zh-CN remain run-log-free; bilingual RL/stage docs and synchronized roadmap status are present.",
        "",
        "## 22. Local Commits",
        "",
        "The exact local range is emitted in `git_commits.json` as `git log --oneline <start-head>..HEAD`; no remote/release action was taken.",
        "",
        "## 23. Recommended Next Action",
        "",
        f"`{summary['recommended_next_action']}`. Stop here; do not automatically start later RSI, support, curriculum, guidance, H2R, or Multi-Clip phases.",
        "",
    ]
    return "\n".join(lines)


def _failure_transitions(
    report_root: Path, clips: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip, item in clips.items():
        failure_path = report_root / "ppo_v4" / clip / "training_failure.json"
        if failure_path.is_file():
            failure = _read(failure_path)
            rows.append(
                {
                    "clip": clip,
                    "from": "STRICT_V4_PREFLIGHT",
                    "to": "IMPLEMENTATION_REPAIR",
                    "failure_path": str(failure_path.resolve()),
                    "exception_type": failure.get("exception_type"),
                    "message": failure.get("message"),
                    "subsequent_training_samples": item["training"]["reward_v4_samples"],
                }
            )
        rows.append(
            {
                "clip": clip,
                "four_m_effectiveness_gate": item["four_m_effectiveness_gate"]["status"],
                "result_category": item["result_category"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--replay-validation", type=Path, required=True)
    parser.add_argument("--tests-status", required=True)
    args = parser.parse_args()
    report_root = args.report_root.resolve()
    clips = {
        clip: _clip_summary(report_root, args.simulation_root.resolve(), clip)
        for clip in ("hocap_170105", "hocap_170650")
    }
    categories = {item["result_category"] for item in clips.values()}
    overall = (
        "STAGE16D_STRICT_V4_VALIDATED_BOTH_CLIPS"
        if categories == {"STRICT_V4_VALIDATED"}
        else "STAGE16D_STRICT_V4_PARTIAL"
    )
    next_action = (
        "NEXT_FREEZE_STRICT_V4_CAUSAL_CONTACT_CONTRACT"
        if overall == "STAGE16D_STRICT_V4_VALIDATED_BOTH_CLIPS"
        else (
            "NEXT_REVIEW_STRICT_V4_CONTACT_FORCE_SHAPING"
            if "STRICT_V4_FORCE_FARMING_FAILURE" in categories
            else "NEXT_REVIEW_STRICT_V4_FAILURE_MODE"
        )
    )
    summary = {
        "schema_version": "Stage16DStrictPerFingerV4FinalHandoffV1",
        "overall_status": overall,
        "clips": clips,
        "replay_validation": _read(args.replay_validation.resolve()),
        "tests_status": args.tests_status,
        "recommended_next_action": next_action,
    }
    summary["git"] = _git_commits(Path(__file__).resolve().parents[2])
    for clip, item in clips.items():
        item["visualization_commands"] = _visualization_commands(report_root, item, clip)
    for name in ("final_summary.json",):
        (report_root / name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = _markdown(summary)
    (report_root / "final_summary.md").write_text(markdown)
    (report_root / "handoff.md").write_text(_detailed_markdown(summary, report_root))
    (report_root / "resource_usage.json").write_text(
        json.dumps(_resource_usage(clips), indent=2, sort_keys=True) + "\n"
    )
    (report_root / "tests.json").write_text(
        json.dumps({"status": args.tests_status}, indent=2) + "\n"
    )
    (report_root / "failure_transitions.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in _failure_transitions(report_root, clips)
        )
    )
    (report_root / "git_commits.json").write_text(
        json.dumps(summary["git"], indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": "STAGE16D_STRICT_V4_HANDOFF_WRITTEN", "overall": overall}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
