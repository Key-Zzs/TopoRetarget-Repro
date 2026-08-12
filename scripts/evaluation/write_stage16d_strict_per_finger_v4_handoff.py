#!/usr/bin/env python3
"""Materialize the Stage 16-D Strict Per-Finger V4 final handoff."""

from __future__ import annotations

import argparse
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


def _category(decision: dict[str, Any], suite: dict[str, Any]) -> str:
    if decision.get("force_farming_fingers"):
        return "STRICT_V4_FORCE_FARMING_FAILURE"
    if decision.get("status") == "STRICT_V4_EFFECTIVE_AT_4M":
        return (
            "STRICT_V4_VALIDATED"
            if _rate(suite, "qualified_success") > 0.0
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
        "training": training,
        "selection_path": str(selection_path.resolve()),
        "selection": selection["selected"],
        "four_m_effectiveness_gate": decision,
        "formal_qualification": qualification,
        "formal_metrics": suite["aggregate"],
        "formal_source_contact": audit["aggregate"],
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
    completed = subprocess.run(
        ["git", "log", "--format=%H%x09%s", "-30"],
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
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "MAIN_MERGED": "NO",
        "TAG_CREATED": "NO",
        "RELEASE_CREATED": "NO",
    }


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
    for name in ("final_summary.json",):
        (report_root / name).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = _markdown(summary)
    (report_root / "final_summary.md").write_text(markdown)
    (report_root / "handoff.md").write_text(markdown)
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
        json.dumps(_git_commits(Path(__file__).resolve().parents[2]), indent=2, sort_keys=True)
        + "\n"
    )
    print(json.dumps({"status": "STAGE16D_STRICT_V4_HANDOFF_WRITTEN", "overall": overall}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
