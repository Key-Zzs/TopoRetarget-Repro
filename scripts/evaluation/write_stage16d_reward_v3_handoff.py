#!/usr/bin/env python3
"""Materialize the final Reward V3 handoff from completed immutable receipts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V3_HANDOFF_JSON_OBJECT_REQUIRED:{path}")
    return value


def _one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"V3_HANDOFF_EXPECTED_ONE:{root}:{pattern}:{len(matches)}")
    return matches[0]


def _last_jsonl(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"V3_HANDOFF_JSONL_EMPTY:{path}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError(f"V3_HANDOFF_JSONL_OBJECT_REQUIRED:{path}")
    return value


def _completed_v3_training(runs_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Select the longest completed V3 segment without hiding prior segments."""

    completed: list[tuple[int, dict[str, Any], dict[str, Any], Path]] = []
    for result_path in sorted(runs_root.glob("*/training_result.json")):
        result = _read(result_path)
        samples = result.get("reward_v3_samples")
        metrics_path = result_path.parent / "training_metrics.jsonl"
        if (
            result.get("status") != "REWARD_V3_TRAINING_SEGMENT_COMPLETE"
            or not isinstance(samples, int)
            or not metrics_path.is_file()
        ):
            continue
        completed.append((samples, result, _last_jsonl(metrics_path), result_path.parent))
    if not completed:
        raise ValueError(f"V3_HANDOFF_COMPLETED_TRAINING_RESULT_MISSING:{runs_root}")
    _, result, metrics, run_root = max(completed, key=lambda item: item[0])
    return result, metrics, run_root


def _rate(suite: dict[str, Any], key: str) -> float:
    value = suite["aggregate"][key]["rate"]
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_HANDOFF_RATE_REQUIRED:{key}")
    return float(value)


def _category(
    *,
    decision: dict[str, Any],
    formal_suite: dict[str, Any],
    v1_suite: dict[str, Any],
    freeflight: dict[str, Any],
) -> str:
    if bool(decision.get("severe_force_farming")):
        return "V3_FORCE_FARMING_FAILURE"
    if (
        decision.get("status") == "V3_EFFECTIVE_AT_4M"
        and _rate(formal_suite, "qualified_success") >= _rate(v1_suite, "qualified_success")
        and freeflight.get("status")
        in {"FREE_FLIGHT_RECATCH_RESOLVED", "FREE_FLIGHT_RECATCH_REDUCED"}
    ):
        return "V3_CONTACT_REWARD_VALIDATED"
    if _rate(formal_suite, "qualified_success") < _rate(v1_suite, "qualified_success"):
        return "V3_CONTACT_REWARD_DEGRADED"
    return "V3_CONTACT_REWARD_NO_CLEAR_GAIN"


def _clip_summary(report_root: Path, simulation_root: Path, clip: str) -> dict[str, Any]:
    v1_root = report_root / "v1_pairforce" / clip
    v1_manifest = _read(v1_root / "pairforce_manifest.json")
    v1_qualification = _read(v1_root / "formal_qualification.json")
    v1_suite = _read(v1_root / "formal_evaluation_suite_v2.json")
    training, training_metrics, training_root = _completed_v3_training(
        report_root / "ppo_v3" / clip / "runs"
    )
    selection = _read(report_root / clip / "dev" / "checkpoint_selection.json")
    decision = _read(report_root / clip / "dev" / "four_m_decision.json")
    formal_root = report_root / clip / "formal"
    qualification_path = _one(formal_root, "v3_formal_selected_*_qualification.json")
    suffix = qualification_path.name.removesuffix("_qualification.json")
    formal_qualification = _read(qualification_path)
    formal_suite = _read(formal_root / f"{suffix}_evaluation_suite_v2.json")
    formal_contact = _read(formal_root / f"{suffix}_contact.json")
    freeflight = _read(formal_root / "v3_vs_v1_freeflight.json")
    simulation_manifest = _read(_one(simulation_root / clip, "*/manifest.json"))
    category = _category(
        decision=decision,
        formal_suite=formal_suite,
        v1_suite=v1_suite,
        freeflight=freeflight,
    )
    return {
        "clip": clip,
        "v1_pairforce": v1_manifest["qualification"],
        "v1_formal_qualification": v1_qualification,
        "v1_formal_metrics": v1_suite["aggregate"],
        "training": training,
        "training_metrics": training_metrics,
        "training_run": str(training_root),
        "selection": selection["selected"],
        "four_m_decision": decision,
        "v3_formal_qualification": formal_qualification,
        "v3_formal_metrics": formal_suite["aggregate"],
        "v3_contact": formal_contact["aggregate"],
        "freeflight": freeflight,
        "simulation": simulation_manifest,
        "result_category": category,
    }


def _markdown(summary: dict[str, Any]) -> str:
    calibration = summary["force_scale_calibration"]
    lines = [
        "# Stage 16-D Reward V3 Pair-Force Unblock and PPO Handoff",
        "",
        f"Overall status: `{summary['overall_status']}`.",
        "",
        "## Exact V1 pair-force contract",
        "",
        "The re-exported tensor is `[321, 20, 5, 3]` in world-frame N, ordered "
        "thumb/index/middle/ring/pinky and captured from the filtered named hand "
        "collision body as force on the active object. Frame 0 is invalid; frames 1--320 "
        "are valid. No aggregate-force reconstruction was used.",
        "",
        "## Shared frozen scale",
        "",
        f"`lambda_c = {calibration['lambda_c_n']:.12g} N`, the pooled V1 positive-contact "
        f"median from {calibration['pooled_positive_contact_statistics']['n']} samples.",
        "",
        "## Clip outcomes",
        "",
        "| Clip | V3 samples | Selected checkpoint samples | Formal qualified | "
        "V1 formal qualified | Free-flight status | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for clip in ("hocap_170650", "hocap_170105"):
        item = summary["clips"][clip]
        training = item["training"]
        selected = item["selection"]
        v3 = item["v3_formal_metrics"]
        v1 = item["v1_formal_metrics"]
        lines.append(
            "| {clip} | {samples} | {chosen} | {v3_ok}/{v3_total} | {v1_ok}/{v1_total} | "
            "{flight} | {result} |".format(
                clip=clip,
                samples=training["reward_v3_samples"],
                chosen=selected["reward_v3_samples"],
                v3_ok=v3["qualified_success"]["pass_count"],
                v3_total=v3["qualified_success"]["total"],
                v1_ok=v1["qualified_success"]["pass_count"],
                v1_total=v1["qualified_success"]["total"],
                flight=item["freeflight"]["status"],
                result=item["result_category"],
            )
        )
    lines.extend(
        [
            "",
            "## Data and replay",
            "",
            "Each clip exports all 20 Formal20 episodes, including failed episodes, as a "
            "reloadable Zarr dataset with Parquet metrics and exact contact provenance. "
            "The replay receipts are recorded in `replay_validation.json`.",
            "",
            "## Next action",
            "",
            "`NEXT_REVIEW_CONTACT_REWARD_CONTRACT`",
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
    names = [name.strip() for name in completed.stdout.splitlines() if name.strip()]
    if len(names) != 1:
        raise ValueError(f"V3_HANDOFF_GPU_QUERY_INVALID:{names}")
    return names[0]


def _resource_usage(clips: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"schema_version": "Stage16DRewardV3ResourceUsageV1", "clips": {}}
    for clip, item in clips.items():
        training = item["training"]
        metrics = item["training_metrics"]
        result["clips"][clip] = {
            "reward_v3_samples": training["reward_v3_samples"],
            "training_status": training["status"],
            "selected_checkpoint_samples": item["selection"]["reward_v3_samples"],
            "num_envs": int(metrics["num_envs"]),
            "last_observed_samples_per_s": float(metrics["samples_per_s"]),
            "gpu": _gpu_name(),
        }
    return result


def _git_commits(repo_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "log", "--format=%H%x09%s", "-12"],
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
        "schema_version": "Stage16DRewardV3GitCommitsV1",
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
    """Preserve transitions and record any repaired training-level failure.

    This handoff can be re-run after a later receipt is produced.  It must not
    erase the preflight transition or conceal a failed continuation merely
    because a bounded retry subsequently completed.
    """

    path = report_root / "failure_transitions.jsonl"
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"V3_HANDOFF_TRANSITION_OBJECT_REQUIRED:{path}")
            rows.append(value)
    for clip in clips:
        run_root = report_root / "ppo_v3" / clip / "runs"
        completed_retry = any(
            _read(result).get("status") == "REWARD_V3_TRAINING_SEGMENT_COMPLETE"
            for result in run_root.glob("*/training_result.json")
        )
        for failure_path in sorted(run_root.glob("*/training_failure.json")):
            failure = _read(failure_path)
            rows.append(
                {
                    "clip": clip,
                    "from": "REWARD_V3_TRAINING",
                    "to": "IMPLEMENTATION_REPAIR",
                    "failure_path": str(failure_path.resolve()),
                    "failure_schema": failure.get("schema_version"),
                    "exception_type": failure.get("exception_type"),
                    "message": failure.get("message"),
                    "subsequent_completed_segment": completed_retry,
                }
            )
    rows.extend(
        {
            "clip": clip,
            "four_m_decision": item["four_m_decision"]["status"],
            "freeflight": item["freeflight"]["status"],
            "result_category": item["result_category"],
        }
        for clip, item in clips.items()
    )
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique[json.dumps(row, sort_keys=True)] = row
    return list(unique.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--replay-validation", type=Path, required=True)
    parser.add_argument("--tests-status", required=True)
    args = parser.parse_args()

    report_root = args.report_root.resolve()
    simulation_root = args.simulation_root.resolve()
    calibration = _read(report_root / "force_scale_calibration.json")
    clips = {
        clip: _clip_summary(report_root, simulation_root, clip)
        for clip in ("hocap_170650", "hocap_170105")
    }
    categories = {item["result_category"] for item in clips.values()}
    if categories == {"V3_CONTACT_REWARD_VALIDATED"}:
        overall = "STAGE16D_REWARD_V3_VALIDATED_BOTH_CLIPS"
    elif "V3_CONTACT_REWARD_VALIDATED" in categories:
        overall = "STAGE16D_REWARD_V3_PARTIAL"
    else:
        overall = "STAGE16D_REWARD_V3_INSUFFICIENT"
    summary = {
        "schema_version": "Stage16DRewardV3FinalHandoffV1",
        "overall_status": overall,
        "force_scale_calibration": calibration,
        "clips": clips,
        "replay_validation": _read(args.replay_validation.resolve()),
        "tests_status": args.tests_status,
        "recommended_next_action": "NEXT_REVIEW_CONTACT_REWARD_CONTRACT",
    }
    (report_root / "final_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = _markdown(summary)
    (report_root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (report_root / "handoff.md").write_text(markdown, encoding="utf-8")
    (report_root / "resource_usage.json").write_text(
        json.dumps(_resource_usage(clips), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_root / "tests.json").write_text(
        json.dumps({"status": args.tests_status}, indent=2) + "\n", encoding="utf-8"
    )
    transitions = _failure_transitions(report_root, clips)
    (report_root / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions), encoding="utf-8"
    )
    (report_root / "git_commits.json").write_text(
        json.dumps(_git_commits(Path(__file__).resolve().parents[2]), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "STAGE16D_REWARD_V3_HANDOFF_WRITTEN", "overall": overall}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
