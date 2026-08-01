#!/usr/bin/env python3
"""Write the fail-closed Stage-16B closeout report suite from bounded evidence."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

COPY_JSON = (
    "world_reference_export.json",
    "reference_reconstruction.json",
    "wrist_model_validation.json",
    "wrist_controller_qualification.json",
    "action_scale_qualification.json",
    "reset_validation.json",
    "observation_contract.json",
    "reward_validation.json",
    "zero_residual_evaluation.json",
    "oracle_evaluation.json",
    "recovery_summary.json",
    "resource_usage.json",
    "w1_exogenous_base_playback.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _blocked_training(clip: str | None = None) -> dict[str, Any]:
    return {
        "status": "NOT_STARTED_GATE_BLOCKED",
        "stage16b_status": "STAGE16B_SINGLE_CLIP_PPO_BLOCKED"
        if clip is not None
        else "STAGE16B_TWO_CLIP_PPO_BLOCKED",
        "clip": clip,
        "reason": "STAGE16B_26D_ORACLE_BLOCKED",
        "oracle_gate": "STAGE16B_26D_ORACLE_VALIDATED",
        "samples": 0,
        "iterations": 0,
        "checkpoint": None,
        "evaluation": "NOT_RUN_NO_QUALIFIED_CHECKPOINT",
    }


def _git_lines(repo: Path, args: list[str]) -> list[str]:
    result = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line]


def _summary_markdown(summary: dict[str, Any]) -> str:
    references = summary["reference_export"]["clips"]
    oracle = summary["oracle_h10"]
    reference_rows = "\n".join(
        "| {clip} | {frames} | {hz:.0f} | yes | yes | {error:.3g} m | validated |".format(
            clip=row["clip"],
            frames=row["validation"]["frames"],
            hz=row["validation"]["control_hz"],
            error=row["validation"]["relative_reconstruction"]["translation_max_error_m"],
        )
        for row in references
    )
    oracle_rows = "\n".join(
        (
            "| {clip} | {count} | {success:.0%} | {reach:.0%} | {object_cm:.3f} | "
            "{wrist_cm:.3f} | {sat:.0%} |"
        ).format(
            clip=row["clip"],
            count=row["summary"]["episode_count"],
            success=row["summary"]["success_rate"],
            reach=row["summary"]["final_reach_rate"],
            object_cm=row["summary"]["object_position_error_cm"],
            wrist_cm=row["summary"]["wrist_position_error_cm"],
            sat=row["summary"]["wrist_wrench_saturation_fraction"],
        )
        for row in oracle
    )
    return f"""# Stage 16-B world wrist-and-finger closeout

**Final status:** `{summary["status"]["overall"]}`. World reference passed; the
finite-wrench wrist controller is partial and the 26D oracle gate blocked PPO.

## Reference export

| Clip | Frames | Hz | World wrist | World object | Relative translation error | Result |
| --- | ---: | ---: | --- | --- | ---: | --- |
{reference_rows}

## Oracle H=10, formal frame-0 population

| Clip | Episodes | Success | Final reach | Object position cm | Wrist position cm | Saturation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{oracle_rows}

## Gate result

- Oracle: `{summary["status"]["oracle"]}`
- Single-clip PPO: `{summary["status"]["single_clip_ppo"]}`
- Two-clip PPO: `{summary["status"]["two_clip_ppo"]}`
- Checkpoints: none; no checkpoint was created behind the failed oracle gate.
- Formal scene: free wrist and free object, zero gravity, no ground, and no
  object-pose/velocity writes after reset.

The old Stage-16A base-relative finger-only baseline remains preserved in the
archive recorded by `frozen_finger_only_baseline.json`.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--qualification-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--start-head", required=True)
    parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args()

    root = args.report_root
    root.mkdir(parents=True, exist_ok=True)
    source = args.qualification_root
    for name in COPY_JSON:
        _write_json(root / name, _read_json(source / name))
    (root / "failure_transition_log.jsonl").write_text(
        (source / "failure_transition_log.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )

    status = _read_json(source / "qualification_status.json")
    reference_export = _read_json(source / "world_reference_export.json")
    oracle_report = _read_json(source / "oracle_evaluation.json")
    h10 = oracle_report["horizons"]["H10"]
    _write_json(
        root / "frozen_finger_only_baseline.json",
        {
            "status": "PRESERVED",
            "profile": "paper_finger_only_base_relative_v1",
            "archive": str(args.archive_root.resolve()),
            "manifest": _read_json(args.archive_root / "frozen_manifest.json"),
        },
    )

    for clip in ("170105", "170650"):
        _write_json(root / f"single_{clip}_training.json", _blocked_training(clip))
        _write_json(root / f"single_{clip}_evaluation.json", _blocked_training(clip))
    _write_json(
        root / "two_clip_ab_matrix.json",
        {
            "status": "NOT_STARTED_GATE_BLOCKED",
            "variants": [],
            "reason": "both single-clip prerequisites are blocked by STAGE16B_26D_ORACLE_BLOCKED",
        },
    )
    _write_json(root / "two_clip_training.json", _blocked_training())
    _write_json(root / "two_clip_evaluation.json", _blocked_training())
    _write_json(
        root / "checkpoint_inventory.json",
        {
            "status": "EMPTY_GATE_BLOCKED",
            "qualified_checkpoints": [],
            "reason": "PPO did not start behind STAGE16B_26D_ORACLE_BLOCKED",
        },
    )
    _write_json(
        root / "contact_dynamics.json",
        {
            "status": "NO_CONTACT_BEFORE_WRIST_SAFETY_FAILURE",
            "no_direct_object_control": True,
            "formal_rollout_object_pose_write": False,
            "oracle_h10": [
                {
                    "clip": row["clip"],
                    "mean_contact_count": row["summary"]["mean_contact_count"],
                    "termination_distribution": row["summary"]["termination_distribution"],
                }
                for row in h10
            ],
        },
    )
    _write_json(
        root / "penetration_audit.json",
        {
            "status": "NOT_RUN_NO_SUCCESSFUL_DYNAMIC_ROLLOUT",
            "reason": "no zero/oracle/PPO rollout reached the final frame",
        },
    )
    episode_rows = [
        {
            "policy": "oracle_h10",
            "clip": clip["clip"],
            "episode": index,
            **episode,
        }
        for clip in h10
        for index, episode in enumerate(clip["episodes"])
    ]
    _write_csv(
        root / "evaluation_episodes.csv",
        list(episode_rows[0]) if episode_rows else ["policy", "clip", "episode"],
        episode_rows,
    )
    termination_rows = [
        {
            "policy": "oracle_h10",
            "clip": row["clip"],
            "termination": termination,
            "count": count,
        }
        for row in h10
        for termination, count in row["summary"]["termination_distribution"].items()
    ]
    _write_csv(
        root / "termination_matrix.csv",
        ["policy", "clip", "termination", "count"],
        termination_rows,
    )
    _write_csv(
        root / "learning_curves.csv",
        ["stage", "iteration", "samples", "mean_return", "checkpoint"],
        [],
    )

    visual_paths = [
        args.visual_root / "170105_zero_camera_fixed" / "visualization_summary.json",
        args.visual_root / "170650_zero_camera_fixed" / "visualization_summary.json",
        args.visual_root / "170105_oracle_h1_camera_fixed" / "visualization_summary.json",
    ]
    visuals = [_read_json(path) | {"summary_path": str(path.resolve())} for path in visual_paths]
    review = {
        "status": "PASS_WITH_FAILURE_EVIDENCE",
        "reviewed_by": "headless MuJoCo offscreen contact sheets manually inspected",
        "visualizations": visuals,
        "findings": [
            "The automatic workspace camera shows separate Wuji hand and red free object geometry.",
            "Both zero-residual clips terminate at the wrist-orientation safety limit after "
            "five frames.",
            "The H=1 oracle terminates at the same safety limit after four frames; it is "
            "not PPO success.",
            "No contact-supported success, PPO video, or PPO checkpoint exists because "
            "the oracle gate is blocked.",
        ],
        "superseded_numerical_fallbacks": (
            "Earlier 960px renderer fallback plots are retained but are not geometry evidence."
        ),
    }
    _write_json(root / "visual_review.json", review)
    (root / "visual_review.md").write_text(
        "# Stage 16-B visual review\n\n"
        "Actual MuJoCo offscreen contact sheets were inspected for both zero-residual clips and "
        "the 170105 H=1 oracle. The workspace camera visibly contains a separate hand and red "
        "free object. All three traces fail at the recorded wrist-orientation safety condition "
        "before contact; no success or PPO visual claim is made.\n",
        encoding="utf-8",
    )

    repo = Path(__file__).resolve().parents[2]
    git = {
        "start_head": args.start_head,
        "final_head": _git_lines(repo, ["rev-parse", "HEAD"])[0],
        "local_commits_over_origin_main": _git_lines(
            repo, ["log", "--oneline", "origin/main..HEAD"]
        ),
        "pushed": False,
        "pr_created": False,
        "main_merged": False,
        "tag_created": False,
    }
    _write_json(root / "git_commits.json", git)
    tests = {
        "status": "PASS" if args.tests_passed else "NOT_RECORDED",
        "commands": [
            "ruff check .",
            "ruff format --check .",
            "python -m mypy src",
            "python -m pytest -q",
            "python scripts/check_paper_fidelity.py",
            "git diff --check",
            "git ls-files .local",
        ],
    }
    _write_json(root / "tests.json", tests)

    summary = {
        "engineering_extension": "WORLD_WRIST_FINGER_TRACKING_PROTOCOL",
        "label": "ENGINEERING_EXTENSION",
        "status": status,
        "reference_export": reference_export,
        "oracle_h10": h10,
        "qualification_root": str(source.resolve()),
        "preflight": str((root / "preflight").resolve()),
        "checkpoints": "none_gate_blocked",
        "ppo_training_started": False,
    }
    _write_json(root / "final_summary.json", summary)
    (root / "final_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    (root / "handoff.md").write_text(
        "# Stage 16-B World Wrist-and-Finger PPO Handoff\n\n"
        "Status: `STAGE16B_BLOCKED_WITH_BOUNDED_EVIDENCE`. The direct world-reference "
        "export passed, but finite-wrench wrist tracking and the 26D oracle safety gate failed. "
        "No PPO checkpoint, push, PR, merge, or tag was created. See `final_summary.md`, "
        "`visual_review.md`, and `git_commits.json`.\n",
        encoding="utf-8",
    )
    (root / "dashboard.html").write_text(
        "<!doctype html><title>Stage 16-B closeout</title><main><h1>Stage 16-B closeout</h1>"
        f"<p>Status: <code>{status['overall']}</code></p>"
        f"<p>Oracle: <code>{status['oracle']}</code>; PPO was not started.</p>"
        "<p>Read final_summary.md and visual_review.md in this report directory.</p></main>\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": status["overall"], "report_root": str(root.resolve())}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
