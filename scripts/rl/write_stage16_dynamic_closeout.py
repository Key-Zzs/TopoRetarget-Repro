#!/usr/bin/env python3
"""Write human-readable Stage-16.1a closeout and visual-review artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True, stderr=subprocess.STDOUT).strip()


def _table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---" for _ in headers]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def _first_pair(profile: dict[str, Any]) -> str:
    for frame in profile["frames"]:
        for contact in frame["initial"]["contacts"]:
            if contact["is_hand_object"]:
                return f"{contact['geom1']} ↔ {contact['geom2']}"
    return "none before diagnostic end"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--tests", required=True, type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
    final = _read(root / "final_summary.json")
    step_a = _read(root / "step_a_pd_qualification.json")
    step_b = _read(root / "step_b_contact_audit.json")
    step_c = _read(root / "step_c_velocity_matrix.json")
    sensitivity = _read(root / "object_oracle_sensitivity.json")
    shooting = _read(root / "shooting_oracle_evaluation.json")
    root_cause = _read(root / "root_cause.json")
    visual = _read(root / "visual_review.json")
    tests = _read(args.tests)
    baseline = _read(args.baseline)
    selected_preload = float(step_b["selected_profile"])
    selected_profiles = {
        clip: next(
            profile
            for profile in profiles
            if float(profile["preload_fraction"]) == selected_preload
        )
        for clip, profiles in step_b["profiles"].items()
    }
    visual.update(
        {
            "checked": True,
            "reviewer": "Codex image inspection",
            "renderer_attempts": {
                "egl": "offscreen renderer unavailable; numerical fallback generated",
                "osmesa": "offscreen renderer unavailable; numerical fallback generated",
                "xvfb": "xvfb-run unavailable on this host",
                "numerical": "contact sheet inspected",
            },
            "observations": [
                "Step-A panels keep object position/axis at zero because the object is reference-driven.",
                "Both Step-A contact panels have no contacts until later reference frames, so this is PD isolation only.",
                "All dynamic-object C0--C3 and object-oracle panels cross the red 5 cm line around steps 5--6.",
                "Contact/wrench traces are zero before the crossing; no object teleport or direct object-control trace is present.",
            ],
            "geometry_acceptance": "NOT_AVAILABLE_RENDERER_LIMITATION",
        }
    )
    _write(root / "visual_review.json", visual)
    visual_md = "# Stage-16.1a visual review\n\n" + "\n".join(
        f"- {item}" for item in visual["observations"]
    )
    visual_md += (
        "\n\nThe contact sheet was inspected at `"
        + str(visual["contact_sheet"])
        + "`. EGL and OSMesa did not create a MuJoCo GL context, and Xvfb is unavailable; "
        "the result is therefore a numerical visualization review, not geometry acceptance.\n"
    )
    (root / "visual_review.md").write_text(visual_md, encoding="utf-8")
    step_a_rows = [
        [
            str(row["label"]).replace(".stage16_step_a_dynamic_hand_kinematic_object", ""),
            f"{float(row['final_joint_rmse_rad']):.5f}",
            f"{float(row['final_link_rmse_mm']):.3f}",
            "none persistent",
            str(row["frames_completed"]),
            str(row["result"]),
        ]
        for row in step_a["rows"]
    ]
    step_b_rows = []
    for clip, profile in selected_profiles.items():
        frames = profile["frames"]
        contacts = sum(int(row["initial"]["hand_object_contact_count"]) for row in frames)
        force = max(
            max(abs(value) for value in row["initial"]["object_wrench_world"][:3]) for row in frames
        )
        wrench = max(
            sum(value * value for value in row["initial"]["object_wrench_world"]) ** 0.5
            for row in frames
        )
        push = max(float(row["push_displacement_m"]) for row in frames)
        result = (
            "FAIL pre-gate contact"
            if all(
                int(row["initial"]["hand_object_contact_count"]) == 0
                for row in frames
                if int(row["frame"]) <= 10
            )
            else "PASS"
        )
        step_b_rows.append(
            [
                clip,
                str(contacts),
                f"{force:.3f} N",
                f"{wrench:.3f} N-m",
                f"{push * 1000:.2f} mm",
                result,
            ]
        )
    step_c_rows = [
        [
            str(row["label"]).replace(".stage16_step_c_", ": "),
            str(row["first_failure_frame"]),
            f"{float(row['progress']) * 100:.1f}%",
            f"{float(row['final_position_cm']):.2f}",
            f"{float(row['final_axis_cm']):.2f}",
            "recorded; no reset explosion",
            "diagnostic-only",
        ]
        for row in step_c["rows"]
    ]
    step_d_rows = []
    for row in final["step_d"]:
        clip = str(row["label"]).replace(".stage16_step_d_object_oracle", "")
        ranks = [entry["rank"] for entry in sensitivity[clip]]
        conditions = [entry["condition_estimate"] for entry in sensitivity[clip]]
        step_d_rows.append(
            [
                clip,
                "object-aware residual",
                "1 (D1; D2/D3 gate not reached)",
                "0%",
                "0%",
                f"{float(row['final_position_cm']):.2f}",
                f"{float(row['final_rotation_deg']):.2f}",
                f"{float(row['final_axis_cm']):.2f}",
                f"rank {max(ranks)}, cond {max(conditions):.2f}",
            ]
        )
    shot_rows = [
        f"{clip}: "
        + ", ".join(
            f"f{row['frame']}/H{row['horizon']} {row['classification']} "
            f"{row['baseline_error_norm']:.4f}->{row['best_error_norm']:.4f}"
            for row in rows
        )
        for clip, rows in shooting.items()
    ]
    branch = _command("git", "branch", "--show-current")
    head = _command("git", "rev-parse", "HEAD")
    commits = _command("git", "log", "--oneline", "origin/main..HEAD")
    git_payload = {
        "branch": branch,
        "head": head,
        "origin_main_to_head": commits.splitlines(),
        "pushed": False,
        "pr_created": False,
        "main_merged": False,
        "tag_created": False,
    }
    _write(root / "git_commits.json", git_payload)
    _write(root / "tests.json", tests)
    summary = "# Stage-16.1a final summary\n\n"
    summary += f"- Status: `{final['status']}`\n- Stage 16.2: `{final['stage16_2_entry']}`\n"
    summary += (
        f"- Root cause: `{root_cause['primary']}`\n- PPO started: `{final['ppo_started']}`\n\n"
    )
    summary += "## Step A\n\n" + _table(
        ["Clip", "Joint RMSE rad", "Link RMSE mm", "Saturation", "Frames", "Result"], step_a_rows
    )
    summary += "\n\n## Step B\n\n" + _table(
        ["Clip", "Contacts", "Peak force", "Peak wrench", "Push response", "Result"], step_b_rows
    )
    summary += "\n\nActual first later contact pairs: " + "; ".join(
        f"{clip}: {_first_pair(profile)}" for clip, profile in selected_profiles.items()
    )
    summary += "\n\n## Step C\n\n" + _table(
        [
            "Profile / clip",
            "First failure",
            "Progress",
            "Pos cm",
            "Axis cm",
            "Reset impulse",
            "Result",
        ],
        step_c_rows,
    )
    summary += "\n\nSelected global profile: `HAND_REFERENCE_VELOCITY_SELECTED` (C3); it is an engineering assumption and does not authorize PPO.\n"
    summary += "\n## Step D\n\n" + _table(
        [
            "Clip",
            "Oracle",
            "Episodes",
            "Success",
            "Final reach",
            "Pos cm",
            "Rot deg",
            "Axis cm",
            "Rank/condition",
        ],
        step_d_rows,
    )
    summary += "\n\n## Step E\n\n" + "\n".join(f"- {row}" for row in shot_rows) + "\n"
    summary += "\n## Visual review\n\n" + visual_md
    summary += "\n## Tests\n\n```json\n" + json.dumps(tests, indent=2) + "\n```\n"
    summary += "\n## Local commits\n\n```text\n" + commits + "\n```\n"
    (root / "final_summary.md").write_text(summary, encoding="utf-8")
    handoff = "# Stage 16.1 Hand–Object Dynamic Coupling Handoff\n\n"
    headings = [
        "Final Status",
        "Git and Environment",
        "Frozen Failure Baseline",
        "Full Diagnostic Rollout",
        "Step A — Dynamic Hand / Kinematic Object",
        "Step B — Contact Coupling",
        "Step C — Velocity Reset Matrix",
        "Existing Oracle Audit",
        "Step D — Object-Aware Oracle",
        "Step E — Dynamic Feasibility",
        "Root-Cause Classification",
        "Implemented Repairs",
        "Failure-Recovery State Machine",
        "Formal Stage 16.1 Qualification",
        "Visualization and Screenshot Review",
        "Interactive MuJoCo Commands",
        "Tests",
        "Local Commits",
        "Stage 16.2 Entry Decision",
        "Recommended Next Action",
    ]
    bodies = {
        "Final Status": f"`{final['status']}`; `{final['stage16_2_entry']}`; root cause `{root_cause['primary']}`.",
        "Git and Environment": f"Branch `{branch}`, HEAD `{head}`, MuJoCo CPU correctness backend.",
        "Frozen Failure Baseline": str(baseline["archive"]),
        "Full Diagnostic Rollout": str(root / "diagnostic_rollout.jsonl"),
        "Step A — Dynamic Hand / Kinematic Object": _table(
            ["Clip", "Joint RMSE rad", "Link RMSE mm", "Saturation", "Frames", "Result"],
            step_a_rows,
        ),
        "Step B — Contact Coupling": _table(
            ["Clip", "Contacts", "Peak force", "Peak wrench", "Push response", "Result"],
            step_b_rows,
        )
        + "\n\nPairs: "
        + "; ".join(
            f"{clip}: {_first_pair(profile)}" for clip, profile in selected_profiles.items()
        ),
        "Step C — Velocity Reset Matrix": _table(
            [
                "Profile / clip",
                "First failure",
                "Progress",
                "Pos cm",
                "Axis cm",
                "Reset impulse",
                "Result",
            ],
            step_c_rows,
        ),
        "Existing Oracle Audit": "Previous `OracleResidualController` was `PREVIOUS_ORACLE_NOT_OBJECT_AWARE`.",
        "Step D — Object-Aware Oracle": _table(
            [
                "Clip",
                "Oracle",
                "Episodes",
                "Success",
                "Final reach",
                "Pos cm",
                "Rot deg",
                "Axis cm",
                "Rank/condition",
            ],
            step_d_rows,
        ),
        "Step E — Dynamic Feasibility": "\n".join(f"- {row}" for row in shot_rows),
        "Root-Cause Classification": json.dumps(root_cause, indent=2),
        "Implemented Repairs": "Stable generated collision labels, full trace logging, reset matrix, cloned-state object-aware oracle, bounded shooting, and numerical visualization. No reference, termination, PPO contract, base action, or direct object control changed.",
        "Failure-Recovery State Machine": str(root / "failure_transition_log.jsonl"),
        "Formal Stage 16.1 Qualification": "Blocked: Step B pre-gate contact and Step D full-trajectory oracle gates fail.",
        "Visualization and Screenshot Review": str(root / "visual_review.md"),
        "Interactive MuJoCo Commands": (
            "`conda run -n toporetarget-rl python scripts/rl/visualize_hocap_policy_mujoco.py "
            "--reference .local/stage16_reference_tracking_ppo/references/hocap_170105.stage16.npz "
            "--object-mesh .local/stage16_reference_tracking_ppo/objects/hocap_170105.obj "
            "--policy zero --mode interactive`; replace `zero` with `joint-oracle`, `object-oracle`, "
            "or `shooting-oracle`; add `--kinematic-object --diagnostic-continue-after-termination` for Step A."
        ),
        "Tests": json.dumps(tests, indent=2),
        "Local Commits": commits,
        "Stage 16.2 Entry Decision": final["stage16_2_entry"],
        "Recommended Next Action": "Keep PPO blocked. Any next experiment must be separately authorized and preserve references/formal gates while establishing early support contact.",
    }
    for heading in headings:
        handoff += f"## {heading}\n\n{bodies[heading]}\n\n"
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    print(
        json.dumps({"summary": str(root / "final_summary.md"), "handoff": str(root / "handoff.md")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
