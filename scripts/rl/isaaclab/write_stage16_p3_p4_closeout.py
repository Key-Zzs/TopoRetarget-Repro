#!/usr/bin/env python3
"""Write an auditable fail-closed Stage16 P3/P4 closeout from existing receipts.

Missing C3/C4/P4 evidence is deliberately recorded as ``NOT_RUN``.  This
script must not replace it with a zero-gravity result or a rejected C2 policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity"
CLIPS = ("hocap_170105", "hocap_170650")
MODES = (("v3", "aggregate_v3"), ("v4", "strict_per_finger_v4"))


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return document


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headings = list(rows[0])
    header = "| " + " | ".join(headings) + " |"
    separator = "| " + " | ".join("---" for _ in headings) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in headings) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def git_commits(start_head: str) -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%H%x1f%s", f"{start_head}..HEAD"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [
        {"commit": line.split("\x1f", 1)[0], "subject": line.split("\x1f", 1)[1]}
        for line in output.splitlines()
        if line
    ]


def baseline_rows() -> list[dict[str, object]]:
    specs = (
        (
            "V3 zero-g",
            REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock",
            "v3_formal_selected_2129920_qualification.json",
        ),
        (
            "V4 zero-g",
            REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4",
            "v4_formal_selected_1064960_qualification.json",
        ),
    )
    rows: list[dict[str, object]] = []
    for protocol, base, filename in specs:
        for clip in CLIPS:
            qualification = read_json(base / clip / "formal" / filename)
            rows.append(
                {
                    "clip": clip,
                    "protocol": protocol,
                    "gravity": "zero-g",
                    "status": qualification["status"],
                    "geometry_absolute_pass": qualification["geometry_absolute_pass"],
                    "SRtask": qualification["ppo_task_success_rate"],
                    "SRphysics": qualification["physics_qualified"],
                    "full_gravity_claim": "NO",
                }
            )
    for clip in CLIPS:
        rows.append(
            {
                "clip": clip,
                "protocol": "P4 full-gravity causal",
                "gravity": "1g",
                "status": "NOT_RUN_G3_PROMOTION_BLOCKED",
                "geometry_absolute_pass": "N/A",
                "SRtask": "N/A",
                "SRphysics": "N/A",
                "full_gravity_claim": "NO_EVIDENCE",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-head", default="ae131ad")
    parser.add_argument("--test-result", action="append", default=[], metavar="NAME=STATUS")
    args = parser.parse_args()
    root = args.output_root.resolve()
    selection = read_json(root / "global_physical_contact_mode_selection.json")
    g3 = read_json(root / "g3/qualification.json")
    if selection["status"] != "GLOBAL_PHYSICAL_CONTACT_MODE_SELECTION_BLOCKED":
        raise ValueError("CLOSEOUT_REQUIRES_BLOCKED_GLOBAL_C2_SELECTION")
    if g3["status"] != "G3_PROMOTION_BLOCKED":
        raise ValueError("CLOSEOUT_REQUIRES_BLOCKED_G3_RECEIPT")

    c2_rows: list[dict[str, object]] = []
    for alias, mode in MODES:
        candidate = selection["candidates"][mode]
        for clip in CLIPS:
            clip_result = candidate["per_clip_metrics"][clip]
            metrics = clip_result["metrics"]
            c2_rows.append(
                {
                    "mode": mode,
                    "mode_alias": alias,
                    "clip": clip,
                    "c2_gravity": 0.5,
                    "c2_friction": 1.5,
                    "safety_pass": clip_result["safety_pass"],
                    "safety_reason": ";".join(clip_result["safety_reasons"]),
                    "SRqualified": metrics["SRqualified"],
                    "SRphysics": metrics["SRphysics"],
                    "p95_penetration_m": metrics["hand_object_p95_penetration_m"],
                }
            )
    write_csv(root / "tables/physical_pilot_v3_v4_macro.csv", c2_rows)
    (root / "tables/physical_pilot_v3_v4_macro.md").write_text(
        "# C2 physical-pilot global-mode selection\n\n"
        "All rows are development-only. A mode must pass both clips; neither did.\n\n"
        + markdown_table(c2_rows)
        + "\n",
        encoding="utf-8",
    )
    comparison_rows = baseline_rows()
    write_csv(root / "tables/hocap_170105_v3_v4_fullgravity.csv", comparison_rows)
    (root / "tables/hocap_170105_v3_v4_fullgravity.md").write_text(
        "# Zero-g baselines and full-gravity causal status\n\n"
        "The zero-g rows are historical baselines only, not full-gravity evidence.\n\n"
        + markdown_table(comparison_rows)
        + "\n",
        encoding="utf-8",
    )

    test_rows: list[dict[str, str]] = []
    for value in args.test_result:
        if "=" not in value:
            raise ValueError(f"TEST_RESULT_EXPECTS_NAME_EQUALS_STATUS:{value}")
        name, status = value.split("=", 1)
        test_rows.append({"name": name, "status": status})
    tests = {"status": "RECORDED" if test_rows else "NOT_RUN", "checks": test_rows}
    write_json(root / "tests.json", tests)

    transitions = (
        {
            "stage": "C2_GLOBAL_MODE_SELECTION",
            "status": selection["status"],
            "reason": selection["selection_reason"],
            "next_stage": "G3",
        },
        {
            "stage": "G3_FULL_GRAVITY_PROMOTION",
            "status": g3["status"],
            "reason": g3["promotion_reason"],
            "next_stage": "C3_C4_P4",
        },
    )
    (root / "failure_transitions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in transitions), encoding="utf-8"
    )
    write_json(
        root / "git_commits.json",
        {"start_head": args.start_head, "commits": git_commits(args.start_head)},
    )

    summary: dict[str, object] = {
        "schema_version": "Stage16P3P4FullGravityCausalCloseoutV1",
        "status": "P3_PHYSICS_CURRICULUM_BLOCKED_AT_C2_SELECTION",
        "p3": {
            "entry": "P3_READY_WITH_CONSTRAINTS",
            "c0_c1_c2": "COMPLETE_DEVELOPMENT_PILOTS",
            "global_c2_selection": selection["status"],
            "g3": g3["status"],
            "c3": "NOT_RUN_UPSTREAM_G3_BLOCKED",
            "c4": "NOT_RUN_UPSTREAM_G3_BLOCKED",
        },
        "p4": {
            "status": "P4_FULL_GRAVITY_CAUSAL_INSUFFICIENT",
            "formal_run": False,
            "replay_export": False,
            "reason": "No global C2 mode passed absolute geometry on both clips.",
        },
        "full_gravity_claim": "NOT_SUPPORTED",
        "selection": {
            "selected_mode": selection["selected_mode"],
            "rejected_modes": selection["rejected_modes"],
            "reason": selection["selection_reason"],
        },
        "g3_contract": g3["physics_contract"],
        "artifacts": {
            "c2_selection": str(root / "global_physical_contact_mode_selection.json"),
            "g3_receipt": str(root / "g3/qualification.json"),
            "c2_table": str(root / "tables/physical_pilot_v3_v4_macro.md"),
            "comparison_table": str(root / "tables/hocap_170105_v3_v4_fullgravity.md"),
        },
        "tests": tests,
    }
    write_json(root / "final_summary.json", summary)
    handoff = "\n".join(
        (
            "# Stage16 P3/P4 full-gravity causal closeout",
            "",
            "## Verdict",
            "",
            "**P3 BLOCKED; P4 INSUFFICIENT.** C0--C2 development pilots completed, "
            "but no V3/V4 C2 mode passed absolute geometry on both clips. G3 was "
            "not executed, and C3, C4, P4, replay export, and any full-gravity "
            "claim remain unavailable.",
            "",
            "## Safe stop boundary",
            "",
            "The frozen G3 C4 contract is 1g, nominal friction, 20 control steps, "
            "and four replicas per retained safe state. It retains the 8 + 25 state "
            "roster as an unexecuted receipt. A rejected or clip-specific C2 policy "
            "must not be substituted.",
            "",
            "## Required next decision",
            "",
            "Repair the C2 absolute-geometry failure under the existing causal "
            "contract, then rerun C2 global selection. Only a mode passing both "
            "clips may restart G3 and unlock C3/C4/P4.",
            "",
            "## Evidence",
            "",
            "- `global_physical_contact_mode_selection.json`: both modes rejected "
            "by the absolute-geometry gate.",
            "- `g3/qualification.json`: upstream promotion block and retained safe-state roster.",
            "- `tables/physical_pilot_v3_v4_macro.md`: C2 per-clip selection inputs.",
            "- `tables/hocap_170105_v3_v4_fullgravity.md`: zero-g baselines "
            "explicitly separated from unavailable 1g P4 evidence.",
            "",
        )
    )
    (root / "final_summary.md").write_text(handoff, encoding="utf-8")
    (root / "handoff.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_root": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
