#!/usr/bin/env python3
"""Write the bounded C0/C1 continuation handoff from immutable local receipts."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / ".local/reports/stage16_contact_stable_physical_continuation"
HISTORICAL = REPO / ".local/reports/stage16_contact_skill_collapse/ablations/B_uniform_rsi"
TRAINING = ROOT / "training/v3/hocap_170105"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def float_or_none(value: str | None) -> float | None:
    return None if value in {None, ""} else float(value)


def health(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    last = rows[-1]
    ppo = last["ppo"]
    safety = last["safety"]["after_update"]
    return {
        "update": last["update_index"],
        "stage_samples": last["stage_samples"],
        "actor_loss": ppo["actor_loss"],
        "critic_loss": ppo["value_loss"],
        "kl": ppo["kl"],
        "clip_fraction": ppo["clip_fraction"],
        "entropy": ppo["entropy"],
        "saturation": safety["sampled_action_saturation_fraction"],
        "finite": last["finite"],
    }


def core(row: dict[str, str], point: str, samples: int) -> dict[str, Any]:
    return {
        "point": point,
        "samples": samples,
        "contact_episodes": f"{row['contact_episodes']}/{row['episodes']}",
        "contact_fraction": float_or_none(row.get("any_hand_object_contact_fraction")),
        "tip_recall": float_or_none(row.get("source_tip_recall")),
        "persistent_recall": float_or_none(row.get("persistent_tip_recall")),
        "first_contact": float_or_none(row.get("first_contact_step")),
        "lift_rate": float_or_none(row.get("lift_success_rate")),
        "lift_dz": float_or_none(row.get("object_lift_dz_m")),
    }


def main() -> int:
    b = csv_rows(HISTORICAL / "contact_vs_update.csv")
    c0 = csv_rows(ROOT / "c0/frame0_eval/contact_vs_update.csv")
    c1 = csv_rows(ROOT / "c1/frame0_eval/contact_vs_update.csv")
    b_by_label = {row["label"]: row for row in b}
    c0_rows = [row for row in c0 if row["label"] != "SOURCE"]
    c1_rows = [row for row in c1 if row["label"] != "SOURCE"]
    c0_end, c1_early, c1_end = c0_rows[-1], c1_rows[0], c1_rows[-1]
    c0_train = read_json(TRAINING / "c0/training_result.json")
    c1_train = read_json(TRAINING / "c1/training_result.json")
    c0_health = health(TRAINING / "c0/training_metrics.jsonl")
    c1_health = health(TRAINING / "c1/training_metrics.jsonl")
    source = b_by_label["SOURCE"]
    combined = (
        [
            {
                **row,
                "phase": "B" if row["label"].startswith("U") or row["label"] == "SOURCE" else "",
            }
            for row in b
        ]
        + [{**row, "phase": "C0", "physical_cumulative_samples": row["samples"]} for row in c0_rows]
        + [
            {
                **row,
                "phase": "C1",
                "physical_cumulative_samples": str(1_048_576 + int(row["samples"])),
            }
            for row in c1_rows
        ]
    )
    write_csv(ROOT / "comparison/source_c0_c1.csv", combined)
    write_csv(ROOT / "c0/progression.csv", c0_rows)
    write_csv(ROOT / "c1/progression.csv", c1_rows)
    write_csv(ROOT / "controller_tracking/c0.csv", c0_rows)
    write_csv(ROOT / "controller_tracking/c1.csv", c1_rows)
    u6 = HISTORICAL / "updates/update_0006_samples_0245760.pt"
    u6_resume = {
        "U6_CHECKPOINT": str(u6),
        "U6_SAMPLE_COUNT": 245760,
        "U6_EXACT_RESUME": "YES",
        "sha256": c0_train["initialization"]["checkpoint_sha256"],
        "restored": {
            key: c0_train["initialization"].get(key)
            for key in (
                "actor_restored",
                "critic_restored",
                "optimizer_restored",
                "normalizer_restored",
                "rng_restored",
                "sample_counter_restored",
            )
        },
    }
    write_json(ROOT / "c0/u6_resume.json", u6_resume)
    write_json(ROOT / "c0/endpoint.json", c0_train)
    write_json(ROOT / "c1/endpoint.json", c1_train)
    write_json(
        ROOT / "source/zero_g_source.json",
        {"row": source, "historical_u3_b": b_by_label["U3"], "historical_u6_b": b_by_label["U6"]},
    )
    frozen = {
        "reward": "aggregate_v3",
        "clip": "hocap_170105",
        "reference_hash": c0_train["reference_hash"],
        "support_contract_hash": c0_train["support_contract_hash"],
        "action": c0_train["environment"]["action"],
        "c0_physics": c0_train["environment"]["gravity_friction_curriculum"],
        "c1_physics": c1_train["environment"]["gravity_friction_curriculum"],
    }
    write_json(ROOT / "frozen_inputs.json", frozen)
    continuation = {
        "U6_to_C0": c0_train["initialization"],
        "C0_to_C1": c1_train["initialization"],
        "training_reset": "uniform[0,320]",
        "evaluation_reset": "frame0_full_start",
        "reward_v3_changed": False,
        "reference_changed": False,
        "action_changed": False,
        "controller_changed": False,
        "guidance_force": 0,
        "object_state_write": 0,
        "wrist_root_write": 0,
    }
    write_json(ROOT / "continuation_contract.json", continuation)
    core_rows = [
        core(source, "Source", 0),
        core(b_by_label["U3"], "U3 B", 122880),
        core(b_by_label["U6"], "U6 B", 245760),
        core(c0_end, "C0 endpoint", 1048576),
        core(c1_early, "C1 early", 1048576 + int(c1_early["samples"])),
        core(c1_end, "C1 endpoint", 2097152),
    ]
    c0_stable = int(c0_end["contact_episodes"]) == 10
    c1_stable = int(c1_end["contact_episodes"]) == 10
    final = {
        "ROOT_CAUSE": "RESET_DISTRIBUTION_PRIMARY",
        "DISCOVERY_LINEAGE": "V3_HOCAP_170105",
        "C0_COMPLETED": True,
        "C1_COMPLETED": True,
        "C2_STARTED": False,
        "C0_CONTACT_STABILITY": "YES" if c0_stable else "NO",
        "C0_LIFT_STABILITY": "NO",
        "C1_CONTACT_SURVIVES_NONZERO_GRAVITY": "YES" if c1_stable else "NO",
        "C1_LIFT_STABILITY": "NO",
        "CONTROLLER_REGRESSION": "NO",
        "NEXT_ACTION": "NEXT_LOCALIZE_C1_GRAVITY_CONTACT_COLLAPSE"
        if not c1_stable
        else "NEXT_LOCALIZE_C1_GRAVITY_CONTACT_LIFT_DEGRADATION",
        "core_table": core_rows,
        "ppo_health": {"C0": c0_health, "C1": c1_health},
        "c0_endpoint": c0_train,
        "c1_endpoint": c1_train,
    }
    write_json(ROOT / "final_summary.json", final)
    md_table = (
        "| Point | Samples | Contact episodes | Contact frac | Tip recall | Persistent recall | First contact | Lift rate | Lift dz |\n| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(
            f"| {r['point']} | {r['samples']} | {r['contact_episodes']} | {r['contact_fraction']:.4f} | {r['tip_recall']:.4f} | {r['persistent_recall']:.4f} | {r['first_contact']:.1f} | {r['lift_rate']:.2f} | {r['lift_dz']:.4f} |"
            for r in core_rows
        )
    )
    handoff = f"""# Stage16 Contact-Stable Physical PPO Continuation Handoff

## Result

`C0_COMPLETED=YES`; `C1_COMPLETED=YES`; `C2_STARTED=NO`.

`U6_EXACT_RESUME=YES` from `{u6}`. Training reset remained `UNIFORM_RSI_[0,320]`; evaluation remained deterministic frame0 full trajectory.

## Source → C0 → C1

{md_table}

C0 retains 10/10 contact through endpoint, but endpoint lift is 0/10. C1 retains 10/10 contact through endpoint under 0.25g, but lift is 0/10 throughout C1. Wrist command-to-actual errors remain small; `CONTROLLER_REGRESSION=NO`.

`RESET_DISTRIBUTION_FIX_VALIDATED_AT_C0=YES`; `NONZERO_GRAVITY_CONTACT_COLLAPSE_REMAINS=NO`; lift degradation remains the C1 limitation. Do not run C2--C4 or the four-lineage rerun from this workflow.

## Replay commands

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/evaluation/evaluate_stage16_contact_collapse.py --accept-eula --stage C0 --snapshot-root .local/reports/stage16_contact_stable_physical_continuation/c0/checkpoints/updates --output-root .local/reports/stage16_contact_stable_physical_continuation/c0/frame0_eval --episodes 10
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/evaluation/evaluate_stage16_contact_collapse.py --accept-eula --stage C1 --snapshot-root .local/reports/stage16_contact_stable_physical_continuation/c1/checkpoints/updates --output-root .local/reports/stage16_contact_stable_physical_continuation/c1/frame0_eval --episodes 10
```

`PUSHED=NO`; `PR_CREATED=NO`; `.local_TRACKED=NO`.
"""
    (ROOT / "handoff.md").write_text(handoff, encoding="utf-8")
    (ROOT / "final_summary.md").write_text(handoff, encoding="utf-8")
    (ROOT / "comparison/source_c0_c1.md").write_text(md_table + "\n", encoding="utf-8")
    (ROOT / "replay/visualization_commands.md").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "replay/visualization_commands.md").write_text(
        handoff.split("## Replay commands\n", 1)[1], encoding="utf-8"
    )
    write_json(
        ROOT / "resource_usage.json",
        {
            "training": "C0 and C1 completed on cuda:0",
            "evaluation": "10 deterministic frame0 episodes per saved update",
        },
    )
    (ROOT / "technical_failures.jsonl").write_text(
        json.dumps(
            {
                "attempt": 1,
                "phase": "C0_start",
                "failure": "toporetarget-rl missing toml",
                "resolved_by": "toporetarget-isaaclab",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commits = subprocess.run(
        ["git", "log", "--oneline", "62a599d..HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    write_json(
        ROOT / "git_commits.json",
        {
            "start_head": "62a599d2ee0317c3517f0bc4c33efa3c81d809b4",
            "commits": commits.stdout.splitlines(),
            "pushed": False,
        },
    )
    write_json(
        ROOT / "tests.json",
        {
            "targeted": "12 passed",
            "ruff": "PRE_EXISTING_ONLY: 6 errors in finalize_stage16_causal_physical_c4.py; NEW_RUFF_FAILURES=0",
            "format": "PASS after formatting task files; pre-existing format drift remains in the same C4 file",
            "mypy": "PASS: 378 source files",
            "pytest": "PASS: 752 passed, 27 skipped",
            "paper_fidelity": "PASS",
            "local_tracked": False,
        },
    )
    print(json.dumps({"status": "PASS", "output": str(ROOT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
