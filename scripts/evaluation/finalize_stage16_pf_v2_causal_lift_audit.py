#!/usr/bin/env python3
"""Write the fail-closed Stage16 PF V2 audit handoff from frozen receipts."""

# ruff: noqa: E501, I001

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    summary_path = REPORT_ROOT / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["classification"] != "PF_V2_SEMANTICS_INVALID":
        raise RuntimeError("PF_V2_FINALIZER_REQUIRES_FAIL_CLOSED_AUDIT")
    current_branch = _git("branch", "--show-current")
    if current_branch != "feature/dexplore-reward-rse":
        raise RuntimeError("PF_V2_FINALIZER_WRONG_BRANCH")
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--short", "--untracked-files=all")
    _write_json(
        REPORT_ROOT / "training/170105/not_run.json",
        {
            "status": "NOT_RUN_STOP_PF_V2_SEMANTICS_INVALID",
            "requested_updates": "U11-U20 (max 10)",
            "actual_updates": 0,
            "actual_samples": 0,
            "reason": "accepted 170650 PF V2 positive-control regression",
        },
    )
    _write_json(
        REPORT_ROOT / "training/170650/not_run.json",
        {
            "status": "NOT_RUN_STOP_PF_V2_SEMANTICS_INVALID",
            "requested_updates": "U01-U10 (max 10)",
            "actual_updates": 0,
            "actual_samples": 0,
            "historical_actor_overwritten": False,
            "reason": "accepted 170650 PF V2 positive-control regression",
        },
    )
    _write_json(
        REPORT_ROOT / "u10_eval20/not_run.json",
        {
            "status": "NOT_RUN_STOP_PF_V2_SEMANTICS_INVALID",
            "optimizer_steps": 0,
            "reason": "task stop condition occurred in frozen existing-trace re-evaluation",
        },
    )
    _write_json(
        REPORT_ROOT / "final_eval20/not_run.json",
        {
            "status": "NOT_RUN_STOP_PF_V2_SEMANTICS_INVALID",
            "reason": "no new PPO checkpoint is authorized",
        },
    )
    _write_csv(
        REPORT_ROOT / "comparison/symmetric_training.csv",
        [
            {
                "clip": "hocap_170105",
                "requested_lineage": "U10_to_U11_U20",
                "actual_updates": 0,
                "actual_samples": 0,
                "status": "NOT_RUN_STOP_PF_V2_SEMANTICS_INVALID",
            },
            {
                "clip": "hocap_170650",
                "requested_lineage": "historical_to_U01_U10",
                "actual_updates": 0,
                "actual_samples": 0,
                "status": "NOT_RUN_STOP_PF_V2_SEMANTICS_INVALID",
            },
        ],
    )
    _write_csv(
        REPORT_ROOT / "comparison/historical_vs_new.csv",
        [
            {
                "metric": "PF_V2_audit",
                "170105_historical": "0/10",
                "170105_new_best": "NOT_RUN",
                "170650_historical": "0/20: support transfer not observable",
                "170650_new_best": "NOT_RUN",
            }
        ],
    )
    _write_text(
        REPORT_ROOT / "replay/visualization_commands.md",
        """# PF V2 audit replay commands

These commands replay recorded PhysX hand/object/table plus raw HOCap MANO and
object with the retarget-reference ghost hidden. `M` toggles raw MOCAP and `R`
toggles the retarget reference; hidden layers do not receive per-frame writes.

## 170105 U10 audit representative (not accepted)

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/reports/stage16_dexplore_reward_rse/training/U10/eval10/traces/episode_00.npz --object hocap_170105 --no-reference-ghost
```

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/reports/stage16_dexplore_reward_rse/training/U10/eval10/traces/episode_00.npz --object hocap_170105 --no-reference-ghost --start-frame 176 --end-frame 260
```

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/reports/stage16_dexplore_reward_rse/training/U10/eval10/traces/episode_00.npz --object hocap_170105 --no-reference-ghost --mocap-object-low-poly --start-frame 176 --end-frame 260
```

Markers: reference LIFT 184; first hand contact 184; persistent multi-contact
188; support-release proxy 192; actual 5 cm lift onset 241.

## Historical accepted 170650 audit representative

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost
```

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost --start-frame 140 --end-frame 240
```

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz --object hocap_170650 --no-reference-ghost --mocap-object-low-poly --start-frame 140 --end-frame 240
```

Markers: first hand contact 152; persistent multi-contact 162; reference LIFT
184; actual 5 cm lift onset 222. The table-support signal is already absent at
the first valid recorded frame, so a support-release event is not observable.

## Historical 170105 failure

```bash
OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop --trace .local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4/episode_00.npz --object hocap_170105 --no-reference-ghost
```
""",
    )
    _write_text(
        REPORT_ROOT / "replay/manual_acceptance.md",
        """# Manual review boundary

PF V2 is not an acceptance authority. Inspect U10 only as a causal-lift audit:
whether hand contact persists through object lift, whether the object separates
or flies ballistically, and whether its support release precedes actual lift.
For 170650, inspect the stable hand-object coupling while noting that the
recorded trace does not include an observed table-supported pre-release state.
Do not manually override `PF_V2_SEMANTICS_INVALID` or authorize PPO from a
visual impression.
""",
    )
    _write_json(
        REPORT_ROOT / "tests.json",
        {
            "targeted_pf_v2_and_profile": "14 passed",
            "ruff_task_files": "PASS",
            "format_task_files": "PASS",
            "mypy_src": "PASS: 388 source files",
            "pytest_full": "PASS: 822 passed, 27 skipped, 1 warning",
            "paper_fidelity": "PASS",
            "PF_V1_source_sha256": "b334a9ff452a94801bbdf653e69d977a71a97c280e5300a5056e9f30b0b77e5b",
            "PF_V1_CHANGED": "NO",
        },
    )
    _write_json(
        REPORT_ROOT / "resource_usage.json",
        {
            "frozen_trace_episodes": 50,
            "new_isaac_evaluations": 0,
            "new_ppo_updates": 0,
            "new_ppo_samples": 0,
            "reason": "PF_V2_SEMANTICS_INVALID stop before PPO",
        },
    )
    _write_text(
        REPORT_ROOT / "technical_failures.jsonl",
        json.dumps(
            {
                "kind": "SEMANTIC_STOP_CONDITION",
                "classification": "PF_V2_SEMANTICS_INVALID",
                "detail": "170650 accepted traces lack observed support before release under SupportTransferProxyV1",
            },
            sort_keys=True,
        ),
    )
    _write_json(
        REPORT_ROOT / "git_commits.json",
        {
            "branch": current_branch,
            "final_head": head,
            "commits": _git("log", "--oneline", "--max-count=8").splitlines(),
            "worktree_status_at_handoff": status.splitlines(),
            "pushed": False,
            "pr_created": False,
            "local_reports_tracked": False,
        },
    )
    _write_text(
        REPORT_ROOT / "final_summary.md",
        f"""# Stage16 PF V2 Causal Lift + Symmetric PPO Handoff

## Decision

`PF_V2_AUDIT=PF_V2_SEMANTICS_INVALID` and `CONFIDENCE=HIGH`.

PF V2 correctly keeps reference-LIFT timing out of its hard gates, but its
required support-transfer proxy is not observable for the accepted 170650
trace set. It therefore scores that positive control 0/20 and cannot replace
PF V1. This is a stop condition.

| Trace | PF V1 | Physical lift | Causal lift | Support transfer proxy | PF V2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical 170105 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| U9 170105 | 0/10 | 9/10 | 9/10 | 9/10 | 9/10 |
| U10 170105 | 0/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Historical 170650 | 20/20 | 20/20 | 20/20 | 0/20 | 0/20 |

`U10` does not receive Eval20. 170105 U11--U20 and the new 170650 U01--U10
experimental lineage are `NOT_RUN`: 0 PPO updates and 0 samples. The accepted
170650 actor and receipt remain untouched.

## Git

`branch={current_branch}`

`FINAL_HEAD={head}`

The handoff git receipt records the final worktree status and local commits.
No push, PR, branch, or worktree was created.

## Recommended next step

`NEXT_REVISE_CAUSAL_LIFT_AUTHORITY`: prospectively capture a support-transfer
authority that can observe the table-supported state and its release before
using causal lift as a physical-functionality gate. Do not revise this PF V2
contract against the frozen U9/U10 outcomes.
""",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
