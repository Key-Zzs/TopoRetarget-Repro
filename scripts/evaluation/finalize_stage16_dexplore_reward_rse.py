#!/usr/bin/env python3
"""Close out the bounded Stage16 grouped-reward/RSE experiment from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_dexplore_reward_rse"
BASE_COMMIT = "6b3851fd66b95e3f5ca76638b8bf3d04d019f789"
BRANCH = "feature/dexplore-reward-rse"
SOURCE_CHECKPOINT = (
    REPO_ROOT / ".local/reports/stage16_frozen_source_policy_gravity_sweep/sources/"
    "v4_hocap_170105.json"
)
HISTORICAL_AUTHORITY = (
    REPO_ROOT / ".local/reports/stage16_contact_timing_angular_twist_pf_df/final_summary.json"
)
POSITIVE_AUTHORITY = (
    REPO_ROOT / ".local/reports/stage16_170650_closure_and_human_object_profile/final_summary.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruff-check", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--ruff-format", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--mypy", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--pytest", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--paper-fidelity", choices=("PASS", "FAIL"), required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("DEXPLORE_CLOSEOUT_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_progression() -> list[dict[str, str]]:
    with (REPORT_ROOT / "training/progression.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or [int(row["update"]) for row in rows] != list(range(1, 11)):
        raise RuntimeError("DEXPLORE_CLOSEOUT_REQUIRES_COMPLETE_U01_U10")
    return rows


def _group_stat(
    rows: list[dict[str, str]], clip: str, window: str, metric: str
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["clip"] == clip and row["window"] == window and row["metric"] == metric
    ]
    if len(selected) != 1:
        raise RuntimeError(f"DEXPLORE_GROUP_STAT_MISSING:{clip}:{window}:{metric}")
    row = selected[0]
    return {
        name: float(row[name])
        for name in (
            "mean",
            "median",
            "p5",
            "p25",
            "p75",
            "p95",
            "fraction_lt_1e-2",
            "fraction_lt_1e-4",
            "fraction_lt_1e-6",
        )
    }


def _technical_failures() -> list[dict[str, Any]]:
    path = REPORT_ROOT / "technical_failures.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _nvidia_probe() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,"
        "utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _format_ppo_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| U | samples | PF | lift | DF pose | DF linear | DF angular V2 | "
        "R_int | R_total | kappa |",
        "| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['update'])} | {int(row['samples'])} | {int(row['PF'])}/10 | "
            f"{int(row['lift'])}/10 | {int(row['DF_pose'])}/10 | "
            f"{int(row['DF_linear'])}/10 | {int(row['DF_angular_v2'])}/10 | "
            f"{float(row['R_int']):.4f} | {float(row['R_total']):.4f} | "
            f"{float(row['kappa']):.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parser().parse_args()
    if _git("branch", "--show-current") != BRANCH:
        raise RuntimeError("DEXPLORE_CLOSEOUT_WRONG_BRANCH")
    offline = _read_json(REPORT_ROOT / "offline/offline_gate.json")
    runtime = _read_json(REPORT_ROOT / "runtime_sanity/gate.json")
    collapse = _read_json(REPORT_ROOT / "offline/reward_collapse.json")
    rse_sanity = _read_json(REPORT_ROOT / "offline/rse_sanity.json")
    source = _read_json(SOURCE_CHECKPOINT)
    historical = _read_json(HISTORICAL_AUTHORITY)
    positive = _read_json(POSITIVE_AUTHORITY)
    rows = _load_progression()
    u10 = _read_json(REPORT_ROOT / "training/U10/eval10/summary.json")
    if (
        offline.get("classification") != "MULTIPLICATIVE_RSE_OFFLINE_VALIDATED"
        or runtime.get("classification") != "RUNTIME_GRADIENT_SANITY_PASS"
        or u10["counts"]["PF"] != 0
        or u10["counts"]["lift"] != 6
    ):
        raise RuntimeError("DEXPLORE_CLOSEOUT_AUTHORITY_DRIFT")

    with (REPORT_ROOT / "offline/group_statistics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        group_rows = list(csv.DictReader(stream))
    positive_all = {
        metric: _group_stat(group_rows, "hocap_170650", "ALL", metric)
        for metric in ("R_obj", "R_hand", "R_int", "R_reg", "R_total")
    }
    positive_contact_lift = {
        metric: _group_stat(group_rows, "hocap_170650", "CONTACT_TO_LIFT", metric)
        for metric in ("R_obj", "R_hand", "R_int", "R_reg", "R_total")
    }
    negative_contact_lift = {
        metric: _group_stat(group_rows, "hocap_170105", "CONTACT_TO_LIFT", metric)
        for metric in ("R_obj", "R_hand", "R_int", "R_reg", "R_total")
    }
    positive_control = {
        "schema_version": "Stage16DexplorePositiveControlV1",
        "clip": "hocap_170650",
        "STATUS": positive["170650_FINAL_DECISION"],
        "PPO_RUN": False,
        "historical_pf": "20/20",
        "historical_df_pose": "20/20",
        "historical_df_linear": "20/20",
        "historical_df_angular_v2": "20/20",
        "offline_group_distribution_all": positive_all,
        "offline_group_distribution_contact_to_lift": positive_contact_lift,
        "pathological_under_new_reward": False,
        "collapse_hard_stop_triggered": False,
        "authority": str(POSITIVE_AUTHORITY.resolve()),
    }
    _write_json(REPORT_ROOT / "positive_control/v4_170650.json", positive_control)

    comparison_rows: list[dict[str, object]] = []
    for row in rows:
        comparison_rows.append(
            {
                "samples": int(row["samples"]),
                "additive_baseline_PF": "NOT_AVAILABLE_AT_MATCHED_BUDGET",
                "multiplicative_RSE_PF": f"{int(row['PF'])}/10",
                "additive_baseline_lift": "NOT_AVAILABLE_AT_MATCHED_BUDGET",
                "multiplicative_RSE_lift": f"{int(row['lift'])}/10",
            }
        )
    comparison_path = REPORT_ROOT / "comparison/additive_vs_multiplicative_rse.csv"
    _write_csv(comparison_path, comparison_rows)
    _write_text(
        REPORT_ROOT / "comparison/additive_vs_multiplicative_rse.md",
        """# Additive versus grouped-multiplicative/RSE

The frozen additive V4/170105 C4 source authority is PF/lift `0/10`. Historical
C4 additive training has no deterministic frame0 Eval10 snapshots at the
40,960--409,600 sample offsets used here, so the matched-budget cells are
explicitly `NOT_AVAILABLE_AT_MATCHED_BUDGET`; no additive rerun was performed.
The older multi-million-sample endpoint is not substituted for a matched early
update. The new combined method reaches lift `6/10` at 409,600 samples but PF
remains `0/10`, so this is partial functional progress, not acceptance.

Source authority:
`.local/reports/stage16_contact_timing_angular_twist_pf_df/final_summary.json`.
""",
    )

    candidate_trace = REPORT_ROOT / "training/U10/eval10/traces/episode_00.npz"
    historical_trace = REPO_ROOT / (
        ".local/sim_data/stage16_full_gravity_capability_closure/technical_remediation/"
        "smoke/former_timeout_v4_170105_c4/v4/hocap_170105/c4/episode_00.npz"
    )
    positive_trace = REPO_ROOT / (
        ".local/sim_data/stage16_full_gravity_capability_closure/formal20/"
        "v4_hocap_170650/episode_015.npz"
    )
    replay_prefix = (
        "OMNI_KIT_ACCEPT_EULA=YES conda run -n toporetarget-isaaclab python "
        "scripts/rl/isaaclab/replay_stage16d_simulation_trace.py --accept-eula --loop"
    )
    historical_replay = (
        f"{replay_prefix} --trace {historical_trace.relative_to(REPO_ROOT)} "
        "--object hocap_170105 --no-reference-ghost"
    )
    historical_window = (
        f"{historical_replay} --mocap-object-low-poly --start-frame 176 --end-frame 225"
    )
    candidate_replay = (
        f"{replay_prefix} --trace {candidate_trace.relative_to(REPO_ROOT)} "
        "--object hocap_170105 --no-reference-ghost"
    )
    candidate_window = (
        f"{candidate_replay} --mocap-object-low-poly --start-frame 176 --end-frame 225"
    )
    positive_replay = (
        f"{replay_prefix} --trace {positive_trace.relative_to(REPO_ROOT)} "
        "--object hocap_170650 --no-reference-ghost"
    )
    _write_text(
        REPORT_ROOT / "replay/visualization_commands.md",
        f"""# Replay commands

All commands show Actual plus raw MANO/object and hide the reference. `M`
toggles raw MOCAP and `R` toggles the reference; hidden layers stop live writes.

## 170105 historical failure

```bash
{historical_replay}
```

## 170105 historical CONTACT-to-LIFT

```bash
{historical_window}
```

## Best observed new 170105 candidate (U10, not accepted)

```bash
{candidate_replay}
```

## New U10 CONTACT-to-LIFT

```bash
{candidate_window}
```

## Accepted 170650 positive control

```bash
{positive_replay}
```
""",
    )
    _write_text(
        REPORT_ROOT / "replay/manual_acceptance.md",
        """# Manual acceptance boundary

U10 is the best observed 170105 result, but it is not an accepted checkpoint:
6/10 episodes lift, while all 10 fail PF because persistent multi-contact is
first established at frame 188 after LIFT at frame 184. Inspect actual/raw
contact timing, object-in-hand motion, table support transfer, and any visible
slip. Manual review cannot override PF or trigger Confirm20. The 170650 replay
remains the frozen accepted positive control and was not trained here.
""",
    )

    validation = {
        "ruff_check_task_files": args.ruff_check,
        "ruff_format_check_task_files": args.ruff_format,
        "mypy_src": args.mypy,
        "pytest_full": args.pytest,
        "paper_fidelity": args.paper_fidelity,
    }
    _write_json(
        REPORT_ROOT / "tests.json",
        {
            "schema_version": "Stage16DexploreValidationReceiptV1",
            "commands": {
                "ruff_check": "conda run -n toporetarget-rl ruff check <task-modified-files>",
                "ruff_format": (
                    "conda run -n toporetarget-rl ruff format --check <task-modified-files>"
                ),
                "mypy": "conda run -n toporetarget-rl python -m mypy src",
                "pytest": "conda run -n toporetarget-rl python -m pytest -q",
                "paper_fidelity": (
                    "conda run -n toporetarget-rl python scripts/check_paper_fidelity.py"
                ),
            },
            "results": validation,
            "all_pass": all(value == "PASS" for value in validation.values()),
        },
    )

    resource = _read_json(REPORT_ROOT / "resource_usage.json")
    resource.update(
        {
            "after": _nvidia_probe(),
            "actual_updates": 10,
            "actual_new_samples": 409600,
            "checkpoints_saved": 10,
            "exact_batches_saved": 10,
            "eval10_episodes": 100,
            "confirm20_episodes": 0,
        }
    )
    _write_json(REPORT_ROOT / "resource_usage.json", resource)

    commits = [
        {"commit": line.split(maxsplit=1)[0], "subject": line.split(maxsplit=1)[1]}
        for line in _git("log", "--format=%H %s", f"{BASE_COMMIT}..HEAD").splitlines()
        if line
    ]
    git_summary = {
        "ORIGINAL_BRANCH": "feature/ppo-physical",
        "BASE_COMMIT": BASE_COMMIT,
        "NEW_BRANCH": BRANCH,
        "DIRECT_SWITCH": True,
        "NEW_WORKTREE_CREATED": False,
        "FINAL_HEAD": _git("rev-parse", "HEAD"),
        "commits": commits,
        "worktree_clean": _git("status", "--short", "--untracked-files=all") == "",
        "PUSHED": False,
        "PR_CREATED": False,
    }
    _write_json(REPORT_ROOT / "git_commits.json", git_summary)

    best = rows[-1]
    failures = _technical_failures()
    final = {
        "schema_version": "Stage16DexploreRewardRSEFinalV1",
        "FINAL_CLASSIFICATION": "MULTIPLICATIVE_RSE_REFINEMENT_PARTIAL",
        "DEXPLORE_STYLE_REFINEMENT_SUCCESS": "NO",
        "OFFLINE_VALIDATION": "PASS",
        "PPO_TRAINING_RUN": "YES",
        "PPO_MAX_UPDATES": 10,
        "PPO_UPDATES_ACTUALLY_RUN": 10,
        "ACTUAL_SAMPLES": 409600,
        "DID_170105_IMPROVE": "PARTIALLY",
        "ACCEPTED_170105": "NO",
        "CONFIRM20_TRIGGERED": "NO",
        "best_observed": {
            "update": 10,
            "checkpoint": u10["checkpoint"],
            "checkpoint_sha256": u10["checkpoint_sha256"],
            "PF": "0/10",
            "lift": "6/10",
            "support_transfer": "0/10",
            "DF_pose": "7/10",
            "DF_linear": "6/10",
            "DF_angular_v2": "6/10",
            "first_contact_median": float(best["first_contact"]),
            "persistent_multi_contact_median": float(best["persistent_multi_contact"]),
            "LIFT": int(best["LIFT"]),
            "pre_LIFT_margin_median": float(best["pre_LIFT_margin"]),
            "R_int": float(best["R_int"]),
            "R_total": float(best["R_total"]),
            "kappa": float(best["kappa"]),
            "rse_termination_rate": float(best["RSE_termination_rate"]),
        },
        "multiplicative_reward_compensation_answer": "PARTIALLY",
        "rse_helped_without_destroying_rsi": "INCONCLUSIVE",
        "uniform_rsi_preserved": True,
        "positive_control": positive_control,
        "offline": {
            "gate": offline,
            "counterfactuals": str((REPORT_ROOT / "offline/counterfactuals.csv").resolve()),
            "reward_collapse": collapse,
            "rse_sanity": rse_sanity,
            "170105_contact_to_lift": negative_contact_lift,
            "170650_contact_to_lift": positive_contact_lift,
        },
        "runtime_sanity": runtime,
        "historical_additive_authority": {
            "PF": historical["PF_DF"]["hocap_170105"]["PF"],
            "lift": "0/10",
            "matched_budget_available": False,
            "rerun": False,
        },
        "technical_failures_preserved": len(failures),
        "tests": validation,
        "git": git_summary,
        "NEXT_ACTION": "NEXT_DIAGNOSE_MULTIPLICATIVE_RSE_RESIDUAL_FAILURE",
        "safety": {
            "BRANCH": BRANCH,
            "NEW_BRANCH_CREATED": "YES",
            "DIRECT_BRANCH_SWITCH": "YES",
            "NEW_WORKTREE_CREATED": "NO",
            "BASE_COMMIT": BASE_COMMIT,
            "SOURCE_PROFILE_FAILED_OBJECTIVE_INCLUDED": "NO",
            "UNIFORM_RSI_TRAINING": "YES",
            "170650_PPO_RUN": "NO",
            "LEGACY_ADDITIVE_REWARD_PRESERVED": "YES",
            "GROUPED_MULTIPLICATIVE_REWARD_IMPLEMENTED": "YES",
            "RSE_IMPLEMENTED": "YES",
            "FIXED_PRE_LIFT_GRASP_GATE_ADDED": "NO",
            "PHASE_HARD_GATE_ADDED": "NO",
            "PROFILE_REWARD_ADDED": "NO",
            "REWARD_WEIGHT_SWEEP_RUN": "NO",
            "LR_SWEEP_RUN": "NO",
            "EPOCH_SWEEP_RUN": "NO",
            "KL_SWEEP_RUN": "NO",
            "FRICTION_CHANGED": "NO",
            "MASS_CHANGED": "NO",
            "MATERIAL_CHANGED": "NO",
            "REFERENCE_CHANGED": "NO",
            "RETIMING_CHANGED": "NO",
            "CONTROLLER_CHANGED": "NO",
            "ACTION_CHANGED": "NO",
            "PF_CHANGED": "NO",
            "ANGULAR_AUTHORITY_V2_CHANGED": "NO",
            "GUIDANCE_ADDED": "NO",
            "OBJECT_STATE_WRITE_ADDED": "NO",
            "WRIST_ROOT_WRITE_ADDED": "NO",
            "HISTORICAL_ARTIFACTS_MODIFIED": "NO",
            "GUIDANCE_WORKTREE_MODIFIED": "NO",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            ".local_TRACKED": "NO",
        },
        "source_actor": source,
    }
    _write_json(REPORT_ROOT / "final_summary.json", final)

    ppo_table = _format_ppo_table(rows)
    handoff = f"""# Stage16 Dexplore-Style Multiplicative Reward + RSE Handoff

## 1. Git

`ORIGINAL_BRANCH=feature/ppo-physical`; `BASE_COMMIT={BASE_COMMIT}`;
`NEW_BRANCH={BRANCH}`; `DIRECT_SWITCH=YES`; `NEW_WORKTREE_CREATED=NO`.
`FINAL_HEAD={git_summary["FINAL_HEAD"]}`. Local commits are listed in
`git_commits.json`; `worktree_clean={str(git_summary["worktree_clean"]).upper()}`.

## 2. Dexplore Inspiration

Adopted: semantic reward groups, bounded group rewards, multiplication between
groups, and soft reference-scoped exploration. Not adopted: Dexplore's entire
runtime, Start-only initialization binding, repository code, or per-object
tuning. This is an inspired Stage16 adaptation, not an exact port.

## 3. Reward Formula

`R_obj = exp(-E_obj)`, where `E_obj` is the frozen weighted normalized object
axis/linear-twist/angular-twist squared error. `R_hand = exp(-w_scope E_hand)`,
where `E_hand` is the frozen weighted normalized link/finger/wrist error and
`w_scope=min(D_ref/0.20,1)`. `R_reg=exp(-0.01 smoothness_26d)`.
`R_total=exp(sum_g log(clamp(R_g,1e-12,1)))` with all four exponents fixed to 1.

## 4. Internal interaction mixture

When reference contact is expected, `R_int=0.5*(R_contact_v4+R_prox)` and
`R_prox=exp(-(1/0.03)*mean(max(d_actual-0.03,0)))`. When no contact is expected,
both internal terms are 1. There is no phase switch or hard pre-LIFT gate.

## 5. RSE Formula

`w_scope(D_ref)=clip(D_ref/0.20,0,1)`. `kappa=clip(N_fail/N_total,0.5,1)` from
initial counts 1/1, and each normalized deviation gate uses
`T_g(kappa)=kappa*T_g_base`. Only RSE deviation terminations count as failures;
normal completions increment the denominator and technical failures do neither.

## 6. Uniform RSI

`UNIFORM_RSI_PRESERVED=YES`; training resets remain uniform over `[0,320]`.
Frame0 is used only for deterministic full-trajectory evaluation.

## 7. Offline Validation

`PASS`. Counterfactual total means decrease from accepted CF0 `0.383141` to
delayed contact `0.368862`, missing contact `0.296276`, hand degradation
`0.260228`, and object degradation `0.004052`. Accepted 170650 CONTACT-to-LIFT
`R_total` mean/median are `0.549428/0.489130`, with fraction below `1e-6 = 0`;
the reward-collapse hard stop did not trigger. RSE monotonicity, kappa, clamp,
and uniform-RSI sanity checks pass.

## 8. Did PPO Run?

`YES`, after both offline and no-step runtime/gradient gates passed.

## 9. Updates

`ACTUAL_UPDATES=10`; `ACTUAL_SAMPLES=409600`. The hard U10 ceiling stopped the
run; no U11, sweep, or additional samples were executed.

{ppo_table}

## 10. Did 170105 Improve?

`PARTIALLY`. Lift improves from frozen historical `0/10` to U10 `6/10`, while
PF remains `0/10`. Historical additive C4 has no matched 40,960-step Eval10
snapshots, so it was not rerun and unavailable cells remain explicit.

## 11. Best 170105 Result

U10: PF `0/10`, lift `6/10`, support transfer `0/10`, DF pose `7/10`, DF linear
`6/10`, DF angular V2 `6/10`, geometry/causality `10/10`. Median first contact
is frame 184; persistent multi-contact is frame 188; LIFT is frame 184; median
pre-LIFT margin is `-4`. The checkpoint SHA is `{u10["checkpoint_sha256"]}`.

## 12. Confirm20

`TRIGGERED=NO`; no update reached the preregistered PF Eval10 trigger of 10/10.

## 13. Accepted 170105?

`NO`. Lift without pre-LIFT persistent multi-contact does not satisfy PF.

## 14. 170650 Positive Control

`STATUS=ACCEPTED_STAGE16_PHYSICAL_HOI`; `PPO_RUN=NO`. Historical PF, DF pose,
DF linear, and DF angular V2 remain 20/20. Offline grouped-reward evaluation is
finite and non-pathological; its CONTACT-to-LIFT `R_total` has mean `0.549428`.

## 15. Final Classification

`MULTIPLICATIVE_RSE_REFINEMENT_PARTIAL`.

## 16. Does Multiplicative Reward Solve Reward Compensation?

`PARTIALLY`. The offline soft-AND counterfactuals are load-bearing and U10
interaction reward rises to `{float(best["R_int"]):.6f}`, but the final policy
still lacks pre-LIFT persistent grasp and PF remains 0/10.

## 17. Did RSE Help Exploration Without Destroying RSI?

`INCONCLUSIVE`. Uniform RSI is intact and lift improves, but the experiment
tests reward aggregation and RSE jointly, so RSE's causal contribution is not
identifiable. U10 kappa is `{float(best["kappa"]):.6f}` and RSE termination rate
is `{float(best["RSE_termination_rate"]):.6f}`.

## 18. Is Per-object Tuning Needed?

`PER_OBJECT_REWARD_TUNING=NO`; `PER_OBJECT_FRICTION_TUNING=NO`;
`MANUAL_GRASP_FRAME=NO`. This task provides no evidence authorizing them.

## 19. Replay

See `replay/visualization_commands.md` for historical 170105, best observed U10,
before/after CONTACT-to-LIFT, and accepted 170650 commands. U10 is explicitly
not an accepted candidate.

## 20. Next Action

Only `NEXT_DIAGNOSE_MULTIPLICATIVE_RSE_RESIDUAL_FAILURE`; it is not executed here.

## 21. Safety Flags

All requested flags are recorded verbatim in `final_summary.json`. In summary:
the requested branch/base/direct-switch boundaries hold; legacy additive mode,
uniform RSI, physics, controller, action, reference, PF, and Authority V2 are
preserved; no SourceProfile objective, phase/pre-LIFT hard gate, tuning sweep,
guidance, state write, worktree, push, PR, or tracked `.local` artifact was added.
The three technical Eval10 failures were retained before the corrected U1
evaluation; none caused an extra optimizer update.
"""
    _write_text(REPORT_ROOT / "handoff.md", handoff)
    _write_text(REPORT_ROOT / "final_summary.md", handoff)
    print(json.dumps({"classification": final["FINAL_CLASSIFICATION"], "git": git_summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
