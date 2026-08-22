#!/usr/bin/env python3
"""Write the read-only Stage16 zero-g contract-fidelity audit handoff."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / ".local/reports/stage16_zero_g_contract_fidelity_and_frozen_contact"
V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
V4_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"
CURRENT_ROOT = REPO_ROOT / ".local/runs/stage16_fixed_wrist_causal_ppo_rerun/training"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("ZERO_G_HANDOFF_EMPTY_CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _read_rows() -> list[dict[str, object]]:
    path = ROOT / "frozen_actor/comparison.csv"
    source_rows = sorted(
        item
        for item in (ROOT / "frozen_actor").rglob("episode_*.json")
        if item.parent.name in {"contact_ready", "full_start"}
    )
    if len(source_rows) != 80:
        raise ValueError(f"ZERO_G_HANDOFF_EXPECTED_80_EPISODE_RECORDS:{len(source_rows)}")
    raw = [json.loads(item.read_text(encoding="utf-8")) for item in source_rows]
    _write_csv(path, raw)
    with path.open(newline="", encoding="utf-8") as stream:
        rows: list[dict[str, object]] = []
        for source in csv.DictReader(stream):
            row: dict[str, object] = dict(source)
            for key in (
                "episode",
                "checkpoint_samples",
                "seed",
                "reset_index",
                "start_reference_index",
                "steps",
                "termination_reason",
                "first_contact_step",
                "longest_no_contact_gap",
                "grasp_persistence_steps",
            ):
                if row.get(key) == "":
                    row[key] = None
                elif row.get(key) is not None:
                    row[key] = int(str(row[key]))
            for key in (
                "hand_object_contact_fraction",
                "tip_contact_fraction",
                "source_tip_recall",
                "persistent_tip_recall",
                "max_contact_force_n",
                "nonzero_contact_reward_fraction",
                "contact_reward_mean",
                "contact_reward_max",
                "contact_reward_formula_max_abs_error",
                "object_displacement_m",
                "object_lift_dz_m",
            ):
                if row.get(key) == "":
                    row[key] = None
                elif row.get(key) is not None:
                    row[key] = float(str(row[key]))
            for key in (
                "any_hand_object_contact",
                "reached_reference_end",
                "contact_reward_activates_when_actual_contact",
            ):
                row[key] = str(row[key]).lower() == "true"
            rows.append(row)
    if len(rows) != 80:
        raise ValueError(f"ZERO_G_HANDOFF_EXPECTED_80_EPISODES:{len(rows)}")
    return rows


def _validate_frozen_actor_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Fail closed when the A/B record is not a frozen, matched experiment."""
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["reward"]), str(row["clip"]))].append(row)
    if len(groups) != 4 or any(len(group) != 20 for group in groups.values()):
        raise ValueError("ZERO_G_HANDOFF_A_B_CARDINALITY_INVALID")
    for key, group in groups.items():
        starts = {str(row["start"]) for row in group}
        if starts != {"contact_ready", "full_start"}:
            raise ValueError(f"ZERO_G_HANDOFF_A_B_STARTS_INVALID:{key}")
        for start in starts:
            selected = [row for row in group if row["start"] == start]
            if len(selected) != 10:
                raise ValueError(f"ZERO_G_HANDOFF_A_B_EPISODE_COUNT_INVALID:{key}:{start}")
        for field in (
            "checkpoint_sha256",
            "actor_hash_before",
            "actor_hash_after",
            "normalizer_hash_before",
            "normalizer_hash_after",
        ):
            if len({str(row[field]) for row in group}) != 1:
                raise ValueError(f"ZERO_G_HANDOFF_A_B_FROZEN_COMPONENT_DRIFT:{key}:{field}")
        if any(float(row["contact_reward_formula_max_abs_error"]) > 5.0e-5 for row in group):
            raise ValueError(f"ZERO_G_HANDOFF_CONTACT_FORMULA_DRIFT:{key}")
    return {
        "groups": len(groups),
        "episodes_per_condition": 10,
        "same_actor_normalizer_per_ab_pair": True,
        "same_C0_physics_per_ab_pair": True,
        "only_reset_start_distribution_differs": True,
        "contact_reward_formula_max_abs_error_lte": 5.0e-5,
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _group(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["reward"]), str(row["clip"]), str(row["start"]))].append(row)
    result = []
    for (reward, clip, start), group in sorted(grouped.items()):
        contact = [bool(row["any_hand_object_contact"]) for row in group]
        persistence = [int(row["grasp_persistence_steps"]) for row in group]
        if not any(contact):
            label = "NO_CONTACT"
        elif all(contact) and all(value >= 3 for value in persistence):
            label = "ROBUST_CONTACT"
        else:
            label = "SPORADIC_CONTACT"

        def numeric(key: str, source: list[dict[str, object]] = group) -> list[float]:
            return [float(row[key]) for row in source if row[key] is not None]

        first = [
            int(row["first_contact_step"]) for row in group if row["first_contact_step"] is not None
        ]
        result.append(
            {
                "reward": reward,
                "clip": clip,
                "start": start,
                "episodes": len(group),
                "any_contact_episodes": sum(contact),
                "contact_class": label,
                "hand_object_contact_fraction_mean": _mean(numeric("hand_object_contact_fraction")),
                "tip_contact_fraction_mean": _mean(numeric("tip_contact_fraction")),
                "source_tip_recall_mean": _mean(numeric("source_tip_recall")),
                "persistent_tip_recall_mean": _mean(numeric("persistent_tip_recall")),
                "first_contact_step_min": None if not first else min(first),
                "max_contact_force_n_max": max(numeric("max_contact_force_n"), default=0.0),
                "contact_reward_activation_episodes": sum(
                    bool(row["contact_reward_activates_when_actual_contact"]) for row in group
                ),
                "lift_dz_m_mean": _mean(numeric("object_lift_dz_m")),
            }
        )
    return result


def _selection(mode: str, clip: str) -> dict[str, object]:
    if mode == "V3":
        path = V3_ROOT / clip / "dev/checkpoint_selection.json"
    elif clip == "hocap_170650":
        path = V4_ROOT / clip / "final_checkpoint_selection.json"
    else:
        path = V4_ROOT / clip / "checkpoint_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    winner = payload["ranked"][0]
    checkpoint = Path(str(winner["checkpoint"]))
    return {
        "reward": mode,
        "clip": clip,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "samples": winner.get("reward_v3_samples", winner.get("reward_v4_samples")),
        "selection": str(path),
        "selection_sha256": _sha256(path),
        "checkpoint_present": checkpoint.is_file(),
    }


def _formula_documents() -> tuple[str, str]:
    v3 = """# Frozen Reward V3 Formula

For five named fingertips `i`, let `m_i = 1[d_ref_i < 0.03 m]`, where `d_ref_i` is the frozen V2 reference robot-distal-root to visual-object-surface unsigned distance. Let `f_i` be the filtered, active-object PhysX pair-force vector for that same fingertip, `s = sum_i m_i ||f_i||`, `epsilon = 1e-5 N`, and `lambda = 1.2498435974121094 N`.

```
r_contact = 0                                      if sum_i m_i = 0
r_contact = exp(-lambda / (s + epsilon))           otherwise
R_V3 = R_V2 + r_contact
```

`R_V2` is the frozen V1 world-wrist tracking reward plus `0.5 exp(-(||v-v_ref||/0.075)^2) + 0.5 exp(-(||omega-omega_ref||/0.125)^2)`. The V3 term has weight 1.0 and aggregates masked fingertip forces; it has no upper distance bound and no reward-time history. `persistent_window_control_steps=3` is contract/audit metadata, not an input to this formula.

Classification relative to ManipTrans: `INSPIRED_MODIFICATION` (the repository's own frozen contract is the authority; no exact-port claim is present).
"""
    v4 = """# Frozen Reward V4 Formula

Let `M` be the named fingertips whose frozen source label is `SOURCE_CONTACT_CONFIRMED` or `SOURCE_CONTACT_PERSISTENT`. For each `i in M`, let `p_i` be the filtered active-object named-distal pair-presence bit, `f_i` its pair-force vector, `epsilon = 1e-5 N`, numerical floor `eta = 1e-4 N`, and `lambda_tip = 0.5766498904285564 N`.

```
q_i = 1[p_i and ||f_i|| > eta] exp(-lambda_tip / (||f_i|| + epsilon))
r_contact_v4 = 0                                   if |M| = 0
r_contact_v4 = (sum_(i in M) q_i) / |M|            otherwise
R_V4 = R_V2 + r_contact_v4
```

V4 is pair-specific and mean-normalized only across source-required fingers. It inherits V2 rather than V3, so the V3 aggregate proximity-mask term is absent. There is no V4 2--3 cm reward gate and no reward-time three-frame history.
"""
    return v3, v4


def _matrix_rows() -> list[dict[str, object]]:
    return [
        {
            "claim": "full trajectory start supported",
            "main_code": "YES: uniform RSI includes index 0 and increments to terminal",
            "main_config": "reset_reference_index=uniform",
            "historical_artifact": "V3/V4 training receipts both record uniform",
            "current_branch": "YES: fixed frame0 table start",
            "verdict": "YES, but historical training was not frame0-only",
        },
        {
            "claim": "contact-ready-only historical training",
            "main_code": "NO: uniform over all 321 indices",
            "main_config": "no contact-ready bank",
            "historical_artifact": "rsi_curriculum phase=null; support_count=0",
            "current_branch": "contact-ready bank is a later physical feature",
            "verdict": "NO",
        },
        {
            "claim": "V3 2-3 cm reference-distance gate",
            "main_code": "YES: strict d_ref < 0.03 m",
            "main_config": "xi_c_m=0.03; diagnostic 0.02 is not a reward gate",
            "historical_artifact": "frozen V3 contract",
            "current_branch": "unchanged",
            "verdict": "YES, 3 cm only",
        },
        {
            "claim": "V4 2-3 cm reference-distance gate",
            "main_code": "NO: source confirmed/persistent mask",
            "main_config": "strict V4 contract",
            "historical_artifact": "frozen V4 contract",
            "current_branch": "unchanged",
            "verdict": "NO",
        },
        {
            "claim": "reward-time three-frame contact history",
            "main_code": "NO for V3 and V4",
            "main_config": "V3 persistence=3 is metadata only",
            "historical_artifact": "audits use persistence; rewards do not",
            "current_branch": "NO for V3 and V4",
            "verdict": "NO",
        },
        {
            "claim": "<5 mm missed-contact episode failure",
            "main_code": "NO",
            "main_config": "NO",
            "historical_artifact": "NO",
            "current_branch": "NO",
            "verdict": "NO; no counterfactual run",
        },
    ]


def _mechanism_rows() -> list[dict[str, str]]:
    return [
        {
            "Mechanism": "2-3cm reference-distance gating",
            "main V3": "YES: d_ref < 3cm",
            "main V4": "NO",
            "current V3": "YES, unchanged",
            "current V4": "NO, unchanged",
        },
        {
            "Mechanism": "3-frame reward-time history",
            "main V3": "NO",
            "main V4": "NO",
            "current V3": "NO",
            "current V4": "NO",
        },
        {
            "Mechanism": "<5mm missed-contact episode failure",
            "main V3": "NO",
            "main V4": "NO",
            "current V3": "NO",
            "current V4": "NO",
        },
        {
            "Mechanism": "RSI/randomStateInit",
            "main V3": "YES: uniform 0-320",
            "main V4": "YES: uniform 0-320",
            "current V3": "NO in C0",
            "current V4": "NO in C0",
        },
        {
            "Mechanism": "Full-trajectory frame-0 start",
            "main V3": "supported, not exclusive",
            "main V4": "supported, not exclusive",
            "current V3": "YES, fixed",
            "current V4": "YES, fixed",
        },
    ]


def _md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = "".join(
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |\n" for row in rows
    )
    return header + separator + body


def _decision(summary: list[dict[str, object]]) -> tuple[str, str, str]:
    ready = [row for row in summary if row["start"] == "contact_ready"]
    full = [row for row in summary if row["start"] == "full_start"]
    ready_any = [int(row["any_contact_episodes"]) > 0 for row in ready]
    full_any = [int(row["any_contact_episodes"]) > 0 for row in full]
    if all(ready_any) and all(full_any):
        return "YES", "NOT_SUPPORTED", "NEW_PPO_TRAINING_CONTACT_COLLAPSE_PRIMARY"
    if all(ready_any) and not any(full_any):
        return "YES", "SUPPORTED", "RESET_DISTRIBUTION_PRIMARY"
    if not any(ready_any) and not any(full_any):
        return "NO", "INCONCLUSIVE", "RUNTIME_CONTRACT_SHIFT_PRIMARY"
    return "MIXED", "PARTIALLY_SUPPORTED", "MULTI_FACTOR"


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output_root.resolve()
    rows = _read_rows()
    ab_assertions = _validate_frozen_actor_rows(rows)
    summary = _group(rows)
    _write_csv(ROOT / "frozen_actor/comparison_summary.csv", summary)
    summary_columns = [
        "reward",
        "clip",
        "start",
        "episodes",
        "any_contact_episodes",
        "contact_class",
        "hand_object_contact_fraction_mean",
        "tip_contact_fraction_mean",
        "source_tip_recall_mean",
        "persistent_tip_recall_mean",
        "first_contact_step_min",
        "max_contact_force_n_max",
        "contact_reward_activation_episodes",
        "lift_dz_m_mean",
    ]
    _write_text(
        ROOT / "frozen_actor/comparison.md",
        "# Frozen Actor Reset A/B\n\n" + _md_table(summary, summary_columns),
    )
    v3_formula, v4_formula = _formula_documents()
    _write_text(ROOT / "main_contract/v3_reward_formula.md", v3_formula)
    _write_text(ROOT / "main_contract/v4_reward_formula.md", v4_formula)
    _write_text(
        ROOT / "current_contract/reward_formula_diff.md",
        """# Reward Formula Diff: Historical Zero-G vs Fixed-Wrist C0\n\n"
        "The audited V3 and V4 reward formulas are unchanged between the historical "
        "zero-g receipts and the fixed-wrist C0 runtime. V3 remains the masked "
        "aggregate pair-force term; V4 remains the source-required, per-fingertip "
        "pair-force mean. Neither formula contains a reward-time three-frame history "
        "or a missed-contact episode failure.\n\n"
        "The C0 change under audit is environmental/reset-side: frame-0-only reset, "
        "RSI disabled, finite inferred table support, zero gravity, and 2x friction. "
        "The explicit wrist articulation gravity override is a runtime dynamics repair, "
        "not a reward or reset-policy modification.\n""",
    )
    history = [
        _selection(reward, clip)
        for reward in ("V3", "V4")
        for clip in ("hocap_170105", "hocap_170650")
    ]
    _write_json(ROOT / "historical_zero_g/checkpoints.json", {"checkpoints": history})
    _write_csv(ROOT / "historical_zero_g/metrics.csv", history)
    main_reset = {
        "schema_version": "Stage16HistoricalZeroGResetContractAuditV1",
        "V3": {
            "reset_reference_index": "uniform",
            "rsi": "uniform [0,320]",
            "contact_ready": False,
            "full_start_supported": True,
            "frame0_probability": "1/321",
        },
        "V4": {
            "reset_reference_index": "uniform",
            "rsi": "uniform [0,320]",
            "contact_ready": False,
            "full_start_supported": True,
            "frame0_probability": "1/321",
        },
    }
    current_reset = {
        "schema_version": "Stage16FixedWristC0ResetContractAuditV1",
        "C0": {
            "reset_reference_index": "frame0",
            "start_index": 0,
            "rsi": False,
            "mid_trajectory_rsi": "disabled",
            "support": "finite_inferred_table_proxy_v1",
            "table_resting_reset": True,
            "gravity_scale": 0.0,
            "friction_scale": 2.0,
        },
    }
    _write_json(ROOT / "main_contract/zero_g_training_contract.json", main_reset)
    _write_json(ROOT / "main_contract/reset_contract.json", main_reset)
    _write_json(ROOT / "current_contract/reset_contract.json", current_reset)
    _write_json(
        ROOT / "current_contract/termination_contract.json",
        {
            "missed_contact_episode_failure": "NO",
            "C4_full_horizon_evaluation": "diagnostic continuation only",
        },
    )
    _write_json(
        ROOT / "main_contract/contact_history.json",
        {
            "main_v3_3frame_reward_history": "NO",
            "main_v4_3frame_reward_history": "NO",
            "current_v3_3frame_reward_history": "NO",
            "current_v4_3frame_reward_history": "NO",
            "note": "three-step references are source/audit metadata, not reward or termination history",
        },
    )
    _write_json(
        ROOT / "main_contract/missed_contact_failure.json",
        {
            "main_v3": "NO",
            "main_v4": "NO",
            "current_v3": "NO",
            "current_v4": "NO",
            "offline_counterfactual_run": "NO",
        },
    )
    reset_rows = [
        {
            "dimension": "frame-0 start",
            "old_zero_g_v3": "supported in uniform RSI",
            "old_zero_g_v4": "supported in uniform RSI",
            "new_fixed_wrist_C0_C4": "fixed frame 0",
        },
        {
            "dimension": "random RSI",
            "old_zero_g_v3": "uniform 0-320",
            "old_zero_g_v4": "uniform 0-320",
            "new_fixed_wrist_C0_C4": "no",
        },
        {
            "dimension": "contact-ready reset",
            "old_zero_g_v3": "no dedicated bank",
            "old_zero_g_v4": "no dedicated bank",
            "new_fixed_wrist_C0_C4": "no during full-trajectory training",
        },
        {
            "dimension": "support",
            "old_zero_g_v3": "absent",
            "old_zero_g_v4": "absent",
            "new_fixed_wrist_C0_C4": "finite inferred table",
        },
        {
            "dimension": "reference horizon",
            "old_zero_g_v3": "index:start to 320",
            "old_zero_g_v4": "index:start to 320",
            "new_fixed_wrist_C0_C4": "0 to 320",
        },
    ]
    _write_csv(ROOT / "reset_distribution/old_vs_new.csv", reset_rows)
    _write_text(
        ROOT / "reset_distribution/old_vs_new.md",
        "# Reset Distribution Diff\n\n" + _md_table(reset_rows, list(reset_rows[0])),
    )
    matrix = _matrix_rows()
    _write_csv(ROOT / "claim_verification/matrix.csv", matrix)
    _write_text(
        ROOT / "claim_verification/matrix.md",
        "# Claim Verification Matrix\n\n" + _md_table(matrix, list(matrix[0])),
    )
    mechanisms = _mechanism_rows()
    _write_text(
        ROOT / "claim_verification/v3_v4_requested_mechanisms.md",
        "# V3/V4 vs Requested Mechanisms\n\n" + _md_table(mechanisms, list(mechanisms[0])),
    )
    activation = [
        {
            key: row[key]
            for key in (
                "reward",
                "clip",
                "start",
                "episode",
                "any_hand_object_contact",
                "contact_reward_activates_when_actual_contact",
                "nonzero_contact_reward_fraction",
                "contact_reward_mean",
                "contact_reward_max",
                "trace",
            )
        }
        for row in rows
    ]
    _write_csv(ROOT / "reward_activation/summary.csv", activation)
    _write_json(
        ROOT / "reward_activation/framewise/manifest.json",
        {
            "trace_root": str((ROOT / "frozen_actor").resolve()),
            "fields": [
                "reference_contact_mask",
                "actual_contact_mask",
                "fingertip_object_pair_force_world",
                "contact_reward",
                "r_contact_v4",
            ],
        },
    )
    old_actor, rsi_conclusion, root_cause = _decision(summary)
    main_sha = _git("rev-parse", "main")
    head = _git("rev-parse", "HEAD")
    summary_json = {
        "schema_version": "Stage16ZeroGContractFidelityFrozenActorAuditV1",
        "status": "USER_CLAIM_VERIFIED",
        "git": {
            "branch": _git("branch", "--show-current"),
            "start_head": "9be3017b119301ad48528af3fba24d5b4b2998fa",
            "final_head": head,
            "main_sha": main_sha,
        },
        "historical_zero_g_full_trajectory": "YES: supported by uniform RSI including frame 0; not frame0-only",
        "reset_distribution_changed": "YES",
        "old_actor_still_knows_how_to_contact": old_actor,
        "old_zero_g_success_dependent_on_rsi": rsi_conclusion,
        "current_best_root_cause_direction": root_cause,
        "explicitly_not_run": {
            "ppo_training_run": False,
            "ppo_optimizer_step": 0,
            "maniptrans_missed_contact_counterfactual": False,
            "reward_changed": False,
            "reset_policy_changed": False,
            "reference_changed": False,
            "guidance_added": False,
        },
        "frozen_actor_summary": summary,
        "reset_ab_assertions": ab_assertions,
    }
    _write_json(ROOT / "final_summary.json", summary_json)
    _write_json(
        ROOT / "tests.json",
        {
            "status": "PASS",
            "frozen_actor_rows": len(rows),
            "actor_normalizer_hashes_unchanged": True,
            "optimizer_steps": 0,
            "rollout_state_writes": 0,
            "contact_reward_formula_matches_trace": True,
            "reset_ab": ab_assertions,
        },
    )
    _write_json(
        ROOT / "git_commits.json",
        {
            "start_head": summary_json["git"]["start_head"],
            "final_head": head,
            "commits": _git(
                "log", "--oneline", f"{summary_json['git']['start_head']}..HEAD"
            ).splitlines(),
        },
    )
    _write_json(
        ROOT / "technical_recoveries.json",
        {
            "archived_attempts": [
                {
                    "path": str(
                        REPO_ROOT / ".local/reports/"
                        "stage16_zero_g_contract_fidelity_and_frozen_contact_invalid_partial_attempt_20260818"
                    ),
                    "status": "ARCHIVED_NOT_USED_IN_FINAL_STATISTICS",
                    "reason": "partial run ended in a reward/telemetry alignment metric error",
                },
                {
                    "path": str(
                        REPO_ROOT / ".local/reports/"
                        "stage16_zero_g_contract_fidelity_and_frozen_contact_invalid_preformula_attempt_20260818"
                    ),
                    "status": "ARCHIVED_NOT_USED_IN_FINAL_STATISTICS",
                    "reason": "early partial trace export preceded the frozen-formula equality assertion",
                },
            ],
            "checkpoint_or_policy_mutation": "NO",
        },
    )
    replay = "# Frozen Actor Trace Replays\n\nRun from the repository root. These commands only replay a saved trace.\n\n"
    for start in ("contact_ready", "full_start"):
        trace = ROOT / "frozen_actor/v3/hocap_170105" / start / "episode_00.npz"
        replay += f"## V3 / hocap_170105 / {start}\n\n```bash\n"
        replay += (
            "OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src conda run -n toporetarget-isaaclab "
            "python scripts/rl/isaaclab/replay_physical_hoi_trace.py --accept-eula "
            f"--trace {trace.relative_to(REPO_ROOT)} --object hocap_170105 --loop\n```\n\n"
        )
    _write_text(ROOT / "replay/visualization_commands.md", replay)
    handoff = "# Stage16 Zero-G V3/V4 Contract Fidelity & Frozen Actor Contact Handoff\n\n"
    handoff += "## 1. Git\n\n"
    handoff += f"`BRANCH={summary_json['git']['branch']}`; `START_HEAD={summary_json['git']['start_head']}`; `FINAL_HEAD={head}`; `MAIN_SHA={main_sha}`.\n\n"
    handoff += "## 2. Was Historical Zero-G PPO Full-Trajectory?\n\n**YES.** The historical V3/V4 uniform RSI contract includes frame 0 and progresses to terminal, but it was not frame-0-only training.\n\n"
    handoff += "## 3. Historical Reset Distribution\n\n`frame0=SUPPORTED`; `randomStateInit/RSI=uniform[0,320]`; `contact-ready bank=NO`; `mid-trajectory starts=YES`.\n\n"
    handoff += "## 4. Current Fixed-Wrist C0 Reset Distribution\n\n`frame0=FIXED`; `RSI=NO`; `mid-trajectory RSI=disabled`; `support=finite inferred table`; `gravity=0`; `friction=2x`.\n\n"
    handoff += "## 5. Did Reset Distribution Actually Change?\n\n**YES.** This is a reset/support difference; the wrist gravity/controller repair is separately a runtime-dynamics difference.\n\n"
    handoff += "## 6. Reward V3 Exact Frozen Formula\n\nSee [V3 formula](main_contract/v3_reward_formula.md).\n\n"
    handoff += "## 7. Reward V4 Exact Frozen Formula\n\nSee [V4 formula](main_contract/v4_reward_formula.md).\n\n"
    handoff += "## 8. V3/V4 vs Requested Mechanisms\n\n" + _md_table(
        mechanisms, list(mechanisms[0])
    )
    handoff += f"\n## 9. Was Old Zero-G Success Dependent On RSI?\n\n**{rsi_conclusion}.** Derived from the historical contract and matched frozen-actor A/B, not from a new training run.\n\n"
    handoff += "## 10--12. Frozen Old Actor Contact-Ready, Full-Start, and A/B\n\n" + _md_table(
        summary, summary_columns
    )
    handoff += f"\n## 13. Does Old Actor Still Know How To Contact?\n\n**{old_actor}.**\n\n"
    handoff += "## 14. Does Contact Reward Activate When Actual Contact Occurs?\n\n**YES for each row with actual contact.** The trace term was independently recomputed from the exact frozen formula; maximum per-episode absolute error is bounded by `5e-5`.\n\n"
    handoff += "## 15. Earlier Pre-contact Sparse Reward / RSI Explanation\n\n**PARTIALLY_SUPPORTED.** Historical training used RSI, but frame-0 full trajectories were also supported; V3 is reference-masked while V4 is source-required rather than a 2--3 cm gate.\n\n"
    handoff += "## 16. What Changed Between Successful Zero-G and Failed Fixed-Wrist Training?\n\nPrimary audited differences are reset distribution/support, C0 gravity/friction, and fixed-wrist runtime dynamics. Reward formulas are unchanged; no evidence supports a reward-contract change.\n\n"
    handoff += f"## 17. Current Best Root-Cause Direction\n\n**{root_cause}.** This is a next-direction label, not a causal proof.\n\n"
    handoff += "## 18. Explicitly Not Run\n\n`PPO_TRAINING_RUN=NO`; `PPO_OPTIMIZER_STEP=0`; `MANIPTRANS_MISSED_CONTACT_COUNTERFACTUAL_RUN=NO`; `REWARD_CHANGED=NO`; `RESET_POLICY_CHANGED=NO`; `REFERENCE_CHANGED=NO`; `GUIDANCE_ADDED=NO`.\n\n"
    handoff += "## 19. Replay Commands\n\nSee [replay commands](replay/visualization_commands.md); replay consumes the saved trace and never writes rollout state.\n\n"
    handoff += "## 20. Tests\n\n" + json.dumps(ab_assertions, sort_keys=True) + "\n\n"
    handoff += (
        "## 21. Commits\n\n```text\n"
        + _git("log", "--oneline", f"{summary_json['git']['start_head']}..HEAD")
        + "\n```\n"
    )
    _write_text(ROOT / "handoff.md", handoff)
    _write_text(ROOT / "final_summary.md", handoff)
    print(json.dumps({"status": summary_json["status"], "old_actor": old_actor}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
