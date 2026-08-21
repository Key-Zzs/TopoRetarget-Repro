#!/usr/bin/env python3
# ruff: noqa: E402, E501, I001
"""Assemble the bounded Stage16 PF V2 symmetric-PPO evidence report.

The script reads immutable sources plus experiment-local receipts.  It does
not create a simulator, collect a rollout, or update a checkpoint.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.rl.isaaclab.run_stage16_pf_v2_symmetric_ppo import (
    assert_symmetric_static_contracts,
)

REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo"
RUN_ROOT = REPO_ROOT / ".local/runs/stage16_pf_v2_causal_lift_and_symmetric_ppo"
CLIPS = ("hocap_170105", "hocap_170650")
U10_SOURCE_SHA256 = "58f18d934679de4a9759a91cb6b0c296e2357eeff88d23dc9915f85e72cceb95"
HISTORICAL_170650_SHA256 = "80da5a3c2c953483f9fe5a668dfe2d4b4c458ab451836ad4b179fec28d0979f3"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"PF_V2_FINAL_REPORT_MISSING:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PF_V2_FINAL_REPORT_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _read_progression(clip: str) -> list[dict[str, str]]:
    path = REPORT_ROOT / "training" / clip / "progression.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"PF_V2_FINAL_REPORT_EMPTY_PROGRESSION:{clip}")
    return rows


def _write_combined_progression(progressions: dict[str, list[dict[str, str]]]) -> None:
    rows = [{"clip": clip, **row} for clip in CLIPS for row in progressions[clip]]
    fieldnames = list(rows[0])
    path = REPORT_ROOT / "comparison/combined_progression.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compact_eval(summary: dict[str, Any]) -> dict[str, object]:
    return {
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "update": summary["update"],
        "stage_samples": summary["samples"],
        "optimizer_steps": summary["optimizer_steps"],
        "counts": summary["counts"],
        "group_means": summary["group_means"],
        "timing": summary["timing"],
        "lift_dz_mean_m": summary["lift_dz_mean_m"],
    }


def _require_eval20(summary: dict[str, Any], *, clip: str, pf_v1: int) -> None:
    counts = summary["counts"]
    required = (
        "PF_V2",
        "physical_lift",
        "causal_lift",
        "support_transfer",
        "sustained_hand_object_coupling",
        "DF_pose",
        "DF_linear",
        "DF_angular_v2",
    )
    if (
        summary.get("optimizer_steps") != 0
        or any(counts.get(key) != 20 for key in required)
        or counts.get("PF_V1") != pf_v1
    ):
        raise RuntimeError(f"PF_V2_FINAL_REPORT_EVAL20_INVALID:{clip}")


def _history_rows(
    *,
    source_170105: dict[str, Any],
    source_170650: dict[str, Any],
    final_170105: dict[str, Any],
    final_170650: dict[str, Any],
) -> list[dict[str, object]]:
    return [
        {
            "lineage": "historical_170105_U10",
            "clip": "hocap_170105",
            "role": "experimental_source",
            "checkpoint_sha256": source_170105["checkpoint_sha256"],
            "new_updates": 0,
            "new_samples": 0,
            "PF_V1": 0,
            "PF_V2": 10,
            "note": "U10 frozen-trace re-evaluation; U10 Eval20 was no-step PF V2 20/20.",
        },
        {
            "lineage": "170105_U11",
            "clip": "hocap_170105",
            "role": "selected_experimental",
            "checkpoint_sha256": final_170105["checkpoint_sha256"],
            "new_updates": 1,
            "new_samples": 40960,
            "PF_V1": final_170105["counts"]["PF_V1"],
            "PF_V2": final_170105["counts"]["PF_V2"],
            "note": "Confirm20 selected checkpoint; PF V1 is deliberately unchanged.",
        },
        {
            "lineage": "historical_accepted_170650",
            "clip": "hocap_170650",
            "role": "frozen_positive_control_source",
            "checkpoint_sha256": source_170650["checkpoint_sha256"],
            "new_updates": 0,
            "new_samples": 0,
            "PF_V1": 20,
            "PF_V2": 20,
            "note": "Historical actor remains frozen and is not overwritten.",
        },
        {
            "lineage": "170650_U02",
            "clip": "hocap_170650",
            "role": "best_observed_experimental",
            "checkpoint_sha256": final_170650["checkpoint_sha256"],
            "new_updates": 2,
            "new_samples": 81920,
            "PF_V1": final_170650["counts"]["PF_V1"],
            "PF_V2": final_170650["counts"]["PF_V2"],
            "note": (
                "First maximum-score experimental checkpoint; U3--U8 regress before U10 recovers "
                "on Eval10."
            ),
        },
    ]


def _write_history(rows: list[dict[str, object]]) -> None:
    path = REPORT_ROOT / "comparison/source_vs_experimental.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_replay_commands() -> None:
    root = REPORT_ROOT
    _write_text(
        root / "replay/visualization_commands.md",
        f"""# Stage16 PF V2 replay commands

Replay is visualization only.  The translucent table is the frozen inferred
support proxy; it does not recompute collision/contact telemetry.  Support
claims below come from recorded `table_object_contact` traces, whose reset
sample is independent of the hand-pair force-validity stream.

```bash
# Historical accepted 170650 source (the user-facing reference command)
OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \\
  --accept-eula --object hocap_170650 \\
  --trace .local/sim_data/stage16_full_gravity_capability_closure/formal20/v4_hocap_170650/episode_000.npz \\
  --loop --speed 2.0 --mocap-object-low-poly

# U11 170105 Confirm20 selected experimental checkpoint, episode 00
OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \\
  --accept-eula --object hocap_170105 \\
  --trace {root}/training/hocap_170105/confirm20/U11/traces/episode_00.npz \\
  --loop --speed 2.0 --mocap-object-low-poly

# U02 170650 best-observed experimental checkpoint, episode 00
OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 conda run -n toporetarget-isaaclab python scripts/rl/isaaclab/replay_stage16d_simulation_trace.py \\
  --accept-eula --object hocap_170650 \\
  --trace {root}/training/hocap_170650/best_eval20/traces/episode_00.npz \\
  --loop --speed 2.0 --mocap-object-low-poly
```
""",
    )


def main() -> int:
    freeze = _read(REPORT_ROOT / "pf_v2/contract_freeze.json")
    if freeze.get("classification") != "PF_V2_CONTRACT_FROZEN" or freeze.get("passed") is not True:
        raise RuntimeError("PF_V2_FINAL_REPORT_FREEZE_INVALID")
    source_170105 = _read(RUN_ROOT / "hocap_170105/training_contract.json")["source"]
    source_170650 = _read(RUN_ROOT / "hocap_170650/training_contract.json")["source"]
    if (
        source_170105["checkpoint_sha256"] != U10_SOURCE_SHA256
        or source_170650["checkpoint_sha256"] != HISTORICAL_170650_SHA256
    ):
        raise RuntimeError("PF_V2_FINAL_REPORT_SOURCE_HASH_DRIFT")
    contracts = {
        clip: _read(REPORT_ROOT / "training" / clip / "lineage_contract.json") for clip in CLIPS
    }
    symmetry = assert_symmetric_static_contracts(
        contracts["hocap_170105"], contracts["hocap_170650"]
    )
    _write_json(
        REPORT_ROOT / "comparison/symmetric_static_contract.json",
        {
            "schema_version": "Stage16Pfv2SymmetricStaticContractComparisonV1",
            **symmetry,
            "allowed_differences": ["clip", "source"],
            "contracts": {
                clip: {
                    "path": str(
                        (REPORT_ROOT / "training" / clip / "lineage_contract.json").resolve()
                    ),
                    "sha256": _sha256(REPORT_ROOT / "training" / clip / "lineage_contract.json"),
                }
                for clip in CLIPS
            },
        },
    )
    progressions = {clip: _read_progression(clip) for clip in CLIPS}
    if len(progressions["hocap_170105"]) != 1 or len(progressions["hocap_170650"]) != 10:
        raise RuntimeError("PF_V2_FINAL_REPORT_UPDATE_BUDGET_INVALID")
    _write_combined_progression(progressions)
    completed = {clip: _read(RUN_ROOT / clip / "complete.json") for clip in CLIPS}
    if (
        completed["hocap_170105"].get("actual_new_updates") != 1
        or completed["hocap_170105"].get("actual_new_samples") != 40960
        or completed["hocap_170650"].get("actual_new_updates") != 10
        or completed["hocap_170650"].get("actual_new_samples") != 409600
    ):
        raise RuntimeError("PF_V2_FINAL_REPORT_COMPLETION_INVALID")
    eval_170105 = _read(REPORT_ROOT / "training/hocap_170105/confirm20/U11/summary.json")
    eval_170650 = _read(REPORT_ROOT / "training/hocap_170650/best_eval20/summary.json")
    _require_eval20(eval_170105, clip="hocap_170105", pf_v1=0)
    _require_eval20(eval_170650, clip="hocap_170650", pf_v1=20)
    final_170105 = _compact_eval(eval_170105)
    final_170650 = _compact_eval(eval_170650)
    history = _history_rows(
        source_170105=source_170105,
        source_170650=source_170650,
        final_170105=final_170105,
        final_170650=final_170650,
    )
    _write_history(history)
    _write_json(
        REPORT_ROOT / "comparison/selected_eval20.json",
        {
            "hocap_170105_U11_confirm20": final_170105,
            "hocap_170650_U02_eval20": final_170650,
        },
    )
    _write_json(
        REPORT_ROOT / "pf_v2/support_transfer_contract.json",
        {
            "schema_version": "Stage16SupportTransferProxyV1",
            "exact_wrench_transfer": False,
            "signal": "table_object_contact_binary_proxy_no_exact_normal_wrench",
            "validity_rule": (
                "table_contact_sensor_validity_is_independent_of_hand_object_pair_force_validity; "
                "a recorded reset support sample is retained"
            ),
            "observed_historical_170650": "20/20 traces have reset table support before release",
            "requirements": [
                "persistent table-support absence after a prior observed support frame",
                "release no later than ActualLiftOnset",
                "persistent hand-object interaction at ActualLiftOnset",
            ],
        },
    )
    _write_replay_commands()
    final = {
        "schema_version": "Stage16Pfv2CausalLiftSymmetricPPOFinalV1",
        "classification": "PF_V2_CAUSAL_LIFT_VALIDATED__170650_CONTINUATION_INSTABILITY",
        "PF_V1_CHANGED": "NO",
        "PF_V2_ADDED": "YES",
        "support_sensor_authority_repaired": "YES",
        "historical_170650_support_source": "recorded reset table ContactSensor sample, not replay geometry",
        "contract_freeze": freeze,
        "symmetric_static_contract": symmetry,
        "lineages": {
            "hocap_170105": {
                "source": source_170105,
                "actual_new_updates": 1,
                "actual_new_samples": 40960,
                "early_stop": "U11 Eval10 PF_V2=10/10 triggered same-checkpoint Confirm20",
                "selected_confirm20": final_170105,
                "interpretation": (
                    "PF V2 causal physical lift and all DF dimensions pass 20/20; PF V1 remains "
                    "0/20 because its pre-reference-LIFT timing gate is intentionally preserved."
                ),
            },
            "hocap_170650": {
                "source": source_170650,
                "actual_new_updates": 10,
                "actual_new_samples": 409600,
                "best_observed_update": 2,
                "selected_eval20": final_170650,
                "continuation_outcome": (
                    "U2 is the first maximum-score checkpoint and preserves 20/20; U8 PF V2 is "
                    "0/10, then U10 recovers to Eval10 10/10. The continuation is non-monotonic."
                ),
                "historical_actor_overwritten": False,
            },
        },
        "safety_flags": {
            "uniform_training_rsi_preserved": True,
            "evaluation_frame0_deterministic": True,
            "per_object_reward_or_friction_tuning": False,
            "manual_grasp_frame": False,
            "phase_or_pre_lift_hard_reward_gate": False,
            "external_guidance": False,
            "rollout_object_or_wrist_root_state_writes": False,
            "new_branch_worktree_push_pr": False,
            "tracked_local_reports": False,
        },
        "next_action": "NEXT_DIAGNOSE_170650_CONTINUATION_INSTABILITY_WITHOUT_TUNING",
    }
    _write_json(REPORT_ROOT / "final_summary.json", final)
    _write_text(
        REPORT_ROOT / "handoff.md",
        f"""# Stage16 PF V2 Causal Lift + Symmetric PPO Handoff

`{final["classification"]}`.

The resolved support source is the independent recorded table ContactSensor:
every historical accepted 170650 trace contains a reset support sample before
release.  Replaying the support proxy only visualizes that scene; the replay
does not recompute contact evidence.

PF V2 is frozen, additive, and does not alter PF V1.  From U10, the 170105
lineage used one new update (U11, 40,960 samples); its same-checkpoint
Confirm20 gives PF V2/physical lift/causal lift/support transfer/sustained
coupling/DF pose/DF linear/DF angular V2 = 20/20.  PF V1 remains 0/20 solely
because the old pre-reference-LIFT timing gate remains unchanged.

The independent 170650 experimental continuation consumed its full ten-update
409,600-sample budget.  U2 is best observed and its Eval20 is 20/20 for PF V1,
PF V2, physical lift, causal lift, support transfer, and all DF dimensions.
The continuation is non-monotonic: U8 has PF V2=0/10 before U10 recovers to
Eval10 10/10. It is not a replacement for the frozen historical accepted
actor, which was never overwritten.

The only next action is `NEXT_DIAGNOSE_170650_CONTINUATION_INSTABILITY_WITHOUT_TUNING`.
""",
    )
    print(json.dumps(final, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
