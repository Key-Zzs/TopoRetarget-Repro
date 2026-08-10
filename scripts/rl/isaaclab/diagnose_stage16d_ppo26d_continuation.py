#!/usr/bin/env python3
"""Apply frozen R6A / PPO-update decisions to evidence-backed PPO-26D artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_continuation import (  # noqa: E402
    classify_ppo_update_bottleneck,
    classify_r6a,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("training metrics must contain JSON-object rows")
    return rows


def _late_or_contact_object_reward(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as trace:
        reward = np.asarray(trace["reward_object"], dtype=np.float64)
        contact = np.asarray(trace["contact_pair_presence"], dtype=bool).any(axis=-1)
    if reward.ndim != 1 or reward.size == 0 or not np.isfinite(reward).all():
        raise ValueError("trace has no finite scalar reward_object series")
    late = np.arange(reward.size) >= int(np.floor(0.75 * reward.size))
    selected = contact | late
    return {
        "value": float(reward[selected].mean()),
        "selection": "contact_frames_union_final_quarter",
        "contact_frames": int(contact.sum()),
        "final_quarter_frames": int(late.sum()),
        "selected_frames": int(selected.sum()),
    }


def _safety_and_exploit(rows: list[dict[str, Any]]) -> tuple[bool, bool, dict[str, object]]:
    finite = all(all(bool(value) for value in row["finite"].values()) for row in rows)
    object_writes = [int(row["reference"]["rsi"]["rollout_object_state_writes"]) for row in rows]
    wrist_writes = [int(row["reference"]["rsi"]["rollout_wrist_root_state_writes"]) for row in rows]
    sampled_saturation = [
        float(row["safety"]["before_update"]["sampled_action_saturation_fraction"]) for row in rows
    ]
    deterministic_saturation = [
        float(row["safety"]["after_update"]["deterministic_action_saturation_fraction"])
        for row in rows
    ]
    saturation_limit = float(rows[0]["safety"]["before_update"]["action_saturation_fraction_limit"])
    no_safety_failure = (
        finite
        and max(object_writes) == 0
        and max(wrist_writes) == 0
        and max(sampled_saturation) <= saturation_limit
        and max(deterministic_saturation) <= saturation_limit
    )
    no_reward_exploit = (
        max(sampled_saturation) <= saturation_limit
        and max(deterministic_saturation) <= saturation_limit
        and finite
    )
    return (
        no_safety_failure,
        no_reward_exploit,
        {
            "all_finite": finite,
            "max_object_rollout_state_writes": max(object_writes),
            "max_wrist_root_state_writes": max(wrist_writes),
            "sampled_action_saturation_max": max(sampled_saturation),
            "deterministic_action_saturation_max": max(deterministic_saturation),
            "saturation_limit": saturation_limit,
            "no_new_safety_failure": no_safety_failure,
            "no_reward_exploit_detected": no_reward_exploit,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-trace", type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--extension-already-used", action="store_true")
    args = parser.parse_args()
    baseline = _read_json(args.baseline_evaluation.resolve())
    candidate = _read_json(args.candidate_evaluation.resolve())
    if baseline["requested_clip"] != candidate["requested_clip"]:
        raise ValueError("R6A diagnosis cannot compare different clips")
    baseline_reward = _late_or_contact_object_reward(args.baseline_trace.resolve())
    candidate_reward = _late_or_contact_object_reward(args.candidate_trace.resolve())
    rows = _read_metrics(args.training_metrics.resolve())
    no_safety_failure, no_reward_exploit, safety = _safety_and_exploit(rows)
    decision = classify_r6a(
        baseline_frame_zero=baseline["frame_zero_summary"],
        baseline_rsi=baseline["rsi_summary"],
        four_m_frame_zero=candidate["frame_zero_summary"],
        four_m_rsi=candidate["rsi_summary"],
        late_object_reward_baseline=float(baseline_reward["value"]),
        late_object_reward_four_m=float(candidate_reward["value"]),
        no_new_safety_failure=no_safety_failure,
        no_reward_exploit=no_reward_exploit,
        extension_already_used=args.extension_already_used,
    )
    ppo_update = classify_ppo_update_bottleneck(rows)
    result = {
        "schema_version": "Stage16DPPO26DContinuationDiagnosisV1",
        "clip": candidate["requested_clip"],
        "baseline": {
            "evaluation": str(args.baseline_evaluation.resolve()),
            "frame_zero_summary": baseline["frame_zero_summary"],
            "rsi_summary": baseline["rsi_summary"],
            "late_or_contact_object_reward": baseline_reward,
        },
        "candidate": {
            "evaluation": str(args.candidate_evaluation.resolve()),
            "frame_zero_summary": candidate["frame_zero_summary"],
            "rsi_summary": candidate["rsi_summary"],
            "late_or_contact_object_reward": candidate_reward,
        },
        "r6a_decision": decision.as_dict(),
        "ppo_update_diagnosis": ppo_update,
        "safety_and_reward_exploit": safety,
        "training_iterations": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.transitions.parent.mkdir(parents=True, exist_ok=True)
    with args.transitions.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "from": "R6A_RESUME_4M",
                    "to": "LEARNING_DIAGNOSIS",
                    "reason": decision.decision.value,
                    "diagnosis": str(args.output.resolve()),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"decision": decision.decision.value, "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
