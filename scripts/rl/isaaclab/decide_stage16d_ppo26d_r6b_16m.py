#!/usr/bin/env python3
"""Record the frozen development-only R6B 16M continuation decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_continuation import decide_r6b_post_16m  # noqa: E402


def _read_json(path: Path, *, development_seed_set: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    actual_seed_set = payload.get("seed_set", {}).get("identifier")
    if actual_seed_set != development_seed_set:
        raise ValueError(
            "R6B continuation requires the requested development-only seed set; "
            f"expected {development_seed_set!r}, got {actual_seed_set!r}"
        )
    return payload


def _safety(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ValueError("training metrics must contain at least one JSON row")
    finite = all(all(bool(value) for value in row["finite"].values()) for row in rows)
    object_writes = [int(row["reference"]["rsi"]["rollout_object_state_writes"]) for row in rows]
    wrist_writes = [int(row["reference"]["rsi"]["rollout_wrist_root_state_writes"]) for row in rows]
    sampled = [
        float(row["safety"]["before_update"]["sampled_action_saturation_fraction"]) for row in rows
    ]
    deterministic = [
        float(row["safety"]["after_update"]["deterministic_action_saturation_fraction"])
        for row in rows
    ]
    limit = float(rows[0]["safety"]["before_update"]["action_saturation_fraction_limit"])
    passed = (
        finite
        and max(object_writes) == 0
        and max(wrist_writes) == 0
        and max([*sampled, *deterministic]) <= limit
    )
    return {
        "all_finite": finite,
        "max_object_rollout_state_writes": max(object_writes),
        "max_wrist_root_state_writes": max(wrist_writes),
        "sampled_action_saturation_max": max(sampled),
        "deterministic_action_saturation_max": max(deterministic),
        "saturation_limit": limit,
        "no_safety_regression": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-m-evaluation", type=Path, required=True)
    parser.add_argument("--sixteen-m-evaluation", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument(
        "--development-seed-set",
        default="development_eval_seed_set_v1",
        help="Exact frozen development-only seed-set identifier required in both evaluations.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    args = parser.parse_args()
    if "formal" in args.development_seed_set.lower():
        raise ValueError("formal holdout evidence is forbidden for R6B continuation")
    four_m = _read_json(
        args.four_m_evaluation.resolve(), development_seed_set=args.development_seed_set
    )
    sixteen_m = _read_json(
        args.sixteen_m_evaluation.resolve(), development_seed_set=args.development_seed_set
    )
    if four_m.get("requested_clip") != sixteen_m.get("requested_clip"):
        raise ValueError("R6B evaluations must use the same clip")
    if int(sixteen_m["cumulative_training_samples"]) < 16_777_216:
        raise ValueError("R6B candidate has not reached the 16M cumulative threshold")
    rows = [
        json.loads(line)
        for line in args.training_metrics.resolve().read_text(encoding="utf-8").splitlines()
        if line
    ]
    safety = _safety(rows)
    decision = decide_r6b_post_16m(
        four_m_frame_zero=four_m["frame_zero_summary"],
        four_m_rsi=four_m["rsi_summary"],
        sixteen_m_frame_zero=sixteen_m["frame_zero_summary"],
        sixteen_m_rsi=sixteen_m["rsi_summary"],
        no_safety_regression=bool(safety["no_safety_regression"]),
    )
    result = {
        **decision,
        "clip": sixteen_m["requested_clip"],
        "four_m_evaluation": str(args.four_m_evaluation.resolve()),
        "sixteen_m_evaluation": str(args.sixteen_m_evaluation.resolve()),
        "development_seed_set": args.development_seed_set,
        "training_metrics": str(args.training_metrics.resolve()),
        "safety": safety,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.transitions.parent.mkdir(parents=True, exist_ok=True)
    with args.transitions.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "from": "R6B_CONTINUE",
                    "to": (
                        "R6B_32M"
                        if decision["continue_to_32m_authorized"]
                        else "CHECKPOINT_SELECTION"
                    ),
                    "reason": decision["decision"],
                    "decision": str(args.output.resolve()),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"decision": decision["decision"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
