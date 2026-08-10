#!/usr/bin/env python3
"""Select a Stage 16-D PPO-26D checkpoint using frozen development-only ordering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_continuation import rank_development_checkpoints  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation is not a JSON object: {path}")
    return payload


def saturation_at_or_before(metrics: Path, *, samples: int) -> float:
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line]
    applicable = [row for row in rows if int(row["cumulative_samples"]) <= samples]
    if not applicable:
        return 0.0
    row = applicable[-1]
    return max(
        float(row["safety"]["before_update"]["sampled_action_saturation_fraction"]),
        float(row["safety"]["after_update"]["deterministic_action_saturation_fraction"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--training-metrics", type=Path, action="append", required=True)
    parser.add_argument(
        "--development-seed-set",
        default="development_eval_seed_set_v1",
        help="Exact frozen development-only seed-set identifier required in every evaluation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "formal" in args.development_seed_set.lower():
        raise ValueError("formal holdout evidence is forbidden for checkpoint selection")
    candidates = []
    for evaluation_path in args.evaluation:
        payload = read_json(evaluation_path.resolve())
        actual_seed_set = payload.get("seed_set", {}).get("identifier")
        if actual_seed_set != args.development_seed_set:
            raise ValueError(
                "checkpoint selection requires the requested development-only seed set; "
                f"expected {args.development_seed_set!r}, got {actual_seed_set!r}"
            )
        sample_count = int(payload["cumulative_training_samples"])
        matching_metrics = [
            path.resolve()
            for path in args.training_metrics
            if path.is_file()
            and any(
                int(json.loads(line)["cumulative_samples"]) <= sample_count
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            )
        ]
        saturation = max(
            (saturation_at_or_before(path, samples=sample_count) for path in matching_metrics),
            default=0.0,
        )
        candidates.append({**payload, "action_saturation": saturation})
    ranked = rank_development_checkpoints(candidates)
    result = {
        "schema_version": "Stage16DPPO26DCheckpointSelectionV1",
        "seed_set": args.development_seed_set,
        "selection_rule": [
            "frame_zero_reference_completion",
            "terminal_contact_rate",
            "terminal_stability_rate",
            "longest_continuous_contact",
            "lower_final_object_position_error",
            "lower_object_rotation_error",
            "higher_rsi_terminal_contact",
            "higher_total_reward",
            "lower_action_saturation",
            "earlier_checkpoint",
        ],
        "selected": ranked[0],
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected": result["selected"]["checkpoint"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
