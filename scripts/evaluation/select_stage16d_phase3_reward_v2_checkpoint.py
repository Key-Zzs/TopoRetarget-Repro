#!/usr/bin/env python3
"""Select a Phase 3 Reward V2 checkpoint from frozen development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PHASE3_SELECTION_EXPECTS_JSON_OBJECT:{path}")
    return value


def _rate(suite: dict[str, Any], name: str) -> float:
    value = suite.get("aggregate", {}).get(name, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"PHASE3_SELECTION_RATE_MISSING:{name}")
    return float(value)


def _mean(suite: dict[str, Any], name: str) -> float:
    value = suite.get("aggregate", {}).get(name, {}).get("mean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"PHASE3_SELECTION_ERROR_MISSING:{name}")
    return float(value)


def _candidate(qualification_path: Path, suite_path: Path, evaluation_path: Path) -> dict[str, Any]:
    qualification = _read(qualification_path.resolve())
    suite = _read(suite_path.resolve())
    evaluation = _read(evaluation_path.resolve())
    frame_zero = evaluation.get("frame_zero")
    if (
        qualification.get("kind") != "development"
        or qualification.get("policy_reference_version") != 2
        or evaluation.get("reference_kinematics_version") != 2
        or evaluation.get("seed_set", {}).get("identifier") != "development_eval_seed_set_v1"
        or not isinstance(frame_zero, list)
        or len(frame_zero) != 20
        or evaluation.get("rsi") != []
    ):
        raise ValueError("PHASE3_SELECTION_REQUIRES_20_EPISODE_V2_DEVELOPMENT_EVIDENCE")
    if qualification.get("checkpoint_sha256") != evaluation.get(
        "checkpoint_sha256"
    ) or qualification.get("checkpoint") != evaluation.get("checkpoint"):
        raise ValueError("PHASE3_SELECTION_CHECKPOINT_PROVENANCE_MISMATCH")
    residuals = qualification.get("twist_residuals", {})
    delta_v = residuals.get("terminal_delta_v_mps", {}).get("per_episode_median")
    delta_omega = residuals.get("terminal_delta_omega_radps", {}).get("per_episode_median")
    samples = evaluation.get("reward_v2_samples")
    if not isinstance(delta_v, (int, float)) or not isinstance(delta_omega, (int, float)):
        raise ValueError("PHASE3_SELECTION_TERMINAL_RESIDUAL_MISSING")
    if not isinstance(samples, int) or samples < 1_048_576:
        raise ValueError("PHASE3_SELECTION_REWARD_V2_SAMPLE_COUNT_INVALID")
    metrics = {
        "SR_qualified": _rate(suite, "qualified_success"),
        "SR_physics": _rate(suite, "physics_success"),
        "terminal_stability_rate": float(qualification["terminal_stability_rate"]),
        "terminal_delta_omega_median_radps": float(delta_omega),
        "terminal_delta_v_median_mps": float(delta_v),
        "SR_kinematic": _rate(suite, "kinematic_success"),
        "E_t_mean_cm": _mean(suite, "E_t_mean_cm"),
        "E_r_mean_deg": _mean(suite, "E_r_mean_deg"),
        "E_ft_mean_cm": _mean(suite, "E_ft_mean_cm"),
        "E_j_mean_cm": _mean(suite, "E_j_mean_cm"),
    }
    selection_key = (
        -metrics["SR_qualified"],
        -metrics["SR_physics"],
        -metrics["terminal_stability_rate"],
        metrics["terminal_delta_omega_median_radps"],
        metrics["terminal_delta_v_median_mps"],
        -metrics["SR_kinematic"],
        metrics["E_t_mean_cm"],
        metrics["E_r_mean_deg"],
        metrics["E_ft_mean_cm"],
        metrics["E_j_mean_cm"],
        samples,
    )
    return {
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "reward_v2_samples": samples,
        "qualification": str(qualification_path.resolve()),
        "evaluation_suite": str(suite_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
        "metrics": metrics,
        "selection_key": list(selection_key),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        nargs=3,
        action="append",
        metavar=("QUALIFICATION", "SUITE", "EVALUATION"),
        required=True,
        help="One V2 development qualification, Evaluation Suite V2, and evaluation receipt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [_candidate(*triplet) for triplet in args.candidate]
    by_checkpoint = {str(row["checkpoint_sha256"]): row for row in candidates}
    if len(by_checkpoint) != len(candidates):
        raise ValueError("PHASE3_SELECTION_DUPLICATE_CHECKPOINT")
    ranked = sorted(candidates, key=lambda row: tuple(float(item) for item in row["selection_key"]))
    result = {
        "schema_version": "Stage16DPhase3RewardV2CheckpointSelectionV1",
        "status": "PHASE3_REWARD_V2_DEVELOPMENT_CHECKPOINT_SELECTED",
        "seed_set": "development_eval_seed_set_v1",
        "selection_rule": [
            "higher_SR_qualified",
            "higher_SR_physics",
            "higher_terminal_stability_rate",
            "lower_terminal_delta_omega_median",
            "lower_terminal_delta_v_median",
            "higher_SR_kinematic",
            "lower_E_t",
            "lower_E_r",
            "lower_E_ft",
            "lower_E_j",
            "earlier_reward_v2_checkpoint",
        ],
        "selected": ranked[0],
        "ranked": ranked,
        "formal_holdout_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
