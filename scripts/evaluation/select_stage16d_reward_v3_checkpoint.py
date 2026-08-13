#!/usr/bin/env python3
"""Select a Reward V3 checkpoint using the frozen 4M contact-first ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V3_SELECTION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _rate(suite: dict[str, Any], name: str) -> float:
    value = suite.get("aggregate", {}).get(name, {}).get("rate")
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_SELECTION_SUITE_RATE_MISSING:{name}")
    return float(value)


def _mean(suite: dict[str, Any], name: str) -> float:
    value = suite.get("aggregate", {}).get(name, {}).get("mean")
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_SELECTION_SUITE_MEAN_MISSING:{name}")
    return float(value)


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"V3_SELECTION_METRIC_MISSING:{name}")
    return float(value)


def _candidate(qualification_path: Path, suite_path: Path, contact_path: Path) -> dict[str, Any]:
    qualification = _read(qualification_path.resolve())
    suite = _read(suite_path.resolve())
    contact = _read(contact_path.resolve())
    evaluation = _read(Path(str(qualification["evaluation"])).resolve())
    frame_zero = evaluation.get("frame_zero")
    seed_set = str(evaluation.get("seed_set", {}).get("identifier", ""))
    if (
        qualification.get("kind") != "development"
        or qualification.get("status") != "STAGE16D_REWARD_V3_DEVELOPMENT_COMPLETE"
        or evaluation.get("reward_contract", {}).get("identifier")
        != "TopoRetargetReferenceTrackingReward26DV3"
        or "formal" in seed_set.lower()
        or evaluation.get("rsi") != []
        or not isinstance(frame_zero, list)
        or len(frame_zero) != 20
        or any(int(row["start_reference_index"]) != 0 for row in frame_zero)
        or suite.get("evaluation_kind") != "development"
        or contact.get("trace_sha256") != qualification.get("trace_sha256")
    ):
        raise ValueError("V3_SELECTION_REQUIRES_20_EPISODE_DEVELOPMENT_ONLY_EVIDENCE")
    samples = qualification.get("reward_v3_samples")
    if not isinstance(samples, int) or samples < 1_000_000:
        raise ValueError("V3_SELECTION_REWARD_V3_SAMPLE_COUNT_INVALID")
    aggregate = contact.get("aggregate", {})
    residuals = qualification.get("twist_residuals", {})
    metrics = {
        "SR_qualified": _rate(suite, "qualified_success"),
        "SR_physics": _rate(suite, "physics_success"),
        "expected_contact_recall": _finite(
            aggregate.get("expected_contact_recall"), "expected_contact_recall"
        ),
        "persistent_contact_recall": _finite(
            aggregate.get("persistent_contact_recall"), "persistent_contact_recall"
        ),
        "terminal_contact_rate": _finite(
            qualification.get("terminal_contact_rate"), "terminal_contact_rate"
        ),
        "terminal_stability_rate": _finite(
            qualification.get("terminal_stability_rate"), "terminal_stability_rate"
        ),
        "longest_contact_loss_gap": _finite(
            aggregate.get("longest_contact_loss_gap"), "longest_contact_loss_gap"
        ),
        "terminal_delta_omega_median_radps": _finite(
            residuals.get("terminal_delta_omega_radps", {}).get("per_episode_median"),
            "terminal_delta_omega_median_radps",
        ),
        "terminal_delta_v_median_mps": _finite(
            residuals.get("terminal_delta_v_mps", {}).get("per_episode_median"),
            "terminal_delta_v_median_mps",
        ),
        "SR_kinematic": _rate(suite, "kinematic_success"),
        "E_t_mean_cm": _mean(suite, "E_t_mean_cm"),
        "E_r_mean_deg": _mean(suite, "E_r_mean_deg"),
        "E_ft_mean_cm": _mean(suite, "E_ft_mean_cm"),
        "E_j_mean_cm": _mean(suite, "E_j_mean_cm"),
    }
    key = (
        -metrics["SR_qualified"],
        -metrics["SR_physics"],
        -metrics["expected_contact_recall"],
        -metrics["persistent_contact_recall"],
        -metrics["terminal_contact_rate"],
        -metrics["terminal_stability_rate"],
        metrics["longest_contact_loss_gap"],
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
        "checkpoint": qualification["checkpoint"],
        "checkpoint_sha256": qualification["checkpoint_sha256"],
        "reward_v3_samples": samples,
        "seed_set": seed_set,
        "qualification": str(qualification_path.resolve()),
        "evaluation_suite": str(suite_path.resolve()),
        "contact_summary": str(contact_path.resolve()),
        "metrics": metrics,
        "selection_key": list(key),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        nargs=3,
        action="append",
        metavar=("QUALIFICATION", "SUITE", "CONTACT"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [_candidate(*candidate) for candidate in args.candidate]
    if len({str(row["checkpoint_sha256"]) for row in candidates}) != len(candidates):
        raise ValueError("V3_SELECTION_DUPLICATE_CHECKPOINT")
    selected = sorted(candidates, key=lambda row: tuple(float(v) for v in row["selection_key"]))
    result = {
        "schema_version": "Stage16DRewardV3CheckpointSelectionV1",
        "status": "STAGE16D_REWARD_V3_DEVELOPMENT_CHECKPOINT_SELECTED",
        "formal_holdout_used_for_selection": False,
        "selection_rule": [
            "higher_SR_qualified",
            "higher_SR_physics",
            "higher_expected_contact_recall",
            "higher_persistent_contact_recall",
            "higher_terminal_contact",
            "higher_terminal_stability",
            "shorter_longest_contact_loss_gap",
            "lower_terminal_delta_omega",
            "lower_terminal_delta_v",
            "higher_SR_kinematic",
            "lower_Et",
            "lower_Er",
            "lower_Eft",
            "lower_Ej",
            "earlier_reward_v3_checkpoint",
        ],
        "selected": selected[0],
        "ranked": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
