#!/usr/bin/env python3
"""Select a Strict Per-Finger V4 checkpoint using frozen development evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_SELECTION_JSON_OBJECT_REQUIRED:{path}")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"STRICT_V4_SELECTION_METRIC_MISSING:{name}")
    return float(value)


def _rate(suite: dict[str, Any], name: str) -> float:
    return _number(suite.get("aggregate", {}).get(name, {}).get("rate"), name)


def _mean(suite: dict[str, Any], name: str) -> float:
    return _number(suite.get("aggregate", {}).get(name, {}).get("mean"), name)


def _cross_finger_mean(audit: dict[str, Any]) -> float:
    values = [
        row.get("cross_finger_group_compensation_fraction")
        for row in audit.get("per_finger", [])
        if isinstance(row, dict) and row.get("cross_finger_group_compensation_fraction") is not None
    ]
    if not values:
        return 0.0
    return float(
        sum(_number(value, "cross_finger_group_compensation_fraction") for value in values)
        / len(values)
    )


def _candidate(qualification_path: Path, suite_path: Path, audit_path: Path) -> dict[str, Any]:
    qualification = _read(qualification_path.resolve())
    suite = _read(suite_path.resolve())
    audit = _read(audit_path.resolve())
    evaluation = _read(Path(str(qualification["evaluation"])).resolve())
    frame_zero = evaluation.get("frame_zero")
    seed_set = str(evaluation.get("seed_set", {}).get("identifier", ""))
    if (
        qualification.get("kind") != "development"
        or qualification.get("status") != "STAGE16D_STRICT_V4_DEVELOPMENT_COMPLETE"
        or evaluation.get("reward_contract", {}).get("identifier")
        != "TopoRetargetReferenceTrackingReward26DV4"
        or "formal" in seed_set.lower()
        or evaluation.get("rsi") != []
        or not isinstance(frame_zero, list)
        or len(frame_zero) != 20
        or any(int(row["start_reference_index"]) != 0 for row in frame_zero)
        or suite.get("evaluation_kind") != "development"
        or audit.get("trace", {}).get("sha256") != qualification.get("trace_sha256")
        or audit.get("status") != "STRICT_V4_SOURCE_CONTACT_AUDIT_COMPLETE"
    ):
        raise ValueError("STRICT_V4_SELECTION_REQUIRES_DEVELOPMENT_ONLY_FORMAL20_SHAPE")
    samples = qualification.get("reward_v4_samples")
    if not isinstance(samples, int) or samples < 1_000_000:
        raise ValueError("STRICT_V4_SELECTION_SAMPLE_COUNT_INVALID")
    source = audit.get("aggregate", {})
    residuals = qualification.get("twist_residuals", {})
    metrics = {
        "SR_qualified": _rate(suite, "qualified_success"),
        "SR_physics": _rate(suite, "physics_success"),
        "persistent_source_tip_recall": _number(
            source.get("persistent_source_tip_recall"), "persistent_source_tip_recall"
        ),
        "source_tip_recall": _number(source.get("source_tip_recall"), "source_tip_recall"),
        "full_source_tip_coverage_rate": _number(
            source.get("full_source_tip_coverage_rate"), "full_source_tip_coverage_rate"
        ),
        "cross_finger_group_compensation_mean": _cross_finger_mean(audit),
        "no_tip_flight_event_count": _number(
            audit.get("no_tip_no_hand_flight_event_counts", {}).get("NO_TIP_CONTACT_FLIGHT"),
            "no_tip_flight_event_count",
        ),
        "no_hand_flight_event_count": _number(
            audit.get("no_tip_no_hand_flight_event_counts", {}).get(
                "NO_HAND_OBJECT_CONTACT_FLIGHT"
            ),
            "no_hand_flight_event_count",
        ),
        "terminal_stability_rate": _number(
            qualification.get("terminal_stability_rate"), "terminal_stability_rate"
        ),
        "terminal_delta_omega_median_radps": _number(
            residuals.get("terminal_delta_omega_radps", {}).get("per_episode_median"),
            "terminal_delta_omega_median_radps",
        ),
        "terminal_delta_v_median_mps": _number(
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
        -metrics["persistent_source_tip_recall"],
        -metrics["source_tip_recall"],
        -metrics["full_source_tip_coverage_rate"],
        metrics["cross_finger_group_compensation_mean"],
        metrics["no_tip_flight_event_count"],
        metrics["no_hand_flight_event_count"],
        -metrics["terminal_stability_rate"],
        metrics["terminal_delta_omega_median_radps"],
        metrics["terminal_delta_v_median_mps"],
        -metrics["SR_kinematic"],
        metrics["E_t_mean_cm"],
        metrics["E_r_mean_deg"],
        metrics["E_ft_mean_cm"],
        metrics["E_j_mean_cm"],
        float(samples),
    )
    return {
        "checkpoint": qualification["checkpoint"],
        "checkpoint_sha256": qualification["checkpoint_sha256"],
        "reward_v4_samples": samples,
        "seed_set": seed_set,
        "qualification": str(qualification_path.resolve()),
        "evaluation_suite": str(suite_path.resolve()),
        "source_audit": str(audit_path.resolve()),
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
        required=True,
        metavar=("QUALIFICATION", "SUITE", "SOURCE_AUDIT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = [_candidate(*candidate) for candidate in args.candidate]
    if len({str(row["checkpoint_sha256"]) for row in candidates}) != len(candidates):
        raise ValueError("STRICT_V4_SELECTION_DUPLICATE_CHECKPOINT")
    ranked = sorted(
        candidates, key=lambda row: tuple(float(value) for value in row["selection_key"])
    )
    result = {
        "schema_version": "Stage16DStrictPerFingerV4CheckpointSelectionV1",
        "status": "STAGE16D_STRICT_V4_DEVELOPMENT_CHECKPOINT_SELECTED",
        "formal_holdout_used_for_selection": False,
        "selection_rule": [
            "higher_SR_qualified",
            "higher_SR_physics",
            "higher_persistent_source_tip_recall",
            "higher_source_tip_recall",
            "higher_full_source_tip_coverage",
            "lower_cross_finger_group_compensation",
            "fewer_no_tip_flight_events",
            "fewer_no_hand_flight_events",
            "higher_terminal_stability",
            "lower_terminal_delta_omega",
            "lower_terminal_delta_v",
            "higher_SR_kinematic",
            "lower_Et",
            "lower_Er",
            "lower_Eft",
            "lower_Ej",
            "earlier_reward_v4_checkpoint",
        ],
        "selected": ranked[0],
        "ranked": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
