#!/usr/bin/env python3
"""Materialize the P4 full-cycle evaluator contract and offline receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.evaluation.physical_functionality_full_cycle_v1 import (
    DF_INTERACTION_TIMING_FULL_CYCLE_V1,
    PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1,
    SUPPORT_TRANSFER_HAND_TO_SURFACE_PROXY_V1,
    PhysicalFunctionalityFullCycleV1Contract,
    evaluate_physical_functionality_full_cycle_v1,
)
from toporetarget.evaluation.stage16_pf_v2_causal_lift import (
    STAGE16_PHYSICAL_FUNCTIONALITY_V2,
    evaluate_stage16_physical_functionality_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".local/reports/raw_to_physical_hardening_v2/p4_full_cycle"),
    )
    parser.add_argument(
        "--hardening-manifest",
        type=Path,
        default=Path(
            ".local/reports/raw_to_physical_hardening_v2/p0_closeout/hardening_set_manifest.json"
        ),
    )
    parser.add_argument(
        "--prior-outcomes",
        type=Path,
        default=Path(
            ".local/reports/raw_to_physical_hardening_v2/p0_closeout/prior_outcome_inventory.csv"
        ),
    )
    parser.add_argument(
        "--pilot-final-result",
        type=Path,
        default=Path(
            ".local/reports/held_out_hocap_raw_to_physical_pilot_post_freeze_"
            "l0_unbounded_eval_retry1/clips/"
            "hocap_subject_9_20231027_125019__right__G16_3__ep00/"
            "final_qualification/final_result.json"
        ),
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _successful_trace() -> dict[str, Any]:
    count = 30
    object_pose = np.zeros((count, 7), dtype=np.float64)
    object_pose[:, 3] = 1.0
    object_pose[:, 2] = np.asarray(
        [0.0, 0.0, 0.0, 0.01, 0.03, 0.05, 0.07, 0.09] + [0.09] * (count - 8)
    )
    object_pose[9:14, 0] = np.linspace(0.02, 0.12, 5)
    object_pose[14:, 0] = 0.12
    wrist_pose = object_pose.copy()
    wrist_pose[:, 2] += 0.02
    wrist_pose[18:, 0] = object_pose[18:, 0] + np.linspace(0.02, 0.20, count - 18)

    contact = np.zeros(count, dtype=bool)
    contact[2:18] = True
    tips = np.zeros((count, 5), dtype=bool)
    tips[:, :2] = contact[:, None]
    hand = np.zeros((count, 21), dtype=bool)
    hand[:, 0] = contact
    table = np.zeros(count, dtype=bool)
    table[:4] = True
    region = np.zeros(count, dtype=bool)
    region[13:] = True
    destination_support = np.zeros(count, dtype=bool)
    destination_support[14:] = True
    return {
        "object_pose_wxyz": object_pose,
        "wrist_pose_wxyz": wrist_pose,
        "tip_pair_presence": tips,
        "hand_object_pair_presence": hand,
        "table_object_contact": table,
        "destination_region": region,
        "destination_support_contact": destination_support,
        "interaction_valid": np.ones(count, dtype=bool),
        "support_valid": np.ones(count, dtype=bool),
        "destination_region_valid": np.ones(count, dtype=bool),
        "destination_support_valid": np.ones(count, dtype=bool),
        "reference_lift_onset": 4,
        "reference_events": {
            "source_contact": 2,
            "persistent_contact": 2,
            "pickup": 4,
            "place": 12,
            "release": 17,
        },
        "causal_execution": True,
        "geometry_safe": True,
        "action_bounds_safe": True,
        "no_hidden_control": True,
    }


def _copy_trace(trace: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in trace.items():
        result[key] = value.copy() if isinstance(value, (np.ndarray, dict)) else value
    return result


def _phase_statuses(result: dict[str, object]) -> dict[str, str]:
    return {
        name: str(result[name]["status"])
        for name in (
            "PF_pick",
            "PF_transport",
            "PF_place",
            "PF_release",
            "PF_retreat",
            "PF_full_cycle",
        )
    }


def _synthetic_receipt() -> dict[str, object]:
    base = _successful_trace()
    scenarios: list[tuple[str, dict[str, Any], dict[str, str], str | None]] = []
    scenarios.append(
        (
            "complete_successful_pick_place_release_retreat",
            _copy_trace(base),
            {
                "PF_pick": "PASS",
                "PF_transport": "PASS",
                "PF_place": "PASS",
                "PF_release": "PASS",
                "PF_retreat": "PASS",
                "PF_full_cycle": "PASS",
            },
            None,
        )
    )
    transport_loss = _copy_trace(base)
    transport_loss["hand_object_pair_presence"][10:13] = False
    scenarios.append(
        (
            "pick_succeeds_but_transport_loses_object",
            transport_loss,
            {
                "PF_pick": "PASS",
                "PF_transport": "FAIL",
                "PF_place": "NOT_REACHED",
                "PF_release": "NOT_REACHED",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            },
            "hand_object_coupling_lost",
        )
    )
    place_failure = _copy_trace(base)
    place_failure["destination_support_contact"][:] = False
    scenarios.append(
        (
            "transport_succeeds_but_place_fails",
            place_failure,
            {
                "PF_pick": "PASS",
                "PF_transport": "PASS",
                "PF_place": "FAIL",
                "PF_release": "NOT_REACHED",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            },
            "destination_support_never_acquired",
        )
    )
    release_failure = _copy_trace(base)
    release_failure["hand_object_pair_presence"][18:, 0] = True
    release_failure["tip_pair_presence"][18:, :2] = True
    scenarios.append(
        (
            "place_succeeds_but_release_fails",
            release_failure,
            {
                "PF_pick": "PASS",
                "PF_transport": "PASS",
                "PF_place": "PASS",
                "PF_release": "FAIL",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            },
            "hand_object_contact_did_not_release",
        )
    )
    retreat_failure = _copy_trace(base)
    retreat_failure["object_pose_wxyz"][23:, 1] += 0.04
    scenarios.append(
        (
            "release_succeeds_but_retreat_disturbs_object",
            retreat_failure,
            {
                "PF_pick": "PASS",
                "PF_transport": "PASS",
                "PF_place": "PASS",
                "PF_release": "PASS",
                "PF_retreat": "FAIL",
                "PF_full_cycle": "FAIL",
            },
            "retreat_disturbed_object_translation",
        )
    )
    ballistic = _copy_trace(base)
    ballistic["tip_pair_presence"][7:] = False
    ballistic["hand_object_pair_presence"][7:] = False
    scenarios.append(
        (
            "ballistic_object_motion",
            ballistic,
            {
                "PF_pick": "FAIL",
                "PF_transport": "NOT_REACHED",
                "PF_place": "NOT_REACHED",
                "PF_release": "NOT_REACHED",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            },
            "sustained_hand_object_coupling",
        )
    )
    teleport = _copy_trace(base)
    teleport["object_pose_wxyz"][11:, 0] += 0.10
    teleport["wrist_pose_wxyz"][11:, 0] += 0.10
    scenarios.append(
        (
            "teleported_object",
            teleport,
            {
                "PF_pick": "PASS",
                "PF_transport": "FAIL",
                "PF_place": "NOT_REACHED",
                "PF_release": "NOT_REACHED",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            },
            "object_translation_discontinuity_teleport",
        )
    )
    support_never_transfers = _copy_trace(base)
    support_never_transfers["destination_support_contact"][:] = False
    scenarios.append(
        (
            "support_never_transfers",
            support_never_transfers,
            {
                "PF_pick": "PASS",
                "PF_transport": "PASS",
                "PF_place": "FAIL",
                "PF_release": "NOT_REACHED",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            },
            "destination_support_never_acquired",
        )
    )

    rows: list[dict[str, object]] = []
    for scenario_name, trace, expected, expected_reason in scenarios:
        result = evaluate_physical_functionality_full_cycle_v1(**trace)
        observed = _phase_statuses(result)
        all_reasons = [
            reason
            for phase in (
                "PF_pick",
                "PF_transport",
                "PF_place",
                "PF_release",
                "PF_retreat",
            )
            for reason in result[phase]["failure_reasons"]
        ]
        passed = observed == expected and (
            expected_reason is None or expected_reason in all_reasons
        )
        rows.append(
            {
                "scenario": scenario_name,
                "passed": passed,
                "expected_statuses": expected,
                "observed_statuses": observed,
                "expected_failure_reason": expected_reason,
                "observed_failure_reasons": all_reasons,
            }
        )

    diagnostic_trace = _copy_trace(base)
    diagnostic_trace["reference_lift_onset"] = 25
    diagnostic_trace["reference_events"] = {
        "source_contact": 25,
        "persistent_contact": 25,
        "pickup": 25,
        "place": 25,
        "release": 25,
    }
    diagnostic_result = evaluate_physical_functionality_full_cycle_v1(**diagnostic_trace)
    rows.append(
        {
            "scenario": "DF_interaction_timing_diagnostic_separation",
            "passed": bool(diagnostic_result["pf_full_cycle"])
            and diagnostic_result["DF_interaction_timing"]["included_in_pf_hard_gate"] is False,
            "expected_statuses": {"PF_full_cycle": "PASS"},
            "observed_statuses": {"PF_full_cycle": diagnostic_result["PF_full_cycle"]["status"]},
            "expected_failure_reason": None,
            "observed_failure_reasons": [],
        }
    )

    standalone = evaluate_stage16_physical_functionality_v2(
        object_pose_wxyz=base["object_pose_wxyz"],
        wrist_pose_wxyz=base["wrist_pose_wxyz"],
        tip_pair_presence=base["tip_pair_presence"],
        hand_object_pair_presence=base["hand_object_pair_presence"],
        table_object_contact=base["table_object_contact"],
        interaction_valid=base["interaction_valid"],
        support_valid=base["support_valid"],
        reference_lift_onset=base["reference_lift_onset"],
        causal_execution=True,
        geometry_safe=True,
        action_bounds_safe=True,
        no_hidden_control=True,
    )
    full = evaluate_physical_functionality_full_cycle_v1(**base)
    parity = full["PF_pick"]["diagnostics"]["detail"] == standalone
    rows.append(
        {
            "scenario": "PhysicalFunctionalityV2_exact_pick_parity",
            "passed": parity,
            "expected_statuses": {"PF_pick": "PASS"},
            "observed_statuses": {"PF_pick": full["PF_pick"]["status"]},
            "expected_failure_reason": None,
            "observed_failure_reasons": full["PF_pick"]["failure_reasons"],
        }
    )
    passed_count = sum(bool(row["passed"]) for row in rows)
    return {
        "schema_version": "PhysicalFunctionalityFullCycleV1SyntheticTestsV1",
        "status": "PASS" if passed_count == len(rows) else "FAIL",
        "passed": passed_count,
        "total": len(rows),
        "scenarios": rows,
    }


def _historical_rows(
    manifest_path: Path, prior_outcomes_path: Path, pilot_final_result_path: Path
) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with prior_outcomes_path.open(newline="", encoding="utf-8") as stream:
        prior_by_id = {row["episode_id"]: row for row in csv.DictReader(stream)}
    pilot = json.loads(pilot_final_result_path.read_text(encoding="utf-8"))
    selected_trace = Path(pilot["selected_trace"]["path"])
    selected = pilot["selected_episode"]
    rows: list[dict[str, object]] = []
    for episode in manifest["episodes"]:
        episode_id = episode["episode_id"]
        prior = prior_by_id[episode_id]
        if episode_id == pilot["clip_id"] and selected_trace.is_file():
            with np.load(selected_trace, allow_pickle=False) as trace:
                count = len(trace["object_pose"])
                result = evaluate_physical_functionality_full_cycle_v1(
                    object_pose_wxyz=trace["object_pose"],
                    wrist_pose_wxyz=trace["wrist_pose"],
                    tip_pair_presence=trace["tip_pair_presence"],
                    hand_object_pair_presence=trace["hand_object_pair_presence"],
                    table_object_contact=trace["table_object_contact"],
                    destination_region=np.zeros(count, dtype=bool),
                    destination_support_contact=np.zeros(count, dtype=bool),
                    interaction_valid=trace["fingertip_object_pair_force_valid"],
                    support_valid=trace["table_object_contact_valid"],
                    destination_region_valid=np.zeros(count, dtype=bool),
                    destination_support_valid=np.zeros(count, dtype=bool),
                    reference_lift_onset=int(selected["LIFT"]),
                    causal_execution=selected["causality"] == "True",
                    geometry_safe=selected["geometry"] == "True",
                    action_bounds_safe=bool(
                        np.isfinite(trace["action"]).all()
                        and (np.abs(trace["action"]) <= 1.0).all()
                    ),
                    no_hidden_control=selected["causality"] == "True",
                )
            statuses = _phase_statuses(result)
            parity = str(result["PF_pick"]["passed"]) == selected["PF_V2"]
            evidence = "SELECTED_PHYSICAL_TRACE_REEVALUATED"
            trace_path = str(selected_trace)
            trace_sha256 = _sha256(selected_trace)
        else:
            statuses = {
                "PF_pick": "NOT_IDENTIFIABLE",
                "PF_transport": "NOT_REACHED",
                "PF_place": "NOT_REACHED",
                "PF_release": "NOT_REACHED",
                "PF_retreat": "NOT_REACHED",
                "PF_full_cycle": "FAIL",
            }
            parity = "NOT_APPLICABLE_NO_PHYSICAL_TRACE"
            evidence = prior["final_evidence"]
            trace_path = ""
            trace_sha256 = ""
        rows.append(
            {
                "index": episode["index"],
                "episode_id": episode_id,
                "dataset_role": manifest["DATASET_ROLE"],
                "historical_evidence": evidence,
                **statuses,
                "PF_V2_parity": parity,
                "selected_trace": trace_path,
                "selected_trace_sha256": trace_sha256,
                "denominator_eligible": False,
            }
        )
    return rows


def _write_historical_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parser().parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2]
    full_cycle_source = (
        source_root / "src/toporetarget/evaluation/physical_functionality_full_cycle_v1.py"
    )
    pf_v2_source = source_root / "src/toporetarget/evaluation/stage16_pf_v2_causal_lift.py"
    contract = PhysicalFunctionalityFullCycleV1Contract()
    evaluator_contract = {
        "schema_version": PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1,
        "status": "FROZEN_FOR_P4_VALIDATION",
        "contract": contract.as_dict(),
        "phase_order": ["PF_pick", "PF_transport", "PF_place", "PF_release", "PF_retreat"],
        "composite_rule": "all_five_phases_PASS",
        "downstream_rule": "upstream_non_PASS_implies_NOT_REACHED",
        "pick_authority": {
            "schema_version": STAGE16_PHYSICAL_FUNCTIONALITY_V2,
            "immutable": True,
            "source": str(pf_v2_source),
            "source_sha256": _sha256(pf_v2_source),
            "reuse_rule": "verbatim_evaluator_result_no_threshold_or_semantic_override",
        },
        "placement_support": {
            "schema_version": SUPPORT_TRANSFER_HAND_TO_SURFACE_PROXY_V1,
            "exact_force_available": False,
            "proxy_labeled": True,
        },
        "interaction_timing": {
            "schema_version": DF_INTERACTION_TIMING_FULL_CYCLE_V1,
            "diagnostic_only": True,
            "included_in_pf_hard_gate": False,
            "fields": [
                "source_contact_timing",
                "persistent_contact_timing",
                "pickup_timing",
                "place_timing",
                "release_timing",
            ],
        },
        "implementation": {
            "source": str(full_cycle_source),
            "source_sha256": _sha256(full_cycle_source),
            "gpu_required": False,
        },
    }
    synthetic = _synthetic_receipt()
    historical = _historical_rows(
        args.hardening_manifest.resolve(),
        args.prior_outcomes.resolve(),
        args.pilot_final_result.resolve(),
    )
    _write_json(output / "evaluator_contract.json", evaluator_contract)
    _write_json(output / "synthetic_tests.json", synthetic)
    _write_historical_csv(output / "historical_reevaluation.csv", historical)

    selected_trace_rows = sum(bool(row["selected_trace"]) for row in historical)
    all_synthetic_pass = synthetic["status"] == "PASS"
    decision = {
        "schema_version": "RawToPhysicalHardeningV2P4DecisionV1",
        "status": "PASS" if all_synthetic_pass else "FAIL",
        "decision": (
            "FULL_CYCLE_PF_V1_VALIDATED" if all_synthetic_pass else "FULL_CYCLE_PF_V1_PARTIAL"
        ),
        "selected_production_branch": PHYSICAL_FUNCTIONALITY_FULL_CYCLE_V1,
        "conservative_fallback": (
            "report_PhysicalFunctionalityV2_pick_only_and_mark_later_phases_NOT_IDENTIFIABLE"
        ),
        "P5_evaluator_authorized": bool(all_synthetic_pass),
        "synthetic_validation": {
            "passed": synthetic["passed"],
            "total": synthetic["total"],
            "receipt": "synthetic_tests.json",
        },
        "historical_reevaluation": {
            "episodes": len(historical),
            "selected_physical_traces_reevaluated": selected_trace_rows,
            "current_full_cycle_passes": sum(row["PF_full_cycle"] == "PASS" for row in historical),
            "scope": "PIPELINE_HARDENING_SET_V1_not_held_out_denominator",
            "limitation": (
                "four episodes have no physical trace; the sole selected physical trace fails "
                "PF_pick, so all downstream phases are correctly NOT_REACHED"
            ),
            "receipt": "historical_reevaluation.csv",
        },
        "PF_V2_immutable": True,
        "DF_interaction_timing_diagnostic_only": True,
        "support_force_limitation": (
            "destination acquisition uses SupportTransferHandToSurfaceProxyV1; exact support "
            "force is not claimed"
        ),
        "gpu_used": False,
    }
    _write_json(output / "final_decision.json", decision)
    return 0 if all_synthetic_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
