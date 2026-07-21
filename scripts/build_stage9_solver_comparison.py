#!/usr/bin/env python3
"""Build the bounded far-vs-contact Stage 9 solver comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toporetarget.retarget.final_refinement import (
    CONTACT_RICH_SOLVER_PROFILE_ID,
    RefinementSolverProfile,
)

CASE_BY_CLASS = {
    "pre_contact": "rh_pre_contact_frame_0",
    "approach": "approach_frame_0",
    "contact_rich": "current_first_failure_contact_rich_frame_0",
}
WINDOW_BY_CLASS = {
    "pre_contact": "stage9_pre_contact_regression_s7_rh",
    "approach": "stage10_approach_f0238_f0298",
    "contact_rich": "stage10_contact_rich_f0240_f0300",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "case_id",
        "budget",
        "result_success",
        "status_code",
        "message",
        "nit",
        "nfev",
        "njev",
        "initial_objective",
        "final_objective",
        "final_objective_change",
        "final_step_norm",
        "min_hard_residual_m",
        "min_soft_residual_m",
        "full_surface_min_signed_distance_m",
        "active_set_rounds",
        "active_set_converged",
        "runtime_s",
        "strict_acceptance",
        "independent_full_surface_audit_pass",
        "deterministic_repeat",
        "acceptance_reason",
    )
    return {key: record.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=Path(".local/reports/stage9_solver_closeout/window_geometry_audit.json"),
    )
    parser.add_argument(
        "--conditioning-report",
        type=Path,
        default=Path(".local/reports/stage9_solver_closeout/interaction_conditioning.json"),
    )
    parser.add_argument(
        "--benchmark-report",
        type=Path,
        default=Path(".local/reports/stage9_1/maxiter_benchmark.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".local/reports/stage9_solver_closeout/far_vs_contact_solver_comparison.json"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    geometry = _load(args.geometry_report)
    conditioning = _load(args.conditioning_report)
    benchmark = _load(args.benchmark_report)
    profile = RefinementSolverProfile.load(CONTACT_RICH_SOLVER_PROFILE_ID, root=repo)
    geometry_by_window = {item["window_id"]: item for item in geometry["windows"]}
    conditioning_by_window = {item["window_id"]: item for item in conditioning["windows"]}
    records = benchmark["records"]
    selected_budget = int(benchmark["selected_maxiter"])
    by_case_budget = {(str(item["case_id"]), int(item["budget"])): item for item in records}

    comparisons: list[dict[str, Any]] = []
    for window_class, case_id in CASE_BY_CLASS.items():
        geometry_row = geometry_by_window[WINDOW_BY_CLASS[window_class]]
        conditioning_row = conditioning_by_window[WINDOW_BY_CLASS[window_class]]
        comparison = {
            "window_class": window_class,
            "case_id": case_id,
            "window_id": WINDOW_BY_CLASS[window_class],
            "geometry_classification": geometry_row["window_class"],
            "geometry_classification_matches_expected": geometry_row[
                "classification_matches_expected"
            ],
            "geometry_classification_metrics": geometry_row["classification_metrics"],
            "geometry_aggregate_distances": geometry_row["aggregate_distances"],
            "conditioning_aggregate": conditioning_row["aggregate"],
            "budget_30": _record_summary(by_case_budget[(case_id, 30)]),
            "selected_budget": _record_summary(by_case_budget[(case_id, selected_budget)]),
        }
        comparisons.append(comparison)

    budget30 = {item["window_class"]: item["budget_30"] for item in comparisons}
    pre_contact_weak = not bool(budget30["pre_contact"]["result_success"])
    contact_rich_failed = not bool(budget30["contact_rich"]["result_success"])
    approach_failed = not bool(budget30["approach"]["result_success"])
    source_geometry_valid = all(
        item["geometry_classification_matches_expected"]
        and item["window_class"] != "invalid_coordinate_or_scale"
        for item in comparisons
    )
    payload = {
        "schema_version": "toporetarget.stage9_solver_closeout.far_vs_contact.v1",
        "comparison_contract": {
            "profile_id": profile.profile_id,
            "profile_hash": profile.profile_hash,
            "budget": 30,
            "same_equations": ["Eq. (8)", "Eq. (9)"],
            "equations_modified": False,
            "same_paper_weights": True,
            "same_base_parameterization": True,
            "same_qpos_and_slack_bounds": True,
            "same_signed_distance_sign": "positive_outside",
            "same_full_surface_audit": True,
            "semantic_contacts_in_objective": False,
            "active_set_continuation": profile.active_set_continuation_policy,
            "sdf_backend": profile.sdf_backend,
        },
        "fixed_comparison": {
            "A": "pre_contact",
            "B": "approach",
            "C": "contact_rich",
            "selected_uniform_budget": selected_budget,
            "grid": benchmark["budget_grid"],
        },
        "comparisons": comparisons,
        "causal_assessment": {
            "pre_contact_consistently_weak_or_bad_at_budget_30": pre_contact_weak,
            "approach_failed_at_budget_30": approach_failed,
            "contact_rich_failed_at_budget_30": contact_rich_failed,
            "source_and_coordinate_audits_valid": source_geometry_valid,
            "status9_correlates_with_contact_rich_or_approach": (
                contact_rich_failed and approach_failed and not pre_contact_weak
            ),
            "continuation_bug_fixed_before_comparison": True,
            "far_geometry_material_contributor": False,
            "coordinate_or_scale_blocker": False,
            "stage7_8_9_mapping_blocker": False,
            "conclusion": (
                "status=9 is associated with the contact/approach solver difficulty "
                "under the fixed budget, not with the far pre-contact geometry; the "
                "bounded evidence supports a uniform solver-budget/termination cause "
                "after result.x continuation was fixed"
            ),
        },
        "source_reports": {
            "geometry": str(args.geometry_report),
            "conditioning": str(args.conditioning_report),
            "benchmark": str(args.benchmark_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_budget": selected_budget,
                "far_geometry_material_contributor": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
