#!/usr/bin/env python3
"""Merge bounded Stage 9 benchmark reports without changing frame results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from toporetarget.retarget.final_refinement import (
    CONTACT_RICH_SOLVER_PROFILE_ID,
    RefinementSolverProfile,
)
from toporetarget.retarget.solver_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    choose_uniform_maxiter,
    validate_benchmark_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--extra", type=Path, action="append", help="additional non-overlapping report"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    extras = [json.loads(path.read_text(encoding="utf-8")) for path in (args.extra or [])]
    if base.get("schema_version") != BENCHMARK_SCHEMA_VERSION or any(
        extra.get("schema_version") != BENCHMARK_SCHEMA_VERSION for extra in extras
    ):
        raise ValueError("benchmark reports use different schemas")
    grid = tuple(int(item) for item in base["budget_grid"])
    if any(tuple(int(item) for item in extra["budget_grid"]) != grid for extra in extras):
        raise ValueError("benchmark reports use different budget grids")
    records: list[dict[str, Any]] = list(base["records"])
    existing = {(str(item["case_id"]), int(item["budget"])) for item in records}
    for extra in extras:
        for item in extra["records"]:
            key = (str(item["case_id"]), int(item["budget"]))
            if key in existing:
                raise ValueError(f"duplicate benchmark record: {key}")
            records.append(item)
            existing.add(key)
    case_ids = tuple(sorted({str(item["case_id"]) for item in records}))
    payload = dict(base)
    payload["case_ids"] = list(case_ids)
    payload["fixed_benchmark_cases"] = list(base.get("fixed_benchmark_cases", [])) + [
        case for extra in extras for case in extra.get("fixed_benchmark_cases", [])
    ]
    payload["records"] = records
    profile = RefinementSolverProfile.load(CONTACT_RICH_SOLVER_PROFILE_ID, root=args.repo.resolve())
    payload["profile"] = profile.as_dict()
    for record in records:
        record["profile_id"] = profile.profile_id
        record["profile_hash"] = profile.profile_hash
    payload["selected_maxiter"] = choose_uniform_maxiter(records, grid, case_ids=case_ids)
    payload["status"] = "pass" if payload["selected_maxiter"] is not None else "blocked"
    payload["deterministic_repeat_requested"] = bool(
        base.get("deterministic_repeat_requested")
        or any(extra.get("deterministic_repeat_requested") for extra in extras)
    )
    validate_benchmark_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_maxiter": payload["selected_maxiter"],
                "records": len(records),
            },
            indent=2,
        )
    )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
