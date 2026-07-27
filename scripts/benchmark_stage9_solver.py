#!/usr/bin/env python3
"""Run the bounded Stage 9.1 maxiter benchmark on explicit fixed frames."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.bones import load_bone_profile
from toporetarget.retarget.final_refinement import (
    CONTACT_RICH_SOLVER_PROFILE_ID,
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    build_final_trajectory,
    load_robot_surface_samples,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_graph,
)
from toporetarget.retarget.solver_benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    DEFAULT_BENCHMARK_GRID,
    choose_uniform_maxiter,
    validate_benchmark_report,
)
from toporetarget.robots.artimano import load_artimano_model


def _default_cases(repo: Path) -> list[dict[str, Any]]:
    airplane = repo / (
        ".local/runs/stage10/s1__airplane_lift__right__artimano_rh__f000240_f000300/artifacts"
    )
    contact_cases = [
        ("current_first_failure_contact_rich_frame_0", airplane, 0),
        # The Stage 10 selection report identifies global frame 270 (local 30)
        # as the maximum semantic-contact frame in [240, 300).
        ("maximum_semantic_contact_frame_30", airplane, 30),
        ("minimum_full_surface_sdf_frame_29", airplane, 29),
        ("maximum_e_im_frame_59", airplane, 59),
    ]
    approach = repo / (
        ".local/runs/stage10/s1__airplane_lift__right__artimano_rh__f000238_f000298/artifacts"
    )
    contact_cases.append(("approach_frame_0", approach, 0))
    # Keep both hands and an existing pre-contact successful frame in the set.
    pre_contact_cases = [
        (
            "rh_pre_contact_frame_0",
            repo / ".local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr",
            repo
            / ".local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr",
            repo / ".local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr",
            "artimano_rh",
        ),
        (
            "lh_pre_contact_frame_0",
            repo / ".local/runs/stage10/"
            "s7__cubemedium_inspect_1__left__artimano_lh__f000513_f000573/artifacts/canonical.zarr",
            repo / ".local/cache/retarget/warm_start/s7_cubemedium_inspect_1_left_artimano_lh.zarr",
            repo / ".local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_left.zarr",
            "artimano_lh",
        ),
    ]
    result: list[dict[str, Any]] = []
    for item in contact_cases:
        case_id, root, frame = item
        result.append(
            {
                "case_id": case_id,
                "canonical": str(root / "canonical.zarr"),
                "warm_start": str(root / "warm_start.zarr"),
                "graph": str(root / "interaction_graph.zarr"),
                "robot": "artimano_rh",
                "collision_samples": str(
                    repo / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz"
                ),
                "frame": frame,
            }
        )
    for case_id, canonical, warm_start, graph, robot in pre_contact_cases:
        result.append(
            {
                "case_id": case_id,
                "canonical": str(canonical),
                "warm_start": str(warm_start),
                "graph": str(graph),
                "robot": robot,
                "collision_samples": str(
                    repo / f".local/cache/geometry/robot_surface/{robot}_neutral.npz"
                ),
                "frame": 0,
            }
        )
    return result


def _load_case(case: dict[str, Any], repo: Path) -> dict[str, Any]:
    canonical = Path(case["canonical"])
    warm_path = Path(case["warm_start"])
    graph_path = Path(case["graph"])
    robot_name = str(case["robot"])
    return {
        "sequence": load_hoi_sequence(canonical),
        "warm": load_warm_start(warm_path),
        "graph": load_interaction_graph(graph_path),
        "robot": load_artimano_model(
            "rh" if robot_name.endswith("rh") else "lh",
            asset_root=repo / "third_party/robot_hands/artimano",
        ),
        "surface": load_robot_surface_samples(Path(case["collision_samples"])),
        "canonical_hash": artifact_hash(canonical),
        "warm_hash": artifact_hash(warm_path),
        "graph_hash": interaction_artifact_hash(graph_path),
    }


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _arrays_equal(left: Any, right: Any) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype.kind in "biufc" and right_array.dtype.kind in "biufc":
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _run_one(
    case: dict[str, Any], loaded: dict[str, Any], profile: RefinementSolverProfile, budget: int
) -> tuple[dict[str, Any], Any]:
    effective = replace(
        profile,
        maxiter=budget,
        maxiter_provenance={
            **profile.maxiter_provenance,
            "effective_budget": budget,
            "benchmark_profile_hash": profile.profile_hash,
        },
    )
    frame = int(case["frame"])
    started = time.perf_counter()
    trajectory, _ = build_final_trajectory(
        loaded["sequence"],
        loaded["warm"],
        loaded["graph"],
        loaded["robot"],
        loaded["surface"],
        load_frame_profile("canonical_keypoint_wrist_v1"),
        load_bone_profile("mediapipe21_full_finger_chain_v1"),
        RefinementCoordinateProfile.load("local_seed_delta_v1"),
        CollisionQueryProfile.load("adaptive_active_set_v1"),
        effective,
        start_frame=frame,
        end_frame=frame + 1,
        warm_artifact_hash=loaded["warm_hash"],
        graph_artifact_hash=loaded["graph_hash"],
        continue_on_failure=True,
    )
    elapsed = time.perf_counter() - started
    arrays = trajectory.arrays
    q0, q1 = int(arrays["query_offsets"][0]), int(arrays["query_offsets"][1])
    hard = arrays["hard_residual_concat"][q0:q1]
    soft = arrays["soft_residual_concat"][q0:q1]
    record = {
        "case_id": str(case["case_id"]),
        "frame": frame,
        "global_frame": int(arrays["frame_indices"][0]),
        "robot": str(case["robot"]),
        "budget": budget,
        "profile_id": profile.profile_id,
        "profile_hash": profile.profile_hash,
        "result_success": bool(arrays["optimizer_converged"][0]),
        "status_code": int(arrays["optimizer_status_code"][0]),
        "message": _decode(arrays["optimizer_message"][0]),
        "nit": int(arrays["optimizer_iterations"][0]),
        "nfev": int(arrays["optimizer_function_evaluations"][0]),
        "njev": int(arrays["optimizer_jacobian_evaluations"][0]),
        "initial_objective": float(arrays["initial_objective"][0]),
        "final_objective": float(arrays["final_objective"][0]),
        "final_objective_change": float(arrays["final_objective_change"][0]),
        "final_step_norm": float(arrays["final_step_norm"][0]),
        "min_hard_residual_m": float(np.min(hard, initial=np.inf)),
        "min_soft_residual_m": float(np.min(soft, initial=np.inf)),
        "full_surface_min_signed_distance_m": float(arrays["min_full_signed_distance"][0]),
        "active_set_rounds": int(arrays["active_set_rounds"][0]),
        "active_set_converged": bool(arrays["active_set_converged"][0]),
        "runtime_s": elapsed,
        "strict_acceptance": bool(arrays["accepted"][0]),
        "independent_full_surface_audit_pass": bool(
            arrays["full_surface_hard_audit_pass"][0] and arrays["full_surface_soft_audit_pass"][0]
        ),
        "qpos_bounds_pass": bool(arrays["qpos_bounds_pass"][0]),
        "slack_bounds_pass": bool(arrays["slack_bounds_pass"][0]),
        "active_constraints_feasible": bool(arrays["active_constraints_feasible"][0]),
        "all_values_finite": bool(arrays["all_values_finite"][0]),
        "deterministic_repeat": False,
        "acceptance_reason": _decode(arrays["acceptance_reason"][0]),
        "source_inputs": {
            "canonical": str(case["canonical"]),
            "warm_start": str(case["warm_start"]),
            "graph": str(case["graph"]),
            "collision_samples": str(case["collision_samples"]),
            "canonical_hash": loaded["canonical_hash"],
            "warm_start_hash": loaded["warm_hash"],
            "graph_hash": loaded["graph_hash"],
        },
    }
    return record, trajectory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path(".local/reports/stage9_1/maxiter_benchmark.json")
    )
    parser.add_argument("--budgets", type=int, nargs="+", help="override benchmark grid")
    parser.add_argument(
        "--case-id",
        action="append",
        help="restrict this invocation to one or more fixed benchmark case IDs",
    )
    parser.add_argument("--no-deterministic-repeat", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    profile = RefinementSolverProfile.load(CONTACT_RICH_SOLVER_PROFILE_ID, root=repo)
    grid = tuple(args.budgets or profile.benchmark_grid or DEFAULT_BENCHMARK_GRID)
    if args.case_manifest is None:
        cases = _default_cases(repo)
    else:
        cases = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark case manifest must be a non-empty list")
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if str(case.get("case_id")) in requested]
        if not cases:
            raise ValueError(f"no fixed benchmark case matches --case-id={sorted(requested)}")
    loaded_by_key: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for budget in grid:
        for case in cases:
            key = "|".join(
                [
                    str(case["canonical"]),
                    str(case["warm_start"]),
                    str(case["graph"]),
                    str(case["frame"]),
                ]
            )
            if key not in loaded_by_key:
                loaded_by_key[key] = _load_case(case, repo)
            record, trajectory = _run_one(case, loaded_by_key[key], profile, budget)
            if not args.no_deterministic_repeat:
                repeat, repeat_trajectory = _run_one(case, loaded_by_key[key], profile, budget)
                record["deterministic_repeat"] = bool(
                    all(
                        _arrays_equal(trajectory.arrays[name], repeat_trajectory.arrays[name])
                        for name in trajectory.arrays
                    )
                    and record["status_code"] == repeat["status_code"]
                    and record["message"] == repeat["message"]
                )
            records.append(record)
            print(
                f"budget={budget} case={case['case_id']} status={record['status_code']} "
                f"success={record['result_success']} accepted={record['strict_acceptance']} "
                f"runtime_s={record['runtime_s']:.3f}",
                flush=True,
            )
    case_ids = tuple(str(case["case_id"]) for case in cases)
    selected = choose_uniform_maxiter(records, grid, case_ids=case_ids)
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "pass" if selected is not None else "blocked",
        "profile": profile.as_dict(),
        "budget_grid": list(grid),
        "fixed_benchmark_cases": cases,
        "case_ids": list(case_ids),
        "selected_maxiter": selected,
        "selection_rule": "minimum_single_budget_passing_all_fixed_benchmark_frames",
        "deterministic_repeat_requested": not args.no_deterministic_repeat,
        "records": records,
    }
    validate_benchmark_report(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"status": payload["status"], "selected_maxiter": selected}, indent=2))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
