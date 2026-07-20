"""Validate Stage 9 objective and constraint Jacobians on one real frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toporetarget.cli.retarget import _object_for_graph, _refinement_components
from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend
from toporetarget.retarget.bones import load_bone_profile
from toporetarget.retarget.final_refinement import (
    ACTIVE_QUERY_PROFILE_ID,
    CollisionQueryProfile,
    PaperRefinementWeights,
    RefinementSolverProfile,
    _make_context,
    build_query_set,
    choose_solver_sdf_backend,
)
from toporetarget.retarget.frames import load_frame_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--collision-samples", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sequence, warm, graph, model, surface, _ = _refinement_components(
        args.canonical,
        args.warm_start,
        args.graph,
        args.robot,
        args.collision_samples,
        None,
    )
    obj = _object_for_graph(sequence, str(graph.metadata["object_id"]))
    reference = build_signed_distance_backend(
        obj.mesh.vertices_local, obj.mesh.faces, sign_mode="strict"
    )
    solver = RefinementSolverProfile.load("scipy_slsqp_active_set_v1")
    sdf, _ = choose_solver_sdf_backend(
        obj.mesh.vertices_local,
        obj.mesh.faces,
        reference,
        solver,
        object_pose_scene=obj.pose_scene.pose_scene[0],
    )
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    paper = PaperRefinementWeights.load()
    context = _make_context(
        sequence,
        graph,
        warm,
        model,
        surface,
        sdf,
        reference,
        frame_profile,
        bone_profile,
        paper,
        args.frame,
        None,
    )
    value_without_slack = np.concatenate([np.zeros(6), context.seed_qpos])
    initial_points = context.candidate_points(value_without_slack)
    initial_sdf = sdf.query_scene(initial_points, context.object_pose_scene)
    query = build_query_set(
        initial_sdf.signed_distance,
        surface.geometry_ids,
        CollisionQueryProfile.load(ACTIVE_QUERY_PROFILE_ID),
    )
    slack = np.clip(
        np.maximum(-paper.tau - query.initial_signed_distance, 0.0),
        0.0,
        paper.b - paper.tau,
    )
    value = np.concatenate([value_without_slack, slack])
    total, analytic_objective, _ = context.objective(value)
    eps = 1e-6
    numerical_objective = np.zeros_like(value)
    for index in range(len(value)):
        plus = value.copy()
        minus = value.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical_objective[index] = (context.objective(plus)[0] - context.objective(minus)[0]) / (
            2 * eps
        )
    constraint_jacobian, jacobian_diag = context.constraint_jacobian(value, query.sample_ids, eps)
    numerical_constraints = np.zeros_like(constraint_jacobian)
    for index in range(len(value)):
        plus = value.copy()
        minus = value.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical_constraints[:, index] = (
            context.constraint_values(plus, query.sample_ids)
            - context.constraint_values(minus, query.sample_ids)
        ) / (2 * eps)
    objective_error = np.abs(analytic_objective - numerical_objective)
    constraint_error = np.abs(constraint_jacobian - numerical_constraints)
    n = context.variable_size_without_slack
    hard_slack_error = float(np.max(np.abs(constraint_jacobian[: query.count, n:])))
    soft_identity_error = float(
        np.max(np.abs(constraint_jacobian[query.count :, n:] - np.eye(query.count)))
    )
    result = {
        "status": "pass"
        if float(np.max(objective_error)) <= 1e-5
        and float(np.sqrt(np.mean(objective_error**2))) <= 1e-6
        and float(np.max(constraint_error)) <= 1e-4
        and hard_slack_error == 0.0
        and soft_identity_error == 0.0
        else "fail",
        "frame": args.frame,
        "query_count": query.count,
        "objective_value": total,
        "objective_gradient_max_abs_error": float(np.max(objective_error)),
        "objective_gradient_rmse": float(np.sqrt(np.mean(objective_error**2))),
        "constraint_jacobian_max_abs_error": float(np.max(constraint_error)),
        "constraint_jacobian_rmse": float(np.sqrt(np.mean(constraint_error**2))),
        "hard_slack_column_max_abs": hard_slack_error,
        "soft_slack_identity_max_abs": soft_identity_error,
        "jacobian_diagnostics": jacobian_diag,
        "finite_difference_epsilon": eps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
