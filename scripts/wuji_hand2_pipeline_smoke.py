"""Bounded generic Stage 7-9 smoke for the Wuji Hand 2 Beta 1 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.bones import load_bone_profile
from toporetarget.retarget.final_refinement import (
    ACTIVE_QUERY_PROFILE_ID,
    SOLVER_PROFILE_ID,
    CollisionQueryProfile,
    PaperRefinementWeights,
    RefinementSolverProfile,
    _make_context,
    build_query_set,
    load_robot_surface_samples,
    prepare_refinement_resources,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import load_interaction_graph
from toporetarget.robots.registry import get_robot_registry
from toporetarget.utils.hashing import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path(".local/cache/hoi/grab/airplane_lift_rh_mp21.zarr"),
    )
    parser.add_argument(
        "--warm-start", type=Path, default=Path(".local/reports/wuji_hand2/warm_start.zarr")
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path(".local/reports/wuji_hand2/interaction_graph.zarr"),
    )
    parser.add_argument(
        "--collision-samples",
        type=Path,
        default=Path(".local/reports/wuji_hand2/wuji_hand2_beta1_rh_neutral.npz"),
    )
    parser.add_argument("--robot", default="wuji_hand2_beta1_rh")
    parser.add_argument(
        "--output", type=Path, default=Path(".local/reports/wuji_hand2/pipeline_smoke.json")
    )
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve()
    canonical = (repo / args.canonical).resolve()
    warm_path = (repo / args.warm_start).resolve()
    graph_path = (repo / args.graph).resolve()
    surface_path = (repo / args.collision_samples).resolve()
    output = (repo / args.output).resolve()

    sequence = load_hoi_sequence(canonical)
    warm = load_warm_start(warm_path)
    graph = load_interaction_graph(graph_path)
    model = get_robot_registry(repo_root=repo).load(args.robot)
    surface = load_robot_surface_samples(surface_path)
    solver_profile = RefinementSolverProfile.load(SOLVER_PROFILE_ID, repo)
    resources = prepare_refinement_resources(sequence, graph, solver_profile)
    frame_profile = load_frame_profile(
        "canonical_keypoint_wrist_v1", config_root=repo / "configs/retarget/frames"
    )
    bone_profile = load_bone_profile(
        "mediapipe21_full_finger_chain_v1", config_root=repo / "configs/retarget/bones"
    )
    paper = PaperRefinementWeights.load(repo)
    context = _make_context(
        sequence,
        graph,
        warm,
        model,
        surface,
        resources.sdf,
        resources.reference_sdf,
        frame_profile,
        bone_profile,
        paper,
        0,
        None,
    )

    seed = np.concatenate([np.zeros(6, dtype=np.float64), warm.arrays["qpos"][0]])
    collision_points = context.candidate_points(seed)
    object_pose = context.object_pose_scene
    initial_query = resources.sdf.query_scene(collision_points, object_pose)
    query_profile = CollisionQueryProfile.load(ACTIVE_QUERY_PROFILE_ID, repo)
    query_set = build_query_set(initial_query.signed_distance, surface.geometry_ids, query_profile)
    value = np.concatenate([seed, np.zeros(query_set.count, dtype=np.float64)])
    objective, gradient, breakdown = context.objective(value, query_set.query_hash)
    constraints = context.constraint_values(value, query_set.sample_ids, query_set.query_hash)
    constraint_jacobian, jacobian_diagnostics = context.constraint_jacobian(
        value,
        query_set.sample_ids,
        solver_profile.finite_difference_epsilon,
        query_set.query_hash,
    )
    reference_probe = build_signed_distance_backend(
        sequence.rigid_object(str(graph.metadata["object_id"])).mesh.vertices_local,
        sequence.rigid_object(str(graph.metadata["object_id"])).mesh.faces,
        sign_mode="strict",
    ).query_scene(collision_points, object_pose)
    instances = model.collision_geometry_instances(model.neutral_q)

    finite = all(
        np.all(np.isfinite(np.asarray(item)))
        for item in (
            collision_points,
            constraints,
            constraint_jacobian,
            reference_probe.signed_distance,
        )
    )
    report = {
        "status": "pass" if finite else "fail",
        "robot": model.name,
        "robot_dof_count": model.num_dofs,
        "frame": int(graph.frame_indices[0]),
        "input_hashes": {
            "canonical": str(canonical),
            "warm_start": str(warm_path),
            "graph": str(graph_path),
            "collision_samples_sha256": sha256_file(surface_path),
        },
        "surface": {
            "sample_count": surface.count,
            "geometry_count": len(instances),
            "profile_id": surface.profile.profile_id,
            "visual_fallback": surface.profile.visual_fallback,
            "tip_visual_fallback": surface.profile.tip_visual_fallback,
        },
        "stage9_construction": {
            "objective_constructed": True,
            "constraint_constructed": True,
            "constraint_jacobian_constructed": True,
            "optimization_performed": False,
            "formal_stage9_solve": "not_run",
            "variable_size": int(len(value)),
            "query_count": query_set.count,
            "objective": float(objective),
            "objective_gradient_norm": float(np.linalg.norm(gradient)),
            "breakdown": breakdown.as_dict(),
            "constraint_shape": list(constraints.shape),
            "constraint_jacobian_shape": list(constraint_jacobian.shape),
            "constraint_jacobian_diagnostics": jacobian_diagnostics,
            "collision_points_shape": list(collision_points.shape),
            "all_values_finite": finite,
            "sdf_backend": resources.sdf.backend_id,
            "reference_sdf_backend": resources.reference_sdf.backend_id,
        },
        "provenance": {
            "source_cache_hash": warm.metadata.get("source_cache_hash"),
            "robot_spec_hash": model.spec_hash,
            "asset_manifest_hash": model.asset_manifest_hash,
            "stage9_started": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if finite else 1


if __name__ == "__main__":
    raise SystemExit(main())
