#!/usr/bin/env python3
"""Separate Stage 10 window geometry from Stage 9 SLSQP termination.

This is a read-only diagnostic.  It consumes the existing Stage 5-8 artifacts,
does not use semantic contacts in the Stage 9 objective, and writes the three
bounded closeout reports requested by Stage 9.1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.se3 import transform_points
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.delaunay import edge_category
from toporetarget.retarget.final_refinement import (
    RefinementSolverProfile,
    choose_solver_sdf_backend,
    dynamic_collision_points_numpy,
    load_final_trajectory,
    load_robot_surface_samples,
    so3_exp,
)
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_evaluation,
    load_interaction_graph,
)
from toporetarget.retarget.interaction_objective import (
    InteractionMeshObjective,
    InteractionMeshResidual,
)
from toporetarget.robots.artimano import load_artimano_model

DIAGNOSTIC_SCHEMA_VERSION = "toporetarget.stage9_solver_closeout.diagnostics.v1"
DIAGNOSTIC_CONFIG = {
    "contact_rich_semantic_frame_ratio_min": 0.5,
    "pre_contact_semantic_frame_ratio_max": 0.1,
    "approach_initial_contact_vertex_count_max": 64,
    "approach_initial_source_separation_min": 0.04,
    "source_contact_geometry_median_max": 0.02,
    "pre_contact_separation_median_min": 0.04,
    "invalid_coordinate_scale_separation_min": 0.2,
    "distance_units": "m",
    "thresholds_are_engineering_diagnostics_not_paper_parameters": True,
}
OVERLAY_EVIDENCE = [
    ".local/reports/stage5_closeout/compare_overlay_semantic.png",
    ".local/reports/stage5_closeout/semantic_contact_first.png",
    ".local/reports/stage5_closeout/semantic_contact_middle.png",
    ".local/reports/stage5_closeout/semantic_contact_last.png",
]


def _stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {
        "min": float(np.min(value)),
        "median": float(np.median(value)),
        "p95": float(np.percentile(value, 95)),
        "max": float(np.max(value)),
    }


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _window_cases(repo: Path) -> list[dict[str, Any]]:
    root = repo / ".local/runs/stage10"
    contact_root = root / "s1__airplane_lift__right__artimano_rh__f000240_f000300/artifacts"
    approach_root = root / "s1__airplane_lift__right__artimano_rh__f000238_f000298/artifacts"
    pre_canonical = repo / ".local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr"
    pre_warm = (
        repo / ".local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr"
    )
    pre_graph = repo / ".local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr"
    pre_evaluation = (
        repo / ".local/cache/retarget/interaction_evaluation/"
        "s7_cubemedium_inspect_1_right_artimano_rh.zarr"
    )
    return [
        {
            "window_id": "stage10_contact_rich_f0240_f0300",
            "sequence": "s1/airplane_lift",
            "hand": "right",
            "global_frame_range": [240, 300],
            "window_class_expected": "contact_rich",
            "canonical": contact_root / "canonical.zarr",
            "warm_start": contact_root / "warm_start.zarr",
            "graph": contact_root / "interaction_graph.zarr",
            "final": None,
            "robot": "artimano_rh",
            "window_role": "contact-rich Stage 10 selection",
        },
        {
            "window_id": "stage10_approach_f0238_f0298",
            "sequence": "s1/airplane_lift",
            "hand": "right",
            "global_frame_range": [238, 298],
            "window_class_expected": "approach",
            "canonical": approach_root / "canonical.zarr",
            "warm_start": approach_root / "warm_start.zarr",
            "graph": approach_root / "interaction_graph.zarr",
            "final": None,
            "robot": "artimano_rh",
            "window_role": "approach comparison",
        },
        {
            "window_id": "stage9_pre_contact_regression_s7_rh",
            "sequence": "s7/cubemedium_inspect_1",
            "hand": "right",
            "global_frame_range": [0, 60],
            "window_class_expected": "pre_contact",
            "canonical": pre_canonical,
            "warm_start": pre_warm,
            "graph": pre_graph,
            "evaluation": pre_evaluation,
            "final": None,
            "robot": "artimano_rh",
            "window_role": "pre-contact regression",
        },
    ]


def _contact_for(sequence: Any, hand_id: str, object_id: str) -> Any | None:
    for contact in sequence.contacts:
        if contact.hand_id == hand_id and contact.object_id == object_id:
            return contact
    return None


def _semantic_mask(contact: Any, frame: int, vertex_count: int) -> np.ndarray:
    if contact is None:
        return np.zeros(vertex_count, dtype=bool)
    if contact.binary is not None:
        value = np.asarray(contact.binary[frame], dtype=bool).reshape(-1)
    elif contact.semantic_labels is not None:
        value = np.asarray(contact.semantic_labels[frame]).reshape(-1) != 0
    elif contact.labels is not None:
        no_contact = int(contact.metadata.get("no_contact_label", 0))
        value = np.asarray(contact.labels[frame]).reshape(-1) != no_contact
    else:
        value = np.zeros(vertex_count, dtype=bool)
    if len(value) != vertex_count:
        raise ValueError("contact object-vertex count does not match object mesh")
    return value


def _frame_distances(
    sequence: Any,
    warm: Any,
    model: Any,
    surface: Any,
    sdf: Any,
    hand: Any,
    obj: Any,
    contact: Any,
    frame: int,
    final: Any | None,
) -> dict[str, Any]:
    object_pose = np.asarray(obj.pose_scene.pose_scene[frame], dtype=np.float64)
    source_surface = np.asarray(hand.vertices_scene[frame], dtype=np.float64)
    source_mp = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene[frame])
    source_sdf = sdf.query_scene(source_surface, object_pose)
    source_mp_sdf = sdf.query_scene(source_mp, object_pose)
    object_vertices = transform_points(object_pose, obj.mesh.vertices_local)
    semantic = _semantic_mask(contact, frame, len(object_vertices))
    semantic_points = object_vertices[semantic]
    tree = cKDTree(source_surface)
    semantic_to_source = (
        tree.query(semantic_points, k=1)[0] if len(semantic_points) else np.empty(0)
    )

    warm_points = dynamic_collision_points_numpy(
        model, surface, warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame]
    )
    warm_mp = _as_numpy(
        model.keypoints_scene(
            warm.arrays["qpos"][frame], warm.arrays["base_pose_scene"][frame], layout="mediapipe21"
        )
    )
    warm_sdf = sdf.query_scene(warm_points, object_pose)
    warm_mp_sdf = sdf.query_scene(warm_mp, object_pose)
    final_row: dict[str, Any] | None = None
    if final is not None:
        final_points = np.asarray(final.arrays["collision_points_scene"][frame])
        final_mp = np.asarray(final.arrays["robot_keypoints_scene"][frame])
        final_sdf = sdf.query_scene(final_points, object_pose)
        final_mp_sdf = sdf.query_scene(final_mp, object_pose)
        final_row = {
            "collision_surface_to_object_m": _stats(final_sdf.unsigned_distance),
            "mediapipe21_to_object_m": _stats(final_mp_sdf.unsigned_distance),
            "full_surface_min_signed_distance_m": float(
                np.min(final.arrays["full_signed_distance"][frame])
            ),
        }
    return {
        "frame": frame,
        "global_frame": int(frame),
        "semantic_contact": bool(np.any(semantic)),
        "semantic_contact_vertex_count": int(np.count_nonzero(semantic)),
        "source_mano_surface_to_object_m": _stats(source_sdf.unsigned_distance),
        "source_mediapipe21_to_object_m": _stats(source_mp_sdf.unsigned_distance),
        "warm_robot_collision_surface_to_object_m": _stats(warm_sdf.unsigned_distance),
        "warm_robot_mediapipe21_to_object_m": _stats(warm_mp_sdf.unsigned_distance),
        "semantic_contact_object_vertices_to_source_hand_surface_m": _stats(semantic_to_source),
        "final_robot": final_row,
    }


def _classify(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    ratio = float(np.mean([bool(row["semantic_contact"]) for row in rows]))
    source_contact = [
        row["semantic_contact_object_vertices_to_source_hand_surface_m"]["median"]
        for row in rows
        if row["semantic_contact_object_vertices_to_source_hand_surface_m"]["median"] is not None
    ]
    source_separation = [row["source_mano_surface_to_object_m"]["median"] for row in rows]
    contact_median = float(np.median(source_contact)) if source_contact else None
    separation_median = float(np.median(source_separation)) if source_separation else None
    initial = rows[0]
    initial_contact_vertices = int(initial["semantic_contact_vertex_count"])
    initial_separation = initial["source_mano_surface_to_object_m"]["median"]
    is_approach_onset = (
        ratio >= DIAGNOSTIC_CONFIG["contact_rich_semantic_frame_ratio_min"]
        and initial_contact_vertices
        <= DIAGNOSTIC_CONFIG["approach_initial_contact_vertex_count_max"]
        and initial_separation is not None
        and initial_separation >= DIAGNOSTIC_CONFIG["approach_initial_source_separation_min"]
    )
    if is_approach_onset:
        classification = "approach"
    elif (
        ratio >= DIAGNOSTIC_CONFIG["contact_rich_semantic_frame_ratio_min"]
        and contact_median is not None
        and contact_median <= DIAGNOSTIC_CONFIG["source_contact_geometry_median_max"]
    ):
        classification = "contact_rich"
    elif (
        ratio <= DIAGNOSTIC_CONFIG["pre_contact_semantic_frame_ratio_max"]
        and separation_median is not None
        and separation_median >= DIAGNOSTIC_CONFIG["pre_contact_separation_median_min"]
    ):
        classification = "pre_contact"
    elif (
        ratio >= DIAGNOSTIC_CONFIG["contact_rich_semantic_frame_ratio_min"]
        and separation_median is not None
        and separation_median >= DIAGNOSTIC_CONFIG["invalid_coordinate_scale_separation_min"]
    ):
        classification = "invalid_coordinate_or_scale"
    else:
        classification = "approach"
    return classification, {
        "semantic_contact_frame_ratio": ratio,
        "initial_semantic_contact_vertex_count": initial_contact_vertices,
        "initial_source_mano_object_m": initial_separation,
        "approach_onset_rule_pass": is_approach_onset,
        "source_contact_geometry_median_m": contact_median,
        "source_mano_object_median_m": separation_median,
    }


def _conditioning_frame(
    graph_frame: Any, evaluation: Any, model: Any, frame: int
) -> dict[str, Any]:
    import torch

    edges = np.asarray(graph_frame.edges, dtype=np.int64)
    categories = np.asarray([edge_category(edge) for edge in edges], dtype=object)
    directed = graph_frame.directed
    directed_categories = np.asarray(
        [
            edge_category((src, dst))
            for src, dst in zip(directed.source_index, directed.destination_index, strict=True)
        ],
        dtype=object,
    )
    hand_object_mask = categories == "hand-object"
    outgoing = np.zeros(21, dtype=np.float64)
    outgoing_mask = (directed.source_index < 21) & (directed.destination_index >= 21)
    np.add.at(outgoing, directed.source_index[outgoing_mask], directed.weights[outgoing_mask])
    hand_object_directed = directed_categories == "hand-object"
    hand_object_lengths = np.linalg.norm(
        graph_frame.source_vertices[edges[hand_object_mask, 0]]
        - graph_frame.source_vertices[edges[hand_object_mask, 1]],
        axis=1,
    )

    source_vertices = np.asarray(graph_frame.source_vertices, dtype=np.float64)
    residual_model = InteractionMeshResidual(
        source_vertices,
        directed.source_index,
        directed.destination_index,
        directed.weights,
    )
    objective = InteractionMeshObjective(residual_model)
    q = torch.as_tensor(evaluation.qpos[frame], dtype=torch.float64).requires_grad_(True)
    base = torch.as_tensor(evaluation.base_pose_scene[frame], dtype=torch.float64)
    xi = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    object_points = torch.as_tensor(source_vertices[21:], dtype=torch.float64)

    def robot_vertices(q_value: Any, xi_value: Any) -> Any:
        rotation = so3_exp(xi_value[3:]) @ base[:3, :3]
        transformed_base = torch.eye(4, dtype=torch.float64)
        transformed_base = transformed_base.clone()
        transformed_base[:3, :3] = rotation
        transformed_base[:3, 3] = base[:3, 3] + xi_value[:3]
        hand = model.keypoints_scene(q_value, transformed_base, layout="mediapipe21")
        return torch.cat([hand, object_points], dim=0)

    loss = objective.loss_tensor(robot_vertices(q, xi))
    grad_q, grad_xi = torch.autograd.grad(loss, (q, xi), create_graph=False)
    qpos_jacobian = np.asarray(evaluation.qpos_jacobian[frame], dtype=np.float64)
    singular = np.linalg.svd(qpos_jacobian, compute_uv=False)
    scale = max(float(singular[0]) if len(singular) else 0.0, 1e-15)
    rank = int(np.count_nonzero(singular > scale * 1e-8))
    base_gradient = _as_numpy(grad_xi)
    base_scale = max(float(np.max(np.abs(base_gradient))), 1e-15)
    base_near_null = int(np.count_nonzero(np.abs(base_gradient) <= base_scale * 1e-3))
    return {
        "frame": int(frame),
        "hand_hand_edge_count": int(np.count_nonzero(categories == "hand-hand")),
        "hand_object_edge_count": int(np.count_nonzero(categories == "hand-object")),
        "object_object_edge_count": int(np.count_nonzero(categories == "object-object")),
        "hand_object_outgoing_weight_sum_by_hand_vertex": outgoing.tolist(),
        "total_hand_object_directed_weight_mass": float(
            np.sum(directed.weights[hand_object_directed])
        ),
        "hand_object_edge_length_m": _stats(hand_object_lengths),
        "e_im_base_translation_gradient_norm": float(np.linalg.norm(_as_numpy(grad_xi)[:3])),
        "e_im_base_rotation_gradient_norm": float(np.linalg.norm(_as_numpy(grad_xi)[3:])),
        "e_im_qpos_gradient_norm": float(np.linalg.norm(_as_numpy(grad_q))),
        "base_residual_jacobian_singular_values": singular.tolist(),
        "jacobian_source": "Stage8 evaluation qpos_jacobian",
        "base_residual_jacobian_numerical_rank": rank,
        "base_residual_jacobian_condition_estimate": float(singular[0] / max(singular[-1], 1e-15))
        if len(singular)
        else None,
        "near_null_base_directions": base_near_null,
        "base_gradient_components": base_gradient.tolist(),
        "e_im": float(loss.detach().cpu()),
    }


def audit_window(case: dict[str, Any], repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sequence = load_hoi_sequence(case["canonical"])
    warm = load_warm_start(case["warm_start"])
    graph = load_interaction_graph(case["graph"])
    model = load_artimano_model(
        "rh" if case["robot"].endswith("rh") else "lh",
        asset_root=repo / ".local/assets/artimano",
    )
    surface = load_robot_surface_samples(
        repo / ".local/cache/geometry/robot_surface" / f"{case['robot']}_neutral.npz"
    )
    obj = sequence.rigid_objects[0]
    hand = next(item for item in sequence.hands if item.side == model.side)
    contact = _contact_for(sequence, hand.hand_id, obj.object_id)
    from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend

    reference_sdf = build_signed_distance_backend(
        obj.mesh.vertices_local, obj.mesh.faces, sign_mode="strict"
    )
    solver_profile = RefinementSolverProfile.load(
        "scipy_slsqp_active_set_contact_rich_v2", root=repo
    )
    sdf, sdf_report = choose_solver_sdf_backend(
        obj.mesh.vertices_local,
        obj.mesh.faces,
        reference_sdf,
        solver_profile,
        object_pose_scene=np.asarray(obj.pose_scene.pose_scene[0]),
    )
    final = None
    if case.get("final") and Path(case["final"]).exists():
        final = load_final_trajectory(case["final"])
    rows = [
        _frame_distances(sequence, warm, model, surface, sdf, hand, obj, contact, frame, final)
        for frame in range(warm.frame_count)
    ]
    classification, class_metrics = _classify(rows)
    geometry = {
        "window_id": case["window_id"],
        "window_identity": {
            "sequence": case["sequence"],
            "hand": case["hand"],
            "global_frame_range": case["global_frame_range"],
        },
        "window_role": case["window_role"],
        "window_class_expected": case["window_class_expected"],
        "window_class": classification,
        "classification_matches_expected": classification == case["window_class_expected"],
        "classification_metrics": class_metrics,
        "frame_count": warm.frame_count,
        "frame_indices_contiguous": bool(
            np.array_equal(graph.frame_indices, np.arange(warm.frame_count))
        ),
        "timestamps_monotonic": bool(np.all(np.diff(warm.arrays["timestamps"]) >= 0)),
        "coordinate_contract": {
            "source_frame": "S",
            "object_pose_frame": "S",
            "robot_collision_frame": "S",
            "units": "m",
            "signed_distance_sign": "positive_outside",
            "sdf_backend": sdf.describe(),
            "sdf_backend_selection": sdf_report,
            "source_cache_hash": warm.metadata.get("source_cache_hash"),
            "canonical_hash": artifact_hash(case["canonical"]),
            "warm_start_hash": artifact_hash(case["warm_start"]),
            "graph_hash": interaction_artifact_hash(case["graph"]),
        },
        "semantic_contact": {
            "representation": None if contact is None else contact.source_contact_representation,
            "frame_ratio": float(np.mean([row["semantic_contact"] for row in rows])),
            "frame_count": int(np.count_nonzero([row["semantic_contact"] for row in rows])),
            "vertex_count_by_frame": [row["semantic_contact_vertex_count"] for row in rows],
        },
        "visual_overlay": {
            "status": "available_for_source_coordinate_review",
            "evidence": [str(repo / item) for item in OVERLAY_EVIDENCE if (repo / item).exists()],
            "review_scope": (
                "Stage 5 source/canonical scene overlay; numerical pose and units "
                "are the authoritative audit"
            ),
        },
        "aggregate_distances": {
            "source_mano_surface_to_object_m": _stats(
                [row["source_mano_surface_to_object_m"]["median"] for row in rows]
            ),
            "source_mediapipe21_to_object_m": _stats(
                [row["source_mediapipe21_to_object_m"]["median"] for row in rows]
            ),
            "warm_robot_collision_surface_to_object_m": _stats(
                [row["warm_robot_collision_surface_to_object_m"]["median"] for row in rows]
            ),
            "warm_robot_mediapipe21_to_object_m": _stats(
                [row["warm_robot_mediapipe21_to_object_m"]["median"] for row in rows]
            ),
            "semantic_contact_object_vertices_to_source_hand_surface_m": _stats(
                [
                    row["semantic_contact_object_vertices_to_source_hand_surface_m"]["median"]
                    for row in rows
                    if row["semantic_contact_object_vertices_to_source_hand_surface_m"]["median"]
                    is not None
                ]
            ),
        },
        "frames": rows,
    }
    conditioning_frames: list[dict[str, Any]] = []
    evaluation_path = Path(
        case.get("evaluation") or case["graph"].parent / "interaction_evaluation.zarr"
    )
    if evaluation_path.exists():
        evaluation = load_interaction_evaluation(evaluation_path)
        for frame in range(graph.frame_count):
            conditioning_frames.append(
                _conditioning_frame(graph.frames[frame], evaluation, model, frame)
            )
    conditioning = {
        "window_id": case["window_id"],
        "window_class": classification,
        "graph_artifact_hash": interaction_artifact_hash(case["graph"]),
        "evaluation_artifact": str(evaluation_path) if evaluation_path.exists() else None,
        "frames": conditioning_frames,
        "aggregate": {
            "hand_object_edge_count": _stats(
                [row["hand_object_edge_count"] for row in conditioning_frames]
            ),
            "total_hand_object_directed_weight_mass": _stats(
                [row["total_hand_object_directed_weight_mass"] for row in conditioning_frames]
            ),
            "e_im_base_translation_gradient_norm": _stats(
                [row["e_im_base_translation_gradient_norm"] for row in conditioning_frames]
            ),
            "e_im_base_rotation_gradient_norm": _stats(
                [row["e_im_base_rotation_gradient_norm"] for row in conditioning_frames]
            ),
            "e_im_qpos_gradient_norm": _stats(
                [row["e_im_qpos_gradient_norm"] for row in conditioning_frames]
            ),
            "base_jacobian_condition_estimate": _stats(
                [row["base_residual_jacobian_condition_estimate"] for row in conditioning_frames]
            ),
            "near_null_base_directions": _stats(
                [row["near_null_base_directions"] for row in conditioning_frames]
            ),
        },
    }
    return geometry, conditioning


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
    args = parser.parse_args()
    repo = args.repo.resolve()
    geometries: list[dict[str, Any]] = []
    conditionings: list[dict[str, Any]] = []
    for case in _window_cases(repo):
        geometry, conditioning = audit_window(case, repo)
        geometries.append(geometry)
        conditionings.append(conditioning)
    geometry_payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_config": DIAGNOSTIC_CONFIG,
        "windows": geometries,
        "source_integrity": "read_only_stage5_to_stage8_inputs",
    }
    conditioning_payload = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_config": DIAGNOSTIC_CONFIG,
        "windows": conditionings,
        "interpretation_guard": (
            "graph edge existence alone is not evidence of effective interaction "
            "coupling; use normalized weight mass and sensitivity metrics"
        ),
    }
    for path, payload in (
        (args.geometry_report, geometry_payload),
        (args.conditioning_report, conditioning_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "geometry_report": str(args.geometry_report),
                "conditioning_report": str(args.conditioning_report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
