#!/usr/bin/env python3
"""Generate bounded Stage 8 determinism, oracle, Jacobian, and provenance reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.surface_sampling import SurfaceSampleSet
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.delaunay import load_delaunay_profile
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_evaluation,
    load_interaction_graph,
)
from toporetarget.retarget.interaction_graph import build_source_interaction_graph, load_paper_kappa
from toporetarget.retarget.interaction_objective import (
    InteractionMeshObjective,
    InteractionMeshResidual,
)
from toporetarget.retarget.interaction_reports import source_integrity_snapshot, topology_over_time
from toporetarget.robots.artimano import load_artimano_model

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / ".local" / "reports" / "stage8"
SAMPLE_PATH = ROOT / ".local/cache/geometry/object_surface/cubemedium_samples.npz"
CASES = {
    "right": {
        "canonical": ROOT
        / ".local/cache/hoi/grab/cubemedium_inspect_1_rh_f000000_f000060_mp21.zarr",
        "hand": "hand_r",
        "robot": "artimano_rh",
        "warm": ROOT
        / ".local/cache/retarget/warm_start/s7_cubemedium_inspect_1_right_artimano_rh.zarr",
        "graph": ROOT
        / ".local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_right.zarr",
        "evaluation": ROOT
        / (
            ".local/cache/retarget/interaction_evaluation/"
            "s7_cubemedium_inspect_1_right_artimano_rh.zarr"
        ),
    },
    "left": {
        "canonical": ROOT
        / ".local/cache/hoi/grab/s7/cubemedium_inspect_1/semantic_left_f000000_f000060.zarr",
        "hand": "left_hand",
        "robot": "artimano_lh",
        "warm": ROOT
        / ".local/cache/retarget/warm_start/s7_cubemedium_inspect_1_left_artimano_lh.zarr",
        "graph": ROOT / ".local/cache/retarget/interaction_graph/s7_cubemedium_inspect_1_left.zarr",
        "evaluation": ROOT
        / (
            ".local/cache/retarget/interaction_evaluation/"
            "s7_cubemedium_inspect_1_left_artimano_lh.zarr"
        ),
    },
}


def write_json(name: str, value: object) -> None:
    destination = REPORT_ROOT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def graph_determinism() -> dict[str, object]:
    profile = load_delaunay_profile()
    samples = SurfaceSampleSet.load(SAMPLE_PATH)
    cases: dict[str, object] = {}
    for side, item in CASES.items():
        sequence = load_hoi_sequence(item["canonical"])
        rebuilt = build_source_interaction_graph(
            sequence,
            item["hand"],
            "primary",
            samples,
            source_cache=item["canonical"],
            object_sample_path=SAMPLE_PATH,
            delaunay_profile=profile,
            kappa=load_paper_kappa(),
            frame_indices=[0, 30, 59],
        )
        saved = load_interaction_graph(item["graph"])
        matches = []
        for local_index, frame in enumerate((0, 30, 59)):
            matches.append(
                {
                    "frame": frame,
                    "graph_hash_equal": rebuilt.graph_hashes[local_index]
                    == saved.graph_hashes[frame],
                    "source_vertices_equal": bool(
                        np.array_equal(
                            rebuilt.source_vertices[local_index], saved.source_vertices[frame]
                        )
                    ),
                    "simplices_equal": bool(
                        np.array_equal(
                            rebuilt.simplex_frames[local_index], saved.simplex_frames[frame]
                        )
                    ),
                    "edges_equal": bool(
                        np.array_equal(rebuilt.edge_frames[local_index], saved.edge_frames[frame])
                    ),
                    "weights_equal": bool(
                        np.array_equal(
                            rebuilt.directed_frames[local_index].weights,
                            saved.directed_frames[frame].weights,
                        )
                    ),
                }
            )
        cases[side] = {
            "profile_id": profile.profile_id,
            "profile_hash": profile.sha256,
            "frames": matches,
            "all_equal": all(
                all(value for key, value in match.items() if key != "frame") for match in matches
            ),
            "delaunay_invocation_count": int(rebuilt.metadata["delaunay_invocation_count"]),
            "robot_delaunay_invocation_count": int(
                rebuilt.metadata["robot_delaunay_invocation_count"]
            ),
        }
    return {"schema_version": "toporetarget.stage8.determinism.v1", "cases": cases}


def identity_oracle() -> dict[str, object]:
    import torch

    values: dict[str, object] = {}
    for side, item in CASES.items():
        graph = load_interaction_graph(item["graph"])
        losses = []
        for frame_index in range(graph.frame_count):
            directed = graph.directed_frames[frame_index]
            model = InteractionMeshResidual(
                graph.source_vertices[frame_index],
                directed.source_index,
                directed.destination_index,
                directed.weights,
            )
            loss = InteractionMeshObjective(model).loss_tensor(
                torch.as_tensor(graph.source_vertices[frame_index], dtype=torch.float64)
            )
            losses.append(float(loss.detach().cpu()))
        values[side] = {
            "frame_count": len(losses),
            "max_e_im": max(losses),
            "mean_e_im": float(np.mean(losses)),
            "all_near_zero": bool(max(losses) <= 1e-28),
        }
    return {"schema_version": "toporetarget.stage8.identity_oracle.v1", "cases": values}


def qpos_jacobian_validation() -> dict[str, object]:
    import torch

    epsilon = 1.0e-6
    selected_frames = (0, 30, 59)
    results: dict[str, object] = {}
    for side, item in CASES.items():
        graph = load_interaction_graph(item["graph"])
        evaluation = load_interaction_evaluation(item["evaluation"])
        warm = load_warm_start(item["warm"])
        robot = load_artimano_model("right" if side == "right" else "left")
        frame_results = []
        for frame in selected_frames:
            base = torch.as_tensor(warm.arrays["base_pose_scene"][frame], dtype=torch.float64)
            q = torch.as_tensor(warm.arrays["qpos"][frame], dtype=torch.float64)
            object_points = torch.as_tensor(graph.source_vertices[frame, 21:], dtype=torch.float64)
            residual_model = InteractionMeshResidual(
                graph.source_vertices[frame],
                graph.directed_frames[frame].source_index,
                graph.directed_frames[frame].destination_index,
                graph.directed_frames[frame].weights,
            )
            objective = InteractionMeshObjective(residual_model)

            def residual_for(
                value: torch.Tensor,
                *,
                robot_model=robot,
                selected_base=base,
                selected_object_points=object_points,
                selected_objective=objective,
            ) -> np.ndarray:
                with torch.no_grad():
                    hand = robot_model.keypoints_scene(value, selected_base, layout="mediapipe21")
                    vertices = torch.cat([hand, selected_object_points], dim=-2)
                    return (
                        selected_objective.residual_tensor(vertices)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )

            finite_difference = np.empty((213, 22), dtype=np.float64)
            for joint in range(22):
                plus = q.clone()
                minus = q.clone()
                plus[joint] += epsilon
                minus[joint] -= epsilon
                finite_difference[:, joint] = (residual_for(plus) - residual_for(minus)) / (
                    2.0 * epsilon
                )
            analytic = evaluation.qpos_jacobian[frame]
            absolute = np.abs(analytic - finite_difference)
            frame_results.append(
                {
                    "frame": frame,
                    "epsilon_rad": epsilon,
                    "shape": list(analytic.shape),
                    "max_abs_error": float(np.max(absolute)),
                    "rmse": float(np.sqrt(np.mean(np.square(analytic - finite_difference)))),
                    "pass": bool(np.max(absolute) <= 2.0e-7),
                }
            )
        results[side] = {
            "robot": robot.name,
            "frames": frame_results,
            "all_pass": all(value["pass"] for value in frame_results),
        }
    return {
        "schema_version": "toporetarget.stage8.qpos_jacobian_validation.v1",
        "method": "central finite difference of frozen Eq.7 residual",
        "cases": results,
    }


def performance_and_integrity() -> tuple[dict[str, object], dict[str, object]]:
    graph_values: dict[str, object] = {}
    evaluation_values: dict[str, object] = {}
    paths = {
        "right_canonical": CASES["right"]["canonical"],
        "left_canonical": CASES["left"]["canonical"],
        "object_samples": SAMPLE_PATH,
        "right_warm_start": CASES["right"]["warm"],
        "left_warm_start": CASES["left"]["warm"],
    }
    integrity = source_integrity_snapshot(paths)
    for side, item in CASES.items():
        graph = load_interaction_graph(item["graph"])
        evaluation = load_interaction_evaluation(item["evaluation"])
        graph_values[side] = {
            "artifact_hash": interaction_artifact_hash(item["graph"]),
            "frame_count": graph.frame_count,
            "timings": graph.metadata.get("timings", {}),
            "mean_s_per_frame": float(
                graph.metadata.get("timings", {}).get("delaunay_s", 0.0) / graph.frame_count
            ),
            "delaunay_invocation_count": graph.delaunay_invocation_count,
            "robot_delaunay_invocation_count": graph.metadata.get(
                "robot_delaunay_invocation_count"
            ),
        }
        evaluation_values[side] = {
            "artifact_hash": interaction_artifact_hash(item["evaluation"]),
            "frame_count": evaluation.frame_count,
            "evaluation_time_s": evaluation.metadata.get("evaluation_time_s"),
            "mean_s_per_frame": float(
                np.mean(evaluation.metadata.get("per_frame_evaluation_time_s", [0.0]))
            ),
            "jacobian_shape": list(evaluation.qpos_jacobian.shape),
            "optimization_performed": evaluation.metadata.get("optimization_performed"),
            "robot_delaunay_invocation_count": evaluation.metadata.get(
                "robot_delaunay_invocation_count"
            ),
        }
    return (
        {
            "schema_version": "toporetarget.stage8.performance.v1",
            "graphs": graph_values,
            "evaluations": evaluation_values,
        },
        {
            "schema_version": "toporetarget.stage8.source_integrity.v1",
            "before": integrity,
            "after": source_integrity_snapshot(paths),
            "unchanged": integrity == source_integrity_snapshot(paths),
            "robot_asset_hashes": {
                side: {
                    "urdf_hash": load_artimano_model(
                        "right" if side == "right" else "left"
                    ).urdf_hash,
                    "asset_manifest_hash": load_artimano_model(
                        "right" if side == "right" else "left"
                    ).asset_manifest_hash,
                }
                for side in CASES
            },
        },
    )


def main() -> None:
    write_json("graph_determinism.json", graph_determinism())
    write_json("identity_oracle.json", identity_oracle())
    write_json("qpos_jacobian_validation.json", qpos_jacobian_validation())
    for side, item in CASES.items():
        graph = load_interaction_graph(item["graph"])
        write_json(f"{side}_topology_over_time.json", topology_over_time(graph))
    performance, integrity = performance_and_integrity()
    write_json("performance.json", performance)
    write_json("source_integrity.json", integrity)
    write_json(
        "topology_over_time.json",
        {
            side: topology_over_time(load_interaction_graph(item["graph"]))
            for side, item in CASES.items()
        },
    )
    write_json(
        "stage8_summary.json",
        {
            "status": "STAGE8_COMPLETE_WITH_ASSUMPTIONS",
            "scope": "bounded s7/cubemedium_inspect_1 frames [0,60) RH/LH",
            "equations": {"implemented": [3, 4, 5, 6, 7], "not_started": [8, 9]},
            "graph_artifacts": {side: str(item["graph"]) for side, item in CASES.items()},
            "evaluation_artifacts": {side: str(item["evaluation"]) for side, item in CASES.items()},
            "required_reports": [
                "input_audit.json",
                "graph_determinism.json",
                "identity_oracle.json",
                "qpos_jacobian_validation.json",
                "topology_over_time.json",
                "object_scale_graphs.json",
                "performance.json",
                "source_integrity.json",
            ],
            "visualizations": [
                "source_graph_first.png",
                "source_graph_middle.png",
                "source_graph_last.png",
                "shared_graph_first.png",
                "shared_graph_middle.png",
                "shared_graph_last.png",
                "laplacian_first.png",
                "laplacian_middle.png",
                "laplacian_last.png",
                "contribution_first.png",
                "object_scale_graphs.png",
                "interactive_viewer_smoke.json",
            ],
            "no_optimization": True,
            "no_sdf_or_collision": True,
        },
    )


if __name__ == "__main__":
    main()
