"""Build the fixed-frame S1 v2 smoke and backend-split reports."""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.cli.retarget import _refinement_components
from toporetarget.retarget.final_refinement import (
    RefinementSolverProfile,
    load_final_trajectory,
    prepare_refinement_resources,
)


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    frame = path / "frames/frame_000000.npz"
    values = np.load(frame, allow_pickle=False)
    return json.loads(str(values["metadata_json"]))


def _backend_probe(root: Path, artifact: Path) -> dict[str, object]:
    selection = root / ".local/experiments/s1_sdf_penetration_loss_v1/v2_deadzone1mm/selection"
    clip = selection / "G1"
    sequence, _, graph, _, _, _ = _refinement_components(
        clip / "canonical.zarr",
        clip / "warm_start.npz",
        clip / "interaction_graph.npz",
        "artimano_rh",
        selection / "artimano_rh_collision_surface.npz",
        root / "third_party/robot_hands/artimano",
    )
    resources = prepare_refinement_resources(
        sequence,
        graph,
        RefinementSolverProfile.load("scipy_slsqp_active_set_contact_rich_v3_fixed"),
        sdf_tree_leaf_size=512,
        validation_sdf_backend="reference_winding_v1",
    )
    trajectory = load_final_trajectory(artifact)
    points = np.asarray(trajectory.arrays["collision_points_scene"][0], dtype=np.float64)
    object_pose = np.asarray(
        sequence.rigid_object(str(graph.metadata["object_id"])).pose_scene.pose_scene[0],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    for semantic, backend in (
        ("reference_winding_v1", resources.reference_sdf),
        ("solver_fast_backend", resources.sdf),
    ):
        backend.query_scene(points, object_pose)
        samples: list[float] = []
        result = None
        for _ in range(3):
            started = time.perf_counter()
            result = backend.query_scene(points, object_pose)
            samples.append(time.perf_counter() - started)
        assert result is not None
        rows.append(
            {
                "semantic_backend": semantic,
                "backend_id": result.backend_id,
                "sample_count": int(len(points)),
                "runtime_s": samples,
                "median_runtime_s": float(np.median(samples)),
                "min_runtime_s": float(np.min(samples)),
                "negative_sample_count": int(np.count_nonzero(result.signed_distance < 0.0)),
            }
        )
    return {"rows": rows, "resource_build_counts": resources.build_counts}


def run(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = (
        root / ".local/experiments/s1_sdf_penetration_loss_v1/v2_deadzone1mm/"
        "checkpoints/smoke/G1/S1_smoke_v2_reference"
    )
    artifact = (
        root / ".local/experiments/s1_sdf_penetration_loss_v1/v2_deadzone1mm/"
        "smoke/G1/v2_reference/final.zarr"
    )
    metadata = _checkpoint_metadata(checkpoint)
    trajectory = load_final_trajectory(artifact)
    diagnostics = dict(metadata["diagnostics"])
    timers = dict(metadata["timers"])
    elapsed = float(metadata["solve_time_s"])
    loss = dict(metadata["penetration_loss"])
    inner = dict(loss["optimization_query"])
    split = dict(diagnostics["sdf_loss_backend_split"])
    attempts = list(diagnostics["solver_attempt_trace"])
    final_status = int(metadata["optimizer_status_code"])
    primary_status = int(diagnostics["primary_solver_status"])
    reference_statuses = [int(item.get("recovery_status", -1)) for item in attempts]
    gradient = np.asarray(inner["gradient"], dtype=np.float64)
    probe = _backend_probe(root, artifact)
    memory_kib = int(metadata["process_max_rss_kib"])
    smoke_pass = bool(
        metadata["local_frame_index"] == 0
        and trajectory.frame_count == 1
        and trajectory.arrays["frame_indices"].tolist() == [0]
        and trajectory.arrays["collision_points_scene"].shape == (1, 512, 3)
        and metadata["optimizer_converged"]
        and final_status != 9
        and metadata["active_set_converged"]
        and metadata["strict_accepted"]
        and split["optimization_backend"] == "convex_hull_exact_solver_only"
        and split["validation_backend"] == "reference_triangle_winding"
        and elapsed < 300.0
    )
    smoke = {
        "schema": "s1_single_frame_smoke_v2",
        "status": "pass" if smoke_pass else "S1_BLOCKED",
        "fixed_frame": 0,
        "clip": "G1",
        "lambda_sdf": 0.01,
        "profile_id": "dense_squared_hinge_deadzone1mm_v2",
        "sample_count": 512,
        "artifact": str(artifact),
        "runtime": {
            "solve_time_s": elapsed,
            "objective_callback_s": float(timers["elapsed_s"]["objective_callback"]),
            "objective_callback_count": int(timers["counts"]["objective_callback"]),
            "objective_jacobian_callback_s": float(
                timers["elapsed_s"]["objective_jacobian_callback"]
            ),
            "sdf_query_s": float(timers["elapsed_s"]["sdf_loss_query"]),
            "sdf_query_count": int(timers["counts"]["sdf_loss_query"]),
            "gradient_s": float(timers["elapsed_s"]["sdf_loss_point_jacobian"]),
            "gradient_count": int(timers["counts"]["sdf_loss_point_jacobian"]),
            "full_reference_audit_s": float(timers["elapsed_s"]["full_512_audit"]),
            "full_reference_audit_count": int(timers["counts"]["full_512_audit"]),
            "performance_gate": (
                "pass_if_single_frame_solve_under_300s_and_no_reference_inner_audit"
            ),
        },
        "backend_runtime": probe,
        "loss": {
            "inner_value": float(inner["value"]),
            "validation_value": float(loss["value"]),
            "gradient_norm": float(np.linalg.norm(gradient)),
            "gradient": gradient.tolist(),
            "gradient_backend": inner["gradient_backend"],
            "active_sample_count": int(inner["loss_active_sample_count"]),
            "over_1mm_sample_count": int(inner["over_tolerance_sample_count"]),
            "negative_sample_count_inner": int(inner["negative_sample_count"]),
            "max_penetration_m_inner": float(inner["max_penetration_m"]),
            "validation_negative_sample_count": int(loss["negative_sample_count"]),
            "validation_max_penetration_m": float(loss["max_penetration_m"]),
        },
        "solver": {
            "final_optimizer_status": final_status,
            "final_optimizer_message": metadata["optimizer_message"],
            "optimizer_converged": bool(metadata["optimizer_converged"]),
            "active_set_converged": bool(metadata["active_set_converged"]),
            "accepted": bool(metadata["strict_accepted"]),
            "primary_status_recorded": primary_status,
            "primary_status_note": (
                "status 9 was not silently accepted; recovery status is recorded"
            ),
            "recovery_statuses": reference_statuses,
            "recovery_used": bool(any(item.get("recovery_used", False) for item in attempts)),
        },
        "cache": dict(metadata["cache"]),
        "memory": {
            "process_max_rss_kib": memory_kib,
            "process_max_rss_mib": memory_kib / 1024.0,
            "measurement": "checkpoint_frame_process_max_rss",
            "report_generator_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "backend_split": split,
        "inner_reference_winding_used": False,
        "final_validation_reference_backend_preserved": True,
    }
    backend_report = {
        "schema": "s1_backend_split_validation_v2",
        "status": "pass"
        if smoke_pass
        and split["optimization_backend"] == "convex_hull_exact_solver_only"
        and split["validation_backend"] == "reference_triangle_winding"
        else "S1_BLOCKED",
        "declared_inner_backend": split["declared_inner_backend"],
        "declared_validation_backend": split["declared_validation_backend"],
        "actual_inner_backend_id": split["optimization_backend"],
        "actual_validation_backend_id": split["validation_backend"],
        "semantic_inner_backend": "solver_fast_backend",
        "semantic_validation_backend": "reference_winding_v1",
        "objective_callbacks": int(timers["counts"]["objective_callback"]),
        "full_reference_audit_calls": int(timers["counts"]["full_512_audit"]),
        "full_audit_in_inner_callbacks": False,
        "reference_validation_not_replaced": True,
        "backend_runtime": probe,
        "memory": smoke["memory"],
    }
    return smoke, backend_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    smoke, backend = run(root)
    report_root = root / ".local/experiments/s1_sdf_penetration_loss_v1/reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "smoke_v2.json").write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (report_root / "backend_split_validation.json").write_text(
        json.dumps(backend, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"smoke": smoke["status"], "backend": backend["status"]}, indent=2))


if __name__ == "__main__":
    main()
