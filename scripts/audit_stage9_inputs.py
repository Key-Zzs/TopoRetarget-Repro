"""Read-only Stage 9 input and solver audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from toporetarget.cli.retarget import _object_for_graph, _refinement_components
from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend
from toporetarget.retarget.final_refinement import (
    RefinementSolverProfile,
    choose_solver_sdf_backend,
    dynamic_collision_points_numpy,
)


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.exists():
        return "missing"
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--right-canonical", type=Path, required=True)
    parser.add_argument("--left-canonical", type=Path, required=True)
    parser.add_argument("--right-warm-start", type=Path, required=True)
    parser.add_argument("--left-warm-start", type=Path, required=True)
    parser.add_argument("--right-graph", type=Path, required=True)
    parser.add_argument("--left-graph", type=Path, required=True)
    parser.add_argument("--right-samples", type=Path, required=True)
    parser.add_argument("--left-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sides = {
        "rh": (
            args.right_canonical,
            args.right_warm_start,
            args.right_graph,
            args.right_samples,
            "artimano_rh",
        ),
        "lh": (
            args.left_canonical,
            args.left_warm_start,
            args.left_graph,
            args.left_samples,
            "artimano_lh",
        ),
    }
    report: dict[str, object] = {
        "status": "pass",
        "schema_version": "toporetarget.stage9.input_solver_audit.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "torch", "zarr")
            if _package_version(name) is not None
        },
        "sides": {},
        "unknown_paper_details": [
            "optimizer and termination details",
            "Q_t construction",
            "base rotation coordinates and prior origin",
            "collision surface density and missing fixed fingertip coverage",
            "SDF derivative convention at closest-face switching",
        ],
    }
    for side, (canonical, warm_path, graph_path, samples_path, robot_name) in sides.items():
        sequence, warm, graph, model, surface, _ = _refinement_components(
            canonical, warm_path, graph_path, robot_name, samples_path, None
        )
        obj = _object_for_graph(sequence, str(graph.metadata["object_id"]))
        reference = build_signed_distance_backend(
            obj.mesh.vertices_local, obj.mesh.faces, sign_mode="strict"
        )
        solver_profile = RefinementSolverProfile.load("scipy_slsqp_active_set_v1")
        solver_sdf, selection = choose_solver_sdf_backend(
            obj.mesh.vertices_local,
            obj.mesh.faces,
            reference,
            solver_profile,
            object_pose_scene=obj.pose_scene.pose_scene[0],
        )
        points = dynamic_collision_points_numpy(
            model, surface, warm.arrays["qpos"][0], warm.arrays["base_pose_scene"][0]
        )
        timings: dict[str, float] = {}
        for count in (1, 32, 128, 512):
            started = time.perf_counter()
            reference.query_scene(points[:count], obj.pose_scene.pose_scene[0])
            timings[str(count)] = time.perf_counter() - started
        report["sides"][side] = {
            "canonical": str(canonical),
            "warm_start": str(warm_path),
            "graph": str(graph_path),
            "collision_samples": str(samples_path),
            "robot": robot_name,
            "frame_count": warm.frame_count,
            "frame_range": [0, warm.frame_count],
            "native_fps": warm.metadata.get("native_fps"),
            "hashes": {
                "canonical": _hash_path(canonical),
                "warm_start": _hash_path(warm_path),
                "graph": _hash_path(graph_path),
                "collision_samples": _hash_path(samples_path),
            },
            "object": {
                "id": obj.object_id,
                "mesh_hash": reference.mesh_hash,
                "mesh_audit": reference.audit(),
                "strict_sign": reference.describe(),
            },
            "collision_surface": {
                "count": surface.count,
                "profile": {
                    **surface.profile.__dict__,
                    "assumptions": list(surface.profile.assumptions),
                },
                "visual_fallback": False,
            },
            "sdf_query_timings_s": timings,
            "solver": {
                "profile": solver_profile.as_dict(),
                "selected_sdf": solver_sdf.describe(),
                "selection": selection,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
