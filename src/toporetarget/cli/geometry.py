"""Stage 6 object geometry, SDF, and collision-surface commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import typer

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.collision_queries import json_ready_probe, probe_robot_surface
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.object_geometry import (
    load_mesh_file,
    sample_object_track,
    validate_temporal_reuse,
)
from toporetarget.geometry.reports import json_ready, write_flat_csv, write_json
from toporetarget.geometry.robot_surface import (
    RobotSurfaceSampleSet,
    load_robot_surface_profile,
    sample_robot_collision_surface,
    transform_robot_surface_to_scene,
)
from toporetarget.geometry.signed_distance.reference import (
    ReferenceSignedDistanceBackend,
    SignedDistanceError,
)
from toporetarget.geometry.signed_distance.validation import (
    make_synthetic_mesh,
    validate_analytic_shape,
    validate_synthetic_shape,
)
from toporetarget.geometry.surface_artifacts import load_surface_artifact
from toporetarget.geometry.surface_sampling import SurfaceSampleSet, load_surface_profile
from toporetarget.robots.artimano import load_artimano_model
from toporetarget.robots.registry import get_robot_registry

app = typer.Typer(
    help="Stage 6 object geometry, deterministic surfaces, SDF, and collision probes."
)


def _print(payload: Any) -> None:
    typer.echo(json.dumps(json_ready(payload), indent=2, sort_keys=True, default=str))


def _canonical_object(canonical: Path, object_id: str) -> Any:
    sequence = load_hoi_sequence(canonical)
    if object_id in {"primary", "object"}:
        return sequence.primary_rigid_object()
    return sequence.rigid_object(object_id)


def _load_mesh_or_canonical(
    mesh: Path | None, canonical: Path | None, object_id: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if mesh is not None:
        vertices, faces = load_mesh_file(mesh)
        return vertices, faces, {"source_path": str(mesh), "object_id": object_id}
    if canonical is None:
        raise typer.BadParameter("provide either --mesh or --canonical")
    track = _canonical_object(canonical, object_id)
    return (
        np.asarray(track.mesh.vertices_local),
        np.asarray(track.mesh.faces),
        {
            "source_path": track.metadata.get("source_mesh"),
            "canonical": str(canonical),
            "object_id": track.object_id,
        },
    )


def _robot_model(robot: str, asset_root: Path | None) -> Any:
    if robot in {"artimano_rh", "artimano_lh", "rh", "lh", "right", "left"}:
        side = "right" if robot in {"artimano_rh", "rh", "right"} else "left"
        return load_artimano_model(side, asset_root=asset_root)
    return get_robot_registry().load(robot, asset_root=asset_root)


def _fk_transform_error(model: Any, qpos: np.ndarray, samples: RobotSurfaceSampleSet) -> float:
    instances = model.collision_geometry_instances(np.asarray(qpos, dtype=np.float64))
    expected_base: list[np.ndarray] = []
    offset = 0
    for instance in instances:
        count = int(np.count_nonzero(samples.geometry_ids == samples.geometry_ids[offset]))
        transform = np.asarray(instance.transform_base, dtype=np.float64)
        expected_base.append(
            samples.points_local[offset : offset + count] @ transform[:3, :3].T + transform[:3, 3]
        )
        offset += count
    expected = np.concatenate(expected_base)
    local_to_base_error = float(np.max(np.abs(expected - samples.points_base)))
    scene_points, _ = transform_robot_surface_to_scene(samples, np.eye(4, dtype=np.float64))
    scene_round_trip_error = float(np.max(np.abs(scene_points - samples.points_scene)))
    return max(local_to_base_error, scene_round_trip_error)


@app.command("inspect-mesh")
def inspect_mesh(
    mesh: Path | None = typer.Option(None, "--mesh"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    object_id: str = typer.Option("primary", "--object-id"),
    output_json: Path | None = typer.Option(None, "--json"),
    csv_output: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Audit a mesh without repairing or rewriting it."""
    try:
        vertices, faces, provenance = _load_mesh_or_canonical(mesh, canonical, object_id)
        report = audit_mesh(
            vertices, faces, source_path=provenance.get("source_path"), source_provenance=provenance
        )
        if output_json is not None:
            report.write_json(output_json)
        if csv_output is not None:
            report.write_csv(csv_output)
        _print(report.as_dict())
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"mesh inspection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("sample-object")
def sample_object(
    mesh: Path | None = typer.Option(None, "--mesh"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    object_id: str = typer.Option("primary", "--object-id"),
    profile: str = typer.Option("paper_strict_area_uniform", "--profile"),
    output: Path = typer.Option(..., "--output"),
    report: Path | None = typer.Option(None, "--report"),
    scale: float = typer.Option(1.0, "--scale"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing derived artifact."),
) -> None:
    """Generate deterministic object-local barycentric surface anchors."""
    try:
        selected_profile = load_surface_profile(profile)
        if canonical is not None and mesh is None:
            track = _canonical_object(canonical, object_id)
            samples = sample_object_track(track, selected_profile)
            provenance = {"canonical": str(canonical), "object_id": track.object_id}
        else:
            vertices, faces, provenance = _load_mesh_or_canonical(mesh, canonical, object_id)
            from toporetarget.geometry.surface_sampling import sample_mesh_surface

            samples = sample_mesh_surface(
                vertices,
                faces,
                selected_profile,
                mesh_id=object_id,
                source_provenance=provenance,
                scale=scale,
            )
        samples.save(output, overwrite=force)
        payload = {"status": "pass", **samples.as_metadata(), "output": str(output)}
        if report is not None:
            write_json(payload, report)
        _print(payload)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"object sampling failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-samples")
def validate_samples(
    samples: Path = typer.Option(..., "--samples"),
    mesh: Path | None = typer.Option(None, "--mesh"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    object_id: str = typer.Option("primary", "--object-id"),
    report: Path | None = typer.Option(None, "--report"),
    csv_output: Path | None = typer.Option(None, "--csv"),
) -> None:
    """Validate exact count, barycentric anchors, and mesh identity."""
    try:
        vertices, faces, _ = _load_mesh_or_canonical(mesh, canonical, object_id)
        artifact = load_surface_artifact(samples, vertices=vertices, faces=faces)
        payload = artifact.validate(vertices, faces)
        payload["status"] = (
            "pass"
            if all(
                (
                    payload["count_exact"],
                    payload["face_indices_valid"],
                    payload["barycentric_nonnegative"],
                    payload["barycentric_sum_max_error"] <= 1e-12,
                    payload["point_reconstruction_max_error"] <= 1e-12,
                )
            )
            else "fail"
        )
        if report is not None:
            write_json(payload, report)
        if csv_output is not None:
            write_flat_csv(payload, csv_output)
        _print(payload)
        if payload["status"] != "pass":
            raise typer.Exit(code=1)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"sample validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _backend_for_mesh(
    vertices: np.ndarray, faces: np.ndarray, sign_mode: str
) -> ReferenceSignedDistanceBackend:
    return ReferenceSignedDistanceBackend(vertices, faces, sign_mode=sign_mode)


@app.command("sdf-query")
def sdf_query(
    points: Path = typer.Option(..., "--points"),
    mesh: Path | None = typer.Option(None, "--mesh"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    object_id: str = typer.Option("primary", "--object-id"),
    frame: int = typer.Option(0, "--frame", min=0),
    points_frame: str = typer.Option("local", "--points-frame"),
    sign_mode: str = typer.Option("strict", "--sign-mode"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Query closest points and positive-outside signed distance."""
    try:
        query_points = np.asarray(np.load(points), dtype=np.float64)
        if canonical is not None and mesh is None:
            track = _canonical_object(canonical, object_id)
            backend = _backend_for_mesh(track.mesh.vertices_local, track.mesh.faces, sign_mode)
            if points_frame == "scene":
                if frame >= track.pose_scene.pose_scene.shape[0]:
                    raise ValueError(f"frame {frame} is outside object pose track")
                result = backend.query_scene(query_points, track.pose_scene.pose_scene[frame])
            elif points_frame == "local":
                result = backend.query_local(query_points)
            else:
                raise ValueError("--points-frame must be local or scene")
        else:
            vertices, faces, _ = _load_mesh_or_canonical(mesh, canonical, object_id)
            backend = _backend_for_mesh(vertices, faces, sign_mode)
            result = backend.query_local(query_points)
        payload = {"backend": backend.describe(), "query": result.as_dict()}
        if output is not None:
            write_json(payload, output)
        _print(payload)
    except SignedDistanceError as exc:
        typer.echo(f"strict signed-distance query refused: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"SDF query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-sdf")
def validate_sdf(
    shape: str = typer.Option("sphere", "--shape"),
    backend: str = typer.Option("reference", "--backend"),
    canonical: Path | None = typer.Option(None, "--canonical"),
    object_id: str = typer.Option("primary", "--object-id"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    """Run analytic synthetic or bounded canonical SDF validation."""
    try:
        if canonical is None:
            strict = validate_synthetic_shape(shape, sign_mode="strict")
            if shape == "open_cube":
                unsigned = validate_synthetic_shape(shape, sign_mode="unsigned_only")
                payload = {
                    "status": "pass"
                    if strict["status"] == "expected_failure" and unsigned["status"] == "pass"
                    else "fail",
                    "shape": shape,
                    "strict": strict,
                    "unsigned_only": unsigned,
                }
            else:
                analytic = validate_analytic_shape(shape)
                payload = {
                    "status": "pass"
                    if strict["status"] == "pass" and analytic["status"] == "pass"
                    else "fail",
                    "shape": shape,
                    "analytic": analytic,
                    "reference_mesh": strict,
                }
        else:
            track = _canonical_object(canonical, object_id)
            sdf = ReferenceSignedDistanceBackend(
                track.mesh.vertices_local, track.mesh.faces, sign_mode="strict"
            )
            probes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
            payload = {
                "status": "pass",
                "backend": sdf.describe(),
                "query": sdf.query_local(probes).as_dict(),
            }
        if report is not None:
            write_json(payload, report)
        _print(payload)
        if payload.get("status") not in {"pass", "expected_failure"}:
            raise typer.Exit(code=1)
    except (OSError, ValueError, RuntimeError, SignedDistanceError) as exc:
        typer.echo(f"SDF validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("sample-robot")
def sample_robot(
    robot: str = typer.Option("artimano_rh", "--robot"),
    pose: str = typer.Option("neutral", "--pose"),
    profile: str = typer.Option("engineering_collision_32_per_geometry", "--profile"),
    seed: int = typer.Option(4, "--seed"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    output: Path = typer.Option(..., "--output"),
    report: Path | None = typer.Option(None, "--report"),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing derived artifact."),
) -> None:
    """Sample only URDF collision geometry in local/base/scene frames."""
    try:
        model = _robot_model(robot, asset_root)
        qpos = model.neutral_q.copy()
        if pose == "random":
            rng = np.random.Generator(np.random.PCG64(seed))
            lower, upper = model.joint_lower, model.joint_upper
            finite = np.isfinite(lower) & np.isfinite(upper)
            qpos[finite] = rng.uniform(lower[finite], upper[finite])
            qpos[~finite] += rng.normal(0.0, 0.2, size=np.count_nonzero(~finite))
        elif pose != "neutral":
            raise ValueError("pose must be neutral or random")
        selected = load_robot_surface_profile(profile)
        samples = sample_robot_collision_surface(model, qpos, selected)
        samples.save(output, overwrite=force)
        geometry = model.describe()["geometry"]
        payload = {
            "status": "pass",
            **samples.as_dict(),
            "pose": pose,
            "seed": seed,
            "missing_collision_links": geometry["missing_collision_links"],
            "tip_collision_links": geometry["fixed_tip_links_with_collision"],
            "qpos": qpos.tolist(),
            "fk_transform_error_m": _fk_transform_error(model, qpos, samples),
            "output": str(output),
        }
        if report is not None:
            write_json(payload, report)
        _print(payload)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"robot surface sampling failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("probe-collision")
def probe_collision(
    robot_samples: Path = typer.Option(..., "--robot-samples"),
    object_shape: str = typer.Option("cube", "--object-shape"),
    sign_mode: str = typer.Option("strict", "--sign-mode"),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    """Probe robot surface points against a synthetic object; no optimization."""
    try:
        with np.load(robot_samples, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            profile_values = metadata["profile"]
            from toporetarget.geometry.robot_surface import RobotSurfaceSamplingProfile

            profile = RobotSurfaceSamplingProfile(**profile_values)
            samples = RobotSurfaceSampleSet(
                robot_name=metadata["robot_name"],
                side=metadata["side"],
                profile=profile,
                geometry_ids=data["geometry_ids"],
                link_names=data["link_names"],
                geometry_types=data["geometry_types"],
                sample_ids=data["sample_ids"],
                points_local=data["points_local"],
                normals_local=data["normals_local"],
                points_base=data["points_base"],
                normals_base=data["normals_base"],
                points_scene=data["points_scene"],
                normals_scene=data["normals_scene"],
                geometry_metadata=metadata["geometry_metadata"],
                source_provenance=metadata["source_provenance"],
            )
        vertices, faces = make_synthetic_mesh(object_shape)
        sdf = ReferenceSignedDistanceBackend(vertices, faces, sign_mode=sign_mode)
        payload = {
            "status": "pass",
            "object_shape": object_shape,
            "probe": json_ready_probe(probe_robot_surface(samples, sdf, np.eye(4))),
            "no_final_query_set": True,
            "no_optimization": True,
        }
        if report is not None:
            write_json(payload, report)
        _print(payload)
    except (OSError, ValueError, KeyError, RuntimeError, SignedDistanceError) as exc:
        typer.echo(f"collision probe failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-object")
def visualize_object(
    canonical: Path = typer.Option(..., "--canonical"),
    samples: Path = typer.Option(..., "--samples"),
    object_id: str = typer.Option("primary", "--object-id"),
    frame: int = typer.Option(0, "--frame", min=0),
    output: Path = typer.Option(..., "--output"),
    show_normals: bool = typer.Option(False, "--show-normals"),
    show_ids: bool = typer.Option(False, "--show-ids"),
    show_object_frame: bool = typer.Option(False, "--show-object-frame"),
    show_scene_frame: bool = typer.Option(False, "--show-scene-frame"),
) -> None:
    """Render object mesh and fixed local sample identities for one frame."""
    try:
        track = _canonical_object(canonical, object_id)
        artifact = SurfaceSampleSet.load(
            samples, vertices=track.mesh.vertices_local, faces=track.mesh.faces
        )
        from toporetarget.geometry.visualization import render_object_samples

        render_object_samples(
            track,
            artifact,
            frame=frame,
            output=output,
            show_normals=show_normals,
            show_ids=show_ids,
            show_object_frame=show_object_frame,
            show_scene_frame=show_scene_frame,
        )
        _print(
            {
                "status": "pass",
                "output": str(output),
                "frame": frame,
                "sample_count": artifact.count,
            }
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"object visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-sdf")
def visualize_sdf(
    shape: str = typer.Option("sphere", "--shape"),
    slice_axis: str = typer.Option("z", "--slice-axis"),
    slice_value: float = typer.Option(0.0, "--slice-value"),
    output: Path = typer.Option(..., "--output"),
) -> None:
    """Render a signed/unsigned 2-D SDF slice with the zero contour."""
    try:
        vertices, faces = make_synthetic_mesh(shape)
        sdf = ReferenceSignedDistanceBackend(vertices, faces, sign_mode="strict")
        from toporetarget.geometry.visualization import render_sdf_slice

        render_sdf_slice(sdf, axis_name=slice_axis, slice_value=slice_value, output=output)
        _print(
            {
                "status": "pass",
                "output": str(output),
                "shape": shape,
                "slice_axis": slice_axis,
                "slice_value": slice_value,
            }
        )
    except (OSError, ValueError, RuntimeError, SignedDistanceError) as exc:
        typer.echo(f"SDF visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("visualize-robot-surface")
def visualize_robot_surface(
    robot: str = typer.Option("artimano_rh", "--robot"),
    pose: str = typer.Option("random", "--pose"),
    seed: int = typer.Option(4, "--seed"),
    profile: str = typer.Option("engineering_collision_32_per_geometry", "--profile"),
    asset_root: Path | None = typer.Option(None, "--asset-root"),
    samples: Path | None = typer.Option(None, "--samples"),
    output: Path = typer.Option(..., "--output"),
    show_sample_normals: bool = typer.Option(False, "--show-sample-normals"),
) -> None:
    """Render collision-only robot surface samples."""
    try:
        if samples is not None:
            with np.load(samples, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                from toporetarget.geometry.robot_surface import RobotSurfaceSamplingProfile

                selected = RobotSurfaceSamplingProfile(**metadata["profile"])
                sample_set = RobotSurfaceSampleSet(
                    robot_name=metadata["robot_name"],
                    side=metadata["side"],
                    profile=selected,
                    geometry_ids=data["geometry_ids"],
                    link_names=data["link_names"],
                    geometry_types=data["geometry_types"],
                    sample_ids=data["sample_ids"],
                    points_local=data["points_local"],
                    normals_local=data["normals_local"],
                    points_base=data["points_base"],
                    normals_base=data["normals_base"],
                    points_scene=data["points_scene"],
                    normals_scene=data["normals_scene"],
                    geometry_metadata=metadata["geometry_metadata"],
                    source_provenance=metadata["source_provenance"],
                )
        else:
            model = _robot_model(robot, asset_root)
            qpos = model.neutral_q
            if pose == "random":
                rng = np.random.Generator(np.random.PCG64(seed))
                lower, upper = model.joint_lower, model.joint_upper
                finite = np.isfinite(lower) & np.isfinite(upper)
                qpos[finite] = rng.uniform(lower[finite], upper[finite])
                qpos[~finite] += rng.normal(0.0, 0.2, size=np.count_nonzero(~finite))
            sample_set = sample_robot_collision_surface(
                model, qpos, load_robot_surface_profile(profile)
            )
        from toporetarget.geometry.visualization import render_robot_surface

        render_robot_surface(sample_set, output=output, show_normals=show_sample_normals)
        _print({"status": "pass", "output": str(output), "sample_count": sample_set.count})
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"robot visualization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("validate-temporal-reuse")
def validate_temporal_reuse_command(
    canonical: Path = typer.Option(..., "--canonical"),
    samples: Path = typer.Option(..., "--samples"),
    object_id: str = typer.Option("primary", "--object-id"),
    start_frame: int = typer.Option(0, "--start-frame", min=0),
    end_frame: int = typer.Option(60, "--end-frame", min=1),
    report: Path | None = typer.Option(None, "--report"),
) -> None:
    """Validate fixed sample identity over a bounded canonical clip."""
    try:
        track = _canonical_object(canonical, object_id)
        artifact = SurfaceSampleSet.load(
            samples, vertices=track.mesh.vertices_local, faces=track.mesh.faces
        )
        payload = validate_temporal_reuse(
            track, artifact, [start_frame, (start_frame + end_frame - 1) // 2, end_frame - 1]
        )
        if report is not None:
            write_json(payload, report)
        _print(payload)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        typer.echo(f"temporal sample validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
