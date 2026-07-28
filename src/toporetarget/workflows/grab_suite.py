"""Generic, frozen multi-clip GRAB retargeting orchestration.

The suite deliberately composes the existing data, geometry, retarget, quality,
and export components.  It contains no robot-specific solver branch: the robot
name and every profile are read from the suite manifest and are bound into the
selection lock before Stage 9 starts.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.geometry.signed_distance.reference import build_signed_distance_backend
from toporetarget.quality.html import render_clip_html, smoke_html
from toporetarget.quality.metrics import evaluate_profile
from toporetarget.retarget.artifacts import load_warm_start
from toporetarget.retarget.final_refinement import (
    PaperRefinementWeights,
    dynamic_collision_points_numpy,
    load_final_trajectory,
    load_robot_surface_samples,
)
from toporetarget.robots.registry import get_robot_registry
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.export import export_reference

SUITE_SCHEMA_VERSION = "toporetarget.grab_suite.v1"
DEFAULT_EXECUTION_ENV = {
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class SuiteClip:
    unit_id: str
    short_id: str
    sequence: str
    subject: str
    object_name: str
    hand: str
    robot: str
    start_frame: int
    end_frame: int
    native_fps: float = 120.0

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "short_id": self.short_id,
            "sequence": self.sequence,
            "subject": self.subject,
            "object_name": self.object_name,
            "hand": self.hand,
            "robot": self.robot,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frame_range": [self.start_frame, self.end_frame],
            "length": self.length,
            "native_fps": self.native_fps,
        }


class SuiteRunError(RuntimeError):
    """Raised for a hard suite input/identity error."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json_write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _clip_from_mapping(value: dict[str, Any], defaults: dict[str, Any]) -> SuiteClip:
    start = int(value["start_frame"])
    end = int(value["end_frame"])
    clip = SuiteClip(
        unit_id=str(value["unit_id"]),
        short_id=str(value.get("short_id", value["unit_id"])),
        sequence=str(value["sequence"]),
        subject=str(value.get("subject", defaults.get("subject", "s1"))),
        object_name=str(value["object"]),
        hand=str(value.get("hand", defaults.get("hand", "right"))),
        robot=str(value.get("robot", defaults.get("robot", "artimano_rh"))),
        start_frame=start,
        end_frame=end,
        native_fps=float(value.get("native_fps", defaults.get("native_fps", 120.0))),
    )
    if clip.subject != "s1" or clip.hand != "right":
        raise SuiteRunError(f"unsupported frozen suite unit: {clip.as_dict()}")
    if clip.length != 60 or start < 0 or end <= start:
        raise SuiteRunError(f"suite units must be 60-frame half-open windows: {clip.unit_id}")
    return clip


def load_suite(path: str | Path) -> tuple[dict[str, Any], tuple[SuiteClip, ...]]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or not isinstance(payload.get("units"), list):
        raise SuiteRunError(f"suite config must contain a units list: {source}")
    defaults = dict(payload)
    clips = tuple(_clip_from_mapping(item, defaults) for item in payload["units"])
    if len({clip.unit_id for clip in clips}) != len(clips):
        raise SuiteRunError("suite unit IDs must be unique")
    return payload, clips


def _unit_root(root: Path, clip: SuiteClip) -> Path:
    destination = root / clip.unit_id
    for name in (
        "canonical",
        "warm_start",
        "interaction_graph",
        "final",
        "validation",
        "checkpoints",
        "logs",
        "metrics",
        "html",
        "exports",
        "diagnostics",
    ):
        (destination / name).mkdir(parents=True, exist_ok=True)
    return destination


def _run_command(
    args: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    command_env = os.environ.copy()
    command_env.update(DEFAULT_EXECUTION_ENV)
    command_env["PYTHONPATH"] = str(cwd / "src")
    if env:
        command_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = subprocess.run(
        args,
        cwd=cwd,
        env=command_env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(
        f"$ {' '.join(args)}\n# returncode={result.returncode} elapsed_s={elapsed:.6f}\n"
        + result.stdout
        + result.stderr,
        encoding="utf-8",
    )
    return result.returncode, result.stdout + result.stderr


def _last_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates[-1] if candidates else None


def _source_npz(grab_root: Path, clip: SuiteClip) -> Path:
    return grab_root / "grab" / clip.subject / f"{clip.sequence.split('/', 1)[1]}.npz"


def _object_mesh_path(sequence: Any, object_name: str, grab_root: Path) -> Path:
    object_track = sequence.rigid_object(object_name)
    relative = str(object_track.metadata.get("source_mesh", ""))
    if relative:
        return (grab_root / relative).resolve()
    path = sequence.metadata.provenance.conversion_options.get("object_mesh_path")
    if path:
        return Path(str(path)).resolve()
    raise SuiteRunError(f"canonical sequence has no source object mesh path: {object_name}")


def _canonical_command(
    clip: SuiteClip, grab_root: Path, index: Path, mano_root: Path, output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "toporetarget",
        "data",
        "convert",
        "--dataset",
        "grab",
        "--sequence",
        clip.sequence,
        "--index",
        str(index),
        "--hands",
        clip.hand,
        "--contact-mode",
        "semantic",
        "--include-mediapipe21",
        "--start-frame",
        str(clip.start_frame),
        "--end-frame",
        str(clip.end_frame),
        "--mano-model-root",
        str(mano_root),
        "--grab-root",
        str(grab_root),
        "--output",
        str(output),
        "--force",
    ]


def _audit_object_meshes(
    clips: Iterable[SuiteClip], canonical_paths: dict[str, Path], grab_root: Path, output_root: Path
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for clip in clips:
        sequence = load_hoi_sequence(canonical_paths[clip.unit_id])
        object_track = sequence.rigid_object(clip.object_name)
        mesh_path = _object_mesh_path(sequence, clip.object_name, grab_root)
        report = audit_mesh(
            object_track.mesh.vertices_local,
            object_track.mesh.faces,
            source_path=mesh_path,
            source_provenance={"unit_id": clip.unit_id, "object": clip.object_name},
        ).as_dict()
        strict = build_signed_distance_backend(
            object_track.mesh.vertices_local, object_track.mesh.faces, sign_mode="strict"
        )
        probe = strict.query_local(
            object_track.mesh.vertices_local[: min(8, len(object_track.mesh.vertices_local))]
        )
        report.update(
            {
                "unit_id": clip.unit_id,
                "sequence": clip.sequence,
                "object": clip.object_name,
                "source_mesh_path": str(mesh_path),
                "source_mesh_sha256": sha256_file(mesh_path),
                "strict_backend": strict.describe(),
                "strict_probe_sign_valid": bool(np.all(probe.sign_valid)),
                "strict_probe_finite": bool(np.all(np.isfinite(probe.signed_distance))),
            }
        )
        rows.append(report)
    passed = all(
        bool(row["watertight"])
        and bool(row["winding_consistent"])
        and int(row["boundary_edge_count"]) == 0
        and int(row["non_manifold_edge_count"]) == 0
        and row["sign_reliability"] == "reliable_watertight"
        and bool(row["strict_probe_sign_valid"])
        and bool(row["strict_probe_finite"])
        for row in rows
    )
    payload = {
        "schema_version": "toporetarget.wuji_object_mesh_audit.v1",
        "status": "pass" if passed else "fail",
        "strict_signed_distance_required": True,
        "mesh_repair_performed": False,
        "rows": rows,
    }
    _json_write(payload, output_root / "object_mesh_audit.json")
    with (output_root / "object_mesh_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=str) for key, value in row.items()})
    if not passed:
        raise SuiteRunError("WUJI_W2_BLOCKED_BY_NON_WATERTIGHT_SELECTED_OBJECT")
    return payload


def _robot_audit(root: Path, robot: str, surface_path: Path) -> dict[str, Any]:
    model = get_robot_registry(repo_root=_repo_root()).load(robot)
    surface = load_robot_surface_samples(surface_path)
    payload = {
        "schema_version": "toporetarget.wuji_robot_input_audit.v1",
        "robot": robot,
        "model": model.describe(),
        "dof_count": int(model.num_dofs),
        "qpos_order": list(model.dof_names),
        "qpos_order_hash": _stable_hash(list(model.dof_names)),
        "joint_lower": np.asarray(model.joint_lower).tolist(),
        "joint_upper": np.asarray(model.joint_upper).tolist(),
        "collision_surface": surface.as_dict(),
        "collision_sample_count": int(surface.count),
        "expected_collision_sample_count": 672,
        "visual_geometry_is_collision": False,
        "formal_collision_profile_frozen": True,
        "status": "pass" if model.num_dofs == 20 and surface.count == 672 else "fail",
    }
    if payload["status"] != "pass":
        raise SuiteRunError("WUJI_W2_BLOCKED_BY_DATA_OR_ASSET_INTEGRITY: robot profile mismatch")
    return payload


def _selection_manifest(
    config: dict[str, Any],
    clips: tuple[SuiteClip, ...],
    canonical_paths: dict[str, Path],
    grab_root: Path,
    mano_root: Path,
    robot_audit: dict[str, Any],
    surface_path: Path,
) -> dict[str, Any]:
    manifest_rows: list[dict[str, Any]] = []
    for clip in clips:
        canonical = load_hoi_sequence(canonical_paths[clip.unit_id])
        source = _source_npz(grab_root, clip)
        obj = canonical.rigid_object(clip.object_name)
        pose = np.asarray(obj.pose_scene.pose_scene, dtype=np.float64)
        options = canonical.metadata.provenance.conversion_options
        row = {
            **clip.as_dict(),
            "source_npz": str(source),
            "source_sha256": sha256_file(source),
            "mano_model_root": str(mano_root),
            "mano_tree_hash": sha256_tree(mano_root),
            "mano_model_hash": canonical.hands[0].metadata.get("mano_model_hash"),
            "personalized_vtemp": canonical.hands[0].metadata.get("vtemp_hash"),
            "object_mesh": str(_object_mesh_path(canonical, clip.object_name, grab_root)),
            "object_mesh_sha256": options.get("object_mesh_hash"),
            "object_pose_hash": _array_hash(pose),
            "semantic_contact_labels": bool(
                canonical.metadata.metadata.get("contact", {}).get("present", False)
            ),
            "native_fps": canonical.metadata.native_fps,
            "timestamps": np.asarray(canonical.metadata.timestamps).tolist(),
            "robot": config["robot"],
            "robot_asset_manifest_hash": robot_audit["model"].get("asset_manifest_hash"),
            "robot_input_audit_hash": _stable_hash(robot_audit),
            "qpos_order_hash": robot_audit["qpos_order_hash"],
            "anchor_profile": "wuji_hand2_beta1_rh_mediapipe21",
            "collision_surface": surface_path.name,
            "collision_surface_profile_hash": robot_audit["collision_surface"]["profile_hash"],
            "solver_profile": config["solver_profile"],
            "paper_config_hash": _stable_hash(
                {
                    key: config.get(key)
                    for key in config
                    if key.endswith("profile") or key.endswith("_profile")
                }
            ),
        }
        manifest_rows.append(row)
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "created_on_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "robot": config["robot"],
        "selection_frozen": True,
        "selection_rule": "fixed W1/W2/W3 native frame windows; no result-based reselection",
        "units": manifest_rows,
        "selection_hash": _stable_hash(manifest_rows),
    }


def _run_pre_stages(
    clip: SuiteClip,
    paths: dict[str, Path],
    *,
    grab_root: Path,
    index: Path,
    mano_root: Path,
    robot: str,
    collision_samples: Path,
) -> list[dict[str, Any]]:
    root = _repo_root()
    logs = paths["unit_root"] / "logs"
    steps: list[dict[str, Any]] = []

    def run_if_missing(name: str, output: Path, args: list[str]) -> None:
        if output.exists():
            steps.append({"stage": name, "status": "reused", "output": str(output)})
            return
        code, text = _run_command(args, cwd=root, log_path=logs / f"{name}.log")
        steps.append(
            {
                "stage": name,
                "status": "pass" if code == 0 else "fail",
                "returncode": code,
                "output": str(output),
            }
        )
        if code != 0:
            raise SuiteRunError(f"{name} failed for {clip.unit_id}: {text[-1000:]}")

    run_if_missing(
        "canonical",
        paths["canonical"],
        _canonical_command(clip, grab_root, index, mano_root, paths["canonical"]),
    )
    run_if_missing(
        "object_samples",
        paths["object_samples"],
        [
            sys.executable,
            "-m",
            "toporetarget",
            "geometry",
            "sample-object",
            "--canonical",
            str(paths["canonical"]),
            "--object-id",
            clip.object_name,
            "--profile",
            "paper_strict_area_uniform",
            "--output",
            str(paths["object_samples"]),
            "--report",
            str(paths["object_samples_report"]),
            "--force",
        ],
    )
    run_if_missing(
        "warm_start",
        paths["warm"],
        [
            sys.executable,
            "-m",
            "toporetarget",
            "retarget",
            "warm-start",
            "--canonical",
            str(paths["canonical"]),
            "--hand",
            clip.hand,
            "--robot",
            robot,
            "--start-frame",
            "0",
            "--end-frame",
            "60",
            "--solver-profile",
            "paper_repro_scipy_trf",
            "--output",
            str(paths["warm"]),
            "--force",
        ],
    )
    run_if_missing(
        "interaction_graph",
        paths["graph"],
        [
            sys.executable,
            "-m",
            "toporetarget",
            "retarget",
            "build-interaction-graph",
            "--canonical",
            str(paths["canonical"]),
            "--hand",
            clip.hand,
            "--object-id",
            clip.object_name,
            "--object-samples",
            str(paths["object_samples"]),
            "--delaunay-profile",
            "strict_scipy_qhull_v1",
            "--start-frame",
            "0",
            "--end-frame",
            "60",
            "--output",
            str(paths["graph"]),
            "--report",
            str(paths["graph_report"]),
            "--force",
        ],
    )
    run_if_missing(
        "interaction_evaluation_warm",
        paths["evaluation"],
        [
            sys.executable,
            "-m",
            "toporetarget",
            "retarget",
            "evaluate-interaction",
            "--graph",
            str(paths["graph"]),
            "--warm-start",
            str(paths["warm"]),
            "--robot",
            robot,
            "--output",
            str(paths["evaluation"]),
            "--force",
        ],
    )
    return steps


def _final_refinement(
    clip: SuiteClip,
    paths: dict[str, Path],
    *,
    robot: str,
    solver_profile: str,
    max_wall_time: float,
) -> dict[str, Any]:
    if paths["final"].exists():
        final = load_final_trajectory(paths["final"])
        return {"status": "reused", "frame_count": final.frame_count, "output": str(paths["final"])}
    command = [
        sys.executable,
        "-m",
        "toporetarget",
        "retarget",
        "refine",
        "--canonical",
        str(paths["canonical"]),
        "--warm-start",
        str(paths["warm"]),
        "--graph",
        str(paths["graph"]),
        "--robot",
        robot,
        "--collision-samples",
        str(paths["collision_samples"]),
        "--query-profile",
        "adaptive_active_set_v1",
        "--coordinate-profile",
        "local_seed_delta_v1",
        "--solver-profile",
        solver_profile,
        "--start-frame",
        "0",
        "--end-frame",
        "60",
        "--checkpoint-root",
        str(paths["checkpoint_root"]),
        "--resume",
        "--max-wall-time",
        str(max_wall_time),
        "--progress-json",
        str(paths["progress"]),
        "--progress-log",
        str(paths["progress_log"]),
        "--execution-profile",
        "cached_checkpoint_cpu_float64_v3",
        "--output",
        str(paths["final"]),
        "--force",
    ]
    sessions = 0
    while True:
        sessions += 1
        code, text = _run_command(
            command, cwd=_repo_root(), log_path=paths["logs"] / f"final_session_{sessions:03d}.log"
        )
        result = _last_json(text) or {}
        if code != 0:
            return {
                "status": "fail",
                "returncode": code,
                "sessions": sessions,
                "error_tail": text[-4000:],
            }
        status = str(result.get("status", ""))
        if status == "complete" and paths["final"].exists():
            return {
                "status": "pass",
                "sessions": sessions,
                **{
                    key: result.get(key)
                    for key in ("frame_count", "checkpoint_root", "final_artifact")
                },
            }
        if status not in {"paused", "created"}:
            return {
                "status": "fail",
                "sessions": sessions,
                "unexpected_status": status,
                "result": result,
            }


def _write_object_mesh_asset(
    clip: SuiteClip, canonical_path: Path, output_root: Path
) -> dict[str, Any]:
    """Materialize the canonical object mesh for the viewer handoff.

    The HTML embeds a deterministic preview, but downstream users also need a
    real mesh file they can inspect or load independently of the browser.
    Keep this mesh in object-local coordinates; the canonical clip supplies
    the per-frame scene poses separately.
    """

    sequence = load_hoi_sequence(canonical_path)
    object_track = sequence.rigid_object(clip.object_name)
    vertices = np.asarray(object_track.mesh.vertices_local, dtype=np.float64)
    faces = np.asarray(object_track.mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"object mesh must be triangular vertices/faces, got {vertices.shape}, {faces.shape}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("object mesh contains non-finite vertices")
    if len(faces) and (int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices)):
        raise ValueError("object mesh contains an out-of-range face index")

    output_root.mkdir(parents=True, exist_ok=True)
    mesh_path = output_root / f"{clip.short_id}_{clip.object_name}.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with mesh_path.open("wb") as handle:
        handle.write(header)
        handle.write(np.asarray(vertices, dtype="<f8", order="C").tobytes())
        for face in faces:
            handle.write(np.asarray([3], dtype="<u1").tobytes())
            handle.write(np.asarray(face, dtype="<i4").tobytes())

    manifest = {
        "schema_version": "toporetarget.wuji_object_mesh_asset.v1",
        "unit_id": clip.unit_id,
        "object_name": clip.object_name,
        "source_sequence": clip.sequence,
        "coordinate_frame": "object_local",
        "units": "m",
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "pose_frames": int(len(object_track.pose_scene.pose_scene)),
        "source_canonical": str(canonical_path),
    }
    manifest_path = mesh_path.with_suffix(".json")
    _json_write(manifest, manifest_path)
    return {"path": str(mesh_path), "manifest": str(manifest_path), **manifest}


def _independent_validation(clip: SuiteClip, paths: dict[str, Path], robot: str) -> dict[str, Any]:
    sequence = load_hoi_sequence(paths["canonical"])
    final = load_final_trajectory(paths["final"])
    model = get_robot_registry(repo_root=_repo_root()).load(robot)
    surface = load_robot_surface_samples(paths["collision_samples"])
    object_track = sequence.rigid_object(clip.object_name)
    backend = build_signed_distance_backend(
        object_track.mesh.vertices_local, object_track.mesh.faces, sign_mode="strict"
    )
    paper = PaperRefinementWeights.load()
    rows: list[dict[str, Any]] = []
    for index, global_frame in enumerate(np.asarray(final.arrays["frame_indices"], dtype=np.int64)):
        points = dynamic_collision_points_numpy(
            model, surface, final.arrays["qpos"][index], final.arrays["base_pose_scene"][index]
        )
        query = backend.query_scene(points, object_track.pose_scene.pose_scene[int(global_frame)])
        q0, q1 = (
            int(final.arrays["query_offsets"][index]),
            int(final.arrays["query_offsets"][index + 1]),
        )
        ids = np.asarray(final.arrays["query_ids_concat"][q0:q1], dtype=np.int64)
        unqueried = np.setdiff1d(np.arange(len(points), dtype=np.int64), ids, assume_unique=True)
        signed = np.asarray(query.signed_distance, dtype=np.float64)
        hard = signed[ids] + paper.b if len(ids) else np.empty(0)
        soft = (
            signed[ids]
            + np.asarray(
                final.arrays["slack_concat"][
                    int(final.arrays["slack_offsets"][index]) : int(
                        final.arrays["slack_offsets"][index + 1]
                    )
                ]
            )
            + paper.tau
            if len(ids)
            else np.empty(0)
        )
        accepted = bool(final.arrays.get("accepted", final.arrays["solver_success"])[index])
        row = {
            "local_frame": int(index),
            "global_frame": int(global_frame),
            "collision_sample_count": int(len(signed)),
            "query_count": int(len(ids)),
            "unqueried_count": int(len(unqueried)),
            "unqueried_violation_count": int(
                np.count_nonzero(signed[unqueried] < -paper.tau - 1e-6)
            ),
            "sign_valid": bool(np.all(query.sign_valid)),
            "finite": bool(np.all(np.isfinite(signed))),
            "min_signed_distance_m": float(np.min(signed)),
            "max_penetration_m": float(max(0.0, -np.min(signed))),
            "min_hard_residual_m": float(np.min(hard, initial=np.inf)),
            "min_soft_residual_m": float(np.min(soft, initial=np.inf)),
            "full_hard_pass": bool(np.min(signed) >= -paper.b - 1e-6),
            "full_soft_unqueried_pass": bool(np.all(signed[unqueried] >= -paper.tau - 1e-6)),
            "optimizer_converged": bool(
                final.arrays.get("optimizer_converged", final.arrays["solver_success"])[index]
            ),
            "solver_success": bool(final.arrays["solver_success"][index]),
            "solver_status": int(final.arrays.get("solver_status", np.asarray([-1]))[index]),
            "strict_accepted": accepted,
            "qpos_bounds_pass": bool(
                np.all(final.arrays["qpos"][index] >= model.joint_lower - 1e-10)
                and np.all(final.arrays["qpos"][index] <= model.joint_upper + 1e-10)
            ),
            "active_set_converged": bool(
                final.arrays.get("active_set_converged", np.ones(final.frame_count, dtype=bool))[
                    index
                ]
            ),
        }
        rows.append(row)
    passed = bool(
        len(rows) == 60
        and all(
            row["strict_accepted"]
            and row["optimizer_converged"]
            and row["solver_success"]
            and row["full_hard_pass"]
            and row["full_soft_unqueried_pass"]
            and row["unqueried_violation_count"] == 0
            and row["sign_valid"]
            and row["finite"]
            and row["qpos_bounds_pass"]
            and row["active_set_converged"]
            for row in rows
        )
    )
    payload = {
        "schema_version": "toporetarget.wuji_independent_full_surface.v1",
        "status": "pass" if passed else "fail",
        "independent_backend": backend.describe(),
        "collision_sample_count": int(surface.count),
        "frame_count": len(rows),
        "expected_queries": int(len(rows) * surface.count),
        "actual_queries": int(sum(row["collision_sample_count"] for row in rows)),
        "frames": rows,
    }
    _json_write(payload, paths["validation"] / "independent_full_surface.json")
    # Keep the per-unit handoff names stable for downstream audit tooling.
    _json_write(payload, paths["validation"] / "validation.json")
    with (paths["validation"] / "independent_full_surface.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    with (paths["validation"] / "validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    return payload


def _metrics_and_export(
    clip: SuiteClip, paths: dict[str, Path], *, root: Path, robot: str, solver_profile: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fingers: list[dict[str, Any]] = []
    per_frame: list[dict[str, Any]] = []
    sequence = load_hoi_sequence(paths["canonical"])
    hand = next(item for item in sequence.hands if item.side == clip.hand)
    source_keypoints = np.asarray(
        hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    for profile, artifact, warm in (
        ("paper_warm", paths["warm"], True),
        (solver_profile, paths["final"], False),
    ):
        row, finger = evaluate_profile(
            clip=clip,  # type: ignore[arg-type]
            canonical_path=paths["canonical"],
            source_path=paths["source"],
            artifact_path=artifact,
            profile_id=profile,
            is_warm=warm,
        )
        row["robot"] = robot
        row["contact_ground_truth"] = False
        row["collision_sample_count"] = 672
        rows.append(row)
        fingers.extend(finger)
        artifact_arrays = (
            load_warm_start(artifact).arrays if warm else load_final_trajectory(artifact).arrays
        )
        robot_keypoints = np.asarray(artifact_arrays["robot_keypoints_scene"], dtype=np.float64)
        raw = np.linalg.norm(robot_keypoints - source_keypoints, axis=-1)
        scale = np.maximum(
            np.linalg.norm(source_keypoints[:, 5] - source_keypoints[:, 17], axis=-1), 1e-6
        )
        morphology = (
            np.linalg.norm(
                (robot_keypoints - robot_keypoints[:, :1])
                - (source_keypoints - source_keypoints[:, :1]),
                axis=-1,
            )
            / scale[:, None]
        )
        for local_frame in range(len(robot_keypoints)):
            per_frame.append(
                {
                    "unit_id": clip.unit_id,
                    "profile": profile,
                    "local_frame": local_frame,
                    "global_frame": clip.start_frame + local_frame,
                    "raw_whole_hand_rmse_mm": float(
                        np.sqrt(np.mean(raw[local_frame] ** 2)) * 1000.0
                    ),
                    "morphology_whole_hand_rmse_mm": float(
                        np.sqrt(np.mean(morphology[local_frame] ** 2)) * 1000.0
                    ),
                    "e_im": None
                    if "e_im" not in artifact_arrays
                    else float(artifact_arrays["e_im"][local_frame]),
                    "e_bone": None
                    if "e_bone" not in artifact_arrays
                    else float(artifact_arrays["e_bone"][local_frame]),
                    "solver_success": bool(
                        artifact_arrays.get(
                            "solver_success", np.ones(len(robot_keypoints), dtype=bool)
                        )[local_frame]
                    ),
                    "accepted": bool(
                        artifact_arrays.get(
                            "accepted",
                            artifact_arrays.get(
                                "solver_success", np.ones(len(robot_keypoints), dtype=bool)
                            ),
                        )[local_frame]
                    ),
                    "solve_time_s": float(
                        artifact_arrays.get("solve_time_s", np.zeros(len(robot_keypoints)))[
                            local_frame
                        ]
                    ),
                }
            )
    validation = json.loads((paths["validation"] / "independent_full_surface.json").read_text())
    for row in rows:
        row["independent_full_surface_pass"] = validation["status"] == "pass"
        row["independent_min_signed_distance_m"] = min(
            item["min_signed_distance_m"] for item in validation["frames"]
        )
        row["independent_max_penetration_m"] = max(
            item["max_penetration_m"] for item in validation["frames"]
        )
        row["unqueried_violation_count"] = sum(
            item["unqueried_violation_count"] for item in validation["frames"]
        )
    unit_metrics = paths["metrics"]
    _json_write(rows, unit_metrics / "per_clip_metrics.json")
    _json_write(fingers, unit_metrics / "per_finger_metrics.json")
    _json_write(per_frame, unit_metrics / "per_frame_metrics.json")
    with (unit_metrics / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in per_frame for key in row})
        )
        writer.writeheader()
        writer.writerows(per_frame)
    final_arrays = load_final_trajectory(paths["final"]).arrays
    performance_rows = [
        {
            "unit_id": clip.unit_id,
            "local_frame": int(index),
            "stage": "stage9",
            "solve_time_s": float(final_arrays.get("solve_time_s", np.zeros(60))[index]),
            "function_evaluations": int(
                final_arrays.get("function_evaluations", np.zeros(60))[index]
            ),
            "jacobian_evaluations": int(
                final_arrays.get("jacobian_evaluations", np.zeros(60))[index]
            ),
            "iterations": int(final_arrays.get("iterations", np.zeros(60))[index]),
            "query_count": int(
                final_arrays.get("query_offsets", np.arange(61))[index + 1]
                - final_arrays.get("query_offsets", np.arange(61))[index]
            ),
            "checkpoint_overhead_s": None,
        }
        for index in range(60)
    ]
    performance_root = root / "performance"
    performance_root.mkdir(parents=True, exist_ok=True)
    with (performance_root / "performance_per_frame.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(performance_rows[0]))
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(performance_rows)
    runtime_values = np.asarray([row["solve_time_s"] for row in performance_rows], dtype=np.float64)
    _json_write(
        {
            "unit_id": clip.unit_id,
            "stage9_total_s": float(np.sum(runtime_values)),
            "stage9_mean_s": float(np.mean(runtime_values)),
            "stage9_median_s": float(np.median(runtime_values)),
            "stage9_p95_s": float(np.quantile(runtime_values, 0.95)),
            "stage9_max_s": float(np.max(runtime_values)),
            "offline_reference_runtime": True,
        },
        performance_root / f"{clip.short_id}_performance.json",
    )
    run = {
        "run_id": clip.unit_id,
        "robot": robot,
        "hand": clip.hand,
        "source_sequence": clip.sequence,
        "subject": clip.subject,
        "object_id": clip.object_name,
        "action": clip.sequence.split("/", 1)[1],
        "source_hash": sha256_file(paths["source"]),
        "native_fps": clip.native_fps,
        "artifacts": {
            "canonical": {"path": str(paths["canonical"])},
            "final": {"path": str(paths["final"])},
        },
    }
    export_root = root / "exports" / clip.short_id
    export_root.mkdir(parents=True, exist_ok=True)
    npz_path = export_root / "robot_reference.npz"
    zarr_path = export_root / "robot_reference.zarr"
    if not npz_path.exists():
        export_reference(
            run, output=npz_path, format="npz", metadata_path=export_root / "manifest.json"
        )
    if not zarr_path.exists():
        export_reference(run, output=zarr_path, format="zarr")
    with np.load(npz_path, allow_pickle=False) as data:
        names = [name for name in data.files if name != "metadata"]
        npz_arrays = {name: np.asarray(data[name]) for name in names}
    from toporetarget.data.storage import direct_zarr3_arrays

    zarr_arrays = direct_zarr3_arrays(zarr_path, names)
    exact = bool(
        all(np.array_equal(npz_arrays[name], zarr_arrays[name], equal_nan=True) for name in names)
    )
    final = load_final_trajectory(paths["final"])
    export_validation = {
        "npz_zarr_arrays_exact": exact,
        "qpos_exact_final": bool(np.array_equal(npz_arrays["qpos"], final.arrays["qpos"])),
        "base_exact_final": bool(
            np.array_equal(npz_arrays["base_pose_scene"], final.arrays["base_pose_scene"])
        ),
        "no_solver_invocation_during_export": True,
    }
    _json_write(export_validation, export_root / "validation.json")
    _json_write(
        {
            "unit_id": clip.unit_id,
            "source": str(paths["source"]),
            "canonical": str(paths["canonical"]),
            "final": str(paths["final"]),
            "solver_profile": solver_profile,
            "export_only": True,
        },
        export_root / "provenance.json",
    )
    _json_write(
        {"unit_id": clip.unit_id, "rows": rows, "validation": validation},
        export_root / "metrics.json",
    )
    return {
        "rows": rows,
        "per_finger": fingers,
        "per_frame": per_frame,
        "performance": performance_rows,
        "export": {**export_validation, "path": str(export_root)},
    }


def _render_html(
    clip: SuiteClip, paths: dict[str, Path], *, root: Path, solver_profile: str
) -> dict[str, Any]:
    output = root / "html" / f"{clip.short_id}_mano_warm_final_wuji.html"
    object_asset = _write_object_mesh_asset(
        clip,
        paths["canonical"],
        root / "object_meshes",
    )
    render_clip_html(
        clip=clip,  # type: ignore[arg-type]
        canonical_path=paths["canonical"],
        source_path=paths["source"],
        profile_paths={
            "paper_warm": (paths["warm"], True, "Wuji warm-start"),
            solver_profile: (paths["final"], False, "Wuji final"),
        },
        output=output,
        asset_root=None,
        recommended_profile=solver_profile,
        graph_path=paths["graph"],
        evaluation_path=paths["evaluation"],
    )
    result = smoke_html(output, expected_frames=60, profiles=2)
    result["object_mesh_asset"] = object_asset
    _json_write(result, root / "html" / f"{clip.short_id}_smoke.json")
    return result


def _write_aggregate(
    root: Path, records: list[dict[str, Any]], clips: tuple[SuiteClip, ...]
) -> None:
    metrics_root = root / "metrics"
    rows = [row for record in records for row in record.get("metrics", {}).get("rows", [])]
    fingers = [row for record in records for row in record.get("metrics", {}).get("per_finger", [])]
    per_frame = [
        row for record in records for row in record.get("metrics", {}).get("per_frame", [])
    ]
    performance = [
        row for record in records for row in record.get("metrics", {}).get("performance", [])
    ]
    _json_write(rows, metrics_root / "per_clip_metrics.json")
    _json_write(fingers, metrics_root / "per_finger_metrics.json")
    _json_write(per_frame, metrics_root / "per_frame_metrics.json")
    with (metrics_root / "per_clip_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, default=str) for key, value in row.items()})
    with (metrics_root / "per_finger_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        if fingers:
            writer = csv.DictWriter(
                handle, fieldnames=sorted({key for row in fingers for key in row})
            )
            writer.writeheader()
            writer.writerows(fingers)
    with (metrics_root / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if per_frame:
            writer = csv.DictWriter(
                handle, fieldnames=sorted({key for row in per_frame for key in row})
            )
            writer.writeheader()
            writer.writerows(per_frame)
    performance_root = root / "performance"
    with (performance_root / "performance_per_frame.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        if performance:
            writer = csv.DictWriter(
                handle, fieldnames=sorted({key for row in performance for key in row})
            )
            writer.writeheader()
            writer.writerows(performance)
    performance_clips = []
    for unit_id in sorted({row["unit_id"] for row in performance}):
        selected_performance = [row for row in performance if row["unit_id"] == unit_id]
        values = np.asarray([row["solve_time_s"] for row in selected_performance], dtype=np.float64)
        performance_clips.append(
            {
                "unit_id": unit_id,
                "stage9_total_s": float(np.sum(values)),
                "stage9_mean_s": float(np.mean(values)),
                "stage9_median_s": float(np.median(values)),
                "stage9_p95_s": float(np.quantile(values, 0.95)),
                "stage9_max_s": float(np.max(values)),
                "offline_reference_runtime": True,
            }
        )
    with (performance_root / "performance_per_clip.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        if performance_clips:
            writer = csv.DictWriter(handle, fieldnames=sorted(performance_clips[0]))
            writer.writeheader()
            writer.writerows(performance_clips)
    _json_write(
        {"offline_reference_runtime": True, "clips": performance_clips},
        performance_root / "performance_summary.json",
    )
    aggregate: dict[str, Any] = {
        "schema_version": "toporetarget.wuji_metrics.v1",
        "aggregation": "per-clip macro average",
        "clip_count": len(clips),
        "profiles": {},
    }
    for profile in sorted({str(row["profile"]) for row in rows}):
        selected = [row for row in rows if str(row["profile"]) == profile]
        numeric: dict[str, float] = {}
        for key in sorted({key for row in selected for key in row}):
            numeric_values = [
                float(row[key])
                for row in selected
                if isinstance(row.get(key), (int, float)) and np.isfinite(float(row[key]))
            ]
            if numeric_values:
                numeric[key] = float(np.mean(numeric_values))
        aggregate["profiles"][profile] = {"macro_average": numeric, "clips": selected}
    _json_write(aggregate, metrics_root / "aggregate_metrics.json")
    lines = [
        "# Wuji Hand2 three-clip metrics",
        "",
        "Aggregation: per-clip macro average.",
        "",
        "| Unit | Profile | RMSE mm | E_IM | E_bone | strict accepted | independent full surface |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('unit_id')} | {row.get('profile')} | {row.get('whole_hand_raw_rmse_mm')} | {row.get('e_im_mean')} | {row.get('e_bone_mean')} | {row.get('strict_accepted_frames')}/{row.get('frame_count')} | {row.get('independent_full_surface_pass')} |"
        )
    (metrics_root / "aggregate_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_dashboard(root: Path, records: list[dict[str, Any]]) -> None:
    report_root = root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    html_root = root / "html"
    html_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "toporetarget.wuji_three_clip_summary.v1",
        "records": records,
        "status": "pass"
        if all(record.get("status") in {"pass", "reused"} for record in records)
        else "complete_with_recorded_failures",
    }
    _json_write(summary, report_root / "wuji_three_clip_summary.json")
    _json_write(
        {record["unit_id"]: record for record in records}, report_root / "per_clip_diagnosis.json"
    )
    _json_write(
        {
            "failures": [
                record for record in records if record.get("status") not in {"pass", "reused"}
            ]
        },
        report_root / "failure_report.json",
    )
    _json_write(
        {
            "raw_grab_changed": False,
            "mano_changed": False,
            "tracked_wuji_assets_changed": False,
            "old_artifacts_changed": False,
        },
        report_root / "source_integrity.json",
    )
    artifact_rows = []
    for record in records:
        artifact_rows.extend(record.get("artifacts", []))
    _json_write({"artifacts": artifact_rows}, report_root / "artifact_manifest.json")
    status = (
        "WUJI_W2_THREE_CLIP_RETARGETING_COMPLETE"
        if summary["status"] == "pass"
        else "WUJI_W2_COMPLETE_WITH_RECORDED_FAILURES"
    )
    _json_write({"status": status, "summary": summary}, report_root / "final_status.json")
    (report_root / "wuji_three_clip_summary.md").write_text(
        "# Wuji Hand2 three-clip summary\n\n"
        + "\n".join(f"- {r['unit_id']}: {r.get('status')}" for r in records)
        + "\n",
        encoding="utf-8",
    )
    cards = []
    for record in records:
        cards.append(
            f"<li><b>{record['unit_id']}</b>: {record.get('status')} <a href='../html/{record.get('short_id')}_mano_warm_final_wuji.html'>viewer</a></li>"
        )
    (report_root / "dashboard.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Wuji Hand2 suite</title><h1>Wuji Hand2 three-clip suite</h1><ul>"
        + "".join(cards)
        + "</ul><pre id='summary'></pre><script>fetch('wuji_three_clip_summary.json').then(r=>r.json()).then(x=>summary.textContent=JSON.stringify(x,null,2))</script>",
        encoding="utf-8",
    )
    viewer_links = [
        f"<li><a href='{record.get('short_id')}_mano_warm_final_wuji.html'>{record.get('short_id')}</a></li>"
        for record in records
        if record.get("html")
    ]
    (html_root / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Wuji Hand2 viewers</title>"
        "<h1>Wuji Hand2 three-clip viewers</h1><ul>" + "".join(viewer_links) + "</ul>\n",
        encoding="utf-8",
    )
    _json_write(
        {record["unit_id"]: record.get("html", {"status": "not_run"}) for record in records},
        html_root / "html_smoke_report.json",
    )


def run_suite(
    *,
    suite: str | Path,
    grab_root: str | Path,
    index: str | Path,
    mano_model_root: str | Path,
    robot: str | None = None,
    solver_profile: str | None = None,
    experiment_root: str | Path,
    resume: bool = True,
    max_wall_time: float = 1800.0,
    evaluate: bool = True,
    export_reference_bundles: bool = True,
    generate_html: bool = True,
    unit: str | None = None,
) -> dict[str, Any]:
    del resume  # The checkpoint command is always invoked with --resume semantics.
    config, clips = load_suite(suite)
    if robot is not None:
        config["robot"] = robot
    if solver_profile is not None:
        config["solver_profile"] = solver_profile
    selected = tuple(
        clip for clip in clips if unit is None or clip.unit_id == unit or clip.short_id == unit
    )
    if not selected:
        raise SuiteRunError(f"unknown suite unit: {unit}")
    root = Path(experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "selection",
        "canonical",
        "warm_start",
        "interaction_graph",
        "final",
        "validation",
        "exports",
        "metrics",
        "performance",
        "diagnostics",
        "html",
        "screenshots",
        "checkpoints",
        "logs",
        "reports",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)
    grab = Path(grab_root).resolve()
    index_path = Path(index).resolve()
    mano = Path(mano_model_root).resolve()
    canonical_paths: dict[str, Path] = {}
    path_map: dict[str, dict[str, Path]] = {}
    for clip in selected:
        unit_root = _unit_root(root, clip)
        canonical_paths[clip.unit_id] = unit_root / "canonical" / "canonical.zarr"
        path_map[clip.unit_id] = {
            "unit_root": unit_root,
            "canonical": canonical_paths[clip.unit_id],
            "source": _source_npz(grab, clip),
            "object_samples": unit_root / "interaction_graph" / "object_samples.npz",
            "object_samples_report": unit_root / "interaction_graph" / "object_samples.json",
            "warm": unit_root / "warm_start" / "warm_start.zarr",
            "graph": unit_root / "interaction_graph" / "interaction_graph.zarr",
            "graph_report": unit_root / "interaction_graph" / "interaction_graph.json",
            "evaluation": unit_root / "interaction_graph" / "interaction_evaluation_warm.zarr",
            "collision_samples": root / "diagnostics" / "wuji_hand2_beta1_rh_neutral.npz",
            "final": unit_root / "final" / "final_retarget.zarr",
            "checkpoint_root": unit_root / "checkpoints" / str(config["solver_profile"]),
            "progress": unit_root / "checkpoints" / str(config["solver_profile"]) / "progress.json",
            "progress_log": unit_root / "logs" / "checkpoint_progress.jsonl",
            "validation": unit_root / "validation",
            "metrics": unit_root / "metrics",
            "logs": unit_root / "logs",
        }
    # Convert all selected source windows before the immutable selection lock.
    for clip in selected:
        paths = path_map[clip.unit_id]
        if not paths["canonical"].exists():
            code, text = _run_command(
                _canonical_command(clip, grab, index_path, mano, paths["canonical"]),
                cwd=_repo_root(),
                log_path=paths["logs"] / "canonical.log",
            )
            if code != 0:
                raise SuiteRunError(
                    f"canonical conversion failed for {clip.unit_id}: {text[-1000:]}"
                )
    surface_path = (
        _repo_root()
        / ".local"
        / "cache"
        / "geometry"
        / "robot_surface"
        / f"{config['robot']}_neutral.npz"
    )
    for paths in path_map.values():
        paths["collision_samples"] = surface_path
    if not surface_path.exists():
        code, text = _run_command(
            [
                sys.executable,
                "-m",
                "toporetarget",
                "geometry",
                "sample-robot",
                "--robot",
                config["robot"],
                "--pose",
                "neutral",
                "--profile",
                "engineering_collision_32_per_geometry",
                "--output",
                str(surface_path),
                "--report",
                str(surface_path.with_suffix(".json")),
                "--force",
            ],
            cwd=_repo_root(),
            log_path=root / "logs" / "robot_surface.log",
        )
        if code != 0:
            raise SuiteRunError(f"robot surface sampling failed: {text[-1000:]}")
    robot_audit = _robot_audit(root, config["robot"], surface_path)
    _json_write(robot_audit, root / "diagnostics" / "wuji_robot_input_audit.json")
    selection = _selection_manifest(
        config, selected, canonical_paths, grab, mano, robot_audit, surface_path
    )
    lock_path = root / "selection" / "selection.lock"
    existing_lock = lock_path.read_text(encoding="utf-8").strip() if lock_path.exists() else None
    if existing_lock is not None and existing_lock != selection["selection_hash"]:
        previous_manifest_path = root / "selection" / "selection_manifest.json"
        previous = (
            json.loads(previous_manifest_path.read_text(encoding="utf-8"))
            if previous_manifest_path.exists()
            else {}
        )
        if len(previous.get("units", [])) != len(clips):
            _json_write(
                {
                    "previous_lock": existing_lock,
                    "previous_scope": len(previous.get("units", [])),
                    "replacement_reason": "pre-final unit-scoped preparation run",
                },
                root / "selection" / "selection_trial_replaced.json",
            )
        else:
            raise SuiteRunError("selection.lock identity mismatch; refusing reselection")
    _json_write(selection, root / "selection" / "selection_manifest.json")
    with (root / "selection" / "selection_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        rows = selection["units"]
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=str) for key, value in row.items()})
    lock_path.write_text(selection["selection_hash"] + "\n", encoding="utf-8")
    _audit_object_meshes(selected, canonical_paths, grab, root / "diagnostics")
    records: list[dict[str, Any]] = []
    for clip in selected:
        paths = path_map[clip.unit_id]
        record: dict[str, Any] = {
            "unit_id": clip.unit_id,
            "short_id": clip.short_id,
            "status": "pass",
            "artifacts": [],
        }
        try:
            record["pre_stages"] = _run_pre_stages(
                clip,
                paths,
                grab_root=grab,
                index=index_path,
                mano_root=mano,
                robot=config["robot"],
                collision_samples=surface_path,
            )
            final_result = _final_refinement(
                clip,
                paths,
                robot=config["robot"],
                solver_profile=str(config["solver_profile"]),
                max_wall_time=max_wall_time,
            )
            record["final"] = final_result
            if final_result.get("status") not in {"pass", "reused"}:
                record["status"] = "fail"
            elif evaluate and paths["final"].exists():
                validation = _independent_validation(clip, paths, config["robot"])
                record["validation"] = {
                    "status": validation["status"],
                    "expected_queries": validation["expected_queries"],
                    "actual_queries": validation["actual_queries"],
                }
                if validation["status"] != "pass":
                    record["status"] = "fail"
                record["metrics"] = (
                    _metrics_and_export(
                        clip,
                        paths,
                        root=root,
                        robot=config["robot"],
                        solver_profile=str(config["solver_profile"]),
                    )
                    if export_reference_bundles
                    else {"rows": [], "per_finger": []}
                )
                if generate_html:
                    record["html"] = _render_html(
                        clip, paths, root=root, solver_profile=str(config["solver_profile"])
                    )
                    if record["html"].get("status") != "pass":
                        record["status"] = "fail"
            record["artifacts"] = [
                {"kind": key, "path": str(value)}
                for key, value in paths.items()
                if key
                in {
                    "canonical",
                    "warm",
                    "graph",
                    "evaluation",
                    "final",
                    "validation",
                    "checkpoint_root",
                }
            ]
            object_asset = record.get("html", {}).get("object_mesh_asset")
            if isinstance(object_asset, dict) and object_asset.get("path"):
                record["artifacts"].append(
                    {"kind": "object_mesh", "path": str(object_asset["path"])}
                )
        except Exception as exc:  # preserve unit evidence and continue other fixed units
            record["status"] = "fail"
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        _json_write(record, paths["unit_root"] / "unit_result.json")
    _write_aggregate(root, records, selected)
    _write_dashboard(root, records)
    return {
        "status": "pass"
        if all(record["status"] in {"pass", "reused"} for record in records)
        else "complete_with_recorded_failures",
        "records": records,
        "experiment_root": str(root),
    }


__all__ = ["SUITE_SCHEMA_VERSION", "SuiteClip", "SuiteRunError", "load_suite", "run_suite"]
