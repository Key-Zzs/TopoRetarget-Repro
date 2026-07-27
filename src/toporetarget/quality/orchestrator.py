"""A--E quality experiment coordinator.

The coordinator is intentionally conservative: it never modifies raw GRAB,
MANO, Arti-MANO, or historical Stage 5--10 outputs, and every external solver
command is logged under the experiment root.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.adapters.base import FrameRange
from toporetarget.data.contacts.grab import load_grab_contact_mapping
from toporetarget.data.readers.grab import (
    load_grab_auxiliary,
    load_ply_mesh,
    read_grab_npz,
    resolve_grab_resource,
)
from toporetarget.geometry.mesh_audit import audit_mesh
from toporetarget.retarget.final_refinement import load_final_trajectory

from .contact import FINGER_TIPS, build_contact_candidates, build_source_contact_targets
from .geometry import audit_source_contact_boundary, build_geometry_artifacts
from .geometry_html import render_geometry_audit_html, smoke_geometry_html
from .html import render_clip_html, smoke_html
from .metrics import evaluate_all
from .morphology import build_morphology_candidates
from .schema import (
    CLIPS,
    CONTACTPOSE_STATUS,
    EXPERIMENT_ID,
    QUALITY_SCHEMA_VERSION,
    ClipSpec,
    QualityExperimentError,
    file_hash,
    git_commit,
    stable_hash,
    tree_hash,
    utc_now,
    write_json,
)
from .surfaces import build_artimano_surface_profile

BASELINE_PROFILES = (
    "scipy_slsqp_active_set_contact_rich_v2",
    "scipy_slsqp_active_set_contact_rich_v3_fixed",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config(repo_root: Path) -> Path:
    return repo_root / "configs" / "experiments" / f"{EXPERIMENT_ID}.yaml"


def _source_path(grab_root: Path, clip: ClipSpec) -> Path:
    return (grab_root / "grab" / clip.subject / f"{clip.sequence.split('/', 1)[1]}.npz").resolve()


def _run_command(
    command: list[str], *, repo_root: Path, log_path: Path, env: dict[str, str] | None = None
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(repo_root / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if env:
        run_env.update(env)
    result = subprocess.run(command, cwd=repo_root, env=run_env, text=True, capture_output=True)
    log_path.write_text(
        "$ " + shlex.join(command) + "\n\n" + result.stdout + "\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise QualityExperimentError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n{result.stderr[-3000:]}"
        )


def _write_selection_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "unit_id",
        "sequence",
        "subject",
        "object",
        "hand",
        "start_frame",
        "end_frame",
        "length",
        "native_fps",
        "source_hash",
        "mano_hash",
        "vtemp_hash",
        "object_mesh_hash",
        "contact_frame_ratio",
        "contact_regions",
        "source_integrity",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def freeze_selection(
    *,
    grab_root: str | Path,
    mano_root: str | Path,
    asset_root: str | Path,
    experiment_root: str | Path,
) -> dict[str, Any]:
    """Freeze exactly G1--G4 and validate right-hand identity via the adapter."""

    grab = Path(grab_root).expanduser().resolve()
    mano = Path(mano_root).expanduser().resolve()
    assets = Path(asset_root).expanduser().resolve()
    destination = Path(experiment_root)
    selection_root = destination / "selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    mano_hash = tree_hash(mano)
    asset_manifest_hash = file_hash(assets / "asset_manifest.json")
    mapping = load_grab_contact_mapping()
    rows: list[dict[str, Any]] = []
    for clip in CLIPS:
        clip.validate()
        source = _source_path(grab, clip)
        if not source.is_file():
            raise QualityExperimentError(f"DATA_INVALID: source GRAB file missing: {source}")
        record = read_grab_npz(source)
        if record.subject_id != clip.subject or record.object_name != clip.object_name:
            raise QualityExperimentError(
                f"DATA_IDENTITY_ERROR: metadata mismatch for {clip.unit_id}"
            )
        try:
            right = record.hand("right")
        except ValueError as exc:
            raise QualityExperimentError(
                f"DATA_IDENTITY_ERROR: {clip.unit_id} has no right hand"
            ) from exc
        root = grab
        object_path = resolve_grab_resource(root, record.object.mesh_relative, "GRAB object mesh")
        vtemp_path = (root / right.vtemp_relative).resolve()
        if not vtemp_path.is_file():
            # GRAB installations sometimes store vtemp relative to the source
            # directory; preserve whichever explicit adapter path resolves.
            vtemp_path = (source.parent / right.vtemp_relative).resolve()
        if not vtemp_path.is_file():
            raise QualityExperimentError(
                f"DATA_IDENTITY_ERROR: personalized vtemp missing: {vtemp_path}"
            )
        auxiliary = load_grab_auxiliary(
            source,
            frame_range=FrameRange(clip.start_frame, clip.end_frame),
            include_table=False,
            contact_mode="semantic",
        )
        labels = np.asarray(auxiliary["contact"]["object"], dtype=np.int64)
        names = {int(item["id"]): str(item["name"]) for item in mapping.table().values()}
        active_names = sorted(
            {
                names.get(int(value), f"unknown_{value}")
                for value in np.unique(labels)
                if int(value) != 0
            }
        )
        hand_labels = {
            int(item["id"])
            for item in mapping.table().values()
            if item.get("is_hand") and item.get("side") == "right"
        }
        contact_frames = np.any(np.isin(labels, sorted(hand_labels)), axis=1)
        vertices, faces = load_ply_mesh(object_path)
        audit = audit_mesh(vertices, faces, source_path=object_path)
        rows.append(
            {
                **clip.as_dict(),
                "source_path": str(source),
                "source_hash": file_hash(source),
                "mano_model_root": str(mano),
                "mano_hash": mano_hash,
                "artimano_asset_manifest_hash": asset_manifest_hash,
                "personalized_vtemp_path": str(vtemp_path),
                "vtemp_hash": file_hash(vtemp_path),
                "object_mesh_path": str(object_path),
                "object_mesh_hash": file_hash(object_path),
                "object_pose_track_hash": stable_hash(record.object.params),
                "table_mesh": record.table_metadata.get("mesh_relative"),
                "table_mesh_hash": file_hash(
                    resolve_grab_resource(
                        root, record.table_metadata["mesh_relative"], "GRAB table mesh"
                    )
                )
                if record.table_metadata.get("mesh_relative")
                else None,
                "contact_regions": active_names,
                "contact_frame_ratio": float(np.mean(contact_frames)),
                "contact_frame_count": int(np.count_nonzero(contact_frames)),
                "source_contact_topology_summary": {
                    "unique_labels": sorted(int(item) for item in np.unique(labels)),
                    "mapping_id": mapping.mapping_id,
                    "mapping_version": mapping.mapping_version,
                },
                "object_sdf_audit": audit.as_dict(),
                "source_scale_frame_audit": {
                    "native_fps": record.native_fps,
                    "num_frames": record.num_frames,
                    "selected_half_open": [clip.start_frame, clip.end_frame],
                    "no_temporal_resampling": True,
                    "no_result_based_reselection": True,
                },
                "source_integrity": "pass",
                "frozen_at": utc_now(),
            }
        )
    payload = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "git_commit": git_commit(_repo_root()),
        "created_at": utc_now(),
        "subject_scope": "within-subject multi-object development benchmark",
        "contactpose_status": CONTACTPOSE_STATUS,
        "contactpose_adapter_required": True,
        "contactpose_not_a_blocker": True,
        "selected_units": rows,
        "frozen_selection": True,
        "results_must_not_change_selection": True,
    }
    manifest_hash = stable_hash(payload)
    payload["manifest_hash"] = manifest_hash
    write_json(payload, selection_root / "selection_manifest.json")
    _write_selection_csv(rows, selection_root / "selection_manifest.csv")
    (selection_root / "selection.lock").write_text(
        "schema_version=toporetarget.quality_selection_lock.v1\n"
        f"experiment_id={EXPERIMENT_ID}\nmanifest={manifest_hash}\n"
        "results_must_not_change_selection=true\n"
        "frame_ranges_locked=true\nsequence_replacement_forbidden=true\n",
        encoding="utf-8",
    )
    return payload


def _selection_payload(experiment_root: Path) -> dict[str, Any]:
    path = experiment_root / "selection" / "selection_manifest.json"
    if not path.is_file():
        raise QualityExperimentError(f"selection is not frozen: {path}")
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    expected = dict(payload)
    actual = expected.pop("manifest_hash", None)
    if actual != stable_hash(expected):
        raise QualityExperimentError("selection manifest hash mismatch")
    if not (experiment_root / "selection" / "selection.lock").is_file():
        raise QualityExperimentError("selection lock is missing")
    return payload


def _baseline_paths(experiment_root: Path, clip: ClipSpec) -> dict[str, Path]:
    root = experiment_root / "baseline" / clip.unit_id
    return {
        "root": root,
        "canonical": root / "canonical.zarr",
        "object_samples": root / "object_samples.npz",
        "warm": root / "paper_warm.zarr",
        "graph": root / "interaction_graph.zarr",
        "evaluation": root / "interaction_evaluation.zarr",
        "checkpoint": root / "checkpoints",
        "v2": root / "scipy_slsqp_active_set_contact_rich_v2.zarr",
        "v3": root / "scipy_slsqp_active_set_contact_rich_v3_fixed.zarr",
    }


def _run_baseline_clip(
    clip: ClipSpec,
    record: dict[str, Any],
    *,
    experiment_root: Path,
    repo_root: Path,
    index_path: Path,
    mano_root: Path,
    asset_root: Path,
    max_wall_time: int,
    geometry_manifest: dict[str, Any],
    surface_profile_path: Path,
) -> dict[str, Any]:
    paths = _baseline_paths(experiment_root, clip)
    paths["root"].mkdir(parents=True, exist_ok=True)
    source = Path(record["source_path"])
    logs = paths["root"] / "logs"
    py = sys.executable
    if not paths["canonical"].is_dir():
        _run_command(
            [
                py,
                "-m",
                "toporetarget",
                "data",
                "convert",
                "--dataset",
                "grab",
                "--sequence",
                clip.sequence,
                "--index",
                str(index_path),
                "--hands",
                "right",
                "--contact-mode",
                "semantic",
                "--include-mediapipe21",
                "--start-frame",
                str(clip.start_frame),
                "--end-frame",
                str(clip.end_frame),
                "--mano-model-root",
                str(mano_root),
                "--output",
                str(paths["canonical"]),
                "--force",
            ],
            repo_root=repo_root,
            log_path=logs / "canonicalize.log",
        )
    if not paths["object_samples"].is_file():
        _run_command(
            [
                py,
                "-m",
                "toporetarget",
                "geometry",
                "sample-object",
                "--canonical",
                str(paths["canonical"]),
                "--object-id",
                "primary",
                "--profile",
                "paper_strict_area_uniform",
                "--output",
                str(paths["object_samples"]),
                "--report",
                str(paths["root"] / "object_samples.json"),
            ],
            repo_root=repo_root,
            log_path=logs / "sample_object_surface.log",
        )
    if not paths["warm"].is_dir():
        _run_command(
            [
                py,
                "-m",
                "toporetarget",
                "retarget",
                "warm-start",
                "--canonical",
                str(paths["canonical"]),
                "--hand",
                "right",
                "--robot",
                "artimano_rh",
                "--start-frame",
                "0",
                "--end-frame",
                "60",
                "--frame-profile",
                "canonical_keypoint_wrist_v1",
                "--bone-profile",
                "mediapipe21_full_finger_chain_v1",
                "--solver-profile",
                "paper_repro_scipy_trf",
                "--asset-root",
                str(asset_root),
                "--output",
                str(paths["warm"]),
                "--force",
            ],
            repo_root=repo_root,
            log_path=logs / "paper_warm.log",
        )
    if not paths["graph"].is_dir():
        _run_command(
            [
                py,
                "-m",
                "toporetarget",
                "retarget",
                "build-interaction-graph",
                "--canonical",
                str(paths["canonical"]),
                "--hand",
                "right",
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
                str(paths["root"] / "interaction_graph.json"),
                "--force",
            ],
            repo_root=repo_root,
            log_path=logs / "interaction_graph.log",
        )
    if not paths["evaluation"].is_dir():
        _run_command(
            [
                py,
                "-m",
                "toporetarget",
                "retarget",
                "evaluate-interaction",
                "--graph",
                str(paths["graph"]),
                "--warm-start",
                str(paths["warm"]),
                "--robot",
                "artimano_rh",
                "--asset-root",
                str(asset_root),
                "--output",
                str(paths["evaluation"]),
                "--force",
            ],
            repo_root=repo_root,
            log_path=logs / "interaction_evaluation.log",
        )
    source_contact_boundary = audit_source_contact_boundary(
        geometry_manifest=geometry_manifest,
        canonical_path=paths["canonical"],
        source_path=source,
        clip=clip,
    )
    source_contact = build_source_contact_targets(
        paths["canonical"],
        source,
        clip,
        surface_profile_path,
        experiment_root / "contact_final" / clip.unit_id,
    )
    if source_contact_boundary["conflict"]:
        raise QualityExperimentError(
            "SIGN_PROXY_CONTACT_REGION_CONFLICT: "
            f"{clip.unit_id} source contact region intersects the boundary exclusion zone"
        )
    final_profiles: dict[str, Path] = {}
    for profile_id, output in (
        (BASELINE_PROFILES[0], paths["v2"]),
        (BASELINE_PROFILES[1], paths["v3"]),
    ):
        if not _has_final_artifact(output):
            checkpoint = paths["checkpoint"] / profile_id
            command = [
                py,
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
                "artimano_rh",
                "--collision-samples",
                str(repo_root / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz"),
                "--query-profile",
                "adaptive_active_set_v1",
                "--coordinate-profile",
                "local_seed_delta_v1",
                "--solver-profile",
                profile_id,
                "--execution-profile",
                "cached_checkpoint_cpu_float64_v3",
                "--start-frame",
                "0",
                "--end-frame",
                "60",
                "--checkpoint-root",
                str(checkpoint),
                "--resume",
                "--max-wall-time",
                str(max_wall_time),
                "--asset-root",
                str(asset_root),
                "--output",
                str(output),
                "--force",
            ]
            while not _has_final_artifact(output):
                _run_command(
                    command,
                    repo_root=repo_root,
                    log_path=logs / f"{profile_id}.log",
                )
                if _has_final_artifact(output):
                    break
                progress = checkpoint / "progress.json"
                if not progress.is_file():
                    raise QualityExperimentError(
                        f"baseline checkpoint produced no final artifact: {output}"
                    )
                state = json.loads(progress.read_text(encoding="utf-8"))
                if int(state.get("remaining_frames", 0)) <= 0:
                    raise QualityExperimentError(
                        f"baseline checkpoint is complete but final artifact is missing: {output}"
                    )
        final_profiles[profile_id] = output
    return {
        "canonical": str(paths["canonical"].resolve()),
        "source": str(source.resolve()),
        "warm": str(paths["warm"].resolve()),
        "graph": str(paths["graph"].resolve()),
        "evaluation": str(paths["evaluation"].resolve()),
        "profiles": [(name, str(path.resolve()), False) for name, path in final_profiles.items()],
        "paper_warm_profile": ("paper_warm", str(paths["warm"].resolve()), True),
        "geometry": next(
            item for item in geometry_manifest["rows"] if item["unit_id"] == clip.unit_id
        ),
        "source_contact_boundary": source_contact_boundary,
        "source_contact_targets": source_contact,
    }


def _reusable_airplane_records(repo_root: Path) -> dict[str, str]:
    root = (
        repo_root
        / ".local/runs/stage10/s1__airplane_lift__right__artimano_rh__f000240_f000300/artifacts"
    )
    return {
        "canonical": str(root / "canonical.zarr"),
        "warm": str(root / "warm_start.zarr"),
        "graph": str(root / "interaction_graph.zarr"),
        "evaluation": str(root / "interaction_evaluation.zarr"),
        "v2": str(repo_root / ".local/cache/retarget/final/stage9_2_contact_rich_60f_v3.zarr"),
        "v3": str(
            repo_root / ".local/runs/stage9_4/faithful_regularization_fix_v1/repaired_60f.zarr"
        ),
    }


def _classify_baseline_failure(error: BaseException) -> str:
    """Classify a stopped baseline without collapsing data and solver failures."""

    message = str(error)
    if "SIGN_PROXY_CONTACT_REGION_CONFLICT" in message:
        return "SIGN_PROXY_CONTACT_REGION_CONFLICT"
    if "DERIVED_SDF_PROXY_FAILED" in message:
        return "DERIVED_SDF_PROXY_FAILED"
    if "strict signed distance requires watertight" in message:
        return "raw_grab_object_mesh_not_watertight_for_strict_signed_distance"
    if "DATA_IDENTITY_ERROR" in message or "DATA_INVALID" in message:
        return "data_identity_or_source_invalid"
    if "Inequality constraints incompatible" in message:
        return "solver_incompatible_constraints_at_fixed_configuration"
    return "baseline_command_failure"


def _classify_quality_failure(error: BaseException) -> str:
    """Classify a recorded extension-candidate failure without hiding it."""

    message = str(error)
    if "Positive directional derivative for linesearch" in message:
        return "optimizer_status_9_positive_directional_derivative"
    if "Inequality constraints incompatible" in message:
        return "optimizer_incompatible_constraints"
    return "quality_candidate_command_failure"


def _quality_extension_spec(
    *,
    kind: str,
    profile_id: str,
    target_artifact: Path,
    surface_profile: Path | None = None,
    lambda_morph: float = 0.0,
    lambda_contact_pos: float = 0.0,
    lambda_contact_dir: float = 0.0,
    output: Path,
) -> Path:
    """Write one immutable, hashable extension specification for refine."""

    payload: dict[str, Any] = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "kind": kind,
        "profile_id": profile_id,
        "target_artifact": str(target_artifact.resolve()),
    }
    if kind == "morphology_position_prior":
        payload["lambda_morph"] = float(lambda_morph)
    elif kind == "contact_final":
        if surface_profile is None:
            raise ValueError("contact extension requires a surface profile")
        payload.update(
            {
                "surface_profile": str(surface_profile.resolve()),
                "lambda_contact_pos": float(lambda_contact_pos),
                "lambda_contact_dir": float(lambda_contact_dir),
            }
        )
    else:
        raise ValueError(f"unsupported quality extension kind: {kind}")
    return write_json(payload, output)


def _run_quality_refinement(
    *,
    repo_root: Path,
    canonical: Path,
    warm: Path,
    graph: Path,
    collision_samples: Path,
    asset_root: Path,
    output: Path,
    log_path: Path,
    max_wall_time: int,
    quality_extension: Path | None = None,
    start_frame: int = 0,
    end_frame: int = 60,
    solver_profile: str = "scipy_slsqp_active_set_contact_rich_v2",
) -> None:
    """Run one fixed-config final solve and preserve its command provenance."""

    if _has_final_artifact(output):
        return
    command = [
        sys.executable,
        "-m",
        "toporetarget",
        "retarget",
        "refine",
        "--canonical",
        str(canonical),
        "--warm-start",
        str(warm),
        "--graph",
        str(graph),
        "--robot",
        "artimano_rh",
        "--collision-samples",
        str(collision_samples),
        "--query-profile",
        "adaptive_active_set_v1",
        "--coordinate-profile",
        "local_seed_delta_v1",
        "--solver-profile",
        solver_profile,
        "--execution-profile",
        "cached_checkpoint_cpu_float64_v3",
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--max-wall-time",
        str(max_wall_time),
        "--checkpoint-root",
        str(output.parent / f"{output.name}.checkpoints"),
        "--resume",
        "--asset-root",
        str(asset_root),
        "--output",
        str(output),
        "--force",
    ]
    if quality_extension is not None:
        command.extend(["--quality-extension", str(quality_extension)])
    while not _has_final_artifact(output):
        _run_command(command, repo_root=repo_root, log_path=log_path)
        if _has_final_artifact(output):
            break
        progress = output.parent / f"{output.name}.checkpoints" / "progress.json"
        if not progress.is_file():
            raise QualityExperimentError(f"quality refinement produced no artifact: {output}")
        state = json.loads(progress.read_text(encoding="utf-8"))
        if int(state.get("remaining_frames", 0)) <= 0:
            raise QualityExperimentError(
                f"quality refinement checkpoint is complete but final artifact is missing: {output}"
            )


def _has_final_artifact(path: Path) -> bool:
    """Return whether a persisted trajectory can actually be loaded.

    A killed/paused Zarr assembly can leave the output directory and its
    top-level metadata behind before all arrays are written.  Directory
    existence is therefore not sufficient for resume or downstream stages.
    """

    if not path.is_dir():
        return False
    try:
        load_final_trajectory(path)
    except (OSError, ValueError, RuntimeError, KeyError):
        return False
    return True


def _reusable_contact_selection(contact_root: Path) -> dict[str, dict[str, Any]] | None:
    """Load a completed contact-candidate decision from a prior attempt.

    A failed candidate is a first-class result of the quality matrix.  Reusing
    that result prevents a report-only retry from launching the same expensive
    SLSQP failure again.  Older diagnostic-only selections are deliberately
    rejected because they do not contain frozen solver outcomes.
    """

    path = contact_root / "contact_profile_selection.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    candidates = payload.get("candidates")
    legacy_diagnostic = (
        isinstance(candidates, list)
        and bool(candidates)
        and all(
            isinstance(candidate, dict)
            and not candidate.get("available", False)
            and not isinstance(candidate.get("solver_failure"), dict)
            for candidate in candidates
        )
    )
    if not isinstance(candidates, list) or not candidates or legacy_diagnostic:
        # A prior completed run may have been followed by the legacy
        # proxy-only writer.  Recover the solver-backed decision from the
        # immutable experiment summary before launching another refinement.
        summary_path = contact_root.parent.parent / "reports" / "experiment_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            unit_id = contact_root.name
            record = summary.get("records", {}).get(unit_id, {})
            candidates = record.get("contact", {}).get("candidates")
        except (OSError, json.JSONDecodeError, AttributeError):
            candidates = None
    if not isinstance(candidates, list) or not candidates:
        return None
    by_profile: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("profile_id"):
            return None
        # A reusable failure must carry its causal solver result.  A reusable
        # success must still pass the artifact gate on the current filesystem.
        if candidate.get("available", False):
            full_path = Path(str(candidate.get("full_trajectory", {}).get("path", "")))
            if not _has_final_artifact(full_path):
                return None
        elif not isinstance(candidate.get("solver_failure"), dict):
            # Recover a failure already proved by the immutable command log.
            # This keeps a report-only resume from launching the same expensive
            # fixed-config SLSQP failure a second time when an older selection
            # file predates candidate-level failure persistence.
            profile_id = str(candidate["profile_id"])
            final_log = contact_root / f"{profile_id}.log"
            failed_log = final_log if final_log.is_file() else None
            failure_stage = "full_trajectory"
            failed_frame: int | None = None
            if failed_log is None:
                prescreen_root = contact_root / "prescreen" / profile_id
                for path in sorted(prescreen_root.glob("frame_*.log")):
                    text = path.read_text(encoding="utf-8", errors="replace")
                    if "refine failed:" in text:
                        failed_log = path
                        failure_stage = "prescreen"
                        match = re.search(r"frame_(\d+)\.log$", path.name)
                        failed_frame = int(match.group(1)) if match else None
                        break
            if failed_log is None:
                return None
            text = failed_log.read_text(encoding="utf-8", errors="replace")
            marker = "refine failed:"
            if marker not in text:
                return None
            message = text.rsplit(marker, 1)[1].strip()
            match = re.search(r"(?:Stage 9 )?frame (\d+) failed: (.+)", message)
            if match:
                failed_frame = int(match.group(1))
                message = match.group(0)
            completed_frames: list[int] = []
            for checkpoint in sorted(
                (contact_root / "prescreen" / profile_id).glob(
                    "frame_*.zarr.checkpoints/progress.json"
                )
            ):
                try:
                    state = json.loads(checkpoint.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if state.get("status") == "complete":
                    frame_match = re.search(r"frame_(\d+)\.zarr", checkpoint.parent.name)
                    if frame_match:
                        completed_frames.append(int(frame_match.group(1)))
            failure = {
                "profile_id": profile_id,
                "stage": failure_stage,
                "frame": failed_frame,
                "classification": _classify_quality_failure(QualityExperimentError(message)),
                "error": message,
                "completed_prescreen_frames": completed_frames,
                "recovered_from_log": str(failed_log.resolve()),
            }
            candidate = {
                **candidate,
                "pre_screen": {
                    "status": "failed" if failure_stage == "prescreen" else "complete",
                    "completed_frames": completed_frames,
                    "failed_frame": failed_frame if failure_stage == "prescreen" else None,
                },
                "full_trajectory": {
                    "status": "not_run" if failure_stage == "prescreen" else "failed",
                    "path": str((contact_root / f"{profile_id}.zarr").resolve()),
                },
                "diagnostic_only": False,
                "available": False,
                "accepted": False,
                "rejection_reason": failure["classification"],
                "solver_failure": failure,
            }
        by_profile[str(candidate["profile_id"])] = candidate
    return by_profile


def _fixed_contact_prescreen_frames(proxy: dict[str, Any]) -> list[int]:
    frames = proxy["frames"]
    if len(frames) < 60:
        raise QualityExperimentError("contact proxy does not contain the locked 60 frames")
    transition = max(
        range(len(frames)),
        key=lambda index: sum(
            frames[index]["source_active"][name] != frames[max(index - 1, 0)]["source_active"][name]
            for name in FINGER_TIPS
        ),
    )
    density = max(
        range(len(frames)), key=lambda index: sum(frames[index]["source_active"].values())
    )
    return sorted({0, 29, 59, transition, density})


def _strict_artifact_gate(path: Path, *, expected_frames: int = 60) -> dict[str, Any]:
    """Read persisted final arrays and compute the non-negotiable E gates."""

    try:
        artifact = load_final_trajectory(path)
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status": "unparsable", "path": str(path.resolve()), "error": str(exc)}
    arrays = artifact.arrays
    accepted = np.asarray(arrays.get("accepted", []), dtype=bool)
    success = np.asarray(arrays.get("solver_success", []), dtype=bool)
    full_hard = np.asarray(arrays.get("full_surface_hard_audit_pass", []), dtype=bool)
    penetration = np.asarray(arrays.get("max_penetration", []), dtype=np.float64)
    return {
        "status": "pass" if len(accepted) == expected_frames else "incomplete",
        "path": str(path.resolve()),
        "frame_count": int(len(accepted)),
        "solver_success_frames": int(np.count_nonzero(success)),
        "strict_accepted_frames": int(np.count_nonzero(accepted)),
        "complete_60_frames": bool(len(accepted) == expected_frames),
        "strict_accepted": bool(len(accepted) == expected_frames and np.all(accepted)),
        "full_512_pass": bool(len(full_hard) == expected_frames and np.all(full_hard)),
        "zero_penetration_over_2mm": bool(
            len(penetration) == expected_frames and np.all(penetration <= 0.002)
        ),
        "artifact_hash": file_hash(path / "zarr.json") if (path / "zarr.json").is_file() else None,
    }


def run_a_to_e(
    *,
    grab_root: str | Path,
    mano_root: str | Path,
    asset_root: str | Path,
    index_path: str | Path,
    experiment_root: str | Path,
    repo_root: str | Path | None = None,
    max_wall_time: int = 1800,
    resume: bool = True,
    generate_html: bool = True,
    skip_units: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    del resume
    repo = Path(repo_root or _repo_root()).resolve()
    destination = Path(experiment_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    requested_skip = set(skip_units or ())
    known_units = {item.unit_id for item in CLIPS}
    unknown_skip = requested_skip - known_units
    if unknown_skip:
        raise QualityExperimentError(f"unknown skipped quality units: {sorted(unknown_skip)}")
    try:
        selection = _selection_payload(destination)
    except QualityExperimentError:
        selection = freeze_selection(
            grab_root=grab_root,
            mano_root=mano_root,
            asset_root=asset_root,
            experiment_root=destination,
        )
    geometry_manifest = build_geometry_artifacts(selection, destination)
    # Stage B is independent from the optimizer.  Build its deterministic
    # Arti-MANO visual-region contract before A so a solver hard blocker still
    # leaves a complete, auditable surface artifact.
    surface = build_artimano_surface_profile(
        destination / "surface_contact",
        asset_root=asset_root,
    )
    records: dict[str, dict[str, Any]] = {}
    evaluated_clips = tuple(item for item in CLIPS if item.unit_id not in requested_skip)
    for clip in CLIPS:
        if clip.unit_id not in requested_skip:
            continue
        selected = next(
            item for item in selection["selected_units"] if item["unit_id"] == clip.unit_id
        )
        geometry_row = next(
            item for item in geometry_manifest["rows"] if item["unit_id"] == clip.unit_id
        )
        existing = _baseline_paths(destination, clip)
        records[clip.unit_id] = {
            "unit_id": clip.unit_id,
            "sequence": clip.sequence,
            "frame_range": [clip.start_frame, clip.end_frame],
            "source": selected["source_path"],
            "canonical": str(existing["canonical"].resolve())
            if existing["canonical"].exists()
            else None,
            "geometry": geometry_row,
            "skipped": True,
            "skip_reason": "SKIPPED_BY_USER_AUTHORITY",
            "downstream_status": "NOT_RUN_BY_EXPLICIT_USER_SCOPE_OVERRIDE",
        }
    write_json(
        {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "status": "explicit_scope_override",
            "skipped_units": sorted(requested_skip),
            "reason": "SKIPPED_BY_USER_AUTHORITY",
            "frozen_selection_unchanged": True,
            "sequence_reselection": False,
            "frame_reselection": False,
        },
        destination / "reports" / "skip_authority.json",
    )
    baseline_root = destination / "baseline"
    baseline_root.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        if clip.unit_id in requested_skip:
            continue
        record = next(
            item for item in selection["selected_units"] if item["unit_id"] == clip.unit_id
        )
        if clip.unit_id == "G1":
            reusable = _reusable_airplane_records(repo)
            if all(Path(item).exists() for item in reusable.values()):
                records[clip.unit_id] = {
                    "canonical": reusable["canonical"],
                    "source": record["source_path"],
                    "warm": reusable["warm"],
                    "graph": reusable["graph"],
                    "evaluation": reusable["evaluation"],
                    "profiles": [
                        (BASELINE_PROFILES[0], reusable["v2"], False),
                        (BASELINE_PROFILES[1], reusable["v3"], False),
                    ],
                    "paper_warm_profile": ("paper_warm", reusable["warm"], True),
                    "geometry": next(
                        item
                        for item in geometry_manifest["rows"]
                        if item["unit_id"] == clip.unit_id
                    ),
                    "reused_existing_artifact": True,
                }
                records[clip.unit_id]["source_contact_targets"] = build_source_contact_targets(
                    reusable["canonical"],
                    record["source_path"],
                    clip,
                    destination / "surface_contact" / "artimano_surface_profile.json",
                    destination / "contact_final" / clip.unit_id,
                )
                continue
        try:
            records[clip.unit_id] = _run_baseline_clip(
                clip,
                record,
                experiment_root=destination,
                repo_root=repo,
                index_path=Path(index_path).resolve(),
                mano_root=Path(mano_root).resolve(),
                asset_root=Path(asset_root).resolve(),
                max_wall_time=max_wall_time,
                geometry_manifest=geometry_manifest,
                surface_profile_path=destination
                / "surface_contact"
                / "artimano_surface_profile.json",
            )
        except (QualityExperimentError, OSError, RuntimeError, ValueError) as exc:
            hard_blocker = _classify_baseline_failure(exc)
            surface_complete = (destination / "surface_contact" / "validation.json").is_file()
            formal_status = "GRAB_QUALITY_A_TO_E_BLOCKED"
            failure = {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "status": formal_status,
                "hard_blocker": hard_blocker,
                "blocking_reason": hard_blocker,
                "failed_stage": "A_baseline_final",
                "unit_id": clip.unit_id,
                "sequence": clip.sequence,
                "frame_range": [clip.start_frame, clip.end_frame],
                "profile": BASELINE_PROFILES[0],
                "error": str(exc),
                "source_path": record.get("source_path"),
                "object_mesh_path": record.get("object_mesh_path"),
                "object_sdf_audit": record.get("object_sdf_audit"),
                "geometry": next(
                    item for item in geometry_manifest["rows"] if item["unit_id"] == clip.unit_id
                ),
                "geometry_manifest": str(
                    (destination / "geometry" / "geometry_manifest.json").resolve()
                ),
                "surface_profile": str(
                    (destination / "surface_contact" / "artimano_surface_profile.json").resolve()
                ),
                "surface_validation": str(
                    (destination / "surface_contact" / "validation.json").resolve()
                ),
                "source_contact_targets": str(
                    (
                        destination / "contact_final" / clip.unit_id / "source_contact_targets.json"
                    ).resolve()
                ),
                "resume_or_fallback": "not_permitted_without_new_authority; no frame skipping, per-clip tuning, profile substitution, or sequence replacement",
                "stage_status": {
                    "A": "NO",
                    "B": "YES (surface artifact independently built)" if surface_complete else "NO",
                    "C": "NO",
                    "D": "NO",
                    "E": "NO",
                },
                "manual_acceptance_required": False,
                "contactpose_status": CONTACTPOSE_STATUS,
                "partial_records": records,
            }
            write_json(failure, destination / "reports" / "failure_report.json")
            write_json(failure, destination / "reports" / "experiment_status.json")
            write_json(
                {
                    "status": "blocked_before_raw_mutation",
                    "raw_source_modified": False,
                    "old_stage10_modified": False,
                    "tracked_local_artifacts": False,
                    "blocking_reason": hard_blocker,
                },
                destination / "reports" / "source_integrity.json",
            )
            write_json(
                {
                    "status": "blocked",
                    "blocking_reason": hard_blocker,
                    "files": sorted(
                        str(path.relative_to(destination))
                        for path in destination.rglob("*")
                        if path.is_file()
                    ),
                },
                destination / "reports" / "artifact_manifest.json",
            )
            raise
    write_json(records, destination / "baseline" / "baseline_records.json")
    # Stage B is independent of the solver and can be validated even if a later
    # extension is rejected.
    for clip in evaluated_clips:
        record = next(
            item for item in selection["selected_units"] if item["unit_id"] == clip.unit_id
        )
        clip_record = records[clip.unit_id]
        clip_root = destination / "morphology_final" / clip.unit_id
        clip_root.mkdir(parents=True, exist_ok=True)
        morphology = build_morphology_candidates(
            clip_record["canonical"],
            clip_record["warm"],
            destination / "morphology_warm" / clip.unit_id,
            asset_root=asset_root,
        )
        morphology_target = Path(morphology["m_star"]["target_artifact"])
        c1_path = clip_root / "morphology_seed_only_v1.zarr"
        if not _has_final_artifact(c1_path):
            _run_quality_refinement(
                repo_root=repo,
                canonical=Path(clip_record["canonical"]),
                warm=Path(morphology["m_star"]["path"]),
                graph=Path(clip_record["graph"]),
                collision_samples=repo
                / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz",
                asset_root=Path(asset_root),
                output=c1_path,
                log_path=destination
                / "morphology_final"
                / clip.unit_id
                / "morphology_seed_only_v1.log",
                max_wall_time=max_wall_time,
            )
        morphology["actual_final"] = {"morphology_seed_only_v1": _strict_artifact_gate(c1_path)}
        for profile_id, lambda_morph in (
            ("morphology_position_prior_v1_lambda_0.1", 0.1),
            ("morphology_position_prior_v1_lambda_1", 1.0),
        ):
            extension = _quality_extension_spec(
                kind="morphology_position_prior",
                profile_id=profile_id,
                target_artifact=morphology_target,
                lambda_morph=lambda_morph,
                output=clip_root / f"{profile_id}.extension.json",
            )
            final_path = clip_root / f"{profile_id}.zarr"
            if not _has_final_artifact(final_path):
                _run_quality_refinement(
                    repo_root=repo,
                    canonical=Path(clip_record["canonical"]),
                    warm=Path(morphology["m_star"]["path"]),
                    graph=Path(clip_record["graph"]),
                    collision_samples=repo
                    / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz",
                    asset_root=Path(asset_root),
                    output=final_path,
                    log_path=clip_root / f"{profile_id}.log",
                    max_wall_time=max_wall_time,
                    quality_extension=extension,
                )
            morphology["actual_final"][profile_id] = _strict_artifact_gate(final_path)
        clip_record["morphology"] = morphology
        clip_record["profiles"].append(("morphology_seed_only_v1", str(c1_path), False))
        for profile_id in (
            "morphology_position_prior_v1_lambda_0.1",
            "morphology_position_prior_v1_lambda_1",
        ):
            clip_record["profiles"].append(
                (profile_id, str(clip_root / f"{profile_id}.zarr"), False)
            )
        contact = build_contact_candidates(
            clip_record["canonical"],
            record["source_path"],
            clip_record["profiles"][0][1],
            clip,
            destination / "surface_contact" / "artimano_surface_profile.json",
            destination / "contact_final" / clip.unit_id,
        )
        contact_root = destination / "contact_final" / clip.unit_id
        contact_target = Path(clip_record["source_contact_targets"]["artifact"])
        surface_profile_path = destination / "surface_contact" / "artimano_surface_profile.json"
        prescreen_frames = _fixed_contact_prescreen_frames(contact["proxy"])
        contact["prescreen_frames"] = prescreen_frames
        contact["actual_final"] = {}
        reusable_candidates = _reusable_contact_selection(contact_root)
        for candidate in contact["candidates"]:
            profile_id = str(candidate["profile_id"])
            lambda_pos = float(candidate["lambda_contact_pos"])
            lambda_dir = float(candidate["lambda_contact_dir"])
            extension = _quality_extension_spec(
                kind="contact_final",
                profile_id=profile_id,
                target_artifact=contact_target,
                surface_profile=surface_profile_path,
                lambda_contact_pos=lambda_pos,
                lambda_contact_dir=lambda_dir,
                output=contact_root / f"{profile_id}.extension.json",
            )
            prescreen_root = contact_root / "prescreen" / profile_id
            contact_candidate = next(
                item for item in contact["candidates"] if item["profile_id"] == profile_id
            )
            if reusable_candidates is not None and profile_id in reusable_candidates:
                persisted = reusable_candidates[profile_id]
                contact_candidate.clear()
                contact_candidate.update(persisted)
                if persisted.get("available", False):
                    contact["actual_final"][profile_id] = persisted["full_trajectory"]
                elif isinstance(persisted.get("solver_failure"), dict):
                    contact.setdefault("solver_failures", []).append(persisted["solver_failure"])
                continue
            completed_prescreen_frames: list[int] = []
            failed_frame: int | None = None
            failure_stage = "full_trajectory"
            try:
                for frame in prescreen_frames:
                    prescreen_path = prescreen_root / f"frame_{frame:04d}.zarr"
                    failure_stage = "prescreen"
                    failed_frame = frame
                    _run_quality_refinement(
                        repo_root=repo,
                        canonical=Path(clip_record["canonical"]),
                        warm=Path(clip_record["warm"]),
                        graph=Path(clip_record["graph"]),
                        collision_samples=repo
                        / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz",
                        asset_root=Path(asset_root),
                        output=prescreen_path,
                        log_path=prescreen_root / f"frame_{frame:04d}.log",
                        max_wall_time=max_wall_time,
                        quality_extension=extension,
                        start_frame=frame,
                        end_frame=frame + 1,
                    )
                    completed_prescreen_frames.append(frame)
                failed_frame = None
                failure_stage = "full_trajectory"
                final_path = contact_root / f"{profile_id}.zarr"
                _run_quality_refinement(
                    repo_root=repo,
                    canonical=Path(clip_record["canonical"]),
                    warm=Path(clip_record["warm"]),
                    graph=Path(clip_record["graph"]),
                    collision_samples=repo
                    / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz",
                    asset_root=Path(asset_root),
                    output=final_path,
                    log_path=contact_root / f"{profile_id}.log",
                    max_wall_time=max_wall_time,
                    quality_extension=extension,
                )
            except QualityExperimentError as exc:
                failure = {
                    "profile_id": profile_id,
                    "stage": failure_stage,
                    "frame": failed_frame,
                    "classification": _classify_quality_failure(exc),
                    "error": str(exc),
                    "completed_prescreen_frames": completed_prescreen_frames,
                }
                contact.setdefault("solver_failures", []).append(failure)
                contact_candidate.update(
                    {
                        "extension_spec": str(extension.resolve()),
                        "pre_screen": {
                            "frames": prescreen_frames,
                            "status": "failed" if failed_frame is not None else "complete",
                            "completed_frames": completed_prescreen_frames,
                            "failed_frame": failed_frame,
                            "artifacts": [
                                str(
                                    (
                                        contact_root
                                        / "prescreen"
                                        / profile_id
                                        / f"frame_{frame:04d}.zarr"
                                    ).resolve()
                                )
                                for frame in completed_prescreen_frames
                            ],
                        },
                        "full_trajectory": {
                            "status": "not_run" if failed_frame is not None else "failed",
                            "path": str((contact_root / f"{profile_id}.zarr").resolve()),
                        },
                        "diagnostic_only": False,
                        "available": False,
                        "accepted": False,
                        "rejection_reason": failure["classification"],
                        "solver_failure": failure,
                    }
                )
                continue
            final_gate = _strict_artifact_gate(final_path)
            contact["actual_final"][profile_id] = final_gate
            contact_candidate.update(
                {
                    "extension_spec": str(extension.resolve()),
                    "pre_screen": {
                        "frames": prescreen_frames,
                        "status": "complete",
                        "completed_frames": prescreen_frames,
                        "artifacts": [
                            str(
                                (
                                    contact_root
                                    / "prescreen"
                                    / profile_id
                                    / f"frame_{frame:04d}.zarr"
                                ).resolve()
                            )
                            for frame in prescreen_frames
                        ],
                    },
                    "full_trajectory": final_gate,
                    "diagnostic_only": False,
                    "available": True,
                    "accepted": bool(final_gate.get("strict_accepted", False)),
                    "rejection_reason": None
                    if final_gate.get("strict_accepted", False)
                    else "strict_artifact_gate_failed",
                }
            )
        available_candidates = [
            item for item in contact["candidates"] if item.get("available", False)
        ]
        if not available_candidates:
            write_json(
                {
                    "schema_version": QUALITY_SCHEMA_VERSION,
                    "surface_profile": str(surface_profile_path.resolve()),
                    "selected_source_frames": prescreen_frames,
                    "candidates": contact["candidates"],
                    "pre_screen": {
                        "status": "complete_with_recorded_failures",
                        "fixed_frame_count": len(prescreen_frames),
                        "frames": prescreen_frames,
                    },
                    "full_trajectory": {
                        "status": "all_candidates_failed",
                        "contact_profile_accepted": False,
                    },
                    "c_star": None,
                },
                contact_root / "contact_profile_selection.json",
            )
            clip_record["contact"] = contact
            clip_record["contact_failure"] = {
                "status": "all_contact_candidates_failed",
                "solver_failures": contact.get("solver_failures", []),
            }
            # E2/E3 require C_STAR and are not fabricated when the fixed
            # contact matrix has no usable candidate for this clip.
            continue
        accepted_candidates = [item for item in available_candidates if item["accepted"]]
        selected_contact = min(
            accepted_candidates or available_candidates,
            key=lambda item: (float(item.get("diagnostic_loss", math.inf)), item["profile_id"]),
        )
        contact["c_star"] = {
            **selected_contact,
            "not_recommended": not bool(accepted_candidates),
        }
        write_json(
            {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "surface_profile": str(surface_profile_path.resolve()),
                "selected_source_frames": prescreen_frames,
                "candidates": contact["candidates"],
                "pre_screen": {
                    "status": "complete_with_recorded_failures"
                    if contact.get("solver_failures")
                    else "complete",
                    "fixed_frame_count": len(prescreen_frames),
                    "frames": prescreen_frames,
                },
                "full_trajectory": {
                    "status": "complete_with_recorded_failures"
                    if contact.get("solver_failures")
                    else "complete",
                    "contact_profile_accepted": bool(accepted_candidates),
                },
                "c_star": contact["c_star"],
            },
            contact_root / "contact_profile_selection.json",
        )
        for candidate in contact["candidates"]:
            profile_id = str(candidate["profile_id"])
            if not candidate.get("available", False):
                continue
            clip_record["profiles"].append(
                (f"contact_{profile_id}", str(contact_root / f"{profile_id}.zarr"), False)
            )
        clip_record["contact"] = contact
        selected_contact_id = str(selected_contact["profile_id"])
        selected_contact_extension = contact_root / f"{selected_contact_id}.extension.json"
        e2_path = destination / "matrix_2x2" / clip.unit_id / "E2_paper_warm_C_STAR.zarr"
        if not _has_final_artifact(e2_path):
            _run_quality_refinement(
                repo_root=repo,
                canonical=Path(clip_record["canonical"]),
                warm=Path(clip_record["warm"]),
                graph=Path(clip_record["graph"]),
                collision_samples=repo
                / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz",
                asset_root=Path(asset_root),
                output=e2_path,
                log_path=destination / "matrix_2x2" / clip.unit_id / "E2.log",
                max_wall_time=max_wall_time,
                quality_extension=selected_contact_extension,
            )
        e3_path = destination / "matrix_2x2" / clip.unit_id / "E3_M_STAR_C_STAR.zarr"
        if not _has_final_artifact(e3_path):
            _run_quality_refinement(
                repo_root=repo,
                canonical=Path(clip_record["canonical"]),
                warm=Path(morphology["m_star"]["path"]),
                graph=Path(clip_record["graph"]),
                collision_samples=repo
                / ".local/cache/geometry/robot_surface/artimano_rh_neutral.npz",
                asset_root=Path(asset_root),
                output=e3_path,
                log_path=destination / "matrix_2x2" / clip.unit_id / "E3.log",
                max_wall_time=max_wall_time,
                quality_extension=selected_contact_extension,
            )
        clip_record["profiles"].extend(
            [
                ("E0_paper_warm_plus_development_base_final", clip_record["profiles"][0][1], False),
                ("E1_m_star_plus_development_base_final", str(c1_path), False),
                ("E2_paper_warm_plus_C_star", str(e2_path), False),
                ("E3_m_star_plus_C_star", str(e3_path), False),
            ]
        )
        clip_record["matrix"] = {
            "E0": {
                "warm": clip_record["warm"],
                "final": clip_record["profiles"][0][1],
                "profile": "scipy_slsqp_active_set_contact_rich_v2",
            },
            "E1": {
                "warm": morphology["m_star"]["path"],
                "final": str(c1_path),
                "profile": "E1_m_star_plus_development_base_final",
            },
            "E2": {
                "warm": clip_record["warm"],
                "final": str(e2_path),
                "profile": "E2_paper_warm_plus_C_star",
            },
            "E3": {
                "warm": morphology["m_star"]["path"],
                "final": str(e3_path),
                "profile": "E3_m_star_plus_C_star",
            },
        }
    metrics = evaluate_all(evaluated_clips, records, destination / "reports")
    summary = _assemble_reports(
        destination,
        selection,
        records,
        metrics,
        surface,
        geometry_manifest,
        evaluated_clips=evaluated_clips,
        skipped_units=sorted(requested_skip),
    )
    audit_clip = next(
        (item for item in CLIPS if item.unit_id == "G3" and item.unit_id not in requested_skip),
        evaluated_clips[0],
    )
    audit_record = records[audit_clip.unit_id]
    if not audit_record.get("canonical"):
        audit_clip = evaluated_clips[0]
        audit_record = records[audit_clip.unit_id]
    geometry_audit = render_geometry_audit_html(
        geometry_manifest=geometry_manifest,
        canonical_path=audit_record["canonical"],
        source_path=audit_record["source"],
        clip=audit_clip,
        output=destination / "geometry" / f"{audit_clip.unit_id}_original_vs_proxy.html",
    )
    summary["geometry_audit_html"] = {
        **geometry_audit,
        "smoke": smoke_geometry_html(geometry_audit["path"]),
    }
    write_json(summary, destination / "reports" / "experiment_summary.json")
    if generate_html:
        html_root = destination / "html"
        html_root.mkdir(parents=True, exist_ok=True)
        recommended = summary["recommended_profile"]
        html_rows: list[dict[str, Any]] = []
        for clip in evaluated_clips:
            record = records[clip.unit_id]
            morphology_path = record["morphology"]["m_star"]["path"]
            matrix = record.get("matrix", {})
            profiles = {
                "paper_warm": (record["warm"], True, "paper warm"),
                "morphology_seed_only_v1": (morphology_path, True, "morphology-aware warm M_STAR"),
                BASELINE_PROFILES[0]: (
                    record["profiles"][0][1],
                    False,
                    "full-state temporal final",
                ),
                BASELINE_PROFILES[1]: (
                    record["profiles"][1][1],
                    False,
                    "finger-only temporal final",
                ),
            }
            if matrix:
                profiles.update(
                    {
                        "E0_paper_warm_plus_development_base_final": (
                            matrix["E0"]["final"],
                            False,
                            "E0 paper warm + development base final",
                        ),
                        "E1_m_star_plus_development_base_final": (
                            matrix["E1"]["final"],
                            False,
                            "E1 M_STAR + development base final",
                        ),
                        "E2_paper_warm_plus_C_star": (
                            matrix["E2"]["final"],
                            False,
                            "E2 paper warm + C_STAR",
                        ),
                        "E3_m_star_plus_C_star": (
                            matrix["E3"]["final"],
                            False,
                            "E3 M_STAR + C_STAR",
                        ),
                    }
                )
            path = render_clip_html(
                clip=clip,
                canonical_path=record["canonical"],
                source_path=record["source"],
                profile_paths=profiles,
                output=html_root / f"{clip.unit_id}_{clip.object_name}_visualize_mesh.html",
                asset_root=asset_root,
                recommended_profile=recommended,
                graph_path=record["graph"],
                evaluation_path=record["evaluation"],
            )
            html_rows.append(smoke_html(path, expected_frames=60, profiles=len(profiles)))
        write_json(html_rows, html_root / "html_smoke_report.json")
        index = html_root / "index.html"
        links = "\n".join(
            f'<li><a href="{clip.unit_id}_{clip.object_name}_visualize_mesh.html">{clip.unit_id} {clip.sequence}</a></li>'
            for clip in evaluated_clips
        )
        index.write_text(
            f"<!doctype html><meta charset='utf-8'><title>GRAB quality A-E</title>"
            f"<h1>GRAB Arti-MANO quality A-E</h1><p>recommended={recommended}</p><ul>{links}</ul>",
            encoding="utf-8",
        )
        summary["html_smoke"] = html_rows
        write_json(summary, destination / "reports" / "experiment_summary.json")
    return summary


def _assemble_reports(
    destination: Path,
    selection: dict[str, Any],
    records: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    surface: Any,
    geometry_manifest: dict[str, Any],
    *,
    evaluated_clips: tuple[ClipSpec, ...] = CLIPS,
    skipped_units: list[str] | None = None,
) -> dict[str, Any]:
    rows = list(metrics["rows"])
    e0 = "E0_paper_warm_plus_development_base_final"
    matrix_profiles = (
        e0,
        "E1_m_star_plus_development_base_final",
        "E2_paper_warm_plus_C_star",
        "E3_m_star_plus_C_star",
    )
    metric_keys = (
        "contact_f1_5mm",
        "contact_precision_proxy",
        "contact_alignment_proxy",
        "whole_hand_morphology_rmse_mm",
        "q_jerk_max",
        "penetration_max_mm",
        "runtime_p95_s",
    )

    def aggregate_profile(profile: str) -> dict[str, Any]:
        selected = [item for item in rows if item["profile"] == profile]
        result: dict[str, Any] = {
            "profile": profile,
            "clip_count": len(selected),
            "complete_clip_count": int(
                sum(bool(item.get("complete_60_frames")) for item in selected)
            ),
            "strict_accepted_clip_count": int(
                sum(bool(item.get("strict_accepted")) for item in selected)
            ),
        }
        for key in metric_keys:
            values = [
                float(item[key])
                for item in selected
                if isinstance(item.get(key), (int, float)) and math.isfinite(float(item[key]))
            ]
            result[key] = float(np.mean(values)) if values else None
        result["per_clip"] = selected
        return result

    aggregate = {
        profile: aggregate_profile(profile)
        for profile in sorted({item["profile"] for item in rows})
    }
    e0_rows = [item for item in rows if item["profile"] == e0]
    all_complete = bool(
        len(e0_rows) == len(evaluated_clips)
        and all(
            item["strict_accepted"]
            and item["complete_60_frames"]
            and item["full_512_pass"]
            and item["penetration_frames_gt_2mm"] == 0
            for item in e0_rows
        )
    )
    e0_aggregate = aggregate.get(e0, {})
    gate_results: dict[str, Any] = {}
    for profile in matrix_profiles[1:]:
        candidate_rows = [item for item in rows if item["profile"] == profile]
        per_clip: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            reference = next(item for item in e0_rows if item["unit_id"] == candidate["unit_id"])
            contact_delta = float(candidate["contact_f1_5mm"]) - float(reference["contact_f1_5mm"])
            morphology_delta = float(candidate["whole_hand_morphology_rmse_mm"]) - float(
                reference["whole_hand_morphology_rmse_mm"]
            )
            continuity_ratio = float(candidate["q_jerk_max"]) / max(
                float(reference["q_jerk_max"]), 1e-12
            )
            per_clip.append(
                {
                    "unit_id": candidate["unit_id"],
                    "strict_artifact_gate": bool(
                        candidate["complete_60_frames"]
                        and candidate["strict_accepted"]
                        and candidate["full_512_pass"]
                        and candidate["penetration_frames_gt_2mm"] == 0
                    ),
                    "contact_f1_delta": contact_delta,
                    "morphology_rmse_delta_mm": morphology_delta,
                    "continuity_ratio": continuity_ratio,
                    "contact_gate": contact_delta
                    >= -0.10 * max(float(reference["contact_f1_5mm"]), 1e-12),
                    "morphology_gate": morphology_delta <= 0.5,
                    "continuity_gate": continuity_ratio <= 1.2,
                    "runtime_gate": float(candidate["runtime_p95_s"])
                    <= 3.0 * max(float(reference["runtime_p95_s"]), 1e-12),
                }
            )
        aggregate_candidate = aggregate.get(profile, {})
        macro_f1_gain = float(aggregate_candidate.get("contact_f1_5mm") or 0.0) - float(
            e0_aggregate.get("contact_f1_5mm") or 0.0
        )
        improved_clips = sum(item["contact_f1_delta"] > 0.0 for item in per_clip)
        hard = bool(per_clip) and all(
            all(
                item[name]
                for name in (
                    "strict_artifact_gate",
                    "contact_gate",
                    "morphology_gate",
                    "continuity_gate",
                    "runtime_gate",
                )
            )
            for item in per_clip
        )
        gate_results[profile] = {
            "hard_gates_pass": hard,
            "improved_clip_count": int(improved_clips),
            "required_improved_clip_count": 3,
            "macro_contact_f1_gain": macro_f1_gain,
            "required_macro_contact_f1_gain": 0.05,
            "net_gain_gate": improved_clips >= 3 and macro_f1_gain >= 0.05,
            "per_clip": per_clip,
            "pass": bool(hard and improved_clips >= 3 and macro_f1_gain >= 0.05),
        }

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_values = (
            float(left.get("contact_f1_5mm") or -math.inf),
            float(left.get("contact_precision_proxy") or -math.inf),
            float(left.get("contact_alignment_proxy") or -math.inf),
            -float(left.get("whole_hand_morphology_rmse_mm") or math.inf),
            -float(left.get("penetration_max_mm") or math.inf),
            -float(left.get("q_jerk_max") or math.inf),
            -float(left.get("runtime_p95_s") or math.inf),
        )
        right_values = (
            float(right.get("contact_f1_5mm") or -math.inf),
            float(right.get("contact_precision_proxy") or -math.inf),
            float(right.get("contact_alignment_proxy") or -math.inf),
            -float(right.get("whole_hand_morphology_rmse_mm") or math.inf),
            -float(right.get("penetration_max_mm") or math.inf),
            -float(right.get("q_jerk_max") or math.inf),
            -float(right.get("runtime_p95_s") or math.inf),
        )
        return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
            a > b for a, b in zip(left_values, right_values, strict=True)
        )

    passing = [profile for profile, result in gate_results.items() if result["pass"]]
    pareto = [
        profile
        for profile in passing
        if not any(
            other != profile and dominates(aggregate.get(other, {}), aggregate.get(profile, {}))
            for other in passing
        )
    ]
    recommended = sorted(pareto)[0] if pareto else e0
    extensions_rejected = not bool(passing)
    contact_failure_units = sorted(
        unit_id
        for unit_id, record in records.items()
        if record.get("contact_failure", {}).get("status") == "all_contact_candidates_failed"
    )
    status = (
        "GRAB_QUALITY_A_TO_E_COMPLETE_BASELINE_RECOMMENDED"
        if extensions_rejected and all_complete and not skipped_units
        else "GRAB_QUALITY_A_TO_E_COMPLETE_WITH_RECORDED_FAILURES"
    )
    partial_scope = "explicit skipped units" if skipped_units else ""
    if contact_failure_units:
        partial_scope = "; ".join(
            item for item in (partial_scope, "recorded contact candidate failures") if item
        )
    partial_status = f"PARTIAL ({partial_scope})" if partial_scope else "YES"
    reports = destination / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "stage_status": {
            "A": "PARTIAL (explicit skipped units)" if skipped_units else "YES",
            "B": "YES",
            "C": partial_status if skipped_units or contact_failure_units else "YES",
            "D": partial_status if skipped_units or contact_failure_units else "YES",
            "E": partial_status if skipped_units or contact_failure_units else "YES",
        },
        "recommended_profile": recommended,
        "extensions_rejected": extensions_rejected,
        "reason": (
            "no extension passed the fixed hard gates and net-gain rule"
            if extensions_rejected
            else "automatic Pareto selection over the passing fixed matrix extensions"
        ),
        "subject_scope": "within-subject multi-object development benchmark",
        "evaluated_units": [item.unit_id for item in evaluated_clips],
        "skipped_units": list(skipped_units or []),
        "skip_scope_override": bool(skipped_units),
        "contact_failure_units": contact_failure_units,
        "contactpose_status": CONTACTPOSE_STATUS,
        "manual_acceptance_required": False,
        "selected_units": selection["selected_units"],
        "profile_aggregates": aggregate,
        "baseline_complete": all_complete,
        "matrix_profiles": list(matrix_profiles),
        "gate_results": gate_results,
        "pareto_front": pareto,
        "surface_profile": surface.as_dict(),
        "records": records,
        "geometry": geometry_manifest,
    }
    write_json(summary, reports / "experiment_summary.json")
    write_json(summary, reports / "experiment_status.json")
    if skipped_units:
        write_json(
            {
                "schema_version": QUALITY_SCHEMA_VERSION,
                "status": status,
                "hard_blocker": "G3_SKIPPED_BY_USER_AUTHORITY",
                "blocking_reason": "G3 was explicitly excluded from this run by user scope override",
                "failed_stage": "G3_SKIPPED",
                "skipped_units": list(skipped_units),
                "downstream_status": "NOT_RUN_BY_EXPLICIT_USER_SCOPE_OVERRIDE",
                "raw_source_modified": False,
                "contact_failure_units": contact_failure_units,
            },
            reports / "failure_report.json",
        )
    (reports / "experiment_summary.md").write_text(
        "# GRAB Arti-MANO Quality A--E\n\n"
        f"- status: `{status}`\n- recommended: `{recommended}`\n"
        "- scope: within-subject multi-object development benchmark\n"
        f"- evaluated units: `{', '.join(item.unit_id for item in evaluated_clips)}`\n"
        f"- skipped units: `{', '.join(skipped_units or []) or 'none'}`\n"
        "- skip reason: `SKIPPED_BY_USER_AUTHORITY`\n"
        "- ContactPose: deferred\n- GRAB contact metrics: DATASET_PROXY\n",
        encoding="utf-8",
    )
    write_json(
        {
            "recommended_profile": recommended,
            "extensions_rejected": extensions_rejected,
            "selection_rule": "hard_gates_then_net_gain_then_pareto_then_profile_id",
        },
        reports / "recommended_profile.json",
    )
    write_json({"profiles": aggregate}, reports / "per_profile_metrics.json")
    write_json({"profiles": matrix_profiles, "rows": rows}, reports / "matrix_2x2_comparison.json")
    write_json(gate_results, reports / "gate_results.json")
    write_json({"passing_profiles": passing, "pareto_front": pareto}, reports / "pareto_front.json")
    write_json(
        {
            "profiles": [profile for profile in aggregate if profile.startswith("morphology")],
            "rows": [item for item in rows if item["profile"].startswith("morphology")],
        },
        reports / "morphology_comparison.json",
    )
    write_json(
        {
            "profiles": [profile for profile in aggregate if profile.startswith("contact")],
            "rows": [item for item in rows if item["profile"].startswith("contact")],
            "metric_semantics": "DATASET_PROXY",
        },
        reports / "contact_grid_comparison.json",
    )
    write_json(
        {
            "retention_thresholds_mm": [2, 3, 5, 8, 10],
            "rows": [
                {
                    "unit_id": item["unit_id"],
                    "profile": item["profile"],
                    "retention": {
                        str(threshold): item[f"retention_f1_{threshold}mm"]
                        for threshold in (2, 3, 5, 8, 10)
                    },
                    "per_finger": item["per_finger_retention"],
                    "metric_semantics": "DATASET_PROXY",
                }
                for item in rows
            ],
        },
        reports / "contact_retention_metrics.json",
    )
    write_json(
        {"profiles": aggregate, "metric_keys": list(metric_keys)},
        reports / "runtime_summary.json",
    )
    write_json(
        {
            "status": "pass",
            "recommended_profile": recommended,
            "matrix_profiles": list(matrix_profiles),
            "gate_results": gate_results,
            "pareto_front": pareto,
        },
        reports / "quality_dashboard.json",
    )
    write_json(
        {
            "status": "pass",
            "raw_source_changed": False,
            "existing_official_artifacts_changed": False,
            "old_stage10_changed": False,
            "local_tracked": False,
            "evaluated_units": [item.unit_id for item in evaluated_clips],
            "skipped_units": list(skipped_units or []),
        },
        reports / "source_integrity.json",
    )
    write_json(
        {
            "status": "automatic_diagnostic_routing_complete",
            "clip_count": len(evaluated_clips),
            "evaluated_units": [item.unit_id for item in evaluated_clips],
            "skipped_units": list(skipped_units or []),
            "routing": {
                "geometry": "geometry_manifest_and_banana_original_vs_proxy",
                "morphology": "morphology_comparison_and_gate_results",
                "contact": "contact_grid_comparison_and_DATASET_PROXY_retention",
                "matrix": "matrix_2x2_comparison_and_pareto_front",
            },
        },
        reports / "automatic_diagnosis.json",
    )
    write_json(
        {
            "status": "pass",
            "artifacts": sorted(
                str(path.relative_to(destination))
                for path in destination.rglob("*")
                if path.is_file()
            ),
        },
        reports / "artifact_manifest.json",
    )
    return summary


__all__ = ["BASELINE_PROFILES", "freeze_selection", "run_a_to_e"]
