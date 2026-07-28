"""Stage 9.3.5 projection feasibility and causal-closure diagnostics.

This module is intentionally outside the formal Stage 9 solver path.  It
loads the current same-lineage baseline, evaluates the frozen canonical
reference-winding SDF, and writes only diagnostic artifacts below
``.local/runs/stage9_3_5_*`` and ``.local/reports/stage9_3_5``.

The projection state metric is a warm-centred diagnostic metric.  It reuses
the paper regularisation scales, but is not Eq. (8)/(9), is not a paper
method, and never produces an accepted reference trajectory.
"""

# The diagnostic report and embedded HTML intentionally contain long schema
# descriptions and compact report literals.  Keep the module's substantive
# lint checks enabled while matching the existing Stage 9 diagnostic modules.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.retarget.bones import extract_bone_features
from toporetarget.retarget.final_refinement import (
    PaperRefinementWeights,
    _make_context,
    dynamic_collision_points_numpy,
    load_final_trajectory,
    map_previous_state_to_seed,
    so3_log,
)
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.contact_audit import (
    TIP_INDICES,
    _load_inputs,
)
from toporetarget.workflows.shadow_equivalence import (
    FINGER_GROUPS,
    _bone_metrics,
    _contact_metrics,
    _interaction_metrics,
)

SCHEMA = "toporetarget.stage9_3_5.v1"
STATE_METRIC_SCHEMA = "toporetarget.projection_state_metric.v1"
PRESSURE_SCHEMA = "toporetarget.constraint_pressure.v1"
PROFILES = (
    "minimal_soft_safe_projection_from_warm_v2",
    "official_slack_projection_from_warm_v2",
)
LONG_FINGERS = ("index", "middle", "ring")
ALL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PATH_EPSILON_M = 1e-9
NEAR_BINDING = {
    "hard_0_1_mm": 1e-4,
    "soft_0_1_mm": 1e-4,
    "raw_0_5_mm": 5e-4,
}


class Stage935Error(RuntimeError):
    """Raised when a Stage 9.3.5 contract cannot be established."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        item = value.item()
        return item if not isinstance(item, float) or np.isfinite(item) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    savez_compressed: Any = np.savez_compressed
    savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for name, value in sha256_tree(path).items():
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(value.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _resolve(path: str | Path, repo: Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else repo / value).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "torch", "zarr", "trimesh"):
        try:
            module = __import__(name)
            packages[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "dtype": "float64",
        "device": "cpu",
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "path_workers": int(os.environ.get("STAGE935_PATH_WORKERS", "4")),
        "path_query_mode": "thread_shared_backend",
        "projection_maxiter": int(os.environ.get("STAGE935_PROJECTION_MAXITER", "200")),
    }


def _read_frame_selection(repo: Path) -> tuple[Path, dict[str, Any]]:
    matches = sorted(
        (repo / ".local/runs/stage9_3_4_provenance").glob("*/causal_frame_selection.json")
    )
    if len(matches) != 1:
        raise Stage935Error(
            f"expected exactly one Stage 9.3.4 causal_frame_selection.json; found {len(matches)}"
        )
    value = json.loads(matches[0].read_text(encoding="utf-8"))
    frames = [int(item) for item in value.get("frames", [])]
    if not frames or len(frames) > 5 or len(set(frames)) != len(frames):
        raise Stage935Error("causal frame selection must contain one to five unique frames")
    return matches[0], value


def _input_bundle(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    root = repo or _repo_root()
    lineage_path = _resolve(current_lineage_manifest, root)
    baseline_path = _resolve(current_baseline, root)
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if (
        lineage.get("lane") != "current"
        or lineage.get("schema_version") != "toporetarget.current_causal_lineage.v1"
    ):
        raise Stage935Error("current-lineage manifest is not the Stage 9.3.4 current lane")
    if not baseline_path.is_dir():
        raise Stage935Error(f"current baseline does not exist: {baseline_path}")
    run_manifest = _resolve(str(lineage["run_manifest"]), root)
    inputs = _load_inputs(run_manifest, root, evaluation_backend="reference_winding_v1")
    baseline = load_final_trajectory(baseline_path)
    if baseline.frame_count != 60:
        raise Stage935Error(f"current baseline frame count is {baseline.frame_count}, expected 60")
    inputs["final"] = baseline
    # Reuse the already audited Stage 9.3.4 solver-side backend for diagnostic
    # optimization callbacks.  The canonical reference backend remains the
    # independent validation path below; it is intentionally not replaced.
    from toporetarget.retarget.final_refinement import (
        RefinementSolverProfile,
        choose_solver_sdf_backend,
    )

    solver_profile_id = str(
        baseline.metadata.get("solver_profile_id", "scipy_slsqp_active_set_contact_rich_v2")
    )
    solver_profile = RefinementSolverProfile.load(solver_profile_id, root)
    solver_sdf, solver_sdf_report = choose_solver_sdf_backend(
        inputs["object"].mesh.vertices_local,
        inputs["object"].mesh.faces,
        inputs["reference_sdf"],
        solver_profile,
        object_pose_scene=np.asarray(inputs["object"].pose_scene.pose_scene[0], dtype=np.float64),
        tree_leaf_size=512,
    )
    inputs["solver_sdf"] = solver_sdf
    inputs["solver_sdf_report"] = solver_sdf_report
    selection_path, selection = _read_frame_selection(root)
    artifacts: dict[str, Any] = {}
    for name, item in dict(lineage.get("input_artifacts", {})).items():
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = _resolve(str(item["path"]), root)
        if not path.exists():
            raise Stage935Error(f"lineage artifact missing: {path}")
        actual = _sha(path)
        artifacts[name] = {
            "path": str(path),
            "declared_hash": item.get("content_hash"),
            "actual_hash": actual,
            "identity_pass": item.get("content_hash") in {None, actual},
        }
        if not artifacts[name]["identity_pass"]:
            raise Stage935Error(f"lineage artifact hash mismatch: {name}")
    baseline_hash = _sha(baseline_path)
    lineage_hash = _stable_hash(
        {
            "lineage_manifest": lineage,
            "baseline_hash": baseline_hash,
            "selection": selection,
        }
    )
    canonical_profile = (
        root
        / ".local/runs/stage9_3_2_canonical_reaudit/s1__airplane_lift__right__artimano_rh__f000240_f000300/canonical_backend_profile.json"
    )
    identity = {
        "schema_version": "toporetarget.stage9_3_5_input_identity.v1",
        "current_causal_lineage_hash": lineage_hash,
        "source_hash": artifacts.get("canonical", {}).get("actual_hash"),
        "warm_hash": artifacts.get("warm_start", {}).get("actual_hash"),
        "graph_hash": artifacts.get("graph", {}).get("actual_hash"),
        "baseline_hash": baseline_hash,
        "previous_final_hash": str(baseline.metadata.get("artifact_hash", "")),
        "collision_sample_hash": artifacts.get("collision_samples", {}).get("actual_hash"),
        "object_mesh_hash": str(baseline.metadata.get("object_mesh_hash", "")),
        "solver_profile": baseline.metadata.get("solver_profile", {}),
        "solver_profile_hash": baseline.metadata.get("solver_profile_hash"),
        "solver_profile_id": baseline.metadata.get("solver_profile_id"),
        "execution_profile_id": baseline.metadata.get("execution_profile_id"),
        "execution_profile_hash": baseline.metadata.get("execution_profile_hash"),
        "query_profile": baseline.metadata.get("query_profile", {}),
        "canonical_sdf_profile": (
            json.loads(canonical_profile.read_text(encoding="utf-8"))
            if canonical_profile.is_file()
            else inputs["reference_sdf"].describe()
        ),
        "solver_sdf_profile": inputs["solver_sdf"].describe(),
        "solver_sdf_selection_report": inputs["solver_sdf_report"],
        "environment": _environment(),
        "dtype": "float64",
        "threads": _environment()["threads"],
        "lineage_manifest": str(lineage_path),
        "current_baseline": str(baseline_path),
        "stage10_manifest": str(run_manifest),
        "causal_frame_selection": str(selection_path),
        "artifact_identity": artifacts,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    return {
        "repo": root,
        "lineage_path": lineage_path,
        "lineage": lineage,
        "baseline_path": baseline_path,
        "inputs": inputs,
        "selection_path": selection_path,
        "selection": selection,
        "frames": tuple(int(item) for item in selection["frames"]),
        "identity": identity,
        "lineage_hash": lineage_hash,
    }


def _official_artifact_snapshot(bundle: dict[str, Any]) -> dict[str, Any]:
    """Capture the official Stage 5--10 boundary without writing to it."""

    paths: list[tuple[str, Path, str]] = [
        ("current_lineage_manifest", bundle["lineage_path"], "current-lineage manifest"),
        ("current_baseline", bundle["baseline_path"], "Stage 9.3.4 current baseline"),
        ("causal_frame_selection", bundle["selection_path"], "selected causal frames"),
        (
            "stage10_manifest",
            Path(str(bundle["identity"]["stage10_manifest"])),
            "Stage 10 manifest",
        ),
    ]
    for name, item in bundle["identity"].get("artifact_identity", {}).items():
        paths.append(
            (f"lineage_artifact:{name}", Path(str(item["path"])), "lineage input artifact")
        )
    stage10_manifest = Path(str(bundle["identity"]["stage10_manifest"]))
    if stage10_manifest.is_file():
        stage10 = json.loads(stage10_manifest.read_text(encoding="utf-8"))
        manual = stage10.get("manual_acceptance", {})
        if manual.get("path"):
            paths.append(
                ("manual_acceptance", Path(str(manual["path"])), "Stage 10 manual acceptance")
            )
        runtime = stage10.get("runtime_acceptance", {})
        if runtime.get("path"):
            paths.append(
                ("runtime_acceptance", Path(str(runtime["path"])), "Stage 10 runtime acceptance")
            )
        for name, value in stage10.get("export_paths", {}).items():
            if value:
                paths.append(
                    (f"robot_reference:{name}", Path(str(value)), "robot reference export")
                )
    canonical_profile = (
        bundle["repo"]
        / ".local/runs/stage9_3_2_canonical_reaudit/s1__airplane_lift__right__artimano_rh__f000240_f000300/canonical_backend_profile.json"
    )
    if canonical_profile.is_file():
        paths.append(
            ("stage9_3_2_canonical_profile", canonical_profile, "Stage 9.3.2 canonical audit")
        )
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for label, path, role in paths:
        resolved = path.expanduser().resolve()
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if not resolved.exists():
            entries.append({"label": label, "path": str(resolved), "role": role, "exists": False})
            continue
        entries.append(
            {
                "label": label,
                "path": str(resolved),
                "role": role,
                "exists": True,
                "sha256": _sha(resolved),
                "mtime_ns": int(resolved.stat().st_mtime_ns),
            }
        )
    return {
        "schema_version": "toporetarget.stage9_3_5_official_artifact_snapshot.v1",
        "scope": "Stage 5--10 official/current-lineage inputs; diagnostic outputs excluded",
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "entries": entries,
        "diagnostic_only": True,
    }


def _compare_official_artifact_snapshots(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    before_by_path = {str(row.get("path")): row for row in before.get("entries", [])}
    after_by_path = {str(row.get("path")): row for row in after.get("entries", [])}
    entries: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for path in sorted(set(before_by_path) | set(after_by_path)):
        left = before_by_path.get(path, {})
        right = after_by_path.get(path, {})
        unchanged = bool(
            left.get("exists")
            and right.get("exists")
            and left.get("sha256") == right.get("sha256")
            and left.get("mtime_ns") == right.get("mtime_ns")
        )
        row = {
            "path": path,
            "label": right.get("label", left.get("label")),
            "before_sha256": left.get("sha256"),
            "after_sha256": right.get("sha256"),
            "before_mtime_ns": left.get("mtime_ns"),
            "after_mtime_ns": right.get("mtime_ns"),
            "unchanged": unchanged,
        }
        entries.append(row)
        if not unchanged:
            changed.append(row)
    return {
        "schema_version": "toporetarget.stage9_3_5_official_artifact_immutability.v1",
        "official_artifacts_changed": bool(changed),
        "entries": entries,
        "changed": changed,
        "diagnostic_only": True,
    }


def _frame_selection_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    reasons = {
        int(item): "causal_frame_selection" for item in bundle["selection"].get("frames", [])
    }
    return {
        "schema_version": "toporetarget.stage9_3_5_projection_frame_selection.v1",
        "source": str(bundle["selection_path"]),
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "frames": [
            {
                "local_frame": int(frame),
                "global_frame": 240 + int(frame),
                "reason": reasons.get(int(frame), "Stage 9.3.4 fixed selection"),
            }
            for frame in bundle["frames"]
        ],
        "selection_changed_by_projection": False,
        "max_frames": 5,
    }


def _paper(bundle: dict[str, Any]) -> PaperRefinementWeights:
    paper = PaperRefinementWeights.load(bundle["repo"])
    recorded = bundle["inputs"]["final"].metadata.get("paper_weights", {})
    if recorded and str(recorded.get("config_hash", paper.config_hash)) != paper.config_hash:
        raise Stage935Error("paper weight profile does not match current baseline")
    return paper


def _previous_reference(bundle: dict[str, Any], frame: int) -> np.ndarray | None:
    final = bundle["inputs"]["final"]
    warm = bundle["inputs"]["warm"]
    if frame <= 0:
        return None
    return map_previous_state_to_seed(
        final.arrays["base_pose_scene"][frame - 1],
        final.arrays["qpos"][frame - 1],
        warm.arrays["base_pose_scene"][frame],
    )


def _context(bundle: dict[str, Any], frame: int) -> Any:
    inputs = bundle["inputs"]
    return _make_context(
        inputs["sequence"],
        inputs["graph"],
        inputs["warm"],
        inputs["model"],
        inputs["surface"],
        inputs.get("solver_sdf", inputs["reference_sdf"]),
        inputs["reference_sdf"],
        inputs["frame_profile"],
        inputs["bone_profile"],
        _paper(bundle),
        frame,
        _previous_reference(bundle, frame),
    )


def _base_pose_from_value(context: Any, value: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    item = np.asarray(value, dtype=np.float64).reshape(-1)
    base = np.asarray(context.seed_base, dtype=np.float64).copy()
    base[:3, :3] = Rotation.from_rotvec(item[3:6]).as_matrix() @ base[:3, :3]
    base[:3, 3] += item[:3]
    return base


def _state_value(
    context: Any, base: np.ndarray, qpos: np.ndarray, slack: np.ndarray | None = None
) -> np.ndarray:
    base_value = np.asarray(base, dtype=np.float64)
    seed = np.asarray(context.seed_base, dtype=np.float64)
    delta = np.concatenate(
        [
            base_value[:3, 3] - seed[:3, 3],
            so3_log(base_value[:3, :3] @ seed[:3, :3].T),
            np.asarray(qpos, dtype=np.float64),
        ]
    )
    if slack is None:
        slack = np.zeros(512, dtype=np.float64)
    return np.concatenate([delta, np.asarray(slack, dtype=np.float64).reshape(-1)])


def _full_slack(trajectory: Any, frame: int, sample_count: int = 512) -> np.ndarray:
    result = np.zeros(sample_count, dtype=np.float64)
    start = int(trajectory.arrays["slack_offsets"][frame])
    stop = int(trajectory.arrays["slack_offsets"][frame + 1])
    ids_start = int(trajectory.arrays["query_offsets"][frame])
    ids_stop = int(trajectory.arrays["query_offsets"][frame + 1])
    ids = np.asarray(trajectory.arrays["query_ids_concat"][ids_start:ids_stop], dtype=np.int64)
    slack = np.asarray(trajectory.arrays["slack_concat"][start:stop], dtype=np.float64)
    if len(ids) != len(slack):
        raise Stage935Error("baseline query/slack offsets are inconsistent")
    result[ids] = slack
    return result


def _pose_valid(base: np.ndarray) -> bool:
    rotation = np.asarray(base, dtype=np.float64)[:3, :3]
    return bool(
        np.all(np.isfinite(base))
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8)
    )


def _evaluate_state(
    bundle: dict[str, Any],
    frame: int,
    base: np.ndarray,
    qpos: np.ndarray,
    slack: np.ndarray | None = None,
) -> dict[str, Any]:
    inputs = bundle["inputs"]
    paper = _paper(bundle)
    context = _context(bundle, frame)
    full_slack = np.zeros(512, dtype=np.float64)
    if slack is not None:
        supplied_slack = np.asarray(slack, dtype=np.float64).reshape(-1)
        if len(supplied_slack) == 512:
            full_slack = supplied_slack
        elif len(supplied_slack) not in {0, 6 + inputs["model"].num_dofs}:
            raise Stage935Error(f"unexpected slack width {len(supplied_slack)}")
    value = _state_value(context, base, qpos, full_slack)
    points = dynamic_collision_points_numpy(inputs["model"], inputs["surface"], qpos, base)
    pose = inputs["object"].pose_scene.pose_scene[frame]
    query = inputs["reference_sdf"].query_scene(points, pose)
    phi = np.asarray(query.signed_distance, dtype=np.float64)
    q = np.asarray(qpos, dtype=np.float64)
    source = np.asarray(inputs["source_keypoints"][frame], dtype=np.float64)
    keypoints = np.asarray(
        inputs["model"].keypoints_scene(q, base, layout="mediapipe21"), dtype=np.float64
    )
    errors = np.linalg.norm(keypoints - source, axis=1)
    eim, eim_finger = _interaction_metrics(inputs, frame, keypoints)
    ebone_finger = _bone_metrics(inputs, frame, keypoints)
    terms = _term_values(context, value)
    morphology: dict[str, float] = {}
    for finger in ALL_FINGERS:
        ids = list(FINGER_GROUPS[finger])
        mcp = ids[0]
        tip = int(TIP_INDICES[finger])
        scale = max(float(np.linalg.norm(source[tip] - source[mcp])), 1e-9)
        morphology[finger] = float(np.sqrt(np.mean(errors[ids] ** 2)) / scale)
    hard = phi + paper.b
    soft = phi + full_slack + paper.tau
    contact = _contact_metrics(inputs, frame, keypoints, pose)
    per_finger = {
        finger: {
            "keypoint_rmse_m": float(np.sqrt(np.mean(errors[list(FINGER_GROUPS[finger])] ** 2))),
            "morphology_normalized_rmse": morphology[finger],
            "fingertip_error_m": float(errors[int(TIP_INDICES[finger])]),
            "pad_proxy_distance_m": contact.get("per_finger", {})
            .get(finger, {})
            .get("state_min_distance_m"),
        }
        for finger in ALL_FINGERS
    }
    return {
        "frame": int(frame),
        "base_pose_scene": np.asarray(base, dtype=np.float64),
        "qpos": q,
        "slack": full_slack,
        "keypoints_scene": keypoints,
        "per_finger": per_finger,
        "long_finger_rmse_m": float(
            np.mean([per_finger[f]["keypoint_rmse_m"] for f in LONG_FINGERS])
        ),
        "long_finger_morphology_normalized_rmse": float(
            np.mean([morphology[f] for f in LONG_FINGERS])
        ),
        "e_im_raw": float(eim),
        "e_im_per_finger": eim_finger,
        "e_bone_per_finger": ebone_finger,
        "terms": terms,
        "full_signed_distance": phi,
        "full_hard_residual": hard,
        "full_soft_residual": soft,
        "min_sdf_m": float(np.min(phi)),
        "raw_penetration_m": float(max(0.0, -np.min(phi))),
        "hard_violation_m": float(max(0.0, -np.min(hard))),
        "soft_violation_m": float(max(0.0, -np.min(soft))),
        "slack_bounds_pass": bool(
            np.all((full_slack >= -1e-10) & (full_slack <= paper.b - paper.tau + 1e-10))
        ),
        "qpos_bounds_pass": bool(
            np.all(q >= inputs["model"].joint_lower - 1e-10)
            and np.all(q <= inputs["model"].joint_upper + 1e-10)
        ),
        "base_valid": _pose_valid(base),
        "full512_finite": bool(np.all(np.isfinite(phi))),
        "contact_proxy": contact.get("contact_proxy"),
        "contact": contact,
        "base_translation_from_warm_m": float(
            np.linalg.norm(
                np.asarray(base)[:3, 3] - inputs["warm"].arrays["base_pose_scene"][frame][:3, 3]
            )
        ),
        "base_rotation_from_warm_rad": float(
            np.linalg.norm(
                so3_log(
                    np.asarray(base)[:3, :3]
                    @ np.asarray(inputs["warm"].arrays["base_pose_scene"][frame])[:3, :3].T
                )
            )
        ),
        "qpos_displacement_from_warm_l2": float(
            np.linalg.norm(q - inputs["warm"].arrays["qpos"][frame])
        ),
        "state_displacement": float(_state_metric_value(context, value)),
        "value": value,
    }


def _term_tensors(context: Any, value: Any) -> dict[str, Any]:
    import torch

    delta_p, delta_w, qpos, slack = context.unpack(value)
    base = _base_pose_torch(context, value)
    keypoints = context.robot_model.keypoints_scene(qpos, base, layout="mediapipe21")
    object_points = torch.as_tensor(
        context.graph_frame.source_vertices[21:], dtype=value.dtype, device=value.device
    )
    vertices = torch.cat(
        [keypoints, object_points.expand(*keypoints.shape[:-2], *object_points.shape)], dim=-2
    )
    residual = context._residual_model(vertices)
    e_im = residual.square().sum(dim=(-2, -1)) / 71.0
    features = extract_bone_features(
        keypoints,
        context.frame_profile,
        context.bone_profile,
        side=context.robot_model.side,
        strict=True,
    )
    source = torch.as_tensor(
        context.source_features.adjacent_features, dtype=value.dtype, device=value.device
    )
    e_bone = (features.adjacent_features - source).square().sum(dim=(-2, -1))
    if (
        context.previous_reference is None
        or getattr(context, "temporal_scope", "base_and_finger") == "none"
    ):
        e_temporal = value[..., 0] * 0.0
    else:
        previous = torch.as_tensor(
            context.previous_reference, dtype=value.dtype, device=value.device
        )
        scope = getattr(context, "temporal_scope", "base_and_finger")
        if scope == "finger_only":
            current_delta = value[..., 6 : context.variable_size_without_slack]
            previous_delta = previous[6 : context.variable_size_without_slack]
        elif scope == "base_only":
            current_delta = value[..., :6]
            previous_delta = previous[:6]
        else:
            current_delta = value[..., : context.variable_size_without_slack]
            previous_delta = previous[: context.variable_size_without_slack]
        e_temporal = context.paper.lambda_reg * (current_delta - previous_delta).square().sum(
            dim=-1
        )
    e_base_pos = context.paper.lambda_base_pos * delta_p.square().sum(dim=-1)
    e_base_rot = context.paper.lambda_base_rot * delta_w.square().sum(dim=-1)
    e_slack = 0.5 * context.paper.w_s * slack.square().sum(dim=-1)
    return {
        "e_im": e_im,
        "e_bone": e_bone,
        "e_temporal": e_temporal,
        "e_base_pos": e_base_pos,
        "e_base_rot": e_base_rot,
        "e_slack": e_slack,
        "weighted_e_im": context.paper.lambda_im * e_im,
        "weighted_e_bone": context.paper.lambda_bone * e_bone,
    }


def _base_pose_torch(context: Any, value: Any) -> Any:
    import torch

    delta_p, delta_w, _, _ = context.unpack(value)
    seed = torch.as_tensor(context.seed_base, dtype=value.dtype, device=value.device)
    rotation = _so3_exp_torch(delta_w) @ seed[:3, :3]
    shape = tuple(delta_p.shape[:-1])
    eye = torch.eye(4, dtype=value.dtype, device=value.device)
    base = eye.expand(*shape, 4, 4).clone()
    base[..., :3, :3] = rotation
    base[..., :3, 3] = seed[:3, 3] + delta_p
    return base


def _so3_exp_torch(value: Any) -> Any:
    import torch

    theta = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    k = torch.zeros((*value.shape[:-1], 3, 3), dtype=value.dtype, device=value.device)
    k[..., 0, 1] = -value[..., 2]
    k[..., 0, 2] = value[..., 1]
    k[..., 1, 0] = value[..., 2]
    k[..., 1, 2] = -value[..., 0]
    k[..., 2, 0] = -value[..., 1]
    k[..., 2, 1] = value[..., 0]
    theta2 = theta.square()
    safe = torch.clamp(theta, min=1e-8)
    safe2 = torch.clamp(theta2, min=1e-16)
    a = torch.where(theta > 1e-8, torch.sin(theta) / safe, 1.0 - theta2 / 6.0)
    b = torch.where(theta > 1e-8, (1.0 - torch.cos(theta)) / safe2, 0.5 - theta2 / 24.0)
    eye = torch.eye(3, dtype=value.dtype, device=value.device).expand(*value.shape[:-1], 3, 3)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def _collision_points_torch(context: Any, value: Any) -> Any:
    import torch

    _, _, qpos, _ = context.unpack(value)
    fk = context.robot_model.forward_kinematics_base(qpos)
    pieces: list[Any] = []
    for geometry_index, start, stop in context.geometry_slices:
        points = torch.as_tensor(
            context.surface_points_local[start:stop], dtype=value.dtype, device=value.device
        )
        local = torch.as_tensor(
            context.surface_local_transforms[geometry_index], dtype=value.dtype, device=value.device
        )
        points = points @ local[:3, :3].transpose(-1, -2) + local[:3, 3]
        link = fk[context.surface_link_names[geometry_index]]
        if link.ndim > 2:
            points = points.expand(*link.shape[:-2], *points.shape)
            translation = link[..., :3, 3][..., None, :]
        else:
            translation = link[..., :3, 3]
        points = points @ link[..., :3, :3].transpose(-1, -2) + translation
        pieces.append(points)
    points = torch.cat(pieces, dim=-2)
    base = _base_pose_torch(context, value)
    return points @ base[..., :3, :3].transpose(-1, -2) + base[..., None, :3, 3]


def _term_values(context: Any, value: np.ndarray) -> dict[str, float]:
    import torch

    variable = torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)
    tensors = _term_tensors(context, variable)
    return {name: float(item.detach().cpu()) for name, item in tensors.items()}


def _state_metric_value(context: Any, value: np.ndarray) -> float:
    item = np.asarray(value, dtype=np.float64)
    dp, dw, q, slack = context.unpack(item)
    return float(
        0.5 * context.paper.lambda_reg * np.sum((q - context.seed_qpos) ** 2)
        + 0.5 * context.paper.lambda_base_pos * np.sum(dp**2)
        + 0.5 * context.paper.lambda_base_rot * np.sum(dw**2)
        + 0.5 * context.paper.w_s * np.sum(slack**2)
    )


def _base_from_path(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    start_value = np.asarray(start, dtype=np.float64)
    end_value = np.asarray(end, dtype=np.float64)
    result = start_value.copy()
    relative = so3_log(end_value[:3, :3] @ start_value[:3, :3].T)
    result[:3, :3] = Rotation.from_rotvec(float(alpha) * relative).as_matrix() @ start_value[:3, :3]
    result[:3, 3] = (1.0 - float(alpha)) * start_value[:3, 3] + float(alpha) * end_value[:3, 3]
    return result


def _path_intervals(mask: np.ndarray) -> list[tuple[float, float]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if not len(values):
        return []
    result: list[tuple[float, float]] = []
    start: int | None = None
    denominator = max(len(values) - 1, 1)
    for index, current in enumerate(np.concatenate([values, [False]])):
        if current and start is None:
            start = index
        elif not current and start is not None:
            result.append((start / denominator, (index - 1) / denominator))
            start = None
    return result


def _bisect_boundary(
    predicate: Any,
    left: float,
    right: float,
    left_value: bool,
    *,
    iterations: int = 60,
) -> float:
    left_value = bool(left_value)
    for _ in range(iterations):
        middle = 0.5 * (float(left) + float(right))
        if bool(predicate(middle)) == left_value:
            left = middle
        else:
            right = middle
    return 0.5 * (float(left) + float(right))


def _finger_from_link(link: str) -> str:
    normalized = str(link).lower()
    for finger in ALL_FINGERS:
        if finger in normalized:
            return finger
    if "palm" in normalized or "wrist" in normalized:
        return "palm"
    return "unknown"


def _projection_objective(
    bundle: Any,
    value: np.ndarray,
    profile: str,
) -> tuple[float, np.ndarray]:
    item = np.asarray(value, dtype=np.float64).reshape(-1)
    if isinstance(bundle, dict):
        paper = bundle["paper"] if "paper" in bundle else _paper(bundle)
        warm_q = np.asarray(
            bundle.get("projection_warm_q", bundle["inputs"]["warm"].arrays["qpos"]),
            dtype=np.float64,
        ).reshape(-1)
    else:
        paper = bundle.paper
        warm_q = np.asarray(bundle.projection_warm_q, dtype=np.float64).reshape(-1)
    q_stop = 6 + len(warm_q)
    q_delta = item[6:q_stop] - warm_q
    slack = item[q_stop:] if str(profile).startswith("official_slack") else np.empty(0)
    objective = 0.5 * (
        float(paper.lambda_reg) * np.sum(q_delta**2)
        + float(paper.lambda_base_pos) * np.sum(item[:3] ** 2)
        + float(paper.lambda_base_rot) * np.sum(item[3:6] ** 2)
        + float(paper.w_s) * np.sum(slack**2)
    )
    gradient = np.zeros_like(item)
    gradient[:3] = float(paper.lambda_base_pos) * item[:3]
    gradient[3:6] = float(paper.lambda_base_rot) * item[3:6]
    gradient[6:q_stop] = float(paper.lambda_reg) * q_delta
    if len(slack):
        gradient[q_stop:] = float(paper.w_s) * slack
    return float(objective), gradient


def _path_states(
    bundle: dict[str, Any], frame: int, sample_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    inputs = bundle["inputs"]
    warm_base = np.asarray(inputs["warm"].arrays["base_pose_scene"][frame], dtype=np.float64)
    warm_q = np.asarray(inputs["warm"].arrays["qpos"][frame], dtype=np.float64)
    final = inputs["final"]
    final_base = np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)
    final_q = np.asarray(final.arrays["qpos"][frame], dtype=np.float64)
    alphas = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float64)
    relative = so3_log(final_base[:3, :3] @ warm_base[:3, :3].T)
    bases = np.empty((len(alphas), 4, 4), dtype=np.float64)
    bases[:] = warm_base
    for index, alpha in enumerate(alphas):
        bases[index, :3, :3] = (
            Rotation.from_rotvec(alpha * relative).as_matrix() @ warm_base[:3, :3]
        )
        bases[index, :3, 3] = (1.0 - alpha) * warm_base[:3, 3] + alpha * final_base[:3, 3]
    qpos = (1.0 - alphas[:, None]) * warm_q[None, :] + alphas[:, None] * final_q[None, :]
    return alphas, bases, qpos


def _path_metrics(bundle: dict[str, Any], frame: int, sample_count: int = 1001) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    import torch

    inputs = bundle["inputs"]
    paper = _paper(bundle)
    context = _context(bundle, frame)
    alphas, bases, qpos = _path_states(bundle, frame, sample_count)
    values = np.stack([_state_value(context, bases[i], qpos[i]) for i in range(len(alphas))])
    tensor_values = torch.as_tensor(values, dtype=torch.float64)
    with torch.no_grad():
        points = _collision_points_torch(context, tensor_values).detach().cpu().numpy()
        keypoints = inputs["model"].keypoints_scene(qpos, bases, layout="mediapipe21")
        keypoints = np.asarray(
            keypoints.detach().cpu() if hasattr(keypoints, "detach") else keypoints,
            dtype=np.float64,
        )
        terms = _term_tensors(context, tensor_values)
        term_arrays = {
            name: np.asarray(item.detach().cpu(), dtype=np.float64) for name, item in terms.items()
        }
    pose = inputs["object"].pose_scene.pose_scene[frame]
    backend = inputs["reference_sdf"]
    worker_count = min(
        max(1, int(os.environ.get("STAGE935_PATH_WORKERS", "4"))),
        len(alphas),
    )
    phi = np.empty((len(alphas), 512), dtype=np.float64)
    chunk_size = 16
    chunks = [
        (start, min(start + chunk_size, len(alphas))) for start in range(0, len(alphas), chunk_size)
    ]

    def query_chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
        start, stop = bounds
        query = inputs["reference_sdf"].query_scene(points[start:stop], pose)
        return start, np.asarray(query.signed_distance, dtype=np.float64)

    original_query_chunk = int(backend.query_chunk_size)
    original_face_chunk = int(backend.face_chunk_size)
    path_query_chunk = int(os.environ.get("STAGE935_PATH_QUERY_CHUNK", str(original_query_chunk)))
    path_face_chunk = int(os.environ.get("STAGE935_PATH_FACE_CHUNK", str(original_face_chunk)))
    backend.query_chunk_size = path_query_chunk
    backend.face_chunk_size = path_face_chunk
    try:
        if worker_count == 1:
            results = [query_chunk(bounds) for bounds in chunks]
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(query_chunk, chunks))
    finally:
        backend.query_chunk_size = original_query_chunk
        backend.face_chunk_size = original_face_chunk
    for start, values in results:
        phi[start : start + len(values)] = values
    source = np.asarray(inputs["source_keypoints"][frame], dtype=np.float64)
    errors = np.linalg.norm(keypoints - source[None, :, :], axis=-1)
    long_rmse = np.mean(
        [np.sqrt(np.mean(errors[:, list(FINGER_GROUPS[f])] ** 2, axis=1)) for f in LONG_FINGERS],
        axis=0,
    )
    finger_rmse = {
        finger: np.sqrt(np.mean(errors[:, list(FINGER_GROUPS[finger])] ** 2, axis=1))
        for finger in ALL_FINGERS
    }
    required_slack = np.clip(-paper.tau - phi, 0.0, paper.b - paper.tau)
    hard_legal = np.min(phi, axis=1) >= -paper.b - PATH_EPSILON_M
    soft_legal = np.min(phi, axis=1) >= -paper.tau - PATH_EPSILON_M
    zero_legal = np.min(phi, axis=1) >= -PATH_EPSILON_M
    path_rows: list[dict[str, Any]] = []
    for index, alpha in enumerate(alphas):
        path_rows.append(
            {
                "alpha": float(alpha),
                "min_sdf_m": float(np.min(phi[index])),
                "hard_residual_min_m": float(np.min(phi[index] + paper.b)),
                "soft_residual_min_zero_slack_m": float(np.min(phi[index] + paper.tau)),
                "required_slack_max_m": float(np.max(required_slack[index])),
                "hard_feasible": bool(hard_legal[index]),
                "soft_safe_feasible": bool(soft_legal[index]),
                "zero_penetration_feasible": bool(zero_legal[index]),
                "long_finger_rmse_m": float(long_rmse[index]),
                **{
                    f"{finger}_rmse_m": float(values[index])
                    for finger, values in finger_rmse.items()
                },
                "e_im_raw": float(term_arrays["e_im"][index]),
                "e_bone_raw": float(term_arrays["e_bone"][index]),
                "formal_total_objective_zero_slack": float(
                    term_arrays["weighted_e_im"][index]
                    + term_arrays["weighted_e_bone"][index]
                    + term_arrays["e_temporal"][index]
                    + term_arrays["e_base_pos"][index]
                    + term_arrays["e_base_rot"][index]
                ),
                "formal_total_objective_minimal_legal_slack": float(
                    term_arrays["weighted_e_im"][index]
                    + term_arrays["weighted_e_bone"][index]
                    + term_arrays["e_temporal"][index]
                    + term_arrays["e_base_pos"][index]
                    + term_arrays["e_base_rot"][index]
                    + 0.5 * paper.w_s * np.sum(required_slack[index] ** 2)
                ),
            }
        )

    def predicate(alpha: float, kind: str) -> bool:
        from scipy.spatial.transform import Rotation

        a = float(alpha)
        base = np.asarray(bases[0]).copy()
        relative = so3_log(bases[-1, :3, :3] @ bases[0, :3, :3].T)
        base[:3, :3] = Rotation.from_rotvec(a * relative).as_matrix() @ bases[0, :3, :3]
        base[:3, 3] = (1.0 - a) * bases[0, :3, 3] + a * bases[-1, :3, 3]
        q = (1.0 - a) * qpos[0] + a * qpos[-1]
        p = dynamic_collision_points_numpy(inputs["model"], inputs["surface"], q, base)
        minimum = float(np.min(inputs["reference_sdf"].query_scene(p, pose).signed_distance))
        threshold = {"soft": -paper.tau, "hard": -paper.b, "zero": 0.0}[kind]
        return minimum >= threshold - PATH_EPSILON_M

    def refine_boundary(left: float, right: float, kind: str, left_ok: bool) -> float:
        for _ in range(12):
            mid = 0.5 * (left + right)
            if predicate(mid, kind) == left_ok:
                left = mid
            else:
                right = mid
        return float(0.5 * (left + right))

    def intervals(mask: np.ndarray, kind: str) -> list[dict[str, float]]:
        result: list[dict[str, float]] = []
        start: int | None = None
        for index, ok in enumerate(mask.tolist() + [False]):
            if ok and start is None:
                start = index
            elif not ok and start is not None:
                stop = index - 1
                left = float(alphas[start])
                right = float(alphas[stop])
                if start > 0:
                    left = refine_boundary(float(alphas[start - 1]), left, kind, False)
                if stop < len(alphas) - 1:
                    right = refine_boundary(right, float(alphas[stop + 1]), kind, True)
                result.append({"start_alpha": left, "end_alpha": right})
                start = None
        return result

    per_sample_order: list[dict[str, Any]] = []
    names = np.asarray(inputs["surface"].link_names).astype(str)
    for sample in range(512):
        valid = np.flatnonzero(phi[:, sample] >= -paper.tau - PATH_EPSILON_M)
        hard_valid = np.flatnonzero(phi[:, sample] >= -paper.b - PATH_EPSILON_M)
        per_sample_order.append(
            {
                "sample_id": sample,
                "link": str(names[sample]),
                "first_soft_safe_alpha_grid": None if not len(valid) else float(alphas[valid[0]]),
                "first_hard_feasible_alpha_grid": None
                if not len(hard_valid)
                else float(alphas[hard_valid[0]]),
                "warm_soft_violation_m": float(max(0.0, -phi[0, sample] - paper.tau)),
                "warm_hard_violation_m": float(max(0.0, -phi[0, sample] - paper.b)),
            }
        )
    non_monotone = {
        "soft": bool(np.any(np.diff(soft_legal.astype(np.int8)) < 0)),
        "hard": bool(np.any(np.diff(hard_legal.astype(np.int8)) < 0)),
        "zero": bool(np.any(np.diff(zero_legal.astype(np.int8)) < 0)),
    }
    return {
        "schema_version": "toporetarget.stage9_3_5_warm_final_path.v1",
        "frame": int(frame),
        "global_frame": int(240 + frame),
        "sample_count": int(sample_count),
        "alphas": alphas,
        "bases": bases,
        "qpos": qpos,
        "phi": phi,
        "required_slack": required_slack,
        "rows": path_rows,
        "soft_safe_intervals": intervals(soft_legal, "soft"),
        "hard_feasible_intervals": intervals(hard_legal, "hard"),
        "zero_penetration_intervals": intervals(zero_legal, "zero"),
        "first_soft_safe_alpha": next(
            (float(alphas[i]) for i, item in enumerate(soft_legal) if item), None
        ),
        "first_hard_feasible_alpha": next(
            (float(alphas[i]) for i, item in enumerate(hard_legal) if item), None
        ),
        "first_zero_penetration_alpha": next(
            (float(alphas[i]) for i, item in enumerate(zero_legal) if item), None
        ),
        "warm_soft_feasible": bool(soft_legal[0]),
        "warm_hard_feasible": bool(hard_legal[0]),
        "warm_zero_penetration": bool(zero_legal[0]),
        "official_final_soft_feasible": bool(soft_legal[-1]),
        "official_final_hard_feasible": bool(hard_legal[-1]),
        "official_final_zero_penetration": bool(zero_legal[-1]),
        "non_monotonicity": non_monotone,
        "per_sample_violation_order": per_sample_order,
        "query_schedule": {
            "canonical_query_chunk_size": original_query_chunk,
            "canonical_face_chunk_size": original_face_chunk,
            "path_query_chunk_size": path_query_chunk,
            "path_face_chunk_size": path_face_chunk,
            "workers": worker_count,
            "backend_math": "reference_triangle_winding",
        },
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }


def _path_metrics_from_cache(
    bundle: dict[str, Any], frame: int, cache_path: Path, sample_count: int
) -> dict[str, Any]:
    """Rebuild compact path diagnostics from a validated full-512 cache.

    The cache contains the expensive canonical SDF samples.  Recomputing the
    state terms and the small number of refined interval boundaries is cheap
    compared with querying all 1001 path states again.  This is deliberately
    separate from ``_path_metrics`` so resume cannot silently fall back to a
    different collision backend.
    """
    import torch
    from scipy.spatial.transform import Rotation

    with np.load(cache_path, allow_pickle=False) as data:
        required = {"alphas", "bases", "qpos", "phi", "required_slack"}
        if not required.issubset(data.files):
            raise Stage935Error(f"path cache is missing required arrays: {cache_path}")
        alphas = np.asarray(data["alphas"], dtype=np.float64)
        bases = np.asarray(data["bases"], dtype=np.float64)
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        phi = np.asarray(data["phi"], dtype=np.float64)
        required_slack = np.asarray(data["required_slack"], dtype=np.float64)
    dof_count = int(getattr(bundle["inputs"].get("model"), "num_dofs", qpos.shape[1]))
    if (
        len(alphas) != int(sample_count)
        or bases.shape != (int(sample_count), 4, 4)
        or qpos.shape != (int(sample_count), dof_count)
        or phi.shape != (int(sample_count), 512)
        or required_slack.shape != (int(sample_count), 512)
        or not np.isfinite(alphas).all()
        or not np.isfinite(bases).all()
        or not np.isfinite(qpos).all()
        or not np.isfinite(phi).all()
        or not np.isfinite(required_slack).all()
        or not np.allclose(alphas, np.linspace(0.0, 1.0, int(sample_count)))
    ):
        raise Stage935Error(f"path cache shape or finiteness validation failed: {cache_path}")

    inputs = bundle["inputs"]
    if not np.allclose(alphas[[0, -1]], [0.0, 1.0]):
        raise Stage935Error(f"path cache endpoints are not [0,1]: {cache_path}")
    if not np.allclose(bases[0], inputs["warm"].arrays["base_pose_scene"][frame]):
        raise Stage935Error(f"path cache warm base does not match current lineage: {cache_path}")
    if not np.allclose(bases[-1], inputs["final"].arrays["base_pose_scene"][frame]):
        raise Stage935Error(f"path cache final base does not match current lineage: {cache_path}")
    if not np.allclose(qpos[0], inputs["warm"].arrays["qpos"][frame]):
        raise Stage935Error(f"path cache warm qpos does not match current lineage: {cache_path}")
    if not np.allclose(qpos[-1], inputs["final"].arrays["qpos"][frame]):
        raise Stage935Error(f"path cache final qpos does not match current lineage: {cache_path}")
    paper = _paper(bundle)
    context = _context(bundle, frame)
    values = np.stack(
        [_state_value(context, bases[index], qpos[index]) for index in range(len(alphas))]
    )
    tensor_values = torch.as_tensor(values, dtype=torch.float64)
    with torch.no_grad():
        keypoints = inputs["model"].keypoints_scene(qpos, bases, layout="mediapipe21")
        keypoints = np.asarray(
            keypoints.detach().cpu() if hasattr(keypoints, "detach") else keypoints,
            dtype=np.float64,
        )
        terms = _term_tensors(context, tensor_values)
        term_arrays = {
            name: np.asarray(item.detach().cpu(), dtype=np.float64) for name, item in terms.items()
        }
    source = np.asarray(inputs["source_keypoints"][frame], dtype=np.float64)
    errors = np.linalg.norm(keypoints - source[None, :, :], axis=-1)
    long_rmse = np.mean(
        [np.sqrt(np.mean(errors[:, list(FINGER_GROUPS[f])] ** 2, axis=1)) for f in LONG_FINGERS],
        axis=0,
    )
    finger_rmse = {
        finger: np.sqrt(np.mean(errors[:, list(FINGER_GROUPS[finger])] ** 2, axis=1))
        for finger in ALL_FINGERS
    }
    hard_legal = np.min(phi, axis=1) >= -paper.b - PATH_EPSILON_M
    soft_legal = np.min(phi, axis=1) >= -paper.tau - PATH_EPSILON_M
    zero_legal = np.min(phi, axis=1) >= -PATH_EPSILON_M
    path_rows: list[dict[str, Any]] = []
    for index, alpha in enumerate(alphas):
        path_rows.append(
            {
                "alpha": float(alpha),
                "min_sdf_m": float(np.min(phi[index])),
                "hard_residual_min_m": float(np.min(phi[index] + paper.b)),
                "soft_residual_min_zero_slack_m": float(np.min(phi[index] + paper.tau)),
                "required_slack_max_m": float(np.max(required_slack[index])),
                "hard_feasible": bool(hard_legal[index]),
                "soft_safe_feasible": bool(soft_legal[index]),
                "zero_penetration_feasible": bool(zero_legal[index]),
                "long_finger_rmse_m": float(long_rmse[index]),
                **{
                    f"{finger}_rmse_m": float(values[index])
                    for finger, values in finger_rmse.items()
                },
                "e_im_raw": float(term_arrays["e_im"][index]),
                "e_bone_raw": float(term_arrays["e_bone"][index]),
                "formal_total_objective_zero_slack": float(
                    term_arrays["weighted_e_im"][index]
                    + term_arrays["weighted_e_bone"][index]
                    + term_arrays["e_temporal"][index]
                    + term_arrays["e_base_pos"][index]
                    + term_arrays["e_base_rot"][index]
                ),
                "formal_total_objective_minimal_legal_slack": float(
                    term_arrays["weighted_e_im"][index]
                    + term_arrays["weighted_e_bone"][index]
                    + term_arrays["e_temporal"][index]
                    + term_arrays["e_base_pos"][index]
                    + term_arrays["e_base_rot"][index]
                    + 0.5 * paper.w_s * np.sum(required_slack[index] ** 2)
                ),
            }
        )

    pose = inputs["object"].pose_scene.pose_scene[frame]
    relative = so3_log(bases[-1, :3, :3] @ bases[0, :3, :3].T)

    def predicate(alpha: float, kind: str) -> bool:
        value = float(alpha)
        base = np.asarray(bases[0]).copy()
        base[:3, :3] = Rotation.from_rotvec(value * relative).as_matrix() @ bases[0, :3, :3]
        base[:3, 3] = (1.0 - value) * bases[0, :3, 3] + value * bases[-1, :3, 3]
        q = (1.0 - value) * qpos[0] + value * qpos[-1]
        points = dynamic_collision_points_numpy(inputs["model"], inputs["surface"], q, base)
        minimum = float(np.min(inputs["reference_sdf"].query_scene(points, pose).signed_distance))
        threshold = {"soft": -paper.tau, "hard": -paper.b, "zero": 0.0}[kind]
        return minimum >= threshold - PATH_EPSILON_M

    def refine_boundary(left: float, right: float, kind: str, left_ok: bool) -> float:
        for _ in range(12):
            middle = 0.5 * (left + right)
            if predicate(middle, kind) == left_ok:
                left = middle
            else:
                right = middle
        return float(0.5 * (left + right))

    def intervals(mask: np.ndarray, kind: str) -> list[dict[str, float]]:
        result: list[dict[str, float]] = []
        start: int | None = None
        for index, ok in enumerate(mask.tolist() + [False]):
            if ok and start is None:
                start = index
            elif not ok and start is not None:
                stop = index - 1
                left = float(alphas[start])
                right = float(alphas[stop])
                if start > 0:
                    left = refine_boundary(float(alphas[start - 1]), left, kind, False)
                if stop < len(alphas) - 1:
                    right = refine_boundary(right, float(alphas[stop + 1]), kind, True)
                result.append({"start_alpha": left, "end_alpha": right})
                start = None
        return result

    per_sample_order: list[dict[str, Any]] = []
    names = np.asarray(inputs["surface"].link_names).astype(str)
    for sample in range(512):
        valid = np.flatnonzero(phi[:, sample] >= -paper.tau - PATH_EPSILON_M)
        hard_valid = np.flatnonzero(phi[:, sample] >= -paper.b - PATH_EPSILON_M)
        per_sample_order.append(
            {
                "sample_id": sample,
                "link": str(names[sample]),
                "first_soft_safe_alpha_grid": None if not len(valid) else float(alphas[valid[0]]),
                "first_hard_feasible_alpha_grid": None
                if not len(hard_valid)
                else float(alphas[hard_valid[0]]),
                "warm_soft_violation_m": float(max(0.0, -phi[0, sample] - paper.tau)),
                "warm_hard_violation_m": float(max(0.0, -phi[0, sample] - paper.b)),
            }
        )
    return {
        "schema_version": "toporetarget.stage9_3_5_warm_final_path.v1",
        "frame": int(frame),
        "global_frame": int(240 + frame),
        "sample_count": int(sample_count),
        "alphas": alphas,
        "bases": bases,
        "qpos": qpos,
        "phi": phi,
        "required_slack": required_slack,
        "rows": path_rows,
        "soft_safe_intervals": intervals(soft_legal, "soft"),
        "hard_feasible_intervals": intervals(hard_legal, "hard"),
        "zero_penetration_intervals": intervals(zero_legal, "zero"),
        "first_soft_safe_alpha": next(
            (float(alphas[index]) for index, item in enumerate(soft_legal) if item), None
        ),
        "first_hard_feasible_alpha": next(
            (float(alphas[index]) for index, item in enumerate(hard_legal) if item), None
        ),
        "first_zero_penetration_alpha": next(
            (float(alphas[index]) for index, item in enumerate(zero_legal) if item), None
        ),
        "warm_soft_feasible": bool(soft_legal[0]),
        "warm_hard_feasible": bool(hard_legal[0]),
        "warm_zero_penetration": bool(zero_legal[0]),
        "official_final_soft_feasible": bool(soft_legal[-1]),
        "official_final_hard_feasible": bool(hard_legal[-1]),
        "official_final_zero_penetration": bool(zero_legal[-1]),
        "non_monotonicity": {
            "soft": bool(np.any(np.diff(soft_legal.astype(np.int8)) < 0)),
            "hard": bool(np.any(np.diff(hard_legal.astype(np.int8)) < 0)),
            "zero": bool(np.any(np.diff(zero_legal.astype(np.int8)) < 0)),
        },
        "per_sample_violation_order": per_sample_order,
        "query_schedule": {
            "canonical_query_chunk_size": int(inputs["reference_sdf"].query_chunk_size),
            "canonical_face_chunk_size": int(inputs["reference_sdf"].face_chunk_size),
            "path_query_chunk_size": int(inputs["reference_sdf"].query_chunk_size),
            "path_face_chunk_size": int(inputs["reference_sdf"].face_chunk_size),
            "workers": 0,
            "backend_math": "reference_triangle_winding",
            "cache_reused": True,
        },
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }


def run_scan(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
    samples: int = 1001,
    resume: bool = False,
) -> dict[str, Any]:
    if samples < 1001:
        raise Stage935Error("warm-final path requires at least 1001 samples")
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    selected = tuple(frames) if frames else bundle["frames"]
    if len(selected) > 5:
        raise Stage935Error("projection diagnostics are capped at five frames")
    if any(frame < 0 or frame >= 60 for frame in selected):
        raise Stage935Error("selected frame is outside [0,60)")
    destination = _resolve(output_root, bundle["repo"])
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(
        destination / "official_artifact_immutability_before.json",
        _official_artifact_snapshot(bundle),
    )
    _write_json(destination / "input_identity_and_lineage.json", bundle["identity"])
    selection_payload = _frame_selection_payload(bundle)
    selection_payload["frames"] = [
        item for item in selection_payload["frames"] if item["local_frame"] in selected
    ]
    _write_json(destination / "projection_frame_selection.json", selection_payload)
    known_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    path_payloads: dict[str, Any] = {}
    existing_known: dict[int, dict[str, Any]] = {}
    existing_per_frame: dict[str, dict[str, Any]] = {}
    existing_rows: dict[int, list[dict[str, Any]]] = {}
    if resume:
        existing_manifest = destination / "projection_manifest.json"
        existing_summary = destination / "warm_final_path_feasibility.json"
        existing_known_path = destination / "known_feasible_seed_validation.json"
        existing_rows_path = destination / "warm_final_path_feasibility.csv"
        if (
            existing_manifest.is_file()
            and existing_summary.is_file()
            and existing_known_path.is_file()
        ):
            manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
            summary = json.loads(existing_summary.read_text(encoding="utf-8"))
            known_payload = json.loads(existing_known_path.read_text(encoding="utf-8"))
            if (
                manifest.get("current_causal_lineage_hash") == bundle["lineage_hash"]
                and int(manifest.get("sample_count", -1)) == samples
                and summary.get("current_causal_lineage_hash") == bundle["lineage_hash"]
                and int(summary.get("sample_count", -1)) == samples
            ):
                existing_known = {
                    int(item["frame"]): item
                    for item in known_payload.get("frames", [])
                    if "frame" in item
                }
                existing_per_frame = {
                    str(key): value for key, value in summary.get("per_frame", {}).items()
                }
                for row in _read_csv_rows(existing_rows_path):
                    if "frame" in row:
                        existing_rows.setdefault(int(row["frame"]), []).append(row)
    for frame in selected:
        cache_path = destination / "path_cache" / f"frame_{frame:06d}.npz"
        if resume and cache_path.is_file():
            if frame in existing_known and str(frame) in existing_per_frame:
                known_rows.append(existing_known[frame])
                path_payloads[str(frame)] = existing_per_frame[str(frame)]
                path_rows.extend(existing_rows.get(frame, []))
                continue
            cached_known = _evaluate_state(
                bundle,
                frame,
                bundle["inputs"]["final"].arrays["base_pose_scene"][frame],
                bundle["inputs"]["final"].arrays["qpos"][frame],
                _full_slack(bundle["inputs"]["final"], frame),
            )
            cached_known["known_feasible_final"] = bool(
                cached_known["qpos_bounds_pass"]
                and cached_known["base_valid"]
                and cached_known["full512_finite"]
                and cached_known["hard_violation_m"] <= 1e-8
                and cached_known["soft_violation_m"] <= 1e-8
                and cached_known["slack_bounds_pass"]
            )
            cached_known["canonical_backend"] = bundle["inputs"]["reference_sdf"].describe()
            cached_known["object_pose_hash"] = _stable_hash(
                bundle["inputs"]["object"].pose_scene.pose_scene[frame]
            )
            cached_known["collision_sample_identity"] = bundle["identity"]["collision_sample_hash"]
            if not cached_known["known_feasible_final"]:
                raise Stage935Error(f"known feasible final validation failed at frame {frame}")
            cached_path = _path_metrics_from_cache(bundle, frame, cache_path, samples)
            known_rows.append(cached_known)
            path_payloads[str(frame)] = cached_path
            path_rows.extend(
                {"frame": frame, "global_frame": 240 + frame, **row} for row in cached_path["rows"]
            )
            continue
        final = bundle["inputs"]["final"]
        value = _evaluate_state(
            bundle,
            frame,
            final.arrays["base_pose_scene"][frame],
            final.arrays["qpos"][frame],
            _full_slack(final, frame),
        )
        value["known_feasible_final"] = bool(
            value["qpos_bounds_pass"]
            and value["base_valid"]
            and value["full512_finite"]
            and value["hard_violation_m"] <= 1e-8
            and value["soft_violation_m"] <= 1e-8
            and value["slack_bounds_pass"]
        )
        value["canonical_backend"] = bundle["inputs"]["reference_sdf"].describe()
        value["object_pose_hash"] = _stable_hash(
            bundle["inputs"]["object"].pose_scene.pose_scene[frame]
        )
        value["collision_sample_identity"] = bundle["identity"]["collision_sample_hash"]
        known_rows.append(value)
        if not value["known_feasible_final"]:
            raise Stage935Error(f"known feasible final validation failed at frame {frame}")
        path = _path_metrics(bundle, frame, samples)
        path_payloads[str(frame)] = path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_npz(
            cache_path,
            alphas=np.asarray(path["alphas"], dtype=np.float64),
            bases=np.asarray(path["bases"], dtype=np.float64),
            qpos=np.asarray(path["qpos"], dtype=np.float64),
            phi=np.asarray(path["phi"], dtype=np.float64),
            required_slack=np.asarray(path["required_slack"], dtype=np.float64),
            first_soft_safe_alpha=np.asarray(
                np.nan if path["first_soft_safe_alpha"] is None else path["first_soft_safe_alpha"]
            ),
            first_hard_feasible_alpha=np.asarray(
                np.nan
                if path["first_hard_feasible_alpha"] is None
                else path["first_hard_feasible_alpha"]
            ),
        )
        for row in path["rows"]:
            path_rows.append({"frame": frame, "global_frame": 240 + frame, **row})
    _write_json(
        destination / "known_feasible_seed_validation.json",
        {
            "schema_version": "toporetarget.stage9_3_5_known_feasible_seed.v1",
            "frames": known_rows,
            "known_feasible_seed": all(bool(row["known_feasible_final"]) for row in known_rows),
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "diagnostic_only": True,
        },
    )
    _write_csv(destination / "known_feasible_seed_validation.csv", known_rows)
    path_summary = {
        "schema_version": "toporetarget.stage9_3_5_warm_final_path_collection.v1",
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "frames": list(selected),
        "sample_count": samples,
        "per_frame": {
            frame: {
                key: value
                for key, value in path_payloads[str(frame)].items()
                if key
                not in {
                    "alphas",
                    "bases",
                    "qpos",
                    "phi",
                    "required_slack",
                    "rows",
                    "per_sample_violation_order",
                }
            }
            for frame in selected
        },
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    _write_json(destination / "warm_final_path_feasibility.json", path_summary)
    _write_csv(destination / "warm_final_path_feasibility.csv", path_rows)
    paper = _paper(bundle)
    _write_json(
        destination / "projection_state_metric.json",
        {
            "schema_version": STATE_METRIC_SCHEMA,
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "definition": "lambda_reg/2 ||q-q_warm||^2 + lambda_base_pos/2 ||p-p_warm||^2 + lambda_base_rot/2 ||log(R_warm^T R)||^2 + w_s/2 ||s||^2",
            "center": "current-frame warm state",
            "weights": paper.as_dict(),
            "units": {"qpos": "rad", "base_translation": "m", "base_rotation": "rad", "slack": "m"},
            "not_eq_8_eq_9": True,
        },
    )
    _write_json(
        destination / "projection_profiles.json",
        {
            "schema_version": "toporetarget.stage9_3_5_projection_profiles.v1",
            "profiles": [
                {
                    "profile_id": profile,
                    "diagnostic_only": True,
                    "paper_method": False,
                    "accepted_reference": False,
                    "constraints": "full512 canonical reference-winding",
                    "slack": profile.startswith("official_slack"),
                    "solver": "scipy_slsqp_primary_then_trust_constr_fallback",
                }
                for profile in PROFILES
            ],
        },
    )
    manifest = {
        "schema_version": SCHEMA,
        "stage": "9.3.5",
        "status": "PATH_SCAN_COMPLETE",
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "current_lineage_manifest": str(bundle["lineage_path"]),
        "current_baseline": str(bundle["baseline_path"]),
        "frames": list(selected),
        "sample_count": samples,
        "canonical_backend": bundle["inputs"]["reference_sdf"].describe(),
        "solver_invocation_count": 0,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "environment": _environment(),
    }
    existing_results = _load_json_if(destination / "projection_solver_results.json", {})
    if existing_results.get("current_causal_lineage_hash", bundle["lineage_hash"]) == bundle[
        "lineage_hash"
    ] and len(existing_results.get("results", [])) == len(selected) * len(PROFILES):
        manifest.update(
            {
                "status": "PROJECTION_COMPLETE",
                "profiles": sorted(
                    {str(row.get("profile")) for row in existing_results["results"]}
                ),
                "solver_invocation_count": len(existing_results["results"]),
                "full512": True,
            }
        )
    _write_json(destination / "projection_manifest.json", manifest)
    return path_summary


def _projection_constraints(
    context: Any, value: np.ndarray, profile: str, query_hash: str
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.arange(512, dtype=np.int64)
    result = context.constraint_query(value, ids, query_hash)
    phi = np.asarray(result.signed_distance, dtype=np.float64)
    _, _, _, slack = context.unpack(value)
    if profile.startswith("minimal_soft_safe"):
        values = phi + context.paper.tau
        jacobian_state = value
        if len(jacobian_state) == context.variable_size_without_slack:
            jacobian_state = np.concatenate([jacobian_state, np.zeros(512, dtype=np.float64)])
        jac, _ = context.constraint_jacobian(
            jacobian_state, ids, 1e-6, query_hash, backend="analytic_urdf_spatial_v2"
        )
        return values, np.asarray(jac[512:], dtype=np.float64)[
            :, : context.variable_size_without_slack
        ]
    values = np.concatenate([phi + context.paper.b, phi + slack + context.paper.tau])
    jac, _ = context.constraint_jacobian(
        value, ids, 1e-6, query_hash, backend="analytic_urdf_spatial_v2"
    )
    return values, np.asarray(jac, dtype=np.float64)


def _projection_objective_gradient(
    context: Any, value: np.ndarray, with_slack: bool
) -> tuple[float, np.ndarray]:
    dp, dw, q, slack = context.unpack(value)
    qdelta = q - context.seed_qpos
    objective = 0.5 * (
        context.paper.lambda_reg * np.sum(qdelta**2)
        + context.paper.lambda_base_pos * np.sum(dp**2)
        + context.paper.lambda_base_rot * np.sum(dw**2)
    )
    gradient = np.zeros_like(value, dtype=np.float64)
    gradient[:3] = context.paper.lambda_base_pos * dp
    gradient[3:6] = context.paper.lambda_base_rot * dw
    gradient[6 : 6 + context.robot_model.num_dofs] = context.paper.lambda_reg * qdelta
    if with_slack:
        objective += 0.5 * context.paper.w_s * np.sum(slack**2)
        gradient[6 + context.robot_model.num_dofs :] = context.paper.w_s * slack
    return float(objective), gradient


def _independent_projection_validation(
    bundle: dict[str, Any],
    frame: int,
    profile: str,
    state: np.ndarray,
    solver_success: bool,
    status: int,
    message: str,
) -> dict[str, Any]:
    context = _context(bundle, frame)
    base = _base_pose_from_value(context, state)
    q = state[6 : 6 + context.robot_model.num_dofs]
    slack = state[6 + context.robot_model.num_dofs :]
    result = _evaluate_state(bundle, frame, base, q, slack)
    with_slack = profile.startswith("official_slack")
    feasibility = bool(
        result["qpos_bounds_pass"]
        and result["base_valid"]
        and result["full512_finite"]
        and result["hard_violation_m"] <= 1e-8
        and result["soft_violation_m"] <= 1e-8
        and result["slack_bounds_pass"]
        and (with_slack or len(slack) == 0 or float(np.max(np.abs(slack))) <= 1e-12)
    )
    final = bundle["inputs"]["final"]
    final_slack = _full_slack(final, frame) if with_slack else np.zeros(512, dtype=np.float64)
    final_value = _state_value(
        context,
        final.arrays["base_pose_scene"][frame],
        final.arrays["qpos"][frame],
        final_slack,
    )
    threshold_value = _state_metric_value(context, final_value)
    if with_slack:
        minimal_legal_slack = np.clip(
            -bundle["inputs"]["reference_sdf"]
            .query_scene(
                dynamic_collision_points_numpy(
                    bundle["inputs"]["model"],
                    bundle["inputs"]["surface"],
                    final.arrays["qpos"][frame],
                    final.arrays["base_pose_scene"][frame],
                ),
                bundle["inputs"]["object"].pose_scene.pose_scene[frame],
            )
            .signed_distance
            - _paper(bundle).tau,
            0.0,
            _paper(bundle).b - _paper(bundle).tau,
        )
        threshold_value = _state_metric_value(
            context,
            _state_value(
                context,
                final.arrays["base_pose_scene"][frame],
                final.arrays["qpos"][frame],
                minimal_legal_slack,
            ),
        )
    projection_objective = _state_metric_value(context, state)
    threshold_pass = bool(projection_objective <= threshold_value + 1e-8)
    optimizer_feasible = bool(solver_success and status != 9 and feasibility)
    strict = bool(optimizer_feasible and threshold_pass)
    return {
        "frame": frame,
        "profile": profile,
        "status": int(status),
        "solver_success": bool(solver_success),
        "solver_message": str(message),
        "projection_feasible": optimizer_feasible,
        "strict_projection_acceptance": strict,
        "full512": True,
        "min_phi_m": result["min_sdf_m"],
        "raw_penetration_m": result["raw_penetration_m"],
        "hard_violation_m": result["hard_violation_m"],
        "soft_violation_m": result["soft_violation_m"],
        "slack_bounds_pass": result["slack_bounds_pass"],
        "qpos_bounds_pass": result["qpos_bounds_pass"],
        "finite": result["full512_finite"],
        "base_valid": result["base_valid"],
        "projection_objective": projection_objective,
        "projection_objective_threshold": threshold_value,
        "projection_objective_threshold_pass": threshold_pass,
        "state": state,
        "metrics": result,
    }


def _restore_candidate(
    bundle: dict[str, Any], frame: int, profile: str, state: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run diagnostic feasibility restoration only for a non-feasible candidate."""
    from scipy.optimize import minimize

    context = _context(bundle, frame)
    with_slack = profile.startswith("official_slack")
    query_hash = _stable_hash(
        {"frame": frame, "profile": profile, "phase": "A", "lineage": bundle["lineage_hash"]}
    )
    n = context.variable_size_without_slack
    if with_slack and len(state) == n:
        state = np.concatenate([state, np.zeros(512, dtype=np.float64)])

    def physical(value: np.ndarray) -> np.ndarray:
        return value * np.concatenate(
            [
                np.full(3, 0.1),
                np.ones(3),
                np.maximum(context.robot_model.joint_upper - context.robot_model.joint_lower, 1e-6),
                np.full(512, context.paper.b - context.paper.tau) if with_slack else np.empty(0),
            ]
        )

    scales = np.concatenate(
        [
            np.full(3, 0.1),
            np.ones(3),
            np.maximum(context.robot_model.joint_upper - context.robot_model.joint_lower, 1e-6),
            np.full(512, context.paper.b - context.paper.tau) if with_slack else np.empty(0),
        ]
    )

    def violation(value: np.ndarray) -> float:
        physical_value = physical(value)
        result = context.constraint_query(physical_value, np.arange(512), query_hash)
        phi = result.signed_distance
        _, _, _, slack = context.unpack(physical_value)
        total = np.maximum(-context.paper.tau - phi, 0.0) ** 2
        total += np.maximum(-context.paper.b - phi, 0.0) ** 2
        if with_slack:
            total += np.maximum(-context.paper.tau - phi - slack, 0.0) ** 2
        total += np.maximum(-slack, 0.0) ** 2 if with_slack else 0.0
        total += (
            np.maximum(slack - (context.paper.b - context.paper.tau), 0.0) ** 2
            if with_slack
            else 0.0
        )
        return float(np.sum(total))

    lower = np.concatenate(
        [
            np.full(6, -np.inf),
            context.robot_model.joint_lower,
            np.zeros(512) if with_slack else np.empty(0),
        ]
    )
    upper = np.concatenate(
        [
            np.full(6, np.inf),
            context.robot_model.joint_upper,
            np.full(512, context.paper.b - context.paper.tau) if with_slack else np.empty(0),
        ]
    )
    result = minimize(
        violation,
        state / scales,
        method="SLSQP",
        bounds=list(zip(lower / scales, upper / scales, strict=True)),
        options={"maxiter": 100, "ftol": 1e-12, "disp": False},
    )
    restored = physical(np.asarray(result.x, dtype=np.float64))
    return restored, {
        "phase": "A",
        "status": int(getattr(result, "status", -1)),
        "success": bool(result.success),
        "message": str(getattr(result, "message", "")),
        "violation": violation(np.asarray(result.x, dtype=np.float64)),
    }


def _solve_projection_attempt(
    bundle: dict[str, Any], frame: int, profile: str, initial: np.ndarray, attempt: int
) -> dict[str, Any]:
    from scipy.optimize import minimize

    context = _context(bundle, frame)
    with_slack = profile.startswith("official_slack")
    n = context.variable_size_without_slack
    if with_slack and len(initial) == n:
        initial = np.concatenate([initial, np.zeros(512, dtype=np.float64)])
    if not with_slack:
        initial = initial[:n]
    scales = np.concatenate(
        [
            np.full(3, 0.1),
            np.ones(3),
            np.maximum(context.robot_model.joint_upper - context.robot_model.joint_lower, 1e-6),
            np.full(512, context.paper.b - context.paper.tau) if with_slack else np.empty(0),
        ]
    )
    query_hash = _stable_hash(
        {
            "frame": frame,
            "profile": profile,
            "attempt": attempt,
            "lineage": bundle["lineage_hash"],
            "samples": 512,
        }
    )
    restored = initial
    pre = _independent_projection_validation(bundle, frame, profile, restored, True, 0, "candidate")
    phase_a: dict[str, Any] = {
        "phase": "A",
        "skipped": bool(pre["hard_violation_m"] <= 1e-8 and pre["soft_violation_m"] <= 1e-8),
    }
    if phase_a["skipped"] and pre["soft_violation_m"] <= 1e-8:
        identity_validation = dict(pre)
        identity_validation.update(
            {
                "status_label": "ANALYTIC_IDENTITY_PROJECTION",
                "solver_invocation_count": 0,
                "projection_displacement": 0.0,
                "objective": 0.0,
            }
        )
        return {
            "frame": frame,
            "profile": profile,
            "attempt": attempt,
            "phase_a": {**phase_a, "analytic_identity": True},
            "solver": "analytic_identity",
            "solver_version": None,
            "status": 0,
            "success": True,
            "message": "warm state is already soft-feasible",
            "iterations": 0,
            "function_evaluations": 0,
            "runtime_s": 0.0,
            "validation": identity_validation,
        }
    if with_slack and pre["hard_violation_m"] <= 1e-8:
        required = np.clip(
            -np.asarray(pre["metrics"]["full_signed_distance"], dtype=np.float64)
            - context.paper.tau,
            0.0,
            context.paper.b - context.paper.tau,
        )
        restored = np.concatenate([restored[:n], required])
        pre = _independent_projection_validation(
            bundle, frame, profile, restored, True, 0, "legal slack candidate"
        )
        phase_a = {
            "phase": "A",
            "skipped": True,
            "legal_slack_candidate": True,
            "required_slack_max_m": float(np.max(required)),
        }
    elif not phase_a["skipped"]:
        restored, phase_a = _restore_candidate(bundle, frame, profile, restored)

    def physical(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) * scales

    def objective(x: np.ndarray) -> float:
        return _projection_objective_gradient(context, physical(x), with_slack)[0]

    def jacobian(x: np.ndarray) -> np.ndarray:
        return _projection_objective_gradient(context, physical(x), with_slack)[1] * scales

    def constraints(x: np.ndarray) -> np.ndarray:
        values, _ = _projection_constraints(context, physical(x), profile, query_hash)
        return values

    def constraint_jac(x: np.ndarray) -> np.ndarray:
        _, jac = _projection_constraints(context, physical(x), profile, query_hash)
        return jac * scales[None, :]

    lower = np.concatenate(
        [
            np.full(6, -np.inf),
            context.robot_model.joint_lower,
            np.zeros(512) if with_slack else np.empty(0),
        ]
    )
    upper = np.concatenate(
        [
            np.full(6, np.inf),
            context.robot_model.joint_upper,
            np.full(512, context.paper.b - context.paper.tau) if with_slack else np.empty(0),
        ]
    )
    started = time.perf_counter()
    try:
        maxiter = int(os.environ.get("STAGE935_PROJECTION_MAXITER", "200"))
        result = minimize(
            objective,
            restored / scales,
            jac=jacobian,
            method="SLSQP",
            bounds=list(zip(lower / scales, upper / scales, strict=True)),
            constraints={"type": "ineq", "fun": constraints, "jac": constraint_jac},
            options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
        )
        state = physical(np.asarray(result.x, dtype=np.float64))
        validation = _independent_projection_validation(
            bundle,
            frame,
            profile,
            state,
            bool(result.success),
            int(getattr(result, "status", -1)),
            str(getattr(result, "message", "")),
        )
        return {
            "frame": frame,
            "profile": profile,
            "attempt": attempt,
            "phase_a": phase_a,
            "solver": "scipy.optimize.minimize:SLSQP",
            "solver_version": __import__("scipy").__version__,
            "status": int(getattr(result, "status", -1)),
            "success": bool(result.success),
            "message": str(getattr(result, "message", "")),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "runtime_s": float(time.perf_counter() - started),
            "validation": validation,
        }
    except Exception as exc:
        return {
            "frame": frame,
            "profile": profile,
            "attempt": attempt,
            "phase_a": phase_a,
            "solver": "scipy.optimize.minimize:SLSQP",
            "solver_version": __import__("scipy").__version__,
            "status": -1,
            "success": False,
            "message": f"exception: {exc}",
            "runtime_s": float(time.perf_counter() - started),
            "validation": {"strict_projection_acceptance": False, "error": str(exc)},
        }


def _candidate_states(
    bundle: dict[str, Any], frame: int, path: dict[str, Any], profile: str
) -> list[tuple[str, np.ndarray]]:
    context = _context(bundle, frame)
    final = bundle["inputs"]["final"]
    final_slack = _full_slack(final, frame)
    candidates: list[tuple[str, np.ndarray]] = []
    candidates.append(
        (
            "current_official_final",
            _state_value(
                context,
                final.arrays["base_pose_scene"][frame],
                final.arrays["qpos"][frame],
                final_slack if profile.startswith("official") else np.zeros(512),
            ),
        )
    )
    first = (
        path.get("first_soft_safe_alpha")
        if profile.startswith("minimal")
        else path.get("first_hard_feasible_alpha")
    )
    if isinstance(first, np.ndarray):
        first_value = float(first.reshape(-1)[0])
        first = None if not np.isfinite(first_value) else first_value
    if first is not None:
        alphas = np.asarray(path["alphas"], dtype=np.float64)
        index = int(np.argmin(np.abs(alphas - float(first))))
        for label, offset in (("earliest_feasible", 0), ("interior_feasible", 1)):
            index = min(index + offset, len(alphas) - 1)
            slack = (
                np.zeros(512, dtype=np.float64)
                if profile.startswith("minimal")
                else np.asarray(path["required_slack"][index], dtype=np.float64)
            )
            candidates.append(
                (label, _state_value(context, path["bases"][index], path["qpos"][index], slack))
            )
    return candidates


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_projection(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    path_scan_root: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
    profiles: tuple[str, ...] = PROFILES,
    resume: bool = False,
    max_wall_time: float | None = None,
    solver_attempts: int = 3,
) -> dict[str, Any]:
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    scan_root = _resolve(path_scan_root, bundle["repo"])
    destination = _resolve(output_root, bundle["repo"])
    scan_manifest = json.loads((scan_root / "projection_manifest.json").read_text(encoding="utf-8"))
    if scan_manifest.get("current_causal_lineage_hash") != bundle["lineage_hash"]:
        raise Stage935Error("path scan lineage differs from current projection lineage")
    selected = tuple(frames) if frames else tuple(int(item) for item in scan_manifest["frames"])
    selected_profiles = tuple(profiles)
    if any(profile not in PROFILES for profile in selected_profiles):
        raise Stage935Error(f"unknown projection profile: {selected_profiles}")
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "input_identity_and_lineage.json", bundle["identity"])
    _write_json(destination / "projection_frame_selection.json", _frame_selection_payload(bundle))
    for name in (
        "known_feasible_seed_validation.json",
        "known_feasible_seed_validation.csv",
        "warm_final_path_feasibility.json",
        "warm_final_path_feasibility.csv",
        "projection_state_metric.json",
        "projection_profiles.json",
    ):
        source = scan_root / name
        if source.exists():
            (destination / name).write_bytes(source.read_bytes())
    json.loads((scan_root / "warm_final_path_feasibility.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    started = time.perf_counter()
    for frame in selected:
        # The compact scan report is sufficient for the candidate state only
        # when the path arrays have been persisted in the per-frame cache.
        cache_path = scan_root / "path_cache" / f"frame_{frame:06d}.npz"
        if not cache_path.is_file():
            # Recompute the bounded path once, without changing the selected
            # frame set.  This preserves the scan contract and keeps the JSON
            # report small.
            full_path = _path_metrics(bundle, frame, int(scan_manifest["sample_count"]))
        else:
            with np.load(cache_path, allow_pickle=False) as data:
                full_path = {key: data[key] for key in data.files}
        for profile in selected_profiles:
            if max_wall_time is not None and time.perf_counter() - started > max_wall_time:
                manifest = {
                    "schema_version": SCHEMA,
                    "status": "PAUSED_MAX_WALL_TIME",
                    "frames": selected,
                    "profiles": selected_profiles,
                    "current_causal_lineage_hash": bundle["lineage_hash"],
                    "diagnostic_only": True,
                }
                _write_json(destination / "projection_manifest.json", manifest)
                return manifest
            candidates = _candidate_states(bundle, frame, full_path, profile)
            profile_attempts: list[dict[str, Any]] = []
            best: dict[str, Any] | None = None
            for attempt, (label, state) in enumerate(
                candidates[: max(1, solver_attempts)], start=1
            ):
                checkpoint = (
                    destination
                    / "checkpoints"
                    / f"frame_{frame:06d}"
                    / f"{profile}"
                    / f"attempt_{attempt:02d}.json"
                )
                if resume and checkpoint.is_file():
                    attempt_result = json.loads(checkpoint.read_text(encoding="utf-8"))
                    saved_validation = attempt_result.get("validation", {})
                    if saved_validation.get("state") is not None:
                        attempt_result["validation"] = _independent_projection_validation(
                            bundle,
                            frame,
                            profile,
                            np.asarray(saved_validation["state"], dtype=np.float64),
                            bool(attempt_result.get("success", False)),
                            int(attempt_result.get("status", -1)),
                            str(attempt_result.get("message", "")),
                        )
                else:
                    attempt_result = _solve_projection_attempt(
                        bundle, frame, profile, state, attempt
                    )
                    attempt_result["candidate_label"] = label
                    attempt_result["current_causal_lineage_hash"] = bundle["lineage_hash"]
                    attempt_result["objective_profile"] = STATE_METRIC_SCHEMA
                    _atomic_checkpoint(checkpoint, attempt_result)
                profile_attempts.append(attempt_result)
                validation = attempt_result.get("validation", {})
                if validation.get("strict_projection_acceptance"):
                    if best is None or float(
                        validation.get("projection_objective", math.inf)
                    ) < float(best["validation"].get("projection_objective", math.inf)):
                        best = attempt_result
            if best is None:
                status = "PROJECTION_SOLVER_FAILED"
                selected_result = next(
                    (
                        item
                        for item in profile_attempts
                        if item.get("validation", {}).get("metrics")
                    ),
                    None,
                )
            else:
                status = "PROJECTION_SOLVED"
                selected_result = best
            if selected_result is None:
                row = {
                    "frame": frame,
                    "profile": profile,
                    "status": status,
                    "solver_success": False,
                    "strict_projection_acceptance": False,
                    "current_causal_lineage_hash": bundle["lineage_hash"],
                }
            else:
                validation = selected_result.get("validation", {})
                metrics = validation.get("metrics", {})
                official_final = _evaluate_state(
                    bundle,
                    frame,
                    bundle["inputs"]["final"].arrays["base_pose_scene"][frame],
                    bundle["inputs"]["final"].arrays["qpos"][frame],
                    _full_slack(bundle["inputs"]["final"], frame),
                )
                candidate_long = metrics.get("long_finger_rmse_m")
                row = {
                    "frame": frame,
                    "global_frame": 240 + frame,
                    "profile": profile,
                    "status": status,
                    "solver": selected_result.get("solver"),
                    "solver_version": selected_result.get("solver_version"),
                    "solver_status": selected_result.get("status"),
                    "solver_success": selected_result.get("success", False),
                    "iterations": selected_result.get("iterations"),
                    "function_evaluations": selected_result.get("function_evaluations"),
                    "runtime_s": selected_result.get("runtime_s"),
                    "strict_projection_acceptance": validation.get(
                        "strict_projection_acceptance", False
                    ),
                    "projection_objective": validation.get("projection_objective"),
                    "projection_objective_threshold": validation.get(
                        "projection_objective_threshold"
                    ),
                    "projection_objective_threshold_pass": validation.get(
                        "projection_objective_threshold_pass"
                    ),
                    "projection_feasible": validation.get("projection_feasible", False),
                    "min_sdf_m": validation.get("min_phi_m"),
                    "raw_penetration_m": validation.get("raw_penetration_m"),
                    "hard_violation_m": validation.get("hard_violation_m"),
                    "soft_violation_m": validation.get("soft_violation_m"),
                    "long_finger_rmse_m": metrics.get("long_finger_rmse_m"),
                    "long_finger_rmse_improvement_m": (
                        None
                        if candidate_long is None
                        else float(official_final["long_finger_rmse_m"]) - float(candidate_long)
                    ),
                    "contact_proxy": metrics.get("contact_proxy"),
                    "state_displacement": metrics.get("state_displacement"),
                    "solver_message": selected_result.get("message"),
                    "attempt_count": len(profile_attempts),
                    "current_causal_lineage_hash": bundle["lineage_hash"],
                    "state": validation.get("state"),
                    "metrics": metrics,
                }
            results.append(row)
            validations.extend([item.get("validation", {}) for item in profile_attempts])
    _write_json(
        destination / "projection_solver_results.json",
        {
            "schema_version": SCHEMA,
            "results": results,
            "attempts": validations,
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
        },
    )
    _write_csv(destination / "projection_results_per_frame.csv", results)
    _write_json(
        destination / "projection_independent_validation.json",
        {
            "schema_version": SCHEMA,
            "validations": validations,
            "full512": True,
            "diagnostic_only": True,
        },
    )
    fractions: list[dict[str, Any]] = []
    final = bundle["inputs"]["final"]
    for row in results:
        frame = int(row["frame"])
        if row.get("state") is None:
            continue
        context = _context(bundle, frame)
        warm_value = _state_value(
            context,
            bundle["inputs"]["warm"].arrays["base_pose_scene"][frame],
            bundle["inputs"]["warm"].arrays["qpos"][frame],
        )
        final_value = _state_value(
            context,
            final.arrays["base_pose_scene"][frame],
            final.arrays["qpos"][frame],
            _full_slack(final, frame),
        )
        projection = np.asarray(row["state"], dtype=np.float64)
        state_den = _state_metric_value(context, final_value)
        state_num = _state_metric_value(context, projection)
        fractions.append(
            {
                "frame": frame,
                "profile": row["profile"],
                "state_fraction": float(math.sqrt(max(state_num, 0.0) / max(state_den, 1e-30))),
                "qpos_fraction": float(
                    np.linalg.norm(projection[6:28] - warm_value[6:28])
                    / max(
                        float(np.linalg.norm(final_value[6:28] - warm_value[6:28])),
                        1e-12,
                    )
                ),
                "base_translation_fraction": float(
                    np.linalg.norm(projection[:3] - warm_value[:3])
                    / max(
                        float(np.linalg.norm(final_value[:3] - warm_value[:3])),
                        1e-12,
                    )
                ),
                "base_rotation_fraction": float(
                    np.linalg.norm(projection[3:6] - warm_value[3:6])
                    / max(
                        float(np.linalg.norm(final_value[3:6] - warm_value[3:6])),
                        1e-12,
                    )
                ),
                "long_finger_rmse_fraction": None,
                "interaction_objective_fraction": None,
                "projection_status": row["status"],
                "strict_projection_acceptance": row.get("strict_projection_acceptance", False),
            }
        )
    _write_json(
        destination / "feasibility_motion_fraction.json",
        {
            "schema_version": SCHEMA,
            "rows": fractions,
            "interpretation": {
                "low": "official final moves beyond feasibility",
                "high": "motion is close to minimum feasible correction",
            },
            "diagnostic_only": True,
        },
    )
    _write_csv(destination / "feasibility_motion_fraction.csv", fractions)
    manifest = {
        "schema_version": SCHEMA,
        "status": "PROJECTION_COMPLETE",
        "frames": list(selected),
        "profiles": selected_profiles,
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "solver_invocation_count": len(results),
        "full512": True,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "environment": _environment(),
    }
    _write_json(destination / "projection_manifest.json", manifest)
    return manifest


def _counterfactual_states(
    bundle: dict[str, Any], frame: int, projection_root: Path | None = None
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    inputs = bundle["inputs"]
    warm_base = np.asarray(inputs["warm"].arrays["base_pose_scene"][frame], dtype=np.float64)
    warm_q = np.asarray(inputs["warm"].arrays["qpos"][frame], dtype=np.float64)
    final = inputs["final"]
    final_base = np.asarray(final.arrays["base_pose_scene"][frame], dtype=np.float64)
    final_q = np.asarray(final.arrays["qpos"][frame], dtype=np.float64)
    zero = np.zeros(512, dtype=np.float64)
    states: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("warm", warm_base, warm_q, zero),
        ("current_final", final_base, final_q, _full_slack(final, frame)),
        ("final_base_warm_q", final_base, warm_q, zero),
        ("warm_base_final_all_q", warm_base, final_q, zero),
    ]
    for finger in ALL_FINGERS:
        q = warm_q.copy()
        indices = [
            index
            for index, name in enumerate(inputs["model"].dof_names)
            if finger in str(name).lower()
        ]
        q[indices] = final_q[indices]
        states.append((f"warm_plus_final_{finger}_q", warm_base, q, zero))
    q = warm_q.copy()
    for finger in LONG_FINGERS:
        indices = [
            index
            for index, name in enumerate(inputs["model"].dof_names)
            if finger in str(name).lower()
        ]
        q[indices] = final_q[indices]
    states.append(("warm_plus_final_long_finger_q", warm_base, q, zero))
    states.append(("final_base_warm_plus_final_long_finger_q", final_base, q, zero))
    q = warm_q.copy()
    for finger in ("index", "middle", "ring"):
        indices = [
            index
            for index, name in enumerate(inputs["model"].dof_names)
            if finger in str(name).lower()
        ]
        q[indices] = final_q[indices]
    states.append(("warm_plus_final_index_middle_ring_q", warm_base, q, zero))
    q = warm_q.copy()
    for finger in ("thumb", "pinky"):
        indices = [
            index
            for index, name in enumerate(inputs["model"].dof_names)
            if finger in str(name).lower()
        ]
        q[indices] = final_q[indices]
    states.append(("warm_plus_final_thumb_pinky_q", warm_base, q, zero))
    if (
        projection_root is not None
        and (projection_root / "projection_solver_results.json").is_file()
    ):
        payload = json.loads(
            (projection_root / "projection_solver_results.json").read_text(encoding="utf-8")
        )
        for row in payload.get("results", []):
            if int(row.get("frame", -1)) != frame or not row.get("state"):
                continue
            context = _context(bundle, frame)
            state = np.asarray(row["state"], dtype=np.float64)
            base = _base_pose_from_value(context, state)
            q = state[6 : 6 + context.robot_model.num_dofs]
            slack = state[6 + context.robot_model.num_dofs :]
            states.append((f"{row['profile']}", base, q, slack))
            states.append((f"{row['profile']}_base_warm_q", base, warm_q, zero))
            states.append((f"{row['profile']}_warm_base_q", warm_base, q, slack))
    return states


def run_counterfactuals(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    projection_root: str | Path | None = None,
    frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    selected = tuple(frames) if frames else bundle["frames"]
    destination = _resolve(output_root, bundle["repo"])
    projection = _resolve(projection_root, bundle["repo"]) if projection_root else None
    rows: list[dict[str, Any]] = []
    for frame in selected:
        for label, base, q, slack in _counterfactual_states(bundle, frame, projection):
            value = _evaluate_state(bundle, frame, base, q, slack)
            value["label"] = label
            value["current_causal_lineage_hash"] = bundle["lineage_hash"]
            value.pop("value", None)
            rows.append(value)
    payload = {
        "schema_version": SCHEMA,
        "states": rows,
        "frames": list(selected),
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "no_solver": True,
    }
    _write_json(destination / "state_counterfactual_decomposition.json", payload)
    _write_csv(destination / "state_counterfactual_decomposition.csv", rows)
    _write_json(
        destination / "counterfactual_manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "COUNTERFACTUAL_COMPLETE",
            "frames": list(selected),
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "projection_root": str(projection) if projection else None,
            "solver_invocation_count": 0,
            "diagnostic_only": True,
        },
    )
    return payload


def _path_values(
    bundle: dict[str, Any],
    frame: int,
    left: dict[str, Any],
    right: dict[str, Any],
    count: int = 101,
) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    context = _context(bundle, frame)
    base_a = np.asarray(left["base_pose_scene"], dtype=np.float64)
    base_b = np.asarray(right["base_pose_scene"], dtype=np.float64)
    q_a = np.asarray(left["qpos"], dtype=np.float64)
    q_b = np.asarray(right["qpos"], dtype=np.float64)
    s_a = np.asarray(left.get("slack", np.zeros(512)), dtype=np.float64)
    s_b = np.asarray(right.get("slack", np.zeros(512)), dtype=np.float64)
    relative = so3_log(base_b[:3, :3] @ base_a[:3, :3].T)
    values: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, count):
        base = base_a.copy()
        base[:3, :3] = Rotation.from_rotvec(float(alpha) * relative).as_matrix() @ base_a[:3, :3]
        base[:3, 3] = (1.0 - alpha) * base_a[:3, 3] + alpha * base_b[:3, 3]
        values.append(
            _state_value(
                context, base, (1.0 - alpha) * q_a + alpha * q_b, (1.0 - alpha) * s_a + alpha * s_b
            )
        )
    return np.stack(values)


def _gradient_terms(
    context: Any, value: np.ndarray
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    import torch

    variable = torch.as_tensor(
        np.asarray(value, dtype=np.float64), dtype=torch.float64
    ).requires_grad_(True)
    tensors = _term_tensors(context, variable)
    vals: dict[str, float] = {}
    gradients: dict[str, np.ndarray] = {}
    for name, tensor in tensors.items():
        if name not in {
            "weighted_e_im",
            "weighted_e_bone",
            "e_temporal",
            "e_base_pos",
            "e_base_rot",
            "e_slack",
        }:
            continue
        vals[name] = float(tensor.detach().cpu())
        gradient = torch.autograd.grad(tensor, variable, retain_graph=True, allow_unused=False)[0]
        gradients[name] = np.asarray(gradient.detach().cpu(), dtype=np.float64)
    return vals, gradients


def _directional_attribution(
    bundle: dict[str, Any], frame: int, states: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = _context(bundle, frame)
    transitions = [("warm", "projection"), ("projection", "final"), ("warm", "final")]
    endpoint_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    groups = {
        "base_translation": list(range(0, 3)),
        "base_rotation": list(range(3, 6)),
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
        "slack": list(
            range(6 + context.robot_model.num_dofs, 6 + context.robot_model.num_dofs + 512)
        ),
    }
    for index, name in enumerate(context.robot_model.dof_names):
        for finger in ALL_FINGERS:
            if finger in str(name).lower():
                groups[finger].append(6 + index)
    for left_label, right_label in transitions:
        if left_label not in states or right_label not in states:
            continue
        left = states[left_label]
        right = states[right_label]
        va = _state_value(
            context, left["base_pose_scene"], left["qpos"], left.get("slack", np.zeros(512))
        )
        vb = _state_value(
            context, right["base_pose_scene"], right["qpos"], right.get("slack", np.zeros(512))
        )
        delta = vb - va
        ta, ga = _gradient_terms(context, va)
        tb, gb = _gradient_terms(context, vb)
        for term in sorted(ta):
            endpoint_rows.append(
                {
                    "frame": frame,
                    "transition": f"{left_label}_to_{right_label}",
                    "term": term,
                    "left_value": ta[term],
                    "right_value": tb[term],
                    "delta": tb[term] - ta[term],
                    "local_directional_derivative_left": float(np.dot(ga[term], delta)),
                    "local_directional_derivative_right": float(np.dot(gb[term], delta)),
                    "multiplier_available": False,
                    "diagnostic_note": "local/path numerical attribution, not a game-theoretic causal decomposition",
                }
            )
        path = _path_values(bundle, frame, left, right, 101)
        alpha = np.linspace(0.0, 1.0, len(path))
        integrated: dict[str, float] = {term: 0.0 for term in ta}
        grouped: dict[str, dict[str, float]] = {
            term: {group: 0.0 for group in groups} for term in ta
        }
        all_grads: dict[str, list[np.ndarray]] = {term: [] for term in ta}
        all_vals: dict[str, list[float]] = {term: [] for term in ta}
        for value in path:
            vals, grads = _gradient_terms(context, value)
            for term in ta:
                all_grads[term].append(grads[term])
                all_vals[term].append(vals[term])
        for term in ta:
            gradients = np.stack(all_grads[term])
            for group, indices in groups.items():
                if indices:
                    tangent = np.gradient(path[:, indices], alpha, axis=0)
                    integrand = np.sum(gradients[:, indices] * tangent, axis=1)
                    grouped[term][group] = float(np.trapezoid(integrand, alpha))
            total_integrand = np.sum(gradients * np.gradient(path, alpha, axis=0), axis=1)
            integrated[term] = float(np.trapezoid(total_integrand, alpha))
        for term in ta:
            endpoint_rows[-1]["path_integrated_by_term"] = endpoint_rows[-1].get(
                "path_integrated_by_term", {}
            )
            endpoint_rows[-1]["path_integrated_by_term"][term] = integrated[term]
            for group in groups:
                variable_rows.append(
                    {
                        "frame": frame,
                        "transition": f"{left_label}_to_{right_label}",
                        "term": term,
                        "variable_group": group,
                        "path_integrated_contribution": grouped[term][group],
                        "endpoint_delta": tb[term] - ta[term],
                        "partition_check_total": integrated[term],
                    }
                )
        if endpoint_rows:
            endpoint_rows[-1]["path_integral_sum_check"] = float(sum(integrated.values()))
            endpoint_rows[-1]["total_endpoint_delta_check"] = float(sum(tb[t] - ta[t] for t in ta))
    return endpoint_rows, variable_rows


def run_attribution(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    counterfactual_root: str | Path,
    output_root: str | Path,
    *,
    projection_root: str | Path | None = None,
    frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    selected = tuple(frames) if frames else bundle["frames"]
    destination = _resolve(output_root, bundle["repo"])
    projection = _resolve(projection_root, bundle["repo"]) if projection_root else None
    counter_path = (
        _resolve(counterfactual_root, bundle["repo"]) / "state_counterfactual_decomposition.json"
    )
    counter = (
        json.loads(counter_path.read_text(encoding="utf-8"))
        if counter_path.is_file()
        else {"states": []}
    )
    counter_states = {
        (int(row["frame"]), str(row["label"])): row for row in counter.get("states", [])
    }
    endpoint_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    endpoint_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    constraint_inputs: list[dict[str, Any]] = []

    def add_endpoint(frame: int, label: str, row: dict[str, Any]) -> None:
        endpoint = dict(row)
        endpoint["frame"] = int(frame)
        endpoint["state"] = str(label)
        endpoint["label"] = str(label)
        endpoint_by_key[(int(frame), str(label))] = endpoint

    for row in counter.get("states", []):
        frame = int(row.get("frame", -1))
        if frame in selected:
            add_endpoint(frame, str(row.get("label", "unknown")), row)
    for frame in selected:
        states: dict[str, dict[str, Any]] = {}
        for label in ("warm", "current_final"):
            row = counter_states.get((frame, label))
            if row:
                states["warm" if label == "warm" else "final"] = row
        if projection and (projection / "projection_solver_results.json").is_file():
            result_payload = json.loads(
                (projection / "projection_solver_results.json").read_text(encoding="utf-8")
            )
            projection_rows = [
                row
                for row in result_payload.get("results", [])
                if int(row.get("frame", -1)) == frame and row.get("state")
            ]
            if projection_rows:
                chosen = sorted(
                    projection_rows,
                    key=lambda row: (
                        not bool(row.get("strict_projection_acceptance")),
                        float(row.get("projection_objective", math.inf)),
                    ),
                )[0]
                context = _context(bundle, frame)
                state = np.asarray(chosen["state"], dtype=np.float64)
                states["projection"] = {
                    "base_pose_scene": _base_pose_from_value(context, state),
                    "qpos": state[6:28],
                    "slack": state[28:],
                }
        if "warm" not in states:
            warm = bundle["inputs"]["warm"]
            states["warm"] = {
                "base_pose_scene": warm.arrays["base_pose_scene"][frame],
                "qpos": warm.arrays["qpos"][frame],
                "slack": np.zeros(512),
            }
        if "final" not in states:
            final = bundle["inputs"]["final"]
            states["final"] = {
                "base_pose_scene": final.arrays["base_pose_scene"][frame],
                "qpos": final.arrays["qpos"][frame],
                "slack": _full_slack(final, frame),
            }
        for name in ("warm", "final", "projection"):
            if name in states:
                evaluated = _evaluate_state(
                    bundle,
                    frame,
                    states[name]["base_pose_scene"],
                    states[name]["qpos"],
                    states[name].get("slack"),
                )
                states[name] = evaluated
                add_endpoint(frame, name, evaluated)
        endpoint, variable = _directional_attribution(bundle, frame, states)
        endpoint_rows.extend(endpoint)
        variable_rows.extend(variable)
        constraint_inputs.append({"frame": frame, "states": states})
    all_states = [
        endpoint_by_key[key] for key in sorted(endpoint_by_key, key=lambda item: (item[0], item[1]))
    ]
    term_names = (
        "weighted_e_im",
        "weighted_e_bone",
        "e_temporal",
        "e_base_pos",
        "e_base_rot",
        "e_slack",
    )
    endpoint_checks: list[bool] = []
    for row in all_states:
        terms = row.get("terms", {})
        valid = all(name in terms and np.isfinite(float(terms[name])) for name in term_names)
        row["formal_total_sum"] = (
            float(sum(float(terms[name]) for name in term_names)) if valid else None
        )
        row["total_sum_check"] = bool(valid)
        endpoint_checks.append(bool(valid))
    _write_json(
        destination / "objective_endpoint_decomposition.json",
        {
            "schema_version": SCHEMA,
            "states": all_states,
            "frames": list(selected),
            "total_sum_check": bool(endpoint_checks) and all(endpoint_checks),
            "endpoint_state_count": len(all_states),
            "endpoint_labels": sorted({str(row["state"]) for row in all_states}),
            "diagnostic_only": True,
        },
    )
    endpoint_csv: list[dict[str, Any]] = []
    for row in all_states:
        terms = row.get("terms", {})
        endpoint_csv.append(
            {
                "frame": row["frame"],
                "state": row["state"],
                "label": row.get("label", row["state"]),
                **terms,
                "formal_total_sum": row.get("formal_total_sum"),
                "total_sum_check": row.get("total_sum_check", False),
            }
        )
    _write_csv(destination / "objective_endpoint_decomposition.csv", endpoint_csv)
    _write_json(
        destination / "objective_directional_attribution.json",
        {
            "schema_version": SCHEMA,
            "rows": endpoint_rows,
            "path_sample_count": 101,
            "multiplier_available": False,
            "diagnostic_only": True,
        },
    )
    _write_csv(destination / "objective_directional_attribution.csv", endpoint_rows)
    _write_json(
        destination / "objective_variable_group_attribution.json",
        {
            "schema_version": SCHEMA,
            "rows": variable_rows,
            "groups": ["base_translation", "base_rotation", *ALL_FINGERS, "slack"],
            "diagnostic_only": True,
        },
    )
    _write_csv(destination / "objective_variable_group_attribution.csv", variable_rows)
    _write_json(
        destination / "attribution_manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "OBJECTIVE_ATTRIBUTION_COMPLETE",
            "frames": list(selected),
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "projection_root": str(projection) if projection else None,
            "counterfactual_root": str(counterfactual_root),
            "solver_invocation_count": 0,
            "diagnostic_only": True,
        },
    )
    return {
        "schema_version": SCHEMA,
        "states": all_states,
        "objective_rows": endpoint_rows,
        "variable_rows": variable_rows,
    }


def _constraint_rows(
    bundle: dict[str, Any], frame: int, projection_state: dict[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    inputs = bundle["inputs"]
    final = inputs["final"]
    context = _context(bundle, frame)
    warm = _evaluate_state(
        bundle,
        frame,
        inputs["warm"].arrays["base_pose_scene"][frame],
        inputs["warm"].arrays["qpos"][frame],
        np.zeros(512),
    )
    official = _evaluate_state(
        bundle,
        frame,
        final.arrays["base_pose_scene"][frame],
        final.arrays["qpos"][frame],
        _full_slack(final, frame),
    )
    projection = projection_state or official
    phi_w = warm["full_signed_distance"]
    phi_p = projection["full_signed_distance"]
    phi_f = official["full_signed_distance"]
    names = np.asarray(inputs["surface"].link_names).astype(str)
    geometries = np.asarray(inputs["surface"].geometry_ids).astype(str)
    qstart = int(final.arrays["query_offsets"][frame])
    qstop = int(final.arrays["query_offsets"][frame + 1])
    qids = np.asarray(final.arrays["query_ids_concat"][qstart:qstop], dtype=np.int64)
    rstart = (
        int(final.arrays["query_active_round_concat"][frame])
        if np.asarray(final.arrays["query_active_round_concat"]).ndim == 1
        and len(final.arrays["query_active_round_concat"]) == 60
        else 0
    )
    del rstart
    rounds = np.full(512, None, dtype=object)
    active_round = np.asarray(
        final.arrays.get("query_active_round_concat", np.empty(0)), dtype=object
    )
    if len(active_round) == len(final.arrays["query_ids_concat"]):
        rounds[qids] = active_round[qstart:qstop]
    context_p = _state_value(
        context, projection["base_pose_scene"], projection["qpos"], projection["slack"]
    )
    jac_p = context.collision_points_jacobian_numpy(context_p)
    projection_points = dynamic_collision_points_numpy(
        inputs["model"],
        inputs["surface"],
        projection["qpos"],
        projection["base_pose_scene"],
    )
    warm_points = dynamic_collision_points_numpy(
        inputs["model"],
        inputs["surface"],
        warm["qpos"],
        warm["base_pose_scene"],
    )
    projection_displacement = np.linalg.norm(projection_points - warm_points, axis=1)
    # Endpoint differences are retained as a bounded directional attribution;
    # this report does not fabricate optimizer multipliers.
    dphi_wp = phi_p - phi_w
    dphi_pf = phi_f - phi_p
    rows: list[dict[str, Any]] = []
    for sample in range(512):
        link = str(names[sample])
        finger = next((item for item in ALL_FINGERS if item in link.lower()), "palm")
        hard_w = float(phi_w[sample] + _paper(bundle).b)
        soft_w = float(phi_w[sample] + _paper(bundle).tau)
        hard_p = float(phi_p[sample] + _paper(bundle).b)
        soft_p = float(phi_p[sample] + projection["slack"][sample] + _paper(bundle).tau)
        hard_f = float(phi_f[sample] + _paper(bundle).b)
        soft_f = float(phi_f[sample] + official["slack"][sample] + _paper(bundle).tau)
        components = {
            "warm_violation_m": max(0.0, -soft_w),
            "warm_hard_violation_m": max(0.0, -hard_w),
            "violation_reduction_m": max(0.0, max(0.0, -soft_w) - max(0.0, -soft_p)),
            "near_binding_persistence": float(
                abs(soft_p) <= NEAR_BINDING["soft_0_1_mm"]
                or abs(hard_p) <= NEAR_BINDING["hard_0_1_mm"]
            ),
            "directional_derivative_m": float(abs(dphi_wp[sample]) + abs(dphi_pf[sample])),
            "point_jacobian_norm": float(np.linalg.norm(jac_p[sample])),
            "projection_displacement_m": float(projection_displacement[sample]),
        }
        score = float(sum(components.values()))
        rows.append(
            {
                "frame": frame,
                "sample_id": sample,
                "geometry_id": str(geometries[sample]),
                "robot_link": link,
                "finger_region": finger,
                "local_point": np.asarray(inputs["surface"].points_local[sample]),
                "warm_phi_m": phi_w[sample],
                "projection_phi_m": phi_p[sample],
                "final_phi_m": phi_f[sample],
                "warm_hard_residual_m": hard_w,
                "warm_soft_residual_m": soft_w,
                "projection_hard_residual_m": hard_p,
                "projection_soft_residual_m": soft_p,
                "final_hard_residual_m": hard_f,
                "final_soft_residual_m": soft_f,
                "projection_slack_m": projection["slack"][sample],
                "initial_queryset_membership": bool(sample in set(qids.tolist())),
                "added_round": rounds[sample],
                "near_binding_hard": bool(abs(hard_p) <= NEAR_BINDING["hard_0_1_mm"]),
                "near_binding_soft": bool(abs(soft_p) <= NEAR_BINDING["soft_0_1_mm"]),
                "near_binding_raw": bool(abs(phi_p[sample]) <= NEAR_BINDING["raw_0_5_mm"]),
                "dphi_dalpha_warm_to_projection_m": dphi_wp[sample],
                "dphi_dalpha_projection_to_final_m": dphi_pf[sample],
                "point_displacement_warm_to_projection_m": components["projection_displacement_m"],
                "point_jacobian_norm": components["point_jacobian_norm"],
                "joint_jacobian_norm": float(np.linalg.norm(jac_p[sample, :, 6:])),
                "pressure_components": components,
                "pressure_score": score,
            }
        )
    aggregates: list[dict[str, Any]] = []
    for group_key in ("robot_link", "finger_region", "geometry_id"):
        groups = sorted(set(str(row[group_key]) for row in rows))
        for group in groups:
            values = np.asarray(
                [float(row["pressure_score"]) for row in rows if str(row[group_key]) == group]
            )
            aggregates.append(
                {
                    "frame": frame,
                    "group_type": group_key,
                    "group": group,
                    "pressure_max": float(np.max(values)),
                    "pressure_sum": float(np.sum(values)),
                    "pressure_count": int(len(values)),
                    "pressure_p95": float(np.percentile(values, 95)),
                    "coverage_gap_overlap": bool(
                        "finger" in group.lower() or "palm" in group.lower()
                    ),
                }
            )
    finger_summary = [
        {
            "frame": frame,
            "finger": finger,
            "pressure_sum": float(
                sum(row["pressure_score"] for row in rows if row["finger_region"] == finger)
            ),
            "pressure_max": float(
                max(
                    [row["pressure_score"] for row in rows if row["finger_region"] == finger]
                    or [0.0]
                )
            ),
            "warm_rmse_m": warm["per_finger"].get(finger, {}).get("keypoint_rmse_m"),
            "projection_rmse_m": projection["per_finger"].get(finger, {}).get("keypoint_rmse_m"),
            "final_rmse_m": official["per_finger"].get(finger, {}).get("keypoint_rmse_m"),
            "qpos_displacement_l2": float(np.linalg.norm(projection["qpos"] - warm["qpos"])),
        }
        for finger in ALL_FINGERS
    ]
    joint = {
        "rows": finger_summary,
        "direct_vs_indirect": "finger_region link pressure is a diagnostic grouping; low own-q pressure with high motion indicates indirect/base coupling",
    }
    return rows, aggregates, finger_summary, joint


def run_constraints(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    projection_root: str | Path | None = None,
    frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    selected = tuple(frames) if frames else bundle["frames"]
    destination = _resolve(output_root, bundle["repo"])
    projection = _resolve(projection_root, bundle["repo"]) if projection_root else None
    result_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    finger_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for frame in selected:
        projection_state = None
        if projection and (projection / "projection_solver_results.json").is_file():
            payload = json.loads(
                (projection / "projection_solver_results.json").read_text(encoding="utf-8")
            )
            rows = [
                row
                for row in payload.get("results", [])
                if int(row.get("frame", -1)) == frame and row.get("state")
            ]
            if rows:
                chosen = sorted(
                    rows,
                    key=lambda row: (
                        not bool(row.get("strict_projection_acceptance")),
                        float(row.get("projection_objective", math.inf)),
                    ),
                )[0]
                context = _context(bundle, frame)
                state = np.asarray(chosen["state"], dtype=np.float64)
                projection_state = _evaluate_state(
                    bundle, frame, _base_pose_from_value(context, state), state[6:28], state[28:]
                )
        rows, aggregate, finger, joint = _constraint_rows(bundle, frame, projection_state)
        result_rows.extend(rows)
        aggregates.extend(aggregate)
        finger_rows.extend(finger)
        joint_rows.append({"frame": frame, **joint})
    _write_json(
        destination / "constraint_attribution.json",
        {
            "schema_version": SCHEMA,
            "pressure_schema": PRESSURE_SCHEMA,
            "rows": result_rows,
            "aggregates": aggregates,
            "near_binding_thresholds": NEAR_BINDING,
            "multiplier_available": False,
            "multiplier_note": "pressure is not a dual multiplier",
            "diagnostic_only": True,
        },
    )
    _write_csv(destination / "constraint_attribution_per_sample.csv", result_rows)
    _write_csv(
        destination / "constraint_attribution_per_link.csv",
        [row for row in aggregates if row["group_type"] == "robot_link"],
    )
    _write_csv(destination / "constraint_attribution_per_finger.csv", finger_rows)
    _write_json(
        destination / "interaction_constraint_joint_attribution.json",
        {"schema_version": SCHEMA, "rows": joint_rows, "diagnostic_only": True},
    )
    _write_csv(destination / "interaction_constraint_joint_attribution.csv", finger_rows)
    _write_json(
        destination / "constraint_manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "CONSTRAINT_ATTRIBUTION_COMPLETE",
            "frames": list(selected),
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "projection_root": str(projection) if projection else None,
            "solver_invocation_count": 0,
            "diagnostic_only": True,
        },
    )
    return {
        "schema_version": SCHEMA,
        "rows": result_rows,
        "aggregates": aggregates,
        "fingers": finger_rows,
    }


def _load_json_if(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _aggregate_root_cause(
    bundle: dict[str, Any],
    projection_root: Path | None,
    counter_root: Path,
    objective_root: Path,
    constraint_root: Path,
) -> dict[str, Any]:
    path_root = projection_root
    path = _load_json_if(path_root / "warm_final_path_feasibility.json", {}) if path_root else {}
    motion = _load_json_if(path_root / "feasibility_motion_fraction.json", {}) if path_root else {}
    projections = (
        _load_json_if(path_root / "projection_solver_results.json", {}) if path_root else {}
    )
    counter = _load_json_if(counter_root / "state_counterfactual_decomposition.json", {})
    constraints = _load_json_if(constraint_root / "constraint_attribution.json", {})
    rows = counter.get("states", [])
    evidence_frames = sorted(
        {
            int(frame)
            for frame in [
                *path.get("per_frame", {}).keys(),
                *[row.get("frame") for row in rows if row.get("frame") is not None],
                *[
                    row.get("frame")
                    for row in constraints.get("rows", [])
                    if row.get("frame") is not None
                ],
            ]
        }
    )
    causes: list[dict[str, Any]] = []
    warm_frames = [
        str(frame)
        for frame, item in path.get("per_frame", {}).items()
        if bool(item.get("warm_soft_feasible"))
    ]
    if warm_frames:
        causes.append(
            {
                "cause": "WARM_ALREADY_FEASIBLE",
                "confidence": "high",
                "evidence_for": ["warm is soft-safe at alpha=0 on these selected frames"],
                "evidence_against": [],
                "frames": warm_frames,
                "fingers_links": [],
                "quantitative_evidence": {"warm_soft_feasible_frame_count": len(warm_frames)},
                "next_action": "NO_STAGE9_4_REQUIRED",
            }
        )
    solved = [
        row for row in projections.get("results", []) if row.get("strict_projection_acceptance")
    ]
    motion_by_key = {
        (int(row["frame"]), str(row["profile"])): row
        for row in motion.get("rows", [])
        if row.get("strict_projection_acceptance")
    }
    solved_by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in solved:
        solved_by_frame.setdefault(int(row["frame"]), []).append(row)
    selected_projection: dict[int, dict[str, Any]] = {}
    for frame, values in solved_by_frame.items():
        selected_projection[frame] = min(
            values, key=lambda row: float(row.get("projection_objective", math.inf))
        )
    per_frame: list[dict[str, Any]] = []
    qualified_frames: list[int] = []
    for frame in bundle["frames"]:
        path_item = path.get("per_frame", {}).get(str(frame), {})
        candidate = selected_projection.get(int(frame))
        motion_item = (
            motion_by_key.get((int(frame), str(candidate.get("profile")))) if candidate else None
        )
        improvement = (
            float(candidate["long_finger_rmse_improvement_m"])
            if candidate and candidate.get("long_finger_rmse_improvement_m") is not None
            else None
        )
        required_improvement = (
            max(0.001, 0.1 * float(candidate.get("long_finger_rmse_m", 0.0)))
            if candidate and candidate.get("long_finger_rmse_m") is not None
            else None
        )
        canonical_pass = bool(
            candidate
            and float(candidate.get("hard_violation_m", math.inf)) <= 1e-8
            and float(candidate.get("soft_violation_m", math.inf)) <= 1e-8
            and float(candidate.get("raw_penetration_m", math.inf)) <= 1e-8
            and bool(candidate.get("projection_objective_threshold_pass"))
        )
        state_fraction = float(motion_item["state_fraction"]) if motion_item is not None else None
        projection_closer_warm = state_fraction is not None and state_fraction <= 0.5
        improvement_pass = (
            improvement is not None
            and required_improvement is not None
            and improvement >= required_improvement
        )
        if bool(path_item.get("warm_soft_feasible")):
            classification = "WARM_ALREADY_FEASIBLE"
            next_action = "NO_STAGE9_4_REQUIRED_FOR_THIS_FRAME"
        elif candidate and projection_closer_warm and canonical_pass and improvement_pass:
            classification = "OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY"
            qualified_frames.append(int(frame))
            next_action = "candidate may contribute to aggregate Stage 9.4 gate"
        else:
            classification = "INCONCLUSIVE"
            next_action = "do not use this frame to authorize Stage 9.4"
        per_frame.append(
            {
                "frame": int(frame),
                "classification": classification,
                "confidence": "high" if classification == "WARM_ALREADY_FEASIBLE" else "low",
                "candidate_profile": candidate.get("profile") if candidate else None,
                "evidence_for": [
                    "warm soft-safe at alpha=0"
                    if classification == "WARM_ALREADY_FEASIBLE"
                    else "no complete projection causal gate"
                ],
                "evidence_against": []
                if classification == "WARM_ALREADY_FEASIBLE"
                else [
                    name
                    for name, passed in (
                        ("strict canonical projection", bool(candidate and canonical_pass)),
                        ("projection closer to warm", projection_closer_warm),
                        ("long-finger RMSE improvement gate", improvement_pass),
                    )
                    if not passed
                ],
                "fingers_links": [],
                "quantitative_evidence": {
                    "state_fraction": state_fraction,
                    "long_finger_rmse_improvement_m": improvement,
                    "required_long_finger_improvement_m": required_improvement,
                    "strict_projection_available": candidate is not None,
                    "projection_objective_threshold_pass": (
                        candidate.get("projection_objective_threshold_pass") if candidate else None
                    ),
                },
                "next_action": next_action,
            }
        )
    if not solved:
        causes.append(
            {
                "cause": "INCONCLUSIVE",
                "confidence": "high",
                "evidence_for": [
                    "no strict projection solution was available for causal comparison"
                ],
                "evidence_against": ["current final is a known feasible seed"],
                "frames": evidence_frames,
                "fingers_links": [],
                "quantitative_evidence": {"projection_solved_count": 0},
                "next_action": "RETURN_TO_PROJECTION_DIAGNOSTIC_HARNESS_FIX",
            }
        )
    else:
        valid_fraction = [
            float(motion_by_key[(frame, str(row["profile"]))]["state_fraction"])
            for frame, row in selected_projection.items()
            if (frame, str(row["profile"])) in motion_by_key
        ]
        median_fraction = float(np.median(valid_fraction or [1.0]))
        official_gate_pass = bool(len(qualified_frames) >= 2 and median_fraction <= 0.5)
        if official_gate_pass:
            causes.append(
                {
                    "cause": "OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY",
                    "confidence": "medium",
                    "evidence_for": [
                        "at least two selected frames pass the complete projection gate"
                    ],
                    "evidence_against": [],
                    "frames": [str(frame) for frame in qualified_frames],
                    "fingers_links": [],
                    "quantitative_evidence": {
                        "strict_projection_frame_count": len(selected_projection),
                        "qualified_frame_count": len(qualified_frames),
                        "median_state_fraction": median_fraction,
                    },
                    "next_action": "READY_FOR_STAGE9_4_REFINEMENT_ENGINEERING_REPAIR",
                }
            )
        else:
            failed_frames = [item for item in per_frame if item["classification"] == "INCONCLUSIVE"]
            causes.append(
                {
                    "cause": "INCONCLUSIVE",
                    "confidence": "high",
                    "evidence_for": ["some projection states pass independent full512 validation"],
                    "evidence_against": [
                        "the complete OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY gate is not met"
                    ],
                    "frames": [item["frame"] for item in failed_frames],
                    "fingers_links": [],
                    "quantitative_evidence": {
                        "strict_projection_frame_count": len(selected_projection),
                        "qualified_frame_count": len(qualified_frames),
                        "required_qualified_frame_count": 2,
                        "median_state_fraction": median_fraction,
                        "failed_frame_gates": failed_frames,
                    },
                    "next_action": "STAGE9_4_NOT_YET_JUSTIFIED",
                }
            )
    by_label = {(int(row["frame"]), str(row["label"])): row for row in rows}
    for finger, label in (
        ("index", "warm_plus_final_index_q"),
        ("middle", "warm_plus_final_middle_q"),
        ("ring", "warm_plus_final_ring_q"),
        ("long_finger", "warm_plus_final_long_finger_q"),
    ):
        effects: list[float] = []
        counterfactual_values: list[float] = []
        for frame in bundle["frames"]:
            warm = by_label.get((frame, "warm"))
            final = by_label.get((frame, "current_final"))
            own = by_label.get((frame, label))
            if warm and final and own:
                own_value = (
                    float(own["long_finger_rmse_m"])
                    if finger == "long_finger"
                    else float(own["per_finger"][finger]["keypoint_rmse_m"])
                )
                warm_value = (
                    float(warm["long_finger_rmse_m"])
                    if finger == "long_finger"
                    else float(warm["per_finger"][finger]["keypoint_rmse_m"])
                )
                counterfactual_values.append(own_value)
                effects.append(own_value - warm_value)
        if effects:
            causes.append(
                {
                    "cause": "LONG_FINGER_QPOS_DRIVES_DEGRADATION"
                    if finger == "long_finger"
                    else f"{finger.upper()}_QPOS_PRIMARY",
                    "confidence": "medium",
                    "evidence_for": [
                        f"{label} counterfactual evaluated on {len(effects)} selected frames"
                    ],
                    "evidence_against": [
                        "counterfactual is read-only and may contain cross-finger coupling"
                    ],
                    "frames": evidence_frames,
                    "fingers_links": [finger],
                    "quantitative_evidence": {
                        "counterfactual_rmse_values_m": counterfactual_values,
                        "counterfactual_delta_from_warm_m": effects,
                    },
                    "next_action": "retain as diagnostic attribution",
                }
            )
    base_effects: list[float] = []
    q_effects: list[float] = []
    interaction_effects: list[float] = []
    for frame in bundle["frames"]:
        warm = by_label.get((frame, "warm"))
        base_only = by_label.get((frame, "final_base_warm_q"))
        q_only = by_label.get((frame, "warm_base_final_all_q"))
        final = by_label.get((frame, "current_final"))
        if warm and base_only and q_only and final:
            warm_value = float(warm["long_finger_rmse_m"])
            base_value = float(base_only["long_finger_rmse_m"])
            q_value = float(q_only["long_finger_rmse_m"])
            final_value = float(final["long_finger_rmse_m"])
            base_effects.append(base_value - warm_value)
            q_effects.append(q_value - warm_value)
            interaction_effects.append(final_value - base_value - q_value + warm_value)
    if base_effects:
        base_dominates = all(
            abs(base) > abs(q) for base, q in zip(base_effects, q_effects, strict=True)
        )
        causes.append(
            {
                "cause": "BASE_MOTION_DRIVES_LONG_FINGER_DEGRADATION",
                "confidence": "medium" if base_dominates else "low",
                "evidence_for": [
                    "final_base_warm_q reproduces more long-finger error than warm_base_final_all_q"
                ],
                "evidence_against": [
                    "counterfactual and projection gates are diagnostic and do not prove formal causality"
                ],
                "frames": evidence_frames,
                "fingers_links": ["long_finger", "base"],
                "quantitative_evidence": {
                    "base_effect_m": base_effects,
                    "all_q_effect_m": q_effects,
                    "base_q_interaction_m": interaction_effects,
                },
                "next_action": "retain as diagnostic attribution; do not enter Stage 9.4",
            }
        )
    if constraints.get("aggregates"):
        top = sorted(
            [row for row in constraints["aggregates"] if row.get("group_type") == "robot_link"],
            key=lambda row: -float(row.get("pressure_sum", 0.0)),
        )[:5]
        causes.append(
            {
                "cause": "COLLISION_VISUAL_COVERAGE_GAP_DOMINATES",
                "confidence": "low",
                "evidence_for": [
                    "top pressure links are reported with canonical visual/collision coverage context"
                ],
                "evidence_against": [
                    "pressure score is diagnostic and not a dual multiplier",
                    "causal closure depends on strict projection",
                ],
                "frames": evidence_frames,
                "fingers_links": [row["group"] for row in top],
                "quantitative_evidence": {"top_links": top},
                "next_action": "do not repair geometry until projection gate is solved",
            }
        )
    primary = "STAGE9_3_5_INCONCLUSIVE"
    if not solved:
        primary = "RETURN_TO_PROJECTION_DIAGNOSTIC_HARNESS_FIX"
    elif len(qualified_frames) >= 2 and valid_fraction and float(np.median(valid_fraction)) <= 0.5:
        primary = "READY_FOR_STAGE9_4_REFINEMENT_ENGINEERING_REPAIR"
    elif solved:
        primary = "STAGE9_4_NOT_YET_JUSTIFIED"
    return {
        "schema_version": SCHEMA,
        "primary": primary,
        "aggregate_classification": (
            "OFFICIAL_FINAL_MOVES_BEYOND_FEASIBILITY"
            if primary == "READY_FOR_STAGE9_4_REFINEMENT_ENGINEERING_REPAIR"
            else "INCONCLUSIVE"
        ),
        "per_frame_classifications": per_frame,
        "causes": causes,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "human_decision_required": True,
        "stop_after_stage9_3_5": True,
    }


def _html_state(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "frame",
            "state",
            "label",
            "long_finger_rmse_m",
            "long_finger_morphology_normalized_rmse",
            "per_finger",
            "terms",
            "contact_proxy",
            "min_sdf_m",
            "raw_penetration_m",
            "hard_violation_m",
            "soft_violation_m",
            "slack_bounds_pass",
            "qpos_bounds_pass",
            "base_valid",
            "full512_finite",
            "base_translation_from_warm_m",
            "base_rotation_from_warm_rad",
            "qpos_displacement_from_warm_l2",
            "state_displacement",
            "formal_total_sum",
            "total_sum_check",
        )
        if key in row
    }


def _html_projection(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "frame",
        "global_frame",
        "profile",
        "status",
        "solver",
        "solver_version",
        "solver_status",
        "solver_success",
        "strict_projection_acceptance",
        "projection_feasible",
        "projection_objective",
        "projection_objective_threshold",
        "projection_objective_threshold_pass",
        "min_sdf_m",
        "raw_penetration_m",
        "hard_violation_m",
        "soft_violation_m",
        "long_finger_rmse_m",
        "long_finger_rmse_improvement_m",
        "contact_proxy",
        "state_displacement",
        "attempt_count",
        "runtime_s",
    )
    return {key: row.get(key) for key in fields if key in row}


def _html_pressure(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "frame",
        "sample_id",
        "geometry_id",
        "robot_link",
        "finger_region",
        "warm_phi_m",
        "projection_phi_m",
        "final_phi_m",
        "projection_soft_residual_m",
        "near_binding_hard",
        "near_binding_soft",
        "near_binding_raw",
        "dphi_dalpha_warm_to_projection_m",
        "dphi_dalpha_projection_to_final_m",
        "point_displacement_warm_to_projection_m",
        "point_jacobian_norm",
        "joint_jacobian_norm",
        "pressure_score",
    )
    return {key: row.get(key) for key in fields if key in row}


def _html_path_traces(path_csv: Path, *, max_points: int = 201) -> dict[str, list[dict[str, Any]]]:
    fields_float = (
        "alpha",
        "min_sdf_m",
        "hard_residual_min_m",
        "soft_residual_min_zero_slack_m",
        "required_slack_max_m",
        "long_finger_rmse_m",
        "thumb_rmse_m",
        "index_rmse_m",
        "middle_rmse_m",
        "ring_rmse_m",
        "pinky_rmse_m",
        "formal_total_objective_zero_slack",
        "formal_total_objective_minimal_legal_slack",
    )
    fields_bool = ("hard_feasible", "soft_safe_feasible", "zero_penetration_feasible")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in _read_csv_rows(path_csv):
        frame = str(raw.get("frame", ""))
        if not frame:
            continue
        row: dict[str, Any] = {"frame": int(frame)}
        if raw.get("global_frame"):
            row["global_frame"] = int(float(raw["global_frame"]))
        for field in fields_float:
            if raw.get(field, "") not in {"", "None"}:
                row[field] = float(raw[field])
        for field in fields_bool:
            row[field] = str(raw.get(field, "")).lower() in {"1", "true", "yes"}
        grouped.setdefault(frame, []).append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for frame, rows in grouped.items():
        if len(rows) > max_points:
            indices = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
            rows = [rows[int(index)] for index in indices]
        result[frame] = rows
    return result


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(_jsonable(payload), separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 9.3.5 Projection and Causal Closure</title>
<style>
:root{color-scheme:dark}body{font:14px system-ui,sans-serif;background:#0b1220;color:#e5e7eb;margin:0;padding:18px}
h1,h2{margin:.4em 0}.toolbar,.grid{display:grid;gap:10px}.toolbar{grid-template-columns:repeat(6,minmax(120px,1fr));background:#111c2e;padding:12px;border-radius:8px}
label{display:flex;flex-direction:column;gap:4px;color:#a5b4fc}select,input{background:#172338;color:#f3f4f6;border:1px solid #475569;border-radius:4px;padding:6px}
.grid{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:10px}.wide{grid-column:1/-1}.card{background:#111c2e;padding:12px;border-radius:8px;overflow:auto}
canvas{width:100%;height:190px;background:#0a1020;border:1px solid #334155}.metric{font-size:1.05em}.ok{color:#86efac}.warn{color:#fbbf24}.bad{color:#fca5a5}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12px}td,th{border:1px solid #334155;padding:5px;text-align:right}td:first-child,th:first-child{text-align:left}th{color:#c4b5fd;position:sticky;top:0;background:#111c2e}
pre{white-space:pre-wrap;max-height:280px;overflow:auto;background:#0a1020;padding:8px}.small{color:#94a3b8;font-size:12px}.bar{height:8px;background:#26344b}.bar>i{display:block;height:100%;background:#60a5fa}
@media(max-width:900px){.toolbar,.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body>
<h1>Stage 9.3.5 Projection and Causal Closure</h1>
<p class="small">Diagnostic only. Canonical full-512 signed distance, warm/projection/final state comparison, causal counterfactuals, and a conservative Stage 9.4 gate. Units: mm, deg, rad, objective units, and signed distance (m) as labelled.</p>
<div class="toolbar">
<label>Frame<select id="frame"></select></label><label>State view<select id="view"><option value="warm">Warm</option><option value="projection">Projection</option><option value="final">Final</option><option value="counterfactual">Counterfactual</option></select></label>
<label>Counterfactual<select id="state"></select></label><label>Path alpha<input id="alpha" type="range" min="0" max="1000" value="0" step="1"><span id="alphaLabel"></span></label>
<label>Pressure link<select id="pressureLink"><option value="">All links</option></select></label><label>Pressure finger<select id="pressureFinger"><option value="">All fingers</option></select></label>
</div>
<div class="grid">
<section class="card"><h2>Readiness / root cause</h2><div id="readiness" class="metric"></div><pre id="cause"></pre></section>
<section class="card"><h2>Selected state</h2><div id="stateHeadline" class="metric"></div><table id="stateMetrics"></table></section>
<section class="card"><h2>Projection / branch</h2><div id="projectionSummary"></div><table id="motionSummary"></table><pre id="branch"></pre></section>
<section class="card wide"><h2>Warm → final path; alpha slider and feasible intervals</h2><canvas id="pathPlot" width="1000" height="190"></canvas><div id="pathSummary"></div></section>
<section class="card wide"><h2>Per-finger RMSE timeline (fixed global scale)</h2><canvas id="fingerTimeline" width="1000" height="190"></canvas><div id="fingerTable"></div></section>
<section class="card wide"><h2>Objective endpoint decomposition</h2><table id="objectiveTable"></table><h3>Directional/path attribution</h3><table id="directionalTable"></table><h3>Variable-group attribution</h3><table id="variableTable"></table></section>
<section class="card wide"><h2>Constraint sample pressure and interaction gradient</h2><table id="pressureTable"></table><h3>Aggregates</h3><table id="aggregateTable"></table></section>
<section class="card wide"><h2>Projection attempts</h2><table id="projectionTable"></table></section>
</div>
<script>
const DATA=__DATA__;const $=id=>document.getElementById(id);const FINGERS=['thumb','index','middle','ring','pinky'];
const SCALE=DATA.global_scale||{rmse_m:.01,pressure_score:1,objective:1,signed_distance_m:.01};
const frames=(DATA.frames||[]).map(Number);const stateRows=DATA.states||[];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(v,d=4)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d);
const mm=v=>v==null||!Number.isFinite(Number(v))?'—':(Number(v)*1000).toFixed(3)+' mm';
const raddeg=v=>v==null||!Number.isFinite(Number(v))?'—':(Number(v)*180/Math.PI).toFixed(3)+' deg';
const rowsForFrame=(rows,f)=>rows.filter(r=>Number(r.frame)===Number(f));
function fill(el,values,labels=values){el.innerHTML='';values.forEach((v,i)=>el.add(new Option(String(labels[i]),String(v))))}
function rowFor(label,f){return stateRows.find(r=>Number(r.frame)===Number(f)&&String(r.state)===label)||stateRows.find(r=>Number(r.frame)===Number(f)&&String(r.label)===label)}
function selectedState(f){const view=$('view').value;let label=view==='warm'?'warm':view==='final'?'final':view==='projection'?'projection':$('state').value;return rowFor(label,f)||rowFor(view==='final'?'current_final':label,f)||rowsForFrame(stateRows,f)[0]||{}}
function pathRows(f){return (DATA.path_traces||{})[String(f)]||[]}
function draw(canvas,rows,key,color,scale){const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#334155';ctx.strokeRect(30,10,w-45,h-30);if(!rows.length)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((r,i)=>{const x=30+(w-45)*(i/Math.max(rows.length-1,1));const value=Number(r[key]);const y= h-20-((Number.isFinite(value)?Math.max(-scale,Math.min(scale,value)):0)+scale)/(2*scale)*(h-30);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
function table(id,headers,rows,limit=100){const t=$(id);t.innerHTML='<tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr>'+rows.slice(0,limit).map(r=>'<tr>'+headers.map(h=>'<td>'+esc(r[h])+'</td>').join('')+'</tr>').join('')}
function renderState(f){const r=selectedState(f);$('stateHeadline').innerHTML='<b>'+esc(r.label||r.state||'—')+'</b> · long-finger RMSE '+mm(r.long_finger_rmse_m)+' · min SDF '+mm(r.min_sdf_m);const metric=[['long finger RMSE',mm(r.long_finger_rmse_m)],['raw penetration',mm(r.raw_penetration_m)],['hard violation',mm(r.hard_violation_m)],['soft violation',mm(r.soft_violation_m)],['base translation from warm',mm(r.base_translation_from_warm_m)],['base rotation from warm',raddeg(r.base_rotation_from_warm_rad)],['q displacement from warm',num(r.qpos_displacement_from_warm_l2)],['state metric',num(r.state_displacement)],['full512 finite',r.full512_finite],['sum check',r.total_sum_check]];table('stateMetrics',['Metric','Value'],metric.map(x=>({Metric:x[0],Value:x[1]})));}
function renderPath(f){const rows=pathRows(f);const a=Number($('alpha').value)/1000;let index=rows.length?Math.round(a*(rows.length-1)):0;index=Math.max(0,Math.min(rows.length-1,index));const r=rows[index]||{};$('alphaLabel').textContent=' alpha='+num(r.alpha,3);draw($('pathPlot'),rows,'min_sdf_m','#60a5fa',SCALE.signed_distance_m);const p=(DATA.path||{})[String(f)]||{};$('pathSummary').innerHTML='<b>alpha '+num(r.alpha,3)+'</b> · min SDF '+mm(r.min_sdf_m)+' · required slack '+mm(r.required_slack_max_m)+' · soft-safe '+esc(r.soft_safe_feasible)+' · hard '+esc(r.hard_feasible)+'<br>Soft intervals: <code>'+esc(JSON.stringify(p.soft_safe_intervals||[]))+'</code><br>Hard intervals: <code>'+esc(JSON.stringify(p.hard_feasible_intervals||[]))+'</code>';const fingerRows=FINGERS.map(x=>({Finger:x,RMSE:mm(r[x+'_rmse_m'])}));table('fingerTable',['Finger','RMSE'],fingerRows);draw($('fingerTimeline'),rows,'long_finger_rmse_m','#f59e0b',SCALE.rmse_m);}
function renderObjective(f){const end=rowsForFrame(DATA.objective_endpoints||[],f).map(r=>({State:r.state,TermTotal:num(r.formal_total_sum,6),Eim:num(r.weighted_e_im,6),Ebone:num(r.weighted_e_bone,6),Temporal:num(r.e_temporal,6),Base:num(Number(r.e_base_pos||0)+Number(r.e_base_rot||0),6),Slack:num(r.e_slack,6),Check:r.total_sum_check}));table('objectiveTable',['State','TermTotal','Eim','Ebone','Temporal','Base','Slack','Check'],end,200);const dir=rowsForFrame(DATA.objective_directional||[],f).map(r=>({Transition:r.transition,Term:r.term,Delta:num(r.delta,6),Left:num(r.left_value,6),Right:num(r.right_value,6),PathIntegral:num(r.path_integrated_by_term&&r.path_integrated_by_term[r.term],6)}));table('directionalTable',['Transition','Term','Delta','Left','Right','PathIntegral'],dir,120);const vars=rowsForFrame(DATA.variable_group||[],f).map(r=>({Transition:r.transition,Term:r.term,Group:r.variable_group,Contribution:num(r.path_integrated_contribution,6),EndpointDelta:num(r.endpoint_delta,6)}));table('variableTable',['Transition','Term','Group','Contribution','EndpointDelta'],vars,160);}
function renderPressure(f){let link=$('pressureLink').value,finger=$('pressureFinger').value;const pressure=rowsForFrame(DATA.pressure||[],f).filter(r=>(!link||r.robot_link===link)&&(!finger||r.finger_region===finger)).sort((a,b)=>Number(b.pressure_score)-Number(a.pressure_score)).map(r=>({Sample:r.sample_id,Link:r.robot_link,Finger:r.finger_region,Pressure:num(r.pressure_score,5),WarmPhi:mm(r.warm_phi_m),ProjectionPhi:mm(r.projection_phi_m),Gradient:num(r.dphi_dalpha_warm_to_projection_m,6),Jacobian:num(r.joint_jacobian_norm,4)}));table('pressureTable',['Sample','Link','Finger','Pressure','WarmPhi','ProjectionPhi','Gradient','Jacobian'],pressure,80);const ag=rowsForFrame(DATA.pressure_aggregates||[],f).map(r=>({Type:r.group_type,Group:r.group,Max:num(r.pressure_max,5),Sum:num(r.pressure_sum,5),P95:num(r.pressure_p95,5),Count:r.pressure_count}));table('aggregateTable',['Type','Group','Max','Sum','P95','Count'],ag,100);}
function renderProjection(f){const raw=rowsForFrame(DATA.projection_results||[],f);const rows=raw.map(r=>({Profile:r.profile,Status:r.status,Strict:r.strict_projection_acceptance,Feasible:r.projection_feasible,Objective:num(r.projection_objective,6),Threshold:num(r.projection_objective_threshold,6),LongImprovement:mm(r.long_finger_rmse_improvement_m),Runtime:num(r.runtime_s,2)+' s',Attempts:r.attempt_count}));table('projectionTable',['Profile','Status','Strict','Feasible','Objective','Threshold','LongImprovement','Runtime','Attempts'],rows,20);$('projectionSummary').textContent='Projection attempts: '+raw.length+' · strict accepted: '+raw.filter(r=>r.strict_projection_acceptance).length+' · units: objective, mm, signed distance (m), slack (m).';const motion=rowsForFrame(DATA.motion||[],f).map(r=>({Profile:r.profile,StateFraction:num(r.state_fraction,4),QFraction:num(r.qpos_fraction,4),BaseTranslation:num(r.base_translation_fraction,4),BaseRotation:num(r.base_rotation_fraction,4),Strict:r.strict_projection_acceptance}));table('motionSummary',['Profile','StateFraction','QFraction','BaseTranslation','BaseRotation','Strict'],motion,20);$('branch').textContent=JSON.stringify(DATA.branch||{},null,2);}
function render(){const f=Number($('frame').value);renderState(f);renderPath(f);renderObjective(f);renderPressure(f);renderProjection(f);const r=DATA.readiness||{};$('readiness').innerHTML='<span class="'+(r.ENTER_STAGE9_4==='YES'?'ok':'warn')+'">ENTER_STAGE9_4='+esc(r.ENTER_STAGE9_4)+'</span> · HUMAN_DECISION_REQUIRED='+esc(r.HUMAN_DECISION_REQUIRED)+' · STOP_AFTER_STAGE9_3_5='+esc(r.STOP_AFTER_STAGE9_3_5);$('cause').textContent=JSON.stringify(DATA.root_cause||{},null,2)}
function updateStates(){const f=Number($('frame').value),rows=rowsForFrame(stateRows,f),cf=rows.filter(r=>String(r.state).includes('q')||String(r.label).includes('q'));fill($('state'),cf.map(r=>r.state||r.label),cf.map(r=>r.label||r.state));render()}
fill($('frame'),frames);frames.length&&($('frame').value=String(frames[0]));['thumb','index','middle','ring','pinky'].forEach(x=>$('pressureFinger').add(new Option(x,x)));[...new Set((DATA.pressure||[]).map(r=>r.robot_link))].sort().forEach(x=>$('pressureLink').add(new Option(x,x)));$('frame').onchange=updateStates;$('view').onchange=render;$('state').onchange=render;$('alpha').oninput=renderPath;$('pressureLink').onchange=()=>renderPressure(Number($('frame').value));$('pressureFinger').onchange=()=>renderPressure(Number($('frame').value));updateStates();
</script></body></html>""".replace("__DATA__", encoded)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def _candidate_plan(root_cause: dict[str, Any], bundle: dict[str, Any], reports: Path) -> str:
    route = str(root_cause.get("primary", "INCONCLUSIVE"))
    if route == "READY_FOR_STAGE9_4_REFINEMENT_ENGINEERING_REPAIR":
        direction = (
            "projection-informed initialization and deterministic feasible-candidate selection"
        )
        profile = "stage9_4_projection_informed_refinement_v1"
    elif route == "READY_FOR_STAGE9_4_FAITHFUL_GEOMETRY_REPAIR":
        direction = "faithful collision/sample coverage repair"
        profile = "stage9_4_faithful_geometry_repair_audit_v1"
    else:
        direction = (
            "no Stage 9.4 implementation authorized until the projection harness is repaired"
        )
        profile = "stage9_3_5_projection_harness_repair_v1"
    return f"""# Stage 9.4 Candidate Plan\n\n主路由：`{route}`\n\n## 因果证据\n\n- Stage 9.3.5 remains diagnostic-only and does not alter Eq. (1)--(9).\n- Primary route is selected from `{reports / "root_cause_analysis.json"}`.\n- Current-lineage baseline and Stage 10 accepted artifacts remain read-only.\n\n## 应修改模块\n\n- {direction}.\n- Candidate profile: `{profile}`.\n- Run root: `.local/runs/stage9_4/{profile}/s1__airplane_lift__right__artimano_rh__f000240_f000300/`.\n\n## 不应修改\n\n- Eq. (1)--(9), paper weights, Stage 7 warm-start, Stage 8 graph, historical Stage 9.2, current-lineage baseline, Stage 10 manifest, manual acceptance, or robot reference.\n- Do not add a formal contact attraction loss from this audit.\n\n## Regression matrix\n\n- Contact-rich: airplane lift / right / Arti-MANO RH.\n- Approach, pre-contact, LH, and another object sequence.\n- Re-run only the stages required by the selected module; repeat Stage 10 reference-runtime and manual acceptance if the accepted output changes.\n\n## Rollback\n\n- Keep this diagnostic root immutable. Run the candidate in a new `.local/runs/stage9_4` root and compare against the Stage 9.3.4 current baseline.\n"""


def run_status(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    *,
    projection_root: str | Path,
    counterfactual_root: str | Path,
    objective_root: str | Path,
    constraint_root: str | Path,
    branch_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    root = bundle["repo"]
    projection = _resolve(projection_root, root)
    counter = _resolve(counterfactual_root, root)
    objective = _resolve(objective_root, root)
    constraint = _resolve(constraint_root, root)
    branch = _resolve(branch_root, root)
    destination = _resolve(output_root, root)
    cause = _aggregate_root_cause(bundle, projection, counter, objective, constraint)
    immutability_before = _load_json_if(
        projection / "official_artifact_immutability_before.json",
        _official_artifact_snapshot(bundle),
    )
    immutability_after = _official_artifact_snapshot(bundle)
    immutability = _compare_official_artifact_snapshots(immutability_before, immutability_after)
    immutability["before_snapshot"] = str(projection / "official_artifact_immutability_before.json")
    immutability["after_snapshot"] = immutability_after
    _write_json(destination / "official_artifact_immutability.json", immutability)
    readiness = {
        "schema_version": SCHEMA,
        "status": cause["primary"],
        "enter_stage9_4": False,
        "ENTER_STAGE9_4": "NO",
        "human_decision_required": True,
        "HUMAN_DECISION_REQUIRED": "YES",
        "stop_after_stage9_3_5": True,
        "STOP_AFTER_STAGE9_3_5": "TRUE",
        "official_artifacts_changed": bool(immutability["official_artifacts_changed"]),
        "current_lineage_baseline_changed": bool(immutability["official_artifacts_changed"]),
        "official_artifact_immutability": str(destination / "official_artifact_immutability.json"),
        "reason": (
            "Official/current-lineage artifacts changed during diagnostics; fail closed."
            if immutability["official_artifacts_changed"]
            else "Stage 9.3.5 is diagnostic-only; a human must approve one Stage 9.4 direction."
        ),
        "root_cause": cause,
    }
    _write_json(destination / "root_cause_analysis.json", cause)
    _write_json(destination / "stage9_4_readiness.json", readiness)
    plan = _candidate_plan(cause, bundle, destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "stage9_4_candidate_plan.md").write_text(plan, encoding="utf-8")
    branch_manifest = _load_json_if(branch / "branch_rollout_manifest.json", {"status": "NOT_RUN"})
    projection_results = _load_json_if(projection / "projection_solver_results.json", {})
    counter_states = _load_json_if(counter / "state_counterfactual_decomposition.json", {})
    objective_rows = _load_json_if(objective / "objective_directional_attribution.json", {})
    objective_endpoints = _load_json_if(objective / "objective_endpoint_decomposition.json", {})
    objective_variable = _load_json_if(objective / "objective_variable_group_attribution.json", {})
    pressure_rows = _load_json_if(constraint / "constraint_attribution.json", {})
    path = _load_json_if(projection / "warm_final_path_feasibility.json", {})
    motion = _load_json_if(projection / "feasibility_motion_fraction.json", {})
    path_frame_ids = [
        int(frame)
        for frame in _load_json_if(projection / "warm_final_path_feasibility.json", {})
        .get("per_frame", {})
        .keys()
    ]
    frame_ids = sorted(set(path_frame_ids))
    path_traces = _html_path_traces(projection / "warm_final_path_feasibility.csv")
    compact_states = [_html_state(row) for row in counter_states.get("states", [])]
    compact_projection = [_html_projection(row) for row in projection_results.get("results", [])]
    compact_pressure = [_html_pressure(row) for row in pressure_rows.get("rows", [])]
    all_rmse = [
        float(row.get("long_finger_rmse_m", 0.0))
        for row in compact_states
        if row.get("long_finger_rmse_m") is not None
    ] + [
        float(row.get("long_finger_rmse_m", 0.0))
        for rows in path_traces.values()
        for row in rows
        if row.get("long_finger_rmse_m") is not None
    ]
    all_pressure = [
        float(row.get("pressure_score", 0.0))
        for row in compact_pressure
        if row.get("pressure_score") is not None
    ]
    all_objective = [
        float(row.get("formal_total_sum", 0.0))
        for row in objective_endpoints.get("states", [])
        if row.get("formal_total_sum") is not None
    ]
    global_scale = {
        "rmse_m": max(max(all_rmse or [0.01]), 1e-6),
        "pressure_score": max(max(all_pressure or [1.0]), 1e-6),
        "objective": max(max(all_objective or [1.0]), 1e-6),
        "signed_distance_m": max(
            max(
                abs(float(row.get("min_sdf_m", 0.0)))
                for rows in path_traces.values()
                for row in rows
                if row.get("min_sdf_m") is not None
            )
            if path_traces
            else 0.01,
            1e-6,
        ),
    }
    html_payload = {
        "frames": frame_ids,
        "requested_frames": list(bundle["frames"]),
        "completed_frames": frame_ids,
        "path": path.get("per_frame", {}),
        "path_traces": path_traces,
        "motion": motion.get("rows", []),
        "states": compact_states,
        "projection_results": compact_projection,
        "objective_endpoints": [_html_state(row) for row in objective_endpoints.get("states", [])],
        "objective": objective_rows.get("rows", []),
        "objective_directional": objective_rows.get("rows", []),
        "variable_group": objective_variable.get("rows", []),
        "pressure": compact_pressure,
        "pressure_aggregates": pressure_rows.get("aggregates", []),
        "root_cause": cause,
        "readiness": readiness,
        "branch": branch_manifest,
        "global_scale": global_scale,
    }
    _write_html(destination / "stage9_3_5_causal_closure.html", html_payload)
    summary = {
        "schema_version": SCHEMA,
        "status": cause["primary"],
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "root_cause": cause,
        "readiness": readiness,
        "projection": projection_results,
        "counterfactual": {"state_count": len(counter_states.get("states", []))},
        "objective": {"row_count": len(objective_rows.get("rows", []))},
        "constraint": {"row_count": len(pressure_rows.get("rows", []))},
        "branch": branch_manifest,
        "immutability": immutability,
        "html": str(destination / "stage9_3_5_causal_closure.html"),
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    _write_json(destination / "stage9_3_5_summary.json", summary)
    (destination / "stage9_3_5_summary.md").write_text(
        f"# Stage 9.3.5 Projection and Causal Closure\n\n- Status: `{cause['primary']}`\n- Primary route: `{cause['primary']}`\n- ENTER_STAGE9_4: `NO`\n- HUMAN_DECISION_REQUIRED: `YES`\n- STOP_AFTER_STAGE9_3_5: `TRUE`\n- Current lineage: `{bundle['lineage_hash']}`\n",
        encoding="utf-8",
    )
    return summary


def run_branch(
    current_lineage_manifest: str | Path,
    current_baseline: str | Path,
    projection_root: str | Path,
    output_root: str | Path,
    *,
    candidate: str = "auto",
    max_frames: int = 10,
    resume: bool = False,
) -> dict[str, Any]:
    bundle = _input_bundle(current_lineage_manifest, current_baseline)
    projection = _resolve(projection_root, bundle["repo"])
    destination = _resolve(output_root, bundle["repo"])
    payload = _load_json_if(projection / "projection_solver_results.json", {})
    rows = [row for row in payload.get("results", []) if row.get("strict_projection_acceptance")]
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(str(row["profile"]), []).append(row)
    gated: list[str] = []
    for profile, values in by_profile.items():
        if len(values) < 2:
            continue
        # The full gate is intentionally conservative.  It is evaluated from
        # independently validated rows and never from solver status alone.
        raw_improvements = [row.get("long_finger_rmse_improvement_m") for row in values]
        if any(item is None for item in raw_improvements):
            continue
        improvements = [float(item) for item in raw_improvements if item is not None]
        if all(
            improvement >= max(0.001, 0.1 * float(row.get("long_finger_rmse_m", 0.0)))
            for improvement, row in zip(improvements, values, strict=True)
        ):
            gated.append(profile)
    if candidate != "auto" and candidate not in gated:
        gated = []
    if not gated:
        result = {
            "schema_version": SCHEMA,
            "branch_rollout_status": "NOT_REQUIRED_BY_GATE",
            "candidate": None,
            "frames": [],
            "solver_invocation_count": 0,
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
        }
        _write_json(destination / "branch_rollout_manifest.json", result)
        _write_csv(destination / "branch_rollout_results.csv", [])
        _write_json(destination / "branch_rollout_summary.json", result)
        return result
    selected = gated[:2]
    results = [
        {
            "candidate": profile,
            "status": "GATE_PASS_BUT_NOT_EXECUTED",
            "reason": "bounded branch executor is reserved for a separately approved Stage 9.4 run",
        }
        for profile in selected
    ]
    result = {
        "schema_version": SCHEMA,
        "branch_rollout_status": "NOT_REQUIRED_BY_GATE",
        "candidate": selected,
        "frames": [],
        "results": results,
        "solver_invocation_count": 0,
        "current_causal_lineage_hash": bundle["lineage_hash"],
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    _write_json(destination / "branch_rollout_manifest.json", result)
    _write_csv(destination / "branch_rollout_results.csv", results)
    _write_json(destination / "branch_rollout_summary.json", result)
    return result


__all__ = [
    "PROFILES",
    "SCHEMA",
    "Stage935Error",
    "run_attribution",
    "run_branch",
    "run_constraints",
    "run_counterfactuals",
    "run_projection",
    "run_scan",
    "run_status",
]
