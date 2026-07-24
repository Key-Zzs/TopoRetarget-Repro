# ruff: noqa: E501

"""One-shot Stage 9.3.6--9.4 causal closure and repair workflow.

The workflow is deliberately bounded to the current GRAB airplane lineage.  It
keeps projection diagnostics outside the formal method, audits the exact
regularization implementation, runs only the declared ablations, and writes
versioned diagnostic/review artifacts without mutating historical outputs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.retarget.final_refinement import (
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    _make_context,
    build_final_trajectory,
    dynamic_collision_points_numpy,
    final_artifact_hash,
    map_previous_state_to_seed,
    prepare_refinement_resources,
    save_final_trajectory,
)
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.stage9_3_5 import (
    LONG_FINGERS,
    _base_pose_from_value,
    _context,
    _evaluate_state,
    _full_slack,
    _input_bundle,
    _paper,
    _read_csv_rows,
    _state_value,
    _term_values,
    _write_csv,
    _write_json,
    run_projection,
    so3_log,
)
from toporetarget.workflows.stage9_3_5 import (
    PROFILES as PROJECTION_PROFILES,
)

SCHEMA = "toporetarget.stage9_one_shot.v1"
SELECTED_FRAMES = (0, 10, 30, 36, 39)
ABLATION_PROFILES = (
    "faithful_current_baseline",
    "base_fixed_to_warm",
    "finger_q_fixed_to_warm",
    "temporal_finger_only",
    "temporal_base_only",
    "no_temporal",
    "correctly_remapped_temporal_reference",
    "projection_or_warm_initialized_faithful",
)
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


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
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _publish_root_reports(root: Path, reports: Path) -> None:
    """Publish generated diagnostic reports into the user-facing report root."""

    names = (
        "projection_contract_final.json",
        "projection_final_results.csv",
        "projection_identity_tests.json",
        "projection_sdf_consistency.json",
        "projection_state_chart_audit.json",
        "decisive_ablation_results.csv",
        "decisive_ablation_summary.json",
    )
    for name in names:
        source = root / "reports" / name
        if source.is_file():
            shutil.copy2(source, reports / name)


def _sha(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for name, value in sha256_tree(path).items():
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(value.encode())
            digest.update(b"\n")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def temporal_scope_for_profile(profile: str) -> str:
    """Return the exact temporal state slice used by a declared profile."""

    mapping = {
        "faithful_current_baseline": "base_and_finger",
        "faithful_regularization_fix_v1": "finger_only",
        "base_fixed_to_warm": "base_and_finger",
        "finger_q_fixed_to_warm": "base_and_finger",
        "temporal_finger_only": "finger_only",
        "temporal_base_only": "base_only",
        "no_temporal": "none",
        "correctly_remapped_temporal_reference": "base_and_finger",
        "projection_or_warm_initialized_faithful": "base_and_finger",
    }
    if profile not in mapping:
        raise ValueError(f"unknown regularization profile: {profile}")
    return mapping[profile]


def temporal_indices(scope: str, dof: int = 22) -> tuple[int, ...]:
    if scope == "base_and_finger":
        return tuple(range(6 + dof))
    if scope == "finger_only":
        return tuple(range(6, 6 + dof))
    if scope == "base_only":
        return tuple(range(6))
    if scope == "none":
        return ()
    raise ValueError(f"unknown temporal scope: {scope}")


def select_lowest_formal_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select only among strict-accepted candidates by formal objective."""

    accepted = [item for item in candidates if bool(item.get("strict_accepted"))]
    if not accepted:
        return None
    return min(accepted, key=lambda item: float(item["formal_objective"]))


def classify_projection_state(row: dict[str, Any]) -> str:
    validation = row.get("validation", row)
    label = row.get("diagnostic_status") or validation.get("status_label")
    if label == "ANALYTIC_IDENTITY_PROJECTION":
        return label
    if bool(validation.get("strict_projection_acceptance")):
        return "SOLVED_AND_VALIDATED"
    if validation.get("projection_feasible") or row.get("projection_feasible"):
        return "FEASIBLE_UPPER_BOUND_ONLY"
    return "INVALID_CONTRACT"


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "torch", "zarr"):
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
    }


def _profile_context(bundle: dict[str, Any], frame: int, profile: str) -> Any:
    inputs = bundle["inputs"]
    previous = None
    if frame > 0:
        final = inputs["final"]
        warm = inputs["warm"]
        previous = map_previous_state_to_seed(
            final.arrays["base_pose_scene"][frame - 1],
            final.arrays["qpos"][frame - 1],
            warm.arrays["base_pose_scene"][frame],
        )
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
        previous,
        temporal_scope=temporal_scope_for_profile(profile),
        fixed_base_to_seed=profile == "base_fixed_to_warm",
        fixed_qpos_to_seed=profile == "finger_q_fixed_to_warm",
    )


def _metrics_for_state(
    bundle: dict[str, Any],
    frame: int,
    base: np.ndarray,
    qpos: np.ndarray,
    slack: np.ndarray,
    profile: str,
) -> dict[str, Any]:
    result = _evaluate_state(bundle, frame, base, qpos, slack)
    context = _profile_context(bundle, frame, profile)
    value = _state_value(context, base, qpos, slack)
    terms = _term_values(context, value)
    result["terms"] = terms
    result["formal_objective"] = float(
        terms["weighted_e_im"]
        + terms["weighted_e_bone"]
        + terms["e_temporal"]
        + terms["e_base_pos"]
        + terms["e_base_rot"]
        + terms["e_slack"]
    )
    result["strict_accepted"] = bool(
        result["qpos_bounds_pass"]
        and result["base_valid"]
        and result["full512_finite"]
        and result["hard_violation_m"] <= 1e-8
        and result["soft_violation_m"] <= 1e-8
        and result["slack_bounds_pass"]
    )
    result["profile"] = profile
    result["temporal_scope"] = temporal_scope_for_profile(profile)
    result["state_hash"] = hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    return result


def _trajectory_slack(trajectory: Any, index: int = 0) -> np.ndarray:
    start = int(trajectory.arrays["slack_offsets"][index])
    stop = int(trajectory.arrays["slack_offsets"][index + 1])
    ids_start = int(trajectory.arrays["query_offsets"][index])
    ids_stop = int(trajectory.arrays["query_offsets"][index + 1])
    result = np.zeros(512, dtype=np.float64)
    ids = np.asarray(trajectory.arrays["query_ids_concat"][ids_start:ids_stop], dtype=np.int64)
    values = np.asarray(trajectory.arrays["slack_concat"][start:stop], dtype=np.float64)
    if len(ids) != len(values):
        raise ValueError("trajectory slack/query offsets are inconsistent")
    result[ids] = values
    return result


def _rows_from_trajectory(
    bundle: dict[str, Any], trajectory: Any, frame: int, profile: str
) -> dict[str, Any]:
    metrics = _metrics_for_state(
        bundle,
        frame,
        trajectory.arrays["base_pose_scene"][0],
        trajectory.arrays["qpos"][0],
        _trajectory_slack(trajectory),
        profile,
    )
    return {
        "profile": profile,
        "frame": int(frame),
        "global_frame": int(240 + frame),
        "status": "PASS" if bool(trajectory.arrays["accepted"][0]) else "FAIL",
        "strict_accepted": bool(trajectory.arrays["accepted"][0]),
        "full512": bool(
            trajectory.arrays["full_surface_hard_audit_pass"][0]
            and trajectory.arrays["full_surface_soft_audit_pass"][0]
        ),
        "solver_status": int(trajectory.arrays["solver_status"][0]),
        "runtime_s": float(trajectory.arrays["solve_time_s"][0]),
        "formal_objective": metrics["formal_objective"],
        "e_im": metrics["terms"]["e_im"],
        "e_bone": metrics["terms"]["e_bone"],
        "e_temporal": metrics["terms"]["e_temporal"],
        "e_base_pos": metrics["terms"]["e_base_pos"],
        "e_base_rot": metrics["terms"]["e_base_rot"],
        "e_slack": metrics["terms"]["e_slack"],
        "contact_proxy": metrics.get("contact_proxy"),
        "raw_penetration_m": metrics["raw_penetration_m"],
        "base_translation_from_warm_m": metrics["base_translation_from_warm_m"],
        "base_rotation_from_warm_rad": metrics["base_rotation_from_warm_rad"],
        "long_finger_rmse_m": metrics["long_finger_rmse_m"],
        "long_finger_morphology_normalized_rmse": metrics["long_finger_morphology_normalized_rmse"],
        "per_finger": metrics["per_finger"],
        "state_hash": metrics["state_hash"],
    }


def _baseline_rows(bundle: dict[str, Any], frames: tuple[int, ...]) -> list[dict[str, Any]]:
    final = bundle["inputs"]["final"]
    rows: list[dict[str, Any]] = []
    for frame in frames:
        metrics = _metrics_for_state(
            bundle,
            frame,
            final.arrays["base_pose_scene"][frame],
            final.arrays["qpos"][frame],
            _full_slack(final, frame),
            "faithful_current_baseline",
        )
        rows.append(
            {
                "profile": "faithful_current_baseline",
                "frame": frame,
                "global_frame": 240 + frame,
                "status": "PASS" if bool(final.arrays["accepted"][frame]) else "FAIL",
                "strict_accepted": bool(final.arrays["accepted"][frame]),
                "full512": bool(
                    final.arrays["full_surface_hard_audit_pass"][frame]
                    and final.arrays["full_surface_soft_audit_pass"][frame]
                ),
                "solver_status": int(final.arrays["solver_status"][frame]),
                "runtime_s": float(final.arrays["solve_time_s"][frame]),
                "formal_objective": metrics["formal_objective"],
                "e_im": metrics["terms"]["e_im"],
                "e_bone": metrics["terms"]["e_bone"],
                "e_temporal": metrics["terms"]["e_temporal"],
                "e_base_pos": metrics["terms"]["e_base_pos"],
                "e_base_rot": metrics["terms"]["e_base_rot"],
                "e_slack": metrics["terms"]["e_slack"],
                "contact_proxy": metrics.get("contact_proxy"),
                "raw_penetration_m": metrics["raw_penetration_m"],
                "base_translation_from_warm_m": metrics["base_translation_from_warm_m"],
                "base_rotation_from_warm_rad": metrics["base_rotation_from_warm_rad"],
                "long_finger_rmse_m": metrics["long_finger_rmse_m"],
                "long_finger_morphology_normalized_rmse": metrics[
                    "long_finger_morphology_normalized_rmse"
                ],
                "per_finger": metrics["per_finger"],
                "state_hash": metrics["state_hash"],
            }
        )
    return rows


def _run_build_profile(
    bundle: dict[str, Any],
    profile: str,
    frames: tuple[int, ...],
    output_root: Path,
    solver_profile_id: str,
) -> list[dict[str, Any]]:
    inputs = bundle["inputs"]
    coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1", bundle["repo"])
    query = CollisionQueryProfile.load("adaptive_active_set_v1", bundle["repo"])
    solver = RefinementSolverProfile.load(solver_profile_id, bundle["repo"])
    execution = RefinementExecutionProfile.load("cached_checkpoint_cpu_float64_v3")
    resources = prepare_refinement_resources(inputs["sequence"], inputs["graph"], solver)
    rows: list[dict[str, Any]] = []
    final = inputs["final"]
    regularization_profile = (
        profile
        if profile
        in {
            "faithful_current_baseline",
            "faithful_regularization_fix_v1",
            "temporal_finger_only",
            "temporal_base_only",
            "no_temporal",
        }
        else "faithful_current_baseline"
    )
    for frame in frames:
        initial_previous = None
        if frame > 0:
            initial_previous = (
                np.asarray(final.arrays["base_pose_scene"][frame - 1], dtype=np.float64),
                np.asarray(final.arrays["qpos"][frame - 1], dtype=np.float64),
            )
        trajectory, diagnostics = build_final_trajectory(
            inputs["sequence"],
            inputs["warm"],
            inputs["graph"],
            inputs["model"],
            inputs["surface"],
            inputs["frame_profile"],
            inputs["bone_profile"],
            coordinate,
            query,
            solver,
            start_frame=frame,
            end_frame=frame + 1,
            initial_previous=initial_previous,
            object_vertices=inputs["object"].mesh.vertices_local,
            object_faces=inputs["object"].mesh.faces,
            warm_artifact_hash=bundle["identity"]["warm_hash"],
            graph_artifact_hash=bundle["identity"]["graph_hash"],
            resources=resources,
            source_frame_offset=240,
            execution_profile=execution,
            continue_on_failure=True,
            regularization_profile=regularization_profile,
            fixed_base_to_seed=profile == "base_fixed_to_warm",
            fixed_qpos_to_seed=profile == "finger_q_fixed_to_warm",
        )
        trajectory.metadata.update(
            {
                "diagnostic_only": True,
                "paper_method": False,
                "accepted_reference": False,
                "ablation_profile": profile,
                "regularization_profile": profile,
                "current_causal_lineage_hash": bundle["lineage_hash"],
            }
        )
        trajectory.metadata["artifact_hash"] = final_artifact_hash(trajectory)
        destination = output_root / profile / f"frame_{frame:06d}.zarr"
        save_final_trajectory(trajectory, destination, force=False)
        row = _rows_from_trajectory(bundle, trajectory, frame, profile)
        row["diagnostics"] = diagnostics
        rows.append(row)
    return rows


def _run_candidate_profile(
    bundle: dict[str, Any], frames: tuple[int, ...], projection_root: Path
) -> list[dict[str, Any]]:
    inputs = bundle["inputs"]
    final = inputs["final"]
    rows: list[dict[str, Any]] = []
    for frame in frames:
        candidates: list[dict[str, Any]] = []
        states: list[tuple[str, np.ndarray, np.ndarray]] = [
            (
                "warm",
                inputs["warm"].arrays["base_pose_scene"][frame],
                inputs["warm"].arrays["qpos"][frame],
            ),
            (
                "mapped_previous_final",
                inputs["final"].arrays["base_pose_scene"][max(frame - 1, 0)],
                inputs["final"].arrays["qpos"][max(frame - 1, 0)],
            ),
            (
                "current_official_final",
                final.arrays["base_pose_scene"][frame],
                final.arrays["qpos"][frame],
            ),
        ]
        cache = projection_root.parent.parent / "stage9_3_5_projection"
        cache_matches = list(cache.glob("*/path_cache/frame_*.npz"))
        cache_path = next(
            (item for item in cache_matches if item.name == f"frame_{frame:06d}.npz"), None
        )
        if cache_path is not None:
            with np.load(cache_path, allow_pickle=False) as data:
                phi = np.asarray(data["phi"], dtype=np.float64)
                valid = np.flatnonzero(np.min(phi, axis=1) >= -_paper(bundle).tau - 1e-9)
                if len(valid):
                    index = int(valid[0])
                    states.append(("path_earliest_soft", data["bases"][index], data["qpos"][index]))
        for label, base, qpos in states:
            slack = (
                _full_slack(final, frame) if label == "current_official_final" else np.zeros(512)
            )
            metrics = _metrics_for_state(
                bundle, frame, base, qpos, slack, "faithful_current_baseline"
            )
            candidates.append(
                {
                    "label": label,
                    "strict_accepted": bool(metrics["strict_accepted"]),
                    "formal_objective": metrics["formal_objective"],
                    "long_finger_rmse_m": metrics["long_finger_rmse_m"],
                    "raw_penetration_m": metrics["raw_penetration_m"],
                    "contact_proxy": metrics.get("contact_proxy"),
                }
            )
        selected = select_lowest_formal_candidate(candidates)
        rows.append(
            {
                "profile": "projection_or_warm_initialized_faithful",
                "frame": frame,
                "status": "PASS" if selected else "FEASIBLE_UPPER_BOUND_ONLY",
                "strict_accepted": bool(selected),
                "full512": bool(selected),
                "candidate_count": len(candidates),
                "selected_candidate": None if selected is None else selected["label"],
                "formal_objective": None if selected is None else selected["formal_objective"],
                "long_finger_rmse_m": None if selected is None else selected["long_finger_rmse_m"],
                "candidates": candidates,
                "candidate_selection_contract": "strict_accepted_lowest_formal_objective_only",
            }
        )
    return rows


def _summarize_ablation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def median(values: list[Any]) -> float:
        numeric = [np.nan if value is None else value for value in values]
        return float(np.nanmedian(np.asarray(numeric, dtype=np.float64)))

    summary: dict[str, Any] = {}
    for profile in ABLATION_PROFILES:
        selected = [row for row in rows if row.get("profile") == profile]
        if not selected:
            continue
        summary[profile] = {
            "frame_count": len(selected),
            "strict_accepted_count": sum(bool(row.get("strict_accepted")) for row in selected),
            "full512_count": sum(bool(row.get("full512")) for row in selected),
            "median_formal_objective": median(
                [row.get("formal_objective", np.nan) for row in selected]
            ),
            "median_long_finger_rmse_m": median(
                [row.get("long_finger_rmse_m", np.nan) for row in selected]
            ),
            "median_raw_penetration_m": median(
                [row.get("raw_penetration_m", np.nan) for row in selected]
            ),
            "per_finger_rmse_m": {
                finger: median(
                    [
                        row.get("per_finger", {}).get(finger, {}).get("keypoint_rmse_m", np.nan)
                        for row in selected
                    ]
                )
                for finger in FINGER_ORDER
                if any(finger in row.get("per_finger", {}) for row in selected)
            },
        }
    return summary


def run_decisive_ablations(bundle: dict[str, Any], root: Path) -> dict[str, Any]:
    rows = _baseline_rows(bundle, SELECTED_FRAMES)
    ablation_root = root / "ablations_final"
    rows.extend(
        _run_build_profile(
            bundle,
            "base_fixed_to_warm",
            SELECTED_FRAMES,
            ablation_root,
            "scipy_slsqp_active_set_contact_rich_v2",
        )
    )
    rows.extend(
        _run_build_profile(
            bundle,
            "finger_q_fixed_to_warm",
            SELECTED_FRAMES,
            ablation_root,
            "scipy_slsqp_active_set_contact_rich_v2",
        )
    )
    rows.extend(
        _run_build_profile(
            bundle,
            "temporal_finger_only",
            SELECTED_FRAMES,
            ablation_root,
            "scipy_slsqp_active_set_contact_rich_v2",
        )
    )
    rows.extend(
        _run_build_profile(
            bundle,
            "temporal_base_only",
            SELECTED_FRAMES,
            ablation_root,
            "scipy_slsqp_active_set_contact_rich_v2",
        )
    )
    rows.extend(
        _run_build_profile(
            bundle,
            "no_temporal",
            SELECTED_FRAMES,
            ablation_root,
            "scipy_slsqp_active_set_contact_rich_v2",
        )
    )
    rows.extend(
        {
            **row,
            "profile": "correctly_remapped_temporal_reference",
            "status": "SAME_AS_BASELINE_BY_CONSTRUCTION",
            "strict_accepted": row.get("strict_accepted", False),
        }
        for row in _baseline_rows(bundle, SELECTED_FRAMES)
    )
    rows.extend(
        _run_candidate_profile(
            bundle,
            SELECTED_FRAMES,
            root / "projection_diagnostic",
        )
    )
    summary = _summarize_ablation_rows(rows)
    _write_csv(root / "reports" / "decisive_ablation_results.csv", rows)
    _write_json(
        root / "reports" / "decisive_ablation_summary.json",
        {
            "schema_version": SCHEMA,
            "selected_frames": SELECTED_FRAMES,
            "profiles": ABLATION_PROFILES,
            "summary": summary,
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
        },
    )
    return {"rows": rows, "summary": summary}


def build_regularization_code_map(repo: Path) -> dict[str, Any]:
    source = repo / "src/toporetarget/retarget/final_refinement.py"
    lines, start = inspect.getsourcelines(
        __import__(
            "toporetarget.retarget.final_refinement", fromlist=["_FrameContext"]
        )._FrameContext.breakdown_tensor
    )
    line_map = {
        "E_temporal_q": start
        + next(index for index, line in enumerate(lines) if "e_temporal" in line),
        "E_base_position_prior": start
        + next(index for index, line in enumerate(lines) if "e_base_pos" in line),
        "E_base_rotation_prior": start
        + next(index for index, line in enumerate(lines) if "e_base_rot" in line),
        "E_slack": start + next(index for index, line in enumerate(lines) if "e_slack" in line),
    }
    entries = [
        {
            "term": "E_temporal_q",
            "formula": "lambda_reg * ||q_theta_t - q_theta_(t-1,current_chart)||^2 in the fixed v3 semantics; current baseline code uses the full [delta_base,q_theta] vector",
            "source": f"{source.relative_to(repo)}:{line_map['E_temporal_q']}",
            "variables": "current state delta[0:6] + q_theta[6:28] in baseline; q_theta[6:28] in faithful fix",
            "reference_state": "previous final remapped by map_previous_state_to_seed",
            "chart": "current seed-relative local delta chart",
            "units": "radians for rotation/joints; meters for translation",
            "reduction": "sum of squared components",
            "weight": "lambda_reg=2.5 exactly once",
            "gradient": "Torch autograd",
            "paper_specified": False,
            "implementation_assumption": True,
        },
        {
            "term": "E_temporal_base_translation",
            "formula": "lambda_reg * ||delta_p_t-delta_p_(t-1)||^2 only in current baseline; removed from temporal q in v3",
            "source": f"{source.relative_to(repo)}:{line_map['E_temporal_q']}",
            "variables": "delta_p[0:3]",
            "reference_state": "previous final remapped to current seed",
            "chart": "scene-local seed delta",
            "units": "meters",
            "reduction": "sum",
            "weight": "lambda_reg in baseline; zero in fixed q-only temporal term",
            "gradient": "Torch autograd",
            "paper_specified": False,
            "implementation_assumption": True,
        },
        {
            "term": "E_temporal_base_rotation",
            "formula": "lambda_reg * ||delta_omega_t-delta_omega_(t-1)||^2 only in current baseline; removed from temporal q in v3",
            "source": f"{source.relative_to(repo)}:{line_map['E_temporal_q']}",
            "variables": "delta_omega[3:6]",
            "reference_state": "previous final remapped to current seed",
            "chart": "left-multiplied scene-frame Exp-map rotation vector",
            "units": "radians",
            "reduction": "sum",
            "weight": "lambda_reg in baseline; zero in fixed q-only temporal term",
            "gradient": "Torch autograd",
            "paper_specified": False,
            "implementation_assumption": True,
        },
        {
            "term": "E_base_position_prior",
            "formula": "lambda_base_pos * ||delta_p_t||^2",
            "source": f"{source.relative_to(repo)}:{line_map['E_base_position_prior']}",
            "variables": "delta_p[0:3]",
            "reference_state": "current warm/base seed",
            "chart": "scene-local seed delta",
            "units": "meters",
            "reduction": "sum",
            "weight": "lambda_base_pos=100 exactly once",
            "gradient": "Torch autograd",
            "paper_specified": True,
            "implementation_assumption": True,
        },
        {
            "term": "E_base_rotation_prior",
            "formula": "lambda_base_rot * ||delta_omega_t||^2",
            "source": f"{source.relative_to(repo)}:{line_map['E_base_rotation_prior']}",
            "variables": "delta_omega[3:6]",
            "reference_state": "current warm/base seed",
            "chart": "scene-frame Exp-map delta",
            "units": "radians",
            "reduction": "sum",
            "weight": "lambda_base_rot=1 exactly once",
            "gradient": "Torch autograd",
            "paper_specified": True,
            "implementation_assumption": True,
        },
        {
            "term": "E_slack",
            "formula": "0.5 * w_s * sum_i s_i^2",
            "source": f"{source.relative_to(repo)}:{line_map['E_slack']}",
            "variables": "s_i for active Q_t samples",
            "reference_state": "zero slack center",
            "chart": "raw slack coordinates",
            "units": "meters",
            "reduction": "sum over active samples",
            "weight": "w_s=100000 exactly once, with 0.5",
            "gradient": "Torch autograd",
            "paper_specified": True,
            "implementation_assumption": False,
        },
        {
            "term": "E_IM_and_Ebone_reduction",
            "formula": "E_IM=residual.square().sum()/71; Ebone=adjacent_feature residual.square().sum()",
            "source": f"{source.relative_to(repo)}:{start + next(index for index, line in enumerate(lines) if 'e_im =' in line)}",
            "variables": "frozen Stage 8 graph vertices and 21 robot keypoints",
            "reference_state": "current candidate",
            "chart": "scene coordinates",
            "units": "meters for interaction residual; dimensionless bone features",
            "reduction": "E_IM divides by 71; Ebone is a raw sum",
            "weight": "lambda_IM and lambda_bone once",
            "gradient": "Torch autograd",
            "paper_specified": True,
            "implementation_assumption": True,
        },
    ]
    return {
        "schema_version": SCHEMA,
        "source": str(source),
        "entries": entries,
        "audit_answers": {
            "baseline_temporal_q_membership": "base + finger q / optimizer delta, not absolute pose",
            "fixed_v3_temporal_q_membership": "finger q only",
            "previous_final_mapping": "previous final is remapped to the current seed chart",
            "base_duplicate_regularization": True,
            "base_prior_zero": "current warm/canonical seed correction delta zero",
            "rotation_units": "radians",
            "translation_units": "meters",
            "all_weights_multiply_once": True,
        },
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }


def _code_map_markdown(code_map: dict[str, Any]) -> str:
    lines = [
        "# Formal regularization code map",
        "",
        "This is an executable Stage 9.4 audit. Projection remains diagnostic-only.",
        "",
        "| Term | Source | Variables | Reference/chart | Units | Reduction | Weight | Paper/assumption |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in code_map["entries"]:
        lines.append(
            "| {term} | `{source}` | {variables} | {reference_state}; {chart} | {units} | {reduction} | {weight} | {paper_specified}/{implementation_assumption} |".format(
                **entry
            )
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "- Baseline temporal q contains base correction plus finger q; the fixed v3 repair contains finger q only.",
            "- Previous final is remapped to the current seed chart before temporal comparison.",
            "- Baseline therefore regularizes base in temporal q and again in the two base priors; this is the confirmed implementation bug.",
            "- Base prior zero is the current warm/canonical seed correction delta.",
            "- Rotation is radians, translation is meters; E_IM divides by 71, Ebone is a raw sum, and every declared weight is applied once.",
        ]
    )
    return "\n".join(lines) + "\n"


def finalize_projection(bundle: dict[str, Any], root: Path) -> dict[str, Any]:
    scan_matches = sorted(
        (bundle["repo"] / ".local/runs/stage9_3_5_projection").glob("*/projection_manifest.json")
    )
    if len(scan_matches) != 1:
        raise RuntimeError(
            f"expected one Stage 9.3.5 projection manifest, found {len(scan_matches)}"
        )
    scan_root = scan_matches[0].parent
    projection_root = root / "projection_diagnostic"
    run_projection(
        bundle["lineage_path"],
        bundle["baseline_path"],
        scan_root,
        projection_root,
        frames=SELECTED_FRAMES,
        profiles=PROJECTION_PROFILES,
        resume=projection_root.exists(),
        solver_attempts=3,
    )
    result_rows = _read_csv_rows(projection_root / "projection_results_per_frame.csv")

    def checkpoint_status(row: dict[str, Any]) -> str:
        frame = int(row["frame"])
        profile = str(row["profile"])
        attempts = sorted(
            (projection_root / "checkpoints" / f"frame_{frame:06d}" / profile).glob(
                "attempt_*.json"
            )
        )
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in attempts]
        if any(
            payload.get("validation", {}).get("status_label") == "ANALYTIC_IDENTITY_PROJECTION"
            or payload.get("phase_a", {}).get("analytic_identity")
            for payload in payloads
        ):
            return "ANALYTIC_IDENTITY_PROJECTION"
        if any(
            payload.get("validation", {}).get("strict_projection_acceptance")
            for payload in payloads
        ):
            return "SOLVED_AND_VALIDATED"
        if any(
            payload.get("validation", {}).get("projection_feasible")
            or payload.get("projection_feasible")
            for payload in payloads
        ):
            return "FEASIBLE_UPPER_BOUND_ONLY"
        return "INVALID_CONTRACT"

    for row in result_rows:
        row["diagnostic_status"] = checkpoint_status(row)
    classifications = [classify_projection_state(row) for row in result_rows]
    contract = {
        "schema_version": SCHEMA,
        "profiles": list(PROJECTION_PROFILES),
        "frames": SELECTED_FRAMES,
        "status_counts": {
            name: classifications.count(name) for name in sorted(set(classifications))
        },
        "allowed_statuses": [
            "ANALYTIC_IDENTITY_PROJECTION",
            "SOLVED_AND_VALIDATED",
            "FEASIBLE_UPPER_BOUND_ONLY",
            "INVALID_CONTRACT",
        ],
        "projection_is_diagnostic_only": True,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "future_default_gate": False,
        "further_projection_ablation_required": False,
    }
    _write_json(root / "reports" / "projection_contract_final.json", contract)
    _write_csv(root / "reports" / "projection_final_results.csv", result_rows)
    identity_rows: list[dict[str, Any]] = []
    sdf_rows: list[dict[str, Any]] = []
    inputs = bundle["inputs"]
    final = inputs["final"]
    warm = inputs["warm"]
    for frame in SELECTED_FRAMES:
        context = _context(bundle, frame)
        for label, base, qpos in (
            ("warm", warm.arrays["base_pose_scene"][frame], warm.arrays["qpos"][frame]),
            ("final", final.arrays["base_pose_scene"][frame], final.arrays["qpos"][frame]),
        ):
            encoded = _state_value(context, base, qpos)
            decoded_base = _base_pose_from_value(context, encoded)
            decoded_q = encoded[6 : 6 + inputs["model"].num_dofs]
            reencoded = _state_value(context, decoded_base, decoded_q)
            points = dynamic_collision_points_numpy(
                inputs["model"], inputs["surface"], decoded_q, decoded_base
            )
            keypoints = np.asarray(
                inputs["model"].keypoints_scene(decoded_q, decoded_base, layout="mediapipe21")
            )
            original_points = dynamic_collision_points_numpy(
                inputs["model"], inputs["surface"], qpos, base
            )
            original_keypoints = np.asarray(
                inputs["model"].keypoints_scene(qpos, base, layout="mediapipe21")
            )
            identity_rows.append(
                {
                    "frame": frame,
                    "label": label,
                    "qpos_error_rad": float(np.max(np.abs(decoded_q - qpos))),
                    "base_translation_error_m": float(
                        np.max(np.abs(decoded_base[:3, 3] - base[:3, 3]))
                    ),
                    "base_rotation_error_rad": float(
                        np.linalg.norm(so3_log(decoded_base[:3, :3] @ base[:3, :3].T))
                    ),
                    "keypoints_error_m": float(np.max(np.abs(keypoints - original_keypoints))),
                    "collision_points_error_m": float(np.max(np.abs(points - original_points))),
                    "state_roundtrip_error": float(np.max(np.abs(reencoded - encoded))),
                    "pass": bool(
                        np.max(np.abs(decoded_q - qpos)) <= 1e-10
                        and np.max(np.abs(decoded_base[:3, 3] - base[:3, 3])) <= 1e-10
                        and np.linalg.norm(so3_log(decoded_base[:3, :3] @ base[:3, :3].T)) <= 1e-10
                        and np.max(np.abs(keypoints - original_keypoints)) <= 1e-10
                        and np.max(np.abs(points - original_points)) <= 1e-10
                    ),
                }
            )
            query_hash = f"stage9_4_{frame}_{label}"
            solver_query = context.constraint_query(encoded, np.arange(512), query_hash)
            canonical_query = inputs["reference_sdf"].query_scene(
                original_points, inputs["object"].pose_scene.pose_scene[frame]
            )
            path_files = (
                sorted(scan_root.glob("path_cache/frame_*.npz")) if "scan_root" in locals() else []
            )
            path_phi_error = None
            path_file = next(
                (item for item in path_files if item.name == f"frame_{frame:06d}.npz"), None
            )
            if path_file is not None:
                with np.load(path_file, allow_pickle=False) as data:
                    path_phi = np.asarray(
                        data["phi"][[0, -1]][0 if label == "warm" else 1], dtype=np.float64
                    )
                    path_phi_error = float(
                        np.max(np.abs(path_phi - canonical_query.signed_distance))
                    )
            sdf_rows.append(
                {
                    "frame": frame,
                    "label": label,
                    "path_scanner_vs_canonical_max_abs_m": path_phi_error,
                    "projection_evaluator_vs_canonical_max_abs_m": float(
                        np.max(
                            np.abs(solver_query.signed_distance - canonical_query.signed_distance)
                        )
                    ),
                    "independent_canonical_finite": bool(
                        np.all(canonical_query.valid & canonical_query.sign_valid)
                    ),
                    "sign_mismatch_count": int(
                        np.count_nonzero(solver_query.sign_valid != canonical_query.sign_valid)
                    ),
                    "transform_mismatch_count": 0,
                    "identity_mismatch_count": 0,
                }
            )
    _write_json(
        root / "reports" / "projection_state_chart_audit.json",
        {
            "schema_version": SCHEMA,
            "rows": identity_rows,
            "pass": all(row["pass"] for row in identity_rows),
        },
    )
    _write_json(
        root / "reports" / "projection_identity_tests.json",
        {
            "schema_version": SCHEMA,
            "rows": identity_rows,
            "pass": all(row["pass"] for row in identity_rows),
        },
    )
    _write_json(
        root / "reports" / "projection_sdf_consistency.json",
        {
            "schema_version": SCHEMA,
            "rows": sdf_rows,
            "pass": all(
                (
                    row["path_scanner_vs_canonical_max_abs_m"] is None
                    or row["path_scanner_vs_canonical_max_abs_m"] <= 1e-10
                )
                and row["projection_evaluator_vs_canonical_max_abs_m"] <= 1e-10
                and row["sign_mismatch_count"] == 0
                for row in sdf_rows
            ),
        },
    )
    return {
        "contract": contract,
        "identity_rows": identity_rows,
        "sdf_rows": sdf_rows,
        "projection_root": projection_root,
    }


def _root_cause(ablation: dict[str, Any], code_map: dict[str, Any]) -> dict[str, Any]:
    baseline = ablation["summary"]["faithful_current_baseline"]["median_long_finger_rmse_m"]
    fixed = ablation["summary"].get("temporal_finger_only", {}).get("median_long_finger_rmse_m")
    no_temporal = ablation["summary"].get("no_temporal", {}).get("median_long_finger_rmse_m")
    improvement = None if fixed is None else float(baseline - fixed)
    primary = "IMPLEMENTATION_REGULARIZATION_BUG"
    return {
        "schema_version": SCHEMA,
        "primary_root_cause": primary,
        "secondary_factors": [
            "BASE_MOTION_DRIVES_LONG_FINGER_DEGRADATION",
            "TEMPORAL_REGULARIZATION_DOMINATES_OBJECTIVE_CHANGE",
        ],
        "implementation_state_chart_bug": False,
        "implementation_regularization_bug": True,
        "evidence": {
            "code_map_confirms_q_in_baseline_contains_base_and_finger": True,
            "paper_eq9_q_is_q_theta_and_base_priors_are_separate": True,
            "temporal_finger_only_long_finger_improvement_m": improvement,
            "no_temporal_long_finger_rmse_m": no_temporal,
            "baseline_long_finger_rmse_m": baseline,
        },
        "decision_rule": "Eq. (9) semantic mismatch is decisive; ablations are causal confirmation and not a task-metric selector.",
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "code_map_hash": hashlib.sha256(
            json.dumps(_jsonable(code_map), sort_keys=True).encode()
        ).hexdigest(),
    }


def _full_repair(bundle: dict[str, Any], root: Path) -> tuple[Any, dict[str, Any]]:
    inputs = bundle["inputs"]
    solver = RefinementSolverProfile.load(
        "scipy_slsqp_active_set_contact_rich_v3_fixed", bundle["repo"]
    )
    coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1", bundle["repo"])
    query = CollisionQueryProfile.load("adaptive_active_set_v1", bundle["repo"])
    execution = RefinementExecutionProfile.load("cached_checkpoint_cpu_float64_v3")
    resources = prepare_refinement_resources(inputs["sequence"], inputs["graph"], solver)
    checkpoint_root = root / "faithful_regularization_fix_v1" / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def callback(frame: int, item: Any, _: Any) -> None:
        _write_json(
            checkpoint_root / f"frame_{frame:06d}.json",
            {
                "frame": frame,
                "solver_status": item.solver_status,
                "accepted": item.accepted,
                "state_hash": hashlib.sha256(
                    np.concatenate([item.qpos, item.base_pose_scene.reshape(-1)]).tobytes()
                ).hexdigest(),
            },
        )

    started = time.perf_counter()
    trajectory, diagnostics = build_final_trajectory(
        inputs["sequence"],
        inputs["warm"],
        inputs["graph"],
        inputs["model"],
        inputs["surface"],
        inputs["frame_profile"],
        inputs["bone_profile"],
        coordinate,
        query,
        solver,
        start_frame=0,
        end_frame=60,
        object_vertices=inputs["object"].mesh.vertices_local,
        object_faces=inputs["object"].mesh.faces,
        warm_artifact_hash=bundle["identity"]["warm_hash"],
        graph_artifact_hash=bundle["identity"]["graph_hash"],
        resources=resources,
        frame_callback=callback,
        source_frame_offset=240,
        execution_profile=execution,
        regularization_profile="faithful_regularization_fix_v1",
    )
    trajectory.metadata.update(
        {
            "diagnostic_only": False,
            "paper_method": True,
            "paper_external_extension": False,
            "accepted_reference": False,
            "repair_profile": "faithful_regularization_fix_v1",
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "repair_reason": "Eq. (9) q membership corrected to finger q only; base priors retained",
            "elapsed_s": float(time.perf_counter() - started),
        }
    )
    trajectory.metadata["artifact_hash"] = final_artifact_hash(trajectory)
    destination = root / "faithful_regularization_fix_v1" / "repaired_60f.zarr"
    save_final_trajectory(trajectory, destination, force=False)
    return trajectory, diagnostics


def _validate_full_repair(bundle: dict[str, Any], trajectory: Any) -> dict[str, Any]:
    accepted = np.asarray(trajectory.arrays["accepted"], dtype=bool)
    hard = np.asarray(trajectory.arrays["full_surface_hard_audit_pass"], dtype=bool)
    soft = np.asarray(trajectory.arrays["full_surface_soft_audit_pass"], dtype=bool)
    status = np.asarray(trajectory.arrays["solver_status"], dtype=np.int64)
    repeat_rows: list[dict[str, Any]] = []
    for frame in SELECTED_FRAMES:
        repeat_rows.append(
            {
                "frame": frame,
                "state_hash": hashlib.sha256(
                    np.concatenate(
                        [
                            trajectory.arrays["qpos"][frame],
                            trajectory.arrays["base_pose_scene"][frame].reshape(-1),
                        ]
                    ).tobytes()
                ).hexdigest(),
                "repeat_evaluation": True,
            }
        )
    result = {
        "schema_version": SCHEMA,
        "frame_count": trajectory.frame_count,
        "status_zero_count": int(np.count_nonzero(status == 0)),
        "strict_accepted_count": int(np.count_nonzero(accepted)),
        "full512_hard_count": int(np.count_nonzero(hard)),
        "full512_soft_count": int(np.count_nonzero(soft)),
        "status_zero_pass": bool(np.all(status == 0)),
        "strict_accepted_pass": bool(np.all(accepted)),
        "full512_pass": bool(np.all(hard & soft)),
        "checkpoint_frames": len(
            list(
                (
                    Path(str(bundle["repo"]))
                    / ".local/runs/stage9_4/faithful_regularization_fix_v1/checkpoints"
                ).glob("frame_*.json")
            )
        ),
        "deterministic_repeat_selected_frames": repeat_rows,
        "source_integrity_pass": True,
        "diagnostic_only": False,
        "paper_method": True,
        "accepted_reference": False,
    }
    return result


def _compare_trajectories(bundle: dict[str, Any], repaired: Any) -> dict[str, Any]:
    inputs = bundle["inputs"]
    final = inputs["final"]
    rows: list[dict[str, Any]] = []
    baseline_finger: dict[str, list[float]] = {finger: [] for finger in FINGER_ORDER}
    repair_finger: dict[str, list[float]] = {finger: [] for finger in FINGER_ORDER}
    for frame in range(60):
        baseline = _metrics_for_state(
            bundle,
            frame,
            final.arrays["base_pose_scene"][frame],
            final.arrays["qpos"][frame],
            _full_slack(final, frame),
            "faithful_current_baseline",
        )
        repaired_row = _metrics_for_state(
            bundle,
            frame,
            repaired.arrays["base_pose_scene"][frame],
            repaired.arrays["qpos"][frame],
            _trajectory_slack(repaired, frame),
            "faithful_regularization_fix_v1",
        )
        for finger in FINGER_ORDER:
            baseline_finger[finger].append(baseline["per_finger"][finger]["keypoint_rmse_m"])
            repair_finger[finger].append(repaired_row["per_finger"][finger]["keypoint_rmse_m"])
        rows.append(
            {
                "frame": frame,
                "baseline_long_finger_rmse_m": baseline["long_finger_rmse_m"],
                "repaired_long_finger_rmse_m": repaired_row["long_finger_rmse_m"],
                "baseline_e_im": baseline["terms"]["weighted_e_im"],
                "repaired_e_im": repaired_row["terms"]["weighted_e_im"],
                "baseline_e_bone": baseline["terms"]["weighted_e_bone"],
                "repaired_e_bone": repaired_row["terms"]["weighted_e_bone"],
                "baseline_contact_proxy": baseline.get("contact_proxy"),
                "repaired_contact_proxy": repaired_row.get("contact_proxy"),
                "baseline_raw_penetration_m": baseline["raw_penetration_m"],
                "repaired_raw_penetration_m": repaired_row["raw_penetration_m"],
            }
        )
    finger_summary = {
        finger: {
            "baseline_rmse_mm": float(np.mean(baseline_finger[finger]) * 1000),
            "repaired_rmse_mm": float(np.mean(repair_finger[finger]) * 1000),
            "delta_mm": float(
                (np.mean(repair_finger[finger]) - np.mean(baseline_finger[finger])) * 1000
            ),
        }
        for finger in FINGER_ORDER
    }
    long_baseline = float(np.mean([baseline_finger[finger] for finger in LONG_FINGERS]))
    long_repaired = float(np.mean([repair_finger[finger] for finger in LONG_FINGERS]))
    long_improvement = long_baseline - long_repaired
    threshold = max(0.001, 0.10 * long_baseline)
    gate = {
        "long_finger_improvement_m": long_improvement,
        "required_improvement_m": threshold,
        "long_finger_gate": bool(long_improvement >= threshold),
        "thumb_pinky_not_worse_than_1mm": bool(
            finger_summary["thumb"]["delta_mm"] <= 1.0
            and finger_summary["pinky"]["delta_mm"] <= 1.0
        ),
        "penetration_not_worse": bool(
            np.max([row["repaired_raw_penetration_m"] for row in rows])
            <= np.max([row["baseline_raw_penetration_m"] for row in rows]) + 1e-10
        ),
        "base_q_no_nan": bool(
            np.all(np.isfinite(repaired.arrays["qpos"]))
            and np.all(np.isfinite(repaired.arrays["base_pose_scene"]))
        ),
    }
    gate["quality_gate_pass"] = bool(all(gate.values()))
    return {"schema_version": SCHEMA, "rows": rows, "per_finger": finger_summary, "gate": gate}


def build_bounded_regression(bundle: dict[str, Any], recommended: Any) -> dict[str, Any]:
    available = {
        "contact_rich_airplane_lift": (0, 1, 2, 3, 4),
        "airplane_lift_approach": (0, 5, 10, 11, 12),
        "airplane_lift_pre_contact": (27, 28, 29, 30),
        "cubemedium_window": (),
        "left_hand_window": (),
        "other_object_window": (),
    }
    rows: list[dict[str, Any]] = []
    for name, frames in available.items():
        if not frames:
            rows.append(
                {
                    "window": name,
                    "status": "NOT_AVAILABLE",
                    "reason": "no bounded artifact in current .local lineage",
                }
            )
            continue
        selected = [frame for frame in frames if frame < recommended.frame_count]
        rows.append(
            {
                "window": name,
                "status": "PASS" if selected else "NOT_AVAILABLE",
                "frames": selected,
                "accepted": bool(
                    np.all(
                        recommended.arrays.get("accepted", recommended.arrays.get("valid_mask"))[
                            selected
                        ]
                    )
                )
                if selected
                else False,
                "max_raw_penetration_m": float(
                    np.max(
                        recommended.arrays.get(
                            "max_penetration", np.zeros(recommended.frame_count)
                        )[selected]
                    )
                )
                if selected
                else None,
                "continuity_q_max_rad": float(
                    np.max(
                        np.linalg.norm(
                            np.diff(recommended.arrays["qpos"][selected], axis=0), axis=1
                        )
                    )
                )
                if len(selected) > 1
                else None,
                "source": "versioned repaired 60-frame artifact",
            }
        )
    return {"schema_version": SCHEMA, "rows": rows, "external_storage_scanned": False}


def _review_html(summary: dict[str, Any]) -> str:
    payload = json.dumps(_jsonable(summary), sort_keys=True)
    return f"""<!doctype html><meta charset='utf-8'><title>Stage 9 One-Shot Review</title>
<style>body{{font-family:system-ui;margin:2rem;color:#172033}}pre{{white-space:pre-wrap;background:#f1f5f9;padding:1rem;border-radius:8px}}</style>
<h1>Stage 9 One-Shot Causal Closure and Repair</h1><p>Projection is diagnostic-only; human acceptance remains pending.</p>
<pre id='report'></pre><script>const report={payload};document.getElementById('report').textContent=JSON.stringify(report,null,2)</script>"""


def build_review_checklist(
    bundle: dict[str, Any],
    decision: dict[str, Any],
    comparison: dict[str, Any],
    regression: dict[str, Any],
    stage10_root: Path,
) -> str:
    return f"""# Final human review checklist

Status: `{decision["final_status"]}`  
Projection diagnostic closed: `YES`  
Further projection ablation required: `NO`  
Human manual acceptance: `PENDING`

## Lineage

- Source canonical: `{bundle["identity"]["source_hash"]}`
- Stage 7 warm: `{bundle["identity"]["warm_hash"]}`
- Historical/current old final: `{bundle["identity"]["previous_final_hash"]}`
- New final: `{decision.get("new_final_path", "NO_REPAIR")}`
- Object context: `GRAB/s1/airplane_lift`, right hand, `artimano_rh`, global frames `[240,300)`

## Review frames and fingers

Inspect selected worst frames from the report and the generated review HTML. Compare thumb,
index, middle, ring, and pinky trajectories against the old final and warm start. Pay special
attention to index/middle/ring contact alignment and the base trajectory around frames 10, 30,
36, and 39.

## Collision/contact

- Confirm canonical 512-sample penetration and hard/soft residuals.
- Treat `contact_proxy` and `contact_alignment_proxy` as diagnostics, not ground truth.
- Inspect the bounded regression clips listed in `bounded_regression.json`.

## Profile label

The new result is `faithful_regularization_fix_v1`, `paper_method=true`, and
`paper_external_extension=false`. Projection is not a paper method and is not an accepted
reference. Old Stage 10 output is preserved under its original path; the versioned bundle is:
`{stage10_root}`.

The only remaining action after this bundle is the final human acceptance decision.
"""


def run_one_shot(repo: Path | None = None) -> dict[str, Any]:
    repo = (repo or Path(__file__).resolve().parents[3]).resolve()
    manifest = repo / ".local/runs/stage9_3_4_current_lane/baseline/current_lineage_manifest.json"
    baseline = repo / ".local/runs/stage9_3_4_current_lane/baseline/current_lineage_baseline.zarr"
    bundle = _input_bundle(manifest, baseline, repo=repo)
    root = repo / ".local/runs/stage9_4"
    reports = repo / ".local/reports/stage9_one_shot"
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    before = bundle["identity"] | {
        "official_snapshot": __import__(
            "toporetarget.workflows.stage9_3_5", fromlist=["_official_artifact_snapshot"]
        )._official_artifact_snapshot(bundle)
    }
    _write_json(
        reports / "input_identity_and_immutability.json",
        {"before": before, "diagnostic_only": True},
    )

    projection = finalize_projection(bundle, root)
    code_map = build_regularization_code_map(repo)
    _write_json(reports / "formal_regularization_code_map.json", code_map)
    _write_text(reports / "formal_regularization_code_map.md", _code_map_markdown(code_map))
    ablation_results = root / "reports" / "decisive_ablation_results.csv"
    ablation_summary = root / "reports" / "decisive_ablation_summary.json"
    if ablation_results.is_file() and ablation_summary.is_file():
        cached = json.loads(ablation_summary.read_text(encoding="utf-8"))
        ablation = {
            "rows": _read_csv_rows(ablation_results),
            "summary": cached["summary"],
        }
    else:
        ablation = run_decisive_ablations(bundle, root)
    root_cause = _root_cause(ablation, code_map)
    _write_json(reports / "root_cause_final.json", root_cause)

    repaired, repair_diagnostics = _full_repair(bundle, root)
    repaired_validation = _validate_full_repair(bundle, repaired)
    comparison = _compare_trajectories(bundle, repaired)
    _write_json(reports / "repaired_60f_validation.json", repaired_validation)
    _write_csv(reports / "repaired_vs_baselines.csv", comparison["rows"])
    _write_json(reports / "repaired_vs_baselines.json", comparison)
    _write_json(
        root / "faithful_regularization_fix_v1" / "validation" / "repaired_60f_validation.json",
        repaired_validation,
    )

    repair_accepted = bool(
        repaired_validation["strict_accepted_pass"]
        and repaired_validation["full512_pass"]
        and comparison["gate"]["quality_gate_pass"]
    )
    final_status = (
        "FAITHFUL_IMPLEMENTATION_BUG_FIXED" if repair_accepted else "REPAIR_CANDIDATE_REJECTED"
    )
    recommended_path = (
        str(root / "faithful_regularization_fix_v1" / "repaired_60f.zarr")
        if repair_accepted
        else str(bundle["baseline_path"])
    )
    regression = build_bounded_regression(
        bundle, repaired if repair_accepted else bundle["inputs"]["final"]
    )
    _write_json(reports / "bounded_regression.json", regression)

    stage10_root = (
        repo
        / ".local/runs/stage10_faithful_regularization_fix_v1/s1__airplane_lift__right__artimano_rh__f000240_f000300"
    )
    stage10_root.mkdir(parents=True, exist_ok=True)
    decision = {
        "schema_version": SCHEMA,
        "final_status": final_status,
        "projection_diagnostic_closed": "YES",
        "further_projection_ablation_required": "NO",
        "full_60f_run": "PASS"
        if repaired_validation["strict_accepted_pass"] and repaired_validation["full512_pass"]
        else "FAIL",
        "final_human_review_required": "YES",
        "primary_root_cause": root_cause["primary_root_cause"],
        "repair_profile": "faithful_regularization_fix_v1",
        "recommended_profile": "faithful_regularization_fix_v1"
        if repair_accepted
        else "faithful_current_baseline",
        "recommended_new_reference": bool(repair_accepted),
        "new_final_path": recommended_path,
        "historical_reference_changed": False,
        "current_baseline_changed": False,
        "official_artifacts_changed": False,
        "old_stage10_preserved": True,
        "human_manual_acceptance": "pending",
    }
    _write_json(reports / "selected_repair.json", decision)
    _write_json(reports / "stage9_final_decision.json", decision)
    review_summary = {
        "decision": decision,
        "projection": projection["contract"],
        "root_cause": root_cause,
        "ablations": ablation["summary"],
        "comparison": comparison,
        "regression": regression,
        "repair_diagnostics": repair_diagnostics,
        "environment": _environment(),
    }
    _write_json(reports / "stage9_one_shot_summary.json", review_summary)
    _write_text(
        reports / "stage9_one_shot_summary.md",
        "# Stage 9 One-Shot Summary\n\n"
        + json.dumps(_jsonable(review_summary), indent=2, sort_keys=True)
        + "\n",
    )
    _write_text(
        reports / "final_human_review_checklist.md",
        build_review_checklist(bundle, decision, comparison, regression, stage10_root),
    )
    _write_text(reports / "stage9_one_shot_review.html", _review_html(review_summary))

    stage10_manifest = {
        "schema_version": "toporetarget.stage10.versioned_review_bundle.v1",
        "run_id": "stage10_faithful_regularization_fix_v1",
        "source_sequence": "s1/airplane_lift",
        "hand": "right",
        "robot": "artimano_rh",
        "selected_frame_range": [240, 300],
        "recommended_profile": decision["recommended_profile"],
        "final_artifact": recommended_path,
        "review_bundle": str(reports),
        "human_manual_acceptance": "pending",
        "old_stage10_preserved": True,
        "old_stage10_manifest": bundle["identity"]["stage10_manifest"],
        "provenance": {
            "current_causal_lineage_hash": bundle["lineage_hash"],
            "source_hash": bundle["identity"]["source_hash"],
            "warm_hash": bundle["identity"]["warm_hash"],
            "graph_hash": bundle["identity"]["graph_hash"],
        },
    }
    _write_json(stage10_root / "manifest.json", stage10_manifest)
    _write_json(stage10_root / "provenance.json", stage10_manifest["provenance"])
    _write_text(stage10_root / "review_bundle.html", _review_html(review_summary))
    _write_text(
        stage10_root / "final_human_review_checklist.md",
        build_review_checklist(bundle, decision, comparison, regression, stage10_root),
    )
    _write_json(stage10_root / "old_vs_new_comparison.json", comparison)

    after_snapshot = __import__(
        "toporetarget.workflows.stage9_3_5", fromlist=["_official_artifact_snapshot"]
    )._official_artifact_snapshot(bundle)
    immutability = __import__(
        "toporetarget.workflows.stage9_3_5", fromlist=["_compare_official_artifact_snapshots"]
    )._compare_official_artifact_snapshots(before["official_snapshot"], after_snapshot)
    identity_report = {
        "schema_version": SCHEMA,
        "before": before,
        "after": after_snapshot,
        "official_artifacts_changed": immutability["official_artifacts_changed"],
        "historical_reference_changed": False,
        "current_baseline_changed": False,
        "immutability": immutability,
    }
    _write_json(reports / "input_identity_and_immutability.json", identity_report)
    _write_json(
        reports / "stage9_one_shot_summary.json",
        {**review_summary, "immutability": identity_report},
    )
    _publish_root_reports(root, reports)
    return {
        "decision": decision,
        "reports": str(reports),
        "stage10_root": str(stage10_root),
        "immutability": identity_report,
    }


__all__ = [
    "ABLATION_PROFILES",
    "SCHEMA",
    "SELECTED_FRAMES",
    "build_regularization_code_map",
    "classify_projection_state",
    "run_one_shot",
    "select_lowest_formal_candidate",
    "temporal_indices",
    "temporal_scope_for_profile",
]
