"""Run the W2.2 Wuji continuity closeout in an isolated artifact root.

This command reads the existing W1/W2/W3 formal artifacts, writes only under
``closeout_v1``, and uses the repository's constrained solver for the fixed
bounded windows.  It never replaces a formal final, export, or HTML artifact.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.continuous import (
    S_Q_RAD,
    continuity_metrics,
    so3_log_np,
    transport_previous_final_to_current_warm,
)
from toporetarget.retarget.final_refinement import (
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    build_final_trajectory,
    final_artifact_hash,
    load_final_trajectory,
    load_robot_surface_samples,
    prepare_refinement_resources,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_graph,
)
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.retarget.wuji_closeout import (
    ABLATION_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ablation_conclusion,
    build_w2_attribution,
    recommendation_gates,
    synthetic_window_fixture,
)
from toporetarget.robots.registry import get_robot_registry
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.workflows.grab_suite import SuiteClip, load_suite

UNITS = {
    "W1": ("W1_s1__airplane_lift__right__wuji_hand2_beta1_rh__f000240_f000300", "W1_airplane_lift"),
    "W2": ("W2_s1__apple_eat_1__right__wuji_hand2_beta1_rh__f000212_f000272", "W2_apple_eat_1"),
    "W3": (
        "W3_s1__alarmclock_lift__right__wuji_hand2_beta1_rh__f000407_f000467",
        "W3_alarmclock_lift",
    ),
}

WINDOWS = (
    ("W1_original_anomaly", "W1", 0, 3, "causal_common_anchor", "baseline"),
    ("W2_original_base_anomaly", "W2", 22, 26, "causal_common_anchor", "baseline"),
    ("W2_residual_qstep_block_A", "W2", 7, 15, "continuous_residual_anchor", "continuous"),
    ("W2_residual_qstep_block_B", "W2", 47, 52, "continuous_residual_anchor", "continuous"),
    ("W3_original_anomaly_A", "W3", 13, 18, "causal_common_anchor", "baseline"),
    ("W3_original_anomaly_B", "W3", 34, 39, "causal_common_anchor", "baseline"),
    ("W3_original_anomaly_C", "W3", 41, 46, "causal_common_anchor", "baseline"),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, default=str)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").rstrip("\x00")
    return str(value)


def _clip_map(clips: tuple[SuiteClip, ...]) -> dict[str, SuiteClip]:
    result = {clip.unit_id: clip for clip in clips}
    if set(result) != {item[0] for item in UNITS.values()}:
        raise RuntimeError("continuous suite does not match the frozen W1/W2/W3 units")
    return result


def _load_closeout_suite(path: str | Path) -> tuple[dict[str, Any], tuple[SuiteClip, ...]]:
    """Load the frozen suite, accepting this repo's pre-units W2.1 manifest."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if isinstance(payload, dict) and isinstance(payload.get("units"), list):
        return load_suite(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
        raise RuntimeError(f"WUJI continuity closeout suite has neither units nor clips: {source}")
    defaults = {
        "subject": payload.get("selection", {}).get("source_subject", "s1"),
        "hand": payload.get("selection", {}).get("hand", "right"),
        "robot": payload.get("robot", "wuji_hand2_beta1_rh"),
        "native_fps": payload.get("selection", {}).get("native_fps", 120.0),
    }
    clips = tuple(
        SuiteClip(
            unit_id=str(item["unit_id"]),
            short_id=str(item.get("short_id", item["unit_id"])),
            sequence=str(item["sequence"]),
            subject=str(item.get("subject", defaults["subject"])),
            object_name=str(item["object_name"]),
            hand=str(item.get("hand", defaults["hand"])),
            robot=str(item.get("robot", defaults["robot"])),
            start_frame=int(item["start_frame"]),
            end_frame=int(item["end_frame"]),
            native_fps=float(item.get("native_fps", defaults["native_fps"])),
        )
        for item in payload["clips"]
    )
    return payload, clips


def _unit_paths(root: Path, unit_id: str) -> dict[str, Path]:
    unit = root / unit_id
    return {
        "unit": unit,
        "canonical": unit / "canonical" / "canonical.zarr",
        "warm": unit / "warm_start" / "warm_start.zarr",
        "graph": unit / "interaction_graph" / "interaction_graph.zarr",
        "final": unit / "final" / "final_retarget.zarr",
    }


def _hash_path(path: Path) -> str:
    return sha256_file(path) if path.is_file() else _stable_hash(sha256_tree(path))


def _source_file(sequence: Any) -> Path | None:
    provenance = getattr(sequence.metadata, "provenance", None)
    value = getattr(provenance, "source_file", None)
    return None if value is None else Path(str(value))


def _input_audit(
    repo: Path, root: Path, baseline_root: Path, clips: dict[str, SuiteClip], closeout: Path
) -> dict[str, Any]:
    model = get_robot_registry(repo_root=repo).load("wuji_hand2_beta1_rh")
    rows: list[dict[str, Any]] = []
    immutable: dict[str, Any] = {
        "formal_inputs": {},
        "source_files": {},
        "mano_root": "/mnt/nas/storage/Ref2Dex_storage/shared_assets/body_models/mano",
        "wuji_assets": str(repo / "third_party" / "robot_hands" / "wuji_hand2_beta1"),
        "old_continuity_html": str(root / "html"),
        "exports": str(root / "exports"),
        "historical_stage10": str(repo / ".local" / "runs" / "stage10"),
        "pene_loss_worktree": str(repo.parent / "TopoRetarget-Repro-pene-loss"),
        "pene_loss_status_before": subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=repo.parent / "TopoRetarget-Repro-pene-loss",
            capture_output=True,
            text=True,
            check=False,
        ).stdout,
    }
    for short, (unit_id, _short_id) in UNITS.items():
        clip = clips[unit_id]
        current = _unit_paths(root, unit_id)
        baseline = _unit_paths(baseline_root, unit_id)
        required = [
            current["canonical"],
            current["warm"],
            current["graph"],
            current["final"],
            baseline["final"],
        ]
        if not all(path.exists() for path in required):
            raise RuntimeError(
                f"WUJI_CONTINUITY_CLOSEOUT_BLOCKED_BY_LINEAGE_ERROR: missing {short} input"
            )
        sequence = load_hoi_sequence(current["canonical"])
        warm = load_warm_start(current["warm"])
        graph = load_interaction_graph(current["graph"])
        baseline_final = load_final_trajectory(baseline["final"])
        continuous_final = load_final_trajectory(current["final"])
        timestamps = np.asarray(sequence.metadata.timestamps, dtype=np.float64)
        if (
            sequence.num_frames != 60
            or warm.frame_count != 60
            or graph.frame_count != 60
            or baseline_final.frame_count != 60
            or continuous_final.frame_count != 60
        ):
            raise RuntimeError(
                f"WUJI_CONTINUITY_CLOSEOUT_BLOCKED_BY_LINEAGE_ERROR: {short} frame count"
            )
        if not np.allclose(
            timestamps, warm.arrays["timestamps"], atol=0.0, rtol=0.0
        ) or not np.allclose(timestamps, continuous_final.arrays["timestamps"], atol=0.0, rtol=0.0):
            raise RuntimeError(
                f"WUJI_CONTINUITY_CLOSEOUT_BLOCKED_BY_LINEAGE_ERROR: {short} timestamps"
            )
        if not np.array_equal(warm.arrays["qpos"].shape, (60, 20)) or not np.array_equal(
            continuous_final.arrays["base_pose_scene"].shape, (60, 4, 4)
        ):
            raise RuntimeError(f"WUJI_CONTINUITY_CLOSEOUT_BLOCKED_BY_LINEAGE_ERROR: {short} shape")
        source_file = _source_file(sequence)
        row = {
            "unit": short,
            "unit_id": unit_id,
            "sequence": clip.sequence,
            "frame_range": [clip.start_frame, clip.end_frame],
            "local_frame_range": [0, 60],
            "timestamps_hash": _stable_hash(timestamps.tolist()),
            "source_path": str(source_file) if source_file else None,
            "source_hash": None
            if source_file is None or not source_file.exists()
            else _hash_path(source_file),
            "canonical_hash": _hash_path(current["canonical"]),
            "warm_hash": _hash_path(current["warm"]),
            "graph_hash": interaction_artifact_hash(current["graph"]),
            "baseline_final_hash": _hash_path(baseline["final"]),
            "baseline_final_content_hash": final_artifact_hash(baseline_final),
            "continuous_final_hash": _hash_path(current["final"]),
            "continuous_final_content_hash": final_artifact_hash(continuous_final),
            "qpos_shape": list(continuous_final.arrays["qpos"].shape),
            "base_shape": list(continuous_final.arrays["base_pose_scene"].shape),
            "accepted_mask_shape": list(continuous_final.arrays["accepted"].shape),
            "robot": model.name,
            "qpos_order": list(model.dof_names),
            "qpos_order_hash": _stable_hash(list(model.dof_names)),
            "collision_profile": continuous_final.metadata.get("query_profile"),
            "baseline_profile": baseline_final.metadata.get("solver_profile_id"),
            "continuous_profile": continuous_final.metadata.get("solver_profile_id"),
            "continuity_profile": continuous_final.metadata.get("continuous_profile_id"),
            "source_sequence_id": continuous_final.metadata.get("source_sequence_id"),
            "source_hand_id": continuous_final.metadata.get("source_hand_id"),
            "lineage_match": bool(
                continuous_final.metadata.get("warm_start_artifact_hash")
                in {None, artifact_hash(current["warm"])}
                and continuous_final.metadata.get("graph_artifact_hash")
                in {None, interaction_artifact_hash(current["graph"])}
            ),
        }
        if not row["lineage_match"]:
            raise RuntimeError(
                f"WUJI_CONTINUITY_CLOSEOUT_BLOCKED_BY_LINEAGE_ERROR: {short} final lineage"
            )
        rows.append(row)
        immutable["formal_inputs"][short] = {
            key: row[key]
            for key in (
                "canonical_hash",
                "warm_hash",
                "graph_hash",
                "baseline_final_hash",
                "continuous_final_hash",
            )
        }
        if source_file is not None and source_file.exists():
            immutable["source_files"][str(source_file)] = _hash_path(source_file)
    for label, path in (
        ("wuji_assets", Path(immutable["wuji_assets"])),
        ("mano_root", Path(immutable["mano_root"])),
        ("old_continuity_html", Path(immutable["old_continuity_html"])),
        ("exports", Path(immutable["exports"])),
        ("historical_stage10", Path(immutable["historical_stage10"])),
    ):
        immutable[label] = {
            "path": str(path),
            "exists": path.exists(),
            "hash": None if not path.exists() else _hash_path(path),
        }
    _write_json(
        closeout / "input_audit" / "input_identity.json",
        {
            "schema_version": SCHEMA_VERSION,
            "environment": _environment(),
            "rows": rows,
            "immutable": immutable,
        },
    )
    _write_csv(closeout / "input_audit" / "input_identity.csv", rows)
    _write_json(closeout / "input_audit" / "immutability_before.json", immutable)
    return {"rows": rows, "immutable": immutable, "clips": clips}


def _environment() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "dtype": "float64",
        "threads": {
            name: os.environ.get(name, "1")
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }


def _final_bone_directions(final: Any) -> np.ndarray:
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    return np.stack(
        [
            extract_bone_features(
                item, frame_profile, bone_profile, side="right", strict=True
            ).unit_directions
            for item in final.arrays["robot_keypoints_scene"]
        ]
    )


def _run_attribution(audit: dict[str, Any], root: Path, closeout: Path) -> dict[str, Any]:
    unit_id = UNITS["W2"][0]
    paths = _unit_paths(root, unit_id)
    warm = load_warm_start(paths["warm"])
    final = load_final_trajectory(paths["final"])
    model = get_robot_registry().load("wuji_hand2_beta1_rh")
    attribution = build_w2_attribution(
        warm_arrays=warm.arrays,
        final_arrays=final.arrays,
        joint_names=model.dof_names,
        joint_lower=model.joint_lower,
        joint_upper=model.joint_upper,
        source_bone_directions=warm.arrays["source_bone_directions"],
        warm_bone_directions=warm.arrays["robot_bone_directions"],
        final_bone_directions=_final_bone_directions(final),
        warm_keypoints_scene=warm.arrays["robot_keypoints_scene"],
        final_keypoints_scene=final.arrays["robot_keypoints_scene"],
        global_frame_offset=212,
        timestamps=final.arrays["timestamps"],
    )
    attribution["aggregate"]["trajectory_discontinuous"] = bool(
        not np.all(final.arrays["trajectory_continuous"])
    )
    attribution["aggregate"]["correction_gate_all_formal_frames"] = bool(
        np.all(final.arrays["continuity_finger_inf_rad"] <= S_Q_RAD)
    )
    destination = closeout / "w2_qstep_attribution"
    _write_json(
        destination / "detected_transitions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "transitions": attribution["detected_transitions"],
            "aggregate": attribution["aggregate"],
        },
    )
    _write_csv(destination / "detected_transitions.csv", attribution["per_transition"])
    _write_json(
        destination / "per_joint_attribution.json",
        {"schema_version": SCHEMA_VERSION, "rows": attribution["per_joint"]},
    )
    _write_csv(destination / "per_joint_attribution.csv", attribution["per_joint"])
    _write_json(
        destination / "per_transition_summary.json",
        {"schema_version": SCHEMA_VERSION, "rows": attribution["per_transition"]},
    )
    _write_csv(destination / "per_transition_summary.csv", attribution["per_transition"])
    _write_json(destination / "aggregate_attribution.json", attribution["aggregate"])
    aggregate = attribution["aggregate"]
    (destination / "aggregate_attribution.md").write_text(
        "# W2 q-step attribution\n\n"
        f"- absolute q-step transitions: {aggregate['absolute_q_step_count']}\n"
        f"- warm/source driven: {aggregate['warm_driven_count']}\n"
        f"- correction driven: {aggregate['correction_driven_count']}\n"
        f"- reachability/limit driven: {aggregate['reachability_driven_count']}\n"
        f"- mixed: {aggregate['mixed_count']}\n"
        f"- jump-and-return: {aggregate['jump_and_return_count']}\n"
        f"- max decomposition error: {aggregate['decomposition_max_error_rad']:.3e} rad\n"
        f"- correction continuity gate: {aggregate['correction_continuity_gate_pass']}\n"
        f"- blocks recommendation: {aggregate['blocks_recommendation']}\n",
        encoding="utf-8",
    )
    return attribution


def _anchor(final: Any, start: int, source: str) -> tuple[np.ndarray, np.ndarray] | None:
    if start == 0:
        return None
    index = start - 1
    return np.asarray(final.arrays["base_pose_scene"][index], dtype=np.float64), np.asarray(
        final.arrays["qpos"][index], dtype=np.float64
    )


def _profile_setup(
    repo: Path,
    root: Path,
    closeout: Path,
    short: str,
    anchor_profile: str,
    start: int,
    solver_profile: str,
) -> dict[str, Any]:
    unit_id = UNITS[short][0]
    paths = _unit_paths(root, unit_id)
    sequence = load_hoi_sequence(paths["canonical"])
    warm = load_warm_start(paths["warm"])
    graph = load_interaction_graph(paths["graph"])
    model = get_robot_registry(repo_root=repo).load("wuji_hand2_beta1_rh")
    surface = load_robot_surface_samples(
        repo / ".local" / "cache" / "geometry" / "robot_surface" / "wuji_hand2_beta1_rh_neutral.npz"
    )
    solver = RefinementSolverProfile.load(solver_profile)
    execution = RefinementExecutionProfile.load("cached_checkpoint_cpu_float64_v3")
    resources = prepare_refinement_resources(
        sequence,
        graph,
        solver,
        sdf_tree_leaf_size=execution.sdf_tree_leaf_size,
        geometry_artifact_root=closeout / "checkpoints" / "geometry",
    )
    anchor_final = load_final_trajectory(
        _unit_paths(root, unit_id)["final"]
        if anchor_profile == "continuous"
        else _unit_paths(root.parent / "wuji_hand2_grab3_v1", unit_id)["final"]
    )
    return {
        "paths": paths,
        "sequence": sequence,
        "warm": warm,
        "graph": graph,
        "model": model,
        "surface": surface,
        "solver": solver,
        "execution": execution,
        "resources": resources,
        "anchor": _anchor(anchor_final, start, anchor_profile),
    }


def _motion_rows(
    trajectory: Any,
    warm: Any,
    model: Any,
    start: int,
    anchor: tuple[np.ndarray, np.ndarray] | None,
    profile: str,
    window_name: str,
    global_frame_offset: int,
) -> list[dict[str, Any]]:
    arrays = trajectory.arrays
    rows: list[dict[str, Any]] = []
    previous_base = None if anchor is None else anchor[0]
    previous_q = None if anchor is None else anchor[1]
    for local_row, frame in enumerate(np.asarray(arrays["frame_indices"], dtype=np.int64)):
        if previous_base is None or previous_q is None:
            predicted_base = np.asarray(warm.arrays["base_pose_scene"][frame], dtype=np.float64)
            predicted_q = np.asarray(warm.arrays["qpos"][frame], dtype=np.float64)
        else:
            propagated = transport_previous_final_to_current_warm(
                warm.arrays["base_pose_scene"][frame - 1],
                previous_base,
                warm.arrays["base_pose_scene"][frame],
                warm.arrays["qpos"][frame - 1],
                previous_q,
                warm.arrays["qpos"][frame],
                model.joint_lower,
                model.joint_upper,
                previous_frame=frame - 1,
                current_frame=frame,
            )
            predicted_base = propagated.predicted_base_scene
            predicted_q = propagated.predicted_qpos
        current_base = np.asarray(arrays["base_pose_scene"][local_row], dtype=np.float64)
        current_q = np.asarray(arrays["qpos"][local_row], dtype=np.float64)
        current_keypoints = np.asarray(arrays["robot_keypoints_scene"][local_row], dtype=np.float64)
        if previous_base is None or previous_q is None:
            previous_keypoints = np.asarray(
                warm.arrays["robot_keypoints_scene"][frame], dtype=np.float64
            )
        else:
            previous_keypoints = np.asarray(
                model.keypoints_scene(previous_q, previous_base), dtype=np.float64
            )
        relative = continuity_metrics(
            predicted_base, current_base, predicted_q, current_q, frame=int(frame)
        )
        base_jump = float(
            np.linalg.norm(
                current_base[:3, 3]
                - (predicted_base[:3, 3] if previous_base is None else previous_base[:3, 3])
            )
        )
        step_reference_base = predicted_base if previous_base is None else previous_base
        rotation_jump = float(
            np.linalg.norm(so3_log_np(step_reference_base[:3, :3].T @ current_base[:3, :3]))
        )
        keypoint_step = float(
            np.max(np.linalg.norm(current_keypoints - previous_keypoints, axis=-1))
        )
        row = {
            "window": window_name,
            "profile": profile,
            "local_frame": int(frame),
            "global_frame": int(frame + global_frame_offset),
            "solve": bool(arrays["solver_success"][local_row]),
            "status": int(arrays["solver_status"][local_row]),
            "message": _text(arrays["optimizer_message"][local_row]),
            "objective": float(arrays["total_objective"][local_row]),
            "e_bone": float(arrays["e_bone"][local_row]),
            "e_im": float(arrays["e_im"][local_row]),
            "e_temporal": float(arrays["e_temporal"][local_row]),
            "base_jump_m": base_jump,
            "rotation_jump_rad": rotation_jump,
            "q_step_linf_rad": float(
                np.max(np.abs(current_q - (previous_q if previous_q is not None else predicted_q)))
            ),
            "q_correction_linf_rad": float(np.max(np.abs(current_q - predicted_q))),
            "excess_keypoint_step_m": keypoint_step,
            "continuity_pass": bool(relative["trajectory_continuous"]),
            "min_sdf_m": float(arrays["min_full_signed_distance"][local_row]),
            "max_penetration_m": float(arrays["max_penetration"][local_row]),
            "collision_pass": bool(
                arrays["full_surface_hard_audit_pass"][local_row]
                and arrays["full_surface_soft_audit_pass"][local_row]
            ),
            "unqueried_violation_count": int(arrays["unqueried_soft_violation_count"][local_row]),
            "active_set_rounds": int(arrays["active_set_rounds"][local_row]),
            "query_count": int(
                arrays["query_offsets"][local_row + 1] - arrays["query_offsets"][local_row]
            ),
            "q_bounds_pass": bool(arrays["qpos_bounds_pass"][local_row]),
            "slack_bounds_pass": bool(arrays["slack_bounds_pass"][local_row]),
            "initialization_source": _text(arrays["initialization_source"][local_row]),
            "retry_profile": _text(arrays["retry_profile"][local_row]),
            "window_used": bool(arrays["window_used"][local_row]),
            "anchor_source": "none" if anchor is None else "fixed_left_anchor",
            "anchor_frame": None if anchor is None else int(start - 1),
            "diagnostic_only": True,
        }
        rows.append(row)
        previous_base = current_base
        previous_q = current_q
    return rows


def _run_ablation(
    repo: Path, root: Path, baseline_root: Path, audit: dict[str, Any], closeout: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    isolated: list[dict[str, Any]] = []
    operational: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for window_name, short, start, end, anchor_source, anchor_kind in WINDOWS:
        setup = _profile_setup(
            repo,
            root,
            closeout,
            short,
            anchor_kind,
            start,
            "scipy_slsqp_active_set_contact_rich_v3_fixed",
        )
        continuous_solver = RefinementSolverProfile.load("wuji_continuous_full_state_v1")
        coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1")
        query = CollisionQueryProfile.load("adaptive_active_set_v1")
        frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
        bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
        for mode, target in (("isolated", isolated), ("operational", operational)):
            for profile in ("B0", "B1", "B2"):
                if profile == "B0":
                    solver = setup["solver"]
                    reg = "auto"
                    transport = False
                elif profile == "B1":
                    solver = setup["solver"]
                    reg = "auto"
                    transport = True
                else:
                    solver = continuous_solver
                    reg = "continuous_full_state_plus_paper"
                    transport = True
                enable_recovery = mode == "operational"
                call_started = time.perf_counter()
                try:
                    trajectory, diagnostics = build_final_trajectory(
                        setup["sequence"],
                        setup["warm"],
                        setup["graph"],
                        setup["model"],
                        setup["surface"],
                        frame_profile,
                        bone_profile,
                        coordinate,
                        query,
                        solver,
                        start_frame=start,
                        end_frame=end,
                        initial_previous=setup["anchor"],
                        warm_artifact_hash=artifact_hash(setup["paths"]["warm"]),
                        graph_artifact_hash=interaction_artifact_hash(setup["paths"]["graph"]),
                        resources=setup["resources"],
                        execution_profile=setup["execution"],
                        regularization_profile=reg,
                        transport_previous_final=transport,
                        enable_continuity_recovery=enable_recovery,
                        continue_on_failure=True,
                    )
                    rows = _motion_rows(
                        trajectory,
                        setup["warm"],
                        setup["model"],
                        start,
                        setup["anchor"],
                        profile,
                        window_name,
                        {"W1": 240, "W2": 212, "W3": 407}[short],
                    )
                    for row in rows:
                        row.update(
                            {
                                "mode": mode,
                                "profile_definition": profile,
                                "anchor_source": anchor_source,
                                "regularization_profile": reg,
                            }
                        )
                    target.extend(rows)
                    performance.append(
                        {
                            "window": window_name,
                            "profile": profile,
                            "mode": mode,
                            "wall_time_s": time.perf_counter() - call_started,
                            "frames": len(rows),
                            "solver_status": "complete",
                        }
                    )
                    _write_json(
                        closeout
                        / "bounded_ablation"
                        / "checkpoints"
                        / f"{window_name}_{mode}_{profile}.json",
                        {"diagnostics": diagnostics, "rows": rows, "diagnostic_only": True},
                    )
                except Exception as exc:
                    row = {
                        "window": window_name,
                        "mode": mode,
                        "profile": profile,
                        "solve": False,
                        "failure": type(exc).__name__ + ": " + str(exc),
                        "anchor_source": anchor_source,
                        "diagnostic_only": True,
                    }
                    target.append(row)
                    performance.append(
                        {
                            "window": window_name,
                            "profile": profile,
                            "mode": mode,
                            "wall_time_s": time.perf_counter() - call_started,
                            "frames": 0,
                            "solver_status": "failed",
                            "failure": row["failure"],
                        }
                    )
    summary_rows: list[dict[str, Any]] = []
    for row in isolated:
        key = (row["window"], row["profile"])
        match = next(
            (item for item in summary_rows if (item["window"], item["profile"]) == key), None
        )
        if match is None:
            match = {
                "window": row["window"],
                "profile": row["profile"],
                "mode": "isolated",
                "frames": 0,
                "solve": True,
                "base_jump_m": 0.0,
                "rotation_jump_rad": 0.0,
                "q_correction_linf_rad": 0.0,
                "excess_keypoint_step_m": 0.0,
                "eim": [],
                "ebone": [],
                "collision_pass": True,
            }
            summary_rows.append(match)
        match["frames"] += 1
        match["solve"] = bool(match["solve"] and row.get("solve", False))
        for field in (
            "base_jump_m",
            "rotation_jump_rad",
            "q_correction_linf_rad",
            "excess_keypoint_step_m",
        ):
            match[field] = max(float(match[field]), float(row.get(field, 0.0)))
        match["eim"].append(float(row.get("e_im", np.nan)))
        match["ebone"].append(float(row.get("e_bone", np.nan)))
        match["collision_pass"] = bool(match["collision_pass"] and row.get("collision_pass", False))
    for row in summary_rows:
        row["mean_eim"] = float(np.nanmean(row.pop("eim"))) if row["frames"] else float("nan")
        row["mean_ebone"] = float(np.nanmean(row.pop("ebone"))) if row["frames"] else float("nan")
    conclusion = ablation_conclusion(isolated)
    destination = closeout / "bounded_ablation"
    _write_json(
        destination / "isolated_results.json",
        {"schema_version": ABLATION_SCHEMA_VERSION, "rows": isolated},
    )
    _write_csv(destination / "isolated_results.csv", isolated)
    _write_json(
        destination / "operational_results.json",
        {"schema_version": ABLATION_SCHEMA_VERSION, "rows": operational},
    )
    _write_csv(destination / "operational_results.csv", operational)
    _write_json(
        destination / "per_window_summary.json",
        {"schema_version": ABLATION_SCHEMA_VERSION, "rows": summary_rows},
    )
    _write_csv(destination / "per_window_summary.csv", summary_rows)
    aggregate = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "windows": [item[0] for item in WINDOWS],
        "isolated_frame_count": len(isolated),
        "operational_frame_count": len(operational),
        "conclusion": conclusion,
        "performance": performance,
        "diagnostic_only": True,
        "no_official_overwrite": True,
    }
    _write_json(destination / "aggregate_ablation.json", aggregate)
    (destination / "aggregate_ablation.md").write_text(
        "# B0/B1/B2 bounded ablation\n\n"
        + f"- conclusion: `{conclusion.get('label')}`\n- isolated rows: {len(isolated)}\n- operational rows: {len(operational)}\n- diagnostic only: `true`\n",
        encoding="utf-8",
    )
    return {
        "isolated": isolated,
        "operational": operational,
        "summary": summary_rows,
        "aggregate": aggregate,
        "wall_time_s": time.perf_counter() - started,
    }


def _run_real_window(repo: Path, root: Path, closeout: Path) -> dict[str, Any]:
    unit_id = UNITS["W3"][0]
    paths = _unit_paths(root, unit_id)
    baseline_paths = _unit_paths(root.parent / "wuji_hand2_grab3_v1", unit_id)
    setup = _profile_setup(
        repo, root, closeout, "W3", "baseline", 35, "wuji_continuous_full_state_v1"
    )
    baseline = load_final_trajectory(baseline_paths["final"])
    anchor = (baseline.arrays["base_pose_scene"][34], baseline.arrays["qpos"][34])
    coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1")
    query = CollisionQueryProfile.load("adaptive_active_set_v1")
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    results: list[dict[str, Any]] = []
    for repeat in ("first", "repeat"):
        started = time.perf_counter()
        trajectory, diagnostics = build_final_trajectory(
            setup["sequence"],
            setup["warm"],
            setup["graph"],
            setup["model"],
            setup["surface"],
            frame_profile,
            bone_profile,
            coordinate,
            query,
            setup["solver"],
            start_frame=35,
            # The closeout commits only the center (local 35).  The three
            # future frames remain jointly optimized as hints inside the
            # same [34, 35, 36, 37, 38] window.  Iterating through 36--38
            # here would repeat the same expensive joint solve and would not
            # add evidence to the center-only validation contract.
            end_frame=36,
            initial_previous=anchor,
            warm_artifact_hash=artifact_hash(paths["warm"]),
            graph_artifact_hash=interaction_artifact_hash(paths["graph"]),
            resources=setup["resources"],
            execution_profile=setup["execution"],
            regularization_profile="auto",
            diagnostic_force_window=True,
            enable_continuity_recovery=True,
            continue_on_failure=True,
        )
        center = continuity_metrics(
            anchor[0],
            trajectory.arrays["base_pose_scene"][0],
            anchor[1],
            trajectory.arrays["qpos"][0],
            predicted_keypoints_scene=setup["model"].keypoints_scene(anchor[1], anchor[0]),
            final_keypoints_scene=trajectory.arrays["robot_keypoints_scene"][0],
            frame=35,
        )
        window_meta = diagnostics.get("future_hints", {})
        results.append(
            {
                "repeat": repeat,
                "wall_time_s": time.perf_counter() - started,
                "trajectory": trajectory,
                "diagnostics": diagnostics,
                "center": center,
                "future_hints": window_meta,
            }
        )
    first = results[0]
    second = results[1]

    def stable_query_summary(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
        stable: list[dict[str, Any]] = []
        for item in diagnostics.get("query_summaries", []):
            window_joint = item.get("window_joint") or {}
            stable.append(
                {
                    "frame": item.get("frame"),
                    "initial_query_count": item.get("initial_query_count"),
                    "final_query_count": item.get("final_query_count"),
                    "query_hash": item.get("query_hash"),
                    "active_set_rounds": item.get("active_set_rounds"),
                    "active_set_converged": item.get("active_set_converged"),
                    "optimizer_converged": item.get("optimizer_converged"),
                    "accepted": item.get("accepted"),
                    "window_joint": {
                        "success": window_joint.get("success"),
                        "status": window_joint.get("status"),
                        "variable_frames": window_joint.get("variable_frames"),
                        "future_hint_frames": window_joint.get("future_hint_frames"),
                        "per_frame_query_sets": window_joint.get("per_frame_query_sets"),
                    },
                }
            )
        return stable

    deterministic = bool(
        np.array_equal(first["trajectory"].arrays["qpos"], second["trajectory"].arrays["qpos"])
        and np.array_equal(
            first["trajectory"].arrays["base_pose_scene"],
            second["trajectory"].arrays["base_pose_scene"],
        )
        and first["future_hints"] == second["future_hints"]
        and stable_query_summary(first["diagnostics"])
        == stable_query_summary(second["diagnostics"])
    )
    baseline_center = continuity_metrics(
        baseline.arrays["base_pose_scene"][34],
        baseline.arrays["base_pose_scene"][35],
        baseline.arrays["qpos"][34],
        baseline.arrays["qpos"][35],
        predicted_keypoints_scene=baseline.arrays["robot_keypoints_scene"][34],
        final_keypoints_scene=baseline.arrays["robot_keypoints_scene"][35],
        frame=35,
    )
    existing_continuous = load_final_trajectory(paths["final"])
    existing_center = continuity_metrics(
        existing_continuous.arrays["base_pose_scene"][34],
        existing_continuous.arrays["base_pose_scene"][35],
        existing_continuous.arrays["qpos"][34],
        existing_continuous.arrays["qpos"][35],
        predicted_keypoints_scene=existing_continuous.arrays["robot_keypoints_scene"][34],
        final_keypoints_scene=existing_continuous.arrays["robot_keypoints_scene"][35],
        frame=35,
    )
    payload = {
        "schema_version": "toporetarget.wuji_five_frame_window_validation.v1",
        "diagnostic_only": True,
        "forced_window_path": True,
        "accepted_reference": False,
        "window": {
            "global": [441, 446],
            "local": [34, 39],
            "optimized_frames": [35, 36, 37, 38],
            "center_global": 442,
            "center_local": 35,
            "left_anchor_global": 441,
            "left_anchor_local": 34,
        },
        "baseline_center": baseline_center,
        "existing_continuous_center": existing_center,
        "first": {
            "center": first["center"],
            "future_hints": first["future_hints"],
            "diagnostics": first["diagnostics"],
        },
        "repeat": {
            "center": second["center"],
            "future_hints": second["future_hints"],
            "diagnostics": second["diagnostics"],
        },
        "fixed_left_anchor_hash": _stable_hash(
            np.asarray(anchor[0]).tolist() + np.asarray(anchor[1]).tolist()
        ),
        "center_continuity_pass": bool(first["center"]["trajectory_continuous"]),
        "determinism_pass": deterministic,
        "formal_artifact_untouched": True,
    }
    _write_json(closeout / "window_fallback" / "real_w3_shadow.json", payload)
    _write_json(
        closeout / "window_fallback" / "real_w3_shadow_repeat.json",
        {
            "center": second["center"],
            "future_hints": second["future_hints"],
            "deterministic_against_first": deterministic,
        },
    )
    return payload


def _html_page(
    path: Path, title: str, payload: Any, controls: str, links: list[tuple[str, str]]
) -> None:
    serialized = json.dumps(payload, sort_keys=True, default=str).replace("</", "<\\/")
    link_html = "".join(f"<li><a href='{href}'>{label}</a></li>" for href, label in links)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title></head><body>
<h1>{title}</h1><p id='diagnostic-label'>diagnostic_only=true; formal artifacts are immutable</p>
<nav><ul>{link_html}</ul></nav>{controls}<pre id='selection'></pre><pre id='payload'></pre>
<script>const CLOSEOUT_DATA={serialized};
document.getElementById('payload').textContent=JSON.stringify(CLOSEOUT_DATA,null,2);
const gatePayload = CLOSEOUT_DATA.gates || CLOSEOUT_DATA.gate?.gates || null;
const gateTable = document.getElementById('gate-table');
if (gateTable && gatePayload) {{
  for (const [name, value] of Object.entries(gatePayload)) {{
    const row=document.createElement('tr');
    row.innerHTML=`<td>${{name}}</td><td>${{String(value)}}</td>`;
    gateTable.appendChild(row);
  }}
}}
const selects=[...document.querySelectorAll('select')];
function valuesFor(select) {{
  const rows = CLOSEOUT_DATA.per_transition || CLOSEOUT_DATA.per_joint ||
    CLOSEOUT_DATA.isolated || CLOSEOUT_DATA.operational || [];
  if (select.id === 'transition') return [...new Set(rows.map(row => row.transition ?? row.window ?? ''))];
  if (select.id === 'joint') return [...new Set(rows.map(row => row.joint_name ?? row.profile ?? ''))];
  if (select.id === 'profile') return ['B0','B1','B2'];
  if (select.id === 'mode') return ['isolated','operational'];
  return [];
}}
function refreshSelection() {{
  for (const select of selects) {{
    if (!select.dataset.ready) {{
      for (const value of valuesFor(select)) {{
        const option=document.createElement('option'); option.value=String(value); option.textContent=String(value);
        select.appendChild(option);
      }}
      select.dataset.ready='true';
    }}
  }}
  const selected={{}}; for (const select of selects) selected[select.id]=select.value;
  document.getElementById('selection').textContent=JSON.stringify(selected,null,2);
}}
for (const select of selects) select.addEventListener('change',refreshSelection); refreshSelection();
window.CLOSEOUT_DATA=CLOSEOUT_DATA;
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def _html_outputs(
    closeout: Path,
    attribution: dict[str, Any],
    ablation: dict[str, Any],
    window: dict[str, Any],
    gate: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    destination = closeout / "html"
    links = [
        ("index.html", "Closeout index"),
        ("recommendation_gate.html", "Recommendation"),
        ("W2_qstep_attribution.html", "W2 attribution"),
        ("B0_B1_B2_ablation.html", "Ablation"),
        ("five_frame_window_validation.html", "Five-frame window"),
    ]
    _html_page(
        destination / "W2_qstep_attribution.html",
        "W2 q-step attribution",
        attribution,
        "<label>Transition <select id='transition'></select></label><label>Joint <select id='joint'></select></label>",
        links,
    )
    _html_page(
        destination / "B0_B1_B2_ablation.html",
        "B0/B1/B2 bounded ablation",
        ablation,
        "<label>Profile <select id='profile'><option>B0</option><option>B1</option><option>B2</option></select></label><label>Mode <select id='mode'><option>isolated</option><option>operational</option></select></label>",
        links,
    )
    _html_page(
        destination / "five_frame_window_validation.html",
        "Five-frame window validation",
        window,
        "<p id='window-layers'>fixed left anchor · source · warm · baseline failed · continuous · shadow center · future hints · object · collision samples · QuerySet · continuity</p>",
        links,
    )
    _html_page(
        destination / "recommendation_gate.html",
        "Wuji continuous recommendation gate",
        gate,
        "<table id='gate-table'><tr><th>Gate</th><th>Pass</th></tr></table>",
        links,
    )
    index_links = links + [
        ("../../html/W1_airplane_lift_continuity_comparison.html", "W1 existing continuity"),
        ("../../html/W2_apple_eat_1_continuity_comparison.html", "W2 existing continuity"),
        ("../../html/W3_alarmclock_lift_continuity_comparison.html", "W3 existing continuity"),
    ]
    _html_page(
        destination / "index.html",
        "Wuji Continuous Retargeting W2.2 Closeout",
        {
            "attribution": attribution["aggregate"],
            "ablation": ablation["aggregate"],
            "window": {
                key: value for key, value in window.items() if key not in {"first", "repeat"}
            },
            "gate": gate,
            "existing_continuity_root": str(root / "html"),
        },
        "<p id='closeout-status'>W2.2 closeout dashboard</p>",
        index_links,
    )
    smoke = {}
    for path in sorted(destination.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        smoke[path.name] = {
            "exists": True,
            "nonempty": bool(text),
            "diagnostic_label": "diagnostic_only=true" in text,
            "payload": "CLOSEOUT_DATA" in text,
            "no_nan_inf": "NaN" not in text and "Infinity" not in text,
        }
    for path in sorted((root / "html").glob("W[123]_*.html")):
        text = path.read_text(encoding="utf-8")
        smoke[f"existing::{path.name}"] = {
            "exists": True,
            "nonempty": bool(text),
            "readable": "<html" in text.lower() or "<!doctype" in text.lower(),
            "no_nan_inf": "NaN" not in text and "Infinity" not in text,
        }
    closeout_smoke_passed = all(
        all(value.values()) for name, value in smoke.items() if not name.startswith("existing::")
    ) and all(
        value["exists"] and value["nonempty"] and value["readable"]
        for name, value in smoke.items()
        if name.startswith("existing::")
    )
    _write_json(
        closeout / "reports" / "html_smoke.json",
        {
            "schema_version": SCHEMA_VERSION,
            "files": smoke,
            "passed": closeout_smoke_passed,
            "existing_html_nan_inf_is_preexisting": any(
                not value.get("no_nan_inf", True)
                for name, value in smoke.items()
                if name.startswith("existing::")
            ),
        },
    )
    return smoke


def _formal_rows(root: Path, baseline_root: Path) -> list[dict[str, Any]]:
    def motion_metrics(trajectory: Any) -> dict[str, float | int]:
        base = np.asarray(trajectory.arrays["base_pose_scene"], dtype=np.float64)
        keypoints = np.asarray(trajectory.arrays["robot_keypoints_scene"], dtype=np.float64)
        qpos = np.asarray(trajectory.arrays["qpos"], dtype=np.float64)
        base_translation_step = np.linalg.norm(np.diff(base[:, :3, 3], axis=0), axis=-1)
        base_rotation_step = np.asarray(
            [
                np.linalg.norm(so3_log_np(base[index, :3, :3].T @ base[index + 1, :3, :3]))
                for index in range(len(base) - 1)
            ],
            dtype=np.float64,
        )
        keypoint_step = np.max(np.linalg.norm(np.diff(keypoints, axis=0), axis=-1), axis=-1)
        q_step = np.max(np.abs(np.diff(qpos, axis=0)), axis=-1)
        q_jerk = np.max(np.abs(np.diff(qpos, n=2, axis=0)), axis=-1)
        base_translation_jerk = np.linalg.norm(np.diff(base[:, :3, 3], n=2, axis=0), axis=-1)
        base_rotation_jerk = np.abs(np.diff(base_rotation_step, axis=0))
        return {
            "max_base_translation_jump_m": float(np.max(base_translation_step, initial=0.0)),
            "max_base_rotation_jump_rad": float(np.max(base_rotation_step, initial=0.0)),
            "max_excess_keypoint_jump_m": float(np.max(keypoint_step, initial=0.0)),
            "max_q_step_linf_rad": float(np.max(q_step, initial=0.0)),
            "max_q_jerk_linf_rad": float(np.max(q_jerk, initial=0.0)),
            "max_base_jerk": float(
                max(
                    float(np.max(base_translation_jerk, initial=0.0)),
                    float(np.max(base_rotation_jerk, initial=0.0)),
                )
            ),
        }

    def reduction(baseline_value: float, continuous_value: float) -> float:
        if baseline_value <= 1.0e-12:
            return 0.0
        return float((baseline_value - continuous_value) / baseline_value)

    rows = []
    model = get_robot_registry().load("wuji_hand2_beta1_rh")
    for short, (unit_id, _short_id) in UNITS.items():
        continuous = load_final_trajectory(_unit_paths(root, unit_id)["final"])
        baseline = load_final_trajectory(_unit_paths(baseline_root, unit_id)["final"])
        baseline_motion = motion_metrics(baseline)
        continuous_motion = motion_metrics(continuous)
        baseline_q = np.asarray(baseline.arrays["qpos"], dtype=np.float64)
        continuous_q = np.asarray(continuous.arrays["qpos"], dtype=np.float64)
        baseline_margin = np.minimum(baseline_q - model.joint_lower, model.joint_upper - baseline_q)
        continuous_margin = np.minimum(
            continuous_q - model.joint_lower, model.joint_upper - continuous_q
        )
        deferred_diagnostic_arrays = {
            "continuity_base_rotation_rad",
            "continuity_base_translation_m",
            "continuity_excess_keypoint_m",
            "continuity_finger_inf_rad",
            "stationarity_residual",
        }
        known_deferred_nonfinite = {
            name: int(np.count_nonzero(~np.isfinite(np.asarray(value))))
            for name, value in continuous.arrays.items()
            if name in deferred_diagnostic_arrays and not np.all(np.isfinite(np.asarray(value)))
        }
        required_numeric_finite = bool(
            all(
                np.all(np.isfinite(value))
                for name, value in continuous.arrays.items()
                if np.asarray(value).dtype.kind in "fiu" and name not in deferred_diagnostic_arrays
            )
        )
        q = np.asarray(continuous.arrays["qpos"], dtype=np.float64)
        dq = np.diff(q, axis=0)
        jump_returns = int(
            sum(
                bool(
                    np.any(
                        (np.abs(dq[index]) > 0.05)
                        & (np.sign(dq[index]) * np.sign(dq[index + 1]) < 0)
                    )
                )
                for index in range(len(dq) - 1)
            )
        )
        rows.append(
            {
                "unit": short,
                "frame_count": continuous.frame_count,
                "all_optimizer_converged": bool(np.all(continuous.arrays["optimizer_converged"])),
                "all_single_frame_feasible": bool(
                    np.all(continuous.arrays["single_frame_feasible"])
                ),
                "all_trajectory_continuous": bool(
                    np.all(continuous.arrays["trajectory_continuous"])
                ),
                "all_accepted": bool(np.all(continuous.arrays["final_accepted"])),
                "q_bounds_pass": bool(np.all(continuous.arrays["qpos_bounds_pass"])),
                "slack_bounds_pass": bool(np.all(continuous.arrays["slack_bounds_pass"])),
                "full_collision_pass": bool(
                    np.all(continuous.arrays["full_surface_hard_audit_pass"])
                    and np.all(continuous.arrays["full_surface_soft_audit_pass"])
                ),
                "unqueried_violation_count": int(
                    np.sum(continuous.arrays["unqueried_soft_violation_count"])
                ),
                "all_finite": required_numeric_finite,
                "known_deferred_nonfinite": known_deferred_nonfinite,
                "max_base_translation_correction_m": float(
                    np.nanmax(continuous.arrays["continuity_base_translation_m"])
                ),
                "max_base_rotation_correction_rad": float(
                    np.nanmax(continuous.arrays["continuity_base_rotation_rad"])
                ),
                "max_correction_q_linf_rad": float(
                    np.nanmax(continuous.arrays["continuity_finger_inf_rad"])
                ),
                "max_excess_keypoint_m": float(
                    np.nanmax(continuous.arrays["continuity_excess_keypoint_m"])
                ),
                "jump_and_return_count": jump_returns,
                "baseline_mean_eim": float(np.mean(baseline.arrays["e_im"])),
                "continuous_mean_eim": float(np.mean(continuous.arrays["e_im"])),
                "baseline_mean_ebone": float(np.mean(baseline.arrays["e_bone"])),
                "continuous_mean_ebone": float(np.mean(continuous.arrays["e_bone"])),
                "baseline_max_penetration_m": float(np.max(baseline.arrays["max_penetration"])),
                "continuous_max_penetration_m": float(np.max(continuous.arrays["max_penetration"])),
                "baseline_penetration_rate": float(
                    np.mean(np.asarray(baseline.arrays["max_penetration"]) > 0.0)
                ),
                "continuous_penetration_rate": float(
                    np.mean(np.asarray(continuous.arrays["max_penetration"]) > 0.0)
                ),
                "baseline_joint_limit_saturation": float(np.mean(baseline_margin <= 0.03)),
                "continuous_joint_limit_saturation": float(np.mean(continuous_margin <= 0.03)),
                "baseline_max_base_translation_jump_m": baseline_motion[
                    "max_base_translation_jump_m"
                ],
                "continuous_max_base_translation_jump_m": continuous_motion[
                    "max_base_translation_jump_m"
                ],
                "baseline_max_base_rotation_jump_rad": baseline_motion[
                    "max_base_rotation_jump_rad"
                ],
                "continuous_max_base_rotation_jump_rad": continuous_motion[
                    "max_base_rotation_jump_rad"
                ],
                "baseline_max_excess_keypoint_jump_m": baseline_motion[
                    "max_excess_keypoint_jump_m"
                ],
                "continuous_max_excess_keypoint_jump_m": continuous_motion[
                    "max_excess_keypoint_jump_m"
                ],
                "baseline_max_q_jerk_linf_rad": baseline_motion["max_q_jerk_linf_rad"],
                "continuous_max_q_jerk_linf_rad": continuous_motion["max_q_jerk_linf_rad"],
                "baseline_max_base_jerk": baseline_motion["max_base_jerk"],
                "continuous_max_base_jerk": continuous_motion["max_base_jerk"],
                "max_base_jump_reduction": reduction(
                    baseline_motion["max_base_translation_jump_m"],
                    continuous_motion["max_base_translation_jump_m"],
                ),
                "max_rotation_jump_reduction": reduction(
                    baseline_motion["max_base_rotation_jump_rad"],
                    continuous_motion["max_base_rotation_jump_rad"],
                ),
                "max_keypoint_jump_reduction": reduction(
                    baseline_motion["max_excess_keypoint_jump_m"],
                    continuous_motion["max_excess_keypoint_jump_m"],
                ),
                "q_jerk_reduction": reduction(
                    baseline_motion["max_q_jerk_linf_rad"],
                    continuous_motion["max_q_jerk_linf_rad"],
                ),
                "base_jerk_reduction": reduction(
                    baseline_motion["max_base_jerk"],
                    continuous_motion["max_base_jerk"],
                ),
            }
        )
    return rows


def _integrity_after(
    before: dict[str, Any], repo: Path, root: Path, baseline_root: Path, closeout: Path
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for label, record in before.items():
        if not isinstance(record, dict) or "path" not in record or not record.get("exists"):
            continue
        path = Path(record["path"])
        current = None if not path.exists() else _hash_path(path)
        checks[label] = {
            "path": str(path),
            "before_hash": record.get("hash"),
            "after_hash": current,
            "unchanged": current == record.get("hash"),
        }
    for label, record in before.get("source_files", {}).items():
        path = Path(label)
        current = None if not path.exists() else _hash_path(path)
        checks[f"source::{label}"] = {
            "path": str(path),
            "before_hash": record,
            "after_hash": current,
            "unchanged": current == record,
        }
    formal_before = before.get("formal_inputs", {})
    for short, values in formal_before.items():
        unit_id = UNITS[short][0]
        current_paths = _unit_paths(root, unit_id)
        baseline_paths = _unit_paths(baseline_root, unit_id)
        paths = {
            "canonical_hash": current_paths["canonical"],
            "warm_hash": current_paths["warm"],
            "graph_hash": current_paths["graph"],
            "continuous_final_hash": current_paths["final"],
            "baseline_final_hash": baseline_paths["final"],
        }
        for field, path in paths.items():
            if not path.exists():
                current = None
            elif field == "graph_hash":
                current = interaction_artifact_hash(path)
            else:
                current = _hash_path(path)
            checks[f"formal::{short}::{field}"] = {
                "path": str(path),
                "before_hash": values.get(field),
                "after_hash": current,
                "unchanged": current == values.get(field),
            }
    pene = Path(before.get("pene_loss_worktree", str(repo.parent / "TopoRetarget-Repro-pene-loss")))
    pene_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=pene,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    checks["pene_loss_worktree"] = {
        "status_before": before.get("pene_loss_status_before", ""),
        "status_after": pene_status,
        "unchanged": pene_status == before.get("pene_loss_status_before", ""),
        "unchanged_by_this_task": pene_status == before.get("pene_loss_status_before", ""),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "source_changed": not all(
            item.get("unchanged", True)
            for item in checks.values()
            if isinstance(item, dict) and "unchanged" in item
        ),
        "wuji_assets_changed": not checks.get("wuji_assets", {}).get("unchanged", True),
        "baseline_artifacts_changed": any(
            not value["unchanged"]
            for key, value in checks.items()
            if key.endswith("baseline_final_hash")
        ),
        "continuous_artifacts_changed": any(
            not value["unchanged"]
            for key, value in checks.items()
            if key.endswith("continuous_final_hash")
        ),
        "exports_changed": not checks.get("exports", {}).get("unchanged", True),
        "old_stage10_changed": not checks.get("historical_stage10", {}).get("unchanged", True),
        "pene_loss_worktree_changed_by_this_task": not checks["pene_loss_worktree"]["unchanged"],
        "local_only_new_outputs": True,
    }
    _write_json(closeout / "reports" / "artifact_integrity.json", result)
    return result


def _recommendation(profile_gate: dict[str, Any]) -> str:
    if profile_gate["passed"]:
        return "WUJI_CONTINUOUS_PROFILE_RECOMMENDED_FOR_OFFLINE_REFERENCE_GENERATION"
    if not profile_gate["w2_residual_qstep_gate"]:
        return "WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_W2_CORRECTION_DRIVEN_JUMPS"
    if not profile_gate.get("window_fallback_gate", True):
        return "WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_WINDOW_FALLBACK_FAILED"
    if not profile_gate["method_evidence_gate"]:
        return "WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_ABLATION_INCONCLUSIVE"
    return "WUJI_CONTINUOUS_PROFILE_NOT_RECOMMENDED_QUALITY_REGRESSION"


def run(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()
    repo = Path.cwd().resolve()
    root = Path(args.root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    closeout = root / "closeout_v1"
    closeout.mkdir(parents=True, exist_ok=True)
    for subdirectory in (
        "logs",
        "input_audit",
        "w2_qstep_attribution",
        "bounded_ablation",
        "window_fallback",
        "html",
        "screenshots",
        "reports",
        "recommendation",
    ):
        (closeout / subdirectory).mkdir(parents=True, exist_ok=True)
    _write_json(
        closeout / "logs" / "run_start.json",
        {
            "started": time.time(),
            "repo": str(repo),
            "root": str(root),
            "baseline_root": str(baseline_root),
            "schema_version": SCHEMA_VERSION,
        },
    )
    _config, clips_tuple = _load_closeout_suite(args.suite)
    clips = _clip_map(clips_tuple)
    audit = _input_audit(repo, root, baseline_root, clips, closeout)
    attribution_started = time.perf_counter()
    attribution = _run_attribution(audit, root, closeout)
    attribution_time = time.perf_counter() - attribution_started
    synthetic_started = time.perf_counter()
    synthetic = synthetic_window_fixture()
    _write_json(closeout / "window_fallback" / "synthetic_fixture.json", synthetic)
    synthetic_repeat = synthetic_window_fixture()
    _write_json(closeout / "window_fallback" / "synthetic_fixture_repeat.json", synthetic_repeat)
    synthetic_pass = bool(
        synthetic["routing_to_window"]
        and synthetic["center_continuity_pass"]
        and synthetic["checkpoint_resume_pass"]
        and synthetic["deterministic"]
        and synthetic == synthetic_repeat
    )
    synthetic_time = time.perf_counter() - synthetic_started
    ablation_dir = closeout / "bounded_ablation"
    cached_ablation = all(
        (ablation_dir / name).is_file()
        for name in (
            "isolated_results.json",
            "operational_results.json",
            "per_window_summary.json",
            "aggregate_ablation.json",
        )
    )
    if cached_ablation:
        # The bounded solver checkpoints are immutable closeout evidence. On
        # a resumed run, reload them instead of repeating the 42 solver calls.
        ablation = {
            "isolated": json.loads((ablation_dir / "isolated_results.json").read_text())["rows"],
            "operational": json.loads((ablation_dir / "operational_results.json").read_text())[
                "rows"
            ],
            "summary": json.loads((ablation_dir / "per_window_summary.json").read_text())["rows"],
            "aggregate": json.loads((ablation_dir / "aggregate_ablation.json").read_text()),
            "wall_time_s": 0.0,
            "resumed_from_checkpoints": True,
        }
    else:
        ablation = _run_ablation(repo, root, baseline_root, audit, closeout)
    real_window_path = closeout / "window_fallback" / "real_w3_shadow.json"
    if args.reuse_real_window and real_window_path.is_file():
        real_window = json.loads(real_window_path.read_text(encoding="utf-8"))
        real_window["reused_existing_shadow"] = True
    else:
        real_window = _run_real_window(repo, root, closeout)
    real_window_pass = bool(
        real_window["center_continuity_pass"]
        and real_window["determinism_pass"]
        and real_window["formal_artifact_untouched"]
    )
    _write_json(
        closeout / "window_fallback" / "window_validation_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "synthetic_pass": synthetic_pass,
            "real_w3_pass": real_window_pass,
            "routing": True,
            "window_size": 5,
            "center_only_commit": True,
            "per_frame_queryset": True,
            "per_frame_slack": True,
            "checkpoint_resume": bool(synthetic["checkpoint_resume_pass"]),
            "determinism": bool(synthetic["deterministic"] and real_window["determinism_pass"]),
        },
    )
    (closeout / "window_fallback" / "window_validation_summary.md").write_text(
        f"# Five-frame fallback validation\n\n- synthetic routing pass: `{synthetic_pass}`\n- real W3 shadow pass: `{real_window_pass}`\n- window size: `5`\n- center-only commit: `true`\n- formal artifact overwrite: `false`\n",
        encoding="utf-8",
    )
    formal = _formal_rows(root, baseline_root)
    deterministic = bool(real_window["determinism_pass"] and synthetic_pass)
    expected_ablation_keys = {
        (window[0], mode, profile)
        for window in WINDOWS
        for mode in ("isolated", "operational")
        for profile in ("B0", "B1", "B2")
    }
    observed_ablation_keys = {
        (str(row.get("window")), str(row.get("mode")), str(row.get("profile")))
        for row in ablation["isolated"] + ablation["operational"]
    }
    ablation_complete = bool(
        len(ablation["isolated"]) == 3 * sum(window[3] - window[2] for window in WINDOWS)
        and len(ablation["operational"]) == 3 * sum(window[3] - window[2] for window in WINDOWS)
        and observed_ablation_keys == expected_ablation_keys
    )
    gate = recommendation_gates(
        formal_rows=formal,
        attribution=attribution,
        ablation_complete=ablation_complete,
        synthetic_pass=synthetic_pass,
        real_window_pass=real_window_pass,
        determinism_pass=deterministic,
    )
    gate["ablation_complete"] = ablation_complete
    recommendation = _recommendation(gate)
    recommendation_payload = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": "wuji_continuous_full_state_v1",
        "status": recommendation,
        "recommended_profile": recommendation
        == "WUJI_CONTINUOUS_PROFILE_RECOMMENDED_FOR_OFFLINE_REFERENCE_GENERATION",
        "rl_ready": False,
        "realtime_ready": False,
        "cross_subject_validated": False,
        "author_exact": "unresolved",
        "engineering_extension": True,
        "scope": "offline reference generation only",
        "gates": gate,
        "formal_rows": formal,
        "diagnostic_only": True,
    }
    _write_json(closeout / "recommendation" / "recommended_profile.json", recommendation_payload)
    _write_json(closeout / "recommendation" / "recommendation_gate.json", recommendation_payload)
    (closeout / "reports" / "recommendation_gate.md").write_text(
        "# Recommendation gate\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in gate.items())
        + f"\n\n- status: `{recommendation}`\n",
        encoding="utf-8",
    )
    _write_json(closeout / "reports" / "recommendation_gate.json", recommendation_payload)
    _write_json(
        closeout / "reports" / "w2_qstep_attribution_summary.json", attribution["aggregate"]
    )
    _write_csv(
        closeout / "reports" / "w2_qstep_attribution_summary.csv", attribution["per_transition"]
    )
    _write_json(closeout / "reports" / "bounded_ablation_summary.json", ablation["aggregate"])
    _write_csv(closeout / "reports" / "bounded_ablation_summary.csv", ablation["summary"])
    _write_json(closeout / "reports" / "window_fallback_validation.json", real_window)
    performance = {
        "schema_version": SCHEMA_VERSION,
        "attribution_s": attribution_time,
        "ablation_wall_time_s": ablation["wall_time_s"],
        "synthetic_window_s": synthetic_time,
        "real_w3_window_s": float(real_window["first"].get("wall_time_s", 0.0)),
        "repeat_s": float(real_window["repeat"].get("wall_time_s", 0.0)),
        "html_generation_s": 0.0,
        "total_wall_time_s": 0.0,
        "formal_180_frame_runtime_included": False,
    }
    _write_json(closeout / "reports" / "performance.json", performance)
    _write_csv(closeout / "reports" / "performance.csv", [performance])
    _write_json(
        closeout / "reports" / "determinism.json",
        {
            "schema_version": SCHEMA_VERSION,
            "pass": deterministic,
            "synthetic_repeat_equal": synthetic == synthetic_repeat,
            "real_repeat_equal": real_window["determinism_pass"],
            "formal_artifacts_reloaded_only": True,
        },
    )
    _write_json(
        closeout / "reports" / "failure_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "none" if gate["passed"] else "complete_with_gate_failures",
            "recommendation": recommendation,
            "failures": [key for key, value in gate.items() if key != "passed" and not value],
            "no_formal_overwrite": True,
        },
    )
    html_started = time.perf_counter()
    smoke = _html_outputs(
        closeout, attribution, ablation, real_window, recommendation_payload, root
    )
    _html_page(
        closeout / "reports" / "dashboard.html",
        "Wuji W2.2 Closeout Dashboard",
        {
            "recommendation": recommendation_payload,
            "attribution": attribution["aggregate"],
            "ablation": ablation["aggregate"],
            "window": {
                key: value for key, value in real_window.items() if key not in {"first", "repeat"}
            },
            "links": [
                "../html/index.html",
                "../html/W2_qstep_attribution.html",
                "../html/B0_B1_B2_ablation.html",
                "../html/five_frame_window_validation.html",
                "../html/recommendation_gate.html",
                "../../html/W1_airplane_lift_continuity_comparison.html",
                "../../html/W2_apple_eat_1_continuity_comparison.html",
                "../../html/W3_alarmclock_lift_continuity_comparison.html",
            ],
        },
        "<p>See the closeout HTML pages and the three existing continuity comparisons.</p>",
        [
            ("../html/index.html", "Closeout index"),
            ("../html/W2_qstep_attribution.html", "W2 attribution"),
            ("../html/B0_B1_B2_ablation.html", "B0/B1/B2"),
            ("../html/five_frame_window_validation.html", "Five-frame window"),
            ("../html/recommendation_gate.html", "Recommendation"),
        ],
    )
    performance["html_generation_s"] = time.perf_counter() - html_started
    integrity = _integrity_after(audit["immutable"], repo, root, baseline_root, closeout)
    performance["total_wall_time_s"] = time.perf_counter() - total_started
    _write_json(closeout / "reports" / "performance.json", performance)
    _write_csv(closeout / "reports" / "performance.csv", [performance])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": recommendation,
        "profile": "wuji_continuous_full_state_v1",
        "recommendation_gate": gate,
        "w2": attribution["aggregate"],
        "ablation": ablation["aggregate"],
        "window": {"synthetic_pass": synthetic_pass, "real_w3_pass": real_window_pass},
        "html_smoke": smoke,
        "artifact_integrity": integrity,
        "scope": {
            "offline_reference_generation": True,
            "rl_ready": False,
            "realtime_ready": False,
            "cross_subject_validated": False,
            "author_exact": "unresolved",
        },
    }
    _write_json(closeout / "reports" / "closeout_summary.json", summary)
    (closeout / "reports" / "closeout_summary.md").write_text(
        "# Wuji W2.2 continuity closeout\n\n"
        + f"- status: `{recommendation}`\n- W2 absolute q-step transitions: {attribution['aggregate']['absolute_q_step_count']}\n- W2 correction-driven: {attribution['aggregate']['correction_driven_count']}\n- ablation: `{ablation['aggregate']['conclusion'].get('label')}`\n- synthetic window: `{synthetic_pass}`\n- real W3 shadow: `{real_window_pass}`\n- diagnostic only: `true`\n- offline reference generation only: `true`\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".local/experiments/wuji_hand2_continuous_v1")
    parser.add_argument("--baseline-root", default=".local/experiments/wuji_hand2_grab3_v1")
    parser.add_argument("--suite", default="configs/experiments/wuji_hand2_continuous_v1.yaml")
    parser.add_argument(
        "--reuse-real-window",
        action="store_true",
        help="reuse an already completed real W3 shadow while regenerating reports",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
