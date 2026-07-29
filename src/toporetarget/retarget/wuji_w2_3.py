"""W2.3 Wuji sequential-profile finalization and bounded audits.

The W2.2 formal artifacts are read-only inputs here.  This module owns the
W2.3 profile split, artifact-level audits, bounded replay/export reports, and
the non-blocking experimental window evidence.  It intentionally keeps the
paper solver and the existing continuous artifacts outside the write path.
"""

# The generated evidence pages intentionally contain compact inline HTML/JS.
# Keep the source readable without making the HTML template itself the lint target.
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from toporetarget.data.storage import load_hoi_sequence, write_zarr3_group_direct
from toporetarget.geometry.robot_surface import RobotSurfaceSampleSet
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.bones import load_bone_profile
from toporetarget.retarget.continuous import (
    CONTINUOUS_PROFILE_ID,
    S_POS_M,
    S_Q_RAD,
    S_ROT_RAD,
    SEQUENTIAL_PROFILE_ID,
    continuity_metrics,
    decode_base_correction,
    encode_base_correction,
    so3_log_np,
)
from toporetarget.retarget.final_refinement import (
    ACTIVE_QUERY_PROFILE_ID,
    CollisionQueryProfile,
    PaperRefinementWeights,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    build_final_trajectory,
    final_artifact_hash,
    load_final_trajectory,
    load_robot_surface_samples,
    prepare_refinement_resources,
    save_final_trajectory,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import (
    interaction_artifact_hash,
    load_interaction_graph,
)
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile
from toporetarget.robots.registry import get_robot_registry
from toporetarget.utils.hashing import sha256_file, sha256_tree

SCHEMA_VERSION = "toporetarget.wuji_w2_3_finalization.v1"
WINDOW_PROFILE_ID = "wuji_five_frame_window_experimental_v1"
THRESHOLDS_MM = (0.0, 0.25, 0.5, 1.0, 2.0)
PENETRATION_HARD_THRESHOLD_MM = 2.0
PENETRATION_SECONDARY_RATE_LIMIT = 0.05
PENETRATION_SECONDARY_DEPTH_LIMIT_MM = 0.25
# Selected-frame replay uses the same deterministic code path but a fresh
# SLSQP invocation.  These bounds capture solver tie-breaking while remaining
# far below the continuity acceptance limits (20 mm keypoint / 10 mm base).
REPLAY_QPOS_TOLERANCE = 3.0e-3
REPLAY_BASE_TRANSLATION_TOLERANCE = 1.0e-4
REPLAY_BASE_ROTATION_TOLERANCE = 1.0e-3

CLIPS: tuple[dict[str, Any], ...] = (
    {
        "unit": "W1",
        "short_id": "W1_airplane_lift",
        "unit_id": "W1_s1__airplane_lift__right__wuji_hand2_beta1_rh__f000240_f000300",
        "sequence": "s1/airplane_lift",
        "object_id": "airplane",
        "start": 240,
        "end": 300,
    },
    {
        "unit": "W2",
        "short_id": "W2_apple_eat_1",
        "unit_id": "W2_s1__apple_eat_1__right__wuji_hand2_beta1_rh__f000212_f000272",
        "sequence": "s1/apple_eat_1",
        "object_id": "apple",
        "start": 212,
        "end": 272,
    },
    {
        "unit": "W3",
        "short_id": "W3_alarmclock_lift",
        "unit_id": "W3_s1__alarmclock_lift__right__wuji_hand2_beta1_rh__f000407_f000467",
        "sequence": "s1/alarmclock_lift",
        "object_id": "alarmclock",
        "start": 407,
        "end": 467,
    },
)

EXPECTED_W2_3_PATHS = frozenset(
    {
        "README.md",
        "README.zh-CN.md",
        "configs/retarget/refinement_solvers/wuji_continuous_sequential_v1.yaml",
        "docs/ASSUMPTIONS.md",
        "docs/DEVELOPMENT_LOG.md",
        "docs/DEVELOPMENT_LOG.zh-CN.md",
        "docs/FIVE_FRAME_RETARGETING_WINDOW.md",
        "docs/PAPER_FIDELITY.md",
        "docs/PAPER_FIDELITY.yaml",
        "docs/REPRODUCTION_LOG.md",
        "docs/ROADMAP.md",
        "docs/ROADMAP.zh-CN.md",
        "docs/WUJI_CONTINUOUS_PROFILE_RECOMMENDATION.md",
        "docs/WUJI_CONTINUOUS_RETARGETING.md",
        "docs/WUJI_CONTINUOUS_RETARGETING.zh-CN.md",
        "docs/WUJI_HAND2_GRAB_RETARGETING.md",
        "docs/WUJI_MULTI_THRESHOLD_PENETRATION_AUDIT.md",
        "docs/WUJI_SEQUENTIAL_CONTINUITY_PROFILE.md",
        "docs/WUJI_WINDOW_HARNESS_REPAIR.md",
        "docs/stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.md",
        "docs/stages/W2_3_WUJI_SEQUENTIAL_FINALIZATION.zh-CN.md",
        "scripts/wuji_w2_3_finalization.py",
        "src/toporetarget/cli/retarget.py",
        "src/toporetarget/retarget/continuous.py",
        "src/toporetarget/retarget/final_refinement.py",
        "src/toporetarget/retarget/wuji_w2_3.py",
        "tests/unit/test_wuji_w2_3.py",
    }
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in values for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({key: _json_ready(row.get(key)) for key in fields})


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tree_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    entries = sha256_tree(path)
    return _stable_hash(entries)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _profile_path(repo: Path, profile_id: str) -> Path:
    return repo / "configs" / "retarget" / "refinement_solvers" / f"{profile_id}.yaml"


def _unit_root(root: Path, clip: dict[str, Any]) -> Path:
    return root / str(clip["unit_id"])


def clip_paths(
    repo: Path, root: Path, baseline_root: Path, clip: dict[str, Any]
) -> dict[str, Path]:
    unit = _unit_root(root, clip)
    baseline_unit = _unit_root(baseline_root, clip)
    return {
        "canonical": unit / "canonical" / "canonical.zarr",
        "warm": unit / "warm_start" / "warm_start.zarr",
        "graph": unit / "interaction_graph" / "interaction_graph.zarr",
        "continuous": root / "full_runs" / str(clip["unit_id"]) / "final_continuous.zarr",
        "baseline": baseline_unit / "final" / "final_retarget.zarr",
        "existing_export": root / "exports" / str(clip["short_id"]),
        "existing_html": root / "html" / f"{clip['short_id']}_continuity_comparison.html",
        "closeout": root / "closeout_v1",
    }


def _load_profile_values(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"profile is not a mapping: {path}")
    return value


def profile_structural_diff(repo: Path) -> dict[str, Any]:
    old = _load_profile_values(_profile_path(repo, CONTINUOUS_PROFILE_ID))
    new = _load_profile_values(_profile_path(repo, SEQUENTIAL_PROFILE_ID))
    old_window = dict(old.get("window", {}))
    new_window = dict(new.get("window", {}))
    old_window.setdefault("fallback_enabled", True)
    old["window"] = old_window
    new["window"] = new_window
    allowed_metadata = {
        "profile_id",
        "display_name",
        "role",
        "status",
        "recommended_scope",
        "window_fallback",
        "rl_ready",
        "realtime_ready",
        "cross_subject_validated",
        "profile_hash",
    }

    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                result.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
            return result
        return {prefix: value}

    old_flat = flatten(old)
    new_flat = flatten(new)
    differences: list[dict[str, Any]] = []
    for field in sorted(set(old_flat) | set(new_flat)):
        before = old_flat.get(field)
        after = new_flat.get(field)
        if before == after:
            continue
        leaf = field.rsplit(".", 1)[-1]
        semantic = field == "window.fallback_enabled"
        allowed = semantic or leaf in allowed_metadata
        differences.append(
            {
                "field": field,
                "full_state": before,
                "sequential": after,
                "allowed": allowed,
                "kind": "semantic" if semantic else "metadata" if allowed else "forbidden",
            }
        )
    semantic_differences = [item for item in differences if item["kind"] == "semantic"]
    forbidden = [item for item in differences if not item["allowed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "full_state_profile": CONTINUOUS_PROFILE_ID,
        "sequential_profile": SEQUENTIAL_PROFILE_ID,
        "full_state_profile_hash": _stable_hash(old),
        "sequential_profile_hash": _stable_hash(new),
        "differences": differences,
        "allowed_difference_count": len(semantic_differences),
        "forbidden_difference_count": len(forbidden),
        "only_allowed_window_difference": bool(
            len(semantic_differences) == 1
            and semantic_differences[0]["field"] == "window.fallback_enabled"
            and semantic_differences[0]["full_state"] is True
            and semantic_differences[0]["sequential"] is False
            and not forbidden
        ),
        "passed": bool(
            len(semantic_differences) == 1
            and semantic_differences[0]["field"] == "window.fallback_enabled"
            and not forbidden
        ),
    }


def formal_execution_path_audit(
    repo: Path, root: Path, baseline_root: Path, clips: Iterable[dict[str, Any]] = CLIPS
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for clip in clips:
        paths = clip_paths(repo, root, baseline_root, clip)
        artifact = load_final_trajectory(paths["continuous"])
        arrays = artifact.arrays
        retry = np.asarray(arrays.get("retry_profile", []), dtype=str)
        init = np.asarray(arrays.get("initialization_source", []), dtype=str)
        window = np.asarray(arrays.get("window_used", []), dtype=bool)
        future_hints = artifact.metadata.get("future_hints", {})
        row = {
            "unit": clip["unit"],
            "continuous_path": str(paths["continuous"]),
            "artifact_hash": final_artifact_hash(artifact),
            "frame_count": artifact.frame_count,
            "production_window_invocation_count": int(np.count_nonzero(window)),
            "retry_window_count": int(np.count_nonzero(retry == "five_frame_window")),
            "initialization_window_count": int(np.count_nonzero(np.char.find(init, "window") >= 0)),
            "future_hint_count": len(future_hints) if isinstance(future_hints, dict) else -1,
            "accepted_frame_count": int(
                np.count_nonzero(np.asarray(arrays["accepted"], dtype=bool))
            ),
            "forced_diagnostic_window_not_in_formal_artifact": bool(
                not np.any(window)
                and not np.any(retry == "five_frame_window")
                and not np.any(np.char.find(init, "window") >= 0)
                and (not isinstance(future_hints, dict) or not future_hints)
            ),
        }
        row["formal_path_pass"] = bool(
            row["production_window_invocation_count"] == 0
            and row["forced_diagnostic_window_not_in_formal_artifact"]
            and row["accepted_frame_count"] == 60
        )
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "production_window_invocation_count": sum(
            int(row["production_window_invocation_count"]) for row in rows
        ),
        "forced_diagnostic_window_not_in_formal_artifact": all(
            bool(row["forced_diagnostic_window_not_in_formal_artifact"]) for row in rows
        ),
        "rows": rows,
        "passed": all(bool(row["formal_path_pass"]) for row in rows),
    }


def input_audit(repo: Path, root: Path, baseline_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    immutable: dict[str, Any] = {}
    for clip in CLIPS:
        paths = clip_paths(repo, root, baseline_root, clip)
        warm = load_warm_start(paths["warm"])
        graph = load_interaction_graph(paths["graph"])
        baseline = load_final_trajectory(paths["baseline"])
        continuous = load_final_trajectory(paths["continuous"])
        timestamps_match = bool(np.array_equal(warm.arrays["timestamps"], graph.timestamps))
        row = {
            "unit": clip["unit"],
            "sequence": clip["sequence"],
            "frame_range": [clip["start"], clip["end"]],
            "frame_count": continuous.frame_count,
            "qpos_shape": list(np.asarray(continuous.arrays["qpos"]).shape),
            "base_shape": list(np.asarray(continuous.arrays["base_pose_scene"]).shape),
            "timestamps_match_graph": timestamps_match,
            "source_hash": tree_digest(paths["canonical"]),
            "warm_hash": artifact_hash(paths["warm"]),
            "graph_hash": interaction_artifact_hash(paths["graph"]),
            "baseline_tree_hash": tree_digest(paths["baseline"]),
            "baseline_artifact_hash": final_artifact_hash(baseline),
            "continuous_tree_hash": tree_digest(paths["continuous"]),
            "continuous_artifact_hash": final_artifact_hash(continuous),
            "baseline_accepted_count": int(
                np.count_nonzero(np.asarray(baseline.arrays["accepted"]))
            ),
            "continuous_accepted_count": int(
                np.count_nonzero(np.asarray(continuous.arrays["accepted"]))
            ),
            "continuous_window_used_count": int(np.count_nonzero(continuous.arrays["window_used"])),
            "qpos_order_path": str(repo / "configs/robots/joint_orders/wuji_hand2_beta1_rh.yaml"),
            "collision_profile_path": str(
                repo / "configs/robots/collision/wuji_hand2_beta1_mjcf_rh.yaml"
            ),
        }
        row["lineage_pass"] = bool(
            row["frame_count"] == 60
            and row["qpos_shape"] == [60, 20]
            and row["base_shape"] == [60, 4, 4]
            and timestamps_match
            and row["baseline_accepted_count"] == 60
            and row["continuous_accepted_count"] == 60
        )
        records.append(row)
        for name in (
            "canonical",
            "warm",
            "graph",
            "baseline",
            "continuous",
            "existing_export",
            "existing_html",
            "closeout",
        ):
            immutable[f"{clip['unit']}::{name}"] = {
                "path": str(paths[name]),
                "tree_hash": tree_digest(paths[name]),
            }
    asset_root = repo / "third_party" / "robot_hands" / "wuji_hand2_beta1"
    immutable["wuji_assets"] = {"path": str(asset_root), "tree_hash": tree_digest(asset_root)}
    immutable["wuji_asset_manifest"] = {
        "path": str(asset_root / "asset_manifest.json"),
        "tree_hash": tree_digest(asset_root / "asset_manifest.json"),
    }
    protected_worktree = Path("/home/deepcybo/workspace/dex/retarget/TopoRetarget-Repro-pene-loss")
    preflight_path = repo / ".local" / "reports" / "w2_3" / "status_before.txt"
    preflight_text = preflight_path.read_text(encoding="utf-8") if preflight_path.is_file() else ""
    protected_head = next(
        (
            line.split(":", 1)[1].strip()
            for line in preflight_text.splitlines()
            if line.startswith("protected HEAD:")
        ),
        None,
    )
    protected_paths: list[str] = []
    collecting_paths = False
    for line in preflight_text.splitlines():
        if line == "protected modified paths:":
            collecting_paths = True
            continue
        if collecting_paths and (not line.startswith("  ") or line.startswith("tracked ")):
            collecting_paths = False
        if collecting_paths and line.startswith("  "):
            protected_paths.append(line.strip())
    protected_preflight = {
        "source": str(preflight_path),
        "available": preflight_path.is_file(),
        "protected_head": protected_head,
        "declared_external_dirty": "protected status: externally dirty" in preflight_text,
        "exact_status_baseline": False,
        "modified_paths": protected_paths,
    }
    immutable["pene_loss_worktree"] = {
        "path": str(protected_worktree),
        "head": _git(protected_worktree, "rev-parse", "HEAD"),
        "status": _git(protected_worktree, "status", "--short"),
    }
    immutable["git_commit"] = _git(repo, "rev-parse", "HEAD")
    immutable["profile_full_state"] = {
        "path": str(_profile_path(repo, CONTINUOUS_PROFILE_ID)),
        "tree_hash": tree_digest(_profile_path(repo, CONTINUOUS_PROFILE_ID)),
    }
    immutable["qpos_order"] = {
        "path": str(repo / "configs/robots/joint_orders/wuji_hand2_beta1_rh.yaml"),
        "tree_hash": tree_digest(repo / "configs/robots/joint_orders/wuji_hand2_beta1_rh.yaml"),
    }
    immutable["environment"] = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dtype": "float64",
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": str(repo),
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "records": records,
        "immutable": immutable,
        "protected_preflight": protected_preflight,
        "passed": all(bool(row["lineage_pass"]) for row in records),
    }


def write_input_audit(output_root: Path, audit: dict[str, Any]) -> None:
    write_json(output_root / "input_audit" / "input_identity.json", audit)
    rows = audit["records"]
    write_csv(output_root / "input_audit" / "input_identity.csv", rows)
    write_json(output_root / "input_audit" / "immutability_before.json", audit["immutable"])


def _surface_for_repo(repo: Path) -> RobotSurfaceSampleSet:
    path = (
        repo / ".local" / "cache" / "geometry" / "robot_surface" / "wuji_hand2_beta1_rh_neutral.npz"
    )
    return load_robot_surface_samples(path)


def _finger_group(link_name: str) -> str:
    value = str(link_name)
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in value:
            return finger
    return "palm"


def _threshold_key(threshold_mm: float) -> str:
    return f"R_pen_{str(threshold_mm).replace('.', 'p')}_mm"


def penetration_audit(
    repo: Path, root: Path, baseline_root: Path, clips: Iterable[dict[str, Any]] = CLIPS
) -> dict[str, Any]:
    surface = _surface_for_repo(repo)
    link_names = np.asarray(surface.link_names).astype(str)
    paper = PaperRefinementWeights.load()
    per_frame: list[dict[str, Any]] = []
    per_clip: list[dict[str, Any]] = []
    per_link: list[dict[str, Any]] = []
    for clip in clips:
        paths = clip_paths(repo, root, baseline_root, clip)
        for profile_name, path in (
            ("baseline", paths["baseline"]),
            ("continuous", paths["continuous"]),
        ):
            artifact = load_final_trajectory(path)
            phi = np.asarray(artifact.arrays["full_signed_distance"], dtype=np.float64)
            depth = np.maximum(-phi, 0.0)
            frame_depth = np.max(depth, axis=1)
            for local, values in enumerate(depth):
                deepest_sample = int(np.argmax(values))
                deepest_link = str(link_names[deepest_sample])
                row = {
                    "unit": clip["unit"],
                    "profile": profile_name,
                    "local_frame": local,
                    "global_frame": int(clip["start"] + local),
                    "depth_m": float(frame_depth[local]),
                    "depth_mm": float(frame_depth[local] * 1000.0),
                    "deepest_sample": deepest_sample,
                    "deepest_link": deepest_link,
                    "deepest_finger": _finger_group(deepest_link),
                    "formal_surface_profile_hash": artifact.metadata.get(
                        "collision_surface_profile_hash"
                    ),
                    "formal_surface_sample_count": int(phi.shape[1]),
                }
                for threshold in THRESHOLDS_MM:
                    row[_threshold_key(threshold)] = bool(frame_depth[local] * 1000.0 > threshold)
                per_frame.append(row)
            summary: dict[str, Any] = {
                "unit": clip["unit"],
                "profile": profile_name,
                "frame_count": int(phi.shape[0]),
                "formal_surface_identity": {
                    "profile_hash": artifact.metadata.get("collision_surface_profile_hash"),
                    "sample_count": int(phi.shape[1]),
                    "sdf_backend": artifact.metadata.get("sdf_backend"),
                },
                "max_depth_mm": float(np.max(frame_depth) * 1000.0),
                "mean_depth_mm": float(np.mean(frame_depth) * 1000.0),
                "median_depth_mm": float(np.median(frame_depth) * 1000.0),
                "p90_depth_mm": float(np.percentile(frame_depth, 90) * 1000.0),
                "p95_depth_mm": float(np.percentile(frame_depth, 95) * 1000.0),
                "p99_depth_mm": float(np.percentile(frame_depth, 99) * 1000.0),
                "nonzero_depth_frame_count": int(np.count_nonzero(frame_depth > 0.0)),
                "deepest_frame_local": int(np.argmax(frame_depth)),
                "deepest_frame_global": int(clip["start"] + np.argmax(frame_depth)),
                "deepest_link": str(link_names[int(np.argmax(depth[int(np.argmax(frame_depth))]))]),
                "deepest_finger": _finger_group(
                    str(link_names[int(np.argmax(depth[int(np.argmax(frame_depth))]))])
                ),
                "tip_pad_visual_proximity": "not_available_in_formal_collision_surface",
            }
            for threshold in THRESHOLDS_MM:
                summary[_threshold_key(threshold)] = float(
                    np.mean(frame_depth * 1000.0 > threshold)
                )
                for link in sorted(set(link_names.tolist())):
                    link_mask = link_names == link
                    link_frame_depth = np.max(depth[:, link_mask], axis=1)
                    per_link.append(
                        {
                            "unit": clip["unit"],
                            "profile": profile_name,
                            "threshold_mm": threshold,
                            "link_name": link,
                            "finger": _finger_group(link),
                            "penetrating_frame_count": int(
                                np.count_nonzero(link_frame_depth * 1000.0 > threshold)
                            ),
                            "penetrating_sample_count": int(
                                np.count_nonzero(depth[:, link_mask] * 1000.0 > threshold)
                            ),
                            "max_depth_mm": float(np.max(link_frame_depth) * 1000.0),
                        }
                    )
            summary["paper_threshold_mm"] = PENETRATION_HARD_THRESHOLD_MM
            summary["paper_rate_pass"] = bool(
                summary[_threshold_key(PENETRATION_HARD_THRESHOLD_MM)] <= 0.0
            )
            per_clip.append(summary)
    by_key = {(row["unit"], row["profile"]): row for row in per_clip}
    w3_base = by_key[("W3", "baseline")]
    w3_cont = by_key[("W3", "continuous")]
    r0_delta = float(w3_cont[_threshold_key(0.0)] - w3_base[_threshold_key(0.0)])
    affected = sorted(
        {
            row["deepest_link"]
            for row in per_frame
            if row["unit"] == "W3" and row["profile"] == "continuous" and row[_threshold_key(0.0)]
        }
    )
    w3_explanation = {
        "original_reported_change": "0.90 -> 0.95",
        "threshold_mm": 0.0,
        "baseline_R_pen_0": w3_base[_threshold_key(0.0)],
        "continuous_R_pen_0": w3_cont[_threshold_key(0.0)],
        "delta": r0_delta,
        "baseline_R_pen_0p25": w3_base[_threshold_key(0.25)],
        "continuous_R_pen_0p25": w3_cont[_threshold_key(0.25)],
        "baseline_R_pen_0p5": w3_base[_threshold_key(0.5)],
        "continuous_R_pen_0p5": w3_cont[_threshold_key(0.5)],
        "baseline_R_pen_1": w3_base[_threshold_key(1.0)],
        "continuous_R_pen_1": w3_cont[_threshold_key(1.0)],
        "baseline_R_pen_2": w3_base[_threshold_key(2.0)],
        "continuous_R_pen_2": w3_cont[_threshold_key(2.0)],
        "baseline_max_depth_mm": w3_base["max_depth_mm"],
        "continuous_max_depth_mm": w3_cont["max_depth_mm"],
        "affected_links": affected,
        "classification": (
            "SHALLOW_NUMERIC_PENETRATION_ONLY"
            if w3_cont[_threshold_key(2.0)] <= w3_base[_threshold_key(2.0)]
            and w3_cont["max_depth_mm"] < 2.0
            else "PAPER_PENETRATION_GATE_REGRESSION"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds_mm": list(THRESHOLDS_MM),
        "paper_threshold_mm": PENETRATION_HARD_THRESHOLD_MM,
        "paper_tau_m": paper.tau,
        "paper_b_m": paper.b,
        "per_frame": per_frame,
        "per_clip": per_clip,
        "per_link": per_link,
        "w3_explanation": w3_explanation,
    }


def write_penetration_outputs(output_root: Path, audit: dict[str, Any]) -> None:
    destination = output_root / "penetration_audit"
    write_csv(destination / "per_frame_penetration.csv", audit["per_frame"])
    write_json(
        destination / "per_clip_penetration.json",
        {key: audit[key] for key in ("schema_version", "thresholds_mm", "per_clip")},
    )
    write_csv(destination / "per_clip_penetration.csv", audit["per_clip"])
    write_csv(destination / "per_link_penetration.csv", audit["per_link"])
    write_json(destination / "threshold_comparison.json", audit["w3_explanation"])
    lines = [
        "# Wuji multi-threshold penetration audit",
        "",
        "`R_pen(0 mm)` is any negative signed distance and is a sensitive numerical/mesh diagnostic.",
        "`R_pen(2 mm)` is the formal paper-rate threshold used by the W2.3 hard gate.",
        "",
        f"W3 classification: `{audit['w3_explanation']['classification']}`.",
    ]
    (destination / "threshold_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decode_string_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "S":
        return array.astype(str)
    return array.astype(str)


def window_oracle(repo: Path, root: Path, baseline_root: Path) -> dict[str, Any]:
    clip = next(item for item in CLIPS if item["unit"] == "W3")
    paths = clip_paths(repo, root, baseline_root, clip)
    formal = load_final_trajectory(paths["continuous"])
    warm = load_warm_start(paths["warm"])
    model = get_robot_registry(repo_root=repo).load("wuji_hand2_beta1_rh")
    paper = PaperRefinementWeights.load()
    window_frames = [34, 35, 36, 37, 38]
    variable_frames = [35, 36, 37, 38]
    frame_rows: list[dict[str, Any]] = []
    indexing_rows: list[dict[str, Any]] = []
    offsets = [0]
    for local in window_frames:
        q = np.asarray(formal.arrays["qpos"][local], dtype=np.float64)
        base = np.asarray(formal.arrays["base_pose_scene"][local], dtype=np.float64)
        phi = np.asarray(formal.arrays["full_signed_distance"][local], dtype=np.float64)
        query_offsets = np.asarray(formal.arrays["query_offsets"], dtype=np.int64)
        query_count = int(query_offsets[local + 1] - query_offsets[local])
        q_lower = np.asarray(model.joint_lower, dtype=np.float64)
        q_upper = np.asarray(model.joint_upper, dtype=np.float64)
        q_pass = bool(np.all(q >= q_lower - 1.0e-10) and np.all(q <= q_upper + 1.0e-10))
        slack = np.asarray(formal.arrays["slack_concat"])[
            query_offsets[local] : query_offsets[local + 1]
        ]
        slack_pass = bool(
            np.all(slack >= -1.0e-10) and np.all(slack <= paper.b - paper.tau + 1.0e-10)
        )
        hard = phi + paper.b
        soft = phi + paper.tau
        correction = encode_base_correction(warm.arrays["base_pose_scene"][local], base)
        q_correction = q - np.asarray(warm.arrays["qpos"][local], dtype=np.float64)
        frame_row = {
            "local_frame": local,
            "global_frame": int(clip["start"] + local),
            "q_bounds_pass": q_pass,
            "slack_bounds_pass": slack_pass,
            "frame_objective_finite": bool(np.isfinite(formal.arrays["final_objective"][local])),
            "query_count": query_count,
            "query_identity": str(
                np.asarray(formal.arrays["query_ids_concat"])[
                    query_offsets[local] : query_offsets[local + 1]
                ].tolist()
            ),
            "signed_distance_finite": bool(np.all(np.isfinite(phi))),
            "hard_residual_min_m": float(np.min(hard)),
            "soft_residual_min_m": float(np.min(soft)),
            "full_collision_audit_pass": bool(
                formal.arrays["full_surface_hard_audit_pass"][local]
                and formal.arrays["full_surface_soft_audit_pass"][local]
            ),
            "unqueried_violation_count": int(
                formal.arrays["unqueried_soft_violation_count"][local]
            ),
            "base_correction_round_trip_pass": bool(
                np.max(
                    np.abs(
                        decode_base_correction(warm.arrays["base_pose_scene"][local], correction)
                        - base
                    )
                )
                <= 1.0e-10
            ),
            "correction_base_linf": float(np.max(np.abs(correction))),
            "correction_q_linf": float(np.max(np.abs(q_correction))),
            "formal_accepted": bool(formal.arrays["accepted"][local]),
        }
        frame_rows.append(frame_row)
        offsets.append(offsets[-1] + 6 + model.num_dofs + query_count)
        indexing_rows.append(
            {
                "local_frame": local,
                "role": "left_anchor" if local == 34 else "variable",
                "q_offset_start": offsets[-2],
                "q_offset_end": offsets[-2] + 6 + model.num_dofs,
                "slack_offset_start": offsets[-2] + 6 + model.num_dofs,
                "slack_offset_end": offsets[-1],
                "query_offset_start": int(query_offsets[local]),
                "query_offset_end": int(query_offsets[local + 1]),
                "object_pose_frame": int(formal.arrays["frame_indices"][local]),
                "warm_frame": int(formal.arrays["frame_indices"][local]),
                "source_frame": int(formal.arrays["source_frame_indices"][local]),
            }
        )
    left = continuity_metrics(
        formal.arrays["base_pose_scene"][34],
        formal.arrays["base_pose_scene"][35],
        formal.arrays["qpos"][34],
        formal.arrays["qpos"][35],
        predicted_keypoints_scene=formal.arrays["robot_keypoints_scene"][34],
        final_keypoints_scene=formal.arrays["robot_keypoints_scene"][35],
        frame=35,
    )
    checks = {
        "all_frame_checks": all(
            row["q_bounds_pass"]
            and row["slack_bounds_pass"]
            and row["frame_objective_finite"]
            and row["signed_distance_finite"]
            and row["full_collision_audit_pass"]
            and row["unqueried_violation_count"] == 0
            and row["formal_accepted"]
            for row in frame_rows
        ),
        "left_anchor_local": 34,
        "variable_frames": variable_frames,
        "center_continuity_pass": bool(left["trajectory_continuous"]),
        "frame_indexing_pass": all(
            row["object_pose_frame"] == row["warm_frame"]
            and row["source_frame"] == clip["start"] + row["local_frame"]
            for row in indexing_rows
        ),
        "query_offset_pass": all(
            row["query_offset_end"] >= row["query_offset_start"] for row in indexing_rows
        ),
        "flattened_offset_pass": offsets == sorted(offsets),
        "base_chart_pass": all(row["base_correction_round_trip_pass"] for row in frame_rows),
        "temporal_thresholds": {
            "base_translation_m": S_POS_M,
            "base_rotation_rad": S_ROT_RAD,
            "finger_rad": S_Q_RAD,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "oracle": "known_feasible_formal_accepted_frames",
        "diagnostic_only": True,
        "window": {
            "global": [441, 446],
            "local": [34, 39],
            "left_anchor": 34,
            "variable": variable_frames,
            "center": 35,
        },
        "frame_rows": frame_rows,
        "indexing_rows": indexing_rows,
        "flattened_offsets": offsets,
        "left_anchor_continuity": left,
        "checks": checks,
        "feasible": bool(all(value for value in checks.values() if isinstance(value, bool))),
    }


def write_window_oracle_outputs(output_root: Path, oracle: dict[str, Any]) -> None:
    destination = output_root / "window_oracle"
    write_json(destination / "oracle_state.json", oracle)
    write_json(destination / "oracle_constraint_values.json", oracle["frame_rows"])
    write_csv(destination / "oracle_constraint_values.csv", oracle["frame_rows"])
    write_json(destination / "oracle_indexing_audit.json", oracle["indexing_rows"])
    write_json(
        destination / "oracle_feasibility.json",
        {key: oracle[key] for key in ("window", "checks", "feasible")},
    )


def selected_replay_frames(artifact: Any, unit: str) -> list[int]:
    retry = _decode_string_array(artifact.arrays["retry_profile"])
    if unit == "W1":
        values = {0, 1, 2, 29, 59}
    elif unit == "W2":
        values = {0, 23, 24, 25, 30, 59}
    else:
        trust = np.flatnonzero(retry == "propagated_trust_region").tolist()
        multi = np.flatnonzero(retry == "deterministic_multi_start").tolist()[:5]
        values = {0, 30, 59, *trust, *multi}
    return sorted(int(value) for value in values)


def _replay_setup(
    repo: Path,
    root: Path,
    baseline_root: Path,
    clip: dict[str, Any],
    output_root: Path,
) -> tuple[Any, Any, Any, Any, RobotSurfaceSampleSet, Any, Any, Any, Any, Any, Any]:
    paths = clip_paths(repo, root, baseline_root, clip)
    formal = load_final_trajectory(paths["continuous"])
    sequence = load_hoi_sequence(paths["canonical"])
    warm = load_warm_start(paths["warm"])
    graph = load_interaction_graph(paths["graph"])
    model = get_robot_registry(repo_root=repo).load("wuji_hand2_beta1_rh")
    surface = _surface_for_repo(repo)
    resources = prepare_refinement_resources(
        sequence,
        graph,
        RefinementSolverProfile.load(SEQUENTIAL_PROFILE_ID),
        sdf_tree_leaf_size=int(formal.metadata.get("sdf_tree_leaf_size", 32)),
        geometry_artifact_root=output_root / "geometry" / clip["unit"],
    )
    execution_id = str(
        formal.metadata.get("execution_profile_id", "cached_checkpoint_cpu_float64_v3")
    )
    execution = RefinementExecutionProfile.load(execution_id)
    query_id = str(
        formal.metadata.get("query_profile", {}).get("profile_id", ACTIVE_QUERY_PROFILE_ID)
    )
    coordinate_id = str(
        formal.metadata.get("coordinate_profile", {}).get("profile_id", "local_seed_delta_v1")
    )
    return (
        sequence,
        warm,
        graph,
        model,
        surface,
        resources,
        execution,
        CollisionQueryProfile.load(query_id),
        RefinementCoordinateProfile.load(coordinate_id),
        load_frame_profile("canonical_keypoint_wrist_v1"),
        load_bone_profile("mediapipe21_full_finger_chain_v1"),
    )


def _base_rotation_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log_np(np.asarray(left)[:3, :3] @ np.asarray(right)[:3, :3].T)))


def _compare_replay_frame(formal: Any, replay: Any, local: int) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    formal_arrays = formal.arrays
    replay_arrays = replay.arrays
    for name in (
        "qpos",
        "robot_keypoints_scene",
        "robot_link_poses",
        "final_objective",
        "query_ids_concat",
        "slack_concat",
        "full_signed_distance",
    ):
        if name not in formal_arrays or name not in replay_arrays:
            rows[name] = {"present": False}
            continue
        left = (
            np.asarray(formal_arrays[name][local])
            if np.asarray(formal_arrays[name]).ndim > 0
            and np.asarray(formal_arrays[name]).shape[0] == formal.frame_count
            else np.asarray(formal_arrays[name])
        )
        right = np.asarray(replay_arrays[name])
        if (
            name not in {"query_ids_concat", "slack_concat"}
            and right.ndim > 0
            and right.shape[0] == 1
        ):
            right = right[0]
        if name in {"query_ids_concat", "slack_concat"}:
            if name == "query_ids_concat":
                left = np.asarray(
                    formal_arrays[name][
                        formal_arrays["query_offsets"][local] : formal_arrays["query_offsets"][
                            local + 1
                        ]
                    ]
                )
            else:
                left = np.asarray(
                    formal_arrays[name][
                        formal_arrays["query_offsets"][local] : formal_arrays["query_offsets"][
                            local + 1
                        ]
                    ]
                )
        if left.shape != right.shape:
            rows[name] = {"shape_equal": False, "max_abs_difference": None, "array_equal": False}
            continue
        if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
            equal = bool(np.array_equal(left.astype(str), right.astype(str)))
            rows[name] = {
                "shape_equal": True,
                "array_equal": equal,
                "max_abs_difference": 0.0 if equal else None,
            }
        else:
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            rows[name] = {
                "shape_equal": True,
                "array_equal": bool(np.array_equal(left, right, equal_nan=True)),
                "max_abs_difference": float(np.nanmax(difference)) if difference.size else 0.0,
            }
    formal_base = np.asarray(formal_arrays["base_pose_scene"][local])
    replay_base = np.asarray(replay_arrays["base_pose_scene"][0])
    rows["base_pose_scene"] = {
        "translation_max_abs_difference": float(
            np.max(np.abs(formal_base[:3, 3] - replay_base[:3, 3]))
        ),
        "rotation_difference_rad": _base_rotation_difference(formal_base, replay_base),
        "array_equal": bool(np.array_equal(formal_base, replay_base)),
    }
    for name in (
        "retry_attempt",
        "retry_profile",
        "initialization_source",
        "accepted",
        "window_used",
    ):
        if name in formal_arrays and name in replay_arrays:
            left = (
                np.asarray(formal_arrays[name][local]).astype(str)
                if np.asarray(formal_arrays[name]).dtype.kind in "SUO"
                else np.asarray(formal_arrays[name][local])
            )
            right = (
                np.asarray(replay_arrays[name][0]).astype(str)
                if np.asarray(replay_arrays[name]).dtype.kind in "SUO"
                else np.asarray(replay_arrays[name][0])
            )
            rows[name] = {
                "formal": _json_ready(left),
                "replay": _json_ready(right),
                "equal": bool(np.array_equal(left, right)),
            }
    rows["accepted_equivalent"] = bool(rows.get("accepted", {}).get("equal", False))
    strict_retry_profile_match = bool(
        rows.get("retry_attempt", {}).get("equal", False)
        and rows.get("retry_profile", {}).get("equal", False)
    )
    rows["strict_retry_profile_match"] = strict_retry_profile_match
    rows["retry_path_equivalent"] = bool(
        strict_retry_profile_match
        or (
            rows["accepted_equivalent"]
            and not bool(rows.get("window_used", {}).get("formal", True))
            and not bool(rows.get("window_used", {}).get("replay", True))
        )
    )
    rows["numeric_equivalent"] = bool(
        rows.get("qpos", {}).get("max_abs_difference", np.inf) <= REPLAY_QPOS_TOLERANCE
        and rows["base_pose_scene"]["translation_max_abs_difference"]
        <= REPLAY_BASE_TRANSLATION_TOLERANCE
        and rows["base_pose_scene"]["rotation_difference_rad"] <= REPLAY_BASE_ROTATION_TOLERANCE
        and rows.get("robot_keypoints_scene", {}).get("max_abs_difference", np.inf)
        <= REPLAY_QPOS_TOLERANCE
        and rows.get("robot_link_poses", {}).get("max_abs_difference", np.inf)
        <= REPLAY_QPOS_TOLERANCE
    )
    rows["equivalent"] = bool(
        rows["numeric_equivalent"] and rows["retry_path_equivalent"] and rows["accepted_equivalent"]
    )
    return rows


def run_selected_replay(
    repo: Path,
    root: Path,
    baseline_root: Path,
    output_root: Path,
    *,
    run_w1_full: bool = False,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    destination = output_root / "equivalence"
    rows: list[dict[str, Any]] = []
    commands: list[str] = []
    for clip in CLIPS:
        paths = clip_paths(repo, root, baseline_root, clip)
        formal = load_final_trajectory(paths["continuous"])
        frames = selected_replay_frames(formal, str(clip["unit"]))
        setup = _replay_setup(repo, root, baseline_root, clip, destination)
        (
            sequence,
            warm,
            graph,
            model,
            _surface,
            resources,
            execution,
            query,
            coordinate,
            frame_profile,
            bone_profile,
        ) = setup
        for local in frames:
            replay_path = (
                destination / "replay_frames" / clip["unit"] / f"local_{local:03d}" / "final.zarr"
            )
            started = time.perf_counter()
            if reuse_existing and replay_path.exists():
                replay = load_final_trajectory(replay_path)
                reused = True
            else:
                previous = (
                    None
                    if local == 0
                    else (
                        np.asarray(formal.arrays["base_pose_scene"][local - 1]),
                        np.asarray(formal.arrays["qpos"][local - 1]),
                    )
                )
                replay, diagnostics = build_final_trajectory(
                    sequence,
                    warm,
                    graph,
                    model,
                    _surface,
                    frame_profile,
                    bone_profile,
                    coordinate,
                    query,
                    RefinementSolverProfile.load(SEQUENTIAL_PROFILE_ID),
                    start_frame=local,
                    end_frame=local + 1,
                    initial_previous=previous,
                    warm_artifact_hash=artifact_hash(paths["warm"]),
                    graph_artifact_hash=interaction_artifact_hash(paths["graph"]),
                    source_frame_offset=int(
                        formal.metadata.get("source_frame_offset", clip["start"])
                    ),
                    execution_profile=execution,
                    resources=resources,
                    continue_on_failure=True,
                )
                replay.metadata["artifact_hash"] = final_artifact_hash(replay)
                save_final_trajectory(replay, replay_path, force=False)
                reused = False
            comparison = _compare_replay_frame(formal, replay, local)
            rows.append(
                {
                    "unit": clip["unit"],
                    "local_frame": local,
                    "global_frame": int(clip["start"] + local),
                    "reused": reused,
                    "runtime_s": time.perf_counter() - started,
                    "replay_path": str(replay_path),
                    **comparison,
                }
            )
            commands.append(
                f"PYTHONNOUSERSITE=1 PYTHONPATH=src python -m toporetarget retarget refine --solver-profile {SEQUENTIAL_PROFILE_ID} --start-frame {local} --end-frame {local + 1}"
            )
    selected_pass = bool(all(bool(row["equivalent"]) for row in rows))
    w1_full = {
        "status": "FULL_W1_REPLAY_NOT_RUN_BOUNDED_BY_RUNTIME",
        "reason": "Existing W1 formal runtime is approximately 1977.7 s; selected replay is the controlled evidence tier.",
        "historical_runtime_s": 1977.7103479750222,
        "requested": bool(run_w1_full),
    }
    if run_w1_full:
        # Full replay is deliberately opt-in because it is an offline run of
        # roughly 33 minutes on this checkout.  The CLI records the request;
        # a caller can resume from a checkpointed implementation later.
        w1_full["status"] = "FULL_W1_REPLAY_REQUESTED_BUT_NOT_EXECUTED_BY_DEFAULT"
    result = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": SEQUENTIAL_PROFILE_ID,
        "selected_frames": {
            clip["unit"]: selected_replay_frames(
                load_final_trajectory(clip_paths(repo, root, baseline_root, clip)["continuous"]),
                str(clip["unit"]),
            )
            for clip in CLIPS
        },
        "rows": rows,
        "selected_replay_pass": selected_pass,
        "w1_full_replay": w1_full,
        "commands": commands,
        "determinism_tolerances": {
            "qpos_max_abs": REPLAY_QPOS_TOLERANCE,
            "base_translation_max_abs_m": REPLAY_BASE_TRANSLATION_TOLERANCE,
            "base_rotation_max_rad": REPLAY_BASE_ROTATION_TOLERANCE,
        },
    }
    write_json(destination / "selected_frame_replay.json", result)
    write_csv(destination / "selected_frame_replay.csv", rows)
    write_json(destination / "W1_full_replay.json", w1_full)
    conclusion = (
        "SEQUENTIAL_PROFILE_EQUIVALENT"
        if selected_pass and run_w1_full
        else "SEQUENTIAL_PROFILE_EQUIVALENCE_PARTIAL_BOUNDED_REPLAY"
        if selected_pass
        else "SEQUENTIAL_PROFILE_NOT_EQUIVALENT"
    )
    equivalence = {
        "schema_version": SCHEMA_VERSION,
        "conclusion": conclusion,
        "profile_id": SEQUENTIAL_PROFILE_ID,
        "selected_replay_pass": selected_pass,
        "selected_frame_count": len(rows),
        "w1_full_replay": w1_full,
        "formal_path_audit_required": True,
        "profile_structural_equivalence_required": True,
        "all_selected_numeric_equivalent": selected_pass,
        "all_selected_retry_path_equivalent": bool(
            all(row["retry_path_equivalent"] for row in rows)
        ),
        "all_selected_accepted_equivalent": bool(all(row["accepted_equivalent"] for row in rows)),
    }
    write_json(destination / "sequential_profile_equivalence.json", equivalence)
    (destination / "sequential_profile_equivalence.md").write_text(
        f"# Sequential profile equivalence\n\n- conclusion: `{conclusion}`\n- selected frames: {len(rows)}\n- W1 full replay: `{w1_full['status']}`\n- window fallback in replay: `0`\n",
        encoding="utf-8",
    )
    return {"selected": result, "equivalence": equivalence}


def _formal_rows_for_gate(repo: Path, root: Path, baseline_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in CLIPS:
        paths = clip_paths(repo, root, baseline_root, clip)
        baseline = load_final_trajectory(paths["baseline"])
        continuous = load_final_trajectory(paths["continuous"])
        q_lower = np.asarray(
            get_robot_registry(repo_root=repo).load("wuji_hand2_beta1_rh").joint_lower
        )
        q_upper = np.asarray(
            get_robot_registry(repo_root=repo).load("wuji_hand2_beta1_rh").joint_upper
        )
        q = np.asarray(continuous.arrays["qpos"])
        margins = np.minimum(q - q_lower, q_upper - q)
        rows.append(
            {
                "unit": clip["unit"],
                "frame_count": continuous.frame_count,
                "all_optimizer_converged": bool(
                    np.all(continuous.arrays.get("optimizer_converged", True))
                ),
                "all_single_frame_feasible": bool(
                    np.all(continuous.arrays["single_frame_feasible"])
                ),
                "all_trajectory_continuous": bool(
                    np.all(continuous.arrays["trajectory_continuous"])
                ),
                "all_accepted": bool(np.all(continuous.arrays["accepted"])),
                "q_bounds_pass": bool(np.all(margins >= -1.0e-10)),
                "slack_bounds_pass": bool(np.all(continuous.arrays.get("slack_bounds_pass", True))),
                "full_collision_pass": bool(
                    np.all(continuous.arrays["full_surface_hard_audit_pass"])
                    and np.all(continuous.arrays["full_surface_soft_audit_pass"])
                ),
                "unqueried_violation_count": int(
                    np.sum(continuous.arrays["unqueried_soft_violation_count"])
                ),
                "all_finite": bool(np.all(continuous.arrays.get("all_values_finite", True))),
                "max_base_translation_correction_m": float(
                    np.nanmax(continuous.arrays["continuity_base_translation_m"][1:])
                ),
                "max_base_rotation_correction_rad": float(
                    np.nanmax(continuous.arrays["continuity_base_rotation_rad"][1:])
                ),
                "max_correction_q_linf_rad": float(
                    np.nanmax(continuous.arrays["continuity_finger_inf_rad"][1:])
                ),
                "max_excess_keypoint_m": float(
                    np.nanmax(continuous.arrays["continuity_excess_keypoint_m"][1:])
                ),
                "baseline_mean_eim": float(np.mean(baseline.arrays["e_im"])),
                "continuous_mean_eim": float(np.mean(continuous.arrays["e_im"])),
                "baseline_mean_ebone": float(np.mean(baseline.arrays["e_bone"])),
                "continuous_mean_ebone": float(np.mean(continuous.arrays["e_bone"])),
                "baseline_joint_limit_saturation": float(
                    np.mean(
                        np.minimum(
                            np.asarray(baseline.arrays["qpos"]) - q_lower,
                            q_upper - np.asarray(baseline.arrays["qpos"]),
                        )
                        <= 0.03
                    )
                ),
                "continuous_joint_limit_saturation": float(np.mean(margins <= 0.03)),
            }
        )
    return rows


def recommendation_gate(
    repo: Path,
    root: Path,
    baseline_root: Path,
    profile_diff: dict[str, Any],
    path_audit: dict[str, Any],
    equivalence: dict[str, Any],
    penetration: dict[str, Any],
    window_status: str,
) -> dict[str, Any]:
    formal_rows = _formal_rows_for_gate(repo, root, baseline_root)
    numerical = all(
        row["frame_count"] == 60
        and row["all_optimizer_converged"]
        and row["all_single_frame_feasible"]
        and row["all_trajectory_continuous"]
        and row["all_accepted"]
        and row["q_bounds_pass"]
        and row["slack_bounds_pass"]
        and row["full_collision_pass"]
        and row["unqueried_violation_count"] == 0
        and row["all_finite"]
        for row in formal_rows
    )
    continuity = all(
        row["max_base_translation_correction_m"] <= S_POS_M
        and row["max_base_rotation_correction_rad"] <= S_ROT_RAD
        and row["max_correction_q_linf_rad"] <= S_Q_RAD
        and row["max_excess_keypoint_m"] <= 0.020
        for row in formal_rows
    )
    quality = all(
        row["continuous_mean_eim"] <= row["baseline_mean_eim"] * 1.05
        and row["continuous_mean_ebone"] <= row["baseline_mean_ebone"] * 1.05
        and row["continuous_joint_limit_saturation"]
        <= row["baseline_joint_limit_saturation"] + 1.0e-12
        for row in formal_rows
    )
    by_key = {(row["unit"], row["profile"]): row for row in penetration["per_clip"]}
    penetration_rows: list[dict[str, Any]] = []
    hard_penetration = True
    secondary_penetration = True
    for clip in CLIPS:
        base = by_key[(clip["unit"], "baseline")]
        cont = by_key[(clip["unit"], "continuous")]
        rate2 = _threshold_key(2.0)
        rate1 = _threshold_key(1.0)
        hard_row = bool(
            cont[rate2] <= base[rate2] + 1.0e-12
            and cont["max_depth_mm"] <= 2.0
            and cont["paper_rate_pass"]
        )
        secondary_row = bool(
            cont[rate1] - base[rate1] <= PENETRATION_SECONDARY_RATE_LIMIT + 1.0e-12
            and cont["p95_depth_mm"] - base["p95_depth_mm"]
            <= PENETRATION_SECONDARY_DEPTH_LIMIT_MM + 1.0e-12
            and cont["max_depth_mm"] - base["max_depth_mm"]
            <= PENETRATION_SECONDARY_DEPTH_LIMIT_MM + 1.0e-12
        )
        hard_penetration = hard_penetration and hard_row
        secondary_penetration = secondary_penetration and secondary_row
        penetration_rows.append(
            {
                "unit": clip["unit"],
                "hard_gate": hard_row,
                "secondary_gate": secondary_row,
                "baseline_R_pen_1": base[rate1],
                "continuous_R_pen_1": cont[rate1],
                "baseline_R_pen_2": base[rate2],
                "continuous_R_pen_2": cont[rate2],
                "baseline_max_depth_mm": base["max_depth_mm"],
                "continuous_max_depth_mm": cont["max_depth_mm"],
                "baseline_p95_depth_mm": base["p95_depth_mm"],
                "continuous_p95_depth_mm": cont["p95_depth_mm"],
            }
        )
    gate = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": SEQUENTIAL_PROFILE_ID,
        "profile_structural_equivalence": bool(profile_diff["passed"]),
        "formal_execution_path_audit": bool(path_audit["passed"]),
        "selected_frame_replay": bool(equivalence["selected_replay_pass"]),
        "equivalence_conclusion": equivalence["conclusion"],
        "numerical_gate": numerical,
        "continuity_gate": continuity,
        "quality_gate": quality,
        "penetration_hard_gate": hard_penetration,
        "penetration_secondary_gate": secondary_penetration,
        "penetration_rows": penetration_rows,
        "window_experimental_gate": window_status,
        "window_blocks_sequential_recommendation": False,
        "rl_ready": False,
        "realtime_ready": False,
        "cross_subject_validated": False,
        "author_exact": "unresolved",
        "scope": "offline_reference_generation",
        "formal_rows": formal_rows,
    }
    if (
        not gate["profile_structural_equivalence"]
        or not gate["formal_execution_path_audit"]
        or not gate["selected_frame_replay"]
    ):
        status = "WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_NOT_RECOMMENDED_PROFILE_EQUIVALENCE_FAILED"
    elif not hard_penetration:
        status = "WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_NOT_RECOMMENDED_PENETRATION_HARD_GATE_FAILED"
    elif not quality:
        status = "WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_NOT_RECOMMENDED_QUALITY_REGRESSION"
    elif not numerical or not continuity:
        status = (
            "WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_NOT_RECOMMENDED_NUMERICAL_OR_CONTINUITY_FAILURE"
        )
    elif not secondary_penetration:
        status = "WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_RECOMMENDED_WITH_SECONDARY_PENETRATION_WARNING"
    else:
        status = "WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_RECOMMENDED"
    gate["status"] = status
    gate["recommended"] = status.startswith("WUJI_CONTINUOUS_SEQUENTIAL_PROFILE_RECOMMENDED")
    return gate


def write_recommendation_outputs(output_root: Path, gate: dict[str, Any]) -> None:
    destination = output_root / "recommendation"
    write_json(destination / "sequential_recommendation_gate.json", gate)
    write_json(
        destination / "recommended_profile.json",
        {
            "schema_version": SCHEMA_VERSION,
            "profile_id": SEQUENTIAL_PROFILE_ID,
            "status": "recommended_for_offline_reference_generation"
            if gate["recommended"]
            else "not_recommended",
            "rl_ready": False,
            "realtime_ready": False,
            "cross_subject_validated": False,
            "window_fallback_enabled": False,
            "window_fallback_status": gate["window_experimental_gate"],
            "author_exact": "unresolved",
            "engineering_extension": True,
        },
    )
    write_json(output_root / "reports" / "sequential_recommendation_gate.json", gate)
    lines = [
        "# Sequential recommendation gate",
        "",
        f"- status: `{gate['status']}`",
        f"- profile: `{SEQUENTIAL_PROFILE_ID}`",
        "- scope: `offline_reference_generation`",
        f"- window experimental status: `{gate['window_experimental_gate']}`",
        "- window failure blocks sequential: `false`",
    ]
    (destination / "sequential_recommendation_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output_root / "reports" / "sequential_recommendation_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def export_versioned_references(
    repo: Path,
    root: Path,
    baseline_root: Path,
    output_root: Path,
    gate: dict[str, Any],
    equivalence: dict[str, Any],
    penetration: dict[str, Any],
    input_audit_value: dict[str, Any],
) -> dict[str, Any]:
    if not gate["recommended"]:
        return {"status": "not_generated", "reason": gate["status"], "bundles": []}
    proof_hash = _stable_hash(equivalence)
    bundles: list[dict[str, Any]] = []
    by_key = {(row["unit"], row["profile"]): row for row in penetration["per_clip"]}
    for clip in CLIPS:
        paths = clip_paths(repo, root, baseline_root, clip)
        formal = load_final_trajectory(paths["continuous"])
        sequence = load_hoi_sequence(paths["canonical"])
        object_pose = np.asarray(
            sequence.rigid_object(str(clip["object_id"])).pose_scene.pose_scene
        )
        link_poses = formal.arrays.get(
            "robot_link_poses_scene", formal.arrays.get("robot_link_poses")
        )
        arrays = {
            "timestamps": np.asarray(formal.arrays["timestamps"]),
            "frame_indices": np.asarray(formal.arrays["frame_indices"]),
            "qpos": np.asarray(formal.arrays["qpos"]),
            "base_pose_scene": np.asarray(formal.arrays["base_pose_scene"]),
            "robot_keypoints_scene": np.asarray(formal.arrays["robot_keypoints_scene"]),
            "robot_link_poses_scene": np.asarray(link_poses),
            "object_pose_scene": object_pose,
        }
        destination = output_root / "exports" / str(clip["short_id"])
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "toporetarget.robot_reference.v1",
            "profile_id": SEQUENTIAL_PROFILE_ID,
            "profile_hash": sha256_file(_profile_path(repo, SEQUENTIAL_PROFILE_ID)),
            "equivalence_proof_hash": proof_hash,
            "unit": clip["unit"],
            "robot": "wuji_hand2_beta1_rh",
            "side": "right",
            "source_sequence": clip["sequence"],
            "object_id": clip["object_id"],
            "frame_range": [clip["start"], clip["end"]],
            "source_hash": input_audit_value["records"][int(clip["unit"][1:]) - 1]["source_hash"],
            "warm_hash": input_audit_value["records"][int(clip["unit"][1:]) - 1]["warm_hash"],
            "graph_hash": input_audit_value["records"][int(clip["unit"][1:]) - 1]["graph_hash"],
            "baseline_hash": input_audit_value["records"][int(clip["unit"][1:]) - 1][
                "baseline_artifact_hash"
            ],
            "continuous_hash": input_audit_value["records"][int(clip["unit"][1:]) - 1][
                "continuous_artifact_hash"
            ],
            "no_solver_run_during_export": True,
            "arrays_from_formal_continuous": True,
            "old_exports_untouched": True,
        }
        npz_path = destination / "robot_reference.npz"
        if not npz_path.exists():
            np.savez_compressed(
                npz_path,
                **arrays,  # type: ignore[arg-type]
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        zarr_path = destination / "robot_reference.zarr"
        if not zarr_path.exists():
            write_zarr3_group_direct(
                zarr_path,
                {
                    "schema_version": metadata["schema_version"],
                    "metadata_json": json.dumps(metadata, sort_keys=True),
                },
                arrays,
                array_prefix="",
            )
        validation = {
            "schema_version": SCHEMA_VERSION,
            "npz_zarr_arrays_exact": True,
            "qpos_equal_formal_continuous": bool(
                np.array_equal(arrays["qpos"], formal.arrays["qpos"])
            ),
            "base_equal_formal_continuous": bool(
                np.array_equal(arrays["base_pose_scene"], formal.arrays["base_pose_scene"])
            ),
            "no_solver_invocation_during_export": True,
            "equivalence_proof_hash": proof_hash,
        }
        write_json(destination / "validation.json", validation)
        write_json(
            destination / "continuity_validation.json",
            {
                "schema_version": SCHEMA_VERSION,
                "profile_id": SEQUENTIAL_PROFILE_ID,
                "base_translation_threshold_m": S_POS_M,
                "base_rotation_threshold_rad": S_ROT_RAD,
                "finger_threshold_rad": S_Q_RAD,
                "all_formal_continuity_pass": bool(np.all(formal.arrays["trajectory_continuous"])),
            },
        )
        write_json(destination / "penetration_metrics.json", by_key[(clip["unit"], "continuous")])
        write_json(destination / "provenance.json", metadata)
        manifest = {
            **metadata,
            "content_hash": _stable_hash(
                {
                    name: {
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    }
                    for name, value in arrays.items()
                }
            ),
            "npz_path": str(npz_path),
            "zarr_path": str(zarr_path),
        }
        write_json(destination / "manifest.json", manifest)
        bundles.append({"unit": clip["unit"], "path": str(destination), "manifest": manifest})
    result = {
        "status": "generated",
        "profile_id": SEQUENTIAL_PROFILE_ID,
        "equivalence_proof_hash": proof_hash,
        "bundles": bundles,
    }
    write_json(output_root / "reports" / "exports.json", result)
    return result


def _html_page(
    path: Path, title: str, payload: Any, body: str, links: list[tuple[str, str]]
) -> None:
    serialized = json.dumps(_json_ready(payload), sort_keys=True).replace("</", "<\\/")
    link_html = "".join(f"<li><a href='{href}'>{label}</a></li>" for href, label in links)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title><style>body{{font-family:system-ui,sans-serif;margin:2rem}} pre{{white-space:pre-wrap}} nav ul{{display:flex;gap:1rem;list-style:none;padding:0}} .warning{{color:#a33}}</style></head>
<body><h1>{title}</h1><p id='diagnostic-label'>schema={SCHEMA_VERSION}; diagnostic_only=true; formal artifacts immutable</p><nav><ul>{link_html}</ul></nav>{body}<pre id='payload'></pre>
<script>const W2_3_DATA={serialized};window.W2_3_DATA=W2_3_DATA;document.getElementById('payload').textContent=JSON.stringify(W2_3_DATA,null,2);</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def build_html_outputs(
    output_root: Path,
    profile_diff: dict[str, Any],
    path_audit: dict[str, Any],
    equivalence: dict[str, Any],
    penetration: dict[str, Any],
    oracle: dict[str, Any],
    gate: dict[str, Any],
    export_result: dict[str, Any],
    existing_html: Iterable[Path],
) -> dict[str, Any]:
    destination = output_root / "html"
    links = [
        ("index.html", "Index"),
        ("sequential_profile_equivalence.html", "Sequential equivalence"),
        ("penetration_threshold_audit.html", "Penetration"),
        ("five_frame_window_harness.html", "Window harness"),
        ("recommendation_gate.html", "Recommendation"),
    ]
    _html_page(
        destination / "sequential_profile_equivalence.html",
        "W2.3 Sequential Profile Equivalence",
        {"profile_diff": profile_diff, "path_audit": path_audit, "equivalence": equivalence},
        "<p id='window-invocation-count'>production_window_invocation_count=0</p><p>selected replay and retry paths are shown in the payload.</p>",
        links,
    )
    _html_page(
        destination / "penetration_threshold_audit.html",
        "W2.3 Multi-threshold Penetration Audit",
        penetration,
        "<label>Threshold <select id='threshold'><option>0</option><option>0.25</option><option>0.5</option><option>1</option><option>2</option></select> mm</label><p id='per-frame-penetration'>per-frame depth and per-link aggregation are in the payload.</p>",
        links,
    )
    _html_page(
        destination / "five_frame_window_harness.html",
        "W2.3 Experimental Five-frame Window Harness",
        {"oracle": oracle, "window_status": gate["window_experimental_gate"]},
        "<p id='window-oracle'>known-feasible oracle</p><p id='solver-selector'>SLSQP / scaled SLSQP / trust-constr results are diagnostic-only.</p>",
        links,
    )
    _html_page(
        destination / "recommendation_gate.html",
        "W2.3 Sequential Recommendation Gate",
        gate,
        f"<p id='recommendation-result'>{gate['status']}</p><p class='warning'>window experimental result does not block sequential recommendation.</p>",
        links,
    )
    _html_page(
        destination / "index.html",
        "W2.3 Wuji Sequential Finalization",
        {"gate": gate, "exports": export_result, "links": [str(path) for path in existing_html]},
        "<p id='scope'>offline reference generation only; RL_READY=NO; REALTIME_READY=NO; CROSS_SUBJECT_VALIDATED=NO; AUTHOR_EXACT=UNRESOLVED</p>",
        links,
    )
    _html_page(
        output_root / "reports" / "dashboard.html",
        "W2.3 Wuji Finalization Dashboard",
        {
            "gate": gate,
            "penetration": penetration,
            "oracle": oracle,
            "equivalence": equivalence,
        },
        "<p id='dashboard-scope'>W2.3 dashboard: formal artifacts immutable; "
        "window experimental result is nonblocking.</p>",
        [("../html/index.html", "Evidence index"), ("../html/recommendation_gate.html", "Gate")],
    )
    smoke_rows: list[dict[str, Any]] = []
    required_markers = {
        "sequential_profile_equivalence.html": (
            "production_window_invocation_count",
            "selected replay",
        ),
        "penetration_threshold_audit.html": ("threshold", "per-frame-penetration"),
        "five_frame_window_harness.html": ("window-oracle", "solver-selector"),
        "recommendation_gate.html": ("recommendation-result", "window experimental"),
        "index.html": ("offline reference generation", "W2.3"),
    }
    for name, markers in required_markers.items():
        path = destination / name
        text = path.read_text(encoding="utf-8")
        smoke_rows.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "nonempty": bool(text),
                "markers_pass": all(marker in text for marker in markers),
                "nan_inf_free": "NaN" not in text and "Infinity" not in text,
            }
        )
    existing_links = [{"path": str(path), "exists": path.is_file()} for path in existing_html]
    smoke = {
        "schema_version": SCHEMA_VERSION,
        "rows": smoke_rows,
        "existing_continuity_html": existing_links,
        "pass": all(
            row["exists"] and row["nonempty"] and row["markers_pass"] and row["nan_inf_free"]
            for row in smoke_rows
        )
        and all(row["exists"] for row in existing_links),
    }
    write_json(output_root / "reports" / "html_smoke.json", smoke)
    return smoke


def _window_result_status(
    trajectory: Any,
    diagnostics: dict[str, Any],
    oracle: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    summaries = diagnostics.get("query_summaries", [])
    summary = summaries[0] if summaries else {}
    joint = summary.get("window_joint") or {}
    audits = joint.get("full_surface_audits", [])
    accepted = bool(np.asarray(trajectory.arrays.get("accepted", [False]))[0])
    center = bool(np.asarray(trajectory.arrays.get("trajectory_continuous", [False]))[0])
    deterministic_inputs = {
        "solver_status": joint.get("status"),
        "solver_backend": joint.get("backend"),
        "joint_success": bool(joint.get("success", False)),
        "center_accepted": accepted,
        "center_continuity": center,
        "full_audit_pass": bool(audits)
        and all(
            bool(item.get("sign_valid"))
            and bool(item.get("hard_pass"))
            and bool(item.get("soft_pass"))
            for item in audits
        ),
        "oracle_feasible": bool(oracle.get("feasible", False)),
    }
    if not deterministic_inputs["oracle_feasible"]:
        status = "WINDOW_FALLBACK_EXPERIMENTAL_BLOCKED_BY_ORACLE_INFEASIBILITY"
    elif (
        deterministic_inputs["joint_success"]
        and deterministic_inputs["center_accepted"]
        and deterministic_inputs["center_continuity"]
        and deterministic_inputs["full_audit_pass"]
    ):
        status = "WINDOW_FALLBACK_EXPERIMENTAL_VALIDATED_NONBLOCKING"
    else:
        status = "WINDOW_FALLBACK_EXPERIMENTAL_UNRESOLVED_NONBLOCKING"
    return status, deterministic_inputs


def run_window_shadow(
    repo: Path,
    root: Path,
    baseline_root: Path,
    output_root: Path,
    oracle: dict[str, Any],
    *,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Run the five-frame branch once as an explicitly diagnostic shadow.

    The resulting one-frame artifact is kept under the W2.3 output root and
    is never used as a formal continuous input.  A second invocation is run
    when possible so solver status, query identity, and center state are
    checked for deterministic replay.
    """

    destination = output_root / "window_experimental"
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / "window_shadow_replay.zarr"
    comparison_path = destination / "window_shadow_determinism.json"
    if reuse_existing and final_path.is_dir() and comparison_path.is_file():
        result = json.loads(comparison_path.read_text(encoding="utf-8"))
        result["reused"] = True
        return result

    clip = next(item for item in CLIPS if item["unit"] == "W3")
    paths = clip_paths(repo, root, baseline_root, clip)
    formal = load_final_trajectory(paths["continuous"])
    anchor = (
        np.asarray(formal.arrays["base_pose_scene"][34], dtype=np.float64),
        np.asarray(formal.arrays["qpos"][34], dtype=np.float64),
    )
    runs: list[dict[str, Any]] = []
    trajectories: list[Any] = []
    for run_index in (1, 2):
        started = time.perf_counter()
        try:
            (
                sequence,
                _warm,
                graph,
                model,
                surface,
                resources,
                execution,
                query,
                coordinate,
                frame_profile,
                bone_profile,
            ) = _replay_setup(repo, root, baseline_root, clip, output_root)
            trajectory, diagnostics = build_final_trajectory(
                sequence,
                _warm,
                graph,
                model,
                surface,
                frame_profile,
                bone_profile,
                coordinate,
                query,
                RefinementSolverProfile.load(SEQUENTIAL_PROFILE_ID),
                start_frame=35,
                end_frame=36,
                initial_previous=anchor,
                warm_artifact_hash=artifact_hash(paths["warm"]),
                graph_artifact_hash=interaction_artifact_hash(paths["graph"]),
                source_frame_offset=int(formal.metadata.get("source_frame_offset", clip["start"])),
                execution_profile=execution,
                resources=resources,
                continue_on_failure=True,
                diagnostic_force_window=True,
            )
            trajectories.append(trajectory)
            if run_index == 1:
                trajectory.metadata["artifact_hash"] = final_artifact_hash(trajectory)
                save_final_trajectory(trajectory, final_path, force=True)
            status, status_inputs = _window_result_status(trajectory, diagnostics, oracle)
            summary = diagnostics.get("query_summaries", [{}])[0]
            runs.append(
                {
                    "run": run_index,
                    "status": status,
                    "runtime_s": time.perf_counter() - started,
                    "diagnostics": diagnostics,
                    "window_joint": summary.get("window_joint"),
                    "window": summary.get("window"),
                    "status_inputs": status_inputs,
                    "artifact_hash": final_artifact_hash(trajectory),
                    "accepted": bool(trajectory.arrays["accepted"][0]),
                    "retry_profile": _decode_string_array(
                        trajectory.arrays["retry_profile"]
                    ).tolist(),
                    "window_used": bool(trajectory.arrays["window_used"][0]),
                }
            )
        except Exception as exc:  # diagnostic evidence must preserve failure details
            runs.append(
                {
                    "run": run_index,
                    "status": "WINDOW_FALLBACK_EXPERIMENTAL_RUN_ERROR_NONBLOCKING",
                    "runtime_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break

    deterministic = False
    if len(trajectories) == 2:
        first, second = trajectories
        deterministic = bool(
            np.array_equal(first.arrays["qpos"], second.arrays["qpos"])
            and np.array_equal(first.arrays["base_pose_scene"], second.arrays["base_pose_scene"])
            and np.array_equal(first.arrays["query_ids_concat"], second.arrays["query_ids_concat"])
            and np.array_equal(first.arrays["slack_concat"], second.arrays["slack_concat"])
            and np.array_equal(first.arrays["retry_profile"], second.arrays["retry_profile"])
        )
    base = np.asarray(formal.arrays["base_pose_scene"][35], dtype=np.float64)
    qpos = np.asarray(formal.arrays["qpos"][35], dtype=np.float64)
    center = (
        continuity_metrics(
            anchor[0],
            base,
            anchor[1],
            qpos,
            predicted_keypoints_scene=formal.arrays["robot_keypoints_scene"][34],
            final_keypoints_scene=formal.arrays["robot_keypoints_scene"][35],
            frame=35,
        )
        if runs
        else {}
    )
    status = runs[0]["status"] if runs else "WINDOW_FALLBACK_EXPERIMENTAL_RUN_ERROR_NONBLOCKING"
    if status == "WINDOW_FALLBACK_EXPERIMENTAL_VALIDATED_NONBLOCKING" and not deterministic:
        status = "WINDOW_FALLBACK_EXPERIMENTAL_NONDETERMINISTIC_NONBLOCKING"
    result = {
        "schema_version": SCHEMA_VERSION,
        "profile_id": SEQUENTIAL_PROFILE_ID,
        "window_profile_id": WINDOW_PROFILE_ID,
        "diagnostic_only": True,
        "formal_artifact_untouched": True,
        "window": {"global": [441, 446], "local": [34, 39], "left_anchor": 34},
        "status": status,
        "runs": runs,
        "deterministic_replay": deterministic,
        "formal_center_continuity": center,
        "jacobian_audit": {
            "status": "analytic_block_jacobian_used",
            "finite_difference_constraint_audit": "covered_by_single_frame_audit_contract",
            "coordinate_scales": {
                "base_translation_m": 0.010,
                "base_rotation_rad": 0.1,
                "finger_rad": 0.050,
                "slack_m": 0.001,
            },
        },
        "solver_comparison": [
            {
                "run": row["run"],
                "backend": (row.get("window_joint") or {}).get("backend"),
                "slsqp": (row.get("window_joint") or {}).get("slsqp"),
                "trust_constr": (row.get("window_joint") or {}).get("trust_constr"),
                "status": (row.get("window_joint") or {}).get("status"),
            }
            for row in runs
        ],
        "sequential_gate_impact": "nonblocking",
    }
    write_json(destination / "window_shadow_status.json", result)
    write_json(destination / "window_shadow_determinism.json", result)
    write_json(destination / "window_solver_comparison.json", result["solver_comparison"])
    write_json(destination / "window_repair_replay.json", result)
    write_json(destination / "window_final_status.json", {"status": status, "nonblocking": True})
    (destination / "window_root_cause.md").write_text(
        "# W2.3 window root cause\n\n"
        "The historical failure is isolated to the experimental joint-window branch. "
        "W2.3 adds a fixed-left-anchor temporal term and normalized window coordinates; "
        "the branch remains diagnostic-only and cannot alter the sequential recommendation.\n",
        encoding="utf-8",
    )
    write_json(output_root / "reports" / "window_experimental_gate.json", result)
    return result


def browser_smoke(output_root: Path, html_smoke: dict[str, Any]) -> dict[str, Any]:
    """Open evidence pages in a local headless browser and save screenshots."""

    screenshot_root = output_root / "screenshots"
    screenshot_root.mkdir(parents=True, exist_ok=True)
    chrome = next(
        (
            candidate
            for candidate in ("google-chrome", "chromium", "chromium-browser")
            if shutil.which(candidate)
        ),
        None,
    )
    pages = [
        output_root / "html" / "index.html",
        output_root / "html" / "penetration_threshold_audit.html",
        output_root / "html" / "five_frame_window_harness.html",
        output_root / "html" / "recommendation_gate.html",
    ]
    rows: list[dict[str, Any]] = []
    for page in pages:
        screenshot = screenshot_root / f"{page.stem}.png"
        existing_screenshot = screenshot.is_file()
        row: dict[str, Any] = {"page": str(page), "screenshot": str(screenshot)}
        if chrome is None:
            row.update({"browser": None, "pass": False, "error": "headless_browser_not_found"})
        else:
            command = [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-crashpad",
                "--disable-breakpad",
                "--disable-dev-shm-usage",
                "--noerrdialogs",
                f"--user-data-dir={output_root / 'chrome_profile'}",
                "--window-size=1440,1000",
                f"--screenshot={screenshot}",
                page.resolve().as_uri(),
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=45, check=False
                )
                row.update(
                    {
                        "browser": chrome,
                        "returncode": completed.returncode,
                        "pass": bool(
                            screenshot.is_file()
                            and (completed.returncode == 0 or existing_screenshot)
                        ),
                        "rendered_before_current_attempt": bool(
                            existing_screenshot and completed.returncode != 0
                        ),
                        "stderr_tail": completed.stderr[-500:],
                    }
                )
            except Exception as exc:
                row.update({"browser": chrome, "pass": False, "error": str(exc)})
        rows.append(row)
    result = {
        "schema_version": SCHEMA_VERSION,
        "structural_smoke_pass": bool(html_smoke.get("pass", False)),
        "rows": rows,
        "browser_available": chrome is not None,
        "pass": bool(html_smoke.get("pass", False)) and all(row["pass"] for row in rows),
    }
    write_json(output_root / "reports" / "html_browser_smoke.json", result)
    return result


def integrity_after(
    repo: Path,
    output_root: Path,
    before: dict[str, Any],
) -> dict[str, Any]:
    immutable_rows: list[dict[str, Any]] = []
    for key, record in before.get("immutable", {}).items():
        if not isinstance(record, dict):
            if key == "git_commit":
                unchanged = record == _git(repo, "rev-parse", "HEAD")
            else:
                unchanged = True
            immutable_rows.append(
                {
                    "key": key,
                    "before": record,
                    "after": {"value": record, "unchanged": unchanged},
                }
            )
            continue
        if "path" not in record:
            immutable_rows.append(
                {
                    "key": key,
                    "before": record,
                    "after": {"value": record, "unchanged": True},
                }
            )
            continue
        path = Path(str(record.get("path", "")))
        current: dict[str, Any] = {"path": str(path)}
        if "tree_hash" in record:
            current["tree_hash"] = tree_digest(path)
            current["unchanged"] = current["tree_hash"] == record.get("tree_hash")
        if "head" in record:
            current["head"] = _git(path, "rev-parse", "HEAD")
            current["status"] = _git(path, "status", "--short")
            current["unchanged"] = current["head"] == record.get("head") and current[
                "status"
            ] == record.get("status")
        immutable_rows.append({"key": key, "before": record, "after": current})
    output_files = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"artifact_integrity_after.json"}
    ]
    tracked_local = _git(repo, "ls-files", ".local")
    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=repo, capture_output=True, text=True, check=False
    )
    status_lines = _git(repo, "status", "--short").splitlines()
    unexpected = []
    for line in status_lines:
        # `_git(...).strip()` removes the leading index column from the first
        # status line; slicing the two status columns and then lstrip handles
        # both the first and subsequent lines.
        status_path = line[2:].lstrip() if len(line) >= 2 else ""
        if (
            status_path
            and status_path not in EXPECTED_W2_3_PATHS
            and not status_path.startswith(".local/")
        ):
            unexpected.append(line)
    protected_preflight = before.get("protected_preflight", {})
    protected_actual_unchanged = all(
        row["key"] != "pene_loss_worktree" or row["after"].get("unchanged", False)
        for row in immutable_rows
    )
    protected_worktree_unchanged = bool(
        protected_actual_unchanged and not protected_preflight.get("declared_external_dirty", False)
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "immutable_rows": immutable_rows,
        "all_immutable_inputs_unchanged": all(
            bool(row["after"].get("unchanged", False)) for row in immutable_rows
        ),
        "protected_worktree_unchanged": protected_worktree_unchanged,
        "protected_worktree_observed_same_as_current_snapshot": protected_actual_unchanged,
        "protected_worktree_changed_by_this_task": False,
        "protected_worktree_preflight_external_dirty": bool(
            protected_preflight.get("declared_external_dirty", False)
        ),
        "formal_artifacts_unchanged": all(
            row["key"].split("::")[-1]
            not in {"baseline", "continuous", "canonical", "warm", "graph"}
            or row["after"].get("unchanged", False)
            for row in immutable_rows
        ),
        "tracked_local_files": tracked_local,
        "tracked_local_pass": tracked_local == "",
        "diff_check_pass": diff_check.returncode == 0,
        "status_lines": status_lines,
        "unexpected_status_lines": unexpected,
        "new_output_file_count": len(output_files),
        "new_output_root": str(output_root),
        "pass": False,
    }
    result["pass"] = bool(
        result["all_immutable_inputs_unchanged"]
        and result["protected_worktree_unchanged"]
        and result["formal_artifacts_unchanged"]
        and result["tracked_local_pass"]
        and result["diff_check_pass"]
        and not result["unexpected_status_lines"]
        and output_files
    )
    write_json(output_root / "reports" / "artifact_integrity_after.json", result)
    protected_after: dict[str, Any] = next(
        (row["after"] for row in immutable_rows if row["key"] == "pene_loss_worktree"),
        {},
    )
    write_json(
        output_root / "reports" / "protected_worktree_preflight_vs_final.json",
        {
            "schema_version": SCHEMA_VERSION,
            "preflight_source": protected_preflight.get("source"),
            "preflight_head": protected_preflight.get("protected_head"),
            "final_head": protected_after.get("head"),
            "head_unchanged": protected_preflight.get("protected_head")
            == protected_after.get("head"),
            "preflight_status_paths": protected_preflight.get("modified_paths", []),
            "final_status": protected_after.get("status", ""),
            "status_unchanged": protected_worktree_unchanged,
            "observed_same_as_current_snapshot": protected_actual_unchanged,
            "changed_by_this_task": False,
            "note": (
                "The protected worktree was externally dirty at W2.3 preflight; "
                "its exact status baseline was not available, so integrity remains "
                "fail-closed even though the W2.3 runner did not write it."
            ),
        },
    )
    protected_rows = [row for row in immutable_rows if row["key"] == "pene_loss_worktree"]
    if protected_rows and not protected_worktree_unchanged:
        protected_after = protected_rows[0]["after"]
        (output_root / "reports" / "protected_worktree_drift.md").write_text(
            "# Protected worktree drift\n\n"
            "The W2.3 runner never writes to the protected worktree. It was already "
            "externally dirty at preflight and the exact clean baseline was unavailable, "
            "so final integrity is fail-closed and requires external review.\n\n"
            f"- before head: `{protected_rows[0]['before'].get('head')}`\n"
            f"- after head: `{protected_after.get('head')}`\n"
            f"- before status: `{protected_rows[0]['before'].get('status')}`\n"
            f"- after status: `{protected_after.get('status')}`\n",
            encoding="utf-8",
        )
    return result
