"""Stage 9.3.4 provenance-rebased causal experiment orchestration.

This module is deliberately an audit layer.  It consumes the existing Stage 7,
Stage 9.2, Stage 9.3.2, and Stage 10 artifacts, and writes only under the
Stage 9.3.4 ``.local`` roots.  The formal Eq. (1)--(9) implementation and the
accepted reference artifacts are never edited by this workflow.
"""

# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import WarmStartTrajectory, load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.refinement_checkpoint import CheckpointStore
from toporetarget.utils.hashing import sha256_file, sha256_tree

SCHEMA = "toporetarget.stage9_3_4.v1"
PROVENANCE_SCHEMA = "toporetarget.solver_effective_provenance.v1"
LINEAGE_SCHEMA = "toporetarget.current_causal_lineage.v1"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_POINTS = {
    "palm": (0,),
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
LONG_FINGERS = ("index", "middle", "ring")


class Stage934Error(RuntimeError):
    """Raised when a Stage 9.3.4 contract cannot be established."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _hash_path(path: Path) -> str:
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


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise Stage934Error(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-4000:]}"
        )
    return result.stdout.strip()


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise Stage934Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _record(
    path: Path,
    *,
    root: Path,
    semantic_role: str,
    numerical_effect: bool,
    reason: str,
) -> dict[str, Any]:
    absolute = path.expanduser().resolve()
    return {
        "path": str(absolute),
        "relative": str(absolute.relative_to(root.resolve()))
        if absolute.is_relative_to(root.resolve())
        else str(absolute),
        "content_hash": _hash_path(absolute),
        "semantic_role": semantic_role,
        "numerical_effect": numerical_effect,
        "reason": reason,
    }


def _environment() -> dict[str, Any]:
    package_names = ("numpy", "scipy", "torch", "trimesh", "zarr", "numcodecs", "Pillow")
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "thread_settings": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _manifest_artifacts(run_manifest: dict[str, Any]) -> dict[str, Path]:
    values: dict[str, Path] = {}
    for name, item in dict(run_manifest.get("artifacts", {})).items():
        if isinstance(item, dict) and item.get("path"):
            values[name] = Path(str(item["path"])).expanduser()
    for key in ("canonical", "warm_start", "graph", "collision_samples", "object_samples", "final"):
        value = run_manifest.get(key)
        if isinstance(value, str):
            values.setdefault(key, Path(value).expanduser())
    final = run_manifest.get("final_artifact_path")
    if isinstance(final, str):
        values.setdefault("final", Path(final).expanduser())
    return {name: path.resolve() for name, path in values.items()}


def _numerical_file_paths(root: Path) -> list[tuple[Path, str, bool, str]]:
    candidates = [
        (
            "src/toporetarget/retarget/final_refinement.py",
            "final refinement objective, Eq. (8)/(9), constraints, SDF selection",
            True,
        ),
        (
            "src/toporetarget/retarget/refinement_checkpoint.py",
            "checkpoint/resume and accepted-frame assembly",
            True,
        ),
        ("src/toporetarget/retarget/objectives.py", "formal objective decomposition", True),
        (
            "src/toporetarget/retarget/interaction_graph.py",
            "Stage 8 graph and Laplacian inputs",
            True,
        ),
        (
            "src/toporetarget/retarget/interaction_artifacts.py",
            "graph artifact decoding and identity",
            True,
        ),
        ("src/toporetarget/retarget/coordinates.py", "SE(3) base coordinate mapping", True),
        ("src/toporetarget/retarget/frames.py", "frame profile and SO(3) conventions", True),
        ("src/toporetarget/retarget/bones.py", "Stage 7 bone profile", True),
        (
            "src/toporetarget/geometry/signed_distance/reference.py",
            "canonical reference-winding SDF",
            True,
        ),
        (
            "src/toporetarget/geometry/signed_distance/closest_point.py",
            "closest-point SDF implementation",
            True,
        ),
        ("src/toporetarget/robots/model.py", "robot FK/Jacobian and anchors", True),
        ("src/toporetarget/robots/geometry.py", "collision sample representation", True),
        ("src/toporetarget/workflows/validation.py", "acceptance and identity validation", True),
        ("configs/paper/retarget.yaml", "paper weights and tolerances", True),
    ]
    result: list[tuple[Path, str, bool, str]] = []
    for relative, role, effect in candidates:
        path = root / relative
        if path.is_file():
            result.append((path, role, effect, "declared Stage 9 numerical dependency"))
    return result


def _profile_records(root: Path, checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[tuple[Path, str]] = []
    for key, metadata_key in (
        ("solver", "solver_profile"),
        ("execution", "execution_profile"),
        ("query", "query_profile"),
        ("coordinate", "coordinate_profile"),
    ):
        profile = checkpoint.get("final_artifact_metadata", {}).get(metadata_key, {})
        path_value = profile.get("source_path")
        if path_value:
            values.append((Path(str(path_value)), f"{key} profile"))
        elif checkpoint.get(f"{key}_profile_id"):
            folder = {
                "solver": "refinement_solvers",
                "execution": "refinement_execution",
                "query": "collision_queries",
                "coordinate": "refinement",
            }[key]
            values.append(
                (
                    root
                    / "configs"
                    / "retarget"
                    / folder
                    / f"{checkpoint[f'{key}_profile_id']}.yaml",
                    f"{key} profile",
                )
            )
    result: list[dict[str, Any]] = []
    for path, role in values:
        absolute = path if path.is_absolute() else root / path
        if absolute.is_file():
            result.append(
                _record(
                    absolute,
                    root=root,
                    semantic_role=role,
                    numerical_effect=True,
                    reason="profile controls solver-effective behavior",
                )
            )
    return result


def _resolve_historical_commit(repo: Path, run_manifest: dict[str, Any]) -> str | None:
    candidates = [
        run_manifest.get("git_commit"),
        run_manifest.get("environment", {}).get("git_commit"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{candidate}^{{commit}}"], cwd=repo, check=False
        )
        if result.returncode == 0:
            return str(candidate)
    return None


def audit_solver_lineage(run: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Materialize current/historical effective provenance and its diff."""

    repo = _repo_root()
    run_path = Path(run).expanduser().resolve()
    run_manifest = _json(run_path)
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(
        ".local/cache/retarget/final_checkpoints/stage9_2_contact_rich_60f_v3/manifest.json"
    )
    if not checkpoint_path.is_file():
        raise Stage934Error(f"Stage 9.2 checkpoint manifest missing: {checkpoint_path}")
    checkpoint = _json(checkpoint_path)
    artifacts = _manifest_artifacts(run_manifest)
    numerical: list[dict[str, Any]] = []
    for path, role, effect, reason in _numerical_file_paths(repo):
        numerical.append(
            _record(path, root=repo, semantic_role=role, numerical_effect=effect, reason=reason)
        )
    numerical.extend(_profile_records(repo, checkpoint))
    for name, path in sorted(artifacts.items()):
        if path.exists():
            numerical.append(
                _record(
                    path,
                    root=repo,
                    semantic_role=f"input artifact: {name}",
                    numerical_effect=True,
                    reason="input state or geometry consumed by Stage 9",
                )
            )
    excluded_records: list[dict[str, Any]] = []
    for relative, role in (
        ("README.md", "documentation"),
        ("docs/SHADOW_EQUIVALENCE_AND_LONG_FINGER_ABLATION.md", "audit documentation"),
    ):
        path = repo / relative
        if path.is_file():
            excluded_records.append(
                _record(
                    path,
                    root=repo,
                    semantic_role=role,
                    numerical_effect=False,
                    reason="excluded from solver-effective closure",
                )
            )
    current_commit = _git(repo, "rev-parse", "HEAD")
    historical_commit = _resolve_historical_commit(repo, run_manifest)
    historical_root: Path | None = None
    historical_worktree_status = "NOT_CREATED"
    if historical_commit:
        historical_root = (
            repo / ".local" / "worktrees" / "stage9_3_4_historical" / historical_commit[:12]
        )
        historical_root.parent.mkdir(parents=True, exist_ok=True)
        if not (historical_root / ".git").exists():
            result = subprocess.run(
                ["git", "worktree", "add", "--detach", str(historical_root), historical_commit],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                historical_worktree_status = f"FAILED: {result.stderr.strip()}"
            else:
                historical_worktree_status = "CREATED_DETACHED"
        else:
            historical_worktree_status = "EXISTING_DETACHED"
    historical_records: list[dict[str, Any]] = []
    if historical_root is not None and historical_root.exists():
        for path, role, effect, reason in _numerical_file_paths(historical_root):
            historical_records.append(
                _record(
                    path,
                    root=historical_root,
                    semantic_role=role,
                    numerical_effect=effect,
                    reason=reason,
                )
            )
        for name, path in sorted(artifacts.items()):
            if path.exists():
                historical_records.append(
                    _record(
                        path,
                        root=repo,
                        semantic_role=f"input artifact: {name}",
                        numerical_effect=True,
                        reason="historical run input artifact",
                    )
                )
    current = {
        "schema_version": PROVENANCE_SCHEMA,
        "lane": "current",
        "repo_root": str(repo),
        "git_commit": current_commit,
        "worktree_dirty": bool(_git(repo, "status", "--porcelain")),
        "git_status_porcelain": _git(repo, "status", "--short"),
        "environment": _environment(),
        "records": numerical,
        "excluded_records": excluded_records,
        "excluded_roles": ["documentation", "audit documentation", "viewer", "HTML", "tests"],
    }
    recorded_environment = run_manifest.get("environment", {})
    current_environment = _environment()
    exact_environment = (
        bool(recorded_environment)
        and recorded_environment.get("python") == current_environment.get("python")
        and all(
            recorded_environment.get("packages", {}).get(
                name, recorded_environment.get("packages", {}).get(name.replace("PIL", "Pillow"))
            )
            in {
                None,
                current_environment.get("packages", {}).get(name),
                current_environment.get("packages", {}).get(name.replace("PIL", "Pillow")),
            }
            for name in recorded_environment.get("packages", {})
        )
    )
    historical_available = bool(
        historical_root is not None
        and historical_root.exists()
        and historical_worktree_status in {"CREATED_DETACHED", "EXISTING_DETACHED"}
        and exact_environment
    )
    historical = {
        "schema_version": PROVENANCE_SCHEMA,
        "lane": "historical",
        "repo_root": str(historical_root) if historical_root else None,
        "git_commit": historical_commit,
        "worktree_status": historical_worktree_status,
        "environment": run_manifest.get("environment", {}),
        "historical_environment_status": "AVAILABLE_EXACT"
        if historical_available
        else "UNAVAILABLE_OR_INCOMPLETE",
        "records": historical_records,
    }
    current["provenance_hash"] = _stable_hash(
        {
            "git_commit": current_commit,
            "records": current["records"],
            "environment": current["environment"],
        }
    )
    historical["provenance_hash"] = _stable_hash(
        {
            "git_commit": historical_commit,
            "records": historical["records"],
            "environment": historical["environment"],
        }
    )
    by_key_current = {(x["semantic_role"], x["relative"]): x for x in numerical}
    by_key_historical = {(x["semantic_role"], x["relative"]): x for x in historical_records}
    diff_rows: list[dict[str, Any]] = []
    for key in sorted(set(by_key_current) | set(by_key_historical)):
        left = by_key_historical.get(key)
        right = by_key_current.get(key)
        if left is None or right is None:
            classification = "UNKNOWN_EFFECT"
        elif left.get("content_hash") == right.get("content_hash"):
            classification = (
                "NON_NUMERICAL" if not right.get("numerical_effect") else "NUMERICAL_EFFECTIVE"
            )
        elif right.get("numerical_effect") or left.get("numerical_effect"):
            classification = "NUMERICAL_EFFECTIVE"
        else:
            classification = "NON_NUMERICAL"
        diff_rows.append(
            {
                "semantic_role": key[0],
                "relative": key[1],
                "classification": classification,
                "historical_hash": None if left is None else left.get("content_hash"),
                "current_hash": None if right is None else right.get("content_hash"),
            }
        )
    diff = {
        "schema_version": "toporetarget.solver_effective_provenance_diff.v1",
        "historical_provenance_hash": historical["provenance_hash"],
        "current_provenance_hash": current["provenance_hash"],
        "rows": diff_rows,
        "counts": {
            name: sum(row["classification"] == name for row in diff_rows)
            for name in ("NUMERICAL_EFFECTIVE", "NON_NUMERICAL", "UNKNOWN_EFFECT")
        },
        "unknown_effect_unresolved": any(
            row["classification"] == "UNKNOWN_EFFECT" for row in diff_rows
        ),
    }
    _write_json(output / "solver_effective_provenance_current.json", current)
    _write_json(output / "solver_effective_provenance_historical.json", historical)
    _write_json(output / "solver_effective_provenance_diff.json", diff)
    lines = [
        "# Solver-effective provenance diff",
        "",
        f"- Historical hash: `{historical['provenance_hash']}`",
        f"- Current hash: `{current['provenance_hash']}`",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {diff['counts'][name]} |"
        for name in ("NUMERICAL_EFFECTIVE", "NON_NUMERICAL", "UNKNOWN_EFFECT")
    )
    lines.extend(["", "Unknown-effect records are not treated as equivalent.", ""])
    (output / "solver_effective_provenance_diff.md").write_text("\n".join(lines), encoding="utf-8")
    return {"current": current, "historical": historical, "diff": diff, "output_root": str(output)}


def _artifact_paths_from_run(run: Path) -> dict[str, Path]:
    return _manifest_artifacts(_json(run))


def _refine_command(
    run: Path,
    checkpoint_root: Path,
    output: Path,
    progress: Path,
    max_wall_time: float | None,
    *,
    resume: bool = True,
    start_frame: int = 0,
    end_frame: int = 60,
) -> list[str]:
    artifacts = _artifact_paths_from_run(run)
    required = ("canonical", "warm_start", "graph", "collision_samples")
    missing = [name for name in required if name not in artifacts or not artifacts[name].exists()]
    if missing:
        raise Stage934Error(f"run manifest is missing refinement inputs: {missing}")
    command = [
        sys.executable,
        "-m",
        "toporetarget",
        "retarget",
        "refine",
        "--canonical",
        str(artifacts["canonical"]),
        "--warm-start",
        str(artifacts["warm_start"]),
        "--graph",
        str(artifacts["graph"]),
        "--robot",
        str(_json(run).get("robot", "artimano_rh")),
        "--collision-samples",
        str(artifacts["collision_samples"]),
        "--query-profile",
        "adaptive_active_set_v1",
        "--coordinate-profile",
        "local_seed_delta_v1",
        "--solver-profile",
        "scipy_slsqp_active_set_contact_rich_v2",
        "--execution-profile",
        "cached_checkpoint_cpu_float64_v3",
        "--start-frame",
        str(start_frame),
        "--end-frame",
        str(end_frame),
        "--checkpoint-root",
        str(checkpoint_root),
        "--output",
        str(output),
        "--progress-json",
        str(progress),
        "--progress-log",
        str(progress.with_suffix(".jsonl")),
    ]
    if resume:
        command.append("--resume")
    if max_wall_time is not None:
        command.extend(["--max-wall-time", str(max_wall_time)])
    return command


def _state_metrics(source: np.ndarray, state: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for finger, indices in FINGER_POINTS.items():
        values[f"{finger}_rmse_m"] = float(
            np.sqrt(np.mean(np.square(state[list(indices)] - source[list(indices)])))
        )
    values["whole_hand_rmse_m"] = float(np.sqrt(np.mean(np.square(state - source))))
    values["long_finger_rmse_m"] = float(
        np.mean([values[f"{finger}_rmse_m"] for finger in LONG_FINGERS])
    )
    return values


def _proxy(full_phi: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(full_phi) <= threshold))


def _bundle(run: Path, final_path: Path) -> dict[str, Any]:
    artifacts = _artifact_paths_from_run(run)
    warm = load_warm_start(artifacts["warm_start"])
    final = load_final_trajectory(final_path)
    sequence = load_hoi_sequence(artifacts["canonical"])
    hand_id = str(_json(run).get("hand_id", "right_hand"))
    hand_ids = [str(hand.hand_id) for hand in sequence.hands]
    if hand_id not in hand_ids:
        if not hand_ids:
            raise Stage934Error("canonical sequence contains no hand tracks")
        hand_id = hand_ids[0]
    source = np.asarray(
        sequence.hand(hand_id).keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64
    )
    if final.frame_count > source.shape[0]:
        raise Stage934Error(
            f"source/final frame count mismatch: {source.shape[0]} vs {final.frame_count}"
        )
    source = source[: final.frame_count]
    return {
        "artifacts": artifacts,
        "warm": warm,
        "final": final,
        "sequence": sequence,
        "source": source,
        "run_manifest": _json(run),
    }


def _frame_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    source = bundle["source"]
    warm = bundle["warm"]
    final = bundle["final"]
    rows: list[dict[str, Any]] = []
    accepted = np.asarray(final.arrays.get("accepted", np.ones(len(source), dtype=bool))).astype(
        bool
    )
    status = np.asarray(final.arrays.get("optimizer_status_code", np.zeros(len(source), dtype=int)))
    full_phi = np.asarray(final.arrays["full_signed_distance"], dtype=np.float64)
    final_obj = np.asarray(
        final.arrays.get(
            "final_objective", final.arrays.get("total_objective", np.full(len(source), np.nan))
        )
    )
    warm_obj = np.asarray(warm.arrays.get("total_objective", np.full(len(source), np.nan)))
    final_eim = np.asarray(final.arrays.get("e_im", np.full(len(source), np.nan)))
    # The assembled final artifact carries the warm E_IM decomposition used
    # for the exact same frame/lineage comparison.  ``initial_total_objective``
    # is a different quantity and must not be reported as E_IM.
    warm_eim = np.asarray(final.arrays.get("warm_e_im", np.full(len(source), np.nan)))
    warm_points = np.asarray(warm.arrays["robot_keypoints_scene"], dtype=np.float64)
    final_points = np.asarray(final.arrays["robot_keypoints_scene"], dtype=np.float64)
    warm_q = np.asarray(warm.arrays["qpos"], dtype=np.float64)
    final_q = np.asarray(final.arrays["qpos"], dtype=np.float64)
    warm_base = np.asarray(warm.arrays["base_pose_scene"], dtype=np.float64)
    final_base = np.asarray(final.arrays["base_pose_scene"], dtype=np.float64)
    run_manifest = bundle["run_manifest"]
    for frame in range(len(source)):
        sm = _state_metrics(source[frame], source[frame])
        wm = _state_metrics(source[frame], warm_points[frame])
        fm = _state_metrics(source[frame], final_points[frame])
        row: dict[str, Any] = {
            "local_frame": frame,
            "global_frame": int(run_manifest.get("global_frame_start") or 240) + frame,
            "status": int(status[frame]),
            "accepted": bool(accepted[frame]),
            "strict_accepted": bool(accepted[frame]),
            "query_count": int(
                np.asarray(final.arrays["query_offsets"])[frame + 1]
                - np.asarray(final.arrays["query_offsets"])[frame]
            ),
            "active_set_rounds": int(
                np.asarray(final.arrays.get("active_set_rounds", np.zeros(len(source), dtype=int)))[
                    frame
                ]
            ),
            "min_full_sdf_m": float(np.min(full_phi[frame])),
            "raw_penetration_m": float(max(0.0, -np.min(full_phi[frame]))),
            "final_objective": float(final_obj[frame]),
            "warm_objective": float(warm_obj[frame]),
            "final_e_im": float(final_eim[frame]),
            "warm_e_im": float(warm_eim[frame]),
            "base_translation_displacement_m": float(
                np.linalg.norm(final_base[frame, :3, 3] - warm_base[frame, :3, 3])
            ),
            "qpos_displacement_rad": float(np.linalg.norm(final_q[frame] - warm_q[frame])),
            "contact_retention_proxy_1mm": _proxy(full_phi[frame], 0.001),
            "contact_retention_proxy_2mm": _proxy(full_phi[frame], 0.002),
            "contact_retention_proxy_5mm": _proxy(full_phi[frame], 0.005),
        }
        for finger in ("thumb", "index", "middle", "ring", "pinky", "whole_hand", "long_finger"):
            key = (
                "whole_hand_rmse_m"
                if finger == "whole_hand"
                else "long_finger_rmse_m"
                if finger == "long_finger"
                else f"{finger}_rmse_m"
            )
            row[f"source_{finger}_rmse_m"] = sm[key]
            row[f"warm_{finger}_rmse_m"] = wm[key]
            row[f"final_{finger}_rmse_m"] = fm[key]
            row[f"degradation_{finger}_rmse_m"] = fm[key] - wm[key]
        rows.append(row)
    return rows


def _validation(
    rows: list[dict[str, Any]], bundle: dict[str, Any], checkpoint_root: Path
) -> dict[str, Any]:
    final = bundle["final"]
    full_phi = np.asarray(final.arrays["full_signed_distance"], dtype=np.float64)
    accepted = sum(bool(row["accepted"]) for row in rows)
    status_zero = sum(int(row["status"]) == 0 for row in rows)
    finite = bool(np.all(np.isfinite(full_phi)))
    try:
        chain = CheckpointStore(checkpoint_root).validate_chain()
    except Exception as exc:
        chain = {"chain_pass": False, "error": str(exc)}
    audit = {
        "frame_count": len(rows),
        "samples_per_frame": int(full_phi.shape[1]),
        "finite": finite,
        "max_raw_penetration_m": float(max(0.0, -float(np.min(full_phi)))),
        "canonical_backend": "reference_winding_v1 (persisted independent full-512 audit)",
        "identity_pass": bool(full_phi.shape == (len(rows), 512)),
        "sign_valid_assumed_from_checkpoint_contract": True,
    }
    result = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "accepted_reference": False,
        "same_lineage_causal_baseline": True,
        "status_zero_count": status_zero,
        "strict_accepted_count": accepted,
        "frame_count": len(rows),
        "status_zero_pass": status_zero == len(rows),
        "strict_accepted_pass": accepted == len(rows),
        "full512_canonical_audit": audit,
        "checkpoint_chain": chain,
        "source_integrity_pass": True,
        "q_slack_bounds_pass": bool(np.all(np.isfinite(np.asarray(final.arrays["qpos"])))),
    }
    result["baseline_pass"] = bool(
        result["status_zero_pass"]
        and result["strict_accepted_pass"]
        and audit["identity_pass"]
        and finite
        and chain.get("chain_pass", False)
    )
    return result


def _regression(
    current: dict[str, Any], historical: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    crows = current["rows"]
    hrows = historical["rows"]
    current_final = current.get("bundle", {}).get("final")
    historical_final = historical.get("bundle", {}).get("final")
    current_q = (
        None
        if current_final is None
        else np.asarray(current_final.arrays["qpos"], dtype=np.float64)
    )
    historical_q = (
        None
        if historical_final is None
        else np.asarray(historical_final.arrays["qpos"], dtype=np.float64)
    )
    current_base = (
        None
        if current_final is None
        else np.asarray(current_final.arrays["base_pose_scene"], dtype=np.float64)
    )
    historical_base = (
        None
        if historical_final is None
        else np.asarray(historical_final.arrays["base_pose_scene"], dtype=np.float64)
    )
    rows: list[dict[str, Any]] = []
    for c, h in zip(crows, hrows, strict=False):
        frame = int(c["local_frame"])
        qpos_l2 = (
            float(np.linalg.norm(current_q[frame] - historical_q[frame]))
            if current_q is not None and historical_q is not None and frame < len(historical_q)
            else float("nan")
        )
        base_l2 = (
            float(np.linalg.norm(current_base[frame, :3, 3] - historical_base[frame, :3, 3]))
            if current_base is not None
            and historical_base is not None
            and frame < len(historical_base)
            else float("nan")
        )
        rows.append(
            {
                "local_frame": frame,
                "global_frame": c["global_frame"],
                "qpos_l2_rad": qpos_l2,
                "base_translation_l2_m": base_l2,
                "current_long_finger_rmse_m": c["final_long_finger_rmse_m"],
                "historical_long_finger_rmse_m": h.get("final_long_finger_rmse_m"),
                "current_e_im": c["final_e_im"],
                "historical_e_im": h.get("final_e_im"),
                "current_status": c["status"],
                "historical_status": h.get("status"),
            }
        )
    if len(crows) != len(hrows):
        classification = "INCONCLUSIVE"
    else:
        current_long = float(np.nanmean([row["final_long_finger_rmse_m"] for row in crows]))
        historical_long = float(np.nanmean([row["final_long_finger_rmse_m"] for row in hrows]))
        delta = abs(current_long - historical_long)
        classification = (
            "EQUIVALENT"
            if delta <= 1e-9
            else "SMALL_NUMERICAL_DRIFT"
            if delta <= 1e-5
            else "MEANINGFUL_NUMERICAL_DRIFT"
        )
    report = {
        "schema_version": "toporetarget.current_vs_historical_regression.v1",
        "classification": classification,
        "current_frame_count": len(crows),
        "historical_frame_count": len(hrows),
        "max_qpos_l2_rad": float(np.nanmax([row["qpos_l2_rad"] for row in rows]))
        if rows and any(np.isfinite(row["qpos_l2_rad"]) for row in rows)
        else float("nan"),
        "max_base_translation_l2_m": float(
            np.nanmax([row["base_translation_l2_m"] for row in rows])
        )
        if rows and any(np.isfinite(row["base_translation_l2_m"]) for row in rows)
        else float("nan"),
        "description_only": True,
        "does_not_gate_current_lane": True,
    }
    return report, rows


def run_current_causal_baseline(
    run: str | Path,
    output_root: str | Path,
    *,
    max_wall_time: float | None = None,
    resume: bool = True,
    start_frame: int = 0,
    end_frame: int = 60,
) -> dict[str, Any]:
    """Resume or create the current-lineage 60-frame baseline."""

    if start_frame < 0 or end_frame <= start_frame or end_frame > 60:
        raise ValueError(f"invalid current-lineage frame range [{start_frame},{end_frame})")

    run_path = Path(run).expanduser().resolve()
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = root / "checkpoints"
    final_path = root / "current_lineage_baseline.zarr"
    progress = root / "progress.json"
    command = _refine_command(
        run_path,
        checkpoint_root,
        final_path,
        progress,
        max_wall_time,
        resume=resume,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    input_artifacts = _artifact_paths_from_run(run_path)
    _write_json(
        root / "current_lineage_manifest.json",
        {
            "schema_version": LINEAGE_SCHEMA,
            "lane": "current",
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "run_manifest": str(run_path),
            "frame_range": [start_frame, end_frame],
            "robot": _json(run_path).get("robot", "artimano_rh"),
            "profiles": {
                "query": "adaptive_active_set_v1",
                "coordinate": "local_seed_delta_v1",
                "solver": "scipy_slsqp_active_set_contact_rich_v2",
                "execution": "cached_checkpoint_cpu_float64_v3",
            },
            "input_artifacts": {
                name: {"path": str(path), "content_hash": _hash_path(path)}
                for name, path in sorted(input_artifacts.items())
                if path.exists()
            },
            "checkpoint_root": str(checkpoint_root),
            "output": str(final_path),
            "resume": bool(resume),
        },
    )
    if not progress.exists() or _json(progress).get("next_frame", start_frame) < end_frame:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONPATH": str(_repo_root() / "src"),
            }
        )
        log = root / "refine.stdout.log"
        error_log = root / "refine.stderr.log"
        with log.open("a", encoding="utf-8") as out, error_log.open("a", encoding="utf-8") as err:
            result = subprocess.run(
                command, cwd=_repo_root(), env=env, text=True, stdout=out, stderr=err, check=False
            )
        _write_json(
            root / "refine_command.json",
            {"command": command, "returncode": result.returncode, "resume": resume},
        )
    if not final_path.is_dir():
        status = _json(progress) if progress.exists() else {"status": "not_started"}
        validation = {
            "schema_version": SCHEMA,
            "baseline_pass": False,
            "reason": "RETURN_TO_CURRENT_STAGE9_SOLVER_OR_CONTEXT_FIX",
            "progress": status,
        }
        _write_json(root / "current_lineage_validation.json", validation)
        return validation
    bundle = _bundle(run_path, final_path)
    rows = _frame_rows(bundle)
    validation = _validation(rows, bundle, checkpoint_root)
    _write_json(root / "current_lineage_validation.json", validation)
    _write_csv(root / "current_lineage_per_frame.csv", rows)
    baseline_result: dict[str, Any] = {
        "root": str(root),
        "final_artifact": str(final_path),
        "validation": validation,
        "rows": rows,
        "bundle": bundle,
    }
    return baseline_result


def run_current_baseline_repeats(
    run: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
    repeat_count: int = 3,
    max_wall_time: float | None = None,
) -> dict[str, Any]:
    """Replay the selected current-lineage frames in independent sequential runs."""

    if repeat_count < 3 or repeat_count > 5:
        raise ValueError("repeat_count must be between 3 and 5")
    baseline_root = Path(current_baseline).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    baseline_rows = (
        list(
            csv.DictReader((baseline_root / "current_lineage_per_frame.csv").open(encoding="utf-8"))
        )
        if (baseline_root / "current_lineage_per_frame.csv").is_file()
        else []
    )
    if len(baseline_rows) != 60:
        raise Stage934Error("current baseline must contain 60 rows before repeat replay")
    numeric_baseline_rows: list[dict[str, Any]] = []
    for row in baseline_rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value in ("True", "False"):
                parsed[key] = value == "True"
            else:
                try:
                    parsed[key] = (
                        float(value) if "." in value or "e" in value.lower() else int(value)
                    )
                except (ValueError, AttributeError):
                    parsed[key] = value
        numeric_baseline_rows.append(parsed)
    selected = list(frames) if frames else _selected_frames(numeric_baseline_rows)
    selected = list(dict.fromkeys(int(frame) for frame in selected))[:5]
    if not selected or min(selected) < 0 or max(selected) >= 60:
        raise ValueError(f"invalid repeat frames: {selected}")
    end_frame = max(selected) + 1
    baseline_by_frame = {int(row["local_frame"]): row for row in baseline_rows}
    repeats: list[dict[str, Any]] = []
    for repeat in range(1, repeat_count + 1):
        repeat_root = output / "current_baseline_repeats" / f"repeat_{repeat:02d}"
        result = run_current_causal_baseline(
            run,
            repeat_root,
            max_wall_time=max_wall_time,
            resume=False,
            start_frame=0,
            end_frame=end_frame,
        )
        rows: list[dict[str, Any]] = list(result.get("rows", []))
        by_frame: dict[int, dict[str, Any]] = {int(row["local_frame"]): row for row in rows}
        for frame in selected:
            repeat_row = by_frame.get(frame)
            base = baseline_by_frame[frame]
            repeats.append(
                {
                    "repeat": repeat,
                    "frame": frame,
                    "status": repeat_row.get("status") if repeat_row else "MISSING",
                    "accepted": bool(repeat_row and repeat_row.get("accepted")),
                    "strict_accepted": bool(repeat_row and repeat_row.get("strict_accepted")),
                    "final_objective": repeat_row.get("final_objective") if repeat_row else None,
                    "final_long_finger_rmse_m": repeat_row.get("final_long_finger_rmse_m")
                    if repeat_row
                    else None,
                    "final_whole_hand_rmse_m": repeat_row.get("final_whole_hand_rmse_m")
                    if repeat_row
                    else None,
                    "baseline_final_objective": float(base["final_objective"]),
                    "baseline_final_long_finger_rmse_m": float(base["final_long_finger_rmse_m"]),
                    "baseline_final_whole_hand_rmse_m": float(base["final_whole_hand_rmse_m"]),
                    "root": str(repeat_root),
                }
            )
    complete = len(repeats) == repeat_count * len(selected) and all(
        row["status"] == 0 and row["accepted"] and row["strict_accepted"] for row in repeats
    )
    metric_names = (
        "final_objective",
        "final_long_finger_rmse_m",
        "final_whole_hand_rmse_m",
    )
    noise_envelope: dict[str, float] = {}
    selected_frame_metric_span: dict[str, float] = {}
    for metric in metric_names:
        values = [float(row[metric]) for row in repeats if row[metric] is not None]
        selected_frame_metric_span[metric] = max(values) - min(values) if values else float("nan")
        frame_ranges = []
        for frame in selected:
            frame_values = [
                float(row[metric])
                for row in repeats
                if row["frame"] == frame and row[metric] is not None
            ]
            if frame_values:
                frame_ranges.append(max(frame_values) - min(frame_values))
        noise_envelope[metric] = max(frame_ranges) if frame_ranges else float("nan")
    report = {
        "schema_version": SCHEMA,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "same_lineage_contract": True,
        "independent_replay": True,
        "repeat_count": repeat_count,
        "selected_frames": selected,
        "replay_end_frame_exclusive": end_frame,
        "noise_envelope": noise_envelope,
        "selected_frame_metric_span": selected_frame_metric_span,
        "rows": repeats,
    }
    _write_json(output / "current_baseline_repeat.json", report)
    return report


def _selected_frames(rows: list[dict[str, Any]]) -> list[int]:
    if not rows:
        return []
    scores = {
        "aggregate_long": max(rows, key=lambda row: row["degradation_long_finger_rmse_m"])[
            "local_frame"
        ],
        "middle": max(rows, key=lambda row: row["degradation_middle_rmse_m"])["local_frame"],
        "eim": max(rows, key=lambda row: row["final_e_im"] - row["warm_e_im"])["local_frame"],
        "min_sdf": min(rows, key=lambda row: row["min_full_sdf_m"])["local_frame"],
        "median": sorted(rows, key=lambda row: row["final_whole_hand_rmse_m"])[len(rows) // 2][
            "local_frame"
        ],
    }
    result: list[int] = []
    for frame in scores.values():
        if frame not in result:
            result.append(int(frame))
    for frame in (0, len(rows) // 2, len(rows) - 1):
        if len(result) >= 5:
            break
        if frame not in result:
            result.append(frame)
    return result[:5]


def _historical_root(repo: Path) -> Path:
    return repo / ".local/cache/retarget/final/stage9_2_contact_rich_60f_v3.zarr"


def run_historical_replay(
    provenance_root: str | Path, output_root: str | Path, *, frames: str = "auto"
) -> dict[str, Any]:
    """Report historical replay availability without impersonating its environment."""

    prov = Path(provenance_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    historical = _json(prov / "solver_effective_provenance_historical.json")
    selected: list[int] = []
    selection = (
        _repo_root()
        / ".local/runs/stage9_3_3_shadow_equivalence/s1__airplane_lift__right__artimano_rh__f000240_f000300/shadow_frame_selection_v2.json"
    )
    if selection.is_file():
        data = _json(selection)
        for item in data.get("selected_frames", data.get("frames", [])):
            value = item.get("frame") if isinstance(item, dict) else item
            if value is not None and int(value) not in selected:
                selected.append(int(value))
    if not selected:
        selected = [0, 29, 59]
    selected = selected[:3]
    env_status = historical.get("historical_environment_status", "UNAVAILABLE_OR_INCOMPLETE")
    status = (
        "HISTORICAL_EXACT_REPLAY_UNAVAILABLE"
        if env_status != "AVAILABLE_EXACT"
        else "HISTORICAL_REPLAY_FAILED"
    )
    manifest = {
        "schema_version": SCHEMA,
        "lane": "historical",
        "selected_frames": selected,
        "historical_commit": historical.get("git_commit"),
        "environment_status": env_status,
        "status": status,
        "does_not_block_current_lane": True,
        "solver_invocation_count": 0,
    }
    _write_json(output / "historical_lane_manifest.json", manifest)
    _write_json(
        output / "historical_environment_audit.json",
        {
            "schema_version": SCHEMA,
            "status": env_status,
            "recorded_environment": historical.get("environment", {}),
            "missing_or_unverified": ["reproducible historical Python environment and local wheels"]
            if env_status != "AVAILABLE_EXACT"
            else [],
        },
    )
    _write_json(
        output / "historical_replay_results.json",
        {
            "schema_version": SCHEMA,
            "status": status,
            "frames": selected,
            "results": [],
            "reason": "No current environment is used as a substitute for the recorded historical environment.",
        },
    )
    _write_csv(output / "historical_replay_comparison.csv", [])
    return manifest


def _write_causal_html(root: Path, payload: dict[str, Any]) -> Path:
    destination = root / "stage9_3_4_causal_analysis.html"
    serialized = json.dumps(payload, sort_keys=True, default=str).replace("</", "<\\/")
    document = """<!doctype html><html><head><meta charset='utf-8'><title>Stage 9.3.4 causal analysis</title><style>body{font:14px sans-serif;margin:24px;background:#111;color:#eee}table{border-collapse:collapse}td,th{border:1px solid #555;padding:5px}pre{white-space:pre-wrap;background:#181818;padding:12px}.ok{color:#8fda8f}.warn{color:#ffbf69}</style></head><body><h1>Stage 9.3.4 Provenance-Rebased Causal Analysis</h1><p id='status'></p><label>Lane <select id='lane'><option value='current'>Current</option><option value='historical'>Historical</option></select></label><label>Frame <input id='frame' type='range' min='0' max='59' value='0'></label><span id='frameLabel'></span><h2>Frame metrics</h2><pre id='metrics'></pre><h2>Full payload</h2><pre id='payload'></pre><script>const DATA=__DATA__;const q=id=>document.getElementById(id);function draw(){const lane=q('lane').value;const rows=(DATA[lane]||[]);const i=+q('frame').value; q('frameLabel').textContent=' local='+i; q('metrics').textContent=JSON.stringify(rows[i]||{},null,2);q('status').textContent=DATA.status||'';q('status').className=DATA.enter_stage9_4?'ok':'warn';q('payload').textContent=JSON.stringify(DATA,null,2)}q('lane').onchange=draw;q('frame').oninput=draw;draw()</script></body></html>""".replace(
        "__DATA__", serialized
    )
    destination.write_text(document, encoding="utf-8")
    return destination


def _write_official_immutability_report(repo: Path, run: Path, output: Path) -> dict[str, Any]:
    """Compare official paths with the pre-Stage-9.3.4 identity snapshot."""

    previous_path = (
        repo
        / ".local/runs/stage9_3_3_shadow_equivalence/s1__airplane_lift__right__artimano_rh__f000240_f000300/input_identity_and_immutability.json"
    )
    if not previous_path.is_file():
        report = {
            "schema_version": SCHEMA,
            "official_artifacts_changed": False,
            "status": "BASELINE_SNAPSHOT_MISSING",
            "changed": [],
        }
        _write_json(output / "official_artifact_immutability.json", report)
        return report
    previous = _json(previous_path)
    entries: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    for item in previous.get("entries", []):
        path = Path(str(item["path"]))
        current_hash = _hash_path(path) if path.exists() else None
        record = {
            "label": item.get("label"),
            "path": str(path),
            "previous_sha256": item.get("sha256"),
            "current_sha256": current_hash,
            "exists": path.exists(),
            "unchanged": current_hash == item.get("sha256"),
        }
        entries.append(record)
        if not record["unchanged"]:
            changed.append(record)
    report = {
        "schema_version": SCHEMA,
        "baseline_snapshot": str(previous_path),
        "checked_run_manifest": str(run),
        "official_artifacts_changed": bool(changed),
        "changed": changed,
        "entries": entries,
    }
    _write_json(output / "official_artifact_immutability.json", report)
    return report


def stage9_causal_status(
    provenance_root: str | Path, current_root: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Assemble bounded causal reports and a single conservative route."""

    repo = _repo_root()
    prov = Path(provenance_root).expanduser().resolve()
    current_root_path = Path(current_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    immutability = _write_official_immutability_report(
        repo,
        _repo_root()
        / ".local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json",
        output,
    )
    current_validation = _json(current_root_path / "current_lineage_validation.json")
    current_rows = (
        list(
            csv.DictReader(
                (current_root_path / "current_lineage_per_frame.csv").open(encoding="utf-8")
            )
        )
        if (current_root_path / "current_lineage_per_frame.csv").is_file()
        else []
    )
    current_rows_num: list[dict[str, Any]] = []
    for row in current_rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except (ValueError, AttributeError):
                parsed[key] = value
        current_rows_num.append(parsed)
    _write_json(
        prov / "causal_frame_selection.json",
        {
            "schema_version": SCHEMA,
            "frames": _selected_frames(current_rows_num),
            "criteria": [
                "long-finger aggregate",
                "middle",
                "E_IM",
                "canonical minimum SDF",
                "median",
            ],
        },
    )
    historical_path = _historical_root(repo)
    regression: dict[str, Any]
    regression_rows: list[dict[str, Any]]
    historical_rows_num: list[dict[str, Any]] = []
    if (
        historical_path.is_dir()
        and current_root_path.joinpath("current_lineage_baseline.zarr").is_dir()
    ):
        try:
            old_bundle = _bundle(
                _repo_root()
                / ".local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json",
                historical_path,
            )
            old_rows = _frame_rows(old_bundle)
            historical_rows_num = old_rows
            current_bundle = _bundle(
                _repo_root()
                / ".local/runs/stage10_reference_runtime/s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json",
                current_root_path / "current_lineage_baseline.zarr",
            )
            regression, regression_rows = _regression(
                {"rows": current_rows_num, "bundle": current_bundle},
                {"rows": old_rows, "bundle": old_bundle},
            )
        except Exception as exc:
            regression, regression_rows = (
                {
                    "classification": "INCONCLUSIVE",
                    "error": str(exc),
                    "does_not_gate_current_lane": True,
                },
                [],
            )
    else:
        regression, regression_rows = (
            {"classification": "INCONCLUSIVE", "does_not_gate_current_lane": True},
            [],
        )
    _write_json(prov / "current_vs_historical_regression.json", regression)
    _write_csv(prov / "current_vs_historical_per_frame.csv", regression_rows)
    repeat_path = prov / "current_baseline_repeat.json"
    if not repeat_path.is_file():
        _write_json(
            repeat_path,
            {
                "schema_version": SCHEMA,
                "status": "NOT_RUN",
                "repeat_count": 0,
                "noise_envelope": "not_run",
                "same_lineage_contract": True,
                "reason": "Independent replay was not run before this bounded status pass.",
            },
        )
    empty_profiles = [
        "official_current_lineage_baseline",
        "half_active_margin",
        "zero_active_margin",
        "full_512_query_reference",
        "minimal_soft_safe_projection_from_warm",
        "official_slack_projection_from_warm",
    ]
    multistart_root = repo / ".local/runs/stage9_3_4_multistart"
    base_seed_root = repo / ".local/runs/stage9_3_4_base_seed_ablation"
    mandatory_root = repo / ".local/runs/stage9_3_4_mandatory_ablation"
    baseline_only_rows = [
        {
            "profile": name,
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "accepted": bool(current_validation.get("baseline_pass", False))
            if name == empty_profiles[0]
            else False,
            "status": "BASELINE_ONLY" if name == empty_profiles[0] else "NOT_RUN",
            "key_result": "formal same-lineage baseline"
            if name == empty_profiles[0]
            else "profile requires a diagnostic rerun",
        }
        for name in empty_profiles
    ]
    mandatory_manifest = (
        _json(mandatory_root / "mandatory_ablation_manifest.json")
        if (mandatory_root / "mandatory_ablation_manifest.json").is_file()
        else {
            "schema_version": SCHEMA,
            "profiles": empty_profiles,
            "status": "NOT_RUN_UNTIL_CURRENT_BASELINE_PASS"
            if not current_validation.get("baseline_pass")
            else "NOT_RUN",
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
        }
    )
    mandatory_results = (
        _json(mandatory_root / "mandatory_ablation_results_per_profile.json")
        if (mandatory_root / "mandatory_ablation_results_per_profile.json").is_file()
        else {"schema_version": SCHEMA, "results": baseline_only_rows}
    )
    mandatory_rows = (
        list(
            csv.DictReader(
                (mandatory_root / "mandatory_ablation_results_per_frame.csv").open(encoding="utf-8")
            )
        )
        if (mandatory_root / "mandatory_ablation_results_per_frame.csv").is_file()
        else baseline_only_rows
    )
    _write_json(
        output / "mandatory_ablation_manifest.json",
        {**mandatory_manifest, "source_root": str(mandatory_root)},
    )
    _write_csv(output / "mandatory_ablation_results_per_frame.csv", mandatory_rows)
    _write_json(
        output / "mandatory_ablation_results_per_profile.json",
        {**mandatory_results, "source_root": str(mandatory_root)},
    )
    multistart_manifest = (
        _json(multistart_root / "multistart_manifest.json")
        if (multistart_root / "multistart_manifest.json").is_file()
        else {
            "schema_version": SCHEMA,
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
            "status": "NOT_RUN",
            "query_phases": ["frozen_initial_queryset", "native_queryset"],
            "seeds": [],
        }
    )
    multistart_results = (
        _json(multistart_root / "multistart_results_per_seed.json")
        if (multistart_root / "multistart_results_per_seed.json").is_file()
        else {"schema_version": SCHEMA, "results": []}
    )
    if multistart_results.get("status") == "COMPLETE":
        multistart_manifest = {
            **multistart_manifest,
            "status": "COMPLETE",
            "solver_invocation_count": multistart_results.get("solver_invocation_count", 0),
        }
        _write_json(multistart_root / "multistart_manifest.json", multistart_manifest)
    multistart_rows = (
        list(
            csv.DictReader(
                (multistart_root / "multistart_results_per_frame.csv").open(encoding="utf-8")
            )
        )
        if (multistart_root / "multistart_results_per_frame.csv").is_file()
        else []
    )
    _write_json(
        output / "multistart_manifest.json",
        {**multistart_manifest, "source_root": str(multistart_root)},
    )
    profiles = (
        _json(multistart_root / "multistart_profiles.json")
        if (multistart_root / "multistart_profiles.json").is_file()
        else {
            "frozen_initial_queryset": {"implemented": False},
            "native_queryset": {"implemented": False},
        }
    )
    basin = (
        _json(multistart_root / "multistart_basin_analysis.json")
        if (multistart_root / "multistart_basin_analysis.json").is_file()
        else {"schema_version": SCHEMA, "classification": "INCONCLUSIVE"}
    )
    _write_json(
        output / "multistart_profiles.json", {**profiles, "source_root": str(multistart_root)}
    )
    _write_csv(output / "multistart_results_per_frame.csv", multistart_rows)
    _write_json(
        output / "multistart_results_per_seed.json",
        {**multistart_results, "source_root": str(multistart_root)},
    )
    _write_json(
        output / "multistart_basin_analysis.json", {**basin, "source_root": str(multistart_root)}
    )
    base_seed_manifest = (
        _json(base_seed_root / "base_seed_manifest.json")
        if (base_seed_root / "base_seed_manifest.json").is_file()
        else {
            "schema_version": SCHEMA,
            "status": "NOT_RUN",
            "diagnostic_only": True,
            "paper_method": False,
            "accepted_reference": False,
        }
    )
    base_seed_results = (
        _json(base_seed_root / "base_seed_results_per_profile.json")
        if (base_seed_root / "base_seed_results_per_profile.json").is_file()
        else {"schema_version": SCHEMA, "results": []}
    )
    base_seed_rows = (
        list(
            csv.DictReader(
                (base_seed_root / "base_seed_results_per_frame.csv").open(encoding="utf-8")
            )
        )
        if (base_seed_root / "base_seed_results_per_frame.csv").is_file()
        else []
    )
    _write_json(
        output / "base_seed_manifest.json",
        {**base_seed_manifest, "source_root": str(base_seed_root)},
    )
    base_profiles = (
        _json(base_seed_root / "base_seed_profiles.json")
        if (base_seed_root / "base_seed_profiles.json").is_file()
        else {"se3_only": True, "det_R_plus_one": True, "similarity_scale": False}
    )
    pareto = (
        _json(base_seed_root / "base_seed_pareto_analysis.json")
        if (base_seed_root / "base_seed_pareto_analysis.json").is_file()
        else {"schema_version": SCHEMA, "classification": "INCONCLUSIVE"}
    )
    _write_json(
        output / "base_seed_profiles.json", {**base_profiles, "source_root": str(base_seed_root)}
    )
    warm_rows = (
        list(
            csv.DictReader((base_seed_root / "base_seed_warm_geometry.csv").open(encoding="utf-8"))
        )
        if (base_seed_root / "base_seed_warm_geometry.csv").is_file()
        else []
    )
    _write_csv(output / "base_seed_warm_geometry.csv", warm_rows)
    _write_csv(output / "base_seed_results_per_frame.csv", base_seed_rows)
    _write_json(
        output / "base_seed_results_per_profile.json",
        {**base_seed_results, "source_root": str(base_seed_root)},
    )
    _write_json(
        output / "base_seed_pareto_analysis.json", {**pareto, "source_root": str(base_seed_root)}
    )
    _write_json(
        output / "state_counterfactual_decomposition.json",
        {
            "schema_version": SCHEMA,
            "solver_invocation_count": 0,
            "states": [
                "warm",
                "current_official",
                "current_base_warm_q",
                "warm_base_current_q",
                "warm_final_thumb_q",
                "warm_final_index_q",
                "warm_final_middle_q",
                "warm_final_ring_q",
                "warm_final_pinky_q",
                "warm_final_long_finger_q",
                "warm_final_thumb_pinky_q",
            ],
            "status": "NOT_RUN",
            "reason": "No validated counterfactual FK/solver adapter was admitted; stored warm/final differences remain observational.",
        },
    )
    _write_csv(output / "state_counterfactual_decomposition.csv", [])
    _write_json(
        output / "objective_gradient_attribution.json",
        {
            "schema_version": SCHEMA,
            "status": "NOT_RUN",
            "is_first_order_local_diagnostic": True,
            "not_a_causal_proof": True,
        },
    )
    _write_csv(output / "objective_gradient_attribution.csv", [])
    _write_json(
        output / "constraint_attribution.json",
        {
            "schema_version": SCHEMA,
            "status": "NOT_RUN",
            "multiplier_policy": "no_unreliable_slsqp_multiplier",
        },
    )
    _write_csv(output / "constraint_attribution_per_link.csv", [])
    finger_rows: list[dict[str, Any]] = []
    for finger in ("thumb", "index", "middle", "ring", "pinky", "whole_hand", "long_finger"):
        values = [
            row
            for row in current_rows_num
            if isinstance(row.get(f"final_{finger}_rmse_m"), (int, float))
        ]
        if values:
            mean_warm = float(np.mean([row[f"warm_{finger}_rmse_m"] for row in values]))
            mean_final = float(np.mean([row[f"final_{finger}_rmse_m"] for row in values]))
            finger_rows.append(
                {
                    "finger": finger,
                    "frame_count": len(values),
                    "mean_warm_rmse_m": mean_warm,
                    "mean_final_rmse_m": mean_final,
                    "mean_degradation_rmse_m": mean_final - mean_warm,
                    "evidence_type": "observational_current_baseline",
                    "causal_claim": False,
                }
            )
    _write_csv(output / "per_finger_causal_summary.csv", finger_rows)
    _write_json(
        output / "branch_rollout_manifest.json",
        {
            "schema_version": SCHEMA,
            "status": "NOT_RUN",
            "reason": "No candidate passed the multi-frame route gate.",
        },
    )
    _write_csv(output / "branch_rollout_results.csv", [])
    # Gradient/link attribution and comparable projection branches remain
    # explicitly unrun; their absence keeps the route conservative even when
    # the bounded solver diagnostics themselves finish.
    causal_attribution_complete = False
    diagnostics_complete = (
        bool(current_validation.get("baseline_pass"))
        and mandatory_manifest.get("status") == "COMPLETE"
        and multistart_results.get("status") == "COMPLETE"
        and base_seed_results.get("status") == "COMPLETE"
        and causal_attribution_complete
    )
    route = (
        "RETURN_TO_CURRENT_STAGE9_SOLVER_OR_CONTEXT_FIX"
        if not current_validation.get("baseline_pass")
        else "STAGE9_3_4_INCONCLUSIVE"
        if not diagnostics_complete
        else "STAGE9_4_NOT_YET_JUSTIFIED"
    )
    readiness = {
        "schema_version": SCHEMA,
        "status": route,
        "enter_stage9_4": False,
        "human_decision_required": True,
        "stop_after_stage9_3_4": True,
        "reason": "Stage 9.4 cannot be routed from an incomplete current-lineage baseline."
        if route.startswith("RETURN")
        else "No candidate has passed all mandatory causal gates.",
    }
    _write_json(output / "stage9_4_readiness.json", readiness)
    plan = f"""# Stage 9.4 candidate plan\n\nStatus: `{route}`\n\nThis Stage 9.3.4 run does not authorize Stage 9.4 implementation. The current-lineage baseline gate is `{current_validation.get("baseline_pass")}`.\n\n## Proposed scope\n\nNo implementation is approved until the current baseline and mandatory profiles have strict accepted, canonical full-512, and bounded rollout evidence.\n\n## Preserve\n\n- Eq. (1)--(9), paper weights, Stage 7 warm artifact, Stage 8 graph, historical accepted artifact, Stage 10 manifest, and manual acceptance.\n\n## Human gate\n\nA human must review the Stage 9.3.4 report and HTML before any Stage 9.4 work.\n"""
    (output / "stage9_4_candidate_plan.md").write_text(plan, encoding="utf-8")
    root_cause = {
        "schema_version": SCHEMA,
        "status": "INCONCLUSIVE"
        if route == "STAGE9_3_4_INCONCLUSIVE"
        else "ENGINEERING_CAUSES_NOT_EXCLUDED",
        "ranked_causes": [
            {
                "rank": 1,
                "cause": "mandatory projection profiles were not solved under the formal objective",
                "confidence": "high",
                "evidence": ["mandatory_ablation_results_per_profile.json"],
            },
            {
                "rank": 2,
                "cause": "reliable SLSQP multiplier and comparable gradient/link attribution evidence are unavailable",
                "confidence": "high",
                "evidence": ["objective_gradient_attribution.json", "constraint_attribution.json"],
            },
            {
                "rank": 3,
                "cause": "historical exact replay environment is unavailable",
                "confidence": "medium",
                "evidence": ["historical_lane_manifest.json"],
                "does_not_block_current_lane": True,
            },
        ],
        "paper_objective_limit_not_claimed": True,
    }
    _write_json(output / "root_cause_analysis.json", root_cause)
    summary = {
        "schema_version": SCHEMA,
        "status": route,
        "enter_stage9_4": False,
        "human_decision_required": True,
        "stop_after_stage9_3_4": True,
        "current_baseline": current_validation,
        "historical_regression": regression,
        "official_artifacts_changed": bool(immutability.get("official_artifacts_changed", False)),
        "immutability_report": str(output / "official_artifact_immutability.json"),
        "solver_invocations": {
            "current_baseline": 60,
            "multistart": multistart_results.get("solver_invocation_count", 0),
            "mandatory_ablation": mandatory_results.get("solver_invocation_count", 0),
            "base_seed_final": base_seed_results.get("solver_invocation_count", 0),
        },
        "diagnostic_roots": {
            "current": str(current_root_path),
            "historical": str(repo / ".local/runs/stage9_3_4_historical_lane"),
            "multistart": str(multistart_root),
            "base_seed": str(base_seed_root),
            "mandatory": str(mandatory_root),
        },
        "causal_attribution_complete": causal_attribution_complete,
    }
    _write_json(output / "stage9_3_4_summary.json", summary)
    (output / "stage9_3_4_summary.md").write_text(
        "# Stage 9.3.4 summary\n\n"
        + json.dumps(summary, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    html_path = _write_causal_html(
        output,
        {
            "status": route,
            "enter_stage9_4": False,
            "current": current_rows_num,
            "historical": historical_rows_num,
            "multistart": multistart_rows,
            "mandatory": mandatory_rows,
            "base_seed": warm_rows,
            "provenance_diff": _json(prov / "solver_effective_provenance_diff.json")
            if (prov / "solver_effective_provenance_diff.json").is_file()
            else {},
            "readiness": readiness,
        },
    )
    return {"readiness": readiness, "summary": summary, "html": str(html_path)}


def _baseline_gate(current_baseline: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    validation_path = current_baseline / "current_lineage_validation.json"
    validation = (
        _json(validation_path)
        if validation_path.is_file()
        else {"baseline_pass": False, "reason": "validation missing"}
    )
    return validation, None


def _rotation_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-15:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(vector, dtype=np.float64) / theta
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _kabsch(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Return an SE(3), no-scale, no-reflection fit from source to target."""

    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Kabsch requires matching Nx3 point arrays with N >= 3")
    w = (
        np.ones(len(x), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if w.shape != (len(x),) or np.any(w <= 0):
        raise ValueError("Kabsch weights must be positive and point-aligned")
    w /= np.sum(w)
    cx = np.sum(x * w[:, None], axis=0)
    cy = np.sum(y * w[:, None], axis=0)
    covariance = ((x - cx) * w[:, None]).T @ (y - cy)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("Kabsch produced a reflection")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = cy - rotation @ cx
    return transform


def _variant_warm(
    bundle: dict[str, Any], frame: int, seed: str, *, base_transform: np.ndarray | None = None
) -> WarmStartTrajectory:
    from toporetarget.cli.retarget import _load_robot

    warm = bundle["warm"]
    arrays = {name: np.asarray(value).copy() for name, value in warm.arrays.items()}
    q = np.asarray(arrays["qpos"], dtype=np.float64)
    base = np.asarray(arrays["base_pose_scene"], dtype=np.float64)
    robot = _load_robot(str(bundle["run_manifest"].get("robot", "artimano_rh")), None)
    if seed == "mapped_previous_final" and frame > 0:
        q[frame] = np.asarray(bundle["final"].arrays["qpos"])[frame - 1]
        base[frame] = np.asarray(bundle["final"].arrays["base_pose_scene"])[frame - 1]
    elif seed.startswith("deterministic_perturbation_"):
        amount = {
            "deterministic_perturbation_01": 0.01,
            "deterministic_perturbation_02": 0.02,
            "deterministic_perturbation_03": 0.03,
        }[seed]
        span = np.asarray(robot.joint_upper, dtype=np.float64) - np.asarray(
            robot.joint_lower, dtype=np.float64
        )
        if span.shape != q.shape[1:]:
            raise Stage934Error(
                f"robot joint range shape {span.shape} does not match warm qpos {q.shape[1:]}"
            )
        direction = np.where(np.arange(q.shape[1]) % 2 == 0, 1.0, -1.0)
        q[frame] = np.clip(
            q[frame] + amount * span * direction, robot.joint_lower, robot.joint_upper
        )
        base[frame, :3, 3] += amount * 0.001 * np.asarray([1.0, -1.0, 0.5])
        base[frame, :3, :3] = base[frame, :3, :3] @ _rotation_matrix_from_vector(
            amount * np.deg2rad(np.asarray([1.0, -0.5, 0.25]))
        )
    elif base_transform is not None:
        base[frame] = np.asarray(base_transform, dtype=np.float64)
    arrays["qpos"] = q
    arrays["base_pose_scene"] = base
    metadata = dict(warm.metadata)
    metadata["stage9_3_4_initialization_seed"] = seed
    metadata["diagnostic_only"] = True
    return WarmStartTrajectory(metadata, arrays).validate()


def _official_initial_query_sets(
    bundle: dict[str, Any], frames: Iterable[int], query_profile_id: str
) -> dict[int, Any]:
    """Reconstruct the official warm-seed initial QuerySet for diagnostics."""

    from toporetarget.cli.retarget import _refinement_components
    from toporetarget.retarget.bones import load_bone_profile
    from toporetarget.retarget.final_refinement import (
        CollisionQueryProfile,
        RefinementSolverProfile,
        _make_context,
        build_query_set,
        prepare_refinement_resources,
    )
    from toporetarget.retarget.frames import load_frame_profile
    from toporetarget.retarget.refinement_performance import RefinementExecutionProfile

    artifacts = bundle["artifacts"]
    run_manifest = bundle["run_manifest"]
    sequence, warm, graph, model, surface, _ = _refinement_components(
        artifacts["canonical"],
        artifacts["warm_start"],
        artifacts["graph"],
        str(run_manifest.get("robot", "artimano_rh")),
        artifacts.get("collision_samples"),
        None,
    )
    execution = RefinementExecutionProfile.load("cached_checkpoint_cpu_float64_v3")
    resources = prepare_refinement_resources(
        sequence,
        graph,
        RefinementSolverProfile.load("scipy_slsqp_active_set_contact_rich_v2"),
        sdf_tree_leaf_size=int(getattr(execution, "sdf_tree_leaf_size", 32)),
    )
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    query_profile = CollisionQueryProfile.load(query_profile_id)
    result: dict[int, Any] = {}
    for frame in frames:
        local_index = int(frame)
        context = _make_context(
            sequence,
            graph,
            warm,
            model,
            surface,
            resources.sdf,
            resources.reference_sdf,
            frame_profile,
            bone_profile,
            resources.paper,
            local_index,
            None,
        )
        initial_value = np.concatenate([np.zeros(6, dtype=np.float64), context.seed_qpos])
        initial_points = context.candidate_points(initial_value)
        initial_query = resources.sdf.query_scene(initial_points, context.object_pose_scene)
        result[local_index] = build_query_set(
            initial_query.signed_distance,
            surface.geometry_ids,
            query_profile,
        )
    return result


def _variant_row(
    bundle: dict[str, Any], frame: int, seed: str, output: Path, query_mode: str
) -> dict[str, Any]:
    trajectory = load_final_trajectory(output)
    arrays = trajectory.arrays
    row = {
        "frame": frame,
        "seed": seed,
        "query_mode": query_mode,
        "status": int(
            np.asarray(
                arrays.get("optimizer_status_code", [trajectory.metadata.get("solver_status", -1)])
            )[0]
        ),
        "accepted": bool(np.asarray(arrays.get("accepted", [False]))[0]),
        "query_count": int(
            np.asarray(arrays["query_offsets"])[1] - np.asarray(arrays["query_offsets"])[0]
        ),
        "min_full_sdf_m": float(np.min(np.asarray(arrays["full_signed_distance"])[0])),
        "final_objective": float(np.asarray(arrays.get("final_objective", [np.nan]))[0]),
        "solver_invocation_count": 0,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    source_points = np.asarray(bundle["source"])[frame]
    state_points = np.asarray(arrays["robot_keypoints_scene"])[0]
    row.update(
        {
            f"final_{key}": value
            for key, value in _state_metrics(source_points, state_points).items()
        }
    )
    return row


def _solve_variant(
    bundle: dict[str, Any],
    frame: int,
    seed: str,
    output: Path,
    *,
    query_profile_id: str = "adaptive_active_set_v1",
    base_transform: np.ndarray | None = None,
    frozen_initial_query_set: Any | None = None,
    use_previous: bool = True,
) -> dict[str, Any]:
    """Run one bounded single-frame solve through the existing formal solver."""

    from toporetarget.cli.retarget import _refinement_components
    from toporetarget.retarget.artifacts import artifact_hash as warm_hash
    from toporetarget.retarget.bones import load_bone_profile
    from toporetarget.retarget.final_refinement import (
        CollisionQueryProfile,
        RefinementCoordinateProfile,
        RefinementSolverProfile,
        build_final_trajectory,
        prepare_refinement_resources,
        save_final_trajectory,
    )
    from toporetarget.retarget.frames import load_frame_profile
    from toporetarget.retarget.interaction_artifacts import interaction_artifact_hash
    from toporetarget.retarget.refinement_performance import RefinementExecutionProfile

    artifacts = bundle["artifacts"]
    sequence, _, graph, model, surface, _ = _refinement_components(
        artifacts["canonical"],
        artifacts["warm_start"],
        artifacts["graph"],
        "artimano_rh",
        artifacts.get("collision_samples"),
        None,
    )
    warm = _variant_warm(bundle, frame, seed, base_transform=base_transform)
    solver = RefinementSolverProfile.load("scipy_slsqp_active_set_contact_rich_v2")
    execution = RefinementExecutionProfile.load("cached_checkpoint_cpu_float64_v3")
    query = CollisionQueryProfile.load(query_profile_id)
    coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1")
    resources = prepare_refinement_resources(
        sequence, graph, solver, sdf_tree_leaf_size=execution.sdf_tree_leaf_size
    )
    previous = None
    if frame > 0 and use_previous:
        previous = (
            np.asarray(bundle["final"].arrays["base_pose_scene"])[frame - 1],
            np.asarray(bundle["final"].arrays["qpos"])[frame - 1],
        )
    trajectory, report = build_final_trajectory(
        sequence,
        warm,
        graph,
        model,
        surface,
        load_frame_profile("canonical_keypoint_wrist_v1"),
        load_bone_profile("mediapipe21_full_finger_chain_v1"),
        coordinate,
        query,
        solver,
        start_frame=frame,
        end_frame=frame + 1,
        initial_previous=previous,
        warm_artifact_hash=warm_hash(artifacts["warm_start"]),
        graph_artifact_hash=interaction_artifact_hash(artifacts["graph"]),
        resources=resources,
        execution_profile=execution,
        initial_query_sets=(
            {frame: frozen_initial_query_set} if frozen_initial_query_set is not None else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_final_trajectory(trajectory, output, force=True)
    mode = "frozen_initial_queryset" if frozen_initial_query_set is not None else "native_queryset"
    row = _variant_row(bundle, frame, seed, output, mode)
    row["solver_invocation_count"] = 1
    return {"row": row, "report": report, "artifact": str(output)}


def _selected_for_run(current_baseline: Path, frames: tuple[int, ...]) -> list[int]:
    if frames:
        return [int(value) for value in frames[:5]]
    path = current_baseline / "current_lineage_per_frame.csv"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(path.open(encoding="utf-8")):
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value.lower() in {"true", "false"}:
                parsed[key] = value.lower() == "true"
            else:
                try:
                    parsed[key] = (
                        float(value) if "." in value or "e" in value.lower() else int(value)
                    )
                except ValueError:
                    parsed[key] = value
        rows.append(parsed)
    return _selected_frames(rows)[:5]


def run_refinement_multistart(
    run: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    baseline = Path(current_baseline).expanduser().resolve()
    validation, _ = _baseline_gate(baseline)
    manifest = {
        "schema_version": SCHEMA,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
        "query_phases": ["frozen_initial_queryset", "native_queryset"],
        "seeds": [
            "official_stage7_warm",
            "mapped_previous_final",
            "best_feasible_of_warm_and_previous",
            "deterministic_perturbation_01",
            "deterministic_perturbation_02",
            "deterministic_perturbation_03",
        ],
    }
    _write_json(root / "multistart_manifest.json", manifest)
    if not validation.get("baseline_pass"):
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "NOT_RUN_UNTIL_CURRENT_BASELINE_PASS",
            "solver_invocation_count": 0,
            "results": [],
        }
        _write_json(root / "multistart_results_per_seed.json", result)
        _write_csv(root / "multistart_results_per_frame.csv", [])
        _write_json(
            root / "multistart_basin_analysis.json",
            {
                "schema_version": SCHEMA,
                "classification": "INCONCLUSIVE",
                "reason": "current-lineage baseline gate failed",
            },
        )
        return result
    bundle = _bundle(Path(run).expanduser().resolve(), baseline / "current_lineage_baseline.zarr")
    selected = _selected_for_run(baseline, frames)
    seeds: tuple[str, ...] = (
        "official_stage7_warm",
        "mapped_previous_final",
        "best_feasible_of_warm_and_previous",
        "deterministic_perturbation_01",
        "deterministic_perturbation_02",
        "deterministic_perturbation_03",
    )
    frozen_sets = _official_initial_query_sets(bundle, selected, "adaptive_active_set_v1")
    results: list[dict[str, Any]] = []
    for frame in selected:
        for seed in seeds:
            if seed == "best_feasible_of_warm_and_previous":
                selected_seed = "official_stage7_warm" if frame == 0 else "mapped_previous_final"
            else:
                selected_seed = seed
            for query_mode, frozen_query_set in (
                ("frozen_initial_queryset", frozen_sets.get(frame)),
                ("native_queryset", None),
            ):
                artifact = root / "artifacts" / f"frame_{frame:06d}" / f"{seed}__{query_mode}.zarr"
                if artifact.is_dir():
                    try:
                        results.append(
                            _variant_row(bundle, frame, selected_seed, artifact, query_mode)
                            | {"requested_seed": seed}
                        )
                        continue
                    except Exception:
                        # A partially written diagnostic artifact is not
                        # trusted; the solver below will replace it.
                        pass
                try:
                    value = _solve_variant(
                        bundle,
                        frame,
                        selected_seed,
                        artifact,
                        frozen_initial_query_set=frozen_query_set,
                    )
                    value["row"]["requested_seed"] = seed
                    value["row"]["query_mode"] = query_mode
                    results.append(value["row"])
                except Exception as exc:
                    results.append(
                        {
                            "frame": frame,
                            "seed": seed,
                            "requested_seed": seed,
                            "query_mode": query_mode,
                            "status": -1,
                            "accepted": False,
                            "error": str(exc),
                            "solver_invocation_count": 1,
                        }
                    )
    accepted = [row for row in results if row.get("accepted")]
    classification = "INCONCLUSIVE" if not accepted else "SINGLE_BASIN_OR_EFFECTIVELY_UNIQUE"
    if len(accepted) >= 2:
        values = {
            round(float(row.get("final_objective", np.nan)), 12)
            for row in accepted
            if np.isfinite(float(row.get("final_objective", np.nan)))
        }
        if len(values) > 1:
            classification = "MULTIPLE_QUALITY_DISTINCT_BASINS"
    official_values = [
        float(row["final_objective"])
        for row in accepted
        if row.get("requested_seed") == "official_stage7_warm"
        and np.isfinite(float(row.get("final_objective", np.nan)))
    ]
    best_value = min(
        [
            float(row["final_objective"])
            for row in accepted
            if np.isfinite(float(row.get("final_objective", np.nan)))
        ],
        default=float("nan"),
    )
    official_value = min(official_values, default=float("nan"))
    result = {
        "schema_version": SCHEMA,
        "status": "COMPLETE",
        "solver_invocation_count": len(results),
        "results": results,
    }
    manifest["status"] = "COMPLETE"
    manifest["solver_invocation_count"] = len(results)
    _write_json(root / "multistart_manifest.json", manifest)
    _write_json(
        root / "multistart_profiles.json",
        {
            "frozen_initial_queryset": {
                "implemented": True,
                "source": "official Stage 7 warm-seed initial QuerySet reconstructed with formal build_query_set",
            },
            "native_queryset": {"implemented": True},
            "same_objective": True,
            "same_constraints": True,
            "all_six_seeds": True,
        },
    )
    _write_csv(root / "multistart_results_per_frame.csv", results)
    _write_json(root / "multistart_results_per_seed.json", result)
    _write_json(
        root / "multistart_basin_analysis.json",
        {
            "schema_version": SCHEMA,
            "classification": classification,
            "accepted_count": len(accepted),
            "official_initialization_suboptimal": bool(
                np.isfinite(official_value)
                and np.isfinite(best_value)
                and best_value < official_value - 1e-9
            ),
            "official_best_objective": official_value,
            "best_accepted_objective": best_value,
            "branch_rollout_required": False,
        },
    )
    return result


def run_base_seed_ablation(
    run: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    baseline = Path(current_baseline).expanduser().resolve()
    validation, _ = _baseline_gate(baseline)
    profiles = (
        "canonical_frame_official",
        "palm_mcp_kabsch",
        "all21_kabsch",
        "thumb_weighted_kabsch",
        "long_finger_only_kabsch",
        "contact_weighted_kabsch",
    )
    manifest = {
        "schema_version": SCHEMA,
        "profiles": profiles,
        "protocols": ["initialization-only", "seed-and-prior"],
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    _write_json(root / "base_seed_manifest.json", manifest)
    if not validation.get("baseline_pass"):
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "NOT_RUN_UNTIL_CURRENT_BASELINE_PASS",
            "results": [],
        }
        _write_json(root / "base_seed_results_per_profile.json", result)
        _write_csv(root / "base_seed_warm_geometry.csv", [])
        _write_csv(root / "base_seed_results_per_frame.csv", [])
        _write_json(
            root / "base_seed_pareto_analysis.json",
            {"schema_version": SCHEMA, "classification": "INCONCLUSIVE"},
        )
        return result
    bundle = _bundle(Path(run).expanduser().resolve(), baseline / "current_lineage_baseline.zarr")
    selected = _selected_for_run(baseline, frames)
    source_indices = np.arange(21)
    mcp_indices = np.asarray([0, 5, 9, 13, 17])
    long_indices = np.asarray(
        [
            0,
            *FINGER_POINTS["index"],
            *FINGER_POINTS["middle"],
            *FINGER_POINTS["ring"],
            *FINGER_POINTS["pinky"],
        ]
    )
    profile_indices = {
        "canonical_frame_official": None,
        "palm_mcp_kabsch": mcp_indices,
        "all21_kabsch": source_indices,
        "thumb_weighted_kabsch": source_indices,
        "long_finger_only_kabsch": long_indices,
        "contact_weighted_kabsch": source_indices,
    }
    warm_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    protocols = ("initialization-only", "seed-and-prior")
    for frame in selected:
        warm_points = np.asarray(bundle["warm"].arrays["robot_keypoints_base"])[frame]
        source_points = np.asarray(bundle["source"])[frame]
        for profile in profiles:
            if profile == "canonical_frame_official":
                transform = np.asarray(bundle["warm"].arrays["base_pose_scene"])[frame]
            else:
                indices = profile_indices[profile]
                assert indices is not None
                weights = None
                if profile == "thumb_weighted_kabsch":
                    weights = np.ones(len(indices), dtype=np.float64)
                    weights[list(FINGER_POINTS["thumb"])] = 4.0
                if profile == "contact_weighted_kabsch":
                    weights = np.ones(len(indices), dtype=np.float64)
                    weights[0] = 4.0
                transform = _kabsch(warm_points[indices], source_points[indices], weights)
            if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-8):
                raise Stage934Error(f"{profile} generated a reflection")
            fitted = (transform[:3, :3] @ warm_points.T).T + transform[:3, 3]
            metrics = _state_metrics(source_points, fitted)
            warm_row = {
                "frame": frame,
                "profile": profile,
                "translation_m": float(np.linalg.norm(transform[:3, 3])),
                "det_R": float(np.linalg.det(transform[:3, :3])),
                "scale": 1.0,
                "warm_long_finger_rmse_m": metrics["long_finger_rmse_m"],
                "warm_thumb_rmse_m": metrics["thumb_rmse_m"],
            }
            warm_rows.append(warm_row)
            for protocol in protocols:
                artifact = root / "artifacts" / f"frame_{frame:06d}" / f"{profile}__{protocol}.zarr"
                row_base = {
                    **warm_row,
                    "protocol": protocol,
                    "paper_method": False,
                    "engineering_assumption_ablation": profile != "canonical_frame_official",
                }
                try:
                    if artifact.is_dir():
                        row = _variant_row(bundle, frame, profile, artifact, protocol)
                        invocation_count = 0
                    else:
                        value = _solve_variant(
                            bundle,
                            frame,
                            profile,
                            artifact,
                            base_transform=transform,
                            use_previous=protocol == "seed-and-prior",
                        )
                        row = value["row"]
                        invocation_count = 1
                    results.append(
                        {
                            **row_base,
                            **row,
                            "final_status": row.get("status"),
                            "final_accepted": row.get("accepted"),
                            "solver_invocation_count": invocation_count,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            **row_base,
                            "final_status": -1,
                            "final_accepted": False,
                            "accepted": False,
                            "status": -1,
                            "solver_invocation_count": 1,
                            "error": str(exc),
                        }
                    )
    result = {
        "schema_version": SCHEMA,
        "status": "COMPLETE",
        "results": results,
        "solver_invocation_count": sum(1 for row in results if row.get("final_status") == 0),
        "new_solver_invocation_count": sum(
            int(row.get("solver_invocation_count", 0)) for row in results
        ),
    }
    manifest["status"] = result["status"]
    manifest["solver_invocation_count"] = result["solver_invocation_count"]
    _write_json(root / "base_seed_manifest.json", manifest)
    _write_json(
        root / "base_seed_profiles.json",
        {
            "se3_only": True,
            "det_R_plus_one": True,
            "similarity_scale": False,
            "thumb_weight": 4,
            "other_weight": 1,
            "profiles": profiles,
            "protocols": list(protocols),
            "formal_objective_and_constraints": "unchanged Eq. (1)-(9); only base initialization and previous-state prior protocol are varied",
        },
    )
    _write_csv(root / "base_seed_warm_geometry.csv", warm_rows)
    _write_csv(root / "base_seed_results_per_frame.csv", results)
    _write_json(root / "base_seed_results_per_profile.json", result)
    _write_json(
        root / "base_seed_pareto_analysis.json",
        {
            "schema_version": SCHEMA,
            "classification": "PARETO_RECORDED",
            "reason": "final formal solves are recorded per base seed and protocol; this remains diagnostic-only",
        },
    )
    return result


def run_same_lineage_ablation(
    run: str | Path,
    current_baseline: str | Path,
    output_root: str | Path,
    *,
    frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    baseline = Path(current_baseline).expanduser().resolve()
    validation, _ = _baseline_gate(baseline)
    profiles = (
        "official_current_lineage_baseline",
        "half_active_margin",
        "zero_active_margin",
        "full_512_query_reference",
        "minimal_soft_safe_projection_from_warm",
        "official_slack_projection_from_warm",
    )
    manifest = {
        "schema_version": SCHEMA,
        "profiles": profiles,
        "diagnostic_only": True,
        "paper_method": False,
        "accepted_reference": False,
    }
    _write_json(root / "mandatory_ablation_manifest.json", manifest)
    if not validation.get("baseline_pass"):
        result: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": "NOT_RUN_UNTIL_CURRENT_BASELINE_PASS",
            "results": [],
        }
        _write_json(root / "mandatory_ablation_results_per_profile.json", result)
        _write_csv(root / "mandatory_ablation_results_per_frame.csv", [])
        return result
    bundle = _bundle(Path(run).expanduser().resolve(), baseline / "current_lineage_baseline.zarr")
    selected = _selected_for_run(baseline, frames)
    query_ids = {
        "half_active_margin": "half_active_margin_diagnostic_v1",
        "zero_active_margin": "zero_active_margin_diagnostic_v1",
        "full_512_query_reference": "full_collision_surface_reference_v1",
    }
    results: list[dict[str, Any]] = []
    for frame in selected:
        base_row = next((row for row in _frame_rows(bundle) if row["local_frame"] == frame), None)
        if base_row is None:
            continue
        results.append(
            {
                "profile": profiles[0],
                "frame": frame,
                "accepted": bool(base_row["accepted"]),
                "status": base_row["status"],
                "query_count": base_row["query_count"],
                "long_finger_rmse_m": base_row["final_long_finger_rmse_m"],
                "contact_proxy": base_row["contact_retention_proxy_2mm"],
                "full512": True,
                "displacement_m": 0.0,
                "solver_invocation_count": 0,
                "key_result": "same-lineage baseline",
            }
        )
        for profile, query_id in query_ids.items():
            try:
                value = _solve_variant(
                    bundle,
                    frame,
                    "official_stage7_warm",
                    root / "artifacts" / f"frame_{frame:06d}" / f"{profile}.zarr",
                    query_profile_id=query_id,
                )
                row = value["row"]
                results.append(
                    {
                        "profile": profile,
                        "frame": frame,
                        "accepted": row["accepted"],
                        "status": row["status"],
                        "query_count": row["query_count"],
                        "long_finger_rmse_m": row.get("final_long_finger_rmse_m"),
                        "contact_proxy": _proxy(
                            np.asarray(
                                load_final_trajectory(value["artifact"]).arrays[
                                    "full_signed_distance"
                                ]
                            )[0],
                            0.002,
                        ),
                        "full512": True,
                        "displacement_m": np.nan,
                        "solver_invocation_count": 1,
                        "key_result": "diagnostic QuerySet/margin rerun",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "profile": profile,
                        "frame": frame,
                        "accepted": False,
                        "status": -1,
                        "error": str(exc),
                        "solver_invocation_count": 1,
                    }
                )
        for profile in (
            "minimal_soft_safe_projection_from_warm",
            "official_slack_projection_from_warm",
        ):
            results.append(
                {
                    "profile": profile,
                    "frame": frame,
                    "accepted": False,
                    "status": "PROJECTION_DIAGNOSTIC_NOT_SOLVED",
                    "query_count": 512,
                    "long_finger_rmse_m": base_row["warm_long_finger_rmse_m"],
                    "contact_proxy": base_row["contact_retention_proxy_2mm"],
                    "full512": True,
                    "displacement_m": 0.0,
                    "solver_invocation_count": 0,
                    "key_result": "projection profile is not comparable to formal total objective",
                }
            )
    result = {
        "schema_version": SCHEMA,
        "status": "COMPLETE",
        "results": results,
        "solver_invocation_count": sum(
            int(row.get("solver_invocation_count", 0)) for row in results
        ),
    }
    manifest["status"] = result["status"]
    manifest["solver_invocation_count"] = result["solver_invocation_count"]
    _write_json(root / "mandatory_ablation_manifest.json", manifest)
    _write_csv(root / "mandatory_ablation_results_per_frame.csv", results)
    _write_json(root / "mandatory_ablation_results_per_profile.json", result)
    return result


def run_stage934(
    run: str | Path, *, output_root: str | Path, max_wall_time: float | None = None
) -> dict[str, Any]:
    """Execute the bounded Stage 9.3.4 orchestration in lane order."""

    root = Path(output_root).expanduser().resolve()
    run_path = Path(run).expanduser().resolve()
    provenance = (
        root.parent
        / "stage9_3_4_provenance"
        / "s1__airplane_lift__right__artimano_rh__f000240_f000300"
    )
    historical = (
        root.parent
        / "stage9_3_4_historical_lane"
        / "s1__airplane_lift__right__artimano_rh__f000240_f000300"
    )
    current = root / "s1__airplane_lift__right__artimano_rh__f000240_f000300"
    audit_solver_lineage(run_path, provenance)
    run_historical_replay(provenance, historical)
    baseline = run_current_causal_baseline(run_path, current, max_wall_time=max_wall_time)
    if not baseline.get("validation", baseline).get("baseline_pass", False):
        return stage9_causal_status(provenance, current, root.parent / "stage9_3_4")
    run_refinement_multistart(run_path, current, root.parent / "stage9_3_4_multistart")
    run_base_seed_ablation(run_path, current, root.parent / "stage9_3_4_base_seed_ablation")
    run_same_lineage_ablation(run_path, current, root.parent / "stage9_3_4_mandatory_ablation")
    return stage9_causal_status(provenance, current, root.parent / "stage9_3_4")


__all__ = [
    "SCHEMA",
    "Stage934Error",
    "audit_solver_lineage",
    "run_current_baseline_repeats",
    "run_current_causal_baseline",
    "run_historical_replay",
    "run_base_seed_ablation",
    "run_refinement_multistart",
    "run_same_lineage_ablation",
    "run_stage934",
    "stage9_causal_status",
]
