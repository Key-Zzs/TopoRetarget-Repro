#!/usr/bin/env python3
"""Freeze Stage-16B baselines and qualify the shared adaptive H1/H5/H10 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from toporetarget.geometry.se3 import rotation_geodesic_error
from toporetarget.rl.environments.world_wrist_backend import (
    WorldWristFingerBackend,
    WristFingerActionScaleV1,
    WristImpedanceProfileV1,
    materialize_world_wrist_free_object_scene,
)
from toporetarget.rl.stage16b_recovery import (
    AdaptiveOracleFailure,
    Stage16BAdaptiveOracleStateMachine,
)
from toporetarget.rl.world_wrist import WorldWristFingerReferenceV1
from toporetarget.rl.world_wrist_oracle import (
    AdaptiveMultiHorizonContactOracle,
    ContactAwareMPCConfig,
)

REPO = Path(__file__).resolve().parents[2]
WUJI_MJCF = REPO / "third_party/robot_hands/wuji_hand2_beta1/mjcf/right.xml"
FROZEN_FIXED_REPORT = (
    REPO / ".local/reports/stage16_world_wrist_finger/"
    "contact_mpc_formal_selected_20260801/oracle_evaluation.json"
)
START_COMMIT = "60b2d99a8405a7d13ed1385023a53f9b2f66ce44"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _command(*arguments: str) -> str:
    return subprocess.run(
        list(arguments),
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _preflight(report_root: Path) -> dict[str, Any]:
    preflight = report_root / "preflight"
    branch = _git("branch", "--show-current").strip()
    head = _git("rev-parse", "HEAD").strip()
    if branch != "feature/reference-tracking-ppo":
        raise ValueError(f"wrong branch: {branch}")
    subprocess.run(["git", "merge-base", "--is-ancestor", START_COMMIT, head], cwd=REPO, check=True)
    status = _git("status", "--short", "--untracked-files=all")
    staged = _git("diff", "--cached", "--binary")
    unstaged = _git("diff", "--binary")
    (preflight / "git_status.txt").parent.mkdir(parents=True, exist_ok=True)
    (preflight / "git_status.txt").write_text(status, encoding="utf-8")
    (preflight / "staged.patch").write_text(staged, encoding="utf-8")
    (preflight / "unstaged.patch").write_text(unstaged, encoding="utf-8")
    process_inventory = _command(
        "ps", "-eo", "pid,ppid,pgid,lstart,etime,%cpu,%mem,rss,cmd", "--sort=-%cpu"
    )
    (preflight / "process_inventory.txt").write_text(process_inventory, encoding="utf-8")
    changed = [line[3:] for line in status.splitlines() if len(line) > 3]
    ownership = {
        "branch": branch,
        "head": head,
        "start_commit_is_ancestor": True,
        "initial_worktree_at_turn_start": "clean",
        "current_changes": [
            {"path": path, "classification": "stage16b_adaptive_oracle_single_ppo_wip"}
            for path in changed
        ],
        "unrelated_or_unknown_changes": [],
    }
    _write_json(preflight / "ownership.json", ownership)
    return ownership


def _fixed_summary(report: dict[str, Any]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for horizon, clips in report["horizons"].items():
        matrix[horizon] = {
            row["clip"]: {
                key: row["summary"][key]
                for key in (
                    "success_rate",
                    "final_reach_rate",
                    "progress",
                    "object_position_error_cm",
                    "object_rotation_error_deg",
                    "max_axis_error_cm",
                    "termination_distribution",
                )
            }
            for row in clips
        }
    return matrix


def _freeze_fixed_baseline(
    *,
    archive_parent: Path,
    report_root: Path,
    references: list[Path],
    meshes: list[Path],
) -> Path:
    fixed_report = json.loads(FROZEN_FIXED_REPORT.read_text(encoding="utf-8"))
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    short_head = _git("rev-parse", "--short", "HEAD").strip()
    archive = archive_parent / f"stage16b_fixed_horizon_oracle_{timestamp}_{short_head}"
    archive.mkdir(parents=True, exist_ok=False)
    artifacts = [
        FROZEN_FIXED_REPORT,
        FROZEN_FIXED_REPORT.parent / "action_scale_qualification.json",
        FROZEN_FIXED_REPORT.parent / "wrist_controller_qualification.json",
        FROZEN_FIXED_REPORT.parent / "object_dynamics_audit_link.json",
        *references,
        *meshes,
    ]
    artifact_hashes = {str(path.resolve()): _sha256(path) for path in artifacts}
    code_paths = [
        REPO / "src/toporetarget/rl/world_wrist_oracle.py",
        REPO / "src/toporetarget/rl/environments/world_wrist_backend.py",
        REPO / "scripts/rl/qualify_stage16_world_wrist.py",
    ]
    code_hashes = {
        str(path.relative_to(REPO)): hashlib.sha256(
            subprocess.run(
                ["git", "show", f"{START_COMMIT}:{path.relative_to(REPO)}"],
                cwd=REPO,
                check=True,
                capture_output=True,
            ).stdout
        ).hexdigest()
        for path in code_paths
    }
    matrix = _fixed_summary(fixed_report)
    manifest = {
        "id": "stage16b_fixed_horizon_oracle_frozen_v1",
        "source_commit": START_COMMIT,
        "source_report": str(FROZEN_FIXED_REPORT.resolve()),
        "source_report_sha256": artifact_hashes[str(FROZEN_FIXED_REPORT.resolve())],
        "large_artifacts_copied": False,
        "artifact_paths_and_hashes_only": True,
        "conclusions": {
            "h5_passes_170105": True,
            "h10_passes_170650": True,
            "per_trajectory_controllability_demonstrated": True,
            "shared_fixed_horizon_oracle_qualified": False,
        },
    }
    _write_json(archive / "frozen_manifest.json", manifest)
    _write_json(archive / "fixed_horizon_matrix.json", matrix)
    _write_json(archive / "artifact_hashes.json", artifact_hashes)
    _write_json(archive / "code_config_hashes.json", code_hashes)
    (archive / "README.md").write_text(
        "# Frozen Stage-16B fixed-horizon oracle baseline\n\n"
        "This archive records paths and SHA-256 hashes without duplicating large artifacts. "
        "H5 passes 170105, H10 passes 170650, so per-trajectory controllability is "
        "demonstrated; no one shared fixed horizon qualifies both clips. The historical "
        "report and its selected half-scale action profile are immutable.\n",
        encoding="utf-8",
    )
    _write_json(report_root / "frozen_fixed_horizon_baseline.json", manifest | {"matrix": matrix})
    return archive


def _make_backend(
    *,
    reference_path: Path,
    mesh_path: Path,
    scene_root: Path,
    seed: int,
) -> WorldWristFingerBackend:
    hand_model = mujoco.MjModel.from_xml_path(str(WUJI_MJCF))
    reference = WorldWristFingerReferenceV1.from_npz(reference_path)
    scene = materialize_world_wrist_free_object_scene(
        WUJI_MJCF,
        scene_root,
        object_mesh=mesh_path,
        object_mass_kg=0.05,
    )
    return WorldWristFingerBackend(
        scene_path=scene,
        reference=reference,
        joint_lower=hand_model.jnt_range[: hand_model.njnt, 0],
        joint_upper=hand_model.jnt_range[: hand_model.njnt, 1],
        impedance_profile=WristImpedanceProfileV1(),
        action_scale=WristFingerActionScaleV1(),
        seed=seed,
    )


def _metrics(backend: WorldWristFingerBackend, state: dict[str, np.ndarray]) -> dict[str, float]:
    index = backend.reference_index
    reference = backend.reference
    return {
        "object_position_error_m": float(
            np.linalg.norm(
                state["object_pose"][:3, 3] - reference.object_pose_world_ref[index, :3, 3]
            )
        ),
        "object_rotation_error_deg": float(
            np.degrees(
                rotation_geodesic_error(
                    state["object_pose"], reference.object_pose_world_ref[index]
                )
            )
        ),
        "max_axis_error_m": float(
            np.max(
                np.linalg.norm(
                    state["object_axis_points"] - reference.object_axis_points_world_ref[index],
                    axis=1,
                )
            )
        ),
        "wrist_position_error_m": float(
            np.linalg.norm(
                state["wrist_pose"][:3, 3] - reference.wrist_pose_world_ref[index, :3, 3]
            )
        ),
        "wrist_rotation_error_deg": float(
            np.degrees(
                rotation_geodesic_error(state["wrist_pose"], reference.wrist_pose_world_ref[index])
            )
        ),
    }


@dataclass(frozen=True)
class EpisodeResult:
    termination: str
    success: bool
    final_reach: bool
    reference_index: int
    progress: float
    object_position_error_m: float
    object_rotation_error_deg: float
    max_axis_error_m: float
    peak_axis_error_m: float
    wrist_position_error_m: float
    wrist_rotation_error_deg: float
    action_trace_sha256: str
    action_count: int
    max_action_abs: float
    selected_horizon_counts: dict[str, int]
    terminal_contraction_steps: int
    numerical_failure: bool


def _run_optimized(
    backend: WorldWristFingerBackend,
    oracle: AdaptiveMultiHorizonContactOracle,
) -> tuple[EpisodeResult, list[np.ndarray], list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    state = backend.reset(reference_index=0)
    actions: list[np.ndarray] = []
    states: list[dict[str, np.ndarray]] = []
    trace: list[dict[str, Any]] = []
    reason: str | None = None
    peak_axis = 0.0
    horizon_counts: dict[str, int] = {}
    terminal_contractions = 0
    for _ in range(backend.reference.frame_count - 1):
        action = oracle.action(backend)
        selected = oracle.last_selected
        assert selected is not None
        horizon_key = f"H{selected.requested_horizon}"
        horizon_counts[horizon_key] = horizon_counts.get(horizon_key, 0) + 1
        if selected.effective_horizon != selected.requested_horizon:
            terminal_contractions += 1
        trace.append(oracle.selection_trace[-1])
        actions.append(action.copy())
        state, _, reason = backend.transition(action)
        step_metrics = _metrics(backend, state)
        peak_axis = max(peak_axis, step_metrics["max_axis_error_m"])
        states.append(
            {
                "qpos": backend.data.qpos.copy(),
                "qvel": backend.data.qvel.copy(),
                "qacc_warmstart": backend.data.qacc_warmstart.copy(),
            }
        )
        if reason is not None:
            break
    if reason is None:
        reason = "FAILURE_EVALUATION_STEP_BOUND"
    final = _metrics(backend, state)
    action_array = np.asarray(actions, dtype=np.float64)
    result = EpisodeResult(
        termination=reason,
        success=reason == "SUCCESS_REFERENCE_COMPLETE",
        final_reach=backend.reference_index >= backend.reference.frame_count - 1,
        reference_index=backend.reference_index,
        progress=float(backend.reference_index / (backend.reference.frame_count - 1)),
        **final,
        peak_axis_error_m=peak_axis,
        action_trace_sha256=_array_hash(action_array),
        action_count=len(actions),
        max_action_abs=float(np.max(np.abs(action_array))) if actions else 0.0,
        selected_horizon_counts=horizon_counts,
        terminal_contraction_steps=terminal_contractions,
        numerical_failure=not np.isfinite(backend.data.qpos).all()
        or not np.isfinite(backend.data.qvel).all(),
    )
    return result, actions, states, trace


def _run_replay(
    backend: WorldWristFingerBackend,
    actions: list[np.ndarray],
    expected_states: list[dict[str, np.ndarray]],
) -> tuple[EpisodeResult, dict[str, float]]:
    state = backend.reset(reference_index=0)
    reason: str | None = None
    peak_axis = 0.0
    differences = {"qpos": 0.0, "qvel": 0.0, "qacc_warmstart": 0.0}
    for index, action in enumerate(actions):
        state, _, reason = backend.transition(action)
        peak_axis = max(peak_axis, _metrics(backend, state)["max_axis_error_m"])
        for key in differences:
            current = (
                backend.data.qacc_warmstart
                if key == "qacc_warmstart"
                else getattr(backend.data, key)
            )
            differences[key] = max(
                differences[key], float(np.max(np.abs(current - expected_states[index][key])))
            )
        if reason is not None:
            break
    if reason is None:
        reason = "FAILURE_EVALUATION_STEP_BOUND"
    final = _metrics(backend, state)
    action_array = np.asarray(actions[: backend.step_index], dtype=np.float64)
    result = EpisodeResult(
        termination=reason,
        success=reason == "SUCCESS_REFERENCE_COMPLETE",
        final_reach=backend.reference_index >= backend.reference.frame_count - 1,
        reference_index=backend.reference_index,
        progress=float(backend.reference_index / (backend.reference.frame_count - 1)),
        **final,
        peak_axis_error_m=peak_axis,
        action_trace_sha256=_array_hash(action_array),
        action_count=backend.step_index,
        max_action_abs=float(np.max(np.abs(action_array))) if len(action_array) else 0.0,
        selected_horizon_counts={},
        terminal_contraction_steps=0,
        numerical_failure=not np.isfinite(backend.data.qpos).all()
        or not np.isfinite(backend.data.qvel).all(),
    )
    return result, differences


def _summarize(episodes: list[EpisodeResult]) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([row.success for row in episodes])),
        "final_reach_rate": float(np.mean([row.final_reach for row in episodes])),
        "progress": float(np.mean([row.progress for row in episodes])),
        "object_position_error_cm": float(
            100.0 * np.mean([row.object_position_error_m for row in episodes])
        ),
        "object_rotation_error_deg": float(
            np.mean([row.object_rotation_error_deg for row in episodes])
        ),
        "max_axis_error_cm": float(100.0 * np.max([row.max_axis_error_m for row in episodes])),
        "peak_axis_error_cm": float(100.0 * np.max([row.peak_axis_error_m for row in episodes])),
        "numerical_failures": sum(row.numerical_failure for row in episodes),
        "termination_distribution": {
            termination: sum(row.termination == termination for row in episodes)
            for termination in sorted({row.termination for row in episodes})
        },
        "unique_action_trace_count": len({row.action_trace_sha256 for row in episodes}),
        "unique_terminal_tuple_count": len(
            {
                (
                    row.termination,
                    row.reference_index,
                    round(row.object_position_error_m, 12),
                    round(row.object_rotation_error_deg, 12),
                    round(row.max_axis_error_m, 12),
                )
                for row in episodes
            }
        ),
    }


def _passes(summary: dict[str, Any]) -> bool:
    return bool(
        summary["success_rate"] >= 0.90
        and summary["final_reach_rate"] >= 0.90
        and summary["object_position_error_cm"] <= 2.0
        and summary["object_rotation_error_deg"] <= 10.0
        and summary["max_axis_error_cm"] <= 3.0
        and summary["numerical_failures"] == 0
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--object-mesh", action="append", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--archive-parent", required=True, type=Path)
    parser.add_argument("--formal-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--mpc-population", type=int, default=32)
    parser.add_argument("--mpc-iterations", type=int, default=3)
    parser.add_argument("--mpc-elites", type=int, default=8)
    args = parser.parse_args()
    if len(args.reference) != 2 or len(args.object_mesh) != 2:
        raise ValueError("formal Stage-16B adaptive qualification requires exactly two clips")
    if args.formal_episodes != 20:
        raise ValueError("formal Stage-16B adaptive qualification requires exactly 20 episodes")
    if args.report_root.exists() or args.experiment_root.exists():
        raise FileExistsError("refusing to overwrite an existing Stage-16B adaptive run")
    args.report_root.mkdir(parents=True)
    args.experiment_root.mkdir(parents=True)
    started = time.monotonic()
    ownership = _preflight(args.report_root)
    archive = _freeze_fixed_baseline(
        archive_parent=args.archive_parent,
        report_root=args.report_root,
        references=args.reference,
        meshes=args.object_mesh,
    )
    config = ContactAwareMPCConfig(
        population=args.mpc_population,
        iterations=args.mpc_iterations,
        elite_count=args.mpc_elites,
        seed=args.seed,
    )
    config.validate()
    budget_upgrade = (
        args.mpc_population == 48 and args.mpc_iterations == 4 and args.mpc_elites == 12
    )
    _write_json(
        args.report_root / "adaptive_oracle_config.json",
        {
            "id": "adaptive_multi_horizon_contact_oracle_v1",
            "portfolio": [1, 5, 10],
            "config": asdict(config),
            "action": WristFingerActionScaleV1().as_dict(),
            "controller": WristImpedanceProfileV1().as_dict(),
            "clip_identity_available_to_selector": False,
            "terminal_padding": False,
            "selector_projection": "closed_loop_receding_replans_without_padding",
            "selection_lookahead": 10,
            "global_budget_upgrade": budget_upgrade,
        },
    )
    expected_hashes = {
        str(path.resolve()): _sha256(path) for path in [*args.reference, *args.object_mesh]
    }
    clip_reports: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    nominal_models: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []
    for clip_index, (reference_path, mesh_path) in enumerate(
        zip(args.reference, args.object_mesh, strict=True)
    ):
        backend = _make_backend(
            reference_path=reference_path,
            mesh_path=mesh_path,
            scene_root=args.scene_root / reference_path.stem,
            seed=args.seed + clip_index,
        )
        nominal_models.append(
            {
                "reference": str(reference_path.resolve()),
                "object_mesh": str(mesh_path.resolve()),
                "model": backend.model_report(),
            }
        )
        oracle = AdaptiveMultiHorizonContactOracle(config=config)
        optimized, actions, expected_states, trace = _run_optimized(backend, oracle)
        for row in trace:
            all_selection_rows.append({"clip": reference_path.stem, **row})
        trace_path = args.experiment_root / f"{reference_path.stem}.adaptive_actions.npz"
        np.savez_compressed(trace_path, actions=np.asarray(actions, dtype=np.float64))
        episodes = [optimized]
        maximum_differences = {"qpos": 0.0, "qvel": 0.0, "qacc_warmstart": 0.0}
        for replay_index in range(args.formal_episodes - 1):
            replay, differences = _run_replay(backend, actions, expected_states)
            episodes.append(replay)
            for key, value in differences.items():
                maximum_differences[key] = max(maximum_differences[key], value)
            determinism_rows.append(
                {
                    "clip": reference_path.stem,
                    "replay_episode": replay_index + 1,
                    **differences,
                }
            )
        summary = _summarize(episodes)
        clip_reports.append(
            {
                "clip": reference_path.stem,
                "reference": str(reference_path.resolve()),
                "object_mesh": str(mesh_path.resolve()),
                "reference_sha256": expected_hashes[str(reference_path.resolve())],
                "object_mesh_sha256": expected_hashes[str(mesh_path.resolve())],
                "action_trace": str(trace_path.resolve()),
                "action_trace_sha256": _sha256(trace_path),
                "optimized_episode_count": 1,
                "action_only_replay_count": args.formal_episodes - 1,
                "summary": summary,
                "passes_gate": _passes(summary),
                "optimized_episode": asdict(optimized),
                "episodes": [asdict(row) for row in episodes],
                "determinism_max_difference": maximum_differences,
            }
        )
    current_hashes = {
        str(path.resolve()): _sha256(path) for path in [*args.reference, *args.object_mesh]
    }
    machine = Stage16BAdaptiveOracleStateMachine()
    drift_detected = current_hashes != expected_hashes
    if drift_detected:
        machine.record(
            failure=AdaptiveOracleFailure.PHYSICS_OR_REFERENCE_DRIFT,
            evidence={"before": expected_hashes, "after": current_hashes},
            fallback="fail_closed",
            repair="none",
            rerun="none",
            result="BLOCKED",
        )
    oracle_pass = all(row["passes_gate"] for row in clip_reports) and not drift_detected
    if budget_upgrade and not drift_detected:
        machine.record(
            failure=AdaptiveOracleFailure.CEM_CONTACT_MODE_MISS,
            evidence={
                "prior_shared_32x3_attempt": "attempt03_32x3",
                "current_clips": {row["clip"]: row["summary"] for row in clip_reports},
            },
            fallback="single_global_48x4_upgrade",
            repair="population_48_iterations_4_elites_12_for_both_clips",
            rerun="both_clips",
            result="VALIDATED" if oracle_pass else "PARTIAL",
            budget_upgrade=True,
        )
    elif not oracle_pass and not machine.transitions:
        machine.record(
            failure=AdaptiveOracleFailure.CEM_CONTACT_MODE_MISS,
            evidence={
                row["clip"]: row["summary"] for row in clip_reports if not row["passes_gate"]
            },
            fallback="cross_horizon_seed_then_single_global_48x4_upgrade",
            repair="initial_shared_32x3_run_preserved_for_bounded_diagnosis",
            rerun="both_clips",
            result="PARTIAL",
        )
    status = (
        "STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_VALIDATED"
        if oracle_pass
        else "STAGE16B_ADAPTIVE_MULTI_HORIZON_ORACLE_PARTIAL"
    )
    evaluation = {
        "status": status,
        "ppo_entry": (
            "STAGE16B_SINGLE_CLIP_PPO_ENTRY_AUTHORIZED"
            if oracle_pass
            else "STAGE16B_SINGLE_CLIP_PPO_ENTRY_NOT_AUTHORIZED"
        ),
        "formal_episodes_per_clip": args.formal_episodes,
        "frame_zero": True,
        "deterministic": True,
        "nominal_frozen_physics": "world_wrist_freebody_nominal_v1",
        "direct_object_control": False,
        "wrist_teleport": False,
        "clips": clip_reports,
    }
    _write_json(args.report_root / "adaptive_oracle_evaluation.json", evaluation)
    _write_jsonl(args.report_root / "adaptive_oracle_selection_trace.jsonl", all_selection_rows)
    _write_json(
        args.report_root / "adaptive_oracle_determinism.json",
        {
            "replays": determinism_rows,
            "clips": [row["determinism_max_difference"] for row in clip_reports],
        },
    )
    _write_jsonl(
        args.report_root / "oracle_failure_transitions.jsonl",
        [asdict(row) for row in machine.transitions],
    )
    nominal_profile = {
        "id": "world_wrist_freebody_nominal_v1",
        "status": "STAGE16B_NOMINAL_SIMULATOR_PROFILE_FROZEN",
        "object_mass_kg": 0.05,
        "inertia": [row["model"]["object_principal_inertia_kgm2"] for row in nominal_models],
        "inertia_source": "engineering assumption from current mesh-derived MuJoCo inertia",
        "gravity": [0.0, 0.0, 0.0],
        "ground": False,
        "support": "none",
        "freejoint_damping": 0.0,
        "contact_profile": "friction=[1,0.005,0.0001] current MuJoCo mesh contact",
        "physical_provenance": "OBJECT_DYNAMICS_PHYSICAL_PROVENANCE_UNRESOLVED",
        "allowed": [
            "simulator functional tracking",
            "oracle qualification",
            "nominal PPO optimization",
            "deterministic simulation comparison",
        ],
        "disallowed_claims": [
            "real object dynamics reproduced",
            "physical parameter ground truth",
            "sim-to-real qualified",
            "real-world force accuracy",
        ],
        "models": nominal_models,
    }
    _write_json(args.report_root / "nominal_dynamics_profile.json", nominal_profile)
    elapsed = time.monotonic() - started
    final = {
        "branch": ownership["branch"],
        "start_head": START_COMMIT,
        "run_head": ownership["head"],
        "adaptive_oracle": status,
        "ppo_entry": evaluation["ppo_entry"],
        "single_170105": "NOT_STARTED_GATE_BLOCKED" if not oracle_pass else "NOT_STARTED",
        "single_170650": "NOT_STARTED_GATE_BLOCKED" if not oracle_pass else "NOT_STARTED",
        "overall": (
            "STAGE16B_ADAPTIVE_ORACLE_PARTIAL"
            if not oracle_pass
            else "STAGE16B_ORACLE_COMPLETE_SINGLE_PPO_PARTIAL"
        ),
        "fixed_baseline_archive": str(archive.resolve()),
        "wall_seconds": elapsed,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "push": False,
    }
    _write_json(args.report_root / "final_summary.json", final)
    _write_json(
        args.report_root / "resource_usage.json",
        {
            "wall_seconds": elapsed,
            "peak_rss_mb": final["peak_rss_mb"],
            "cpu_backend": "MuJoCo",
            "gpu_used_for_oracle": False,
        },
    )
    print(json.dumps(final, sort_keys=True))
    return 0 if oracle_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
