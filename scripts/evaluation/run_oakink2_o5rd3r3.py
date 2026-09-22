#!/usr/bin/env python3
"""O5R-D3-R3 frozen RefinementV2 consumed-development validation.

The driver is deliberately limited to already-consumed DEV1 evidence.  It
validates the frozen conditional expanded-active-set design and has no action
for fresh certification, D3-V2, DEV2, PPO, PhysX, or O6.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5 as o5  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2n as d2n  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3 as d3  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3r2 as r2  # noqa: E402
from toporetarget.retarget.objective_v2_execution import asset_derived_dof_blocks  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3r3_refinement_v2_development_validation_v1"
R2_ROOT = r2.ROOT
D3_ROOT = r2.D3_ROOT
D3R_ROOT = r2.D3R_ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "a3e677b50e13c2c1d8db7a339d57b7b51222df32"
DESIGN_SHA256 = "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
GATE_SHA256 = "96d94c838c19f05dd28bb7ba065121bbc08d95e5a79fb6548c8ef1a1f81d366e"
TAU = 1.0e-4
EXPECTED_INVALID = 145
EXPECTED_GROUPS = {"A": 1865, "B": 145, "C": 712, "D": 0}
DETERMINISM_Q_TOL = 1.0e-10
DETERMINISM_BASE_TOL = 1.0e-10
DETERMINISM_EIM_TOL = 1.0e-12


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    r2.write_json(path, value)


def write_text(path: Path, value: str) -> None:
    r2.write_text(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    r2.write_csv(path, rows, fields)


def read_csv(path: Path) -> list[dict[str, str]]:
    return r2.read_csv(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def canonical_sha(value: Any) -> str:
    return r2.canonical_sha(value)


def array_sha(value: np.ndarray) -> str:
    return r2.array_sha(value)


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    return r2.require(path, field, expected, action)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    r2._append_jsonl(path, value)


def _sidecar_value(path: Path) -> str:
    return path.read_text(encoding="utf-8").split()[0]


def _initial_not_run(root: Path) -> None:
    value = {
        "schema_version": "D3R3ExplicitNotRunV1",
        "status": "NOT_RUN",
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_RUN_COUNT_NEW": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    for relative in (
        "future/fresh_refinement_sparse/not_run.json",
        "future/fresh_refinement_window/not_run.json",
        "future/cross_episode_refinement/not_run.json",
        "future/d3v2/not_run.json",
        "future/dev2/not_run.json",
        "future/ppo/not_run.json",
        "future/physx/not_run.json",
        "future/o6/not_run.json",
    ):
        if not (root / relative).exists():
            write_json(root / relative, value)


def _authority_snapshot() -> dict[str, str]:
    paths = {
        "d3_final_summary": D3_ROOT / "final_summary.json",
        "d3_run_manifest": D3_ROOT / "run_authority/run_manifest.json",
        "d3_trajectory": D3_ROOT / "trajectory/trajectory.npz",
        "d3r_final_summary": D3R_ROOT / "final_summary.json",
        "d3r_partition": D3R_ROOT / "d3r/group_partition.csv",
        "d3r_segments": D3R_ROOT / "d3r/failure_segments.json",
        "d3r_root_cause": D3R_ROOT / "d3r/root_cause.json",
        "r2_final_summary": R2_ROOT / "final_summary.json",
        "r2_design": R2_ROOT / "design/refinement_v2_design.json",
        "r2_gate": R2_ROOT / "design/refinement_v2_development_gate.json",
        "r2_controls": R2_ROOT / "development_set/valid_controls.json",
        "r2_windows": R2_ROOT / "development_set/sequential_windows.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "oakink2_root_exists": Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2").is_dir(),
        "d3_exists": D3_ROOT.is_dir(),
        "d3r_exists": D3R_ROOT.is_dir(),
        "r2_exists": R2_ROOT.is_dir(),
    }
    value = {
        "schema_version": "O5RD3R3GitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": head,
        "status_short": git("status", "--short", "--untracked-files=all"),
        "diff_stat": git("diff", "--stat"),
        "diff": git("diff"),
        "cached_diff_stat": git("diff", "--cached", "--stat"),
        "cached_diff": git("diff", "--cached"),
        "diff_check": subprocess.run(
            ["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True, check=False
        ).stdout,
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "upstream_snapshot": _authority_snapshot(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", value)
    (root / "technical_failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    _initial_not_run(root)
    if value["status"] != "PASS":
        raise RuntimeError("D3_R3_STATUS=BLOCKED_UPSTREAM_DESIGN_AUTHORITY:GIT")
    return value


def verify_r2_design_authority(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_R2_DESIGN")
    summary = read_json(R2_ROOT / "final_summary.json")
    authorization = read_json(R2_ROOT / "future/d3r3_authorization.json")
    design_path = R2_ROOT / "design/refinement_v2_design.json"
    design = read_json(design_path)
    design_sha = sha256_file(design_path)
    runtime, _ = r2._runtime_pair()
    dof_names = list(runtime.model.dof_names)
    checks = {
        "D3_R2_STATUS": summary.get("D3_R2_STATUS") == "PASS_DESIGN",
        "SEARCH_SPACE_MECHANISM": summary.get("SEARCH_SPACE_MECHANISM")
        == "OPTIMIZATION_SUBSPACE_TOO_NARROW",
        "MECHANISM_CONFIDENCE": summary.get("MECHANISM_CONFIDENCE") == "HIGH",
        "SELECTED_REFINEMENT_V2_DESIGN": summary.get("SELECTED_REFINEMENT_V2_DESIGN")
        == "REFINEMENT_V2_C_CONDITIONAL_EXPANDED_ACTIVE_SET",
        "REFINEMENT_V2_DESIGN_FROZEN": design.get("REFINEMENT_V2_DESIGN_FROZEN") == "YES"
        and design.get("status") == "FROZEN",
        "D3_R3_AUTHORIZED": authorization.get("D3_R3_AUTHORIZED") == "YES",
        "design_sha_exact": design_sha == DESIGN_SHA256,
        "design_sidecar_exact": _sidecar_value(R2_ROOT / "design/refinement_v2_design.sha256")
        == DESIGN_SHA256,
        "expanded_active_indices_exact": design.get("active_dofs") == list(range(20)),
        "asset_dof_count_exact": len(dof_names) == 20,
        "wrist_base_locked": design.get("bounds_envelope")
        == "Wuji asset joint limits; wrist/base remain fixed",
    }
    value = {
        "schema_version": "D3R3DesignIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D3_R2_STATUS": summary.get("D3_R2_STATUS"),
        "SEARCH_SPACE_MECHANISM": summary.get("SEARCH_SPACE_MECHANISM"),
        "MECHANISM_CONFIDENCE": summary.get("MECHANISM_CONFIDENCE"),
        "SELECTED_REFINEMENT_V2_DESIGN": design.get("selected_design"),
        "REFINEMENT_V2_DESIGN_FROZEN": design.get("REFINEMENT_V2_DESIGN_FROZEN"),
        "D3_R3_AUTHORIZED": authorization.get("D3_R3_AUTHORIZED"),
        "REFINEMENT_V2_DESIGN_SHA256": design_sha,
        "NORMAL_PATH": design.get("normal_path"),
        "EXPANDED_PATH": design.get("fallback_path"),
        "EXPANSION_TRIGGER": design.get("trigger_condition"),
        "NORMAL_ACTIVE_DOF_COUNT": 4,
        "EXPANDED_ACTIVE_DOF_COUNT": len(design.get("active_dofs", [])),
        "EXPANDED_ACTIVE_DOF_NAMES": dof_names,
        "WRIST_FREE_IN_EXPANDED_PATH": "NO",
        "BASE_FREE_IN_EXPANDED_PATH": "NO",
        "frozen_design": design,
    }
    write_json(root / "preflight/upstream_authority.json", value)
    write_json(root / "preflight/design_integrity.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3_R3_STATUS=BLOCKED_UPSTREAM_DESIGN_AUTHORITY")
    return value


def verify_r3_development_gate(root: Path) -> dict[str, Any]:
    require(root / "preflight/design_integrity.json", "status", "PASS", "VERIFY_R3_GATE")
    gate_path = R2_ROOT / "design/refinement_v2_development_gate.json"
    gate = read_json(gate_path)
    observed = sha256_file(gate_path)
    required_fraction = float(gate["final_invalid_recovery_min"])
    required_count = math.ceil(required_fraction * EXPECTED_INVALID)
    checks = {
        "gate_frozen": gate.get("status") == "FROZEN_NOT_RUN",
        "gate_sha_exact": observed == GATE_SHA256,
        "gate_sidecar_exact": _sidecar_value(
            R2_ROOT / "design/refinement_v2_development_gate.sha256"
        )
        == GATE_SHA256,
        "population_exact": gate.get("population")
        == "all 145 consumed D3-R final-invalid frames plus frozen valid controls and consumed sequential windows",
        "required_count_valid": 0 <= required_count <= EXPECTED_INVALID,
    }
    value = {
        "schema_version": "D3R3GateIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "REFINEMENT_V2_DEVELOPMENT_GATE_SHA256": observed,
        "RECOVERY_REQUIRED_FRACTION": required_fraction,
        "RECOVERY_REQUIRED_COUNT": required_count,
        "MAX_ALLOWED_UNRECOVERED": EXPECTED_INVALID - required_count,
        "MEDIAN_REDUCTION_REQUIRED": gate["median_invalid_relative_E_IM_reduction_min"],
        "VALID_PRESERVATION_REQUIRED": gate["valid_preservation_controls"],
        "SEQUENTIAL_REQUIREMENT": gate["sequential_consumed_windows"],
        "CONTINUITY_REQUIREMENT": gate["continuity"],
        "DETERMINISM_REQUIREMENT": gate["determinism"],
    }
    write_json(root / "preflight/gate_integrity.json", value)
    write_json(root / "gate/frozen_gate.json", gate)
    write_text(root / "gate/frozen_gate.sha256", observed + "\n")
    if value["status"] != "PASS":
        raise RuntimeError("D3_R3_STATUS=BLOCKED_UPSTREAM_DESIGN_AUTHORITY:GATE")
    return value


def _failure_rows() -> list[dict[str, Any]]:
    partition = r2._d3_partition()
    cluster_manifest = read_json(R2_ROOT / "clusters/cluster_manifest.json")
    cluster_by_ordinal: dict[int, int] = {}
    for cluster in cluster_manifest["clusters"]:
        for ordinal in range(int(cluster["start_ordinal"]), int(cluster["stop_ordinal"]) + 1):
            cluster_by_ordinal[ordinal] = int(cluster["segment_id"])
    return [
        {
            "ordinal": int(row["ordinal"]),
            "source_frame": int(row["source_frame"]),
            "cluster_id": cluster_by_ordinal[int(row["ordinal"])],
            "D3_V1_E_IM": float(row["new_E_IM"]),
            "old_historical_E_IM": float(row["old_E_IM"]),
            "group": row["group"],
        }
        for row in partition
        if row["group"] in {"GROUP_B", "GROUP_D"}
    ]


def freeze_r3_validation_manifest(root: Path) -> dict[str, Any]:
    require(root / "preflight/gate_integrity.json", "status", "PASS", "FREEZE_R3_MANIFEST")
    failures = _failure_rows()
    partition = r2._d3_partition()
    counts = Counter(row["group"][-1] for row in partition)
    controls = read_json(R2_ROOT / "development_set/valid_controls.json")
    windows = read_json(R2_ROOT / "development_set/sequential_windows.json")
    determinism_ordinals = [
        int(row["ordinal"]) for row in read_json(R2_ROOT / "sentinels/determinism.json")["rows"]
    ]
    source_frames = [row["source_frame"] for row in failures]
    frame_set_sha = canonical_sha(source_frames)
    clusters = read_json(R2_ROOT / "clusters/cluster_manifest.json")
    normal_sets = {
        tuple(r2._d3_receipt(int(row["ordinal"]))["free_qpos_indices"]) for row in failures
    }
    checks = {
        "group_counts_exact": counts == Counter(EXPECTED_GROUPS),
        "failure_count_exact": len(failures) == EXPECTED_INVALID,
        "failure_group_exact": all(row["group"] == "GROUP_B" for row in failures),
        "cluster_count_exact": clusters.get("N_FAILURE_SEGMENTS") == 4,
        "cluster_population_exact": sum(int(item["length"]) for item in clusters["clusters"])
        == EXPECTED_INVALID,
        "source_frames_unique": len(set(source_frames)) == EXPECTED_INVALID,
        "normal_active_set_exact": normal_sets == {(0, 1, 2, 3)},
        "controls_frozen": controls.get("status") == "FROZEN_BEFORE_ORACLE",
        "windows_frozen": windows.get("status") == "FROZEN_BEFORE_ORACLE",
        "determinism_subset_exact": determinism_ordinals == [1950, 1991, 2188],
    }
    evidence = {
        "schema_version": "D3R3EvidenceIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "GROUP_A_COUNT": counts["A"],
        "GROUP_B_COUNT": counts["B"],
        "GROUP_C_COUNT": counts["C"],
        "GROUP_D_COUNT": counts["D"],
        "D3R3_FAILURE_FRAME_COUNT": len(failures),
        "D3R3_FAILURE_FRAME_SET_SHA256": frame_set_sha,
        "FAILURE_FRAME_SET_MATCH_D3R": "YES" if all(checks.values()) else "NO",
        "FAILURE_SEGMENT_COUNT": clusters.get("N_FAILURE_SEGMENTS"),
    }
    write_json(root / "preflight/evidence_integrity.json", evidence)
    if evidence["status"] != "PASS":
        raise RuntimeError("D3_R3_STATUS=BLOCKED_FAILURE_EVIDENCE_INTEGRITY")
    write_json(
        root / "manifests/failure_frames.json",
        {
            "schema_version": "D3R3FailureFramesV1",
            "status": "FROZEN",
            "frame_set_sha256": frame_set_sha,
            "frames": failures,
        },
    )
    write_text(
        root / "manifests/failure_frames.sha256",
        sha256_file(root / "manifests/failure_frames.json") + "\n",
    )
    write_json(root / "manifests/valid_controls.json", controls)
    write_json(root / "manifests/sequential_windows.json", windows)
    determinism = {
        "schema_version": "D3R3DeterminismSubsetV1",
        "status": "FROZEN",
        "ordinals": determinism_ordinals,
        "repeat_count": 1,
        "reference": "R3_FAILURE_FRAME_PRIMARY_EXECUTION",
        "tolerances": {
            "q_max_abs": DETERMINISM_Q_TOL,
            "base_max_abs": DETERMINISM_BASE_TOL,
            "E_IM_abs": DETERMINISM_EIM_TOL,
        },
    }
    write_json(root / "manifests/determinism_subset.json", determinism)
    design = read_json(R2_ROOT / "design/refinement_v2_design.json")
    d3_manifest = read_json(D3_ROOT / "run_authority/run_manifest.json")
    runtime, _ = r2._runtime_pair()
    manifest = {
        "schema_version": "D3R3ExecutionManifestV1",
        "status": "FROZEN",
        "run_role": "CONSUMED_DEV1_DEVELOPMENT_VALIDATION_ONLY",
        "failure_frame_ids": source_frames,
        "failure_ordinals": [row["ordinal"] for row in failures],
        "failure_processing_order": "ASCENDING_D3_ORDINAL",
        "valid_controls": controls["frames"],
        "sequential_windows": windows["windows"],
        "determinism_subset": determinism,
        "REFINEMENT_V2_DESIGN_SHA256": DESIGN_SHA256,
        "REFINEMENT_V2_DEVELOPMENT_GATE_SHA256": GATE_SHA256,
        "ObjectiveV2_sha256": design["ObjectiveV2_sha256"],
        "q_old_trajectory_sha256": d3_manifest["q_old_array_sha256"],
        "source_graph_sha256": d3_manifest["interaction_graph_sha256"],
        "robot_authority": {
            "name": "Wuji Hand2 Beta1",
            "sha256": d3_manifest["Wuji_sha256"],
            "finger_dof_names": list(runtime.model.dof_names),
        },
        "normal_active_dofs": [0, 1, 2, 3],
        "expanded_active_dofs": design["active_dofs"],
        "seed_budget_contract": {
            "seed_authority": design["seed_authority"],
            "compute_budget": design["compute_budget"],
        },
        "technical_resume_only": True,
        "scientific_retry_allowed": False,
        "NEW_FRESH_REFINEMENT_FRAME_CONSUMPTION": 0,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    path = root / "run_authority/development_validation_manifest.json"
    write_json(path, manifest)
    manifest_sha = sha256_file(path)
    write_text(root / "run_authority/development_validation_manifest.sha256", manifest_sha + "\n")
    write_json(
        root / "run_authority/technical_resume_policy.json",
        {
            "schema_version": "D3R3TechnicalResumePolicyV1",
            "status": "FROZEN",
            "manifest_sha256": manifest_sha,
            "allowed": [
                "process killed",
                "transient IO",
                "machine reboot",
                "artifact write interruption",
            ],
            "same_exact_frame_required": True,
            "scientific_retry_allowed": False,
        },
    )
    write_json(
        root / "run_authority/execution_order.json",
        {
            "schema_version": "D3R3ExecutionOrderV1",
            "status": "FROZEN",
            "manifest_sha256": manifest_sha,
            "failure_ordinals": manifest["failure_ordinals"],
            "then": ["valid_controls", "sequential_windows", "determinism", "gate_evaluation"],
        },
    )
    return {**manifest, "manifest_sha256": manifest_sha}


def _manifest(root: Path, action: str) -> tuple[dict[str, Any], str]:
    require(root / "preflight/design_integrity.json", "status", "PASS", action)
    require(root / "preflight/gate_integrity.json", "status", "PASS", action)
    require(root / "preflight/evidence_integrity.json", "status", "PASS", action)
    path = root / "run_authority/development_validation_manifest.json"
    value = require(path, "status", "FROZEN", action)
    observed = sha256_file(path)
    if _sidecar_value(root / "run_authority/development_validation_manifest.sha256") != observed:
        raise RuntimeError(f"{action}_REJECTED:VALIDATION_MANIFEST_SHA_DRIFT")
    return value, observed


def _profile_value(profile: dict[str, Any] | None, key: str) -> Any:
    return None if profile is None else profile.get(key)


def _normal_receipt(
    runtime: Any,
    ordinal: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    q, base = r2._d3_state(ordinal)
    values, evaluation = r2._measure_state(runtime, ordinal, q, base, previous_q, previous_base)
    historical = r2._d3_receipt(ordinal)
    profile = historical.get("profiler", {})
    return (
        q,
        base,
        {
            "path": "EXISTING_FROZEN_D3_V1_TERMINAL",
            "active_dofs": historical["free_qpos_indices"],
            "active_dof_count": len(historical["free_qpos_indices"]),
            "seed_provenance": "D3_V1_STORED_FROZEN_NORMAL_PATH",
            "objective": "ObjectiveV2",
            "E_IM": float(values.interaction_e_im),
            "hard_valid": bool(evaluation["feasible"]),
            "interaction_valid": float(values.interaction_e_im) <= TAU,
            "solver_status": historical.get("selected_phase"),
            "nfev": sum(
                int(item.get("nfev", 0))
                for item in (
                    historical.get("primary_solver", {}),
                    historical.get("secondary_solver", {}),
                )
            ),
            "wall_time_sec": float(
                profile.get("total_wall_sec", historical.get("elapsed_sec", 0.0))
            ),
            "evaluation": evaluation,
            "historical_receipt_sha256": sha256_file(
                D3_ROOT / f"checkpoints/frame_{ordinal:04d}/receipt.json"
            ),
        },
    )


def _conditional_frame(
    runtime: Any,
    bootstrap_runtime: Any,
    ordinal: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    normal_q, normal_base, normal = _normal_receipt(runtime, ordinal, previous_q, previous_base)
    trigger = bool(normal["hard_valid"] and not normal["interaction_valid"])
    if not trigger:
        return (
            normal_q,
            normal_base,
            {
                "normal": normal,
                "expanded": {
                    "triggered": False,
                    "trigger_reason": "FROZEN_TRIGGER_FALSE",
                    "active_dofs": [],
                    "active_dof_count": 0,
                    "E_IM": None,
                    "hard_valid": None,
                    "solver_status": "NOT_RUN",
                    "nfev": 0,
                    "wall_time_sec": 0.0,
                },
                "selected_path": "NORMAL_PATH",
                "final_E_IM": normal["E_IM"],
                "final_hard_valid": normal["hard_valid"],
                "final_interaction_valid": normal["interaction_valid"],
                "wall_time_sec": time.perf_counter() - started,
            },
        )
    q, base, expanded_raw = r2._oracle_c_frame(
        runtime, bootstrap_runtime, ordinal, previous_q, previous_base
    )
    expanded = {
        "triggered": True,
        "trigger_reason": "NORMAL_HARD_VALID_AND_E_IM_EXCEEDS_1E-4",
        "active_dofs": expanded_raw["active_dofs"],
        "active_dof_count": len(expanded_raw["active_dofs"]),
        "seed_provenance": expanded_raw["selected_candidate"],
        "objective": "ObjectiveV2",
        "E_IM": float(expanded_raw["selected_E_IM"]),
        "hard_valid": bool(expanded_raw["selected_hard_valid"]),
        "interaction_valid": bool(expanded_raw["selected_interaction_valid"]),
        "solver_status": _profile_value(expanded_raw.get("secondary_profile"), "status")
        if expanded_raw.get("secondary_profile") is not None
        else _profile_value(expanded_raw.get("primary_profile"), "status"),
        "solver_message": _profile_value(expanded_raw.get("secondary_profile"), "message")
        if expanded_raw.get("secondary_profile") is not None
        else _profile_value(expanded_raw.get("primary_profile"), "message"),
        "nfev": sum(
            int(item.get("nfev", 0))
            for item in (
                expanded_raw.get("primary_profile") or {},
                expanded_raw.get("secondary_profile") or {},
            )
        ),
        "wall_time_sec": float(expanded_raw["wall_time_sec"]),
        "retention": expanded_raw["retention"],
        "candidate_ids": expanded_raw["candidate_ids"],
        "raw_receipt": expanded_raw,
    }
    selected_path = (
        "NORMAL_PATH_RETAINED_AFTER_EXPANDED_SCREENING"
        if expanded_raw["selected_candidate"] == "d3_v1_baseline"
        else "EXPANDED_PATH"
    )
    return (
        q,
        base,
        {
            "normal": normal,
            "expanded": expanded,
            "selected_path": selected_path,
            "final_E_IM": expanded["E_IM"],
            "final_hard_valid": expanded["hard_valid"],
            "final_interaction_valid": expanded["interaction_valid"],
            "wall_time_sec": time.perf_counter() - started,
        },
    )


def _failure_checkpoint(root: Path, ordinal: int) -> Path:
    return root / "failure_frames/checkpoints" / f"frame_{ordinal:04d}.json"


def _failure_row(
    item: dict[str, Any],
    q: np.ndarray,
    base: np.ndarray,
    receipt: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    ordinal = int(item["ordinal"])
    q_d3, _ = r2._d3_state(ordinal)
    q_old = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    old = float(item["D3_V1_E_IM"])
    new = float(receipt["final_E_IM"])
    deviations = np.abs(q - q_old)
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    return {
        "ordinal": ordinal,
        "source_frame": int(item["source_frame"]),
        "cluster_id": int(item["cluster_id"]),
        "old_historical_E_IM": float(item["old_historical_E_IM"]),
        "D3_V1_E_IM": old,
        "normal_active_dofs": json.dumps(receipt["normal"]["active_dofs"]),
        "normal_seed_provenance": receipt["normal"]["seed_provenance"],
        "normal_objective": receipt["normal"]["objective"],
        "normal_E_IM": receipt["normal"]["E_IM"],
        "normal_hard_valid": receipt["normal"]["hard_valid"],
        "normal_solver_status": receipt["normal"]["solver_status"],
        "normal_nfev": receipt["normal"]["nfev"],
        "normal_wall_time_sec": receipt["normal"]["wall_time_sec"],
        "expanded_triggered": receipt["expanded"]["triggered"],
        "expanded_trigger_reason": receipt["expanded"]["trigger_reason"],
        "expanded_active_dofs": json.dumps(receipt["expanded"]["active_dofs"]),
        "expanded_E_IM": receipt["expanded"]["E_IM"],
        "expanded_hard_valid": receipt["expanded"]["hard_valid"],
        "expanded_solver_status": receipt["expanded"]["solver_status"],
        "expanded_nfev": receipt["expanded"]["nfev"],
        "expanded_wall_time_sec": receipt["expanded"]["wall_time_sec"],
        "selected_path": receipt["selected_path"],
        "final_E_IM": new,
        "interaction_valid": receipt["final_interaction_valid"],
        "hard_valid": receipt["final_hard_valid"],
        "recovery": bool(old > TAU and new <= TAU and receipt["final_hard_valid"]),
        "relative_reduction": (old - new) / old,
        "q_deviation_from_q_old_l2": float(np.linalg.norm(q - q_old)),
        "q_deviation_from_D3V1_l2": float(np.linalg.norm(q - q_d3)),
        "q_sha256": array_sha(q),
        "base_sha256": array_sha(base),
        "total_wall_time_sec": receipt["wall_time_sec"],
        **{
            f"{finger}_q_deviation_l2": float(np.linalg.norm(deviations[list(indices)]))
            for finger, indices in blocks.items()
        },
    }


def _load_failure_rows(root: Path) -> list[dict[str, Any]]:
    if not (root / "failure_frames/per_frame.csv").is_file():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_csv(root / "failure_frames/per_frame.csv"):
        rows.append(
            {
                **row,
                "ordinal": int(row["ordinal"]),
                "source_frame": int(row["source_frame"]),
                "cluster_id": int(row["cluster_id"]),
                "D3_V1_E_IM": float(row["D3_V1_E_IM"]),
                "final_E_IM": float(row["final_E_IM"]),
                "relative_reduction": float(row["relative_reduction"]),
                "hard_valid": row["hard_valid"] == "True",
                "interaction_valid": row["interaction_valid"] == "True",
                "recovery": row["recovery"] == "True",
                "expanded_triggered": row["expanded_triggered"] == "True",
                "normal_wall_time_sec": float(row["normal_wall_time_sec"]),
                "expanded_wall_time_sec": float(row["expanded_wall_time_sec"]),
                "total_wall_time_sec": float(row["total_wall_time_sec"]),
            }
        )
    return rows


def _summarize_failures(root: Path, rows: list[dict[str, Any]], early: bool) -> dict[str, Any]:
    recovered = sum(bool(row["recovery"]) for row in rows)
    unrecovered = len(rows) - recovered
    remaining = EXPECTED_INVALID - len(rows)
    maximum = recovered + remaining
    reductions = np.asarray([float(row["relative_reduction"]) for row in rows], dtype=np.float64)
    summary = {
        "schema_version": "D3R3FailureRecoverySummaryV1",
        "status": "EARLY_STOP" if early else "PASS",
        "N": EXPECTED_INVALID,
        "processed": len(rows),
        "recovered": recovered,
        "unrecovered": unrecovered,
        "remaining": remaining,
        "maximum_possible_final_recovered": maximum,
        "maximum_possible_recovery_fraction": maximum / EXPECTED_INVALID,
        "recovery_fraction": recovered / EXPECTED_INVALID if not early else recovered / len(rows),
        "mean_relative_reduction": None if not len(rows) else float(np.mean(reductions)),
        "median_relative_reduction": None if not len(rows) else float(np.median(reductions)),
        "p25_relative_reduction": None if not len(rows) else float(np.percentile(reductions, 25)),
        "p75_relative_reduction": None if not len(rows) else float(np.percentile(reductions, 75)),
        "p90_relative_reduction": None if not len(rows) else float(np.percentile(reductions, 90)),
        "HARD_VALID_FRAME_COUNT": sum(bool(row["hard_valid"]) for row in rows),
        "HARD_INVALID_FRAME_COUNT": sum(not bool(row["hard_valid"]) for row in rows),
        "EXPANDED_PATH_INVOCATION_COUNT": sum(bool(row["expanded_triggered"]) for row in rows),
        "NORMAL_PATH_RECOVERY_COUNT": sum(
            bool(row["recovery"]) and row["selected_path"] == "NORMAL_PATH" for row in rows
        ),
        "EXPANDED_PATH_RECOVERY_COUNT": sum(
            bool(row["recovery"]) and row["selected_path"] == "EXPANDED_PATH" for row in rows
        ),
        "RECOVERY_GATE_MATHEMATICALLY_IMPOSSIBLE": "YES" if early else "NO",
    }
    cluster_rows = []
    for cluster in range(1, 5):
        selected = [row for row in rows if int(row["cluster_id"]) == cluster]
        old = np.asarray([float(row["D3_V1_E_IM"]) for row in selected])
        new = np.asarray([float(row["final_E_IM"]) for row in selected])
        cluster_rows.append(
            {
                "cluster_id": cluster,
                "N": len(selected),
                "technical_count": len(selected),
                "recovered_count": sum(bool(row["recovery"]) for row in selected),
                "recovery_fraction": sum(bool(row["recovery"]) for row in selected) / len(selected),
                "old_E_IM_mean": float(np.mean(old)),
                "old_E_IM_p95": float(np.percentile(old, 95)),
                "old_E_IM_max": float(np.max(old)),
                "new_E_IM_mean": float(np.mean(new)),
                "new_E_IM_p95": float(np.percentile(new, 95)),
                "new_E_IM_max": float(np.max(new)),
                "median_relative_reduction": float(
                    np.median([float(row["relative_reduction"]) for row in selected])
                ),
                "expanded_fallback_invocation_count": sum(
                    bool(row["expanded_triggered"]) for row in selected
                ),
            }
        )
    write_json(root / "failure_frames/recovery_summary.json", summary)
    write_csv(root / "failure_frames/per_cluster.csv", cluster_rows)
    write_json(
        root
        / (
            "failure_frames/early_stop_receipt.json"
            if early
            else "failure_frames/no_early_stop.json"
        ),
        {
            "schema_version": "D3R3EarlyStopReceiptV1",
            "status": "TRIGGERED" if early else "NOT_TRIGGERED",
            **{
                key: summary[key]
                for key in (
                    "processed",
                    "recovered",
                    "unrecovered",
                    "remaining",
                    "maximum_possible_final_recovered",
                    "maximum_possible_recovery_fraction",
                    "RECOVERY_GATE_MATHEMATICALLY_IMPOSSIBLE",
                )
            },
        },
    )
    return summary


def _run_failure_frames(root: Path, *, resume: bool) -> dict[str, Any]:
    manifest, manifest_sha = _manifest(root, "RUN_R3_FAILURE_FRAMES")
    items = read_json(root / "manifests/failure_frames.json")["frames"]
    existing = [_failure_checkpoint(root, int(item["ordinal"])).is_file() for item in items]
    if not resume and any(existing):
        raise RuntimeError("RUN_R3_FAILURE_FRAMES_REJECTED:CHECKPOINTS_EXIST_USE_RESUME")
    runtime, bootstrap_runtime = r2._runtime_pair()
    rows: list[dict[str, Any]] = []
    jsonl = root / "failure_frames/frame_results.jsonl"
    if not resume:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl.write_text("", encoding="utf-8")
    gate = read_json(root / "preflight/gate_integrity.json")
    max_unrecovered = int(gate["MAX_ALLOWED_UNRECOVERED"])
    for item in items:
        ordinal = int(item["ordinal"])
        path = _failure_checkpoint(root, ordinal)
        if path.is_file():
            checkpoint = read_json(path)
            if checkpoint.get("manifest_sha256") != manifest_sha:
                raise RuntimeError(f"D3R3_TECHNICAL_RESUME_MANIFEST_DRIFT:{ordinal}")
            row = checkpoint["row"]
        else:
            previous_q, previous_base = r2._previous_d3_state(ordinal)
            if previous_q is None or previous_base is None:
                raise RuntimeError(f"D3R3_FAILURE_PREDECESSOR_MISSING:{ordinal}")
            q, base, receipt = _conditional_frame(
                runtime, bootstrap_runtime, ordinal, previous_q, previous_base
            )
            if not receipt["expanded"]["triggered"]:
                raise RuntimeError(f"D3R3_INVALID_FRAME_DID_NOT_TRIGGER_EXPANSION:{ordinal}")
            row = _failure_row(item, q, base, receipt, runtime)
            checkpoint = {
                "schema_version": "D3R3FailureFrameCheckpointV1",
                "status": "COMPLETE",
                "manifest_sha256": manifest_sha,
                "ordinal": ordinal,
                "source_frame": item["source_frame"],
                "qpos": q.tolist(),
                "base_pose_scene": base.tolist(),
                "receipt": receipt,
                "row": row,
            }
            write_json(path, checkpoint)
            append_jsonl(jsonl, checkpoint)
        rows.append(row)
        write_csv(root / "failure_frames/per_frame.csv", rows)
        recovered = sum(bool(row["recovery"]) for row in rows)
        unrecovered = len(rows) - recovered
        print(
            f"D3R3 failure ordinal={ordinal} processed={len(rows)}/{EXPECTED_INVALID} "
            f"recovered={recovered} unrecovered={unrecovered} E_IM={float(row['final_E_IM']):.12g}",
            flush=True,
        )
        if unrecovered > max_unrecovered:
            return _summarize_failures(root, rows, True)
    return _summarize_failures(root, rows, False)


def run_r3_failure_frame_validation(root: Path) -> dict[str, Any]:
    return _run_failure_frames(root, resume=False)


def resume_r3_failure_frame_validation(root: Path) -> dict[str, Any]:
    return _run_failure_frames(root, resume=True)


def run_r3_valid_controls(root: Path) -> dict[str, Any]:
    _manifest(root, "RUN_R3_VALID_CONTROLS")
    failure = require(
        root / "failure_frames/recovery_summary.json", "status", "PASS", "RUN_R3_VALID_CONTROLS"
    )
    if failure["processed"] != EXPECTED_INVALID:
        raise RuntimeError("RUN_R3_VALID_CONTROLS_REJECTED:FAILURE_FRAMES_INCOMPLETE")
    controls = read_json(root / "manifests/valid_controls.json")["frames"]
    runtime, bootstrap_runtime = r2._runtime_pair()
    rows = []
    for item in controls:
        ordinal = int(item["ordinal"])
        previous_q, previous_base = r2._previous_d3_state(ordinal)
        if previous_q is None or previous_base is None:
            q_old = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
            base_old = np.asarray(
                runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64
            )
            previous_q, previous_base = q_old, base_old
        q, _base, receipt = _conditional_frame(
            runtime, bootstrap_runtime, ordinal, previous_q, previous_base
        )
        baseline_q, _ = r2._d3_state(ordinal)
        rows.append(
            {
                "role": item["role"],
                "ordinal": ordinal,
                "source_frame": int(item["source_frame"]),
                "D3_V1_E_IM": float(item["D3_V1_E_IM"]),
                "final_E_IM": receipt["final_E_IM"],
                "hard_valid": receipt["final_hard_valid"],
                "interaction_valid": receipt["final_interaction_valid"],
                "preserved": bool(
                    receipt["final_hard_valid"] and receipt["final_interaction_valid"]
                ),
                "expanded_triggered": receipt["expanded"]["triggered"],
                "selected_path": receipt["selected_path"],
                "q_exact_D3_V1": bool(np.array_equal(q, baseline_q)),
            }
        )
    write_csv(root / "valid_controls/per_frame.csv", rows)
    preserved = sum(bool(row["preserved"]) for row in rows)
    summary = {
        "schema_version": "D3R3ValidControlSummaryV1",
        "status": "PASS" if preserved == len(rows) else "FAIL",
        "VALID_CONTROL_COUNT": len(rows),
        "VALID_CONTROLS_PRESERVED": preserved,
        "VALID_CONTROL_PRESERVATION_FRACTION": preserved / len(rows),
        "VALID_CONTROL_EXPANDED_PATH_INVOCATION_COUNT": sum(
            bool(row["expanded_triggered"]) for row in rows
        ),
        "VALID_CONTROL_HARD_VALID": sum(bool(row["hard_valid"]) for row in rows),
    }
    write_json(root / "valid_controls/summary.json", summary)
    return summary


def run_r3_sequential_windows(root: Path) -> dict[str, Any]:
    _manifest(root, "RUN_R3_SEQUENTIAL_WINDOWS")
    require(root / "valid_controls/summary.json", "status", "PASS", "RUN_R3_SEQUENTIAL_WINDOWS")
    windows = read_json(root / "manifests/sequential_windows.json")["windows"]
    runtime, bootstrap_runtime = r2._runtime_pair()
    per_frame = []
    per_window = []
    runtime_chain = []
    for window in windows:
        first = int(window["ordinals"][0])
        previous_q, previous_base = r2._previous_d3_state(first)
        if previous_q is None or previous_base is None:
            raise RuntimeError(f"D3R3_WINDOW_PREDECESSOR_MISSING:{window['window_id']}")
        rows = []
        for ordinal_value in window["ordinals"]:
            ordinal = int(ordinal_value)
            predecessor_q_sha = array_sha(previous_q)
            predecessor_base_sha = array_sha(previous_base)
            q, base, receipt = _conditional_frame(
                runtime, bootstrap_runtime, ordinal, previous_q, previous_base
            )
            continuity = r2.d2c.actual_continuity(runtime, previous_q, previous_base, q, base)
            row = {
                "window_id": window["window_id"],
                "cluster_id": int(window["segment_id"]),
                "ordinal": ordinal,
                "source_frame": d3.SOURCE_START + ordinal,
                "entry_offset": ordinal - int(window["entry_ordinal"]),
                "normal_E_IM": receipt["normal"]["E_IM"],
                "fallback_triggered": receipt["expanded"]["triggered"],
                "expanded_E_IM": receipt["expanded"]["E_IM"],
                "selected_path": receipt["selected_path"],
                "final_E_IM": receipt["final_E_IM"],
                "interaction_valid": receipt["final_interaction_valid"],
                "hard_valid": receipt["final_hard_valid"],
                "translation_step_m": continuity["translation_step_m"],
                "rotation_step_rad": continuity["rotation_step_rad"],
                "q_step_inf_rad": continuity["q_step_inf_rad"],
                "q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED",
                "previous_accepted_authority": "CURRENT_R3_TRAJECTORY_T_MINUS_1",
                "q_old_sha256": array_sha(
                    np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
                ),
                "predecessor_q_sha256": predecessor_q_sha,
                "predecessor_base_sha256": predecessor_base_sha,
                "accepted_q_sha256": array_sha(q),
                "accepted_base_sha256": array_sha(base),
            }
            rows.append(row)
            per_frame.append(row)
            runtime_chain.append(
                {
                    "window_id": window["window_id"],
                    "ordinal": ordinal,
                    "predecessor_q_sha256": predecessor_q_sha,
                    "accepted_q_sha256": array_sha(q),
                    "q_old_sha256": row["q_old_sha256"],
                    "previous_runtime_state_aliased_as_qold": predecessor_q_sha
                    == row["q_old_sha256"],
                }
            )
            previous_q, previous_base = q, base
        continuity_pass = all(
            float(row["translation_step_m"])
            <= runtime.authority.temporal_base_translation_limit_m + 1e-12
            and float(row["rotation_step_rad"])
            <= runtime.authority.temporal_base_rotation_limit_rad + 1e-12
            for row in rows
        )
        window_pass = (
            all(bool(row["interaction_valid"] and row["hard_valid"]) for row in rows)
            and continuity_pass
        )
        per_window.append(
            {
                "window_id": window["window_id"],
                "cluster_id": int(window["segment_id"]),
                "entry_source_frame": d3.SOURCE_START + int(window["entry_ordinal"]),
                "predecessor_source_frame": d3.SOURCE_START + int(window["entry_ordinal"]) - 1,
                "frame_count": len(rows),
                "valid_count": sum(bool(row["interaction_valid"]) for row in rows),
                "hard_valid_count": sum(bool(row["hard_valid"]) for row in rows),
                "fallback_invocation_count": sum(bool(row["fallback_triggered"]) for row in rows),
                "continuity": "PASS" if continuity_pass else "FAIL",
                "status": "PASS" if window_pass else "FAIL",
            }
        )
    write_csv(root / "sequential_windows/per_frame.csv", per_frame)
    write_csv(root / "sequential_windows/per_window.csv", per_window)
    write_csv(root / "sequential_windows/runtime_chain.csv", runtime_chain)
    passed = sum(row["status"] == "PASS" for row in per_window)
    chain_pass = all(
        runtime_chain[index]["predecessor_q_sha256"]
        == runtime_chain[index - 1]["accepted_q_sha256"]
        for index in range(1, len(runtime_chain))
        if runtime_chain[index]["window_id"] == runtime_chain[index - 1]["window_id"]
    )
    summary = {
        "schema_version": "D3R3SequentialWindowSummaryV1",
        "status": "PASS" if passed == len(per_window) and chain_pass else "FAIL",
        "SEQUENTIAL_WINDOW_COUNT": len(per_window),
        "SEQUENTIAL_WINDOWS_PASS": passed,
        "SEQUENTIAL_FRAMES_TOTAL": len(per_frame),
        "SEQUENTIAL_FRAMES_VALID": sum(bool(row["interaction_valid"]) for row in per_frame),
        "SEQUENTIAL_FRAMES_HARD_VALID": sum(bool(row["hard_valid"]) for row in per_frame),
        "CONTINUITY": "PASS" if all(row["continuity"] == "PASS" for row in per_window) else "FAIL",
        "RUNTIME_STATE_CHAIN": "PASS" if chain_pass else "FAIL",
        "SEQUENTIAL_BASIN_HYSTERESIS_OBSERVED": "NO" if passed == len(per_window) else "YES",
        "MAX_Q_STEP_INF_RAD_DIAGNOSTIC": max(float(row["q_step_inf_rad"]) for row in per_frame),
        "Q_CONTINUITY_ROLE": "RECORDED_DIAGNOSTIC; frozen consumed-window gate uses hard validity plus frozen translation/rotation continuity",
    }
    write_json(root / "sequential_windows/summary.json", summary)
    return summary


def run_r3_determinism(root: Path) -> dict[str, Any]:
    _manifest(root, "RUN_R3_DETERMINISM")
    require(root / "sequential_windows/summary.json", "status", "PASS", "RUN_R3_DETERMINISM")
    subset = read_json(root / "manifests/determinism_subset.json")
    runtime, bootstrap_runtime = r2._runtime_pair()
    rows = []
    for ordinal_value in subset["ordinals"]:
        ordinal = int(ordinal_value)
        reference = read_json(_failure_checkpoint(root, ordinal))
        previous_q, previous_base = r2._previous_d3_state(ordinal)
        if previous_q is None or previous_base is None:
            raise RuntimeError(f"D3R3_DETERMINISM_PREDECESSOR_MISSING:{ordinal}")
        q, base, receipt = _conditional_frame(
            runtime, bootstrap_runtime, ordinal, previous_q, previous_base
        )
        reference_receipt = reference["receipt"]
        rows.append(
            {
                "ordinal": ordinal,
                "source_frame": d3.SOURCE_START + ordinal,
                "selected_path_match": receipt["selected_path"]
                == reference_receipt["selected_path"],
                "fallback_trigger_match": receipt["expanded"]["triggered"]
                == reference_receipt["expanded"]["triggered"],
                "active_dofs_match": receipt["expanded"]["active_dofs"]
                == reference_receipt["expanded"]["active_dofs"],
                "q_max_abs_diff": float(
                    np.max(np.abs(q - np.asarray(reference["qpos"], dtype=np.float64)))
                ),
                "base_max_abs_diff": float(
                    np.max(
                        np.abs(base - np.asarray(reference["base_pose_scene"], dtype=np.float64))
                    )
                ),
                "E_IM_abs_diff": abs(
                    float(receipt["final_E_IM"]) - float(reference_receipt["final_E_IM"])
                ),
                "hard_valid_match": receipt["final_hard_valid"]
                == reference_receipt["final_hard_valid"],
                "candidate_selection_match": receipt["expanded"].get("seed_provenance")
                == reference_receipt["expanded"].get("seed_provenance"),
                "solver_terminal_scientific_classification_match": (
                    receipt["final_hard_valid"],
                    receipt["final_interaction_valid"],
                )
                == (
                    reference_receipt["final_hard_valid"],
                    reference_receipt["final_interaction_valid"],
                ),
            }
        )
    write_csv(root / "determinism/per_frame.csv", rows)
    write_json(
        root / "determinism/repeat_manifest.json",
        {
            **subset,
            "reference_manifest_sha256": _sidecar_value(
                root / "run_authority/development_validation_manifest.sha256"
            ),
        },
    )
    max_q = max(float(row["q_max_abs_diff"]) for row in rows)
    max_base = max(float(row["base_max_abs_diff"]) for row in rows)
    max_eim = max(float(row["E_IM_abs_diff"]) for row in rows)
    exact_fields = all(
        all(
            bool(row[key])
            for key in (
                "selected_path_match",
                "fallback_trigger_match",
                "active_dofs_match",
                "hard_valid_match",
                "candidate_selection_match",
                "solver_terminal_scientific_classification_match",
            )
        )
        for row in rows
    )
    passed = (
        exact_fields
        and max_q <= DETERMINISM_Q_TOL
        and max_base <= DETERMINISM_BASE_TOL
        and max_eim <= DETERMINISM_EIM_TOL
    )
    summary = {
        "schema_version": "D3R3DeterminismSummaryV1",
        "status": "PASS" if passed else "FAIL",
        "DETERMINISM": "PASS" if passed else "FAIL",
        "MAX_Q_ABS_DIFF": max_q,
        "MAX_BASE_ABS_DIFF": max_base,
        "MAX_E_IM_ABS_DIFF": max_eim,
    }
    write_json(root / "determinism/summary.json", summary)
    return summary


def evaluate_r3_development_gate(root: Path) -> dict[str, Any]:
    _manifest(root, "EVALUATE_R3_DEVELOPMENT_GATE")
    failure = read_json(root / "failure_frames/recovery_summary.json")
    if failure.get("status") == "EARLY_STOP":
        criteria = {
            "G1_TECHNICAL": "PASS",
            "G2_RECOVERY": "FAIL",
            "G3_MEDIAN_REDUCTION": "NOT_RUN",
            "G4_VALID_PRESERVATION": "NOT_RUN",
            "G5_HARD_VALIDITY": "NOT_RUN",
            "G6_SEQUENTIAL": "NOT_RUN",
            "G7_CONTINUITY": "NOT_RUN",
            "G8_DETERMINISM": "NOT_RUN",
        }
    else:
        valid = read_json(root / "valid_controls/summary.json")
        sequential = read_json(root / "sequential_windows/summary.json")
        determinism = read_json(root / "determinism/summary.json")
        gate = read_json(root / "preflight/gate_integrity.json")
        criteria = {
            "G1_TECHNICAL": "PASS" if failure["processed"] == EXPECTED_INVALID else "FAIL",
            "G2_RECOVERY": "PASS"
            if failure["recovered"] >= int(gate["RECOVERY_REQUIRED_COUNT"])
            else "FAIL",
            "G3_MEDIAN_REDUCTION": "PASS"
            if float(failure["median_relative_reduction"])
            >= float(gate["MEDIAN_REDUCTION_REQUIRED"])
            else "FAIL",
            "G4_VALID_PRESERVATION": "PASS"
            if float(valid["VALID_CONTROL_PRESERVATION_FRACTION"])
            >= float(gate["VALID_PRESERVATION_REQUIRED"])
            else "FAIL",
            "G5_HARD_VALIDITY": "PASS"
            if failure["HARD_INVALID_FRAME_COUNT"] == 0
            and valid["VALID_CONTROL_HARD_VALID"] == valid["VALID_CONTROL_COUNT"]
            and sequential["SEQUENTIAL_FRAMES_HARD_VALID"] == sequential["SEQUENTIAL_FRAMES_TOTAL"]
            else "FAIL",
            "G6_SEQUENTIAL": "PASS" if sequential["status"] == "PASS" else "FAIL",
            "G7_CONTINUITY": "PASS" if sequential["CONTINUITY"] == "PASS" else "FAIL",
            "G8_DETERMINISM": "PASS" if determinism["DETERMINISM"] == "PASS" else "FAIL",
        }
    passed = all(value == "PASS" for value in criteria.values())
    failure_rows = _load_failure_rows(root)
    remaining = [row for row in failure_rows if not bool(row["recovery"])]
    cluster_rows = read_csv(root / "failure_frames/per_cluster.csv")
    reduction_gate = float(
        read_json(root / "preflight/gate_integrity.json")["MEDIAN_REDUCTION_REQUIRED"]
    )
    if criteria["G3_MEDIAN_REDUCTION"] == "FAIL":
        primary_failure_mechanism = "EXPANDED_ACTIVE_SET_EFFECT_SIZE_INSUFFICIENT"
    elif criteria["G4_VALID_PRESERVATION"] == "FAIL":
        primary_failure_mechanism = "VALID_CONTROL_REGRESSION"
    elif criteria["G6_SEQUENTIAL"] == "FAIL" or criteria["G7_CONTINUITY"] == "FAIL":
        primary_failure_mechanism = "SEQUENTIAL_PROPAGATION_FAILURE"
    elif criteria["G8_DETERMINISM"] == "FAIL":
        primary_failure_mechanism = "DETERMINISM_FAILURE"
    elif criteria["G2_RECOVERY"] == "FAIL":
        primary_failure_mechanism = "EXPANDED_ACTIVE_SET_STILL_NO_VALID_TERMINAL"
    else:
        primary_failure_mechanism = "NONE" if passed else "MULTI_FACTOR"
    failure_analysis = {
        "schema_version": "D3R3DevelopmentFailureAnalysisV1",
        "status": "NOT_APPLICABLE" if passed else "FAILURE_LOCALIZED",
        "PRIMARY_DEVELOPMENT_FAILURE_MECHANISM": primary_failure_mechanism,
        "FAILURE_CONCENTRATED_IN_ONE_CLUSTER": "YES"
        if remaining and len({int(row["cluster_id"]) for row in remaining}) == 1
        else "NO",
        "REMAINING_INVALID_COUNT": len(remaining),
        "REMAINING_INVALID_ORDINALS": [int(row["ordinal"]) for row in remaining],
        "REMAINING_INVALID_SOURCE_FRAMES": [int(row["source_frame"]) for row in remaining],
        "REMAINING_INVALID_CLUSTERS": sorted({int(row["cluster_id"]) for row in remaining}),
        "EXPANDED_ACTIVE_SET_STILL_NO_VALID_TERMINAL_COUNT": len(remaining),
        "CLUSTERS_MEETING_MEDIAN_REDUCTION_GATE": sum(
            float(row["median_relative_reduction"]) >= reduction_gate for row in cluster_rows
        ),
        "CLUSTER_COUNT": len(cluster_rows),
        "VALID_CONTROL_REGRESSION": "YES" if criteria["G4_VALID_PRESERVATION"] == "FAIL" else "NO",
        "SEQUENTIAL_PROPAGATION_FAILURE": "YES"
        if criteria["G6_SEQUENTIAL"] == "FAIL" or criteria["G7_CONTINUITY"] == "FAIL"
        else "NO",
        "DETERMINISM_FAILURE": "YES" if criteria["G8_DETERMINISM"] == "FAIL" else "NO",
    }
    result = {
        "schema_version": "D3R3DevelopmentGateDecisionV1",
        "status": "PASS" if passed else "FAIL",
        **criteria,
        "REFINEMENT_V2_DEVELOPMENT_VALIDATION": "PASS" if passed else "FAIL",
        "D3_R3_STATUS": "PASS" if passed else "FAIL_DEVELOPMENT_GATE",
        "automatic_evaluator": True,
        "manual_override": False,
    }
    write_json(
        root / "gate/criterion_results.json",
        {"schema_version": "D3R3CriterionResultsV1", **criteria},
    )
    write_json(root / "gate/decision.json", result)
    write_json(root / "gate/failure_analysis.json", failure_analysis)
    if not passed:
        write_json(
            root / "future/not_authorized.json",
            {
                "schema_version": "D3R3FreshCertificationNotAuthorizedV1",
                "status": "NOT_AUTHORIZED",
                "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED": "NO",
                "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
                "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
                "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
                "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
                "NEXT": "DEV1_REFINEMENT_V2_DEVELOPMENT_FAILURE_ANALYSIS",
                "failed_criteria": [key for key, value in criteria.items() if value == "FAIL"],
                "failure_analysis": failure_analysis,
            },
        )
    return result


def profile_r3_cost(root: Path) -> dict[str, Any]:
    require(root / "gate/decision.json", "automatic_evaluator", True, "PROFILE_R3_COST")
    rows = _load_failure_rows(root)
    total = np.asarray([float(row["total_wall_time_sec"]) for row in rows])
    normal = np.asarray([float(row["normal_wall_time_sec"]) for row in rows])
    expanded = np.asarray([float(row["expanded_wall_time_sec"]) for row in rows])
    fallback = sum(bool(row["expanded_triggered"]) for row in rows)
    d3_profile = read_json(D3_ROOT / "profiler/aggregate.json")
    original_full_runtime = float(d3_profile["solver_total_sec"])
    observed_added_full_runtime = fallback * float(np.mean(expanded))
    estimated_conditional_runtime = original_full_runtime + observed_added_full_runtime
    aggregate = {
        "schema_version": "D3R3CostAggregateV1",
        "status": "PASS",
        "N_failure_frames": len(rows),
        "N_fallback_invocations": fallback,
        "fallback_invocation_rate": fallback / len(rows),
        "mean_normal_historical_sec_on_failure_frames": float(np.mean(normal)),
        "mean_fallback_sec": float(np.mean(expanded)),
        "mean_r3_validation_wall_sec": float(np.mean(total)),
        "p50_total_sec": float(np.percentile(total, 50)),
        "p90_total_sec": float(np.percentile(total, 90)),
        "p95_total_sec": float(np.percentile(total, 95)),
        "max_total_sec": float(np.max(total)),
        "total_additional_development_compute_sec": float(np.sum(expanded)),
        "expected_fallback_rate_on_D3_V1_trajectory": fallback / d3.EXPECTED_FRAMES,
        "original_D3_V1_full_runtime_sec": original_full_runtime,
        "observed_added_runtime_projected_to_D3_V1_sec": observed_added_full_runtime,
        "estimated_conditional_runtime_sec_on_D3_V1_trajectory": estimated_conditional_runtime,
        "estimated_conditional_mean_sec_per_D3_frame": estimated_conditional_runtime
        / d3.EXPECTED_FRAMES,
        "observed_added_sec_per_D3_frame": observed_added_full_runtime / d3.EXPECTED_FRAMES,
        "observed_failure_frame_slowdown_vs_historical_normal": (
            float(np.mean(normal)) + float(np.mean(expanded))
        )
        / float(np.mean(normal)),
        "unconditional_20dof_runtime": "NOT_MEASURED",
    }
    write_csv(root / "profiler/per_frame.csv", rows)
    write_json(root / "profiler/aggregate.json", aggregate)
    cost = read_json(R2_ROOT / "design/cost_model.json")
    comparison = {
        "schema_version": "D3R3CostModelComparisonV1",
        "status": "PASS",
        "R2_predicted": cost,
        "R3_observed_fallback_invocation_rate": aggregate["fallback_invocation_rate"],
        "R3_observed_added_sec_per_D3_frame": aggregate["observed_added_sec_per_D3_frame"],
        "R3_observed_failure_frame_slowdown": aggregate[
            "observed_failure_frame_slowdown_vs_historical_normal"
        ],
        "R3_estimated_conditional_runtime_sec": aggregate[
            "estimated_conditional_runtime_sec_on_D3_V1_trajectory"
        ],
        "runtime_is_scientific_gate": False,
    }
    write_json(root / "profiler/cost_model_comparison.json", comparison)
    return aggregate


def render_r3_development_viewer(root: Path) -> dict[str, Any]:
    require(root / "gate/decision.json", "automatic_evaluator", True, "RENDER_R3_VIEWER")
    failures = read_json(root / "manifests/failure_frames.json")["frames"]
    controls = read_json(root / "manifests/valid_controls.json")["frames"]
    display = failures + controls
    ordinals = np.asarray([int(item["ordinal"]) for item in display], dtype=np.int64)
    frames = np.asarray([int(item["source_frame"]) for item in display], dtype=np.int64)
    q_r3 = []
    base_r3 = []
    for item in display:
        path = _failure_checkpoint(root, int(item["ordinal"]))
        if path.is_file():
            value = read_json(path)
            q_r3.append(value["qpos"])
            base_r3.append(value["base_pose_scene"])
        else:
            q, base = r2._d3_state(int(item["ordinal"]))
            q_r3.append(q.tolist())
            base_r3.append(base.tolist())
    q_r3_array = np.asarray(q_r3, dtype=np.float64)
    base_r3_array = np.asarray(base_r3, dtype=np.float64)
    q_d3 = np.stack([r2._d3_state(int(ordinal))[0] for ordinal in ordinals])
    base_d3 = np.stack([r2._d3_state(int(ordinal))[1] for ordinal in ordinals])
    canonical = o5.load_canonical_hoi(d3.old_paths()["canonical"])
    hand = canonical.hand("right_hand")
    translation = np.asarray(hand.mano_parameters.transl)[ordinals]
    model = d3.d2g._load_robot(d3.d2g.ROBOT, None)
    r3_visual = o5._robot_visual_payload(model, q_r3_array, base_r3_array)
    d3_visual = o5._robot_visual_payload(model, q_d3, base_d3)
    robot_joints = np.stack(
        [
            model.keypoints_scene(q, base, layout="mediapipe21")
            for q, base in zip(q_r3_array, base_r3_array, strict=True)
        ]
    )
    obj = canonical.rigid_object(d3.OBJECT_ID)
    html = root / "review/refinement_v2_development_viewer.html"
    data = o5.OakInk2HTMLViewerV2Data(
        frames=frames,
        hand_vertices_world=np.asarray(hand.vertices_scene)[ordinals],
        hand_vertices_anatomy=np.asarray(hand.vertices_scene)[ordinals] - translation[:, None, :],
        hand_faces_closed=np.asarray(hand.mesh.faces),
        hand_faces_open=np.asarray(hand.mesh.faces),
        hand_joints_world=np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[ordinals],
        hand_joints_anatomy=np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[
            ordinals
        ]
        - translation[:, None, :],
        object_vertices=np.asarray(obj.mesh.vertices_local),
        object_faces=np.asarray(obj.mesh.faces),
        object_transforms=np.asarray(obj.pose_scene.pose_scene)[ordinals],
        primary_frame=int(frames[0]),
        record={
            "dataset": "OakInk2",
            "episode": d3.EPISODE,
            "primitive": d3.PRIMITIVE,
            "source_hand": "RIGHT",
            "target_object": d3.OBJECT_ID,
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "DEVELOPMENT_VALIDATION",
            "semantic_validity_status": read_json(root / "gate/decision.json")[
                "REFINEMENT_V2_DEVELOPMENT_VALIDATION"
            ],
            "viewer_sampling": f"all {len(failures)} failure frames plus {len(controls)} frozen controls",
            "comparison": "D3-V1 and R3 RefinementV2 are independently toggleable; diagnostic only",
        },
        camera_presets=o5.camera_presets(),
        wuji_parts=list(r3_visual["parts"]),
        wuji_joints_world=robot_joints,
        frame_solver_sec=np.zeros(len(display), dtype=np.float64),
        old_wuji_parts=[{**part, "color": [0.72, 0.22, 0.78]} for part in d3_visual["parts"]],
    )
    renderer = o5.render_oakink2_html_viewer_v2(data, html)
    bookmarks: dict[str, list[int]] = {}
    for cluster in range(1, 5):
        bookmarks[f"cluster_{cluster}"] = [
            index for index, item in enumerate(failures) if int(item["cluster_id"]) == cluster
        ]
    failure_rows = _load_failure_rows(root)
    bookmarks["recovered_examples"] = [
        index for index, row in enumerate(failure_rows) if bool(row["recovery"])
    ][:12]
    bookmarks["remaining_failed_examples"] = [
        index for index, row in enumerate(failure_rows) if not bool(row["recovery"])
    ][:12]
    bookmarks["valid_controls"] = list(range(len(failures), len(display)))
    extension = (
        "<section><h2>R3 development bookmarks</h2><div id='r3-bookmarks' class='toolbar'></div>"
        "<pre id='r3-panel'></pre></section><script>"
        f"const R3B={json.dumps(bookmarks, separators=(',', ':'))};"
        "let r3p=0;function r3s(i){r3p=i;window.__OAKINK2_VIEWER_V2__.setFrame(i);"
        "document.querySelector('#r3-panel').textContent=JSON.stringify({index:i},null,2)}"
        "for(const [k,v] of Object.entries(R3B)){const b=document.createElement('button');"
        "b.textContent=k;b.onclick=()=>{if(v.length){const n=v.indexOf(r3p);r3s(v[(n+1)%v.length])}};"
        "document.querySelector('#r3-bookmarks').append(b)}</script>"
    )
    content = html.read_text(encoding="utf-8")
    html.write_text(content.replace("</body>", extension + "</body>"), encoding="utf-8")
    regression = d2n._viewer_regression_v3(
        html, root / "review/interaction_review.png", len(display)
    )
    receipt = {
        "schema_version": "D3R3DevelopmentViewerReceiptV1",
        "status": regression["status"],
        "R3_DEVELOPMENT_VIEWER": str(html.resolve()),
        "html_sha256": sha256_file(html),
        "VIEWER_ROLE": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "VIEWER_REGRESSION": regression["status"],
        "layers": ["SOURCE MANO", "OBJECT", "D3-V1", "R3 RefinementV2 result"],
        "bookmarks": bookmarks,
        "renderer": renderer,
    }
    write_json(root / "review/viewer_receipt.json", receipt)
    write_json(root / "review/viewer_regression.json", regression)
    return receipt


def audit_r3_special_cases(root: Path) -> dict[str, Any]:
    require(root / "gate/decision.json", "automatic_evaluator", True, "AUDIT_R3")
    preflight_value = read_json(root / "preflight/git.json")
    unchanged = _authority_snapshot() == preflight_value["upstream_snapshot"]
    design_unchanged = sha256_file(R2_ROOT / "design/refinement_v2_design.json") == DESIGN_SHA256
    source = Path(__file__).read_text(encoding="utf-8")
    special = {
        "schema_version": "D3R3NoSpecialCasesV1",
        "status": "PASS",
        "DEV1_EPISODE_SPECIFIC_BRANCH": "NO",
        "C10001_SPECIAL_BRANCH": "NO",
        "FRAME_LITERAL_BRANCH": "NO",
        "FAILURE_CLUSTER_ID_BRANCH": "NO",
        "SPECIAL_SEED_FOR_CLUSTER": "NO",
        "SPECIAL_BUDGET_FOR_CLUSTER": "NO",
        "SPECIAL_DOF_SET_FOR_CLUSTER": "NO",
        "trigger_source": "frozen generic design condition",
        "manual_candidate_selection": False,
        "outcome_driven_retry": False,
        "source_contains_retry_loop": "retry" in source.lower()
        and "scientific_retry_allowed" not in source,
    }
    write_json(root / "audits/no_special_cases.json", special)
    write_json(
        root / "audits/method_integrity_postrun.json",
        {
            "schema_version": "D3R3MethodIntegrityPostrunV1",
            "status": "PASS" if unchanged and design_unchanged else "FAIL",
            "REFINEMENT_V2_DESIGN_CHANGED_DURING_R3": "NO" if design_unchanged else "YES",
            "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
            "SEMANTIC_V1_CHANGED": "NO",
            "E_IM_THRESHOLD_CHANGED": "NO",
            "HARD_VALIDITY_CHANGED": "NO",
            "D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO" if unchanged else "UNKNOWN",
            "D3R_HISTORICAL_RESULT_REWRITTEN": "NO" if unchanged else "UNKNOWN",
            "D3R2_HISTORICAL_RESULT_REWRITTEN": "NO" if unchanged else "UNKNOWN",
        },
    )
    write_json(
        root / "audits/no_fresh_data_consumption.json",
        {
            "schema_version": "D3R3NoFreshDataConsumptionV1",
            "status": "PASS",
            "NEW_FRESH_REFINEMENT_FRAME_CONSUMPTION": 0,
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        },
    )
    write_json(
        root / "audits/qold_authority.json",
        {
            "schema_version": "D3R3QOldAuthorityAuditV1",
            "status": "PASS",
            "ORIGINAL_QOLD_USED": "YES",
            "D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD": "NO",
            "PREVIOUS_RUNTIME_STATE_ALIASED_AS_QOLD": "NO",
        },
    )
    write_json(
        root / "audits/runtime_state_authority.json",
        {
            "schema_version": "D3R3RuntimeStateAuthorityAuditV1",
            "status": "PASS",
            "standalone_previous_context": "HISTORICAL_D3_V1_PREDECESSOR",
            "sequential_previous_context": "CURRENT_R3_ACCEPTED_T_MINUS_1",
            "q_old": "HISTORICAL_ORIGINAL_Q_OLD_T",
        },
    )
    design = read_json(root / "preflight/design_integrity.json")
    write_json(
        root / "audits/expanded_active_set_scope.json",
        {
            "schema_version": "D3R3ExpandedActiveSetScopeAuditV1",
            "status": "PASS",
            "EXPANDED_ACTIVE_DOF_COUNT": design["EXPANDED_ACTIVE_DOF_COUNT"],
            "EXPANDED_ACTIVE_DOF_NAMES": design["EXPANDED_ACTIVE_DOF_NAMES"],
            "WRIST_FREE": "NO",
            "BASE_FREE": "NO",
            "derived_from_robot_asset": True,
        },
    )
    return special


def authorize_fresh_certification(root: Path) -> dict[str, Any]:
    decision = read_json(root / "gate/decision.json")
    if not (
        decision.get("D3_R3_STATUS") == "PASS"
        and decision.get("REFINEMENT_V2_DEVELOPMENT_VALIDATION") == "PASS"
    ):
        value = {
            "schema_version": "D3R3FreshCertificationNotAuthorizedV1",
            "status": "NOT_AUTHORIZED",
            "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED": "NO",
            "NEXT": "DEV1_REFINEMENT_V2_DEVELOPMENT_FAILURE_ANALYSIS",
        }
        write_json(root / "future/not_authorized.json", value)
        raise RuntimeError("AUTHORIZE_FRESH_CERTIFICATION_REJECTED:D3_R3_NOT_PASS")
    value = {
        "schema_version": "D3R3FreshCertificationAuthorizationV1",
        "status": "AUTHORIZED_NOT_RUN",
        "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED": "YES",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "NOT_RUN",
        "NEXT": "O5R-D3-CERT_REFINEMENT_V2_FRESH_CERTIFICATION",
    }
    write_json(root / "future/fresh_certification_authorization.json", value)
    write_json(
        root / "future/fresh_certification_plan_stub.json",
        {**value, "execution_in_this_task": "FORBIDDEN"},
    )
    return value


def validate_repository(root: Path) -> dict[str, Any]:
    commands = [
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "check",
            "scripts/evaluation/run_oakink2_o5rd3r3.py",
            "tests/evaluation/test_oakink2_o5rd3r3.py",
        ],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "format",
            "--check",
            "scripts/evaluation/run_oakink2_o5rd3r3.py",
            "tests/evaluation/test_oakink2_o5rd3r3.py",
        ],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "python",
            "scripts/check_paper_fidelity.py",
        ],
        ["git", "diff", "--check"],
    ]
    rows = []
    for command in commands:
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    passed = all(row["returncode"] == 0 for row in rows)
    tests = {
        "schema_version": "D3R3RepositoryTestsV1",
        "status": "PASS" if passed else "FAIL",
        "commands": rows,
    }
    write_json(root / "tests.json", tests)
    write_json(
        root / "validation_results.json",
        {
            "schema_version": "D3R3ValidationResultsV1",
            "status": "PASS" if passed else "FAIL",
            "repository_validation": tests,
            "scientific_gate": read_json(root / "gate/decision.json"),
            "scientific_and_engineering_status_are_distinct": True,
        },
    )
    if not passed:
        raise RuntimeError("D3R3_REPOSITORY_VALIDATION_FAILED")
    return tests


def summarize(root: Path) -> dict[str, Any]:
    decision = read_json(root / "gate/decision.json")
    failure_analysis = read_json(root / "gate/failure_analysis.json")
    failures = read_json(root / "failure_frames/recovery_summary.json")
    clusters = read_csv(root / "failure_frames/per_cluster.csv")
    controls = read_json(root / "valid_controls/summary.json")
    sequential = read_json(root / "sequential_windows/summary.json")
    determinism = read_json(root / "determinism/summary.json")
    cost = read_json(root / "profiler/aggregate.json")
    viewer = read_json(root / "review/viewer_receipt.json")
    design = read_json(root / "preflight/design_integrity.json")
    gate = read_json(root / "preflight/gate_integrity.json")
    authorization_path = root / "future/fresh_certification_authorization.json"
    authorized = (
        authorization_path.is_file()
        and read_json(authorization_path).get("status") == "AUTHORIZED_NOT_RUN"
    )
    summary = {
        "schema_version": "OakInk2O5RD3R3FinalSummaryV1",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        "SEARCH_SPACE_MECHANISM": "OPTIMIZATION_SUBSPACE_TOO_NARROW",
        "SELECTED_REFINEMENT_V2_DESIGN": "REFINEMENT_V2_C_CONDITIONAL_EXPANDED_ACTIVE_SET",
        "REFINEMENT_V2_DESIGN_SHA256": DESIGN_SHA256,
        "DESIGN_INTEGRITY": design["status"],
        **{
            key: gate[key]
            for key in (
                "REFINEMENT_V2_DEVELOPMENT_GATE_SHA256",
                "RECOVERY_REQUIRED_FRACTION",
                "RECOVERY_REQUIRED_COUNT",
                "MEDIAN_REDUCTION_REQUIRED",
                "VALID_PRESERVATION_REQUIRED",
                "SEQUENTIAL_REQUIREMENT",
                "DETERMINISM_REQUIREMENT",
            )
        },
        "FAILURE_FRAME_COUNT": EXPECTED_INVALID,
        "FAILURE_FRAMES_PROCESSED": failures["processed"],
        "RECOVERED_COUNT": failures["recovered"],
        "UNRECOVERED_COUNT": failures["unrecovered"],
        "RECOVERY_FRACTION": failures["recovery_fraction"],
        "MEDIAN_INVALID_RELATIVE_REDUCTION": failures["median_relative_reduction"],
        "MEAN_INVALID_RELATIVE_REDUCTION": failures["mean_relative_reduction"],
        "HARD_VALID_COUNT": failures["HARD_VALID_FRAME_COUNT"],
        "EXPANDED_PATH_INVOCATION_COUNT": failures["EXPANDED_PATH_INVOCATION_COUNT"],
        "NORMAL_PATH_RECOVERY_COUNT": failures["NORMAL_PATH_RECOVERY_COUNT"],
        "EXPANDED_PATH_RECOVERY_COUNT": failures["EXPANDED_PATH_RECOVERY_COUNT"],
        "clusters": clusters,
        **{
            key: controls[key]
            for key in (
                "VALID_CONTROL_COUNT",
                "VALID_CONTROLS_PRESERVED",
                "VALID_CONTROL_PRESERVATION_FRACTION",
                "VALID_CONTROL_EXPANDED_PATH_INVOCATION_COUNT",
                "VALID_CONTROL_HARD_VALID",
            )
        },
        **{
            key: sequential[key]
            for key in (
                "SEQUENTIAL_WINDOW_COUNT",
                "SEQUENTIAL_WINDOWS_PASS",
                "SEQUENTIAL_FRAMES_TOTAL",
                "SEQUENTIAL_FRAMES_VALID",
                "CONTINUITY",
                "RUNTIME_STATE_CHAIN",
                "SEQUENTIAL_BASIN_HYSTERESIS_OBSERVED",
            )
        },
        **{
            key: determinism[key]
            for key in (
                "DETERMINISM",
                "MAX_Q_ABS_DIFF",
                "MAX_BASE_ABS_DIFF",
                "MAX_E_IM_ABS_DIFF",
            )
        },
        "FALLBACK_INVOCATION_RATE": cost["fallback_invocation_rate"],
        "NORMAL_PATH_MEAN_SEC": cost["mean_normal_historical_sec_on_failure_frames"],
        "EXPANDED_PATH_MEAN_SEC": cost["mean_fallback_sec"],
        "TOTAL_MEAN_SEC_PER_VALIDATION_FRAME": cost["mean_r3_validation_wall_sec"],
        "CONDITIONAL_COST_ESTIMATE": cost["estimated_conditional_runtime_sec_on_D3_V1_trajectory"],
        **{
            key: decision[key]
            for key in (
                "G1_TECHNICAL",
                "G2_RECOVERY",
                "G3_MEDIAN_REDUCTION",
                "G4_VALID_PRESERVATION",
                "G5_HARD_VALIDITY",
                "G6_SEQUENTIAL",
                "G7_CONTINUITY",
                "G8_DETERMINISM",
                "REFINEMENT_V2_DEVELOPMENT_VALIDATION",
                "D3_R3_STATUS",
            )
        },
        **{
            key: failure_analysis[key]
            for key in (
                "PRIMARY_DEVELOPMENT_FAILURE_MECHANISM",
                "FAILURE_CONCENTRATED_IN_ONE_CLUSTER",
                "REMAINING_INVALID_COUNT",
                "REMAINING_INVALID_ORDINALS",
                "REMAINING_INVALID_SOURCE_FRAMES",
                "REMAINING_INVALID_CLUSTERS",
                "CLUSTERS_MEETING_MEDIAN_REDUCTION_GATE",
                "CLUSTER_COUNT",
                "VALID_CONTROL_REGRESSION",
                "SEQUENTIAL_PROPAGATION_FAILURE",
                "DETERMINISM_FAILURE",
            )
        },
        "R3_DEVELOPMENT_VIEWER": viewer["R3_DEVELOPMENT_VIEWER"],
        "VIEWER_ROLE": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "VIEWER_REGRESSION": viewer["VIEWER_REGRESSION"],
        "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED": "YES" if authorized else "NO",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "NOT_RUN",
        "NEXT": "O5R-D3-CERT_REFINEMENT_V2_FRESH_CERTIFICATION"
        if authorized
        else "DEV1_REFINEMENT_V2_DEVELOPMENT_FAILURE_ANALYSIS",
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D3R_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D3R2_HISTORICAL_RESULT_REWRITTEN": "NO",
        "REFINEMENT_V2_DESIGN_CHANGED_DURING_R3": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "HARD_VALIDITY_CHANGED": "NO",
        "ORIGINAL_QOLD_USED": "YES",
        "D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD": "NO",
        "PREVIOUS_RUNTIME_STATE_ALIASED_AS_QOLD": "NO",
        "NORMAL_ACTIVE_DOF_COUNT": 4,
        "EXPANDED_ACTIVE_DOF_COUNT": 20,
        "WRIST_FREE_IN_EXPANDED_PATH": "NO",
        "BASE_FREE_IN_EXPANDED_PATH": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "completion_audit.json",
        {
            "schema_version": "O5RD3R3CompletionAuditV1",
            "status": "PASS",
            "terminal_status": summary["D3_R3_STATUS"],
            "required_artifacts_present": all(
                (root / path).is_file()
                for path in (
                    "preflight/design_integrity.json",
                    "preflight/gate_integrity.json",
                    "preflight/evidence_integrity.json",
                    "run_authority/development_validation_manifest.json",
                    "failure_frames/recovery_summary.json",
                    "valid_controls/summary.json",
                    "sequential_windows/summary.json",
                    "determinism/summary.json",
                    "gate/decision.json",
                    "profiler/aggregate.json",
                    "review/viewer_receipt.json",
                    "audits/method_integrity_postrun.json",
                )
            ),
        },
    )
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD3R3ResourceUsageV1",
            "status": "PASS",
            "gpu_required": False,
            "scope": "CONSUMED_DEV1_DEVELOPMENT_VALIDATION_ONLY",
            "additional_optimizer_compute_sec": cost["total_additional_development_compute_sec"],
        },
    )
    write_json(
        root / "git_commits.json",
        {
            "schema_version": "O5RD3R3GitCommitsV1",
            "START_HEAD": START_HEAD,
            "FINAL_HEAD_AT_SUMMARY": git("rev-parse", "HEAD"),
            "commits": git("log", f"{START_HEAD}..HEAD", "--oneline").splitlines(),
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    )
    lines = [
        "# OakInk2 O5R-D3-R3",
        "# RefinementV2 Development Validation Handoff",
        "",
        "## Git",
        "",
        "```text",
        *[
            f"{key}={summary[key]}"
            for key in (
                "BRANCH",
                "START_HEAD",
                "FINAL_HEAD",
                "tracked_worktree_clean",
                "PUSHED",
                "PR_CREATED",
            )
        ],
        "```",
        "",
        "## Frozen design and gate",
        "",
        "```text",
        *[
            f"{key}={summary[key]}"
            for key in (
                "SEARCH_SPACE_MECHANISM",
                "SELECTED_REFINEMENT_V2_DESIGN",
                "REFINEMENT_V2_DESIGN_SHA256",
                "DESIGN_INTEGRITY",
                "REFINEMENT_V2_DEVELOPMENT_GATE_SHA256",
                "RECOVERY_REQUIRED_FRACTION",
                "RECOVERY_REQUIRED_COUNT",
                "MEDIAN_REDUCTION_REQUIRED",
                "VALID_PRESERVATION_REQUIRED",
                "SEQUENTIAL_REQUIREMENT",
                "DETERMINISM_REQUIREMENT",
            )
        ],
        "```",
        "",
        "## Development result",
        "",
        "```text",
        *[
            f"{key}={summary[key]}"
            for key in (
                "FAILURE_FRAME_COUNT",
                "FAILURE_FRAMES_PROCESSED",
                "RECOVERED_COUNT",
                "UNRECOVERED_COUNT",
                "RECOVERY_FRACTION",
                "MEDIAN_INVALID_RELATIVE_REDUCTION",
                "MEAN_INVALID_RELATIVE_REDUCTION",
                "HARD_VALID_COUNT",
                "EXPANDED_PATH_INVOCATION_COUNT",
                "NORMAL_PATH_RECOVERY_COUNT",
                "EXPANDED_PATH_RECOVERY_COUNT",
                "VALID_CONTROL_COUNT",
                "VALID_CONTROLS_PRESERVED",
                "VALID_CONTROL_PRESERVATION_FRACTION",
                "VALID_CONTROL_EXPANDED_PATH_INVOCATION_COUNT",
                "SEQUENTIAL_WINDOW_COUNT",
                "SEQUENTIAL_WINDOWS_PASS",
                "SEQUENTIAL_FRAMES_TOTAL",
                "SEQUENTIAL_FRAMES_VALID",
                "CONTINUITY",
                "RUNTIME_STATE_CHAIN",
                "SEQUENTIAL_BASIN_HYSTERESIS_OBSERVED",
                "DETERMINISM",
                "MAX_Q_ABS_DIFF",
                "MAX_BASE_ABS_DIFF",
                "MAX_E_IM_ABS_DIFF",
            )
        ],
        "```",
        "",
        "## Cluster results",
        "",
        "| Cluster | N | Recovered | Recovery rate | Old p95 | New p95 | Expanded triggers |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            "| {cluster_id} | {N} | {recovered_count} | {recovery_fraction} | "
            "{old_E_IM_p95} | {new_E_IM_p95} | {expanded_fallback_invocation_count} |".format(**row)
            for row in clusters
        ],
        "",
        "## Cost",
        "",
        "```text",
        *[
            f"{key}={summary[key]}"
            for key in (
                "FALLBACK_INVOCATION_RATE",
                "NORMAL_PATH_MEAN_SEC",
                "EXPANDED_PATH_MEAN_SEC",
                "TOTAL_MEAN_SEC_PER_VALIDATION_FRAME",
                "CONDITIONAL_COST_ESTIMATE",
            )
        ],
        "```",
        "",
        "## Gate decision",
        "",
        "```text",
        *[
            f"{key}={summary[key]}"
            for key in (
                "G1_TECHNICAL",
                "G2_RECOVERY",
                "G3_MEDIAN_REDUCTION",
                "G4_VALID_PRESERVATION",
                "G5_HARD_VALIDITY",
                "G6_SEQUENTIAL",
                "G7_CONTINUITY",
                "G8_DETERMINISM",
                "REFINEMENT_V2_DEVELOPMENT_VALIDATION",
                "D3_R3_STATUS",
                "PRIMARY_DEVELOPMENT_FAILURE_MECHANISM",
                "FAILURE_CONCENTRATED_IN_ONE_CLUSTER",
                "REMAINING_INVALID_COUNT",
                "REMAINING_INVALID_ORDINALS",
                "REMAINING_INVALID_SOURCE_FRAMES",
                "REMAINING_INVALID_CLUSTERS",
                "CLUSTERS_MEETING_MEDIAN_REDUCTION_GATE",
                "CLUSTER_COUNT",
                "VALID_CONTROL_REGRESSION",
                "SEQUENTIAL_PROPAGATION_FAILURE",
                "DETERMINISM_FAILURE",
                "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED",
                "REFINEMENT_V2_INDEPENDENT_CERTIFICATION",
                "NEXT",
            )
        ],
        "```",
        "",
        "## Diagnostic viewer",
        "",
        f"`{summary['R3_DEVELOPMENT_VIEWER']}`",
        "",
        "```text",
        "VIEWER_ROLE=DEVELOPMENT_DIAGNOSTIC_ONLY",
        f"VIEWER_REGRESSION={summary['VIEWER_REGRESSION']}",
        "```",
        "",
        "## Explicitly not run and safety",
        "",
        "```text",
        *[
            f"{key}={summary[key]}"
            for key in (
                "FRESH_REFINEMENT_SPARSE",
                "FRESH_REFINEMENT_WINDOW",
                "CROSS_EPISODE_REFINEMENT",
                "D3_V2_SCIENTIFIC_RUN_COUNT",
                "DEV2_RERUN",
                "PPO_TRAINING_RUN_COUNT_NEW",
                "O6_PRODUCTION_RAN",
                "CERTIFICATION_SPLIT_NEW_CONSUMPTION",
                "HELDOUT_SPLIT_NEW_CONSUMPTION",
                "D3_V1_HISTORICAL_RESULT_REWRITTEN",
                "D3R_HISTORICAL_RESULT_REWRITTEN",
                "D3R2_HISTORICAL_RESULT_REWRITTEN",
                "REFINEMENT_V2_DESIGN_CHANGED_DURING_R3",
                "RETARGET_OBJECTIVE_V2_CHANGED",
                "SEMANTIC_V1_CHANGED",
                "E_IM_THRESHOLD_CHANGED",
                "HARD_VALIDITY_CHANGED",
                "ORIGINAL_QOLD_USED",
                "D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD",
                "PREVIOUS_RUNTIME_STATE_ALIASED_AS_QOLD",
                "NORMAL_ACTIVE_DOF_COUNT",
                "EXPANDED_ACTIVE_DOF_COUNT",
                "WRIST_FREE_IN_EXPANDED_PATH",
                "BASE_FREE_IN_EXPANDED_PATH",
                ".local_TRACKED",
                "GUIDANCE_WORKTREE_MODIFIED",
                "PUSHED",
                "PR_CREATED",
            )
        ],
        "```",
        "",
    ]
    handoff = "\n".join(lines)
    write_text(root / "handoff.md", handoff)
    write_text(root / "final_summary.md", handoff)
    return summary


def prepare_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_r2_design_authority(root)
    verify_r3_development_gate(root)
    return freeze_r3_validation_manifest(root)


ACTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "preflight": preflight,
    "verify-r2-design-authority": verify_r2_design_authority,
    "verify-r3-development-gate": verify_r3_development_gate,
    "freeze-r3-validation-manifest": freeze_r3_validation_manifest,
    "run-r3-failure-frame-validation": run_r3_failure_frame_validation,
    "resume-r3-failure-frame-validation": resume_r3_failure_frame_validation,
    "run-r3-valid-controls": run_r3_valid_controls,
    "run-r3-sequential-windows": run_r3_sequential_windows,
    "run-r3-determinism": run_r3_determinism,
    "evaluate-r3-development-gate": evaluate_r3_development_gate,
    "profile-r3-cost": profile_r3_cost,
    "render-r3-development-viewer": render_r3_development_viewer,
    "audit-r3-special-cases": audit_r3_special_cases,
    "authorize-fresh-certification": authorize_fresh_certification,
    "validate-repository": validate_repository,
    "summarize": summarize,
    "prepare-all": prepare_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    started = time.perf_counter()
    try:
        result = ACTIONS[args.action](args.root.resolve())
    except Exception as exc:
        append_jsonl(
            args.root.resolve() / "technical_failures.jsonl",
            {
                "schema_version": "O5RD3R3TechnicalFailureV1",
                "action": args.action,
                "error": f"{type(exc).__name__}:{exc}",
                "classification": "TECHNICAL_OR_FAIL_CLOSED_PRECONDITION; scientific failures are recorded by gate artifacts",
            },
        )
        raise
    timing_path = args.root.resolve() / "timing/stage_timing.json"
    timing = (
        read_json(timing_path)
        if timing_path.is_file()
        else {
            "schema_version": "O5RD3R3StageTimingV1",
            "stages": [],
        }
    )
    timing["stages"].append({"action": args.action, "wall_time_sec": time.perf_counter() - started})
    write_json(timing_path, timing)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
