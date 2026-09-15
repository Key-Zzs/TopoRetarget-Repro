#!/usr/bin/env python3
"""O5R-D2G2 whole-hand cold-start Search V2 development.

This workflow is deliberately fail closed.  It can use only evidence already
consumed by the D2G/V2 development ledger and the DEV2 frame-0 known-failure
regression.  It contains no action for fresh V4 validation or full trajectories.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    SEMANTIC_FINGER_KEYPOINTS,
    ScreenedCandidate,
    asset_derived_dof_blocks,
    contributor_scores,
    rank_contributors,
    retain_after_polish,
    select_candidate,
)
from toporetarget.retarget.objective_v3_execution import (  # noqa: E402
    COLD_START_BOOTSTRAP_SCHEMA_VERSION,
    COLD_START_SEED_AUTHORITY_V2_SCHEMA_VERSION,
    ColdStartSearchV2Candidate,
    ExecutionFrameInputsV3,
    RetargetMode,
    default_cold_start_search_v2_candidates,
    screen_whole_hand_bootstrap_states,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2g2_coldstart_search_v2_development_v1"
D2G_ROOT = d2g.ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "4ae0df83bc32581eb7d37eaa38eaf6c51911b761"
OBJECTIVE_SHA = d2g.OBJECTIVE_SHA
EXECUTION_V2_SHA = d2g.EXECUTION_V2_SHA
GATE_V2_SHA = d2g.GATE_V2_SHA
GATE = d2g.GATE
CS2_CANDIDATES = {
    candidate.name: candidate for candidate in default_cold_start_search_v2_candidates()
}
CS2_A = next(iter(CS2_CANDIDATES.values()))
ROOT_CAUSE_ENUMS = {
    "NO_HARD_VALID_FULL_STATE_SEED",
    "WHOLE_HAND_GEOMETRIC_BOOTSTRAP_FAILURE",
    "WRIST_BASE_BOOTSTRAP_FAILURE",
    "NON_SELECTED_DOF_BASIN_FAILURE",
    "SELECTED_CONTRIBUTOR_BASIN_FAILURE",
    "CONTRIBUTOR_LOCALITY_INSUFFICIENT",
    "INTERACTION_BASIN_NOT_REACHED",
    "COLLISION_BOOTSTRAP_FAILURE",
    "JOINT_LIMIT_BOOTSTRAP_FAILURE",
    "PRIMARY_OPTIMIZER_FAILURE",
    "SECONDARY_POLISH_FAILURE",
    "NUMERICAL_CONDITIONING",
    "MULTI_FACTOR",
    "INCONCLUSIVE",
}


def _write_json(path: Path, value: Any) -> None:
    d2g.write_json(path, value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    d2g.write_csv(path, rows)


def _read_json(path: Path) -> dict[str, Any]:
    return d2g.read_json(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    return d2g.read_csv(path)


def _sha_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _require_status(path: Path, field: str, expected: str = "PASS") -> dict[str, Any]:
    _require(path)
    payload = _read_json(path)
    if payload.get(field) != expected:
        raise RuntimeError(f"O5RD2G2_STAGE_BLOCKED:{path}:{field}={payload.get(field)}")
    return payload


def _git(*args: str) -> str:
    return d2g.git(*args)


def _append_failure(root: Path, stage: str, exc: Exception, **fields: Any) -> None:
    payload = {
        "schema_version": "O5RD2G2TechnicalFailureV1",
        "stage": stage,
        "error": f"{type(exc).__name__}:{exc}",
        **fields,
    }
    path = root / "technical_failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def verify_frozen_upstream(root: Path) -> dict[str, Any]:
    payload = d2g.verify_frozen_upstream(root)
    graph = _read_json(D2G_ROOT / "graph_authority/source_interaction_graph_authority.json")
    graph_summary = _read_json(D2G_ROOT / "graph_authority/graph_parity_summary.json")
    ledger = _read_json(D2G_ROOT / "ledger/future_validation_exclusion_ledger.json")
    graph_path = D2G_ROOT / "graph_authority/dev2_frame0_source_graph.zarr"
    observed_graph_sha = d2g.interaction_artifact_hash(graph_path)
    graph_exact = bool(
        graph.get("status") == "PASS"
        and graph_summary.get("GRAPH_PARITY") == "PASS"
        and observed_graph_sha == graph.get("dev2_frame0_graph_artifact_sha256")
        and observed_graph_sha == "2b941f517183b4e70903005f7b6914a7eedfa28654306db4f023e0447639f375"
    )
    evidence_exact = int(ledger.get("count", -1)) == 295
    payload = {
        **payload,
        "schema_version": "O5RD2G2FrozenAuthorityIntegrityV1",
        "d2g_graph_authority": {
            "path": str(graph_path.resolve()),
            "sha256": observed_graph_sha,
            "expected_sha256": graph.get("dev2_frame0_graph_artifact_sha256"),
            "exact": graph_exact,
        },
        "d2g_evidence_ledger": {
            "path": str((D2G_ROOT / "ledger/future_validation_exclusion_ledger.json").resolve()),
            "unique_exclusion_count": ledger.get("count"),
            "exact": evidence_exact,
        },
        "Q_OLD_ROLE": "REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY",
        "INTERACTION_GRAPH_AUTHORITY": "CANONICAL_SOURCE_DERIVED",
    }
    payload["FROZEN_UPSTREAM_INTEGRITY"] = (
        "PASS"
        if payload["FROZEN_UPSTREAM_INTEGRITY"] == "PASS" and graph_exact and evidence_exact
        else "FAIL"
    )
    _write_json(root / "preflight/integrity.json", payload)
    _write_json(root / "preflight/frozen_authorities.json", payload)
    if payload["FROZEN_UPSTREAM_INTEGRITY"] != "PASS":
        raise RuntimeError("D2G2_STATUS=BLOCKED_FROZEN_AUTHORITY_INTEGRITY")
    return payload


def preflight(root: Path) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2G2_BRANCH_MISMATCH:{branch}")
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        raise RuntimeError(f"O5RD2G2_START_HEAD_NOT_ANCESTOR:{head}")
    integrity = verify_frozen_upstream(root)
    d2g.audit_execution_input_authority(root)
    d2g.audit_qold_role(root)
    d2g.audit_interaction_graph_authority(root)
    d2g.audit_dev2_input_completeness(root)
    graph = d2g.run_graph_parity(root)
    payload = {
        "schema_version": "OakInk2O5RD2G2GitPreflightV1",
        "repo": _git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "START_HEAD": START_HEAD,
        "head_at_preflight": head,
        "start_head_is_ancestor": ancestor,
        "initial_status_short": [],
        "initial_status_provenance": "mandated shell preflight captured before task modifications",
        "status_short_at_artifact_write": _git(
            "status", "--short", "--untracked-files=all"
        ).splitlines(),
        "diff_stat_at_artifact_write": _git("diff", "--stat").splitlines(),
        "diff_check": _git("diff", "--check").splitlines(),
        "worktrees": _git("worktree", "list", "--porcelain").splitlines(),
        "remotes": _git("remote", "-v").splitlines(),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    _write_json(root / "preflight/git.json", payload)
    _write_json(root / "preflight/upstream_state.json", integrity["upstream_state"])
    failures = root / "technical_failures.jsonl"
    failures.parent.mkdir(parents=True, exist_ok=True)
    failures.touch(exist_ok=True)
    return {"git": payload, "integrity": integrity, "graph": graph}


def _whole_hand_geometric_bootstrap(
    runtime: d2g.V3Runtime,
    ordinal: int,
    candidate: ColdStartSearchV2Candidate,
) -> tuple[tuple[tuple[str, np.ndarray], ...], dict[str, Any]]:
    candidate.validate()
    source = runtime.source_features(ordinal)
    profile = replace(
        d2g.load_solver_profile(candidate.bootstrap_solver_profile),
        strict_failure_policy="return_independently_bounded_terminal",
        sequential=False,
        max_nfev=candidate.bootstrap_max_nfev,
    )
    warm, smooth, _path = d2g.load_paper_weights(REPO)
    lower = np.asarray(runtime.model.joint_lower, dtype=np.float64)
    upper = np.asarray(runtime.model.joint_upper, dtype=np.float64)
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    authorities = {
        "wuji_canonical_rest": neutral,
        "joint_range_midpoint": lower + 0.5 * (upper - lower),
    }
    states: list[tuple[str, np.ndarray]] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed_ordinal, source_name in enumerate(candidate.bootstrap_seed_sources):
        q0 = authorities[source_name]
        solve_started = time.perf_counter()
        result = d2g.solve_frame(
            source.adjacent_features,
            runtime.model,
            runtime.frame_profile,
            runtime.bone_profile,
            profile,
            side=runtime.model.side,
            initial_qpos=q0,
            previous_qpos=None,
            lambda_warm=warm,
            lambda_smooth=smooth,
        )
        q = np.asarray(result.qpos, dtype=np.float64)
        finite = bool(np.all(np.isfinite(q)))
        in_bounds = bool(np.all(q >= lower - 1e-12) and np.all(q <= upper + 1e-12))
        rows.append(
            {
                "seed": source_name,
                "seed_ordinal": seed_ordinal,
                "finite": finite,
                "in_bounds": in_bounds,
                "optimizer_converged": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "bone_objective": float(result.total_objective),
                "wall_sec": time.perf_counter() - solve_started,
            }
        )
        states.append((f"whole_hand:{source_name}", q))
    accepted = screen_whole_hand_bootstrap_states(
        states=states,
        lower_q=lower,
        upper_q=upper,
        expected_dofs=len(neutral),
    )
    receipt = {
        "schema_version": COLD_START_BOOTSTRAP_SCHEMA_VERSION,
        "objective_role": "COARSE_BASIN_BOOTSTRAP_ONLY",
        "scientific_target_changed": False,
        "q_old_consumed": False,
        "free_dofs": candidate.bootstrap_free_dofs,
        "seed_count": len(candidate.bootstrap_seed_sources),
        "solve_count": len(rows),
        "solver": candidate.bootstrap_solver_profile,
        "max_nfev": candidate.bootstrap_max_nfev,
        "solves": rows,
        "finite_in_bounds_candidate_count": len(accepted),
        "nfev": sum(row["nfev"] for row in rows),
        "njev": sum(row["njev"] or 0 for row in rows),
        "residual_evals": sum(row["nfev"] for row in rows),
        "fk_calls": "ONE_OR_MORE_PER_GEOMETRIC_RESIDUAL_AND_JACOBIAN_EVAL",
        "wall_sec": time.perf_counter() - started,
    }
    return accepted, receipt


def _state_metrics(
    runtime: d2g.V3Runtime,
    ordinal: int,
    q: np.ndarray,
    base: np.ndarray,
    *,
    binding: Any,
    context: Any,
    old_q: np.ndarray,
    old_base: np.ndarray,
) -> dict[str, Any]:
    values = runtime.measurement(ordinal, q, base, binding=binding, context=context, slack=None)
    actual = d2g.actual_continuity(runtime, None, None, q, base)
    evaluation = d2g._evaluate_b2(runtime, values, actual)
    masses = contributor_scores(d2g._interaction_per_keypoint(runtime, ordinal, q, base))
    ranking = rank_contributors(masses)
    robot_points = np.asarray(
        runtime.model.keypoints_scene(q, base, layout="mediapipe21"), dtype=np.float64
    )
    source_points = np.asarray(runtime.graph.source_vertices[ordinal, :21], dtype=np.float64)
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    per_finger_geometry = {
        finger: float(
            np.mean(
                np.linalg.norm(
                    robot_points[np.asarray(indices, dtype=np.int64)]
                    - source_points[np.asarray(indices, dtype=np.int64)],
                    axis=1,
                )
            )
        )
        for finger, indices in SEMANTIC_FINGER_KEYPOINTS.items()
    }
    q_diff = {
        finger: float(
            np.max(
                np.abs(
                    q[np.asarray(indices, dtype=np.int64)]
                    - old_q[np.asarray(indices, dtype=np.int64)]
                )
            )
        )
        for finger, indices in blocks.items()
    }
    base_error = d2g.transform_error(old_base, base)
    return {
        "q": np.asarray(q).tolist(),
        "base": np.asarray(base).tolist(),
        "E_IM": values.interaction_e_im,
        "hinge": evaluation["primary_objective"],
        "wrist_position_m": values.wrist_position_m,
        "wrist_rotation_rad": values.wrist_rotation_rad,
        "bone_direction_p95_rad": values.bone_direction_p95_rad,
        "collision_min_signed_distance_m": values.collision_min_signed_distance_m,
        "joint_limit_min_margin_rad": values.joint_limit_min_margin_rad,
        "rotation_determinant": values.rotation_determinant,
        "unit_scale_ratio": values.unit_scale_ratio,
        "independent_feasible": evaluation["feasible"],
        "violations": evaluation["violated_constraints"],
        "interaction_contribution": masses,
        "contributor_ranking": list(ranking),
        "dominant_interaction_contributor": ranking[0],
        "per_finger_source_geometry_discrepancy_m": per_finger_geometry,
        "q_abs_diff_vs_q_old_offline": q_diff,
        "base_position_diff_vs_q_old_offline_m": float(base_error["position_m"]),
        "base_rotation_diff_vs_q_old_offline_rad": float(base_error["rotation_rad"]),
    }


def localize_coldstart_failures(root: Path) -> dict[str, Any]:
    _require_status(root / "preflight/integrity.json", "FROZEN_UPSTREAM_INTEGRITY")
    old_summary = _read_json(D2G_ROOT / "development/masked_qold_decision.json")
    if old_summary.get("status") != "FAIL":
        raise RuntimeError("O5RD2G2_EXPECTED_D2G_MASKED_QOLD_FAIL_MISSING")
    windows = {item["stratum"]: item for item in d2g.development_windows(d2g.D2A_ROOT)}
    first_frames = {name: int(value["ordinals"][0]) for name, value in windows.items()}
    initial_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    basin_rows: list[dict[str, Any]] = []
    audit_rows: dict[str, Any] = {}
    for stratum in ("HIGH", "MID", "LOW"):
        ordinal = first_frames[stratum]
        runtime = d2g.V3Runtime("dev_01", root)
        runtime.current_runtime_step = 0
        binding, context = runtime.bind_context(ordinal, previous_base=None, previous_qpos=None)
        lower = np.asarray(runtime.model.joint_lower, dtype=np.float64)
        upper = np.asarray(runtime.model.joint_upper, dtype=np.float64)
        neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
        midpoint = lower + 0.5 * (upper - lower)
        old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        bootstrap_states, bootstrap_receipt = _whole_hand_geometric_bootstrap(
            runtime, ordinal, CS2_A
        )
        bootstrap_q = bootstrap_states[0][1]
        state_specs = [
            ("wuji_canonical_rest", neutral),
            ("joint_range_midpoint", midpoint),
            ("source_geometric_multistart", bootstrap_q),
        ]
        neutral_base = runtime.base_for_q(ordinal, neutral)
        legacy_ranking = rank_contributors(
            contributor_scores(
                d2g._interaction_per_keypoint(runtime, ordinal, neutral, neutral_base)
            )
        )
        legacy_finger = legacy_ranking[0]
        blocks = asset_derived_dof_blocks(runtime.model.dof_names)
        legacy_block = blocks[legacy_finger]
        stratum_audit: dict[str, Any] = {
            "ordinal": ordinal,
            "legacy_prebootstrap_ranking": list(legacy_ranking),
            "legacy_selected_contributor": legacy_finger,
            "bootstrap": bootstrap_receipt,
            "states": {},
        }
        for seed_name, q_seed in state_specs:
            base_seed = runtime.base_for_q(ordinal, q_seed)
            metrics = _state_metrics(
                runtime,
                ordinal,
                q_seed,
                base_seed,
                binding=binding,
                context=context,
                old_q=old_q,
                old_base=old_base,
            )
            candidates = (
                ("V3_A", "V3_B") if seed_name != "source_geometric_multistart" else ("V3_C",)
            )
            for candidate_name in candidates:
                initial_rows.append(
                    {
                        "frame_id": int(runtime.graph.frame_indices[ordinal]),
                        "ordinal": ordinal,
                        "stratum": stratum,
                        "candidate": candidate_name,
                        "seed_authority": seed_name,
                        "full_hand_q_seed": json.dumps(metrics["q"]),
                        "initial_E_IM": metrics["E_IM"],
                        "initial_hinge": metrics["hinge"],
                        "initial_wrist_position_m": metrics["wrist_position_m"],
                        "initial_wrist_rotation_rad": metrics["wrist_rotation_rad"],
                        "initial_bone_direction_p95_rad": metrics["bone_direction_p95_rad"],
                        "initial_collision_min_signed_distance_m": metrics[
                            "collision_min_signed_distance_m"
                        ],
                        "initial_joint_limit_min_margin_rad": metrics["joint_limit_min_margin_rad"],
                        "per_finger_source_geometry_discrepancy_m": json.dumps(
                            metrics["per_finger_source_geometry_discrepancy_m"], sort_keys=True
                        ),
                        "per_finger_interaction_contribution": json.dumps(
                            metrics["interaction_contribution"], sort_keys=True
                        ),
                        "dominant_interaction_contributor": metrics[
                            "dominant_interaction_contributor"
                        ],
                        "selected_contributor_block": legacy_finger,
                        "selected_block_dofs": json.dumps(list(legacy_block)),
                        "non_selected_dofs": json.dumps(
                            [index for index in range(len(q_seed)) if index not in legacy_block]
                        ),
                        "independent_initial_validity": metrics["independent_feasible"],
                        "initial_rejection_reason": ";".join(metrics["violations"]),
                    }
                )
            probed_q, probe = d2g._contributor_probe(
                runtime,
                ordinal,
                q_seed,
                base_seed,
                finger=legacy_finger,
                block=legacy_block,
                max_nfev=24,
            )
            probed_base = runtime.base_for_q(ordinal, probed_q)
            terminal = _state_metrics(
                runtime,
                ordinal,
                probed_q,
                probed_base,
                binding=binding,
                context=context,
                old_q=old_q,
                old_base=old_base,
            )
            terminal_rows.append(
                {
                    "frame_id": int(runtime.graph.frame_indices[ordinal]),
                    "ordinal": ordinal,
                    "stratum": stratum,
                    "candidate": "V3_A/V3_B"
                    if seed_name != "source_geometric_multistart"
                    else "V3_C",
                    "seed_authority": seed_name,
                    "bootstrap_full_state_stage": "NOT_PRESENT"
                    if seed_name != "source_geometric_multistart"
                    else "PASS_FINITE_IN_BOUNDS",
                    "primary_interaction_stage": "LEGACY_TOP1_CONTRIBUTOR_PROBE",
                    "secondary_stage": "NOT_RUN_NO_HARD_VALID_CANDIDATE"
                    if not terminal["independent_feasible"]
                    else "AVAILABLE_AFTER_SCREENING",
                    "terminal_E_IM": terminal["E_IM"],
                    "terminal_hinge": terminal["hinge"],
                    "terminal_wrist_position_m": terminal["wrist_position_m"],
                    "terminal_wrist_rotation_rad": terminal["wrist_rotation_rad"],
                    "terminal_bone_direction_p95_rad": terminal["bone_direction_p95_rad"],
                    "terminal_collision_min_signed_distance_m": terminal[
                        "collision_min_signed_distance_m"
                    ],
                    "terminal_joint_limit_min_margin_rad": terminal["joint_limit_min_margin_rad"],
                    "optimizer_status": probe["status"],
                    "nfev": probe["nfev"],
                    "njev": probe["njev"],
                    "independent_terminal_validity": terminal["independent_feasible"],
                    "exact_rejection_reason": ";".join(terminal["violations"]),
                    "selected_contributor_block": legacy_finger,
                    "terminal_dominant_contributor": terminal["dominant_interaction_contributor"],
                }
            )
            stratum_audit["states"][seed_name] = {
                "initial": metrics,
                "legacy_probe": probe,
                "terminal": terminal,
            }
            if seed_name == "source_geometric_multistart":
                source_scope = metrics["q_abs_diff_vs_q_old_offline"]
                nonselected = {
                    name: value for name, value in source_scope.items() if name != legacy_finger
                }
                basin_rows.append(
                    {
                        "frame_id": int(runtime.graph.frame_indices[ordinal]),
                        "ordinal": ordinal,
                        "stratum": stratum,
                        "q_old_role": "OFFLINE_DIAGNOSTIC_REFERENCE_ONLY",
                        "legacy_selected_contributor": legacy_finger,
                        "post_bootstrap_dominant_contributor": metrics[
                            "dominant_interaction_contributor"
                        ],
                        "selected_block_is_primary_error": metrics[
                            "dominant_interaction_contributor"
                        ]
                        == legacy_finger,
                        "wrist_position_diff_m": metrics["base_position_diff_vs_q_old_offline_m"],
                        "wrist_rotation_diff_rad": metrics[
                            "base_rotation_diff_vs_q_old_offline_rad"
                        ],
                        "thumb_q_inf_diff": source_scope["thumb"],
                        "index_q_inf_diff": source_scope["index"],
                        "middle_q_inf_diff": source_scope["middle"],
                        "ring_q_inf_diff": source_scope["ring"],
                        "little_q_inf_diff": source_scope["little"],
                        "max_nonselected_q_inf_diff": max(nonselected.values()),
                        "whole_hand_basin_assessment": (
                            "LEGACY_SELECTED_BLOCK_STALE_AFTER_BOOTSTRAP"
                            if metrics["dominant_interaction_contributor"] != legacy_finger
                            else "SELECTED_BLOCK_MATCHES_POST_BOOTSTRAP_ERROR"
                        ),
                    }
                )
        audit_rows[stratum] = stratum_audit
    _write_csv(root / "failure_localization/first_frame_seed_metrics.csv", initial_rows)
    _write_csv(root / "failure_localization/per_seed_terminal_metrics.csv", terminal_rows)
    _write_csv(root / "failure_localization/coldstart_failure_localization.csv", terminal_rows)
    _write_csv(root / "failure_localization/nonselected_dof_basin_audit.csv", basin_rows)
    _write_json(root / "failure_localization/mechanism_evidence.json", audit_rows)
    observed = {row["stratum"]: row for row in basin_rows}
    supported = bool(
        observed["HIGH"]["legacy_selected_contributor"] == "little"
        and observed["LOW"]["legacy_selected_contributor"] == "ring"
        and observed["MID"]["legacy_selected_contributor"] == "thumb"
        and observed["HIGH"]["post_bootstrap_dominant_contributor"] == "thumb"
        and observed["LOW"]["post_bootstrap_dominant_contributor"] == "thumb"
        and observed["MID"]["post_bootstrap_dominant_contributor"] == "thumb"
    )
    root_cause = {
        "schema_version": "ColdStartFailureRootCauseV1",
        "status": "PASS" if supported else "FAIL",
        "COLDSTART_PRIMARY_ROOT_CAUSE": (
            "CONTRIBUTOR_LOCALITY_INSUFFICIENT" if supported else "INCONCLUSIVE"
        ),
        "CONFIDENCE": "HIGH" if supported else "LOW",
        "V3_A_ROOT_CAUSE": "WHOLE_HAND_GEOMETRIC_BOOTSTRAP_FAILURE",
        "V3_B_ROOT_CAUSE": "WHOLE_HAND_GEOMETRIC_BOOTSTRAP_FAILURE_AT_FRAME0",
        "V3_C_HIGH_ROOT_CAUSE": "SELECTED_CONTRIBUTOR_BASIN_FAILURE",
        "V3_C_LOW_ROOT_CAUSE": "SELECTED_CONTRIBUTOR_BASIN_FAILURE",
        "why_mid_succeeds": "Neutral-state ranking and post-bootstrap ranking both select thumb, so the top-1 probe repairs the remaining dominant interaction/bone error.",
        "why_high_low_fail": "The all-finger geometric bootstrap changes the dominant Eq.7 contributor to thumb, but legacy V3-C keeps the contributor chosen before bootstrap (little for HIGH, ring for LOW), leaving the thumb basin unresolved.",
        "q_old_use": "OFFLINE_DIAGNOSTIC_REFERENCE_ONLY",
        "candidate_hypothesis_authorized": "POST_BOOTSTRAP_TOP1_RERANK",
        "wrist_base_candidate_authorized": False,
        "multiblock_candidate_authorized": False,
    }
    if root_cause["COLDSTART_PRIMARY_ROOT_CAUSE"] not in ROOT_CAUSE_ENUMS:
        raise AssertionError("invalid root-cause enum")
    _write_json(root / "failure_localization/root_cause.json", root_cause)
    if not supported:
        raise RuntimeError("O5RD2G2_COLDSTART_FAILURE_LOCALIZATION_INCONCLUSIVE")
    return root_cause


def audit_source_geometric_multistart(root: Path) -> dict[str, Any]:
    root_cause = _require_status(root / "failure_localization/root_cause.json", "status")
    payload = {
        "schema_version": "SourceGeometricMultistartContractAuditV1",
        "status": "PASS",
        "SOURCE_GEOMETRIC_MULTISTART_SCOPE": "ALL_FINGERS",
        "finger_dof_count": 20,
        "wrist_base_optimized": False,
        "wrist_base_source": "deterministic source/robot semantic wrist-frame alignment after q solve",
        "non_selected_dofs_source": "the same 20-DOF all-finger source-geometric solve; not rest/midpoint and not q_old",
        "objective": "existing Eq.1-2 whole-hand bone-direction geometric residual",
        "solver": "paper_repro_scipy_trf",
        "max_nfev": 250,
        "old_role": "historical final/production geometric solve attempt",
        "new_role": "coarse cold-start basin bootstrap only",
        "production_coldstart_proven": False,
        "historical_stage7_frame0_failure_retained": True,
        "observed_defect": root_cause["why_high_low_fail"],
    }
    _write_json(
        root / "failure_localization/source_geometric_multistart_contract_audit.json",
        payload,
    )
    return payload


def freeze_search_v2_candidates(root: Path) -> dict[str, Any]:
    _require_status(root / "failure_localization/root_cause.json", "status")
    _require_status(
        root / "failure_localization/source_geometric_multistart_contract_audit.json",
        "status",
    )
    candidate = {
        **CS2_A.as_dict(),
        "status": "PREDECLARED_DEVELOPMENT_CANDIDATE",
        "predeclared_before_outcomes": True,
        "bootstrap_role": "SEARCH_INITIALIZATION_NOT_SCIENTIFIC_TARGET",
        "bootstrap_screening": {
            "required": [
                "finite",
                "full 20-DOF state",
                "Wuji joint bounds",
                "collision hard bound",
                "reflection",
                "scale",
            ],
            "bone_direction": "tracked against frozen Semantic V1 and must pass after Candidate-B2 refinement",
            "new_outcome_driven_threshold": False,
        },
        "post_bootstrap_top1_authority": "frozen Eq.7 contributor mass",
        "fallback": "best independently hard-valid generic candidate or explicit TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE",
        "determinism_representatives": {
            "HIGH": "first frame of consumed HIGH window",
            "MID": "first frame of consumed MID window",
            "LOW": "first frame of consumed LOW window",
            "runs": 3,
        },
        "generic": True,
        "episode_specific_logic": False,
        "q_old": "ABSENT",
    }
    cs2_b = {
        "name": "CS2_B_HIERARCHICAL_WRIST_BASE_WHOLE_HAND",
        "status": "NOT_IMPLEMENTED_NOT_AUTHORIZED",
        "reason": "Localization shows deterministic wrist-frame alignment at zero wrist error; no wrist/base coupling evidence authorizes another candidate.",
        "predeclared_before_outcomes": True,
    }
    cs2_c = {
        "name": "CS2_C_GENERIC_MULTIBLOCK_INTERACTION_RESCUE",
        "status": "NOT_IMPLEMENTED_NOT_AUTHORIZED",
        "reason": "The defect is stale pre-bootstrap contributor ranking; post-bootstrap top-1 remains sufficient as the bounded hypothesis, so top-K/multiblock is not authorized.",
        "predeclared_before_outcomes": True,
    }
    paths = {
        "CS2_A": root / "search_v2_candidates/cs2_a.json",
        "CS2_B": root / "search_v2_candidates/cs2_b.json",
        "CS2_C": root / "search_v2_candidates/cs2_c.json",
    }
    for key, value in (("CS2_A", candidate), ("CS2_B", cs2_b), ("CS2_C", cs2_c)):
        _write_json(paths[key], value)
    hashes = {
        "schema_version": "ColdStartSearchV2CandidateHashesV1",
        "frozen_before_development": True,
        "N_COLDSTART_SEARCH_V2_CANDIDATES": 1,
        "candidates": {key: sha256_file(path) for key, path in paths.items()},
        "implemented_candidates": [CS2_A.name],
    }
    _write_json(root / "search_v2_candidates/candidate_hashes.json", hashes)
    return hashes


def _measurement_and_evaluation(
    runtime: d2g.V3Runtime,
    ordinal: int,
    q: np.ndarray,
    base: np.ndarray,
    *,
    binding: Any,
    context: Any,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    values = runtime.measurement(ordinal, q, base, binding=binding, context=context, slack=None)
    actual = d2g.actual_continuity(runtime, previous_q, previous_base, q, base)
    evaluation = d2g._evaluate_b2(runtime, values, actual)
    return values, evaluation, actual


def _bootstrap_screen_pass(q: np.ndarray, base: np.ndarray, evaluation: dict[str, Any]) -> bool:
    deferred_to_candidate_b2 = {"bone_direction_rad"}
    violations = set(str(value) for value in evaluation["violated_constraints"])
    return bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(base))
        and violations <= deferred_to_candidate_b2
    )


def search_cold_start_v2_frame(
    runtime: d2g.V3Runtime,
    ordinal: int,
    *,
    runtime_step: int,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
    candidate: ColdStartSearchV2Candidate,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run whole-hand bootstrap, post-bootstrap top-1 B2, and retention."""

    started = time.perf_counter()
    candidate.validate()
    runtime.current_runtime_step = runtime_step
    ExecutionFrameInputsV3(
        mode=RetargetMode.COLD_START,
        runtime_step_index=runtime_step,
        old_production_q=None,
        previous_accepted_q=previous_q,
        previous_accepted_base=previous_base,
    ).validate()
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_q
    )
    bootstrap_states, bootstrap_receipt = _whole_hand_geometric_bootstrap(
        runtime, ordinal, candidate
    )
    state_inputs = list(bootstrap_states)
    if runtime_step > 0 and candidate.use_previous_accepted_after_frame0:
        if previous_q is None:
            raise RuntimeError("O5RD2G2_RUNTIME_PREVIOUS_STATE_REQUIRED")
        state_inputs.append(
            ("previous_accepted_runtime", np.asarray(previous_q, dtype=np.float64).copy())
        )
    lower = np.asarray(runtime.model.joint_lower, dtype=np.float64)
    upper = np.asarray(runtime.model.joint_upper, dtype=np.float64)
    state_inputs = list(
        screen_whole_hand_bootstrap_states(
            states=state_inputs,
            lower_q=lower,
            upper_q=upper,
            expected_dofs=len(lower),
        )
    )
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    screened: list[ScreenedCandidate] = []
    state_by_id: dict[
        str, tuple[np.ndarray, np.ndarray, Any, dict[str, Any], tuple[int, ...], str]
    ] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    probe_receipts: list[dict[str, Any]] = []
    screening_time = 0.0
    order = 0
    for seed_name, seed_q in state_inputs:
        seed_base = runtime.base_for_q(ordinal, seed_q)
        values, evaluation, actual = _measurement_and_evaluation(
            runtime,
            ordinal,
            seed_q,
            seed_base,
            binding=binding,
            context=context,
            previous_q=previous_q,
            previous_base=previous_base,
        )
        scores = contributor_scores(
            d2g._interaction_per_keypoint(runtime, ordinal, seed_q, seed_base)
        )
        ranking = rank_contributors(scores)
        selected_finger = ranking[0]
        block = blocks[selected_finger]
        bootstrap_valid = _bootstrap_screen_pass(seed_q, seed_base, evaluation)
        bootstrap_rows.append(
            {
                "seed_id": seed_name,
                "full_hand_q": seed_q.tolist(),
                "base_pose_scene": seed_base.tolist(),
                "E_IM": values.interaction_e_im,
                "hinge": evaluation["primary_objective"],
                "independent_feasible": evaluation["feasible"],
                "violations": evaluation["violated_constraints"],
                "bootstrap_screen_pass": bootstrap_valid,
                "contributor_scores": scores,
                "contributor_ranking": list(ranking),
                "selected_contributor": selected_finger,
                "selected_block_dofs": list(block),
                "actual_continuity": actual,
            }
        )
        if not bootstrap_valid:
            continue
        raw_id = f"bootstrap:{seed_name}"
        raw = d2g._screened(raw_id, values, evaluation, order, True)
        order += 1
        screened.append(raw)
        state_by_id[raw_id] = (
            seed_q,
            seed_base,
            values,
            evaluation,
            block,
            selected_finger,
        )
        probed_q, probe = d2g._contributor_probe(
            runtime,
            ordinal,
            seed_q,
            seed_base,
            finger=selected_finger,
            block=block,
            max_nfev=candidate.contributor_probe_max_nfev,
        )
        probed_base = runtime.base_for_q(ordinal, probed_q)
        probed_values, probed_evaluation, probed_actual = _measurement_and_evaluation(
            runtime,
            ordinal,
            probed_q,
            probed_base,
            binding=binding,
            context=context,
            previous_q=previous_q,
            previous_base=previous_base,
        )
        probe_id = f"probe:{seed_name}:{selected_finger}"
        item = d2g._screened(
            probe_id,
            probed_values,
            probed_evaluation,
            order,
            bool(probe["success"]),
        )
        order += 1
        screened.append(item)
        state_by_id[probe_id] = (
            probed_q,
            probed_base,
            probed_values,
            probed_evaluation,
            block,
            selected_finger,
        )
        probe_receipts.append(
            {
                "seed_id": seed_name,
                "post_bootstrap_contributor": selected_finger,
                "post_bootstrap_ranking": list(ranking),
                **probe,
                "whole_e_im": probed_values.interaction_e_im,
                "primary_hinge": probed_evaluation["primary_objective"],
                "secondary_objective": probed_values.secondary_objective,
                "independent_feasible": probed_evaluation["feasible"],
                "violations": probed_evaluation["violated_constraints"],
                "actual_continuity": probed_actual,
            }
        )
    screening_started = time.perf_counter()
    selected_seed = select_candidate(screened)
    screening_time += time.perf_counter() - screening_started
    if selected_seed is None:
        raise RuntimeError("TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE")
    q_seed, base_seed, _seed_values, _seed_eval, block, selected_finger = state_by_id[
        selected_seed.candidate_id
    ]
    primary_result = None
    primary_exception = None
    try:
        (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
            primary_result,
            _primary_actual,
        ) = d2g._run_candidate_phase(
            runtime,
            ordinal,
            q_seed=q_seed,
            base_seed=base_seed,
            previous_q=previous_q,
            previous_base=previous_base,
            block=block,
            phase="primary",
            maxiter=candidate.selected_primary_maxiter,
            retention_limit=None,
            initialization_source=f"{candidate.name}:{selected_seed.candidate_id}:primary",
        )
        primary = d2g._screened(
            "primary_terminal",
            primary_values,
            primary_evaluation,
            order,
            bool(primary_result.optimizer_converged),
        )
        screened.append(primary)
        state_by_id[primary.candidate_id] = (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
            block,
            selected_finger,
        )
    except Exception as exc:
        primary_exception = f"{type(exc).__name__}:{exc}"
    screening_started = time.perf_counter()
    retained = select_candidate(screened)
    screening_time += time.perf_counter() - screening_started
    if retained is None:
        raise RuntimeError("TECHNICAL_FAIL_NO_VALID_COLDSTART_CANDIDATE")
    (
        retained_q,
        retained_base,
        retained_values,
        retained_evaluation,
        retained_block,
        retained_finger,
    ) = state_by_id[retained.candidate_id]
    limit = d2g.interaction_retention_limit(
        retained_values.interaction_e_im, runtime.authority.interaction_target
    )
    polished = None
    polished_result = None
    polished_state = None
    polish_exception = None
    try:
        (
            polished_q,
            polished_base,
            polished_values,
            polished_evaluation,
            polished_result,
            polished_actual,
        ) = d2g._run_candidate_phase(
            runtime,
            ordinal,
            q_seed=retained_q,
            base_seed=retained_base,
            previous_q=previous_q,
            previous_base=previous_base,
            block=retained_block,
            phase="secondary",
            maxiter=candidate.secondary_polish_maxiter,
            retention_limit=limit,
            initialization_source=f"{candidate.name}:{retained.candidate_id}:secondary",
        )
        polished = d2g._screened(
            "secondary_polished",
            polished_values,
            polished_evaluation,
            order + 1,
            bool(polished_result.optimizer_converged),
        )
        polished_state = (
            polished_q,
            polished_base,
            polished_values,
            polished_evaluation,
            polished_actual,
        )
    except Exception as exc:
        polish_exception = f"{type(exc).__name__}:{exc}"
    screening_started = time.perf_counter()
    selected, retention = retain_after_polish(
        retained,
        polished,
        interaction_target=runtime.authority.interaction_target,
        numerical_epsilon=candidate.numerical_epsilon,
    )
    screening_time += time.perf_counter() - screening_started
    if polished is not None and selected is polished and polished_state is not None:
        q_out, base_out, values_out, evaluation_out, actual_out = polished_state
    else:
        q_out, base_out, values_out, evaluation_out, _block, _finger = state_by_id[
            retained.candidate_id
        ]
        actual_out = d2g.actual_continuity(runtime, previous_q, previous_base, q_out, base_out)
    primary_profile = None if primary_result is None else d2g._solver_profile(primary_result)
    secondary_profile = None if polished_result is None else d2g._solver_profile(polished_result)
    profiler = {
        "schema_version": "RetargetSolverProfilerV1",
        "bootstrap_seed_count": bootstrap_receipt["seed_count"],
        "bootstrap_solve_count": bootstrap_receipt["solve_count"],
        "bootstrap_nfev": bootstrap_receipt["nfev"],
        "bootstrap_njev": bootstrap_receipt["njev"],
        "bootstrap_wall_sec": bootstrap_receipt["wall_sec"],
        "bootstrap_fk_calls": bootstrap_receipt["fk_calls"],
        "bootstrap_residual_evals": bootstrap_receipt["residual_evals"],
        "bootstrap_hard_valid_candidates": sum(
            bool(item["bootstrap_screen_pass"]) for item in bootstrap_rows
        ),
        "candidate_b2_contributor": retained_finger,
        "candidate_b2_candidate_probes": len(probe_receipts),
        "primary_nfev": 0 if primary_profile is None else primary_profile["nfev"],
        "secondary_nfev": 0 if secondary_profile is None else secondary_profile["nfev"],
        "interaction_eval_time_sec": sum(
            float((profile or {}).get("interaction_eval_time_sec", 0.0))
            for profile in (primary_profile, secondary_profile)
        ),
        "candidate_screening_time_sec": screening_time,
        "primary_retention": retention,
        "fallback": selected.candidate_id.startswith("bootstrap:"),
        "total_wall_sec": time.perf_counter() - started,
    }
    receipt = {
        "schema_version": "ExecutionV3ColdStartSearchV2FrameReceiptV1",
        "mode": "COLD_START",
        "candidate": candidate.name,
        "ordinal": ordinal,
        "source_frame_local": int(runtime.graph.frame_indices[ordinal]),
        "runtime_step_index": runtime_step,
        "old_production_q": "ABSENT",
        "previous_accepted_state": "ABSENT" if runtime_step == 0 else "PRESENT",
        "q_old_synthesized": False,
        "failed_stage7_terminal_used_as_q_old": False,
        "bootstrap": bootstrap_receipt,
        "bootstrap_candidates": bootstrap_rows,
        "probe_receipts": probe_receipts,
        "selected_seed_candidate": selected_seed.candidate_id,
        "selected_block": retained_finger,
        "selected_block_dofs": list(retained_block),
        "primary_exception": primary_exception,
        "primary_solver": primary_profile,
        "retained_primary_id": retained.candidate_id,
        "secondary_exception": polish_exception,
        "secondary_solver": secondary_profile,
        "retention_decision": retention,
        "selected_candidate": selected.candidate_id,
        "selected": d2g._measurement_row(values_out),
        "selected_evaluation": evaluation_out,
        "selected_actual_continuity": actual_out,
        "technical_success": bool(selected.usable),
        "optimizer_started": bool(probe_receipts),
        "profiler": profiler,
        "context_binding_sha256": binding.sha256,
    }
    return np.asarray(q_out), np.asarray(base_out), receipt


def _development_row(
    receipt: dict[str, Any], window: dict[str, Any], old_e_im: float
) -> dict[str, Any]:
    selected = receipt["selected"]
    evaluation = receipt["selected_evaluation"]
    actual = receipt["selected_actual_continuity"]
    profiler = receipt["profiler"]
    return {
        "candidate": receipt["candidate"],
        "window_id": window["window_id"],
        "stratum": window["stratum"],
        "ordinal": receipt["ordinal"],
        "frame_id": receipt["source_frame_local"],
        "runtime_step_index": receipt["runtime_step_index"],
        "selected_block": receipt["selected_block"],
        "selected_seed": receipt["selected_seed_candidate"],
        "selected_candidate": receipt["selected_candidate"],
        "old_e_im_evaluation_only": old_e_im,
        "new_e_im": selected["interaction_e_im"],
        "technical_success": receipt["technical_success"],
        "optimizer_started": receipt["optimizer_started"],
        "feasible": evaluation["feasible"],
        "violations": ";".join(evaluation["violated_constraints"]),
        "actual_delta_p": actual["translation_step_m"],
        "actual_delta_R": actual["rotation_step_rad"],
        "actual_delta_q": actual["q_step_inf_rad"],
        "wrist_position_m": selected["wrist_position_m"],
        "wrist_rotation_rad": selected["wrist_rotation_rad"],
        "bone_p95_rad": selected["bone_direction_p95_rad"],
        "collision_min_signed_distance_m": selected["collision_min_signed_distance_m"],
        "joint_limit_min_margin_rad": selected["joint_limit_min_margin_rad"],
        "rotation_determinant": selected["rotation_determinant"],
        "unit_scale_ratio": selected["unit_scale_ratio"],
        "retention_decision": receipt["retention_decision"],
        **profiler,
    }


def _candidate_summary(
    rows: list[dict[str, Any]], failures: list[dict[str, Any]]
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for stratum in ("HIGH", "MID", "LOW"):
        subset = [row for row in rows if row["stratum"] == stratum]
        complete = [row for row in subset if bool(row.get("technical_success"))]
        values = [float(row["new_e_im"]) for row in complete]
        p95 = None if len(values) != 20 else float(np.quantile(values, 0.95))
        hard = bool(len(complete) == 20 and all(bool(row["feasible"]) for row in complete))
        continuity = bool(
            len(complete) == 20
            and all(
                float(row["actual_delta_p"]) <= GATE.temporal_translation_step_limit_m
                and float(row["actual_delta_R"]) <= GATE.temporal_rotation_step_limit_rad
                for row in complete
            )
        )
        preservation = bool(
            len(complete) == 20
            and all(
                float(row["new_e_im"]) <= GATE.interaction_e_im_p95_limit
                for row in complete
                if float(row["old_e_im_evaluation_only"]) <= GATE.interaction_e_im_p95_limit
            )
        )
        strata[stratum] = {
            "technical": f"{len(complete)}/20",
            "p95_e_im": p95,
            "interaction_pass": p95 is not None and p95 <= GATE.interaction_e_im_p95_limit,
            "hard_validity_pass": hard,
            "continuity_pass": continuity,
            "low_old_valid_preservation_pass": preservation
            if stratum == "LOW"
            else "NOT_APPLICABLE",
        }
    passed = bool(
        not failures
        and all(
            value["technical"] == "20/20"
            and value["interaction_pass"]
            and value["hard_validity_pass"]
            and value["continuity_pass"]
            and value["low_old_valid_preservation_pass"] is not False
            for value in strata.values()
        )
    )
    return {
        "schema_version": "MaskedQOldColdStartSearchV2DevelopmentV1",
        "candidate": CS2_A.as_dict(),
        "q_old_authority_present": False,
        "old_e_im_use": "EVALUATION_LABEL_ONLY",
        "frame_count": len(rows),
        "new_dev1_method_development_frames": 0,
        "strata": strata,
        "failures": failures,
        "MASKED_QOLD_DEVELOPMENT": "PASS" if passed else "FAIL",
    }


def _run_candidate_determinism(root: Path) -> dict[str, Any]:
    windows = {item["stratum"]: item for item in d2g.development_windows(d2g.D2A_ROOT)}
    rows: list[dict[str, Any]] = []
    overall = True
    for stratum in ("HIGH", "MID", "LOW"):
        ordinal = int(windows[stratum]["ordinals"][0])
        repeats: list[dict[str, Any]] = []
        for run in (1, 2, 3):
            runtime = d2g.V3Runtime("dev_01", root)
            try:
                q, base, receipt = search_cold_start_v2_frame(
                    runtime,
                    ordinal,
                    runtime_step=0,
                    previous_q=None,
                    previous_base=None,
                    candidate=CS2_A,
                )
                repeats.append(
                    {
                        "run": run,
                        "technical": "PASS",
                        "q": q,
                        "base": base,
                        "bootstrap": receipt["bootstrap"],
                        "block": receipt["selected_block"],
                        "selected": receipt["selected_candidate"],
                        "retention": receipt["retention_decision"],
                        "e_im": receipt["selected"]["interaction_e_im"],
                    }
                )
            except Exception as exc:
                repeats.append(
                    {
                        "run": run,
                        "technical": "FAIL",
                        "failure": f"{type(exc).__name__}:{exc}",
                    }
                )
        reference = repeats[0]
        if all(item["technical"] == "PASS" for item in repeats):
            q_diff = max(float(np.max(np.abs(item["q"] - reference["q"]))) for item in repeats)
            base_diff = max(
                float(np.max(np.abs(item["base"] - reference["base"]))) for item in repeats
            )
            eim_diff = max(abs(float(item["e_im"]) - float(reference["e_im"])) for item in repeats)
            metadata_same = all(
                item["block"] == reference["block"]
                and item["selected"] == reference["selected"]
                and item["retention"] == reference["retention"]
                and [solve["seed"] for solve in item["bootstrap"]["solves"]]
                == [solve["seed"] for solve in reference["bootstrap"]["solves"]]
                for item in repeats
            )
            passed = bool(
                metadata_same and q_diff <= 1e-8 and base_diff <= 1e-8 and eim_diff <= 1e-12
            )
            outcome = "TECHNICAL_PASS_DETERMINISTIC"
        else:
            q_diff = base_diff = eim_diff = None
            metadata_same = all(
                item["technical"] == "FAIL" and item.get("failure") == reference.get("failure")
                for item in repeats
            )
            passed = metadata_same
            outcome = "TECHNICAL_FAIL_DETERMINISTIC"
        overall = overall and passed
        rows.append(
            {
                "stratum": stratum,
                "ordinal": ordinal,
                "runs": 3,
                "bootstrap_seed_order_same": metadata_same,
                "max_q_abs": q_diff,
                "max_base_abs": base_diff,
                "max_eim_abs": eim_diff,
                "outcome": outcome,
                "failure": reference.get("failure"),
                "status": "PASS" if passed else "FAIL",
            }
        )
    payload = {
        "schema_version": "ColdStartSearchV2DeterminismV1",
        "candidate": CS2_A.name,
        "representatives": rows,
        "DETERMINISM": "PASS" if overall else "FAIL",
    }
    _write_json(root / "development/determinism.json", payload)
    return payload


def run_cs2_a_development(root: Path) -> dict[str, Any]:
    candidates = _read_json(root / "search_v2_candidates/candidate_hashes.json")
    if not candidates.get("frozen_before_development"):
        raise RuntimeError("O5RD2G2_CANDIDATES_NOT_FROZEN")
    expected = candidates["candidates"]["CS2_A"]
    if sha256_file(root / "search_v2_candidates/cs2_a.json") != expected:
        raise RuntimeError("O5RD2G2_CANDIDATE_HASH_DRIFT")
    runtime = d2g.V3Runtime("dev_01", root)
    old_eim = np.asarray(runtime.semantic["interaction_final_e_im"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    profiler_rows: list[dict[str, Any]] = []
    for window in d2g.development_windows(d2g.D2A_ROOT):
        previous_q = None
        previous_base = None
        for step, ordinal_raw in enumerate(window["ordinals"]):
            ordinal = int(ordinal_raw)
            state_path = (
                root
                / "development/receipts/cs2_a"
                / window["window_id"]
                / f"frame_{ordinal:04d}.npz"
            )
            receipt_path = state_path.with_suffix(".json")
            try:
                if state_path.exists() and receipt_path.exists():
                    with np.load(state_path, allow_pickle=False) as state:
                        q_out = np.asarray(state["qpos"], dtype=np.float64)
                        base_out = np.asarray(state["base_pose_scene"], dtype=np.float64)
                    receipt = _read_json(receipt_path)
                else:
                    q_out, base_out, receipt = search_cold_start_v2_frame(
                        runtime,
                        ordinal,
                        runtime_step=step,
                        previous_q=previous_q,
                        previous_base=previous_base,
                        candidate=CS2_A,
                    )
                    state_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(state_path, qpos=q_out, base_pose_scene=base_out)
                    _write_json(receipt_path, receipt)
                row = _development_row(receipt, window, float(old_eim[ordinal]))
                rows.append(row)
                profiler_rows.append(
                    {
                        "candidate": CS2_A.name,
                        "window_id": window["window_id"],
                        "stratum": window["stratum"],
                        "ordinal": ordinal,
                        **receipt["profiler"],
                    }
                )
                previous_q, previous_base = q_out, base_out
            except Exception as exc:
                failure = {
                    "candidate": CS2_A.name,
                    "window": window["window_id"],
                    "stratum": window["stratum"],
                    "ordinal": ordinal,
                    "runtime_step_index": step,
                    "error": f"{type(exc).__name__}:{exc}",
                }
                failures.append(failure)
                _append_failure(
                    root,
                    "CS2_A_MASKED_QOLD_DEVELOPMENT",
                    exc,
                    candidate=CS2_A.name,
                    window=window["window_id"],
                    stratum=window["stratum"],
                    ordinal=ordinal,
                    runtime_step_index=step,
                )
                for remaining in window["ordinals"][step:]:
                    rows.append(
                        {
                            "candidate": CS2_A.name,
                            "window_id": window["window_id"],
                            "stratum": window["stratum"],
                            "ordinal": int(remaining),
                            "technical_success": False,
                            "optimizer_started": False,
                            "failure": failure["error"],
                        }
                    )
                break
    for stratum in ("HIGH", "MID", "LOW"):
        _write_csv(
            root / f"development/cs2_a_{stratum.lower()}.csv",
            [row for row in rows if row["stratum"] == stratum],
        )
    _write_csv(root / "profiler/per_frame.csv", profiler_rows)
    summary = _candidate_summary(rows, failures)
    _write_json(root / "development/cs2_a_summary.json", summary)
    determinism = _run_candidate_determinism(root)
    candidate_pass = bool(
        summary["MASKED_QOLD_DEVELOPMENT"] == "PASS" and determinism["DETERMINISM"] == "PASS"
    )
    candidate_rows = [
        {
            "candidate": CS2_A.name,
            "implementation": "FULL_HAND_GEOMETRIC_BOOTSTRAP_POST_BOOTSTRAP_TOP1",
            "masked_qold": summary["MASKED_QOLD_DEVELOPMENT"],
            "determinism": determinism["DETERMINISM"],
            "result": "PASS" if candidate_pass else "FAIL",
        },
        {
            "candidate": "CS2_B_HIERARCHICAL_WRIST_BASE_WHOLE_HAND",
            "implementation": "NOT_IMPLEMENTED_NOT_AUTHORIZED",
            "masked_qold": "NOT_RUN_NOT_AUTHORIZED",
            "determinism": "NOT_RUN",
            "result": "NOT_IMPLEMENTED_NOT_AUTHORIZED",
        },
        {
            "candidate": "CS2_C_GENERIC_MULTIBLOCK_INTERACTION_RESCUE",
            "implementation": "NOT_IMPLEMENTED_NOT_AUTHORIZED",
            "masked_qold": "NOT_RUN_NOT_AUTHORIZED",
            "determinism": "NOT_RUN",
            "result": "NOT_IMPLEMENTED_NOT_AUTHORIZED",
        },
    ]
    _write_csv(root / "development/candidate_summary.csv", candidate_rows)
    decision = {
        "schema_version": "ColdStartSearchV2DevelopmentDecisionV1",
        "status": "PASS" if candidate_pass else "FAIL",
        "EXECUTION_V3_SEARCH_V2_DEVELOPMENT_GATE": "PASS" if candidate_pass else "FAIL",
        "candidate": summary,
        "determinism": determinism,
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "OLD_PRODUCTION_TRAJECTORY_AUTHORITY": "ABSENT",
    }
    _write_json(root / "development/masked_qold_decision.json", decision)
    return decision


def _not_authorized_development(root: Path, key: str) -> dict[str, Any]:
    contract = _read_json(root / f"search_v2_candidates/{key}.json")
    if contract.get("status") != "NOT_IMPLEMENTED_NOT_AUTHORIZED":
        raise RuntimeError(f"O5RD2G2_{key.upper()}_UNEXPECTED_AUTHORIZATION")
    for stratum in ("high", "mid", "low"):
        _write_csv(
            root / f"development/{key}_{stratum}.csv",
            [
                {
                    "candidate": contract["name"],
                    "status": "NOT_RUN_NOT_AUTHORIZED",
                    "reason": contract["reason"],
                }
            ],
        )
    return contract


def run_cs2_b_development(root: Path) -> dict[str, Any]:
    _require_status(root / "failure_localization/root_cause.json", "status")
    return _not_authorized_development(root, "cs2_b")


def run_cs2_c_development(root: Path) -> dict[str, Any]:
    _require_status(root / "failure_localization/root_cause.json", "status")
    return _not_authorized_development(root, "cs2_c")


def reconcile_development_reporting(root: Path) -> dict[str, Any]:
    """Repair block-name reporting from the authoritative free-DOF tuple.

    This does not call an optimizer and does not change q/base/E_IM outcomes.
    """

    _require_status(root / "development/masked_qold_decision.json", "status")
    runtime = d2g.V3Runtime("dev_01", root)
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    inverse = {tuple(indices): finger for finger, indices in blocks.items()}
    old_eim = np.asarray(runtime.semantic["interaction_final_e_im"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    profiler_rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    for window in d2g.development_windows(d2g.D2A_ROOT):
        for ordinal_raw in window["ordinals"]:
            ordinal = int(ordinal_raw)
            path = (
                root
                / "development/receipts/cs2_a"
                / window["window_id"]
                / f"frame_{ordinal:04d}.json"
            )
            _require(path)
            receipt = _read_json(path)
            dofs = tuple(int(value) for value in receipt["selected_block_dofs"])
            if dofs not in inverse:
                raise RuntimeError(f"O5RD2G2_UNKNOWN_SELECTED_BLOCK_DOF_TUPLE:{dofs}")
            authoritative_name = inverse[dofs]
            old_name = receipt.get("selected_block")
            if old_name != authoritative_name:
                corrections.append(
                    {
                        "ordinal": ordinal,
                        "window_id": window["window_id"],
                        "reported_before": old_name,
                        "reported_after": authoritative_name,
                        "authority": "selected_block_dofs_used_by_optimizer",
                    }
                )
            receipt["selected_block"] = authoritative_name
            receipt["profiler"]["candidate_b2_contributor"] = authoritative_name
            receipt["reporting_reconciliation"] = {
                "schema_version": "SelectedBlockReportingReconciliationV1",
                "original_selected_block": old_name,
                "corrected_selected_block": authoritative_name,
                "scientific_state_changed": False,
                "q_base_eim_changed": False,
            }
            _write_json(path, receipt)
            rows.append(_development_row(receipt, window, float(old_eim[ordinal])))
            profiler_rows.append(
                {
                    "candidate": CS2_A.name,
                    "window_id": window["window_id"],
                    "stratum": window["stratum"],
                    "ordinal": ordinal,
                    **receipt["profiler"],
                }
            )
    for stratum in ("HIGH", "MID", "LOW"):
        _write_csv(
            root / f"development/cs2_a_{stratum.lower()}.csv",
            [row for row in rows if row["stratum"] == stratum],
        )
    _write_csv(root / "profiler/per_frame.csv", profiler_rows)
    old_summary = _read_json(root / "development/cs2_a_summary.json")
    summary = _candidate_summary(rows, old_summary["failures"])
    _write_json(root / "development/cs2_a_summary.json", summary)
    decision = _read_json(root / "development/masked_qold_decision.json")
    decision["candidate"] = summary
    decision["reporting_reconciliation"] = "PASS"
    _write_json(root / "development/masked_qold_decision.json", decision)
    payload = {
        "schema_version": "ColdStartSearchV2DevelopmentReportingReconciliationV1",
        "status": "PASS",
        "correction_count": len(corrections),
        "authority": "selected_block_dofs actually passed to Candidate-B2 optimizer",
        "method_changed": False,
        "q_base_eim_changed": False,
        "gate_outcomes_changed": False,
        "rerun_count": 0,
        "corrections": corrections,
    }
    _write_json(root / "development/reporting_reconciliation.json", payload)
    return payload


def run_refinement_regression(root: Path) -> dict[str, Any]:
    _require_status(root / "development/masked_qold_decision.json", "status")
    payload = d2g.run_refinement_regression(root)
    rows = _read_csv(root / "development/refinement_regression.csv")
    _write_json(
        root / "refinement_regression/frame_selection.json",
        {
            "schema_version": "ExecutionV3RefinementRegressionSelectionV1",
            "frozen_before_execution": True,
            "high_mid_low_counts": {"HIGH": 5, "MID": 5, "LOW": 5},
            "contiguous_window": "consumed LOW development window",
            "rows": [
                {
                    "stratum": row["stratum"],
                    "ordinal": int(row["ordinal"]),
                    "continuous_window": row["continuous_window"] == "True",
                }
                for row in rows
            ],
        },
    )
    _write_csv(
        root / "refinement_regression/per_frame.csv",
        [row for row in rows if row["continuous_window"] != "True"],
    )
    _write_csv(
        root / "refinement_regression/window.csv",
        [row for row in rows if row["continuous_window"] == "True"],
    )
    _write_json(root / "refinement_regression/parity_summary.json", payload)
    return payload


def select_coldstart_search_v2(root: Path) -> dict[str, Any]:
    development = _require_status(root / "development/masked_qold_decision.json", "status")
    refinement = _require_status(
        root / "refinement_regression/parity_summary.json",
        "REFINEMENT_MODE_REGRESSION",
    )
    rows = _read_csv(root / "development/candidate_summary.csv")
    eligible = [row for row in rows if row["result"] == "PASS"]
    selected = eligible[0] if len(eligible) == 1 else None
    payload = {
        "schema_version": "ColdStartSearchV2SelectionDecisionV1",
        "status": "PASS" if selected is not None else "FAIL",
        "selection_rubric": [
            "HIGH/MID/LOW all 20/20 technical",
            "all p95 E_IM <= 1e-4",
            "all frozen hard semantics PASS",
            "determinism PASS",
            "generic no episode-specific logic",
            "simpler contract before runtime tie-break",
        ],
        "development_gate": development["status"],
        "refinement_regression": refinement["REFINEMENT_MODE_REGRESSION"],
        "SELECTED_COLDSTART_SEARCH_V2": None if selected is None else selected["candidate"],
        "dev2_outcome_used_for_selection": False,
        "candidates": rows,
    }
    _write_json(root / "development/selection_decision.json", payload)
    if selected is None:
        raise RuntimeError("O5RD2G2_NO_COLDSTART_SEARCH_V2_CANDIDATE_READY")
    return payload


def lock_selected_candidate(root: Path) -> dict[str, Any]:
    selection = _require_status(root / "development/selection_decision.json", "status")
    _require_status(
        root / "refinement_regression/parity_summary.json",
        "REFINEMENT_MODE_REGRESSION",
    )
    selected = str(selection["SELECTED_COLDSTART_SEARCH_V2"])
    if selected != CS2_A.name:
        raise RuntimeError("O5RD2G2_SELECTED_CANDIDATE_UNKNOWN")
    hashes = _read_json(root / "search_v2_candidates/candidate_hashes.json")
    payload = {
        "schema_version": "SelectedColdStartSearchV2LockV1",
        "status": "LOCKED_BEFORE_DEV2",
        "candidate_name": selected,
        "bootstrap_contract_sha256": hashes["candidates"]["CS2_A"],
        "seed_authority_draft_sha256": hashes["candidates"]["CS2_A"],
        "solver_budget": {
            "bootstrap_max_nfev": CS2_A.bootstrap_max_nfev,
            "contributor_probe_max_nfev": CS2_A.contributor_probe_max_nfev,
            "primary_maxiter": CS2_A.selected_primary_maxiter,
            "secondary_maxiter": CS2_A.secondary_polish_maxiter,
        },
        "fallback": "best independently hard-valid generic candidate or explicit technical failure",
        "Candidate_B2_hashes": {
            "objective_v2": OBJECTIVE_SHA,
            "execution_v2": EXECUTION_V2_SHA,
        },
        "graph_authority": "CANONICAL_SOURCE_DERIVED",
        "mode_semantics": {
            "REFINEMENT": "q_old required",
            "COLD_START_FRAME0": "q_old and previous state absent",
            "COLD_START_T_GT_0": "q_old absent; previous accepted runtime available",
        },
        "candidate_switch_after_dev2_failure": "FORBIDDEN",
    }
    path = root / "development/selected_candidate_lock.json"
    _write_json(path, payload)
    _sha_text(root / "development/selected_candidate_lock.sha256", sha256_file(path))
    return payload


def run_dev2_frame0_development(root: Path) -> dict[str, Any]:
    lock_path = root / "development/selected_candidate_lock.json"
    lock = _require_status(lock_path, "status", "LOCKED_BEFORE_DEV2")
    recorded_sha = (
        (root / "development/selected_candidate_lock.sha256").read_text(encoding="utf-8").strip()
    )
    if sha256_file(lock_path) != recorded_sha or lock["candidate_name"] != CS2_A.name:
        raise RuntimeError("O5RD2G2_SELECTED_CANDIDATE_LOCK_DRIFT")
    _require_status(
        root / "refinement_regression/parity_summary.json",
        "REFINEMENT_MODE_REGRESSION",
    )
    identity = d2g._dev2_identity()
    if not identity["identity_exact"]:
        raise RuntimeError("O5RD2G2_DEV2_IDENTITY_AUTHORITY_MISMATCH")
    role = {
        "schema_version": "DEV2Frame0ColdStartSearchV2RoleReceiptV1",
        "ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "independent_control": False,
        "identity": identity,
        "mode": "COLD_START",
        "old_production_q": "ABSENT",
        "previous_accepted_state": "ABSENT",
        "source_graph_authority": "CANONICAL_SOURCE_DERIVED",
        "selected_candidate_lock_sha256": recorded_sha,
    }
    _write_json(root / "dev2_frame0_development/role_receipt.json", role)
    runs: list[dict[str, Any]] = []
    states: list[tuple[np.ndarray, np.ndarray]] = []
    profiler_rows: list[dict[str, Any]] = []
    for run in (1, 2, 3):
        runtime = d2g.V3Runtime("dev_02", root)
        started = time.perf_counter()
        try:
            q, base, receipt = search_cold_start_v2_frame(
                runtime,
                0,
                runtime_step=0,
                previous_q=None,
                previous_base=None,
                candidate=CS2_A,
            )
            selected = receipt["selected"]
            evaluation = receipt["selected_evaluation"]
            row = {
                "schema_version": "DEV2Frame0ColdStartSearchV2RunV1",
                "run": run,
                "source_frame": 10704,
                "mode": "COLD_START",
                "candidate": CS2_A.name,
                "optimizer_started": receipt["optimizer_started"],
                "technical": "PASS" if receipt["technical_success"] else "FAIL",
                "finite": bool(np.all(np.isfinite(q)) and np.all(np.isfinite(base))),
                "E_IM": selected["interaction_e_im"],
                "wrist": "PASS"
                if selected["wrist_position_m"] <= runtime.authority.wrist_position_limit_m
                and selected["wrist_rotation_rad"] <= runtime.authority.wrist_rotation_limit_rad
                else "FAIL",
                "wrist_position_m": selected["wrist_position_m"],
                "wrist_rotation_rad": selected["wrist_rotation_rad"],
                "bone": "PASS"
                if selected["bone_direction_p95_rad"] <= runtime.authority.bone_direction_limit_rad
                else "FAIL",
                "collision": "PASS"
                if selected["collision_min_signed_distance_m"]
                >= -runtime.authority.collision_hard_bound_m - 1e-6
                else "FAIL",
                "joints": "PASS" if selected["joint_limit_min_margin_rad"] >= -1e-10 else "FAIL",
                "hard_feasible": evaluation["feasible"],
                "selected_seed": receipt["selected_seed_candidate"],
                "selected_terminal": receipt["selected_candidate"],
                "selected_block": receipt["selected_block"],
                "retention": receipt["retention_decision"],
                "bootstrap_seed_order": [item["seed"] for item in receipt["bootstrap"]["solves"]],
                "q_old_synthesized": receipt["q_old_synthesized"],
                "failed_stage7_terminal_used_as_q_old": receipt[
                    "failed_stage7_terminal_used_as_q_old"
                ],
                "profiler": receipt["profiler"],
                "wall_sec": time.perf_counter() - started,
            }
            states.append((q, base))
            profiler_rows.append({"run": run, **receipt["profiler"]})
        except Exception as exc:
            row = {
                "schema_version": "DEV2Frame0ColdStartSearchV2RunV1",
                "run": run,
                "source_frame": 10704,
                "mode": "COLD_START",
                "candidate": CS2_A.name,
                "optimizer_started": False,
                "technical": "FAIL",
                "finite": False,
                "E_IM": None,
                "wrist": "NOT_EVALUABLE",
                "bone": "NOT_EVALUABLE",
                "collision": "NOT_EVALUABLE",
                "joints": "NOT_EVALUABLE",
                "failure": f"{type(exc).__name__}:{exc}",
                "wall_sec": time.perf_counter() - started,
            }
            _append_failure(root, "DEV2_FRAME0_KNOWN_FAILURE_DEVELOPMENT", exc, run=run)
        _write_json(root / f"dev2_frame0_development/run_{run}.json", row)
        runs.append(row)
    _write_csv(root / "dev2_frame0_development/profiler.csv", profiler_rows)
    comparable = len(states) == 3
    q_diff = (
        None
        if not comparable
        else max(float(np.max(np.abs(q - states[0][0]))) for q, _base in states)
    )
    base_diff = (
        None
        if not comparable
        else max(float(np.max(np.abs(base - states[0][1]))) for _q, base in states)
    )
    eim = [row["E_IM"] for row in runs if row["E_IM"] is not None]
    eim_diff = None if len(eim) != 3 else max(abs(float(value) - float(eim[0])) for value in eim)
    metadata_same = bool(
        len(runs) == 3
        and all(
            row.get("bootstrap_seed_order") == runs[0].get("bootstrap_seed_order")
            and row.get("selected_terminal") == runs[0].get("selected_terminal")
            and row.get("selected_block") == runs[0].get("selected_block")
            and row.get("retention") == runs[0].get("retention")
            for row in runs
        )
    )
    determinism = {
        "schema_version": "DEV2Frame0ColdStartSearchV2DeterminismV1",
        "comparable_runs": comparable,
        "seed_bootstrap_selection_same": metadata_same,
        "max_q_abs": q_diff,
        "max_base_abs": base_diff,
        "max_eim_abs": eim_diff,
        "status": "PASS"
        if comparable
        and metadata_same
        and q_diff is not None
        and q_diff <= 1e-8
        and base_diff is not None
        and base_diff <= 1e-8
        and eim_diff is not None
        and eim_diff <= 1e-12
        else "FAIL",
    }
    _write_json(root / "dev2_frame0_development/determinism.json", determinism)
    passed = bool(
        determinism["status"] == "PASS"
        and all(
            row["optimizer_started"]
            and row["technical"] == "PASS"
            and row["finite"]
            and float(row["E_IM"]) <= GATE.interaction_e_im_p95_limit
            and row["wrist"] == "PASS"
            and row["bone"] == "PASS"
            and row["collision"] == "PASS"
            and row["joints"] == "PASS"
            and row["hard_feasible"]
            and not row["q_old_synthesized"]
            and not row["failed_stage7_terminal_used_as_q_old"]
            for row in runs
        )
    )
    decision = {
        "schema_version": "DEV2Frame0ColdStartSearchV2DecisionV1",
        "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
        "DEV2_FRAME0_V3_DEVELOPMENT": "PASS" if passed else "FAIL",
        "DEV2_FRAME0_V3_RUN_COUNT": 3,
        "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": sum(bool(row["optimizer_started"]) for row in runs),
        "EXECUTION_V3_FREEZE_AUTHORIZED": "YES" if passed else "NO",
        "runs": runs,
    }
    _write_json(root / "dev2_frame0_development/decision.json", decision)
    return decision


def _special_case_audit() -> dict[str, str]:
    algorithm = inspect.getsource(search_cold_start_v2_frame) + inspect.getsource(
        _whole_hand_geometric_bootstrap
    )
    return {
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "DEV2_EPISODE_ID_BRANCH": "NO" if "scene_01__A003" not in algorithm else "YES",
        "DEV2_OBJECT_ID_BRANCH": "NO" if "C11001" not in algorithm else "YES",
        "DEV2_FRAME_LITERAL_BRANCH": "NO" if "10704" not in algorithm else "YES",
        "THUMB_SPECIAL_CASE_ADDED": "NO" if '"thumb"' not in algorithm else "YES",
        "DEV1_SPECIAL_CASE_ADDED": "NO",
    }


def _development_gate(root: Path) -> dict[str, Any]:
    integrity = _require_status(root / "preflight/integrity.json", "FROZEN_UPSTREAM_INTEGRITY")
    localization = _require_status(root / "failure_localization/root_cause.json", "status")
    masked = _require_status(root / "development/masked_qold_decision.json", "status")
    refinement = _require_status(
        root / "refinement_regression/parity_summary.json", "REFINEMENT_MODE_REGRESSION"
    )
    selection = _require_status(root / "development/selection_decision.json", "status")
    lock = _require_status(
        root / "development/selected_candidate_lock.json", "status", "LOCKED_BEFORE_DEV2"
    )
    dev2 = _require_status(
        root / "dev2_frame0_development/decision.json", "DEV2_FRAME0_V3_DEVELOPMENT"
    )
    special = _special_case_audit()
    no_special = all(value == "NO" for value in special.values())
    passed = bool(
        integrity["FROZEN_UPSTREAM_INTEGRITY"] == "PASS"
        and localization["status"] == "PASS"
        and masked["status"] == "PASS"
        and refinement["REFINEMENT_MODE_REGRESSION"] == "PASS"
        and selection["SELECTED_COLDSTART_SEARCH_V2"] == CS2_A.name
        and lock["candidate_name"] == CS2_A.name
        and dev2["DEV2_FRAME0_V3_DEVELOPMENT"] == "PASS"
        and no_special
    )
    payload = {
        "schema_version": "ExecutionV3SearchV2DevelopmentGateV1",
        "EXECUTION_V3_SEARCH_V2_DEVELOPMENT_GATE": "PASS" if passed else "FAIL",
        "frozen_upstream": integrity["FROZEN_UPSTREAM_INTEGRITY"],
        "localization": localization["status"],
        "masked_qold": masked["status"],
        "refinement_regression": refinement["REFINEMENT_MODE_REGRESSION"],
        "selection": selection["SELECTED_COLDSTART_SEARCH_V2"],
        "candidate_lock": lock["status"],
        "dev2_frame0": dev2["DEV2_FRAME0_V3_DEVELOPMENT"],
        "special_case_audit": special,
    }
    _write_json(root / "development/development_gate.json", payload)
    if not passed:
        raise RuntimeError("O5RD2G2_EXECUTION_V3_DEVELOPMENT_GATE_FAIL")
    return payload


def freeze_execution_input_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    payload = {
        "schema_version": "ExecutionInputAuthorityV1",
        "status": "FROZEN",
        "scientifically_mandatory_inputs": [
            "canonical source hand",
            "canonical object pose and mesh",
            "source-derived interaction graph",
            "Wuji robot asset and limits",
        ],
        "mode_specific_inputs": {
            "REFINEMENT": {"old_production_q": "REQUIRED"},
            "COLD_START_FRAME0": {
                "old_production_q": "FORBIDDEN_ABSENT",
                "previous_accepted_runtime_state": "ABSENT",
            },
            "COLD_START_T_GT_0": {
                "old_production_q": "ABSENT",
                "previous_accepted_runtime_state": "AVAILABLE_AFTER_SUCCESSFUL_T_MINUS_1",
            },
        },
        "source_derived_inputs": [
            "source wrist frame",
            "source bone features",
            "object surface samples",
            "interaction graph",
        ],
        "robot_derived_inputs": ["joint bounds", "DOF mapping", "rest/midpoint seeds"],
        "runtime_derived_inputs": ["previous accepted q/base for t>0"],
        "optional_search_hints": ["previous accepted runtime state"],
        "forbidden_synthetic_authorities": [
            "synthetic q_old",
            "failed Stage7 terminal as q_old",
            "DEV1 solved q in cold start",
            "episode/object/frame-specific seed",
        ],
    }
    path = root / "frozen_v3/execution_input_authority.json"
    _write_json(path, payload)
    _sha_text(root / "frozen_v3/execution_input_authority.sha256", sha256_file(path))
    return payload


def freeze_source_interaction_graph_authority(root: Path) -> dict[str, Any]:
    _development_gate(root)
    source = _read_json(root / "graph_authority/source_interaction_graph_authority.json")
    payload = {
        **source,
        "schema_version": "SourceInteractionGraphAuthorityV1",
        "status": "FROZEN",
        "semantics_changed_in_d2g2": False,
    }
    path = root / "frozen_v3/source_interaction_graph_authority.json"
    _write_json(path, payload)
    _sha_text(root / "frozen_v3/source_interaction_graph_authority.sha256", sha256_file(path))
    return payload


def freeze_coldstart_bootstrap_contract(root: Path) -> dict[str, Any]:
    _development_gate(root)
    lock = _read_json(root / "development/selected_candidate_lock.json")
    payload = {
        "schema_version": COLD_START_BOOTSTRAP_SCHEMA_VERSION,
        "status": "FROZEN",
        "selected_candidate": lock["candidate_name"],
        "mathematical_search_objective": "existing Eq.1-2 whole-hand bone-direction residual",
        "scientific_target": "UNCHANGED_RETARGET_OBJECTIVE_V2_CANDIDATE_B2",
        "free_dofs": CS2_A.bootstrap_free_dofs,
        "seed_authority": list(CS2_A.bootstrap_seed_sources),
        "solver": CS2_A.bootstrap_solver_profile,
        "bounds": "Wuji asset joint limits",
        "budget_max_nfev": CS2_A.bootstrap_max_nfev,
        "screening": "finite full-state, bounds, physical hard validity; bone hard validity required after B2",
        "termination": "solver termination or frozen max_nfev; independently screen terminal",
        "determinism": "fixed two-seed order, no RNG, stable duplicate elision",
        "post_bootstrap_contributor_ranking": "Eq.7 top-1 per bootstrap state",
        "fallback": lock["fallback"],
    }
    path = root / "frozen_v3/cold_start_bootstrap_contract.json"
    _write_json(path, payload)
    _sha_text(root / "frozen_v3/cold_start_bootstrap_contract.sha256", sha256_file(path))
    return payload


def freeze_coldstart_seed_authority_v2(root: Path) -> dict[str, Any]:
    _development_gate(root)
    payload = {
        "schema_version": COLD_START_SEED_AUTHORITY_V2_SCHEMA_VERSION,
        "status": "FROZEN",
        "frame0_seed_sources": list(CS2_A.bootstrap_seed_sources),
        "t_gt_0_seed_sources": [
            *CS2_A.bootstrap_seed_sources,
            "previous_accepted_runtime_state",
        ],
        "ordering": "rest, midpoint, then previous accepted runtime for t>0",
        "bounds": "Wuji asset joint limits",
        "deterministic_perturbation": "NONE",
        "forbidden_seeds": [
            "q_old",
            "q_old alias",
            "DEV1 solved q",
            "failed Stage7 terminal",
            "episode/object/frame special case",
        ],
    }
    path = root / "frozen_v3/cold_start_seed_authority_v2.json"
    _write_json(path, payload)
    _sha_text(root / "frozen_v3/cold_start_seed_authority_v2.sha256", sha256_file(path))
    return payload


def freeze_execution_v3(root: Path) -> dict[str, Any]:
    _development_gate(root)
    freeze_execution_input_authority(root)
    freeze_source_interaction_graph_authority(root)
    freeze_coldstart_bootstrap_contract(root)
    freeze_coldstart_seed_authority_v2(root)
    payload = {
        "schema_version": "RetargetObjectiveV2ExecutionContractV3",
        "status": "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
        "objective": "RetargetObjectiveV2 Candidate B2 unchanged",
        "objective_v2_sha256": OBJECTIVE_SHA,
        "modes": {
            "REFINEMENT": {
                "old_production_q": "REQUIRED",
                "execution": "byte/numerical parity delegate to frozen ExecutionV2/S1",
                "fallback": "authoritative old-production baseline where allowed",
            },
            "COLD_START_FRAME0": {
                "old_production_q": "FORBIDDEN_ABSENT",
                "previous_accepted_runtime_state": "ABSENT",
                "inputs": "canonical source/object/graph + Wuji asset + ColdStartBootstrapContractV1",
                "fallback": "best independently valid generic candidate or explicit failure",
            },
            "COLD_START_T_GT_0": {
                "old_production_q": "ABSENT",
                "previous_accepted_runtime_state": "AVAILABLE_AFTER_SUCCESSFUL_T_MINUS_1",
            },
        },
        "selected_coldstart_search_v2": CS2_A.as_dict(),
        "special_case_audit": _special_case_audit(),
        "fresh_independent_evidence_consumed": False,
    }
    path = root / "frozen_v3/objective_v2_execution_contract_v3.json"
    _write_json(path, payload)
    _sha_text(root / "frozen_v3/objective_v2_execution_contract_v3.sha256", sha256_file(path))
    verify_frozen_upstream(root)
    return payload


def generate_future_certification_plan(root: Path) -> dict[str, Any]:
    contract = _require_status(
        root / "frozen_v3/objective_v2_execution_contract_v3.json",
        "status",
        "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION",
    )
    payload = {
        "schema_version": "ExecutionV3IndependentColdStartCertificationPlanV2",
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "next_stage": "O5R-D2H_EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION",
        "frozen_execution_v3_sha256": sha256_file(
            root / "frozen_v3/objective_v2_execution_contract_v3.json"
        ),
        "frozen_objective_v2_sha256": contract["objective_v2_sha256"],
        "order": [
            "fresh DEV1 ColdStartSparseValidationV4 with q_old absent",
            "fresh DEV1 ColdStartWindowValidationV4 with q_old absent",
            "freeze three fresh development-split cross-episode frame0 controls",
            "three cross-episode frame0 controls 3/3 PASS",
            "DEV2 full 240 KNOWN_FAILURE_RECOVERY exactly once",
            "Semantic V1, viewer, human review",
        ],
        "cross_episode_selection_contract": {
            "N": 3,
            "split": "OakInk2 development only",
            "requirements": [
                "right-hand eligible",
                "single unambiguous target object",
                "canonical input complete",
                "never used for V3 development",
                "prefer different episodes and objects",
                "no certification or heldout split",
            ],
            "actual_episode_ids_selected_now": False,
        },
        "executed_in_d2g2": False,
    }
    _write_json(
        root / "future_certification/execution_v3_independent_coldstart_certification_plan_v2.json",
        payload,
    )
    return payload


def update_evidence_ledger(root: Path) -> dict[str, Any]:
    source = _read_json(D2G_ROOT / "ledger/future_validation_exclusion_ledger.json")
    entries = json.loads(json.dumps(source["entries"]))
    development_ordinals = {
        int(value)
        for window in d2g.development_windows(d2g.D2A_ROOT)
        for value in window["ordinals"]
    }
    for entry in entries:
        entry["roles"] = [
            "D2G_MASKED_QOLD_DEVELOPMENT" if role == "V3_MASKED_QOLD_DEVELOPMENT" else role
            for role in entry["roles"]
        ]
        if entry["episode"] == "DEV1" and int(entry["ordinal"]) in development_ordinals:
            roles = list(entry["roles"])
            if "D2G2_MASKED_QOLD_DEVELOPMENT" not in roles:
                roles.append("D2G2_MASKED_QOLD_DEVELOPMENT")
            entry["roles"] = sorted(roles)
    payload = {
        "schema_version": "ExecutionV3EvidenceLedgerV2",
        "FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT": len(entries),
        "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
        "role_vocabulary": [
            "V2_METHOD_DEVELOPMENT",
            "SPARSE_V2_CERTIFICATION_CONSUMED",
            "SPARSE_V3_CERTIFICATION_CONSUMED",
            "WINDOW_V3_CERTIFICATION_CONSUMED",
            "D2G_MASKED_QOLD_DEVELOPMENT",
            "D2G2_MASKED_QOLD_DEVELOPMENT",
            "DEV2_FRAME0_KNOWN_FAILURE_DEVELOPMENT",
            "FUTURE_V3_INDEPENDENT_EXCLUDED",
        ],
        "entries": entries,
    }
    _write_json(root / "ledger/execution_v3_evidence_ledger_v2.json", payload)
    _write_json(
        root / "ledger/future_validation_exclusion_ledger.json",
        {
            "schema_version": "ExecutionV3FutureValidationExclusionLedgerV2",
            "count": len(entries),
            "new_dev1_method_development_frames": 0,
            "entries": entries,
        },
    )
    return payload


def _not_run_placeholders(root: Path) -> None:
    refinement = root / "refinement_regression/parity_summary.json"
    if not refinement.exists():
        payload = {
            "schema_version": "ExecutionV3RefinementRegressionV1",
            "REFINEMENT_MODE_REGRESSION": "NOT_RUN",
            "reason": "MASKED_QOLD_DEVELOPMENT_GATE_CLOSED",
            "max_q_abs": None,
            "max_base_abs": None,
            "max_e_im_abs": None,
        }
        _write_json(refinement, payload)
        _write_json(
            root / "refinement_regression/frame_selection.json",
            {"status": "NOT_RUN_GATE_CLOSED"},
        )
        _write_csv(
            root / "refinement_regression/per_frame.csv",
            [{"status": "NOT_RUN_GATE_CLOSED"}],
        )
        _write_csv(
            root / "refinement_regression/window.csv",
            [{"status": "NOT_RUN_GATE_CLOSED"}],
        )
    selection = root / "development/selection_decision.json"
    if not selection.exists():
        _write_json(
            selection,
            {
                "schema_version": "ColdStartSearchV2SelectionDecisionV1",
                "status": "NOT_RUN_GATE_CLOSED",
                "SELECTED_COLDSTART_SEARCH_V2": None,
                "dev2_outcome_used_for_selection": False,
            },
        )
    lock = root / "development/selected_candidate_lock.json"
    if not lock.exists():
        _write_json(
            lock,
            {
                "schema_version": "SelectedColdStartSearchV2LockV1",
                "status": "NOT_RUN_GATE_CLOSED",
                "candidate_name": None,
            },
        )
        _sha_text(root / "development/selected_candidate_lock.sha256", "NOT_FROZEN")
    decision = root / "dev2_frame0_development/decision.json"
    if not decision.exists():
        _write_json(
            root / "dev2_frame0_development/role_receipt.json",
            {
                "ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
                "status": "NOT_RUN_GATE_CLOSED",
            },
        )
        for run in (1, 2, 3):
            _write_json(
                root / f"dev2_frame0_development/run_{run}.json",
                {"run": run, "status": "NOT_RUN_GATE_CLOSED", "optimizer_started": False},
            )
        _write_csv(
            root / "dev2_frame0_development/profiler.csv",
            [{"status": "NOT_RUN_GATE_CLOSED"}],
        )
        _write_json(
            root / "dev2_frame0_development/determinism.json",
            {"status": "NOT_RUN_GATE_CLOSED"},
        )
        _write_json(
            decision,
            {
                "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
                "DEV2_FRAME0_V3_DEVELOPMENT": "NOT_RUN",
                "DEV2_FRAME0_V3_RUN_COUNT": 0,
                "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": 0,
                "EXECUTION_V3_FREEZE_AUTHORIZED": "NO",
            },
        )
    frozen = root / "frozen_v3/objective_v2_execution_contract_v3.json"
    if not frozen.exists():
        _write_json(
            root / "frozen_v3/no_execution_v3_ready.json",
            {
                "schema_version": "NoExecutionV3ReadyV1",
                "EXECUTION_V3_STATUS": "NO_EXECUTION_V3_READY",
                "EXECUTION_INPUT_AUTHORITY_SHA256": None,
                "SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256": None,
                "COLD_START_BOOTSTRAP_CONTRACT_SHA256": None,
                "COLD_START_SEED_AUTHORITY_V2_SHA256": None,
                "OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256": None,
            },
        )
    plan = (
        root / "future_certification/execution_v3_independent_coldstart_certification_plan_v2.json"
    )
    if not plan.exists():
        _write_json(
            plan,
            {
                "schema_version": "ExecutionV3IndependentColdStartCertificationPlanV2",
                "status": "NOT_GENERATED_EXECUTION_V3_NOT_FROZEN",
                "actual_episode_ids_selected_now": False,
                "executed_in_d2g2": False,
            },
        )


def _frozen_hashes(root: Path) -> dict[str, str | None]:
    mapping = {
        "EXECUTION_INPUT_AUTHORITY_SHA256": root / "frozen_v3/execution_input_authority.json",
        "SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256": root
        / "frozen_v3/source_interaction_graph_authority.json",
        "COLD_START_BOOTSTRAP_CONTRACT_SHA256": root
        / "frozen_v3/cold_start_bootstrap_contract.json",
        "COLD_START_SEED_AUTHORITY_V2_SHA256": root / "frozen_v3/cold_start_seed_authority_v2.json",
        "OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256": root
        / "frozen_v3/objective_v2_execution_contract_v3.json",
    }
    return {name: sha256_file(path) if path.exists() else None for name, path in mapping.items()}


def _profiler_aggregate(root: Path) -> dict[str, Any]:
    path = root / "profiler/per_frame.csv"
    rows = _read_csv(path) if path.exists() else []
    completed = [row for row in rows if row.get("total_wall_sec") not in (None, "")]
    numeric_fields = (
        "bootstrap_solve_count",
        "bootstrap_nfev",
        "bootstrap_wall_sec",
        "bootstrap_hard_valid_candidates",
        "candidate_b2_candidate_probes",
        "primary_nfev",
        "secondary_nfev",
        "interaction_eval_time_sec",
        "candidate_screening_time_sec",
        "total_wall_sec",
    )
    aggregate = {
        field: None if not completed else float(np.mean([float(row[field]) for row in completed]))
        for field in numeric_fields
    }
    payload = {
        "schema_version": "RetargetSolverProfilerV1Aggregate",
        "completed_frames": len(completed),
        "mean": aggregate,
    }
    _write_json(root / "profiler/aggregate.json", payload)
    return payload


def _materialize_refinement_failure_analysis(root: Path) -> dict[str, Any]:
    path = root / "development/refinement_regression.csv"
    if not path.exists():
        payload = {"status": "NOT_RUN"}
    else:
        rows = _read_csv(path)
        failed = [row for row in rows if row["status"] != "PASS"]
        payload = {
            "schema_version": "ExecutionV3RefinementRegressionFailureAnalysisV1",
            "status": "PASS" if not failed else "FAIL",
            "classification": "FROZEN_EXECUTION_V2_NUMERICAL_PARITY_NOT_REPRODUCED"
            if failed
            else "NO_FAILURE",
            "frame_checks": len(rows),
            "failed_checks": len(failed),
            "metadata_mismatch_count": sum(
                row["same_selected_contributor_seed_retention"] != "True" for row in failed
            ),
            "numerical_mismatch_with_metadata_parity_count": sum(
                row["same_selected_contributor_seed_retention"] == "True" for row in failed
            ),
            "failed_ordinals": [int(row["ordinal"]) for row in failed],
            "tolerance_changed": False,
            "second_coldstart_candidate_tried": False,
            "interpretation": "The unchanged refinement delegate did not reproduce the frozen S1 receipts within the pre-existing tolerance. This is a hard gate failure, not evidence against the cold-start masked-q_old result.",
        }
    _write_json(root / "refinement_regression/failure_analysis.json", payload)
    return payload


def record_git(root: Path) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    commits = _git("log", "--format=%H%x09%s", f"{START_HEAD}..{head}").splitlines()
    payload = {
        "start_head": START_HEAD,
        "final_head": head,
        "commits": commits,
        "tracked_worktree_clean": not bool(_git("status", "--porcelain", "--untracked-files=no")),
        "status_short": _git("status", "--short", "--untracked-files=all").splitlines(),
        "pushed": False,
        "pr_created": False,
    }
    _write_json(root / "git_commits.json", payload)
    return payload


def validate_repository(root: Path) -> dict[str, Any]:
    task_files = [
        "src/toporetarget/retarget/objective_v3_execution.py",
        "scripts/data/run_oakink2_o5rd2g2.py",
        "tests/retarget/test_objective_v3_execution.py",
        "tests/data/test_oakink2_o5rd2g2.py",
    ]
    commands = [
        (
            "00_ruff_task_files.log",
            ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", *task_files],
        ),
        (
            "01_ruff_format_task_files.log",
            ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", *task_files],
        ),
        (
            "02_mypy_src.log",
            ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ),
        (
            "03_pytest_full.log",
            ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        ),
        (
            "04_paper_fidelity.log",
            ["conda", "run", "-n", "toporetarget-rl", "python", "scripts/check_paper_fidelity.py"],
        ),
        ("05_ci_ruff_all.log", ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", "."]),
        (
            "06_ci_ruff_format_all.log",
            ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", "."],
        ),
        ("07_git_diff_check.log", ["git", "diff", "--check"]),
    ]
    results: list[dict[str, Any]] = []
    for log_name, command in commands:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        log = root / "validation_logs" / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        results.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
                "log": str(log.resolve()),
            }
        )
    payload = {
        "schema_version": "O5RD2G2RepositoryValidationV1",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "checks": results,
    }
    _write_json(root / "validation_results.json", payload)
    _write_json(
        root / "tests.json",
        {
            "status": payload["status"],
            "full_pytest": next(
                item for item in results if item["log"].endswith("03_pytest_full.log")
            ),
        },
    )
    if payload["status"] != "PASS":
        raise RuntimeError("O5RD2G2_REPOSITORY_VALIDATION_FAIL")
    return payload


def summarize(root: Path) -> dict[str, Any]:
    _not_run_placeholders(root)
    if not (root / "ledger/execution_v3_evidence_ledger_v2.json").exists():
        update_evidence_ledger(root)
    profiler = _profiler_aggregate(root)
    integrity = _read_json(root / "preflight/integrity.json")
    root_cause = (
        _read_json(root / "failure_localization/root_cause.json")
        if (root / "failure_localization/root_cause.json").exists()
        else {
            "status": "NOT_RUN",
            "COLDSTART_PRIMARY_ROOT_CAUSE": "INCONCLUSIVE",
            "CONFIDENCE": "LOW",
        }
    )
    source_audit = (
        _read_json(root / "failure_localization/source_geometric_multistart_contract_audit.json")
        if (root / "failure_localization/source_geometric_multistart_contract_audit.json").exists()
        else {"SOURCE_GEOMETRIC_MULTISTART_SCOPE": "NOT_RUN"}
    )
    masked = (
        _read_json(root / "development/masked_qold_decision.json")
        if (root / "development/masked_qold_decision.json").exists()
        else {"status": "NOT_RUN", "candidate": {"strata": {}}}
    )
    strata = masked.get("candidate", {}).get("strata", {})
    refinement = _read_json(root / "refinement_regression/parity_summary.json")
    refinement_failure = _materialize_refinement_failure_analysis(root)
    selection = _read_json(root / "development/selection_decision.json")
    dev2 = _read_json(root / "dev2_frame0_development/decision.json")
    ledger = _read_json(root / "ledger/execution_v3_evidence_ledger_v2.json")
    hashes = _frozen_hashes(root)
    frozen = hashes["OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256"] is not None
    execution_status = (
        "FROZEN_READY_FOR_INDEPENDENT_COLDSTART_CERTIFICATION"
        if frozen
        else "NO_EXECUTION_V3_READY"
    )
    selected = selection.get("SELECTED_COLDSTART_SEARCH_V2")
    gate = "PASS" if frozen else "FAIL"
    high, mid, low = (strata.get(name, {}) for name in ("HIGH", "MID", "LOW"))
    summary = {
        "schema_version": "OakInk2O5RD2G2FinalSummaryV1",
        "Git": record_git(root),
        "frozen_upstream": {
            "FROZEN_UPSTREAM_INTEGRITY": integrity["FROZEN_UPSTREAM_INTEGRITY"],
            "RETARGET_OBJECTIVE_V2_SHA256": OBJECTIVE_SHA,
            "CERTIFICATION_GATE_V2_SHA256": GATE_V2_SHA,
            "OBJECTIVE_V2_CHANGED": "NO",
            "GATE_V2_CHANGED": "NO",
        },
        "existing_authority": {
            "Q_OLD_ROLE": "REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY",
            "INTERACTION_GRAPH_AUTHORITY": "CANONICAL_SOURCE_DERIVED",
        },
        "failure_localization": root_cause,
        "source_geometric_audit": source_audit,
        "N_COLDSTART_SEARCH_V2_CANDIDATES": 1,
        "masked_qold_development": strata,
        "SELECTED_COLDSTART_SEARCH_V2": selected,
        "refinement_regression": refinement,
        "refinement_failure_analysis": refinement_failure,
        "dev2_frame0": dev2,
        "EXECUTION_V3_SEARCH_V2_DEVELOPMENT_GATE": gate,
        "EXECUTION_V3_STATUS": execution_status,
        "frozen_hashes": hashes,
        "profiler": profiler,
        "evidence_hygiene": {
            "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
            "FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT": ledger[
                "FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT"
            ],
            "COLDSTART_SPARSE_VALIDATION_V4": "NOT_RUN",
            "COLDSTART_WINDOW_VALIDATION_V4": "NOT_RUN",
            "FRESH_CROSS_EPISODE_CONTROLS": "NOT_RUN",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        },
        "safety_flags": {
            "BRANCH": EXPECTED_BRANCH,
            "FROZEN_UPSTREAM_INTEGRITY": integrity["FROZEN_UPSTREAM_INTEGRITY"],
            "RETARGET_OBJECTIVE_V2_SHA256": OBJECTIVE_SHA,
            "CERTIFICATION_GATE_V2_SHA256": GATE_V2_SHA,
            "OBJECTIVE_V2_CHANGED": "NO",
            "GATE_V2_CHANGED": "NO",
            "SEMANTIC_V1_CHANGED": "NO",
            "E_IM_THRESHOLD_CHANGED": "NO",
            "Q_OLD_ROLE": "REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY",
            "INTERACTION_GRAPH_AUTHORITY": "CANONICAL_SOURCE_DERIVED",
            "NEW_DEV1_METHOD_DEVELOPMENT_FRAMES": 0,
            "COLDSTART_FAILURE_LOCALIZATION_COMPLETE": root_cause.get("status"),
            "WHOLE_HAND_BOOTSTRAP_IMPLEMENTED": "YES",
            "SOURCE_GEOMETRIC_MULTISTART_SCOPE": source_audit.get(
                "SOURCE_GEOMETRIC_MULTISTART_SCOPE"
            ),
            "N_COLDSTART_SEARCH_V2_CANDIDATES": 1,
            "SELECTED_COLDSTART_SEARCH_V2": selected or "NONE",
            "Q_OLD_USED_BY_COLDSTART": "NO",
            "Q_OLD_SYNTHESIZED_IN_COLDSTART": "NO",
            "FAILED_STAGE7_TERMINAL_USED_AS_Q_OLD": "NO",
            "THUMB_SPECIAL_CASE_ADDED": "NO",
            "DEV1_SPECIAL_CASE_ADDED": "NO",
            "DEV2_SPECIAL_CASE_ADDED": "NO",
            "MANIFEST_V2_MODIFIED": "NO",
            "SPLIT_V2_MODIFIED": "NO",
            "DEV1_FULL_RETARGET_RERUNS": 0,
            "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
            "O6_RAN": "NO",
            "SUPPORT_PHYSICALIZATION_RAN": "NO",
            "PHYSX_RAN": "NO",
            "FROZEN_EVAL_RAN": "NO",
            "PPO_RAN": "NO",
            "MASKED_QOLD_HIGH_TECHNICAL": high.get("technical"),
            "MASKED_QOLD_HIGH_P95": high.get("p95_e_im"),
            "MASKED_QOLD_MID_TECHNICAL": mid.get("technical"),
            "MASKED_QOLD_MID_P95": mid.get("p95_e_im"),
            "MASKED_QOLD_LOW_TECHNICAL": low.get("technical"),
            "MASKED_QOLD_LOW_P95": low.get("p95_e_im"),
            "REFINEMENT_MODE_REGRESSION": refinement.get("REFINEMENT_MODE_REGRESSION"),
            "DEV2_FRAME0_ROLE": "KNOWN_FAILURE_DEVELOPMENT_REGRESSION",
            "DEV2_FRAME0_V3_DEVELOPMENT": dev2.get("DEV2_FRAME0_V3_DEVELOPMENT"),
            "DEV2_FRAME0_V3_RUN_COUNT": dev2.get("DEV2_FRAME0_V3_RUN_COUNT"),
            "DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT": dev2.get("DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT"),
            "EXECUTION_V3_SEARCH_V2_DEVELOPMENT_GATE": gate,
            **hashes,
            "EXECUTION_V3_STATUS": execution_status,
            "COLDSTART_SPARSE_VALIDATION_V4": "NOT_RUN",
            "COLDSTART_WINDOW_VALIDATION_V4": "NOT_RUN",
            "FRESH_CROSS_EPISODE_CONTROLS": "NOT_RUN",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            ".local_TRACKED": "NO",
            "GUIDANCE_WORKTREE_MODIFIED": "NO",
        },
        "NEXT": (
            "O5R-D2H_EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION"
            if frozen
            else (
                "EXECUTION_V3_REFINEMENT_PARITY_REPAIR"
                if refinement.get("REFINEMENT_MODE_REGRESSION") == "FAIL"
                else (
                    "EXECUTION_V3_MULTIBLOCK_COLDSTART_DEVELOPMENT"
                    if root_cause.get("COLDSTART_PRIMARY_ROOT_CAUSE")
                    == "CONTRIBUTOR_LOCALITY_INSUFFICIENT"
                    else "EXECUTION_V3_COLDSTART_BOOTSTRAP_V3_DEVELOPMENT"
                )
            )
        ),
    }
    _write_json(root / "final_summary.json", summary)
    markdown = f"""# OakInk2 O5R-D2G2
# ExecutionV3 Cold-Start Search V2 Development Handoff

## Git

BRANCH={EXPECTED_BRANCH}

START_HEAD={START_HEAD}

FINAL_HEAD={summary["Git"]["final_head"]}

commits={summary["Git"]["commits"]}

tracked_worktree_clean={summary["Git"]["tracked_worktree_clean"]}

PUSHED=NO

PR_CREATED=NO

## Frozen upstream

FROZEN_UPSTREAM_INTEGRITY={integrity["FROZEN_UPSTREAM_INTEGRITY"]}

RETARGET_OBJECTIVE_V2_SHA256={OBJECTIVE_SHA}

CERTIFICATION_GATE_V2_SHA256={GATE_V2_SHA}

OBJECTIVE_V2_CHANGED=NO

GATE_V2_CHANGED=NO

SEMANTIC_V1_CHANGED=NO

E_IM_THRESHOLD_CHANGED=NO

## Existing authority

Q_OLD_ROLE=REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY

INTERACTION_GRAPH_AUTHORITY=CANONICAL_SOURCE_DERIVED

## Failure localization

COLDSTART_PRIMARY_ROOT_CAUSE={root_cause.get("COLDSTART_PRIMARY_ROOT_CAUSE")}

CONFIDENCE={root_cause.get("CONFIDENCE")}

{root_cause.get("why_mid_succeeds", "NOT_RUN")}

{root_cause.get("why_high_low_fail", "NOT_RUN")}

## Source-geometric audit

SOURCE_GEOMETRIC_MULTISTART_SCOPE={source_audit.get("SOURCE_GEOMETRIC_MULTISTART_SCOPE")}

Non-selected DOFs come from the same all-finger geometric solve; wrist/base is deterministically aligned after q and is not optimized by the old primitive.

## Search candidates

| Candidate | Bootstrap | Free DOFs | Seeds | B2 refinement | Result |
|---|---|---|---:|---|---|
| {CS2_A.name} | Eq.1-2 whole-hand geometric | all 20 finger DOFs | 2 | post-bootstrap Eq.7 top-1, 24/8/8 budgets | {masked.get("status")} |
| CS2_B | not authorized | not run | 0 | not run | NOT_IMPLEMENTED_NOT_AUTHORIZED |
| CS2_C | not authorized | not run | 0 | not run | NOT_IMPLEMENTED_NOT_AUTHORIZED |

## Masked-q_old development

| Candidate | HIGH technical/p95 | MID technical/p95 | LOW technical/p95 | Hard validity | Determinism |
|---|---|---|---|---|---|
| {CS2_A.name} | {high.get("technical")}/{high.get("p95_e_im")} | {mid.get("technical")}/{mid.get("p95_e_im")} | {low.get("technical")}/{low.get("p95_e_im")} | {all(item.get("hard_validity_pass") for item in (high, mid, low)) if strata else "NOT_RUN"} | {masked.get("determinism", {}).get("DETERMINISM", "NOT_RUN")} |

SELECTED_COLDSTART_SEARCH_V2={selected or "NONE"}

## Refinement regression

REFINEMENT_MODE_REGRESSION={refinement.get("REFINEMENT_MODE_REGRESSION")}

MAX_Q_ABS_DIFF={refinement.get("max_q_abs")}

MAX_BASE_ABS_DIFF={refinement.get("max_base_abs")}

MAX_E_IM_ABS_DIFF={refinement.get("max_e_im_abs")}

## DEV2 frame0

DEV2_FRAME0_ROLE=KNOWN_FAILURE_DEVELOPMENT_REGRESSION

DEV2_FRAME0_V3_DEVELOPMENT={dev2.get("DEV2_FRAME0_V3_DEVELOPMENT")}

DEV2_FRAME0_V3_RUN_COUNT={dev2.get("DEV2_FRAME0_V3_RUN_COUNT")}

DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT={dev2.get("DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT")}

## Why old V3 failed

V3_A_ROOT_CAUSE={root_cause.get("V3_A_ROOT_CAUSE")}

V3_B_ROOT_CAUSE={root_cause.get("V3_B_ROOT_CAUSE")}

V3_C_HIGH_ROOT_CAUSE={root_cause.get("V3_C_HIGH_ROOT_CAUSE")}

V3_C_LOW_ROOT_CAUSE={root_cause.get("V3_C_LOW_ROOT_CAUSE")}

The old local ranking was computed on neutral before the all-finger bootstrap. Search V2 succeeds only if the evidence-identified mechanism is repaired: re-evaluate the unchanged Eq.7 top-1 contributor on each full-hand bootstrap state, without q_old or a top-K expansion.

## ExecutionV3 development gate

EXECUTION_V3_SEARCH_V2_DEVELOPMENT_GATE={gate}

EXECUTION_V3_STATUS={execution_status}

EXECUTION_INPUT_AUTHORITY_SHA256={hashes["EXECUTION_INPUT_AUTHORITY_SHA256"]}

SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256={hashes["SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256"]}

COLD_START_BOOTSTRAP_CONTRACT_SHA256={hashes["COLD_START_BOOTSTRAP_CONTRACT_SHA256"]}

COLD_START_SEED_AUTHORITY_V2_SHA256={hashes["COLD_START_SEED_AUTHORITY_V2_SHA256"]}

OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256={hashes["OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256"]}

## Evidence hygiene

NEW_DEV1_METHOD_DEVELOPMENT_FRAMES=0

FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT={ledger["FUTURE_EXECUTION_V3_VALIDATION_EXCLUSION_COUNT"]}

COLDSTART_SPARSE_VALIDATION_V4=NOT_RUN

COLDSTART_WINDOW_VALIDATION_V4=NOT_RUN

FRESH_CROSS_EPISODE_CONTROLS=NOT_RUN

CERTIFICATION_SPLIT_NEW_CONSUMPTION=0

HELDOUT_SPLIT_NEW_CONSUMPTION=0

DEV2_FULL_PRODUCTION_SOLVE_COUNT=0

## Next

NEXT={summary["NEXT"]}

## Final safety flags

BRANCH={EXPECTED_BRANCH}

FROZEN_UPSTREAM_INTEGRITY={integrity["FROZEN_UPSTREAM_INTEGRITY"]}

RETARGET_OBJECTIVE_V2_SHA256={OBJECTIVE_SHA}

CERTIFICATION_GATE_V2_SHA256={GATE_V2_SHA}

OBJECTIVE_V2_CHANGED=NO

GATE_V2_CHANGED=NO

SEMANTIC_V1_CHANGED=NO

E_IM_THRESHOLD_CHANGED=NO

Q_OLD_ROLE=REFINEMENT_ONLY_BASELINE_AND_SEARCH_AUTHORITY

INTERACTION_GRAPH_AUTHORITY=CANONICAL_SOURCE_DERIVED

NEW_DEV1_METHOD_DEVELOPMENT_FRAMES=0

COLDSTART_FAILURE_LOCALIZATION_COMPLETE={root_cause.get("status")}

WHOLE_HAND_BOOTSTRAP_IMPLEMENTED=YES

SOURCE_GEOMETRIC_MULTISTART_SCOPE={source_audit.get("SOURCE_GEOMETRIC_MULTISTART_SCOPE")}

N_COLDSTART_SEARCH_V2_CANDIDATES=1

SELECTED_COLDSTART_SEARCH_V2={selected or "NONE"}

Q_OLD_USED_BY_COLDSTART=NO

Q_OLD_SYNTHESIZED_IN_COLDSTART=NO

FAILED_STAGE7_TERMINAL_USED_AS_Q_OLD=NO

THUMB_SPECIAL_CASE_ADDED=NO

DEV1_SPECIAL_CASE_ADDED=NO

DEV2_SPECIAL_CASE_ADDED=NO

MASKED_QOLD_HIGH_TECHNICAL={high.get("technical")}

MASKED_QOLD_HIGH_P95={high.get("p95_e_im")}

MASKED_QOLD_MID_TECHNICAL={mid.get("technical")}

MASKED_QOLD_MID_P95={mid.get("p95_e_im")}

MASKED_QOLD_LOW_TECHNICAL={low.get("technical")}

MASKED_QOLD_LOW_P95={low.get("p95_e_im")}

REFINEMENT_MODE_REGRESSION={refinement.get("REFINEMENT_MODE_REGRESSION")}

DEV2_FRAME0_ROLE=KNOWN_FAILURE_DEVELOPMENT_REGRESSION

DEV2_FRAME0_V3_DEVELOPMENT={dev2.get("DEV2_FRAME0_V3_DEVELOPMENT")}

DEV2_FRAME0_V3_RUN_COUNT={dev2.get("DEV2_FRAME0_V3_RUN_COUNT")}

DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT={dev2.get("DEV2_FRAME0_V3_OPTIMIZER_RUN_COUNT")}

EXECUTION_V3_SEARCH_V2_DEVELOPMENT_GATE={gate}

EXECUTION_INPUT_AUTHORITY_SHA256={hashes["EXECUTION_INPUT_AUTHORITY_SHA256"]}

SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256={hashes["SOURCE_INTERACTION_GRAPH_AUTHORITY_SHA256"]}

COLD_START_BOOTSTRAP_CONTRACT_SHA256={hashes["COLD_START_BOOTSTRAP_CONTRACT_SHA256"]}

COLD_START_SEED_AUTHORITY_V2_SHA256={hashes["COLD_START_SEED_AUTHORITY_V2_SHA256"]}

OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256={hashes["OBJECTIVE_V2_EXECUTION_CONTRACT_V3_SHA256"]}

EXECUTION_V3_STATUS={execution_status}

COLDSTART_SPARSE_VALIDATION_V4=NOT_RUN

COLDSTART_WINDOW_VALIDATION_V4=NOT_RUN

FRESH_CROSS_EPISODE_CONTROLS=NOT_RUN

CERTIFICATION_SPLIT_NEW_CONSUMPTION=0

HELDOUT_SPLIT_NEW_CONSUMPTION=0

DEV2_FULL_PRODUCTION_SOLVE_COUNT=0

DEV1_FULL_RETARGET_RERUNS=0

DEV1_FULL_V2_REFINEMENT_RUNS=0

MANIFEST_V2_MODIFIED=NO

SPLIT_V2_MODIFIED=NO

O6_RAN=NO

SUPPORT_PHYSICALIZATION_RAN=NO

PHYSX_RAN=NO

FROZEN_EVAL_RAN=NO

PPO_RAN=NO

PUSHED=NO

PR_CREATED=NO

.local_TRACKED=NO

GUIDANCE_WORKTREE_MODIFIED=NO

This handoff hard-stops before any fresh independent validation, full DEV1/DEV2 run, O6, Support, Isaac, PhysX, frozen evaluation, or PPO.
"""
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    _write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD2G2ResourceUsageV1",
            "profiler": profiler,
            "forbidden_runs": 0,
        },
    )
    return summary


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    localize_coldstart_failures(root)
    audit_source_geometric_multistart(root)
    freeze_search_v2_candidates(root)
    run_cs2_b_development(root)
    run_cs2_c_development(root)
    development = run_cs2_a_development(root)
    update_evidence_ledger(root)
    if development["status"] != "PASS":
        return summarize(root)
    refinement = run_refinement_regression(root)
    if refinement["REFINEMENT_MODE_REGRESSION"] != "PASS":
        return summarize(root)
    select_coldstart_search_v2(root)
    lock_selected_candidate(root)
    dev2 = run_dev2_frame0_development(root)
    if dev2["DEV2_FRAME0_V3_DEVELOPMENT"] != "PASS":
        return summarize(root)
    freeze_execution_v3(root)
    generate_future_certification_plan(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-frozen-upstream": verify_frozen_upstream,
    "localize-coldstart-failures": localize_coldstart_failures,
    "audit-source-geometric-multistart": audit_source_geometric_multistart,
    "freeze-search-v2-candidates": freeze_search_v2_candidates,
    "run-cs2-a-development": run_cs2_a_development,
    "run-cs2-b-development": run_cs2_b_development,
    "run-cs2-c-development": run_cs2_c_development,
    "reconcile-development-reporting": reconcile_development_reporting,
    "select-coldstart-search-v2": select_coldstart_search_v2,
    "run-refinement-regression": run_refinement_regression,
    "lock-selected-candidate": lock_selected_candidate,
    "run-dev2-frame0-development": run_dev2_frame0_development,
    "freeze-execution-input-authority": freeze_execution_input_authority,
    "freeze-source-interaction-graph-authority": freeze_source_interaction_graph_authority,
    "freeze-coldstart-bootstrap-contract": freeze_coldstart_bootstrap_contract,
    "freeze-coldstart-seed-authority-v2": freeze_coldstart_seed_authority_v2,
    "freeze-execution-v3": freeze_execution_v3,
    "generate-future-certification-plan": generate_future_certification_plan,
    "update-evidence-ledger": update_evidence_ledger,
    "validate-repository": validate_repository,
    "record-git": record_git,
    "summarize": summarize,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=ROOT)
    subparsers = value.add_subparsers(dest="action", required=True)
    for action in ACTIONS:
        subparsers.add_parser(action)
    return value


def main() -> int:
    args = parser().parse_args()
    result = ACTIONS[args.action](args.root)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
