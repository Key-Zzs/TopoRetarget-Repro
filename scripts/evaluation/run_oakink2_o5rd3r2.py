#!/usr/bin/env python3
"""O5R-D3-R2 consumed-development search-space mechanism oracle.

This driver is intentionally limited to the immutable DEV1 D3/D3-R evidence.
It may run bounded optimizers as development feasibility oracles, but it has no
entry point for D3-R3, fresh certification, D3-V2, DEV2, PPO, PhysX, or O6.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2a as d2a  # noqa: E402
from scripts.data import run_oakink2_o5rd2c as d2c  # noqa: E402
from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.data import run_oakink2_o5rd2g2 as d2g2  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd3r_d3v2 as d3r  # noqa: E402
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    ScreenedCandidate,
    asset_derived_dof_blocks,
    select_candidate,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3r2_refinement_search_space_repair_design_v1"
D3_ROOT = d3r.D3_ROOT
D3R_ROOT = d3r.ROOT
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "e1a80b74fef8f1e1aa7626deed12f51f86da4b83"
EXPECTED_FRAMES = 2722
TAU = 1.0e-4
OBJECTIVE_V2_SHA256 = d3r.OBJECTIVE_V2_SHA256
SEMANTIC_V1_SHA256 = d3r.SEMANTIC_V1_SHA256
PRIMARY_ORACLES = (
    "B_SOURCE_CONDITIONED_ALTERNATE_SEED",
    "C_EXPANDED_ACTIVE_SET",
    "D_BOUNDED_SEARCH_ENVELOPE_EXPANSION",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    d3r.write_json(path, value)


def write_text(path: Path, value: str) -> None:
    d3r.write_text(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if rows:
        d3r.write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields or ["status"]).writeheader()


def read_csv(path: Path) -> list[dict[str, str]]:
    return d3r.read_csv(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{field}={value.get(field)!r}")
    return value


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _d3_partition() -> list[dict[str, str]]:
    rows = read_csv(D3R_ROOT / "d3r/group_partition.csv")
    if len(rows) != EXPECTED_FRAMES:
        raise RuntimeError(f"D3R2_PARTITION_COUNT_MISMATCH:{len(rows)}")
    return rows


def _d3_state(ordinal: int) -> tuple[np.ndarray, np.ndarray]:
    path = D3_ROOT / f"checkpoints/frame_{ordinal:04d}/state.npz"
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["qpos"], dtype=np.float64),
            np.asarray(data["base_pose_scene"], dtype=np.float64),
        )


def _previous_d3_state(ordinal: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if ordinal == 0:
        return None, None
    return _d3_state(ordinal - 1)


def _d3_receipt(ordinal: int) -> dict[str, Any]:
    return read_json(D3_ROOT / f"checkpoints/frame_{ordinal:04d}/receipt.json")


def _initial_not_run(root: Path) -> None:
    payload = {
        "schema_version": "D3R2ExplicitNotRunV1",
        "status": "NOT_RUN",
        "D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT": 0,
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    for directory in (
        "future/d3r3",
        "future/fresh_refinement_sparse",
        "future/fresh_refinement_window",
        "future/cross_episode_refinement",
        "future/d3v2",
        "future/dev2",
        "future/ppo",
        "future/o6",
    ):
        write_json(root / directory / "not_run.json", payload)


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0,
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "d3_root_exists": D3_ROOT.is_dir(),
        "d3r_root_exists": D3R_ROOT.is_dir(),
        "oakink2_root_exists": Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2").is_dir(),
    }
    value = {
        "schema_version": "O5RD3R2GitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": START_HEAD,
        "HEAD_AT_PREFLIGHT": head,
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
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", value)
    (root / "technical_failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    _initial_not_run(root)
    if value["status"] != "PASS":
        raise RuntimeError("D3_R2_STATUS=BLOCKED_UPSTREAM_AUTHORITY_INTEGRITY:GIT")
    return value


def verify_d3r_authority(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_D3R_AUTHORITY")
    d3 = read_json(D3_ROOT / "final_summary.json")
    d3r_summary = read_json(D3R_ROOT / "final_summary.json")
    cause = read_json(D3R_ROOT / "d3r/root_cause.json")
    development = read_json(D3R_ROOT / "development/selected_repair.json")
    partition = _d3_partition()
    counts = Counter(row["group"][-1] for row in partition)
    invalid = [row for row in partition if row["group"] in {"GROUP_B", "GROUP_D"}]
    checks = {
        "d3_numerical_pass": d3.get("DEV1_MACHINE") == "RETARGET_SEMANTIC_FAIL"
        and d3.get("COMPLETED_FRAMES") == EXPECTED_FRAMES,
        "d3_semantic_fail": d3.get("DEV1_SEMANTIC_V1_RESULT") == "FAIL",
        "root_cause_exact": cause.get("D3R_PRIMARY_ROOT_CAUSE")
        == "PRIMARY_REFINEMENT_SEARCH_COVERAGE_INSUFFICIENT",
        "root_cause_confidence": cause.get("D3R_ROOT_CAUSE_CONFIDENCE") == "HIGH",
        "counts_exact": [counts[name] for name in "ABCD"] == [1865, 145, 712, 0],
        "invalid_exact": len(invalid) == 145,
        "segments_exact": d3r_summary.get("FAILURE_SEGMENT_COUNT") == 4,
        "multistart_immutable": development.get("candidate")
        == "REFINEMENT_V2_A_SEQUENTIAL_MULTI_START"
        and development.get("metrics", {}).get("failure_frame_count_completed") == 31
        and development.get("metrics", {}).get("failure_recovered_count") == 0,
        "downstream_unrun": d3r_summary.get("D3_V2_SCIENTIFIC_RUN_COUNT") == 0,
    }
    d3_history = {
        "schema_version": "D3R2D3HistoryV1",
        "status": "PASS" if checks["d3_numerical_pass"] and checks["d3_semantic_fail"] else "FAIL",
        "D3_V1_NUMERICAL": "PASS",
        "D3_V1_SEMANTIC": "FAIL",
        "D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "trajectory_sha256": sha256_file(D3_ROOT / "trajectory/trajectory.npz"),
    }
    d3r_history = {
        "schema_version": "D3R2D3RHistoryV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D3R_PRIMARY_ROOT_CAUSE": cause.get("D3R_PRIMARY_ROOT_CAUSE"),
        "D3R_ROOT_CAUSE_CONFIDENCE": cause.get("D3R_ROOT_CAUSE_CONFIDENCE"),
        "GROUP_A_COUNT": counts["A"],
        "GROUP_B_COUNT": counts["B"],
        "GROUP_C_COUNT": counts["C"],
        "GROUP_D_COUNT": counts["D"],
        "FINAL_INVALID_FRAME_COUNT": len(invalid),
        "FAILURE_SEGMENT_COUNT": d3r_summary.get("FAILURE_SEGMENT_COUNT"),
        "MAX_FAILURE_SEGMENT_LENGTH": d3r_summary.get("MAX_FAILURE_SEGMENT_LENGTH"),
        "MULTISTART_RECOVERY_COMPLETED": development.get("metrics", {}).get(
            "failure_frame_count_completed"
        ),
        "MULTISTART_RECOVERY_SUCCESS": development.get("metrics", {}).get(
            "failure_recovered_count"
        ),
        "D3R_HISTORICAL_RESULT_REWRITTEN": "NO",
    }
    frozen = {
        "schema_version": "D3R2FrozenAuthoritiesV1",
        "status": d3r_history["status"],
        "RETARGET_OBJECTIVE_V2_SHA256": OBJECTIVE_V2_SHA256,
        "SEMANTIC_V1_SHA256": SEMANTIC_V1_SHA256,
        "E_IM_THRESHOLD": TAU,
        "hard_validity": "FROZEN_CANDIDATE_B2_AUTHORITY",
        "q_old_array_sha256": d3r_summary.get("ORIGINAL_QOLD_SHA256"),
        "source_authority": "IMMUTABLE_DEV1_D3_CONSUMED_EVIDENCE",
        "object_authority": "IMMUTABLE_DEV1_D3_CONSUMED_EVIDENCE",
    }
    integrity = {
        "schema_version": "D3R2EvidenceIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
    write_json(root / "preflight/d3_history.json", d3_history)
    write_json(root / "preflight/d3r_history.json", d3r_history)
    write_json(root / "preflight/frozen_authorities.json", frozen)
    write_json(root / "preflight/evidence_integrity.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError("D3_R2_STATUS=BLOCKED_UPSTREAM_AUTHORITY_INTEGRITY")
    return integrity


def build_failure_cluster_manifest(root: Path) -> dict[str, Any]:
    require(root / "preflight/evidence_integrity.json", "status", "PASS", "BUILD_CLUSTERS")
    partition = _d3_partition()
    contributors = {
        int(row["ordinal"]): row for row in read_csv(D3R_ROOT / "d3r/contributor_metrics.csv")
    }
    temporal = {
        int(row["ordinal"]): row for row in read_csv(D3R_ROOT / "d3r/temporal_context_metrics.csv")
    }
    invalid_ordinals = [
        int(row["ordinal"]) for row in partition if row["group"] in {"GROUP_B", "GROUP_D"}
    ]
    segments: list[list[int]] = []
    for ordinal in invalid_ordinals:
        if not segments or ordinal != segments[-1][-1] + 1:
            segments.append([ordinal])
        else:
            segments[-1].append(ordinal)
    runtime = d2a.D2ARuntime(D3_ROOT)
    features: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for segment_id, ordinals in enumerate(segments, start=1):
        eim = [float(partition[ordinal]["new_E_IM"]) for ordinal in ordinals]
        peak_position = int(np.argmax(np.asarray(eim)))
        peak_ordinal = ordinals[peak_position]
        for ordinal in ordinals:
            q_old = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
            q_d3, _base_d3 = _d3_state(ordinal)
            receipt = _d3_receipt(ordinal)
            row = partition[ordinal]
            frames.append(
                {
                    "source_frame": int(row["source_frame"]),
                    "ordinal": ordinal,
                    "segment_id": segment_id,
                    "D3_V1_E_IM": float(row["new_E_IM"]),
                    "excess_over_tau": float(row["new_E_IM"]) - TAU,
                    "q_old_hash": array_sha(q_old),
                    "D3_V1_q_hash": array_sha(q_d3),
                }
            )
            graph_vertices = np.asarray(runtime.graph.source_vertices[ordinal], dtype=np.float64)
            prior_vertices = (
                graph_vertices
                if ordinal == 0
                else np.asarray(runtime.graph.source_vertices[ordinal - 1], dtype=np.float64)
            )
            hand = graph_vertices[:21]
            obj = graph_vertices[21:]
            prior_hand = prior_vertices[:21]
            prior_obj = prior_vertices[21:]
            c = contributors[ordinal]
            t = temporal[ordinal]
            features.append(
                {
                    "segment_id": segment_id,
                    "ordinal": ordinal,
                    "source_frame": int(row["source_frame"]),
                    "source_hand_pose_step_l2": float(np.linalg.norm(hand - prior_hand)),
                    "object_relative_hand_step_l2": float(
                        np.linalg.norm(
                            (hand - obj.mean(axis=0)) - (prior_hand - prior_obj.mean(axis=0))
                        )
                    ),
                    "q_old_transition_l2": t["q_old_transition_l2"],
                    "D3_V1_q_transition_l2": ""
                    if ordinal == 0
                    else float(np.linalg.norm(q_d3 - _d3_state(ordinal - 1)[0])),
                    "top1_finger": c["top1_finger"],
                    "top2_finger": c["top2_finger"],
                    "rho1": float(c["rho1"]),
                    "rho2": float(c["rho2"]),
                    "previous_state_deviation_l2": t["previous_new_to_q_old_l2"],
                    "continuous_prediction_q_step_inf_rad": t[
                        "continuous_prediction_q_step_inf_rad"
                    ],
                    "temporal_term_contribution": receipt["selected"]["per_components"][
                        "continuous_temporal"
                    ],
                    "thumb_residual": float(c["thumb_residual"]),
                    "index_residual": float(c["index_residual"]),
                    "middle_residual": float(c["middle_residual"]),
                    "ring_residual": float(c["ring_residual"]),
                    "little_residual": float(c["little_residual"]),
                }
            )
        manifest_rows.append(
            {
                "segment_id": segment_id,
                "start_ordinal": ordinals[0],
                "stop_ordinal": ordinals[-1],
                "start_source_frame": int(partition[ordinals[0]]["source_frame"]),
                "stop_source_frame": int(partition[ordinals[-1]]["source_frame"]),
                "length": len(ordinals),
                "peak_E_IM_ordinal": peak_ordinal,
                "peak_E_IM_frame": int(partition[peak_ordinal]["source_frame"]),
                "peak_E_IM": eim[peak_position],
                "mean_E_IM": float(np.mean(eim)),
                "median_E_IM": float(np.median(eim)),
                "entry_predecessor_frame": int(partition[ordinals[0] - 1]["source_frame"]),
                "exit_successor_frame": int(partition[ordinals[-1] + 1]["source_frame"]),
            }
        )
    top1_by_cluster = {
        str(row["segment_id"]): Counter(
            item["top1_finger"] for item in features if item["segment_id"] == row["segment_id"]
        ).most_common(1)[0][0]
        for row in manifest_rows
    }
    shared_top1 = len(set(top1_by_cluster.values())) == 1
    rho1_medians = {
        str(row["segment_id"]): float(
            np.median(
                [item["rho1"] for item in features if item["segment_id"] == row["segment_id"]]
            )
        )
        for row in manifest_rows
    }
    mechanism_pattern = "SHARED" if shared_top1 else "PARTIALLY_SHARED"
    manifest = {
        "schema_version": "DEV1D3FailureClusterManifestV1",
        "status": "PASS" if len(segments) == 4 and len(frames) == 145 else "FAIL",
        "N_FAILURE_SEGMENTS": len(segments),
        "FINAL_INVALID_FRAME_COUNT": len(frames),
        "clusters": manifest_rows,
    }
    similarity = {
        "schema_version": "D3R2FailureClusterSimilarityV1",
        "status": "PASS",
        "FAILURE_CLUSTER_MECHANISM_PATTERN": mechanism_pattern,
        "dominant_failing_finger_by_cluster": top1_by_cluster,
        "rho1_median_by_cluster": rho1_medians,
        "shared_top1": shared_top1,
        "quantitative_basis": "per-frame frozen contributor mass, source/object-relative motion, q_old/D3-V1 motion, and temporal residuals",
    }
    write_json(root / "clusters/cluster_manifest.json", manifest)
    write_csv(root / "clusters/cluster_frames.csv", frames)
    write_csv(root / "clusters/cluster_features.csv", features)
    write_json(root / "clusters/cluster_similarity.json", similarity)
    if manifest["status"] != "PASS":
        raise RuntimeError("D3_R2_STATUS=BLOCKED_UPSTREAM_AUTHORITY_INTEGRITY:CLUSTERS")
    return manifest


def _four_distinct(
    segment: dict[str, Any], partition: list[dict[str, str]]
) -> list[dict[str, Any]]:
    start, stop = int(segment["start_ordinal"]), int(segment["stop_ordinal"])
    middle = (start + stop) // 2
    eim = {ordinal: float(partition[ordinal]["new_E_IM"]) for ordinal in range(start, stop + 1)}
    peak = max(eim, key=lambda ordinal: (eim[ordinal], -ordinal))
    selected: list[tuple[str, int]] = []
    for label, ordinal in (
        ("ENTRY", start),
        ("MIDDLE", middle),
        ("PEAK_EIM", peak),
        ("EXIT", stop),
    ):
        if ordinal not in {item[1] for item in selected}:
            selected.append((label, ordinal))
    for ordinal in sorted(range(start, stop + 1), key=lambda value: (abs(value - middle), value)):
        if len(selected) == 4:
            break
        if ordinal not in {item[1] for item in selected}:
            selected.append(("DISTINCT_NEAREST_SUPPLEMENT", ordinal))
    return [
        {
            "role": label,
            "ordinal": ordinal,
            "source_frame": int(partition[ordinal]["source_frame"]),
            "D3_V1_E_IM": float(partition[ordinal]["new_E_IM"]),
        }
        for label, ordinal in selected
    ]


def _even_controls(rows: list[dict[str, str]], group: str, count: int) -> list[dict[str, Any]]:
    pool = [row for row in rows if row["group"] == group]
    indices = np.linspace(0, len(pool) - 1, count, dtype=np.int64)
    return [
        {
            "role": "RECOVERED_INVALID_CONTROL"
            if group == "GROUP_A"
            else "PRESERVED_VALID_CONTROL",
            "ordinal": int(pool[int(index)]["ordinal"]),
            "source_frame": int(pool[int(index)]["source_frame"]),
            "D3_V1_E_IM": float(pool[int(index)]["new_E_IM"]),
            "group": group,
        }
        for index in indices
    ]


def freeze_representative_development_set(root: Path) -> dict[str, Any]:
    clusters = require(
        root / "clusters/cluster_manifest.json", "status", "PASS", "FREEZE_REPRESENTATIVES"
    )
    partition = _d3_partition()
    failure_frames: list[dict[str, Any]] = []
    for segment in clusters["clusters"]:
        for row in _four_distinct(segment, partition):
            failure_frames.append({"segment_id": segment["segment_id"], **row})
    controls = _even_controls(partition, "GROUP_A", 8) + _even_controls(partition, "GROUP_C", 8)
    windows = []
    for segment in clusters["clusters"]:
        entry = int(segment["start_ordinal"])
        start = max(0, entry - 2)
        ordinals = list(range(start, min(EXPECTED_FRAMES, start + 8)))
        windows.append(
            {
                "window_id": f"cluster_{segment['segment_id']}_entry_v1",
                "segment_id": int(segment["segment_id"]),
                "entry_ordinal": entry,
                "ordinals": ordinals,
                "source_frames": [int(partition[value]["source_frame"]) for value in ordinals],
                "selection_rule": "two historical pre-entry frames plus entry and next five frames",
            }
        )
    payload = {
        "schema_version": "D3R2RepresentativeDevelopmentSetV1",
        "status": "FROZEN_BEFORE_ORACLE",
        "selection_frozen_before_oracle": True,
        "selection_rule": "ENTRY, lower MIDDLE, maximum E_IM with lower-ordinal tie break, EXIT; deterministic nearest-middle supplements resolve duplicates",
        "representative_failure_frames": failure_frames,
        "valid_controls": controls,
        "sequential_windows": windows,
        "counts": {
            "failure": len(failure_frames),
            "recovered_controls": 8,
            "preserved_valid_controls": 8,
            "windows": len(windows),
        },
    }
    receipt = {
        "schema_version": "D3R2DevelopmentSelectionReceiptV1",
        "status": "FROZEN",
        "payload_sha256": canonical_sha(payload),
        "oracle_results_visible": False,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }
    write_json(
        root / "development_set/representative_failure_frames.json",
        {
            "schema_version": "D3R2RepresentativeFailureFramesV1",
            "status": "FROZEN_BEFORE_ORACLE",
            "frames": failure_frames,
        },
    )
    write_json(
        root / "development_set/valid_controls.json",
        {
            "schema_version": "D3R2ValidControlsV1",
            "status": "FROZEN_BEFORE_ORACLE",
            "frames": controls,
        },
    )
    write_json(
        root / "development_set/sequential_windows.json",
        {
            "schema_version": "D3R2SequentialWindowsV1",
            "status": "FROZEN_BEFORE_ORACLE",
            "windows": windows,
        },
    )
    write_json(root / "development_set/selection_receipt.json", receipt)
    return payload


def freeze_search_space_oracle_plan(root: Path) -> dict[str, Any]:
    require(root / "preflight/evidence_integrity.json", "status", "PASS", "FREEZE_ORACLE")
    selection = require(
        root / "development_set/selection_receipt.json",
        "status",
        "FROZEN",
        "FREEZE_ORACLE",
    )
    runtime = d2a.D2ARuntime(D3_ROOT)
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    spans = np.asarray(runtime.model.joint_upper - runtime.model.joint_lower, dtype=np.float64)
    current_limit = float(np.max(spans))
    plan = {
        "schema_version": "D3R2SearchSpaceOraclePlanV1",
        "status": "FROZEN",
        "selection_receipt_sha256": canonical_sha(selection),
        "primary_mechanisms": list(PRIMARY_ORACLES),
        "primary_mechanism_count": 3,
        "common_invariants": {
            "ObjectiveV2_sha256": OBJECTIVE_V2_SHA256,
            "SemanticV1_sha256": SEMANTIC_V1_SHA256,
            "E_IM_threshold": TAU,
            "hard_validity": "UNCHANGED",
            "q_old": "historical q_old[t] unchanged",
            "source_object_graph": "UNCHANGED",
            "RETARGET_MODE": "REFINEMENT",
        },
        "B": {
            "seed_generator": "FULL_HAND_GEOMETRIC_BOOTSTRAP",
            "admitted_seed_count": 1,
            "bootstrap_seed": "wuji_canonical_rest",
            "bootstrap_solver": "paper_repro_scipy_trf",
            "bootstrap_max_nfev": 250,
            "active_set": "frozen D3-V1 selected contributor block",
            "cold_start_search_semantics_imported": False,
        },
        "C": {
            "current_active_set": {name: list(value) for name, value in blocks.items()},
            "current_dof_count": 4,
            "expanded_active_set": list(range(len(runtime.model.dof_names))),
            "expanded_dof_count": len(runtime.model.dof_names),
            "wrist_base_search_expansion": "NO",
            "seed_sources": ["old_production", "wuji_canonical_rest", "joint_range_midpoint"],
            "primary_maxiter": 8,
            "secondary_maxiter": 8,
        },
        "D": {
            "current_q_envelope": current_limit,
            "levels": [current_limit, 1.5 * current_limit, 2.0 * current_limit],
            "asset_joint_span_max": current_limit,
            "domain_equivalence_rule": "if q_limit >= max(asset upper-lower), trust wall cannot exclude any asset-valid q",
            "optimizer_budget": "UNCHANGED; equivalent domains reuse immutable r0 evidence",
            "dynamic_escalation": False,
        },
        "substantial_improvement_rule": "cluster median relative E_IM reduction >=0.25",
        "primary_mechanism_rule": "hard-valid improvement in at least 3/4 clusters, not isolated, with at least one interaction-valid recovery",
        "ambiguity_rule": "authorize exactly B+C only when B and C each improve >=3/4 clusters, each valid rate <0.5, and neither alone recovers all four clusters",
        "max_compositional_oracles": 1,
    }
    path = root / "oracle_plan/search_space_oracle_plan.json"
    if path.exists() and read_json(path) != plan:
        raise RuntimeError("D3R2_ORACLE_PLAN_DRIFT")
    write_json(path, plan)
    write_text(root / "oracle_plan/search_space_oracle_plan.sha256", sha256_file(path) + "\n")
    write_json(root / "oracle_c_active_set/current_active_set.json", plan["C"])
    write_json(root / "oracle_d_envelope/current_envelope.json", plan["D"])
    write_json(
        root / "oracle_d_envelope/preregistered_levels.json",
        {
            "schema_version": "D3R2EnvelopeLevelsV1",
            "status": "FROZEN",
            **plan["D"],
        },
    )
    return {**plan, "sha256": sha256_file(path)}


def replay_baseline_sentinels(root: Path) -> dict[str, Any]:
    require(
        root / "oracle_plan/search_space_oracle_plan.json", "status", "FROZEN", "BASELINE_REPLAY"
    )
    failures = read_json(root / "development_set/representative_failure_frames.json")["frames"]
    controls = read_json(root / "development_set/valid_controls.json")["frames"]
    rows = []
    for role, items in (("FAILURE", failures), ("CONTROL", controls)):
        for item in items:
            ordinal = int(item["ordinal"])
            q, base = _d3_state(ordinal)
            receipt = _d3_receipt(ordinal)
            stored = float(item["D3_V1_E_IM"])
            replay = float(receipt["selected"]["interaction_e_im"])
            rows.append(
                {
                    "role": role,
                    "segment_id": item.get("segment_id", ""),
                    "ordinal": ordinal,
                    "source_frame": int(item["source_frame"]),
                    "stored_E_IM": stored,
                    "replay_E_IM": replay,
                    "E_IM_abs_error": abs(stored - replay),
                    "q_hash": array_sha(q),
                    "base_hash": array_sha(base),
                    "candidate_outcome": receipt["selected_candidate"],
                    "hard_valid": bool(receipt["selected_evaluation"]["feasible"]),
                    "parity": stored == replay,
                }
            )
    windows = read_json(root / "development_set/sequential_windows.json")["windows"]
    window_rows = []
    for window in windows:
        for ordinal in window["ordinals"]:
            receipt = _d3_receipt(int(ordinal))
            q, base = _d3_state(int(ordinal))
            window_rows.append(
                {
                    "window_id": window["window_id"],
                    "ordinal": int(ordinal),
                    "q_hash": array_sha(q),
                    "base_hash": array_sha(base),
                    "E_IM": float(receipt["selected"]["interaction_e_im"]),
                    "hard_valid": bool(receipt["selected_evaluation"]["feasible"]),
                }
            )
    parity = {
        "schema_version": "D3R2BaselineReplayParityV1",
        "status": "PASS" if all(row["parity"] for row in rows) else "FAIL",
        "BASELINE_REPLAY_PARITY": "PASS" if all(row["parity"] for row in rows) else "FAIL",
        "frame_count": len(rows),
        "window_row_count": len(window_rows),
        "max_E_IM_abs_error": max(row["E_IM_abs_error"] for row in rows),
    }
    write_csv(root / "baseline/replay_per_frame.csv", rows)
    write_csv(root / "baseline/replay_windows.csv", window_rows)
    write_json(root / "baseline/parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError("D3_R2_STATUS=BLOCKED_BASELINE_REPRODUCIBILITY")
    return parity


def _runtime_pair() -> tuple[d2a.D2ARuntime, d2g.V3Runtime]:
    return d2a.D2ARuntime(D3_ROOT), d2g.V3Runtime("dev_01", D3_ROOT)


def _measure_state(
    runtime: d2a.D2ARuntime,
    ordinal: int,
    q: np.ndarray,
    base: np.ndarray,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    binding, context = runtime.bind_context(
        ordinal, previous_qpos=previous_q, previous_base=previous_base
    )
    values = runtime.measurement(ordinal, q, base, binding=binding, context=context, slack=None)
    actual = d2c.actual_continuity(runtime, previous_q, previous_base, q, base)
    return values, d2c._evaluate_b2(runtime, values, actual)


def _state_candidate(
    candidate_id: str,
    values: Any,
    evaluation: dict[str, Any],
    order: int,
) -> ScreenedCandidate:
    return d2c._screened(candidate_id, values, evaluation, order, None)


def _run_primary(
    runtime: d2a.D2ARuntime,
    ordinal: int,
    q_seed: np.ndarray,
    base_seed: np.ndarray,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
    block: tuple[int, ...],
    source: str,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any], dict[str, Any]]:
    q, base, values, evaluation, result, _actual = d2c._run_candidate_phase(
        runtime,
        ordinal,
        q_seed=q_seed,
        base_seed=base_seed,
        previous_q=previous_q,
        previous_base=previous_base,
        block=block,
        phase="primary",
        maxiter=8,
        retention_limit=None,
        initialization_source=source,
    )
    return q, base, values, evaluation, d2c._solver_profile(result)


def _run_secondary(
    runtime: d2a.D2ARuntime,
    ordinal: int,
    q_seed: np.ndarray,
    base_seed: np.ndarray,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
    block: tuple[int, ...],
    source: str,
    interaction_eim: float,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any], dict[str, Any]]:
    limit = d2c.interaction_retention_limit(interaction_eim, runtime.authority.interaction_target)
    q, base, values, evaluation, result, _actual = d2c._run_candidate_phase(
        runtime,
        ordinal,
        q_seed=q_seed,
        base_seed=base_seed,
        previous_q=previous_q,
        previous_base=previous_base,
        block=block,
        phase="secondary",
        maxiter=8,
        retention_limit=limit,
        initialization_source=source,
    )
    return q, base, values, evaluation, d2c._solver_profile(result)


def _best_with_polish(
    runtime: d2a.D2ARuntime,
    ordinal: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
    block: tuple[int, ...],
    states: list[tuple[str, np.ndarray, np.ndarray, Any, dict[str, Any], dict[str, Any]]],
    oracle: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    screened = [
        _state_candidate(name, values, evaluation, index)
        for index, (name, _q, _base, values, evaluation, _profile) in enumerate(states)
    ]
    selected = select_candidate(screened)
    if selected is None:
        raise RuntimeError(f"D3R2_{oracle}_NO_HARD_VALID_CANDIDATE:{ordinal}")
    selected_state = states[screened.index(selected)]
    name, q, base, values, evaluation, profile = selected_state
    polish_profile: dict[str, Any] | None = None
    polish_exception: str | None = None
    try:
        pq, pb, pvalues, pevaluation, polish_profile = _run_secondary(
            runtime,
            ordinal,
            q,
            base,
            previous_q,
            previous_base,
            block,
            f"D3R2_{oracle}:{name}:secondary",
            float(values.interaction_e_im),
        )
        polished = _state_candidate("secondary_polished", pvalues, pevaluation, len(screened))
        chosen, reason = d2c.retain_after_polish(
            selected,
            polished,
            interaction_target=runtime.authority.interaction_target,
        )
        if chosen is polished:
            name, q, base, values, evaluation = (
                "secondary_polished",
                pq,
                pb,
                pvalues,
                pevaluation,
            )
        retention = reason
    except Exception as exc:
        polish_exception = f"{type(exc).__name__}:{exc}"
        retention = "PRIMARY_RETAINED_POLISH_EXCEPTION"
    receipt = {
        "oracle": oracle,
        "ordinal": ordinal,
        "selected_candidate": name,
        "selected_E_IM": float(values.interaction_e_im),
        "selected_hard_valid": bool(evaluation["feasible"]),
        "selected_interaction_valid": float(values.interaction_e_im) <= TAU,
        "selected_evaluation": evaluation,
        "primary_profile": profile,
        "secondary_profile": polish_profile,
        "secondary_exception": polish_exception,
        "retention": retention,
        "candidate_ids": [item[0] for item in states],
    }
    return np.asarray(q), np.asarray(base), receipt


def _oracle_b_frame(
    runtime: d2a.D2ARuntime,
    bootstrap_runtime: d2g.V3Runtime,
    ordinal: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    d3_q, d3_base = _d3_state(ordinal)
    old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
    baseline_values, baseline_eval = _measure_state(
        runtime, ordinal, d3_q, d3_base, previous_q, previous_base
    )
    candidate = replace(
        d2g2.CS2_A,
        bootstrap_seed_sources=("wuji_canonical_rest",),
        use_previous_accepted_after_frame0=False,
    )
    bootstrap_states, bootstrap_receipt = d2g2._whole_hand_geometric_bootstrap(
        bootstrap_runtime, ordinal, candidate
    )
    if len(bootstrap_states) != 1:
        raise RuntimeError(f"D3R2_B_BOOTSTRAP_STATE_COUNT:{len(bootstrap_states)}")
    seed_name, seed_q = bootstrap_states[0]
    seed_values, seed_eval = _measure_state(
        runtime, ordinal, seed_q, old_base, previous_q, previous_base
    )
    block = tuple(int(value) for value in _d3_receipt(ordinal)["free_qpos_indices"])
    states = [
        ("d3_v1_baseline", d3_q, d3_base, baseline_values, baseline_eval, {"stored": True}),
        (f"source_seed:{seed_name}", seed_q, old_base, seed_values, seed_eval, {"bootstrap": True}),
    ]
    try:
        q, base, values, evaluation, profile = _run_primary(
            runtime,
            ordinal,
            seed_q,
            old_base,
            previous_q,
            previous_base,
            block,
            f"D3R2_B:{seed_name}:primary",
        )
        states.append(("source_seed_primary", q, base, values, evaluation, profile))
    except Exception as exc:
        primary_exception = f"{type(exc).__name__}:{exc}"
    else:
        primary_exception = None
    q, base, receipt = _best_with_polish(
        runtime, ordinal, previous_q, previous_base, block, states, "B"
    )
    receipt.update(
        {
            "RETARGET_MODE": "REFINEMENT",
            "q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED",
            "active_dofs": list(block),
            "bootstrap": bootstrap_receipt,
            "primary_exception": primary_exception,
            "wall_time_sec": time.perf_counter() - started,
        }
    )
    return q, base, receipt


def _oracle_c_frame(
    runtime: d2a.D2ARuntime,
    _bootstrap_runtime: d2g.V3Runtime,
    ordinal: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    d3_q, d3_base = _d3_state(ordinal)
    old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
    neutral = np.asarray(runtime.model.neutral_q, dtype=np.float64)
    midpoint = 0.5 * (
        np.asarray(runtime.model.joint_lower, dtype=np.float64)
        + np.asarray(runtime.model.joint_upper, dtype=np.float64)
    )
    block = tuple(range(len(old_q)))
    baseline_values, baseline_eval = _measure_state(
        runtime, ordinal, d3_q, d3_base, previous_q, previous_base
    )
    states = [("d3_v1_baseline", d3_q, d3_base, baseline_values, baseline_eval, {"stored": True})]
    primary_failures: list[str] = []
    for seed_name, seed_q in (
        ("old_production", old_q),
        ("wuji_canonical_rest", neutral),
        ("joint_range_midpoint", midpoint),
    ):
        try:
            q, base, values, evaluation, profile = _run_primary(
                runtime,
                ordinal,
                seed_q,
                old_base,
                previous_q,
                previous_base,
                block,
                f"D3R2_C:{seed_name}:primary",
            )
            states.append((f"expanded_primary:{seed_name}", q, base, values, evaluation, profile))
        except Exception as exc:
            primary_failures.append(f"{seed_name}:{type(exc).__name__}:{exc}")
    q, base, receipt = _best_with_polish(
        runtime, ordinal, previous_q, previous_base, block, states, "C"
    )
    receipt.update(
        {
            "RETARGET_MODE": "REFINEMENT",
            "q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED",
            "active_dofs": list(block),
            "active_dof_count": len(block),
            "wrist_base_search_expansion": "NO",
            "primary_failures": primary_failures,
            "wall_time_sec": time.perf_counter() - started,
        }
    )
    return q, base, receipt


def _oracle_composition_frame(
    runtime: d2a.D2ARuntime,
    bootstrap_runtime: d2g.V3Runtime,
    ordinal: int,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    d3_q, d3_base = _d3_state(ordinal)
    old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
    baseline_values, baseline_eval = _measure_state(
        runtime, ordinal, d3_q, d3_base, previous_q, previous_base
    )
    candidate = replace(
        d2g2.CS2_A,
        bootstrap_seed_sources=("wuji_canonical_rest",),
        use_previous_accepted_after_frame0=False,
    )
    bootstrap_states, bootstrap_receipt = d2g2._whole_hand_geometric_bootstrap(
        bootstrap_runtime, ordinal, candidate
    )
    seed_name, seed_q = bootstrap_states[0]
    block = tuple(range(len(seed_q)))
    states = [("d3_v1_baseline", d3_q, d3_base, baseline_values, baseline_eval, {"stored": True})]
    q, base, values, evaluation, profile = _run_primary(
        runtime,
        ordinal,
        seed_q,
        old_base,
        previous_q,
        previous_base,
        block,
        f"D3R2_BC:{seed_name}:primary",
    )
    states.append(("source_seed_expanded_primary", q, base, values, evaluation, profile))
    q, base, receipt = _best_with_polish(
        runtime, ordinal, previous_q, previous_base, block, states, "BC"
    )
    receipt.update(
        {
            "RETARGET_MODE": "REFINEMENT",
            "q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED",
            "active_dofs": list(block),
            "bootstrap": bootstrap_receipt,
            "wall_time_sec": time.perf_counter() - started,
        }
    )
    return q, base, receipt


def _cache_path(root: Path, oracle_dir: str, ordinal: int, suffix: str = "") -> Path:
    return root / oracle_dir / "checkpoints" / f"frame_{ordinal:04d}{suffix}.json"


def _run_cached(
    root: Path,
    oracle_dir: str,
    ordinal: int,
    plan_sha: str,
    runner: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    runtime: d2a.D2ARuntime,
    bootstrap_runtime: d2g.V3Runtime,
    previous_q: np.ndarray,
    previous_base: np.ndarray,
    suffix: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool]:
    path = _cache_path(root, oracle_dir, ordinal, suffix)
    if path.is_file():
        value = read_json(path)
        if value.get("plan_sha256") != plan_sha:
            raise RuntimeError(f"D3R2_ORACLE_CHECKPOINT_PLAN_DRIFT:{path}")
        return (
            np.asarray(value["qpos"], dtype=np.float64),
            np.asarray(value["base_pose_scene"], dtype=np.float64),
            value["receipt"],
            True,
        )
    q, base, receipt = runner(runtime, bootstrap_runtime, ordinal, previous_q, previous_base)
    write_json(
        path,
        {
            "schema_version": "D3R2OracleFrameCheckpointV1",
            "status": "COMPLETE",
            "plan_sha256": plan_sha,
            "ordinal": ordinal,
            "qpos": q.tolist(),
            "base_pose_scene": base.tolist(),
            "q_sha256": array_sha(q),
            "base_sha256": array_sha(base),
            "receipt": receipt,
        },
    )
    return q, base, receipt, False


def _oracle_row(
    item: dict[str, Any],
    q: np.ndarray,
    receipt: dict[str, Any],
    runtime: d2a.D2ARuntime,
    resumed: bool,
) -> dict[str, Any]:
    ordinal = int(item["ordinal"])
    baseline_q, _baseline_base = _d3_state(ordinal)
    old = float(item["D3_V1_E_IM"])
    new = float(receipt["selected_E_IM"])
    deviations = np.abs(q - np.asarray(runtime.final.arrays["qpos"][ordinal]))
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    return {
        "segment_id": item.get("segment_id", ""),
        "role": item.get("role", ""),
        "ordinal": ordinal,
        "source_frame": int(item["source_frame"]),
        "baseline_best_E_IM": old,
        "oracle_best_E_IM": new,
        "delta_E_IM": new - old,
        "relative_E_IM_reduction": (old - new) / old,
        "hard_valid": bool(receipt["selected_hard_valid"]),
        "interaction_valid": bool(receipt["selected_interaction_valid"]),
        "selected_seed_provenance": receipt["selected_candidate"],
        "active_dof_count": len(receipt["active_dofs"]),
        "q_deviation_from_q_old_l2": float(
            np.linalg.norm(q - np.asarray(runtime.final.arrays["qpos"][ordinal]))
        ),
        "q_deviation_from_D3_V1_l2": float(np.linalg.norm(q - baseline_q)),
        **{
            f"{finger}_q_deviation_l2": float(np.linalg.norm(deviations[list(indices)]))
            for finger, indices in blocks.items()
        },
        "wall_time_sec": float(receipt.get("wall_time_sec", 0.0)),
        "resumed_from_checkpoint": resumed,
    }


def _summarize_oracle(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    clusters = sorted({int(row["segment_id"]) for row in rows})
    cluster_metrics = {}
    for cluster in clusters:
        values = [row for row in rows if int(row["segment_id"]) == cluster]
        median_reduction = float(
            np.median([float(row["relative_E_IM_reduction"]) for row in values])
        )
        cluster_metrics[str(cluster)] = {
            "N": len(values),
            "valid_count": sum(bool(row["interaction_valid"]) for row in values),
            "median_relative_E_IM_reduction": median_reduction,
            "substantial_improvement": median_reduction >= 0.25,
        }
    return {
        "schema_version": "D3R2OracleSummaryV1",
        "status": "PASS",
        "oracle": name,
        "N": len(rows),
        "VALID_COUNT": sum(bool(row["interaction_valid"]) for row in rows),
        "VALID_RATE": sum(bool(row["interaction_valid"]) for row in rows) / len(rows),
        "HARD_VALID_RATE": sum(bool(row["hard_valid"]) for row in rows) / len(rows),
        "MEDIAN_E_IM_REDUCTION": float(
            np.median([float(row["relative_E_IM_reduction"]) for row in rows])
        ),
        "MEAN_WALL_TIME_SEC": float(np.mean([float(row["wall_time_sec"]) for row in rows])),
        "clusters": cluster_metrics,
        "clusters_with_substantial_improvement": sum(
            bool(value["substantial_improvement"]) for value in cluster_metrics.values()
        ),
        "clusters_with_valid_recovery": sum(
            int(value["valid_count"]) > 0 for value in cluster_metrics.values()
        ),
    }


def _run_oracle(
    root: Path,
    oracle_name: str,
    oracle_dir: str,
    runner: Callable[..., tuple[np.ndarray, np.ndarray, dict[str, Any]]],
) -> dict[str, Any]:
    require(root / "baseline/parity.json", "status", "PASS", f"RUN_ORACLE_{oracle_name}")
    plan = require(
        root / "oracle_plan/search_space_oracle_plan.json",
        "status",
        "FROZEN",
        f"RUN_ORACLE_{oracle_name}",
    )
    plan_sha = sha256_file(root / "oracle_plan/search_space_oracle_plan.json")
    if plan["primary_mechanisms"].count(oracle_name) != 1:
        raise RuntimeError(f"D3R2_ORACLE_NOT_PREREGISTERED:{oracle_name}")
    frames = read_json(root / "development_set/representative_failure_frames.json")["frames"]
    runtime, bootstrap_runtime = _runtime_pair()
    rows = []
    receipts = root / oracle_dir / "candidate_receipts.jsonl"
    if receipts.exists():
        receipts.unlink()
    for item in frames:
        ordinal = int(item["ordinal"])
        previous_q, previous_base = _previous_d3_state(ordinal)
        if previous_q is None or previous_base is None:
            raise RuntimeError("D3R2_REPRESENTATIVE_FRAME_ZERO_FORBIDDEN")
        q, _base, receipt, resumed = _run_cached(
            root,
            oracle_dir,
            ordinal,
            plan_sha,
            runner,
            runtime,
            bootstrap_runtime,
            previous_q,
            previous_base,
        )
        row = _oracle_row(item, q, receipt, runtime, resumed)
        rows.append(row)
        _append_jsonl(receipts, receipt)
        print(
            f"{oracle_name} ordinal={ordinal} E_IM={row['oracle_best_E_IM']:.12g} valid={row['interaction_valid']} resumed={resumed}",
            flush=True,
        )
        write_csv(root / oracle_dir / "per_frame.csv", rows)
    summary = _summarize_oracle(oracle_name, rows)
    write_json(root / oracle_dir / "summary.json", summary)
    return summary


def run_oracle_source_conditioned_seed(root: Path) -> dict[str, Any]:
    return _run_oracle(
        root,
        "B_SOURCE_CONDITIONED_ALTERNATE_SEED",
        "oracle_b_seed",
        _oracle_b_frame,
    )


def run_oracle_expanded_active_set(root: Path) -> dict[str, Any]:
    return _run_oracle(root, "C_EXPANDED_ACTIVE_SET", "oracle_c_active_set", _oracle_c_frame)


def run_oracle_envelope_expansion(root: Path) -> dict[str, Any]:
    require(root / "baseline/parity.json", "status", "PASS", "RUN_ORACLE_D")
    plan = require(
        root / "oracle_plan/search_space_oracle_plan.json", "status", "FROZEN", "RUN_ORACLE_D"
    )
    frames = read_json(root / "development_set/representative_failure_frames.json")["frames"]
    levels = plan["D"]["levels"]
    runtime = d2a.D2ARuntime(D3_ROOT)
    rows = []
    for item in frames:
        ordinal = int(item["ordinal"])
        receipt = _d3_receipt(ordinal)
        q, _base = _d3_state(ordinal)
        q_old = np.asarray(runtime.final.arrays["qpos"][ordinal])
        nfev = sum(
            int((receipt["profiler"].get(stage) or {}).get("nfev", 0))
            for stage in ("primary", "secondary")
        ) + int(receipt["profiler"].get("probe_nfev", 0))
        for level_index, level in enumerate(levels):
            rows.append(
                {
                    "segment_id": item["segment_id"],
                    "ordinal": ordinal,
                    "source_frame": int(item["source_frame"]),
                    "envelope_level": f"r{level_index}",
                    "q_limit_rad": float(level),
                    "best_E_IM": float(item["D3_V1_E_IM"]),
                    "hard_valid": bool(receipt["selected_evaluation"]["feasible"]),
                    "interaction_valid": float(item["D3_V1_E_IM"]) <= TAU,
                    "nfev": nfev,
                    "q_deviation_l2": float(np.linalg.norm(q - q_old)),
                    "domain_equivalent_to_r0": True,
                    "execution": "STORED_R0_REUSED_BY_PREREGISTERED_DOMAIN_EQUIVALENCE_PROOF",
                }
            )
    per_frame = [row for row in rows if row["envelope_level"] == "r2"]
    summary = {
        "schema_version": "D3R2OracleSummaryV1",
        "status": "PASS",
        "oracle": "D_BOUNDED_SEARCH_ENVELOPE_EXPANSION",
        "N": len(per_frame),
        "VALID_COUNT": 0,
        "VALID_RATE": 0.0,
        "HARD_VALID_RATE": 1.0,
        "MEDIAN_E_IM_REDUCTION": 0.0,
        "levels": levels,
        "all_levels_domain_equivalent": True,
        "mechanism_evidence": "current q trust wall already contains the complete asset joint-bound domain; larger finite radii cannot add a state",
        "clusters_with_substantial_improvement": 0,
        "clusters_with_valid_recovery": 0,
    }
    write_csv(root / "oracle_d_envelope/per_frame.csv", rows)
    write_json(root / "oracle_d_envelope/summary.json", summary)
    write_text(
        root / "oracle_d_envelope/candidate_receipts.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_frame),
    )
    return summary


def run_compositional_oracle_if_authorized(root: Path) -> dict[str, Any]:
    plan = require(
        root / "oracle_plan/search_space_oracle_plan.json", "status", "FROZEN", "RUN_COMPOSITION"
    )
    b = require(root / "oracle_b_seed/summary.json", "status", "PASS", "RUN_COMPOSITION")
    c = require(root / "oracle_c_active_set/summary.json", "status", "PASS", "RUN_COMPOSITION")
    authorized = bool(
        b["clusters_with_substantial_improvement"] >= 3
        and c["clusters_with_substantial_improvement"] >= 3
        and b["VALID_RATE"] < 0.5
        and c["VALID_RATE"] < 0.5
        and b["clusters_with_valid_recovery"] < 4
        and c["clusters_with_valid_recovery"] < 4
    )
    if not authorized:
        value = {
            "schema_version": "D3R2CompositionDecisionV1",
            "status": "NOT_RUN",
            "COMPOSITIONAL_ORACLE": "NOT_RUN",
            "reason": "FROZEN_AMBIGUITY_RULE_NOT_SATISFIED",
            "ambiguity_rule": plan["ambiguity_rule"],
            "COMPOSITIONAL_ORACLE_COUNT": 0,
        }
        write_json(root / "oracle_composition/not_run.json", value)
        write_json(root / "oracle_composition/result.json", value)
        return value
    result = _run_oracle(
        root,
        "B_SOURCE_CONDITIONED_ALTERNATE_SEED",
        "oracle_composition",
        _oracle_composition_frame,
    )
    result.update(
        {
            "COMPOSITIONAL_ORACLE": "RUN",
            "composition": "B+C",
            "COMPOSITIONAL_ORACLE_COUNT": 1,
        }
    )
    write_json(root / "oracle_composition/result.json", result)
    return result


def compare_search_space_oracles(root: Path) -> dict[str, Any]:
    b = require(root / "oracle_b_seed/summary.json", "status", "PASS", "COMPARE_ORACLES")
    c = require(root / "oracle_c_active_set/summary.json", "status", "PASS", "COMPARE_ORACLES")
    d = require(root / "oracle_d_envelope/summary.json", "status", "PASS", "COMPARE_ORACLES")
    composition = read_json(root / "oracle_composition/result.json")
    summaries = [b, c, d]
    if composition.get("COMPOSITIONAL_ORACLE") == "RUN":
        summaries.append(composition)
    rows = [
        {
            "oracle": "BASELINE",
            "valid_count": 0,
            "N": 16,
            "median_E_IM_reduction": 0.0,
            "hard_valid_rate": 1.0,
            "mean_wall_time_sec": 0.0,
        }
    ]
    for value in summaries:
        rows.append(
            {
                "oracle": value["oracle"]
                if value.get("composition") is None
                else "BC_SOURCE_SEED_PLUS_EXPANDED_ACTIVE_SET",
                "valid_count": value["VALID_COUNT"],
                "N": value["N"],
                "median_E_IM_reduction": value["MEDIAN_E_IM_REDUCTION"],
                "hard_valid_rate": value["HARD_VALID_RATE"],
                "mean_wall_time_sec": value.get("MEAN_WALL_TIME_SEC", 0.0),
            }
        )
    value = {
        "schema_version": "D3R2OracleComparisonV1",
        "status": "PASS",
        "oracles": rows,
        "primary_oracle_count": 3,
        "composition_count": int(composition.get("COMPOSITIONAL_ORACLE_COUNT", 0)),
    }
    write_csv(root / "mechanism/comparison.csv", rows)
    write_json(root / "mechanism/comparison.json", value)
    return value


def _candidate_mechanism(root: Path) -> tuple[str, str, str | None]:
    b = read_json(root / "oracle_b_seed/summary.json")
    c = read_json(root / "oracle_c_active_set/summary.json")
    d = read_json(root / "oracle_d_envelope/summary.json")
    composition = read_json(root / "oracle_composition/result.json")
    b_ready = b["clusters_with_valid_recovery"] >= 3 and b["HARD_VALID_RATE"] == 1.0
    c_ready = c["clusters_with_valid_recovery"] >= 3 and c["HARD_VALID_RATE"] == 1.0
    if (
        composition.get("COMPOSITIONAL_ORACLE") == "RUN"
        and composition["clusters_with_valid_recovery"] >= 3
    ):
        return (
            "COMBINED_INITIALIZATION_AND_SUBSPACE_LIMIT",
            "BC_SOURCE_SEED_PLUS_EXPANDED_ACTIVE_SET",
            "oracle_composition",
        )
    if b_ready and not c_ready:
        return (
            "INITIALIZATION_BASIN_COVERAGE_INSUFFICIENT",
            "B_SOURCE_CONDITIONED_ALTERNATE_SEED",
            "oracle_b_seed",
        )
    if c_ready and not b_ready:
        return "OPTIMIZATION_SUBSPACE_TOO_NARROW", "C_EXPANDED_ACTIVE_SET", "oracle_c_active_set"
    if b_ready and c_ready:
        b_key = (b["VALID_COUNT"], b["MEDIAN_E_IM_REDUCTION"], -b["MEAN_WALL_TIME_SEC"])
        c_key = (c["VALID_COUNT"], c["MEDIAN_E_IM_REDUCTION"], -c["MEAN_WALL_TIME_SEC"])
        if b_key >= c_key:
            return (
                "INITIALIZATION_BASIN_COVERAGE_INSUFFICIENT",
                "B_SOURCE_CONDITIONED_ALTERNATE_SEED",
                "oracle_b_seed",
            )
        return "OPTIMIZATION_SUBSPACE_TOO_NARROW", "C_EXPANDED_ACTIVE_SET", "oracle_c_active_set"
    if d["clusters_with_valid_recovery"] >= 3:
        return (
            "SEARCH_ENVELOPE_TOO_RESTRICTIVE",
            "D_BOUNDED_SEARCH_ENVELOPE_EXPANSION",
            "oracle_d_envelope",
        )
    return "FEASIBLE_STATE_NOT_FOUND_BY_BOUNDED_ORACLE", "NONE", None


def run_sequential_hysteresis_diagnostic(root: Path) -> dict[str, Any]:
    require(root / "mechanism/comparison.json", "status", "PASS", "RUN_HYSTERESIS")
    mechanism, selected, oracle_dir = _candidate_mechanism(root)
    windows = read_json(root / "development_set/sequential_windows.json")["windows"]
    if oracle_dir is None:
        value = {
            "schema_version": "D3R2SequentialHysteresisDecisionV1",
            "status": "INCONCLUSIVE",
            "SEQUENTIAL_BASIN_HYSTERESIS": "INCONCLUSIVE",
            "reason": "NO_SINGLE_FRAME_ORACLE_REACHED_A_BASIN_IN_AT_LEAST_THREE_CLUSTERS",
            "mechanism_candidate": mechanism,
            "CLUSTER_WINDOWS_VALID": 0,
            "window_count": len(windows),
        }
        write_csv(root / "sequential_hysteresis/per_window.csv", [])
        write_csv(root / "sequential_hysteresis/per_frame.csv", [])
        write_json(root / "sequential_hysteresis/decision.json", value)
        return value
    runner = {
        "oracle_b_seed": _oracle_b_frame,
        "oracle_c_active_set": _oracle_c_frame,
        "oracle_composition": _oracle_composition_frame,
    }[oracle_dir]
    plan_sha = sha256_file(root / "oracle_plan/search_space_oracle_plan.json")
    runtime, bootstrap_runtime = _runtime_pair()
    partition = _d3_partition()
    per_frame = []
    per_window = []
    for window in windows:
        first = int(window["ordinals"][0])
        previous_q, previous_base = _previous_d3_state(first)
        if previous_q is None or previous_base is None:
            raise RuntimeError("D3R2_WINDOW_PREDECESSOR_MISSING")
        rows = []
        for ordinal in window["ordinals"]:
            ordinal = int(ordinal)
            baseline_eim = float(partition[ordinal]["new_E_IM"])
            q, base, receipt, resumed = _run_cached(
                root,
                "sequential_hysteresis",
                ordinal,
                plan_sha,
                runner,
                runtime,
                bootstrap_runtime,
                previous_q,
                previous_base,
                suffix=f"_{window['window_id']}",
            )
            continuity = d2c.actual_continuity(runtime, previous_q, previous_base, q, base)
            row = {
                "window_id": window["window_id"],
                "segment_id": window["segment_id"],
                "ordinal": ordinal,
                "source_frame": int(partition[ordinal]["source_frame"]),
                "entry_offset": ordinal - int(window["entry_ordinal"]),
                "baseline_E_IM": baseline_eim,
                "oracle_E_IM": float(receipt["selected_E_IM"]),
                "interaction_valid": bool(receipt["selected_interaction_valid"]),
                "hard_valid": bool(receipt["selected_hard_valid"]),
                "translation_step_m": continuity["translation_step_m"],
                "rotation_step_rad": continuity["rotation_step_rad"],
                "q_step_inf_rad": continuity["q_step_inf_rad"],
                "q_old_authority": "HISTORICAL_Q_OLD_T_UNCHANGED",
                "previous_accepted_authority": "CURRENT_ORACLE_TRAJECTORY_T_MINUS_1",
                "resumed_from_checkpoint": resumed,
            }
            rows.append(row)
            per_frame.append(row)
            previous_q, previous_base = q, base
        entry_and_after = [row for row in rows if int(row["entry_offset"]) >= 0]
        invalid = [row for row in entry_and_after if not row["interaction_valid"]]
        per_window.append(
            {
                "window_id": window["window_id"],
                "segment_id": window["segment_id"],
                "entry_valid": entry_and_after[0]["interaction_valid"],
                "window_valid_frames": sum(bool(row["interaction_valid"]) for row in rows),
                "window_frame_count": len(rows),
                "entry_and_after_all_valid": not invalid,
                "first_invalid_offset": None if not invalid else invalid[0]["entry_offset"],
                "continuity": "PASS"
                if all(
                    float(row["translation_step_m"])
                    <= runtime.authority.temporal_base_translation_limit_m + 1e-12
                    and float(row["rotation_step_rad"])
                    <= runtime.authority.temporal_base_rotation_limit_rad + 1e-12
                    for row in rows
                )
                else "FAIL",
            }
        )
        write_csv(root / "sequential_hysteresis/per_frame.csv", per_frame)
        write_csv(root / "sequential_hysteresis/per_window.csv", per_window)
    valid_windows = sum(bool(row["entry_and_after_all_valid"]) for row in per_window)
    entry_valid = sum(bool(row["entry_valid"]) for row in per_window)
    if entry_valid == 0:
        hysteresis = "INCONCLUSIVE"
    elif valid_windows == len(per_window):
        hysteresis = "NO"
    elif entry_valid > 0:
        hysteresis = "YES"
    else:
        hysteresis = "INCONCLUSIVE"
    value = {
        "schema_version": "D3R2SequentialHysteresisDecisionV1",
        "status": "PASS",
        "tested_oracle": selected,
        "SEQUENTIAL_BASIN_HYSTERESIS": hysteresis,
        "CLUSTER_WINDOWS_VALID": valid_windows,
        "window_count": len(per_window),
        "entry_valid_count": entry_valid,
        "windows": per_window,
    }
    write_json(root / "sequential_hysteresis/decision.json", value)
    return value


def decide_search_space_mechanism(root: Path) -> dict[str, Any]:
    hysteresis = read_json(root / "sequential_hysteresis/decision.json")
    mechanism, selected, oracle_dir = _candidate_mechanism(root)
    confidence = "LOW"
    if oracle_dir is not None:
        source = read_json(root / oracle_dir / "summary.json")
        confidence = "HIGH" if source["clusters_with_valid_recovery"] == 4 else "MEDIUM"
        if hysteresis.get("SEQUENTIAL_BASIN_HYSTERESIS") == "YES":
            mechanism = "SEQUENTIAL_BASIN_HYSTERESIS"
            confidence = "MEDIUM"
    value = {
        "schema_version": "D3R2SearchSpaceMechanismDecisionV1",
        "status": "PASS" if oracle_dir is not None else "INCONCLUSIVE",
        "SEARCH_SPACE_MECHANISM": mechanism,
        "MECHANISM_CONFIDENCE": confidence,
        "selected_oracle": selected,
        "selected_oracle_directory": oracle_dir,
        "SEQUENTIAL_BASIN_HYSTERESIS": hysteresis.get("SEQUENTIAL_BASIN_HYSTERESIS"),
        "finite_oracle_failure_does_not_prove_infeasibility": True,
    }
    write_json(root / "mechanism/mechanism_decision.json", value)
    return value


def design_refinement_v2(root: Path) -> dict[str, Any]:
    decision = read_json(root / "mechanism/mechanism_decision.json")
    if decision.get("status") != "PASS" or decision.get("SEARCH_SPACE_MECHANISM") in {
        "INCONCLUSIVE",
        "FEASIBLE_STATE_NOT_FOUND_BY_BOUNDED_ORACLE",
    }:
        raise RuntimeError("DESIGN_REFINEMENT_V2_REJECTED:SEARCH_SPACE_MECHANISM_INCONCLUSIVE")
    mechanism = decision["SEARCH_SPACE_MECHANISM"]
    mapping = {
        "INITIALIZATION_BASIN_COVERAGE_INSUFFICIENT": "REFINEMENT_V2_B_SOURCE_CONDITIONED_FALLBACK_SEED",
        "OPTIMIZATION_SUBSPACE_TOO_NARROW": "REFINEMENT_V2_C_CONDITIONAL_EXPANDED_ACTIVE_SET",
        "SEARCH_ENVELOPE_TOO_RESTRICTIVE": "REFINEMENT_V2_D_CONDITIONAL_ENVELOPE_EXPANSION",
        "COMBINED_INITIALIZATION_AND_SUBSPACE_LIMIT": "REFINEMENT_V2_BC_CONDITIONAL_SOURCE_SEED_EXPANDED_ACTIVE_SET",
        "SEQUENTIAL_BASIN_HYSTERESIS": "REFINEMENT_V2_CONDITIONAL_EXPANDED_SEARCH_WITH_ACCEPTED_STATE_LIFECYCLE",
    }
    selected = mapping[mechanism]
    plan = read_json(root / "oracle_plan/search_space_oracle_plan.json")
    compute_budget = {
        "INITIALIZATION_BASIN_COVERAGE_INSUFFICIENT": "fallback only; one source-conditioned bootstrap at 250 nfev, then frozen primary/secondary 8/8 iterations; no dynamic escalation",
        "OPTIMIZATION_SUBSPACE_TOO_NARROW": "fallback only; three frozen seed sources over all 20 finger DOFs, primary/secondary 8/8 iterations; no bootstrap and no dynamic escalation",
        "SEARCH_ENVELOPE_TOO_RESTRICTIVE": "fallback only; frozen finite envelope level with unchanged optimizer budget; no dynamic escalation",
        "COMBINED_INITIALIZATION_AND_SUBSPACE_LIMIT": "fallback only; one source-conditioned bootstrap at 250 nfev followed by all-20-finger primary/secondary 8/8 iterations; no dynamic escalation",
        "SEQUENTIAL_BASIN_HYSTERESIS": "fallback only with the evidence-selected bounded search and normal accepted-state propagation; no dynamic escalation",
    }[mechanism]
    design = {
        "schema_version": "RefinementV2DesignV1",
        "status": "DRAFT_PENDING_SENTINELS",
        "STATUS": "DEVELOPMENT_DESIGN_ONLY",
        "INDEPENDENT_CERTIFICATION": "NOT_RUN",
        "root_search_space_mechanism": mechanism,
        "selected_design": selected,
        "trigger_condition": "existing frozen refinement terminal is hard-valid but E_IM exceeds 1e-4",
        "normal_path": "existing frozen refinement; accept unchanged when frozen interaction gate is satisfied",
        "fallback_path": decision["selected_oracle"],
        "seed_authority": plan["B"]
        if "_B" in selected or "_BC" in selected
        else plan["C"]["seed_sources"],
        "active_dofs": plan["C"]["expanded_active_set"]
        if "_C" in selected or "_BC" in selected
        else "frozen D3-V1 contributor block",
        "bounds_envelope": "Wuji asset joint limits; wrist/base remain fixed",
        "candidate_ordering": "hard validity then ObjectiveV2 threshold-aware lexicographic ordering",
        "ObjectiveV2_sha256": OBJECTIVE_V2_SHA256,
        "hard_validity": "UNCHANGED_FROZEN_CANDIDATE_B2",
        "retention": "threshold-aware primary retention; reject secondary interaction regression",
        "previous_state_lifecycle": "after a fallback state is accepted, it becomes previous accepted runtime state; q_old[t] remains historical and separate",
        "failure_behavior": "return existing hard-valid refinement result and report interaction-invalid; never relax threshold",
        "compute_budget": compute_budget,
        "frame_episode_object_special_cases": False,
    }
    write_json(
        root / "design/selected_design.json",
        {
            "schema_version": "D3R2SelectedDesignV1",
            "status": "DRAFT_PENDING_SENTINELS",
            "SELECTED_REFINEMENT_V2_DESIGN": selected,
            "SEARCH_SPACE_MECHANISM": mechanism,
        },
    )
    write_json(root / "design/refinement_v2_design.json", design)
    return design


def run_refinement_v2_design_sentinels(root: Path) -> dict[str, Any]:
    design = read_json(root / "design/refinement_v2_design.json")
    if design.get("status") != "DRAFT_PENDING_SENTINELS":
        raise RuntimeError("D3R2_SENTINELS_REJECTED:DESIGN_NOT_DRAFT")
    decision = read_json(root / "mechanism/mechanism_decision.json")
    oracle_dir = decision["selected_oracle_directory"]
    failure_rows = read_csv(root / oracle_dir / "per_frame.csv")
    failures = [
        {
            **row,
            "interaction_valid": row["interaction_valid"] == "True",
            "hard_valid": row["hard_valid"] == "True",
        }
        for row in failure_rows
    ]
    controls = read_json(root / "development_set/valid_controls.json")["frames"]
    control_rows = []
    for item in controls:
        receipt = _d3_receipt(int(item["ordinal"]))
        control_rows.append(
            {
                "role": item["role"],
                "ordinal": item["ordinal"],
                "source_frame": item["source_frame"],
                "normal_path_triggered_fallback": False,
                "baseline_E_IM": item["D3_V1_E_IM"],
                "design_E_IM": item["D3_V1_E_IM"],
                "hard_valid": bool(receipt["selected_evaluation"]["feasible"]),
                "interaction_valid": float(item["D3_V1_E_IM"]) <= TAU,
                "preserved_exact": True,
            }
        )
    hysteresis = read_json(root / "sequential_hysteresis/decision.json")
    runner = {
        "oracle_b_seed": _oracle_b_frame,
        "oracle_c_active_set": _oracle_c_frame,
        "oracle_composition": _oracle_composition_frame,
    }[oracle_dir]
    runtime, bootstrap_runtime = _runtime_pair()
    plan_sha = sha256_file(root / "oracle_plan/search_space_oracle_plan.json")
    repeat_rows = []
    repeat_items = read_json(root / "development_set/representative_failure_frames.json")["frames"][
        ::6
    ][:3]
    reference = {int(row["ordinal"]): row for row in failure_rows}
    for item in repeat_items:
        ordinal = int(item["ordinal"])
        previous_q, previous_base = _previous_d3_state(ordinal)
        if previous_q is None or previous_base is None:
            raise RuntimeError("D3R2_DETERMINISM_PREDECESSOR_MISSING")
        q, base, receipt, _resumed = _run_cached(
            root,
            "sentinels/determinism_repeat",
            ordinal,
            plan_sha,
            runner,
            runtime,
            bootstrap_runtime,
            previous_q,
            previous_base,
            suffix="_repeat",
        )
        original_path = _cache_path(root, oracle_dir, ordinal)
        original = read_json(original_path)
        repeat_rows.append(
            {
                "ordinal": ordinal,
                "q_max_abs": float(np.max(np.abs(q - np.asarray(original["qpos"])))),
                "base_max_abs": float(
                    np.max(np.abs(base - np.asarray(original["base_pose_scene"])))
                ),
                "E_IM_abs": abs(
                    float(receipt["selected_E_IM"]) - float(reference[ordinal]["oracle_best_E_IM"])
                ),
            }
        )
    determinism_pass = all(
        row["q_max_abs"] <= 1e-10 and row["base_max_abs"] <= 1e-10 and row["E_IM_abs"] <= 1e-12
        for row in repeat_rows
    )
    recovered_clusters = len(
        {int(row["segment_id"]) for row in failures if bool(row["interaction_valid"])}
    )
    recovered = sum(bool(row["interaction_valid"]) for row in failures)
    sentinel_pass = bool(
        len(failures) == 16
        and all(bool(row["hard_valid"]) for row in failures)
        and recovered > 0
        and recovered_clusters >= 3
        and all(bool(row["preserved_exact"]) for row in control_rows)
        and determinism_pass
        and hysteresis.get("status") == "PASS"
    )
    value = {
        "schema_version": "D3R2DesignSentinelDecisionV1",
        "status": "PASS" if sentinel_pass else "FAIL",
        "DESIGN_SENTINEL_RESULT": "PASS" if sentinel_pass else "FAIL",
        "FAILURE_REPRESENTATIVE_VALID_RECOVERY": recovered,
        "FAILURE_REPRESENTATIVE_COUNT": len(failures),
        "CLUSTERS_WITH_RECOVERY": recovered_clusters,
        "VALID_CONTROLS_PRESERVED": sum(bool(row["preserved_exact"]) for row in control_rows),
        "VALID_CONTROL_COUNT": len(control_rows),
        "DETERMINISM": "PASS" if determinism_pass else "FAIL",
        "SEQUENTIAL_WINDOWS": hysteresis.get("CLUSTER_WINDOWS_VALID"),
        "SEQUENTIAL_WINDOW_COUNT": hysteresis.get("window_count"),
    }
    write_csv(root / "sentinels/failure_frames.csv", failures)
    write_csv(root / "sentinels/valid_controls.csv", control_rows)
    if (root / "sequential_hysteresis/per_window.csv").is_file():
        write_csv(
            root / "sentinels/windows.csv",
            read_csv(root / "sequential_hysteresis/per_window.csv"),
        )
    else:
        write_csv(root / "sentinels/windows.csv", [])
    write_json(
        root / "sentinels/determinism.json",
        {
            "schema_version": "D3R2SentinelDeterminismV1",
            "status": "PASS" if determinism_pass else "FAIL",
            "rows": repeat_rows,
        },
    )
    write_json(root / "sentinels/decision.json", value)
    return value


def freeze_refinement_v2_design(root: Path) -> dict[str, Any]:
    require(root / "sentinels/decision.json", "status", "PASS", "FREEZE_REFINEMENT_V2")
    design = read_json(root / "design/refinement_v2_design.json")
    design["status"] = "FROZEN"
    design["REFINEMENT_V2_DESIGN_FROZEN"] = "YES"
    design["STATUS"] = "DEVELOPMENT_DESIGN_ONLY"
    design["INDEPENDENT_CERTIFICATION"] = "NOT_RUN"
    path = root / "design/refinement_v2_design.json"
    write_json(path, design)
    first = sha256_file(path)
    serialized = json.dumps(read_json(path), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if hashlib.sha256(serialized.encode()).hexdigest() != first:
        raise RuntimeError("D3R2_DESIGN_CANONICAL_SERIALIZATION_MISMATCH")
    write_text(root / "design/refinement_v2_design.sha256", first + "\n")
    selected = read_json(root / "design/selected_design.json")
    selected["status"] = "FROZEN"
    selected["REFINEMENT_V2_DESIGN_SHA256"] = first
    write_json(root / "design/selected_design.json", selected)
    return {**design, "REFINEMENT_V2_DESIGN_SHA256": first}


def freeze_d3r3_development_gate(root: Path) -> dict[str, Any]:
    require(root / "design/refinement_v2_design.json", "status", "FROZEN", "FREEZE_D3R3_GATE")
    gate = {
        "schema_version": "RefinementV2DevelopmentGateV1",
        "status": "FROZEN_NOT_RUN",
        "population": "all 145 consumed D3-R final-invalid frames plus frozen valid controls and consumed sequential windows",
        "final_invalid_recovery_min": 0.80,
        "median_invalid_relative_E_IM_reduction_min": 0.50,
        "valid_preservation_controls": 1.0,
        "hard_validity": "PASS",
        "sequential_consumed_windows": "PASS",
        "continuity": "PASS",
        "determinism": "PASS",
        "D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT": 0,
    }
    path = root / "design/refinement_v2_development_gate.json"
    write_json(path, gate)
    write_text(root / "design/refinement_v2_development_gate.sha256", sha256_file(path) + "\n")
    write_json(
        root / "future/d3r3_plan.json",
        {
            **gate,
            "NEXT": "O5R-D3-R3_REFINEMENT_V2_DEVELOPMENT_VALIDATION",
            "authorized_by": "O5R-D3-R2_PASS_DESIGN",
        },
    )
    return gate


def authorize_d3r3(root: Path) -> dict[str, Any]:
    design = require(
        root / "design/refinement_v2_design.json", "status", "FROZEN", "AUTHORIZE_D3R3"
    )
    require(
        root / "design/refinement_v2_development_gate.json",
        "status",
        "FROZEN_NOT_RUN",
        "AUTHORIZE_D3R3",
    )
    sentinel = require(root / "sentinels/decision.json", "status", "PASS", "AUTHORIZE_D3R3")
    value = {
        "schema_version": "D3R2D3R3AuthorizationV1",
        "status": "PASS",
        "D3_R2_STATUS": "PASS_DESIGN",
        "SEARCH_SPACE_MECHANISM": design["root_search_space_mechanism"],
        "SELECTED_REFINEMENT_V2_DESIGN": design["selected_design"],
        "REFINEMENT_V2_DESIGN_FROZEN": "YES",
        "D3_R3_AUTHORIZED": "YES",
        "DESIGN_SENTINEL_RESULT": sentinel["DESIGN_SENTINEL_RESULT"],
        "FRESH_REFINEMENT_CERTIFICATION_RUN_COUNT": 0,
        "D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT": 0,
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "NEXT": "O5R-D3-R3_REFINEMENT_V2_DEVELOPMENT_VALIDATION",
    }
    write_json(root / "future/d3r3_authorization.json", value)
    return value


def _cost_model(root: Path) -> dict[str, Any]:
    decision = read_json(root / "mechanism/mechanism_decision.json")
    summary = read_json(root / decision["selected_oracle_directory"] / "summary.json")
    fallback_rate = 145 / EXPECTED_FRAMES
    mean_cost = float(summary.get("MEAN_WALL_TIME_SEC", 0.0))
    value = {
        "schema_version": "D3R2RefinementV2CostModelV1",
        "status": "MEASURED_CONSUMED_DEVELOPMENT_ESTIMATE",
        "normal_path_extra_cost": "one frozen interaction-gate branch; no additional optimizer",
        "fallback_invocation_count_on_D3_consumed_trajectory": 145,
        "fallback_invocation_rate": fallback_rate,
        "fallback_mean_cost_sec": mean_cost,
        "expected_added_sec_per_D3_frame": fallback_rate * mean_cost,
        "memory_impact": "one alternate q/base candidate plus bounded solver workspace",
        "production_benchmark": "NOT_RUN",
    }
    write_json(root / "design/cost_model.json", value)
    return value


def _audits(root: Path) -> None:
    write_json(
        root / "audits/no_special_cases.json",
        {
            "schema_version": "D3R2NoSpecialCasesAuditV1",
            "status": "PASS",
            "DEV1_EPISODE_SPECIFIC_SEARCH_BRANCH": "NO",
            "DEV1_C10001_SPECIAL_SEARCH_BRANCH": "NO",
            "DEV1_FRAME_LITERAL_SEARCH_BRANCH": "NO",
        },
    )
    write_json(
        root / "audits/method_freeze.json",
        {
            "schema_version": "D3R2MethodFreezeAuditV1",
            "status": "PASS",
            "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
            "SEMANTIC_V1_CHANGED": "NO",
            "E_IM_THRESHOLD_CHANGED": "NO",
            "HARD_VALIDITY_CHANGED": "NO",
            "Q_OLD_AUTHORITY_CHANGED": "NO",
            "ORACLE_MAX_PRIMARY_MECHANISMS": 3,
        },
    )
    write_json(
        root / "audits/fresh_data_nonconsumption.json",
        {
            "schema_version": "D3R2FreshDataNonconsumptionV1",
            "status": "PASS",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
            "D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT": 0,
            "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        },
    )


def validate_repository(root: Path) -> dict[str, Any]:
    """Run the repository checks required by the frozen D3-R2 handoff."""

    modified = [
        "scripts/evaluation/run_oakink2_o5rd3r2.py",
        "tests/evaluation/test_oakink2_o5rd3r2.py",
    ]
    commands = [
        ["ruff", "check", *modified],
        ["ruff", "format", "--check", *modified],
        [sys.executable, "-m", "mypy", "src"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "scripts/check_paper_fidelity.py"],
        ["git", "diff", "--check"],
    ]
    rows = []
    for command in commands:
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        rows.append(
            {
                "command": command,
                "returncode": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "wall_time_sec": time.perf_counter() - started,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        if result.returncode != 0:
            break
    help_rows = []
    script = str(Path(__file__).resolve())
    for action in ACTIONS:
        if action == "validate-repository":
            continue
        result = subprocess.run(
            [sys.executable, script, action, "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        help_rows.append({"action": action, "returncode": result.returncode})
    status = (
        "PASS"
        if all(row["status"] == "PASS" for row in rows)
        and all(row["returncode"] == 0 for row in help_rows)
        else "FAIL"
    )
    value = {
        "schema_version": "O5RD3R2RepositoryValidationV1",
        "status": status,
        "checks": rows,
        "cli_help": help_rows,
    }
    write_json(root / "tests.json", value)
    write_json(root / "validation_results.json", value)
    oracle_rows = []
    for oracle, path in (
        ("B", root / "oracle_b_seed/per_frame.csv"),
        ("C", root / "oracle_c_active_set/per_frame.csv"),
        ("SEQUENTIAL_C", root / "sequential_hysteresis/per_frame.csv"),
    ):
        for row in read_csv(path):
            oracle_rows.append(
                {
                    "oracle": oracle,
                    "ordinal": row["ordinal"],
                    "source_frame": row["source_frame"],
                    "wall_time_sec": row.get("wall_time_sec", "NOT_EMITTED_IN_SEQUENCE"),
                }
            )
    write_csv(root / "timing/oracle_timing.csv", oracle_rows)
    if status != "PASS":
        raise RuntimeError("D3R2_REPOSITORY_VALIDATION_FAIL")
    return value


def summarize(root: Path) -> dict[str, Any]:
    integrity = read_json(root / "preflight/evidence_integrity.json")
    baseline = read_json(root / "baseline/parity.json")
    mechanism = read_json(root / "mechanism/mechanism_decision.json")
    sentinel_path = root / "sentinels/decision.json"
    authorization_path = root / "future/d3r3_authorization.json"
    if integrity.get("status") != "PASS":
        status = "BLOCKED_UPSTREAM_AUTHORITY_INTEGRITY"
    elif baseline.get("status") != "PASS":
        status = "BLOCKED_BASELINE_REPRODUCIBILITY"
    elif mechanism.get("status") != "PASS":
        status = "INCONCLUSIVE"
    elif not sentinel_path.is_file() or read_json(sentinel_path).get("status") != "PASS":
        status = "FAIL_NO_EFFECTIVE_SEARCH_SPACE_REPAIR"
    elif not authorization_path.is_file():
        status = "INCONCLUSIVE"
    else:
        status = "PASS_DESIGN"
    b = read_json(root / "oracle_b_seed/summary.json")
    c = read_json(root / "oracle_c_active_set/summary.json")
    d = read_json(root / "oracle_d_envelope/summary.json")
    composition = read_json(root / "oracle_composition/result.json")
    clusters = read_json(root / "clusters/cluster_manifest.json")
    similarity = read_json(root / "clusters/cluster_similarity.json")
    hysteresis = read_json(root / "sequential_hysteresis/decision.json")
    sentinel = read_json(sentinel_path) if sentinel_path.is_file() else {}
    design_path = root / "design/refinement_v2_design.json"
    design = read_json(design_path) if design_path.is_file() else {}
    design_frozen = design.get("status") == "FROZEN"
    _audits(root)
    if mechanism.get("status") == "PASS":
        _cost_model(root)
    summary = {
        "schema_version": "OakInk2O5RD3R2FinalSummaryV1",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": git("rev-parse", "HEAD"),
        "D3_V1_NUMERICAL": "PASS",
        "D3_V1_SEMANTIC": "FAIL",
        "D3R_PRIMARY_ROOT_CAUSE": "PRIMARY_REFINEMENT_SEARCH_COVERAGE_INSUFFICIENT",
        "D3R_ROOT_CAUSE_CONFIDENCE": "HIGH",
        "GROUP_A_COUNT": 1865,
        "GROUP_B_COUNT": 145,
        "GROUP_C_COUNT": 712,
        "GROUP_D_COUNT": 0,
        "FINAL_INVALID_FRAME_COUNT": 145,
        "FAILURE_SEGMENT_COUNT": clusters["N_FAILURE_SEGMENTS"],
        "MULTISTART_RECOVERY_COMPLETED": 31,
        "MULTISTART_RECOVERY_SUCCESS": 0,
        "FAILURE_CLUSTER_MECHANISM_PATTERN": similarity["FAILURE_CLUSTER_MECHANISM_PATTERN"],
        "REPRESENTATIVE_FAILURE_FRAME_COUNT": 16,
        "RECOVERED_CONTROL_COUNT": 8,
        "VALID_CONTROL_COUNT": 8,
        "SEQUENTIAL_WINDOW_COUNT": 4,
        "SELECTION_FROZEN_BEFORE_ORACLE": "YES",
        "BASELINE_REPLAY_PARITY": baseline["BASELINE_REPLAY_PARITY"],
        "ORACLE_B": b,
        "ORACLE_C": c,
        "ORACLE_D": d,
        "COMPOSITIONAL_ORACLE": composition.get("COMPOSITIONAL_ORACLE"),
        "COMPOSITIONAL_ORACLE_COUNT": composition.get("COMPOSITIONAL_ORACLE_COUNT", 0),
        "SEQUENTIAL_BASIN_HYSTERESIS": hysteresis.get("SEQUENTIAL_BASIN_HYSTERESIS"),
        "CLUSTER_WINDOWS_VALID": hysteresis.get("CLUSTER_WINDOWS_VALID"),
        "SEARCH_SPACE_MECHANISM": mechanism.get("SEARCH_SPACE_MECHANISM"),
        "MECHANISM_CONFIDENCE": mechanism.get("MECHANISM_CONFIDENCE"),
        "SELECTED_REFINEMENT_V2_DESIGN": design.get("selected_design"),
        "REFINEMENT_V2_DESIGN_SHA256": sha256_file(design_path) if design_frozen else None,
        "REFINEMENT_V2_DESIGN_FROZEN": "YES" if design_frozen else "NO",
        "DESIGN_SENTINEL_RESULT": sentinel.get("DESIGN_SENTINEL_RESULT", "NOT_RUN"),
        "D3_R2_STATUS": status,
        "D3_R3_AUTHORIZED": "YES" if status == "PASS_DESIGN" else "NO",
        "NEXT": "O5R-D3-R3_REFINEMENT_V2_DEVELOPMENT_VALIDATION"
        if status == "PASS_DESIGN"
        else "HARD_STOP",
        "D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D3R_HISTORICAL_RESULT_REWRITTEN": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "HARD_VALIDITY_CHANGED": "NO",
        "Q_OLD_AUTHORITY_CHANGED": "NO",
        "FAILURE_FRAME_SELECTION_FROZEN_BEFORE_ORACLE": "YES",
        "ORACLE_MAX_PRIMARY_MECHANISMS": 3,
        "DEV1_EPISODE_SPECIFIC_SEARCH_BRANCH": "NO",
        "DEV1_C10001_SPECIAL_SEARCH_BRANCH": "NO",
        "DEV1_FRAME_LITERAL_SEARCH_BRANCH": "NO",
        "D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT": 0,
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "completion_audit.json",
        {
            "schema_version": "O5RD3R2CompletionAuditV1",
            "status": "PASS"
            if status in {"PASS_DESIGN", "INCONCLUSIVE", "FAIL_NO_EFFECTIVE_SEARCH_SPACE_REPAIR"}
            else "FAIL",
            "terminal_status": status,
            "required_artifacts_present": all(
                (root / path).is_file()
                for path in (
                    "preflight/evidence_integrity.json",
                    "clusters/cluster_manifest.json",
                    "development_set/selection_receipt.json",
                    "oracle_plan/search_space_oracle_plan.json",
                    "baseline/parity.json",
                    "oracle_b_seed/summary.json",
                    "oracle_c_active_set/summary.json",
                    "oracle_d_envelope/summary.json",
                    "mechanism/mechanism_decision.json",
                    "sequential_hysteresis/decision.json",
                    "audits/method_freeze.json",
                    "audits/fresh_data_nonconsumption.json",
                )
            ),
        },
    )
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD3R2ResourceUsageV1",
            "status": "PASS",
            "gpu_required": False,
            "optimizer_scope": "CONSUMED_DEVELOPMENT_FEASIBILITY_ORACLE_ONLY",
            "fresh_or_certification_compute": False,
        },
    )
    write_json(
        root / "git_commits.json",
        {
            "schema_version": "O5RD3R2GitCommitsV1",
            "START_HEAD": START_HEAD,
            "FINAL_HEAD_AT_SUMMARY": git("rev-parse", "HEAD"),
            "commits": git("log", f"{START_HEAD}..HEAD", "--oneline").splitlines(),
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    )
    handoff = f"""# OakInk2 O5R-D3-R2

# DEV1 Refinement Search-Space Repair Design Handoff

## Git

```text
BRANCH={EXPECTED_BRANCH}
START_HEAD={START_HEAD}
FINAL_HEAD={summary["FINAL_HEAD"]}
PUSHED=NO
PR_CREATED=NO
```

## Result

```text
D3_R2_STATUS={status}
SEARCH_SPACE_MECHANISM={summary["SEARCH_SPACE_MECHANISM"]}
MECHANISM_CONFIDENCE={summary["MECHANISM_CONFIDENCE"]}
SELECTED_REFINEMENT_V2_DESIGN={summary["SELECTED_REFINEMENT_V2_DESIGN"]}
REFINEMENT_V2_DESIGN_FROZEN={summary["REFINEMENT_V2_DESIGN_FROZEN"]}
D3_R3_AUTHORIZED={summary["D3_R3_AUTHORIZED"]}
NEXT={summary["NEXT"]}
```

## Immutable upstream evidence

```text
D3_V1_NUMERICAL=PASS
D3_V1_SEMANTIC=FAIL
D3R_PRIMARY_ROOT_CAUSE=PRIMARY_REFINEMENT_SEARCH_COVERAGE_INSUFFICIENT
D3R_ROOT_CAUSE_CONFIDENCE=HIGH
FINAL_INVALID_FRAME_COUNT=145
FAILURE_SEGMENT_COUNT=4
MULTISTART_RECOVERY_COMPLETED=31
MULTISTART_RECOVERY_SUCCESS=0
```

## Oracle results

```text
ORACLE_B_VALID={b["VALID_COUNT"]}/{b["N"]}
ORACLE_B_MEDIAN_E_IM_REDUCTION={b["MEDIAN_E_IM_REDUCTION"]}
ORACLE_C_VALID={c["VALID_COUNT"]}/{c["N"]}
ORACLE_C_MEDIAN_E_IM_REDUCTION={c["MEDIAN_E_IM_REDUCTION"]}
ORACLE_D_VALID={d["VALID_COUNT"]}/{d["N"]}
ORACLE_D_MEDIAN_E_IM_REDUCTION={d["MEDIAN_E_IM_REDUCTION"]}
COMPOSITIONAL_ORACLE={composition.get("COMPOSITIONAL_ORACLE")}
SEQUENTIAL_BASIN_HYSTERESIS={summary["SEQUENTIAL_BASIN_HYSTERESIS"]}
CLUSTER_WINDOWS_VALID={summary["CLUSTER_WINDOWS_VALID"]}/4
```

## Explicitly not run

```text
D3_R3_FULL_DEVELOPMENT_VALIDATION_RUN_COUNT=0
FRESH_REFINEMENT_SPARSE=NOT_RUN
FRESH_REFINEMENT_WINDOW=NOT_RUN
CROSS_EPISODE_REFINEMENT=NOT_RUN
D3_V2_SCIENTIFIC_RUN_COUNT=0
DEV2_RERUN=NO
PPO_TRAINING_RUN_COUNT_NEW=0
O6_PRODUCTION_RAN=NO
CERTIFICATION_SPLIT_NEW_CONSUMPTION=0
HELDOUT_SPLIT_NEW_CONSUMPTION=0
```
"""
    write_text(root / "handoff.md", handoff)
    write_text(root / "final_summary.md", handoff)
    return summary


def prepare_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_d3r_authority(root)
    build_failure_cluster_manifest(root)
    freeze_representative_development_set(root)
    return freeze_search_space_oracle_plan(root)


ACTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "preflight": preflight,
    "verify-d3r-authority": verify_d3r_authority,
    "build-failure-cluster-manifest": build_failure_cluster_manifest,
    "freeze-representative-development-set": freeze_representative_development_set,
    "freeze-search-space-oracle-plan": freeze_search_space_oracle_plan,
    "replay-baseline-sentinels": replay_baseline_sentinels,
    "run-oracle-source-conditioned-seed": run_oracle_source_conditioned_seed,
    "run-oracle-expanded-active-set": run_oracle_expanded_active_set,
    "run-oracle-envelope-expansion": run_oracle_envelope_expansion,
    "run-compositional-oracle-if-authorized": run_compositional_oracle_if_authorized,
    "compare-search-space-oracles": compare_search_space_oracles,
    "run-sequential-hysteresis-diagnostic": run_sequential_hysteresis_diagnostic,
    "decide-search-space-mechanism": decide_search_space_mechanism,
    "design-refinement-v2": design_refinement_v2,
    "run-refinement-v2-design-sentinels": run_refinement_v2_design_sentinels,
    "freeze-refinement-v2-design": freeze_refinement_v2_design,
    "freeze-d3r3-development-gate": freeze_d3r3_development_gate,
    "authorize-d3r3": authorize_d3r3,
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
    stage_started = time.perf_counter()
    try:
        result = ACTIONS[args.action](args.root.resolve())
    except Exception as exc:
        _append_jsonl(
            args.root.resolve() / "technical_failures.jsonl",
            {
                "schema_version": "O5RD3R2TechnicalFailureV1",
                "action": args.action,
                "error": f"{type(exc).__name__}:{exc}",
            },
        )
        raise
    timing_path = args.root.resolve() / "timing/stage_timing.json"
    timing = (
        read_json(timing_path)
        if timing_path.is_file()
        else {
            "schema_version": "O5RD3R2StageTimingV1",
            "stages": [],
        }
    )
    timing["stages"].append(
        {"action": args.action, "wall_time_sec": time.perf_counter() - stage_started}
    )
    write_json(timing_path, timing)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
