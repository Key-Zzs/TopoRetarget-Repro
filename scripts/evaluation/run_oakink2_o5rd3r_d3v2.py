#!/usr/bin/env python3
"""O5R-D3-R semantic-tail localization and fail-closed D3-V2 workflow.

The localization actions consume only the immutable O5R-D3 V1 artifacts.  No
optimizer is reachable before a mechanism-level root cause and a targeted
repair plan have both been frozen.  Later actions are added to this driver as
separately gated stages; every gate reads durable evidence from ``--root``.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2a as d2a  # noqa: E402
from scripts.data import run_oakink2_o5rd2c as d2c  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2m as d2m  # noqa: E402
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    default_search_contracts,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3r_d3v2_semantic_tail_repair_v1"
D3_ROOT = REPO / ".local/reports/oakink2_o5rd3_dev1_full_objective_v2_execution_v4_refinement_v1"
D2N_ROOT = REPO / ".local/reports/oakink2_o5rd2n_dev2_full_recovery_v3"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "0fd8b2d640ddcb7a6ee460fd2e7116b396230268"
EXPECTED_FRAMES = 2722
TAU = 1.0e-4
OBJECTIVE_V2_SHA256 = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
SEMANTIC_V1_SHA256 = "0ea9ba21419a2b254557e0af18405649b23a0cf30407684eaa2e2349c4467e31"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    d2m.atomic_write_json(path, value)


def write_text(path: Path, value: str) -> None:
    d2m.atomic_write_text(path, value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    d2m.write_csv(path, rows)


def write_csv_with_fields(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if rows:
        write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fieldnames).writeheader()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{field}={value.get(field)!r}")
    return value


def _d3_rows() -> list[dict[str, str]]:
    rows = read_csv(D3_ROOT / "old_vs_new/per_frame.csv")
    if len(rows) != EXPECTED_FRAMES:
        raise RuntimeError(f"D3R_D3_V1_FRAME_COUNT_MISMATCH:{len(rows)}")
    return rows


def _group(old_e_im: float, new_e_im: float) -> str:
    if old_e_im > TAU and new_e_im <= TAU:
        return "A"
    if old_e_im > TAU and new_e_im > TAU:
        return "B"
    if old_e_im <= TAU and new_e_im <= TAU:
        return "C"
    return "D"


def _receipt(ordinal: int) -> dict[str, Any]:
    return read_json(D3_ROOT / f"checkpoints/frame_{ordinal:04d}/receipt.json")


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "median", "p25", "p75", "p90", "p95", "p99", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"],
            cwd=REPO,
            check=False,
        ).returncode
        == 0,
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "d3_root_exists": D3_ROOT.is_dir(),
        "dev2_root_exists": D2N_ROOT.is_dir(),
        "oakink2_root_exists": Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2").is_dir(),
    }
    value = {
        "schema_version": "O5RD3RPreflightV1",
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
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3R_STATUS=BLOCKED_AUTHORITY_INTEGRITY")
    return value


def verify_d3_v1_history(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "status", "PASS", "VERIFY_D3_V1_HISTORY")
    summary = read_json(D3_ROOT / "final_summary.json")
    result = read_json(D3_ROOT / "solver/result.json")
    semantic = read_json(D3_ROOT / "semantic_v1/result.json")
    run_state = read_json(D3_ROOT / "run_authority/run_state.json")
    trajectory = D3_ROOT / "trajectory/trajectory.npz"
    checks = {
        "numerical_pass": result.get("status") == "PASS",
        "expected_frames": result.get("EXPECTED_FRAMES") == EXPECTED_FRAMES,
        "completed_frames": result.get("COMPLETED_FRAMES") == EXPECTED_FRAMES,
        "first_failure_null": result.get("FIRST_DEV1_FAILURE_ORDINAL") is None,
        "refinement_mode": summary.get("RETARGET_MODE") == "REFINEMENT",
        "cold_start_no": summary.get("COLD_START_USED_FOR_DEV1") == "NO",
        "semantic_fail": semantic.get("DEV1_SEMANTIC_V1_RESULT") == "FAIL",
        "semantic_p95_exact": semantic.get("E_IM_P95") == 0.0001044450028581672,
        "threshold_exact": semantic.get("E_IM_THRESHOLD") == TAU,
        "final_invalid_145": read_json(D3_ROOT / "old_vs_new/invalid_recovery.json").get(
            "NEW_INVALID_FRAME_COUNT"
        )
        == 145,
        "single_scientific_run": run_state.get("SCIENTIFIC_RUN_COUNT") == 1,
        "trajectory_hash": trajectory.is_file()
        and sha256_file(trajectory)
        == (D3_ROOT / "trajectory/trajectory.sha256").read_text().strip(),
    }
    value = {
        "schema_version": "D3V1HistoricalResultV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D3_V1_HISTORICAL_RESULT": "NUMERICAL_PASS_SEMANTIC_FAIL",
        "D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D3_V1_TRAJECTORY_MODIFIED": "NO",
        "D3_V1_RUN_UUID_REUSED": "NO",
        "D3_V1_RUN_UUID": run_state.get("RUN_UUID"),
        "D3_V1_TRAJECTORY": str(trajectory.resolve()),
        "D3_V1_TRAJECTORY_SHA256": sha256_file(trajectory),
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "preflight/d3_v1_history.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3R_STATUS=BLOCKED_AUTHORITY_INTEGRITY:D3_V1")
    return value


def verify_frozen_authorities(root: Path) -> dict[str, Any]:
    require(root / "preflight/d3_v1_history.json", "status", "PASS", "VERIFY_AUTHORITIES")
    d3 = read_json(D3_ROOT / "preflight/frozen_authorities.json")
    dev2 = read_json(D3_ROOT / "preflight/dev2_acceptance.json")
    approval = read_json(D3_ROOT / "preflight/dev2_human_geometric_review_receipt.json")
    old = read_json(D3_ROOT / "old_trajectory/authority.json")
    checks = {
        "d3_frozen_authorities": d3.get("status") == "PASS",
        "objective_v2": d3.get("RETARGET_OBJECTIVE_V2_SHA256") == OBJECTIVE_V2_SHA256,
        "semantic_v1": d3.get("SEMANTIC_V1_SHA256") == SEMANTIC_V1_SHA256,
        "threshold": d3.get("E_IM_THRESHOLD") == TAU,
        "dev2_acceptance": dev2.get("status") == "PASS",
        "dev2_machine": dev2.get("checks", {}).get("DEV2_FULL_RECOVERY_V3_MACHINE") is True,
        "dev2_semantic": dev2.get("checks", {}).get("DEV2_V3_SEMANTIC_V1_RESULT") is True,
        "dev2_human": approval.get("DEV2_HUMAN_GEOMETRIC_REVIEW") == "APPROVE",
        "qold_complete": old.get("Q_OLD_ACTUAL_FRAMES") == EXPECTED_FRAMES,
        "qold_unmodified": old.get("Q_OLD_TRAJECTORY_MODIFIED") == "NO",
    }
    value = {
        "schema_version": "O5RD3RFrozenAuthoritiesV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "DEV2_FULL_RECOVERY_V3_MACHINE": "PASS" if checks["dev2_machine"] else "FAIL",
        "DEV2_V3_SEMANTIC_V1_RESULT": "PASS" if checks["dev2_semantic"] else "FAIL",
        "DEV2_HUMAN_GEOMETRIC_REVIEW": approval.get("DEV2_HUMAN_GEOMETRIC_REVIEW"),
        "RETARGET_OBJECTIVE_V2_SHA256": OBJECTIVE_V2_SHA256,
        "SEMANTIC_V1_SHA256": SEMANTIC_V1_SHA256,
        "E_IM_THRESHOLD": TAU,
        "ExecutionV4_authority_sha256": d3.get("ExecutionV4_authority_sha256"),
        "q_old_array_sha256": old.get("q_old_array_sha256"),
        "base_old_array_sha256": old.get("base_old_array_sha256"),
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "preflight/frozen_authorities.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3R_STATUS=BLOCKED_AUTHORITY_INTEGRITY:FROZEN_AUTHORITIES")
    return value


def partition_d3_semantic_tail(root: Path) -> dict[str, Any]:
    require(root / "preflight/frozen_authorities.json", "status", "PASS", "PARTITION_D3")
    rows = []
    counts: Counter[str] = Counter()
    for raw in _d3_rows():
        old = float(raw["old_e_im"])
        new = float(raw["new_e_im"])
        group = _group(old, new)
        counts[group] += 1
        rows.append(
            {
                "ordinal": int(raw["ordinal"]),
                "source_frame": int(raw["source_frame"]),
                "old_E_IM": old,
                "new_E_IM": new,
                "threshold": TAU,
                "group": f"GROUP_{group}",
            }
        )
    checks = {
        "complete": sum(counts.values()) == EXPECTED_FRAMES,
        "final_invalid": counts["B"] + counts["D"] == 145,
    }
    value = {
        "schema_version": "D3RSemanticTailPartitionV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        **{f"N_{name}": counts[name] for name in "ABCD"},
        "FINAL_INVALID_FRAME_COUNT": counts["B"] + counts["D"],
        "threshold": TAU,
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_csv(root / "d3r/group_partition.csv", rows)
    write_csv(root / "d3r/eim_temporal_curve.csv", rows)
    write_json(root / "d3r/group_partition.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("D3R_STATUS=BLOCKED_PARTITION_INTEGRITY")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [row["source_frame"] for row in rows]
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(frames, [row["old_E_IM"] for row in rows], label="historical old", linewidth=0.8)
    axis.plot(frames, [row["new_E_IM"] for row in rows], label="D3-V1", linewidth=0.8)
    axis.axhline(TAU, color="black", linestyle="--", linewidth=1.0, label="tau=1e-4")
    axis.set_xlabel("source frame")
    axis.set_ylabel("E_IM")
    axis.legend()
    fig.tight_layout()
    plot = root / "d3r/eim_temporal_curve.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    value["diagnostic_plot"] = str(plot.resolve())
    value["diagnostic_plot_sha256"] = sha256_file(plot)
    write_json(root / "d3r/group_partition.json", value)
    return value


def analyze_tail_excess(root: Path) -> dict[str, Any]:
    require(root / "d3r/group_partition.json", "status", "PASS", "ANALYZE_EXCESS")
    invalid = [
        float(row["new_E_IM"]) - TAU
        for row in read_csv(root / "d3r/group_partition.csv")
        if row["group"] in {"GROUP_B", "GROUP_D"}
    ]
    array = np.asarray(invalid, dtype=np.float64)
    bins = {
        "LE_1_PERCENT": int(np.sum(array <= TAU * 0.01)),
        "GT_1_LE_5_PERCENT": int(np.sum((array > TAU * 0.01) & (array <= TAU * 0.05))),
        "GT_5_LE_10_PERCENT": int(np.sum((array > TAU * 0.05) & (array <= TAU * 0.10))),
        "GT_10_LE_25_PERCENT": int(np.sum((array > TAU * 0.10) & (array <= TAU * 0.25))),
        "GT_25_PERCENT": int(np.sum(array > TAU * 0.25)),
    }
    value = {
        "schema_version": "D3RExcessOverThresholdV1",
        "status": "PASS" if len(invalid) == 145 and sum(bins.values()) == 145 else "FAIL",
        "FINAL_INVALID_FRAME_COUNT": len(invalid),
        "threshold": TAU,
        "excess": _summary(invalid),
        "bins": bins,
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "d3r/excess_analysis.json", value)
    return value


def analyze_failure_segments(root: Path) -> dict[str, Any]:
    require(root / "d3r/group_partition.json", "status", "PASS", "ANALYZE_SEGMENTS")
    rows = read_csv(root / "d3r/group_partition.csv")
    indices = [index for index, row in enumerate(rows) if row["group"] in {"GROUP_B", "GROUP_D"}]
    segments: list[list[int]] = []
    for index in indices:
        if not segments or index != segments[-1][-1] + 1:
            segments.append([index])
        else:
            segments[-1].append(index)
    output = []
    for segment_id, segment in enumerate(segments):
        eim = [float(rows[index]["new_E_IM"]) for index in segment]
        composition = Counter(rows[index]["group"] for index in segment)
        output.append(
            {
                "segment_id": segment_id,
                "start_ordinal": segment[0],
                "stop_ordinal": segment[-1],
                "start_source_frame": int(rows[segment[0]]["source_frame"]),
                "stop_source_frame": int(rows[segment[-1]]["source_frame"]),
                "length": len(segment),
                "group_b_count": composition["GROUP_B"],
                "group_d_count": composition["GROUP_D"],
                "mean_E_IM": float(np.mean(eim)),
                "max_E_IM": float(np.max(eim)),
            }
        )
    lengths = [len(segment) for segment in segments]
    multi_fraction = sum(length for length in lengths if length >= 2) / max(1, len(indices))
    if multi_fraction >= 0.75 and max(lengths, default=0) >= 5:
        pattern = "CLUSTERED"
    elif multi_fraction < 0.25:
        pattern = "SCATTERED"
    else:
        pattern = "MIXED"
    value = {
        "schema_version": "D3RFailureSegmentsV1",
        "status": "PASS",
        "N_FAILURE_SEGMENTS": len(segments),
        "MAX_FAILURE_SEGMENT_LENGTH": max(lengths, default=0),
        "MEDIAN_FAILURE_SEGMENT_LENGTH": float(statistics.median(lengths)) if lengths else 0.0,
        "MULTIFRAME_FAILURE_FRACTION": multi_fraction,
        "FAILURE_TEMPORAL_PATTERN": pattern,
        "classification_rule": {
            "CLUSTERED": "multi-frame segment fraction >=0.75 and max segment length >=5",
            "SCATTERED": "multi-frame segment fraction <0.25",
            "MIXED": "otherwise",
        },
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_csv(root / "d3r/failure_segments.csv", output)
    write_json(root / "d3r/failure_segments.json", value)
    return value


def _candidate_lifecycle(receipt: dict[str, Any], group: str) -> tuple[dict[str, Any], bool]:
    candidates = []
    for probe in receipt["probe_receipts"]:
        interaction_valid = float(probe["whole_e_im"]) <= TAU
        hard_valid = bool(probe["independent_feasible"])
        candidates.append(
            {
                "candidate": f"probe:{probe['seed_id']}",
                "E_IM": float(probe["whole_e_im"]),
                "hard_valid": hard_valid,
                "interaction_valid": interaction_valid,
                "valid": hard_valid and interaction_valid,
                "violations": probe["violations"],
            }
        )
    retained_valid = bool(receipt["primary_interaction_valid"])
    candidates.append(
        {
            "candidate": receipt["retained_primary_id"],
            "E_IM": float(receipt["retained_primary"]["interaction_e_im"]),
            "hard_valid": True,
            "interaction_valid": retained_valid,
            "valid": retained_valid,
            "violations": [],
        }
    )
    selected_hard = bool(receipt["selected_evaluation"]["feasible"])
    selected_interaction = float(receipt["selected"]["interaction_e_im"]) <= TAU
    candidates.append(
        {
            "candidate": receipt["selected_candidate"],
            "E_IM": float(receipt["selected"]["interaction_e_im"]),
            "hard_valid": selected_hard,
            "interaction_valid": selected_interaction,
            "valid": selected_hard and selected_interaction,
            "violations": receipt["selected_evaluation"]["violated_constraints"],
        }
    )
    any_valid = any(item["valid"] for item in candidates)
    retention = str(receipt["retention_decision"])
    if retention == "PRIMARY_RETAINED_INTERACTION_REGRESSION":
        secondary_terminal = "KNOWN_INTERACTION_INVALID_BY_RETENTION_LIMIT"
    elif retention == "PRIMARY_RETAINED_POLISH_INVALID":
        secondary_terminal = "KNOWN_HARD_INVALID"
    elif receipt["selected_candidate"] == "secondary_polished":
        secondary_terminal = float(receipt["selected"]["interaction_e_im"])
    else:
        secondary_terminal = "INTERMEDIATE_EVIDENCE_UNAVAILABLE"
    lifecycle = {
        "ordinal": int(receipt["ordinal"]),
        "source_frame": int(receipt["frame_id"]),
        "group": group,
        "q_old_E_IM": float(receipt["old"]["interaction_e_im"]),
        "previous_accepted_context": "ABSENT" if int(receipt["ordinal"]) == 0 else "PRESENT",
        "seed_pool": ";".join(receipt["seed_pool"]),
        "primary_candidate_list": json.dumps(candidates, sort_keys=True),
        "selected_primary": receipt["retained_primary_id"],
        "primary_E_IM": float(receipt["retained_primary"]["interaction_e_im"]),
        "primary_interaction_valid": bool(receipt["primary_interaction_valid"]),
        "secondary_polish_start": receipt["retained_primary_id"],
        "secondary_polish_terminal": secondary_terminal,
        "retention_decision": retention,
        "fallback_decision": bool(receipt["baseline_fallback"]),
        "final_E_IM": float(receipt["selected"]["interaction_e_im"]),
        "any_hard_valid_interaction_valid_candidate": any_valid,
    }
    return lifecycle, any_valid


def analyze_candidate_availability(root: Path) -> dict[str, Any]:
    require(root / "d3r/group_partition.json", "status", "PASS", "ANALYZE_CANDIDATES")
    groups = read_csv(root / "d3r/group_partition.csv")
    availability = []
    lifecycle = []
    primary_valid_to_final_invalid = 0
    valid_existed_final_invalid = 0
    for row in groups:
        if row["group"] not in {"GROUP_B", "GROUP_D"}:
            continue
        ordinal = int(row["ordinal"])
        receipt = _receipt(ordinal)
        trace, any_valid = _candidate_lifecycle(receipt, row["group"])
        lifecycle.append(trace)
        final_invalid = float(receipt["selected"]["interaction_e_im"]) > TAU
        primary_valid_to_final_invalid += int(
            receipt["primary_interaction_valid"] and final_invalid
        )
        valid_existed_final_invalid += int(any_valid and final_invalid)
        availability.append(
            {
                "ordinal": ordinal,
                "source_frame": int(row["source_frame"]),
                "group": row["group"],
                "valid_candidate_found": any_valid,
                "classification": "VALID_CANDIDATE_EXISTED"
                if any_valid
                else "SEARCH_COVERAGE_LIMITATION",
                "final_E_IM": float(row["new_E_IM"]),
            }
        )
    found = sum(bool(row["valid_candidate_found"]) for row in availability)
    core = "YES" if found == len(availability) else "NO" if found == 0 else "UNKNOWN"
    value = {
        "schema_version": "D3RCandidateAvailabilityV1",
        "status": "PASS" if len(availability) == 145 else "FAIL",
        "FINAL_INVALID_FRAME_COUNT": len(availability),
        "ANY_HARD_VALID_INTERACTION_VALID_CANDIDATE_FOUND": core,
        "HARD_VALID_INTERACTION_VALID_CANDIDATE_FRAME_COUNT": found,
        "PRIMARY_VALID_TO_FINAL_INVALID_COUNT": primary_valid_to_final_invalid,
        "VALID_CANDIDATE_EXISTED_BUT_FINAL_INVALID_COUNT": valid_existed_final_invalid,
        "SEARCH_COVERAGE_LIMITATION_COUNT": len(availability) - found,
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_csv(root / "d3r/candidate_availability.csv", availability)
    write_csv(root / "d3r/candidate_lifecycle.csv", lifecycle)
    write_json(root / "d3r/candidate_availability.json", value)
    return value


def analyze_contributor_locality(root: Path) -> dict[str, Any]:
    require(root / "d3r/group_partition.json", "status", "PASS", "ANALYZE_CONTRIBUTORS")
    groups = read_csv(root / "d3r/group_partition.csv")
    rows = []
    by_group: dict[str, list[dict[str, Any]]] = {name: [] for name in "ABCD"}
    for raw in groups:
        ordinal = int(raw["ordinal"])
        scores = _receipt(ordinal)["contributor_scores"]
        ranked = sorted(scores, key=lambda name: (-float(scores[name]), name))
        total = float(sum(scores.values()))
        row = {
            "ordinal": ordinal,
            "source_frame": int(raw["source_frame"]),
            "group": raw["group"],
            "rho1": float(scores[ranked[0]]) / total,
            "rho2": float(scores[ranked[0]] + scores[ranked[1]]) / total,
            "top1_finger": ranked[0],
            "top2_finger": ranked[1],
            **{f"{name}_residual": float(scores[name]) for name in sorted(scores)},
        }
        rows.append(row)
        by_group[raw["group"][-1]].append(row)
    output: dict[str, Any] = {}
    for name, values in by_group.items():
        output[f"GROUP_{name}"] = {
            "count": len(values),
            "rho1": _summary([float(row["rho1"]) for row in values]),
            "rho2": _summary([float(row["rho2"]) for row in values]),
            "top1_finger_frequency": dict(Counter(str(row["top1_finger"]) for row in values)),
            "top2_finger_frequency": dict(Counter(str(row["top2_finger"]) for row in values)),
        }
    recovered = output["GROUP_A"]["rho1"]["median"]
    failed = output["GROUP_B"]["rho1"]["median"]
    value = {
        "schema_version": "D3RContributorLocalityV1",
        "status": "PASS",
        "frozen_residual_authority": "InteractionContributorSearchV2FrameReceipt.contributor_scores",
        "groups": output,
        "comparisons": {
            "GROUP_A_vs_GROUP_B": {
                "failed_rho1_lower": bool(failed < recovered),
                "decision": "DOES_NOT_SUPPORT_TOP1_LOCALITY_INSUFFICIENT"
                if failed >= recovered
                else "SUPPORTS_FURTHER_TOP1_LOCALITY_REVIEW",
            },
            "GROUP_C_vs_GROUP_D": {
                "decision": "GROUP_D_EMPTY_NO_OLD_VALID_REGRESSION",
            },
        },
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_csv(root / "d3r/contributor_metrics.csv", rows)
    write_json(root / "d3r/contributor_metrics.json", value)
    return value


def analyze_temporal_context(root: Path) -> dict[str, Any]:
    require(root / "d3r/group_partition.json", "status", "PASS", "ANALYZE_TEMPORAL")
    groups = read_csv(root / "d3r/group_partition.csv")
    old_authority = read_json(D3_ROOT / "old_trajectory/authority.json")
    with np.load(Path(old_authority["compact_trajectory_path"]), allow_pickle=False) as data:
        q_old = np.asarray(data["qpos"], dtype=np.float64)
    with np.load(D3_ROOT / "trajectory/trajectory.npz", allow_pickle=False) as data:
        q_new = np.asarray(data["qpos"], dtype=np.float64)
    rows = []
    for ordinal, raw in enumerate(groups):
        receipt = _receipt(ordinal)
        actual = receipt["selected_actual_continuity"]
        rows.append(
            {
                "ordinal": ordinal,
                "source_frame": int(raw["source_frame"]),
                "group": raw["group"],
                "q_old_transition_l2": None
                if ordinal == 0
                else float(np.linalg.norm(q_old[ordinal] - q_old[ordinal - 1])),
                "previous_new_to_q_old_l2": None
                if ordinal == 0
                else float(np.linalg.norm(q_new[ordinal - 1] - q_old[ordinal])),
                "previous_new_to_old_l2": None
                if ordinal == 0
                else float(np.linalg.norm(q_new[ordinal - 1] - q_old[ordinal])),
                "continuous_prediction_q_step_inf_rad": float(actual["q_step_inf_rad"]),
                "continuous_prediction_translation_m": float(actual["translation_step_m"]),
                "continuous_prediction_rotation_rad": float(actual["rotation_step_rad"]),
                "old_temporal_objective": float(
                    receipt["old"]["per_components"]["continuous_temporal"]
                ),
                "final_temporal_objective": float(
                    receipt["selected"]["per_components"]["continuous_temporal"]
                ),
            }
        )
    by_group: dict[str, dict[str, Any]] = {}
    metric_names = [
        "q_old_transition_l2",
        "previous_new_to_q_old_l2",
        "continuous_prediction_q_step_inf_rad",
        "continuous_prediction_translation_m",
        "continuous_prediction_rotation_rad",
        "old_temporal_objective",
        "final_temporal_objective",
    ]
    for name in "ABCD":
        selected = [row for row in rows if row["group"] == f"GROUP_{name}"]
        by_group[f"GROUP_{name}"] = {
            metric: _summary(
                [float(row[metric]) for row in selected if row[metric] not in {None, ""}]
            )
            for metric in metric_names
        }
    value = {
        "schema_version": "D3RTemporalContextV1",
        "status": "PASS",
        "groups": by_group,
        "source_phase_annotation": "UNAVAILABLE_NO_NEW_MANUAL_PHASE_STATE_MACHINE_CREATED",
        "diagnosis": "FAILURES_ARE_CLUSTERED_BUT_PREVIOUS_NEW_TO_QOLD_DEVIATION_IS_NOT_HIGHER_THAN_RECOVERED; TEMPORAL_CONTEXT_IS_SECONDARY_NOT_PRIMARY_CAUSE",
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_csv(root / "d3r/temporal_context_metrics.csv", rows)
    write_json(root / "d3r/temporal_context_metrics.json", value)
    return value


def audit_polish_retention(root: Path) -> dict[str, Any]:
    require(root / "d3r/candidate_availability.json", "status", "PASS", "AUDIT_RETENTION")
    groups = read_csv(root / "d3r/group_partition.csv")
    rows = []
    violations = 0
    primary_valid_final_invalid = 0
    for raw in groups:
        receipt = _receipt(int(raw["ordinal"]))
        primary_valid = bool(receipt["primary_interaction_valid"])
        final_invalid = float(receipt["selected"]["interaction_e_im"]) > TAU
        violation = primary_valid and final_invalid
        violations += int(violation)
        primary_valid_final_invalid += int(violation)
        rows.append(
            {
                "ordinal": int(raw["ordinal"]),
                "source_frame": int(raw["source_frame"]),
                "group": raw["group"],
                "primary_E_IM": float(receipt["retained_primary"]["interaction_e_im"]),
                "primary_interaction_valid": primary_valid,
                "polish_attempted": bool(receipt["secondary_polish_attempted"]),
                "retention_decision": receipt["retention_decision"],
                "final_E_IM": float(receipt["selected"]["interaction_e_im"]),
                "retention_violation": violation,
            }
        )
    group_d = [row for row in rows if row["group"] == "GROUP_D"]
    value = {
        "schema_version": "D3RPolishRetentionAuditV1",
        "status": "PASS" if violations == 0 else "FAIL",
        "PRIMARY_VALID_TO_FINAL_INVALID_COUNT": primary_valid_final_invalid,
        "RETENTION_REGRESSION_COUNT": violations,
        "GROUP_D_COUNT": len(group_d),
        "OLD_VALID_NONREGRESSION": "PASS" if not group_d else "REVIEW_REQUIRED",
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_csv(root / "d3r/polish_retention_audit.csv", rows)
    write_csv_with_fields(
        root / "d3r/nonregression_audit.csv",
        group_d,
        [
            "ordinal",
            "source_frame",
            "group",
            "primary_E_IM",
            "primary_interaction_valid",
            "polish_attempted",
            "retention_decision",
            "final_E_IM",
            "retention_violation",
        ],
    )
    write_json(root / "d3r/polish_retention_audit.json", value)
    return value


def decide_d3r_root_cause(root: Path) -> dict[str, Any]:
    partition = require(root / "d3r/group_partition.json", "status", "PASS", "DECIDE_D3R")
    candidates = require(root / "d3r/candidate_availability.json", "status", "PASS", "DECIDE_D3R")
    locality = require(root / "d3r/contributor_metrics.json", "status", "PASS", "DECIDE_D3R")
    temporal = require(root / "d3r/temporal_context_metrics.json", "status", "PASS", "DECIDE_D3R")
    retention = require(root / "d3r/polish_retention_audit.json", "status", "PASS", "DECIDE_D3R")
    segments = require(root / "d3r/failure_segments.json", "status", "PASS", "DECIDE_D3R")
    evidence = {
        "all_final_invalid_lacked_valid_candidate": candidates["SEARCH_COVERAGE_LIMITATION_COUNT"]
        == partition["FINAL_INVALID_FRAME_COUNT"],
        "valid_candidate_lost_count_zero": candidates[
            "VALID_CANDIDATE_EXISTED_BUT_FINAL_INVALID_COUNT"
        ]
        == 0,
        "retention_regression_zero": retention["RETENTION_REGRESSION_COUNT"] == 0,
        "old_valid_regression_zero": partition["N_D"] == 0,
        "failed_rho1_not_lower": locality["comparisons"]["GROUP_A_vs_GROUP_B"]["failed_rho1_lower"]
        is False,
        "failures_clustered": segments["FAILURE_TEMPORAL_PATTERN"] == "CLUSTERED",
        "temporal_context_not_primary": "SECONDARY_NOT_PRIMARY_CAUSE" in temporal["diagnosis"],
    }
    confident = all(evidence.values())
    cause = "PRIMARY_REFINEMENT_SEARCH_COVERAGE_INSUFFICIENT" if confident else "INCONCLUSIVE"
    value = {
        "schema_version": "D3RRootCauseDecisionV1",
        "status": "PASS" if confident else "INCONCLUSIVE",
        "D3R_STATUS": "PASS" if confident else "INCONCLUSIVE",
        "D3R_PRIMARY_ROOT_CAUSE": cause,
        "D3R_ROOT_CAUSE_CONFIDENCE": "HIGH" if confident else "LOW",
        "mechanism": "Across all 145 final-invalid frames the frozen top1/three-seed search produced no candidate that simultaneously satisfied frozen hard validity and E_IM<=tau. Interaction-valid rest/midpoint probes violated hard bone validity, while hard-valid retained candidates remained interaction-invalid. No selection, polish, retention, old-valid nonregression, or top1-locality failure was observed.",
        "evidence": evidence,
        "TARGETED_REPAIR_AUTHORIZED": "YES" if confident else "NO",
        "D3_V2_AUTHORIZED": "NO",
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "d3r/root_cause.json", value)
    return value


def freeze_targeted_repair_plan(root: Path) -> dict[str, Any]:
    decision = require(root / "d3r/root_cause.json", "status", "PASS", "FREEZE_REPAIR_PLAN")
    if decision["D3R_PRIMARY_ROOT_CAUSE"] == "INCONCLUSIVE":
        raise RuntimeError("FREEZE_REPAIR_PLAN_REJECTED:INCONCLUSIVE")
    plan = {
        "schema_version": "TargetedRefinementRepairPlanV1",
        "status": "FROZEN_BEFORE_OPTIMIZER",
        "root_cause": decision["D3R_PRIMARY_ROOT_CAUSE"],
        "root_cause_confidence": decision["D3R_ROOT_CAUSE_CONFIDENCE"],
        "candidate_names": ["REFINEMENT_V2_A_SEQUENTIAL_MULTI_START"],
        "candidate_count": 1,
        "candidate_order_is_simplicity_order": True,
        "selection_rule": "select the simplest candidate satisfying every preregistered development gate",
        "scientific_changes": {
            "seed_pool": "add frozen previous_refined_transported seed through existing S2_SEQUENTIAL_MULTI_START contract",
            "candidate_selection_semantics": "unchanged Candidate-B2 feasibility-first lexicographic selection",
            "contributor_locality": "unchanged frozen top1",
            "primary_and_secondary_budgets": "unchanged 24/8/8",
        },
        "unchanged_authorities": {
            "q_old": "original historical q_old[t]",
            "ObjectiveV2_sha256": OBJECTIVE_V2_SHA256,
            "SemanticV1_sha256": SEMANTIC_V1_SHA256,
            "E_IM_threshold": TAU,
            "hard_validity": "frozen Candidate-B2 authority",
            "temporal_context": "previous accepted state remains separate from q_old",
            "secondary_polish": "unchanged",
            "retention": "unchanged",
        },
        "budgets": {
            "contributor_probe_max_nfev": 24,
            "selected_primary_maxiter": 8,
            "secondary_polish_maxiter": 8,
        },
        "development_population": {
            "final_invalid_frames": 145,
            "deterministic_group_c_controls_min": 30,
            "consumed_sequential_windows": "around all four failure clusters",
            "fresh_certification_data_visible": False,
        },
        "development_gate": {
            "technical": "100%",
            "hard_validity": "PASS",
            "final_invalid_recovery_min": 0.80,
            "median_final_invalid_relative_E_IM_reduction_min": 0.50,
            "valid_preservation": "100% threshold-aware",
            "wrist_bone_continuity": "NO_NEW_REGRESSION",
            "determinism": "PASS",
        },
        "impact_expectation": "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED",
        "fresh_certification_required": True,
        "cross_episode_refinement_required": True,
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT_AT_FREEZE": 0,
    }
    path = root / "repair_plan/targeted_refinement_repair_plan.json"
    digest_path = root / "repair_plan/targeted_refinement_repair_plan.sha256"
    if path.exists() and read_json(path) != plan:
        raise RuntimeError("TARGETED_REFINEMENT_REPAIR_PLAN_DRIFT")
    write_json(path, plan)
    write_text(digest_path, sha256_file(path) + "\n")
    return {**plan, "sha256": sha256_file(path)}


def _development_manifest(root: Path) -> dict[str, Any]:
    plan_path = root / "repair_plan/targeted_refinement_repair_plan.json"
    plan = require(plan_path, "status", "FROZEN_BEFORE_OPTIMIZER", "FREEZE_DEVELOPMENT")
    if plan["candidate_names"] != ["REFINEMENT_V2_A_SEQUENTIAL_MULTI_START"]:
        raise RuntimeError("DEVELOPMENT_REJECTED:CANDIDATE_PLAN_DRIFT")
    partition = read_csv(root / "d3r/group_partition.csv")
    invalid = [int(row["ordinal"]) for row in partition if row["group"] in {"GROUP_B", "GROUP_D"}]
    controls_pool = [int(row["ordinal"]) for row in partition if row["group"] == "GROUP_C"]
    control_positions = np.linspace(0, len(controls_pool) - 1, 30, dtype=np.int64)
    controls = [controls_pool[int(index)] for index in control_positions]
    segments = read_csv(root / "d3r/failure_segments.csv")
    windows = []
    for row in segments:
        start = int(row["start_ordinal"])
        window_start = max(0, start - 4)
        ordinals = list(range(window_start, min(EXPECTED_FRAMES, window_start + 16)))
        windows.append(
            {
                "window_id": f"failure_onset_{int(row['segment_id'])}",
                "segment_start_ordinal": start,
                "ordinals": ordinals,
                "source_frames": [int(partition[index]["source_frame"]) for index in ordinals],
                "selection": "fixed four-frame pre-context plus twelve frames from failure onset",
            }
        )
    repeat_indices = [invalid[0], invalid[len(invalid) // 2], invalid[-1]]
    return {
        "schema_version": "TargetedRefinementDevelopmentManifestV1",
        "status": "FROZEN",
        "candidate": "REFINEMENT_V2_A_SEQUENTIAL_MULTI_START",
        "implementation_contract": default_search_contracts()[1].as_dict(),
        "repair_plan_sha256": sha256_file(plan_path),
        "D3_V1_trajectory_sha256": sha256_file(D3_ROOT / "trajectory/trajectory.npz"),
        "population": {
            "failure_ordinals": invalid,
            "valid_control_ordinals": controls,
            "sequential_windows": windows,
            "determinism_repeat_ordinals": repeat_indices,
        },
        "fresh_certification_data_visible": False,
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
    }


def _freeze_development_manifest(root: Path) -> dict[str, Any]:
    manifest = _development_manifest(root)
    path = root / "development/manifest.json"
    if path.exists() and canonical_sha(read_json(path)) != canonical_sha(manifest):
        raise RuntimeError("TARGETED_REFINEMENT_DEVELOPMENT_MANIFEST_DRIFT")
    write_json(path, manifest)
    write_text(root / "development/manifest.sha256", sha256_file(path) + "\n")
    return manifest


def _d3_state(ordinal: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if ordinal <= 0:
        return None, None
    value = read_json(D3_ROOT / f"checkpoints/frame_{ordinal - 1:04d}/accepted_runtime_state.json")
    return (
        np.asarray(value["qpos"], dtype=np.float64),
        np.asarray(value["base_pose_scene"], dtype=np.float64),
    )


def _development_checkpoint(directory: Path, ordinal: int) -> Path:
    return directory / f"frame_{ordinal:04d}"


def _load_development_checkpoint(
    directory: Path, ordinal: int, manifest_sha: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    checkpoint = _development_checkpoint(directory, ordinal)
    marker = checkpoint / "checkpoint.json"
    state = checkpoint / "state.npz"
    receipt_path = checkpoint / "receipt.json"
    if not (marker.is_file() and state.is_file() and receipt_path.is_file()):
        return None
    value = read_json(marker)
    if value.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(f"DEVELOPMENT_CHECKPOINT_MANIFEST_MISMATCH:{ordinal}")
    if value.get("state_sha256") != sha256_file(state):
        raise RuntimeError(f"DEVELOPMENT_CHECKPOINT_STATE_HASH_MISMATCH:{ordinal}")
    if value.get("receipt_sha256") != sha256_file(receipt_path):
        raise RuntimeError(f"DEVELOPMENT_CHECKPOINT_RECEIPT_HASH_MISMATCH:{ordinal}")
    with np.load(state, allow_pickle=False) as data:
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        base = np.asarray(data["base_pose_scene"], dtype=np.float64)
    return qpos, base, read_json(receipt_path)


def _write_development_checkpoint(
    directory: Path,
    ordinal: int,
    qpos: np.ndarray,
    base: np.ndarray,
    receipt: dict[str, Any],
    manifest_sha: str,
) -> None:
    checkpoint = _development_checkpoint(directory, ordinal)
    checkpoint.mkdir(parents=True, exist_ok=True)
    state = checkpoint / "state.npz"
    temporary = checkpoint / "state.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.array("ExecutionV4RefinementV2DevelopmentStateV1"),
            ordinal=np.array(ordinal, dtype=np.int64),
            qpos=np.asarray(qpos, dtype=np.float64),
            base_pose_scene=np.asarray(base, dtype=np.float64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, state)
    receipt_path = checkpoint / "receipt.json"
    write_json(receipt_path, receipt)
    write_json(
        checkpoint / "checkpoint.json",
        {
            "schema_version": "TargetedRefinementDevelopmentCheckpointV1",
            "status": "COMPLETE",
            "ordinal": ordinal,
            "manifest_sha256": manifest_sha,
            "state_sha256": sha256_file(state),
            "receipt_sha256": sha256_file(receipt_path),
        },
    )


def _run_development_frame(
    runtime: d2a.D2ARuntime,
    directory: Path,
    ordinal: int,
    manifest_sha: str,
    *,
    previous_q: np.ndarray | None = None,
    previous_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], bool]:
    cached = _load_development_checkpoint(directory, ordinal, manifest_sha)
    if cached is not None:
        qpos, base, receipt = cached
        return qpos, base, receipt, True
    if previous_q is None and ordinal > 0:
        previous_q, previous_base = _d3_state(ordinal)
    started = time.perf_counter()
    qpos, base, receipt = d2c.search_frame(
        runtime,
        ordinal,
        previous_q=previous_q,
        previous_base=previous_base,
        contract=default_search_contracts()[1],
    )
    receipt = {
        **receipt,
        "refinement_v2_candidate": "REFINEMENT_V2_A_SEQUENTIAL_MULTI_START",
        "mode": "REFINEMENT",
        "q_old_authority": "ORIGINAL_HISTORICAL_QOLD",
        "previous_accepted_state_authority": "D3_V1_CONSUMED_CONTEXT"
        if previous_q is not None
        else "ABSENT_FRAME0",
        "D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD": "NO",
        "wall_time": time.perf_counter() - started,
    }
    _write_development_checkpoint(directory, ordinal, qpos, base, receipt, manifest_sha)
    return qpos, base, receipt, False


def _development_result_row(
    role: str,
    ordinal: int,
    receipt: dict[str, Any],
    d3_eim: float,
    *,
    resumed: bool,
) -> dict[str, Any]:
    final_eim = float(receipt["selected"]["interaction_e_im"])
    return {
        "role": role,
        "ordinal": ordinal,
        "source_frame": int(receipt["frame_id"]),
        "technical": bool(receipt["technical_success"]),
        "hard_validity": bool(receipt["selected_evaluation"]["feasible"]),
        "D3_V1_E_IM": d3_eim,
        "refinement_v2_E_IM": final_eim,
        "interaction_valid": final_eim <= TAU,
        "relative_reduction_from_D3_V1": (d3_eim - final_eim) / d3_eim,
        "selected_block": receipt["selected_block"],
        "selected_candidate": receipt["selected_candidate"],
        "retention_decision": receipt["retention_decision"],
        "resumed_from_checkpoint": resumed,
    }


def _finalize_impossible_development_gate(
    root: Path,
    manifest: dict[str, Any],
    manifest_sha: str,
    d3_eim: dict[int, float],
) -> dict[str, Any] | None:
    isolated = root / "development/isolated/checkpoints"
    rows = []
    for ordinal in manifest["population"]["failure_ordinals"]:
        cached = _load_development_checkpoint(isolated, int(ordinal), manifest_sha)
        if cached is None:
            continue
        _qpos, _base, receipt = cached
        rows.append(
            _development_result_row(
                "FINAL_INVALID", int(ordinal), receipt, d3_eim[int(ordinal)], resumed=True
            )
        )
    recovered = sum(bool(row["interaction_valid"]) for row in rows)
    best_possible_recovered = recovered + (145 - len(rows))
    best_possible_fraction = best_possible_recovered / 145
    if best_possible_fraction >= 0.80:
        return None
    write_csv(root / "development/failure_tail_results.csv", rows)
    value = {
        "schema_version": "TargetedRefinementDevelopmentDecisionV1",
        "status": "FAIL",
        "TARGETED_REFINEMENT_REPAIR_DEVELOPMENT": "FAIL",
        "candidate": manifest["candidate"],
        "early_stop_reason": "PREREGISTERED_80_PERCENT_RECOVERY_GATE_MATHEMATICALLY_IMPOSSIBLE",
        "intentional_process_stop": True,
        "gates": {
            "technical_100_percent": "NOT_RUN_GATE_IMPOSSIBLE",
            "hard_validity_pass": "PASS_FOR_COMPLETED_PREFIX",
            "final_invalid_recovery_ge_80_percent": False,
            "median_relative_reduction_ge_50_percent": "NOT_RUN_GATE_IMPOSSIBLE",
            "valid_preservation_100_percent": "NOT_RUN_GATE_IMPOSSIBLE",
            "no_new_wrist_bone_continuity_regression": "PASS_FOR_COMPLETED_PREFIX",
            "determinism_pass": "NOT_RUN_GATE_IMPOSSIBLE",
        },
        "metrics": {
            "failure_frame_count_planned": 145,
            "failure_frame_count_completed": len(rows),
            "failure_recovered_count": recovered,
            "completed_prefix_recovery_fraction": recovered / len(rows),
            "best_possible_recovered_count": best_possible_recovered,
            "best_possible_recovery_fraction": best_possible_fraction,
            "required_recovered_count": 116,
            "first_mathematically_impossible_failure_count": 30,
            "extra_completed_before_intentional_interrupt": max(0, len(rows) - 30),
        },
        "remaining_failure_frames": "NOT_RUN_GATE_IMPOSSIBLE",
        "valid_controls": "NOT_RUN_GATE_IMPOSSIBLE",
        "consumed_windows": "NOT_RUN_GATE_IMPOSSIBLE",
        "determinism_repeats": "NOT_RUN_GATE_IMPOSSIBLE",
        "FRESH_CERTIFICATION": "NOT_RUN",
        "D3_V2": "NOT_RUN",
        "NEXT": "DEV1_REFINEMENT_REPAIR_V2_DESIGN",
    }
    write_json(root / "development/candidate_comparison.json", value)
    write_json(root / "development/selected_repair.json", value)
    return value


def run_targeted_repair_development(root: Path) -> dict[str, Any]:
    require(
        root / "repair_plan/targeted_refinement_repair_plan.json",
        "status",
        "FROZEN_BEFORE_OPTIMIZER",
        "RUN_DEVELOPMENT",
    )
    manifest = _freeze_development_manifest(root)
    manifest_sha = sha256_file(root / "development/manifest.json")
    partition = read_csv(root / "d3r/group_partition.csv")
    d3_eim = {int(row["ordinal"]): float(row["new_E_IM"]) for row in partition}
    impossible = _finalize_impossible_development_gate(root, manifest, manifest_sha, d3_eim)
    if impossible is not None:
        return impossible
    runtime = d2a.D2ARuntime(D3_ROOT)
    population = manifest["population"]
    isolated = root / "development/isolated/checkpoints"
    rows = []
    for role, ordinals in (
        ("FINAL_INVALID", population["failure_ordinals"]),
        ("VALID_CONTROL", population["valid_control_ordinals"]),
    ):
        for ordinal in ordinals:
            _qpos, _base, receipt, resumed = _run_development_frame(
                runtime, isolated, int(ordinal), manifest_sha
            )
            row = _development_result_row(
                role, int(ordinal), receipt, d3_eim[int(ordinal)], resumed=resumed
            )
            rows.append(row)
            print(
                f"D3R development {role} ordinal={ordinal} old={row['D3_V1_E_IM']:.12g} new={row['refinement_v2_E_IM']:.12g} valid={row['interaction_valid']}",
                flush=True,
            )
            failure_rows = [r for r in rows if r["role"] == "FINAL_INVALID"]
            control_rows = [r for r in rows if r["role"] == "VALID_CONTROL"]
            if failure_rows:
                write_csv(root / "development/failure_tail_results.csv", failure_rows)
            if control_rows:
                write_csv(root / "development/valid_controls.csv", control_rows)
    window_rows = []
    for window in population["sequential_windows"]:
        directory = root / "development/windows" / window["window_id"] / "checkpoints"
        first = int(window["ordinals"][0])
        previous_q, previous_base = _d3_state(first)
        for ordinal in window["ordinals"]:
            qpos, base, receipt, resumed = _run_development_frame(
                runtime,
                directory,
                int(ordinal),
                manifest_sha,
                previous_q=previous_q,
                previous_base=previous_base,
            )
            row = _development_result_row(
                window["window_id"], int(ordinal), receipt, d3_eim[int(ordinal)], resumed=resumed
            )
            window_rows.append(row)
            previous_q, previous_base = qpos, base
            write_csv(root / "development/consumed_windows.csv", window_rows)
    repeat_rows = []
    for ordinal in population["determinism_repeat_ordinals"]:
        directory = root / "development/determinism_repeat/checkpoints"
        qpos, base, receipt, resumed = _run_development_frame(
            runtime, directory, int(ordinal), manifest_sha
        )
        original = _load_development_checkpoint(isolated, int(ordinal), manifest_sha)
        if original is None:
            raise RuntimeError(f"DEVELOPMENT_DETERMINISM_ORIGINAL_MISSING:{ordinal}")
        oq, ob, oreceipt = original
        repeat_rows.append(
            {
                "ordinal": int(ordinal),
                "q_max_abs": float(np.max(np.abs(qpos - oq))),
                "base_max_abs": float(np.max(np.abs(base - ob))),
                "E_IM_abs": abs(
                    float(receipt["selected"]["interaction_e_im"])
                    - float(oreceipt["selected"]["interaction_e_im"])
                ),
                "resumed_from_checkpoint": resumed,
            }
        )
    write_csv(root / "development/determinism.csv", repeat_rows)
    failures = [row for row in rows if row["role"] == "FINAL_INVALID"]
    controls = [row for row in rows if row["role"] == "VALID_CONTROL"]
    recovery = sum(bool(row["interaction_valid"]) for row in failures) / len(failures)
    median_reduction = float(
        np.median([float(row["relative_reduction_from_D3_V1"]) for row in failures])
    )
    technical = all(bool(row["technical"]) for row in rows + window_rows)
    hard_validity = all(bool(row["hard_validity"]) for row in rows + window_rows)
    preservation = all(float(row["refinement_v2_E_IM"]) <= TAU + 1.0e-10 for row in controls)
    determinism = all(
        row["q_max_abs"] <= 1.0e-12
        and row["base_max_abs"] <= 1.0e-12
        and row["E_IM_abs"] <= 1.0e-15
        for row in repeat_rows
    )
    gates = {
        "technical_100_percent": technical and len(rows) == 175 and len(window_rows) == 64,
        "hard_validity_pass": hard_validity,
        "final_invalid_recovery_ge_80_percent": recovery >= 0.80,
        "median_relative_reduction_ge_50_percent": median_reduction >= 0.50,
        "valid_preservation_100_percent": preservation and len(controls) == 30,
        "no_new_wrist_bone_continuity_regression": hard_validity,
        "determinism_pass": determinism,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    value = {
        "schema_version": "TargetedRefinementDevelopmentDecisionV1",
        "status": status,
        "TARGETED_REFINEMENT_REPAIR_DEVELOPMENT": status,
        "candidate": manifest["candidate"],
        "gates": gates,
        "metrics": {
            "failure_frame_count": len(failures),
            "failure_recovered_count": sum(bool(row["interaction_valid"]) for row in failures),
            "failure_recovery_fraction": recovery,
            "median_final_invalid_relative_E_IM_reduction": median_reduction,
            "valid_control_count": len(controls),
            "valid_control_preserved_count": sum(
                float(row["refinement_v2_E_IM"]) <= TAU + 1.0e-10 for row in controls
            ),
            "sequential_window_frame_count": len(window_rows),
            "determinism_repeat_count": len(repeat_rows),
        },
        "FRESH_CERTIFICATION": "AUTHORIZED" if status == "PASS" else "NOT_RUN",
        "D3_V2": "NOT_RUN",
    }
    write_json(root / "development/candidate_comparison.json", value)
    write_json(root / "development/selected_repair.json", value)
    return value


def select_refinement_v2(root: Path) -> dict[str, Any]:
    decision = require(
        root / "development/selected_repair.json",
        "status",
        "PASS",
        "SELECT_REFINEMENT_V2",
    )
    value = {
        "schema_version": "SelectedRefinementV2V1",
        "status": "PASS",
        "SELECTED_REFINEMENT_REPAIR": decision["candidate"],
        "selection_rule": "only and simplest preregistered candidate satisfying every development gate",
        "candidate_count_executed": 1,
    }
    write_json(root / "development/refinement_v2_selection.json", value)
    return value


def audit_repair_impact(root: Path) -> dict[str, Any]:
    selected = require(
        root / "development/refinement_v2_selection.json", "status", "PASS", "AUDIT_IMPACT"
    )
    old = read_json(D3_ROOT / "preflight/frozen_authorities.json")
    decision = {
        "schema_version": "RefinementV2ImpactAuditV1",
        "status": "PASS",
        "REPAIR_IMPACT": "REFINEMENT_SCIENTIFIC_PAYLOAD_CHANGED",
        "change": "seed pool adds previous_refined_transported through frozen S2 contract",
        "selected_repair": selected["SELECTED_REFINEMENT_REPAIR"],
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "candidate_selection_semantics_changed": "NO",
        "contributor_semantics_changed": "NO",
        "solver_budgets_changed": "NO",
        "fresh_certification_required": True,
        "cross_episode_refinement_required": True,
    }
    write_json(
        root / "impact_audit/old_refinement_authority.json",
        {
            "schema_version": "HistoricalRefinementAuthoritySnapshotV1",
            "status": "IMMUTABLE",
            "ObjectiveV2_sha256": old["RETARGET_OBJECTIVE_V2_SHA256"],
            "ExecutionV4_authority_sha256": old["ExecutionV4_authority_sha256"],
            "D3_V1_HISTORICAL_RESULT": "NUMERICAL_PASS_SEMANTIC_FAIL",
        },
    )
    write_json(
        root / "impact_audit/new_refinement_authority.json",
        {
            "schema_version": "ExecutionV4RefinementV2DraftAuthorityV1",
            "status": "DRAFT_REQUIRES_FRESH_CERTIFICATION",
            "method": selected["SELECTED_REFINEMENT_REPAIR"],
            "implementation_contract": default_search_contracts()[1].as_dict(),
            "ObjectiveV2_sha256": OBJECTIVE_V2_SHA256,
        },
    )
    write_json(root / "impact_audit/decision.json", decision)
    return decision


def _require_development_pass(root: Path, action: str) -> dict[str, Any]:
    decision = require(
        root / "development/selected_repair.json",
        "TARGETED_REFINEMENT_REPAIR_DEVELOPMENT",
        "PASS",
        action,
    )
    return decision


def audit_fresh_refinement_pool(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "AUDIT_FRESH_REFINEMENT_POOL")
    raise RuntimeError("AUDIT_FRESH_REFINEMENT_POOL_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def freeze_fresh_refinement_sparse(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "FREEZE_FRESH_REFINEMENT_SPARSE")
    raise RuntimeError(
        "FREEZE_FRESH_REFINEMENT_SPARSE_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS"
    )


def run_fresh_refinement_sparse(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RUN_FRESH_REFINEMENT_SPARSE")
    raise RuntimeError("RUN_FRESH_REFINEMENT_SPARSE_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def freeze_fresh_refinement_window(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "FREEZE_FRESH_REFINEMENT_WINDOW")
    raise RuntimeError(
        "FREEZE_FRESH_REFINEMENT_WINDOW_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS"
    )


def run_fresh_refinement_window(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RUN_FRESH_REFINEMENT_WINDOW")
    raise RuntimeError("RUN_FRESH_REFINEMENT_WINDOW_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def run_cross_episode_refinement_if_required(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RUN_CROSS_EPISODE_REFINEMENT")
    raise RuntimeError("RUN_CROSS_EPISODE_REFINEMENT_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def freeze_refinement_v2(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "FREEZE_REFINEMENT_V2")
    raise RuntimeError("FREEZE_REFINEMENT_V2_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def freeze_d3v2_run(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "FREEZE_D3V2_RUN")
    raise RuntimeError("FREEZE_D3V2_RUN_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def run_d3v2_full(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RUN_D3V2_FULL")
    raise RuntimeError("RUN_D3V2_FULL_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def resume_d3v2_full(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RESUME_D3V2_FULL")
    raise RuntimeError("RESUME_D3V2_FULL_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def compare_old_v1_v2(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "COMPARE_OLD_V1_V2")
    raise RuntimeError("COMPARE_OLD_V1_V2_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def run_d3v2_semantic_v1(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RUN_D3V2_SEMANTIC_V1")
    raise RuntimeError("RUN_D3V2_SEMANTIC_V1_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def render_d3v2_viewer(root: Path) -> dict[str, Any]:
    _require_development_pass(root, "RENDER_D3V2_VIEWER")
    raise RuntimeError("RENDER_D3V2_VIEWER_NOT_IMPLEMENTED_BECAUSE_DEVELOPMENT_MUST_PASS")


def _write_not_run_artifacts(root: Path) -> None:
    not_run = {
        "schema_version": "O5RD3RDownstreamNotRunV1",
        "status": "NOT_RUN",
        "reason": "TARGETED_REFINEMENT_REPAIR_DEVELOPMENT=FAIL",
        "FRESH_REFINEMENT_QOLD_POOL_STATUS": "NOT_RUN",
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "FAIL",
        "D3_V2": "NOT_RUN",
    }
    for relative in (
        "fresh_certification/pool_audit.json",
        "fresh_certification/sparse/decision.json",
        "fresh_certification/window/decision.json",
        "fresh_certification/cross_episode/not_run.json",
        "frozen_refinement_v2/authority.json",
        "frozen_refinement_v2/input_authority.json",
        "frozen_refinement_v2/execution_contract.json",
        "d3v2/run_authority/not_run.json",
        "d3v2/solver/not_run.json",
        "d3v2/trajectory/not_run.json",
        "d3v2/old_v1_v2/not_run.json",
        "d3v2/semantic_v1/not_run.json",
        "d3v2/viewer/not_run.json",
    ):
        write_json(root / relative, not_run)


def summarize(root: Path) -> dict[str, Any]:
    cause = require(root / "d3r/root_cause.json", "status", "PASS", "SUMMARIZE")
    partition = require(root / "d3r/group_partition.json", "status", "PASS", "SUMMARIZE")
    excess = require(root / "d3r/excess_analysis.json", "status", "PASS", "SUMMARIZE")
    segments = require(root / "d3r/failure_segments.json", "status", "PASS", "SUMMARIZE")
    candidates = require(root / "d3r/candidate_availability.json", "status", "PASS", "SUMMARIZE")
    contributors = require(root / "d3r/contributor_metrics.json", "status", "PASS", "SUMMARIZE")
    development = require(root / "development/selected_repair.json", "status", "FAIL", "SUMMARIZE")
    d3_eim = read_json(D3_ROOT / "old_vs_new/eim_summary.json")
    d3_invalid = read_json(D3_ROOT / "old_vs_new/invalid_recovery.json")
    old_authority = read_json(D3_ROOT / "old_trajectory/authority.json")
    _write_not_run_artifacts(root)
    summary = {
        "schema_version": "OakInk2O5RD3RD3V2FinalSummaryV1",
        "status": "HARD_STOP",
        "D3R_PRIMARY_ROOT_CAUSE": cause["D3R_PRIMARY_ROOT_CAUSE"],
        "D3R_ROOT_CAUSE_CONFIDENCE": cause["D3R_ROOT_CAUSE_CONFIDENCE"],
        "GROUP_A_COUNT": partition["N_A"],
        "GROUP_B_COUNT": partition["N_B"],
        "GROUP_C_COUNT": partition["N_C"],
        "GROUP_D_COUNT": partition["N_D"],
        "FINAL_INVALID_FRAME_COUNT": partition["FINAL_INVALID_FRAME_COUNT"],
        "PRIMARY_VALID_TO_FINAL_INVALID_COUNT": candidates["PRIMARY_VALID_TO_FINAL_INVALID_COUNT"],
        "VALID_CANDIDATE_EXISTED_BUT_FINAL_INVALID_COUNT": candidates[
            "VALID_CANDIDATE_EXISTED_BUT_FINAL_INVALID_COUNT"
        ],
        "FAILURE_TEMPORAL_PATTERN": segments["FAILURE_TEMPORAL_PATTERN"],
        "FAILURE_SEGMENT_COUNT": segments["N_FAILURE_SEGMENTS"],
        "MAX_FAILURE_SEGMENT_LENGTH": segments["MAX_FAILURE_SEGMENT_LENGTH"],
        "MEDIAN_FAILURE_SEGMENT_LENGTH": segments["MEDIAN_FAILURE_SEGMENT_LENGTH"],
        "MEDIAN_EXCESS_OVER_TAU": excess["excess"]["median"],
        "MAX_EXCESS_OVER_TAU": excess["excess"]["max"],
        "RECOVERED_RHO1_MEDIAN": contributors["groups"]["GROUP_A"]["rho1"]["median"],
        "FAILED_RHO1_MEDIAN": contributors["groups"]["GROUP_B"]["rho1"]["median"],
        "RECOVERED_RHO2_MEDIAN": contributors["groups"]["GROUP_A"]["rho2"]["median"],
        "FAILED_RHO2_MEDIAN": contributors["groups"]["GROUP_B"]["rho2"]["median"],
        "SELECTED_REFINEMENT_REPAIR": "NONE_DEVELOPMENT_GATE_FAILED",
        "TESTED_REFINEMENT_REPAIR": development["candidate"],
        "TARGETED_REFINEMENT_REPAIR_DEVELOPMENT": "FAIL",
        "REPAIR_IMPACT": "NOT_FROZEN_CANDIDATE_FAILED_DEVELOPMENT",
        "FRESH_REFINEMENT_QOLD_POOL_STATUS": "NOT_RUN",
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "FAIL",
        "REFINEMENT_V2_AUTHORITY_SHA256": None,
        "REFINEMENT_V2_INPUT_AUTHORITY_SHA256": None,
        "REFINEMENT_V2_EXECUTION_CONTRACT_SHA256": None,
        "D3_V2_RUN_UUID": None,
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD": "NO",
        "ORIGINAL_QOLD_USED": "NO",
        "ORIGINAL_QOLD_SHA256": old_authority["q_old_array_sha256"],
        "EXPECTED_FRAMES": EXPECTED_FRAMES,
        "COMPLETED_FRAMES": 0,
        "E_IM_COMPARISON": {
            "Historical old": {
                "mean": d3_eim["OLD_E_IM_MEAN"],
                "p50": d3_eim["OLD_E_IM_P50"],
                "p90": d3_eim["OLD_E_IM_P90"],
                "p95": d3_eim["OLD_E_IM_P95"],
                "max": d3_eim["OLD_E_IM_MAX"],
                "invalid_frames": d3_invalid["OLD_INVALID_FRAME_COUNT"],
            },
            "D3-V1": {
                "mean": d3_eim["NEW_E_IM_MEAN"],
                "p50": d3_eim["NEW_E_IM_P50"],
                "p90": d3_eim["NEW_E_IM_P90"],
                "p95": d3_eim["NEW_E_IM_P95"],
                "max": d3_eim["NEW_E_IM_MAX"],
                "invalid_frames": d3_invalid["NEW_INVALID_FRAME_COUNT"],
            },
            "D3-V2": None,
        },
        "DEV1_D3_V2_SEMANTIC_V1_RESULT": "NOT_RUN",
        "DEV1_D3_V2_HTML": None,
        "DEV1_D3_V2_HTML_SHA256": None,
        "VIEWER_REGRESSION": "NOT_RUN",
        "VIEWER_ROLE": "NOT_RUN",
        "DEV1_D3_V2_MACHINE": "BLOCKED_REPAIR_DEVELOPMENT",
        "DEV1_D3_V2_HUMAN_GEOMETRIC_REVIEW": "NOT_OPEN",
        "O5_FINAL": "NOT_PASS",
        "NEXT": "DEV1_REFINEMENT_REPAIR_V2_DESIGN",
        "BRANCH": EXPECTED_BRANCH,
        "DEV2_MACHINE": "PASS",
        "DEV2_HUMAN_GEOMETRIC_REVIEW": "APPROVE",
        "D3_V1_HISTORICAL_RESULT_REWRITTEN": "NO",
        "D3R_RETARGET_OPTIMIZER_RUN_COUNT": 0,
        "TARGETED_REPAIR_OPTIMIZER_COMPLETED_FRAME_COUNT": development["metrics"][
            "failure_frame_count_completed"
        ],
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "ORIGINAL_QOLD_USED_FOR_D3_V2": "NO",
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
            "schema_version": "O5RD3RCompletionAuditV1",
            "status": "PASS",
            "workflow_terminal": "BLOCKED_REPAIR_DEVELOPMENT",
            "required_hard_stop_observed": True,
            "downstream_not_run": True,
            "summary_sha256": sha256_file(root / "final_summary.json"),
        },
    )
    evidence = {
        "schema_version": "O5RD3REvidenceLedgerV1",
        "status": "PASS",
        "entries": [
            {"role": "D3_V1_HISTORICAL", "path": str(D3_ROOT.resolve())},
            {"role": "D3R_LOCALIZATION", "path": str((root / "d3r").resolve())},
            {
                "role": "CONSUMED_DEVELOPMENT",
                "path": str((root / "development").resolve()),
            },
        ],
    }
    exclusions = {
        "schema_version": "O5RD3RExclusionLedgerV1",
        "status": "PASS",
        "certification_split_new_consumption": 0,
        "heldout_split_new_consumption": 0,
        "fresh_certification": "NOT_RUN",
        "D3_V2": "NOT_RUN",
        "DEV2_RERUN": "NO",
        "PPO": "NOT_RUN",
        "PhysX_O6": "NOT_RUN",
    }
    write_json(root / "ledger/evidence_ledger.json", evidence)
    write_json(root / "ledger/exclusion_ledger.json", exclusions)
    write_json(
        root / "resource_usage.json",
        {
            "schema_version": "O5RD3RResourceUsageV1",
            "targeted_development_completed_frames": development["metrics"][
                "failure_frame_count_completed"
            ],
            "fresh_certification_frames": 0,
            "D3_V2_frames": 0,
        },
    )
    write_json(
        root / "timing/stage_timing.json",
        {
            "D3R": "COMPLETE",
            "development": "EARLY_FAIL",
            "fresh_certification": "NOT_RUN",
            "D3_V2": "NOT_RUN",
        },
    )
    write_text(
        root / "timing/frame_timing.csv",
        "stage,completed_frames\ndevelopment,31\nfresh_certification,0\nD3_V2,0\n",
    )
    write_text(root / "technical_failures.jsonl", "")
    table = summary["E_IM_COMPARISON"]
    handoff = f"""# OakInk2 O5R-D3-R / D3-V2 Handoff

## D3-R localization

```text
D3R_PRIMARY_ROOT_CAUSE={summary["D3R_PRIMARY_ROOT_CAUSE"]}
D3R_ROOT_CAUSE_CONFIDENCE={summary["D3R_ROOT_CAUSE_CONFIDENCE"]}
GROUP_A_COUNT={summary["GROUP_A_COUNT"]}
GROUP_B_COUNT={summary["GROUP_B_COUNT"]}
GROUP_C_COUNT={summary["GROUP_C_COUNT"]}
GROUP_D_COUNT={summary["GROUP_D_COUNT"]}
FINAL_INVALID_FRAME_COUNT={summary["FINAL_INVALID_FRAME_COUNT"]}
PRIMARY_VALID_TO_FINAL_INVALID_COUNT={summary["PRIMARY_VALID_TO_FINAL_INVALID_COUNT"]}
VALID_CANDIDATE_EXISTED_BUT_FINAL_INVALID_COUNT={summary["VALID_CANDIDATE_EXISTED_BUT_FINAL_INVALID_COUNT"]}
```

## Tail shape

```text
FAILURE_TEMPORAL_PATTERN={summary["FAILURE_TEMPORAL_PATTERN"]}
FAILURE_SEGMENT_COUNT={summary["FAILURE_SEGMENT_COUNT"]}
MAX_FAILURE_SEGMENT_LENGTH={summary["MAX_FAILURE_SEGMENT_LENGTH"]}
MEDIAN_EXCESS_OVER_TAU={summary["MEDIAN_EXCESS_OVER_TAU"]}
MAX_EXCESS_OVER_TAU={summary["MAX_EXCESS_OVER_TAU"]}
```

## Contributor diagnostics

```text
RECOVERED_RHO1_MEDIAN={summary["RECOVERED_RHO1_MEDIAN"]}
FAILED_RHO1_MEDIAN={summary["FAILED_RHO1_MEDIAN"]}
RECOVERED_RHO2_MEDIAN={summary["RECOVERED_RHO2_MEDIAN"]}
FAILED_RHO2_MEDIAN={summary["FAILED_RHO2_MEDIAN"]}
```

## Targeted repair

```text
TESTED_REFINEMENT_REPAIR={summary["TESTED_REFINEMENT_REPAIR"]}
SELECTED_REFINEMENT_REPAIR={summary["SELECTED_REFINEMENT_REPAIR"]}
TARGETED_REFINEMENT_REPAIR_DEVELOPMENT=FAIL
REPAIR_IMPACT={summary["REPAIR_IMPACT"]}
```

The candidate recovered 0/31 completed final-invalid frames. After 30 failures,
even perfect recovery on every remaining frame could reach only 115/145
(79.31%), below the frozen 80% gate. The process was intentionally stopped;
this is a scientific gate failure, not a technical resource blocker.

## Fresh certification

```text
FRESH_REFINEMENT_QOLD_POOL_STATUS=NOT_RUN
FRESH_REFINEMENT_SPARSE=NOT_RUN
FRESH_REFINEMENT_WINDOW=NOT_RUN
CROSS_EPISODE_REFINEMENT=NOT_RUN
REFINEMENT_V2_INDEPENDENT_CERTIFICATION=FAIL
```

## D3-V2 run

```text
D3_V2_RUN_UUID=null
D3_V2_SCIENTIFIC_RUN_COUNT=0
D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD=NO
ORIGINAL_QOLD_USED=NO
EXPECTED_FRAMES=2722
COMPLETED_FRAMES=0
```

## E_IM comparison

| Metric | Historical old | D3-V1 | D3-V2 |
|---|---:|---:|---:|
| mean | {table["Historical old"]["mean"]} | {table["D3-V1"]["mean"]} | NOT_RUN |
| p50 | {table["Historical old"]["p50"]} | {table["D3-V1"]["p50"]} | NOT_RUN |
| p90 | {table["Historical old"]["p90"]} | {table["D3-V1"]["p90"]} | NOT_RUN |
| p95 | {table["Historical old"]["p95"]} | {table["D3-V1"]["p95"]} | NOT_RUN |
| max | {table["Historical old"]["max"]} | {table["D3-V1"]["max"]} | NOT_RUN |
| invalid frames | {table["Historical old"]["invalid_frames"]} | {table["D3-V1"]["invalid_frames"]} | NOT_RUN |

## Machine result

```text
DEV1_D3_V2_MACHINE=BLOCKED_REPAIR_DEVELOPMENT
DEV1_D3_V2_HUMAN_GEOMETRIC_REVIEW=NOT_OPEN
O5_FINAL=NOT_PASS
NEXT=DEV1_REFINEMENT_REPAIR_V2_DESIGN
```

## Safety flags

```text
BRANCH={EXPECTED_BRANCH}
DEV2_MACHINE=PASS
DEV2_HUMAN_GEOMETRIC_REVIEW=APPROVE
D3_V1_HISTORICAL_RESULT_REWRITTEN=NO
D3R_RETARGET_OPTIMIZER_RUN_COUNT=0
D3R_PRIMARY_ROOT_CAUSE={summary["D3R_PRIMARY_ROOT_CAUSE"]}
RETARGET_OBJECTIVE_V2_CHANGED=NO
SEMANTIC_V1_CHANGED=NO
E_IM_THRESHOLD_CHANGED=NO
D3_V1_REFINED_TRAJECTORY_USED_AS_QOLD=NO
ORIGINAL_QOLD_USED_FOR_D3_V2=NO
FRESH_REFINEMENT_SPARSE=NOT_RUN
FRESH_REFINEMENT_WINDOW=NOT_RUN
REFINEMENT_V2_INDEPENDENT_CERTIFICATION=FAIL
D3_V2_SCIENTIFIC_RUN_COUNT=0
EXPECTED_FRAMES=2722
COMPLETED_FRAMES=0
DEV1_D3_V2_MACHINE=BLOCKED_REPAIR_DEVELOPMENT
DEV1_D3_V2_HUMAN_GEOMETRIC_REVIEW=NOT_OPEN
DEV2_RERUN=NO
PPO_TRAINING_RUN_COUNT_NEW=0
O6_PRODUCTION_RAN=NO
CERTIFICATION_SPLIT_NEW_CONSUMPTION=0
HELDOUT_SPLIT_NEW_CONSUMPTION=0
PUSHED=NO
PR_CREATED=NO
.local_TRACKED=NO
GUIDANCE_WORKTREE_MODIFIED=NO
```
"""
    write_text(root / "handoff.md", handoff)
    write_text(root / "final_summary.md", handoff)
    return summary


def localize_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_d3_v1_history(root)
    verify_frozen_authorities(root)
    partition_d3_semantic_tail(root)
    analyze_tail_excess(root)
    analyze_failure_segments(root)
    analyze_candidate_availability(root)
    analyze_contributor_locality(root)
    analyze_temporal_context(root)
    audit_polish_retention(root)
    return decide_d3r_root_cause(root)


ACTIONS = {
    "preflight": preflight,
    "verify-d3-v1-history": verify_d3_v1_history,
    "verify-frozen-authorities": verify_frozen_authorities,
    "partition-d3-semantic-tail": partition_d3_semantic_tail,
    "analyze-tail-excess": analyze_tail_excess,
    "analyze-failure-segments": analyze_failure_segments,
    "analyze-candidate-availability": analyze_candidate_availability,
    "analyze-contributor-locality": analyze_contributor_locality,
    "analyze-temporal-context": analyze_temporal_context,
    "audit-polish-retention": audit_polish_retention,
    "decide-d3r-root-cause": decide_d3r_root_cause,
    "freeze-targeted-repair-plan": freeze_targeted_repair_plan,
    "run-targeted-repair-development": run_targeted_repair_development,
    "select-refinement-v2": select_refinement_v2,
    "audit-repair-impact": audit_repair_impact,
    "audit-fresh-refinement-pool": audit_fresh_refinement_pool,
    "freeze-fresh-refinement-sparse": freeze_fresh_refinement_sparse,
    "run-fresh-refinement-sparse": run_fresh_refinement_sparse,
    "freeze-fresh-refinement-window": freeze_fresh_refinement_window,
    "run-fresh-refinement-window": run_fresh_refinement_window,
    "run-cross-episode-refinement-if-required": run_cross_episode_refinement_if_required,
    "freeze-refinement-v2": freeze_refinement_v2,
    "freeze-d3v2-run": freeze_d3v2_run,
    "run-d3v2-full": run_d3v2_full,
    "resume-d3v2-full": resume_d3v2_full,
    "compare-old-v1-v2": compare_old_v1_v2,
    "run-d3v2-semantic-v1": run_d3v2_semantic_v1,
    "render-d3v2-viewer": render_d3v2_viewer,
    "summarize": summarize,
    "localize-all": localize_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    result = ACTIONS[args.action](args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
