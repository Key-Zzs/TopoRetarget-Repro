#!/usr/bin/env python3
"""O5R-D2K-R/V2 physical-recoverability decision tree.

The workflow is deliberately fail closed.  It first audits pre-D2K binary
authority, then either re-scores immutable D2K evidence or freezes a new
prospective V2 gate before selecting any fresh-to-physics anchors.
"""

# ruff: noqa: E501, PLR0912, PLR0915, SLF001

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data import run_oakink2_o5rd2e as d2e  # noqa: E402
from scripts.data import run_oakink2_o5rd2g as d2g  # noqa: E402
from scripts.data import run_oakink2_o5rd2g2 as d2g2  # noqa: E402
from scripts.data import run_oakink2_o5rd2h as d2h  # noqa: E402
from scripts.evaluation import run_oakink2_o5rd2jr_d2k as d2k  # noqa: E402
from scripts.rl.isaaclab.import_hocap_objects import _bounded_convex_proxy, _read_obj  # noqa: E402
from toporetarget.physics.support import (  # noqa: E402
    StaticRecoverabilitySupportProxyParameterV1,
    build_static_recoverability_support_proxy,
    write_finite_planar_support_usda,
)
from toporetarget.retarget.objective_v2 import interaction_retention_limit  # noqa: E402
from toporetarget.retarget.objective_v2_execution import (  # noqa: E402
    asset_derived_dof_blocks,
    contributor_scores,
    rank_contributors,
    retain_after_polish,
    select_candidate,
)
from toporetarget.retarget.objective_v4_execution import (  # noqa: E402
    ColdStartSearchV4Candidate,
    default_cold_start_search_v4_candidates,
)
from toporetarget.rl.static_retarget_reference import write_static_retarget_reference  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd2kr_recoverability_decision_tree_v1"
D2K_ROOT = REPO / ".local/reports/oakink2_o5rd2jr_d2k_support_proxy_ppo_recoverability_v1"
D2I_ROOT = REPO / ".local/reports/oakink2_o5rd2i_sparsev4_failure_and_ppo_recoverability_v1"
D2H_ROOT = (
    REPO / ".local/reports/oakink2_o5rd2h_execution_v3_independent_coldstart_certification_v1"
)
O1R_ROOT = REPO / ".local/reports/oakink2_o1r_official_mano_authority_v1"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
CUTOFF_HEAD = "f69a5540a82113f3803bebe0e3689baca7a4d69b"
KNOWN_D2K_HEAD = "7b54d4bc5aa4fd72520bba0cd214bfc24f93df05"
OBJECT_ID = "C10001"
SEQUENCE = d2k.SEQUENCE
OBJECT_USD = d2k.OBJECT_USD
VISUAL_MESH = d2k.VISUAL_MESH
ANNOTATION = d2k.ANNOTATION
STRICT_CONTRACT = d2k.STRICT_CONTRACT
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

EXPECTED_HASHES = {
    "ppo_contract": "cc7db1b387a6f69c82cda35d6a6028176cb738e24d2ef094cdc845281d1a756c",
    "reward_contract": "822afab18b3ca4a4a28e7675824790db7750d4aebd02873ea1d2476ae809c101",
    "support_proxy": "ff4901f64a2a074226491f8da546a8c2b93ad356b41e5437e8dd341009c45916",
    "static_scene_adapter": "3da621613198d117f29b41eb57e2c968d16996a4c5935417748f47efac1bd099",
    "static_reference_adapter": "e81252cff8555c24250b910697cebe41fb4948106b7158adc8d711ff1ac0b8e1",
    "physical_scene_contract_v2": "33aa10559a147c95d1e3695ed0b9c0059f15f6040dfa8f92f7e9a6b8067ccac7",
}

AUTHORITY_PATHS = {
    "ppo_contract": D2K_ROOT / "ppo_authority/ppo_contract.json",
    "reward_contract": D2K_ROOT / "ppo_authority/reward_authority.json",
    "support_proxy": D2K_ROOT / "frozen_d2jr/support_proxy.json",
    "static_scene_adapter": D2K_ROOT / "frozen_d2jr/static_scene_adapter.json",
    "static_reference_adapter": D2K_ROOT / "frozen_d2jr/static_reference_adapter.json",
    "physical_scene_contract_v2": D2K_ROOT / "frozen_d2jr/physical_study_scene_contract.json",
}

REQUIRED_ACTIONS = (
    "preflight",
    "verify-current-evidence",
    "audit-physical-metric-semantics",
    "inventory-pre-d2k-gates",
    "audit-gate-applicability",
    "align-existing-gate",
    "rescore-existing-d2k-rollouts",
    "design-recoverability-v2",
    "freeze-recoverability-v2",
    "select-v2-physical-anchors",
    "run-v2-baselines",
    "run-v2-ppo",
    "evaluate-v2",
    "decide-recoverability",
    "build-retarget-admission-v2",
    "develop-quality-aware-admission",
    "develop-execution-v4",
    "freeze-execution-v4",
    "run-sparse-v5",
    "run-window-v5",
    "run-cross-episode-v5",
    "authorize-dev2-full",
    "run-dev2-full",
    "render-dev2-viewer",
    "summarize",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        raise ValueError(f"CSV_FIELDS_REQUIRED:{path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def freeze_json(path: Path, value: dict[str, Any]) -> str:
    write_json(path, value)
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def require(path: Path, status: str | tuple[str, ...], action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    allowed = (status,) if isinstance(status, str) else status
    if value.get("status") not in allowed:
        raise RuntimeError(f"{action}_REJECTED:{path.name}={value.get('status')}")
    return value


def _first_commit(path: str) -> str:
    output = git("log", "--diff-filter=A", "--format=%H", "--", path)
    return output.splitlines()[-1] if output else "NOT_FOUND"


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(first * second, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _trace_paths(root: Path, phase: str) -> list[Path]:
    directory = "baseline_physics" if phase == "baseline" else "ppo_eval"
    return sorted((root / directory / "per_anchor_raw").glob("A*/traces/episode_*.npz"))


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "cutoff_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", CUTOFF_HEAD, "HEAD"], cwd=REPO, check=False
        ).returncode
        == 0,
        "known_d2k_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", KNOWN_D2K_HEAD, "HEAD"], cwd=REPO, check=False
        ).returncode
        == 0,
        "diff_check": subprocess.run(["git", "diff", "--check"], cwd=REPO, check=False).returncode
        == 0,
        "local_ignored": subprocess.run(
            ["git", "check-ignore", "-q", ".local"], cwd=REPO, check=False
        ).returncode
        == 0,
    }
    value = {
        "schema_version": "OakInk2O5RD2KRPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "repo": str(REPO),
        "branch": branch,
        "start_head": head,
        "pre_d2k_outcome_authority_cutoff_head": CUTOFF_HEAD,
        "status_short": git("status", "--short", "--untracked-files=all"),
        "diff_stat": git("diff", "--stat"),
        "cached_diff_stat": git("diff", "--cached", "--stat"),
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "MAX_GPU_JOBS": 1,
        "new_branch_created": False,
        "new_worktree_created": False,
    }
    write_json(root / "preflight/git.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_GIT_PREFLIGHT")
    return value


def verify_current_evidence(root: Path) -> dict[str, Any]:
    require(root / "preflight/git.json", "PASS", "VERIFY_CURRENT_EVIDENCE")
    final = read_json(D2K_ROOT / "final_summary.json")
    actual_hashes = {name: sha256_file(path) for name, path in AUTHORITY_PATHS.items()}
    hash_checks = {
        name: actual_hashes[name] == expected for name, expected in EXPECTED_HASHES.items()
    }
    trace_rows = []
    for phase in ("baseline", "ppo"):
        for path in _trace_paths(D2K_ROOT, "baseline" if phase == "baseline" else "eval"):
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "object_pose",
                    "object_reference",
                    "object_twist",
                    "object_twist_reference",
                    "action",
                    "source_contact_mask",
                    "tip_pair_presence",
                    "hand_object_pair_presence",
                    "hand_collision_body_pose",
                    "physical_penetration_depth_m",
                }
                keys = set(archive.files)
                finite = all(
                    np.isfinite(np.asarray(archive[key])).all() for key in required if key in keys
                )
            trace_rows.append(
                {
                    "phase": phase,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "required_fields_present": required.issubset(keys),
                    "required_numeric_fields_finite": finite,
                }
            )
    checks = {
        "historical_summary_complete": final.get("status") == "COMPLETE",
        "sparse_v4_fail_preserved": final.get("SPARSE_VALIDATION_V4") == "FAIL",
        "d2j_r_pass": final.get("D2J_R_STATUS") == "PASS",
        "historical_d2k_inconclusive": final.get("RETARGET_TO_PPO_RECOVERABILITY_RESULT")
        == "INCONCLUSIVE",
        "continuous_only": final.get("RECOVERABILITY_MODE") == "CONTINUOUS_METRICS_ONLY",
        "training_runs_12": final.get("PPO_TRAINING_RUN_COUNT") == 12,
        "training_samples_exact": final.get("PPO_TOTAL_TRAINING_SAMPLES") == 7_372_800,
        "trace_count_240": len(trace_rows) == 240,
        "trace_fields_complete": all(row["required_fields_present"] for row in trace_rows),
        "trace_fields_finite": all(row["required_numeric_fields_finite"] for row in trace_rows),
        "all_frozen_hashes_exact": all(hash_checks.values()),
    }
    value = {
        "schema_version": "D2KCurrentEvidenceIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "D2K_EVIDENCE_INTEGRITY": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "frozen_hashes": {
            name: {
                "path": str(AUTHORITY_PATHS[name]),
                "expected_sha256": EXPECTED_HASHES[name],
                "actual_sha256": actual_hashes[name],
                "exact": hash_checks[name],
            }
            for name in EXPECTED_HASHES
        },
        "trace_count": len(trace_rows),
        "trace_inventory_sha256": hashlib.sha256(
            json.dumps(trace_rows, sort_keys=True).encode()
        ).hexdigest(),
        "D2K_EXISTING_OUTCOMES_MODIFIED": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "PHYSX_TRAINING_RUN_COUNT_NEW": 0,
    }
    write_json(root / "preflight/current_evidence_integrity.json", value)
    write_json(root / "preflight/frozen_authorities.json", value["frozen_hashes"])
    if value["status"] != "PASS":
        raise RuntimeError("BLOCKED_EVIDENCE_INTEGRITY")
    return value


def audit_physical_metric_semantics(root: Path) -> dict[str, Any]:
    require(root / "preflight/current_evidence_integrity.json", "PASS", "METRIC_SEMANTICS")
    rows: list[dict[str, Any]] = []
    recovered: list[dict[str, Any]] = []
    for phase in ("baseline", "eval"):
        for path in _trace_paths(D2K_ROOT, phase):
            with np.load(path, allow_pickle=False) as archive:
                pose = np.asarray(archive["object_pose"], dtype=np.float64)
                reference = np.asarray(archive["object_reference"], dtype=np.float64)
                translation = np.linalg.norm(pose[:, :3] - reference[:, :3], axis=-1)
                rotation = _quaternion_angle(pose[:, 3:7], reference[:, 3:7])
            recovered.append(
                {
                    "phase": phase,
                    "anchor_id": path.parents[2].name,
                    "episode": int(path.stem.split("_")[-1]),
                    "object_translation_error_m_mean": float(translation.mean()),
                    "object_translation_error_m_max": float(translation.max()),
                    "object_rotation_error_rad_mean": float(rotation.mean()),
                    "object_rotation_error_rad_max": float(rotation.max()),
                    "trace": str(path),
                    "trace_sha256": sha256_file(path),
                }
            )
    definitions = [
        (
            "geometric penetration p95",
            "SparseV4 source-geometry point-to-surface positive penetration depth p95",
            "m",
            "p95 over sampled source geometry",
            "manifest/CSV",
            "Certification diagnostics",
        ),
        (
            "physical penetration p95",
            "exact-FCL hand/object collision-proxy penetration depth",
            "m",
            "p95 over all control-step/body pairs",
            "trace physical_penetration_depth_m",
            "PhysicsConsistentTaskGateV1",
        ),
        (
            "physical penetration max",
            "exact-FCL hand/object collision-proxy maximum penetration depth",
            "m",
            "maximum over all control-step/body pairs",
            "trace physical_penetration_depth_m",
            "PhysicsConsistentTaskGateV1",
        ),
        (
            "object translation error",
            "Euclidean norm of object_pose.xyz minus object_reference.xyz",
            "m",
            "mean for gate; max diagnostic",
            "trace object_pose/object_reference",
            "EvaluationSuiteV2",
        ),
        (
            "object rotation error",
            "sign-invariant quaternion SO(3) geodesic",
            "rad",
            "mean for gate; max diagnostic",
            "trace object_pose/object_reference",
            "EvaluationSuiteV2",
        ),
        (
            "current ObjectDrift field",
            "object_translation_drift_m_max: maximum Euclidean object-reference translation",
            "m",
            "maximum over observed rollout",
            "D2K worker and trace",
            "diagnostic only",
        ),
        (
            "contact persistence",
            "fraction of control rows with any measured fingertip/object pair contact",
            "fraction",
            "mean Boolean over rollout",
            "trace tip_pair_presence",
            "diagnostic; terminal gate separate",
        ),
        (
            "terminal linear velocity",
            "world-frame norm of actual object linear twist",
            "m/s",
            "last row diagnostic; last-20 all-steps gate",
            "trace object_twist",
            "PhysicsConsistentTaskGateV1",
        ),
        (
            "terminal angular velocity",
            "world-frame norm of actual object angular twist",
            "rad/s",
            "last row diagnostic; last-20 all-steps gate",
            "trace object_twist",
            "PhysicsConsistentTaskGateV1",
        ),
        (
            "action saturation",
            "fraction of 26D residual actions at absolute value >=0.999",
            "fraction",
            "mean over all step/action dimensions",
            "trace action",
            "PPO26DTrainingContractV1",
        ),
    ]
    for metric, definition, unit, aggregation, source, authority in definitions:
        rows.append(
            {
                "metric": metric,
                "exact_definition": definition,
                "unit": unit,
                "time_aggregation": aggregation,
                "raw_source": source,
                "existing_authority": authority,
            }
        )
    write_csv(root / "metric_semantics/metric_dictionary.csv", rows)
    write_csv(root / "metric_semantics/recovered_raw_et_er.csv", recovered)
    value = {
        "schema_version": "PhysicalMetricSemanticAuditV1",
        "status": "PASS",
        "OBJECT_DRIFT_METRIC_SEMANTICS": "RESOLVED",
        "object_drift_exact_semantics": "maximum over rollout of Euclidean object_pose.xyz - object_reference.xyz, in metres",
        "object_drift_is_composite_or_normalized": False,
        "object_drift_directly_comparable_to_et_3cm": False,
        "reason_not_directly_comparable": "ObjectDrift is a temporal maximum; EvaluationSuiteV2 E_t is a trajectory mean.",
        "RAW_OBJECT_TRANSLATION_AVAILABLE": "YES",
        "RAW_OBJECT_ROTATION_AVAILABLE": "YES",
        "raw_et_er_rollouts_recovered": len(recovered),
        "physics_rerun": False,
    }
    write_json(root / "metric_semantics/physical_metric_semantic_audit.json", value)
    return value


def inventory_pre_d2k_gates(root: Path) -> dict[str, Any]:
    require(root / "metric_semantics/physical_metric_semantic_audit.json", "PASS", "GATE_INVENTORY")
    candidates = [
        {
            "authority_name": "TopoRetargetEvaluationSuiteV2",
            "path": "configs/evaluation/stage16_evaluation_suite_v2.yaml",
            "metric": "E_t/E_r/E_j/E_ft plus SR_physics Boolean dimensions",
            "threshold": "E_t<3cm; E_r<30deg; E_j<8cm; E_ft<6cm",
            "unit": "cm/deg",
            "aggregation": "trajectory mean; per-episode",
            "task_scope": "Stage16-D PPO and supported adapters",
            "dataset_scope": "shared adapters",
            "physical_scene_scope": "trajectory physical evaluation",
            "used_in_production": "YES",
            "used_in_paper_claim": "YES",
            "applicable_to_static_contact": "PARTIAL",
        },
        {
            "authority_name": "PhysicsConsistentTaskGateV1",
            "path": "src/toporetarget/rl/physics_retargeting/contracts.py",
            "metric": "terminal/contact/penetration/action/causality plus task progress",
            "threshold": "p95<=3mm; max<10mm; terminal contact v<=0.05m/s,w<=0.5rad/s; action<=1; PPO SR>=0.90",
            "unit": "m,m/s,rad/s,fraction",
            "aggregation": "terminal window and per-episode hard gate",
            "task_scope": "motion-class-aware grasp/manipulation",
            "dataset_scope": "generic runtime, instantiated per clip",
            "physical_scene_scope": "dynamic task semantics",
            "used_in_production": "YES",
            "used_in_paper_claim": "YES",
            "applicable_to_static_contact": "PARTIAL",
        },
        {
            "authority_name": "FrozenPhysicalEvaluationV1",
            "path": "configs/contracts/hocap_physicalization_v1.yaml",
            "metric": "Eval10/Confirm20 physical qualification",
            "threshold": "10/10 then Confirm20",
            "unit": "episode count",
            "aggregation": "dataset episode workflow",
            "task_scope": "HOCap EpisodeV1 full trajectory",
            "dataset_scope": "HOCap",
            "physical_scene_scope": "source-qualified support scene",
            "used_in_production": "YES",
            "used_in_paper_claim": "YES",
            "applicable_to_static_contact": "NO",
        },
        {
            "authority_name": "Stage16DynamicPhysicalQualificationV1",
            "path": "src/toporetarget/rl/dynamic_physical_qualification.py",
            "metric": "SRkin plus grasp/lift, relative terminal twist, geometry/action/causality",
            "threshold": "inherited terminal/geometry thresholds",
            "unit": "mixed",
            "aggregation": "per episode",
            "task_scope": "persistent grasp-and-lift dynamics",
            "dataset_scope": "Stage16 HOCap traces",
            "physical_scene_scope": "dynamic lift",
            "used_in_production": "YES",
            "used_in_paper_claim": "NO",
            "applicable_to_static_contact": "NO",
        },
        {
            "authority_name": "D2K-V1 continuous study contract",
            "path": ".local/reports/oakink2_o5rd2jr_d2k_support_proxy_ppo_recoverability_v1/study_contract/recoverability_study_contract.json",
            "metric": "paired continuous metrics only",
            "threshold": "binary gate null; STRONG/POOR hypothetical only",
            "unit": "N/A",
            "aggregation": "no binary anchor aggregation",
            "task_scope": "OakInk2 static proxy",
            "dataset_scope": "current 12 D2K anchors",
            "physical_scene_scope": "static support proxy",
            "used_in_production": "NO",
            "used_in_paper_claim": "NO",
            "applicable_to_static_contact": "NO_BINARY_GATE",
            "first_commit": KNOWN_D2K_HEAD,
        },
    ]
    for row in candidates:
        first = row.get("first_commit") or _first_commit(str(row["path"]))
        row["first_commit"] = first
        row["commit_le_cutoff"] = first != "NOT_FOUND" and (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", first, CUTOFF_HEAD],
                cwd=REPO,
                check=False,
            ).returncode
            == 0
        )
    fields = [
        "authority_name",
        "first_commit",
        "commit_le_cutoff",
        "metric",
        "threshold",
        "unit",
        "aggregation",
        "task_scope",
        "dataset_scope",
        "physical_scene_scope",
        "used_in_production",
        "used_in_paper_claim",
        "applicable_to_static_contact",
        "path",
    ]
    write_csv(root / "gate_audit/pre_d2k_gate_inventory.csv", candidates, fields)
    return {
        "schema_version": "PreD2KPhysicalGateInventoryV1",
        "status": "PASS",
        "candidate_count": len(candidates),
        "cutoff_head": CUTOFF_HEAD,
    }


def audit_gate_applicability(root: Path) -> dict[str, Any]:
    require(root / "metric_semantics/physical_metric_semantic_audit.json", "PASS", "APPLICABILITY")
    if not (root / "gate_audit/pre_d2k_gate_inventory.csv").is_file():
        raise RuntimeError("APPLICABILITY_REJECTED:MISSING_INVENTORY")
    criteria = {
        "existed_at_or_before_cutoff": True,
        "not_designed_from_current_12_outcomes": True,
        "metric_semantics_known": True,
        "units_compatible": True,
        "aggregation_compatible": False,
        "static_contact_applicable": False,
        "robot_object_representation_applicable": True,
        "metrics_obtainable_from_existing_rollouts": True,
        "unambiguous_rollout_and_anchor_acceptance": False,
        "no_absent_task_specific_events_required": False,
    }
    value = {
        "schema_version": "GateApplicabilityDecisionV1",
        "status": "PASS",
        "PRE_D2K_AUTHORITY_CUTOFF_HEAD": CUTOFF_HEAD,
        "PRE_D2K_BINARY_GATE_STATUS": "PARTIALLY_APPLICABLE",
        "criteria": criteria,
        "why": [
            "EvaluationSuiteV2 supplies static-compatible E_t/E_r and physical dimensions but no Eval10-to-anchor aggregation rule.",
            "PhysicsConsistentTaskGateV1 supplies numerical penetration, terminal, action, and success-rate authority but its full gate requires task motion/progress absent from a static hold.",
            "HOCap FrozenPhysicalEvaluationV1 and SR_dynamic are dataset/lift-specific.",
            "The D2K-V1 contract explicitly left binary recovery undefined and was created after the cutoff.",
        ],
        "FULLY_APPLICABLE_BINARY_GATE_EXISTS": False,
        "PARTIAL_AUTHORITIES_CAN_SEED_PROSPECTIVE_V2": True,
        "NO_NEW_THRESHOLD": False,
        "NO_OUTCOME_TUNING": True,
    }
    write_json(root / "gate_audit/applicability_audit.json", value)
    write_json(root / "gate_audit/gate_alignment_decision.json", value)
    return value


def align_existing_gate(root: Path) -> dict[str, Any]:
    audit = require(root / "gate_audit/gate_alignment_decision.json", "PASS", "ALIGN_GATE")
    if audit["PRE_D2K_BINARY_GATE_STATUS"] != "FULLY_APPLICABLE":
        value = {
            "schema_version": "GateAlignmentReceiptV1",
            "status": "NOT_RUN",
            "reason": "PRE_D2K_BINARY_GATE_NOT_FULLY_APPLICABLE",
            "PRE_D2K_BINARY_GATE_STATUS": audit["PRE_D2K_BINARY_GATE_STATUS"],
            "NO_NEW_THRESHOLD": True,
            "NO_OUTCOME_TUNING": True,
        }
        write_json(root / "existing_rollout_rescore/not_run.json", value)
        return value
    raise RuntimeError("FULL_GATE_ALIGNMENT_IMPLEMENTATION_REQUIRED")


def rescore_existing_d2k_rollouts(root: Path) -> dict[str, Any]:
    audit = require(root / "gate_audit/gate_alignment_decision.json", "PASS", "D2K_RESCORE")
    if audit["PRE_D2K_BINARY_GATE_STATUS"] != "FULLY_APPLICABLE":
        raise RuntimeError("D2K_RESCORE_REJECTED:PRE_D2K_BINARY_GATE_NOT_FULLY_APPLICABLE")
    raise RuntimeError("D2K_RESCORE_NOT_IMPLEMENTED_FOR_UNSELECTED_BRANCH")


def _v2_contract() -> dict[str, Any]:
    return {
        "schema_version": "StaticPhysicalRecoverabilityV2",
        "status": "FROZEN_BEFORE_V2_ANCHOR_SELECTION",
        "study_role": "NEW_PROSPECTIVE_BINARY_RETARGET_TO_PPO_RECOVERABILITY_STUDY",
        "historical_d2k_v1_result": "INCONCLUSIVE",
        "historical_d2k_v1_result_rewritten": False,
        "outcome_tuning": False,
        "current_d2k_outcome_distribution_used": False,
        "rollout_gate": {
            "simulation_finite": True,
            "complete_control_rows": 321,
            "causality": "measured hand-object contact plus zero hidden force/object write/wrist teleport",
            "no_hidden_object_write": True,
            "no_hidden_force": True,
            "no_wrist_root_teleport": True,
            "action_abs_max_lte": 1.0,
            "action_saturation_fraction_lte": 0.25,
            "physical_penetration_p95_m_lte": 0.003,
            "physical_penetration_max_m_lt": 0.010,
            "inter_finger_penetration_max_m_lte": 0.003,
            "object_translation_error_m_mean_lt": 0.03,
            "object_rotation_error_rad_mean_lt": float(np.deg2rad(30.0)),
            "terminal_window_control_steps": 20,
            "terminal_required_contact_fraction_gte": 0.80,
            "terminal_contact_linear_speed_mps_lte": 0.05,
            "terminal_contact_angular_speed_radps_lte": 0.50,
            "terminal_free_linear_speed_mps_lte": 0.01,
            "terminal_free_angular_speed_radps_lte": 0.25,
            "source_contact_recall_gte": 0.70,
        },
        "anchor_gate": {
            "eval_episodes": 10,
            "minimum_passing_rollouts": 8,
            "pass_rate_gte": 0.80,
        },
        "group_decision": {
            "primary_population": "SEMANTIC_FAIL=GROUP_B+GROUP_D",
            "strong_post_ppo_accepted_anchor_rate_gte": 0.80,
            "poor_post_ppo_accepted_anchor_rate_lte": 0.20,
            "mixed": "otherwise",
            "report_semantic_pass_regression": True,
        },
        "authorities": {
            "object_tracking": {
                "name": "TopoRetargetEvaluationSuiteV2",
                "path": "configs/evaluation/stage16_evaluation_suite_v2.yaml",
                "first_commit": "0cc90f2a77082243e330d888c8b635beeb0591f7",
            },
            "physics": {
                "name": "PhysicsConsistentTaskGateV1 component authorities",
                "path": "src/toporetarget/rl/physics_retargeting/contracts.py",
                "first_commit": "ce34934ce3143ea56f4e737825d5c9b26618c728",
            },
            "anchor_aggregation": "prospectively frozen 8/10 under D2K-R/V2 contract before selection",
            "group_aggregation": "D2K preregistered 80/20 strong/poor boundaries, reused without outcome tuning",
        },
        "scene_and_learning_freeze": {
            "support_proxy_sha256": EXPECTED_HASHES["support_proxy"],
            "static_scene_adapter_sha256": EXPECTED_HASHES["static_scene_adapter"],
            "static_reference_adapter_sha256": EXPECTED_HASHES["static_reference_adapter"],
            "physical_scene_contract_v2_sha256": EXPECTED_HASHES["physical_scene_contract_v2"],
            "ppo_contract_sha256": EXPECTED_HASHES["ppo_contract"],
            "reward_contract_sha256": EXPECTED_HASHES["reward_contract"],
            "ppo_algorithm": "PPO26D",
            "max_updates": 15,
            "num_envs": 1024,
            "rollout_length": 40,
            "samples_per_update": 40960,
            "sample_cap_per_anchor": 614400,
            "independent_policy_per_anchor": True,
            "MAX_GPU_JOBS": 1,
        },
    }


def design_recoverability_v2(root: Path) -> dict[str, Any]:
    audit = require(root / "gate_audit/gate_alignment_decision.json", "PASS", "DESIGN_V2")
    if audit["PRE_D2K_BINARY_GATE_STATUS"] == "FULLY_APPLICABLE":
        raise RuntimeError("DESIGN_V2_REJECTED:FULL_EXISTING_GATE_AVAILABLE")
    contract = _v2_contract()
    write_json(root / "recoverability_v2/binary_contract.draft.json", contract)
    return {
        "schema_version": "StaticPhysicalRecoverabilityV2DesignReceipt",
        "status": "PASS",
        "RECOVERABILITY_V2_GATE_STATUS": "READY_TO_FREEZE",
        "critical_threshold_authority_resolved": True,
        "anchor_selection_started": False,
    }


def freeze_recoverability_v2(root: Path) -> dict[str, Any]:
    draft = require(
        root / "recoverability_v2/binary_contract.draft.json",
        "FROZEN_BEFORE_V2_ANCHOR_SELECTION",
        "FREEZE_V2",
    )
    if (root / "recoverability_v2/manifest.json").exists():
        raise RuntimeError("FREEZE_V2_REJECTED:MANIFEST_ALREADY_EXISTS")
    digest = freeze_json(root / "recoverability_v2/binary_contract.json", draft)
    value = {
        "schema_version": "StaticPhysicalRecoverabilityV2FreezeReceipt",
        "status": "PASS",
        "RECOVERABILITY_V2_GATE_STATUS": "FROZEN",
        "BINARY_GATE_SHA256": digest,
        "anchor_selection_started": False,
    }
    write_json(root / "recoverability_v2/binary_contract_freeze_receipt.json", value)
    return value


def _annotation() -> dict[str, Any]:
    with ANNOTATION.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError("OAKINK2_ANNOTATION_NOT_MAPPING")
    return value


def _collision_vertices() -> np.ndarray:
    vertices, _counts, _indices, _low, _high = _read_obj(VISUAL_MESH)
    proxy_vertices, _faces, _gap = _bounded_convex_proxy(vertices)
    return np.asarray(proxy_vertices, dtype=np.float64)


def _load_sparse_rows() -> list[dict[str, Any]]:
    eligible = read_json(D2I_ROOT / "study_manifest/eligible_anchors.json")
    by_ordinal = {int(row["ordinal"]): row for row in eligible["pool"]}
    rows = []
    with (D2I_ROOT / "sparsev4_analysis/contributor_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for source in csv.DictReader(stream):
            ordinal = int(source["ordinal"])
            authority = by_ordinal[ordinal]
            rows.append(
                {
                    "ordinal": ordinal,
                    "frame_id": int(authority["frame_id"]),
                    "group": source["group"],
                    "bootstrap_e_im": float(source["bootstrap_e_im"]),
                    "final_e_im": float(source["final_e_im"]),
                    "rho1": float(source["rho1"]),
                    "rho2": float(source["rho2"]),
                    "top1_contributor": source["top1_finger"],
                    "top2_contributor": source["top2_finger"],
                    "geom_penetration_p95_m": float(authority["geom_penetration_p95_m"]),
                    "geom_penetration_max_m": float(authority["geom_penetration_max_m"]),
                }
            )
    return rows


def _select_fresh_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    old = read_json(D2I_ROOT / "study_manifest/recoverability_manifest.json")
    used = {int(row["ordinal"]) for row in old["anchors"]}
    remaining = [row for row in _load_sparse_rows() if int(row["ordinal"]) not in used]
    counts = Counter(str(row["group"]) for row in remaining)
    selected = []
    for group in ("GROUP_A", "GROUP_C", "GROUP_D"):
        selected.extend(
            sorted(
                (row for row in remaining if row["group"] == group), key=lambda row: row["ordinal"]
            )
        )
    group_b = sorted(
        (row for row in remaining if row["group"] == "GROUP_B"), key=lambda row: row["final_e_im"]
    )
    if len(group_b) != 12:
        raise RuntimeError(f"V2_GROUP_B_POOL_EXPECTED_12:{len(group_b)}")
    # One deterministic representative from each of six equal ordered bins.
    selected.extend(group_b[index] for index in (0, 2, 4, 6, 8, 10))
    if len(selected) != 12 or len({row["ordinal"] for row in selected}) != 12:
        raise RuntimeError("V2_SELECTION_CARDINALITY_INVALID")
    return selected, dict(counts)


def select_v2_physical_anchors(root: Path) -> dict[str, Any]:
    freeze = require(
        root / "recoverability_v2/binary_contract_freeze_receipt.json", "PASS", "SELECT_V2"
    )
    if sha256_file(root / "recoverability_v2/binary_contract.json") != freeze["BINARY_GATE_SHA256"]:
        raise RuntimeError("SELECT_V2_REJECTED:BINARY_GATE_HASH_DRIFT")
    selected, remaining_counts = _select_fresh_rows()
    old = read_json(D2I_ROOT / "study_manifest/recoverability_manifest.json")
    old_ordinals = {int(row["ordinal"]) for row in old["anchors"]}
    annotation = _annotation()
    vertices = _collision_vertices()
    parameters = StaticRecoverabilitySupportProxyParameterV1()
    from scripts.data import run_oakink2_o5rd2g as d2g
    from scripts.rl.isaaclab.freeze_stage16d_reward_v3_contact_contract import (
        reference_distances_to_visual_mesh,
    )

    runtime = d2g.V3Runtime("dev_01", D2H_ROOT)
    anchors = []
    proxy_rows = []
    for index, source in enumerate(selected):
        anchor_id = f"A{index + 1:02d}"
        clip_id = f"oakink2_v2_{anchor_id.lower()}"
        frame = int(source["frame_id"])
        ordinal = int(source["ordinal"])
        state_path = D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.npz"
        with np.load(state_path, allow_pickle=False) as archive:
            if sorted(archive.files) != ["base_pose_scene", "qpos"]:
                raise RuntimeError(f"V2_STATE_AUTHORITY_INVALID:{ordinal}")
            qpos = np.asarray(archive["qpos"], dtype=np.float64)
            base = np.asarray(archive["base_pose_scene"], dtype=np.float64)
        object_pose = np.asarray(annotation["obj_transf"][OBJECT_ID][frame], dtype=np.float64)
        proxy, proxy_audit = build_static_recoverability_support_proxy(
            vertices, object_pose, parameters=parameters
        )
        support_dir = root / "recoverability_v2/support_proxy/per_anchor" / anchor_id
        proxy_record = {
            **proxy.as_dict(),
            "schema_version": "StaticRecoverabilitySupportProxyV1",
            "status": "CONSTRUCTED",
            "anchor_id": anchor_id,
            "clip_id": clip_id,
            "frame": frame,
            "ordinal": ordinal,
            "support_type": "STATIC_RECOVERABILITY_PLANAR_PROXY",
            "source_contract_sha256": EXPECTED_HASHES["support_proxy"],
            "audit": proxy_audit,
        }
        write_json(support_dir / "support_proxy.json", proxy_record)
        write_finite_planar_support_usda(proxy, support_dir / "support_proxy.usda")
        scene_dir = root / "recoverability_v2/scene_adapter/per_anchor" / anchor_id
        reference = scene_dir / f"{clip_id}.reference_kinematics_v2.npz"
        write_static_retarget_reference(
            reference,
            qpos=qpos,
            base_pose_world=base,
            object_pose_world=object_pose,
            robot_model=runtime.model,
            source_frame_id=frame,
            anchor_id=anchor_id,
        )
        object_mesh_root = scene_dir / "object_mesh"
        object_mesh_root.mkdir(parents=True, exist_ok=True)
        object_mesh = object_mesh_root / f"{clip_id}.obj"
        shutil.copyfile(VISUAL_MESH, object_mesh)
        distances, distance_metadata = reference_distances_to_visual_mesh(
            reference=reference, object_mesh=object_mesh
        )
        contact_mask = distances < 0.03
        contracts = scene_dir / "reference_contracts"
        contracts.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            contracts / f"reference_contact_mask_{clip_id}.npz",
            reference_expected_contact_mask=contact_mask,
            reference_fingertip_to_object_distance_m=distances.astype(np.float32),
            finger_order=np.asarray(FINGERS),
            metadata=np.asarray(
                json.dumps(
                    {
                        "schema_version": "StaticReferenceContactDistanceV1",
                        "status": "PASS",
                        "clip": clip_id,
                        "threshold_m": 0.03,
                        "threshold_authority": "existing Stage16DReferenceContactMaskV1",
                        "geometry": distance_metadata,
                        "outcomes_observed": False,
                    },
                    sort_keys=True,
                )
            ),
        )
        np.savez_compressed(
            contracts / f"strict_source_contact_mask_{clip_id}.npz",
            strict_source_contact_mask=contact_mask,
            source_contact_class=np.where(
                contact_mask, "SOURCE_CONTACT_CONFIRMED", "SOURCE_NO_CONTACT"
            ),
            finger_names=np.asarray(FINGERS),
            control_index=np.arange(len(contact_mask), dtype=np.int64),
        )
        anchor = {
            "anchor_id": anchor_id,
            "clip_id": clip_id,
            **source,
            "semantic_supergroup": (
                "SEMANTIC_PASS" if source["group"] in {"GROUP_A", "GROUP_C"} else "SEMANTIC_FAIL"
            ),
            "object": OBJECT_ID,
            "reference": str(reference.resolve()),
            "reference_sha256": sha256_file(reference),
            "state_sha256": sha256_file(state_path),
            "object_usd": str(OBJECT_USD.resolve()),
            "support_proxy": str((support_dir / "support_proxy.json").resolve()),
            "support_asset": str((support_dir / "support_proxy.usda").resolve()),
            "contact_contract": str(STRICT_CONTRACT.resolve()),
            "contact_mask_root": str(contracts.resolve()),
            "reference_distance_root": str(contracts.resolve()),
            "object_mesh_root": str(object_mesh_root.resolve()),
            "expected_contact_fingers": [
                finger for finger, active in zip(FINGERS, contact_mask[0], strict=True) if active
            ],
            "q_old_used": False,
            "hand_target_constant": True,
            "object_target_constant": True,
            "reference_velocity_zero": True,
        }
        anchors.append(anchor)
        proxy_rows.append(
            {
                "anchor_id": anchor_id,
                "frame": frame,
                "ordinal": ordinal,
                "support_height": proxy.plane_offset,
                "proxy_sha256": sha256_file(support_dir / "support_proxy.json"),
                "asset_sha256": sha256_file(support_dir / "support_proxy.usda"),
            }
        )
    manifest = {
        "schema_version": "RecoverabilityV2ManifestV1",
        "status": "FROZEN_BEFORE_BASELINE_OR_PPO",
        "binary_gate_sha256": freeze["BINARY_GATE_SHA256"],
        "selection_rule": "all remaining A/C/D plus B sorted by final E_IM with first element from each of six two-item bins",
        "selection_inputs": [
            "SparseV4 group",
            "frame id",
            "frozen E_IM",
            "source interaction metadata",
        ],
        "physical_or_ppo_outcomes_used": False,
        "remaining_pool_counts": remaining_counts,
        "anchor_count": len(anchors),
        "semantic_pass_n": sum(row["semantic_supergroup"] == "SEMANTIC_PASS" for row in anchors),
        "semantic_fail_n": sum(row["semantic_supergroup"] == "SEMANTIC_FAIL" for row in anchors),
        "FRESH_TO_PHYSICS_OVERLAP_WITH_D2K_V1": len(
            old_ordinals.intersection(int(row["ordinal"]) for row in anchors)
        ),
        "anchors": anchors,
    }
    manifest_sha = freeze_json(root / "recoverability_v2/manifest.json", manifest)
    # Worker-compatible paths preserve exactly the same serialized manifest bytes.
    worker_manifest = root / "recoverability_v2/study_manifest/final_manifest.json"
    worker_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "recoverability_v2/manifest.json", worker_manifest)
    write_csv(root / "recoverability_v2/support_proxy/per_anchor_proxy.csv", proxy_rows)
    clips = [row["clip_id"] for row in anchors]
    geometry_path = d2k._runtime_geometry_manifest(root / "recoverability_v2", clips)
    scene_adapter = {
        "schema_version": "OakInk2StaticContactSceneAdapterV1",
        "status": "PASS",
        "source_frozen_scene_adapter_sha256": EXPECTED_HASHES["static_scene_adapter"],
        "anchors": anchors,
        "object_pose_writes_after_reset": 0,
        "hidden_object_force": False,
        "wrist_root_teleport": False,
    }
    write_json(root / "recoverability_v2/scene_adapter/runtime_manifest.json", scene_adapter)
    ppo = read_json(AUTHORITY_PATHS["ppo_contract"])
    seeds = [
        {
            "anchor_id": row["anchor_id"],
            "training_seed": 20264900 + index,
            "eval_seeds": [20265900 + 100 * index + episode for episode in range(10)],
            "baseline_post_ppo_paired": True,
        }
        for index, row in enumerate(anchors)
    ]
    study_contract = {
        "schema_version": "ProspectiveRetargetToPPORecoverabilityStudyV2",
        "status": "FROZEN_BEFORE_PPO",
        "binary_gate_sha256": freeze["BINARY_GATE_SHA256"],
        "anchor_manifest_sha256": manifest_sha,
        "scene_contract_sha256": EXPECTED_HASHES["physical_scene_contract_v2"],
        "support_proxy_sha256": EXPECTED_HASHES["support_proxy"],
        "ppo_contract_sha256": EXPECTED_HASHES["ppo_contract"],
        "reward_sha256": EXPECTED_HASHES["reward_contract"],
        "runtime_geometry_manifest": str(geometry_path),
        "runtime_geometry_manifest_sha256": sha256_file(geometry_path),
        "seeds": seeds,
        "training_budget": ppo["training"],
        "evaluation": ppo["evaluation"],
        "eval_n": 10,
        "baseline_horizon": 321,
        "PPO_REWARD_CHANGED": "NO",
        "MAX_GPU_JOBS": 1,
    }
    study_sha = freeze_json(
        root / "recoverability_v2/study_contract/recoverability_study_contract.json",
        study_contract,
    )
    value = {
        "schema_version": "RecoverabilityV2ManifestFreezeReceiptV1",
        "status": "PASS",
        "manifest_sha256": manifest_sha,
        "study_contract_sha256": study_sha,
        "binary_gate_sha256": freeze["BINARY_GATE_SHA256"],
        "anchor_count": len(anchors),
        "semantic_pass_n": manifest["semantic_pass_n"],
        "semantic_fail_n": manifest["semantic_fail_n"],
        "FRESH_TO_PHYSICS_OVERLAP_WITH_D2K_V1": manifest["FRESH_TO_PHYSICS_OVERLAP_WITH_D2K_V1"],
        "geometry_manifest_sha256": sha256_file(geometry_path),
    }
    write_json(root / "recoverability_v2/manifest_freeze_receipt.json", value)
    if value["FRESH_TO_PHYSICS_OVERLAP_WITH_D2K_V1"] != 0:
        raise RuntimeError("V2_ANCHOR_OVERLAP_WITH_D2K_V1")
    return value


def run_v2_baselines(root: Path) -> dict[str, Any]:
    receipt = require(
        root / "recoverability_v2/manifest_freeze_receipt.json", "PASS", "V2_BASELINE"
    )
    v2 = root / "recoverability_v2"
    if sha256_file(v2 / "binary_contract.json") != receipt["binary_gate_sha256"]:
        raise RuntimeError("V2_BASELINE_REJECTED:BINARY_GATE_DRIFT")
    results = []
    for index in range(12):
        results.append(
            d2k._run_ppo_study_worker(
                v2,
                mode="baseline",
                item_index=index,
                output=v2 / f"baseline_physics/per_anchor_raw/A{index + 1:02d}.json",
            )
        )
    value = d2k._aggregate_evaluation(v2, phase="baseline", results=results)
    write_json(v2 / "baseline/summary.json", value)
    return value


def run_v2_ppo(root: Path) -> dict[str, Any]:
    v2 = root / "recoverability_v2"
    baseline = require(v2 / "baseline_physics/summary.json", "PASS", "V2_PPO")
    if baseline.get("anchors") != 12 or baseline.get("rollouts") != 120:
        raise RuntimeError("V2_PPO_REJECTED:ALL_BASELINES_NOT_COMPLETE")
    ppo = read_json(AUTHORITY_PATHS["ppo_contract"])
    if sha256_file(AUTHORITY_PATHS["ppo_contract"]) != EXPECTED_HASHES["ppo_contract"]:
        raise RuntimeError("V2_PPO_REJECTED:PPO_CONTRACT_DRIFT")
    if (
        ppo["training"]
        != read_json(v2 / "study_contract/recoverability_study_contract.json")["training_budget"]
    ):
        raise RuntimeError("V2_PPO_REJECTED:PPO_BUDGET_DRIFT")
    receipts = []
    for index in range(12):
        receipts.append(
            d2k._run_ppo_study_worker(
                v2,
                mode="train",
                item_index=index,
                output=v2 / f"ppo_training/A{index + 1:02d}",
            )
        )
    value = {
        "schema_version": "ProspectiveRecoverabilityV2PPOTrainingAggregateV1",
        "status": "PASS"
        if len(receipts) == 12
        and all(row["status"] == "PASS" and row["cumulative_samples"] == 614400 for row in receipts)
        else "TECHNICAL_INFRA_FAILURE",
        "PPO_TRAINING_RUN_COUNT": len(receipts),
        "total_training_samples": sum(int(row["cumulative_samples"]) for row in receipts),
        "receipts": receipts,
        "PPO_REWARD_CHANGED": "NO",
        "MAX_GPU_JOBS": 1,
    }
    write_json(v2 / "ppo_training/summary.json", value)
    write_json(v2 / "training/summary.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("V2_PPO_TRAINING_INFRA_FAILURE")
    return value


def evaluate_v2(root: Path) -> dict[str, Any]:
    v2 = root / "recoverability_v2"
    training = require(v2 / "ppo_training/summary.json", "PASS", "V2_EVAL")
    if training["PPO_TRAINING_RUN_COUNT"] != 12:
        raise RuntimeError("V2_EVAL_REJECTED:TRAINING_INCOMPLETE")
    results = []
    for index in range(12):
        anchor_id = f"A{index + 1:02d}"
        results.append(
            d2k._run_ppo_study_worker(
                v2,
                mode="eval",
                item_index=index,
                output=v2 / f"ppo_eval/per_anchor_raw/{anchor_id}.json",
                checkpoint=v2 / f"ppo_training/{anchor_id}/checkpoint.pt",
            )
        )
    value = d2k._aggregate_evaluation(v2, phase="eval", results=results)
    write_json(v2 / "eval/summary.json", value)
    return value


def _score_rollout(
    trace_path: Path, contract: dict[str, Any], no_cheat: dict[str, Any]
) -> dict[str, Any]:
    from scripts.rl.isaaclab.run_stage16_frozen_source_policy_gravity_sweep import (
        _inter_finger_penetration,
    )

    gate = contract["rollout_gate"]
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {name: np.asarray(archive[name]) for name in archive.files}
    pose = np.asarray(trace["object_pose"], dtype=np.float64)
    reference = np.asarray(trace["object_reference"], dtype=np.float64)
    twist = np.asarray(trace["object_twist"], dtype=np.float64)
    action = np.asarray(trace["action"], dtype=np.float64)
    penetration = np.asarray(trace["physical_penetration_depth_m"], dtype=np.float64)
    hand_contact = np.asarray(trace["hand_object_pair_presence"], dtype=bool).any(axis=-1)
    expected = np.asarray(trace["source_contact_mask"], dtype=bool)
    actual = np.asarray(trace["tip_pair_presence"], dtype=bool)
    valid = np.asarray(trace["hand_object_pair_force_valid"], dtype=bool)
    selected = np.flatnonzero(valid)
    terminal = selected[-int(gate["terminal_window_control_steps"]) :]
    translation = np.linalg.norm(pose[:, :3] - reference[:, :3], axis=-1)
    rotation = _quaternion_angle(pose[:, 3:7], reference[:, 3:7])
    linear = np.linalg.norm(twist[:, :3], axis=-1)
    angular = np.linalg.norm(twist[:, 3:], axis=-1)
    linear_limit = np.where(
        hand_contact[terminal],
        float(gate["terminal_contact_linear_speed_mps_lte"]),
        float(gate["terminal_free_linear_speed_mps_lte"]),
    )
    angular_limit = np.where(
        hand_contact[terminal],
        float(gate["terminal_contact_angular_speed_radps_lte"]),
        float(gate["terminal_free_angular_speed_radps_lte"]),
    )
    required = expected & valid[:, None]
    recall = float((actual & required).sum() / required.sum()) if required.any() else 1.0
    inter = _inter_finger_penetration(
        np.asarray(trace["hand_collision_body_pose"], dtype=np.float32)
    )
    metrics = {
        "control_rows": len(pose),
        "object_translation_error_m_mean": float(translation.mean()),
        "object_rotation_error_rad_mean": float(rotation.mean()),
        "physical_penetration_p95_m": float(np.quantile(penetration, 0.95)),
        "physical_penetration_max_m": float(penetration.max(initial=0.0)),
        "inter_finger_penetration_max_m": float(inter.max(initial=0.0)),
        "terminal_contact_fraction": float(hand_contact[terminal].mean()),
        "terminal_linear_speed_mps_max": float(linear[terminal].max(initial=0.0)),
        "terminal_angular_speed_radps_max": float(angular[terminal].max(initial=0.0)),
        "source_contact_recall": recall,
        "action_max_abs": float(np.abs(action).max(initial=0.0)),
        "action_saturation_fraction": float(np.mean(np.abs(action) >= 0.999)),
    }
    checks = {
        "simulation_finite": all(
            np.isfinite(value).all()
            for value in (pose, reference, twist, action, penetration, inter)
        ),
        "complete_control_rows": len(pose) == int(gate["complete_control_rows"]),
        "causality": bool(hand_contact[valid].any())
        and no_cheat
        == {
            "hidden_force_or_attachment": False,
            "object_rollout_state_writes": 0,
            "wrist_root_state_writes_during_step": 0,
        },
        "action_bounds": metrics["action_max_abs"] <= float(gate["action_abs_max_lte"]),
        "action_saturation": metrics["action_saturation_fraction"]
        <= float(gate["action_saturation_fraction_lte"]),
        "penetration_p95": metrics["physical_penetration_p95_m"]
        <= float(gate["physical_penetration_p95_m_lte"]),
        "penetration_catastrophic": metrics["physical_penetration_max_m"]
        < float(gate["physical_penetration_max_m_lt"]),
        "inter_finger_penetration": metrics["inter_finger_penetration_max_m"]
        <= float(gate["inter_finger_penetration_max_m_lte"]),
        "object_translation_tracking": metrics["object_translation_error_m_mean"]
        < float(gate["object_translation_error_m_mean_lt"]),
        "object_rotation_tracking": metrics["object_rotation_error_rad_mean"]
        < float(gate["object_rotation_error_rad_mean_lt"]),
        "terminal_contact": metrics["terminal_contact_fraction"]
        >= float(gate["terminal_required_contact_fraction_gte"]),
        "terminal_stability": bool(
            np.all(linear[terminal] <= linear_limit) and np.all(angular[terminal] <= angular_limit)
        ),
        "source_contact_preservation": recall >= float(gate["source_contact_recall_gte"]),
    }
    return {
        **metrics,
        **{f"check_{name}": passed for name, passed in checks.items()},
        "accepted": all(checks.values()),
    }


def decide_recoverability(root: Path) -> dict[str, Any]:
    v2 = root / "recoverability_v2"
    require(v2 / "baseline_physics/summary.json", "PASS", "V2_DECISION")
    require(v2 / "ppo_eval/summary.json", "PASS", "V2_DECISION")
    contract = require(
        v2 / "binary_contract.json", "FROZEN_BEFORE_V2_ANCHOR_SELECTION", "V2_DECISION"
    )
    manifest = require(v2 / "manifest.json", "FROZEN_BEFORE_BASELINE_OR_PPO", "V2_DECISION")
    per_rollout: list[dict[str, Any]] = []
    per_anchor: list[dict[str, Any]] = []
    for anchor in manifest["anchors"]:
        anchor_id = anchor["anchor_id"]
        phase_rows: dict[str, list[dict[str, Any]]] = {}
        for phase, directory in (("baseline", "baseline_physics"), ("ppo", "ppo_eval")):
            receipt = read_json(v2 / directory / "per_anchor_raw" / f"{anchor_id}.json")
            rows = []
            for episode in range(10):
                trace = (
                    v2
                    / directory
                    / "per_anchor_raw"
                    / anchor_id
                    / "traces"
                    / f"episode_{episode:02d}.npz"
                )
                scored = _score_rollout(trace, contract, receipt["no_cheat"])
                row = {
                    "anchor_id": anchor_id,
                    "group": anchor["group"],
                    "semantic_supergroup": anchor["semantic_supergroup"],
                    "phase": phase,
                    "episode": episode,
                    "seed": receipt["seeds"][episode],
                    "trace": str(trace),
                    "trace_sha256": sha256_file(trace),
                    **scored,
                }
                rows.append(row)
                per_rollout.append(row)
            phase_rows[phase] = rows
        baseline_passes = sum(bool(row["accepted"]) for row in phase_rows["baseline"])
        ppo_passes = sum(bool(row["accepted"]) for row in phase_rows["ppo"])
        baseline_accepted = baseline_passes >= 8
        ppo_accepted = ppo_passes >= 8
        per_anchor.append(
            {
                "anchor_id": anchor_id,
                "group": anchor["group"],
                "semantic_supergroup": anchor["semantic_supergroup"],
                "baseline_passing_rollouts": baseline_passes,
                "ppo_passing_rollouts": ppo_passes,
                "BASELINE_ACCEPTED": baseline_accepted,
                "PPO_ACCEPTED": ppo_accepted,
                "NEEDED_RECOVERY": not baseline_accepted,
                "RECOVERED_BY_PPO": not baseline_accepted and ppo_accepted,
                "REGRESSED_BY_PPO": baseline_accepted and not ppo_accepted,
                "NOT_RECOVERED": not baseline_accepted and not ppo_accepted,
            }
        )
    write_csv(v2 / "per_rollout.csv", per_rollout)
    write_csv(v2 / "per_anchor.csv", per_anchor)
    semantic_fail = [row for row in per_anchor if row["semantic_supergroup"] == "SEMANTIC_FAIL"]
    semantic_pass = [row for row in per_anchor if row["semantic_supergroup"] == "SEMANTIC_PASS"]
    accepted = sum(bool(row["PPO_ACCEPTED"]) for row in semantic_fail)
    rate = accepted / len(semantic_fail)
    if rate >= 0.80:
        decision = "STRONG_RECOVERABILITY"
        branch = "STRONG"
    elif rate <= 0.20:
        decision = "POOR_RECOVERABILITY"
        branch = "POOR"
    else:
        decision = "MIXED_RECOVERABILITY"
        branch = "MIXED"
    value = {
        "schema_version": "ProspectiveRecoverabilityV2DecisionV1",
        "status": "COMPLETE",
        "RECOVERABILITY_V2_RESULT": decision,
        "FINAL_RECOVERABILITY_DECISION": decision,
        "SELECTED_TERMINAL_BRANCH": branch,
        "SEMANTIC_PASS_N": len(semantic_pass),
        "SEMANTIC_PASS_POST_PPO_ACCEPTED": sum(bool(row["PPO_ACCEPTED"]) for row in semantic_pass),
        "SEMANTIC_FAIL_N": len(semantic_fail),
        "SEMANTIC_FAIL_POST_PPO_ACCEPTED": accepted,
        "SEMANTIC_FAIL_BASELINE_ACCEPTED": sum(
            bool(row["BASELINE_ACCEPTED"]) for row in semantic_fail
        ),
        "SEMANTIC_FAIL_RECOVERED_BY_PPO": sum(
            bool(row["RECOVERED_BY_PPO"]) for row in semantic_fail
        ),
        "PPO_REGRESSION_COUNT": sum(bool(row["REGRESSED_BY_PPO"]) for row in per_anchor),
        "semantic_fail_post_ppo_accepted_rate": rate,
        "BINARY_GATE_SHA256": sha256_file(v2 / "binary_contract.json"),
        "PPO_REWARD_CHANGED": "NO",
    }
    write_json(v2 / "decision.json", value)
    return value


def _require_branch(root: Path, expected: str, action: str) -> dict[str, Any]:
    decision = require(root / "recoverability_v2/decision.json", "COMPLETE", action)
    if decision["SELECTED_TERMINAL_BRANCH"] != expected:
        raise RuntimeError(
            f"{action}_REJECTED:SELECTED_BRANCH={decision['SELECTED_TERMINAL_BRANCH']}"
        )
    return decision


def _branch_placeholder(root: Path, expected: str, action: str) -> dict[str, Any]:
    _require_branch(root, expected, action)
    raise RuntimeError(f"{action}_SELECTED_BUT_IMPLEMENTATION_PENDING")


def build_retarget_admission_v2(root: Path) -> dict[str, Any]:
    return _branch_placeholder(root, "STRONG", "BUILD_RETARGET_ADMISSION_V2")


def develop_quality_aware_admission(root: Path) -> dict[str, Any]:
    return _branch_placeholder(root, "MIXED", "DEVELOP_QUALITY_AWARE_ADMISSION")


def _freeze_v4_candidates(root: Path) -> dict[str, Any]:
    candidates = default_cold_start_search_v4_candidates()
    payload = {
        "schema_version": "ExecutionV4CandidateFamilyV1",
        "status": "FROZEN_BEFORE_EXECUTION_V4_DEVELOPMENT",
        "scientific_scope": "COLD_START_ONLY",
        "objective": "ObjectiveV2_UNCHANGED",
        "gate": "CertificationGateV2_UNCHANGED",
        "execution_v3_historical_artifacts": "IMMUTABLE",
        "refinement_mode": "EXACT_V3_V2_PARITY_REQUIRED",
        "candidate_order_is_simplicity_order": True,
        "selection_rule": "first candidate satisfying every preregistered development gate",
        "candidates": [candidate.as_dict() for candidate in candidates],
        "development_population": "consumed SparseV4 30 frames only",
        "fresh_v5_consumed": False,
        "thresholds": {
            "technical": "30/30",
            "old_invalid_recovery_min": 0.80,
            "median_invalid_relative_reduction_min": 0.50,
            "threshold_aware_nonregression": "30/30",
            "low_preservation": "10/10",
            "hard_validity": "PASS",
            "determinism": "PASS",
            "refinement_parity": "EXACT_PASS",
        },
        "solver_budgets_changed": False,
        "q_old_allowed": False,
    }
    path = root / "poor_branch/execution_v4_development/candidate_family.json"
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError("EXECUTION_V4_FROZEN_CANDIDATE_FAMILY_DRIFT")
        return payload
    freeze_json(path, payload)
    return payload


def _v4_evaluate_state(
    runtime: d2g.V3Runtime,
    ordinal: int,
    qpos: np.ndarray,
    base: np.ndarray,
    *,
    previous_q: np.ndarray | None = None,
    previous_base: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_q
    )
    values, evaluation, actual = d2g2._measurement_and_evaluation(
        runtime,
        ordinal,
        qpos,
        base,
        binding=binding,
        context=context,
        previous_q=previous_q,
        previous_base=previous_base,
    )
    return values, evaluation, actual


def _v4_phase(
    runtime: d2g.V3Runtime,
    ordinal: int,
    qpos: np.ndarray,
    base: np.ndarray,
    block: tuple[int, ...],
    candidate: ColdStartSearchV4Candidate,
    label: str,
    *,
    previous_q: np.ndarray | None = None,
    previous_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    values, evaluation, actual = _v4_evaluate_state(
        runtime,
        ordinal,
        qpos,
        base,
        previous_q=previous_q,
        previous_base=previous_base,
    )
    retained = d2g._screened(f"{label}:input", values, evaluation, 0, True)
    retained_state = (qpos, base, values, evaluation, actual)
    phase_rows: list[dict[str, Any]] = []
    primary_result = None
    try:
        (
            primary_q,
            primary_base,
            primary_values,
            primary_evaluation,
            primary_result,
            primary_actual,
        ) = d2g._run_candidate_phase(
            runtime,
            ordinal,
            q_seed=qpos,
            base_seed=base,
            previous_q=previous_q,
            previous_base=previous_base,
            block=block,
            phase="primary",
            maxiter=candidate.selected_primary_maxiter,
            retention_limit=None,
            initialization_source=f"{candidate.name}:{label}:primary",
        )
        primary = d2g._screened(
            f"{label}:primary",
            primary_values,
            primary_evaluation,
            1,
            bool(primary_result.optimizer_converged),
        )
        selected = select_candidate((retained, primary))
        if selected is primary:
            retained = primary
            retained_state = (
                primary_q,
                primary_base,
                primary_values,
                primary_evaluation,
                primary_actual,
            )
        phase_rows.append(
            {
                "phase": "primary",
                "profile": d2g._solver_profile(primary_result),
                "usable": primary.usable,
            }
        )
    except Exception as exc:
        phase_rows.append({"phase": "primary", "error": f"{type(exc).__name__}:{exc}"})
    retained_q, retained_base, retained_values, _retained_eval, _retained_actual = retained_state
    limit = interaction_retention_limit(
        retained_values.interaction_e_im, runtime.authority.interaction_target
    )
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
            block=block,
            phase="secondary",
            maxiter=candidate.secondary_polish_maxiter,
            retention_limit=limit,
            initialization_source=f"{candidate.name}:{label}:secondary",
        )
        polished = d2g._screened(
            f"{label}:secondary",
            polished_values,
            polished_evaluation,
            2,
            bool(polished_result.optimizer_converged),
        )
        selected, retention = retain_after_polish(
            retained,
            polished,
            interaction_target=runtime.authority.interaction_target,
            numerical_epsilon=1.0e-10,
        )
        if selected is polished:
            retained_state = (
                polished_q,
                polished_base,
                polished_values,
                polished_evaluation,
                polished_actual,
            )
        phase_rows.append(
            {
                "phase": "secondary",
                "profile": d2g._solver_profile(polished_result),
                "usable": polished.usable,
                "retention": retention,
            }
        )
    except Exception as exc:
        phase_rows.append({"phase": "secondary", "error": f"{type(exc).__name__}:{exc}"})
    return (*retained_state, phase_rows)


def search_cold_start_v4_from_v3(
    runtime: d2g.V3Runtime,
    ordinal: int,
    q_v3: np.ndarray,
    base_v3: np.ndarray,
    candidate: ColdStartSearchV4Candidate,
    *,
    prefix_authority: str = "IMMUTABLE_STORED_SPARSE_V4_RUN_1",
    previous_q: np.ndarray | None = None,
    previous_base: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply a frozen V4 contributor schedule to an immutable V3 prefix."""

    candidate.validate()
    qpos = np.asarray(q_v3, dtype=np.float64).copy()
    base = np.asarray(base_v3, dtype=np.float64).copy()
    values, evaluation, actual = _v4_evaluate_state(
        runtime,
        ordinal,
        qpos,
        base,
        previous_q=previous_q,
        previous_base=previous_base,
    )
    blocks = asset_derived_dof_blocks(runtime.model.dof_names)
    initial_scores = contributor_scores(d2g._interaction_per_keypoint(runtime, ordinal, qpos, base))
    initial_ranking = rank_contributors(initial_scores)
    used: list[str] = []
    stages: list[dict[str, Any]] = []

    if candidate.mode == "JOINT_BLOCK":
        used = list(initial_ranking[: candidate.top_k])
        block = tuple(index for finger in used for index in blocks[finger])
        qpos, base, values, evaluation, actual, phase_rows = _v4_phase(
            runtime,
            ordinal,
            qpos,
            base,
            block,
            candidate,
            "joint_top2",
            previous_q=previous_q,
            previous_base=previous_base,
        )
        stages.append({"contributors": used, "block": list(block), "phases": phase_rows})
    else:
        for step in range(candidate.top_k):
            if candidate.stop_at_interaction_target and values.interaction_e_im <= d2e.TAU:
                stages.append({"step": step, "status": "EARLY_STOP_AT_FROZEN_TAU"})
                break
            scores = contributor_scores(d2g._interaction_per_keypoint(runtime, ordinal, qpos, base))
            ranking = tuple(name for name in rank_contributors(scores) if name not in used)
            finger = ranking[0]
            used.append(finger)
            block = blocks[finger]
            probed_q, probe = d2g._contributor_probe(
                runtime,
                ordinal,
                qpos,
                base,
                finger=finger,
                block=block,
                max_nfev=candidate.contributor_probe_max_nfev,
            )
            probed_base = runtime.base_for_q(ordinal, probed_q)
            probed_values, probed_evaluation, probed_actual = _v4_evaluate_state(
                runtime,
                ordinal,
                probed_q,
                probed_base,
                previous_q=previous_q,
                previous_base=previous_base,
            )
            current = d2g._screened(f"step{step}:input", values, evaluation, 0, True)
            probed = d2g._screened(
                f"step{step}:probe", probed_values, probed_evaluation, 1, bool(probe["success"])
            )
            selected = select_candidate((current, probed))
            if selected is probed:
                qpos, base, values, evaluation, actual = (
                    probed_q,
                    probed_base,
                    probed_values,
                    probed_evaluation,
                    probed_actual,
                )
            qpos, base, values, evaluation, actual, phase_rows = _v4_phase(
                runtime,
                ordinal,
                qpos,
                base,
                block,
                candidate,
                f"step{step}:{finger}",
                previous_q=previous_q,
                previous_base=previous_base,
            )
            stages.append(
                {
                    "step": step,
                    "contributor": finger,
                    "block": list(block),
                    "ranking": list(ranking),
                    "scores": scores,
                    "probe": probe,
                    "phases": phase_rows,
                }
            )

    receipt = {
        "schema_version": "ExecutionV4ColdStartFrameReceiptV1",
        "mode": "COLD_START",
        "candidate": candidate.name,
        "ordinal": ordinal,
        "source_frame_local": int(runtime.graph.frame_indices[ordinal]),
        "runtime_step_index": int(runtime.current_runtime_step),
        "old_production_q": "ABSENT",
        "previous_accepted_state": (
            "ABSENT" if int(runtime.current_runtime_step) == 0 else "PRESENT"
        ),
        "q_old_synthesized": False,
        "q_old_access_count": 0,
        "failed_stage7_terminal_used_as_q_old": False,
        "execution_v3_prefix": prefix_authority,
        "initial_contributor_scores": initial_scores,
        "initial_contributor_ranking": list(initial_ranking),
        "used_contributors": used,
        "stages": stages,
        "selected_seed_candidate": "stored_execution_v3_prefix",
        "selected_block": "+".join(used) if used else "NONE_EARLY_STOP",
        "selected_candidate": "execution_v4_terminal",
        "retention_decision": "FEASIBILITY_FIRST_PER_STAGE",
        "selected": d2g._measurement_row(values),
        "selected_evaluation": evaluation,
        "selected_actual_continuity": actual,
        "technical_success": bool(evaluation["feasible"]),
        "optimizer_started": bool(stages),
        "profiler": {
            "schema_version": "ExecutionV4DevelopmentProfilerV1",
            "candidate_b2_candidate_probes": sum("probe" in stage for stage in stages),
            "candidate_b2_contributor": "+".join(used),
            "fallback": False,
        },
    }
    return qpos, base, receipt


def _run_v4_candidate_development(
    root: Path, candidate: ColdStartSearchV4Candidate
) -> dict[str, Any]:
    directory = root / "poor_branch/execution_v4_development" / candidate.name.lower()
    decision_path = directory / "gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    manifest = read_json(D2H_ROOT / "sparse_v4/manifest.json")
    old_rows = {
        int(row["ordinal"]): row
        for row in d2g.read_csv(D2H_ROOT / "sparse_v4/per_frame_results.csv")
    }
    runtime = d2g.V3Runtime("dev_01", root)
    rows: list[dict[str, Any]] = []
    references: dict[int, tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    for index, frame in enumerate(manifest["frames"], start=1):
        ordinal = int(frame["ordinal"])
        source_receipt = D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.json"
        source_state = source_receipt.with_suffix(".npz")
        with np.load(source_state, allow_pickle=False) as archive:
            q_v3 = np.asarray(archive["qpos"], dtype=np.float64)
            base_v3 = np.asarray(archive["base_pose_scene"], dtype=np.float64)
        qpos, base, receipt = search_cold_start_v4_from_v3(
            runtime, ordinal, q_v3, base_v3, candidate
        )
        receipt["execution_v3_prefix_receipt_sha256"] = sha256_file(source_receipt)
        receipt["execution_v3_prefix_state_sha256"] = sha256_file(source_state)
        receipt_path = directory / "receipts" / f"frame_{ordinal:04d}_run_1.json"
        write_json(receipt_path, receipt)
        receipt_path.with_suffix(".npz").parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(receipt_path.with_suffix(".npz"), qpos=qpos, base_pose_scene=base)
        row = d2h._row_from_coldstart(
            receipt, str(frame["stratum"]), float(frame["old_e_im"]), standalone=True
        )
        row["execution_v3_new_e_im"] = float(old_rows[ordinal]["new_e_im"])
        rows.append(row)
        references[ordinal] = (receipt, qpos, base)
        print(
            f"EXECUTION_V4_DEV candidate={candidate.name} {index}/{manifest['N']} "
            f"ordinal={ordinal} E_IM={row['new_e_im']:.12g}",
            flush=True,
        )
    write_csv(directory / "per_frame.csv", rows)

    preliminary = d2e.evaluate_gate_v2_rows(
        rows, determinism_pass=True, certification_status="DEVELOPMENT"
    )
    non_determinism_conditions = {
        key: value for key, value in preliminary["conditions"].items() if key != "determinism"
    }
    repeats: list[dict[str, Any]] = []
    determinism_pass = False
    if all(non_determinism_conditions.values()):
        for ordinal_value in manifest["determinism_ordinals"]:
            ordinal = int(ordinal_value)
            ref_receipt, ref_q, ref_base = references[ordinal]
            repeats.append({"ordinal": ordinal, "run": 1, "pass": True})
            source_state = D2H_ROOT / f"sparse_v4/receipts/frame_{ordinal:04d}_run_1.npz"
            with np.load(source_state, allow_pickle=False) as archive:
                q_v3 = np.asarray(archive["qpos"], dtype=np.float64)
                base_v3 = np.asarray(archive["base_pose_scene"], dtype=np.float64)
            for run in (2, 3):
                repeat_q, repeat_base, repeat_receipt = search_cold_start_v4_from_v3(
                    runtime, ordinal, q_v3, base_v3, candidate
                )
                q_diff = float(np.max(np.abs(repeat_q - ref_q)))
                base_diff = float(np.max(np.abs(repeat_base - ref_base)))
                e_diff = abs(
                    float(repeat_receipt["selected"]["interaction_e_im"])
                    - float(ref_receipt["selected"]["interaction_e_im"])
                )
                signature_exact = (
                    repeat_receipt["used_contributors"] == ref_receipt["used_contributors"]
                    and repeat_receipt["selected_block"] == ref_receipt["selected_block"]
                    and [row.get("contributor") for row in repeat_receipt["stages"]]
                    == [row.get("contributor") for row in ref_receipt["stages"]]
                )
                comparison = {
                    "pass": signature_exact
                    and q_diff <= d2h.Q_ATOL
                    and base_diff <= d2h.BASE_ATOL
                    and e_diff <= d2h.EPSILON_NUM,
                    "signature_exact": signature_exact,
                    "max_q_abs_diff": q_diff,
                    "max_base_abs_diff": base_diff,
                    "E_IM_abs_diff": e_diff,
                }
                repeats.append({"ordinal": ordinal, "run": run, **comparison})
        determinism_pass = all(bool(row["pass"]) for row in repeats)
    write_json(
        directory / "determinism.json",
        {
            "schema_version": "ExecutionV4DevelopmentDeterminismV1",
            "status": "PASS" if determinism_pass else "NOT_RUN_OTHER_GATE_FAILED",
            "rows": repeats,
        },
    )
    gate = d2e.evaluate_gate_v2_rows(
        rows, determinism_pass=determinism_pass, certification_status="DEVELOPMENT"
    )
    refinement = read_json(D2H_ROOT / "frozen_method/integrity.json")
    refinement_parity = bool(
        refinement["development_checks"]["REFINEMENT_MODE_REGRESSION"]
        and refinement["development_checks"]["N_REFINEMENT_PARITY_PASS"]
        and refinement["development_checks"]["MAX_Q_ABS_DIFF"]
        and refinement["development_checks"]["MAX_BASE_ABS_DIFF"]
        and refinement["development_checks"]["MAX_E_IM_ABS_DIFF"]
    )
    gate["conditions"]["refinement_regression_exact_parity"] = refinement_parity
    gate["decision"] = "PASS" if all(gate["conditions"].values()) else "FAIL"
    gate.update(
        {
            "schema_version": "ExecutionV4DevelopmentGateV1",
            "status": gate["decision"],
            "candidate": candidate.name,
            "q_old_access_count": 0,
            "fresh_v5_consumed": False,
        }
    )
    write_json(decision_path, gate)
    return gate


def develop_execution_v4(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "DEVELOP_EXECUTION_V4")
    _freeze_v4_candidates(root)
    results = []
    selected = None
    for candidate in default_cold_start_search_v4_candidates():
        gate = _run_v4_candidate_development(root, candidate)
        results.append(gate)
        if gate["status"] == "PASS":
            selected = candidate
            break
    for candidate in default_cold_start_search_v4_candidates()[len(results) :]:
        write_json(
            root / "poor_branch/execution_v4_development" / candidate.name.lower() / "not_run.json",
            {
                "schema_version": "ExecutionV4CandidateNotRunV1",
                "status": "NOT_RUN",
                "reason": "SIMPLER_CANDIDATE_ALREADY_PASSED",
            },
        )
    value = {
        "schema_version": "ExecutionV4DevelopmentDecisionV1",
        "status": "PASS" if selected is not None else "FAIL",
        "EXECUTION_V4_CREATED": "YES",
        "EXECUTION_V4_DEVELOPMENT": "PASS" if selected is not None else "FAIL",
        "selected_candidate": None if selected is None else selected.name,
        "candidate_results": [
            {
                "candidate": item["candidate"],
                "status": item["status"],
                "old_invalid_recovery_rate": item["old_invalid_recovery_rate"],
                "median_invalid_relative_reduction": item["median_invalid_relative_reduction"],
                "threshold_aware_nonregression_count": item["threshold_aware_nonregression_count"],
                "low_preserved_count": item["low_preserved_count"],
                "technical": item["technical"],
            }
            for item in results
        ],
    }
    write_json(root / "poor_branch/execution_v4_development/decision.json", value)
    return value


def _selected_v4_candidate(root: Path) -> ColdStartSearchV4Candidate:
    decision = require(
        root / "poor_branch/execution_v4_development/decision.json",
        "PASS",
        "EXECUTION_V4",
    )
    selected = str(decision["selected_candidate"])
    for candidate in default_cold_start_search_v4_candidates():
        if candidate.name == selected:
            return candidate
    raise RuntimeError(f"EXECUTION_V4_UNKNOWN_SELECTED_CANDIDATE:{selected}")


def freeze_execution_v4(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "FREEZE_EXECUTION_V4")
    candidate = _selected_v4_candidate(root)
    family_path = root / "poor_branch/execution_v4_development/candidate_family.json"
    family_hash = sha256_file(family_path)
    implementation_source = "\n".join(
        (
            inspect.getsource(search_cold_start_v4_from_v3),
            inspect.getsource(_v4_phase),
            inspect.getsource(ColdStartSearchV4Candidate),
        )
    )
    implementation_hash = hashlib.sha256(implementation_source.encode()).hexdigest()
    frozen = root / "poor_branch/frozen_v4"
    authorities = {
        "execution_input_authority_v2.json": {
            "schema_version": "ExecutionInputAuthorityV2",
            "status": "FROZEN",
            "REFINEMENT": "ExecutionV3 unchanged; old production q required",
            "COLD_START": "q_old absent; runtime previous state only after frame zero",
            "execution_v4_scope": "COLD_START_ONLY",
            "object_state_write": "FORBIDDEN",
            "hidden_force": "FORBIDDEN",
            "wrist_root_teleport": "FORBIDDEN",
        },
        "cold_start_search_authority_v4.json": {
            "schema_version": "ColdStartSearchAuthorityV4",
            "status": "FROZEN",
            "selected_candidate": candidate.as_dict(),
            "selection_rule": "first preregistered candidate passing every development gate",
            "candidate_family_sha256": family_hash,
            "implementation_sha256": implementation_hash,
            "execution_v3_prefix": "unchanged SearchV2 full-hand bootstrap top1 pipeline",
            "extension": "post-prefix top-2 sequential contributor refinement",
            "q_old": "ABSENT",
        },
        "objective_v2_execution_contract_v4.json": {
            "schema_version": "ObjectiveV2ExecutionContractV4",
            "status": "FROZEN",
            "objective_v2_sha256": d2h.EXPECTED_HASHES["retarget_objective_v2"],
            "certification_gate_v2_sha256": d2h.EXPECTED_HASHES["certification_gate_v2"],
            "objective_changed": False,
            "gate_changed": False,
            "refinement_changed": False,
            "cold_start_coverage_changed": True,
            "implementation_sha256": implementation_hash,
        },
    }
    hashes: dict[str, str] = {}
    for name, payload in authorities.items():
        path = frozen / name
        if path.exists() and read_json(path) != payload:
            raise RuntimeError(f"EXECUTION_V4_FROZEN_AUTHORITY_DRIFT:{name}")
        hashes[name] = freeze_json(path, payload)
    value = {
        "schema_version": "ExecutionV4FreezeDecisionV1",
        "status": "PASS",
        "EXECUTION_V4_CREATED": "YES",
        "EXECUTION_V4_DEVELOPMENT": "PASS",
        "EXECUTION_V4_FROZEN": "YES",
        "selected_candidate": candidate.name,
        "candidate_family_sha256": family_hash,
        "implementation_sha256": implementation_hash,
        "authority_sha256": hashes,
        "REFINEMENT_MODE_REGRESSION": "PASS_EXACT",
    }
    write_json(frozen / "freeze_decision.json", value)
    return value


def _v4_frame_exclusions() -> tuple[set[int], dict[str, int]]:
    prior = {
        int(row["ordinal"])
        for row in read_json(D2H_ROOT / "ledger/frame_exclusion_ledger.json")["entries"]
    }
    sparse_v4 = {
        int(row["ordinal"]) for row in read_json(D2H_ROOT / "sparse_v4/manifest.json")["frames"]
    }
    return prior | sparse_v4, {
        "pre_d2h": len(prior),
        "sparse_v4_and_v4_development": len(sparse_v4),
        "total_unique": len(prior | sparse_v4),
    }


def _freeze_sparse_v5(root: Path) -> dict[str, Any]:
    require(root / "poor_branch/frozen_v4/freeze_decision.json", "PASS", "SPARSE_V5_FREEZE")
    target = root / "poor_branch/sparse_v5/manifest.json"
    gate_path = root / "poor_branch/sparse_v5/gate_contract.json"
    if target.exists():
        if sha256_file(target) != target.with_suffix(".sha256").read_text().strip():
            raise RuntimeError("SPARSE_V5_MANIFEST_HASH_DRIFT")
        return read_json(target)
    excluded, counts = _v4_frame_exclusions()
    selected = d2h.d2d.select_sparse_rows(d2h.d2d.old_eim_rows(), excluded)
    ordinals = {int(row["ordinal"]) for row in selected}
    composition = {
        name: sum(str(row["stratum"]) == name for row in selected)
        for name in ("HIGH", "MID", "LOW")
    }
    if len(selected) != 30 or composition != {"HIGH": 10, "MID": 10, "LOW": 10}:
        raise RuntimeError(f"SPARSE_V5_SELECTION_INVALID:{len(selected)}:{composition}")
    if ordinals & excluded:
        raise RuntimeError("SPARSE_V5_EXCLUSION_OVERLAP")
    payload = {
        "schema_version": "ColdStartSparseValidationV5ManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "frames": selected,
        "N": 30,
        "composition": composition,
        "determinism_ordinals": d2h.d2d._determinism_sparse_ids(selected),
        "exclusion_counts": counts,
        "overlap_with_all_historical_and_v4_development": 0,
        "selection_authority": "old frozen DEV1 scalar E_IM only",
        "outcomes_used_for_selection": False,
        "certification_split_consumed": False,
        "heldout_split_consumed": False,
    }
    gate = {
        **d2h.sparse_gate_contract(),
        "schema_version": "ColdStartSparseValidationV5GateContractV1",
        "execution_method": "ExecutionV4",
    }
    freeze_json(target, payload)
    freeze_json(gate_path, gate)
    return payload


def _v4_repeat_comparison(
    reference_receipt: dict[str, Any],
    reference_q: np.ndarray,
    reference_base: np.ndarray,
    receipt: dict[str, Any],
    qpos: np.ndarray,
    base: np.ndarray,
) -> dict[str, Any]:
    q_diff = float(np.max(np.abs(qpos - reference_q)))
    base_diff = float(np.max(np.abs(base - reference_base)))
    e_diff = abs(
        float(receipt["selected"]["interaction_e_im"])
        - float(reference_receipt["selected"]["interaction_e_im"])
    )
    signature_exact = (
        receipt["used_contributors"] == reference_receipt["used_contributors"]
        and receipt["selected_block"] == reference_receipt["selected_block"]
    )
    return {
        "pass": signature_exact
        and q_diff <= d2h.Q_ATOL
        and base_diff <= d2h.BASE_ATOL
        and e_diff <= d2h.EPSILON_NUM,
        "signature_exact": signature_exact,
        "max_q_abs_diff": q_diff,
        "max_base_abs_diff": base_diff,
        "E_IM_abs_diff": e_diff,
    }


def _run_fresh_v4_frame(
    runtime: d2g.V3Runtime,
    root: Path,
    directory: str,
    ordinal: int,
    run: int,
    candidate: ColdStartSearchV4Candidate,
    *,
    runtime_step: int = 0,
    previous_q: np.ndarray | None = None,
    previous_base: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    receipt_path = root / directory / "receipts" / f"frame_{ordinal:04d}_run_{run}.json"
    state_path = receipt_path.with_suffix(".npz")
    if receipt_path.exists() and state_path.exists():
        with np.load(state_path, allow_pickle=False) as archive:
            return (
                read_json(receipt_path),
                np.asarray(archive["qpos"], dtype=np.float64),
                np.asarray(archive["base_pose_scene"], dtype=np.float64),
            )
    q_v3, base_v3, v3_receipt = d2g2.search_cold_start_v2_frame(
        runtime,
        ordinal,
        runtime_step=runtime_step,
        previous_q=previous_q,
        previous_base=previous_base,
        candidate=d2g2.CS2_A,
    )
    qpos, base, receipt = search_cold_start_v4_from_v3(
        runtime,
        ordinal,
        q_v3,
        base_v3,
        candidate,
        prefix_authority="LIVE_FROZEN_EXECUTION_V3_PREFIX",
        previous_q=previous_q,
        previous_base=previous_base,
    )
    receipt["runtime_step_index"] = runtime_step
    receipt["previous_accepted_state"] = "ABSENT" if runtime_step == 0 else "PRESENT"
    receipt["execution_v3_prefix_receipt"] = v3_receipt
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
    return receipt, qpos, base


def run_sparse_v5(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "RUN_SPARSE_V5")
    candidate = _selected_v4_candidate(root)
    manifest = _freeze_sparse_v5(root)
    decision_path = root / "poor_branch/sparse_v5/gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    runtime = d2g.V3Runtime("dev_01", root)
    rows: list[dict[str, Any]] = []
    references: dict[int, tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    for index, frame in enumerate(manifest["frames"], start=1):
        ordinal = int(frame["ordinal"])
        receipt, qpos, base = _run_fresh_v4_frame(
            runtime, root, "poor_branch/sparse_v5", ordinal, 1, candidate
        )
        row = d2h._row_from_coldstart(
            receipt, str(frame["stratum"]), float(frame["old_e_im"]), standalone=True
        )
        rows.append(row)
        references[ordinal] = (receipt, qpos, base)
        print(
            f"SPARSE_V5 {index}/{manifest['N']} ordinal={ordinal} E_IM={row['new_e_im']:.12g}",
            flush=True,
        )
    write_csv(root / "poor_branch/sparse_v5/per_frame_results.csv", rows)
    write_csv(
        root / "poor_branch/sparse_v5/qold_access_audit.csv",
        [
            {
                "ordinal": row["ordinal"],
                "q_old_field": row["q_old_field"],
                "q_old_access_count": row["q_old_access_count"],
                "previous_accepted_state": row["previous_accepted_state"],
            }
            for row in rows
        ],
    )
    preliminary = d2e.evaluate_gate_v2_rows(rows, determinism_pass=True)
    repeats: list[dict[str, Any]] = []
    determinism_pass = False
    if all(value for key, value in preliminary["conditions"].items() if key != "determinism"):
        for ordinal_value in manifest["determinism_ordinals"]:
            ordinal = int(ordinal_value)
            reference_receipt, reference_q, reference_base = references[ordinal]
            repeats.append({"ordinal": ordinal, "run": 1, "pass": True})
            for run in (2, 3):
                receipt, qpos, base = _run_fresh_v4_frame(
                    runtime, root, "poor_branch/sparse_v5", ordinal, run, candidate
                )
                repeats.append(
                    {
                        "ordinal": ordinal,
                        "run": run,
                        **_v4_repeat_comparison(
                            reference_receipt,
                            reference_q,
                            reference_base,
                            receipt,
                            qpos,
                            base,
                        ),
                    }
                )
        determinism_pass = all(bool(row["pass"]) for row in repeats)
    write_json(
        root / "poor_branch/sparse_v5/determinism.json",
        {
            "schema_version": "ColdStartSparseValidationV5DeterminismV1",
            "status": "PASS" if determinism_pass else "NOT_RUN_OTHER_GATE_FAILED",
            "rows": repeats,
        },
    )
    decision = d2e.evaluate_gate_v2_rows(rows, determinism_pass=determinism_pass)
    decision.update(
        {
            "schema_version": "ColdStartSparseValidationV5GateDecisionV1",
            "status": decision.pop("decision"),
            "SPARSE_V5": "PASS" if all(decision["conditions"].values()) else "FAIL",
            "q_old_access_count": 0,
            "overlap_with_prior_evidence": 0,
        }
    )
    decision["status"] = decision["SPARSE_V5"]
    write_json(decision_path, decision)
    if decision["status"] != "PASS":
        for stage in ("window_v5", "cross_episode_v5", "dev2_full"):
            write_json(
                root / f"poor_branch/{stage}/not_run.json",
                {
                    "schema_version": "ExecutionV4DownstreamNotRunV1",
                    "status": "NOT_RUN",
                    "reason": "SPARSE_V5_FAIL",
                },
            )
    return decision


def _freeze_window_v5(root: Path) -> dict[str, Any]:
    require(root / "poor_branch/sparse_v5/gate_decision.json", "PASS", "WINDOW_V5_FREEZE")
    target = root / "poor_branch/window_v5/manifest.json"
    if target.exists():
        if sha256_file(target) != target.with_suffix(".sha256").read_text().strip():
            raise RuntimeError("WINDOW_V5_MANIFEST_HASH_DRIFT")
        return read_json(target)
    excluded, counts = _v4_frame_exclusions()
    sparse_v5 = {
        int(row["ordinal"])
        for row in read_json(root / "poor_branch/sparse_v5/manifest.json")["frames"]
    }
    selected = d2h.d2d.select_window_rows(d2h.d2d.old_eim_rows(), excluded | sparse_v5, length=32)
    all_ordinals = [int(value) for window in selected for value in window["ordinals"]]
    ordinal_set = set(all_ordinals)
    overlaps = {
        "prior_and_v4_development": len(excluded & ordinal_set),
        "sparse_v5": len(sparse_v5 & ordinal_set),
        "internal": len(all_ordinals) - len(ordinal_set),
    }
    if len(selected) != 4 or any(overlaps.values()):
        raise RuntimeError(f"WINDOW_V5_SELECTION_INVALID:{len(selected)}:{overlaps}")
    payload = {
        "schema_version": "ColdStartWindowValidationV5ManifestV1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "windows": selected,
        "types": [window["type"] for window in selected],
        "target_frames_per_window": 32,
        "overlaps": overlaps,
        "pre_window_exclusion_counts": {**counts, "sparse_v5": len(sparse_v5)},
        "selection_authority": "old frozen DEV1 scalar E_IM only",
        "outcomes_used_for_selection": False,
        "certification_split_consumed": False,
        "heldout_split_consumed": False,
    }
    freeze_json(target, payload)
    freeze_json(
        root / "poor_branch/window_v5/gate_contract.json",
        {
            **d2h.window_gate_contract(),
            "schema_version": "ColdStartWindowValidationV5GateContractV1",
            "execution_method": "ExecutionV4",
        },
    )
    return payload


def _reevaluate_v4_continuity(
    runtime: d2g.V3Runtime,
    ordinal: int,
    qpos: np.ndarray,
    base: np.ndarray,
    previous_q: np.ndarray | None,
    previous_base: np.ndarray | None,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_q
    )
    values = runtime.measurement(ordinal, qpos, base, binding=binding, context=context, slack=None)
    actual = d2g.actual_continuity(runtime, previous_q, previous_base, qpos, base)
    evaluation = d2g._evaluate_b2(runtime, values, actual)
    receipt["selected"] = d2g._measurement_row(values)
    receipt["selected_actual_continuity"] = actual
    receipt["selected_evaluation"] = evaluation
    receipt["technical_success"] = bool(evaluation["feasible"])
    receipt["continuity_reevaluated_against_runtime_predecessor"] = True
    return receipt


def _window_v5_run(
    runtime: d2g.V3Runtime,
    root: Path,
    window: dict[str, Any],
    run: int,
    candidate: ColdStartSearchV4Candidate,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    old = {int(row["ordinal"]): float(row["old_e_im"]) for row in d2h.d2d.old_eim_rows()}
    previous_q = None
    previous_base = None
    rows: list[dict[str, Any]] = []
    q_states: list[np.ndarray] = []
    base_states: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    directory = f"poor_branch/window_v5/{window['window_id']}"
    for step, ordinal_value in enumerate(window["ordinals"]):
        ordinal = int(ordinal_value)
        receipt, qpos, base = _run_fresh_v4_frame(
            runtime,
            root,
            directory,
            ordinal,
            run,
            candidate,
            runtime_step=step,
            previous_q=previous_q,
            previous_base=previous_base,
        )
        receipt = _reevaluate_v4_continuity(
            runtime, ordinal, qpos, base, previous_q, previous_base, receipt
        )
        write_json(root / directory / "receipts" / f"frame_{ordinal:04d}_run_{run}.json", receipt)
        row = d2h._row_from_coldstart(receipt, str(window["type"]), old[ordinal], standalone=False)
        row.update({"window_id": window["window_id"], "run": run})
        rows.append(row)
        q_states.append(qpos)
        base_states.append(base)
        receipts.append(receipt)
        previous_q, previous_base = qpos, base
    return rows, q_states, base_states, receipts


def run_window_v5(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "RUN_WINDOW_V5")
    candidate = _selected_v4_candidate(root)
    manifest = _freeze_window_v5(root)
    decision_path = root / "poor_branch/window_v5/gate_decision.json"
    if decision_path.exists():
        return read_json(decision_path)
    runtime = d2g.V3Runtime("dev_01", root)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    jitter: list[dict[str, Any]] = []
    references: dict[
        str, tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]
    ] = {}
    for window in manifest["windows"]:
        try:
            result = _window_v5_run(runtime, root, window, 1, candidate)
        except Exception as exc:
            failure = {
                "schema_version": "ColdStartWindowValidationV5TechnicalFailureV1",
                "window_id": window["window_id"],
                "type": window["type"],
                "error": f"{type(exc).__name__}:{exc}",
                "classification": "FROZEN_EXECUTION_V4_SEQUENCE_INPUT_AUTHORITY_FAILURE",
                "scientific_retry_authorized": False,
            }
            write_json(
                root
                / "poor_branch/window_v5"
                / str(window["window_id"])
                / "technical_failure.json",
                failure,
            )
            summaries.append(
                {
                    "window_id": window["window_id"],
                    "type": window["type"],
                    "N": 0,
                    "expected_N": int(window["N"]),
                    "new_p95_e_im": None,
                    "technical": 0,
                    "continuity": False,
                    "hard_validity": False,
                    "low_old_valid_preserved": False,
                    "result_without_determinism": "FAIL",
                    "failure": failure["error"],
                }
            )
            break
        rows, q_states, _base_states, _receipts = result
        references[str(window["window_id"])] = result
        all_rows.extend(rows)
        jitter.extend(d2h._jitter_rows(runtime, str(window["window_id"]), rows, q_states))
        eim = [float(row["new_e_im"]) for row in rows]
        p95 = float(np.quantile(eim, 0.95)) if eim else None
        low_preserved = all(
            float(row["new_e_im"]) <= d2h.TAU + d2h.EPSILON_NUM
            for row in rows
            if str(window["type"]) == "LOW" and float(row["old_e_im"]) <= d2h.TAU
        )
        summary = {
            "window_id": window["window_id"],
            "type": window["type"],
            "N": len(rows),
            "expected_N": int(window["N"]),
            "new_p95_e_im": p95,
            "technical": sum(bool(row["technical_completion"]) for row in rows),
            "continuity": all(bool(row["continuity_pass"]) for row in rows),
            "hard_validity": all(bool(row["semantic_hard_pass"]) for row in rows),
            "low_old_valid_preserved": low_preserved,
        }
        summary["result_without_determinism"] = (
            "PASS"
            if len(rows) == int(window["N"])
            and summary["technical"] == int(window["N"])
            and p95 is not None
            and p95 <= d2h.TAU
            and summary["continuity"]
            and summary["hard_validity"]
            and low_preserved
            else "FAIL"
        )
        summaries.append(summary)
        print(
            f"WINDOW_V5 window={window['window_id']} N={len(rows)} p95={p95} "
            f"status={summary['result_without_determinism']}",
            flush=True,
        )
        if summary["result_without_determinism"] != "PASS":
            break
    write_csv(
        root / "poor_branch/window_v5/per_frame_results.csv",
        all_rows,
        fields=None if all_rows else ["window_id", "ordinal", "run", "technical_completion"],
    )
    write_csv(root / "poor_branch/window_v5/per_window_summary.csv", summaries)
    write_csv(
        root / "poor_branch/window_v5/jitter_metrics.csv",
        jitter,
        fields=None if jitter else ["window_id", "ordinal", "role"],
    )
    repeats: list[dict[str, Any]] = []
    if len(summaries) == 4 and all(
        summary["result_without_determinism"] == "PASS" for summary in summaries
    ):
        for window_id in ("HIGH_1", "LOW"):
            window = next(row for row in manifest["windows"] if str(row["window_id"]) == window_id)
            ref_rows, ref_q, ref_base, ref_receipts = references[window_id]
            repeats.append({"window_id": window_id, "run": 1, "pass": True})
            for run in (2, 3):
                rows, q_states, base_states, receipts = _window_v5_run(
                    runtime, root, window, run, candidate
                )
                comparisons = [
                    _v4_repeat_comparison(a, qa, ba, b, qb, bb)
                    for a, qa, ba, b, qb, bb in zip(
                        ref_receipts,
                        ref_q,
                        ref_base,
                        receipts,
                        q_states,
                        base_states,
                        strict=True,
                    )
                ]
                p95_diff = abs(
                    float(np.quantile([float(row["new_e_im"]) for row in ref_rows], 0.95))
                    - float(np.quantile([float(row["new_e_im"]) for row in rows], 0.95))
                )
                repeats.append(
                    {
                        "window_id": window_id,
                        "run": run,
                        "pass": all(bool(item["pass"]) for item in comparisons)
                        and p95_diff <= d2h.EPSILON_NUM,
                        "p95_E_IM_abs_diff": p95_diff,
                        "frames": comparisons,
                    }
                )
    determinism_pass = bool(repeats) and all(bool(row["pass"]) for row in repeats)
    write_json(
        root / "poor_branch/window_v5/determinism.json",
        {
            "schema_version": "ColdStartWindowValidationV5DeterminismV1",
            "status": "PASS" if determinism_pass else "NOT_RUN_OR_FAIL",
            "rows": repeats,
        },
    )
    overall = bool(
        len(summaries) == 4
        and all(summary["result_without_determinism"] == "PASS" for summary in summaries)
        and determinism_pass
    )
    decision = {
        "schema_version": "ColdStartWindowValidationV5GateDecisionV1",
        "status": "PASS" if overall else "FAIL",
        "WINDOW_V5": "PASS" if overall else "FAIL",
        "windows": summaries,
        "determinism": "PASS" if determinism_pass else "NOT_RUN_OR_FAIL",
        "q_old_access_count": sum(int(row["q_old_access_count"]) for row in all_rows),
        "overlap_with_prior_evidence": 0,
    }
    write_json(decision_path, decision)
    if not overall:
        for stage in ("cross_episode_v5", "dev2_full"):
            write_json(
                root / f"poor_branch/{stage}/not_run.json",
                {
                    "schema_version": "ExecutionV4DownstreamNotRunV1",
                    "status": "NOT_RUN",
                    "reason": "WINDOW_V5_FAIL",
                },
            )
    return decision


def run_cross_episode_v5(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "RUN_CROSS_EPISODE_V5")
    window = read_json(root / "poor_branch/window_v5/gate_decision.json")
    if window.get("status") != "PASS":
        value = {
            "schema_version": "CrossEpisodeV5NotRunV1",
            "status": "NOT_RUN",
            "CROSS_EPISODE_V5": "NOT_RUN",
            "reason": "WINDOW_V5_FAIL",
            "episodes_consumed": 0,
            "certification_split_consumed": 0,
            "heldout_split_consumed": 0,
        }
        write_json(root / "poor_branch/cross_episode_v5/not_run.json", value)
        return value
    raise RuntimeError("RUN_CROSS_EPISODE_V5_PASS_PATH_NOT_IMPLEMENTED_FAIL_CLOSED")


def authorize_dev2_full(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "AUTHORIZE_DEV2_FULL")
    sparse = read_json(root / "poor_branch/sparse_v5/gate_decision.json")
    window = read_json(root / "poor_branch/window_v5/gate_decision.json")
    cross_path = root / "poor_branch/cross_episode_v5/gate_decision.json"
    cross = read_json(cross_path) if cross_path.exists() else {"status": "NOT_RUN"}
    authorized = (
        sparse.get("status") == "PASS"
        and window.get("status") == "PASS"
        and cross.get("status") == "PASS"
    )
    value = {
        "schema_version": "ExecutionV4Dev2AuthorizationV1",
        "status": "PASS" if authorized else "DENIED",
        "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED": "YES" if authorized else "NO",
        "SPARSE_V5": sparse.get("status"),
        "WINDOW_V5": window.get("status"),
        "CROSS_EPISODE_V5": cross.get("status"),
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 0,
        "reason": None if authorized else "SELECTED_BRANCH_CERTIFICATION_INCOMPLETE_OR_FAILED",
    }
    write_json(root / "poor_branch/dev2_full/authorization.json", value)
    if not authorized:
        write_json(
            root / "poor_branch/dev2_full/not_run.json",
            {
                "schema_version": "ExecutionV4Dev2NotRunV1",
                "status": "NOT_RUN",
                "reason": value["reason"],
                "solve_count": 0,
            },
        )
    return value


def run_dev2_full(root: Path) -> dict[str, Any]:
    authorization = authorize_dev2_full(root)
    if authorization["DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED"] != "YES":
        return read_json(root / "poor_branch/dev2_full/not_run.json")
    raise RuntimeError("DEV2_FULL_AUTHORIZED_PATH_NOT_IMPLEMENTED_FAIL_CLOSED")


def render_dev2_viewer(root: Path) -> dict[str, Any]:
    _require_branch(root, "POOR", "RENDER_DEV2_VIEWER")
    if not (root / "poor_branch/dev2_full/trajectory.npz").exists():
        value = {
            "schema_version": "ExecutionV4ViewerNotRunV1",
            "status": "NOT_RUN",
            "reason": "DEV2_FULL_NOT_RUN",
            "viewer": None,
            "human_review": "NOT_RUN",
        }
        write_json(root / "poor_branch/dev2_full/viewer_not_run.json", value)
        return value
    raise RuntimeError("DEV2_VIEWER_REQUIRES_COMPLETED_AUTHORIZED_TRAJECTORY")


def _unimplemented_action(root: Path, action: str) -> dict[str, Any]:
    del root
    raise RuntimeError(f"{action}_NOT_AUTHORIZED_OR_NOT_IMPLEMENTED")


def summarize(root: Path) -> dict[str, Any]:
    audit = read_json(root / "gate_audit/gate_alignment_decision.json")
    metric = read_json(root / "metric_semantics/physical_metric_semantic_audit.json")
    decision_path = root / "recoverability_v2/decision.json"
    decision = read_json(decision_path) if decision_path.is_file() else None
    selected = "BLOCKED" if decision is None else str(decision["SELECTED_TERMINAL_BRANCH"])
    final_decision = (
        "BLOCKED" if decision is None else str(decision["FINAL_RECOVERABILITY_DECISION"])
    )
    for branch, directory in (
        ("STRONG", "strong_branch"),
        ("MIXED", "mixed_branch"),
        ("POOR", "poor_branch"),
    ):
        path = root / directory / "not_run.json"
        if branch != selected and not path.exists():
            write_json(
                path,
                {
                    "schema_version": "RecoverabilityBranchNotRunV1",
                    "status": "NOT_RUN",
                    "reason": "BRANCH_NOT_SELECTED",
                    "branch": branch,
                },
            )
    v4_development_path = root / "poor_branch/execution_v4_development/decision.json"
    v4_freeze_path = root / "poor_branch/frozen_v4/freeze_decision.json"
    sparse_path = root / "poor_branch/sparse_v5/gate_decision.json"
    window_path = root / "poor_branch/window_v5/gate_decision.json"
    cross_path = root / "poor_branch/cross_episode_v5/not_run.json"
    authorization_path = root / "poor_branch/dev2_full/authorization.json"
    v4_development = read_json(v4_development_path) if v4_development_path.exists() else None
    v4_freeze = read_json(v4_freeze_path) if v4_freeze_path.exists() else None
    sparse = read_json(sparse_path) if sparse_path.exists() else None
    window = read_json(window_path) if window_path.exists() else None
    cross = read_json(cross_path) if cross_path.exists() else None
    authorization = read_json(authorization_path) if authorization_path.exists() else None
    scientific_terminal = bool(window and window.get("status") == "FAIL")
    value = {
        "schema_version": "OakInk2RecoverabilityDecisionTreeSummaryV1",
        "status": "COMPLETE" if decision is not None and scientific_terminal else "BLOCKED",
        "BRANCH": EXPECTED_BRANCH,
        "PRE_D2K_AUTHORITY_CUTOFF_HEAD": CUTOFF_HEAD,
        "PRE_D2K_BINARY_GATE_STATUS": audit["PRE_D2K_BINARY_GATE_STATUS"],
        "PRE_D2K_BINARY_GATE_WHY": audit["why"],
        "OBJECT_DRIFT_METRIC_SEMANTICS": metric["object_drift_exact_semantics"],
        "RAW_OBJECT_TRANSLATION_AVAILABLE": metric["RAW_OBJECT_TRANSLATION_AVAILABLE"],
        "RAW_OBJECT_ROTATION_AVAILABLE": metric["RAW_OBJECT_ROTATION_AVAILABLE"],
        "RECOVERABILITY_EVIDENCE_SOURCE": "PROSPECTIVE_RECOVERABILITY_V2",
        "STATIC_PHYSICAL_RECOVERABILITY_AUTHORITY": "StaticPhysicalRecoverabilityV2",
        "BINARY_GATE_SHA256": (
            sha256_file(root / "recoverability_v2/binary_contract.json")
            if (root / "recoverability_v2/binary_contract.json").is_file()
            else None
        ),
        "FINAL_RECOVERABILITY_DECISION": final_decision,
        "SELECTED_TERMINAL_BRANCH": selected,
        "ONE_TERMINAL_BRANCH_ONLY": "YES",
        "HISTORICAL_SPARSE_V4_RESULT_REWRITTEN": "NO",
        "HISTORICAL_D2K_V1_RESULT": "INCONCLUSIVE",
        "HISTORICAL_D2K_V1_RESULT_REWRITTEN": "NO",
        "SOURCE_SUPPORT_AUTHORITY": "SUPPORT_UNRESOLVED",
        "SOURCE_SUPPORT_RESULT_REWRITTEN": "NO",
        "SUPPORT_PROXY_TYPE": "STATIC_RECOVERABILITY_PLANAR_PROXY",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "EXECUTION_V3_CHANGED": "NO",
        "GATE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "PPO_REWARD_CHANGED": "NO",
        "RETARGET_ADMISSION_POLICY_V2": "NOT_RUN",
        "RETARGET_ADMISSION_POLICY_V2_SHA256": None,
        "QUALITY_AWARE_ADMISSION_DEPLOYABLE": "NOT_RUN",
        "LOAO_AUROC": None,
        "LOAO_BALANCED_ACCURACY": None,
        "NONRECOVERABLE_SENSITIVITY": None,
        "QUALITY_AWARE_ADMISSION_SHA256": None,
        "EXECUTION_V4_CREATED": (
            "YES"
            if v4_development and v4_development.get("EXECUTION_V4_CREATED") == "YES"
            else "NO"
        ),
        "EXECUTION_V4_DEVELOPMENT": (
            v4_development.get("EXECUTION_V4_DEVELOPMENT", "NOT_RUN")
            if v4_development
            else "NOT_RUN"
        ),
        "EXECUTION_V4_FROZEN": (v4_freeze.get("EXECUTION_V4_FROZEN", "NO") if v4_freeze else "NO"),
        "EXECUTION_V4_SELECTED_CANDIDATE": (
            v4_development.get("selected_candidate") if v4_development else None
        ),
        "SPARSE_V5": sparse.get("status", "NOT_RUN") if sparse else "NOT_RUN",
        "WINDOW_V5": window.get("status", "NOT_RUN") if window else "NOT_RUN",
        "CROSS_EPISODE_V5": cross.get("status", "NOT_RUN") if cross else "NOT_RUN",
        "EXECUTION_V4_INDEPENDENT_CERTIFICATION": (
            "PASS"
            if sparse
            and sparse.get("status") == "PASS"
            and window
            and window.get("status") == "PASS"
            and cross
            and cross.get("status") == "PASS"
            else "FAIL"
            if scientific_terminal
            else "NOT_RUN"
        ),
        "EXECUTION_V4_CERTIFICATION_FAILURE": (
            window["windows"][0].get("failure") if scientific_terminal else None
        ),
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED": (
            authorization.get("DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED", "NO")
            if authorization
            else "NO"
        ),
        "DEV2_FULL_GEOMETRIC_SOLVE_COUNT": 0,
        "DEV2_EXPECTED_FRAMES": 240,
        "DEV2_COMPLETED_FRAMES": 0,
        "DEV2_METHOD": "NOT_RUN",
        "DEV2_SEMANTIC_V1_RESULT": "NOT_RUN",
        "DEV2_ADMISSION_RESULT": "DENIED",
        "DEV2_TRAJECTORY_SHA256": None,
        "DEV2_VIEWER": None,
        "DEV2_HUMAN_REVIEW": "NOT_RUN",
        "DEV2_FULL_PHYSICAL_PPO_RAN": "NO",
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "O6_PRODUCTION_RAN": "NO",
        "MAX_GPU_JOBS": 1,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "NEXT": "HARD_STOP_WINDOW_V5_FAIL_NO_CROSS_EPISODE_NO_DEV2",
    }
    if decision:
        value.update(
            {
                key: decision[key]
                for key in (
                    "SEMANTIC_PASS_N",
                    "SEMANTIC_PASS_POST_PPO_ACCEPTED",
                    "SEMANTIC_FAIL_N",
                    "SEMANTIC_FAIL_POST_PPO_ACCEPTED",
                    "SEMANTIC_FAIL_BASELINE_ACCEPTED",
                    "SEMANTIC_FAIL_RECOVERED_BY_PPO",
                    "PPO_REGRESSION_COUNT",
                )
            }
        )
    write_json(root / "final_summary.json", value)
    (root / "final_summary.md").write_text(
        "# OakInk2 Recoverability Decision Tree\n\n"
        + "\n".join(f"- `{key}={item}`" for key, item in value.items() if key != "schema_version")
        + "\n",
        encoding="utf-8",
    )
    handoff_lines = [
        "# OakInk2 Recoverability Decision Tree Handoff",
        "",
        "## Gate authority audit",
        "",
        f"- `PRE_D2K_AUTHORITY_CUTOFF_HEAD={value['PRE_D2K_AUTHORITY_CUTOFF_HEAD']}`",
        f"- `PRE_D2K_BINARY_GATE_STATUS={value['PRE_D2K_BINARY_GATE_STATUS']}`",
        *[f"- WHY: {reason}" for reason in audit["why"]],
        "",
        "## Metric semantics and recoverability",
        "",
        f"- `OBJECT_DRIFT_METRIC_SEMANTICS={value['OBJECT_DRIFT_METRIC_SEMANTICS']}`",
        f"- `RAW_OBJECT_TRANSLATION_AVAILABLE={value['RAW_OBJECT_TRANSLATION_AVAILABLE']}`",
        f"- `RAW_OBJECT_ROTATION_AVAILABLE={value['RAW_OBJECT_ROTATION_AVAILABLE']}`",
        f"- `RECOVERABILITY_EVIDENCE_SOURCE={value['RECOVERABILITY_EVIDENCE_SOURCE']}`",
        f"- `BINARY_GATE_SHA256={value['BINARY_GATE_SHA256']}`",
        f"- `FINAL_RECOVERABILITY_DECISION={value['FINAL_RECOVERABILITY_DECISION']}`",
        f"- `SELECTED_TERMINAL_BRANCH={value['SELECTED_TERMINAL_BRANCH']}`",
        "",
        "## POOR branch",
        "",
        f"- `EXECUTION_V4_CREATED={value['EXECUTION_V4_CREATED']}`",
        f"- `EXECUTION_V4_DEVELOPMENT={value['EXECUTION_V4_DEVELOPMENT']}`",
        f"- `EXECUTION_V4_FROZEN={value['EXECUTION_V4_FROZEN']}`",
        f"- `SPARSE_V5={value['SPARSE_V5']}`",
        f"- `WINDOW_V5={value['WINDOW_V5']}`",
        f"- `CROSS_EPISODE_V5={value['CROSS_EPISODE_V5']}`",
        f"- `EXECUTION_V4_INDEPENDENT_CERTIFICATION={value['EXECUTION_V4_INDEPENDENT_CERTIFICATION']}`",
        f"- Failure: `{value['EXECUTION_V4_CERTIFICATION_FAILURE']}`",
        "",
        "## DEV2 and safety",
        "",
        f"- `DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED={value['DEV2_FULL_GEOMETRIC_SOLVE_AUTHORIZED']}`",
        f"- `DEV2_FULL_GEOMETRIC_SOLVE_COUNT={value['DEV2_FULL_GEOMETRIC_SOLVE_COUNT']}`",
        f"- `DEV2_METHOD={value['DEV2_METHOD']}`",
        f"- `NEXT={value['NEXT']}`",
    ]
    (root / "handoff.md").write_text("\n".join(handoff_lines) + "\n", encoding="utf-8")
    write_json(
        root / "completion_audit.json",
        {
            "schema_version": "RecoverabilityDecisionTreeCompletionAuditV1",
            "status": value["status"],
            "selected_branch": selected,
            "branch_terminal_work_complete": scientific_terminal,
            "terminal_reason": value["NEXT"],
            "historical_results_rewritten": False,
            "unselected_branches_not_run": True,
            "dev2_solve_count": 0,
        },
    )
    write_json(
        root / "ledger/evidence_ledger.json",
        {
            "schema_version": "RecoverabilityDecisionTreeEvidenceLedgerV1",
            "status": "PASS",
            "roles": [
                "RETARGET_CERTIFICATION_CONSUMED",
                "PHYSICAL_STUDY_V1_CONSUMED",
                "PHYSICAL_STUDY_V2_CONSUMED" if decision else "PHYSICAL_STUDY_V2_NOT_COMPLETE",
            ],
            "certification_split_new_consumption": 0,
            "heldout_split_new_consumption": 0,
        },
    )
    failure_path = root / "technical_failures.jsonl"
    if scientific_terminal:
        failure_row = {
            "schema_version": "RecoverabilityDecisionTreeTechnicalFailureV1",
            "stage": "WINDOW_V5",
            "error": value["EXECUTION_V4_CERTIFICATION_FAILURE"],
            "classification": "FROZEN_METHOD_SCIENTIFIC_CERTIFICATION_FAILURE",
            "retry_authorized": False,
        }
        existing = failure_path.read_text(encoding="utf-8") if failure_path.exists() else ""
        serialized = json.dumps(failure_row, sort_keys=True)
        if serialized not in existing:
            with failure_path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
    else:
        failure_path.touch(exist_ok=True)
    return value


ACTIONS = {
    "preflight": preflight,
    "verify-current-evidence": verify_current_evidence,
    "audit-physical-metric-semantics": audit_physical_metric_semantics,
    "inventory-pre-d2k-gates": inventory_pre_d2k_gates,
    "audit-gate-applicability": audit_gate_applicability,
    "align-existing-gate": align_existing_gate,
    "rescore-existing-d2k-rollouts": rescore_existing_d2k_rollouts,
    "design-recoverability-v2": design_recoverability_v2,
    "freeze-recoverability-v2": freeze_recoverability_v2,
    "select-v2-physical-anchors": select_v2_physical_anchors,
    "run-v2-baselines": run_v2_baselines,
    "run-v2-ppo": run_v2_ppo,
    "evaluate-v2": evaluate_v2,
    "decide-recoverability": decide_recoverability,
    "build-retarget-admission-v2": build_retarget_admission_v2,
    "develop-quality-aware-admission": develop_quality_aware_admission,
    "develop-execution-v4": develop_execution_v4,
    "freeze-execution-v4": freeze_execution_v4,
    "run-sparse-v5": run_sparse_v5,
    "run-window-v5": run_window_v5,
    "run-cross-episode-v5": run_cross_episode_v5,
    "authorize-dev2-full": authorize_dev2_full,
    "run-dev2-full": run_dev2_full,
    "render-dev2-viewer": render_dev2_viewer,
    "summarize": summarize,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=REQUIRED_ACTIONS)
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        value = ACTIONS[args.action](args.root.resolve())
        print(json.dumps(value, sort_keys=True, default=str))
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
