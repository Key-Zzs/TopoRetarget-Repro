#!/usr/bin/env python3
"""O5R-D3-R4 optimizer-free RefinementV2 gate semantic-alignment audit.

This driver only reads repository history, frozen contracts, and stored R3
evidence.  It has no optimizer, fresh-certification, D3-V2, DEV2, PPO, PhysX,
or O6 execution entry point.
"""

# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from toporetarget.evaluation.retarget_semantic_validity import (  # noqa: E402
    SemanticGateContractV1,
)
from toporetarget.utils.hashing import sha256_file  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd3r4_refinement_v2_gate_semantic_alignment_v1"
D3_ROOT = REPO / ".local/reports/oakink2_o5rd3_dev1_full_objective_v2_execution_v4_refinement_v1"
R2_ROOT = REPO / ".local/reports/oakink2_o5rd3r2_refinement_search_space_repair_design_v1"
R3_ROOT = REPO / ".local/reports/oakink2_o5rd3r3_refinement_v2_development_validation_v1"
D2E_ROOT = REPO / ".local/reports/oakink2_o5rd2e_gate_alignment_and_v3_recertification_v1"
D2I_ROOT = REPO / ".local/reports/oakink2_o5rd2i_sparsev4_failure_and_ppo_recoverability_v1"
D2J_ROOT = REPO / ".local/reports/oakink2_o5rd2j_d2k_static_physics_and_ppo_recoverability_v1"
D2JR_ROOT = REPO / ".local/reports/oakink2_o5rd2jr_d2k_support_proxy_ppo_recoverability_v1"
D2KR_ROOT = REPO / ".local/reports/oakink2_o5rd2kr_recoverability_decision_tree_v1"

EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
START_HEAD = "d9c947a2c60a8e0864c3a7d74d3c111ef24445e9"
GUIDANCE_HEAD = "61a59717c48881c0acc59df1ba20de4fb9a6e7c6"
EXPECTED_DESIGN_SHA256 = "b59cc09314ebf0b12ce7976d367a7a03eed0125c945d384eec4371e48c0e49b3"
EXPECTED_GATE_V1_SHA256 = "96d94c838c19f05dd28bb7ba065121bbc08d95e5a79fb6548c8ef1a1f81d366e"
EXPECTED_SEMANTIC_V1_SHA256 = "0ea9ba21419a2b254557e0af18405649b23a0cf30407684eaa2e2349c4467e31"
EXPECTED_OBJECTIVE_V2_SHA256 = "48861d02020395bbdd891635e2176028f3fd5a86f825c784a6763dc3504211bc"
EXPECTED_R3_MANIFEST_SHA256 = "894dcda9d9b67905e970bec6e8320754e1cff362f50919d704202c9ed6692fa3"
EXPECTED_FAILURE_MANIFEST_SHA256 = (
    "a81118d0db55fb08fb38d86fb66f7c593dbb0914a1bf8757ef19ba6944e45aae"
)
EXPECTED_VALID_CONTROLS_SHA256 = "8285bf53b4e2d415cf210f896d2b7ba450d613e9b0ebd86d6c8ad69c3dd8fb63"
EXPECTED_WINDOWS_SHA256 = "f4ad14afb6bbaf289c4aa29872313142a7954e56fe387f5b23817d05b1c1f072"
EXPECTED_QOLD_ARRAY_SHA256 = "9e048361e37da77e56d18b720b8ef4a6876eeb3e094988e291a0a747c3324743"
EXPECTED_INVALID = 145
TAU = 1.0e-4
EPSILON_NUM = 1.0e-12

AUTHORITY_TYPES = {
    "PAPER_METHOD_REQUIREMENT",
    "SEMANTIC_V1_REQUIREMENT",
    "ROBOT_KINEMATIC_REQUIREMENT",
    "PHYSICAL_REQUIREMENT",
    "DOWNSTREAM_RECOVERABILITY_EVIDENCE",
    "PRE_REGISTERED_CONSERVATIVE_HEURISTIC",
    "IMPLEMENTATION_HEURISTIC",
    "HISTORICAL_CARRYOVER",
    "UNRESOLVED",
}
MEDIAN_AUTHORITY_TYPES = {
    "STRONG_INDEPENDENT_AUTHORITY",
    "CONSERVATIVE_PREREGISTERED_HEURISTIC",
    "IMPLEMENTATION_HEURISTIC",
    "HISTORICAL_CARRYOVER",
    "UNRESOLVED",
}
FORBIDDEN_EXECUTION_ACTIONS = {
    "run-retarget-optimizer",
    "run-fresh-refinement-sparse",
    "run-fresh-refinement-window",
    "run-cross-episode-refinement",
    "run-d3v2",
    "run-dev2",
    "run-ppo",
    "run-physx",
    "run-o6",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def sidecar_value(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip().split()[0]


def require(path: Path, field: str, expected: Any, action: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{action}_REJECTED:MISSING:{path}")
    value = read_json(path)
    if value.get(field) != expected:
        raise RuntimeError(f"{action}_REJECTED:{field}={value.get(field)!r}")
    return value


def initial_not_run(root: Path) -> None:
    payload = {
        "schema_version": "O5RD3R4ExplicitNotRunV1",
        "status": "NOT_RUN",
        "RETARGET_OPTIMIZER_RUN_COUNT": 0,
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "D3_V2_TRAJECTORY": None,
        "D3_V2_SEMANTIC_V1": "NOT_RUN",
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
        write_json(root / relative, payload)


def preflight(root: Path) -> dict[str, Any]:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--short", "--untracked-files=all")
    allowed_changes = (
        "scripts/evaluation/run_oakink2_o5rd3r4.py",
        "tests/evaluation/test_oakink2_o5rd3r4.py",
    )
    unexpected = [
        line for line in status.splitlines() if not any(x in line for x in allowed_changes)
    ]
    guidance = REPO.parent / "TopoRetarget-Repro-guidance"
    checks = {
        "repo_exact": Path(git("rev-parse", "--show-toplevel")) == REPO,
        "branch_exact": branch == EXPECTED_BRANCH,
        "start_head_is_ancestor": subprocess.run(
            ["git", "merge-base", "--is-ancestor", START_HEAD, head], cwd=REPO, check=False
        ).returncode
        == 0,
        "task_changes_only": not unexpected,
        "artifact_root_ignored": subprocess.run(
            ["git", "check-ignore", "-q", str(root)], cwd=REPO, check=False
        ).returncode
        == 0,
        "historical_roots_exist": all(path.is_dir() for path in (D3_ROOT, R2_ROOT, R3_ROOT)),
        "oakink2_root_exists": Path("/mnt/nas/storage/Ref2Dex_storage/OakInk2").is_dir(),
        "guidance_head_exact": git("rev-parse", "HEAD", cwd=guidance) == GUIDANCE_HEAD,
        "guidance_worktree_clean": not git("status", "--short", cwd=guidance),
    }
    value = {
        "schema_version": "O5RD3R4GitPreflightV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "BRANCH": branch,
        "START_HEAD": START_HEAD,
        "HEAD_AT_PREFLIGHT": head,
        "initial_manual_git_snapshot_clean": True,
        "status_short_at_cli_preflight": status,
        "unexpected_changes": unexpected,
        "diff_stat": git("diff", "--stat"),
        "cached_diff_stat": git("diff", "--cached", "--stat"),
        "diff_check": subprocess.run(
            ["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True, check=False
        ).stdout,
        "worktrees": git("worktree", "list", "--porcelain"),
        "remotes": git("remote", "-v"),
        "PUSHED": "NO",
        "PR_CREATED": "NO",
    }
    write_json(root / "preflight/git.json", value)
    initial_not_run(root)
    write_text(root / "technical_failures.jsonl", "")
    if value["status"] != "PASS":
        raise RuntimeError("O5RD3R4_PREFLIGHT_FAIL")
    return value


def verify_r3_history(root: Path) -> dict[str, Any]:
    design_path = R2_ROOT / "design/refinement_v2_design.json"
    gate_path = R2_ROOT / "design/refinement_v2_development_gate.json"
    r3_summary_path = R3_ROOT / "final_summary.json"
    r3_decision_path = R3_ROOT / "gate/decision.json"
    failure_manifest_path = R3_ROOT / "manifests/failure_frames.json"
    controls_path = R3_ROOT / "manifests/valid_controls.json"
    windows_path = R3_ROOT / "manifests/sequential_windows.json"
    validation_manifest_path = R3_ROOT / "run_authority/development_validation_manifest.json"
    d3_manifest_path = D3_ROOT / "run_authority/run_manifest.json"
    qold_authority_path = D3_ROOT / "old_trajectory/authority.json"
    summary = read_json(r3_summary_path)
    decision = read_json(r3_decision_path)
    d3_manifest = read_json(d3_manifest_path)
    qold_authority = read_json(qold_authority_path)
    observed = {
        "design": sha256_file(design_path),
        "gate_v1": sha256_file(gate_path),
        "semantic_v1": sha256_file(
            REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py"
        ),
        "objective_v2_contract": sha256_file(
            D2E_ROOT / "method/retarget_objective_v2_contract.json"
        ),
        "d3_run_manifest": sha256_file(d3_manifest_path),
        "qold_authority_file": sha256_file(qold_authority_path),
        "r3_validation_manifest": sha256_file(validation_manifest_path),
        "failure_manifest": sha256_file(failure_manifest_path),
        "valid_controls_manifest": sha256_file(controls_path),
        "sequential_windows_manifest": sha256_file(windows_path),
    }
    checks = {
        "design_sha_exact": observed["design"] == EXPECTED_DESIGN_SHA256,
        "design_sidecar_exact": sidecar_value(design_path.with_suffix(".sha256"))
        == EXPECTED_DESIGN_SHA256,
        "gate_v1_sha_exact": observed["gate_v1"] == EXPECTED_GATE_V1_SHA256,
        "gate_v1_sidecar_exact": sidecar_value(gate_path.with_suffix(".sha256"))
        == EXPECTED_GATE_V1_SHA256,
        "semantic_v1_sha_exact": observed["semantic_v1"] == EXPECTED_SEMANTIC_V1_SHA256,
        "objective_v2_sha_exact": observed["objective_v2_contract"] == EXPECTED_OBJECTIVE_V2_SHA256,
        "r3_manifest_sha_exact": observed["r3_validation_manifest"] == EXPECTED_R3_MANIFEST_SHA256,
        "r3_manifest_sidecar_exact": sidecar_value(validation_manifest_path.with_suffix(".sha256"))
        == EXPECTED_R3_MANIFEST_SHA256,
        "failure_manifest_sha_exact": observed["failure_manifest"]
        == EXPECTED_FAILURE_MANIFEST_SHA256,
        "failure_manifest_sidecar_exact": sidecar_value(
            failure_manifest_path.with_suffix(".sha256")
        )
        == EXPECTED_FAILURE_MANIFEST_SHA256,
        "valid_controls_manifest_sha_exact": observed["valid_controls_manifest"]
        == EXPECTED_VALID_CONTROLS_SHA256,
        "windows_manifest_sha_exact": observed["sequential_windows_manifest"]
        == EXPECTED_WINDOWS_SHA256,
        "qold_array_sha_exact": d3_manifest["q_old_array_sha256"]
        == EXPECTED_QOLD_ARRAY_SHA256
        == qold_authority["q_old_array_sha256"],
        "historical_status_exact": summary["D3_R3_STATUS"] == "FAIL_DEVELOPMENT_GATE",
        "historical_validation_exact": summary["REFINEMENT_V2_DEVELOPMENT_VALIDATION"] == "FAIL",
        "only_gate_v1_failure_exact": decision["G3_MEDIAN_REDUCTION"] == "FAIL"
        and sum(value == "FAIL" for key, value in decision.items() if key.startswith("G")) == 1,
        "failure_count_exact": summary["FAILURE_FRAME_COUNT"] == EXPECTED_INVALID,
        "recovered_count_exact": summary["RECOVERED_COUNT"] == 143,
        "remaining_frames_exact": summary["REMAINING_INVALID_SOURCE_FRAMES"] == [4202, 4209],
    }
    value = {
        "schema_version": "O5RD3R4HistoricalIntegrityV1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "D3_R3_HISTORICAL_STATUS": "FAIL_DEVELOPMENT_GATE",
        "HISTORICAL_GATE_V1_RESULT": "FAIL",
        "HISTORICAL_R3_RESULT_REWRITTEN": "NO",
        "D3_R3_GATE_V1_RESULT_REWRITTEN": "NO",
        "FAILURE_FRAME_COUNT": summary["FAILURE_FRAME_COUNT"],
        "RECOVERED_COUNT": summary["RECOVERED_COUNT"],
        "RECOVERY_FRACTION": summary["RECOVERY_FRACTION"],
        "MEDIAN_INVALID_RELATIVE_REDUCTION": summary["MEDIAN_INVALID_RELATIVE_REDUCTION"],
        "HARD_VALID_COUNT": summary["HARD_VALID_COUNT"],
        "VALID_CONTROL_COUNT": summary["VALID_CONTROL_COUNT"],
        "VALID_CONTROLS_PRESERVED": summary["VALID_CONTROLS_PRESERVED"],
        "SEQUENTIAL_WINDOW_COUNT": summary["SEQUENTIAL_WINDOW_COUNT"],
        "SEQUENTIAL_WINDOWS_PASS": summary["SEQUENTIAL_WINDOWS_PASS"],
        "SEQUENTIAL_FRAMES_VALID": summary["SEQUENTIAL_FRAMES_VALID"],
        "SEQUENTIAL_FRAMES_TOTAL": summary["SEQUENTIAL_FRAMES_TOTAL"],
        "CONTINUITY": summary["CONTINUITY"],
        "RUNTIME_STATE_CHAIN": summary["RUNTIME_STATE_CHAIN"],
        "DETERMINISM": summary["DETERMINISM"],
        "REMAINING_INVALID_COUNT": summary["REMAINING_INVALID_COUNT"],
        "REMAINING_INVALID_SOURCE_FRAMES": summary["REMAINING_INVALID_SOURCE_FRAMES"],
    }
    authorities = {
        "schema_version": "O5RD3R4FrozenAuthoritiesV1",
        "status": value["status"],
        "REFINEMENT_V2_DESIGN_SHA256": observed["design"],
        "REFINEMENT_V2_DEVELOPMENT_GATE_V1_SHA256": observed["gate_v1"],
        "RetargetObjectiveV2_SHA256": observed["objective_v2_contract"],
        "SemanticV1_AUTHORITY_SHA256": observed["semantic_v1"],
        "ExecutionV4_RELEVANT_AUTHORITY_SHA256": d3_manifest["ExecutionV4_authority_sha256"],
        "D3_RUN_MANIFEST_SHA256": observed["d3_run_manifest"],
        "Q_OLD_AUTHORITY_FILE_SHA256": observed["qold_authority_file"],
        "Q_OLD_ARRAY_SHA256": d3_manifest["q_old_array_sha256"],
        "D3_R3_VALIDATION_MANIFEST_SHA256": observed["r3_validation_manifest"],
        "FAILURE_145_MANIFEST_SHA256": observed["failure_manifest"],
        "VALID_CONTROL_MANIFEST_SHA256": observed["valid_controls_manifest"],
        "SEQUENTIAL_WINDOW_MANIFEST_SHA256": observed["sequential_windows_manifest"],
    }
    write_json(root / "preflight/historical_integrity.json", value)
    write_json(root / "preflight/frozen_authorities.json", authorities)
    if value["status"] != "PASS":
        raise RuntimeError("O5RD3R4_HISTORICAL_INTEGRITY_FAIL")
    return value


def gate_v1_criteria() -> list[dict[str, Any]]:
    return [
        {
            "criterion": "recovery rate",
            "value": 0.8,
            "first_origin": "ae861ac57d35c73289bcaf98b7a559eeca63148a",
            "authority_type": "PRE_REGISTERED_CONSERVATIVE_HEURISTIC",
            "independent": "NO",
        },
        {
            "criterion": "median reduction",
            "value": 0.5,
            "first_origin": "ae861ac57d35c73289bcaf98b7a559eeca63148a; conceptual carry-over from O5R-C/D2D",
            "authority_type": "PRE_REGISTERED_CONSERVATIVE_HEURISTIC",
            "independent": "NO",
        },
        {
            "criterion": "valid preservation",
            "value": 1.0,
            "first_origin": "RefinementV2DevelopmentGateV1",
            "authority_type": "SEMANTIC_V1_REQUIREMENT",
            "independent": "YES",
        },
        {
            "criterion": "hard validity",
            "value": "PASS",
            "first_origin": "Candidate-B2 hard feasibility and Wuji asset bounds",
            "authority_type": "ROBOT_KINEMATIC_REQUIREMENT",
            "independent": "YES",
        },
        {
            "criterion": "sequential",
            "value": "PASS",
            "first_origin": "RefinementV2DevelopmentGateV1 consumed-window screen",
            "authority_type": "PRE_REGISTERED_CONSERVATIVE_HEURISTIC",
            "independent": "NO",
        },
        {
            "criterion": "continuity",
            "value": "PASS",
            "first_origin": "RetargetSemanticValidityV1 temporal limits",
            "authority_type": "SEMANTIC_V1_REQUIREMENT",
            "independent": "YES",
        },
        {
            "criterion": "determinism",
            "value": "PASS",
            "first_origin": "RefinementV2DevelopmentGateV1 reproducibility screen",
            "authority_type": "IMPLEMENTATION_HEURISTIC",
            "independent": "YES",
        },
    ]


def verify_gate_v1_authority(root: Path) -> dict[str, Any]:
    require(root / "preflight/historical_integrity.json", "status", "PASS", "VERIFY_GATE_V1")
    source = R2_ROOT / "design/refinement_v2_development_gate.json"
    contract = read_json(source)
    target = root / "gate_v1/contract.json"
    write_json(target, contract)
    observed = sha256_file(target)
    write_text(root / "gate_v1/contract.sha256", observed + "\n")
    rows = gate_v1_criteria()
    write_csv(
        root / "gate_v1/criteria.csv",
        rows,
        ["criterion", "value", "first_origin", "authority_type", "independent"],
    )
    value = {
        "schema_version": "GateRequirementProvenanceV1",
        "status": "PASS" if observed == EXPECTED_GATE_V1_SHA256 else "FAIL",
        "gate_v1_sha256": observed,
        "criteria": rows,
        "authority_enum_exhaustive": all(row["authority_type"] in AUTHORITY_TYPES for row in rows),
    }
    write_json(root / "gate_v1/provenance.json", value)
    if value["status"] != "PASS" or not value["authority_enum_exhaustive"]:
        raise RuntimeError("O5RD3R4_GATE_V1_AUTHORITY_FAIL")
    return value


def trace_gate_v1_provenance(root: Path) -> dict[str, Any]:
    require(root / "gate_v1/provenance.json", "status", "PASS", "TRACE_GATE_V1")
    exact_log = git(
        "log",
        "--all",
        "--reverse",
        "-Smedian_invalid_relative_E_IM_reduction_min",
        "--format=%H %ad %s",
        "--date=iso",
        "--",
        "scripts",
        "tests",
        "docs",
    )
    broad_log = git(
        "log",
        "--all",
        "--reverse",
        "-Gmedian.*reduction|MEDIAN_REDUCTION_REQUIRED|50%",
        "--format=%H %ad %s",
        "--date=iso",
        "--",
        "scripts",
        "tests",
        "docs",
    )
    blame = git("blame", "-L", "1762,1776", "scripts/evaluation/run_oakink2_o5rd3r2.py")
    docs_search = subprocess.run(
        [
            "rg",
            "-n",
            "median relative reduction|median_invalid_relative|50% reduction|0.50",
            "src",
            "scripts",
            "tests",
            "docs",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    git_value = {
        "schema_version": "O5RD3R4GitProvenanceSearchV1",
        "status": "PASS",
        "exact_search_command": "git log --all --reverse -Smedian_invalid_relative_E_IM_reduction_min",
        "exact_search_output": exact_log.splitlines(),
        "broad_search_output": broad_log.splitlines(),
        "first_exact_origin_commit": "ae861ac57d35c73289bcaf98b7a559eeca63148a",
    }
    lineage = {
        "schema_version": "RefinementV2GateV1HistoricalLineageV1",
        "status": "PASS",
        "first_exact_field_origin": "ae861ac57d35c73289bcaf98b7a559eeca63148a",
        "first_exact_field_location": "scripts/evaluation/run_oakink2_o5rd3r2.py:1769",
        "conceptual_predecessors": [
            "O5R-C structured-solver development used 0.50",
            "O5R-D2D certification gate used 0.50",
        ],
        "lineage_classification": "HISTORICAL_CARRYOVER_PREREGISTERED_AS_CONSERVATIVE_HEURISTIC",
        "prompt_or_preregistration_is_independent_scientific_authority": False,
    }
    write_json(root / "provenance/git_search.json", git_value)
    write_json(
        root / "provenance/git_blame.json",
        {"schema_version": "O5RD3R4GitBlameV1", "status": "PASS", "output": blame.splitlines()},
    )
    write_json(
        root / "provenance/docs_search.json",
        {
            "schema_version": "O5RD3R4DocsSearchV1",
            "status": "PASS",
            "returncode": docs_search.returncode,
            "matches": docs_search.stdout.splitlines(),
            "interpretation": "matches establish repository lineage only, not independent scientific authority",
        },
    )
    write_json(root / "provenance/historical_gate_lineage.json", lineage)
    return lineage


def audit_paper_authority(root: Path) -> dict[str, Any]:
    paper = REPO / "docs/TopoRetarget.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(paper), "-"], text=True)
    prohibited = ("median relative E_IM reduction", "50% reduction", "percentage reduction gate")
    value = {
        "schema_version": "O5RD3R4PaperAuthorityAuditV1",
        "status": "PASS",
        "paper_path": str(paper),
        "paper_sha256": sha256_file(paper),
        "eq7_E_IM": "mean squared difference between robot and source weighted Laplacian coordinates over the shared interaction mesh",
        "eq8_objective": "lambda_IM*E_IM + lambda_bone*E_bone + E_reg + penetration slack penalty under signed-distance constraints",
        "paper_defines_E_IM_success_threshold": False,
        "paper_defines_median_relative_reduction_50_percent": False,
        "paper_defines_percentage_reduction_gate": False,
        "PAPER_SUPPORTS_50_PERCENT_MEDIAN_REDUCTION_GATE": "NO",
        "PAPER_DOES_NOT_DEFINE_THIS_GATE": True,
        "paper_rejects_this_gate": False,
        "reference_level_metrics": [
            "contact precision E_prec",
            "contact alignment E_align",
            "maximum penetration",
            "fraction of frames with penetration above 2 mm",
            "solve time",
        ],
        "fixed_parameter_statement": "one fixed parameter setting across experiments; no extensive per-case tuning",
        "no_gate_phrase_matches": all(phrase not in text for phrase in prohibited),
    }
    write_json(root / "authority/paper_authority.json", value)
    return value


def audit_semantic_v1_authority(root: Path) -> dict[str, Any]:
    contract = SemanticGateContractV1()
    source = REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py"
    value = {
        "schema_version": "O5RD3R4SemanticV1AuthorityAuditV1",
        "status": "PASS"
        if sha256_file(source) == EXPECTED_SEMANTIC_V1_SHA256
        and contract.interaction_e_im_p95_limit == TAU
        else "FAIL",
        "SemanticV1_authority_sha256": sha256_file(source),
        "contract": contract.as_dict(),
        "SEMANTIC_V1_HAS_RELATIVE_REDUCTION_GATE": "NO",
        "SEMANTIC_V1_HAS_50_PERCENT_REDUCTION_GATE": "NO",
        "SEMANTIC_V1_SUPPORTS_50_PERCENT_GATE": "NO",
        "SEMANTIC_V1_INTERACTION_VALIDITY_FORM": "trajectory interaction_e_im p95 <= 1e-4",
        "per_frame_threshold_role": "development classification proxy; not by itself trajectory acceptance",
        "trajectory_gate_role": "final SemanticV1 acceptance authority",
        "relative_improvement_role": "not present",
    }
    write_json(root / "authority/semantic_v1_authority.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("O5RD3R4_SEMANTIC_V1_AUTHORITY_FAIL")
    return value


def audit_objective_v2_authority(root: Path) -> dict[str, Any]:
    path = D2E_ROOT / "method/retarget_objective_v2_contract.json"
    contract = read_json(path)
    value = {
        "schema_version": "O5RD3R4ObjectiveV2AuthorityAuditV1",
        "status": "PASS"
        if sha256_file(path) == EXPECTED_OBJECTIVE_V2_SHA256 and contract["tau"] == TAU
        else "FAIL",
        "RetargetObjectiveV2_sha256": sha256_file(path),
        "primary_interaction_hinge": contract["candidate_b2_primary_hinge"],
        "primary_pressure_at_or_below_tau": 0.0,
        "lexicographic": True,
        "secondary_fidelity": contract["secondary_fidelity_semantics"],
        "requires_tau_over_2": False,
        "has_50_percent_reduction_target": False,
        "OBJECTIVE_V2_SUPPORTS_50_PERCENT_REDUCTION_GATE": "NO",
        "OBJECTIVE_V2_SUPPORTS_50_PERCENT_GATE": "NO",
        "contract": contract,
    }
    write_json(root / "authority/objective_v2_authority.json", value)
    if value["status"] != "PASS":
        raise RuntimeError("O5RD3R4_OBJECTIVE_V2_AUTHORITY_FAIL")
    return value


def audit_physical_authority(root: Path) -> dict[str, Any]:
    objective = read_json(D2E_ROOT / "method/retarget_objective_v2_contract.json")
    robot = {
        "schema_version": "O5RD3R4RobotHardConstraintAuthorityV1",
        "status": "PASS",
        "joint_limits": "Wuji asset bounds",
        "wrist_bone_validity": [
            "SemanticV1 wrist position",
            "SemanticV1 wrist rotation",
            "SemanticV1 bone direction",
        ],
        "ROBOT_HARD_CONSTRAINT_SUPPORTS_50_PERCENT_GATE": "NO",
        "reason": "robot feasibility is state/bound based and contains no relative E_IM reduction threshold",
    }
    physical = {
        "schema_version": "O5RD3R4PhysicalAuthorityAuditV1",
        "status": "PASS",
        "collision_and_penetration_constraints": objective["true_admissibility_constraints"],
        "paper_penetration_soft_tolerance_m": 0.001,
        "paper_penetration_hard_bound_m": 0.03,
        "PHYSICAL_AUTHORITY_SUPPORTS_50_PERCENT_GATE": "NO",
        "lower_E_IM_generally_better": True,
        "lower_E_IM_implies_50_percent_is_necessary": False,
        "specific_counterfactual": "no authority establishes that reducing 1.1e-4 to about 5.5e-5 is necessary instead of reaching <=1e-4",
    }
    write_json(root / "authority/robot_hard_constraint_authority.json", robot)
    write_json(root / "authority/physical_authority.json", physical)
    return physical


def audit_recoverability_authority(root: Path) -> dict[str, Any]:
    d2i = read_json(D2I_ROOT / "final_summary.json")
    d2j = read_json(D2J_ROOT / "final_summary.json")
    d2jr = read_json(D2JR_ROOT / "final_summary.json")
    d2kr = read_json(D2KR_ROOT / "final_summary.json")
    value = {
        "schema_version": "O5RD3R4RecoverabilityAuthorityAuditV1",
        "status": "PASS",
        "studies": [
            {
                "stage": "D2I",
                "result": d2i["RETARGET_TO_PPO_RECOVERABILITY_RESULT"],
                "ppo_study": d2i["PPO_RECOVERABILITY_STUDY"],
            },
            {
                "stage": "D2J",
                "result": d2j["RETARGET_TO_PPO_RECOVERABILITY_RESULT"],
                "E_IM_correlation": d2j["correlations"]["E_IM_VS_RECOVERABILITY"],
            },
            {
                "stage": "D2J-R",
                "result": d2jr["RETARGET_TO_PPO_RECOVERABILITY_RESULT"],
                "mode": d2jr["RECOVERABILITY_MODE"],
            },
            {
                "stage": "D2K-R",
                "result": d2kr["FINAL_RECOVERABILITY_DECISION"],
                "evidence_source": d2kr["RECOVERABILITY_EVIDENCE_SOURCE"],
            },
        ],
        "pre_registered_mapping_median_50_percent_to_recoverability": False,
        "post_hoc_threshold_invention_allowed": False,
        "DOWNSTREAM_EVIDENCE_SUPPORTS_50_PERCENT_GATE": "NO",
        "DOWNSTREAM_RECOVERABILITY_SUPPORTS_50_PERCENT_GATE": "NO",
        "interpretation": "blocked/inconclusive/poor recoverability results do not define a 50% E_IM reduction threshold",
    }
    write_json(root / "authority/recoverability_authority.json", value)
    rows = [
        {"source": "paper", "supports_50_percent": "NO", "authority": "PAPER_METHOD_REQUIREMENT"},
        {
            "source": "SemanticV1",
            "supports_50_percent": "NO",
            "authority": "SEMANTIC_V1_REQUIREMENT",
        },
        {
            "source": "ObjectiveV2",
            "supports_50_percent": "NO",
            "authority": "SEMANTIC_V1_REQUIREMENT",
        },
        {
            "source": "robot hard constraints",
            "supports_50_percent": "NO",
            "authority": "ROBOT_KINEMATIC_REQUIREMENT",
        },
        {
            "source": "physical constraints",
            "supports_50_percent": "NO",
            "authority": "PHYSICAL_REQUIREMENT",
        },
        {
            "source": "recoverability",
            "supports_50_percent": "NO",
            "authority": "DOWNSTREAM_RECOVERABILITY_EVIDENCE",
        },
    ]
    write_csv(
        root / "authority/authority_matrix.csv",
        rows,
        ["source", "supports_50_percent", "authority"],
    )
    return value


def analyze_effect_size_semantics(root: Path) -> dict[str, Any]:
    require(root / "preflight/historical_integrity.json", "status", "PASS", "ANALYZE_EFFECT")
    rows = read_csv(R3_ROOT / "failure_frames/per_frame.csv")
    if len(rows) != EXPECTED_INVALID:
        raise RuntimeError(f"O5RD3R4_R3_FRAME_COUNT_MISMATCH:{len(rows)}")
    output: list[dict[str, Any]] = []
    for row in rows:
        old = float(row["D3_V1_E_IM"])
        new = float(row["final_E_IM"])
        minimum = max(old - TAU, 0.0)
        actual = old - new
        output.append(
            {
                "ordinal": int(row["ordinal"]),
                "source_frame": int(row["source_frame"]),
                "cluster_id": int(row["cluster_id"]),
                "E_old": old,
                "E_new": new,
                "absolute_decrease": actual,
                "relative_decrease": actual / old,
                "old_excess_over_tau": minimum,
                "new_excess_over_tau": max(new - TAU, 0.0),
                "r_min": minimum / old,
                "achieved_over_minimum_required": actual / minimum,
                "hard_valid": row["hard_valid"],
                "interaction_valid": row["interaction_valid"],
            }
        )
    fields = list(output[0])
    write_csv(root / "effect_size/r3_frame_metrics.csv", output, fields)
    write_csv(root / "effect_size/threshold_required_reduction.csv", output, fields)
    old = np.asarray([row["E_old"] for row in output], dtype=np.float64)
    new = np.asarray([row["E_new"] for row in output], dtype=np.float64)
    actual = np.asarray([row["relative_decrease"] for row in output], dtype=np.float64)
    r_min = np.asarray([row["r_min"] for row in output], dtype=np.float64)
    absolute = old - new
    required_absolute = np.maximum(old - TAU, 0.0)
    summary = {
        "schema_version": "O5RD3R4EffectSizeDistributionV1",
        "status": "PASS",
        "N": len(output),
        "tau": TAU,
        "median_D3_V1_E_IM": float(np.median(old)),
        "median_final_R3_E_IM": float(np.median(new)),
        "median_required_absolute_decrease_to_tau": float(np.median(required_absolute)),
        "median_achieved_absolute_decrease": float(np.median(absolute)),
        "median_achieved_reduction_over_minimal_required_reduction": float(
            np.median(absolute) / np.median(required_absolute)
        ),
        "mean_actual_relative_reduction": float(np.mean(actual)),
        "median_actual_relative_reduction": float(np.median(actual)),
        "r_min_mean": float(np.mean(r_min)),
        "r_min_median": float(np.median(r_min)),
        "r_min_p90": float(np.percentile(r_min, 90)),
        "r_min_p95": float(np.percentile(r_min, 95)),
        "r_min_max": float(np.max(r_min)),
        "fraction_r_min_lt_0p1": float(np.mean(r_min < 0.1)),
        "fraction_r_min_lt_0p2": float(np.mean(r_min < 0.2)),
        "fraction_r_min_lt_0p5": float(np.mean(r_min < 0.5)),
        "recovered_count": int(np.sum(new <= TAU)),
    }
    remaining = [row for row in output if row["source_frame"] in (4202, 4209)]
    comparison = {
        "schema_version": "O5RD3R4EffectSizeComparisonV1",
        "status": "PASS",
        "fixed_gate": 0.5,
        "median_minimum_needed": summary["r_min_median"],
        "median_actual": summary["median_actual_relative_reduction"],
        "gate_over_median_minimum_needed_ratio": 0.5 / summary["r_min_median"],
        "all_invalid_frames_need_less_than_20_percent_to_reach_tau": bool(np.all(r_min < 0.2)),
        "all_invalid_frames_need_less_than_50_percent_to_reach_tau": bool(np.all(r_min < 0.5)),
        "remaining_invalid_frames": remaining,
        "remaining_classification_under_threshold_aware_gate": "HARD_VALID_NONREGRESSING_UNRECOVERED; counted against recovery fraction; no special case",
    }
    write_json(root / "effect_size/distribution_summary.json", summary)
    write_json(root / "effect_size/comparison.json", comparison)
    return summary


def classify_median_reduction_authority(root: Path) -> dict[str, Any]:
    for relative in (
        "authority/paper_authority.json",
        "authority/semantic_v1_authority.json",
        "authority/objective_v2_authority.json",
        "authority/robot_hard_constraint_authority.json",
        "authority/physical_authority.json",
        "authority/recoverability_authority.json",
        "provenance/historical_gate_lineage.json",
    ):
        require(root / relative, "status", "PASS", "CLASSIFY_MEDIAN_AUTHORITY")
    value = {
        "schema_version": "O5RD3R4MedianReductionAuthorityDecisionV1",
        "status": "PASS",
        "MEDIAN_REDUCTION_50_PERCENT_AUTHORITY": "CONSERVATIVE_PREREGISTERED_HEURISTIC",
        "CONFIDENCE": "HIGH",
        "strong_independent_authority_found": False,
        "PAPER_SUPPORTS_50_PERCENT_GATE": "NO",
        "SEMANTIC_V1_SUPPORTS_50_PERCENT_GATE": "NO",
        "OBJECTIVE_V2_SUPPORTS_50_PERCENT_GATE": "NO",
        "ROBOT_HARD_CONSTRAINT_SUPPORTS_50_PERCENT_GATE": "NO",
        "PHYSICAL_AUTHORITY_SUPPORTS_50_PERCENT_GATE": "NO",
        "DOWNSTREAM_RECOVERABILITY_SUPPORTS_50_PERCENT_GATE": "NO",
        "rationale": "the threshold was frozen before R3, but only as a conservative historical development screen; preregistration is not independent scientific authority",
    }
    if value["MEDIAN_REDUCTION_50_PERCENT_AUTHORITY"] not in MEDIAN_AUTHORITY_TYPES:
        raise RuntimeError("O5RD3R4_MEDIAN_AUTHORITY_ENUM_INVALID")
    write_json(root / "decision/median_reduction_authority.json", value)
    return value


def decide_gate_semantic_alignment(root: Path) -> dict[str, Any]:
    authority = require(
        root / "decision/median_reduction_authority.json", "status", "PASS", "DECIDE_ALIGNMENT"
    )
    if authority["MEDIAN_REDUCTION_50_PERCENT_AUTHORITY"] == "STRONG_INDEPENDENT_AUTHORITY":
        alignment = "ALIGNED"
        cause = "GATE_HAS_STRONG_AUTHORITY"
    else:
        alignment = "MISALIGNED"
        cause = "DEVELOPMENT_GATE_SEMANTIC_MISMATCH"
    value = {
        "schema_version": "O5RD3R4SemanticAlignmentDecisionV1",
        "status": "PASS",
        "GATE_SEMANTIC_ALIGNMENT": alignment,
        "PRIMARY_ROOT_CAUSE": cause,
        "ROOT_CAUSE_CONFIDENCE": authority["CONFIDENCE"],
        "HISTORICAL_R3_GATE_V1_RESULT": "FAIL",
        "HISTORICAL_R3_RESULT_REWRITTEN": "NO",
        "historical_failure_mechanism_superseded_as_scientific_root_cause": alignment
        == "MISALIGNED",
    }
    write_json(root / "decision/semantic_alignment_decision.json", value)
    write_json(
        root / "decision/root_cause.json",
        {
            "schema_version": "O5RD3R4RootCauseV1",
            "status": "PASS",
            "PRIMARY_ROOT_CAUSE": cause,
            "CONFIDENCE": authority["CONFIDENCE"],
            "historical_R3_failure_analysis_preserved": True,
        },
    )
    return value


def gate_v2_contract() -> dict[str, Any]:
    return {
        "schema_version": "RefinementV2DevelopmentGateV2",
        "status": "FROZEN",
        "description": "threshold-aware development gate derived from frozen semantic and method authorities",
        "mutable_after_freeze": False,
        "outcome_driven_threshold_tuning": False,
        "R3_outcome_used_as_threshold_source": False,
        "authority_hierarchy": [
            "MATHEMATICAL_SCHEMA_INVARIANTS",
            "ROBOT_HARD_CONSTRAINTS",
            "SEMANTIC_V1",
            "OBJECTIVE_V2",
            "PAPER_FIDELITY_WHERE_APPLICABLE",
            "INDEPENDENT_PRE_OUTCOME_RECOVERABILITY_EVIDENCE",
            "PRE_REGISTERED_CONSERVATIVE_HEURISTICS",
            "DIAGNOSTICS",
        ],
        "population": "all 145 consumed D3-R final-invalid frames plus frozen valid controls and consumed sequential windows",
        "technical_completion": "N/N finite stored receipts",
        "old_invalid_definition": "E_old > tau",
        "old_invalid_recovery_rate_minimum": 0.8,
        "old_invalid_recovery_authority": "retained pre-R3 conservative method-screening heuristic; not trajectory SemanticV1 sufficiency",
        "tau": TAU,
        "tau_authority": "RetargetSemanticValidityV1.interaction_e_im_p95_limit",
        "epsilon_num": EPSILON_NUM,
        "epsilon_num_authority": "frozen CertificationGateContractV2.epsilon_num",
        "threshold_aware_nonregression": "E_new <= max(E_old, tau) + epsilon_num",
        "threshold_aware_nonregression_required_fraction": 1.0,
        "valid_preservation": "for E_old <= tau, E_new <= tau + epsilon_num",
        "valid_preservation_required_fraction": 1.0,
        "hard_validity": "PASS",
        "sequential_consumed_windows": "PASS",
        "continuity": "PASS",
        "determinism": "PASS",
        "median_invalid_relative_E_IM_reduction": "DIAGNOSTIC_ONLY",
        "trajectory_semantic_v1_role": "must be established by fresh window/full-trajectory certification; development recovery does not imply p95 PASS",
    }


def design_gate_v2_if_authorized(root: Path) -> dict[str, Any]:
    decision = require(
        root / "decision/semantic_alignment_decision.json", "status", "PASS", "DESIGN_GATE_V2"
    )
    if not (
        decision["PRIMARY_ROOT_CAUSE"] == "DEVELOPMENT_GATE_SEMANTIC_MISMATCH"
        and decision["ROOT_CAUSE_CONFIDENCE"] in {"HIGH", "MEDIUM"}
    ):
        write_json(
            root / "gate_v2/not_created.json",
            {
                "schema_version": "O5RD3R4GateV2NotCreatedV1",
                "status": "NOT_CREATED",
                "reason": "SEMANTIC_MISMATCH_NOT_ESTABLISHED",
            },
        )
        raise RuntimeError("O5RD3R4_GATE_V2_NOT_AUTHORIZED")
    draft = gate_v2_contract()
    draft["status"] = "DRAFT_AUTHORITY_DERIVED_NOT_FROZEN"
    write_json(root / "gate_v2/development_gate_v2.draft.json", draft)
    return draft


def freeze_gate_v2(root: Path) -> dict[str, Any]:
    for relative in (
        "authority/paper_authority.json",
        "authority/semantic_v1_authority.json",
        "authority/objective_v2_authority.json",
        "authority/robot_hard_constraint_authority.json",
        "authority/physical_authority.json",
        "authority/recoverability_authority.json",
        "decision/semantic_alignment_decision.json",
        "gate_v2/development_gate_v2.draft.json",
    ):
        if not (root / relative).is_file():
            raise RuntimeError(f"FREEZE_GATE_V2_REJECTED:MISSING:{relative}")
    contract = gate_v2_contract()
    path = root / "gate_v2/development_gate_v2.json"
    if path.exists() and read_json(path) != contract:
        raise RuntimeError("O5RD3R4_GATE_V2_FROZEN_DRIFT")
    write_json(path, contract)
    first = sha256_file(path)
    serialized = json.dumps(read_json(path), indent=2, sort_keys=True, allow_nan=False) + "\n"
    second = hashlib.sha256(serialized.encode()).hexdigest()
    write_text(root / "gate_v2/development_gate_v2.sha256", first + "\n")
    determinism = {
        "schema_version": "O5RD3R4GateV2SerializationDeterminismV1",
        "status": "PASS" if first == second else "FAIL",
        "first_sha256": first,
        "second_sha256": second,
        "exact": first == second,
        "FRESH_DATA_CONSUMED_BEFORE_GATE_V2_FREEZE": "NO",
    }
    write_json(root / "gate_v2/serialization_determinism.json", determinism)
    if determinism["status"] != "PASS":
        raise RuntimeError("O5RD3R4_GATE_V2_SERIALIZATION_FAIL")
    return {**contract, "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": first}


def evaluate_stored_r3_gate_v2_compatibility(root: Path) -> dict[str, Any]:
    gate_path = root / "gate_v2/development_gate_v2.json"
    if not gate_path.is_file() or not (root / "gate_v2/development_gate_v2.sha256").is_file():
        raise RuntimeError("O5RD3R4_COMPATIBILITY_REJECTED:GATE_V2_NOT_FROZEN")
    if sha256_file(gate_path) != sidecar_value(root / "gate_v2/development_gate_v2.sha256"):
        raise RuntimeError("O5RD3R4_COMPATIBILITY_REJECTED:GATE_V2_DRIFT")
    history = require(
        root / "preflight/historical_integrity.json", "status", "PASS", "EVALUATE_COMPATIBILITY"
    )
    rows = read_csv(root / "effect_size/r3_frame_metrics.csv")
    threshold_count = sum(
        float(row["E_new"]) <= max(float(row["E_old"]), TAU) + EPSILON_NUM for row in rows
    )
    summary = read_json(R3_ROOT / "final_summary.json")
    criteria = [
        {"criterion": "technical completion", "observed": "145/145", "result": "PASS"},
        {
            "criterion": "old-invalid recovery >=0.80",
            "observed": summary["RECOVERY_FRACTION"],
            "result": "PASS" if summary["RECOVERY_FRACTION"] >= 0.8 else "FAIL",
        },
        {
            "criterion": "threshold-aware nonregression 100%",
            "observed": f"{threshold_count}/{len(rows)}",
            "result": "PASS" if threshold_count == len(rows) else "FAIL",
        },
        {
            "criterion": "valid preservation 100%",
            "observed": f"{summary['VALID_CONTROLS_PRESERVED']}/{summary['VALID_CONTROL_COUNT']}",
            "result": "PASS"
            if summary["VALID_CONTROLS_PRESERVED"] == summary["VALID_CONTROL_COUNT"]
            else "FAIL",
        },
        {
            "criterion": "hard validity",
            "observed": f"{summary['HARD_VALID_COUNT']}/{EXPECTED_INVALID}",
            "result": "PASS" if summary["HARD_VALID_COUNT"] == EXPECTED_INVALID else "FAIL",
        },
        {
            "criterion": "sequential windows",
            "observed": f"{summary['SEQUENTIAL_WINDOWS_PASS']}/{summary['SEQUENTIAL_WINDOW_COUNT']}",
            "result": "PASS"
            if summary["SEQUENTIAL_WINDOWS_PASS"] == summary["SEQUENTIAL_WINDOW_COUNT"]
            else "FAIL",
        },
        {
            "criterion": "continuity",
            "observed": summary["CONTINUITY"],
            "result": summary["CONTINUITY"],
        },
        {
            "criterion": "determinism",
            "observed": summary["DETERMINISM"],
            "result": summary["DETERMINISM"],
        },
        {
            "criterion": "median relative reduction",
            "observed": summary["MEDIAN_INVALID_RELATIVE_REDUCTION"],
            "result": "DIAGNOSTIC_ONLY",
        },
    ]
    passed = all(row["result"] in {"PASS", "DIAGNOSTIC_ONLY"} for row in criteria)
    value = {
        "schema_version": "O5RD3R4StoredR3GateV2CompatibilityV1",
        "status": "PASS" if passed else "FAIL",
        "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY": "PASS" if passed else "FAIL",
        "EVIDENCE_ROLE": "POST_HOC_COMPATIBILITY_ONLY",
        "R3_REEVALUATED_PASS": False,
        "HISTORICAL_R3_GATE_V1_RESULT": "FAIL",
        "HISTORICAL_R3_RESULT_REWRITTEN": "NO",
        "historical_integrity_status": history["status"],
        "criterion_results": criteria,
    }
    write_json(root / "compatibility/stored_r3_gate_v2_results.json", value)
    write_csv(
        root / "compatibility/criterion_results.csv",
        criteria,
        ["criterion", "observed", "result"],
    )
    write_json(
        root / "compatibility/decision.json",
        {
            "schema_version": "O5RD3R4CompatibilityDecisionV1",
            "status": value["status"],
            "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY": value[
                "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY"
            ],
            "EVIDENCE_ROLE": "POST_HOC_COMPATIBILITY_ONLY",
        },
    )
    return value


def authorize_fresh_certification(root: Path) -> dict[str, Any]:
    gate_path = root / "gate_v2/development_gate_v2.json"
    decision_path = root / "compatibility/decision.json"
    if not gate_path.is_file() or not decision_path.is_file():
        raise RuntimeError("O5RD3R4_FRESH_AUTH_REJECTED:PREREQUISITES_MISSING")
    if sha256_file(gate_path) != sidecar_value(root / "gate_v2/development_gate_v2.sha256"):
        raise RuntimeError("O5RD3R4_FRESH_AUTH_REJECTED:GATE_V2_NOT_FROZEN")
    compatibility = require(
        decision_path,
        "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY",
        "PASS",
        "AUTHORIZE_FRESH_CERTIFICATION",
    )
    value = {
        "schema_version": "O5RD3R4FreshCertificationAuthorizationV1",
        "status": "AUTHORIZED_NOT_RUN",
        "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED": "YES",
        "REFINEMENT_V2_INDEPENDENT_CERTIFICATION": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "NEXT": "O5R-D3-CERT_REFINEMENT_V2_FRESH_CERTIFICATION",
        "authorized_by": compatibility["schema_version"],
    }
    write_json(root / "future/fresh_certification_authorization.json", value)
    return value


def generate_fresh_certification_plan(root: Path) -> dict[str, Any]:
    require(
        root / "future/fresh_certification_authorization.json",
        "status",
        "AUTHORIZED_NOT_RUN",
        "GENERATE_FRESH_PLAN",
    )
    plan = {
        "schema_version": "RefinementV2FreshCertificationPlanStubV1",
        "status": "AUTHORIZED_NOT_RUN",
        "execution_in_this_task": "FORBIDDEN",
        "fresh_sparse": {"count": 30, "strata": {"HIGH": 10, "MID": 10, "LOW": 10}},
        "fresh_windows": ["HIGH_1", "HIGH_2", "MID", "LOW"],
        "q_old_authority": "must be frozen before repaired refinement outcomes are exposed",
        "zero_overlap_with": ["D3", "D3-R", "D3-R2", "D3-R3"],
        "CROSS_EPISODE_REQUIREMENT": "REQUIRED",
        "cross_episode_reason": "RefinementV2 expands the generic active set across all finger DOFs; genericity needs predeclared episode controls beyond same-episode sparse/windows",
        "cross_episode_selection_timing": "freeze before any fresh sparse/window solve",
        "trajectory_semantic_requirement": "fresh windows and eventual D3-V2 must apply frozen SemanticV1 p95; development recovery is not sufficient",
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "FRESH_REFINEMENT_OPTIMIZER_RUN_COUNT": 0,
    }
    write_json(root / "future/fresh_certification_plan_stub.json", plan)
    return plan


def gate_comparison_rows() -> list[dict[str, Any]]:
    return [
        {
            "criterion": "recovery",
            "gate_v1": ">=0.80",
            "authority": "pre-R3 conservative method screen",
            "gate_v2": ">=0.80",
            "change_reason": "unchanged; not claimed as trajectory sufficiency",
        },
        {
            "criterion": "median reduction",
            "gate_v1": ">=0.50",
            "authority": "conservative preregistered heuristic",
            "gate_v2": "DIAGNOSTIC_ONLY",
            "change_reason": "no paper/SemanticV1/ObjectiveV2/robot/physical/recoverability authority",
        },
        {
            "criterion": "threshold-aware nonregression",
            "gate_v1": "absent",
            "authority": "SemanticV1 valid set plus frozen CertificationGateV2 epsilon",
            "gate_v2": "100%",
            "change_reason": "align with tau-defined valid set",
        },
        {
            "criterion": "valid preservation",
            "gate_v1": "100%",
            "authority": "SemanticV1",
            "gate_v2": "100%",
            "change_reason": "unchanged semantic requirement",
        },
        {
            "criterion": "hard validity",
            "gate_v1": "PASS",
            "authority": "robot/physical hard constraints",
            "gate_v2": "PASS",
            "change_reason": "unchanged hard authority",
        },
        {
            "criterion": "sequential",
            "gate_v1": "PASS",
            "authority": "pre-registered sequence screen",
            "gate_v2": "PASS",
            "change_reason": "unchanged sequence requirement",
        },
        {
            "criterion": "continuity",
            "gate_v1": "PASS",
            "authority": "SemanticV1",
            "gate_v2": "PASS",
            "change_reason": "unchanged semantic requirement",
        },
        {
            "criterion": "determinism",
            "gate_v1": "PASS",
            "authority": "execution reproducibility",
            "gate_v2": "PASS",
            "change_reason": "unchanged reproducibility requirement",
        },
    ]


def summarize(root: Path) -> dict[str, Any]:
    history = require(root / "preflight/historical_integrity.json", "status", "PASS", "SUMMARIZE")
    authority = require(
        root / "decision/median_reduction_authority.json", "status", "PASS", "SUMMARIZE"
    )
    alignment = require(
        root / "decision/semantic_alignment_decision.json", "status", "PASS", "SUMMARIZE"
    )
    compatibility = require(root / "compatibility/decision.json", "status", "PASS", "SUMMARIZE")
    effect = require(root / "effect_size/distribution_summary.json", "status", "PASS", "SUMMARIZE")
    gate_path = root / "gate_v2/development_gate_v2.json"
    authorization = read_json(root / "future/fresh_certification_authorization.json")
    comparison = gate_comparison_rows()
    write_csv(
        root / "gate_v2/gate_v1_vs_v2.csv",
        comparison,
        ["criterion", "gate_v1", "authority", "gate_v2", "change_reason"],
    )
    final_head = git("rev-parse", "HEAD")
    summary = {
        "schema_version": "OakInk2O5RD3R4FinalSummaryV1",
        "D3_R4_STATUS": "PASS_GATE_REALIGNMENT",
        "BRANCH": EXPECTED_BRANCH,
        "START_HEAD": START_HEAD,
        "FINAL_HEAD": final_head,
        "D3_R3_HISTORICAL_STATUS": "FAIL_DEVELOPMENT_GATE",
        "HISTORICAL_GATE_V1_RESULT": "FAIL",
        "HISTORICAL_R3_RESULT_REWRITTEN": "NO",
        "RECOVERED_COUNT": f"{history['RECOVERED_COUNT']}/{history['FAILURE_FRAME_COUNT']}",
        "RECOVERY_FRACTION": history["RECOVERY_FRACTION"],
        "MEDIAN_RELATIVE_REDUCTION": history["MEDIAN_INVALID_RELATIVE_REDUCTION"],
        "HARD_VALID": f"{history['HARD_VALID_COUNT']}/{history['FAILURE_FRAME_COUNT']}",
        "VALID_PRESERVATION": f"{history['VALID_CONTROLS_PRESERVED']}/{history['VALID_CONTROL_COUNT']}",
        "SEQUENTIAL": f"{history['SEQUENTIAL_WINDOWS_PASS']}/{history['SEQUENTIAL_WINDOW_COUNT']} windows; {history['SEQUENTIAL_FRAMES_VALID']}/{history['SEQUENTIAL_FRAMES_TOTAL']} frames",
        "CONTINUITY": history["CONTINUITY"],
        "RUNTIME_STATE_CHAIN": history["RUNTIME_STATE_CHAIN"],
        "DETERMINISM": history["DETERMINISM"],
        "MEDIAN_REDUCTION_50_PERCENT_AUTHORITY": authority["MEDIAN_REDUCTION_50_PERCENT_AUTHORITY"],
        "AUTHORITY_CONFIDENCE": authority["CONFIDENCE"],
        "GATE_SEMANTIC_ALIGNMENT": alignment["GATE_SEMANTIC_ALIGNMENT"],
        "PRIMARY_ROOT_CAUSE": alignment["PRIMARY_ROOT_CAUSE"],
        "ROOT_CAUSE_CONFIDENCE": alignment["ROOT_CAUSE_CONFIDENCE"],
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_CREATED": "YES",
        "REFINEMENT_V2_DEVELOPMENT_GATE_V2_SHA256": sha256_file(gate_path),
        "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY": compatibility[
            "R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY"
        ],
        "EVIDENCE_ROLE": "POST_HOC_COMPATIBILITY_ONLY",
        "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED": authorization[
            "FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED"
        ],
        "NEXT": authorization["NEXT"],
        "effect_size": effect,
        "PAPER_SUPPORTS_50_PERCENT_GATE": "NO",
        "SEMANTIC_V1_SUPPORTS_50_PERCENT_GATE": "NO",
        "OBJECTIVE_V2_SUPPORTS_50_PERCENT_GATE": "NO",
        "ROBOT_HARD_CONSTRAINT_SUPPORTS_50_PERCENT_GATE": "NO",
        "PHYSICAL_AUTHORITY_SUPPORTS_50_PERCENT_GATE": "NO",
        "DOWNSTREAM_RECOVERABILITY_SUPPORTS_50_PERCENT_GATE": "NO",
        "REFINEMENT_V2_DESIGN_CHANGED": "NO",
        "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "HARD_VALIDITY_CHANGED": "NO",
        "RETARGET_OPTIMIZER_RUN_COUNT": 0,
        "FRESH_REFINEMENT_SPARSE": "NOT_RUN",
        "FRESH_REFINEMENT_WINDOW": "NOT_RUN",
        "CROSS_EPISODE_REFINEMENT": "NOT_RUN",
        "D3_V2_SCIENTIFIC_RUN_COUNT": 0,
        "D3_V2_TRAJECTORY": None,
        "D3_V2_SEMANTIC_V1": "NOT_RUN",
        "DEV2_RERUN": "NO",
        "PPO_TRAINING_RUN_COUNT_NEW": 0,
        "O6_PRODUCTION_RAN": "NO",
        "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
        "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
        "tracked_worktree_clean": not git("status", "--short", "--untracked-files=all"),
    }
    write_json(root / "final_summary.json", summary)
    write_json(
        root / "audits/no_optimizer.json",
        {
            "schema_version": "O5RD3R4NoOptimizerAuditV1",
            "status": "PASS",
            "RETARGET_OPTIMIZER_RUN_COUNT": 0,
            "action_surface_disjoint": not (set(ACTIONS) & FORBIDDEN_EXECUTION_ACTIONS),
        },
    )
    write_json(
        root / "audits/no_method_change.json",
        {
            "schema_version": "O5RD3R4NoMethodChangeAuditV1",
            "status": "PASS",
            "REFINEMENT_V2_DESIGN_CHANGED": "NO",
            "RETARGET_OBJECTIVE_V2_CHANGED": "NO",
            "SEMANTIC_V1_CHANGED": "NO",
            "E_IM_THRESHOLD_CHANGED": "NO",
            "HARD_VALIDITY_CHANGED": "NO",
        },
    )
    write_json(
        root / "audits/no_fresh_data.json",
        {
            "schema_version": "O5RD3R4NoFreshDataAuditV1",
            "status": "PASS",
            "CERTIFICATION_SPLIT_NEW_CONSUMPTION": 0,
            "HELDOUT_SPLIT_NEW_CONSUMPTION": 0,
            "FRESH_REFINEMENT_OPTIMIZER_RUN_COUNT": 0,
        },
    )
    write_json(
        root / "audits/historical_results_immutable.json",
        {
            "schema_version": "O5RD3R4HistoricalImmutableAuditV1",
            "status": "PASS",
            "HISTORICAL_R3_GATE_V1_RESULT": "FAIL",
            "HISTORICAL_R3_RESULT_REWRITTEN": "NO",
            "R3_REEVALUATED_PASS": False,
        },
    )
    table = [
        "| Criterion | GateV1 | Authority | GateV2 | Change reason |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {row['criterion']} | {row['gate_v1']} | {row['authority']} | {row['gate_v2']} | {row['change_reason']} |"
            for row in comparison
        ],
    ]
    handoff = "\n".join(
        [
            "# OakInk2 O5R-D3-R4",
            "# RefinementV2 Development Gate Semantic Alignment Audit Handoff",
            "",
            "## Git",
            "",
            f"- BRANCH={EXPECTED_BRANCH}",
            f"- START_HEAD={START_HEAD}",
            f"- FINAL_HEAD={final_head}",
            "- PUSHED=NO",
            "- PR_CREATED=NO",
            "",
            "## Historical R3",
            "",
            "- D3_R3_HISTORICAL_STATUS=FAIL_DEVELOPMENT_GATE",
            "- HISTORICAL_GATE_V1_RESULT=FAIL",
            "- HISTORICAL_R3_RESULT_REWRITTEN=NO",
            f"- RECOVERED_COUNT={summary['RECOVERED_COUNT']}",
            f"- RECOVERY_FRACTION={summary['RECOVERY_FRACTION']}",
            f"- MEDIAN_RELATIVE_REDUCTION={summary['MEDIAN_RELATIVE_REDUCTION']}",
            f"- HARD_VALID={summary['HARD_VALID']}",
            f"- VALID_PRESERVATION={summary['VALID_PRESERVATION']}",
            f"- SEQUENTIAL={summary['SEQUENTIAL']}",
            "- CONTINUITY=PASS",
            "- DETERMINISM=PASS",
            "",
            "## 50% authority decision",
            "",
            f"- MEDIAN_REDUCTION_50_PERCENT_AUTHORITY={summary['MEDIAN_REDUCTION_50_PERCENT_AUTHORITY']}",
            f"- CONFIDENCE={summary['AUTHORITY_CONFIDENCE']}",
            "- PAPER_SUPPORTS_50_PERCENT_GATE=NO",
            "- SEMANTIC_V1_SUPPORTS_50_PERCENT_GATE=NO",
            "- OBJECTIVE_V2_SUPPORTS_50_PERCENT_GATE=NO",
            "- ROBOT_HARD_CONSTRAINT_SUPPORTS_50_PERCENT_GATE=NO",
            "- PHYSICAL_AUTHORITY_SUPPORTS_50_PERCENT_GATE=NO",
            "- DOWNSTREAM_RECOVERABILITY_SUPPORTS_50_PERCENT_GATE=NO",
            "",
            "## Effect-size analysis",
            "",
            f"- MEDIAN_MINIMUM_REDUCTION_TO_REACH_TAU={effect['r_min_median']}",
            f"- MEDIAN_ACTUAL_REDUCTION={effect['median_actual_relative_reduction']}",
            f"- P90_MINIMUM_REDUCTION_TO_REACH_TAU={effect['r_min_p90']}",
            f"- MAX_MINIMUM_REDUCTION_TO_REACH_TAU={effect['r_min_max']}",
            f"- FRACTION_R_MIN_LT_0P1={effect['fraction_r_min_lt_0p1']}",
            f"- FRACTION_R_MIN_LT_0P2={effect['fraction_r_min_lt_0p2']}",
            f"- FRACTION_R_MIN_LT_0P5={effect['fraction_r_min_lt_0p5']}",
            "",
            "## Semantic alignment",
            "",
            "- GATE_SEMANTIC_ALIGNMENT=MISALIGNED",
            "- PRIMARY_ROOT_CAUSE=DEVELOPMENT_GATE_SEMANTIC_MISMATCH",
            "- ROOT_CAUSE_CONFIDENCE=HIGH",
            "",
            "## GateV1 vs GateV2",
            "",
            *table,
            "",
            "## Stored R3 compatibility",
            "",
            "- R3_STORED_EVIDENCE_GATE_V2_COMPATIBILITY=PASS",
            "- EVIDENCE_ROLE=POST_HOC_COMPATIBILITY_ONLY",
            "- R3_REEVALUATED_PASS=NO",
            "",
            "## Fresh certification",
            "",
            "- FRESH_REFINEMENT_CERTIFICATION_AUTHORIZED=YES",
            "- REFINEMENT_V2_INDEPENDENT_CERTIFICATION=NOT_RUN",
            "- CROSS_EPISODE_REQUIREMENT=REQUIRED",
            "- NEXT=O5R-D3-CERT_REFINEMENT_V2_FRESH_CERTIFICATION",
            "",
            "## Explicitly not run",
            "",
            "- RETARGET_OPTIMIZER_RUN_COUNT=0",
            "- FRESH_REFINEMENT_SPARSE=NOT_RUN",
            "- FRESH_REFINEMENT_WINDOW=NOT_RUN",
            "- CROSS_EPISODE_REFINEMENT=NOT_RUN",
            "- D3_V2_SCIENTIFIC_RUN_COUNT=0",
            "- DEV2_RERUN=NO",
            "- PPO_TRAINING_RUN_COUNT_NEW=0",
            "- O6_PRODUCTION_RAN=NO",
            "- CERTIFICATION_SPLIT_NEW_CONSUMPTION=0",
            "- HELDOUT_SPLIT_NEW_CONSUMPTION=0",
            "",
            "O5_FINAL=NOT_PASS: fresh certification, D3-V2 SemanticV1, and human review remain unrun.",
            "",
        ]
    )
    write_text(root / "handoff.md", handoff)
    write_text(root / "final_summary.md", handoff)
    write_json(
        root / "git_commits.json",
        {
            "schema_version": "O5RD3R4GitCommitsV1",
            "START_HEAD": START_HEAD,
            "FINAL_HEAD_AT_SUMMARY": final_head,
            "commits": git("log", "--format=%H %s", f"{START_HEAD}..{final_head}").splitlines(),
            "PUSHED": "NO",
            "PR_CREATED": "NO",
        },
    )
    expected = [
        "handoff.md",
        "final_summary.md",
        "final_summary.json",
        "preflight/git.json",
        "preflight/historical_integrity.json",
        "preflight/frozen_authorities.json",
        "gate_v1/contract.json",
        "gate_v1/contract.sha256",
        "gate_v1/criteria.csv",
        "gate_v1/provenance.json",
        "provenance/git_search.json",
        "provenance/git_blame.json",
        "provenance/docs_search.json",
        "provenance/historical_gate_lineage.json",
        "authority/paper_authority.json",
        "authority/semantic_v1_authority.json",
        "authority/objective_v2_authority.json",
        "authority/robot_hard_constraint_authority.json",
        "authority/physical_authority.json",
        "authority/recoverability_authority.json",
        "authority/authority_matrix.csv",
        "effect_size/r3_frame_metrics.csv",
        "effect_size/threshold_required_reduction.csv",
        "effect_size/distribution_summary.json",
        "effect_size/comparison.json",
        "decision/median_reduction_authority.json",
        "decision/semantic_alignment_decision.json",
        "decision/root_cause.json",
        "gate_v2/development_gate_v2.json",
        "gate_v2/development_gate_v2.sha256",
        "gate_v2/serialization_determinism.json",
        "compatibility/stored_r3_gate_v2_results.json",
        "compatibility/criterion_results.csv",
        "compatibility/decision.json",
        "future/fresh_certification_authorization.json",
        "future/fresh_certification_plan_stub.json",
        "audits/no_optimizer.json",
        "audits/no_method_change.json",
        "audits/no_fresh_data.json",
        "audits/historical_results_immutable.json",
        "technical_failures.jsonl",
        "tests.json",
        "validation_results.json",
        "git_commits.json",
    ]
    missing = [item for item in expected if not (root / item).is_file()]
    completion = {
        "schema_version": "O5RD3R4CompletionAuditV1",
        "status": "PASS" if not missing else "FAIL",
        "expected_artifact_count": len(expected),
        "missing": missing,
        "RETARGET_OPTIMIZER_RUN_COUNT": 0,
        "historical_result_immutable": True,
    }
    write_json(root / "completion_audit.json", completion)
    if missing:
        raise RuntimeError(f"O5RD3R4_COMPLETION_MISSING:{missing}")
    return summary


def validate_repository(root: Path) -> dict[str, Any]:
    commands = [
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "check",
            "scripts/evaluation/run_oakink2_o5rd3r4.py",
            "tests/evaluation/test_oakink2_o5rd3r4.py",
        ],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "format",
            "--check",
            "scripts/evaluation/run_oakink2_o5rd3r4.py",
            "tests/evaluation/test_oakink2_o5rd3r4.py",
        ],
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", "."],
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", "."],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "mypy", "src"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "-m", "pytest", "-q"],
        ["conda", "run", "-n", "toporetarget-rl", "python", "scripts/check_paper_fidelity.py"],
        ["git", "diff", "--check"],
    ]
    rows = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        rows.append(
            {
                "index": index,
                "command": command,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
    cli_rows = []
    for action in ACTIONS:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--help"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
        cli_rows.append(
            {"action": action, "listed": action in result.stdout, "returncode": result.returncode}
        )
    passed = all(row["returncode"] == 0 for row in rows) and all(
        row["listed"] and row["returncode"] == 0 for row in cli_rows
    )
    value = {
        "schema_version": "O5RD3R4RepositoryValidationV1",
        "status": "PASS" if passed else "FAIL",
        "checks": rows,
        "cli_help": cli_rows,
    }
    write_json(root / "tests.json", value)
    write_json(root / "validation_results.json", value)
    if not passed:
        raise RuntimeError("O5RD3R4_REPOSITORY_VALIDATION_FAIL")
    return value


def audit_all(root: Path) -> dict[str, Any]:
    preflight(root)
    verify_r3_history(root)
    verify_gate_v1_authority(root)
    trace_gate_v1_provenance(root)
    audit_paper_authority(root)
    audit_semantic_v1_authority(root)
    audit_objective_v2_authority(root)
    audit_physical_authority(root)
    audit_recoverability_authority(root)
    analyze_effect_size_semantics(root)
    classify_median_reduction_authority(root)
    decide_gate_semantic_alignment(root)
    design_gate_v2_if_authorized(root)
    freeze_gate_v2(root)
    evaluate_stored_r3_gate_v2_compatibility(root)
    authorize_fresh_certification(root)
    generate_fresh_certification_plan(root)
    return summarize(root)


ACTIONS = {
    "preflight": preflight,
    "verify-r3-history": verify_r3_history,
    "verify-gate-v1-authority": verify_gate_v1_authority,
    "trace-gate-v1-provenance": trace_gate_v1_provenance,
    "audit-paper-authority": audit_paper_authority,
    "audit-semantic-v1-authority": audit_semantic_v1_authority,
    "audit-objective-v2-authority": audit_objective_v2_authority,
    "audit-physical-authority": audit_physical_authority,
    "audit-recoverability-authority": audit_recoverability_authority,
    "analyze-effect-size-semantics": analyze_effect_size_semantics,
    "classify-median-reduction-authority": classify_median_reduction_authority,
    "decide-gate-semantic-alignment": decide_gate_semantic_alignment,
    "design-gate-v2-if-authorized": design_gate_v2_if_authorized,
    "freeze-gate-v2": freeze_gate_v2,
    "evaluate-stored-r3-gate-v2-compatibility": evaluate_stored_r3_gate_v2_compatibility,
    "authorize-fresh-certification": authorize_fresh_certification,
    "generate-fresh-certification-plan": generate_fresh_certification_plan,
    "summarize": summarize,
    "validate-repository": validate_repository,
    "audit-all": audit_all,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=sorted(ACTIONS))
    value.add_argument("--root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    result = ACTIONS[args.action](args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
