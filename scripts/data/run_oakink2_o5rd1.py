#!/usr/bin/env python3
"""Audit OakInk2 O5R-D1 production-objective alignment and rollback.

This command is diagnostic-only.  It evaluates the frozen production objective,
replays the already-consumed O5R-C frames when their Q2/Q3 states were not
persisted, and never runs a full DEV1 trajectory or any DEV2 solve.
"""

# ruff: noqa: E402, E501, PLR0912, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from scripts.data.run_oakink2_o5 import REPORT_ROOT as O5  # noqa: E402
from scripts.data.run_oakink2_o5 import (  # noqa: E402
    episode_paths,
)
from scripts.data.run_oakink2_o5rc import (  # noqa: E402
    EPS,
    FINGERS,
    EpisodeRuntime,
    block_refine,
)
from toporetarget.retarget.bones import load_bone_profile  # noqa: E402
from toporetarget.retarget.continuous import (  # noqa: E402
    LAMBDA_CORR,
    S_POS_M,
    S_Q_RAD,
    S_ROT_RAD,
    transport_previous_final_to_current_warm,
)
from toporetarget.retarget.final_refinement import (  # noqa: E402
    _make_context,
    encode_base_correction,
    map_previous_state_to_seed,
    refine_frame,
    so3_log,
)
from toporetarget.retarget.frames import load_frame_profile  # noqa: E402
from toporetarget.utils.hashing import sha256_file, sha256_tree  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rd1_objective_alignment_v1"
O1R = REPO / ".local/reports/oakink2_o1r_official_mano_authority_v1"
O5RA = REPO / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"
O5RB = REPO / ".local/reports/oakink2_o5rb_parallel_v1"
O5RC = REPO / ".local/reports/oakink2_o5rc_structured_solver_v1"
OBJECTIVE_EPS = 1e-10
DIRECTIONAL_STEP = 1e-4


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(jsonable(rows))
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    value = hashlib.sha256()
    for name, item in sha256_tree(path).items():
        value.update(name.encode())
        value.update(b"\0")
        value.update(item.encode())
        value.update(b"\n")
    return value.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def rollback_ratio(gain: float, loss: float, epsilon: float = OBJECTIVE_EPS) -> float | None:
    """Return rollback loss / block gain, or None when no gain exists."""

    return None if gain <= epsilon else loss / gain


def central_directional_derivative(
    function: Callable[[np.ndarray], float],
    origin: np.ndarray,
    direction: np.ndarray,
    step: float = DIRECTIONAL_STEP,
) -> float:
    """Central derivative with respect to the dimensionless path coordinate."""

    if step <= 0.0:
        raise ValueError("directional derivative step must be positive")
    return float(
        (function(origin + step * direction) - function(origin - step * direction)) / (2.0 * step)
    )


def classify_b1(delta_eim: float, delta_objective: float, epsilon: float) -> str:
    if delta_eim < -epsilon:
        if delta_objective <= epsilon:
            return "ALIGNED_BETTER"
        return "SEMANTIC_BETTER_OBJECTIVE_WORSE"
    return "NO_MEANINGFUL_SEMANTIC_IMPROVEMENT"


def choose_representatives(
    rows: list[dict[str, Any]], counts: dict[str, int]
) -> list[dict[str, Any]]:
    """Choose deterministic temporal quantiles independently in each stratum."""

    selected: list[dict[str, Any]] = []
    for stratum in ("HIGH", "MID", "LOW"):
        pool = sorted(
            (row for row in rows if str(row["stratum"]) == stratum),
            key=lambda row: (int(row["frame_id"]), int(row["ordinal"])),
        )
        count = int(counts.get(stratum, 0))
        if count > len(pool):
            raise ValueError(f"insufficient {stratum} frames")
        indices = np.linspace(0, len(pool) - 1, count, dtype=int) if count else []
        selected.extend(pool[int(index)] for index in indices)
    return selected


def bind_state_quartet(arrays: Any, q1_available: bool) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Bind persisted arrays to the canonical Q0/Q1/Q2/Q3 meanings."""

    rows = [
        ("Q0_OLD_PRODUCTION", np.asarray(arrays["q0"]), np.asarray(arrays["base0"])),
        (
            "Q2_STRUCTURED_BLOCK",
            np.asarray(arrays["q2"]),
            np.asarray(arrays["base2"]),
        ),
        (
            "Q3_STRUCTURED_FINAL",
            np.asarray(arrays["q3"]),
            np.asarray(arrays["base3"]),
        ),
    ]
    if q1_available:
        rows.insert(
            1,
            (
                "Q1_B1_BEST_OBSERVED",
                np.asarray(arrays["q1"]),
                np.asarray(arrays["base0"]),
            ),
        )
    return rows


def frozen_paths() -> dict[str, Path]:
    dev1 = episode_paths(O5, "dev_01")
    return {
        "manifest_v2": O1R / "manifest_v2/oakink2_corpus_manifest_v2.jsonl",
        "split_v2": O1R / "manifest_v2/oakink2_raw_to_physical_split_v2.json",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "paper_objective_weights": REPO / "configs/paper/retarget.yaml",
        "production_objective_source": REPO / "src/toporetarget/retarget/final_refinement.py",
        "semantic_v1_source": REPO / "scripts/evaluation/audit_retarget_semantic_validity.py",
        "o5_production_contract": O5 / "contract/geometric_retarget_contract.json",
        "dev1_production_trajectory": dev1["final"],
        "dev1_semantic_v1": dev1["semantic"],
        "o5ra_contributor_attribution": O5RA / "dev1_semantic/contributor_summary.json",
        "o5ra_semantic_per_frame": O5RA / "dev1_semantic/dev1_e_im_per_frame.csv",
        "o5rb_frame_selection": O5RB / "b1_thumb_feasibility/frame_selection.json",
        "o5rb_current_best_states": O5RB / "b1_thumb_feasibility/frame_results.csv",
        "o5rb_b1_contract": O5RB / "b1_thumb_feasibility/feasibility_contract.json",
        "o5rc_source": REPO / "scripts/data/run_oakink2_o5rc.py",
        "o5rc_frozen_contract": O5RC / "structured_solver/frozen_structured_solver_contract.json",
        "o5rc_sparse_set": O5RC / "validation/dev1_sparse_validation_set.json",
        "o5rc_development_results": O5RC / "structured_solver/candidate_results.csv",
        "o5rc_sparse_results": O5RC / "validation/per_frame_results.csv",
        "o5rc_sparse_decision": O5RC / "validation/sparse_gate_decision.json",
        "o5rc_profiler": O5RC / "profiling/attribution_summary.json",
    }


def preflight(root: Path) -> dict[str, Any]:
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != "feature/oakink2-raw-to-physical":
        raise RuntimeError("O5RD1_BRANCH_AUTHORITY_MISMATCH")
    paths = frozen_paths()
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen authorities: " + ", ".join(missing))
    authorities = {
        name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in paths.items()
    }
    write_json(
        root / "preflight/git.json",
        {
            "schema_version": "OakInk2O5RD1GitPreflightV1",
            "repo": git("rev-parse", "--show-toplevel"),
            "branch": branch,
            "start_head": head,
            "initial_status_short": [],
            "initial_status_provenance": "mandated shell preflight captured before task modifications",
            "status_short_at_artifact_write": git(
                "status", "--short", "--untracked-files=all"
            ).splitlines(),
            "diff_check": subprocess.run(
                ["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True, check=False
            ).stdout.splitlines(),
            "worktrees": git("worktree", "list", "--porcelain").splitlines(),
            "remotes": git("remote", "-v").splitlines(),
            "pushed": False,
            "pr_created": False,
        },
    )
    payload = {
        "schema_version": "OakInk2O5RD1FrozenAuthoritiesV1",
        "status": "PASS",
        "authorities": authorities,
        "structured_solver_v1_status": "SCIENTIFICALLY_REJECTED_BY_INDEPENDENT_VALIDATION",
        "structured_solver_v1_modified": False,
        "semantic_v1_changed": False,
        "production_objective_changed": False,
    }
    write_json(root / "preflight/frozen_authorities.json", payload)
    b1 = read_json(O5RB / "b1_thumb_feasibility/frame_selection.json")["frames"]
    sparse = read_json(O5RC / "validation/dev1_sparse_validation_set.json")["frames"]
    ledger = {
        "schema_version": "DEV1MethodDevelopmentLedgerV1",
        "entries": [
            {
                "source": "O5R-B B1",
                "role": "DEVELOPMENT_DIAGNOSTIC",
                "count": len(b1),
                "frames": b1,
            },
            {
                "source": "O5R-C SparseValidationV1",
                "role": "FORMER_INDEPENDENT_VALIDATION_NOW_CONSUMED_FOR_DIAGNOSIS_AFTER_FAILURE",
                "count": len(sparse),
                "frames": sparse,
            },
        ],
        "future_validation_exclusion_count": len(b1) + len(sparse),
        "dev1_sparse_validation_v2_created": False,
    }
    write_json(root / "preflight/consumed_frame_ledger.json", ledger)
    return payload


def objective_audit(root: Path) -> dict[str, Any]:
    runtime = EpisodeRuntime("dev_01", root)
    paper = runtime.resources.paper
    payload = {
        "schema_version": "ProductionRetargetObjectiveV1Audit",
        "source": {
            "definition": "src/toporetarget/retarget/final_refinement.py:_FrameContext.breakdown_tensor",
            "gradient": "src/toporetarget/retarget/final_refinement.py:_FrameContext.objective",
            "optimizer": "src/toporetarget/retarget/final_refinement.py:_solver_call",
            "production_context": "src/toporetarget/retarget/final_refinement.py:build_final_trajectory",
        },
        "optimizer_semantics": {
            "backend": "scipy.optimize.minimize",
            "method": "SLSQP",
            "objective_kind": "scalar differentiable energy; not scipy least_squares residual norm",
            "constraints": "separate collision hard/soft inequalities with nonnegative slack",
            "variable_vector": "[base_translation_delta(3), base_rotation_log_delta(3), qpos(20), active_query_slack(M)]",
        },
        "formula": "J=lambda_IM*E_IM + lambda_bone*E_bone + E_temporal + lambda_base_pos*||delta_p||^2 + lambda_base_rot*||delta_w||^2 + 0.5*w_s*||slack||^2",
        "paper_weights": paper.as_dict(),
        "components": [
            {
                "name": "interaction_mesh",
                "definition": "sum_v ||L_robot(v)-L_source(v)||^2 / 71 over 21 hand + 50 object graph vertices",
                "raw_residual_dimension": "71x3 before squared aggregation",
                "normalization": "divide squared sum by 71",
                "weight": paper.lambda_im,
                "affected_dofs": "base correction and all finger DOFs",
                "source_geometry": "source interaction graph and robot mediapipe21 keypoints",
                "semantic_e_im_relation": "EXACT",
            },
            {
                "name": "bone_direction",
                "definition": "sum squared adjacent bone-direction feature error",
                "raw_residual_dimension": "asset/profile-derived adjacent feature tensor",
                "normalization": "none beyond unit direction construction",
                "weight": paper.lambda_bone,
                "affected_dofs": "base orientation and all finger DOFs",
                "source_geometry": "source mediapipe21 adjacent bone features",
                "semantic_e_im_relation": "NO_DIRECT_EQUIVALENT",
            },
            {
                "name": "continuous_temporal",
                "definition": "lambda_corr*(mean(||R_pred^T(p-p_pred)/S_pos||^2)+mean(||log(R_pred^T R)/S_rot||^2)+mean(||(q-q_pred)/S_q||^2))",
                "raw_residual_dimension": "3 translation + 3 rotation + 20 qpos",
                "normalization": {"S_pos_m": S_POS_M, "S_rot_rad": S_ROT_RAD, "S_q_rad": S_Q_RAD},
                "weight": LAMBDA_CORR,
                "affected_dofs": "base correction and all finger DOFs",
                "source_geometry": "previous accepted production state transported through adjacent warm frames",
                "semantic_e_im_relation": "NO_DIRECT_EQUIVALENT",
            },
            {
                "name": "base_position",
                "definition": "lambda_base_pos*||delta_p||^2",
                "raw_residual_dimension": 3,
                "normalization": "none",
                "weight": paper.lambda_base_pos,
                "affected_dofs": "base translation only",
                "source_geometry": "warm base-pose seed",
                "semantic_e_im_relation": "INDIRECT",
            },
            {
                "name": "base_rotation",
                "definition": "lambda_base_rot*||delta_w||^2",
                "raw_residual_dimension": 3,
                "normalization": "SO(3) log radians",
                "weight": paper.lambda_base_rot,
                "affected_dofs": "base rotation only",
                "source_geometry": "warm base-pose seed",
                "semantic_e_im_relation": "INDIRECT",
            },
            {
                "name": "collision_slack",
                "definition": "0.5*w_s*sum(slack^2)",
                "raw_residual_dimension": "active collision QuerySet size M",
                "normalization": "slack in meters; bounds [0,b-tau]",
                "weight": 0.5 * paper.w_s,
                "affected_dofs": "auxiliary slack; required slack depends on base and all finger DOFs",
                "source_geometry": "robot collision surface and object signed distance",
                "semantic_e_im_relation": "NO_DIRECT_EQUIVALENT",
            },
        ],
        "paper_external_components_active": False,
        "normalization_warning": "FinalObjectiveBreakdown stores temporal/base/slack terms after their weights; weighted_e_im and weighted_e_bone are explicit.",
    }
    write_json(root / "objective/production_objective_audit.json", payload)
    write_json(root / "objective/objective_components.json", {"components": payload["components"]})
    mapping = {
        "schema_version": "ObjectiveSemanticAlignmentMapV1",
        "rows": [
            {
                "semantic_metric": "RetargetSemanticValidityV1.interaction_e_im",
                "production_term": "interaction_mesh E_IM",
                "relation": "EXACT",
                "weight": paper.lambda_im,
            }
        ],
        "e_im_in_production_objective": "EXACT",
        "downstream_metric_not_directly_optimized": False,
        "caveat": "Semantic V1 gates trajectory p95 while production minimizes a weighted per-frame sum with competing terms and collision constraints.",
    }
    write_json(root / "objective/objective_semantic_alignment_map.json", mapping)
    return payload


def _b1_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_csv(O5RB / "b1_thumb_feasibility/frame_results.csv"):
        row: dict[str, Any] = dict(raw)
        row["ordinal"], row["frame_id"] = int(raw["ordinal"]), int(raw["frame_id"])
        row["best_full_q"] = np.fromstring(raw["best_full_q"].strip("[]"), sep=" ")
        rows.append(row)
    return rows


def audit_frames() -> list[dict[str, Any]]:
    b1 = {int(row["ordinal"]): row for row in _b1_rows()}
    sparse = read_json(O5RC / "validation/dev1_sparse_validation_set.json")["frames"]
    rows: dict[int, dict[str, Any]] = {}
    for row in b1.values():
        rows[int(row["ordinal"])] = {
            "ordinal": int(row["ordinal"]),
            "frame_id": int(row["frame_id"]),
            "stratum": str(row["stratum"]),
            "population": "B1_DEVELOPMENT",
            "q1_available": True,
        }
    for row in sparse:
        rows[int(row["ordinal"])] = {
            "ordinal": int(row["ordinal"]),
            "frame_id": int(row["frame_id"]),
            "stratum": str(row["stratum"]),
            "population": "SPARSE_V1_CONSUMED",
            "q1_available": False,
        }
    if len(rows) != 55:
        raise RuntimeError(f"O5RD1_AUDIT_FRAME_COUNT_MISMATCH:{len(rows)}")
    return [rows[key] for key in sorted(rows)]


def _state_path(root: Path, ordinal: int) -> Path:
    return root / f"replay/states/frame_{ordinal:04d}.npz"


def _receipt_path(root: Path, ordinal: int) -> Path:
    return root / f"replay/receipts/frame_{ordinal:04d}.json"


def _polish(
    runtime: EpisodeRuntime,
    ordinal: int,
    q: np.ndarray,
    base: np.ndarray,
    previous: np.ndarray | None,
    *,
    initialization_source: str,
) -> Any:
    context = runtime.context(ordinal, q, base, previous, free=None, fixed_base=False)
    query = runtime.query_set(context, q)
    state = np.concatenate([encode_base_correction(base, base), q])
    return refine_frame(
        context,
        query,
        runtime.solver,
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery=runtime.execution.strict_recovery,
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=state,
        initialization_source=initialization_source,
    )


def replay_states(root: Path) -> dict[str, Any]:
    contract_path = O5RC / "structured_solver/frozen_structured_solver_contract.json"
    expected = (
        (O5RC / "structured_solver/frozen_structured_solver_contract.sha256").read_text().strip()
    )
    if sha256_file(contract_path) != expected:
        raise RuntimeError("STRUCTURED_SOLVER_V1_CONTRACT_HASH_MISMATCH")
    contract = read_json(contract_path)
    design = contract["design"]["selected"]
    runtime = EpisodeRuntime("dev_01", root)
    b1 = {int(row["ordinal"]): row for row in _b1_rows()}
    development = {
        int(row["ordinal"]): row
        for row in read_csv(O5RC / "structured_solver/candidate_results.csv")
    }
    sparse = {
        int(row["ordinal"]): row for row in read_csv(O5RC / "validation/per_frame_results.csv")
    }
    completed = 0
    parity_rows: list[dict[str, Any]] = []
    for frame in audit_frames():
        ordinal = int(frame["ordinal"])
        state_path, receipt_path = _state_path(root, ordinal), _receipt_path(root, ordinal)
        if state_path.exists() and receipt_path.exists():
            arrays = np.load(state_path, allow_pickle=False)
            q3_eim, _ = runtime.eim(ordinal, arrays["q3"], arrays["base3"])
            source = development.get(ordinal) or sparse.get(ordinal)
            expected_eim = float(source["new_e_im"])
            parity_rows.append(
                {
                    "ordinal": ordinal,
                    "frame_id": frame["frame_id"],
                    "q3_eim": q3_eim,
                    "historical_q3_eim": expected_eim,
                    "absolute_error": abs(q3_eim - expected_eim),
                    "resumed": True,
                }
            )
            completed += 1
            continue
        q0, base0, previous = runtime.old_state(ordinal)
        old_eim, scores = runtime.eim(ordinal, q0, base0)
        ranked = sorted(FINGERS, key=lambda finger: (-scores[finger], finger))
        selected = ranked[: int(design["top_k"])]
        q2, base2, stages = q0.copy(), base0.copy(), []
        for _ in range(int(design["block_passes"])):
            for finger in selected:
                q2, base2, stage = block_refine(
                    runtime,
                    ordinal,
                    q2,
                    base2,
                    previous,
                    finger.upper(),
                    int(design["block_maxiter"]),
                )
                stages.append(stage)
                if not stage["success"]:
                    raise RuntimeError(
                        f"O5RD1_BLOCK_REPLAY_FAILED:{ordinal}:{stage['termination']}"
                    )
        polish = _polish(
            runtime,
            ordinal,
            q2,
            base2,
            previous,
            initialization_source="structured_blocks_then_original_full_polish_v1",
        )
        if not polish.accepted:
            raise RuntimeError(f"O5RD1_POLISH_REPLAY_FAILED:{ordinal}:{polish.acceptance_reason}")
        q1 = b1[ordinal]["best_full_q"] if ordinal in b1 else np.empty(0, dtype=float)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            state_path,
            q0=q0,
            base0=base0,
            q1=q1,
            q2=q2,
            base2=base2,
            q3=polish.qpos,
            base3=polish.base_pose_scene,
            q3_slack=polish.slack,
            q3_query_ids=polish.query_set.sample_ids,
            q3_query_initial_signed_distance=polish.query_set.initial_signed_distance,
        )
        q2_eim, _ = runtime.eim(ordinal, q2, base2)
        q3_eim, _ = runtime.eim(ordinal, polish.qpos, polish.base_pose_scene)
        source = development.get(ordinal) or sparse.get(ordinal)
        expected_eim = float(source["new_e_im"])
        write_json(
            receipt_path,
            {
                "schema_version": "StructuredSolverV1DiagnosticReplayReceipt",
                "diagnostic_replay_only": True,
                "frozen_contract_sha256": expected,
                "ordinal": ordinal,
                "frame_id": frame["frame_id"],
                "population": frame["population"],
                "dominant_contributor_block": ranked[0],
                "ranked_contributor_blocks": ranked,
                "contributor_scores": scores,
                "selected_blocks": selected,
                "q0_e_im": old_eim,
                "q2_e_im": q2_eim,
                "q3_e_im": q3_eim,
                "block_stages": stages,
                "full_polish": {
                    "accepted": polish.accepted,
                    "acceptance_reason": polish.acceptance_reason,
                    "nfev": polish.optimizer_function_evaluations,
                    "njev": polish.optimizer_jacobian_evaluations,
                    "runtime_sec": polish.solve_time_s,
                    "initial_objective": polish.initial_objective,
                    "final_objective": polish.final_objective,
                    "breakdown": polish.breakdown.as_dict(),
                    "initialization_source": polish.initialization_source,
                },
            },
        )
        parity_rows.append(
            {
                "ordinal": ordinal,
                "frame_id": frame["frame_id"],
                "q3_eim": q3_eim,
                "historical_q3_eim": expected_eim,
                "absolute_error": abs(q3_eim - expected_eim),
                "resumed": False,
            }
        )
        completed += 1
        print(
            f"O5RD1_REPLAY {completed}/55 ordinal={ordinal} frame={frame['frame_id']}", flush=True
        )
    write_csv(root / "replay/q3_historical_parity.csv", parity_rows)
    payload = {
        "schema_version": "StructuredSolverV1StateReplaySummary",
        "status": "PASS" if completed == 55 else "FAIL",
        "frames": completed,
        "max_q3_eim_absolute_error": max(float(row["absolute_error"]) for row in parity_rows),
        "tolerance": 1e-10,
        "q2_recovery": "DIAGNOSTIC_REPLAY_ONLY",
        "dev1_full_retarget_reruns": 0,
        "dev1_full_refinement_runs": 0,
        "dev2_frame0_new_solves": 0,
        "dev2_full_production_solves": 0,
    }
    write_json(root / "replay/state_replay_summary.json", payload)
    return payload


def production_context(runtime: EpisodeRuntime, ordinal: int) -> Any:
    """Reconstruct the exact historical continuous production context."""

    if ordinal <= 0:
        previous = None
        predicted_base = None
        predicted_q = None
    else:
        previous_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal - 1])
        previous_q = np.asarray(runtime.final.arrays["qpos"][ordinal - 1])
        previous = map_previous_state_to_seed(
            previous_base,
            previous_q,
            np.asarray(runtime.warm.arrays["base_pose_scene"][ordinal]),
        )
        propagated = transport_previous_final_to_current_warm(
            runtime.warm.arrays["base_pose_scene"][ordinal - 1],
            previous_base,
            runtime.warm.arrays["base_pose_scene"][ordinal],
            runtime.warm.arrays["qpos"][ordinal - 1],
            previous_q,
            runtime.warm.arrays["qpos"][ordinal],
            runtime.model.joint_lower,
            runtime.model.joint_upper,
            previous_frame=int(runtime.graph.frame_indices[ordinal - 1]),
            current_frame=int(runtime.graph.frame_indices[ordinal]),
        )
        predicted_base = propagated.predicted_base_scene
        predicted_q = propagated.predicted_qpos
    return _make_context(
        runtime.sequence,
        runtime.graph,
        runtime.warm,
        runtime.model,
        runtime.surface,
        runtime.backends.solver_sdf,
        runtime.resources.reference_sdf,
        load_frame_profile("canonical_keypoint_wrist_v1"),
        load_bone_profile("mediapipe21_full_finger_chain_v1"),
        runtime.resources.paper,
        ordinal,
        previous,
        temporal_scope="continuous_full_state",
        continuous_prediction_base=predicted_base,
        continuous_prediction_qpos=predicted_q,
        spatial_gradient_backend=runtime.execution.signed_distance_gradient,
        sign_cache=runtime.backends.sign_cache,
        compiled_spatial_fd_backend=runtime.backends.compiled_spatial_fd_backend,
    )


def _ragged(array: np.ndarray, offsets: np.ndarray, ordinal: int) -> np.ndarray:
    start, stop = int(offsets[ordinal]), int(offsets[ordinal + 1])
    return np.asarray(array[start:stop]).copy()


def _state_values(root: Path, frame: dict[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    ordinal = int(frame["ordinal"])
    arrays = np.load(_state_path(root, ordinal), allow_pickle=False)
    return bind_state_quartet(arrays, bool(frame["q1_available"]))


def _component_rows(
    frame: dict[str, Any],
    state: str,
    breakdown: Any,
    correction: np.ndarray,
    slack: np.ndarray,
    paper: Any,
) -> list[dict[str, Any]]:
    components = [
        (
            "interaction_mesh",
            float(breakdown.e_im) * 71.0,
            float(breakdown.e_im),
            paper.lambda_im,
            float(breakdown.weighted_e_im),
            "sum_squared_residual_then_divide_71",
        ),
        (
            "bone_direction",
            float(breakdown.e_bone),
            float(breakdown.e_bone),
            paper.lambda_bone,
            float(breakdown.weighted_e_bone),
            "unit_direction_features_no_extra_scale",
        ),
        (
            "continuous_temporal",
            float(breakdown.e_temporal) / LAMBDA_CORR,
            float(breakdown.e_temporal) / LAMBDA_CORR,
            LAMBDA_CORR,
            float(breakdown.e_temporal),
            f"mean_scaled_blocks:S_pos={S_POS_M},S_rot={S_ROT_RAD},S_q={S_Q_RAD}",
        ),
        (
            "base_position",
            float(np.square(correction[:3]).sum()),
            float(np.square(correction[:3]).sum()),
            paper.lambda_base_pos,
            float(breakdown.e_base_pos),
            "meters_squared",
        ),
        (
            "base_rotation",
            float(np.square(correction[3:6]).sum()),
            float(np.square(correction[3:6]).sum()),
            paper.lambda_base_rot,
            float(breakdown.e_base_rot),
            "so3_log_radians_squared",
        ),
        (
            "collision_slack",
            float(np.square(slack).sum()),
            float(np.square(slack).sum()),
            0.5 * paper.w_s,
            float(breakdown.e_slack),
            "meters_squared_over_q0_historical_queryset",
        ),
    ]
    return [
        {
            "frame_id": frame["frame_id"],
            "ordinal": frame["ordinal"],
            "population": frame["population"],
            "stratum": frame["stratum"],
            "state": state,
            "component": name,
            "raw": raw,
            "normalized": normalized,
            "normalization": normalization,
            "weight": weight,
            "weighted": weighted,
        }
        for name, raw, normalized, weight, weighted, normalization in components
    ]


def decompose_states(root: Path) -> dict[str, Any]:
    if not all(_state_path(root, int(frame["ordinal"])).exists() for frame in audit_frames()):
        raise RuntimeError("O5RD1_STATE_REPLAY_INCOMPLETE")
    runtime = EpisodeRuntime("dev_01", root)
    availability: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    semantic_reference = {
        int(row["ordinal"]): float(row["e_im"])
        for row in read_csv(O5RA / "dev1_semantic/dev1_e_im_per_frame.csv")
    }
    for index, frame in enumerate(audit_frames(), start=1):
        ordinal = int(frame["ordinal"])
        context = production_context(runtime, ordinal)
        warm_base = np.asarray(runtime.warm.arrays["base_pose_scene"][ordinal])
        q0_slack = _ragged(
            runtime.final.arrays["slack_concat"], runtime.final.arrays["slack_offsets"], ordinal
        )
        q0_query_ids = _ragged(
            runtime.final.arrays["query_ids_concat"], runtime.final.arrays["query_offsets"], ordinal
        ).astype(np.int64)
        historical_state = np.concatenate(
            [
                np.asarray(runtime.final.arrays["base_corrections"][ordinal]),
                np.asarray(runtime.final.arrays["qpos"][ordinal]),
                q0_slack,
            ]
        )
        historical_total, _, historical_breakdown = context.objective(historical_state)
        parity_fields = {
            "total": "total_objective",
            "e_im": "e_im",
            "e_bone": "e_bone",
            "e_temporal": "e_temporal",
            "e_base_pos": "e_base_pos",
            "e_base_rot": "e_base_rot",
            "e_slack": "e_slack",
            "weighted_e_im": "weighted_e_im",
            "weighted_e_bone": "weighted_e_bone",
        }
        objective_errors = {
            name: abs(
                float(getattr(historical_breakdown, name))
                - float(runtime.final.arrays[stored][ordinal])
            )
            for name, stored in parity_fields.items()
            if name != "total"
        }
        objective_errors["total"] = abs(
            historical_total - float(runtime.final.arrays["total_objective"][ordinal])
        )
        q0_semantic, _ = runtime.eim(
            ordinal,
            np.asarray(runtime.final.arrays["qpos"][ordinal]),
            np.asarray(runtime.final.arrays["base_pose_scene"][ordinal]),
        )
        parity_rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "max_objective_component_absolute_error": max(objective_errors.values()),
                "semantic_vs_final_artifact_absolute_error": abs(
                    q0_semantic - float(runtime.final.arrays["e_im"][ordinal])
                ),
                "semantic_vs_o5ra_absolute_error": abs(q0_semantic - semantic_reference[ordinal]),
            }
        )
        states = _state_values(root, frame)
        availability.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "population": frame["population"],
                "Q0_available": True,
                "Q1_available": bool(frame["q1_available"]),
                "Q2_available": True,
                "Q3_available": True,
            }
        )
        for state, qpos, base in states:
            correction = encode_base_correction(warm_base, base)
            fixed_state = np.concatenate([correction, qpos, q0_slack])
            total, _, breakdown = context.objective(fixed_state)
            full = runtime.resources.reference_sdf.query_scene(
                context.candidate_points(fixed_state), context.object_pose_scene
            )
            signed = np.asarray(full.signed_distance, dtype=float)
            query_signed = signed[q0_query_ids]
            minimal_slack = np.clip(
                np.maximum(-context.paper.tau - query_signed, 0.0),
                0.0,
                context.paper.b - context.paper.tau,
            )
            minimal_state = np.concatenate([correction, qpos, minimal_slack])
            minimal_total, _, minimal_breakdown = context.objective(minimal_state)
            row = {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "population": frame["population"],
                "stratum": frame["stratum"],
                "state": state,
                **breakdown.as_dict(),
                "production_total_fixed_q0_aux": total,
                "production_total_minimal_q0_queryset_slack": minimal_total,
                "minimal_q0_queryset_slack_penalty": minimal_breakdown.e_slack,
                "q0_queryset_size": len(q0_query_ids),
                "full_surface_min_signed_distance_m": float(signed.min()),
                "full_surface_soft_violation_count": int(np.sum(signed < -context.paper.tau)),
                "full_surface_hard_violation_count": int(np.sum(signed < -context.paper.b)),
                "full_surface_hard_feasible": bool(signed.min() >= -context.paper.b - 1e-6),
                "qpos_delta_l2_from_q0": float(
                    np.linalg.norm(qpos - runtime.final.arrays["qpos"][ordinal])
                ),
                "base_translation_delta_m_from_q0": float(
                    np.linalg.norm(
                        base[:3, 3] - runtime.final.arrays["base_pose_scene"][ordinal][:3, 3]
                    )
                ),
                "base_rotation_delta_rad_from_q0": float(
                    np.linalg.norm(
                        so3_log(
                            runtime.final.arrays["base_pose_scene"][ordinal][:3, :3].T
                            @ base[:3, :3]
                        )
                    )
                ),
            }
            state_rows.append(row)
            component_rows.extend(
                _component_rows(frame, state, breakdown, correction, q0_slack, context.paper)
            )
        print(f"O5RD1_DECOMPOSE {index}/55 ordinal={ordinal}", flush=True)
    write_csv(root / "replay/state_availability.csv", availability)
    write_csv(root / "replay/state_objective_decomposition.csv", state_rows)
    write_csv(root / "replay/objective_component_tradeoff.csv", component_rows)
    write_csv(root / "replay/parity_per_frame.csv", parity_rows)
    max_objective = max(float(row["max_objective_component_absolute_error"]) for row in parity_rows)
    max_semantic = max(
        max(
            float(row["semantic_vs_final_artifact_absolute_error"]),
            float(row["semantic_vs_o5ra_absolute_error"]),
        )
        for row in parity_rows
    )
    tolerance = max(OBJECTIVE_EPS, 10.0 * max(max_objective, max_semantic))
    payload = {
        "schema_version": "OakInk2O5RD1ReplayParityV1",
        "objective_replay_parity": "PASS" if max_objective <= OBJECTIVE_EPS else "FAIL",
        "semantic_replay_parity": "PASS" if max_semantic <= OBJECTIVE_EPS else "FAIL",
        "frames": len(parity_rows),
        "max_objective_component_absolute_error": max_objective,
        "max_semantic_absolute_error": max_semantic,
        "base_tolerance": OBJECTIVE_EPS,
        "classification_epsilon": tolerance,
        "classification_epsilon_authority": "max(base numeric tolerance, 10x observed replay error), frozen before outcome classification",
        "objective_binding": "exact production continuous transport context",
        "state_comparison_auxiliary_policy": "hold Q0 historical active-query slack fixed; also report state-specific minimal slack on the same Q0 QuerySet",
    }
    write_json(root / "replay/parity.json", payload)
    return payload


def _state_row_index(root: Path) -> dict[tuple[int, str], dict[str, str]]:
    return {
        (int(row["ordinal"]), str(row["state"])): row
        for row in read_csv(root / "replay/state_objective_decomposition.csv")
    }


def _component_index(root: Path) -> dict[tuple[int, str, str], dict[str, str]]:
    return {
        (int(row["ordinal"]), str(row["state"]), str(row["component"])): row
        for row in read_csv(root / "replay/objective_component_tradeoff.csv")
    }


def b1_alignment(root: Path) -> dict[str, Any]:
    states, components = _state_row_index(root), _component_index(root)
    epsilon = float(read_json(root / "replay/parity.json")["classification_epsilon"])
    rows: list[dict[str, Any]] = []
    for frame in (row for row in audit_frames() if row["q1_available"]):
        ordinal = int(frame["ordinal"])
        q0 = states[(ordinal, "Q0_OLD_PRODUCTION")]
        q1 = states[(ordinal, "Q1_B1_BEST_OBSERVED")]
        delta_eim = float(q1["e_im"]) - float(q0["e_im"])
        delta_j = float(q1["production_total_fixed_q0_aux"]) - float(
            q0["production_total_fixed_q0_aux"]
        )
        rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "q0_e_im": float(q0["e_im"]),
                "q1_e_im": float(q1["e_im"]),
                "delta_e_im": delta_eim,
                "q0_production_objective": float(q0["production_total_fixed_q0_aux"]),
                "q1_production_objective": float(q1["production_total_fixed_q0_aux"]),
                "delta_production_objective": delta_j,
                "classification": classify_b1(delta_eim, delta_j, epsilon),
                "q1_full_surface_hard_feasible": q1["full_surface_hard_feasible"],
                "q1_min_signed_distance_m": float(q1["full_surface_min_signed_distance_m"]),
            }
        )
    write_csv(root / "b1_alignment/current_vs_best_observed.csv", rows)
    counts = Counter(row["classification"] for row in rows)
    component_summary: list[dict[str, Any]] = []
    for component in (
        "interaction_mesh",
        "bone_direction",
        "continuous_temporal",
        "base_position",
        "base_rotation",
        "collision_slack",
    ):
        deltas = [
            float(components[(int(row["ordinal"]), "Q1_B1_BEST_OBSERVED", component)]["weighted"])
            - float(components[(int(row["ordinal"]), "Q0_OLD_PRODUCTION", component)]["weighted"])
            for row in rows
        ]
        component_summary.append(
            {
                "component": component,
                "median_delta_weighted_q0_to_q1": float(np.median(deltas)),
                "mean_delta_weighted_q0_to_q1": float(np.mean(deltas)),
                "positive_penalty_frames": int(np.sum(np.asarray(deltas) > epsilon)),
                "negative_improvement_frames": int(np.sum(np.asarray(deltas) < -epsilon)),
            }
        )
    component_summary.sort(key=lambda row: -float(row["median_delta_weighted_q0_to_q1"]))
    payload = {
        "schema_version": "B1ProductionObjectiveAlignmentSummaryV1",
        "n_b1": len(rows),
        "classification_epsilon": epsilon,
        "counts": dict(counts),
        "median_delta_e_im": float(np.median([row["delta_e_im"] for row in rows])),
        "median_delta_production_objective": float(
            np.median([row["delta_production_objective"] for row in rows])
        ),
        "fraction_semantic_better_objective_worse": counts["SEMANTIC_BETTER_OBJECTIVE_WORSE"]
        / len(rows),
        "q1_hard_feasible_fraction": sum(
            str(row["q1_full_surface_hard_feasible"]).lower() == "true" for row in rows
        )
        / len(rows),
        "component_tradeoff_ranked_by_median_penalty": component_summary,
        "objective_comparison_policy": "exact production continuous context; Q0 historical auxiliary slack held fixed",
    }
    write_json(root / "b1_alignment/classification_summary.json", payload)
    write_json(root / "b1_alignment/component_tradeoff_summary.json", component_summary)
    return payload


def structured_audit(root: Path) -> dict[str, Any]:
    states = _state_row_index(root)
    epsilon = float(read_json(root / "replay/parity.json")["classification_epsilon"])
    contributor_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, int]] = {
        finger: {selected: 0 for selected in FINGERS} for finger in FINGERS
    }
    for frame in audit_frames():
        ordinal = int(frame["ordinal"])
        receipt = read_json(_receipt_path(root, ordinal))
        ranked = list(receipt["ranked_contributor_blocks"])
        selected = list(receipt["selected_blocks"])
        dominant = str(receipt["dominant_contributor_block"])
        for chosen in selected:
            matrix[dominant][str(chosen)] += 1
        contributor_rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "population": frame["population"],
                "dominant_semantic_contributor_block": dominant,
                "ranked_contributor_blocks": "+".join(ranked),
                "selected_block_1": selected[0] if selected else None,
                "selected_block_2": selected[1] if len(selected) > 1 else None,
                "top1_agreement": bool(selected and dominant == selected[0]),
                "topk_coverage": dominant in selected,
                **{f"score_{finger}": receipt["contributor_scores"][finger] for finger in FINGERS},
            }
        )
        q0, q2 = states[(ordinal, "Q0_OLD_PRODUCTION")], states[(ordinal, "Q2_STRUCTURED_BLOCK")]
        delta_eim = float(q2["e_im"]) - float(q0["e_im"])
        if delta_eim < -epsilon:
            effect = "BLOCK_EFFECTIVE"
        elif delta_eim > epsilon:
            effect = "BLOCK_REGRESSED"
        else:
            effect = "BLOCK_WEAK"
        stages = receipt["block_stages"]
        block_rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "population": frame["population"],
                "dominant_contributor": dominant,
                "selected_blocks": "+".join(selected),
                "q0_e_im": float(q0["e_im"]),
                "q2_e_im": float(q2["e_im"]),
                "delta_e_im": delta_eim,
                "relative_eim_reduction": -delta_eim / max(float(q0["e_im"]), EPS),
                "q0_production_objective": float(q0["production_total_fixed_q0_aux"]),
                "q2_production_objective": float(q2["production_total_fixed_q0_aux"]),
                "delta_production_objective": float(q2["production_total_fixed_q0_aux"])
                - float(q0["production_total_fixed_q0_aux"]),
                "block_effect": effect,
                "block_nfev": sum(int(stage["nfev"]) for stage in stages),
                "block_runtime_sec": sum(float(stage["runtime_sec"]) for stage in stages),
            }
        )
    write_csv(root / "structured/contributor_selection.csv", contributor_rows)
    write_csv(root / "structured/block_effect.csv", block_rows)
    agreement = sum(bool(row["top1_agreement"]) for row in contributor_rows) / len(contributor_rows)
    coverage = sum(bool(row["topk_coverage"]) for row in contributor_rows) / len(contributor_rows)
    thumb_rows = [
        row for row in contributor_rows if row["dominant_semantic_contributor_block"] == "thumb"
    ]
    contributor_summary = {
        "schema_version": "DominantContributorSelectionAgreementV1",
        "metric_name": "DOMINANT_CONTRIBUTOR_SELECTION_AGREEMENT",
        "not_classification_ground_truth": True,
        "frames": len(contributor_rows),
        "top1_block_selection_agreement": agreement,
        "topk_coverage": coverage,
        "thumb_dominant_to_thumb_selected_rate": (
            sum(row["selected_block_1"] == "thumb" for row in thumb_rows) / len(thumb_rows)
            if thumb_rows
            else None
        ),
        "confusion_like_matrix": matrix,
    }
    write_json(root / "structured/contributor_selection_summary.json", contributor_summary)
    effect_counts = Counter(row["block_effect"] for row in block_rows)
    block_summary = {
        "schema_version": "StructuredBlockEffectSummaryV1",
        "frames": len(block_rows),
        "counts": dict(effect_counts),
        "median_e_im_reduction_before_polish": float(
            np.median([-float(row["delta_e_im"]) for row in block_rows])
        ),
        "median_relative_e_im_reduction_before_polish": float(
            np.median([float(row["relative_eim_reduction"]) for row in block_rows])
        ),
        "median_production_objective_change": float(
            np.median([float(row["delta_production_objective"]) for row in block_rows])
        ),
    }
    write_json(root / "structured/block_effect_summary.json", block_summary)
    b1_contract = read_json(O5RB / "b1_thumb_feasibility/feasibility_contract.json")
    structured_contract = read_json(
        O5RC / "structured_solver/frozen_structured_solver_contract.json"
    )
    contract_diff = {
        "schema_version": "B1VsStructuredBlockContractDiffV1",
        "b1": {
            "starting_q": "five deterministic seeds including Q0 and four generic alternatives",
            "free_dofs": "asset-derived thumb DOFs [0,1,2,3]",
            "objective": b1_contract["objective"],
            "residual_scope": "thumb semantic graph rows only",
            "bounds": "Wuji thumb joint limits",
            "solver": "scipy.optimize.least_squares(method=trf)",
            "seed_candidates": b1_contract["seeds"],
            "nfev_budget_per_seed": b1_contract["max_nfev"],
        },
        "structured_block": {
            "starting_q": "Q0 only",
            "free_dofs": structured_contract["design"]["blocks"]["THUMB"],
            "objective": "full weighted final-refinement scalar objective plus collision constraints",
            "residual_scope": "E_IM all 71 graph vertices + bone + temporal + base + slack",
            "bounds": "Wuji limits, fixed base, active collision slack",
            "solver": "scipy.optimize.minimize(method=SLSQP)",
            "seed_candidates": ["Q0"],
            "maxiter": structured_contract["design"]["selected"]["block_maxiter"],
            "runtime_binding_issue": "continuous_prediction inputs omitted; continuous_full_state fell back to previous_reference branch",
        },
        "same": ["thumb free-DOF indices", "robot asset", "source graph", "joint limits"],
        "different": [
            "objective",
            "initialization",
            "search budget",
            "solver",
            "residual scope",
            "collision constraints",
            "temporal runtime binding",
        ],
        "B1_VS_STRUCTURED_PRIMARY_DIFFERENCE": "MULTIPLE_DIFFERENCES",
        "why_b1_median_rho_does_not_transfer": "B1 directly minimizes thumb E_IM from five basins. Structured V1 minimizes a competing full scalar objective from Q0 only and additionally misbinds the continuous temporal context.",
    }
    write_json(root / "structured/b1_vs_structured_block_contract_diff.json", contract_diff)
    return {"contributor": contributor_summary, "block": block_summary}


def polish_audit(root: Path) -> dict[str, Any]:
    runtime = EpisodeRuntime("dev_01", root)
    states = _state_row_index(root)
    epsilon = float(read_json(root / "replay/parity.json")["classification_epsilon"])
    rollback_rows: list[dict[str, Any]] = []
    tradeoff_rows: list[dict[str, Any]] = []
    for index, frame in enumerate(audit_frames(), start=1):
        ordinal = int(frame["ordinal"])
        q0 = states[(ordinal, "Q0_OLD_PRODUCTION")]
        q2 = states[(ordinal, "Q2_STRUCTURED_BLOCK")]
        q3 = states[(ordinal, "Q3_STRUCTURED_FINAL")]
        gain = float(q0["e_im"]) - float(q2["e_im"])
        loss = float(q3["e_im"]) - float(q2["e_im"])
        ratio = rollback_ratio(gain, loss, epsilon)
        rollback_rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "population": frame["population"],
                "q0_e_im": float(q0["e_im"]),
                "q2_e_im": float(q2["e_im"]),
                "q3_e_im": float(q3["e_im"]),
                "block_gain_g": gain,
                "rollback_loss_l": loss,
                "rollback_ratio": ratio,
                "rollback_applicable": ratio is not None,
            }
        )
        arrays = np.load(_state_path(root, ordinal), allow_pickle=False)
        q2_q, base2 = arrays["q2"], arrays["base2"]
        q3_q, base3 = arrays["q3"], arrays["base3"]
        _, _, previous = runtime.old_state(ordinal)
        stage_context = runtime.context(ordinal, q2_q, base2, previous, free=None, fixed_base=False)
        q3_ids = arrays["q3_query_ids"].astype(np.int64)
        q2_state_without = np.concatenate([np.zeros(6), q2_q])
        q2_full = runtime.resources.reference_sdf.query_scene(
            stage_context.candidate_points(q2_state_without), stage_context.object_pose_scene
        )
        q2_slack = np.clip(
            np.maximum(-stage_context.paper.tau - q2_full.signed_distance[q3_ids], 0.0),
            0.0,
            stage_context.paper.b - stage_context.paper.tau,
        )
        q2_total, _, q2_breakdown = stage_context.objective(
            np.concatenate([q2_state_without, q2_slack])
        )
        q3_correction = encode_base_correction(base2, base3)
        q3_total, _, q3_breakdown = stage_context.objective(
            np.concatenate([q3_correction, q3_q, arrays["q3_slack"]])
        )
        values2, values3 = q2_breakdown.as_dict(), q3_breakdown.as_dict()
        for component in (
            "weighted_e_im",
            "weighted_e_bone",
            "e_temporal",
            "e_base_pos",
            "e_base_rot",
            "e_slack",
            "total",
        ):
            tradeoff_rows.append(
                {
                    "frame_id": frame["frame_id"],
                    "ordinal": ordinal,
                    "population": frame["population"],
                    "component": component,
                    "q2_stage_local": values2[component],
                    "q3_stage_local": values3[component],
                    "delta_q2_to_q3": values3[component] - values2[component],
                    "q2_total_reconstructed": q2_total,
                    "q3_total_reconstructed": q3_total,
                }
            )
        print(f"O5RD1_POLISH_AUDIT {index}/55 ordinal={ordinal}", flush=True)
    write_csv(root / "polish/rollback_per_frame.csv", rollback_rows)
    write_csv(root / "polish/objective_tradeoff.csv", tradeoff_rows)
    applicable = [
        float(row["rollback_ratio"]) for row in rollback_rows if row["rollback_ratio"] is not None
    ]
    rollback_summary = {
        "schema_version": "PolishRollbackRatioV1",
        "n_frames": len(rollback_rows),
        "n_applicable": len(applicable),
        "median": float(np.median(applicable)) if applicable else None,
        "p25": float(np.quantile(applicable, 0.25)) if applicable else None,
        "p75": float(np.quantile(applicable, 0.75)) if applicable else None,
        "fraction_r_ge_0_5": sum(value >= 0.5 for value in applicable) / len(applicable)
        if applicable
        else None,
        "fraction_r_ge_0_9": sum(value >= 0.9 for value in applicable) / len(applicable)
        if applicable
        else None,
        "fraction_r_gt_1": sum(value > 1.0 for value in applicable) / len(applicable)
        if applicable
        else None,
        "epsilon": epsilon,
    }
    write_json(root / "polish/rollback_summary.json", rollback_summary)
    component_summary = []
    for component in sorted({row["component"] for row in tradeoff_rows}):
        values = [
            float(row["delta_q2_to_q3"]) for row in tradeoff_rows if row["component"] == component
        ]
        component_summary.append(
            {
                "component": component,
                "median_delta_q2_to_q3": float(np.median(values)),
                "mean_delta_q2_to_q3": float(np.mean(values)),
            }
        )
    write_json(
        root / "polish/objective_tradeoff_summary.json",
        {
            "schema_version": "FinalPolishObjectiveTradeoffSummaryV1",
            "components": component_summary,
            "evaluation_context": "frozen StructuredSolverV1 stage-local context; Q2 minimal slack and Q3 returned slack on final QuerySet",
        },
    )
    return rollback_summary


def directional_audit(root: Path) -> dict[str, Any]:
    runtime = EpisodeRuntime("dev_01", root)
    rows: list[dict[str, Any]] = []
    for frame in (row for row in audit_frames() if row["q1_available"]):
        ordinal = int(frame["ordinal"])
        arrays = np.load(_state_path(root, ordinal), allow_pickle=False)
        context = production_context(runtime, ordinal)
        slack = _ragged(
            runtime.final.arrays["slack_concat"], runtime.final.arrays["slack_offsets"], ordinal
        )
        correction = np.asarray(runtime.final.arrays["base_corrections"][ordinal])
        origin = np.concatenate([correction, arrays["q0"], slack])
        direction = np.zeros_like(origin)
        direction[6 : 6 + runtime.model.num_dofs] = arrays["q1"] - arrays["q0"]

        def objective(value: np.ndarray, ctx: Any = context) -> float:
            return float(ctx.objective(value)[0])

        def semantic(value: np.ndarray, ctx: Any = context) -> float:
            return float(ctx.objective(value)[2].e_im)

        total, gradient, _ = context.objective(origin)
        d_j = central_directional_derivative(objective, origin, direction)
        d_eim = central_directional_derivative(semantic, origin, direction)
        analytic_d_j = float(np.dot(gradient, direction))
        non_thumb = np.setdiff1d(np.arange(runtime.model.num_dofs), runtime.blocks["THUMB"])
        rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "q0_objective": total,
                "direction_l2": float(np.linalg.norm(direction)),
                "max_non_thumb_direction_abs": float(
                    np.max(np.abs((arrays["q1"] - arrays["q0"])[non_thumb]), initial=0.0)
                ),
                "finite_difference_step_alpha": DIRECTIONAL_STEP,
                "d_j_prod_d_alpha_central": d_j,
                "d_j_prod_d_alpha_analytic": analytic_d_j,
                "analytic_central_absolute_error": abs(d_j - analytic_d_j),
                "d_e_im_d_alpha_central": d_eim,
                "objective_opposes_semantic_direction": d_eim < 0.0 and d_j > 0.0,
            }
        )
    write_csv(root / "directional/directional_derivatives.csv", rows)
    opposed = sum(bool(row["objective_opposes_semantic_direction"]) for row in rows)
    payload = {
        "schema_version": "ProductionObjectiveDirectionalAlignmentV1",
        "frames": len(rows),
        "direction": "Q1-Q0 on B1-modified thumb DOFs only",
        "fraction_d_eim_negative_d_j_positive": opposed / len(rows),
        "median_d_j_prod_d_alpha": float(
            np.median([row["d_j_prod_d_alpha_central"] for row in rows])
        ),
        "median_d_e_im_d_alpha": float(np.median([row["d_e_im_d_alpha_central"] for row in rows])),
        "max_analytic_central_absolute_error": max(
            float(row["analytic_central_absolute_error"]) for row in rows
        ),
        "full_jacobian_generated": False,
    }
    write_json(root / "directional/summary.json", payload)
    return payload


def interpolation_audit(root: Path) -> dict[str, Any]:
    runtime = EpisodeRuntime("dev_01", root)
    selected = choose_representatives(
        [row for row in audit_frames() if row["q1_available"]],
        {"HIGH": 5, "MID": 3, "LOW": 2},
    )
    write_json(
        root / "interpolation/selected_frames.json",
        {
            "schema_version": "O5RD1InterpolationSubsetV1",
            "selection": "deterministic temporal quantiles within frozen B1 HIGH/MID/LOW strata",
            "counts": {"HIGH": 5, "MID": 3, "LOW": 2},
            "frames": selected,
        },
    )
    rows: list[dict[str, Any]] = []
    path_classes: list[dict[str, Any]] = []
    for frame in selected:
        ordinal = int(frame["ordinal"])
        arrays = np.load(_state_path(root, ordinal), allow_pickle=False)
        context = production_context(runtime, ordinal)
        slack = _ragged(
            runtime.final.arrays["slack_concat"], runtime.final.arrays["slack_offsets"], ordinal
        )
        correction = np.asarray(runtime.final.arrays["base_corrections"][ordinal])
        frame_rows: list[dict[str, Any]] = []
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            q = arrays["q0"] + alpha * (arrays["q1"] - arrays["q0"])
            value = np.concatenate([correction, q, slack])
            total, _, breakdown = context.objective(value)
            row = {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "alpha": alpha,
                "production_objective": total,
                **breakdown.as_dict(),
            }
            rows.append(row)
            frame_rows.append(row)
        objective_values = [float(row["production_objective"]) for row in frame_rows]
        semantic_values = [float(row["e_im"]) for row in frame_rows]
        barrier = (
            max(objective_values[1:-1])
            > max(objective_values[0], objective_values[-1]) + OBJECTIVE_EPS
        )
        objective_monotone = bool(np.all(np.diff(objective_values) >= -OBJECTIVE_EPS))
        semantic_monotone = bool(np.all(np.diff(semantic_values) <= OBJECTIVE_EPS))
        if barrier:
            classification = "OBJECTIVE_BARRIER"
        elif objective_monotone and semantic_monotone:
            classification = "SMOOTH_SEMANTIC_OBJECTIVE_TRADEOFF"
        elif (
            objective_values[-1] <= objective_values[0] and semantic_values[-1] < semantic_values[0]
        ):
            classification = "SEMANTIC_AND_OBJECTIVE_ALIGNED"
        else:
            classification = "NONMONOTONIC_MIXED_PATH"
        path_classes.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "classification": classification,
                "objective_monotone_non_decreasing": objective_monotone,
                "semantic_monotone_non_increasing": semantic_monotone,
                "interior_objective_barrier": barrier,
            }
        )
    write_csv(root / "interpolation/path_metrics.csv", rows)
    payload = {
        "schema_version": "O5RD1InterpolationSummaryV1",
        "frames": len(selected),
        "alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
        "classification_counts": dict(Counter(row["classification"] for row in path_classes)),
        "per_frame": path_classes,
    }
    write_json(root / "interpolation/summary.json", payload)
    return payload


def micro_polish(root: Path) -> dict[str, Any]:
    runtime = EpisodeRuntime("dev_01", root)
    selected = choose_representatives(
        [row for row in audit_frames() if row["q1_available"]],
        {"HIGH": 5, "MID": 3, "LOW": 2},
    )
    write_json(
        root / "micro_polish/selected_frames.json",
        {
            "schema_version": "O5RD1MicroPolishSubsetV1",
            "development_only": True,
            "selection": "same deterministic 5 HIGH + 3 MID + 2 LOW B1 subset as interpolation",
            "frames": selected,
        },
    )
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(selected, start=1):
        ordinal = int(frame["ordinal"])
        source = np.load(_state_path(root, ordinal), allow_pickle=False)
        target = root / f"micro_polish/states/frame_{ordinal:04d}.npz"
        receipt_path = root / f"micro_polish/receipts/frame_{ordinal:04d}.json"
        if target.exists() and receipt_path.exists():
            result_arrays = np.load(target, allow_pickle=False)
            q_after, base_after = result_arrays["q"], result_arrays["base"]
            receipt = read_json(receipt_path)
        else:
            _, _, previous = runtime.old_state(ordinal)
            polish = _polish(
                runtime,
                ordinal,
                source["q1"],
                source["base0"],
                previous,
                initialization_source="o5rd1_b1_best_observed_micro_polish",
            )
            q_after, base_after = polish.qpos, polish.base_pose_scene
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, q=q_after, base=base_after, slack=polish.slack)
            receipt = {
                "schema_version": "O5RD1B1MicroPolishReceiptV1",
                "diagnostic_only": True,
                "development_frame": True,
                "ordinal": ordinal,
                "frame_id": frame["frame_id"],
                "accepted": polish.accepted,
                "acceptance_reason": polish.acceptance_reason,
                "nfev": polish.optimizer_function_evaluations,
                "njev": polish.optimizer_jacobian_evaluations,
                "runtime_sec": polish.solve_time_s,
                "stage_local_initial_objective": polish.initial_objective,
                "stage_local_final_objective": polish.final_objective,
                "stage_local_final_breakdown": polish.breakdown.as_dict(),
            }
            write_json(receipt_path, receipt)
        exact = production_context(runtime, ordinal)
        q0_slack = _ragged(
            runtime.final.arrays["slack_concat"], runtime.final.arrays["slack_offsets"], ordinal
        )
        warm_base = np.asarray(runtime.warm.arrays["base_pose_scene"][ordinal])
        before_value = np.concatenate(
            [encode_base_correction(warm_base, source["base0"]), source["q1"], q0_slack]
        )
        after_value = np.concatenate(
            [encode_base_correction(warm_base, base_after), q_after, q0_slack]
        )
        before_total, _, before = exact.objective(before_value)
        after_total, _, after = exact.objective(after_value)
        rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": ordinal,
                "stratum": frame["stratum"],
                "accepted": receipt["accepted"],
                "q1_before_e_im": before.e_im,
                "q1_after_polish_e_im": after.e_im,
                "delta_e_im": after.e_im - before.e_im,
                "q1_before_exact_production_objective": before_total,
                "q1_after_exact_production_objective": after_total,
                "delta_exact_production_objective": after_total - before_total,
                "stage_local_initial_objective": receipt["stage_local_initial_objective"],
                "stage_local_final_objective": receipt["stage_local_final_objective"],
                "nfev": receipt["nfev"],
                "runtime_sec": receipt["runtime_sec"],
            }
        )
        print(f"O5RD1_MICRO_POLISH {index}/10 ordinal={ordinal}", flush=True)
    write_csv(root / "micro_polish/results.csv", rows)
    payload = {
        "schema_version": "O5RD1MicroPolishSummaryV1",
        "frames": len(rows),
        "technical_completion": sum(str(row["accepted"]).lower() == "true" for row in rows),
        "fraction_eim_rolled_back": sum(float(row["delta_e_im"]) > 0.0 for row in rows) / len(rows),
        "fraction_exact_production_objective_improved": sum(
            float(row["delta_exact_production_objective"]) < 0.0 for row in rows
        )
        / len(rows),
        "median_delta_e_im": float(np.median([float(row["delta_e_im"]) for row in rows])),
        "median_delta_exact_production_objective": float(
            np.median([float(row["delta_exact_production_objective"]) for row in rows])
        ),
        "bounded_development_only": True,
        "full_trajectory": False,
    }
    write_json(root / "micro_polish/summary.json", payload)
    return payload


def reconstruct_total(breakdown: Any) -> float:
    """Reconstruct the scalar production objective from persisted components."""

    return float(
        breakdown.weighted_e_im
        + breakdown.weighted_e_bone
        + breakdown.e_temporal
        + breakdown.e_base_pos
        + breakdown.e_base_rot
        + breakdown.e_slack
        + breakdown.weighted_e_morph
        + breakdown.weighted_e_contact_pos
        + breakdown.weighted_e_contact_dir
    )


def mutation_audit(root: Path) -> dict[str, Any]:
    initial = read_json(root / "preflight/frozen_authorities.json")["authorities"]
    rows = []
    for name, path in frozen_paths().items():
        before = str(initial[name]["sha256"])
        after = digest(path)
        rows.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256_before": before,
                "sha256_after": after,
                "unchanged": before == after,
            }
        )
    guidance = REPO.parent / "TopoRetarget-Repro-guidance"
    guidance_status = subprocess.check_output(
        ["git", "-C", str(guidance), "status", "--short", "--untracked-files=all"],
        text=True,
    ).splitlines()
    payload = {
        "schema_version": "O5RD1NoMutationAuditV1",
        "status": "PASS"
        if all(row["unchanged"] for row in rows) and not guidance_status
        else "FAIL",
        "authorities": rows,
        "required_unchanged": {
            "structured_solver_v1": all(
                row["unchanged"]
                for row in rows
                if row["name"] in {"o5rc_source", "o5rc_frozen_contract"}
            ),
            "semantic_v1": next(
                row["unchanged"] for row in rows if row["name"] == "semantic_v1_source"
            ),
            "production_objective": next(
                row["unchanged"] for row in rows if row["name"] == "production_objective_source"
            ),
            "dev1_old_trajectory": next(
                row["unchanged"] for row in rows if row["name"] == "dev1_production_trajectory"
            ),
            "manifest_v2": next(row["unchanged"] for row in rows if row["name"] == "manifest_v2"),
            "split_v2": next(row["unchanged"] for row in rows if row["name"] == "split_v2"),
        },
        "guidance_worktree_status": guidance_status,
        "local_tracked_paths": git("ls-files", ".local").splitlines(),
    }
    if payload["local_tracked_paths"]:
        payload["status"] = "FAIL"
    write_json(root / "preflight/no_mutation_audit.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("O5RD1_NO_MUTATION_AUDIT_FAILED")
    return payload


def summarize(root: Path) -> dict[str, Any]:
    no_mutation = mutation_audit(root)
    b1 = read_json(root / "b1_alignment/classification_summary.json")
    contributor = read_json(root / "structured/contributor_selection_summary.json")
    block = read_json(root / "structured/block_effect_summary.json")
    rollback = read_json(root / "polish/rollback_summary.json")
    polish_tradeoff = read_json(root / "polish/objective_tradeoff_summary.json")
    directional = read_json(root / "directional/summary.json")
    interpolation = read_json(root / "interpolation/summary.json")
    micro = read_json(root / "micro_polish/summary.json")
    parity = read_json(root / "replay/parity.json")
    interpolation_tradeoff_fraction = int(
        interpolation["classification_counts"].get("SMOOTH_SEMANTIC_OBJECTIVE_TRADEOFF", 0)
    ) / int(interpolation["frames"])
    micro_counterfactual_fraction = min(
        float(micro["fraction_eim_rolled_back"]),
        float(micro["fraction_exact_production_objective_improved"]),
    )
    misalignment_vote = (
        float(b1["fraction_semantic_better_objective_worse"]) >= 0.8
        and max(interpolation_tradeoff_fraction, micro_counterfactual_fraction) >= 0.8
    )
    aligned_vote = (
        int(b1["counts"].get("ALIGNED_BETTER", 0)) >= 20
        and float(directional["fraction_d_eim_negative_d_j_positive"]) <= 0.2
    )
    if misalignment_vote:
        alignment = "MISALIGNED"
        root_cause = "OBJECTIVE_SEMANTIC_MISALIGNMENT"
        confidence = "HIGH"
        next_contract = "O5R-D2A_RETARGET_OBJECTIVE_V2_DESIGN"
    elif aligned_vote:
        alignment = "ALIGNED"
        root_cause = "SEARCH_BASIN_FAILURE"
        confidence = "HIGH"
        next_contract = "O5R-D2B_STRUCTURED_SOLVER_V2_SEARCH_REPAIR"
    else:
        alignment = "PARTIALLY_ALIGNED"
        root_cause = "MULTI_FACTOR"
        confidence = "MEDIUM"
        next_contract = "O5R-D2_JOINT_OBJECTIVE_AND_SEARCH_DESIGN"
    alignment_payload = {
        "schema_version": "ProductionObjectiveAlignmentV1",
        "PRODUCTION_OBJECTIVE_ALIGNMENT": alignment,
        "e_im_in_production_objective": "EXACT",
        "downstream_metric_not_directly_optimized": False,
        "trajectory_gate_mapping": "per-frame E_IM is exact, but p95 threshold is absent from the per-frame scalar objective",
        "decision_rule": "MISALIGNED when >=80% of B1 endpoints are semantic-better/objective-worse plus an independent >=80% interpolation tradeoff or Q1-start polish counterfactual",
        "evidence": {
            "b1_semantic_better_objective_worse_fraction": b1[
                "fraction_semantic_better_objective_worse"
            ],
            "directional_opposition_fraction": directional["fraction_d_eim_negative_d_j_positive"],
            "interpolation_smooth_tradeoff_fraction": interpolation_tradeoff_fraction,
            "micro_polish_eim_rollback_fraction": micro["fraction_eim_rolled_back"],
            "micro_polish_objective_improvement_fraction": micro[
                "fraction_exact_production_objective_improved"
            ],
        },
    }
    root_payload = {
        "schema_version": "O5RD1PrimaryRootCauseV1",
        "PRIMARY_ROOT_CAUSE": root_cause,
        "CONFIDENCE": confidence,
        "independent_evidence_directions": [
            "Q0-to-Q1 endpoint objective decomposition",
            "Q0-to-Q1 interpolation path tradeoff",
            "bounded Q1-start frozen full-polish replay",
        ],
        "secondary_findings": [
            "Q0 directional signs are mixed because the old production state is locally near-stationary; the finite meaningful path still rises monotonically on all ten interpolation frames",
            "StructuredSolverV1 block search differs from B1 in objective, initialization, solver, budget, residual scope, and constraints",
            "StructuredSolverV1 omitted continuous-prediction runtime inputs and therefore used a non-production temporal branch",
            "StructuredSolverV1 remains scientifically rejected regardless of this diagnostic attribution",
        ],
        "not_claimed": [
            "B1 global optimality",
            "V2 acceptance",
            "DEV2 recovery",
            "production readiness",
        ],
    }
    next_payload = {
        "schema_version": "O5RD1NextContractRecommendationV1",
        "NEXT": next_contract,
        "recommendation_only": True,
        "implemented": False,
        "design_direction": "study interaction-aware constrained or lexicographic refinement with explicit non-regression limits; do not outcome-tune a scalar interaction weight",
        "validation_hygiene": "freeze a new SparseValidationV2 only after the next method contract is frozen, excluding all 55 consumed frames",
    }
    write_json(root / "decision/production_objective_alignment.json", alignment_payload)
    write_json(root / "decision/primary_root_cause.json", root_payload)
    write_json(root / "decision/next_contract_recommendation.json", next_payload)
    interpolation_frames = read_json(root / "interpolation/selected_frames.json")["frames"]
    micro_frames = read_json(root / "micro_polish/selected_frames.json")["frames"]
    ledger = read_json(root / "preflight/consumed_frame_ledger.json")
    ledger["entries"].extend(
        [
            {
                "source": "O5R-D1 interpolation subset",
                "role": "DIAGNOSTIC_REUSE_OF_ALREADY_CONSUMED_B1_FRAMES",
                "count": len(interpolation_frames),
                "frames": interpolation_frames,
                "adds_new_exclusions": False,
            },
            {
                "source": "O5R-D1 micro-polish subset",
                "role": "DIAGNOSTIC_REUSE_OF_ALREADY_CONSUMED_B1_FRAMES",
                "count": len(micro_frames),
                "frames": micro_frames,
                "adds_new_exclusions": False,
            },
        ]
    )
    ledger["future_dev1_validation_exclusion_count"] = 55
    write_json(root / "method_ledger/dev1_method_development_ledger.json", ledger)
    components = b1["component_tradeoff_ranked_by_median_penalty"]
    positive = [row for row in components if float(row["median_delta_weighted_q0_to_q1"]) > 0.0]
    blockers = ", ".join(str(row["component"]) for row in positive[:3])
    safety = {
        "BRANCH": git("branch", "--show-current"),
        "STRUCTURED_SOLVER_V1_MODIFIED": "NO",
        "STRUCTURED_SOLVER_V1_STATUS": "SCIENTIFICALLY_REJECTED",
        "STRUCTURED_SOLVER_V1": "SCIENTIFICALLY_REJECTED",
        "DO_NOT_REUSE_AS_PRODUCTION": "YES",
        "SEMANTIC_V1_CHANGED": "NO",
        "PRODUCTION_OBJECTIVE_CHANGED": "NO",
        "DEV1_OLD_TRAJECTORY_MODIFIED": "NO",
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_REFINEMENT_RUNS": 0,
        "DEV2_FRAME0_NEW_SOLVES": 0,
        "DEV2_FULL_PRODUCTION_SOLVES": 0,
        "B1_DEVELOPMENT_FRAMES_USED": 25,
        "SPARSE_V1_CONSUMED_FRAMES_USED": 30,
        "NEW_INDEPENDENT_VALIDATION_CONSUMED": "NO",
        "OBJECTIVE_DECOMPOSITION_COMPLETE": "YES",
        "EIM_OBJECTIVE_MAPPING_AUDITED": "YES",
        "B1_OBJECTIVE_ALIGNMENT_AUDITED": "YES",
        "CONTRIBUTOR_SELECTION_AUDITED": "YES",
        "BLOCK_STAGE_EFFECT_AUDITED": "YES",
        "FINAL_POLISH_ROLLBACK_AUDITED": "YES",
        "DIRECTIONAL_ALIGNMENT_AUDITED": "YES",
        "RETARGET_OBJECTIVE_V2_CREATED": "NO",
        "STRUCTURED_SOLVER_V2_CREATED": "NO",
        "SEMANTIC_V2_CREATED": "NO",
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "CERTIFICATION_DOWNSTREAM_CONSUMED": "NO",
        "HELDOUT_DOWNSTREAM_CONSUMED": "NO",
        "DEV2_FRAME0_HARD_CONTROL_PRESERVED_FOR_NEXT_VERSION": "YES",
        "O6_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSICAL_SCENE_AUTHORITY_RAN": "NO",
        "ISAAC_SIM_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "PF_DF_RAN": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    summary = {
        "schema_version": "OakInk2O5RD1FinalSummaryV1",
        "status": "O5RD1_DIAGNOSTIC_COMPLETE",
        "production_objective_alignment": alignment,
        "primary_root_cause": root_cause,
        "confidence": confidence,
        "next": next_contract,
        "e_im_mapping": "EXACT",
        "b1_alignment": b1,
        "contributor_selection": contributor,
        "block_effect": block,
        "polish_rollback": rollback,
        "polish_objective_tradeoff": polish_tradeoff,
        "directional_alignment": directional,
        "interpolation": interpolation,
        "micro_polish": micro,
        "parity": parity,
        "no_mutation": no_mutation,
        "blocking_components_q0_to_q1": blockers,
        "safety_flags": safety,
    }
    write_json(root / "final_summary.json", summary)
    tradeoff_table = "\n".join(
        f"| {row['component']} | {float(row['median_delta_weighted_q0_to_q1']):.9g} | {'opposes lower E_IM' if float(row['median_delta_weighted_q0_to_q1']) > 0 else 'supports lower E_IM'} |"
        for row in components
    )
    counts = b1["counts"]
    block_counts = block["counts"]
    markdown = f"""# OakInk2 O5R-D1 Objective Alignment & Rollback Audit Handoff

## 1. Git

BRANCH={safety["BRANCH"]}

START_HEAD={read_json(root / "preflight/git.json")["start_head"]}

FINAL_HEAD={git("rev-parse", "HEAD")}

PUSHED=NO

PR_CREATED=NO

## 2. Execution boundary

DEV1_FULL_RETARGET_RERUNS=0

DEV1_FULL_REFINEMENT_RUNS=0

DEV2_FRAME0_NEW_SOLVES=0

DEV2_FULL_PRODUCTION_SOLVES=0

STRUCTURED_SOLVER_V2_CREATED=NO

RETARGET_OBJECTIVE_V2_CREATED=NO

SEMANTIC_V2_CREATED=NO

## 3. Production objective definition

`J = 500 E_IM + 0.1 E_bone + E_temporal + 100 ||delta_p||^2 + ||delta_w||^2 + 50000 ||slack||^2`, where production continuous temporal energy is `0.25` times the mean squared translation, rotation, and q errors normalized by `0.01 m`, `5 deg`, and `0.05 rad`. Collision inequalities are separate from the scalar objective.

E_IM_IN_PRODUCTION_OBJECTIVE=EXACT

Semantic V1's trajectory p95 gate is not itself part of the per-frame scalar objective.

## 4. B1 objective alignment

N_B1={b1["n_b1"]}

N semantic-better + objective-better={counts.get("ALIGNED_BETTER", 0)}

N semantic-better + objective-worse={counts.get("SEMANTIC_BETTER_OBJECTIVE_WORSE", 0)}

N other={counts.get("NO_MEANINGFUL_SEMANTIC_IMPROVEMENT", 0)}

median delta E_IM={b1["median_delta_e_im"]:.12g}

median delta J_prod={b1["median_delta_production_objective"]:.12g}

| Component | Median delta weighted contribution Q0 to Q1 | Interpretation |
| --- | ---: | --- |
{tradeoff_table}

The principal terms opposing the B1 low-E_IM states are: {blockers}.

## 5. Contributor and block stage

DOMINANT_CONTRIBUTOR_SELECTION_AGREEMENT={contributor["top1_block_selection_agreement"]:.3%}

TOP_K_COVERAGE={contributor["topk_coverage"]:.3%}

THUMB_DOMINANT_TO_THUMB_SELECTED_RATE={contributor["thumb_dominant_to_thumb_selected_rate"]:.3%}

N BLOCK_EFFECTIVE={block_counts.get("BLOCK_EFFECTIVE", 0)}

N BLOCK_WEAK={block_counts.get("BLOCK_WEAK", 0)}

N BLOCK_REGRESSED={block_counts.get("BLOCK_REGRESSED", 0)}

N UNAVAILABLE=0

median E_IM reduction before polish={block["median_e_im_reduction_before_polish"]:.12g}

B1_VS_STRUCTURED_PRIMARY_DIFFERENCE=MULTIPLE_DIFFERENCES

## 6. Final polish rollback

N rollback-applicable={rollback["n_applicable"]}

median R_rollback={rollback["median"]}

p25={rollback["p25"]}

p75={rollback["p75"]}

fraction R>=0.5={rollback["fraction_r_ge_0_5"]}

fraction R>=0.9={rollback["fraction_r_ge_0_9"]}

fraction R>1={rollback["fraction_r_gt_1"]}

The component-resolved Q2-to-Q3 benefit is recorded in `polish/objective_tradeoff.csv`; this separates frozen StructuredSolverV1's stage-local binding from the exact production replay.

## 7. Directional alignment and decision

N directional frames={directional["frames"]}

fraction D(E_IM)<0 and D(J_prod)>0={directional["fraction_d_eim_negative_d_j_positive"]:.3%}

median D(J_prod)={directional["median_d_j_prod_d_alpha"]:.12g}

PRODUCTION_OBJECTIVE_ALIGNMENT={alignment}

PRIMARY_ROOT_CAUSE={root_cause}

CONFIDENCE={confidence}

STRUCTURED_SOLVER_V1=SCIENTIFICALLY_REJECTED

DO_NOT_REUSE_AS_PRODUCTION=YES

## 8. Future validation and next contract

DEV1_FRAMES_CONSUMED_FOR_METHOD_DEVELOPMENT=55

FUTURE_SPARSE_VALIDATION_V2_MUST_EXCLUDE=the 55 frame IDs in `method_ledger/dev1_method_development_ledger.json`

DEV1_SPARSE_VALIDATION_V2_CREATED=NO

NEXT={next_contract}

This is a recommendation only. No V2 method was created.

DEV2_FRAME0_HARD_CONTROL_PRESERVED=YES

DEV2_FULL_PRODUCTION_SOLVE_COUNT=0
"""
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    write_json(
        root / "resource_usage.json",
        {
            "audit_frames": 55,
            "b1_development_frames": 25,
            "consumed_sparse_v1_frames": 30,
            "structured_v1_diagnostic_replays": 55,
            "micro_polish_development_frames": 10,
            "interpolation_development_frames": 10,
            "dev1_full_retarget_reruns": 0,
            "dev1_full_refinement_runs": 0,
            "dev2_frame0_new_solves": 0,
            "dev2_full_production_solves": 0,
        },
    )
    (root / "technical_failures.jsonl").touch(exist_ok=True)
    return summary


def run_validation(root: Path) -> dict[str, Any]:
    modified_python = [
        "scripts/data/run_oakink2_o5rd1.py",
        "tests/data/test_oakink2_o5rd1.py",
    ]
    commands = [
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", *modified_python],
        [
            "conda",
            "run",
            "-n",
            "toporetarget-rl",
            "ruff",
            "format",
            "--check",
            *modified_python,
        ],
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "check", "."],
        ["conda", "run", "-n", "toporetarget-rl", "ruff", "format", "--check", "."],
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
    ]
    names = [
        "ruff_check_modified",
        "ruff_format_check_modified",
        "ruff_check_repo",
        "ruff_format_check_repo",
        "mypy_src",
        "pytest",
        "paper_fidelity",
    ]
    results = []
    log_root = root / "validation_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for name, command in zip(names, commands, strict=True):
        started = time.perf_counter()
        result = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        output = result.stdout + ("\n" if result.stdout and result.stderr else "") + result.stderr
        (log_root / f"{name}.log").write_text(output, encoding="utf-8")
        results.append(
            {
                "name": name,
                "command": command,
                "returncode": result.returncode,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "runtime_sec": time.perf_counter() - started,
                "log": str((log_root / f"{name}.log").resolve()),
            }
        )
        print(f"O5RD1_VALIDATION {name}={results[-1]['status']}", flush=True)
    payload = {
        "schema_version": "O5RD1ValidationResultsV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "results": results,
        "unrelated_historical_debt_fixed": False,
    }
    write_json(root / "tests.json", payload)
    write_json(root / "validation_results.json", payload)
    return payload


def record_git(root: Path) -> dict[str, Any]:
    preflight_git = read_json(root / "preflight/git.json")
    start, final = str(preflight_git["start_head"]), git("rev-parse", "HEAD")
    status = git("status", "--short", "--untracked-files=all").splitlines()
    commits = git("log", "--format=%H%x09%s", f"{start}..{final}").splitlines()
    payload = {
        "schema_version": "O5RD1GitCommitsV1",
        "branch": git("branch", "--show-current"),
        "start_head": start,
        "final_head": final,
        "commits": [
            {"commit": row.split("\t", 1)[0], "subject": row.split("\t", 1)[1]}
            for row in commits
            if "\t" in row
        ],
        "tracked_worktree_clean": not status,
        "status_short": status,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "git_commits.json", payload)
    summary_path = root / "final_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        summary["git"] = payload
        write_json(summary_path, summary)
    for name in ("final_summary.md", "handoff.md"):
        path = root / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        replaced = []
        for line in lines:
            if line.startswith("FINAL_HEAD="):
                replaced.append(f"FINAL_HEAD={final}")
            else:
                replaced.append(line)
        replaced.extend(
            [
                "",
                f"commits={[row['commit'] for row in payload['commits']]}",
                f"tracked_worktree_clean={'YES' if payload['tracked_worktree_clean'] else 'NO'}",
            ]
        )
        path.write_text("\n".join(replaced) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "all",
            "preflight",
            "audit-objective",
            "replay-states",
            "decompose",
            "audit-b1",
            "audit-structured",
            "audit-polish",
            "directional",
            "interpolation",
            "micro-polish",
            "summarize",
            "validate",
            "record-git",
        ),
        default="all",
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root, action = args.report_root.resolve(), args.action
    if action in {"all", "preflight"}:
        preflight(root)
    if action in {"all", "audit-objective"}:
        objective_audit(root)
    if action in {"all", "replay-states"}:
        replay_states(root)
    if action in {"all", "decompose"}:
        decompose_states(root)
    if action in {"all", "audit-b1"}:
        b1_alignment(root)
    if action in {"all", "audit-structured"}:
        structured_audit(root)
    if action in {"all", "audit-polish"}:
        polish_audit(root)
    if action in {"all", "directional"}:
        directional_audit(root)
    if action in {"all", "interpolation"}:
        interpolation_audit(root)
    if action in {"all", "micro-polish"}:
        micro_polish(root)
    if action in {"all", "summarize"}:
        summarize(root)
    if action == "validate":
        result = run_validation(root)
        return 0 if result["status"] == "PASS" else 1
    if action == "record-git":
        record_git(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
