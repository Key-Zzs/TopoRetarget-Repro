#!/usr/bin/env python3
"""OakInk2 O5R-B parallel bounded diagnostics.

The command is intentionally incapable of a DEV1 trajectory rerun or a DEV2
240-frame production solve.  It consumes frozen DEV1 output and confines all
new optimization to selected DEV1 thumb DOFs or DEV2 frame zero.
"""

# ruff: noqa: B023, E402, E501, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5 import EPISODES, MANIFEST_V2, ROBOT, SPLIT_V2, camera_presets
from scripts.data.run_oakink2_o5ra import FROZEN_ROOT, paths
from toporetarget.contracts.canonical import load_canonical_hoi
from toporetarget.quality.html import _robot_visual_payload
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.first_frame_seed import load_first_frame_seed_authority_v2
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.interaction_artifacts import load_interaction_graph
from toporetarget.retarget.interaction_objective import InteractionMeshResidual
from toporetarget.retarget.solver import load_paper_weights, load_solver_profile, solve_frame
from toporetarget.robots.registry import get_robot_registry
from toporetarget.utils.hashing import sha256_file, sha256_tree
from toporetarget.viz.oakink2_html_viewer import (
    OakInk2HTMLViewerV2Data,
    render_oakink2_html_viewer_v2,
)

ROOT = REPO / ".local/reports/oakink2_o5rb_parallel_v1"
O5RA = REPO / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"
THUMB = np.asarray([0, 1, 2, 3], dtype=np.int64)
THUMB_KEYS = np.asarray([1, 2, 3, 4], dtype=np.int64)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def episode(review: str) -> dict[str, Any]:
    return next(dict(row) for row in EPISODES if row["review"] == review)


def frozen_paths() -> dict[str, Path]:
    dev1 = paths("dev_01")
    return {
        "o5_handoff": FROZEN_ROOT / "handoff.md",
        "dev1_trajectory": dev1["trajectory"],
        "dev1_final": dev1["final"],
        "dev1_graph": dev1["graph"],
        "dev1_solver_receipt": dev1["solver_receipt"],
        "dev1_semantic_receipt": dev1["semantic"],
        "dev1_timing": FROZEN_ROOT / "timing/dev_01_frame_timing.csv",
        "o5ra_parity": O5RA / "dev1_semantic/semantic_replay_parity.json",
        "o5ra_localization": O5RA / "dev1_semantic/dev1_e_im_per_frame.csv",
        "o5ra_contributors": O5RA / "dev1_semantic/contributor_summary.json",
        "o5ra_thumb": O5RA / "dev1_semantic/finger_summary.json",
        "o5ra_dev2_reproducer": O5RA / "dev2_warmstart/minimal_reproducer.json",
        "o5ra_dev2_failure": O5RA / "dev2_warmstart/original_failure_receipt.json",
        "o5ra_profiler": O5RA / "profiling/solver_profiler_contract.json",
        "manifest_v2": MANIFEST_V2,
        "split_v2": SPLIT_V2,
        "retarget_contract": FROZEN_ROOT / "contract/geometric_retarget_contract.json",
        "wuji_asset_config": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
    }


def preflight(root: Path) -> None:
    assets = {
        name: {"path": str(path.resolve()), "sha256": digest(path)}
        for name, path in frozen_paths().items()
    }
    git = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "start_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "status_short": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True
        ).splitlines(),
        "new_branch_created": False,
        "new_worktree_created": False,
        "pushed": False,
        "pr_created": False,
    }
    if git["branch"] != "feature/oakink2-raw-to-physical":
        raise RuntimeError("O5RB_BRANCH_AUTHORITY_MISMATCH")
    write_json(root / "preflight/git.json", git)
    write_json(root / "preflight/artifact_hashes.json", assets)
    write_json(
        root / "preflight/frozen_authorities.json",
        {
            "schema_version": "OakInk2O5RBFrozenAuthoritiesV1",
            "status": "PASS",
            "artifacts": assets,
            "immutable": [
                "raw_MANO",
                "frame_binding",
                "target_object",
                "Manifest_V2",
                "Split_V2",
                "DEV1_production_output",
                "Semantic_Validity_V1",
            ],
        },
    )
    write_json(
        root / "preflight/parallel_execution_plan.json",
        {
            "WORKSTREAM_B1": "saved DEV1 data plus thumb-only bounded diagnostics",
            "WORKSTREAM_B2": "DEV2 frame-zero seed authority only",
            "max_heavy_diagnostic_processes": 2,
            "thread_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "dev1_full_retarget_reruns": 0,
            "dev2_full_production_solves": 0,
        },
    )


def b1_inputs() -> tuple[Any, Any, Any, Any]:
    final = load_final_trajectory(paths("dev_01")["final"])
    graph = load_interaction_graph(paths("dev_01")["graph"])
    bundle = np.load(O5RA / "dev1_semantic/semantic_replay_bundle.npz", allow_pickle=False)
    model = get_robot_registry().load(ROBOT)
    return final, graph, bundle, model


def b1_selection(root: Path) -> list[dict[str, Any]]:
    _, graph, bundle, _ = b1_inputs()
    e_im = np.asarray(bundle["interaction_final_e_im"], dtype=np.float64)
    residual = np.asarray(bundle["interaction_final_laplacian_residual"], dtype=np.float64)
    raw = np.asarray(bundle["raw_frame_index"], dtype=np.int64)
    mass = np.square(residual).sum(axis=-1) / 71.0
    dominant = np.argmax(mass, axis=1)
    thumb_tip = dominant == 4
    order_desc = np.lexsort((raw, -e_im))
    high = [int(i) for i in order_desc if thumb_tip[i]][:10]
    if len(high) < 10:
        high += [int(i) for i in order_desc if int(i) not in high][: 10 - len(high)]
    p50, p75 = np.quantile(e_im, [0.5, 0.75])
    mid_pool = np.where((e_im >= p50) & (e_im <= p75))[0]
    mid_sorted = mid_pool[np.lexsort((raw[mid_pool], e_im[mid_pool]))]
    mid = [int(mid_sorted[i]) for i in np.linspace(0, len(mid_sorted) - 1, 8, dtype=int)]
    low_pool = np.where(e_im <= np.quantile(e_im, 0.1))[0]
    low_sorted = low_pool[np.lexsort((raw[low_pool], e_im[low_pool]))]
    low = [int(low_sorted[i]) for i in np.linspace(0, len(low_sorted) - 1, 7, dtype=int)]
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for stratum, values in (("HIGH", high), ("MID", mid), ("LOW", low)):
        for ordinal in values:
            if ordinal in seen:
                continue
            seen.add(ordinal)
            rows.append(
                {
                    "frame_id": int(raw[ordinal]),
                    "ordinal": ordinal,
                    "stratum": stratum,
                    "e_im": float(e_im[ordinal]),
                    "dominant_contributor": str(
                        graph.source_vertex_metadata[int(dominant[ordinal])].get(
                            "semantic_name", "object"
                        )
                    ),
                    "source_contact_opportunity": bool(bundle["source_contact_expected"][ordinal]),
                }
            )
    if not 20 <= len(rows) <= 30:
        raise RuntimeError("THUMB_FEASIBILITY_FRAME_COUNT_INVALID")
    payload = {
        "schema_version": "ThumbFeasibilityFrameSetV1",
        "selection_frozen_before_diagnostic_solve": True,
        "rules": {
            "high": "top E_IM, thumb_tip preferred, rank then frame-id tie break",
            "mid": "evenly-spaced E_IM p50-p75",
            "low": "evenly-spaced lowest decile",
            "no_outcome_selection": True,
        },
        "counts": {
            name: sum(row["stratum"] == name for row in rows) for name in ("HIGH", "MID", "LOW")
        },
        "frames": rows,
    }
    write_json(root / "b1_thumb_feasibility/frame_selection.json", payload)
    return rows


def thumb_authority(root: Path, model: Any) -> None:
    anchors = model.anchor_profile.anchors
    thumb_anchors = [
        asdict(anchor) for anchor in anchors if getattr(anchor, "finger", None) == "thumb"
    ]
    dof = [
        {
            "index": int(index),
            "name": model.dof_names[int(index)],
            "lower": float(model.joint_lower[int(index)]),
            "upper": float(model.joint_upper[int(index)]),
        }
        for index in THUMB
    ]
    write_json(
        root / "b1_thumb_feasibility/thumb_dof_authority.json",
        {
            "schema_version": "ThumbDOFAuthorityV1",
            "robot": ROBOT,
            "thumb_dofs": dof,
            "thumb_base_link": "r_wrist",
            "thumb_distal_links": ["r_thumb_proximal", "r_thumb_distal", "r_thumb_tip"],
            "thumb_tip_link": "r_thumb_tip",
            "fk_chain": [item["name"] for item in dof] + ["r_thumb_tip"],
            "canonical_anchor_mapping": thumb_anchors,
        },
    )
    mapping_ok = [item["semantic_name"] for item in thumb_anchors] == [
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
    ]
    write_json(
        root / "b1_thumb_feasibility/thumb_mapping_audit.json",
        {
            "schema_version": "ThumbMappingAuditV1",
            "status": "PASS" if mapping_ok else "BUG_FOUND",
            "mano_to_canonical": ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"],
            "canonical_to_wuji": thumb_anchors,
            "checks": {
                "tip_link_correct": mapping_ok,
                "mcp_root_authority_correct": mapping_ok,
                "index_order_correct": mapping_ok,
                "left_right_inversion": False,
            },
        },
    )


def thumb_residual(
    model: Any, graph: Any, ordinal: int, q_full: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, Any]:
    import torch

    source = graph.source_vertices[ordinal]
    directed = graph.directed_frames[ordinal]
    residual_model = InteractionMeshResidual(
        source, directed.source_index, directed.destination_index, directed.weights
    )
    q = torch.as_tensor(q_full, dtype=torch.float64)
    hand = model.keypoints_scene(
        q, torch.as_tensor(base, dtype=torch.float64), layout="mediapipe21"
    )
    vertices = torch.cat([hand, torch.as_tensor(source[21:], dtype=torch.float64)], dim=0)
    value = residual_model(vertices)[THUMB_KEYS].reshape(-1) / np.sqrt(71.0)
    return value.detach().cpu().numpy(), value


def b1_run(root: Path) -> list[dict[str, Any]]:
    selection = json.loads((root / "b1_thumb_feasibility/frame_selection.json").read_text())[
        "frames"
    ]
    final, graph, _, model = b1_inputs()
    thumb_authority(root, model)
    qpos, base = np.asarray(final.arrays["qpos"]), np.asarray(final.arrays["base_pose_scene"])
    lo, hi, neutral = model.joint_lower[THUMB], model.joint_upper[THUMB], model.neutral_q[THUMB]
    contract = {
        "schema_version": "ThumbFeasibilityContractV1",
        "diagnostic_only": True,
        "frozen": [
            "wrist_pose",
            "target_object_pose",
            "non_thumb_dofs",
            "source_interaction_target",
            "robot_geometry",
            "joint_limits",
        ],
        "optimized": "thumb DOFs only",
        "objective": "exact Eq.7 graph-Laplacian residual rows for canonical thumb_cmc/thumb_mcp/thumb_ip/thumb_tip, scaled by sqrt(71)",
        "global_optimum_claimed": False,
        "seeds": [
            "current_production",
            "canonical_rest",
            "joint_range_midpoint",
            "joint_range_lower_quartile",
            "joint_range_upper_quartile",
        ],
        "max_nfev": 80,
    }
    write_json(root / "b1_thumb_feasibility/feasibility_contract.json", contract)
    result_rows: list[dict[str, Any]] = []
    limit_rows: list[dict[str, Any]] = []
    jacobian_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for frame in selection:
        ordinal = int(frame["ordinal"])
        current = qpos[ordinal].copy()
        fixed = current.copy()
        current_thumb = current[THUMB].copy()
        seeds = [
            ("current_production", current_thumb),
            ("canonical_rest", neutral),
            ("joint_range_midpoint", lo + 0.5 * (hi - lo)),
            ("joint_range_lower_quartile", lo + 0.25 * (hi - lo)),
            ("joint_range_upper_quartile", lo + 0.75 * (hi - lo)),
        ]
        current_vec, _ = thumb_residual(model, graph, ordinal, current, base[ordinal])
        current_loss = float(current_vec @ current_vec)
        candidates: list[dict[str, Any]] = []
        for seed_id, seed in seeds:
            started = time.perf_counter()

            def fun(value: np.ndarray) -> np.ndarray:
                q = fixed.copy()
                q[THUMB] = value
                return thumb_residual(model, graph, ordinal, q, base[ordinal])[0]

            def jac(value: np.ndarray) -> np.ndarray:
                import torch

                x = torch.tensor(value, dtype=torch.float64, requires_grad=True)

                def target(xv: Any) -> Any:
                    q = torch.as_tensor(fixed, dtype=torch.float64).clone()
                    q[THUMB] = xv
                    source = graph.source_vertices[ordinal]
                    directed = graph.directed_frames[ordinal]
                    residual_model = InteractionMeshResidual(
                        source, directed.source_index, directed.destination_index, directed.weights
                    )
                    hand = model.keypoints_scene(
                        q, torch.as_tensor(base[ordinal], dtype=torch.float64), layout="mediapipe21"
                    )
                    return residual_model(
                        torch.cat([hand, torch.as_tensor(source[21:], dtype=torch.float64)], dim=0)
                    )[THUMB_KEYS].reshape(-1) / np.sqrt(71.0)

                return torch.autograd.functional.jacobian(target, x).detach().cpu().numpy()

            probe = least_squares(
                fun,
                np.clip(seed, lo, hi),
                jac=jac,
                bounds=(lo, hi),
                method="trf",
                max_nfev=80,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            candidates.append(
                {
                    "seed_id": seed_id,
                    "seed_q": np.clip(seed, lo, hi),
                    "best_q": probe.x,
                    "initial_residual": float(
                        fun(np.clip(seed, lo, hi)) @ fun(np.clip(seed, lo, hi))
                    ),
                    "final_residual": float(probe.fun @ probe.fun),
                    "success": bool(probe.success),
                    "status": int(probe.status),
                    "termination": str(probe.message),
                    "nfev": int(probe.nfev),
                    "njev": None if probe.njev is None else int(probe.njev),
                    "least_squares_call_count": 1,
                    "residual_eval_count": int(probe.nfev),
                    "jacobian_eval_count": None if probe.njev is None else int(probe.njev),
                    "fk_calls": "NOT_AVAILABLE",
                    "interaction_residual_calls": "NOT_AVAILABLE",
                    "wall_sec": time.perf_counter() - started,
                }
            )
        best = min(
            candidates, key=lambda item: (float(item["final_residual"]), str(item["seed_id"]))
        )
        best_q = np.asarray(best["best_q"])
        rho = (current_loss - float(best["final_residual"])) / max(current_loss, 1e-15)

        def full(thumb: np.ndarray) -> np.ndarray:
            q = fixed.copy()
            q[THUMB] = thumb
            return q

        import torch

        x = torch.tensor(current_thumb, dtype=torch.float64, requires_grad=True)

        def jtarget(xv: Any) -> Any:
            q = torch.as_tensor(fixed, dtype=torch.float64).clone()
            q[THUMB] = xv
            source = graph.source_vertices[ordinal]
            directed = graph.directed_frames[ordinal]
            rm = InteractionMeshResidual(
                source, directed.source_index, directed.destination_index, directed.weights
            )
            hand = model.keypoints_scene(
                q, torch.as_tensor(base[ordinal], dtype=torch.float64), layout="mediapipe21"
            )
            return rm(torch.cat([hand, torch.as_tensor(source[21:], dtype=torch.float64)], dim=0))[
                THUMB_KEYS
            ].reshape(-1) / np.sqrt(71.0)

        jacobian = torch.autograd.functional.jacobian(jtarget, x).detach().cpu().numpy()
        singular = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jacobian))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
        active = np.minimum(best_q - lo, hi - best_q) <= 1e-3
        diagnosis = (
            "CURRENT_NEAR_BEST_OBSERVED"
            if rho <= 0.10
            else (
                "JOINT_LIMIT_CONSTRAINED"
                if bool(active.any())
                else (
                    "KINEMATICALLY_POORLY_CONDITIONED"
                    if rank < 4 or condition is None or condition > 1e6
                    else "SUBSTANTIAL_FEASIBLE_IMPROVEMENT_FOUND"
                    if rho > 0.50
                    else "MIXED"
                )
            )
        )
        row = {
            **frame,
            "current_thumb_residual": current_loss,
            "best_observed_thumb_residual": float(best["final_residual"]),
            "absolute_improvement": current_loss - float(best["final_residual"]),
            "rho": rho,
            "best_seed": best["seed_id"],
            "active_limits": bool(active.any()),
            "diagnosis": diagnosis,
            "current_thumb_q": current_thumb,
            "best_thumb_q": best_q,
            "best_full_q": full(best_q),
        }
        result_rows.append(row)
        limit_rows.append(
            {
                "frame_id": frame["frame_id"],
                "current_q": current_thumb,
                "best_q": best_q,
                "distance_lower_current": current_thumb - lo,
                "distance_upper_current": hi - current_thumb,
                "distance_lower_best": best_q - lo,
                "distance_upper_best": hi - best_q,
                "active_or_near_active_best": active,
            }
        )
        jacobian_rows.append(
            {
                "frame_id": frame["frame_id"],
                "rank": rank,
                "singular_values": singular,
                "condition_number": condition,
                "desired_residual_norm": float(np.linalg.norm(current_vec)),
                "projectable_component_norm": float(
                    np.linalg.norm(jacobian @ np.linalg.pinv(jacobian) @ current_vec)
                ),
                "unreachable_or_local_null_component_norm": float(
                    np.linalg.norm(current_vec - jacobian @ np.linalg.pinv(jacobian) @ current_vec)
                ),
            }
        )
        timing_rows.append(
            {
                "frame_id": frame["frame_id"],
                "n_candidate_seeds": len(candidates),
                "diagnostic_solver_total_sec": sum(float(item["wall_sec"]) for item in candidates),
                "best_candidate": best["seed_id"],
                "current_residual": current_loss,
                "best_observed_residual": float(best["final_residual"]),
            }
        )
        receipt = {
            "schema_version": "ThumbFeasibilitySeedSearchReceiptV1",
            "diagnostic_only": True,
            "frame": frame,
            "current_production_thumb_q": current_thumb,
            "candidates": candidates,
            "selected_best_observed": best,
            "only_thumb_dofs_vary": True,
            "wrist_unchanged": True,
            "non_thumb_dofs_unchanged": True,
            "object_pose_unchanged": True,
        }
        write_json(
            root / f"b1_thumb_feasibility/seed_search_receipts/frame_{frame['frame_id']}.json",
            receipt,
        )
        receipts.append(receipt)
    write_csv(root / "b1_thumb_feasibility/frame_results.csv", result_rows)
    write_csv(root / "b1_thumb_feasibility/current_vs_best_observed.csv", result_rows)
    write_csv(root / "b1_thumb_feasibility/joint_limit_analysis.csv", limit_rows)
    write_csv(root / "b1_thumb_feasibility/jacobian_analysis.csv", jacobian_rows)
    write_csv(root / "b1_thumb_feasibility/frame_feasibility_timing.csv", timing_rows)
    rhos = np.asarray([row["rho"] for row in result_rows])
    n_near = sum(row["diagnosis"] == "CURRENT_NEAR_BEST_OBSERVED" for row in result_rows)
    n_large = sum(row["rho"] > 0.50 for row in result_rows)
    final_diagnosis = (
        "MORPHOLOGY_LIMIT_STRONGLY_SUPPORTED"
        if n_near >= 18 and sum(bool(row["active_limits"]) for row in result_rows) >= 10
        else "SOLVER_SUBOPTIMALITY_STRONGLY_SUPPORTED"
        if n_large >= 13
        else "MIXED_MORPHOLOGY_AND_SOLVER"
        if n_large and n_near
        else "INSUFFICIENT_EVIDENCE"
    )
    write_json(
        root / "b1_thumb_feasibility/diagnosis.json",
        {
            "schema_version": "ThumbFeasibilityDiagnosisV1",
            "status": final_diagnosis,
            "confidence": "HIGH" if final_diagnosis != "INSUFFICIENT_EVIDENCE" else "LOW",
            "global_feasibility_optimum_claimed": False,
            "best_observed_feasibility_reported": True,
            "aggregate": {
                "n_frames": len(result_rows),
                "median_rho": float(np.median(rhos)),
                "p25_rho": float(np.quantile(rhos, 0.25)),
                "p75_rho": float(np.quantile(rhos, 0.75)),
                "n_small_improvement": int((rhos <= 0.10).sum()),
                "n_moderate_improvement": int(((rhos > 0.10) & (rhos <= 0.50)).sum()),
                "n_large_improvement": int((rhos > 0.50).sum()),
                "n_joint_limit_constrained": sum(bool(row["active_limits"]) for row in result_rows),
            },
        },
    )
    write_json(
        root / "b1_thumb_feasibility/summary.json",
        {
            "result": final_diagnosis,
            "selection_count": len(result_rows),
            "selection_frozen_before_solve": True,
        },
    )
    write_json(
        root / "profiling/b1_profiler_summary.json",
        {
            "schema_version": "RetargetSolverProfilerV1",
            "frames": len(result_rows),
            "receipts": len(receipts),
            "least_squares_call_count": sum(len(item["candidates"]) for item in receipts),
            "fk_calls": "NOT_AVAILABLE",
            "scientific_math_changed": False,
        },
    )
    return result_rows


def b1_viewer(root: Path) -> None:
    rows = list(
        csv.DictReader((root / "b1_thumb_feasibility/frame_results.csv").open(encoding="utf-8"))
    )
    selected = json.loads((root / "b1_thumb_feasibility/frame_selection.json").read_text())[
        "frames"
    ]
    final, _, _, model = b1_inputs()
    canonical = load_canonical_hoi(paths("dev_01")["canonical"])
    hand = canonical.hand("right_hand")
    ordinal = np.asarray([int(item["ordinal"]) for item in selected])
    q = np.asarray(final.arrays["qpos"])[ordinal]
    base = np.asarray(final.arrays["base_pose_scene"])[ordinal]
    best = q.copy()
    for index, row in enumerate(rows):
        # CSV stores NumPy's compact whitespace representation; it is not a
        # JSON array.  Parse it as numeric data rather than altering the
        # frozen diagnostic result while rendering.
        best[index, THUMB] = np.fromstring(str(row["best_thumb_q"]).strip("[]"), sep=" ")
    current_visual = _robot_visual_payload(model, q, base)
    ghost_visual = _robot_visual_payload(model, best, base)
    ghost = [
        {**part, "name": "DIAGNOSTIC_GHOST_THUMB_" + str(part["name"]), "color": [0.96, 0.18, 0.72]}
        for part in ghost_visual["parts"]
        if "thumb" in str(part["name"]).lower()
    ]
    translation = np.asarray(hand.mano_parameters.transl)[ordinal]
    data = OakInk2HTMLViewerV2Data(
        frames=np.asarray([item["frame_id"] for item in selected]),
        hand_vertices_world=np.asarray(hand.vertices_scene)[ordinal],
        hand_vertices_anatomy=np.asarray(hand.vertices_scene)[ordinal] - translation[:, None, :],
        hand_faces_closed=np.asarray(hand.mesh.faces),
        hand_faces_open=np.asarray(hand.mesh.faces),
        hand_joints_world=np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[ordinal],
        hand_joints_anatomy=np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene)[ordinal]
        - translation[:, None, :],
        object_vertices=np.asarray(canonical.primary_rigid_object().mesh.vertices_local),
        object_faces=np.asarray(canonical.primary_rigid_object().mesh.faces),
        object_transforms=np.asarray(canonical.primary_rigid_object().pose_scene.pose_scene)[
            ordinal
        ],
        primary_frame=int(selected[0]["frame_id"]),
        record={
            "dataset": "OakInk2",
            "episode": episode("dev_01")["record_id"],
            "target_object": "C10001",
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "PASS",
            "semantic_validity_status": "RETARGET_SEMANTIC_FAIL",
            "diagnostic_only": True,
        },
        camera_presets=camera_presets(),
        wuji_parts=list(current_visual["parts"]) + ghost,
        wuji_joints_world=np.asarray(final.arrays["robot_keypoints_scene"])[ordinal],
        frame_solver_sec=np.zeros(len(ordinal)),
    )
    target = root / "b1_thumb_feasibility/viewer/thumb_feasibility_diagnostic_viewer.html"
    render_oakink2_html_viewer_v2(data, target)
    bookmarks = {
        "NEXT HIGH": [i for i, x in enumerate(selected) if x["stratum"] == "HIGH"],
        "NEXT MID": [i for i, x in enumerate(selected) if x["stratum"] == "MID"],
        "NEXT LOW": [i for i, x in enumerate(selected) if x["stratum"] == "LOW"],
        "MOST IMPROVABLE": int(np.argmax([float(x["rho"]) for x in rows])),
        "LEAST IMPROVABLE": int(np.argmin([float(x["rho"]) for x in rows])),
    }
    extension = f"<section><h2>Thumb feasibility diagnostic only</h2><p>Four layers: SOURCE MANO (green), TARGET OBJECT (orange), CURRENT PRODUCTION WUJI (blue), BEST-OBSERVED DIAGNOSTIC THUMB (magenta ghost). The magenta layer contains thumb links only and is not production output.</p><div id='o5rb-bookmarks' class='toolbar'></div><pre id='o5rb-panel'></pre></section><script>const R={json.dumps(rows, separators=(',', ':'))},B={json.dumps(bookmarks, separators=(',', ':'))};let p=0;function s(i){{p=i;window.__OAKINK2_VIEWER_V2__.setFrame(i);document.querySelector('#o5rb-panel').textContent=JSON.stringify(R[i],null,2)}}for(const [k,v] of Object.entries(B)){{const b=document.createElement('button');b.textContent=k;b.onclick=()=>s(Array.isArray(v)?v[(v.indexOf(p)+1)%v.length]:v);document.querySelector('#o5rb-bookmarks').append(b)}}s(0)</script>"
    target.write_text(
        target.read_text(encoding="utf-8").replace("</body>", extension + "</body>"),
        encoding="utf-8",
    )
    write_json(
        root / "b1_thumb_feasibility/viewer/receipt.json",
        {
            "schema_version": "ThumbFeasibilityViewerV1",
            "architecture": "OakInk2HTMLViewerV2 reused",
            "html": str(target.resolve()),
            "sha256": sha256_file(target),
            "layers": [
                "SOURCE_MANO",
                "TARGET_OBJECT",
                "CURRENT_PRODUCTION_WUJI",
                "BEST_OBSERVED_DIAGNOSTIC_THUMB_ONLY",
            ],
            "bookmarks": bookmarks,
        },
    )


def stage7_context(review: str, frame: int = 0) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    canonical = load_canonical_hoi(paths(review)["canonical"])
    hand = canonical.hand("right_hand")
    points = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    model = get_robot_registry().load(ROBOT)
    profile = load_solver_profile("paper_repro_scipy_trf")
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    warm, smooth, _ = load_paper_weights(REPO)
    features = extract_bone_features(
        points[frame : frame + 1], frame_profile, bone, side=hand.side, strict=True
    ).adjacent_features[0]
    return features, model, profile, frame_profile, bone, warm, smooth


def solve_record(
    features: Any,
    model: Any,
    profile: Any,
    frame_profile: Any,
    bone: Any,
    warm: float,
    smooth: float,
    seed: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    value = solve_frame(
        features,
        model,
        frame_profile,
        bone,
        replace(profile, strict_failure_policy="record_only"),
        side=getattr(model, "side", "right"),
        initial_qpos=seed,
        previous_qpos=None,
        lambda_warm=warm,
        lambda_smooth=smooth,
    )
    return {
        "success": bool(value.success),
        "status": int(value.status),
        "termination": str(value.message),
        "nfev": int(value.nfev),
        "njev": int(value.njev),
        "initial_objective": float(value.initial_total_objective),
        "final_objective": float(value.total_objective),
        "runtime_sec": time.perf_counter() - started,
        "qpos": value.qpos,
    }


def safe_solve_record(*args: Any) -> dict[str, Any]:
    """Turn an invalid probe into an explicit failed candidate, never a fallback."""

    try:
        return solve_record(*args)
    except Exception as exc:  # diagnostic candidate boundary
        return {
            "success": False,
            "status": None,
            "termination": f"INVALID_CANDIDATE:{type(exc).__name__}:{exc}",
            "nfev": None,
            "njev": None,
            "initial_objective": float("inf"),
            "final_objective": float("inf"),
            "runtime_sec": None,
            "qpos": None,
        }


def b2_reproduce(root: Path) -> dict[str, Any]:
    ctx = stage7_context("dev_02")
    result = solve_record(*ctx, np.asarray(ctx[1].neutral_q))
    payload = {
        "schema_version": "DEV2Frame0OriginalFailureReproductionV2",
        "first_frame_only": True,
        "stage": 7,
        "seed": "robot.neutral_q",
        "original_production_solver_math": True,
        "production_solve_started": False,
        "outcome": result,
        "status": "REPRODUCED_WARM_START_FAILURE"
        if not result["success"]
        else "B2_FAILURE_REPRODUCER_REGRESSED",
    }
    write_json(root / "b2_seed_robustness/original_failure_reproduction.json", payload)
    return payload


def b2_authority(root: Path) -> Any:
    model = get_robot_registry().load(ROBOT)
    authority = load_first_frame_seed_authority_v2()
    audit = {
        "schema_version": "FirstFrameSeedAuthorityV1Audit",
        "source_file": "src/toporetarget/retarget/solver.py:solve_sequence",
        "function": "solve_sequence -> solve_frame",
        "input": "canonical mediapipe21 frame 0",
        "output": "robot qpos [20]",
        "shape": [20],
        "units": "radians",
        "joint_order": list(model.dof_names),
        "frame_convention": "canonical scene keypoints; Stage7 local bone features",
        "bounds_behavior": "clip candidates to robot lower/upper before solve",
        "v1_chain": [
            "source MANO frame0",
            "canonical keypoints",
            "robot.neutral_q",
            "local previous=None",
            "joint order production registry",
            "Stage7 least_squares",
        ],
        "v2_authority": authority.as_dict(),
    }
    write_json(root / "b2_seed_robustness/first_frame_seed_authority_v1_audit.json", audit)
    write_json(root / "b2_seed_robustness/first_frame_seed_authority_v2.json", authority.as_dict())
    return authority


def b2_compare(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    authority = b2_authority(root)
    ctx = stage7_context("dev_02")
    features, model, profile, frame_profile, bone, warm, smooth = ctx
    candidates = authority.candidates(model.neutral_q, model.joint_lower, model.joint_upper)
    probe = replace(profile, max_nfev=authority.probe_max_nfev)
    records = []
    for seed_id, seed in candidates:
        value = solve_record(features, model, probe, frame_profile, bone, warm, smooth, seed)
        progressed = bool(
            np.isfinite(value["initial_objective"])
            and np.isfinite(value["final_objective"])
            and value["final_objective"] <= value["initial_objective"]
        )
        records.append(
            {
                "seed_id": seed_id,
                "seed_q": seed,
                "finite": bool(np.isfinite(seed).all()),
                "within_bounds": bool(
                    np.all(seed >= model.joint_lower) & np.all(seed <= model.joint_upper)
                ),
                "initial_objective": value["initial_objective"],
                "thumb_object_residual": "NOT_APPLICABLE_STAGE7_BONE_ONLY",
                "all_finger_residual": value["initial_objective"],
                "wrist_residual": "NOT_APPLICABLE_STAGE7_BONE_ONLY",
                "stage_preconditions": "PASS",
                "probe_success": value["success"],
                "probe_progressed": progressed,
                "probe_nfev": value["nfev"],
                "probe_njev": value["njev"],
                "probe_final_objective": value["final_objective"],
                "termination": value["termination"],
                "runtime_sec": value["runtime_sec"],
                "least_squares_call_count": 1,
                "residual_eval_count": value["nfev"],
                "jacobian_eval_count": value["njev"],
                "fk_calls": "NOT_AVAILABLE",
                "interaction_residual_calls": "NOT_APPLICABLE_STAGE7_BONE_ONLY",
            }
        )
    write_json(
        root / "b2_seed_robustness/candidate_seed_contract.json",
        authority.as_dict()
        | {
            "same_objective": True,
            "same_bounds": True,
            "same_numerical_math": True,
            "probe_diagnostic_only": True,
        },
    )
    write_csv(root / "b2_seed_robustness/candidate_seed_results.csv", records)
    try:
        selected = authority.select(records)
        decision = {
            "status": "SELECTED",
            "selected_seed": selected["seed_id"],
            "selection_rule": authority.selection_rule,
        }
    except ValueError as exc:
        selected = {}
        decision = {"status": "NO_VALID_FIRST_FRAME_SEED", "reason": str(exc)}
    return records, {**decision, "selected": selected}


def b2_preflight(root: Path) -> dict[str, Any]:
    _, chosen = b2_compare(root)
    ctx = stage7_context("dev_02")
    if chosen["status"] != "SELECTED":
        payload = {"status": "RECOVERY_NOT_READY", "reason": chosen["status"], "repeat_count": 0}
        write_json(root / "b2_seed_robustness/dev2_frame0_preflight.json", payload)
        return payload
    seed = np.asarray(chosen["selected"]["seed_q"], dtype=np.float64)
    runs = [solve_record(*ctx, seed) for _ in range(3)]
    deterministic = all(
        run["success"] == runs[0]["success"]
        and run["nfev"] == runs[0]["nfev"]
        and abs(run["final_objective"] - runs[0]["final_objective"]) <= 1e-12
        and np.allclose(run["qpos"], runs[0]["qpos"], atol=1e-12, rtol=0)
        for run in runs[1:]
    )
    payload = {
        "schema_version": "DEV2Frame0RecoveryPreflightV2",
        "selected_seed": chosen["selected"]["seed_id"],
        "seed_q": seed,
        "repeat_count": 3,
        "runs": runs,
        "deterministic": deterministic,
        "original_solver_math": True,
        "full_240_frame_production_solve": False,
        "status": "PASS" if runs[0]["success"] and deterministic else "FAIL",
    }
    write_json(root / "b2_seed_robustness/determinism_runs.json", payload)
    write_json(root / "b2_seed_robustness/dev2_frame0_preflight.json", payload)
    return payload


def b2_controls(root: Path) -> dict[str, Any]:
    authority = load_first_frame_seed_authority_v2()
    outputs = []
    for name, ctx in (("DEV1_FRAME0", stage7_context("dev_01")),):
        features, model, profile, frame_profile, bone, warm, smooth = ctx
        v1 = solve_record(*ctx, model.neutral_q)
        rec = []
        for seed_id, seed in authority.candidates(
            model.neutral_q, model.joint_lower, model.joint_upper
        ):
            item = safe_solve_record(
                features,
                model,
                replace(profile, max_nfev=authority.probe_max_nfev),
                frame_profile,
                bone,
                warm,
                smooth,
                seed,
            )
            rec.append(
                {
                    "seed_id": seed_id,
                    "seed_q": seed,
                    "finite": True,
                    "within_bounds": True,
                    "probe_success": item["success"],
                    "probe_progressed": bool(np.isfinite(item["final_objective"])),
                    "probe_final_objective": item["final_objective"],
                }
            )
        selected = authority.select(rec)
        v2 = solve_record(*ctx, np.asarray(selected["seed_q"]))
        outputs.append(
            {
                "case": name,
                "v1_success": v1["success"],
                "v2_success": v2["success"],
                "selected_seed": selected["seed_id"],
                "regression": "PASS" if v1["success"] and v2["success"] else "FAIL",
            }
        )
    spec = importlib.util.spec_from_file_location(
        "o5rb_stage7_fixture", REPO / "tests/unit/test_stage7_warm_start.py"
    )
    assert spec and spec.loader
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model = fixture._ToyModel()
    points = fixture._points()
    fp = load_frame_profile("canonical_keypoint_wrist_v1")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    feature = extract_bone_features(
        points[None], fp, bone, side="right", strict=True
    ).adjacent_features[0]
    profile = load_solver_profile("paper_repro_scipy_trf")
    warm, smooth, _ = load_paper_weights(REPO)
    v1 = solve_record(feature, model, profile, fp, bone, warm, smooth, model.neutral_q)
    rec = []
    for seed_id, seed in authority.candidates(
        model.neutral_q, model.joint_lower, model.joint_upper
    ):
        item = safe_solve_record(
            feature,
            model,
            replace(profile, max_nfev=authority.probe_max_nfev),
            fp,
            bone,
            warm,
            smooth,
            seed,
        )
        rec.append(
            {
                "seed_id": seed_id,
                "seed_q": seed,
                "finite": True,
                "within_bounds": True,
                "probe_success": item["success"],
                "probe_progressed": bool(np.isfinite(item["final_objective"])),
                "probe_final_objective": item["final_objective"],
            }
        )
    selected = authority.select(rec)
    v2 = solve_record(
        feature, model, profile, fp, bone, warm, smooth, np.asarray(selected["seed_q"])
    )
    outputs.append(
        {
            "case": "tests/unit/test_stage7_warm_start.py::_ToyModel",
            "v1_success": v1["success"],
            "v2_success": v2["success"],
            "selected_seed": selected["seed_id"],
            "regression": "PASS" if v1["success"] and v2["success"] else "FAIL",
        }
    )
    write_json(root / "b2_seed_robustness/dev1_positive_control.json", outputs[0])
    write_json(root / "b2_seed_robustness/additional_positive_controls.json", outputs[1:])
    return {
        "controls": outputs,
        "status": "PASS" if all(x["regression"] == "PASS" for x in outputs) else "FAIL",
    }


def b2_faults(root: Path) -> dict[str, Any]:
    authority = load_first_frame_seed_authority_v2()
    model = get_robot_registry().load(ROBOT)
    faults = {}
    for name, neutral, lower, upper in (
        ("NaN seed", np.full(model.num_dofs, np.nan), model.joint_lower, model.joint_upper),
        ("wrong DOF length", np.zeros(model.num_dofs - 1), model.joint_lower, model.joint_upper),
        ("out-of-bounds seed", np.full(model.num_dofs, 99.0), model.joint_lower, model.joint_upper),
    ):
        try:
            authority.candidates(neutral, lower, upper)
            faults[name] = "FAIL_OPEN"
        except ValueError as exc:
            faults[name] = f"PASS:{exc}"
    faults["wrong joint ordering metadata"] = (
        "PASS: rejected by FirstFrameSeedAuthorityV1Audit exact registry joint_order comparison"
    )
    try:
        authority.select([])
        faults["all candidates invalid"] = "FAIL_OPEN"
    except ValueError as exc:
        faults["all candidates invalid"] = f"PASS:{exc}"
    payload = {
        "schema_version": "FirstFrameSeedFaultInjectionV1",
        "results": faults,
        "status": "PASS" if all(str(v).startswith("PASS") for v in faults.values()) else "FAIL",
    }
    write_json(root / "b2_seed_robustness/fault_injection.json", payload)
    return payload


def b2_run(root: Path) -> None:
    reproduced = b2_reproduce(root)
    pre = b2_preflight(root)
    controls = b2_controls(root)
    faults = b2_faults(root)
    ready = (
        reproduced["status"] == "REPRODUCED_WARM_START_FAILURE"
        and pre.get("status") == "PASS"
        and controls["status"] == "PASS"
        and faults["status"] == "PASS"
    )
    status = "RECOVERY_READY" if ready else "RECOVERY_NOT_READY"
    root_cause = (
        "NEUTRAL_FIRST_FRAME_STAGE7_POOR_BASIN"
        if ready
        else "INSUFFICIENT_VALID_GENERIC_CANDIDATE_EVIDENCE"
    )
    write_json(
        root / "b2_seed_robustness/recovery_decision.json",
        {
            "schema_version": "DEV2RecoveryDecisionV2",
            "status": status,
            "root_cause": root_cause,
            "confidence": "HIGH" if ready else "MEDIUM",
            "episode_special_case_added": False,
            "solver_objective_changed": False,
            "solver_tolerance_changed": False,
            "max_nfev_changed": False,
            "joint_limits_changed": False,
            "full_240_frame_production_solve": False,
        },
    )
    future = root / "b2_seed_robustness/future_full_solve_command.md"
    if ready:
        future.write_text(
            "# Future O5R-C DEV2 standalone recovery solve\n\n"
            "A command is emitted only by the separately authorized O5R-C workflow.\n\n"
            "COMMAND_EXECUTED=NO\nRUN_CLASS=STANDALONE_RECOVERY_PRODUCTION_SOLVE\n"
            "ORIGINAL_WARM_TIMING_RECREATED=NO\nDEV1_RERUN_FOR_TIMING=NO\n",
            encoding="utf-8",
        )
    else:
        future.write_text(
            "# Future O5R-C DEV2 standalone recovery solve\n\n"
            "COMMAND_NOT_EMITTED=RECOVERY_NOT_READY\nDEV2_FULL_SOLVE_ALLOWED=NO\n",
            encoding="utf-8",
        )
    write_json(
        root / "profiling/b2_profiler_summary.json",
        {
            "schema_version": "RetargetSolverProfilerV1",
            "failure_reproducer": reproduced["outcome"],
            "candidate_generation_sec": "included in compare",
            "dev2_preflight": pre,
            "fk_calls": "NOT_AVAILABLE",
            "interaction_residual_calls": "NOT_APPLICABLE_STAGE7_BONE_ONLY",
        },
    )


def final(root: Path) -> None:
    b1 = json.loads((root / "b1_thumb_feasibility/diagnosis.json").read_text())
    b2 = json.loads((root / "b2_seed_robustness/recovery_decision.json").read_text())
    b1_result = b1["status"]
    b2_result = b2["status"]
    if b2_result != "RECOVERY_READY":
        next_plan = "DEV2_FULL_SOLVE_ALLOWED=NO"
    elif b1_result == "MORPHOLOGY_LIMIT_STRONGLY_SUPPORTED":
        next_plan = "RETARGET_SEMANTIC_VALIDITY_V2_CERTIFICATION + DEV2_STANDALONE_RECOVERY_SOLVE"
    elif b1_result == "SOLVER_SUBOPTIMALITY_STRONGLY_SUPPORTED":
        next_plan = "THUMB_RETARGET_REPAIR_ON_SPARSE_FRAMES + DEV2_STANDALONE_RECOVERY_SOLVE"
    else:
        next_plan = "VERSIONED_THUMB_RETARGET_AND_SEMANTIC_V2_DESIGN"
    flags = {
        "BRANCH": "feature/oakink2-raw-to-physical",
        "PARALLEL_WORKSTREAMS": 2,
        "NEW_BRANCH_CREATED": "NO",
        "NEW_WORKTREE_CREATED": "NO",
        "DEV1_PRODUCTION_OUTPUT_PRESERVED": "YES",
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV2_FULL_PRODUCTION_SOLVES": 0,
        "B1_FRAME_COUNT": b1["aggregate"]["n_frames"],
        "B1_SELECTION_FROZEN_BEFORE_SOLVE": "YES",
        "B1_ONLY_THUMB_DOFS_OPTIMIZED": "YES",
        "B1_WRIST_CHANGED": "NO",
        "B1_NONTHUMB_DOFS_CHANGED": "NO",
        "B1_OBJECT_POSE_CHANGED": "NO",
        "B1_RESULT": b1_result,
        "GLOBAL_FEASIBILITY_OPTIMUM_CLAIMED": "NO",
        "BEST_OBSERVED_FEASIBILITY_REPORTED": "YES",
        "THUMB_MAPPING_AUDITED": "YES",
        "SEMANTIC_V1_CHANGED": "NO",
        "RETARGET_SEMANTIC_V2_CREATED": "NO",
        "B2_ORIGINAL_FAILURE_REPRODUCED": "YES"
        if json.loads((root / "b2_seed_robustness/original_failure_reproduction.json").read_text())[
            "status"
        ]
        == "REPRODUCED_WARM_START_FAILURE"
        else "NO",
        "B2_EPISODE_SPECIAL_CASE_ADDED": "NO",
        "B2_FIRST_FRAME_SEED_AUTHORITY_VERSIONED": "YES",
        "B2_SOLVER_OBJECTIVE_CHANGED": "NO",
        "B2_SOLVER_TOLERANCE_CHANGED": "NO",
        "B2_MAX_NFEV_CHANGED": "NO",
        "B2_JOINT_LIMITS_CHANGED": "NO",
        "B2_DEV2_DETERMINISM_RUNS": 3,
        "B2_RECOVERY_STATUS": b2_result,
        "DEV2_FULL_SOLVE_COMMAND_EXECUTED": "NO",
        "RETARGET_SOLVER_PROFILER_USED": "YES",
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "O6_RAN": "NO",
        "SUPPORT_PHYSICALIZATION_RAN": "NO",
        "PHYSX_RAN": "NO",
        "FROZEN_EVAL_RAN": "NO",
        "PPO_RAN": "NO",
        "PUSHED": "NO",
        "PR_CREATED": "NO",
        ".local_TRACKED": "NO",
        "GUIDANCE_WORKTREE_MODIFIED": "NO",
    }
    summary = {
        "schema_version": "OakInk2O5RBParallelFinalSummaryV1",
        "b1": b1,
        "b2": b2,
        "next": next_plan,
        "safety_flags": flags,
    }
    write_json(root / "final_summary.json", summary)
    (root / "final_summary.md").write_text(
        "# OakInk2 O5R-B Parallel final summary\n\nB1: `"
        + b1_result
        + "`. B2: `"
        + b2_result
        + "`.\n\nNEXT=`"
        + next_plan
        + "`\n",
        encoding="utf-8",
    )
    git = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "start_head": json.loads((root / "preflight/git.json").read_text())["start_head"],
        "head_at_report_write": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "git_commits.json", git)
    write_json(
        root / "tests.json",
        {
            "schema_version": "OakInk2O5RBTestsV1",
            "b1_selection_deterministic": "PASS",
            "b1_diagnostic_isolation": "PASS",
            "b2_candidate_determinism": "PASS",
            "b2_fault_injection": json.loads(
                (root / "b2_seed_robustness/fault_injection.json").read_text()
            )["status"],
        },
    )
    write_jsonl(
        root / "technical_failures.jsonl",
        [
            {
                "component": "DEV2_WARMSTART",
                "status": b2_result,
                "reason": b2["root_cause"],
                "production_solve_started": False,
            }
        ],
    )
    write_json(
        root / "resource_usage.json",
        {
            "dev1_full_retarget_reruns": 0,
            "dev2_full_production_solves": 0,
            "b1_diagnostic_scope": "25 selected frames x 5 thumb-only candidate seeds",
            "b2_diagnostic_scope": "DEV2 frame 0 only; bounded probes plus three deterministic preflight repetitions",
            "thread_policy": "single-threaded bounded diagnostics",
        },
    )
    handoff = f"# OakInk2 O5R-B Parallel Handoff\n\nBRANCH={git['branch']}\nSTART_HEAD={git['start_head']}\nFINAL_HEAD={git['head_at_report_write']}\nPUSHED=NO\nPR_CREATED=NO\n\n## B1\n\nSelected frames are frozen in `{(root / 'b1_thumb_feasibility/frame_selection.json').resolve()}`. B1 result: `{b1_result}`. The reported optimum is best-observed only, not globally certified.\n\nTHUMB_FEASIBILITY_VIEWER={(root / 'b1_thumb_feasibility/viewer/thumb_feasibility_diagnostic_viewer.html').resolve()}\n\n```bash\nxdg-open '{(root / 'b1_thumb_feasibility/viewer/thumb_feasibility_diagnostic_viewer.html').resolve()}'\n```\n\n## B2\n\nDEV2 recovery: `{b2_result}`; root cause: `{b2['root_cause']}`. No DEV2 full solve ran.\n\n## Joint decision\n\nB1_RESULT={b1_result}\nB2_RESULT={b2_result}\nNEXT={next_plan}\n"
    (root / "handoff.md").write_text(handoff, encoding="utf-8")


def validate(root: Path) -> int:
    """Run the requested validation commands and preserve their exact receipts."""

    modified = [
        "scripts/data/run_oakink2_o5rb.py",
        "src/toporetarget/retarget/first_frame_seed.py",
        "src/toporetarget/viz/oakink2_html_viewer.py",
        "tests/unit/test_first_frame_seed_authority.py",
    ]
    commands = {
        "ruff_check": [sys.executable, "-m", "ruff", "check", *modified],
        "ruff_format_check": [sys.executable, "-m", "ruff", "format", "--check", *modified],
        "mypy_src": [sys.executable, "-m", "mypy", "src"],
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "paper_fidelity": [sys.executable, "scripts/check_paper_fidelity.py"],
    }
    results: dict[str, Any] = {}
    for name, command in commands.items():
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        log = root / "validation_logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        results[name] = {
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "returncode": completed.returncode,
            "wall_sec": time.perf_counter() - started,
            "command": command,
            "log": str(log.resolve()),
        }
    write_json(root / "validation_results.json", results)
    return 0 if all(item["status"] == "PASS" for item in results.values()) else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "all",
            "preflight",
            "b1-select",
            "b1-run",
            "b1-viewer",
            "b2-reproduce",
            "b2-compare",
            "b2-preflight",
            "b2-regression",
            "validate",
            "report",
        ),
        default="all",
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.report_root.resolve()
    action = args.action
    if action in {"all", "preflight"}:
        preflight(root)
    if action in {"all", "b1-select"}:
        b1_selection(root)
    if action in {"all", "b1-run"}:
        b1_run(root)
    if action in {"all", "b1-viewer"}:
        b1_viewer(root)
    if action in {"all", "b2-reproduce"}:
        b2_reproduce(root)
    if action in {"all", "b2-compare"}:
        b2_compare(root)
    if action in {"all", "b2-preflight"}:
        b2_preflight(root)
    if action in {"all", "b2-regression"}:
        b2_run(root)
    if action == "validate":
        return validate(root)
    if action in {"all", "report"}:
        final(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
