#!/usr/bin/env python3
"""OakInk2 O5R-C structured full-objective solver certification.

The command is deliberately fail-closed.  It develops one generic,
asset-derived block schedule only on the frozen O5R-B B1 frames, freezes that
schedule and an independent DEV1 sparse set, then runs the sparse gate and the
known DEV2 frame-zero hard control.  A DEV2 240-frame command is refused until
both gates have passed.  DEV1 full reruns and full refinements are absent.
"""

# ruff: noqa: E402, E501, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5 import (
    EXECUTION_PROFILE,
    ROBOT,
    SOLVER_PROFILE,
    episode_paths,
)
from scripts.data.run_oakink2_o5 import (  # noqa: E402
    REPORT_ROOT as O5_ROOT,
)
from toporetarget.cli.retarget import _refinement_components  # noqa: E402
from toporetarget.retarget.final_refinement import (  # noqa: E402
    CollisionQueryProfile,
    RefinementCoordinateProfile,
    RefinementSolverProfile,
    _make_context,
    build_query_set,
    encode_base_correction,
    load_final_trajectory,
    map_previous_state_to_seed,
    prepare_refinement_resources,
    prepare_refinement_runtime_backends,
    refine_frame,
)
from toporetarget.retarget.interaction_objective import InteractionMeshResidual  # noqa: E402
from toporetarget.retarget.refinement_performance import RefinementExecutionProfile  # noqa: E402
from toporetarget.utils.hashing import sha256_file, sha256_tree  # noqa: E402

ROOT = REPO / ".local/reports/oakink2_o5rc_structured_solver_v1"
O5RA = REPO / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"
O5RB = REPO / ".local/reports/oakink2_o5rb_parallel_v1"
EPS = 1e-10
FINGERS = ("thumb", "index", "middle", "ring", "little")
KEYPOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "little": (17, 18, 19, 20),
}


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
        writer.writerows(jsonable(rows))


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_paths() -> dict[str, Path]:
    dev1, dev2 = episode_paths(O5_ROOT, "dev_01"), episode_paths(O5_ROOT, "dev_02")
    return {
        "manifest_v2": REPO
        / ".local/reports/oakink2_o1r_official_mano_authority_v1/manifest_v2/oakink2_corpus_manifest_v2.jsonl",
        "split_v2": REPO
        / ".local/reports/oakink2_o1r_official_mano_authority_v1/manifest_v2/oakink2_raw_to_physical_split_v2.json",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "v2_contract": O5_ROOT / "contract/geometric_retarget_contract.json",
        "dev1_final_trajectory": dev1["final"],
        "dev1_semantic_v1": dev1["semantic"],
        "dev1_timing": O5_ROOT / "timing/dev_01_frame_timing.csv",
        "o5ra_eim": O5RA / "dev1_semantic/dev1_e_im_per_frame.csv",
        "o5rb_b1_set": O5RB / "b1_thumb_feasibility/frame_selection.json",
        "o5rb_b1_results": O5RB / "b1_thumb_feasibility/frame_results.csv",
        "seed_authority_v2": O5RB / "b2_seed_robustness/first_frame_seed_authority_v2.json",
        "dev2_failure": O5RA / "dev2_warmstart/minimal_reproducer.json",
        "profiler_v1": O5RA / "profiling/solver_profiler_contract.json",
        "dev2_canonical": dev2["canonical"],
    }


def preflight(root: Path) -> None:
    entries = {
        name: {"path": str(path.resolve()), "sha256": digest(path)}
        for name, path in frozen_paths().items()
    }
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if branch != "feature/oakink2-raw-to-physical":
        raise RuntimeError("O5RC_BRANCH_AUTHORITY_MISMATCH")
    write_json(
        root / "preflight/git.json",
        {
            "branch": branch,
            "start_head": head,
            "status_short": subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=all"], cwd=REPO, text=True
            ).splitlines(),
            "pushed": False,
            "pr_created": False,
        },
    )
    write_json(
        root / "preflight/frozen_authorities.json",
        {
            "schema_version": "OakInk2O5RCFrozenAuthoritiesV1",
            "status": "PASS",
            "artifacts": entries,
            "immutable": [
                "Manifest V2",
                "Split V2",
                "raw MANO",
                "Semantic V1",
                "Wuji limits",
                "DEV1 old trajectory",
            ],
        },
    )
    write_json(
        root / "preflight/execution_authorization.json",
        {
            "DEV1_FROM_SCRATCH_FULL_RETARGET_RERUNS": 0,
            "DEV1_FULL_REFINEMENT_EXECUTED": "NO",
            "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
            "FULL_DEV2_COMPUTE_AUTHORIZED": "NO",
            "forbidden": ["O6", "SupportPhysicalization", "PhysX", "Frozen Eval", "PPO"],
        },
    )


def schedule_audit(root: Path) -> None:
    profile = RefinementSolverProfile.load(SOLVER_PROFILE)
    warm = REPO / "configs/retarget/warm_start/paper_repro_scipy_trf.yaml"
    write_json(
        root / "development/current_solver_schedule_audit.json",
        {
            "schema_version": "CurrentRetargetOptimizationScheduleV2Audit",
            "stages": [
                {
                    "stage": 1,
                    "name": "canonical MANO/frame authority",
                    "optimized_dofs": [],
                    "solver": None,
                    "entry": "Manifest V2 record",
                    "exit": "canonical HOI",
                },
                {
                    "stage": 2,
                    "name": "warm source bone features",
                    "optimized_dofs": [],
                    "solver": None,
                    "entry": "canonical keypoints",
                    "exit": "bone features",
                },
                {
                    "stage": 3,
                    "name": "Stage7 warm start",
                    "optimized_dofs": "all Wuji finger DOFs",
                    "fixed_dofs": [],
                    "objective": "BoneDirectionObjective Eq.1-2",
                    "bounds": "Wuji joint limits",
                    "solver": "scipy least_squares/trf",
                    "max_nfev": 250,
                    "warm_start": "neutral then previous frame",
                    "exit": "strict success; fail-fast",
                },
                {
                    "stage": 4,
                    "name": "interaction graph",
                    "optimized_dofs": [],
                    "solver": None,
                    "objective": "Eq.7 source graph",
                    "exit": "71-vertex graph",
                },
                {
                    "stage": 5,
                    "name": "active collision query selection",
                    "optimized_dofs": [],
                    "solver": None,
                    "bounds": "adaptive fixed source surface query",
                    "exit": "query set",
                },
                {
                    "stage": 6,
                    "name": "continuous initialization/transport",
                    "optimized_dofs": [],
                    "solver": None,
                    "warm_start": "previous accepted final transported to current warm seed",
                    "exit": "seed delta state",
                },
                {
                    "stage": 7,
                    "name": "production final refinement",
                    "optimized_dofs": "base correction + all 20 Wuji DOFs + query slack",
                    "fixed_dofs": [],
                    "objective": "unchanged full production objective: interaction, bone, temporal and slack",
                    "bounds": "production base/free, Wuji limits, slack",
                    "solver": profile.as_dict(),
                    "entry": "Stage7 seed and active set",
                    "exit": "strict SLSQP acceptance + independent full audit",
                    "fallback": "production deterministic continuity recovery only",
                },
            ],
            "source_locations": {
                "warm_solver": str(warm),
                "full_objective": "src/toporetarget/retarget/final_refinement.py:_FrameContext.objective",
                "full_polish": "src/toporetarget/retarget/final_refinement.py:refine_frame",
            },
        },
    )


def bundle() -> Any:
    return np.load(O5RA / "dev1_semantic/semantic_replay_bundle.npz", allow_pickle=False)


def b1_frames() -> list[dict[str, Any]]:
    return list(read_json(O5RB / "b1_thumb_feasibility/frame_selection.json")["frames"])


def freeze_sparse_set(root: Path) -> list[dict[str, Any]]:
    destination = root / "validation/dev1_sparse_validation_set.json"
    if destination.exists():
        return list(read_json(destination)["frames"])
    data = bundle()
    eim, raw = (
        np.asarray(data["interaction_final_e_im"], dtype=float),
        np.asarray(data["raw_frame_index"], dtype=int),
    )
    forbidden = {int(row["ordinal"]) for row in b1_frames()}
    allowed = np.asarray([index for index in range(len(eim)) if index not in forbidden], dtype=int)
    p40, p60 = np.quantile(eim, [0.4, 0.6])
    pools = {
        "HIGH": allowed[eim[allowed] > 1e-4],
        "MID": allowed[(eim[allowed] >= p40) & (eim[allowed] <= p60)],
        "LOW": allowed[eim[allowed] <= 1e-4],
    }
    rows: list[dict[str, Any]] = []
    for stratum in ("HIGH", "MID", "LOW"):
        pool = pools[stratum]
        if len(pool) < 10:
            raise RuntimeError(f"O5RC_SPARSE_SELECTION_INSUFFICIENT_{stratum}")
        # The frozen strata occupy different portions of this episode (the
        # LOW controls precede the HIGH tail), so global equal-width coverage
        # is infeasible without weakening a stratum.  Spread each stratum
        # over its available temporal support instead, before any new solve.
        ordered = pool[np.argsort(raw[pool], kind="stable")]
        picked = [int(ordered[index]) for index in np.linspace(0, len(ordered) - 1, 10, dtype=int)]
        for ordinal in picked:
            rows.append(
                {
                    "stratum": stratum,
                    "ordinal": int(ordinal),
                    "frame_id": int(raw[ordinal]),
                    "old_e_im": float(eim[ordinal]),
                    "temporal_bin": len([row for row in rows if row["stratum"] == stratum]),
                }
            )
    if (
        len(rows) != 30
        or len({row["ordinal"] for row in rows}) != 30
        or any(row["ordinal"] in forbidden for row in rows)
    ):
        raise RuntimeError("O5RC_SPARSE_SELECTION_INVALID")
    payload = {
        "schema_version": "DEV1SparseValidationSetV1",
        "selection_frozen_before_structured_solver_outcome": True,
        "selection_rule": "ten deterministic quantiles over each stratum's available temporal support; strata are frozen from old E_IM only",
        "temporal_coverage_note": "global equal-width bins are infeasible without breaking HIGH/MID/LOW definitions; each stratum is spread within its available time support",
        "frames": rows,
        "counts": {
            key: sum(row["stratum"] == key for row in rows) for key in ("HIGH", "MID", "LOW")
        },
        "overlap_with_b1_development": 0,
        "validation_reselected_after_outcome": False,
    }
    write_json(destination, payload)
    (root / "validation/dev1_sparse_validation_set.sha256").write_text(
        sha256_file(destination) + "\n", encoding="utf-8"
    )
    return rows


def block_map(model: Any) -> dict[str, list[int]]:
    names = list(model.dof_names)
    groups: dict[str, list[int]] = {"WRIST": [], **{name.upper(): [] for name in FINGERS}}
    aliases = {
        "thumb": ("thumb",),
        "index": ("index",),
        "middle": ("middle",),
        "ring": ("ring",),
        "little": ("little", "pinky"),
    }
    for idx, name in enumerate(names):
        lower = str(name).lower()
        matches = [
            finger for finger, tokens in aliases.items() if any(token in lower for token in tokens)
        ]
        if len(matches) != 1:
            raise RuntimeError(f"O5RC_ASSET_DOF_BLOCK_UNMAPPED:{name}")
        groups[matches[0].upper()].append(idx)
    if any(len(groups[name.upper()]) == 0 for name in FINGERS):
        raise RuntimeError("O5RC_ASSET_DOF_BLOCK_EMPTY")
    return groups


class EpisodeRuntime:
    def __init__(self, review: str, root: Path):
        paths = episode_paths(O5_ROOT, review)
        self.review, self.paths, self.root = review, paths, root
        self.final = load_final_trajectory(paths["final"]) if review == "dev_01" else None
        self.sequence, self.warm, self.graph, self.model, self.surface, _ = _refinement_components(
            paths["canonical"], paths["warm"], paths["graph"], ROBOT, None, None
        )
        self.solver = RefinementSolverProfile.load(SOLVER_PROFILE)
        self.execution = RefinementExecutionProfile.load(EXECUTION_PROFILE, REPO)
        self.query = CollisionQueryProfile.load("adaptive_active_set_v1")
        self.coordinate = RefinementCoordinateProfile.load("local_seed_delta_v1")
        self.resources = prepare_refinement_resources(
            self.sequence,
            self.graph,
            self.solver,
            geometry_artifact_root=root / "structured_solver/geometry",
        )
        self.backends = prepare_refinement_runtime_backends(self.resources, self.execution)
        self.blocks = block_map(self.model)

    def old_state(self, ordinal: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.final is None:
            raise RuntimeError("O5RC_DEV2_HAS_NO_OLD_PRODUCTION_TRAJECTORY")
        q = np.asarray(self.final.arrays["qpos"][ordinal], dtype=float)
        base = np.asarray(self.final.arrays["base_pose_scene"][ordinal], dtype=float)
        previous = None
        if ordinal > 0:
            previous = map_previous_state_to_seed(
                np.asarray(self.final.arrays["base_pose_scene"][ordinal - 1], dtype=float),
                np.asarray(self.final.arrays["qpos"][ordinal - 1], dtype=float),
                np.asarray(self.warm.arrays["base_pose_scene"][ordinal], dtype=float),
            )
        return q, base, previous

    def context(
        self,
        ordinal: int,
        q: np.ndarray,
        base: np.ndarray,
        previous: np.ndarray | None,
        *,
        free: tuple[int, ...] | None,
        fixed_base: bool,
    ) -> Any:
        value = _make_context(
            self.sequence,
            self.graph,
            self.warm,
            self.model,
            self.surface,
            self.backends.solver_sdf,
            self.resources.reference_sdf,
            __import__(
                "toporetarget.retarget.frames", fromlist=["load_frame_profile"]
            ).load_frame_profile("canonical_keypoint_wrist_v1"),
            __import__(
                "toporetarget.retarget.bones", fromlist=["load_bone_profile"]
            ).load_bone_profile("mediapipe21_full_finger_chain_v1"),
            self.resources.paper,
            ordinal,
            previous,
            temporal_scope="continuous_full_state",
            fixed_base_to_seed=fixed_base,
            spatial_gradient_backend=self.execution.signed_distance_gradient,
            sign_cache=self.backends.sign_cache,
            compiled_spatial_fd_backend=self.backends.compiled_spatial_fd_backend,
        )
        value.seed_qpos = np.asarray(q, dtype=float).copy()
        value.seed_base = np.asarray(base, dtype=float).copy()
        value.free_qpos_indices = free
        return value

    def query_set(self, context: Any, q: np.ndarray) -> Any:
        state = np.concatenate([np.zeros(6), q])
        result = self.backends.solver_sdf.query_scene(
            context.candidate_points(state), context.object_pose_scene
        )
        return build_query_set(result.signed_distance, self.surface.geometry_ids, self.query)

    def eim(self, ordinal: int, q: np.ndarray, base: np.ndarray) -> tuple[float, dict[str, float]]:
        import torch

        directed = self.graph.directed_frames[ordinal]
        residual = InteractionMeshResidual(
            self.graph.source_vertices[ordinal],
            directed.source_index,
            directed.destination_index,
            directed.weights,
        )
        hand = self.model.keypoints_scene(
            torch.as_tensor(q, dtype=torch.float64),
            torch.as_tensor(base, dtype=torch.float64),
            layout="mediapipe21",
        )
        vertices = torch.cat(
            [hand, torch.as_tensor(self.graph.source_vertices[ordinal, 21:], dtype=torch.float64)],
            dim=0,
        )
        per = residual(vertices).square().sum(dim=-1).detach().cpu().numpy() / 71.0
        return float(per.sum()), {
            finger: float(per[list(KEYPOINTS[finger])].sum()) for finger in FINGERS
        }


def block_refine(
    runtime: EpisodeRuntime,
    ordinal: int,
    q: np.ndarray,
    base: np.ndarray,
    previous: np.ndarray | None,
    block: str,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    free = tuple(runtime.blocks[block])
    context = runtime.context(ordinal, q, base, previous, free=free, fixed_base=True)
    query = runtime.query_set(context, q)
    result = refine_frame(
        context,
        query,
        replace(runtime.solver, maxiter=budget),
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery="none",
        initial_state_without_slack=np.concatenate([np.zeros(6), q]),
        initialization_source=f"structured_block_{block.lower()}_v1",
    )
    return (
        result.qpos,
        result.base_pose_scene,
        {
            "block": block,
            "success": bool(result.accepted),
            "termination": result.acceptance_reason,
            "nfev": int(result.optimizer_function_evaluations),
            "njev": int(result.optimizer_jacobian_evaluations),
            "runtime_sec": float(result.solve_time_s),
            "objective": float(result.final_objective),
            "profiler": result.jacobian_diagnostics,
        },
    )


def structured_frame(
    runtime: EpisodeRuntime,
    ordinal: int,
    q0: np.ndarray,
    base0: np.ndarray,
    previous: np.ndarray | None,
    design: dict[str, Any],
) -> dict[str, Any]:
    old_eim, scores = runtime.eim(ordinal, q0, base0)
    ranked = sorted(FINGERS, key=lambda finger: (-scores[finger], finger))
    selected = ranked[: int(design["top_k"])]
    q, base, stages = q0.copy(), base0.copy(), []
    for _ in range(int(design["block_passes"])):
        for finger in selected:
            q, base, receipt = block_refine(
                runtime, ordinal, q, base, previous, finger.upper(), int(design["block_maxiter"])
            )
            stages.append(receipt)
            if not receipt["success"]:
                return {
                    "success": False,
                    "qpos": q,
                    "base": base,
                    "old_e_im": old_eim,
                    "new_e_im": None,
                    "scores": scores,
                    "selected_blocks": selected,
                    "stages": stages,
                    "full_polish": None,
                }
    context = runtime.context(ordinal, q, base, previous, free=None, fixed_base=False)
    query = runtime.query_set(context, q)
    state = np.concatenate([encode_base_correction(base, base), q])
    polish = refine_frame(
        context,
        query,
        runtime.solver,
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery=runtime.execution.strict_recovery,
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=state,
        initialization_source="structured_blocks_then_original_full_polish_v1",
    )
    new_eim, new_scores = runtime.eim(ordinal, polish.qpos, polish.base_pose_scene)
    return {
        "success": bool(polish.accepted),
        "qpos": polish.qpos,
        "base": polish.base_pose_scene,
        "old_e_im": old_eim,
        "new_e_im": new_eim,
        "scores": scores,
        "new_scores": new_scores,
        "selected_blocks": selected,
        "stages": stages,
        "full_polish": {
            "success": bool(polish.accepted),
            "termination": polish.acceptance_reason,
            "nfev": int(polish.optimizer_function_evaluations),
            "njev": int(polish.optimizer_jacobian_evaluations),
            "runtime_sec": float(polish.solve_time_s),
            "objective": float(polish.final_objective),
            "initial_objective": float(polish.initial_objective),
            "profiler": polish.jacobian_diagnostics,
        },
    }


def development(root: Path) -> dict[str, Any]:
    target = root / "structured_solver/candidate_results.csv"
    if target.exists():
        return read_json(root / "structured_solver/selected_design.json")
    runtime = EpisodeRuntime("dev_01", root)
    frames = b1_frames()
    designs = [{"id": "top1_one_pass_b40", "top_k": 1, "block_passes": 1, "block_maxiter": 40}]
    write_json(
        root / "development/b1_development_set.json",
        {
            "schema_version": "O5RBB1StructuredSolverDevelopmentSetV1",
            "role": "STRUCTURED_SOLVER_DEVELOPMENT",
            "frames": frames,
            "count": len(frames),
        },
    )
    rows: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []
    for design in designs:
        outcomes = []
        for frame in frames:
            q, base, previous = runtime.old_state(int(frame["ordinal"]))
            out = structured_frame(runtime, int(frame["ordinal"]), q, base, previous, design)
            outcomes.append(out)
            rows.append(
                {
                    "design_id": design["id"],
                    "frame_id": frame["frame_id"],
                    "ordinal": frame["ordinal"],
                    "success": out["success"],
                    "old_e_im": out["old_e_im"],
                    "new_e_im": out["new_e_im"],
                    "selected_blocks": "+".join(out["selected_blocks"]),
                    "total_nfev": sum(int(stage["nfev"]) for stage in out["stages"])
                    + (0 if out["full_polish"] is None else int(out["full_polish"]["nfev"])),
                    "total_runtime_sec": sum(float(stage["runtime_sec"]) for stage in out["stages"])
                    + (
                        0.0
                        if out["full_polish"] is None
                        else float(out["full_polish"]["runtime_sec"])
                    ),
                }
            )
        successful = [out for out in outcomes if out["success"]]
        aggregate.append(
            {
                "design": design,
                "success_count": len(successful),
                "median_final_objective": float(
                    np.median([out["full_polish"]["objective"] for out in successful])
                )
                if successful
                else float("inf"),
                "median_eim": float(np.median([out["new_e_im"] for out in successful]))
                if successful
                else float("inf"),
            }
        )
    write_csv(target, rows)
    selected = min(
        aggregate,
        key=lambda row: (
            -int(row["success_count"]),
            float(row["median_final_objective"]),
            str(row["design"]["id"]),
        ),
    )
    write_json(
        root / "structured_solver/design_candidates.json",
        {
            "schema_version": "StructuredSolverCandidatesV1",
            "count": len(designs),
            "selection_rule": "highest B1 completion, then lowest final original objective, then lexical ID",
            "candidates": designs,
            "results": aggregate,
        },
    )
    write_json(
        root / "structured_solver/selected_design.json",
        {
            "schema_version": "WujiStructuredSequentialRefinementV1",
            "development_only": True,
            "selected": selected["design"],
            "selection": selected,
            "objective_changed": False,
            "weights_changed": False,
            "joint_limits_changed": False,
            "thumb_special_case": False,
            "dev2_special_case": False,
            "blocks": runtime.blocks,
        },
    )
    return read_json(root / "structured_solver/selected_design.json")


def freeze_contract(root: Path) -> dict[str, Any]:
    target = root / "structured_solver/frozen_structured_solver_contract.json"
    if target.exists():
        return read_json(target)
    selected = read_json(root / "structured_solver/selected_design.json")
    contract = {
        "schema_version": "StructuredSolverFrozenContractV1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "design": selected,
        "final_polish": RefinementSolverProfile.load(SOLVER_PROFILE).as_dict(),
        "objective_invariants": {
            "final_scientific_objective_changed": False,
            "weights_changed": False,
            "bounds_changed": False,
            "final_tolerance_changed": False,
            "final_maxiter_changed": False,
        },
        "block_budget_frozen": True,
        "seed_authority": "DEV1 old production q/base plus mapped previous production state; DEV2 FirstFrameSeedAuthorityV2 frame zero",
        "contract_mutable_after_validation_starts": False,
    }
    write_json(target, contract)
    (root / "structured_solver/frozen_structured_solver_contract.sha256").write_text(
        sha256_file(target) + "\n", encoding="utf-8"
    )
    write_json(
        root / "structured_solver/optimization_schedule_contract.json",
        {
            "initial_seed": contract["seed_authority"],
            "blocks": selected["blocks"],
            "contributor_selection": "exact Eq.7 per-hand-keypoint mass, top-K deterministic",
            "final_polish": "full production objective/all DOFs/original production bounds and tolerance",
        },
    )
    write_json(
        root / "structured_solver/interaction_contributor_refinement_contract.json",
        {
            "schema_version": "InteractionContributorRefinementV1",
            "additivity": "exact Eq.7 vertex-mass grouping",
            "mapping": KEYPOINTS,
            "selection": "descending contributor mass then semantic name",
            "no_finger_special_case": True,
        },
    )
    return contract


def sparse_validation(root: Path) -> dict[str, Any]:
    target = root / "validation/sparse_gate_decision.json"
    if target.exists():
        return read_json(target)
    frames, design = (
        freeze_sparse_set(root),
        read_json(root / "structured_solver/frozen_structured_solver_contract.json")["design"][
            "selected"
        ],
    )
    runtime = EpisodeRuntime("dev_01", root)
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for frame in frames:
        q, base, previous = runtime.old_state(int(frame["ordinal"]))
        out = structured_frame(runtime, int(frame["ordinal"]), q, base, previous, design)
        outcomes.append(out)
        full = out["full_polish"] or {}
        rows.append(
            {
                "frame_id": frame["frame_id"],
                "ordinal": frame["ordinal"],
                "stratum": frame["stratum"],
                "old_e_im": out["old_e_im"],
                "new_e_im": out["new_e_im"],
                "relative_reduction": None
                if out["new_e_im"] is None
                else (out["old_e_im"] - out["new_e_im"]) / max(out["old_e_im"], EPS),
                "success": out["success"],
                "selected_blocks": "+".join(out["selected_blocks"]),
                "original_obj_old": float(runtime.final.arrays["final_objective"][int(frame["ordinal"])]),
                "original_obj_new": full.get("objective"),
                "termination": full.get("termination"),
            }
        )
    write_csv(root / "validation/per_frame_results.csv", rows)
    highs = [row for row in rows if float(row["old_e_im"]) > 1e-4]
    low = [row for row in rows if row["stratum"] == "LOW"]
    finite_success = all(
        bool(row["success"]) and row["new_e_im"] is not None and np.isfinite(float(row["new_e_im"]))
        for row in rows
    )
    recovered = sum(float(row["new_e_im"]) <= 1e-4 for row in highs) / max(1, len(highs))
    reductions = [
        (float(row["old_e_im"]) - float(row["new_e_im"])) / max(float(row["old_e_im"]), EPS)
        for row in highs
        if row["new_e_im"] is not None
    ]
    nonreg = sum(
        float(row["new_e_im"]) <= float(row["old_e_im"]) + EPS
        for row in rows
        if row["new_e_im"] is not None
    ) / len(rows)
    catastrophic = sum(
        float(row["new_e_im"]) > 1.25 * float(row["old_e_im"])
        for row in rows
        if row["new_e_im"] is not None
    )
    low_ok = all(row["new_e_im"] is not None and float(row["new_e_im"]) <= 1e-4 for row in low)
    objective_ok = all(
        row["original_obj_new"] is not None
        and float(row["original_obj_new"]) <= float(row["original_obj_old"]) + EPS
        for row in rows
    )
    status = (
        "PASS"
        if finite_success
        and recovered >= 0.8
        and reductions
        and float(np.median(reductions)) >= 0.5
        and nonreg >= 0.9
        and catastrophic == 0
        and low_ok
        and objective_ok
        else "FAIL"
    )
    metrics = {
        "technical_success": finite_success,
        "above_threshold_recovery_rate": recovered,
        "median_relative_eim_reduction": None if not reductions else float(np.median(reductions)),
        "all_frame_nonregression_rate": nonreg,
        "catastrophic_regressions": catastrophic,
        "low_controls_retained": low_ok,
        "objective_nonregression": objective_ok,
        "semantic_nonregression": "NOT_RUN: gate is already fail-closed if technical/metric requirements fail",
    }
    write_json(root / "validation/gate_metrics.json", metrics)
    write_json(
        root / "validation/determinism.json",
        {
            "status": "NOT_RUN",
            "reason": "only permitted after the frozen schedule completes all sparse frames; never substitutes for a failed sparse gate",
        },
    )
    decision = {
        "STRUCTURED_SOLVER_SPARSE_GATE": status,
        "metrics": metrics,
        "validation_count": len(rows),
        "overlap_with_b1_development": 0,
    }
    write_json(target, decision)
    return decision


def dev2_frame0(root: Path) -> dict[str, Any]:
    target = root / "dev2_frame0/hard_control_decision.json"
    if target.exists():
        return read_json(target)
    sparse = read_json(root / "validation/sparse_gate_decision.json")
    if sparse["STRUCTURED_SOLVER_SPARSE_GATE"] != "PASS":
        decision = {
            "DEV2_FRAME0_HARD_CONTROL": "NOT_RUN",
            "reason": "SparseGate failed; frozen contract prevents outcome-driven DEV2 tuning",
        }
        write_json(target, decision)
        return decision
    # DEV2 has no accepted legacy final output.  Its established hard failure
    # is Stage-7 before full refinement; no structured full-objective control
    # may be fabricated without a production-consistent Stage-7 completion.
    old = read_json(O5RA / "dev2_warmstart/original_failure_receipt.json")
    write_json(root / "dev2_frame0/old_failure_reference.json", old)
    decision = {
        "DEV2_FRAME0_HARD_CONTROL": "FAIL",
        "OLD_RESULT": "MAX_NFEV_FAILURE",
        "NEW_RESULT": "NOT_RUN",
        "reason": "Structured full-objective schedule requires a valid Stage-7 seed; the frozen DEV2 frame-zero Stage-7 failure remains terminal and SeedAuthorityV2 was insufficient. No seed-only or DEV2-special path is permitted.",
    }
    write_json(target, decision)
    return decision


def full_dev2(root: Path) -> dict[str, Any]:
    sparse = read_json(root / "validation/sparse_gate_decision.json")
    hard = read_json(root / "dev2_frame0/hard_control_decision.json")
    allowed = (
        sparse["STRUCTURED_SOLVER_SPARSE_GATE"] == "PASS"
        and hard["DEV2_FRAME0_HARD_CONTROL"] == "PASS"
    )
    payload = {
        "FULL_DEV2_COMPUTE_AUTHORIZED": "YES" if allowed else "NO",
        "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0,
        "reason": "both frozen gates must PASS"
        if not allowed
        else "implementation intentionally requires an explicit separately monitored production invocation",
    }
    write_json(root / "dev2_full/authorization.json", payload)
    if not allowed:
        write_json(root / "dev2_full/not_run.json", payload)
    return payload


def report(root: Path) -> None:
    sparse = (
        read_json(root / "validation/sparse_gate_decision.json")
        if (root / "validation/sparse_gate_decision.json").exists()
        else {"STRUCTURED_SOLVER_SPARSE_GATE": "NOT_RUN"}
    )
    hard = (
        read_json(root / "dev2_frame0/hard_control_decision.json")
        if (root / "dev2_frame0/hard_control_decision.json").exists()
        else {"DEV2_FRAME0_HARD_CONTROL": "NOT_RUN"}
    )
    full = (
        read_json(root / "dev2_full/authorization.json")
        if (root / "dev2_full/authorization.json").exists()
        else {"FULL_DEV2_COMPUTE_AUTHORIZED": "NO", "DEV2_FULL_PRODUCTION_SOLVE_COUNT": 0}
    )
    flags = {
        "BRANCH": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True
        ).strip(),
        "STRUCTURED_SOLVER_VERSIONED": "YES",
        "OLD_V2_SOLVER_OVERWRITTEN": "NO",
        "FINAL_OBJECTIVE_CHANGED": "NO",
        "OBJECTIVE_WEIGHTS_CHANGED": "NO",
        "JOINT_LIMITS_CHANGED": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "FULL_POLISH_MAX_NFEV_CHANGED": "NO",
        "THUMB_SPECIAL_CASE_ADDED": "NO",
        "DEV2_SPECIAL_CASE_ADDED": "NO",
        "B1_DEVELOPMENT_FRAME_COUNT": 25,
        "DEV1_SPARSE_VALIDATION_FROZEN": "YES",
        "DEV1_SPARSE_VALIDATION_COUNT": 30,
        "B1_VALIDATION_OVERLAP": 0,
        "VALIDATION_RESELECTED_AFTER_OUTCOME": "NO",
        **sparse,
        **hard,
        **full,
        "DEV1_FROM_SCRATCH_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_REFINEMENT_EXECUTED": "NO",
        "RETARGET_SOLVER_PROFILER_USED": "YES",
        "MANIFEST_V2_MODIFIED": "NO",
        "SPLIT_V2_MODIFIED": "NO",
        "CERTIFICATION_DOWNSTREAM_CONSUMED": "NO",
        "HELDOUT_DOWNSTREAM_CONSUMED": "NO",
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
    status = (
        "STRUCTURED_SOLVER_VALIDATION_FAIL"
        if sparse["STRUCTURED_SOLVER_SPARSE_GATE"] == "FAIL"
        else "DEV2_HARD_CONTROL_FAIL"
        if hard["DEV2_FRAME0_HARD_CONTROL"] != "PASS"
        else "INCOMPLETE"
    )
    write_json(
        root / "final_summary.json",
        {"schema_version": "OakInk2O5RCFinalSummaryV1", "o5_status": status, "safety_flags": flags},
    )
    (root / "final_summary.md").write_text(
        f"# OakInk2 O5R-C Structured Solver Robustness Handoff\n\nO5_STATUS={status}\n\nSTRUCTURED_SOLVER_SPARSE_GATE={sparse['STRUCTURED_SOLVER_SPARSE_GATE']}\nDEV2_FRAME0_HARD_CONTROL={hard['DEV2_FRAME0_HARD_CONTROL']}\nFULL_DEV2_COMPUTE_AUTHORIZED={full['FULL_DEV2_COMPUTE_AUTHORIZED']}\nDEV2_FULL_PRODUCTION_SOLVE_COUNT={full['DEV2_FULL_PRODUCTION_SOLVE_COUNT']}\n",
        encoding="utf-8",
    )
    (root / "handoff.md").write_text(
        (root / "final_summary.md").read_text(encoding="utf-8")
        + "\nDEV1_FULL_REFINEMENT_EXECUTED=NO\nPUSHED=NO\nPR_CREATED=NO\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "all",
            "preflight",
            "audit",
            "freeze-validation",
            "develop",
            "freeze-solver",
            "sparse-validation",
            "dev2-frame0",
            "dev2-full",
            "report",
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
    if action in {"all", "audit"}:
        schedule_audit(root)
    if action in {"all", "freeze-validation"}:
        freeze_sparse_set(root)
    if action in {"all", "develop"}:
        development(root)
    if action in {"all", "freeze-solver"}:
        freeze_contract(root)
    if action in {"all", "sparse-validation"}:
        sparse_validation(root)
    if action in {"all", "dev2-frame0"}:
        dev2_frame0(root)
    if action in {"all", "dev2-full"}:
        full_dev2(root)
    if action in {"all", "report"}:
        report(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
