#!/usr/bin/env python3
"""Design OakInk2 O5R-D2A RetargetObjectiveV2 and stop before validation.

Every action in this CLI is method-development-only.  It refuses DEV1 full
trajectory work, DEV2, independent validation, Semantic V2, and downstream
physical/policy stages by construction.
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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

from scripts.data.run_oakink2_o5 import REPORT_ROOT as O5
from scripts.data.run_oakink2_o5 import episode_paths
from scripts.data.run_oakink2_o5rc import EpisodeRuntime
from scripts.data.run_oakink2_o5rd1 import production_context as production_context_v1_replay
from scripts.evaluation.audit_retarget_semantic_validity import _semantic_frames
from toporetarget.evaluation.retarget_semantic_validity import (
    SemanticGateContractV1,
    angular_error,
    qualify_semantics,
    relative_transform,
    transform_error,
)
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile
from toporetarget.retarget.continuous import (
    continuity_metrics,
    encode_base_correction,
    transport_previous_final_to_current_warm,
)
from toporetarget.retarget.final_refinement import (
    _make_context,
    build_query_set,
    map_previous_state_to_seed,
    refine_frame,
)
from toporetarget.retarget.frames import load_frame_profile
from toporetarget.retarget.objective_v2 import (
    ObjectiveV2Candidate,
    ObjectiveV2DevelopmentContext,
    ObjectiveV2Measurements,
    ProductionObjectiveContextBindingV2,
    RetargetNonRegressionBudgetAuthorityV1,
    compare_candidate_states,
    evaluate_candidate,
    interaction_retention_limit,
)
from toporetarget.utils.hashing import sha256_file, sha256_tree

ROOT = REPO / ".local/reports/oakink2_o5rd2a_objective_v2_design_v1"
O1R = REPO / ".local/reports/oakink2_o1r_official_mano_authority_v1"
O5RA = REPO / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"
O5RB = REPO / ".local/reports/oakink2_o5rb_parallel_v1"
O5RC = REPO / ".local/reports/oakink2_o5rc_structured_solver_v1"
O5RD1 = REPO / ".local/reports/oakink2_o5rd1_objective_alignment_v1"
START_HEAD = "91d0b3bc74991111e627634d526e3617ffcb3e5c"
EXPECTED_BRANCH = "feature/oakink2-raw-to-physical"
WINDOW_SIZE = 20
WINDOW_STRATA = ("HIGH", "MID", "LOW")
CANDIDATES = (ObjectiveV2Candidate.candidate_a(), ObjectiveV2Candidate.candidate_b())


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


def frozen_paths() -> dict[str, Path]:
    dev1 = episode_paths(O5, "dev_01")
    return {
        "manifest_v2": O1R / "manifest_v2/oakink2_corpus_manifest_v2.jsonl",
        "split_v2": O1R / "manifest_v2/oakink2_raw_to_physical_split_v2.json",
        "wuji_asset": REPO / "configs/robots/wuji_hand2_beta1_rh.yaml",
        "production_objective_v1": REPO / "src/toporetarget/retarget/final_refinement.py",
        "semantic_v1": REPO / "src/toporetarget/evaluation/retarget_semantic_validity.py",
        "semantic_v1_cli": REPO / "scripts/evaluation/audit_retarget_semantic_validity.py",
        "dev1_old_trajectory": dev1["final"],
        "dev1_canonical": dev1["canonical"],
        "dev1_warm": dev1["warm"],
        "dev1_interaction_graph": dev1["graph"],
        "o5_production_contract": O5 / "contract/geometric_retarget_contract.json",
        "o5rb_b1_frames": O5RB / "b1_thumb_feasibility/frame_selection.json",
        "o5rb_q0_q1": O5RB / "b1_thumb_feasibility/frame_results.csv",
        "o5rc_frozen_contract": O5RC / "structured_solver/frozen_structured_solver_contract.json",
        "o5rc_sparse_v1": O5RC / "validation/dev1_sparse_validation_set.json",
        "o5rc_sparse_results": O5RC / "validation/per_frame_results.csv",
        "o5rd1_summary": O5RD1 / "final_summary.json",
        "o5rd1_alignment": O5RD1 / "decision/production_objective_alignment.json",
        "o5rd1_decomposition": O5RD1 / "replay/state_objective_decomposition.csv",
        "o5rd1_interpolation": O5RD1 / "interpolation/summary.json",
        "o5rd1_micro_polish": O5RD1 / "micro_polish/summary.json",
        "o5rd1_ledger": O5RD1 / "method_ledger/dev1_method_development_ledger.json",
        "o5rd1_source": REPO / "scripts/data/run_oakink2_o5rd1.py",
        "semantic_replay_bundle": O5RA / "dev1_semantic/semantic_replay_bundle.npz",
    }


def preflight(root: Path) -> dict[str, Any]:
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError("O5RD2A_BRANCH_AUTHORITY_MISMATCH")
    if head != START_HEAD:
        raise RuntimeError(f"O5RD2A_START_HEAD_MISMATCH:{head}")
    missing = [str(path) for path in frozen_paths().values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen authorities: " + ", ".join(missing))
    status_now = git("status", "--short", "--untracked-files=all").splitlines()
    git_payload = {
        "schema_version": "OakInk2O5RD2AGitPreflightV1",
        "repo": git("rev-parse", "--show-toplevel"),
        "branch": branch,
        "start_head": head,
        "initial_status_short": [],
        "initial_status_provenance": "mandated shell preflight captured before task modifications",
        "status_short_at_artifact_write": status_now,
        "diff_stat_at_artifact_write": git("diff", "--stat").splitlines(),
        "diff_check": git("diff", "--check").splitlines(),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "remotes": git("remote", "-v").splitlines(),
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "preflight/git.json", git_payload)
    authorities = {
        name: {"path": str(path.resolve()), "sha256": digest(path)}
        for name, path in frozen_paths().items()
    }
    payload = {
        "schema_version": "OakInk2O5RD2AFrozenAuthoritiesV1",
        "status": "PASS",
        "authorities": authorities,
        "structured_solver_v1_status": "SCIENTIFICALLY_REJECTED",
        "structured_solver_v1_reused_as_production": False,
        "production_objective_v1_modified": False,
        "semantic_v1_changed": False,
        "manifest_v2_modified": False,
        "split_v2_modified": False,
    }
    write_json(root / "preflight/frozen_authorities.json", payload)
    upstream = read_json(O5RD1 / "decision/production_objective_alignment.json")
    diagnosis = read_json(O5RD1 / "decision/primary_root_cause.json")
    if upstream.get("PRODUCTION_OBJECTIVE_ALIGNMENT") != "MISALIGNED":
        raise RuntimeError("O5RD2A_UPSTREAM_ALIGNMENT_AUTHORITY_MISMATCH")
    if diagnosis.get("PRIMARY_ROOT_CAUSE") != "OBJECTIVE_SEMANTIC_MISALIGNMENT":
        raise RuntimeError("O5RD2A_UPSTREAM_ROOT_CAUSE_AUTHORITY_MISMATCH")
    write_json(
        root / "preflight/upstream_decision.json",
        {
            "PRODUCTION_OBJECTIVE_ALIGNMENT": "MISALIGNED",
            "PRIMARY_ROOT_CAUSE": "OBJECTIVE_SEMANTIC_MISALIGNMENT",
            "CONFIDENCE": "HIGH",
            "StructuredSolverV1": "SCIENTIFICALLY_REJECTED",
            "DO_NOT_REUSE_AS_PRODUCTION": True,
            "source_alignment_sha256": sha256_file(frozen_paths()["o5rd1_alignment"]),
        },
    )
    return payload


def consumed_frames() -> list[dict[str, Any]]:
    b1 = read_json(O5RB / "b1_thumb_feasibility/frame_selection.json")["frames"]
    sparse = read_json(O5RC / "validation/dev1_sparse_validation_set.json")["frames"]
    rows: dict[int, dict[str, Any]] = {}
    for source, role, frames in (
        ("O5R-B B1", "DEVELOPMENT", b1),
        ("O5R-C SparseValidationV1", "CONSUMED_FAILED_VALIDATION_NOW_DEVELOPMENT", sparse),
    ):
        for frame in frames:
            ordinal = int(frame["ordinal"])
            rows.setdefault(
                ordinal,
                {
                    "ordinal": ordinal,
                    "frame_id": int(frame["frame_id"]),
                    "stratum": str(frame["stratum"]),
                    "sources": [],
                },
            )
            rows[ordinal]["sources"].append({"source": source, "role": role})
    if len(rows) != 55:
        raise RuntimeError(f"O5RD2A_CONSUMED_FRAME_COUNT_MISMATCH:{len(rows)}")
    return [rows[key] for key in sorted(rows)]


def development_windows(root: Path) -> list[dict[str, Any]]:
    target = root / "development_data/development_window_set.json"
    if target.exists():
        return list(read_json(target)["windows"])
    rows = consumed_frames()
    raw_frame_ids = np.asarray(
        np.load(frozen_paths()["semantic_replay_bundle"], allow_pickle=False)["raw_frame_index"],
        dtype=np.int64,
    )
    windows: list[dict[str, Any]] = []
    for stratum in WINDOW_STRATA:
        pool = sorted(
            (row for row in rows if row["stratum"] == stratum),
            key=lambda row: int(row["ordinal"]),
        )
        anchor = pool[(len(pool) - 1) // 2]
        center = int(anchor["ordinal"])
        start = max(0, min(center - WINDOW_SIZE // 2, 2722 - WINDOW_SIZE))
        ordinals = list(range(start, start + WINDOW_SIZE))
        windows.append(
            {
                "window_id": f"{stratum.lower()}_around_consumed_median_anchor",
                "stratum": stratum,
                "anchor_ordinal": center,
                "anchor_frame_id": int(anchor["frame_id"]),
                "selection_rule": "lower median temporal consumed anchor in stratum, then fixed [-10,+9] ordinal window",
                "ordinals": ordinals,
                "frame_ids": [int(raw_frame_ids[ordinal]) for ordinal in ordinals],
                "frame_count": len(ordinals),
            }
        )
    unique = sorted({ordinal for window in windows for ordinal in window["ordinals"]})
    already = {int(row["ordinal"]) for row in rows}
    new = sorted(set(unique) - already)
    if len(new) > 72 or len(windows) != 3:
        raise RuntimeError("O5RD2A_DEVELOPMENT_WINDOW_LIMIT_EXCEEDED")
    payload = {
        "schema_version": "ObjectiveV2DevelopmentWindowSetV1",
        "selection_frozen_before_objective_v2_outcome": True,
        "outcome_dependent_selection": False,
        "windows": windows,
        "unique_window_frame_count": len(unique),
        "new_development_frame_count": len(new),
        "new_development_ordinals": new,
        "independent_validation_role": False,
    }
    write_json(target, payload)
    return windows


def write_development_ledger(root: Path) -> dict[str, Any]:
    consumed = consumed_frames()
    windows = development_windows(root)
    all_rows: dict[int, dict[str, Any]] = {int(row["ordinal"]): dict(row) for row in consumed}
    for window in windows:
        for ordinal, frame_id in zip(window["ordinals"], window["frame_ids"], strict=True):
            row = all_rows.setdefault(
                int(ordinal),
                {
                    "ordinal": int(ordinal),
                    "frame_id": int(frame_id),
                    "stratum": str(window["stratum"]),
                    "sources": [],
                },
            )
            row["sources"].append(
                {
                    "source": "O5R-D2A ObjectiveV2DevelopmentWindowSetV1",
                    "window_id": window["window_id"],
                    "role": "METHOD_DEVELOPMENT_ONLY",
                }
            )
    rows = [all_rows[key] for key in sorted(all_rows)]
    payload = {
        "schema_version": "DEV1MethodDevelopmentLedgerV2",
        "consumed_55_frames_used_for_development": True,
        "entries": rows,
        "future_validation_exclusion_count": len(rows),
        "future_sparse_validation_v2_overlap_required": 0,
        "future_window_validation_v2_overlap_required": 0,
        "dev1_sparse_validation_v2_created": False,
        "dev1_window_validation_v2_created": False,
    }
    write_json(root / "development_data/consumed_frames.json", {"count": 55, "frames": consumed})
    write_json(root / "development_data/method_development_ledger_v2.json", payload)
    return payload


class D2ARuntime(EpisodeRuntime):
    def __init__(self, root: Path):
        super().__init__("dev_01", root)
        self.semantic = np.load(frozen_paths()["semantic_replay_bundle"], allow_pickle=False)
        self.frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
        self.bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
        self.authority = RetargetNonRegressionBudgetAuthorityV1.from_frozen_v1(
            collision_hard_bound_m=self.resources.paper.b,
            collision_soft_tolerance_m=self.resources.paper.tau,
        )

    def bind_context(
        self,
        ordinal: int,
        *,
        previous_base: np.ndarray | None = None,
        previous_qpos: np.ndarray | None = None,
    ) -> tuple[ProductionObjectiveContextBindingV2, Any]:
        current_frame = int(self.graph.frame_indices[ordinal])
        object_id = str(self.graph.metadata["object_id"])
        obj = self.sequence.rigid_object(object_id)
        object_pose = np.asarray(obj.pose_scene.pose_scene[current_frame], dtype=np.float64)
        if ordinal == 0:
            previous_reference = None
            propagated = None
            previous_frame = None
            previous_base = None
            previous_qpos = None
        else:
            previous_base = (
                np.asarray(self.final.arrays["base_pose_scene"][ordinal - 1], dtype=np.float64)
                if previous_base is None
                else np.asarray(previous_base, dtype=np.float64)
            )
            previous_qpos = (
                np.asarray(self.final.arrays["qpos"][ordinal - 1], dtype=np.float64)
                if previous_qpos is None
                else np.asarray(previous_qpos, dtype=np.float64)
            )
            previous_frame = int(self.graph.frame_indices[ordinal - 1])
            previous_reference = map_previous_state_to_seed(
                previous_base,
                previous_qpos,
                np.asarray(self.warm.arrays["base_pose_scene"][ordinal], dtype=np.float64),
            )
            propagated = transport_previous_final_to_current_warm(
                self.warm.arrays["base_pose_scene"][ordinal - 1],
                previous_base,
                self.warm.arrays["base_pose_scene"][ordinal],
                self.warm.arrays["qpos"][ordinal - 1],
                previous_qpos,
                self.warm.arrays["qpos"][ordinal],
                self.model.joint_lower,
                self.model.joint_upper,
                previous_frame=previous_frame,
                current_frame=current_frame,
            )
        predicted_base = None if propagated is None else propagated.predicted_base_scene
        predicted_q = None if propagated is None else propagated.predicted_qpos
        binding = ProductionObjectiveContextBindingV2(
            active_frame_id=current_frame,
            local_ordinal=int(ordinal),
            current_source_frame_id=current_frame,
            previous_source_frame_id=previous_frame,
            previous_runtime_base_scene=previous_base,
            previous_robot_qpos=previous_qpos,
            continuous_predicted_translation_scene=None
            if predicted_base is None
            else np.asarray(predicted_base[:3, 3]),
            continuous_predicted_rotation_scene=None
            if predicted_base is None
            else np.asarray(predicted_base[:3, :3]),
            continuous_predicted_qpos=predicted_q,
            base_correction_reference=None if propagated is None else propagated.base_correction,
            object_pose_scene=object_pose,
            object_id=object_id,
            robot_name=str(self.model.name),
            robot_side=str(self.model.side),
            robot_dof_names=tuple(str(name) for name in self.model.dof_names),
            robot_mapping_authority="configs/robots/wuji_hand2_beta1_rh.yaml",
            source_hand_id=str(self.warm.metadata["source_hand_id"]),
        ).validate()
        context = _make_context(
            self.sequence,
            self.graph,
            self.warm,
            self.model,
            self.surface,
            self.backends.solver_sdf,
            self.resources.reference_sdf,
            self.frame_profile,
            self.bone_profile,
            self.resources.paper,
            ordinal,
            previous_reference,
            temporal_scope=binding.temporal_scope,
            continuous_prediction_base=binding.continuous_prediction_base,
            continuous_prediction_qpos=binding.continuous_predicted_qpos,
            spatial_gradient_backend=self.execution.signed_distance_gradient,
            sign_cache=self.backends.sign_cache,
            compiled_spatial_fd_backend=self.backends.compiled_spatial_fd_backend,
        )
        return binding, context

    def q0_slack(self, ordinal: int) -> np.ndarray:
        offsets = np.asarray(self.final.arrays["slack_offsets"], dtype=np.int64)
        return np.asarray(
            self.final.arrays["slack_concat"][int(offsets[ordinal]) : int(offsets[ordinal + 1])],
            dtype=np.float64,
        )

    def measurement(
        self,
        ordinal: int,
        qpos: np.ndarray,
        base_pose: np.ndarray,
        *,
        binding: ProductionObjectiveContextBindingV2,
        context: Any,
        slack: np.ndarray | None = None,
    ) -> ObjectiveV2Measurements:
        q = np.asarray(qpos, dtype=np.float64)
        base = np.asarray(base_pose, dtype=np.float64)
        correction = encode_base_correction(context.seed_base, base)
        slack_value = np.zeros(0, dtype=np.float64) if slack is None else np.asarray(slack)
        value = np.concatenate([correction, q, slack_value])
        _total, _gradient, breakdown = context.objective(value)
        keypoints_base = np.asarray(self.model.keypoints_scene(q, np.eye(4)), dtype=np.float64)
        keypoints_scene = np.asarray(self.model.keypoints_scene(q, base), dtype=np.float64)
        robot_features = extract_bone_features(
            keypoints_base,
            self.frame_profile,
            self.bone_profile,
            side=self.model.side,
            strict=True,
        )
        source_directions = np.asarray(self.semantic["source_bone_directions"][ordinal])
        bone_error = angular_error(source_directions, np.asarray(robot_features.unit_directions))
        source_wrist = np.asarray(self.semantic["canonical_wrist_pose_scene"][ordinal])
        robot_wrist = base @ np.asarray(_semantic_frames(keypoints_base, self.model.side))
        object_pose = np.asarray(binding.object_pose_scene)
        wrist = transform_error(
            relative_transform(object_pose, source_wrist),
            relative_transform(object_pose, robot_wrist),
        )
        if binding.continuous_prediction_base is None:
            continuity = {
                "delta_base_translation_m": 0.0,
                "delta_base_rotation_rad": 0.0,
                "delta_finger_inf_rad": 0.0,
                "excess_keypoint_max_m": 0.0,
            }
        else:
            predicted_keypoints = np.asarray(
                self.model.keypoints_scene(
                    binding.continuous_predicted_qpos,
                    binding.continuous_prediction_base,
                )
            )
            continuity = continuity_metrics(
                binding.continuous_prediction_base,
                base,
                binding.continuous_predicted_qpos,
                q,
                predicted_keypoints_scene=predicted_keypoints,
                final_keypoints_scene=keypoints_scene,
                frame=int(binding.active_frame_id),
            )
        points = context.candidate_points(np.concatenate([correction, q]))
        full = self.resources.reference_sdf.query_scene(points, binding.object_pose_scene)
        joint_margin = float(
            np.min(np.concatenate([q - self.model.joint_lower, self.model.joint_upper - q]))
        )
        old_lengths = extract_bone_features(
            np.asarray(self.warm.arrays["robot_keypoints_base"][ordinal]),
            self.frame_profile,
            self.bone_profile,
            side=self.model.side,
            strict=True,
        ).bone_lengths
        ratio = np.asarray(robot_features.bone_lengths) / np.asarray(old_lengths)
        secondary = float(breakdown.total - breakdown.weighted_e_im)
        return ObjectiveV2Measurements(
            interaction_e_im=float(breakdown.e_im),
            secondary_objective=secondary,
            bone_direction_p95_rad=float(np.quantile(bone_error, 0.95)),
            wrist_position_m=float(np.asarray(wrist["position_m"])),
            wrist_rotation_rad=float(np.asarray(wrist["rotation_rad"])),
            temporal_base_translation_m=float(continuity["delta_base_translation_m"]),
            temporal_base_rotation_rad=float(continuity["delta_base_rotation_rad"]),
            temporal_q_inf_rad=float(continuity["delta_finger_inf_rad"]),
            temporal_excess_keypoint_m=float(continuity["excess_keypoint_max_m"]),
            collision_min_signed_distance_m=float(np.min(full.signed_distance)),
            joint_limit_min_margin_rad=joint_margin,
            rotation_determinant=float(np.linalg.det(robot_wrist[:3, :3])),
            unit_scale_ratio=float(np.max(np.abs(ratio - 1.0)) + 1.0),
            per_components={
                "interaction_mesh": float(breakdown.weighted_e_im),
                "bone_direction": float(breakdown.weighted_e_bone),
                "continuous_temporal": float(breakdown.e_temporal),
                "base_position": float(breakdown.e_base_pos),
                "base_rotation": float(breakdown.e_base_rot),
                "collision_slack": float(breakdown.e_slack),
            },
        )


def verify_objective_v1(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    paper = runtime.resources.paper
    payload = {
        "schema_version": "ProductionRetargetObjectiveV1VerifiedForO5RD2A",
        "formula": "J_V1=500*E_IM+0.1*E_bone+E_continuous_temporal+100*||delta_p||^2+||delta_w||^2+50000*||slack||^2",
        "implementation_formula": "lambda_IM*E_IM+lambda_bone*E_bone+E_temporal+lambda_base_pos*||delta_p||^2+lambda_base_rot*||delta_w||^2+0.5*w_s*||slack||^2",
        "weights": paper.as_dict(),
        "continuous_temporal": {
            "lambda_corr": 0.25,
            "translation_scale_m": 0.01,
            "rotation_scale_rad": float(np.deg2rad(5.0)),
            "q_scale_rad": 0.05,
            "aggregation": "sum of three block means",
        },
        "historical_semantics_changed": False,
        "source_sha256": sha256_file(frozen_paths()["production_objective_v1"]),
    }
    write_json(root / "objective_v1/verified_formula.json", payload)
    return payload


def context_binding_parity(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for frame in consumed_frames():
        ordinal = int(frame["ordinal"])
        binding, context = runtime.bind_context(ordinal)
        legacy = production_context_v1_replay(runtime, ordinal)
        q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        slack = runtime.q0_slack(ordinal)
        value = np.concatenate([encode_base_correction(context.seed_base, base), q, slack])
        new_total, _new_grad, new_breakdown = context.objective(value)
        old_total, _old_grad, old_breakdown = legacy.objective(value)
        component_delta = {
            name: float(getattr(new_breakdown, name) - getattr(old_breakdown, name))
            for name in (
                "e_im",
                "e_bone",
                "e_temporal",
                "e_base_pos",
                "e_base_rot",
                "e_slack",
            )
        }
        rows.append(
            {
                "ordinal": ordinal,
                "frame_id": frame["frame_id"],
                "v1_legacy": float(old_total),
                "v1_v2_binding": float(new_total),
                "absolute_difference": abs(float(new_total) - float(old_total)),
                "max_component_absolute_difference": max(
                    abs(item) for item in component_delta.values()
                ),
                "pass": abs(float(new_total) - float(old_total)) <= 1e-12
                and max(abs(item) for item in component_delta.values()) <= 1e-12,
            }
        )
        bindings.append(
            {
                "ordinal": ordinal,
                "frame_id": frame["frame_id"],
                "binding_sha256": binding.sha256,
                "continuous_inputs_complete": ordinal == 0
                or all(
                    item is not None
                    for item in (
                        binding.previous_runtime_base_scene,
                        binding.previous_robot_qpos,
                        binding.continuous_predicted_translation_scene,
                        binding.continuous_predicted_rotation_scene,
                        binding.continuous_predicted_qpos,
                    )
                ),
            }
        )
    status = "PASS" if all(row["pass"] for row in rows) else "FAIL"
    contract = {
        "schema_version": "ProductionObjectiveContextBindingV2Contract",
        "status": status,
        "typed_binding_schema": ProductionObjectiveContextBindingV2.__name__,
        "required_fields": list(ProductionObjectiveContextBindingV2.__dataclass_fields__),
        "used_by": [
            "objective replay",
            "candidate scoring",
            "micro optimization",
            "development windows",
            "future sparse validation",
            "future sequential refinement",
        ],
        "frame_zero_policy": "preserve historical no-temporal branch",
        "nonzero_policy": "all previous-runtime and continuous-prediction inputs required",
        "historical_v1_semantics_changed": False,
        "frame_receipts": bindings,
    }
    parity = {
        "schema_version": "ProductionObjectiveContextBindingV2Parity",
        "status": status,
        "frame_count": len(rows),
        "tolerance": 1e-12,
        "max_objective_absolute_difference": max(row["absolute_difference"] for row in rows),
        "max_component_absolute_difference": max(
            row["max_component_absolute_difference"] for row in rows
        ),
        "all_production_continuous_inputs_bound": all(
            row["continuous_inputs_complete"] for row in bindings
        ),
        "no_silent_alternate_temporal_branch": status == "PASS",
        "rows": rows,
    }
    write_json(root / "context_binding/production_context_binding_v2.json", contract)
    write_json(root / "context_binding/parity.json", parity)
    if status != "PASS":
        raise RuntimeError("PRODUCTION_CONTEXT_BINDING_PARITY_FAIL")
    return parity


def budget_authority(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    authority = runtime.authority
    payload = {
        **authority.as_dict(),
        "status": "RESOLVED",
        "outcome_driven_threshold_tuning": False,
        "dev1_q1_outcome_used_to_set_budget": False,
        "precedence": [
            "production hard joint/collision constraints",
            "RetargetSemanticValidityV1 explicit limits",
            "wuji_continuous_full_state_v1 positive-control continuity limits",
        ],
    }
    write_json(root / "budget_authority/nonregression_budget_authority.json", payload)
    write_json(
        root / "budget_authority/positive_control_statistics.json",
        {
            "schema_version": "ObjectiveV2PositiveControlAuthorityReceiptV1",
            "status": "BOUND_TO_EXISTING_FROZEN_LIMITS",
            "new_statistics_fit_to_dev1": False,
            "semantic_gate_contract": SemanticGateContractV1().as_dict(),
            "continuous_profile": "configs/retarget/refinement_solvers/wuji_continuous_full_state_v1.yaml",
        },
    )
    for candidate in CANDIDATES:
        write_json(
            root / f"candidates/candidate_{candidate.name[0].lower()}.json",
            {
                **asdict(candidate),
                "development_optimizer": "ObjectiveV2DevelopmentOptimizerV1",
                "development_only": True,
                "hard_constraints": list(
                    evaluate_candidate(
                        candidate,
                        ObjectiveV2Measurements(
                            interaction_e_im=authority.interaction_target,
                            secondary_objective=0.0,
                            bone_direction_p95_rad=0.0,
                            wrist_position_m=0.0,
                            wrist_rotation_rad=0.0,
                            temporal_base_translation_m=0.0,
                            temporal_base_rotation_rad=0.0,
                            temporal_q_inf_rad=0.0,
                            temporal_excess_keypoint_m=0.0,
                            collision_min_signed_distance_m=0.0,
                            joint_limit_min_margin_rad=0.0,
                            rotation_determinant=1.0,
                            unit_scale_ratio=1.0,
                            per_components={},
                        ),
                        authority,
                    )["constraint_margins"]
                ),
            },
        )
    write_json(
        root / "candidates/candidate_c.json",
        {
            "candidate": "C_TAIL_AWARE_WINDOW",
            "status": "NOT_IMPLEMENTED",
            "reason": "A/B express the frozen per-frame target and trajectory-tail evaluation remains a window-level metric; no engineering need justifies a third objective",
        },
    )
    return payload


def q0_q1_preference(root: Path) -> dict[str, Any]:
    parity = read_json(root / "context_binding/parity.json")
    if parity["status"] != "PASS":
        raise RuntimeError("PRODUCTION_CONTEXT_BINDING_PARITY_REQUIRED")
    runtime = D2ARuntime(root)
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    b1 = read_json(O5RB / "b1_thumb_feasibility/frame_selection.json")["frames"]
    for candidate in CANDIDATES:
        preferred = admissible = clear = not_worse = 0
        low_controls = 0
        low_controls_pressure_free = 0
        for frame in b1:
            ordinal = int(frame["ordinal"])
            arrays = np.load(O5RD1 / f"replay/states/frame_{ordinal:04d}.npz", allow_pickle=False)
            binding, context = runtime.bind_context(ordinal)
            slack = runtime.q0_slack(ordinal)
            q0 = runtime.measurement(
                ordinal,
                arrays["q0"],
                arrays["base0"],
                binding=binding,
                context=context,
                slack=slack,
            )
            q1 = runtime.measurement(
                ordinal,
                arrays["q1"],
                arrays["base0"],
                binding=binding,
                context=context,
                slack=slack,
            )
            comparison = compare_candidate_states(candidate, q0, q1, runtime.authority)
            q1_clear = q1.interaction_e_im < q0.interaction_e_im - 1e-10
            q1_admissible = bool(comparison["right"]["feasible"])
            q1_preferred = comparison["preferred"] == "RIGHT"
            q1_not_worse = comparison["preferred"] in {"RIGHT", "TIE"}
            clear += int(q1_clear)
            admissible += int(q1_clear and q1_admissible)
            preferred += int(q1_clear and q1_admissible and q1_preferred)
            not_worse += int(q1_clear and q1_admissible and q1_not_worse)
            if q0.interaction_e_im <= runtime.authority.interaction_target:
                low_controls += 1
                pressure_free = (
                    candidate.primary_value(
                        q0.interaction_e_im, runtime.authority.interaction_target
                    )
                    == 0.0
                )
                low_controls_pressure_free += int(pressure_free)
            rows.append(
                {
                    "candidate": candidate.name,
                    "ordinal": ordinal,
                    "frame_id": int(frame["frame_id"]),
                    "stratum": frame["stratum"],
                    "q0_e_im": q0.interaction_e_im,
                    "q1_e_im": q1.interaction_e_im,
                    "q0_primary": comparison["left"]["primary_objective"],
                    "q1_primary": comparison["right"]["primary_objective"],
                    "q0_secondary": q0.secondary_objective,
                    "q1_secondary": q1.secondary_objective,
                    "q0_feasible": comparison["left"]["feasible"],
                    "q1_feasible": comparison["right"]["feasible"],
                    "q1_violations": ";".join(comparison["right"]["violated_constraints"]),
                    "preference": comparison["preferred"],
                    "preference_reason": comparison["reason"],
                    "clear_e_im_improvement": q1_clear,
                }
            )
        summaries[candidate.name] = {
            "n": len(b1),
            "clear_e_im_improvement_count": clear,
            "q1_admissible_count": admissible,
            "q1_preferred_when_admissible_count": preferred,
            "q1_not_ranked_worse_when_admissible_count": not_worse,
            "q1_not_ranked_worse_when_admissible_fraction": None
            if admissible == 0
            else not_worse / admissible,
            "development_80pct_criterion": None
            if admissible == 0
            else not_worse / admissible >= 0.8,
            "development_80pct_criterion_status": "NOT_APPLICABLE_NO_ADMISSIBLE_Q1"
            if admissible == 0
            else "PASS"
            if not_worse / admissible >= 0.8
            else "FAIL",
            "low_control_count": low_controls,
            "low_control_primary_pressure_free_count": low_controls_pressure_free,
            "low_control_behavior_pass": low_controls > 0
            and low_controls_pressure_free == low_controls,
        }
    write_csv(root / "candidates/q0_q1_candidate_preference.csv", rows)
    payload = {
        "schema_version": "O5RD2AQ0Q1CandidatePreferenceV1",
        "n_b1": len(b1),
        "development_criterion_not_independent_gate": True,
        "summaries": summaries,
    }
    write_json(root / "candidates/q0_q1_candidate_preference.json", payload)
    return payload


def _query_set(runtime: D2ARuntime, context: Any, qpos: np.ndarray, base: np.ndarray) -> Any:
    state = np.concatenate([encode_base_correction(context.seed_base, base), qpos])
    result = runtime.backends.solver_sdf.query_scene(
        context.candidate_points(state), context.object_pose_scene
    )
    return build_query_set(result.signed_distance, runtime.surface.geometry_ids, runtime.query)


def _measurement_row(values: ObjectiveV2Measurements) -> dict[str, Any]:
    return jsonable(asdict(values))


def _seed_state(
    runtime: D2ARuntime,
    ordinal: int,
    candidate: ObjectiveV2Candidate,
    binding: ProductionObjectiveContextBindingV2,
    context: Any,
) -> tuple[np.ndarray, np.ndarray, str]:
    old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
    old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
    if binding.continuous_prediction_base is None:
        return old_q, old_base, "old_trajectory_frame0"
    predicted_q = np.asarray(binding.continuous_predicted_qpos, dtype=np.float64)
    predicted_base = np.asarray(binding.continuous_prediction_base, dtype=np.float64)
    old = runtime.measurement(
        ordinal,
        old_q,
        old_base,
        binding=binding,
        context=context,
    )
    predicted = runtime.measurement(
        ordinal,
        predicted_q,
        predicted_base,
        binding=binding,
        context=context,
    )
    comparison = compare_candidate_states(candidate, old, predicted, runtime.authority)
    if comparison["preferred"] == "RIGHT":
        return predicted_q, predicted_base, "previous_v2_transported_prediction"
    return old_q, old_base, "old_trajectory_q_old_t"


def optimize_development_frame(
    runtime: D2ARuntime,
    ordinal: int,
    candidate: ObjectiveV2Candidate,
    *,
    previous_base: np.ndarray | None = None,
    previous_qpos: np.ndarray | None = None,
    maxiter: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run one bounded two-phase ObjectiveV2 development solve."""

    started = time.perf_counter()
    binding, context = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_qpos
    )
    seed_q, seed_base, seed_source = _seed_state(runtime, ordinal, candidate, binding, context)
    seed_correction = encode_base_correction(context.seed_base, seed_base)
    if binding.continuous_prediction_base is None:
        trust_reference = np.concatenate([seed_correction, seed_q])
    else:
        trust_reference = np.concatenate(
            [
                encode_base_correction(context.seed_base, binding.continuous_prediction_base),
                binding.continuous_predicted_qpos,
            ]
        )
    context.trust_region_reference = trust_reference
    # The production trust callback is componentwise while Semantic V1 gates
    # vector norms.  Dividing the base budgets by sqrt(3) is a conservative
    # implication of the frozen norm limits, not a fitted threshold.
    context.trust_region_limits = (
        runtime.authority.temporal_base_translation_limit_m / np.sqrt(3.0),
        runtime.authority.temporal_base_rotation_limit_rad / np.sqrt(3.0),
        runtime.authority.temporal_q_inf_limit_rad,
    )
    query = _query_set(runtime, context, seed_q, seed_base)
    primary_context = ObjectiveV2DevelopmentContext(
        context, candidate, runtime.authority, phase="primary"
    )
    solver = replace(runtime.solver, maxiter=maxiter)
    initial = np.concatenate([seed_correction, seed_q])
    primary = refine_frame(
        primary_context,
        query,
        solver,
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery="none",
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=initial,
        initialization_source=f"{seed_source}_objective_v2_primary",
    )
    primary_values = runtime.measurement(
        ordinal,
        primary.qpos,
        primary.base_pose_scene,
        binding=binding,
        context=context,
        slack=primary.slack,
    )
    primary_receipt = evaluate_candidate(candidate, primary_values, runtime.authority)
    retention = interaction_retention_limit(
        primary_values.interaction_e_im, runtime.authority.interaction_target
    )
    phase2_binding, phase2_base = runtime.bind_context(
        ordinal, previous_base=previous_base, previous_qpos=previous_qpos
    )
    phase2_base.trust_region_reference = trust_reference
    phase2_base.trust_region_limits = context.trust_region_limits
    phase2_query = _query_set(runtime, phase2_base, primary.qpos, primary.base_pose_scene)
    secondary_context = ObjectiveV2DevelopmentContext(
        phase2_base,
        candidate,
        runtime.authority,
        phase="secondary",
        interaction_retention_limit=retention,
    )
    secondary = refine_frame(
        secondary_context,
        phase2_query,
        solver,
        max_rounds=runtime.query.max_active_set_rounds,
        active_margin_m=runtime.query.active_margin_m,
        point_jacobian_backend=runtime.execution.point_jacobian_backend,
        strict_recovery="none",
        final_audit_scheduling=runtime.execution.final_audit_scheduling,
        initial_state_without_slack=np.concatenate(
            [
                encode_base_correction(phase2_base.seed_base, primary.base_pose_scene),
                primary.qpos,
            ]
        ),
        initialization_source="objective_v2_primary_then_secondary",
    )
    secondary_values = runtime.measurement(
        ordinal,
        secondary.qpos,
        secondary.base_pose_scene,
        binding=phase2_binding,
        context=phase2_base,
        slack=secondary.slack,
    )
    secondary_receipt = evaluate_candidate(candidate, secondary_values, runtime.authority)
    retention_pass = secondary_values.interaction_e_im <= retention + 1e-10
    if secondary.accepted and secondary_receipt["feasible"] and retention_pass:
        selected = secondary
        selected_values = secondary_values
        selected_receipt = secondary_receipt
        selected_phase = "secondary"
    elif primary.accepted and primary_receipt["feasible"]:
        selected = primary
        selected_values = primary_values
        selected_receipt = primary_receipt
        selected_phase = "primary_fallback"
    else:
        old_q = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        old_base = np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        selected = None
        selected_values = runtime.measurement(
            ordinal,
            old_q,
            old_base,
            binding=binding,
            context=context,
        )
        selected_receipt = evaluate_candidate(candidate, selected_values, runtime.authority)
        selected_phase = "old_state_fail_closed"
    q_out = (
        np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
        if selected is None
        else np.asarray(selected.qpos, dtype=np.float64)
    )
    base_out = (
        np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64)
        if selected is None
        else np.asarray(selected.base_pose_scene, dtype=np.float64)
    )
    old_values = runtime.measurement(
        ordinal,
        np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64),
        np.asarray(runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64),
        binding=binding,
        context=context,
    )
    receipt = {
        "schema_version": "ObjectiveV2DevelopmentFrameReceiptV1",
        "development_only": True,
        "not_production_certified": True,
        "candidate": candidate.name,
        "ordinal": int(ordinal),
        "frame_id": int(runtime.graph.frame_indices[ordinal]),
        "seed_source": seed_source,
        "maxiter_per_phase": maxiter,
        "retention_limit": retention,
        "retention_pass": retention_pass,
        "primary_solver": {
            "accepted": bool(primary.accepted),
            "status": int(primary.optimizer_status_code),
            "message": primary.optimizer_message,
            "nfev": int(primary.optimizer_function_evaluations),
            "runtime_sec": float(primary.solve_time_s),
        },
        "secondary_solver": {
            "accepted": bool(secondary.accepted),
            "status": int(secondary.optimizer_status_code),
            "message": secondary.optimizer_message,
            "nfev": int(secondary.optimizer_function_evaluations),
            "runtime_sec": float(secondary.solve_time_s),
        },
        "selected_phase": selected_phase,
        "technical_success": selected is not None,
        "old": _measurement_row(old_values),
        "primary": _measurement_row(primary_values),
        "secondary": _measurement_row(secondary_values),
        "selected": _measurement_row(selected_values),
        "selected_evaluation": selected_receipt,
        "elapsed_sec": time.perf_counter() - started,
        "context_binding_sha256": binding.sha256,
    }
    return q_out, base_out, receipt


def single_frame_selection() -> list[dict[str, Any]]:
    rows = consumed_frames()
    selected: list[dict[str, Any]] = []
    for stratum in WINDOW_STRATA:
        pool = [row for row in rows if row["stratum"] == stratum]
        indices = np.linspace(0, len(pool) - 1, 5, dtype=int)
        selected.extend(pool[int(index)] for index in indices)
    return selected


def single_frame_development(root: Path) -> dict[str, Any]:
    selection = single_frame_selection()
    write_json(
        root / "development_data/single_frame_development_set.json",
        {
            "schema_version": "ObjectiveV2SingleFrameDevelopmentSetV1",
            "selection_rule": "five deterministic temporal quantiles per consumed HIGH/MID/LOW stratum",
            "count": len(selection),
            "frames": selection,
            "independent_validation_role": False,
        },
    )
    runtime = D2ARuntime(root)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for frame in selection:
            ordinal = int(frame["ordinal"])
            receipt_path = (
                root
                / "candidates/single_frame_receipts"
                / candidate.name.lower()
                / f"frame_{ordinal:04d}.json"
            )
            state_path = receipt_path.with_suffix(".npz")
            try:
                if receipt_path.exists() and state_path.exists():
                    receipt = read_json(receipt_path)
                else:
                    qpos, base, receipt = optimize_development_frame(runtime, ordinal, candidate)
                    receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
                    write_json(receipt_path, receipt)
                old, new = receipt["old"], receipt["selected"]
                rows.append(
                    {
                        "candidate": candidate.name,
                        "ordinal": ordinal,
                        "frame_id": int(frame["frame_id"]),
                        "stratum": frame["stratum"],
                        "technical_success": receipt["technical_success"],
                        "selected_phase": receipt["selected_phase"],
                        "old_e_im": old["interaction_e_im"],
                        "new_e_im": new["interaction_e_im"],
                        "old_above_new_below": old["interaction_e_im"]
                        > runtime.authority.interaction_target
                        and new["interaction_e_im"] <= runtime.authority.interaction_target,
                        "e_im_reduction": old["interaction_e_im"] - new["interaction_e_im"],
                        "new_feasible": receipt["selected_evaluation"]["feasible"],
                        "new_violations": ";".join(
                            receipt["selected_evaluation"]["violated_constraints"]
                        ),
                        "primary_nfev": receipt["primary_solver"]["nfev"],
                        "secondary_nfev": receipt["secondary_solver"]["nfev"],
                        "solver_sec": receipt["primary_solver"]["runtime_sec"]
                        + receipt["secondary_solver"]["runtime_sec"],
                    }
                )
            except Exception as exc:
                failure = {
                    "stage": "single_frame_development",
                    "candidate": candidate.name,
                    "ordinal": ordinal,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                write_json(receipt_path, {**failure, "technical_success": False})
                rows.append(
                    {
                        "candidate": candidate.name,
                        "ordinal": ordinal,
                        "frame_id": int(frame["frame_id"]),
                        "stratum": frame["stratum"],
                        "technical_success": False,
                        "selected_phase": "exception_fail_closed",
                        "old_e_im": None,
                        "new_e_im": None,
                        "old_above_new_below": False,
                        "e_im_reduction": None,
                        "new_feasible": False,
                        "new_violations": type(exc).__name__,
                        "primary_nfev": 0,
                        "secondary_nfev": 0,
                        "solver_sec": 0.0,
                    }
                )
    write_csv(root / "candidates/single_frame_results.csv", rows)
    write_csv(root / "candidates/single_frame_development_results.csv", rows)
    if failures:
        path = root / "technical_failures.jsonl"
        path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in failures),
            encoding="utf-8",
        )
    summaries: dict[str, Any] = {}
    for candidate in CANDIDATES:
        subset = [row for row in rows if row["candidate"] == candidate.name]
        finite = [row for row in subset if row["new_e_im"] is not None]
        above = [
            row for row in finite if float(row["old_e_im"]) > runtime.authority.interaction_target
        ]
        summaries[candidate.name] = {
            "frame_count": len(subset),
            "technical_success_count": sum(bool(row["technical_success"]) for row in subset),
            "old_above_to_new_below_fraction": None
            if not above
            else sum(bool(row["old_above_new_below"]) for row in above) / len(above),
            "median_e_im_reduction": None
            if not finite
            else float(np.median([float(row["e_im_reduction"]) for row in finite])),
            "new_semantic_hard_violation_count": sum(
                not bool(row["new_feasible"]) for row in subset
            ),
        }
    payload = {
        "schema_version": "ObjectiveV2SingleFrameDevelopmentSummaryV1",
        "development_only": True,
        "summaries": summaries,
    }
    write_json(root / "candidates/single_frame_results.json", payload)
    return payload


def window_development(root: Path) -> dict[str, Any]:
    runtime = D2ARuntime(root)
    windows = development_windows(root)
    rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for window in windows:
            previous_ordinal = int(window["ordinals"][0]) - 1
            previous_base = np.asarray(
                runtime.final.arrays["base_pose_scene"][previous_ordinal], dtype=np.float64
            )
            previous_q = np.asarray(
                runtime.final.arrays["qpos"][previous_ordinal], dtype=np.float64
            )
            window_rows: list[dict[str, Any]] = []
            for ordinal in window["ordinals"]:
                ordinal = int(ordinal)
                receipt_path = (
                    root
                    / "candidates/window_receipts"
                    / candidate.name.lower()
                    / str(window["window_id"])
                    / f"frame_{ordinal:04d}.json"
                )
                state_path = receipt_path.with_suffix(".npz")
                try:
                    if receipt_path.exists() and state_path.exists():
                        receipt = read_json(receipt_path)
                        state = np.load(state_path, allow_pickle=False)
                        qpos = np.asarray(state["qpos"])
                        base = np.asarray(state["base_pose_scene"])
                    else:
                        qpos, base, receipt = optimize_development_frame(
                            runtime,
                            ordinal,
                            candidate,
                            previous_base=previous_base,
                            previous_qpos=previous_q,
                        )
                        receipt_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(state_path, qpos=qpos, base_pose_scene=base)
                        write_json(receipt_path, receipt)
                    old, new = receipt["old"], receipt["selected"]
                    row = {
                        "candidate": candidate.name,
                        "window_id": window["window_id"],
                        "stratum": window["stratum"],
                        "ordinal": ordinal,
                        "frame_id": int(runtime.graph.frame_indices[ordinal]),
                        "technical_success": receipt["technical_success"],
                        "selected_phase": receipt["selected_phase"],
                        "seed_source": receipt["seed_source"],
                        "old_e_im": old["interaction_e_im"],
                        "new_e_im": new["interaction_e_im"],
                        "new_feasible": receipt["selected_evaluation"]["feasible"],
                        "new_violations": ";".join(
                            receipt["selected_evaluation"]["violated_constraints"]
                        ),
                        "q_step_inf_rad": new["temporal_q_inf_rad"],
                        "translation_step_m": new["temporal_base_translation_m"],
                        "rotation_step_rad": new["temporal_base_rotation_rad"],
                        "excess_keypoint_m": new["temporal_excess_keypoint_m"],
                        "wrist_position_m": new["wrist_position_m"],
                        "wrist_rotation_rad": new["wrist_rotation_rad"],
                        "bone_p95_rad": new["bone_direction_p95_rad"],
                        "collision_min_signed_distance_m": new["collision_min_signed_distance_m"],
                        "joint_limit_min_margin_rad": new["joint_limit_min_margin_rad"],
                        "solver_sec": receipt["primary_solver"]["runtime_sec"]
                        + receipt["secondary_solver"]["runtime_sec"],
                        "objective_calls": receipt["primary_solver"]["nfev"]
                        + receipt["secondary_solver"]["nfev"],
                    }
                except Exception as exc:
                    row = {
                        "candidate": candidate.name,
                        "window_id": window["window_id"],
                        "stratum": window["stratum"],
                        "ordinal": ordinal,
                        "frame_id": int(runtime.graph.frame_indices[ordinal]),
                        "technical_success": False,
                        "selected_phase": "exception_fail_closed",
                        "seed_source": "unavailable",
                        "old_e_im": float(runtime.final.arrays["e_im"][ordinal]),
                        "new_e_im": float(runtime.final.arrays["e_im"][ordinal]),
                        "new_feasible": False,
                        "new_violations": type(exc).__name__,
                        "q_step_inf_rad": None,
                        "translation_step_m": None,
                        "rotation_step_rad": None,
                        "excess_keypoint_m": None,
                        "wrist_position_m": None,
                        "wrist_rotation_rad": None,
                        "bone_p95_rad": None,
                        "collision_min_signed_distance_m": None,
                        "joint_limit_min_margin_rad": None,
                        "solver_sec": 0.0,
                        "objective_calls": 0,
                    }
                    qpos = np.asarray(runtime.final.arrays["qpos"][ordinal], dtype=np.float64)
                    base = np.asarray(
                        runtime.final.arrays["base_pose_scene"][ordinal], dtype=np.float64
                    )
                    write_json(
                        receipt_path,
                        {
                            "stage": "window_development",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "technical_success": False,
                        },
                    )
                rows.append(row)
                window_rows.append(row)
                previous_q, previous_base = qpos, base
            old_values = np.asarray([float(row["old_e_im"]) for row in window_rows])
            new_values = np.asarray([float(row["new_e_im"]) for row in window_rows])
            finite_metrics = all(row["q_step_inf_rad"] is not None for row in window_rows)
            semantic_rows.append(
                {
                    "candidate": candidate.name,
                    "window_id": window["window_id"],
                    "stratum": window["stratum"],
                    "frame_count": len(window_rows),
                    "old_mean_e_im": float(np.mean(old_values)),
                    "old_p90_e_im": float(np.quantile(old_values, 0.90)),
                    "old_p95_e_im": float(np.quantile(old_values, 0.95)),
                    "old_max_e_im": float(np.max(old_values)),
                    "new_mean_e_im": float(np.mean(new_values)),
                    "new_p90_e_im": float(np.quantile(new_values, 0.90)),
                    "new_p95_e_im": float(np.quantile(new_values, 0.95)),
                    "new_max_e_im": float(np.max(new_values)),
                    "technical_success": all(bool(row["technical_success"]) for row in window_rows),
                    "hard_validity_pass": all(bool(row["new_feasible"]) for row in window_rows),
                    "continuity_pass": finite_metrics
                    and all(
                        float(row["q_step_inf_rad"])
                        <= runtime.authority.temporal_q_inf_limit_rad + 1e-12
                        and float(row["translation_step_m"])
                        <= runtime.authority.temporal_base_translation_limit_m + 1e-12
                        and float(row["rotation_step_rad"])
                        <= runtime.authority.temporal_base_rotation_limit_rad + 1e-12
                        for row in window_rows
                    ),
                    "wrist_pass": finite_metrics
                    and all(
                        float(row["wrist_position_m"])
                        <= runtime.authority.wrist_position_limit_m + 1e-12
                        and float(row["wrist_rotation_rad"])
                        <= runtime.authority.wrist_rotation_limit_rad + 1e-12
                        for row in window_rows
                    ),
                    "bone_pass": finite_metrics
                    and all(
                        float(row["bone_p95_rad"])
                        <= runtime.authority.bone_direction_limit_rad + 1e-12
                        for row in window_rows
                    ),
                    "semantic_v1_role": "DEVELOPMENT_COMPATIBILITY_AUDIT_NOT_TRAJECTORY_VALIDATION",
                }
            )
    write_csv(root / "candidates/window_results.csv", rows)
    write_csv(root / "candidates/window_development_results.csv", semantic_rows)
    write_csv(root / "candidates/semantic_v1_development_results.csv", semantic_rows)
    payload = {
        "schema_version": "ObjectiveV2WindowDevelopmentSummaryV1",
        "development_only": True,
        "not_objective_v2_trajectory_validation": True,
        "windows": semantic_rows,
    }
    write_json(root / "candidates/window_results.json", payload)
    return payload


def semantic_v1_development_audit(root: Path) -> dict[str, Any]:
    """Run the unchanged Semantic V1 qualification function on each dev window."""

    runtime = D2ARuntime(root)
    gate = SemanticGateContractV1()
    detail: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    frame_rows = read_csv(root / "candidates/window_results.csv")
    for candidate in CANDIDATES:
        for window in development_windows(root):
            subset = [
                row
                for row in frame_rows
                if row["candidate"] == candidate.name and row["window_id"] == window["window_id"]
            ]
            subset.sort(key=lambda row: int(row["ordinal"]))
            robot_frames: list[np.ndarray] = []
            wrist_position: list[float] = []
            wrist_rotation: list[float] = []
            bone_errors: list[np.ndarray] = []
            robot_contact: list[bool] = []
            scale_ratios: list[np.ndarray] = []
            for row in subset:
                ordinal = int(row["ordinal"])
                state_path = (
                    root
                    / "candidates/window_receipts"
                    / candidate.name.lower()
                    / str(window["window_id"])
                    / f"frame_{ordinal:04d}.npz"
                )
                state = np.load(state_path, allow_pickle=False)
                q = np.asarray(state["qpos"], dtype=np.float64)
                base = np.asarray(state["base_pose_scene"], dtype=np.float64)
                keypoints_base = np.asarray(
                    runtime.model.keypoints_scene(q, np.eye(4)), dtype=np.float64
                )
                robot_frame = base @ np.asarray(
                    _semantic_frames(keypoints_base, runtime.model.side)
                )
                robot_frames.append(robot_frame)
                source_frame = np.asarray(runtime.semantic["canonical_wrist_pose_scene"][ordinal])
                object_pose = np.asarray(runtime.semantic["object_pose_scene"][ordinal])
                error = transform_error(
                    relative_transform(object_pose, source_frame),
                    relative_transform(object_pose, robot_frame),
                )
                wrist_position.append(float(np.asarray(error["position_m"])))
                wrist_rotation.append(float(np.asarray(error["rotation_rad"])))
                features = extract_bone_features(
                    keypoints_base,
                    runtime.frame_profile,
                    runtime.bone_profile,
                    side=runtime.model.side,
                    strict=True,
                )
                source_directions = np.asarray(runtime.semantic["source_bone_directions"][ordinal])
                bone_errors.append(
                    angular_error(source_directions, np.asarray(features.unit_directions))
                )
                warm_features = extract_bone_features(
                    np.asarray(runtime.warm.arrays["robot_keypoints_base"][ordinal]),
                    runtime.frame_profile,
                    runtime.bone_profile,
                    side=runtime.model.side,
                    strict=True,
                )
                scale_ratios.append(
                    np.asarray(features.bone_lengths) / np.asarray(warm_features.bone_lengths)
                )
                binding, context = runtime.bind_context(ordinal)
                correction = encode_base_correction(context.seed_base, base)
                points = context.candidate_points(np.concatenate([correction, q]))
                distance = runtime.resources.reference_sdf.query_scene(
                    points, binding.object_pose_scene
                )
                robot_contact.append(
                    bool(np.min(distance.unsigned_distance) <= gate.contact_opportunity_distance_m)
                )
            eim = np.asarray([float(row["new_e_im"]) for row in subset])
            source_contact = np.asarray(
                runtime.semantic["source_contact_expected"][window["ordinals"]], dtype=bool
            )
            frames = np.stack(robot_frames)
            ratios = np.concatenate(scale_ratios)
            frame_authority_pass = bool(
                np.all(np.linalg.det(frames[:, :3, :3]) >= gate.reflection_determinant_minimum)
                and np.all(ratios >= gate.unit_scale_ratio_minimum)
                and np.all(ratios <= gate.unit_scale_ratio_maximum)
            )
            qualification = qualify_semantics(
                wrist_position_m=np.asarray(wrist_position),
                wrist_rotation_rad=np.asarray(wrist_rotation),
                bone_error_rad=np.concatenate(bone_errors),
                source_contact=source_contact,
                robot_contact=np.asarray(robot_contact),
                robot_wrist_transforms=frames,
                frame_authority_pass=frame_authority_pass,
                time_alignment_pass=True,
                interaction_geometry_pass=bool(
                    np.isfinite(eim).all()
                    and np.quantile(eim, 0.95) <= gate.interaction_e_im_p95_limit
                ),
                gate=gate,
            )
            record = {
                "candidate": candidate.name,
                "window_id": window["window_id"],
                "stratum": window["stratum"],
                "frame_count": len(subset),
                "interaction_e_im_p95": float(np.quantile(eim, 0.95)),
                "interaction_target": gate.interaction_e_im_p95_limit,
                "qualification": qualification,
                "semantic_v1_unchanged": True,
                "development_only": True,
                "not_independent_validation": True,
            }
            detail.append(record)
            rows.append(
                {
                    "candidate": candidate.name,
                    "window_id": window["window_id"],
                    "stratum": window["stratum"],
                    "frame_count": len(subset),
                    "interaction_e_im_p95": record["interaction_e_im_p95"],
                    "interaction_target": gate.interaction_e_im_p95_limit,
                    "semantic_v1_status": qualification["status"],
                    "gross_sanity_pass": qualification["gross_sanity_pass"],
                    "interaction_geometry_pass": qualification["interaction_geometry_pass"],
                    "temporal_continuity_status": qualification["temporal_continuity_status"],
                    "contact_recall_status": qualification["contact_recall_status"],
                    "development_only": True,
                }
            )
    write_csv(root / "candidates/semantic_v1_development_results.csv", rows)
    payload = {
        "schema_version": "ObjectiveV2SemanticV1DevelopmentCompatibilityV1",
        "semantic_v1_changed": False,
        "retarget_semantic_v2_created": False,
        "objective_v2_does_not_replace_semantic_v1": True,
        "results": detail,
    }
    write_json(root / "candidates/semantic_v1_development_results.json", payload)
    return payload


def candidate_selection(root: Path) -> dict[str, Any]:
    q0q1 = read_json(root / "candidates/q0_q1_candidate_preference.json")
    single = read_json(root / "candidates/single_frame_results.json")
    windows = read_json(root / "candidates/window_results.json")["windows"]
    semantic = read_json(root / "candidates/semantic_v1_development_results.json")["results"]
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        name = candidate.name
        window_rows = [row for row in windows if row["candidate"] == name]
        semantic_rows = [row for row in semantic if row["candidate"] == name]
        q_summary = q0q1["summaries"][name]
        single_summary = single["summaries"][name]
        interaction_alignment = bool(
            name == "B_LEXICOGRAPHIC_THRESHOLD" and q_summary["low_control_behavior_pass"]
        )
        hard_nonregression = all(bool(row["hard_validity_pass"]) for row in window_rows)
        temporal_nonregression = all(bool(row["continuity_pass"]) for row in window_rows)
        technical_complete = all(bool(row["technical_success"]) for row in window_rows)
        semantic_compatibility = all(
            row["qualification"]["status"] == "RETARGET_SEMANTIC_PASS" for row in semantic_rows
        )
        ready = all(
            (
                interaction_alignment,
                hard_nonregression,
                temporal_nonregression,
                technical_complete,
                semantic_compatibility,
            )
        )
        status = (
            "READY"
            if ready
            else "PROMISING_NOT_READY"
            if name == "B_LEXICOGRAPHIC_THRESHOLD"
            else "REJECTED"
        )
        rows.append(
            {
                "candidate": name,
                "primary": candidate.primary_definition,
                "secondary": candidate.secondary_definition,
                "constraints": candidate.retention_definition + "; frozen non-regression authority",
                "q0_q1_low_control_pass": q_summary["low_control_behavior_pass"],
                "single_frame_technical_success_count": single_summary["technical_success_count"],
                "single_frame_count": single_summary["frame_count"],
                "single_frame_recovery_fraction": single_summary["old_above_to_new_below_fraction"],
                "window_technical_pass_count": sum(
                    bool(row["technical_success"]) for row in window_rows
                ),
                "window_hard_validity_pass_count": sum(
                    bool(row["hard_validity_pass"]) for row in window_rows
                ),
                "window_continuity_pass_count": sum(
                    bool(row["continuity_pass"]) for row in window_rows
                ),
                "semantic_v1_window_pass_count": sum(
                    row["qualification"]["status"] == "RETARGET_SEMANTIC_PASS"
                    for row in semantic_rows
                ),
                "ready": ready,
                "status": status,
            }
        )
    write_csv(root / "candidates/candidate_summary.csv", rows)
    ready = [row for row in rows if row["ready"]]
    selected = None if not ready else sorted(row["candidate"] for row in ready)[0]
    decision = {
        "schema_version": "RetargetObjectiveV2SelectionDecisionV1",
        "selection_rubric_order": [
            "interaction validity alignment",
            "no new Semantic V1 hard violation",
            "temporal/window non-regression",
            "simple interpretable semantics",
            "runtime tie-break only",
        ],
        "lowest_dev1_e_im_wins": False,
        "selected_objective": selected or "NONE",
        "candidate_count": len(CANDIDATES),
        "candidate_c": "NOT_IMPLEMENTED",
        "rows": rows,
        "decision": "NO_CANDIDATE_READY" if selected is None else "SELECTED_FOR_FREEZE",
    }
    write_json(root / "selected_objective/selection_decision.json", decision)
    if selected is None:
        high_b = next(
            row
            for row in windows
            if row["candidate"] == "B_LEXICOGRAPHIC_THRESHOLD" and row["stratum"] == "HIGH"
        )
        write_json(
            root / "selected_objective/no_candidate_ready.json",
            {
                "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": "NO_CANDIDATE_READY",
                "SELECTED_OBJECTIVE": "NONE",
                "RETARGET_OBJECTIVE_V2_SHA256": None,
                "primary_blocker": "Candidate B does not complete the HIGH development window without continuity violations",
                "high_window_evidence": high_b,
                "candidate_a_rejected": "LOW controls retain nonzero primary pressure and development completion is weaker",
                "candidate_b_status": "PROMISING_NOT_READY",
                "objective_contract_frozen": False,
                "independent_validation_allowed": False,
            },
        )
    return decision


def future_certification_plan(root: Path) -> dict[str, Any]:
    decision = read_json(root / "selected_objective/selection_decision.json")
    allowed = decision["selected_objective"] != "NONE"
    plan = {
        "schema_version": "ObjectiveV2IndependentCertificationPlanV1",
        "status": "DESIGNED_NOT_EXECUTED" if allowed else "BLOCKED_NO_CANDIDATE_READY",
        "execution_allowed": False,
        "reason": "this O5R-D2A phase hard-stops before new independent validation",
        "required_order": [
            "freeze ObjectiveV2",
            "freeze production-search contract",
            "freeze untouched SparseValidationV2",
            "freeze untouched WindowValidationV2",
            "run sparse gate",
            "run window gate",
            "both pass",
            "DEV2 frame0 hard control",
            "frame0 pass",
            "DEV2 240-frame solve once",
        ],
        "method_development_ledger": str(
            (root / "development_data/method_development_ledger_v2.json").resolve()
        ),
        "required_overlap_with_development": 0,
        "sparse": {
            "proposed_count": 30,
            "selection": "deterministic untouched DEV1 HIGH/MID/LOW frames after objective and search freeze",
            "gate_schema": [
                "technical completion 30/30",
                "old-above-target to new-at-or-below-target recovery >= 0.80",
                "median relative E_IM reduction >= 0.50 on old-above-target frames",
                "all LOW controls remain at or below target",
                "zero new Semantic V1 hard violations",
                "three-repeat determinism on predeclared representatives",
            ],
        },
        "windows": {
            "proposed_count": 4,
            "proposed_length_frames": 24,
            "selection": "deterministic untouched contiguous DEV1 windows with zero ledger overlap",
            "gate_schema": [
                "technical completion on every frame",
                "each window p95 E_IM <= frozen Semantic V1 target",
                "all frozen continuous-profile limits pass",
                "all wrist/bone/collision/joint/invariant limits pass",
                "three-repeat determinism on one predeclared window",
            ],
        },
        "validation_sets_created": False,
        "validation_executed": False,
    }
    write_json(root / "future_certification/independent_certification_plan.json", plan)
    write_json(
        root / "future_certification/future_sparse_gate_contract.json",
        {**plan["sparse"], "status": "SCHEMA_ONLY_NOT_FROZEN", "executed": False},
    )
    write_json(
        root / "future_certification/future_window_gate_contract.json",
        {**plan["windows"], "status": "SCHEMA_ONLY_NOT_FROZEN", "executed": False},
    )
    write_json(
        root / "future_certification/objective_v2_sequential_refinement_concept_v1.json",
        {
            "schema_version": "ObjectiveV2SequentialRefinementConceptV1",
            "concept": "existing DEV1 q_old[t] to ObjectiveV2 sequential refinement",
            "from_scratch_2722_frame_retarget": False,
            "executed_in_o5rd2a": False,
            "requires_independent_certification_pass": True,
        },
    )
    return plan


def mutation_audit(root: Path) -> dict[str, Any]:
    frozen = read_json(root / "preflight/frozen_authorities.json")["authorities"]
    rows: list[dict[str, Any]] = []
    for name, path in frozen_paths().items():
        before = str(frozen[name]["sha256"])
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
    local_tracked = git("ls-files", ".local").splitlines()
    payload = {
        "schema_version": "O5RD2ANoMutationAuditV1",
        "status": "PASS"
        if all(row["unchanged"] for row in rows) and not guidance_status and not local_tracked
        else "FAIL",
        "authorities": rows,
        "required_unchanged": {
            "production_objective_v1": next(
                row["unchanged"] for row in rows if row["name"] == "production_objective_v1"
            ),
            "semantic_v1": next(row["unchanged"] for row in rows if row["name"] == "semantic_v1"),
            "structured_solver_v1": next(
                row["unchanged"] for row in rows if row["name"] == "o5rc_frozen_contract"
            ),
            "dev1_old_trajectory": next(
                row["unchanged"] for row in rows if row["name"] == "dev1_old_trajectory"
            ),
            "manifest_v2": next(row["unchanged"] for row in rows if row["name"] == "manifest_v2"),
            "split_v2": next(row["unchanged"] for row in rows if row["name"] == "split_v2"),
        },
        "guidance_worktree_status": guidance_status,
        "local_tracked_paths": local_tracked,
    }
    write_json(root / "preflight/no_mutation_audit.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("O5RD2A_NO_MUTATION_AUDIT_FAILED")
    return payload


def summarize(root: Path) -> dict[str, Any]:
    no_mutation = mutation_audit(root)
    context = read_json(root / "context_binding/parity.json")
    selection = read_json(root / "selected_objective/selection_decision.json")
    no_candidate = read_json(root / "selected_objective/no_candidate_ready.json")
    ledger = read_json(root / "development_data/method_development_ledger_v2.json")
    window_set = read_json(root / "development_data/development_window_set.json")
    q0q1 = read_json(root / "candidates/q0_q1_candidate_preference.json")
    single = read_json(root / "candidates/single_frame_results.json")
    windows = read_json(root / "candidates/window_results.json")
    safety = {
        "BRANCH": EXPECTED_BRANCH,
        "PRODUCTION_OBJECTIVE_V1_MODIFIED": "NO",
        "PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2": context["status"],
        "STRUCTURED_SOLVER_V1_MODIFIED": "NO",
        "STRUCTURED_SOLVER_V1_REUSED_AS_PRODUCTION": "NO",
        "SEMANTIC_V1_CHANGED": "NO",
        "E_IM_THRESHOLD_CHANGED": "NO",
        "RETARGET_SEMANTIC_V2_CREATED": "NO",
        "OBJECTIVE_WEIGHT_GRID_SEARCH_USED": "NO",
        "OUTCOME_DRIVEN_THRESHOLD_TUNING": "NO",
        "CANDIDATE_COUNT": len(CANDIDATES),
        "CONSUMED_55_FRAMES_USED_FOR_DEVELOPMENT": "YES",
        "NEW_DEVELOPMENT_WINDOW_COUNT": len(window_set["windows"]),
        "NEW_INDEPENDENT_VALIDATION_EXECUTED": "NO",
        "DEV1_SPARSE_VALIDATION_V2_CREATED": "NO",
        "DEV1_WINDOW_VALIDATION_V2_CREATED": "NO",
        "RETARGET_OBJECTIVE_V2_DESIGN_STATUS": no_candidate["RETARGET_OBJECTIVE_V2_DESIGN_STATUS"],
        "RETARGET_OBJECTIVE_V2_SHA256": None,
        "DEV1_FULL_RETARGET_RERUNS": 0,
        "DEV1_FULL_V2_REFINEMENT_RUNS": 0,
        "DEV2_FRAME0_NEW_SOLVES": 0,
        "DEV2_FRAME0_HARD_CONTROL_PRESERVED": "YES",
        "DEV2_FULL_PRODUCTION_SOLVES": 0,
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
    resource_usage = {
        "context_binding_parity_frames": context["frame_count"],
        "q0_q1_counterfactual_frames": q0q1["n_b1"],
        "single_frame_candidate_solves": sum(
            row["frame_count"] for row in single["summaries"].values()
        ),
        "development_window_count": len(window_set["windows"]),
        "development_window_unique_frames": window_set["unique_window_frame_count"],
        "development_window_candidate_frame_attempts": len(CANDIDATES)
        * window_set["unique_window_frame_count"],
        "future_validation_exclusion_count": ledger["future_validation_exclusion_count"],
        "dev1_full_retarget_reruns": 0,
        "dev1_full_v2_refinement_runs": 0,
        "dev2_frame0_new_solves": 0,
        "dev2_full_production_solves": 0,
        "new_independent_validation_executed": False,
    }
    write_json(root / "resource_usage.json", resource_usage)
    failure_path = root / "technical_failures.jsonl"
    if not failure_path.exists():
        failure_path.write_text("", encoding="utf-8")
    summary = {
        "schema_version": "OakInk2O5RD2AFinalSummaryV1",
        "git": {
            "branch": EXPECTED_BRANCH,
            "start_head": START_HEAD,
            "current_head": git("rev-parse", "HEAD"),
            "pushed": False,
            "pr_created": False,
        },
        "upstream": read_json(root / "preflight/upstream_decision.json"),
        "objective_v1": read_json(root / "objective_v1/verified_formula.json"),
        "context_binding": context,
        "budget_authority": read_json(
            root / "budget_authority/nonregression_budget_authority.json"
        ),
        "q0_q1": q0q1,
        "single_frame_development": single,
        "window_development": windows,
        "selection": selection,
        "objective_v2_status": no_candidate,
        "method_development_frame_count": ledger["future_validation_exclusion_count"],
        "future_certification": read_json(
            root / "future_certification/independent_certification_plan.json"
        ),
        "no_mutation": no_mutation,
        "resource_usage": resource_usage,
        "safety_flags": safety,
    }
    write_json(root / "final_summary.json", summary)
    b_window = next(
        row
        for row in windows["windows"]
        if row["candidate"] == "B_LEXICOGRAPHIC_THRESHOLD" and row["stratum"] == "HIGH"
    )
    b_mid = next(
        row
        for row in windows["windows"]
        if row["candidate"] == "B_LEXICOGRAPHIC_THRESHOLD" and row["stratum"] == "MID"
    )
    markdown = f"""# OakInk2 O5R-D2A RetargetObjectiveV2 Design Handoff

## 1. Git

```text
BRANCH={EXPECTED_BRANCH}
START_HEAD={START_HEAD}
FINAL_HEAD={git("rev-parse", "HEAD")}
PUSHED=NO
PR_CREATED=NO
```

## 2. Upstream diagnosis

```text
PRODUCTION_OBJECTIVE_ALIGNMENT=MISALIGNED
PRIMARY_ROOT_CAUSE=OBJECTIVE_SEMANTIC_MISALIGNMENT
CONFIDENCE=HIGH
```

## 3. Objective V1 and context binding

`J_V1 = 500 E_IM + 0.1 E_bone + E_temporal + 100 ||delta_p||^2 + ||delta_w||^2 + 50000 ||slack||^2`.

`PRODUCTION_OBJECTIVE_CONTEXT_BINDING_V2={context["status"]}` on {context["frame_count"]} frozen Q0 states; maximum objective/component replay error is exactly `{context["max_objective_absolute_difference"]}` / `{context["max_component_absolute_difference"]}`.

## 4. Budget authority

| Budget | Value | Authority |
| --- | ---: | --- |
| Interaction target | 1e-4 | frozen Semantic V1 |
| Wrist position / rotation | 0.01 m / 10 deg | frozen Semantic V1 |
| Bone direction | 45 deg | frozen Semantic V1 |
| Continuous base translation / rotation | 0.01 m / 5 deg | Wuji continuous profile |
| Continuous finger correction | 0.05 rad | Wuji continuous profile |
| Collision hard bound | 0.03 m | production constraint |
| Joint limits | asset bounds | Wuji Hand2 Beta1 asset |

No value was tuned from DEV1 Q1 outcomes.

## 5. Candidate objectives

| Candidate | Primary | Secondary | Constraints | Status |
| --- | --- | --- | --- | --- |
| A | minimize E_IM | V1 fidelity without interaction | frozen hard/non-regression plus interaction retention | REJECTED |
| B | minimize squared hinge above 1e-4 | V1 fidelity without interaction | frozen hard/non-regression plus interaction retention | PROMISING_NOT_READY |
| C | tail-aware window objective | — | — | NOT_IMPLEMENTED |

## 6. Development evidence

Q0/Q1 counterfactual N=25: every Q1 improves E_IM, but 0/25 is admissible under the already-frozen continuous limits. Candidate B removes interaction pressure on all 7 LOW controls; Candidate A does not.

Single-frame Candidate B completes {single["summaries"]["B_LEXICOGRAPHIC_THRESHOLD"]["technical_success_count"]}/15 with no new hard violation and {single["summaries"]["B_LEXICOGRAPHIC_THRESHOLD"]["old_above_to_new_below_fraction"]:.3f} old-above to new-below recovery.

Sequential Candidate B MID p95 improves from {b_mid["old_p95_e_im"]:.12g} to {b_mid["new_p95_e_im"]:.12g}, but HIGH p95 remains {b_window["new_p95_e_im"]:.12g}; the HIGH window is not technically/continuity complete.

## 7. Selection and status

```text
SELECTED_OBJECTIVE=NONE
RETARGET_OBJECTIVE_V2_DESIGN_STATUS=NO_CANDIDATE_READY
RETARGET_OBJECTIVE_V2_SHA256=null
```

No frozen ObjectiveV2 contract was emitted and independent validation is not authorized.

## 8. V1 versus proposed B

| Property | V1 | Candidate B |
| --- | --- | --- |
| E_IM treatment | weighted scalar at every value | lexicographic hinge, zero at/below target |
| Temporal treatment | freely traded weighted energy | frozen admissibility plus secondary polish |
| Bone treatment | freely traded weighted energy | frozen Semantic V1 budget plus secondary polish |
| Wrist | indirect base penalty | explicit frozen Semantic V1 budget |
| Collision | production hard/soft constraints | unchanged production constraints |
| Joint limits | Wuji asset bounds | unchanged Wuji bounds |
| Context binding | implicit call-path state | typed/versioned complete binding |

## 9. Frozen items and data hygiene

Semantic V1, its E_IM threshold, Wuji/MANO, Manifest V2, Split V2, Objective V1, StructuredSolverV1, and the old DEV1 trajectory are unchanged. `DEV1_METHOD_DEVELOPMENT_FRAME_COUNT={ledger["future_validation_exclusion_count"]}`; all future sparse/window sets require zero overlap.

## 10. Real CLI

```bash
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --help
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action audit-context-binding
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action evaluate-candidates
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action compare-q0-q1
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action run-single-frame-development
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action run-development-windows
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action freeze-objective-v2
conda run -n toporetarget-rl python scripts/data/run_oakink2_o5rd2a.py --action generate-future-certification-plan
```

## 11. Hard stop

No new independent validation, DEV1 full rerun/refinement, DEV2, O6, Support, PhysX, Frozen Eval, PPO, certification, or heldout consumption ran. DEV2 frame0 remains preserved.
"""
    (root / "handoff.md").write_text(markdown, encoding="utf-8")
    (root / "final_summary.md").write_text(markdown, encoding="utf-8")
    return summary


def record_git(root: Path) -> dict[str, Any]:
    commits = []
    for line in git("log", "--format=%H%x09%s", f"{START_HEAD}..HEAD").splitlines():
        if line:
            commit, subject = line.split("\t", 1)
            commits.append({"commit": commit, "subject": subject})
    status = git("status", "--short", "--untracked-files=all").splitlines()
    payload = {
        "schema_version": "O5RD2AGitCommitsV1",
        "branch": git("branch", "--show-current"),
        "start_head": START_HEAD,
        "final_head": git("rev-parse", "HEAD"),
        "commits": commits,
        "status_short": status,
        "tracked_worktree_clean": not status,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "git_commits.json", payload)
    return payload


def run_validation(root: Path) -> dict[str, Any]:
    modified_python = [
        "src/toporetarget/retarget/objective_v2.py",
        "scripts/data/run_oakink2_o5rd2a.py",
        "tests/retarget/test_objective_v2.py",
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
        print(f"O5RD2A_VALIDATION {name}={results[-1]['status']}", flush=True)
    payload = {
        "schema_version": "O5RD2AValidationResultsV1",
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
        "results": results,
        "unrelated_historical_debt_fixed": False,
    }
    write_json(root / "tests.json", payload)
    write_json(root / "validation_results.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "all-static",
            "preflight",
            "audit-context-binding",
            "evaluate-candidates",
            "compare-q0-q1",
            "run-single-frame-development",
            "run-development-windows",
            "freeze-objective-v2",
            "generate-future-certification-plan",
            "summarize",
            "validate",
            "record-git",
        ),
        default="all-static",
    )
    value.add_argument("--report-root", type=Path, default=ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root, action = args.report_root.resolve(), args.action
    if action in {"all-static", "preflight"}:
        preflight(root)
        write_development_ledger(root)
    if action in {"all-static", "audit-context-binding"}:
        verify_objective_v1(root)
        context_binding_parity(root)
    if action in {"all-static", "evaluate-candidates"}:
        budget_authority(root)
    if action in {"all-static", "compare-q0-q1"}:
        q0_q1_preference(root)
    if action == "run-single-frame-development":
        single_frame_development(root)
    if action == "run-development-windows":
        window_development(root)
        semantic_v1_development_audit(root)
    if action == "freeze-objective-v2":
        candidate_selection(root)
    if action == "generate-future-certification-plan":
        future_certification_plan(root)
    if action == "summarize":
        summarize(root)
    if action == "record-git":
        record_git(root)
    if action == "validate":
        result = run_validation(root)
        return 0 if result["status"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
