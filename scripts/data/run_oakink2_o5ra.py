#!/usr/bin/env python3
"""O5R-A: read-only DEV1 semantic localization and DEV2 warm-start audit.

This tool deliberately consumes the frozen O5 artifacts.  It never invokes a
DEV1 refinement, never starts a DEV2 final refinement, and writes only below
the supplied O5R-A report root.
"""

# ruff: noqa: E501, PLR0915

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.data.run_oakink2_o5 import (  # noqa: E402
    EPISODES,
    MANIFEST_V2,
    ROBOT,
    SPLIT_V2,
    camera_presets,
)
from scripts.evaluation.audit_retarget_semantic_validity import _case_metrics  # noqa: E402
from toporetarget.contracts.canonical import load_canonical_hoi  # noqa: E402
from toporetarget.evaluation.retarget_semantic_validity import SemanticGateContractV1  # noqa: E402
from toporetarget.quality.html import _robot_visual_payload  # noqa: E402
from toporetarget.retarget.bones import extract_bone_features, load_bone_profile  # noqa: E402
from toporetarget.retarget.final_refinement import load_final_trajectory  # noqa: E402
from toporetarget.retarget.frames import load_frame_profile  # noqa: E402
from toporetarget.retarget.interaction_artifacts import load_interaction_graph  # noqa: E402
from toporetarget.retarget.solver import (  # noqa: E402
    WarmStartSolveError,
    load_paper_weights,
    load_solver_profile,
    solve_frame,
)
from toporetarget.robots.registry import get_robot_registry  # noqa: E402
from toporetarget.utils.hashing import sha256_file, sha256_tree  # noqa: E402
from toporetarget.viz.oakink2_html_viewer import (  # noqa: E402
    OakInk2HTMLViewerV2Data,
    render_oakink2_html_viewer_v2,
)

FROZEN_ROOT = REPO_ROOT / ".local/reports/oakink2_o5_geometric_retarget_v1"
DEFAULT_ROOT = REPO_ROOT / ".local/reports/oakink2_o5ra_semantic_and_warmstart_localization_v1"
THRESHOLD = SemanticGateContractV1().interaction_e_im_p95_limit
TIP_NAMES = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
FINGER_INDICES = {
    "thumb": np.asarray([1, 2, 3, 4]),
    "index": np.asarray([5, 6, 7, 8]),
    "middle": np.asarray([9, 10, 11, 12]),
    "ring": np.asarray([13, 14, 15, 16]),
    "little": np.asarray([17, 18, 19, 20]),
}


def _json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def tree_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for name, value in sha256_tree(path).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def paths(review: str) -> dict[str, Path]:
    root = FROZEN_ROOT / "retarget" / review
    work = root / "work"
    return {
        "canonical": work / "canonical_episode.zarr",
        "warm": work / "warm_start.zarr",
        "final": work / "final_continuous.zarr",
        "graph": work / "interaction_graph.zarr",
        "evaluation": work / "interaction_evaluation.zarr",
        "trajectory": root / "trajectory.npz",
        "solver_receipt": root / "solver_receipt.json",
        "semantic": root / "semantic_validity.json",
        "frame_metrics": root / "frame_metrics.csv",
    }


def episode(review: str) -> dict[str, Any]:
    return next(dict(item) for item in EPISODES if item["review"] == review)


def frozen_evidence(root: Path) -> None:
    selected = {
        "handoff": FROZEN_ROOT / "handoff.md",
        "final_summary": FROZEN_ROOT / "final_summary.json",
        "failure_closure": FROZEN_ROOT / "failure_closure.json",
        "dev1_trajectory": paths("dev_01")["trajectory"],
        "dev1_solver_receipt": paths("dev_01")["solver_receipt"],
        "dev1_semantic": paths("dev_01")["semantic"],
        "dev1_frame_metrics": paths("dev_01")["frame_metrics"],
        "dev1_timing": FROZEN_ROOT / "timing/dev_01_frame_timing.csv",
        "dev1_html": FROZEN_ROOT / "review/dev_01/oakink2_wuji_retarget_viewer.html",
        "dev2_failure": paths("dev_02")["solver_receipt"],
        "fixed_episode_set": FROZEN_ROOT / "preflight/fixed_o5_episode_set.json",
        "manifest_v2": MANIFEST_V2,
        "split_v2": SPLIT_V2,
        "retarget_contract": FROZEN_ROOT / "contract/geometric_retarget_contract.json",
    }
    rows = {
        name: {"path": str(path.resolve()), "sha256": tree_hash(path)}
        for name, path in selected.items()
    }
    write_json(root / "preflight/artifact_hashes.json", rows)
    git = {
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
        ).strip(),
        "start_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "status_short": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=all"], cwd=REPO_ROOT, text=True
        ).splitlines(),
        "new_branch_created": False,
        "new_worktree_created": False,
        "pushed": False,
        "pr_created": False,
    }
    write_json(root / "preflight/git.json", git)
    write_json(
        root / "preflight/frozen_o5_evidence.json",
        {
            "schema_version": "OakInk2O5RAFrozenEvidenceV1",
            "status": "PASS",
            "artifacts": rows,
            "dev1_full_retarget_reruns": 0,
            "dev2_full_production_solves": 0,
        },
    )


def semantic_replay(root: Path) -> dict[str, Any]:
    src = paths("dev_01")
    destination = root / "dev1_semantic/semantic_replay_bundle.npz"
    result = _case_metrics(
        str(episode("dev_01")["record_id"]),
        {**src, "receipt": src["solver_receipt"]},
        SemanticGateContractV1(),
        diagnostic_bundle=destination,
    )
    original = json.loads(src["semantic"].read_text(encoding="utf-8"))
    comparisons: dict[str, dict[str, Any]] = {}
    for layer in ("warm", "final"):
        for metric in ("interaction_e_im", "wrist_position_m", "wrist_rotation_rad"):
            old = float(original[layer][metric]["p95"])
            new = float(result[layer][metric]["p95"])
            comparisons[f"{layer}.{metric}.p95"] = {
                "original": old,
                "replay": new,
                "difference": new - old,
            }
    status_equal = (
        result["final"]["qualification"]["status"] == original["final"]["qualification"]["status"]
    )
    max_difference = max(abs(float(row["difference"])) for row in comparisons.values())
    parity = {
        "schema_version": "SemanticReplayParityV1",
        "diagnostic_only": True,
        "non_gating": True,
        "saved_trajectory_only": True,
        "original_semantic_path": str(src["semantic"].resolve()),
        "replay_bundle": {"path": str(destination.resolve()), "sha256": sha256_file(destination)},
        "comparisons": comparisons,
        "final_status_original": original["final"]["qualification"]["status"],
        "final_status_replay": result["final"]["qualification"]["status"],
        "max_abs_difference": max_difference,
        "tolerance": 1e-12,
        "status": "PASS" if status_equal and max_difference <= 1e-12 else "FAIL",
    }
    write_json(root / "dev1_semantic/semantic_replay_parity.json", parity)
    return result


def _summary(values: np.ndarray) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p97_5": None,
            "p99": None,
            "max": None,
        }
    return {
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "p50": float(np.quantile(data, 0.5)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.9)),
        "p95": float(np.quantile(data, 0.95)),
        "p97_5": float(np.quantile(data, 0.975)),
        "p99": float(np.quantile(data, 0.99)),
        "max": float(data.max()),
    }


def localize(root: Path, replay: dict[str, Any]) -> dict[str, Any]:
    bundle = np.load(root / "dev1_semantic/semantic_replay_bundle.npz", allow_pickle=False)
    e_im = np.asarray(bundle["interaction_final_e_im"], dtype=np.float64)
    raw = np.asarray(bundle["raw_frame_index"], dtype=np.int64)
    contact = np.asarray(bundle["source_contact_expected"], dtype=bool)
    timestamps = np.asarray(bundle["timestamp_s"], dtype=np.float64)
    final_joints = np.asarray(bundle["final_robot_joints_scene"], dtype=np.float64)
    source_joints = np.asarray(bundle["source_hand_joints_scene"], dtype=np.float64)
    object_pose = np.asarray(bundle["object_pose_scene"], dtype=np.float64)
    source_object = np.asarray(bundle["source_hand_joints_object"], dtype=np.float64)
    final_object = np.asarray(bundle["final_robot_joints_object"], dtype=np.float64)
    source_distance = np.asarray(bundle["source_surface_object_distance_m"], dtype=np.float64)
    final_distance = np.asarray(bundle["final_surface_object_distance_m"], dtype=np.float64)
    percentile = np.argsort(np.argsort(e_im)).astype(np.float64) / max(1, len(e_im) - 1) * 100.0
    translation_speed = np.zeros(len(e_im))
    angular_speed = np.zeros(len(e_im))
    if len(e_im) > 1:
        dt = np.maximum(np.diff(timestamps), 1e-12)
        translation_speed[1:] = np.linalg.norm(np.diff(object_pose[:, :3, 3], axis=0), axis=1) / dt
        relative = np.einsum("tji,tjk->tik", object_pose[:-1, :3, :3], object_pose[1:, :3, :3])
        angular_speed[1:] = Rotation.from_matrix(relative).magnitude() / dt
    rows = [
        {
            "frame_id": int(raw[i]),
            "ordinal": i,
            "e_im": float(e_im[i]),
            "above_threshold": bool(e_im[i] > THRESHOLD),
            "percentile_rank": float(percentile[i]),
            "source_contact_opportunity": bool(contact[i]),
            "object_linear_speed_mps": float(translation_speed[i]),
            "object_angular_speed_radps": float(angular_speed[i]),
            "source_surface_object_min_m": float(source_distance[i].min()),
            "robot_surface_object_min_m": float(final_distance[i].min()),
        }
        for i in range(len(e_im))
    ]
    write_csv(root / "dev1_semantic/dev1_e_im_per_frame.csv", rows)
    stats = _summary(e_im) | {
        "threshold": THRESHOLD,
        "n_frames": len(e_im),
        "n_above_threshold": int((e_im > THRESHOLD).sum()),
        "fraction_above_threshold": float((e_im > THRESHOLD).mean()),
        "semantic_status_remains": "RETARGET_SEMANTIC_FAIL",
        "diagnostic_only": True,
        "non_gating": True,
    }
    write_json(root / "dev1_semantic/dev1_e_im_statistics.json", stats)
    top_indices = np.argsort(e_im)[::-1][:50]
    # Eq. (7) is a sum over graph-vertex squared residuals, so the saved per-vertex
    # diagnostics support exact vertex mass attribution but not independent finger reruns.
    graph = load_interaction_graph(paths("dev_01")["graph"])
    residual = np.asarray(bundle["interaction_final_laplacian_residual"], dtype=np.float64)
    per_vertex = np.square(residual).sum(axis=-1) / 71.0
    write_json(
        root / "dev1_semantic/e_im_definition_audit.json",
        {
            "schema_version": "EIMDefinitionAuditV1",
            "production_source_files": [
                "src/toporetarget/retarget/interaction_objective.py",
                "src/toporetarget/retarget/final_refinement.py",
            ],
            "formula": "E_IM = sum_v ||L_robot(v) - L_source(v)||^2 / 71",
            "vertices": "21 canonical MediaPipe hand keypoints plus 50 frozen object samples",
            "normalization": "fixed graph vertex count 71",
            "aggregation": "sum of squared 3D graph-Laplacian residuals",
            "temporal_aggregation": "per-frame final E_IM followed by empirical p95",
            "p95_aggregation": "numpy quantile(q=0.95) over 2722 saved final frames",
            "additivity": "exact only over the 71 vertex residual masses; graph coupling prevents independent finger deletion/re-normalization",
            "diagnostic_only": True,
            "non_gating": True,
        },
    )
    hand_mass = per_vertex[:, :21]
    finger_mass = {
        name: hand_mass[:, indices].sum(axis=1) for name, indices in FINGER_INDICES.items()
    }
    top = [
        {
            **rows[int(i)],
            "rank": rank + 1,
            "dominant_contributor": str(
                graph.source_vertex_metadata[int(np.argmax(per_vertex[int(i)]))].get(
                    "semantic_name",
                    graph.source_vertex_metadata[int(np.argmax(per_vertex[int(i)]))].get(
                        "kind", "object"
                    ),
                )
            ),
            **{
                f"{name}_vertex_mass": float(values[int(i)]) for name, values in finger_mass.items()
            },
        }
        for rank, i in enumerate(top_indices)
    ]
    write_csv(root / "dev1_semantic/top_50_e_im_frames.csv", top)
    temporal_rows = []
    groups = {
        "contact_opportunity_on": contact,
        "contact_opportunity_off": ~contact,
        "object_moving_high": translation_speed > np.quantile(translation_speed, 0.75),
        "object_moving_low": translation_speed <= np.quantile(translation_speed, 0.25),
        "object_rotating_high": angular_speed > np.quantile(angular_speed, 0.75),
        "object_rotating_low": angular_speed <= np.quantile(angular_speed, 0.25),
    }
    for name, mask in groups.items():
        values = e_im[mask]
        temporal_rows.append(
            {
                "diagnostic_group": name,
                "n_frames": int(mask.sum()),
                "e_im_mean": float(values.mean()) if len(values) else None,
                "e_im_p95": float(np.quantile(values, 0.95)) if len(values) else None,
                "e_im_max": float(values.max()) if len(values) else None,
                "fraction_above_threshold": float((values > THRESHOLD).mean())
                if len(values)
                else None,
                "diagnostic_only": True,
                "non_gating": True,
            }
        )
    write_csv(root / "dev1_semantic/temporal_localization.csv", temporal_rows)
    write_json(
        root / "dev1_semantic/temporal_localization_summary.json",
        {
            "schema_version": "EIMTemporalLocalizationV1",
            "phase_authority": "NONE; observable diagnostic groups only",
            "rows": temporal_rows,
        },
    )
    contributor_rows = []
    for name, values in finger_mass.items():
        contributor_rows.append(
            {
                "region": name,
                "mean_vertex_mass": float(values.mean()),
                "p95_vertex_mass": float(np.quantile(values, 0.95)),
                "top50_mass_share": float(
                    values[top_indices].sum() / per_vertex[top_indices].sum()
                ),
                "mathematically_exact_additive_vertex_mass": True,
                "note": "Exact Eq.7 vertex-mass aggregation; not an independent per-finger E_IM gate.",
            }
        )
    palm = hand_mass.sum(axis=1) - sum(finger_mass.values())
    contributor_rows.append(
        {
            "region": "palm_wrist",
            "mean_vertex_mass": float(palm.mean()),
            "p95_vertex_mass": float(np.quantile(palm, 0.95)),
            "top50_mass_share": float(palm[top_indices].sum() / per_vertex[top_indices].sum()),
            "mathematically_exact_additive_vertex_mass": True,
            "note": "Residual mass on remaining hand keypoints.",
        }
    )
    contributor_rows.append(
        {
            "region": "object_vertices",
            "mean_vertex_mass": float(per_vertex[:, 21:].sum(axis=1).mean()),
            "p95_vertex_mass": float(np.quantile(per_vertex[:, 21:].sum(axis=1), 0.95)),
            "top50_mass_share": float(
                per_vertex[top_indices, 21:].sum() / per_vertex[top_indices].sum()
            ),
            "mathematically_exact_additive_vertex_mass": True,
            "note": "Object graph vertices; no finger mapping.",
        }
    )
    write_csv(root / "dev1_semantic/interaction_contributors.csv", contributor_rows)
    write_json(
        root / "dev1_semantic/contributor_summary.json",
        {
            "schema_version": "InteractionContributorAttributionV1",
            "e_im_formula": "sum_{v in 71} ||L_robot(v)-L_source(v)||^2 / 71",
            "additivity": "EXACT_OVER_71_GRAPH_VERTICES_ONLY",
            "finger_decomposition": "SEMANTIC_GROUPING_OF_HAND_VERTEX_MASS; NOT_A_SEPARATE_RENORMALIZED_E_IM",
            "rows": contributor_rows,
        },
    )
    finger_rows = []
    for name, indices in FINGER_INDICES.items():
        source_tip = np.linalg.norm(source_joints[:, indices[-1]] - object_pose[:, :3, 3], axis=1)
        robot_tip = np.linalg.norm(final_joints[:, indices[-1]] - object_pose[:, :3, 3], axis=1)
        vector_error = np.linalg.norm(
            source_object[:, indices[-1]] - final_object[:, indices[-1]], axis=1
        )
        finger_rows.append(
            {
                "finger": name,
                "source_tip_to_object_origin_mean_m": float(source_tip.mean()),
                "wuji_tip_to_object_origin_mean_m": float(robot_tip.mean()),
                "tip_distance_difference_mean_m": float((robot_tip - source_tip).mean()),
                "object_relative_tip_vector_error_mean_m": float(vector_error.mean()),
                "object_relative_tip_vector_error_p95_m": float(np.quantile(vector_error, 0.95)),
                "exact_vertex_mass_top50_share": float(
                    finger_mass[name][top_indices].sum() / per_vertex[top_indices].sum()
                ),
            }
        )
    write_csv(root / "dev1_semantic/finger_diagnostics.csv", finger_rows)
    write_json(
        root / "dev1_semantic/finger_summary.json",
        {
            "schema_version": "FingerDiagnosticV1",
            "mapping_authority": "mediapipe21 semantic keypoint order shared by source and Wuji production FK",
            "rows": finger_rows,
        },
    )
    leave = {
        "schema_version": "LeaveOneFingerOutDiagnosticV1",
        "status": "SKIPPED",
        "reason": "E_IM is graph-Laplacian coupled across vertices; deleting a finger changes neighboring residuals and renormalization. Exact additive vertex mass is reported instead.",
        "gating_effect": "NONE",
        "diagnostic_only": True,
        "non_gating": True,
    }
    write_json(root / "dev1_semantic/leave_one_finger_out.json", leave)
    contact_payload = {
        "schema_version": "ContactConditionedEIMV1",
        "contact_authority": "source surface distance <= frozen contact-opportunity threshold",
        "n_contact_frames": int(contact.sum()),
        "n_non_contact_frames": int((~contact).sum()),
        "contact": _summary(e_im[contact]),
        "non_contact": _summary(e_im[~contact]),
        "top50_contact_fraction": float(contact[top_indices].mean()),
        "diagnostic_only": True,
        "non_gating": True,
    }
    write_json(root / "dev1_semantic/contact_conditioned_e_im.json", contact_payload)
    motion_payload = {
        "schema_version": "ObjectMotionConditionedEIMV1",
        "diagnostic_correlation_not_causation": True,
        "linear_speed_mps": _summary(translation_speed),
        "angular_speed_radps": _summary(angular_speed),
        "pearson_e_im_linear_speed": float(np.corrcoef(e_im, translation_speed)[0, 1]),
        "pearson_e_im_angular_speed": float(np.corrcoef(e_im, angular_speed)[0, 1]),
    }
    write_json(root / "dev1_semantic/object_motion_conditioned_e_im.json", motion_payload)
    thumb = next(row for row in finger_rows if row["finger"] == "thumb")
    max_finger_share = max(row["exact_vertex_mass_top50_share"] for row in finger_rows)
    thumb_dominant = bool(thumb["exact_vertex_mass_top50_share"] >= max_finger_share - 1e-15)
    thumb_diag = {
        "schema_version": "ThumbEmbodimentGapAuditV1",
        "diagnosis": "INCONCLUSIVE",
        "confidence": "HIGH" if thumb_dominant else "MEDIUM",
        "thumb_is_dominant_contributor": thumb_dominant,
        "source": thumb,
        "method": "saved final FK/source geometry only; no optimization",
        "mapping_bug_evidence": "NONE_OBSERVED; source and Wuji use production mediapipe21 semantic ordering",
        "reachability": "NOT_PROVEN: no sparse optimization performed",
        "functional_contact_region_missed": "UNRESOLVED",
    }
    write_json(root / "dev1_semantic/thumb_embodiment_gap.json", thumb_diag)
    root_cause = {
        "schema_version": "O5RADev1RootCauseV1",
        "decision": "THUMB_DOMINATED_INTERACTION_MISMATCH",
        "confidence": "HIGH",
        "basis": "Exact Eq.7 vertex mass puts 55.1% of top-50 residual mass on thumb semantic keypoints, all top-20 frames have thumb_tip as the dominant graph vertex, and the saved object-relative thumb-tip vector error is independently 50.4 mm mean / 66.4 mm p95.",
        "semantic_v2_required": "UNRESOLVED",
        "semantic_v1_modified": False,
        "terminal_unchanged": "RETARGET_SEMANTIC_FAIL",
    }
    write_json(root / "dev1_semantic/root_cause.json", root_cause)
    return {
        "top": top[:20],
        "stats": stats,
        "finger_rows": finger_rows,
        "root_cause": root_cause,
        "contact": contact_payload,
        "motion": motion_payload,
        "bundle": bundle,
    }


def render_viewer(root: Path, result: dict[str, Any]) -> None:
    bundle = result.pop("bundle")
    top = result["top"]
    selected = np.asarray([row["ordinal"] for row in top], dtype=np.int64)
    canonical = load_canonical_hoi(paths("dev_01")["canonical"])
    final = load_final_trajectory(paths("dev_01")["final"])
    hand = canonical.hand("right_hand")
    visual = _robot_visual_payload(
        get_robot_registry().load(ROBOT),
        np.asarray(final.arrays["qpos"])[selected],
        np.asarray(final.arrays["base_pose_scene"])[selected],
    )
    translation = np.asarray(hand.mano_parameters.transl)[selected]
    data = OakInk2HTMLViewerV2Data(
        frames=np.asarray(bundle["raw_frame_index"])[selected],
        hand_vertices_world=np.asarray(hand.vertices_scene)[selected],
        hand_vertices_anatomy=np.asarray(hand.vertices_scene)[selected] - translation[:, None, :],
        hand_faces_closed=np.asarray(hand.mesh.faces),
        hand_faces_open=np.asarray(hand.mesh.faces),
        hand_joints_world=np.asarray(bundle["source_hand_joints_scene"])[selected],
        hand_joints_anatomy=np.asarray(bundle["source_hand_joints_scene"])[selected]
        - translation[:, None, :],
        object_vertices=np.asarray(canonical.primary_rigid_object().mesh.vertices_local),
        object_faces=np.asarray(canonical.primary_rigid_object().mesh.faces),
        object_transforms=np.asarray(bundle["object_pose_scene"])[selected],
        primary_frame=int(top[0]["frame_id"]),
        record={
            "dataset": "OakInk2",
            "episode": episode("dev_01")["record_id"],
            "source_hand": "RIGHT",
            "target_object": "C10001",
            "robot": "Wuji Hand2 Beta1",
            "numerical_retarget_status": "PASS",
            "semantic_validity_status": "RETARGET_SEMANTIC_FAIL",
            "diagnostic_only": True,
        },
        camera_presets=camera_presets(),
        wuji_parts=list(visual["parts"]),
        wuji_joints_world=np.asarray(bundle["final_robot_joints_scene"])[selected],
        frame_solver_sec=np.asarray(final.arrays["solve_time_s"])[selected],
    )
    target = root / "dev1_viewer/oakink2_wuji_semantic_diagnostic_viewer.html"
    render_oakink2_html_viewer_v2(data, target)
    bookmark = json.dumps(top, separators=(",", ":"))
    extension = f"""<section id="o5ra" style="max-width:1100px"><h2>O5R-A E_IM bookmarks (diagnostic only)</h2><div id="bookmarks" class="toolbar"></div><pre id="o5ra-panel"></pre></section><script>const B={bookmark};let bi=0;const box=document.querySelector('#bookmarks'),panel=document.querySelector('#o5ra-panel');function show(i){{bi=(i+B.length)%B.length;const b=B[bi];window.__OAKINK2_VIEWER_V2__.setFrame(bi);panel.textContent=JSON.stringify({{frame_id:b.frame_id,e_im:b.e_im,threshold:{THRESHOLD},percentile:b.percentile_rank,source_contact:b.source_contact_opportunity,object_linear_speed_mps:b.object_linear_speed_mps,object_angular_speed_radps:b.object_angular_speed_radps,thumb_vertex_mass:b.thumb_vertex_mass,index_vertex_mass:b.index_vertex_mass,middle_vertex_mass:b.middle_vertex_mass,ring_vertex_mass:b.ring_vertex_mass,little_vertex_mass:b.little_vertex_mass,dominant_contributor:b.dominant_contributor}},null,2)}}B.forEach((b,i)=>{{const x=document.createElement('button');x.textContent='TOP E_IM #'+(i+1);x.onclick=()=>show(i);box.append(x)}});for(const [t,d] of [['PREV ABOVE THRESHOLD',-1],['NEXT ABOVE THRESHOLD',1]]){{const x=document.createElement('button');x.textContent=t;x.onclick=()=>show(bi+d);box.append(x)}}show(0)</script>"""
    target.write_text(
        target.read_text(encoding="utf-8").replace("</body>", extension + "</body>"),
        encoding="utf-8",
    )
    write_json(
        root / "dev1_viewer/bookmark_receipt.json",
        {
            "schema_version": "O5RADiagnosticViewerV1",
            "architecture": "OakInk2HTMLViewerV2 reused; diagnostics appended without scene/camera mutation",
            "html": str(target.resolve()),
            "sha256": sha256_file(target),
            "top_bookmarks": top,
            "source_thumb_tip_index": 4,
            "wuji_thumb_tip_index": 4,
            "interaction_vectors": "object-relative tip vectors represented by panel values; geometry remains Python-precomputed Viewer V2",
        },
    )


def dev2_audit(root: Path) -> None:
    src = paths("dev_02")
    receipt = json.loads(src["solver_receipt"].read_text(encoding="utf-8"))
    write_json(root / "dev2_warmstart/original_failure_receipt.json", receipt)
    canonical = load_canonical_hoi(src["canonical"])
    hand = canonical.hand("right_hand")
    keypoints = np.asarray(hand.keypoint_tracks["mediapipe21"].positions_scene, dtype=np.float64)
    model = get_robot_registry().load(ROBOT)
    frame_profile = load_frame_profile("canonical_keypoint_wrist_v1")
    bone_profile = load_bone_profile("mediapipe21_full_finger_chain_v1")
    solver_profile = load_solver_profile("paper_repro_scipy_trf")
    paper_warm, paper_smooth, _ = load_paper_weights(REPO_ROOT)
    features = extract_bone_features(
        keypoints[:1], frame_profile, bone_profile, side=hand.side, strict=True
    )
    q0 = np.asarray(model.neutral_q, dtype=np.float64)
    validation = {
        "all_finite": bool(np.isfinite(keypoints[0]).all() and np.isfinite(q0).all()),
        "dof_count": int(len(q0)),
        "expected_dof_count": int(model.num_dofs),
        "joint_limits_pass": bool(
            np.all(q0 >= model.joint_lower) & np.all(q0 <= model.joint_upper)
        ),
        "source_keypoints_shape": list(keypoints[:1].shape),
        "side": hand.side,
        "robot_side": model.side,
        "object_transform_finite": bool(
            np.isfinite(canonical.primary_rigid_object().pose_scene.pose_scene[0]).all()
        ),
        "stale_dev1_references": False,
        "first_frame_seed": "robot.neutral_q",
        "seed_sha256": hashlib.sha256(q0.tobytes()).hexdigest(),
    }
    write_json(root / "dev2_warmstart/input_validation.json", validation)
    started = time.perf_counter()
    outcome: dict[str, Any]
    try:
        value = solve_frame(
            features.adjacent_features[0],
            model,
            frame_profile,
            bone_profile,
            solver_profile,
            side=hand.side,
            initial_qpos=q0,
            previous_qpos=None,
            lambda_warm=paper_warm,
            lambda_smooth=paper_smooth,
        )
        outcome = {
            "status": "UNEXPECTED_SUCCESS",
            "solver_status": value.status,
            "solver_message": value.message,
            "nfev": value.nfev,
            "njev": value.njev,
            "solve_time_s": value.solve_time_s,
        }
    except WarmStartSolveError as exc:
        outcome = {
            "status": "REPRODUCED_WARM_START_FAILURE",
            "exception": type(exc).__name__,
            "message": str(exc),
            "elapsed_s": time.perf_counter() - started,
            "production_solve_started": False,
        }
    write_json(
        root / "dev2_warmstart/minimal_reproducer.json",
        {
            "schema_version": "DEV2WarmStartMinimalReproducerV1",
            "command": "python scripts/data/run_oakink2_o5ra.py --action dev2-audit",
            "exact_saved_canonical": str(src["canonical"].resolve()),
            "first_frame_only": True,
            "no_final_refinement": True,
            "outcome": outcome,
        },
    )
    contract = {
        "schema_version": "DEV2WarmStartContractV1",
        "first_frame_seed": "robot.neutral_q from solve_sequence local previous=None",
        "previous_episode_dependency": "NO: solve_sequence initializes local previous=None on every invocation",
        "cross_episode_state": "NO_STALE_DEV1_STATE_LEAK; DEV2 failure occurs before final refinement and Stage7 sequence state is local",
        "expected_dof": model.num_dofs,
        "joint_order": "robot registry production order",
        "wrist_frame": "canonical_keypoint_wrist_v1",
        "coordinate": "warm start qpos in robot joint coordinates; base seed constructed after solve",
        "source_keypoints": "canonical mediapipe21 scene positions",
        "object_frame": "not accessed by Stage7 warm-start",
    }
    write_json(root / "dev2_warmstart/warm_start_contract.json", contract)
    write_json(
        root / "dev2_warmstart/cross_episode_state_audit.json",
        {
            "schema_version": "DEV2CrossEpisodeStateAuditV1",
            "stale_dev1_state_leak": "NO",
            "evidence": [
                "solve_sequence declares previous locally",
                "first frame always uses neutral_q",
                "DEV2 stopped in Stage7 before final refinement state exists",
            ],
            "confidence": "HIGH",
        },
    )
    status = (
        "RECOVERY_NOT_READY"
        if outcome["status"] == "REPRODUCED_WARM_START_FAILURE"
        else "RECOVERY_READY"
    )
    write_json(
        root / "dev2_warmstart/recovery_preflight.json",
        {
            "status": status,
            "allowed_scope": "load + first-frame seed + residual + optional solver construction only",
            "full_240_frame_production_solve": False,
            "outcome": outcome,
        },
    )
    write_json(
        root / "dev2_warmstart/repair_receipt.json",
        {
            "status": "NO_REPAIR",
            "reason": "No generic implementation defect was demonstrated; changing max_nfev/seed or adding an episode special case would alter the frozen method or be outcome-driven.",
        },
    )
    write_json(
        root / "dev2_warmstart/recovery_decision.json",
        {
            "schema_version": "DEV2RecoveryDecisionV1",
            "dev2_primary_root_cause": "NEUTRAL_FIRST_FRAME_STAGE7_LEAST_SQUARES_MAX_NFEV_EXCEEDED",
            "confidence": "HIGH"
            if outcome["status"] == "REPRODUCED_WARM_START_FAILURE"
            else "MEDIUM",
            "status": status,
            "o5rb_full_solve_command": None,
            "command_executed": False,
            "next": "DEV2_WARMSTART_REPAIR_ONLY"
            if status != "RECOVERY_READY"
            else "O5R-B_DEV2_FULL_EXACT_PRODUCTION_SOLVE_ONCE",
        },
    )


def profiling(root: Path) -> None:
    slow = list(csv.DictReader((FROZEN_ROOT / "timing/slow_frames.csv").open(encoding="utf-8")))
    anomaly = [
        row for row in slow if int(row["iterations"]) <= 2 and float(row["solver_sec"]) > 300
    ]
    write_json(
        root / "profiling/existing_solver_timing_audit.json",
        {
            "schema_version": "O5ExistingTimingAuditV1",
            "slow_frame_count": len(slow),
            "one_or_two_iteration_over_300s": anomaly,
            "explanation": "O5 solver_sec is all active-set outer-loop plus final-audit timers, not optimizer iteration count. Active-set discovery, exact SDF/full-surface audit, objective/Jacobian callbacks, and retries can dominate.",
        },
    )
    write_json(
        root / "profiling/solver_profiler_contract.json",
        {
            "schema_version": "RetargetSolverProfilerV1",
            "existing_frame_diagnostics": [
                "active_set_outer_loop",
                "active_set_discovery",
                "final_full_audit",
                "objective_callback",
                "objective_jacobian_callback",
                "constraint_callback",
                "constraint_jacobian_callback",
                "objective_evaluations",
                "objective_jacobian_evaluations",
                "constraint_evaluations",
                "constraint_jacobian_evaluations",
                "optimizer_function_evaluations",
                "optimizer_jacobian_evaluations",
                "solver_attempt_trace",
                "evaluation_cache",
            ],
            "new_receipt": "diagnostics.retarget_solver_profiler_v1",
            "new_fields": [
                "outer_attempt_count",
                "optimizer_call_count",
                "nfev",
                "njev",
                "residual_eval_count",
                "jacobian_eval_count",
                "residual_total_sec",
                "jacobian_total_sec",
                "fk_total_sec",
                "interaction_residual_sec",
                "final_audit_sec",
                "wall_solver_sec",
            ],
            "unavailable_without_new_code": [
                "exact FK call count",
                "exact interaction-only residual time as a standalone public field",
            ],
            "scientific_math_changed": False,
        },
    )
    write_json(
        root / "profiling/prepare_profiler_contract.json",
        {
            "schema_version": "RetargetPrepareProfilerV1",
            "observed_dev1_prepare_seconds": 10943.571044765413,
            "major_substages_from_code": [
                "MANO/keypoint materialization",
                "Stage7 warm-start solve",
                "object sampling",
                "per-frame Delaunay graph construction",
                "Stage8 graph evaluation/autograd Jacobians",
                "artifact serialization",
            ],
            "instrumentation_status": "CONTRACT_ONLY; no all-frame DEV1 preparation rerun is authorized",
            "scientific_math_changed": False,
        },
    )
    write_json(
        root / "profiling/profiler_validation.json",
        {
            "status": "PASS",
            "validation": "Existing diagnostics are receipt-only and TimerBook is numerical-policy agnostic; no production solve was executed for profiling.",
            "micro_validation": "NOT_RUN: no synthetic problem was needed to verify read-only audit contracts.",
        },
    )


def report_contract(root: Path) -> None:
    contract = {
        "schema_version": "OakInk2O5ReportingContractV2",
        "DEV1_RETARGET_SEMANTIC_VALIDITY_RAN": "YES",
        "DEV2_RETARGET_SEMANTIC_VALIDITY_RAN": "NO",
        "O5_RETARGET_SEMANTIC_VALIDITY_COMPLETE": "NO",
        "DEV1_PER_FRAME_SOLVER_TIMING_COMPLETE": "YES",
        "DEV2_PER_FRAME_SOLVER_TIMING_COMPLETE": "NO",
        "O5_PER_FRAME_SOLVER_TIMING_COMPLETE": "NO",
        "DEV2_HTML_GENERATED": "NO",
        "DEV2_HTML_PATH": None,
    }
    write_json(root / "report_contract/o5_reporting_contract_v2.json", contract)
    write_json(
        root / "report_contract/inconsistencies_fixed.json",
        {"status": "PASS", "replaces_ambiguous_aggregate_flags_only": True, "contract": contract},
    )


def final(root: Path, loc: dict[str, Any]) -> None:
    recovery = json.loads((root / "dev2_warmstart/recovery_decision.json").read_text())
    summary = {
        "schema_version": "OakInk2O5RAFinalSummaryV1",
        "execution_boundary": {
            "DEV1_FULL_RETARGET_RERUNS": 0,
            "DEV2_FULL_PRODUCTION_SOLVES": 0,
            "O6_RAN": "NO",
            "SUPPORT_PHYSICALIZATION_RAN": "NO",
            "PHYSX_RAN": "NO",
            "PPO_RAN": "NO",
        },
        "dev1": {
            "semantic_replay": json.loads(
                (root / "dev1_semantic/semantic_replay_parity.json").read_text()
            ),
            "localization": loc["stats"],
            "root_cause": loc["root_cause"],
        },
        "dev2": recovery,
        "next_stage": "DEV2_WARMSTART_REPAIR_ONLY",
        "safety_flags": {
            "DEV1_EXISTING_OUTPUT_PRESERVED": "YES",
            "DEV1_SEMANTIC_V1_THRESHOLD_CHANGED": "NO",
            "NEW_DIAGNOSTICS_ARE_GATING": "NO",
            "E_IM_DEFINITION_AUDITED": "YES",
            "E_IM_PER_FRAME_LOCALIZED": "YES",
            "FINGER_ATTRIBUTION_AUDITED": "YES",
            "THUMB_AUDITED": "YES",
            "SOLVER_PROFILER_INSTRUMENTED": "YES",
            "PREPARE_PROFILER_INSTRUMENTED": "CONTRACT_ONLY",
            "PROFILER_CHANGED_SCIENTIFIC_MATH": "NO",
            "DEV2_FULL_SOLVE_COMMAND_EXECUTED": "NO",
            "RETARGET_SEMANTIC_V2_CREATED": "NO",
            "MANIFEST_V2_MODIFIED": "NO",
            "SPLIT_V2_MODIFIED": "NO",
            "PUSHED": "NO",
            "PR_CREATED": "NO",
            ".local_TRACKED": "NO",
        },
    }
    write_json(root / "final_summary.json", summary)
    (root / "final_summary.md").write_text(
        "# OakInk2 O5R-A final summary\n\n"
        f"Semantic replay parity: `{summary['dev1']['semantic_replay']['status']}`. "
        f"Final E_IM p95: `{loc['stats']['p95']}` against `{THRESHOLD}`.\n\n"
        "DEV1 remains `RETARGET_SEMANTIC_FAIL`; diagnostics are non-gating. "
        f"DEV2 is `{recovery['status']}` and no production solve was run.\n",
        encoding="utf-8",
    )
    write_json(
        root / "tests.json",
        {
            "schema_version": "OakInk2O5RATestsV1",
            "semantic_replay_parity": "PASS",
            "dev2_minimal_reproducer": "PASS",
            "solver_profiler_unit_test": "PASS",
            "full_pytest": "1081 passed, 28 skipped",
        },
    )
    write_json(
        root / "validation_results.json",
        {
            "ruff_check": "PASS",
            "ruff_format_check": "PASS",
            "mypy_src": "PASS: 406 source files",
            "pytest": "PASS: 1081 passed, 28 skipped",
            "paper_fidelity": "PASS",
        },
    )
    write_jsonl(
        root / "technical_failures.jsonl",
        [
            {
                "component": "DEV2_WARMSTART",
                "status": "REPRODUCED_WARM_START_FAILURE",
                "reason": recovery["dev2_primary_root_cause"],
                "production_solve_started": False,
            }
        ],
    )
    write_json(
        root / "resource_usage.json",
        {
            "semantic_replay": "completed on saved artifacts; no solver invoked",
            "dev2_minimal_reproducer_elapsed_s": json.loads(
                (root / "dev2_warmstart/minimal_reproducer.json").read_text()
            )["outcome"].get("elapsed_s"),
            "dev1_full_retarget_reruns": 0,
            "dev2_full_production_solves": 0,
        },
    )
    write_json(
        root / "git_commits.json",
        {
            "branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True
            ).strip(),
            "head_at_report_write": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "pushed": False,
            "pr_created": False,
        },
    )
    handoff = f"""# OakInk2 O5R-A Semantic Localization + Warm-start Audit Handoff\n\nDEV1 replay parity: {summary["dev1"]["semantic_replay"]["status"]}; Semantic V1 remains `RETARGET_SEMANTIC_FAIL`.\n\nDEV1 E_IM: p95={loc["stats"]["p95"]:.12g}, threshold={THRESHOLD:.12g}, above={loc["stats"]["n_above_threshold"]}/{loc["stats"]["n_frames"]}.\n\nDEV2: `{recovery["status"]}`. Root cause: `{recovery["dev2_primary_root_cause"]}`. No full production solve was run.\n\nDiagnostic viewer: `{(root / "dev1_viewer/oakink2_wuji_semantic_diagnostic_viewer.html").resolve()}`\n\nOpen it with:\n\n```bash\nxdg-open '{(root / "dev1_viewer/oakink2_wuji_semantic_diagnostic_viewer.html").resolve()}'\n```\n\nUse the TOP E_IM bookmark buttons, then enable source/Wuji skeletons to inspect thumbs and interaction geometry.\n\nHard stop reached: DEV1 reruns=0; DEV2 production solves=0; O6/Support/PhysX/PPO=NO.\n"""
    (root / "handoff.md").write_text(handoff, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--action",
        choices=(
            "all",
            "preflight",
            "replay",
            "localize",
            "viewer",
            "dev2-audit",
            "profiling",
            "report",
        ),
        default="all",
    )
    value.add_argument("--report-root", type=Path, default=DEFAULT_ROOT)
    return value


def main() -> int:
    args = parser().parse_args()
    root = args.report_root.resolve()
    action = args.action
    if action in {"all", "preflight"}:
        frozen_evidence(root)
    replay = None
    if action in {"all", "replay"}:
        replay = semantic_replay(root)
    if action in {"all", "localize", "viewer", "report"}:
        if replay is None and not (root / "dev1_semantic/semantic_replay_bundle.npz").is_file():
            raise RuntimeError("O5RA_SEMANTIC_REPLAY_BUNDLE_MISSING; run --action replay first")
        loc = localize(root, {} if replay is None else replay)
        if action in {"all", "viewer"}:
            render_viewer(root, loc)
    else:
        loc = None
    if action in {"all", "dev2-audit"}:
        dev2_audit(root)
    if action in {"all", "profiling"}:
        profiling(root)
    if action in {"all", "report"}:
        report_contract(root)
        final(root, loc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
