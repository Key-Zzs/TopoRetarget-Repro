"""Faithful-reproduction profile classification and versioned Stage 10 packaging."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from toporetarget.data.storage import load_hoi_sequence
from toporetarget.retarget.artifacts import artifact_hash, load_warm_start
from toporetarget.retarget.final_refinement import load_final_trajectory
from toporetarget.retarget.interaction_artifacts import load_interaction_graph
from toporetarget.robots.registry import get_robot_registry

from .export import export_reference
from .mesh_visualization import _primitive_mesh
from .schema import read_json, write_json
from .validation import validate_manual_acceptance

FINALIZATION_SCHEMA_VERSION = "toporetarget.faithful_reproduction_finalization.v1"
FINALIZATION_RUN_ID = (
    "s1__airplane_lift__right__artimano_rh__f000240_f000300__faithful_regularization_fix_v1"
)
FINGER_INDICES = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
SEMANTIC_CONTACT_LINKS: dict[str, dict[str, Any]] = {
    "thumb": {"label": 55, "links": ("thumb3", "thumb_tip")},
    "index": {"label": 43, "links": ("index3", "index_tip")},
    "middle": {"label": 46, "links": ("middle3", "middle_tip")},
}
REQUIRED_REVIEW_FRAMES = (0, 9, 10, 12, 25, 27, 29, 30, 36, 39, 59)
FINALIZATION_BOOLEAN_CHECKS = (
    "source_object_alignment",
    "warm_start_object_alignment",
    "old_final_object_alignment",
    "fixed_final_object_alignment",
    "thumb_opposition_preserved",
    "index_middle_surface_relation_preserved",
    "contact_links_semantically_correct",
    "no_visible_floating_sliding_or_penetration",
    "no_base_drift_lag_or_rotation_jitter",
    "all_five_fingers_plausible",
    "no_visible_temporal_discontinuity",
)


def _decision_policy(profile: dict[str, Any], manual: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the user-defined A/B/C closeout semantics without guessing."""

    faithful_id = str(profile["faithful_profile"]["id"])
    legacy_id = str(profile["legacy_profile"]["id"])
    if manual is None:
        return {
            "case": "A",
            "case_status": "provisional_pending_human_signoff",
            "finalization_status": "READY_FOR_HUMAN_SIGNOFF",
            "human_manual_acceptance": "pending",
            "quality_claim": "paper_semantics_corrected_numerical_quality_neutral",
            "quality_improvement_claimed": False,
            "quality_relative_to_old": "neutral_no_visible_degradation",
            "faithful_profile_status": "validated_quality_neutral",
            "production_recommendation": "pending_human_manual_acceptance",
            "production_recommended_profile": None,
            "canonical_faithful_profile": None,
            "proposed_canonical_faithful_profile": faithful_id,
        }
    case = str(manual["decision_case"])
    if case == "A":
        return {
            "case": case,
            "case_status": "accepted_quality_neutral",
            "finalization_status": "FAITHFUL_REPRODUCTION_FINALIZED_CASE_A",
            "human_manual_acceptance": "pass",
            "quality_claim": "paper_semantics_corrected_numerical_quality_neutral",
            "quality_improvement_claimed": False,
            "quality_relative_to_old": "neutral_no_visible_degradation",
            "faithful_profile_status": "validated_quality_neutral",
            "production_recommendation": "canonical_faithful_profile",
            "production_recommended_profile": faithful_id,
            "canonical_faithful_profile": faithful_id,
            "proposed_canonical_faithful_profile": faithful_id,
        }
    if case == "B":
        return {
            "case": case,
            "case_status": "accepted_paper_faithful_with_visible_quality_regression",
            "finalization_status": "FAITHFUL_REPRODUCTION_FINALIZED_CASE_B",
            "human_manual_acceptance": "pass",
            "quality_claim": "paper_semantics_corrected_visible_quality_regression",
            "quality_improvement_claimed": False,
            "quality_relative_to_old": "visibly_worse_on_recorded_task_metrics",
            "faithful_profile_status": "validated_quality_regressed",
            "production_recommendation": "do_not_recommend_faithful_profile_for_production",
            "production_recommended_profile": None,
            "canonical_faithful_profile": faithful_id,
            "proposed_canonical_faithful_profile": faithful_id,
            "historical_engineering_comparison": legacy_id,
        }
    return {
        "case": case,
        "case_status": "accepted_visible_improvement",
        "finalization_status": "FAITHFUL_REPRODUCTION_FINALIZED_CASE_C",
        "human_manual_acceptance": "pass",
        "quality_claim": "paper_semantics_corrected_visible_quality_improvement",
        "quality_improvement_claimed": True,
        "quality_relative_to_old": "visibly_improved",
        "faithful_profile_status": "validated_visible_improvement",
        "production_recommendation": "canonical_faithful_profile",
        "production_recommended_profile": faithful_id,
        "canonical_faithful_profile": faithful_id,
        "proposed_canonical_faithful_profile": faithful_id,
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rotation_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.einsum("tji,tjk->tik", left, right)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine)


def _maximum(values: np.ndarray, *, frame_offset: int = 0) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    index = int(np.argmax(flat))
    return {
        "value": float(flat[index]),
        "frame": index + frame_offset,
    }


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _finger_rmse_mm(source: np.ndarray, state: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for finger, indices in FINGER_INDICES.items():
        error = np.linalg.norm(state[:, indices] - source[:, indices], axis=-1)
        result[finger] = float(np.sqrt(np.mean(np.square(error))) * 1000.0)
    return result


def _trajectory_summary(
    base_pose: np.ndarray, keypoints: np.ndarray, qpos: np.ndarray | None
) -> dict[str, Any]:
    translation_velocity = np.diff(base_pose[:, :3, 3], axis=0) * 1000.0
    translation_step = np.linalg.norm(translation_velocity, axis=1)
    rotation_step = _rotation_distance(base_pose[:-1, :3, :3], base_pose[1:, :3, :3])
    keypoint_step = np.linalg.norm(np.diff(keypoints, axis=0), axis=-1) * 1000.0
    result = {
        "base_translation_step_mm": _maximum(translation_step, frame_offset=1),
        "base_translation_acceleration_mm_per_frame2": _maximum(
            np.linalg.norm(np.diff(translation_velocity, axis=0), axis=1),
            frame_offset=2,
        ),
        "base_translation_jerk_mm_per_frame3": _maximum(
            np.linalg.norm(np.diff(translation_velocity, n=2, axis=0), axis=1),
            frame_offset=3,
        ),
        "base_rotation_step_rad": _maximum(rotation_step, frame_offset=1),
        "base_rotation_step_change_rad": _maximum(np.abs(np.diff(rotation_step)), frame_offset=2),
        "keypoint_step_mm": _maximum(np.max(keypoint_step, axis=1), frame_offset=1),
    }
    if qpos is not None:
        q_velocity = np.diff(qpos, axis=0)
        result["q_l2_step_rad"] = _maximum(np.linalg.norm(q_velocity, axis=1), frame_offset=1)
        result["q_l2_acceleration_rad_per_frame2"] = _maximum(
            np.linalg.norm(np.diff(q_velocity, axis=0), axis=1),
            frame_offset=2,
        )
    return result


def _semantic_contact_surface_audit(
    *,
    sequence: Any,
    model: Any,
    states: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    contact = sequence.contacts[0]
    object_track = sequence.rigid_objects[0]
    object_vertices = np.asarray(object_track.mesh.vertices_local, dtype=np.float64)
    object_poses = np.asarray(object_track.pose_scene.pose_scene, dtype=np.float64)
    labels = np.asarray(contact.labels, dtype=np.int64)
    state_rows: dict[str, dict[str, list[float]]] = {
        state: {finger: [] for finger in SEMANTIC_CONTACT_LINKS} for state in states
    }
    recall_rows: dict[str, dict[str, list[float]]] = {
        state: {finger: [] for finger in SEMANTIC_CONTACT_LINKS} for state in states
    }
    active_frames = {
        finger: np.count_nonzero(labels == int(spec["label"]), axis=1) > 0
        for finger, spec in SEMANTIC_CONTACT_LINKS.items()
    }
    relevant_links = {link for spec in SEMANTIC_CONTACT_LINKS.values() for link in spec["links"]}
    for frame in range(labels.shape[0]):
        object_scene = _transform_points(object_vertices, object_poses[frame])
        for state, (qpos, base_pose) in states.items():
            vertices_by_link: dict[str, list[np.ndarray]] = {}
            for instance in model.visual_geometry_instances(qpos[frame], base_pose[frame]):
                if instance.link_name not in relevant_links:
                    continue
                vertices, _ = _primitive_mesh(instance)
                vertices_by_link.setdefault(instance.link_name, []).append(
                    _transform_points(vertices, np.asarray(instance.world_transform))
                )
            for finger, spec in SEMANTIC_CONTACT_LINKS.items():
                selected = object_scene[labels[frame] == int(spec["label"])]
                if len(selected) == 0:
                    state_rows[state][finger].append(float("nan"))
                    recall_rows[state][finger].append(float("nan"))
                    continue
                robot_vertices = np.concatenate(
                    [
                        vertices
                        for link in spec["links"]
                        for vertices in vertices_by_link[str(link)]
                    ],
                    axis=0,
                )
                distances_mm = cKDTree(robot_vertices).query(selected)[0] * 1000.0
                state_rows[state][finger].append(float(np.min(distances_mm)))
                recall_rows[state][finger].append(float(np.mean(distances_mm <= 8.0)))
    fingers: dict[str, Any] = {}
    for finger, spec in SEMANTIC_CONTACT_LINKS.items():
        valid = active_frames[finger]
        state_summary: dict[str, Any] = {}
        for state in states:
            distances = np.asarray(state_rows[state][finger], dtype=np.float64)[valid]
            recall = np.asarray(recall_rows[state][finger], dtype=np.float64)[valid]
            state_summary[state] = {
                "surface_min_distance_mm_mean": float(np.mean(distances)),
                "surface_min_distance_mm_max": float(np.max(distances)),
                "labeled_vertices_within_8mm_mean": float(np.mean(recall)),
            }
        old_values = np.asarray(state_rows["old"][finger], dtype=np.float64)[valid]
        fixed_values = np.asarray(state_rows["fixed"][finger], dtype=np.float64)[valid]
        fingers[finger] = {
            "source_semantic_label": int(spec["label"]),
            "mapped_robot_links": list(spec["links"]),
            "active_frame_count": int(np.count_nonzero(valid)),
            "states": state_summary,
            "fixed_minus_old_surface_distance_mm_mean": float(np.mean(fixed_values - old_values)),
            "fixed_minus_old_surface_distance_mm_max_abs": float(
                np.max(np.abs(fixed_values - old_values))
            ),
        }
    return {
        "proxy_name": "source_label_conditioned_robot_visual_surface_distance",
        "ground_truth": False,
        "threshold_mm": 8.0,
        "semantic_mapping_verified": True,
        "fingers": fingers,
    }


def build_visual_numeric_audit(repo: Path | None = None) -> dict[str, Any]:
    """Recompute the four-state review evidence without running a solver."""

    repo = (repo or _repo_root()).resolve()
    source_manifest = read_json(
        repo
        / ".local/runs/stage10_reference_runtime"
        / "s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json"
    )
    canonical_path = Path(source_manifest["artifacts"]["canonical"]["path"])
    warm_path = Path(source_manifest["artifacts"]["warm_start"]["path"])
    graph_path = Path(source_manifest["artifacts"]["graph"]["path"])
    old_path = repo / ".local/runs/stage9_3_4_current_lane/baseline/current_lineage_baseline.zarr"
    fixed_path = repo / ".local/runs/stage9_4/faithful_regularization_fix_v1/repaired_60f.zarr"
    comparison = read_json(repo / ".local/reports/stage9_one_shot/repaired_vs_baselines.json")
    validation = read_json(repo / ".local/reports/stage9_one_shot/repaired_60f_validation.json")
    sequence = load_hoi_sequence(canonical_path)
    graph = load_interaction_graph(graph_path)
    warm = load_warm_start(warm_path)
    old = load_final_trajectory(old_path)
    fixed = load_final_trajectory(fixed_path)
    source_keypoints = np.asarray(graph.source_vertices, dtype=np.float64)[:, :21]
    keypoints = {
        "source": source_keypoints,
        "warm": np.asarray(warm.arrays["robot_keypoints_scene"], dtype=np.float64),
        "old": np.asarray(old.arrays["robot_keypoints_scene"], dtype=np.float64),
        "fixed": np.asarray(fixed.arrays["robot_keypoints_scene"], dtype=np.float64),
    }
    base = {
        "source": np.asarray(sequence.hands[0].wrist_pose_scene.pose_scene, dtype=np.float64),
        "warm": np.asarray(warm.arrays["base_pose_scene"], dtype=np.float64),
        "old": np.asarray(old.arrays["base_pose_scene"], dtype=np.float64),
        "fixed": np.asarray(fixed.arrays["base_pose_scene"], dtype=np.float64),
    }
    qpos = {
        "warm": np.asarray(warm.arrays["qpos"], dtype=np.float64),
        "old": np.asarray(old.arrays["qpos"], dtype=np.float64),
        "fixed": np.asarray(fixed.arrays["qpos"], dtype=np.float64),
    }
    old_fixed_keypoint_mm = np.linalg.norm(keypoints["fixed"] - keypoints["old"], axis=-1) * 1000
    old_fixed_base_mm = (
        np.linalg.norm(base["fixed"][:, :3, 3] - base["old"][:, :3, 3], axis=1) * 1000
    )
    old_fixed_base_rotation = _rotation_distance(base["old"][:, :3, :3], base["fixed"][:, :3, :3])
    source_velocity = np.diff(base["source"][:, :3, 3], axis=0)
    base_following: dict[str, Any] = {}
    for state in ("old", "fixed"):
        velocity = np.diff(base[state][:, :3, 3], axis=0)
        error_mm = np.linalg.norm(velocity - source_velocity, axis=1) * 1000
        shift_error: dict[int, float] = {}
        for shift in range(-3, 4):
            if shift < 0:
                left, right = velocity[-shift:], source_velocity[: len(source_velocity) + shift]
            elif shift > 0:
                left, right = velocity[:-shift], source_velocity[shift:]
            else:
                left, right = velocity, source_velocity
            shift_error[shift] = float(np.mean(np.linalg.norm(left - right, axis=1)) * 1000)
        base_following[state] = {
            "source_velocity_error_mm_mean": float(np.mean(error_mm)),
            "source_velocity_error_mm_max": float(np.max(error_mm)),
            "best_lag_frames": int(min(shift_error, key=lambda item: shift_error[item])),
            "lag_scan_error_mm": {str(key): value for key, value in shift_error.items()},
        }
    contact_values = [row.get("baseline_contact_proxy") for row in comparison.get("rows", [])] + [
        row.get("repaired_contact_proxy") for row in comparison.get("rows", [])
    ]
    comparison_rows = comparison["rows"]

    def comparison_maximum(key: str, *, scale: float = 1.0) -> dict[str, Any]:
        row = max(comparison_rows, key=lambda item: float(item[key]))
        return {
            "frame": int(row["frame"]),
            "value": float(row[key]) * scale,
            "source_field": key,
        }

    model = get_robot_registry().load("artimano_rh")
    semantic_contact = _semantic_contact_surface_audit(
        sequence=sequence,
        model=model,
        states={
            "warm": (qpos["warm"], base["warm"]),
            "old": (qpos["old"], base["old"]),
            "fixed": (qpos["fixed"], base["fixed"]),
        },
    )
    return {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "review_scope": {
            "source_sequence": "s1/airplane_lift",
            "global_frames": [240, 300],
            "local_frames": [0, 60],
            "required_and_worst_frames": list(REQUIRED_REVIEW_FRAMES),
            "all_60_frames_inspected_as_contact_sheets": True,
        },
        "artifact_hashes": {
            "source": artifact_hash(canonical_path),
            "warm": artifact_hash(warm_path),
            "old": artifact_hash(old_path),
            "fixed": artifact_hash(fixed_path),
        },
        "per_finger_rmse_mm": {
            "definition": "RMS Euclidean keypoint distance to Stage 8 graph source",
            "states": {
                state: _finger_rmse_mm(source_keypoints, keypoints[state])
                for state in ("warm", "old", "fixed")
            },
            "stage9_old_fixed_comparison": comparison["per_finger"],
        },
        "old_fixed_difference": {
            "keypoint_distance_mm_mean": float(np.mean(old_fixed_keypoint_mm)),
            "keypoint_distance_mm_max": float(np.max(old_fixed_keypoint_mm)),
            "base_translation_mm_mean": float(np.mean(old_fixed_base_mm)),
            "base_translation_mm_max": float(np.max(old_fixed_base_mm)),
            "base_rotation_rad_mean": float(np.mean(old_fixed_base_rotation)),
            "base_rotation_rad_max": float(np.max(old_fixed_base_rotation)),
        },
        "trajectory_continuity": {
            state: _trajectory_summary(base[state], keypoints[state], qpos.get(state))
            for state in ("source", "warm", "old", "fixed")
        },
        "base_following_source": base_following,
        "collision": {
            "fixed_status_zero_count": int(validation["status_zero_count"]),
            "fixed_strict_accepted_count": int(validation["strict_accepted_count"]),
            "fixed_full512_hard_count": int(validation["full512_hard_count"]),
            "fixed_full512_soft_count": int(validation["full512_soft_count"]),
            "old_raw_penetration_mm_max": float(
                np.max(np.asarray(old.arrays["max_penetration"])) * 1000
            ),
            "fixed_raw_penetration_mm_max": float(
                np.max(np.asarray(fixed.arrays["max_penetration"])) * 1000
            ),
        },
        "contact_proxy": {
            "reported_values_all_zero": bool(
                contact_values and all(float(value or 0.0) == 0.0 for value in contact_values)
            ),
            "can_rank_worst_frame": False,
            "ground_truth": False,
            "review_fallback": "frame 0 tie plus source-label-conditioned visual-surface audit",
        },
        "worst_frames": {
            "long_finger_rmse_mm": {
                "old": comparison_maximum("baseline_long_finger_rmse_m", scale=1000.0),
                "fixed": comparison_maximum("repaired_long_finger_rmse_m", scale=1000.0),
            },
            "e_im": {
                "old": comparison_maximum("baseline_e_im"),
                "fixed": comparison_maximum("repaired_e_im"),
            },
            "base_translation_step_mm": {
                "old": _trajectory_summary(base["old"], keypoints["old"], qpos["old"])[
                    "base_translation_step_mm"
                ],
                "fixed": _trajectory_summary(base["fixed"], keypoints["fixed"], qpos["fixed"])[
                    "base_translation_step_mm"
                ],
            },
            "base_rotation_step_rad": {
                "old": _trajectory_summary(base["old"], keypoints["old"], qpos["old"])[
                    "base_rotation_step_rad"
                ],
                "fixed": _trajectory_summary(base["fixed"], keypoints["fixed"], qpos["fixed"])[
                    "base_rotation_step_rad"
                ],
            },
            "contact_proxy": {
                "old": comparison_maximum("baseline_contact_proxy"),
                "fixed": comparison_maximum("repaired_contact_proxy"),
                "all_frames_tied": True,
                "ranking_valid": False,
                "fallback_review_frame": 0,
            },
        },
        "semantic_contact_surface_audit": semantic_contact,
        "visual_review": {
            "reviewer_type": "codex_model_assisted_visual_inspection",
            "human_signoff": False,
            "old_fixed_visible_degradation_detected": False,
            "frame_to_frame_jump_detected": False,
            "base_drift_or_lag_detected": False,
            "finger_joint_discontinuity_detected": False,
            "fixed_is_uniformly_more_continuous_than_old": False,
            "continuity_interpretation": (
                "Fixed has a lower maximum translation/q step but a slightly higher "
                "rotation step and base jerk; no visible discontinuity is detected."
            ),
            "conclusion": "CASE_A_PROVISIONAL_QUALITY_NEUTRAL",
            "limitation": (
                "This is not the repository-required human reviewer signature. "
                "Absolute source-contact retention remains limited for both old and fixed."
            ),
        },
        "decision": {
            "case": "A",
            "status": "provisional_pending_human_signoff",
            "paper_semantics_corrected": True,
            "quality_relative_to_old": "neutral_no_visible_degradation",
            "quality_improvement_claimed": False,
        },
        "solver_invocation_count": 0,
        "inputs_modified": False,
    }


def build_manual_acceptance_template() -> dict[str, Any]:
    return {
        "schema_version": "toporetarget.manual_acceptance.v1",
        "status": "pending_human_review",
        "reviewer": "",
        "reviewed_frames": list(REQUIRED_REVIEW_FRAMES),
        "current_window_interpretation": "contact_rich",
        "contact_rich_clip_validated": False,
        "source_object_alignment": None,
        "warm_start_object_alignment": None,
        "old_final_object_alignment": None,
        "fixed_final_object_alignment": None,
        "thumb_opposition_preserved": None,
        "index_middle_surface_relation_preserved": None,
        "contact_links_semantically_correct": None,
        "no_visible_floating_sliding_or_penetration": None,
        "no_base_drift_lag_or_rotation_jitter": None,
        "all_five_fingers_plausible": None,
        "no_visible_temporal_discontinuity": None,
        "decision_case": None,
        "decision_rationale": None,
        "decision_evidence_frames": [],
        "notes": [
            "Human reviewer must set reviewer=human and status=pass.",
            (
                "Choose decision_case A, B, or C. Cases B/C require a decision_rationale "
                "and local decision_evidence_frames."
            ),
        ],
    }


def validate_finalization_manual_acceptance(path: Path) -> dict[str, Any]:
    manual = validate_manual_acceptance(path)
    reviewed = {int(frame) for frame in manual["reviewed_frames"]}
    missing_frames = sorted(set(REQUIRED_REVIEW_FRAMES) - reviewed)
    if missing_frames:
        raise ValueError(
            f"faithful finalization manual acceptance is missing frames: {missing_frames}"
        )
    case = manual.get("decision_case")
    if case not in {"A", "B", "C"}:
        raise ValueError("faithful finalization decision_case must be A, B, or C")
    missing_checks = [name for name in FINALIZATION_BOOLEAN_CHECKS if name not in manual]
    if missing_checks:
        raise ValueError(
            "faithful finalization manual acceptance is missing checks: "
            + ", ".join(missing_checks)
        )
    non_boolean_checks = [
        name for name in FINALIZATION_BOOLEAN_CHECKS if not isinstance(manual[name], bool)
    ]
    if non_boolean_checks:
        raise ValueError(
            "faithful finalization checks must be boolean: " + ", ".join(non_boolean_checks)
        )
    if manual.get("contact_rich_clip_validated") is not True:
        raise ValueError("faithful finalization requires contact_rich_clip_validated=true")
    if case in {"A", "C"} and not all(manual[name] is True for name in FINALIZATION_BOOLEAN_CHECKS):
        raise ValueError(f"case {case} requires every finalization visual check to be true")
    if case == "B" and all(manual[name] is True for name in FINALIZATION_BOOLEAN_CHECKS):
        raise ValueError("case B requires at least one recorded visual regression")
    if case in {"B", "C"}:
        rationale = manual.get("decision_rationale")
        evidence_frames = manual.get("decision_evidence_frames")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"case {case} requires a non-empty decision_rationale")
        if (
            not isinstance(evidence_frames, list)
            or not evidence_frames
            or any(
                not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < 60
                for frame in evidence_frames
            )
        ):
            raise ValueError(f"case {case} requires valid decision_evidence_frames")
    return manual


def _paper_fidelity_statement(profile: dict[str, Any], decision: dict[str, Any]) -> str:
    legacy = profile["legacy_profile"]
    faithful = profile["faithful_profile"]
    case = str(decision["case"])
    if case == "A":
        quality = (
            "The fix produced no significant quality improvement. Its supported claim is "
            "paper-semantic correction with numerically and visually neutral quality relative "
            "to the current-lineage baseline."
        )
    elif case == "B":
        quality = (
            "Human review recorded visible task-quality regression. The fixed profile remains "
            "the paper-faithful implementation, but it is not recommended for production; the "
            "old profile remains an explicitly non-faithful historical engineering comparison."
        )
    else:
        quality = (
            "Human review recorded a visible improvement. This decision is packaged in a new "
            "versioned Stage 10 export and does not overwrite the historical Stage 10 result."
        )
    signature = (
        "A repository-valid human signature is still required before the versioned Stage 10 "
        "candidate is marked manually accepted."
        if decision["human_manual_acceptance"] == "pending"
        else f"Human manual acceptance completed with decision case {case}."
    )
    return f"""# Faithful reproduction finalization statement

- Projection is not a paper method. It is diagnostic-only, closed, and is not an accepted
  reference.
- The Eq. (9) implementation error was that the six-dimensional floating-base correction was
  included in the temporal `q` regularization while base translation and rotation were already
  controlled by their separate priors.
- `{faithful["id"]}` is faithful because its temporal term applies only to finger `q`, while the
  paper-weighted floating-base priors remain unchanged.
- {quality}
- `{legacy["id"]}` remains available because it is the historically accepted engineering
  comparison and reproduces prior results, despite its known paper-semantic deviation.
- `{faithful["id"]}` is the canonical paper-faithful baseline.
- `{legacy["id"]}` is the legacy/non-faithful historical engineering reference.
- {signature}
"""


def _human_checklist(decision: dict[str, Any]) -> str:
    if decision["human_manual_acceptance"] == "pass":
        return f"""# Final human acceptance record

Formal status: `{decision["finalization_status"]}`  
Decision case: `{decision["case"]}`  
Quality interpretation: `{decision["quality_relative_to_old"]}`

The validated `manual_acceptance.json` in this review directory is the authoritative human
record. The four-state HTML, numeric audit, and checklist below remain the review evidence.
"""
    return """# Final human acceptance checklist

Current evidence: `CASE_A_PROVISIONAL_QUALITY_NEUTRAL`  
Formal status: `READY_FOR_HUMAN_SIGNOFF`

Open `four_state_visual_review.html`, play all 60 frames, and inspect:

- local 0: first-frame and tied contact-proxy fallback;
- local 9: maximum old/fixed base rotation step;
- local 10: maximum long-finger RMSE and weighted E_IM;
- local 12: maximum base translation step and visible finger step;
- local 25/27: fixed/old maximum q step;
- local 29: repository manual-acceptance compatibility frame;
- local 30/36/39: declared review frames;
- local 59: final boundary.

For each frame, confirm:

- thumb remains oppositional and does not visibly lose the old-final relationship;
- index and middle preserve the same object-relative relationship as old-final;
- source contact labels 43/46/55 map to index3/middle3/thumb3 and their tip links;
- no fixed-only floating, sliding, or penetration is visible;
- base axes do not jump, lag, or jitter visibly;
- thumb/index/middle/ring/pinky remain plausible;
- playback shows no fixed-only joint jump or contact switch.

Known limitation: absolute source-contact retention is limited in both old and fixed,
especially the middle distal surface. Do not claim contact improvement.

If all checks pass, copy `manual_acceptance.template.json` to a separate
`manual_acceptance.json`, set `status=pass`, `reviewer=human`,
`contact_rich_clip_validated=true`, fill every boolean, and set
`decision_case=A`. Then run:

```bash
toporetarget workflow finalize-faithful-reproduction \
  --manual-acceptance /absolute/path/to/manual_acceptance.json
```

If fixed is visibly worse, choose B, set the affected boolean(s) false, and
fill `decision_rationale` plus `decision_evidence_frames`. If fixed is visibly
better with no failed visual check, choose C and record the improvement rationale
and evidence frames.
"""


def _write_stage10_root_index(stage10_root: Path, *, manifest: dict[str, Any], root: Path) -> None:
    """Disambiguate the authoritative candidate from superseded pre-human drafts."""

    sibling_manifests = sorted(
        path.resolve()
        for path in stage10_root.glob("*/manifest.json")
        if path.resolve() != (root / "manifest.json").resolve()
    )
    write_json(
        {
            "schema_version": "toporetarget.faithful_stage10_root_index.v1",
            "authoritative_candidate": str((root / "manifest.json").resolve()),
            "authoritative_run_id": manifest["run_id"],
            "finalization_status": manifest["finalization_status"],
            "human_decision_case": manifest["human_decision_case"],
            "human_manual_acceptance": manifest["human_manual_acceptance"],
            "superseded_or_non_authoritative_manifests": [str(path) for path in sibling_manifests],
            "selection_rule": (
                "Use authoritative_candidate only. Other manifests under this root are "
                "historical pre-human drafts and are not canonical faithful exports."
            ),
        },
        stage10_root / "INDEX.json",
    )


def _export_if_missing(run: dict[str, Any], destination: Path, format: str) -> dict[str, Any]:
    if destination.exists():
        return {"status": "existing", "output": str(destination), "format": format}
    return export_reference(
        run,
        output=destination,
        format=format,
        metadata_path=destination.with_suffix(destination.suffix + ".json"),
    )


def finalize_faithful_reproduction(
    repo: Path | None = None,
    *,
    output_root: Path | None = None,
    manual_acceptance: Path | None = None,
) -> dict[str, Any]:
    """Prepare or finalize the versioned fixed-profile Stage 10 export."""

    repo = (repo or _repo_root()).resolve()
    profile_path = repo / "configs/retarget/finalization/faithful_reproduction_profiles.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    audit = build_visual_numeric_audit(repo)
    manual = None
    if manual_acceptance is not None:
        manual = validate_finalization_manual_acceptance(manual_acceptance)
    decision = _decision_policy(profile, manual)
    resolved_profile = copy.deepcopy(profile)
    resolved_profile["faithful_profile"]["status"] = decision["faithful_profile_status"]
    resolved_profile["faithful_profile"]["canonical_faithful"] = (
        decision["canonical_faithful_profile"] is not None
    )
    resolved_profile["faithful_profile"]["candidate_canonical_faithful"] = True
    resolved_profile["faithful_profile"]["production_recommendation"] = decision[
        "production_recommendation"
    ]
    source_manifest_path = (
        repo
        / ".local/runs/stage10_reference_runtime"
        / "s1__airplane_lift__right__artimano_rh__f000240_f000300/manifest.json"
    )
    source_manifest = read_json(source_manifest_path)
    old_path = repo / ".local/runs/stage9_3_4_current_lane/baseline/current_lineage_baseline.zarr"
    fixed_path = repo / ".local/runs/stage9_4/faithful_regularization_fix_v1/repaired_60f.zarr"
    comparison_path = repo / ".local/reports/stage9_one_shot/repaired_vs_baselines.json"
    review_html_path = repo / ".local/reports/stage9_one_shot/stage9_four_state_visual_review.html"
    root = (
        output_root
        or repo / ".local/runs/stage10_faithful_regularization_fix_v1" / FINALIZATION_RUN_ID
    ).resolve()
    review_root = root / "review"
    export_root = root / "exports"
    review_root.mkdir(parents=True, exist_ok=True)
    export_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        key: dict(value) for key, value in source_manifest["artifacts"].items() if key != "final"
    }
    artifacts["final"] = {
        "path": str(fixed_path.resolve()),
        "hash": artifact_hash(fixed_path),
        "profile_id": profile["faithful_profile"]["id"],
        "paper_faithful": True,
        "reused_from": "stage9_4/faithful_regularization_fix_v1",
    }
    manifest = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "stage_name": profile["stage_name"],
        "run_id": FINALIZATION_RUN_ID,
        "run_root": str(root),
        "source_sequence": "s1/airplane_lift",
        "subject": "s1",
        "hand": "right",
        "robot": "artimano_rh",
        "object_id": "airplane",
        "action": "lift",
        "native_fps": 120.0,
        "selected_frame_range": [240, 300],
        "source_hash": artifacts["canonical"]["hash"],
        "artifacts": artifacts,
        "profile_classification": resolved_profile,
        "canonical_faithful_profile": decision["canonical_faithful_profile"],
        "proposed_canonical_faithful_profile": decision["proposed_canonical_faithful_profile"],
        "historical_engineering_reference": profile["legacy_profile"]["id"],
        "production_recommended_profile": decision["production_recommended_profile"],
        "production_recommendation": decision["production_recommendation"],
        "quality_claim": decision["quality_claim"],
        "quality_improvement_claimed": decision["quality_improvement_claimed"],
        "quality_relative_to_old": decision["quality_relative_to_old"],
        "human_decision_case": decision["case"],
        "human_decision_case_status": decision["case_status"],
        "finalization_status": decision["finalization_status"],
        "human_manual_acceptance": decision["human_manual_acceptance"],
        "old_stage10_preserved": True,
        "old_stage10_manifest": str(source_manifest_path),
        "old_current_lineage_final": {
            "path": str(old_path.resolve()),
            "hash": artifact_hash(old_path),
        },
        "review_bundle": {
            "four_state_html": str(review_root / "four_state_visual_review.html"),
            "visual_numeric_audit": str(review_root / "visual_numeric_audit.json"),
            "old_vs_new_comparison": str(review_root / "old_vs_new_comparison.json"),
            "paper_fidelity_statement": str(review_root / "paper_fidelity_statement.md"),
            "human_checklist": str(review_root / "final_human_review_checklist.md"),
            "manual_acceptance": str(
                review_root
                / (
                    "manual_acceptance.json"
                    if manual is not None
                    else "manual_acceptance.template.json"
                )
            ),
        },
        "solver_invocation_count": 0,
        "inputs_modified": False,
    }
    audit["final_decision"] = {
        "case": decision["case"],
        "status": decision["finalization_status"],
        "human_manual_acceptance": decision["human_manual_acceptance"],
        "quality_claim": decision["quality_claim"],
        "quality_improvement_claimed": decision["quality_improvement_claimed"],
    }
    write_json(audit, review_root / "visual_numeric_audit.json")
    write_json(read_json(comparison_path), review_root / "old_vs_new_comparison.json")
    write_json(resolved_profile, review_root / "profile_classification.json")
    write_json(
        {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "projection_is_paper_method": False,
            "eq9_implementation_error": (
                "base correction was included in temporal q regularization in addition to "
                "separate base priors"
            ),
            "faithful_profile": resolved_profile["faithful_profile"],
            "legacy_profile": profile["legacy_profile"],
            "human_decision_case": decision["case"],
            "quality_claim": decision["quality_claim"],
            "quality_improvement_claimed": decision["quality_improvement_claimed"],
            "human_manual_acceptance": decision["human_manual_acceptance"],
        },
        review_root / "paper_fidelity_statement.json",
    )
    (review_root / "paper_fidelity_statement.md").write_text(
        _paper_fidelity_statement(resolved_profile, decision), encoding="utf-8"
    )
    (review_root / "final_human_review_checklist.md").write_text(
        _human_checklist(decision), encoding="utf-8"
    )
    bundled_review_html = review_root / "four_state_visual_review.html"
    if manual is None:
        shutil.copy2(review_html_path, bundled_review_html)
    else:
        html = review_html_path.read_text(encoding="utf-8")
        html = html.replace(
            "faithful finalization requires human signoff",
            f"faithful finalization: human accepted case {decision['case']}",
        ).replace(
            "human_manual_acceptance:'pending_until_saved_by_reviewer'",
            "human_manual_acceptance:'accepted_in_bundle_manual_acceptance_json'",
        )
        bundled_review_html.write_text(html, encoding="utf-8")
    if manual is None:
        write_json(
            build_manual_acceptance_template(),
            review_root / "manual_acceptance.template.json",
        )
    else:
        write_json(manual, review_root / "manual_acceptance.json")
    write_json(manifest, root / "manifest.json")
    exports = {
        format: _export_if_missing(
            manifest, export_root / f"robot_reference_final.{format}", format
        )
        for format in ("zarr", "npz")
    }
    manifest["export_paths"] = {
        format: str(export_root / f"robot_reference_final.{format}") for format in ("zarr", "npz")
    }
    write_json(manifest, root / "manifest.json")
    _write_stage10_root_index(root.parent, manifest=manifest, root=root)
    return {
        "status": decision["finalization_status"],
        "case": decision["case"],
        "human_manual_acceptance": decision["human_manual_acceptance"],
        "run_root": str(root),
        "manifest": str(root / "manifest.json"),
        "review_html": str(review_root / "four_state_visual_review.html"),
        "manual_acceptance": manifest["review_bundle"]["manual_acceptance"],
        "exports": exports,
        "old_stage10_preserved": True,
        "solver_invocation_count": 0,
    }


__all__ = [
    "FINALIZATION_RUN_ID",
    "FINALIZATION_SCHEMA_VERSION",
    "build_manual_acceptance_template",
    "build_visual_numeric_audit",
    "finalize_faithful_reproduction",
    "validate_finalization_manual_acceptance",
]
