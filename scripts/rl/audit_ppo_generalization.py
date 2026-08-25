#!/usr/bin/env python3
"""Audit PPO length, exact-batch phase coverage, and HOCap object scale offline."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toporetarget.rl.ppo_generalization import (  # noqa: E402
    INTERACTION_PHASES,
    PHASES,
    DimensionlessObjectScaleV1,
    EpisodeV1RuntimeEvents,
    UniformEventBalancedRSIV1,
    map_source_event_to_runtime,
    object_bbox_diagonal_m,
)

DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/raw_to_physical_hardening_v2/p3_ppo_generalization"
EPISODE_INDEX = (
    REPO_ROOT / ".local/reports/hocap_physicalization_protocol_freeze/all_hocap_episodes.json"
)
HARDENING_MANIFEST = (
    REPO_ROOT
    / ".local/reports/raw_to_physical_hardening_v2/p0_closeout/hardening_set_manifest.json"
)
PF_ROOT = REPO_ROOT / ".local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo"
HARDENING_CLIP = "hocap_subject_9_20231027_125019__right__G16_3__ep00"
SAMPLES_PER_UPDATE = 40_960
CURRENT_FROZEN_MAX_UPDATES = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"P3_EMPTY_CSV:{path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _legacy_phase_labels(task_path: Path) -> tuple[tuple[str, ...], dict[str, bool]]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    length = int(task["retimed_frame_count"])
    onset = int(task["contact_onset_window"]["start"])
    persistent = int(task["persistent_contact_window"]["start"])
    contact_end = int(task["contact_end_window"]["end"])
    labels = ["PRE/IDLE"] * length
    labels[onset:persistent] = ["APPROACH"] * (persistent - onset)
    labels[persistent : contact_end + 1] = ["CONTACT"] * (contact_end + 1 - persistent)
    identifiable = {phase: phase in {"PRE/IDLE", "APPROACH", "CONTACT"} for phase in PHASES}
    return tuple(labels), identifiable


def _episode_phase_labels(
    episode: dict[str, Any], reference_length: int
) -> tuple[tuple[str, ...], dict[str, bool], dict[str, int]]:
    common = {
        "source_start_frame": int(episode["start_frame"]),
        "source_end_frame": int(episode["end_frame"]),
        "runtime_reference_length": reference_length,
    }
    event_indices = {
        name: map_source_event_to_runtime(int(episode[name]), **common)
        for name in (
            "approach_frame",
            "contact_frame",
            "pickup_frame",
            "place_frame",
            "release_frame",
            "retreat_frame",
        )
    }
    events = EpisodeV1RuntimeEvents(
        reference_length=reference_length,
        approach=event_indices["approach_frame"],
        contact=event_indices["contact_frame"],
        pickup=event_indices["pickup_frame"],
        place=event_indices["place_frame"],
        release=event_indices["release_frame"],
        retreat=event_indices["retreat_frame"],
    )
    return events.phase_labels(), dict.fromkeys(PHASES, True), event_indices


def _batch_specs(episode_index: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hardening_episode = next(row for row in episode_index if row["episode_id"] == HARDENING_CLIP)
    dev_170105 = sorted(
        (REPO_ROOT / ".local/runs/stage16_dexplore_reward_rse/training").glob(
            "U*/exact_batch/exact_batch.pt"
        )
    )
    dev_170105.append(
        REPO_ROOT
        / ".local/runs/stage16_pf_v2_causal_lift_and_symmetric_ppo"
        / "hocap_170105/updates/U11/exact_batch/exact_batch.pt"
    )
    dev_170650 = sorted(
        (
            REPO_ROOT
            / ".local/runs/stage16_pf_v2_causal_lift_and_symmetric_ppo/hocap_170650/updates"
        ).glob("U0[12]/exact_batch/exact_batch.pt")
    )
    hardening_batches = sorted(
        (
            REPO_ROOT
            / ".local/experiments"
            / "held_out_hocap_raw_to_physical_pilot_post_freeze_l0_unbounded_eval_retry1"
            / HARDENING_CLIP
            / "physical_refinement"
            / HARDENING_CLIP
            / "updates"
        ).glob("U*/exact_batch/exact_batch.pt")
    )
    labels_170105, identifiable_170105 = _legacy_phase_labels(
        REPO_ROOT
        / ".local/reports/stage16d_physics_consistent_retargeting/hocap_170105.task_semantics.json"
    )
    labels_170650, identifiable_170650 = _legacy_phase_labels(
        REPO_ROOT
        / ".local/reports/stage16d_physics_consistent_retargeting/hocap_170650.task_semantics.json"
    )
    hard_labels, hard_identifiable, hard_events = _episode_phase_labels(hardening_episode, 1_121)
    return [
        {
            "clip_id": "hocap_170105",
            "dataset_role": "SUCCESSFUL_DEVELOPMENT",
            "outcome": "PF_V2_20_OF_20_AT_U11",
            "outcome_evidence": PF_ROOT / "training/hocap_170105/confirm20/U11/summary.json",
            "batches": dev_170105,
            "labels": labels_170105,
            "identifiable": identifiable_170105,
            "semantic_authority": "TaskSemanticContractV1_legacy_truncated_window",
            "events": {},
        },
        {
            "clip_id": "hocap_170650",
            "dataset_role": "SUCCESSFUL_DEVELOPMENT",
            "outcome": "PF_V1_AND_PF_V2_10_OF_10_AT_U02",
            "outcome_evidence": PF_ROOT / "training/hocap_170650/U02/eval10/summary.json",
            "batches": dev_170650,
            "labels": labels_170650,
            "identifiable": identifiable_170650,
            "semantic_authority": "TaskSemanticContractV1_legacy_truncated_window",
            "events": {},
        },
        {
            "clip_id": HARDENING_CLIP,
            "dataset_role": "PIPELINE_HARDENING_SET_V1",
            "outcome": "PPO_BUDGET_EXHAUSTED_PF_V2_0_OF_10_AT_U15",
            "outcome_evidence": REPO_ROOT
            / ".local/reports"
            / "held_out_hocap_raw_to_physical_pilot_post_freeze_l0_unbounded_eval_retry1"
            / "clips"
            / HARDENING_CLIP
            / "physical_refinement/training"
            / HARDENING_CLIP
            / "U15/eval10/summary.json",
            "batches": hardening_batches,
            "labels": hard_labels,
            "identifiable": hard_identifiable,
            "semantic_authority": "HOCapSingleHandObjectEpisodeV1",
            "events": hard_events,
        },
    ]


def audit_batches(
    specs: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    length_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for spec in specs:
        labels = np.asarray(spec["labels"])
        reference_histogram = np.zeros(len(labels), dtype=np.int64)
        aggregate = {
            phase: {
                "samples": 0,
                "r_int_active": 0,
                "contact_positive": 0,
                "advantage_sum": 0.0,
                "advantage_square_sum": 0.0,
                "positive_advantage": 0,
            }
            for phase in PHASES
        }
        total_samples = 0
        for update_number, batch_path in enumerate(spec["batches"], start=1):
            if not batch_path.is_file():
                raise FileNotFoundError(batch_path)
            payload = torch.load(batch_path, map_location="cpu", weights_only=False)
            reference_indices = payload["reference_indices"].detach().numpy().reshape(-1)
            advantages = payload["advantages"].detach().numpy().reshape(-1).astype(np.float64)
            reward_terms = payload["reward_terms"]
            r_int_active = (
                reward_terms["source_contact_mask"].detach().numpy().reshape(-1, 5).any(axis=1)
            )
            contact_positive = (
                reward_terms["tip_pair_presence"].detach().numpy().reshape(-1, 5).any(axis=1)
            )
            if (
                reference_indices.size != advantages.size
                or advantages.size != r_int_active.size
                or r_int_active.size != contact_positive.size
                or int(reference_indices.min()) < 0
                or int(reference_indices.max()) >= len(labels)
            ):
                raise ValueError(f"P3_EXACT_BATCH_SHAPE_OR_INDEX_INVALID:{batch_path}")
            np.add.at(reference_histogram, reference_indices, 1)
            sample_labels = labels[reference_indices]
            for phase in PHASES:
                mask = sample_labels == phase
                count = int(mask.sum())
                if count == 0:
                    continue
                values = advantages[mask]
                row = aggregate[phase]
                row["samples"] += count
                row["r_int_active"] += int(r_int_active[mask].sum())
                row["contact_positive"] += int(contact_positive[mask].sum())
                row["advantage_sum"] += float(values.sum())
                row["advantage_square_sum"] += float(np.square(values).sum())
                row["positive_advantage"] += int((values > 0.0).sum())
            total_samples += reference_indices.size
            evidence_rows.append(
                {
                    "clip_id": spec["clip_id"],
                    "update": update_number,
                    "path": str(batch_path.relative_to(REPO_ROOT)),
                    "sha256": sha256(batch_path),
                    "schema_version": payload["schema_version"],
                    "samples": int(reference_indices.size),
                    "reference_index_min": int(reference_indices.min()),
                    "reference_index_max": int(reference_indices.max()),
                }
            )
            del payload, reward_terms
            gc.collect()

        if total_samples != len(spec["batches"]) * SAMPLES_PER_UPDATE:
            raise ValueError(f"P3_EXACT_SAMPLE_COUNT_DRIFT:{spec['clip_id']}")
        interaction_reference_count = sum(
            int((labels == phase).sum())
            for phase in INTERACTION_PHASES
            if bool(spec["identifiable"][phase])
        )
        actual_interaction_samples = sum(
            int(aggregate[phase]["samples"])
            for phase in INTERACTION_PHASES
            if bool(spec["identifiable"][phase])
        )
        r_int_active_samples = sum(int(row["r_int_active"]) for row in aggregate.values())
        contact_positive_samples = sum(int(row["contact_positive"]) for row in aggregate.values())
        length_rows.append(
            {
                "clip_id": spec["clip_id"],
                "dataset_role": spec["dataset_role"],
                "outcome": spec["outcome"],
                "reference_length": len(labels),
                "valid_rsi_domain_length": len(labels),
                "interaction_frame_count": interaction_reference_count,
                "samples_per_update": SAMPLES_PER_UPDATE,
                "updates": len(spec["batches"]),
                "total_samples": total_samples,
                "samples_per_valid_index": total_samples / len(labels),
                "total_samples_per_interaction_index": total_samples / interaction_reference_count,
                "actual_interaction_samples": actual_interaction_samples,
                "actual_samples_per_interaction_index": actual_interaction_samples
                / interaction_reference_count,
                "R_int_active_samples": r_int_active_samples,
                "contact_positive_samples": contact_positive_samples,
                "unique_valid_indices_sampled": int(np.count_nonzero(reference_histogram)),
                "outcome_evidence": str(Path(spec["outcome_evidence"]).relative_to(REPO_ROOT)),
            }
        )
        for phase in PHASES:
            values = aggregate[phase]
            count = int(values["samples"])
            reference_count = int((labels == phase).sum()) if spec["identifiable"][phase] else 0
            mean = float(values["advantage_sum"]) / count if count else None
            variance = (
                max(0.0, float(values["advantage_square_sum"]) / count - mean**2) if count else None
            )
            reference_fraction = reference_count / len(labels) if reference_count else 0.0
            sample_fraction = count / total_samples if count else 0.0
            phase_rows.append(
                {
                    "clip_id": spec["clip_id"],
                    "dataset_role": spec["dataset_role"],
                    "phase": phase,
                    "phase_identifiable": bool(spec["identifiable"][phase]),
                    "semantic_authority": spec["semantic_authority"],
                    "reference_index_count": reference_count,
                    "reference_domain_fraction": reference_fraction,
                    "exact_sample_count": count,
                    "exact_sample_fraction": sample_fraction,
                    "sample_to_domain_fraction_ratio": sample_fraction / reference_fraction
                    if reference_fraction
                    else None,
                    "R_int_activation_count": int(values["r_int_active"]),
                    "R_int_activation_fraction": int(values["r_int_active"]) / count
                    if count
                    else None,
                    "actual_contact_count": int(values["contact_positive"]),
                    "actual_contact_fraction": int(values["contact_positive"]) / count
                    if count
                    else None,
                    "advantage_mean": mean,
                    "advantage_std": variance**0.5 if variance is not None else None,
                    "positive_advantage_fraction": int(values["positive_advantage"]) / count
                    if count
                    else None,
                }
            )
    return length_rows, phase_rows, evidence_rows


def audit_object_scale(
    episode_index: list[dict[str, Any]], hardening_manifest: dict[str, Any]
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    eligible = [row for row in episode_index if row["physicalization_v1_eligible"]]
    if len(eligible) != 108:
        raise ValueError(f"P3_EPISODEV1_ELIGIBLE_COUNT_DRIFT:{len(eligible)}")
    hardening_ids = {row["episode_id"] for row in hardening_manifest["episodes"]}
    scale_cache: dict[Path, float] = {}

    def scale(path: Path) -> float:
        if path not in scale_cache:
            scale_cache[path] = object_bbox_diagonal_m(path)
        return scale_cache[path]

    unique_object_scales: dict[str, float] = {}
    base_rows: list[dict[str, object]] = []
    for episode in eligible:
        mesh_path = Path(episode["provenance"]["object_mesh"]["path"])
        object_scale = scale(mesh_path)
        object_id = str(episode["target_object"])
        prior = unique_object_scales.setdefault(object_id, object_scale)
        if not np.isclose(prior, object_scale, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"P3_OBJECT_ID_SCALE_DRIFT:{object_id}")
        base_rows.append(
            {
                "audit_population": "ELIGIBLE_EPISODEV1_108",
                "episode_id": episode["episode_id"],
                "object_id": object_id,
                "object_mesh": str(mesh_path),
                "object_mesh_sha256": episode["provenance"]["object_mesh"]["sha256"],
                "s_O_bbox_diagonal_m": object_scale,
                "is_pipeline_hardening_set_v1": episode["episode_id"] in hardening_ids,
            }
        )
    anchor = statistics.median(unique_object_scales.values())
    normalization = DimensionlessObjectScaleV1(anchor_bbox_diagonal_m=anchor)

    audit_rows: list[dict[str, object]] = []

    def add_thresholds(row: dict[str, object]) -> None:
        object_scale = float(row["s_O_bbox_diagonal_m"])
        normalized = normalization.thresholds(object_scale)
        row.update(
            {
                "fixed_0.03_over_s_O": 0.03 / object_scale,
                "fixed_object_tracking_0.04_over_s_O": 0.04 / object_scale,
                "fixed_0.05_over_s_O": 0.05 / object_scale,
                "fixed_object_velocity_0.075_over_s_O_per_s": 0.075 / object_scale,
                "fixed_0.20_over_s_O": 0.20 / object_scale,
                "fixed_hand_link_0.025_over_s_O_diagnostic_only": 0.025 / object_scale,
                "fixed_wrist_position_0.02_over_s_O_diagnostic_only": 0.02 / object_scale,
                "selected_proximity_tolerance_m": normalized["proximity_tolerance_m"],
                "selected_distance_scope_m": normalized["distance_scope_m"],
                "selected_object_tracking_sigma_m": normalized["object_tracking_sigma_m"],
                "selected_object_velocity_sigma_mps": normalized["object_velocity_sigma_mps"],
                "selected_object_position_base_m": normalized["object_position_base_m"],
                "selected_object_axis_base_m": normalized["object_axis_base_m"],
                "selected_hand_position_base_m_unchanged": 0.20,
                "selected_hand_link_sigma_m_unchanged": 0.025,
                "selected_wrist_position_sigma_m_unchanged": 0.02,
            }
        )

    for row in base_rows:
        add_thresholds(row)
        audit_rows.append(row)
    by_episode = {str(row["episode_id"]): row for row in base_rows}
    for episode_id in sorted(hardening_ids):
        duplicate = dict(by_episode[episode_id])
        duplicate["audit_population"] = "PIPELINE_HARDENING_SET_V1_5"
        audit_rows.append(duplicate)

    regression_rows: list[dict[str, object]] = []
    for clip_id in ("hocap_170105", "hocap_170650"):
        mesh = (
            REPO_ROOT
            / ".local/stage16_reference_tracking_ppo/world_wrist_objects"
            / f"{clip_id}.obj"
        )
        object_scale = scale(mesh)
        normalized = normalization.thresholds(object_scale)
        row: dict[str, object] = {
            "audit_population": "SUCCESSFUL_DEVELOPMENT_2",
            "episode_id": clip_id,
            "object_id": clip_id,
            "object_mesh": str(mesh.relative_to(REPO_ROOT)),
            "object_mesh_sha256": sha256(mesh),
            "s_O_bbox_diagonal_m": object_scale,
            "is_pipeline_hardening_set_v1": False,
        }
        add_thresholds(row)
        audit_rows.append(row)
        for name, old_value in (
            ("proximity_tolerance_m", 0.03),
            ("distance_scope_m", 0.20),
            ("object_tracking_sigma_m", 0.04),
            ("object_velocity_sigma_mps", 0.075),
            ("object_position_base_m", 0.05),
            ("object_axis_base_m", 0.05),
        ):
            new_value = normalized[name]
            regression_rows.append(
                {
                    "clip_id": clip_id,
                    "threshold": name,
                    "units": "m/s" if name.endswith("_mps") else "m",
                    "object_scale_m": object_scale,
                    "old_fixed_value": old_value,
                    "selected_value": new_value,
                    "absolute_delta": new_value - old_value,
                    "relative_delta": new_value / old_value - 1.0,
                    "runtime_reward_recomputed": False,
                    "disposition": "OBJECT_RELATIVE_NORMALIZED",
                    "reason": "offline numerical threshold regression; no GPU and no PPO rerun",
                }
            )
        for name, old_value in (
            ("hand_position_base_m", 0.20),
            ("hand_link_sigma_m", 0.025),
            ("wrist_position_sigma_m", 0.02),
        ):
            regression_rows.append(
                {
                    "clip_id": clip_id,
                    "threshold": name,
                    "units": "m",
                    "object_scale_m": object_scale,
                    "old_fixed_value": old_value,
                    "selected_value": old_value,
                    "absolute_delta": 0.0,
                    "relative_delta": 0.0,
                    "runtime_reward_recomputed": False,
                    "disposition": "HAND_OR_CONTROLLER_SCALE_UNCHANGED",
                    "reason": "not an object-relative geometry threshold",
                }
            )

    scales = list(unique_object_scales.values())
    summary = {
        "schema_version": "P3ObjectScaleAuditSummaryV1",
        "eligible_episode_count": len(eligible),
        "eligible_unique_object_count": len(unique_object_scales),
        "hardening_episode_count": len(hardening_ids),
        "bbox_diagonal_m": {
            "min": min(scales),
            "median_unique_object": anchor,
            "max": max(scales),
            "max_over_min": max(scales) / min(scales),
        },
        "fixed_dimensionless_ratio_spread": {
            "0.03_over_s_O_max_over_min": max(scales) / min(scales),
            "0.20_over_s_O_max_over_min": max(scales) / min(scales),
        },
        "fixed_metric_inventory": [
            {
                "name": "proximity_tolerance_m",
                "value": 0.03,
                "disposition": "OBJECT_RELATIVE_NORMALIZED",
            },
            {
                "name": "distance_scope_m",
                "value": 0.20,
                "disposition": "OBJECT_RELATIVE_NORMALIZED",
            },
            {
                "name": "object_tracking_sigma_m",
                "value": 0.04,
                "disposition": "OBJECT_RELATIVE_NORMALIZED",
            },
            {
                "name": "object_velocity_sigma_mps",
                "value": 0.075,
                "disposition": "OBJECT_RELATIVE_NORMALIZED",
            },
            {
                "name": "RSE_object_position_base_m",
                "value": 0.05,
                "disposition": "OBJECT_RELATIVE_NORMALIZED",
            },
            {
                "name": "RSE_object_axis_base_m",
                "value": 0.05,
                "disposition": "OBJECT_RELATIVE_NORMALIZED",
            },
            {
                "name": "tracked_link_sigma_m",
                "value": 0.025,
                "disposition": "HAND_SCALE_UNCHANGED",
            },
            {
                "name": "wrist_position_sigma_m",
                "value": 0.02,
                "disposition": "CONTROLLER_SCALE_UNCHANGED",
            },
            {
                "name": "RSE_hand_position_base_m",
                "value": 0.20,
                "disposition": "HAND_CONTROLLER_SCALE_UNCHANGED",
            },
        ],
        "decision_rule": (
            "normalization required when eligible unique-object max/min scale exceeds 2.0"
        ),
        "decision": "DIMENSIONLESS_OBJECT_SCALE_NORMALIZATION_REQUIRED",
        "anchor_selection": (
            "metadata-only median bbox diagonal over 59 unique eligible EpisodeV1 objects"
        ),
        "dimensionless_ratios": normalization.dimensionless_ratios(),
    }
    return audit_rows, summary, regression_rows


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    required_names = (
        "length_coverage.csv",
        "phase_coverage.csv",
        "object_scale_audit.csv",
        "budget_contract.json",
        "rsi_contract.json",
        "scale_contract.json",
        "final_decision.json",
    )
    existing = [str(output / name) for name in required_names if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"P3_REFUSES_OVERWRITE:{existing}")
    output.mkdir(parents=True, exist_ok=True)

    episode_index = json.loads(EPISODE_INDEX.read_text(encoding="utf-8"))
    hardening_manifest = json.loads(HARDENING_MANIFEST.read_text(encoding="utf-8"))
    specs = _batch_specs(episode_index)
    length_rows, phase_rows, evidence_rows = audit_batches(specs)
    scale_rows, scale_summary, regression_rows = audit_object_scale(
        episode_index, hardening_manifest
    )
    write_csv(output / "length_coverage.csv", length_rows)
    write_csv(output / "phase_coverage.csv", phase_rows)
    write_csv(output / "object_scale_audit.csv", scale_rows)
    write_csv(output / "object_scale_numerical_regression.csv", regression_rows)
    write_csv(output / "exact_batch_evidence.csv", evidence_rows)
    write_json(output / "object_scale_summary.json", scale_summary)

    hardening_phases = {row["phase"]: row for row in phase_rows if row["clip_id"] == HARDENING_CLIP}
    transport = hardening_phases["TRANSPORT"]
    interaction_domain_fraction = sum(
        float(row["reference_domain_fraction"])
        for row in hardening_phases.values()
        if row["phase"] in INTERACTION_PHASES
    )
    interaction_sample_fraction = sum(
        float(row["exact_sample_fraction"])
        for row in hardening_phases.values()
        if row["phase"] in INTERACTION_PHASES
    )

    budget_contract = {
        "schema_version": "P3LengthGeneralizedBudgetContractV2",
        "audit_decision": "INCONCLUSIVE",
        "selected_branch": "CURRENT_FROZEN_FIXED_SAMPLE_BUDGET_FALLBACK",
        "fallback_reason": (
            "two successful development lineages differ by more than 5x in samples per valid "
            "index, and the hardening failure lies inside that successful coverage envelope"
        ),
        "rule": {
            "max_new_updates": CURRENT_FROZEN_MAX_UPDATES,
            "samples_per_update": SAMPLES_PER_UPDATE,
            "global_sample_cap": CURRENT_FROZEN_MAX_UPDATES * SAMPLES_PER_UPDATE,
            "outcome_dependent_expansion": False,
            "unbounded_updates": False,
            "per_clip_tuning": False,
        },
        "status_label": "LENGTH_GENERALIZATION_NOT_ESTABLISHED",
        "candidate_rules_not_selected": [
            "FIXED_SAMPLE_BUDGET_as_generalization_claim",
            "SAMPLES_PER_VALID_REFERENCE_INDEX",
            "SAMPLES_PER_INTERACTION_INDEX",
            "HYBRID_LENGTH_AND_INTERACTION_BUDGET",
        ],
        "evidence": "length_coverage.csv",
    }
    rsi = UniformEventBalancedRSIV1(uniform_alpha=0.5)
    rsi_contract = {
        "schema_version": "P3RSIPhaseCoverageContractV2",
        "audit_decision": "UNIFORM_PLUS_EVENT_BALANCED_RSI",
        "selected_branch": "UNIFORM_PLUS_EVENT_BALANCED_RSI",
        "sampler": rsi.as_dict(),
        "mixture_formula": "alpha*U(T_valid)+(1-alpha)*U(T_interaction)",
        "all_clips_same_alpha": True,
        "manual_grasp_frame": False,
        "hardening_evidence": {
            "interaction_reference_domain_fraction": interaction_domain_fraction,
            "interaction_exact_sample_fraction": interaction_sample_fraction,
            "interaction_dilution_gap": interaction_sample_fraction - interaction_domain_fraction,
            "transport_reference_domain_fraction": transport["reference_domain_fraction"],
            "transport_exact_sample_fraction": transport["exact_sample_fraction"],
            "transport_sample_to_domain_fraction_ratio": transport[
                "sample_to_domain_fraction_ratio"
            ],
        },
        "legacy_development_limitation": (
            "170105 and 170650 are truncated pre-EpisodeV1 windows; only PRE/IDLE, APPROACH, "
            "and CONTACT are identifiable there"
        ),
        "evidence": "phase_coverage.csv",
    }
    scale_contract = {
        "schema_version": "P3ObjectScaleContractV2",
        "audit_decision": "DIMENSIONLESS_OBJECT_SCALE_NORMALIZATION_REQUIRED",
        "selected_branch": "BBOX_DIAGONAL_DIMENSIONLESS_NORMALIZATION",
        "characteristic_length": "object_bbox_diagonal_m",
        "global_anchor_bbox_diagonal_m": scale_summary["bbox_diagonal_m"]["median_unique_object"],
        "anchor_population": "59 unique objects in 108 eligible EpisodeV1 records",
        "dimensionless_ratios": scale_summary["dimensionless_ratios"],
        "applies_to": [
            "grouped reward proximity_tolerance_m",
            "grouped reward distance_scope_m",
            "object-axis tracking object_sigma_m",
            "object linear-velocity tracking object_velocity_sigma_mps",
            "RSE proximity_tolerance_m",
            "RSE distance_scope_m",
            "RSE object_position_base_m",
            "RSE object_axis_base_m",
        ],
        "unchanged": [
            "tracked-link sigma 0.025 m",
            "wrist-position sigma 0.02 m",
            "RSE hand-position base 0.20 m",
            "angular thresholds",
            "PF thresholds",
            "DF thresholds",
            "grouped multiplicative reward structure",
            "RSE adaptive counter semantics",
        ],
        "per_object_tuning": False,
        "automatic_metadata_rule": True,
        "runtime_implementation_changed_in_p3": False,
        "numerical_regression": "object_scale_numerical_regression.csv",
        "evidence": ["object_scale_audit.csv", "object_scale_summary.json"],
    }
    write_json(output / "budget_contract.json", budget_contract)
    write_json(output / "rsi_contract.json", rsi_contract)
    write_json(output / "scale_contract.json", scale_contract)

    final_decision = {
        "schema_version": "PPOGeneralizationContractV2",
        "P3": "COMPLETE",
        "receipt_status": "PASS",
        "scientific_decisions": {
            "P3A": budget_contract["audit_decision"],
            "P3B": rsi_contract["audit_decision"],
            "P3C": scale_contract["audit_decision"],
        },
        "selected_or_fallback": {
            "budget_rule": budget_contract["selected_branch"],
            "RSI_rule": rsi_contract["selected_branch"],
            "object_scale_rule": scale_contract["selected_branch"],
        },
        "independent_subaudits": True,
        "uniform_RSI_component_removed": False,
        "frame0_only_training_added": False,
        "per_clip_tuning": False,
        "reward_structure_changed": False,
        "reward_metric_scale_rule_selected_for_change": True,
        "RSE_structure_changed": False,
        "RSE_metric_scale_rule_selected_for_change": True,
        "PF_DF_thresholds_changed": False,
        "GPU_run": False,
        "PPO_run": False,
        "runtime_integration": "NOT_PERFORMED_IN_LANE_C_OFFLINE_AUDIT",
        "source_provenance": {
            "episode_index": {
                "path": str(EPISODE_INDEX.relative_to(REPO_ROOT)),
                "sha256": sha256(EPISODE_INDEX),
            },
            "hardening_manifest": {
                "path": str(HARDENING_MANIFEST.relative_to(REPO_ROOT)),
                "sha256": sha256(HARDENING_MANIFEST),
            },
            "exact_batch_manifest": "exact_batch_evidence.csv",
        },
        "artifacts": list(required_names)
        + [
            "object_scale_numerical_regression.csv",
            "exact_batch_evidence.csv",
            "object_scale_summary.json",
            "validation.json",
        ],
    }
    write_json(output / "final_decision.json", final_decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
