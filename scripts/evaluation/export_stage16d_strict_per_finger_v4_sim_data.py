#!/usr/bin/env python3
"""Export one completed Strict Per-Finger V4 Formal20 run without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

FRAME_COUNT = 321
EPISODE_COUNT = 20
CONTROL_DT_S = 1.0 / 20.0
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_EXPORT_JSON_OBJECT_REQUIRED:{path}")
    return value


def _arrays(trace_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(trace_path, allow_pickle=False) as archive:
        required = {
            "replica_object_pose",
            "replica_object_twist",
            "replica_wrist_pose",
            "replica_wrist_twist_world",
            "replica_virtual_wrist_q",
            "replica_virtual_wrist_qdot",
            "replica_finger_q",
            "replica_finger_qdot",
            "replica_action",
            "replica_wrist_target_pose",
            "replica_finger_target_q",
            "replica_actuator_effort",
            "replica_source_contact_mask",
            "replica_tip_pair_presence",
            "replica_tip_pair_force_world",
            "replica_hand_object_pair_presence",
            "replica_hand_object_pair_force_world",
            "replica_hand_object_pair_force_valid",
            "replica_per_finger_contact_reward",
            "replica_r_contact_v4",
            "replica_reward_total",
            "replica_object_twist_reference",
            "replica_embedded_reference_object_pose",
            "replica_reference_index",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"STRICT_V4_EXPORT_TRACE_FIELDS_MISSING:{missing}")
        values = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name.startswith("replica_") and np.asarray(archive[name]).shape[:2] == (321, 20)
        }
        metadata = {
            "clip": str(np.asarray(archive["clip"]).item()),
            "checkpoint_path": str(np.asarray(archive["checkpoint_path"]).item()),
            "checkpoint_sha256": str(np.asarray(archive["checkpoint_sha256"]).item()),
            "reference_hash": str(np.asarray(archive["reference_hash"]).item()),
            "reference_kinematics_version": int(
                np.asarray(archive["reference_kinematics_version"]).item()
            ),
            "reward_v4_samples": int(np.asarray(archive["reward_v4_samples"]).item()),
            "action_contract": str(np.asarray(archive["action_contract"]).item()),
            "fingertip_link_names": [
                str(value) for value in np.asarray(archive["fingertip_link_names"]).tolist()
            ],
            "hand_body_names": [
                str(value) for value in np.asarray(archive["hand_body_names"]).tolist()
            ],
        }
    for name, value in values.items():
        if value.shape[:2] != (FRAME_COUNT, EPISODE_COUNT):
            raise ValueError(f"STRICT_V4_EXPORT_TRACE_SHAPE_INVALID:{name}:{value.shape}")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"STRICT_V4_EXPORT_TRACE_NONFINITE:{name}")
    if metadata["action_contract"] != "26D_reference_residual":
        raise ValueError("STRICT_V4_EXPORT_ACTION_CONTRACT_DRIFT")
    return values, metadata


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError("STRICT_V4_EXPORT_PARQUET_ROWS_EMPTY")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _write_zarr(
    output: Path,
    *,
    values: dict[str, np.ndarray],
    metadata: dict[str, Any],
    source_classes: np.ndarray,
) -> None:
    import zarr

    root = zarr.open_group(str(output), mode="w-")
    root.attrs.update(
        {
            "schema_version": "Stage16DStrictPerFingerV4FormalSimulationDataV1",
            "reward_version": "V4",
            "control_dt_s": CONTROL_DT_S,
            "formal_frame_count": FRAME_COUNT,
            "formal_episode_count": EPISODE_COUNT,
            "causal_physics": True,
            "external_guidance": False,
            **metadata,
        }
    )
    groups = {
        "robot": (
            "wrist_pose",
            "wrist_twist_world",
            "virtual_wrist_q",
            "virtual_wrist_qdot",
            "virtual_wrist_target_q",
            "virtual_wrist_target_qdot",
            "finger_q",
            "finger_qdot",
            "action",
            "wrist_target_pose",
            "finger_target_q",
            "actuator_effort",
        ),
        "object": ("object_pose", "object_twist", "object_axis_points"),
        "reference": (
            "reference_index",
            "embedded_reference_object_pose",
            "embedded_reference_wrist_pose",
            "embedded_reference_finger_q",
            "embedded_reference_tracked_links",
            "object_twist_reference",
        ),
        "source_semantics": ("source_contact_mask",),
        "contact": (
            "tip_pair_presence",
            "tip_pair_force_world",
            "hand_object_pair_presence",
            "hand_object_pair_force_world",
            "hand_object_pair_force_valid",
        ),
        "reward": (
            "reward_total",
            "reward_object",
            "reward_link",
            "reward_finger",
            "reward_wrist_translation",
            "reward_wrist_rotation",
            "reward_smoothness",
            "reward_obj_vel",
            "reward_obj_ang_vel",
            "per_finger_contact_reward",
            "r_contact_v4",
        ),
    }
    episodes = root.create_group("episodes")
    for episode in range(EPISODE_COUNT):
        group = episodes.create_group(f"episode_{episode:03d}")
        group.attrs.update(
            {
                "episode": episode,
                "causal_physics": True,
                "external_guidance": False,
                "source_contact_localization": "NOT_MATERIALIZED_IN_RUNTIME_EVIDENCE_V1",
            }
        )
        for group_name, names in groups.items():
            child = group.create_group(group_name)
            for name in names:
                key = f"replica_{name}"
                if key not in values:
                    continue
                child.create_dataset(name, data=values[key][:, episode], overwrite=False)
            if group_name == "source_semantics":
                child.create_dataset("source_contact_class", data=source_classes, overwrite=False)


def _episode_rows(
    *, qualification: dict[str, Any], suite: dict[str, Any], audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qualification_rows = qualification.get("episodes")
    suite_rows = suite.get("episodes")
    audit_rows = audit.get("per_replica")
    if (
        not isinstance(qualification_rows, list)
        or not isinstance(suite_rows, list)
        or not isinstance(audit_rows, list)
        or len(qualification_rows) != EPISODE_COUNT
        or len(suite_rows) != EPISODE_COUNT
        or len(audit_rows) != EPISODE_COUNT
    ):
        raise ValueError("STRICT_V4_EXPORT_FORMAL20_ROWS_MISSING")
    rows: list[dict[str, Any]] = []
    for episode, (q, suite_row, contact) in enumerate(
        zip(qualification_rows, suite_rows, audit_rows, strict=True)
    ):
        if int(q["replica"]) != episode or int(suite_row["replica"]) != episode:
            raise ValueError("STRICT_V4_EXPORT_REPLICA_ORDER_INVALID")
        source_pass = bool(
            float(contact["persistent_source_tip_recall"] or 0.0) >= 0.50
            and float(contact["source_tip_recall"] or 0.0) >= 0.50
        )
        rows.append(
            {
                "episode": episode,
                "seed": int(q["seed"]),
                "kinematic_success": bool(suite_row["kinematic_success"]),
                "physics_success": bool(suite_row["physics_success"]),
                "qualified_success": bool(suite_row["qualified_success"]),
                "source_contact_semantics_pass_v1": source_pass,
                "SR_interaction_qualified_v1": bool(suite_row["qualified_success"]) and source_pass,
                "source_tip_recall": contact["source_tip_recall"],
                "persistent_source_tip_recall": contact["persistent_source_tip_recall"],
                "full_source_tip_coverage_rate": contact["full_source_tip_coverage_rate"],
                "longest_source_contact_loss_gap": contact["longest_source_contact_loss_gap"],
                "source_contact_loss_event_count": contact["source_contact_loss_event_count"],
                "recontact_event_count": contact["recontact_event_count"],
                "E_r_mean_deg": suite_row["E_r_mean_deg"],
                "E_t_mean_cm": suite_row["E_t_mean_cm"],
                "E_j_mean_cm": suite_row["E_j_mean_cm"],
                "E_ft_mean_cm": suite_row["E_ft_mean_cm"],
                "terminal_delta_v_mps": q["terminal_delta_v_mps"],
                "terminal_delta_omega_radps": q["terminal_delta_omega_radps"],
                "max_inter_finger_penetration_m": q["max_inter_finger_penetration_m"],
            }
        )
    return rows, audit_rows


def _source_rows(
    *, values: dict[str, np.ndarray], source_classes: np.ndarray, audit: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = np.asarray(values["replica_source_contact_mask"], dtype=bool)
    presence = np.asarray(values["replica_tip_pair_presence"], dtype=bool)
    force = np.asarray(values["replica_tip_pair_force_world"], dtype=np.float64)
    reward = np.asarray(values["replica_per_finger_contact_reward"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for episode in range(EPISODE_COUNT):
        for frame in range(FRAME_COUNT):
            for finger, name in enumerate(FINGERS):
                rows.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "finger": name,
                        "source_contact_mask": bool(expected[frame, episode, finger]),
                        "source_contact_class": str(source_classes[frame, finger]),
                        "source_contact_localization": "NOT_MATERIALIZED_IN_RUNTIME_EVIDENCE_V1",
                        "tip_pair_presence": bool(presence[frame, episode, finger]),
                        "tip_pair_force_x_n": float(force[frame, episode, finger, 0]),
                        "tip_pair_force_y_n": float(force[frame, episode, finger, 1]),
                        "tip_pair_force_z_n": float(force[frame, episode, finger, 2]),
                        "tip_pair_force_norm_n": float(
                            np.linalg.norm(force[frame, episode, finger])
                        ),
                        "per_finger_contact_reward": float(reward[frame, episode, finger]),
                    }
                )
    return rows


def _pair_rows(values: dict[str, np.ndarray], hand_names: list[str]) -> list[dict[str, Any]]:
    presence = np.asarray(values["replica_hand_object_pair_presence"], dtype=bool)
    force = np.asarray(values["replica_hand_object_pair_force_world"], dtype=np.float64)
    valid = np.asarray(values["replica_hand_object_pair_force_valid"], dtype=bool)
    rows: list[dict[str, Any]] = []
    for episode in range(EPISODE_COUNT):
        for frame in range(FRAME_COUNT):
            for body, name in enumerate(hand_names):
                rows.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "hand_body_index": body,
                        "hand_body_name": name,
                        "pair_force_valid": bool(valid[frame, episode]),
                        "pair_presence": bool(presence[frame, episode, body]),
                        "pair_force_x_n": float(force[frame, episode, body, 0]),
                        "pair_force_y_n": float(force[frame, episode, body, 1]),
                        "pair_force_z_n": float(force[frame, episode, body, 2]),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--evaluation-suite", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--source-runtime", type=Path, required=True)
    parser.add_argument("--strict-v4-contract", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_path = args.trace.resolve()
    output = args.output.resolve()
    qualification_path = args.qualification.resolve()
    suite_path = args.evaluation_suite.resolve()
    audit_path = args.source_audit.resolve()
    contract_path = args.strict_v4_contract.resolve()
    reference_path = args.reference.resolve()
    if output.exists():
        raise FileExistsError(f"STRICT_V4_EXPORT_OUTPUT_ALREADY_EXISTS:{output}")
    values, metadata = _arrays(trace_path)
    qualification = _read(qualification_path)
    suite = _read(suite_path)
    audit = _read(audit_path)
    contract = _read(contract_path)
    if (
        qualification.get("status") != "STAGE16D_STRICT_V4_FORMAL_COMPLETE"
        or suite.get("schema_version") != "TopoRetargetEvaluationSuiteV2ResultV1"
        or audit.get("status") != "STRICT_V4_SOURCE_CONTACT_AUDIT_COMPLETE"
        or contract.get("status") != "STRICT_V4_CONTACT_CONTRACT_FROZEN"
        or qualification.get("trace_sha256") != _sha256(trace_path)
        or audit.get("trace", {}).get("sha256") != _sha256(trace_path)
    ):
        raise ValueError("STRICT_V4_EXPORT_REQUIRES_COMPLETED_FROZEN_FORMAL_INPUTS")
    with np.load(args.source_runtime.resolve(), allow_pickle=False) as source:
        classes = np.asarray(source["class_label"])
        names = tuple(str(value) for value in source["finger_order"].tolist())
    if classes.shape != (FRAME_COUNT, 5) or names != FINGERS:
        raise ValueError("STRICT_V4_EXPORT_SOURCE_RUNTIME_CONTRACT_INVALID")
    episode_rows, _ = _episode_rows(qualification=qualification, suite=suite, audit=audit)
    source_rows = _source_rows(values=values, source_classes=classes, audit=audit)
    pair_rows = _pair_rows(values, metadata["hand_body_names"])
    output.mkdir(parents=True, exist_ok=False)
    enriched_metadata = {
        **metadata,
        "reward_contract_sha256": _sha256(contract_path),
        "source_contact_contract_sha256": _sha256(contract_path),
        "physics_contract_sha256": qualification.get("physics_contract_sha256"),
        "source_contact_semantics_pass_v1": "source and persistent tip recall each >= 0.50",
    }
    _write_zarr(
        output / "rollouts.zarr",
        values=values,
        metadata=enriched_metadata,
        source_classes=classes,
    )
    _write_parquet(output / "per_episode_metrics.parquet", episode_rows)
    _write_parquet(output / "source_contact_metrics.parquet", source_rows)
    _write_parquet(output / "pair_contact_metrics.parquet", pair_rows)
    shutil.copyfile(reference_path, output / "reference.npz")
    summary = {
        "schema_version": "Stage16DStrictPerFingerV4FormalEvaluationSummaryV1",
        "clip": metadata["clip"],
        "evaluation_suite": suite,
        "source_contact": audit["aggregate"],
        "qualification": {
            "physics_qualified": qualification["physics_qualified"],
            "ppo_task_success_rate": qualification["ppo_task_success_rate"],
        },
        "SR_source_contact_v1": float(
            np.mean([row["source_contact_semantics_pass_v1"] for row in episode_rows])
        ),
        "SR_interaction_qualified_v1": float(
            np.mean([row["SR_interaction_qualified_v1"] for row in episode_rows])
        ),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "Stage16DStrictPerFingerV4FormalSimulationManifestV1",
        "status": "STAGE16D_STRICT_V4_FORMAL_SIM_DATA_EXPORTED",
        "clip": metadata["clip"],
        "episode_count": EPISODE_COUNT,
        "frame_count": FRAME_COUNT,
        "qualified_episode_indices": [
            row["episode"] for row in episode_rows if row["qualified_success"]
        ],
        "failed_episode_indices": [
            row["episode"] for row in episode_rows if not row["qualified_success"]
        ],
        "metadata": enriched_metadata,
        "inputs": {
            "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
            "qualification": {
                "path": str(qualification_path),
                "sha256": _sha256(qualification_path),
            },
            "evaluation_suite": {"path": str(suite_path), "sha256": _sha256(suite_path)},
            "source_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
            "strict_v4_contract": {"path": str(contract_path), "sha256": _sha256(contract_path)},
            "reference": {"path": str(reference_path), "sha256": _sha256(reference_path)},
        },
        "files": {
            name: _sha256(output / name)
            for name in (
                "per_episode_metrics.parquet",
                "source_contact_metrics.parquet",
                "pair_contact_metrics.parquet",
                "reference.npz",
                "evaluation_summary.json",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
