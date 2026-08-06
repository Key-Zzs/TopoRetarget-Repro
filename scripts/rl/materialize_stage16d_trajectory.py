#!/usr/bin/env python3
"""Materialize a non-destructive PhysicsConsistentRetargetedTrajectoryV1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from toporetarget.rl.physics_retargeting.export import export_physics_consistent_trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retime(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] != 41:
        raise ValueError("Stage16D source reference must have 41 frames")
    return np.concatenate((np.repeat(array[:-1], 8, axis=0), array[-1:]), axis=0)


def load_reference(clip: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = (
        REPO_ROOT
        / ".local/stage16_reference_tracking_ppo/world_wrist_references"
        / f"{clip}.world_wrist.stage16.npz"
    )
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
    return path, arrays


def write_contacts(path: Path, trace: dict[str, np.ndarray], body_names: list[str]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Stage16D contacts.parquet requires pyarrow") from exc
    presence = np.asarray(trace["contact_pair_presence"], dtype=bool)
    forces = np.asarray(trace["contact_force_world"], dtype=np.float64)
    impulses = np.asarray(trace["contact_impulse_world"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    duration = 0
    for step in range(321):
        duration = duration + 1 if bool(presence[step].any()) else 0
        rows.append(
            {
                "step": step,
                "body_pairs": [body_names[i] for i in np.flatnonzero(presence[step])],
                "force_x_n": float(forces[step, 0]),
                "force_y_n": float(forces[step, 1]),
                "force_z_n": float(forces[step, 2]),
                "impulse_x_ns": float(impulses[step, 0]),
                "impulse_y_ns": float(impulses[step, 1]),
                "impulse_z_ns": float(impulses[step, 2]),
                "contact_duration_steps": duration,
                "separation_penetration": "independent_geometry_audit",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def main() -> int:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Stage16D refuses overwrite: {args.output}")
    qualification = json.loads(args.qualification.read_text(encoding="utf-8"))
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    with np.load(args.trace, allow_pickle=False) as source:
        trace = {name: np.asarray(source[name]) for name in source.files}
    actions = np.asarray(np.load(args.actions, allow_pickle=False), dtype=np.float32)
    reference_path, reference = load_reference(args.clip)
    source_wrist_pose = np.concatenate(
        (
            reference["wrist_pose_translation_world_ref"],
            reference["wrist_pose_quaternion_world_ref_wxyz"],
        ),
        axis=-1,
    )
    source_object_pose = np.concatenate(
        (
            reference["object_pose_translation_world_ref"],
            reference["object_pose_quaternion_world_ref_wxyz"],
        ),
        axis=-1,
    )
    penetration = np.asarray(
        geometry["per_step_replica"]["penetration_lower_bound_m"], dtype=np.float64
    )[:, 0]
    arrays = {
        "actions": actions,
        "observation": trace["observation"],
        "wrist_pose": trace["wrist_pose"],
        "wrist_twist": trace["wrist_twist"],
        "virtual_wrist_q": trace["virtual_wrist_q"],
        "virtual_wrist_qd": trace["virtual_wrist_qd"],
        "finger_q": trace["finger_q"],
        "finger_qd": trace["finger_qd"],
        "targets": trace["joint_targets"],
        "actuator_effort": trace["actuator_effort"],
        "saturation": trace["saturation"],
        "object_pose": trace["object_pose"],
        "object_twist": trace["object_twist"],
        "object_axis_points": trace["object_axis_points"],
        "hand_collision_body_pose": trace["hand_collision_body_pose"],
        "contact_pair_presence": trace["contact_pair_presence"],
        "contact_force": trace["contact_force_world"],
        "contact_impulse": trace["contact_impulse_world"],
        "contact_duration": np.maximum.accumulate(trace["contact_persistence"]),
        "penetration_lower_bound": penetration,
        "semantic_progress": trace["semantic_progress"],
        "contact_recall": trace["contact_recall"],
        "contact_persistence": trace["contact_persistence"],
        "contact_causality": trace["contact_causality"],
        "terminal_stable": trace["terminal_stable"],
        "termination_reason_code": trace["reason_code"],
        "source_wrist_pose": retime(source_wrist_pose),
        "source_finger_q": retime(reference["q_finger_ref"]),
        "source_object_pose": retime(source_object_pose),
        "source_reference_index": np.minimum(np.arange(321) // 8, 40),
    }
    manifest = {
        "clip": args.clip,
        "source_reference": str(reference_path.resolve()),
        "source_reference_sha256": sha256(reference_path),
        "corrected_action_source": str(args.actions.resolve()),
        "corrected_action_sha256": sha256(args.actions),
        "qualification": str(args.qualification.resolve()),
        "qualification_sha256": sha256(args.qualification),
        "geometry_audit": str(args.geometry.resolve()),
        "geometry_audit_sha256": sha256(args.geometry),
        "source_key_indices": list(range(41)),
        "factor_8_mapping": "retimed_index -> min(retimed_index // 8, 40)",
        "physics_contract": "nominal_stage16_physx_contract_v1",
        "random_seed": 20260806,
        "formal_object_state_writes": 0,
        "formal_wrist_state_writes": 0,
        "hidden_force": False,
        "hidden_attachment": False,
    }
    quality = {
        "schema_version": "Stage16DPhysicsConsistentTrajectoryQualityV1",
        "status": "STAGE16D_PHYSICS_CONSISTENT_TRAJECTORY_PARTIAL_BLOCKED",
        "task_success_rate": qualification["success_rate"],
        "semantic_reach_rate": qualification["semantic_reach_rate"],
        "contact_topology_pass_rate": qualification["contact_topology_pass_rate"],
        "contact_causality_pass_rate": qualification["contact_causality_pass_rate"],
        "terminal_stability_pass_rate": qualification["terminal_stability_pass_rate"],
        "source_robot_deviation": "stored_in_trajectory_arrays",
        "source_object_deviation": "stored_in_trajectory_arrays",
        "penetration": geometry["collision_proxy"],
        "geometry_gate": geometry["formal_geometry_gate"],
        "action_smoothness_l2_mean": float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean()),
        "termination": "see termination_reason_code",
        "blockers": [
            "independent penetration result is a lower bound",
            "runtime visual OBJ has no watertight sign",
            "Stage12 SDF metric is not directly comparable to runtime collision proxy",
        ],
    }
    outputs = export_physics_consistent_trajectory(
        output_dir=args.output, arrays=arrays, manifest=manifest, quality=quality
    )
    body_names = [str(value) for value in trace["hand_collision_body_names"]]
    contacts_path = args.output / "contacts.parquet"
    write_contacts(contacts_path, trace, body_names)
    comparison_path = args.output / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "wrist_translation_deviation_m",
                "object_translation_deviation_m",
                "penetration_lower_bound_m",
                "semantic_progress",
                "contact_recall",
            ),
        )
        writer.writeheader()
        for step in range(321):
            writer.writerow(
                {
                    "step": step,
                    "wrist_translation_deviation_m": float(
                        np.linalg.norm(
                            arrays["wrist_pose"][step, :3]
                            - source_wrist_pose[min(step // 8, 40), :3]
                        )
                    ),
                    "object_translation_deviation_m": float(
                        np.linalg.norm(
                            arrays["object_pose"][step, :3]
                            - source_object_pose[min(step // 8, 40), :3]
                        )
                    ),
                    "penetration_lower_bound_m": float(penetration[step]),
                    "semantic_progress": float(arrays["semantic_progress"][step]),
                    "contact_recall": float(arrays["contact_recall"][step]),
                }
            )
    outputs.update({"contacts": str(contacts_path), "comparison": str(comparison_path)})
    print(json.dumps({"status": quality["status"], "outputs": outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
