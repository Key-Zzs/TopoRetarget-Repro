#!/usr/bin/env python3
"""Extract shared Stage 16-D task and contact-topology contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.physics_retargeting.contact_topology import (  # noqa: E402
    extract_persistent_contact_topology,
)
from toporetarget.rl.physics_retargeting.contracts import derive_task_gate  # noqa: E402
from toporetarget.rl.physics_retargeting.rewards import (  # noqa: E402
    PhysicsConsistentRewardProfileV1,
)
from toporetarget.rl.physics_retargeting.task_semantics import (  # noqa: E402
    extract_task_semantics,
)

CLIPS = ("hocap_170105", "hocap_170650")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".local/reports/stage16d_physics_consistent_retargeting",
    )
    parser.add_argument("--reference-time-scale", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write(path: Path, payload: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Stage16D refuses overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def object_bbox_diagonal(path: Path) -> float:
    vertices: list[list[float]] = []
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(values[1]), float(values[2]), float(values[3])])
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"invalid object OBJ vertices: {path}")
    return float(np.linalg.norm(array.max(axis=0) - array.min(axis=0)))


def main() -> int:
    args = parse_args()
    output = args.output_root.resolve()
    semantics: dict[str, Any] = {}
    topology: dict[str, Any] = {}
    gates: dict[str, Any] = {}
    for clip in CLIPS:
        reference_path = (
            REPO_ROOT
            / ".local/stage16_reference_tracking_ppo/world_wrist_references"
            / f"{clip}.world_wrist.stage16.npz"
        )
        worker_path = (
            REPO_ROOT
            / ".local/reports/stage16c3r5_reference_retiming_c4"
            / ".contact_causality_scale8_workers"
            / f"{clip}.json"
        )
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        with np.load(reference_path, allow_pickle=False) as source:
            reference = {
                name: np.asarray(source[name]) for name in source.files if name != "metadata"
            }
        semantic = extract_task_semantics(
            clip=clip,
            reference=reference,
            contact_records=worker["contact_records"],
            reference_time_scale=args.reference_time_scale,
        )
        contact = extract_persistent_contact_topology(
            clip=clip,
            contact_records=worker["contact_records"],
            retimed_frame_count=semantic.retimed_frame_count,
        )
        mesh = REPO_ROOT / f".local/stage16_reference_tracking_ppo/world_wrist_objects/{clip}.obj"
        gate = derive_task_gate(semantic, object_bbox_diagonal_m=object_bbox_diagonal(mesh))
        semantics[clip], topology[clip], gates[clip] = (
            semantic.as_dict(),
            contact.as_dict(),
            gate.as_dict(),
        )
        suffix = clip.removeprefix("hocap_")
        write(
            output / f"task_semantics_{suffix}.json", semantic.as_dict(), overwrite=args.overwrite
        )
        write(output / f"{clip}.task_semantics.json", semantic.as_dict(), overwrite=args.overwrite)
    comparison = {
        "schema_version": "Stage16DTaskSemanticComparisonV1",
        "shared_extraction_algorithm": True,
        "clip_specific_conditionals": False,
        "clips": {
            clip: {
                "task_class": semantics[clip]["task_class"],
                "source_motion_class": semantics[clip]["source_motion_class"],
                "confidence": semantics[clip]["classification_confidence"],
                "translation_m": semantics[clip]["source_object_translation_m"],
                "rotation_deg": semantics[clip]["source_object_rotation_deg"],
                "contact_steps": semantics[clip]["source_contact_control_steps"],
            }
            for clip in CLIPS
        },
        "status": "STAGE16D_TASK_SEMANTICS_PARTIAL",
        "reason": "both validated C3 traces are too sparse for high-confidence persistent topology",
    }
    reward = PhysicsConsistentRewardProfileV1()
    reward_contract = {
        "schema_version": "physics_consistent_retargeting_reward_v1",
        "selected_globally_before_optimization": True,
        "selection_rule": "shared worst-case semantic/contact support across both clips",
        "available_profiles": [
            "semantic_balanced_v1",
            "contact_priority_v1",
            "source_fidelity_priority_v1",
        ],
        "profile": reward.as_dict(),
        "strict_object_world_tracking_hard_reward": False,
    }
    termination = {
        "schema_version": "Stage16DTerminationContractV1",
        "failure": [
            "numerical_failure",
            "simulator_failure",
            "catastrophic_penetration",
            "object_leaves_workspace",
            "wrist_safety_violation",
            "joint_limit_safety_violation",
            "action_invalid",
            "uncontrolled_object_explosion",
            "timeout",
        ],
        "not_immediate_failure": [
            "source_object_world_pose_deviation",
            "source_object_axis_error",
            "source_object_orientation_error",
        ],
        "success": [
            "semantic_task_progress_complete",
            "required_contact_topology_satisfied",
            "stable_terminal_window",
            "no_hard_failure",
        ],
    }
    anti = {
        "schema_version": "PhysicsConsistentTaskGateV1Set",
        "frozen_before_optimization": True,
        "clips": gates,
    }
    write(output / "task_semantic_comparison.json", comparison, overwrite=args.overwrite)
    write(
        output / "contact_topology.json",
        {"schema_version": "PersistentContactTopologyV1Set", "clips": topology},
        overwrite=args.overwrite,
    )
    write(
        output / "contact_topology_contract.json",
        {"schema_version": "PersistentContactTopologyV1Set", "clips": topology},
        overwrite=args.overwrite,
    )
    write(output / "reward_contract.json", reward_contract, overwrite=args.overwrite)
    write(output / "termination_contract.json", termination, overwrite=args.overwrite)
    write(output / "anti_degenerate_contract.json", anti, overwrite=args.overwrite)
    print(
        json.dumps(
            {"status": comparison["status"], "output_root": str(output), "clips": list(CLIPS)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
