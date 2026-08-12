#!/usr/bin/env python3
"""Validate the named 21-body Formal20 telemetry artifact without IsaacLab."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.evaluation.full_hand_contact import HAND_BODY_GROUPS  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"FULL_PAIR_TELEMETRY_JSON_OBJECT_REQUIRED:{path}")
    return value


def qualify(trace_path: Path, evaluation_path: Path, *, clip: str) -> dict[str, object]:
    """Check schema and diagnostic isolation before the offline audit runs."""

    with np.load(trace_path, allow_pickle=False) as archive:
        required = {
            "replica_hand_object_pair_force_world",
            "replica_hand_object_pair_presence",
            "replica_hand_object_pair_force_valid",
            "hand_body_names",
            "hand_body_indices",
            "hand_body_groups",
            "hand_collision_shape_mapping",
            "hand_palm_mapping",
            "replica_object_pose",
            "replica_hand_collision_body_pose",
            "replica_action",
            "replica_reference_index",
            "replica_reason_code",
            "replica_terminated",
            "replica_timed_out",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"FULL_PAIR_TELEMETRY_FIELDS_MISSING:{missing}")
        force = np.asarray(archive["replica_hand_object_pair_force_world"])
        presence = np.asarray(archive["replica_hand_object_pair_presence"], dtype=bool)
        valid = np.asarray(archive["replica_hand_object_pair_force_valid"], dtype=bool)
        names = tuple(str(value) for value in archive["hand_body_names"].tolist())
        indices = np.asarray(archive["hand_body_indices"], dtype=np.int64)
        groups = tuple(str(value) for value in archive["hand_body_groups"].tolist())
        clip_value = str(np.asarray(archive["requested_clip"]).item())
    if force.shape != (321, 20, 21, 3):
        raise ValueError(f"FULL_PAIR_TELEMETRY_FORCE_SHAPE_INVALID:{force.shape}")
    if presence.shape != (321, 20, 21):
        raise ValueError(f"FULL_PAIR_TELEMETRY_PRESENCE_SHAPE_INVALID:{presence.shape}")
    if valid.shape not in {(321, 20), (321, 20, 21)}:
        raise ValueError(f"FULL_PAIR_TELEMETRY_VALIDITY_SHAPE_INVALID:{valid.shape}")
    valid_per_pair = valid if valid.ndim == 3 else np.broadcast_to(valid[..., None], presence.shape)
    if valid_per_pair[0].any() or not valid_per_pair[1:].all():
        raise ValueError("FULL_PAIR_TELEMETRY_RESET_VALIDITY_INVALID")
    if presence[0].any() or np.any(force[0] != 0.0):
        raise ValueError("FULL_PAIR_TELEMETRY_RESET_ROW_MUST_BE_EXPLICITLY_EMPTY")
    if not np.isfinite(force[valid_per_pair]).all():
        raise ValueError("FULL_PAIR_TELEMETRY_FORCE_NONFINITE")
    expected_presence = np.linalg.norm(force, axis=-1) > 1.0e-4
    if not np.array_equal(presence, expected_presence):
        raise ValueError("FULL_PAIR_TELEMETRY_PRESENCE_FORCE_MISMATCH")
    if len(names) != 21 or len(set(names)) != 21 or not np.array_equal(indices, np.arange(21)):
        raise ValueError("FULL_PAIR_TELEMETRY_BODY_MAPPING_INVALID")
    if len(groups) != 21 or any(group not in HAND_BODY_GROUPS for group in groups):
        raise ValueError("FULL_PAIR_TELEMETRY_BODY_GROUP_INVALID")
    if clip_value != clip:
        raise ValueError("FULL_PAIR_TELEMETRY_ACTIVE_OBJECT_MISMATCH")
    evaluation = _read_json(evaluation_path)
    ppo = evaluation.get("physics_contract", {}).get("ppo26d", {})
    writes = {
        "object_rollout_state_writes": ppo.get("object_rollout_state_writes"),
        "wrist_root_state_writes_during_step": ppo.get("wrist_root_state_writes_during_step"),
        "hidden_force_or_attachment": ppo.get("hidden_force_or_attachment"),
    }
    if writes != {
        "object_rollout_state_writes": 0,
        "wrist_root_state_writes_during_step": 0,
        "hidden_force_or_attachment": False,
    }:
        raise ValueError(f"FULL_PAIR_TELEMETRY_DIAGNOSTIC_ISOLATION_FAILED:{writes}")
    return {
        "schema_version": "FullHandObjectPairTelemetryV1",
        "status": "FULL_PAIR_TELEMETRY_QUALIFIED",
        "clip": clip,
        "trace": str(trace_path.resolve()),
        "trace_sha256": _sha256(trace_path),
        "shapes": {
            "replica_hand_object_pair_force_world": list(force.shape),
            "replica_hand_object_pair_presence": list(presence.shape),
            "replica_hand_object_pair_force_valid": list(valid.shape),
        },
        "body_mapping": {"names": list(names), "indices": indices.tolist(), "groups": list(groups)},
        "active_object": clip,
        "force": {
            "frame": "world",
            "units": "N",
            "semantics": "force on active object from named hand collision body",
            "finite_valid_samples": True,
        },
        "frame_zero": "explicitly_invalid_no_post_physics_sensor_sample",
        "contamination": {
            "self_collision": (
                "not a hand-object pair because the object-side sensor filters only hand bodies"
            ),
            "inactive_object": (
                "excluded by active object-side sensor selection and fixed clip check"
            ),
        },
        "diagnostic_isolation": writes,
        "palm_mapping": _palm_mapping(trace_path),
    }


def _palm_mapping(trace_path: Path) -> dict[str, object]:
    with np.load(trace_path, allow_pickle=False) as archive:
        return json.loads(str(np.asarray(archive["hand_palm_mapping"]).item()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--clip", required=True, choices=("hocap_170105", "hocap_170650"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(args.trace.resolve(), args.evaluation.resolve(), clip=args.clip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
