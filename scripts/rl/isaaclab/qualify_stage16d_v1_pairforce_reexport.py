#!/usr/bin/env python3
"""Qualify a V1 Formal20 re-export that adds only exact pair-force telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTINUATION_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d_continuation"
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock/v1_pairforce"
FINGER_LINKS = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)
FINGER_INDICES = (20, 4, 8, 16, 12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        "max_abs": float(np.abs(values).max()),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", choices=("hocap_170105", "hocap_170650"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    output = args.output_root.resolve() / args.clip
    trace_path = output / "trace.npz"
    evaluation_path = output / "r7_pairforce_evaluation.json"
    old_root = CONTINUATION_ROOT / args.clip
    old_trace_path = old_root / "ppo_r7_formal_trace_replica0.npz"
    old_evaluation_path = old_root / "r7_formal_evaluation.json"
    frozen_inputs_path = args.output_root.resolve().parent / "v1_pairforce_frozen_inputs.json"
    if not all(
        path.is_file()
        for path in (
            trace_path,
            evaluation_path,
            old_trace_path,
            old_evaluation_path,
            frozen_inputs_path,
        )
    ):
        raise FileNotFoundError("V1_PAIRFORCE_REEXPORT_QUALIFICATION_INPUT_MISSING")
    frozen = _load_json(frozen_inputs_path)
    source = frozen["clips"][args.clip]
    evaluation = _load_json(evaluation_path)
    old_evaluation = _load_json(old_evaluation_path)
    if evaluation["checkpoint_sha256"] != source["checkpoint_sha256"]:
        raise ValueError("V1_PAIRFORCE_REEXPORT_CONTRACT_MISMATCH:checkpoint")
    if [int(row["seed"]) for row in evaluation["frame_zero"]] != source["formal_seeds"]:
        raise ValueError("V1_PAIRFORCE_REEXPORT_CONTRACT_MISMATCH:seeds")
    with (
        np.load(old_trace_path, allow_pickle=False) as old,
        np.load(trace_path, allow_pickle=False) as new,
    ):
        required = {
            "replica_fingertip_object_pair_force_world",
            "replica_fingertip_object_pair_force_valid",
            "fingertip_link_names",
            "fingertip_force_sensor_indices",
            "pair_force_frame",
            "pair_force_units",
            "pair_force_semantics",
        }
        missing = sorted(required.difference(new.files))
        if missing:
            raise ValueError(f"V1_PAIRFORCE_REEXPORT_FIELD_MISSING:{missing}")
        pair_force = np.asarray(new["replica_fingertip_object_pair_force_world"])
        valid = np.asarray(new["replica_fingertip_object_pair_force_valid"], dtype=bool)
        names = tuple(str(value) for value in new["fingertip_link_names"].tolist())
        indices = tuple(int(value) for value in new["fingertip_force_sensor_indices"].tolist())
        if pair_force.shape != (321, 20, 5, 3) or valid.shape != (321, 20):
            raise ValueError("V1_PAIRFORCE_REEXPORT_SHAPE_INVALID")
        if pair_force.dtype != np.float32 or names != FINGER_LINKS or indices != FINGER_INDICES:
            raise ValueError("V1_PAIRFORCE_REEXPORT_MAPPING_OR_DTYPE_INVALID")
        if (
            str(new["pair_force_frame"].item()) != "world"
            or str(new["pair_force_units"].item()) != "N"
        ):
            raise ValueError("V1_PAIRFORCE_REEXPORT_SEMANTICS_INVALID")
        if bool(valid[0].any()) or not np.isfinite(pair_force[valid]).all():
            raise ValueError("V1_PAIRFORCE_REEXPORT_VALIDITY_INVALID")
        old_actions = np.asarray(old["replica_action"], dtype=np.float64)
        new_actions = np.asarray(new["replica_action"], dtype=np.float64)
        old_indices = np.asarray(old["reference_index"], dtype=np.int64)
        new_indices = np.asarray(new["reference_index"], dtype=np.int64)
        if old_actions.shape != new_actions.shape or old_indices.shape != new_indices.shape:
            raise ValueError("V1_PAIRFORCE_REEXPORT_CONTRACT_MISMATCH:trace_shape")
        if not np.array_equal(old_indices, new_indices):
            raise ValueError("V1_PAIRFORCE_REEXPORT_CONTRACT_MISMATCH:reference_progression")
        pair_norm = np.linalg.vector_norm(pair_force, axis=-1)
        old_contact = np.asarray(old["replica_contact_pair_presence"], dtype=bool)
        new_contact = np.asarray(new["replica_contact_pair_presence"], dtype=bool)
        trajectory_diverged = not np.array_equal(
            np.asarray(old["replica_object_pose"]), np.asarray(new["replica_object_pose"])
        )
        action_delta = new_actions - old_actions
        qualification = {
            "schema_version": "Stage16DV1PairForceReexportQualificationV1",
            "status": "V1_PAIRFORCE_REEXPORT_VALIDATED",
            "artifact_label": "V1_PAIRFORCE_REEXPORT_DIAGNOSTIC",
            "clip": args.clip,
            "reward_version": "TopoRetargetReferenceTrackingReward26DV1",
            "policy_mode": "deterministic_mean_action",
            "rsi": [],
            "frame_zero_replicas": 20,
            "trace": str(trace_path),
            "trace_sha256": _sha256(trace_path),
            "old_r7_trace": str(old_trace_path.resolve()),
            "old_r7_trace_sha256": _sha256(old_trace_path),
            "checkpoint_sha256": source["checkpoint_sha256"],
            "formal_seed_set": source["formal_seed_set"],
            "formal_seed_count": len(source["formal_seeds"]),
            "reference_progression_identical": True,
            "episode_length": {
                "old": int(old_actions.shape[0]),
                "reexport": int(new_actions.shape[0]),
            },
            "action_statistics": {
                "old": _summary(old_actions),
                "reexport": _summary(new_actions),
                "delta": _summary(action_delta),
            },
            "contact_rate": {
                "old": float(old_contact.any(axis=-1).mean()),
                "reexport": float(new_contact.any(axis=-1).mean()),
            },
            "pair_force": {
                "shape": list(pair_force.shape),
                "dtype": str(pair_force.dtype),
                "valid_sample_count": int(valid.sum()),
                "initial_valid_sample_count": int(valid[0].sum()),
                "fingertip_link_names": list(names),
                "fingertip_force_sensor_indices": list(indices),
                "force_frame": str(new["pair_force_frame"].item()),
                "force_units": str(new["pair_force_units"].item()),
                "force_semantics": str(new["pair_force_semantics"].item()),
                "p95_magnitude_n": float(np.quantile(pair_norm[valid], 0.95)),
                "max_magnitude_n": float(pair_norm[valid].max()),
            },
            "physical_difference": "EXPECTED_PHYSX_CONTACT_DIVERGENCE"
            if trajectory_diverged
            else "NONE",
            "old_r7_status": old_evaluation["status"],
            "reexport_status": evaluation["status"],
        }
    manifest = {
        "schema_version": "Stage16DV1PairForceReexportManifestV1",
        "artifact_label": "V1_PAIRFORCE_REEXPORT_DIAGNOSTIC",
        "qualification": qualification,
        "trace_fields": [
            "replica_fingertip_object_pair_force_world",
            "replica_fingertip_object_pair_force_valid",
            "fingertip_link_names",
            "fingertip_force_sensor_indices",
            "pair_force_frame",
            "pair_force_units",
            "pair_force_semantics",
        ],
    }
    _write_json(output / "qualification.json", qualification)
    _write_json(output / "pairforce_manifest.json", manifest)
    print(json.dumps({"status": qualification["status"], "clip": args.clip}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
