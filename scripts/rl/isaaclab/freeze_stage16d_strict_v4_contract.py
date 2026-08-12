#!/usr/bin/env python3
"""Freeze SourcePerFingerContactEvidenceV1 inputs for Strict Per-Finger V4.

This program is deliberately Isaac-free.  It converts only the already-frozen
source-audit labels into the runtime mask, calibrates a shared single-tip scale
from V1 Formal20 exact pair-force telemetry, and records every lineage input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.reference_tracking.strict_per_finger_contact import (  # noqa: E402
    SOURCE_CONTACT_REQUIRED_CLASSES,
    StrictPerFingerContactRewardV4,
    strict_source_contact_mask,
)

CLIPS = ("hocap_170105", "hocap_170650")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SOURCE_ROOT = REPO_ROOT / ".local/reports/stage16d_source_contact_semantics_final_audit"
V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_strict_per_finger_v4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"STRICT_V4_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"STRICT_V4_REQUIRED_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("STRICT_V4_CALIBRATION_VALUES_INVALID")
    if not array.size:
        return {
            "n": 0,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    return {
        "n": int(array.size),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _source_mask(source_root: Path, clip: str, output: Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = source_root / clip / "source_contact_evidence_runtime.npz"
    with np.load(path, allow_pickle=False) as archive:
        required = {"class_label", "finger_order", "control_index"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"STRICT_V4_SOURCE_RUNTIME_FIELDS_MISSING:{clip}:{missing}")
        class_label = np.asarray(archive["class_label"])
        finger_order = tuple(str(value) for value in archive["finger_order"].tolist())
        control_index = np.asarray(archive["control_index"], dtype=np.int64)
    if finger_order != FINGERS or not np.array_equal(control_index, np.arange(321)):
        raise ValueError(f"SOURCE_CONTACT_RUNTIME_MAPPING_INVALID:{clip}")
    mask = strict_source_contact_mask(class_label)
    output_path = output / f"strict_source_contact_mask_{clip}.npz"
    np.savez_compressed(
        output_path,
        strict_source_contact_mask=mask,
        source_contact_class=class_label,
        finger_names=np.asarray(FINGERS),
        control_index=control_index,
    )
    return mask, {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "source_runtime": _receipt(path),
        "shape": list(mask.shape),
        "finger_order": list(FINGERS),
        "required_classes": list(SOURCE_CONTACT_REQUIRED_CLASSES),
        "counts_by_finger": {
            finger: int(mask[:, index].sum()) for index, finger in enumerate(FINGERS)
        },
        "required_frame_count": int(mask.any(axis=1).sum()),
    }


def _calibration_values(
    *, trace: Path, mask: np.ndarray, clip: str, floor_n: float
) -> tuple[dict[str, Any], list[np.ndarray]]:
    with np.load(trace, allow_pickle=False) as archive:
        required = {
            "replica_fingertip_object_pair_force_world",
            "replica_fingertip_object_pair_force_valid",
            "replica_contact_pair_presence",
            "fingertip_force_sensor_indices",
            "fingertip_link_names",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"PAIR_FORCE_MAPPING_INVALID:{clip}:{missing}")
        force = np.asarray(archive["replica_fingertip_object_pair_force_world"], dtype=np.float64)
        valid = np.asarray(archive["replica_fingertip_object_pair_force_valid"], dtype=bool)
        presence = np.asarray(archive["replica_contact_pair_presence"], dtype=bool)
        sensor_indices = tuple(
            int(value) for value in np.asarray(archive["fingertip_force_sensor_indices"]).tolist()
        )
        names = tuple(str(value) for value in archive["fingertip_link_names"].tolist())
    if force.shape != (321, 20, 5, 3) or valid.shape != (321, 20):
        raise ValueError(f"PAIR_FORCE_MAPPING_INVALID:{clip}:shape")
    if (
        presence.shape != (321, 20, 21)
        or sensor_indices != (20, 4, 8, 16, 12)
        or names
        != (
            "r_thumb_distal",
            "r_index_finger_distal",
            "r_middle_finger_distal",
            "r_ring_finger_distal",
            "r_pinky_distal",
        )
    ):
        raise ValueError(f"PAIR_FORCE_MAPPING_INVALID:{clip}:semantics")
    presence = presence[:, :, sensor_indices]
    norm = np.linalg.vector_norm(force, axis=-1)
    if not np.isfinite(norm[valid]).all():
        raise ValueError(f"PAIR_FORCE_MAPPING_INVALID:{clip}:nonfinite")
    included = mask[:, None, :] & valid[:, :, None] & presence & (norm > floor_n)
    values = [
        np.asarray(norm[:, :, index][included[:, :, index]], dtype=np.float64) for index in range(5)
    ]
    coverage = {
        finger: {
            "source_required_samples": int((mask[:, index, None] & valid).sum()),
            "positive_contact_samples": int(values[index].size),
            **_stats(values[index]),
        }
        for index, finger in enumerate(FINGERS)
    }
    return {
        "trace": _receipt(trace),
        "clip": clip,
        "positive_calibration_sample_count": int(sum(value.size for value in values)),
        "coverage_by_finger": coverage,
    }, values


def _response(lambda_tip_n: float, floor_n: float) -> list[dict[str, float]]:
    forces = [
        0.0,
        floor_n,
        lambda_tip_n * 0.05,
        lambda_tip_n * 0.10,
        lambda_tip_n * 0.25,
        lambda_tip_n,
        lambda_tip_n * 2,
        lambda_tip_n * 4,
    ]
    return [
        {
            "force_n": force,
            "per_finger_reward": 0.0
            if force <= floor_n
            else float(np.exp(-lambda_tip_n / (force + 1.0e-5))),
        }
        for force in forces
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--v3-root", type=Path, default=V3_ROOT)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    v3_root = args.v3_root.resolve()
    reference_root = args.reference_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = StrictPerFingerContactRewardV4()

    decision = _json(source_root / "decision.json")
    source_contract = _json(source_root / "source_contact_contract.json")
    if (
        decision.get("primary_recommendation") != "STRICT_PER_FINGER_V4_RECOMMENDED"
        or decision.get("decision_confidence") != "HIGH"
        or source_contract.get("identifier") != contract.source_contact_semantics_identifier
    ):
        raise RuntimeError("STRICT_V4_INPUT_PROVENANCE_DRIFT")
    v3_contract = _json(v3_root / "reward_v3_contract.json")
    if v3_contract.get("status") != "CONTACT_REWARD_CONTRACT_FROZEN":
        raise RuntimeError("STRICT_V4_INPUT_PROVENANCE_DRIFT")
    checkpoint_manifest = _json(v3_root / "checkpoint_manifest.json")
    formal_seed_manifest = _receipt(v3_root / "formal_seed_manifest.json")
    source_masks: dict[str, np.ndarray] = {}
    source_mask_receipts: dict[str, Any] = {}
    calibration_by_clip: dict[str, Any] = {}
    pooled: list[np.ndarray] = []
    positive_families: set[str] = set()
    for clip in CLIPS:
        mask, mask_receipt = _source_mask(source_root, clip, output)
        source_masks[clip] = mask
        source_mask_receipts[clip] = mask_receipt
        trace = v3_root / "v1_pairforce" / clip / "trace.npz"
        report, values = _calibration_values(
            trace=trace, mask=mask, clip=clip, floor_n=contract.numerical_floor_n
        )
        calibration_by_clip[clip] = report
        pooled.extend(value for value in values if value.size)
        positive_families.update(
            finger for finger, value in zip(FINGERS, values, strict=True) if value.size
        )
    all_positive = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    if all_positive.size < 100 or len(positive_families) < 2:
        status = "STRICT_V4_CALIBRATION_COVERAGE_FAILURE"
        lambda_tip_n: float | None = None
    else:
        status = "STRICT_V4_CONTACT_CONTRACT_FROZEN"
        lambda_tip_n = float(np.median(all_positive))
    calibration = {
        "schema_version": "StrictPerFingerForceScaleCalibrationV1",
        "status": status,
        "source": "V1_Formal20_exact_named_tip_pair_force_only",
        "lambda_rule": "pooled_positive_source_required_tip_force_p50",
        "numerical_floor_n": contract.numerical_floor_n,
        "coverage_by_clip": calibration_by_clip,
        "pooled_positive_contact_statistics": _stats(all_positive),
        "positive_finger_families": sorted(positive_families),
        "lambda_tip_n": lambda_tip_n,
    }
    _write_json(output / "strict_v4_force_scale_calibration.json", calibration)
    if lambda_tip_n is None:
        if args.require_ready:
            return 2
        return 0
    response = _response(lambda_tip_n, contract.numerical_floor_n)
    if not np.isclose(response[5]["per_finger_reward"], np.exp(-1.0), atol=1.0e-5):
        raise AssertionError("STRICT_V4_RESPONSE_LAMBDA_INVARIANT_FAILED")
    with (output / "strict_v4_reward_response.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("force_n", "per_finger_reward"))
        writer.writeheader()
        writer.writerows(response)
    _write_json(
        output / "strict_v4_reward_response.json",
        {
            "schema_version": "StrictPerFingerContactRewardResponseV1",
            "lambda_tip_n": lambda_tip_n,
            "rows": response,
        },
    )
    checkpoint_receipts: dict[str, Any] = {}
    for clip in CLIPS:
        row = checkpoint_manifest.get("clips", {}).get(clip)
        if not isinstance(row, dict):
            raise RuntimeError("STRICT_V4_INPUT_PROVENANCE_DRIFT")
        checkpoint = Path(str(row["checkpoint"]))
        if _sha256(checkpoint) != row.get("checkpoint_sha256"):
            raise RuntimeError("V1_L0_INITIALIZATION_DRIFT")
        checkpoint_receipts[clip] = {
            "v1_l0_checkpoint": _receipt(checkpoint),
            "manifest_entry": row,
        }
    frozen_inputs = {
        "schema_version": "Stage16DStrictPerFingerV4FrozenInputsV1",
        "status": status,
        "source_contact_decision": _receipt(source_root / "decision.json"),
        "source_contact_contract": _receipt(source_root / "source_contact_contract.json"),
        "source_mano_object": _receipt(source_root / "source_provenance.json"),
        "source_runtime_masks": source_mask_receipts,
        "reference_kinematics": {
            clip: _receipt(reference_root / f"{clip}.reference_kinematics_v2.npz") for clip in CLIPS
        },
        "v1_formal_pair_force": {
            clip: _receipt(v3_root / "v1_pairforce" / clip / "trace.npz") for clip in CLIPS
        },
        "v3_contract": _receipt(v3_root / "reward_v3_contract.json"),
        "formal_seed_manifest": formal_seed_manifest,
        "checkpoint_provenance": checkpoint_receipts,
        "physics_action_observation_controller": _receipt(v3_root / "checkpoint_manifest.json"),
    }
    _write_json(output / "frozen_inputs.json", frozen_inputs)
    _write_json(output / "source_contact_provenance.json", source_mask_receipts)
    _write_json(output / "checkpoint_provenance.json", checkpoint_receipts)
    _write_json(output / "reference_provenance.json", frozen_inputs["reference_kinematics"])
    _write_json(
        output / "physics_provenance.json",
        {"checkpoint_manifest": frozen_inputs["physics_action_observation_controller"]},
    )
    _write_json(
        output / "v3_baseline_provenance.json", {"reward_v3_contract": frozen_inputs["v3_contract"]}
    )
    _write_json(
        output / "evaluation_contract_provenance.json",
        {"formal_seed_manifest": formal_seed_manifest},
    )
    v4_contract = {
        "schema_version": "Stage16DStrictPerFingerContactRewardV4ContractV1",
        "status": status,
        "frozen_parameters": {**contract.as_dict(), "lambda_tip_n": lambda_tip_n},
        "reward": "RewardV4 = RewardV2 + r_contact_v4",
        "v3_aggregate_contact_term": "absent",
        "source_mask_semantics": "confirmed_or_persistent_confirmed_only",
        "forbidden_source_states": [
            "SOURCE_CONTACT_PROBABLE",
            "SOURCE_CONTACT_TRANSITION",
            "SOURCE_PROXIMITY_ONLY",
            "SOURCE_NO_CONTACT",
            "ambiguous",
        ],
    }
    _write_json(output / "strict_v4_contract.json", v4_contract)
    _write_json(
        output / "strict_v4_source_mask_contract.json",
        {
            "schema_version": "StrictV4SourceContactMaskContractV1",
            "status": status,
            "clips": source_mask_receipts,
            "immutable_from_policy": True,
        },
    )
    print(json.dumps({"status": status, "lambda_tip_n": lambda_tip_n, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
