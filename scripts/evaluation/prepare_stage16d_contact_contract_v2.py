#!/usr/bin/env python3
"""Freeze R2 inputs and materialize diagnostic ReferenceContactContractV2."""

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

from toporetarget.evaluation.reference_contact_contract import (  # noqa: E402
    FINGER_ORDER,
    ReferenceContactContractV2,
    evaluate_reference_contact,
    persistent_windows,
)

CLIPS = ("hocap_170105", "hocap_170650")
V3_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"
V3_CONTACT_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_contact"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"
OLD_AUDIT_ROOT = REPO_ROOT / ".local/reports/stage16d_per_finger_contact_audit"
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16d_contact_contract_v2_audit"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"CONTACT_AUDIT_R2_JSON_OBJECT_REQUIRED:{path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"CONTACT_AUDIT_R2_INPUT_MISSING:{path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _old_summary() -> dict[str, object]:
    path = OLD_AUDIT_ROOT / "final_summary.json"
    return _read_json(path) if path.is_file() else {"status": "UNAVAILABLE", "path": str(path)}


def materialize(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    contract = ReferenceContactContractV2()
    checkpoints: dict[str, object] = {}
    seed_provenance: dict[str, object] = {}
    reference_provenance: dict[str, object] = {}
    physics_provenance: dict[str, object] = {}
    old_mask_contract: dict[str, object] = {
        "identifier": "V3_PRIMARY_EXPECTED_CONTACT_MASK",
        "formula": "distance_m < 0.03",
        "historical_training_contract": True,
        "mutated_by_r2": False,
        "clips": {},
    }
    all_inputs: dict[str, object] = {
        "schema_version": "Stage16DContactAuditR2FrozenInputsV1",
        "clips": {},
    }
    for clip in CLIPS:
        short = clip.removeprefix("hocap_")
        selection_path = V3_ROOT / clip / "dev/checkpoint_selection.json"
        selection = _read_json(selection_path)
        selected = selection.get("selected")
        if not isinstance(selected, dict):
            raise ValueError("CONTACT_AUDIT_R2_CHECKPOINT_SELECTION_INVALID")
        checkpoint = Path(str(selected["checkpoint"]))
        checkpoint_hash = _sha256(checkpoint)
        if checkpoint_hash != selected.get("checkpoint_sha256"):
            raise RuntimeError("CONTACT_AUDIT_R2_INPUT_DRIFT:checkpoint")
        qualification_path = V3_ROOT / clip / "formal/v3_formal_selected_2129920_qualification.json"
        qualification = _read_json(qualification_path)
        if qualification.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("CONTACT_AUDIT_R2_INPUT_DRIFT:formal_checkpoint")
        seed = qualification.get("seed_set")
        if not isinstance(seed, dict) or len(seed.get("frame_zero", [])) != 20:
            raise RuntimeError("CONTACT_AUDIT_R2_INPUT_DRIFT:formal_seed_set")
        mask_path = V3_CONTACT_ROOT / f"reference_contact_mask_{short}.npz"
        ref_path = REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz"
        with np.load(mask_path, allow_pickle=False) as archive:
            historical_mask = np.asarray(archive["reference_expected_contact_mask"], dtype=bool)
            distances = np.asarray(
                archive["reference_fingertip_to_object_distance_m"], dtype=np.float64
            )
            order = tuple(str(value) for value in archive["finger_order"].tolist())
        if (
            order != FINGER_ORDER
            or historical_mask.shape != (321, 5)
            or distances.shape != (321, 5)
        ):
            raise ValueError("CONTACT_AUDIT_R2_HISTORICAL_MASK_INVALID")
        result = evaluate_reference_contact(distances, contract=contract)
        if not np.array_equal(historical_mask, result["historical_v3_primary_mask"]):
            raise RuntimeError("CONTACT_AUDIT_R2_INPUT_DRIFT:historical_v3_mask")
        windows = persistent_windows(
            result["strong_contact_expected"],
            evidence_source=result["reference_evidence_source"],
            distances_m=result["reference_distance_m"],
            contract=contract,
        )
        npz_path = output / f"reference_contact_contract_v2_{short}.npz"
        np.savez_compressed(
            npz_path,
            **result,
            finger_order=np.asarray(FINGER_ORDER),
            persistent_window_control_steps=np.asarray(contract.persistent_window_control_steps),
        )
        evidence = {
            "schema_version": "Stage16DReferenceContactEvidenceV2",
            "clip": clip,
            "status": "REFERENCE_CONTACT_EVIDENCE_AVAILABLE_WITH_LIMITATIONS",
            "source_evidence_audit": {
                "SOURCE_EXPLICIT": "UNAVAILABLE: no frozen HOCap per-finger annotation found",
                "SOURCE_DERIVED": (
                    "UNAVAILABLE: no frozen source hand-object distance/contact field mapped "
                    "to the five Wuji tips"
                ),
                "TOPOLOGY_DERIVED": (
                    "UNAVAILABLE: existing Stage16 topology is runtime V3 telemetry and is "
                    "not source evidence"
                ),
                "GEOMETRIC_PROXIMITY": (
                    "AVAILABLE: frozen V3 reference fingertip-to-object distance"
                ),
            },
            "evidence_counts": {
                key: int((result["reference_evidence_class"] == key).sum())
                for key in np.unique(result["reference_evidence_class"])
            },
            "persistent_windows": windows,
            "source_or_topology_supported_fraction": 0.0,
            "geometric_strong_contact_candidate_fraction": float(
                result["strong_contact_expected"].mean()
            ),
            "reference_contact_evidence_conflict_count": int(
                result["reference_contact_evidence_conflict"].sum()
            ),
        }
        _write_json(output / f"contact_evidence_{short}.json", evidence)
        checkpoints[clip] = {
            "checkpoint_selection": _artifact(selection_path),
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_hash},
            "formal_qualification": _artifact(qualification_path),
        }
        seed_provenance[clip] = {
            "seed_manifest": _artifact(Path(str(seed["manifest"]))),
            "identifier": seed["identifier"],
            "frame_zero_seeds": seed["frame_zero"],
        }
        reference_provenance[clip] = {
            "reference": _artifact(ref_path),
            "historical_v3_mask": _artifact(mask_path),
            "reference_distance_field": "reference_fingertip_to_object_distance_m",
        }
        physics_provenance[clip] = {
            "formal_qualification": _artifact(qualification_path),
            "physics_contract_sha256": qualification.get("physics_contract_sha256"),
            "frozen_policy_contract": qualification.get("physics_contract", {}).get("ppo26d"),
        }
        old_mask_contract["clips"][clip] = {
            **_artifact(mask_path),
            "frame_count": 321,
            "finger_order": list(FINGER_ORDER),
        }
        all_inputs["clips"][clip] = {
            "checkpoint": checkpoints[clip],
            "seeds": seed_provenance[clip],
            "reference": reference_provenance[clip],
            "physics": {"physics_contract_sha256": qualification.get("physics_contract_sha256")},
        }
    _write_json(output / "frozen_inputs.json", all_inputs)
    _write_json(output / "checkpoint_provenance.json", checkpoints)
    _write_json(output / "formal_seed_provenance.json", seed_provenance)
    _write_json(output / "reference_provenance.json", reference_provenance)
    _write_json(output / "physics_provenance.json", physics_provenance)
    _write_json(output / "old_v3_mask_contract.json", old_mask_contract)
    _write_json(output / "old_per_finger_audit_summary.json", _old_summary())
    receipt = {
        "schema_version": contract.schema_version,
        "status": "REFERENCE_CONTACT_CONTRACT_V2_MATERIALIZED",
        "contract": contract.as_dict(),
        "v3_historical_mask_preserved": True,
        "training_or_reward_modified": False,
        "source_evidence_status": "SOURCE_PER_FINGER_EVIDENCE_UNAVAILABLE",
    }
    _write_json(output / "reference_contact_contract_v2.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
