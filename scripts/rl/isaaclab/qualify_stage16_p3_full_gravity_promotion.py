#!/usr/bin/env python3
"""Write a fail-closed G3 receipt when C2 has no safe global policy mode.

G3's policy arm is defined only for the selected global C2 mode.  This tool
does not substitute a rejected V3/V4 checkpoint, nor reinterpret the old P1
zero-residual diagnostic as a new promotion pass.  It writes the complete
safe-state/replica roster so the exact blocked scope remains auditable.
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.full_gravity_promotion import (  # noqa: E402
    G3_BLOCKED,
    expected_g3_state_replica_pairs,
    validate_g3_contract,
)
from toporetarget.rl.rsi.contact_ready_v2 import load_safe_bank  # noqa: E402

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16_p3_p4_full_gravity"
DEFAULT_SAFE_BANK_ROOT = REPO_ROOT / ".local/reports/stage16_physical_p0_p2/p1"
DEFAULT_CURRICULUM = REPO_ROOT / "configs/rl/stage16/stage16_gravity_friction_curriculum_v1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--safe-bank-root", type=Path, default=DEFAULT_SAFE_BANK_ROOT)
    parser.add_argument("--curriculum-contract", type=Path, default=DEFAULT_CURRICULUM)
    args = parser.parse_args()
    root = args.output_root.resolve()
    selection_path = root / "global_physical_contact_mode_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "GLOBAL_PHYSICAL_CONTACT_MODE_SELECTION_BLOCKED":
        raise ValueError("G3_BLOCKED_RECEIPT_REQUIRES_BLOCKED_GLOBAL_C2_SELECTION")
    curriculum_document = yaml.safe_load(args.curriculum_contract.read_text(encoding="utf-8"))
    if not isinstance(curriculum_document, dict):
        raise ValueError("G3_CURRICULUM_DOCUMENT_INVALID")
    physics = validate_g3_contract(curriculum_document["g3_promotion"])
    output = root / "g3"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"G3_OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    banks: dict[str, dict[str, object]] = {}
    for clip in CLIPS:
        bank_path = args.safe_bank_root.resolve() / f"safe_bank_{clip.removeprefix('hocap_')}.npz"
        bank = load_safe_bank(bank_path)
        indices = tuple(int(value) for value in np.asarray(bank["runtime_index"], dtype=np.int64))
        labels = {
            int(index): str(label)
            for index, label in zip(bank["runtime_index"], bank["safe_bank"], strict=True)
        }
        for runtime_index, replica in sorted(expected_g3_state_replica_pairs(indices)):
            rows.append(
                {
                    "clip": clip,
                    "runtime_index": runtime_index,
                    "safe_bank": labels[runtime_index],
                    "replica": replica,
                    "status": "NOT_RUN_NO_SAFETY_QUALIFIED_GLOBAL_C2_MODE",
                    "gravity_scale": 1.0,
                    "friction_scale": 1.0,
                    "control_steps": 20,
                }
            )
        banks[clip] = {
            "safe_bank_path": str(bank_path),
            "safe_bank_sha256": _sha256(bank_path),
            "candidate_safe_state_count": len(indices),
            "replicas_per_safe_state": 4,
            "expected_row_count": len(indices) * 4,
            "final_safe_bank_status": "NOT_PROMOTED",
        }
    with (output / "per_state.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for clip, bank in banks.items():
        _write_json(output / "final_safe_banks" / f"{clip}.json", bank)
    qualification: dict[str, Any] = {
        "schema_version": "Stage16P3FullGravityPromotionQualificationV1",
        "status": G3_BLOCKED,
        "execution": "NOT_RUN_UPSTREAM_GLOBAL_C2_SELECTION_BLOCKED",
        "physics_contract": physics,
        "selection_input": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "status": selection["status"],
            "selected_mode": selection["selected_mode"],
        },
        "safe_state_roster": banks,
        "zero_residual_reference_following": {
            "status": "NOT_RUN_UPSTREAM_GLOBAL_C2_SELECTION_BLOCKED",
            "reason": "A rejected C2 mode may not be substituted for the required selected mode.",
        },
        "selected_c2_policy_diagnostic": {
            "status": "NOT_RUN_NO_SELECTED_GLOBAL_C2_POLICY",
            "reason": "Both C2 modes failed the mandatory absolute-geometry safety gate.",
        },
        "promotion_reason": (
            "No global C2 contact mode passed absolute geometry on both clips. G3 and all C3/C4 "
            "training are therefore blocked rather than using clip-specific or rejected policies."
        ),
        "training_authorized": False,
    }
    _write_json(output / "qualification.json", qualification)
    print(json.dumps({"status": G3_BLOCKED, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
