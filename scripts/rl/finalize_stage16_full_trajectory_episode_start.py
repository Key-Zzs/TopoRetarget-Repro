#!/usr/bin/env python3
"""Materialize immutable full-trajectory episode-start receipts.

The script is intentionally Isaac-free.  A separate PhysX smoke consumes the
receipts; this avoids misrepresenting offline selection as a simulator result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.full_trajectory_episode_start import (
    CLIPS,
    select_full_trajectory_episode_start,
)

DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/stage16_p3_full_trajectory_restart"
VALIDITY_ROOT = REPO_ROOT / ".local/reports/stage16_p3b6_scene_rsi_requalification"
SUPPORT_ROOT = REPO_ROOT / ".local/reports/stage16_support_reconstruction/inference"
REFERENCE_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/references"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    frozen: dict[str, Any] = {
        "schema_version": "Stage16FullTrajectoryFrozenInputsV1",
        "reference_modified": False,
        "reference_geometry_role": "DIAGNOSTIC",
        "mid_trajectory_rsi": "disabled",
        "inputs": {},
    }
    summaries: list[dict[str, object]] = []
    for clip in CLIPS:
        validity_path = VALIDITY_ROOT / clip / "physical_reference_validity_mask.npz"
        stable_path = SUPPORT_ROOT / clip / "stable_interval.json"
        table_path = SUPPORT_ROOT / clip / "table_proxy.json"
        reference_path = REFERENCE_ROOT / f"{clip}.reference_kinematics_v2.npz"
        with np.load(validity_path, allow_pickle=False) as archive:
            rows = {name: np.asarray(archive[name]) for name in archive.files}
        stable = json.loads(stable_path.read_text(encoding="utf-8"))["interval"]
        start = int(stable["start_frame"])
        stop = int(stable["end_frame_exclusive"])
        receipt = select_full_trajectory_episode_start(
            clip=clip,
            validity_rows=rows,
            stable_indices=tuple(range(start, stop)),
            reference_hash=_sha256(reference_path),
            support_contract_hash=_sha256(table_path),
        ).as_dict()
        receipt.update(
            {
                "stable_interval": {"start": start, "end_exclusive": stop},
                "table_proxy": str(table_path.resolve()),
                "table_proxy_sha256": _sha256(table_path),
                "selection_reason": (
                    "frame0_preferred"
                    if receipt["start_index"] == 0
                    else "earliest_individually_valid_stable_pre_contact"
                ),
                "hard_reset_geometry": "PASS_FROM_FROZEN_FORMAL_THRESHOLDS",
                "one_g_support_sanity": "PENDING_PHYSX",
                "reference_annotation_twist_replaced_at_reset": True,
                "table_resting_reset_semantics": {
                    "identifier": "TABLE_RESTING_RESET_SEMANTICS_V1",
                    "applies_only_to": "verified_TABLE_SUPPORTED_PRE_CONTACT_episode_start",
                    "object_pose": "immutable_reference_pose",
                    "linear_velocity": "zero_resting_velocity",
                    "angular_velocity": "zero_resting_velocity",
                },
            }
        )
        _write(output / "episode_start" / f"{clip}.json", receipt)
        frozen["inputs"][clip] = {
            "reference": {
                "path": str(reference_path.resolve()),
                "sha256": _sha256(reference_path),
            },
            "validity_mask": {
                "path": str(validity_path.resolve()),
                "sha256": _sha256(validity_path),
            },
            "stable_interval": {
                "path": str(stable_path.resolve()),
                "sha256": _sha256(stable_path),
            },
            "table_proxy": {
                "path": str(table_path.resolve()),
                "sha256": _sha256(table_path),
            },
        }
        summaries.append(
            {
                "clip": clip,
                "start_index": receipt["start_index"],
                "phase": receipt["semantic_class"],
                "support_state": receipt["support_state"],
                "reference_hash": receipt["reference_hash"],
            }
        )
    _write(output / "frozen_inputs.json", frozen)
    _write(
        output / "episode_start" / "summary.json",
        {"schema_version": "Stage16FullTrajectoryEpisodeStartSummaryV1", "starts": summaries},
    )
    print(
        json.dumps(
            {"status": "FULL_TRAJECTORY_EPISODE_START_MATERIALIZED", "starts": summaries},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
