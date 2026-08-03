#!/usr/bin/env python3
"""Materialize immutable-key 3P+3R q/qd/qdd references outside Isaac Sim."""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".local/experiments/stage16c3r3_joint_dynamics_c5/joint_reference",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT
        / ".local/reports/stage16c3r3_joint_dynamics_c5/explicit_joint_reference.json",
    )
    return parser.parse_args()


def main() -> int:
    from toporetarget.rl.environments.isaaclab_backend.explicit_wrist_reference import (
        ExplicitWristJointReferenceV2,
    )
    from toporetarget.rl.environments.isaaclab_backend.reference_bank import WorldWristReferenceBank

    args = parse_args()
    reference_root = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
    bank = WorldWristReferenceBank(
        {
            "hocap_170105": reference_root / "hocap_170105.world_wrist.stage16.npz",
            "hocap_170650": reference_root / "hocap_170650.world_wrist.stage16.npz",
        },
        device="cpu",
    )
    reference = ExplicitWristJointReferenceV2.from_reference_bank(bank)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = reference.validation()
    for clip_index, clip in enumerate(reference.clip_ids):
        path = args.output_dir / f"joint_reference_{clip.removeprefix('hocap_')}.npz"
        np.savez_compressed(
            path,
            q_w_ref=reference.q_wrist_ref[clip_index].numpy(),
            qd_w_ref=reference.qd_wrist_ref[clip_index].numpy(),
            qdd_w_ref=np.gradient(
                reference.qd_wrist_ref[clip_index].numpy(), reference.dt_s, axis=0
            ),
            q_f_ref=reference.q_finger_ref[clip_index].numpy(),
            qd_f_ref=reference.qd_finger_ref[clip_index].numpy(),
            qdd_f_ref=np.gradient(
                reference.qd_finger_ref[clip_index].numpy(), reference.dt_s, axis=0
            ),
        )
    report.update(
        {
            "status": "C3R3_EXPLICIT_JOINT_REFERENCE_VALIDATED",
            "source_reference_hashes": bank.hashes,
            "output_dir": str(args.output_dir),
            "generated_npz_tracked": False,
            "substep_contract": "cubic Hermite q/qd/analytic qdd at 120 Hz",
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
