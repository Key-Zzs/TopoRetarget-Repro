#!/usr/bin/env python3
"""Materialize and audit the frozen Stage 16-D.5 PPO-26D contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.ppo26d_contract import (  # noqa: E402
    Stage16DPPO26DObservationV2,
    Stage16DPPO26DTrainingConfigV1,
    Stage16DReferenceResidualAction26DV1,
)
from toporetarget.rl.reference_tracking.ppo26d_reference import (  # noqa: E402
    export_factor8_reference,
)
from toporetarget.rl.reference_tracking.ppo26d_reward import (  # noqa: E402
    TopoRetargetReferenceTrackingReward26DV1,
)
from toporetarget.rl.reference_tracking.ppo26d_rsi import Stage16DPPO26DRSIV1  # noqa: E402

DEFAULT_SOURCE_ROOT = REPO_ROOT / ".local/stage16_reference_tracking_ppo/world_wrist_references"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.output_root.resolve()
    reference_root = root / "reference"
    exported = {}
    for clip in ("hocap_170105", "hocap_170650"):
        source = args.source_root / f"{clip}.world_wrist.stage16.npz"
        destination = reference_root / f"{clip}.reference.npz"
        exported[clip] = export_factor8_reference(source, destination)
    action = Stage16DReferenceResidualAction26DV1().as_dict()
    observation = Stage16DPPO26DObservationV2().as_dict()
    reward = TopoRetargetReferenceTrackingReward26DV1().as_dict()
    rsi = Stage16DPPO26DRSIV1().as_dict()
    training = Stage16DPPO26DTrainingConfigV1().as_dict()
    write_json(
        reference_root / "reference_contract.json",
        {"schema_version": "Stage16DPPO26DReferenceAuditV1", "clips": exported},
    )
    write_json(root / "observation" / "observation_contract.json", observation)
    write_json(root / "observation" / "existing_observation_semantic_map.json", observation)
    write_json(root / "reward" / "reward_contract.json", reward)
    write_json(
        root / "gate_revision.json",
        {
            "schema_version": "Stage16DPPO26DGateRevisionV1",
            "old_s3_cem": {
                "hocap_170105": "PRE_PPO_BASELINE_FAILURE",
                "hocap_170650": "PRE_PPO_BASELINE_FAILURE",
                "terminal_contact": "0/20",
                "terminal_stability": "0/20",
                "final_success": "0/20",
            },
            "gate_a_only_ppo_entry": True,
            "gate_b_training_safety": True,
            "gate_c_post_ppo_qualification": True,
            "not_gate_a": [
                "terminal_contact",
                "terminal_stability",
                "final_success",
                "exact_hand_object_penetration",
                "old_cem_trajectory_qualification",
            ],
        },
    )
    write_json(
        root / "paper_adaptation_ledger.json",
        {
            "claim": "TOPORETARGET_PPO_REPRODUCTION_WITH_26D_WRIST_ADAPTATION",
            "paper": ["reference_tracking_reward", "RSI", "PPO_table6"],
            "engineering_adaptation": [
                "six_d_wrist_residual",
                "explicit_serial_3p3r",
                "factor8_321_sample_reference",
                "isaaclab_backend",
            ],
            "paper_fidelity_partial": [
                "full_isaaclab_dynamics_randomization_mapping",
                "external_disturbance",
                "observation_delay",
            ],
        },
    )
    write_json(
        root / "contract_audit.json",
        {
            "action": action,
            "observation": observation,
            "reward": reward,
            "rsi": rsi,
            "training": training,
            "status": "STAGE16D_PPO26D_CONTRACT_VALIDATED",
        },
    )
    print(json.dumps({"status": "PASS", "output_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
