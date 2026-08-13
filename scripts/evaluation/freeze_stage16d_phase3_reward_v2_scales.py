#!/usr/bin/env python3
"""Freeze Reward V2 dynamics scales from both qualified V2 references."""

# ruff: noqa: E402

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

from toporetarget.rl.reference_tracking.ppo26d_reward import (
    TopoRetargetReferenceTrackingReward26DV2,
)

CLIPS = ("hocap_170105", "hocap_170650")
DEFAULT_ROOT = REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        key: float(np.quantile(values, percentile))
        for key, percentile in (("p50", 0.5), ("p95", 0.95), ("max", 1.0))
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    qualification = json.loads((root / "reference_kinematics_qualification.json").read_text())
    if qualification.get("status") != "STAGE16D_REFERENCE_KINEMATICS_V2_VALIDATED":
        raise RuntimeError("PHASE3_REWARD_SCALE_FREEZE_REQUIRES_V2_QUALIFICATION")
    linear: list[np.ndarray] = []
    angular: list[np.ndarray] = []
    inputs: dict[str, Any] = {}
    for clip in CLIPS:
        path = root / "references" / f"{clip}.reference_kinematics_v2.npz"
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("reference_kinematics_version") != 2:
                raise ValueError("PHASE3_REWARD_SCALE_FREEZE_REQUIRES_V2_REFERENCE")
            twist = np.asarray(archive["object_twist_world_ref"], dtype=np.float64)
        linear.append(np.linalg.norm(twist[:, :3], axis=-1))
        angular.append(np.linalg.norm(twist[:, 3:], axis=-1))
        inputs[clip] = {"path": str(path), "sha256": _hash(path)}
    combined_linear = np.concatenate(linear)
    combined_angular = np.concatenate(angular)
    v_scale = 0.01
    omega_scale = 0.25
    # Conservative rounded engineering fallbacks, selected once from both clips:
    # max(V2 combined p95, existing terminal scale contribution), then round upward.
    sigma_v = 0.075
    sigma_omega = 0.125
    profile = TopoRetargetReferenceTrackingReward26DV2(
        object_velocity_sigma_mps=sigma_v,
        object_angular_velocity_sigma_radps=sigma_omega,
    )
    payload = {
        "schema_version": "Stage16DPhase3RewardV2ScaleFreezeV1",
        "status": "PHASE3_REWARD_V2_SCALES_FROZEN",
        "reference_kinematics_version": 2,
        "inputs": inputs,
        "combined_reference_dynamics": {
            "sample_count": int(combined_linear.size),
            "linear_speed_mps": _quantiles(combined_linear),
            "angular_speed_radps": _quantiles(combined_angular),
        },
        "existing_terminal_stability_scales": {
            "free_object_linear_speed_mps": v_scale,
            "free_object_angular_speed_radps": omega_scale,
        },
        "frozen_reward_v2": profile.as_dict(),
        "selection_rule": {
            "source_priority": (
                "engineering fallback after no paper or author-sourced V2 twist scale was present"
            ),
            "sigma_v_mps": "rounded upward from max(combined V2 p95, 5 * terminal linear scale)",
            "sigma_omega_radps": (
                "rounded upward from max(combined V2 p95, 0.5 * terminal angular scale)"
            ),
            "combined_maximum_contribution": 1.0,
            "maximum_allowed_contribution": 2.0,
        },
        "forbidden_reward_terms": [
            "contact_reward",
            "terminal_reward",
            "penetration_reward",
            "guidance_reward",
            "gravity_reward",
        ],
    }
    destination = root / "phase3" / "reward_v2_scale_freeze.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    contract_destination = root / "phase3" / "reward_v2_contract.json"
    contract_destination.write_text(
        json.dumps(
            {
                "schema_version": "Stage16DPhase3RewardV2ContractV1",
                "status": "PHASE3_REWARD_V2_CONTRACT_FROZEN",
                "identifier": profile.identifier,
                "reference_kinematics_version": 2,
                "signed_world_twist_objective": {
                    "linear": "exp(-(norm(v_actual_world - v_ref_world_v2) / sigma_v)^2)",
                    "angular": (
                        "exp(-(norm(omega_actual_world - omega_ref_world_v2) / sigma_omega)^2)"
                    ),
                    "speed_magnitude_only_penalty": False,
                },
                "profile": profile.as_dict(),
                "forbidden_reward_terms": payload["forbidden_reward_terms"],
                "scale_freeze": str(destination.resolve()),
                "scale_freeze_sha256": _hash(destination),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(destination),
                "contract": str(contract_destination),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
