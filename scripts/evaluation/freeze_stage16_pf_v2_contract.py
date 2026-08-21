#!/usr/bin/env python3
"""Freeze the audited Stage16 PF V2 contract before any new PPO update.

This intentionally read-only gate binds the evaluator implementation, its
outcome-independent constants, the accepted-positive-control re-evaluation,
and U10's no-optimizer Eval20 receipt.  The output is a local experiment
receipt; it never changes PF V1 or any frozen source trace/checkpoint.
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = REPO_ROOT / ".local/reports/stage16_pf_v2_causal_lift_and_symmetric_ppo"
OUTPUT = REPORT_ROOT / "pf_v2/contract_freeze.json"
CONTRACT = REPORT_ROOT / "pf_v2/pf_v2_contract.json"
AUDIT = REPORT_ROOT / "pf_v2/audit_classification.json"
U10_EVAL20 = REPORT_ROOT / "u10_eval20/summary.json"
U10_CHECKPOINT = (
    REPO_ROOT / ".local/runs/stage16_dexplore_reward_rse/training/U10/checkpoint/checkpoint.pt"
)
U10_CHECKPOINT_SHA256 = "58f18d934679de4a9759a91cb6b0c296e2357eeff88d23dc9915f85e72cceb95"
IMPLEMENTATION = REPO_ROOT / "src/toporetarget/evaluation/stage16_pf_v2_causal_lift.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"PF_V2_CONTRACT_FREEZE_MISSING:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PF_V2_CONTRACT_FREEZE_OBJECT_REQUIRED:{path}")
    return value


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"PF_V2_CONTRACT_FREEZE_EXISTS:{OUTPUT}")
    contract = _read(CONTRACT)
    audit = _read(AUDIT)
    eval20 = _read(U10_EVAL20)
    expected_contract = {
        "schema_version": "Stage16PhysicalFunctionalityV2",
        "lift_threshold_m": 0.05,
        "persistence_control_steps": 3,
        "multifinger_minimum": 2,
        "control_period_s": 0.05,
        "reference_lift_hard_gate": False,
        "exact_wrench_transfer_claimed": False,
        "exact_surface_slip_claimed": False,
        "outcome_tuned": False,
        "support_validity_rule": (
            "table_contact_sensor_validity_is_independent_of_hand_object_pair_force_validity; "
            "a recorded reset support sample is retained"
        ),
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise RuntimeError("PF_V2_CONTRACT_FREEZE_CONTRACT_DRIFT")
    if (
        audit.get("PF_V2_AUDIT")
        not in {
            "PF_V1_PRELIFT_GATE_OVERCONSTRAINED_CONFIRMED",
            "PF_V1_PRELIFT_GATE_PARTIALLY_OVERCONSTRAINED",
        }
        or audit.get("ppo_authorized") is not True
    ):
        raise RuntimeError("PF_V2_CONTRACT_FREEZE_AUDIT_NOT_AUTHORIZED")
    counts = eval20.get("counts", {})
    if (
        eval20.get("optimizer_steps") != 0
        or counts.get("PF_V2") != 20
        or counts.get("physical_lift") != 20
        or counts.get("causal_lift") != 20
        or counts.get("support_transfer") != 20
        or counts.get("sustained_hand_object_coupling") != 20
        or not (REPORT_ROOT / "u10_eval20/traces").is_dir()
        or len(list((REPORT_ROOT / "u10_eval20/traces").glob("episode_*.npz"))) != 20
    ):
        raise RuntimeError("PF_V2_CONTRACT_FREEZE_U10_EVAL20_INVALID")
    if not U10_CHECKPOINT.is_file() or _sha256(U10_CHECKPOINT) != U10_CHECKPOINT_SHA256:
        raise RuntimeError("PF_V2_CONTRACT_FREEZE_U10_CHECKPOINT_HASH_INVALID")
    receipt = {
        "schema_version": "Stage16PhysicalFunctionalityV2ContractFreezeV1",
        "classification": "PF_V2_CONTRACT_FROZEN",
        "passed": True,
        "PF_V1_CHANGED": "NO",
        "REFERENCE_LIFT_HARD_GATE": "NO",
        "frozen_contract": {"path": str(CONTRACT.resolve()), "sha256": _sha256(CONTRACT)},
        "implementation": {
            "path": str(IMPLEMENTATION.resolve()),
            "sha256": _sha256(IMPLEMENTATION),
        },
        "audit": {"path": str(AUDIT.resolve()), "sha256": _sha256(AUDIT)},
        "u10_eval20": {
            "path": str(U10_EVAL20.resolve()),
            "sha256": _sha256(U10_EVAL20),
            "optimizer_steps": 0,
            "PF_V2": 20,
        },
        "u10_checkpoint": {
            "path": str(U10_CHECKPOINT.resolve()),
            "sha256": U10_CHECKPOINT_SHA256,
        },
        "training_authorized": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
