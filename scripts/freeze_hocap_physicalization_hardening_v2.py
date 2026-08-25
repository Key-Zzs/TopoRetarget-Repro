#!/usr/bin/env python3
"""Freeze the immutable HOCap hardening V2 authority before P5 starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/raw_to_physical_hardening_v2/contracts"
CONFIG = REPO_ROOT / "configs/contracts/hocap_physicalization_hardening_v2.json"

AUTHORITY_SOURCES = (
    "configs/contracts/hocap_physicalization_hardening_v2.json",
    "configs/contracts/source_controller_auto_v1.yaml",
    "configs/retarget/refinement_execution/wuji_continuous_sequential_fast_exact_v2.yaml",
    "src/toporetarget/retarget/input_quality.py",
    "src/toporetarget/retarget/refinement_checkpoint.py",
    "src/toporetarget/rl/source_controller.py",
    "src/toporetarget/rl/ppo_generalization.py",
    "src/toporetarget/rl/environments/isaaclab_backend/explicit_virtual_wrist.py",
    "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py",
    "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env_cfg.py",
    "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py",
    "src/toporetarget/evaluation/physical_functionality_full_cycle_v1.py",
)

LANE_RECEIPTS = {
    "P1": ".local/reports/raw_to_physical_hardening_v2/p1_retarget/final_decision.json",
    "P2": ".local/reports/raw_to_physical_hardening_v2/p2_source_controller/final_decision.json",
    "P3": ".local/reports/raw_to_physical_hardening_v2/p3_ppo_generalization/final_decision.json",
    "P4": ".local/reports/raw_to_physical_hardening_v2/p4_full_cycle/final_decision.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"HARDENING_V2_FREEZE_OUTPUT_EXISTS:{path}")
    path.write_text(value, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"HARDENING_V2_JSON_OBJECT_REQUIRED:{path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    config = _load_json(CONFIG)
    if config.get("schema_version") != "HOCapPhysicalizationHardeningProtocolV2":
        raise ValueError("HARDENING_V2_CONFIG_SCHEMA_INVALID")

    lanes: dict[str, object] = {}
    for lane, relative in LANE_RECEIPTS.items():
        path = REPO_ROOT / relative
        payload = _load_json(path)
        lanes[lane] = {
            "path": relative,
            "sha256": _sha256(path),
            "decision": payload.get("decision"),
            "status": payload.get("status"),
            "schema_version": payload.get("schema_version"),
        }

    sources: list[dict[str, str]] = []
    for relative in AUTHORITY_SOURCES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"HARDENING_V2_AUTHORITY_SOURCE_MISSING:{path}")
        sources.append({"path": relative, "sha256": _sha256(path)})

    contract = {
        **config,
        "lane_final_receipts": lanes,
        "authority_sources": sources,
        "freeze_semantics": {
            "p1_p4_final_receipts_exist": True,
            "p5_started": False,
            "mutable_after_first_p5_result": False,
        },
    }
    contract_hash = _stable_hash(contract)
    authorities = {
        "schema_version": "HOCapPhysicalizationCurrentAuthoritiesV2",
        "hardening_v2_contract_hash": contract_hash,
        "authorities": {
            "episode": "HOCapSingleHandObjectEpisodeV1",
            "retarget_input": "RetargetInputQualityV1",
            "source_controller": "SourceControllerMode.AUTO",
            "ppo_sampling": "UniformEventBalancedRSIV1",
            "object_scale": "DimensionlessObjectScaleV1",
            "support": "SupportResolutionV1",
            "reward": "Stage16GroupedMultiplicativeRewardV1",
            "rse": "Stage16ReferenceScopedExplorationV1",
            "angular": "AngularAuthorityV2",
            "pf": "PhysicalFunctionalityV2+PhysicalFunctionalityFullCycleV1",
        },
    }
    _write_new(
        output / "hardening_physicalization_v2.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    _write_new(output / "hardening_v2_contract_hash.txt", contract_hash + "\n")
    _write_new(
        output / "current_authorities.json",
        json.dumps(authorities, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"HARDENING_V2_CONTRACT_HASH": contract_hash, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
