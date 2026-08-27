#!/usr/bin/env python3
"""Freeze H3PhysicalizationProtocolV1 on the clean integrated execution head."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/contracts/h3_physicalization_protocol_v1.json"
DEFAULT_H3A = (
    REPO_ROOT
    / ".local/reports/h3_unseen_object_generalization/h3a_source_controller/final_decision.json"
)
DEFAULT_H3B = (
    REPO_ROOT
    / ".local/reports/h3_unseen_object_generalization/h3b_retarget_throughput/final_decision.json"
)
DEFAULT_OUTPUT = REPO_ROOT / ".local/reports/h3_unseen_object_generalization/contracts"

COMMON_AUTHORITY_SOURCES = (
    "configs/contracts/h3_physicalization_protocol_v1.json",
    "configs/contracts/source_controller_auto_v2.yaml",
    "src/toporetarget/retarget/input_quality.py",
    "src/toporetarget/retarget/refinement_checkpoint.py",
    "src/toporetarget/rl/source_controller.py",
    "src/toporetarget/rl/ppo_generalization.py",
    "src/toporetarget/rl/independent_physical_refinement.py",
    "src/toporetarget/rl/environments/isaaclab_backend/explicit_virtual_wrist.py",
    "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py",
    "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env_cfg.py",
    "src/toporetarget/rl/environments/isaaclab_backend/world_wrist_direct_env.py",
    "src/toporetarget/evaluation/physical_functionality_full_cycle_v1.py",
    "scripts/rl/isaaclab/run_independent_source_policy.py",
    "scripts/physics/run_independent_physical_support.py",
    "scripts/evaluation/run_independent_frozen_physical_evaluation.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3a-decision", type=Path, default=DEFAULT_H3A)
    parser.add_argument("--h3b-decision", type=Path, default=DEFAULT_H3B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"H3_PROTOCOL_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3_PROTOCOL_OUTPUT_EXISTS:{path}")
    path.write_text(text, encoding="utf-8")


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"H3_PROTOCOL_GIT_COMMAND_FAILED:{arguments}:{result.stdout}")
    return result.stdout.strip()


def main() -> int:
    args = _parser().parse_args()
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("H3_PROTOCOL_TRACKED_WORKTREE_NOT_CLEAN")
    branch = _git("branch", "--show-current")
    if branch != "feature/dexplore-reward-rse":
        raise RuntimeError(f"H3_PROTOCOL_BRANCH_INVALID:{branch}")
    execution_head = _git("rev-parse", "HEAD")
    config = _load_json(CONFIG)
    if config.get("schema_version") != "H3PhysicalizationProtocolV1":
        raise ValueError("H3_PROTOCOL_CONFIG_SCHEMA_INVALID")
    h3a_path = args.h3a_decision.resolve()
    h3b_path = args.h3b_decision.resolve()
    h3a = _load_json(h3a_path)
    h3b = _load_json(h3b_path)
    if h3a.get("decision") not in {
        "H3A_SOURCE_ADMISSION_V2_VALIDATED",
        "H3A_SOURCE_ADMISSION_V2_PARTIAL",
        "H3A_TRUE_SOURCE_CONTROLLER_HARD_FAILURES_IDENTIFIED",
        "H3A_INCONCLUSIVE",
    }:
        raise ValueError("H3_PROTOCOL_H3A_DECISION_INVALID")
    selected_source = h3a.get("H3A_SELECTED_SOURCE_CONTROLLER_CONTRACT")
    if not isinstance(selected_source, str) or not selected_source:
        raise ValueError("H3_PROTOCOL_H3A_SELECTED_CONTRACT_MISSING")
    if h3b.get("decision") not in {
        "H3B_THROUGHPUT_HARDENING_VALIDATED",
        "H3B_IO_OVERHEAD_REDUCED",
        "H3B_NO_MEASURABLE_SPEEDUP",
        "H3B_REGRESSION_REVERTED",
        "H3B_INCONCLUSIVE",
    }:
        raise ValueError("H3_PROTOCOL_H3B_DECISION_INVALID")
    selected_execution = h3b.get("H3B_SELECTED_RETARGET_EXECUTION_CONTRACT")
    if not isinstance(selected_execution, str) or not selected_execution:
        raise ValueError("H3_PROTOCOL_H3B_SELECTED_CONTRACT_MISSING")
    execution_profile_path = (
        REPO_ROOT / "configs/retarget/refinement_execution" / f"{selected_execution}.yaml"
    )
    if not execution_profile_path.is_file():
        raise FileNotFoundError(
            f"H3_PROTOCOL_SELECTED_EXECUTION_PROFILE_MISSING:{execution_profile_path}"
        )
    sources: list[dict[str, str]] = []
    for relative in (*COMMON_AUTHORITY_SOURCES, str(execution_profile_path.relative_to(REPO_ROOT))):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"H3_PROTOCOL_AUTHORITY_SOURCE_MISSING:{path}")
        sources.append({"path": relative, "sha256": _sha256(path)})
    contract = {
        **config,
        "retarget": {
            **config["retarget"],
            "execution_contract": selected_execution,
            "execution_profile_sha256": _sha256(execution_profile_path),
        },
        "source_controller": {
            **config["source_controller"],
            "selected_contract": selected_source,
        },
        "lane_final_receipts": {
            "H3A": {
                "path": str(h3a_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(h3a_path),
                "decision": h3a["decision"],
            },
            "H3B": {
                "path": str(h3b_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(h3b_path),
                "decision": h3b["decision"],
            },
        },
        "authority_sources": sources,
        "freeze": {
            "branch": branch,
            "H3_EXECUTION_HEAD": execution_head,
            "tracked_worktree_clean": True,
            "h3c_started": False,
            "mutable_after_first_h3c_result": False,
        },
    }
    contract_hash = _stable_hash(contract)
    authorities = {
        "schema_version": "H3CurrentAuthoritiesV1",
        "H3_PROTOCOL_HASH": contract_hash,
        "H3_EXECUTION_HEAD": execution_head,
        "authorities": {
            "episode": "HOCapSingleHandObjectEpisodeV1",
            "retarget_input": "RetargetInputQualityV1",
            "retarget_execution": selected_execution,
            "source_admission": "SourceControllerExecutableV2",
            "source_fidelity": "SourceControllerFidelityV2",
            "source_route": "SourceControllerAutoV2",
            "support": "SupportResolutionV1",
            "gpu": "GPURuntimePreflightV1",
            "reward": "Stage16GroupedMultiplicativeRewardV1",
            "rse": "Stage16ReferenceScopedExplorationV1",
            "rsi": "UniformEventBalancedRSIV1",
            "object_scale": "DimensionlessObjectScaleV1",
            "pf": "PhysicalFunctionalityV2+PhysicalFunctionalityFullCycleV1",
            "df": "pose+linear+AngularAuthorityV2",
        },
    }
    output = args.output.resolve()
    _write_new(
        output / "h3_physicalization_protocol.json",
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
    )
    _write_new(output / "h3_protocol_hash.txt", contract_hash + "\n")
    _write_new(
        output / "current_authorities.json",
        json.dumps(authorities, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        REPO_ROOT / ".local/reports/h3_unseen_object_generalization/workspace/execution_head.json",
        json.dumps(
            {
                "schema_version": "H3ExecutionHeadV1",
                "branch": branch,
                "H3_EXECUTION_HEAD": execution_head,
                "tracked_worktree_clean": True,
                "H3_PROTOCOL_HASH": contract_hash,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        json.dumps(
            {
                "H3_PROTOCOL_HASH": contract_hash,
                "H3_EXECUTION_HEAD": execution_head,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
