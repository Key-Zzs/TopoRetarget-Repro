#!/usr/bin/env python3
"""Materialize an identically-zero PPO residual actor from a V2 qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.independent_physical_refinement import atomic_write_json  # noqa: E402
from toporetarget.rl.ppo.checkpoint import load_checkpoint  # noqa: E402
from toporetarget.rl.ppo.ppo26d_trainer import PPO26DTrainer  # noqa: E402
from toporetarget.rl.source_controller import make_zero_output_residual_actor_  # noqa: E402
from toporetarget.utils.hashing import sha256_file  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("ZERO_RESIDUAL_QUALIFICATION_OBJECT_REQUIRED")
    return value


def main() -> int:
    args = _parser().parse_args()
    qualification_path = args.qualification.resolve()
    checkpoint = args.checkpoint.resolve()
    result_path = args.result.resolve()
    if checkpoint.exists() or result_path.exists():
        raise FileExistsError("ZERO_RESIDUAL_SOURCE_REFUSES_OVERWRITE")
    qualification = _json(qualification_path)
    if (
        qualification.get("schema_version") != "SourceControllerQualificationV2"
        or qualification.get("mode") != "ZERO_RESIDUAL_DETERMINISTIC"
        or qualification.get("source_controller_executability_v2") != "PASS"
    ):
        raise ValueError("ZERO_RESIDUAL_SOURCE_QUALIFICATION_INVALID")
    runtime = qualification.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("ZERO_RESIDUAL_SOURCE_RUNTIME_CONTRACT_MISSING")
    clip = qualification.get("clip_id")
    reference = runtime.get("reference_bank", {})
    frame_count = int(reference.get("frame_count", -1))
    if (
        runtime.get("source_controller_admission_v2") is not True
        or not isinstance(clip, str)
        or frame_count < 17
    ):
        raise ValueError("ZERO_RESIDUAL_SOURCE_RUNTIME_CONTRACT_INVALID")

    torch.manual_seed(0)
    trainer = PPO26DTrainer(
        observation_dim=764,
        device="cpu",
        runtime_reference_samples=frame_count,
    )
    make_zero_output_residual_actor_(trainer.model)
    trainer.trainer.freeze_observation_normalizer()
    observations = torch.randn(7, 764)
    with torch.no_grad():
        before = trainer.trainer.distribution(observations).mean
    if not torch.equal(before, torch.zeros_like(before)):
        raise RuntimeError("ZERO_RESIDUAL_SOURCE_ACTOR_NOT_IDENTICALLY_ZERO")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(
        checkpoint,
        environment_contract=runtime,
        selected_num_envs=int(qualification["episodes"]),
        extra_payload={
            "source_controller_route": "ZERO_RESIDUAL",
            "source_controller_executability_v2": "PASS",
            "source_qualification_path": str(qualification_path),
            "source_qualification_sha256": sha256_file(qualification_path),
            "optimizer_steps": 0,
            "training_samples": 0,
        },
    )
    payload = load_checkpoint(checkpoint, map_location="cpu")
    restored = PPO26DTrainer(
        observation_dim=764,
        device="cpu",
        runtime_reference_samples=frame_count,
    )
    restored.model.load_state_dict(payload["actor_critic"])
    restored.trainer.normalizer.load_state_dict(payload["observation_normalization"])
    with torch.no_grad():
        after = restored.trainer.distribution(observations).mean
    if not torch.equal(before, after) or not torch.equal(after, torch.zeros_like(after)):
        raise RuntimeError("ZERO_RESIDUAL_SOURCE_CHECKPOINT_ROUNDTRIP_FAILURE")

    result = {
        "schema_version": "Stage16DZeroResidualSourceTrainingV1",
        "status": "ZERO_RESIDUAL_SOURCE_CONTROLLER_MATERIALIZED",
        "clip": clip,
        "source_controller_route": "ZERO_RESIDUAL",
        "source_controller_executability_v2": "PASS",
        "optimizer_steps": 0,
        "training_samples": 0,
        "cumulative_samples": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "qualification": {
            "path": str(qualification_path),
            "sha256": sha256_file(qualification_path),
        },
        "deterministic_action_identically_zero": True,
        "checkpoint_roundtrip_identical": True,
        "network_architecture": "Stage16D_PPO26D_ActorCritic",
    }
    atomic_write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
