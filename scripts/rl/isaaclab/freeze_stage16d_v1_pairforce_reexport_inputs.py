#!/usr/bin/env python3
"""Freeze the immutable V1 inputs for a pair-force-only Formal20 re-export."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from toporetarget.rl.ppo.checkpoint import load_checkpoint  # noqa: E402


CLIPS = ("hocap_170105", "hocap_170650")
CONTINUATION_ROOT = REPO_ROOT / ".local/reports/stage16d_ppo26d_continuation"
OUTPUT_ROOT = REPO_ROOT / ".local/reports/stage16d_reward_v3_pairforce_unblock"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_manifest_path(clip: str) -> Path:
    dedicated = CONTINUATION_ROOT / f"frozen_seed_sets_{clip.removeprefix('hocap_')}.json"
    return dedicated if dedicated.is_file() else CONTINUATION_ROOT / "frozen_seed_sets.json"


def _formal_seed_set(payload: dict[str, Any], *, clip: str) -> tuple[str, list[int]]:
    candidates = [
        key
        for key, value in payload.items()
        if key.startswith("formal_holdout_seed_set")
        and isinstance(value, dict)
        and isinstance(value.get("seeds"), list)
    ]
    expected = f"formal_holdout_seed_set_{clip.removeprefix('hocap_')}_v1"
    key = expected if expected in candidates else candidates[0] if len(candidates) == 1 else None
    if key is None:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_FORMAL_SEED_SET_MISSING:{clip}:{candidates}")
    seeds = [int(value) for value in payload[key]["seeds"]]
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_FORMAL_SEED_SET_INVALID:{clip}")
    return key, seeds


def _trace_scalar(archive: Any, name: str) -> str:
    if name not in archive.files:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_OLD_R7_FIELD_MISSING:{name}")
    return str(np.asarray(archive[name]).item())


def _clip_provenance(clip: str) -> dict[str, Any]:
    root = CONTINUATION_ROOT / clip
    selection_path = root / "checkpoint_selection.json"
    formal_evaluation_path = root / "r7_formal_evaluation.json"
    if not selection_path.is_file() or not formal_evaluation_path.is_file():
        raise FileNotFoundError(f"V1_PAIRFORCE_REEXPORT_R7_PROVENANCE_MISSING:{clip}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    formal = json.loads(formal_evaluation_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(selection["checkpoint"])).resolve()
    trace = Path(str(formal["trace"])).resolve()
    if not checkpoint.is_file() or not trace.is_file():
        raise FileNotFoundError(f"V1_PAIRFORCE_REEXPORT_SOURCE_ARTIFACT_MISSING:{clip}")
    checkpoint_hash = _sha256(checkpoint)
    if (
        checkpoint_hash != selection["checkpoint_sha256"]
        or checkpoint_hash != formal["checkpoint_sha256"]
    ):
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_INPUT_DRIFT:{clip}:checkpoint_hash")
    if Path(str(formal["checkpoint"])).resolve() != checkpoint:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_INPUT_DRIFT:{clip}:checkpoint_path")
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload.get("schema_version") != "Stage16DPPO26DCheckpointV1":
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_CHECKPOINT_SCHEMA_INVALID:{clip}")
    if payload.get("clip") != clip:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_INPUT_DRIFT:{clip}:checkpoint_clip")
    environment = payload.get("environment_contract")
    ppo = environment.get("ppo26d") if isinstance(environment, dict) else None
    if (
        not isinstance(ppo, dict)
        or ppo.get("reward", {}).get("identifier") != "TopoRetargetReferenceTrackingReward26DV1"
    ):
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_REWARD_CONTRACT_DRIFT:{clip}")
    if int(ppo.get("observation", {}).get("dimension", -1)) != 764:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_OBSERVATION_CONTRACT_DRIFT:{clip}")
    with np.load(trace, allow_pickle=False) as archive:
        if _trace_scalar(archive, "checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"V1_PAIRFORCE_REEXPORT_INPUT_DRIFT:{clip}:r7_trace_checkpoint")
        if _trace_scalar(archive, "checkpoint_path") != str(checkpoint):
            raise ValueError(f"V1_PAIRFORCE_REEXPORT_INPUT_DRIFT:{clip}:r7_trace_path")
        if _trace_scalar(archive, "action_contract") != "26D_reference_residual":
            raise ValueError(f"V1_PAIRFORCE_REEXPORT_ACTION_CONTRACT_DRIFT:{clip}")
        reference_hash = json.loads(_trace_scalar(archive, "reference_hash"))
        if archive["replica_action"].shape != (321, 20, 26):
            raise ValueError(f"V1_PAIRFORCE_REEXPORT_R7_ACTION_SHAPE_INVALID:{clip}")
    manifest_path = _seed_manifest_path(clip)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed_set, seeds = _formal_seed_set(manifest, clip=clip)
    formal_rows = formal.get("frame_zero")
    formal_seeds = (
        [int(row["seed"]) for row in formal_rows] if isinstance(formal_rows, list) else []
    )
    if formal_seeds != seeds:
        raise ValueError(f"V1_PAIRFORCE_REEXPORT_INPUT_DRIFT:{clip}:formal_seed_order")
    return {
        "clip": clip,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_schema": payload["schema_version"],
        "checkpoint_samples": int(payload["cumulative_samples"]),
        "old_r7_evaluation": {
            "path": str(formal_evaluation_path.resolve()),
            "sha256": _sha256(formal_evaluation_path),
        },
        "old_r7_trace": {"path": str(trace), "sha256": _sha256(trace)},
        "reference_hash": reference_hash,
        "physics_contract_hash": payload.get("physics_contract_hash"),
        "action_contract": payload.get("action_contract"),
        "observation_contract": payload.get("observation_contract"),
        "reward_version": "TopoRetargetReferenceTrackingReward26DV1",
        "seed_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256(manifest_path)},
        "formal_seed_set": seed_set,
        "formal_seeds": seeds,
        "formal_episode_count": 20,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output_root.resolve()
    clips = {clip: _clip_provenance(clip) for clip in CLIPS}
    source_reports = {
        str(path.relative_to(REPO_ROOT)): {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in (
            REPO_ROOT / ".local/reports/stage16d_reward_v3_contact/final_summary.json",
            CONTINUATION_ROOT / "final_summary.json",
            REPO_ROOT / ".local/reports/stage16d_ppo26d_clip_repair/final_summary.json",
            REPO_ROOT / ".local/reports/stage16d_reference_kinematics_v2/final_summary.json",
            REPO_ROOT / ".local/reports/stage16d_phase1_phase2/final_summary.json",
        )
    }
    for path, receipt in source_reports.items():
        if not Path(str(receipt["path"])).is_file():
            raise FileNotFoundError(f"V1_PAIRFORCE_REEXPORT_REQUIRED_REPORT_MISSING:{path}")
    contract_files = (
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env.py",
        REPO_ROOT
        / "src/toporetarget/rl/environments/isaaclab_backend/ppo26d_reference_tracking_env_cfg.py",
        REPO_ROOT / "src/toporetarget/rl/ppo/ppo26d_contract.py",
        CONTINUATION_ROOT / "global_ppo_contract.json",
    )
    contract_hashes = {
        str(path.relative_to(REPO_ROOT)): {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in contract_files
    }
    frozen = {
        "schema_version": "Stage16DV1PairForceReexportFrozenInputsV1",
        "status": "V1_PAIRFORCE_REEXPORT_INPUTS_FROZEN",
        "purpose": "V1_PAIRFORCE_REEXPORT_DIAGNOSTIC",
        "reward_version": "TopoRetargetReferenceTrackingReward26DV1",
        "policy_mode": "deterministic_mean_action",
        "frame_zero_only": True,
        "rsi": False,
        "clips": clips,
        "source_reports": source_reports,
    }
    _write_json(output / "v1_pairforce_frozen_inputs.json", frozen)
    _write_json(output / "checkpoint_manifest.json", {"clips": clips})
    _write_json(
        output / "formal_seed_manifest.json",
        {
            clip: {
                "seed_manifest": row["seed_manifest"],
                "seed_set": row["formal_seed_set"],
                "seeds": row["formal_seeds"],
            }
            for clip, row in clips.items()
        },
    )
    _write_json(output / "contract_hashes.json", contract_hashes)
    _write_json(
        output / "old_v3_blocker.json",
        {
            "path": str(
                (
                    REPO_ROOT
                    / ".local/reports/stage16d_reward_v3_contact/force_scale_calibration.json"
                ).resolve()
            ),
            "sha256": _sha256(
                REPO_ROOT / ".local/reports/stage16d_reward_v3_contact/force_scale_calibration.json"
            ),
            "status": "CONTACT_REWARD_PAIR_FORCE_UNRESOLVED",
        },
    )
    print(json.dumps({"status": frozen["status"], "output_root": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
