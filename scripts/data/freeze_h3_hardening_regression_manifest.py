#!/usr/bin/env python3
"""Freeze the five known H3-C hardening episodes as regression inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.freeze_hocap_episode_held_out import _manifest_episode  # noqa: E402

EXPECTED_EPISODES = (
    "hocap_subject_9_20231027_125019__right__G16_3__ep00",
    "hocap_subject_6_20231025_112332__right__G09_4__ep00",
    "hocap_subject_2_20231023_164741__right__G22_3__ep00",
    "hocap_subject_3_20231024_161209__right__G16_2__ep00",
    "hocap_subject_1_20231025_170231__right__G10_3__ep00",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--old-hardening-manifest", type=Path, required=True)
    parser.add_argument("--h3-protocol", type=Path, required=True)
    parser.add_argument("--h3-protocol-hash", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"H3C_REGRESSION_MANIFEST_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"H3C_GIT_HEAD_FAILED:{result.stdout}")
    return result.stdout.strip()


def main() -> int:
    args = _parser().parse_args()
    episode_path = args.episode_index.resolve()
    rows = json.loads(episode_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("H3C_EPISODE_INDEX_LIST_REQUIRED")
    indexed = {str(row["episode_id"]): row for row in rows}
    if any(episode not in indexed for episode in EXPECTED_EPISODES):
        raise ValueError("H3C_EXPECTED_EPISODE_MISSING")
    old_path = args.old_hardening_manifest.resolve()
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if old.get("schema_version") != "PipelineHardeningSetManifestV1":
        raise ValueError("H3C_OLD_HARDENING_MANIFEST_INVALID")
    old_ids = tuple(str(row["episode_id"]) for row in old.get("episodes", ()))
    if old_ids != EXPECTED_EPISODES:
        raise ValueError("H3C_OLD_HARDENING_EPISODE_ORDER_DRIFT")
    protocol_path = args.h3_protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "H3PhysicalizationProtocolV1":
        raise ValueError("H3C_PROTOCOL_INVALID")
    protocol_hash = args.h3_protocol_hash.resolve().read_text(encoding="utf-8").strip()
    if protocol_hash != _stable_hash(protocol):
        raise ValueError("H3C_PROTOCOL_HASH_DRIFT")
    execution_head = str(protocol.get("freeze", {}).get("H3_EXECUTION_HEAD", ""))
    if len(execution_head) != 40 or execution_head != _git_head():
        raise ValueError("H3C_EXECUTION_HEAD_DRIFT")

    clips: list[dict[str, Any]] = []
    for rank, episode_id in enumerate(EXPECTED_EPISODES, start=1):
        row = dict(indexed[episode_id])
        if not (
            row.get("active_hand") == "right"
            and row.get("physicalization_v1_eligible") is True
            and row.get("complete") is True
            and row.get("episode_type") == "SINGLE_HAND_PICK_PLACE"
            and not row.get("other_hand_same_target")
        ):
            raise ValueError(f"H3C_EPISODE_CONTRACT_DRIFT:{episode_id}")
        row["selection_rank"] = rank
        row["selection_key"] = _stable_hash(
            {"dataset_role": "PIPELINE_HARDENING_SET_V1", "episode_id": episode_id}
        )
        clip = _manifest_episode(row)
        clip["dataset_role"] = "PIPELINE_HARDENING_SET_V1"
        clip["held_out"] = False
        clip["historical_outcome_observed"] = True
        clip["execution_purpose"] = "PIPELINE_REGRESSION_NOT_SCIENTIFIC_HELDOUT_RATE"
        clip["exclusion_audit"] = {
            "outcome_observed": True,
            "allowed_because_dataset_role_is_regression": True,
        }
        clips.append(clip)
    manifest: dict[str, Any] = {
        "schema_version": "H3PipelineHardeningRegressionManifestV1",
        "status": "FROZEN_NOT_EXECUTED",
        "REGRESSION_SET_FROZEN": "YES",
        "dataset": "hocap",
        "dataset_role": "PIPELINE_HARDENING_SET_V1",
        "held_out": False,
        "episode_contract": "HOCapSingleHandObjectEpisodeV1",
        "physicalization_protocol": "H3PhysicalizationProtocolV1",
        "h3_protocol_hash": protocol_hash,
        "H3_EXECUTION_HEAD": execution_head,
        "episode_count": 5,
        "held_out_rate_denominator": False,
        "historical_outcomes_acknowledged": True,
        "fresh_raw_to_final_execution_required": True,
        "old_hardening_manifest": {
            "path": str(old_path),
            "sha256": _sha256(old_path),
        },
        "episode_index": {"path": str(episode_path), "sha256": _sha256(episode_path)},
        "clips": clips,
        "episodes": clips,
    }
    manifest["manifest_sha256"] = _stable_hash(manifest)
    receipt = {
        "schema_version": "H3HardeningRegressionManifestReceiptV1",
        "status": "FROZEN_NOT_EXECUTED",
        "dataset_role": manifest["dataset_role"],
        "held_out": False,
        "episode_ids": list(EXPECTED_EPISODES),
        "manifest_sha256": manifest["manifest_sha256"],
        "h3_protocol_hash": protocol_hash,
        "H3_EXECUTION_HEAD": execution_head,
        "historical_outcome_reuse": False,
        "fresh_execution_required": True,
    }
    output = args.output_root.resolve()
    _write_new(
        output / "hardening_regression_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        output / "hardening_regression_manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
    )
    _write_new(output / "manifest_sha256.txt", manifest["manifest_sha256"] + "\n")
    _write_new(
        output / "manifest_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
