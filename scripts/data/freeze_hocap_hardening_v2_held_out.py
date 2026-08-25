#!/usr/bin/env python3
"""Freeze a new metadata-only HOCap Frozen5 for hardening protocol V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.freeze_hocap_episode_held_out import (  # noqa: E402
    DEFAULT_DEVELOPMENT_SEQUENCE_SUFFIXES,
    _manifest_episode,
    select_episodes,
)

SELECTION_SEED = 20260825
HARDENING_SEQUENCE_SUFFIXES = ("125019", "112332", "164741", "161209", "170231")


def _canonical_hash(value: object) -> str:
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
        raise FileExistsError(f"HARDENING_V2_HELD_OUT_OUTPUT_EXISTS:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(f"HARDENING_V2_HELD_OUT_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--hardening-manifest", type=Path, required=True)
    parser.add_argument("--hardening-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    args = parser.parse_args()

    rows = json.loads(args.episode_index.resolve().read_text(encoding="utf-8"))
    hardening = json.loads(args.hardening_manifest.resolve().read_text(encoding="utf-8"))
    contract = json.loads(args.hardening_contract.resolve().read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("HARDENING_V2_EPISODE_INDEX_LIST_REQUIRED")
    if hardening.get("schema_version") != "PipelineHardeningSetManifestV1":
        raise ValueError("HARDENING_V2_HARDENING_MANIFEST_INVALID")
    if contract.get("schema_version") != "HOCapPhysicalizationHardeningProtocolV2":
        raise ValueError("HARDENING_V2_CONTRACT_INVALID")
    hardening_episodes = {str(row["episode_id"]) for row in hardening["episodes"]}
    hardening_sequences = {str(row["sequence"]) for row in hardening["episodes"]}
    if {sequence.rsplit("_", 1)[-1] for sequence in hardening_sequences} != set(
        HARDENING_SEQUENCE_SUFFIXES
    ):
        raise ValueError("HARDENING_V2_EXCLUSION_LIST_DRIFT")

    suffixes = tuple(
        sorted(set((*DEFAULT_DEVELOPMENT_SEQUENCE_SUFFIXES, *HARDENING_SEQUENCE_SUFFIXES)))
    )
    selected, audited = select_episodes(
        rows,
        count=args.count,
        seed=args.seed,
        excluded_suffixes=suffixes,
        active_hand="right",
    )
    selected_ids = {str(row["episode_id"]) for row in selected}
    selected_sequences = {str(row["raw_sequence"]) for row in selected}
    selected_objects = {str(row["target_object"]) for row in selected}
    if (
        selected_ids & hardening_episodes
        or selected_sequences & hardening_sequences
        or len(selected_sequences) != args.count
        or len(selected_objects) != args.count
    ):
        raise RuntimeError("HARDENING_V2_HELD_OUT_DIVERSITY_OR_EXCLUSION_FAILURE")

    manifest: dict[str, Any] = {
        "schema_version": "HOCapPhysicalizationHeldOutEpisodeManifestV2",
        "status": "FROZEN_NOT_EXECUTED",
        "HELD_OUT_SET_FROZEN": "YES",
        "dataset": "hocap",
        "selection_unit": "HOCapSingleHandObjectEpisodeV1",
        "physicalization_protocol": "HOCapPhysicalizationHardeningProtocolV2",
        "hardening_v2_contract_sha256": _canonical_hash(contract),
        "episode_contract_sha256": selected[0]["contract_sha256"],
        "held_out_count": args.count,
        "selection_basis": "metadata_only_no_retarget_support_l0_pf_ppo_outcomes",
        "selection_seed": args.seed,
        "active_hand_backend_capability_filter": "right",
        "development_sequence_suffix_exclusions": list(suffixes),
        "diversity_priority": ["raw_sequence", "target_object", "subject"],
        "episodes": [_manifest_episode(row) for row in selected],
        "clips": [_manifest_episode(row) for row in selected],
        "geometric_retarget_run": False,
        "source_policy_l0_run": False,
        "support_physx_run": False,
        "frozen_evaluation_run": False,
        "physical_ppo_run": False,
        "downstream_outcomes_observed": False,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)

    fields = [
        "episode_id",
        "raw_sequence",
        "subject",
        "active_hand",
        "target_object",
        "start_frame",
        "end_frame",
        "duration_frames",
        "selection_key",
        "selection_rank",
        "development_exclusion_reason",
    ]
    candidate_rows = []
    for row in audited:
        if row["development_exclusion_reason"]:
            continue
        rank = next(
            (
                item["selection_rank"]
                for item in selected
                if item["episode_id"] == row["episode_id"]
            ),
            "",
        )
        candidate_rows.append({**row, "selection_rank": rank})
    exclusions = [row for row in audited if row["development_exclusion_reason"]]
    receipt = {
        "schema_version": "HOCapHardeningV2HeldOutSelectionReceiptV1",
        "status": "FROZEN_NOT_EXECUTED",
        "selection_seed": args.seed,
        "selection_algorithm": "sha256_seed_episode_with_sequence_object_subject_diversity_v1",
        "metadata_only": True,
        "outcome_fields_used": [],
        "hardening_set_excluded": True,
        "development_sequences_excluded": list(suffixes),
        "candidate_count": len(candidate_rows),
        "excluded_count": len(exclusions),
        "selected_episode_ids": [row["episode_id"] for row in selected],
        "selected_sequences_unique": len(selected_sequences) == args.count,
        "selected_objects_unique": len(selected_objects) == args.count,
        "selected_subject_count": len({str(row["subject"]) for row in selected}),
        "episode_index_sha256": _sha256(args.episode_index.resolve()),
        "hardening_manifest_sha256": _sha256(args.hardening_manifest.resolve()),
        "hardening_contract_canonical_sha256": _canonical_hash(contract),
        "manifest_sha256": manifest["manifest_sha256"],
        "downstream_execution": "FORBIDDEN_NOT_RUN",
    }
    output = args.output_root.resolve()
    _write_new(
        output / "new_held_out_5_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_new(
        output / "new_held_out_5_manifest.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
    )
    _write_new(output / "manifest_sha256.txt", str(manifest["manifest_sha256"]) + "\n")
    _write_new(
        output / "selection_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    _write_csv(output / "candidate_pool.csv", candidate_rows, fields)
    _write_csv(output / "development_exclusions.csv", exclusions, fields)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
